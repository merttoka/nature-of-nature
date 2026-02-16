#include "post_effects.h"
#include <imgui.h>
#include <cstring>

void PostEffects::init(WGPUDevice device, WGPUQueue queue, uint32_t w, uint32_t h) {
    m_device = device;
    m_queue = queue;
    m_width = w;
    m_height = h;

    // Uniform buffer
    {
        WGPUBufferDescriptor desc = {};
        desc.size = sizeof(GpuParams);
        desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
        desc.label = "post_effects_params";
        m_uniformBuffer = wgpuDeviceCreateBuffer(m_device, &desc);
    }

    createTextures();
    createPipelines();
}

void PostEffects::createTextures() {
    auto makeTex = [&](const char* label) -> WGPUTexture {
        WGPUTextureDescriptor desc = {};
        desc.size = { m_width, m_height, 1 };
        desc.format = WGPUTextureFormat_RGBA8Unorm;
        desc.usage = WGPUTextureUsage_StorageBinding | WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopySrc;
        desc.mipLevelCount = 1;
        desc.sampleCount = 1;
        desc.dimension = WGPUTextureDimension_2D;
        desc.label = label;
        return wgpuDeviceCreateTexture(m_device, &desc);
    };
    auto makeView = [](WGPUTexture tex) -> WGPUTextureView {
        return wgpuTextureCreateView(tex, nullptr);
    };

    m_bloomATex = makeTex("post_bloomA");
    m_bloomBTex = makeTex("post_bloomB");
    m_outputTex = makeTex("post_output");
    m_bloomAView = makeView(m_bloomATex);
    m_bloomBView = makeView(m_bloomBTex);
    m_outputView = makeView(m_outputTex);
}

void PostEffects::destroyTextures() {
    if (m_bloomAView) wgpuTextureViewRelease(m_bloomAView);
    if (m_bloomBView) wgpuTextureViewRelease(m_bloomBView);
    if (m_outputView) wgpuTextureViewRelease(m_outputView);
    if (m_bloomATex) { wgpuTextureDestroy(m_bloomATex); wgpuTextureRelease(m_bloomATex); }
    if (m_bloomBTex) { wgpuTextureDestroy(m_bloomBTex); wgpuTextureRelease(m_bloomBTex); }
    if (m_outputTex) { wgpuTextureDestroy(m_outputTex); wgpuTextureRelease(m_outputTex); }
    m_bloomAView = m_bloomBView = m_outputView = nullptr;
    m_bloomATex = m_bloomBTex = m_outputTex = nullptr;
}

void PostEffects::resize(uint32_t w, uint32_t h) {
    if (w == m_width && h == m_height) return;
    m_width = w;
    m_height = h;
    destroyTextures();
    createTextures();
}

void PostEffects::createPipelines() {
    std::string code = loadShaderFile("shaders/post_effects.wgsl");
    if (code.empty()) return;

    WGPUShaderModuleWGSLDescriptor wgslDesc = {};
    wgslDesc.chain.sType = WGPUSType_ShaderModuleWGSLDescriptor;
    wgslDesc.code = code.c_str();
    WGPUShaderModuleDescriptor smDesc = {};
    smDesc.nextInChain = &wgslDesc.chain;
    m_shaderModule = wgpuDeviceCreateShaderModule(m_device, &smDesc);

    // Bind group layout: uniform, inputTex, bloomTex (read), outputTex (write)
    {
        WGPUBindGroupLayoutEntry entries[4] = {};

        // b0: uniform
        entries[0].binding = 0;
        entries[0].visibility = WGPUShaderStage_Compute;
        entries[0].buffer.type = WGPUBufferBindingType_Uniform;
        entries[0].buffer.minBindingSize = sizeof(GpuParams);

        // b1: input texture (read)
        entries[1].binding = 1;
        entries[1].visibility = WGPUShaderStage_Compute;
        entries[1].texture.sampleType = WGPUTextureSampleType_Float;
        entries[1].texture.viewDimension = WGPUTextureViewDimension_2D;

        // b2: secondary texture (read)
        entries[2].binding = 2;
        entries[2].visibility = WGPUShaderStage_Compute;
        entries[2].texture.sampleType = WGPUTextureSampleType_Float;
        entries[2].texture.viewDimension = WGPUTextureViewDimension_2D;

        // b3: output texture (write)
        entries[3].binding = 3;
        entries[3].visibility = WGPUShaderStage_Compute;
        entries[3].storageTexture.access = WGPUStorageTextureAccess_WriteOnly;
        entries[3].storageTexture.format = WGPUTextureFormat_RGBA8Unorm;
        entries[3].storageTexture.viewDimension = WGPUTextureViewDimension_2D;

        WGPUBindGroupLayoutDescriptor desc = {};
        desc.entryCount = 4;
        desc.entries = entries;
        m_bindGroupLayout = wgpuDeviceCreateBindGroupLayout(m_device, &desc);
    }

    // Pipeline layout
    {
        WGPUPipelineLayoutDescriptor desc = {};
        desc.bindGroupLayoutCount = 1;
        desc.bindGroupLayouts = &m_bindGroupLayout;
        m_pipelineLayout = wgpuDeviceCreatePipelineLayout(m_device, &desc);
    }

    auto makePipeline = [&](const char* entry) -> WGPUComputePipeline {
        WGPUComputePipelineDescriptor desc = {};
        desc.layout = m_pipelineLayout;
        desc.compute.module = m_shaderModule;
        desc.compute.entryPoint = entry;
        return wgpuDeviceCreateComputePipeline(m_device, &desc);
    };

    m_bloomHPipeline = makePipeline("bloom_h");
    m_bloomVPipeline = makePipeline("bloom_v");
    m_compositePipeline = makePipeline("composite");
}

void PostEffects::apply(WGPUCommandEncoder encoder, WGPUTextureView simOutput) {
    // Upload params
    GpuParams gp = {};
    gp.width = m_width;
    gp.height = m_height;
    gp.brightness = brightness;
    gp.contrast = contrast;
    gp.bloomThreshold = bloomThreshold;
    gp.bloomIntensity = bloomIntensity;
    gp.bloomRadius = bloomRadius;
    gp.saturationPost = saturationPost;
    gp.vignette = vignette;
    wgpuQueueWriteBuffer(m_queue, m_uniformBuffer, 0, &gp, sizeof(gp));

    uint32_t wg = (m_width + 7) / 8;
    uint32_t hg = (m_height + 7) / 8;

    // Helper to build bind group
    auto buildBG = [&](WGPUTextureView input, WGPUTextureView secondary, WGPUTextureView output) -> WGPUBindGroup {
        WGPUBindGroupEntry entries[4] = {};
        entries[0].binding = 0;
        entries[0].buffer = m_uniformBuffer;
        entries[0].size = sizeof(GpuParams);
        entries[1].binding = 1;
        entries[1].textureView = input;
        entries[2].binding = 2;
        entries[2].textureView = secondary;
        entries[3].binding = 3;
        entries[3].textureView = output;

        WGPUBindGroupDescriptor desc = {};
        desc.layout = m_bindGroupLayout;
        desc.entryCount = 4;
        desc.entries = entries;
        return wgpuDeviceCreateBindGroup(m_device, &desc);
    };

    // Pass 1: Horizontal bloom blur (simOutput -> bloomA)
    // secondary is unused but we need a valid view
    {
        WGPUBindGroup bg = buildBG(simOutput, simOutput, m_bloomAView);
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);
        wgpuComputePassEncoderSetPipeline(pass, m_bloomHPipeline);
        wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, wg, hg, 1);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
        wgpuBindGroupRelease(bg);
    }

    // Pass 2: Vertical bloom blur (bloomA -> bloomB)
    {
        WGPUBindGroup bg = buildBG(m_bloomAView, m_bloomAView, m_bloomBView);
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);
        wgpuComputePassEncoderSetPipeline(pass, m_bloomVPipeline);
        wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, wg, hg, 1);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
        wgpuBindGroupRelease(bg);
    }

    // Pass 3: Composite (simOutput + bloomB -> output)
    {
        WGPUBindGroup bg = buildBG(simOutput, m_bloomBView, m_outputView);
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);
        wgpuComputePassEncoderSetPipeline(pass, m_compositePipeline);
        wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, wg, hg, 1);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
        wgpuBindGroupRelease(bg);
    }
}

WGPUTextureView PostEffects::getOutputView() const {
    return m_outputView;
}

void PostEffects::onGui() {
    if (ImGui::Button("Reset")) {
        brightness = 0.0f;
        contrast = 1.0f;
        saturationPost = 1.0f;
        vignette = 0.0f;
        bloomThreshold = 0.2f;
        bloomIntensity = 0.5f;
        bloomRadius = 5.0f;
    }
    ImGui::SliderFloat("Brightness", &brightness, -0.5f, 1.0f);
    ImGui::SliderFloat("Contrast", &contrast, 0.5f, 1.5f);
    ImGui::SliderFloat("Saturation", &saturationPost, 0.0f, 2.0f);
    ImGui::SliderFloat("Vignette", &vignette, 0.0f, 0.5f);
    ImGui::Separator();
    ImGui::SliderFloat("Bloom Threshold", &bloomThreshold, 0.1f, 1.0f);
    ImGui::SliderFloat("Bloom Intensity", &bloomIntensity, 0.0f, 0.5f);
    ImGui::SliderFloat("Bloom Radius", &bloomRadius, 1.0f, 12.0f);
}

void PostEffects::shutdown() {
    destroyTextures();
    if (m_uniformBuffer) { wgpuBufferDestroy(m_uniformBuffer); wgpuBufferRelease(m_uniformBuffer); }
    if (m_bloomHPipeline) wgpuComputePipelineRelease(m_bloomHPipeline);
    if (m_bloomVPipeline) wgpuComputePipelineRelease(m_bloomVPipeline);
    if (m_compositePipeline) wgpuComputePipelineRelease(m_compositePipeline);
    if (m_bindGroupLayout) wgpuBindGroupLayoutRelease(m_bindGroupLayout);
    if (m_pipelineLayout) wgpuPipelineLayoutRelease(m_pipelineLayout);
    if (m_shaderModule) wgpuShaderModuleRelease(m_shaderModule);
}
