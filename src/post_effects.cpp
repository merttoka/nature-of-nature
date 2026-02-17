#include "post_effects.h"
#include <imgui.h>
#include <cstring>
#include <cmath>

// Gradient control points: {t, r, g, b}
struct GradPoint { float t, r, g, b; };

static void lerpGradient(const GradPoint* pts, int n, uint8_t out[256*4]) {
    for (int i = 0; i < 256; i++) {
        float t = i / 255.0f;
        // Find segment
        int seg = 0;
        for (int j = 0; j < n - 1; j++) {
            if (t >= pts[j].t && t <= pts[j+1].t) { seg = j; break; }
            if (j == n - 2) seg = j;
        }
        float frac = (pts[seg+1].t > pts[seg].t)
            ? (t - pts[seg].t) / (pts[seg+1].t - pts[seg].t) : 0.0f;
        out[i*4+0] = (uint8_t)((pts[seg].r + (pts[seg+1].r - pts[seg].r) * frac) * 255.0f);
        out[i*4+1] = (uint8_t)((pts[seg].g + (pts[seg+1].g - pts[seg].g) * frac) * 255.0f);
        out[i*4+2] = (uint8_t)((pts[seg].b + (pts[seg+1].b - pts[seg].b) * frac) * 255.0f);
        out[i*4+3] = 255;
    }
}

static const char* colormapNames[] = { "Viridis", "Inferno", "Magma", "Plasma", "Grayscale" };
static const int colormapCount = 5;

static void generateColormap(int index, uint8_t out[256*4]) {
    switch (index) {
    case 0: { // Viridis
        GradPoint pts[] = {
            {0.0f, 0.267f, 0.004f, 0.329f},
            {0.25f, 0.282f, 0.140f, 0.458f},
            {0.5f, 0.127f, 0.566f, 0.551f},
            {0.75f, 0.544f, 0.774f, 0.247f},
            {1.0f, 0.993f, 0.906f, 0.144f},
        };
        lerpGradient(pts, 5, out);
    } break;
    case 1: { // Inferno
        GradPoint pts[] = {
            {0.0f, 0.001f, 0.000f, 0.014f},
            {0.25f, 0.341f, 0.062f, 0.429f},
            {0.5f, 0.735f, 0.215f, 0.330f},
            {0.75f, 0.978f, 0.557f, 0.035f},
            {1.0f, 0.988f, 1.000f, 0.644f},
        };
        lerpGradient(pts, 5, out);
    } break;
    case 2: { // Magma
        GradPoint pts[] = {
            {0.0f, 0.001f, 0.000f, 0.014f},
            {0.25f, 0.316f, 0.072f, 0.485f},
            {0.5f, 0.717f, 0.215f, 0.475f},
            {0.75f, 0.983f, 0.533f, 0.382f},
            {1.0f, 0.987f, 0.991f, 0.750f},
        };
        lerpGradient(pts, 5, out);
    } break;
    case 3: { // Plasma
        GradPoint pts[] = {
            {0.0f, 0.050f, 0.030f, 0.528f},
            {0.25f, 0.494f, 0.012f, 0.658f},
            {0.5f, 0.798f, 0.280f, 0.470f},
            {0.75f, 0.973f, 0.585f, 0.253f},
            {1.0f, 0.940f, 0.975f, 0.131f},
        };
        lerpGradient(pts, 5, out);
    } break;
    case 4: { // Grayscale
        for (int i = 0; i < 256; i++) {
            out[i*4+0] = out[i*4+1] = out[i*4+2] = (uint8_t)i;
            out[i*4+3] = 255;
        }
    } break;
    }
}

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
    createLutTexture();
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

void PostEffects::createLutTexture() {
    // Generate default colormap data
    uint8_t data[256 * 4];
    generateColormap(colormapIndex, data);

    WGPUTextureDescriptor desc = {};
    desc.size = { 256, 1, 1 };
    desc.format = WGPUTextureFormat_RGBA8Unorm;
    desc.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
    desc.mipLevelCount = 1;
    desc.sampleCount = 1;
    desc.dimension = WGPUTextureDimension_2D;
    desc.label = "lut_texture";
    m_lutTex = wgpuDeviceCreateTexture(m_device, &desc);
    m_lutView = wgpuTextureCreateView(m_lutTex, nullptr);

    // Upload data
    WGPUImageCopyTexture dst = {};
    dst.texture = m_lutTex;
    dst.mipLevel = 0;
    WGPUTextureDataLayout layout = {};
    layout.bytesPerRow = 256 * 4;
    layout.rowsPerImage = 1;
    WGPUExtent3D extent = { 256, 1, 1 };
    wgpuQueueWriteTexture(m_queue, &dst, data, 256 * 4, &layout, &extent);

    // Sampler
    WGPUSamplerDescriptor sampDesc = {};
    sampDesc.magFilter = WGPUFilterMode_Linear;
    sampDesc.minFilter = WGPUFilterMode_Linear;
    sampDesc.addressModeU = WGPUAddressMode_ClampToEdge;
    sampDesc.addressModeV = WGPUAddressMode_ClampToEdge;
    sampDesc.maxAnisotropy = 1;
    m_lutSampler = wgpuDeviceCreateSampler(m_device, &sampDesc);
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

    // Bind group layout: uniform, inputTex, bloomTex (read), outputTex (write), lutSampler, lutTex
    {
        WGPUBindGroupLayoutEntry entries[6] = {};

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

        // b4: LUT sampler
        entries[4].binding = 4;
        entries[4].visibility = WGPUShaderStage_Compute;
        entries[4].sampler.type = WGPUSamplerBindingType_Filtering;

        // b5: LUT texture (256x1 2D)
        entries[5].binding = 5;
        entries[5].visibility = WGPUShaderStage_Compute;
        entries[5].texture.sampleType = WGPUTextureSampleType_Float;
        entries[5].texture.viewDimension = WGPUTextureViewDimension_2D;

        WGPUBindGroupLayoutDescriptor desc = {};
        desc.entryCount = 6;
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
    // Re-upload LUT if colormap changed
    {
        static int lastColormapIndex = -1;
        if (colormapIndex != lastColormapIndex) {
            lastColormapIndex = colormapIndex;
            uint8_t data[256 * 4];
            generateColormap(colormapIndex, data);
            WGPUImageCopyTexture dst = {};
            dst.texture = m_lutTex;
            dst.mipLevel = 0;
            WGPUTextureDataLayout layout = {};
            layout.bytesPerRow = 256 * 4;
            layout.rowsPerImage = 1;
            WGPUExtent3D extent = { 256, 1, 1 };
            wgpuQueueWriteTexture(m_queue, &dst, data, 256 * 4, &layout, &extent);
        }
    }

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
    gp.useLut = useColormap ? 1u : 0u;
    wgpuQueueWriteBuffer(m_queue, m_uniformBuffer, 0, &gp, sizeof(gp));

    uint32_t wg = (m_width + 7) / 8;
    uint32_t hg = (m_height + 7) / 8;

    // Helper to build bind group
    auto buildBG = [&](WGPUTextureView input, WGPUTextureView secondary, WGPUTextureView output) -> WGPUBindGroup {
        WGPUBindGroupEntry entries[6] = {};
        entries[0].binding = 0;
        entries[0].buffer = m_uniformBuffer;
        entries[0].size = sizeof(GpuParams);
        entries[1].binding = 1;
        entries[1].textureView = input;
        entries[2].binding = 2;
        entries[2].textureView = secondary;
        entries[3].binding = 3;
        entries[3].textureView = output;
        entries[4].binding = 4;
        entries[4].sampler = m_lutSampler;
        entries[5].binding = 5;
        entries[5].textureView = m_lutView;

        WGPUBindGroupDescriptor desc = {};
        desc.layout = m_bindGroupLayout;
        desc.entryCount = 6;
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
    ImGui::Separator();
    ImGui::Checkbox("Colormap", &useColormap);
    if (useColormap) {
        ImGui::Combo("Preset", &colormapIndex, colormapNames, colormapCount);
    }
}

void PostEffects::shutdown() {
    destroyTextures();
    if (m_lutView) wgpuTextureViewRelease(m_lutView);
    if (m_lutTex) { wgpuTextureDestroy(m_lutTex); wgpuTextureRelease(m_lutTex); }
    if (m_lutSampler) wgpuSamplerRelease(m_lutSampler);
    m_lutView = nullptr; m_lutTex = nullptr; m_lutSampler = nullptr;
    if (m_uniformBuffer) { wgpuBufferDestroy(m_uniformBuffer); wgpuBufferRelease(m_uniformBuffer); }
    if (m_bloomHPipeline) wgpuComputePipelineRelease(m_bloomHPipeline);
    if (m_bloomVPipeline) wgpuComputePipelineRelease(m_bloomVPipeline);
    if (m_compositePipeline) wgpuComputePipelineRelease(m_compositePipeline);
    if (m_bindGroupLayout) wgpuBindGroupLayoutRelease(m_bindGroupLayout);
    if (m_pipelineLayout) wgpuPipelineLayoutRelease(m_pipelineLayout);
    if (m_shaderModule) wgpuShaderModuleRelease(m_shaderModule);
}
