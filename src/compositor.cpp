#include "compositor.h"
#include "compute_pass.h"
#include <imgui.h>
#include <cstring>

void Compositor::init(WGPUDevice device, WGPUQueue queue, uint32_t w, uint32_t h) {
    m_device = device;
    m_queue = queue;
    m_width = w;
    m_height = h;

    {
        WGPUBufferDescriptor desc = {};
        desc.size = sizeof(GpuParams);
        desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
        desc.label = "compositor_params";
        m_uniformBuffer = wgpuDeviceCreateBuffer(m_device, &desc);
    }

    createTextures();
    createPipelines();
}

void Compositor::createTextures() {
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

    m_texA = makeTex("compositor_A");
    m_texB = makeTex("compositor_B");
    m_viewA = wgpuTextureCreateView(m_texA, nullptr);
    m_viewB = wgpuTextureCreateView(m_texB, nullptr);
    m_current = 0;
}

void Compositor::destroyTextures() {
    if (m_viewA) wgpuTextureViewRelease(m_viewA);
    if (m_viewB) wgpuTextureViewRelease(m_viewB);
    if (m_texA) { wgpuTextureDestroy(m_texA); wgpuTextureRelease(m_texA); }
    if (m_texB) { wgpuTextureDestroy(m_texB); wgpuTextureRelease(m_texB); }
    m_viewA = m_viewB = nullptr;
    m_texA = m_texB = nullptr;
}

void Compositor::resize(uint32_t w, uint32_t h) {
    if (w == m_width && h == m_height) return;
    m_width = w;
    m_height = h;
    destroyTextures();
    createTextures();
}

void Compositor::createPipelines() {
    std::string code = loadShaderFile("shaders/compositor.wgsl");
    if (code.empty()) return;

    WGPUShaderModuleWGSLDescriptor wgslDesc = {};
    wgslDesc.chain.sType = WGPUSType_ShaderModuleWGSLDescriptor;
    wgslDesc.code = code.c_str();
    WGPUShaderModuleDescriptor smDesc = {};
    smDesc.nextInChain = &wgslDesc.chain;
    m_shaderModule = wgpuDeviceCreateShaderModule(m_device, &smDesc);

    // Bind group layout: uniform, layerTex(read), accumTex(read), outputTex(write)
    {
        WGPUBindGroupLayoutEntry entries[4] = {};

        entries[0].binding = 0;
        entries[0].visibility = WGPUShaderStage_Compute;
        entries[0].buffer.type = WGPUBufferBindingType_Uniform;
        entries[0].buffer.minBindingSize = sizeof(GpuParams);

        entries[1].binding = 1;
        entries[1].visibility = WGPUShaderStage_Compute;
        entries[1].texture.sampleType = WGPUTextureSampleType_Float;
        entries[1].texture.viewDimension = WGPUTextureViewDimension_2D;

        entries[2].binding = 2;
        entries[2].visibility = WGPUShaderStage_Compute;
        entries[2].texture.sampleType = WGPUTextureSampleType_Float;
        entries[2].texture.viewDimension = WGPUTextureViewDimension_2D;

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

    {
        WGPUPipelineLayoutDescriptor desc = {};
        desc.bindGroupLayoutCount = 1;
        desc.bindGroupLayouts = &m_bindGroupLayout;
        m_pipelineLayout = wgpuDeviceCreatePipelineLayout(m_device, &desc);
    }

    WGPUComputePipelineDescriptor desc = {};
    desc.layout = m_pipelineLayout;
    desc.compute.module = m_shaderModule;
    desc.compute.entryPoint = "blend";
    m_pipeline = wgpuDeviceCreateComputePipeline(m_device, &desc);
}

void Compositor::composite(WGPUCommandEncoder encoder) {
    uint32_t wg = (m_width + 7) / 8;
    uint32_t hg = (m_height + 7) / 8;
    m_current = 0;

    bool isFirst = true;

    auto buildBG = [&](WGPUTextureView layer, WGPUTextureView accum, WGPUTextureView output) -> WGPUBindGroup {
        WGPUBindGroupEntry entries[4] = {};
        entries[0].binding = 0;
        entries[0].buffer = m_uniformBuffer;
        entries[0].size = sizeof(GpuParams);
        entries[1].binding = 1;
        entries[1].textureView = layer;
        entries[2].binding = 2;
        entries[2].textureView = accum;
        entries[3].binding = 3;
        entries[3].textureView = output;

        WGPUBindGroupDescriptor desc = {};
        desc.layout = m_bindGroupLayout;
        desc.entryCount = 4;
        desc.entries = entries;
        return wgpuDeviceCreateBindGroup(m_device, &desc);
    };

    for (auto& layer : layers) {
        if (!layer.enabled || !layer.sim) continue;

        GpuParams gp = {};
        gp.width = m_width;
        gp.height = m_height;
        gp.blendMode = (uint32_t)layer.blendMode;
        gp.opacity = layer.opacity;
        gp.isFirstLayer = isFirst ? 1 : 0;
        wgpuQueueWriteBuffer(m_queue, m_uniformBuffer, 0, &gp, sizeof(gp));

        // Determine ping-pong: accum is current read, output is current write
        WGPUTextureView accumView = (m_current == 0) ? m_viewA : m_viewB;
        WGPUTextureView outView   = (m_current == 0) ? m_viewB : m_viewA;

        // For first layer, accum is unused but must be valid — use the layer itself
        WGPUTextureView accumArg = isFirst ? layer.sim->getOutputView() : accumView;

        WGPUBindGroup bg = buildBG(layer.sim->getOutputView(), accumArg, outView);
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);
        wgpuComputePassEncoderSetPipeline(pass, m_pipeline);
        wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, wg, hg, 1);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
        wgpuBindGroupRelease(bg);

        // Swap: output becomes the new accumulator
        m_current = 1 - m_current;
        isFirst = false;
    }

    // If no layers were enabled, clear output to black
    if (isFirst) {
        GpuParams gp = {};
        gp.width = m_width;
        gp.height = m_height;
        gp.isFirstLayer = 1;
        gp.opacity = 0.0f;
        wgpuQueueWriteBuffer(m_queue, m_uniformBuffer, 0, &gp, sizeof(gp));

        WGPUBindGroup bg = buildBG(m_viewA, m_viewA, m_viewB);
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);
        wgpuComputePassEncoderSetPipeline(pass, m_pipeline);
        wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, wg, hg, 1);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
        wgpuBindGroupRelease(bg);
        m_current = 1; // output is in B
    }
}

WGPUTextureView Compositor::getOutputView() const {
    // After composite(), the latest result is in the texture we just wrote to.
    // m_current was swapped after the last write, so current read side has the result.
    return (m_current == 0) ? m_viewA : m_viewB;
}

void Compositor::onGui() {
    const char* blendNames[] = { "Additive", "Multiply", "Screen", "Normal" };

    for (int i = 0; i < (int)layers.size(); i++) {
        auto& l = layers[i];
        ImGui::PushID(i);

        ImGui::Checkbox(l.sim->name(), &l.enabled);
        if (l.enabled) {
            ImGui::SameLine();
            ImGui::SetNextItemWidth(80);
            int bm = (int)l.blendMode;
            if (ImGui::Combo("##blend", &bm, blendNames, 4)) {
                l.blendMode = (BlendMode)bm;
            }
            ImGui::SameLine();
            ImGui::SetNextItemWidth(80);
            ImGui::SliderFloat("##opacity", &l.opacity, 0.0f, 1.0f);
        }

        // Drag reorder
        if (ImGui::IsItemActive() && !ImGui::IsItemHovered()) {
            int next = i + (ImGui::GetMouseDragDelta(0).y < 0.0f ? -1 : 1);
            if (next >= 0 && next < (int)layers.size()) {
                std::swap(layers[i], layers[next]);
                ImGui::ResetMouseDragDelta();
            }
        }

        ImGui::PopID();
    }
}

void Compositor::shutdown() {
    destroyTextures();
    if (m_uniformBuffer) { wgpuBufferDestroy(m_uniformBuffer); wgpuBufferRelease(m_uniformBuffer); }
    if (m_pipeline) wgpuComputePipelineRelease(m_pipeline);
    if (m_bindGroupLayout) wgpuBindGroupLayoutRelease(m_bindGroupLayout);
    if (m_pipelineLayout) wgpuPipelineLayoutRelease(m_pipelineLayout);
    if (m_shaderModule) wgpuShaderModuleRelease(m_shaderModule);
    m_uniformBuffer = nullptr;
    m_pipeline = nullptr;
    m_bindGroupLayout = nullptr;
    m_pipelineLayout = nullptr;
    m_shaderModule = nullptr;
}
