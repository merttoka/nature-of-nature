#include "physarum.h"
#include "../preset.h"
#include <imgui.h>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <random>

static constexpr float DEG2RAD = 3.14159265359f / 180.0f;

void PhysarumSim::init(WGPUDevice device, WGPUQueue queue, uint32_t w, uint32_t h) {
    m_device = device;
    m_queue = queue;
    params.width = w;
    params.height = h;

    m_trailTextures.init(device, w, h, WGPUTextureFormat_RGBA16Float);
    m_outputTextures.init(device, w, h, WGPUTextureFormat_RGBA8Unorm);

    createBuffers();
    createPipelines();
    m_needsReset = true;
}

void PhysarumSim::createBuffers() {
    // Agent buffer: 16 bytes per agent (vec2f position + vec2f direction)
    {
        WGPUBufferDescriptor desc = {};
        desc.size = (uint64_t)m_agentCount * 16;
        desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
        desc.label = "physarum_agents";
        m_agentBuffer = wgpuDeviceCreateBuffer(m_device, &desc);
    }
    // Uniform buffer: 160 bytes
    {
        WGPUBufferDescriptor desc = {};
        desc.size = sizeof(GpuParams);
        desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
        desc.label = "physarum_params";
        m_uniformBuffer = wgpuDeviceCreateBuffer(m_device, &desc);
    }
}

void PhysarumSim::createPipelines() {
    // Load shader
    std::string code = loadShaderFile("shaders/physarum.wgsl");
    if (code.empty()) return;

    WGPUShaderModuleWGSLDescriptor wgslDesc = {};
    wgslDesc.chain.sType = WGPUSType_ShaderModuleWGSLDescriptor;
    wgslDesc.code = code.c_str();
    WGPUShaderModuleDescriptor smDesc = {};
    smDesc.nextInChain = &wgslDesc.chain;
    m_shaderModule = wgpuDeviceCreateShaderModule(m_device, &smDesc);

    // Group 0 layout: uniform, trailRead, trailWrite, outRead, outWrite
    {
        WGPUBindGroupLayoutEntry entries[5] = {};

        // b0: uniform
        entries[0].binding = 0;
        entries[0].visibility = WGPUShaderStage_Compute;
        entries[0].buffer.type = WGPUBufferBindingType_Uniform;
        entries[0].buffer.minBindingSize = sizeof(GpuParams);

        // b1: trailRead (texture_2d<f32>)
        entries[1].binding = 1;
        entries[1].visibility = WGPUShaderStage_Compute;
        entries[1].texture.sampleType = WGPUTextureSampleType_Float;
        entries[1].texture.viewDimension = WGPUTextureViewDimension_2D;

        // b2: trailWrite (storage rgba16float)
        entries[2].binding = 2;
        entries[2].visibility = WGPUShaderStage_Compute;
        entries[2].storageTexture.access = WGPUStorageTextureAccess_WriteOnly;
        entries[2].storageTexture.format = WGPUTextureFormat_RGBA16Float;
        entries[2].storageTexture.viewDimension = WGPUTextureViewDimension_2D;

        // b3: outRead (texture_2d<f32>)
        entries[3].binding = 3;
        entries[3].visibility = WGPUShaderStage_Compute;
        entries[3].texture.sampleType = WGPUTextureSampleType_Float;
        entries[3].texture.viewDimension = WGPUTextureViewDimension_2D;

        // b4: outWrite (storage rgba8unorm)
        entries[4].binding = 4;
        entries[4].visibility = WGPUShaderStage_Compute;
        entries[4].storageTexture.access = WGPUStorageTextureAccess_WriteOnly;
        entries[4].storageTexture.format = WGPUTextureFormat_RGBA8Unorm;
        entries[4].storageTexture.viewDimension = WGPUTextureViewDimension_2D;

        WGPUBindGroupLayoutDescriptor desc = {};
        desc.entryCount = 5;
        desc.entries = entries;
        m_group0Layout = wgpuDeviceCreateBindGroupLayout(m_device, &desc);
    }

    // Group 1 layout: agents storage buffer
    {
        WGPUBindGroupLayoutEntry entry = {};
        entry.binding = 0;
        entry.visibility = WGPUShaderStage_Compute;
        entry.buffer.type = WGPUBufferBindingType_Storage;
        entry.buffer.minBindingSize = 16; // at least 1 agent

        WGPUBindGroupLayoutDescriptor desc = {};
        desc.entryCount = 1;
        desc.entries = &entry;
        m_group1Layout = wgpuDeviceCreateBindGroupLayout(m_device, &desc);
    }

    // Pipeline layout with 2 bind groups
    {
        WGPUBindGroupLayout layouts[2] = { m_group0Layout, m_group1Layout };
        WGPUPipelineLayoutDescriptor desc = {};
        desc.bindGroupLayoutCount = 2;
        desc.bindGroupLayouts = layouts;
        m_pipelineLayout = wgpuDeviceCreatePipelineLayout(m_device, &desc);
    }

    // Create all 6 pipelines sharing shader module and layout
    auto makePipeline = [&](const char* entry) -> WGPUComputePipeline {
        WGPUComputePipelineDescriptor desc = {};
        desc.layout = m_pipelineLayout;
        desc.compute.module = m_shaderModule;
        desc.compute.entryPoint = entry;
        return wgpuDeviceCreateComputePipeline(m_device, &desc);
    };

    m_resetTexturePipeline  = makePipeline("reset_texture");
    m_resetAgentsPipeline   = makePipeline("reset_agents");
    m_moveAgentsPipeline    = makePipeline("move_agents");
    m_writeTrailsPipeline   = makePipeline("write_trails");
    m_diffuseTexturePipeline = makePipeline("diffuse_texture");
    m_renderPipeline        = makePipeline("render");

    // Group 1 bind group (agents buffer — doesn't change unless agent count changes)
    {
        WGPUBindGroupEntry entry = {};
        entry.binding = 0;
        entry.buffer = m_agentBuffer;
        entry.size = (uint64_t)m_agentCount * 16;

        WGPUBindGroupDescriptor desc = {};
        desc.layout = m_group1Layout;
        desc.entryCount = 1;
        desc.entries = &entry;
        m_group1 = wgpuDeviceCreateBindGroup(m_device, &desc);
    }
}

WGPUBindGroup PhysarumSim::buildGroup0() {
    WGPUBindGroupEntry entries[5] = {};

    entries[0].binding = 0;
    entries[0].buffer = m_uniformBuffer;
    entries[0].size = sizeof(GpuParams);

    entries[1].binding = 1;
    entries[1].textureView = m_trailTextures.readView();

    entries[2].binding = 2;
    entries[2].textureView = m_trailTextures.writeView();

    entries[3].binding = 3;
    entries[3].textureView = m_outputTextures.readView();

    entries[4].binding = 4;
    entries[4].textureView = m_outputTextures.writeView();

    WGPUBindGroupDescriptor desc = {};
    desc.layout = m_group0Layout;
    desc.entryCount = 5;
    desc.entries = entries;
    return wgpuDeviceCreateBindGroup(m_device, &desc);
}

void PhysarumSim::uploadParams() {
    GpuParams gp = {};
    gp.rezX = params.width;
    gp.rezY = params.height;
    gp.agentsCount = m_agentCount;
    gp.time = m_frameCounter;

    for (int i = 0; i < 4; i++) {
        gp.senseAngles[i]    = m_senseAngle[i] * DEG2RAD;
        gp.senseDistances[i] = m_senseDistance[i];
        gp.turnAngles[i]     = m_turnAngle[i] * DEG2RAD;
        gp.moveSpeeds[i]     = m_moveSpeed[i];
        gp.depositAmounts[i] = m_deposit[i];
        gp.eatAmounts[i]     = m_eat[i];
        gp.diffuseRates[i]   = m_diffuseRate[i];
        gp.hues[i]           = m_hue[i];
        gp.saturations[i]    = m_saturation[i];
    }

    wgpuQueueWriteBuffer(m_queue, m_uniformBuffer, 0, &gp, sizeof(gp));
}

void PhysarumSim::clearTextures() {
    // Zero-fill all 4 textures via CPU upload
    uint32_t w = params.width, h = params.height;

    // Trail textures: rgba16float = 8 bytes per pixel
    {
        std::vector<uint8_t> zeros(w * h * 8, 0);
        WGPUImageCopyTexture dst = {};
        WGPUTextureDataLayout layout = {};
        layout.bytesPerRow = w * 8;
        layout.rowsPerImage = h;
        WGPUExtent3D size = { w, h, 1 };

        dst.texture = m_trailTextures.texA;
        wgpuQueueWriteTexture(m_queue, &dst, zeros.data(), zeros.size(), &layout, &size);
        dst.texture = m_trailTextures.texB;
        wgpuQueueWriteTexture(m_queue, &dst, zeros.data(), zeros.size(), &layout, &size);
    }

    // Output textures: rgba8unorm = 4 bytes per pixel
    {
        std::vector<uint8_t> zeros(w * h * 4, 0);
        WGPUImageCopyTexture dst = {};
        WGPUTextureDataLayout layout = {};
        layout.bytesPerRow = w * 4;
        layout.rowsPerImage = h;
        WGPUExtent3D size = { w, h, 1 };

        dst.texture = m_outputTextures.texA;
        wgpuQueueWriteTexture(m_queue, &dst, zeros.data(), zeros.size(), &layout, &size);
        dst.texture = m_outputTextures.texB;
        wgpuQueueWriteTexture(m_queue, &dst, zeros.data(), zeros.size(), &layout, &size);
    }
}

void PhysarumSim::dispatchReset(WGPUCommandEncoder encoder) {
    m_trailTextures.current = 0;
    m_outputTextures.current = 0;
    m_frameCounter = 0;

    // Recreate agent buffer if size changed
    uint64_t requiredSize = (uint64_t)m_agentCount * 16;
    uint64_t currentSize = m_agentBuffer ? wgpuBufferGetSize(m_agentBuffer) : 0;

    if (currentSize != requiredSize) {
        if (m_agentBuffer) { wgpuBufferDestroy(m_agentBuffer); wgpuBufferRelease(m_agentBuffer); }
        if (m_group1) wgpuBindGroupRelease(m_group1);

        WGPUBufferDescriptor desc = {};
        desc.size = requiredSize;
        desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
        desc.label = "physarum_agents";
        m_agentBuffer = wgpuDeviceCreateBuffer(m_device, &desc);

        WGPUBindGroupEntry entry = {};
        entry.binding = 0;
        entry.buffer = m_agentBuffer;
        entry.size = requiredSize;
        WGPUBindGroupDescriptor bgDesc = {};
        bgDesc.layout = m_group1Layout;
        bgDesc.entryCount = 1;
        bgDesc.entries = &entry;
        m_group1 = wgpuDeviceCreateBindGroup(m_device, &bgDesc);
    }

    clearTextures();
    uploadParams();

    WGPUBindGroup bg0 = buildGroup0();

    // Reset agents kernel
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);
    wgpuComputePassEncoderSetPipeline(pass, m_resetAgentsPipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bg0, 0, nullptr);
    wgpuComputePassEncoderSetBindGroup(pass, 1, m_group1, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(pass, (m_agentCount + 255) / 256, 1, 1);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);

    wgpuBindGroupRelease(bg0);
}

void PhysarumSim::step(WGPUCommandEncoder encoder) {
    if (m_needsReset) {
        m_needsReset = false;
        dispatchReset(encoder);
        return;
    }

    if (params.paused && !m_doStep) return;
    m_doStep = false;

    uint32_t wgTex = (params.width + 7) / 8;
    uint32_t hgTex = (params.height + 7) / 8;
    uint32_t wgAgent = (m_agentCount + 255) / 256;

    for (int s = 0; s < m_stepsPerFrame; s++) {
        m_frameCounter++;
        uploadParams();

        // 1. Build group 0 for current ping-pong state
        WGPUBindGroup bg0 = buildGroup0();

        // 2. MoveAgents — reads trailRead, updates agents
        {
            WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);
            wgpuComputePassEncoderSetPipeline(pass, m_moveAgentsPipeline);
            wgpuComputePassEncoderSetBindGroup(pass, 0, bg0, 0, nullptr);
            wgpuComputePassEncoderSetBindGroup(pass, 1, m_group1, 0, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(pass, wgAgent, 1, 1);
            wgpuComputePassEncoderEnd(pass);
            wgpuComputePassEncoderRelease(pass);
        }

        // 3. DiffuseTexture — trailRead -> trailWrite (blur)
        {
            WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);
            wgpuComputePassEncoderSetPipeline(pass, m_diffuseTexturePipeline);
            wgpuComputePassEncoderSetBindGroup(pass, 0, bg0, 0, nullptr);
            wgpuComputePassEncoderSetBindGroup(pass, 1, m_group1, 0, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(pass, wgTex, hgTex, 1);
            wgpuComputePassEncoderEnd(pass);
            wgpuComputePassEncoderRelease(pass);
        }

        // 4. Copy trailWrite -> trailRead so WriteTrails can read diffused data
        {
            WGPUImageCopyTexture src = {};
            src.texture = m_trailTextures.current == 0 ? m_trailTextures.texB : m_trailTextures.texA;
            WGPUImageCopyTexture dst = {};
            dst.texture = m_trailTextures.current == 0 ? m_trailTextures.texA : m_trailTextures.texB;
            WGPUExtent3D size = { params.width, params.height, 1 };
            wgpuCommandEncoderCopyTextureToTexture(encoder, &src, &dst, &size);
        }

        wgpuBindGroupRelease(bg0);

        // 5. WriteTrails — reads trailRead (now has diffused data), writes trailWrite
        // Need new bind group since we just copied (trailRead has diffused data, trailWrite same)
        bg0 = buildGroup0();
        {
            WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);
            wgpuComputePassEncoderSetPipeline(pass, m_writeTrailsPipeline);
            wgpuComputePassEncoderSetBindGroup(pass, 0, bg0, 0, nullptr);
            wgpuComputePassEncoderSetBindGroup(pass, 1, m_group1, 0, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(pass, wgAgent, 1, 1);
            wgpuComputePassEncoderEnd(pass);
            wgpuComputePassEncoderRelease(pass);
        }

        // 6. Swap trail ping-pong
        m_trailTextures.swap();

        wgpuBindGroupRelease(bg0);

        // 7. Render — reads trailRead + outRead, writes outWrite
        bg0 = buildGroup0();
        {
            WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);
            wgpuComputePassEncoderSetPipeline(pass, m_renderPipeline);
            wgpuComputePassEncoderSetBindGroup(pass, 0, bg0, 0, nullptr);
            wgpuComputePassEncoderSetBindGroup(pass, 1, m_group1, 0, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(pass, wgTex, hgTex, 1);
            wgpuComputePassEncoderEnd(pass);
            wgpuComputePassEncoderRelease(pass);
        }

        // 8. Swap output ping-pong
        m_outputTextures.swap();

        wgpuBindGroupRelease(bg0);
    }
}

void PhysarumSim::reset() {
    m_needsReset = true;
}

WGPUTextureView PhysarumSim::getOutputView() {
    return m_outputTextures.readView();
}

WGPUTexture PhysarumSim::getOutputTexture() {
    return m_outputTextures.current == 0 ? m_outputTextures.texA : m_outputTextures.texB;
}

void PhysarumSim::onGui() {
    ImGui::Text("Physarum");
    ImGui::Separator();

    if (ImGui::Button(params.paused ? "Play" : "Pause")) {
        params.paused = !params.paused;
    }
    ImGui::SameLine();
    if (ImGui::Button("Step")) {
        params.paused = true;
        m_doStep = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset")) {
        reset();
    }

    ImGui::SliderInt("Steps/Frame", &m_stepsPerFrame, 1, 20);

    {
        int ac = (int)m_agentCount;
        if (ImGui::InputInt("Agents (reset)", &ac, 1000, 1000000)) {
            if (ac < 1024) ac = 1024;
            if (ac > 5000000) ac = 5000000;
            if ((uint32_t)ac != m_agentCount) {
                m_agentCount = (uint32_t)ac;
                m_needsReset = true;
            }
        }
    }

    if (ImGui::Button("Randomize")) {
        static std::mt19937 rng(std::random_device{}());
        auto rf = [&](float lo, float hi) {
            return std::uniform_real_distribution<float>(lo, hi)(rng);
        };
        m_linkTypes = false;
        for (int i = 0; i < 4; i++) {
            m_senseAngle[i]   = rf(0.1f, 360.0f);
            m_senseDistance[i] = rf(0.1f, 200.0f);
            m_turnAngle[i]    = rf(0.1f, 360.0f);
            m_moveSpeed[i]    = rf(0.01f, 5.0f);
            m_deposit[i]      = rf(0.001f, 0.5f);
            m_eat[i]          = rf(0.001f, 0.5f);
            m_diffuseRate[i]  = rf(0.0f, 1.0f);
            m_hue[i]          = rf(0.0f, 1.0f);
            m_saturation[i]   = rf(0.3f, 1.0f);
        }
    }

    static char presetName[64] = "default";
    ImGui::InputText("Preset Name", presetName, sizeof(presetName));

    if (ImGui::Button("Save Preset")) {
        std::map<std::string, std::vector<float>> data;
        data["agentCount"] = {(float)m_agentCount};
        data["linkTypes"] = {m_linkTypes ? 1.0f : 0.0f};
        data["senseAngle"]   = {m_senseAngle[0], m_senseAngle[1], m_senseAngle[2], m_senseAngle[3]};
        data["senseDistance"] = {m_senseDistance[0], m_senseDistance[1], m_senseDistance[2], m_senseDistance[3]};
        data["turnAngle"]    = {m_turnAngle[0], m_turnAngle[1], m_turnAngle[2], m_turnAngle[3]};
        data["moveSpeed"]    = {m_moveSpeed[0], m_moveSpeed[1], m_moveSpeed[2], m_moveSpeed[3]};
        data["deposit"]      = {m_deposit[0], m_deposit[1], m_deposit[2], m_deposit[3]};
        data["eat"]          = {m_eat[0], m_eat[1], m_eat[2], m_eat[3]};
        data["diffuseRate"]  = {m_diffuseRate[0], m_diffuseRate[1], m_diffuseRate[2], m_diffuseRate[3]};
        data["hue"]          = {m_hue[0], m_hue[1], m_hue[2], m_hue[3]};
        data["saturation"]   = {m_saturation[0], m_saturation[1], m_saturation[2], m_saturation[3]};
        savePreset(std::string("physarum_") + presetName, data);
    }
    ImGui::SameLine();
    if (ImGui::Button("Load Preset")) {
        auto data = loadPreset(std::string("physarum_") + presetName);
        if (!data.empty()) {
            auto load4 = [&](const char* key, float* dst) {
                auto it = data.find(key);
                if (it != data.end()) for (size_t i = 0; i < 4 && i < it->second.size(); i++) dst[i] = it->second[i];
            };
            if (data.count("agentCount") && !data["agentCount"].empty()) {
                m_agentCount = (uint32_t)data["agentCount"][0];
                m_needsReset = true;
            }
            if (data.count("linkTypes") && !data["linkTypes"].empty())
                m_linkTypes = data["linkTypes"][0] > 0.5f;
            load4("senseAngle", m_senseAngle);
            load4("senseDistance", m_senseDistance);
            load4("turnAngle", m_turnAngle);
            load4("moveSpeed", m_moveSpeed);
            load4("deposit", m_deposit);
            load4("eat", m_eat);
            load4("diffuseRate", m_diffuseRate);
            load4("hue", m_hue);
            load4("saturation", m_saturation);
        }
    }

    ImGui::Checkbox("Link All Types", &m_linkTypes);

    if (m_linkTypes) {
        // Single set of sliders, apply to all 4 types
        bool changed = false;
        changed |= ImGui::SliderFloat("Sense Angle", &m_senseAngle[0], 0.1f, 360.0f);
        changed |= ImGui::SliderFloat("Sense Distance", &m_senseDistance[0], 0.1f, 200.0f);
        changed |= ImGui::SliderFloat("Turn Angle", &m_turnAngle[0], 0.1f, 360.0f);
        changed |= ImGui::SliderFloat("Move Speed", &m_moveSpeed[0], 0.01f, 5.0f);
        changed |= ImGui::SliderFloat("Deposit", &m_deposit[0], 0.001f, 0.5f);
        changed |= ImGui::SliderFloat("Eat", &m_eat[0], 0.001f, 0.5f);
        changed |= ImGui::SliderFloat("Diffuse Rate", &m_diffuseRate[0], 0.0f, 1.0f);
        changed |= ImGui::SliderFloat("Hue", &m_hue[0], 0.0f, 1.0f);
        changed |= ImGui::SliderFloat("Saturation", &m_saturation[0], 0.0f, 1.0f);
        if (changed) {
            for (int i = 1; i < 4; i++) {
                m_senseAngle[i]   = m_senseAngle[0];
                m_senseDistance[i] = m_senseDistance[0];
                m_turnAngle[i]    = m_turnAngle[0];
                m_moveSpeed[i]    = m_moveSpeed[0];
                m_deposit[i]      = m_deposit[0];
                m_eat[i]          = m_eat[0];
                m_diffuseRate[i]  = m_diffuseRate[0];
                m_hue[i]          = m_hue[0];
                m_saturation[i]   = m_saturation[0];
            }
        }
    } else {
        for (int t = 0; t < 4; t++) {
            char label[32];
            snprintf(label, sizeof(label), "Type %d", t);
            if (ImGui::TreeNode(label)) {
                ImGui::PushID(t);
                ImGui::SliderFloat("Sense Angle", &m_senseAngle[t], 0.1f, 360.0f);
                ImGui::SliderFloat("Sense Distance", &m_senseDistance[t], 0.1f, 200.0f);
                ImGui::SliderFloat("Turn Angle", &m_turnAngle[t], 0.1f, 360.0f);
                ImGui::SliderFloat("Move Speed", &m_moveSpeed[t], 0.01f, 5.0f);
                ImGui::SliderFloat("Deposit", &m_deposit[t], 0.001f, 0.5f);
                ImGui::SliderFloat("Eat", &m_eat[t], 0.001f, 0.5f);
                ImGui::SliderFloat("Diffuse Rate", &m_diffuseRate[t], 0.0f, 1.0f);
                ImGui::SliderFloat("Hue", &m_hue[t], 0.0f, 1.0f);
                ImGui::SliderFloat("Saturation", &m_saturation[t], 0.0f, 1.0f);
                ImGui::PopID();
                ImGui::TreePop();
            }
        }
    }
}

void PhysarumSim::shutdown() {
    if (m_group1) wgpuBindGroupRelease(m_group1);
    if (m_group0Layout) wgpuBindGroupLayoutRelease(m_group0Layout);
    if (m_group1Layout) wgpuBindGroupLayoutRelease(m_group1Layout);
    if (m_pipelineLayout) wgpuPipelineLayoutRelease(m_pipelineLayout);

    if (m_resetTexturePipeline)   wgpuComputePipelineRelease(m_resetTexturePipeline);
    if (m_resetAgentsPipeline)    wgpuComputePipelineRelease(m_resetAgentsPipeline);
    if (m_moveAgentsPipeline)     wgpuComputePipelineRelease(m_moveAgentsPipeline);
    if (m_writeTrailsPipeline)    wgpuComputePipelineRelease(m_writeTrailsPipeline);
    if (m_diffuseTexturePipeline) wgpuComputePipelineRelease(m_diffuseTexturePipeline);
    if (m_renderPipeline)         wgpuComputePipelineRelease(m_renderPipeline);

    if (m_shaderModule) wgpuShaderModuleRelease(m_shaderModule);
    if (m_agentBuffer) { wgpuBufferDestroy(m_agentBuffer); wgpuBufferRelease(m_agentBuffer); }
    if (m_uniformBuffer) { wgpuBufferDestroy(m_uniformBuffer); wgpuBufferRelease(m_uniformBuffer); }

    m_trailTextures.destroy();
    m_outputTextures.destroy();
}
