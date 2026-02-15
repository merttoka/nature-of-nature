#include "boids.h"
#include "../preset.h"
#include <imgui.h>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <random>

void BoidsSim::init(WGPUDevice device, WGPUQueue queue, uint32_t w, uint32_t h) {
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

void BoidsSim::createBuffers() {
    // Agent buffer: 48 bytes per agent
    {
        WGPUBufferDescriptor desc = {};
        desc.size = (uint64_t)m_agentCount * 48;
        desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
        desc.label = "boids_agents";
        m_agentBuffer = wgpuDeviceCreateBuffer(m_device, &desc);
    }
    // Uniform buffer: 256 bytes
    {
        WGPUBufferDescriptor desc = {};
        desc.size = sizeof(GpuParams);
        desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
        desc.label = "boids_params";
        m_uniformBuffer = wgpuDeviceCreateBuffer(m_device, &desc);
    }
    // Grid buffers
    m_gridW = (uint32_t)ceilf((float)params.width / m_cellSize);
    m_gridH = (uint32_t)ceilf((float)params.height / m_cellSize);
    uint32_t totalCells = m_gridW * m_gridH;
    {
        WGPUBufferDescriptor desc = {};
        desc.size = totalCells * sizeof(uint32_t);
        desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
        desc.label = "boids_cellCount";
        m_cellCountBuffer = wgpuDeviceCreateBuffer(m_device, &desc);
    }
    {
        WGPUBufferDescriptor desc = {};
        desc.size = (uint64_t)MAX_PER_CELL * totalCells * sizeof(uint32_t);
        desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
        desc.label = "boids_cellAgents";
        m_cellAgentsBuffer = wgpuDeviceCreateBuffer(m_device, &desc);
    }
}

void BoidsSim::createPipelines() {
    std::string code = loadShaderFile("shaders/boids.wgsl");
    if (code.empty()) return;

    WGPUShaderModuleWGSLDescriptor wgslDesc = {};
    wgslDesc.chain.sType = WGPUSType_ShaderModuleWGSLDescriptor;
    wgslDesc.code = code.c_str();
    WGPUShaderModuleDescriptor smDesc = {};
    smDesc.nextInChain = &wgslDesc.chain;
    m_shaderModule = wgpuDeviceCreateShaderModule(m_device, &smDesc);

    // Group 0 layout: uniform, trailRead, trailWrite, outRead, outWrite (same as Physarum)
    {
        WGPUBindGroupLayoutEntry entries[5] = {};

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
        entries[2].storageTexture.access = WGPUStorageTextureAccess_WriteOnly;
        entries[2].storageTexture.format = WGPUTextureFormat_RGBA16Float;
        entries[2].storageTexture.viewDimension = WGPUTextureViewDimension_2D;

        entries[3].binding = 3;
        entries[3].visibility = WGPUShaderStage_Compute;
        entries[3].texture.sampleType = WGPUTextureSampleType_Float;
        entries[3].texture.viewDimension = WGPUTextureViewDimension_2D;

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
        entry.buffer.minBindingSize = 48;

        WGPUBindGroupLayoutDescriptor desc = {};
        desc.entryCount = 1;
        desc.entries = &entry;
        m_group1Layout = wgpuDeviceCreateBindGroupLayout(m_device, &desc);
    }

    // Group 2 layout: cellCount + cellAgents storage buffers
    {
        WGPUBindGroupLayoutEntry entries[2] = {};

        entries[0].binding = 0;
        entries[0].visibility = WGPUShaderStage_Compute;
        entries[0].buffer.type = WGPUBufferBindingType_Storage;
        entries[0].buffer.minBindingSize = 4;

        entries[1].binding = 1;
        entries[1].visibility = WGPUShaderStage_Compute;
        entries[1].buffer.type = WGPUBufferBindingType_Storage;
        entries[1].buffer.minBindingSize = 4;

        WGPUBindGroupLayoutDescriptor desc = {};
        desc.entryCount = 2;
        desc.entries = entries;
        m_group2Layout = wgpuDeviceCreateBindGroupLayout(m_device, &desc);
    }

    // Pipeline layout with 3 bind groups
    {
        WGPUBindGroupLayout layouts[3] = { m_group0Layout, m_group1Layout, m_group2Layout };
        WGPUPipelineLayoutDescriptor desc = {};
        desc.bindGroupLayoutCount = 3;
        desc.bindGroupLayouts = layouts;
        m_pipelineLayout = wgpuDeviceCreatePipelineLayout(m_device, &desc);
    }

    // Create all 8 pipelines
    auto makePipeline = [&](const char* entry) -> WGPUComputePipeline {
        WGPUComputePipelineDescriptor desc = {};
        desc.layout = m_pipelineLayout;
        desc.compute.module = m_shaderModule;
        desc.compute.entryPoint = entry;
        return wgpuDeviceCreateComputePipeline(m_device, &desc);
    };

    m_resetTexturePipeline   = makePipeline("reset_texture");
    m_resetAgentsPipeline    = makePipeline("reset_agents");
    m_clearGridPipeline      = makePipeline("clear_grid");
    m_assignCellsPipeline    = makePipeline("assign_cells");
    m_moveAgentsPipeline     = makePipeline("move_agents");
    m_writeTrailsPipeline    = makePipeline("write_trails");
    m_diffuseTexturePipeline = makePipeline("diffuse_texture");
    m_renderPipeline         = makePipeline("render");

    // Group 1 bind group (agents buffer)
    {
        WGPUBindGroupEntry entry = {};
        entry.binding = 0;
        entry.buffer = m_agentBuffer;
        entry.size = (uint64_t)m_agentCount * 48;

        WGPUBindGroupDescriptor desc = {};
        desc.layout = m_group1Layout;
        desc.entryCount = 1;
        desc.entries = &entry;
        m_group1 = wgpuDeviceCreateBindGroup(m_device, &desc);
    }

    // Group 2 bind group (grid buffers)
    {
        uint32_t totalCells = m_gridW * m_gridH;
        WGPUBindGroupEntry entries[2] = {};

        entries[0].binding = 0;
        entries[0].buffer = m_cellCountBuffer;
        entries[0].size = totalCells * sizeof(uint32_t);

        entries[1].binding = 1;
        entries[1].buffer = m_cellAgentsBuffer;
        entries[1].size = (uint64_t)MAX_PER_CELL * totalCells * sizeof(uint32_t);

        WGPUBindGroupDescriptor desc = {};
        desc.layout = m_group2Layout;
        desc.entryCount = 2;
        desc.entries = entries;
        m_group2 = wgpuDeviceCreateBindGroup(m_device, &desc);
    }
}

WGPUBindGroup BoidsSim::buildGroup0() {
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

void BoidsSim::uploadParams() {
    GpuParams gp = {};
    gp.rezX = params.width;
    gp.rezY = params.height;
    gp.agentsCount = m_agentCount;
    gp.time = m_frameCounter;

    gp.cellSize = m_cellSize;
    gp.gridWf = (float)m_gridW;
    gp.gridHf = (float)m_gridH;
    gp.maxPerCellf = (float)MAX_PER_CELL;

    for (int i = 0; i < 4; i++) {
        gp.maxSpeeds[i]           = m_maxSpeed[i];
        gp.maxForces[i]           = m_maxForce[i];
        gp.typeSeparateRanges[i]  = m_typeSeparateRange[i];
        gp.globalSeparateRanges[i]= m_globalSeparateRange[i];
        gp.alignRanges[i]         = m_alignRange[i];
        gp.attractRanges[i]       = m_attractRange[i];
        gp.foodSensorDistances[i] = m_foodSensorDist[i];
        gp.sensorAngles[i]        = m_sensorAngle[i];
        gp.foodStrengths[i]       = m_foodStrength[i];
        gp.depositAmounts[i]      = m_deposit[i];
        gp.eatAmounts[i]          = m_eat[i];
        gp.diffuseRates[i]        = m_diffuseRate[i];
        gp.hues[i]                = m_hue[i];
        gp.saturations[i]         = m_saturation[i];
    }

    wgpuQueueWriteBuffer(m_queue, m_uniformBuffer, 0, &gp, sizeof(gp));
}

void BoidsSim::clearTextures() {
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

void BoidsSim::dispatchReset(WGPUCommandEncoder encoder) {
    m_trailTextures.current = 0;
    m_outputTextures.current = 0;
    m_frameCounter = 0;

    // Recreate agent buffer if size changed
    uint64_t requiredAgentSize = (uint64_t)m_agentCount * 48;
    uint64_t currentAgentSize = m_agentBuffer ? wgpuBufferGetSize(m_agentBuffer) : 0;
    bool rebuildGroup1 = (currentAgentSize != requiredAgentSize);

    if (rebuildGroup1) {
        if (m_agentBuffer) { wgpuBufferDestroy(m_agentBuffer); wgpuBufferRelease(m_agentBuffer); }

        WGPUBufferDescriptor desc = {};
        desc.size = requiredAgentSize;
        desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
        desc.label = "boids_agents";
        m_agentBuffer = wgpuDeviceCreateBuffer(m_device, &desc);

        if (m_group1) wgpuBindGroupRelease(m_group1);
        WGPUBindGroupEntry entry = {};
        entry.binding = 0;
        entry.buffer = m_agentBuffer;
        entry.size = requiredAgentSize;
        WGPUBindGroupDescriptor bgDesc = {};
        bgDesc.layout = m_group1Layout;
        bgDesc.entryCount = 1;
        bgDesc.entries = &entry;
        m_group1 = wgpuDeviceCreateBindGroup(m_device, &bgDesc);
    }

    // Recalculate grid, recreate if dimensions changed
    uint32_t newGridW = (uint32_t)ceilf((float)params.width / m_cellSize);
    uint32_t newGridH = (uint32_t)ceilf((float)params.height / m_cellSize);
    bool rebuildGroup2 = (newGridW != m_gridW || newGridH != m_gridH);

    if (rebuildGroup2) {
        m_gridW = newGridW;
        m_gridH = newGridH;
        uint32_t totalCells = m_gridW * m_gridH;

        if (m_cellCountBuffer) { wgpuBufferDestroy(m_cellCountBuffer); wgpuBufferRelease(m_cellCountBuffer); }
        if (m_cellAgentsBuffer) { wgpuBufferDestroy(m_cellAgentsBuffer); wgpuBufferRelease(m_cellAgentsBuffer); }

        {
            WGPUBufferDescriptor desc = {};
            desc.size = totalCells * sizeof(uint32_t);
            desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
            desc.label = "boids_cellCount";
            m_cellCountBuffer = wgpuDeviceCreateBuffer(m_device, &desc);
        }
        {
            WGPUBufferDescriptor desc = {};
            desc.size = (uint64_t)MAX_PER_CELL * totalCells * sizeof(uint32_t);
            desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
            desc.label = "boids_cellAgents";
            m_cellAgentsBuffer = wgpuDeviceCreateBuffer(m_device, &desc);
        }

        if (m_group2) wgpuBindGroupRelease(m_group2);
        WGPUBindGroupEntry entries[2] = {};
        entries[0].binding = 0;
        entries[0].buffer = m_cellCountBuffer;
        entries[0].size = totalCells * sizeof(uint32_t);
        entries[1].binding = 1;
        entries[1].buffer = m_cellAgentsBuffer;
        entries[1].size = (uint64_t)MAX_PER_CELL * totalCells * sizeof(uint32_t);
        WGPUBindGroupDescriptor bgDesc = {};
        bgDesc.layout = m_group2Layout;
        bgDesc.entryCount = 2;
        bgDesc.entries = entries;
        m_group2 = wgpuDeviceCreateBindGroup(m_device, &bgDesc);
    }

    clearTextures();
    uploadParams();

    WGPUBindGroup bg0 = buildGroup0();

    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);
    wgpuComputePassEncoderSetPipeline(pass, m_resetAgentsPipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bg0, 0, nullptr);
    wgpuComputePassEncoderSetBindGroup(pass, 1, m_group1, 0, nullptr);
    wgpuComputePassEncoderSetBindGroup(pass, 2, m_group2, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(pass, (m_agentCount + 255) / 256, 1, 1);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);

    wgpuBindGroupRelease(bg0);
}

void BoidsSim::step(WGPUCommandEncoder encoder) {
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
    uint32_t totalCells = m_gridW * m_gridH;
    uint32_t wgGrid = (totalCells + 255) / 256;

    for (int s = 0; s < m_stepsPerFrame; s++) {
        m_frameCounter++;
        uploadParams();

        WGPUBindGroup bg0 = buildGroup0();

        // 1. Clear grid
        {
            WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);
            wgpuComputePassEncoderSetPipeline(pass, m_clearGridPipeline);
            wgpuComputePassEncoderSetBindGroup(pass, 0, bg0, 0, nullptr);
            wgpuComputePassEncoderSetBindGroup(pass, 1, m_group1, 0, nullptr);
            wgpuComputePassEncoderSetBindGroup(pass, 2, m_group2, 0, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(pass, wgGrid, 1, 1);
            wgpuComputePassEncoderEnd(pass);
            wgpuComputePassEncoderRelease(pass);
        }

        // 2. Assign cells
        {
            WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);
            wgpuComputePassEncoderSetPipeline(pass, m_assignCellsPipeline);
            wgpuComputePassEncoderSetBindGroup(pass, 0, bg0, 0, nullptr);
            wgpuComputePassEncoderSetBindGroup(pass, 1, m_group1, 0, nullptr);
            wgpuComputePassEncoderSetBindGroup(pass, 2, m_group2, 0, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(pass, wgAgent, 1, 1);
            wgpuComputePassEncoderEnd(pass);
            wgpuComputePassEncoderRelease(pass);
        }

        // 3. Move agents (reads trailRead for food sensing)
        {
            WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);
            wgpuComputePassEncoderSetPipeline(pass, m_moveAgentsPipeline);
            wgpuComputePassEncoderSetBindGroup(pass, 0, bg0, 0, nullptr);
            wgpuComputePassEncoderSetBindGroup(pass, 1, m_group1, 0, nullptr);
            wgpuComputePassEncoderSetBindGroup(pass, 2, m_group2, 0, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(pass, wgAgent, 1, 1);
            wgpuComputePassEncoderEnd(pass);
            wgpuComputePassEncoderRelease(pass);
        }

        // 4. Diffuse texture (trailRead -> trailWrite)
        {
            WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);
            wgpuComputePassEncoderSetPipeline(pass, m_diffuseTexturePipeline);
            wgpuComputePassEncoderSetBindGroup(pass, 0, bg0, 0, nullptr);
            wgpuComputePassEncoderSetBindGroup(pass, 1, m_group1, 0, nullptr);
            wgpuComputePassEncoderSetBindGroup(pass, 2, m_group2, 0, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(pass, wgTex, hgTex, 1);
            wgpuComputePassEncoderEnd(pass);
            wgpuComputePassEncoderRelease(pass);
        }

        // 5. Copy trailWrite -> trailRead
        {
            WGPUImageCopyTexture src = {};
            src.texture = m_trailTextures.current == 0 ? m_trailTextures.texB : m_trailTextures.texA;
            WGPUImageCopyTexture dst = {};
            dst.texture = m_trailTextures.current == 0 ? m_trailTextures.texA : m_trailTextures.texB;
            WGPUExtent3D size = { params.width, params.height, 1 };
            wgpuCommandEncoderCopyTextureToTexture(encoder, &src, &dst, &size);
        }

        wgpuBindGroupRelease(bg0);

        // 6. Write trails (deposit/eat on diffused data)
        bg0 = buildGroup0();
        {
            WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);
            wgpuComputePassEncoderSetPipeline(pass, m_writeTrailsPipeline);
            wgpuComputePassEncoderSetBindGroup(pass, 0, bg0, 0, nullptr);
            wgpuComputePassEncoderSetBindGroup(pass, 1, m_group1, 0, nullptr);
            wgpuComputePassEncoderSetBindGroup(pass, 2, m_group2, 0, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(pass, wgAgent, 1, 1);
            wgpuComputePassEncoderEnd(pass);
            wgpuComputePassEncoderRelease(pass);
        }

        // 7. Swap trail ping-pong
        m_trailTextures.swap();

        wgpuBindGroupRelease(bg0);

        // 8. Render (trail -> output)
        bg0 = buildGroup0();
        {
            WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);
            wgpuComputePassEncoderSetPipeline(pass, m_renderPipeline);
            wgpuComputePassEncoderSetBindGroup(pass, 0, bg0, 0, nullptr);
            wgpuComputePassEncoderSetBindGroup(pass, 1, m_group1, 0, nullptr);
            wgpuComputePassEncoderSetBindGroup(pass, 2, m_group2, 0, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(pass, wgTex, hgTex, 1);
            wgpuComputePassEncoderEnd(pass);
            wgpuComputePassEncoderRelease(pass);
        }

        // 9. Swap output ping-pong
        m_outputTextures.swap();

        wgpuBindGroupRelease(bg0);
    }
}

void BoidsSim::reset() {
    m_needsReset = true;
}

WGPUTextureView BoidsSim::getOutputView() {
    return m_outputTextures.readView();
}

WGPUTexture BoidsSim::getOutputTexture() {
    return m_outputTextures.current == 0 ? m_outputTextures.texA : m_outputTextures.texB;
}

void BoidsSim::onGui() {
    ImGui::Text("Boids");
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
        if (ImGui::InputInt("Agents (reset)", &ac, 1000, 10000)) {
            if (ac < 256) ac = 256;
            if (ac > 500000) ac = 500000;
            if ((uint32_t)ac != m_agentCount) {
                m_agentCount = (uint32_t)ac;
                m_needsReset = true;
            }
        }
    }

    {
        float cs = m_cellSize;
        if (ImGui::SliderFloat("Cell Size (reset)", &cs, 10.0f, 100.0f)) {
            if (cs != m_cellSize) {
                m_cellSize = cs;
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
            m_maxSpeed[i]          = rf(0.1f, 10.0f);
            m_maxForce[i]          = rf(0.01f, 1.0f);
            m_typeSeparateRange[i] = rf(1.0f, 2000.0f);
            m_globalSeparateRange[i]= rf(1.0f, 2000.0f);
            m_alignRange[i]        = rf(1.0f, 5000.0f);
            m_attractRange[i]      = rf(1.0f, 10000.0f);
            m_foodSensorDist[i]    = rf(1.0f, 100.0f);
            m_sensorAngle[i]       = rf(0.01f, 3.14f);
            m_foodStrength[i]      = rf(0.0f, 5.0f);
            m_deposit[i]           = rf(0.001f, 0.5f);
            m_eat[i]               = rf(0.001f, 0.5f);
            m_diffuseRate[i]       = rf(0.0f, 1.0f);
            m_hue[i]               = rf(0.0f, 1.0f);
            m_saturation[i]        = rf(0.3f, 1.0f);
        }
    }

    static char presetName[64] = "default";
    ImGui::InputText("Preset Name", presetName, sizeof(presetName));

    if (ImGui::Button("Save Preset")) {
        std::map<std::string, std::vector<float>> data;
        data["agentCount"] = {(float)m_agentCount};
        data["cellSize"] = {m_cellSize};
        data["linkTypes"] = {m_linkTypes ? 1.0f : 0.0f};
        data["maxSpeed"]          = {m_maxSpeed[0], m_maxSpeed[1], m_maxSpeed[2], m_maxSpeed[3]};
        data["maxForce"]          = {m_maxForce[0], m_maxForce[1], m_maxForce[2], m_maxForce[3]};
        data["typeSeparateRange"] = {m_typeSeparateRange[0], m_typeSeparateRange[1], m_typeSeparateRange[2], m_typeSeparateRange[3]};
        data["globalSeparateRange"]= {m_globalSeparateRange[0], m_globalSeparateRange[1], m_globalSeparateRange[2], m_globalSeparateRange[3]};
        data["alignRange"]        = {m_alignRange[0], m_alignRange[1], m_alignRange[2], m_alignRange[3]};
        data["attractRange"]      = {m_attractRange[0], m_attractRange[1], m_attractRange[2], m_attractRange[3]};
        data["foodSensorDist"]    = {m_foodSensorDist[0], m_foodSensorDist[1], m_foodSensorDist[2], m_foodSensorDist[3]};
        data["sensorAngle"]       = {m_sensorAngle[0], m_sensorAngle[1], m_sensorAngle[2], m_sensorAngle[3]};
        data["foodStrength"]      = {m_foodStrength[0], m_foodStrength[1], m_foodStrength[2], m_foodStrength[3]};
        data["deposit"]           = {m_deposit[0], m_deposit[1], m_deposit[2], m_deposit[3]};
        data["eat"]               = {m_eat[0], m_eat[1], m_eat[2], m_eat[3]};
        data["diffuseRate"]       = {m_diffuseRate[0], m_diffuseRate[1], m_diffuseRate[2], m_diffuseRate[3]};
        data["hue"]               = {m_hue[0], m_hue[1], m_hue[2], m_hue[3]};
        data["saturation"]        = {m_saturation[0], m_saturation[1], m_saturation[2], m_saturation[3]};
        savePreset(std::string("boids_") + presetName, data);
    }
    ImGui::SameLine();
    if (ImGui::Button("Load Preset")) {
        auto data = loadPreset(std::string("boids_") + presetName);
        if (!data.empty()) {
            auto load4 = [&](const char* key, float* dst) {
                auto it = data.find(key);
                if (it != data.end()) for (size_t i = 0; i < 4 && i < it->second.size(); i++) dst[i] = it->second[i];
            };
            if (data.count("agentCount") && !data["agentCount"].empty()) {
                m_agentCount = (uint32_t)data["agentCount"][0];
                m_needsReset = true;
            }
            if (data.count("cellSize") && !data["cellSize"].empty()) {
                m_cellSize = data["cellSize"][0];
                m_needsReset = true;
            }
            if (data.count("linkTypes") && !data["linkTypes"].empty())
                m_linkTypes = data["linkTypes"][0] > 0.5f;
            load4("maxSpeed", m_maxSpeed);
            load4("maxForce", m_maxForce);
            load4("typeSeparateRange", m_typeSeparateRange);
            load4("globalSeparateRange", m_globalSeparateRange);
            load4("alignRange", m_alignRange);
            load4("attractRange", m_attractRange);
            load4("foodSensorDist", m_foodSensorDist);
            load4("sensorAngle", m_sensorAngle);
            load4("foodStrength", m_foodStrength);
            load4("deposit", m_deposit);
            load4("eat", m_eat);
            load4("diffuseRate", m_diffuseRate);
            load4("hue", m_hue);
            load4("saturation", m_saturation);
        }
    }

    ImGui::Checkbox("Link All Types", &m_linkTypes);

    if (m_linkTypes) {
        bool changed = false;
        changed |= ImGui::SliderFloat("Max Speed", &m_maxSpeed[0], 0.1f, 10.0f);
        changed |= ImGui::SliderFloat("Max Force", &m_maxForce[0], 0.01f, 1.0f);
        changed |= ImGui::SliderFloat("Type Sep Range", &m_typeSeparateRange[0], 1.0f, 2000.0f);
        changed |= ImGui::SliderFloat("Global Sep Range", &m_globalSeparateRange[0], 1.0f, 2000.0f);
        changed |= ImGui::SliderFloat("Align Range", &m_alignRange[0], 1.0f, 5000.0f);
        changed |= ImGui::SliderFloat("Attract Range", &m_attractRange[0], 1.0f, 10000.0f);
        changed |= ImGui::SliderFloat("Food Sensor Dist", &m_foodSensorDist[0], 1.0f, 100.0f);
        changed |= ImGui::SliderFloat("Sensor Angle", &m_sensorAngle[0], 0.01f, 3.14f);
        changed |= ImGui::SliderFloat("Food Strength", &m_foodStrength[0], 0.0f, 5.0f);
        changed |= ImGui::SliderFloat("Deposit", &m_deposit[0], 0.001f, 0.5f);
        changed |= ImGui::SliderFloat("Eat", &m_eat[0], 0.001f, 0.5f);
        changed |= ImGui::SliderFloat("Diffuse Rate", &m_diffuseRate[0], 0.0f, 1.0f);
        changed |= ImGui::SliderFloat("Hue", &m_hue[0], 0.0f, 1.0f);
        changed |= ImGui::SliderFloat("Saturation", &m_saturation[0], 0.0f, 1.0f);
        if (changed) {
            for (int i = 1; i < 4; i++) {
                m_maxSpeed[i]          = m_maxSpeed[0];
                m_maxForce[i]          = m_maxForce[0];
                m_typeSeparateRange[i] = m_typeSeparateRange[0];
                m_globalSeparateRange[i]= m_globalSeparateRange[0];
                m_alignRange[i]        = m_alignRange[0];
                m_attractRange[i]      = m_attractRange[0];
                m_foodSensorDist[i]    = m_foodSensorDist[0];
                m_sensorAngle[i]       = m_sensorAngle[0];
                m_foodStrength[i]      = m_foodStrength[0];
                m_deposit[i]           = m_deposit[0];
                m_eat[i]               = m_eat[0];
                m_diffuseRate[i]       = m_diffuseRate[0];
                m_hue[i]               = m_hue[0];
                m_saturation[i]        = m_saturation[0];
            }
        }
    } else {
        for (int t = 0; t < 4; t++) {
            char label[32];
            snprintf(label, sizeof(label), "Type %d", t);
            if (ImGui::TreeNode(label)) {
                ImGui::PushID(t);
                ImGui::SliderFloat("Max Speed", &m_maxSpeed[t], 0.1f, 10.0f);
                ImGui::SliderFloat("Max Force", &m_maxForce[t], 0.01f, 1.0f);
                ImGui::SliderFloat("Type Sep Range", &m_typeSeparateRange[t], 1.0f, 2000.0f);
                ImGui::SliderFloat("Global Sep Range", &m_globalSeparateRange[t], 1.0f, 2000.0f);
                ImGui::SliderFloat("Align Range", &m_alignRange[t], 1.0f, 5000.0f);
                ImGui::SliderFloat("Attract Range", &m_attractRange[t], 1.0f, 10000.0f);
                ImGui::SliderFloat("Food Sensor Dist", &m_foodSensorDist[t], 1.0f, 100.0f);
                ImGui::SliderFloat("Sensor Angle", &m_sensorAngle[t], 0.01f, 3.14f);
                ImGui::SliderFloat("Food Strength", &m_foodStrength[t], 0.0f, 5.0f);
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

void BoidsSim::shutdown() {
    if (m_group1) wgpuBindGroupRelease(m_group1);
    if (m_group2) wgpuBindGroupRelease(m_group2);
    if (m_group0Layout) wgpuBindGroupLayoutRelease(m_group0Layout);
    if (m_group1Layout) wgpuBindGroupLayoutRelease(m_group1Layout);
    if (m_group2Layout) wgpuBindGroupLayoutRelease(m_group2Layout);
    if (m_pipelineLayout) wgpuPipelineLayoutRelease(m_pipelineLayout);

    if (m_resetTexturePipeline)   wgpuComputePipelineRelease(m_resetTexturePipeline);
    if (m_resetAgentsPipeline)    wgpuComputePipelineRelease(m_resetAgentsPipeline);
    if (m_clearGridPipeline)      wgpuComputePipelineRelease(m_clearGridPipeline);
    if (m_assignCellsPipeline)    wgpuComputePipelineRelease(m_assignCellsPipeline);
    if (m_moveAgentsPipeline)     wgpuComputePipelineRelease(m_moveAgentsPipeline);
    if (m_writeTrailsPipeline)    wgpuComputePipelineRelease(m_writeTrailsPipeline);
    if (m_diffuseTexturePipeline) wgpuComputePipelineRelease(m_diffuseTexturePipeline);
    if (m_renderPipeline)         wgpuComputePipelineRelease(m_renderPipeline);

    if (m_shaderModule) wgpuShaderModuleRelease(m_shaderModule);
    if (m_agentBuffer) { wgpuBufferDestroy(m_agentBuffer); wgpuBufferRelease(m_agentBuffer); }
    if (m_uniformBuffer) { wgpuBufferDestroy(m_uniformBuffer); wgpuBufferRelease(m_uniformBuffer); }
    if (m_cellCountBuffer) { wgpuBufferDestroy(m_cellCountBuffer); wgpuBufferRelease(m_cellCountBuffer); }
    if (m_cellAgentsBuffer) { wgpuBufferDestroy(m_cellAgentsBuffer); wgpuBufferRelease(m_cellAgentsBuffer); }

    m_trailTextures.destroy();
    m_outputTextures.destroy();
}
