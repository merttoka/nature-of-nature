#include "game_of_life.h"
#include <imgui.h>
#include <cstdlib>
#include <ctime>
#include <cstring>

void GameOfLife::init(WGPUDevice device, WGPUQueue queue, uint32_t w, uint32_t h) {
    m_device = device;
    m_queue = queue;
    params.width = w;
    params.height = h;

    m_textures.init(device, w, h);

    // Bind group layout: texture_2d (read) + storage texture (write)
    m_bindGroupLayout = createPingPongBindGroupLayout(device, false);

    // Pipeline
    m_pipeline = createComputePipeline(device, "shaders/game_of_life.wgsl", "main", m_bindGroupLayout);

    rebuildBindGroups();
    seedRandom();
}

void GameOfLife::rebuildBindGroups() {
    if (m_bindGroupA) wgpuBindGroupRelease(m_bindGroupA);
    if (m_bindGroupB) wgpuBindGroupRelease(m_bindGroupB);
    m_bindGroupA = createPingPongBindGroup(m_device, m_bindGroupLayout,
        m_textures.viewA, m_textures.viewB);
    m_bindGroupB = createPingPongBindGroup(m_device, m_bindGroupLayout,
        m_textures.viewB, m_textures.viewA);
}

void GameOfLife::seedRandom() {
    srand((unsigned)time(nullptr));
    uint32_t w = params.width, h = params.height;
    m_cpuState.resize(w * h * 4);
    for (uint32_t i = 0; i < w * h; i++) {
        uint8_t alive = ((float)rand() / RAND_MAX) < m_fillDensity ? 255 : 0;
        m_cpuState[i * 4 + 0] = alive;
        m_cpuState[i * 4 + 1] = alive;
        m_cpuState[i * 4 + 2] = alive;
        m_cpuState[i * 4 + 3] = 255;
    }
    uploadState();
}

void GameOfLife::seedGlider(int x, int y) {
    uint32_t w = params.width;
    auto set = [&](int dx, int dy) {
        uint32_t idx = ((y + dy) * w + (x + dx)) * 4;
        m_cpuState[idx] = 255; m_cpuState[idx+1] = 255; m_cpuState[idx+2] = 255;
    };
    // Standard glider pattern
    set(1, 0); set(2, 1); set(0, 2); set(1, 2); set(2, 2);
}

void GameOfLife::uploadState() {
    // Write to current read texture via queue
    WGPUImageCopyTexture dst = {};
    dst.texture = m_textures.current == 0 ? m_textures.texA : m_textures.texB;
    dst.mipLevel = 0;

    WGPUTextureDataLayout layout = {};
    layout.bytesPerRow = params.width * 4;
    layout.rowsPerImage = params.height;

    WGPUExtent3D size = { params.width, params.height, 1 };
    wgpuQueueWriteTexture(m_queue, &dst, m_cpuState.data(),
                          m_cpuState.size(), &layout, &size);
}

void GameOfLife::step(WGPUCommandEncoder encoder) {
    if (params.paused) return;

    for (int i = 0; i < m_stepsPerFrame; i++) {
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);
        wgpuComputePassEncoderSetPipeline(pass, m_pipeline);

        WGPUBindGroup bg = m_textures.current == 0 ? m_bindGroupA : m_bindGroupB;
        wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);

        uint32_t wgX = (params.width + 7) / 8;
        uint32_t wgY = (params.height + 7) / 8;
        wgpuComputePassEncoderDispatchWorkgroups(pass, wgX, wgY, 1);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);

        m_textures.swap();
    }
}

void GameOfLife::reset() {
    m_textures.current = 0;
    seedRandom();
}

WGPUTextureView GameOfLife::getOutputView() {
    return m_textures.readView();
}

WGPUTexture GameOfLife::getOutputTexture() {
    return m_textures.current == 0 ? m_textures.texA : m_textures.texB;
}

void GameOfLife::onGui() {
    ImGui::Text("Game of Life");
    ImGui::Separator();

    if (ImGui::Button(params.paused ? "Play" : "Pause")) {
        params.paused = !params.paused;
    }
    ImGui::SameLine();
    if (ImGui::Button("Step")) {
        params.paused = true;
        // Will do one step next frame via flag — simplified: just unpause for 1 frame
        // Actually let's just directly dispatch. For now, toggle pause off then on.
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset")) {
        reset();
    }

    ImGui::SliderInt("Steps/Frame", &m_stepsPerFrame, 1, 20);
    if (ImGui::SliderFloat("Fill Density", &m_fillDensity, 0.01f, 0.99f)) {
        // Will apply on next reset
    }

    if (ImGui::Button("Seed Glider")) {
        m_textures.current = 0;
        memset(m_cpuState.data(), 0, m_cpuState.size());
        // Set alpha to 255
        for (uint32_t i = 0; i < params.width * params.height; i++)
            m_cpuState[i * 4 + 3] = 255;
        seedGlider(10, 10);
        seedGlider(30, 30);
        seedGlider(50, 20);
        uploadState();
    }
}

void GameOfLife::shutdown() {
    if (m_bindGroupA) wgpuBindGroupRelease(m_bindGroupA);
    if (m_bindGroupB) wgpuBindGroupRelease(m_bindGroupB);
    if (m_bindGroupLayout) wgpuBindGroupLayoutRelease(m_bindGroupLayout);
    if (m_pipeline) wgpuComputePipelineRelease(m_pipeline);
    m_textures.destroy();
}
