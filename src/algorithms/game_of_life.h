#pragma once
#include "simulation.h"
#include "compute_pass.h"
#include <vector>
#include <cstdint>

class GameOfLife : public Simulation {
public:
    const char* name() const override { return "Game of Life"; }
    void init(WGPUDevice device, WGPUQueue queue, uint32_t w, uint32_t h) override;
    void step(WGPUCommandEncoder encoder) override;
    void reset() override;
    WGPUTextureView getOutputView() override;
    WGPUTexture getOutputTexture() override;
    void onGui() override;
    void shutdown() override;

private:
    void seedRandom();
    void seedGlider(int x, int y);
    void uploadState();
    void rebuildBindGroups();

    WGPUDevice m_device = nullptr;
    WGPUQueue m_queue = nullptr;
    PingPongTextures m_textures;
    WGPUComputePipeline m_pipeline = nullptr;
    WGPUBindGroupLayout m_bindGroupLayout = nullptr;
    WGPUBindGroup m_bindGroupA = nullptr; // read A, write B
    WGPUBindGroup m_bindGroupB = nullptr; // read B, write A

    // CPU state for seeding
    std::vector<uint8_t> m_cpuState;

    float m_fillDensity = 0.3f;
    int m_stepsPerFrame = 1;
};
