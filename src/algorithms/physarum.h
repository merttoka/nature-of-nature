#pragma once
#include "../simulation.h"
#include "../compute_pass.h"
#include <vector>
#include <cstdint>

class PhysarumSim : public Simulation {
public:
    const char* name() const override { return "Physarum"; }
    void init(WGPUDevice device, WGPUQueue queue, uint32_t w, uint32_t h) override;
    void step(WGPUCommandEncoder encoder) override;
    void reset() override;
    WGPUTextureView getOutputView() override;
    WGPUTexture getOutputTexture() override;
    void onGui() override;
    void shutdown() override;

private:
    void createPipelines();
    void createBuffers();
    void clearTextures();
    void dispatchReset(WGPUCommandEncoder encoder);
    void uploadParams();
    WGPUBindGroup buildGroup0();

    WGPUDevice m_device = nullptr;
    WGPUQueue m_queue = nullptr;

    // Textures
    PingPongTextures m_trailTextures;   // rgba16float
    PingPongTextures m_outputTextures;  // rgba8unorm

    // Buffers
    WGPUBuffer m_agentBuffer = nullptr;
    WGPUBuffer m_uniformBuffer = nullptr;

    // Pipelines (all share same layout)
    WGPUShaderModule m_shaderModule = nullptr;
    WGPUPipelineLayout m_pipelineLayout = nullptr;
    WGPUBindGroupLayout m_group0Layout = nullptr;
    WGPUBindGroupLayout m_group1Layout = nullptr;

    WGPUComputePipeline m_resetTexturePipeline = nullptr;
    WGPUComputePipeline m_resetAgentsPipeline = nullptr;
    WGPUComputePipeline m_moveAgentsPipeline = nullptr;
    WGPUComputePipeline m_writeTrailsPipeline = nullptr;
    WGPUComputePipeline m_diffuseTexturePipeline = nullptr;
    WGPUComputePipeline m_renderPipeline = nullptr;

    WGPUBindGroup m_group1 = nullptr; // agents buffer — stable

    // Params
    uint32_t m_agentCount = 100000;
    uint32_t m_frameCounter = 0;
    int m_stepsPerFrame = 1;
    bool m_needsReset = true;
    bool m_doStep = false;
    bool m_linkTypes = true;

    // Per-type params (indices 0-3)
    float m_senseAngle[4]    = {22.5f, 22.5f, 22.5f, 22.5f};       // degrees
    float m_senseDistance[4]  = {9.0f, 9.0f, 9.0f, 9.0f};
    float m_turnAngle[4]     = {45.0f, 45.0f, 45.0f, 45.0f};       // degrees
    float m_moveSpeed[4]     = {0.4f, 0.4f, 0.4f, 0.4f};
    float m_deposit[4]       = {0.01f, 0.01f, 0.01f, 0.01f};
    float m_eat[4]           = {0.05f, 0.05f, 0.05f, 0.05f};
    float m_diffuseRate[4]   = {0.95f, 0.95f, 0.95f, 0.95f};
    float m_hue[4]           = {0.0f, 0.0f, 0.0f, 0.0f};
    float m_saturation[4]    = {0.5f, 0.5f, 0.5f, 0.5f};

    // GPU uniform struct (must match shader)
    struct GpuParams {
        uint32_t rezX, rezY, agentsCount, time;
        float senseAngles[4];
        float senseDistances[4];
        float turnAngles[4];
        float moveSpeeds[4];
        float depositAmounts[4];
        float eatAmounts[4];
        float diffuseRates[4];
        float hues[4];
        float saturations[4];
    };
    static_assert(sizeof(GpuParams) == 160, "GpuParams must be 160 bytes");
};
