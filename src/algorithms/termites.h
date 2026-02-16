#pragma once
#include "../simulation.h"
#include "../compute_pass.h"
#include <vector>
#include <cstdint>

class TermitesSim : public Simulation {
public:
    const char* name() const override { return "Termites"; }
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

    PingPongTextures m_trailTextures;   // pheromone (decays, for sensing)
    PingPongTextures m_moundTextures;   // persistent deposits (no decay)
    PingPongTextures m_outputTextures;  // rgba8unorm render

    WGPUBuffer m_agentBuffer = nullptr;
    WGPUBuffer m_uniformBuffer = nullptr;

    WGPUShaderModule m_shaderModule = nullptr;
    WGPUPipelineLayout m_pipelineLayout = nullptr;
    WGPUBindGroupLayout m_group0Layout = nullptr;
    WGPUBindGroupLayout m_group1Layout = nullptr;

    WGPUComputePipeline m_resetTexturePipeline = nullptr;
    WGPUComputePipeline m_resetAgentsPipeline = nullptr;
    WGPUComputePipeline m_moveAgentsPipeline = nullptr;
    WGPUComputePipeline m_decayTexturePipeline = nullptr;
    WGPUComputePipeline m_writeTrailsPipeline = nullptr;
    WGPUComputePipeline m_renderPipeline = nullptr;

    WGPUBindGroup m_group1 = nullptr;

    uint32_t m_agentCount = 100000;
    uint32_t m_frameCounter = 0;
    int m_stepsPerFrame = 1;
    bool m_needsReset = true;
    bool m_doStep = false;
    bool m_linkTypes = true;

    float m_senseAngle[4]    = {45.0f, 45.0f, 45.0f, 45.0f};
    float m_senseDistance[4]  = {20.5f, 20.5f, 20.5f, 20.5f};
    float m_turnAngle[4]     = {15.0f, 15.0f, 15.0f, 15.0f};
    float m_moveSpeed[4]     = {0.5f, 0.5f, 0.5f, 0.5f};
    float m_deposit[4]       = {0.5f, 0.5f, 0.5f, 0.5f};
    float m_depositRate[4]   = {0.09f, 0.09f, 0.09f, 0.09f};
    float m_decayRate[4]     = {0.95f, 0.95f, 0.95f, 0.95f};
    float m_hue[4]           = {0.0f, 0.25f, 0.5f, 0.75f};
    float m_saturation[4]    = {0.7f, 0.7f, 0.7f, 0.7f};
    float m_typeWeight[4]    = {25.0f, 25.0f, 25.0f, 25.0f};

    struct GpuParams {
        uint32_t rezX, rezY, agentsCount, time;
        float senseAngles[4];
        float senseDistances[4];
        float turnAngles[4];
        float moveSpeeds[4];
        float depositAmounts[4];
        float depositRates[4];
        float decayRates[4];
        float hues[4];
        float saturations[4];
        float typeRatios[4];
    };
    static_assert(sizeof(GpuParams) == 176, "GpuParams must be 176 bytes");
};
