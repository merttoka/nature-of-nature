#pragma once
#include "../simulation.h"
#include "../compute_pass.h"
#include <vector>
#include <cstdint>

class BoidsSim : public Simulation {
public:
    const char* name() const override { return "Boids"; }
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
    PingPongTextures m_trailTextures;
    PingPongTextures m_outputTextures;

    // Buffers
    WGPUBuffer m_agentBuffer = nullptr;
    WGPUBuffer m_uniformBuffer = nullptr;
    WGPUBuffer m_cellCountBuffer = nullptr;
    WGPUBuffer m_cellAgentsBuffer = nullptr;

    // Pipelines
    WGPUShaderModule m_shaderModule = nullptr;
    WGPUPipelineLayout m_pipelineLayout = nullptr;
    WGPUBindGroupLayout m_group0Layout = nullptr;
    WGPUBindGroupLayout m_group1Layout = nullptr;
    WGPUBindGroupLayout m_group2Layout = nullptr;

    WGPUComputePipeline m_resetTexturePipeline = nullptr;
    WGPUComputePipeline m_resetAgentsPipeline = nullptr;
    WGPUComputePipeline m_clearGridPipeline = nullptr;
    WGPUComputePipeline m_assignCellsPipeline = nullptr;
    WGPUComputePipeline m_moveAgentsPipeline = nullptr;
    WGPUComputePipeline m_writeTrailsPipeline = nullptr;
    WGPUComputePipeline m_diffuseTexturePipeline = nullptr;
    WGPUComputePipeline m_renderPipeline = nullptr;

    WGPUBindGroup m_group1 = nullptr;
    WGPUBindGroup m_group2 = nullptr;

    // Params
    uint32_t m_agentCount = 20000;
    uint32_t m_frameCounter = 0;
    int m_stepsPerFrame = 1;
    bool m_needsReset = true;
    bool m_doStep = false;
    bool m_linkTypes = true;

    // Spatial hash
    float m_cellSize = 30.0f;
    uint32_t m_gridW = 0, m_gridH = 0;
    static constexpr uint32_t MAX_PER_CELL = 64;

    // Per-type params (4 types)
    float m_maxSpeed[4]           = {2.0f, 2.0f, 2.0f, 2.0f};
    float m_maxForce[4]           = {0.1f, 0.1f, 0.1f, 0.1f};
    float m_typeSeparateRange[4]  = {100.0f, 100.0f, 100.0f, 100.0f};
    float m_globalSeparateRange[4]= {50.0f, 50.0f, 50.0f, 50.0f};
    float m_alignRange[4]         = {400.0f, 400.0f, 400.0f, 400.0f};
    float m_attractRange[4]       = {900.0f, 900.0f, 900.0f, 900.0f};
    float m_foodSensorDist[4]     = {15.0f, 15.0f, 15.0f, 15.0f};
    float m_sensorAngle[4]        = {0.5f, 0.5f, 0.5f, 0.5f};
    float m_foodStrength[4]       = {0.5f, 0.5f, 0.5f, 0.5f};
    float m_deposit[4]            = {0.02f, 0.02f, 0.02f, 0.02f};
    float m_eat[4]                = {0.01f, 0.01f, 0.01f, 0.01f};
    float m_diffuseRate[4]        = {0.95f, 0.95f, 0.95f, 0.95f};
    float m_hue[4]                = {0.0f, 0.25f, 0.5f, 0.75f};
    float m_saturation[4]         = {0.7f, 0.7f, 0.7f, 0.7f};

    // GPU uniform struct (must match shader)
    struct GpuParams {
        uint32_t rezX, rezY, agentsCount, time;
        float cellSize, gridWf, gridHf, maxPerCellf;
        float maxSpeeds[4];
        float maxForces[4];
        float typeSeparateRanges[4];
        float globalSeparateRanges[4];
        float alignRanges[4];
        float attractRanges[4];
        float foodSensorDistances[4];
        float sensorAngles[4];
        float foodStrengths[4];
        float depositAmounts[4];
        float eatAmounts[4];
        float diffuseRates[4];
        float hues[4];
        float saturations[4];
    };
    static_assert(sizeof(GpuParams) == 256, "GpuParams must be 256 bytes");
};
