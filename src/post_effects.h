#pragma once
#include <webgpu/webgpu.h>
#include "compute_pass.h"
#include <cstdint>

class PostEffects {
public:
    void init(WGPUDevice device, WGPUQueue queue, uint32_t w, uint32_t h);
    void resize(uint32_t w, uint32_t h);
    void apply(WGPUCommandEncoder encoder, WGPUTextureView simOutput);
    WGPUTextureView getOutputView() const;
    void onGui();
    void shutdown();

    // Params
    float brightness = 0.0f;   // -1 to 1
    float contrast   = 1.0f;   // 0 to 3
    float bloomThreshold = 0.6f; // 0 to 1
    float bloomIntensity = 0.3f; // 0 to 2
    float bloomRadius    = 4.0f; // 1 to 20 (sigma)
    float saturationPost = 1.0f; // 0 to 2
    float vignette       = 0.0f; // 0 to 1

private:
    void createTextures();
    void createPipelines();
    void destroyTextures();

    WGPUDevice m_device = nullptr;
    WGPUQueue m_queue = nullptr;
    uint32_t m_width = 0, m_height = 0;

    // Textures: bloomA (h-blur result), bloomB (v-blur result), output (final)
    WGPUTexture m_bloomATex = nullptr, m_bloomBTex = nullptr, m_outputTex = nullptr;
    WGPUTextureView m_bloomAView = nullptr, m_bloomBView = nullptr, m_outputView = nullptr;

    // Pipelines
    WGPUShaderModule m_shaderModule = nullptr;
    WGPUPipelineLayout m_pipelineLayout = nullptr;
    WGPUBindGroupLayout m_bindGroupLayout = nullptr;
    WGPUComputePipeline m_bloomHPipeline = nullptr;
    WGPUComputePipeline m_bloomVPipeline = nullptr;
    WGPUComputePipeline m_compositePipeline = nullptr;

    // Uniform buffer
    WGPUBuffer m_uniformBuffer = nullptr;

    struct GpuParams {
        uint32_t width, height;
        float brightness, contrast;
        float bloomThreshold, bloomIntensity, bloomRadius, saturationPost;
        float vignette;
        float _pad[3];
    };
    static_assert(sizeof(GpuParams) == 48, "PostEffects GpuParams must be 48 bytes");
};
