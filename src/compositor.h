#pragma once
#include <webgpu/webgpu.h>
#include "simulation.h"
#include <vector>
#include <string>
#include <cstdint>

enum class BlendMode : uint32_t {
    Additive = 0,
    Multiply = 1,
    Screen   = 2,
    Normal   = 3,
};

struct Layer {
    Simulation* sim = nullptr;
    bool enabled = false;
    float opacity = 1.0f;
    BlendMode blendMode = BlendMode::Additive;
};

class Compositor {
public:
    void init(WGPUDevice device, WGPUQueue queue, uint32_t w, uint32_t h);
    void resize(uint32_t w, uint32_t h);
    void composite(WGPUCommandEncoder encoder);
    WGPUTextureView getOutputView() const;
    WGPUTexture getOutputTexture() const { return m_texA; }
    void onGui();
    void shutdown();

    std::vector<Layer> layers;

private:
    void createTextures();
    void destroyTextures();
    void createPipelines();

    WGPUDevice m_device = nullptr;
    WGPUQueue m_queue = nullptr;
    uint32_t m_width = 0, m_height = 0;

    // Ping-pong textures for iterative blending
    WGPUTexture m_texA = nullptr, m_texB = nullptr;
    WGPUTextureView m_viewA = nullptr, m_viewB = nullptr;
    int m_current = 0; // tracks which tex has the latest result

    WGPUShaderModule m_shaderModule = nullptr;
    WGPUPipelineLayout m_pipelineLayout = nullptr;
    WGPUBindGroupLayout m_bindGroupLayout = nullptr;
    WGPUComputePipeline m_pipeline = nullptr;
    WGPUBuffer m_uniformBuffer = nullptr;

    struct GpuParams {
        uint32_t width, height, blendMode;
        float opacity;
        uint32_t isFirstLayer;
        uint32_t _pad[3];
    };
    static_assert(sizeof(GpuParams) == 32, "Compositor GpuParams must be 32 bytes");
};
