#pragma once
#include <webgpu/webgpu.h>
#include <string>

// Manages a pair of ping-pong storage textures for compute shaders
struct PingPongTextures {
    WGPUTexture texA = nullptr;
    WGPUTexture texB = nullptr;
    WGPUTextureView viewA = nullptr;
    WGPUTextureView viewB = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;
    int current = 0; // 0 = A is read, B is write; 1 = swapped

    void init(WGPUDevice device, uint32_t w, uint32_t h,
              WGPUTextureFormat format = WGPUTextureFormat_RGBA8Unorm);
    void swap();
    WGPUTextureView readView() const;
    WGPUTextureView writeView() const;
    void destroy();
};

// Helper to create a compute pipeline from WGSL shader file
WGPUComputePipeline createComputePipeline(
    WGPUDevice device,
    const char* shaderPath,
    const char* entryPoint,
    WGPUBindGroupLayout layout);

// Helper to load shader file as string
std::string loadShaderFile(const char* path);

// Helper to create a bind group layout for ping-pong compute
WGPUBindGroupLayout createPingPongBindGroupLayout(WGPUDevice device, bool withUniform = true);

// Create bind group for ping-pong textures + optional uniform buffer
WGPUBindGroup createPingPongBindGroup(
    WGPUDevice device,
    WGPUBindGroupLayout layout,
    WGPUTextureView readView,
    WGPUTextureView writeView,
    WGPUBuffer uniformBuffer = nullptr,
    uint64_t uniformSize = 0);
