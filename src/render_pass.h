#pragma once
#include <webgpu/webgpu.h>

// Fullscreen quad renderer — samples a texture and draws to screen
struct RenderPass {
    WGPURenderPipeline pipeline = nullptr;
    WGPUBindGroupLayout bindGroupLayout = nullptr;
    WGPUSampler sampler = nullptr;

    void init(WGPUDevice device, WGPUTextureFormat surfaceFormat);
    WGPUBindGroup createBindGroup(WGPUDevice device, WGPUTextureView textureView);
    void draw(WGPUCommandEncoder encoder, WGPUTextureView targetView,
              WGPUBindGroup bindGroup);
    void shutdown();
};
