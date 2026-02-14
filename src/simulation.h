#pragma once
#include <webgpu/webgpu.h>
#include <string>

struct SimParams {
    uint32_t width = 512;
    uint32_t height = 512;
    bool paused = false;
    float speed = 1.0f; // steps per frame
};

class Simulation {
public:
    virtual ~Simulation() = default;
    virtual const char* name() const = 0;
    virtual void init(WGPUDevice device, WGPUQueue queue, uint32_t w, uint32_t h) = 0;
    virtual void step(WGPUCommandEncoder encoder) = 0;
    virtual void reset() = 0;
    virtual WGPUTextureView getOutputView() = 0;
    virtual WGPUTexture getOutputTexture() = 0;
    virtual void onGui() = 0; // ImGui controls
    virtual void shutdown() = 0;

    SimParams params;
};
