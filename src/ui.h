#pragma once
#include "gpu_context.h"
#include <webgpu/webgpu.h>

struct UI {
    void init(GpuContext& ctx);
    void beginFrame();
    void endFrame(WGPURenderPassEncoder renderPass);
    void shutdown();
};
