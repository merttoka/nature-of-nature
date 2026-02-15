#pragma once
#include <webgpu/webgpu.h>
#include <GLFW/glfw3.h>

struct GpuContext {
    GLFWwindow* window = nullptr;
    WGPUInstance instance = nullptr;
    WGPUSurface surface = nullptr;
    WGPUAdapter adapter = nullptr;
    WGPUDevice device = nullptr;
    WGPUQueue queue = nullptr;
    WGPUTextureFormat surfaceFormat = WGPUTextureFormat_BGRA8Unorm;
    uint32_t width = 1280;
    uint32_t height = 720;

    bool init(uint32_t w, uint32_t h, const char* title);
    void configureSurface();
    void updateSize();
    WGPUTextureView getNextSurfaceTextureView();
    void present();
    void shutdown();
};
