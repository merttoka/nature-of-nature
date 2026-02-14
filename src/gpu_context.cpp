#include "gpu_context.h"
#include <glfw3webgpu.h>
#include <cstdio>
#include <cstdlib>
#include <cassert>

static void onDeviceError(WGPUErrorType type, const char* message, void* userdata) {
    fprintf(stderr, "[WebGPU Error] type=%d: %s\n", (int)type, message);
}

bool GpuContext::init(uint32_t w, uint32_t h, const char* title) {
    width = w;
    height = h;

    if (!glfwInit()) return false;
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!window) return false;

    // Instance
    WGPUInstanceDescriptor instanceDesc = {};
    instance = wgpuCreateInstance(&instanceDesc);
    if (!instance) { fprintf(stderr, "Failed to create WebGPU instance\n"); return false; }

    // Surface from GLFW
    surface = glfwGetWGPUSurface(instance, window);
    if (!surface) { fprintf(stderr, "Failed to get WebGPU surface\n"); return false; }

    // Adapter (synchronous request via callback)
    WGPURequestAdapterOptions adapterOpts = {};
    adapterOpts.compatibleSurface = surface;
    adapterOpts.powerPreference = WGPUPowerPreference_HighPerformance;

    struct AdapterData { WGPUAdapter adapter = nullptr; bool done = false; };
    AdapterData adapterData;

    wgpuInstanceRequestAdapter(instance, &adapterOpts,
        [](WGPURequestAdapterStatus status, WGPUAdapter adapter, const char* message, void* ud) {
            auto* data = (AdapterData*)ud;
            if (status == WGPURequestAdapterStatus_Success) data->adapter = adapter;
            else fprintf(stderr, "Adapter request failed: %s\n", message ? message : "unknown");
            data->done = true;
        }, &adapterData);

    // wgpu-native completes synchronously, but poll just in case
    assert(adapterData.done);
    adapter = adapterData.adapter;
    if (!adapter) return false;

    // Device (synchronous request via callback)
    WGPUDeviceDescriptor deviceDesc = {};
    deviceDesc.label = "nature-of-nature device";

    // Use default device limits (no explicit required limits)
    // Setting requiredLimits with zero-initialized "min" fields causes validation
    // errors because 0 is stricter than what the GPU supports.

    struct DeviceData { WGPUDevice device = nullptr; bool done = false; };
    DeviceData deviceData;

    wgpuAdapterRequestDevice(adapter, &deviceDesc,
        [](WGPURequestDeviceStatus status, WGPUDevice device, const char* message, void* ud) {
            auto* data = (DeviceData*)ud;
            if (status == WGPURequestDeviceStatus_Success) data->device = device;
            else fprintf(stderr, "Device request failed: %s\n", message ? message : "unknown");
            data->done = true;
        }, &deviceData);

    assert(deviceData.done);
    device = deviceData.device;
    if (!device) return false;

    wgpuDeviceSetUncapturedErrorCallback(device, onDeviceError, nullptr);
    queue = wgpuDeviceGetQueue(device);

    // Surface format — use preferred format API
    surfaceFormat = wgpuSurfaceGetPreferredFormat(surface, adapter);

    configureSurface();
    return true;
}

void GpuContext::configureSurface() {
    WGPUSurfaceConfiguration config = {};
    config.device = device;
    config.format = surfaceFormat;
    config.usage = WGPUTextureUsage_RenderAttachment;
    config.width = width;
    config.height = height;
    config.presentMode = WGPUPresentMode_Fifo;
    wgpuSurfaceConfigure(surface, &config);
}

WGPUTextureView GpuContext::getNextSurfaceTextureView() {
    WGPUSurfaceTexture surfTex;
    wgpuSurfaceGetCurrentTexture(surface, &surfTex);
    if (surfTex.status != WGPUSurfaceGetCurrentTextureStatus_Success) return nullptr;

    WGPUTextureViewDescriptor viewDesc = {};
    viewDesc.format = surfaceFormat;
    viewDesc.dimension = WGPUTextureViewDimension_2D;
    viewDesc.mipLevelCount = 1;
    viewDesc.arrayLayerCount = 1;
    return wgpuTextureCreateView(surfTex.texture, &viewDesc);
}

void GpuContext::present() {
    wgpuSurfacePresent(surface);
}

void GpuContext::shutdown() {
    if (queue) wgpuQueueRelease(queue);
    if (device) wgpuDeviceRelease(device);
    if (adapter) wgpuAdapterRelease(adapter);
    if (surface) wgpuSurfaceRelease(surface);
    if (instance) wgpuInstanceRelease(instance);
    if (window) glfwDestroyWindow(window);
    glfwTerminate();
}
