#pragma once
#include <webgpu/webgpu.h>
#include <string>

// Read back a GPU texture to CPU and save as PNG
bool exportTextureToPNG(WGPUDevice device, WGPUQueue queue,
                        WGPUTexture texture, uint32_t width, uint32_t height,
                        const std::string& filename);
