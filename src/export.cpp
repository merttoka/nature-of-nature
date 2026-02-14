#include "export.h"
#include <webgpu/wgpu.h>
#include <cstdio>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

bool exportTextureToPNG(WGPUDevice device, WGPUQueue queue,
                        WGPUTexture texture, uint32_t width, uint32_t height,
                        const std::string& filename)
{
    uint32_t bytesPerRow = ((width * 4 + 255) / 256) * 256; // 256-byte aligned
    uint64_t bufferSize = bytesPerRow * height;

    WGPUBufferDescriptor bufDesc = {};
    bufDesc.size = bufferSize;
    bufDesc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
    WGPUBuffer readbackBuf = wgpuDeviceCreateBuffer(device, &bufDesc);

    WGPUCommandEncoderDescriptor encDesc = {};
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, &encDesc);

    WGPUImageCopyTexture src = {};
    src.texture = texture;

    WGPUImageCopyBuffer dst = {};
    dst.buffer = readbackBuf;
    dst.layout.bytesPerRow = bytesPerRow;
    dst.layout.rowsPerImage = height;

    WGPUExtent3D size = { width, height, 1 };
    wgpuCommandEncoderCopyTextureToBuffer(encoder, &src, &dst, &size);

    WGPUCommandBufferDescriptor cbDesc = {};
    WGPUCommandBuffer cmdBuf = wgpuCommandEncoderFinish(encoder, &cbDesc);
    wgpuQueueSubmit(queue, 1, &cmdBuf);
    wgpuCommandBufferRelease(cmdBuf);
    wgpuCommandEncoderRelease(encoder);

    // Map buffer synchronously
    struct MapData { bool done = false; WGPUBufferMapAsyncStatus status; };
    MapData mapData;
    wgpuBufferMapAsync(readbackBuf, WGPUMapMode_Read, 0, bufferSize,
        [](WGPUBufferMapAsyncStatus status, void* ud) {
            auto* data = (MapData*)ud;
            data->status = status;
            data->done = true;
        }, &mapData);

    // Poll device until mapped
    while (!mapData.done) {
        wgpuDevicePoll(device, true, nullptr); // wgpu-native extension
    }

    bool ok = false;
    if (mapData.status == WGPUBufferMapAsyncStatus_Success) {
        const uint8_t* mapped = (const uint8_t*)wgpuBufferGetConstMappedRange(readbackBuf, 0, bufferSize);

        // Remove row padding
        std::vector<uint8_t> pixels(width * height * 4);
        for (uint32_t y = 0; y < height; y++) {
            memcpy(&pixels[y * width * 4], &mapped[y * bytesPerRow], width * 4);
        }
        wgpuBufferUnmap(readbackBuf);

        ok = stbi_write_png(filename.c_str(), width, height, 4, pixels.data(), width * 4) != 0;
        if (ok) printf("Exported: %s\n", filename.c_str());
        else fprintf(stderr, "Failed to write PNG: %s\n", filename.c_str());
    }

    wgpuBufferRelease(readbackBuf);
    return ok;
}
