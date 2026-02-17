#include "export.h"
#include <webgpu/wgpu.h>
#include <cstdio>
#include <cstring>

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

    while (!mapData.done) {
        wgpuDevicePoll(device, true, nullptr);
    }

    bool ok = false;
    if (mapData.status == WGPUBufferMapAsyncStatus_Success) {
        const uint8_t* mapped = (const uint8_t*)wgpuBufferGetConstMappedRange(readbackBuf, 0, bufferSize);

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

// --- AsyncExporter ---

void AsyncExporter::start() {
    if (m_running) return;
    m_running = true;
    m_thread = std::thread(&AsyncExporter::workerLoop, this);
}

void AsyncExporter::stop() {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_running = false;
    }
    m_cv.notify_one();
    if (m_thread.joinable()) m_thread.join();
}

void AsyncExporter::enqueue(std::vector<uint8_t>&& pixels, uint32_t w, uint32_t h,
                             const std::string& filename) {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_jobs.push({std::move(pixels), w, h, filename});
        m_pending++;
    }
    m_cv.notify_one();
}

void AsyncExporter::workerLoop() {
    while (true) {
        Job job;
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_cv.wait(lock, [&]{ return !m_jobs.empty() || !m_running; });
            if (!m_running && m_jobs.empty()) break;
            job = std::move(m_jobs.front());
            m_jobs.pop();
        }

        bool ok = stbi_write_png(job.filename.c_str(), job.width, job.height, 4,
                                  job.pixels.data(), job.width * 4) != 0;
        if (ok) printf("Exported: %s\n", job.filename.c_str());
        else fprintf(stderr, "Failed to write PNG: %s\n", job.filename.c_str());
        m_pending--;
    }
}
