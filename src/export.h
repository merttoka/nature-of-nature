#pragma once
#include <webgpu/webgpu.h>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>

// Synchronous single-frame export
bool exportTextureToPNG(WGPUDevice device, WGPUQueue queue,
                        WGPUTexture texture, uint32_t width, uint32_t height,
                        const std::string& filename);

// Async exporter for sequence recording — PNG encoding happens on a worker thread
class AsyncExporter {
public:
    void start();
    void stop(); // blocks until queue is drained

    // Enqueue pixel data for PNG encoding on worker thread
    void enqueue(std::vector<uint8_t>&& pixels, uint32_t w, uint32_t h,
                 const std::string& filename);

    int pending() const { return m_pending.load(); }

private:
    void workerLoop();

    struct Job {
        std::vector<uint8_t> pixels;
        uint32_t width, height;
        std::string filename;
    };

    std::thread m_thread;
    std::mutex m_mutex;
    std::condition_variable m_cv;
    std::queue<Job> m_jobs;
    std::atomic<bool> m_running{false};
    std::atomic<int> m_pending{0};
};
