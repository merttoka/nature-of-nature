#include "gpu_context.h"
#include <webgpu/wgpu.h>
#include "render_pass.h"
#include "compositor.h"
#include "post_effects.h"
#include "ui.h"
#include "export.h"
#include "algorithms/game_of_life.h"
#include "algorithms/physarum.h"
#include "algorithms/boids.h"
#include "algorithms/termites.h"
#include <imgui.h>
#include <GLFW/glfw3.h>
#include <cstdio>
#include <ctime>
#include <cstring>
#include <string>
#include <memory>
#include <vector>
#include <sys/stat.h>

struct ViewTransform {
    float offsetX = 0.0f, offsetY = 0.0f;
    float zoom = 1.0f;
};

struct AppUserData {
    GpuContext* gpu = nullptr;
    ViewTransform* view = nullptr;
};

int main() {
    GpuContext gpu;
    if (!gpu.init(1280, 1280, "nature of nature")) {
        fprintf(stderr, "Failed to initialize GPU context\n");
        return 1;
    }

    // Update to actual framebuffer size (handles Retina displays)
    gpu.updateSize();

    UI ui;
    ui.init(gpu);

    RenderPass renderPass;
    renderPass.init(gpu.device, gpu.surfaceFormat);

    // Simulations
    std::unique_ptr<Simulation> sims[] = {
        std::make_unique<GameOfLife>(),
        std::make_unique<PhysarumSim>(),
        std::make_unique<BoidsSim>(),
        std::make_unique<TermitesSim>(),
    };
    constexpr int simCount = 4;

    int rezX = 1536, rezY = 1536;

    // Init all sims upfront
    for (int i = 0; i < simCount; i++)
        sims[i]->init(gpu.device, gpu.queue, rezX, rezY);

    // Compositor
    Compositor compositor;
    compositor.init(gpu.device, gpu.queue, rezX, rezY);
    for (int i = 0; i < simCount; i++) {
        Layer l;
        l.sim = sims[i].get();
        l.enabled = (i == 1); // default: only Physarum
        l.opacity = 1.0f;
        l.blendMode = BlendMode::Additive;
        compositor.layers.push_back(l);
    }

    PostEffects postFx;
    postFx.init(gpu.device, gpu.queue, rezX, rezY);

    // Upscale pipeline for hi-res export
    WGPUBindGroupLayout upscaleBGL = nullptr;
    WGPUPipelineLayout upscalePL = nullptr;
    WGPUComputePipeline upscalePipeline = nullptr;
    WGPUShaderModule upscaleSM = nullptr;
    {
        std::string code = loadShaderFile("shaders/upscale.wgsl");
        WGPUShaderModuleWGSLDescriptor wgslDesc = {};
        wgslDesc.chain.sType = WGPUSType_ShaderModuleWGSLDescriptor;
        wgslDesc.code = code.c_str();
        WGPUShaderModuleDescriptor smDesc = {};
        smDesc.nextInChain = &wgslDesc.chain;
        upscaleSM = wgpuDeviceCreateShaderModule(gpu.device, &smDesc);

        WGPUBindGroupLayoutEntry entries[4] = {};
        entries[0].binding = 0;
        entries[0].visibility = WGPUShaderStage_Compute;
        entries[0].buffer.type = WGPUBufferBindingType_Uniform;
        entries[0].buffer.minBindingSize = 16;
        entries[1].binding = 1;
        entries[1].visibility = WGPUShaderStage_Compute;
        entries[1].texture.sampleType = WGPUTextureSampleType_Float;
        entries[1].texture.viewDimension = WGPUTextureViewDimension_2D;
        entries[2].binding = 2;
        entries[2].visibility = WGPUShaderStage_Compute;
        entries[2].sampler.type = WGPUSamplerBindingType_Filtering;
        entries[3].binding = 3;
        entries[3].visibility = WGPUShaderStage_Compute;
        entries[3].storageTexture.access = WGPUStorageTextureAccess_WriteOnly;
        entries[3].storageTexture.format = WGPUTextureFormat_RGBA8Unorm;
        entries[3].storageTexture.viewDimension = WGPUTextureViewDimension_2D;

        WGPUBindGroupLayoutDescriptor bglDesc = {};
        bglDesc.entryCount = 4;
        bglDesc.entries = entries;
        upscaleBGL = wgpuDeviceCreateBindGroupLayout(gpu.device, &bglDesc);

        WGPUPipelineLayoutDescriptor plDesc = {};
        plDesc.bindGroupLayoutCount = 1;
        plDesc.bindGroupLayouts = &upscaleBGL;
        upscalePL = wgpuDeviceCreatePipelineLayout(gpu.device, &plDesc);

        WGPUComputePipelineDescriptor cpDesc = {};
        cpDesc.layout = upscalePL;
        cpDesc.compute.module = upscaleSM;
        cpDesc.compute.entryPoint = "main";
        upscalePipeline = wgpuDeviceCreateComputePipeline(gpu.device, &cpDesc);
    }

    // Upscale sampler
    WGPUSamplerDescriptor upSampDesc = {};
    upSampDesc.magFilter = WGPUFilterMode_Linear;
    upSampDesc.minFilter = WGPUFilterMode_Linear;
    upSampDesc.addressModeU = WGPUAddressMode_ClampToEdge;
    upSampDesc.addressModeV = WGPUAddressMode_ClampToEdge;
    upSampDesc.maxAnisotropy = 1;
    WGPUSampler upscaleSampler = wgpuDeviceCreateSampler(gpu.device, &upSampDesc);

    // Upscale uniform buffer
    WGPUBufferDescriptor upUniDesc = {};
    upUniDesc.size = 16;
    upUniDesc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    WGPUBuffer upscaleUniform = wgpuDeviceCreateBuffer(gpu.device, &upUniDesc);

    bool shouldExport = false;
    bool recording = false;
    int seqFrame = 0;
    int seqInterval = 1;
    int exportScale = 1;
    std::string seqDir; // subdirectory for current sequence
    AsyncExporter asyncExporter;
    double lastTime = glfwGetTime();
    float fps = 0.0f;
    int frameCount = 0;

    ViewTransform view;
    bool dragging = false;
    double lastMouseX = 0.0, lastMouseY = 0.0;
    double lastClickTime = 0.0;

    // Wrap user pointer so resize callback (in gpu_context) and scroll both work
    AppUserData appData;
    appData.gpu = &gpu;
    appData.view = &view;
    glfwSetWindowUserPointer(gpu.window, &appData);

    // Re-register framebuffer resize callback with new user pointer type
    glfwSetFramebufferSizeCallback(gpu.window, [](GLFWwindow* win, int w, int h) {
        auto* app = (AppUserData*)glfwGetWindowUserPointer(win);
        if (app && app->gpu && w > 0 && h > 0) {
            app->gpu->width = (uint32_t)w;
            app->gpu->height = (uint32_t)h;
            app->gpu->configureSurface();
        }
    });

    glfwSetScrollCallback(gpu.window, [](GLFWwindow* w, double, double yoff) {
        if (ImGui::GetIO().WantCaptureMouse) return;
        auto* app = (AppUserData*)glfwGetWindowUserPointer(w);
        float factor = yoff > 0 ? 1.1f : (1.0f / 1.1f);
        app->view->zoom *= factor;
        if (app->view->zoom < 0.1f) app->view->zoom = 0.1f;
        if (app->view->zoom > 100.0f) app->view->zoom = 100.0f;
    });

    while (!glfwWindowShouldClose(gpu.window)) {
        glfwPollEvents();

        // FPS counter
        frameCount++;
        double now = glfwGetTime();
        if (now - lastTime >= 0.5) {
            fps = (float)(frameCount / (now - lastTime));
            frameCount = 0;
            lastTime = now;
        }

        // --- Input: pan with mouse drag ---
        if (!ImGui::GetIO().WantCaptureMouse) {
            double mx, my;
            glfwGetCursorPos(gpu.window, &mx, &my);
            int winW, winH;
            glfwGetWindowSize(gpu.window, &winW, &winH);

            if (glfwGetMouseButton(gpu.window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
                if (!dragging) {
                    dragging = true;
                    // Double-click detection
                    if (now - lastClickTime < 0.3) {
                        view.offsetX = 0.0f;
                        view.offsetY = 0.0f;
                        view.zoom = 1.0f;
                    }
                    lastClickTime = now;
                } else {
                    float dx = (float)(mx - lastMouseX) / (float)winW;
                    float dy = (float)(my - lastMouseY) / (float)winH;
                    view.offsetX += dx / view.zoom;
                    view.offsetY += dy / view.zoom;
                }
                lastMouseX = mx;
                lastMouseY = my;
            } else {
                dragging = false;
                lastMouseX = mx;
                lastMouseY = my;
            }
        } else {
            dragging = false;
        }

        // --- Input: keyboard pan/zoom ---
        if (!ImGui::GetIO().WantCaptureKeyboard) {
            float panSpeed = 0.01f / view.zoom;
            if (glfwGetKey(gpu.window, GLFW_KEY_W) == GLFW_PRESS) view.offsetY += panSpeed;
            if (glfwGetKey(gpu.window, GLFW_KEY_S) == GLFW_PRESS) view.offsetY -= panSpeed;
            if (glfwGetKey(gpu.window, GLFW_KEY_A) == GLFW_PRESS) view.offsetX += panSpeed;
            if (glfwGetKey(gpu.window, GLFW_KEY_D) == GLFW_PRESS) view.offsetX -= panSpeed;
            if (glfwGetKey(gpu.window, GLFW_KEY_Z) == GLFW_PRESS) {
                view.zoom *= 1.02f;
                if (view.zoom > 100.0f) view.zoom = 100.0f;
            }
            if (glfwGetKey(gpu.window, GLFW_KEY_X) == GLFW_PRESS) {
                view.zoom /= 1.02f;
                if (view.zoom < 0.1f) view.zoom = 0.1f;
            }
            if (glfwGetKey(gpu.window, GLFW_KEY_0) == GLFW_PRESS) {
                view.offsetX = 0.0f;
                view.offsetY = 0.0f;
                view.zoom = 1.0f;
            }
        }

        // Upload view transform with aspect ratio correction
        int winW, winH;
        glfwGetFramebufferSize(gpu.window, &winW, &winH);
        float windowAspect = (winH > 0) ? (float)winW / (float)winH : 1.0f;
        float texAspect = (float)rezX / (float)rezY;
        float aspectRatio = windowAspect / texAspect;
        renderPass.setTransform(gpu.queue, view.offsetX, view.offsetY, view.zoom, aspectRatio);

        // Begin frame
        WGPUTextureView surfaceView = gpu.getNextSurfaceTextureView();
        if (!surfaceView) continue;

        WGPUCommandEncoderDescriptor encDesc = {};
        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(gpu.device, &encDesc);

        // ImGui — process UI first so sim/resolution changes happen before compute
        ui.beginFrame();

        // Stats overlay (hold Tab)
        if (glfwGetKey(gpu.window, GLFW_KEY_TAB) == GLFW_PRESS) {
            ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x - 160, 10));
            ImGui::SetNextWindowBgAlpha(0.6f);
            ImGui::Begin("##stats", nullptr,
                ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs |
                ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings);
            ImGui::Text("FPS: %.0f", fps);
            ImGui::Text("Res: %ux%u", (uint32_t)rezX, (uint32_t)rezY);
            ImGui::Text("Zoom: %.1fx", view.zoom);
            ImGui::End();
        }

        // Settings window
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(280, 0), ImGuiCond_FirstUseEver);
        ImGui::Begin("Settings");

        int prevRezX = rezX, prevRezY = rezY;
        ImGui::DragInt("RezX", &rezX, 8.0f, 64, 4096);
        ImGui::DragInt("RezY", &rezY, 8.0f, 64, 4096);
        if (rezX != prevRezX || rezY != prevRezY) {
            for (int i = 0; i < simCount; i++) {
                sims[i]->shutdown();
                sims[i]->init(gpu.device, gpu.queue, rezX, rezY);
            }
            compositor.resize(rezX, rezY);
            postFx.resize(rezX, rezY);
        }

        if (ImGui::Button("Export PNG")) shouldExport = true;
        ImGui::SameLine();
        ImGui::DragInt("Scale", &exportScale, 0.1f, 1, 4);
        ImGui::Separator();
        if (recording) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.1f, 0.1f, 1.0f));
            if (ImGui::Button("Stop Recording")) {
                recording = false;
                seqFrame = 0;
                asyncExporter.stop();
            }
            ImGui::PopStyleColor();
            ImGui::SameLine();
            ImGui::Text("Frame %d", seqFrame);
            int pend = asyncExporter.pending();
            if (pend > 0) { ImGui::SameLine(); ImGui::Text("(%d queued)", pend); }
        } else {
            if (ImGui::Button("Record Sequence")) {
                mkdir("exports", 0755);
                time_t t = time(nullptr);
                struct tm* tm_info = localtime(&t);
                char ts[32];
                strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", tm_info);
                seqDir = std::string("exports/seq_") + ts;
                mkdir(seqDir.c_str(), 0755);
                recording = true;
                seqFrame = 0;
                asyncExporter.start();
            }
        }
        ImGui::DragInt("Interval", &seqInterval, 0.1f, 1, 60);
        ImGui::TextDisabled("ffmpeg -framerate 30 -i exports/seq_%%06d.png -c:v libx264 out.mp4");
        ImGui::End();

        // Layers window
        ImGui::SetNextWindowPos(ImVec2(10, 100), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(280, 0), ImGuiCond_FirstUseEver);
        ImGui::Begin("Layers");
        compositor.onGui();
        ImGui::End();

        // Simulation controls window
        ImGui::SetNextWindowPos(ImVec2(10, 200), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(280, 400), ImGuiCond_FirstUseEver);
        ImGui::Begin("Controls");
        for (int i = 0; i < (int)compositor.layers.size(); i++) {
            auto& layer = compositor.layers[i];
            if (layer.enabled && layer.sim) {
                ImGui::PushID(i);
                if (ImGui::CollapsingHeader(layer.sim->name(), ImGuiTreeNodeFlags_DefaultOpen)) {
                    layer.sim->onGui();
                }
                ImGui::PopID();
            }
        }
        ImGui::End();

        // Post Effects window
        ImGui::SetNextWindowPos(ImVec2(300, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(250, 0), ImGuiCond_FirstUseEver);
        ImGui::Begin("Post Effects");
        postFx.onGui();
        ImGui::End();

        // Compute step: step all enabled sims, then composite
        for (auto& layer : compositor.layers) {
            if (layer.enabled && layer.sim) {
                layer.sim->step(encoder);
            }
        }
        compositor.composite(encoder);

        // Post-processing
        postFx.apply(encoder, compositor.getOutputView());

        // Render post-processed output to screen
        WGPUBindGroup quadBG = renderPass.createBindGroup(gpu.device, postFx.getOutputView());

        // Render pass for fullscreen quad + ImGui
        WGPURenderPassColorAttachment colorAtt = {};
        colorAtt.view = surfaceView;
        colorAtt.loadOp = WGPULoadOp_Clear;
        colorAtt.storeOp = WGPUStoreOp_Store;
        colorAtt.clearValue = { 0.0, 0.0, 0.0, 1.0 };

        WGPURenderPassDescriptor rpDesc = {};
        rpDesc.colorAttachmentCount = 1;
        rpDesc.colorAttachments = &colorAtt;

        WGPURenderPassEncoder rpass = wgpuCommandEncoderBeginRenderPass(encoder, &rpDesc);

        // Draw fullscreen quad
        wgpuRenderPassEncoderSetPipeline(rpass, renderPass.pipeline);
        wgpuRenderPassEncoderSetBindGroup(rpass, 0, quadBG, 0, nullptr);
        wgpuRenderPassEncoderDraw(rpass, 6, 1, 0, 0);

        ui.endFrame(rpass);

        wgpuRenderPassEncoderEnd(rpass);
        wgpuRenderPassEncoderRelease(rpass);

        // Submit
        WGPUCommandBufferDescriptor cbDesc = {};
        WGPUCommandBuffer cmdBuf = wgpuCommandEncoderFinish(encoder, &cbDesc);
        wgpuQueueSubmit(gpu.queue, 1, &cmdBuf);
        wgpuCommandBufferRelease(cmdBuf);
        wgpuCommandEncoderRelease(encoder);

        wgpuBindGroupRelease(quadBG);

        gpu.present();
        wgpuTextureViewRelease(surfaceView);

        // Export after frame
        if (shouldExport) {
            shouldExport = false;
            mkdir("exports", 0755);

            time_t t = time(nullptr);
            struct tm* tm_info = localtime(&t);
            char timestamp[32];
            strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", tm_info);

            std::string name = "Composite";
            int enabledCount = 0;
            for (auto& l : compositor.layers) {
                if (l.enabled && l.sim) {
                    if (enabledCount == 0) name = l.sim->name();
                    else name = "Composite";
                    enabledCount++;
                }
            }
            for (auto& c : name) if (c == ' ') c = '_';

            uint32_t outW = rezX * exportScale, outH = rezY * exportScale;
            char filename[256];
            snprintf(filename, sizeof(filename), "exports/%s_%ux%u_%s.png",
                     name.c_str(), outW, outH, timestamp);

            if (exportScale == 1) {
                exportTextureToPNG(gpu.device, gpu.queue, postFx.getOutputTexture(),
                                   rezX, rezY, filename);
            } else {
                // Create hi-res temp texture
                WGPUTextureDescriptor hiDesc = {};
                hiDesc.size = { outW, outH, 1 };
                hiDesc.format = WGPUTextureFormat_RGBA8Unorm;
                hiDesc.usage = WGPUTextureUsage_StorageBinding | WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopySrc;
                hiDesc.mipLevelCount = 1;
                hiDesc.sampleCount = 1;
                hiDesc.dimension = WGPUTextureDimension_2D;
                WGPUTexture hiTex = wgpuDeviceCreateTexture(gpu.device, &hiDesc);
                WGPUTextureView hiView = wgpuTextureCreateView(hiTex, nullptr);

                // Upload upscale params
                uint32_t upParams[4] = { (uint32_t)rezX, (uint32_t)rezY, outW, outH };
                wgpuQueueWriteBuffer(gpu.queue, upscaleUniform, 0, upParams, 16);

                // Build bind group
                WGPUBindGroupEntry bgEntries[4] = {};
                bgEntries[0].binding = 0;
                bgEntries[0].buffer = upscaleUniform;
                bgEntries[0].size = 16;
                bgEntries[1].binding = 1;
                bgEntries[1].textureView = postFx.getOutputView();
                bgEntries[2].binding = 2;
                bgEntries[2].sampler = upscaleSampler;
                bgEntries[3].binding = 3;
                bgEntries[3].textureView = hiView;

                WGPUBindGroupDescriptor bgDesc = {};
                bgDesc.layout = upscaleBGL;
                bgDesc.entryCount = 4;
                bgDesc.entries = bgEntries;
                WGPUBindGroup bg = wgpuDeviceCreateBindGroup(gpu.device, &bgDesc);

                // Dispatch upscale
                WGPUCommandEncoderDescriptor eDesc = {};
                WGPUCommandEncoder enc2 = wgpuDeviceCreateCommandEncoder(gpu.device, &eDesc);
                WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc2, nullptr);
                wgpuComputePassEncoderSetPipeline(pass, upscalePipeline);
                wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
                wgpuComputePassEncoderDispatchWorkgroups(pass, (outW + 7) / 8, (outH + 7) / 8, 1);
                wgpuComputePassEncoderEnd(pass);
                wgpuComputePassEncoderRelease(pass);

                WGPUCommandBufferDescriptor cb2Desc = {};
                WGPUCommandBuffer cmd2 = wgpuCommandEncoderFinish(enc2, &cb2Desc);
                wgpuQueueSubmit(gpu.queue, 1, &cmd2);
                wgpuCommandBufferRelease(cmd2);
                wgpuCommandEncoderRelease(enc2);

                exportTextureToPNG(gpu.device, gpu.queue, hiTex, outW, outH, filename);

                wgpuBindGroupRelease(bg);
                wgpuTextureViewRelease(hiView);
                wgpuTextureDestroy(hiTex);
                wgpuTextureRelease(hiTex);
            }
        }

        // Sequence recording — GPU readback here, PNG encode on worker thread
        if (recording) {
            if (seqFrame % seqInterval == 0) {
                char seqFilename[256];
                snprintf(seqFilename, sizeof(seqFilename), "%s/%06d.png", seqDir.c_str(), seqFrame);

                WGPUTexture tex = postFx.getOutputTexture();
                uint32_t w = rezX, h = rezY;
                uint32_t bytesPerRow = ((w * 4 + 255) / 256) * 256;
                uint64_t bufferSize = (uint64_t)bytesPerRow * h;

                WGPUBufferDescriptor bufDesc = {};
                bufDesc.size = bufferSize;
                bufDesc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
                WGPUBuffer readbackBuf = wgpuDeviceCreateBuffer(gpu.device, &bufDesc);

                WGPUCommandEncoderDescriptor eDesc = {};
                WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(gpu.device, &eDesc);
                WGPUImageCopyTexture src = {};
                src.texture = tex;
                WGPUImageCopyBuffer dst = {};
                dst.buffer = readbackBuf;
                dst.layout.bytesPerRow = bytesPerRow;
                dst.layout.rowsPerImage = h;
                WGPUExtent3D extent = { w, h, 1 };
                wgpuCommandEncoderCopyTextureToBuffer(enc, &src, &dst, &extent);
                WGPUCommandBufferDescriptor cbDesc2 = {};
                WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, &cbDesc2);
                wgpuQueueSubmit(gpu.queue, 1, &cmd);
                wgpuCommandBufferRelease(cmd);
                wgpuCommandEncoderRelease(enc);

                struct MapData { bool done = false; WGPUBufferMapAsyncStatus status; };
                MapData mapData;
                wgpuBufferMapAsync(readbackBuf, WGPUMapMode_Read, 0, bufferSize,
                    [](WGPUBufferMapAsyncStatus status, void* ud) {
                        auto* d = (MapData*)ud;
                        d->status = status;
                        d->done = true;
                    }, &mapData);
                while (!mapData.done)
                    wgpuDevicePoll(gpu.device, true, nullptr);

                if (mapData.status == WGPUBufferMapAsyncStatus_Success) {
                    const uint8_t* mapped = (const uint8_t*)wgpuBufferGetConstMappedRange(readbackBuf, 0, bufferSize);
                    std::vector<uint8_t> pixels(w * h * 4);
                    for (uint32_t y = 0; y < h; y++)
                        memcpy(&pixels[y * w * 4], &mapped[y * bytesPerRow], w * 4);
                    wgpuBufferUnmap(readbackBuf);
                    asyncExporter.enqueue(std::move(pixels), w, h, seqFilename);
                }
                wgpuBufferRelease(readbackBuf);
            }
            seqFrame++;
        }
    }

    asyncExporter.stop();
    for (int i = 0; i < simCount; i++)
        sims[i]->shutdown();
    compositor.shutdown();
    postFx.shutdown();
    wgpuComputePipelineRelease(upscalePipeline);
    wgpuPipelineLayoutRelease(upscalePL);
    wgpuBindGroupLayoutRelease(upscaleBGL);
    wgpuShaderModuleRelease(upscaleSM);
    wgpuSamplerRelease(upscaleSampler);
    wgpuBufferDestroy(upscaleUniform);
    wgpuBufferRelease(upscaleUniform);
    renderPass.shutdown();
    ui.shutdown();
    gpu.shutdown();
    return 0;
}
