#include "gpu_context.h"
#include "render_pass.h"
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
    const char* simNames[] = { "Game of Life", "Physarum", "Boids", "Termites" };
    int currentSim = 1; // default to Physarum

    int simResolution = 512;
    const int resOptions[] = { 256, 512, 768, 1024, 1536, 2048 };
    constexpr int resCount = 6;
    const char* resLabels[] = { "256", "512", "768", "1024", "1536", "2048", "4096" };
    int resIndex = 4; // default 1536

    sims[currentSim]->init(gpu.device, gpu.queue, simResolution, simResolution);

    PostEffects postFx;
    postFx.init(gpu.device, gpu.queue, simResolution, simResolution);

    bool shouldExport = false;
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
                    view.offsetX += dx * view.zoom;
                    view.offsetY += dy * view.zoom;
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
        float texAspect = (float)sims[currentSim]->params.width / (float)sims[currentSim]->params.height;
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
            ImGui::Text("Sim: %s", simNames[currentSim]);
            ImGui::Text("Res: %ux%u", sims[currentSim]->params.width, sims[currentSim]->params.height);
            ImGui::Text("Zoom: %.1fx", view.zoom);
            ImGui::End();
        }

        // Settings window
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(280, 0), ImGuiCond_FirstUseEver);
        ImGui::Begin("Settings");

        int prevSim = currentSim;
        ImGui::Combo("Simulation", &currentSim, simNames, simCount);
        if (currentSim != prevSim) {
            sims[prevSim]->shutdown();
            sims[currentSim]->init(gpu.device, gpu.queue, simResolution, simResolution);
            postFx.resize(simResolution, simResolution);
        }

        int prevRes = resIndex;
        ImGui::Combo("Resolution", &resIndex, resLabels, resCount);
        if (resIndex != prevRes) {
            simResolution = resOptions[resIndex];
            sims[currentSim]->shutdown();
            sims[currentSim]->init(gpu.device, gpu.queue, simResolution, simResolution);
            postFx.resize(simResolution, simResolution);
        }

        if (ImGui::Button("Export PNG")) shouldExport = true;
        ImGui::End();

        // Simulation controls window
        ImGui::SetNextWindowPos(ImVec2(10, 130), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(280, 400), ImGuiCond_FirstUseEver);
        ImGui::Begin("Controls");
        sims[currentSim]->onGui();
        ImGui::End();

        // Post Effects window
        ImGui::SetNextWindowPos(ImVec2(300, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(250, 0), ImGuiCond_FirstUseEver);
        ImGui::Begin("Post Effects");
        postFx.onGui();
        ImGui::End();

        // Compute step (after UI so sim changes are applied)
        sims[currentSim]->step(encoder);

        // Post-processing
        postFx.apply(encoder, sims[currentSim]->getOutputView());

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

            // Create exports/ directory
            mkdir("exports", 0755);

            // Build filename: exports/{SimName}_{WxH}_{YYYYMMDD_HHMMSS}.png
            time_t t = time(nullptr);
            struct tm* tm_info = localtime(&t);
            char timestamp[32];
            strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", tm_info);

            // Sanitize sim name (replace spaces with _)
            std::string name(simNames[currentSim]);
            for (auto& c : name) if (c == ' ') c = '_';

            char filename[256];
            snprintf(filename, sizeof(filename), "exports/%s_%ux%u_%s.png",
                     name.c_str(), sims[currentSim]->params.width,
                     sims[currentSim]->params.height, timestamp);

            exportTextureToPNG(gpu.device, gpu.queue, postFx.getOutputTexture(),
                               sims[currentSim]->params.width, sims[currentSim]->params.height, filename);
        }
    }

    sims[currentSim]->shutdown();
    postFx.shutdown();
    renderPass.shutdown();
    ui.shutdown();
    gpu.shutdown();
    return 0;
}
