#include "gpu_context.h"
#include "render_pass.h"
#include "post_effects.h"
#include "ui.h"
#include "export.h"
#include "algorithms/game_of_life.h"
#include "algorithms/physarum.h"
#include "algorithms/boids.h"
#include <imgui.h>
#include <GLFW/glfw3.h>
#include <cstdio>
#include <memory>

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
    };
    constexpr int simCount = 3;
    const char* simNames[] = { "Game of Life", "Physarum", "Boids" };
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
            exportTextureToPNG(gpu.device, gpu.queue, sims[currentSim]->getOutputTexture(),
                               sims[currentSim]->params.width, sims[currentSim]->params.height, "export.png");
        }
    }

    sims[currentSim]->shutdown();
    postFx.shutdown();
    renderPass.shutdown();
    ui.shutdown();
    gpu.shutdown();
    return 0;
}
