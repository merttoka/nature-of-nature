#include "gpu_context.h"
#include "render_pass.h"
#include "ui.h"
#include "export.h"
#include "algorithms/game_of_life.h"
#include "algorithms/physarum.h"
#include <imgui.h>
#include <cstdio>
#include <memory>

int main() {
    GpuContext gpu;
    if (!gpu.init(1280, 720, "nature of nature")) {
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
    };
    constexpr int simCount = 2;
    const char* simNames[] = { "Game of Life", "Physarum" };
    int currentSim = 1; // default to Physarum

    sims[currentSim]->init(gpu.device, gpu.queue, 512, 512);

    bool shouldExport = false;

    while (!glfwWindowShouldClose(gpu.window)) {
        glfwPollEvents();

        // Begin frame
        WGPUTextureView surfaceView = gpu.getNextSurfaceTextureView();
        if (!surfaceView) continue;

        WGPUCommandEncoderDescriptor encDesc = {};
        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(gpu.device, &encDesc);

        // Compute step
        sims[currentSim]->step(encoder);

        // Render sim output to screen
        WGPUBindGroup quadBG = renderPass.createBindGroup(gpu.device, sims[currentSim]->getOutputView());

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

        // ImGui
        ui.beginFrame();

        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(280, 400), ImGuiCond_FirstUseEver);
        ImGui::Begin("Controls");

        // Sim switcher
        int prevSim = currentSim;
        ImGui::Combo("Simulation", &currentSim, simNames, simCount);
        if (currentSim != prevSim) {
            sims[prevSim]->shutdown();
            sims[currentSim]->init(gpu.device, gpu.queue, 512, 512);
        }

        ImGui::Separator();
        sims[currentSim]->onGui();

        ImGui::Separator();
        if (ImGui::Button("Export PNG")) shouldExport = true;
        ImGui::End();

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
    renderPass.shutdown();
    ui.shutdown();
    gpu.shutdown();
    return 0;
}
