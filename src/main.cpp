#include "gpu_context.h"
#include "render_pass.h"
#include "ui.h"
#include "export.h"
#include "algorithms/game_of_life.h"
#include <imgui.h>
#include <cstdio>

int main() {
    GpuContext gpu;
    if (!gpu.init(1280, 720, "nature of nature")) {
        fprintf(stderr, "Failed to initialize GPU context\n");
        return 1;
    }

    UI ui;
    ui.init(gpu);

    RenderPass renderPass;
    renderPass.init(gpu.device, gpu.surfaceFormat);

    // Simulation
    GameOfLife sim;
    sim.init(gpu.device, gpu.queue, 512, 512);

    bool shouldExport = false;

    while (!glfwWindowShouldClose(gpu.window)) {
        glfwPollEvents();

        // Begin frame
        WGPUTextureView surfaceView = gpu.getNextSurfaceTextureView();
        if (!surfaceView) continue;

        WGPUCommandEncoderDescriptor encDesc = {};
        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(gpu.device, &encDesc);

        // Compute step
        sim.step(encoder);

        // Render sim output to screen
        WGPUBindGroup quadBG = renderPass.createBindGroup(gpu.device, sim.getOutputView());

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
        ImGui::SetNextWindowSize(ImVec2(280, 300), ImGuiCond_FirstUseEver);
        ImGui::Begin("Controls");
        sim.onGui();
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
            exportTextureToPNG(gpu.device, gpu.queue, sim.getOutputTexture(),
                               sim.params.width, sim.params.height, "export.png");
        }
    }

    sim.shutdown();
    renderPass.shutdown();
    ui.shutdown();
    gpu.shutdown();
    return 0;
}
