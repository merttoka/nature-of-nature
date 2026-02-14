#include "ui.h"
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_wgpu.h>

void UI::init(GpuContext& ctx) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOther(ctx.window, true);

    ImGui_ImplWGPU_InitInfo initInfo = {};
    initInfo.Device = ctx.device;
    initInfo.RenderTargetFormat = ctx.surfaceFormat;
    initInfo.DepthStencilFormat = WGPUTextureFormat_Undefined;
    initInfo.NumFramesInFlight = 1;
    ImGui_ImplWGPU_Init(&initInfo);
}

void UI::beginFrame() {
    ImGui_ImplWGPU_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void UI::endFrame(WGPURenderPassEncoder renderPass) {
    ImGui::Render();
    ImGui_ImplWGPU_RenderDrawData(ImGui::GetDrawData(), renderPass);
}

void UI::shutdown() {
    ImGui_ImplWGPU_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}
