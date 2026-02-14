#include "render_pass.h"
#include "compute_pass.h" // for loadShaderFile
#include <cstdio>

void RenderPass::init(WGPUDevice device, WGPUTextureFormat surfaceFormat) {
    // Bind group layout: sampler + texture
    WGPUBindGroupLayoutEntry entries[2] = {};
    entries[0].binding = 0;
    entries[0].visibility = WGPUShaderStage_Fragment;
    entries[0].sampler.type = WGPUSamplerBindingType_Filtering;

    entries[1].binding = 1;
    entries[1].visibility = WGPUShaderStage_Fragment;
    entries[1].texture.sampleType = WGPUTextureSampleType_Float;
    entries[1].texture.viewDimension = WGPUTextureViewDimension_2D;

    WGPUBindGroupLayoutDescriptor bglDesc = {};
    bglDesc.entryCount = 2;
    bglDesc.entries = entries;
    bindGroupLayout = wgpuDeviceCreateBindGroupLayout(device, &bglDesc);

    // Sampler
    WGPUSamplerDescriptor samplerDesc = {};
    samplerDesc.magFilter = WGPUFilterMode_Nearest;
    samplerDesc.minFilter = WGPUFilterMode_Nearest;
    samplerDesc.addressModeU = WGPUAddressMode_ClampToEdge;
    samplerDesc.addressModeV = WGPUAddressMode_ClampToEdge;
    samplerDesc.addressModeW = WGPUAddressMode_ClampToEdge;
    samplerDesc.mipmapFilter = WGPUMipmapFilterMode_Nearest;
    samplerDesc.maxAnisotropy = 1;
    sampler = wgpuDeviceCreateSampler(device, &samplerDesc);

    // Shader
    std::string code = loadShaderFile("shaders/fullscreen_quad.wgsl");
    WGPUShaderModuleWGSLDescriptor wgslDesc = {};
    wgslDesc.chain.sType = WGPUSType_ShaderModuleWGSLDescriptor;
    wgslDesc.code = code.c_str();

    WGPUShaderModuleDescriptor smDesc = {};
    smDesc.nextInChain = &wgslDesc.chain;
    WGPUShaderModule module = wgpuDeviceCreateShaderModule(device, &smDesc);

    // Pipeline layout
    WGPUPipelineLayoutDescriptor plDesc = {};
    plDesc.bindGroupLayoutCount = 1;
    plDesc.bindGroupLayouts = &bindGroupLayout;
    WGPUPipelineLayout layout = wgpuDeviceCreatePipelineLayout(device, &plDesc);

    // Render pipeline
    WGPUColorTargetState colorTarget = {};
    colorTarget.format = surfaceFormat;
    colorTarget.writeMask = WGPUColorWriteMask_All;

    WGPUFragmentState fragState = {};
    fragState.module = module;
    fragState.entryPoint = "fs_main";
    fragState.targetCount = 1;
    fragState.targets = &colorTarget;

    WGPURenderPipelineDescriptor rpDesc = {};
    rpDesc.layout = layout;
    rpDesc.vertex.module = module;
    rpDesc.vertex.entryPoint = "vs_main";
    rpDesc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    rpDesc.multisample.count = 1;
    rpDesc.multisample.mask = 0xFFFFFFFF;
    rpDesc.fragment = &fragState;

    pipeline = wgpuDeviceCreateRenderPipeline(device, &rpDesc);

    wgpuPipelineLayoutRelease(layout);
    wgpuShaderModuleRelease(module);
}

WGPUBindGroup RenderPass::createBindGroup(WGPUDevice device, WGPUTextureView textureView) {
    WGPUBindGroupEntry entries[2] = {};
    entries[0].binding = 0;
    entries[0].sampler = sampler;
    entries[1].binding = 1;
    entries[1].textureView = textureView;

    WGPUBindGroupDescriptor desc = {};
    desc.layout = bindGroupLayout;
    desc.entryCount = 2;
    desc.entries = entries;
    return wgpuDeviceCreateBindGroup(device, &desc);
}

void RenderPass::draw(WGPUCommandEncoder encoder, WGPUTextureView targetView,
                      WGPUBindGroup bindGroup) {
    WGPURenderPassColorAttachment colorAtt = {};
    colorAtt.view = targetView;
    colorAtt.loadOp = WGPULoadOp_Clear;
    colorAtt.storeOp = WGPUStoreOp_Store;
    colorAtt.clearValue = { 0.0, 0.0, 0.0, 1.0 };

    WGPURenderPassDescriptor rpDesc = {};
    rpDesc.colorAttachmentCount = 1;
    rpDesc.colorAttachments = &colorAtt;

    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &rpDesc);
    wgpuRenderPassEncoderSetPipeline(pass, pipeline);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, bindGroup, 0, nullptr);
    wgpuRenderPassEncoderDraw(pass, 6, 1, 0, 0); // fullscreen quad (2 triangles)
    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);
}

void RenderPass::shutdown() {
    if (sampler) wgpuSamplerRelease(sampler);
    if (bindGroupLayout) wgpuBindGroupLayoutRelease(bindGroupLayout);
    if (pipeline) wgpuRenderPipelineRelease(pipeline);
}
