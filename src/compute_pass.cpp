#include "compute_pass.h"
#include <fstream>
#include <sstream>
#include <cstdio>

void PingPongTextures::init(WGPUDevice device, uint32_t w, uint32_t h, WGPUTextureFormat format) {
    width = w;
    height = h;
    current = 0;

    WGPUTextureDescriptor desc = {};
    desc.size = { w, h, 1 };
    desc.format = format;
    desc.usage = WGPUTextureUsage_StorageBinding | WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopySrc | WGPUTextureUsage_CopyDst;
    desc.dimension = WGPUTextureDimension_2D;
    desc.mipLevelCount = 1;
    desc.sampleCount = 1;

    desc.label = "pingpong_A";
    texA = wgpuDeviceCreateTexture(device, &desc);
    desc.label = "pingpong_B";
    texB = wgpuDeviceCreateTexture(device, &desc);

    WGPUTextureViewDescriptor viewDesc = {};
    viewDesc.format = format;
    viewDesc.dimension = WGPUTextureViewDimension_2D;
    viewDesc.mipLevelCount = 1;
    viewDesc.arrayLayerCount = 1;

    viewA = wgpuTextureCreateView(texA, &viewDesc);
    viewB = wgpuTextureCreateView(texB, &viewDesc);
}

void PingPongTextures::swap() { current = 1 - current; }
WGPUTextureView PingPongTextures::readView() const { return current == 0 ? viewA : viewB; }
WGPUTextureView PingPongTextures::writeView() const { return current == 0 ? viewB : viewA; }

void PingPongTextures::destroy() {
    if (viewA) wgpuTextureViewRelease(viewA);
    if (viewB) wgpuTextureViewRelease(viewB);
    if (texA) wgpuTextureDestroy(texA);
    if (texB) wgpuTextureDestroy(texB);
    if (texA) wgpuTextureRelease(texA);
    if (texB) wgpuTextureRelease(texB);
    viewA = viewB = nullptr;
    texA = texB = nullptr;
}

std::string loadShaderFile(const char* path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "Failed to load shader: %s\n", path);
        return "";
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

WGPUComputePipeline createComputePipeline(
    WGPUDevice device, const char* shaderPath, const char* entryPoint, WGPUBindGroupLayout layout)
{
    std::string code = loadShaderFile(shaderPath);
    if (code.empty()) return nullptr;

    WGPUShaderModuleWGSLDescriptor wgslDesc = {};
    wgslDesc.chain.sType = WGPUSType_ShaderModuleWGSLDescriptor;
    wgslDesc.code = code.c_str();

    WGPUShaderModuleDescriptor smDesc = {};
    smDesc.nextInChain = &wgslDesc.chain;
    WGPUShaderModule module = wgpuDeviceCreateShaderModule(device, &smDesc);

    WGPUPipelineLayoutDescriptor plDesc = {};
    plDesc.bindGroupLayoutCount = 1;
    plDesc.bindGroupLayouts = &layout;
    WGPUPipelineLayout pipelineLayout = wgpuDeviceCreatePipelineLayout(device, &plDesc);

    WGPUComputePipelineDescriptor cpDesc = {};
    cpDesc.layout = pipelineLayout;
    cpDesc.compute.module = module;
    cpDesc.compute.entryPoint = entryPoint;
    WGPUComputePipeline pipeline = wgpuDeviceCreateComputePipeline(device, &cpDesc);

    wgpuPipelineLayoutRelease(pipelineLayout);
    wgpuShaderModuleRelease(module);
    return pipeline;
}

WGPUBindGroupLayout createPingPongBindGroupLayout(WGPUDevice device, bool withUniform) {
    WGPUBindGroupLayoutEntry entries[3] = {};

    // Binding 0: read texture (texture_2d<f32>)
    entries[0].binding = 0;
    entries[0].visibility = WGPUShaderStage_Compute;
    entries[0].texture.sampleType = WGPUTextureSampleType_Float;
    entries[0].texture.viewDimension = WGPUTextureViewDimension_2D;

    // Binding 1: write texture (storage texture, rgba8unorm, write)
    entries[1].binding = 1;
    entries[1].visibility = WGPUShaderStage_Compute;
    entries[1].storageTexture.access = WGPUStorageTextureAccess_WriteOnly;
    entries[1].storageTexture.format = WGPUTextureFormat_RGBA8Unorm;
    entries[1].storageTexture.viewDimension = WGPUTextureViewDimension_2D;

    uint32_t count = 2;

    // Binding 2: uniform buffer (optional)
    if (withUniform) {
        entries[2].binding = 2;
        entries[2].visibility = WGPUShaderStage_Compute;
        entries[2].buffer.type = WGPUBufferBindingType_Uniform;
        entries[2].buffer.minBindingSize = 0;
        count = 3;
    }

    WGPUBindGroupLayoutDescriptor desc = {};
    desc.entryCount = count;
    desc.entries = entries;
    return wgpuDeviceCreateBindGroupLayout(device, &desc);
}

WGPUBindGroup createPingPongBindGroup(
    WGPUDevice device, WGPUBindGroupLayout layout,
    WGPUTextureView readView, WGPUTextureView writeView,
    WGPUBuffer uniformBuffer, uint64_t uniformSize)
{
    WGPUBindGroupEntry entries[3] = {};

    entries[0].binding = 0;
    entries[0].textureView = readView;

    entries[1].binding = 1;
    entries[1].textureView = writeView;

    uint32_t count = 2;
    if (uniformBuffer) {
        entries[2].binding = 2;
        entries[2].buffer = uniformBuffer;
        entries[2].size = uniformSize;
        count = 3;
    }

    WGPUBindGroupDescriptor desc = {};
    desc.layout = layout;
    desc.entryCount = count;
    desc.entries = entries;
    return wgpuDeviceCreateBindGroup(device, &desc);
}
