# CLAUDE.md

## Build
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
```
Known issue: wgpu-native ships `macos-aarch64` dir but CMake expects `macos-arm64`. Symlink created in build step. If clean build fails on macOS ARM, run: `ln -s macos-aarch64 build/_deps/webgpu-src/bin/macos-arm64`

## Architecture
- `Simulation` base class in `simulation.h` — all algos inherit from it
- `PingPongTextures` in `compute_pass.h` — shared ping-pong infra for grid-based sims
- Shaders in `shaders/` dir, copied to build dir post-build
- WebGPU API version: wgpu-native v0.19.4.1 (old spec, NOT the new WGPUStringView API)
- glfw3webgpu pinned to v1.0.1 (must match wgpu v0.19 API)
- ImGui WebGPU backend needs `IMGUI_IMPL_WEBGPU_BACKEND_WGPU=1`

## Porting references
- Physarum primary: Unity VISAP (`~/Desktop/Making/Graphics/UNITY_EoC_GPU/Assets/Workspace/9.1 VISAP/`)
- Boids/Common: Metal port (`~/Desktop/Making/Graphics/METAL_EoC_GPU/EdgeOfChaos/`)
- Termites: Processing (`~/Desktop/Making/Graphics/PDE_Nefeli_Termites/`)

## Conventions
- HLSL→WGSL porting: `float4`→`vec4f`, `SamplerState`+`SampleLevel`→`textureSample`, `RWTexture2D`→`texture_storage_2d`
- Agent buffers: storage buffers with `read_write` access
- Thread groups: 8x8 for texture ops, 256 for agent ops
