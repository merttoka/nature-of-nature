# Nature of Nature

Cross-platform symbolic generative art toolbox. Nature-inspired simple systems producing complex emergent forms.

Porting and extending algorithms from [Edge of Chaos](https://github.com/merttoka/edge-of-chaos-unity-compute) (Unity/Metal) into a standalone C++/WebGPU application with WGSL compute shaders.

## Algorithms

- **Game of Life** (cellular automata)
- **Physarum** (slime mold transport networks, 4 competitive agent types) 
- **Boids** (flocking with 4 competing types, GPU spatial hashing, trail competition) 
- **Termites** (biased random walk, probabilistic pheromone deposition, 4 competitive types) 

## Features

- **Compositing layers** — run multiple sims simultaneously, blend with additive/multiply/screen/normal modes + per-layer opacity
- Post-processing pipeline (bloom, brightness/contrast, saturation, vignette, colormap LUTs)
- **Colormap system** — built-in scientific colormaps (Viridis, Inferno, Magma, Plasma, Grayscale) applied via luminance remapping
- Zoom/pan viewport (scroll wheel, click-drag, WASD/ZX keys, nearest-neighbor sampling)
- Granular parameter randomization (Movement / Deposition / Colors buttons)
- Color controls always per-type, independent of Link All Types
- Preset save/load system (`presets/` directory)
- **PNG export** with metadata filenames, post-effects, and **1x–4x hi-res upscale**
- **PNG sequence recording** with configurable frame interval for video creation

## Stack

- C++17, WebGPU (wgpu-native), WGSL compute shaders
- GLFW (windowing), Dear ImGui (controls), stb_image_write (PNG export)
- CMake + FetchContent for all dependencies

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
./build/nature-of-nature
```

Requires CMake 3.21+ and a C++17 compiler. All other dependencies are fetched automatically.

## Project Structure

```
src/
  main.cpp              # window, main loop
  gpu_context.h/cpp     # WebGPU device/surface/queue
  compute_pass.h/cpp    # ping-pong textures, compute pipeline helpers
  render_pass.h/cpp     # fullscreen quad renderer (nearest-neighbor, zoom/pan)
  compositor.h/cpp      # N-layer blending (additive, multiply, screen, normal)
  post_effects.h/cpp    # bloom, brightness, contrast, saturation, vignette
  preset.h              # save/load preset helpers
  simulation.h          # base simulation interface
  ui.h/cpp              # ImGui setup
  export.h/cpp          # GPU texture readback -> PNG
  algorithms/           # one file pair per algorithm
shaders/                # WGSL compute + render shaders
presets/                # saved parameter presets
```

## License

MIT
