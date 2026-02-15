# nature-of-nature

Cross-platform symbolic generative art toolbox. Nature-inspired simple systems producing complex emergent forms.

Porting and extending algorithms from [Edge of Chaos](https://github.com/merttoka/edge-of-chaos-unity-compute) (Unity/Metal) into a standalone C++/WebGPU application with WGSL compute shaders.

## Algorithms

- **Game of Life** (cellular automata) — ✅ implemented
- **Physarum** (slime mold transport networks, 4 competitive agent types) — ✅ implemented
- **Boids** (flocking with cross-simulation food-seeking)
- **Termites** (biased random walk, pheromone sensing)
- **Cyclic Cellular Automata** (N-state threshold systems)

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
  render_pass.h/cpp     # fullscreen quad renderer
  simulation.h          # base simulation interface
  ui.h/cpp              # ImGui setup
  export.h/cpp          # GPU texture readback -> PNG
  algorithms/           # one file pair per algorithm
shaders/                # WGSL compute + render shaders
```

## License

MIT
