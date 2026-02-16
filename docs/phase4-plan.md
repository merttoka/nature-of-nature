# Phase 4 — Termites + Compositing

## Overview

Phase 4 adds the fourth simulation algorithm (Termites) and a compositing system for blending multiple simulation outputs into a single frame.

---

## Part A: Termites Simulation

### Algorithm Description

Termites simulate nest-building insects that pick up and deposit material particles based on local pheromone concentrations. The original algorithm comes from a Processing implementation (referenced in CLAUDE.md). The GPU port follows the same architecture as Physarum and Boids: agent buffers + pheromone trail textures + ping-pong compute kernels.

**Core behavior per agent per step:**
1. Sense pheromone field at 3 sensor positions (left, center, right)
2. Turn toward highest pheromone concentration (biased random walk)
3. Move forward
4. If carrying material AND local pheromone > deposit threshold: drop material (probabilistic)
5. If not carrying AND pixel has material: pick up (probabilistic)
6. Deposit pheromone at current position

### Agent State

```wgsl
struct TermiteAgent {
    position: vec2f,     // xy position on grid
    direction: vec2f,    // normalized heading
    carrying: f32,       // 0.0 = empty, 1.0 = carrying material
    _pad: f32,           // alignment padding (24 bytes total)
};
```

**Note:** 24 bytes per agent (vs 16 for Physarum). Agent buffer size = `agentCount * 24`.

### Textures

| Texture | Format | Purpose |
|---------|--------|---------|
| `trailTextures` (ping-pong) | `rgba16float` | R = pheromone, G = material density, B = unused, A = unused |
| `outputTextures` (ping-pong) | `rgba8unorm` | Final rendered output for display |

Using 2 channels (pheromone + material) keeps the architecture consistent with existing sims. 4 termite types can share the same texture by encoding type-specific pheromone in RGBA channels if we want competition (like Physarum), but the base implementation uses a single shared pheromone field for simplicity, with type differentiation only in rendering color.

### Compute Kernels (6 total)

Following the Physarum pattern:

| # | Kernel | Workgroup | Description |
|---|--------|-----------|-------------|
| 1 | `reset_texture` | 8x8 | Clear trail + material textures to zero |
| 2 | `reset_agents` | 256 | Random scatter with random carrying state |
| 3 | `move_agents` | 256 | Sense pheromone, turn, move, pick/drop material |
| 4 | `write_trails` | 256 | Deposit pheromone at agent positions |
| 5 | `diffuse_texture` | 8x8 | 3x3 box blur + evaporation on pheromone channel |
| 6 | `render` | 8x8 | HSB color mapping from pheromone + material |

### Per-Type Parameters (4 types)

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `senseAngle` | 30.0 | 0.1–360 deg | Angle between center and side sensors |
| `senseDistance` | 12.0 | 0.1–100 | Distance to sensor positions |
| `turnAngle` | 45.0 | 0.1–360 deg | Max turn per step |
| `moveSpeed` | 0.5 | 0.01–5.0 | Pixels per step |
| `depositPheromone` | 0.02 | 0.001–0.5 | Pheromone deposited per step |
| `pickupProb` | 0.3 | 0.0–1.0 | Probability of picking up material |
| `dropProb` | 0.2 | 0.0–1.0 | Probability of dropping material |
| `pheromoneThreshold` | 0.1 | 0.0–1.0 | Min pheromone to trigger drop |
| `diffuseRate` | 0.92 | 0.0–1.0 | Pheromone evaporation/diffusion rate |
| `hue` | 0.0 | 0.0–1.0 | Display color hue |
| `saturation` | 0.6 | 0.0–1.0 | Display color saturation |

### GPU Uniform Struct

```cpp
struct GpuParams {
    uint32_t rezX, rezY, agentsCount, time;     // 16 bytes
    float senseAngles[4];                        // 16 bytes
    float senseDistances[4];                     // 16 bytes
    float turnAngles[4];                         // 16 bytes
    float moveSpeeds[4];                         // 16 bytes
    float depositPheromones[4];                  // 16 bytes
    float pickupProbs[4];                        // 16 bytes
    float dropProbs[4];                          // 16 bytes
    float pheromoneThresholds[4];                // 16 bytes
    float diffuseRates[4];                       // 16 bytes
    float hues[4];                               // 16 bytes
    float saturations[4];                        // 16 bytes
};
// Total: 192 bytes
```

### Bind Group Layout

Same 2-group pattern as Physarum:
- **Group 0:** uniform buffer, trailRead, trailWrite, outRead, outWrite
- **Group 1:** agent storage buffer (read_write)

### Step Execution Order

Same as Physarum with the material pick/drop integrated into `move_agents`:

1. Upload uniforms
2. Build group 0 for current ping-pong state
3. Dispatch `move_agents` (sense + turn + move + pick/drop)
4. Dispatch `diffuse_texture` (trailRead → trailWrite, blur pheromone)
5. Copy trailWrite → trailRead
6. Rebuild group 0
7. Dispatch `write_trails` (deposit pheromone at agent positions)
8. Swap trail ping-pong
9. Dispatch `render` (trailRead + outRead → outWrite)
10. Swap output ping-pong

### Files to Create

- `src/algorithms/termites.h` — class declaration (~90 lines, mirrors physarum.h)
- `src/algorithms/termites.cpp` — implementation (~600 lines, mirrors physarum.cpp)
- `shaders/termites.wgsl` — 6 compute kernels (~300 lines)

### Files to Modify

- `src/main.cpp` — Add `#include "algorithms/termites.h"`, add to `sims[]` array, bump `simCount` to 4, add name to `simNames[]`
- `CMakeLists.txt` — Add `src/algorithms/termites.cpp` to sources, add `termites.wgsl` to shader copy list

### ImGui Controls

Same pattern as Physarum:
- Play/Pause/Step/Reset buttons
- Steps/Frame slider
- Agent count input (triggers reset)
- Randomize button
- Preset save/load
- "Link All Types" checkbox
- Per-type parameter sliders (in tree nodes when unlinked)

---

## Part B: Compositing System

### Description

A compute-based compositing pass that blends the output of multiple simulations into a single output texture. This runs as a separate compute pipeline after individual simulation steps, taking N simulation output textures and producing a blended result.

### Design

The compositor is a new class (`Compositor`) that:
1. Takes references to N simulation output views
2. Applies per-layer blend modes and opacity
3. Outputs to its own texture (fed into PostEffects)

### Blend Modes

| Mode | Formula | Description |
|------|---------|-------------|
| Add | `A + B` | Additive blending (clamped) |
| Multiply | `A * B` | Darkening blend |
| Screen | `1 - (1-A)(1-B)` | Lightening blend |
| Max | `max(A, B)` | Brightest pixel wins |
| Average | `(A + B) / N` | Equal weighted average |

### Per-Layer Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `enabled` | true | bool | Enable/disable layer |
| `opacity` | 1.0 | 0.0–1.0 | Layer opacity |
| `blendMode` | Add | enum | Blend mode selector |

### Architecture

```
Sim A ──→ outputA ──┐
Sim B ──→ outputB ──┤──→ Compositor ──→ compositeOutput ──→ PostEffects ──→ screen
Sim C ──→ outputC ──┘
```

When only 1 simulation is active (the common case), the compositor passes through directly.

### Implementation Details

**Shader approach:** A single compute shader kernel that reads from up to 4 input textures and writes a blended output. The uniform buffer specifies which layers are enabled, their opacity, and blend mode.

```wgsl
// compositor.wgsl
@group(0) @binding(0) var<uniform> params: CompositorParams;
@group(0) @binding(1) var layer0: texture_2d<f32>;
@group(0) @binding(2) var layer1: texture_2d<f32>;
@group(0) @binding(3) var layer2: texture_2d<f32>;
@group(0) @binding(4) var layer3: texture_2d<f32>;
@group(0) @binding(5) var output: texture_storage_2d<rgba8unorm, write>;
```

**CompositorParams:**
```wgsl
struct CompositorParams {
    width: u32,
    height: u32,
    layerCount: u32,
    _pad: u32,
    // per-layer: vec4f where x=enabled, y=opacity, z=blendMode, w=unused
    layer0: vec4f,
    layer1: vec4f,
    layer2: vec4f,
    layer3: vec4f,
};
```

### Files to Create

- `src/compositor.h` — class declaration (~50 lines)
- `src/compositor.cpp` — implementation (~200 lines)
- `shaders/compositor.wgsl` — blend kernel (~80 lines)

### Files to Modify

- `src/main.cpp` — Add compositor between simulation step and post-effects; modify main loop to optionally run multiple sims and composite their outputs
- `CMakeLists.txt` — Add compositor source and shader

### Main Loop Changes

The main loop needs to support two modes:
1. **Single sim mode** (current behavior) — one sim active, output goes directly to post-effects
2. **Composite mode** — multiple sims run simultaneously, outputs blended via compositor, result goes to post-effects

A checkbox in Settings window toggles composite mode. When enabled, each sim gets its own enable/opacity/blend controls.

### ImGui Controls

New "Compositor" window:
- Enable/disable composite mode checkbox
- Per-simulation layer controls (when composite mode is on):
  - Enable checkbox
  - Opacity slider (0–1)
  - Blend mode combo (Add, Multiply, Screen, Max, Average)

---

## Implementation Order

### Step 1: Termites Core (Issue #4)
- Create `termites.h`, `termites.cpp`, `termites.wgsl`
- 6 compute kernels matching the Physarum pattern
- Register in main.cpp as 4th simulation
- Basic ImGui controls (play/pause/step/reset, per-type params)

### Step 2: Termites Polish (Issue #5)
- Randomize button
- Preset save/load
- Link all types toggle
- Agent count resize with buffer recreation
- Toroidal boundary wrapping

### Step 3: Compositor (Issue #6)
- Create `compositor.h`, `compositor.cpp`, `compositor.wgsl`
- Blend modes: Add, Multiply, Screen, Max, Average
- Per-layer enable/opacity/blend controls
- Integrate into main loop (single sim passthrough + multi-sim composite)
- ImGui compositor window

---

## Dependencies & Risks

1. **Agent buffer size change (24 bytes):** The termite agent struct is larger than Physarum's 16-byte struct. This is handled naturally since each simulation manages its own buffer.

2. **Concurrent simulation execution:** The compositor requires multiple sims to be initialized simultaneously. Currently only one sim is active. Main loop needs modification to maintain multiple active sims when composite mode is on.

3. **Memory pressure:** Running 4 sims simultaneously at 1536x1536 with rgba16float trails requires ~72MB of texture memory (4 sims × 2 ping-pong × 2 textures × 1536² × 8 bytes). This is manageable on modern GPUs.

4. **Shader compilation:** Each new WGSL file adds compile time at startup. The existing pattern (compile on init, cache pipeline) handles this adequately.
