#!/usr/bin/env bash
# Phase 4 GitHub Issue Creation Script
# Run with: bash docs/phase4-issues.sh
# Requires: gh CLI authenticated (gh auth login)

set -euo pipefail

REPO="merttoka/nature-of-nature"

echo "Creating Phase 4 GitHub issues..."

# Issue #4: Termites Core
gh issue create \
  --repo "$REPO" \
  --title "Phase 4a: Termites Simulation — Core Implementation" \
  --label "phase-4" \
  --body "$(cat <<'EOF'
## Termites Simulation — Core Implementation

Port the termites algorithm from Processing to GPU compute, following the Physarum architecture pattern.

### Algorithm

Nest-building agents that pick up and deposit material based on local pheromone concentrations:
1. Sense pheromone at 3 sensor positions (left, center, right)
2. Turn toward highest pheromone (biased random walk)
3. Move forward
4. Pick up / drop material (probabilistic, based on pheromone threshold)
5. Deposit pheromone at current position

### Agent Struct (24 bytes)

```wgsl
struct TermiteAgent {
    position: vec2f,
    direction: vec2f,
    carrying: f32,
    _pad: f32,
};
```

### Textures

- `trailTextures` (rgba16float ping-pong): R=pheromone, G=material density
- `outputTextures` (rgba8unorm ping-pong): rendered display output

### Compute Kernels (6)

| # | Kernel | WG | Description |
|---|--------|----|-------------|
| 1 | `reset_texture` | 8×8 | Clear textures |
| 2 | `reset_agents` | 256 | Random scatter, random carrying state |
| 3 | `move_agents` | 256 | Sense + turn + move + pick/drop |
| 4 | `write_trails` | 256 | Deposit pheromone |
| 5 | `diffuse_texture` | 8×8 | Blur + evaporation |
| 6 | `render` | 8×8 | HSB color map |

### Per-Type Parameters (4 types)

senseAngle, senseDistance, turnAngle, moveSpeed, depositPheromone, pickupProb, dropProb, pheromoneThreshold, diffuseRate, hue, saturation

### Files to Create

- `src/algorithms/termites.h`
- `src/algorithms/termites.cpp`
- `shaders/termites.wgsl`

### Files to Modify

- `src/main.cpp` — register as 4th simulation
- `CMakeLists.txt` — add source + shader

### Acceptance Criteria

- [ ] Termites appears in simulation combo dropdown
- [ ] Agents visibly move, deposit pheromone trails, and form material clusters
- [ ] Play/Pause/Step/Reset controls work
- [ ] Per-type parameter sliders in ImGui
- [ ] Post-effects pipeline works with termites output
- [ ] No GPU validation errors
EOF
)"

echo "Created: Termites Core"

# Issue #5: Termites Polish
gh issue create \
  --repo "$REPO" \
  --title "Phase 4b: Termites — Polish & Presets" \
  --label "phase-4" \
  --body "$(cat <<'EOF'
## Termites — Polish & Presets

Complete termites UI features to match Physarum and Boids quality level.

**Depends on:** Phase 4a (Termites Core)

### Tasks

- [ ] **Randomize button** — random per-type parameters within valid ranges
- [ ] **Preset save/load** — key-value text format matching existing preset system
- [ ] **Link All Types toggle** — single set of sliders controlling all 4 types
- [ ] **Agent count resize** — InputInt with buffer recreation on change
- [ ] **Toroidal boundary wrapping** — X wrap, Y flip-wrap (matching Physarum behavior)
- [ ] **Steps/Frame slider** — multiple compute steps per render frame

### Files to Modify

- `src/algorithms/termites.cpp` — add randomize, preset, link-types UI
- `src/algorithms/termites.h` — add link-types flag, steps-per-frame

### Acceptance Criteria

- [ ] Randomize produces visually diverse results
- [ ] Presets save/load round-trip correctly
- [ ] Link All Types syncs all 4 types from single slider set
- [ ] Agent count change triggers clean reset (no crashes)
- [ ] Boundary behavior wraps correctly (no edge artifacts)
EOF
)"

echo "Created: Termites Polish"

# Issue #6: Compositor
gh issue create \
  --repo "$REPO" \
  --title "Phase 4c: Compositing System" \
  --label "phase-4" \
  --body "$(cat <<'EOF'
## Compositing System

Blend multiple simulation outputs into a single frame via a GPU compute compositor.

### Architecture

```
Sim A → outputA ─┐
Sim B → outputB ─┤→ Compositor → compositeOutput → PostEffects → screen
Sim C → outputC ─┘
```

Single sim mode passes through directly (no overhead).

### Blend Modes

| Mode | Formula |
|------|---------|
| Add | `A + B` (clamped) |
| Multiply | `A * B` |
| Screen | `1 - (1-A)(1-B)` |
| Max | `max(A, B)` |
| Average | `(A + B) / N` |

### Per-Layer Controls

- Enable/disable (checkbox)
- Opacity (0–1 slider)
- Blend mode (combo selector)

### Shader Design

Single compute kernel reading up to 4 input textures:

```wgsl
@group(0) @binding(0) var<uniform> params: CompositorParams;
@group(0) @binding(1) var layer0: texture_2d<f32>;
@group(0) @binding(2) var layer1: texture_2d<f32>;
@group(0) @binding(3) var layer2: texture_2d<f32>;
@group(0) @binding(4) var layer3: texture_2d<f32>;
@group(0) @binding(5) var output: texture_storage_2d<rgba8unorm, write>;
```

### Main Loop Changes

- Support running multiple sims simultaneously when composite mode is enabled
- Toggle between single-sim mode (current) and composite mode
- Each sim maintains its own init/step/shutdown lifecycle

### Files to Create

- `src/compositor.h`
- `src/compositor.cpp`
- `shaders/compositor.wgsl`

### Files to Modify

- `src/main.cpp` — composite mode toggle, multi-sim execution, compositor integration
- `CMakeLists.txt` — add compositor source + shader

### ImGui Controls

New "Compositor" window:
- Composite mode toggle
- Per-sim layer enable/opacity/blend controls (when composite mode is on)

### Acceptance Criteria

- [ ] Single-sim mode works identically to current behavior
- [ ] Enabling composite mode allows multiple sims to run simultaneously
- [ ] All 5 blend modes produce correct visual results
- [ ] Per-layer opacity modulates blend contribution
- [ ] Toggling layers on/off updates output in real-time
- [ ] Post-effects apply to composited output
- [ ] No performance regression in single-sim mode
EOF
)"

echo "Created: Compositor"

echo ""
echo "Updating master issue #1 with Phase 4 sub-issue references..."

# Update master issue #1 Phase 4 section
# Note: You'll need to manually update the Phase 4 section of issue #1 to reference these new issues
echo ""
echo "Done! Please update issue #1's Phase 4 section to reference the newly created issues."
echo "The new issues should be #4, #5, and #6."
