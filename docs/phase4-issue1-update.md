## Phase 4 — Termites + Compositing

> **Status: Planned** — Sub-issues: #4, #5, #6

### 4a. Termites Core (#4)
- [ ] 6 compute kernels (reset_texture, reset_agents, move_agents, write_trails, diffuse_texture, render)
- [ ] 4 termite types with biased random walk, pheromone sensing, probabilistic pick/drop
- [ ] 24-byte agent struct (position + direction + carrying state)
- [ ] Pheromone trail textures (rgba16float) + material density
- [ ] Register as 4th simulation in main loop
- [ ] Per-type parameter sliders in ImGui

### 4b. Termites Polish (#5)
- [ ] Randomize button for parameter exploration
- [ ] Preset save/load (text format)
- [ ] Link All Types toggle
- [ ] Agent count resize with buffer recreation
- [ ] Toroidal boundary wrapping

### 4c. Compositing System (#6)
- [ ] Compute-based compositor: blend N simulation outputs
- [ ] 5 blend modes: Add, Multiply, Screen, Max, Average
- [ ] Per-layer enable/opacity/blend controls
- [ ] Single-sim passthrough (zero overhead when not compositing)
- [ ] Multi-sim simultaneous execution mode
- [ ] ImGui compositor window
