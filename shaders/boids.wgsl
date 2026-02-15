// Boids flocking — 4 competitive flock types with GPU spatial hashing

struct BoidAgent {
    position: vec2f,
    velocity: vec2f,
    acceleration: vec2f,
    type_id: u32,
    cell_id: u32,
    separateCount: f32,
    alignCount: f32,
    cohesionCount: f32,
    padding: f32,
};

struct Params {
    rez_agents_time: vec4u,        // rezX, rezY, agentsCount, time
    grid_params: vec4f,            // cellSize, gridW, gridH, maxPerCell
    maxSpeeds: vec4f,
    maxForces: vec4f,
    typeSeparateRanges: vec4f,
    globalSeparateRanges: vec4f,
    alignRanges: vec4f,
    attractRanges: vec4f,
    foodSensorDistances: vec4f,
    sensorAngles: vec4f,
    foodStrengths: vec4f,
    depositAmounts: vec4f,
    eatAmounts: vec4f,
    diffuseRates: vec4f,
    hues: vec4f,
    saturations: vec4f,
};

// Group 0: textures + uniforms
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var trailRead: texture_2d<f32>;
@group(0) @binding(2) var trailWrite: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var outRead: texture_2d<f32>;
@group(0) @binding(4) var outWrite: texture_storage_2d<rgba8unorm, write>;

// Group 1: agent buffer
@group(1) @binding(0) var<storage, read_write> agents: array<BoidAgent>;

// Group 2: spatial hash grid
@group(2) @binding(0) var<storage, read_write> cellCount: array<atomic<u32>>;
@group(2) @binding(1) var<storage, read_write> cellAgents: array<u32>;

// ---- Helpers ----

fn random2(p: vec2f) -> vec2f {
    var a = fract(vec3f(p.x, p.y, p.x) * vec3f(123.34, 234.34, 345.65));
    a = a + dot(a, a + 34.45);
    return fract(vec2f(a.x * a.y, a.y * a.z));
}

fn hsb2rgb(c: vec3f, a: f32) -> vec4f {
    let rgb = clamp(
        abs(((c.x * 6.0 + vec3f(0.0, 4.0, 2.0)) % 6.0) - 3.0) - 1.0,
        vec3f(0.0), vec3f(1.0)
    );
    let smoothed = rgb * rgb * (3.0 - 2.0 * rgb);
    let o = c.z * mix(vec3f(1.0), smoothed, c.y);
    return vec4f(o, a);
}

fn select_channel(v: vec4f, ch: i32) -> f32 {
    if (ch == 0) { return v.x; }
    if (ch == 1) { return v.y; }
    if (ch == 2) { return v.z; }
    return v.w;
}

fn get_rez() -> vec2u {
    return vec2u(params.rez_agents_time.x, params.rez_agents_time.y);
}

fn get_agents_count() -> u32 {
    return params.rez_agents_time.z;
}

fn get_time() -> u32 {
    return params.rez_agents_time.w;
}

fn limit_vec(v: vec2f, maxLen: f32) -> vec2f {
    let lenSq = dot(v, v);
    if (lenSq > maxLen * maxLen) {
        return normalize(v) * maxLen;
    }
    return v;
}

fn rotate_vec2(v: vec2f, angle: f32) -> vec2f {
    let c = cos(angle);
    let s = sin(angle);
    return vec2f(v.x * c - v.y * s, v.x * s + v.y * c);
}

fn sample_trail(pos: vec2i, rez: vec2u) -> vec4f {
    let wrapped = vec2i(
        (pos.x + i32(rez.x)) % i32(rez.x),
        (pos.y + i32(rez.y)) % i32(rez.y)
    );
    return textureLoad(trailRead, vec2u(u32(wrapped.x), u32(wrapped.y)), 0);
}

fn toroidal_diff(a: vec2f, b: vec2f, rez: vec2f) -> vec2f {
    var d = a - b;
    if (abs(d.x) > rez.x * 0.5) { d.x -= sign(d.x) * rez.x; }
    if (abs(d.y) > rez.y * 0.5) { d.y -= sign(d.y) * rez.y; }
    return d;
}

// ---- Kernel 1: Reset Texture ----
@compute @workgroup_size(8, 8)
fn reset_texture(@builtin(global_invocation_id) gid: vec3u) {
    let rez = get_rez();
    if (gid.x >= rez.x || gid.y >= rez.y) { return; }
    textureStore(trailWrite, gid.xy, vec4f(0.0));
}

// ---- Kernel 2: Reset Agents ----
@compute @workgroup_size(256)
fn reset_agents(@builtin(global_invocation_id) gid: vec3u) {
    let count = get_agents_count();
    if (gid.x >= count) { return; }
    let rez = get_rez();
    let rezF = vec2f(f32(rez.x), f32(rez.y));
    let t = f32(get_time());

    // Ring distribution
    let c = random2(vec2f(f32(gid.x)) * 0.0001 + t * 0.001);
    let ang = c.x * 6.28318530718;
    let outerRadius = min(rezF.x, rezF.y) * 0.25;
    let ringThickness = rezF.x / 10.0;
    let innerRadius = outerRadius - ringThickness;
    let rad = mix(innerRadius, outerRadius, c.y);
    let center = rezF * 0.5;
    let pos = center + vec2f(cos(ang), sin(ang)) * rad;

    // Random velocity
    let r2 = random2(vec2f(f32(gid.x), f32(gid.x)) * 0.001 + sin(t));
    let vel = normalize(2.0 * (r2 - 0.5)) * 0.5;

    // Type by quarter
    var typeId = 3u;
    if (gid.x < count / 4u) { typeId = 0u; }
    else if (gid.x < count / 2u) { typeId = 1u; }
    else if (gid.x < count * 3u / 4u) { typeId = 2u; }

    agents[gid.x] = BoidAgent(pos, vel, vec2f(0.0), typeId, 0u, 0.0, 0.0, 0.0, 0.0);
}

// ---- Kernel 3: Clear Grid ----
@compute @workgroup_size(256)
fn clear_grid(@builtin(global_invocation_id) gid: vec3u) {
    let gridW = u32(params.grid_params.y);
    let gridH = u32(params.grid_params.z);
    let totalCells = gridW * gridH;
    if (gid.x >= totalCells) { return; }
    atomicStore(&cellCount[gid.x], 0u);
}

// ---- Kernel 4: Assign Cells ----
@compute @workgroup_size(256)
fn assign_cells(@builtin(global_invocation_id) gid: vec3u) {
    let count = get_agents_count();
    if (gid.x >= count) { return; }

    var b = agents[gid.x];
    let cellSz = params.grid_params.x;
    let gridW = u32(params.grid_params.y);
    let gridH = u32(params.grid_params.z);
    let maxPerCell = u32(params.grid_params.w);

    let cx = min(u32(max(floor(b.position.x / cellSz), 0.0)), gridW - 1u);
    let cy = min(u32(max(floor(b.position.y / cellSz), 0.0)), gridH - 1u);
    let cellIdx = cx + cy * gridW;

    let localIdx = atomicAdd(&cellCount[cellIdx], 1u);
    if (localIdx < maxPerCell) {
        cellAgents[cellIdx * maxPerCell + localIdx] = gid.x;
    }

    b.cell_id = cellIdx;
    agents[gid.x] = b;
}

// ---- Kernel 5: Move Agents ----
@compute @workgroup_size(256)
fn move_agents(@builtin(global_invocation_id) gid: vec3u) {
    let count = get_agents_count();
    if (gid.x >= count) { return; }

    var b = agents[gid.x];
    let agentType = i32(b.type_id);
    let rez = get_rez();
    let rezF = vec2f(f32(rez.x), f32(rez.y));

    let maxSpd = select_channel(params.maxSpeeds, agentType);
    let maxFrc = select_channel(params.maxForces, agentType);
    let typeSepRange = select_channel(params.typeSeparateRanges, agentType);
    let globSepRange = select_channel(params.globalSeparateRanges, agentType);
    let alignRng = select_channel(params.alignRanges, agentType);
    let attractRng = select_channel(params.attractRanges, agentType);

    let cellSz = params.grid_params.x;
    let gridW = u32(params.grid_params.y);
    let gridH = u32(params.grid_params.z);
    let maxPerCell = u32(params.grid_params.w);

    let cellX = i32(floor(b.position.x / cellSz));
    let cellY = i32(floor(b.position.y / cellSz));

    var typeSepSum = vec2f(0.0);
    var typeSepCnt = 0.0;
    var globSepSum = vec2f(0.0);
    var globSepCnt = 0.0;
    var alignSum = vec2f(0.0);
    var alignCnt = 0.0;
    var cohesionDir = vec2f(0.0);
    var cohesionCnt = 0.0;

    // 3x3 neighbor cell lookup
    for (var dx = -1; dx <= 1; dx++) {
        for (var dy = -1; dy <= 1; dy++) {
            let nx = (cellX + dx + i32(gridW)) % i32(gridW);
            let ny = (cellY + dy + i32(gridH)) % i32(gridH);
            let nIdx = u32(nx) + u32(ny) * gridW;
            let cnt = min(atomicLoad(&cellCount[nIdx]), maxPerCell);

            for (var k = 0u; k < cnt; k++) {
                let otherIdx = cellAgents[nIdx * maxPerCell + k];
                if (otherIdx == gid.x) { continue; }

                let other = agents[otherIdx];
                let d = toroidal_diff(b.position, other.position, rezF);
                let sqDist = dot(d, d);

                // Type separation (same type)
                if (other.type_id == b.type_id && sqDist < typeSepRange && sqDist > 0.0) {
                    typeSepSum += d / sqDist;
                    typeSepCnt += 1.0;
                }

                // Global separation (all types)
                if (sqDist < globSepRange && sqDist > 0.0) {
                    globSepSum += d / sqDist;
                    globSepCnt += 1.0;
                }

                // Alignment (same type)
                if (other.type_id == b.type_id && sqDist < alignRng && sqDist > 0.0) {
                    alignSum += other.velocity;
                    alignCnt += 1.0;
                }

                // Cohesion (same type) — accumulate direction toward neighbor
                if (other.type_id == b.type_id && sqDist < attractRng && sqDist > 0.0) {
                    cohesionDir += -d;
                    cohesionCnt += 1.0;
                }
            }
        }
    }

    // Compute forces
    b.acceleration = vec2f(0.0);

    if (typeSepCnt > 0.0) {
        let desired = normalize(typeSepSum / typeSepCnt) * maxSpd;
        b.acceleration += limit_vec(desired - b.velocity, maxFrc);
    }

    if (globSepCnt > 0.0) {
        let desired = normalize(globSepSum / globSepCnt) * maxSpd;
        b.acceleration += limit_vec(desired - b.velocity, maxFrc);
    }

    if (alignCnt > 0.0) {
        let desired = normalize(alignSum / alignCnt) * maxSpd;
        b.acceleration += limit_vec(desired - b.velocity, maxFrc);
    }

    if (cohesionCnt > 0.0) {
        let desired = normalize(cohesionDir / cohesionCnt) * maxSpd;
        b.acceleration += limit_vec(desired - b.velocity, maxFrc);
    }

    // Food sensing from trail (3 sensors)
    let foodDist = select_channel(params.foodSensorDistances, agentType);
    let foodAngle = select_channel(params.sensorAngles, agentType);
    let foodStr = select_channel(params.foodStrengths, agentType);

    if (foodStr > 0.0 && dot(b.velocity, b.velocity) > 0.001) {
        let normVel = normalize(b.velocity);
        let posAhead = b.position + normVel * foodDist;
        let posLeft = b.position + rotate_vec2(normVel, foodAngle) * foodDist;
        let posRight = b.position + rotate_vec2(normVel, -foodAngle) * foodDist;

        let fAhead = select_channel(sample_trail(vec2i(i32(posAhead.x), i32(posAhead.y)), rez), agentType);
        let fLeft = select_channel(sample_trail(vec2i(i32(posLeft.x), i32(posLeft.y)), rez), agentType);
        let fRight = select_channel(sample_trail(vec2i(i32(posRight.x), i32(posRight.y)), rez), agentType);

        var foodForce = vec2f(0.0);
        if (fLeft > fAhead && fLeft > fRight) {
            let desiredVel = rotate_vec2(normVel, foodAngle) * maxSpd;
            foodForce = limit_vec(desiredVel - b.velocity, maxFrc * foodStr);
        } else if (fRight > fAhead && fRight > fLeft) {
            let desiredVel = rotate_vec2(normVel, -foodAngle) * maxSpd;
            foodForce = limit_vec(desiredVel - b.velocity, maxFrc * foodStr);
        } else if (fAhead > 0.01) {
            let desiredVel = normVel * maxSpd;
            foodForce = limit_vec(desiredVel - b.velocity, maxFrc * foodStr * 0.5);
        }
        b.acceleration += foodForce;
    }

    // Update velocity and position
    b.velocity = limit_vec(b.velocity + b.acceleration, maxSpd);
    b.position += b.velocity;

    // Toroidal wrap with x-flip on y boundary
    if (b.position.x < 0.0) { b.position.x += rezF.x; }
    if (b.position.x >= rezF.x) { b.position.x -= rezF.x; }
    if (b.position.y < 0.0 || b.position.y >= rezF.y) {
        b.position.x = rezF.x - b.position.x;
    }
    if (b.position.y < 0.0) { b.position.y += rezF.y; }
    if (b.position.y >= rezF.y) { b.position.y -= rezF.y; }

    b.separateCount = typeSepCnt;
    b.alignCount = alignCnt;
    b.cohesionCount = cohesionCnt;

    agents[gid.x] = b;
}

// ---- Kernel 6: Write Trails ----
@compute @workgroup_size(256)
fn write_trails(@builtin(global_invocation_id) gid: vec3u) {
    let count = get_agents_count();
    if (gid.x >= count) { return; }

    let a = agents[gid.x];
    let agentType = i32(a.type_id);
    let px = vec2u(u32(round(a.position.x)), u32(round(a.position.y)));
    let rez = get_rez();
    if (px.x >= rez.x || px.y >= rez.y) { return; }

    var env = textureLoad(trailRead, px, 0);
    let deposit = select_channel(params.depositAmounts, agentType);
    let eat = select_channel(params.eatAmounts, agentType);

    if (agentType == 0) {
        env.x = clamp(env.x + deposit, 0.0, 1.0);
        env.y = clamp(env.y - eat, 0.0, 1.0);
        env.z = clamp(env.z - eat, 0.0, 1.0);
        env.w = clamp(env.w - eat, 0.0, 1.0);
    } else if (agentType == 1) {
        env.x = clamp(env.x - eat, 0.0, 1.0);
        env.y = clamp(env.y + deposit, 0.0, 1.0);
        env.z = clamp(env.z - eat, 0.0, 1.0);
        env.w = clamp(env.w - eat, 0.0, 1.0);
    } else if (agentType == 2) {
        env.x = clamp(env.x - eat, 0.0, 1.0);
        env.y = clamp(env.y - eat, 0.0, 1.0);
        env.z = clamp(env.z + deposit, 0.0, 1.0);
        env.w = clamp(env.w - eat, 0.0, 1.0);
    } else {
        env.x = clamp(env.x - eat, 0.0, 1.0);
        env.y = clamp(env.y - eat, 0.0, 1.0);
        env.z = clamp(env.z - eat, 0.0, 1.0);
        env.w = clamp(env.w + deposit, 0.0, 1.0);
    }

    textureStore(trailWrite, px, env);
}

// ---- Kernel 7: Diffuse Texture ----
@compute @workgroup_size(8, 8)
fn diffuse_texture(@builtin(global_invocation_id) gid: vec3u) {
    let rez = get_rez();
    if (gid.x >= rez.x || gid.y >= rez.y) { return; }

    var avg = vec4f(0.0);
    for (var dx = -1; dx <= 1; dx++) {
        for (var dy = -1; dy <= 1; dy++) {
            avg += sample_trail(vec2i(i32(gid.x) + dx, i32(gid.y) + dy), rez);
        }
    }
    avg = avg / 9.0;

    var oc = vec4f(
        avg.x * params.diffuseRates.x,
        avg.y * params.diffuseRates.y,
        avg.z * params.diffuseRates.z,
        avg.w * params.diffuseRates.w
    );
    oc = clamp(oc, vec4f(0.0), vec4f(1.0));

    textureStore(trailWrite, gid.xy, oc);
}

// ---- Kernel 8: Render ----
@compute @workgroup_size(8, 8)
fn render(@builtin(global_invocation_id) gid: vec3u) {
    let rez = get_rez();
    if (gid.x >= rez.x || gid.y >= rez.y) { return; }

    let trail = textureLoad(trailRead, gid.xy, 0);

    let c1 = hsb2rgb(vec3f(params.hues.x, params.saturations.x, 0.8 * trail.x), trail.x);
    let c2 = hsb2rgb(vec3f(params.hues.y, params.saturations.y, 0.8 * trail.y), trail.y);
    let c3 = hsb2rgb(vec3f(params.hues.z, params.saturations.z, 0.8 * trail.z), trail.z);
    let c4 = hsb2rgb(vec3f(params.hues.w, params.saturations.w, 0.8 * trail.w), trail.w);

    var currentColor = textureLoad(outRead, gid.xy, 0);
    currentColor = currentColor + c1 * 0.25 + c2 * 0.25 + c3 * 0.25 + c4 * 0.25;
    currentColor = currentColor * 0.65;

    textureStore(outWrite, gid.xy, currentColor);
}
