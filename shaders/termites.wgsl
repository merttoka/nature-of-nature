// Termites simulation — 4 competitive agent types
// Trail texture = pheromone (decays, used for sensing/navigation)
// Mound texture = persistent material deposits (no decay, probabilistic)

struct Agent {
    position: vec2f,
    direction: vec2f,
};

struct Params {
    rez_agents_time: vec4u,    // x=rezX, y=rezY, z=agentsCount, w=time
    senseAngles: vec4f,        // per-type (radians)
    senseDistances: vec4f,
    turnAngles: vec4f,         // per-type (radians)
    moveSpeeds: vec4f,
    depositAmounts: vec4f,
    depositRates: vec4f,       // probability 0-1 of mound deposit per frame
    decayRates: vec4f,         // pheromone trail decay (no blur)
    hues: vec4f,
    saturations: vec4f,
    typeRatios: vec4f,         // cumulative thresholds for type distribution
};

// Group 0: textures + uniforms
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var trailRead: texture_2d<f32>;
@group(0) @binding(2) var trailWrite: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var moundRead: texture_2d<f32>;
@group(0) @binding(4) var moundWrite: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var outRead: texture_2d<f32>;
@group(0) @binding(6) var outWrite: texture_storage_2d<rgba8unorm, write>;

// Group 1: agent buffer
@group(1) @binding(0) var<storage, read_write> agents: array<Agent>;

// ---- Helpers ----

fn random2(p: vec2f) -> vec2f {
    var a = fract(vec3f(p.x, p.y, p.x) * vec3f(123.34, 234.34, 345.65));
    a = a + dot(a, a + 34.45);
    return fract(vec2f(a.x * a.y, a.y * a.z));
}

fn rotate_vec2(v: vec2f, angle: f32) -> vec2f {
    let c = cos(angle);
    let s = sin(angle);
    return vec2f(v.x * c - v.y * s, v.x * s + v.y * c);
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

fn get_agent_type(id: u32, count: u32) -> i32 {
    let frac = f32(id) / f32(count);
    if (frac < params.typeRatios.x) { return 0; }
    if (frac < params.typeRatios.y) { return 1; }
    if (frac < params.typeRatios.z) { return 2; }
    return 3;
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

fn select_channel(v: vec4f, ch: i32) -> f32 {
    if (ch == 0) { return v.x; }
    if (ch == 1) { return v.y; }
    if (ch == 2) { return v.z; }
    return v.w;
}

fn sample_trail(pos: vec2i, rez: vec2u) -> vec4f {
    let wrapped = vec2i(
        (pos.x + i32(rez.x)) % i32(rez.x),
        (pos.y + i32(rez.y)) % i32(rez.y)
    );
    return textureLoad(trailRead, vec2u(u32(wrapped.x), u32(wrapped.y)), 0);
}

// ---- Kernel 1: Reset Texture ----
@compute @workgroup_size(8, 8)
fn reset_texture(@builtin(global_invocation_id) gid: vec3u) {
    let rez = get_rez();
    if (gid.x >= rez.x || gid.y >= rez.y) { return; }
    textureStore(trailWrite, gid.xy, vec4f(0.0));
    textureStore(moundWrite, gid.xy, vec4f(0.0));
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

    // Random direction
    let r2 = random2(vec2f(f32(gid.x), f32(gid.x)) * 0.001 + sin(t));
    let dir = normalize(2.0 * (r2 - 0.5));

    agents[gid.x] = Agent(pos, dir);
}

// ---- Kernel 3: Move Agents (biased random walk, senses trails) ----
@compute @workgroup_size(256)
fn move_agents(@builtin(global_invocation_id) gid: vec3u) {
    let count = get_agents_count();
    if (gid.x >= count) { return; }
    let rez = get_rez();
    let rezF = vec2f(f32(rez.x), f32(rez.y));
    let t = f32(get_time());

    var a = agents[gid.x];
    let agentType = get_agent_type(gid.x, count);

    let direction = normalize(a.direction);

    let senseAngle = select_channel(params.senseAngles, agentType);
    let senseDist = select_channel(params.senseDistances, agentType);
    let turnAngle = select_channel(params.turnAngles, agentType);
    let speed = select_channel(params.moveSpeeds, agentType);

    // 3 sensors — read from trail (pheromone) texture
    let leftSensor = rotate_vec2(direction, -senseAngle) * senseDist;
    let middleSensor = direction * senseDist;
    let rightSensor = rotate_vec2(direction, senseAngle) * senseDist;

    let leftCoord = a.position + leftSensor;
    let middleCoord = a.position + middleSensor;
    let rightCoord = a.position + rightSensor;

    let leftVal = sample_trail(vec2i(i32(leftCoord.x), i32(leftCoord.y)), rez);
    let middleVal = sample_trail(vec2i(i32(middleCoord.x), i32(middleCoord.y)), rez);
    let rightVal = sample_trail(vec2i(i32(rightCoord.x), i32(rightCoord.y)), rez);

    // Competitive sensing: own channel - sum of other channels
    let ownL = select_channel(leftVal, agentType);
    let othersL = (leftVal.x + leftVal.y + leftVal.z + leftVal.w) - ownL;
    let leftLevel = ownL - othersL;

    let ownM = select_channel(middleVal, agentType);
    let othersM = (middleVal.x + middleVal.y + middleVal.z + middleVal.w) - ownM;
    let middleLevel = ownM - othersM;

    let ownR = select_channel(rightVal, agentType);
    let othersR = (rightVal.x + rightVal.y + rightVal.z + rightVal.w) - ownR;
    let rightLevel = ownR - othersR;

    // Turn decision (biased random walk)
    var d = direction;
    if (middleLevel > leftLevel && middleLevel > rightLevel) {
        d = direction;
    } else if (leftLevel > rightLevel) {
        d = rotate_vec2(d, -turnAngle);
    } else if (rightLevel > leftLevel) {
        d = rotate_vec2(d, turnAngle);
    } else {
        let rnd = random2(vec2f(f32(gid.x), f32(gid.x)) * 0.01 + sin(t) * 0.01);
        var sign = 1.0;
        if (rnd.x > 0.5) { sign = -1.0; }
        d = rotate_vec2(d, sign * turnAngle);
    }

    d = normalize(d);
    a.direction = d * speed;
    a.position = a.position + a.direction;

    // Toroidal wrapping
    if (a.position.x < 0.0) { a.position.x += rezF.x; }
    if (a.position.x >= rezF.x) { a.position.x -= rezF.x; }
    if (a.position.y < 0.0) { a.position.y += rezF.y; }
    if (a.position.y >= rezF.y) { a.position.y -= rezF.y; }

    agents[gid.x] = a;
}

// ---- Kernel 4: Decay + Copy (trail decays, mound persists) ----
@compute @workgroup_size(8, 8)
fn decay_texture(@builtin(global_invocation_id) gid: vec3u) {
    let rez = get_rez();
    if (gid.x >= rez.x || gid.y >= rez.y) { return; }

    // Trail: exponential decay
    let val = textureLoad(trailRead, gid.xy, 0);
    let decayed = vec4f(
        val.x * params.decayRates.x,
        val.y * params.decayRates.y,
        val.z * params.decayRates.z,
        val.w * params.decayRates.w
    );
    textureStore(trailWrite, gid.xy, clamp(decayed, vec4f(0.0), vec4f(1.0)));

    // Mound: identity copy (no decay — persists until reset)
    let mound = textureLoad(moundRead, gid.xy, 0);
    textureStore(moundWrite, gid.xy, mound);
}

// ---- Kernel 5: Write Trails + Mounds ----
@compute @workgroup_size(256)
fn write_trails(@builtin(global_invocation_id) gid: vec3u) {
    let count = get_agents_count();
    if (gid.x >= count) { return; }
    let t = f32(get_time());

    let a = agents[gid.x];
    let agentType = get_agent_type(gid.x, count);
    let px = vec2u(u32(round(a.position.x)), u32(round(a.position.y)));
    let deposit = select_channel(params.depositAmounts, agentType);

    // Always deposit pheromone trail (for navigation)
    var trail = textureLoad(trailRead, px, 0);
    if (agentType == 0) {
        trail.x = clamp(trail.x + deposit, 0.0, 1.0);
    } else if (agentType == 1) {
        trail.y = clamp(trail.y + deposit, 0.0, 1.0);
    } else if (agentType == 2) {
        trail.z = clamp(trail.z + deposit, 0.0, 1.0);
    } else {
        trail.w = clamp(trail.w + deposit, 0.0, 1.0);
    }
    textureStore(trailWrite, px, trail);

    // Probabilistic mound deposit (persistent material)
    let rnd = random2(vec2f(f32(gid.x) * 0.0137, t * 0.0031));
    let depositRate = select_channel(params.depositRates, agentType);
    if (rnd.x < depositRate) {
        var mound = textureLoad(moundRead, px, 0);
        if (agentType == 0) {
            mound.x = clamp(mound.x + deposit, 0.0, 1.0);
        } else if (agentType == 1) {
            mound.y = clamp(mound.y + deposit, 0.0, 1.0);
        } else if (agentType == 2) {
            mound.z = clamp(mound.z + deposit, 0.0, 1.0);
        } else {
            mound.w = clamp(mound.w + deposit, 0.0, 1.0);
        }
        textureStore(moundWrite, px, mound);
    }
}

// ---- Kernel 6: Render (composite trail + mound) ----
@compute @workgroup_size(8, 8)
fn render(@builtin(global_invocation_id) gid: vec3u) {
    let rez = get_rez();
    if (gid.x >= rez.x || gid.y >= rez.y) { return; }

    let trail = textureLoad(trailRead, gid.xy, 0);
    let mound = textureLoad(moundRead, gid.xy, 0);

    // Mound color — bright, persistent
    let m1 = hsb2rgb(vec3f(params.hues.x, params.saturations.x, mound.x), 1.0).rgb * mound.x;
    let m2 = hsb2rgb(vec3f(params.hues.y, params.saturations.y, mound.y), 1.0).rgb * mound.y;
    let m3 = hsb2rgb(vec3f(params.hues.z, params.saturations.z, mound.z), 1.0).rgb * mound.z;
    let m4 = hsb2rgb(vec3f(params.hues.w, params.saturations.w, mound.w), 1.0).rgb * mound.w;
    let moundColor = m1 + m2 + m3 + m4;

    // Trail color — faint overlay showing navigation pheromones
    let t1 = hsb2rgb(vec3f(params.hues.x, params.saturations.x * 0.5, trail.x), 1.0).rgb * trail.x * 0.3;
    let t2 = hsb2rgb(vec3f(params.hues.y, params.saturations.y * 0.5, trail.y), 1.0).rgb * trail.y * 0.3;
    let t3 = hsb2rgb(vec3f(params.hues.z, params.saturations.z * 0.5, trail.z), 1.0).rgb * trail.z * 0.3;
    let t4 = hsb2rgb(vec3f(params.hues.w, params.saturations.w * 0.5, trail.w), 1.0).rgb * trail.w * 0.3;
    let trailColor = t1 + t2 + t3 + t4;

    let combined = moundColor + trailColor;

    textureStore(outWrite, gid.xy, vec4f(combined, 1.0));
}
