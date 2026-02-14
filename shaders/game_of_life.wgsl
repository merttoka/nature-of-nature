@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(input_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let x = i32(gid.x);
    let y = i32(gid.y);
    let w = i32(dims.x);
    let h = i32(dims.y);

    // Count live neighbors (toroidal wrap)
    var neighbors: i32 = 0;
    for (var dy: i32 = -1; dy <= 1; dy++) {
        for (var dx: i32 = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) { continue; }
            let nx = (x + dx + w) % w;
            let ny = (y + dy + h) % h;
            let cell = textureLoad(input_tex, vec2i(nx, ny), 0);
            if (cell.r > 0.5) {
                neighbors++;
            }
        }
    }

    let current = textureLoad(input_tex, vec2i(x, y), 0);
    let alive = current.r > 0.5;

    var next_alive = false;
    if (alive) {
        next_alive = (neighbors == 2 || neighbors == 3);
    } else {
        next_alive = (neighbors == 3);
    }

    let v = select(0.0, 1.0, next_alive);
    textureStore(output_tex, vec2i(x, y), vec4f(v, v, v, 1.0));
}
