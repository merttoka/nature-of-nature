struct Params {
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var inputTex: texture_2d<f32>;
@group(0) @binding(2) var inputSampler: sampler;
@group(0) @binding(3) var outputTex: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= params.dst_w || gid.y >= params.dst_h) { return; }

    let uv = (vec2f(f32(gid.x), f32(gid.y)) + 0.5) / vec2f(f32(params.dst_w), f32(params.dst_h));
    let color = textureSampleLevel(inputTex, inputSampler, uv, 0.0);
    textureStore(outputTex, gid.xy, color);
}
