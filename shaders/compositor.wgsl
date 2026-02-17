struct Params {
    width: u32,
    height: u32,
    blendMode: u32,
    opacity: f32,
    isFirstLayer: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var layerTex: texture_2d<f32>;
@group(0) @binding(2) var accumTex: texture_2d<f32>;
@group(0) @binding(3) var outputTex: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn blend(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }

    let coord = vec2i(gid.xy);
    let layer = textureLoad(layerTex, coord, 0);
    let layerRGB = layer.rgb * params.opacity;

    if (params.isFirstLayer == 1u) {
        textureStore(outputTex, coord, vec4f(layerRGB, 1.0));
        return;
    }

    let accum = textureLoad(accumTex, coord, 0);

    var result: vec3f;
    switch (params.blendMode) {
        // Additive
        case 0u: {
            result = accum.rgb + layerRGB;
        }
        // Multiply
        case 1u: {
            result = accum.rgb * mix(vec3f(1.0), layer.rgb, params.opacity);
        }
        // Screen
        case 2u: {
            let screened = vec3f(1.0) - (vec3f(1.0) - accum.rgb) * (vec3f(1.0) - layer.rgb);
            result = mix(accum.rgb, screened, params.opacity);
        }
        // Normal (alpha over)
        default: {
            result = mix(accum.rgb, layer.rgb, params.opacity);
        }
    }

    textureStore(outputTex, coord, vec4f(clamp(result, vec3f(0.0), vec3f(1.0)), 1.0));
}
