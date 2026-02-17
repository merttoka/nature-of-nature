struct Params {
    width: u32,
    height: u32,
    brightness: f32,
    contrast: f32,
    bloomThreshold: f32,
    bloomIntensity: f32,
    bloomRadius: f32,
    saturationPost: f32,
    vignette: f32,
    useLut: u32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var inputTex: texture_2d<f32>;
@group(0) @binding(2) var secondaryTex: texture_2d<f32>;
@group(0) @binding(3) var outputTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var lutSampler: sampler;
@group(0) @binding(5) var lutTex: texture_2d<f32>;

fn gaussian(x: f32, sigma: f32) -> f32 {
    return exp(-(x * x) / (2.0 * sigma * sigma));
}

// Pass 1: Horizontal bloom blur — extract bright pixels and blur horizontally
@compute @workgroup_size(8, 8)
fn bloom_h(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }

    let sigma = params.bloomRadius;
    let radius = i32(ceil(sigma * 2.5));

    var total = vec3f(0.0);
    var weightSum = 0.0;

    for (var dx = -radius; dx <= radius; dx++) {
        let sx = clamp(i32(gid.x) + dx, 0, i32(params.width) - 1);
        let pixel = textureLoad(inputTex, vec2u(u32(sx), gid.y), 0).rgb;

        // Extract bright parts
        let lum = dot(pixel, vec3f(0.2126, 0.7152, 0.0722));
        let bright = pixel * smoothstep(params.bloomThreshold, params.bloomThreshold + 0.2, lum);

        let w = gaussian(f32(dx), sigma);
        total += bright * w;
        weightSum += w;
    }

    textureStore(outputTex, gid.xy, vec4f(total / weightSum, 1.0));
}

// Pass 2: Vertical bloom blur
@compute @workgroup_size(8, 8)
fn bloom_v(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }

    let sigma = params.bloomRadius;
    let radius = i32(ceil(sigma * 2.5));

    var total = vec3f(0.0);
    var weightSum = 0.0;

    for (var dy = -radius; dy <= radius; dy++) {
        let sy = clamp(i32(gid.y) + dy, 0, i32(params.height) - 1);
        let pixel = textureLoad(inputTex, vec2u(gid.x, u32(sy)), 0).rgb;

        let w = gaussian(f32(dy), sigma);
        total += pixel * w;
        weightSum += w;
    }

    textureStore(outputTex, gid.xy, vec4f(total / weightSum, 1.0));
}

// Pass 3: Composite — combine original with bloom, apply brightness/contrast/saturation/vignette
@compute @workgroup_size(8, 8)
fn composite(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }

    var color = textureLoad(inputTex, gid.xy, 0).rgb;
    let bloom = textureLoad(secondaryTex, gid.xy, 0).rgb;

    // Add bloom
    color = color + bloom * params.bloomIntensity;

    // Brightness (multiplicative — preserves black)
    color = color * (1.0 + params.brightness);

    // Contrast (around 0.5 midpoint)
    color = (color - 0.5) * params.contrast + 0.5;

    // Saturation
    let gray = dot(color, vec3f(0.2126, 0.7152, 0.0722));
    color = mix(vec3f(gray), color, params.saturationPost);

    // Vignette
    let uv = vec2f(f32(gid.x) / f32(params.width), f32(gid.y) / f32(params.height));
    let dist = distance(uv, vec2f(0.5));
    let vig = 1.0 - smoothstep(0.3, 0.9, dist) * params.vignette;
    color = color * vig;

    color = clamp(color, vec3f(0.0), vec3f(1.0));

    // Colormap LUT
    if (params.useLut == 1u) {
        let luma = dot(color, vec3f(0.2126, 0.7152, 0.0722));
        color = textureSampleLevel(lutTex, lutSampler, vec2f(luma, 0.5), 0.0).rgb;
    }

    textureStore(outputTex, gid.xy, vec4f(color, 1.0));
}
