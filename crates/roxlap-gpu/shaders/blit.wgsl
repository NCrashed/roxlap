// GPU.0 probe blit — upscales the low-res compute output into the
// swapchain via a fullscreen triangle. Nearest-neighbour sampling
// preserves the chunky pixel aesthetic.

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    // Single triangle covering NDC [-1, 3] × [-1, 3]; clipped to the
    // viewport. vid ∈ {0, 1, 2}.
    let x = f32((vid << 1u) & 2u) * 2.0 - 1.0;
    let y = 1.0 - f32(vid & 2u) * 2.0;
    var out: VsOut;
    out.clip = vec4<f32>(x, y, 0.0, 1.0);
    // Map NDC to UV; flip y so (0,0) is top-left like the compute
    // shader's pixel coordinates.
    out.uv = vec2<f32>((x + 1.0) * 0.5, 1.0 - (y + 1.0) * 0.5);
    return out;
}

@group(0) @binding(0) var src: texture_2d<f32>;
// Kept for bind-group/layout compatibility; the pixel-exact `textureLoad`
// path below doesn't sample, so this sampler is unused.
@group(0) @binding(1) var samp: sampler;

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Pixel-exact fetch instead of `textureSample`. Normalised-UV
    // sampling of the compute pass's storage texture mis-strided on
    // some WebGPU/Dawn drivers (the output tiled across the screen);
    // resolving the source texel by `uv * dimensions` + `textureLoad`
    // reads the logical texel directly, with no sampler/row-stride
    // ambiguity. Nearest-neighbour by construction — preserves the
    // chunky retro look and still upscales a lower-res source (the
    // GPU.0 probe) correctly.
    let dims = vec2<f32>(textureDimensions(src));
    let texel = vec2<i32>(clamp(in.uv, vec2<f32>(0.0), vec2<f32>(0.999999)) * dims);
    return textureLoad(src, texel, 0);
}
