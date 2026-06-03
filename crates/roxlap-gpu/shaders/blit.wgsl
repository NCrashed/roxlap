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
@group(0) @binding(1) var samp: sampler;

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return textureSample(src, samp, in.uv);
}
