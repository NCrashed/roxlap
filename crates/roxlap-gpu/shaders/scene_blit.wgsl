// Scene blit — presents the compute pass's framebuffer STORAGE BUFFER
// (packed `rgba8unorm` per pixel, row stride = `dims.size.x`) into the
// swapchain via a fullscreen triangle. A buffer (not a storage texture)
// is used because Chrome's Dawn tiles write storage textures in a way
// the sampled read disagrees with; a linear buffer is unambiguous on
// every backend. Pixel-exact (nearest) by construction.

struct VsOut {
    @builtin(position) clip: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    // Single triangle covering NDC [-1, 3]²; clipped to the viewport.
    let x = f32((vid << 1u) & 2u) * 2.0 - 1.0;
    let y = 1.0 - f32(vid & 2u) * 2.0;
    var out: VsOut;
    out.clip = vec4<f32>(x, y, 0.0, 1.0);
    return out;
}

struct Dims {
    size: vec2<u32>,
    flip_x: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> fb: array<u32>;
@group(0) @binding(1) var<uniform> dims: Dims;

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // `in.clip.xy` is the framebuffer pixel centre (px + 0.5). The
    // compute pass wrote `fb[py * width + px]`, so read the same texel —
    // mirrored to `width-1-px` when the horizontal flip is on.
    var px = u32(in.clip.x);
    let py = u32(in.clip.y);
    if dims.flip_x != 0u {
        px = dims.size.x - 1u - px;
    }
    let idx = py * dims.size.x + px;
    return unpack4x8unorm(fb[idx]);
}
