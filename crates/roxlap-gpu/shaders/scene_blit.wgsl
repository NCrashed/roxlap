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

// RP.0 — the framebuffer is the **render** (logical) size `src`; this pass
// renders to the swapchain `dst` and nearest-upscales. Native (`src == dst`)
// maps every dst pixel to itself (integer math ⇒ byte-identical to pre-RP).
struct Dims {
    src_size: vec2<u32>,   // render (compute) framebuffer size
    dst_size: vec2<u32>,   // swapchain size
    flip_x: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> fb: array<u32>;
@group(0) @binding(1) var<uniform> dims: Dims;

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // `in.clip.xy` is the swapchain pixel centre (px + 0.5). Map it to the
    // render-grid texel with integer nearest sampling (hard pixels).
    let dx = u32(in.clip.x);
    let dy = u32(in.clip.y);
    var sx = (dx * dims.src_size.x) / dims.dst_size.x;
    var sy = (dy * dims.src_size.y) / dims.dst_size.y;
    sx = min(sx, dims.src_size.x - 1u);
    sy = min(sy, dims.src_size.y - 1u);
    // Mirror to `src_w-1-sx` when the horizontal flip is on (the compute
    // pass wrote the framebuffer unflipped).
    if dims.flip_x != 0u {
        sx = dims.src_size.x - 1u - sx;
    }
    let idx = sy * dims.src_size.x + sx;
    return unpack4x8unorm(fb[idx]);
}
