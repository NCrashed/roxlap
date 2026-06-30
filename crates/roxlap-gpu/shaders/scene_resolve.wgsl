// roxlap-gpu scene_resolve.wgsl — RP.1 SSAA resolve.
//
// Box-downfilter the march-resolution scene framebuffer (`src`, sized
// `logical × ssaa`, packed `rgba8unorm`) into the logical-resolution
// `dst` buffer: one thread per *logical* pixel averages its `ssaa × ssaa`
// march block. `ssaa == 1` is an exact 1×1 copy (unpack→repack with the
// scene's alpha=1 round-trips byte-for-byte), so the blit reads `dst`
// unconditionally. Posterize + dither (RP.2) will hook in here, at the
// logical resolution, so each hard pixel quantizes once.

struct Resolve {
    src_size: vec2<u32>,   // march framebuffer size (logical * ssaa)
    dst_size: vec2<u32>,   // logical size
    ssaa: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> r: Resolve;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= r.dst_size.x || gid.y >= r.dst_size.y {
        return;
    }
    let s = max(r.ssaa, 1u);
    var acc = vec3<f32>(0.0, 0.0, 0.0);
    for (var j: u32 = 0u; j < s; j = j + 1u) {
        let sy = gid.y * s + j;
        let row = sy * r.src_size.x;
        for (var i: u32 = 0u; i < s; i = i + 1u) {
            let sx = gid.x * s + i;
            acc = acc + unpack4x8unorm(src[row + sx]).rgb;
        }
    }
    let inv = 1.0 / f32(s * s);
    dst[gid.y * r.dst_size.x + gid.x] = pack4x8unorm(vec4<f32>(acc * inv, 1.0));
}
