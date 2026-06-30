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
    // RP.2 — per-channel posterize level counts (<=1 ⇒ untouched) + dither
    // mode (0 none, 1 Bayer 4x4, 2 blue-noise/IGN).
    levels_r: u32,
    levels_g: u32,
    levels_b: u32,
    dither: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> r: Resolve;

// RP.2 — dither threshold in [0,1) for logical pixel (x,y). None ⇒ 0.5 so
// floor(scaled + 0.5) is round-to-nearest.
fn dither_offset(mode: u32, x: u32, y: u32) -> f32 {
    if mode == 1u {
        var bayer = array<f32, 16>(
            0.0, 8.0, 2.0, 10.0,
            12.0, 4.0, 14.0, 6.0,
            3.0, 11.0, 1.0, 9.0,
            15.0, 7.0, 13.0, 5.0,
        );
        let idx = (y % 4u) * 4u + (x % 4u);
        return (bayer[idx] + 0.5) / 16.0;
    }
    if mode == 2u {
        // Jimenez interleaved-gradient noise.
        let f = f32(x) * 0.06711056 + f32(y) * 0.00583715;
        return fract(52.9829189 * fract(f));
    }
    return 0.5;
}

// RP.2 — quantize channel `c` (in [0,1]) to `levels` steps with dither `off`.
fn quantize(c: f32, levels: u32, off: f32) -> f32 {
    if levels <= 1u {
        return c;
    }
    let m = f32(levels - 1u);
    let q = clamp(floor(c * m + off), 0.0, m);
    return q / m;
}

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
    var color = acc / f32(s * s);
    // RP.2 — posterize + dither at the logical resolution (one quantize per
    // hard pixel). All-`1` levels ⇒ untouched (the RP.1 box-avg only).
    let off = dither_offset(r.dither, gid.x, gid.y);
    color = vec3<f32>(
        quantize(color.r, r.levels_r, off),
        quantize(color.g, r.levels_g, off),
        quantize(color.b, r.levels_b, off),
    );
    dst[gid.y * r.dst_size.x + gid.x] = pack4x8unorm(vec4<f32>(color, 1.0));
}
