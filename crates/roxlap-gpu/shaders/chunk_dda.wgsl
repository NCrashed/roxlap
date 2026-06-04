// GPU.3 — single-chunk Amanatides–Woo voxel marcher.
//
// One thread per output pixel: build a ray from the camera basis,
// step voxel-by-voxel in chunk-local coordinates, look up colour
// via per-column rank-count on hit, write to a storage texture.
//
// Camera convention (matches roxlap-core's `Camera`):
//   * Z is DOWN (voxlap). `camera_down` is the +z direction in
//     world space; `camera_forward` and `camera_right` complete a
//     right-handed orthonormal basis.
//   * Pixel (0, 0) is top-left. Rays from the top half of the
//     screen point UP (negative camera_down).
//
// Chunk-local space: voxels at integer coords (x, y, z), x ∈
// [0, vsid), y ∈ [0, vsid), z ∈ [0, CHUNK_Z=256). The host
// translates the world-space camera into chunk-local before
// passing it here.

const OCC_WORDS_PER_COLUMN: u32 = 8u; // CHUNK_Z (256) / 32
const CHUNK_Z: u32 = 256u;

struct Uniforms {
    camera_pos: vec3<f32>,
    _pad0: f32,
    camera_right: vec3<f32>,
    _pad1: f32,
    camera_down: vec3<f32>,
    _pad2: f32,
    camera_forward: vec3<f32>,
    fov_y_rad: f32,
    screen_size: vec2<u32>,
    vsid: u32,
    max_scan_dist: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> occupancy: array<u32>;
@group(0) @binding(2) var<storage, read> color_offsets: array<u32>;
@group(0) @binding(3) var<storage, read> colors: array<u32>;
@group(0) @binding(4) var output: texture_storage_2d<rgba8unorm, write>;

fn voxel_solid(p: vec3<i32>) -> bool {
    if (p.x < 0 || p.y < 0 || p.z < 0 ||
        u32(p.x) >= u.vsid || u32(p.y) >= u.vsid || u32(p.z) >= CHUNK_Z) {
        return false;
    }
    let col_idx = u32(p.x) + u32(p.y) * u.vsid;
    let col_word_base = col_idx * OCC_WORDS_PER_COLUMN;
    let z_word = u32(p.z) >> 5u;
    let z_bit = u32(p.z) & 31u;
    return (occupancy[col_word_base + z_word] & (1u << z_bit)) != 0u;
}

fn voxel_color(p: vec3<i32>) -> vec3<f32> {
    let col_idx = u32(p.x) + u32(p.y) * u.vsid;
    let col_word_base = col_idx * OCC_WORDS_PER_COLUMN;
    let z_word = u32(p.z) >> 5u;
    let z_bit = u32(p.z) & 31u;
    var rank: u32 = 0u;
    for (var w: u32 = 0u; w < z_word; w = w + 1u) {
        rank = rank + countOneBits(occupancy[col_word_base + w]);
    }
    var mask: u32 = 0u;
    if (z_bit > 0u) {
        mask = (1u << z_bit) - 1u;
    }
    rank = rank + countOneBits(occupancy[col_word_base + z_word] & mask);
    let packed = colors[color_offsets[col_idx] + rank];
    // Voxlap colour layout = 0xAARRGGBB (little-endian u32 from BGRA
    // bytes). The A byte is voxlap's "brightness" multiplier baked
    // by lightmode-1: 0x80 (=128) is neutral; CPU rasterizer mirrors
    // `(rgb * A) >> 7` per channel. Apply it here so GPU-rendered
    // chunks match the CPU's baked shading.
    let a = f32((packed >> 24u) & 0xffu);
    let r = f32((packed >> 16u) & 0xffu);
    let g = f32((packed >> 8u) & 0xffu);
    let b = f32(packed & 0xffu);
    let brightness = a * (1.0 / 128.0);
    return vec3<f32>(r, g, b) * (brightness / 255.0);
}

// Voxlap-style two-band sky: blue zenith, lighter horizon. dir.z is
// positive going down (into the world), so dir.z = -1 is straight
// up = zenith.
fn sky_color(dir: vec3<f32>) -> vec3<f32> {
    let down_amount = clamp(dir.z * 0.5 + 0.5, 0.0, 1.0);
    let zenith = vec3<f32>(0.18, 0.28, 0.55);
    let horizon = vec3<f32>(0.66, 0.74, 0.88);
    return mix(zenith, horizon, down_amount);
}

@compute @workgroup_size(8, 8)
fn render_chunk(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.screen_size.x || gid.y >= u.screen_size.y) {
        return;
    }

    // Per-pixel ray direction. Pixel (0, 0) is top-left; the top
    // half maps to -camera_down (rays point UP in voxlap coords).
    let aspect = f32(u.screen_size.x) / f32(u.screen_size.y);
    let half_h = tan(u.fov_y_rad * 0.5);
    let half_w = half_h * aspect;
    let ndc_x = (f32(gid.x) + 0.5) / f32(u.screen_size.x) * 2.0 - 1.0;
    let ndc_y_top_pos = 1.0 - (f32(gid.y) + 0.5) / f32(u.screen_size.y) * 2.0;
    let dir = normalize(
        u.camera_forward
        + ndc_x * half_w * u.camera_right
        - ndc_y_top_pos * half_h * u.camera_down
    );

    // Amanatides–Woo 3D DDA.
    var p = vec3<i32>(floor(u.camera_pos));
    let step = vec3<i32>(sign(dir));
    let t_delta = abs(1.0 / dir);
    let next_boundary = vec3<f32>(
        select(f32(p.x), f32(p.x + 1), step.x > 0),
        select(f32(p.y), f32(p.y + 1), step.y > 0),
        select(f32(p.z), f32(p.z + 1), step.z > 0),
    );
    var t_max = (next_boundary - u.camera_pos) / dir;
    if (dir.x == 0.0) { t_max.x = 1.0e30; }
    if (dir.y == 0.0) { t_max.y = 1.0e30; }
    if (dir.z == 0.0) { t_max.z = 1.0e30; }

    var hit_color = sky_color(dir);

    // Voxlap's lightmode-1 bake already encodes face direction +
    // sun angle into each voxel's alpha byte (consumed in
    // `voxel_color`), so the marcher doesn't apply additional
    // per-face shading — that would double-darken and produce the
    // "flat / pastel" mismatch the user flagged.
    for (var i = 0u; i < u.max_scan_dist; i = i + 1u) {
        if (voxel_solid(p)) {
            hit_color = voxel_color(p);
            break;
        }
        if (t_max.x < t_max.y && t_max.x < t_max.z) {
            p.x = p.x + step.x;
            t_max.x = t_max.x + t_delta.x;
        } else if (t_max.y < t_max.z) {
            p.y = p.y + step.y;
            t_max.y = t_max.y + t_delta.y;
        } else {
            p.z = p.z + step.z;
            t_max.z = t_max.z + t_delta.z;
        }
    }

    textureStore(output, vec2<i32>(gid.xy), vec4<f32>(hit_color, 1.0));
}
