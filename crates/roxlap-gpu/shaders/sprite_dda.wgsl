// GPU.9 — KV6 voxel sprite splatter, two-pass atomic design.
//
// Pass 1 `splat`: one thread per sprite voxel. Projects via
// cameras[0] (world camera; grid 0 = ground at identity), then for
// each covered screen pixel does a single `atomicMax` into the
// per-pixel key buffer. The key packs (nearness, voxel index) so the
// nearest voxel deterministically wins — no read-modify-write, no
// texture write, no inter-thread `textureStore` race (the cause of
// the white-pixel bleed the plain one-pass version had).
//
// Pass 2 `resolve`: one thread per screen pixel. Reads the winning
// voxel, world-occludes it against the scene depth buffer, then
// writes its colour once. Moving colour + occlusion here means the
// expensive per-pixel work happens once per pixel instead of once
// per overlapping splat.

const T_INF: f32 = 1.0e30;
// Depth-quantisation range for the packed key. Sprite voxels nearer
// than MAX_SPRITE_T quantise across 16 bits (~0.06 unit steps at
// 4096); farther ones saturate (their splats are sub-pixel anyway).
const MAX_SPRITE_T: f32 = 4096.0;

struct PerGridCamera {
    pos: vec3<f32>, _pad0: f32,
    right: vec3<f32>, _pad1: f32,
    down: vec3<f32>, _pad2: f32,
    forward: vec3<f32>, _pad3: f32,
};
struct Uniforms {
    fov_y_rad: f32, grid_count: u32, max_outer_steps: u32, _pad0: u32,
    screen_size: vec2<u32>, _pad1: vec2<u32>,
    cameras: array<PerGridCamera, 16>,
    fog_color: vec4<f32>, fog_far: f32,
    write_depth: u32, occ_page_words: u32, occ_num_pages: u32,
};
struct SpriteVoxel { world_pos: vec3<f32>, color: u32, world_size: f32, };

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> sprite_voxels: array<SpriteVoxel>;
// Per-pixel winner key: (nearness16 << 16) | (voxel_index + 1).
// 0 = empty. Cleared to 0 each frame; written by `splat`, read by
// `resolve`.
@group(0) @binding(2) var<storage, read_write> sprite_keys: array<atomic<u32>>;
// Scene world-t depth (read-only here) for sprite-vs-world occlusion.
@group(0) @binding(3) var<storage, read> depth_buffer: array<u32>;
@group(0) @binding(4) var output: texture_storage_2d<rgba8unorm, write>;

fn apply_fog(hit_color: vec3<f32>, t: f32) -> vec3<f32> {
    let fog_near = u.fog_color.w;
    let factor = smoothstep(fog_near, u.fog_far, t);
    return mix(hit_color, u.fog_color.rgb, factor);
}

// Project a sprite voxel to screen + report its splat extent.
// Returns false if the voxel is behind the camera. `ok` callers read
// sx_f/sy_f/half_ext/voxel_t.
struct Proj { ok: bool, cx_i: i32, cy_i: i32, half_ext: i32, voxel_t: f32 };
fn project(v: SpriteVoxel) -> Proj {
    var p: Proj;
    p.ok = false;
    let cam = u.cameras[0];
    let rel = v.world_pos - cam.pos;
    let z_cam = dot(rel, cam.forward);
    if (z_cam <= 0.1) { return p; }
    let x_cam = dot(rel, cam.right);
    let y_cam = dot(rel, cam.down);
    let aspect = f32(u.screen_size.x) / f32(u.screen_size.y);
    let half_h = tan(u.fov_y_rad * 0.5);
    let half_w = half_h * aspect;
    let cx = f32(u.screen_size.x) * 0.5;
    let cy = f32(u.screen_size.y) * 0.5;
    // `+` on y inverts scene_dda's ray gen exactly (see git history).
    let sx_f = cx + (x_cam / z_cam) * (cx / half_w);
    let sy_f = cy + (y_cam / z_cam) * (cy / half_h);
    let px_per_world = cx / half_w;
    let splat_px = px_per_world * v.world_size / z_cam;
    p.half_ext = i32(clamp(ceil(splat_px * 0.65), 0.0, 128.0));
    p.cx_i = i32(floor(sx_f));
    p.cy_i = i32(floor(sy_f));
    p.voxel_t = length(rel);
    p.ok = true;
    return p;
}

@compute @workgroup_size(64, 1, 1)
fn splat(@builtin(global_invocation_id) gid: vec3<u32>) {
    let voxel_count = arrayLength(&sprite_voxels);
    if (gid.x >= voxel_count) { return; }
    let p = project(sprite_voxels[gid.x]);
    if (!p.ok) { return; }

    // Pack nearness (larger = nearer) so atomicMax keeps the closest
    // voxel; ties break toward the higher index. +1 so index 0 at the
    // far plane never produces the empty key 0.
    let nearness = u32(clamp((MAX_SPRITE_T - p.voxel_t) / MAX_SPRITE_T * 65535.0, 0.0, 65535.0));
    let key = (nearness << 16u) | ((gid.x + 1u) & 0xffffu);

    // Clamp the splat square to its on-screen intersection.
    let w = i32(u.screen_size.x);
    let h = i32(u.screen_size.y);
    let y0 = max(p.cy_i - p.half_ext, 0);
    let y1 = min(p.cy_i + p.half_ext, h - 1);
    let x0 = max(p.cx_i - p.half_ext, 0);
    let x1 = min(p.cx_i + p.half_ext, w - 1);
    for (var py = y0; py <= y1; py = py + 1) {
        let row = u32(py) * u.screen_size.x;
        for (var px = x0; px <= x1; px = px + 1) {
            atomicMax(&sprite_keys[row + u32(px)], key);
        }
    }
}

@compute @workgroup_size(8, 8, 1)
fn resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.screen_size.x || gid.y >= u.screen_size.y) { return; }
    let pix_idx = gid.y * u.screen_size.x + gid.x;
    let key = atomicLoad(&sprite_keys[pix_idx]);
    if (key == 0u) { return; }
    let index = (key & 0xffffu) - 1u;
    let v = sprite_voxels[index];

    // World occlusion: skip if the scene surface is nearer.
    let cam = u.cameras[0];
    let rel = v.world_pos - cam.pos;
    let voxel_t = length(rel);
    let scene_t = bitcast<f32>(depth_buffer[pix_idx]);
    if (voxel_t >= scene_t) { return; }

    let packed = v.color;
    let a = f32((packed >> 24u) & 0xffu);
    let r = f32((packed >> 16u) & 0xffu);
    let g = f32((packed >> 8u) & 0xffu);
    let b = f32(packed & 0xffu);
    let brightness = a * (1.0 / 128.0);
    var col = vec3<f32>(r, g, b) * (brightness / 255.0);
    col = apply_fog(col, voxel_t);
    textureStore(output, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(col, 1.0));
}
