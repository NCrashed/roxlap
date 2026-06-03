// GPU.0 probe — single-chunk Amanatides–Woo DDA marcher.
//
// One thread per output pixel: build a world-space ray from the
// camera, march voxel-by-voxel, write the hit colour into a storage
// texture. The companion `blit.wgsl` upscales the storage texture to
// the swapchain in a fullscreen-triangle pass.

struct Uniforms {
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad0: f32,
    screen_size: vec2<u32>,
    chunk_size: u32,
    max_scan_dist: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> occupancy: array<u32>;
@group(0) @binding(2) var<storage, read> colors: array<u32>;
@group(0) @binding(3) var output: texture_storage_2d<rgba8unorm, write>;

fn linear_index(p: vec3<i32>) -> u32 {
    return u32(p.x) + u32(p.y) * u.chunk_size + u32(p.z) * u.chunk_size * u.chunk_size;
}

fn in_bounds(p: vec3<i32>) -> bool {
    let s = i32(u.chunk_size);
    return p.x >= 0 && p.y >= 0 && p.z >= 0 && p.x < s && p.y < s && p.z < s;
}

fn is_solid(p: vec3<i32>) -> bool {
    if (!in_bounds(p)) {
        return false;
    }
    let idx = linear_index(p);
    let word = occupancy[idx >> 5u];
    return (word & (1u << (idx & 31u))) != 0u;
}

fn voxel_color(p: vec3<i32>) -> vec3<f32> {
    let packed = colors[linear_index(p)];
    return vec3<f32>(
        f32((packed >> 16u) & 0xffu) / 255.0,
        f32((packed >> 8u) & 0xffu) / 255.0,
        f32(packed & 0xffu) / 255.0,
    );
}

// Voxlap-style two-band sky: lighter near the horizon, darker
// overhead. Keeps the retro-aesthetic the GPU stage is preserving.
fn sky_color(dir: vec3<f32>) -> vec3<f32> {
    let t = clamp(dir.z * 0.5 + 0.5, 0.0, 1.0);
    let horizon = vec3<f32>(0.66, 0.74, 0.88);
    let zenith = vec3<f32>(0.18, 0.28, 0.55);
    return mix(horizon, zenith, t);
}

@compute @workgroup_size(8, 8)
fn render_frame(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.screen_size.x || gid.y >= u.screen_size.y) {
        return;
    }

    // Reconstruct world-space ray from the inverse view-projection.
    let ndc = vec2<f32>(
        (f32(gid.x) + 0.5) / f32(u.screen_size.x) * 2.0 - 1.0,
        1.0 - (f32(gid.y) + 0.5) / f32(u.screen_size.y) * 2.0,
    );
    let far_h = u.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    let far_w = far_h.xyz / far_h.w;
    let dir = normalize(far_w - u.camera_pos);

    // Amanatides–Woo 3D DDA with one cell step per iteration.
    var p = vec3<i32>(floor(u.camera_pos));
    let step = vec3<i32>(sign(dir));
    let t_delta = abs(1.0 / dir);

    // For each axis, distance along the ray to the next integer
    // boundary. If the ray is exactly parallel to an axis the
    // corresponding `t_max` stays at +inf so that axis never wins
    // the per-iteration min.
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
    var face_shade: f32 = 1.0;

    for (var i = 0u; i < u.max_scan_dist; i = i + 1u) {
        if (is_solid(p)) {
            hit_color = voxel_color(p) * face_shade;
            break;
        }
        // Step the smallest t_max axis; remember which face we
        // crossed for a cheap flat shading tint.
        if (t_max.x < t_max.y && t_max.x < t_max.z) {
            p.x = p.x + step.x;
            t_max.x = t_max.x + t_delta.x;
            face_shade = 0.78;
        } else if (t_max.y < t_max.z) {
            p.y = p.y + step.y;
            t_max.y = t_max.y + t_delta.y;
            face_shade = 0.88;
        } else {
            p.z = p.z + step.z;
            t_max.z = t_max.z + t_delta.z;
            face_shade = select(0.6, 1.0, step.z < 0); // top brighter than bottom
        }
    }

    textureStore(output, vec2<i32>(gid.xy), vec4<f32>(hit_color, 1.0));
}
