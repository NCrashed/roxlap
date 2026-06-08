// GPU.10.0 — single KV6 sprite as a DDA-marched voxel model.
//
// One thread per screen pixel: build the world ray (same convention
// as scene_dda), transform it into the model's local voxel space,
// clip to the model AABB, and 3D-DDA-march the volume to the first
// solid voxel. Composite against the terrain depth buffer (the scene
// pass's per-pixel best_t) so the sprite occludes / is occluded by
// the world correctly. No overdraw, no atomics — the precise path.

const T_INF: f32 = 1.0e30;

struct ModelUniform {
    cam_pos: vec3<f32>, _p0: f32,
    cam_right: vec3<f32>, _p1: f32,
    cam_down: vec3<f32>, _p2: f32,
    cam_forward: vec3<f32>, _p3: f32,
    // Inverse model→world rotation, columns (w unused).
    inv_rot0: vec4<f32>,
    inv_rot1: vec4<f32>,
    inv_rot2: vec4<f32>,
    inst_pos: vec3<f32>, _p4: f32,
    pivot: vec3<f32>, _p5: f32,
    fog_color: vec4<f32>, // rgb + fog_near in w
    screen_size: vec2<u32>,
    dims: vec2<u32>, // mx, my
    mz: u32,
    occ_words_per_col: u32,
    fog_far: f32,
    fov_y_rad: f32,
};

@group(0) @binding(0) var<uniform> u: ModelUniform;
@group(0) @binding(1) var<storage, read> occupancy: array<u32>;
@group(0) @binding(2) var<storage, read> colors: array<u32>;
@group(0) @binding(3) var<storage, read> color_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> depth_buffer: array<u32>;
@group(0) @binding(5) var output: texture_storage_2d<rgba8unorm, write>;

fn apply_inv(v: vec3<f32>) -> vec3<f32> {
    return u.inv_rot0.xyz * v.x + u.inv_rot1.xyz * v.y + u.inv_rot2.xyz * v.z;
}

fn apply_fog(hit_color: vec3<f32>, t: f32) -> vec3<f32> {
    let fog_near = u.fog_color.w;
    let factor = smoothstep(fog_near, u.fog_far, t);
    return mix(hit_color, u.fog_color.rgb, factor);
}

fn col_base(p: vec3<i32>) -> u32 {
    return (u32(p.x) + u32(p.y) * u.dims.x) * u.occ_words_per_col;
}

fn model_solid(p: vec3<i32>) -> bool {
    let base = col_base(p);
    let zw = u32(p.z) >> 5u;
    let zb = u32(p.z) & 31u;
    return (occupancy[base + zw] & (1u << zb)) != 0u;
}

fn model_color(p: vec3<i32>) -> vec3<f32> {
    let col = u32(p.x) + u32(p.y) * u.dims.x;
    let base = col * u.occ_words_per_col;
    let zw = u32(p.z) >> 5u;
    let zb = u32(p.z) & 31u;
    var rank: u32 = 0u;
    for (var w: u32 = 0u; w < zw; w = w + 1u) {
        rank = rank + countOneBits(occupancy[base + w]);
    }
    var mask: u32 = 0u;
    if (zb > 0u) { mask = (1u << zb) - 1u; }
    rank = rank + countOneBits(occupancy[base + zw] & mask);

    let packed = colors[color_offsets[col] + rank];
    let a = f32((packed >> 24u) & 0xffu);
    let r = f32((packed >> 16u) & 0xffu);
    let g = f32((packed >> 8u) & 0xffu);
    let b = f32(packed & 0xffu);
    let brightness = a * (1.0 / 128.0);
    return vec3<f32>(r, g, b) * (brightness / 255.0);
}

fn shield_parallel(t: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    var o = t;
    if (dir.x == 0.0) { o.x = T_INF; }
    if (dir.y == 0.0) { o.y = T_INF; }
    if (dir.z == 0.0) { o.z = T_INF; }
    return o;
}

@compute @workgroup_size(8, 8, 1)
fn march(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.screen_size.x || gid.y >= u.screen_size.y) { return; }
    let pix = gid.y * u.screen_size.x + gid.x;

    // World ray (matches scene_dda's ray construction exactly).
    let aspect = f32(u.screen_size.x) / f32(u.screen_size.y);
    let half_h = tan(u.fov_y_rad * 0.5);
    let half_w = half_h * aspect;
    let ndc_x = (f32(gid.x) + 0.5) / f32(u.screen_size.x) * 2.0 - 1.0;
    let ndc_y_top = 1.0 - (f32(gid.y) + 0.5) / f32(u.screen_size.y) * 2.0;
    let ray_dir = normalize(
        u.cam_forward + ndc_x * half_w * u.cam_right - ndc_y_top * half_h * u.cam_down
    );

    // World → model-local. For an orthonormal basis `apply_inv`
    // preserves length, so the local ray parameter t equals the
    // world-space distance and composites directly against the scene
    // depth buffer.
    let o = apply_inv(u.cam_pos - u.inst_pos) + u.pivot;
    let d = apply_inv(ray_dir);

    // Slab-clip the local ray to the model AABB [0,dims].
    let box_max = vec3<f32>(f32(u.dims.x), f32(u.dims.y), f32(u.mz));
    let inv_d = 1.0 / d;
    let t0 = (vec3<f32>(0.0) - o) * inv_d;
    let t1 = (box_max - o) * inv_d;
    let tlo = min(t0, t1);
    let thi = max(t0, t1);
    let t_enter = max(max(tlo.x, tlo.y), max(tlo.z, 0.0));
    let t_exit = min(thi.x, min(thi.y, thi.z));
    if (t_exit < t_enter) { return; } // misses the box

    // Only worth marching if the sprite could be nearer than the world.
    let scene_t = bitcast<f32>(depth_buffer[pix]);
    if (t_enter >= scene_t) { return; }

    // 3D-DDA from the entry point.
    let entry = o + t_enter * d;
    let dim_i = vec3<i32>(i32(u.dims.x), i32(u.dims.y), i32(u.mz));
    var p = clamp(vec3<i32>(floor(entry)), vec3<i32>(0), dim_i - vec3<i32>(1));
    let step = vec3<i32>(sign(d));
    let t_delta = abs(inv_d);
    let next_b = vec3<f32>(
        select(f32(p.x), f32(p.x + 1), step.x > 0),
        select(f32(p.y), f32(p.y + 1), step.y > 0),
        select(f32(p.z), f32(p.z + 1), step.z > 0),
    );
    var t_max = shield_parallel((next_b - o) * inv_d, d);
    var t_hit = t_enter;

    // Bound steps to the volume's Manhattan diagonal.
    let max_steps = u.dims.x + u.dims.y + u.mz + 3u;
    for (var i: u32 = 0u; i < max_steps; i = i + 1u) {
        if (model_solid(p)) {
            if (t_hit < scene_t) {
                let col = apply_fog(model_color(p), t_hit);
                textureStore(output, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(col, 1.0));
            }
            return;
        }
        if (t_max.x < t_max.y && t_max.x < t_max.z) {
            t_hit = t_max.x;
            p.x = p.x + step.x;
            t_max.x = t_max.x + t_delta.x;
            if (p.x < 0 || p.x >= dim_i.x) { return; }
        } else if (t_max.y < t_max.z) {
            t_hit = t_max.y;
            p.y = p.y + step.y;
            t_max.y = t_max.y + t_delta.y;
            if (p.y < 0 || p.y >= dim_i.y) { return; }
        } else {
            t_hit = t_max.z;
            p.z = p.z + step.z;
            t_max.z = t_max.z + t_delta.z;
            if (p.z < 0 || p.z >= dim_i.z) { return; }
        }
        if (t_hit >= scene_t) { return; } // world already nearer
    }
}
