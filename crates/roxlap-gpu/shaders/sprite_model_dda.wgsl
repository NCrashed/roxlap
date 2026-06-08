// GPU.10.1 — instanced KV6 sprites as DDA-marched voxel models.
//
// One thread per screen pixel. Build the world ray, then loop every
// instance (naive — frustum cull + screen-tile binning come in 10.2 /
// 10.3): transform the ray into that instance's model-local space,
// AABB-clip to the model box, 3D-DDA-march to the first solid voxel,
// and keep the nearest hit across all instances. Composite against the
// terrain depth buffer so the world occludes / is occluded correctly.
// Precise, no overdraw, no atomics.

const T_INF: f32 = 1.0e30;

struct ModelMeta {
    occupancy_offset: u32,
    colors_offset: u32,
    color_offsets_offset: u32,
    occ_words_per_col: u32,
    dims: vec3<u32>,
    _pad0: u32,
    pivot: vec3<f32>,
    _pad1: f32,
};
struct Instance {
    inv_rot0: vec4<f32>,
    inv_rot1: vec4<f32>,
    inv_rot2: vec4<f32>,
    pos: vec3<f32>,
    model_id: u32,
};
struct Uniform {
    cam_pos: vec3<f32>, _p0: f32,
    cam_right: vec3<f32>, _p1: f32,
    cam_down: vec3<f32>, _p2: f32,
    cam_forward: vec3<f32>, _p3: f32,
    fog_color: vec4<f32>, // rgb + fog_near in w
    screen_size: vec2<u32>,
    instance_count: u32,
    fog_far: f32,
    fov_y_rad: f32,
    _p4: f32, _p5: f32, _p6: f32,
};

@group(0) @binding(0) var<uniform> u: Uniform;
@group(0) @binding(1) var<storage, read> occupancy: array<u32>;
@group(0) @binding(2) var<storage, read> colors: array<u32>;
@group(0) @binding(3) var<storage, read> color_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> models: array<ModelMeta>;
@group(0) @binding(5) var<storage, read> instances: array<Instance>;
@group(0) @binding(6) var<storage, read> depth_buffer: array<u32>;
@group(0) @binding(7) var output: texture_storage_2d<rgba8unorm, write>;

fn apply_fog(hit_color: vec3<f32>, t: f32) -> vec3<f32> {
    let fog_near = u.fog_color.w;
    let factor = smoothstep(fog_near, u.fog_far, t);
    return mix(hit_color, u.fog_color.rgb, factor);
}

fn model_solid(m: ModelMeta, p: vec3<i32>) -> bool {
    let col = u32(p.x) + u32(p.y) * m.dims.x;
    let base = m.occupancy_offset + col * m.occ_words_per_col;
    let zw = u32(p.z) >> 5u;
    let zb = u32(p.z) & 31u;
    return (occupancy[base + zw] & (1u << zb)) != 0u;
}

fn model_color(m: ModelMeta, p: vec3<i32>) -> vec3<f32> {
    let col = u32(p.x) + u32(p.y) * m.dims.x;
    let base = m.occupancy_offset + col * m.occ_words_per_col;
    let zw = u32(p.z) >> 5u;
    let zb = u32(p.z) & 31u;
    var rank: u32 = 0u;
    for (var w: u32 = 0u; w < zw; w = w + 1u) {
        rank = rank + countOneBits(occupancy[base + w]);
    }
    var mask: u32 = 0u;
    if (zb > 0u) { mask = (1u << zb) - 1u; }
    rank = rank + countOneBits(occupancy[base + zw] & mask);

    let local_off = color_offsets[m.color_offsets_offset + col];
    let packed = colors[m.colors_offset + local_off + rank];
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

// March one instance; returns the hit t (or `limit` on miss) and
// writes the colour into `out_color` when it improves on `limit`.
struct Hit { t: f32, color: vec3<f32>, hit: bool };
fn march_instance(inst: Instance, ray_dir: vec3<f32>, limit: f32) -> Hit {
    var res: Hit;
    res.hit = false;
    res.t = limit;
    res.color = vec3<f32>(0.0);

    let m = models[inst.model_id];
    let inv = mat3x3<f32>(inst.inv_rot0.xyz, inst.inv_rot1.xyz, inst.inv_rot2.xyz);
    // World → model-local. For an orthonormal basis this preserves
    // length, so the local ray parameter equals world distance.
    let o = inv * (u.cam_pos - inst.pos) + m.pivot;
    let d = inv * ray_dir;

    let box_max = vec3<f32>(f32(m.dims.x), f32(m.dims.y), f32(m.dims.z));
    let inv_d = 1.0 / d;
    let t0 = (vec3<f32>(0.0) - o) * inv_d;
    let t1 = (box_max - o) * inv_d;
    let tlo = min(t0, t1);
    let thi = max(t0, t1);
    let t_enter = max(max(tlo.x, tlo.y), max(tlo.z, 0.0));
    let t_exit = min(thi.x, min(thi.y, thi.z));
    if (t_exit < t_enter || t_enter >= limit) { return res; }

    let entry = o + t_enter * d;
    let dim_i = vec3<i32>(i32(m.dims.x), i32(m.dims.y), i32(m.dims.z));
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
    let max_steps = m.dims.x + m.dims.y + m.dims.z + 3u;

    for (var i: u32 = 0u; i < max_steps; i = i + 1u) {
        if (model_solid(m, p)) {
            if (t_hit < limit) {
                res.hit = true;
                res.t = t_hit;
                res.color = model_color(m, p);
            }
            return res;
        }
        if (t_max.x < t_max.y && t_max.x < t_max.z) {
            t_hit = t_max.x; p.x = p.x + step.x; t_max.x = t_max.x + t_delta.x;
            if (p.x < 0 || p.x >= dim_i.x) { return res; }
        } else if (t_max.y < t_max.z) {
            t_hit = t_max.y; p.y = p.y + step.y; t_max.y = t_max.y + t_delta.y;
            if (p.y < 0 || p.y >= dim_i.y) { return res; }
        } else {
            t_hit = t_max.z; p.z = p.z + step.z; t_max.z = t_max.z + t_delta.z;
            if (p.z < 0 || p.z >= dim_i.z) { return res; }
        }
        if (t_hit >= limit) { return res; }
    }
    return res;
}

@compute @workgroup_size(8, 8, 1)
fn march(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.screen_size.x || gid.y >= u.screen_size.y) { return; }
    let pix = gid.y * u.screen_size.x + gid.x;

    let aspect = f32(u.screen_size.x) / f32(u.screen_size.y);
    let half_h = tan(u.fov_y_rad * 0.5);
    let half_w = half_h * aspect;
    let ndc_x = (f32(gid.x) + 0.5) / f32(u.screen_size.x) * 2.0 - 1.0;
    let ndc_y_top = 1.0 - (f32(gid.y) + 0.5) / f32(u.screen_size.y) * 2.0;
    let ray_dir = normalize(
        u.cam_forward + ndc_x * half_w * u.cam_right - ndc_y_top * half_h * u.cam_down
    );

    // Start the nearest-hit search at the terrain depth; only sprites
    // closer than the world matter.
    var best_t = bitcast<f32>(depth_buffer[pix]);
    var best_color = vec3<f32>(0.0);
    var any = false;

    for (var i: u32 = 0u; i < u.instance_count; i = i + 1u) {
        let h = march_instance(instances[i], ray_dir, best_t);
        if (h.hit) {
            best_t = h.t;
            best_color = h.color;
            any = true;
        }
    }

    if (any) {
        let col = apply_fog(best_color, best_t);
        textureStore(output, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(col, 1.0));
    }
}
