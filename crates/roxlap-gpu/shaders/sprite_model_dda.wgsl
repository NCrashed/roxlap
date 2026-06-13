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
    voxel_world_size: f32, // GPU.10.4 LOD: world size of one voxel
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
    tiles_x: u32,    // GPU.10.3 screen-tile grid width
    tile_size: u32,  // GPU.10.3 tile edge in pixels
    _p6: f32,
};

@group(0) @binding(0) var<uniform> u: Uniform;
@group(0) @binding(1) var<storage, read> occupancy: array<u32>;
@group(0) @binding(2) var<storage, read> colors: array<u32>;
@group(0) @binding(3) var<storage, read> color_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> models: array<ModelMeta>;
@group(0) @binding(5) var<storage, read> instances: array<Instance>;
@group(0) @binding(6) var<storage, read> depth_buffer: array<u32>;
// Framebuffer as a storage BUFFER (packed `rgba8unorm`), shared with
// the scene pass — see the note in `scene_dda.wgsl`.
@group(0) @binding(7) var<storage, read_write> output: array<u32>;
// GPU.10.3 — screen-tile binning: per-tile (offset,count) into the
// flat grouped index list, so each pixel loops only its tile's sprites.
@group(0) @binding(8) var<storage, read> tile_ranges: array<u32>;
@group(0) @binding(9) var<storage, read> tile_instances: array<u32>;
// Per-voxel surface-normal index, parallel to `colors`.
@group(0) @binding(10) var<storage, read> dirs: array<u32>;
// Per-visible-instance voxlap kv6colmul[256] tables: two u32 per u64
// entry (lanes 0|1, then 2|3), 512 u32 per instance, packed in the
// same order as `instances`. Indexed `colmul[inst*512u + dir*2u + n]`.
@group(0) @binding(11) var<storage, read> colmul: array<u32>;

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

// Resolve the voxel at `p`, then shade it with this instance's
// kv6colmul table indexed by the voxel's surface normal — the same
// per-channel `_mm_mulhi_epu16` modulation the CPU rasteriser applies
// (voxlap `drawboundcubesse`): `out[c] = min(255, (rgb[c] * mul[c]) >> 8)`.
// Returns linear-ish 0..1 RGB (the marcher composites + fogs afterward).
fn model_color(m: ModelMeta, p: vec3<i32>, inst_idx: u32) -> vec3<f32> {
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
    let vidx = m.colors_offset + local_off + rank;
    let packed = colors[vidx];
    let dir = dirs[vidx] & 0xffu;

    // kv6colmul[dir] for this instance: lanes (B,G) in lo, (R,A) in hi.
    let cbase = inst_idx * 512u + dir * 2u;
    let lo = colmul[cbase];
    let hi = colmul[cbase + 1u];
    let r = min(255u, (((packed >> 16u) & 0xffu) * (hi & 0xffffu)) >> 8u);
    let g = min(255u, (((packed >> 8u) & 0xffu) * (lo >> 16u)) >> 8u);
    let b = min(255u, ((packed & 0xffu) * (lo & 0xffffu)) >> 8u);
    return vec3<f32>(f32(r), f32(g), f32(b)) / 255.0;
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
fn march_instance(inst: Instance, inst_idx: u32, ray_dir: vec3<f32>, limit: f32) -> Hit {
    var res: Hit;
    res.hit = false;
    res.t = limit;
    res.color = vec3<f32>(0.0);

    let m = models[inst.model_id];
    let inv = mat3x3<f32>(inst.inv_rot0.xyz, inst.inv_rot1.xyz, inst.inv_rot2.xyz);
    // World → model-local voxel space. Dividing by voxel_world_size
    // makes one voxel span that many world units (GPU.10.4 LOD: coarse
    // mips have larger voxels) and keeps the ray parameter `t` in world
    // units, so the depth composite stays correct. mip-0 has size 1.
    let s = m.voxel_world_size;
    let o = inv * (u.cam_pos - inst.pos) / s + m.pivot;
    let d = inv * ray_dir / s;

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
                res.color = model_color(m, p, inst_idx);
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

    // GPU.10.3 — loop only the instances binned to this pixel's tile.
    let tile = (gid.y / u.tile_size) * u.tiles_x + (gid.x / u.tile_size);
    let offset = tile_ranges[2u * tile];
    let count = tile_ranges[2u * tile + 1u];
    for (var k: u32 = 0u; k < count; k = k + 1u) {
        let inst_idx = tile_instances[offset + k];
        let h = march_instance(instances[inst_idx], inst_idx, ray_dir, best_t);
        if (h.hit) {
            best_t = h.t;
            best_color = h.color;
            any = true;
        }
    }

    if (any) {
        let col = apply_fog(best_color, best_t);
        output[gid.y * u.screen_size.x + gid.x] = pack4x8unorm(vec4<f32>(col, 1.0));
    }
}
