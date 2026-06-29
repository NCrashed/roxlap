// XS.4.3 — sprite-cast shadow march for the scene pass (sprites CAST onto
// terrain). SNIPPET, not a standalone module: the renderer splices it into
// `scene_dda.wgsl` (replacing the `sprites_occlude` stub) only on
// sprite-shadow-capable devices, so it relies on that shader's `u` uniform and
// `shield_parallel`. It binds the sprite registry's instances / model meta /
// occupancy (bindings 19..21 — the scene pass reaches 19 storage buffers) and
// duplicates the sprite occupancy reader from sprite_model_dda.wgsl. The march
// loops the visible sprite instances (`u.sprite_cast_count`) and reports the
// first solid sprite voxel blocking a world-space shadow ray within `max_t`.
struct SpriteModelMetaS {
    occupancy_offset: u32,
    colors_offset: u32,
    color_offsets_offset: u32,
    occ_words_per_col: u32,
    dims: vec3<u32>,
    has_vox_materials: u32,
    pivot: vec3<f32>,
    voxel_world_size: f32,
};
struct SpriteInstanceS {
    inv_rot0: vec4<f32>,
    inv_rot1: vec4<f32>,
    inv_rot2: vec4<f32>,
    pos: vec3<f32>,
    model_id: u32,
    material: u32,
    alpha_mul: f32,
    flags: u32,
    _pad1: u32,
};
@group(0) @binding(19) var<storage, read> sprite_instances: array<SpriteInstanceS>;
@group(0) @binding(20) var<storage, read> sprite_models: array<SpriteModelMetaS>;
@group(0) @binding(21) var<storage, read> sprite_occupancy: array<u32>;

fn sprite_model_solid(m: SpriteModelMetaS, p: vec3<i32>) -> bool {
    let col = u32(p.x) + u32(p.y) * m.dims.x;
    let base = m.occupancy_offset + col * m.occ_words_per_col;
    let zw = u32(p.z) >> 5u;
    let zb = u32(p.z) & 31u;
    return (sprite_occupancy[base + zw] & (1u << zb)) != 0u;
}

// True if any casting sprite blocks the world-space ray within `max_t`.
fn sprites_occlude(origin_w: vec3<f32>, dir_w: vec3<f32>, max_t: f32) -> bool {
    for (var i: u32 = 0u; i < u.sprite_cast_count; i = i + 1u) {
        let inst = sprite_instances[i];
        // bit4 = NO_SHADOW_CAST → this sprite doesn't cast.
        if ((inst.flags & 16u) != 0u) { continue; }
        let m = sprite_models[inst.model_id];
        let inv = mat3x3<f32>(inst.inv_rot0.xyz, inst.inv_rot1.xyz, inst.inv_rot2.xyz);
        let s = m.voxel_world_size;
        // World → model-local. `d` includes the 1/s so the ray parameter `t`
        // stays in world units (== the caller's `max_t`).
        let o = inv * (origin_w - inst.pos) / s + m.pivot;
        let d = inv * dir_w / s;
        let box_max = vec3<f32>(f32(m.dims.x), f32(m.dims.y), f32(m.dims.z));
        let inv_d = 1.0 / d;
        let t0 = (vec3<f32>(0.0) - o) * inv_d;
        let t1 = (box_max - o) * inv_d;
        let tlo = min(t0, t1);
        let thi = max(t0, t1);
        let t_enter = max(max(tlo.x, tlo.y), max(tlo.z, 0.0));
        let t_exit = min(thi.x, min(thi.y, thi.z));
        if (t_exit < t_enter || t_enter >= max_t) { continue; }
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
        for (var k: u32 = 0u; k < max_steps; k = k + 1u) {
            if (t_hit >= max_t) { break; }
            if (sprite_model_solid(m, p)) { return true; }
            if (t_max.x < t_max.y && t_max.x < t_max.z) {
                t_hit = t_max.x; p.x = p.x + step.x; t_max.x = t_max.x + t_delta.x;
                if (p.x < 0 || p.x >= dim_i.x) { break; }
            } else if (t_max.y < t_max.z) {
                t_hit = t_max.y; p.y = p.y + step.y; t_max.y = t_max.y + t_delta.y;
                if (p.y < 0 || p.y >= dim_i.y) { break; }
            } else {
                t_hit = t_max.z; p.z = p.z + step.z; t_max.z = t_max.z + t_delta.z;
                if (p.z < 0 || p.z >= dim_i.z) { break; }
            }
        }
    }
    return false;
}
