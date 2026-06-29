// XS.4.2 — terrain shadow march for the sprite pass (sprites RECEIVE terrain
// shadows). This is a SNIPPET, not a standalone module: it's spliced into
// `sprite_model_dda.wgsl` (replacing the stub `shadow_occluded_world`) only on
// sprite-shadow-capable devices, so it relies on that shader's `u` uniform and
// `shield_parallel`. It re-declares the terrain occupancy bindings (16..23) and
// duplicates scene_dda.wgsl's occupancy-reader + shadow-marcher ABI verbatim —
// kept in lockstep with scene_dda (same buffer layouts: GridStaticMeta 144 B,
// PerGridCamera 144 B). The 8 storage buffers push the sprite pass to 22, which
// the renderer only binds when the device grants that many.
const CHUNK_Z: u32 = 256u;
const MAX_GPU_MIPS: u32 = 6u;

struct PerGridCamera {
    pos: vec3<f32>, _pc0: f32,
    right: vec3<f32>, _pc1: f32,
    down: vec3<f32>, _pc2: f32,
    forward: vec3<f32>, _pc3: f32,
    sun_dir: vec4<f32>,
    world_origin: vec4<f32>,
    rot0: vec4<f32>,
    rot1: vec4<f32>,
    rot2: vec4<f32>,
};
struct GridStaticMeta {
    occupancy_offset: u32,
    color_offsets_offset: u32,
    colors_offset: u32,
    chunk_colors_base_offset: u32,
    chunk_occupancy_offset: u32,
    slot_chunk_idx_offset: u32,
    vsid: u32,
    total_slots: u32,
    pool_dims: vec3<u32>,
    _gm0: u32,
    occ_words_per_slot: u32,
    offsets_words_per_slot: u32,
    mip_count: u32,
    _gm1: u32,
    mip_occ_rel: array<u32, MAX_GPU_MIPS>,
    mip_coff_rel: array<u32, MAX_GPU_MIPS>,
    aabb_min: vec3<i32>,
    aabb_max: vec3<i32>,
};
@group(0) @binding(16) var<storage, read> occ_page0: array<u32>;
@group(0) @binding(17) var<storage, read> occ_page1: array<u32>;
@group(0) @binding(18) var<storage, read> occ_page2: array<u32>;
@group(0) @binding(19) var<storage, read> occ_page3: array<u32>;
@group(0) @binding(20) var<storage, read> all_chunk_occupancy: array<u32>;
@group(0) @binding(21) var<storage, read> all_slot_chunk_idx: array<vec3<i32>>;
@group(0) @binding(22) var<storage, read> grid_static_meta: array<GridStaticMeta>;
@group(0) @binding(23) var<storage, read> grid_cameras: array<PerGridCamera>;

fn occ_word(i: u32) -> u32 {
    if (u.occ_num_pages <= 1u) { return occ_page0[i]; }
    let page = i / u.occ_page_words;
    let local = i % u.occ_page_words;
    if (page == 0u) { return occ_page0[local]; }
    if (page == 1u) { return occ_page1[local]; }
    if (page == 2u) { return occ_page2[local]; }
    return occ_page3[local];
}
fn occ_words_per_col_for_mip(mip: u32) -> u32 {
    return max(1u, (CHUNK_Z >> mip) / 32u);
}
fn col_word_base_mip(g: u32, meta_id: u32, mip: u32, p_voxel: vec3<i32>) -> u32 {
    let vsid_mip = grid_static_meta[g].vsid >> mip;
    let col_idx = u32(p_voxel.x) + u32(p_voxel.y) * vsid_mip;
    let occ_base = grid_static_meta[g].occupancy_offset
        + meta_id * grid_static_meta[g].occ_words_per_slot
        + grid_static_meta[g].mip_occ_rel[mip];
    return occ_base + col_idx * occ_words_per_col_for_mip(mip);
}
fn mip_occ_block_words(g: u32, mip: u32) -> u32 {
    let vsid_mip = grid_static_meta[g].vsid >> mip;
    return vsid_mip * vsid_mip * occ_words_per_col_for_mip(mip);
}
fn voxel_solid_in(g: u32, meta_id: u32, mip: u32, p_voxel: vec3<i32>) -> bool {
    let solid_base = col_word_base_mip(g, meta_id, mip, p_voxel) + mip_occ_block_words(g, mip);
    let z_word = u32(p_voxel.z) >> 5u;
    let z_bit = u32(p_voxel.z) & 31u;
    return (occ_word(solid_base + z_word) & (1u << z_bit)) != 0u;
}
fn slot_idx_of(g: u32, chunk_idx: vec3<i32>) -> u32 {
    let m = grid_static_meta[g];
    let mask = vec3<i32>(m.pool_dims) - vec3<i32>(1, 1, 1);
    let s = chunk_idx & mask;
    return u32(s.x) + u32(s.y) * m.pool_dims.x + u32(s.z) * m.pool_dims.x * m.pool_dims.y;
}
fn aabb_passed(g: u32, p: vec3<i32>, step: vec3<i32>) -> bool {
    let mn = grid_static_meta[g].aabb_min;
    let mx = grid_static_meta[g].aabb_max;
    if (step.x > 0 && p.x > mx.x) { return true; }
    if (step.x < 0 && p.x < mn.x) { return true; }
    if (step.x == 0 && (p.x < mn.x || p.x > mx.x)) { return true; }
    if (step.y > 0 && p.y > mx.y) { return true; }
    if (step.y < 0 && p.y < mn.y) { return true; }
    if (step.y == 0 && (p.y < mn.y || p.y > mx.y)) { return true; }
    if (step.z > 0 && p.z > mx.z) { return true; }
    if (step.z < 0 && p.z < mn.z) { return true; }
    if (step.z == 0 && (p.z < mn.z || p.z > mx.z)) { return true; }
    return false;
}
fn chunk_has_content(g: u32, slot_idx: u32, chunk_idx: vec3<i32>) -> bool {
    let m = grid_static_meta[g];
    let stored = all_slot_chunk_idx[m.slot_chunk_idx_offset / 4u + slot_idx];
    if (stored.x != chunk_idx.x || stored.y != chunk_idx.y || stored.z != chunk_idx.z) {
        return false;
    }
    return (all_chunk_occupancy[m.chunk_occupancy_offset + (slot_idx >> 5u)]
        & (1u << (slot_idx & 31u))) != 0u;
}
fn world_to_grid_local(g: u32, w: vec3<f32>) -> vec3<f32> {
    let c = grid_cameras[g];
    let v = w - c.world_origin.xyz;
    return vec3<f32>(dot(c.rot0.xyz, v), dot(c.rot1.xyz, v), dot(c.rot2.xyz, v));
}
fn world_dir_to_grid_local(g: u32, d: vec3<f32>) -> vec3<f32> {
    let c = grid_cameras[g];
    return vec3<f32>(dot(c.rot0.xyz, d), dot(c.rot1.xyz, d), dot(c.rot2.xyz, d));
}
// Per-grid intra-grid shadow march — verbatim copy of scene_dda's `shadow_occluded`.
fn shadow_occluded(g: u32, origin: vec3<f32>, dir: vec3<f32>, max_t: f32) -> bool {
    let m = grid_static_meta[g];
    let chunk_dim = vec3<f32>(f32(m.vsid), f32(m.vsid), f32(CHUNK_Z));
    let vsid_i = i32(m.vsid);
    let cz_i = i32(CHUNK_Z);
    var p_chunk = vec3<i32>(floor(origin / chunk_dim));
    let step_chunk = vec3<i32>(sign(dir));
    let t_delta_chunk = abs(chunk_dim / dir);
    let next_boundary_chunk = vec3<f32>(
        select(f32(p_chunk.x), f32(p_chunk.x + 1), step_chunk.x > 0) * chunk_dim.x,
        select(f32(p_chunk.y), f32(p_chunk.y + 1), step_chunk.y > 0) * chunk_dim.y,
        select(f32(p_chunk.z), f32(p_chunk.z + 1), step_chunk.z > 0) * chunk_dim.z,
    );
    var t_max_chunk = shield_parallel((next_boundary_chunk - origin) / dir, dir);
    var t_enter: f32 = 0.0;
    var steps: u32 = 0u;
    for (var oc: u32 = 0u; oc < u.max_outer_steps; oc = oc + 1u) {
        if (t_enter > max_t) { return false; }
        if (aabb_passed(g, p_chunk, step_chunk)) { return false; }
        let slot_id = slot_idx_of(g, p_chunk);
        if (chunk_has_content(g, slot_id, p_chunk)) {
            let entry_world = origin + t_enter * dir;
            let chunk_origin_world = vec3<f32>(p_chunk) * chunk_dim;
            let entry_in_chunk = entry_world - chunk_origin_world;
            var p_voxel = clamp(
                vec3<i32>(floor(entry_in_chunk)),
                vec3<i32>(0),
                vec3<i32>(vsid_i - 1, vsid_i - 1, cz_i - 1),
            );
            let next_voxel_world = vec3<f32>(
                select(f32(p_voxel.x), f32(p_voxel.x + 1), step_chunk.x > 0) + chunk_origin_world.x,
                select(f32(p_voxel.y), f32(p_voxel.y + 1), step_chunk.y > 0) + chunk_origin_world.y,
                select(f32(p_voxel.z), f32(p_voxel.z + 1), step_chunk.z > 0) + chunk_origin_world.z,
            );
            var t_max_voxel = shield_parallel((next_voxel_world - origin) / dir, dir);
            let t_delta_voxel = abs(vec3<f32>(1.0) / dir);
            loop {
                if (voxel_solid_in(g, slot_id, 0u, p_voxel)) { return true; }
                steps = steps + 1u;
                if (steps >= u.shadow_max_steps) { return false; }
                if (t_max_voxel.x < t_max_voxel.y && t_max_voxel.x < t_max_voxel.z) {
                    if (t_max_voxel.x > max_t) { return false; }
                    p_voxel.x = p_voxel.x + step_chunk.x;
                    t_max_voxel.x = t_max_voxel.x + t_delta_voxel.x;
                    if (p_voxel.x < 0 || p_voxel.x >= vsid_i) { break; }
                } else if (t_max_voxel.y < t_max_voxel.z) {
                    if (t_max_voxel.y > max_t) { return false; }
                    p_voxel.y = p_voxel.y + step_chunk.y;
                    t_max_voxel.y = t_max_voxel.y + t_delta_voxel.y;
                    if (p_voxel.y < 0 || p_voxel.y >= vsid_i) { break; }
                } else {
                    if (t_max_voxel.z > max_t) { return false; }
                    p_voxel.z = p_voxel.z + step_chunk.z;
                    t_max_voxel.z = t_max_voxel.z + t_delta_voxel.z;
                    if (p_voxel.z < 0 || p_voxel.z >= cz_i) { break; }
                }
            }
        }
        if (t_max_chunk.x < t_max_chunk.y && t_max_chunk.x < t_max_chunk.z) {
            t_enter = t_max_chunk.x;
            p_chunk.x = p_chunk.x + step_chunk.x;
            t_max_chunk.x = t_max_chunk.x + t_delta_chunk.x;
        } else if (t_max_chunk.y < t_max_chunk.z) {
            t_enter = t_max_chunk.y;
            p_chunk.y = p_chunk.y + step_chunk.y;
            t_max_chunk.y = t_max_chunk.y + t_delta_chunk.y;
        } else {
            t_enter = t_max_chunk.z;
            p_chunk.z = p_chunk.z + step_chunk.z;
            t_max_chunk.z = t_max_chunk.z + t_delta_chunk.z;
        }
    }
    return false;
}
// World-space shadow query over every terrain grid (XS.3 cross-grid).
fn shadow_occluded_world(origin_w: vec3<f32>, dir_w: vec3<f32>, max_t: f32) -> bool {
    for (var g: u32 = 0u; g < u.grid_count; g = g + 1u) {
        let o = world_to_grid_local(g, origin_w);
        let d = world_dir_to_grid_local(g, dir_w);
        if (shadow_occluded(g, o, d, max_t)) { return true; }
    }
    return false;
}
