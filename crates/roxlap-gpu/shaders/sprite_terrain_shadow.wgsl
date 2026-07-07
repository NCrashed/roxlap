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
const OCC_WORDS_PER_COLUMN_S: u32 = 8u; // CHUNK_Z (256) / 32, mip-0

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
// PF.1 — mirrors scene_dda.wgsl: loop-invariant meta fields are hoisted by
// the marcher; the solid mip-0 word base is computed once per slot and bits
// are tested inline with a last-word cache (see scene_dda's `shadow_occluded`).
fn slot_idx_of(pool_dims: vec3<u32>, chunk_idx: vec3<i32>) -> u32 {
    let mask = vec3<i32>(pool_dims) - vec3<i32>(1, 1, 1);
    let s = chunk_idx & mask;
    return u32(s.x) + u32(s.y) * pool_dims.x + u32(s.z) * pool_dims.x * pool_dims.y;
}
fn aabb_passed(mn: vec3<i32>, mx: vec3<i32>, p: vec3<i32>, step: vec3<i32>) -> bool {
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
fn chunk_has_content(slot_base: u32, chunk_occ_off: u32, slot_idx: u32, chunk_idx: vec3<i32>) -> bool {
    let stored = all_slot_chunk_idx[slot_base + slot_idx];
    if (stored.x != chunk_idx.x || stored.y != chunk_idx.y || stored.z != chunk_idx.z) {
        return false;
    }
    return (all_chunk_occupancy[chunk_occ_off + (slot_idx >> 5u)]
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
    // PF.1 — hoist loop-invariant meta fields once per march (field reads,
    // not a whole-struct copy: naga would materialise both mip arrays).
    let vsid = grid_static_meta[g].vsid;
    let occ_off = grid_static_meta[g].occupancy_offset;
    let occ_words_slot = grid_static_meta[g].occ_words_per_slot;
    let pool_dims = grid_static_meta[g].pool_dims;
    let slot_base = grid_static_meta[g].slot_chunk_idx_offset / 4u;
    let chunk_occ_off = grid_static_meta[g].chunk_occupancy_offset;
    let aabb_mn = grid_static_meta[g].aabb_min;
    let aabb_mx = grid_static_meta[g].aabb_max;
    // mip-0 SOLID block start within a slot (mip_occ_rel[0] == 0).
    let solid_rel0 = vsid * vsid * OCC_WORDS_PER_COLUMN_S;
    // SC.4 — world-local march: chunk + (mip-0, size-1) voxel cells scale by
    // the grid's voxel_world_size so a scaled terrain grid occludes sprites at
    // its true world footprint. 1.0 ⇒ identity (byte-identical).
    let vws = grid_cameras[g].world_origin.w;
    let chunk_dim = vec3<f32>(f32(vsid), f32(vsid), f32(CHUNK_Z)) * vws;
    let vsid_i = i32(vsid);
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
        if (aabb_passed(aabb_mn, aabb_mx, p_chunk, step_chunk)) { return false; }
        let slot_id = slot_idx_of(pool_dims, p_chunk);
        if (chunk_has_content(slot_base, chunk_occ_off, slot_id, p_chunk)) {
            let solid_col0 = occ_off + slot_id * occ_words_slot + solid_rel0;
            var occ_idx_cached: u32 = 0xffffffffu;
            var occ_word_cached: u32 = 0u;
            let entry_world = origin + t_enter * dir;
            let chunk_origin_world = vec3<f32>(p_chunk) * chunk_dim;
            let entry_in_chunk = entry_world - chunk_origin_world;
            var p_voxel = clamp(
                vec3<i32>(floor(entry_in_chunk / vws)), // SC.4 — world → voxel index
                vec3<i32>(0),
                vec3<i32>(vsid_i - 1, vsid_i - 1, cz_i - 1),
            );
            // SC.4 — voxel boundaries at integer-index × vws (world units).
            let next_voxel_world = vec3<f32>(
                select(f32(p_voxel.x), f32(p_voxel.x + 1), step_chunk.x > 0) * vws + chunk_origin_world.x,
                select(f32(p_voxel.y), f32(p_voxel.y + 1), step_chunk.y > 0) * vws + chunk_origin_world.y,
                select(f32(p_voxel.z), f32(p_voxel.z + 1), step_chunk.z > 0) * vws + chunk_origin_world.z,
            );
            var t_max_voxel = shield_parallel((next_voxel_world - origin) / dir, dir);
            let t_delta_voxel = abs(vec3<f32>(vws) / dir); // SC.4 — world voxel size
            loop {
                let z_u = u32(p_voxel.z);
                let widx = solid_col0
                    + (u32(p_voxel.x) + u32(p_voxel.y) * vsid) * OCC_WORDS_PER_COLUMN_S
                    + (z_u >> 5u);
                if (widx != occ_idx_cached) {
                    occ_idx_cached = widx;
                    occ_word_cached = occ_word(widx);
                }
                if ((occ_word_cached & (1u << (z_u & 31u))) != 0u) { return true; }
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
