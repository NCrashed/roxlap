//! XS.1 — a world-space scene occlusion oracle for cross-grid (and, in
//! XS.2, cross-sprite) hard shadows.
//!
//! Dynamic-lighting shadow rays on the CPU were single-grid: a hit in grid
//! A could only be shadowed by grid A's own voxels (`dda::SamplerShadow`).
//! [`SceneOccluder`] lifts the test to **world space** over every grid in the
//! scene: given a world-space shadow ray it transforms the ray into each
//! grid's local frame and marches that grid's voxels, returning `true` on the
//! first solid hit. The CPU DDA reaches it through
//! [`roxlap_core::DdaEnv::world_shadow`] (a [`roxlap_core::WorldOccluder`]
//! trait object) + the current grid's local→world transform, so a sun /
//! point-light ray that leaves the hit grid keeps testing the rest of the
//! scene — the ship drops a shadow on the ground, etc.
//!
//! Built once per frame, borrowing the scene immutably (it coexists with the
//! immutable render pass). Only constructed when shadows are actually active,
//! so the no-shadow path allocates nothing.

use glam::{DQuat, DVec3, IVec3, Vec3};

use crate::{Grid, Scene, CHUNK_SIZE_XY, CHUNK_SIZE_Z};

/// Safety cap on a shadow ray's voxel steps within one grid (the `max_t` /
/// AABB bound is the real limit; this only backstops a degenerate ray).
const SHADOW_MAX_STEPS: u32 = 4096;

/// One grid's contribution to the scene occluder: an immutable borrow plus
/// the cached world→local transform and the grid-local voxel AABB (so a ray
/// that never enters the grid is rejected without marching).
struct GridOcc<'a> {
    grid: &'a Grid,
    /// Grid world origin.
    origin: DVec3,
    /// World→grid-local rotation (the inverse of the grid's rotation).
    rot_inv: DQuat,
    /// SC.2 — the grid's `voxel_world_size`. The world ray is divided by this
    /// (after the inverse-rotation) to reach the grid's VOXEL frame, in which
    /// `lo`/`hi` and the DDA march live. `1.0` for an unscaled grid.
    vws: f32,
    /// Grid-local voxel AABB `[lo, hi)` (mip-0 voxel coords) covering every
    /// materialised chunk; the shadow march is clipped to it.
    lo: [f32; 3],
    hi: [f32; 3],
}

/// World-space scene occlusion oracle (see module docs).
pub struct SceneOccluder<'a> {
    grids: Vec<GridOcc<'a>>,
}

impl<'a> SceneOccluder<'a> {
    /// Build the occluder over every materialised grid in `scene`. Cheap:
    /// per grid it walks the chunk keys once for the AABB (the same cost the
    /// render loop's `grid_bounds` early-out already pays).
    #[must_use]
    pub fn build(scene: &'a Scene) -> Self {
        let mut grids = Vec::new();
        for (_id, grid) in scene.grids() {
            if let Some((mut lo, hi)) = grid_voxel_aabb(grid) {
                // CA.2 — cutaway: everything above the clip plane
                // (`z < z_clip`, z-down) is air to the renderer, so
                // shadow rays must not be blocked by it either
                // ("world as if removed"). Raising the occluder box
                // floor to the plane makes the march skip the hidden
                // band entirely; each grid applies its OWN clip, so a
                // clipped ship stops casting onto the ground while the
                // unclipped ground keeps shadowing normally. A fully
                // hidden grid stops casting altogether.
                #[allow(clippy::cast_precision_loss)]
                if let Some(zc) = grid.z_clip {
                    lo[2] = lo[2].max(zc as f32);
                    if lo[2] >= hi[2] {
                        continue;
                    }
                }
                #[allow(clippy::cast_possible_truncation)]
                grids.push(GridOcc {
                    grid,
                    origin: grid.transform.origin,
                    rot_inv: grid.transform.rotation.inverse(),
                    vws: grid.transform.voxel_world_size as f32,
                    lo,
                    hi,
                });
            }
        }
        Self { grids }
    }

    /// Whether the occluder holds anything (an empty scene casts no shadows).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.grids.is_empty()
    }
}

impl roxlap_core::WorldOccluder for SceneOccluder<'_> {
    fn occluded_world(&self, origin: [f64; 3], dir: [f32; 3], max_t: f32) -> bool {
        // SC.2 — `origin` arrives f64 (full world precision); only `dir`
        // (unit-ish) is widened from f32.
        let ow = DVec3::new(origin[0], origin[1], origin[2]);
        let dw = DVec3::new(f64::from(dir[0]), f64::from(dir[1]), f64::from(dir[2]));
        for g in &self.grids {
            if occluded_in_grid(g, ow, dw, max_t) {
                return true;
            }
        }
        false
    }
}

/// World-space voxel AABB → grid-local voxel AABB `[lo, hi)` from the grid's
/// materialised chunk extent. `None` for an empty grid.
fn grid_voxel_aabb(grid: &Grid) -> Option<([f32; 3], [f32; 3])> {
    let mut min = IVec3::splat(i32::MAX);
    let mut max = IVec3::splat(i32::MIN);
    let mut any = false;
    for idx in grid.chunks.keys() {
        any = true;
        min = min.min(*idx);
        max = max.max(*idx);
    }
    if !any {
        return None;
    }
    #[allow(clippy::cast_precision_loss)]
    let cs_xy = CHUNK_SIZE_XY as i32;
    #[allow(clippy::cast_precision_loss)]
    let cs_z = CHUNK_SIZE_Z as i32;
    let lo = [
        (min.x * cs_xy) as f32,
        (min.y * cs_xy) as f32,
        (min.z * cs_z) as f32,
    ];
    let hi = [
        ((max.x + 1) * cs_xy) as f32,
        ((max.y + 1) * cs_xy) as f32,
        ((max.z + 1) * cs_z) as f32,
    ];
    Some((lo, hi))
}

/// March one grid's voxels along a world-space ray (transformed to grid-local)
/// and report whether a solid voxel blocks it within `max_t` world units.
///
/// PF.9 — three-tier empty-space skip, mirroring the primary DDA's
/// leak-free fast-forward: an ABSENT chunk jumps its whole
/// 128×128×256 box; a present chunk with an empty 64³ super-brick /
/// 8³ brick (from the grid's `dda_brick_cache`, mip 0) jumps that box.
/// Skipping only guaranteed-empty boxes can never hide an occluder; the
/// landing cell pins the exit axis to the integer boundary so the next
/// box's entry cell is visited densely (the `cell_walk_skip` contract).
/// The step budget is consumed in Manhattan cell distance — exactly
/// what the dense walk would have spent — so the `SHADOW_MAX_STEPS`
/// truncation fires at the identical point (bit-compatible results).
/// Chunks with no cached mip-0 brick map (e.g. Mid-LOD grids) fall back
/// to the dense per-voxel walk — conservative, never wrong.
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn occluded_in_grid(g: &GridOcc<'_>, ow: DVec3, dw: DVec3, max_t: f32) -> bool {
    // World → grid-local VOXEL frame: inverse-rotate (rigid) then divide by
    // `vws` (world units → voxel indices, in which `lo`/`hi` live). Dividing
    // both `o` and `d` by the same factor preserves the ray parameter `t`, so
    // `max_t` still clips at the right point. `vws == 1.0` is byte-identical.
    let o: Vec3 = (g.rot_inv * (ow - g.origin)).as_vec3() / g.vws;
    let d: Vec3 = (g.rot_inv * dw).as_vec3() / g.vws;
    let o = [o.x, o.y, o.z];
    let d = [d.x, d.y, d.z];

    // Clip to the grid's voxel AABB — a ray that never enters can't occlude.
    let Some((t0, t1)) = intersect_aabb(o, d, g.lo, g.hi) else {
        return false;
    };
    let t_enter = t0.max(0.0);
    let t_exit = t1.min(max_t);
    if t_enter > t_exit {
        return false;
    }

    let start = t_enter + 1e-4;
    let p = [
        o[0] + d[0] * start,
        o[1] + d[1] * start,
        o[2] + d[2] * start,
    ];
    let mut cell = [
        (p[0].floor() as i32).clamp(g.lo[0] as i32, g.hi[0] as i32 - 1),
        (p[1].floor() as i32).clamp(g.lo[1] as i32, g.hi[1] as i32 - 1),
        (p[2].floor() as i32).clamp(g.lo[2] as i32, g.hi[2] as i32 - 1),
    ];
    let (step, mut t_max, t_delta) = dda_setup(o, d, cell);
    // Reciprocal direction: box-boundary `t`s and the post-jump t_max
    // refresh use multiplies (0.0 where step == 0 — those axes stay +∞).
    let inv = [
        if step[0] != 0 { 1.0 / d[0] } else { 0.0 },
        if step[1] != 0 { 1.0 / d[1] } else { 0.0 },
        if step[2] != 0 { 1.0 / d[2] } else { 0.0 },
    ];
    let mut t_curr = t_enter;
    let lo_i = [g.lo[0] as i32, g.lo[1] as i32, g.lo[2] as i32];
    let hi_i = [g.hi[0] as i32, g.hi[1] as i32, g.hi[2] as i32];
    let (cs_xy, cs_z) = (CHUNK_SIZE_XY as i32, CHUNK_SIZE_Z as i32);
    // PF.6 — chunk-cached sampler: one HashMap probe per chunk crossing
    // (not per step), and the solid test walks the slab chain in place
    // (no per-step allocation / whole-column decode).
    let mut sampler = g.grid.solid_sampler();
    let cache = &g.grid.dda_brick_cache;
    let mut used = 0u32;
    while used < SHADOW_MAX_STEPS {
        if cell[0] < lo_i[0]
            || cell[0] >= hi_i[0]
            || cell[1] < lo_i[1]
            || cell[1] >= hi_i[1]
            || cell[2] < lo_i[2]
            || cell[2] >= hi_i[2]
            || t_curr > t_exit
        {
            return false;
        }
        let (chunk_idx, in_chunk) = crate::voxel_split(IVec3::new(cell[0], cell[1], cell[2]));
        // Empty-box tier: absent chunk → whole chunk box; else the brick
        // cache's super/brick blocks (global 8/64 alignment coincides with
        // the chunk-local maps because chunk dims are multiples of 64,
        // and `>>` floors for negative coords).
        let vxl = sampler.chunk_at(chunk_idx);
        let skip_box: Option<([i32; 3], [i32; 3])> = if vxl.is_none() {
            let lo = [chunk_idx.x * cs_xy, chunk_idx.y * cs_xy, chunk_idx.z * cs_z];
            Some((lo, [lo[0] + cs_xy, lo[1] + cs_xy, lo[2] + cs_z]))
        } else {
            let ch = [chunk_idx.x, chunk_idx.y, chunk_idx.z];
            let in_c = [in_chunk.x as i32, in_chunk.y as i32, in_chunk.z as i32];
            if cache.super_occupied_at(ch, 0, in_c) == Some(false) {
                let lo = [
                    (cell[0] >> 6) << 6,
                    (cell[1] >> 6) << 6,
                    (cell[2] >> 6) << 6,
                ];
                Some((lo, [lo[0] + 64, lo[1] + 64, lo[2] + 64]))
            } else if cache.brick_occupied_at(ch, 0, in_c) == Some(false) {
                let lo = [
                    (cell[0] >> 3) << 3,
                    (cell[1] >> 3) << 3,
                    (cell[2] >> 3) << 3,
                ];
                Some((lo, [lo[0] + 8, lo[1] + 8, lo[2] + 8]))
            } else {
                None
            }
        };
        if let Some((blo, bhi)) = skip_box {
            let mut best_t = f32::INFINITY;
            let mut best_axis = 3usize;
            let mut plane = [0i32; 3];
            for a in 0..3 {
                if step[a] == 0 {
                    continue;
                }
                plane[a] = if step[a] > 0 { bhi[a] } else { blo[a] };
                let tb = (plane[a] as f32 - o[a]) * inv[a];
                if tb < best_t {
                    best_t = tb;
                    best_axis = a;
                }
            }
            if best_axis == 3 {
                return false;
            }
            // Landing mirrors `cell_walk_skip`: exit axis pinned to the
            // boundary cell, other axes advanced by exact crossing
            // counts from t-differences (an absolute-position re-floor
            // is ill-conditioned when the ray origin is far away).
            let mut nc = cell;
            for a in 0..3 {
                if a == best_axis || step[a] == 0 || t_max[a] > best_t {
                    continue;
                }
                let k = ((best_t - t_max[a]) / t_delta[a]) as i32 + 1;
                nc[a] += k * step[a];
                t_max[a] += k as f32 * t_delta[a];
            }
            nc[best_axis] = if step[best_axis] > 0 {
                plane[best_axis]
            } else {
                plane[best_axis] - 1
            };
            t_max[best_axis] = best_t + t_delta[best_axis];
            // Budget: the dense walk would have spent one step per crossed
            // cell; if it runs out inside the empty box it would have
            // returned `false` there (nothing solid inside to find).
            let crossed =
                cell[0].abs_diff(nc[0]) + cell[1].abs_diff(nc[1]) + cell[2].abs_diff(nc[2]);
            if used.saturating_add(crossed) >= SHADOW_MAX_STEPS {
                return false;
            }
            used += crossed;
            cell = nc;
            t_curr = best_t.max(t_curr);
            continue;
        }
        // Occupied brick (or no cached map): dense per-voxel test through
        // the already-probed chunk. A cell occludes iff it is
        // **render-solid** — the same `surface_color_mip` rule the
        // primary rays use — NOT the collision-level `vxl_voxel_solid`:
        // the voxlap zero-RGB bedrock placeholder (the single voxel at
        // the bottom of every never-written column of a materialised
        // chunk) is invisible to the renderer, and counting it as an
        // occluder makes a stacked-chunk grid's WHOLE 128×128 chunk
        // footprint cast a phantom shadow plate (the Decks report: the
        // CPU over-shadowed a full-width band the GPU — whose
        // decompressor drops placeholders — correctly didn't).
        // `vxl_voxel_solid` stays as the cheap first gate; the colour
        // walk runs only on solid cells (at most a handful per ray).
        if let Some(vxl) = vxl {
            if crate::chunks::vxl_voxel_solid(vxl, in_chunk.x, in_chunk.y, in_chunk.z)
                && roxlap_core::GridView::from_single_vxl(vxl)
                    .surface_color_mip(in_chunk.x, in_chunk.y, in_chunk.z, 0)
                    .is_some()
            {
                return true;
            }
        }
        let a = min_axis(t_max);
        t_curr = t_max[a];
        cell[a] += step[a];
        t_max[a] += t_delta[a];
        used += 1;
    }
    false
}

/// Slab-method ray/AABB intersection (`[lo, hi]`, voxel units). Returns the
/// entry/exit ray parameters, or `None` if the ray misses.
fn intersect_aabb(o: [f32; 3], d: [f32; 3], lo: [f32; 3], hi: [f32; 3]) -> Option<(f32, f32)> {
    let mut tmin = f32::NEG_INFINITY;
    let mut tmax = f32::INFINITY;
    for a in 0..3 {
        if d[a].abs() < 1e-9 {
            if o[a] < lo[a] || o[a] > hi[a] {
                return None;
            }
        } else {
            let inv = 1.0 / d[a];
            let mut t0 = (lo[a] - o[a]) * inv;
            let mut t1 = (hi[a] - o[a]) * inv;
            if t0 > t1 {
                std::mem::swap(&mut t0, &mut t1);
            }
            tmin = tmin.max(t0);
            tmax = tmax.min(t1);
            if tmin > tmax {
                return None;
            }
        }
    }
    Some((tmin, tmax))
}

/// 3D-DDA setup for a 1-voxel grid: per-axis step, first boundary `t`, and
/// per-cell `t` increment (mirror of `roxlap_core`'s private helper).
fn dda_setup(o: [f32; 3], d: [f32; 3], cell: [i32; 3]) -> ([i32; 3], [f32; 3], [f32; 3]) {
    let mut step = [0i32; 3];
    let mut t_max = [f32::INFINITY; 3];
    let mut t_delta = [f32::INFINITY; 3];
    for a in 0..3 {
        if d[a] > 1e-9 {
            step[a] = 1;
            #[allow(clippy::cast_precision_loss)]
            let boundary = (cell[a] + 1) as f32;
            t_max[a] = (boundary - o[a]) / d[a];
            t_delta[a] = 1.0 / d[a];
        } else if d[a] < -1e-9 {
            step[a] = -1;
            #[allow(clippy::cast_precision_loss)]
            let boundary = cell[a] as f32;
            t_max[a] = (boundary - o[a]) / d[a];
            t_delta[a] = -1.0 / d[a];
        }
    }
    (step, t_max, t_delta)
}

#[inline]
fn min_axis(t: [f32; 3]) -> usize {
    if t[0] < t[1] && t[0] < t[2] {
        0
    } else if t[1] < t[2] {
        1
    } else {
        2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GridTransform, Scene};
    use roxlap_formats::color::VoxColor;

    /// The pre-PF.9 dense per-voxel shadow march, kept verbatim as the
    /// equivalence oracle for the skipping version (`Grid::voxel_solid`
    /// per cell, one step per iteration).
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn occluded_dense(g: &GridOcc<'_>, ow: DVec3, dw: DVec3, max_t: f32) -> bool {
        let o: Vec3 = (g.rot_inv * (ow - g.origin)).as_vec3() / g.vws;
        let d: Vec3 = (g.rot_inv * dw).as_vec3() / g.vws;
        let o = [o.x, o.y, o.z];
        let d = [d.x, d.y, d.z];
        let Some((t0, t1)) = intersect_aabb(o, d, g.lo, g.hi) else {
            return false;
        };
        let t_enter = t0.max(0.0);
        let t_exit = t1.min(max_t);
        if t_enter > t_exit {
            return false;
        }
        let start = t_enter + 1e-4;
        let p = [
            o[0] + d[0] * start,
            o[1] + d[1] * start,
            o[2] + d[2] * start,
        ];
        let mut cell = [
            (p[0].floor() as i32).clamp(g.lo[0] as i32, g.hi[0] as i32 - 1),
            (p[1].floor() as i32).clamp(g.lo[1] as i32, g.hi[1] as i32 - 1),
            (p[2].floor() as i32).clamp(g.lo[2] as i32, g.hi[2] as i32 - 1),
        ];
        let (step, mut t_max, t_delta) = dda_setup(o, d, cell);
        let mut t_curr = t_enter;
        let lo_i = [g.lo[0] as i32, g.lo[1] as i32, g.lo[2] as i32];
        let hi_i = [g.hi[0] as i32, g.hi[1] as i32, g.hi[2] as i32];
        for _ in 0..SHADOW_MAX_STEPS {
            if cell[0] < lo_i[0]
                || cell[0] >= hi_i[0]
                || cell[1] < lo_i[1]
                || cell[1] >= hi_i[1]
                || cell[2] < lo_i[2]
                || cell[2] >= hi_i[2]
                || t_curr > t_exit
            {
                return false;
            }
            // Render-solid, like the production march (the zero-RGB
            // bedrock placeholder must not occlude — see there).
            let (chunk_idx, in_chunk) = crate::voxel_split(IVec3::new(cell[0], cell[1], cell[2]));
            if let Some(vxl) = g.grid.chunks.get(&chunk_idx) {
                if crate::chunks::vxl_voxel_solid(vxl, in_chunk.x, in_chunk.y, in_chunk.z)
                    && roxlap_core::GridView::from_single_vxl(vxl)
                        .surface_color_mip(in_chunk.x, in_chunk.y, in_chunk.z, 0)
                        .is_some()
                {
                    return true;
                }
            }
            let a = min_axis(t_max);
            t_curr = t_max[a];
            cell[a] += step[a];
            t_max[a] += t_delta[a];
        }
        false
    }

    /// PF.9 — the three-tier skipping march must agree with the dense
    /// reference for every ray: hits, misses, AND the step-budget
    /// truncation point (Manhattan accounting). The scene has caves,
    /// thin diagonal walls, an ABSENT chunk gap (exercises the
    /// chunk-box jump), and chunks with no brick maps until
    /// `ensure_dda_bricks` runs (exercises the dense fallback first).
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn skip_march_matches_dense_reference() {
        let mut scene = Scene::new();
        let gid = scene.add_grid(GridTransform::identity());
        let grid = scene.grid_mut(gid).unwrap();
        // Terrain across chunks (0,0,0), (1,0,0) and (3,0,0) — chunk
        // (2,0,0) is deliberately ABSENT (air gap the skip must jump).
        grid.set_rect(
            IVec3::new(0, 0, 160),
            IVec3::new(255, 127, 255),
            Some(VoxColor(0x80_55_66_77)),
        );
        grid.set_rect(
            IVec3::new(384, 0, 160),
            IVec3::new(511, 127, 255),
            Some(VoxColor(0x80_55_66_77)),
        );
        // Caves + pillars for thin/diagonal occluders.
        for i in 0..14 {
            let (x, y) = (29 * i % 220 + 10, 41 * i % 100 + 10);
            grid.set_sphere(IVec3::new(x, y, 170), 7, None);
            grid.set_rect(
                IVec3::new(x + 3, y + 3, 120),
                IVec3::new(x + 4, y + 4, 159),
                Some(VoxColor(0x80_99_88_77)),
            );
        }

        // Phase 1: no brick maps yet → every present chunk takes the
        // dense fallback; absent-chunk jumps still fire.
        let run = |scene: &Scene| {
            let occ = SceneOccluder::build(scene);
            let g = &occ.grids[0];
            // Deterministic LCG ray sweep: origins above/inside the
            // terrain band, directions over the sphere (incl. near-axis).
            let mut seed = 0x1234_5678u64;
            #[allow(clippy::cast_possible_truncation)]
            let mut rng = move || {
                seed = seed
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                // Top 24 bits → uniform [0, 1).
                ((seed >> 40) as u32) as f32 / 16_777_216.0
            };
            let mut hits = 0u32;
            for i in 0..4000u32 {
                let ox = rng() * 560.0 - 20.0;
                let oy = rng() * 160.0 - 16.0;
                let oz = rng() * 240.0;
                let dx = rng() * 2.0 - 1.0;
                let dy = rng() * 2.0 - 1.0;
                // Bias some rays to near-axis / horizontal (worst cases).
                let dz = match i % 5 {
                    0 => 0.0,
                    1 => 1e-6,
                    _ => rng() * 2.0 - 1.0,
                };
                let ow = DVec3::new(f64::from(ox), f64::from(oy), f64::from(oz));
                let dw = DVec3::new(f64::from(dx), f64::from(dy), f64::from(dz));
                if dw.length() < 1e-9 {
                    continue;
                }
                let max_t = if i % 3 == 0 { 96.0 } else { 512.0 };
                let dense = occluded_dense(g, ow, dw, max_t);
                let skip = occluded_in_grid(g, ow, dw, max_t);
                assert_eq!(
                    skip, dense,
                    "ray {i}: o={ow:?} d={dw:?} max_t={max_t} skip={skip} dense={dense}",
                );
                hits += u32::from(dense);
            }
            hits
        };
        let hits_no_maps = run(&scene);

        // Phase 2: with mip-0 brick maps → super/brick jumps active.
        scene.grid_mut(gid).unwrap().ensure_dda_bricks(0);
        let hits_with_maps = run(&scene);
        assert_eq!(hits_no_maps, hits_with_maps, "map presence changed results");
        assert!(hits_with_maps > 200, "sweep should hit terrain often");
    }

    /// CA follow-up (Decks report) — the occluder must be
    /// **render-solid**: the zero-RGB bedrock placeholder at the bottom
    /// of never-written columns is invisible to the renderer, so it
    /// must not occlude either. Pre-fix, a stacked-chunk grid (the
    /// shiplet) cast a phantom 128×128 shadow plate from its chz=0
    /// chunk's placeholder floor. Real bedrock BELOW a coloured surface
    /// keeps occluding (the render draws its faces via the run-top
    /// fallback).
    #[test]
    fn placeholder_bedrock_does_not_occlude() {
        use roxlap_core::WorldOccluder;
        let mut scene = Scene::new();
        let ground = scene.add_grid(GridTransform::identity());
        scene.grid_mut(ground).unwrap().set_rect(
            IVec3::new(0, 0, 420),
            IVec3::new(255, 127, 440),
            Some(VoxColor(0x80_4A_5E_3C)),
        );
        // A ship-like grid above the ground: one small roof plate high
        // in chunk (0,0,0) — the REST of that chunk's columns keep the
        // invisible placeholder voxel at local z=255 (world 361).
        let ship = scene.add_grid(GridTransform::at(DVec3::new(0.0, 0.0, 106.0)));
        scene.grid_mut(ship).unwrap().set_rect(
            IVec3::new(24, 32, 240),
            IVec3::new(103, 95, 241),
            Some(VoxColor(0x80_62_66_70)),
        );
        let occ = SceneOccluder::build(&scene);
        let to_sun = [-0.4506, -0.3505, -0.8211];
        // Under the plate's true shadow: occluded.
        assert!(
            occ.occluded_world([120.0, 75.0, 418.5], to_sun, 200.0),
            "the real roof plate must occlude"
        );
        // A ray that only crosses PLACEHOLDER columns of the ship's
        // chunk (misses the plate): must NOT occlude — pre-fix this
        // reported the phantom plate at world z=361.
        assert!(
            !occ.occluded_world([141.6, 26.7, 418.5], to_sun, 200.0),
            "the invisible bedrock placeholder must not occlude"
        );
        // Real bedrock below a coloured surface still occludes: aim a
        // shallow ray through the ground plate's own interior.
        assert!(
            occ.occluded_world([130.0, 60.0, 435.0], [1.0, 0.0, 0.0], 64.0),
            "coloured-surface bedrock interiors keep occluding"
        );
    }

    /// CA.2 — cross-grid shadow rays apply each TARGET grid's own
    /// cutaway clip: a clipped-away slab stops occluding while another
    /// grid's (unclipped) slab keeps blocking the same ray, and a clip
    /// that hides a whole grid drops it from the occluder outright.
    #[test]
    fn cutaway_clip_applies_per_grid() {
        let mut scene = Scene::new();
        // Two overlapping identity grids: slab A at z 100..110,
        // slab B at z 180..190.
        let ga = scene.add_grid(GridTransform::identity());
        scene.grid_mut(ga).unwrap().set_rect(
            IVec3::new(0, 0, 100),
            IVec3::new(127, 127, 110),
            Some(VoxColor(0x80_55_66_77)),
        );
        let gb = scene.add_grid(GridTransform::identity());
        scene.grid_mut(gb).unwrap().set_rect(
            IVec3::new(0, 0, 180),
            IVec3::new(127, 127, 190),
            Some(VoxColor(0x80_77_66_55)),
        );
        // A "sun ray" from below both slabs, straight up (z-down: -z).
        let occluded = |scene: &Scene| {
            use roxlap_core::WorldOccluder;
            SceneOccluder::build(scene).occluded_world([50.0, 50.0, 250.0], [0.0, 0.0, -1.0], 512.0)
        };
        assert!(occluded(&scene), "both slabs visible: blocked");
        // Hide slab A only — slab B (its own clip unset) still blocks.
        scene.grid_mut(ga).unwrap().z_clip = Some(150);
        assert!(occluded(&scene), "B is unclipped and must keep blocking");
        // Hide slab B too (plane below it) — the ray reaches the sun.
        scene.grid_mut(gb).unwrap().z_clip = Some(195);
        assert!(
            !occluded(&scene),
            "both slabs hidden: hidden geometry must not shadow"
        );
        // Per-grid independence the other way round: only B clipped.
        scene.grid_mut(ga).unwrap().z_clip = None;
        assert!(occluded(&scene), "A is unclipped and must keep blocking");
        // A clip below the whole grid box drops the grid from the
        // occluder entirely (the `lo >= hi` early-continue).
        scene.grid_mut(ga).unwrap().z_clip = Some(300);
        assert!(!occluded(&scene), "fully hidden grids never occlude");
    }
}
