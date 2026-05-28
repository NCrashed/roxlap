//! Multi-chunk edit API on [`Grid`].
//!
//! Region-shape edit operations ([`Grid::set_voxel`],
//! [`Grid::set_rect`], [`Grid::set_sphere`]) take grid-local voxel
//! coordinates spanning any number of chunks. The implementation:
//!
//! 1. Computes the chunk index range the operation touches.
//! 2. For each chunk in that range, intersects the edit region
//!    with the chunk's voxel footprint and translates to
//!    chunk-local coordinates.
//! 3. Calls the corresponding [`roxlap_formats::edit`] primitive
//!    with the per-chunk slice of the edit.
//!
//! Implicit-air chunks are materialised on demand via
//! [`Grid::ensure_chunk`] when the operation inserts voxels
//! (`color = Some(_)`); pure-carve operations (`color = None`)
//! skip materialisation for chunks that don't already exist —
//! carving from already-air voxels is a no-op.

use glam::IVec3;
use roxlap_formats::edit::{set_cube, set_rect, set_sphere};

use crate::addr::{voxel_split, GridLocalPos};
use crate::{Grid, CHUNK_SIZE_XY, CHUNK_SIZE_Z};

/// Per-axis chunk size as an [`IVec3`]. Duplicated from
/// [`crate::addr`]'s private helper; kept local because exposing
/// it would leak an implementation detail.
#[inline]
fn chunk_size_ivec3() -> IVec3 {
    #[allow(clippy::cast_possible_wrap)]
    IVec3::new(
        CHUNK_SIZE_XY as i32,
        CHUNK_SIZE_XY as i32,
        CHUNK_SIZE_Z as i32,
    )
}

impl Grid {
    /// Set or carve a single voxel at grid-local coordinate
    /// `voxel`. `color = Some(c)` inserts a solid voxel of colour
    /// `c`; `color = None` carves to air.
    ///
    /// Inserting in an implicit-air chunk materialises that chunk
    /// (allocates a fresh [`Vxl`]); carving from a missing chunk
    /// is a no-op.
    ///
    /// [`Vxl`]: roxlap_formats::vxl::Vxl
    pub fn set_voxel(&mut self, voxel: IVec3, color: Option<u32>) {
        // S6.2: any edit invalidates the billboard impostor cache;
        // S6.3 will rebuild on next Far-tier use.
        self.billboards = None;
        let (chunk_idx, in_chunk) = voxel_split(voxel);
        if color.is_some() {
            let vxl = self.ensure_chunk(chunk_idx);
            #[allow(clippy::cast_possible_wrap)]
            set_cube(
                vxl,
                in_chunk.x as i32,
                in_chunk.y as i32,
                in_chunk.z as i32,
                color,
            );
        } else if let Some(vxl) = self.chunks.get_mut(&chunk_idx) {
            #[allow(clippy::cast_possible_wrap)]
            set_cube(
                vxl,
                in_chunk.x as i32,
                in_chunk.y as i32,
                in_chunk.z as i32,
                None,
            );
        }
    }

    /// Set or carve an axis-aligned box `[lo, hi]` in grid-local
    /// voxel coordinates. Inclusive on both ends, like
    /// [`roxlap_formats::edit::set_rect`].
    ///
    /// The box is decomposed per chunk: each touched chunk receives
    /// a `set_rect` call with the box clipped to its footprint and
    /// translated to chunk-local. Inserts materialise missing
    /// chunks; carves skip them.
    ///
    /// `lo` and `hi` may be in any order on each axis — the
    /// decomposition normalises them.
    pub fn set_rect(&mut self, lo: IVec3, hi: IVec3, color: Option<u32>) {
        // S6.2: edit invalidates billboard cache (see set_voxel doc).
        self.billboards = None;
        let lo_n = lo.min(hi);
        let hi_n = lo.max(hi);
        let (lo_c, _) = voxel_split(lo_n);
        let (hi_c, _) = voxel_split(hi_n);
        let cs = chunk_size_ivec3();

        for cz in lo_c.z..=hi_c.z {
            for cy in lo_c.y..=hi_c.y {
                for cx in lo_c.x..=hi_c.x {
                    let chunk_idx = IVec3::new(cx, cy, cz);
                    let chunk_origin = chunk_idx * cs;
                    let chunk_end = chunk_origin + cs - IVec3::ONE;
                    let local_lo = lo_n.max(chunk_origin) - chunk_origin;
                    let local_hi = hi_n.min(chunk_end) - chunk_origin;
                    apply_set_rect(self, chunk_idx, local_lo, local_hi, color);
                }
            }
        }
    }

    /// Set or carve a sphere of voxels at grid-local centre
    /// `centre` with the given `radius`. Euclidean distance, like
    /// [`roxlap_formats::edit::set_sphere`].
    ///
    /// The bounding box of the sphere is enumerated chunk by chunk;
    /// each touched chunk receives a `set_sphere` call with the
    /// centre re-expressed in chunk-local coords (the per-chunk
    /// call clips the sphere to the chunk's footprint internally).
    /// Chunks the sphere doesn't actually reach get materialised
    /// only if `color.is_some()` and they fall within the AABB —
    /// the per-chunk `set_sphere` is a no-op for non-overlapping
    /// chunks but the materialisation cost remains. A subsequent
    /// pre-pass that filters chunks against `radius²` could avoid
    /// this; out of scope for v1.
    pub fn set_sphere(&mut self, centre: IVec3, radius: u32, color: Option<u32>) {
        // S6.2: edit invalidates billboard cache (see set_voxel doc).
        self.billboards = None;
        #[allow(clippy::cast_possible_wrap)]
        let r_i = radius as i32;
        let lo = centre - IVec3::splat(r_i);
        let hi = centre + IVec3::splat(r_i);
        let (lo_c, _) = voxel_split(lo);
        let (hi_c, _) = voxel_split(hi);
        let cs = chunk_size_ivec3();

        for cz in lo_c.z..=hi_c.z {
            for cy in lo_c.y..=hi_c.y {
                for cx in lo_c.x..=hi_c.x {
                    let chunk_idx = IVec3::new(cx, cy, cz);
                    let chunk_origin = chunk_idx * cs;
                    let local_centre = centre - chunk_origin;
                    apply_set_sphere(self, chunk_idx, local_centre, radius, color);
                }
            }
        }
    }
}

fn apply_set_rect(
    grid: &mut Grid,
    chunk_idx: IVec3,
    local_lo: IVec3,
    local_hi: IVec3,
    color: Option<u32>,
) {
    if color.is_some() {
        let vxl = grid.ensure_chunk(chunk_idx);
        set_rect(vxl, local_lo.into(), local_hi.into(), color);
    } else if let Some(vxl) = grid.chunks.get_mut(&chunk_idx) {
        set_rect(vxl, local_lo.into(), local_hi.into(), None);
    }
}

fn apply_set_sphere(
    grid: &mut Grid,
    chunk_idx: IVec3,
    local_centre: IVec3,
    radius: u32,
    color: Option<u32>,
) {
    if color.is_some() {
        let vxl = grid.ensure_chunk(chunk_idx);
        set_sphere(vxl, local_centre.into(), radius, color);
    } else if let Some(vxl) = grid.chunks.get_mut(&chunk_idx) {
        set_sphere(vxl, local_centre.into(), radius, None);
    }
}

/// Convenience: forward a [`GridLocalPos`]-style decomposition
/// back to [`voxel_global`] for callers that already hold one.
/// Stays here rather than in [`crate::addr`] because it's only
/// useful in the edit-API call shape.
///
/// [`voxel_global`]: crate::addr::voxel_global
#[must_use]
pub fn voxel_at(local: &GridLocalPos) -> IVec3 {
    crate::addr::voxel_global(local.chunk, local.voxel)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunks::tests::voxel_is_solid;
    use crate::GridTransform;

    const TEST_COL: u32 = 0x80_aa_bb_cc;

    #[test]
    fn set_voxel_inserts_in_correct_chunk() {
        // Voxel at grid-local (5, 6, 7) sits in chunk (0, 0, 0)
        // at local (5, 6, 7).
        let mut g = Grid::new(GridTransform::identity());
        g.set_voxel(IVec3::new(5, 6, 7), Some(TEST_COL));
        let vxl = g.chunk(IVec3::ZERO).expect("chunk created");
        assert!(voxel_is_solid(vxl, 5, 6, 7));
        // Adjacent voxel still air.
        assert!(!voxel_is_solid(vxl, 5, 6, 8));
        assert_eq!(g.chunk_count(), 1);
    }

    #[test]
    fn set_voxel_negative_coords_use_neg_chunk() {
        // Voxel (-1, 0, 0) sits in chunk (-1, 0, 0) at local
        // (CHUNK_SIZE_XY - 1, 0, 0).
        let mut g = Grid::new(GridTransform::identity());
        g.set_voxel(IVec3::new(-1, 0, 0), Some(TEST_COL));
        assert!(g.chunk(IVec3::new(-1, 0, 0)).is_some());
        let vxl = g.chunk(IVec3::new(-1, 0, 0)).unwrap();
        assert!(voxel_is_solid(vxl, CHUNK_SIZE_XY - 1, 0, 0));
        // Chunk (0, 0, 0) was NOT created.
        assert!(g.chunk(IVec3::ZERO).is_none());
    }

    #[test]
    fn set_voxel_carve_then_insert_round_trips() {
        let mut g = Grid::new(GridTransform::identity());
        g.set_voxel(IVec3::new(10, 10, 10), Some(TEST_COL));
        assert!(voxel_is_solid(g.chunk(IVec3::ZERO).unwrap(), 10, 10, 10));
        g.set_voxel(IVec3::new(10, 10, 10), None);
        assert!(!voxel_is_solid(g.chunk(IVec3::ZERO).unwrap(), 10, 10, 10));
    }

    #[test]
    fn set_voxel_carve_in_missing_chunk_is_noop() {
        // Carving in a chunk that doesn't exist should NOT create
        // it (it's already implicit-air; nothing to do).
        let mut g = Grid::new(GridTransform::identity());
        g.set_voxel(IVec3::new(5, 5, 5), None);
        assert_eq!(g.chunk_count(), 0);
    }

    #[test]
    fn set_rect_within_one_chunk() {
        let mut g = Grid::new(GridTransform::identity());
        g.set_rect(IVec3::new(0, 0, 0), IVec3::new(3, 3, 3), Some(TEST_COL));
        assert_eq!(g.chunk_count(), 1);
        let vxl = g.chunk(IVec3::ZERO).unwrap();
        for z in 0..=3 {
            for y in 0..=3 {
                for x in 0..=3 {
                    assert!(voxel_is_solid(vxl, x, y, z), "({x},{y},{z}) air");
                }
            }
        }
        // Just outside the rect.
        assert!(!voxel_is_solid(vxl, 4, 0, 0));
        assert!(!voxel_is_solid(vxl, 0, 4, 0));
        assert!(!voxel_is_solid(vxl, 0, 0, 4));
    }

    #[test]
    fn set_rect_spans_two_chunks_x() {
        // Box [(126, 0, 0) .. (129, 0, 0)] crosses the chunk-0 /
        // chunk-1 boundary on x at 128.
        let mut g = Grid::new(GridTransform::identity());
        g.set_rect(IVec3::new(126, 0, 0), IVec3::new(129, 0, 0), Some(TEST_COL));
        assert_eq!(g.chunk_count(), 2);

        // Chunk (0,0,0): voxels x=126, 127 at (y=0, z=0) solid.
        let v0 = g.chunk(IVec3::ZERO).unwrap();
        assert!(voxel_is_solid(v0, 126, 0, 0));
        assert!(voxel_is_solid(v0, 127, 0, 0));
        assert!(!voxel_is_solid(v0, 125, 0, 0));

        // Chunk (1,0,0): voxels x=0, 1 at (y=0, z=0) solid.
        let v1 = g.chunk(IVec3::new(1, 0, 0)).unwrap();
        assert!(voxel_is_solid(v1, 0, 0, 0));
        assert!(voxel_is_solid(v1, 1, 0, 0));
        assert!(!voxel_is_solid(v1, 2, 0, 0));
    }

    #[test]
    fn set_rect_spans_z_boundary() {
        // Box at z=255..256 crosses chunk boundary on z (256 = 1
        // chunk on z-axis).
        let mut g = Grid::new(GridTransform::identity());
        g.set_rect(IVec3::new(0, 0, 254), IVec3::new(0, 0, 257), Some(TEST_COL));
        assert_eq!(g.chunk_count(), 2);
        let v0 = g.chunk(IVec3::ZERO).unwrap();
        assert!(voxel_is_solid(v0, 0, 0, 254));
        assert!(voxel_is_solid(v0, 0, 0, 255));
        let v1 = g.chunk(IVec3::new(0, 0, 1)).unwrap();
        assert!(voxel_is_solid(v1, 0, 0, 0));
        assert!(voxel_is_solid(v1, 0, 0, 1));
        assert!(!voxel_is_solid(v1, 0, 0, 2));
    }

    #[test]
    fn set_rect_unsorted_lo_hi_normalised() {
        // Passing hi < lo should produce the same result as lo < hi.
        let mut g1 = Grid::new(GridTransform::identity());
        let mut g2 = Grid::new(GridTransform::identity());
        g1.set_rect(IVec3::new(0, 0, 0), IVec3::new(3, 3, 3), Some(TEST_COL));
        g2.set_rect(IVec3::new(3, 3, 3), IVec3::new(0, 0, 0), Some(TEST_COL));
        let v1 = g1.chunk(IVec3::ZERO).unwrap();
        let v2 = g2.chunk(IVec3::ZERO).unwrap();
        for z in 0..=3 {
            for y in 0..=3 {
                for x in 0..=3 {
                    assert_eq!(voxel_is_solid(v1, x, y, z), voxel_is_solid(v2, x, y, z));
                }
            }
        }
    }

    #[test]
    fn set_sphere_within_one_chunk() {
        let mut g = Grid::new(GridTransform::identity());
        g.set_sphere(IVec3::new(64, 64, 100), 5, Some(TEST_COL));
        assert_eq!(g.chunk_count(), 1);
        let vxl = g.chunk(IVec3::ZERO).unwrap();
        // Centre is solid.
        assert!(voxel_is_solid(vxl, 64, 64, 100));
        // 1 voxel from centre is solid (radius 5).
        assert!(voxel_is_solid(vxl, 65, 64, 100));
        assert!(voxel_is_solid(vxl, 64, 64, 105));
        // Just outside radius is air.
        assert!(!voxel_is_solid(vxl, 70, 64, 100));
    }

    #[test]
    fn set_sphere_spans_chunk_boundary() {
        // Centre at (127, 64, 100), radius 4 → reaches into chunk
        // (1,0,0) on the +x side.
        let mut g = Grid::new(GridTransform::identity());
        g.set_sphere(IVec3::new(127, 64, 100), 4, Some(TEST_COL));
        // 2 chunks: (0,0,0) and (1,0,0).
        assert_eq!(g.chunk_count(), 2);

        let v0 = g.chunk(IVec3::ZERO).unwrap();
        // (127, 64, 100) is the centre, in chunk 0 at local
        // (127, 64, 100).
        assert!(voxel_is_solid(v0, 127, 64, 100));
        // (124, 64, 100) is 3 voxels away, inside the sphere.
        assert!(voxel_is_solid(v0, 124, 64, 100));

        let v1 = g.chunk(IVec3::new(1, 0, 0)).unwrap();
        // Voxel (128, 64, 100) is centre + 1x → in chunk 1 at
        // local (0, 64, 100).
        assert!(voxel_is_solid(v1, 0, 64, 100));
        // Voxel (130, 64, 100) is 3 voxels from centre, still inside.
        assert!(voxel_is_solid(v1, 2, 64, 100));
    }

    // ---- S6.2: billboard cache invalidation ----

    /// Helper: stamp a sentinel billboard cache onto a grid so the
    /// invalidation tests can detect when the edit cleared it. We
    /// use a 32-resolution `new_empty` cache — populating real
    /// snapshots would work too but is needlessly expensive when
    /// the test only cares about Some/None state.
    fn stamp_sentinel_cache(g: &mut Grid) {
        g.billboards = Some(crate::BillboardCache::new_empty(32));
    }

    #[test]
    fn set_voxel_invalidates_billboard_cache() {
        let mut g = Grid::new(GridTransform::identity());
        stamp_sentinel_cache(&mut g);
        assert!(g.billboards.is_some());
        g.set_voxel(IVec3::new(5, 5, 5), Some(TEST_COL));
        assert!(
            g.billboards.is_none(),
            "set_voxel should clear the billboard cache"
        );
    }

    #[test]
    fn set_voxel_carve_also_invalidates() {
        // Even no-op carves invalidate — the cache must be conservative.
        let mut g = Grid::new(GridTransform::identity());
        stamp_sentinel_cache(&mut g);
        g.set_voxel(IVec3::new(5, 5, 5), None); // missing-chunk no-op
        assert!(
            g.billboards.is_none(),
            "carve should clear the cache (conservative)"
        );
    }

    #[test]
    fn set_rect_invalidates_billboard_cache() {
        let mut g = Grid::new(GridTransform::identity());
        stamp_sentinel_cache(&mut g);
        g.set_rect(IVec3::new(0, 0, 0), IVec3::new(3, 3, 3), Some(TEST_COL));
        assert!(g.billboards.is_none(), "set_rect should clear the cache");
    }

    #[test]
    fn set_sphere_invalidates_billboard_cache() {
        let mut g = Grid::new(GridTransform::identity());
        stamp_sentinel_cache(&mut g);
        g.set_sphere(IVec3::new(64, 64, 100), 5, Some(TEST_COL));
        assert!(g.billboards.is_none(), "set_sphere should clear the cache");
    }

    #[test]
    fn set_voxel_dispatches_to_correct_chunk_on_y_z_axes() {
        // Sanity check the y / z chunk dispatches use the right
        // chunk size (XY=128, Z=256). voxel (200, 300, 500)
        // should go to chunk (1, 2, 1), local (72, 44, 244).
        let mut g = Grid::new(GridTransform::identity());
        g.set_voxel(IVec3::new(200, 300, 500), Some(TEST_COL));
        let vxl = g
            .chunk(IVec3::new(1, 2, 1))
            .expect("expected chunk (1, 2, 1)");
        assert!(voxel_is_solid(vxl, 72, 44, 244));
    }
}
