//! Per-column slab walker used at the top of opticast to locate which
//! air gap contains the camera. Port of the `gstartv` walk in
//! `voxlaptest`'s `opticast` (`voxlap5.c:2314..2325`).
//!
//! Two entry points, layered:
//!
//! - [`camera_column_air_gap`] — operates on a raw column byte slice.
//!   Bit-stable since R4.3a-rewire-3; covered by the goldens.
//! - [`camera_chunk_air_gap`] (S4B.1) — chunk-aware wrapper: takes a
//!   [`crate::grid_view::GridView`] + the [`crate::opticast_prelude::
//!   OpticastPrelude`] and finds the air gap inside the chunk the
//!   camera sits in, OR synthesises the OOB-XY bedrock seed when the
//!   camera is past the grid's AABB. Today's single-chunk grids
//!   degenerate this to `camera_column_air_gap` over the flat
//!   `column_offsets` table; the seam exists so S4B.2's
//!   cross-chunk-XY DDA can grow the chunk lookup without touching
//!   opticast.
//!
//! Voxlap stores each map column as a chain of slab records (see
//! `roxlap-formats::vxl`):
//!
//! ```text
//! slab header (4 bytes):
//!   byte 0  nextptr  — offset to next slab in dwords (0 = last)
//!   byte 1  z1       — top z of floor-colour list
//!   byte 2  z1c      — bottom z of floor-colour list MINUS 1
//!   byte 3  z0       — ceiling z (additional slabs); dummy in first
//! ```
//!
//! [`camera_column_air_gap`] walks the chain to find the air gap (the
//! `[z0, z1)` range with no voxels) the camera is in, or reports
//! `None` if the camera is inside solid voxel material (in which case
//! voxlap returns from opticast early — no render this frame).

/// Walk a column's slab list to find which air gap contains z = `cz`.
///
/// Returns:
/// - `Some((z0, z1, vptr_offset))` — voxlap's `gstartz0` /
///   `gstartz1` / `v - *ixy_sptr_col` triple. Camera sits in the
///   air gap with top z = `z0` and bottom z = `z1`. `vptr_offset`
///   is the byte offset within `column` to the slab whose top
///   bounds the air gap from below. `0` means the air gap is
///   above the first slab (z0 = 0); a non-zero value means the
///   camera is in an interior air gap and `vptr_offset` points to
///   the slab header below it. R4.3a-rewire-3 grouscan uses this
///   to dispatch the initial drawflor (top-of-column) vs drawceil
///   (interior) branch.
/// - `None` — camera is inside a slab (hidden interior or below the
///   final slab without the `treat_z_max_as_air` synthesis below);
///   opticast returns early with no render.
///
/// `treat_z_max_as_air`: when `true`, the bedrock placeholder slab
/// (z1 = 0xff = MAXZDIM-1 = 255) is treated as transparent. A camera
/// past every real slab synthesises an air gap whose top is the
/// previous slab's floor and whose bottom is the bedrock's top z.
/// Without this, a camera below the bedrock returns `None` and the
/// renderer reports `SkippedCameraInSolid` (entire frame stays at
/// the host-cleared sky colour).
#[must_use]
pub fn camera_column_air_gap(
    column: &[u8],
    cz: i32,
    treat_z_max_as_air: bool,
) -> Option<(i32, i32, usize)> {
    if column.len() < 4 {
        return None;
    }
    let first_z1 = i32::from(column[1]);
    if cz < first_z1 {
        // Air above the first slab — gstartz0 = 0 in voxlap's else
        // branch. vptr_offset = 0 marks "column-top" for grouscan.
        return Some((0, first_z1, 0));
    }

    // Track the previous slab's floor so the bedrock-as-air synthesis
    // below has a sensible "ceiling" for the synthetic gap.
    let mut pos = 0usize;
    let mut prev_floor_z1c = first_z1; // ceiling defaults to first slab's top
    loop {
        let nextptr = column[pos];
        if nextptr == 0 {
            // Reached the last slab without finding an air gap above
            // the camera. Without treat_z_max_as_air this matches
            // voxlap C: returns None → opticast bails as
            // SkippedCameraInSolid.
            //
            // With treat_z_max_as_air on AND the last slab is the
            // bedrock placeholder (z1 == 0xff), the bedrock IS air, so
            // a camera past every solid slab is in air below the
            // world. Synthesise an air gap so grouscan can render the
            // visible scene above the camera. The gap's ceiling is
            // the previous slab's floor (where solids are still real);
            // its floor is the bedrock placeholder's z1 (= 255). The
            // vptr offset stays at the bedrock slab so drawcwall /
            // drawceil's `v[-4]` lookups land in the previous slab's
            // last colour entry (the floor's underside colour).
            if treat_z_max_as_air {
                let last_z1 = i32::from(column[pos + 1]);
                if last_z1 == 0xff {
                    return Some((prev_floor_z1c, last_z1, pos));
                }
            }
            return None;
        }
        // z1c = bottom of floor-colour list - 1 (= last solid voxel
        // z of the slab + 1 == ceiling z of the air gap below this
        // slab). Used as the bedrock-air synthesis's ceiling.
        prev_floor_z1c = i32::from(column[pos + 2]) + 1;
        pos = pos.checked_add(usize::from(nextptr) * 4)?;
        // Need 4 header bytes at the new position.
        if pos.checked_add(4)? > column.len() {
            return None;
        }
        let z1 = i32::from(column[pos + 1]);
        if cz < z1 {
            let z0 = i32::from(column[pos + 3]);
            if cz < z0 {
                // Hidden interior between this slab's ceiling colours
                // and floor colours — no render.
                return None;
            }
            return Some((z0, z1, pos));
        }
    }
}

/// Chunk-aware variant of [`camera_column_air_gap`]. Given the
/// per-frame grid view and the prelude's chunk-local camera info,
/// find the air gap inside the column the camera sits in (or
/// synthesise an air seed when the camera is past the grid's AABB).
///
/// Returns `(gstartz0, gstartz1, camera_vptr_offset, seed_chz)`:
/// the first three are voxlap's `gstartz0` / `gstartz1` /
/// `camera_vptr_offset` (now in **world-z** post-S4B.6.c), and
/// `seed_chz` is the chunk-z of the chunk whose column owns the
/// `vptr_offset`. For the unstacked path or a camera in a chunk
/// with real terrain `seed_chz == camera_chunk_idx[2]` (byte-
/// identical to pre-S4B.6.e). When the camera's column is fully
/// air-bedrock and `treat_z_max_as_air` is set, this walks the
/// chunk stack downward (chz+1, chz+2, …) and returns the first
/// chunk whose column has a real floor — that's `seed_chz`. The
/// caller then seeds the rasterizer with `seed_chz`'s chunk data
/// (`state.column` / `state.slab_buf` / etc.) so rays start
/// walking the real-terrain chunk directly. See
/// `project_s4b_6_e_landed.md` for the cross-chunk look-down case
/// this unblocks.
///
/// Returns `None` only when the camera is inside solid voxel
/// material — OOB-XY cameras are treated as "in air outside the
/// grid" and return the bedrock placeholder seed `(0, world_z_max,
/// 0, origin_chunk_z)`.
#[must_use]
pub fn camera_chunk_air_gap(
    grid: crate::grid_view::GridView<'_>,
    prelude: &crate::opticast_prelude::OpticastPrelude,
    treat_z_max_as_air: bool,
) -> Option<(i32, i32, usize, i32)> {
    if !prelude.in_bounds_xy {
        // OOB-XY camera: synthesise the bedrock placeholder seed
        // covering the grid's full world-z extent. `vptr=0` is a
        // placeholder; the cf seed's `i0/i1` range gets drained by
        // `phase_startsky` so no slab data is read from `column[]`
        // anyway. `seed_chz = origin_chunk_z` (= top chunk of the
        // stack) so callers route to that chunk's data — which is
        // not read anyway for OOB-XY, but keeps the indexing
        // canonical.
        let chunks_z_signed = grid.chunk_grid.map_or(1, |cg| cg.chunks_z) as i32;
        let seed_chz = grid.chunk_grid.map_or(0, |cg| cg.origin_chunk_z);
        #[allow(clippy::cast_possible_wrap)]
        let world_z_max = chunks_z_signed * grid.chunk_size_z as i32 - 1;
        return Some((0, world_z_max, 0, seed_chz));
    }

    let chunks_z = grid.chunk_grid.map_or(1, |cg| cg.chunks_z) as i32;
    let origin_chunk_z = grid.chunk_grid.map_or(0, |cg| cg.origin_chunk_z);
    let max_chz = origin_chunk_z + chunks_z - 1;
    let camera_chz = prelude.camera_chunk_idx[2];
    #[allow(clippy::cast_possible_wrap)]
    let chunk_size_z_signed = grid.chunk_size_z as i32;

    // S4B.6.e: walk the chunk stack downward when the camera's
    // column reports "air all the way to bedrock-as-air" — i.e. the
    // column-top air gap reaches the chunk's bedrock placeholder
    // and there's a chunk below it. Each iteration after the first
    // re-queries the air gap with `cz_local = 0` (camera is "above"
    // that chunk for the purposes of the air-gap walk).
    let mut chz = camera_chz;
    let mut z0_world: Option<i32> = None;
    loop {
        let chunk = grid.chunk_at_xyz([
            prelude.camera_chunk_idx[0],
            prelude.camera_chunk_idx[1],
            chz,
        ])?;
        #[allow(clippy::cast_sign_loss)]
        let column_idx_in_chunk = (prelude.camera_local_xyz[1] as u32)
            .wrapping_mul(chunk.chunk_size_xy)
            .wrapping_add(prelude.camera_local_xyz[0] as u32);
        let column = crate::opticast::camera_column_slice(
            chunk.slab_buf,
            chunk.column_offsets,
            column_idx_in_chunk,
        )?;
        let cz_local = if chz == camera_chz {
            prelude.camera_local_xyz[2]
        } else {
            0
        };
        let (local_z0, local_z1, vptr) =
            camera_column_air_gap(column, cz_local, treat_z_max_as_air)?;
        let chunk_world_z_base = chz * chunk_size_z_signed;
        // Lock z0_world on the first iteration only — subsequent
        // chunks contribute only their floor (z1) to the gap.
        let z0w = *z0_world.get_or_insert(local_z0 + chunk_world_z_base);
        // All-air-bedrock-from-top detection: column-top air gap
        // (vptr==0) ends at the bedrock placeholder (z1==0xff). With
        // `treat_z_max_as_air` the gap extends past the chunk; if a
        // chunk below exists, look at its top to find the real
        // floor.
        if local_z1 == 0xff && vptr == 0 && treat_z_max_as_air && chz < max_chz {
            chz += 1;
            continue;
        }
        return Some((z0w, local_z1 + chunk_world_z_base, vptr, chz));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Single solid slab at z = 5..15 (inclusive of bottom). Last slab,
    /// no air-gap below it. Header [0, 5, 14, 0].
    fn single_slab_5_15() -> Vec<u8> {
        // 4-byte header. Floor colours would be (z1c - z1 + 1) = 10
        // entries × 4 bytes = 40 bytes; we omit them here because the
        // walker only reads the header.
        vec![0, 5, 14, 0]
    }

    /// Two slabs with an air gap at z = 20..30.
    ///
    /// Slab 0 (first): visible block at z = 10..14, 5 voxels.
    ///   header = [nextptr=6, z1=10, z1c=14, dummy=0]
    ///   5 colour records × 4 bytes = 20 bytes
    ///   total slab size = nextptr × 4 = 24 bytes ✓
    ///
    /// Slab 1 (last): visible block at z = 30..39, 10 voxels above
    /// the air starting at z = 20.
    ///   header = [nextptr=0, z1=30, z1c=39, z0=20]
    ///   10 colour records × 4 bytes = 40 bytes
    fn two_slabs_air_at_20_30() -> Vec<u8> {
        let mut col = Vec::new();
        // Slab 0 header.
        col.extend_from_slice(&[6, 10, 14, 0]);
        // 5 colour records (content irrelevant — the walker doesn't read).
        col.resize(col.len() + 5 * 4, 0xab);
        // Slab 1 header.
        col.extend_from_slice(&[0, 30, 39, 20]);
        // 10 colour records.
        col.resize(col.len() + 10 * 4, 0xcd);
        col
    }

    #[test]
    fn camera_above_first_slab_returns_zero_to_z1() {
        let col = single_slab_5_15();
        assert_eq!(camera_column_air_gap(&col, 0, false), Some((0, 5, 0)));
        assert_eq!(camera_column_air_gap(&col, 4, false), Some((0, 5, 0)));
    }

    #[test]
    fn camera_inside_solid_returns_none() {
        let col = single_slab_5_15();
        // cz = 10 is inside the only slab. We walk forward, hit
        // nextptr = 0, return None.
        assert_eq!(camera_column_air_gap(&col, 10, false), None);
        // cz = 100 (way past the column) does the same thing.
        assert_eq!(camera_column_air_gap(&col, 100, false), None);
    }

    #[test]
    fn camera_above_first_slab_in_two_slab_column() {
        let col = two_slabs_air_at_20_30();
        // cz = 5 is above slab 0's z1 = 10.
        // Column-top — vptr_offset = 0.
        assert_eq!(camera_column_air_gap(&col, 5, false), Some((0, 10, 0)));
    }

    #[test]
    fn camera_in_air_gap_between_slabs() {
        let col = two_slabs_air_at_20_30();
        // cz = 25 is in the [20, 30) gap above slab 1.
        // Interior gap — vptr_offset points to slab 1's header.
        // Slab 0's nextptr = 6 → 24-byte stride (6 × 4); slab 1
        // starts at column[24].
        assert_eq!(camera_column_air_gap(&col, 25, false), Some((20, 30, 24)));
        assert_eq!(camera_column_air_gap(&col, 20, false), Some((20, 30, 24)));
        assert_eq!(camera_column_air_gap(&col, 29, false), Some((20, 30, 24)));
    }

    #[test]
    fn camera_in_first_slabs_hidden_interior_returns_none() {
        let col = two_slabs_air_at_20_30();
        // cz = 12 is inside slab 0 (z1 = 10..14). The walker advances
        // to slab 1, which has z1 = 30 (cz < z1) and z0 = 20 (cz < z0)
        // → hidden interior → None.
        assert_eq!(camera_column_air_gap(&col, 12, false), None);
    }

    #[test]
    fn camera_in_last_slab_returns_none() {
        let col = two_slabs_air_at_20_30();
        // cz = 35 is inside slab 1 (z1 = 30..39). The walker advances
        // to slab 1, sees cz >= z1 = 30, hits nextptr = 0 → None.
        assert_eq!(camera_column_air_gap(&col, 35, false), None);
        // cz = 1000 — same path.
        assert_eq!(camera_column_air_gap(&col, 1000, false), None);
    }

    #[test]
    fn malformed_too_short_returns_none() {
        // Less than a single header.
        assert_eq!(camera_column_air_gap(&[1, 2, 3], 10, false), None);
    }

    #[test]
    fn malformed_nextptr_overruns_returns_none() {
        // Single slab with nextptr = 99 but no following slab.
        let col = vec![99, 10, 14, 0];
        // cz = 100 forces walk-forward; nextptr advances past EOF.
        assert_eq!(camera_column_air_gap(&col, 100, false), None);
    }

    /// S4B.1: camera_chunk_air_gap with a single-chunk GridView
    /// matches camera_column_air_gap on the column the prelude's
    /// `column_index` points at. Validates the degenerate path —
    /// today's only path — so the goldens stay byte-identical.
    #[test]
    fn camera_chunk_air_gap_inbounds_matches_column_path() {
        // Single-column world (vsid = 1), column at index 0 is the
        // two-slab fixture from above. column_index = 0,
        // mip_base = [0, 2] (2 entries: column 0 start + sentinel).
        let col = two_slabs_air_at_20_30();
        let column_offsets = [0u32, col.len() as u32];
        let mip_base = [0usize, column_offsets.len()];
        let grid = crate::grid_view::GridView::from_parts(1, &col, &column_offsets, &mip_base);
        // Hand-built prelude with the fields camera_chunk_air_gap
        // reads.
        let prelude = crate::opticast_prelude::OpticastPrelude {
            forward_z_sign: 1,
            li_pos: [0, 0, 25], // camera in the [20, 30) air gap
            column_index: 0,
            pos_xfrac: [1.0, 0.0],
            pos_yfrac: [1.0, 0.0],
            pos_z: 0,
            y_lookup: Vec::new(),
            mip_levels: 1,
            x_mip: 0,
            max_scan_dist: 0,
            cx: 0,
            cy: 0,
            in_bounds_xy: true,
            camera_chunk_idx: [0, 0, 0],
            camera_local_xyz: [0, 0, 25],
        };
        let chunk_path = camera_chunk_air_gap(grid, &prelude, false);
        let column_path = camera_column_air_gap(&col, 25, false);
        // chunk_path adds seed_chz to the tuple; strip it for the
        // column_path comparison.
        assert_eq!(chunk_path.map(|(z0, z1, vp, _)| (z0, z1, vp)), column_path);
        assert_eq!(chunk_path, Some((20, 30, 24, 0)));
    }

    /// S4B.1: camera_chunk_air_gap synthesises the OOB-XY bedrock
    /// seed when `prelude.in_bounds_xy = false`, independent of the
    /// grid's column data. The OOB path used to live inline in
    /// `opticast`; the wrapper owns it now.
    #[test]
    fn camera_chunk_air_gap_oob_synthesises_bedrock_seed() {
        // Empty grid — wrapper must not touch the slab buffer.
        let mip_base = [0usize, 0];
        let grid = crate::grid_view::GridView::from_parts(1, &[], &[], &mip_base);
        let prelude = crate::opticast_prelude::OpticastPrelude {
            forward_z_sign: 1,
            li_pos: [-5, 0, 30],
            column_index: u32::MAX, // junk: as u32 of -5; never read on OOB path
            pos_xfrac: [1.0, 0.0],
            pos_yfrac: [1.0, 0.0],
            pos_z: 0,
            y_lookup: Vec::new(),
            mip_levels: 1,
            x_mip: 0,
            max_scan_dist: 0,
            cx: -5,
            cy: 0,
            in_bounds_xy: false,
            camera_chunk_idx: [0, 0, 0],
            camera_local_xyz: [-5, 0, 30],
        };
        assert_eq!(
            camera_chunk_air_gap(grid, &prelude, true),
            Some((0, 255, 0, 0))
        );
    }

    /// S4B.1: camera-in-solid returns None so opticast can bail.
    /// Bridges the existing single-slab fixture through the new
    /// wrapper.
    #[test]
    fn camera_chunk_air_gap_inbounds_in_solid_returns_none() {
        let col = single_slab_5_15();
        let column_offsets = [0u32, col.len() as u32];
        let mip_base = [0usize, column_offsets.len()];
        let grid = crate::grid_view::GridView::from_parts(1, &col, &column_offsets, &mip_base);
        let prelude = crate::opticast_prelude::OpticastPrelude {
            forward_z_sign: 1,
            li_pos: [0, 0, 10], // inside the solid slab z = 5..15
            column_index: 0,
            pos_xfrac: [1.0, 0.0],
            pos_yfrac: [1.0, 0.0],
            pos_z: 0,
            y_lookup: Vec::new(),
            mip_levels: 1,
            x_mip: 0,
            max_scan_dist: 0,
            cx: 0,
            cy: 0,
            in_bounds_xy: true,
            camera_chunk_idx: [0, 0, 0],
            camera_local_xyz: [0, 0, 10],
        };
        assert_eq!(camera_chunk_air_gap(grid, &prelude, false), None);
    }

    /// S4B.6.e: cross-chunk look-down. Camera in chz=0 with an
    /// all-air-bedrock column should walk into chz+1 and seed
    /// against chz+1's real floor. The returned `seed_chz` lets
    /// the caller swap the rasterizer's chunk borrow to chz+1.
    #[test]
    fn camera_chunk_air_gap_cross_chunk_lookdown_to_chz1_floor() {
        use crate::grid_view::{ChunkGrid, GridView};

        // chz=0: bedrock-only column (all-air-bedrock).
        let col_chz0: Vec<u8> = vec![0, 0xff, 0xff, 0];
        // chz=1: floor at local z=50 (= world z=306).
        let mut col_chz1 = Vec::new();
        col_chz1.extend_from_slice(&[2, 50, 50, 0]); // floor header
        col_chz1.extend_from_slice(&[0xcc; 4]); // floor colour
        col_chz1.extend_from_slice(&[0, 0xff, 0xff, 51]); // bedrock

        let cols0 = [0u32, col_chz0.len() as u32];
        let cols1 = [0u32, col_chz1.len() as u32];
        let mips0 = [0usize, cols0.len()];
        let mips1 = [0usize, cols1.len()];
        let c0 = GridView::from_parts(1, &col_chz0, &cols0, &mips0);
        let c1 = GridView::from_parts(1, &col_chz1, &cols1, &mips1);
        let chunks = [Some(c0), Some(c1)];
        let cg = ChunkGrid {
            chunks: &chunks,
            origin_chunk_xy: [0, 0],
            origin_chunk_z: 0,
            chunks_x: 1,
            chunks_y: 1,
            chunks_z: 2,
        };
        let parent = GridView::from_chunk_grid(&cg, 1);

        // Camera at world z=100 in chz=0's all-air column.
        let prelude = crate::opticast_prelude::OpticastPrelude {
            forward_z_sign: 1,
            li_pos: [0, 0, 100],
            column_index: 0,
            pos_xfrac: [1.0, 0.0],
            pos_yfrac: [1.0, 0.0],
            pos_z: 0,
            y_lookup: Vec::new(),
            mip_levels: 1,
            x_mip: 0,
            max_scan_dist: 0,
            cx: 0,
            cy: 0,
            in_bounds_xy: true,
            camera_chunk_idx: [0, 0, 0],
            camera_local_xyz: [0, 0, 100],
        };
        // Without treat_z_max_as_air: walk stops in chz=0 with the
        // bedrock seed unchanged → seed_chz==0.
        assert_eq!(
            camera_chunk_air_gap(parent, &prelude, false),
            Some((0, 0xff, 0, 0))
        );
        // With treat_z_max_as_air: walk extends into chz=1, finds
        // floor at local z=50 (= world z=306), vptr=0 (top of
        // chz=1's column), seed_chz=1.
        assert_eq!(
            camera_chunk_air_gap(parent, &prelude, true),
            Some((0, 306, 0, 1))
        );
    }

    /// Floor + bedrock placeholder: cz past the bedrock returns None
    /// without `treat_z_max_as_air`, but synthesises an air gap when
    /// the flag is on. Mirrors the test scene used by
    /// `outside_pentagon_repro::below_bedrock_camera_renders_world`.
    #[test]
    fn below_bedrock_synthesises_air_when_flag_set() {
        // Slab 0: 1-voxel floor at z = 200. nextptr = 2 (header + 1
        // colour record = 8 bytes = 2 dwords). z1 = z1c = 200.
        // Slab 1: bedrock placeholder at z = 255. nextptr = 0 (last).
        let mut col = Vec::new();
        col.extend_from_slice(&[2, 200, 200, 0]); // floor header
        col.extend_from_slice(&[0xff; 4]); // floor colour
        col.extend_from_slice(&[0, 255, 255, 201]); // bedrock header
        col.extend_from_slice(&[0xff; 4]); // bedrock colour
                                           // Without flag — voxlap-canonical None.
        assert_eq!(camera_column_air_gap(&col, 261, false), None);
        // With flag — synthesises (z0=201, z1=255, vptr=8).
        // z0 = floor's z1c+1 = 201 (top of synthetic gap, just below
        // floor's last solid voxel).
        // z1 = bedrock's z1 = 255.
        // vptr = bedrock's offset = 8 (so drawceil reads
        // column[vptr-4..vptr] = floor's colour).
        assert_eq!(camera_column_air_gap(&col, 261, true), Some((201, 255, 8)));
    }

    /// S4B.6.b: in a vertically-stacked grid (`chunks_z > 1`),
    /// `camera_chunk_air_gap` queries the chunk at the camera's
    /// actual chz. Two chunks at chz=0 and chz=1 each carry their
    /// own column; the air-gap result reflects the camera's z
    /// layer.
    #[test]
    fn camera_chunk_air_gap_dispatches_by_camera_chz() {
        use crate::grid_view::{ChunkGrid, GridView};

        // Two chunks (one column each), both with a floor at
        // local z=50. Distinct colour bytes so we can verify which
        // chunk's column was walked.
        let mut col_chz0 = Vec::new();
        col_chz0.extend_from_slice(&[2, 50, 50, 0]); // floor header
        col_chz0.extend_from_slice(&[0xaa; 4]); // distinct colour
        let mut col_chz1 = Vec::new();
        col_chz1.extend_from_slice(&[2, 50, 50, 0]); // floor header
        col_chz1.extend_from_slice(&[0xbb; 4]); // distinct colour

        let cols0 = [0u32, col_chz0.len() as u32];
        let cols1 = [0u32, col_chz1.len() as u32];
        let mips = [0usize, cols0.len()];
        let c0 = GridView::from_parts(1, &col_chz0, &cols0, &mips);
        let c1 = GridView::from_parts(1, &col_chz1, &cols1, &mips);
        let chunks = [Some(c0), Some(c1)];
        let cg = ChunkGrid {
            chunks: &chunks,
            origin_chunk_xy: [0, 0],
            origin_chunk_z: 0,
            chunks_x: 1,
            chunks_y: 1,
            chunks_z: 2,
        };
        let parent = GridView::from_chunk_grid(&cg, 1);

        // Camera in chz=0 at local z=20 (= world z=20) — air gap
        // (0, 50, 0) within chz=0's column.
        let prelude_chz0 = crate::opticast_prelude::OpticastPrelude {
            forward_z_sign: 1,
            li_pos: [0, 0, 20],
            column_index: 0,
            pos_xfrac: [1.0, 0.0],
            pos_yfrac: [1.0, 0.0],
            pos_z: 0,
            y_lookup: Vec::new(),
            mip_levels: 1,
            x_mip: 0,
            max_scan_dist: 0,
            cx: 0,
            cy: 0,
            in_bounds_xy: true,
            camera_chunk_idx: [0, 0, 0],
            camera_local_xyz: [0, 0, 20],
        };
        let gap0 = camera_chunk_air_gap(parent, &prelude_chz0, false);
        // S4B.6.c: cf seed returns world-z. For chz=0 (chunk-z
        // base = 0), world-z == chunk-local. S4B.6.e: seed_chz==0
        // (camera's column has a real floor in chz=0).
        assert_eq!(gap0, Some((0, 50, 0, 0)));

        // Camera in chz=1 at world z=280 (= local z=24 within
        // chz=1's chunk). camera_chunk_idx=[0, 0, 1].
        // camera_local_xyz=[0, 0, 24]. Air gap chunk-local
        // (0, 50, 0) translates to world-z (256, 306, 0).
        let prelude_chz1 = crate::opticast_prelude::OpticastPrelude {
            forward_z_sign: 1,
            li_pos: [0, 0, 280],
            column_index: 0,
            pos_xfrac: [1.0, 0.0],
            pos_yfrac: [1.0, 0.0],
            pos_z: 0,
            y_lookup: Vec::new(),
            mip_levels: 1,
            x_mip: 0,
            max_scan_dist: 0,
            cx: 0,
            cy: 0,
            in_bounds_xy: true,
            camera_chunk_idx: [0, 0, 1],
            camera_local_xyz: [0, 0, 24],
        };
        let gap1 = camera_chunk_air_gap(parent, &prelude_chz1, false);
        // S4B.6.e: seed_chz==1 (camera in chz=1, real floor there).
        assert_eq!(gap1, Some((256, 306, 0, 1)));

        // Confirm the chunk dispatch is correct by reading slab
        // bytes via the chunk-z-specific view.
        let cv0 = parent.chunk_at_xyz([0, 0, 0]).unwrap();
        let cv1 = parent.chunk_at_xyz([0, 0, 1]).unwrap();
        assert_eq!(cv0.slab_buf, &col_chz0[..]);
        assert_eq!(cv1.slab_buf, &col_chz1[..]);
    }
}
