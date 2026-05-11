//! Per-grid "combined" virtual world — Approach C from the S4
//! kickoff. Stitches a [`Grid`]'s sparse chunk map into a single
//! `(slab_buf, column_offsets, vsid)` triple shaped exactly like
//! the `(Vxl::data, Vxl::column_offset, Vxl::vsid)` the per-chunk
//! renderer already accepts.
//!
//! ## Why this works
//!
//! Voxlap's per-column slab format encodes the next-slab pointer
//! as a **delta in dwords within the column's own byte range**
//! (see `roxlap-formats::vxl` module docs and `slng`). Each
//! column's slab chain is therefore self-contained: copying the
//! bytes verbatim into a different buffer at a different offset
//! leaves the chain valid. We exploit that to materialise a
//! "virtual" world at `vsid = N_CHUNKS * CHUNK_SIZE_XY` whose
//! `column_offset[]` table indexes per-column slab bytes lifted
//! from each populated chunk.
//!
//! The 2D-DDA inside `roxlap-core::grouscan` already steps
//! one voxel-column at a time across an arbitrary `vsid × vsid`
//! lattice, so chunk **boundaries collapse into ordinary column
//! steps**. No engine change needed.
//!
//! ## S4.0 scope
//!
//! Single-z-chunk grids only — chunks at `chz != 0` are silently
//! ignored. Voxlap's slab format encodes z bounds as `u8`, so
//! stacking chunks vertically into one virtual column (a 2048-tall
//! slab list for 8 stacked chunks) doesn't fit the byte format.
//! Vertical-stack support comes later in S4.3 via a different path.
//!
//! Negative chunk indices are likewise out of S4.0 scope —
//! `column_index = vy * vsid + vx` requires `vx, vy ≥ 0`.
//!
//! ## Caching
//!
//! The combined view is cached on [`crate::Grid`] and rebuilt
//! lazily on the first [`crate::Grid::combined_world`] call after
//! any edit ([`crate::Grid::set_voxel`] / `set_rect` / `set_sphere`
//! / [`crate::Grid::ensure_chunk`] all invalidate). External
//! callers that mutate `chunks` directly (e.g. the lightmode bake
//! in `roxlap-scene-demo::scene`) must call
//! [`crate::Grid::invalidate_combined`] themselves once their
//! mutations finish.

use std::collections::HashMap;

use glam::IVec3;
use roxlap_formats::vxl::Vxl;

use crate::CHUNK_SIZE_XY;

/// Stitched per-grid view of every populated chunk's column data,
/// shaped to be passed straight to
/// [`roxlap_core::opticast::opticast`] alongside
/// [`roxlap_core::scalar_rasterizer::ScalarRasterizer`].
///
/// Mip-0 only at S4.0 — chunk-level mip switching lands in S6.
#[derive(Debug, Clone)]
pub struct CombinedGridView {
    /// World dimension. `vsid * vsid + 1 == column_offset.len()`.
    /// Always `>= CHUNK_SIZE_XY`. Square (voxlap requires it); for
    /// non-square chunk lattices the smaller axis is padded with
    /// all-air placeholder columns.
    pub vsid: u32,
    /// Concatenated per-column slab bytes.
    pub data: Vec<u8>,
    /// Per-column byte offsets into [`Self::data`]. Length
    /// `vsid * vsid + 1`, trailing sentinel equals `data.len()`.
    pub column_offset: Vec<u32>,
    /// Single-mip `[0, vsid² + 1]` boundary table — single entry of
    /// state needed by `ScalarRasterizer::new`.
    pub mip_base_offsets: Vec<usize>,
}

/// All-air column placeholder bytes. Mirrors what
/// [`crate::chunks::empty_chunk_vxl`] writes per column after the
/// full-z-range carve: a single bedrock-placeholder slab at z=255.
///
/// Header: `[nextptr=0, z1=255, z1c=255, dummy=0]` + 1 colour byte
/// record (4 bytes). Total 8 bytes.
const ALL_AIR_COLUMN: [u8; 8] = [0, 255, 255, 0, 0, 0, 0, 0];

impl CombinedGridView {
    /// Build a fresh combined view from `chunks`. Linear in total
    /// virtual-column count (`vsid²`) and total slab byte length;
    /// expensive enough that callers should rely on
    /// [`crate::Grid::combined_world`]'s lazy cache rather than
    /// invoking this directly per-frame.
    ///
    /// # Panics
    ///
    /// Debug builds panic if any chunk has a negative XY index —
    /// out-of-scope for S4.0. Release builds silently skip such
    /// chunks. Non-zero `chz` chunks (vertical stacking) are
    /// skipped silently in both modes.
    #[must_use]
    #[allow(clippy::similar_names)] // max_chx / max_chy are voxlap-canonical pair names
    pub fn build(chunks: &HashMap<IVec3, Vxl>) -> Self {
        let cs = CHUNK_SIZE_XY;

        // Determine the chunk-XY extent. We only count chunks at
        // chz=0; others are skipped (S4.0 single-z-chunk scope).
        let mut max_chx: i32 = 0;
        let mut max_chy: i32 = 0;
        for &chunk_idx in chunks.keys() {
            if chunk_idx.z != 0 {
                continue;
            }
            debug_assert!(
                chunk_idx.x >= 0 && chunk_idx.y >= 0,
                "S4.0 only supports non-negative chunk indices (got {chunk_idx:?})"
            );
            if chunk_idx.x < 0 || chunk_idx.y < 0 {
                continue;
            }
            max_chx = max_chx.max(chunk_idx.x);
            max_chy = max_chy.max(chunk_idx.y);
        }
        // Square padding — voxlap's vsid is one number for both
        // axes. Empty grids still produce a 1-chunk virtual world
        // so the consumer always has something to render against.
        #[allow(clippy::cast_sign_loss)]
        let n_chunks = ((max_chx.max(max_chy)) as u32) + 1;
        let vsid = n_chunks * cs;
        let n_cols = (vsid as usize) * (vsid as usize);

        let mut data: Vec<u8> = Vec::new();
        let mut column_offset: Vec<u32> = Vec::with_capacity(n_cols + 1);

        for vy in 0..vsid {
            #[allow(clippy::cast_possible_wrap)]
            let chy = (vy / cs) as i32;
            let ly = vy % cs;
            for vx in 0..vsid {
                #[allow(clippy::cast_possible_wrap)]
                let chx = (vx / cs) as i32;
                let lx = vx % cs;
                let off = u32::try_from(data.len()).expect("combined data offset fits in u32");
                column_offset.push(off);
                match chunks.get(&IVec3::new(chx, chy, 0)) {
                    Some(chunk) => {
                        let local_idx = (ly * cs + lx) as usize;
                        let bytes = chunk.column_data(local_idx);
                        data.extend_from_slice(bytes);
                    }
                    None => {
                        // Implicit-air chunk → all-air placeholder.
                        // Same shape the per-chunk renderer's
                        // `treat_z_max_as_air` flag treats as sky.
                        data.extend_from_slice(&ALL_AIR_COLUMN);
                    }
                }
            }
        }
        let trailer = u32::try_from(data.len()).expect("combined data trailer fits in u32");
        column_offset.push(trailer);

        let mip_base_offsets = vec![0, n_cols + 1];

        Self {
            vsid,
            data,
            column_offset,
            mip_base_offsets,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Grid, GridTransform};
    use roxlap_formats::vxl::slng;

    /// Empty grid → 1-chunk virtual world, every column is the
    /// all-air placeholder.
    #[test]
    fn empty_grid_yields_one_chunk_virtual_world() {
        let grid = Grid::new(GridTransform::identity());
        let view = CombinedGridView::build(&grid.chunks);
        assert_eq!(view.vsid, CHUNK_SIZE_XY);
        let n_cols = (view.vsid as usize) * (view.vsid as usize);
        assert_eq!(view.column_offset.len(), n_cols + 1);
        assert_eq!(view.mip_base_offsets, vec![0, n_cols + 1]);
        // Every column is the 8-byte placeholder.
        for ci in 0..n_cols {
            let start = view.column_offset[ci] as usize;
            assert_eq!(slng(&view.data[start..]), ALL_AIR_COLUMN.len());
            assert_eq!(&view.data[start..start + 8], &ALL_AIR_COLUMN);
        }
    }

    /// Single populated chunk at (0, 0, 0) → virtual world has
    /// the same column data the chunk's `Vxl` does, byte for byte.
    #[test]
    fn single_chunk_view_matches_underlying_vxl() {
        let mut grid = Grid::new(GridTransform::identity());
        // Build one populated chunk by setting a few voxels.
        grid.set_voxel(IVec3::new(5, 6, 100), Some(0x80_aa_bb_cc));
        grid.set_voxel(IVec3::new(50, 50, 200), Some(0x80_11_22_33));
        let view = CombinedGridView::build(&grid.chunks);
        assert_eq!(view.vsid, CHUNK_SIZE_XY);

        let chunk = grid.chunk(IVec3::ZERO).unwrap();
        // Spot-check: voxlap's column_index = y * vsid + x.
        for &(x, y) in &[(0, 0), (5, 6), (50, 50), (127, 127)] {
            let local_idx = (y * CHUNK_SIZE_XY + x) as usize;
            let combined_off = view.column_offset[local_idx] as usize;
            let combined_len = slng(&view.data[combined_off..]);
            let chunk_bytes = chunk.column_data(local_idx);
            assert_eq!(
                &view.data[combined_off..combined_off + combined_len],
                chunk_bytes,
                "column ({x}, {y}) mismatch"
            );
        }
    }

    /// 2-chunk-wide grid: `vsid` bumps to `2 × CHUNK_SIZE_XY`, x in
    /// `[0, 128)` reads chunk 0, x in `[128, 256)` reads chunk 1.
    #[test]
    fn two_chunk_x_grid_stitches_left_right() {
        let mut grid = Grid::new(GridTransform::identity());
        // Chunk 0 voxel at (10, 0, 100); chunk 1 voxel at local
        // (10, 0, 100) i.e. grid-local (138, 0, 100).
        grid.set_voxel(IVec3::new(10, 0, 100), Some(0x80_aa_00_00));
        grid.set_voxel(IVec3::new(138, 0, 100), Some(0x80_00_aa_00));
        assert_eq!(grid.chunk_count(), 2);

        let view = CombinedGridView::build(&grid.chunks);
        assert_eq!(view.vsid, 2 * CHUNK_SIZE_XY);

        let c0 = grid.chunk(IVec3::ZERO).unwrap();
        let c1 = grid.chunk(IVec3::new(1, 0, 0)).unwrap();

        // Virtual column (10, 0) ↔ chunk 0 local (10, 0).
        let v_idx_left: u32 = 10; // y=0, x=10 → idx = 0*vsid + 10
        let off_left = view.column_offset[v_idx_left as usize] as usize;
        let len_left = slng(&view.data[off_left..]);
        let local_left = c0.column_data(10);
        assert_eq!(&view.data[off_left..off_left + len_left], local_left);

        // Virtual column (138, 0) ↔ chunk 1 local (10, 0).
        let v_idx_right: u32 = 138; // y=0, x=138
        let off_right = view.column_offset[v_idx_right as usize] as usize;
        let len_right = slng(&view.data[off_right..]);
        let local_right = c1.column_data(10);
        assert_eq!(&view.data[off_right..off_right + len_right], local_right);

        // Virtual column at the seam y row's far edge (y=130, well
        // past the populated chunk's y extent of 1) → all-air
        // placeholder because chunk (0, 1, 0) doesn't exist.
        let v_idx_pad = 130_u32 * view.vsid;
        let off_pad = view.column_offset[v_idx_pad as usize] as usize;
        assert_eq!(&view.data[off_pad..off_pad + 8], &ALL_AIR_COLUMN);
    }

    /// Z-stacked chunks (chz != 0) silently skipped at S4.0.
    #[test]
    fn z_stacked_chunks_skipped() {
        let mut grid = Grid::new(GridTransform::identity());
        // Chunk at (0, 0, 1) — outside S4.0 scope. Build should
        // not materialise its data into the view.
        grid.ensure_chunk(IVec3::new(0, 0, 1));
        let view = CombinedGridView::build(&grid.chunks);
        // No populated z=0 chunks → still 1-chunk virtual world.
        assert_eq!(view.vsid, CHUNK_SIZE_XY);
    }

    /// Non-square 2x1 lattice gets square-padded to 2x2; the second
    /// chunk row is all-air placeholders.
    #[test]
    fn non_square_lattice_squared_with_padding() {
        let mut grid = Grid::new(GridTransform::identity());
        grid.set_voxel(IVec3::new(0, 0, 0), Some(0x80_aa_00_00));
        grid.set_voxel(IVec3::new(128, 0, 0), Some(0x80_00_aa_00));
        // 2-wide, 1-tall in chunks. After build, padded to 2x2.
        let view = CombinedGridView::build(&grid.chunks);
        assert_eq!(view.vsid, 2 * CHUNK_SIZE_XY);
        // Column at (0, 200) — past the populated y row → all-air.
        let v_idx_pad = 200_u32 * view.vsid;
        let off = view.column_offset[v_idx_pad as usize] as usize;
        assert_eq!(&view.data[off..off + 8], &ALL_AIR_COLUMN);
    }
}
