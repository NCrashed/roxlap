//! Dense-grid → voxlap-slab packing.
//!
//! The cave-shape algorithm in [`crate::Generator::generate`] (CD.5.2)
//! produces a dense `(VSID × VSID × MAXZDIM)` voxel mask + colour
//! grid. [`pack_dense_grid_to_vxl`] folds that grid into voxlap's
//! slab-RLE column format.
//!
//! Implementation strategy (post-CD.5 architectural refactor): seed
//! an all-solid Vxl matching voxlap's `loadnul` shape, then carve
//! the air voxels via [`set_spans_with_colfunc`]. The colfunc closure
//! supplies the dense grid's colour for any newly-exposed voxel
//! `compilerle` queries — this is the same buried-voxel-aware
//! encoding the edit pipeline uses, so cave-gen output composes with
//! later runtime edits without surprises.
//!
//! [`set_spans_with_colfunc`]: roxlap_formats::edit::set_spans_with_colfunc

use roxlap_formats::edit::{set_spans_with_colfunc, SpanOp, Vspan};
use roxlap_formats::vxl::Vxl;

/// Voxlap's `MAXZDIM` (`voxlap5.h:10`) — world height, one byte
/// per z value → 256 voxels.
pub const MAXZDIM: i32 = 256;

/// Build a [`Vxl`] from a dense voxel-mask + colour grid.
///
/// `grid` and `color` are sized `VSID × VSID × MAXZDIM` in `(y, x,
/// z)` order — i.e., `grid[(y * vsid + x) * MAXZDIM + z]`. A non-
/// zero `grid` byte marks the voxel as solid; the corresponding
/// `color[..]` u32 is the BGRA colour stored when that voxel becomes
/// exposed (top of solid run, or skip-forward landing inside a run).
/// Air voxels (`grid == 0`) ignore `color`.
///
/// **Implementation**: seeds an all-solid Vxl (one floor colour per
/// column at z=0), then carves the air voxels via [`set_spans`]'
/// colfunc-flavoured variant. The carve walks in `(y, x)` ascending
/// order so it satisfies voxlap's setspans sort contract.
///
/// # Panics
///
/// - Panics if `grid.len() != vsid² × MAXZDIM` (or `color`).
/// - Panics if a column's solid voxels cannot fit in the slab pool;
///   the headroom estimate (256 bytes per column on average) is
///   generous for typical cave geometry.
///
/// [`set_spans`]: roxlap_formats::edit::set_spans
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn pack_dense_grid_to_vxl(grid: &[u8], color: &[u32], vsid: u32) -> Vxl {
    let vsid_u = vsid as usize;
    let maxzdim_u = MAXZDIM as usize;
    let expected = vsid_u * vsid_u * maxzdim_u;
    assert_eq!(grid.len(), expected, "grid size");
    assert_eq!(color.len(), expected, "color size");

    let n_cols = vsid_u * vsid_u;

    // 1. Seed Vxl: every column = single-voxel slab at z=0 with that
    //    voxel's colour from the grid (placeholder for air-at-z=0
    //    columns; carved away in step 3). Voxlap's slab convention
    //    treats this as "solid extending implicitly to MAXZDIM" until
    //    delslab pokes holes.
    let mut data: Vec<u8> = Vec::with_capacity(n_cols * 8);
    let mut column_offset: Vec<u32> = Vec::with_capacity(n_cols + 1);
    for y in 0..vsid_u {
        for x in 0..vsid_u {
            column_offset.push(u32::try_from(data.len()).expect("offset fits in u32"));
            // Header [nextptr=0, z1=0, z1c=0, z0=0]
            data.extend_from_slice(&[0, 0, 0, 0]);
            // 1 floor colour at z=0 — pulled from grid for the
            // common solid-floor case. For columns with air at z=0,
            // any value is fine since this voxel gets carved.
            let base = (y * vsid_u + x) * maxzdim_u;
            let c0 = if grid[base] == 0 { 0 } else { color[base] };
            data.extend_from_slice(&c0.to_le_bytes());
        }
    }
    column_offset.push(u32::try_from(data.len()).expect("offset fits in u32"));

    let mut vxl = Vxl {
        vsid,
        ipo: [0.0; 3],
        ist: [1.0, 0.0, 0.0],
        ihe: [0.0, 0.0, 1.0],
        ifo: [0.0, 1.0, 0.0],
        data: data.into_boxed_slice(),
        column_offset: column_offset.into_boxed_slice(),
        mip_base_offsets: Box::new([0, n_cols + 1]),
        vbit: Box::new([]),
        vbiti: 0,
    };

    // 2. Reserve edit capacity. Cave columns post-carve average ~50-
    //    150 bytes (multi-slab with sparse colour lists); 512 bytes
    //    per column is comfortable headroom.
    vxl.reserve_edit_capacity(n_cols * 512);

    // 3. Build sorted air-span list ((y, x) ascending; within each
    //    column, z runs are naturally ascending).
    let mut air_spans: Vec<Vspan> = Vec::new();
    for y in 0..vsid_u {
        for x in 0..vsid_u {
            let base = (y * vsid_u + x) * maxzdim_u;
            let mut z = 0usize;
            while z < maxzdim_u {
                if grid[base + z] == 0 {
                    let zs = z;
                    while z < maxzdim_u && grid[base + z] == 0 {
                        z += 1;
                    }
                    air_spans.push(Vspan {
                        x: x as u32,
                        y: y as u32,
                        z0: zs as u8,
                        z1: (z - 1) as u8, // inclusive — voxlap's vspans convention
                    });
                } else {
                    z += 1;
                }
            }
        }
    }

    // 4. Carve. The colfunc supplies colour for every newly-exposed
    //    voxel from the dense grid.
    if !air_spans.is_empty() {
        set_spans_with_colfunc(&mut vxl, &air_spans, SpanOp::Carve, |x, y, z| {
            let idx = (y as usize * vsid_u + x as usize) * maxzdim_u + z as usize;
            #[allow(clippy::cast_possible_wrap)]
            let c = color[idx] as i32;
            c
        });
    }

    vxl
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
mod tests {
    use super::*;
    use roxlap_formats::edit::expandrle;

    /// Build a small (vsid × vsid × MAXZDIM) grid with a known cave
    /// pattern. Returns `(grid, color)`. Grid is non-zero where the
    /// voxel is solid; air elsewhere.
    fn build_test_grid(
        vsid: u32,
        solid_predicate: impl Fn(u32, u32, i32) -> bool,
        colour_fn: impl Fn(u32, u32, i32) -> u32,
    ) -> (Vec<u8>, Vec<u32>) {
        let vsid_u = vsid as usize;
        let maxzdim_u = MAXZDIM as usize;
        let n = vsid_u * vsid_u * maxzdim_u;
        let mut grid = vec![0u8; n];
        let mut color = vec![0u32; n];
        for y in 0..vsid {
            for x in 0..vsid {
                for z in 0..MAXZDIM {
                    let idx = (y as usize * vsid_u + x as usize) * maxzdim_u + z as usize;
                    if solid_predicate(x, y, z) {
                        grid[idx] = 1;
                        color[idx] = colour_fn(x, y, z);
                    }
                }
            }
        }
        (grid, color)
    }

    #[test]
    fn pack_all_solid_grid_yields_minimal_columns() {
        // Every voxel solid — no air to carve. Each column should
        // remain at its seed encoding (1 colour at z=0).
        let (grid, color) = build_test_grid(2, |_, _, _| true, |_, _, _| 0x80aa_bbcc);
        let vxl = pack_dense_grid_to_vxl(&grid, &color, 2);
        assert_eq!(vxl.vsid, 2);
        for idx in 0..4 {
            let mut b2 = vec![0i32; 256];
            expandrle(vxl.column_data(idx), &mut b2);
            // Solid run [0, MAXZDIM).
            assert_eq!(b2[0], 0, "col {idx} run top");
            assert_eq!(b2[1], MAXZDIM, "col {idx} run bot");
        }
    }

    #[test]
    fn pack_top_carve_yields_z_offset_solid() {
        // 2x2 world; carve [0, 50) on every column → solid runs
        // start at z=50.
        let (grid, color) =
            build_test_grid(2, |_, _, z| z >= 50, |_, _, z| 0x8000_0000 | (z as u32));
        let vxl = pack_dense_grid_to_vxl(&grid, &color, 2);
        for idx in 0..4 {
            let mut b2 = vec![0i32; 256];
            expandrle(vxl.column_data(idx), &mut b2);
            assert_eq!(b2[0], 50, "col {idx} run top");
            assert_eq!(b2[1], MAXZDIM, "col {idx} run bot");
        }
    }

    #[test]
    fn pack_cave_in_middle_yields_two_runs() {
        // 2x2 world; carve [50, 100) on every column → two solid
        // runs: [0, 50) and [100, MAXZDIM).
        let (grid, color) = build_test_grid(
            2,
            |_, _, z| !(50..100).contains(&z),
            |_, _, z| 0x8000_0000 | (z as u32),
        );
        let vxl = pack_dense_grid_to_vxl(&grid, &color, 2);
        for idx in 0..4 {
            let mut b2 = vec![0i32; 256];
            expandrle(vxl.column_data(idx), &mut b2);
            assert_eq!(b2[0], 0, "col {idx}");
            assert_eq!(b2[1], 50, "col {idx}");
            assert_eq!(b2[2], 100, "col {idx}");
            assert_eq!(b2[3], MAXZDIM, "col {idx}");
        }
    }

    #[test]
    fn pack_top_solid_voxel_uses_grid_color() {
        // Single column world (vsid=1 isn't supported — neighbors must
        // be valid). Use vsid=2 and test column (0, 0).
        // Solid only at z=64..70; check that top voxel (z=64) has the
        // colour from the grid.
        let (grid, color) = build_test_grid(
            2,
            |_, _, z| (64..70).contains(&z),
            |_, _, z| 0x8011_2233 | (z as u32),
        );
        let vxl = pack_dense_grid_to_vxl(&grid, &color, 2);
        let column = vxl.column_data(0);
        // Walk to the slab with z1=64 (top of solid run).
        let mut v = 0usize;
        let mut top_color = None;
        loop {
            let nextptr = column[v];
            let z1 = column[v + 1];
            if z1 == 64 {
                let off = v + 4;
                top_color = Some(u32::from_le_bytes([
                    column[off],
                    column[off + 1],
                    column[off + 2],
                    column[off + 3],
                ]));
                break;
            }
            if nextptr == 0 {
                break;
            }
            v += usize::from(nextptr) * 4;
        }
        assert_eq!(
            top_color,
            Some(0x8011_2233 | 64),
            "exposed top voxel at z=64 should use grid colour"
        );
    }

    #[test]
    fn pack_round_trip_through_worley_grid() {
        // worley_classify_grid → pack_dense_grid_to_vxl yields a Vxl
        // whose every column is parseable. Tests the integration
        // between cave shape generation and slab packing.
        use crate::{worley_classify_grid, CaveParams};
        let params = CaveParams {
            seed: 42,
            seed_count: 16,
            air_ratio: 0.5,
            anisotropy: 1.0,
            perlin_octaves: 0,
            perlin_amplitude: 0.0,
        };
        let vsid = 8u32;
        let grid = worley_classify_grid(&params, vsid);
        // Build a colour grid: each solid voxel gets z-as-low-byte.
        let vsid_u = vsid as usize;
        let maxzdim_u = MAXZDIM as usize;
        let mut color = vec![0u32; vsid_u * vsid_u * maxzdim_u];
        for (i, &g) in grid.iter().enumerate() {
            if g != 0 {
                let z = (i % maxzdim_u) as u32;
                color[i] = 0x8060_4020 | z;
            }
        }
        let vxl = pack_dense_grid_to_vxl(&grid, &color, vsid);
        // Sanity: every column should be expandrle-decodable into a
        // valid b2 buffer (sentinel-terminated).
        for idx in 0..(vsid_u * vsid_u) {
            let mut b2 = vec![0i32; 256];
            expandrle(vxl.column_data(idx), &mut b2);
            // Walk to sentinel; runs must satisfy top < bot.
            let mut i = 0;
            while b2[i + 1] < MAXZDIM {
                assert!(
                    b2[i] < b2[i + 1],
                    "col {idx} run {i}: top={} bot={}",
                    b2[i],
                    b2[i + 1]
                );
                i += 2;
            }
        }
    }

    #[test]
    fn pack_grid_size_assert() {
        // Wrong-sized grid panics.
        let result = std::panic::catch_unwind(|| {
            let grid = vec![0u8; 4];
            let color = vec![0u32; 4];
            let _ = pack_dense_grid_to_vxl(&grid, &color, 2);
        });
        assert!(result.is_err(), "wrong grid size should panic");
    }
}
