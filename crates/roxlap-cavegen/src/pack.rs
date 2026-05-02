//! Dense-grid → voxlap-slab packing.
//!
//! The cave-shape algorithm in [`crate::Generator::generate`] (CD.5.2)
//! produces a dense `(VSID × VSID × MAXZDIM)` voxel mask + colour
//! grid. [`pack_dense_grid_to_vxl`] folds that grid into voxlap's
//! slab-RLE column format directly, without round-tripping through
//! the [`set_spans`] edit pipeline.
//!
//! The encoder is "fat": every solid voxel's colour is stored,
//! splitting runs into the voxlap top-of-run-in-ceiling-list pattern
//! so adjacent slabs stay compatible with `expandrle`'s
//! `v[3] >= v[1] → skip` check.
//!
//! Runs longer than 253 voxels are split into multiple slabs to keep
//! `nextptr` within the byte-sized field — voxlap C has the same
//! constraint (`v[0]*4` walks via signed `char` arithmetic on x86).
//!
//! [`set_spans`]: ../roxlap_formats/edit/fn.set_spans.html

use roxlap_formats::vxl::Vxl;

/// Voxlap's `MAXZDIM` (`voxlap5.h:10`) — world height, one byte
/// per z value → 256 voxels.
pub const MAXZDIM: i32 = 256;

/// Build a [`Vxl`] from a dense voxel-mask + colour grid.
///
/// `grid` and `color` are sized `VSID × VSID × MAXZDIM` in `(y, x,
/// z)` order — i.e., `grid[(y * vsid + x) * MAXZDIM + z]`. A non-
/// zero `grid` byte marks the voxel as solid; the corresponding
/// `color[..]` u32 is the BGRA colour stored in the slab. Air
/// voxels (`grid == 0`) ignore `color`.
///
/// **Stub**: CD.5.1 will fill in the encoder. Today this returns
/// an empty all-air placeholder Vxl.
///
/// # Panics
///
/// Panics if `grid.len() != vsid * vsid * MAXZDIM` (or `color`).
#[must_use]
pub fn pack_dense_grid_to_vxl(grid: &[u8], color: &[u32], vsid: u32) -> Vxl {
    let expected = (vsid as usize) * (vsid as usize) * (MAXZDIM as usize);
    assert_eq!(grid.len(), expected, "grid size");
    assert_eq!(color.len(), expected, "color size");
    let _ = grid;
    let _ = color;

    // CD.5.1 placeholder — return a Vxl with one minimal solid voxel
    // per column at z=0 (matches voxlap's loadnul shape). Tests will
    // pin a richer encoding in CD.5.1.
    let n_cols = (vsid as usize) * (vsid as usize);
    let mut data: Vec<u8> = Vec::with_capacity(n_cols * 8);
    let mut column_offset: Vec<u32> = Vec::with_capacity(n_cols + 1);
    for _ in 0..n_cols {
        column_offset.push(u32::try_from(data.len()).expect("column offset fits in u32"));
        // Header [nextptr=0, z1=0, z1c=0, z0=0] + 1 black floor color.
        data.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0]);
    }
    column_offset.push(u32::try_from(data.len()).expect("column offset fits in u32"));

    Vxl {
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
    }
}
