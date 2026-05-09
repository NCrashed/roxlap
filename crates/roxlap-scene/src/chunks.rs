//! Sparse chunk storage helpers.
//!
//! A grid's [`Grid::chunks`] map holds populated chunks keyed by
//! their `(chx, chy, chz)` index. A missing entry is an implicit
//! all-air chunk; this module provides the constructor for fresh
//! all-air chunks plus the `chunk` / `chunk_mut` / `ensure_chunk`
//! lookup API.
//!
//! [`Grid::chunks`]: crate::Grid::chunks

use glam::IVec3;
use roxlap_formats::edit::{set_spans, Vspan};
use roxlap_formats::vxl::Vxl;

use crate::{Grid, CHUNK_SIZE_XY};

/// Bytes of edit-pool headroom reserved per chunk on creation.
/// 256 bytes/column × 128² columns ≈ 4 MiB; a generous budget for
/// runtime edits within a single chunk before [`voxalloc`] starts
/// returning out-of-space. Tunable later if memory becomes an
/// issue.
///
/// [`voxalloc`]: roxlap_formats::vxl::Vxl::voxalloc
const CHUNK_EDIT_HEADROOM_PER_COLUMN: usize = 256;

/// Construct a fresh all-air [`Vxl`] sized for one chunk
/// (`vsid = CHUNK_SIZE_XY`).
///
/// Strategy mirrors `roxlap_cavegen::pack_dense_grid_to_vxl`: seed
/// each column with one solid voxel at z=0 + implicit-solid below
/// (the voxlap "loadnul" shape), then carve the entire z range to
/// air via [`set_spans`]. Finishes with [`Vxl::reserve_edit_capacity`]
/// so subsequent runtime edits don't need a separate upgrade pass.
///
/// This is the canonical empty-chunk constructor — every code
/// path that materialises a sparse chunk goes through it (see
/// [`Grid::ensure_chunk`]).
fn empty_chunk_vxl() -> Vxl {
    let vsid = CHUNK_SIZE_XY;
    let n_cols = (vsid as usize) * (vsid as usize);

    // 1. Seed: every column = 4-byte slab header + 1 colour. Colour
    //    is irrelevant — the whole column gets carved below.
    let mut data: Vec<u8> = Vec::with_capacity(n_cols * 8);
    let mut column_offset: Vec<u32> = Vec::with_capacity(n_cols + 1);
    for _ in 0..n_cols {
        column_offset.push(u32::try_from(data.len()).expect("offset fits in u32"));
        data.extend_from_slice(&[0, 0, 0, 0]); // header
        data.extend_from_slice(&[0, 0, 0, 0]); // 1 placeholder colour
    }
    column_offset.push(u32::try_from(data.len()).expect("offset fits in u32"));

    let mut vxl = Vxl {
        vsid,
        // Per-grid placement lives on `GridTransform`; the per-chunk
        // Vxl's intrinsic camera fields are unused at this layer.
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
    vxl.reserve_edit_capacity(n_cols * CHUNK_EDIT_HEADROOM_PER_COLUMN);

    // 2. Carve [0, 255] in every column to make it all-air.
    //    `Vspan.z1` is inclusive per voxlap's vspans convention.
    let mut spans: Vec<Vspan> = Vec::with_capacity(n_cols);
    for y in 0..vsid {
        for x in 0..vsid {
            spans.push(Vspan {
                x,
                y,
                z0: 0,
                z1: u8::MAX,
            });
        }
    }
    set_spans(&mut vxl, &spans, None);

    vxl
}

impl Grid {
    /// Borrow the chunk at `chunk_idx` if it has been materialised.
    /// `None` means the chunk is implicitly all-air.
    #[must_use]
    pub fn chunk(&self, chunk_idx: IVec3) -> Option<&Vxl> {
        self.chunks.get(&chunk_idx)
    }

    /// Mutably borrow a materialised chunk. Returns `None` for
    /// implicit-air chunks; use [`Grid::ensure_chunk`] when you
    /// need a `&mut Vxl` for an edit that may write voxels.
    pub fn chunk_mut(&mut self, chunk_idx: IVec3) -> Option<&mut Vxl> {
        self.chunks.get_mut(&chunk_idx)
    }

    /// Borrow `chunk_idx`'s [`Vxl`], creating an empty all-air
    /// chunk first if it doesn't exist yet. The returned `&mut`
    /// is valid for editing via [`roxlap_formats::edit`] — the new
    /// chunk has [`Vxl::reserve_edit_capacity`] already applied.
    pub fn ensure_chunk(&mut self, chunk_idx: IVec3) -> &mut Vxl {
        self.chunks.entry(chunk_idx).or_insert_with(empty_chunk_vxl)
    }

    /// Number of materialised chunks. Implicit-air chunks don't
    /// count.
    #[must_use]
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::{GridTransform, CHUNK_SIZE_Z};
    use roxlap_formats::edit::expandrle;

    /// Decode `column`'s slab bytes and return `true` iff `z` is
    /// covered by any solid run. Mirrors voxlap's column-walk
    /// semantics — the b2 buffer is `[top0, bot0, top1, bot1, ...,
    /// MAXZDIM_sentinel]`, with each `[top, bot)` pair denoting a
    /// solid range.
    #[allow(clippy::cast_possible_wrap)]
    pub(crate) fn voxel_is_solid(vxl: &Vxl, x: u32, y: u32, z: u32) -> bool {
        let idx = (y * vxl.vsid + x) as usize;
        let column = vxl.column_data(idx);
        // Pre-fill with MAXZDIM so unwritten slots terminate the walk
        // (matches voxlap's b2 init convention in `all_air_neighbor`
        // and friends — expandrle only writes the prefix it needs).
        let maxzdim = CHUNK_SIZE_Z as i32;
        let mut b2 = vec![maxzdim; 2 * (CHUNK_SIZE_Z as usize) + 4];
        expandrle(column, &mut b2);
        let z = z as i32;
        let mut i = 0;
        while b2[i] < maxzdim {
            let top = b2[i];
            let bot = b2[i + 1];
            if z >= top && z < bot {
                return true;
            }
            i += 2;
        }
        false
    }

    #[test]
    fn empty_chunk_has_correct_vsid() {
        let vxl = empty_chunk_vxl();
        assert_eq!(vxl.vsid, CHUNK_SIZE_XY);
    }

    #[test]
    fn empty_chunk_is_all_air() {
        let vxl = empty_chunk_vxl();
        // Sample a few representative voxels — full coverage is in
        // `empty_chunk_no_voxel_solid_anywhere` below.
        for &(x, y, z) in &[
            (0u32, 0u32, 0u32),
            (0, 0, 100),
            (0, 0, 200),
            (CHUNK_SIZE_XY - 1, CHUNK_SIZE_XY - 1, 0),
            (64, 64, 128),
        ] {
            assert!(
                !voxel_is_solid(&vxl, x, y, z),
                "voxel ({x}, {y}, {z}) should be air"
            );
        }
    }

    #[test]
    fn empty_chunk_air_above_bedrock_on_grid_sample() {
        // Stride 16 across the chunk catches structural breakage
        // (a corner column wrong, a z-band wrong, etc.) without the
        // 4M-query cost of a brute-force scan in debug mode.
        // Voxlap's slab format keeps z=255 solid as the "below the
        // world" sentinel; the renderer's `treat_z_max_as_air` flag
        // handles displaying it as transparent. See
        // `project_below_bedrock_all_sky.md` for the S1.X fix.
        let vxl = empty_chunk_vxl();
        let bedrock_z = CHUNK_SIZE_Z - 1;
        for y in (0..CHUNK_SIZE_XY).step_by(16) {
            for x in (0..CHUNK_SIZE_XY).step_by(16) {
                for z in (0..bedrock_z).step_by(16) {
                    assert!(
                        !voxel_is_solid(&vxl, x, y, z),
                        "voxel ({x}, {y}, {z}) leaked solid in empty chunk"
                    );
                }
                // bedrock z is solid (placeholder).
                assert!(voxel_is_solid(&vxl, x, y, bedrock_z));
            }
        }
    }

    #[test]
    fn empty_chunk_keeps_bedrock_placeholder() {
        // Voxlap's invariant: every column carries an implicit
        // solid voxel at z = MAXZDIM-1 = 255 even after a full
        // carve. The renderer reads this as the bedrock placeholder.
        let vxl = empty_chunk_vxl();
        assert!(voxel_is_solid(&vxl, 0, 0, CHUNK_SIZE_Z - 1));
        assert!(voxel_is_solid(&vxl, 64, 64, CHUNK_SIZE_Z - 1));
    }

    #[test]
    fn ensure_chunk_creates_when_missing() {
        let mut g = Grid::new(GridTransform::identity());
        assert_eq!(g.chunk_count(), 0);
        assert!(g.chunk(IVec3::ZERO).is_none());
        let _ = g.ensure_chunk(IVec3::ZERO);
        assert_eq!(g.chunk_count(), 1);
        assert!(g.chunk(IVec3::ZERO).is_some());
    }

    #[test]
    fn ensure_chunk_returns_existing() {
        // Calling ensure_chunk a second time on the same index
        // doesn't replace the chunk. Verify by writing through the
        // first call and reading through the second.
        let mut g = Grid::new(GridTransform::identity());
        let chunk = IVec3::new(2, -1, 0);
        g.ensure_chunk(chunk);
        // Voxel local (5, 6, 7) inside chunk (2, -1, 0) is
        // grid-local global (2*128 + 5, -1*128 + 6, 0*256 + 7) =
        // (261, -122, 7).
        g.set_voxel(IVec3::new(261, -122, 7), Some(0x80_aa_bb_cc));
        let vxl = g.ensure_chunk(chunk);
        assert!(voxel_is_solid(vxl, 5, 6, 7));
        assert_eq!(g.chunk_count(), 1);
    }

    #[test]
    fn chunk_mut_returns_none_for_missing() {
        let mut g = Grid::new(GridTransform::identity());
        assert!(g.chunk_mut(IVec3::ZERO).is_none());
    }
}
