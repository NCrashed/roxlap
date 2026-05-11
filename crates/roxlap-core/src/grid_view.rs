//! Per-frame voxel-world borrow shape.
//!
//! Wraps the `(vsid, slab_buf, column_offsets, mip_base_offsets)`
//! tuple that [`crate::opticast`] and
//! [`crate::scalar_rasterizer::ScalarRasterizer`] both need. Today
//! always represents a single chunk so callers building from a
//! [`roxlap_formats::vxl::Vxl`] keep the existing flat-world
//! semantics byte-identically.
//!
//! Substage S4B.0 introduced the shape as a pure rename — opticast
//! drove a single flat world behind a typed borrow. Subsequent
//! S4B.x sub-substages grow this into a multi-chunk view:
//!
//! * S4B.1 — carry the camera's chunk index alongside the borrow.
//! * S4B.2.a — `chunk_size_xy` field + `chunk_at_xy` method.
//!   Today's single-chunk callers set `chunk_size_xy = vsid` and
//!   the lookup only succeeds for `[0, 0]`.
//! * S4B.2.b — grouscan column-step calls `chunk_at_xy` and swaps
//!   active per-chunk `(slab_buf, column_offsets)` when `(cx, cy)`
//!   crosses a chunk boundary. Single-chunk goldens byte-identical.
//! * S4B.2.c.1 (this file) — [`ChunkGrid`] backend so `chunk_at_xy`
//!   can resolve real per-chunk borrows. Pure infrastructure; no
//!   caller integration yet.
//! * S4B.2.c.2 — scene-side multi-chunk constructor + 32×32
//!   ground seam test.
//! * S4B.3 — chunk-z extent + handoff for cross-chunk-Z rays.
//!
//! See `project_s4_b_plan.md` for the full sub-substage plan.

use roxlap_formats::vxl::Vxl;

/// Per-frame world borrow that opticast + the rasterizer share.
///
/// Today: a single chunk's `(vsid, slab_buf, column_offsets,
/// mip_base_offsets)`. `Copy` so callers can pass it to opticast
/// and stash it on the rasterizer without ceremony — every field
/// is a borrow or a `u32`.
///
/// Fields are public on purpose. External callers usually go
/// through the [`from_single_vxl`](Self::from_single_vxl) /
/// [`from_parts`](Self::from_parts) constructors, but the engine's
/// internals destructure directly. Keeping the fields exposed
/// avoids a layer of accessor methods that the borrow checker
/// would otherwise force at every read.
#[derive(Clone, Copy)]
pub struct GridView<'a> {
    /// Square dimension of the currently-active chunk view (matches
    /// the source `Vxl`'s `vsid` for single-chunk callers). The
    /// per-chunk `column_offsets` table holds `(vsid² + 1)` entries.
    pub vsid: u32,
    /// S4B.2.a: square dimension of each chunk in XY voxel units.
    /// For today's single-chunk callers, `chunk_size_xy == vsid` so
    /// `(cx, cy)` in `[0, vsid)` never crosses a chunk boundary.
    /// For multi-chunk callers (S4B.2.c+), `chunk_size_xy` is the
    /// per-chunk dimension (typically 128) and `vsid` is the same
    /// per-chunk value — they may diverge in S4B.4 when GridView
    /// stops carrying a "default" chunk's flat fields.
    pub chunk_size_xy: u32,
    /// Flat slab byte buffer for every column at every built mip.
    pub slab_buf: &'a [u8],
    /// Per-column byte offsets into [`Self::slab_buf`], concatenated
    /// across every mip's sub-table. Mip-0 occupies indices
    /// `mip_base_offsets[0]..mip_base_offsets[1]`.
    pub column_offsets: &'a [u32],
    /// Mip-level boundaries inside [`Self::column_offsets`].
    /// Length `mip_count + 1`; trailing sentinel equals
    /// `column_offsets.len()`. Single-mip callers pass
    /// `&[0, vsid² + 1]`.
    pub mip_base_offsets: &'a [usize],
    /// S4B.2.c.1: chunk-grid backend. `None` for single-chunk views
    /// (e.g. anything built via [`Self::from_single_vxl`] /
    /// [`Self::from_parts`]) — [`Self::chunk_at_xy`] falls back to
    /// the `Some(Self) for [0, 0]` behaviour. `Some(&...)` for
    /// multi-chunk views built via [`Self::from_chunk_grid`] —
    /// [`Self::chunk_at_xy`] consults the table.
    pub chunk_grid: Option<&'a ChunkGrid<'a>>,
}

/// S4B.2.c.1: chunk-grid metadata for multi-chunk [`GridView`]
/// lookups.
///
/// Stores a 2D table of optional per-chunk [`GridView`] borrows so
/// [`GridView::chunk_at_xy`] can resolve an XY chunk index to the
/// matching chunk's `(slab_buf, column_offsets, ...)` view. Empty
/// table entries (`None`) signal "no chunk at that index" — the
/// grouscan column-step treats them as fully-air (the chunk renders
/// as sky / empty until the ray crosses into a populated chunk).
///
/// Sized to match the chunk grid's XY footprint:
/// `chunks.len() == chunks_x * chunks_y`. Chunk at relative position
/// `(dx, dy)` from `origin_chunk_xy` lives at index
/// `dy * chunks_x + dx`. The per-chunk [`GridView`] entries should
/// have `chunk_grid: None` (they describe individual chunks, not
/// the parent grid).
#[derive(Clone, Copy)]
pub struct ChunkGrid<'a> {
    /// Per-chunk views, row-major. Length `chunks_x * chunks_y`.
    pub chunks: &'a [Option<GridView<'a>>],
    /// XY index of the chunk at `chunks[0]`. Subsequent chunks lie
    /// along `+x` (next in the row) then `+y` (next row).
    pub origin_chunk_xy: [i32; 2],
    /// Number of chunks along the X axis. Row stride in
    /// [`Self::chunks`].
    pub chunks_x: u32,
    /// Number of chunks along the Y axis.
    pub chunks_y: u32,
}

impl<'a> GridView<'a> {
    /// Build from explicit fields. Test fixtures use this directly;
    /// production callers usually go through
    /// [`from_single_vxl`](Self::from_single_vxl).
    ///
    /// Sets `chunk_size_xy = vsid` (single-chunk semantics). Use
    /// [`with_chunk_size_xy`](Self::with_chunk_size_xy) to mark the
    /// view as part of a chunk grid.
    #[must_use]
    pub fn from_parts(
        vsid: u32,
        slab_buf: &'a [u8],
        column_offsets: &'a [u32],
        mip_base_offsets: &'a [usize],
    ) -> Self {
        Self {
            vsid,
            chunk_size_xy: vsid,
            slab_buf,
            column_offsets,
            mip_base_offsets,
            chunk_grid: None,
        }
    }

    /// Borrow a parsed `.vxl` map as a single-chunk grid view. The
    /// scene-graph stage's eventual multi-chunk constructor will
    /// live alongside this one (`from_grid` over
    /// `roxlap_scene::Grid`).
    #[must_use]
    pub fn from_single_vxl(vxl: &'a Vxl) -> Self {
        Self {
            vsid: vxl.vsid,
            chunk_size_xy: vxl.vsid,
            slab_buf: &vxl.data,
            column_offsets: &vxl.column_offset,
            mip_base_offsets: &vxl.mip_base_offsets,
            chunk_grid: None,
        }
    }

    /// S4B.2.c.1: build a multi-chunk view from a [`ChunkGrid`].
    ///
    /// The returned [`GridView`]'s flat `(vsid, slab_buf,
    /// column_offsets, mip_base_offsets)` fields are seeded from
    /// the first populated chunk in the grid (so opticast's prelude
    /// has a sensible default before its camera-chunk lookup
    /// refreshes them). `chunk_size_xy` carries the caller-supplied
    /// per-chunk dimension; [`Self::chunk_grid`] points at
    /// `chunk_grid` so [`Self::chunk_at_xy`] resolves every
    /// in-range index to its actual chunk borrow.
    ///
    /// Empty-grid case: if every chunk is `None`, the flat fields
    /// fall back to empty slices and `vsid = chunk_size_xy`. The
    /// grouscan column-step swap will see `chunk_at_xy → None` for
    /// every index and render the whole grid as sky.
    #[must_use]
    pub fn from_chunk_grid(chunk_grid: &'a ChunkGrid<'a>, chunk_size_xy: u32) -> Self {
        let default_chunk = chunk_grid.chunks.iter().find_map(|c| c.as_ref()).copied();
        match default_chunk {
            Some(c) => Self {
                vsid: c.vsid,
                chunk_size_xy,
                slab_buf: c.slab_buf,
                column_offsets: c.column_offsets,
                mip_base_offsets: c.mip_base_offsets,
                chunk_grid: Some(chunk_grid),
            },
            None => Self {
                vsid: chunk_size_xy,
                chunk_size_xy,
                slab_buf: &[],
                column_offsets: &[],
                mip_base_offsets: &[],
                chunk_grid: Some(chunk_grid),
            },
        }
    }

    /// S4B.2.a builder: override [`Self::chunk_size_xy`]. Multi-chunk
    /// callers (S4B.2.c+) use this to mark the view as one chunk of
    /// a larger grid. Today no caller needs it; the existence makes
    /// the seam testable in isolation.
    #[must_use]
    pub fn with_chunk_size_xy(mut self, chunk_size_xy: u32) -> Self {
        self.chunk_size_xy = chunk_size_xy;
        self
    }

    /// S4B.2.d: voxel-space XY axis-aligned bounding box of the
    /// grid. Returns `([xmin, ymin], [xmax, ymax])` in voxel units;
    /// the grid contains voxels with coordinates in
    /// `[xmin, xmax) × [ymin, ymax)`.
    ///
    /// - Single-chunk (`chunk_grid: None`): returns
    ///   `([0, 0], [vsid, vsid])`. Byte-identical to the historical
    ///   single-chunk world-edge math.
    /// - Multi-chunk (`chunk_grid: Some(&cg)`): derived from
    ///   `cg.origin_chunk_xy + cg.chunks_x/y * chunk_size_xy`.
    ///
    /// Consumed by [`crate::opticast_prelude::recompute_in_bounds_xy`]
    /// (camera-inside-grid check) and the rasterizer's gline
    /// world-edge gxmax clip.
    #[must_use]
    pub fn aabb_xy(&self) -> ([i32; 2], [i32; 2]) {
        if let Some(cg) = self.chunk_grid {
            #[allow(clippy::cast_possible_wrap)]
            let cs = self.chunk_size_xy as i32;
            #[allow(clippy::cast_possible_wrap)]
            let chunks_x = cg.chunks_x as i32;
            #[allow(clippy::cast_possible_wrap)]
            let chunks_y = cg.chunks_y as i32;
            let xmin = cg.origin_chunk_xy[0] * cs;
            let ymin = cg.origin_chunk_xy[1] * cs;
            let xmax = xmin + chunks_x * cs;
            let ymax = ymin + chunks_y * cs;
            ([xmin, ymin], [xmax, ymax])
        } else {
            #[allow(clippy::cast_possible_wrap)]
            let v = self.vsid as i32;
            ([0, 0], [v, v])
        }
    }

    /// S4B.2.a: chunk lookup for the cross-chunk-XY DDA.
    ///
    /// Returns the [`GridView`] for the chunk at XY index
    /// `chunk_idx` if one exists, `None` otherwise. Routes by
    /// [`Self::chunk_grid`]:
    ///
    /// * `chunk_grid: Some(&cg)` — multi-chunk view. Resolves
    ///   `chunk_idx` against `cg.origin_chunk_xy` /
    ///   `cg.chunks_x` / `cg.chunks_y` and returns `cg.chunks[..]`
    ///   at the matching slot (which may itself be `None` for
    ///   empty chunks).
    /// * `chunk_grid: None` — single-chunk view. Returns `Some(Self)`
    ///   for `[0, 0]`, `None` for any other index. Matches today's
    ///   single-chunk callers (every `from_single_vxl` /
    ///   `from_parts` consumer).
    ///
    /// The grouscan column-step treats `None` as an empty chunk
    /// (renders as sky / empty until the ray re-enters a populated
    /// chunk).
    #[must_use]
    pub fn chunk_at_xy(&self, chunk_idx: [i32; 2]) -> Option<GridView<'a>> {
        if let Some(cg) = self.chunk_grid {
            let dx = chunk_idx[0] - cg.origin_chunk_xy[0];
            let dy = chunk_idx[1] - cg.origin_chunk_xy[1];
            if dx < 0 || dy < 0 {
                return None;
            }
            #[allow(clippy::cast_sign_loss)]
            let (dx, dy) = (dx as u32, dy as u32);
            if dx >= cg.chunks_x || dy >= cg.chunks_y {
                return None;
            }
            let i = dy as usize * cg.chunks_x as usize + dx as usize;
            cg.chunks.get(i).copied().flatten()
        } else if chunk_idx == [0, 0] {
            Some(*self)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_parts_preserves_fields_byte_identically() {
        let slab = [0u8, 200, 254, 0];
        let cols = [0u32, 4];
        let mips = [0usize, 2];
        let gv = GridView::from_parts(1, &slab, &cols, &mips);
        assert_eq!(gv.vsid, 1);
        assert_eq!(gv.slab_buf, &slab[..]);
        assert_eq!(gv.column_offsets, &cols[..]);
        assert_eq!(gv.mip_base_offsets, &mips[..]);
    }

    #[test]
    fn grid_view_is_copy() {
        // Compile-time check: GridView must be Copy so opticast +
        // ScalarRasterizer can stash independent copies without a
        // borrow-checker dance.
        fn assert_copy<T: Copy>() {}
        assert_copy::<GridView<'_>>();
    }

    #[test]
    fn from_parts_defaults_chunk_size_xy_to_vsid() {
        let mips = [0usize, 2];
        let gv = GridView::from_parts(2048, &[], &[], &mips);
        assert_eq!(gv.chunk_size_xy, 2048);
    }

    #[test]
    fn chunk_at_xy_returns_self_for_origin_chunk() {
        let slab = [0u8, 200, 254, 0];
        let cols = [0u32, 4];
        let mips = [0usize, 2];
        let gv = GridView::from_parts(1, &slab, &cols, &mips);
        let inner = gv.chunk_at_xy([0, 0]).expect("origin chunk present");
        assert_eq!(inner.vsid, gv.vsid);
        assert_eq!(inner.slab_buf, gv.slab_buf);
        assert_eq!(inner.column_offsets, gv.column_offsets);
    }

    #[test]
    fn chunk_at_xy_returns_none_for_off_origin_idx() {
        let mips = [0usize, 2];
        let gv = GridView::from_parts(1, &[], &[], &mips);
        assert!(gv.chunk_at_xy([1, 0]).is_none());
        assert!(gv.chunk_at_xy([-1, 0]).is_none());
        assert!(gv.chunk_at_xy([0, 1]).is_none());
        assert!(gv.chunk_at_xy([5, -7]).is_none());
    }

    #[test]
    fn with_chunk_size_xy_overrides_default() {
        let mips = [0usize, 2];
        let gv = GridView::from_parts(2048, &[], &[], &mips).with_chunk_size_xy(128);
        assert_eq!(gv.vsid, 2048);
        assert_eq!(gv.chunk_size_xy, 128);
    }

    /// S4B.2.c.1: tiny 2-chunk-x-stripe ChunkGrid scaffolding for
    /// the multi-chunk lookup tests. Two synthetic 1×1 chunks at
    /// XY indices [0, 0] and [1, 0]. Their slab/column data is
    /// distinct so the lookup tests can tell them apart.
    fn build_two_chunk_x_stripe() -> ([u8; 4], [u8; 4], [u32; 2], [u32; 2], [usize; 2], [usize; 2])
    {
        let slab_a = [10u8, 200, 254, 0];
        let slab_b = [20u8, 200, 254, 0];
        let cols_a = [0u32, 4];
        let cols_b = [0u32, 4];
        let mips_a = [0usize, 2];
        let mips_b = [0usize, 2];
        (slab_a, slab_b, cols_a, cols_b, mips_a, mips_b)
    }

    #[test]
    fn chunk_at_xy_via_chunk_grid_returns_in_range_chunks() {
        let (slab_a, slab_b, cols_a, cols_b, mips_a, mips_b) = build_two_chunk_x_stripe();
        let chunks = [
            Some(GridView::from_parts(64, &slab_a, &cols_a, &mips_a)),
            Some(GridView::from_parts(64, &slab_b, &cols_b, &mips_b)),
        ];
        let cg = ChunkGrid {
            chunks: &chunks,
            origin_chunk_xy: [0, 0],
            chunks_x: 2,
            chunks_y: 1,
        };
        let gv = GridView::from_chunk_grid(&cg, 64);
        // [0, 0] resolves to chunk A.
        let c0 = gv.chunk_at_xy([0, 0]).expect("chunk [0, 0] present");
        assert_eq!(c0.slab_buf, &slab_a[..]);
        // [1, 0] resolves to chunk B.
        let c1 = gv.chunk_at_xy([1, 0]).expect("chunk [1, 0] present");
        assert_eq!(c1.slab_buf, &slab_b[..]);
        // [0, 0] and [1, 0] carry distinct slab buffers.
        assert_ne!(c0.slab_buf, c1.slab_buf);
    }

    #[test]
    fn chunk_at_xy_via_chunk_grid_returns_none_out_of_range() {
        let (slab_a, _, cols_a, _, mips_a, _) = build_two_chunk_x_stripe();
        let chunks = [
            Some(GridView::from_parts(64, &slab_a, &cols_a, &mips_a)),
            None,
        ];
        let cg = ChunkGrid {
            chunks: &chunks,
            origin_chunk_xy: [0, 0],
            chunks_x: 2,
            chunks_y: 1,
        };
        let gv = GridView::from_chunk_grid(&cg, 64);
        // [-1, 0] is below origin.
        assert!(gv.chunk_at_xy([-1, 0]).is_none());
        // [0, -1] is below origin (y).
        assert!(gv.chunk_at_xy([0, -1]).is_none());
        // [2, 0] is past chunks_x.
        assert!(gv.chunk_at_xy([2, 0]).is_none());
        // [0, 1] is past chunks_y.
        assert!(gv.chunk_at_xy([0, 1]).is_none());
        // [1, 0] is an empty slot inside the grid.
        assert!(gv.chunk_at_xy([1, 0]).is_none());
    }

    #[test]
    fn chunk_at_xy_handles_negative_origin() {
        let (slab_a, slab_b, cols_a, cols_b, mips_a, mips_b) = build_two_chunk_x_stripe();
        // Origin at [-1, 0]: chunks live at XY indices [-1, 0] and [0, 0].
        let chunks = [
            Some(GridView::from_parts(64, &slab_a, &cols_a, &mips_a)),
            Some(GridView::from_parts(64, &slab_b, &cols_b, &mips_b)),
        ];
        let cg = ChunkGrid {
            chunks: &chunks,
            origin_chunk_xy: [-1, 0],
            chunks_x: 2,
            chunks_y: 1,
        };
        let gv = GridView::from_chunk_grid(&cg, 64);
        let cm1 = gv.chunk_at_xy([-1, 0]).expect("chunk [-1, 0] present");
        assert_eq!(cm1.slab_buf, &slab_a[..]);
        let c0 = gv.chunk_at_xy([0, 0]).expect("chunk [0, 0] present");
        assert_eq!(c0.slab_buf, &slab_b[..]);
        // Past the right edge.
        assert!(gv.chunk_at_xy([1, 0]).is_none());
        // Past the left edge.
        assert!(gv.chunk_at_xy([-2, 0]).is_none());
    }

    #[test]
    fn from_chunk_grid_seeds_flat_fields_from_first_populated_chunk() {
        let (slab_a, slab_b, cols_a, cols_b, mips_a, mips_b) = build_two_chunk_x_stripe();
        // First slot empty, second populated. Flat fields should
        // seed from chunk B.
        let chunks = [
            None,
            Some(GridView::from_parts(64, &slab_b, &cols_b, &mips_b)),
        ];
        let cg = ChunkGrid {
            chunks: &chunks,
            origin_chunk_xy: [0, 0],
            chunks_x: 2,
            chunks_y: 1,
        };
        let gv = GridView::from_chunk_grid(&cg, 64);
        assert_eq!(gv.slab_buf, &slab_b[..]);
        assert_eq!(gv.chunk_size_xy, 64);
        // Single-chunk fields stay populated; tests beyond rely on
        // opticast's prelude refreshing them via the camera lookup.
        let _ = (slab_a, cols_a, mips_a);
    }

    #[test]
    fn aabb_xy_single_chunk_returns_0_to_vsid() {
        let mips = [0usize, 2];
        let gv = GridView::from_parts(2048, &[], &[], &mips);
        let (lo, hi) = gv.aabb_xy();
        assert_eq!(lo, [0, 0]);
        assert_eq!(hi, [2048, 2048]);
    }

    #[test]
    fn aabb_xy_multi_chunk_covers_full_extent() {
        let (slab_a, _, cols_a, _, mips_a, _) = build_two_chunk_x_stripe();
        // 2-wide × 3-tall chunk grid starting at origin [-1, 0].
        // Each chunk is 128² (matches the constructor's chunk_size_xy).
        let chunks = [
            Some(GridView::from_parts(128, &slab_a, &cols_a, &mips_a)),
            None,
            None,
            None,
            None,
            None,
        ];
        let cg = ChunkGrid {
            chunks: &chunks,
            origin_chunk_xy: [-1, 0],
            chunks_x: 2,
            chunks_y: 3,
        };
        let gv = GridView::from_chunk_grid(&cg, 128);
        let (lo, hi) = gv.aabb_xy();
        // xmin = -1 * 128 = -128; xmax = (-1 + 2) * 128 = 128.
        // ymin = 0; ymax = 3 * 128 = 384.
        assert_eq!(lo, [-128, 0]);
        assert_eq!(hi, [128, 384]);
    }

    #[test]
    fn from_chunk_grid_empty_table_falls_back_to_empty_slices() {
        let chunks: [Option<GridView<'_>>; 1] = [None];
        let cg = ChunkGrid {
            chunks: &chunks,
            origin_chunk_xy: [0, 0],
            chunks_x: 1,
            chunks_y: 1,
        };
        let gv = GridView::from_chunk_grid(&cg, 128);
        assert!(gv.slab_buf.is_empty());
        assert!(gv.column_offsets.is_empty());
        assert!(gv.mip_base_offsets.is_empty());
        assert_eq!(gv.vsid, 128);
        assert_eq!(gv.chunk_size_xy, 128);
        // The chunk_grid backend still gates chunk_at_xy.
        assert!(gv.chunk_at_xy([0, 0]).is_none());
    }
}
