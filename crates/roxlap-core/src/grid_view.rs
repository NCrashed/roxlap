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
//! * S4B.2.a (this file) — `chunk_size_xy` field + `chunk_at_xy`
//!   method. Today's single-chunk callers set `chunk_size_xy =
//!   vsid` and the lookup only succeeds for `[0, 0]`. The seam
//!   exists for S4B.2.b's grouscan column-step swap.
//! * S4B.2.b — grouscan column-step calls `chunk_at_xy` and swaps
//!   active per-chunk `(slab_buf, column_offsets)` when `(cx, cy)`
//!   crosses a chunk boundary.
//! * S4B.2.c — new multi-chunk constructor (scene-side) + a 32×32
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

    /// S4B.2.a: chunk lookup for the cross-chunk-XY DDA.
    ///
    /// Returns the [`GridView`] for the chunk at XY index
    /// `chunk_idx` if one exists, `None` otherwise. Today's single-
    /// chunk callers store one chunk under index `[0, 0]`; any other
    /// index returns `None` (the grouscan column-step swap will treat
    /// that as an empty chunk, matching the OOB-XY behaviour the
    /// existing 2D-DDA already produces for columns past
    /// `[0, vsid)`).
    ///
    /// **Today's degenerate behaviour.** Because every single-chunk
    /// caller has `chunk_size_xy == vsid`, the column-step swap in
    /// `grouscan` never fires within `[0, vsid)` — the boundary is
    /// at `cx = vsid`, which is already OOB. S4B.2.b wires the swap;
    /// S4B.2.c lands the first caller with `chunk_size_xy < vsid`.
    #[must_use]
    pub fn chunk_at_xy(&self, chunk_idx: [i32; 2]) -> Option<GridView<'a>> {
        if chunk_idx == [0, 0] {
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
}
