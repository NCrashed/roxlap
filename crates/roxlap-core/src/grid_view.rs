//! Per-frame voxel-world borrow shape.
//!
//! Wraps the `(vsid, slab_buf, column_offsets, mip_base_offsets)`
//! tuple that [`crate::opticast`] and
//! [`crate::scalar_rasterizer::ScalarRasterizer`] both need. Today
//! always represents a single chunk so callers building from a
//! [`roxlap_formats::vxl::Vxl`] keep the existing flat-world
//! semantics byte-identically.
//!
//! Substage S4B.0 (the "GridView abstraction"). The shape is
//! introduced as a pure rename — opticast still drives a single
//! flat world, just behind a typed borrow. Subsequent S4B.x
//! sub-substages grow this into a multi-chunk view:
//!
//! * S4B.1 — carry the camera's chunk index alongside the borrow.
//! * S4B.2 — `chunk_at_xy(idx) -> Option<...>` lookup for the
//!   cross-chunk-XY 3D-DDA.
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
    /// Square world dimension (matches the source `Vxl`).
    pub vsid: u32,
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
    #[must_use]
    pub fn from_parts(
        vsid: u32,
        slab_buf: &'a [u8],
        column_offsets: &'a [u32],
        mip_base_offsets: &'a [usize],
    ) -> Self {
        Self {
            vsid,
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
            slab_buf: &vxl.data,
            column_offsets: &vxl.column_offset,
            mip_base_offsets: &vxl.mip_base_offsets,
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
}
