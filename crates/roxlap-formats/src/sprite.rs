//! KV6 sprite — a parsed [`Kv6`] paired with a world-space pose.
//!
//! Mirror of voxlap's `vx5sprite` (`voxlap5.h:63-79`) for the kv6
//! case (`flags & SPRITE_FLAG_KFA == 0`). Pure data — owns the
//! [`Kv6`] voxel grid plus the four `point3d` fields voxlap calls
//! `p` (pivot position), `s` (x-basis), `h` (y-basis), `f`
//! (z-basis), and the bitfield `flags`.
//!
//! Rendering lives in `roxlap-core` (`draw_sprite`); this module
//! only models the data shape so downstream tools (file
//! converters, model inspectors, asset pipelines) can build and
//! manipulate sprites without depending on the full renderer.
//!
//! Voxlap's 64-byte layout, for reference:
//!
//! ```text
//! point3d p;       // position
//! int32_t flags;   // bit 0: 0=normal shading
//!                  // bit 1: 0=kv6data, 1=kfatype
//!                  // bit 2: 0=normal, 1=invisible
//!                  // bit 3: 0=z-tested, 1=overlay (no-z)
//! point3d s;       // x-basis (kv6data.xsiz direction)
//! kv6data *voxnum; // (or kfatype *kfaptr if flag bit 1 set)
//! point3d h;       // y-basis
//! int32_t kfatim;
//! point3d f;       // z-basis
//! int32_t okfatim;
//! ```

use crate::kv6::Kv6;

/// Voxlap's sprite-flags bit 0: disable normal-based face shading.
pub const SPRITE_FLAG_NO_SHADING: u32 = 1 << 0;
/// Voxlap's sprite-flags bit 1: voxnum points at a `kfatype`
/// (animated). When clear (default), points at a `kv6data`.
pub const SPRITE_FLAG_KFA: u32 = 1 << 1;
/// Voxlap's sprite-flags bit 2: skip rendering entirely.
pub const SPRITE_FLAG_INVISIBLE: u32 = 1 << 2;
/// Voxlap's sprite-flags bit 3: render without z-buffer test.
pub const SPRITE_FLAG_NO_Z: u32 = 1 << 3;

/// A KV6 voxel sprite positioned in world space.
///
/// Mirror of voxlap's `vx5sprite` for the kv6 case
/// (`flags & SPRITE_FLAG_KFA == 0`; see [`SPRITE_FLAG_KFA`]).
/// Owns its [`Kv6`] by value. `p` / `s` / `h` / `f` are voxlap's
/// per-axis world-space basis: `s` is the `kv6.xsiz` direction,
/// `h` the `ysiz` direction, `f` the `zsiz` direction. For an
/// axis-aligned sprite, `s = [1,0,0]`, `h = [0,1,0]`,
/// `f = [0,0,1]`.
#[derive(Debug, Clone)]
pub struct Sprite {
    /// Voxel data + bounding-box pivots. Loaded from a `.kv6`
    /// file via [`crate::kv6::parse`] or built procedurally.
    pub kv6: Kv6,
    /// World-space position of the sprite's pivot (xpiv, ypiv,
    /// zpiv inside the kv6 maps to this point).
    pub p: [f32; 3],
    /// World-space basis vector for the kv6's local +x. Length
    /// scales the sprite along that axis (typically `1.0` for
    /// unit-scale).
    pub s: [f32; 3],
    /// World-space basis vector for the kv6's local +y.
    pub h: [f32; 3],
    /// World-space basis vector for the kv6's local +z.
    pub f: [f32; 3],
    /// Voxlap-style flags bitfield. See [`SPRITE_FLAG_NO_SHADING`],
    /// [`SPRITE_FLAG_KFA`], [`SPRITE_FLAG_INVISIBLE`],
    /// [`SPRITE_FLAG_NO_Z`].
    pub flags: u32,
}

impl Sprite {
    /// Convenience constructor for an axis-aligned sprite at
    /// world position `pos`. Basis is identity, flags = 0
    /// (kv6 + normal shading + visible + z-tested).
    ///
    /// # Examples
    ///
    /// ```
    /// use roxlap_formats::kv6::Kv6;
    /// use roxlap_formats::sprite::Sprite;
    ///
    /// # let kv6 = Kv6 {
    /// #     xsiz: 1, ysiz: 1, zsiz: 1,
    /// #     xpiv: 0.5, ypiv: 0.5, zpiv: 0.5,
    /// #     voxels: vec![], xlen: vec![0], ylen: vec![vec![0]],
    /// #     palette: None,
    /// # };
    /// // ... after `let kv6 = kv6::parse(&bytes)?;` or similar:
    /// let sprite = Sprite::axis_aligned(kv6, [1024.0, 1024.0, 100.0]);
    /// assert_eq!(sprite.flags, 0);
    /// assert_eq!(sprite.s, [1.0, 0.0, 0.0]);
    /// ```
    #[must_use]
    pub fn axis_aligned(kv6: Kv6, pos: [f32; 3]) -> Self {
        Self {
            kv6,
            p: pos,
            s: [1.0, 0.0, 0.0],
            h: [0.0, 1.0, 0.0],
            f: [0.0, 0.0, 1.0],
            flags: 0,
        }
    }
}
