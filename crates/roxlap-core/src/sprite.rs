//! KV6 sprite type + the `draw_sprite` dispatcher.
//!
//! Mirror of voxlap's `vx5sprite` (voxlap5.h:63-79) plus the
//! `drawsprite` entry point (voxlap5.c:9818). For R6.1 the
//! dispatcher is a stub — just enough API surface for the host to
//! plumb a sprite reference through. R6.2-R6.4 fill in the actual
//! kv6 frustum-cull + per-voxel rasterization behind it.
//!
//! Voxlap's vx5sprite is a 64-byte struct:
//!
//! ```text
//! point3d p;       // position
//! int32_t flags;   // bit 0: 0=normal shading
//!                  // bit 1: 0=kv6data, 1=kfatype  (oracle uses 0)
//!                  // bit 2: 0=normal, 1=invisible
//! point3d s;       // x-basis (kv6data.xsiz direction)
//! kv6data *voxnum; // (or kfatype *kfaptr if flag bit 1 set)
//! point3d h;       // y-basis
//! int32_t kfatim;
//! point3d f;       // z-basis
//! int32_t okfatim;
//! ```
//!
//! For R6 we only handle kv6 sprites with `flags = 0` (the four
//! oracle sprite poses all use this). KFA animation + the no-z and
//! invisible flags are deferred.

use roxlap_formats::kv6::Kv6;

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
/// Mirror of voxlap's `vx5sprite` for the kv6 case (`flags &
/// SPRITE_FLAG_KFA == 0`). Owns its `Kv6` by value — the sprite
/// data the engine needs to access during rendering. `p` / `s` /
/// `h` / `f` are voxlap's per-axis world-space basis: `s` is
/// the `kv6.xsiz` direction, `h` the `ysiz` direction, `f` the
/// `zsiz` direction. For an axis-aligned sprite, `s = [1,0,0]`,
/// `h = [0,1,0]`, `f = [0,0,1]`.
#[derive(Debug, Clone)]
pub struct Sprite {
    /// Voxel data + bounding-box pivots. Built either by
    /// [`crate::meltsphere::meltsphere`] (extracted from a world)
    /// or loaded from a `.kv6` file via `roxlap_formats::kv6::parse`.
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
    /// Voxlap-style flags bitfield. See `SPRITE_FLAG_*` constants.
    pub flags: u32,
}

impl Sprite {
    /// Convenience constructor for an axis-aligned sprite at
    /// world position `pos`. Basis is identity, flags = 0
    /// (kv6 + normal shading + visible + z-tested).
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

/// Draw a sprite into the engine's framebuffer + z-buffer.
///
/// Top-level dispatcher mirroring voxlap5.c:9818-9828:
/// - Skips on `flags & INVISIBLE`.
/// - Picks `kv6draw` vs `kv6draw_noz` based on `flags & NO_Z`.
/// - Picks the kv6 vs kfa path based on `flags & KFA`.
///
/// **R6.1 status**: the dispatcher is a no-op stub for every
/// branch. The signature is fixed so R6.2 can plug in the real
/// frustum-cull + per-voxel iteration without further callsite
/// churn. Returns `false` so callers (and tests) can detect that
/// no actual rendering happened yet.
#[must_use]
pub fn draw_sprite(sprite: &Sprite) -> bool {
    if sprite.flags & SPRITE_FLAG_INVISIBLE != 0 {
        return false;
    }
    if sprite.flags & SPRITE_FLAG_KFA != 0 {
        // KFA animation path is out of scope for R6 (no oracle
        // pose exercises it). Mirror voxlap's silent dispatch but
        // without rendering anything.
        return false;
    }
    // R6.2+ will replace these stubs with the real kv6draw /
    // kv6draw_noz calls. For R6.1, just touch the fields we'll
    // need (suppresses dead-code warnings) and return.
    let _ = (sprite.kv6.xsiz, sprite.kv6.ysiz, sprite.kv6.zsiz);
    let _ = (sprite.p, sprite.s, sprite.h, sprite.f);
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use roxlap_formats::kv6::Kv6;

    fn empty_kv6() -> Kv6 {
        Kv6 {
            xsiz: 1,
            ysiz: 1,
            zsiz: 1,
            xpiv: 0.5,
            ypiv: 0.5,
            zpiv: 0.5,
            voxels: Vec::new(),
            xlen: vec![0],
            ylen: vec![vec![0]],
            palette: None,
        }
    }

    #[test]
    fn axis_aligned_sets_identity_basis() {
        // Compare bit patterns: these are integer-valued floats so
        // bit-equality is well-defined and dodges clippy::float_cmp.
        let bits = |a: [f32; 3]| a.map(f32::to_bits);
        let s = Sprite::axis_aligned(empty_kv6(), [10.0, 20.0, 30.0]);
        assert_eq!(bits(s.p), bits([10.0, 20.0, 30.0]));
        assert_eq!(bits(s.s), bits([1.0, 0.0, 0.0]));
        assert_eq!(bits(s.h), bits([0.0, 1.0, 0.0]));
        assert_eq!(bits(s.f), bits([0.0, 0.0, 1.0]));
        assert_eq!(s.flags, 0);
    }

    #[test]
    fn invisible_flag_skips_dispatch() {
        let mut s = Sprite::axis_aligned(empty_kv6(), [0.0; 3]);
        s.flags = SPRITE_FLAG_INVISIBLE;
        assert!(!draw_sprite(&s));
    }

    #[test]
    fn kfa_flag_skips_dispatch() {
        let mut s = Sprite::axis_aligned(empty_kv6(), [0.0; 3]);
        s.flags = SPRITE_FLAG_KFA;
        assert!(!draw_sprite(&s));
    }

    #[test]
    fn r61_stub_returns_false_for_normal_sprite() {
        // R6.1 stub: even a normally-flagged sprite "renders"
        // nothing yet. R6.2+ will flip this expectation.
        let s = Sprite::axis_aligned(empty_kv6(), [0.0; 3]);
        assert!(!draw_sprite(&s));
    }
}
