//! KV6 sprite — a parsed [`Kv6`] paired with a world-space pose.
//!
//! Mirror of voxlap's `vx5sprite` (``) for the kv6
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
/// roxlap extension (XS.4), bit 4: this sprite does **not** cast a hard
/// shadow onto terrain / other sprites. Clear (the default) ⇒ it casts.
pub const SPRITE_FLAG_NO_SHADOW_CAST: u32 = 1 << 4;
/// roxlap extension (XS.4), bit 5: this sprite does **not** receive hard
/// shadows (it isn't darkened by occluders). Clear (the default) ⇒ it
/// receives. Both shadow bits default to participating, matching the
/// dynamic-lighting rig's "shadows on" intent; set a bit to opt a sprite
/// out (e.g. a glowing effect that shouldn't be shadowed).
pub const SPRITE_FLAG_NO_SHADOW_RECEIVE: u32 = 1 << 5;

/// The neutral (no-op) value for [`Sprite::tint`] — white, so every channel
/// multiplies by 1.0.
pub const NO_TINT: u32 = 0x00FF_FFFF;

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
    /// Voxel-material id (TV stage) applied uniformly to this sprite's
    /// voxels — indexes the renderer's global
    /// [`MaterialTable`](crate::material::MaterialTable) for opacity + blend
    /// mode. `0` (the default) is opaque, so an un-set sprite renders exactly
    /// as before. See `PORTING-TRANSPARENCY.md`.
    pub material: u8,
    /// Per-instance opacity multiplier (TV stage), `0..=255` (`255` =
    /// unscaled, the default). Scales the material's alpha so an effect can
    /// fade out by cheap per-frame updates without re-uploading its volume.
    pub alpha_mul: u8,
    /// Per-instance RGB colour tint, packed `0x00RRGGBB`. Each rendered voxel's
    /// colour is multiplied by this (per channel, `out = c * tint / 255`), so a
    /// host can recolour instances of one model cheaply. `0x00FF_FFFF` (white,
    /// the default) is a no-op. Alpha is **not** carried here — use a
    /// translucent material + [`alpha_mul`](Self::alpha_mul) for transparency.
    pub tint: u32,
    /// Per-voxel material colour map (TV.3): `(rgb, material_id)` pairs that
    /// classify this model's voxels into materials by colour — a mixed model
    /// (opaque frame + glass, bottle + potion). **Empty** (the default) means
    /// the whole sprite uses [`material`](Self::material) uniformly (the TV.1
    /// path). See [`crate::material::material_for_color`].
    pub material_map: Vec<(u32, u8)>,
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
            material: 0,
            alpha_mul: 255,
            tint: NO_TINT,
            material_map: Vec::new(),
        }
    }

    /// Set this sprite's RGB colour tint (`0x00RRGGBB`, white = no-op). Returns
    /// `self` for chaining. See [`tint`](Self::tint).
    #[must_use]
    pub fn with_tint(mut self, tint: u32) -> Self {
        self.tint = tint & 0x00FF_FFFF;
        self
    }

    /// XS.4 — whether this sprite casts a hard shadow (the [dynamic-lighting
    /// stage](../../../PORTING-DYNLIGHT.md) decides it darkens terrain / other
    /// sprites). `true` unless [`SPRITE_FLAG_NO_SHADOW_CAST`] is set.
    #[must_use]
    pub fn casts_shadow(&self) -> bool {
        self.flags & SPRITE_FLAG_NO_SHADOW_CAST == 0
    }

    /// XS.4 — whether this sprite receives hard shadows (is darkened by
    /// occluders). `true` unless [`SPRITE_FLAG_NO_SHADOW_RECEIVE`] is set.
    #[must_use]
    pub fn receives_shadow(&self) -> bool {
        self.flags & SPRITE_FLAG_NO_SHADOW_RECEIVE == 0
    }

    /// XS.4 — set whether this sprite casts a hard shadow (clears/sets
    /// [`SPRITE_FLAG_NO_SHADOW_CAST`]). Returns `self` for chaining.
    #[must_use]
    pub fn with_casts_shadow(mut self, casts: bool) -> Self {
        if casts {
            self.flags &= !SPRITE_FLAG_NO_SHADOW_CAST;
        } else {
            self.flags |= SPRITE_FLAG_NO_SHADOW_CAST;
        }
        self
    }

    /// XS.4 — set whether this sprite receives hard shadows (clears/sets
    /// [`SPRITE_FLAG_NO_SHADOW_RECEIVE`]). Returns `self` for chaining.
    #[must_use]
    pub fn with_receives_shadow(mut self, receives: bool) -> Self {
        if receives {
            self.flags &= !SPRITE_FLAG_NO_SHADOW_RECEIVE;
        } else {
            self.flags |= SPRITE_FLAG_NO_SHADOW_RECEIVE;
        }
        self
    }

    /// Carve a sphere out of this sprite's voxel model, controlling
    /// the colour of the interior the cut exposes. Thin delegate to
    /// [`Kv6::carve_sphere_with_colfunc`] — `centre` / `radius`,
    /// `solid`, and `colfunc` are all in **kv6-local** voxel
    /// coordinates (the sprite pose `p`/`s`/`h`/`f` is untouched). See
    /// that method for why the `solid` occupancy predicate is required.
    pub fn carve_sphere_with_colfunc<S, C>(
        &mut self,
        centre: [i32; 3],
        radius: u32,
        solid: S,
        colfunc: C,
    ) where
        S: Fn(i32, i32, i32) -> bool,
        C: Fn(i32, i32, i32) -> u32,
    {
        self.kv6
            .carve_sphere_with_colfunc(centre, radius, solid, colfunc);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// XS.4 — sprite shadow flags default to participating and toggle cleanly,
    /// independently of each other.
    #[test]
    fn sprite_shadow_flags_default_on_and_toggle() {
        let kv6 = Kv6::from_fn_shaded(2, 2, 2, |_, _, _| Some(0x8033_4455));
        let s = Sprite::axis_aligned(kv6, [0.0; 3]);
        // Default: casts + receives.
        assert!(s.casts_shadow() && s.receives_shadow());

        let no_cast = s.clone().with_casts_shadow(false);
        assert!(!no_cast.casts_shadow() && no_cast.receives_shadow());
        assert_eq!(
            no_cast.flags & SPRITE_FLAG_NO_SHADOW_CAST,
            SPRITE_FLAG_NO_SHADOW_CAST
        );

        let no_recv = s.clone().with_receives_shadow(false);
        assert!(no_recv.casts_shadow() && !no_recv.receives_shadow());

        // Toggling one bit leaves the other (and unrelated flags) untouched.
        let neither = s
            .clone()
            .with_casts_shadow(false)
            .with_receives_shadow(false);
        assert!(!neither.casts_shadow() && !neither.receives_shadow());
        let back = neither.with_casts_shadow(true);
        assert!(back.casts_shadow() && !back.receives_shadow());
    }

    #[test]
    fn carve_sphere_delegates_to_kv6_and_leaves_pose() {
        const BASE: u32 = 0x8033_4455;
        let kv6 = Kv6::from_fn_shaded(16, 16, 16, |_, _, _| Some(BASE));
        let mut sprite = Sprite::axis_aligned(kv6, [10.0, 20.0, 30.0]);
        let before = sprite.kv6.voxels.len();

        sprite.carve_sphere_with_colfunc([8, 8, 8], 4, |_, _, _| true, |_, _, _| 0x8000_FF00);

        // The hollowed-out shell has more surface voxels than the
        // solid hull did, so the model definitely changed.
        assert_ne!(sprite.kv6.voxels.len(), before);
        // Pose untouched — carving is kv6-local only.
        assert_eq!(sprite.p, [10.0, 20.0, 30.0]);
        assert_eq!(sprite.s, [1.0, 0.0, 0.0]);
    }
}
