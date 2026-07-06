//! QE-B6 — the packed-colour newtype family.
//!
//! roxlap inherited **three distinct `u32` colour packings** from
//! voxlap, and every API taking a bare `u32` invited mixing them up
//! (the classic: passing `0xFF...` "opaque white" where the high byte
//! is *brightness*, getting an over-bright voxel). The newtypes make
//! each packing a distinct type, so a mix-up is a compile error:
//!
//! | Type | Packing | High byte means | Used by |
//! |---|---|---|---|
//! | [`VoxColor`] | `0xBB_RR_GG_BB` | baked brightness/AO (`0x80` neutral) | voxels: edits, kv6/vxl builders, colour queries |
//! | [`Rgb`] | `0x00_RR_GG_BB` | nothing (must be 0) | tints, sky/fog/clear colours, colour→material map keys |
//! | [`OverlayColor`] | `0xAA_RR_GG_BB` | real alpha | overlay lines (`Line3`) |
//!
//! All three are `#[repr(transparent)]` wrappers with a **public**
//! `.0` field: on-disk formats and GPU buffers keep raw `u32`s, and
//! wrapping/unwrapping at the boundary is free and explicit.

/// A voxel colour in voxlap's packing: RGB in the low 24 bits, the
/// baked **brightness/AO byte** on top — `0x80` is neutral ("unlit"),
/// and lighting bakes rewrite it per voxel. NOT alpha: translucency
/// is a material property (see the material palette), never a colour
/// channel.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct VoxColor(pub u32);

impl VoxColor {
    /// A voxel colour at the neutral `0x80` brightness.
    #[must_use]
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self(0x8000_0000 | ((r as u32) << 16) | ((g as u32) << 8) | b as u32)
    }

    /// Replace the brightness byte (bakes do this per voxel; hand-set
    /// it only for pre-shaded art).
    #[must_use]
    pub const fn with_brightness(self, brightness: u8) -> Self {
        Self((self.0 & 0x00ff_ffff) | ((brightness as u32) << 24))
    }

    /// The colour part as an [`Rgb`] (brightness stripped) — e.g. to
    /// tint debris particles with the voxel colour they were carved
    /// from.
    #[must_use]
    pub const fn rgb_part(self) -> Rgb {
        Rgb(self.0 & 0x00ff_ffff)
    }
}

/// A plain `0x00_RR_GG_BB` colour: sprite/actor/particle **tints**,
/// the sky / fog / clear colours, and the colour keys of
/// colour→material maps. No packing surprises — the high byte is
/// unused and must stay zero.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Rgb(pub u32);

impl Rgb {
    /// The no-op tint.
    pub const WHITE: Self = Self(0x00ff_ffff);

    #[must_use]
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self(((r as u32) << 16) | ((g as u32) << 8) | b as u32)
    }
}

/// An overlay colour with **real alpha**: `0xAA_RR_GG_BB` (`0xFF` =
/// opaque). Used by the overlay-line API ([`Line3`]) — the one place
/// in the engine where the high byte genuinely is opacity.
///
/// [`Line3`]: https://docs.rs/roxlap-render
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct OverlayColor(pub u32);

impl OverlayColor {
    #[must_use]
    pub const fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self(((a as u32) << 24) | ((r as u32) << 16) | ((g as u32) << 8) | b as u32)
    }

    /// Fully opaque.
    #[must_use]
    pub const fn opaque(r: u8, g: u8, b: u8) -> Self {
        Self::rgba(r, g, b, 0xff)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packings_match_their_wire_layouts() {
        assert_eq!(VoxColor::rgb(0x4d, 0x8a, 0x3a).0, 0x804d_8a3a);
        assert_eq!(
            VoxColor::rgb(0x4d, 0x8a, 0x3a).with_brightness(0xff).0,
            0xff4d_8a3a
        );
        assert_eq!(VoxColor(0x804d_8a3a).rgb_part(), Rgb(0x004d_8a3a));
        assert_eq!(Rgb::new(0x8f, 0xbc, 0xd4).0, 0x008f_bcd4);
        assert_eq!(OverlayColor::rgba(0xff, 0xd0, 0x40, 0xff).0, 0xffff_d040);
        assert_eq!(
            OverlayColor::opaque(1, 2, 3),
            OverlayColor::rgba(1, 2, 3, 0xff)
        );
    }
}
