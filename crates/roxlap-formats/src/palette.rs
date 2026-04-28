//! Shared 6-bit-per-channel palette types used by `.kvx` and `.kv6`.
//!
//! Voxlap's on-disk palettes encode each RGB component as 6 bits
//! (`0..=63`). The engine widens to 8-bit by left-shifting 2 (low 2 bits
//! zero), then sets the `0x80000000` "brightness" bit when packing to
//! a 32-bit colour word. [`Rgb6::to_voxlap_argb`] performs that
//! conversion exactly as voxlaptest's `setkvx` / `loadkv6` do.

/// A 6-bit-per-channel palette entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Rgb6 {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Rgb6 {
    /// Voxlap-style 32-bit packed colour: each component widened to 8
    /// bits with the low 2 bits zero, plus the `0x80000000` brightness
    /// bit the engine treats as "full intensity".
    #[must_use]
    pub fn to_voxlap_argb(self) -> u32 {
        0x8000_0000
            | (u32::from(self.r) << 18)
            | (u32::from(self.g) << 10)
            | (u32::from(self.b) << 2)
    }
}
