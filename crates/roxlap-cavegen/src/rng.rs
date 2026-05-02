//! Tiny deterministic PRNG shared by [`crate::worley`] and
//! [`crate::perlin`]. Reference: `SplitMix64`,
//! <http://prng.di.unimi.it/splitmix64.c>.
//!
//! Crate-private — exposed so worley + perlin can share a seed
//! convention, not part of the public API.

pub(crate) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
        }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform `f32` in `[0.0, 1.0)`. 24 random bits → exact f32
    /// mantissa.
    #[allow(clippy::cast_precision_loss)]
    pub(crate) fn next_f32_unit(&mut self) -> f32 {
        let bits = self.next_u64() >> (64 - 24);
        (bits as f32) / ((1u32 << 24) as f32)
    }
}
