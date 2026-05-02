//! Worley cave-shape classification.
//!
//! Algorithm (Tom Dobrowolski's `CaveGen`, simplified):
//!
//! 1. Place `seed_count` random seeds uniformly in the voxel
//!    volume. The first `air_ratio × seed_count` are tagged "air";
//!    the rest are tagged "solid".
//! 2. For each voxel `p`:
//!    - `d_a` = distance to the nearest **air** seed
//!    - `d_s` = distance to the nearest **solid** seed
//!    - The voxel is air iff `d_a < d_s`, else solid.
//! 3. (CD.5.3) Add a Perlin overlay to `d_a` to break up the clean
//!    Worley facets. Currently disabled (overlay = 0).
//!
//! `anisotropy` scales the z component of the distance: `> 1.0`
//! produces caves that are wider than tall (mineshaft-style); `< 1.0`
//! gives tall narrow caves; `1.0` is isotropic.
//!
//! Determinism: same `seed` + same params + same code → byte-stable
//! grid (within toolchain — cross-CPU FP might diverge slightly,
//! same caveat as the rendering tests).

use crate::{CaveParams, MAXZDIM};

/// One Worley seed point.
#[derive(Debug, Clone, Copy)]
pub struct Seed {
    /// World coordinates `(x, y, z)`.
    pub pos: [f32; 3],
    /// `true` for air-tagged seeds, `false` for solid-tagged.
    pub is_air: bool,
}

/// Place `params.seed_count` deterministic random seeds across the
/// `vsid × vsid × MAXZDIM` volume. The first `params.air_ratio ×
/// params.seed_count` (rounded) are tagged air; the rest solid.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
pub fn place_seeds(params: &CaveParams, vsid: u32) -> Vec<Seed> {
    let mut rng = SplitMix64::new(params.seed);
    let total = params.seed_count;
    // air_ratio is clamped to [0.0, 1.0] by the caller; we trust it here.
    let n_air = ((total as f32) * params.air_ratio).round() as usize;
    let n_air = n_air.min(total);
    let xy_max = vsid as f32;
    let z_max = MAXZDIM as f32;
    let mut seeds = Vec::with_capacity(total);
    for i in 0..total {
        let pos_x = rng.next_f32_unit() * xy_max;
        let pos_y = rng.next_f32_unit() * xy_max;
        let pos_z = rng.next_f32_unit() * z_max;
        seeds.push(Seed {
            pos: [pos_x, pos_y, pos_z],
            is_air: i < n_air,
        });
    }
    seeds
}

/// Classify a single voxel as solid (`true`) or air (`false`)
/// against the seed set. The voxel position `(x, y, z)` is in
/// integer voxel coordinates.
///
/// CD.5.3 will fold a Perlin overlay into `d_a` here.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn classify_voxel(seeds: &[Seed], x: u32, y: u32, z: i32, anisotropy: f32) -> bool {
    let p = [x as f32, y as f32, z as f32];
    let mut d_air_sq = f32::INFINITY;
    let mut d_solid_sq = f32::INFINITY;
    for seed in seeds {
        let d_sq = anisotropic_dist_sq(p, seed.pos, anisotropy);
        if seed.is_air {
            if d_sq < d_air_sq {
                d_air_sq = d_sq;
            }
        } else if d_sq < d_solid_sq {
            d_solid_sq = d_sq;
        }
    }
    // d_a < d_s means closer to air seed → voxel is air.
    // Comparing squared distances is fine since both are ≥ 0.
    d_air_sq >= d_solid_sq
}

/// Build the dense `(VSID × VSID × MAXZDIM)` solidness grid by
/// classifying every voxel against the seed set. Output: 1 byte
/// per voxel; non-zero = solid.
///
/// O(`vsid² × MAXZDIM × seed_count`). At `vsid = 256`,
/// `seed_count = 128` this is ~2 G ops — minutes on a single core.
/// CD.5.3+ will add spatial bucketing if perf bites.
#[must_use]
#[allow(clippy::cast_sign_loss)]
pub fn worley_classify_grid(params: &CaveParams, vsid: u32) -> Vec<u8> {
    let seeds = place_seeds(params, vsid);
    let vsid_u = vsid as usize;
    let maxzdim_u = MAXZDIM as usize;
    let n_voxels = vsid_u * vsid_u * maxzdim_u;
    let mut grid = vec![0u8; n_voxels];
    for y in 0..vsid {
        for x in 0..vsid {
            for z in 0..MAXZDIM {
                let idx = (y as usize * vsid_u + x as usize) * maxzdim_u + z as usize;
                if classify_voxel(&seeds, x, y, z, params.anisotropy) {
                    grid[idx] = 1;
                }
            }
        }
    }
    grid
}

#[inline]
fn anisotropic_dist_sq(a: [f32; 3], b: [f32; 3], anisotropy: f32) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = (a[2] - b[2]) * anisotropy;
    dx * dx + dy * dy + dz * dz
}

// ---- splitmix64 PRNG ------------------------------------------------
//
// Tiny deterministic PRNG, no external dep. Reference:
// http://prng.di.unimi.it/splitmix64.c. Produces u64 streams indexed
// by an internal counter; `next_f32_unit` maps to `[0.0, 1.0)`.

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform `f32` in `[0.0, 1.0)`. 24 random bits → exact
    /// representation as f32 mantissa.
    #[allow(clippy::cast_precision_loss)]
    fn next_f32_unit(&mut self) -> f32 {
        let bits = self.next_u64() >> (64 - 24);
        (bits as f32) / ((1u32 << 24) as f32)
    }
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
mod tests {
    use super::*;

    fn test_params(seed: u64, seed_count: usize, air_ratio: f32) -> CaveParams {
        CaveParams {
            seed,
            seed_count,
            air_ratio,
            anisotropy: 1.0,
            perlin_octaves: 0,
            perlin_amplitude: 0.0,
        }
    }

    #[test]
    fn place_seeds_deterministic_in_seed() {
        let p = test_params(42, 16, 0.5);
        let a = place_seeds(&p, 64);
        let b = place_seeds(&p, 64);
        assert_eq!(a.len(), b.len());
        for (sa, sb) in a.iter().zip(b.iter()) {
            assert_eq!(sa.pos[0].to_bits(), sb.pos[0].to_bits(), "x");
            assert_eq!(sa.pos[1].to_bits(), sb.pos[1].to_bits(), "y");
            assert_eq!(sa.pos[2].to_bits(), sb.pos[2].to_bits(), "z");
            assert_eq!(sa.is_air, sb.is_air);
        }
    }

    #[test]
    fn place_seeds_different_seed_yields_different_seeds() {
        let a = place_seeds(&test_params(1, 16, 0.5), 64);
        let b = place_seeds(&test_params(2, 16, 0.5), 64);
        // Should differ on every position; allow at most a quarter
        // of accidental coincidences.
        let same = a
            .iter()
            .zip(b.iter())
            .filter(|(x, y)| x.pos[0].to_bits() == y.pos[0].to_bits())
            .count();
        assert!(same * 4 < a.len(), "too many coincident x positions");
    }

    #[test]
    fn place_seeds_air_ratio_split() {
        let p = test_params(7, 100, 0.4);
        let seeds = place_seeds(&p, 64);
        let n_air = seeds.iter().filter(|s| s.is_air).count();
        assert_eq!(n_air, 40, "40% of 100 seeds tagged air");
    }

    #[test]
    fn place_seeds_within_volume_bounds() {
        let p = test_params(7, 256, 0.5);
        let seeds = place_seeds(&p, 64);
        for s in &seeds {
            assert!((0.0..64.0).contains(&s.pos[0]), "x in bounds");
            assert!((0.0..64.0).contains(&s.pos[1]), "y in bounds");
            assert!((0.0..MAXZDIM as f32).contains(&s.pos[2]), "z in bounds");
        }
    }

    #[test]
    fn classify_at_air_seed_returns_air() {
        // One air seed at (10, 20, 30); one solid seed far away. Voxel
        // at (10, 20, 30) should be air.
        let seeds = vec![
            Seed {
                pos: [10.0, 20.0, 30.0],
                is_air: true,
            },
            Seed {
                pos: [100.0, 100.0, 100.0],
                is_air: false,
            },
        ];
        assert!(!classify_voxel(&seeds, 10, 20, 30, 1.0), "should be air");
    }

    #[test]
    fn classify_at_solid_seed_returns_solid() {
        let seeds = vec![
            Seed {
                pos: [100.0, 100.0, 100.0],
                is_air: true,
            },
            Seed {
                pos: [10.0, 20.0, 30.0],
                is_air: false,
            },
        ];
        assert!(classify_voxel(&seeds, 10, 20, 30, 1.0), "should be solid");
    }

    #[test]
    fn classify_anisotropy_squishes_caves_vertically() {
        // One air seed at (10, 20, 30); one solid seed at (10, 20, 50).
        // Voxel at (10, 20, 40) — equidistant in isotropic, so the
        // tiebreak goes to solid (d_air >= d_solid → solid).
        // With aniso=2, z distance is amplified, but still
        // equidistant — same outcome.
        let seeds = vec![
            Seed {
                pos: [10.0, 20.0, 30.0],
                is_air: true,
            },
            Seed {
                pos: [10.0, 20.0, 50.0],
                is_air: false,
            },
        ];
        // Move voxel slightly toward air seed.
        assert!(
            !classify_voxel(&seeds, 10, 20, 39, 1.0),
            "isotropic — closer to air seed"
        );
        // Same with aniso=2 — z distance amplified equally for both.
        assert!(
            !classify_voxel(&seeds, 10, 20, 39, 2.0),
            "aniso=2 — closer to air seed"
        );
    }

    #[test]
    #[allow(clippy::naive_bytecount)]
    fn worley_classify_grid_air_ratio_roughly_matches() {
        // Small world, count solid vs air. With air_ratio=0.5 we
        // expect ~50% air.
        let p = test_params(7, 32, 0.5);
        let vsid = 16u32;
        let grid = worley_classify_grid(&p, vsid);
        let n_air = grid.iter().filter(|&&b| b == 0).count();
        let total = grid.len();
        let ratio = n_air as f32 / total as f32;
        assert!(
            (0.30..=0.70).contains(&ratio),
            "expected ~50% air, got {:.2}",
            ratio * 100.0
        );
    }

    #[test]
    fn worley_classify_grid_deterministic_in_seed() {
        let p = test_params(1234, 32, 0.5);
        let g1 = worley_classify_grid(&p, 16);
        let g2 = worley_classify_grid(&p, 16);
        assert_eq!(g1, g2, "same seed → byte-stable grid");
    }
}
