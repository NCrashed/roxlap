//! Classical 3D Perlin noise + fractal Brownian motion (fBm) helper.
//!
//! Hand-rolled to avoid the `fastnoise2` C++ dep and the `noise`
//! crate's overhead. Reference: Ken Perlin's
//! [Improving Noise](https://mrl.cs.nyu.edu/~perlin/paper445.pdf)
//! (2002). The permutation table is shuffled deterministically via
//! [`crate::rng::SplitMix64`] so same-seed runs produce byte-stable
//! noise within a toolchain (cross-CPU FP determinism is best-
//! effort, same caveat as the renderer's tests).
//!
//! Used by [`crate::worley::worley_classify_grid`] as the perlin-
//! overlay term that breaks up Worley's clean cell facets, giving
//! caves an organic-looking surface roughness instead of straight
//! polygonal walls.

use crate::rng::SplitMix64;

const PERM_SIZE: usize = 256;

/// 3D Perlin noise sampler with a deterministic permutation table.
///
/// Output of [`PerlinNoise3D::sample`] is in roughly `[-1, 1]`; the
/// theoretical max for classical 3D Perlin is around `±√(3)/2 ≈
/// ±0.866`, so consumers shouldn't expect the full unit range.
/// [`PerlinNoise3D::fbm`] sums octaves and normalises by the
/// summed amplitude.
#[derive(Debug, Clone)]
pub struct PerlinNoise3D {
    /// Permutation table — `[0..256)` shuffled by the seed,
    /// duplicated for unbranched index wrapping.
    perm: [u8; PERM_SIZE * 2],
}

impl PerlinNoise3D {
    /// Build a Perlin sampler with a deterministic permutation
    /// derived from `seed`.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed);
        // Initialise table to the identity, then Fisher-Yates shuffle.
        let mut p = [0u8; PERM_SIZE];
        for (i, slot) in p.iter_mut().enumerate() {
            #[allow(clippy::cast_possible_truncation)]
            {
                *slot = i as u8;
            }
        }
        for i in (1..PERM_SIZE).rev() {
            #[allow(clippy::cast_possible_truncation)]
            let j = (rng.next_u64() as usize) % (i + 1);
            p.swap(i, j);
        }
        let mut perm = [0u8; PERM_SIZE * 2];
        perm[..PERM_SIZE].copy_from_slice(&p);
        perm[PERM_SIZE..].copy_from_slice(&p);
        Self { perm }
    }

    /// Single-octave 3D Perlin noise. Output roughly in `[-1, 1]`
    /// (theoretical bound `≈ ±0.866`).
    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::many_single_char_names
    )]
    pub fn sample(&self, x: f32, y: f32, z: f32) -> f32 {
        let xfl = x.floor();
        let yfl = y.floor();
        let zfl = z.floor();
        let xi = (xfl as i32) & 255;
        let yi = (yfl as i32) & 255;
        let zi = (zfl as i32) & 255;
        let xf = x - xfl;
        let yf = y - yfl;
        let zf = z - zfl;

        let u = fade(xf);
        let v = fade(yf);
        let w = fade(zf);

        let xi = xi as usize;
        let yi = yi as usize;
        let zi = zi as usize;

        let a = self.perm[xi] as usize;
        let aa = self.perm[a + yi] as usize;
        let ab = self.perm[a + yi + 1] as usize;
        let b = self.perm[xi + 1] as usize;
        let ba = self.perm[b + yi] as usize;
        let bb = self.perm[b + yi + 1] as usize;

        let aaa = self.perm[aa + zi];
        let aba = self.perm[ab + zi];
        let baa = self.perm[ba + zi];
        let bba = self.perm[bb + zi];
        let aab = self.perm[aa + zi + 1];
        let abb = self.perm[ab + zi + 1];
        let bab = self.perm[ba + zi + 1];
        let bbb = self.perm[bb + zi + 1];

        let x1 = lerp(grad(aaa, xf, yf, zf), grad(baa, xf - 1.0, yf, zf), u);
        let x2 = lerp(
            grad(aba, xf, yf - 1.0, zf),
            grad(bba, xf - 1.0, yf - 1.0, zf),
            u,
        );
        let y1 = lerp(x1, x2, v);

        let x3 = lerp(
            grad(aab, xf, yf, zf - 1.0),
            grad(bab, xf - 1.0, yf, zf - 1.0),
            u,
        );
        let x4 = lerp(
            grad(abb, xf, yf - 1.0, zf - 1.0),
            grad(bbb, xf - 1.0, yf - 1.0, zf - 1.0),
            u,
        );
        let y2 = lerp(x3, x4, v);

        lerp(y1, y2, w)
    }

    /// Multi-octave fractal Brownian motion. `octaves = 0` returns
    /// `0.0`; otherwise sums `octaves` Perlin samples at doubling
    /// frequencies + halving amplitudes, normalised by the total
    /// amplitude so the output stays in roughly `[-1, 1]`.
    ///
    /// `frequency` scales the input coordinates uniformly; pick it
    /// to control the lowest-octave wavelength relative to your
    /// world coordinates (e.g., `1.0 / 32.0` puts the lowest
    /// octave's wavelength at 32 voxels).
    #[must_use]
    pub fn fbm(&self, x: f32, y: f32, z: f32, octaves: u32, frequency: f32) -> f32 {
        if octaves == 0 {
            return 0.0;
        }
        let mut sum = 0.0f32;
        let mut amp = 1.0f32;
        let mut freq = frequency;
        let mut max_amp = 0.0f32;
        for _ in 0..octaves {
            sum += self.sample(x * freq, y * freq, z * freq) * amp;
            max_amp += amp;
            amp *= 0.5;
            freq *= 2.0;
        }
        sum / max_amp
    }
}

#[inline]
fn fade(t: f32) -> f32 {
    // Ken Perlin's smootherstep: 6t⁵ − 15t⁴ + 10t³.
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

#[inline]
#[allow(clippy::many_single_char_names)]
fn grad(hash: u8, x: f32, y: f32, z: f32) -> f32 {
    // Voxlap-equivalent of Ken's gradient table: pick one of 16
    // canonical gradients via the low 4 hash bits.
    let h = hash & 15;
    let u = if h < 8 { x } else { y };
    let v = if h < 4 {
        y
    } else if h == 12 || h == 14 {
        x
    } else {
        z
    };
    (if h & 1 == 0 { u } else { -u }) + (if h & 2 == 0 { v } else { -v })
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss, clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn sample_deterministic_for_same_seed() {
        let a = PerlinNoise3D::new(42);
        let b = PerlinNoise3D::new(42);
        for &p in &[(0.5, 0.5, 0.5), (3.7, 1.2, 7.9), (-2.5, 4.1, 0.3)] {
            assert_eq!(
                a.sample(p.0, p.1, p.2).to_bits(),
                b.sample(p.0, p.1, p.2).to_bits(),
                "Perlin sample at {p:?} should match across instances"
            );
        }
    }

    #[test]
    fn sample_different_seed_yields_different_values() {
        let a = PerlinNoise3D::new(1);
        let b = PerlinNoise3D::new(2);
        let p = (3.7, 1.2, 7.9);
        let va = a.sample(p.0, p.1, p.2);
        let vb = b.sample(p.0, p.1, p.2);
        assert!(
            (va - vb).abs() > 0.01,
            "different seeds should yield different samples (got {va} vs {vb})"
        );
    }

    #[test]
    fn sample_zero_at_integer_grid_points() {
        // Classical Perlin noise is zero at every integer grid
        // point — the gradient at the corner contributes 0 to all
        // adjacent cells.
        let n = PerlinNoise3D::new(7);
        for &(x, y, z) in &[(0.0, 0.0, 0.0), (1.0, 2.0, 3.0), (-5.0, 7.0, 11.0)] {
            let v = n.sample(x, y, z);
            assert!(
                v.abs() < 1e-5,
                "Perlin should be ~0 at integer grid point ({x},{y},{z}), got {v}"
            );
        }
    }

    #[test]
    fn sample_in_expected_range() {
        // Stochastic check: across many random sample positions, the
        // value stays comfortably in [-1, 1] (theoretical bound is
        // ~±0.866).
        let n = PerlinNoise3D::new(42);
        let mut max_abs = 0.0f32;
        for i in 0..10_000 {
            let s = i as f32 * 0.123;
            let v = n.sample(s, s * 1.7, s * 0.4);
            max_abs = max_abs.max(v.abs());
        }
        assert!(max_abs <= 1.0, "Perlin sample exceeded 1.0 (got {max_abs})");
        assert!(
            max_abs > 0.3,
            "Perlin sample suspiciously small over 10k draws (got {max_abs})"
        );
    }

    #[test]
    fn fbm_zero_octaves_is_zero() {
        let n = PerlinNoise3D::new(1);
        assert_eq!(n.fbm(1.0, 2.0, 3.0, 0, 1.0), 0.0);
    }

    #[test]
    fn fbm_single_octave_matches_sample() {
        let n = PerlinNoise3D::new(1);
        let p = (0.3, 0.7, 1.1);
        let single = n.sample(p.0, p.1, p.2);
        let fbm1 = n.fbm(p.0, p.1, p.2, 1, 1.0);
        assert_eq!(
            single.to_bits(),
            fbm1.to_bits(),
            "single-octave fBm should equal sample"
        );
    }

    #[test]
    fn fbm_stays_in_range() {
        let n = PerlinNoise3D::new(7);
        let mut max_abs = 0.0f32;
        for i in 0..1000 {
            let s = i as f32 * 0.123;
            let v = n.fbm(s, s * 1.7, s * 0.4, 4, 0.5);
            max_abs = max_abs.max(v.abs());
        }
        assert!(max_abs <= 1.0, "fBm exceeded 1.0 (got {max_abs})");
    }
}
