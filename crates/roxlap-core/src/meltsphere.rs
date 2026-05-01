//! Sphere-region voxel extraction (`meltsphere`) and its support
//! tables.
//!
//! Port of voxlap's `meltsphere` (voxlap5.c:10222-10344). Walks an
//! AABB-bounded sphere of voxels in the world and packages the
//! visible (border) voxels into a fresh `kv6` sprite.
//!
//! R6.0b/c (this file) ships the supporting helpers:
//! - [`lightvox`] — alpha-byte face shader (voxlap5.c:623-632).
//! - [`PowerTables`] + [`build_tempfloatbuf`] — the factr / logint /
//!   tempfloatbuf machinery (voxlap5.c:118-120 statics,
//!   :12224-12236 init, :10240-10252 per-meltsphere-call build).
//!
//! R6.0d will land [`meltsphere`] itself on top of these.

// Entire module is a port of voxlap C bit-twiddle; the casts
// implicit in the source map cleanly to Rust's `as` casts.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]

/// Voxlap's `SETSPHMAXRAD` (voxlap5.c:117). Upper bound on
/// `hitrad`; the fast-pow tables are sized to this.
pub const SETSPHMAXRAD: usize = 256;

/// Apply alpha-byte face shading to a packed voxlap colour.
///
/// Port of `lightvox` (voxlap5.c:623-632). The high byte of `i` is
/// treated as a brightness multiplier (`0x80` is neutral); each RGB
/// channel is multiplied by it, shifted right by 7, and clamped to
/// 255. The returned colour has its alpha byte cleared.
///
/// Voxlap uses this to bake alpha-byte intensity into the per-voxel
/// colour stored in a kv6 sprite: meltsphere copies the world's
/// `BR(rgb)`-style packed colour through `lightvox` so the resulting
/// `Voxel::col` is plain `0x00rrggbb`.
#[must_use]
pub fn lightvox(i: u32) -> u32 {
    let b = i >> 24;
    let r = ((((i >> 16) & 0xff) * b) >> 7).min(255);
    let g = ((((i >> 8) & 0xff) * b) >> 7).min(255);
    let bl = (((i & 0xff) * b) >> 7).min(255);
    (r << 16) | (g << 8) | bl
}

/// Tables that voxlap precomputes once in `initvoxlap` and reuses
/// across every `meltsphere` / `setsphere` call.
///
/// Both tables are indexed by an integer `z ∈ [0, SETSPHMAXRAD)`:
///
/// - `factr[z]` is voxlap's prime-decomposition cache. If `z` is
///   prime, `factr[z][0] == 0`. If `z` is composite,
///   `factr[z][0] * factr[z][1] == z` and `factr[z][0]` is the
///   smallest prime divisor of `z`. `factr[2][0]` is forced to 0.
/// - `logint[z] = ln(z)` (natural log) for `z >= 1`; `logint[0]`
///   is unused.
///
/// Used by [`PowerTables::build_tempfloatbuf`] to compute
/// `i^curpow` cheaply: prime indices go through `exp(ln(i) *
/// curpow)`, composite indices fold into a multiplication of two
/// already-computed entries (avoiding `pow` per index).
#[derive(Debug, Clone)]
pub struct PowerTables {
    pub factr: [[u32; 2]; SETSPHMAXRAD],
    pub logint: [f64; SETSPHMAXRAD],
}

impl Default for PowerTables {
    fn default() -> Self {
        Self::new()
    }
}

impl PowerTables {
    /// Mirror of voxlap's `initvoxlap` factr-/logint-init block
    /// (voxlap5.c:12224-12236). Sieves the prime decomposition and
    /// fills `logint[i] = ln(i)`.
    #[must_use]
    pub fn new() -> Self {
        let mut factr = [[0u32; 2]; SETSPHMAXRAD];

        // Voxlap's hand-written prime sieve. `i` tracks the largest
        // prime ≤ √z; `j` is the next perfect square at which `i`
        // increments by 2. `k` is the previous prime — `factr[k][1]`
        // is rewritten on every iteration so that, by the time the
        // loop has crossed the next prime, `factr[k][1]` holds that
        // next prime (used to walk the prime list during composite
        // testing).
        factr[2][0] = 0;
        let mut i: u32 = 1;
        let mut j: u32 = 9;
        let mut k: usize = 0;
        let mut z: u32 = 3;
        while (z as usize) < SETSPHMAXRAD {
            if z == j {
                j += (i << 2) + 12;
                i += 2;
            }
            factr[z as usize][0] = 0;
            factr[k][1] = z;
            // Walk primes ≤ i looking for a divisor of z.
            let mut zz: u32 = 3;
            while zz <= i {
                if z % zz == 0 {
                    factr[z as usize][0] = zz;
                    factr[z as usize][1] = z / zz;
                    break;
                }
                zz = factr[zz as usize][1];
            }
            if factr[z as usize][0] == 0 {
                k = z as usize;
            }
            // Even number z + 1 is always 2 × ((z+1)/2).
            if (z as usize) + 1 < SETSPHMAXRAD {
                factr[(z as usize) + 1][0] = (z + 1) >> 1;
                factr[(z as usize) + 1][1] = 2;
            }
            z += 2;
        }

        let mut logint = [0.0f64; SETSPHMAXRAD];
        // logint[0] stays 0.0 (unused; voxlap leaves it uninitialised
        // but the meltsphere loop starts at i=2 after special-casing
        // tempfloatbuf[1]=1.0).
        for (zz, slot) in logint.iter_mut().enumerate().skip(1) {
            *slot = f64::ln(zz as f64);
        }

        Self { factr, logint }
    }

    /// Build a per-call `tempfloatbuf` such that
    /// `tempfloatbuf[i] ≈ i.powf(curpow)` for `i ∈ [0, hitrad]`.
    /// `hitrad + 1` is filled with the IEEE-754 max-finite-float
    /// sentinel (`0x7f7fffff`) so the int-bit-pattern comparisons
    /// in meltsphere terminate cleanly at the table edge.
    ///
    /// Port of voxlap5.c:10240-10252. `hitrad` is clamped to
    /// `SETSPHMAXRAD - 2`.
    #[must_use]
    pub fn build_tempfloatbuf(&self, hitrad: i32, curpow: f32) -> [f32; SETSPHMAXRAD] {
        let mut buf = [0.0f32; SETSPHMAXRAD];
        let hitrad_clamped = (hitrad.max(0) as usize).min(SETSPHMAXRAD - 2);

        buf[0] = 0.0;
        if hitrad_clamped >= 1 {
            buf[1] = 1.0;
        }
        // Voxlap mixes f64 / f32 here: logint[i] is f64, curpow is
        // f32 promoted to f64 for the multiply, exp is f64,
        // assignment to tempfloatbuf truncates to f32.
        let curpow_d = f64::from(curpow);
        for i in 2..=hitrad_clamped {
            if self.factr[i][0] == 0 {
                // Prime: tempfloatbuf[i] = exp(log(i) * curpow).
                buf[i] = (self.logint[i] * curpow_d).exp() as f32;
            } else {
                // Composite: factor[a] * factor[b] where a*b == i.
                let a = self.factr[i][0] as usize;
                let b = self.factr[i][1] as usize;
                buf[i] = buf[a] * buf[b];
            }
        }
        // Sentinel: 0x7f7fffff (= f32::MAX bit pattern) lives at
        // hitrad + 1 so the binary-search loops in meltsphere
        // hit a "guaranteed > anything finite" boundary.
        buf[hitrad_clamped + 1] = f32::from_bits(0x7f7f_ffff);
        buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- lightvox -------------------------------------------------------

    #[test]
    fn lightvox_neutral_brightness_passes_rgb_through() {
        // Alpha 0x80 = 128, multiplier (x * 128) >> 7 = x.
        assert_eq!(lightvox(0x80ff_4030), 0x00ff_4030);
        assert_eq!(lightvox(0x80ff_ffff), 0x00ff_ffff);
        assert_eq!(lightvox(0x8000_0000), 0x0000_0000);
    }

    #[test]
    fn lightvox_zero_alpha_blackens() {
        assert_eq!(lightvox(0x00ff_ffff), 0);
        assert_eq!(lightvox(0x0080_4020), 0);
    }

    #[test]
    fn lightvox_clamps_at_255() {
        // Alpha 0xff = 255; multiplier (0xff * 255) >> 7 = 510 → clamp 255.
        assert_eq!(lightvox(0xffff_ffff), 0x00ff_ffff);
        // Alpha 0xc0 = 192; (0x80 * 192) >> 7 = 0xc0 = 192. Not clamped.
        assert_eq!(lightvox(0xc080_8080), 0x00c0_c0c0);
        // Alpha 0xc0 with 0xff channel: (0xff * 192) >> 7 = 382 → clamp 255.
        assert_eq!(lightvox(0xc0ff_4030), 0x00ff_6048);
    }

    #[test]
    fn lightvox_half_brightness() {
        // Alpha 0x40 = 64; (x * 64) >> 7 = x / 2.
        assert_eq!(lightvox(0x4080_8080), 0x0040_4040);
    }

    // --- factr / prime sieve --------------------------------------------

    #[test]
    fn factr_known_primes_have_zero_factor() {
        let pt = PowerTables::new();
        // factr[2][0] is force-zero in init.
        for &p in &[
            2u32, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 251,
        ] {
            assert_eq!(
                pt.factr[p as usize][0], 0,
                "prime {p} should have factr[{p}][0] == 0"
            );
        }
    }

    #[test]
    fn factr_composites_decompose_to_their_factors() {
        let pt = PowerTables::new();
        // Spot-check: each is `(z, expected_a, expected_b)` with a*b == z.
        let cases = [
            (4u32, 2u32, 2u32),
            (6, 3, 2),
            (8, 4, 2),
            (9, 3, 3),
            (10, 5, 2),
            (12, 6, 2),
            (15, 3, 5),
            (21, 3, 7),
            (25, 5, 5),
            (27, 3, 9),
            (35, 5, 7),
            (49, 7, 7),
            (121, 11, 11),
            (169, 13, 13),
            // 255 = 3 × 5 × 17; smallest-prime-first → 3 × 85.
            (255, 3, 85),
        ];
        for (z, a, b) in cases {
            assert_eq!(
                pt.factr[z as usize],
                [a, b],
                "factr[{z}] should be [{a}, {b}]"
            );
        }
    }

    #[test]
    fn factr_invariant_holds_for_all_composites() {
        // Voxlap's invariant: if factr[z][0] != 0 then a*b == z.
        let pt = PowerTables::new();
        for z in 2..SETSPHMAXRAD as u32 {
            let a = pt.factr[z as usize][0];
            if a != 0 {
                let b = pt.factr[z as usize][1];
                assert_eq!(a * b, z, "factr[{z}] = [{a}, {b}], product {}", a * b);
            }
        }
    }

    // --- logint ----------------------------------------------------------

    #[test]
    fn logint_matches_natural_log() {
        let pt = PowerTables::new();
        // Compare bit patterns: these are exact-equal floats produced
        // by the same `ln` call, so any drift is a real bug.
        assert_eq!(pt.logint[1].to_bits(), 0.0f64.to_bits());
        assert_eq!(pt.logint[10].to_bits(), (10.0f64).ln().to_bits());
        assert_eq!(pt.logint[100].to_bits(), (100.0f64).ln().to_bits());
        assert_eq!(pt.logint[255].to_bits(), (255.0f64).ln().to_bits());
    }

    // --- tempfloatbuf ---------------------------------------------------

    #[test]
    fn tempfloatbuf_curpow_two_approximates_squares() {
        let pt = PowerTables::new();
        let buf = pt.build_tempfloatbuf(64, 2.0);
        // tempfloatbuf[i] should be very close to i² for curpow=2.
        // Not exactly equal because the prime path goes through
        // exp(ln*2), which carries ULP rounding. Tolerance: 1 ULP
        // of i² in f32, or just relative 1e-6.
        for i in 0..=64u32 {
            let want = (i * i) as f32;
            let got = buf[i as usize];
            let rel = ((got - want) / want.max(1.0)).abs();
            assert!(rel < 1e-5, "tempfloatbuf[{i}] = {got}, want {want}");
        }
    }

    #[test]
    fn tempfloatbuf_zero_and_one_are_exact() {
        let pt = PowerTables::new();
        let buf = pt.build_tempfloatbuf(8, 2.0);
        // Voxlap hard-codes tempfloatbuf[0]=0, tempfloatbuf[1]=1
        // before the prime-walk loop.
        assert_eq!(buf[0].to_bits(), 0.0f32.to_bits());
        assert_eq!(buf[1].to_bits(), 1.0f32.to_bits());
    }

    #[test]
    fn tempfloatbuf_sentinel_is_max_finite_float() {
        let pt = PowerTables::new();
        let buf = pt.build_tempfloatbuf(8, 2.0);
        assert_eq!(buf[9].to_bits(), 0x7f7f_ffff);
    }

    #[test]
    fn tempfloatbuf_clamps_huge_hitrad() {
        let pt = PowerTables::new();
        // hitrad past SETSPHMAXRAD-2 should clamp; the function must
        // not panic and the sentinel should sit at index 255.
        let buf = pt.build_tempfloatbuf(10_000, 2.0);
        assert_eq!(buf[SETSPHMAXRAD - 1].to_bits(), 0x7f7f_ffff);
    }

    #[test]
    fn tempfloatbuf_curpow_three_matches_cubes() {
        let pt = PowerTables::new();
        let buf = pt.build_tempfloatbuf(20, 3.0);
        for i in 0..=20u32 {
            let want = (i as f32).powi(3);
            let got = buf[i as usize];
            let rel = ((got - want) / want.max(1.0)).abs();
            assert!(rel < 1e-4, "tempfloatbuf[{i}] = {got}, want {want}");
        }
    }
}
