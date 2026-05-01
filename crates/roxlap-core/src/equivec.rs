//! Voxlap's `univec[256]` / `iunivec[256][4]` Fibonacci-spiral sphere
//! direction tables.
//!
//! Port of `equivecinit(255)` (`voxlap5.c:7866`) plus the iunivec
//! quantisation loop (`voxlap5.c:12302-12308`). Voxlap calls this
//! once during `initvoxlap`; roxlap builds the tables lazily via
//! [`OnceLock`] on first access from `updatereflects`.
//!
//! The direction at index `i` (for `n = 255`) is:
//!
//! ```text
//! z = i * (2/n) + (1/n - 1)         // monotone -1..1
//! r = sqrt(1 - z^2)
//! x = cos(i * GOLDRAT * 2π) * r
//! y = sin(i * GOLDRAT * 2π) * r
//! ```
//!
//! with `GOLDRAT = 0.3819660112501052`. Index 255 is the special
//! zero vector (voxlap's "n is odd, slot 255 is the all-zero
//! vector" hack).
//!
//! `iunivec[i] = ((short)(univec[i].x*4096), ..., 4096)`.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::needless_range_loop,
    clippy::many_single_char_names
)]

use std::sync::OnceLock;

/// Voxlap's `GOLDRAT` constant (`voxlap5.c:18`): `1 - 1/φ`. Stored
/// as `f64` because the original C expression `(float)i * (GOLDRAT *
/// PI * 2)` evaluates the angle in `double` and only casts to
/// `float` when calling `cosf`/`sinf` — preserving that precision
/// path is important for matching voxlap's `iunivec` byte-for-byte.
pub const GOLDRAT: f64 = 0.381_966_011_250_105_2;

/// Number of distinct unit directions voxlap quantises into. Slot
/// 256 (= `n`) is the all-zero vector when `n` is odd.
pub const UNIVEC_N: usize = 255;

/// Total slots, including the n+1 zero slot.
pub const UNIVEC_LEN: usize = 256;

static UNIVEC: OnceLock<[[f32; 3]; UNIVEC_LEN]> = OnceLock::new();
static IUNIVEC: OnceLock<[[i16; 4]; UNIVEC_LEN]> = OnceLock::new();

fn build_univec() -> [[f32; 3]; UNIVEC_LEN] {
    let mut out = [[0.0f32; 3]; UNIVEC_LEN];
    // Voxlap stores zmulk/zaddk as f32 (equivectyp); compute the
    // same way to preserve their rounding.
    let n = UNIVEC_N as f32;
    let zmulk = 2.0_f32 / n;
    let zaddk = zmulk * 0.5_f32 - 1.0_f32;
    // The angle multiplier is computed in f64 to mirror C's
    // promotion: `(float)i * (GOLDRAT * PI * 2)` evaluates as
    // double, then casts to float at the cosf/sinf call.
    let goldrat_2pi: f64 = GOLDRAT * std::f64::consts::PI * 2.0;
    for i in 0..UNIVEC_N {
        let z = (i as f32) * zmulk + zaddk;
        let r = (1.0_f32 - z * z).sqrt();
        let a = ((i as f64) * goldrat_2pi) as f32;
        let x = a.cos() * r;
        let y = a.sin() * r;
        out[i] = [x, y, z];
    }
    // Slot 255 is the all-zero direction (voxlap's odd-n hack).
    out[UNIVEC_N] = [0.0, 0.0, 0.0];
    out
}

fn build_iunivec(univec: &[[f32; 3]; UNIVEC_LEN]) -> [[i16; 4]; UNIVEC_LEN] {
    let mut out = [[0i16; 4]; UNIVEC_LEN];
    for i in 0..UNIVEC_LEN {
        // Voxlap's `(short)(f * 4096.0)` is a C truncating cast. For
        // negative values it truncates toward zero — same as Rust's
        // `as i16` once the value is in range.
        out[i][0] = (univec[i][0] * 4096.0) as i16;
        out[i][1] = (univec[i][1] * 4096.0) as i16;
        out[i][2] = (univec[i][2] * 4096.0) as i16;
        out[i][3] = 4096;
    }
    out
}

/// Lazily-initialised 256-entry sphere-direction table. Slot 255 is
/// the all-zero vector. Mirror of voxlap's `univec[256]`.
#[must_use]
pub fn univec() -> &'static [[f32; 3]; UNIVEC_LEN] {
    UNIVEC.get_or_init(build_univec)
}

/// Integer-quantised version of [`univec`]. Each component is
/// `(univec[i].k * 4096.0) as i16`. Lane 3 is the constant `4096`
/// — voxlap uses it as the "1.0 in the `dot4` product" slot for the
/// `lightlist[k][3]` bias.
#[must_use]
pub fn iunivec() -> &'static [[i16; 4]; UNIVEC_LEN] {
    IUNIVEC.get_or_init(|| build_iunivec(univec()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Voxlap's `equiind2vec` distributes 255 unit vectors over the
    /// sphere via a Fibonacci spiral. The first few should be near
    /// the south pole (z ≈ -1) and the last few near the north pole.
    #[test]
    fn univec_endpoints_are_polar() {
        let u = univec();
        // i=0: z = 0 * 2/255 + (1/255 - 1) = 1/255 - 1 ≈ -0.9961.
        assert!(u[0][2] < -0.99, "got z = {}", u[0][2]);
        // i=254: z = 254 * 2/255 + (1/255 - 1) = 508/255 + 1/255 - 1
        //          = 509/255 - 1 ≈ 0.9961.
        assert!(u[254][2] > 0.99, "got z = {}", u[254][2]);
    }

    #[test]
    fn univec_unit_length_for_active_slots() {
        let u = univec();
        for (i, v) in u.iter().enumerate().take(UNIVEC_N) {
            let len2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
            // Each direction is built from sqrt(1-z²) sin/cos and z;
            // round-trip should give length ≈ 1 within float error.
            assert!((len2 - 1.0).abs() < 1e-5, "slot {i} len² = {len2}");
        }
        // Slot 255 is the special zero vector.
        let v = u[UNIVEC_N];
        assert_eq!(v[0].to_bits(), 0);
        assert_eq!(v[1].to_bits(), 0);
        assert_eq!(v[2].to_bits(), 0);
    }

    #[test]
    fn iunivec_lane3_is_4096() {
        let iu = iunivec();
        for (i, v) in iu.iter().enumerate() {
            assert_eq!(v[3], 4096, "iunivec[{i}][3] should be 4096");
        }
    }

    #[test]
    fn iunivec_matches_quantised_univec() {
        let u = univec();
        let iu = iunivec();
        for i in 0..UNIVEC_LEN {
            assert_eq!(iu[i][0], (u[i][0] * 4096.0) as i16);
            assert_eq!(iu[i][1], (u[i][1] * 4096.0) as i16);
            assert_eq!(iu[i][2], (u[i][2] * 4096.0) as i16);
        }
    }
}
