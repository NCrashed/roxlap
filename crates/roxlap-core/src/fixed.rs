//! Fixed-point integer helper(s).
//
// `ftol`'s cast is fundamentally a bit-twiddle; the lints flag
// behaviour that *is* the operation.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

/// Round-to-nearest (ties to even) float → `i32` with **wrapping**
/// overflow, matching C's `lrintf` + `(int32_t)` cast.
///
/// Rust's plain `as i32` *saturates* at `i32::MAX`, which silently
/// diverged from the C reference for any float wider than `i32::MAX`
/// — visible historically as a floor-hairline artifact when a
/// near-axis-aligned ray's slope lane saturated (see
/// `project_roxlap_floor_hairline.md`). Going via `i64` first
/// reproduces the wrap for floats in `[i64::MIN, i64::MAX]`.
#[must_use]
#[inline]
pub fn ftol(f: f32) -> i32 {
    f.round_ties_even() as i64 as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ftol_rounds_ties_to_even_in_range() {
        // round-half-to-even: 0.5 → 0, 1.5 → 2, 2.5 → 2.
        assert_eq!(ftol(0.5), 0);
        assert_eq!(ftol(1.5), 2);
        assert_eq!(ftol(2.5), 2);
        assert_eq!(ftol(-0.5), 0);
        assert_eq!(ftol(-1.5), -2);
        // Plain rounding for non-tie inputs.
        assert_eq!(ftol(1.4), 1);
        assert_eq!(ftol(1.6), 2);
    }

    #[test]
    fn ftol_wraps_modulo_2_to_32_on_overflow() {
        // The defining property: floats whose rounded value exceeds
        // i32::MAX must WRAP modulo 2^32 (matching C's `lrintf +
        // (int32_t)cast`), not saturate at i32::MAX.
        assert_eq!(ftol(2_147_483_648.0_f32), i32::MIN);
        assert_eq!(ftol(4_294_967_296.0_f32), 0);

        let f = 3.9e10_f32;
        let want = f.round_ties_even() as i64 as i32;
        assert_eq!(ftol(f), want);
        assert_ne!(ftol(f), i32::MAX);
    }
}
