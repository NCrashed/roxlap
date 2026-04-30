//! Fixed-point integer helpers — port of voxlap's portable C shims
//! that wrap the bit-tricks the original asm did inline.
//
// Module-wide allows: every function here is fundamentally a cast-
// laden bit-twiddle that voxlap's C does inline. The lints flag
// behaviour that *is* the operation.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
//!
//! References:
//! - `mulshr16` (`voxlap5.c:343`): `(a * d) >> 16` via `i64`.
//! - `shldiv16` (`voxlap5.c:353`): `(a << 16) / b` via `i64`.
//! - `isshldiv16safe` (`voxlap5.c:358`): is `(a << 16) / b`
//!   non-overflowing? The branchless sign-dance below mirrors the
//!   original asm exactly, including correct behaviour for
//!   `i32::MIN` (where naive `-x` would overflow).
//! - `lbound0` (`voxlap5.c:399`): clamp `a` to `[0, b]` (`b ≥ 0`).
//!
//! These show up in voxlap's scan-loop arithmetic (per-row `u`, `ui`
//! delta, screen-space column ranges); the four quadrant drivers
//! that R4.1f3+ lays down call them on every iteration.

/// Voxlap C's `ftol(f, &out)` — `lrintf` rounds to the nearest
/// integer (ties to even), then the result is cast to `int32_t`.
/// On Linux `x86_64` `lrintf` returns `long` (64-bit), so the
/// `(int32_t)` cast **wraps modulo 2³²** for floats whose rounded
/// value exceeds `i32::MAX`. Rust's `as i32` instead **saturates**,
/// which silently diverges from voxlap C for any float wider than
/// `i32::MAX` — visible as the floor-hairline artifact when
/// `gline`'s `gdz` lane saturated for near-axis-aligned rays
/// (see `project_roxlap_floor_hairline.md`).
///
/// Going via `i64` first matches voxlap's wrap behaviour exactly
/// for floats in `[i64::MIN, i64::MAX]`. Floats outside that
/// range still saturate at the i64 step — but voxlap's `lrintf`
/// raises `FE_INVALID` and returns an implementation-defined
/// value at that magnitude too, so callers always need an
/// upstream sanity-check at boundaries that extreme.
///
/// Use this anywhere a roxlap port mirrors a voxlap5.c
/// `ftol(float, &int32)` call.
#[must_use]
#[inline]
pub fn ftol(f: f32) -> i32 {
    f.round_ties_even() as i64 as i32
}

/// `(a * d) >> 16`, computed in `i64` to avoid intermediate overflow.
#[must_use]
pub fn mulshr16(a: i32, d: i32) -> i32 {
    let prod = i64::from(a) * i64::from(d);
    (prod >> 16) as i32
}

/// `(a << 16) / b`, computed in `i64`.
///
/// # Panics
///
/// Panics on `b == 0` (division by zero) and on the i64 overflow
/// case `i64::MIN / -1`. Voxlap's C wraps both as undefined behaviour;
/// callers in opticast guard with [`isshldiv16safe`] before calling
/// this, so the panics shouldn't fire on well-formed inputs.
#[must_use]
pub fn shldiv16(a: i32, b: i32) -> i32 {
    let num = i64::from(a) << 16;
    (num / i64::from(b)) as i32
}

/// Returns `1` if `(a << 16) / b` would not overflow `i32`, else `0`.
/// Branchless, deliberately uses subtraction-of-negatives to stay
/// correct at `i32::MIN` (where `-i32::MIN` would overflow).
///
/// Mirrors the original asm in `voxlap5.c:358` instruction-for-
/// instruction; the unsigned-right-shift-by-31 at the end is the
/// branchless sign-bit extract that voxlap's asm produces via `shr`.
#[must_use]
pub fn isshldiv16safe(a: i32, b: i32) -> i32 {
    let mut edx = a;
    if edx >= 0 {
        edx = edx.wrapping_neg();
    }
    edx >>= 14;
    let mut eax = b;
    if eax >= 0 {
        eax = eax.wrapping_neg();
    }
    eax = eax.wrapping_sub(edx);
    ((eax as u32) >> 31) as i32
}

/// Clamp `a` to `[0, b]`. `b` must be `>= 0`.
///
/// The implementation matches voxlap's branchless variant
/// (`voxlap5.c:399`): the `(a as u32) <= b` test catches both `a < 0`
/// (where the unsigned cast wraps to a huge number) and `a > b` in
/// one branch. The fall-through expression `(!(a >> 31)) & b` returns
/// `b` when `a > b` (sign bit clear → `~(...)` is all-ones → `& b`)
/// and `0` when `a < 0` (sign bit set → `~(...)` is zero).
#[must_use]
pub fn lbound0(a: i32, b: i32) -> i32 {
    debug_assert!(b >= 0, "lbound0: b must be >= 0");
    if (a as u32) <= (b as u32) {
        return a;
    }
    (!(a >> 31)) & b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mulshr16_smoke() {
        // 256 * 256 = 65536; >> 16 = 1.
        assert_eq!(mulshr16(256, 256), 1);
        // 1.0 fixed = 65536; * 1.0 fixed = 65536^2; >> 16 = 65536.
        assert_eq!(mulshr16(65536, 65536), 65536);
        // 0.5 * 0.5 = 0.25 → (32768 * 32768) >> 16 = 16384.
        assert_eq!(mulshr16(32768, 32768), 16384);
        // No intermediate overflow: i32::MAX * 2 then shr 16.
        assert_eq!(
            mulshr16(i32::MAX, 2),
            ((i64::from(i32::MAX) * 2) >> 16) as i32
        );
        // Negative inputs: arithmetic shift preserves sign.
        assert_eq!(mulshr16(-65536, 256), -256);
    }

    #[test]
    fn shldiv16_smoke() {
        // 1 << 16 / 1 = 65536.
        assert_eq!(shldiv16(1, 1), 65536);
        // 1 << 16 / 65536 = 1.
        assert_eq!(shldiv16(1, 65536), 1);
        // (-100 << 16) / 200 = -32768.
        assert_eq!(shldiv16(-100, 200), -32768);
    }

    #[test]
    fn isshldiv16safe_returns_one_when_safe() {
        // Small numerator, large denominator: clearly safe.
        assert_eq!(isshldiv16safe(1, 100), 1);
        assert_eq!(isshldiv16safe(-100, -100), 1);
        // 0 numerator: trivially safe (result is 0).
        assert_eq!(isshldiv16safe(0, 1), 1);
    }

    #[test]
    fn isshldiv16safe_returns_zero_when_unsafe() {
        // a >> 14 has too many bits → result of (a << 16) / b would
        // overflow i32. Pick a near i32::MAX with small b.
        assert_eq!(isshldiv16safe(i32::MAX, 1), 0);
        assert_eq!(isshldiv16safe(i32::MAX, 2), 0);
    }

    #[test]
    fn isshldiv16safe_handles_int_min() {
        // i32::MIN: the predicate must not panic on -i32::MIN
        // (would be u32 overflow). The wrapping_neg in the impl
        // protects against that.
        let _ = isshldiv16safe(i32::MIN, 1);
        let _ = isshldiv16safe(1, i32::MIN);
    }

    #[test]
    fn lbound0_in_range_passes_through() {
        assert_eq!(lbound0(0, 100), 0);
        assert_eq!(lbound0(50, 100), 50);
        assert_eq!(lbound0(100, 100), 100);
    }

    #[test]
    fn lbound0_below_zero_clamps_to_zero() {
        assert_eq!(lbound0(-1, 100), 0);
        assert_eq!(lbound0(-1_000_000, 100), 0);
        assert_eq!(lbound0(i32::MIN, 100), 0);
    }

    #[test]
    fn lbound0_above_b_clamps_to_b() {
        assert_eq!(lbound0(101, 100), 100);
        assert_eq!(lbound0(i32::MAX, 100), 100);
    }

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
        // i32::MAX must WRAP modulo 2^32 (matching voxlap C's
        // `lrintf + (int32_t)cast`), not saturate at i32::MAX. This
        // is the regression test for the floor-hairline bug.

        // Exactly-representable boundary cases: 2^31 wraps to
        // i32::MIN; 2^32 wraps to 0.
        assert_eq!(ftol(2_147_483_648.0_f32), i32::MIN);
        assert_eq!(ftol(4_294_967_296.0_f32), 0);

        // For arbitrary huge finite f32, the result must equal
        // `(rounded as i64 as i32)` — i.e., the low 32 bits, not
        // i32::MAX. f32 can't represent 3.9e10 exactly (rounds to a
        // multiple of ~4096), but whatever it rounds to, the wrap
        // identity holds.
        let f = 3.9e10_f32;
        let want = f.round_ties_even() as i64 as i32;
        assert_eq!(ftol(f), want);
        // Roxlap's pre-fix `as i32` saturated to i32::MAX here.
        // ftol must not.
        assert_ne!(ftol(f), i32::MAX);
    }
}
