//! Per-pixel ray-step coefficients — the `optistr*` / `optihei*` /
//! `optiadd*` values voxlap derives just before the column-scan
//! loops, plus the `cx*65536` / `cy*65536` Q16.16 forms of the
//! projection centre.
//!
//! Port of `voxlap5.c:opticast` lines 2359-2371.
//!
//! For each pixel (sx, sy) in screen space, the camera-relative ray
//! direction in the (x, y) ground plane is approximated as
//!
//! ```text
//! dirx = optistrx * sx + optiheix * sy + optiaddx
//! diry = optistry * sx + optiheiy * sy + optiaddy
//! ```
//!
//! The Q16.16 `cx16` / `cy16` fields are used by the four scan loops
//! when they iterate sx / sy in fixed-point shldiv16-style arithmetic
//! (see voxlap5.c lines 2395-2440).
//!
//! The `opti4` SSE batch table voxlap precomputes for the
//! 4-pixel-unrolled hrend/vrend rasterizers is intentionally NOT in
//! this module — it belongs to the SSE-recover work in R5, mirroring
//! voxlaptest's stage 4.9 which scoped it the same way.

use crate::camera_math::CameraState;
use crate::fixed::ftol;
use crate::opticast_prelude::PREC;

/// Coefficients the four-quadrant scan loops use to step a ray
/// direction by one screen pixel.
#[derive(Debug, Clone, Copy)]
pub struct RayStep {
    pub strx: f32,
    pub stry: f32,
    pub heix: f32,
    pub heiy: f32,
    pub addx: f32,
    pub addy: f32,
    /// Projection centre × 65536, rounded to int via `ftol`
    /// (round-ties-to-even). Used directly in the shldiv16-based
    /// per-pixel `u` / `ui` arithmetic in opticast's scan loops.
    pub cx16: i32,
    pub cy16: i32,
}

/// Derive the per-pixel ray-step coefficients.
///
/// Inputs:
/// - `camera_state` — supplies `right.x/y` (`gistr`), `down.x/y`
///   (`gihei`), and `corn[0].x/y` (the top-left view-frustum corner).
/// - `cx` / `cy` — projection centre from
///   `projection::derive_projection`.
/// - `hz` — voxlap's `gihz` screen-projection param (the same value
///   passed to `setcamera`'s `dahz` argument).
//
// clippy::cast_precision_loss covers `i32 as f32` for the PREC scale
// constant. The other `as i32` is intentional truncation post-round.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    // strx/stry, heix/heiy, addx/addy, cx16/cy16 are voxlap names.
    clippy::similar_names
)]
#[must_use]
pub fn derive_ray_step(camera_state: &CameraState, cx: f32, cy: f32, hz: f32) -> RayStep {
    let f = (PREC as f32) / hz;
    let strx = camera_state.right[0] * f;
    let stry = camera_state.right[1] * f;
    let heix = camera_state.down[0] * f;
    let heiy = camera_state.down[1] * f;
    let addx = camera_state.corn[0][0] * f;
    let addy = camera_state.corn[0][1] * f;
    // cx / cy can sit just past F_CLAMP = 32000 + screen-half
    // (≈33k) for level-camera projections, so cx*65536 overflows
    // i32::MAX. Voxlap C's ftol wraps; route through the helper.
    let cx16 = ftol(cx * 65536.0);
    let cy16 = ftol(cy * 65536.0);
    RayStep {
        strx,
        stry,
        heix,
        heiy,
        addx,
        addy,
        cx16,
        cy16,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera_math;
    use crate::Camera;

    fn bit(v: f32) -> u32 {
        v.to_bits()
    }

    /// Camera looking straight down at the origin, 640×480 viewport,
    /// hx/hy/hz = (320, 240, 320). Picked because the resulting cx/cy
    /// land at viewport centre `(320, 240)` — both exactly
    /// representable in f32 and fitting comfortably below `2^24`
    /// after the ×65536 scale, so `cx16` / `cy16` are exact.
    fn looking_down_state() -> CameraState {
        let cam = Camera {
            pos: [0.0, 0.0, 0.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 1.0, 0.0],
            forward: [0.0, 0.0, 1.0],
        };
        camera_math::derive(&cam, 640, 480, 320.0, 240.0, 320.0)
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn ray_step_for_looking_down_camera() {
        // hz = 320 → f = 1_048_576 / 320 = 3276.8.
        let s = looking_down_state();
        let r = derive_ray_step(&s, 320.0, 240.0, 320.0);
        let f = (PREC as f32) / 320.0;

        // gistr = (1, 0, 0) → strx = f, stry = 0.
        assert_eq!(bit(r.strx), bit(f));
        assert_eq!(bit(r.stry), bit(0.0));

        // gihei = (0, 1, 0) → heix = 0, heiy = f.
        assert_eq!(bit(r.heix), bit(0.0));
        assert_eq!(bit(r.heiy), bit(f));

        // For looking-down camera with hx=hy=hz/2, gcorn[0] should be
        // computed by camera_math: -hx*right - hy*down + hz*forward
        // = -320*[1,0,0] - 240*[0,1,0] + 320*[0,0,1]
        // = [-320, -240, 320]. addx = -320*f, addy = -240*f.
        assert_eq!(bit(r.addx), bit(-320.0 * f));
        assert_eq!(bit(r.addy), bit(-240.0 * f));
    }

    #[test]
    fn fixed_point_centre_for_integer_inputs() {
        let s = looking_down_state();
        let r = derive_ray_step(&s, 320.0, 240.0, 320.0);
        // 320 * 65536 = 20_971_520 — exactly representable in f32.
        assert_eq!(r.cx16, 20_971_520);
        // 240 * 65536 = 15_728_640.
        assert_eq!(r.cy16, 15_728_640);
    }

    #[test]
    fn fixed_point_centre_round_ties_to_even_at_half() {
        // cx = 0.5 / 65536 = exactly half a fixed-point unit. With
        // round-ties-to-even, 0.5 → 0 (the even integer of the two).
        let s = looking_down_state();
        let r = derive_ray_step(&s, 0.5_f32 / 65536.0_f32, 1.5_f32 / 65536.0_f32, 320.0);
        // Both inputs round-half-to-even on the multiply-then-round
        // result. 0.5 → 0; 1.5 → 2.
        assert_eq!(r.cx16, 0);
        assert_eq!(r.cy16, 2);
    }

    #[test]
    fn fixed_point_centre_for_negative_half() {
        // -0.5 → 0 (round-half-to-even).
        let s = looking_down_state();
        let r = derive_ray_step(&s, -0.5_f32 / 65536.0_f32, -1.5_f32 / 65536.0_f32, 320.0);
        assert_eq!(r.cx16, 0);
        assert_eq!(r.cy16, -2);
    }

    #[test]
    fn z_components_not_present_in_ray_step() {
        // The ray step is a 2D (x, y) world-plane vector; voxlap does
        // not derive optistrz / optiheiz / optiaddz. RayStep only
        // exposes 2D coefficients. This test is mostly a compile-time
        // assertion via field access — the struct definition is the
        // contract.
        let s = looking_down_state();
        let r = derive_ray_step(&s, 320.0, 240.0, 320.0);
        let _ = (r.strx, r.stry, r.heix, r.heiy, r.addx, r.addy);
    }
}
