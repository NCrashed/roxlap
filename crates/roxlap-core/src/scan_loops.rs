//! Four-quadrant column-scan dispatch — port of `voxlap5.c:opticast`
//! lines 2373..2440+ (the scan loops). Filling out across the
//! `R4.1f*` sub-substages.
//!
//! - **R4.1f1** (this commit): `vline_clip` / `hline_clip` clip
//!   helpers — the line-against-viewport math from voxlap's `vline` /
//!   `hline` (voxlap5.c:1861, 1879). The actual `gline` ray-cast
//!   call those functions also issue is grouscan and lands in R4.3;
//!   here we expose only the integer endpoints, which is what the
//!   four-quadrant scan loops consume to know which screen pixels
//!   the column falls within.
//! - **R4.1f2..f4**: rasterizer trait + per-quadrant drivers (top,
//!   right, bottom, left).

use crate::projection::ProjectionRect;

/// Clip a vertical-direction line `(x0, y0) → (x1, y1)` against the
/// viewport's x-bounds, then clamp the start endpoint to the
/// y-bounds. Returns the integer `(iy0, iy1)` voxlap's `vline`
/// (voxlap5.c:1879) writes via its out-parameters.
///
/// `grd` is voxlap's `grd` global — `1 / (y1 - y0)`, set by the
/// caller before each `vline` invocation.
///
/// Note: only `iy0` is clamped to `[iwy0, iwy1]`; voxlap leaves
/// `iy1` unclamped (with a commented-out clamp in the C source —
/// this asymmetry is intentional).
//
// The y-projection formula `(x_target - x0) / dxy + y0` is anchored
// at the start point (x0, y0); voxlap uses the same anchor for both
// iy0 and iy1, so a line fully outside the viewport on one x-side
// produces iy0 == iy1 and the downstream gline call gets length 0.
#[allow(
    clippy::cast_possible_truncation,
    // wx0/wx1 vs iwy0/iwy1 trip similar_names; voxlap names.
    clippy::similar_names
)]
#[must_use]
pub fn vline_clip(x0: f32, y0: f32, x1: f32, y1: f32, grd: f32, p: &ProjectionRect) -> (i32, i32) {
    let dxy = (x1 - x0) * grd; // dx/dy
    let project_y = |x_target: f32| ((x_target - x0) / dxy + y0).round_ties_even() as i32;
    let identity_y = |y_in: f32| y_in.round_ties_even() as i32;

    let iy0_raw = if x0 < p.wx0 {
        project_y(p.wx0)
    } else if x0 > p.wx1 {
        project_y(p.wx1)
    } else {
        identity_y(y0)
    };
    let iy1 = if x1 < p.wx0 {
        project_y(p.wx0)
    } else if x1 > p.wx1 {
        project_y(p.wx1)
    } else {
        identity_y(y1)
    };

    let iy0 = iy0_raw.clamp(p.iwy0, p.iwy1);
    (iy0, iy1)
}

/// Clip a horizontal-direction line `(x0, y0) → (x1, y1)` against
/// the viewport's y-bounds, then clamp the start endpoint to the
/// x-bounds. Returns the integer `(ix0, ix1)` voxlap's `hline`
/// (voxlap5.c:1861) writes via its out-parameters.
///
/// `grd` here is `1 / (x1 - x0)`, again set by the caller. Same
/// asymmetric clip — `ix0` clamped to `[iwx0, iwx1]`, `ix1` not.
#[allow(clippy::cast_possible_truncation, clippy::similar_names)]
#[must_use]
pub fn hline_clip(x0: f32, y0: f32, x1: f32, y1: f32, grd: f32, p: &ProjectionRect) -> (i32, i32) {
    let dyx = (y1 - y0) * grd; // dy/dx
    let project_x = |y_target: f32| ((y_target - y0) / dyx + x0).round_ties_even() as i32;
    let identity_x = |x_in: f32| x_in.round_ties_even() as i32;

    let ix0_raw = if y0 < p.wy0 {
        project_x(p.wy0)
    } else if y0 > p.wy1 {
        project_x(p.wy1)
    } else {
        identity_x(x0)
    };
    let ix1 = if y1 < p.wy0 {
        project_x(p.wy0)
    } else if y1 > p.wy1 {
        project_x(p.wy1)
    } else {
        identity_x(x1)
    };

    let ix0 = ix0_raw.clamp(p.iwx0, p.iwx1);
    (ix0, ix1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera_math;
    use crate::projection;
    use crate::Camera;

    /// Looking-down camera so we get a viewport rectangle aligned with
    /// the screen — the clip helpers work off `ProjectionRect.iwx*` /
    /// `iwy*` regardless of the camera, but a deterministic camera
    /// keeps the test setup simple.
    fn looking_down_projection() -> ProjectionRect {
        let cam = Camera {
            pos: [0.0, 0.0, 0.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 1.0, 0.0],
            forward: [0.0, 0.0, 1.0],
        };
        let s = camera_math::derive(&cam, 640, 480, 320.0, 240.0, 320.0);
        projection::derive_projection(&s, 640, 480, 320.0, 240.0, 320.0, 1)
    }

    #[test]
    fn vline_clip_line_inside_viewport_uses_y0_y1() {
        // Line from (100, 50) to (100, 400). Both x's inside [-1, 640].
        // y0 = 50 → iy0 = 50; y1 = 400 → iy1 = 400. Clamp doesn't fire
        // because [50, 400] ⊂ [-1, 480].
        let p = looking_down_projection();
        let grd = 1.0 / (400.0 - 50.0); // 1 / dy
        let (iy0, iy1) = vline_clip(100.0, 50.0, 100.0, 400.0, grd, &p);
        assert_eq!((iy0, iy1), (50, 400));
    }

    #[test]
    #[allow(clippy::cast_possible_truncation)]
    fn vline_clip_x0_left_of_viewport_projects() {
        // Line starts left of viewport at x0 = -100, ends inside at
        // x1 = 320. y0 = 100, y1 = 200. dxy = (320 - -100)/(200-100)
        //   = 420 / 100 = 4.2.
        // x0 < wx0 = -1 → iy0_raw = (wx0 - x0)/dxy + y0
        //                  = (-1 - -100)/4.2 + 100
        //                  = 99/4.2 + 100 ≈ 123.57 → round_ties_even → 124.
        // x1 inside → iy1 = round(y1) = 200.
        let p = looking_down_projection();
        let grd = 1.0 / (200.0 - 100.0);
        let (iy0, iy1) = vline_clip(-100.0, 100.0, 320.0, 200.0, grd, &p);
        let want_iy0 = (((-1.0 - -100.0_f32) / 4.2_f32) + 100.0).round_ties_even() as i32;
        assert_eq!(iy0, want_iy0);
        assert_eq!(iy1, 200);
    }

    #[test]
    fn vline_clip_iy0_clamped_to_viewport_y_range() {
        // Line crosses left edge at y far above viewport: (wx0 - x0)/dxy
        // projects to a y < iwy0. Clamp to iwy0.
        // x0 = -10, y0 = -100, x1 = 100, y1 = -50.
        // dxy = (100 - -10)/(-50 - -100) = 110/50 = 2.2.
        // x0 < wx0 = -1 → iy0_raw = (-1 - -10)/2.2 + -100 = 9/2.2 + -100
        //               ≈ 4.09 - 100 = -95.91 → round → -96.
        // -96 < iwy0 = -1 → clamped to -1.
        let p = looking_down_projection();
        let grd = 1.0 / (-50.0 - -100.0);
        let (iy0, _) = vline_clip(-10.0, -100.0, 100.0, -50.0, grd, &p);
        assert_eq!(iy0, p.iwy0);
    }

    #[test]
    fn vline_clip_iy1_not_clamped_even_outside_range() {
        // Line entirely above viewport: x0=10, y0=-1000, x1=20, y1=-500.
        // Both x's inside; iy0 = round(-1000) = -1000, iy1 = round(-500).
        // Voxlap clamps iy0 → iwy0 = -1. iy1 stays at -500 (NOT clamped).
        let p = looking_down_projection();
        let grd = 1.0 / (-500.0 - -1000.0);
        let (iy0, iy1) = vline_clip(10.0, -1000.0, 20.0, -500.0, grd, &p);
        assert_eq!(iy0, p.iwy0); // clamped
        assert_eq!(iy1, -500); // unclamped
    }

    #[test]
    fn vline_clip_line_entirely_outside_left_zero_length() {
        // Both x's < wx0 = -1. Both endpoints project to the same y at
        // x = wx0; iy0 == iy1, gline would receive length 0.
        let p = looking_down_projection();
        let grd = 1.0 / (300.0 - 100.0); // dy = 200
        let (iy0, iy1) = vline_clip(-50.0, 100.0, -10.0, 300.0, grd, &p);
        // The clamp on iy0 may move it; the unclamped iy1 reflects the
        // raw projection. Their post-clamp values should still produce
        // a zero-length scan IF iy0 was inside the y-range. Here both
        // raw values lie in [iwy0, iwy1] so clamp doesn't fire and they
        // stay equal.
        assert_eq!(iy0, iy1);
    }

    #[test]
    fn hline_clip_mirrors_vline_clip_for_horizontal_lines() {
        // Mirror of vline test #1: line from (50, 100) to (400, 100).
        let p = looking_down_projection();
        let grd = 1.0 / (400.0 - 50.0);
        let (ix0, ix1) = hline_clip(50.0, 100.0, 400.0, 100.0, grd, &p);
        assert_eq!((ix0, ix1), (50, 400));
    }

    #[test]
    #[allow(clippy::cast_possible_truncation)]
    fn hline_clip_y0_above_viewport_projects() {
        // Line starts above viewport at y0 = -100, ends inside at
        // y1 = 320. x0 = 100, x1 = 200. dyx = (320 - -100)/(200-100)
        //   = 4.2.
        // y0 < wy0 = -1 → ix0_raw = (wy0 - y0)/dyx + x0
        //                = (-1 - -100)/4.2 + 100
        //                = 99/4.2 + 100 ≈ 123.57 → 124.
        // y1 inside → ix1 = round(x1) = 200.
        let p = looking_down_projection();
        let grd = 1.0 / (200.0 - 100.0);
        let (ix0, ix1) = hline_clip(100.0, -100.0, 200.0, 320.0, grd, &p);
        let want_ix0 = (((-1.0 - -100.0_f32) / 4.2_f32) + 100.0).round_ties_even() as i32;
        assert_eq!(ix0, want_ix0);
        assert_eq!(ix1, 200);
    }

    #[test]
    fn hline_clip_ix0_clamped_to_viewport_x_range() {
        // Line entirely left of viewport: x0=-1000, y0=10, x1=-500, y1=20.
        // y's inside; ix0 = round(-1000) = -1000, clamped to iwx0 = -1.
        let p = looking_down_projection();
        let grd = 1.0 / (-500.0 - -1000.0);
        let (ix0, ix1) = hline_clip(-1000.0, 10.0, -500.0, 20.0, grd, &p);
        assert_eq!(ix0, p.iwx0);
        assert_eq!(ix1, -500);
    }
}
