//! Per-frame projection setup — port of `voxlap5.c:opticast` lines
//! 2327..2357.
//!
//! Computes the screen-space data the four-quadrant scan loops need:
//!
//! - **`cx` / `cy`** — projection centre; where the camera's forward
//!   ray hits the screen plane (or its asymptote when forward is
//!   parallel to the screen plane). Each scan fan radiates outward
//!   from this point.
//! - **`wx0..wx1`, `wy0..wy1`** — viewport bounds with `anginc`
//!   padding. `iwx*` / `iwy*` are the same values rounded to int via
//!   voxlap's `ftol` (round-ties-to-even).
//! - **`fx, fy, gx, gy`** — relative offsets from `(cx, cy)` to the
//!   four viewport corners (top-left, bottom-right minus centre).
//!   Their signs select which quadrant cuts engage.
//! - **`(x0, y0)..(x3, y3)`** — corner-cut quadrilateral the four
//!   scan loops walk around. When `(cx, cy)` lies outside the
//!   viewport, the cuts collapse the bounding rectangle into a
//!   smaller fan-shaped region. Final `±0.01` bias matches voxlap's
//!   "let pixels overwrite rather than omit" rounding-error guard.
//!
//! Fan layout for the four scan loops that follow (R4.1f and on):
//! - top quadrant: `(cx, cy) → (x0, wy0) → (x1, wy0)`
//! - right quadrant: `(cx, cy) → (wx1, y1) → (wx1, y2)`
//! - bottom quadrant: `(cx, cy) → (x2, wy1) → (x3, wy1)`
//! - left quadrant: `(cx, cy) → (wx0, y3) → (wx0, y0)`

use crate::camera_math::CameraState;

/// Voxlap clamps the forward-z reciprocal to `±32000` to avoid
/// `inf` blowing up the projection when the camera is nearly
/// parallel to the screen plane.
const F_CLAMP: f32 = 32000.0;

#[derive(Debug, Clone, Copy)]
pub struct ProjectionRect {
    /// Projection-centre x — `gistr.z * f + gihx`.
    pub cx: f32,
    /// Projection-centre y — `gihei.z * f + gihy`.
    pub cy: f32,

    /// Viewport bounds with `anginc` padding (`-anginc`, `xres-1+anginc`, …).
    pub wx0: f32,
    pub wx1: f32,
    pub wy0: f32,
    pub wy1: f32,
    /// `wx*` / `wy*` rounded to int via `ftol` (round-ties-to-even).
    pub iwx0: i32,
    pub iwx1: i32,
    pub iwy0: i32,
    pub iwy1: i32,

    /// Relative offsets from `(cx, cy)` to the four viewport corners.
    pub fx: f32,
    pub fy: f32,
    pub gx: f32,
    pub gy: f32,

    /// Corner-cut quadrilateral. `(x0, y0) (x1, y1) (x2, y2) (x3, y3)`
    /// in CCW order matching voxlap's scan loops.
    pub x0: f32,
    pub x1: f32,
    pub x2: f32,
    pub x3: f32,
    pub y0: f32,
    pub y1: f32,
    pub y2: f32,
    pub y3: f32,
}

/// Derive the per-frame projection rectangle.
///
/// Inputs `xres`, `yres`, `hx`, `hy`, `hz`, `anginc` mirror voxlap's
/// `xres` / `yres` globals plus the `(dahx, dahy, dahz)` parameters
/// of `setcamera` and the `vx5.anginc` config knob. Oracle defaults
/// are `(width / 2, height / 2, width / 2)` and `anginc = 1`.
///
/// Equivalent to [`derive_projection_with_y_range`] with
/// `y_start = 0, y_end = yres` (full-frame opticast — the pre-R12.3
/// behaviour preserved bit-exactly). Kept for the in-tree projection
/// tests that exercise the full-frame math; opticast itself calls
/// the `with_y_range` variant directly.
#[allow(dead_code)]
#[must_use]
pub fn derive_projection(
    camera_state: &CameraState,
    xres: u32,
    yres: u32,
    hx: f32,
    hy: f32,
    hz: f32,
    anginc: f32,
) -> ProjectionRect {
    derive_projection_with_y_range(camera_state, xres, yres, 0, yres, hx, hy, hz, anginc)
}

/// Derive the projection rectangle clipped to a horizontal strip
/// `[y_start, y_end)`. R12.3.0 entry point — the per-strip parallel
/// dispatch (R12.3.1) calls this once per strip with the strip's
/// y-range.
///
/// The camera projection center `(cx, cy)` is computed in absolute
/// screen coords, exactly as for the full-frame call — `cy` may
/// even fall outside the strip (e.g. a top strip with the camera
/// horizon above it). Only the viewport edges (`wy0` / `wy1` /
/// `iwy0` / `iwy1`) and the corner-cut quadrilateral are clipped.
/// `vline_clip` / `hline_clip` consume the clipped wy0/wy1, so
/// pass-1 ray casts in `scan_loops` automatically restrict to the
/// strip.
//
// clippy::float_cmp at `forward_z == 0.0` is intentional: voxlap's C
// is `if (gifor.z == 0)` — exact zero check before dividing by it.
// Other allows cover the f32 / i32 / u32 cross-casting needed to
// match the per-frame f32 math voxlap uses internally.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]
#[must_use]
pub fn derive_projection_with_y_range(
    camera_state: &CameraState,
    xres: u32,
    _yres: u32,
    y_start: u32,
    y_end: u32,
    hx: f32,
    hy: f32,
    hz: f32,
    anginc: f32,
) -> ProjectionRect {
    // Project camera-forward onto the screen plane → centre point.
    let forward_z = camera_state.forward[2];
    let f = if forward_z == 0.0 {
        F_CLAMP
    } else {
        (hz / forward_z).clamp(-F_CLAMP, F_CLAMP)
    };
    let cx = camera_state.right[2] * f + hx;
    let cy = camera_state.down[2] * f + hy;

    // anginc-padded viewport, with the strip's y-range substituted
    // for the full-frame `0..yres`. wx0 / wx1 are unaffected — strips
    // are full-x.
    let anginc_f = anginc;
    let wx0 = -anginc_f;
    let wx1 = (xres as i32 - 1) as f32 + anginc_f;
    let wy0 = (y_start as i32) as f32 - anginc_f;
    let wy1 = (y_end as i32 - 1) as f32 + anginc_f;
    let iwx0 = wx0.round_ties_even() as i32;
    let iwx1 = wx1.round_ties_even() as i32;
    let iwy0 = wy0.round_ties_even() as i32;
    let iwy1 = wy1.round_ties_even() as i32;

    // Offsets to viewport corners.
    let fx = wx0 - cx;
    let fy = wy0 - cy;
    let gx = wx1 - cx;
    let gy = wy1 - cy;

    // Corner-cut quadrilateral; initial = full viewport.
    let mut x0 = wx0;
    let mut x3 = wx0;
    let mut x1 = wx1;
    let mut x2 = wx1;
    let mut y0 = wy0;
    let mut y1 = wy0;
    let mut y2 = wy1;
    let mut y3 = wy1;

    // First pass — circular-arc-style cuts when (cx, cy) is on the
    // outside of a viewport edge. Each surviving corner moves along
    // a 45° fan from the centre.
    if fy < 0.0 {
        if fx < 0.0 {
            let s = (fx * fy).sqrt();
            x0 = cx - s;
            y0 = cy - s;
        }
        if gx > 0.0 {
            let s = (-gx * fy).sqrt();
            x1 = cx + s;
            y1 = cy - s;
        }
    }
    if gy > 0.0 {
        if gx > 0.0 {
            let s = (gx * gy).sqrt();
            x2 = cx + s;
            y2 = cy + s;
        }
        if fx < 0.0 {
            let s = (-fx * gy).sqrt();
            x3 = cx - s;
            y3 = cy + s;
        }
    }

    // Second pass — clamp pairs that crossed each other in pass 1
    // back to a viewport edge using a linear-interpolation projection.
    if x0 > x1 {
        if fx < 0.0 {
            y0 = fx / gx * fy + cy;
        } else {
            y1 = gx / fx * fy + cy;
        }
    }
    if y1 > y2 {
        if fy < 0.0 {
            x1 = fy / gy * gx + cx;
        } else {
            x2 = gy / fy * gx + cx;
        }
    }
    if x2 < x3 {
        if fx < 0.0 {
            y3 = fx / gx * gy + cy;
        } else {
            y2 = gx / fx * gy + cy;
        }
    }
    if y3 < y0 {
        if fy < 0.0 {
            x0 = fy / gy * fx + cx;
        } else {
            x3 = gy / fy * fx + cx;
        }
    }

    // 0.01 outward bias on each edge so boundary pixels overwrite
    // rather than fall through fp-precision cracks (voxlap's
    // verbatim comment: "This makes precision errors cause pixels to
    // overwrite rather than omit").
    x0 -= 0.01;
    x1 += 0.01;
    y1 -= 0.01;
    y2 += 0.01;
    x3 -= 0.01;
    x2 += 0.01;
    y0 -= 0.01;
    y3 += 0.01;

    ProjectionRect {
        cx,
        cy,
        wx0,
        wx1,
        wy0,
        wy1,
        iwx0,
        iwx1,
        iwy0,
        iwy1,
        fx,
        fy,
        gx,
        gy,
        x0,
        x1,
        x2,
        x3,
        y0,
        y1,
        y2,
        y3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera_math;
    use crate::Camera;

    fn level_north_camera_state() -> CameraState {
        let cam = Camera {
            pos: [1024.0, 1024.0, 128.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 0.0, 1.0],
            forward: [0.0, 1.0, 0.0],
        };
        camera_math::derive(&cam, 640, 480, 320.0, 240.0, 320.0)
    }

    fn looking_down_camera_state() -> CameraState {
        // Camera looking straight down: forward = +z, right = +x, down
        // = +y. gistr.z = 0, gihei.z = 0, gifor.z = 1.
        let cam = Camera {
            pos: [1024.0, 1024.0, 128.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 1.0, 0.0],
            forward: [0.0, 0.0, 1.0],
        };
        camera_math::derive(&cam, 640, 480, 320.0, 240.0, 320.0)
    }

    fn bit(v: f32) -> u32 {
        v.to_bits()
    }

    #[test]
    fn level_camera_takes_forward_z_zero_branch() {
        // gifor.z == 0 → f = 32000 unconditionally, regardless of clamp.
        // gistr.z == 0 → cx = hx. gihei.z == 1 → cy = 32000 + hy.
        let s = level_north_camera_state();
        let p = derive_projection(&s, 640, 480, 320.0, 240.0, 320.0, 1.0);
        assert_eq!(bit(p.cx), bit(320.0));
        assert_eq!(bit(p.cy), bit(32000.0 + 240.0));
    }

    #[test]
    fn looking_down_camera_centre_at_viewport_centre() {
        // gifor.z == 1 → f = 320 (= hz/1). gistr.z == 0 → cx = 320.
        // gihei.z == 0 → cy = 240. Centre point at viewport centre.
        let s = looking_down_camera_state();
        let p = derive_projection(&s, 640, 480, 320.0, 240.0, 320.0, 1.0);
        assert_eq!(bit(p.cx), bit(320.0));
        assert_eq!(bit(p.cy), bit(240.0));
    }

    #[test]
    fn viewport_bounds_with_anginc() {
        // anginc = 1 padding: wx0 = -1, wx1 = 640, wy0 = -1, wy1 = 480.
        let s = looking_down_camera_state();
        let p = derive_projection(&s, 640, 480, 320.0, 240.0, 320.0, 1.0);
        assert_eq!(bit(p.wx0), bit(-1.0));
        assert_eq!(bit(p.wx1), bit(640.0));
        assert_eq!(bit(p.wy0), bit(-1.0));
        assert_eq!(bit(p.wy1), bit(480.0));
        assert_eq!(p.iwx0, -1);
        assert_eq!(p.iwx1, 640);
        assert_eq!(p.iwy0, -1);
        assert_eq!(p.iwy1, 480);
        // anginc = 4 widens by 3 each side.
        let q = derive_projection(&s, 640, 480, 320.0, 240.0, 320.0, 4.0);
        assert_eq!(bit(q.wx0), bit(-4.0));
        assert_eq!(bit(q.wx1), bit(643.0));
        assert_eq!(q.iwx0, -4);
        assert_eq!(q.iwx1, 643);
    }

    #[test]
    fn relative_corner_offsets_around_centre() {
        // looking-down camera puts (cx, cy) = (320, 240) inside the
        // viewport; fx / fy negative (top-left below centre), gx / gy
        // positive (bottom-right above centre).
        let s = looking_down_camera_state();
        let p = derive_projection(&s, 640, 480, 320.0, 240.0, 320.0, 1.0);
        // wx0 = -1, cx = 320 → fx = -321.
        assert_eq!(bit(p.fx), bit(-1.0 - 320.0));
        // wy0 = -1, cy = 240 → fy = -241.
        assert_eq!(bit(p.fy), bit(-1.0 - 240.0));
        // wx1 = 640, cx = 320 → gx = 320.
        assert_eq!(bit(p.gx), bit(640.0 - 320.0));
        // wy1 = 480, cy = 240 → gy = 240.
        assert_eq!(bit(p.gy), bit(480.0 - 240.0));
    }

    #[test]
    #[allow(clippy::similar_names)]
    fn corner_cut_quadrilateral_for_centre_inside_viewport() {
        // Looking-down camera: all four corner-cuts engage, each
        // moves a corner along a 45° fan from (cx, cy).
        let s = looking_down_camera_state();
        let p = derive_projection(&s, 640, 480, 320.0, 240.0, 320.0, 1.0);
        // Corner 0 (top-left): x0 = cx - sqrt(fx*fy) - 0.01,
        // y0 = cy - sqrt(fx*fy) - 0.01 (final bias applied).
        let s_topleft = ((-321.0_f32) * (-241.0_f32)).sqrt();
        assert_eq!(bit(p.x0), bit(320.0 - s_topleft - 0.01));
        assert_eq!(bit(p.y0), bit(240.0 - s_topleft - 0.01));
        // Corner 2 (bottom-right) symmetrical with +bias.
        let s_botright = (320.0_f32 * 240.0_f32).sqrt();
        assert_eq!(bit(p.x2), bit(320.0 + s_botright + 0.01));
        assert_eq!(bit(p.y2), bit(240.0 + s_botright + 0.01));
    }

    #[test]
    fn forward_z_clamps_at_positive_extreme() {
        // Tiny but non-zero forward.z should still produce f = 32000
        // because hz/forward_z would exceed F_CLAMP.
        let cam = Camera {
            pos: [0.0, 0.0, 0.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 0.0, 1.0],
            // forward.z very small positive ⇒ hz/forward_z huge ⇒ clamp.
            forward: [0.0, 0.99999, 0.001],
        };
        let s = camera_math::derive(&cam, 640, 480, 320.0, 240.0, 320.0);
        let p = derive_projection(&s, 640, 480, 320.0, 240.0, 320.0, 1.0);
        // gistr.z = 0 → cx = hx. gihei.z = 1 → cy = 32000 + hy.
        // (Even though forward.z is non-zero now, the clamp keeps f at 32000.)
        assert_eq!(bit(p.cx), bit(320.0));
        assert_eq!(bit(p.cy), bit(32000.0 + 240.0));
    }
}
