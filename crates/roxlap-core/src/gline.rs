//! `gline` per-scanline frustum setup — port of the projection math
//! at the top of `voxlap5.c:gline` (lines 1146..1175).
//!
//! For each screen-space scanline `(x0, y0) → (x1, y1)`, this
//! computes the 2D voxel-grid frustum the per-ray voxel-column walker
//! (`grouscanasm` / R4.3c+) consumes:
//!
//! - **World-space ray endpoints** (`vd0`, `vd1`, `vz0`, `vx1`,
//!   `vy1`, `vz1`): the screen scanline projected through the camera
//!   basis to world coordinates.
//! - **`gixy[2]`**: voxel-column step (±1 along x, ±vsid along y) in
//!   the direction the ray travels. voxlap stores byte-strides into
//!   `sptr[VSID*VSID]`; our port uses logical column-index units.
//! - **`gpz[2]`**: distance to the next voxel-grid line along x and
//!   y from the current ray position, scaled by `PREC`. The lane
//!   with the smaller `gpz` is the leading lane — that's the axis
//!   whose voxel boundary comes up next.
//! - **`gdz[2]`**: per-column-step delta added to `gpz` after a
//!   column advance. Constant per scanline.
//!
//! The trailing pieces of voxlap's `gline` (`cf[128]` seeding, gxmax
//! edge clipping, sky-radar bookkeeping, the `grouscanasm_scalar`
//! call) land in R4.3c+; this module only ships the projection
//! math because it's hand-verifiable in isolation.

use crate::camera_math::CameraState;
use crate::fixed::ftol;
use crate::opticast_prelude::PREC;

/// Per-scanline frustum data — what the `grouscan` ray-walker
/// consumes (R4.3c+). Values match voxlap's globals one-to-one.
#[derive(Debug, Clone, Copy)]
pub struct GlineFrustum {
    /// Left ray's projected world (x, y). Voxlap's `vd0` then
    /// rescales it to ground-plane principal-axis units; this struct
    /// holds the rescaled value.
    pub vd0: f32,
    /// Magnitude of the right ray's ground-plane vector
    /// (`sqrt(vx1² + vy1²)`). Voxlap's `vd1`.
    pub vd1: f32,
    /// Left ray's projected world z.
    pub vz0: f32,
    /// Right ray's projected world (x, y, z).
    pub vx1: f32,
    pub vy1: f32,
    pub vz1: f32,

    /// Voxel-column step in the ray's direction.
    /// `gixy[0]` = ±1 (x step), `gixy[1]` = ±`vsid` (y step).
    pub gixy: [i32; 2],
    /// Distance to next voxel-grid line in `PREC`-fixed-point.
    /// Lane with smaller value is the next-to-cross.
    pub gpz: [i32; 2],
    /// Per-column-step delta added to `gpz` after a column advance.
    pub gdz: [i32; 2],
}

/// Compute the per-scanline frustum.
///
/// `pos_xfrac` / `pos_yfrac` are the bilinear lane weights for the
/// initial DDA position — for an inside-grid camera these come from
/// `OpticastPrelude::{pos_xfrac, pos_yfrac}`; for the S1.3 outside-
/// camera path each scanline supplies a different pair derived from
/// the AABB-entry XY (the gline DDA starts at the entry point, not
/// at the camera position, since the camera sits past `vsid`).
///
/// `leng` is voxlap's `leng` parameter (the pixel run length the
/// caller will write into `radar`). It is *not* used by the
/// projection math here — it's part of the broader gline signature
/// and consumed in R4.3c when the `cf[128]` seed and `gi0` / `gi1`
/// step coefficients land. Kept in the signature now to flag the
/// dependency for the next sub-substage.
//
// Many casts cross signed↔unsigned and f32↔i32 by design; the
// projection math is voxlap's verbatim. clippy::float_cmp fires on
// the asm-style `vd0 sign-bit < 0` check; we use is_sign_negative()
// instead which is equivalent for valid floats.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::too_many_arguments
)]
#[must_use]
pub fn derive_gline_frustum(
    cs: &CameraState,
    pos_xfrac: [f32; 2],
    pos_yfrac: [f32; 2],
    vsid: u32,
    _leng: u32,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
) -> GlineFrustum {
    // World-space ray endpoints. Voxlap5.c:1153-1158.
    let vd0_x = x0 * cs.right[0] + y0 * cs.down[0] + cs.corn[0][0];
    let vd0_y = x0 * cs.right[1] + y0 * cs.down[1] + cs.corn[0][1];
    let vz0 = x0 * cs.right[2] + y0 * cs.down[2] + cs.corn[0][2];
    let vx1 = x1 * cs.right[0] + y1 * cs.down[0] + cs.corn[0][0];
    let vy1 = x1 * cs.right[1] + y1 * cs.down[1] + cs.corn[0][1];
    let vz1 = x1 * cs.right[2] + y1 * cs.down[2] + cs.corn[0][2];

    // f = sqrt(vx1² + vy1²): magnitude of the right ray's ground
    // projection. f1 = f/vx1, f2 = f/vy1.
    let f = (vx1 * vx1 + vy1 * vy1).sqrt();
    let f1 = f / vx1;
    let f2 = f / vy1;

    // Pick the dominant axis. Asm uses `fabs(vx1) > fabs(vy1)`.
    let mut vd0 = if vx1.abs() > vy1.abs() {
        vd0_x * f1
    } else {
        vd0_y * f2
    };
    // "vd0 MUST NOT be negative: bad for asm" — voxlap clamps via
    // bit-test on the sign bit. is_sign_negative() captures the same
    // bit (and treats -0.0 as negative, like voxlap).
    if vd0.is_sign_negative() {
        vd0 = 0.0;
    }
    let vd1 = f;

    // gdz lanes — fixed-point per-step deltas. Voxlap C's
    // `ftol(fabs(f1)*PREC, &gdz[0])` wraps modulo 2³² for floats
    // exceeding i32::MAX; Rust's `as i32` saturates instead. The
    // mismatch caused the floor-hairline artifact for near-axis-
    // aligned rays (see `project_roxlap_floor_hairline.md`); the
    // `ftol` helper in `fixed.rs` mirrors voxlap's wrap-on-overflow.
    let f1_abs = f1.abs();
    let gdz_0 = if f1_abs.is_finite() {
        ftol(f1_abs * PREC as f32)
    } else {
        -1
    };
    let f2_abs = f2.abs();
    let gdz_1 = if f2_abs.is_finite() {
        ftol(f2_abs * PREC as f32)
    } else {
        -1
    };

    // gixy — voxel-column step in the ray's direction. Asm masks the
    // sign bit of the float-as-int and uses it to flip the
    // SPTR_STRIDE / VSID*SPTR_STRIDE constant. Our port operates on
    // logical column indices, so the constants are 1 / vsid.
    let vsid_signed = vsid as i32;

    // gixy[0] = sign(vx1) * 1
    let gixy_0 = if vx1.is_sign_negative() { -1 } else { 1 };
    // gixy[1] = sign(vy1) * vsid
    let gixy_1 = if vy1.is_sign_negative() {
        -vsid_signed
    } else {
        vsid_signed
    };

    // gpz lanes — distance to next voxel-grid crossing, scaled by
    // PREC. Voxlap's `gposxfrac[sign-bit]` selects the right
    // fractional weight.
    let xfrac_idx = (vx1.to_bits() >> 31) as usize;
    let yfrac_idx = (vy1.to_bits() >> 31) as usize;

    // Asm clamp on overflow — gdz <= 0 means f1/f2 was huge or NaN.
    // Same wrap-not-saturate constraint applies to the gpz cast:
    // for `pos_xfrac ≈ 1` and `gdz` near i32::MAX, the product is
    // at the i32 boundary.
    let (gdz_clamped_0, gpz_0) = if gdz_0 <= 0 {
        (0, i32::MAX)
    } else {
        let gp = ftol(pos_xfrac[xfrac_idx] * gdz_0 as f32);
        (gdz_0, gp)
    };
    let (gdz_clamped_1, gpz_1) = if gdz_1 <= 0 {
        (0, i32::MAX)
    } else {
        let gp = ftol(pos_yfrac[yfrac_idx] * gdz_1 as f32);
        (gdz_1, gp)
    };

    GlineFrustum {
        vd0,
        vd1,
        vz0,
        vx1,
        vy1,
        vz1,
        gixy: [gixy_0, gixy_1],
        gpz: [gpz_0, gpz_1],
        gdz: [gdz_clamped_0, gdz_clamped_1],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera_math;
    use crate::opticast_prelude;
    use crate::opticast_prelude::OpticastPrelude;
    use crate::Camera;

    /// Looking-down camera at world origin, 640×480 viewport.
    /// gistr=[1,0,0], gihei=[0,1,0], gifor=[0,0,1].
    /// gcorn[0] = [-320, -240, 320].
    fn looking_down_state() -> (CameraState, OpticastPrelude) {
        let cam = Camera {
            pos: [0.0, 0.0, 0.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 1.0, 0.0],
            forward: [0.0, 0.0, 1.0],
        };
        let cs = camera_math::derive(&cam, 640, 480, 320.0, 240.0, 320.0);
        let prelude = opticast_prelude::derive_prelude(&cs, 2048, 1, 4, 1024, 1);
        (cs, prelude)
    }

    fn bit(v: f32) -> u32 {
        v.to_bits()
    }

    #[test]
    fn world_space_ray_endpoints_for_top_scanline() {
        // Scanline along the top edge of viewport: (0,0) → (640,0).
        let (cs, prelude) = looking_down_state();
        let g = derive_gline_frustum(
            &cs,
            prelude.pos_xfrac,
            prelude.pos_yfrac,
            2048,
            640,
            0.0,
            0.0,
            640.0,
            0.0,
        );

        // Right ray endpoint: vx1 = 640*1 + 0*0 + (-320) = 320.
        // vy1 = 640*0 + 0*1 + (-240) = -240.
        // vz1 = 640*0 + 0*0 + 320 = 320.
        assert_eq!(bit(g.vx1), bit(320.0));
        assert_eq!(bit(g.vy1), bit(-240.0));
        assert_eq!(bit(g.vz1), bit(320.0));
        // vz0 = 0*0 + 0*0 + 320 = 320.
        assert_eq!(bit(g.vz0), bit(320.0));
    }

    #[test]
    fn vd1_equals_ground_plane_magnitude() {
        // For (0,0)→(640,0): vd1 = sqrt(320² + 240²) = sqrt(160000) = 400.
        let (cs, prelude) = looking_down_state();
        let g = derive_gline_frustum(
            &cs,
            prelude.pos_xfrac,
            prelude.pos_yfrac,
            2048,
            640,
            0.0,
            0.0,
            640.0,
            0.0,
        );
        assert_eq!(bit(g.vd1), bit(400.0));
    }

    #[test]
    fn vd0_clamped_to_zero_when_negative() {
        // For (0,0)→(640,0): |vx1|=320 > |vy1|=240, so vd0 picks the x
        // branch. vd0_x = -320 (gcorn[0].x), f1 = 400/320 = 1.25.
        // vd0 = -320 * 1.25 = -400 → clamped to 0.
        let (cs, prelude) = looking_down_state();
        let g = derive_gline_frustum(
            &cs,
            prelude.pos_xfrac,
            prelude.pos_yfrac,
            2048,
            640,
            0.0,
            0.0,
            640.0,
            0.0,
        );
        assert_eq!(bit(g.vd0), bit(0.0));
    }

    #[test]
    fn gixy_signs_match_ray_direction() {
        // (0,0)→(640,0): vx1=320 > 0 → +1. vy1=-240 < 0 → -vsid.
        let (cs, prelude) = looking_down_state();
        let g = derive_gline_frustum(
            &cs,
            prelude.pos_xfrac,
            prelude.pos_yfrac,
            2048,
            640,
            0.0,
            0.0,
            640.0,
            0.0,
        );
        assert_eq!(g.gixy[0], 1);
        // For our origin-position prelude column_index = 0 (x=0,
        // y=0); vsid_signed extraction is sketchy when li_pos[1] = 0.
        // Move the camera slightly so the prelude has non-zero li_pos
        // and verify the sign of gixy[1].
        let cam = Camera {
            pos: [10.0, 10.0, 0.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 1.0, 0.0],
            forward: [0.0, 0.0, 1.0],
        };
        let cs2 = camera_math::derive(&cam, 640, 480, 320.0, 240.0, 320.0);
        let prelude2 = opticast_prelude::derive_prelude(&cs2, 2048, 1, 4, 1024, 1);
        let g2 = derive_gline_frustum(
            &cs2,
            prelude2.pos_xfrac,
            prelude2.pos_yfrac,
            2048,
            640,
            0.0,
            0.0,
            640.0,
            0.0,
        );
        assert!(
            g2.gixy[1] < 0,
            "gixy[1] = {}, expected negative",
            g2.gixy[1]
        );
        // Magnitude should be vsid (= 2048).
        assert_eq!(g2.gixy[1].unsigned_abs(), 2048);
    }

    #[test]
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn gdz_lanes_for_known_scanline() {
        // f1 = 400/320 = 1.25 → gdz[0] = 1.25 * PREC = 1310720.
        // f2 = 400/(-240) ≈ -1.6667 → |f2| * PREC ≈ 1747626.7 → 1747627.
        let (cs, prelude) = looking_down_state();
        let g = derive_gline_frustum(
            &cs,
            prelude.pos_xfrac,
            prelude.pos_yfrac,
            2048,
            640,
            0.0,
            0.0,
            640.0,
            0.0,
        );
        assert_eq!(g.gdz[0], (1.25_f32 * PREC as f32).round_ties_even() as i32);
        assert_eq!(
            g.gdz[1],
            ((400.0_f32 / 240.0_f32) * PREC as f32).round_ties_even() as i32
        );
    }

    #[test]
    fn gdz_clamps_to_zero_on_overflow() {
        // Force vx1 = 0 (vertical scanline) → f1 = inf → fabs * PREC
        // overflows i32 → ftol returns INT32_MIN → gdz <= 0 → clamp.
        // The lane with vy1 = 0 then has the same overflow.
        // For our looking-down camera, picking (320, 0) → (320, 480)
        // puts vx1 = vd0_x = (320*1 + 0*0 - 320) = 0 (but at endpoint
        // x1 = 320, f_at_x1 = 320*1 + 480*0 - 320 = 0 too). So vx1=0.
        let (cs, prelude) = looking_down_state();
        let g = derive_gline_frustum(
            &cs,
            prelude.pos_xfrac,
            prelude.pos_yfrac,
            2048,
            640,
            320.0,
            0.0,
            320.0,
            480.0,
        );
        // vx1 = 0 → f1 = inf → gdz[0] clamped to 0, gpz[0] = i32::MAX.
        assert_eq!(g.gdz[0], 0);
        assert_eq!(g.gpz[0], i32::MAX);
    }
}
