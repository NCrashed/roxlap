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

use crate::camera_math::CameraState;
use crate::fixed::{ftol, isshldiv16safe, lbound0, mulshr16, shldiv16};
use crate::opticast_prelude::OpticastPrelude;
use crate::projection::ProjectionRect;
use crate::rasterizer::{Rasterizer, ScanScratch};
use crate::ray_step::RayStep;

/// Per-frame state the four-quadrant scan loops read. Bundling it
/// keeps the per-quadrant driver signatures from sprouting eight
/// arguments each. None of the borrows are mutable — the loops only
/// produce calls through the [`Rasterizer`] trait + side effects on
/// [`ScanScratch`].
///
/// `y_start..y_end` is the iteration range in screen-y this scan
/// covers. For full-frame opticast this is `0..yres` (pre-R12.3
/// behaviour). For per-strip parallel rendering (R12.3.1+) each
/// strip narrows the range; the four scan loops clip their sy
/// iteration accordingly.
#[derive(Debug, Clone, Copy)]
pub struct ScanContext<'a> {
    pub proj: &'a ProjectionRect,
    pub rs: &'a RayStep,
    pub prelude: &'a OpticastPrelude,
    pub xres: i32,
    /// First sy this strip processes (inclusive).
    pub y_start: i32,
    /// One past the last sy (exclusive). Equals the framebuffer's
    /// `yres` for full-frame opticast.
    pub y_end: i32,
    pub anginc: f32,
    /// Camera basis + frustum corners. Concrete rasterizers' real
    /// `gline` (R4.3a-rewire-3b) projects per-ray endpoints through
    /// `camera_state.right` / `down` / `corn[0]` to derive the
    /// world-space ray.
    pub camera_state: &'a CameraState,
    /// Voxlap's `gstartz0` — top of the camera's air gap. `0` for
    /// the column-top case (above the first slab).
    pub camera_gstartz0: i32,
    /// Voxlap's `gstartz1` — bottom of the camera's air gap (= top
    /// of the slab below it).
    pub camera_gstartz1: i32,
    /// Byte offset within the camera's column to the slab whose top
    /// bounds the air gap from below. `0` means column-top.
    pub camera_vptr_offset: usize,
    /// S4B.6.e: chunk-z that owns `camera_vptr_offset`. For the
    /// in-camera-chunk case this equals `prelude.camera_chunk_idx[2]`.
    /// For cross-chunk look-down (camera in all-air-bedrock column
    /// with terrain in a deeper chunk) it points to the chunk that
    /// holds the real floor. `gline_seed` reads it to route
    /// state.column / `slab_buf` to the right chunk so rays start
    /// walking the real-terrain chunk directly.
    pub camera_seed_chunk_z: i32,
}

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

/// Drive the **top quadrant** scan — the fan rooted at the projection
/// centre `(cx, cy)` opening upward to the viewport's top edge.
///
/// Port of `voxlap5.c:opticast` lines 2373..2406. Two passes:
///
/// 1. **Ray cast pass.** Issue `j` rays via `vline_clip` + the
///    rasterizer's `gline`. Each ray's hit-record range is recorded in
///    `scratch.angstart[i]` so the scanline pass can dereference it.
/// 2. **Scanline pass.** For each screen row `sy` from `cy` upward,
///    compute the column range `(p0, p1)` that maps into this fan and
///    dispatch `rasterizer.hrend(p0, sy, p1, u, ui, i)`. The `u` /
///    `ui` are voxlap's `shldiv16`-based fixed-point per-pixel
///    counters that the SSE rasterizers (R5) consume.
///
/// Early-out when `proj.fy >= 0` (no fan above `cy`) or `j == 0` (no
/// pixels wide enough to scan), matching voxlap's outer guard
/// (`(fy < 0) && (j > 0)`).
//
// Length / arg count: the math is dense and one-shot. Splitting into
// helper functions hides the relationship between voxlap's source and
// this port; readers cross-checking against voxlap5.c benefit from a
// single contiguous body. See PORTING-RUST.md / R4.1f3 for context.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::too_many_lines
)]
pub fn top_quadrant<R: Rasterizer>(
    rasterizer: &mut R,
    scratch: &mut ScanScratch,
    ctx: &ScanContext<'_>,
) {
    let p = ctx.proj;
    let rs = ctx.rs;
    let forward_z_sign = ctx.prelude.forward_z_sign;

    let j_count = ((p.x1 - p.x0) / ctx.anginc).round_ties_even() as i32;
    if p.fy >= 0.0 || j_count <= 0 {
        return;
    }

    let ff_x = (p.x1 - p.x0) / j_count as f32;
    let grd = 1.0 / (p.wy0 - p.cy);

    // Voxlap stamps -giforzsgn into skycurdir at the start of every
    // quadrant; reset_for_quadrant takes care of the cursor + sign.
    scratch.reset_for_quadrant(-forward_z_sign);

    // -- Pass 1: cast j_count rays from (cx, cy) toward y = wy0. --
    let mut f_ray = p.x0 + ff_x * 0.5;
    for i in 0..j_count as usize {
        let (iy0, iy1) = vline_clip(p.cx, p.cy, f_ray, p.wy0, grd, p);

        // angstart[i] = gscanptr ± (p0 / p1) per giforzsgn — voxlap's
        // pointer-arithmetic stores a `castdat*` that may land before
        // radar[0]. Mirror with isize.
        let gscan = scratch.gscanptr as isize;
        scratch.angstart[i] = if forward_z_sign < 0 {
            gscan + iy0 as isize
        } else {
            gscan - iy1 as isize
        };

        // Issue gline. Endpoints are projected back to the screen
        // x = (iy - wy0) * dxy + f_ray (voxlap's vline formula).
        let length = iy1.abs_diff(iy0);
        let dxy = (f_ray - p.cx) * grd;
        let cast_x0 = (iy0 as f32 - p.wy0) * dxy + f_ray;
        let cast_x1 = (iy1 as f32 - p.wy0) * dxy + f_ray;
        rasterizer.gline(scratch, length, cast_x0, iy0 as f32, cast_x1, iy1 as f32);

        // Advance gscanptr by |iy1 - iy0| + 1 castdat slots.
        scratch.gscanptr += length as usize + 1;

        f_ray += ff_x;
    }

    // -- Pass 2: scanline rasterization. --
    // Voxlap shifts j into Q16 fixed-point; range stays in i32 for
    // realistic xres.
    let j_fixed = j_count << 16;
    let f_scale = j_fixed as f32 / ((p.x1 - p.x0) * grd);
    // f_scale = j_fixed / ((x_range)*grd) blows up for level cameras
    // where grd ≈ 1/cy is tiny; the `kadd` and `kmul` products land
    // past i32::MAX. Voxlap C's ftol wraps; route through the helper.
    let kadd = ftol((p.cx - p.x0) * grd * f_scale);

    let p1_init = (p.cx - 0.5).round_ties_even() as i32;
    let mut p0 = lbound0(p1_init + 1, ctx.xres);
    let mut p1 = lbound0(p1_init, ctx.xres);

    let mut sy = (p.cy - 0.50005).round_ties_even() as i32;
    if sy >= ctx.y_end {
        sy = ctx.y_end - 1;
    }

    // Anti-crash: voxlap's float-overflow guard. Step sy down until
    // ftol(f_scale) won't push (sy<<16)-cy16 into a value the
    // shldiv16 below would overflow on. Clipped to ctx.y_start so
    // strip renders don't iterate below the strip's top edge.
    let ff_check = ((p1 as f32 - p.cx).abs() + 1.0) * f_scale / 2_147_483_647.0 + p.cy;
    while ff_check < sy as f32 && sy >= ctx.y_start {
        sy -= 1;
    }
    if sy < ctx.y_start {
        return;
    }

    let kmul = ftol(f_scale);
    while sy >= ctx.y_start {
        if isshldiv16safe(kmul, (sy << 16) - rs.cy16) != 0 {
            break;
        }
        sy -= 1;
    }

    // Ray index `i` walks paired with `sy`. Sign of step matches
    // -giforzsgn (i.e. when looking up sy decreases, i increases for
    // -1 sign or decreases for +1 sign).
    let mut i = if forward_z_sign < 0 { -sy } else { sy };
    while sy >= ctx.y_start {
        let ui = shldiv16(kmul, (sy << 16) - rs.cy16);
        let mut u = mulshr16((p0 << 16) - rs.cx16, ui) + kadd;

        // Walk p0 left while the ray index is past the previous one.
        while p0 > 0 && u >= ui {
            u -= ui;
            p0 -= 1;
        }
        // Walk p1 right until the ray index passes the j_fixed cap.
        let mut u1 = (p1 - p0) * ui + u;
        while p1 < ctx.xres && u1 < j_fixed {
            u1 += ui;
            p1 += 1;
        }

        if p0 < p1 {
            rasterizer.hrend(scratch, p0, sy, p1, u, ui, i);
        }

        sy -= 1;
        i -= forward_z_sign;
    }
}

/// Drive the **bottom quadrant** scan — the mirror of [`top_quadrant`]
/// fanning *downward* from `(cx, cy)` to `y = wy1`.
///
/// Port of `voxlap5.c:opticast` lines 2447..2480. Same vline + hrend
/// shape as top, with these flips:
/// - guard `gy > 0` (centre is above the viewport bottom)
/// - rays iterate from `x3` to `x2` (the bottom-edge corner-cut Xs)
/// - target y = `wy1`, not `wy0`
/// - `skycurdir = +giforzsgn` (top uses `-giforzsgn`)
/// - `angstart[i]` sign convention is flipped: `gscanptr - p0` /
///   `gscanptr + p1` (top: `+ p0` / `- p1`)
/// - scanline walks `sy` *upward* through `0..yres`
/// - ray index `i` starts at `+sy` / `-sy` (top: opposite)
//
// See top_quadrant for the rationale on body length / arg count.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::too_many_lines
)]
pub fn bottom_quadrant<R: Rasterizer>(
    rasterizer: &mut R,
    scratch: &mut ScanScratch,
    ctx: &ScanContext<'_>,
) {
    let p = ctx.proj;
    let rs = ctx.rs;
    let forward_z_sign = ctx.prelude.forward_z_sign;

    let j_count = ((p.x2 - p.x3) / ctx.anginc).round_ties_even() as i32;
    if p.gy <= 0.0 || j_count <= 0 {
        return;
    }

    let ff_x = (p.x2 - p.x3) / j_count as f32;
    let grd = 1.0 / (p.wy1 - p.cy);

    // Bottom quadrant uses +giforzsgn for skycurdir; top used -.
    scratch.reset_for_quadrant(forward_z_sign);

    // -- Pass 1: cast j_count rays from (cx, cy) toward y = wy1. --
    let mut f_ray = p.x3 + ff_x * 0.5;
    for i in 0..j_count as usize {
        let (iy0, iy1) = vline_clip(p.cx, p.cy, f_ray, p.wy1, grd, p);

        // Sign flip from top: gscanptr - p0 (giforzsgn < 0)
        // / gscanptr + p1 (else).
        let gscan = scratch.gscanptr as isize;
        scratch.angstart[i] = if forward_z_sign < 0 {
            gscan - iy0 as isize
        } else {
            gscan + iy1 as isize
        };

        let length = iy1.abs_diff(iy0);
        let dxy = (f_ray - p.cx) * grd;
        let cast_x0 = (iy0 as f32 - p.wy1) * dxy + f_ray;
        let cast_x1 = (iy1 as f32 - p.wy1) * dxy + f_ray;
        rasterizer.gline(scratch, length, cast_x0, iy0 as f32, cast_x1, iy1 as f32);

        scratch.gscanptr += length as usize + 1;

        f_ray += ff_x;
    }

    // -- Pass 2: scanline rasterization (sy walks upward). --
    let j_fixed = j_count << 16;
    let f_scale = j_fixed as f32 / ((p.x2 - p.x3) * grd);
    let kadd = ftol((p.cx - p.x3) * grd * f_scale);

    let p1_init = (p.cx - 0.5).round_ties_even() as i32;
    let mut p0 = lbound0(p1_init + 1, ctx.xres);
    let mut p1 = lbound0(p1_init, ctx.xres);

    let mut sy = (p.cy + 0.50005).round_ties_even() as i32;
    if sy < ctx.y_start {
        sy = ctx.y_start;
    }

    let ff_check = ((p1 as f32 - p.cx).abs() + 1.0) * f_scale / 2_147_483_647.0 + p.cy;
    while ff_check > sy as f32 && sy < ctx.y_end {
        sy += 1;
    }
    if sy >= ctx.y_end {
        return;
    }

    let kmul = ftol(f_scale);
    while sy < ctx.y_end {
        if isshldiv16safe(kmul, (sy << 16) - rs.cy16) != 0 {
            break;
        }
        sy += 1;
    }

    let mut i = if forward_z_sign < 0 { sy } else { -sy };
    while sy < ctx.y_end {
        let ui = shldiv16(kmul, (sy << 16) - rs.cy16);
        let mut u = mulshr16((p0 << 16) - rs.cx16, ui) + kadd;

        while p0 > 0 && u >= ui {
            u -= ui;
            p0 -= 1;
        }
        let mut u1 = (p1 - p0) * ui + u;
        while p1 < ctx.xres && u1 < j_fixed {
            u1 += ui;
            p1 += 1;
        }

        if p0 < p1 {
            rasterizer.hrend(scratch, p0, sy, p1, u, ui, i);
        }

        sy += 1;
        i -= forward_z_sign;
    }
}

/// Drive the **right quadrant** scan — the fan rooted at `(cx, cy)`
/// opening rightward to the viewport's right edge (`x = wx1`).
///
/// Port of `voxlap5.c:opticast` lines 2408..2444. Hline-based ray
/// cast, then a column walk that populates `scratch.uurend` /
/// `scratch.lastx` per screen-x, finally a vrend dispatch loop over
/// the resulting per-row x-boundary.
//
// See top_quadrant for the rationale on body length / arg count.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::too_many_lines
)]
pub fn right_quadrant<R: Rasterizer>(
    rasterizer: &mut R,
    scratch: &mut ScanScratch,
    ctx: &ScanContext<'_>,
) {
    let p = ctx.proj;
    let rs = ctx.rs;
    let forward_z_sign = ctx.prelude.forward_z_sign;

    let j_count = ((p.y2 - p.y1) / ctx.anginc).round_ties_even() as i32;
    if p.gx <= 0.0 || j_count <= 0 {
        return;
    }

    let ff_y = (p.y2 - p.y1) / j_count as f32;
    let grd = 1.0 / (p.wx1 - p.cx);

    scratch.reset_for_quadrant(-forward_z_sign);

    // -- Pass 1: cast j_count rays from (cx, cy) toward x = wx1. --
    let mut f_ray = p.y1 + ff_y * 0.5;
    for i in 0..j_count as usize {
        let (ix0, ix1) = hline_clip(p.cx, p.cy, p.wx1, f_ray, grd, p);

        // Right quadrant: gscanptr - p0 (giforzsgn < 0)
        // / gscanptr + p1 (else).
        let gscan = scratch.gscanptr as isize;
        scratch.angstart[i] = if forward_z_sign < 0 {
            gscan - ix0 as isize
        } else {
            gscan + ix1 as isize
        };

        // gline endpoints projected back to screen y at each ix.
        let length = ix1.abs_diff(ix0);
        let dyx = (f_ray - p.cy) * grd;
        let cast_y0 = (ix0 as f32 - p.wx1) * dyx + f_ray;
        let cast_y1 = (ix1 as f32 - p.wx1) * dyx + f_ray;
        rasterizer.gline(scratch, length, ix0 as f32, cast_y0, ix1 as f32, cast_y1);

        scratch.gscanptr += length as usize + 1;

        f_ray += ff_y;
    }

    // -- Pass 2: column walk + vrend dispatch. --
    let j_fixed = j_count << 16;
    let f_scale = j_fixed as f32 / ((p.y2 - p.y1) * grd);
    let kadd = ftol((p.cy - p.y1) * grd * f_scale);

    let p1_init = (p.cy - 0.5).round_ties_even() as i32;
    // Strip-aware clamp: p0/p1 are sy values that must stay inside
    // [y_start, y_end). lbound0 only clamps to [0, b]; for strip
    // renders that y_start > 0, the extra `.max(y_start)` floors it.
    let mut p0 = lbound0(p1_init + 1, ctx.y_end).max(ctx.y_start);
    let mut p1 = lbound0(p1_init, ctx.y_end).max(ctx.y_start);

    let mut sx = (p.cx + 0.50005).round_ties_even() as i32;
    if sx < 0 {
        sx = 0;
    }

    let ff_check = ((p1 as f32 - p.cy).abs() + 1.0) * f_scale / 2_147_483_647.0 + p.cx;
    while ff_check > sx as f32 && sx < ctx.xres {
        sx += 1;
    }
    if sx >= ctx.xres {
        return;
    }

    let kmul = ftol(f_scale);
    while sx < ctx.xres {
        if isshldiv16safe(kmul, (sx << 16) - rs.cx16) != 0 {
            break;
        }
        sx += 1;
    }

    while sx < ctx.xres {
        let ui = shldiv16(kmul, (sx << 16) - rs.cx16);
        let mut u = mulshr16((p0 << 16) - rs.cy16, ui) + kadd;

        // Walk p0 left (toward strip top), populating lastx.
        while p0 > ctx.y_start && u >= ui {
            u -= ui;
            p0 -= 1;
            scratch.lastx[p0 as usize] = sx;
        }
        // Stamp this column's u/ui pair for the vrend dispatch.
        scratch.uurend[sx as usize] = u;
        scratch.uurend[sx as usize + scratch.uurend_half_stride] = ui;
        u += (p1 - p0) * ui;
        // Walk p1 right (toward strip bottom), populating lastx.
        while p1 < ctx.y_end && u < j_fixed {
            u += ui;
            scratch.lastx[p1 as usize] = sx;
            p1 += 1;
        }
        sx += 1;
    }

    // vrend dispatch: for each y row in [p0, p1), call vrend with
    // the row's x boundary. The plc / sign flips with giforzsgn.
    if forward_z_sign < 0 {
        for sy in p0..p1 {
            let lx = scratch.lastx[sy as usize];
            rasterizer.vrend(scratch, lx, sy, ctx.xres, lx, 1);
        }
    } else {
        for sy in p0..p1 {
            let lx = scratch.lastx[sy as usize];
            rasterizer.vrend(scratch, lx, sy, ctx.xres, -lx, -1);
        }
    }
}

/// Drive the **left quadrant** scan — the fan rooted at `(cx, cy)`
/// opening leftward to the viewport's left edge (`x = wx0`).
///
/// Port of `voxlap5.c:opticast` lines 2482..2517. Hline-based ray
/// cast (mirror of right), then column walk going leftward, then a
/// vrend dispatch over the resulting per-row x-boundary. Unlike the
/// right quadrant the dispatch passes `xres + 1` style range with
/// `0` as the start; voxlap calls `vrend(0, sy, lastx[sy]+1, 0,
/// giforzsgn)` to render the left slice.
//
// See top_quadrant for the rationale on body length / arg count.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::too_many_lines
)]
pub fn left_quadrant<R: Rasterizer>(
    rasterizer: &mut R,
    scratch: &mut ScanScratch,
    ctx: &ScanContext<'_>,
) {
    let p = ctx.proj;
    let rs = ctx.rs;
    let forward_z_sign = ctx.prelude.forward_z_sign;

    let j_count = ((p.y3 - p.y0) / ctx.anginc).round_ties_even() as i32;
    if p.fx >= 0.0 || j_count <= 0 {
        return;
    }

    let ff_y = (p.y3 - p.y0) / j_count as f32;
    let grd = 1.0 / (p.wx0 - p.cx);

    // Left quadrant: skycurdir = +giforzsgn (opposite of top).
    scratch.reset_for_quadrant(forward_z_sign);

    // -- Pass 1: cast j_count rays from (cx, cy) toward x = wx0. --
    let mut f_ray = p.y0 + ff_y * 0.5;
    for i in 0..j_count as usize {
        let (ix0, ix1) = hline_clip(p.cx, p.cy, p.wx0, f_ray, grd, p);

        // Left quadrant: gscanptr + p0 (giforzsgn < 0)
        // / gscanptr - p1 (else).
        let gscan = scratch.gscanptr as isize;
        scratch.angstart[i] = if forward_z_sign < 0 {
            gscan + ix0 as isize
        } else {
            gscan - ix1 as isize
        };

        let length = ix1.abs_diff(ix0);
        let dyx = (f_ray - p.cy) * grd;
        let cast_y0 = (ix0 as f32 - p.wx0) * dyx + f_ray;
        let cast_y1 = (ix1 as f32 - p.wx0) * dyx + f_ray;
        rasterizer.gline(scratch, length, ix0 as f32, cast_y0, ix1 as f32, cast_y1);

        scratch.gscanptr += length as usize + 1;

        f_ray += ff_y;
    }

    // -- Pass 2: column walk (sx walks leftward) + vrend dispatch. --
    let j_fixed = j_count << 16;
    let f_scale = j_fixed as f32 / ((p.y3 - p.y0) * grd);
    let kadd = ftol((p.cy - p.y0) * grd * f_scale);

    let p1_init = (p.cy - 0.5).round_ties_even() as i32;
    // Strip-aware p0/p1 clamp; see right_quadrant for the same shape.
    let mut p0 = lbound0(p1_init + 1, ctx.y_end).max(ctx.y_start);
    let mut p1 = lbound0(p1_init, ctx.y_end).max(ctx.y_start);

    let mut sx = (p.cx - 0.50005).round_ties_even() as i32;
    if sx >= ctx.xres {
        sx = ctx.xres - 1;
    }

    let ff_check = ((p1 as f32 - p.cy).abs() + 1.0) * f_scale / 2_147_483_647.0 + p.cx;
    while ff_check < sx as f32 && sx >= 0 {
        sx -= 1;
    }
    if sx < 0 {
        return;
    }

    let kmul = ftol(f_scale);
    while sx >= 0 {
        if isshldiv16safe(kmul, (sx << 16) - rs.cx16) != 0 {
            break;
        }
        sx -= 1;
    }

    while sx >= 0 {
        let ui = shldiv16(kmul, (sx << 16) - rs.cx16);
        let mut u = mulshr16((p0 << 16) - rs.cy16, ui) + kadd;

        while p0 > ctx.y_start && u >= ui {
            u -= ui;
            p0 -= 1;
            scratch.lastx[p0 as usize] = sx;
        }
        scratch.uurend[sx as usize] = u;
        scratch.uurend[sx as usize + scratch.uurend_half_stride] = ui;
        u += (p1 - p0) * ui;
        while p1 < ctx.y_end && u < j_fixed {
            u += ui;
            scratch.lastx[p1 as usize] = sx;
            p1 += 1;
        }
        sx -= 1;
    }

    for sy in p0..p1 {
        let lx = scratch.lastx[sy as usize];
        rasterizer.vrend(scratch, 0, sy, lx + 1, 0, forward_z_sign);
    }
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
        projection::derive_projection(&s, 640, 480, 320.0, 240.0, 320.0, 1.0)
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

    // --- top quadrant tests ---

    use crate::opticast_prelude;
    use crate::rasterizer::{Rasterizer, ScanScratch};
    use crate::ray_step;

    /// Recording rasterizer that counts gline / hrend / vrend calls
    /// and stores them as flat events for assertions.
    #[derive(Debug, Default)]
    struct Recorder {
        gline_calls: u32,
        hrend_calls: u32,
        vrend_calls: u32,
        first_hrend: Option<(i32, i32, i32)>, // (sx, sy, p1) snapshot
    }

    impl Rasterizer for Recorder {
        fn gline(&mut self, _: &mut ScanScratch, _: u32, _: f32, _: f32, _: f32, _: f32) {
            self.gline_calls += 1;
        }
        fn hrend(
            &mut self,
            _: &mut ScanScratch,
            sx: i32,
            sy: i32,
            p1: i32,
            _: i32,
            _: i32,
            _: i32,
        ) {
            if self.first_hrend.is_none() {
                self.first_hrend = Some((sx, sy, p1));
            }
            self.hrend_calls += 1;
        }
        fn vrend(&mut self, _: &mut ScanScratch, _: i32, _: i32, _: i32, _: i32, _: i32) {
            self.vrend_calls += 1;
        }
    }

    /// Build a `ScanContext` for a "looking down" camera at the
    /// origin of a 2048-wide world. cx/cy land at viewport centre, so
    /// all four quadrant fans are non-trivial — the top fan in
    /// particular covers the half of the screen above y = cy = 240.
    fn looking_down_context() -> (
        crate::camera_math::CameraState,
        crate::projection::ProjectionRect,
        crate::ray_step::RayStep,
        crate::opticast_prelude::OpticastPrelude,
    ) {
        let cam = crate::Camera {
            pos: [1024.0, 1024.0, 128.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 1.0, 0.0],
            forward: [0.0, 0.0, 1.0],
        };
        let s = camera_math::derive(&cam, 640, 480, 320.0, 240.0, 320.0);
        let proj = crate::projection::derive_projection(&s, 640, 480, 320.0, 240.0, 320.0, 1.0);
        let rs = ray_step::derive_ray_step(&s, proj.cx, proj.cy, 320.0);
        let prelude = opticast_prelude::derive_prelude(&s, 2048, 1, 4, 1024, 1);
        (s, proj, rs, prelude)
    }

    #[test]
    fn top_quadrant_skips_when_centre_below_top_edge() {
        // Construct a synthetic ProjectionRect with cy way above wy0
        // (centre is far above the viewport — fy = wy0 - cy is large
        // positive, NOT < 0). This is the early-out case.
        let (cs, proj_lookdown, rs, prelude) = looking_down_context();
        let mut proj = proj_lookdown;
        // Force fy >= 0 by moving cy way up.
        proj.cy = -1000.0;
        proj.fy = proj.wy0 - proj.cy; // positive now

        let mut rec = Recorder::default();
        let mut scratch = ScanScratch::new_for_size(640, 480, 2048);
        let ctx = ScanContext {
            proj: &proj,
            rs: &rs,
            prelude: &prelude,
            xres: 640,
            y_start: 0,
            y_end: 480,
            anginc: 1.0,
            camera_state: &cs,
            camera_gstartz0: 0,
            camera_gstartz1: 0,
            camera_vptr_offset: 0,
            camera_seed_chunk_z: 0,
        };
        top_quadrant(&mut rec, &mut scratch, &ctx);
        assert_eq!(rec.gline_calls, 0);
        assert_eq!(rec.hrend_calls, 0);
    }

    #[test]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn top_quadrant_casts_one_ray_per_anginc_step() {
        // Looking-down camera: cx = 320, cy = 240. Top fan covers the
        // [x0, x1] horizontal range above cy. After the corner-cut
        // pass, x1 - x0 ≈ 2 * sqrt(320 * 240) ≈ 554, plus the ±0.01
        // bias. j_count = round((x1 - x0) / 1) = ~554.
        let (cs, proj, rs, prelude) = looking_down_context();
        let mut rec = Recorder::default();
        let mut scratch = ScanScratch::new_for_size(640, 480, 2048);
        let ctx = ScanContext {
            proj: &proj,
            rs: &rs,
            prelude: &prelude,
            xres: 640,
            y_start: 0,
            y_end: 480,
            anginc: 1.0,
            camera_state: &cs,
            camera_gstartz0: 0,
            camera_gstartz1: 0,
            camera_vptr_offset: 0,
            camera_seed_chunk_z: 0,
        };
        top_quadrant(&mut rec, &mut scratch, &ctx);
        // We don't pin the exact value (anginc rounding + ±0.01 bias),
        // but the count must be in the expected ballpark of ~ |x1-x0|.
        let expected = ((proj.x1 - proj.x0) / 1.0).round_ties_even() as u32;
        assert_eq!(rec.gline_calls, expected);
    }

    #[test]
    fn top_quadrant_emits_at_least_one_hrend() {
        // For the looking-down camera, sy walks downward from cy ≈ 240
        // through all rows above, so hrend should fire at least once.
        let (cs, proj, rs, prelude) = looking_down_context();
        let mut rec = Recorder::default();
        let mut scratch = ScanScratch::new_for_size(640, 480, 2048);
        let ctx = ScanContext {
            proj: &proj,
            rs: &rs,
            prelude: &prelude,
            xres: 640,
            y_start: 0,
            y_end: 480,
            anginc: 1.0,
            camera_state: &cs,
            camera_gstartz0: 0,
            camera_gstartz1: 0,
            camera_vptr_offset: 0,
            camera_seed_chunk_z: 0,
        };
        top_quadrant(&mut rec, &mut scratch, &ctx);
        assert!(rec.hrend_calls > 0, "expected ≥ 1 hrend, got 0");
        // The first hrend's sy is the topmost row scanned; it should
        // be ≤ initial sy (= round(cy - 0.50005) = 239).
        let (_, sy, _) = rec.first_hrend.expect("first hrend recorded");
        assert!(sy <= 239, "first hrend sy = {sy}, expected ≤ 239");
    }

    #[test]
    fn bottom_quadrant_skips_when_centre_below_bottom_edge() {
        // Force gy <= 0 by placing cy below the viewport bottom.
        let (cs, proj_lookdown, rs, prelude) = looking_down_context();
        let mut proj = proj_lookdown;
        proj.cy = 1000.0;
        proj.gy = proj.wy1 - proj.cy; // negative now
        let mut rec = Recorder::default();
        let mut scratch = ScanScratch::new_for_size(640, 480, 2048);
        let ctx = ScanContext {
            proj: &proj,
            rs: &rs,
            prelude: &prelude,
            xres: 640,
            y_start: 0,
            y_end: 480,
            anginc: 1.0,
            camera_state: &cs,
            camera_gstartz0: 0,
            camera_gstartz1: 0,
            camera_vptr_offset: 0,
            camera_seed_chunk_z: 0,
        };
        bottom_quadrant(&mut rec, &mut scratch, &ctx);
        assert_eq!(rec.gline_calls, 0);
        assert_eq!(rec.hrend_calls, 0);
    }

    #[test]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn bottom_quadrant_casts_one_ray_per_anginc_step() {
        // Looking-down camera has both top and bottom fans engaged;
        // bottom uses x3..x2 width which equals top's x0..x1 width
        // by symmetry of the corner-cut quadrilateral.
        let (cs, proj, rs, prelude) = looking_down_context();
        let mut rec = Recorder::default();
        let mut scratch = ScanScratch::new_for_size(640, 480, 2048);
        let ctx = ScanContext {
            proj: &proj,
            rs: &rs,
            prelude: &prelude,
            xres: 640,
            y_start: 0,
            y_end: 480,
            anginc: 1.0,
            camera_state: &cs,
            camera_gstartz0: 0,
            camera_gstartz1: 0,
            camera_vptr_offset: 0,
            camera_seed_chunk_z: 0,
        };
        bottom_quadrant(&mut rec, &mut scratch, &ctx);
        let expected = ((proj.x2 - proj.x3) / 1.0).round_ties_even() as u32;
        assert_eq!(rec.gline_calls, expected);
    }

    #[test]
    fn bottom_quadrant_emits_at_least_one_hrend() {
        let (cs, proj, rs, prelude) = looking_down_context();
        let mut rec = Recorder::default();
        let mut scratch = ScanScratch::new_for_size(640, 480, 2048);
        let ctx = ScanContext {
            proj: &proj,
            rs: &rs,
            prelude: &prelude,
            xres: 640,
            y_start: 0,
            y_end: 480,
            anginc: 1.0,
            camera_state: &cs,
            camera_gstartz0: 0,
            camera_gstartz1: 0,
            camera_vptr_offset: 0,
            camera_seed_chunk_z: 0,
        };
        bottom_quadrant(&mut rec, &mut scratch, &ctx);
        assert!(rec.hrend_calls > 0, "expected ≥ 1 hrend, got 0");
        // First hrend's sy is the topmost scanned row of the bottom
        // fan; with cy = 240 it should land at >= 240 (we walk down).
        let (_, sy, _) = rec.first_hrend.expect("first hrend recorded");
        assert!(sy >= 240, "first hrend sy = {sy}, expected ≥ 240");
    }

    #[test]
    fn right_quadrant_skips_when_centre_right_of_viewport() {
        let (cs, proj_lookdown, rs, prelude) = looking_down_context();
        let mut proj = proj_lookdown;
        proj.cx = 1000.0;
        proj.gx = proj.wx1 - proj.cx; // negative
        let mut rec = Recorder::default();
        let mut scratch = ScanScratch::new_for_size(640, 480, 2048);
        let ctx = ScanContext {
            proj: &proj,
            rs: &rs,
            prelude: &prelude,
            xres: 640,
            y_start: 0,
            y_end: 480,
            anginc: 1.0,
            camera_state: &cs,
            camera_gstartz0: 0,
            camera_gstartz1: 0,
            camera_vptr_offset: 0,
            camera_seed_chunk_z: 0,
        };
        right_quadrant(&mut rec, &mut scratch, &ctx);
        assert_eq!(rec.gline_calls + rec.vrend_calls, 0);
    }

    #[test]
    fn left_quadrant_skips_when_centre_left_of_viewport() {
        let (cs, proj_lookdown, rs, prelude) = looking_down_context();
        let mut proj = proj_lookdown;
        proj.cx = -1000.0;
        proj.fx = proj.wx0 - proj.cx; // positive (large)
        let mut rec = Recorder::default();
        let mut scratch = ScanScratch::new_for_size(640, 480, 2048);
        let ctx = ScanContext {
            proj: &proj,
            rs: &rs,
            prelude: &prelude,
            xres: 640,
            y_start: 0,
            y_end: 480,
            anginc: 1.0,
            camera_state: &cs,
            camera_gstartz0: 0,
            camera_gstartz1: 0,
            camera_vptr_offset: 0,
            camera_seed_chunk_z: 0,
        };
        left_quadrant(&mut rec, &mut scratch, &ctx);
        assert_eq!(rec.gline_calls + rec.vrend_calls, 0);
    }

    #[test]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn right_quadrant_casts_one_ray_per_anginc_step() {
        let (cs, proj, rs, prelude) = looking_down_context();
        let mut rec = Recorder::default();
        let mut scratch = ScanScratch::new_for_size(640, 480, 2048);
        let ctx = ScanContext {
            proj: &proj,
            rs: &rs,
            prelude: &prelude,
            xres: 640,
            y_start: 0,
            y_end: 480,
            anginc: 1.0,
            camera_state: &cs,
            camera_gstartz0: 0,
            camera_gstartz1: 0,
            camera_vptr_offset: 0,
            camera_seed_chunk_z: 0,
        };
        right_quadrant(&mut rec, &mut scratch, &ctx);
        let expected = ((proj.y2 - proj.y1) / 1.0).round_ties_even() as u32;
        assert_eq!(rec.gline_calls, expected);
    }

    #[test]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn left_quadrant_casts_one_ray_per_anginc_step() {
        let (cs, proj, rs, prelude) = looking_down_context();
        let mut rec = Recorder::default();
        let mut scratch = ScanScratch::new_for_size(640, 480, 2048);
        let ctx = ScanContext {
            proj: &proj,
            rs: &rs,
            prelude: &prelude,
            xres: 640,
            y_start: 0,
            y_end: 480,
            anginc: 1.0,
            camera_state: &cs,
            camera_gstartz0: 0,
            camera_gstartz1: 0,
            camera_vptr_offset: 0,
            camera_seed_chunk_z: 0,
        };
        left_quadrant(&mut rec, &mut scratch, &ctx);
        let expected = ((proj.y3 - proj.y0) / 1.0).round_ties_even() as u32;
        assert_eq!(rec.gline_calls, expected);
    }

    #[test]
    fn right_and_left_quadrants_emit_vrend() {
        let (cs, proj, rs, prelude) = looking_down_context();
        let mut scratch = ScanScratch::new_for_size(640, 480, 2048);
        let ctx = ScanContext {
            proj: &proj,
            rs: &rs,
            prelude: &prelude,
            xres: 640,
            y_start: 0,
            y_end: 480,
            anginc: 1.0,
            camera_state: &cs,
            camera_gstartz0: 0,
            camera_gstartz1: 0,
            camera_vptr_offset: 0,
            camera_seed_chunk_z: 0,
        };
        let mut rec_right = Recorder::default();
        right_quadrant(&mut rec_right, &mut scratch, &ctx);
        assert!(
            rec_right.vrend_calls > 0,
            "right quadrant: expected ≥ 1 vrend"
        );

        scratch = ScanScratch::new_for_size(640, 480, 2048);
        let mut rec_left = Recorder::default();
        left_quadrant(&mut rec_left, &mut scratch, &ctx);
        assert!(
            rec_left.vrend_calls > 0,
            "left quadrant: expected ≥ 1 vrend"
        );
    }

    #[test]
    fn top_quadrant_advances_gscanptr_by_ray_lengths() {
        // After all rays cast, scratch.gscanptr should equal sum of
        // (|iy1-iy0|+1) over all rays.
        let (cs, proj, rs, prelude) = looking_down_context();
        let mut rec = Recorder::default();
        let mut scratch = ScanScratch::new_for_size(640, 480, 2048);
        let ctx = ScanContext {
            proj: &proj,
            rs: &rs,
            prelude: &prelude,
            xres: 640,
            y_start: 0,
            y_end: 480,
            anginc: 1.0,
            camera_state: &cs,
            camera_gstartz0: 0,
            camera_gstartz1: 0,
            camera_vptr_offset: 0,
            camera_seed_chunk_z: 0,
        };
        top_quadrant(&mut rec, &mut scratch, &ctx);
        // Lower bound: gscanptr advanced at least once per ray (every
        // ray contributes length+1 ≥ 1).
        assert!(scratch.gscanptr >= rec.gline_calls as usize);
    }
}
