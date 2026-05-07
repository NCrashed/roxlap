//! Per-frame orchestrator — wires the R4.1 builders into a single
//! `opticast` entry point.
//!
//! Port of the top-of-`opticast` execution order in
//! `voxlap5.c:opticast` (lines 2284..end-of-function), minus the
//! globals voxlap mutates inline:
//!
//! 1. `camera_math::derive` → per-frame f32 basis.
//! 2. `opticast_prelude::derive_prelude` → integer / fixed-point cache.
//! 3. `column_walk::camera_column_air_gap` → early-out if the camera
//!    is inside solid voxel material.
//! 4. `projection::derive_projection` → cx / cy / corner-cut quad.
//! 5. `ray_step::derive_ray_step` → per-pixel ray-step coefficients.
//! 6. Four-quadrant scan dispatch (top, right, bottom, left).
//!
//! [`OpticastSettings`] bundles the constants the four-quadrant scan
//! loops need (xres / yres / projection params / mip + scan-dist
//! controls) so the orchestrator's signature stays compact.

use rayon::prelude::*;

use crate::camera_math;
use crate::camera_math::CameraState;
use crate::column_walk;
use crate::opticast_prelude;
use crate::opticast_prelude::OpticastPrelude;
use crate::projection;
use crate::rasterizer::{Rasterizer, ScratchPool};
use crate::ray_step;
use crate::scan_loops::{
    bottom_quadrant, left_quadrant, right_quadrant, top_quadrant, ScanContext,
};
use crate::Camera;

/// Per-frame settings the orchestrator forwards to the builders. Most
/// fields map 1:1 onto a voxlap global (`vx5.anginc`, `vx5.mipscandist`,
/// `vx5.maxscandist`) or a `setcamera` argument (`dahx` / `dahy` /
/// `dahz`). `mip_levels` is voxlap's `gmipnum` — `1` for the oracle
/// scene.
///
/// `y_start..y_end` is the strip-render iteration bound (R12.3).
/// Default is the full framebuffer (`0..yres`), giving pre-R12.3
/// full-frame opticast behaviour bit-exactly. Tile / strip callers
/// set a sub-range to render only that horizontal strip — pass-1
/// gline ray casts and pass-2 hrend / vrend writes both stay
/// inside the strip's y-range. The camera projection center stays
/// in absolute screen coords; only the viewport edges shrink.
#[derive(Debug, Clone, Copy)]
pub struct OpticastSettings {
    pub xres: u32,
    pub yres: u32,
    /// First y-row this opticast call renders (inclusive). `0` for
    /// full-frame.
    pub y_start: u32,
    /// One past the last y-row (exclusive). `yres` for full-frame.
    pub y_end: u32,
    pub hx: f32,
    pub hy: f32,
    pub hz: f32,
    pub anginc: i32,
    pub mip_levels: u32,
    pub mip_scan_dist: i32,
    pub max_scan_dist: i32,
}

impl OpticastSettings {
    /// Default settings for a `width × height` framebuffer with the
    /// voxlap-oracle convention `(hx, hy, hz) = (w/2, h/2, w/2)` and
    /// `anginc = 1`, matching `tests/oracle/oracle.c`. Renders the
    /// full frame (`y_start = 0, y_end = height`).
    //
    // `width` / `height` cast to f32 is bounded by realistic screen
    // sizes (≤ 16M, well within f32's 24-bit mantissa).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn for_oracle_framebuffer(width: u32, height: u32) -> Self {
        let half_w = (width as f32) * 0.5;
        let half_h = (height as f32) * 0.5;
        Self {
            xres: width,
            yres: height,
            y_start: 0,
            y_end: height,
            hx: half_w,
            hy: half_h,
            hz: half_w,
            anginc: 1,
            mip_levels: 1,
            mip_scan_dist: 4,
            max_scan_dist: 1024,
        }
    }

    /// Restrict this settings struct to the `[y_start, y_end)`
    /// horizontal strip. Used by the per-strip parallel dispatch
    /// (R12.3.1) — each strip clones the base settings and clamps
    /// the y-range. Caller is responsible for ensuring `y_start <
    /// y_end <= yres`.
    #[must_use]
    pub fn with_y_range(mut self, y_start: u32, y_end: u32) -> Self {
        self.y_start = y_start;
        self.y_end = y_end;
        self
    }
}

/// Outcome of one [`opticast`] call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpticastOutcome {
    /// All four quadrants dispatched (some or all may have early-
    /// outed on their own geometry guards — that is normal).
    Rendered,
    /// Camera position lies in solid voxel material. Voxlap returns
    /// from `opticast` early in this case (no render, screen retains
    /// previous contents — the host can pre-fill with sky).
    SkippedCameraInSolid,
    /// S1.2: camera position lies outside the column grid in X or Y.
    /// Voxlap C cannot render this case (the per-frame
    /// `camera_column_air_gap` walk has no valid camera column);
    /// roxlap routes through [`opticast_outside`] which clips each
    /// ray against the world AABB and starts gline at the entry
    /// face. S1.2 wires the dispatch but the body is still a stub —
    /// S1.3 fills in the per-ray walk.
    OutsideCamera,
}

/// S1.2: is the camera's integer position outside the column grid's
/// XY footprint? `li_pos` comes from
/// [`opticast_prelude::OpticastPrelude::li_pos`] — voxlap floors the
/// f32 camera pos to i32 to derive `column_index`, and that index
/// silently wraps when either component is `>= vsid` or `< 0`. This
/// helper is the precondition the dispatch fork in [`opticast`]
/// checks.
///
/// Z is intentionally not checked: voxlap's existing sky-walk-down
/// already handles "camera above the world top," and "camera below
/// `z = 0`" is not in S1's scope.
#[must_use]
#[inline]
fn camera_outside_xy(li_pos: &[i32; 3], vsid: u32) -> bool {
    #[allow(clippy::cast_possible_wrap)]
    let vsid_signed = vsid as i32;
    li_pos[0] < 0 || li_pos[1] < 0 || li_pos[0] >= vsid_signed || li_pos[1] >= vsid_signed
}

/// Drive one frame of opticast. The caller supplies:
/// - `camera`: pose to render from.
/// - `settings`: framebuffer + projection + scan-dist constants.
/// - `vsid`: world dimension (square map).
/// - `slab_buf` + `column_offsets`: world-level voxel data —
///   `slab_buf` is the flat byte buffer holding all columns'
///   slab lists concatenated; `column_offsets[i]` is the byte
///   offset where column `i`'s slabs start.
///   `column_offsets.len()` must equal `vsid * vsid + 1` (the
///   final entry is `slab_buf.len()`, so column slices are
///   `slab_buf[column_offsets[i]..column_offsets[i + 1]]`).
///
/// Whatever real or stub [`Rasterizer`] is plugged in receives the
/// `gline` / `hrend` / `vrend` calls the four-quadrant scan loops
/// produce; the [`ScratchPool`]'s slots accumulate the radar /
/// angstart / lastx / uurend buffers between those calls.
///
/// Threading dial lives on the pool:
/// - `pool.n_threads() == 1` → sequential. The four quadrants run
///   on the calling thread against `pool.slot_mut(0)`. Pre-R12
///   shape; the byte-stable golden baseline.
/// - `pool.n_threads() >= 2` → R12.3.1 per-strip parallel. The
///   framebuffer's y-range splits into N horizontal strips of
///   `~yres/N` rows each. Each strip runs its own opticast pass
///   (4 quadrants) against its own slot from
///   `pool.slots_mut_slice()`, with [`OpticastSettings::y_start`] /
///   `y_end` clipped to the strip. Strips run via
///   `rayon::par_iter_mut`, each with a cloned rasterizer (raw
///   fb / zb pointers shared, strip-disjoint row writes).
///
/// **Byte-stability caveat** (R12.3.1): per-strip rendering produces
/// different pixel hashes than single-strip. Voxlap's screen-line
/// interpolation in `gline` parameterises rays by viewport-y bounds
/// (via the corner-cut quad's grd / dxy); strips have narrower
/// y-bounds, so the per-strip ray fan discretises slightly
/// differently. The image is geometrically valid — each pixel still
/// samples a camera-correct ray — but the 1/N strip discretisation
/// drifts by a fraction of a voxel from the full-frame
/// discretisation. For CI, oracle goldens are frozen at
/// `--threads 1` (single strip = full frame, byte-stable).
///
/// `R: Clone + Send + Sync` is required by the parallel branch even
/// when it doesn't fire — keeping the bound consistent across both
/// paths means the generic body monomorphizes once. The `Sync` bound
/// shows up because `rayon::par_iter_mut`'s closure shares `&R` (the
/// strip-cloning template) across worker threads. Test rasterizers
/// (`Counts`, `RecordingRasterizer`) derive Clone + auto-Send/Sync
/// so they satisfy the bound at no runtime cost.
//
// Sign convention: voxlap's opticast forwards everything as-is from
// the static state; here it's all explicit parameters. The clippy
// arg-count lint is allowed because each parameter pulls its weight
// (a struct-of-args variant just renames the same data). The
// xres / yres → i32 casts are bounded by realistic framebuffer
// dimensions and won't wrap.
#[allow(clippy::too_many_arguments, clippy::cast_possible_wrap)]
#[must_use]
pub fn opticast<R: Rasterizer + Clone + Send + Sync>(
    rasterizer: &mut R,
    pool: &mut ScratchPool,
    camera: &Camera,
    settings: &OpticastSettings,
    vsid: u32,
    slab_buf: &[u8],
    column_offsets: &[u32],
) -> OpticastOutcome {
    let cs = camera_math::derive(
        camera,
        settings.xres,
        settings.yres,
        settings.hx,
        settings.hy,
        settings.hz,
    );

    let prelude = opticast_prelude::derive_prelude(
        &cs,
        vsid,
        settings.mip_levels,
        settings.mip_scan_dist,
        settings.max_scan_dist,
    );

    // S1.Z: outside-XY camera path. When the camera sits past
    // [0, vsid)² in X or Y, `prelude.column_index` is wrapped junk
    // (`li_pos.y * vsid + li_pos.x` with `as u32` cast aliases
    // negative values to large in-range integers). Skip the
    // per-frame `camera_column_slice + camera_column_air_gap` walk
    // and synthesise a full-air-gap ScanContext — the column-walk
    // DDA will then traverse OOB columns as empty until it crosses
    // into the world (handled per-step in grouscan.rs's
    // `phase_after_delete_kept_presync`).
    //
    // For inside-XY camera, the original gstartv walk runs as
    // before; the negative-index path is a no-op.
    let (gstartz0, gstartz1, camera_vptr_offset) = if prelude.in_bounds_xy {
        let camera_column = camera_column_slice(slab_buf, column_offsets, prelude.column_index);
        let Some(camera_column_data) = camera_column else {
            return OpticastOutcome::SkippedCameraInSolid;
        };
        let Some(triple) =
            column_walk::camera_column_air_gap(camera_column_data, prelude.li_pos[2])
        else {
            return OpticastOutcome::SkippedCameraInSolid;
        };
        triple
    } else {
        // Camera sits in the void outside the column grid. There is
        // no camera column; synthesise the "full air gap" answer
        // (z0 = 0, z1 = MAXZDIM, vptr_offset = 0) so the per-frame
        // setup proceeds. Per-scanline cf-seed semantics are
        // unchanged because gline reads gstartz0/z1 only as the
        // initial cf z bounds — nothing's solid above the void
        // camera so the full-z-range answer is geometrically
        // correct.
        // z bounds match what camera_column_air_gap returns for
        // "camera above all slabs" of a typical column: (0,
        // first_z1, 0). 255 is voxlap's MAXZDIM - 1 — the "below
        // any solid" sentinel.
        (0, 255, 0)
    };

    // Per-frame setup hook needs a `ScanContext` with cy / camera
    // state populated; build a "setup-only" projection over the
    // FULL frame y-range so frame_setup sees the same projection
    // center the strips inherit.
    let setup_proj = projection::derive_projection_with_y_range(
        &cs,
        settings.xres,
        settings.yres,
        settings.y_start,
        settings.y_end,
        settings.hx,
        settings.hy,
        settings.hz,
        settings.anginc,
    );
    let setup_rs = ray_step::derive_ray_step(&cs, setup_proj.cx, setup_proj.cy, settings.hz);
    let setup_ctx = ScanContext {
        proj: &setup_proj,
        rs: &setup_rs,
        prelude: &prelude,
        xres: settings.xres as i32,
        y_start: settings.y_start as i32,
        y_end: settings.y_end as i32,
        anginc: settings.anginc,
        camera_state: &cs,
        camera_gstartz0: gstartz0,
        camera_gstartz1: gstartz1,
        camera_vptr_offset,
    };

    // Per-frame setup hook — concrete rasterizers (R4.2) cache the
    // bits of CameraState / RayStep / OpticastPrelude they need for
    // the per-pixel math. Runs on the calling thread before any
    // parallel fan-out so subsequent clones inherit the populated
    // FrameCache. Stub rasterizers ignore via the trait's default
    // no-op.
    rasterizer.frame_setup(&setup_ctx);

    let n_strips = pool.n_threads();
    if n_strips <= 1 {
        // Sequential — slot 0, full settings. Byte-stable golden
        // baseline.
        let scratch = pool.slot_mut(0);
        top_quadrant(rasterizer, scratch, &setup_ctx);
        right_quadrant(rasterizer, scratch, &setup_ctx);
        bottom_quadrant(rasterizer, scratch, &setup_ctx);
        left_quadrant(rasterizer, scratch, &setup_ctx);
    } else {
        // Per-strip parallel (R12.3.1). Slice the y-range into N
        // strips of `~strip_height` rows each. Each strip runs its
        // own opticast against its own slot. See
        // `run_strip_parallel` for the per-strip body.
        run_strip_parallel(
            rasterizer,
            pool,
            settings,
            &cs,
            &prelude,
            gstartz0,
            gstartz1,
            camera_vptr_offset,
        );
    }

    OpticastOutcome::Rendered
}

/// Per-strip parallel body. Splits `[settings.y_start, settings.y_end)`
/// into `pool.n_threads()` contiguous row strips and runs one
/// opticast pass per strip via `rayon::par_iter_mut`. Each strip:
///
/// * clones `rasterizer` (raw fb / zb pointers in the
///   [`crate::scalar_rasterizer::RasterTarget`] are `Copy`; the
///   strip-disjoint row writes make the aliasing safe);
/// * gets exclusive `&mut ScanScratch` access to one pool slot via
///   `par_iter_mut`'s borrow split;
/// * derives its own [`crate::projection::ProjectionRect`] with
///   wy0 / wy1 clipped to the strip — `gline` and the four scan
///   loops then auto-clip ray casts and pixel writes;
/// * runs the four quadrants over its strip.
//
// Per-strip projection re-derivation is fast (a handful of f32
// ops). prelude + camera_state are shared `&` borrows — Sync, no
// per-strip allocation.
#[allow(clippy::too_many_arguments)]
fn run_strip_parallel<R: Rasterizer + Clone + Send + Sync>(
    rasterizer: &mut R,
    pool: &mut ScratchPool,
    settings: &OpticastSettings,
    cs: &CameraState,
    prelude: &OpticastPrelude,
    gstartz0: i32,
    gstartz1: i32,
    camera_vptr_offset: usize,
) {
    let n_strips = pool.n_threads();
    let y_start_total = settings.y_start;
    let y_end_total = settings.y_end;
    let span = y_end_total.saturating_sub(y_start_total);
    if span == 0 {
        return;
    }

    // `(span + n - 1) / n` → ceiling-divide so trailing rows aren't
    // dropped on non-divisible splits. Last strip may be smaller.
    #[allow(clippy::cast_possible_truncation)]
    let strip_height: u32 = ((span + n_strips as u32 - 1) / n_strips as u32).max(1);

    // Capture borrowed copies for the parallel closure — closure
    // needs `move` for the cloned rasterizer + slot, but the
    // shared `&` borrows below are Send + Sync via auto-impl.
    let rasterizer_template = &*rasterizer;
    let cs_ref: &CameraState = cs;
    let prelude_ref: &OpticastPrelude = prelude;
    let settings_ref: &OpticastSettings = settings;

    let strip_body = |(i, scratch): (usize, &mut crate::rasterizer::ScanScratch)| {
        #[allow(clippy::cast_possible_truncation)]
        let strip_y_start = y_start_total.saturating_add((i as u32).saturating_mul(strip_height));
        let strip_y_end = strip_y_start.saturating_add(strip_height).min(y_end_total);
        if strip_y_start >= strip_y_end {
            // Tail strip past the actual y-range — happens when
            // n_strips > span (e.g., 16 strips on a 12-row span).
            return;
        }

        let strip_proj = projection::derive_projection_with_y_range(
            cs_ref,
            settings_ref.xres,
            settings_ref.yres,
            strip_y_start,
            strip_y_end,
            settings_ref.hx,
            settings_ref.hy,
            settings_ref.hz,
            settings_ref.anginc,
        );
        let strip_rs =
            ray_step::derive_ray_step(cs_ref, strip_proj.cx, strip_proj.cy, settings_ref.hz);
        #[allow(clippy::cast_possible_wrap)]
        let strip_ctx = ScanContext {
            proj: &strip_proj,
            rs: &strip_rs,
            prelude: prelude_ref,
            xres: settings_ref.xres as i32,
            y_start: strip_y_start as i32,
            y_end: strip_y_end as i32,
            anginc: settings_ref.anginc,
            camera_state: cs_ref,
            camera_gstartz0: gstartz0,
            camera_gstartz1: gstartz1,
            camera_vptr_offset,
        };

        let mut strip_rasterizer: R = rasterizer_template.clone();
        top_quadrant(&mut strip_rasterizer, scratch, &strip_ctx);
        right_quadrant(&mut strip_rasterizer, scratch, &strip_ctx);
        bottom_quadrant(&mut strip_rasterizer, scratch, &strip_ctx);
        left_quadrant(&mut strip_rasterizer, scratch, &strip_ctx);
    };

    pool.slots_mut_slice()
        .par_iter_mut()
        .enumerate()
        .for_each(strip_body);
}

/// S1.3: outside-camera render entry point. Voxlap C bails before
/// reaching the per-frame projection setup when the camera is
/// outside the column grid; roxlap reroutes here instead.
///
/// Strategy: install
/// [`Rasterizer::set_outside_camera_active`]`(true)` on the
/// rasterizer, run the standard four-quadrant scan loops, then
/// clear the flag. While active, [`Rasterizer::gline`] auto-clips
/// each scanline's ray against the world XY AABB and synthesises
/// the per-scanline `OutsideRayContext` (entry column, fractional
/// XY, air-gap z-bounds). Rays that miss the AABB take a sky-fill
/// fast path; rays that hit run gline as if the camera were at
/// the entry point.
///
/// First cut is sequential-only — the per-strip parallel branch
/// from the inside path is gated to `pool.n_threads() == 1`. The
/// outside flag is set on the rasterizer before clones happen, so
/// each clone inherits it; a follow-up can lift the parallel guard
/// once we've verified the per-strip path doesn't shake the auto-
/// clip output.
///
/// `camera_gstartz0` / `camera_gstartz1` / `camera_vptr_offset` in
/// the synthesised `ScanContext` are placeholders (0). The inside
/// path uses them as the cf-seed defaults; outside mode supplies
/// per-scanline overrides via the rasterizer's auto-clip path so
/// the placeholders are never read for actual rays.
#[allow(clippy::too_many_arguments)]
fn opticast_outside<R: Rasterizer + Clone + Send + Sync>(
    rasterizer: &mut R,
    pool: &mut ScratchPool,
    cs: &CameraState,
    prelude: &OpticastPrelude,
    settings: &OpticastSettings,
    _vsid: u32,
    _slab_buf: &[u8],
    _column_offsets: &[u32],
) -> OpticastOutcome {
    let setup_proj = projection::derive_projection_with_y_range(
        cs,
        settings.xres,
        settings.yres,
        settings.y_start,
        settings.y_end,
        settings.hx,
        settings.hy,
        settings.hz,
        settings.anginc,
    );
    let setup_rs = ray_step::derive_ray_step(cs, setup_proj.cx, setup_proj.cy, settings.hz);
    #[allow(clippy::cast_possible_wrap)]
    let setup_ctx = ScanContext {
        proj: &setup_proj,
        rs: &setup_rs,
        prelude,
        xres: settings.xres as i32,
        y_start: settings.y_start as i32,
        y_end: settings.y_end as i32,
        anginc: settings.anginc,
        camera_state: cs,
        camera_gstartz0: 0,
        camera_gstartz1: 0,
        camera_vptr_offset: 0,
    };

    rasterizer.set_outside_camera_active(true);
    rasterizer.frame_setup(&setup_ctx);

    // Sequential only for S1 first-cut. The per-strip parallel
    // branch clones the rasterizer; each clone inherits the active
    // flag, but we haven't yet verified that the per-strip y-range
    // restriction doesn't interact badly with the auto-clip path.
    let scratch = pool.slot_mut(0);
    top_quadrant(rasterizer, scratch, &setup_ctx);
    right_quadrant(rasterizer, scratch, &setup_ctx);
    bottom_quadrant(rasterizer, scratch, &setup_ctx);
    left_quadrant(rasterizer, scratch, &setup_ctx);

    rasterizer.set_outside_camera_active(false);

    OpticastOutcome::OutsideCamera
}

/// Slice `slab_buf` at column `idx`'s byte range (per the
/// `column_offsets` table). Returns `None` if the index is out of
/// range or the offsets are malformed (non-monotonic, past the
/// buffer end). Treated as camera-in-solid by the caller.
pub(crate) fn camera_column_slice<'a>(
    slab_buf: &'a [u8],
    column_offsets: &[u32],
    idx: u32,
) -> Option<&'a [u8]> {
    let i = idx as usize;
    if i >= column_offsets.len() {
        return None;
    }
    let start = column_offsets[i] as usize;
    if start >= slab_buf.len() {
        return None;
    }
    // Slice to end-of-buffer; the slab walker self-terminates on
    // `nextptr == 0`. Using `column_offsets[i + 1]` as the end was
    // wrong post-edit (voxalloc scatters columns across vbuf, so
    // adjacent table indices are no longer adjacent in memory).
    Some(&slab_buf[start..])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rasterizer::ScanScratch;

    /// Recording rasterizer that counts the three callback kinds.
    /// `Clone` so the rasterizer satisfies opticast's `R: Clone +
    /// Send` bound (R12.2.1).
    #[derive(Debug, Default, Clone)]
    struct Counts {
        gline: u32,
        hrend: u32,
        vrend: u32,
    }

    impl Rasterizer for Counts {
        fn gline(&mut self, _: &mut ScanScratch, _: u32, _: f32, _: f32, _: f32, _: f32) {
            self.gline += 1;
        }
        fn hrend(&mut self, _: &mut ScanScratch, _: i32, _: i32, _: i32, _: i32, _: i32, _: i32) {
            self.hrend += 1;
        }
        fn vrend(&mut self, _: &mut ScanScratch, _: i32, _: i32, _: i32, _: i32, _: i32) {
            self.vrend += 1;
        }
    }

    /// Single solid slab at z = 200..254. cz < 200 → air gap (0, 200).
    /// cz inside [200, 254] → in solid → opticast skips.
    fn solid_slab_z200_to_254() -> Vec<u8> {
        // Header [nextptr=0, z1=200, z1c=254, dummy=0]. The walker
        // doesn't read past the header, so no colour bytes needed.
        vec![0, 200, 254, 0]
    }

    fn looking_down_camera() -> Camera {
        Camera {
            pos: [1024.0, 1024.0, 128.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 1.0, 0.0],
            forward: [0.0, 0.0, 1.0],
        }
    }

    /// Build a `(slab_buf, column_offsets)` pair where one column —
    /// `camera_column_index` — holds `column_data`'s bytes and
    /// every other column is empty. Lets opticast tests target the
    /// camera column without allocating per-column slab data for
    /// the full `vsid²` grid.
    #[allow(clippy::cast_possible_truncation)]
    fn synthetic_world_with_camera_column(
        column_data: &[u8],
        camera_column_index: u32,
        vsid: u32,
    ) -> (Vec<u8>, Vec<u32>) {
        let vsid_sq = (vsid as usize) * (vsid as usize);
        let len_u32 = column_data.len() as u32;
        let cam_idx = camera_column_index as usize;
        let mut column_offsets = vec![0u32; vsid_sq + 1];
        for offset in &mut column_offsets[(cam_idx + 1)..] {
            *offset = len_u32;
        }
        (column_data.to_vec(), column_offsets)
    }

    /// `looking_down_camera` at pos = (1024, 1024) with vsid = 2048
    /// → `column_index` = 1024 * 2048 + 1024 = `2_099_200`.
    const LOOKING_DOWN_COL_INDEX: u32 = 1024 * 2048 + 1024;

    #[test]
    fn opticast_dispatches_all_four_quadrants() {
        let cam = looking_down_camera();
        let settings = OpticastSettings::for_oracle_framebuffer(640, 480);
        let mut counts = Counts::default();
        let mut pool = ScratchPool::new(640, 480, 2048);
        let (slab_buf, column_offsets) = synthetic_world_with_camera_column(
            &solid_slab_z200_to_254(),
            LOOKING_DOWN_COL_INDEX,
            2048,
        );

        let outcome = opticast(
            &mut counts,
            &mut pool,
            &cam,
            &settings,
            2048,
            &slab_buf,
            &column_offsets,
        );

        assert_eq!(outcome, OpticastOutcome::Rendered);
        // Looking-down camera: each quadrant fires. gline counts ≈
        // 2 × x-fan-width + 2 × y-fan-width; positive total.
        assert!(counts.gline > 0, "expected ≥ 1 gline call");
        // Top + bottom quadrants both produce hrend; right + left
        // produce vrend.
        assert!(counts.hrend > 0, "expected ≥ 1 hrend call");
        assert!(counts.vrend > 0, "expected ≥ 1 vrend call");
    }

    #[test]
    fn opticast_skips_when_camera_in_solid() {
        // Place the camera inside the solid slab z = 200..254 by
        // moving pos.z to 220.
        let mut cam = looking_down_camera();
        cam.pos[2] = 220.0;
        let settings = OpticastSettings::for_oracle_framebuffer(640, 480);
        let mut counts = Counts::default();
        let mut pool = ScratchPool::new(640, 480, 2048);
        let (slab_buf, column_offsets) = synthetic_world_with_camera_column(
            &solid_slab_z200_to_254(),
            LOOKING_DOWN_COL_INDEX,
            2048,
        );

        let outcome = opticast(
            &mut counts,
            &mut pool,
            &cam,
            &settings,
            2048,
            &slab_buf,
            &column_offsets,
        );

        assert_eq!(outcome, OpticastOutcome::SkippedCameraInSolid);
        assert_eq!(counts.gline, 0);
        assert_eq!(counts.hrend, 0);
        assert_eq!(counts.vrend, 0);
    }

    #[test]
    fn for_oracle_framebuffer_defaults() {
        let s = OpticastSettings::for_oracle_framebuffer(640, 480);
        assert_eq!(s.xres, 640);
        assert_eq!(s.yres, 480);
        // hx / hy / hz: voxlap-oracle convention.
        assert!((s.hx - 320.0).abs() < f32::EPSILON);
        assert!((s.hy - 240.0).abs() < f32::EPSILON);
        assert!((s.hz - 320.0).abs() < f32::EPSILON);
        assert_eq!(s.anginc, 1);
        assert_eq!(s.mip_levels, 1);
        assert_eq!(s.max_scan_dist, 1024);
    }
}
