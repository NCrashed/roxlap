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

use crate::camera_math;
use crate::column_walk;
use crate::opticast_prelude;
use crate::projection;
use crate::rasterizer::{Rasterizer, ScanScratch};
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
#[derive(Debug, Clone, Copy)]
pub struct OpticastSettings {
    pub xres: u32,
    pub yres: u32,
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
    /// `anginc = 1`, matching `tests/oracle/oracle.c`.
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
            hx: half_w,
            hy: half_h,
            hz: half_w,
            anginc: 1,
            mip_levels: 1,
            mip_scan_dist: 4,
            max_scan_dist: 1024,
        }
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
/// produce; `scratch` accumulates the radar / angstart / lastx /
/// uurend buffers between those calls.
//
// Sign convention: voxlap's opticast forwards everything as-is from
// the static state; here it's all explicit parameters. The clippy
// arg-count lint is allowed because each parameter pulls its weight
// (a struct-of-args variant just renames the same data). The
// xres / yres → i32 casts are bounded by realistic framebuffer
// dimensions and won't wrap.
#[allow(clippy::too_many_arguments, clippy::cast_possible_wrap)]
pub fn opticast<R: Rasterizer>(
    rasterizer: &mut R,
    scratch: &mut ScanScratch,
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

    // gstartv walk — early-out if the camera is inside solid voxel
    // material. Slice `slab_buf` at the camera column's range
    // (computed by the prelude as `column_index = li_pos.y * vsid +
    // li_pos.x`); a malformed or out-of-bounds offset table is
    // treated as "camera in solid" so we early-out cleanly.
    let camera_column = camera_column_slice(slab_buf, column_offsets, prelude.column_index);
    let Some(camera_column_data) = camera_column else {
        return OpticastOutcome::SkippedCameraInSolid;
    };
    if column_walk::camera_column_air_gap(camera_column_data, prelude.li_pos[2]).is_none() {
        return OpticastOutcome::SkippedCameraInSolid;
    }

    let proj = projection::derive_projection(
        &cs,
        settings.xres,
        settings.yres,
        settings.hx,
        settings.hy,
        settings.hz,
        settings.anginc,
    );
    let rs = ray_step::derive_ray_step(&cs, proj.cx, proj.cy, settings.hz);

    let ctx = ScanContext {
        proj: &proj,
        rs: &rs,
        prelude: &prelude,
        xres: settings.xres as i32,
        yres: settings.yres as i32,
        anginc: settings.anginc,
    };

    // Per-frame setup hook — concrete rasterizers (R4.2) cache the
    // bits of CameraState / RayStep / OpticastPrelude they need for
    // the per-pixel math. Stub rasterizers ignore via the trait's
    // default no-op.
    rasterizer.frame_setup(&ctx);

    top_quadrant(rasterizer, scratch, &ctx);
    right_quadrant(rasterizer, scratch, &ctx);
    bottom_quadrant(rasterizer, scratch, &ctx);
    left_quadrant(rasterizer, scratch, &ctx);

    OpticastOutcome::Rendered
}

/// Slice `slab_buf` at column `idx`'s byte range (per the
/// `column_offsets` table). Returns `None` if the index is out of
/// range or the offsets are malformed (non-monotonic, past the
/// buffer end). Treated as camera-in-solid by the caller.
fn camera_column_slice<'a>(
    slab_buf: &'a [u8],
    column_offsets: &[u32],
    idx: u32,
) -> Option<&'a [u8]> {
    let i = idx as usize;
    if i + 1 >= column_offsets.len() {
        return None;
    }
    let start = column_offsets[i] as usize;
    let end = column_offsets[i + 1] as usize;
    if start > end || end > slab_buf.len() {
        return None;
    }
    Some(&slab_buf[start..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Recording rasterizer that counts the three callback kinds.
    #[derive(Debug, Default)]
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
        let mut scratch = ScanScratch::new_for_size(640, 480, 2048);
        let (slab_buf, column_offsets) = synthetic_world_with_camera_column(
            &solid_slab_z200_to_254(),
            LOOKING_DOWN_COL_INDEX,
            2048,
        );

        let outcome = opticast(
            &mut counts,
            &mut scratch,
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
        let mut scratch = ScanScratch::new_for_size(640, 480, 2048);
        let (slab_buf, column_offsets) = synthetic_world_with_camera_column(
            &solid_slab_z200_to_254(),
            LOOKING_DOWN_COL_INDEX,
            2048,
        );

        let outcome = opticast(
            &mut counts,
            &mut scratch,
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
