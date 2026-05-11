//! Per-frame opticast prelude — the integer / fixed-point state cache
//! voxlap derives at the top of `opticast` (`voxlap5.c:2284..2303`)
//! before the column-scan loops run.
//!
//! Inputs:
//! - [`CameraState`] from `camera_math::derive`.
//! - `vsid`: world dimension (square map).
//! - `mip_levels`: number of mip levels, voxlap calls this `gmipnum`.
//!   Defaults to 1 in the oracle scene.
//! - `mip_scan_dist` / `max_scan_dist`: voxlap's `vx5.mipscandist` /
//!   `vx5.maxscandist` controls.
//!
//! Outputs (`OpticastPrelude`):
//! - `forward_z_sign`: `giforzsgn` — sign of `forward.z` as ±1.
//! - `li_pos`: integer-truncated camera position (`glipos`).
//! - `column_index`: linear index `li_pos.y * vsid + li_pos.x`. The
//!   voxlap C carries this as a pointer (`gpixy`) into the
//!   `sptr[VSID*VSID]` column-pointer table; in Rust we keep it as an
//!   integer index because the column data lives in
//!   `roxlap-formats::vxl::Vxl::column_offset / data` and there is no
//!   pointer-array equivalent to step through.
//! - `pos_xfrac` / `pos_yfrac`: `[1 - frac, frac]` per axis. The
//!   grouscan ray-stepper uses these as bilinear weights between
//!   neighbouring voxel columns.
//! - `pos_z`: camera z scaled into voxlap's fixed-point grid
//!   (`gposz`). `lrintf(pos.z * PREC - 0.5)` — `PREC = 256 * 4096`.
//! - `y_lookup`: `gylookup` mip table; per level `j`, holds
//!   `(512 >> j) + 4` entries that the rasterizer indexes by depth
//!   to convert world z into screen-row offsets.
//! - `x_mip`: `gxmip = max(mip_scan_dist, 4) * PREC`.
//! - `max_scan_dist`: `gmaxscandist`, clamped to `[1, 4095]` then
//!   PREC-scaled.

use crate::camera_math::CameraState;
use crate::fixed::ftol;

/// Voxlap's fixed-point Q12.20-style scale factor (`PREC` in
/// `voxlap5.c:13`). `gposz` is `pos.z * PREC` rounded; `gylookup`
/// entries are computed from `(gposz >> j) - i * PREC`.
pub const PREC: i32 = 256 * 4096;

#[derive(Debug, Clone)]
pub struct OpticastPrelude {
    pub forward_z_sign: i32,
    pub li_pos: [i32; 3],
    pub column_index: u32,
    pub pos_xfrac: [f32; 2],
    pub pos_yfrac: [f32; 2],
    pub pos_z: i32,
    pub y_lookup: Vec<i32>,
    pub x_mip: i32,
    pub max_scan_dist: i32,
    /// S1.Z: signed integer column coordinates of the camera. For
    /// in-bounds camera, equal to `(li_pos[0], li_pos[1])`. For
    /// outside-XY camera (camera position past `[0, vsid)`), these
    /// can be negative or `>= vsid` and the grouscan walk traverses
    /// out-of-bounds columns as empty until it crosses into the
    /// world. `column_index` cannot be used for OOB classification
    /// because `as u32` wraps negative values to large positive
    /// integers that may alias to in-range columns.
    pub cx: i32,
    pub cy: i32,
    /// S1.Z: `cx, cy ∈ [0, vsid)`. Inside-camera path uses this to
    /// gate the existing `camera_column_air_gap` precondition (which
    /// requires real column data); outside path skips that and
    /// synthesises full-air-gap placeholders.
    pub in_bounds_xy: bool,
    /// S4B.1: chunk index the camera sits in. For single-chunk grids
    /// (today's only shape) always `[0, 0, 0]`. S4B.2 grows this to
    /// `floor_div(cx, chunk_xy)` / `floor_div(cy, chunk_xy)` /
    /// `floor_div(li_pos.z, chunk_z)` so the XY-cross-chunk DDA can
    /// hand off when (cx, cy) crosses a chunk boundary.
    pub camera_chunk_idx: [i32; 3],
    /// S4B.1: camera coordinates relative to its chunk's origin.
    /// Single-chunk grid: equal to `li_pos`. Multi-chunk:
    /// `li_pos[i] - camera_chunk_idx[i] * chunk_size`. Drives the
    /// chunk-local column-table lookup `chunk.column_offsets[
    /// camera_local_xyz.y * chunk_vsid + camera_local_xyz.x]`.
    pub camera_local_xyz: [i32; 3],
}

/// Derive the per-frame [`OpticastPrelude`].
///
/// Mirrors the integer / fixed-point setup at the top of
/// `voxlap5.c:opticast` (lines ~2289..2303).
//
// clippy::cast_possible_truncation fires on `f32 as i32` for `li_pos`
// and `cast_precision_loss` fires on `i32 as f32` for the fractional-
// position math; both casts are intentional and bounded — voxel
// positions are within `0..VSID` so they fit f32 exactly.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    // pos_xfrac / pos_yfrac differ by one letter; that one letter is
    // load-bearing (matches voxlap's gposxfrac / gposyfrac names) so
    // suppressing the similar_names lint is the right call here.
    clippy::similar_names
)]
#[must_use]
pub fn derive_prelude(
    camera_state: &CameraState,
    vsid: u32,
    mip_levels: u32,
    mip_scan_dist: i32,
    max_scan_dist: i32,
) -> OpticastPrelude {
    let forward_z_sign = if camera_state.forward[2] < 0.0 { -1 } else { 1 };

    // Floor (not `as i32` truncate) — voxlap C uses lrintf which
    // for integer-position cameras (the 12 oracle poses) rounds to
    // the same value, but for fractional-position OOB cameras with
    // negative coords (e.g. cy=-18.29), `as i32` truncates to -18
    // while floor produces -19. Voxlap's pos_xfrac/pos_yfrac
    // convention requires the fractional part to be in [0, 1) —
    // truncation puts it in (-1, 0] for negative positions, which
    // makes gline's gpz[lane] init negative and the per-column DDA
    // skips ahead by 1 voxel-col on the first y-step. Visible as
    // "world doesn't render at all" when camera flies into the
    // -Y OOB strip with a fractional Y position.
    let li_pos = [
        camera_state.pos[0].floor() as i32,
        camera_state.pos[1].floor() as i32,
        camera_state.pos[2].floor() as i32,
    ];

    // S1.Z: signed copies for the negative-index outside-camera walk.
    // li_pos[0] / [1] can legitimately be negative when the camera
    // sits past [0, vsid) in X or Y; the cast-to-u32 below would
    // alias such values to wrap-around in-range column indices, which
    // is wrong. We carry the signed values forward separately.
    let cx = li_pos[0];
    let cy = li_pos[1];
    let vsid_signed = vsid as i32;
    let in_bounds_xy = cx >= 0 && cy >= 0 && cx < vsid_signed && cy < vsid_signed;

    // For in-bounds cameras, this is the canonical voxlap
    // `gpixy = ipy*VSID + ipx` index. For OOB cameras the value is
    // garbage (u32 wrap) but is never read — every read site gates
    // on `in_bounds_xy` first. `wrapping_*` makes the intent
    // explicit and avoids debug-mode overflow panics for cameras
    // far outside the world (large negative `li_pos` casts to
    // u32::MAX-ish values whose product with `vsid` exceeds `u32::MAX`).
    let column_index = (li_pos[1] as u32)
        .wrapping_mul(vsid)
        .wrapping_add(li_pos[0] as u32);

    let xfrac1 = camera_state.pos[0] - li_pos[0] as f32;
    let yfrac1 = camera_state.pos[1] - li_pos[1] as f32;
    let pos_xfrac = [1.0 - xfrac1, xfrac1];
    let pos_yfrac = [1.0 - yfrac1, yfrac1];

    // gposz = ftol(gipos.z * PREC - 0.5). Voxlap's ftol wraps on
    // overflow; for camera positions with pos.z above ~2048 (PREC
    // = 2²⁰) the product exceeds i32::MAX, so we go through the
    // wrap-not-saturate `ftol` helper rather than a bare `as i32`.
    let pos_z = ftol(camera_state.pos[2] * PREC as f32 - 0.5);

    // gylookup: per mip level j, fill (512 >> j) + 4 entries computed
    // from ((pos_z >> j) - i * PREC) >> (16 - j), masked to 16 bits.
    let mut y_lookup = Vec::new();
    for j in 0..mip_levels {
        let count = (512u32 >> j) + 4;
        let pz_shifted = pos_z >> j;
        let shift = 16i32 - j as i32;
        for i in 0..count {
            let val = ((pz_shifted - (i as i32) * PREC) >> shift) & 0xFFFF;
            y_lookup.push(val);
        }
    }

    // x_mip and max_scan_dist multiply by PREC (= 2^20). At the
    // upper clamp of 4095 the multiplication overflows i32 silently
    // in voxlap C (4095 * 2^20 = 4_293_918_720 > i32::MAX). Rust's
    // `*` panics in debug; mirror C's de-facto wrap with wrapping_mul.
    let x_mip = mip_scan_dist.max(4).wrapping_mul(PREC);
    let max_scan_dist_clamped = max_scan_dist.clamp(1, 4095).wrapping_mul(PREC);

    // S4B.1 scaffold: today's GridView is a single chunk, so the
    // camera is always in chunk (0, 0, 0) and its local position
    // equals li_pos. S4B.2 grows this into a real chunk-grid lookup;
    // for now these fields just give downstream code the shape it
    // will eventually consume.
    let camera_chunk_idx = [0, 0, 0];
    let camera_local_xyz = li_pos;

    OpticastPrelude {
        forward_z_sign,
        li_pos,
        column_index,
        pos_xfrac,
        pos_yfrac,
        pos_z,
        y_lookup,
        x_mip,
        max_scan_dist: max_scan_dist_clamped,
        cx,
        cy,
        in_bounds_xy,
        camera_chunk_idx,
        camera_local_xyz,
    }
}

/// S4B.2.c.2: refine [`OpticastPrelude::camera_chunk_idx`] and
/// [`OpticastPrelude::camera_local_xyz`] using the per-chunk
/// dimension carried on the [`crate::grid_view::GridView`].
///
/// [`derive_prelude`] populates the chunk-aware fields with the
/// single-chunk placeholder `[0, 0, 0]` / `li_pos`; callers with
/// access to a `GridView` chain this on so multi-chunk grids see
/// the right `(chunk_idx, in_chunk_xy)` split. Single-chunk callers
/// (`chunk_size_xy == vsid`) get the same numerical values as the
/// placeholder for in-bounds cameras — `li_pos.div_euclid(vsid)`
/// is `0` for any `li_pos ∈ [0, vsid)`, and
/// `li_pos.rem_euclid(vsid) == li_pos`.
///
/// OOB-XY cameras get non-trivial `camera_chunk_idx` values (the
/// chunk past the grid edge they're floating over), but
/// `prelude.in_bounds_xy = false` gates downstream code so the
/// new field values aren't dereferenced — the OOB-XY bedrock seed
/// synthesis in [`crate::column_walk::camera_chunk_air_gap`] short-
/// circuits first.
//
// The z axis stays at the single-chunk placeholder until S4B.3
// adds chunk-z (`li_pos.z.div_euclid(chunk_size_z)`); chunk_z
// handoff in grouscan is also S4B.3.
#[allow(clippy::cast_possible_wrap)]
pub fn recompute_camera_chunk(prelude: &mut OpticastPrelude, chunk_size_xy: u32) {
    let chunk_size_signed = chunk_size_xy as i32;
    prelude.camera_chunk_idx = [
        prelude.li_pos[0].div_euclid(chunk_size_signed),
        prelude.li_pos[1].div_euclid(chunk_size_signed),
        0,
    ];
    prelude.camera_local_xyz = [
        prelude.li_pos[0].rem_euclid(chunk_size_signed),
        prelude.li_pos[1].rem_euclid(chunk_size_signed),
        prelude.li_pos[2],
    ];
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera_math;
    use crate::Camera;

    fn bits2(a: [f32; 2]) -> [u32; 2] {
        a.map(f32::to_bits)
    }

    /// Camera at (1024, 1024, 128) in a 2048-VSID world, looking +y
    /// forward (matching voxlaptest's oracle "north" pose).
    fn oracle_north_state() -> CameraState {
        let cam = Camera {
            pos: [1024.0, 1024.0, 128.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 0.0, 1.0],
            forward: [0.0, 1.0, 0.0],
        };
        camera_math::derive(&cam, 640, 480, 320.0, 240.0, 320.0)
    }

    #[test]
    fn integer_position_and_column_index() {
        let s = oracle_north_state();
        let p = derive_prelude(&s, 2048, 1, 4, 1024);
        assert_eq!(p.li_pos, [1024, 1024, 128]);
        assert_eq!(p.column_index, 1024 * 2048 + 1024);
        assert_eq!(p.forward_z_sign, 1); // forward.z == 0 → non-negative branch
    }

    #[test]
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    fn integer_position_pos_at_voxel_centre() {
        // Place camera at half-voxel offsets to verify fractional bits.
        let cam = Camera {
            pos: [10.25, 20.75, 30.5],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 0.0, 1.0],
            forward: [0.0, 1.0, 0.0],
        };
        let s = camera_math::derive(&cam, 640, 480, 320.0, 240.0, 320.0);
        let p = derive_prelude(&s, 2048, 1, 4, 1024);
        assert_eq!(p.li_pos, [10, 20, 30]);
        // gposxfrac[1] = 10.25 - 10 = 0.25; gposxfrac[0] = 0.75.
        assert_eq!(bits2(p.pos_xfrac), bits2([0.75, 0.25]));
        assert_eq!(bits2(p.pos_yfrac), bits2([0.25, 0.75]));
        // gposz = lrintf(30.5 * PREC - 0.5)
        //       = lrintf(31_981_567.5) — exact representation; ties-to-even rounds to 31_981_568.
        let want_pz = (30.5_f32 * PREC as f32 - 0.5).round_ties_even() as i32;
        assert_eq!(p.pos_z, want_pz);
    }

    #[test]
    fn forward_z_sign_negative_when_looking_down() {
        let cam = Camera {
            pos: [0.0, 0.0, 0.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 1.0, 0.0],
            forward: [0.0, 0.0, -1.0],
        };
        let s = camera_math::derive(&cam, 640, 480, 320.0, 240.0, 320.0);
        let p = derive_prelude(&s, 2048, 1, 4, 1024);
        assert_eq!(p.forward_z_sign, -1);
    }

    #[test]
    fn y_lookup_table_for_pos_z_at_voxel_grid() {
        // pos.z = 128.0 → pos_z = 128 * PREC = 134_217_728. PREC = 2^20,
        // so (pos_z - i*PREC) >> 16 = (128 - i) * 16 for any i.
        let s = oracle_north_state();
        let p = derive_prelude(&s, 2048, 1, 4, 1024);
        // Single mip level → (512 >> 0) + 4 = 516 entries.
        assert_eq!(p.y_lookup.len(), 516);
        // y_lookup[0] = 128 * 16 = 2048.
        assert_eq!(p.y_lookup[0], 2048);
        // y_lookup[128] = 0 * 16 = 0.
        assert_eq!(p.y_lookup[128], 0);
        // y_lookup[129] = -1 * 16 = -16, then & 0xFFFF = 0xFFF0 = 65520.
        assert_eq!(p.y_lookup[129], 65520);
    }

    #[test]
    fn y_lookup_two_mip_levels_have_correct_lengths() {
        let s = oracle_north_state();
        let p = derive_prelude(&s, 2048, 2, 4, 1024);
        // Level 0: (512 >> 0) + 4 = 516. Level 1: (512 >> 1) + 4 = 260.
        assert_eq!(p.y_lookup.len(), 516 + 260);
    }

    /// S4B.2.c.2: single-chunk-grid recompute is a no-op for the
    /// in-bounds camera (li_pos < vsid → div_euclid yields 0,
    /// rem_euclid yields li_pos). Locks in byte-identity for the
    /// 12-pose oracle goldens after recompute is wired into opticast.
    #[test]
    fn recompute_camera_chunk_single_chunk_in_bounds_is_no_op() {
        let s = oracle_north_state(); // pos.x = pos.y = 1024
        let mut p = derive_prelude(&s, 2048, 1, 4, 1024);
        let before_idx = p.camera_chunk_idx;
        let before_local = p.camera_local_xyz;
        recompute_camera_chunk(&mut p, 2048);
        assert_eq!(p.camera_chunk_idx, before_idx);
        assert_eq!(p.camera_local_xyz, before_local);
        assert_eq!(p.camera_chunk_idx, [0, 0, 0]);
        assert_eq!(p.camera_local_xyz, [1024, 1024, 128]);
    }

    /// S4B.2.c.2: multi-chunk recompute splits li_pos via
    /// `div_euclid` / `rem_euclid` against `chunk_size_xy`. Camera
    /// at (1024, 100) with `chunk_size_xy = 128` lands in chunk
    /// `[8, 0]` with local `(0, 100)`.
    #[test]
    fn recompute_camera_chunk_multi_chunk_splits_li_pos() {
        let cam = crate::Camera {
            pos: [1024.0, 100.0, 50.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 0.0, 1.0],
            forward: [0.0, 1.0, 0.0],
        };
        let s = camera_math::derive(&cam, 640, 480, 320.0, 240.0, 320.0);
        let mut p = derive_prelude(&s, 2048, 1, 4, 1024);
        recompute_camera_chunk(&mut p, 128);
        assert_eq!(p.camera_chunk_idx, [8, 0, 0]);
        assert_eq!(p.camera_local_xyz, [0, 100, 50]);
    }

    /// S4B.2.c.2: OOB-XY camera's recomputed chunk_idx can be
    /// non-trivial (negative for camera past `[0, vsid)` on the
    /// left edge). Downstream code gates on `in_bounds_xy`, so
    /// the OOB-XY values don't get dereferenced — but the test
    /// pins the arithmetic.
    #[test]
    fn recompute_camera_chunk_oob_xy_yields_negative_chunk_idx() {
        let cam = crate::Camera {
            pos: [-5.0, 100.0, 50.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 0.0, 1.0],
            forward: [0.0, 1.0, 0.0],
        };
        let s = camera_math::derive(&cam, 640, 480, 320.0, 240.0, 320.0);
        let mut p = derive_prelude(&s, 2048, 1, 4, 1024);
        recompute_camera_chunk(&mut p, 128);
        assert!(!p.in_bounds_xy);
        // -5.div_euclid(128) = -1; -5.rem_euclid(128) = 123 (positive).
        assert_eq!(p.camera_chunk_idx, [-1, 0, 0]);
        assert_eq!(p.camera_local_xyz, [123, 100, 50]);
    }

    #[test]
    fn xmip_and_maxscandist_clamping() {
        let s = oracle_north_state();
        // mip_scan_dist below the floor of 4 gets clamped up.
        let p = derive_prelude(&s, 2048, 1, 0, 99999);
        assert_eq!(p.x_mip, 4 * PREC);
        // max_scan_dist clamps to [1, 4095]. Note 4095 × PREC overflows
        // i32 in voxlap C and we mirror the wrap.
        assert_eq!(p.max_scan_dist, 4095_i32.wrapping_mul(PREC));
        let q = derive_prelude(&s, 2048, 1, 16, 0);
        assert_eq!(q.x_mip, 16 * PREC);
        // 0 clamps up to 1, then * PREC.
        assert_eq!(q.max_scan_dist, PREC);
        // Oracle's 1024 is well below the overflow threshold.
        let r = derive_prelude(&s, 2048, 1, 4, 1024);
        assert_eq!(r.max_scan_dist, 1024 * PREC);
    }
}
