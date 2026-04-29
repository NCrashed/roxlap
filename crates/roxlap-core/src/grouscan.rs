//! grouscan = `gline`'s per-ray voxel-column raycaster — port of
//! `voxlap5.c:grouscanasm_scalar` (~600 lines, voxlap5.c:11575).
//!
//! Substaged across R4.3c..f, mirroring voxlaptest's own grouscan
//! port (Stage 4.5b.2..6):
//!
//! - **R4.3c (this commit)**: cftype data model + `grouscan_run`
//!   prologue. Caches the cf[128] seed slot's state into local
//!   scalars, picks the leading raycast lane. The dispatch skeleton
//!   + draw-phase stubs land in R4.3d.
//! - **R4.3d**: drawcwall / drawfwall / drawceil / drawflor stubs +
//!   the prologue's `v == *ixy_sptr_col ? drawflor : drawceil`
//!   initial dispatch.
//! - **R4.3e**: findslab / slab-split / deletez column advance.
//! - **R4.3f**: remiporend (mip transition) + startsky.

/// One entry on grouscan's `cf` stack — voxlap's `cftype`
/// (`voxlap5.c:128`):
///
/// ```c
/// typedef struct {
///     castdat *i0, *i1;
///     int32_t z0, z1, cx0, cy0, cx1, cy1;
/// } cftype;
/// ```
///
/// `i0` / `i1` are pointers into the `radar` buffer; we mirror with
/// `isize` offsets to match the rest of the port (voxlap's pointer
/// arithmetic can produce values that land before `radar[0]`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CfType {
    pub i0: isize,
    pub i1: isize,
    pub z0: i32,
    pub z1: i32,
    pub cx0: i32,
    pub cy0: i32,
    pub cx1: i32,
    pub cy1: i32,
}

/// Length of the `cf` stack. Voxlap's asm allocates `cfasm dd 32*129`
/// (32 bytes × 129 entries); we keep the same 129-slot footprint.
pub const CF_LEN: usize = 129;

/// Index of the seed slot `gline` populates before invoking
/// `grouscan_run`. Voxlap calls this `cf[128]`.
pub const CF_SEED_INDEX: usize = 128;

use crate::rasterizer::ScanScratch;

/// Snapshot of the prologue state — the local scalars voxlap caches
/// from `cf[128]` before walking the ray. Returned by
/// [`grouscan_run`] in R4.3c so the caller can verify the prologue
/// did its work; later sub-substages will keep this state internal
/// once the dispatch loop consumes it directly.
#[derive(Debug, Clone, Copy)]
pub struct GrouscanPrologue {
    pub z0: i32,
    pub z1: i32,
    pub cx0: i32,
    pub cy0: i32,
    pub cx1: i32,
    pub cy1: i32,
    /// Leading raycast lane: `0` if the next x-grid crossing is
    /// closer than the next y-grid crossing, `1` otherwise.
    pub lane: usize,
    /// `ogx` — voxlap's "previous gx", seeded with `gpz[lane] &
    /// 0xFFFF0000` (the integer part of the leading lane's depth).
    pub ogx: i32,
    /// `gx` starts at `0`; voxlap accumulates depth into it as it
    /// walks columns.
    pub gx: i32,
    /// `ngxmax` = `min(gxmax, gxmip)` when multiple mips exist; for
    /// `gmipnum == 1` this just equals `gxmax`.
    pub ngxmax: i32,
}

/// Run grouscan for one ray. R4.3c stops after the prologue and
/// returns the snapshot; R4.3d+ adds the dispatch loop.
///
/// Side effects on `scratch`:
/// - `gpz[lane] += gdz[lane]` (voxlap's first column advance, baked
///   into the prologue).
//
// The full grouscan body is sub-staged across R4.3c..f; this stub
// returns the prologue snapshot so the prologue's behaviour is
// unit-testable in isolation.
#[must_use]
pub fn grouscan_run(scratch: &mut ScanScratch, gxmip: i32, gmipnum: u32) -> GrouscanPrologue {
    // --- Cache cf[128] state. Voxlap5.c:11601-11606. ---
    let c = scratch.cf[CF_SEED_INDEX];
    let z0 = c.z0;
    let z1 = c.z1;
    let cx0 = c.cx0;
    let cy0 = c.cy0;
    let cx1 = c.cx1;
    let cy1 = c.cy1;

    // --- ngxmax = min(gxmax, gxmip) when multiple mips exist. ---
    let mut ngxmax = scratch.gxmax;
    if gmipnum > 1 && gxmip < ngxmax {
        ngxmax = gxmip;
    }

    // --- Pick the leading raycast lane. Voxlap5.c:11621-11624. ---
    let lane: usize = usize::from(scratch.gpz[1] < scratch.gpz[0]);
    // ogx = gpz[lane] & 0xFFFF0000 — keep only the integer part of
    // the fixed-point depth.
    let ogx = scratch.gpz[lane] & -0x1_0000_i32;
    let gx = 0;
    // First column advance — voxlap's `gpz[lane] += gdz[lane]`.
    scratch.gpz[lane] = scratch.gpz[lane].wrapping_add(scratch.gdz[lane]);

    GrouscanPrologue {
        z0,
        z1,
        cx0,
        cy0,
        cx1,
        cy1,
        lane,
        ogx,
        gx,
        ngxmax,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh_scratch() -> ScanScratch {
        let mut s = ScanScratch::new_for_size(64, 64, 64);
        // Seed cf[128] with recognisable values.
        s.cf[CF_SEED_INDEX] = CfType {
            i0: 10,
            i1: 20,
            z0: 5,
            z1: 50,
            cx0: 100,
            cy0: 200,
            cx1: 300,
            cy1: 400,
        };
        s
    }

    #[test]
    fn prologue_caches_cf_seed_state() {
        let mut s = fresh_scratch();
        s.gpz = [1_000, 2_000];
        s.gdz = [10, 20];
        s.gxmax = 999_999;
        let p = grouscan_run(&mut s, 50_000, 1);
        assert_eq!(p.z0, 5);
        assert_eq!(p.z1, 50);
        assert_eq!(p.cx0, 100);
        assert_eq!(p.cy0, 200);
        assert_eq!(p.cx1, 300);
        assert_eq!(p.cy1, 400);
    }

    #[test]
    fn prologue_picks_leading_lane_with_smaller_gpz() {
        // gpz[0] = 1000 < gpz[1] = 2000 → lane 0 wins.
        let mut s = fresh_scratch();
        s.gpz = [1_000, 2_000];
        s.gdz = [10, 20];
        let p = grouscan_run(&mut s, 1, 1);
        assert_eq!(p.lane, 0);

        // Reverse: gpz[1] = 500 < gpz[0] = 800 → lane 1 wins.
        let mut s = fresh_scratch();
        s.gpz = [800, 500];
        s.gdz = [10, 20];
        let p = grouscan_run(&mut s, 1, 1);
        assert_eq!(p.lane, 1);
    }

    #[test]
    fn prologue_advances_winning_lane_by_gdz() {
        // gpz[0] = 1000 wins; gdz[0] = 256. After: gpz[0] = 1256.
        let mut s = fresh_scratch();
        s.gpz = [1_000, 2_000];
        s.gdz = [256, 999];
        let _ = grouscan_run(&mut s, 1, 1);
        assert_eq!(s.gpz[0], 1_256);
        // Lane 1 is untouched.
        assert_eq!(s.gpz[1], 2_000);
    }

    #[test]
    fn prologue_ngxmax_clamps_to_gxmip_when_multimip() {
        // gxmax = 1_000_000, gxmip = 50_000. ngxmax = 50_000.
        let mut s = fresh_scratch();
        s.gpz = [1_000, 2_000];
        s.gdz = [10, 20];
        s.gxmax = 1_000_000;
        let p = grouscan_run(&mut s, 50_000, 2);
        assert_eq!(p.ngxmax, 50_000);

        // Single-mip case: ngxmax = gxmax regardless of gxmip.
        let mut s = fresh_scratch();
        s.gpz = [1_000, 2_000];
        s.gdz = [10, 20];
        s.gxmax = 1_000_000;
        let p = grouscan_run(&mut s, 50_000, 1);
        assert_eq!(p.ngxmax, 1_000_000);
    }

    #[test]
    fn prologue_ogx_keeps_integer_part_of_gpz() {
        // gpz[0] = 0x12345678 → ogx = 0x12340000.
        let mut s = fresh_scratch();
        s.gpz = [0x1234_5678, 0x7FFF_FFFF];
        s.gdz = [0, 0];
        let p = grouscan_run(&mut s, 1, 1);
        assert_eq!(p.lane, 0);
        // 0x1234_0000 fits i32 positively (high bit clear).
        assert_eq!(p.ogx, 0x1234_0000_i32);
        assert_eq!(p.gx, 0);
    }
}
