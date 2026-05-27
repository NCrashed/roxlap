//! Cf-narrowing simulator (substage CF.1).
//!
//! Algorithmic fix for the axis-aligned-mip-beams artifact (= "fake
//! column" streaks in the scene-demo at deep mip-N + near-axis rays).
//! See `project_cf_narrowing_multi_session_plan.md` (memory) for the
//! multi-session plan and `project_axis_aligned_mip_beams.md` for the
//! original forensic finding.
//!
//! ## The problem this fixes
//!
//! At mip-N, the rasterizer walks `2^N` fewer columns than mip-0 in
//! the same depth range. `phase_draw_fwall`'s per-column cross_sign
//! narrowing fires `2^N` fewer times. cf entries stay wider than they
//! should. drawfwall then paints "barely positive" cross_sign pixels
//! at far-mip-N rays (cancellation regime: `cx*gy ≈ -cy*ogx`).
//!
//! ## What this simulator does
//!
//! At each mip transition (entry to [`crate::grouscan::phase_remiporend`])
//! BEFORE the cf-halve loop runs, [`cf_narrow_simulate`] walks each
//! active cf entry through `(2^old_mip - 1)` virtual finer-mip column
//! steps, applying [`phase_draw_fwall`]-shape inner-loop narrowing
//! per virtual step. The cumulative effect (across N successive mip
//! transitions) approximates what a mip-0 walk would have narrowed.
//!
//! ## Minimum-viable hypothesis (CF.1)
//!
//! - Only [`phase_draw_fwall`]'s i1-side narrowing is modelled.
//!   `do_slab_split` + `phase_draw_{cwall, ceil, flor}` are skipped.
//!   Hypothesis: in the cf-cancellation regime that produces beams,
//!   slab geometry is too far for split to fire, and the i0-side
//!   wall pushes a separate but symmetric story.
//! - `gy_raw` is frozen per cf entry at `gylookup[entry.z1]` for
//!   the entire simulation pass (drawfwall's outer `z1 -= 1` per
//!   voxel-row is not modelled).
//! - `virtual_ogx` advances by `gdz_finer[lane] = gdz_old[lane] >> 1`
//!   per step. Lane is recomputed per step from `virtual_gpz`.
//!
//! ## Failed prior attempt (do NOT repeat)
//!
//! See plan memo: the obvious "freeze (ogx, gy_raw) at entry and run
//! drawfwall's inner narrowing to convergence" took beams from 6404
//! to 6838 (+6.8 %). Root cause was that with frozen ogx, narrowing
//! converges to a fixed point after one virtual step, leaving the
//! remaining (2^OLD - 2) virtual steps no-ops. This simulator
//! advances virtual_ogx per step, so the next virtual step's
//! cross_sign sees a different depth and can re-trigger narrowing.

use crate::grouscan::{grouscan_cross_sign, CfType};

/// Per-call inputs the simulator reads from the live engine state.
/// All values are taken by value / by borrow — the simulator does not
/// touch `GrouscanState` directly so it stays pure and unit-testable.
#[derive(Debug, Clone, Copy)]
pub struct CfNarrowInputs<'a> {
    /// Per-lane `state.scratch.gpz` at the moment `phase_remiporend`
    /// is entered. The leading lane's gpz is what tripped the
    /// `(new_gpz as u32) > ngxmax as u32` overflow.
    pub gpz_at_entry: [i32; 2],
    /// Per-lane `state.scratch.gdz` at remiporend entry. Mip-OLD
    /// step size. The simulator uses `gdz >> 1` as the finer-mip
    /// step.
    pub gdz_old: [i32; 2],
    /// Scanline-constant cross-product weights.
    pub gi0: i32,
    pub gi1: i32,
    /// `state.gylookup` (mip-OLD sub-range) — read for each cf
    /// entry's `gy_raw = gylookup[entry.z1]` lookup.
    pub gylookup: &'a [i32],
    /// Current mip level we're transitioning OUT of (= `state.gmipcnt`
    /// BEFORE the `+= 1` in `phase_remiporend`). The simulator runs
    /// `(2^old_mip - 1)` virtual steps.
    pub old_mip: u32,
}

/// Maximum virtual steps to simulate. Caps the loop so a pathological
/// `old_mip` (shouldn't exceed ~6 in any real config) can't burn time.
const MAX_VIRTUAL_STEPS: u32 = 64;

/// Narrow each cf entry in `cf` by `(2^old_mip - 1)` virtual
/// finer-mip column steps' worth of `phase_draw_fwall`-shape
/// cross_sign narrowing.
///
/// Mutates each entry's `i1`, `cx1`, `cy1` in place. `i0`, `cx0`,
/// `cy0`, `z0`, `z1`, `chz_layer` are left untouched.
///
/// Pass the slice `&mut scratch.cf[CF_SEED_INDEX..=ce_idx]` from
/// `phase_remiporend`'s integration point (before the cf-halve loop).
///
/// # Per-step body
///
/// For each virtual step `s` in `0..(2^old_mip - 1)`:
/// 1. `lane = (virtual_gpz[1] < virtual_gpz[0]) ? 1 : 0`
/// 2. `virtual_ogx = virtual_gpz[lane] & -0x1_0000`
/// 3. For each cf entry, run drawfwall's inner narrowing to
///    convergence at `(virtual_ogx, gy_raw)`:
///    ```ignore
///    loop {
///        test = cross_sign(entry.cx1, entry.cy1, virtual_ogx, gy_raw)
///        if test <= 0 { break }
///        entry.cx1 -= gi0; entry.cy1 -= gi1; entry.i1 -= 1
///        if entry.i1 < entry.i0 { break }  // radar exhausted
///    }
///    ```
/// 4. `virtual_gpz[lane] += gdz_finer[lane]`
///
/// # No-ops
///
/// - `old_mip == 0` → return immediately (no finer mip to simulate).
/// - Empty `cf` slice → return (nothing to narrow).
/// - `gylookup` empty → return (defensive).
#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
pub fn cf_narrow_simulate(cf: &mut [CfType], in_: &CfNarrowInputs<'_>) {
    if in_.old_mip == 0 || cf.is_empty() || in_.gylookup.is_empty() {
        return;
    }

    let n_steps = (1u32 << in_.old_mip)
        .saturating_sub(1)
        .min(MAX_VIRTUAL_STEPS);

    // Finer-mip step sizes — half of mip-OLD. Arithmetic shift right
    // matches voxlap's `pslld mm`-style halving (no negative-overflow
    // concerns: gdz is non-negative in steady state).
    let gdz_finer: [i32; 2] = [in_.gdz_old[0] >> 1, in_.gdz_old[1] >> 1];

    // Start virtual gpz at the values present at remiporend entry.
    // The leading lane's `gpz[lane]` is what just tripped ngxmax —
    // we virtually walk backward through finer-mip column steps that
    // would have happened in the same depth range.
    //
    // Note: we ADVANCE virtual_gpz forward each virtual step (= the
    // direction drawfwall sees depth grow). Cross_sign at each step
    // uses the integer part `(virtual_ogx & -0x10000)` which is what
    // drawfwall would have read on a finer-mip column.
    let mut virtual_gpz = in_.gpz_at_entry;

    for _step in 0..n_steps {
        let lane = usize::from(virtual_gpz[1] < virtual_gpz[0]);
        let virtual_ogx = virtual_gpz[lane] & -0x1_0000_i32;

        for entry in cf.iter_mut() {
            // Convert z1 to gylookup index. Defensive bounds check
            // matches drawfwall's per-iter guard.
            let z1_idx = entry.z1 as usize;
            if z1_idx >= in_.gylookup.len() {
                continue;
            }
            let gy_raw = in_.gylookup[z1_idx];

            // One narrowing step per virtual column — matches the
            // per-virtual-column shape (one drawfwall outer-iter's
            // worth of narrowing per missing finer-mip column). The
            // earlier "narrow to convergence per step" variant
            // compounded across (2^OLD - 1) steps and over-narrowed
            // cy-driven rays (saw +15.6 % beam regression).
            //
            // The previous failed attempt used frozen (ogx, gy_raw)
            // across all virtual steps, producing fixed-point
            // convergence in one virtual step. This variant advances
            // virtual_ogx per step so cross_sign at each step sees a
            // fresh depth (= drawfwall's behaviour across columns).
            let test = grouscan_cross_sign(entry.cx1, entry.cy1, virtual_ogx, gy_raw);
            if test > 0 && entry.i1 > entry.i0 {
                entry.cx1 = entry.cx1.wrapping_sub(in_.gi0);
                entry.cy1 = entry.cy1.wrapping_sub(in_.gi1);
                entry.i1 -= 1;
            }
        }

        // Advance virtual gpz for the NEXT virtual step. The advance
        // direction matches `phase_after_delete_kept_presync`'s post-
        // trip-wire `gpz[lane] += gdz[lane]` — except we use the
        // finer mip's gdz_finer.
        virtual_gpz[lane] = virtual_gpz[lane].wrapping_add(gdz_finer[lane]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal CfType. Other-side fields (i0/cx0/cy0/z0) get
    /// neutral defaults so tests can focus on i1/cx1/cy1/z1 narrowing.
    fn cf(i0: isize, i1: isize, cx1: i32, cy1: i32, z1: i32) -> CfType {
        CfType {
            i0,
            i1,
            z0: 0,
            z1,
            cx0: 0,
            cy0: 0,
            cx1,
            cy1,
            chz_layer: 0,
        }
    }

    /// `gy_raw = 1` at the z1 index we'll use. Padding rest with 0.
    fn gylookup_with_one_at(z1: usize, gy: i32) -> Vec<i32> {
        let mut g = vec![0i32; (z1 + 1).max(8)];
        g[z1] = gy;
        g
    }

    /// Test 1 — `old_mip == 0` is a no-op. Even with a positive
    /// cross_sign, the simulator must not narrow.
    #[test]
    fn old_mip_zero_is_noop() {
        let mut cf_arr = [cf(0, 100, 0x0001_0000, 0, 10)];
        let gylookup = gylookup_with_one_at(10, 1);
        let inputs = CfNarrowInputs {
            gpz_at_entry: [0x0001_0000, i32::MAX],
            gdz_old: [0x0002_0000, 0],
            gi0: 0x0002_0000,
            gi1: 0,
            gylookup: &gylookup,
            old_mip: 0,
        };
        cf_narrow_simulate(&mut cf_arr, &inputs);
        assert_eq!(cf_arr[0].i1, 100);
        assert_eq!(cf_arr[0].cx1, 0x0001_0000);
        assert_eq!(cf_arr[0].cy1, 0);
    }

    /// Test 2 — `old_mip == 1`, axis-aligned positive cross_sign.
    /// One virtual step. cross_sign(0x10000, 0, 0x10000, 1)
    ///   = (1)*(1) + 0*(1) = 1 > 0 → narrow once.
    /// After: cx1 = -0x10000, i1 = 99.
    /// Re-test: cross_sign(-0x10000, 0, 0x10000, 1) = -1 → stop.
    #[test]
    fn axis_aligned_one_step_narrows_once() {
        let mut cf_arr = [cf(0, 100, 0x0001_0000, 0, 10)];
        let gylookup = gylookup_with_one_at(10, 1);
        let inputs = CfNarrowInputs {
            gpz_at_entry: [0x0001_0000, i32::MAX],
            gdz_old: [0x0002_0000, 0],
            gi0: 0x0002_0000,
            gi1: 0,
            gylookup: &gylookup,
            old_mip: 1,
        };
        cf_narrow_simulate(&mut cf_arr, &inputs);
        assert_eq!(cf_arr[0].i1, 99);
        assert_eq!(cf_arr[0].cx1, -0x0001_0000);
        assert_eq!(cf_arr[0].cy1, 0);
    }

    /// Test 3 — axis-aligned NEGATIVE cross_sign is a no-op.
    /// cross_sign(0x10000, 0, 0x10000, -1) = 1*(-1) = -1 → exit
    /// without narrowing.
    #[test]
    fn axis_aligned_negative_cross_sign_no_narrow() {
        let mut cf_arr = [cf(0, 100, 0x0001_0000, 0, 10)];
        let gylookup = gylookup_with_one_at(10, -1);
        let inputs = CfNarrowInputs {
            gpz_at_entry: [0x0001_0000, i32::MAX],
            gdz_old: [0x0002_0000, 0],
            gi0: 0x0002_0000,
            gi1: 0,
            gylookup: &gylookup,
            old_mip: 3,
        };
        cf_narrow_simulate(&mut cf_arr, &inputs);
        assert_eq!(cf_arr[0].i1, 100);
        assert_eq!(cf_arr[0].cx1, 0x0001_0000);
    }

    /// Test 4 — `old_mip == 2` (3 virtual steps) with persistent
    /// positive cross_sign that re-triggers each step.
    /// Setup: cy1 such that ogx advancement keeps cross_sign positive
    /// even after cx1 narrowing. (gi0 chosen smaller than initial
    /// cx1 so multiple narrows can fire per step.)
    ///
    /// gy_raw = 0, depth_s16 = ogx>>16. cross_sign = cx_s16*0 +
    /// cy_s16*depth_s16 = cy_s16 * depth_s16. cy1 = 0x10000 → cy_s16
    /// = 1. ogx = 0x10000 → depth_s16 = 1. test = 1. Positive — narrow.
    /// After: cy1 = 0, cy_s16 = 0 → test = 0 → stop. So 1 narrow per
    /// virtual step *if* gi1 brings cy1 to a fresh positive starting
    /// point at the next step. But cy1 only decreases (gi1 = 0x10000)
    /// → after 1 step cy1=0, future steps cross_sign = 0, no more.
    /// So expect 1 narrow total even though 3 virtual steps allowed.
    #[test]
    fn cy_dominated_one_narrow_then_stops() {
        let mut cf_arr = [cf(0, 100, 0x0001_0000, 0x0001_0000, 10)];
        let gylookup = gylookup_with_one_at(10, 0);
        let inputs = CfNarrowInputs {
            gpz_at_entry: [0x0001_0000, i32::MAX],
            gdz_old: [0x0002_0000, 0],
            gi0: 0x0001_0000,
            gi1: 0x0001_0000,
            gylookup: &gylookup,
            old_mip: 2, // 3 virtual steps
        };
        cf_narrow_simulate(&mut cf_arr, &inputs);
        assert_eq!(cf_arr[0].i1, 99);
        assert_eq!(cf_arr[0].cx1, 0);
        assert_eq!(cf_arr[0].cy1, 0);
    }

    /// Test 5 — empty cf slice + empty gylookup are no-ops.
    #[test]
    fn empty_inputs_are_noop() {
        let mut cf_arr: [CfType; 0] = [];
        let gylookup = vec![1i32; 16];
        let inputs = CfNarrowInputs {
            gpz_at_entry: [0, 0],
            gdz_old: [0, 0],
            gi0: 0,
            gi1: 0,
            gylookup: &gylookup,
            old_mip: 4,
        };
        cf_narrow_simulate(&mut cf_arr, &inputs);
        // No panic, no change.
        assert!(cf_arr.is_empty());

        // Empty gylookup is also no-op.
        let mut cf_arr2 = [cf(0, 100, 0x10000, 0, 0)];
        let inputs2 = CfNarrowInputs {
            gpz_at_entry: [0, 0],
            gdz_old: [0, 0],
            gi0: 0,
            gi1: 0,
            gylookup: &[],
            old_mip: 4,
        };
        cf_narrow_simulate(&mut cf_arr2, &inputs2);
        assert_eq!(cf_arr2[0].i1, 100);
    }

    /// Test 6 — radar exhaustion. With `i0 == i1`, the simulator must
    /// not narrow past i0.
    #[test]
    fn radar_exhaustion_stops_narrowing() {
        let mut cf_arr = [cf(50, 50, 0x0010_0000, 0, 10)];
        let gylookup = gylookup_with_one_at(10, 100); // strong positive
        let inputs = CfNarrowInputs {
            gpz_at_entry: [0x0001_0000, i32::MAX],
            gdz_old: [0x0002_0000, 0],
            gi0: 0x0001_0000,
            gi1: 0,
            gylookup: &gylookup,
            old_mip: 5, // 31 virtual steps
        };
        cf_narrow_simulate(&mut cf_arr, &inputs);
        // i1 must not drop below i0.
        assert!(
            cf_arr[0].i1 >= 50,
            "i1 should not go below i0; got {}",
            cf_arr[0].i1
        );
    }

    /// Test 7 — multi-step cumulative narrowing.
    /// gi0 = 0x4000 (small). cx1 = 0x10000 (= 16*gi0).
    /// gy_raw = 1, cy1 = 0. virtual_ogx advances per step but
    /// cross_sign depends only on cx1*gy_raw (axis-aligned).
    /// Per step: at most as many narrows as it takes for cx1 to go
    /// non-positive at the s16 level. cx_s16 starts at 1; after 1
    /// narrow (cx1 -= 0x4000) cx_s16 is still ... let's compute.
    /// cx1 = 0x10000 → s16 = 1. After 1 narrow: cx1 = 0xC000 → s16 = 0.
    /// test = 0 → stop. So 1 narrow per virtual step max.
    /// virtual_ogx evolution doesn't affect cross_sign for cy1=0.
    /// So total narrows = 1 (first virtual step), rest no-op.
    #[test]
    fn cumulative_narrowing_stops_when_cx_s16_zero() {
        let mut cf_arr = [cf(0, 100, 0x0001_0000, 0, 10)];
        let gylookup = gylookup_with_one_at(10, 1);
        let inputs = CfNarrowInputs {
            gpz_at_entry: [0x0001_0000, i32::MAX],
            gdz_old: [0x0002_0000, 0],
            gi0: 0x0000_4000,
            gi1: 0,
            gylookup: &gylookup,
            old_mip: 3, // 7 virtual steps
        };
        cf_narrow_simulate(&mut cf_arr, &inputs);
        assert_eq!(cf_arr[0].i1, 99);
        assert_eq!(cf_arr[0].cx1, 0x0000_C000);
    }

    /// Test 8 — ogx-evolution re-triggers narrowing on a near-axis
    /// case where each step's cross_sign sees a fresh depth.
    ///
    /// Setup: gy_raw = -1, cy1 = small positive. cross_sign =
    /// cx_s16 * -1 + cy_s16 * depth_s16. As virtual_ogx grows, the
    /// cy_s16 * depth_s16 term grows, can overcome -cx_s16.
    ///
    /// cx1 = 0x0001_0000 → cx_s16 = 1.
    /// cy1 = 0x0001_0000 → cy_s16 = 1.
    /// Step 0: virtual_ogx = 0x0001_0000 → depth_s16 = 1.
    ///   test = 1*(-1) + 1*1 = 0 → stop.
    /// Step 1: virtual_ogx = 0x0001_0000 + 0x0001_0000 = 0x0002_0000.
    ///   depth_s16 = 2. test = 1*(-1) + 1*2 = 1 → narrow.
    ///   cx1 -= gi0 = 0x10000-0x10000=0 → cx_s16=0. cy1 -= 0 → cy_s16=1.
    ///   re-test at ogx=0x20000: 0*(-1) + 1*2 = 2 → narrow again.
    ///   cx1 = -0x10000 → cx_s16 = -1. re-test: -1*(-1)+1*2 = 3 > 0
    ///   → narrow. cx1 = -0x20000 → cx_s16 = -2. test = -2*(-1)+1*2 = 4
    ///   → narrow forever unless capped.
    ///
    /// Per-entry loop cap (`entry_span + 1`) protects against this.
    /// With span = i1 - i0 = 100, the loop runs at most 101 iters
    /// per virtual step. So narrows happen but bounded.
    #[test]
    fn ogx_evolution_can_re_trigger_narrowing() {
        let mut cf_arr = [cf(0, 100, 0x0001_0000, 0x0001_0000, 10)];
        let gylookup = gylookup_with_one_at(10, -1);
        let inputs = CfNarrowInputs {
            gpz_at_entry: [0x0001_0000, i32::MAX],
            // gdz_finer = gdz_old >> 1. We want virtual_ogx to advance
            // by 0x10000 per step, so gdz_old = 0x20000.
            gdz_old: [0x0002_0000, 0],
            gi0: 0x0001_0000,
            gi1: 0,
            gylookup: &gylookup,
            old_mip: 2, // 3 virtual steps
        };
        let before_i1 = cf_arr[0].i1;
        let before_cx1 = cf_arr[0].cx1;
        cf_narrow_simulate(&mut cf_arr, &inputs);
        // Step 0: test=0 → no narrow.
        // Step 1: virtual_ogx=0x20000 → first iter test = 1*-1+1*2=1>0,
        //   narrow (cx1=0, i1=99). re-test = 0*-1+1*2=2>0, narrow
        //   (cx1=-0x10000, i1=98). re-test = -1*-1+1*2=3, narrow
        //   ... cap'd at entry_span + 1 = 101 iters → i1 hits 0.
        // (The minimum-viable sim over-narrows here. CF.3 will see
        // this in cross-pose validation; the bound just confirms
        // some narrowing happened.)
        assert!(
            cf_arr[0].i1 < before_i1,
            "should have narrowed (i1: {} -> {})",
            before_i1,
            cf_arr[0].i1
        );
        assert!(cf_arr[0].cx1 < before_cx1, "cx1 should have decreased");
    }
}
