//! Voxel-edit primitives: column z-range buffer manipulation.
//!
//! Ported from voxlap5.c's column-edit helpers. Each column is
//! represented during edit processing as a flat `[i32]` "z-range
//! buffer" (`b2` in voxlap C):
//!
//! ```text
//! [top0, bot0, top1, bot1, ..., top_sentinel, bot_sentinel]
//! ```
//!
//! Each `(top_k, bot_k)` pair represents a contiguous SOLID region
//! `[top_k, bot_k)`. Voxlap's z-axis grows downward (z=0 is sky), so
//! `top_k < bot_k` and `bot_k` is exclusive. The list is terminated
//! by a sentinel pair whose `bot` is `>= MAXZDIM`; air gaps live
//! between adjacent slabs (`bot_k..top_{k+1}`).
//!
//! The buffer is owned by the caller; both helpers run in place and
//! assume the caller sized `b2` with enough tail capacity to absorb
//! the worst-case growth (one extra slab pair per `delslab` split,
//! never more for `insslab` — it can only collapse). voxlap C sizes
//! these via `SCPITCH * 3` slots; the Rust port inherits the
//! contract.
//!
//! These helpers are CD.1 of the cave-demo plan. CD.2+ wraps them in
//! `scum2_line` / `scum2_finish` and on-disk slab encode / decode.

#![allow(dead_code)] // CD.1 lands the helpers; CD.2+ wires them up.

/// Voxlap's `MAXZDIM` (voxlap5.h:10). World z is one byte → at most
/// 256 voxels tall.
pub(crate) const MAXZDIM: i32 = 256;

/// Carve voxels in `[y0, y1)` to air on the column `b2`.
///
/// Port of `delslab` (voxlap5.c:4231). `b2` is mutated in place.
///
/// - `y0 >= y1` is a no-op.
/// - `y1 >= MAXZDIM` is clamped to `MAXZDIM - 1` (matches C).
/// - `b2.is_empty()` returns early (matches the C null-pointer
///   guard).
///
/// In the worst case the carve splits a single solid slab in two,
/// growing the list by one pair. The caller is responsible for
/// sizing `b2` to absorb this. The helper does not allocate.
pub(crate) fn delslab(b2: &mut [i32], y0: i32, mut y1: i32) {
    if y1 >= MAXZDIM {
        y1 = MAXZDIM - 1;
    }
    if y0 >= y1 || b2.is_empty() {
        return;
    }
    let mut z = 0usize;
    while y0 >= b2[z + 1] {
        z += 2;
    }
    if y0 > b2[z] {
        if y1 < b2[z + 1] {
            // Carve sits strictly inside slab z: split it in two and
            // shift the rest of the list right by one pair to make
            // room.
            let mut i = z;
            while b2[i + 1] < MAXZDIM {
                i += 2;
            }
            while i > z {
                b2[i + 3] = b2[i + 1];
                b2[i + 2] = b2[i];
                i -= 2;
            }
            b2[z + 3] = b2[z + 1];
            b2[z + 1] = y0;
            b2[z + 2] = y1;
            return;
        }
        // y1 reaches into (or past) the bottom of slab z: shrink slab
        // z's bot to y0, then move on to handle slabs below.
        b2[z + 1] = y0;
        z += 2;
    }
    if y1 >= b2[z + 1] {
        // y1 spans through slab z (and possibly further). Find the
        // slab i that y1 lands in (above its bottom), adopt it as
        // the new slab z, and shift the tail back to close the gap.
        let mut i = z + 2;
        while y1 >= b2[i + 1] {
            i += 2;
        }
        let delta = i - z;
        b2[z] = b2[i];
        b2[z + 1] = b2[i + 1];
        while b2[i + 1] < MAXZDIM {
            i += 2;
            b2[i - delta] = b2[i];
            b2[i - delta + 1] = b2[i + 1];
        }
    }
    if y1 > b2[z] {
        // y1 falls inside slab z: clamp top.
        b2[z] = y1;
    }
}

/// Insert solid voxels in `[y0, y1)` on the column `b2`.
///
/// Port of `insslab` (voxlap5.c:4259). Mirrors the shape of
/// [`delslab`]: walks `b2` to find where `[y0, y1)` lands and either
/// inserts a fresh slab into an air gap or merges with adjacent
/// slabs.
///
/// - `y0 >= y1` is a no-op.
/// - `b2.is_empty()` returns early (matches the C null-pointer
///   guard).
/// - Unlike `delslab`, `insslab` does **not** clamp `y1` against
///   `MAXZDIM`; voxlap relies on the caller for that. A `y1` value
///   `>= MAXZDIM` collapses the column into a single solid slab
///   that acts as the sentinel.
pub(crate) fn insslab(b2: &mut [i32], y0: i32, y1: i32) {
    if y0 >= y1 || b2.is_empty() {
        return;
    }
    let mut z = 0usize;
    while y0 > b2[z + 1] {
        z += 2;
    }
    if y1 < b2[z] {
        // [y0, y1) lives entirely in the air gap above slab z.
        // Shift slabs [z..=last] right by one pair, then drop the
        // new slab into slot z.
        let mut i = z;
        while b2[i + 1] < MAXZDIM {
            i += 2;
        }
        loop {
            b2[i + 3] = b2[i + 1];
            b2[i + 2] = b2[i];
            if i == z {
                break;
            }
            i -= 2;
        }
        b2[z + 1] = y1;
        b2[z] = y0;
        return;
    }
    if y0 < b2[z] {
        // [y0, y1) overlaps the top of slab z: extend the top up.
        b2[z] = y0;
    }
    if y1 >= b2[z + 2] && b2[z + 1] < MAXZDIM {
        // The insert reaches into slab z+2 (or further); merge slabs
        // z..i into a single slab, where i is the last slab whose
        // top is at or below y1.
        let mut i = z + 2;
        while y1 >= b2[i + 2] && b2[i + 1] < MAXZDIM {
            i += 2;
        }
        let delta = i - z;
        b2[z + 1] = b2[i + 1];
        while b2[i + 1] < MAXZDIM {
            i += 2;
            b2[i - delta] = b2[i];
            b2[i - delta + 1] = b2[i + 1];
        }
    }
    if y1 > b2[z + 1] {
        // y1 reaches past the bottom of slab z: extend the bot down.
        b2[z + 1] = y1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a sentinel-terminated `b2` from a list of solid slabs.
    /// The buffer has slack at the tail so split-style ops have room
    /// to shift.
    fn build_b2(slabs: &[(i32, i32)]) -> Vec<i32> {
        let mut buf: Vec<i32> = Vec::new();
        for &(top, bot) in slabs {
            assert!(top < bot, "slab top must be < bot");
            assert!(bot < MAXZDIM, "slab bot must fit below MAXZDIM");
            buf.push(top);
            buf.push(bot);
        }
        // Sentinel pair. voxlap's expandrle terminates with
        // bot = MAXZDIM; top is unread (writes only).
        buf.push(MAXZDIM);
        buf.push(MAXZDIM);
        // Slack — accommodates worst-case growth for any test.
        buf.resize(buf.len() + 32, 0);
        buf
    }

    /// Read back the slab list before the sentinel.
    fn read_slabs(b2: &[i32]) -> Vec<(i32, i32)> {
        let mut out = Vec::new();
        let mut i = 0;
        while b2[i + 1] < MAXZDIM {
            out.push((b2[i], b2[i + 1]));
            i += 2;
        }
        out
    }

    // ---- delslab ----------------------------------------------------

    #[test]
    fn delslab_noop_y0_ge_y1() {
        let mut b2 = build_b2(&[(10, 20)]);
        delslab(&mut b2, 15, 15);
        assert_eq!(read_slabs(&b2), [(10, 20)]);
        delslab(&mut b2, 20, 10);
        assert_eq!(read_slabs(&b2), [(10, 20)]);
    }

    #[test]
    fn delslab_split_inside_one_slab() {
        let mut b2 = build_b2(&[(10, 30)]);
        delslab(&mut b2, 15, 20);
        assert_eq!(read_slabs(&b2), [(10, 15), (20, 30)]);
    }

    #[test]
    fn delslab_shrink_bot_of_slab() {
        let mut b2 = build_b2(&[(10, 30)]);
        delslab(&mut b2, 20, 30);
        assert_eq!(read_slabs(&b2), [(10, 20)]);
    }

    #[test]
    fn delslab_shrink_top_of_slab() {
        let mut b2 = build_b2(&[(10, 30)]);
        delslab(&mut b2, 5, 15);
        assert_eq!(read_slabs(&b2), [(15, 30)]);
    }

    #[test]
    fn delslab_carve_full_slab() {
        let mut b2 = build_b2(&[(10, 30)]);
        delslab(&mut b2, 5, 35);
        assert_eq!(read_slabs(&b2), Vec::<(i32, i32)>::new());
    }

    #[test]
    fn delslab_in_air_noop() {
        let mut b2 = build_b2(&[(10, 30)]);
        delslab(&mut b2, 0, 8);
        assert_eq!(read_slabs(&b2), [(10, 30)]);
        delslab(&mut b2, 35, 50);
        assert_eq!(read_slabs(&b2), [(10, 30)]);
    }

    #[test]
    fn delslab_span_two_slabs_carve_middle() {
        let mut b2 = build_b2(&[(10, 30), (50, 70)]);
        delslab(&mut b2, 20, 60);
        assert_eq!(read_slabs(&b2), [(10, 20), (60, 70)]);
    }

    #[test]
    fn delslab_carve_two_full_slabs_keep_third() {
        let mut b2 = build_b2(&[(10, 20), (30, 40), (50, 60)]);
        delslab(&mut b2, 5, 45);
        assert_eq!(read_slabs(&b2), [(50, 60)]);
    }

    #[test]
    fn delslab_y1_clamped_to_maxzdim_minus_1() {
        let mut b2 = build_b2(&[(10, 200)]);
        delslab(&mut b2, 100, MAXZDIM);
        assert_eq!(read_slabs(&b2), [(10, 100)]);
    }

    #[test]
    fn delslab_carve_top_edge_of_slab() {
        // y1 == top of slab → should leave the slab untouched (the
        // carve range ends right at the surface).
        let mut b2 = build_b2(&[(10, 30)]);
        delslab(&mut b2, 5, 10);
        assert_eq!(read_slabs(&b2), [(10, 30)]);
    }

    #[test]
    fn delslab_carve_bot_edge_of_slab() {
        // y0 == bot of slab → no overlap.
        let mut b2 = build_b2(&[(10, 30)]);
        delslab(&mut b2, 30, 35);
        assert_eq!(read_slabs(&b2), [(10, 30)]);
    }

    #[test]
    fn delslab_carve_exact_full_slab_keeps_neighbors() {
        let mut b2 = build_b2(&[(10, 20), (30, 40), (50, 60)]);
        delslab(&mut b2, 30, 40);
        assert_eq!(read_slabs(&b2), [(10, 20), (50, 60)]);
    }

    // ---- insslab ----------------------------------------------------

    #[test]
    fn insslab_noop_y0_ge_y1() {
        let mut b2 = build_b2(&[(10, 20)]);
        insslab(&mut b2, 15, 15);
        assert_eq!(read_slabs(&b2), [(10, 20)]);
        insslab(&mut b2, 20, 10);
        assert_eq!(read_slabs(&b2), [(10, 20)]);
    }

    #[test]
    fn insslab_into_pure_air() {
        let mut b2 = build_b2(&[]);
        insslab(&mut b2, 10, 30);
        assert_eq!(read_slabs(&b2), [(10, 30)]);
    }

    #[test]
    fn insslab_into_air_gap_above_slab() {
        let mut b2 = build_b2(&[(50, 70)]);
        insslab(&mut b2, 10, 30);
        assert_eq!(read_slabs(&b2), [(10, 30), (50, 70)]);
    }

    #[test]
    fn insslab_into_air_gap_between_slabs() {
        let mut b2 = build_b2(&[(10, 20), (60, 70)]);
        insslab(&mut b2, 30, 50);
        assert_eq!(read_slabs(&b2), [(10, 20), (30, 50), (60, 70)]);
    }

    #[test]
    fn insslab_into_air_gap_below_all_slabs() {
        let mut b2 = build_b2(&[(10, 20)]);
        insslab(&mut b2, 30, 50);
        assert_eq!(read_slabs(&b2), [(10, 20), (30, 50)]);
    }

    #[test]
    fn insslab_extend_top_of_slab() {
        let mut b2 = build_b2(&[(50, 70)]);
        insslab(&mut b2, 30, 60);
        assert_eq!(read_slabs(&b2), [(30, 70)]);
    }

    #[test]
    fn insslab_extend_bot_of_slab() {
        let mut b2 = build_b2(&[(50, 70)]);
        insslab(&mut b2, 60, 80);
        assert_eq!(read_slabs(&b2), [(50, 80)]);
    }

    #[test]
    fn insslab_touch_top_merges() {
        // y1 == top of slab → adjacent insert merges (extends top).
        let mut b2 = build_b2(&[(50, 70)]);
        insslab(&mut b2, 30, 50);
        assert_eq!(read_slabs(&b2), [(30, 70)]);
    }

    #[test]
    fn insslab_touch_bot_merges() {
        // y0 == bot of slab → adjacent insert merges (extends bot).
        let mut b2 = build_b2(&[(50, 70)]);
        insslab(&mut b2, 70, 80);
        assert_eq!(read_slabs(&b2), [(50, 80)]);
    }

    #[test]
    fn insslab_merge_two_slabs() {
        let mut b2 = build_b2(&[(10, 30), (50, 70)]);
        insslab(&mut b2, 20, 60);
        assert_eq!(read_slabs(&b2), [(10, 70)]);
    }

    #[test]
    fn insslab_engulf_inner_slabs() {
        let mut b2 = build_b2(&[(10, 20), (30, 40), (50, 60)]);
        insslab(&mut b2, 5, 70);
        assert_eq!(read_slabs(&b2), [(5, 70)]);
    }

    #[test]
    fn insslab_engulf_then_keep_lower() {
        let mut b2 = build_b2(&[(10, 20), (30, 40), (60, 80)]);
        insslab(&mut b2, 5, 50);
        assert_eq!(read_slabs(&b2), [(5, 50), (60, 80)]);
    }

    #[test]
    fn insslab_engulf_then_merge_lower() {
        let mut b2 = build_b2(&[(10, 20), (30, 40), (60, 80)]);
        insslab(&mut b2, 5, 60);
        assert_eq!(read_slabs(&b2), [(5, 80)]);
    }

    #[test]
    fn insslab_chain_of_touching_inserts() {
        let mut b2 = build_b2(&[]);
        insslab(&mut b2, 10, 20);
        insslab(&mut b2, 20, 30);
        insslab(&mut b2, 30, 40);
        assert_eq!(read_slabs(&b2), [(10, 40)]);
    }

    #[test]
    fn insslab_carve_then_insert_round_trip() {
        // Land on slab, carve the middle, fill it back: end result
        // is identical to the original.
        let original = [(10, 50)];
        let mut b2 = build_b2(&original);
        delslab(&mut b2, 20, 30);
        assert_eq!(read_slabs(&b2), [(10, 20), (30, 50)]);
        insslab(&mut b2, 20, 30);
        assert_eq!(read_slabs(&b2), original);
    }

    #[test]
    fn insslab_into_sentinel_only_buffer_with_z_advance() {
        // Insert below an existing slab — z advances past slab[0].
        let mut b2 = build_b2(&[(10, 20)]);
        insslab(&mut b2, 100, 150);
        assert_eq!(read_slabs(&b2), [(10, 20), (100, 150)]);
    }
}
