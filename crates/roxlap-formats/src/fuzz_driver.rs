//! CT.3 — the `edit` fuzz target's shared driver.
//!
//! Interprets fuzzer bytes as a stream of edit operations — inserts,
//! carves, sphere carves and the CT carve-through-floor shapes
//! (bottom-reaching carves that empty columns or leave air-terminal
//! tails) — applies them to a small world, then asserts the format
//! invariants every consumer relies on. Lives in the library (not the
//! fuzz crate) so the committed seed streams also run as plain unit
//! tests on stable CI (`edit_fuzz_seeds_hold_invariants`); the
//! libFuzzer target (`fuzz/fuzz_targets/edit.rs`) is a one-line
//! wrapper.

use crate::color::VoxColor;
use crate::edit::{expandrle, set_rect, set_sphere, MAXZDIM};
use crate::vxl::{parse, serialize, Vxl};

/// World side used by the edit fuzzer — small keeps iterations fast
/// while still exercising multi-column exposure re-encodes.
pub const FUZZ_VSID: u32 = 16;

/// Decode `data` as an edit-op stream (8 bytes per op, at most 48
/// ops) against a fresh all-air world, then check:
///
/// 1. the on-disk round-trip is byte-stable (`serialize → parse →
///    serialize`),
/// 2. every column decodes to sane runs — non-overlapping, in
///    z-order, bounded by `MAXZDIM`, properly terminated (including
///    the CT pure terminator),
/// 3. the mip ladder builds without panicking and every mip column's
///    slab chain is walkable.
///
/// Op layout: `[kind, x0, y0, z0, x1, y1, z1, colour]`; `kind & 3`
/// selects insert rect / carve rect / carve sphere / bottom-reaching
/// carve (`z1` forced to 255 — the shape that was impossible before
/// CT.1). Panics on any violation — the fuzzer reports it as a crash.
#[allow(clippy::missing_panics_doc)]
pub fn run_edit_ops(data: &[u8]) {
    if data.len() < 8 {
        return;
    }
    let vsid = FUZZ_VSID;
    let m = i32::try_from(vsid).expect("small vsid");
    let mut vxl = Vxl::empty(vsid);
    // MAXCSIZ-bound headroom per column so a legitimate op stream can
    // never exhaust the pool (voxalloc panics are caller errors, not
    // format bugs).
    vxl.reserve_edit_capacity((vsid * vsid) as usize * 1100);

    for op in data.chunks_exact(8).take(48) {
        let x0 = i32::from(op[1]) % m;
        let y0 = i32::from(op[2]) % m;
        let z0 = i32::from(op[3]);
        let x1 = i32::from(op[4]) % m;
        let y1 = i32::from(op[5]) % m;
        let z1 = i32::from(op[6]);
        let colour = VoxColor(0x8000_0040 | (u32::from(op[7]) << 8));
        match op[0] & 3 {
            0 => set_rect(&mut vxl, [x0, y0, z0], [x1, y1, z1], Some(colour)),
            1 => set_rect(&mut vxl, [x0, y0, z0], [x1, y1, z1], None),
            2 => set_sphere(&mut vxl, [x0, y0, z0], u32::from(op[4]) % 8, None),
            _ => set_rect(&mut vxl, [x0, y0, z0], [x1, y1, 255], None),
        }
    }

    // Invariant 1 — byte-stable round-trip.
    let bytes = serialize(&vxl);
    let back = parse(&bytes).expect("serialized world must parse");
    assert_eq!(serialize(&back), bytes, "round-trip must be byte-stable");

    // Invariant 2 — sane runs per column.
    let mut spans = vec![0i32; 2 * MAXZDIM as usize + 4];
    for idx in 0..(vsid * vsid) as usize {
        spans.fill(0);
        expandrle(vxl.column_data(idx), &mut spans);
        let mut prev_bot = 0i32;
        let mut i = 0usize;
        loop {
            let (top, bot) = (spans[i], spans[i + 1]);
            assert!(top >= prev_bot, "col {idx} run {i}: out of order");
            assert!(top <= bot, "col {idx} run {i}: inverted");
            if bot >= MAXZDIM {
                assert!(top <= MAXZDIM, "col {idx}: terminator top");
                break;
            }
            prev_bot = bot;
            i += 2;
        }
    }

    // Invariant 3 — the mip ladder builds and walks.
    vxl.generate_mips(3);
    for mip in 0..vxl.mip_count() {
        let vs = (vsid >> mip).max(1);
        for idx in 0..(vs * vs) as usize {
            // `column_data_for_mip` runs `slng`'s bounded slab walk —
            // a malformed mip column panics here.
            let _ = vxl.column_data_for_mip(mip, idx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::run_edit_ops;

    /// The committed `fuzz/corpus/edit/` seeds, byte-mirrored (the
    /// corpus files are not packaged with the crate): CT shapes run
    /// through the full invariant battery on stable CI, not just
    /// under the nightly fuzzer.
    #[test]
    fn edit_fuzz_seeds_hold_invariants() {
        const SEEDS: [&[u8]; 3] = [
            // fill + carve-through (empties / air-tails most columns)
            &[
                0, 0, 0, 10, 15, 15, 200, 170, //
                3, 2, 2, 0, 12, 12, 0, 0,
            ],
            // fill to bottom + bottom-reaching rect carve + reinsert
            &[
                0, 0, 0, 100, 15, 15, 255, 85, //
                1, 4, 4, 180, 10, 10, 255, 0, //
                0, 5, 5, 200, 8, 8, 220, 102,
            ],
            // sphere carves incl. into the very bottom + final
            // bottom-layer carve-through
            &[
                2, 8, 8, 128, 6, 0, 0, 0, //
                0, 0, 0, 250, 15, 15, 255, 119, //
                2, 8, 8, 252, 5, 0, 0, 0, //
                3, 0, 0, 254, 15, 15, 0, 0,
            ],
        ];
        for seed in SEEDS {
            run_edit_ops(seed);
        }
    }
}
