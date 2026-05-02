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

/// Decode a column's slab bytes into the `b2` z-range buffer.
///
/// Port of `expandrle` (voxlap5.c:4131). Walks the slab chain, writes
/// `[top0, bot0, top1, bot1, ..., MAXZDIM_sentinel]` into `uind`.
/// `uind` MUST be sized to hold every solid run plus the sentinel pair
/// — voxlap allocates `MAXZDIM` slots, which is the worst-case bound
/// (one slab per z value).
///
/// The `if (v[3] >= v[1]) continue` branch handles a degenerate slab
/// where ceiling-z is at or below floor-z (no air gap above this
/// slab); voxlap merges it implicitly into the previous solid run by
/// skipping the slab in `uind`.
pub(crate) fn expandrle(slab: &[u8], uind: &mut [i32]) {
    uind[0] = i32::from(slab[1]);
    let mut i = 2usize;
    let mut v = 0usize;
    while slab[v] != 0 {
        v += usize::from(slab[v]) * 4;
        if slab[v + 3] >= slab[v + 1] {
            continue;
        }
        uind[i - 1] = i32::from(slab[v + 3]);
        uind[i] = i32::from(slab[v + 1]);
        i += 2;
    }
    uind[i - 1] = MAXZDIM;
}

/// One color-lookup record: original column's color for `z` in
/// `[z_start, z_end)`. Built from a column's slab bytes by
/// [`build_color_table`].
#[derive(Debug)]
struct ColorRange<'s> {
    z_start: i32,
    z_end: i32,
    /// Colors for `[z_start, z_end)`, BGRA, 4 bytes per voxel,
    /// ordered by `z`. Length `(z_end - z_start) * 4`.
    colors: &'s [u8],
}

/// Build the `tbuf2` color-lookup table for a column. Voxlap's
/// initial loop in `compilerle` (voxlap5.c:4163-4174). For each slab
/// emits a floor-color range and (for non-first slabs) a ceiling-
/// color range. Sentinel-terminated by a record at `z_start = MAXZDIM`.
fn build_color_table(slab: &[u8]) -> Vec<ColorRange<'_>> {
    let mut ranges = Vec::new();
    let mut v = 0usize;
    loop {
        let z_start = i32::from(slab[v + 1]);
        let z1c = i32::from(slab[v + 2]);
        let z_end = z1c + 1;
        let n_voxels = usize::try_from((z_end - z_start).max(0)).expect("voxel count >= 0");
        let off = v + 4;
        ranges.push(ColorRange {
            z_start,
            z_end,
            colors: &slab[off..off + n_voxels * 4],
        });

        let nextptr = slab[v];
        if nextptr == 0 {
            break;
        }
        let prev_v = v;
        v += usize::from(nextptr) * 4;
        let ze = i32::from(slab[v + 3]);
        // Ceiling color list of new slab. Voxlap stores these in the
        // tail of the *previous* slab's bytes — between its floor
        // colors and the new slab's header.
        //
        // C: ic[0] = ze + p.z - ia - i + 2, ic[1] = ze, ic[2] = v - ze*4
        // where p.z = z1c_prev, ia = z1_prev, i = nextptr_prev.
        let prev_z1 = i32::from(slab[prev_v + 1]);
        let prev_z1c = i32::from(slab[prev_v + 2]);
        let prev_nextptr = i32::from(slab[prev_v]);
        let ceil_z_start = ze + prev_z1c - prev_z1 - prev_nextptr + 2;
        let ceil_z_end = ze;
        let ceil_n =
            usize::try_from((ceil_z_end - ceil_z_start).max(0)).expect("ceiling voxel count >= 0");
        // Colors live at slab[v - ceil_n*4 .. v].
        let ceil_start = v - ceil_n * 4;
        ranges.push(ColorRange {
            z_start: ceil_z_start,
            z_end: ceil_z_end,
            colors: &slab[ceil_start..v],
        });
    }
    ranges.push(ColorRange {
        z_start: MAXZDIM,
        z_end: MAXZDIM,
        colors: &[],
    });
    ranges
}

/// Re-encode a column's `b2` z-range buffer to slab bytes.
///
/// Port of `compilerle` (voxlap5.c:4154). Walks `n0` (this column's
/// b2) voxel-by-voxel, writing one BGRA color record per exposed
/// solid voxel into `cbuf`. Color values are pulled from the
/// `original_column`'s slab bytes (where present) or from
/// `colfunc(px, py, z)` for newly-exposed voxels created by edits.
///
/// `n1..n4` are the four neighbor columns' b2 buffers (left, right,
/// north, south, in voxlap's order); they drive the "exposed" flag
/// `ia` that decides whether each voxel needs a color record.
///
/// Returns the number of bytes written to `cbuf`. Voxlap sizes
/// `cbuf` to `MAXCSIZ = 1028` bytes — the caller must do the same.
///
/// # Panics
///
/// Panics if a `b2` z value (always in `0..=MAXZDIM`) doesn't fit in
/// `u8` — would indicate a malformed b2 buffer.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::missing_panics_doc
)]
pub(crate) fn compilerle(
    n0: &[i32],
    n1: &[i32],
    n2: &[i32],
    n3: &[i32],
    n4: &[i32],
    cbuf: &mut [u8],
    original_column: &[u8],
    px: i32,
    py: i32,
    colfunc: &mut dyn FnMut(i32, i32, i32) -> i32,
) -> usize {
    let tbuf2 = build_color_table(original_column);

    let mut p_z: i32 = n0[0];
    // Voxlap's char narrowing semantics: cbuf is byte-addressed, and
    // `cbuf[n+1] = n0[i]` in C truncates to the low 8 bits. The
    // sentinel run at the tail of `n0` carries MAXZDIM (=256) which
    // wraps to 0 — voxlap uses that as part of the terminator slab.
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    let to_u8 = |v: i32| (v & 0xff) as u8;

    cbuf[1] = to_u8(p_z);
    let mut ze: i32 = n0[1];
    cbuf[2] = to_u8(ze - 1);
    cbuf[3] = 0;

    let mut i = 0usize;
    let mut onext = 0usize;
    let mut ic = 0usize;
    let mut ia: i32 = 15;
    let mut n = 4usize;
    let mut zend = if ze == MAXZDIM { -1 } else { ze - 1 };

    let mut n1_idx = 0usize;
    let mut n2_idx = 0usize;
    let mut n3_idx = 0usize;
    let mut n4_idx = 0usize;

    'outer: loop {
        let mut dacnt = 0;
        'middle: loop {
            // do { write voxel; ... } while (ia || p_z == zend)
            let exit_to_rlendit2 = loop {
                while p_z >= tbuf2[ic].z_end {
                    ic += 1;
                }
                let color: i32 = if p_z >= tbuf2[ic].z_start {
                    let off =
                        usize::try_from((p_z - tbuf2[ic].z_start) * 4).expect("color offset >= 0");
                    let bytes = &tbuf2[ic].colors[off..off + 4];
                    i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
                } else {
                    colfunc(px, py, p_z)
                };
                cbuf[n..n + 4].copy_from_slice(&color.to_le_bytes());
                n += 4;
                p_z += 1;
                if p_z >= ze {
                    break true; // goto rlendit2
                }
                while p_z >= n1[n1_idx] {
                    n1_idx += 1;
                    ia ^= 1;
                }
                while p_z >= n2[n2_idx] {
                    n2_idx += 1;
                    ia ^= 2;
                }
                while p_z >= n3[n3_idx] {
                    n3_idx += 1;
                    ia ^= 4;
                }
                while p_z >= n4[n4_idx] {
                    n4_idx += 1;
                    ia ^= 8;
                }
                if !(ia != 0 || p_z == zend) {
                    break false; // exit do-while: buried voxel
                }
            };

            if exit_to_rlendit2 {
                if ze >= MAXZDIM {
                    break 'outer;
                }
                i += 2;
                cbuf[onext] = u8::try_from((n - onext) >> 2).expect("slab dword count fits in u8");
                onext = n;
                p_z = n0[i];
                cbuf[n + 1] = to_u8(p_z);
                cbuf[n + 3] = to_u8(ze);
                ze = n0[i + 1];
                cbuf[n + 2] = to_u8(ze - 1);
                n += 4;
                zend = if ze == MAXZDIM { -1 } else { ze - 1 };
                break 'middle; // restart 'outer with dacnt = 0
            }

            // Buried voxel: close the floor list (or open a sub-slab
            // for the next exposed run).
            if dacnt == 0 {
                cbuf[onext + 2] = to_u8(p_z - 1);
                dacnt = 1;
            } else {
                cbuf[onext] = u8::try_from((n - onext) >> 2).expect("slab dword count fits in u8");
                onext = n;
                cbuf[n + 1] = to_u8(p_z);
                cbuf[n + 2] = to_u8(p_z - 1);
                cbuf[n + 3] = to_u8(p_z);
                n += 4;
            }

            // Skip forward to the smallest neighbor breakpoint.
            let n1_v = n1[n1_idx];
            let n2_v = n2[n2_idx];
            let n3_v = n3[n3_idx];
            let n4_v = n4[n4_idx];
            if n1_v < n2_v && n1_v < n3_v && n1_v < n4_v {
                if n1_v >= ze {
                    p_z = ze - 1;
                } else {
                    p_z = n1_v;
                    n1_idx += 1;
                    ia ^= 1;
                }
            } else if n2_v < n3_v && n2_v < n4_v {
                if n2_v >= ze {
                    p_z = ze - 1;
                } else {
                    p_z = n2_v;
                    n2_idx += 1;
                    ia ^= 2;
                }
            } else if n3_v < n4_v {
                if n3_v >= ze {
                    p_z = ze - 1;
                } else {
                    p_z = n3_v;
                    n3_idx += 1;
                    ia ^= 4;
                }
            } else if n4_v >= ze {
                p_z = ze - 1;
            } else {
                p_z = n4_v;
                n4_idx += 1;
                ia ^= 8;
            }

            if p_z == MAXZDIM - 1 {
                break 'outer;
            }
            // continue 'middle: re-enter inner do-while with same dacnt
        }
    }

    cbuf[onext] = 0;
    n
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
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

    // ---- expandrle (CD.2.3) ------------------------------------------

    /// Strip the `[top, bot, top, bot, ..., MAXZDIM]` decoded shape
    /// off a `b2` produced by [`expandrle`]. Last bot is the sentinel
    /// (== MAXZDIM); preceding pairs are the solid runs.
    fn read_uind(uind: &[i32]) -> Vec<(i32, i32)> {
        let mut out = Vec::new();
        let mut i = 0;
        while uind[i + 1] < MAXZDIM {
            out.push((uind[i], uind[i + 1]));
            i += 2;
        }
        // Last solid run terminated by sentinel.
        out.push((uind[i], uind[i + 1]));
        out
    }

    #[test]
    fn expandrle_single_slab_fully_solid_column() {
        // Voxlap encoding for solid [0, MAXZDIM) — the on-disk fixture
        // we see for "ground all the way down" columns.
        // [nextptr=0, z1=0, z1c=MAXZDIM-1, z0=0] + MAXZDIM × 4 colours.
        let z1c = u8::try_from(MAXZDIM - 1).expect("MAXZDIM-1 fits in u8");
        let mut slab = vec![0u8, 0, z1c, 0];
        slab.extend(std::iter::repeat_n(0u8, (MAXZDIM as usize) * 4));
        let mut uind = vec![0i32; 16];
        expandrle(&slab, &mut uind);
        // One solid run from z=0 to MAXZDIM, followed by sentinel.
        assert_eq!(uind[0], 0);
        assert_eq!(uind[1], MAXZDIM);
    }

    #[test]
    fn expandrle_single_slab_partial_floor() {
        // [nextptr=0, z1=64, z1c=66, z0=0] + 3 colours (don't matter).
        // Solid run = [64, MAXZDIM) — voxlap treats below-floor as
        // implicit solid.
        let slab = [0u8, 64, 66, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0];
        let mut uind = vec![0i32; 16];
        expandrle(&slab, &mut uind);
        assert_eq!(uind[0], 64);
        assert_eq!(uind[1], MAXZDIM);
    }

    #[test]
    fn expandrle_two_slabs_with_cave() {
        // Slab 0: nextptr=2 (advance 8 bytes), z1=10, z1c=12, z0=0.
        // Floor list: 3 colours = 12 bytes; total slab 0 = 4 + 12 - 4 = 12?
        // Actually slab 0 size = nextptr * 4 = 8 bytes. So with floor
        // list shorter than (z1c - z1 + 1) = 3 colours = 12 bytes, size
        // would be 16. To fit nextptr=2 (= 8 bytes), z1c-z1+1 must be 1
        // colour = 4 bytes (8 = 4 header + 4 colour).
        //
        // Layout: [2, 10, 10, 0, c0, c0, c0, c0, 0, 50, 52, 30, ...].
        // Slab 0: 1-voxel floor, then implicit solid [11, 30) (between
        // slab 0 floor end and slab 1 ceiling top).
        // Slab 1: ceiling [30, 50), floor [50, 53), implicit below.
        //
        // expandrle output:
        //   uind[0] = v[1]_s0 = 10
        //   advance to slab 1; v[3]=30 < v[1]=50 → write
        //   uind[1] = 30, uind[2] = 50, i = 4
        //   slab 1 nextptr=0 → exit loop
        //   uind[3] = MAXZDIM
        let slab = [
            2u8, 10, 10, 0, // slab 0 header
            0xaa, 0, 0, 0, // slab 0 1-voxel floor colour
            0, 50, 52, 30, // slab 1 (last) header
            0xbb, 0, 0, 0, // slab 1 floor 0
            0xcc, 0, 0, 0, // slab 1 floor 1
            0xdd, 0, 0, 0, // slab 1 floor 2
        ];
        let mut uind = vec![0i32; 16];
        expandrle(&slab, &mut uind);
        assert_eq!(uind[0], 10);
        assert_eq!(uind[1], 30);
        assert_eq!(uind[2], 50);
        assert_eq!(uind[3], MAXZDIM);
    }

    #[test]
    fn expandrle_skips_degenerate_slab_with_no_ceiling_gap() {
        // Slab 1 has v[3] >= v[1]: ceiling collapses with floor → no
        // air gap above. Voxlap's `continue` skips emitting a new
        // solid run for this slab, merging it with the previous run.
        //
        // Layout:
        //   slab 0: nextptr=2, z1=10, z1c=10, z0=0, 1 floor colour
        //   slab 1: nextptr=0, z1=20, z1c=22, z0=20 (z0 == z1 → skip)
        let slab = [
            2u8, 10, 10, 0, // slab 0 header
            0xaa, 0, 0, 0, // 1 floor colour
            0, 20, 22, 20, // slab 1 (degenerate: z0=z1=20)
            0xbb, 0, 0, 0, 0xcc, 0, 0, 0, 0xdd, 0, 0, 0, // floor colours
        ];
        let mut uind = vec![0i32; 16];
        expandrle(&slab, &mut uind);
        // Only one solid run (slab 0's), since slab 1 was skipped.
        assert_eq!(uind[0], 10);
        assert_eq!(uind[1], MAXZDIM);
    }

    #[test]
    fn expandrle_round_trips_through_b2_helpers() {
        // Decode a 2-slab column, then verify delslab can carve a
        // hole into the air gap (the `read_uind` shape matches the b2
        // shape that delslab/insslab consume).
        let slab = [
            2u8, 10, 10, 0, 0xaa, 0, 0, 0, 0, 50, 52, 30, 0xbb, 0, 0, 0, 0xcc, 0, 0, 0, 0xdd, 0, 0,
            0,
        ];
        let mut uind = vec![0i32; 16];
        expandrle(&slab, &mut uind);
        let runs = read_uind(&uind[..4]);
        assert_eq!(runs, [(10, 30), (50, MAXZDIM)]);
    }

    // ---- build_color_table + compilerle (CD.2.4) ----------------------

    /// Build a sentinel-terminated all-air b2 buffer for a neighbor.
    /// Sized with slack so compilerle's index walks don't run off
    /// the end (worst case = n0's voxel count + 1).
    fn all_air_neighbor() -> Vec<i32> {
        // Just the sentinel pair + slack.
        let mut buf = vec![MAXZDIM, MAXZDIM];
        buf.resize(buf.len() + MAXZDIM as usize, MAXZDIM);
        buf
    }

    /// Build a sentinel-terminated b2 from a list of solid runs.
    /// Compatible with [`compilerle`]'s input shape — the trailing
    /// sentinel must come last with bot == MAXZDIM.
    fn b2_from_runs(runs: &[(i32, i32)]) -> Vec<i32> {
        let mut buf = Vec::new();
        for &(top, bot) in runs {
            buf.push(top);
            buf.push(bot);
        }
        buf.push(MAXZDIM);
        buf.push(MAXZDIM);
        buf.resize(buf.len() + MAXZDIM as usize, MAXZDIM);
        buf
    }

    #[test]
    fn build_color_table_single_slab_one_floor_voxel() {
        // [nextptr=0, z1=10, z1c=10, z0=0] + 1 colour.
        let slab = [0u8, 10, 10, 0, 0xa1, 0xa2, 0xa3, 0xa4];
        let table = build_color_table(&slab);
        assert_eq!(table.len(), 2);
        assert_eq!(table[0].z_start, 10);
        assert_eq!(table[0].z_end, 11);
        assert_eq!(table[0].colors, &[0xa1, 0xa2, 0xa3, 0xa4]);
        // Sentinel.
        assert_eq!(table[1].z_start, MAXZDIM);
        assert_eq!(table[1].z_end, MAXZDIM);
    }

    #[test]
    fn build_color_table_two_slabs_with_ceiling() {
        // Slab 0: nextptr=4 (16 bytes total) — 4 hdr + 1 floor + 8 ceiling-of-slab1
        //   Layout: [4, 10, 10, 0, F0,F0,F0,F0, C0,C0,C0,C0, C1,C1,C1,C1]
        // Slab 1: [0, 50, 52, 30, ...]
        //   Slab 1's ceiling list = 2 voxels at z=28..30 (stored in slab 0's tail).
        let slab = [
            4u8, 10, 10, 0, // slab 0 header
            0xf0, 0xf0, 0xf0, 0xf0, // 1 floor color
            0xc0, 0xc0, 0xc0, 0xc0, // ceiling color for z=28
            0xc1, 0xc1, 0xc1, 0xc1, // ceiling color for z=29
            0u8, 50, 52, 30, // slab 1 header
            0xfa, 0xfa, 0xfa, 0xfa, // floor z=50
            0xfb, 0xfb, 0xfb, 0xfb, // floor z=51
            0xfc, 0xfc, 0xfc, 0xfc, // floor z=52
        ];
        let table = build_color_table(&slab);
        assert_eq!(table.len(), 4);
        // Slab 0 floor.
        assert_eq!(table[0].z_start, 10);
        assert_eq!(table[0].z_end, 11);
        assert_eq!(table[0].colors.len(), 4);
        // Slab 1 ceiling — 2 voxels at z=28..30.
        assert_eq!(table[1].z_start, 28);
        assert_eq!(table[1].z_end, 30);
        assert_eq!(
            table[1].colors,
            &[0xc0, 0xc0, 0xc0, 0xc0, 0xc1, 0xc1, 0xc1, 0xc1]
        );
        // Slab 1 floor.
        assert_eq!(table[2].z_start, 50);
        assert_eq!(table[2].z_end, 53);
        assert_eq!(table[2].colors.len(), 12);
        // Sentinel.
        assert_eq!(table[3].z_start, MAXZDIM);
    }

    #[test]
    fn compilerle_round_trip_single_slab_solid_to_maxzdim() {
        // Original column: solid from z=10 to MAXZDIM, full floor
        // color list. compilerle with all-air neighbors should
        // re-encode bit-equivalently in b2 shape.
        let mut slab = vec![0u8, 10, (MAXZDIM - 1) as u8, 0];
        for z in 10..MAXZDIM {
            // Distinct color per z so we can verify exact output bytes.
            slab.extend_from_slice(&[z as u8, (z + 1) as u8, (z + 2) as u8, 0]);
        }

        // Decode to b2.
        let mut b2 = vec![0i32; (MAXZDIM as usize) + 4];
        expandrle(&slab, &mut b2);
        assert_eq!(b2[0], 10);
        assert_eq!(b2[1], MAXZDIM);

        // Re-encode with all-air neighbors → no buried-voxel skip.
        let n_air = all_air_neighbor();
        let mut cbuf = vec![0u8; 1028];
        let mut colfunc_called = 0;
        let mut colfunc = |_x: i32, _y: i32, _z: i32| -> i32 {
            colfunc_called += 1;
            0
        };
        let written = compilerle(
            &b2,
            &n_air,
            &n_air,
            &n_air,
            &n_air,
            &mut cbuf,
            &slab,
            0,
            0,
            &mut colfunc,
        );
        assert_eq!(colfunc_called, 0, "all colors should come from tbuf2");

        // Output should match the input slab byte-for-byte (it's
        // already the minimal full-floor encoding).
        assert_eq!(written, slab.len());
        assert_eq!(&cbuf[..written], &slab[..]);

        // And expandrle on the output reproduces the same b2.
        let mut b2_round = vec![0i32; (MAXZDIM as usize) + 4];
        expandrle(&cbuf[..written], &mut b2_round);
        assert_eq!(b2_round[0], 10);
        assert_eq!(b2_round[1], MAXZDIM);
    }

    #[test]
    fn compilerle_round_trip_two_solid_runs_with_cave() {
        // b2 = [10, 30, 50, MAXZDIM] — one cave between two solid runs.
        // Build a synthetic original column that has full floor color
        // lists for both slabs; ceiling list for slab 1 is non-empty.
        // For simplicity construct via compilerle from an all-air-
        // neighbor first encode of the desired b2, then round-trip.

        // Step 1: build a SEED column. We'll compilerle from a
        // hand-rolled "all is colfunc" variant — colfunc returns z as
        // its color, deterministic.
        let dummy = vec![0u8, 0, (MAXZDIM - 1) as u8, 0];
        let mut dummy_full = dummy;
        dummy_full.extend(std::iter::repeat_n(0u8, (MAXZDIM as usize) * 4));

        let n_air = all_air_neighbor();
        let b2 = b2_from_runs(&[(10, 30), (50, MAXZDIM)]);
        let mut seed = vec![0u8; 1028];
        let mut colfunc = |_x: i32, _y: i32, z: i32| -> i32 { z };
        let seed_len = compilerle(
            &b2,
            &n_air,
            &n_air,
            &n_air,
            &n_air,
            &mut seed,
            &dummy_full,
            0,
            0,
            &mut colfunc,
        );
        seed.truncate(seed_len);

        // Step 2: decode the seed back to b2.
        let mut b2_round = vec![0i32; (MAXZDIM as usize) + 4];
        expandrle(&seed, &mut b2_round);
        // Two solid runs followed by sentinel.
        assert_eq!(b2_round[0], 10);
        assert_eq!(b2_round[1], 30);
        assert_eq!(b2_round[2], 50);
        assert_eq!(b2_round[3], MAXZDIM);

        // Step 3: compilerle again using the seed as the original
        // column — should produce byte-identical output (idempotent
        // round trip).
        let mut cbuf = vec![0u8; 1028];
        let mut never_called = 0;
        let mut colfunc2 = |_x: i32, _y: i32, _z: i32| -> i32 {
            never_called += 1;
            0
        };
        let written = compilerle(
            &b2,
            &n_air,
            &n_air,
            &n_air,
            &n_air,
            &mut cbuf,
            &seed,
            0,
            0,
            &mut colfunc2,
        );
        assert_eq!(never_called, 0, "second pass needs no colfunc");
        assert_eq!(written, seed_len);
        assert_eq!(&cbuf[..written], &seed[..]);
    }

    #[test]
    fn compilerle_buried_voxel_optimization_with_all_solid_neighbors() {
        // All 4 neighbors solid means every voxel below the top one is
        // buried — compilerle's dacnt path closes the floor list right
        // after writing the first exposed voxel.

        // Self column: solid [10, MAXZDIM). Voxlap's b2 convention has
        // the last real solid run extending to MAXZDIM (solid below is
        // implicit), so we never transition into the sentinel slab.
        let b2 = b2_from_runs(&[(10, MAXZDIM)]);
        let n_solid = b2_from_runs(&[(0, MAXZDIM)]);
        // Original column over-encoded with a full floor color list so
        // colfunc is never needed (verifies tbuf2 lookup for buried
        // voxels we'd otherwise skip).
        let mut slab = vec![0u8, 10, (MAXZDIM - 1) as u8, 0];
        for z in 10..MAXZDIM {
            slab.extend_from_slice(&[z as u8, 0, 0, 0]);
        }
        let mut cbuf = vec![0u8; 1028];
        let mut colfunc_called = 0;
        let mut colfunc = |_x: i32, _y: i32, _z: i32| -> i32 {
            colfunc_called += 1;
            0
        };
        let written = compilerle(
            &b2,
            &n_solid,
            &n_solid,
            &n_solid,
            &n_solid,
            &mut cbuf,
            &slab,
            0,
            0,
            &mut colfunc,
        );
        assert_eq!(colfunc_called, 0, "tbuf2 should cover every voxel");
        // Compressed output: only the top voxel exposed (z=10) →
        // 4-byte header + 1 color = 8 bytes.
        assert_eq!(written, 8);
        assert_eq!(cbuf[0], 0); // terminator nextptr
        assert_eq!(cbuf[1], 10); // z1
        assert_eq!(cbuf[2], 10); // z1c (only one exposed voxel)
        assert_eq!(cbuf[3], 0); // z0 dummy
                                // expandrle on output reproduces the b2 shape (still solid
                                // from z=10 onward, despite the compressed encoding).
        let mut b2_round = vec![0i32; (MAXZDIM as usize) + 4];
        expandrle(&cbuf[..written], &mut b2_round);
        assert_eq!(b2_round[0], 10);
        assert_eq!(b2_round[1], MAXZDIM);
    }
}
