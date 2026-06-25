//! Voxel-edit primitives: column z-range buffer manipulation.
//!
//! While editing, each `.vxl` column is held as a flat `[i32]` list of
//! solid z-runs (`spans`):
//!
//! ```text
//! [top0, bot0, top1, bot1, ..., top_sentinel, bot_sentinel]
//! ```
//!
//! Each `(top_k, bot_k)` pair is a contiguous SOLID region
//! `[top_k, bot_k)`. The `.vxl` z-axis grows downward (z=0 is sky), so
//! `top_k < bot_k` and `bot_k` is exclusive. The list ends in a
//! sentinel pair whose `bot` is `>= MAXZDIM`; air gaps live between
//! adjacent runs (`bot_k..top_{k+1}`).
//!
//! The buffer is owned by the caller; the helpers run in place and
//! assume it was sized with enough tail capacity for the worst-case
//! growth (one extra run pair per [`delslab`] split; [`insslab`] only
//! ever collapses). The [`ScumCtx`] driver sizes its row cache at
//! `SPAN_STRIDE * 3` ints per column to honour that.

#![allow(dead_code)]

/// World z is one byte → at most 256 voxels tall.
pub const MAXZDIM: i32 = 256;

/// Carve voxels in `[y0, y1)` to air on the span list `spans`,
/// mutated in place.
///
/// - `y0 >= y1` is a no-op.
/// - `y1 >= MAXZDIM` is clamped to `MAXZDIM - 1`.
/// - an empty `spans` returns early.
///
/// In the worst case the carve splits one solid run in two, growing
/// the list by one pair; the caller must have sized `spans` to absorb
/// it. Does not allocate.
pub fn delslab(spans: &mut [i32], y0: i32, mut y1: i32) {
    if y1 >= MAXZDIM {
        y1 = MAXZDIM - 1;
    }
    if y0 >= y1 || spans.is_empty() {
        return;
    }
    let mut z = 0usize;
    while y0 >= spans[z + 1] {
        z += 2;
    }
    if y0 > spans[z] {
        if y1 < spans[z + 1] {
            // Carve sits strictly inside slab z: split it in two and
            // shift the rest of the list right by one pair to make
            // room.
            let mut i = z;
            while spans[i + 1] < MAXZDIM {
                i += 2;
            }
            while i > z {
                spans[i + 3] = spans[i + 1];
                spans[i + 2] = spans[i];
                i -= 2;
            }
            spans[z + 3] = spans[z + 1];
            spans[z + 1] = y0;
            spans[z + 2] = y1;
            return;
        }
        // y1 reaches into (or past) the bottom of slab z: shrink slab
        // z's bot to y0, then move on to handle slabs below.
        spans[z + 1] = y0;
        z += 2;
    }
    if y1 >= spans[z + 1] {
        // y1 spans through slab z (and possibly further). Find the
        // slab i that y1 lands in (above its bottom), adopt it as
        // the new slab z, and shift the tail back to close the gap.
        let mut i = z + 2;
        while y1 >= spans[i + 1] {
            i += 2;
        }
        let delta = i - z;
        spans[z] = spans[i];
        spans[z + 1] = spans[i + 1];
        while spans[i + 1] < MAXZDIM {
            i += 2;
            spans[i - delta] = spans[i];
            spans[i - delta + 1] = spans[i + 1];
        }
    }
    if y1 > spans[z] {
        // y1 falls inside slab z: clamp top.
        spans[z] = y1;
    }
}

/// Insert solid voxels in `[y0, y1)` on the column `spans`.
///
/// Mirrors the shape of
/// [`delslab`]: walks `spans` to find where `[y0, y1)` lands and either
/// inserts a fresh slab into an air gap or merges with adjacent
/// slabs.
///
/// - `y0 >= y1` is a no-op.
/// - `spans.is_empty()` returns early (matches the C null-pointer
///   guard).
/// - Unlike `delslab`, `insslab` does **not** clamp `y1` against
///   `MAXZDIM`; the algorithm relies on the caller for that. A `y1` value
///   `>= MAXZDIM` collapses the column into a single solid slab
///   that acts as the sentinel.
pub fn insslab(spans: &mut [i32], y0: i32, y1: i32) {
    if y0 >= y1 || spans.is_empty() {
        return;
    }
    let mut z = 0usize;
    while y0 > spans[z + 1] {
        z += 2;
    }
    if y1 < spans[z] {
        // [y0, y1) lives entirely in the air gap above slab z.
        // Shift slabs [z..=last] right by one pair, then drop the
        // new slab into slot z.
        let mut i = z;
        while spans[i + 1] < MAXZDIM {
            i += 2;
        }
        loop {
            spans[i + 3] = spans[i + 1];
            spans[i + 2] = spans[i];
            if i == z {
                break;
            }
            i -= 2;
        }
        spans[z + 1] = y1;
        spans[z] = y0;
        return;
    }
    if y0 < spans[z] {
        // [y0, y1) overlaps the top of slab z: extend the top up.
        spans[z] = y0;
    }
    if y1 >= spans[z + 2] && spans[z + 1] < MAXZDIM {
        // The insert reaches into slab z+2 (or further); merge slabs
        // z..i into a single slab, where i is the last slab whose
        // top is at or below y1.
        let mut i = z + 2;
        while y1 >= spans[i + 2] && spans[i + 1] < MAXZDIM {
            i += 2;
        }
        let delta = i - z;
        spans[z + 1] = spans[i + 1];
        while spans[i + 1] < MAXZDIM {
            i += 2;
            spans[i - delta] = spans[i];
            spans[i - delta + 1] = spans[i + 1];
        }
        // Stamp a sentinel at the now-vacated `i+2-delta` slot.
        // The shift loop above exits with `spans[i+1] >= MAXZDIM`
        // (slab `i` is the sentinel) WITHOUT copying it forward —
        // so the merged slabs' old top/bot values are left in
        // place between `spans[z+2]` and `spans[i+1]`. Walkers using
        // `spans[i] < MAXZDIM` (top-check, e.g. `voxel_is_solid`)
        // and the subsequent compilerle re-emit then see phantom
        // overlapping runs that corrupt the column.
        //
        // Writing both top and bot ensures BOTH walker conventions
        // (top-check and bot-check `< MAXZDIM`) terminate here.
        spans[i + 2 - delta] = MAXZDIM;
        spans[i + 3 - delta] = MAXZDIM;
    }
    if y1 > spans[z + 1] {
        // y1 reaches past the bottom of slab z: extend the bot down.
        spans[z + 1] = y1;
    }
}

/// Decode a column's slab bytes into the `spans` z-range buffer.
///
/// Decode a `.vxl` slab column into a per-z solid-run list. Walks the slab chain, writes
/// `[top0, bot0, top1, bot1, ..., MAXZDIM_sentinel]` into `uind`.
/// `uind` MUST be sized to hold every solid run plus the sentinel pair
/// — we allocate `MAXZDIM` slots, which is the worst-case bound
/// (one slab per z value).
///
/// The `if (v[3] >= v[1]) continue` branch handles a degenerate slab
/// where ceiling-z is at or below floor-z (no air gap above this
/// slab); it merges implicitly into the previous solid run by
/// skipping the slab in `uind`.
pub fn expandrle(slab: &[u8], uind: &mut [i32]) {
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

/// Build the colour-lookup table for a column. The
/// initial loop in `compilerle` (-4174). For each slab
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
        // Ceiling color list of new slab. The format stores these in the
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

/// Re-encode a column's `spans` z-range buffer to slab bytes.
///
/// Encode a column's span list back to `.vxl` slab bytes. Walks `n0` (this column's
/// spans) voxel-by-voxel, writing one BGRA color record per exposed
/// solid voxel into `cbuf`. Color values are pulled from the
/// `original_column`'s slab bytes (where present) or from
/// `colfunc(px, py, z)` for newly-exposed voxels created by edits.
///
/// `n1..n4` are the four neighbor columns' spans buffers (left, right,
/// north, south, in N/E/W/S order); they drive the "exposed" flag
/// `ia` that decides whether each voxel needs a color record.
///
/// Returns the number of bytes written to `cbuf`. We size
/// `cbuf` to `MAXCSIZ = 1028` bytes — the caller must do the same.
///
/// # Panics
///
/// Panics if a `spans` z value (always in `0..=MAXZDIM`) doesn't fit in
/// `u8` — would indicate a malformed spans buffer.
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
    // Char narrowing semantics: cbuf is byte-addressed, and
    // `cbuf[n+1] = n0[i]` in C truncates to the low 8 bits. The
    // sentinel run at the tail of `n0` carries MAXZDIM (=256) which
    // wraps to 0 — that is part of the terminator slab.
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

// ====================================================================
// ScumCtx: scum2_line + scum2_finish + scum2 dispatcher (CD.2.5)
// ====================================================================
//
// The column-edit batch context.
//
// Acquires a `&mut Vxl` for the duration of an edit batch, manages
// the rolling 3-row spans buffer cache (`row_cache`), and re-encodes each
// finished y-row through `compilerle` + `voxalloc`/`voxdealloc` /
// `column_offset` updates.
//
// User flow (set_spans / set_sphere / set_cube):
//
// ```ignore
// vxl.reserve_edit_capacity(headroom);
// let mut ctx = ScumCtx::new(&mut vxl);
// ctx.set_colfunc(|x, y, z| 0xff_8080_80);
// for span in spans {
//     let spans = ctx.scum2(span.x, span.y).unwrap();
//     insslab(spans, span.z0, span.z1);
// }
// ctx.finish();
// ```
//
// Edits are accumulated per-column, then flushed when the y row
// advances (so the 3-row neighborhood is stable). `finish` drains
// the last 2 rows.

use crate::vxl::Vxl;

/// Per-column, per-row stride
/// (in i32 units) inside the row_cache buffer. spans buffer for column X
/// in a row whose base offset in row_cache is R lives at
/// `row_cache[R + X * SPAN_STRIDE * 3 .. R + X * SPAN_STRIDE * 3 + SPAN_STRIDE]`.
pub(crate) const SPAN_STRIDE: usize = 256;

/// Maximum bytes a single
/// column can occupy after re-encoding through [`compilerle`].
pub(crate) const MAXCSIZ: usize = 1028;

/// Sentinel "no row started yet". Encoded as
/// `0x80000000` = `i32::MIN`.
const SCOY_NONE: i32 = i32::MIN;

/// Initial row_cache offset for `cur_row_base` (`SPAN_STRIDE*6`).
const ROW_BASE_INITIAL: usize = SPAN_STRIDE * 6;

/// When `cur_row_base` reaches this offset, wrap back to
/// `ROW_BASE_INITIAL` (wrap at `SPAN_STRIDE*9`).
const ROW_BASE_WRAP: usize = SPAN_STRIDE * 9;

/// Column-edit batch context. Construct via [`ScumCtx::new`] after
/// calling [`Vxl::reserve_edit_capacity`] on the world.
///
/// Holds a `&mut Vxl` borrow plus the rolling 3-row spans buffer cache.
/// Mutate columns via [`ScumCtx::scum2`] — it returns the spans buffer
/// for the requested column; mutate via [`delslab`] / [`insslab`].
/// Edits are committed when the y row advances (or when
/// [`ScumCtx::finish`] drains the last 2 rows).
///
/// Caller MUST invoke [`ScumCtx::finish`] explicitly. Drop without
/// finish leaks the trailing 2 rows of edits — by contract.
pub struct ScumCtx<'v> {
    vxl: &'v mut Vxl,
    /// Rolling spans buffer cache. Sized `(vsid + 4) * 3 * SPAN_STRIDE` ints.
    row_cache: Vec<i32>,
    /// `compilerle` output scratch.
    cbuf: Vec<u8>,
    /// Color callback (= the colour callback). Called by
    /// `compilerle` for each newly-exposed voxel.
    colfunc: Box<dyn FnMut(i32, i32, i32) -> i32 + 'v>,

    // ---- rolling state ----------------------------------------------
    scoy: i32,
    cur_row_base: usize,
    scx0: i32,
    scx1: i32,
    scox0: i32,
    scox1: i32,
    scoox0: i32,
    scoox1: i32,
    scex0: i32,
    scex1: i32,
    sceox0: i32,
    sceox1: i32,

    /// `(x, y)` of the most-recent successful [`ScumCtx::scum2`] call,
    /// or `None` if no column has been loaded since the last row
    /// advance / context creation. Used by [`ScumCtx::with_column`]
    /// to skip the redundant `expandrle` when successive edits hit
    /// the same column — the per-column edit contract:
    /// re-loading from `sptr` would wipe pending in-row_cache edits.
    last_scum2: Option<(i32, i32)>,
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::if_not_else,
    clippy::similar_names
)]
impl<'v> ScumCtx<'v> {
    /// Open a new column-edit batch on a Vxl. The Vxl MUST have been
    /// upgraded with [`Vxl::reserve_edit_capacity`] beforehand (the
    /// slab allocator must be initialised).
    ///
    /// # Panics
    ///
    /// Panics if `vxl.vbit` is empty (no edit capacity reserved).
    pub fn new(vxl: &'v mut Vxl) -> Self {
        assert!(
            !vxl.vbit.is_empty(),
            "ScumCtx::new requires Vxl::reserve_edit_capacity to be called first"
        );
        let radar_size = (vxl.vsid as usize + 4) * 3 * SPAN_STRIDE;
        Self {
            vxl,
            row_cache: vec![0i32; radar_size],
            cbuf: vec![0u8; MAXCSIZ],
            colfunc: Box::new(|_, _, _| 0),
            scoy: SCOY_NONE,
            cur_row_base: ROW_BASE_INITIAL,
            scx0: 0,
            scx1: 0,
            scox0: 0,
            scox1: 0,
            scoox0: 0,
            scoox1: 0,
            scex0: 0,
            scex1: 0,
            sceox0: 0,
            sceox1: 0,
            last_scum2: None,
        }
    }

    /// Install the colour callback. Called
    /// for each newly-exposed voxel produced by edits.
    pub fn set_colfunc<F>(&mut self, f: F)
    where
        F: FnMut(i32, i32, i32) -> i32 + 'v,
    {
        self.colfunc = Box::new(f);
    }

    /// Open column `(x, y)` for editing; returns its spans buffer.
    /// Caller mutates via [`delslab`] / [`insslab`].
    ///
    /// Auto-flushes any prior y row that's no longer in the rolling
    /// window. Returns `None` if `(x, y)` is out of world bounds.
    pub fn scum2(&mut self, x: i32, y: i32) -> Option<&mut [i32]> {
        let vsid = self.vxl.vsid as i32;
        if x < 0 || x >= vsid || y < 0 || y >= vsid {
            return None;
        }

        if y != self.scoy {
            if self.scoy != SCOY_NONE {
                self.scum2_line();
                while self.scoy < y - 1 {
                    self.scx0 = i32::MAX;
                    self.scx1 = i32::MIN;
                    self.advance_row();
                    self.scum2_line();
                }
                self.advance_row();
            } else {
                self.scoox0 = i32::MAX;
                self.scox0 = i32::MAX;
                self.sceox0 = x + 1;
                self.scex0 = x + 1;
                self.sceox1 = x;
                self.scex1 = x;
                self.scoy = y;
                self.cur_row_base = ROW_BASE_INITIAL;
            }
            self.scx0 = x;
        } else {
            // Same y row as previous call: fill any skipped columns
            // between scx1 and x so the 3-row window stays continuous.
            while self.scx1 < x - 1 {
                self.scx1 += 1;
                let scx1 = self.scx1;
                self.expand_column_into_row(scx1, y, self.cur_row_base);
            }
        }

        let radar_idx = self.cur_row_base + (x as usize) * SPAN_STRIDE * 3;
        self.scx1 = x;
        self.expand_column_into_row(x, y, self.cur_row_base);
        self.last_scum2 = Some((x, y));
        Some(&mut self.row_cache[radar_idx..radar_idx + SPAN_STRIDE])
    }

    /// Edit one column with closure-based access. If `(x, y)` matches
    /// the immediately-previous successful [`ScumCtx::scum2`] /
    /// `with_column` call, reuses the cached spans buffer in row_cache
    /// (skipping the redundant `expandrle` that would wipe pending
    /// edits). Otherwise calls `scum2` to load the column.
    ///
    /// This is the primary edit API for span-style batch operations
    /// where multiple z ranges land on the same column —
    /// [`set_spans`] is a thin wrapper. Returns `false` and skips
    /// the closure if `(x, y)` is out of world bounds.
    pub fn with_column<F>(&mut self, x: i32, y: i32, f: F) -> bool
    where
        F: FnOnce(&mut [i32]),
    {
        if self.last_scum2 != Some((x, y)) && self.scum2(x, y).is_none() {
            return false;
        }
        // At this point either the cache hit or scum2 succeeded —
        // both leave the column's spans at cur_row_base + x * SPAN_STRIDE * 3.
        let radar_idx = self.cur_row_base + (x as usize) * SPAN_STRIDE * 3;
        let spans = &mut self.row_cache[radar_idx..radar_idx + SPAN_STRIDE];
        f(spans);
        true
    }

    /// Drain the last 2 rows and consume the context. MUST be called
    /// — Drop does not auto-finish (by contract).
    pub fn finish(mut self) {
        if self.scoy == SCOY_NONE {
            return;
        }
        for _ in 0..2 {
            self.scum2_line();
            self.scx0 = i32::MAX;
            self.scx1 = i32::MIN;
            self.advance_row();
        }
        self.scum2_line();
        self.scoy = SCOY_NONE;
    }

    /// Bump scoy by 1 and advance cur_row_base in the row_cache ring.
    /// Invalidates the [`ScumCtx::with_column`] cache: the prior
    /// row's column slots are still in the row_cache but their relative
    /// offset to the new `cur_row_base` has shifted, so re-using the
    /// cached `(x, y)` would index the wrong slot.
    fn advance_row(&mut self) {
        self.scoy += 1;
        self.cur_row_base += SPAN_STRIDE;
        if self.cur_row_base == ROW_BASE_WRAP {
            self.cur_row_base = ROW_BASE_INITIAL;
        }
        self.last_scum2 = None;
    }

    /// Load column `(x, y)` from the slab pool into the row_cache slot
    /// at `row_base + x * SPAN_STRIDE * 3`. Out-of-world columns get the
    /// all-solid sentinel `[0, MAXZDIM]` (matches the slab decode
    /// out-of-bounds behaviour).
    fn expand_column_into_row(&mut self, x: i32, y: i32, row_base: usize) {
        let vsid = self.vxl.vsid as i32;
        // Radar offset; the algorithm relies on the prefix slack for x = -1.
        let radar_idx_signed = (row_base as isize) + (x as isize) * (SPAN_STRIDE as isize) * 3;
        if radar_idx_signed < 0 {
            return;
        }
        #[allow(clippy::cast_sign_loss)]
        let radar_idx = radar_idx_signed as usize;
        if radar_idx + SPAN_STRIDE > self.row_cache.len() {
            return;
        }
        if x < 0 || x >= vsid || y < 0 || y >= vsid {
            self.row_cache[radar_idx] = 0;
            self.row_cache[radar_idx + 1] = MAXZDIM;
            return;
        }
        let idx = (y as usize) * (vsid as usize) + (x as usize);
        let slab = self.vxl.column_data(idx);
        expandrle(
            slab,
            &mut self.row_cache[radar_idx..radar_idx + SPAN_STRIDE],
        );
    }

    /// Flush row `scoy - 1` (the middle of the rolling 3-row window).
    ///
    #[allow(clippy::too_many_lines)]
    fn scum2_line(&mut self) {
        let vsid = self.vxl.vsid as i32;

        // x0 = min(scox0-1, min(scx0, scoox0)); x1 = max(scox1+1, max(scx1, scoox1))
        let x0 = (self.scox0 - 1).min(self.scx0).min(self.scoox0);
        self.scoox0 = self.scox0;
        self.scox0 = self.scx0;
        let x1 = (self.scox1 + 1).max(self.scx1).max(self.scoox1);
        self.scoox1 = self.scox1;
        self.scox1 = self.scx1;

        let uptr = wrap_radar(self.cur_row_base + SPAN_STRIDE);
        let mptr = wrap_radar(uptr + SPAN_STRIDE);

        // Load row scoy-2 (uptr) for [x0, x1] minus [sceox0, sceox1].
        let scoy_2 = self.scoy - 2;
        if x1 < self.sceox0 || x0 > self.sceox1 {
            for x in x0..=x1 {
                self.expand_column_into_row(x, scoy_2, uptr);
            }
        } else {
            for x in x0..self.sceox0 {
                self.expand_column_into_row(x, scoy_2, uptr);
            }
            let mut x = x1;
            while x > self.sceox1 {
                self.expand_column_into_row(x, scoy_2, uptr);
                x -= 1;
            }
        }

        // Load row scoy-1 (mptr) for [x0-1, x1+1].
        let scoy_1 = self.scoy - 1;
        if (self.scex1 | x1) >= 0 {
            for x in (x1 + 2)..self.scex0 {
                self.expand_column_into_row(x, scoy_1, mptr);
            }
            let mut x = x0 - 2;
            while x > self.scex1 {
                self.expand_column_into_row(x, scoy_1, mptr);
                x -= 1;
            }
        }
        if x1 + 1 < self.scex0 || x0 - 1 > self.scex1 {
            for x in (x0 - 1)..=(x1 + 1) {
                self.expand_column_into_row(x, scoy_1, mptr);
            }
        } else {
            for x in (x0 - 1)..self.scex0 {
                self.expand_column_into_row(x, scoy_1, mptr);
            }
            let mut x = x1 + 1;
            while x > self.scex1 {
                self.expand_column_into_row(x, scoy_1, mptr);
                x -= 1;
            }
        }
        self.sceox0 = (x0 - 1).min(self.scex0);
        self.sceox1 = (x1 + 1).max(self.scex1);

        // Load row scoy (cur_row_base) for [x0, x1] minus [scx0, scx1].
        let scoy_0 = self.scoy;
        let cur_row_base = self.cur_row_base;
        if x1 < self.scx0 || x0 > self.scx1 {
            for x in x0..=x1 {
                self.expand_column_into_row(x, scoy_0, cur_row_base);
            }
        } else {
            for x in x0..self.scx0 {
                self.expand_column_into_row(x, scoy_0, cur_row_base);
            }
            let mut x = x1;
            while x > self.scx1 {
                self.expand_column_into_row(x, scoy_0, cur_row_base);
                x -= 1;
            }
        }
        self.scex0 = x0;
        self.scex1 = x1;

        // Flush row scoy-1: re-encode each column in [x0, x1] within
        // [0, vsid).
        let y = self.scoy - 1;
        if !(0..vsid).contains(&y) {
            return;
        }
        let x0_clamped = x0.max(0);
        let x1_clamped = x1.min(vsid - 1);

        for x in x0_clamped..=x1_clamped {
            self.flush_column(x, y, mptr, uptr, cur_row_base);
        }
    }

    /// Re-encode column (x, y) using its spans buffer in `mptr` and
    /// neighbor b2s in mptr (left/right) + uptr (above) + cur_row_base
    /// (below). Commits the new bytes to the slab pool.
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    fn flush_column(&mut self, x: i32, y: i32, mptr: usize, uptr: usize, cur_row_base: usize) {
        let vsid = self.vxl.vsid as usize;
        let k = (x as usize) * SPAN_STRIDE * 3;
        let n0_pos = mptr + k;
        let n1_pos_signed = (mptr as isize) + (k as isize) - (SPAN_STRIDE as isize) * 3;
        let n2_pos = mptr + k + SPAN_STRIDE * 3;
        let n3_pos = uptr + k;
        let n4_pos = cur_row_base + k;

        // n1_pos may be at a negative offset for x = 0; we rely
        // on row_cache prefix slack. Skip if the slot is outside our row_cache.
        if n1_pos_signed < 0 {
            return;
        }
        let n1_pos = n1_pos_signed as usize;

        let idx = (y as usize) * vsid + (x as usize);

        // Snapshot original column bytes — compilerle reads colors
        // from them, and we overwrite them later via voxalloc + copy.
        let original_bytes: Vec<u8> = self.vxl.column_data(idx).to_vec();

        let written = {
            let row_cache = &self.row_cache;
            let n0 = &row_cache[n0_pos..n0_pos + SPAN_STRIDE];
            let n1 = &row_cache[n1_pos..n1_pos + SPAN_STRIDE];
            let n2 = &row_cache[n2_pos..n2_pos + SPAN_STRIDE];
            let n3 = &row_cache[n3_pos..n3_pos + SPAN_STRIDE];
            let n4 = &row_cache[n4_pos..n4_pos + SPAN_STRIDE];
            compilerle(
                n0,
                n1,
                n2,
                n3,
                n4,
                &mut self.cbuf,
                &original_bytes,
                x,
                y,
                &mut *self.colfunc,
            )
        };

        let old_offset = self.vxl.column_offset[idx];
        self.vxl.voxdealloc(old_offset);
        let new_offset = self.vxl.voxalloc(written as u32);
        self.vxl.data[new_offset as usize..new_offset as usize + written]
            .copy_from_slice(&self.cbuf[..written]);
        self.vxl.column_offset[idx] = new_offset;
    }
}

/// Wrap a row_cache offset back into the rolling-window range
/// `[ROW_BASE_INITIAL, ROW_BASE_WRAP)`.
fn wrap_radar(off: usize) -> usize {
    if off == ROW_BASE_WRAP {
        ROW_BASE_INITIAL
    } else {
        off
    }
}

// ====================================================================
// set_spans (CD.3) — thin wrapper over ScumCtx + delslab/insslab.
// ====================================================================

/// One vertical span on a column: `(x, y, z0..=z1)` solid voxels.
///
/// `z1` is INCLUSIVE per the `.vxl` span convention — the actual
/// edited range is the half-open `[z0, z1 + 1)`. Same convention
/// makes a `Vspan { z0: 100, z1: 100 }` carve / fill exactly one
/// voxel at z=100.
///
/// `x` / `y` are full-world `u32` coordinates (no patch-relative +
/// offset machinery — not needed here).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Vspan {
    pub x: u32,
    pub y: u32,
    pub z0: u8,
    pub z1: u8,
}

/// Operation for span-style edits.
///
/// `Carve` flips the listed voxels to air (per-span [`delslab`]) —
/// the colfunc is consulted by the internal RLE re-compile step for
/// newly-exposed voxels just outside the carved range (above and
/// below) which weren't previously in the column's color list.
///
/// `Insert` flips the listed voxels to solid (per-span [`insslab`])
/// — the colfunc is consulted for the inserted voxels themselves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpanOp {
    Carve,
    Insert,
}

/// Apply a list of column-aligned vertical spans with a custom colour
/// callback — the closure can capture arbitrary state to implement
/// flat, jittered, textured, or otherwise procedural colour patterns.
///
/// `colfunc(x, y, z) -> i32` returns the BGRA colour for any voxel
/// that needs one: the inserted voxels (Insert op) or the newly-
/// exposed voxels just outside the carved range (Carve op).
///
/// **Contract**: `spans` MUST be sorted ascending by `(y, x)` and,
/// within each `(x, y)` group, ascending by `z0`. The driver relies on
/// this for correct row-flush ordering and for `with_column`'s
/// caching invariant. Out-of-bounds spans (x or y >= vsid) are
/// silently skipped.
///
/// Empty input is a no-op (no `ScumCtx` is created).
///
/// # Panics
///
/// Panics if `world.vbit` is empty — call
/// [`Vxl::reserve_edit_capacity`] first.
#[allow(clippy::cast_possible_wrap)]
pub fn set_spans_with_colfunc<F>(world: &mut Vxl, spans: &[Vspan], op: SpanOp, colfunc: F)
where
    F: FnMut(i32, i32, i32) -> i32,
{
    if spans.is_empty() {
        return;
    }
    let inserting = op == SpanOp::Insert;
    let mut ctx = ScumCtx::new(world);
    ctx.set_colfunc(colfunc);
    for span in spans {
        let x = span.x as i32;
        let y = span.y as i32;
        let z0 = i32::from(span.z0);
        let z1 = i32::from(span.z1) + 1; // inclusive → half-open exclusive
        ctx.with_column(x, y, |spans| {
            if inserting {
                insslab(spans, z0, z1);
            } else {
                delslab(spans, z0, z1);
            }
        });
    }
    ctx.finish();
}

/// Apply a list of column-aligned vertical spans with a constant
/// colour. Convenience wrapper over [`set_spans_with_colfunc`] for
/// the common cases:
///
/// - `color = None` → carve. Newly-exposed voxels get colour 0.
/// - `color = Some(c)` → insert. Inserted voxels get colour `c`.
///
/// For non-constant colours (jitter, texture-mapped, position-
/// dependent, gradient by depth), use [`set_spans_with_colfunc`]
/// directly with a closure capturing the relevant state.
///
/// See [`set_spans_with_colfunc`] for the sort-order contract and
/// panic semantics.
pub fn set_spans(world: &mut Vxl, spans: &[Vspan], color: Option<u32>) {
    let op = if color.is_some() {
        SpanOp::Insert
    } else {
        SpanOp::Carve
    };
    #[allow(clippy::cast_possible_wrap)]
    let c_i32 = color.unwrap_or(0) as i32;
    set_spans_with_colfunc(world, spans, op, move |_, _, _| c_i32);
}

// ====================================================================
// set_cube / set_rect / set_sphere (CD.4) — region wrappers.
// ====================================================================

/// Edit a single voxel at `(x, y, z)`. (`setcube`).
///
/// `color = None` carves to air; `Some(c)` inserts solid coloured
/// `c`. Out-of-bounds coordinates are silently skipped.
///
/// **Note**: this skips the "exposed-solid in-place
/// colour overwrite" optimization — every call goes through the
/// scum2 + delslab/insslab + compilerle pipeline. Per-voxel edits
/// are rare in the cave-demo workload; the optimization can land
/// later if needed.
///
/// # Panics
///
/// Panics if `world.vbit` is empty — call
/// [`Vxl::reserve_edit_capacity`] first.
pub fn set_cube(world: &mut Vxl, x: i32, y: i32, z: i32, color: Option<u32>) {
    let op = if color.is_some() {
        SpanOp::Insert
    } else {
        SpanOp::Carve
    };
    #[allow(clippy::cast_possible_wrap)]
    let c_i32 = color.unwrap_or(0) as i32;
    set_cube_with_colfunc(world, x, y, z, op, move |_, _, _| c_i32);
}

/// [`set_cube`] with a custom colour callback.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
pub fn set_cube_with_colfunc<F>(world: &mut Vxl, x: i32, y: i32, z: i32, op: SpanOp, colfunc: F)
where
    F: FnMut(i32, i32, i32) -> i32,
{
    let vsid = world.vsid as i32;
    if x < 0 || x >= vsid || y < 0 || y >= vsid || !(0..MAXZDIM).contains(&z) {
        return;
    }
    let span = Vspan {
        x: x as u32,
        y: y as u32,
        z0: z as u8,
        z1: z as u8,
    };
    set_spans_with_colfunc(world, &[span], op, colfunc);
}

/// Edit an axis-aligned box `[lo, hi]` (inclusive on both ends in
/// every axis). (`setrect`).
///
/// `color = None` carves to air; `Some(c)` inserts solid coloured
/// `c`. The box is sorted and clamped to world bounds before
/// iteration; an empty box (any axis where lo > hi after clamp) is
/// a no-op.
///
/// # Panics
///
/// Panics if `world.vbit` is empty — call
/// [`Vxl::reserve_edit_capacity`] first.
pub fn set_rect(world: &mut Vxl, lo: [i32; 3], hi: [i32; 3], color: Option<u32>) {
    let op = if color.is_some() {
        SpanOp::Insert
    } else {
        SpanOp::Carve
    };
    #[allow(clippy::cast_possible_wrap)]
    let c_i32 = color.unwrap_or(0) as i32;
    set_rect_with_colfunc(world, lo, hi, op, move |_, _, _| c_i32);
}

/// [`set_rect`] with a custom colour callback.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
pub fn set_rect_with_colfunc<F>(world: &mut Vxl, lo: [i32; 3], hi: [i32; 3], op: SpanOp, colfunc: F)
where
    F: FnMut(i32, i32, i32) -> i32,
{
    let vsid = world.vsid as i32;
    let xs = lo[0].min(hi[0]).max(0);
    let xe = lo[0].max(hi[0]).min(vsid - 1);
    let ys = lo[1].min(hi[1]).max(0);
    let ye = lo[1].max(hi[1]).min(vsid - 1);
    let zs = lo[2].min(hi[2]).max(0);
    let ze = lo[2].max(hi[2]).min(MAXZDIM - 1);
    if xs > xe || ys > ye || zs > ze {
        return;
    }
    let inserting = op == SpanOp::Insert;
    let mut ctx = ScumCtx::new(world);
    ctx.set_colfunc(colfunc);
    for y in ys..=ye {
        for x in xs..=xe {
            ctx.with_column(x, y, |spans| {
                if inserting {
                    insslab(spans, zs, ze + 1);
                } else {
                    delslab(spans, zs, ze + 1);
                }
            });
        }
    }
    ctx.finish();
}

/// Edit a sphere of voxels centred at `center` with the given
/// `radius`. (`setsphere`).
///
/// Uses Euclidean distance (the
/// round-sphere case). For non-Euclidean shapes (octahedron at
/// `curpow = 1.0`, etc.) the user can drop down to [`ScumCtx`] and
/// roll their own iteration; the cave-demo's spherical bullet
/// impacts only need the Euclidean case.
///
/// `color = None` carves to air; `Some(c)` inserts solid coloured
/// `c`. The bounding box is clamped to world bounds; a sphere fully
/// outside the world is a no-op.
///
/// # Panics
///
/// Panics if `world.vbit` is empty — call
/// [`Vxl::reserve_edit_capacity`] first.
pub fn set_sphere(world: &mut Vxl, center: [i32; 3], radius: u32, color: Option<u32>) {
    let op = if color.is_some() {
        SpanOp::Insert
    } else {
        SpanOp::Carve
    };
    #[allow(clippy::cast_possible_wrap)]
    let c_i32 = color.unwrap_or(0) as i32;
    set_sphere_with_colfunc(world, center, radius, op, move |_, _, _| c_i32);
}

/// [`set_sphere`] with a custom colour callback.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::similar_names
)]
pub fn set_sphere_with_colfunc<F>(
    world: &mut Vxl,
    center: [i32; 3],
    radius: u32,
    op: SpanOp,
    colfunc: F,
) where
    F: FnMut(i32, i32, i32) -> i32,
{
    let vsid = world.vsid as i32;
    let cx = center[0];
    let cy = center[1];
    let cz = center[2];
    let r = radius as i32;
    let xs = (cx - r).max(0);
    let xe = (cx + r).min(vsid - 1);
    let ys = (cy - r).max(0);
    let ye = (cy + r).min(vsid - 1);
    let zs = (cz - r).max(0);
    let ze = (cz + r).min(MAXZDIM - 1);
    if xs > xe || ys > ye || zs > ze {
        return;
    }
    let r_sq = r * r;
    let inserting = op == SpanOp::Insert;
    let mut ctx = ScumCtx::new(world);
    ctx.set_colfunc(colfunc);
    for y in ys..=ye {
        let dy = y - cy;
        let dy_sq = dy * dy;
        if dy_sq > r_sq {
            continue;
        }
        for x in xs..=xe {
            let dx = x - cx;
            let dx_sq = dx * dx;
            let xy_sq = dx_sq + dy_sq;
            if xy_sq > r_sq {
                continue;
            }
            // dz_max satisfies dx² + dy² + dz² <= r²; voxel range is
            // z = cz - dz_max ..= cz + dz_max.
            let dz_max_sq = r_sq - xy_sq;
            let dz_max = (dz_max_sq as f32).sqrt() as i32;
            let z_lo = (cz - dz_max).max(zs);
            let z_hi = (cz + dz_max).min(ze);
            if z_lo > z_hi {
                continue;
            }
            ctx.with_column(x, y, |spans| {
                if inserting {
                    insslab(spans, z_lo, z_hi + 1);
                } else {
                    delslab(spans, z_lo, z_hi + 1);
                }
            });
        }
    }
    ctx.finish();
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::items_after_statements
)]
mod tests {
    use super::*;

    /// Build a sentinel-terminated `spans` from a list of solid slabs.
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
        // Sentinel pair. the slab decode terminates with
        // bot = MAXZDIM; top is unread (writes only).
        buf.push(MAXZDIM);
        buf.push(MAXZDIM);
        // Slack — accommodates worst-case growth for any test.
        buf.resize(buf.len() + 32, 0);
        buf
    }

    /// Read back the slab list before the sentinel.
    fn read_slabs(spans: &[i32]) -> Vec<(i32, i32)> {
        let mut out = Vec::new();
        let mut i = 0;
        while spans[i + 1] < MAXZDIM {
            out.push((spans[i], spans[i + 1]));
            i += 2;
        }
        out
    }

    // ---- delslab ----------------------------------------------------

    #[test]
    fn delslab_noop_y0_ge_y1() {
        let mut spans = build_b2(&[(10, 20)]);
        delslab(&mut spans, 15, 15);
        assert_eq!(read_slabs(&spans), [(10, 20)]);
        delslab(&mut spans, 20, 10);
        assert_eq!(read_slabs(&spans), [(10, 20)]);
    }

    #[test]
    fn delslab_split_inside_one_slab() {
        let mut spans = build_b2(&[(10, 30)]);
        delslab(&mut spans, 15, 20);
        assert_eq!(read_slabs(&spans), [(10, 15), (20, 30)]);
    }

    #[test]
    fn delslab_shrink_bot_of_slab() {
        let mut spans = build_b2(&[(10, 30)]);
        delslab(&mut spans, 20, 30);
        assert_eq!(read_slabs(&spans), [(10, 20)]);
    }

    #[test]
    fn delslab_shrink_top_of_slab() {
        let mut spans = build_b2(&[(10, 30)]);
        delslab(&mut spans, 5, 15);
        assert_eq!(read_slabs(&spans), [(15, 30)]);
    }

    #[test]
    fn delslab_carve_full_slab() {
        let mut spans = build_b2(&[(10, 30)]);
        delslab(&mut spans, 5, 35);
        assert_eq!(read_slabs(&spans), Vec::<(i32, i32)>::new());
    }

    #[test]
    fn delslab_in_air_noop() {
        let mut spans = build_b2(&[(10, 30)]);
        delslab(&mut spans, 0, 8);
        assert_eq!(read_slabs(&spans), [(10, 30)]);
        delslab(&mut spans, 35, 50);
        assert_eq!(read_slabs(&spans), [(10, 30)]);
    }

    #[test]
    fn delslab_span_two_slabs_carve_middle() {
        let mut spans = build_b2(&[(10, 30), (50, 70)]);
        delslab(&mut spans, 20, 60);
        assert_eq!(read_slabs(&spans), [(10, 20), (60, 70)]);
    }

    #[test]
    fn delslab_carve_two_full_slabs_keep_third() {
        let mut spans = build_b2(&[(10, 20), (30, 40), (50, 60)]);
        delslab(&mut spans, 5, 45);
        assert_eq!(read_slabs(&spans), [(50, 60)]);
    }

    #[test]
    fn delslab_y1_clamped_to_maxzdim_minus_1() {
        let mut spans = build_b2(&[(10, 200)]);
        delslab(&mut spans, 100, MAXZDIM);
        assert_eq!(read_slabs(&spans), [(10, 100)]);
    }

    #[test]
    fn delslab_carve_top_edge_of_slab() {
        // y1 == top of slab → should leave the slab untouched (the
        // carve range ends right at the surface).
        let mut spans = build_b2(&[(10, 30)]);
        delslab(&mut spans, 5, 10);
        assert_eq!(read_slabs(&spans), [(10, 30)]);
    }

    #[test]
    fn delslab_carve_bot_edge_of_slab() {
        // y0 == bot of slab → no overlap.
        let mut spans = build_b2(&[(10, 30)]);
        delslab(&mut spans, 30, 35);
        assert_eq!(read_slabs(&spans), [(10, 30)]);
    }

    #[test]
    fn delslab_carve_exact_full_slab_keeps_neighbors() {
        let mut spans = build_b2(&[(10, 20), (30, 40), (50, 60)]);
        delslab(&mut spans, 30, 40);
        assert_eq!(read_slabs(&spans), [(10, 20), (50, 60)]);
    }

    // ---- insslab ----------------------------------------------------

    #[test]
    fn insslab_noop_y0_ge_y1() {
        let mut spans = build_b2(&[(10, 20)]);
        insslab(&mut spans, 15, 15);
        assert_eq!(read_slabs(&spans), [(10, 20)]);
        insslab(&mut spans, 20, 10);
        assert_eq!(read_slabs(&spans), [(10, 20)]);
    }

    #[test]
    fn insslab_into_pure_air() {
        let mut spans = build_b2(&[]);
        insslab(&mut spans, 10, 30);
        assert_eq!(read_slabs(&spans), [(10, 30)]);
    }

    #[test]
    fn insslab_into_air_gap_above_slab() {
        let mut spans = build_b2(&[(50, 70)]);
        insslab(&mut spans, 10, 30);
        assert_eq!(read_slabs(&spans), [(10, 30), (50, 70)]);
    }

    #[test]
    fn insslab_into_air_gap_between_slabs() {
        let mut spans = build_b2(&[(10, 20), (60, 70)]);
        insslab(&mut spans, 30, 50);
        assert_eq!(read_slabs(&spans), [(10, 20), (30, 50), (60, 70)]);
    }

    #[test]
    fn insslab_into_air_gap_below_all_slabs() {
        let mut spans = build_b2(&[(10, 20)]);
        insslab(&mut spans, 30, 50);
        assert_eq!(read_slabs(&spans), [(10, 20), (30, 50)]);
    }

    #[test]
    fn insslab_extend_top_of_slab() {
        let mut spans = build_b2(&[(50, 70)]);
        insslab(&mut spans, 30, 60);
        assert_eq!(read_slabs(&spans), [(30, 70)]);
    }

    #[test]
    fn insslab_extend_bot_of_slab() {
        let mut spans = build_b2(&[(50, 70)]);
        insslab(&mut spans, 60, 80);
        assert_eq!(read_slabs(&spans), [(50, 80)]);
    }

    #[test]
    fn insslab_merge_into_last_slab_writes_sentinel() {
        // Repro for the multi-call set_spans / terrain bug: inserting
        // [105, 255) into a column that already has runs (100, 105)
        // and (255, MAXZDIM) should produce a single run (100,
        // MAXZDIM), with a clean sentinel at spans[2..]. Pre-fix this
        // left phantom values at spans[2..4] = (255, MAXZDIM) that
        // compilerle then re-emitted as overlapping garbage (column
        // collapsed to just the first voxel at z=100).
        // Built manually because `build_b2` rejects bot == MAXZDIM —
        // expandrle does produce that pattern (the bedrock slab at
        // z=255 lands as the last run with bot=MAXZDIM).
        let mut spans: Vec<i32> = vec![100, 105, 255, MAXZDIM, MAXZDIM, MAXZDIM];
        spans.resize(spans.len() + 32, 0); // slack
        insslab(&mut spans, 105, 255);
        // After: single run from z=100 to the bedrock-equivalent
        // sentinel. read_slabs returns until spans[i+1] < MAXZDIM.
        assert_eq!(spans[0], 100);
        assert!(
            spans[1] >= MAXZDIM,
            "expected merged run to extend to MAXZDIM, got spans[1] = {}",
            spans[1]
        );
        // The phantom (255, MAXZDIM) slot must NOT still be there —
        // sentinel-stamping in the merge branch fix.
        assert!(
            spans[2] >= MAXZDIM,
            "spans[2] should be sentinel, got {} (pre-fix this was 255 from the un-shifted phantom slab)",
            spans[2]
        );
    }

    #[test]
    fn insslab_touch_top_merges() {
        // y1 == top of slab → adjacent insert merges (extends top).
        let mut spans = build_b2(&[(50, 70)]);
        insslab(&mut spans, 30, 50);
        assert_eq!(read_slabs(&spans), [(30, 70)]);
    }

    #[test]
    fn insslab_touch_bot_merges() {
        // y0 == bot of slab → adjacent insert merges (extends bot).
        let mut spans = build_b2(&[(50, 70)]);
        insslab(&mut spans, 70, 80);
        assert_eq!(read_slabs(&spans), [(50, 80)]);
    }

    #[test]
    fn insslab_merge_two_slabs() {
        let mut spans = build_b2(&[(10, 30), (50, 70)]);
        insslab(&mut spans, 20, 60);
        assert_eq!(read_slabs(&spans), [(10, 70)]);
    }

    #[test]
    fn insslab_engulf_inner_slabs() {
        let mut spans = build_b2(&[(10, 20), (30, 40), (50, 60)]);
        insslab(&mut spans, 5, 70);
        assert_eq!(read_slabs(&spans), [(5, 70)]);
    }

    #[test]
    fn insslab_engulf_then_keep_lower() {
        let mut spans = build_b2(&[(10, 20), (30, 40), (60, 80)]);
        insslab(&mut spans, 5, 50);
        assert_eq!(read_slabs(&spans), [(5, 50), (60, 80)]);
    }

    #[test]
    fn insslab_engulf_then_merge_lower() {
        let mut spans = build_b2(&[(10, 20), (30, 40), (60, 80)]);
        insslab(&mut spans, 5, 60);
        assert_eq!(read_slabs(&spans), [(5, 80)]);
    }

    #[test]
    fn insslab_chain_of_touching_inserts() {
        let mut spans = build_b2(&[]);
        insslab(&mut spans, 10, 20);
        insslab(&mut spans, 20, 30);
        insslab(&mut spans, 30, 40);
        assert_eq!(read_slabs(&spans), [(10, 40)]);
    }

    #[test]
    fn insslab_carve_then_insert_round_trip() {
        // Land on slab, carve the middle, fill it back: end result
        // is identical to the original.
        let original = [(10, 50)];
        let mut spans = build_b2(&original);
        delslab(&mut spans, 20, 30);
        assert_eq!(read_slabs(&spans), [(10, 20), (30, 50)]);
        insslab(&mut spans, 20, 30);
        assert_eq!(read_slabs(&spans), original);
    }

    #[test]
    fn insslab_into_sentinel_only_buffer_with_z_advance() {
        // Insert below an existing slab — z advances past slab[0].
        let mut spans = build_b2(&[(10, 20)]);
        insslab(&mut spans, 100, 150);
        assert_eq!(read_slabs(&spans), [(10, 20), (100, 150)]);
    }

    // ---- expandrle (CD.2.3) ------------------------------------------

    /// Strip the `[top, bot, top, bot, ..., MAXZDIM]` decoded shape
    /// off a `spans` produced by [`expandrle`]. Last bot is the sentinel
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
        // On-disk encoding for solid [0, MAXZDIM) — the on-disk fixture
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
        // Solid run = [64, MAXZDIM) — the format treats below-floor as
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
        // air gap above. We skip emitting a new
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
        // hole into the air gap (the `read_uind` shape matches the spans
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

    /// Build a sentinel-terminated all-air spans buffer for a neighbor.
    /// Sized with slack so compilerle's index walks don't run off
    /// the end (worst case = n0's voxel count + 1).
    fn all_air_neighbor() -> Vec<i32> {
        // Just the sentinel pair + slack.
        let mut buf = vec![MAXZDIM, MAXZDIM];
        buf.resize(buf.len() + MAXZDIM as usize, MAXZDIM);
        buf
    }

    /// Build a sentinel-terminated spans from a list of solid runs.
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
        // re-encode bit-equivalently in spans shape.
        let mut slab = vec![0u8, 10, (MAXZDIM - 1) as u8, 0];
        for z in 10..MAXZDIM {
            // Distinct color per z so we can verify exact output bytes.
            slab.extend_from_slice(&[z as u8, (z + 1) as u8, (z + 2) as u8, 0]);
        }

        // Decode to spans.
        let mut spans = vec![0i32; (MAXZDIM as usize) + 4];
        expandrle(&slab, &mut spans);
        assert_eq!(spans[0], 10);
        assert_eq!(spans[1], MAXZDIM);

        // Re-encode with all-air neighbors → no buried-voxel skip.
        let n_air = all_air_neighbor();
        let mut cbuf = vec![0u8; 1028];
        let mut colfunc_called = 0;
        let mut colfunc = |_x: i32, _y: i32, _z: i32| -> i32 {
            colfunc_called += 1;
            0
        };
        let written = compilerle(
            &spans,
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

        // And expandrle on the output reproduces the same spans.
        let mut b2_round = vec![0i32; (MAXZDIM as usize) + 4];
        expandrle(&cbuf[..written], &mut b2_round);
        assert_eq!(b2_round[0], 10);
        assert_eq!(b2_round[1], MAXZDIM);
    }

    #[test]
    fn compilerle_round_trip_two_solid_runs_with_cave() {
        // spans = [10, 30, 50, MAXZDIM] — one cave between two solid runs.
        // Build a synthetic original column that has full floor color
        // lists for both slabs; ceiling list for slab 1 is non-empty.
        // For simplicity construct via compilerle from an all-air-
        // neighbor first encode of the desired spans, then round-trip.

        // Step 1: build a SEED column. We'll compilerle from a
        // hand-rolled "all is colfunc" variant — colfunc returns z as
        // its color, deterministic.
        let dummy = vec![0u8, 0, (MAXZDIM - 1) as u8, 0];
        let mut dummy_full = dummy;
        dummy_full.extend(std::iter::repeat_n(0u8, (MAXZDIM as usize) * 4));

        let n_air = all_air_neighbor();
        let spans = b2_from_runs(&[(10, 30), (50, MAXZDIM)]);
        let mut seed = vec![0u8; 1028];
        let mut colfunc = |_x: i32, _y: i32, z: i32| -> i32 { z };
        let seed_len = compilerle(
            &spans,
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

        // Step 2: decode the seed back to spans.
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
            &spans,
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

        // Self column: solid [10, MAXZDIM). The spans convention has
        // the last real solid run extending to MAXZDIM (solid below is
        // implicit), so we never transition into the sentinel slab.
        let spans = b2_from_runs(&[(10, MAXZDIM)]);
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
            &spans,
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
                                // expandrle on output reproduces the spans shape (still solid
                                // from z=10 onward, despite the compressed encoding).
        let mut b2_round = vec![0i32; (MAXZDIM as usize) + 4];
        expandrle(&cbuf[..written], &mut b2_round);
        assert_eq!(b2_round[0], 10);
        assert_eq!(b2_round[1], MAXZDIM);
    }

    // ---- ScumCtx (CD.2.5) ---------------------------------------------

    /// Build a 1×1 Vxl with a single fully-solid column. Minimal slab
    /// encoding: 1 floor color = 8 bytes total.
    fn build_1x1_min_solid_vxl() -> Vxl {
        let column = vec![0u8, 0, 0, 0, 0xff, 0x80, 0x40, 0x20];
        let column_offset = vec![0u32, column.len() as u32].into_boxed_slice();
        Vxl {
            vsid: 1,
            ipo: [0.0; 3],
            ist: [1.0, 0.0, 0.0],
            ihe: [0.0, 0.0, 1.0],
            ifo: [0.0, 1.0, 0.0],
            data: column.into_boxed_slice(),
            column_offset,
            mip_base_offsets: Box::new([0, 2]),
            vbit: Box::new([]),
            vbiti: 0,
        }
    }

    #[test]
    fn scum2_no_edit_round_trip_1x1_minimal_column() {
        // Open a batch on a minimal-encoded 1×1 column, run scum2 +
        // finish without mutating, verify the column's spans shape is
        // preserved.
        let mut vxl = build_1x1_min_solid_vxl();
        vxl.reserve_edit_capacity(4096);

        let mut ctx = ScumCtx::new(&mut vxl);
        let _b2 = ctx.scum2(0, 0).expect("column 0,0 in bounds");
        ctx.finish();

        let column = vxl.column_data(0);
        let mut b2_after = vec![0i32; SPAN_STRIDE];
        expandrle(column, &mut b2_after);
        assert_eq!(b2_after[0], 0);
        assert_eq!(b2_after[1], MAXZDIM);
    }

    #[test]
    fn scum2_carve_edit_1x1_creates_air_gap() {
        // Carve a hole in a fully-solid column; verify the post-edit
        // spans reflects the carve.
        let mut vxl = build_1x1_min_solid_vxl();
        vxl.reserve_edit_capacity(4096);

        let mut ctx = ScumCtx::new(&mut vxl);
        ctx.set_colfunc(|_x, _y, _z| 0x80_60_40_20u32 as i32);
        {
            let spans = ctx.scum2(0, 0).expect("column 0,0 in bounds");
            // Carve [50, 100) to air.
            delslab(spans, 50, 100);
        }
        ctx.finish();

        let column = vxl.column_data(0);
        let mut b2_after = vec![0i32; SPAN_STRIDE];
        expandrle(column, &mut b2_after);
        // Two solid runs now: [0, 50) and [100, MAXZDIM).
        assert_eq!(b2_after[0], 0);
        assert_eq!(b2_after[1], 50);
        assert_eq!(b2_after[2], 100);
        assert_eq!(b2_after[3], MAXZDIM);
    }

    /// Build a 4×4 Vxl with all 16 columns sharing the same minimal
    /// fully-solid encoding. Useful for testing batch edits.
    fn build_4x4_min_solid_vxl() -> Vxl {
        const COL: [u8; 8] = [0, 0, 0, 0, 0xff, 0x80, 0x40, 0x20];
        let mut data = Vec::with_capacity(16 * 8);
        let mut offsets = Vec::with_capacity(17);
        for i in 0..16 {
            offsets.push((i * 8) as u32);
            data.extend_from_slice(&COL);
        }
        offsets.push((16 * 8) as u32);
        Vxl {
            vsid: 4,
            ipo: [0.0; 3],
            ist: [1.0, 0.0, 0.0],
            ihe: [0.0, 0.0, 1.0],
            ifo: [0.0, 1.0, 0.0],
            data: data.into_boxed_slice(),
            column_offset: offsets.into_boxed_slice(),
            mip_base_offsets: Box::new([0, 17]),
            vbit: Box::new([]),
            vbiti: 0,
        }
    }

    #[test]
    fn scum2_batch_edits_multiple_columns_same_row() {
        // Edit columns (1, 2) and (2, 2) — same y row. Both should
        // get the same carve.
        let mut vxl = build_4x4_min_solid_vxl();
        vxl.reserve_edit_capacity(8192);

        let mut ctx = ScumCtx::new(&mut vxl);
        ctx.set_colfunc(|_x, _y, _z| 0);
        {
            let spans = ctx.scum2(1, 2).unwrap();
            delslab(spans, 50, 100);
        }
        {
            let spans = ctx.scum2(2, 2).unwrap();
            delslab(spans, 50, 100);
        }
        ctx.finish();

        for x in [1, 2] {
            let idx = 2 * 4 + x;
            let mut b2_after = vec![0i32; SPAN_STRIDE];
            expandrle(vxl.column_data(idx), &mut b2_after);
            assert_eq!(b2_after[0], 0);
            assert_eq!(b2_after[1], 50);
            assert_eq!(b2_after[2], 100);
            assert_eq!(b2_after[3], MAXZDIM);
        }
        // Untouched columns retain their original spans.
        for x in [0, 3] {
            let idx = 2 * 4 + x;
            let mut b2_after = vec![0i32; SPAN_STRIDE];
            expandrle(vxl.column_data(idx), &mut b2_after);
            assert_eq!(b2_after[0], 0);
            assert_eq!(b2_after[1], MAXZDIM);
        }
    }

    #[test]
    fn scum2_batch_edits_across_rows() {
        // Edit column (1, 1) then column (1, 2) — y advances, prior
        // row gets flushed automatically.
        let mut vxl = build_4x4_min_solid_vxl();
        vxl.reserve_edit_capacity(8192);

        let mut ctx = ScumCtx::new(&mut vxl);
        ctx.set_colfunc(|_x, _y, _z| 0);
        {
            let spans = ctx.scum2(1, 1).unwrap();
            delslab(spans, 60, 80);
        }
        {
            let spans = ctx.scum2(1, 2).unwrap();
            delslab(spans, 60, 80);
        }
        ctx.finish();

        for y in [1, 2] {
            let idx = y * 4 + 1;
            let mut b2_after = vec![0i32; SPAN_STRIDE];
            expandrle(vxl.column_data(idx), &mut b2_after);
            assert_eq!(b2_after[0], 0);
            assert_eq!(b2_after[1], 60);
            assert_eq!(b2_after[2], 80);
            assert_eq!(b2_after[3], MAXZDIM);
        }
    }

    #[test]
    fn scum2_finish_without_any_edit_is_noop() {
        // Begin then immediately finish without any scum2 call.
        let mut vxl = build_1x1_min_solid_vxl();
        vxl.reserve_edit_capacity(4096);
        let original = vxl.column_data(0).to_vec();
        let ctx = ScumCtx::new(&mut vxl);
        ctx.finish();
        assert_eq!(vxl.column_data(0), &original[..]);
    }

    #[test]
    fn scum2_returns_none_for_out_of_bounds() {
        let mut vxl = build_1x1_min_solid_vxl();
        vxl.reserve_edit_capacity(4096);
        let mut ctx = ScumCtx::new(&mut vxl);
        assert!(ctx.scum2(-1, 0).is_none());
        assert!(ctx.scum2(0, -1).is_none());
        assert!(ctx.scum2(1, 0).is_none());
        assert!(ctx.scum2(0, 1).is_none());
        ctx.finish();
    }

    // ---- set_spans (CD.3) --------------------------------------------

    #[test]
    fn set_spans_empty_is_noop() {
        let mut vxl = build_1x1_min_solid_vxl();
        let original = vxl.column_data(0).to_vec();
        set_spans(&mut vxl, &[], None);
        // Nothing should have changed — and we never reserved edit
        // capacity, so this also tests that empty input is fully
        // short-circuited (no ScumCtx::new + assert).
        assert_eq!(vxl.column_data(0), &original[..]);
    }

    #[test]
    fn set_spans_single_carve_creates_air_gap() {
        let mut vxl = build_1x1_min_solid_vxl();
        vxl.reserve_edit_capacity(4096);
        set_spans(
            &mut vxl,
            &[Vspan {
                x: 0,
                y: 0,
                z0: 50,
                z1: 99,
            }],
            None,
        );
        let mut spans = vec![0i32; SPAN_STRIDE];
        expandrle(vxl.column_data(0), &mut spans);
        // Half-open exclusive end is z1+1 = 100.
        assert_eq!(spans[0], 0);
        assert_eq!(spans[1], 50);
        assert_eq!(spans[2], 100);
        assert_eq!(spans[3], MAXZDIM);
    }

    #[test]
    fn set_spans_multi_span_same_column_accumulates() {
        // Two non-overlapping carves on the same column. The
        // setspans correctness relies on the with_column dedup —
        // re-calling scum2 between spans would wipe the first carve.
        let mut vxl = build_1x1_min_solid_vxl();
        vxl.reserve_edit_capacity(4096);
        set_spans(
            &mut vxl,
            &[
                Vspan {
                    x: 0,
                    y: 0,
                    z0: 30,
                    z1: 49,
                },
                Vspan {
                    x: 0,
                    y: 0,
                    z0: 100,
                    z1: 119,
                },
            ],
            None,
        );
        let mut spans = vec![0i32; SPAN_STRIDE];
        expandrle(vxl.column_data(0), &mut spans);
        // Three solid runs: [0, 30), [50, 100), [120, MAXZDIM).
        assert_eq!(spans[0], 0);
        assert_eq!(spans[1], 30);
        assert_eq!(spans[2], 50);
        assert_eq!(spans[3], 100);
        assert_eq!(spans[4], 120);
        assert_eq!(spans[5], MAXZDIM);
    }

    #[test]
    fn set_spans_insert_color_fills_air() {
        // Start with a column that's already partly carved, fill the
        // air gap back in with a known color.
        let mut vxl = build_1x1_min_solid_vxl();
        vxl.reserve_edit_capacity(4096);
        // First carve [50, 100) to create air.
        set_spans(
            &mut vxl,
            &[Vspan {
                x: 0,
                y: 0,
                z0: 50,
                z1: 99,
            }],
            None,
        );
        // Now fill [60, 80) back to solid with a known color.
        const FILL: u32 = 0x80_aa_bb_cc;
        set_spans(
            &mut vxl,
            &[Vspan {
                x: 0,
                y: 0,
                z0: 60,
                z1: 79,
            }],
            Some(FILL),
        );
        let mut spans = vec![0i32; SPAN_STRIDE];
        expandrle(vxl.column_data(0), &mut spans);
        // Three solid runs: [0, 50), [60, 80), [100, MAXZDIM).
        assert_eq!(spans[0], 0);
        assert_eq!(spans[1], 50);
        assert_eq!(spans[2], 60);
        assert_eq!(spans[3], 80);
        assert_eq!(spans[4], 100);
        assert_eq!(spans[5], MAXZDIM);
    }

    #[test]
    fn set_spans_skips_out_of_bounds_silently() {
        let mut vxl = build_1x1_min_solid_vxl();
        vxl.reserve_edit_capacity(4096);
        set_spans(
            &mut vxl,
            &[Vspan {
                x: 7,
                y: 9,
                z0: 50,
                z1: 99,
            }],
            None,
        );
        // Column (0,0) untouched — out-of-bounds span had no effect.
        let mut spans = vec![0i32; SPAN_STRIDE];
        expandrle(vxl.column_data(0), &mut spans);
        assert_eq!(spans[0], 0);
        assert_eq!(spans[1], MAXZDIM);
    }

    #[test]
    fn set_spans_with_colfunc_z_dependent_colour() {
        // Insert solid with a colour that depends on z. To make every
        // voxel in [60, 80) exposed (and thus needing a colfunc call
        // each), use a 4x4 world with neighbors carved to air at the
        // same z range. Center column (1, 1) is then surrounded by
        // air on all 4 sides at [60, 80), giving compilerle a full
        // floor list to fill via the closure.
        let mut vxl = build_4x4_min_solid_vxl();
        vxl.reserve_edit_capacity(8192);
        // Step 1: carve [50, 100) on every column so the insert sits
        // in air on every side.
        let carve_spans: Vec<Vspan> = (0..4)
            .flat_map(|y| {
                (0..4).map(move |x| Vspan {
                    x,
                    y,
                    z0: 50,
                    z1: 99,
                })
            })
            .collect();
        set_spans(&mut vxl, &carve_spans, None);
        // Step 2: insert [60, 80) on the center column with a
        // z-dependent colour.
        set_spans_with_colfunc(
            &mut vxl,
            &[Vspan {
                x: 1,
                y: 1,
                z0: 60,
                z1: 79,
            }],
            SpanOp::Insert,
            |_x, _y, z| (0x80ff_ff00u32 as i32) | z,
        );
        // Walk column (1,1) and find the slab with z1=60; verify each
        // floor colour matches the closure's output.
        let idx = 4 + 1; // y=1, x=1 in a 4-wide world
        let column = vxl.column_data(idx);
        let mut v = 0usize;
        let mut found = false;
        loop {
            let nextptr = column[v];
            let z1 = column[v + 1];
            if z1 == 60 {
                let z1c = column[v + 2];
                assert_eq!(z1c, 79, "z1c");
                let n_voxels = usize::from(z1c) - usize::from(z1) + 1;
                for i in 0..n_voxels {
                    let off = v + 4 + i * 4;
                    let c = u32::from_le_bytes([
                        column[off],
                        column[off + 1],
                        column[off + 2],
                        column[off + 3],
                    ]);
                    let z = u32::from(z1) + (i as u32);
                    assert_eq!(
                        c,
                        0x80ff_ff00 | z,
                        "z={z}: expected colour {:#010x}, got {:#010x}",
                        0x80ff_ff00 | z,
                        c
                    );
                }
                found = true;
                break;
            }
            if nextptr == 0 {
                break;
            }
            v += usize::from(nextptr) * 4;
        }
        assert!(found, "did not find a slab with z1=60");
    }

    // ---- set_cube / set_rect / set_sphere (CD.4) ---------------------

    #[test]
    fn set_cube_carves_single_voxel() {
        let mut vxl = build_4x4_min_solid_vxl();
        vxl.reserve_edit_capacity(4096);
        set_cube(&mut vxl, 1, 1, 100, None);
        let mut spans = vec![0i32; SPAN_STRIDE];
        expandrle(vxl.column_data(4 + 1), &mut spans);
        // Solid runs: [0, 100), [101, MAXZDIM).
        assert_eq!(spans[0], 0);
        assert_eq!(spans[1], 100);
        assert_eq!(spans[2], 101);
        assert_eq!(spans[3], MAXZDIM);
    }

    #[test]
    fn set_cube_skips_oob() {
        let mut vxl = build_4x4_min_solid_vxl();
        vxl.reserve_edit_capacity(4096);
        // Negative x, oversized y, z >= MAXZDIM, all silently no-op.
        set_cube(&mut vxl, -1, 1, 100, None);
        set_cube(&mut vxl, 5, 1, 100, None);
        set_cube(&mut vxl, 1, 1, 256, None);
        // Column (1, 1) untouched.
        let mut spans = vec![0i32; SPAN_STRIDE];
        expandrle(vxl.column_data(4 + 1), &mut spans);
        assert_eq!(spans[0], 0);
        assert_eq!(spans[1], MAXZDIM);
    }

    #[test]
    fn set_rect_carves_aabb() {
        let mut vxl = build_4x4_min_solid_vxl();
        vxl.reserve_edit_capacity(8192);
        // Carve a 2x2x50 box at (1..=2, 1..=2, 50..=99).
        set_rect(&mut vxl, [1, 1, 50], [2, 2, 99], None);
        for y in 1..=2 {
            for x in 1..=2 {
                let idx = (y * 4 + x) as usize;
                let mut spans = vec![0i32; SPAN_STRIDE];
                expandrle(vxl.column_data(idx), &mut spans);
                assert_eq!(spans[0], 0, "col ({x},{y})");
                assert_eq!(spans[1], 50, "col ({x},{y})");
                assert_eq!(spans[2], 100, "col ({x},{y})");
                assert_eq!(spans[3], MAXZDIM, "col ({x},{y})");
            }
        }
        // Untouched corners still solid through the full column.
        for &(x, y) in &[(0, 0), (3, 3)] {
            let idx = (y * 4 + x) as usize;
            let mut spans = vec![0i32; SPAN_STRIDE];
            expandrle(vxl.column_data(idx), &mut spans);
            assert_eq!(spans[0], 0);
            assert_eq!(spans[1], MAXZDIM);
        }
    }

    #[test]
    fn set_rect_clamps_to_world() {
        let mut vxl = build_4x4_min_solid_vxl();
        vxl.reserve_edit_capacity(8192);
        // Box extends well past world bounds — clamps to [0, 3] in
        // each axis.
        set_rect(&mut vxl, [-10, -10, -10], [100, 100, 1000], None);
        // Every column carved over [0, MAXZDIM) → all-air.
        for idx in 0..16 {
            let mut spans = vec![0i32; SPAN_STRIDE];
            expandrle(vxl.column_data(idx), &mut spans);
            // delslab clamps z1 to MAXZDIM-1, leaving voxel at
            // z=MAXZDIM-1 solid. The spans reflects this: solid run
            // [255, MAXZDIM) only.
            assert_eq!(spans[0], 255, "col {idx}");
            assert_eq!(spans[1], MAXZDIM, "col {idx}");
        }
    }

    #[test]
    fn set_sphere_carves_centred_sphere() {
        // 4x4 world; sphere radius 1 carves a "+" pattern at z=128
        // (center voxel + 4 axis-adjacent + 2 z-axis voxels).
        let mut vxl = build_4x4_min_solid_vxl();
        vxl.reserve_edit_capacity(8192);
        set_sphere(&mut vxl, [1, 1, 128], 1, None);
        // Voxels carved at: (1,1,127), (1,1,128), (1,1,129) [z axis],
        // (0,1,128), (2,1,128) [x axis], (1,0,128), (1,2,128) [y axis].
        // Center column (1,1) has z range [127, 130) carved.
        let mut spans = vec![0i32; SPAN_STRIDE];
        expandrle(vxl.column_data(4 + 1), &mut spans);
        assert_eq!(spans[0], 0);
        assert_eq!(spans[1], 127);
        assert_eq!(spans[2], 130);
        assert_eq!(spans[3], MAXZDIM);
        // Adjacent column (0, 1) has only z=128 carved.
        let mut spans = vec![0i32; SPAN_STRIDE];
        expandrle(vxl.column_data(4), &mut spans);
        assert_eq!(spans[0], 0);
        assert_eq!(spans[1], 128);
        assert_eq!(spans[2], 129);
        assert_eq!(spans[3], MAXZDIM);
    }

    #[test]
    fn set_sphere_radius_zero_is_single_voxel() {
        let mut vxl = build_4x4_min_solid_vxl();
        vxl.reserve_edit_capacity(4096);
        set_sphere(&mut vxl, [1, 1, 100], 0, None);
        // Same as set_cube — only (1, 1, 100) carved.
        let mut spans = vec![0i32; SPAN_STRIDE];
        expandrle(vxl.column_data(4 + 1), &mut spans);
        assert_eq!(spans[0], 0);
        assert_eq!(spans[1], 100);
        assert_eq!(spans[2], 101);
        assert_eq!(spans[3], MAXZDIM);
    }

    #[test]
    fn set_sphere_with_colfunc_position_dependent_color() {
        // Carve a cave first (so the inserted sphere has air on every
        // side, exposing its surface voxels), then insert a sphere
        // with a colfunc that returns z in the low byte. Verify
        // (a) spans reflects the sphere shape and (b) the top exposed
        // voxel's colour is the colfunc output.
        //
        // Note: the encoder stores colours only for EXPOSED
        // voxels (top of run + skip-forward landings); buried voxels
        // in the middle of a slab don't get colfunc-derived colours
        // recorded. So we can only verify colours for voxels at the
        // run boundary or near them.
        let mut vxl = build_4x4_min_solid_vxl();
        vxl.reserve_edit_capacity(8192);
        // Carve [50, 200) on every column.
        set_rect(&mut vxl, [0, 0, 50], [3, 3, 199], None);
        // Insert a sphere of radius 2 at (1, 1, 128). Center column
        // gets z=126..130 inclusive (5 voxels) inserted.
        set_sphere_with_colfunc(&mut vxl, [1, 1, 128], 2, SpanOp::Insert, |_, _, z| {
            (0x80ff_ff00u32 as i32) | z
        });
        // (a) spans has the expected three solid runs.
        let mut spans = vec![0i32; SPAN_STRIDE];
        expandrle(vxl.column_data(4 + 1), &mut spans);
        assert_eq!(spans[0], 0, "spans first run top");
        assert_eq!(spans[1], 50, "spans first run bot");
        assert_eq!(spans[2], 126, "spans sphere run top");
        assert_eq!(spans[3], 131, "spans sphere run bot");
        assert_eq!(spans[4], 200, "spans third run top");
        assert_eq!(spans[5], MAXZDIM, "spans third run bot");

        // (b) top of the sphere (z=126) is exposed (air above from
        // the carve). Its colour is the FIRST byte of the slab's
        // floor list.
        let column = vxl.column_data(4 + 1).to_vec();
        let mut v = 0usize;
        let mut top_color = None;
        loop {
            let nextptr = column[v];
            let z1 = column[v + 1];
            if z1 == 126 {
                let off = v + 4;
                top_color = Some(u32::from_le_bytes([
                    column[off],
                    column[off + 1],
                    column[off + 2],
                    column[off + 3],
                ]));
                break;
            }
            if nextptr == 0 {
                break;
            }
            v += usize::from(nextptr) * 4;
        }
        assert_eq!(
            top_color,
            Some(0x80ff_ff7e),
            "exposed voxel at z=126 should have colfunc-derived colour"
        );
    }

    #[test]
    fn set_spans_4x4_batch_carves_each_listed_column() {
        // Sorted (y, x) ascending; each column gets the same carve.
        let mut vxl = build_4x4_min_solid_vxl();
        vxl.reserve_edit_capacity(8192);
        let spans: Vec<Vspan> = (0..4)
            .flat_map(|y| {
                (0..4).map(move |x| Vspan {
                    x,
                    y,
                    z0: 50,
                    z1: 99,
                })
            })
            .collect();
        set_spans(&mut vxl, &spans, None);
        // Every column should have the [50, 100) carve.
        for idx in 0..16 {
            let mut spans = vec![0i32; SPAN_STRIDE];
            expandrle(vxl.column_data(idx), &mut spans);
            assert_eq!(spans[0], 0, "col {idx}");
            assert_eq!(spans[1], 50, "col {idx}");
            assert_eq!(spans[2], 100, "col {idx}");
            assert_eq!(spans[3], MAXZDIM, "col {idx}");
        }
    }
}
