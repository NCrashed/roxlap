//! `.vxl` voxel-map format (Voxlap world / heightmap + slab columns).
//!
//! Reference: voxlaptest's `loadvxl` / `savevxl` (`voxlap5.c:3828` /
//! `:3887`) and the `vbuf` slab layout comment at
//! `voxlap5.c:75`. File layout (all multi-byte fields are little-
//! endian):
//!
//! ```text
//! offset  size                            description
//! 0x00    u32                             magic = 0x09072000
//! 0x04    u32                             xdim (must equal ydim — VSID)
//! 0x08    u32                             ydim
//! 0x0c    24 bytes                        ipo: starting position (3 × f64)
//! 0x24    24 bytes                        ist: right vector       (3 × f64)
//! 0x3c    24 bytes                        ihe: down vector        (3 × f64)
//! 0x54    24 bytes                        ifo: forward vector     (3 × f64)
//! 0x6c    variable                        column slab data (`vsid * vsid` columns)
//! ```
//!
//! Each column's slab data is a chain of slabs:
//!
//! ```text
//! slab header (4 bytes):
//!     byte 0  nextptr  — offset to next slab in dwords (== 0 for last slab)
//!     byte 1  z1       — top z of floor-colour list
//!     byte 2  z1c      — bottom z of floor-colour list MINUS 1
//!     byte 3  z0       — ceiling z (additional slabs); dummy in the first
//! followed by (per-voxel) 4-byte BGRA colour records.
//! ```
//!
//! Walker semantics, copied from `loadvxl`:
//!
//! - Non-last slab: total bytes = `nextptr * 4`. Advance by `nextptr * 4`.
//! - Last slab (`nextptr == 0`): total bytes = `4 + (z1c - z1 + 1) * 4`
//!   (header + floor colours; no ceiling colours).
//!
//! This module preserves column slab bytes verbatim in [`Vxl::data`] and
//! exposes a per-column byte-offset table in [`Vxl::column_offset`].
//! Iterating individual slabs (interpreting ceiling vs floor colour
//! lists) is left for a follow-up — the world fixture is large enough
//! that the test workload favours a flat byte representation, and the
//! engine port (R4) walks the bytes directly anyway.

use core::fmt;

use crate::bytes::{Cursor, OutOfBounds};

const MAGIC: u32 = 0x0907_2000;
const HEADER_LEN: usize = 4 + 4 + 4 + 4 * 24;

/// Parsed `.vxl` map. Round-trips byte-equally via [`parse`] +
/// [`serialize`].
#[derive(Debug, Clone)]
pub struct Vxl {
    /// Square map dimension. xdim and ydim are both equal to this in
    /// the file format (the loader rejects non-square maps).
    pub vsid: u32,
    /// Starting camera position (`dpoint3d`).
    pub ipo: [f64; 3],
    /// Right vector (`dpoint3d`).
    pub ist: [f64; 3],
    /// Down vector (`dpoint3d`).
    pub ihe: [f64; 3],
    /// Forward vector (`dpoint3d`).
    pub ifo: [f64; 3],
    /// Concatenated raw slab data for all columns across every built
    /// mip level. `parse` returns mip-0 data only; [`Vxl::generate_mips`]
    /// appends mip-1..mip-N onto the tail.
    pub data: Box<[u8]>,
    /// Per-column byte offsets into [`Vxl::data`], concatenated across
    /// every built mip level. Mip-N's sub-table lives at indices
    /// `mip_base_offsets[N]..mip_base_offsets[N + 1]` and contains
    /// `(vsid >> N)² + 1` entries — the trailing sentinel equals the
    /// byte offset where mip-(N+1)'s data starts (or `data.len()`
    /// for the topmost mip). After `parse`, the table is just mip-0
    /// (`vsid² + 1` entries) and the layout matches the historical
    /// single-mip shape, so callers passing `&vxl.column_offset`
    /// straight into the rasterizer keep working.
    pub column_offset: Box<[u32]>,
    /// `mip_base_offsets[mip]` is the index in [`Vxl::column_offset`]
    /// where mip-`mip`'s sub-table begins. `len() == mip_count + 1`;
    /// the trailing sentinel equals `column_offset.len()`. Initial
    /// state after `parse`: `[0, vsid² + 1]` (one mip).
    pub mip_base_offsets: Box<[usize]>,
}

impl Vxl {
    /// Raw slab bytes for mip-0 column `idx` (`idx < vsid * vsid`).
    /// Equivalent to `column_data_for_mip(0, idx)` — kept for the
    /// pre-multi-mip call sites.
    #[must_use]
    pub fn column_data(&self, idx: usize) -> &[u8] {
        let start = self.column_offset[idx] as usize;
        let end = self.column_offset[idx + 1] as usize;
        &self.data[start..end]
    }

    /// How many mip levels are currently built. Always `>= 1`
    /// (mip-0 is the parsed file). [`Vxl::generate_mips`] grows this
    /// up to its `max_mips` argument (capped by the world's
    /// `vsid` halving).
    ///
    /// # Panics
    ///
    /// Cannot panic in practice — `mip_base_offsets.len() - 1`
    /// fits in `u32` for any realistic `vsid`.
    #[must_use]
    pub fn mip_count(&self) -> u32 {
        u32::try_from(self.mip_base_offsets.len() - 1).expect("mip count fits in u32")
    }

    /// Per-column byte-offset sub-table for mip `mip`. Length
    /// `(vsid >> mip)² + 1`; the trailing sentinel is the byte
    /// offset where this mip's data ends inside [`Vxl::data`].
    ///
    /// # Panics
    ///
    /// Panics if `mip >= mip_count()`.
    #[must_use]
    pub fn column_offset_for_mip(&self, mip: u32) -> &[u32] {
        let mip_idx = mip as usize;
        let lo = self.mip_base_offsets[mip_idx];
        let hi = self.mip_base_offsets[mip_idx + 1];
        &self.column_offset[lo..hi]
    }

    /// Raw slab bytes for column `idx` at mip `mip`. `idx` must be
    /// `< (vsid >> mip)²`.
    ///
    /// # Panics
    ///
    /// Panics if `mip >= mip_count()` or `idx` is past this mip's
    /// column count.
    #[must_use]
    pub fn column_data_for_mip(&self, mip: u32, idx: usize) -> &[u8] {
        let table = self.column_offset_for_mip(mip);
        let start = table[idx] as usize;
        let end = table[idx + 1] as usize;
        &self.data[start..end]
    }

    /// Drop any built mip-1+ data, returning the Vxl to its
    /// post-`parse` single-mip shape. Cheap when already single-mip.
    fn reset_to_single_mip(&mut self) {
        let n_cols = (self.vsid as usize) * (self.vsid as usize);
        if self.mip_base_offsets.len() <= 2 {
            return;
        }
        let mip0_end_in_data = self.column_offset[n_cols] as usize;
        self.data = self.data[..mip0_end_in_data].to_vec().into_boxed_slice();
        self.column_offset = self.column_offset[..=n_cols].to_vec().into_boxed_slice();
        self.mip_base_offsets = Box::new([0, n_cols + 1]);
    }

    /// Build mip-1..mip-`max_mips` column data in place, mirroring
    /// voxlap's `genmipvxl` (`voxlap5.c:4710-4944`). Mip-0 is preserved.
    /// The loop halves dims each level and stops early when either
    /// dim drops to 1 or `max_mips` is reached.
    ///
    /// Calling this method more than once recomputes mips from
    /// scratch (matches voxlap's idempotent semantics — `genmipvxl`
    /// is invoked anywhere setcolumn-style mutations happen, and it
    /// always rebuilds against the current mip-0).
    ///
    /// # Panics
    ///
    /// Panics on a logic bug: the per-iteration `debug_assert_eq!`
    /// guards the invariant that the prior trailing
    /// `mip_base_offsets` entry equals the new sub-table's start.
    /// Production builds skip the assert.
    #[allow(clippy::missing_panics_doc)] // covered above
    pub fn generate_mips(&mut self, max_mips: u32) {
        self.reset_to_single_mip();
        if max_mips <= 1 {
            return;
        }

        // Outer mip loop. Voxlap5.c:4724-4932: while dims still halve
        // and we haven't reached `mipmax`, build mip-(mipnum) from
        // mip-(mipnum-1).
        let mut mipnum: u32 = 1;
        let mut src_vsid: u32 = self.vsid;
        let mut src_z_bound: i32 = MAXZDIM;
        while src_vsid > 1 && src_z_bound > 1 && mipnum < max_mips {
            let dst_vsid = src_vsid >> 1;
            let dst_z_bound = src_z_bound >> 1;

            // Snapshot the source mip's offsets/data before we mutate
            // self. The source mip is `mipnum - 1` (already built).
            let src_offsets_lo = self.mip_base_offsets[(mipnum - 1) as usize];
            let src_offsets_hi = self.mip_base_offsets[mipnum as usize];
            let src_offsets = self.column_offset[src_offsets_lo..src_offsets_hi].to_vec();

            // Build the new mip into fresh buffers; merge afterwards.
            let (new_data_segment, new_offsets) =
                build_mip_level(&self.data, &src_offsets, src_vsid, dst_vsid);

            // Splice into self. New offsets are returned in absolute
            // byte coords (the source-data prefix is unchanged so
            // they're already valid when treated as offsets into the
            // grown data buffer).
            let mut combined_data = self.data.to_vec();
            combined_data.extend_from_slice(&new_data_segment);
            self.data = combined_data.into_boxed_slice();

            let mut combined_offsets = self.column_offset.to_vec();
            combined_offsets.extend_from_slice(&new_offsets);
            self.column_offset = combined_offsets.into_boxed_slice();

            // The previous trailing entry was `src_offsets_hi` (the
            // mip-N sentinel and therefore the start of mip-N+1's
            // sub-table). Pushing the post-extension `column_offset`
            // length adds the mip-N+1 sentinel.
            debug_assert_eq!(
                *self
                    .mip_base_offsets
                    .last()
                    .expect("mip_base_offsets non-empty"),
                src_offsets_hi
            );
            let mut combined_mips = self.mip_base_offsets.to_vec();
            combined_mips.push(self.column_offset.len());
            self.mip_base_offsets = combined_mips.into_boxed_slice();

            mipnum += 1;
            src_vsid = dst_vsid;
            src_z_bound = dst_z_bound;
        }
    }
}

// ---------- multi-mip generation -----------------------------------------
//
// `build_mip_level` and friends below are a dense, cast-heavy port of
// voxlap5.c:4710-4944. The pedantic-cast lints fire on every line
// that mirrors a C `int32_t`/`char *` interaction; they're allowed
// scoped to each function rather than module-wide so the parser
// keeps its full lint coverage.

/// Maximum z-extent of a column — voxlap's `MAXZDIM` from
/// `voxlap5.h:10`. Each mip level halves this bound.
const MAXZDIM: i32 = 256;

/// Number of z-buckets in the per-cell colour-mixing accumulator.
/// Mip-N+1's z range is `MAXZDIM >> (N+1)`, so the largest first-
/// transition (mip-0 → mip-1) needs `MAXZDIM/2 = 128` buckets.
/// Sized once; later mips use a prefix.
const MIXC_BUCKETS: usize = (MAXZDIM as usize) >> 1;

/// Up to 8 source colours per bucket: 4 source columns × 2 source
/// z values per `(z >> 1)` bucket = 8.
const MIXC_LANES: usize = 8;

/// Voxlap5.c:4703-4707 (`qmulmip`). Multiplier table for averaging
/// `n` colour bytes after `*2 + 1`, then `>> 16`. Originally a
/// 4-lane packed `int64` (e.g. `0x7fff7fff7fff7fff`); we only need
/// the low 16 bits because the scalar fallback at
/// `voxlap5.c:4815-4837` reads the bottom u16 and broadcasts it.
const QMULMIP: [u32; 8] = [
    0x7fff, 0x4000, 0x2aaa, 0x2000, 0x1999, 0x1555, 0x1249, 0x1000,
];

/// Average up to 8 packed BGRA colours (voxlap5.c:4815-4837 scalar
/// translation). `n` is `1..=8`; `lanes[..n]` are the source
/// `int32_t` BGRA records.
#[allow(clippy::cast_sign_loss, clippy::cast_possible_wrap)]
fn average_packed_colours(lanes: &[i32], n: usize) -> i32 {
    debug_assert!((1..=MIXC_LANES).contains(&n));
    let mul = QMULMIP[n - 1];
    let mut sum = [0u32; 4];
    for &c in &lanes[..n] {
        let c = c as u32;
        sum[0] += c & 0xff;
        sum[1] += (c >> 8) & 0xff;
        sum[2] += (c >> 16) & 0xff;
        sum[3] += (c >> 24) & 0xff;
    }
    let mut out = 0u32;
    // Voxlap rounds via `(sum*2 + 1) * mul >> 16` then saturates to
    // `0..=255`. The unsigned arithmetic can't go negative, so only
    // the upper clamp is reachable.
    for (b, &s) in sum.iter().enumerate() {
        let v = s.wrapping_mul(2).wrapping_add(1).wrapping_mul(mul) >> 16;
        let v = v.min(255);
        out |= v << (b * 8);
    }
    out as i32
}

/// Build mip-N+1 column data + offsets from mip-N source. `data` is
/// the global byte buffer (mip-0..mip-N concatenated). `src_offsets`
/// is mip-N's per-column offset sub-table (length `src_vsid² + 1`).
///
/// Returns `(new_segment_bytes, new_offsets)`. `new_offsets` is sized
/// `dst_vsid² + 1` and gives ABSOLUTE byte offsets into the post-
/// extension data buffer (i.e. starts at `data.len()` and grows from
/// there).
#[allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::unnecessary_cast,
    clippy::needless_range_loop
)]
fn build_mip_level(
    data: &[u8],
    src_offsets: &[u32],
    src_vsid: u32,
    dst_vsid: u32,
) -> (Vec<u8>, Vec<u32>) {
    let src_vsid_us = src_vsid as usize;
    let dst_vsid_us = dst_vsid as usize;
    debug_assert_eq!(src_offsets.len(), src_vsid_us * src_vsid_us + 1);

    let dst_n_cols = dst_vsid_us * dst_vsid_us;
    let mut new_data: Vec<u8> = Vec::with_capacity(dst_n_cols * 8);
    let mut new_offsets: Vec<u32> = Vec::with_capacity(dst_n_cols + 1);
    let data_base = u32::try_from(data.len()).expect("data offset within u32");

    // Per-cell scratch reused across all (x, y) at this mip level.
    let mut mixc: Vec<i32> = vec![0; MIXC_BUCKETS * MIXC_LANES];
    let mut mixn: Vec<u8> = vec![0; MIXC_BUCKETS];
    // tbuf: per-column slab byte stream. Voxlap caps at MAXCSIZ=1028;
    // we grow as needed.
    let mut tbuf: Vec<u8> = Vec::with_capacity(1028);

    for y in 0..dst_vsid_us {
        for x in 0..dst_vsid_us {
            // Reset per-cell scratch.
            mixn.fill(0);
            tbuf.clear();
            tbuf.resize(4, 0); // header placeholder (voxlap5.c:4779: tbuf[3] = 0; n = 4)

            // 4 source-column byte offsets at (2x, 2y), (2x+1, 2y),
            // (2x, 2y+1), (2x+1, 2y+1). Voxlap's `oysiz`/`oxsiz` are
            // equal at every mip (square world).
            let src_idx = [
                (2 * y) * src_vsid_us + (2 * x),
                (2 * y) * src_vsid_us + (2 * x + 1),
                (2 * y + 1) * src_vsid_us + (2 * x),
                (2 * y + 1) * src_vsid_us + (2 * x + 1),
            ];
            let mut v_offset = [0usize; 4]; // current slab offset per source col
            for k in 0..4 {
                v_offset[k] = src_offsets[src_idx[k]] as usize;
            }

            // ---- Phase 1: flatten each source column's voxels into
            // `mixc`/`mixn` keyed on `z >> 1`. Voxlap5.c:4754-4778.
            let mut curz = [0i32; 4];
            let mut curzn = [[0i32; 4]; 4];
            for i in 0..4 {
                let mut tv = v_offset[i];
                // Initial state: top of floor and end-of-floor + 1.
                curz[i] = i32::from(data[tv + 1]);
                curzn[i][0] = curz[i];
                curzn[i][1] = i32::from(data[tv + 2]) + 1;

                loop {
                    let oz = i32::from(data[tv + 1]);
                    let z1c = i32::from(data[tv + 2]);
                    // Floor records at z = oz..=z1c.
                    let mut z = oz;
                    while z <= z1c {
                        let nz = (z >> 1) as usize;
                        let rec_off = tv + (((z - oz) << 2) + 4) as usize;
                        let rec = i32::from_le_bytes([
                            data[rec_off],
                            data[rec_off + 1],
                            data[rec_off + 2],
                            data[rec_off + 3],
                        ]);
                        let n_lane = mixn[nz] as usize;
                        mixc[nz * MIXC_LANES + n_lane] = rec;
                        mixn[nz] += 1;
                        z += 1;
                    }
                    // Carry-over for the post-advance ceiling loop.
                    let nextptr = i32::from(data[tv]);
                    let mut z_carry = (z - oz) - (nextptr - 1);
                    if nextptr == 0 {
                        break;
                    }
                    tv += (nextptr as usize) << 2;
                    let oz_new = i32::from(data[tv + 3]);
                    while z_carry < 0 {
                        let nz = ((z_carry + oz_new) >> 1) as usize;
                        // Read backwards: tv[z_carry * 4] sits inside
                        // the previous slab's tail, where voxlap
                        // stores the new slab's ceiling records.
                        let signed_off = (z_carry << 2) as isize;
                        let rec_off = (tv as isize + signed_off) as usize;
                        let rec = i32::from_le_bytes([
                            data[rec_off],
                            data[rec_off + 1],
                            data[rec_off + 2],
                            data[rec_off + 3],
                        ]);
                        let n_lane = mixn[nz] as usize;
                        mixc[nz * MIXC_LANES + n_lane] = rec;
                        mixn[nz] += 1;
                        z_carry += 1;
                    }
                    v_offset[i] = tv;
                }
                // After the flatten, restore v_offset[i] to the
                // FIRST slab — phase 2 walks v[besti] from the top
                // independently of phase 1's tv cursor.
                v_offset[i] = src_offsets[src_idx[i]] as usize;
            }

            // ---- Phase 2: 4-way z-merge → emit mip-N+1 slab bytes
            // into `tbuf`. Voxlap5.c:4779-4918.
            let mut cstat: i32 = 0;
            let mut oldn: usize = 0;
            let mut n: usize = 4;
            let mut z: i32 = i32::MIN; // 0x80000000 sentinel (voxlap5.c:4779)
            let mut cz: i32 = -1;

            loop {
                let oz = z;

                // Min of curz[0..4] using voxlap's branchless dance
                // (line 4785-4787).
                let mut besti = (((curz[1].wrapping_sub(curz[0])) as u32) >> 31) as i32;
                let i_alt =
                    ((((curz[3].wrapping_sub(curz[2])) as u32) >> 31) as i32).wrapping_add(2);
                let delta = curz[i_alt as usize].wrapping_sub(curz[besti as usize]);
                besti = besti.wrapping_add((delta >> 31) & (i_alt - besti));
                z = curz[besti as usize];
                if z >= MAXZDIM {
                    break;
                }

                // Maybe begin a new slab in tbuf.
                if cstat == 0 && (z >> 1) >= ((oz + 1) >> 1) {
                    if oz >= 0 {
                        tbuf[oldn] = ((n - oldn) >> 2) as u8;
                        tbuf[oldn + 2] = tbuf[oldn + 2].wrapping_sub(1);
                        // tbuf[n+3] = (oz + 1) >> 1 — z0 of the slab
                        // we're ABOUT to write next.
                        ensure_capacity(&mut tbuf, n + 4);
                        tbuf[n + 3] = (((oz + 1) >> 1) & 0xff) as u8;
                        oldn = n;
                        n += 4;
                    }
                    ensure_capacity(&mut tbuf, oldn + 4);
                    tbuf[oldn] = 0;
                    let initial = ((z >> 1) & 0xff) as u8;
                    tbuf[oldn + 1] = initial;
                    tbuf[oldn + 2] = initial;
                    cz = -1;
                }

                if cstat & 0x1111 != 0 {
                    let tbuf_z1c = i32::from(tbuf[oldn + 2]);
                    if (tbuf_z1c << 1) + 1 >= oz && cz < 0 {
                        // Continue the floor list: emit averaged
                        // colours per zz until we catch up.
                        while (i32::from(tbuf[oldn + 2]) << 1) < z {
                            let zz = i32::from(tbuf[oldn + 2]) as usize;
                            let n_vox = mixn[zz] as usize;
                            // Voxlap requires n_vox >= 1 here. If it
                            // somehow lands at 0, write zero — keeps
                            // the slab walker invariants intact.
                            let avg = if n_vox == 0 {
                                0
                            } else {
                                let lo = zz * MIXC_LANES;
                                average_packed_colours(&mixc[lo..lo + n_vox], n_vox)
                            };
                            mixn[zz] = 0;
                            ensure_capacity(&mut tbuf, n + 4);
                            tbuf[n..n + 4].copy_from_slice(&avg.to_le_bytes());
                            tbuf[oldn + 2] = tbuf[oldn + 2].wrapping_add(1);
                            n += 4;
                        }
                    } else {
                        if cz < 0 {
                            cz = oz >> 1;
                        } else if (cz << 1) + 1 < oz {
                            // Insert fake (single-voxel?) slab boundary.
                            tbuf[oldn] = ((n - oldn) >> 2) as u8;
                            tbuf[oldn + 2] = tbuf[oldn + 2].wrapping_sub(1);
                            ensure_capacity(&mut tbuf, n + 4);
                            tbuf[n] = 0;
                            let cz_byte = (cz & 0xff) as u8;
                            tbuf[n + 1] = cz_byte;
                            tbuf[n + 2] = cz_byte;
                            tbuf[n + 3] = cz_byte;
                            oldn = n;
                            n += 4;
                            cz = oz >> 1;
                        }
                        while (cz << 1) < z {
                            let zz = cz as usize;
                            let n_vox = mixn[zz] as usize;
                            let avg = if n_vox == 0 {
                                0
                            } else {
                                let lo = zz * MIXC_LANES;
                                average_packed_colours(&mixc[lo..lo + n_vox], n_vox)
                            };
                            mixn[zz] = 0;
                            ensure_capacity(&mut tbuf, n + 4);
                            tbuf[n..n + 4].copy_from_slice(&avg.to_le_bytes());
                            cz += 1;
                            n += 4;
                        }
                    }
                }

                // State machine update for besti (voxlap5.c:4887-4908).
                let bit_pos = (besti << 2) as i32;
                cstat = ((1i32 << bit_pos).wrapping_add(cstat)) & 0x3333;
                let state = (cstat >> bit_pos) & 3;
                let bi = besti as usize;
                match state {
                    0 => curz[bi] = curzn[bi][0],
                    1 => curz[bi] = curzn[bi][1],
                    2 => {
                        let tv = v_offset[bi];
                        if data[tv] == 0 {
                            curz[bi] = MAXZDIM;
                        } else {
                            let n_floor = i32::from(data[tv + 2]) - i32::from(data[tv + 1]) + 1;
                            let i_carry = n_floor - (i32::from(data[tv]) - 1);
                            let new_tv = tv + ((i32::from(data[tv]) as usize) << 2);
                            curz[bi] = i32::from(data[new_tv + 3]) + i_carry;
                            curzn[bi][3] = i32::from(data[new_tv + 3]);
                            curzn[bi][0] = i32::from(data[new_tv + 1]);
                            curzn[bi][1] = i32::from(data[new_tv + 2]) + 1;
                            v_offset[bi] = new_tv;
                        }
                    }
                    3 => curz[bi] = curzn[bi][3],
                    _ => unreachable!("state is masked to 0..=3"),
                }
            }

            // After loop: emit the final slab tail (voxlap5.c:4910-4918).
            tbuf[oldn + 2] = tbuf[oldn + 2].wrapping_sub(1);
            if cz >= 0 {
                tbuf[oldn] = ((n - oldn) >> 2) as u8;
                ensure_capacity(&mut tbuf, n + 4);
                tbuf[n] = 0;
                let cz_byte = (cz & 0xff) as u8;
                tbuf[n + 1] = cz_byte;
                tbuf[n + 2] = (cz - 1) as u8;
                tbuf[n + 3] = cz_byte;
                n += 4;
            }

            // Commit this column's slab bytes to new_data.
            let col_start = data_base
                + u32::try_from(new_data.len()).expect("mip data fits in u32 byte addressing");
            new_offsets.push(col_start);
            new_data.extend_from_slice(&tbuf[..n]);
        }
    }
    new_offsets.push(
        data_base + u32::try_from(new_data.len()).expect("mip data fits in u32 byte addressing"),
    );

    (new_data, new_offsets)
}

/// Grow `tbuf` so that index `len_inclusive - 1` is a valid write.
fn ensure_capacity(tbuf: &mut Vec<u8>, len_inclusive: usize) {
    if tbuf.len() < len_inclusive {
        tbuf.resize(len_inclusive, 0);
    }
}

/// Errors returned by [`parse`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    /// File too small to even contain the 108-byte header.
    TooSmall { got: usize },
    /// Magic bytes are not `0x09072000`.
    BadMagic { got: u32 },
    /// xdim and ydim disagree (file format requires square maps).
    NonSquareVsid { x: u32, y: u32 },
    /// A read of `need` bytes at offset `at` would run past EOF.
    Truncated { at: usize, need: usize },
    /// While walking column `idx`'s slab chain, the cursor at offset
    /// `at` would have run past the end of the column data region.
    BadColumn { idx: u32, at: usize },
    /// File total size > `u32::MAX`. The internal `column_offset`
    /// table uses `u32` because realistic maps fit comfortably.
    FileTooLarge { got: usize },
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::TooSmall { got } => write!(
                f,
                "vxl file too small ({got} bytes; need at least 108 byte header)"
            ),
            Self::BadMagic { got } => {
                write!(f, "vxl bad magic: got {got:#010x}, expected 0x09072000")
            }
            Self::NonSquareVsid { x, y } => write!(
                f,
                "vxl non-square dimensions: xdim={x}, ydim={y} (must be equal)"
            ),
            Self::Truncated { at, need } => {
                write!(f, "vxl truncated: need {need} bytes at offset {at}")
            }
            Self::BadColumn { idx, at } => write!(
                f,
                "vxl column {idx}: slab walker overran data region at offset {at}"
            ),
            Self::FileTooLarge { got } => write!(
                f,
                "vxl file size {got} exceeds {} bytes that this parser handles",
                u32::MAX
            ),
        }
    }
}

impl std::error::Error for ParseError {}

impl From<OutOfBounds> for ParseError {
    fn from(e: OutOfBounds) -> Self {
        Self::Truncated {
            at: e.at,
            need: e.need,
        }
    }
}

/// Parse a `.vxl` file's bytes into a [`Vxl`].
///
/// # Errors
///
/// Returns [`ParseError`] if `bytes` is shorter than the 108-byte
/// header, if the magic mismatches, if xdim ≠ ydim, if the file size
/// exceeds `u32::MAX` bytes, if a header field would run past EOF, or
/// if any column's slab chain runs off the end of the data region.
///
/// # Panics
///
/// Cannot panic on valid input: `pos` is bounded by `data.len()` which
/// the [`ParseError::FileTooLarge`] gate at the top of the function
/// proves fits in `u32`. The internal `expect` calls would only fire
/// on a logic bug in the walker.
///
/// # Examples
///
/// ```no_run
/// use roxlap_formats::vxl;
///
/// let bytes = std::fs::read("oracle.vxl")?;
/// let world = vxl::parse(&bytes)?;
/// println!(
///     "{}×{} VSID, {} mip levels, camera at {:?}",
///     world.vsid,
///     world.vsid,
///     world.mip_base_offsets.len() - 1,
///     world.ipo,
/// );
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn parse(bytes: &[u8]) -> Result<Vxl, ParseError> {
    if bytes.len() < HEADER_LEN {
        return Err(ParseError::TooSmall { got: bytes.len() });
    }
    if u32::try_from(bytes.len()).is_err() {
        return Err(ParseError::FileTooLarge { got: bytes.len() });
    }

    let mut cur = Cursor::new(bytes);
    let magic = cur.read_u32()?;
    if magic != MAGIC {
        return Err(ParseError::BadMagic { got: magic });
    }
    let xdim = cur.read_u32()?;
    let ydim = cur.read_u32()?;
    if xdim != ydim {
        return Err(ParseError::NonSquareVsid { x: xdim, y: ydim });
    }
    let vsid = xdim;

    let ipo = read_dpoint3d(&mut cur)?;
    let ist = read_dpoint3d(&mut cur)?;
    let ihe = read_dpoint3d(&mut cur)?;
    let ifo = read_dpoint3d(&mut cur)?;

    // Column data begins immediately after the header and runs to EOF.
    let data_start = cur.pos;
    let data: Box<[u8]> = bytes[data_start..].to_vec().into_boxed_slice();

    let n_cols = (vsid as usize) * (vsid as usize);
    let mut column_offset = Vec::with_capacity(n_cols + 1);
    let mut pos = 0usize;
    for i in 0..n_cols {
        column_offset.push(u32::try_from(pos).expect("data offset within u32"));
        loop {
            if pos + 4 > data.len() {
                return Err(ParseError::BadColumn {
                    idx: u32::try_from(i).unwrap_or(u32::MAX),
                    at: pos,
                });
            }
            let nextptr = data[pos];
            if nextptr == 0 {
                // Last slab. Length = 4 + n_floor * 4 where
                //   n_floor = max(0, z1c - z1 + 1).
                let z1 = data[pos + 1];
                let z1c = data[pos + 2];
                // n_floor = max(0, z1c - z1 + 1). Promote to i32 to
                // avoid u8 underflow when z1c == z1 - 1 (the "no floor
                // colours" case voxlap allows).
                let n_floor_signed = i32::from(z1c) - i32::from(z1) + 1;
                let n_floor = usize::try_from(n_floor_signed.max(0))
                    .expect("n_floor non-negative after .max(0)");
                let last_size = 4 + n_floor * 4;
                if pos + last_size > data.len() {
                    return Err(ParseError::BadColumn {
                        idx: u32::try_from(i).unwrap_or(u32::MAX),
                        at: pos,
                    });
                }
                pos += last_size;
                break;
            }
            let advance = usize::from(nextptr) * 4;
            // Guard against `nextptr * 4 < 4` which would loop forever.
            if advance < 4 {
                return Err(ParseError::BadColumn {
                    idx: u32::try_from(i).unwrap_or(u32::MAX),
                    at: pos,
                });
            }
            pos += advance;
        }
    }
    column_offset.push(u32::try_from(pos).expect("data offset within u32"));

    let mip_base_offsets = Box::new([0usize, n_cols + 1]);
    Ok(Vxl {
        vsid,
        ipo,
        ist,
        ihe,
        ifo,
        data,
        column_offset: column_offset.into_boxed_slice(),
        mip_base_offsets,
    })
}

/// Serialise a [`Vxl`] back to bytes. Round-trips byte-equally with
/// the input that produced this `Vxl` via [`parse`].
#[must_use]
pub fn serialize(vxl: &Vxl) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER_LEN + vxl.data.len());
    out.extend_from_slice(&MAGIC.to_le_bytes());
    out.extend_from_slice(&vxl.vsid.to_le_bytes());
    out.extend_from_slice(&vxl.vsid.to_le_bytes());
    write_dpoint3d(&mut out, &vxl.ipo);
    write_dpoint3d(&mut out, &vxl.ist);
    write_dpoint3d(&mut out, &vxl.ihe);
    write_dpoint3d(&mut out, &vxl.ifo);
    out.extend_from_slice(&vxl.data);
    out
}

fn read_dpoint3d(cur: &mut Cursor<'_>) -> Result<[f64; 3], OutOfBounds> {
    let mut out = [0.0f64; 3];
    for v in &mut out {
        let buf = cur.read_bytes(8)?;
        *v = f64::from_bits(u64::from_le_bytes([
            buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
        ]));
    }
    Ok(out)
}

fn write_dpoint3d(out: &mut Vec<u8>, p: &[f64; 3]) {
    for v in p {
        out.extend_from_slice(&v.to_bits().to_le_bytes());
    }
}

// --- tests --------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::io::Read;

    use flate2::read::GzDecoder;

    use super::*;

    /// Gzipped `oracle.vxl` produced by voxlaptest's oracle binary
    /// (run with `ROXLAP_SAVE_VXL=oracle.vxl`). At VSID = 2048 the raw
    /// file is ~37 MB but compresses to ~200 KB thanks to the largely
    /// uniform solid block surrounding the 448³ playable carve.
    const ORACLE_VXL_GZ: &[u8] = include_bytes!("../../../assets/oracle.vxl.gz");

    fn decode_oracle() -> Vec<u8> {
        let mut decoder = GzDecoder::new(ORACLE_VXL_GZ);
        let mut out = Vec::with_capacity(40 * 1024 * 1024);
        decoder.read_to_end(&mut out).expect("ungzip oracle.vxl.gz");
        out
    }

    #[test]
    fn parse_oracle_header() {
        let bytes = decode_oracle();
        let vxl = parse(&bytes).expect("parse oracle.vxl");
        // voxlaptest's fork uses VSID = 2048.
        assert_eq!(vxl.vsid, 2048);
        // The placeholder camera vectors written by oracle.c match the
        // values we set in tests/oracle/oracle.c's savevxl call. Compare
        // bit patterns to dodge clippy::float_cmp — these are exact
        // integer-valued doubles so the comparison is well-defined.
        let bits = |a: [f64; 3]| a.map(f64::to_bits);
        assert_eq!(bits(vxl.ipo), bits([1024.0, 1024.0, 128.0]));
        assert_eq!(bits(vxl.ist), bits([1.0, 0.0, 0.0]));
        assert_eq!(bits(vxl.ihe), bits([0.0, 0.0, 1.0]));
        assert_eq!(bits(vxl.ifo), bits([0.0, 1.0, 0.0]));
        // 2048 * 2048 = 4_194_304 columns; column_offset has one extra entry.
        assert_eq!(vxl.column_offset.len(), 4_194_304 + 1);
    }

    #[test]
    fn oracle_columns_partition_data_exactly() {
        let bytes = decode_oracle();
        let vxl = parse(&bytes).expect("parse oracle.vxl");
        // First column starts at offset 0, last sentinel equals data.len().
        assert_eq!(vxl.column_offset[0], 0);
        assert_eq!(
            vxl.column_offset[vxl.column_offset.len() - 1] as usize,
            vxl.data.len()
        );
        // Every column has at least one 4-byte slab header.
        let n_cols = (vxl.vsid as usize) * (vxl.vsid as usize);
        let min_col_len = (0..n_cols)
            .map(|i| vxl.column_data(i).len())
            .min()
            .expect("at least one column");
        assert!(min_col_len >= 4);
    }

    #[test]
    fn oracle_solid_corner_column_has_minimal_slab() {
        let bytes = decode_oracle();
        let vxl = parse(&bytes).expect("parse oracle.vxl");
        // The carve in tests/oracle/oracle.c is at x=800..1248, y=800..1248.
        // Column (0, 0) is well outside that range — solid block all the
        // way down, so its slab list should be the minimal one-slab form.
        let col = vxl.column_data(0);
        // Last slab nextptr == 0 must occur somewhere; the simplest valid
        // column is exactly one slab (header only or header + a few colours).
        // We assert column length is small (≤ 32 bytes — much less than a
        // carved column would be).
        assert!(
            col.len() <= 32,
            "solid corner column should be tiny; got {} bytes",
            col.len()
        );
    }

    #[test]
    fn oracle_roundtrips_byte_equal() {
        let bytes = decode_oracle();
        let vxl = parse(&bytes).expect("parse oracle.vxl");
        let out = serialize(&vxl);
        assert_eq!(out.len(), bytes.len(), "length differs");
        assert_eq!(out, bytes, "byte content differs");
    }

    #[test]
    fn parse_truncated_header_fails() {
        let r = parse(&[0u8; 32]);
        assert!(matches!(r, Err(ParseError::TooSmall { .. })));
    }

    #[test]
    fn parse_bad_magic_fails() {
        let mut bad = decode_oracle();
        bad[0] ^= 0xff;
        let r = parse(&bad);
        assert!(matches!(r, Err(ParseError::BadMagic { .. })));
    }

    /// Minimal valid Vxl with `vsid = 2`, four columns, each one slab
    /// with a single floor voxel at z = 10. Every column carries a
    /// distinct BGRA colour.
    fn build_synthetic_2x2(colours: [u32; 4]) -> Vxl {
        // Per-column slab bytes: header [0, 10, 10, 0] + colour record
        // (4 bytes) = 8 bytes per column. 4 columns = 32 bytes total.
        let mut data = Vec::with_capacity(32);
        for col_colour in colours {
            data.extend_from_slice(&[0, 10, 10, 0]);
            data.extend_from_slice(&col_colour.to_le_bytes());
        }
        let column_offset: Box<[u32]> = vec![0u32, 8, 16, 24, 32].into_boxed_slice();
        Vxl {
            vsid: 2,
            ipo: [0.0; 3],
            ist: [1.0, 0.0, 0.0],
            ihe: [0.0, 0.0, 1.0],
            ifo: [0.0, 1.0, 0.0],
            data: data.into_boxed_slice(),
            column_offset,
            mip_base_offsets: Box::new([0, 5]),
        }
    }

    #[test]
    fn generate_mips_skips_when_max_le_1() {
        let mut vxl = build_synthetic_2x2([1, 2, 3, 4]);
        let before_data_len = vxl.data.len();
        vxl.generate_mips(0);
        vxl.generate_mips(1);
        assert_eq!(vxl.mip_count(), 1);
        assert_eq!(vxl.data.len(), before_data_len);
        assert_eq!(vxl.mip_base_offsets.as_ref(), &[0usize, 5]);
    }

    #[test]
    fn generate_mips_2x2_produces_one_voxel_at_z5() {
        // Source: 4 columns, each one floor voxel at z = 10 with a
        // unique BGRA colour. Mip-1 collapses them to one column with
        // a single voxel at z = 5 (= 10 >> 1) coloured by the average.
        let colours = [
            0x0001_0101u32,
            0x0002_0202u32,
            0x0003_0303u32,
            0x0004_0404u32,
        ];
        let mut vxl = build_synthetic_2x2(colours);
        vxl.generate_mips(2);

        assert_eq!(vxl.mip_count(), 2);
        // Mip-0 sub-table preserved.
        assert_eq!(vxl.column_offset_for_mip(0).len(), 5);
        // Mip-1 has 1 column + sentinel = 2 entries.
        assert_eq!(vxl.column_offset_for_mip(1).len(), 2);

        // Single mip-1 column: header + 1 voxel = 8 bytes.
        let col = vxl.column_data_for_mip(1, 0);
        assert_eq!(col.len(), 8, "mip-1 column bytes: {col:?}");
        // Header: nextptr=0 (last slab), z1=z1c=5, z0=0 (dummy).
        assert_eq!(col[0], 0);
        assert_eq!(col[1], 5);
        assert_eq!(col[2], 5);
        assert_eq!(col[3], 0);

        // Voxel colour: average of inputs through voxlap's QMULMIP[3]
        // kernel. Sum of B/G/R is 1+2+3+4 = 10 per channel; sum of A
        // bytes is 0. avg(B/G/R) = ((10*2+1) * 0x2000) >> 16 = 2;
        // avg(A) = ((0*2+1) * 0x2000) >> 16 = 0.
        assert_eq!(col[4], 2, "B");
        assert_eq!(col[5], 2, "G");
        assert_eq!(col[6], 2, "R");
        assert_eq!(col[7], 0, "A");
    }

    #[test]
    fn generate_mips_idempotent_across_calls() {
        // Second invocation should yield bit-identical state.
        let colours = [0x10u32, 0x20, 0x30, 0x40];
        let mut a = build_synthetic_2x2(colours);
        let mut b = build_synthetic_2x2(colours);
        a.generate_mips(2);
        b.generate_mips(2);
        b.generate_mips(2);
        assert_eq!(a.data, b.data);
        assert_eq!(a.column_offset, b.column_offset);
        assert_eq!(a.mip_base_offsets, b.mip_base_offsets);
    }

    #[test]
    fn generate_mips_oracle_full_depth() {
        // Smoke test: oracle.vxl is 2048×2048, so 4 mips fit
        // (2048 → 1024 → 512 → 256). Verify each mip's sub-table
        // sizing and that mip-0 stays untouched.
        let bytes = decode_oracle();
        let mut vxl = parse(&bytes).expect("parse oracle.vxl");
        let mip0_data_len = vxl.column_offset[(2048 * 2048) as usize] as usize;
        let mip0_data_snapshot = vxl.data[..mip0_data_len].to_vec();

        vxl.generate_mips(4);
        assert_eq!(vxl.mip_count(), 4);
        for mip in 0..4u32 {
            let dim = (2048u32 >> mip) as usize;
            assert_eq!(
                vxl.column_offset_for_mip(mip).len(),
                dim * dim + 1,
                "mip-{mip} offset table length"
            );
        }
        // Mip-0 byte data must be untouched (multi-mip layout appends).
        assert_eq!(&vxl.data[..mip0_data_len], &mip0_data_snapshot[..]);
    }
}
