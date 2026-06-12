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
    /// Slab-pool allocation bitmap — 1 bit per dword in [`Vxl::data`].
    /// A set bit marks the dword as belonging to an allocated column
    /// or mip-segment; clear bits are free space available to
    /// [`Vxl::voxalloc`]. Voxlap's `vbit` (`voxlap5.c:72`).
    ///
    /// Empty after [`parse`] — read-only Vxls have no allocator
    /// state. Call [`Vxl::reserve_edit_capacity`] to upgrade to a
    /// mutation-ready slab pool.
    pub vbit: Box<[u32]>,
    /// Roving allocator index, in dwords. Voxlap's `vbiti`
    /// (`voxlap5.c:72`). Advances past each successful
    /// [`Vxl::voxalloc`] to amortise the search.
    pub vbiti: u32,
}

impl Vxl {
    /// Raw slab bytes for mip-0 column `idx` (`idx < vsid * vsid`).
    /// Equivalent to `column_data_for_mip(0, idx)` — kept for the
    /// pre-multi-mip call sites.
    ///
    /// Length is recovered by walking the slab chain via [`slng`].
    /// This works both pre-edit (when columns are contiguous and
    /// `column_offset[idx + 1]` would also bound the column) and
    /// post-edit (when [`Vxl::voxalloc`] has scattered columns
    /// across the vbuf pool).
    #[must_use]
    pub fn column_data(&self, idx: usize) -> &[u8] {
        let start = self.column_offset[idx] as usize;
        let end = start + slng(&self.data[start..]);
        &self.data[start..end]
    }

    /// Packed BGRA colour of the **textured** voxel at `(x, y, z)`, or
    /// `None` if that voxel is air or an untextured (bedrock / interior
    /// / placeholder) cell. Walks the column's slab chain — both the
    /// floor colour list (`[z1, z1c]`) of each slab and the ceiling
    /// colour list stored in the previous slab's tail — mirroring
    /// voxlap's `vbuf` layout. Short-circuits on the first range that
    /// covers `z`, so it's an O(slabs) read with no decompression.
    ///
    /// The returned `u32` is the same packed colour the renderer reads
    /// (`B | G<<8 | R<<16 | A<<24`, with `A` carrying voxlap brightness);
    /// a zero-RGB cell (the `empty_chunk` placeholder) reads as `None`.
    /// Surface voxels — what a from-air raycast or click hits — are
    /// textured, so this returns their colour. Use [`column_data`] +
    /// the bitmap path for full interior/solidity semantics.
    ///
    /// [`column_data`]: Self::column_data
    #[must_use]
    pub fn voxel_color(&self, x: u32, y: u32, z: u32) -> Option<u32> {
        if x >= self.vsid || y >= self.vsid || z >= MAXZDIM as u32 {
            return None;
        }
        let zi = z as i32;
        let slab = self.column_data((y * self.vsid + x) as usize);
        let texel = |b: &[u8]| -> Option<u32> {
            let rgb = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            // Zero RGB = empty_chunk placeholder → treat as untextured.
            (rgb & 0x00ff_ffff != 0).then_some(rgb)
        };
        let mut v = 0usize;
        loop {
            // Floor colour list of the current slab: z ∈ [z1, z1c].
            let z_start = i32::from(slab[v + 1]);
            let z1c = i32::from(slab[v + 2]);
            if zi >= z_start && zi <= z1c {
                let off = v + 4 + ((zi - z_start) as usize) * 4;
                return texel(&slab[off..off + 4]);
            }
            let nextptr = slab[v];
            if nextptr == 0 {
                return None; // last slab, z not in its floor list
            }
            let (prev_z1, prev_z1c, prev_nextptr) = (z_start, z1c, i32::from(nextptr));
            v += usize::from(nextptr) * 4;
            // Ceiling colour list for the NEW slab — stored in the tail
            // of the previous slab's bytes (voxlap `vbuf` convention).
            let ze = i32::from(slab[v + 3]);
            let ceil_z_start = ze + prev_z1c - prev_z1 - prev_nextptr + 2;
            if zi >= ceil_z_start && zi < ze {
                let ceil_n = (ze - ceil_z_start) as usize;
                let ceil_start = v - ceil_n * 4;
                let off = ceil_start + ((zi - ceil_z_start) as usize) * 4;
                return texel(&slab[off..off + 4]);
            }
        }
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
    /// `< (vsid >> mip)²`. Length recovered via [`slng`] (works
    /// both pre-edit and after [`Vxl::voxalloc`]-driven scatter).
    ///
    /// # Panics
    ///
    /// Panics if `mip >= mip_count()` or `idx` is past this mip's
    /// column count.
    #[must_use]
    pub fn column_data_for_mip(&self, mip: u32, idx: usize) -> &[u8] {
        let table = self.column_offset_for_mip(mip);
        let start = table[idx] as usize;
        let end = start + slng(&self.data[start..]);
        &self.data[start..end]
    }

    /// Drop any built mip-1+ data, returning the Vxl to its
    /// post-`parse` single-mip shape. Cheap when already single-mip.
    ///
    /// **Stale-sentinel handling.** `column_offset[n_cols]` is set
    /// at chunk creation to the initial seed-data length and is
    /// **not** bumped when [`Vxl::voxalloc`] scatters columns into
    /// the edit pool past it. On a second `generate_mips` call
    /// against a post-edit chunk, trusting the stored sentinel
    /// would slice `self.data` back to its pre-edit-pool footprint
    /// and destroy every voxalloc'd column. Recover the real
    /// end-of-mip-0 by walking columns + summing [`slng`] lengths
    /// before slicing, and bump the sentinel to the recomputed
    /// value so downstream reads see a consistent view.
    fn reset_to_single_mip(&mut self) {
        let n_cols = (self.vsid as usize) * (self.vsid as usize);
        if self.mip_base_offsets.len() <= 2 {
            return;
        }
        let mip0_end_in_data: usize = (0..n_cols)
            .map(|i| {
                let start = self.column_offset[i] as usize;
                start + slng(&self.data[start..])
            })
            .max()
            .unwrap_or(0);
        self.data = self.data[..mip0_end_in_data].to_vec().into_boxed_slice();
        let mut new_offsets = self.column_offset[..=n_cols].to_vec();
        new_offsets[n_cols] = u32::try_from(mip0_end_in_data).expect("mip-0 end fits in u32");
        self.column_offset = new_offsets.into_boxed_slice();
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

    // ---- slab allocator (CD.2 cave-demo edit API) ---------------------
    //
    // Voxlap's `vbuf`/`vbit`/`voxalloc`/`voxdealloc` (`voxlap5.c:64-862`)
    // are a bitmap free-list allocator over the slab byte pool. Roxlap
    // re-uses [`Vxl::data`] as that pool: existing column bytes stay
    // put, headroom appended at the tail is the free region. Granularity
    // is one bit per dword (matches voxlap; column data is dword-aligned
    // by construction since slabs are 4-byte records).
    //
    // The allocator is opt-in. Read-only Vxls leave `vbit` empty and
    // never call alloc/dealloc.

    /// Upgrade this Vxl to a mutation-ready slab pool.
    ///
    /// Grows [`Vxl::data`] by `headroom_bytes` (rounded up to dword)
    /// and initialises [`Vxl::vbit`] with one bit per dword: bits in
    /// the prefix covered by every existing mip's columns are marked
    /// allocated; the appended headroom is free space available to
    /// [`Vxl::voxalloc`].
    ///
    /// Cost: O(total mip dwords) for the bit-marking pass, plus the
    /// data grow (one allocation + memcpy of existing bytes).
    ///
    /// After this call, [`serialize`] no longer round-trips byte-equally
    /// with the parsed file (the appended headroom is included in
    /// `data` and would be emitted as trailing zeros). Post-edit save
    /// requires walking columns in index order — landing in CD.2.5
    /// when on-disk save matters.
    ///
    /// # Panics
    ///
    /// Panics if the new total byte length does not fit in `u32`
    /// (would prevent column offsets from indexing into the pool).
    pub fn reserve_edit_capacity(&mut self, headroom_bytes: usize) {
        let headroom_aligned = (headroom_bytes + 3) & !3;
        let old_len = self.data.len();
        let new_len = old_len + headroom_aligned;
        u32::try_from(new_len).expect("vbuf size fits in u32");

        let mut new_data = Vec::with_capacity(new_len);
        new_data.extend_from_slice(&self.data);
        new_data.resize(new_len, 0);
        self.data = new_data.into_boxed_slice();

        let total_dwords = new_len / 4;
        let n_words = total_dwords.div_ceil(32);
        let mut vbit = vec![0u32; n_words].into_boxed_slice();

        // Mark every column's dword range as allocated. Iterates over
        // all built mips so mip-1+ tail data is preserved as
        // "allocated" (never repurposed by voxalloc, which finds runs
        // only in genuinely-clear bits).
        for mip in 0..self.mip_count() {
            let table = self.column_offset_for_mip(mip);
            for window in table.windows(2) {
                let lo = (window[0] / 4) as usize;
                let hi = (window[1] / 4) as usize;
                for d in lo..hi {
                    vbit[d >> 5] |= 1u32 << (d & 31);
                }
            }
        }

        self.vbit = vbit;
        self.vbiti = 0;
    }

    /// Allocate `n_bytes` of slab pool, returning a byte offset into
    /// [`Vxl::data`]. Voxlap's `voxalloc` (`voxlap5.c:841`).
    ///
    /// `n_bytes` MUST be a positive multiple of 4 (slab records are
    /// dword-aligned).
    ///
    /// # Panics
    ///
    /// - Panics if [`Vxl::reserve_edit_capacity`] was not called first
    ///   (`vbit` is empty).
    /// - Panics if `n_bytes == 0` or not a multiple of 4.
    /// - Panics if the pool is full (after two scan passes from
    ///   `vbiti` and from 0). Callers should size the headroom in
    ///   `reserve_edit_capacity` to absorb expected churn.
    pub fn voxalloc(&mut self, n_bytes: u32) -> u32 {
        assert!(
            !self.vbit.is_empty(),
            "voxalloc requires reserve_edit_capacity"
        );
        assert!(
            n_bytes > 0 && n_bytes % 4 == 0,
            "voxalloc n_bytes must be a positive multiple of 4 (got {n_bytes})"
        );

        let danum = n_bytes / 4;
        let total_dwords = u32::try_from(self.data.len() / 4).expect("pool dwords fit in u32");
        assert!(
            danum <= total_dwords,
            "voxalloc: requested span > pool size"
        );
        let vend = total_dwords - danum;

        for _badcnt in 0..2 {
            while self.vbiti < vend {
                if vbit_is_set(&self.vbit, self.vbiti) {
                    self.vbiti += danum;
                    continue;
                }
                // Walk back to the first dword of the surrounding
                // free run.
                let mut p0 = self.vbiti;
                while p0 > 0 && !vbit_is_set(&self.vbit, p0 - 1) {
                    p0 -= 1;
                }
                // Verify [p0, p0+danum) is entirely free by probing
                // backward from p0+danum-1 down to vbiti+1 (the
                // [p0..=vbiti] half is already proven free by the walk
                // above, plus vbiti's own bit which we know is clear).
                let mut p1 = p0 + danum - 1;
                let mut found = true;
                while p1 > self.vbiti {
                    if vbit_is_set(&self.vbit, p1) {
                        found = false;
                        break;
                    }
                    p1 -= 1;
                }
                if !found {
                    self.vbiti += danum;
                    continue;
                }

                self.vbiti = p0 + danum;
                for k in p0..self.vbiti {
                    self.vbit[(k >> 5) as usize] |= 1u32 << (k & 31);
                }
                return p0 * 4;
            }
            self.vbiti = 0;
        }
        panic!("voxalloc: vbuf full (cannot allocate {n_bytes} bytes)");
    }

    /// Free the slab at `byte_offset` (which must point at the start
    /// of a slab chain previously returned by [`Vxl::voxalloc`] or
    /// recorded in a column table). Voxlap's `voxdealloc`
    /// (`voxlap5.c:822`).
    ///
    /// Length is recovered by walking the slab chain via [`slng`] —
    /// the caller does not pass it.
    ///
    /// # Panics
    ///
    /// Panics if [`Vxl::reserve_edit_capacity`] was not called first.
    pub fn voxdealloc(&mut self, byte_offset: u32) {
        assert!(
            !self.vbit.is_empty(),
            "voxdealloc requires reserve_edit_capacity"
        );
        let len_bytes = u32::try_from(slng(&self.data[byte_offset as usize..]))
            .expect("slab length fits in u32");
        let i = byte_offset / 4;
        let j = (byte_offset + len_bytes) / 4;
        let i_word = (i >> 5) as usize;
        let j_word = (j >> 5) as usize;
        let i_bit = i & 31;
        let j_bit = j & 31;

        if i_word == j_word {
            // Range fits in a single word.
            let mask = p2m(j_bit) ^ p2m(i_bit);
            self.vbit[i_word] &= !mask;
        } else {
            self.vbit[i_word] &= p2m(i_bit);
            self.vbit[j_word] &= !p2m(j_bit);
            for w in (i_word + 1)..j_word {
                self.vbit[w] = 0;
            }
        }
    }
}

/// Slab-chain length helper. Voxlap's `slng` (`voxlap5.c:814`). Walks
/// the next-slab pointer chain rooted at `slab[0]` until the
/// terminator (`v[0] == 0`), then adds the last slab's floor-colour
/// bytes (`(z1c - z1 + 1) * 4`) plus the terminating slab's 4-byte
/// header. Returns total bytes used by the column.
///
/// # Panics
///
/// Panics on a malformed slab where the chain walk runs past the end
/// of `slab`, or where the last-slab header reaches outside the
/// slice. Both are caller errors — `slab` must be a valid voxlap slab
/// chain originating at `slab[0]`.
#[must_use]
pub fn slng(slab: &[u8]) -> usize {
    let mut i = 0usize;
    while slab[i] != 0 {
        i += usize::from(slab[i]) * 4;
    }
    let z1 = i32::from(slab[i + 1]);
    let z1c = i32::from(slab[i + 2]);
    let n_floor = usize::try_from((z1c - z1 + 1).max(0)).expect("n_floor non-negative");
    i + n_floor * 4 + 4
}

/// `p2m[k]` — bitmask with the low `k` bits set (`(1 << k) - 1`).
/// Voxlap's `p2m` table (a 32-entry static array). `k` MUST be in
/// `0..=31`.
#[inline]
fn p2m(k: u32) -> u32 {
    debug_assert!(k <= 31, "p2m takes 0..=31 (got {k})");
    if k == 0 {
        0
    } else {
        (1u32 << k) - 1
    }
}

#[inline]
fn vbit_is_set(vbit: &[u32], dword_idx: u32) -> bool {
    let word = (dword_idx >> 5) as usize;
    let bit = dword_idx & 31;
    (vbit[word] >> bit) & 1 != 0
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
        vbit: Box::new([]),
        vbiti: 0,
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
            vbit: Box::new([]),
            vbiti: 0,
        }
    }

    #[test]
    fn voxel_color_reads_floor_surface() {
        // 4 columns, each a floor voxel at z=10 with a distinct BGRA.
        let colours = [0x0080_ff00u32, 0x00ff_0000, 0x0000_ff00, 0x8040_2010];
        let vxl = build_synthetic_2x2(colours);
        for (i, &c) in colours.iter().enumerate() {
            #[allow(clippy::cast_possible_truncation)]
            let (x, y) = ((i % 2) as u32, (i / 2) as u32);
            assert_eq!(
                vxl.voxel_color(x, y, 10),
                Some(c),
                "surface colour, col {i}"
            );
            assert_eq!(vxl.voxel_color(x, y, 9), None, "air below the floor");
            assert_eq!(vxl.voxel_color(x, y, 11), None, "air above the floor");
        }
        // Out of bounds → None.
        assert_eq!(vxl.voxel_color(2, 0, 10), None, "x OOB");
        assert_eq!(vxl.voxel_color(0, 0, 256), None, "z OOB");
    }

    #[test]
    fn voxel_color_zero_rgb_is_untextured() {
        // A column whose only "voxel" has zero RGB (the empty-chunk
        // placeholder shape) reads as air, not a black surface.
        let vxl = build_synthetic_2x2([0x8000_0000, 0x8000_0000, 0x8000_0000, 0x8000_0000]);
        assert_eq!(vxl.voxel_color(0, 0, 10), None);
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
    fn generate_mips_twice_survives_post_edit_voxalloc_scatter() {
        // Regression for the `reset_to_single_mip` stale-sentinel bug
        // surfaced 2026-05-29 by the scene-demo streaming bake tracker.
        //
        // Setup: a 2x2 chunk, reserve edit capacity (grows `data`),
        // then `set_spans` to force `voxalloc` to scatter columns into
        // the edit pool past the original `column_offset[n_cols]`
        // sentinel. First `generate_mips` works (early-return in
        // `reset_to_single_mip` because `mip_base_offsets.len() == 2`),
        // but the second one used to slice `data[..stale_sentinel]`
        // and panic when subsequent column reads went OOB.
        //
        // After the fix, `reset_to_single_mip` walks columns + `slng`
        // to recompute the actual mip-0 end, so re-mip is sound.
        let mut vxl = build_synthetic_2x2([0xaa, 0xbb, 0xcc, 0xdd]);
        // Grow the data buffer with reserve_edit_capacity so
        // voxalloc has room to scatter columns.
        vxl.reserve_edit_capacity(64);
        // Now carve + re-insert at column (0, 0) so voxalloc moves
        // it into the edit pool past the original 32-byte mip-0
        // footprint. Use the edit-module helpers via the formats
        // crate.
        crate::edit::set_spans(
            &mut vxl,
            &[crate::edit::Vspan {
                x: 0,
                y: 0,
                z0: 5,
                z1: 15,
            }],
            None,
        );
        crate::edit::set_spans(
            &mut vxl,
            &[crate::edit::Vspan {
                x: 0,
                y: 0,
                z0: 5,
                z1: 15,
            }],
            Some(0x80_42_42_42),
        );

        // First mip build — must succeed.
        vxl.generate_mips(2);
        assert_eq!(vxl.mip_count(), 2, "first generate_mips → 2 mips");
        // Second build (the path the bug fired on). Must not panic
        // and still produce 2 mips.
        vxl.generate_mips(2);
        assert_eq!(
            vxl.mip_count(),
            2,
            "second generate_mips after post-edit scatter → still 2 mips"
        );
        // Third for good measure.
        vxl.generate_mips(2);
        assert_eq!(vxl.mip_count(), 2);
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

    // ---- slab allocator (CD.2.0/2.1/2.2) ------------------------------

    #[test]
    fn slng_single_slab_with_one_floor_voxel() {
        // [nextptr=0, z1=10, z1c=10, z0=0] + 4-byte colour = 8 bytes.
        let slab = [0u8, 10, 10, 0, 0xff, 0, 0, 0];
        assert_eq!(slng(&slab), 8);
    }

    #[test]
    fn slng_single_slab_with_three_floor_voxels() {
        // [nextptr=0, z1=10, z1c=12, z0=0] + 3 colours = 16 bytes.
        let slab = [0u8, 10, 12, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0];
        assert_eq!(slng(&slab), 16);
    }

    #[test]
    fn slng_single_slab_empty_floor() {
        // z1c < z1 → no floor colours; voxlap5.c stores n_floor as 0.
        // [nextptr=0, z1=10, z1c=9, z0=0] = 4 bytes only.
        let slab = [0u8, 10, 9, 0];
        assert_eq!(slng(&slab), 4);
    }

    #[test]
    fn slng_two_slab_chain() {
        // Slab 0: nextptr=2 (8 bytes) + 1 ceiling colour record + ...
        //   [2, 10, 11, 0, c0, c0, c0, c0]
        // Slab 1 (last): [0, 20, 22, 12] + 3 floor colours = 16 bytes
        //   total = 8 + 16 = 24.
        let slab = [
            2u8, 10, 11, 0, // slab 0 header
            0xaa, 0, 0, 0, // ceiling colour
            0, 20, 22, 12, // slab 1 (last) header
            0xbb, 0, 0, 0, // floor 0
            0xcc, 0, 0, 0, // floor 1
            0xdd, 0, 0, 0, // floor 2
        ];
        assert_eq!(slng(&slab), 24);
    }

    #[test]
    fn reserve_edit_capacity_grows_data_and_marks_existing() {
        let mut vxl = build_synthetic_2x2([0xaa, 0xbb, 0xcc, 0xdd]);
        let original_len = vxl.data.len();
        assert_eq!(original_len, 32);

        vxl.reserve_edit_capacity(64);

        // Data grew by aligned headroom.
        assert_eq!(vxl.data.len(), 32 + 64);
        // Existing bytes unchanged.
        assert_eq!(vxl.data[0..4], [0, 10, 10, 0]);
        // vbit sized = ceil(96 / 4 / 32) = 1.
        assert_eq!(vxl.vbit.len(), 1);
        // First 8 dwords (32 bytes / 4) marked allocated.
        // bits 0..=7 set → 0xff.
        assert_eq!(vxl.vbit[0], 0xff);
        // vbiti reset.
        assert_eq!(vxl.vbiti, 0);
    }

    #[test]
    fn reserve_edit_capacity_aligns_headroom_up_to_dword() {
        let mut vxl = build_synthetic_2x2([1, 2, 3, 4]);
        // Asking for 5 bytes should round up to 8.
        vxl.reserve_edit_capacity(5);
        assert_eq!(vxl.data.len(), 32 + 8);
    }

    #[test]
    fn voxalloc_returns_offset_in_headroom() {
        let mut vxl = build_synthetic_2x2([1, 2, 3, 4]);
        vxl.reserve_edit_capacity(64);
        let off = vxl.voxalloc(8);
        // First free dword is at byte offset 32 (just past the
        // 8 packed columns).
        assert_eq!(off, 32);
        // Bits 8 and 9 (dwords past existing data) now set.
        assert_eq!(vxl.vbit[0] & ((1 << 8) | (1 << 9)), (1 << 8) | (1 << 9));
    }

    #[test]
    fn voxalloc_successive_returns_non_overlapping() {
        let mut vxl = build_synthetic_2x2([1, 2, 3, 4]);
        vxl.reserve_edit_capacity(128);
        let a = vxl.voxalloc(8);
        let b = vxl.voxalloc(8);
        let c = vxl.voxalloc(16);
        assert_eq!(a, 32);
        assert_eq!(b, 40);
        assert_eq!(c, 48);
    }

    #[test]
    fn voxdealloc_clears_bits() {
        let mut vxl = build_synthetic_2x2([1, 2, 3, 4]);
        vxl.reserve_edit_capacity(128);
        let off = vxl.voxalloc(8);
        // off=32 → dword 8, len=8 → dwords 8..10 marked allocated.
        assert_eq!((vxl.vbit[0] >> 8) & 1, 1);
        assert_eq!((vxl.vbit[0] >> 9) & 1, 1);
        // Plant a valid slab at off so slng can recover the length.
        // [nextptr=0, z1=5, z1c=5, z0=0] + 1 colour = 8 bytes.
        vxl.data[off as usize..off as usize + 8]
            .copy_from_slice(&[0, 5, 5, 0, 0xa1, 0xa2, 0xa3, 0xa4]);
        vxl.voxdealloc(off);
        assert_eq!((vxl.vbit[0] >> 8) & 1, 0);
        assert_eq!((vxl.vbit[0] >> 9) & 1, 0);
    }

    #[test]
    fn voxdealloc_freed_region_reused_after_full_scan() {
        // Pool: 32 (existing) + 32 headroom = 64 bytes = 16 dwords.
        // Three 8-byte allocs fill dwords 8..14; vbiti reaches vend=14.
        // After freeing alloc #1, the next voxalloc resets vbiti to 0
        // and finds the freed dwords on the second pass.
        let mut vxl = build_synthetic_2x2([1, 2, 3, 4]);
        vxl.reserve_edit_capacity(32);
        let a = vxl.voxalloc(8);
        let _b = vxl.voxalloc(8);
        let _c = vxl.voxalloc(8);
        vxl.data[a as usize..a as usize + 8].copy_from_slice(&[0, 5, 5, 0, 0xa1, 0xa2, 0xa3, 0xa4]);
        vxl.voxdealloc(a);
        let reused = vxl.voxalloc(8);
        assert_eq!(reused, a, "freed region should be reused on rescan");
    }

    #[test]
    fn voxdealloc_cross_word_boundary() {
        let mut vxl = build_synthetic_2x2([1, 2, 3, 4]);
        // Build a ~200-byte pool: 32 (existing) + 192 headroom = 224 bytes
        // = 56 dwords. Words 0..=1 cover this — dword 31 is bit 31 of word 0,
        // dword 32 is bit 0 of word 1.
        vxl.reserve_edit_capacity(192);
        // Force vbiti up so successive allocs cross word boundary 31/32.
        // At this point bits 0..=7 are set (existing columns).
        // Allocate 24 dwords (96 bytes) starting at dword 8 → covers
        // dwords 8..=31 (still in word 0).
        let _ = vxl.voxalloc(96);
        assert_eq!(vxl.vbiti, 32);
        // Allocate another 16 bytes (4 dwords) — covers dwords 32..=35
        // in word 1.
        let off = vxl.voxalloc(16);
        assert_eq!(off, 32 * 4); // dword 32 → byte 128
                                 // Place a slab whose slng = 16 at off.
                                 // [nextptr=0, z1=0, z1c=2, z0=0] + 3 colours = 16 bytes.
        vxl.data[off as usize..off as usize + 16]
            .copy_from_slice(&[0, 0, 2, 0, 0xa, 0, 0, 0, 0xb, 0, 0, 0, 0xc, 0, 0, 0]);
        // The dword range to clear is [32, 36) — but it's within
        // word 1 entirely, so single-word path. Construct a different
        // case: free a span that crosses 31/32.
        // We need an allocation that spans dwords 30..=33 (cross word).
        // Reset: do a fresh upgrade.
        let mut vxl2 = build_synthetic_2x2([1, 2, 3, 4]);
        vxl2.reserve_edit_capacity(192);
        // Skip past existing 8 dwords with a 22-dword pad alloc.
        let pad = vxl2.voxalloc(22 * 4);
        assert_eq!(pad, 32);
        // Now allocate 16 bytes (4 dwords) at dword 30 — crosses 31/32.
        let cross = vxl2.voxalloc(16);
        assert_eq!(cross, 30 * 4);
        // Verify bits 30, 31 in word 0 and bits 0, 1 in word 1 set.
        assert!(
            (vxl2.vbit[0] >> 30) & 1 == 1
                && (vxl2.vbit[0] >> 31) & 1 == 1
                && vxl2.vbit[1] & 1 == 1
                && (vxl2.vbit[1] >> 1) & 1 == 1,
            "bits across word boundary should all be set"
        );
        // Plant a 16-byte slab so slng works.
        vxl2.data[cross as usize..cross as usize + 16]
            .copy_from_slice(&[0, 0, 2, 0, 0xa, 0, 0, 0, 0xb, 0, 0, 0, 0xc, 0, 0, 0]);
        vxl2.voxdealloc(cross);
        // Cross-boundary bits cleared.
        assert_eq!((vxl2.vbit[0] >> 30) & 1, 0);
        assert_eq!((vxl2.vbit[0] >> 31) & 1, 0);
        assert_eq!(vxl2.vbit[1] & 1, 0);
        assert_eq!((vxl2.vbit[1] >> 1) & 1, 0);
    }

    #[test]
    #[should_panic(expected = "voxalloc requires reserve_edit_capacity")]
    fn voxalloc_panics_without_reserve() {
        let mut vxl = build_synthetic_2x2([1, 2, 3, 4]);
        let _ = vxl.voxalloc(8);
    }

    #[test]
    #[should_panic(expected = "voxalloc n_bytes must be a positive multiple of 4")]
    fn voxalloc_panics_on_bad_size() {
        let mut vxl = build_synthetic_2x2([1, 2, 3, 4]);
        vxl.reserve_edit_capacity(64);
        let _ = vxl.voxalloc(7);
    }

    #[test]
    #[should_panic(expected = "voxalloc: vbuf full")]
    fn voxalloc_panics_when_pool_full() {
        let mut vxl = build_synthetic_2x2([1, 2, 3, 4]);
        // Headroom = 16 bytes = 4 dwords. Alloc 8 bytes twice → 4 dwords used,
        // 0 left. Third alloc must fail.
        vxl.reserve_edit_capacity(16);
        let _ = vxl.voxalloc(8);
        let _ = vxl.voxalloc(8);
        let _ = vxl.voxalloc(8); // panics
    }
}
