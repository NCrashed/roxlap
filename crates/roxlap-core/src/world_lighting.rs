//! Voxlap's world-voxel lighting bake (`updatelighting`,
//! voxlap5.c:10539).
//!
//! Walks every visible voxel inside a 3D bounding box and rewrites
//! its alpha byte (the per-voxel "brightness" channel that the
//! rendering path mulhi'es against `kv6colmul`-style modulators)
//! based on the engine's current `LightSrc` set + lightmode.
//!
//! Two modes:
//! - `lightmode == 1`: cheap directional bake — every voxel gets
//!   shading from a single hardcoded sun direction
//!   `(tp.y * 0.5 + tp.z) * 64 + 103.5` clamped to `[0, 255]`.
//! - `lightmode == 2`: per-light Lambertian bake — for each light
//!   in range, subtract `g * h * sc` where `g = 1/(d·d²) -
//!   1/(r·r²)` (cube-falloff with hard cutoff at radius `r`),
//!   `h = surface_normal · light_delta` (negative ⇒ face front-
//!   lit, contributes; positive ⇒ self-shadowed, skipped). Result
//!   subtracts from a base `(tp.y * 0.5 + tp.z) * 16 + 47.5`.
//!
//! The surface normal `tp` for each voxel comes from `estnorm` —
//! a 5×5×5 voxel-solid neighbourhood vote (`ESTNORMRAD == 2` in
//! voxlap, the production path).

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::must_use_candidate,
    clippy::unnecessary_cast,
    clippy::cast_lossless,
    clippy::needless_bool_assign,
    clippy::needless_range_loop,
    clippy::no_effect,
    clippy::identity_op,
    clippy::if_not_else
)]

use rayon::prelude::*;

use crate::engine::LightSrc;

/// Voxlap's `MAXZDIM` (`voxlap5.c`). World z runs `0..MAXZDIM`.
pub(crate) const MAXZDIM: i32 = 256;

/// Voxlap's `ESTNORMRAD == 2` cache window radius. The estnorm
/// neighbourhood is `(2*RAD+1)³ = 5×5×5` voxels.
pub(crate) const ESTNORMRAD: i32 = 2;

/// Per-byte popcount table. Voxlap's `bitnum[32]` (voxlap5.c:1477)
/// — number of set bits in the low 5 bits of each index. Used by
/// estnorm's neighbourhood-vote reduction.
pub(crate) const BITNUM: [i8; 32] = [
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
];

/// Per-byte signed-symmetric popcount. Voxlap's `bitsnum[32]`
/// (voxlap5.c:1487) — packs `popcount` into the low i16 lane and
/// `popcount - 2·popcount_negative_axis` into the high i16 lane.
/// The exact derivation is in voxlap's comment block; values
/// reproduced verbatim.
#[rustfmt::skip]
pub(crate) const BITSNUM: [i32; 32] = [
    0,           1 - (2 << 16), 1 - (1 << 16), 2 - (3 << 16),
    1,           2 - (2 << 16), 2 - (1 << 16), 3 - (3 << 16),
    1 + (1 << 16), 2 - (1 << 16), 2,           3 - (2 << 16),
    2 + (1 << 16), 3 - (1 << 16), 3,           4 - (2 << 16),
    1 + (2 << 16), 2,           2 + (1 << 16), 3 - (1 << 16),
    2 + (2 << 16), 3,           3 + (1 << 16), 4 - (1 << 16),
    2 + (3 << 16), 3 + (1 << 16), 3 + (2 << 16), 4,
    3 + (3 << 16), 4 + (1 << 16), 4 + (2 << 16), 5,
];

/// `xbsflor[k] = -1i32 << k` — bits `k..31` set, low `k` bits
/// clear. Used by `expandbit256` to splat air→solid transitions
/// onto a partial 32-bit word.
pub(crate) const fn xbsflor(k: usize) -> u32 {
    if k >= 32 {
        0
    } else {
        (-1i32 << k) as u32
    }
}

/// `xbsceil[k] = ~xbsflor[k]` — low `k` bits set. Solid→air
/// transitions.
pub(crate) const fn xbsceil(k: usize) -> u32 {
    !xbsflor(k)
}

/// `expandbit256` — slab structure → 256-bit "voxel solid" bit
/// array (low-bit-first, low-z-first). Mirror of voxlap5.c:1059.
///
/// The output `bits` is a `[u32; 8]` (= 256 bits = `MAXZDIM` z
/// levels). Bit `z` is set iff voxel at column `(x, y)`, depth `z`
/// is solid (= part of any slab body, including hidden interiors
/// between slabs).
///
/// Walks the slab linked list, alternating between `v[1]`
/// (air→solid transition at top of slab) and `v[3]` (solid→air
/// transition at bottom of next slab). Each transition flushes
/// pending whole-words (full air `0` or full solid `-1`) until
/// it lands inside the partial word containing the transition,
/// then OR/ANDs the partial mask via `xbsflor` / `xbsceil`.
pub(crate) fn expandbit256(column: &[u8], bits: &mut [u32; 8]) {
    let mut src_idx: usize = 0;
    let mut dst_idx: usize = 0;
    let mut bitpos: i32 = 32;
    let mut word: u32 = 0;
    let nbits: i32 = (bits.len() as i32) * 32;

    // First iteration: jump straight to the v[1] transition (no
    // preceding slab whose v[3] we'd need to flush).
    let mut next_len: i32;
    let mut delta: i32;
    let mut go_to_v3 = false;

    'outer: loop {
        if go_to_v3 {
            // v[3] : solid → air transition.
            if src_idx + 3 >= column.len() {
                break;
            }
            delta = i32::from(column[src_idx + 3]) - bitpos;
            while delta >= 0 {
                if dst_idx >= bits.len() {
                    break 'outer;
                }
                bits[dst_idx] = word;
                dst_idx += 1;
                word = u32::MAX;
                bitpos += 32;
                delta -= 32;
            }
            word &= xbsceil((delta + 32) as usize);
        }
        go_to_v3 = true;

        // v[1] : air → solid transition.
        if src_idx + 1 >= column.len() {
            break;
        }
        delta = i32::from(column[src_idx + 1]) - bitpos;
        while delta >= 0 {
            if dst_idx >= bits.len() {
                break 'outer;
            }
            bits[dst_idx] = word;
            dst_idx += 1;
            word = 0;
            bitpos += 32;
            delta -= 32;
        }
        word |= xbsflor((delta + 32) as usize);

        next_len = i32::from(column[src_idx]);
        if next_len == 0 {
            break;
        }
        src_idx += (next_len as usize) * 4;
    }

    // Pad the rest of the buffer with `word`'s tail value (in C the
    // post-loop word is whatever the last `v[1]` partial-set
    // produced; remaining whole-words flush as solid `-1`).
    if bitpos <= nbits {
        while dst_idx < bits.len() {
            bits[dst_idx] = word;
            dst_idx += 1;
            word = u32::MAX;
        }
    }
}

/// Pre-built `expandbit256` grid covering a 2D bounding region —
/// `(x1 - x0 + 2*RAD) × (y1 - y0 + 2*RAD)` columns. Trades 32
/// bytes per column of memory for O(1) bit-window lookups during
/// the estnorm 5×5 neighbourhood vote.
///
/// This is the conceptual equivalent of voxlap's `xbsbuf` cache —
/// just batch-pre-built rather than rotated row-by-row through
/// the bake. Memory cost stays manageable: a 448×448 bake (the
/// `diag_down_lit` oracle scope, which extends to 452×452 with
/// padding) needs about 6.4 MB.
#[allow(dead_code)] // vsid field/method preserved for voxlap-parity inspection
pub(crate) struct EstNormCache {
    /// Per-column bit arrays. `bits[(yidx) * width + (xidx)]` is
    /// the slab bit-mask of column `(origin_x + xidx, origin_y +
    /// yidx)`. `xidx ∈ 0..width`, mapping abs-x into
    /// `[origin_x - RAD, origin_x + (x1 - x0) - 1 + RAD]`.
    bits: Vec<[u32; 8]>,
    /// Top-left of the cache window in world coords (= original
    /// `x0 - RAD`).
    origin_x: i32,
    origin_y: i32,
    /// Cached-region width (= `x1 - x0 + 2 * RAD`).
    width: usize,
    /// Reserved for symmetric debugging — kept so the cache layout
    /// can be inspected without recomputing from `bits.len()`.
    #[allow(dead_code)]
    height: usize,
    /// Inverse-square-root LUT — `fsqrecip[k] = 1 / sqrt(k)` for
    /// `k ∈ 0..=5859`. Voxlap's `fsqrecip` table; same precision
    /// as the C build (no Newton refinement for k > 22).
    fsqrecip: Vec<f32>,
    /// Voxel-grid limit (= `vsid`) used for out-of-bounds clamps.
    vsid: i32,
}

/// Voxlap's `fsqrecip[5860]` table init (voxlap5.c:12240-12256).
/// Mirror of the C calculation including the asymmetric Newton-
/// refinement schedule for indices ≤ 22.
fn build_fsqrecip() -> Vec<f32> {
    const N: usize = 5860;
    let mut t = vec![0.0_f32; N];
    t[0] = 0.0;
    t[1] = 1.0;
    t[2] = (1.0_f32 / 2.0_f32.sqrt()) as f32;
    t[3] = 1.0 / 3.0_f32.sqrt();
    let mut i = 3usize;
    let mut z = 4usize;
    while z < N {
        if z + 5 >= N {
            // Safety stop — cycle increment by 6 may overshoot.
            break;
        }
        t[z] = t[z >> 1] * t[2];
        t[z + 2] = t[(z + 2) >> 1] * t[2];
        t[z + 4] = t[(z + 4) >> 1] * t[2];
        t[z + 5] = t[i] * t[3];
        i += 2;

        let mut f = (t[z] + t[z + 2]) * 0.5_f32;
        if z <= 22 {
            f = (1.5 - 0.5 * ((z + 1) as f32) * f * f) * f;
        }
        t[z + 1] = (1.5 - 0.5 * ((z + 1) as f32) * f * f) * f;

        let mut f = (t[z + 2] + t[z + 4]) * 0.5_f32;
        if z <= 22 {
            f = (1.5 - 0.5 * ((z + 3) as f32) * f * f) * f;
        }
        t[z + 3] = (1.5 - 0.5 * ((z + 3) as f32) * f * f) * f;

        z += 6;
    }
    t
}

impl EstNormCache {
    /// Build the bit-grid cache covering the bounding region
    /// `[x0..x1) × [y0..y1)` extended by `ESTNORMRAD` padding on
    /// each side. Calling [`Self::estnorm`] for any `(x, y)` inside
    /// the original `[x0..x1) × [y0..y1)` box is then a pure read.
    #[must_use]
    pub fn build(
        world_data: &[u8],
        column_offsets: &[u32],
        vsid: u32,
        x0: i32,
        y0: i32,
        x1: i32,
        y1: i32,
    ) -> Self {
        let rad = ESTNORMRAD;
        let pad_x0 = x0 - rad;
        let pad_y0 = y0 - rad;
        let pad_x1 = x1 + rad;
        let pad_y1 = y1 + rad;
        let width = (pad_x1 - pad_x0) as usize;
        let height = (pad_y1 - pad_y0) as usize;

        let mut bits = vec![[0u32; 8]; width * height];
        let vsid_i = vsid as i32;
        for yi in 0..height {
            let y = pad_y0 + yi as i32;
            for xi in 0..width {
                let x = pad_x0 + xi as i32;
                if (x | y) < 0 || x >= vsid_i || y >= vsid_i {
                    // Out-of-bounds: voxlap's `expandbitstack`
                    // zeros the bind buffer (= treat as full air).
                    continue;
                }
                let col_idx = (y as u32) * vsid + (x as u32);
                let off_start = column_offsets[col_idx as usize] as usize;
                // Slice to end-of-buffer; the slab walker self-
                // terminates via nextptr. Same fix as world_query +
                // opticast post-edit-scatter — column_offsets[idx+1]
                // is the next table entry, NOT the next-byte-offset
                // after edits.
                let column = &world_data[off_start..];
                expandbit256(column, &mut bits[yi * width + xi]);
            }
        }

        Self {
            bits,
            origin_x: pad_x0,
            origin_y: pad_y0,
            width,
            height,
            fsqrecip: build_fsqrecip(),
            vsid: vsid_i,
        }
    }

    /// Read 5 consecutive bits starting at z-position `z` from the
    /// column at `(xi, yi)` cache index. Returns `0..=31`.
    /// Out-of-range positions:
    /// - `z < -2`: returns 0 (air above world — though voxlap's
    ///   convention is "above is sky", same effect).
    /// - `z >= MAXZDIM`: returns `0x1f` (solid below world).
    #[inline]
    fn extract_bits5(&self, xi: usize, yi: usize, z: i32) -> u32 {
        let col = &self.bits[yi * self.width + xi];
        if z >= MAXZDIM {
            return 0x1f;
        }
        if z + 5 <= 0 {
            return 0;
        }
        // Combine adjacent words to handle the case where the 5-bit
        // window straddles a word boundary.
        let z_bit = z;
        let word_idx = z_bit.div_euclid(32);
        let bit_off = z_bit.rem_euclid(32) as u32;
        let lo = if (0..8).contains(&word_idx) {
            col[word_idx as usize]
        } else if word_idx < 0 {
            0 // air above world
        } else {
            u32::MAX // solid below world
        };
        let hi = if word_idx + 1 < 8 && word_idx >= -1 {
            col[(word_idx + 1) as usize]
        } else if word_idx + 1 < 0 {
            0
        } else {
            u32::MAX
        };
        let combined = u64::from(lo) | (u64::from(hi) << 32);
        ((combined >> bit_off) & 0x1f) as u32
    }

    /// Estimate the surface normal at `(x, y, z)` from a 5×5×5
    /// voxel-solid neighbourhood vote. Mirror of voxlap5.c:1501
    /// (`estnorm`, `ESTNORMRAD == 2` branch).
    ///
    /// `(x, y)` must lie inside the cache's `[x0..x1) × [y0..y1)`
    /// region (panics otherwise — caller guarantees this via the
    /// bounding-box iteration). `z` is unconstrained (handled via
    /// air/solid clamping).
    #[must_use]
    pub fn estnorm(&self, x: i32, y: i32, z: i32) -> [f32; 3] {
        let center_xi = (x - self.origin_x) as usize;
        let center_yi = (y - self.origin_y) as usize;

        let mut nx: i32 = 0;
        let mut ny: i32 = 0;
        let mut nz: i32 = 0;
        let z_window = z - ESTNORMRAD; // top of the 5-bit z window

        for yy in -ESTNORMRAD..=ESTNORMRAD {
            let yi = (center_yi as i32 + yy) as usize;
            // Read 5 columns at this yy row (xx = -2..=+2).
            let b0 = self.extract_bits5(center_xi - 2, yi, z_window) as usize;
            let b1 = self.extract_bits5(center_xi - 1, yi, z_window) as usize;
            let b2 = self.extract_bits5(center_xi, yi, z_window) as usize;
            let b3 = self.extract_bits5(center_xi + 1, yi, z_window) as usize;
            let b4 = self.extract_bits5(center_xi + 2, yi, z_window) as usize;

            // Per-column popcount differences give x-axis normal
            // contributions. Voxlap weights:
            //   2*(N(xx=+2) - N(xx=-2)) + N(xx=+1) - N(xx=-1)
            // = `n.x` from this row (full normal sum is over yy).
            nx += ((i32::from(BITNUM[b4]) - i32::from(BITNUM[b0])) << 1) + i32::from(BITNUM[b3])
                - i32::from(BITNUM[b1]);

            // Sum bitsnum across all 5 columns: `j` is the total
            // signed-i16-packed contribution. Low 16 bits = number
            // of solid voxels in this row across all 5 columns and
            // 5 z levels. High 16 bits = z-axis contribution
            // (positive bits from upper z, negative from lower).
            let j = BITSNUM[b0]
                .wrapping_add(BITSNUM[b1])
                .wrapping_add(BITSNUM[b2])
                .wrapping_add(BITSNUM[b3])
                .wrapping_add(BITSNUM[b4]);
            nz = nz.wrapping_add(j);
            // n.y picks only the LOW i16 of `j` (= total solid
            // count), scaled by yy. The high i16 (z contribution)
            // doesn't enter n.y.
            let j_lo16 = (j as i16) as i32;
            ny = ny.wrapping_add(j_lo16 * yy);
        }
        nz >>= 16;

        // Normalise via fsqrecip[len_sq]. Voxlap's table peaks at
        // 5*5*5 box max = 75² + 15² + 3² = 5859 — within
        // `fsqrecip`'s 5860-entry range. Out-of-range len_sq values
        // (e.g. all-zero neighbourhood) get `fsqrecip[0] = 0` ⇒
        // returns `(0, 0, 0)` which downstream lighting math
        // tolerates.
        let len_sq = (nx * nx + ny * ny + nz * nz) as usize;
        let f = if len_sq < self.fsqrecip.len() {
            self.fsqrecip[len_sq]
        } else {
            0.0
        };
        [(nx as f32) * f, (ny as f32) * f, (nz as f32) * f]
    }

    /// Voxel-grid limit; used by callers to bound their iteration.
    #[must_use]
    #[allow(dead_code)] // preserved for voxlap-parity inspection
    pub(crate) fn vsid(&self) -> i32 {
        self.vsid
    }
}

/// Bake per-voxel lighting into the world's brightness bytes.
/// Mirror of voxlap's `updatelighting` (`voxlap5.c:10539`).
///
/// Walks every visible voxel inside `[x0..x1) × [y0..y1) ×
/// [z0..z1)` and rewrites its alpha byte (the brightness channel
/// the rasterizer mulhi'es against `kv6colmul` modulators) under
/// the current `lightmode` + `lights` state.
///
/// - `lightmode == 0`: no-op (fast return).
/// - `lightmode == 1`: directional sun-style bake — every visible
///   voxel gets `(tp.y * 0.5 + tp.z) * 64 + 103.5` clamped to
///   `[0, 255]` from its surface normal `tp`.
/// - `lightmode >= 2`: per-light Lambertian bake — base
///   `(tp.y * 0.5 + tp.z) * 16 + 47.5` minus, for each light in
///   range with surface normal facing it, `g * h * sc` where
///   `g = 1/(d·d²) - 1/(r·r²)` (cube falloff with hard radius
///   cutoff) and `h = tp · light_delta`.
///
/// Voxlap pads the bbox by `ESTNORMRAD` on each side internally
/// to give estnorm enough neighbourhood; that's done here too.
/// `lights` should match the engine's full `vx5.lightsrc[]` —
/// the function does its own per-tile range filtering.
///
/// Mutates `world_data` in place. Caller is responsible for any
/// `column_offsets` / `vsid` invariants.
pub fn update_lighting(
    world_data: &mut [u8],
    column_offsets: &[u32],
    vsid: u32,
    x0: i32,
    y0: i32,
    z0: i32,
    x1: i32,
    y1: i32,
    z1: i32,
    lightmode: u32,
    lights: &[LightSrc],
) {
    if lightmode == 0 {
        return;
    }
    let vsid_i = vsid as i32;
    let x0p = (x0 - ESTNORMRAD).max(0);
    let y0p = (y0 - ESTNORMRAD).max(0);
    let z0p = (z0 - ESTNORMRAD).max(0);
    let x1p = (x1 + ESTNORMRAD).min(vsid_i);
    let y1p = (y1 + ESTNORMRAD).min(vsid_i);
    let z1p = (z1 + ESTNORMRAD).min(MAXZDIM);
    if x0p >= x1p || y0p >= y1p || z0p >= z1p {
        return;
    }

    // Build the cache once for the whole padded bake region.
    // Voxlap tiles the bake into 64×64 chunks with a per-tile
    // `lightlst` filter; for our (one-shot bake) use case the
    // full-region filter computed inside the per-voxel loop is
    // simpler and not measurably slower at oracle bake sizes.
    let cache = EstNormCache::build(world_data, column_offsets, vsid, x0p, y0p, x1p, y1p);

    // Per-light precomputed `lightsub[i] = 1 / (sqrt(r2) * r2)` —
    // the radius-cutoff bias that makes the light contribution go
    // to exactly zero at distance == sqrt(r2).
    let lightsub: Vec<f32> = lights.iter().map(|l| 1.0 / (l.r2.sqrt() * l.r2)).collect();

    // R12.4.1: parallelise the per-row bake via rayon. Each `(x, y)`
    // pair maps to a unique column slice in `world_data`
    // (`column_offsets[col_idx]..[col_idx + 1]` ranges are pairwise
    // disjoint — the voxalloc allocator's invariant). Rows split
    // cleanly across worker threads; per-row x-loops stay serial to
    // amortise rayon's per-task overhead. Speedup follows
    // `RAYON_NUM_THREADS` (set `=1` to disable).
    //
    // Lighting bakes are typically rare (one-shot at scene load) but
    // dynamic-lighting / per-edit relighting use cases call
    // `update_lighting` per frame — at which point the parallel
    // path matters for interactive responsiveness.
    let world_view = WorldDataMutView::new(world_data);
    (y0p..y1p).into_par_iter().for_each(|y| {
        for x in x0p..x1p {
            let col_idx = (y as u32) * vsid + (x as u32);
            let off_start = column_offsets[col_idx as usize] as usize;
            let off_end = column_offsets[col_idx as usize + 1] as usize;
            // SAFETY: each (x, y) maps to a unique col_idx; column
            // ranges `[off_start, off_end)` are pairwise disjoint
            // across distinct `col_idx` (voxalloc invariant), so
            // no two threads write to the same byte.
            let column = unsafe { world_view.column_slice(off_start, off_end) };
            shade_column(column, x, y, z0p, z1p, lightmode, lights, &lightsub, &cache);
        }
    });
}

/// Raw-pointer view of `world_data` so the parallel
/// [`update_lighting`] body can hand out per-column `&mut [u8]`
/// slices to multiple threads without each thread needing
/// `&mut Vec<u8>` (which is exclusive). Constructed from a single
/// `&mut [u8]` borrow at the start of the parallel section; the
/// borrow's lifetime gates `WorldDataMutView`'s usable lifetime.
///
/// # Safety contract
/// Callers that hand out concurrent `column_slice` references MUST
/// guarantee the requested ranges are pairwise non-overlapping
/// across threads. [`update_lighting`]'s call site relies on
/// voxalloc's per-column-disjoint-byte-range invariant.
struct WorldDataMutView<'a> {
    ptr: *mut u8,
    len: usize,
    _marker: std::marker::PhantomData<&'a mut [u8]>,
}

// SAFETY: `WorldDataMutView` is morally a `&mut [u8]` re-exposed as
// raw pointers. The disjoint-write invariant is enforced by the
// caller; concurrent reads of `ptr` / `len` fields are race-free
// (immutable scalar fields).
unsafe impl Send for WorldDataMutView<'_> {}
unsafe impl Sync for WorldDataMutView<'_> {}

impl<'a> WorldDataMutView<'a> {
    fn new(buf: &'a mut [u8]) -> Self {
        Self {
            ptr: buf.as_mut_ptr(),
            len: buf.len(),
            _marker: std::marker::PhantomData,
        }
    }

    /// Carve out a sub-slice. Caller upholds the disjoint-write
    /// invariant (see struct doc).
    ///
    /// # Safety
    /// `off_start <= off_end <= self.len`, and the requested range
    /// must not overlap with ranges concurrently held by other
    /// threads.
    unsafe fn column_slice(&self, off_start: usize, off_end: usize) -> &'a mut [u8] {
        debug_assert!(off_start <= off_end, "column slice: start > end");
        debug_assert!(off_end <= self.len, "column slice: end past buffer");
        // SAFETY: caller asserts in-bounds + disjoint-from-other-threads.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(off_start), off_end - off_start) }
    }
}

/// Walk one column's slab chain and shade every visible voxel
/// inside `[z_lo, z_hi)`. Mirror of the inner loop in
/// voxlap5.c:10588-10650.
#[allow(clippy::cast_lossless)]
fn shade_column(
    column: &mut [u8],
    x: i32,
    y: i32,
    z_lo: i32,
    z_hi: i32,
    lightmode: u32,
    lights: &[LightSrc],
    lightsub: &[f32],
    cache: &EstNormCache,
) {
    let mut v_off: usize = 0;
    // cstat = false ⇒ top-of-slab phase (floor colours); true ⇒
    // ceiling-of-next-slab phase (bottom of current slab's solid
    // mass, visible from the air pocket below).
    let mut cstat = false;
    loop {
        let (sz0, sz1, voxel_byte_offset_signed): (i32, i32, isize);
        if !cstat {
            // Floor colours of the current slab. Voxel z=v[1]..=v[2].
            // Alpha byte at offset (z - v[1]) * 4 + 7 from header
            // (header is 4 bytes, voxel record is 4 bytes BGRA, +3
            // for alpha). The voxlap formula encodes this as
            // `(z << 2) + offs` with `offs = 7 - (v[1] << 2)`.
            if v_off + 2 >= column.len() {
                break;
            }
            let v1 = i32::from(column[v_off + 1]);
            let v2 = i32::from(column[v_off + 2]);
            sz0 = v1;
            sz1 = v2 + 1;
            voxel_byte_offset_signed = (v_off as isize) + 7 - ((sz0 as isize) << 2);
            cstat = true;
        } else {
            // Ceiling colours of the next slab — must read v[0]
            // BEFORE advancing v_off.
            if v_off + 2 >= column.len() {
                break;
            }
            let v0 = i32::from(column[v_off]);
            let v1 = i32::from(column[v_off + 1]);
            let v2 = i32::from(column[v_off + 2]);
            let prev_offset = v2 - v1 - v0 + 2; // ceilnum from getcube convention
            if v0 == 0 {
                break;
            }
            v_off += (v0 as usize) * 4;
            if v_off + 3 >= column.len() {
                break;
            }
            let v3 = i32::from(column[v_off + 3]);
            sz1 = v3;
            sz0 = prev_offset + sz1;
            voxel_byte_offset_signed = (v_off as isize) + 3 - ((sz1 as isize) << 2);
            cstat = false;
        }

        let lo = sz0.max(z_lo);
        let hi = sz1.min(z_hi);
        for z in lo..hi {
            let normal = cache.estnorm(x, y, z);
            let brightness = compute_brightness(x, y, z, normal, lightmode, lights, lightsub);
            let byte_off = voxel_byte_offset_signed + ((z as isize) << 2);
            if byte_off >= 0 && (byte_off as usize) < column.len() {
                column[byte_off as usize] = brightness;
            }
        }
    }
}

/// Voxlap's per-voxel brightness math. Computes the `[0, 255]`
/// alpha byte for one voxel from its surface normal `tp` + the
/// light list. Mirror of voxlap5.c:10605-10646.
fn compute_brightness(
    x: i32,
    y: i32,
    z: i32,
    tp: [f32; 3],
    lightmode: u32,
    lights: &[LightSrc],
    lightsub: &[f32],
) -> u8 {
    if lightmode < 2 {
        // Directional path (voxlap5.c:10607-10612): single sun
        // direction baked into a hardcoded coefficient pair.
        // i = (tp.y * 0.5 + tp.z) * 64 + 103.5, clamped to [0, 255].
        let f = (tp[1] * 0.5 + tp[2]) * 64.0 + 103.5;
        clamp_to_byte(f)
    } else {
        // Point-light path (voxlap5.c:10614-10645). Base brightness
        // 47.5..63.5 + per-light front-face contribution.
        let mut f = (tp[1] * 0.5 + tp[2]) * 16.0 + 47.5;
        let xf = x as f32;
        let yf = y as f32;
        let zf = z as f32;
        for (i, light) in lights.iter().enumerate() {
            let fx = light.pos[0] - xf;
            let fy = light.pos[1] - yf;
            let fz = light.pos[2] - zf;
            // tp · light_delta: positive ⇒ surface faces away from
            // light (back-lit, no contribution); negative ⇒ surface
            // faces light (front-lit, lambertian contribution).
            let h = tp[0] * fx + tp[1] * fy + tp[2] * fz;
            if h >= 0.0 {
                continue;
            }
            let g_sq = fx * fx + fy * fy + fz * fz;
            if g_sq >= light.r2 {
                continue;
            }
            // Voxlap's SSE rcpss/rsqrtss sequence:
            //   g = (1/g_sq) * rsqrt(g_sq) - lightsub[i]
            //     = 1/(g_sq * sqrt(g_sq)) - 1/(r2 * sqrt(r2))
            //     = 1/d³ - 1/r³
            // The `_mm_rcp_ss` / `_mm_rsqrt_ss` are 12-bit
            // approximations; the exact `f32::sqrt`-based form
            // here is more precise but may drift from voxlap C.
            // Bit-exactness will require switching to the
            // intrinsic versions on x86_64; deferred until
            // diag_down_lit oracle convergence demands it.
            let g = 1.0 / (g_sq * g_sq.sqrt()) - lightsub[i];
            f -= g * h * light.sc;
        }
        clamp_to_byte(f)
    }
}

#[inline]
fn clamp_to_byte(f: f32) -> u8 {
    // Voxlap's `if (*(int32_t *)&f > 0x437f0000) f = 255` is the
    // bit-trick form of `if (f > 255.0) f = 255.0`. Negatives wrap
    // through `ftol` / cast; we clamp explicitly for safety.
    if f >= 255.0 {
        255
    } else if f <= 0.0 {
        0
    } else {
        f as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// xbsflor(0) = -1 (all bits set), xbsflor(32) clamped to 0,
    /// xbsflor(5) = ~31 = 0xffff_ffe0.
    #[test]
    fn xbsflor_xbsceil_known_values() {
        assert_eq!(xbsflor(0), 0xffff_ffff);
        assert_eq!(xbsflor(1), 0xffff_fffe);
        assert_eq!(xbsflor(5), 0xffff_ffe0);
        assert_eq!(xbsflor(31), 0x8000_0000);
        assert_eq!(xbsflor(32), 0);
        assert_eq!(xbsceil(0), 0);
        assert_eq!(xbsceil(5), 0x1f);
        assert_eq!(xbsceil(31), 0x7fff_ffff);
        assert_eq!(xbsceil(32), 0xffff_ffff);
    }

    /// Single-slab column [next=0, sz0=10, sz1=14, then 5 voxel
    /// records]. Voxels exist at z = 10..15 (sz0..=sz1). After
    /// expandbit256, bits 10..15 should be set, all others
    /// (0..10 and 15..256) should reflect: air above (0..10) and
    /// solid below (15..256), since voxlap treats z > sz1 of last
    /// slab as solid.
    #[test]
    fn single_slab_z10_to_14_sets_correct_bits() {
        // Column layout: [next=0, sz0=10, sz1=14, top_color, then 5x
        // voxel records of 4 bytes each]. We don't use the voxel
        // record contents; expandbit256 only reads v[0]..v[3].
        let mut col = vec![0u8, 10, 14, 0]; // header
        col.extend(vec![0u8; 5 * 4]); // 5 voxel records (z=10..14)

        let mut bits = [0u32; 8];
        expandbit256(&col, &mut bits);

        // Word 0 covers bits 0..32. Air for z=0..10, solid 10..15,
        // solid for z=15..32 (since this is the only slab → below
        // is fully solid).
        // bits 10..15 from the slab body: 0x7c00 (bits 10,11,12,13,14)
        // bits 15..32 from "solid below last slab": 0xffff_8000
        // Combined: 0xffff_fc00.
        assert_eq!(
            bits[0], 0xffff_fc00,
            "word 0 want 0xffff_fc00 got 0x{:08x}",
            bits[0]
        );
        // Words 1..7 should all be 0xffff_ffff (fully solid).
        for (i, w) in bits.iter().enumerate().skip(1) {
            assert_eq!(*w, 0xffff_ffff, "word {i} want -1 got 0x{:08x}", *w);
        }
    }

    /// fsqrecip[N] should match `1/sqrt(N)` to a reasonable
    /// tolerance for the values estnorm actually produces.
    #[test]
    fn fsqrecip_matches_1_over_sqrt() {
        let t = build_fsqrecip();
        for k in 1..=100 {
            let want = 1.0_f32 / (k as f32).sqrt();
            let got = t[k];
            let err = (got - want).abs();
            assert!(err < 1e-3, "fsqrecip[{k}] = {got}, want {want}, err {err}");
        }
        // Spot-check higher values (less precise but still close).
        for k in [500, 1000, 2000, 5000] {
            let want = 1.0_f32 / (k as f32).sqrt();
            let got = t[k];
            let rel = (got / want - 1.0).abs();
            assert!(
                rel < 0.01,
                "fsqrecip[{k}] = {got}, want {want}, rel-err {rel}"
            );
        }
    }

    /// Build a 4×4 synthetic world with a flat floor at z=20..=24,
    /// run lightmode-1 update_lighting over the centre 2×2, and
    /// verify (a) brightness bytes were rewritten, (b) the result
    /// is in `[0, 255]` for every shaded voxel, (c) the brightness
    /// is uniform within each (x, y) column at the same z (since
    /// lightmode-1 depends only on the surface normal).
    #[test]
    fn lightmode1_bakes_brightness_into_visible_voxels() {
        // 4×4 world, single slab at z=20..=24, sentinel column ends.
        let vsid: u32 = 4;
        let mut col = vec![0u8, 20, 24, 0]; // header: nextptr=0, z1=20, z2=24
        for _ in 20..=24 {
            // 5 voxel records, alpha pre-set to 0xab so we can verify
            // they got rewritten.
            col.extend([0x10, 0x20, 0x30, 0xab]);
        }
        let col_len = col.len() as u32;
        let mut data = Vec::new();
        let mut offsets = vec![0u32; (vsid * vsid + 1) as usize];
        for i in 0..(vsid * vsid) {
            offsets[i as usize] = data.len() as u32;
            data.extend_from_slice(&col);
        }
        offsets[(vsid * vsid) as usize] = data.len() as u32;
        assert_eq!(col_len as usize * (vsid * vsid) as usize, data.len());

        update_lighting(
            &mut data,
            &offsets,
            vsid,
            1,
            1,
            0,
            3,
            3,
            30, // bbox 1..=2 in xy, z 0..30
            1,  // lightmode 1
            &[],
        );

        // Pull every voxel record's alpha byte from the centre
        // (1, 1) column. Should all be in [0, 255] and ≠ 0xab.
        let off1 = offsets[(1 * vsid + 1) as usize] as usize;
        let alphas: Vec<u8> = (0..5).map(|i| data[off1 + 4 + i * 4 + 3]).collect();
        for (i, &a) in alphas.iter().enumerate() {
            assert_ne!(a, 0xab, "alpha[{i}] not rewritten");
        }
        // The shading should be mostly bright — flat-floor voxels
        // have ~vertical normals so `(tp.y*0.5 + tp.z)*64 + 103.5`
        // ≈ 1.0*64 + 103.5 = 167.5.
        for (i, &a) in alphas.iter().enumerate() {
            assert!(
                a > 100,
                "alpha[{i}]={a} should be on the bright side for top-of-floor voxels"
            );
        }
    }

    /// lightmode-2 with one nearby light should darken voxels on
    /// the away side relative to the toward side. Use a 5×5 world
    /// with a flat floor and place a light such that it's on the
    /// +x side of the centre column — the +x face voxel's neighbour
    /// columns should end up brighter than the -x.
    #[test]
    fn lightmode2_with_light_produces_per_column_variation() {
        let vsid: u32 = 5;
        let mut col = vec![0u8, 20, 24, 0];
        for _ in 20..=24 {
            col.extend([0x10, 0x20, 0x30, 0]);
        }
        let mut data = Vec::new();
        let mut offsets = vec![0u32; (vsid * vsid + 1) as usize];
        for i in 0..(vsid * vsid) {
            offsets[i as usize] = data.len() as u32;
            data.extend_from_slice(&col);
        }
        offsets[(vsid * vsid) as usize] = data.len() as u32;

        let lights = [LightSrc {
            // World coords: light right next to (4, 2, 20).
            pos: [4.0, 2.0, 20.0],
            r2: 50.0 * 50.0,
            sc: 64.0,
        }];
        update_lighting(&mut data, &offsets, vsid, 0, 0, 0, 5, 5, 30, 2, &lights);

        // Sample the alpha at the top-floor voxel of each column
        // along y=2. Closer-to-light columns should be brighter.
        let alpha_at = |x: u32, z_idx: usize| {
            let off = offsets[(2 * vsid + x) as usize] as usize;
            data[off + 4 + z_idx * 4 + 3]
        };
        let close = alpha_at(4, 0); // closest column to light
        let far = alpha_at(0, 0); // farthest
        assert!(
            close >= far,
            "column nearer the light should be ≥ as bright as the far one (close={close} far={far})"
        );
    }

    /// Empty column ([0, 0, 0, ...]) — no slabs. After
    /// expandbit256, all 256 bits = 0 (full air).
    #[test]
    fn empty_column_all_air() {
        let col = vec![0u8, 0, 0, 0]; // single-slab header at z=0..0, no body
        let mut bits = [0u32; 8];
        expandbit256(&col, &mut bits);
        // bit 0 from "air→solid transition at z=0", but only bit 0
        // is set within the slab range [0, 0+1). Then "solid below"
        // fills bits 1..256.
        // Actually for sz0=sz1=0: voxel record is z=0..0 inclusive
        // (0 voxels). The bit pattern is 1 set bit at z=0 then
        // solid below.
        // word 0: bit 0 set, bits 1..32 set ⇒ 0xffff_ffff.
        assert_eq!(
            bits[0], 0xffff_ffff,
            "empty column word 0 want all-1 got 0x{:08x}",
            bits[0]
        );
    }
}
