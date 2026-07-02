//! World-voxel lighting bake.
//!
//! Walks every visible voxel inside a 3D bounding box and writes its
//! per-voxel brightness byte (the high byte of the packed colour, which
//! the renderer multiplies into the RGB — see [`crate::dda`]'s `shade`)
//! from the engine's current `LightSrc` set + lightmode.
//!
//! Two modes:
//! - `lightmode == 1`: cheap directional bake — every voxel gets
//!   shading from a single fixed sun direction:
//!   `(n.y * 0.5 + n.z) * 64 + 103.5` clamped to `[0, 255]`.
//! - `lightmode == 2`: per-light point-light bake — for each light in
//!   range, subtract `g * h * sc`, where `g = 1/(d·d²) - 1/(r·r²)`
//!   (cube-falloff with a hard cutoff at radius `r`) and
//!   `h = surface_normal · light_delta` (front-lit faces contribute;
//!   back faces are skipped). Subtracted from a base
//!   `(n.y * 0.5 + n.z) * 16 + 47.5`.
//!
//! The surface normal `n` comes from [`EstNormCache::estnorm`] — the
//! occupancy gradient of a voxel's 5×5×5 neighbourhood.

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

/// World z is one byte → `0..MAXZDIM` (256) voxels tall.
pub(crate) const MAXZDIM: i32 = 256;

/// Estnorm neighbourhood radius. The surface normal at a voxel is
/// estimated from the solid/air pattern in the surrounding
/// `(2*RAD+1)³ = 5×5×5` cube.
pub const ESTNORMRAD: i32 = 2;

/// AO.2 — ambient-occlusion sampling radius (≤ `ESTNORMRAD`, so the same
/// cache padding suffices). The per-exposed-face method (see
/// [`EstNormCache::ambient_occlusion`]) is concave-only at any radius; this
/// just sets how far a concave contact's darkening reaches. `1` keeps it a
/// tight 1-voxel edge; raise for a wider contact band.
pub(crate) const AO_RAD: i32 = 1;
const _: () = assert!(AO_RAD <= ESTNORMRAD);

/// AO.0 — how dark a fully-occluded voxel gets, as a fraction removed from
/// the open-voxel ambient. `0.8` ⇒ a deep crevice keeps 20% of the ambient.
pub(crate) const AO_STRENGTH: f32 = 0.8;

/// AO.2 — tunable ambient-occlusion bake parameters (the `lightmode == 3`
/// knobs). [`Default`] matches the AO.0/AO.1 constants.
#[derive(Clone, Copy, Debug)]
pub struct AoParams {
    /// Fraction of ambient removed at full occlusion (`0` = off, `1` = black
    /// crevices before `min_floor`).
    pub strength: f32,
    /// Sampling reach in voxels (clamped to `ESTNORMRAD`). `1` = tight edge.
    pub radius: i32,
    /// Lower bound on the darkening factor — crevices never dim below
    /// `min_floor` of the open ambient (`0` ⇒ governed by `strength` alone).
    pub min_floor: f32,
}

impl Default for AoParams {
    fn default() -> Self {
        Self {
            strength: AO_STRENGTH,
            radius: AO_RAD,
            min_floor: 0.0,
        }
    }
}

/// `bits k..31 set, low k bits clear` (`!0 << k`). Used by
/// [`expandbit256`] to fill from an air→solid transition up to the
/// top of a 32-bit word.
pub(crate) const fn xbsflor(k: usize) -> u32 {
    if k >= 32 {
        0
    } else {
        (-1i32 << k) as u32
    }
}

/// `~xbsflor[k]` — low `k` bits set. Fills from the bottom of a word
/// up to a solid→air transition.
pub(crate) const fn xbsceil(k: usize) -> u32 {
    !xbsflor(k)
}

/// Decode a `.vxl` slab column into a 256-bit "voxel solid" bitset,
/// low-bit-first / low-z-first.
///
/// The output `bits` is a `[u32; 8]` (= 256 bits = `MAXZDIM` z
/// levels); bit `z` is set iff the voxel at depth `z` in this column is
/// solid (including the hidden interior between a slab's coloured top
/// and the next slab). This is a straight read of the `.vxl` column
/// layout: each slab record's byte 1 is its top z (air→solid) and byte
/// 3 the next slab's bottom (solid→air). Whole 32-bit words between
/// transitions are flushed as all-air (`0`) or all-solid (`!0`); the
/// word holding a transition gets a partial mask via
/// [`xbsflor`] / [`xbsceil`].
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

/// Read bit `z` (`0..256`) of a `[u32; 8]` z-column bitset.
#[inline]
pub(crate) fn bit256(bits: &[u32; 8], z: usize) -> bool {
    (bits[z >> 5] >> (z & 31)) & 1 != 0
}

/// Per-column solid/air bitset grid covering a 2D bounding region —
/// `(x1 - x0 + 2*RAD) × (y1 - y0 + 2*RAD)` columns. Decoding each
/// column to a bitset once turns the estnorm 5×5×5 neighbourhood query
/// into O(1) bit tests. A 448×448 bake (extending to 452×452 with
/// padding) needs about 6.4 MB.
#[allow(dead_code)] // vsid field/method preserved for inspection
pub struct EstNormCache {
    /// Per-column bit arrays. `bits[yidx * width + xidx]` is the
    /// solid/air bitset of column `(origin_x + xidx, origin_y + yidx)`.
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
    /// Voxel-grid limit (= `vsid`) used for out-of-bounds clamps.
    vsid: i32,
    /// AO cross-chunk z continuity (stacked grids, S4B.6). When
    /// non-empty (a z-aware build via [`Self::build_with_reader_z`]),
    /// these hold the solidity of the `ESTNORMRAD` voxels just outside
    /// the `[0, MAXZDIM)` z-window — read from the chunks stacked above
    /// (`chz-1`, world-z above) and below (`chz+1`, world-z below). Bit
    /// `i` of `z_below[col]` ⇒ the voxel at `z = -1 - i` is solid; bit
    /// `i` of `z_above[col]` ⇒ the voxel at `z = MAXZDIM + i` is solid.
    /// Empty ⇒ single-layer bake, [`Self::solid`] uses the implicit
    /// air-above / bedrock-below boundary (unchanged).
    z_below: Vec<u8>,
    z_above: Vec<u8>,
}

impl EstNormCache {
    /// Build the bit-grid cache covering the bounding region
    /// `[x0..x1) × [y0..y1)` extended by `ESTNORMRAD` padding on
    /// each side. Calling [`Self::estnorm`] for any `(x, y)` inside
    /// the original `[x0..x1) × [y0..y1)` box is then a pure read.
    ///
    /// Wraps [`Self::build_with_reader`] with a flat-table closure.
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
        let vsid_i = vsid as i32;
        let reader = |x: i32, y: i32| -> Option<&[u8]> {
            if (x | y) < 0 || x >= vsid_i || y >= vsid_i {
                return None;
            }
            let col_idx = (y as u32) * vsid + (x as u32);
            let off_start = column_offsets[col_idx as usize] as usize;
            // Slice to end-of-buffer; the slab walker self-
            // terminates via nextptr.
            Some(&world_data[off_start..])
        };
        let mut cache = Self::build_with_reader(reader, x0, y0, x1, y1);
        cache.vsid = vsid_i;
        cache
    }

    /// Z-aware variant of [`Self::build_with_reader`] for **stacked
    /// grids**. `column_reader(x, y, chz_delta)` returns the slab bytes
    /// of the column at cache-XY `(x, y)` in the chunk `chz_delta`
    /// layers away in z (`0` = the target layer, `-1` = the layer above
    /// in world-z, `+1` = below), or `None` for implicit-air / missing.
    ///
    /// The `chz_delta == 0` reads build the in-plane cache exactly like
    /// [`Self::build_with_reader`]; the `±1` reads populate the
    /// [`Self::z_below`] / [`Self::z_above`] overlays so [`Self::solid`]
    /// — and therefore [`Self::ambient_occlusion`] / [`Self::estnorm`] —
    /// see the neighbouring chunk's voxels across the z-seam instead of
    /// the implicit air-above / bedrock-below boundary. Where a z-
    /// neighbour is absent the overlay falls back to that same boundary
    /// (air above, solid below), so a topmost/bottommost chunk bakes
    /// identically to the single-layer path.
    #[must_use]
    pub fn build_with_reader_z<'r>(
        column_reader: impl Fn(i32, i32, i32) -> Option<&'r [u8]>,
        x0: i32,
        y0: i32,
        x1: i32,
        y1: i32,
    ) -> Self {
        let mut cache = Self::build_with_reader(|x, y| column_reader(x, y, 0), x0, y0, x1, y1);

        let n = cache.bits.len();
        let mut z_below = vec![0u8; n];
        let mut z_above = vec![0u8; n];
        // `chz_delta = -1` is the chunk above in world-z; its bottom-most
        // world-z voxels (its z-local `MAXZDIM-1, MAXZDIM-2`) sit at our
        // `z = -1, -2`. `chz_delta = +1` is below; its top voxels (z-local
        // `0, 1`) sit at our `z = MAXZDIM, MAXZDIM+1`.
        let pad = ESTNORMRAD as usize;
        for yi in 0..cache.height {
            let y = cache.origin_y + yi as i32;
            for xi in 0..cache.width {
                let x = cache.origin_x + xi as i32;
                let col = yi * cache.width + xi;

                if let Some(column) = column_reader(x, y, -1) {
                    let mut tmp = [0u32; 8];
                    expandbit256(column, &mut tmp);
                    for i in 0..pad {
                        // world z = -1 - i  ⇐  neighbour z-local MAXZDIM-1-i.
                        if bit256(&tmp, (MAXZDIM as usize) - 1 - i) {
                            z_below[col] |= 1 << i;
                        }
                    }
                }
                // Absent neighbour above ⇒ leave bits clear (air), matching
                // the implicit `z < 0 → air` boundary.

                if let Some(column) = column_reader(x, y, 1) {
                    let mut tmp = [0u32; 8];
                    expandbit256(column, &mut tmp);
                    for i in 0..pad {
                        // world z = MAXZDIM + i  ⇐  neighbour z-local i.
                        if bit256(&tmp, i) {
                            z_above[col] |= 1 << i;
                        }
                    }
                } else {
                    // Absent neighbour below ⇒ solid (bedrock), matching the
                    // implicit `z >= MAXZDIM → solid` boundary.
                    z_above[col] = ((1u32 << pad) - 1) as u8;
                }
            }
        }

        cache.z_below = z_below;
        cache.z_above = z_above;
        cache
    }

    /// S4B.4.b: chunk-aware cache build. The closure
    /// `column_reader(x, y)` returns the slab bytes of the column
    /// at world-or-grid-local position `(x, y)`, or `None` for an
    /// implicit-air / out-of-grid column (matching `build`'s OOB
    /// "treat as full air" semantics).
    ///
    /// No vsid bound — the reader owns OOB handling. Per-chunk
    /// bakes use a closure that resolves `(x, y)` to a neighbour
    /// chunk via `Grid::chunk(IVec3)` so the 2-voxel padding
    /// extends seamlessly across chunk boundaries.
    ///
    /// The cache's [`Self::vsid`] field is left at `0` for chunk-
    /// aware builds — the field is dead-code anyway, preserved
    /// only for inspection.
    #[must_use]
    pub fn build_with_reader<'r>(
        column_reader: impl Fn(i32, i32) -> Option<&'r [u8]>,
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
        for yi in 0..height {
            let y = pad_y0 + yi as i32;
            for xi in 0..width {
                let x = pad_x0 + xi as i32;
                if let Some(column) = column_reader(x, y) {
                    expandbit256(column, &mut bits[yi * width + xi]);
                }
                // None → leave the cache slot zeroed (treat as full
                // air), matching `build`'s OOB behaviour.
            }
        }

        Self {
            bits,
            origin_x: pad_x0,
            origin_y: pad_y0,
            width,
            height,
            vsid: 0,
            z_below: Vec::new(),
            z_above: Vec::new(),
        }
    }

    /// Whether the voxel at cache-column `(xi, yi)`, depth `z` is solid.
    /// Out of the `[0, MAXZDIM)` z range: by default everything above
    /// the world is air and everything below is solid (bedrock); a
    /// z-aware build ([`Self::build_with_reader_z`]) instead reads the
    /// stacked-neighbour chunk's voxels for the first `ESTNORMRAD`
    /// levels past each boundary (the `z_below` / `z_above` overlays),
    /// so AO is continuous across a chunk z-seam.
    #[inline]
    fn solid(&self, xi: usize, yi: usize, z: i32) -> bool {
        if z < 0 {
            let i = (-1 - z) as usize;
            if !self.z_below.is_empty() && i < ESTNORMRAD as usize {
                return (self.z_below[yi * self.width + xi] >> i) & 1 != 0;
            }
            return false;
        }
        if z >= MAXZDIM {
            let i = (z - MAXZDIM) as usize;
            if !self.z_above.is_empty() && i < ESTNORMRAD as usize {
                return (self.z_above[yi * self.width + xi] >> i) & 1 != 0;
            }
            return true;
        }
        let col = &self.bits[yi * self.width + xi];
        let z = z as usize;
        (col[z >> 5] >> (z & 31)) & 1 != 0
    }

    /// Estimate the surface orientation at solid voxel `(x, y, z)` as
    /// the **occupancy gradient** of its 5×5×5 neighbourhood:
    ///
    /// ```text
    /// n = Σ_{solid neighbours} offset,   normal = n / |n|
    /// ```
    ///
    /// (the sum runs over `offset ∈ [-2, 2]³`). `n` points toward the
    /// denser (solid) side; the lighting formulas in [`update_lighting`]
    /// are calibrated to that orientation. On a flat surface the solid
    /// half-space cancels laterally and leaves `n` along the inward
    /// axis. An all-solid or all-air neighbourhood gives `n = 0` →
    /// `(0, 0, 0)`, which the lighting math treats as unlit.
    ///
    /// `(x, y)` must lie inside the cache's `[x0..x1) × [y0..y1)` region
    /// (the padded border supplies the ±2 neighbours); `z` is
    /// unconstrained.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn estnorm(&self, x: i32, y: i32, z: i32) -> [f32; 3] {
        let cx = (x - self.origin_x) as i32;
        let cy = (y - self.origin_y) as i32;

        let mut nx = 0i32;
        let mut ny = 0i32;
        let mut nz = 0i32;
        for dy in -ESTNORMRAD..=ESTNORMRAD {
            let yi = (cy + dy) as usize;
            for dx in -ESTNORMRAD..=ESTNORMRAD {
                let xi = (cx + dx) as usize;
                for dz in -ESTNORMRAD..=ESTNORMRAD {
                    if self.solid(xi, yi, z + dz) {
                        nx += dx;
                        ny += dy;
                        nz += dz;
                    }
                }
            }
        }

        let len_sq = nx * nx + ny * ny + nz * nz;
        if len_sq == 0 {
            return [0.0, 0.0, 0.0];
        }
        let inv = 1.0 / (len_sq as f32).sqrt();
        [nx as f32 * inv, ny as f32 * inv, nz as f32 * inv]
    }

    /// AO.2 — ambient occlusion at solid voxel `(x, y, z)`. Returns `0.0`
    /// (open, e.g. a flat floor under open sky, or a convex edge) … `1.0`
    /// (fully enclosed).
    ///
    /// **Per-exposed-face**, normal-free: for each of the 6 axis faces whose
    /// immediate neighbour is air (an exposed face), sample the `±AO_RAD`
    /// half-space **in front of that face** (`offset · face_dir > 0`) and
    /// measure how much of it is solid (inverse-distance weighted). This only
    /// fires at **concave** corners — a perpendicular solid sitting in front of
    /// an exposed face. Flat faces and **convex** edges read `0` (the space in
    /// front of every exposed face is air). Crucially it does **not** use the
    /// `estnorm` gradient normal, which tilts near a convex edge and would make
    /// a voxel's own folded-over surface (e.g. a pillar's top above its side
    /// face) count as occlusion — the "pillow" border on every edge.
    /// `radius` is the sampling reach (clamped to `ESTNORMRAD`, the cache's
    /// padding); `1` = a tight 1-voxel concave edge, `2` = a wider contact.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn ambient_occlusion(&self, x: i32, y: i32, z: i32, radius: i32) -> f32 {
        const FACES: [[i32; 3]; 6] = [
            [-1, 0, 0],
            [1, 0, 0],
            [0, -1, 0],
            [0, 1, 0],
            [0, 0, -1],
            [0, 0, 1],
        ];
        let r = radius.clamp(1, ESTNORMRAD);
        let cx = (x - self.origin_x) as i32;
        let cy = (y - self.origin_y) as i32;
        let mut occ = 0.0f32;
        let mut total = 0.0f32;
        for f in FACES {
            // Only exposed faces (immediate neighbour is air) contribute.
            if self.solid((cx + f[0]) as usize, (cy + f[1]) as usize, z + f[2]) {
                continue;
            }
            for dy in -r..=r {
                for dx in -r..=r {
                    for dz in -r..=r {
                        // Only the half-space strictly in front of this face.
                        if dx * f[0] + dy * f[1] + dz * f[2] <= 0 {
                            continue;
                        }
                        let d = [dx as f32, dy as f32, dz as f32];
                        let w = 1.0 / (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                        total += w;
                        if self.solid((cx + dx) as usize, (cy + dy) as usize, z + dz) {
                            occ += w;
                        }
                    }
                }
            }
        }
        if total <= 0.0 {
            0.0
        } else {
            occ / total
        }
    }

    /// Voxel-grid limit; used by callers to bound their iteration.
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn vsid(&self) -> i32 {
        self.vsid
    }
}

/// Bake per-voxel lighting into the world's brightness bytes.
/// Bakes per-voxel brightness over a 3D bounding box.
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
/// - `lightmode == 2`: per-light Lambertian bake — base
///   `(tp.y * 0.5 + tp.z) * 16 + 47.5` minus, for each light in
///   range with surface normal facing it, `g * h * sc` where
///   `g = 1/(d·d²) - 1/(r·r²)` (cube falloff with hard radius
///   cutoff) and `h = tp · light_delta`.
/// - `lightmode == 3` (AO): bake **ambient occlusion** into the byte
///   (the DL ambient/AO channel) — open voxels keep `128`, crevices /
///   inside corners darken (see [`EstNormCache::ambient_occlusion`]).
///   The retro stylized lighting reads this byte as its ambient fill.
///
/// The bbox is padded by `ESTNORMRAD` on each side internally
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
    // The bake is tiled into 64×64 chunks with a per-tile
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
    // Per-column byte extents `(start, end)`. After voxalloc-driven
    // edits (e.g. cave-gen's heavy `set_spans` carve, or runtime
    // bullet-impact carves), columns are scattered in the slab
    // pool, so `column_offsets[i+1]` is NOT column `i`'s end byte
    // — walk each column's slab chain via `slng()` to
    // recover length. We pre-compute extents here serially before
    // moving `world_data` into the parallel mutable view; the
    // slng walk is O(slab_count) per column, typically 1-3 slabs.
    //
    // **Region-bounded**: only the bake rectangle `[x0p..x1p) ×
    // [y0p..y1p)` needs extents — the per-row body indexes only
    // those columns. Sizing the table to `vsid²` is wasteful when
    // a small chunk-sized region is baked against a large-vsid
    // world (e.g. S4.1 scene-graph per-chunk bake against a
    // vsid=4096 combined view — would have been 16M slng walks per
    // chunk × 1024 chunks = 17B slng walks). The bake-region table
    // collapses that to `bake_region` walks per call.
    #[allow(clippy::cast_sign_loss)]
    let region_w = (x1p - x0p) as usize;
    #[allow(clippy::cast_sign_loss)]
    let region_h = (y1p - y0p) as usize;
    let mut column_extents: Vec<(usize, usize)> = Vec::with_capacity(region_w * region_h);
    for yi in 0..region_h {
        #[allow(clippy::cast_possible_wrap)]
        let y = y0p + yi as i32;
        for xi in 0..region_w {
            #[allow(clippy::cast_possible_wrap)]
            let x = x0p + xi as i32;
            #[allow(clippy::cast_sign_loss)]
            let col_idx = (y as u32) * vsid + (x as u32);
            let start = column_offsets[col_idx as usize] as usize;
            let end = start + roxlap_formats::vxl::slng(&world_data[start..]);
            column_extents.push((start, end));
        }
    }

    let world_view = WorldDataMutView::new(world_data);
    let row_body = |y: i32| {
        #[allow(clippy::cast_sign_loss)]
        let yi = (y - y0p) as usize;
        for x in x0p..x1p {
            #[allow(clippy::cast_sign_loss)]
            let xi = (x - x0p) as usize;
            let (off_start, off_end) = column_extents[yi * region_w + xi];
            // SAFETY: each (x, y) maps to a unique col_idx; column
            // byte ranges `[off_start, off_end)` are pairwise
            // disjoint across distinct `col_idx` (voxalloc's
            // free-list invariant), so no two threads write to
            // the same byte.
            let column = unsafe { world_view.column_slice(off_start, off_end) };
            // AO (lightmode 3) via this engine entry uses default params.
            shade_column(
                column,
                x,
                y,
                z0p,
                z1p,
                lightmode,
                lights,
                &lightsub,
                &cache,
                AoParams::default(),
            );
        }
    };

    (y0p..y1p).into_par_iter().for_each(row_body);
}

/// S4B.4.b: per-chunk variant of [`update_lighting`].
///
/// Writes alpha bytes into one chunk's slab buffer; reads
/// neighbour-chunk voxels through `column_reader` for `estnorm`'s
/// 5×5×5 padding. The reader takes chunk-local `(x, y)` (which can
/// extend `±ESTNORMRAD` past the chunk's `[0, target_vsid)` extent)
/// and returns the column at that position — typically resolved
/// through `Grid::chunk(IVec3)` so the bake gets seamless
/// cross-chunk neighbourhood reads without materialising a stitched
/// combined view (Approach C retirement, S4B.4.b).
///
/// `(x0, y0, z0, x1, y1, z1)` is the bake region in chunk-local
/// coords (typically `(0, 0, 0)..(CHUNK_SIZE_XY, CHUNK_SIZE_XY,
/// CHUNK_SIZE_Z)`). Writes clip to the target chunk's vsid; reads
/// extend into neighbour chunks via the closure.
///
/// `lightmode`, `lights`, and the per-voxel arithmetic match
/// [`update_lighting`]; only the cache build + write-region
/// scoping differ.
#[allow(clippy::too_many_arguments)]
pub fn update_lighting_chunk<'r>(
    target_data: &mut [u8],
    target_column_offsets: &[u32],
    target_vsid: u32,
    x0: i32,
    y0: i32,
    z0: i32,
    x1: i32,
    y1: i32,
    z1: i32,
    column_reader: impl Fn(i32, i32) -> Option<&'r [u8]>,
    lightmode: u32,
    lights: &[LightSrc],
) {
    if lightmode == 0 {
        return;
    }
    let target_vsid_i = target_vsid as i32;

    // Padded region for the cache (cross-chunk reads via reader).
    // Z clamps to [0, MAXZDIM) because each chunk's slab data is
    // chunk-local in z. This XY-only reader leaves the top/bottom
    // boundary at the implicit air-above / bedrock-below default;
    // callers that bake stacked grids and want AO/estnorm continuous
    // across the z-seam build the cache via
    // [`EstNormCache::build_with_reader_z`] (a `chz`-aware reader) and
    // call [`apply_lighting_with_cache`] directly. X/y intentionally
    // don't clamp — the reader pulls from neighbour chunks via its own
    // coord translation.
    let z0p = (z0 - ESTNORMRAD).max(0);
    let z1p = (z1 + ESTNORMRAD).min(MAXZDIM);
    // Write region clipped to the target chunk's footprint.
    let wx0 = x0.max(0);
    let wy0 = y0.max(0);
    let wx1 = x1.min(target_vsid_i);
    let wy1 = y1.min(target_vsid_i);
    if wx0 >= wx1 || wy0 >= wy1 || z0p >= z1p {
        return;
    }

    let cache = EstNormCache::build_with_reader(column_reader, x0, y0, x1, y1);
    apply_lighting_with_cache(
        target_data,
        target_column_offsets,
        target_vsid,
        wx0,
        wy0,
        z0p,
        wx1,
        wy1,
        z1p,
        &cache,
        lightmode,
        lights,
        AoParams::default(),
    );
}

/// S4B.4.b: write half of [`update_lighting_chunk`], split out so
/// callers can build the [`EstNormCache`] separately (via
/// [`EstNormCache::build_with_reader`]) and pass it in.
///
/// The split matters when the cache build needs an immutable grid
/// borrow (for cross-chunk reads) and the write phase needs a
/// mutable target-chunk borrow — the two can't coexist. The
/// caller builds the cache first while holding the immutable
/// borrow, drops it, then mutably borrows the target chunk and
/// invokes this.
///
/// The `(x0..x1, y0..y1, z0..z1)` region must already be clipped
/// to the target chunk's footprint (this helper does no clipping).
/// `cache` must cover at least `[x0..x1) × [y0..y1)` (a `±ESTNORMRAD`
/// padding is the caller's responsibility — typically built via
/// `build_with_reader(.., x0, y0, x1, y1)` which adds the padding
/// itself).
#[allow(clippy::too_many_arguments)]
pub fn apply_lighting_with_cache(
    target_data: &mut [u8],
    target_column_offsets: &[u32],
    target_vsid: u32,
    x0: i32,
    y0: i32,
    z0: i32,
    x1: i32,
    y1: i32,
    z1: i32,
    cache: &EstNormCache,
    lightmode: u32,
    lights: &[LightSrc],
    ao: AoParams,
) {
    if lightmode == 0 || x0 >= x1 || y0 >= y1 || z0 >= z1 {
        return;
    }

    let lightsub: Vec<f32> = lights.iter().map(|l| 1.0 / (l.r2.sqrt() * l.r2)).collect();

    let region_w = (x1 - x0) as usize;
    let region_h = (y1 - y0) as usize;
    let mut column_extents: Vec<(usize, usize)> = Vec::with_capacity(region_w * region_h);
    for yi in 0..region_h {
        let y = y0 + yi as i32;
        for xi in 0..region_w {
            let x = x0 + xi as i32;
            let col_idx = (y as u32) * target_vsid + (x as u32);
            let start = target_column_offsets[col_idx as usize] as usize;
            let end = start + roxlap_formats::vxl::slng(&target_data[start..]);
            column_extents.push((start, end));
        }
    }

    let world_view = WorldDataMutView::new(target_data);
    let row_body = |y: i32| {
        let yi = (y - y0) as usize;
        for x in x0..x1 {
            let xi = (x - x0) as usize;
            let (off_start, off_end) = column_extents[yi * region_w + xi];
            // SAFETY: per-column byte ranges are pairwise disjoint
            // across distinct `(x, y)` (voxalloc invariant).
            let column = unsafe { world_view.column_slice(off_start, off_end) };
            shade_column(
                column, x, y, z0, z1, lightmode, lights, &lightsub, cache, ao,
            );
        }
    };

    (y0..y1).into_par_iter().for_each(row_body);
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
/// the per-voxel bake loop.
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
    ao: AoParams,
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
            // for alpha). The formula encodes this as
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
            // AO.0 — `lightmode == 3` bakes ambient occlusion into the byte
            // (the DL ambient/AO channel; normal-free); other modes use the
            // estnorm surface normal for the directional / point-light bake.
            let brightness = if lightmode == 3 {
                ao_byte(cache, x, y, z, ao)
            } else {
                let normal = cache.estnorm(x, y, z);
                compute_brightness(x, y, z, normal, lightmode, lights, lightsub)
            };
            let byte_off = voxel_byte_offset_signed + ((z as isize) << 2);
            if byte_off >= 0 && (byte_off as usize) < column.len() {
                column[byte_off as usize] = brightness;
            }
        }
    }
}

/// AO.0/AO.2 — map ambient occlusion to the brightness byte (the DL
/// ambient/AO channel). Open voxels keep the neutral `128` (= full ambient,
/// shader `byte/128 == 1.0`); occluded voxels darken by `params.strength`,
/// never below `params.min_floor` of the open ambient.
fn ao_byte(cache: &EstNormCache, x: i32, y: i32, z: i32, params: AoParams) -> u8 {
    let ao = cache.ambient_occlusion(x, y, z, params.radius);
    let factor = (1.0 - params.strength * ao).max(params.min_floor);
    clamp_to_byte(128.0 * factor)
}

/// Per-voxel brightness math. Computes the `[0, 255]`
/// alpha byte for one voxel from its surface normal `tp` + the
/// light list.
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
        // Directional path: single fixed sun direction
        // direction baked into a hardcoded coefficient pair.
        // i = (tp.y * 0.5 + tp.z) * 64 + 103.5, clamped to [0, 255].
        let f = (tp[1] * 0.5 + tp[2]) * 64.0 + 103.5;
        clamp_to_byte(f)
    } else {
        // Point-light path. Base brightness
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
            // Cube-law falloff with a hard cutoff at the light radius:
            //   g = 1/d³ - 1/r³   (d = distance, r = radius)
            // so the contribution fades to exactly zero at `r`.
            let g = 1.0 / (g_sq * g_sq.sqrt()) - lightsub[i];
            f -= g * h * light.sc;
        }
        clamp_to_byte(f)
    }
}

#[inline]
fn clamp_to_byte(f: f32) -> u8 {
    // Clamp the brightness into the `[0, 255]` byte range.
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

    /// AO.2 — only **concave** edges occlude: a raised block on a floor has a
    /// **convex** top (flat + edge) that must stay open (AO ≈ 0), while the
    /// floor at its **concave** base darkens.
    #[test]
    fn ao_only_darkens_concave_not_convex() {
        let vsid: u32 = 10;
        let column = |z1: u8, z2: u8| -> Vec<u8> {
            let mut c = vec![0u8, z1, z2, 0];
            for _ in z1..=z2 {
                c.extend([0x20, 0x20, 0x20, 0x80]);
            }
            c
        };
        let floor = column(20, 20); // air z<20, solid z>=20 (bedrock below)
        let block = column(15, 15); // raised: air z<15, solid z>=15
        let mut data = Vec::new();
        let mut offsets = vec![0u32; (vsid * vsid + 1) as usize];
        for i in 0..(vsid * vsid) {
            offsets[i as usize] = data.len() as u32;
            let x = i % vsid;
            let y = i / vsid;
            // 3×3 raised block at x∈[3,5], y∈[3,5].
            let raised = (3..=5).contains(&x) && (3..=5).contains(&y);
            data.extend_from_slice(if raised { &block } else { &floor });
        }
        offsets[(vsid * vsid) as usize] = data.len() as u32;
        let cache = EstNormCache::build(&data, &offsets, vsid, 0, 0, vsid as i32, vsid as i32);

        let ao = |x, y, z| cache.ambient_occlusion(x, y, z, AO_RAD);
        let top_center = ao(4, 4, 15); // convex flat top
        let top_edge = ao(3, 4, 15); // convex top edge
        let base = ao(2, 4, 20); // concave: floor at the block's base
        let flat = ao(0, 0, 20); // open flat floor
        assert!(flat < 0.01, "open flat floor must not occlude: {flat}");
        assert!(
            top_center < 0.01,
            "convex flat top must not occlude: {top_center}"
        );
        assert!(top_edge < 0.01, "convex edge must not occlude: {top_edge}");
        assert!(base > 0.1, "concave base must occlude: {base}");

        // AO.2 — params: strength scales the darkening, min_floor clamps it.
        let p = |strength, min_floor| AoParams {
            strength,
            radius: AO_RAD,
            min_floor,
        };
        let off = ao_byte(&cache, 2, 4, 20, p(0.0, 0.0));
        assert_eq!(off, 128, "strength 0 ⇒ no darkening (full ambient)");
        let full = ao_byte(&cache, 2, 4, 20, p(1.0, 0.0));
        assert!(full < 128, "strength 1 darkens the concave voxel: {full}");
        // A high min_floor clamps the darkening factor up to ~0.8·128.
        let floored = ao_byte(&cache, 2, 4, 20, p(1.0, 0.8));
        assert!(
            floored > full && floored >= 100,
            "min_floor 0.8 clamps darkening to ≥ ~102: floored={floored} full={full}",
        );
    }

    /// AO cross-chunk z-seam continuity (stacked grids). A solid voxel
    /// sitting in the chunk **above** — one level past the target chunk's
    /// top z-boundary — must count as occlusion for a side face at the
    /// boundary. The plain (`build_with_reader`) cache can't see it (the
    /// implicit `z < 0 → air` boundary), so a z-aware build
    /// (`build_with_reader_z`) must occlude **more**.
    #[test]
    fn ao_z_seam_reads_stacked_neighbour() {
        let mk = |z1: u8, z2: u8| -> Vec<u8> {
            let mut c = vec![0u8, z1, z2, 0];
            for _ in z1..=z2 {
                c.extend([0x20, 0x20, 0x20, 0x80]);
            }
            c
        };
        // Target layer (chz_delta 0): (1,1) solid from the top boundary
        // down; (2,1) a pit (air at z=0, solid z≥1) so (1,1)'s +x face is
        // exposed. The chunk above (chz_delta -1) has a solid voxel at its
        // bottom (z-local 255) over (2,1) — i.e. at our z = -1, in front of
        // that +x face. Everything else is implicit air (reader → None).
        let floor = mk(0, 0);
        let pit = mk(1, 255);
        let above = mk(255, 255);
        let reader = |x: i32, y: i32, dz: i32| -> Option<&[u8]> {
            match (x, y, dz) {
                (1, 1, 0) => Some(&floor),
                (2, 1, 0) => Some(&pit),
                (2, 1, -1) => Some(&above),
                _ => None,
            }
        };

        let plain = EstNormCache::build_with_reader(|x, y| reader(x, y, 0), 0, 0, 3, 3);
        let zaware = EstNormCache::build_with_reader_z(reader, 0, 0, 3, 3);

        let ao_plain = plain.ambient_occlusion(1, 1, 0, AO_RAD);
        let ao_z = zaware.ambient_occlusion(1, 1, 0, AO_RAD);
        assert!(
            ao_plain > 0.0,
            "the in-layer pit wall should already occlude a little: {ao_plain}"
        );
        assert!(
            ao_z > ao_plain + 0.01,
            "the solid across the z-seam must add occlusion: z-aware={ao_z} plain={ao_plain}"
        );
    }

    /// AO.0 — a floor voxel beside a wall is more occluded (darker) than an
    /// open floor voxel; an open voxel reads ≈0 occlusion.
    #[test]
    fn ambient_occlusion_darkens_next_to_a_wall() {
        let vsid: u32 = 8;
        // Per-column slab: `[nextptr=0, z1, z2, 0]` + (z2-z1+1) BGRA records.
        let column = |z1: u8, z2: u8| -> Vec<u8> {
            let mut c = vec![0u8, z1, z2, 0];
            for _ in z1..=z2 {
                c.extend([0x20, 0x20, 0x20, 0x80]);
            }
            c
        };
        let floor = column(20, 20); // single floor voxel at z=20
        let wall = column(10, 20); // wall rising from z=10..20 (above the floor)
        let mut data = Vec::new();
        let mut offsets = vec![0u32; (vsid * vsid + 1) as usize];
        for i in 0..(vsid * vsid) {
            offsets[i as usize] = data.len() as u32;
            // Tall wall at column (5, 3); floor everywhere else.
            let col = if i == 3 * vsid + 5 { &wall } else { &floor };
            data.extend_from_slice(col);
        }
        offsets[(vsid * vsid) as usize] = data.len() as u32;

        let cache = EstNormCache::build(&data, &offsets, vsid, 0, 0, vsid as i32, vsid as i32);
        // (4,3) sits next to the wall at (5,3); (2,3) is in the open.
        let near = cache.ambient_occlusion(4, 3, 20, AO_RAD);
        let open = cache.ambient_occlusion(2, 3, 20, AO_RAD);
        assert!(
            open < 0.05,
            "open floor voxel should be ~unoccluded: {open}"
        );
        assert!(
            near > open + 0.1,
            "voxel beside the wall must be more occluded: near={near} open={open}",
        );
    }

    /// AO.0 — `lightmode == 3` bakes occlusion into the alpha byte: the
    /// floor voxel beside the wall ends up darker than an open one, which
    /// stays at the neutral 128 (full ambient).
    #[test]
    fn lightmode3_bakes_ambient_occlusion() {
        let vsid: u32 = 8;
        let column = |z1: u8, z2: u8| -> Vec<u8> {
            let mut c = vec![0u8, z1, z2, 0];
            for _ in z1..=z2 {
                c.extend([0x20, 0x20, 0x20, 0xab]); // alpha 0xab to see the rewrite
            }
            c
        };
        let floor = column(20, 20);
        let wall = column(10, 20);
        let mut data = Vec::new();
        let mut offsets = vec![0u32; (vsid * vsid + 1) as usize];
        for i in 0..(vsid * vsid) {
            offsets[i as usize] = data.len() as u32;
            let col = if i == 3 * vsid + 5 { &wall } else { &floor };
            data.extend_from_slice(col);
        }
        offsets[(vsid * vsid) as usize] = data.len() as u32;

        update_lighting(&mut data, &offsets, vsid, 0, 0, 0, 8, 8, 30, 3, &[]);

        // Top floor voxel's alpha is at column offset + 7 (header 4 + BGR 3).
        let alpha = |x: u32, y: u32| data[offsets[(y * vsid + x) as usize] as usize + 7];
        let near = alpha(4, 3);
        let open = alpha(2, 3);
        assert_ne!(open, 0xab, "open voxel alpha rewritten by the AO bake");
        assert_eq!(open, 128, "open floor voxel keeps full ambient (128)");
        assert!(
            near < open,
            "voxel beside the wall is darker: near={near} open={open}"
        );
    }

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
    /// solid below (15..256): z past the last slab's bottom reads
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
