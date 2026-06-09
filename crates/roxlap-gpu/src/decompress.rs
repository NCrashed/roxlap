//! GPU.2 — Vxl → (occupancy bitmap, colour offsets, packed colour
//! array). Pure CPU; no wgpu deps in this module. Shape:
//!
//! * `occupancy[x, y]` is 8 contiguous u32 words covering z=0..256,
//!   one bit per voxel with z-innermost ordering. Bit position of
//!   voxel `(x, y, z)` is `z + (x + y*vsid)*CHUNK_Z`; the word
//!   index is `(x + y*vsid)*8 + z/32` and the bit-in-word is `z & 31`.
//!   This packs each column's 256 z-bits into 8 contiguous u32s so
//!   the GPU shader can rank-count solid voxels in O(8 popcount)
//!   instead of O(z) sequential bit fetches.
//! * `color_offsets[x + y*vsid]` — u32 = base index into `colors`
//!   for that column's voxels in ascending z. `vsid*vsid + 1`
//!   entries; trailing sentinel = `colors.len()`.
//! * `colors[..]` — packed u32 per occupied voxel, ordered first by
//!   column index then by ascending z within the column.
//!
//! The voxlap slab format interleaves floor and ceiling colour
//! ranges across slab boundaries, with implicit "bedrock" voxels
//! filling the gap between a slab's textured floor and the next
//! slab's air-gap top. Bedrock has no per-voxel colour in the slab
//! data — voxlap stores only textured surfaces.
//!
//! **Bedrock-as-solid** (cliff-face fix): each [`MipUpload`] carries
//! *two* bitmaps — `occupancy` (textured voxels only, for the colour
//! rank) and `solid_occupancy` (textured surfaces **plus** the
//! implicit bedrock interior below them). The marcher hit-tests
//! `solid_occupancy`, so vertical wall/cliff faces are opaque; a
//! bedrock hit (solid but uncoloured) inherits the colour of the
//! textured surface above it. Bedrock is still stored as **1 bit**,
//! not a per-voxel colour, so the colour array stays
//! `O(textured voxels)` — storing bedrock colours would balloon a
//! vsid=128 chunk from ~80 KiB to ~10 MiB. The cost is one extra
//! occupancy bitmap (occupancy storage doubles; colours unchanged).
//!
//! (Originally "bedrock-as-air", GPU.4: bedrock was dropped entirely,
//! which left cliff faces transparent to the sky.)
//!
//! This is `O(textured voxels)` work; not on the render hot path.

#![allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::many_single_char_names,
    clippy::missing_panics_doc,
    clippy::verbose_bit_mask
)]

use roxlap_formats::vxl::Vxl;

/// Z-extent of every voxlap column — matches `roxlap_formats`'
/// private `MAXZDIM` (`voxlap5.h:10`). Re-declared here so this
/// module stays a pure consumer of the public `Vxl` surface.
pub const CHUNK_Z: u32 = 256;

/// Historic sentinel BGRA for bedrock voxels — kept exported so
/// callers that want voxlap-CPU bedrock parity can render their own
/// pass. **Not used by the default GPU decompressor**: the
/// "bedrock-as-air" refactor (GPU.4 prereq) skips bedrock entirely.
pub const BEDROCK_RGB: u32 = 0x0040_4040;

/// CPU-decompressed chunk ready to upload to the GPU. Each field
/// maps onto one storage buffer in GPU.3+; for GPU.2 the buffers
/// also serve the read-back validator.
#[derive(Debug, Clone)]
pub struct ChunkUpload {
    /// XY extent of the chunk in voxels — typically `roxlap-scene`'s
    /// `CHUNK_SIZE_XY = 128`. Same as `Vxl::vsid`.
    pub vsid: u32,
    /// 1 bit per voxel, packed little-endian within each u32.
    /// `bit(x, y, z) = (occupancy[i >> 5] >> (i & 31)) & 1`
    /// where `i = x + y*vsid + z*vsid*vsid`.
    pub occupancy: Vec<u32>,
    /// `vsid*vsid + 1` entries. Column `(x, y)`'s colours live at
    /// `colors[offsets[x + y*vsid] .. offsets[x + y*vsid + 1]]`,
    /// in ascending-z order across all solid voxels of that column.
    pub color_offsets: Vec<u32>,
    /// Packed BGRA u32 per solid voxel (textured + bedrock).
    pub colors: Vec<u32>,
    /// GPU.11 — the full mip ladder, finest (mip-0) first. `mips[0]`
    /// is identical to the [`Self::occupancy`] / [`Self::color_offsets`]
    /// / [`Self::colors`] fields above (which the older single-chunk
    /// and single-grid GPU paths still read directly). The scene
    /// path concatenates every mip per slot; the shader marches the
    /// mip its LOD picker selects. Always at least one entry; count
    /// is [`gpu_mip_count`]`(vsid)`.
    pub mips: Vec<MipUpload>,
}

/// Number of u32 words per column in the occupancy bitmap
/// (`CHUNK_Z` bits packed 32-per-word). With `CHUNK_Z = 256` this is
/// exactly 8 — the rank-count loop in the GPU shader runs in 8
/// iterations max.
pub const OCC_WORDS_PER_COLUMN: u32 = CHUNK_Z / 32;

/// GPU.11 — number of mip levels [`decompress_chunk`] builds per
/// chunk (capped by the chunk's own `vsid` / `CHUNK_Z` halving).
/// Matches the CPU demo's `OpticastSettings::mip_levels = 6`, so the
/// GPU mip ladder reaches the same ray-depth as the CPU path
/// (`mip_scan_dist · 2⁵`). The per-mip relative-offset tables in
/// [`crate::scene::GridStaticMeta`] are sized to this.
pub const GPU_MAX_MIPS: u32 = 6;

/// GPU.11 — how many mip levels a chunk of side `vsid` actually
/// yields under [`GPU_MAX_MIPS`]. Mirrors the stopping rule in
/// [`Vxl::generate_mips`] (`src_vsid > 1 && src_z > 1 && n < max`)
/// so the upload, the per-slot stride math in [`crate::scene`], and
/// the shader all agree on the level count for a given `vsid`.
/// Always `>= 1` (mip-0).
#[must_use]
pub fn gpu_mip_count(vsid: u32) -> u32 {
    let mut n = 1u32;
    let mut v = vsid;
    let mut z = CHUNK_Z as i32;
    while v > 1 && z > 1 && n < GPU_MAX_MIPS {
        v >>= 1;
        z >>= 1;
        n += 1;
    }
    n
}

/// GPU.11 — number of occupancy u32 words per column at a given mip
/// (`(CHUNK_Z >> mip)` bits packed 32-per-word, min 1). Mip-0 is
/// [`OCC_WORDS_PER_COLUMN`] (= 8).
#[must_use]
pub fn occ_words_per_column_for_mip(mip: u32) -> u32 {
    (CHUNK_Z >> mip).div_ceil(32).max(1)
}

/// GPU.11 — one mip level of a chunk in the GPU upload shape. Mip-N
/// has `(vsid >> mip)²` columns spanning z = 0..`CHUNK_Z >> mip`.
///
/// `color_offsets` are **absolute within the chunk's whole colour
/// block** (cumulative across mips): mip-0's run [0..n0], mip-1's
/// run [n0..n0+n1], etc. So `colors` of all mips concatenated in
/// level order index directly via `chunk_colors_base + offset +
/// rank` — the same formula the shader already uses for mip-0,
/// independent of which mip is being read.
#[derive(Debug, Clone)]
pub struct MipUpload {
    /// XY column extent at this mip = `vsid >> mip`.
    pub vsid: u32,
    /// Z-extent at this mip = `CHUNK_Z >> mip`.
    pub cz: u32,
    /// Occupancy words per column at this mip = `cz.div_ceil(32)`.
    pub occ_words_per_col: u32,
    /// `vsid² * occ_words_per_col` packed **textured** occupancy bits
    /// (one set bit per voxel that has an explicit colour). The shader
    /// rank-counts these for the colour lookup.
    pub occupancy: Vec<u32>,
    /// Same shape as `occupancy`, but one set bit per **solid** voxel
    /// — textured surfaces *and* the implicit bedrock interior below
    /// them. The marcher hit-tests against this so vertical
    /// wall/cliff faces are opaque; bedrock hits inherit the colour of
    /// the textured surface above them. (Fixes the "cliff face shows
    /// sky" bedrock-as-air artifact.)
    pub solid_occupancy: Vec<u32>,
    /// `vsid² + 1` cumulative-within-chunk colour offsets.
    pub color_offsets: Vec<u32>,
    /// This mip's packed BGRA colours (ascending z within a column,
    /// columns in `x + y*vsid` order).
    pub colors: Vec<u32>,
}

impl ChunkUpload {
    /// Helper for tests / debug — looks up the colour at `(x, y, z)`
    /// if solid, else `None`. CPU-side mirror of what the GPU shader
    /// computes.
    #[must_use]
    pub fn voxel_at(&self, x: u32, y: u32, z: u32) -> Option<u32> {
        if x >= self.vsid || y >= self.vsid || z >= CHUNK_Z {
            return None;
        }
        let col_idx = (x + y * self.vsid) as usize;
        let col_word_base = col_idx * OCC_WORDS_PER_COLUMN as usize;
        let z_word = (z / 32) as usize;
        let z_bit = z & 31;
        let bit = (self.occupancy[col_word_base + z_word] >> z_bit) & 1;
        if bit == 0 {
            return None;
        }
        // Rank-count solid voxels at z' < z in the same column —
        // popcount of `z_word` full words + masked partial.
        let mut rank = 0u32;
        for w in 0..z_word {
            rank += self.occupancy[col_word_base + w].count_ones();
        }
        let mask = if z_bit == 0 {
            0u32
        } else {
            (1u32 << z_bit) - 1
        };
        rank += (self.occupancy[col_word_base + z_word] & mask).count_ones();

        let base = self.color_offsets[col_idx];
        Some(self.colors[(base + rank) as usize])
    }
}

/// Decompress a `Vxl` chunk into the GPU upload shape, building the
/// full mip ladder ([`gpu_mip_count`]`(vsid)` levels). Caller
/// guarantees `vxl` is shaped as a roxlap-scene chunk (`vsid`
/// square). If `vxl` already carries at least that many mips (the
/// common scene path — the bake generates 6), they are read
/// directly; otherwise the chunk is cloned and re-mipped so the
/// upload always carries a deterministic, vsid-uniform level count.
///
/// `mips[0]` is the legacy mip-0 data, also mirrored into the
/// top-level [`ChunkUpload`] fields for the older single-chunk /
/// single-grid paths.
#[must_use]
pub fn decompress_chunk(vxl: &Vxl) -> ChunkUpload {
    let vsid = vxl.vsid;
    let target = gpu_mip_count(vsid);

    // Ensure `target` mips are available without mutating the
    // caller's borrow. The terrain + streaming bake already builds
    // 6, so the fast path takes the existing tables (no clone).
    let owned;
    let src: &Vxl = if vxl.mip_count() >= target {
        vxl
    } else {
        let mut c = vxl.clone();
        c.generate_mips(GPU_MAX_MIPS);
        owned = c;
        &owned
    };

    let mut mips: Vec<MipUpload> = Vec::with_capacity(target as usize);
    let mut color_base = 0u32;
    for m in 0..target {
        let mip = decompress_mip(src, m, color_base);
        color_base = *mip.color_offsets.last().expect("offsets non-empty");
        mips.push(mip);
    }

    let m0 = &mips[0];
    ChunkUpload {
        vsid,
        occupancy: m0.occupancy.clone(),
        color_offsets: m0.color_offsets.clone(),
        colors: m0.colors.clone(),
        mips,
    }
}

/// Decompress a single mip level `mip` of `src` into a [`MipUpload`].
/// `color_base` is the cumulative colour count of all finer mips
/// (mips `< mip`) so this level's `color_offsets` stay absolute
/// within the chunk's whole colour block.
#[must_use]
fn decompress_mip(src: &Vxl, mip: u32, color_base: u32) -> MipUpload {
    let vsid = src.vsid >> mip;
    let cz = CHUNK_Z >> mip;
    let occ_words_per_col = occ_words_per_column_for_mip(mip);
    let vsid_usize = vsid as usize;
    let n_cols = vsid_usize * vsid_usize;
    let n_occ_words = n_cols * (occ_words_per_col as usize);

    let mut occupancy = vec![0u32; n_occ_words];
    let mut solid_occupancy = vec![0u32; n_occ_words];
    let mut color_offsets = vec![0u32; n_cols + 1];
    let mut colors: Vec<u32> = Vec::with_capacity(n_cols * 4);

    for y in 0..vsid {
        for x in 0..vsid {
            let col_idx = (y as usize) * vsid_usize + (x as usize);
            color_offsets[col_idx] =
                color_base + u32::try_from(colors.len()).expect("colours fit in u32");

            let slab = src.column_data_for_mip(mip, col_idx);
            decompress_column(
                slab,
                x,
                y,
                vsid,
                cz,
                occ_words_per_col,
                &mut occupancy,
                &mut solid_occupancy,
                &mut colors,
            );
        }
    }
    color_offsets[n_cols] = color_base + u32::try_from(colors.len()).expect("colours fit in u32");

    MipUpload {
        vsid,
        cz,
        occ_words_per_col,
        occupancy,
        solid_occupancy,
        color_offsets,
        colors,
    }
}

/// Walk one column's slab chain, producing two bitmaps + the colour
/// list:
/// * `occupancy` — one bit per **textured** voxel (has an explicit
///   colour); pushed into `colors` in ascending z. The shader
///   rank-counts these for the colour lookup.
/// * `solid_occupancy` — one bit per **solid** voxel: textured
///   surfaces *and* the implicit bedrock interior below them. The
///   marcher hit-tests this, so cliff/wall faces are opaque.
///
/// Bedrock (solid but uncoloured) is marked solid only *after* the
/// column's first real textured surface, so the `empty_chunk_vxl`
/// all-zero placeholder columns stay fully air (no spurious black
/// floor/ceiling). `cz` is the column z-extent at this mip; the runs
/// from [`expand_solid_runs`] already exclude overhang air gaps.
#[allow(clippy::too_many_arguments)]
fn decompress_column(
    slab: &[u8],
    x: u32,
    y: u32,
    vsid: u32,
    cz: u32,
    occ_words_per_col: u32,
    occupancy: &mut [u32],
    solid_occupancy: &mut [u32],
    colors: &mut Vec<u32>,
) {
    let vsid_usize = vsid as usize;
    let runs = expand_solid_runs(slab, cz);
    let ranges = build_color_ranges(slab);

    let col_idx = (x as usize) + (y as usize) * vsid_usize;
    let col_word_base = col_idx * (occ_words_per_col as usize);
    // Once a column has a real textured surface, everything solid
    // below it is bedrock to fill (opaque). Before the first surface
    // the run voxels are placeholder/air — leave them clear.
    let mut have_surface = false;

    let mut range_cursor = 0usize;
    for (top, bot) in runs {
        for z in top..bot {
            while range_cursor < ranges.len() && z >= ranges[range_cursor].z_end {
                range_cursor += 1;
            }
            let in_range = range_cursor < ranges.len() && z >= ranges[range_cursor].z_start;
            // A textured voxel has a non-zero RGB inside a colour
            // range. `empty_chunk_vxl`'s placeholder keeps RGB 0
            // ([0,0,0,0]); treat it as untextured so unbaked columns
            // don't paint a black surface.
            let mut rgb = 0u32;
            if in_range {
                let off = ((z - ranges[range_cursor].z_start) as usize) * 4;
                let bytes = &ranges[range_cursor].colours[off..off + 4];
                rgb = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            }
            let z_word = (z as usize) / 32;
            let z_bit = (z as u32) & 31;

            if in_range && (rgb & 0x00ff_ffff) != 0 {
                // Textured surface voxel: solid + coloured.
                occupancy[col_word_base + z_word] |= 1u32 << z_bit;
                solid_occupancy[col_word_base + z_word] |= 1u32 << z_bit;
                colors.push(rgb);
                have_surface = true;
            } else if !in_range && have_surface {
                // GENUINE bedrock — uncoloured solid below a surface
                // (no colour range covers it). Mark solid; it inherits
                // the surface colour above at render time. A voxel that
                // IS in a colour range but reads RGB 0 is the
                // `empty_chunk_vxl` placeholder (the column's z=255
                // air filler) — leave it air, else floating objects
                // grow a spurious floor plane below them.
                solid_occupancy[col_word_base + z_word] |= 1u32 << z_bit;
            }
        }
    }
}

/// Port of `expandrle` (voxlap5.c:4131) but emitting `(top, bot)`
/// pairs as half-open ranges instead of the in-place `uind` layout.
/// Solid for `z ∈ [top, bot)`. Last run's `bot` is always `cz` (the
/// column's z-extent at this mip — matches the voxlap "implicit
/// bedrock below" assumption, halved per mip level).
fn expand_solid_runs(slab: &[u8], cz: u32) -> Vec<(i32, i32)> {
    // Worst case = MAXZDIM/2 alternating solid/air runs (mip-0 bound;
    // coarser mips use fewer entries).
    let mut uind = [0i32; (CHUNK_Z as usize) + 2];
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
    uind[i - 1] = cz as i32;

    let n_runs = i / 2;
    let mut runs = Vec::with_capacity(n_runs);
    for k in 0..n_runs {
        runs.push((uind[2 * k], uind[2 * k + 1]));
    }
    runs
}

/// One colour-record range = colours for voxels at `z ∈ [z_start, z_end)`.
struct ColorRange<'s> {
    z_start: i32,
    z_end: i32,
    colours: &'s [u8],
}

/// Build the per-column colour lookup table — port of voxlap's
/// `compilerle` colour-table loop (voxlap5.c:4163-4174) + the
/// matching ceiling-colour walk. Mirrors `roxlap-formats`'
/// private `build_color_table` field-for-field.
fn build_color_ranges(slab: &[u8]) -> Vec<ColorRange<'_>> {
    let mut ranges: Vec<ColorRange<'_>> = Vec::new();
    let mut v = 0usize;
    loop {
        let z_start = i32::from(slab[v + 1]);
        let z1c = i32::from(slab[v + 2]);
        let z_end = z1c + 1;
        let n_voxels = usize::try_from((z_end - z_start).max(0)).expect("non-negative");
        let off = v + 4;
        ranges.push(ColorRange {
            z_start,
            z_end,
            colours: &slab[off..off + n_voxels * 4],
        });

        let nextptr = slab[v];
        if nextptr == 0 {
            break;
        }
        let prev_z1 = z_start;
        let prev_z1c = z1c;
        let prev_nextptr = i32::from(nextptr);
        v += usize::from(nextptr) * 4;

        // Ceiling colour list for the NEW slab — stored in the tail
        // of the previous slab's bytes, between its floor colours
        // and the next slab's header.
        let ze = i32::from(slab[v + 3]);
        let ceil_z_start = ze + prev_z1c - prev_z1 - prev_nextptr + 2;
        let ceil_z_end = ze;
        let ceil_n = usize::try_from((ceil_z_end - ceil_z_start).max(0)).expect("non-negative");
        let ceil_start = v - ceil_n * 4;
        ranges.push(ColorRange {
            z_start: ceil_z_start,
            z_end: ceil_z_end,
            colours: &slab[ceil_start..v],
        });
    }
    ranges
}

#[cfg(test)]
mod tests {
    use super::*;
    use roxlap_formats::vxl::Vxl;

    /// Build a tiny `Vxl` (4×4 columns) where every column is a
    /// single-slab "one floor voxel at z=100" with colour
    /// `0xAARRGGBB = 0x80ff_8000` (red-orange). Used as the
    /// canonical fixture for both CPU and GPU round-trip tests.
    pub(crate) fn fixture_one_voxel_per_column() -> Vxl {
        let vsid: u32 = 4;
        let n_cols = (vsid as usize) * (vsid as usize);
        let mut data = Vec::with_capacity(n_cols * 8);
        let mut column_offset = Vec::with_capacity(n_cols + 1);
        // 0x80ff_8000 little-endian bytes = [0x00, 0x80, 0xff, 0x80]
        // = [B=0x00, G=0x80, R=0xff, A=0x80 (neutral brightness)].
        // Alpha=0x80 keeps the fixture non-empty under the
        // alpha-zero placeholder filter in `decompress_column`.
        let bgra = [0x00, 0x80, 0xff, 0x80];
        for _ in 0..n_cols {
            column_offset.push(u32::try_from(data.len()).expect("offset fits"));
            data.extend_from_slice(&[0, 100, 100, 0]); // nextptr=0, z1=100, z1c=100, z0=0
            data.extend_from_slice(&bgra);
        }
        column_offset.push(u32::try_from(data.len()).expect("offset fits"));

        Vxl {
            vsid,
            ipo: [0.0; 3],
            ist: [1.0, 0.0, 0.0],
            ihe: [0.0, 0.0, 1.0],
            ifo: [0.0, 1.0, 0.0],
            data: data.into_boxed_slice(),
            column_offset: column_offset.into_boxed_slice(),
            mip_base_offsets: Box::new([0, n_cols + 1]),
            vbit: Box::new([]),
            vbiti: 0,
        }
    }

    #[test]
    fn fixture_textured_voxel_carries_slab_colour() {
        let vxl = fixture_one_voxel_per_column();
        let chunk = decompress_chunk(&vxl);
        assert_eq!(chunk.voxel_at(1, 2, 100), Some(0x80ff_8000));
    }

    #[test]
    fn fixture_air_above_textured_is_empty() {
        let vxl = fixture_one_voxel_per_column();
        let chunk = decompress_chunk(&vxl);
        for z in 0..100 {
            assert_eq!(chunk.voxel_at(1, 2, z), None, "z={z} expected air");
        }
    }

    #[test]
    fn fixture_below_textured_is_air_after_bedrock_strip() {
        // Bedrock-as-air refactor (GPU.4 prereq): z>z1c is no
        // longer reported as solid by the GPU decompressor.
        let vxl = fixture_one_voxel_per_column();
        let chunk = decompress_chunk(&vxl);
        for z in 101..CHUNK_Z {
            assert_eq!(
                chunk.voxel_at(1, 2, z),
                None,
                "z={z} expected air (bedrock stripped)"
            );
        }
    }

    #[test]
    fn only_textured_voxels_are_marked_solid() {
        let vxl = fixture_one_voxel_per_column();
        let chunk = decompress_chunk(&vxl);
        // 1 textured voxel per column.
        let solid: u32 = chunk.occupancy.iter().map(|w| w.count_ones()).sum();
        let expected = chunk.vsid * chunk.vsid;
        assert_eq!(solid, expected);
    }

    // ---- GPU.11 mip-ladder tests ------------------------------------

    #[test]
    fn gpu_mip_count_matches_generate_mips() {
        // vsid=128 chunk supports the full GPU_MAX_MIPS=6.
        assert_eq!(gpu_mip_count(128), 6);
        // Small vsid caps the ladder where halving hits 1.
        assert_eq!(gpu_mip_count(4), 3); // 4 -> 2 -> 1
        assert_eq!(gpu_mip_count(2), 2); // 2 -> 1
        assert_eq!(gpu_mip_count(1), 1);
    }

    #[test]
    fn occ_words_per_column_halves_with_z() {
        assert_eq!(occ_words_per_column_for_mip(0), 8); // 256/32
        assert_eq!(occ_words_per_column_for_mip(1), 4); // 128/32
        assert_eq!(occ_words_per_column_for_mip(2), 2); // 64/32
        assert_eq!(occ_words_per_column_for_mip(3), 1); // 32/32
        assert_eq!(occ_words_per_column_for_mip(4), 1); // 16 -> min 1
        assert_eq!(occ_words_per_column_for_mip(5), 1); // 8 -> min 1
    }

    #[test]
    fn mip0_mirrors_legacy_top_level_fields() {
        let vxl = fixture_one_voxel_per_column();
        let chunk = decompress_chunk(&vxl);
        assert!(chunk.mips.len() >= 2, "fixture should build a mip ladder");
        let m0 = &chunk.mips[0];
        assert_eq!(m0.vsid, 4);
        assert_eq!(m0.cz, CHUNK_Z);
        assert_eq!(m0.occ_words_per_col, OCC_WORDS_PER_COLUMN);
        assert_eq!(m0.occupancy, chunk.occupancy, "mip-0 occupancy == legacy");
        assert_eq!(m0.colors, chunk.colors, "mip-0 colours == legacy");
        assert_eq!(
            m0.color_offsets, chunk.color_offsets,
            "mip-0 offsets == legacy"
        );
        assert_eq!(m0.color_offsets[0], 0, "mip-0 starts at colour 0");
    }

    #[test]
    fn each_mip_popcount_equals_color_count() {
        // The 1:1 occupancy-bit ↔ colour invariant must hold at every
        // mip level (the shader's rank-count colour lookup relies on
        // it). The fixture's clone+generate_mips path exercises the
        // coarse levels.
        let vxl = fixture_one_voxel_per_column();
        let chunk = decompress_chunk(&vxl);
        assert_eq!(chunk.mips.len() as u32, gpu_mip_count(4));
        for (m, mip) in chunk.mips.iter().enumerate() {
            let solid: u32 = mip.occupancy.iter().map(|w| w.count_ones()).sum();
            let in_mip = mip.colors.len() as u32;
            assert_eq!(
                solid, in_mip,
                "mip {m}: {solid} solid bits but {in_mip} colours",
            );
            assert_eq!(mip.vsid, 4 >> m as u32);
            assert_eq!(mip.cz, CHUNK_Z >> m as u32);
            assert_eq!(
                mip.occ_words_per_col,
                occ_words_per_column_for_mip(m as u32)
            );
            assert_eq!(
                mip.occupancy.len() as u32,
                mip.vsid * mip.vsid * mip.occ_words_per_col,
            );
            assert_eq!(mip.color_offsets.len() as u32, mip.vsid * mip.vsid + 1);
        }
    }

    #[test]
    fn solid_occupancy_fills_bedrock_below_surface() {
        // The cliff-face fix: every mip's `solid_occupancy` is solid
        // from the textured surface down through the bedrock interior,
        // while `occupancy` (textured) marks only the surface.
        let vxl = fixture_one_voxel_per_column(); // textured at z=100
        let chunk = decompress_chunk(&vxl);
        let m0 = &chunk.mips[0];
        let base = 0usize; // column (0,0)
        let bit = |buf: &[u32], z: u32| (buf[base + (z / 32) as usize] >> (z & 31)) & 1 == 1;

        assert!(bit(&m0.occupancy, 100), "surface textured");
        assert!(bit(&m0.solid_occupancy, 100), "surface solid");
        assert!(!bit(&m0.occupancy, 150), "bedrock is not textured");
        assert!(
            bit(&m0.solid_occupancy, 150),
            "bedrock below surface is solid"
        );
        assert!(bit(&m0.solid_occupancy, 255), "bedrock fills to the bottom");
        assert!(
            !bit(&m0.solid_occupancy, 50),
            "air above the surface stays air"
        );
        // The textured popcount still equals the colour count.
        let solid: u32 = m0.solid_occupancy.iter().map(|w| w.count_ones()).sum();
        let tex: u32 = m0.occupancy.iter().map(|w| w.count_ones()).sum();
        assert!(solid > tex, "solid (surface + bedrock) exceeds textured");
    }

    #[test]
    fn color_offsets_are_absolute_and_monotonic_across_mips() {
        let vxl = fixture_one_voxel_per_column();
        let chunk = decompress_chunk(&vxl);
        let mut prev_end = 0u32;
        for (m, mip) in chunk.mips.iter().enumerate() {
            // Within a mip, offsets are non-decreasing.
            for w in mip.color_offsets.windows(2) {
                assert!(w[0] <= w[1], "mip {m} offsets not monotonic");
            }
            // First offset continues where the previous mip's colours
            // ended (cumulative within the chunk's whole colour block).
            assert_eq!(
                mip.color_offsets[0], prev_end,
                "mip {m} colour base not contiguous",
            );
            // Trailing sentinel == base + this mip's colour count.
            assert_eq!(
                *mip.color_offsets.last().unwrap(),
                prev_end + mip.colors.len() as u32,
            );
            prev_end = *mip.color_offsets.last().unwrap();
        }
    }

    #[test]
    fn color_offsets_partition_colours_correctly() {
        let vxl = fixture_one_voxel_per_column();
        let chunk = decompress_chunk(&vxl);
        let n_cols = (chunk.vsid * chunk.vsid) as usize;
        assert_eq!(chunk.color_offsets.len(), n_cols + 1);
        assert_eq!(chunk.color_offsets[0], 0);
        // Bedrock is stripped — only the 1 textured voxel/column
        // ends up in colours.
        let per_col = 1;
        for i in 0..=n_cols {
            assert_eq!(
                chunk.color_offsets[i],
                u32::try_from(i).expect("test fixture small") * per_col,
            );
        }
        assert_eq!(
            *chunk.color_offsets.last().unwrap() as usize,
            chunk.colors.len()
        );
    }
}
