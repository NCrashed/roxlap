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
//! **Bedrock-as-air** (GPU.4 prerequisite): the GPU decompressor
//! treats bedrock voxels as empty. Rays heading into bedrock fall
//! through to the far surface (or sky); for the typical demo view
//! (camera above terrain, looking out) this is visually
//! indistinguishable from voxlap-CPU. Storing bedrock explicitly
//! would balloon a vsid=128 chunk's colour array from ~80 KiB to
//! ~10 MiB, blocking GPU.4's 32×32-chunk grid upload.
//!
//! This is `O(textured voxels)` work; not on the render hot path.

#![allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::many_single_char_names,
    clippy::missing_panics_doc
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
}

/// Number of u32 words per column in the occupancy bitmap
/// (`CHUNK_Z` bits packed 32-per-word). With `CHUNK_Z = 256` this is
/// exactly 8 — the rank-count loop in the GPU shader runs in 8
/// iterations max.
pub const OCC_WORDS_PER_COLUMN: u32 = CHUNK_Z / 32;

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

/// Decompress a `Vxl` chunk's mip-0 slab data into the GPU upload
/// shape. Caller guarantees `vxl` is shaped as a roxlap-scene chunk
/// (`vsid` square, mip-0 only required).
#[must_use]
pub fn decompress_chunk(vxl: &Vxl) -> ChunkUpload {
    let vsid = vxl.vsid;
    let vsid_usize = vsid as usize;
    let n_cols = vsid_usize * vsid_usize;
    let n_occ_words = n_cols * (OCC_WORDS_PER_COLUMN as usize);

    let mut occupancy = vec![0u32; n_occ_words];
    let mut color_offsets = vec![0u32; n_cols + 1];
    // Heuristic: each column ends up ~CHUNK_Z bedrock + ~10 textured
    // on average for a typical scene-demo terrain chunk.
    let mut colors: Vec<u32> = Vec::with_capacity(n_cols * 16);

    for y in 0..vsid {
        for x in 0..vsid {
            let col_idx = (y as usize) * vsid_usize + (x as usize);
            color_offsets[col_idx] = u32::try_from(colors.len()).expect("colours fit in u32");

            let slab = vxl.column_data(col_idx);
            decompress_column(slab, x, y, vsid, &mut occupancy, &mut colors);
        }
    }
    color_offsets[n_cols] = u32::try_from(colors.len()).expect("colours fit in u32");

    ChunkUpload {
        vsid,
        occupancy,
        color_offsets,
        colors,
    }
}

/// Walk one column's slab chain. For each **textured** voxel sets
/// the occupancy bit and pushes its packed BGRA u32 into `colors`.
/// Bedrock voxels (implicit solid below a slab's textured floor)
/// are skipped — treated as air for the GPU marcher.
fn decompress_column(
    slab: &[u8],
    x: u32,
    y: u32,
    vsid: u32,
    occupancy: &mut [u32],
    colors: &mut Vec<u32>,
) {
    let vsid_usize = vsid as usize;
    let runs = expand_solid_runs(slab);
    let ranges = build_color_ranges(slab);

    let mut range_cursor = 0usize;
    for (top, bot) in runs {
        for z in top..bot {
            while range_cursor < ranges.len() && z >= ranges[range_cursor].z_end {
                range_cursor += 1;
            }
            // Skip bedrock z values — outside every colour range.
            if range_cursor >= ranges.len() || z < ranges[range_cursor].z_start {
                continue;
            }
            let off = ((z - ranges[range_cursor].z_start) as usize) * 4;
            let bytes = &ranges[range_cursor].colours[off..off + 4];
            let rgb = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);

            // z-innermost packing: each column owns 8 contiguous u32
            // words covering z=0..256.
            let col_idx = (x as usize) + (y as usize) * vsid_usize;
            let col_word_base = col_idx * (OCC_WORDS_PER_COLUMN as usize);
            let z_word = (z as usize) / 32;
            let z_bit = (z as u32) & 31;
            occupancy[col_word_base + z_word] |= 1u32 << z_bit;
            colors.push(rgb);
        }
    }
}

/// Port of `expandrle` (voxlap5.c:4131) but emitting `(top, bot)`
/// pairs as half-open ranges instead of the in-place `uind` layout.
/// Solid for `z ∈ [top, bot)`. Last run's `bot` is always `CHUNK_Z`
/// (matches the voxlap "implicit bedrock below" assumption).
fn expand_solid_runs(slab: &[u8]) -> Vec<(i32, i32)> {
    // Worst case = MAXZDIM/2 alternating solid/air runs.
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
    uind[i - 1] = CHUNK_Z as i32;

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
    /// `0xAARRGGBB = 0x00ff_8000` (red-orange). Used as the
    /// canonical fixture for both CPU and GPU round-trip tests.
    pub(crate) fn fixture_one_voxel_per_column() -> Vxl {
        let vsid: u32 = 4;
        let n_cols = (vsid as usize) * (vsid as usize);
        let mut data = Vec::with_capacity(n_cols * 8);
        let mut column_offset = Vec::with_capacity(n_cols + 1);
        // 0x00ff_8000 little-endian bytes = [0x00, 0x80, 0xff, 0x00]
        // = [B=0x00, G=0x80, R=0xff, A=0x00].
        let bgra = [0x00, 0x80, 0xff, 0x00];
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
        assert_eq!(chunk.voxel_at(1, 2, 100), Some(0x00ff_8000));
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
