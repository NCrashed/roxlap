//! GPU.2 — Vxl → (occupancy bitmap, colour offsets, packed colour
//! array). Pure CPU; no wgpu deps in this module. The shape mirrors
//! `PORTING-GPU.md` §"Data representation":
//!
//! * `occupancy[chx, chy, chz][x, y, z]` — 1 bit per voxel, packed
//!   into u32s. ⌈XY·XY·Z/32⌉ words for a single chunk.
//! * `color_offsets[chx, chy, chz][x, y]` — u32 per column = base
//!   index into `colors` for that column's voxels in ascending z.
//! * `colors[grid_id][...]` — packed u32 per occupied voxel. The
//!   columns are stored back-to-back; column `(x, y)` runs from
//!   `offsets[x + y*XY] .. offsets[x + y*XY + 1]`.
//!
//! The voxlap slab format interleaves floor and ceiling colour
//! ranges across slab boundaries, with implicit "bedrock" voxels
//! filling the gap between a slab's textured floor and the next
//! slab's air-gap top. We faithfully expand both: textured runs
//! get their per-voxel colour from the slab data; bedrock voxels
//! get a sentinel grey colour so rays from below still terminate.
//!
//! This is `O(occupied voxels)` work; not on the render hot path.

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

/// Sentinel BGRA the decompressor stamps onto bedrock voxels — the
/// implicit-solid region below a slab's textured floor. Dark grey
/// in voxlap's 0xAARRGGBB convention. Chosen to be visually
/// distinct from typical terrain so a stray bedrock hit shows up
/// in screenshots rather than passing for terrain.
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

impl ChunkUpload {
    /// Helper for tests / debug — looks up the colour at `(x, y, z)`
    /// if solid, else `None`. CPU-side mirror of what the GPU shader
    /// computes.
    #[must_use]
    pub fn voxel_at(&self, x: u32, y: u32, z: u32) -> Option<u32> {
        if x >= self.vsid || y >= self.vsid || z >= CHUNK_Z {
            return None;
        }
        let vsid = self.vsid as usize;
        let i = (x as usize) + (y as usize) * vsid + (z as usize) * vsid * vsid;
        let bit = (self.occupancy[i >> 5] >> (i & 31)) & 1;
        if bit == 0 {
            return None;
        }
        // Find which colour goes here by counting solid voxels
        // above us in the same column.
        let col_idx = (x as usize) + (y as usize) * vsid;
        let mut rank = 0u32;
        for zi in 0..z {
            let j = (x as usize) + (y as usize) * vsid + (zi as usize) * vsid * vsid;
            rank += (self.occupancy[j >> 5] >> (j & 31)) & 1;
        }
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
    let n_voxels = n_cols * (CHUNK_Z as usize);

    let mut occupancy = vec![0u32; n_voxels.div_ceil(32)];
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

/// Walk one column's slab chain. For each solid voxel sets the
/// occupancy bit and pushes its packed BGRA u32 into `colors`.
/// Bedrock voxels (implicit solid below a slab's textured floor)
/// get `BEDROCK_RGB`.
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
            let rgb = if range_cursor < ranges.len() && z >= ranges[range_cursor].z_start {
                let off = ((z - ranges[range_cursor].z_start) as usize) * 4;
                let bytes = &ranges[range_cursor].colours[off..off + 4];
                u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
            } else {
                BEDROCK_RGB
            };

            let voxel_idx =
                (x as usize) + (y as usize) * vsid_usize + (z as usize) * vsid_usize * vsid_usize;
            occupancy[voxel_idx >> 5] |= 1u32 << (voxel_idx & 31);
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
    fn fixture_below_textured_is_bedrock() {
        let vxl = fixture_one_voxel_per_column();
        let chunk = decompress_chunk(&vxl);
        for z in 101..CHUNK_Z {
            assert_eq!(
                chunk.voxel_at(1, 2, z),
                Some(BEDROCK_RGB),
                "z={z} expected bedrock"
            );
        }
    }

    #[test]
    fn fixture_solid_run_length_matches_expandrle() {
        let vxl = fixture_one_voxel_per_column();
        let chunk = decompress_chunk(&vxl);
        // 156 solid voxels per column: textured 1 + bedrock 155.
        let solid: u32 = chunk.occupancy.iter().map(|w| w.count_ones()).sum();
        let expected = (chunk.vsid * chunk.vsid) * (CHUNK_Z - 100);
        assert_eq!(solid, expected);
    }

    #[test]
    fn color_offsets_partition_colours_correctly() {
        let vxl = fixture_one_voxel_per_column();
        let chunk = decompress_chunk(&vxl);
        let n_cols = (chunk.vsid * chunk.vsid) as usize;
        assert_eq!(chunk.color_offsets.len(), n_cols + 1);
        assert_eq!(chunk.color_offsets[0], 0);
        // First column has CHUNK_Z - 100 colours (1 textured + bedrock).
        let per_col = CHUNK_Z - 100;
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
