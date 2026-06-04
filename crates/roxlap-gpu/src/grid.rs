//! GPU.4 — grid-of-chunks upload + storage layout.
//!
//! Concatenates every chunk of one `roxlap-scene::Grid` into a few
//! flat storage buffers so a single compute dispatch can outer-DDA
//! through chunk-space + inner-DDA into any chunk it hits.
//!
#![allow(
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::doc_markdown,
    clippy::field_reassign_with_default
)]

//! Memory layout (post-bedrock-strip):
//!
//! * `occupancy[meta_idx]` — one chunk's 128 KiB occupancy slice
//!   starts at `meta_idx * vsid² * OCC_WORDS_PER_COLUMN` u32 words.
//!   Uniform per chunk (all chunks are vsid² × CHUNK_Z voxels).
//! * `color_offsets[meta_idx]` — one chunk's `vsid² + 1` u32
//!   offsets start at `meta_idx * (vsid² + 1)` u32 words. Uniform
//!   per chunk.
//! * `colors` — variable per chunk. Per-chunk base index lives in
//!   `chunk_colors_base[meta_idx]`.
//! * `chunk_occupancy` — 1 bit per chunk position. Bit at
//!   `meta_idx` set iff that chunk has any textured voxels. The
//!   outer DDA uses this to skip empty chunks in one step.
//!
//! The `meta_idx` for a chunk at `(chx, chy, chz)` is its row-major
//! offset within the grid's `chunks_dims` bounding box:
//!
//! ```text
//! rel = chunk_idx - origin_chunk
//! meta_idx = rel.x + rel.y * chunks_dims.x + rel.z * chunks_dims.x * chunks_dims.y
//! ```

use wgpu::util::DeviceExt;

use crate::decompress::{ChunkUpload, CHUNK_Z, OCC_WORDS_PER_COLUMN};

/// CPU-side aggregation of a grid's chunks ready to upload. Host
/// (e.g. `roxlap-scene-demo`) builds this by iterating its
/// `roxlap-scene::Grid` and calling [`crate::decompress_chunk`] per
/// materialised chunk.
pub struct GridUpload {
    /// Shared XY extent of every chunk in voxels. Matches
    /// `roxlap-scene::CHUNK_SIZE_XY = 128`.
    pub vsid: u32,
    /// Lowest chunk index present in the grid `(min_chx, min_chy,
    /// min_chz)`. The grid's bounding box runs from `origin_chunk`
    /// to `origin_chunk + chunks_dims` exclusive.
    pub origin_chunk: [i32; 3],
    /// Chunk-count along each axis = `max - min + 1`.
    pub chunks_dims: [u32; 3],
    /// GPU.7 slot-pool dimensions for modular chunk indexing.
    /// Every component MUST be a power of 2. A chunk at index
    /// `(chx, chy, chz)` maps to slot
    /// `(chx & (pool_dims.x - 1), chy & (pool_dims.y - 1),
    /// chz & (pool_dims.z - 1))`. As long as
    /// `pool_dims_axis ≥ active_range_along_axis`, no two
    /// simultaneously-resident chunks collide. Set this larger than
    /// `chunks_dims` only when streaming may install chunks at
    /// indices outside the initial bbox.
    pub pool_dims: [u32; 3],
    /// `(chunk_idx, decompressed)` pairs. Chunks outside the
    /// pool's collision-free active range are still accepted —
    /// modular indexing will assign them slots; the caller is
    /// responsible for avoiding collisions with other resident
    /// chunks.
    pub chunks: Vec<([i32; 3], ChunkUpload)>,
}

impl GridUpload {
    #[must_use]
    pub fn total_chunks(&self) -> u32 {
        self.chunks_dims[0] * self.chunks_dims[1] * self.chunks_dims[2]
    }

    /// Default GPU.7 [`Self::pool_dims`] derived from
    /// `chunks_dims` — each axis rounded up to the next power of 2.
    /// Use this when the grid is static + slots map 1:1 to bbox
    /// positions; for streaming grids, callers should pick a
    /// larger pool that covers `2 × r_active_chunks + 1` along
    /// each axis.
    #[must_use]
    pub fn default_pool_dims(chunks_dims: [u32; 3]) -> [u32; 3] {
        [
            ceil_pow2(chunks_dims[0]),
            ceil_pow2(chunks_dims[1]),
            ceil_pow2(chunks_dims[2]),
        ]
    }

    /// Linear chunk index `(meta_idx)` for `(chx, chy, chz)` in the
    /// grid's row-major bounding-box order. `None` if the index is
    /// outside the grid.
    #[must_use]
    pub fn meta_idx_of(&self, chunk_idx: [i32; 3]) -> Option<u32> {
        let dx = chunk_idx[0] - self.origin_chunk[0];
        let dy = chunk_idx[1] - self.origin_chunk[1];
        let dz = chunk_idx[2] - self.origin_chunk[2];
        if dx < 0
            || dy < 0
            || dz < 0
            || (dx as u32) >= self.chunks_dims[0]
            || (dy as u32) >= self.chunks_dims[1]
            || (dz as u32) >= self.chunks_dims[2]
        {
            return None;
        }
        Some(
            (dx as u32)
                + (dy as u32) * self.chunks_dims[0]
                + (dz as u32) * self.chunks_dims[0] * self.chunks_dims[1],
        )
    }
}

/// GPU-resident storage for one grid's chunks. Lives until the
/// host drops it; in GPU.6 (edit invalidation) we'll re-upload
/// individual chunks via partial buffer writes.
pub struct GpuGridResident {
    pub vsid: u32,
    pub origin_chunk: [i32; 3],
    pub chunks_dims: [u32; 3],
    pub total_chunks: u32,
    pub occupancy: wgpu::Buffer,
    pub color_offsets: wgpu::Buffer,
    pub colors: wgpu::Buffer,
    pub chunk_colors_base: wgpu::Buffer,
    pub chunk_occupancy: wgpu::Buffer,
    pub occupancy_bytes: u64,
    pub color_offsets_bytes: u64,
    pub colors_bytes: u64,
}

impl GpuGridResident {
    /// Pack + upload `info`. All buffers are sized to fit
    /// `total_chunks` regardless of which chunks are actually
    /// present in `info.chunks` — missing chunks have their
    /// `chunk_occupancy` bit clear and their per-chunk slices
    /// zero-filled, which the GPU.4 marcher reads as "no voxels
    /// here, skip".
    ///
    /// # Panics
    /// If a chunk's `vsid` doesn't match `info.vsid`.
    pub fn upload(device: &wgpu::Device, info: &GridUpload) -> Self {
        let vsid = info.vsid;
        let vsid_usize = vsid as usize;
        let cols_per_chunk = vsid_usize * vsid_usize;
        let occ_words_per_chunk = cols_per_chunk * (OCC_WORDS_PER_COLUMN as usize);
        let offsets_words_per_chunk = cols_per_chunk + 1;

        let total_chunks = info.total_chunks();
        let total_chunks_usize = total_chunks as usize;

        let mut occupancy = vec![0u32; total_chunks_usize * occ_words_per_chunk];
        let mut color_offsets = vec![0u32; total_chunks_usize * offsets_words_per_chunk];
        let mut chunk_colors_base = vec![0u32; total_chunks_usize];
        let mut chunk_occupancy = vec![0u32; total_chunks_usize.div_ceil(32)];
        let mut colors: Vec<u32> = Vec::new();

        let mut populated = 0u32;
        for (chunk_idx, chunk) in &info.chunks {
            let Some(meta_idx) = info.meta_idx_of(*chunk_idx) else {
                continue;
            };
            assert_eq!(
                chunk.vsid, vsid,
                "GpuGridResident: chunk vsid {} disagrees with grid vsid {}",
                chunk.vsid, vsid,
            );
            let meta_idx_us = meta_idx as usize;

            // Per-chunk occupancy slice.
            let occ_start = meta_idx_us * occ_words_per_chunk;
            occupancy[occ_start..occ_start + occ_words_per_chunk].copy_from_slice(&chunk.occupancy);

            // Per-chunk color_offsets slice. Rebase: the per-column
            // offsets in `chunk.color_offsets` are local to this
            // chunk's colours; the GPU shader adds
            // `chunk_colors_base[meta_idx]` on the outside, so the
            // values copy in verbatim.
            let off_start = meta_idx_us * offsets_words_per_chunk;
            color_offsets[off_start..off_start + offsets_words_per_chunk]
                .copy_from_slice(&chunk.color_offsets);

            // Append this chunk's colours; record the base.
            chunk_colors_base[meta_idx_us] =
                u32::try_from(colors.len()).expect("colours fit in u32");
            colors.extend_from_slice(&chunk.colors);

            // Mark chunk as non-empty iff it has any colour entries
            // (every textured voxel writes one). Empty chunks (all
            // air) skip cheaply in the outer DDA.
            if !chunk.colors.is_empty() {
                chunk_occupancy[meta_idx_us >> 5] |= 1u32 << (meta_idx_us & 31);
                populated += 1;
            }
        }

        // Storage buffers can't be empty — `colors` may be all-zero
        // length when every chunk is air. Pad with one sentinel u32
        // (never read because no chunk's `chunk_occupancy` bit is
        // set in that case).
        if colors.is_empty() {
            colors.push(0);
        }

        let occupancy_buf = create_storage(device, "roxlap-gpu grid.occupancy", &occupancy);
        let color_offsets_buf =
            create_storage(device, "roxlap-gpu grid.color_offsets", &color_offsets);
        let colors_buf = create_storage(device, "roxlap-gpu grid.colors", &colors);
        let chunk_colors_base_buf = create_storage(
            device,
            "roxlap-gpu grid.chunk_colors_base",
            &chunk_colors_base,
        );
        let chunk_occupancy_buf =
            create_storage(device, "roxlap-gpu grid.chunk_occupancy", &chunk_occupancy);

        let occupancy_bytes = (occupancy.len() * 4) as u64;
        let color_offsets_bytes = (color_offsets.len() * 4) as u64;
        let colors_bytes = (colors.len() * 4) as u64;
        let _ = populated; // reserved for future telemetry

        Self {
            vsid,
            origin_chunk: info.origin_chunk,
            chunks_dims: info.chunks_dims,
            total_chunks,
            occupancy: occupancy_buf,
            color_offsets: color_offsets_buf,
            colors: colors_buf,
            chunk_colors_base: chunk_colors_base_buf,
            chunk_occupancy: chunk_occupancy_buf,
            occupancy_bytes,
            color_offsets_bytes,
            colors_bytes,
        }
    }

    /// Total resident bytes — sum of all five storage buffers.
    /// Used by the demo's startup print.
    pub fn resident_bytes(&self) -> u64 {
        self.occupancy_bytes
            + self.color_offsets_bytes
            + self.colors_bytes
            + (self.total_chunks as u64) * 4 // chunk_colors_base
            + (u64::from(self.total_chunks).div_ceil(32)) * 4 // chunk_occupancy
    }
}

fn create_storage(device: &wgpu::Device, label: &str, data: &[u32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

/// Round `n` up to the nearest power of 2. `0` and `1` both return
/// `1`. Used to derive a GPU.7 [`GridUpload::pool_dims`] from a
/// non-pow2 `chunks_dims`.
#[must_use]
pub fn ceil_pow2(n: u32) -> u32 {
    if n <= 1 {
        return 1;
    }
    1u32 << (32 - (n - 1).leading_zeros())
}

/// Compute the smallest bounding box that contains every
/// `(chunk_idx, _)` in `chunks`. Returns `None` if `chunks` is
/// empty.
#[must_use]
pub fn bounding_box_of(chunks: impl IntoIterator<Item = [i32; 3]>) -> Option<([i32; 3], [u32; 3])> {
    let mut min = [i32::MAX; 3];
    let mut max = [i32::MIN; 3];
    let mut any = false;
    for idx in chunks {
        for i in 0..3 {
            if idx[i] < min[i] {
                min[i] = idx[i];
            }
            if idx[i] > max[i] {
                max[i] = idx[i];
            }
        }
        any = true;
    }
    if !any {
        return None;
    }
    #[allow(clippy::cast_sign_loss)]
    let dims = [
        (max[0] - min[0] + 1) as u32,
        (max[1] - min[1] + 1) as u32,
        (max[2] - min[2] + 1) as u32,
    ];
    Some((min, dims))
}

/// Number of u32 words a single chunk's per-chunk occupancy slice
/// occupies in the concatenated grid occupancy buffer. Useful for
/// host-side memory budgeting.
#[must_use]
pub fn occ_words_per_chunk(vsid: u32) -> u32 {
    vsid * vsid * OCC_WORDS_PER_COLUMN
}

/// Z-extent of every chunk — re-export of the `CHUNK_Z` constant
/// so hosts can budget without pulling `crate::decompress` in.
pub const GRID_CHUNK_Z: u32 = CHUNK_Z;
