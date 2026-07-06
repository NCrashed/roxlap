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
    /// Capacity of the grid's bounding box in chunks
    /// (`chunks_dims.x · y · z`) — an upper bound on `chunks.len()`,
    /// which may be smaller when interior chunks are absent.
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
