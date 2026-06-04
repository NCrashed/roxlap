//! GPU.5 — multi-grid scene upload + shared storage layout.
//!
//! Concatenates every chunk of every grid into one set of storage
//! buffers + a per-grid offsets table. Each grid keeps its own
//! `vsid`, `chunks_dims`, `origin_chunk`, and runtime transform;
//! the shader iterates grids 0..grid_count, transforms the world
//! camera into each grid's local frame, runs that grid's outer-DDA
//! over chunks, and tracks the closest hit across all grids.
//!
//! Why concatenate rather than one bind group per grid? wgpu's
//! `MAX_BIND_GROUPS` default is 4; demos with 10+ grids
//! (`roxlap-scene-demo` has ground + ship + 10 marker pillars =
//! 12) need a single bind-group layout that scales.

#![allow(
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::doc_markdown,
    clippy::missing_panics_doc,
    clippy::needless_range_loop,
    clippy::pub_underscore_fields
)]

use bytemuck::Zeroable;
use wgpu::util::DeviceExt;

use crate::decompress::{ChunkUpload, OCC_WORDS_PER_COLUMN};
use crate::grid::GridUpload;

/// Maximum number of grids the shader's per-grid camera uniform
/// array can hold. The scene-demo has 12 (1 ground + 1 ship + 10
/// markers); 16 leaves headroom for a future +4 without re-cooking
/// the shader. The runtime check rejects scenes that overflow.
pub const MAX_SCENE_GRIDS: u32 = 16;

/// Per-chunk colour-slot stride, in u32 words (256 KiB). Each
/// chunk's colour data lives at `meta_idx * COLORS_PER_CHUNK_WORDS`
/// within its grid's colours range. Fixed-stride layout means
/// every slot — present or absent at upload time — has the same
/// capacity, so [`GpuSceneResident::refresh_chunk`] can always
/// write new colour data into the slot when a chunk arrives via
/// streaming or is re-baked.
///
/// 65536 u32s = 256 KiB. Scene-demo's densest ground-hills chunks
/// run ~36 k colour entries (~144 KiB) — multiple textured voxels
/// per column at slopes/cliffs; 256 KiB gives ~1.8× headroom.
/// Memory cost on the demo's 32×32×1 static grid: 1024 slots ×
/// 256 KiB = 256 MiB colours (~830 MiB resident scene total).
/// Chunks past the cap truncate with a stderr warn; GPU.7
/// sliding-window storage removes the cap entirely.
pub const COLORS_PER_CHUNK_WORDS: u32 = 65536;

/// Per-grid runtime transform — voxlap-style (world → grid-local).
/// `rotation` is column-major and encodes the inverse rotation
/// applied to the world camera basis before passing it to that
/// grid's marcher. Identity for the ground; non-trivial for the
/// rotating ship.
#[derive(Debug, Clone, Copy)]
pub struct GridRuntimeTransform {
    /// Grid-local position of the world origin = `-rotation⁻¹ ·
    /// grid.position` for a `GridTransform { position, rotation }`.
    /// The host computes this once per frame.
    pub grid_origin_world: [f64; 3],
    /// 3×3 inverse rotation (column-major).
    pub world_to_grid_rotation: [[f32; 3]; 3],
}

impl Default for GridRuntimeTransform {
    fn default() -> Self {
        Self {
            grid_origin_world: [0.0, 0.0, 0.0],
            world_to_grid_rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }
}

/// CPU-side aggregation of every grid in a scene. Built once at
/// startup; per-grid transforms are recomputed each frame and
/// passed to `render_scene` separately.
pub struct SceneUpload {
    pub grids: Vec<GridUpload>,
}

impl SceneUpload {
    #[must_use]
    pub fn grid_count(&self) -> u32 {
        u32::try_from(self.grids.len()).unwrap_or(u32::MAX)
    }
}

/// Per-grid static metadata: offsets into the concatenated storage
/// buffers + the grid's slot-pool dimensions. Uploaded once.
///
/// GPU.7 changes: `chunks_dims` and `origin_chunk` were dropped.
/// The shader uses modular slot indexing
/// (`chunk_idx & (pool_dims - 1)`) and verifies slot identity via
/// `slot_chunk_idx[slot]`, so the upload-time bbox is no longer
/// relevant to the shader.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug)]
pub struct GridStaticMeta {
    /// `occupancy` u32-word offset where this grid's data starts.
    pub occupancy_offset: u32,
    pub color_offsets_offset: u32,
    pub colors_offset: u32,
    pub chunk_colors_base_offset: u32,
    pub chunk_occupancy_offset: u32,
    /// New in GPU.7: u32-word offset where this grid's
    /// `slot_chunk_idx` array starts (one `vec3<i32>` per slot,
    /// i.e. 3 u32 words each, plus 1 padding word for std430).
    pub slot_chunk_idx_offset: u32,
    pub vsid: u32,
    pub total_slots: u32,
    pub pool_dims: [u32; 3],
    pub _pad0: u32,
}

/// Sentinel chunk_idx written into empty slot_chunk_idx entries.
/// Real chunk indices never use `i32::MIN`, so the shader can
/// distinguish empty slots from collisions via a single equality
/// check.
pub const SLOT_EMPTY_SENTINEL: [i32; 3] = [i32::MIN, i32::MIN, i32::MIN];

/// GPU-resident storage for an entire scene's grids.
pub struct GpuSceneResident {
    pub grid_count: u32,
    pub all_occupancy: wgpu::Buffer,
    pub all_color_offsets: wgpu::Buffer,
    pub all_colors: wgpu::Buffer,
    pub all_chunk_colors_base: wgpu::Buffer,
    pub all_chunk_occupancy: wgpu::Buffer,
    /// GPU.7 — per-slot chunk_idx for identity verification in the
    /// shader. Stored as `vec3<i32>` with std430 16-byte stride
    /// (each entry is `[i32; 4]` on the host: x, y, z, _pad).
    pub all_slot_chunk_idx: wgpu::Buffer,
    pub grid_static_meta: wgpu::Buffer,
    pub total_bytes: u64,
    /// Cached static metadata for the host's frame-loop work.
    pub static_meta: Vec<GridStaticMeta>,
    /// CPU shadow of the per-grid chunk-occupancy bitmap. Each entry
    /// is the u32 word at `chunk_occupancy_offset + (mi >> 5)`.
    /// `refresh_chunk` / `evict_chunk` flip the right bit + write
    /// the affected word back to the GPU.
    pub(crate) chunk_occupancy_shadow: Vec<Vec<u32>>,
    /// CPU shadow of `slot_chunk_idx`. Indexed `[scene_idx][slot]`
    /// → `[i32; 4]` (vec3 + pad). Host uses this to detect "slot is
    /// holding a different chunk than expected" + as the eviction
    /// origin.
    pub(crate) slot_chunk_idx_shadow: Vec<Vec<[i32; 4]>>,
}

impl GpuSceneResident {
    /// Pack + upload `info`. Each grid is uploaded as a contiguous
    /// slab inside the shared storage buffers; per-grid offsets
    /// live in `grid_static_meta`.
    ///
    /// # Panics
    /// If `info.grids.len() > MAX_SCENE_GRIDS`.
    pub fn upload(device: &wgpu::Device, info: &SceneUpload) -> Self {
        let grid_count = info.grid_count();
        assert!(
            grid_count <= MAX_SCENE_GRIDS,
            "GpuSceneResident: scene has {grid_count} grids, shader supports {MAX_SCENE_GRIDS}",
        );

        let mut all_occupancy: Vec<u32> = Vec::new();
        let mut all_color_offsets: Vec<u32> = Vec::new();
        let mut all_colors: Vec<u32> = Vec::new();
        let mut all_chunk_colors_base: Vec<u32> = Vec::new();
        let mut all_chunk_occupancy: Vec<u32> = Vec::new();
        let mut all_slot_chunk_idx: Vec<i32> = Vec::new();
        let mut static_meta: Vec<GridStaticMeta> = Vec::with_capacity(info.grids.len());
        let mut chunk_occupancy_shadow: Vec<Vec<u32>> = Vec::with_capacity(info.grids.len());
        let mut slot_chunk_idx_shadow: Vec<Vec<[i32; 4]>> = Vec::with_capacity(info.grids.len());

        for grid in &info.grids {
            let vsid = grid.vsid;
            let cols_per_chunk = (vsid * vsid) as usize;
            let occ_words_per_slot = cols_per_chunk * (OCC_WORDS_PER_COLUMN as usize);
            let offsets_words_per_slot = cols_per_chunk + 1;
            let colors_stride = COLORS_PER_CHUNK_WORDS as usize;

            // Validate pool_dims are powers of 2 — required for the
            // shader's `chunk_idx & (pool_dims - 1)` modular slot
            // indexing.
            assert!(
                grid.pool_dims[0].is_power_of_two()
                    && grid.pool_dims[1].is_power_of_two()
                    && grid.pool_dims[2].is_power_of_two(),
                "scene grid: pool_dims {:?} must all be powers of 2",
                grid.pool_dims,
            );
            let pool_x = grid.pool_dims[0] as usize;
            let pool_y = grid.pool_dims[1] as usize;
            let pool_z = grid.pool_dims[2] as usize;
            let total_slots = pool_x * pool_y * pool_z;

            let mut grid_occupancy = vec![0u32; total_slots * occ_words_per_slot];
            let mut grid_color_offsets = vec![0u32; total_slots * offsets_words_per_slot];
            let mut grid_colors = vec![0u32; total_slots * colors_stride];
            let mut grid_chunk_colors_base = vec![0u32; total_slots];
            for i in 0..total_slots {
                grid_chunk_colors_base[i] = (i * colors_stride) as u32;
            }
            let mut grid_chunk_occupancy = vec![0u32; total_slots.div_ceil(32)];
            // slot_chunk_idx: vec3<i32> per slot, std430 stride = 16
            // bytes (4 u32 words: x, y, z, _pad). Initialise every
            // slot to the empty sentinel; populated slots overwrite
            // with the actual chunk_idx below.
            let mut grid_slot_chunk_idx: Vec<[i32; 4]> = Vec::with_capacity(total_slots);
            for _ in 0..total_slots {
                grid_slot_chunk_idx.push([
                    SLOT_EMPTY_SENTINEL[0],
                    SLOT_EMPTY_SENTINEL[1],
                    SLOT_EMPTY_SENTINEL[2],
                    0,
                ]);
            }

            let mask_x = (grid.pool_dims[0] - 1) as i32;
            let mask_y = (grid.pool_dims[1] - 1) as i32;
            let mask_z = (grid.pool_dims[2] - 1) as i32;
            let chunks_per_layer = pool_x * pool_y;

            for (chunk_idx, chunk) in &grid.chunks {
                assert_eq!(chunk.vsid, vsid, "scene grid: chunk vsid mismatch");
                let sx = (chunk_idx[0] & mask_x) as usize;
                let sy = (chunk_idx[1] & mask_y) as usize;
                let sz = (chunk_idx[2] & mask_z) as usize;
                let slot_idx = sx + sy * pool_x + sz * chunks_per_layer;

                let occ_start = slot_idx * occ_words_per_slot;
                grid_occupancy[occ_start..occ_start + occ_words_per_slot]
                    .copy_from_slice(&chunk.occupancy);
                let off_start = slot_idx * offsets_words_per_slot;
                grid_color_offsets[off_start..off_start + offsets_words_per_slot]
                    .copy_from_slice(&chunk.color_offsets);

                let col_start = slot_idx * colors_stride;
                let n = chunk.colors.len().min(colors_stride);
                if chunk.colors.len() > colors_stride {
                    eprintln!(
                        "roxlap-gpu SceneUpload: scene grid chunk {chunk_idx:?} has {} colours \
                         > COLORS_PER_CHUNK_WORDS ({colors_stride}); truncating",
                        chunk.colors.len(),
                    );
                }
                grid_colors[col_start..col_start + n].copy_from_slice(&chunk.colors[..n]);

                if !chunk.colors.is_empty() {
                    grid_chunk_occupancy[slot_idx >> 5] |= 1u32 << (slot_idx & 31);
                }
                grid_slot_chunk_idx[slot_idx] = [chunk_idx[0], chunk_idx[1], chunk_idx[2], 0];
            }

            // Slot_chunk_idx storage offset: each entry is 4 u32
            // words (vec3 padded to 16 bytes in std430).
            let slot_chunk_idx_offset = u32::try_from(all_slot_chunk_idx.len()).expect("fits");
            let meta = GridStaticMeta {
                occupancy_offset: u32::try_from(all_occupancy.len()).expect("fits"),
                color_offsets_offset: u32::try_from(all_color_offsets.len()).expect("fits"),
                colors_offset: u32::try_from(all_colors.len()).expect("fits"),
                chunk_colors_base_offset: u32::try_from(all_chunk_colors_base.len()).expect("fits"),
                chunk_occupancy_offset: u32::try_from(all_chunk_occupancy.len()).expect("fits"),
                slot_chunk_idx_offset,
                vsid,
                total_slots: total_slots as u32,
                pool_dims: grid.pool_dims,
                _pad0: 0,
            };

            chunk_occupancy_shadow.push(grid_chunk_occupancy.clone());
            slot_chunk_idx_shadow.push(grid_slot_chunk_idx.clone());

            all_occupancy.extend_from_slice(&grid_occupancy);
            all_color_offsets.extend_from_slice(&grid_color_offsets);
            all_colors.extend_from_slice(&grid_colors);
            all_chunk_colors_base.extend_from_slice(&grid_chunk_colors_base);
            all_chunk_occupancy.extend_from_slice(&grid_chunk_occupancy);
            for entry in &grid_slot_chunk_idx {
                all_slot_chunk_idx.extend_from_slice(entry);
            }
            static_meta.push(meta);
        }

        // Pad an empty scene's storage buffers — wgpu rejects
        // zero-size storage bindings.
        if all_occupancy.is_empty() {
            all_occupancy.push(0);
        }
        if all_color_offsets.is_empty() {
            all_color_offsets.push(0);
        }
        if all_colors.is_empty() {
            all_colors.push(0);
        }
        if all_chunk_colors_base.is_empty() {
            all_chunk_colors_base.push(0);
        }
        if all_chunk_occupancy.is_empty() {
            all_chunk_occupancy.push(0);
        }
        if all_slot_chunk_idx.is_empty() {
            // 4 zeros = single padded vec3<i32>. wgpu rejects
            // zero-sized storage bindings.
            all_slot_chunk_idx.extend_from_slice(&[0; 4]);
        }
        if static_meta.is_empty() {
            static_meta.push(GridStaticMeta::zeroed());
        }

        let occupancy_bytes = (all_occupancy.len() * 4) as u64;
        let color_offsets_bytes = (all_color_offsets.len() * 4) as u64;
        let colors_bytes = (all_colors.len() * 4) as u64;
        let chunk_colors_base_bytes = (all_chunk_colors_base.len() * 4) as u64;
        let chunk_occupancy_bytes = (all_chunk_occupancy.len() * 4) as u64;
        let slot_chunk_idx_bytes = (all_slot_chunk_idx.len() * 4) as u64;
        let static_meta_bytes = (static_meta.len() * std::mem::size_of::<GridStaticMeta>()) as u64;
        let total_bytes = occupancy_bytes
            + color_offsets_bytes
            + colors_bytes
            + chunk_colors_base_bytes
            + chunk_occupancy_bytes
            + slot_chunk_idx_bytes
            + static_meta_bytes;

        let all_occupancy = create_storage(device, "roxlap-gpu scene.occupancy", &all_occupancy);
        let all_color_offsets =
            create_storage(device, "roxlap-gpu scene.color_offsets", &all_color_offsets);
        let all_colors = create_storage(device, "roxlap-gpu scene.colors", &all_colors);
        let all_chunk_colors_base = create_storage(
            device,
            "roxlap-gpu scene.chunk_colors_base",
            &all_chunk_colors_base,
        );
        let all_chunk_occupancy = create_storage(
            device,
            "roxlap-gpu scene.chunk_occupancy",
            &all_chunk_occupancy,
        );
        // GPU.7 slot identity verification buffer. i32 storage.
        let all_slot_chunk_idx_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("roxlap-gpu scene.slot_chunk_idx"),
            contents: bytemuck::cast_slice(&all_slot_chunk_idx),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let grid_static_meta = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("roxlap-gpu scene.grid_static_meta"),
            contents: bytemuck::cast_slice(&static_meta),
            usage: wgpu::BufferUsages::STORAGE,
        });

        Self {
            grid_count,
            all_occupancy,
            all_color_offsets,
            all_colors,
            all_chunk_colors_base,
            all_chunk_occupancy,
            all_slot_chunk_idx: all_slot_chunk_idx_buf,
            grid_static_meta,
            total_bytes,
            static_meta,
            chunk_occupancy_shadow,
            slot_chunk_idx_shadow,
        }
    }

    pub fn resident_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// Install or refresh a chunk in its modular pool slot. GPU.7
    /// generalises GPU.6's in-place refresh: any chunk_idx maps to
    /// a slot via `chunk_idx & (pool_dims - 1)`. The previous
    /// occupant (if a different chunk) is silently replaced — the
    /// host is responsible for guaranteeing that the pool is sized
    /// large enough that two simultaneously-resident chunks never
    /// collide on the same slot.
    pub fn refresh_chunk(
        &mut self,
        queue: &wgpu::Queue,
        scene_idx: usize,
        chunk_idx: [i32; 3],
        chunk: &ChunkUpload,
    ) -> RefreshOutcome {
        let Some(meta) = self.static_meta.get(scene_idx).copied() else {
            return RefreshOutcome::SceneIdxOob;
        };
        let slot_idx = modular_slot_idx(chunk_idx, meta.pool_dims);

        let vsid = meta.vsid as usize;
        let cols_per_chunk = vsid * vsid;
        let occ_words_per_slot = cols_per_chunk * (OCC_WORDS_PER_COLUMN as usize);
        let offsets_words_per_slot = cols_per_chunk + 1;
        let colors_stride = COLORS_PER_CHUNK_WORDS as usize;

        assert_eq!(
            chunk.occupancy.len(),
            occ_words_per_slot,
            "refresh_chunk: occupancy length mismatch",
        );
        assert_eq!(
            chunk.color_offsets.len(),
            offsets_words_per_slot,
            "refresh_chunk: color_offsets length mismatch",
        );

        // ---- occupancy ----
        let occ_word_offset = meta.occupancy_offset as usize + slot_idx * occ_words_per_slot;
        queue.write_buffer(
            &self.all_occupancy,
            (occ_word_offset * 4) as u64,
            bytemuck::cast_slice(&chunk.occupancy),
        );

        // ---- color_offsets ----
        let off_word_offset =
            meta.color_offsets_offset as usize + slot_idx * offsets_words_per_slot;
        queue.write_buffer(
            &self.all_color_offsets,
            (off_word_offset * 4) as u64,
            bytemuck::cast_slice(&chunk.color_offsets),
        );

        // ---- colours (truncate to slot stride) ----
        let new_len = chunk.colors.len();
        let outcome = if new_len > colors_stride {
            eprintln!(
                "roxlap-gpu refresh_chunk: scene_idx={scene_idx} chunk_idx={chunk_idx:?} colours \
                 {new_len} > stride {colors_stride}; truncating",
            );
            RefreshOutcome::ColorsTruncated
        } else {
            RefreshOutcome::Ok
        };
        let write_len = new_len.min(colors_stride);
        if write_len > 0 {
            let colors_word_offset = meta.colors_offset as usize + slot_idx * colors_stride;
            queue.write_buffer(
                &self.all_colors,
                (colors_word_offset * 4) as u64,
                bytemuck::cast_slice(&chunk.colors[..write_len]),
            );
        }

        // ---- chunk_occupancy bit ----
        self.set_chunk_occupancy_bit(queue, scene_idx, &meta, slot_idx, !chunk.colors.is_empty());

        // ---- slot_chunk_idx (identity for the shader) ----
        self.set_slot_chunk_idx(queue, scene_idx, &meta, slot_idx, chunk_idx);

        outcome
    }

    /// Evict a chunk's slot — clear its `chunk_occupancy` bit and
    /// reset `slot_chunk_idx` to the empty sentinel. Used by the
    /// host when a chunk disappears from the CPU-side `Grid::chunks`
    /// (e.g. streaming eviction past `r_evict`).
    ///
    /// Returns `false` if `scene_idx` is past `grid_count` (no-op);
    /// `true` otherwise.
    pub fn evict_chunk(
        &mut self,
        queue: &wgpu::Queue,
        scene_idx: usize,
        chunk_idx: [i32; 3],
    ) -> bool {
        let Some(meta) = self.static_meta.get(scene_idx).copied() else {
            return false;
        };
        let slot_idx = modular_slot_idx(chunk_idx, meta.pool_dims);
        // Only evict if this slot still claims to hold `chunk_idx`.
        // Otherwise we'd be wiping out a different (newer) chunk
        // that happens to share the slot.
        let shadow_entry = self.slot_chunk_idx_shadow[scene_idx][slot_idx];
        if shadow_entry[0] != chunk_idx[0]
            || shadow_entry[1] != chunk_idx[1]
            || shadow_entry[2] != chunk_idx[2]
        {
            return true;
        }
        self.set_chunk_occupancy_bit(queue, scene_idx, &meta, slot_idx, false);
        self.set_slot_chunk_idx(queue, scene_idx, &meta, slot_idx, SLOT_EMPTY_SENTINEL);
        true
    }

    fn set_chunk_occupancy_bit(
        &mut self,
        queue: &wgpu::Queue,
        scene_idx: usize,
        meta: &GridStaticMeta,
        slot_idx: usize,
        new_bit: bool,
    ) {
        let word_idx = slot_idx >> 5;
        let bit = slot_idx & 31;
        let shadow = &mut self.chunk_occupancy_shadow[scene_idx][word_idx];
        let was_bit = (*shadow >> bit) & 1 == 1;
        if new_bit == was_bit {
            return;
        }
        if new_bit {
            *shadow |= 1u32 << bit;
        } else {
            *shadow &= !(1u32 << bit);
        }
        let global_word_idx = meta.chunk_occupancy_offset as usize + word_idx;
        queue.write_buffer(
            &self.all_chunk_occupancy,
            (global_word_idx * 4) as u64,
            bytemuck::bytes_of(shadow),
        );
    }

    fn set_slot_chunk_idx(
        &mut self,
        queue: &wgpu::Queue,
        scene_idx: usize,
        meta: &GridStaticMeta,
        slot_idx: usize,
        chunk_idx: [i32; 3],
    ) {
        let entry = [chunk_idx[0], chunk_idx[1], chunk_idx[2], 0];
        self.slot_chunk_idx_shadow[scene_idx][slot_idx] = entry;
        let global_word_idx = meta.slot_chunk_idx_offset as usize + slot_idx * 4;
        queue.write_buffer(
            &self.all_slot_chunk_idx,
            (global_word_idx * 4) as u64,
            bytemuck::cast_slice(&entry),
        );
    }
}

/// Modular slot index for `chunk_idx` given the grid's
/// power-of-2 `pool_dims`. Negative `chunk_idx` components map via
/// two's-complement bitwise AND, matching the shader's
/// `chunk_idx & (pool_dims - 1)`.
#[must_use]
pub fn modular_slot_idx(chunk_idx: [i32; 3], pool_dims: [u32; 3]) -> usize {
    let mask_x = (pool_dims[0] - 1) as i32;
    let mask_y = (pool_dims[1] - 1) as i32;
    let mask_z = (pool_dims[2] - 1) as i32;
    let sx = (chunk_idx[0] & mask_x) as usize;
    let sy = (chunk_idx[1] & mask_y) as usize;
    let sz = (chunk_idx[2] & mask_z) as usize;
    sx + sy * (pool_dims[0] as usize) + sz * (pool_dims[0] as usize) * (pool_dims[1] as usize)
}

/// Outcome of `GpuSceneResident::refresh_chunk`. Most callers
/// can ignore the result; `ColorsTruncated` indicates the chunk's
/// colour data overflowed the per-slot stride and was clipped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefreshOutcome {
    Ok,
    /// The chunk's colour count exceeded `COLORS_PER_CHUNK_WORDS`;
    /// the GPU sees the first `stride` colours. Bump
    /// `COLORS_PER_CHUNK_WORDS` for content that hits this.
    ColorsTruncated,
    /// Retained for ABI compatibility; the GPU.7 modular pool no
    /// longer rejects chunks by bbox.
    ChunkOutOfBbox,
    /// `scene_idx` is past `grid_count`. Programming error.
    SceneIdxOob,
}

fn create_storage(device: &wgpu::Device, label: &str, data: &[u32]) -> wgpu::Buffer {
    // GPU.6: include COPY_DST so `refresh_chunk` can `queue.write_buffer`
    // into existing slots without rebuilding the resident.
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    })
}
