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
    clippy::doc_markdown,
    clippy::missing_panics_doc,
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
/// buffers + the grid's voxel extents. Uploaded once.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug)]
pub struct GridStaticMeta {
    /// `occupancy` u32-word offset where this grid's data starts.
    pub occupancy_offset: u32,
    pub color_offsets_offset: u32,
    pub colors_offset: u32,
    pub chunk_colors_base_offset: u32,
    pub chunk_occupancy_offset: u32,
    pub vsid: u32,
    pub total_chunks: u32,
    pub _pad0: u32,
    pub chunks_dims: [u32; 3],
    pub _pad1: u32,
    pub origin_chunk: [i32; 3],
    pub _pad2: u32,
}

/// GPU-resident storage for an entire scene's grids.
pub struct GpuSceneResident {
    pub grid_count: u32,
    pub all_occupancy: wgpu::Buffer,
    pub all_color_offsets: wgpu::Buffer,
    pub all_colors: wgpu::Buffer,
    pub all_chunk_colors_base: wgpu::Buffer,
    pub all_chunk_occupancy: wgpu::Buffer,
    pub grid_static_meta: wgpu::Buffer,
    pub total_bytes: u64,
    /// Cached static metadata for the host's frame-loop
    /// `world_camera_for_grid(idx)` computations.
    pub static_meta: Vec<GridStaticMeta>,
    /// CPU shadow of per-chunk colour-slot base offsets (in u32
    /// words, relative to that grid's `colors_offset`). Indexed
    /// `[scene_idx][meta_idx]`. GPU.6 `refresh_chunk` uses this
    /// to write new colour bytes into the existing slot.
    pub(crate) chunk_colors_base: Vec<Vec<u32>>,
    /// Allocated colour-slot capacity per chunk (in u32 words).
    /// `[scene_idx][meta_idx]`. `refresh_chunk` truncates a chunk's
    /// new colour data to this length and warns on overflow.
    pub(crate) chunk_colors_capacity: Vec<Vec<u32>>,
    /// CPU shadow of the per-grid chunk-occupancy bitmap. Each entry
    /// is the u32 word at `chunk_occupancy_offset + (mi >> 5)`.
    /// `refresh_chunk` flips the right bit in this shadow + writes
    /// the affected word back to the GPU.
    pub(crate) chunk_occupancy_shadow: Vec<Vec<u32>>,
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
        let mut static_meta: Vec<GridStaticMeta> = Vec::with_capacity(info.grids.len());
        let mut chunk_colors_base: Vec<Vec<u32>> = Vec::with_capacity(info.grids.len());
        let mut chunk_colors_capacity: Vec<Vec<u32>> = Vec::with_capacity(info.grids.len());
        let mut chunk_occupancy_shadow: Vec<Vec<u32>> = Vec::with_capacity(info.grids.len());

        for grid in &info.grids {
            let vsid = grid.vsid;
            let cols_per_chunk = (vsid * vsid) as usize;
            let occ_words_per_chunk = cols_per_chunk * (OCC_WORDS_PER_COLUMN as usize);
            let offsets_words_per_chunk = cols_per_chunk + 1;
            let total_chunks = grid.total_chunks();
            let total_chunks_us = total_chunks as usize;

            let mut grid_occupancy = vec![0u32; total_chunks_us * occ_words_per_chunk];
            let mut grid_color_offsets = vec![0u32; total_chunks_us * offsets_words_per_chunk];
            let mut grid_colors: Vec<u32> = Vec::new();
            let mut grid_chunk_colors_base = vec![0u32; total_chunks_us];
            let mut grid_chunk_occupancy = vec![0u32; total_chunks_us.div_ceil(32)];

            for (chunk_idx, chunk) in &grid.chunks {
                let Some(meta_idx) = grid.meta_idx_of(*chunk_idx) else {
                    continue;
                };
                assert_eq!(chunk.vsid, vsid, "scene grid: chunk vsid mismatch");
                let mi = meta_idx as usize;
                let occ_start = mi * occ_words_per_chunk;
                grid_occupancy[occ_start..occ_start + occ_words_per_chunk]
                    .copy_from_slice(&chunk.occupancy);
                let off_start = mi * offsets_words_per_chunk;
                grid_color_offsets[off_start..off_start + offsets_words_per_chunk]
                    .copy_from_slice(&chunk.color_offsets);
                grid_chunk_colors_base[mi] =
                    u32::try_from(grid_colors.len()).expect("colours fit in u32");
                grid_colors.extend_from_slice(&chunk.colors);
                if !chunk.colors.is_empty() {
                    grid_chunk_occupancy[mi >> 5] |= 1u32 << (mi & 31);
                }
            }
            if grid_colors.is_empty() {
                grid_colors.push(0);
            }

            let meta = GridStaticMeta {
                occupancy_offset: u32::try_from(all_occupancy.len()).expect("fits"),
                color_offsets_offset: u32::try_from(all_color_offsets.len()).expect("fits"),
                colors_offset: u32::try_from(all_colors.len()).expect("fits"),
                chunk_colors_base_offset: u32::try_from(all_chunk_colors_base.len()).expect("fits"),
                chunk_occupancy_offset: u32::try_from(all_chunk_occupancy.len()).expect("fits"),
                vsid,
                total_chunks,
                _pad0: 0,
                chunks_dims: grid.chunks_dims,
                _pad1: 0,
                origin_chunk: grid.origin_chunk,
                _pad2: 0,
            };
            // Compute per-chunk slot capacities (= next-base
            // minus this-base; last slot gets trailing bytes up to
            // grid_colors.len()). Stored as cheap CPU shadow for
            // GPU.6 in-place re-uploads.
            let grid_colors_len = u32::try_from(grid_colors.len()).expect("fits");
            let mut grid_chunk_capacity = vec![0u32; total_chunks_us];
            for i in 0..total_chunks_us {
                let next_base = if i + 1 < total_chunks_us {
                    grid_chunk_colors_base[i + 1]
                } else {
                    grid_colors_len
                };
                grid_chunk_capacity[i] = next_base.saturating_sub(grid_chunk_colors_base[i]);
            }
            chunk_colors_base.push(grid_chunk_colors_base.clone());
            chunk_colors_capacity.push(grid_chunk_capacity);
            chunk_occupancy_shadow.push(grid_chunk_occupancy.clone());

            all_occupancy.extend_from_slice(&grid_occupancy);
            all_color_offsets.extend_from_slice(&grid_color_offsets);
            all_colors.extend_from_slice(&grid_colors);
            all_chunk_colors_base.extend_from_slice(&grid_chunk_colors_base);
            all_chunk_occupancy.extend_from_slice(&grid_chunk_occupancy);
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
        if static_meta.is_empty() {
            static_meta.push(GridStaticMeta::zeroed());
        }

        let occupancy_bytes = (all_occupancy.len() * 4) as u64;
        let color_offsets_bytes = (all_color_offsets.len() * 4) as u64;
        let colors_bytes = (all_colors.len() * 4) as u64;
        let chunk_colors_base_bytes = (all_chunk_colors_base.len() * 4) as u64;
        let chunk_occupancy_bytes = (all_chunk_occupancy.len() * 4) as u64;
        let static_meta_bytes = (static_meta.len() * std::mem::size_of::<GridStaticMeta>()) as u64;
        let total_bytes = occupancy_bytes
            + color_offsets_bytes
            + colors_bytes
            + chunk_colors_base_bytes
            + chunk_occupancy_bytes
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
            grid_static_meta,
            total_bytes,
            static_meta,
            chunk_colors_base,
            chunk_colors_capacity,
            chunk_occupancy_shadow,
        }
    }

    pub fn resident_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// GPU.6 — refresh one chunk's data in-place. Used by the host
    /// each frame when [`roxlap_scene::Grid::chunk_version`] reports
    /// a bump (e.g. the streaming-hills bake tracker re-baking a
    /// newly-installed chunk's neighbours).
    ///
    /// The chunk's slot in the concatenated buffers is identified
    /// by `(scene_idx, chunk_idx)`. Occupancy + `color_offsets` are
    /// fixed-size and always written; the colour data is written up
    /// to the slot's allocated capacity and truncated with a stderr
    /// warn beyond that (chunks growing past their initial colour
    /// count is a GPU.7 follow-up — sliding-window streaming).
    pub fn refresh_chunk(
        &mut self,
        queue: &wgpu::Queue,
        scene_idx: usize,
        chunk_idx: [i32; 3],
        chunk: &ChunkUpload,
    ) -> RefreshOutcome {
        let Some(meta) = self.static_meta.get(scene_idx) else {
            return RefreshOutcome::SceneIdxOob;
        };
        let dx = chunk_idx[0] - meta.origin_chunk[0];
        let dy = chunk_idx[1] - meta.origin_chunk[1];
        let dz = chunk_idx[2] - meta.origin_chunk[2];
        if dx < 0
            || dy < 0
            || dz < 0
            || (dx as u32) >= meta.chunks_dims[0]
            || (dy as u32) >= meta.chunks_dims[1]
            || (dz as u32) >= meta.chunks_dims[2]
        {
            return RefreshOutcome::ChunkOutOfBbox;
        }
        let meta_idx = ((dx as u32)
            + (dy as u32) * meta.chunks_dims[0]
            + (dz as u32) * meta.chunks_dims[0] * meta.chunks_dims[1])
            as usize;

        let vsid = meta.vsid as usize;
        let cols_per_chunk = vsid * vsid;
        let occ_words_per_chunk = cols_per_chunk * (OCC_WORDS_PER_COLUMN as usize);
        let offsets_words_per_chunk = cols_per_chunk + 1;

        assert_eq!(
            chunk.occupancy.len(),
            occ_words_per_chunk,
            "refresh_chunk: occupancy length mismatch",
        );
        assert_eq!(
            chunk.color_offsets.len(),
            offsets_words_per_chunk,
            "refresh_chunk: color_offsets length mismatch",
        );

        // ---- occupancy ----
        let occ_word_offset = meta.occupancy_offset as usize + meta_idx * occ_words_per_chunk;
        let occ_byte_offset = (occ_word_offset * 4) as u64;
        queue.write_buffer(
            &self.all_occupancy,
            occ_byte_offset,
            bytemuck::cast_slice(&chunk.occupancy),
        );

        // ---- color_offsets ----
        let off_word_offset =
            meta.color_offsets_offset as usize + meta_idx * offsets_words_per_chunk;
        let off_byte_offset = (off_word_offset * 4) as u64;
        queue.write_buffer(
            &self.all_color_offsets,
            off_byte_offset,
            bytemuck::cast_slice(&chunk.color_offsets),
        );

        // ---- colours (truncate to slot capacity) ----
        let slot_base = self.chunk_colors_base[scene_idx][meta_idx] as usize;
        let slot_capacity = self.chunk_colors_capacity[scene_idx][meta_idx] as usize;
        let new_len = chunk.colors.len();
        let outcome = if new_len > slot_capacity {
            eprintln!(
                "roxlap-gpu refresh_chunk: scene_idx={scene_idx} chunk_idx={chunk_idx:?} colours \
                 {new_len} > slot capacity {slot_capacity}; truncating (GPU.7 sliding pool fixes this)",
            );
            RefreshOutcome::ColorsTruncated
        } else {
            RefreshOutcome::Ok
        };
        let write_len = new_len.min(slot_capacity);
        if write_len > 0 {
            let colors_word_offset = meta.colors_offset as usize + slot_base;
            let colors_byte_offset = (colors_word_offset * 4) as u64;
            queue.write_buffer(
                &self.all_colors,
                colors_byte_offset,
                bytemuck::cast_slice(&chunk.colors[..write_len]),
            );
        }

        // ---- chunk_occupancy bit (read-modify-write the word) ----
        let chunk_bit_word_idx = meta_idx >> 5;
        let chunk_bit = meta_idx & 31;
        let shadow = &mut self.chunk_occupancy_shadow[scene_idx][chunk_bit_word_idx];
        let new_bit = !chunk.colors.is_empty();
        let was_bit = (*shadow >> chunk_bit) & 1 == 1;
        if new_bit != was_bit {
            if new_bit {
                *shadow |= 1u32 << chunk_bit;
            } else {
                *shadow &= !(1u32 << chunk_bit);
            }
            let global_word_idx = meta.chunk_occupancy_offset as usize + chunk_bit_word_idx;
            queue.write_buffer(
                &self.all_chunk_occupancy,
                (global_word_idx * 4) as u64,
                bytemuck::bytes_of(shadow),
            );
        }

        outcome
    }
}

/// Outcome of `GpuSceneResident::refresh_chunk`. Most callers
/// can ignore the result; `ColorsTruncated` indicates the chunk
/// grew past its slot capacity and GPU.7 sliding-window streaming
/// is the proper fix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefreshOutcome {
    Ok,
    /// The chunk's new colour count exceeded its pre-allocated
    /// slot capacity. The chunk was truncated; GPU will render
    /// the first `slot_capacity` colours.
    ColorsTruncated,
    /// `chunk_idx` is outside the scene's `(origin_chunk,
    /// chunks_dims)` bbox for `scene_idx`. The chunk wasn't
    /// refreshed; GPU.7 will install new chunks via the sliding
    /// window.
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
