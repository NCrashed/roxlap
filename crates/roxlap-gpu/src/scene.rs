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

use crate::decompress::OCC_WORDS_PER_COLUMN;
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
        }
    }

    pub fn resident_bytes(&self) -> u64 {
        self.total_bytes
    }
}

fn create_storage(device: &wgpu::Device, label: &str, data: &[u32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE,
    })
}
