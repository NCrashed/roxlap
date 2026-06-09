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

use crate::decompress::{gpu_mip_count, occ_words_per_column_for_mip, ChunkUpload};
use crate::grid::GridUpload;

/// GPU.11 — max mip levels the per-slot layout reserves room for in
/// [`GridStaticMeta`]'s relative-offset tables. Matches
/// [`crate::decompress::GPU_MAX_MIPS`]; the shader's `array<u32, N>`
/// must use the same N.
pub const MAX_GPU_MIPS: usize = 6;

/// GPU.11 — per-slot occupancy/color-offset strides + per-mip
/// within-slot relative offsets for a grid of side `vsid`. All
/// chunks of a grid share these (uniform mip count by
/// [`gpu_mip_count`]). `colors` keep their fixed
/// [`COLORS_PER_CHUNK_WORDS`] stride; each mip's colours are
/// concatenated within that block and indexed by the chunk's own
/// (absolute) `color_offsets`.
#[derive(Debug, Clone, Copy)]
pub struct MipLayout {
    pub mip_count: u32,
    pub occ_words_per_slot: u32,
    pub offsets_words_per_slot: u32,
    /// Within-slot u32 offset where mip `m`'s occupancy starts.
    pub mip_occ_rel: [u32; MAX_GPU_MIPS],
    /// Within-slot u32 offset where mip `m`'s color_offsets start.
    pub mip_coff_rel: [u32; MAX_GPU_MIPS],
}

impl MipLayout {
    #[must_use]
    pub fn for_vsid(vsid: u32) -> Self {
        let mip_count = gpu_mip_count(vsid);
        let mut mip_occ_rel = [0u32; MAX_GPU_MIPS];
        let mut mip_coff_rel = [0u32; MAX_GPU_MIPS];
        let mut occ_acc = 0u32;
        let mut coff_acc = 0u32;
        for m in 0..mip_count {
            mip_occ_rel[m as usize] = occ_acc;
            mip_coff_rel[m as usize] = coff_acc;
            let vsid_m = vsid >> m;
            let cols = vsid_m * vsid_m;
            // Each mip stores TWO bitmaps back-to-back: the textured
            // occupancy then the solid occupancy (cliff-face fix). The
            // shader reads solid at `tex_base + cols*occ_words_per_col`.
            occ_acc += 2 * cols * occ_words_per_column_for_mip(m);
            coff_acc += cols + 1;
        }
        Self {
            mip_count,
            occ_words_per_slot: occ_acc,
            offsets_words_per_slot: coff_acc,
            mip_occ_rel,
            mip_coff_rel,
        }
    }
}

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

/// Number of separate storage bindings the concatenated occupancy
/// buffer is split ("paged") across. A single storage binding may
/// not exceed the device's `max_storage_buffer_binding_size` — on
/// strict drivers that's a hard 128 MiB (lavapipe), which the
/// streaming demo's occupancy already reaches. Splitting into pages
/// keeps every binding under the limit while preserving a single
/// global word index in the shader (each page is a whole number of
/// chunk slots, so no slot ever straddles a page boundary).
///
/// On GPUs with multi-GiB binding limits (NVK, native Vulkan) the
/// whole buffer fits in page 0, the other bindings get a 1-word
/// dummy, and the shader's page select is a single perfectly-
/// predicted uniform branch → zero hot-loop cost. 4 pages covers
/// 512 MiB of occupancy even on a 128 MiB-per-binding device.
pub const MAX_OCC_PAGES: usize = 4;

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
    /// GPU.11 — per-slot occupancy stride (sum over all mips).
    /// `meta_id`'s occupancy slab starts at
    /// `occupancy_offset + meta_id * occ_words_per_slot`.
    pub occ_words_per_slot: u32,
    /// GPU.11 — per-slot color_offsets stride (sum over all mips).
    pub offsets_words_per_slot: u32,
    /// GPU.11 — number of mip levels stored per slot.
    pub mip_count: u32,
    pub _pad1: u32,
    /// GPU.11 — within-slot u32 offset where mip `m`'s occupancy
    /// starts. `mip_occ_rel[0] == 0` so mip-0 reads are unchanged.
    pub mip_occ_rel: [u32; MAX_GPU_MIPS],
    /// GPU.11 — within-slot u32 offset where mip `m`'s color_offsets
    /// start. `mip_coff_rel[0] == 0`.
    pub mip_coff_rel: [u32; MAX_GPU_MIPS],
    /// GPU.13.0 — occupied chunk-AABB (inclusive) in chunk-index space.
    /// The outer DDA stops once `p_chunk` passes this box along the
    /// ray's travel direction (no resident chunk can lie ahead). An
    /// empty grid uses the inverted sentinel (`aabb_min = i32::MAX`,
    /// `aabb_max = i32::MIN`) so every ray early-outs immediately.
    /// Maintained live: [`GpuSceneResident::refresh_chunk`] /
    /// [`GpuSceneResident::evict_chunk`] recompute + re-upload it.
    pub aabb_min: [i32; 3],
    pub _pad2: i32,
    pub aabb_max: [i32; 3],
    pub _pad3: i32,
}

/// Sentinel chunk_idx written into empty slot_chunk_idx entries.
/// Real chunk indices never use `i32::MIN`, so the shader can
/// distinguish empty slots from collisions via a single equality
/// check.
pub const SLOT_EMPTY_SENTINEL: [i32; 3] = [i32::MIN, i32::MIN, i32::MIN];

/// GPU-resident storage for an entire scene's grids.
pub struct GpuSceneResident {
    pub grid_count: u32,
    /// Concatenated per-slot occupancy, split into up to
    /// [`MAX_OCC_PAGES`] storage bindings so no single binding
    /// exceeds the device's `max_storage_buffer_binding_size`. The
    /// vec is always exactly `MAX_OCC_PAGES` long — pages past
    /// `occupancy_num_pages` are 1-word dummies kept only so the
    /// bind group has a buffer for every layout entry. Page p holds
    /// the global word range `[p*occupancy_page_words,
    /// (p+1)*occupancy_page_words)`; `occupancy_page_words` is a
    /// whole number of chunk slots so no slot straddles a boundary.
    pub occupancy_pages: Vec<wgpu::Buffer>,
    /// Words per occupancy page (a multiple of `occ_words_per_slot`).
    pub occupancy_page_words: u32,
    /// Number of real (non-dummy) pages in `occupancy_pages`.
    pub occupancy_num_pages: u32,
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
            // GPU.11 — per-slot strides span the whole mip ladder.
            let layout = MipLayout::for_vsid(vsid);
            let occ_words_per_slot = layout.occ_words_per_slot as usize;
            let offsets_words_per_slot = layout.offsets_words_per_slot as usize;
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

                // GPU.11 — write each mip at its within-slot offset.
                // occupancy + color_offsets land in per-mip sub-blocks
                // (mip-0 first, so its data is byte-identical to the
                // pre-mip layout); colours of every mip concatenate
                // into the slot's fixed COLORS_PER_CHUNK_WORDS block in
                // level order, indexed by each chunk's own absolute
                // `color_offsets`.
                let occ_start = slot_idx * occ_words_per_slot;
                let off_start = slot_idx * offsets_words_per_slot;
                let col_start = slot_idx * colors_stride;
                let mut color_cursor = 0usize;
                for (m, mip) in chunk.mips.iter().enumerate() {
                    let occ_dst = occ_start + layout.mip_occ_rel[m] as usize;
                    grid_occupancy[occ_dst..occ_dst + mip.occupancy.len()]
                        .copy_from_slice(&mip.occupancy);
                    // Solid bitmap immediately follows the textured one.
                    let solid_dst = occ_dst + mip.occupancy.len();
                    grid_occupancy[solid_dst..solid_dst + mip.solid_occupancy.len()]
                        .copy_from_slice(&mip.solid_occupancy);
                    let coff_dst = off_start + layout.mip_coff_rel[m] as usize;
                    grid_color_offsets[coff_dst..coff_dst + mip.color_offsets.len()]
                        .copy_from_slice(&mip.color_offsets);

                    let remaining = colors_stride.saturating_sub(color_cursor);
                    let n = mip.colors.len().min(remaining);
                    if n < mip.colors.len() {
                        eprintln!(
                            "roxlap-gpu SceneUpload: scene grid chunk {chunk_idx:?} mip {m} \
                             colours overflow COLORS_PER_CHUNK_WORDS ({colors_stride}); \
                             truncating",
                        );
                    }
                    grid_colors[col_start + color_cursor..col_start + color_cursor + n]
                        .copy_from_slice(&mip.colors[..n]);
                    color_cursor += n;
                }

                if !chunk.mips[0].colors.is_empty() {
                    grid_chunk_occupancy[slot_idx >> 5] |= 1u32 << (slot_idx & 31);
                }
                grid_slot_chunk_idx[slot_idx] = [chunk_idx[0], chunk_idx[1], chunk_idx[2], 0];
            }

            // Slot_chunk_idx storage offset: each entry is 4 u32
            // words (vec3 padded to 16 bytes in std430).
            let slot_chunk_idx_offset = u32::try_from(all_slot_chunk_idx.len()).expect("fits");
            // GPU.13.0 — occupied chunk-AABB for the outer-DDA early-out.
            let (aabb_min, aabb_max) = aabb_of_slots(&grid_slot_chunk_idx);
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
                occ_words_per_slot: layout.occ_words_per_slot,
                offsets_words_per_slot: layout.offsets_words_per_slot,
                mip_count: layout.mip_count,
                _pad1: 0,
                mip_occ_rel: layout.mip_occ_rel,
                mip_coff_rel: layout.mip_coff_rel,
                aabb_min,
                _pad2: 0,
                aabb_max,
                _pad3: 0,
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

        // Split the concatenated occupancy across storage pages so no
        // single binding exceeds the device limit. Page size is a
        // whole number of chunk slots (slot-aligned) so no per-slot
        // refresh write ever straddles two pages.
        // GPU.11 — page alignment is now the whole-ladder per-slot
        // occupancy stride so a slot (all its mips) never straddles a
        // page boundary.
        let slot_align_words = info
            .grids
            .iter()
            .map(|g| u64::from(MipLayout::for_vsid(g.vsid).occ_words_per_slot))
            .max()
            .unwrap_or(1)
            .max(1);
        let (occupancy_pages, occupancy_page_words, occupancy_num_pages) =
            split_occupancy_pages(device, &all_occupancy, slot_align_words);
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
            // GPU.13.0 — COPY_DST so the live chunk-AABB can be patched
            // into a grid's meta on refresh_chunk / evict_chunk.
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            grid_count,
            occupancy_pages,
            occupancy_page_words,
            occupancy_num_pages,
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

        // GPU.11 — the per-slot strides span the full mip ladder; the
        // resident's layout was built from the same `MipLayout`.
        let layout = MipLayout::for_vsid(meta.vsid);
        let occ_words_per_slot = layout.occ_words_per_slot as usize;
        let offsets_words_per_slot = layout.offsets_words_per_slot as usize;
        let colors_stride = COLORS_PER_CHUNK_WORDS as usize;

        assert_eq!(
            chunk.mips.len() as u32,
            layout.mip_count,
            "refresh_chunk: mip count mismatch (chunk {} vs grid {})",
            chunk.mips.len(),
            layout.mip_count,
        );

        // ---- occupancy ----
        // Route each mip's write to its page. Page size is slot-
        // aligned (see `split_occupancy_pages`) so the whole slot's
        // occupancy ladder lands in a single page.
        let slot_occ_base = meta.occupancy_offset as usize + slot_idx * occ_words_per_slot;
        let page_words = self.occupancy_page_words as usize;
        let page = slot_occ_base / page_words;
        let slot_local_word = slot_occ_base % page_words;
        debug_assert!(
            slot_local_word + occ_words_per_slot <= page_words,
            "occupancy slot straddles a page boundary — page size not slot-aligned",
        );
        let off_slot_base = meta.color_offsets_offset as usize + slot_idx * offsets_words_per_slot;
        let col_slot_base = meta.colors_offset as usize + slot_idx * colors_stride;

        let mut outcome = RefreshOutcome::Ok;
        let mut color_cursor = 0usize;
        for (m, mip) in chunk.mips.iter().enumerate() {
            // occupancy (textured) then solid, back-to-back.
            let local = slot_local_word + layout.mip_occ_rel[m] as usize;
            queue.write_buffer(
                &self.occupancy_pages[page],
                (local * 4) as u64,
                bytemuck::cast_slice(&mip.occupancy),
            );
            queue.write_buffer(
                &self.occupancy_pages[page],
                ((local + mip.occupancy.len()) * 4) as u64,
                bytemuck::cast_slice(&mip.solid_occupancy),
            );
            // color_offsets
            let coff = off_slot_base + layout.mip_coff_rel[m] as usize;
            queue.write_buffer(
                &self.all_color_offsets,
                (coff * 4) as u64,
                bytemuck::cast_slice(&mip.color_offsets),
            );
            // colours (concatenated per slot, truncate to stride)
            let remaining = colors_stride.saturating_sub(color_cursor);
            let n = mip.colors.len().min(remaining);
            if n < mip.colors.len() {
                eprintln!(
                    "roxlap-gpu refresh_chunk: scene_idx={scene_idx} chunk_idx={chunk_idx:?} \
                     mip {m} colours overflow stride {colors_stride}; truncating",
                );
                outcome = RefreshOutcome::ColorsTruncated;
            }
            if n > 0 {
                queue.write_buffer(
                    &self.all_colors,
                    ((col_slot_base + color_cursor) * 4) as u64,
                    bytemuck::cast_slice(&mip.colors[..n]),
                );
            }
            color_cursor += n;
        }

        // ---- chunk_occupancy bit ----
        self.set_chunk_occupancy_bit(
            queue,
            scene_idx,
            &meta,
            slot_idx,
            !chunk.mips[0].colors.is_empty(),
        );

        // ---- slot_chunk_idx (identity for the shader) ----
        self.set_slot_chunk_idx(queue, scene_idx, &meta, slot_idx, chunk_idx);

        // ---- GPU.13.0 grid-AABB early-out box ----
        self.sync_aabb(queue, scene_idx);

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
        // GPU.13.0 — eviction may shrink the occupied box; recompute.
        self.sync_aabb(queue, scene_idx);
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

    /// GPU.13.0 — recompute the grid's occupied chunk-AABB from its
    /// `slot_chunk_idx` shadow and, if it changed, patch the grid's
    /// [`GridStaticMeta`] on the GPU. Cheap: scans `total_slots`
    /// entries and writes 144 bytes only when the box actually moves
    /// (steady-state re-bakes leave it unchanged → no GPU write).
    /// Called after every install/eviction so streaming grids keep a
    /// tight, always-conservative early-out box.
    fn sync_aabb(&mut self, queue: &wgpu::Queue, scene_idx: usize) {
        let (aabb_min, aabb_max) = aabb_of_slots(&self.slot_chunk_idx_shadow[scene_idx]);
        let meta = &mut self.static_meta[scene_idx];
        if meta.aabb_min == aabb_min && meta.aabb_max == aabb_max {
            return;
        }
        meta.aabb_min = aabb_min;
        meta.aabb_max = aabb_max;
        let off = (scene_idx * std::mem::size_of::<GridStaticMeta>()) as u64;
        queue.write_buffer(&self.grid_static_meta, off, bytemuck::bytes_of(meta));
    }
}

/// GPU.13.0 — inclusive chunk-AABB over a grid's `slot_chunk_idx`
/// shadow, skipping the [`SLOT_EMPTY_SENTINEL`] entries. Returns the
/// inverted sentinel box (`min = i32::MAX`, `max = i32::MIN`) when no
/// slot is occupied, which makes the shader's `aabb_passed` early-out
/// fire for every ray (an empty grid renders nothing).
fn aabb_of_slots(slots: &[[i32; 4]]) -> ([i32; 3], [i32; 3]) {
    let mut min = [i32::MAX; 3];
    let mut max = [i32::MIN; 3];
    for e in slots {
        if e[0] == SLOT_EMPTY_SENTINEL[0]
            && e[1] == SLOT_EMPTY_SENTINEL[1]
            && e[2] == SLOT_EMPTY_SENTINEL[2]
        {
            continue;
        }
        for k in 0..3 {
            if e[k] < min[k] {
                min[k] = e[k];
            }
            if e[k] > max[k] {
                max[k] = e[k];
            }
        }
    }
    (min, max)
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

/// Split the concatenated occupancy words into up to
/// [`MAX_OCC_PAGES`] storage buffers, each no larger than the
/// device's `max_storage_buffer_binding_size`, then pad the page
/// list with 1-word dummy buffers so the returned vec is always
/// exactly `MAX_OCC_PAGES` long (one buffer per bind-group entry).
///
/// `slot_align_words` is the per-slot occupancy stride: page size is
/// rounded down to a multiple of it so no chunk slot — and therefore
/// no per-slot `refresh_chunk` write — straddles a page boundary.
/// Returns `(pages, page_words, num_pages)`.
fn split_occupancy_pages(
    device: &wgpu::Device,
    words: &[u32],
    slot_align_words: u64,
) -> (Vec<wgpu::Buffer>, u32, u32) {
    let total_words = words.len() as u64;
    let limit_words = u64::from(device.limits().max_storage_buffer_binding_size) / 4;
    // Largest slot-aligned page that fits one binding (≥ 1 slot).
    let page_slots = (limit_words / slot_align_words).max(1);
    let mut page_words = page_slots.saturating_mul(slot_align_words);
    // A tiny scene (or the empty-scene 1-word pad) isn't slot-aligned;
    // cap the page at the data length so we don't allocate emptiness.
    page_words = page_words.min(total_words.max(1));
    let num_pages = total_words.div_ceil(page_words);
    assert!(
        num_pages as usize <= MAX_OCC_PAGES,
        "occupancy needs {num_pages} pages (>{MAX_OCC_PAGES}) at this device's \
         {limit_words}-word binding limit; shrink the streaming pool or raise MAX_OCC_PAGES",
    );

    let mut pages: Vec<wgpu::Buffer> = Vec::with_capacity(MAX_OCC_PAGES);
    let page_words_usize = page_words as usize;
    for p in 0..num_pages as usize {
        let start = p * page_words_usize;
        let end = ((p + 1) * page_words_usize).min(words.len());
        pages.push(create_storage(
            device,
            &format!("roxlap-gpu scene.occupancy.page{p}"),
            &words[start..end],
        ));
    }
    // Dummy 1-word buffers for the unused bindings.
    while pages.len() < MAX_OCC_PAGES {
        pages.push(create_storage(
            device,
            "roxlap-gpu scene.occupancy.page_dummy",
            &[0u32],
        ));
    }
    (
        pages,
        u32::try_from(page_words).expect("page_words fits u32"),
        num_pages as u32,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_static_meta_matches_wgsl_std430_size() {
        // scene_dda.wgsl's GridStaticMeta is read as
        // array<GridStaticMeta>; the std430 array stride must equal
        // the Rust size_of or wgpu rejects the binding.
        // Concretely: 8 u32 (32) + vec3+pad (16) + 4 u32 (16) +
        // 2*[u32;6] (48) = 112, then GPU.13.0 adds two vec3<i32>+pad
        // (aabb_min, aabb_max) = 32 → 144 bytes.
        assert_eq!(std::mem::size_of::<GridStaticMeta>(), 144);
        assert_eq!(std::mem::align_of::<GridStaticMeta>(), 4);
    }

    #[test]
    fn mip_layout_offsets_accumulate() {
        // vsid=128 → 6 mips. Relative offsets are cumulative; mip-0
        // sits at 0 so mip-0 reads are byte-identical to pre-mip.
        let l = MipLayout::for_vsid(128);
        assert_eq!(l.mip_count, 6);
        assert_eq!(l.mip_occ_rel[0], 0);
        assert_eq!(l.mip_coff_rel[0], 0);

        // Recompute the strides independently and compare. Each mip
        // stores TWO occupancy bitmaps (textured + solid) back-to-back.
        let mut occ = 0u32;
        let mut coff = 0u32;
        for m in 0..6u32 {
            assert_eq!(l.mip_occ_rel[m as usize], occ, "occ rel mip {m}");
            assert_eq!(l.mip_coff_rel[m as usize], coff, "coff rel mip {m}");
            let v = 128u32 >> m;
            occ += 2 * v * v * occ_words_per_column_for_mip(m);
            coff += v * v + 1;
        }
        assert_eq!(l.occ_words_per_slot, occ);
        assert_eq!(l.offsets_words_per_slot, coff);

        // mip-0 occupancy stride is 2 × the historical vsid²·8 (tex +
        // solid bitmaps).
        assert_eq!(l.mip_occ_rel[1], 2 * 128 * 128 * 8);
        // The whole ladder is only ~1/7 larger than mip-0 alone
        // (geometric 1 + 1/8 + 1/64 + …) — here on the doubled base.
        assert!(l.occ_words_per_slot < 2 * 128 * 128 * 8 * 5 / 4);
    }
}
