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
    /// Number of mip levels stored per slot = [`gpu_mip_count`]`(vsid)`,
    /// always `1..=`[`MAX_GPU_MIPS`].
    pub mip_count: u32,
    /// Occupancy u32 words per chunk slot, summed over all mips — each
    /// mip stores its textured **and** solid bitmap back-to-back, so
    /// this is `Σ 2·(vsid>>m)²·occ_words_per_column_for_mip(m)`.
    pub occ_words_per_slot: u32,
    /// `color_offsets` u32 words per chunk slot, summed over all mips
    /// (`Σ (vsid>>m)² + 1` — each mip keeps its own `cols + 1` prefix
    /// table).
    pub offsets_words_per_slot: u32,
    /// Within-slot u32 offset where mip `m`'s occupancy starts.
    pub mip_occ_rel: [u32; MAX_GPU_MIPS],
    /// Within-slot u32 offset where mip `m`'s color_offsets start.
    pub mip_coff_rel: [u32; MAX_GPU_MIPS],
}

impl MipLayout {
    /// Compute the per-slot layout for a grid whose chunks are
    /// `vsid × vsid × CHUNK_Z` voxels. Deterministic in `vsid`, so the
    /// upload, [`GpuSceneResident::refresh_chunk`], and the shader all
    /// derive identical strides independently.
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
    /// One [`GridUpload`] per scene grid, in the order the shader
    /// iterates them; index = the `scene_idx` handed to
    /// [`GpuSceneResident::refresh_chunk`] / `evict_chunk` and the
    /// per-grid camera slot.
    pub grids: Vec<GridUpload>,
}

impl SceneUpload {
    /// Number of grids, saturated into `u32` (the type the shader's
    /// `grid_count` uniform uses).
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
    /// `all_color_offsets` (binding 2) u32-word offset where this
    /// grid's per-slot colour-offset tables start; slot `meta_id`'s
    /// window is `offsets_words_per_slot` words from
    /// `color_offsets_offset + meta_id * offsets_words_per_slot`.
    pub color_offsets_offset: u32,
    /// `all_colors` (binding 3) u32-word offset where this grid's
    /// packed voxel colours start (per-slot blocks of the grid's
    /// colour stride).
    pub colors_offset: u32,
    /// `all_chunk_colors_base` (binding 4) u32-word offset of this
    /// grid's per-slot table mapping `meta_id` → the slot's colour
    /// base (in words, relative to `colors_offset`).
    pub chunk_colors_base_offset: u32,
    /// `all_chunk_occupancy` (binding 5) u32-word offset of this
    /// grid's chunk-occupancy bitmap: bit `slot_idx & 31` of word
    /// `slot_idx >> 5` is set iff the slot holds a non-empty chunk —
    /// the outer DDA's whole-chunk skip test.
    pub chunk_occupancy_offset: u32,
    /// New in GPU.7: u32-word offset where this grid's
    /// `slot_chunk_idx` array starts (one `vec3<i32>` per slot,
    /// i.e. 3 u32 words each, plus 1 padding word for std430).
    pub slot_chunk_idx_offset: u32,
    /// Chunk XY extent in voxels (typically 128); the chunk's Z extent
    /// is the fixed `CHUNK_Z = 256`.
    pub vsid: u32,
    /// Slot count in the modular pool
    /// (`pool_dims.x · pool_dims.y · pool_dims.z`).
    pub total_slots: u32,
    /// GPU.7 slot-pool dimensions (each a power of 2). Chunk
    /// `(chx, chy, chz)` lives in slot
    /// `(chx & (pool_dims.x - 1), chy & …, chz & …)`.
    pub pool_dims: [u32; 3],
    /// std430 padding; always 0.
    pub _pad0: u32,
    /// GPU.11 — per-slot occupancy stride (sum over all mips).
    /// `meta_id`'s occupancy slab starts at
    /// `occupancy_offset + meta_id * occ_words_per_slot`.
    pub occ_words_per_slot: u32,
    /// GPU.11 — per-slot color_offsets stride (sum over all mips).
    pub offsets_words_per_slot: u32,
    /// GPU.11 — number of mip levels stored per slot.
    pub mip_count: u32,
    /// std430 padding; always 0.
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
    /// std430 `vec3<i32>` padding; always 0.
    pub _pad2: i32,
    /// Inclusive upper corner of the occupied chunk-AABB (chunk-index
    /// space); `i32::MIN` sentinel when the grid holds no chunks. See
    /// [`Self::aabb_min`].
    pub aabb_max: [i32; 3],
    /// std430 `vec3<i32>` padding; always 0.
    pub _pad3: i32,
    /// GPU.13.1 — `all_chunk_occupancy` u32-word offsets of this
    /// grid's chunk-occupancy pyramid levels 1..=4 (entry `l - 1` =
    /// level `l`; entries past [`Self::chunk_occ_levels`] are 0).
    /// Level `l` has `max(pool_dims >> l, 1)` cells per axis; a set
    /// bit means "some resident non-empty chunk maps into this
    /// slot-block" — the outer DDA's read-free empty-block skip.
    pub chunk_occ_mip_off: [u32; 4],
    /// GPU.13.1 — pyramid levels stored above L0 (0 = no pyramid,
    /// the single-chunk-pool degenerate case).
    pub chunk_occ_levels: u32,
    /// CA perf — grid-local SOLID voxel z-extent (inclusive) over all
    /// resident chunks: the marchers clamp their entry fast-forward
    /// box and ray caps to it, so a grid whose content occupies a
    /// slice of its chunk stack (a ship spanning a chz boundary sits
    /// in a 512-voxel-tall chunk AABB but is ~70 voxels of hull) is
    /// entered AT the content and abandoned right past it. Inverted
    /// sentinel (`lo = i32::MAX`, `hi = i32::MIN`) when the grid holds
    /// no solid voxels. Maintained live alongside the chunk AABB
    /// ([`GpuSceneResident::refresh_chunk`] / `evict_chunk`).
    pub vox_z_lo: i32,
    /// See [`Self::vox_z_lo`]; inclusive.
    pub vox_z_hi: i32,
    /// std430 padding; always 0.
    pub _pad4: u32,
}

/// Sentinel chunk_idx written into empty slot_chunk_idx entries.
/// Real chunk indices never use `i32::MIN`, so the shader can
/// distinguish empty slots from collisions via a single equality
/// check.
pub const SLOT_EMPTY_SENTINEL: [i32; 3] = [i32::MIN, i32::MIN, i32::MIN];

/// GPU-resident storage for an entire scene's grids.
pub struct GpuSceneResident {
    /// Number of grids uploaded (= `SceneUpload::grid_count()`); the
    /// shader's grid loop bound and the length of [`Self::static_meta`].
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
    /// Binding 2 — concatenated per-slot `color_offsets` prefix tables
    /// (all grids, all mips). Grid `g`'s region starts at
    /// `static_meta[g].color_offsets_offset` words.
    pub all_color_offsets: wgpu::Buffer,
    /// Binding 3 — concatenated packed voxel colours (all grids).
    /// Each word is the voxlap wire format the shader unpacks: blue in
    /// bits 0-7, green 8-15, red 16-23, and a **brightness** byte (not
    /// alpha) in bits 24-31 with `0x80` = neutral.
    pub all_colors: wgpu::Buffer,
    /// Binding 4 — per-slot colour base table: word `meta_id` of grid
    /// `g`'s region holds the slot's colour start (in words, relative
    /// to `static_meta[g].colors_offset`) = `meta_id × colour stride`.
    pub all_chunk_colors_base: wgpu::Buffer,
    /// Binding 5 — per-grid chunk-occupancy bitmaps (one bit per pool
    /// slot: set iff the slot holds a non-empty chunk). The outer DDA
    /// tests it to skip whole 128×128×256 chunks per step.
    pub all_chunk_occupancy: wgpu::Buffer,
    /// GPU.7 — per-slot chunk_idx for identity verification in the
    /// shader. Stored as `vec3<i32>` with std430 16-byte stride
    /// (each entry is `[i32; 4]` on the host: x, y, z, _pad).
    pub all_slot_chunk_idx: wgpu::Buffer,
    /// Binding 6 — the [`GridStaticMeta`] array, one element per grid.
    /// Mostly upload-time constant; the live chunk-AABB fields are
    /// patched in place on `refresh_chunk` / `evict_chunk`.
    pub grid_static_meta: wgpu::Buffer,
    /// Total bytes allocated across all the resident storage buffers,
    /// for VRAM accounting (see [`Self::resident_bytes`]).
    pub total_bytes: u64,
    /// Cached static metadata for the host's frame-loop work.
    pub static_meta: Vec<GridStaticMeta>,
    /// CPU shadow of the per-grid chunk-occupancy bitmap. Each entry
    /// is the u32 word at `chunk_occupancy_offset + (mi >> 5)`.
    /// `refresh_chunk` / `evict_chunk` flip the right bit + write
    /// the affected word back to the GPU.
    pub(crate) chunk_occupancy_shadow: Vec<Vec<u32>>,
    /// GPU.13.1 — CPU mirror of each grid's chunk-occupancy pyramid
    /// (outer Vec: grid; middle: level 1.. — level 0 IS
    /// [`Self::chunk_occupancy_shadow`]). Ancestor re-OR on
    /// refresh/evict reads children from here instead of the GPU.
    pub(crate) chunk_occ_pyramid_shadow: Vec<Vec<Vec<u32>>>,
    /// CPU shadow of `slot_chunk_idx`. Indexed `[scene_idx][slot]`
    /// → `[i32; 4]` (vec3 + pad). Host uses this to detect "slot is
    /// holding a different chunk than expected" + as the eviction
    /// origin.
    pub(crate) slot_chunk_idx_shadow: Vec<Vec<[i32; 4]>>,
    /// Per-grid colour stride in u32 words (the adaptive
    /// [`COLORS_PER_CHUNK_WORDS`]-or-larger value chosen at upload to
    /// fit the grid's densest chunk). `refresh_chunk` reads it so a
    /// streamed re-upload addresses colours with the same stride the
    /// initial upload used.
    pub(crate) colors_stride_shadow: Vec<u32>,
    /// PF.12.c — CPU mirror of each installed slot's per-mip
    /// `color_offsets` tables (`offsets_words_per_slot` words, the exact
    /// content of the GPU window). [`Self::refresh_chunk_partial`] reads
    /// it to (a) place a dirty column's colours at the resident offset
    /// and (b) verify the column's colour COUNT is unchanged — a count
    /// change reflows the packed colour block and forces the full-path
    /// fallback. ~87 KB per 128² chunk; dropped on evict.
    pub(crate) color_offsets_shadow: Vec<std::collections::HashMap<usize, Vec<u32>>>,
    /// CA perf — CPU mirror of each slot's IN-CHUNK solid z-extent
    /// (`[lo, hi]` inclusive; `[i32::MAX, i32::MIN]` for empty slots).
    /// Combined with the slot's chunk-z on every refresh/evict to
    /// recompute the grid's [`GridStaticMeta::vox_z_lo`]/`vox_z_hi`
    /// marcher clamp (same maintenance path as the chunk AABB).
    pub(crate) slot_z_ext_shadow: Vec<Vec<[i32; 2]>>,
}

impl GpuSceneResident {
    /// Pack + upload `info`. Each grid is uploaded as a contiguous
    /// slab inside the shared storage buffers; per-grid offsets
    /// live in `grid_static_meta`. The grid count is bounded only by
    /// the device's storage-buffer limits (per-grid cameras + metadata
    /// are runtime-sized storage arrays, not a fixed shader array).
    pub fn upload(device: &wgpu::Device, info: &SceneUpload) -> Self {
        let grid_count = info.grid_count();

        let mut all_occupancy: Vec<u32> = Vec::new();
        let mut all_color_offsets: Vec<u32> = Vec::new();
        let mut all_colors: Vec<u32> = Vec::new();
        let mut all_chunk_colors_base: Vec<u32> = Vec::new();
        let mut all_chunk_occupancy: Vec<u32> = Vec::new();
        let mut all_slot_chunk_idx: Vec<i32> = Vec::new();
        let mut static_meta: Vec<GridStaticMeta> = Vec::with_capacity(info.grids.len());
        let mut chunk_occupancy_shadow: Vec<Vec<u32>> = Vec::with_capacity(info.grids.len());
        let mut chunk_occ_pyramid_shadow: Vec<Vec<Vec<u32>>> = Vec::with_capacity(info.grids.len());
        let mut slot_chunk_idx_shadow: Vec<Vec<[i32; 4]>> = Vec::with_capacity(info.grids.len());
        let mut slot_z_ext_shadow: Vec<Vec<[i32; 2]>> = Vec::with_capacity(info.grids.len());
        let mut color_offsets_shadow: Vec<std::collections::HashMap<usize, Vec<u32>>> =
            Vec::with_capacity(info.grids.len());
        // Per-grid colour stride (words/slot) — adaptive to the grid's
        // densest chunk (see the in-loop derivation). `refresh_chunk`
        // reads it back so streamed re-uploads use the same stride.
        let mut grid_colors_strides: Vec<u32> = Vec::with_capacity(info.grids.len());

        for grid in &info.grids {
            let vsid = grid.vsid;
            // GPU.11 — per-slot strides span the whole mip ladder.
            let layout = MipLayout::for_vsid(vsid);
            let occ_words_per_slot = layout.occ_words_per_slot as usize;
            let offsets_words_per_slot = layout.offsets_words_per_slot as usize;
            // Per-slot colour stride. The fixed-stride layout gives every
            // slot — present or not — the same capacity, so streaming /
            // re-bake can write a fresh chunk's colours into any slot.
            // [`COLORS_PER_CHUNK_WORDS`] is sized for sparse terrain
            // chunks (~36 k colours); a *fully dense* chunk (the cave
            // demo's single 128×128×256 chunk carries ~207 k colours
            // across its mip ladder) needs more, or its colours truncate
            // and the chunk's high-`y` columns render black. Grow the
            // stride to the grid's densest chunk, floored at the default
            // so denser chunks that stream in later still fit the common
            // case. Per-grid: a sparse grid keeps the small stride; only
            // a grid that actually holds dense chunks pays for the
            // bigger one.
            let max_chunk_colors = grid
                .chunks
                .iter()
                .map(|(_, c)| c.mips.iter().map(|m| m.colors.len()).sum::<usize>())
                .max()
                .unwrap_or(0);
            let colors_stride = (COLORS_PER_CHUNK_WORDS as usize).max(max_chunk_colors);
            grid_colors_strides.push(colors_stride as u32);

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
            let mut grid_offsets_shadow: std::collections::HashMap<usize, Vec<u32>> =
                std::collections::HashMap::new();
            let mut grid_slot_chunk_idx: Vec<[i32; 4]> = Vec::with_capacity(total_slots);
            for _ in 0..total_slots {
                grid_slot_chunk_idx.push([
                    SLOT_EMPTY_SENTINEL[0],
                    SLOT_EMPTY_SENTINEL[1],
                    SLOT_EMPTY_SENTINEL[2],
                    0,
                ]);
            }
            // CA perf — per-slot in-chunk solid z-extent (marcher clamp).
            let mut grid_slot_z_ext: Vec<[i32; 2]> = vec![[i32::MAX, i32::MIN]; total_slots];

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
                // CA perf — the slot's in-chunk solid z-extent.
                grid_slot_z_ext[slot_idx] = match crate::decompress::solid_z_extent(&chunk.mips[0])
                {
                    Some((lo, hi)) => [lo as i32, hi as i32],
                    None => [i32::MAX, i32::MIN],
                };
                // PF.12.c — mirror the slot's color_offsets window.
                grid_offsets_shadow.insert(
                    slot_idx,
                    grid_color_offsets[off_start..off_start + offsets_words_per_slot].to_vec(),
                );
            }

            // Slot_chunk_idx storage offset: each entry is 4 u32
            // words (vec3 padded to 16 bytes in std430).
            let slot_chunk_idx_offset = u32::try_from(all_slot_chunk_idx.len()).expect("fits");
            // GPU.13.0 — occupied chunk-AABB for the outer-DDA early-out.
            let (aabb_min, aabb_max) = aabb_of_slots(&grid_slot_chunk_idx);
            // CA perf — grid-local solid voxel z-extent (marcher clamp).
            let (vox_z_lo, vox_z_hi) = vox_z_of_slots(&grid_slot_chunk_idx, &grid_slot_z_ext);
            // GPU.13.1 — chunk-occupancy pyramid: levels 1.. appended
            // right after this grid's L0 words, offsets recorded per
            // level (word offsets are into `all_chunk_occupancy`).
            let chunk_occupancy_offset = u32::try_from(all_chunk_occupancy.len()).expect("fits");
            let pyramid = build_occ_pyramid(grid.pool_dims, &grid_chunk_occupancy);
            let mut chunk_occ_mip_off = [0u32; 4];
            {
                let mut cursor = all_chunk_occupancy.len() + grid_chunk_occupancy.len();
                for (i, level) in pyramid.iter().enumerate() {
                    chunk_occ_mip_off[i] = u32::try_from(cursor).expect("fits");
                    cursor += level.len();
                }
            }
            let meta = GridStaticMeta {
                occupancy_offset: u32::try_from(all_occupancy.len()).expect("fits"),
                color_offsets_offset: u32::try_from(all_color_offsets.len()).expect("fits"),
                colors_offset: u32::try_from(all_colors.len()).expect("fits"),
                chunk_colors_base_offset: u32::try_from(all_chunk_colors_base.len()).expect("fits"),
                chunk_occupancy_offset,
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
                chunk_occ_mip_off,
                chunk_occ_levels: u32::try_from(pyramid.len()).expect("small"),
                vox_z_lo,
                vox_z_hi,
                _pad4: 0,
            };

            chunk_occupancy_shadow.push(grid_chunk_occupancy.clone());
            chunk_occ_pyramid_shadow.push(pyramid.clone());
            slot_chunk_idx_shadow.push(grid_slot_chunk_idx.clone());
            slot_z_ext_shadow.push(grid_slot_z_ext);
            color_offsets_shadow.push(grid_offsets_shadow);

            all_occupancy.extend_from_slice(&grid_occupancy);
            all_color_offsets.extend_from_slice(&grid_color_offsets);
            all_colors.extend_from_slice(&grid_colors);
            all_chunk_colors_base.extend_from_slice(&grid_chunk_colors_base);
            all_chunk_occupancy.extend_from_slice(&grid_chunk_occupancy);
            for level in &pyramid {
                all_chunk_occupancy.extend_from_slice(level);
            }
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
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let grid_static_meta = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("roxlap-gpu scene.grid_static_meta"),
            contents: bytemuck::cast_slice(&static_meta),
            // GPU.13.0 — COPY_DST so the live chunk-AABB can be patched
            // into a grid's meta on refresh_chunk / evict_chunk.
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
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
            chunk_occ_pyramid_shadow,
            slot_chunk_idx_shadow,
            color_offsets_shadow,
            colors_stride_shadow: grid_colors_strides,
            slot_z_ext_shadow,
        }
    }

    /// GPU memory held by the scene's storage buffers, in bytes —
    /// [`Self::total_bytes`] as computed at upload time (in-place
    /// refreshes never change it).
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
        // Same adaptive stride the initial upload chose for this grid.
        let colors_stride = self
            .colors_stride_shadow
            .get(scene_idx)
            .map_or(COLORS_PER_CHUNK_WORDS as usize, |&s| s as usize);

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

        // ---- PF.12.c — mirror the slot's color_offsets window ----
        // (`refresh_chunk_partial` verifies counts + places colours
        // against it). Rebuilt exactly as the GPU windows were written.
        let mut window = vec![0u32; offsets_words_per_slot];
        for (m, mip) in chunk.mips.iter().enumerate() {
            let coff = layout.mip_coff_rel[m] as usize;
            window[coff..coff + mip.color_offsets.len()].copy_from_slice(&mip.color_offsets);
        }
        self.color_offsets_shadow[scene_idx].insert(slot_idx, window);

        // CA perf — the slot's in-chunk solid z-extent (the grid clamp
        // recomputes inside sync_aabb below).
        self.slot_z_ext_shadow[scene_idx][slot_idx] =
            match crate::decompress::solid_z_extent(&chunk.mips[0]) {
                Some((lo, hi)) => [lo as i32, hi as i32],
                None => [i32::MAX, i32::MIN],
            };

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
    /// PF.12.c — partial refresh: re-derive + re-upload ONLY the columns
    /// inside the inclusive chunk-local mip-0 column rect `[x0..=x1] ×
    /// [y0..=y1]` (pre-padded by the caller with the edit's ±1 adjacency
    /// reach), for every mip. Requires the slot to already hold
    /// `chunk_idx` with a mirrored offsets table, and every dirty
    /// column's colour COUNT to be unchanged (a count change reflows the
    /// packed colour block). Returns `false` — with **nothing written**
    /// — when any precondition fails; the caller falls back to the full
    /// [`Self::refresh_chunk`] path.
    ///
    /// The count-stable case is the streaming bake tracker's per-frame
    /// path (brightness-byte rewrites) and recolour edits: those now
    /// upload a few KB instead of decompressing + rewriting the whole
    /// ~1–2 MB chunk ladder.
    #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
    pub fn refresh_chunk_partial(
        &mut self,
        queue: &wgpu::Queue,
        scene_idx: usize,
        chunk_idx: [i32; 3],
        vxl: &roxlap_formats::vxl::Vxl,
        x0: i32,
        y0: i32,
        x1: i32,
        y1: i32,
    ) -> bool {
        let Some(meta) = self.static_meta.get(scene_idx).copied() else {
            return false;
        };
        let layout = MipLayout::for_vsid(meta.vsid);
        if vxl.mip_count() < layout.mip_count {
            return false;
        }
        let slot_idx = modular_slot_idx(chunk_idx, meta.pool_dims);
        // The slot must currently hold THIS chunk (modular pools reuse
        // slots; a partial write over another chunk's data = garbage).
        let held = self.slot_chunk_idx_shadow[scene_idx][slot_idx];
        if held[0] != chunk_idx[0] || held[1] != chunk_idx[1] || held[2] != chunk_idx[2] {
            return false;
        }
        let Some(offs_shadow) = self.color_offsets_shadow[scene_idx].get(&slot_idx) else {
            return false;
        };
        let colors_stride = self
            .colors_stride_shadow
            .get(scene_idx)
            .map_or(COLORS_PER_CHUNK_WORDS as usize, |&s| s as usize);

        // Phase 1 — recompute every dirty column per mip into row-run
        // buffers (rows are contiguous in both the occupancy layout and
        // the packed colour block), verifying colour counts. NOTHING is
        // written until the whole extent verifies.
        struct RowRun {
            /// Textured-occupancy word offset within the slot.
            occ_word: usize,
            /// Solid block sits `block_words` after the textured one.
            block_words: usize,
            occ: Vec<u32>,
            solid: Vec<u32>,
            /// Colour word offset within the slot's colour block.
            color_word: usize,
            colors: Vec<u32>,
        }
        let mut runs: Vec<RowRun> = Vec::new();
        // CA perf — mip-0 solid z-extent of the re-derived columns, to
        // WIDEN the slot's clamp below (widen-only is safe: an edit
        // that removed the extremes leaves the clamp conservatively
        // large; the next full refresh re-tightens it).
        let mut dirty_z = [i32::MAX, i32::MIN];
        for m in 0..layout.mip_count {
            let vsid_m = (meta.vsid >> m).max(1) as i32;
            let cz_m = crate::decompress::CHUNK_Z >> m;
            let wpc = occ_words_per_column_for_mip(m) as usize;
            let block_words = (vsid_m as usize) * (vsid_m as usize) * wpc;
            let rx0 = (x0 >> m).clamp(0, vsid_m - 1);
            let ry0 = (y0 >> m).clamp(0, vsid_m - 1);
            let rx1 = (x1 >> m).clamp(0, vsid_m - 1);
            let ry1 = (y1 >> m).clamp(0, vsid_m - 1);
            let coff_base = layout.mip_coff_rel[m as usize] as usize;
            for y in ry0..=ry1 {
                let row_col0 = (y * vsid_m + rx0) as usize;
                let n_cols = (rx1 - rx0 + 1) as usize;
                let mut occ = vec![0u32; n_cols * wpc];
                let mut solid = vec![0u32; n_cols * wpc];
                let mut colors: Vec<u32> = Vec::new();
                for i in 0..n_cols {
                    let col_idx = row_col0 + i;
                    let slab = vxl.column_data_for_mip(m, col_idx);
                    let before = colors.len();
                    // vsid=1 / (0,0) → the column scratch windows index
                    // from word 0 of the per-column slices.
                    crate::decompress::decompress_column(
                        slab,
                        0,
                        0,
                        1,
                        cz_m,
                        wpc as u32,
                        &mut occ[i * wpc..(i + 1) * wpc],
                        &mut solid[i * wpc..(i + 1) * wpc],
                        &mut colors,
                    );
                    // Count stability vs the mirrored offsets table.
                    let old_count = offs_shadow[coff_base + col_idx + 1]
                        .saturating_sub(offs_shadow[coff_base + col_idx])
                        as usize;
                    if colors.len() - before != old_count {
                        return false; // reflow → full path
                    }
                    // CA perf — fold this column's mip-0 solid extent.
                    if m == 0 {
                        for (k, &w) in solid[i * wpc..(i + 1) * wpc].iter().enumerate() {
                            if w == 0 {
                                continue;
                            }
                            let zb = (k as i32) * 32;
                            dirty_z[0] = dirty_z[0].min(zb + w.trailing_zeros() as i32);
                            dirty_z[1] = dirty_z[1].max(zb + 31 - w.leading_zeros() as i32);
                        }
                    }
                }
                let color_word = offs_shadow[coff_base + row_col0] as usize;
                if color_word + colors.len() > colors_stride {
                    return false; // stride overflow → full path handles
                }
                runs.push(RowRun {
                    occ_word: layout.mip_occ_rel[m as usize] as usize + row_col0 * wpc,
                    block_words,
                    occ,
                    solid,
                    color_word,
                    colors,
                });
            }
        }

        // Phase 2 — verified: write the row runs.
        let occ_words_per_slot = layout.occ_words_per_slot as usize;
        let slot_occ_base = meta.occupancy_offset as usize + slot_idx * occ_words_per_slot;
        let page_words = self.occupancy_page_words as usize;
        let page = slot_occ_base / page_words;
        let slot_local_word = slot_occ_base % page_words;
        let col_slot_base = meta.colors_offset as usize + slot_idx * colors_stride;
        for run in &runs {
            let tex = slot_local_word + run.occ_word;
            queue.write_buffer(
                &self.occupancy_pages[page],
                (tex * 4) as u64,
                bytemuck::cast_slice(&run.occ),
            );
            queue.write_buffer(
                &self.occupancy_pages[page],
                ((tex + run.block_words) * 4) as u64,
                bytemuck::cast_slice(&run.solid),
            );
            if !run.colors.is_empty() {
                queue.write_buffer(
                    &self.all_colors,
                    ((col_slot_base + run.color_word) * 4) as u64,
                    bytemuck::cast_slice(&run.colors),
                );
            }
        }
        // Counts unchanged ⇒ offsets, chunk-occupancy bit, AABB and the
        // mirrors all stay valid untouched.
        // CA perf — widen the slot's solid z-extent by the re-derived
        // columns (never narrows; see `dirty_z`'s comment).
        if dirty_z[0] <= dirty_z[1] {
            let e = &mut self.slot_z_ext_shadow[scene_idx][slot_idx];
            let widened = dirty_z[0] < e[0] || dirty_z[1] > e[1];
            e[0] = e[0].min(dirty_z[0]);
            e[1] = e[1].max(dirty_z[1]);
            if widened {
                self.sync_aabb(queue, scene_idx);
            }
        }
        true
    }

    /// Evict `chunk_idx` from grid `scene_idx`: clear the slot's
    /// chunk-occupancy bit, stamp [`SLOT_EMPTY_SENTINEL`] into its
    /// `slot_chunk_idx` entry, and shrink the grid's chunk-AABB if the
    /// box tightened. The bulk voxel data is left in place — the
    /// cleared occupancy bit + sentinel already make the shader treat
    /// the slot as empty. A no-op (returning `true`) when the slot
    /// meanwhile holds a *different* chunk, so a stale evict can never
    /// wipe a newer occupant. Returns `false` only for an out-of-range
    /// `scene_idx`.
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
        // PF.12.c — drop the evicted slot's offsets mirror.
        self.color_offsets_shadow[scene_idx].remove(&slot_idx);
        // CA perf — the slot no longer contributes to the z-extent.
        self.slot_z_ext_shadow[scene_idx][slot_idx] = [i32::MAX, i32::MIN];
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

        // GPU.13.1 — re-OR the pyramid ancestors of the touched slot.
        // Setting a bit can only turn ancestors ON; clearing one can
        // only turn them OFF — either way, an unchanged ancestor means
        // every level above it is unchanged too, so stop early.
        let pool = meta.pool_dims;
        let slot = [
            (slot_idx as u32) % pool[0],
            ((slot_idx as u32) / pool[0]) % pool[1],
            (slot_idx as u32) / (pool[0] * pool[1]),
        ];
        for l in 1..=meta.chunk_occ_levels {
            let cell = [slot[0] >> l, slot[1] >> l, slot[2] >> l];
            let bit = {
                let child: &[u32] = if l == 1 {
                    &self.chunk_occupancy_shadow[scene_idx]
                } else {
                    &self.chunk_occ_pyramid_shadow[scene_idx][(l - 2) as usize]
                };
                occ_cell_from_children(pool, l, cell, child)
            };
            let d = occ_level_dims(pool, l);
            let idx = (cell[0] + cell[1] * d[0] + cell[2] * d[0] * d[1]) as usize;
            let word = &mut self.chunk_occ_pyramid_shadow[scene_idx][(l - 1) as usize][idx >> 5];
            let was = (*word >> (idx & 31)) & 1 == 1;
            if bit == was {
                break;
            }
            if bit {
                *word |= 1u32 << (idx & 31);
            } else {
                *word &= !(1u32 << (idx & 31));
            }
            let global = meta.chunk_occ_mip_off[(l - 1) as usize] as usize + (idx >> 5);
            let word_copy = *word;
            queue.write_buffer(
                &self.all_chunk_occupancy,
                (global * 4) as u64,
                bytemuck::bytes_of(&word_copy),
            );
        }
    }

    /// Read-only view of the chunk-occupancy pyramid shadows (per
    /// grid, per level above L0) — the CPU mirror the incremental
    /// maintenance updates; exposed for integration tests to assert
    /// the re-OR bookkeeping (rendering itself only reads the GPU
    /// copy).
    #[must_use]
    pub fn chunk_occ_pyramid_shadow(&self) -> &[Vec<Vec<u32>>] {
        &self.chunk_occ_pyramid_shadow
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
        // CA perf — the voxel z-extent clamp travels with the AABB.
        let (vox_z_lo, vox_z_hi) = vox_z_of_slots(
            &self.slot_chunk_idx_shadow[scene_idx],
            &self.slot_z_ext_shadow[scene_idx],
        );
        let meta = &mut self.static_meta[scene_idx];
        if meta.aabb_min == aabb_min
            && meta.aabb_max == aabb_max
            && meta.vox_z_lo == vox_z_lo
            && meta.vox_z_hi == vox_z_hi
        {
            return;
        }
        meta.aabb_min = aabb_min;
        meta.aabb_max = aabb_max;
        meta.vox_z_lo = vox_z_lo;
        meta.vox_z_hi = vox_z_hi;
        let off = (scene_idx * std::mem::size_of::<GridStaticMeta>()) as u64;
        queue.write_buffer(&self.grid_static_meta, off, bytemuck::bytes_of(meta));
    }
}

/// CA perf — inclusive grid-local SOLID voxel z-extent over a grid's
/// occupied slots: each slot contributes `chunk_z · CHUNK_Z + its
/// in-chunk extent`. Inverted sentinel when nothing is solid — the
/// marcher's entry slab test then rejects every ray, matching the
/// empty chunk-AABB behaviour.
fn vox_z_of_slots(slots: &[[i32; 4]], exts: &[[i32; 2]]) -> (i32, i32) {
    let mut lo = i32::MAX;
    let mut hi = i32::MIN;
    for (s, e) in slots.iter().zip(exts.iter()) {
        if s[..3] == SLOT_EMPTY_SENTINEL[..3] || e[0] > e[1] {
            continue;
        }
        let base = s[2] * crate::decompress::CHUNK_Z as i32;
        lo = lo.min(base + e[0]);
        hi = hi.max(base + e[1]);
    }
    (lo, hi)
}

/// GPU.13.0 — inclusive chunk-AABB over a grid's `slot_chunk_idx`
/// shadow, skipping the [`SLOT_EMPTY_SENTINEL`] entries. Returns the
/// inverted sentinel box (`min = i32::MAX`, `max = i32::MIN`) when no
/// slot is occupied, which makes the shader's `aabb_passed` early-out
/// fire for every ray (an empty grid renders nothing).
/// GPU.13.1 — cells per axis of chunk-occupancy pyramid level `l`
/// (`l >= 1`): the pool box halves per level, floored at 1.
fn occ_level_dims(pool: [u32; 3], l: u32) -> [u32; 3] {
    [
        (pool[0] >> l).max(1),
        (pool[1] >> l).max(1),
        (pool[2] >> l).max(1),
    ]
}

/// u32 words holding level `l`'s bits.
fn occ_level_words(pool: [u32; 3], l: u32) -> usize {
    let d = occ_level_dims(pool, l);
    (d[0] * d[1] * d[2]).div_ceil(32) as usize
}

/// Pyramid levels above L0: enough that the largest pool axis
/// reaches one cell, capped by the meta's 4 offset slots.
fn occ_pyramid_levels(pool: [u32; 3]) -> u32 {
    let max_dim = pool[0].max(pool[1]).max(pool[2]).max(1);
    max_dim.ilog2().min(4)
}

/// One level-`l` cell's bit, recomputed as the OR of its (up to
/// 2×2×2) children in level `l - 1` (level 0 = the per-slot bitmap).
fn occ_cell_from_children(pool: [u32; 3], l: u32, cell: [u32; 3], child_words: &[u32]) -> bool {
    let cd = if l == 1 {
        pool
    } else {
        occ_level_dims(pool, l - 1)
    };
    for dz in 0..2u32 {
        for dy in 0..2u32 {
            for dx in 0..2u32 {
                let c = [cell[0] * 2 + dx, cell[1] * 2 + dy, cell[2] * 2 + dz];
                if c[0] >= cd[0] || c[1] >= cd[1] || c[2] >= cd[2] {
                    continue;
                }
                let idx = (c[0] + c[1] * cd[0] + c[2] * cd[0] * cd[1]) as usize;
                if (child_words[idx >> 5] >> (idx & 31)) & 1 == 1 {
                    return true;
                }
            }
        }
    }
    false
}

/// Build the whole pyramid (levels 1..=`occ_pyramid_levels`) from the
/// L0 per-slot bitmap. Returns one word-vec per level.
fn build_occ_pyramid(pool: [u32; 3], l0: &[u32]) -> Vec<Vec<u32>> {
    let levels = occ_pyramid_levels(pool);
    let mut out: Vec<Vec<u32>> = Vec::with_capacity(levels as usize);
    for l in 1..=levels {
        let d = occ_level_dims(pool, l);
        let child: &[u32] = if l == 1 { l0 } else { &out[(l - 2) as usize] };
        let mut words = vec![0u32; occ_level_words(pool, l)];
        for z in 0..d[2] {
            for y in 0..d[1] {
                for x in 0..d[0] {
                    if occ_cell_from_children(pool, l, [x, y, z], child) {
                        let idx = (x + y * d[0] + z * d[0] * d[1]) as usize;
                        words[idx >> 5] |= 1 << (idx & 31);
                    }
                }
            }
        }
        out.push(words);
    }
    out
}

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
    /// The chunk was installed/refreshed in full — every mip's
    /// occupancy, offsets, and colours are resident.
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
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
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
    // wgpu 29 widened `max_storage_buffer_binding_size` to `u64`.
    let limit_words = device.limits().max_storage_buffer_binding_size / 4;
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
        // (aabb_min, aabb_max) = 32 → 144, and GPU.13.1 adds
        // [u32;4] pyramid offsets + levels = 20; CA perf appends
        // vox_z_lo/hi + pad = 12 → 176.
        assert_eq!(std::mem::size_of::<GridStaticMeta>(), 176);
        assert_eq!(std::mem::align_of::<GridStaticMeta>(), 4);
        // The size matching is NECESSARY but not sufficient: WGSL
        // packs a bare `vec3<i32>` as 12 bytes and the next member
        // lands at ITS OWN alignment, so without the shaders'
        // `@size(16)` on aabb_min/aabb_max every tail field reads 4
        // bytes early while the total stride still agrees (this
        // silently disabled the GPU.13.1 pyramid — its level count
        // read `mip_off[3]`). Pin the host-side tail offsets the
        // annotated WGSL mirrors are written against.
        assert_eq!(std::mem::offset_of!(GridStaticMeta, aabb_min), 112);
        assert_eq!(std::mem::offset_of!(GridStaticMeta, aabb_max), 128);
        assert_eq!(std::mem::offset_of!(GridStaticMeta, chunk_occ_mip_off), 144);
        assert_eq!(std::mem::offset_of!(GridStaticMeta, chunk_occ_levels), 160);
        assert_eq!(std::mem::offset_of!(GridStaticMeta, vox_z_lo), 164);
        assert_eq!(std::mem::offset_of!(GridStaticMeta, vox_z_hi), 168);
    }

    /// GPU.13.1 — the pyramid built from an L0 bitmap must equal a
    /// brute-force OR over every level's block, and incremental
    /// child recompute must agree with the builder.
    #[test]
    fn occ_pyramid_matches_brute_force() {
        let pool = [8u32, 8, 4];
        // A scattered pattern: bits at slots (0,0,0), (7,7,3), (3,4,1).
        let mut l0 = vec![0u32; (8 * 8 * 4) / 32];
        for slot in [(0u32, 0u32, 0u32), (7, 7, 3), (3, 4, 1)] {
            let idx = (slot.0 + slot.1 * 8 + slot.2 * 64) as usize;
            l0[idx >> 5] |= 1 << (idx & 31);
        }
        let pyr = build_occ_pyramid(pool, &l0);
        assert_eq!(pyr.len(), 3, "log2(8) levels above L0");
        for l in 1..=3u32 {
            let d = occ_level_dims(pool, l);
            for z in 0..d[2] {
                for y in 0..d[1] {
                    for x in 0..d[0] {
                        // Brute force: OR of every L0 slot inside the
                        // 2^l block (clamped by the pool).
                        let mut expect = false;
                        for sz in (z << l)..(((z + 1) << l).min(pool[2])) {
                            for sy in (y << l)..(((y + 1) << l).min(pool[1])) {
                                for sx in (x << l)..(((x + 1) << l).min(pool[0])) {
                                    let i = (sx + sy * 8 + sz * 64) as usize;
                                    expect |= (l0[i >> 5] >> (i & 31)) & 1 == 1;
                                }
                            }
                        }
                        let idx = (x + y * d[0] + z * d[0] * d[1]) as usize;
                        let got = (pyr[(l - 1) as usize][idx >> 5] >> (idx & 31)) & 1 == 1;
                        assert_eq!(got, expect, "level {l} cell ({x},{y},{z})");
                        // Incremental recompute path agrees.
                        let child: &[u32] = if l == 1 { &l0 } else { &pyr[(l - 2) as usize] };
                        assert_eq!(occ_cell_from_children(pool, l, [x, y, z], child), expect);
                    }
                }
            }
        }
        // Top level is a single cell and it is occupied.
        assert_eq!(pyr[2].len(), 1);
        assert_eq!(pyr[2][0] & 1, 1);
        // An empty L0 yields an all-empty pyramid.
        let empty = build_occ_pyramid(pool, &[0u32; 8]);
        assert!(empty.iter().all(|lvl| lvl.iter().all(|&w| w == 0)));
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
