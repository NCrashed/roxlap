//! GPU backend — `roxlap-gpu` compute marcher.
//!
//! RF.2: owns the [`GpuRenderer`] plus the `Scene`→GPU bridge that
//! used to live in the scene-demo: the one-time scene upload, the
//! per-frame dirty-chunk refresh, and the per-grid world→grid-local
//! camera transform. The host hands a `Scene` + world `Camera`; this
//! backend keeps GPU residency in sync and marches it.
//!
//! Streaming/edits stay the host's job (it mutates the `Scene` before
//! calling render); this backend only *observes* chunk versions to
//! decide what to re-upload.

// The GPU bridge crosses the f64-world → f32-GPU boundary (camera
// transform) and prints a u64 byte count as MiB — both deliberate.
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use std::collections::HashMap;

use crate::{
    DynSpriteTransform, FrameParams, ImageId, KfaSprite, Kv6, Line3, QuadDraw, RenderOptions,
    Sprite, SpriteSet,
};
#[cfg(not(target_arch = "wasm32"))]
use crate::{HasDisplayHandle, HasWindowHandle};
use glam::{DVec3, IVec3};
use roxlap_core::kfa_draw::solve_kfa_limbs;
use roxlap_core::Camera;
use roxlap_formats::material::{Material, MaterialTable};
use roxlap_formats::voxel_clip::{DecodedClip, VoxelFrame};
use roxlap_gpu::{
    build_sprite_model, build_sprite_model_with_materials,
    sprite_model_from_clip_frame_with_materials, sprite_model_from_voxel_frame_with_materials,
    GpuInitError, GpuRenderer, GpuSceneResident, SpriteInstance, SpriteInstanceTransform,
    SpriteModelRegistry,
};
use roxlap_scene::{GridId, Scene};

/// Unpack a `0x00RRGGBB` packed colour (the framebuffer / `FrameParams`
/// convention) into `[R, G, B]` bytes.
fn unpack_rgb(packed: u32) -> [u8; 3] {
    [
        ((packed >> 16) & 0xff) as u8,
        ((packed >> 8) & 0xff) as u8,
        (packed & 0xff) as u8,
    ]
}

pub(crate) struct GpuBackend {
    gpu: GpuRenderer,
    /// Whole-scene residency; `None` until the first non-empty render.
    resident: Option<GpuSceneResident>,
    /// Lazily-built `grid_count == 0` resident used for the sprite-only
    /// path (a scene with no grids but with sprites — e.g. an asset
    /// viewer). Lets `render_scene` fill the sky background + far depth
    /// and run the sprite pass without any voxel grids. Kept separate
    /// from [`resident`](Self::resident) (which stays `None` for empty
    /// scenes) so [`upload_scene`](Self::upload_scene) still re-runs and
    /// picks up grids added later.
    empty_resident: Option<GpuSceneResident>,
    /// Grid ids in upload order — index = per-grid camera slot.
    grid_ids: Vec<GridId>,
    /// Sorted raw grid ids of the scene the [`resident`](Self::resident)
    /// was last built for. A scene switch swaps the whole grid set; when
    /// this no longer matches the incoming scene the resident is stale and
    /// gets rebuilt — otherwise the previous scene's grids (e.g. the World
    /// scene's ship) would ghost into a gridless scene. (The CPU backend
    /// renders straight from the scene each frame, so it can't go stale.)
    resident_scene_grids: Vec<u32>,
    /// Per-grid `chunk_idx → last-uploaded version` for the dirty poll.
    versions: Vec<HashMap<IVec3, u64>>,
    /// Instanced sprite registry + the uploaded instance list; `None`
    /// until [`set_sprites`](Self::set_sprites).
    sprite_registry: Option<SpriteModelRegistry>,
    sprite_instances: Vec<SpriteInstance>,
    /// Forward-basis [`Sprite`] per instance, parallel to
    /// [`sprite_instances`](Self::sprite_instances) (static then KFA
    /// limbs). Kept so [`render`](Self::render) can rebuild each
    /// instance's `kv6colmul` lighting table from its current pose +
    /// the frame's [`FrameParams::sprite_lighting`]. The kv6 is cloned
    /// once at registration; per-frame KFA updates only copy the basis.
    sprite_basis: Vec<Sprite>,
    /// GPU.10 KFA — per registered KFA sprite, the registry model id of
    /// each limb (in limb order). Built once by [`set_kfa_sprites`].
    kfa_limb_models: Vec<Vec<u32>>,
    /// Index into [`sprite_instances`] where the KFA limb instances
    /// begin (static [`SpriteSet`] instances occupy `[0, kfa_base)`).
    kfa_base: usize,
    /// Model templates from the last [`SpriteSet`] (`set.models`), kept so
    /// [`Self::add_dyn_instance_posed`] can clone a model's base pose/kv6 for the
    /// per-instance lighting basis. The CPU backend keeps the analogous
    /// `models`.
    sprite_models_tpl: Vec<Sprite>,
    /// Count of dynamically added instances (see [`Self::add_dyn_instance_posed`]),
    /// which occupy the tail of [`sprite_instances`] after the static set +
    /// KFA limbs. Their base index is `sprite_instances.len() - dyn_count`.
    dyn_count: usize,
    /// Per dynamic instance: `Some((clip_idx, frame))` if it plays an
    /// animated voxel clip (VCL.4), else `None`. Parallel to the dynamic
    /// tail of `sprite_instances`; swap-removed alongside it.
    /// `set_clip_frame` resolves `clips[clip_idx][frame]` → the chain to
    /// repoint the instance's model at, and records the current `frame` so
    /// it can be queried (`clip_instance_frame`).
    dyn_clip: Vec<Option<(usize, usize)>>,
    /// Registered animated voxel clips: per clip, one registry LOD-chain
    /// id per frame (the flipbook, VCL.2). A clip instance is a regular
    /// dynamic instance whose `model_id` we swap between these chains.
    clips: Vec<Vec<u32>>,
    /// GPU.12 incremental — registry LOD-chain id per static
    /// [`SpriteSet::models`] index (built in [`set_sprites`]), so
    /// [`update_sprite_model`](Self::update_sprite_model) can map a host
    /// model index to its chain for a single-model re-upload.
    sprite_model_ids: Vec<u32>,
    /// Registry model id the `G`-carve edits + its next z-layer.
    carve_model_id: Option<u32>,
    carve_z: u32,
    /// `true` once the host uploads a real sky panorama via
    /// [`set_sky_panorama`](Self::set_sky_panorama). Until then the
    /// backend mirrors [`FrameParams::sky_color`] into a 1×1 sky
    /// texture each render so the GPU sky matches the CPU's flat sky
    /// (the engine otherwise samples a default grey panorama).
    host_sky_set: bool,
    /// Last `sky_color` auto-uploaded under the parity path above —
    /// re-uploads the 1×1 texture only when it changes.
    auto_sky_color: Option<u32>,
    /// Max dirty chunks installed per frame in [`Self::refresh_dirty`].
    /// Bounds the streaming upload spike: a frame that would otherwise
    /// decompress + upload a whole batch of newly-streamed chunks at once
    /// (a multi-hundred-ms freeze) installs at most this many and lets the
    /// rest ride the next frames (refresh runs every frame). `u32::MAX`
    /// (env `ROXLAP_GPU_CHUNK_BUDGET=0`) restores the old unbounded path.
    chunk_upload_budget: u32,
    /// Flush the device staging pool every this-many frame uploads while
    /// registering a flipbook clip ([`Self::add_voxel_clip`]): an N-frame
    /// flipbook (or many clips at once) would otherwise stage N volumes of
    /// `write_buffer`s before the next submit and exhaust the pool (#4).
    /// `u32::MAX` (env `ROXLAP_GPU_CLIP_BUDGET=0`) restores the unbounded
    /// path. (Streaming clips upload one model, so they never hit this.)
    clip_upload_budget: u32,
    /// CPU shadow copy of each uploaded image (`rgba`, `w`, `h`), keyed by
    /// the [`ImageId`] `roxlap-gpu` hands back. The GPU texture isn't read
    /// back, so `pick_image`'s alpha test samples this instead. Indexed by
    /// id (resized on demand); a dropped slot is `None`.
    image_pixels: Vec<Option<(Vec<u8>, u32, u32)>>,
    /// Set when [`Self::set_dyn_instance_transform`] mutates a dynamic
    /// instance's pose; [`Self::render`] flushes all pending poses to the
    /// device once (full ordered slice) and clears it. Coalesces a whole
    /// frame's per-instance updates into one upload (avoids O(n²)).
    transforms_dirty: bool,
    /// Global voxel-material palette (TV stage): per-voxel material ids index
    /// this for opacity + blend mode. Defaults to all-[`Material::OPAQUE`], so
    /// it is inert until the host defines a translucent material via
    /// [`SceneRenderer::define_material`](crate::SceneRenderer::define_material).
    /// The device-side palette + per-voxel material buffers + blending shader
    /// land in a later TV sub-stage; this holds the authoritative table.
    materials: MaterialTable,
    /// TV: terrain colour→material map. Retained for the TV.6 GPU terrain
    /// path; the CPU backend already composites with it.
    terrain_materials: Vec<(u32, u8)>,
}

impl GpuBackend {
    /// Backend-agnostic field seeding shared by the native + wasm
    /// constructors, given an already-initialised [`GpuRenderer`].
    fn from_gpu(gpu: GpuRenderer) -> Self {
        Self {
            gpu,
            resident: None,
            empty_resident: None,
            grid_ids: Vec::new(),
            resident_scene_grids: Vec::new(),
            versions: Vec::new(),
            sprite_registry: None,
            sprite_instances: Vec::new(),
            sprite_basis: Vec::new(),
            kfa_limb_models: Vec::new(),
            kfa_base: 0,
            sprite_models_tpl: Vec::new(),
            dyn_count: 0,
            dyn_clip: Vec::new(),
            clips: Vec::new(),
            sprite_model_ids: Vec::new(),
            carve_model_id: None,
            carve_z: 0,
            host_sky_set: false,
            auto_sky_color: None,
            chunk_upload_budget: Self::chunk_upload_budget_from_env(),
            clip_upload_budget: Self::clip_upload_budget_from_env(),
            image_pixels: Vec::new(),
            transforms_dirty: false,
            materials: MaterialTable::new(),
            terrain_materials: Vec::new(),
        }
    }

    /// Frames-per-staging-flush while registering a flipbook clip — default
    /// 8, overridable via `ROXLAP_GPU_CLIP_BUDGET` (`0` = unbounded). See
    /// [`Self::clip_upload_budget`].
    fn clip_upload_budget_from_env() -> u32 {
        match std::env::var("ROXLAP_GPU_CLIP_BUDGET")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
        {
            Some(0) => u32::MAX,
            Some(n) => n,
            None => 8,
        }
    }

    /// Per-frame dirty-chunk install budget — default 2, overridable via
    /// `ROXLAP_GPU_CHUNK_BUDGET` (`0` = unbounded, the old behaviour).
    ///
    /// Each chunk install issues several `queue.write_buffer` calls
    /// (occupancy pages, colours, offsets, …); a big batch in one frame
    /// can exhaust the device staging pool, which then makes egui's own
    /// `write_buffer_with` fail (a hard panic in egui-wgpu) — seen while
    /// flying fast on the Mesa/NVK Vulkan driver. A small budget keeps the
    /// per-frame write volume bounded; the trade-off is slower chunk fill
    /// (more terrain pop-in) when moving quickly. Raise it on a driver
    /// that tolerates the bursts.
    fn chunk_upload_budget_from_env() -> u32 {
        match std::env::var("ROXLAP_GPU_CHUNK_BUDGET")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
        {
            Some(0) => u32::MAX,
            Some(n) => n,
            None => 2,
        }
    }

    /// Native: block on the async wgpu init against a window handle.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn new<W>(
        window: std::sync::Arc<W>,
        size: (u32, u32),
        opts: &RenderOptions,
    ) -> Result<Self, GpuInitError>
    where
        W: HasWindowHandle + HasDisplayHandle + Send + Sync + 'static,
    {
        let gpu = GpuRenderer::new_blocking(window, size, opts.gpu)?;
        Ok(Self::from_gpu(gpu))
    }

    /// wasm/WebGPU: await the async wgpu init against an HTML canvas.
    /// The browser drives the adapter/device futures through its event
    /// loop, so there's no blocking wrapper here.
    #[cfg(target_arch = "wasm32")]
    pub(crate) async fn new_async(
        canvas: web_sys::HtmlCanvasElement,
        size: (u32, u32),
        opts: &RenderOptions,
    ) -> Result<Self, GpuInitError> {
        let gpu = GpuRenderer::new_from_canvas(canvas, size, opts.gpu).await?;
        Ok(Self::from_gpu(gpu))
    }

    /// Build an instanced model registry from `set` and upload it.
    /// One registry model per [`SpriteSet::models`] entry; each
    /// instance references its model + carries its placed transform.
    pub(crate) fn set_sprites(&mut self, set: &SpriteSet) {
        let mut registry = SpriteModelRegistry::new();
        let model_ids: Vec<u32> = set
            .models
            .iter()
            .map(|m| registry.add_lod(build_sprite_model(&m.kv6), 4))
            .collect();

        let mut instances = Vec::with_capacity(set.instances.len());
        let mut basis = Vec::with_capacity(set.instances.len());
        for inst in &set.instances {
            let Some(&model_id) = model_ids.get(inst.model) else {
                continue;
            };
            // Per-instance Sprite = model template with the instance
            // position, so the GPU transform matches the CPU draw.
            let mut s = set.models[inst.model].clone();
            s.p = inst.pos;
            instances.push(SpriteInstance {
                model_id,
                transform: SpriteInstanceTransform::from_sprite(&s),
                material: s.material,
                alpha_mul: s.alpha_mul,
                flags: s.flags,
            });
            basis.push(s);
        }
        self.gpu.set_sprite_instances(&registry, &instances);
        self.carve_model_id = set.carve_model.and_then(|i| model_ids.get(i).copied());
        self.sprite_model_ids = model_ids;
        self.carve_z = 0;
        // Static instances reset the KFA region; re-register if needed.
        self.kfa_base = instances.len();
        self.kfa_limb_models.clear();
        self.sprite_registry = Some(registry);
        self.sprite_instances = instances;
        self.sprite_basis = basis;
        // Retain model templates for dynamic adds; a new set drops dynamics
        // (including animated-clip instances + the clip chains, which lived
        // in the now-replaced registry — the host re-registers clips after
        // a `set_sprites`, like the streamed sprite models).
        self.sprite_models_tpl.clone_from(&set.models);
        self.dyn_count = 0;
        self.dyn_clip.clear();
        self.clips.clear();
    }

    /// Append one dynamic instance of `model_index` pre-posed by `xf`;
    /// returns its dynamic-sublist index (the new last). Uses the
    /// incremental `append_sprite_instances` (no registry rebuild) and
    /// mirrors the instance into the parallel
    /// `sprite_instances`/`sprite_basis` so the per-frame lighting +
    /// transform updates keep covering it.
    pub(crate) fn add_dyn_instance_posed(
        &mut self,
        model_index: usize,
        xf: DynSpriteTransform,
    ) -> usize {
        let idx = self.dyn_count;
        let (Some(&chain_id), Some(model), Some(registry)) = (
            self.sprite_model_ids.get(model_index),
            self.sprite_models_tpl.get(model_index),
            self.sprite_registry.as_ref(),
        ) else {
            return idx;
        };
        let mut s = model.clone();
        xf.apply_to(&mut s);
        let inst = SpriteInstance {
            model_id: chain_id,
            transform: SpriteInstanceTransform::from_sprite(&s),
            material: s.material,
            alpha_mul: s.alpha_mul,
            flags: s.flags,
        };
        self.gpu.append_sprite_instances(registry, &[inst]);
        self.sprite_instances.push(inst);
        self.sprite_basis.push(s);
        self.dyn_clip.push(None);
        self.dyn_count += 1;
        idx
    }

    /// Update dynamic instance `idx`'s pose (position + orientation) in
    /// the parallel CPU-side mirrors and flag the instance buffer dirty;
    /// the new transform is flushed to the GPU once per [`Self::render`]
    /// (so spinning a whole cluster of instances is a single device
    /// upload, not one per instance). No-op if `idx` is out of range.
    pub(crate) fn set_dyn_instance_transform(&mut self, idx: usize, xf: DynSpriteTransform) {
        if idx >= self.dyn_count {
            return;
        }
        let gpu_index = (self.sprite_instances.len() - self.dyn_count) + idx;
        if let Some(b) = self.sprite_basis.get_mut(gpu_index) {
            xf.apply_to(b);
            let t = SpriteInstanceTransform::from_sprite(b);
            self.sprite_instances[gpu_index].transform = t;
            self.transforms_dirty = true;
        }
    }

    /// Set dynamic instance `idx`'s voxel-material id (TV stage). Updates the
    /// host-side `sprite_basis` + `sprite_instances` mirrors and flags the
    /// instance buffer dirty; the new material rides the next [`Self::render`]
    /// flush (coalesced with pose updates) and the per-frame cull. No-op if
    /// out of range.
    pub(crate) fn set_dyn_instance_material(&mut self, idx: usize, material: u8) {
        if idx >= self.dyn_count {
            return;
        }
        let gpu_index = (self.sprite_instances.len() - self.dyn_count) + idx;
        if let Some(b) = self.sprite_basis.get_mut(gpu_index) {
            b.material = material;
        }
        self.sprite_instances[gpu_index].material = material;
        self.transforms_dirty = true;
    }

    /// Set dynamic instance `idx`'s per-instance alpha multiplier (TV stage,
    /// `255` = unscaled). Same coalesced-flush path as
    /// [`Self::set_dyn_instance_material`]. No-op if out of range.
    pub(crate) fn set_dyn_instance_alpha(&mut self, idx: usize, alpha_mul: u8) {
        if idx >= self.dyn_count {
            return;
        }
        let gpu_index = (self.sprite_instances.len() - self.dyn_count) + idx;
        if let Some(b) = self.sprite_basis.get_mut(gpu_index) {
            b.alpha_mul = alpha_mul;
        }
        self.sprite_instances[gpu_index].alpha_mul = alpha_mul;
        self.transforms_dirty = true;
    }

    /// Register a new sprite model incrementally (its full LOD chain),
    /// returning its positional host index (== registry chain id). Lazily
    /// creates the registry + resident if none exists yet, so this works
    /// before any `set_sprites`. Mirrors the new chain into the host-side
    /// `sprite_model_ids` / `sprite_models_tpl`.
    pub(crate) fn add_model(&mut self, kv6: &Kv6) -> usize {
        self.add_model_chain(build_sprite_model(kv6), kv6)
    }

    /// Register a model whose voxels carry per-voxel material ids (TV.3),
    /// classified by colour from `material_map`. The per-voxel `materials`
    /// ride the `SpriteModel`; the device-side material buffer + shader
    /// lookup land in TV.3b (until then the GPU renders these with the
    /// instance's uniform material).
    pub(crate) fn add_model_with_materials(
        &mut self,
        kv6: &Kv6,
        material_map: &[(u32, u8)],
    ) -> usize {
        self.add_model_chain(build_sprite_model_with_materials(kv6, material_map), kv6)
    }

    /// Shared body of [`Self::add_model`] / [`Self::add_model_with_materials`]:
    /// add the model's LOD chain to the resident registry + mirror the host
    /// template.
    fn add_model_chain(&mut self, model: roxlap_gpu::SpriteModel, kv6: &Kv6) -> usize {
        let mut registry = self.sprite_registry.take().unwrap_or_default();
        let chain_id = registry.add_lod(model, 4);
        // `gpu.add_sprite_model` establishes residency (zero-instance
        // upload) if none yet, else appends just this chain's volume.
        self.gpu.add_sprite_model(&registry, chain_id);
        let host_idx = self.sprite_model_ids.len();
        self.sprite_model_ids.push(chain_id);
        self.sprite_models_tpl
            .push(Sprite::axis_aligned(kv6.clone(), [0.0, 0.0, 0.0]));
        self.sprite_registry = Some(registry);
        host_idx
    }

    /// Remove host model `host_idx`: tombstone its chain on the GPU
    /// resident **first** (sets the `dead` flags), then free its voxel
    /// data in the CPU-side registry, then drop the host-side template.
    /// Ordering matters — `gpu.remove_sprite_model` must run before
    /// `registry.remove` so the resident's dead-aware compact only ever
    /// reads live entries. No-op if `host_idx` is unknown.
    pub(crate) fn remove_model(&mut self, host_idx: usize) {
        let Some(&chain_id) = self.sprite_model_ids.get(host_idx) else {
            return;
        };
        // 1. Resident tombstone (frees GPU colour slots, marks `dead`).
        self.gpu.remove_sprite_model(chain_id);
        // 2. CPU registry free (must follow step 1 — see method doc).
        if let Some(reg) = self.sprite_registry.as_mut() {
            reg.remove(chain_id);
        }
        // 3. Drop the host template's kv6; keep the slot (id never reused).
        if let Some(t) = self.sprite_models_tpl.get_mut(host_idx) {
            *t = Sprite::axis_aligned(Kv6::from_fn(1, 1, 1, |_, _, _| None), [0.0, 0.0, 0.0]);
        }
    }

    /// Reclaim the GPU buffer holes left by [`Self::remove_model`] by
    /// repacking the resident registry to its live models only. Ids are
    /// preserved. No-op if no registry is resident.
    pub(crate) fn compact_models(&mut self) {
        if let Some(reg) = self.sprite_registry.as_ref() {
            self.gpu.compact_sprite_models(reg);
        }
    }

    /// Define global voxel-material `id` (TV stage). Id 0 is reserved as
    /// [`Material::OPAQUE`]; defining it is a no-op returning `false`. The
    /// table is held authoritatively here; device upload + blending land in
    /// a later TV sub-stage.
    pub(crate) fn define_material(&mut self, id: u8, mat: Material) -> bool {
        self.materials.set(id, mat)
    }

    /// The material at `id` ([`Material::OPAQUE`] for any never-defined id).
    pub(crate) fn material(&self, id: u8) -> Material {
        self.materials.get(id)
    }

    /// Set the terrain colour→material map (TV.4). Retained for the TV.6 GPU
    /// terrain transparency path.
    pub(crate) fn set_terrain_materials(&mut self, map: &[(u32, u8)]) {
        self.terrain_materials = map.to_vec();
    }

    /// Remove the dynamic instance at dynamic-sublist index `idx` by
    /// swap-remove. Returns `Some(old_last)` (dynamic-local) if a
    /// different instance filled the hole, else `None` — matching the CPU
    /// backend so the facade's handle fixup is identical.
    pub(crate) fn remove_dyn_instance(&mut self, idx: usize) -> Option<usize> {
        if idx >= self.dyn_count {
            return None;
        }
        let base = self.sprite_instances.len() - self.dyn_count;
        let gpu_index = base + idx;
        let moved = self.gpu.remove_sprite_instance(gpu_index);
        // Mirror the swap-remove on the parallel arrays (swap_remove on a
        // Vec swaps with the last element — the last dynamic instance,
        // since dynamics are the tail — exactly as the GPU cull does).
        self.sprite_instances.swap_remove(gpu_index);
        self.sprite_basis.swap_remove(gpu_index);
        self.dyn_clip.swap_remove(idx); // dyn-local index (parallel to the tail)
        self.dyn_count -= 1;
        moved.map(|m| m - base)
    }

    /// Register an animated voxel clip (VCL.4): upload every frame as an
    /// LOD chain (the flipbook). With a non-empty `material_map` (TV.3), each
    /// frame's voxels are classified into per-voxel material ids by colour —
    /// the clip analogue of [`Self::add_model_with_materials`]. An empty map
    /// is the plain all-opaque clip. Returns its positional clip index.
    /// Lazily creates the registry/resident if none exists yet (like
    /// [`Self::add_model`]).
    pub(crate) fn add_voxel_clip_with_materials(
        &mut self,
        clip: &DecodedClip,
        material_map: &[(u32, u8)],
    ) -> usize {
        let mut registry = self.sprite_registry.take().unwrap_or_default();
        let mut chains = Vec::with_capacity(clip.frames.len());
        for frame in 0..clip.frames.len() {
            let chain = registry.add_lod(
                sprite_model_from_clip_frame_with_materials(clip, frame, material_map),
                4,
            );
            // Establishes residency (zero-instance upload) if none yet, else
            // appends just this chain's volume.
            self.gpu.add_sprite_model(&registry, chain);
            chains.push(chain);
            // #4: keep the staging pool bounded — an N-frame flipbook stages
            // N volumes of `write_buffer`s before the next submit; flush in
            // batches so a big clip (or many at once) can't exhaust it.
            if self.clip_upload_budget != u32::MAX
                && (frame as u32 + 1) % self.clip_upload_budget == 0
            {
                self.gpu.flush_writes();
            }
        }
        self.sprite_registry = Some(registry);
        let idx = self.clips.len();
        self.clips.push(chains);
        // Final flush of this clip's residual (< budget) batch so a later
        // clip registered the same frame starts from a drained pool.
        if self.clip_upload_budget != u32::MAX {
            self.gpu.flush_writes();
        }
        idx
    }

    /// Tombstone clip `clip_idx`: remove each frame chain from the resident
    /// registry (instances of it then draw nothing). Chain ids are never
    /// reused, so other clips stay valid. No-op if out of range.
    pub(crate) fn remove_voxel_clip(&mut self, clip_idx: usize) {
        let Some(chains) = self.clips.get(clip_idx).cloned() else {
            return;
        };
        for chain in chains {
            // Resident tombstone first, then CPU registry free (see
            // `remove_model`'s ordering note).
            self.gpu.remove_sprite_model(chain);
            if let Some(reg) = self.sprite_registry.as_mut() {
                reg.remove(chain);
            }
        }
        if let Some(c) = self.clips.get_mut(clip_idx) {
            c.clear();
        }
        // Detach instances that were playing this clip so they're no longer
        // treated as clip instances (their now-tombstoned chain draws
        // nothing regardless; this keeps the bookkeeping honest).
        for slot in &mut self.dyn_clip {
            if matches!(slot, Some((c, _)) if *c == clip_idx) {
                *slot = None;
            }
        }
    }

    /// Append a dynamic instance playing clip `clip_idx`, posed by `xf`,
    /// starting on frame 0. Returns its dynamic-sublist index. A clip
    /// instance is a regular dynamic instance whose `model_id` is one of
    /// the clip's frame chains; [`Self::set_clip_frame`] swaps it.
    pub(crate) fn add_clip_instance(&mut self, clip_idx: usize, xf: DynSpriteTransform) -> usize {
        let idx = self.dyn_count;
        let (Some(&chain0), Some(registry)) = (
            self.clips.get(clip_idx).and_then(|c| c.first()),
            self.sprite_registry.as_ref(),
        ) else {
            return idx;
        };
        // Pose carrier — only `s/h/f/p` matter (the volume is the chain's).
        let mut s = Sprite::axis_aligned(Kv6::from_fn(1, 1, 1, |_, _, _| None), [0.0, 0.0, 0.0]);
        xf.apply_to(&mut s);
        let inst = SpriteInstance {
            model_id: chain0,
            transform: SpriteInstanceTransform::from_sprite(&s),
            material: s.material,
            alpha_mul: s.alpha_mul,
            flags: s.flags,
        };
        self.gpu.append_sprite_instances(registry, &[inst]);
        self.sprite_instances.push(inst);
        self.sprite_basis.push(s);
        self.dyn_clip.push(Some((clip_idx, 0)));
        self.dyn_count += 1;
        idx
    }

    /// Select the playback frame of clip instance `idx`: repoint its model
    /// at `clips[clip_idx][frame]` (the cheap per-frame flipbook step — no
    /// volume re-upload). No-op if `idx` isn't a clip instance / out of
    /// range.
    pub(crate) fn set_clip_frame(&mut self, idx: usize, frame: usize) {
        if idx >= self.dyn_count {
            return;
        }
        let Some(Some((clip_idx, _))) = self.dyn_clip.get(idx).copied() else {
            return;
        };
        let Some(&chain) = self.clips.get(clip_idx).and_then(|c| c.get(frame)) else {
            return;
        };
        let gpu_index = (self.sprite_instances.len() - self.dyn_count) + idx;
        self.sprite_instances[gpu_index].model_id = chain;
        if let Some(reg) = self.sprite_registry.as_ref() {
            self.gpu.set_sprite_instance_model(reg, gpu_index, chain);
        }
        self.dyn_clip[idx] = Some((clip_idx, frame));
    }

    /// The frame a clip instance is currently showing, or `None` if `idx`
    /// isn't a (live) clip instance.
    pub(crate) fn clip_instance_frame(&self, idx: usize) -> Option<usize> {
        match self.dyn_clip.get(idx) {
            Some(Some((_, frame))) => Some(*frame),
            _ => None,
        }
    }

    /// Re-upload **one** frame's volume of clip `clip_idx` in place (the
    /// editor's single-voxel paint), without touching the other frames. The
    /// frame's LOD chain is rebuilt + re-uploaded — O(1 frame), not the
    /// whole flipbook. No-op if out of range.
    pub(crate) fn update_clip_frame(
        &mut self,
        clip_idx: usize,
        frame: usize,
        vf: &VoxelFrame,
        dims: [u32; 3],
        pivot: [f32; 3],
        voxel_world_size: f32,
        material_map: &[(u32, u8)],
    ) -> bool {
        let Some(&chain_id) = self.clips.get(clip_idx).and_then(|c| c.get(frame)) else {
            return false;
        };
        let Some(reg) = self.sprite_registry.as_mut() else {
            return false;
        };
        // Recompute dirs so the edited frame's model is byte-identical to the
        // register path (`sprite_model_from_clip_frame`), not flat zeros —
        // matters only if per-instance shading is ever applied to a clip.
        // `material_map` re-classifies the edited frame's voxels (TV.3) so an
        // in-place edit keeps the clip's per-voxel materials.
        let dirs = vf.dirs(dims);
        *reg.model_mut(chain_id) = sprite_model_from_voxel_frame_with_materials(
            vf,
            &dirs,
            dims,
            pivot,
            voxel_world_size,
            material_map,
        );
        reg.rebuild_lod(chain_id);
        self.gpu.update_sprite_model(reg, chain_id);
        true
    }

    /// Register KFA sprites: append each limb's kv6 as an instanced
    /// model (with an LOD chain, like static sprites) and seed one
    /// instance per limb at its current pose. Volumes upload once here;
    /// [`update_kfa_poses`](Self::update_kfa_poses) only moves them.
    pub(crate) fn set_kfa_sprites(&mut self, kfas: &mut [KfaSprite]) {
        // Build on top of whatever static sprites already exist so the
        // single GPU sprite pass draws both. `set_sprites` left
        // `kfa_base` at the static instance count.
        let mut registry = self.sprite_registry.take().unwrap_or_default();
        let mut instances = std::mem::take(&mut self.sprite_instances);
        // Truncate any prior KFA basis; static basis stays in [0, kfa_base).
        self.sprite_basis.truncate(self.kfa_base);
        self.kfa_base = instances.len();
        self.kfa_limb_models.clear();

        for kfa in kfas.iter_mut() {
            // Pose the limbs so the seed instances are correct frame 0.
            solve_kfa_limbs(kfa);
            let mut limb_models = Vec::with_capacity(kfa.limbs.len());
            for limb in &kfa.limbs {
                let id = registry.add_lod(build_sprite_model(&limb.kv6), 4);
                limb_models.push(id);
                instances.push(SpriteInstance {
                    model_id: id,
                    transform: SpriteInstanceTransform::from_sprite(limb),
                    material: limb.material,
                    alpha_mul: limb.alpha_mul,
                    flags: limb.flags,
                });
                self.sprite_basis.push(limb.clone());
            }
            self.kfa_limb_models.push(limb_models);
        }

        self.gpu.set_sprite_instances(&registry, &instances);
        self.sprite_registry = Some(registry);
        self.sprite_instances = instances;
    }

    /// Re-pose registered KFA limbs and push the new transforms to the
    /// GPU without re-uploading any model volume (GPU.10 cheap path).
    pub(crate) fn update_kfa_poses(&mut self, kfas: &mut [KfaSprite]) {
        if self.kfa_limb_models.is_empty() {
            return;
        }
        let mut idx = self.kfa_base;
        for kfa in kfas.iter_mut() {
            solve_kfa_limbs(kfa);
            for limb in &kfa.limbs {
                if let Some(inst) = self.sprite_instances.get_mut(idx) {
                    inst.transform = SpriteInstanceTransform::from_sprite(limb);
                }
                // Copy only the posed basis (no kv6 re-clone) so the
                // next `render` rebuilds this limb's lighting table.
                if let Some(b) = self.sprite_basis.get_mut(idx) {
                    b.p = limb.p;
                    b.s = limb.s;
                    b.h = limb.h;
                    b.f = limb.f;
                }
                idx += 1;
            }
        }
        self.gpu
            .update_sprite_instance_transforms(&self.sprite_instances);
    }

    /// Carve the next z-layer off the carve model, rebuild its LOD
    /// chain, and re-upload — GPU.12 copy-on-modify. Returns voxels
    /// removed (`0` when nothing to carve / no carve model).
    pub(crate) fn carve_active_sprite(&mut self) -> u32 {
        let Some(id) = self.carve_model_id else {
            return 0;
        };
        let Some(reg) = self.sprite_registry.as_mut() else {
            return 0;
        };
        let dims = reg.model(id).dims;
        let z = self.carve_z;
        if z >= dims[2] {
            return 0;
        }
        let m = reg.model_mut(id);
        let mut removed = 0u32;
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                if m.set_voxel(x, y, z, None) {
                    removed += 1;
                }
            }
        }
        reg.rebuild_lod(id);
        self.carve_z = z + 1;
        // GPU.12 incremental: re-upload only this model's chain, not the
        // whole registry (instances/cull/bounds are unchanged by a carve).
        self.gpu.update_sprite_model(reg, id);
        removed
    }

    /// GPU.12 incremental — re-register host model `model_index`'s
    /// geometry from the (already-edited) `kv6`, refreshing only that LOD
    /// chain's GPU data. The instance set is untouched. No-op if no
    /// registry is resident or `model_index` is unknown.
    pub(crate) fn update_sprite_model(&mut self, model_index: usize, kv6: &Kv6) {
        self.update_sprite_model_with_materials(model_index, kv6, &[]);
    }

    /// Like [`Self::update_sprite_model`] but classifies the rebuilt voxels
    /// into per-voxel material ids by colour (TV.3) via `material_map` — the
    /// material-aware refresh behind the streaming-clip path. An empty map is
    /// identical to [`Self::update_sprite_model`].
    pub(crate) fn update_sprite_model_with_materials(
        &mut self,
        model_index: usize,
        kv6: &Kv6,
        material_map: &[(u32, u8)],
    ) {
        let Some(&chain_id) = self.sprite_model_ids.get(model_index) else {
            return;
        };
        let Some(reg) = self.sprite_registry.as_mut() else {
            return;
        };
        // Rebuild mip-0 from the edited kv6, then refresh the coarse mips
        // so every LOD level matches before the single-chain re-upload.
        *reg.model_mut(chain_id) = build_sprite_model_with_materials(kv6, material_map);
        reg.rebuild_lod(chain_id);
        self.gpu.update_sprite_model(reg, chain_id);
    }

    pub(crate) fn adapter_info(&self) -> &str {
        self.gpu.adapter_info()
    }

    /// World-t depth at window pixel `(x, y)` from the last frame (for
    /// screen→world picking). See [`SceneRenderer::pick_depth`].
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn pick_depth(&self, x: u32, y: u32) -> Option<f32> {
        self.gpu.read_depth_pixel(x, y)
    }

    /// wasm: depth picking is deferred on the GPU path — WebGPU has no
    /// blocking readback, so the staging-buffer map can't be awaited
    /// synchronously here. Always `None`; the CPU fallback picks
    /// normally from its in-memory z-buffer.
    #[cfg(target_arch = "wasm32")]
    #[allow(clippy::unused_self)]
    pub(crate) fn pick_depth(&self, _x: u32, _y: u32) -> Option<f32> {
        None
    }

    /// World-space view ray for pixel `(x, y)` under the GPU marcher's
    /// projection. See [`SceneRenderer::pixel_ray`].
    pub(crate) fn pixel_ray(&self, camera: &Camera, x: f64, y: f64) -> Option<[f64; 3]> {
        self.gpu
            .pixel_ray(camera.right, camera.down, camera.forward, x, y)
    }

    pub(crate) fn resize(&mut self, width: u32, height: u32) {
        self.gpu.resize(width, height);
    }

    /// Upload a sky panorama for the GPU shader's sky sampling.
    pub(crate) fn set_sky_panorama(&mut self, rgba: &[u8], w: u32, h: u32) {
        self.gpu.set_sky_panorama(rgba, w, h);
        // The host owns the sky now — stop mirroring `sky_color`.
        self.host_sky_set = true;
    }

    /// Mirror the CPU path's flat sky + distance fog onto the GPU from
    /// the per-frame [`FrameParams`]. The GPU marcher samples its own
    /// sky *texture* (default grey) and carries its own fog state, so
    /// without this the GPU diverges from the CPU's `sky_color` /
    /// `fog_color` every frame. Skips the sky mirror once the host has
    /// uploaded a real panorama.
    fn sync_sky_and_fog(&mut self, frame: &FrameParams) {
        if !self.host_sky_set && self.auto_sky_color != Some(frame.sky_color) {
            let [r, g, b] = unpack_rgb(frame.sky_color);
            self.gpu.set_sky_panorama(&[r, g, b, 0xff], 1, 1);
            self.auto_sky_color = Some(frame.sky_color);
        }

        // Config-driven fog, matching the CPU/DDA path (which reads the
        // pool's fog state): on iff `fog_max_scan_dist > 0`, a linear
        // ramp from t=0 to that distance toward `fog_color`. Off ⇒ a huge
        // far ≈ no fog. The host (e.g. the scene demo) drives the fog
        // colour/distance via `FrameParams`.
        #[allow(clippy::cast_precision_loss)]
        let far = if frame.fog_max_scan_dist > 0 {
            frame.fog_max_scan_dist as f32
        } else {
            1.0e30
        };
        let [r, g, b] = unpack_rgb(frame.fog_color);
        let color = [
            f32::from(r) / 255.0,
            f32::from(g) / 255.0,
            f32::from(b) / 255.0,
        ];
        // near = 0 + linear curve (matched in the shader) = the CPU LUT
        // / DDA `clamp(t / far)` blend.
        self.gpu.set_fog(color, 0.0, far);
    }

    pub(crate) fn render(&mut self, scene: &mut Scene, camera: &Camera, frame: &FrameParams) {
        // CPU/GPU parity: mirror the frame's flat sky + fog onto the GPU
        // (which carries its own sky texture + fog state).
        self.sync_sky_and_fog(frame);

        // Drop a resident built for a different scene (a scene switch swaps
        // the whole grid set). Without this the previous scene's grids —
        // e.g. the World scene's ship saucer — ghost into a gridless scene,
        // since `refresh_dirty` only walks the resident's own grid ids.
        if self.resident.is_some() && !self.resident_matches_scene(scene) {
            self.resident = None;
        }

        if self.resident.is_none() {
            self.upload_scene(scene);
        } else {
            self.refresh_dirty(scene);
        }

        // Flush any dynamic-instance pose changes accumulated this frame
        // via `set_dyn_instance_transform` in a single device upload (the
        // full ordered slice), coalescing N per-instance setters into one.
        // The per-frame cull re-reads the new `cull` data, so no extra GPU
        // work beyond this write.
        if self.transforms_dirty {
            self.gpu
                .update_sprite_instance_transforms(&self.sprite_instances);
            self.transforms_dirty = false;
        }

        // Per-frame GPU scene-LOD knob (GPU.11.1).
        self.gpu.set_scene_mip_scan_dist(frame.gpu_mip_scan_dist);

        // Per-face grid shading (voxlap setsideshades) — the scene-DDA
        // pass darkens a hit voxel's brightness by the hit face's shade,
        // matching the CPU rasteriser. With a flat (un-baked) brightness
        // byte it's pure runtime side-shading; with baked light it
        // stacks, exactly as voxlap does. Default [0;6] = no shading.
        self.gpu.set_scene_side_shades(frame.side_shades);

        // DL — translate this frame's world-space LightRig into each grid's
        // local frame and hand it to the GPU (GPU-only; the CPU backend
        // ignores lights). `None` clears them ⇒ the pre-DL render.
        self.sync_lights(scene, frame);

        // TV — mirror the global voxel-material palette to the sprite pass
        // (cheap; the shader only leaves the opaque fast-path once a
        // translucent material is defined).
        self.gpu.set_sprite_materials(&self.materials);
        // TV.6 — and the terrain palette + colour→material map to the scene
        // pass (glass/water as world geometry).
        self.gpu
            .set_scene_terrain_materials(&self.materials, &self.terrain_materials);

        // Sprites render flat-lit (identity `kv6colmul`, the GPU default)
        // to match the CPU backend's clean-room DDA sprite raycaster —
        // the voxlap directional `sprite_colmul` shading is not used.

        let cameras = self.grid_cameras(scene, camera);
        // XS.3 — per-grid world transforms (parallel to `cameras`) so the
        // scene shader can lift a shadow ray to world space and test it
        // against every grid (cross-grid shadows).
        let grid_world = self.grid_world_transforms(scene);
        // Sprites are world-space, so they project through the world
        // camera (identity transform), not any grid-local one. Without
        // this the GPU sprite pass used `cameras[0]` and shifted every
        // instance by grid 0's origin/rotation.
        let sprite_camera = grid_local_camera(glam::DQuat::IDENTITY, DVec3::ZERO, camera);
        if let Some(resident) = &self.resident {
            self.gpu.render_scene(
                resident,
                &cameras,
                &grid_world,
                &sprite_camera,
                frame.gpu_fov_y_rad,
                frame.gpu_max_outer_steps,
            );
        } else if !self.sprite_instances.is_empty() {
            // Sprite-only scene (no voxel grids — e.g. an asset/model
            // viewer). Render through a zero-grid resident so the scene
            // pass fills the sky background + far depth and the sprite
            // pass composites the models over it (CPU/GPU parity). The
            // sky comes from the 1×1 auto-sky (= `frame.sky_color`), so
            // the background matches the CPU backend.
            if self.empty_resident.is_none() {
                let info = roxlap_gpu::SceneUpload { grids: Vec::new() };
                self.empty_resident = Some(GpuSceneResident::upload(self.gpu.device(), &info));
            }
            let empty = self.empty_resident.as_ref().expect("just built");
            self.gpu.render_scene(
                empty,
                &[],
                &[],
                &sprite_camera,
                frame.gpu_fov_y_rad,
                frame.gpu_max_outer_steps,
            );
        } else {
            // Truly empty (no grids, no sprites) — clear to colour
            // (deferred, so a HUD can still be painted over it).
            self.gpu.render_clear_deferred();
        }
    }

    /// Present the frame `render` composited, with no UI overlay.
    pub(crate) fn present(&mut self) {
        self.gpu.present();
    }

    /// Drain in-flight GPU work before teardown (see
    /// [`SceneRenderer::wait_idle`](crate::SceneRenderer::wait_idle)).
    pub(crate) fn wait_idle(&mut self) {
        self.gpu.wait_idle();
    }

    /// Horizontal scene flip — mirrors the marched scene + line/image
    /// overlays on present, leaving egui upright.
    pub(crate) fn set_flip_x(&mut self, flip: bool) {
        self.gpu.set_flip_x(flip);
    }

    /// Draw depth-tested world-space line segments over the pending frame
    /// (L3.2). Converts the facade [`Line3`]s + world `camera` to the GPU
    /// line types and runs the `roxlap-gpu` line pipeline, which projects
    /// the endpoints (marcher pinhole), expands them to screen quads, and
    /// composites with a `LoadOp::Load` pass. Depth-tested lines are
    /// occluded by nearer marched geometry (euclidean `best_t`).
    pub(crate) fn draw_lines(&mut self, camera: &Camera, lines: &[Line3]) {
        if lines.is_empty() {
            return;
        }
        let cam = roxlap_gpu::GpuLineCamera {
            pos: camera.pos.map(|v| v as f32),
            right: camera.right.map(|v| v as f32),
            down: camera.down.map(|v| v as f32),
            forward: camera.forward.map(|v| v as f32),
        };
        let glines: Vec<roxlap_gpu::GpuLine> = lines
            .iter()
            .map(|l| {
                // 0xAARRGGBB → straight RGBA in 0..=1 (alpha = over-blend).
                let a = ((l.color >> 24) & 0xff) as f32 / 255.0;
                let r = ((l.color >> 16) & 0xff) as f32 / 255.0;
                let g = ((l.color >> 8) & 0xff) as f32 / 255.0;
                let b = (l.color & 0xff) as f32 / 255.0;
                roxlap_gpu::GpuLine {
                    a: [l.a[0] as f32, l.a[1] as f32, l.a[2] as f32],
                    b: [l.b[0] as f32, l.b[1] as f32, l.b[2] as f32],
                    color: [r, g, b, a],
                    width_px: l.width_px,
                    depth_test: l.depth_test,
                }
            })
            .collect();
        self.gpu.draw_lines_deferred(&cam, &glines);
    }

    /// Upload (or replace) an RGBA8 image-sprite texture, keeping a CPU
    /// shadow copy so `pick_image`'s alpha test can sample it (the GPU
    /// texture isn't read back).
    pub(crate) fn upload_image(&mut self, rgba: &[u8], width: u32, height: u32) -> ImageId {
        let id = self.gpu.upload_image(rgba, width, height);
        let valid =
            width != 0 && height != 0 && rgba.len() == (width as usize) * (height as usize) * 4;
        let shadow = valid.then(|| (rgba.to_vec(), width, height));
        if id >= self.image_pixels.len() {
            self.image_pixels.resize_with(id + 1, || None);
        }
        self.image_pixels[id] = shadow;
        ImageId(id)
    }

    /// Release a previously uploaded image-sprite texture.
    pub(crate) fn drop_image(&mut self, id: ImageId) {
        self.gpu.drop_image(id.0);
        if let Some(slot) = self.image_pixels.get_mut(id.0) {
            *slot = None;
        }
    }

    /// Source `(width, height)` of an uploaded image, for `pick_image`.
    pub(crate) fn image_dims(&self, id: ImageId) -> Option<(u32, u32)> {
        self.image_pixels
            .get(id.0)
            .and_then(Option::as_ref)
            .map(|(_, w, h)| (*w, *h))
    }

    /// Alpha byte of texel `(tx, ty)` from the shadow copy; `0` for an
    /// unknown id / out-of-range texel.
    pub(crate) fn image_alpha_at(&self, id: ImageId, tx: u32, ty: u32) -> u8 {
        let Some(Some((rgba, w, h))) = self.image_pixels.get(id.0) else {
            return 0;
        };
        if tx >= *w || ty >= *h {
            return 0;
        }
        let idx = ((ty * w + tx) * 4 + 3) as usize;
        rgba.get(idx).copied().unwrap_or(0)
    }

    /// Project a world point to window pixels under the marcher's
    /// projection. See [`SceneRenderer::project_point`].
    pub(crate) fn project_point(&self, camera: &Camera, world: [f32; 3]) -> Option<(f32, f32)> {
        self.gpu.project_point(
            camera.pos.map(|v| v as f32),
            camera.right.map(|v| v as f32),
            camera.down.map(|v| v as f32),
            camera.forward.map(|v| v as f32),
            world,
        )
    }

    /// Draw world-space 2D image sprites over the pending frame — the
    /// textured-quad sibling of [`Self::draw_lines`]. Converts the
    /// facade-resolved [`QuadDraw`]s + world `camera` to the GPU image
    /// types and runs the `roxlap-gpu` image pipeline (perspective-correct
    /// UV, manual depth test against the marched `best_t`).
    pub(crate) fn draw_images(&mut self, camera: &Camera, quads: &[QuadDraw]) {
        if quads.is_empty() {
            return;
        }
        let cam = roxlap_gpu::GpuLineCamera {
            pos: camera.pos.map(|v| v as f32),
            right: camera.right.map(|v| v as f32),
            down: camera.down.map(|v| v as f32),
            forward: camera.forward.map(|v| v as f32),
        };
        let gquads: Vec<roxlap_gpu::GpuImageQuad> = quads
            .iter()
            .map(|q| {
                // 0xAARRGGBB tint → straight RGBA in 0..=1.
                let a = ((q.tint >> 24) & 0xff) as f32 / 255.0;
                let r = ((q.tint >> 16) & 0xff) as f32 / 255.0;
                let g = ((q.tint >> 8) & 0xff) as f32 / 255.0;
                let b = (q.tint & 0xff) as f32 / 255.0;
                roxlap_gpu::GpuImageQuad {
                    corners: q.corners,
                    image: q.image.0,
                    tint: [r, g, b, a],
                    depth_test: q.depth_test,
                    alpha_cutoff: q.alpha_cutoff,
                }
            })
            .collect();
        self.gpu.draw_images_deferred(&cam, &gquads);
    }

    /// Overlay egui on the pending frame, then present (`hud` feature).
    #[cfg(feature = "hud")]
    pub(crate) fn paint_egui(
        &mut self,
        jobs: &[egui::ClippedPrimitive],
        textures: &egui::TexturesDelta,
        pixels_per_point: f32,
    ) {
        self.gpu.paint_egui(jobs, textures, pixels_per_point);
    }

    /// Decompress every materialised chunk of every grid and upload as
    /// one [`GpuSceneResident`]; record the grid order + seed the
    /// dirty-version trackers. Moved verbatim from the scene-demo's
    /// `upload_first_scene` (minus the streaming pump, which the host
    /// drives before calling render).
    /// Whether the cached [`resident`](Self::resident) was built for
    /// `scene`'s grid set (sorted raw ids). A mismatch ⇒ a scene switch or
    /// grid add/remove invalidated it.
    fn resident_matches_scene(&self, scene: &Scene) -> bool {
        let mut ids: Vec<u32> = scene.grids().map(|(g, _)| g.raw()).collect();
        ids.sort_unstable();
        ids == self.resident_scene_grids
    }

    fn upload_scene(&mut self, scene: &Scene) {
        // Snapshot the scene's grid set so a later scene switch is detected
        // (see `resident_matches_scene`). Recorded even when no grids are
        // uploadable yet — the empty set still has to match next frame.
        self.resident_scene_grids = {
            let mut ids: Vec<u32> = scene.grids().map(|(g, _)| g.raw()).collect();
            ids.sort_unstable();
            ids
        };

        let mut grids_by_id: Vec<_> = scene.grids().collect();
        grids_by_id.sort_by_key(|(gid, _)| gid.raw());

        let mut scene_grids: Vec<roxlap_gpu::GridUpload> = Vec::new();
        let mut grid_ids: Vec<GridId> = Vec::new();
        let mut total_chunks = 0usize;
        for (gid, grid) in grids_by_id {
            let is_streaming = grid.generator.is_some();
            // Skip truly-static empty grids (they'll never gain
            // chunks). A STREAMING grid is registered even when empty
            // so it lands in `grid_ids` — otherwise its chunks, which
            // arrive over later frames via the background generator,
            // would never be installed by `refresh_dirty` (the
            // streaming "no hills" regression).
            if grid.chunks.is_empty() && !is_streaming {
                continue;
            }
            let chunk_idxs: Vec<[i32; 3]> = grid.chunks.keys().map(|i| [i.x, i.y, i.z]).collect();
            // Empty streaming grid → placeholder bbox; the modular pool
            // ignores the bbox for slot assignment anyway.
            let (origin_chunk, chunks_dims) =
                roxlap_gpu::bounding_box_of(chunk_idxs.iter().copied())
                    .unwrap_or(([0, 0, 0], [1, 1, 1]));
            let chunks: Vec<([i32; 3], roxlap_gpu::ChunkUpload)> = grid
                .chunks
                .iter()
                .map(|(idx, vxl)| ([idx.x, idx.y, idx.z], roxlap_gpu::decompress_chunk(vxl)))
                .collect();
            total_chunks += chunks.len();
            // Streaming grids get a generous modular pool so chunks
            // arriving at new indices never collide; static grids fit
            // their bbox exactly.
            let pool_dims = if is_streaming {
                [8, 8, 4]
            } else {
                roxlap_gpu::GridUpload::default_pool_dims(chunks_dims)
            };
            scene_grids.push(roxlap_gpu::GridUpload {
                vsid: roxlap_scene::CHUNK_SIZE_XY,
                origin_chunk,
                chunks_dims,
                pool_dims,
                chunks,
            });
            grid_ids.push(gid);
        }

        if scene_grids.is_empty() {
            // No grids yet (e.g. streaming hasn't materialised the
            // first chunk) — leave `resident` None; render clears.
            return;
        }

        let info = roxlap_gpu::SceneUpload { grids: scene_grids };
        let resident = GpuSceneResident::upload(self.gpu.device(), &info);
        eprintln!(
            "roxlap-render: uploaded scene — {} grids, {total_chunks} chunks, {:.1} MiB resident",
            grid_ids.len(),
            resident.resident_bytes() as f64 / (1024.0 * 1024.0),
        );

        // Seed dirty trackers with each chunk's current version.
        let mut versions: Vec<HashMap<IVec3, u64>> = Vec::with_capacity(grid_ids.len());
        for gid in &grid_ids {
            let mut gv: HashMap<IVec3, u64> = HashMap::new();
            if let Some(grid) = scene.grid(*gid) {
                for c in grid.chunks.keys() {
                    gv.insert(*c, grid.chunk_version(*c));
                }
            }
            versions.push(gv);
        }

        self.resident = Some(resident);
        self.grid_ids = grid_ids;
        self.versions = versions;
    }

    /// Re-upload any chunk whose `chunk_version` bumped since last
    /// frame; evict chunks the streamer dropped. Moved verbatim from
    /// the scene-demo's `refresh_dirty_chunks`.
    fn refresh_dirty(&mut self, scene: &Scene) {
        let Some(resident) = self.resident.as_mut() else {
            return;
        };
        let queue = self.gpu.queue();
        let mut decompressed = 0u32;
        let mut evicted = 0u32;
        for (scene_idx, gid) in self.grid_ids.iter().enumerate() {
            let Some(grid) = scene.grid(*gid) else {
                continue;
            };
            let tracker = &mut self.versions[scene_idx];

            // Install / refresh current chunks, up to the per-frame
            // budget — the rest stay dirty and ride the next frames, so a
            // big streamed-in batch spreads its upload cost instead of
            // freezing one frame.
            for (chunk_ivec3, vxl) in &grid.chunks {
                let cur = grid.chunk_version(*chunk_ivec3);
                if tracker.get(chunk_ivec3).copied() == Some(cur) {
                    continue;
                }
                if decompressed >= self.chunk_upload_budget {
                    break;
                }
                let upload = roxlap_gpu::decompress_chunk(vxl);
                let outcome = resident.refresh_chunk(
                    queue,
                    scene_idx,
                    [chunk_ivec3.x, chunk_ivec3.y, chunk_ivec3.z],
                    &upload,
                );
                if outcome != roxlap_gpu::RefreshOutcome::ChunkOutOfBbox {
                    tracker.insert(*chunk_ivec3, cur);
                    decompressed += 1;
                }
            }

            // Evict chunks dropped since last frame.
            let stale: Vec<IVec3> = tracker
                .keys()
                .filter(|i| !grid.chunks.contains_key(*i))
                .copied()
                .collect();
            for c in stale {
                resident.evict_chunk(queue, scene_idx, [c.x, c.y, c.z]);
                tracker.remove(&c);
                evicted += 1;
            }
        }
        if decompressed > 8 || evicted > 0 {
            eprintln!("roxlap-render: refreshed {decompressed} chunks, evicted {evicted}");
        }
    }

    /// DL — translate the per-frame world-space [`LightRig`] into each
    /// grid's local frame (sun direction as a vector, point positions as
    /// points — mirroring [`grid_local_camera`]) and upload it. `None`
    /// clears all lights (the pre-DL render). Iterates `self.grid_ids` in
    /// the same order as [`Self::grid_cameras`], so per-grid light rows
    /// line up with the per-grid cameras `render_scene` marches with.
    fn sync_lights(&mut self, scene: &Scene, frame: &FrameParams) {
        let Some(rig) = frame.lights.as_ref() else {
            self.gpu
                .set_scene_lights(roxlap_gpu::SceneLights::default());
            return;
        };
        let mut lights = roxlap_gpu::SceneLights {
            // A rig is present ⇒ take the lit path (so `ambient` applies even
            // with no sun/points). `None` keeps the default `enabled: false`.
            enabled: true,
            sun_color: rig.sun.map_or([0.0; 3], |s| s.color),
            sun_intensity: rig.sun.map_or(0.0, |s| s.intensity),
            sun_casts_shadow: rig.sun.is_some_and(|s| s.casts_shadow),
            ambient: rig.ambient,
            shadow_strength: rig.shadow_strength,
            shadow_bias: rig.shadow_bias_voxels,
            shadow_max_dist: rig.shadow_max_dist,
            // Shadow-ray step budget (consumed in DL.3); a sane default.
            shadow_max_steps: 256,
            // DL.6 — stylized lighting (cel banding + gradient-map ramp).
            style_bands: rig.bands,
            shadow_tint: rig.shadow_tint,
            ..Default::default()
        };
        for gid in &self.grid_ids {
            let (rotation, origin) = scene
                .grid(*gid)
                .map_or((glam::DQuat::IDENTITY, DVec3::ZERO), |g| {
                    (g.transform.rotation, g.transform.origin)
                });
            if let Some(sun) = rig.sun {
                lights
                    .grid_sun_dirs
                    .push(grid_local_sun_dir(rotation, sun.direction));
            }
            let mut pts = Vec::with_capacity(rig.points.len());
            for p in rig.points {
                pts.push(roxlap_gpu::GpuLight {
                    position: grid_local_point(rotation, origin, p.position),
                    radius: p.radius,
                    color: p.color,
                    intensity: p.intensity,
                    casts_shadow: p.casts_shadow,
                });
            }
            lights.grid_point_lights.push(pts);
        }
        // DL.4 — world-space copies for the sprite pass (sprites render in
        // world space, not grid-local). Sun dir = normalized −travel.
        if let Some(sun) = rig.sun {
            let to_sun = (-DVec3::from_array(sun.direction.map(f64::from))).normalize_or_zero();
            lights.world_sun_dir = [to_sun.x as f32, to_sun.y as f32, to_sun.z as f32];
        }
        lights.world_points = rig
            .points
            .iter()
            .map(|p| roxlap_gpu::GpuLight {
                position: p.position,
                radius: p.radius,
                color: p.color,
                intensity: p.intensity,
                casts_shadow: p.casts_shadow,
            })
            .collect();
        self.gpu.set_scene_lights(lights);
    }

    /// One per-grid [`roxlap_gpu::Camera`]: the world camera
    /// transformed into each grid's local frame via the inverse
    /// `GridTransform`. Moved from the scene-demo's `redraw_gpu`.
    fn grid_cameras(&self, scene: &Scene, camera: &Camera) -> Vec<roxlap_gpu::Camera> {
        let mut cameras = Vec::with_capacity(self.grid_ids.len());
        for gid in &self.grid_ids {
            let Some(grid) = scene.grid(*gid) else {
                cameras.push(roxlap_gpu::Camera::default());
                continue;
            };
            cameras.push(grid_local_camera(
                grid.transform.rotation,
                grid.transform.origin,
                camera,
            ));
        }
        cameras
    }

    /// XS.3 — per-grid world transforms (parallel to [`Self::grid_cameras`]):
    /// world origin + the local→world rotation columns, for cross-grid shadows.
    #[allow(clippy::cast_possible_truncation)]
    fn grid_world_transforms(&self, scene: &Scene) -> Vec<roxlap_gpu::GridWorldTransform> {
        let mut out = Vec::with_capacity(self.grid_ids.len());
        for gid in &self.grid_ids {
            let Some(grid) = scene.grid(*gid) else {
                out.push(roxlap_gpu::GridWorldTransform::default());
                continue;
            };
            let o = grid.transform.origin;
            let r = grid.transform.rotation;
            let col = |v: DVec3| {
                let w = r * v;
                [w.x as f32, w.y as f32, w.z as f32]
            };
            out.push(roxlap_gpu::GridWorldTransform {
                origin: [o.x as f32, o.y as f32, o.z as f32],
                rot_cols: [col(DVec3::X), col(DVec3::Y), col(DVec3::Z)],
            });
        }
        out
    }
}

/// Transform a world [`Camera`] into a grid's local frame: apply the
/// inverse grid rotation to the basis + the origin-relative position.
/// Rigid transforms preserve handedness, so `right × down == forward`
/// carries through — important, since a flipped basis silently culls
/// the whole grid (see the voxlap-basis-chirality note).
pub(crate) fn grid_local_camera(
    rotation: glam::DQuat,
    origin: DVec3,
    camera: &Camera,
) -> roxlap_gpu::Camera {
    let inv_rot = rotation.inverse();
    let local_pos = inv_rot * (DVec3::from_array(camera.pos) - origin);
    let local_right = inv_rot * DVec3::from_array(camera.right);
    let local_down = inv_rot * DVec3::from_array(camera.down);
    let local_forward = inv_rot * DVec3::from_array(camera.forward);
    roxlap_gpu::Camera {
        position: [local_pos.x as f32, local_pos.y as f32, local_pos.z as f32],
        right: [
            local_right.x as f32,
            local_right.y as f32,
            local_right.z as f32,
        ],
        down: [
            local_down.x as f32,
            local_down.y as f32,
            local_down.z as f32,
        ],
        forward: [
            local_forward.x as f32,
            local_forward.y as f32,
            local_forward.z as f32,
        ],
        // fov is passed to render_scene separately; the per-grid
        // Camera's fov field is unused by the marcher.
        fov_y_rad: 60_f32.to_radians(),
    }
}

/// DL — a world-space sun **travel** direction → the unit direction **to**
/// the sun in a grid's local frame. A direction is a vector, so only the
/// inverse rotation applies (no translation); the negation flips travel →
/// toward-light for the shader's N·L. Handedness is preserved (rigid
/// rotation), so the sign convention stays consistent under rotation — the
/// chirality footgun the per-grid cameras also have to respect.
pub(crate) fn grid_local_sun_dir(rotation: glam::DQuat, travel_world: [f32; 3]) -> [f32; 3] {
    let to_sun = (rotation.inverse() * (-DVec3::from_array(travel_world.map(f64::from))))
        .normalize_or_zero();
    [to_sun.x as f32, to_sun.y as f32, to_sun.z as f32]
}

/// DL — a world-space point-light position → its position in a grid's
/// local frame (inverse rotation + origin-relative translation, exactly
/// like [`grid_local_camera`]'s position).
pub(crate) fn grid_local_point(
    rotation: glam::DQuat,
    origin: DVec3,
    pos_world: [f32; 3],
) -> [f32; 3] {
    let local = rotation.inverse() * (DVec3::from_array(pos_world.map(f64::from)) - origin);
    [local.x as f32, local.y as f32, local.z as f32]
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // exact pass-through values are intended
mod tests {
    use super::*;

    fn world_cam() -> Camera {
        Camera {
            pos: [10.0, 20.0, 30.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 0.0, 1.0],
            forward: [0.0, 1.0, 0.0],
        }
    }

    /// Sign of the basis triple product `(right × down) · forward` —
    /// the handedness a flipped transform would silently invert
    /// (→ whole-grid cull, per the voxlap-basis-chirality note).
    fn handedness(r: [f32; 3], d: [f32; 3], f: [f32; 3]) -> f32 {
        glam::Vec3::from_array(r)
            .cross(glam::Vec3::from_array(d))
            .dot(glam::Vec3::from_array(f))
            .signum()
    }

    #[test]
    fn identity_transform_is_pass_through() {
        let c = grid_local_camera(glam::DQuat::IDENTITY, DVec3::ZERO, &world_cam());
        assert_eq!(c.position, [10.0, 20.0, 30.0]);
        assert_eq!(c.right, [1.0, 0.0, 0.0]);
        assert_eq!(c.down, [0.0, 0.0, 1.0]);
        assert_eq!(c.forward, [0.0, 1.0, 0.0]);
    }

    #[test]
    fn origin_offset_shifts_position_only() {
        let c = grid_local_camera(
            glam::DQuat::IDENTITY,
            DVec3::new(10.0, 20.0, 30.0),
            &world_cam(),
        );
        assert_eq!(c.position, [0.0, 0.0, 0.0]);
        assert_eq!(c.forward, [0.0, 1.0, 0.0], "basis unaffected by origin");
    }

    #[test]
    fn rotation_preserves_basis_handedness() {
        // A proper rotation must NOT flip handedness — a flipped local
        // basis silently culls the whole grid in the marcher.
        let cam = world_cam();
        let world_h = handedness(
            [
                cam.right[0] as f32,
                cam.right[1] as f32,
                cam.right[2] as f32,
            ],
            [cam.down[0] as f32, cam.down[1] as f32, cam.down[2] as f32],
            [
                cam.forward[0] as f32,
                cam.forward[1] as f32,
                cam.forward[2] as f32,
            ],
        );
        let rot = glam::DQuat::from_euler(glam::EulerRot::XYZ, 0.5, -0.8, 0.3);
        let c = grid_local_camera(rot, DVec3::new(1.0, 2.0, 3.0), &cam);
        assert_eq!(
            handedness(c.right, c.down, c.forward),
            world_h,
            "grid-local transform flipped the basis handedness",
        );
    }

    // ───────────────────────── DL — light transforms ─────────────────────────

    fn approx(a: [f32; 3], b: [f32; 3]) {
        for i in 0..3 {
            assert!((a[i] - b[i]).abs() < 1e-5, "{a:?} != {b:?}");
        }
    }

    #[test]
    fn sun_dir_negates_travel_and_normalizes() {
        // Identity grid: a sun travelling +z (straight down, voxlap z-down)
        // → direction TO the sun is -z (straight up), unit length.
        let d = grid_local_sun_dir(glam::DQuat::IDENTITY, [0.0, 0.0, 5.0]);
        approx(d, [0.0, 0.0, -1.0]);
    }

    #[test]
    fn sun_dir_follows_inverse_grid_rotation() {
        // Grid yawed +90° about z: a world sun travelling +x maps to a
        // to-sun direction of -x in world, then into grid-local via the
        // inverse rotation. Cross-check against the raw quat math.
        let rot = glam::DQuat::from_rotation_z(std::f64::consts::FRAC_PI_2);
        let got = grid_local_sun_dir(rot, [3.0, 0.0, 0.0]);
        let want = rot.inverse() * DVec3::new(-1.0, 0.0, 0.0); // -travel, normalized
        approx(got, [want.x as f32, want.y as f32, want.z as f32]);
        // Still unit length after the rotation.
        let len = (got[0] * got[0] + got[1] * got[1] + got[2] * got[2]).sqrt();
        assert!((len - 1.0).abs() < 1e-5, "sun dir not unit: {len}");
    }

    #[test]
    fn point_pos_is_translation_plus_inverse_rotation() {
        // Identity rotation: only the origin-relative shift applies.
        let p = grid_local_point(
            glam::DQuat::IDENTITY,
            DVec3::new(10.0, 20.0, 30.0),
            [12.0, 24.0, 33.0],
        );
        approx(p, [2.0, 4.0, 3.0]);
        // A light sitting on the grid origin maps to the local origin
        // regardless of rotation.
        let rot = glam::DQuat::from_euler(glam::EulerRot::XYZ, 0.4, -0.7, 0.2);
        let origin = DVec3::new(5.0, -6.0, 7.0);
        let at_origin = grid_local_point(rot, origin, [5.0, -6.0, 7.0]);
        approx(at_origin, [0.0, 0.0, 0.0]);
    }
}
