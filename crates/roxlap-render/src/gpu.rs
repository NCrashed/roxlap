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

use glam::{DVec3, IVec3};
use roxlap_core::Camera;
use roxlap_gpu::{
    build_sprite_model, GpuInitError, GpuRenderer, GpuSceneResident, SpriteInstance,
    SpriteInstanceTransform, SpriteModelRegistry,
};
use roxlap_scene::{GridId, Scene};
use winit::window::Window;

use crate::{FrameParams, RenderOptions, SpriteSet};

pub(crate) struct GpuBackend {
    gpu: GpuRenderer,
    /// Whole-scene residency; `None` until the first non-empty render.
    resident: Option<GpuSceneResident>,
    /// Grid ids in upload order — index = per-grid camera slot.
    grid_ids: Vec<GridId>,
    /// Per-grid `chunk_idx → last-uploaded version` for the dirty poll.
    versions: Vec<HashMap<IVec3, u64>>,
    /// Instanced sprite registry + the uploaded instance list; `None`
    /// until [`set_sprites`](Self::set_sprites).
    sprite_registry: Option<SpriteModelRegistry>,
    sprite_instances: Vec<SpriteInstance>,
    /// Registry model id the `G`-carve edits + its next z-layer.
    carve_model_id: Option<u32>,
    carve_z: u32,
}

impl GpuBackend {
    pub(crate) fn new(
        window: std::sync::Arc<Window>,
        opts: &RenderOptions,
    ) -> Result<Self, GpuInitError> {
        let gpu = GpuRenderer::new_blocking(window, opts.gpu)?;
        Ok(Self {
            gpu,
            resident: None,
            grid_ids: Vec::new(),
            versions: Vec::new(),
            sprite_registry: None,
            sprite_instances: Vec::new(),
            carve_model_id: None,
            carve_z: 0,
        })
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
            });
        }
        self.gpu.set_sprite_instances(&registry, &instances);
        self.carve_model_id = set.carve_model.and_then(|i| model_ids.get(i).copied());
        self.carve_z = 0;
        self.sprite_registry = Some(registry);
        self.sprite_instances = instances;
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
        self.gpu.set_sprite_instances(reg, &self.sprite_instances);
        removed
    }

    pub(crate) fn adapter_info(&self) -> &str {
        self.gpu.adapter_info()
    }

    pub(crate) fn resize(&mut self, width: u32, height: u32) {
        self.gpu.resize(width, height);
    }

    /// Upload a sky panorama for the GPU shader's sky sampling.
    pub(crate) fn set_sky_panorama(&mut self, rgba: &[u8], w: u32, h: u32) {
        self.gpu.set_sky_panorama(rgba, w, h);
    }

    pub(crate) fn render(&mut self, scene: &mut Scene, camera: &Camera, frame: &FrameParams) {
        if self.resident.is_none() {
            self.upload_scene(scene);
        } else {
            self.refresh_dirty(scene);
        }

        // Per-frame GPU scene-LOD knob (GPU.11.1).
        self.gpu.set_scene_mip_scan_dist(frame.gpu_mip_scan_dist);

        let cameras = self.grid_cameras(scene, camera);
        if let Some(resident) = &self.resident {
            self.gpu.render_scene(
                resident,
                &cameras,
                frame.gpu_fov_y_rad,
                frame.gpu_max_outer_steps,
            );
        } else {
            // No materialised grids yet — clear to colour.
            self.gpu.render();
        }
    }

    /// Decompress every materialised chunk of every grid and upload as
    /// one [`GpuSceneResident`]; record the grid order + seed the
    /// dirty-version trackers. Moved verbatim from the scene-demo's
    /// `upload_first_scene` (minus the streaming pump, which the host
    /// drives before calling render).
    fn upload_scene(&mut self, scene: &Scene) {
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
            if scene_grids.len() == roxlap_gpu::MAX_SCENE_GRIDS as usize {
                eprintln!(
                    "roxlap-render: scene cap ({} grids) reached — skipping grid {}+",
                    roxlap_gpu::MAX_SCENE_GRIDS,
                    gid.raw(),
                );
                break;
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

            // Install / refresh current chunks.
            for (chunk_ivec3, vxl) in &grid.chunks {
                let cur = grid.chunk_version(*chunk_ivec3);
                if tracker.get(chunk_ivec3).copied() == Some(cur) {
                    continue;
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
}
