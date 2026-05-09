//! Scene-level rendering — drives [`roxlap_core::opticast::opticast`]
//! across the grids of a [`Scene`].
//!
//! S3.0 (this module's first cut) handles **axis-aligned,
//! single-chunk grids only**. For each grid, the world camera is
//! translated by `-grid.transform.origin` (rotation-free) and the
//! resulting grid-local pose is fed straight into the existing
//! per-chunk opticast renderer over `grid.chunks[(0, 0, 0)]`.
//!
//! Multi-grid composition (z-buffer merge of overlapping output)
//! is **not yet correct** at S3.0 — opticast writes pixels and z
//! values unconditionally, so a second grid's render overwrites
//! the first. S3.1 adds per-grid temporary buffers + z-merge.
//! Single-grid scenes match a direct opticast call byte-for-byte
//! (the round-trip test below pins this).
//!
//! Higher-stage upgrades that go through this entry point:
//! - **S3.1** — multi-grid z-composition.
//! - **S4** — cross-chunk gline (replaces the
//!   `chunks.get(&IVec3::ZERO)` lookup with a 3D DDA across the
//!   sparse chunk map).
//! - **S5** — full quaternion grid rotation (the camera transform
//!   adds `rotation.inverse() *` after the translate).
//! - **S6** — per-grid LOD selection (Near voxel raycast / Mid
//!   coarse mip / Far billboard).

use glam::IVec3;
use roxlap_core::opticast::{opticast, OpticastOutcome, OpticastSettings};
use roxlap_core::rasterizer::{Rasterizer, ScratchPool};
use roxlap_core::Camera;

use crate::Scene;

/// Outcome of a [`render_scene`] call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderOutcome {
    /// At least one grid produced a render.
    Rendered {
        /// Number of grids whose opticast pass returned
        /// [`OpticastOutcome::Rendered`].
        grids_drawn: usize,
    },
    /// No grid rendered. Either the scene was empty, all grids had
    /// no chunk at `(0, 0, 0)` (the only address S3.0 inspects),
    /// or every per-grid opticast call returned
    /// [`OpticastOutcome::SkippedCameraInSolid`].
    Empty,
}

/// Render every grid in `scene` into the rasterizer's framebuffer.
///
/// Each grid's render uses a grid-local camera pose obtained by
/// translating the world camera by `-grid.transform.origin`. S3.0
/// ignores rotation (axis-aligned grids only) and only the chunk
/// at `(0, 0, 0)` (single-chunk grids only) — non-zero chunks are
/// skipped silently.
///
/// Caller responsibilities (same as a direct
/// [`roxlap_core::opticast::opticast`] call):
/// - Pre-fill the framebuffer with the desired sky colour.
/// - Pre-fill the zbuffer (commonly to `0.0` matching the
///   per-chunk renderer's convention).
/// - Configure `pool` with skycast / fog colours.
/// - Build the rasterizer over the framebuffer + zbuffer.
///
/// **Caveat (S3.0)**: when the scene has > 1 grid, output is
/// last-grid-wins because opticast writes unconditionally.
/// S3.1 will land per-grid temporary buffers + z-buffer
/// composition.
pub fn render_scene<R: Rasterizer + Clone + Send + Sync>(
    rasterizer: &mut R,
    pool: &mut ScratchPool,
    scene: &Scene,
    camera: &Camera,
    settings: &OpticastSettings,
) -> RenderOutcome {
    let mut grids_drawn = 0usize;
    for (_id, grid) in scene.grids() {
        let Some(chunk) = grid.chunks.get(&IVec3::ZERO) else {
            // S3.0: only chunk (0,0,0) is rendered. Cross-chunk
            // dispatch lands in S4.
            continue;
        };
        let local_cam = Camera {
            pos: [
                camera.pos[0] - grid.transform.origin.x,
                camera.pos[1] - grid.transform.origin.y,
                camera.pos[2] - grid.transform.origin.z,
            ],
            right: camera.right,
            down: camera.down,
            forward: camera.forward,
        };
        let outcome = opticast(
            rasterizer,
            pool,
            &local_cam,
            settings,
            chunk.vsid,
            &chunk.data,
            &chunk.column_offset,
        );
        if outcome == OpticastOutcome::Rendered {
            grids_drawn += 1;
        }
    }
    if grids_drawn == 0 {
        RenderOutcome::Empty
    } else {
        RenderOutcome::Rendered { grids_drawn }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GridTransform, Scene, CHUNK_SIZE_XY};
    use glam::{DVec3, IVec3};
    use roxlap_core::opticast::{opticast as core_opticast, OpticastSettings};
    use roxlap_core::rasterizer::ScratchPool;
    use roxlap_core::scalar_rasterizer::ScalarRasterizer;
    use roxlap_core::{Camera, Engine};

    const XRES: u32 = 320;
    const YRES: u32 = 200;

    /// Build a single-grid scene at the given world origin with a
    /// recognisable shape inside its chunk (0, 0, 0): a 16-voxel
    /// box plus a 6-radius sphere. Returns `(scene, grid_id)`.
    fn build_one_grid_scene(world_origin: DVec3) -> (Scene, crate::GridId) {
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::at(world_origin));
        let grid = scene.grid_mut(id).unwrap();
        // Box covering [40..56]³ in chunk-local coords.
        grid.set_rect(
            IVec3::new(40, 40, 40),
            IVec3::new(55, 55, 55),
            Some(0x80_88_88_88),
        );
        // Sphere at (80, 80, 80) radius 6.
        grid.set_sphere(IVec3::new(80, 80, 80), 6, Some(0x80_22_aa_22));
        (scene, id)
    }

    fn camera_at(pos: [f64; 3]) -> Camera {
        // Look +y axis; voxlap z-down convention. Right-handed:
        // right × down == forward.
        Camera {
            pos,
            right: [-1.0, 0.0, 0.0],
            down: [0.0, 0.0, 1.0],
            forward: [0.0, 1.0, 0.0],
        }
    }

    /// Spin up an engine + `ScratchPool` + framebuffers ready for
    /// one `opticast` / `render_scene` pass. Mirrors the
    /// boilerplate the `outside_*_repro` tests use.
    fn render_setup() -> (Engine, ScratchPool, Vec<u32>, Vec<f32>) {
        let engine = Engine::new();
        let mut pool = ScratchPool::new(XRES, YRES, CHUNK_SIZE_XY);
        let sky = engine.sky_color();
        let sky_col_i = i32::from_ne_bytes(sky.to_ne_bytes());
        pool.set_skycast(sky_col_i, 0);
        let fog_col_i = i32::from_ne_bytes(engine.fog_color().to_ne_bytes());
        pool.set_fog(fog_col_i, engine.fog_max_scan_dist());
        pool.set_treat_z_max_as_air(true);
        let pixel_count = (XRES as usize) * (YRES as usize);
        let framebuffer = vec![sky; pixel_count];
        let zbuffer = vec![0.0f32; pixel_count];
        (engine, pool, framebuffer, zbuffer)
    }

    /// Render `scene` via [`render_scene`] and return the resulting
    /// framebuffer.
    fn render_via_scene(scene: &Scene, camera: &Camera) -> Vec<u32> {
        let (_engine, mut pool, mut fb, mut zb) = render_setup();
        let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
        let chunk_vsid = CHUNK_SIZE_XY;
        // Need a chunk to seed the rasterizer with — borrow grid 0's
        // chunk (0,0,0) since render_scene reads through it anyway.
        let grid = scene.grids().next().expect("test scene has a grid").1;
        let chunk = grid.chunk(IVec3::ZERO).expect("chunk (0,0,0) present");
        let mut rasterizer = ScalarRasterizer::new(
            &mut fb,
            &mut zb,
            XRES as usize,
            &chunk.data,
            &chunk.column_offset,
            &chunk.mip_base_offsets,
            chunk_vsid,
        );
        let outcome = render_scene(&mut rasterizer, &mut pool, scene, camera, &settings);
        drop(rasterizer);
        assert_eq!(outcome, RenderOutcome::Rendered { grids_drawn: 1 });
        fb
    }

    /// Render the same chunk as a direct opticast call, with the
    /// camera already in grid-local frame. The reference output
    /// for the round-trip test.
    fn render_via_direct_opticast(scene: &Scene, local_camera: &Camera) -> Vec<u32> {
        let (_engine, mut pool, mut fb, mut zb) = render_setup();
        let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
        let grid = scene.grids().next().unwrap().1;
        let chunk = grid.chunk(IVec3::ZERO).unwrap();
        let mut rasterizer = ScalarRasterizer::new(
            &mut fb,
            &mut zb,
            XRES as usize,
            &chunk.data,
            &chunk.column_offset,
            &chunk.mip_base_offsets,
            chunk.vsid,
        );
        let _ = core_opticast(
            &mut rasterizer,
            &mut pool,
            local_camera,
            &settings,
            chunk.vsid,
            &chunk.data,
            &chunk.column_offset,
        );
        drop(rasterizer);
        fb
    }

    #[test]
    fn render_scene_at_origin_matches_direct_opticast() {
        // Grid at world origin → grid-local camera == world camera.
        let (scene, _) = build_one_grid_scene(DVec3::ZERO);
        let cam = camera_at([64.0, 0.0, 64.0]);
        let via_scene = render_via_scene(&scene, &cam);
        let via_direct = render_via_direct_opticast(&scene, &cam);
        assert_eq!(
            via_scene, via_direct,
            "render_scene with single grid at origin should match direct opticast"
        );
    }

    #[test]
    fn render_scene_translated_grid_matches_grid_local_opticast() {
        // Grid at world (1000, 2000, 3000). World camera at
        // (1064, 2000, 3064) — grid-local (64, 0, 64). render_scene
        // should produce the same output as a direct opticast call
        // with grid-local camera.
        let world_origin = DVec3::new(1000.0, 2000.0, 3000.0);
        let (scene, _) = build_one_grid_scene(world_origin);
        let world_cam = camera_at([1064.0, 2000.0, 3064.0]);
        let local_cam = camera_at([64.0, 0.0, 64.0]);
        let via_scene = render_via_scene(&scene, &world_cam);
        let via_direct = render_via_direct_opticast(&scene, &local_cam);
        assert_eq!(
            via_scene, via_direct,
            "render_scene of translated grid should match opticast with grid-local camera"
        );
    }

    #[test]
    fn empty_scene_returns_empty_outcome() {
        let scene = Scene::new();
        let (_engine, mut pool, mut fb, mut zb) = render_setup();
        let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
        // No chunk to seed with — use a placeholder of zeroed bytes.
        let placeholder_data = [0u8; 8];
        let placeholder_offsets = [0u32, 8];
        let placeholder_mip_offsets = [0usize, 2];
        let mut rasterizer = ScalarRasterizer::new(
            &mut fb,
            &mut zb,
            XRES as usize,
            &placeholder_data,
            &placeholder_offsets,
            &placeholder_mip_offsets,
            1,
        );
        let outcome = render_scene(
            &mut rasterizer,
            &mut pool,
            &scene,
            &camera_at([0.0, 0.0, 0.0]),
            &settings,
        );
        assert_eq!(outcome, RenderOutcome::Empty);
    }

    #[test]
    fn scene_with_grid_but_no_chunk_zero_returns_empty() {
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::identity());
        // Add a chunk at (1, 0, 0) but NOT at (0, 0, 0).
        scene
            .grid_mut(id)
            .unwrap()
            .ensure_chunk(IVec3::new(1, 0, 0));
        let (_engine, mut pool, mut fb, mut zb) = render_setup();
        let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
        let chunk = scene.grid(id).unwrap().chunk(IVec3::new(1, 0, 0)).unwrap();
        let mut rasterizer = ScalarRasterizer::new(
            &mut fb,
            &mut zb,
            XRES as usize,
            &chunk.data,
            &chunk.column_offset,
            &chunk.mip_base_offsets,
            chunk.vsid,
        );
        let outcome = render_scene(
            &mut rasterizer,
            &mut pool,
            &scene,
            &camera_at([64.0, 0.0, 64.0]),
            &settings,
        );
        // S3.0 only inspects (0,0,0); (1,0,0) is silently skipped.
        assert_eq!(outcome, RenderOutcome::Empty);
    }
}
