//! Scene-level rendering — drives [`roxlap_core::opticast::opticast`]
//! across the grids of a [`Scene`].
//!
//! Two entry points:
//!
//! - [`render_scene_composed`] (recommended for multi-grid scenes):
//!   per grid, allocates a temporary framebuffer + zbuffer, runs
//!   opticast into the temp, then merges into the shared output via
//!   per-pixel min-z. Correctly composites overlapping grid output.
//! - [`render_scene`] (single-grid trusting caller): writes every
//!   grid directly into the shared rasterizer. For single-grid
//!   scenes this matches a direct opticast call byte-for-byte; for
//!   multi-grid it's last-grid-wins (sky writes from grid B
//!   overwrite grid A's hits). Useful for tests / single-grid
//!   sanity checks.
//!
//! ## S4B.2.e: Approach B multi-chunk dispatch
//!
//! Both APIs route per-grid rendering through
//! [`crate::Grid::chunk_xy_backing`] → [`roxlap_core::ChunkGrid`] →
//! [`roxlap_core::GridView::from_chunk_grid`] → [`opticast`].
//! `opticast`'s prelude looks up the camera's chunk via
//! [`roxlap_core::GridView::chunk_at_xy`]; the grouscan column-step
//! swaps the active per-chunk `(slab_buf, column_offsets)` when
//! rays cross a chunk-XY boundary. The combined-world stitch
//! (Approach C, S4.0..S4.2) is no longer in the render path — the
//! lighting bake still uses it until S4B.4 lands a per-chunk bake.
//!
//! Per-grid rotation (S5) and per-grid LOD (S6) plug in at the
//! same dispatch point: rotate the world camera into grid-local
//! before the chunk-grid lookup, then dispatch coarse / fine /
//! billboard based on grid-camera distance.

// `fb` / `zb` (framebuffer / zbuffer) and the `_fb` / `_zb` suffixes
// throughout this module are voxlap-canonical pairs — drilling them
// apart with longer names just hurts readability.
#![allow(clippy::similar_names)]

use roxlap_core::opticast::{opticast, OpticastOutcome, OpticastSettings};
use roxlap_core::rasterizer::ScratchPool;
use roxlap_core::scalar_rasterizer::ScalarRasterizer;
use roxlap_core::sky::Sky;
use roxlap_core::Camera;

use crate::{Scene, CHUNK_SIZE_XY};

/// Outcome of a [`render_scene`] / [`render_scene_composed`] call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderOutcome {
    /// At least one grid produced a render.
    Rendered {
        /// Number of grids whose opticast pass returned
        /// [`OpticastOutcome::Rendered`].
        grids_drawn: usize,
    },
    /// No grid rendered. Either the scene was empty or every
    /// per-grid opticast call returned
    /// [`OpticastOutcome::SkippedCameraInSolid`].
    Empty,
}

/// Render every grid in `scene` directly into `(fb, zb)` — no
/// per-grid temp buffer, no compose merge. For multi-grid scenes
/// this is last-grid-wins (later grids' opticast writes overwrite
/// earlier grids' pixels indiscriminately, including sky), so it's
/// only correct for single-grid scenes.
///
/// Use this when you have one grid and want the byte-stable
/// matches-direct-opticast property — the test suite uses it as a
/// sanity check that the combined-world stitch + render harness
/// doesn't drift vs. a raw [`opticast`] call.
///
/// Caller pre-fills `fb` with the desired sky colour and `zb` with
/// any value (typically `0.0` matching the per-chunk renderer's
/// convention or `f32::INFINITY` for compose-friendly init); the
/// rasterizer overwrites both per pixel that gets a hit.
#[allow(clippy::too_many_arguments)]
pub fn render_scene(
    fb: &mut [u32],
    zb: &mut [f32],
    pitch_pixels: usize,
    width: u32,
    height: u32,
    pool: &mut ScratchPool,
    scene: &mut Scene,
    camera: &Camera,
    settings: &OpticastSettings,
    sky: Option<&Sky>,
) -> RenderOutcome {
    debug_assert_eq!(fb.len(), zb.len());
    let pixel_count = (width as usize) * (height as usize);
    debug_assert_eq!(fb.len(), pixel_count);

    let mut grids_drawn = 0usize;
    for (_id, grid) in scene.grids_mut() {
        // S4B.2.e: Approach B render path. World → grid-local
        // camera transform doesn't need a voxel-offset adjustment
        // anymore — Approach B's chunks live at their signed
        // (chx, chy) indices and `chunk_at_xy` handles negative-
        // index lookups natively.
        let grid_origin = grid.transform.origin;
        let Some(backing) = grid.chunk_xyz_backing() else {
            // Empty grid (no populated chz=0 chunks) — skip.
            continue;
        };
        let local_cam = Camera {
            pos: [
                camera.pos[0] - grid_origin.x,
                camera.pos[1] - grid_origin.y,
                camera.pos[2] - grid_origin.z,
            ],
            right: camera.right,
            down: camera.down,
            forward: camera.forward,
        };
        let cg = roxlap_core::ChunkGrid {
            chunks: &backing.chunks,
            origin_chunk_xy: backing.origin_chunk_xy,
            origin_chunk_z: backing.origin_chunk_z,
            chunks_x: backing.chunks_x,
            chunks_y: backing.chunks_y,
            chunks_z: backing.chunks_z,
        };
        let grid_view = roxlap_core::GridView::from_chunk_grid(&cg, CHUNK_SIZE_XY);
        let outcome = {
            let mut rasterizer = ScalarRasterizer::new(fb, zb, pitch_pixels, grid_view);
            if let Some(sky_ref) = sky {
                rasterizer = rasterizer.with_sky(sky_ref);
            }
            opticast(&mut rasterizer, pool, &local_cam, settings, grid_view)
        };
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

/// Per-pixel "min-z wins" merge of `(temp_fb, temp_zb)` into
/// `(shared_fb, shared_zb)`.
///
/// Voxlap's z-buffer convention: `z` = perpendicular distance from
/// camera; **smaller `z` = closer to camera**. This helper picks
/// the closer pixel per slot. Sky pixels emerge with a large `z`
/// (`scratch.skycast.dist`, set to `gxmax` or `i32::MAX` per
/// `phase_startsky`) so they always lose to any hit's finite
/// distance.
///
/// `temp_fb` / `temp_zb` are read-only inputs; both must have the
/// same length as `shared_fb` / `shared_zb` (debug-asserted).
pub fn compose_into(
    shared_fb: &mut [u32],
    shared_zb: &mut [f32],
    temp_fb: &[u32],
    temp_zb: &[f32],
) {
    debug_assert_eq!(shared_fb.len(), shared_zb.len());
    debug_assert_eq!(shared_fb.len(), temp_fb.len());
    debug_assert_eq!(shared_fb.len(), temp_zb.len());
    for i in 0..shared_fb.len() {
        if temp_zb[i] < shared_zb[i] {
            shared_fb[i] = temp_fb[i];
            shared_zb[i] = temp_zb[i];
        }
    }
}

/// Render every grid in `scene` with per-grid temporary buffers +
/// z-buffer composition. The canonical multi-grid scene render
/// path.
///
/// Algorithm:
/// 1. Caller pre-fills `fb` with the desired sky colour and `zb`
///    with [`f32::INFINITY`] (so any rendered pixel wins the
///    initial composition).
/// 2. For each grid, allocate a temporary `(temp_fb, temp_zb)` of
///    the same size, pre-fill them with sky / `INFINITY`, and run
///    [`opticast`] into them via a [`ScalarRasterizer`] over the
///    temporary buffers AND the grid's combined-world view (S4.0).
/// 3. Merge the temporary buffers into the shared `(fb, zb)` via
///    [`compose_into`] — closer pixels (smaller `z`) win.
///
/// Pixel correctness across overlapping grids: sky pixels emerge
/// with `z` = `gxmax` / `i32::MAX` (a very large value), so they
/// always lose to any hit. Hits compete on actual perpendicular
/// distance — the closer grid's surface is what gets composited.
///
/// `pitch_pixels` is the framebuffer's row stride in pixels (×4 for
/// bytes). `width` × `height` must equal `fb.len()` /
/// `zb.len()`. `sky` is the optional textured sky resource the
/// rasterizer threads through to `phase_startsky`; `None` ⇒ solid
/// `pool.skycast` fill.
///
/// **Heap allocation per call:** two `Vec` allocations per grid (a
/// temp framebuffer and zbuffer). For repeated frame rendering an
/// owned scratch struct that pre-allocates these is the obvious
/// optimisation; deferred until profiling shows it matters.
#[allow(clippy::too_many_arguments)]
pub fn render_scene_composed(
    fb: &mut [u32],
    zb: &mut [f32],
    pitch_pixels: usize,
    width: u32,
    height: u32,
    pool: &mut ScratchPool,
    scene: &mut Scene,
    camera: &Camera,
    settings: &OpticastSettings,
    sky_color: u32,
    sky: Option<&Sky>,
) -> RenderOutcome {
    debug_assert_eq!(fb.len(), zb.len());
    let pixel_count = (width as usize) * (height as usize);
    debug_assert_eq!(fb.len(), pixel_count);

    let mut grids_drawn = 0usize;
    let mut temp_fb = vec![sky_color; pixel_count];
    let mut temp_zb = vec![f32::INFINITY; pixel_count];

    for (_id, grid) in scene.grids_mut() {
        // Reset temp to sky / INFINITY so each grid starts fresh.
        // The reset cost is O(pixels) per grid; for small grid counts
        // this is negligible vs the opticast work.
        temp_fb.fill(sky_color);
        temp_zb.fill(f32::INFINITY);

        // S4B.2.e: Approach B render path. See `render_scene`'s
        // body for the camera transform + ChunkGrid construction
        // commentary; the only difference is this writes to
        // (temp_fb, temp_zb) and composes via `compose_into`.
        let grid_origin = grid.transform.origin;
        let Some(backing) = grid.chunk_xyz_backing() else {
            continue;
        };
        let local_cam = Camera {
            pos: [
                camera.pos[0] - grid_origin.x,
                camera.pos[1] - grid_origin.y,
                camera.pos[2] - grid_origin.z,
            ],
            right: camera.right,
            down: camera.down,
            forward: camera.forward,
        };
        let cg = roxlap_core::ChunkGrid {
            chunks: &backing.chunks,
            origin_chunk_xy: backing.origin_chunk_xy,
            origin_chunk_z: backing.origin_chunk_z,
            chunks_x: backing.chunks_x,
            chunks_y: backing.chunks_y,
            chunks_z: backing.chunks_z,
        };
        let grid_view = roxlap_core::GridView::from_chunk_grid(&cg, CHUNK_SIZE_XY);

        let outcome = {
            let mut rasterizer =
                ScalarRasterizer::new(&mut temp_fb, &mut temp_zb, pitch_pixels, grid_view);
            if let Some(sky_ref) = sky {
                rasterizer = rasterizer.with_sky(sky_ref);
            }
            opticast(&mut rasterizer, pool, &local_cam, settings, grid_view)
        };
        if outcome == OpticastOutcome::Rendered {
            compose_into(fb, zb, &temp_fb, &temp_zb);
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
#[allow(clippy::float_cmp)]
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
    /// one `opticast` / `render_scene` pass. `pool_vsid` should
    /// cover the largest grid's combined vsid.
    fn render_setup(pool_vsid: u32) -> (Engine, ScratchPool, Vec<u32>, Vec<f32>) {
        let engine = Engine::new();
        let mut pool = ScratchPool::new(XRES, YRES, pool_vsid);
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

    /// Render `scene` via [`render_scene`] (single-grid no-compose
    /// path) and return the resulting framebuffer.
    fn render_via_scene(scene: &mut Scene, camera: &Camera) -> Vec<u32> {
        let (_engine, mut pool, mut fb, mut zb) = render_setup(CHUNK_SIZE_XY);
        let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
        let outcome = render_scene(
            &mut fb,
            &mut zb,
            XRES as usize,
            XRES,
            YRES,
            &mut pool,
            scene,
            camera,
            &settings,
            None,
        );
        assert_eq!(outcome, RenderOutcome::Rendered { grids_drawn: 1 });
        fb
    }

    /// Render the same chunk as a direct opticast call, with the
    /// camera already in grid-local frame. The reference output
    /// for the round-trip test.
    fn render_via_direct_opticast(scene: &Scene, local_camera: &Camera) -> Vec<u32> {
        let (_engine, mut pool, mut fb, mut zb) = render_setup(CHUNK_SIZE_XY);
        let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
        let grid = scene.grids().next().unwrap().1;
        let chunk = grid.chunk(IVec3::ZERO).unwrap();
        let grid_view = roxlap_core::GridView::from_single_vxl(chunk);
        let mut rasterizer = ScalarRasterizer::new(&mut fb, &mut zb, XRES as usize, grid_view);
        let _ = core_opticast(
            &mut rasterizer,
            &mut pool,
            local_camera,
            &settings,
            grid_view,
        );
        drop(rasterizer);
        fb
    }

    #[test]
    fn render_scene_at_origin_matches_direct_opticast() {
        // Grid at world origin → grid-local camera == world camera.
        // Single 1-chunk grid: combined view's bytes are byte-identical
        // to the chunk's own column data (slng-walk equivalence), so
        // render_scene must produce the same framebuffer as a direct
        // opticast on the chunk.
        let (mut scene, _) = build_one_grid_scene(DVec3::ZERO);
        let cam = camera_at([64.0, 0.0, 64.0]);
        let via_scene = render_via_scene(&mut scene, &cam);
        let via_direct = render_via_direct_opticast(&scene, &cam);
        assert_eq!(
            via_scene, via_direct,
            "render_scene with single 1-chunk grid at origin should match direct opticast"
        );
    }

    #[test]
    fn render_scene_translated_grid_matches_grid_local_opticast() {
        // Grid at world (1000, 2000, 3000). World camera at
        // (1064, 2000, 3064) — grid-local (64, 0, 64). render_scene
        // should produce the same output as a direct opticast call
        // with grid-local camera.
        let world_origin = DVec3::new(1000.0, 2000.0, 3000.0);
        let (mut scene, _) = build_one_grid_scene(world_origin);
        let world_cam = camera_at([1064.0, 2000.0, 3064.0]);
        let local_cam = camera_at([64.0, 0.0, 64.0]);
        let via_scene = render_via_scene(&mut scene, &world_cam);
        let via_direct = render_via_direct_opticast(&scene, &local_cam);
        assert_eq!(
            via_scene, via_direct,
            "render_scene of translated grid should match opticast with grid-local camera"
        );
    }

    #[test]
    fn empty_scene_returns_empty_outcome() {
        let mut scene = Scene::new();
        let (_engine, mut pool, mut fb, mut zb) = render_setup(CHUNK_SIZE_XY);
        let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
        let outcome = render_scene(
            &mut fb,
            &mut zb,
            XRES as usize,
            XRES,
            YRES,
            &mut pool,
            &mut scene,
            &camera_at([0.0, 0.0, 0.0]),
            &settings,
            None,
        );
        assert_eq!(outcome, RenderOutcome::Empty);
    }

    // ---- S3.1 / S4.0: render_scene_composed + 2-grid composition ----

    /// Build a 2-grid scene with two distinguishable boxes placed
    /// side-by-side in world space along the camera's right axis.
    /// Each grid holds one chunk (`(0, 0, 0)`) containing a single
    /// 16-voxel box with a uniquely-coloured surface so the
    /// composited framebuffer is partitionable by colour.
    fn build_two_grid_side_by_side() -> (Scene, u32, u32) {
        let mut scene = Scene::new();
        // Grid 0 at world (0, 200, 0): box centred chunk-local (64, 64, 100).
        let g0 = scene.add_grid(GridTransform::at(DVec3::new(0.0, 200.0, 0.0)));
        scene.grid_mut(g0).unwrap().set_rect(
            IVec3::new(56, 56, 92),
            IVec3::new(71, 71, 107),
            Some(0x80_88_22_22), // dark red
        );
        // Grid 1 at world (200, 200, 0): box centred chunk-local (64, 64, 100).
        let _g1 = scene.add_grid(GridTransform::at(DVec3::new(200.0, 200.0, 0.0)));
        // Borrow-checker dance: re-borrow grid 1 mutably.
        let g1_id = scene
            .grids()
            .filter(|(id, _)| *id != g0)
            .map(|(id, _)| id)
            .next()
            .unwrap();
        scene.grid_mut(g1_id).unwrap().set_rect(
            IVec3::new(56, 56, 92),
            IVec3::new(71, 71, 107),
            Some(0x80_22_22_88), // dark blue
        );
        (scene, 0x80_88_22_22, 0x80_22_22_88)
    }

    fn make_composed_pool(pool_vsid: u32) -> (Engine, ScratchPool, u32) {
        let engine = Engine::new();
        let mut pool = ScratchPool::new(XRES, YRES, pool_vsid);
        let sky_color = engine.sky_color();
        let sky_col_i = i32::from_ne_bytes(sky_color.to_ne_bytes());
        pool.set_skycast(sky_col_i, 0);
        let fog_col_i = i32::from_ne_bytes(engine.fog_color().to_ne_bytes());
        pool.set_fog(fog_col_i, engine.fog_max_scan_dist());
        pool.set_treat_z_max_as_air(true);
        (engine, pool, sky_color)
    }

    fn pixel_count(width: u32, height: u32) -> usize {
        (width as usize) * (height as usize)
    }

    #[test]
    fn compose_into_takes_smaller_z() {
        let mut shared_fb = vec![0xff_ff_ff_ff_u32; 4];
        let mut shared_zb = vec![10.0f32; 4];
        let temp_fb = [0xaa_aa_aa_aa, 0x11_22_33_44, 0x55_66_77_88, 0xde_ad_be_ef];
        let temp_zb = [5.0f32, 20.0, 10.0, f32::INFINITY];
        compose_into(&mut shared_fb, &mut shared_zb, &temp_fb, &temp_zb);
        // i=0: 5 < 10 → take temp.
        assert_eq!(shared_fb[0], 0xaa_aa_aa_aa);
        assert_eq!(shared_zb[0], 5.0);
        // i=1: 20 > 10 → keep shared.
        assert_eq!(shared_fb[1], 0xff_ff_ff_ff);
        assert_eq!(shared_zb[1], 10.0);
        // i=2: 10 == 10 → keep shared (`<` not `<=`).
        assert_eq!(shared_fb[2], 0xff_ff_ff_ff);
        // i=3: INFINITY > 10 → keep shared.
        assert_eq!(shared_fb[3], 0xff_ff_ff_ff);
    }

    #[test]
    fn render_scene_composed_two_grids_both_visible() {
        // Camera positioned to see both grids' boxes. Grid 0's box
        // at world (~64, ~264, ~100); grid 1's box at world
        // (~264, ~264, ~100). Camera at world (160, 100, 100)
        // looking +y centres both in view.
        let (mut scene, red, blue) = build_two_grid_side_by_side();
        let (_engine, mut pool, sky_color) = make_composed_pool(CHUNK_SIZE_XY);
        let mut fb = vec![sky_color; pixel_count(XRES, YRES)];
        let mut zb = vec![f32::INFINITY; pixel_count(XRES, YRES)];

        let camera = camera_at([160.0, 100.0, 100.0]);
        let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
        let outcome = render_scene_composed(
            &mut fb,
            &mut zb,
            XRES as usize,
            XRES,
            YRES,
            &mut pool,
            &mut scene,
            &camera,
            &settings,
            sky_color,
            None,
        );
        assert_eq!(outcome, RenderOutcome::Rendered { grids_drawn: 2 });

        // Both colours should appear somewhere in the framebuffer.
        let red_count = fb.iter().filter(|&&p| p == red).count();
        let blue_count = fb.iter().filter(|&&p| p == blue).count();
        assert!(
            red_count > 0,
            "no red pixels: grid 0 (red box) not visible after compose"
        );
        assert!(
            blue_count > 0,
            "no blue pixels: grid 1 (blue box) not visible after compose"
        );
    }

    #[test]
    fn render_scene_composed_grid_a_in_front_of_grid_b() {
        // Two grids stacked along +y so grid A (closer) occludes
        // grid B (farther). After composition only grid A's colour
        // should appear on the overlap.
        let mut scene = Scene::new();
        let g_a = scene.add_grid(GridTransform::at(DVec3::new(0.0, 50.0, 0.0)));
        scene.grid_mut(g_a).unwrap().set_rect(
            IVec3::new(56, 56, 92),
            IVec3::new(71, 71, 107),
            Some(0x80_aa_00_00), // red
        );
        let _g_b = scene.add_grid(GridTransform::at(DVec3::new(0.0, 200.0, 0.0)));
        let g_b_id = scene
            .grids()
            .filter(|(id, _)| *id != g_a)
            .map(|(id, _)| id)
            .next()
            .unwrap();
        scene.grid_mut(g_b_id).unwrap().set_rect(
            IVec3::new(56, 56, 92),
            IVec3::new(71, 71, 107),
            Some(0x80_00_00_aa), // blue
        );

        let (_engine, mut pool, sky_color) = make_composed_pool(CHUNK_SIZE_XY);
        let mut fb = vec![sky_color; pixel_count(XRES, YRES)];
        let mut zb = vec![f32::INFINITY; pixel_count(XRES, YRES)];

        // Camera at (64, -10, 100) looking +y — both boxes line up
        // along the camera's forward axis.
        let camera = camera_at([64.0, -10.0, 100.0]);
        let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
        let outcome = render_scene_composed(
            &mut fb,
            &mut zb,
            XRES as usize,
            XRES,
            YRES,
            &mut pool,
            &mut scene,
            &camera,
            &settings,
            sky_color,
            None,
        );
        assert_eq!(outcome, RenderOutcome::Rendered { grids_drawn: 2 });

        // Red (closer grid) should be visible. Blue (farther grid)
        // may peek around the edges but the central pixels should
        // be red where both boxes project.
        let red_count = fb.iter().filter(|&&p| p == 0x80_aa_00_00).count();
        assert!(
            red_count > 0,
            "expected red pixels (closer box should win z-test)"
        );

        // Reverse the registration order (force grid B drawn first)
        // and verify that's irrelevant — composition is commutative.
        let mut scene2 = Scene::new();
        let g_b2 = scene2.add_grid(GridTransform::at(DVec3::new(0.0, 200.0, 0.0)));
        scene2.grid_mut(g_b2).unwrap().set_rect(
            IVec3::new(56, 56, 92),
            IVec3::new(71, 71, 107),
            Some(0x80_00_00_aa),
        );
        let g_a2 = scene2.add_grid(GridTransform::at(DVec3::new(0.0, 50.0, 0.0)));
        scene2.grid_mut(g_a2).unwrap().set_rect(
            IVec3::new(56, 56, 92),
            IVec3::new(71, 71, 107),
            Some(0x80_aa_00_00),
        );

        let mut fb2 = vec![sky_color; pixel_count(XRES, YRES)];
        let mut zb2 = vec![f32::INFINITY; pixel_count(XRES, YRES)];
        let outcome2 = render_scene_composed(
            &mut fb2,
            &mut zb2,
            XRES as usize,
            XRES,
            YRES,
            &mut pool,
            &mut scene2,
            &camera,
            &settings,
            sky_color,
            None,
        );
        assert_eq!(outcome2, RenderOutcome::Rendered { grids_drawn: 2 });
        assert_eq!(
            fb, fb2,
            "composition should be order-independent — same scene in different add order should produce identical output"
        );
    }

    #[test]
    fn render_scene_composed_empty_scene_returns_empty() {
        let mut scene = Scene::new();
        let (_engine, mut pool, sky_color) = make_composed_pool(CHUNK_SIZE_XY);
        let mut fb = vec![sky_color; pixel_count(XRES, YRES)];
        let mut zb = vec![f32::INFINITY; pixel_count(XRES, YRES)];
        let camera = camera_at([0.0, 0.0, 0.0]);
        let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
        let outcome = render_scene_composed(
            &mut fb,
            &mut zb,
            XRES as usize,
            XRES,
            YRES,
            &mut pool,
            &mut scene,
            &camera,
            &settings,
            sky_color,
            None,
        );
        assert_eq!(outcome, RenderOutcome::Empty);
        // fb should be unchanged (still all sky).
        assert!(fb.iter().all(|&p| p == sky_color));
    }

    /// FNV-1a 64-bit hash. Same offset/prime as the
    /// `roxlap-oracle::fnv1a64` helper used by the wasm-render
    /// goldens; pinning a render hash here is the same flavour of
    /// regression catch.
    fn fnv1a64(data: &[u8]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in data {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    // ---- S4.0 cross-chunk smoke test ----

    /// Two-chunk-wide grid: a recognisable shape spans the chunk
    /// boundary at `virtual_x = 128`. The render must not have a
    /// horizontal seam line at the boundary.
    #[test]
    fn render_scene_two_chunk_x_grid_no_seam() {
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::at(DVec3::new(0.0, 200.0, 0.0)));
        let g = scene.grid_mut(id).unwrap();
        // 100-voxel-tall stripe spanning x=[120..136] across the
        // x=128 chunk seam at z=200, y=[60..68]. After bake-free
        // render, every column in the stripe paints the same colour
        // at the same z; a seam at x=128 would show as missing
        // pixels in the column at virtual_x=128 / 129 / ...
        g.set_rect(
            IVec3::new(120, 60, 200),
            IVec3::new(136, 67, 215),
            Some(0x80_aa_55_22),
        );
        // Sanity: ensure both chunks were materialised.
        assert_eq!(g.chunk_count(), 2);

        // Render with a camera positioned to look at the stripe
        // straight on. Stripe at world (120..136, 260..268, 200..215).
        // Camera at (128, 100, 207) looking +y centres on it.
        let (_engine, mut pool, sky_color) = make_composed_pool(2 * CHUNK_SIZE_XY);
        let mut fb = vec![sky_color; pixel_count(XRES, YRES)];
        let mut zb = vec![f32::INFINITY; pixel_count(XRES, YRES)];
        let camera = camera_at([128.0, 100.0, 207.0]);
        let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
        let outcome = render_scene_composed(
            &mut fb,
            &mut zb,
            XRES as usize,
            XRES,
            YRES,
            &mut pool,
            &mut scene,
            &camera,
            &settings,
            sky_color,
            None,
        );
        assert_eq!(outcome, RenderOutcome::Rendered { grids_drawn: 1 });

        // Stripe colour should appear in roughly the centre of the
        // framebuffer. A chunk-edge seam would manifest as a thin
        // sky-coloured vertical line splitting the stripe in two.
        let stripe = 0x80_aa_55_22;
        let stripe_count = fb.iter().filter(|&&p| p == stripe).count();
        assert!(
            stripe_count > 200,
            "stripe rendered too few pixels ({stripe_count}) — chunks may not be stitching"
        );

        // Walk the centre row left-to-right looking for a sky-pixel
        // gap inside a stripe run. A gap 1+ pixels wide flags a
        // chunk-edge seam.
        let centre_y = (YRES / 2) as usize;
        let row_start = centre_y * (XRES as usize);
        let row = &fb[row_start..row_start + (XRES as usize)];
        let mut in_stripe = false;
        let mut seam_gaps = 0usize;
        for &px in row {
            if px == stripe {
                in_stripe = true;
            } else if in_stripe && px == sky_color {
                // Stripe ended; if we re-enter it on this row that's
                // a seam.
                if row.iter().skip_while(|&&p| p != px).any(|&p| p == stripe) {
                    // Look ahead for any further stripe pixel.
                    seam_gaps += 1;
                }
                in_stripe = false;
            }
        }
        // We allow seam_gaps to count the legitimate "stripe ended,
        // didn't restart" transition once; more than that means
        // multiple disjoint runs on the row → seam.
        assert!(
            seam_gaps <= 1,
            "centre row has {seam_gaps} disjoint stripe runs — expected 1 (chunk-edge seam suspected)"
        );
    }

    /// Pin the byte-exact FNV-1a64 of a 2-chunk render. Catches
    /// any drift in the cross-chunk stitch / opticast path.
    /// S4B.5: regression test for the cf-halving + column-step
    /// interaction at chunk-vsid (=128) under multi-mip rendering.
    /// Prior to the fix in `grouscan.rs:phase_after_delete_kept_presync`
    /// the single-chunk column-step recomputed `ixy_sptr_col_idx`
    /// from `cy * vsid + cx`, mapping the post-remiporend mip-N
    /// sub-table index back into mip-0's range. The next mip
    /// transition then underflowed at
    /// `state.ixy_sptr_col_idx - mip_base_offsets[old_mip]`.
    ///
    /// `mip_scan_dist=32` chosen so the 3-mip depth ladder
    /// (32→64→128 PREC scan budgets) reaches the floor 36 voxels
    /// away at mip-1 or mip-2 — exercising the post-transition
    /// rendering path that was broken pre-fix.
    #[test]
    fn vxl_generate_mips_on_set_voxel_chunk_renders() {
        let mut grid = crate::Grid::new(GridTransform::identity());
        // Solid floor at z=100..254 across the entire chunk —
        // looks like the oracle test's terrain.
        grid.set_rect(
            IVec3::new(0, 0, 100),
            IVec3::new(127, 127, 254),
            Some(0x80_88_88_88),
        );
        let chunk = grid.chunks.get_mut(&IVec3::ZERO).unwrap();
        chunk.generate_mips(3);
        let (_engine, mut pool, sky_color) = make_composed_pool(CHUNK_SIZE_XY);
        let mut fb = vec![sky_color; pixel_count(XRES, YRES)];
        let mut zb = vec![f32::INFINITY; pixel_count(XRES, YRES)];
        let camera = camera_at([64.0, 0.0, 64.0]);
        let mut settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
        settings.mip_levels = 3;
        settings.mip_scan_dist = 32;
        let grid_view = roxlap_core::GridView::from_single_vxl(&chunk);
        let mut rasterizer = ScalarRasterizer::new(&mut fb, &mut zb, XRES as usize, grid_view);
        let _ = core_opticast(&mut rasterizer, &mut pool, &camera, &settings, grid_view);
        drop(rasterizer);
        let non_sky = fb.iter().filter(|&&p| p != sky_color).count();
        assert!(
            non_sky > 0,
            "Vxl::generate_mips on a set_voxel-built chunk should render to something non-sky (got {non_sky})"
        );
    }

    /// Mip-0 preservation when mips are generated on the combined
    /// view but `mip_levels = 1` in the rasterizer's settings.
    /// Confirms `generate_mips` only APPENDS data — mip-0
    /// prefix is unchanged.
    #[test]
    fn render_with_mips_present_still_renders_mip0() {
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::at(DVec3::ZERO));
        scene.grid_mut(id).unwrap().set_rect(
            IVec3::new(40, 40, 40),
            IVec3::new(55, 55, 55),
            Some(0x80_88_88_88),
        );
        // S4B.4.a: force mip-1..mip-2 generation on the single
        // chunk directly (the Grid's combined-view cache API was
        // removed). The chunk's own Vxl::generate_mips builds its
        // own mip tables and the renderer happens to render through
        // them via Approach B's chunk_at_xy lookup.
        {
            let grid = scene.grid_mut(id).unwrap();
            let chunk = grid.chunks.get_mut(&IVec3::ZERO).unwrap();
            chunk.generate_mips(3);
        }

        let (_engine, mut pool, sky_color) = make_composed_pool(CHUNK_SIZE_XY);
        let mut fb = vec![sky_color; pixel_count(XRES, YRES)];
        let mut zb = vec![f32::INFINITY; pixel_count(XRES, YRES)];
        let camera = camera_at([64.0, 0.0, 64.0]);
        // mip_scan_dist huge → renderer never transitions past mip-0
        // so this test pins mip-0 correctness only.
        let mut settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
        settings.mip_scan_dist = 100_000;
        let outcome = render_scene_composed(
            &mut fb,
            &mut zb,
            XRES as usize,
            XRES,
            YRES,
            &mut pool,
            &mut scene,
            &camera,
            &settings,
            sky_color,
            None,
        );
        assert_eq!(outcome, RenderOutcome::Rendered { grids_drawn: 1 });
        let non_sky = fb.iter().filter(|&&p| p != sky_color).count();
        assert!(
            non_sky > 0,
            "render of single-grid scene with mips present rendered all-sky: mip-0 may be corrupted by generate_mips"
        );
    }

    #[test]
    fn render_scene_two_chunk_x_grid_hash_is_stable() {
        // Frozen 2026-05-10 at S4.0 landing on x86_64.
        const GOLDEN: u64 = 0x215e_d66d_7359_4725;
        // Same scene shape as `render_scene_two_chunk_x_grid_no_seam`
        // — kept distinct so the hash assertion doesn't share its
        // setup with the structural seam check.
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::at(DVec3::new(0.0, 200.0, 0.0)));
        scene.grid_mut(id).unwrap().set_rect(
            IVec3::new(120, 60, 200),
            IVec3::new(136, 67, 215),
            Some(0x80_aa_55_22),
        );
        let (_engine, mut pool, sky_color) = make_composed_pool(2 * CHUNK_SIZE_XY);
        let mut fb = vec![sky_color; pixel_count(XRES, YRES)];
        let mut zb = vec![f32::INFINITY; pixel_count(XRES, YRES)];
        let camera = camera_at([128.0, 100.0, 207.0]);
        let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
        let outcome = render_scene_composed(
            &mut fb,
            &mut zb,
            XRES as usize,
            XRES,
            YRES,
            &mut pool,
            &mut scene,
            &camera,
            &settings,
            sky_color,
            None,
        );
        assert_eq!(outcome, RenderOutcome::Rendered { grids_drawn: 1 });

        let bytes: Vec<u8> = fb.iter().flat_map(|p| p.to_ne_bytes()).collect();
        let hash = fnv1a64(&bytes);
        if GOLDEN == SENTINEL {
            // First-run capture mode — print the hash so the
            // developer can paste it into GOLDEN above.
            eprintln!("render_scene_two_chunk_x_grid_hash_is_stable: capture hash = 0x{hash:016x}");
            panic!("GOLDEN is the SENTINEL placeholder — paste 0x{hash:016x} into GOLDEN above");
        }
        assert_eq!(
            hash, GOLDEN,
            "2-chunk render hash drifted: expected 0x{GOLDEN:016x}, got 0x{hash:016x}"
        );
    }

    /// Sentinel for first-run hash capture in
    /// [`render_scene_two_chunk_x_grid_hash_is_stable`]. Replace
    /// `GOLDEN`'s definition with the printed value once captured.
    const SENTINEL: u64 = 0xDEAD_BEEF_DEAD_BEEF;

    /// S4B.2.c.3: render a 2-chunk x-stripe scene via Approach B
    /// (multi-chunk GridView + direct opticast). Validates the full
    /// chain — `Grid::chunk_xy_backing` → `ChunkGrid` →
    /// `GridView::from_chunk_grid` → opticast prelude
    /// `recompute_camera_chunk` → `camera_chunk_air_gap` lookup →
    /// gline's chunk-aware seed → grouscan's chunk-swap column-step.
    ///
    /// Geometry: a floor in chunk (0, 0) at grid-local
    /// `(0..127, 0..127, 200..205)` plus a recognisable box in
    /// chunk (1, 0) at `(160..170, 50..60, 150..165)`. Camera in
    /// chunk (0, 0) looking +x so rays cross into chunk (1, 0) and
    /// must trigger the cross-chunk DDA. (Camera in chunk (1, 0) is
    /// blocked by the in_bounds_xy check that still uses the per-
    /// chunk vsid; full grid-AABB in_bounds gets revisited later.)
    ///
    /// Hash-pinned. Non-sky pixel count is the primary correctness
    /// signal — if the cross-chunk DDA were broken, only the floor
    /// in chunk (0, 0) would render.
    #[test]
    fn approach_b_renders_two_chunk_x_stripe_via_chunk_grid() {
        const SENTINEL_B: u64 = 0xDEAD_BEEF_DEAD_BEEF;
        // Frozen 2026-05-11 on x86_64 — Approach B's first
        // multi-chunk render (S4B.2.c.3). Refreeze when changing
        // the rasterizer; the cross-chunk DDA stays validated by
        // the `floor_count` + `box_count` assertions above.
        const GOLDEN_B: u64 = 0x5ee1_e81c_66a8_d1f1;

        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::identity());
        let g = scene.grid_mut(id).unwrap();
        // Floor across chunk (0, 0) so rays looking +x always have
        // a hit before they exit the grid AABB.
        g.set_rect(
            IVec3::new(0, 0, 200),
            IVec3::new(127, 127, 205),
            Some(0x80_44_44_aa),
        );
        // Recognisable box deep in chunk (1, 0) — only visible if
        // the cross-chunk DDA fires.
        g.set_rect(
            IVec3::new(160, 50, 150),
            IVec3::new(170, 60, 165),
            Some(0x80_aa_55_22),
        );
        assert_eq!(g.chunk_count(), 2);

        // Build the multi-chunk GridView.
        let backing = g.chunk_xyz_backing().expect("at least one chunk populated");
        assert_eq!(backing.chunks_x, 2);
        assert_eq!(backing.chunks_y, 1);
        assert_eq!(backing.origin_chunk_xy, [0, 0]);
        let cg = roxlap_core::ChunkGrid {
            chunks: &backing.chunks,
            origin_chunk_xy: backing.origin_chunk_xy,
            origin_chunk_z: backing.origin_chunk_z,
            chunks_x: backing.chunks_x,
            chunks_y: backing.chunks_y,
            chunks_z: backing.chunks_z,
        };
        let grid_view = roxlap_core::GridView::from_chunk_grid(&cg, CHUNK_SIZE_XY);

        // Camera in chunk (0, 0) looking +x toward chunk (1, 0).
        // Voxlap z-down basis: right × down == forward.
        let camera = Camera {
            pos: [10.0, 64.0, 160.0],
            right: [0.0, 1.0, 0.0],
            down: [0.0, 0.0, 1.0],
            forward: [1.0, 0.0, 0.0],
        };
        let (_engine, mut pool, mut fb, mut zb) = render_setup(2 * CHUNK_SIZE_XY);
        let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
        let mut rasterizer = ScalarRasterizer::new(&mut fb, &mut zb, XRES as usize, grid_view);
        let outcome = core_opticast(&mut rasterizer, &mut pool, &camera, &settings, grid_view);
        drop(rasterizer);
        assert_eq!(outcome, OpticastOutcome::Rendered);

        // Hits BOTH the floor (chunk 0, 0) AND the box (chunk 1, 0).
        let floor_count = fb.iter().filter(|&&p| p == 0x80_44_44_aa).count();
        let box_count = fb.iter().filter(|&&p| p == 0x80_aa_55_22).count();
        assert!(
            floor_count > 1000,
            "floor not visible — only {floor_count} floor pixels (single-chunk path?)"
        );
        assert!(
            box_count > 50,
            "box in chunk (1, 0) not visible — only {box_count} box pixels — cross-chunk DDA may have failed to fire"
        );

        // Hash-pin the output. Refreeze when changing the rasterizer.
        let bytes: Vec<u8> = fb.iter().flat_map(|p| p.to_ne_bytes()).collect();
        let hash = fnv1a64(&bytes);
        if GOLDEN_B == SENTINEL_B {
            eprintln!("approach_b_renders_two_chunk_x_stripe_via_chunk_grid: capture hash = 0x{hash:016x}");
            panic!(
                "GOLDEN_B is the SENTINEL placeholder — paste 0x{hash:016x} into GOLDEN_B above"
            );
        }
        assert_eq!(
            hash, GOLDEN_B,
            "Approach B 2-chunk render hash drifted: expected 0x{GOLDEN_B:016x}, got 0x{hash:016x}"
        );
    }

    /// S4B.2.d: multi-chunk camera that sits **past** the first
    /// chunk's vsid (i.e., inside chunk (1, 0)). Validates that
    /// `recompute_in_bounds_xy` against the grid AABB recognises
    /// the camera as in-bounds — and that the seed path looks up
    /// chunk (1, 0)'s column via `chunk_at_xy(camera_chunk_idx)`
    /// instead of returning the OOB-XY bedrock placeholder.
    ///
    /// Scene shape: a 2-chunk x-stripe with a floor in chunk (1, 0)
    /// (where the camera sits) and the recognisable box in chunk
    /// (0, 0). The camera looks `-x` so rays cross from chunk
    /// (1, 0) back into chunk (0, 0). Tests the OTHER direction of
    /// cross-chunk DDA + the in-bounds AABB fix together.
    #[test]
    fn approach_b_camera_in_chunk_1_0_renders_neighbour() {
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::identity());
        let g = scene.grid_mut(id).unwrap();
        // Floor under the camera in chunk (1, 0).
        g.set_rect(
            IVec3::new(128, 0, 200),
            IVec3::new(255, 127, 205),
            Some(0x80_44_44_aa),
        );
        // Box deep in chunk (0, 0) — only visible if the camera in
        // chunk (1, 0) is recognised as in-bounds AND the
        // cross-chunk DDA fires westward.
        g.set_rect(
            IVec3::new(20, 50, 150),
            IVec3::new(30, 60, 165),
            Some(0x80_aa_55_22),
        );
        assert_eq!(g.chunk_count(), 2);

        let backing = g.chunk_xyz_backing().expect("populated");
        let cg = roxlap_core::ChunkGrid {
            chunks: &backing.chunks,
            origin_chunk_xy: backing.origin_chunk_xy,
            origin_chunk_z: backing.origin_chunk_z,
            chunks_x: backing.chunks_x,
            chunks_y: backing.chunks_y,
            chunks_z: backing.chunks_z,
        };
        let grid_view = roxlap_core::GridView::from_chunk_grid(&cg, CHUNK_SIZE_XY);
        let (aabb_min, aabb_max) = grid_view.aabb_xy();
        assert_eq!(aabb_min, [0, 0]);
        assert_eq!(aabb_max, [256, 128]);

        // Camera deep in chunk (1, 0): world (200, 64, 160). Past
        // the single-chunk vsid=128 OOB cutoff but inside the
        // multi-chunk AABB. Look -x toward chunk (0, 0).
        let camera = Camera {
            pos: [200.0, 64.0, 160.0],
            right: [0.0, -1.0, 0.0],
            down: [0.0, 0.0, 1.0],
            forward: [-1.0, 0.0, 0.0],
        };
        let (_engine, mut pool, mut fb, mut zb) = render_setup(2 * CHUNK_SIZE_XY);
        let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
        let mut rasterizer = ScalarRasterizer::new(&mut fb, &mut zb, XRES as usize, grid_view);
        let outcome = core_opticast(&mut rasterizer, &mut pool, &camera, &settings, grid_view);
        drop(rasterizer);
        assert_eq!(outcome, OpticastOutcome::Rendered);

        let floor_count = fb.iter().filter(|&&p| p == 0x80_44_44_aa).count();
        let box_count = fb.iter().filter(|&&p| p == 0x80_aa_55_22).count();
        assert!(
            floor_count > 1000,
            "floor under camera in chunk (1, 0) not visible — only {floor_count} floor pixels — in_bounds_xy fix may not have taken effect"
        );
        assert!(
            box_count > 50,
            "box in chunk (0, 0) not visible — only {box_count} box pixels — westward cross-chunk DDA failed"
        );
    }

    /// S4B.6.c: stacked-grid scaffold — camera in chz=1 (= world
    /// z=256..511) of a 2-chunk-tall grid should render its own
    /// chunk's terrain. Verifies cf seed + slab-byte reads + chunk-
    /// XY swaps all use world-z consistently.
    ///
    /// Cross-chunk look-down (= camera in chz=0 sees terrain in
    /// chz=1) needs cf z range extension at air-gap-lookup time;
    /// that's a follow-up to S4B.6.c.
    #[test]
    fn stacked_two_chunk_z_camera_in_chz1_sees_own_chunk_floor() {
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::at(DVec3::ZERO));
        let g = scene.grid_mut(id).unwrap();
        // chz=0: all-air (materialised so chunk_xyz_backing enumerates).
        g.ensure_chunk(IVec3::new(0, 0, 0));
        // chz=1: floor at local z=50 (= world z=306).
        g.set_rect(
            IVec3::new(60, 60, 306),
            IVec3::new(72, 72, 310),
            Some(0x80_33_66_99),
        );
        assert!(g.chunk(IVec3::new(0, 0, 1)).is_some());

        let (_engine, mut pool, sky_color) = make_composed_pool(2 * CHUNK_SIZE_XY);
        let mut fb = vec![sky_color; pixel_count(XRES, YRES)];
        let mut zb = vec![f32::INFINITY; pixel_count(XRES, YRES)];
        pool.set_treat_z_max_as_air(true);
        // Camera at world (66, 66, 280) — directly above the
        // floor at world z=306. Look STRAIGHT DOWN (z increases =
        // down in voxlap z-down).
        let camera = Camera {
            pos: [66.0, 66.0, 280.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 1.0, 0.0],
            forward: [0.0, 0.0, 1.0],
        };
        let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
        let outcome = render_scene_composed(
            &mut fb,
            &mut zb,
            XRES as usize,
            XRES,
            YRES,
            &mut pool,
            &mut scene,
            &camera,
            &settings,
            sky_color,
            None,
        );
        assert_eq!(outcome, RenderOutcome::Rendered { grids_drawn: 1 });
        let floor_count = fb.iter().filter(|&&p| p == 0x80_33_66_99).count();
        assert!(
            floor_count > 100,
            "camera at chz=1 with floor in same chunk should see it — got {floor_count} floor pixels"
        );
    }

    /// S4B.6.e: cross-chunk look-down. Camera in chz=0's all-air
    /// chunk should see chz=1's floor below it. This was deferred
    /// from S4B.6.c because the cf seed's z range capped at the
    /// camera-chunk's bedrock (world z=255); S4B.6.e extends the
    /// air-gap walk in `camera_chunk_air_gap` to step into the
    /// next chunk down when the camera's column is all-air-bedrock,
    /// and the rasterizer routes state.column / slab_buf to the
    /// chunk holding the real floor via `seed_chunk_z`.
    #[test]
    fn stacked_two_chunk_z_camera_in_chz0_sees_chz1_floor() {
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::at(DVec3::ZERO));
        let g = scene.grid_mut(id).unwrap();
        // chz=0: all-air. Materialised so chunk_xyz_backing
        // enumerates it.
        g.ensure_chunk(IVec3::new(0, 0, 0));
        // chz=1: floor at world z=306..310 (= local z=50..54).
        g.set_rect(
            IVec3::new(60, 60, 306),
            IVec3::new(72, 72, 310),
            Some(0x80_77_aa_44),
        );
        assert!(g.chunk(IVec3::new(0, 0, 1)).is_some());

        let (_engine, mut pool, sky_color) = make_composed_pool(2 * CHUNK_SIZE_XY);
        let mut fb = vec![sky_color; pixel_count(XRES, YRES)];
        let mut zb = vec![f32::INFINITY; pixel_count(XRES, YRES)];
        pool.set_treat_z_max_as_air(true);
        // Camera at world (66, 66, 100) — in chz=0's all-air
        // chunk. Look STRAIGHT DOWN (z+) toward chz=1's floor at
        // world z=306.
        let camera = Camera {
            pos: [66.0, 66.0, 100.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 1.0, 0.0],
            forward: [0.0, 0.0, 1.0],
        };
        let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
        let outcome = render_scene_composed(
            &mut fb,
            &mut zb,
            XRES as usize,
            XRES,
            YRES,
            &mut pool,
            &mut scene,
            &camera,
            &settings,
            sky_color,
            None,
        );
        assert_eq!(outcome, RenderOutcome::Rendered { grids_drawn: 1 });
        let floor_count = fb.iter().filter(|&&p| p == 0x80_77_aa_44).count();
        assert!(
            floor_count > 50,
            "camera in chz=0 air-gap should see chz=1 floor via cross-chunk look-down — got {floor_count} floor pixels"
        );
    }

    /// S4B.6.h: mid-render chunk-Z handoff. Camera column has
    /// content in chz=0 (= a mountain at the camera's XY) so
    /// seed-time cross-chunk look-down does NOT fire — seed_chz=0.
    /// As rays DDA across the scene, they visit XY columns where
    /// chz=0 is all-air-bedrock. Mid-render handoff should swap
    /// state to chz=1's column at those XY positions and reveal
    /// hill content sitting under the camera's chz=0 layer.
    ///
    /// This is the "tall mountains breaching chunk-Z boundary"
    /// case the demo aims for.
    #[test]
    fn mid_render_handoff_reveals_chz1_hills_under_mountain_camera() {
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::at(DVec3::ZERO));
        let g = scene.grid_mut(id).unwrap();
        // chz=0: a small "mountain peak" at the camera's XY.
        // Mountain at world z=150..200 — solid block.
        g.set_rect(
            IVec3::new(60, 60, 150),
            IVec3::new(72, 72, 200),
            Some(0x80_88_44_22), // brown mountain
        );
        // chz=1: hills at world z=336..360 across the WHOLE chunk
        // (so DDA rays hit them when chz=0 is air).
        g.set_rect(
            IVec3::new(0, 0, 336),
            IVec3::new(128, 128, 360),
            Some(0x80_22_88_44), // green hills
        );
        // Carve a hole in chz=1's hill at the mountain's footprint
        // so the mountain doesn't appear to "float" on green.
        g.set_rect(IVec3::new(60, 60, 336), IVec3::new(72, 72, 360), None);
        assert!(g.chunk(IVec3::new(0, 0, 0)).is_some());
        assert!(g.chunk(IVec3::new(0, 0, 1)).is_some());

        let (_engine, mut pool, sky_color) = make_composed_pool(2 * CHUNK_SIZE_XY);
        let mut fb = vec![sky_color; pixel_count(XRES, YRES)];
        let mut zb = vec![f32::INFINITY; pixel_count(XRES, YRES)];
        pool.set_treat_z_max_as_air(true);
        // Camera at world (66, 66, 100) — directly above the
        // mountain peak (at z=150). Camera column has the
        // mountain in chz=0. Look straight down.
        let camera = Camera {
            pos: [66.0, 66.0, 100.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 1.0, 0.0],
            forward: [0.0, 0.0, 1.0],
        };
        let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
        let outcome = render_scene_composed(
            &mut fb,
            &mut zb,
            XRES as usize,
            XRES,
            YRES,
            &mut pool,
            &mut scene,
            &camera,
            &settings,
            sky_color,
            None,
        );
        assert_eq!(outcome, RenderOutcome::Rendered { grids_drawn: 1 });
        let mountain_count = fb.iter().filter(|&&p| p == 0x80_88_44_22).count();
        let hill_count = fb.iter().filter(|&&p| p == 0x80_22_88_44).count();
        // Verify the hills render at approximately the correct
        // world-z by sampling the z-buffer at hill pixels. Camera
        // at z=100 looking straight down; hills at world z=336.
        // Expected depth = 236 for directly-below pixels. If
        // state.z1 stays stuck at the mountain peak's z=150 the
        // hills would render with depth ≈ 50 → orders of magnitude
        // off.
        let mut hill_depths: Vec<f32> = fb
            .iter()
            .zip(zb.iter())
            .filter_map(|(&p, &d)| if p == 0x80_22_88_44 { Some(d) } else { None })
            .collect();
        hill_depths.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_hill_depth = hill_depths[hill_depths.len() / 2];
        eprintln!(
            "mid-render handoff: mountain={mountain_count} hill={hill_count} median_hill_depth={median_hill_depth:.1}"
        );
        assert!(
            mountain_count > 50,
            "should see mountain peak via chz=0 — got {mountain_count} mountain pixels"
        );
        assert!(
            hill_count > 50,
            "should see chz=1 hills via mid-render handoff — got {hill_count} hill pixels"
        );
        assert!(
            (median_hill_depth - 236.0).abs() < 80.0,
            "hill median depth should be ≈236 (camera→z=336); got {median_hill_depth:.1} — state.z1 may be stale at the mountain peak's z"
        );
    }

    /// S4B.6.g: cross-chunk look-down under multi-mip. Same scene
    /// as `stacked_two_chunk_z_camera_in_chz0_sees_chz1_floor` but
    /// with `mip_levels=2, mip_scan_dist=16` so the rasterizer
    /// transitions to mip-1 well within the chz=1 terrain. Locks in
    /// the slab_z_at mip-N offset fix (= `chunk_world_z_base >>
    /// gmipcnt`). Pre-fix produced a green / brown "wall in a circle
    /// around the camera" because mip-1 rendered the floor at
    /// world-z ≈ 178 instead of 306.
    #[test]
    fn stacked_two_chunk_z_camera_in_chz0_sees_chz1_floor_multi_mip() {
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::at(DVec3::ZERO));
        let g = scene.grid_mut(id).unwrap();
        g.ensure_chunk(IVec3::new(0, 0, 0));
        g.set_rect(
            IVec3::new(60, 60, 306),
            IVec3::new(72, 72, 310),
            Some(0x80_77_aa_44),
        );
        assert!(g.chunk(IVec3::new(0, 0, 1)).is_some());

        let (_engine, mut pool, sky_color) = make_composed_pool(2 * CHUNK_SIZE_XY);
        let mut fb = vec![sky_color; pixel_count(XRES, YRES)];
        let mut zb = vec![f32::INFINITY; pixel_count(XRES, YRES)];
        pool.set_treat_z_max_as_air(true);
        let camera = Camera {
            pos: [66.0, 66.0, 100.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 1.0, 0.0],
            forward: [0.0, 0.0, 1.0],
        };
        let mut settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
        settings.mip_levels = 2;
        settings.mip_scan_dist = 16;
        let outcome = render_scene_composed(
            &mut fb,
            &mut zb,
            XRES as usize,
            XRES,
            YRES,
            &mut pool,
            &mut scene,
            &camera,
            &settings,
            sky_color,
            None,
        );
        assert_eq!(outcome, RenderOutcome::Rendered { grids_drawn: 1 });
        let floor_count = fb.iter().filter(|&&p| p == 0x80_77_aa_44).count();
        assert!(
            floor_count > 50,
            "multi-mip cross-chunk look-down should still see chz=1 floor — got {floor_count} floor pixels"
        );
    }

    /// S4B.6.d: 3-chunk-tall stack stresses the widened gylookup
    /// (`(chunks_z * 512) >> mip + 4` per mip). Pre-S4B.6.d, gylookup
    /// was hardcoded at `(512 >> mip) + 4`, which would OOB or alias
    /// for any z > 511. This test renders a floor at world z=562
    /// (= chz=2, local z=50) with the camera at world z=540, looking
    /// straight down. Multi-mip is on so we exercise the mip slide
    /// path in `phase_remiporend` that scales `advance` by chunks_z.
    #[test]
    fn stacked_three_chunk_z_camera_in_chz2_sees_own_chunk_floor_multi_mip() {
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::at(DVec3::ZERO));
        let g = scene.grid_mut(id).unwrap();
        // Materialise chz=0 + chz=1 so chunk_xyz_backing enumerates
        // the full stack.
        g.ensure_chunk(IVec3::new(0, 0, 0));
        g.ensure_chunk(IVec3::new(0, 0, 1));
        // chz=2: floor at world z=562..566 (= local z=50..54).
        g.set_rect(
            IVec3::new(60, 60, 562),
            IVec3::new(72, 72, 566),
            Some(0x80_aa_55_22),
        );
        assert!(g.chunk(IVec3::new(0, 0, 2)).is_some());

        let (_engine, mut pool, sky_color) = make_composed_pool(2 * CHUNK_SIZE_XY);
        let mut fb = vec![sky_color; pixel_count(XRES, YRES)];
        let mut zb = vec![f32::INFINITY; pixel_count(XRES, YRES)];
        pool.set_treat_z_max_as_air(true);
        let camera = Camera {
            pos: [66.0, 66.0, 540.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 1.0, 0.0],
            forward: [0.0, 0.0, 1.0],
        };
        // Multi-mip on to exercise the gylookup-slide path.
        let mut settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
        settings.mip_levels = 2;
        settings.mip_scan_dist = 16;
        let outcome = render_scene_composed(
            &mut fb,
            &mut zb,
            XRES as usize,
            XRES,
            YRES,
            &mut pool,
            &mut scene,
            &camera,
            &settings,
            sky_color,
            None,
        );
        assert_eq!(outcome, RenderOutcome::Rendered { grids_drawn: 1 });
        let floor_count = fb.iter().filter(|&&p| p == 0x80_aa_55_22).count();
        assert!(
            floor_count > 100,
            "camera at chz=2 with floor in same chunk should see it — got {floor_count} floor pixels"
        );
    }
}
