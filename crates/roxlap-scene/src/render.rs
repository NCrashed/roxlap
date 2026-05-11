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
//! ## S4.0: Approach C combined-world stitch
//!
//! Both APIs route per-grid rendering through
//! [`crate::Grid::combined_world`] — the per-grid combined virtual
//! world covers every populated chunk in the grid's XY footprint
//! and is rendered as a single opticast pass. The 2D-DDA inside
//! [`roxlap_core::grouscan`] already walks across an arbitrary
//! `vsid × vsid` lattice, so cross-chunk seams collapse into normal
//! column steps. Empty chunks contribute all-air placeholder
//! columns.
//!
//! Per-grid rotation (S5) and per-grid LOD (S6) plug in at the
//! same dispatch point: rotate the world camera into grid-local
//! before the combined-view lookup, then dispatch coarse / fine /
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

use crate::Scene;

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
        let grid_origin = grid.transform.origin;
        let combined = grid.combined_world();
        let offset = combined.voxel_offset();
        // World → grid-local → virtual: subtract grid origin, then
        // add the combined view's voxel offset so negative-index
        // chunks (origin_chunk.x/y < 0) map to virtual coords in
        // [0, vsid).
        let local_cam = Camera {
            pos: [
                camera.pos[0] - grid_origin.x + f64::from(offset.x),
                camera.pos[1] - grid_origin.y + f64::from(offset.y),
                camera.pos[2] - grid_origin.z,
            ],
            right: camera.right,
            down: camera.down,
            forward: camera.forward,
        };
        let outcome = {
            let mut rasterizer = ScalarRasterizer::new(
                fb,
                zb,
                pitch_pixels,
                &combined.data,
                &combined.column_offset,
                &combined.mip_base_offsets,
                combined.vsid,
            );
            if let Some(sky_ref) = sky {
                rasterizer = rasterizer.with_sky(sky_ref);
            }
            opticast(
                &mut rasterizer,
                pool,
                &local_cam,
                settings,
                combined.vsid,
                &combined.data,
                &combined.column_offset,
            )
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

        let grid_origin = grid.transform.origin;
        let combined = grid.combined_world();
        let offset = combined.voxel_offset();
        // World → grid-local → virtual: subtract grid origin, then
        // add the combined view's voxel offset so negative-index
        // chunks map into the virtual-coord range opticast expects.
        let local_cam = Camera {
            pos: [
                camera.pos[0] - grid_origin.x + f64::from(offset.x),
                camera.pos[1] - grid_origin.y + f64::from(offset.y),
                camera.pos[2] - grid_origin.z,
            ],
            right: camera.right,
            down: camera.down,
            forward: camera.forward,
        };

        let outcome = {
            let mut rasterizer = ScalarRasterizer::new(
                &mut temp_fb,
                &mut temp_zb,
                pitch_pixels,
                &combined.data,
                &combined.column_offset,
                &combined.mip_base_offsets,
                combined.vsid,
            );
            if let Some(sky_ref) = sky {
                rasterizer = rasterizer.with_sky(sky_ref);
            }
            opticast(
                &mut rasterizer,
                pool,
                &local_cam,
                settings,
                combined.vsid,
                &combined.data,
                &combined.column_offset,
            )
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
    /// Refreeze the hash if a deliberate change to the rendering
    /// pipeline lands; the seam test above is the
    /// look-and-feel-correct check.
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
}
