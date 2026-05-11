//! Regression test for the OOB-XY chunk-edge streaking bug
//! resolved 2026-05-10 (see
//! `memory/project_oob_xy_chunk_edge_streaking.md`).
//!
//! The user found a tiny-delta bug↔no-bug capture pair that
//! crossed the bug threshold by ~1 voxel of camera movement.
//! Bisecting the axes pinned the trigger to integer floors of
//! `li_pos[0]` / `li_pos[2]` against the representative column's
//! `surface_z`; the fix in `crates/roxlap-core/src/opticast.rs`
//! drops the synthesised cf seed and always uses `(0, 255, 0)`
//! for OOB-XY cameras.
//!
//! This module keeps the regression assertion + a PPM-dumping
//! test for quick visual inspection. The full bisection harness
//! used during the investigation is preserved in
//! `git show <fix-commit>~1` if needed for future OOB-XY work.

#![cfg(test)]
#![allow(clippy::cast_precision_loss)]

use glam::DVec3;
use roxlap_core::opticast::OpticastSettings;
use roxlap_core::rasterizer::ScratchPool;
use roxlap_core::{Camera, Engine};
use roxlap_scene::render::render_scene_composed;
use roxlap_scene::{GridTransform, Scene};

use crate::terrain;

/// 800×600 — same as the live demo so PPMs are comparable to
/// captures from the running binary.
const W: u32 = 800;
const H: u32 = 600;

fn camera_for_yaw_pitch(pos: [f64; 3], yaw: f64, pitch: f64) -> Camera {
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    Camera {
        pos,
        right: [-sy, cy, 0.0],
        down: [-cy * sp, -sy * sp, cp],
        forward: [cy * cp, sy * cp, sp],
    }
}

/// Render `scene` from `camera` via `render_scene_composed` with
/// the live demo's pool config (`treat_z_max_as_air` on, fog +
/// skycast set from the engine defaults). Pool vsid is sized for
/// S4.0's 2-chunk-wide ground (combined vsid = 256).
fn render_pose(scene: &mut Scene, camera: &Camera) -> Vec<u32> {
    let engine = Engine::new();
    let mut pool = ScratchPool::new(W, H, 2 * roxlap_scene::CHUNK_SIZE_XY);
    let sky = engine.sky_color();
    let sky_col_i = i32::from_ne_bytes(sky.to_ne_bytes());
    pool.set_skycast(sky_col_i, 0);
    let fog_col_i = i32::from_ne_bytes(engine.fog_color().to_ne_bytes());
    pool.set_fog(fog_col_i, engine.fog_max_scan_dist());
    pool.set_treat_z_max_as_air(true);

    let pixel_count = (W as usize) * (H as usize);
    let mut fb = vec![sky; pixel_count];
    let mut zb = vec![f32::INFINITY; pixel_count];
    let settings = OpticastSettings::for_oracle_framebuffer(W, H);
    let _ = render_scene_composed(
        &mut fb, &mut zb, W as usize, W, H, &mut pool, scene, camera, &settings, sky, None,
    );
    fb
}

#[allow(clippy::cast_possible_truncation)]
fn write_ppm(path: &str, fb: &[u32]) {
    let mut bytes = format!("P6\n{W} {H}\n255\n").into_bytes();
    for &px in fb {
        bytes.push(((px >> 16) & 0xff) as u8); // R
        bytes.push(((px >> 8) & 0xff) as u8); // G
        bytes.push((px & 0xff) as u8); // B
    }
    std::fs::write(path, bytes).expect("write ppm");
}

// User-captured poses that triggered the streaking bug.
//
// `BUG` and `NOBUG` differ by ~1 voxel of camera movement —
// `li_pos` goes from `(110, -38, 192)` (no bug) to
// `(111, -40, 191)` (bug). Both are OOB-XY (y < 0). The bug
// flipped on when the representative-column synthesis switched
// from "camera in solid → fallback `(0, 255, 0)`" to "camera in
// air → `Some((0, surface_z, 0))`".
const NOBUG_POS: [f64; 3] = [
    110.496_288_504_681_33,
    -37.748_461_580_480_56,
    192.670_231_071_822_88,
];
const NOBUG_YAW: f64 = 2.170_796_326_794_883_3;
const NOBUG_PITCH: f64 = 0.554_999_999_999_998_6;
const BUG_POS: [f64; 3] = [
    111.345_628_963_082_2,
    -39.051_928_328_994_1,
    191.727_053_679_962_84,
];
const BUG_YAW: f64 = 2.148_296_326_794_884;
const BUG_PITCH: f64 = 0.544_999_999_999_998_8;

/// Build a minimal 2×1 ground scene for the streaking regression
/// tests. The full demo (32×32 ground + ship + lighting bake)
/// takes ~7 s to construct; the streaking bug only depends on
/// having SOME terrain past which the camera sits OOB-y. 2×1 is
/// the smallest extent that still exercises the cross-chunk
/// stitch path (it's the OOB-XY render that triggers, not the
/// per-chunk gline).
fn build_ground_only() -> Scene {
    let mut scene = Scene::new();
    let id = scene.add_grid(GridTransform::at(DVec3::ZERO));
    let grid = scene.grid_mut(id).expect("ground grid present");
    terrain::build_ground_extent(grid, 2, 1);
    scene
}

/// Regression test: the bug-pose's framebuffer used to have ~20%
/// fewer sky pixels than the no-bug pose because vertical voxel
/// streaks displaced sky. Post-fix the two poses' sky fractions
/// are within a couple of percent — the test fails (loudly) if
/// the streaks come back.
#[test]
fn chunk_edge_streaking_bug_is_fixed() {
    let mut scene = build_ground_only();
    let engine = Engine::new();
    let sky = engine.sky_color();

    let nobug_fb = {
        let cam = camera_for_yaw_pitch(NOBUG_POS, NOBUG_YAW, NOBUG_PITCH);
        render_pose(&mut scene, &cam)
    };
    let bug_fb = {
        let cam = camera_for_yaw_pitch(BUG_POS, BUG_YAW, BUG_PITCH);
        render_pose(&mut scene, &cam)
    };

    let sky_pre = nobug_fb.iter().filter(|&&p| p == sky).count();
    let sky_post = bug_fb.iter().filter(|&&p| p == sky).count();
    let pixel_count = nobug_fb.len();
    let pre_pct = 100.0 * sky_pre as f64 / pixel_count as f64;
    let post_pct = 100.0 * sky_post as f64 / pixel_count as f64;
    eprintln!("nobug sky pixels: {sky_pre}/{pixel_count} ({pre_pct:.2}%)");
    eprintln!("bug   sky pixels: {sky_post}/{pixel_count} ({post_pct:.2}%)");

    let diff = sky_pre.abs_diff(sky_post);
    let diff_frac = diff as f64 / pixel_count as f64;
    assert!(
        diff_frac < 0.05,
        "sky-pixel fraction drift {diff} ({:.2}%) exceeds 5% — chunk-edge streaks may be back",
        100.0 * diff_frac,
    );
}

/// Smoke-test the demo's full 32×32×1 ground at startup: build
/// it via [`build_demo`], render one frame, assert no panic and
/// non-trivial output. Catches build-time regressions in the
/// large-vsid path (4096²-column combined view, 64 MB column-
/// offset table) without running the interactive binary.
///
/// Ignored by default because the scene build dominates wall
/// time (~3 s) and locks up a CI worker. Run with
/// `cargo test --release -p roxlap-scene-demo --
/// --ignored` to exercise it.
#[test]
#[ignore = "expensive: builds the full 32x32 ground (~3 s on dev hardware)"]
fn full_demo_scene_renders_without_panic() {
    let mut scene_and_camera = crate::scene::build_demo();
    let engine = Engine::new();
    let mut pool = ScratchPool::new(W, H, 32 * roxlap_scene::CHUNK_SIZE_XY);
    let sky = engine.sky_color();
    let sky_col_i = i32::from_ne_bytes(sky.to_ne_bytes());
    pool.set_skycast(sky_col_i, 0);
    let fog_col_i = i32::from_ne_bytes(engine.fog_color().to_ne_bytes());
    pool.set_fog(fog_col_i, engine.fog_max_scan_dist());
    pool.set_treat_z_max_as_air(true);

    let pixel_count = (W as usize) * (H as usize);
    let mut fb = vec![sky; pixel_count];
    let mut zb = vec![f32::INFINITY; pixel_count];
    let settings = OpticastSettings::for_oracle_framebuffer(W, H);
    let _ = render_scene_composed(
        &mut fb,
        &mut zb,
        W as usize,
        W,
        H,
        &mut pool,
        &mut scene_and_camera.scene,
        &scene_and_camera.camera,
        &settings,
        sky,
        None,
    );
    let non_sky_count = fb.iter().filter(|&&p| p != sky).count();
    assert!(
        non_sky_count > 0,
        "full demo render at start camera produced no non-sky pixels"
    );
    eprintln!(
        "full demo render: {non_sky_count}/{pixel_count} non-sky pixels ({:.1}%)",
        100.0 * non_sky_count as f64 / pixel_count as f64
    );
}

/// Convenience: dump the bug-pose's framebuffer to `/tmp/` for
/// quick visual inspection. Useful if S4 (cross-chunk gline) or
/// S5 (rotation) regress the OOB-XY render path.
#[test]
fn dump_bug_pose_ppm_for_inspection() {
    let mut scene = build_ground_only();
    let cam = camera_for_yaw_pitch(BUG_POS, BUG_YAW, BUG_PITCH);
    let fb = render_pose(&mut scene, &cam);
    write_ppm("/tmp/scene-demo-bug-pose.ppm", &fb);
    eprintln!("wrote /tmp/scene-demo-bug-pose.ppm");
}
