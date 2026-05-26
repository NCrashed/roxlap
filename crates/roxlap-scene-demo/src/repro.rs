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
    // Mirror the live demo's settings since `build_demo` now
    // generates mips on the combined view; the rasterizer must
    // see `mip_levels > 1` to consume them.
    let mut settings = OpticastSettings::for_oracle_framebuffer(W, H);
    settings.mip_levels = 4;
    settings.mip_scan_dist = 64;
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

/// S4B.6.f stacked-ground smoke: build the demo with the chunks_z=2
/// terrain (chz=0 all-air + chz=1 hilly) and the matching spawn
/// pitch, render the spawn pose, assert non-trivial output. Validates
/// that S4B.6.e cross-chunk look-down + the chz-aware bake actually
/// produce visible terrain pixels in the live demo path.
#[test]
#[ignore = "expensive: builds the full 32x32 ground (~3 s on dev hardware)"]
fn stacked_demo_scene_renders_terrain_from_chz0() {
    std::env::set_var("ROXLAP_STACKED_GROUND", "1");
    let mut scene_and_camera = crate::scene::build_demo();
    std::env::remove_var("ROXLAP_STACKED_GROUND");
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
    let mut settings = OpticastSettings::for_oracle_framebuffer(W, H);
    settings.mip_levels = 4;
    settings.mip_scan_dist = 64;
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
    const MOUNTAIN_STONE: u32 = 0x80_8a_82_7a;
    let mountain_count = fb.iter().filter(|&&p| p == MOUNTAIN_STONE).count();
    eprintln!(
        "stacked demo render: {non_sky_count}/{pixel_count} non-sky pixels ({:.1}%), mountain={mountain_count}",
        100.0 * non_sky_count as f64 / pixel_count as f64
    );
    // Camera at world z=200 pitched -0.35 rad sees ~250+ voxels of
    // hilly terrain in chz=1 below. Reasonable floor: 5% non-sky.
    let non_sky_frac = non_sky_count as f64 / pixel_count as f64;
    assert!(
        non_sky_frac > 0.05,
        "stacked demo rendered <5% non-sky pixels — cross-chunk look-down may not be hitting chz=1 terrain"
    );
}

/// S4B.6.i: render from the user-reported capture pose near a tall
/// mountain (camera at world (239, 298, 101), close to the
/// `(220, 320)` mountain) and verify that mountain pixels cover
/// both halves — the chz=0 portion (= peak side) AND the chz=1
/// portion (= base side via mid-render handoff). Pre-S4B.6.i the
/// bedrock placeholder at chz=0's z=255 was overwritten by the
/// mountain step crossing the chunk boundary; with the placeholder
/// gone, the handoff sentinel never fires and only the chz=0 top
/// of the mountain rendered (the bug the user reported: "mountains
/// floating in mid air, only top layer visible").
///
/// S4B.6.j follow-up poses (currently FAILING):
/// - Pose A `(13.50, 169.45, 193.12)` yaw 3.01 pitch 1.26:
///   user reports "only bottom layer visible (= chunk z=0 portion)".
/// - Pose B `(13.71, 171.04, 193.12)` same yaw/pitch: a 1.6-unit
///   step from A renders ALL SKY. Thin band suggests a chunk-XY
///   boundary issue.
/// - Pose C `(21.50, 181.72, 177.21)` yaw 3.19 pitch 0.76: renders
///   both layers correctly.
/// These help isolate why some poses lose half the mountain.
#[test]
#[ignore = "expensive: builds the full 32x32 stacked ground (~3 s on dev hardware)"]
fn stacked_demo_renders_full_mountain_at_user_capture_pose() {
    std::env::set_var("ROXLAP_STACKED_GROUND", "1");
    let mut scene_and_camera = crate::scene::build_demo();
    std::env::remove_var("ROXLAP_STACKED_GROUND");

    // The exact pose captured by the F key in the demo.
    scene_and_camera.cam_pos = [
        239.015_574_238_198_4,
        298.289_282_655_482,
        101.626_834_407_672_85,
    ];
    scene_and_camera.yaw = -3.484_203_673_205_028_6;
    scene_and_camera.pitch = 0.997_500_000_000_006_3;
    scene_and_camera.refresh_camera();

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
    let mut settings = OpticastSettings::for_oracle_framebuffer(W, H);
    settings.mip_levels = 4;
    settings.mip_scan_dist = 64;
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
    const MOUNTAIN_STONE: u32 = 0x80_8a_82_7a;
    // Camera at z=101 looking down at mountain peak at z=100 (just
    // 1 voxel above camera). Mountain base at z=336 = depth 235.
    // Sample depths of mountain pixels: shallow ones (= chz=0 top)
    // vs deep ones (= chz=1 base via handoff).
    let mountain_depths: Vec<f32> = fb
        .iter()
        .zip(zb.iter())
        .filter_map(|(&p, &d)| if p == MOUNTAIN_STONE { Some(d) } else { None })
        .collect();
    let mountain_count = mountain_depths.len();
    // For camera at (239,298,101), chz=0 mountain voxels live at
    // world z=224..254 — distance to closest chz=0 voxel from the
    // camera is sqrt(19²+22²+(101-254)²) ≈ 156. chz=1 voxels at
    // z=256..336 — distance >= 158. Picking 200 as a clean
    // threshold for "definitely chz=1 base of the mountain".
    let chz1_count = mountain_depths.iter().filter(|&&d| d >= 200.0).count();
    // Count green hill pixels — chz=1 hills filling the ground
    // around the mountains via mid-render handoff.
    let hill_green = fb
        .iter()
        .filter(|&&p| {
            let r = (p >> 16) & 0xff;
            let g = (p >> 8) & 0xff;
            let b = p & 0xff;
            g > r && g > b && g > 60
        })
        .count();
    eprintln!(
        "user capture pose: mountain_total={mountain_count}, chz1(d>=200)={chz1_count}, hill_green={hill_green}"
    );
    // Three S4B.6.i fixes feed this test:
    //   1. `stamp_mountain_step_preserving_bedrock` keeps the chz=0
    //      bedrock placeholder intact under mountain steps that
    //      cross world z=255.
    //   2. `build_ground_extent_at_chz` caps hill spans at
    //      `z_max - 2` so insslab can't merge the hill with the
    //      bedrock placeholder into one "last slab".
    //   3. Column-step's chunk-XY swap reads `current_chunk_z`
    //      (= the chunk-z the mid-render handoff is currently
    //      in) instead of the pinned `camera_chunk_z` — so after
    //      the first handoff the DDA stays in chz=1 across XY
    //      crossings and reads the hills there.
    // Pre-fix: chz1_count = 87 (mountain bases barely visible),
    // hill_green ≈ 0 (hills entirely black). Post-fix: ~50k+ hill
    // pixels + the mountain bases continue uninterrupted.
    assert!(
        chz1_count > 30,
        "expected chz=1 base of the mountain (depth>=200) to render via mid-render handoff — got {chz1_count} pixels"
    );
    assert!(
        hill_green > 50_000,
        "expected ample chz=1 hill green pixels via mid-render handoff — got {hill_green}"
    );
}

/// S4B.6.j: render the three follow-up poses the user captured to
/// isolate the remaining bug.  Saves each framebuffer to /tmp so
/// the visual can be compared. Asserts nothing — purely diagnostic.
#[test]
#[ignore = "expensive diagnostic: ~3s scene build + 3 renders"]
fn stacked_demo_diagnostic_three_capture_poses() {
    std::env::set_var("ROXLAP_STACKED_GROUND", "1");
    let mut scene_and_camera = crate::scene::build_demo();
    std::env::remove_var("ROXLAP_STACKED_GROUND");
    let engine = Engine::new();
    let mut pool = ScratchPool::new(W, H, 32 * roxlap_scene::CHUNK_SIZE_XY);
    let sky = engine.sky_color();
    let sky_col_i = i32::from_ne_bytes(sky.to_ne_bytes());
    pool.set_skycast(sky_col_i, 0);
    let fog_col_i = i32::from_ne_bytes(engine.fog_color().to_ne_bytes());
    pool.set_fog(fog_col_i, engine.fog_max_scan_dist());
    pool.set_treat_z_max_as_air(true);

    let pixel_count = (W as usize) * (H as usize);
    let mut settings = OpticastSettings::for_oracle_framebuffer(W, H);
    settings.mip_levels = 4;
    settings.mip_scan_dist = 64;
    const MOUNTAIN_STONE: u32 = 0x80_8a_82_7a;
    let poses: [(&str, [f64; 3], f64, f64); 4] = [
        ("poseA_partial", [13.50, 169.45, 193.12], 3.0133, 1.2559),
        ("poseB_sky_only", [13.71, 171.04, 193.12], 3.0133, 1.2559),
        ("poseC_works", [21.50, 181.72, 177.21], 3.1908, 0.7559),
        // S4B.6.k: user reports the "usual render path" is broken
        // at this pose — far from any mountain, looking steeply
        // down. Should render hills + distant mountains cleanly.
        ("poseD_regression", [0.43, -4.61, 225.73], 1.2258, 1.3259),
    ];
    for (name, pos, yaw, pitch) in poses {
        scene_and_camera.cam_pos = pos;
        scene_and_camera.yaw = yaw;
        scene_and_camera.pitch = pitch;
        scene_and_camera.refresh_camera();
        let mut fb = vec![sky; pixel_count];
        let mut zb = vec![f32::INFINITY; pixel_count];
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
        let mut sky_count = 0usize;
        let mut hill_green = 0usize;
        let mut mountain_lit = 0usize;
        let mut mountain_dark = 0usize;
        for &p in &fb {
            let r = (p >> 16) & 0xff;
            let g = (p >> 8) & 0xff;
            let b = p & 0xff;
            if p == sky {
                sky_count += 1;
            } else if g > r && g > b && g > 60 {
                hill_green += 1;
            } else if r > 60 && (r as i32 - b as i32).abs() < 30 && (r as i32 - g as i32).abs() < 30
            {
                if r > 100 {
                    mountain_lit += 1;
                } else {
                    mountain_dark += 1;
                }
            }
        }
        let mountain_total = fb.iter().filter(|&&p| p == MOUNTAIN_STONE).count();
        let path = format!("/tmp/stacked-{name}.ppm");
        write_ppm(&path, &fb);
        // Bucket mountain (= gray) pixel depths into chz=0 vs
        // chz=1 ranges. For camera at z≈193 (pose A/B), the
        // chz boundary at world z=256 is depth ≈ 60-80 from
        // camera (depending on ray angle). Use 75 as the
        // cutoff: shallower = chz=0, deeper = chz=1.
        let mut mountain_chz0 = 0;
        let mut mountain_chz1 = 0;
        let cutoff = match name {
            "poseA_partial" | "poseB_sky_only" => 75.0_f32, // camera at z=193
            "poseC_works" => 90.0_f32,                      // camera at z=177
            _ => 100.0_f32,
        };
        for (&p, &d) in fb.iter().zip(zb.iter()) {
            let r = (p >> 16) & 0xff;
            let g = (p >> 8) & 0xff;
            let b = p & 0xff;
            let is_gray =
                r > 60 && (r as i32 - b as i32).abs() < 30 && (r as i32 - g as i32).abs() < 30;
            if is_gray && p != sky {
                if d < cutoff {
                    mountain_chz0 += 1;
                } else {
                    mountain_chz1 += 1;
                }
            }
        }
        eprintln!(
            "{name}: sky={sky_count} hill_green={hill_green} mountain_lit={mountain_lit} mountain_dark={mountain_dark} mountain_raw={mountain_total} mountain_chz0={mountain_chz0} mountain_chz1={mountain_chz1} (PPM at {path})"
        );
    }
}

/// S4B.6.k: ablation across mip configs to localise when the
/// triangular BLACK wedge appears at pose D. Runs the same camera
/// pose under (mip_levels=1, =2, =3, =4) × (mip_scan_dist=64),
/// counts exact-RGB-0 pixels in each, and writes PPMs for visual
/// diff. Useful to isolate whether the bug is in a specific mip
/// transition.
#[test]
#[ignore = "expensive diagnostic: builds full stacked demo + 4 renders (~10 s)"]
fn poseD_mip_ablation() {
    std::env::set_var("ROXLAP_STACKED_GROUND", "1");
    let mut scene_and_cam = crate::scene::build_demo();
    std::env::remove_var("ROXLAP_STACKED_GROUND");
    scene_and_cam.cam_pos = [0.43, -4.61, 225.73];
    scene_and_cam.yaw = 1.2258;
    scene_and_cam.pitch = 1.3259;
    scene_and_cam.refresh_camera();

    let engine = Engine::new();
    let mut pool = ScratchPool::new(W, H, 32 * roxlap_scene::CHUNK_SIZE_XY);
    let sky = engine.sky_color();
    let sky_col_i = i32::from_ne_bytes(sky.to_ne_bytes());
    pool.set_skycast(sky_col_i, 0);
    let fog_col_i = i32::from_ne_bytes(engine.fog_color().to_ne_bytes());
    pool.set_fog(fog_col_i, engine.fog_max_scan_dist());
    pool.set_treat_z_max_as_air(true);

    let pixel_count = (W as usize) * (H as usize);

    for &(mip_levels, mip_scan_dist, tag) in &[
        (1u32, 1024i32, "m1"),
        (2u32, 64i32, "m2_64"),
        (3u32, 64i32, "m3_64"),
        (4u32, 64i32, "m4_64"),
        (4u32, 128i32, "m4_128"),
        (4u32, 256i32, "m4_256"),
    ] {
        let mut fb = vec![sky; pixel_count];
        let mut zb = vec![f32::INFINITY; pixel_count];
        let mut settings = OpticastSettings::for_oracle_framebuffer(W, H);
        settings.mip_levels = mip_levels;
        settings.mip_scan_dist = mip_scan_dist;
        let _ = render_scene_composed(
            &mut fb,
            &mut zb,
            W as usize,
            W,
            H,
            &mut pool,
            &mut scene_and_cam.scene,
            &scene_and_cam.camera,
            &settings,
            sky,
            None,
        );
        // Count exact-RGB-0 pixels (the black wedge) and compute the
        // bounding box in screen space.
        let mut n_black = 0usize;
        let mut x_min = u32::MAX;
        let mut y_min = u32::MAX;
        let mut x_max = 0u32;
        let mut y_max = 0u32;
        for (i, &p) in fb.iter().enumerate() {
            let rgb = p & 0x00_ff_ff_ff;
            if rgb == 0 {
                n_black += 1;
                let x = (i as u32) % W;
                let y = (i as u32) / W;
                x_min = x_min.min(x);
                y_min = y_min.min(y);
                x_max = x_max.max(x);
                y_max = y_max.max(y);
            }
        }
        let path = format!("/tmp/poseD_ablate_{tag}.ppm");
        write_ppm(&path, &fb);
        eprintln!(
            "{tag} (mip_levels={mip_levels}, mip_scan_dist={mip_scan_dist}): \
             black_RGB0={n_black} bbox=[x{x_min}..{x_max}, y{y_min}..{y_max}] → {path}"
        );
    }
}

/// Quick FPS benchmark: render N frames from the spawn camera at
/// the full 32×32 ground + 4×6×1 ship, print average frame time.
/// Ignored by default — only run when investigating perf
/// regressions. Pool is sized for the full vsid=4096 ground and
/// uses rayon's default thread count.
#[test]
#[ignore = "expensive: builds full demo + renders N frames (~15-30 s)"]
fn bench_full_demo_render_fps() {
    use std::time::Instant;
    const N_FRAMES: usize = 30;

    // Mirror the live demo's constants so the bench numbers
    // reflect what users actually see.
    const DEMO_RENDER_THREADS: usize = 4;
    const DEMO_MAX_SCAN_DIST: i32 = 512;

    let mut scene_and_cam = crate::scene::build_demo();
    let mut engine = Engine::new();
    engine.set_fog(engine.sky_color(), DEMO_MAX_SCAN_DIST);
    // `BENCH_THREADS` / `BENCH_MAX_SCAN_DIST` env overrides let
    // the bench be swept from a shell loop for tuning.
    let n_threads = std::env::var("BENCH_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(DEMO_RENDER_THREADS)
        .max(1);
    let mut pool = ScratchPool::new_parallel(W, H, 32 * roxlap_scene::CHUNK_SIZE_XY, n_threads);
    let sky = engine.sky_color();
    let sky_col_i = i32::from_ne_bytes(sky.to_ne_bytes());
    pool.set_skycast(sky_col_i, 0);
    let fog_col_i = i32::from_ne_bytes(engine.fog_color().to_ne_bytes());
    pool.set_fog(fog_col_i, engine.fog_max_scan_dist());
    let treat_z_max_as_air = std::env::var("BENCH_TREAT_Z_MAX_AS_AIR")
        .ok()
        .map_or(true, |v| v == "1");
    pool.set_treat_z_max_as_air(treat_z_max_as_air);

    let pixel_count = (W as usize) * (H as usize);
    let mut fb = vec![sky; pixel_count];
    let mut zb = vec![f32::INFINITY; pixel_count];
    let mut settings = OpticastSettings::for_oracle_framebuffer(W, H);
    settings.max_scan_dist = std::env::var("BENCH_MAX_SCAN_DIST")
        .ok()
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(DEMO_MAX_SCAN_DIST);
    // S4B.5: chunks are mipped (`generate_mips(4)` in scene::bake_lightmode_1).
    // Sweep mip_scan_dist via BENCH_MIP_SCAN_DIST; set to a value ≥ max_scan_dist
    // to disable transitions (single-mip baseline).
    settings.mip_levels = std::env::var("BENCH_MIP_LEVELS")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(4);
    settings.mip_scan_dist = std::env::var("BENCH_MIP_SCAN_DIST")
        .ok()
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(64);

    // One warmup frame so cache populates + jit-y bits settle.
    let _ = render_scene_composed(
        &mut fb,
        &mut zb,
        W as usize,
        W,
        H,
        &mut pool,
        &mut scene_and_cam.scene,
        &scene_and_cam.camera,
        &settings,
        sky,
        None,
    );

    let t_start = Instant::now();
    for _ in 0..N_FRAMES {
        fb.fill(sky);
        zb.fill(f32::INFINITY);
        let _ = render_scene_composed(
            &mut fb,
            &mut zb,
            W as usize,
            W,
            H,
            &mut pool,
            &mut scene_and_cam.scene,
            &scene_and_cam.camera,
            &settings,
            sky,
            None,
        );
    }
    let elapsed = t_start.elapsed();
    let avg_ms = elapsed.as_secs_f64() * 1000.0 / N_FRAMES as f64;
    let fps = 1000.0 / avg_ms;
    let non_sky = fb.iter().filter(|&&p| p != sky).count();
    let non_sky_pct = 100.0 * non_sky as f64 / pixel_count as f64;
    eprintln!(
        "bench: {N_FRAMES} frames in {:.2} s — {avg_ms:.1} ms/frame ({fps:.1} FPS, {n_threads} threads, vsid=4096 ground + vsid=768 ship); non-sky pixels {non_sky}/{pixel_count} ({non_sky_pct:.1}%)",
        elapsed.as_secs_f64()
    );
}

/// User-reported "black wall around ship grid" at this pose.
/// Mip ablation: m1 = no mip transitions, m4 = full 4-level mips
/// (= the live demo's setting). If m1 renders clean and m4 shows
/// the wall, it's a multi-mip regression on the ship grid.
#[test]
#[ignore = "diagnostic — builds full demo + dumps PPM at the bug pose"]
fn dump_ship_black_wall_pose_m1() {
    dump_ship_black_wall_pose_at_mip(1, 1024, "m1");
}

#[test]
#[ignore = "diagnostic — builds full demo + dumps PPM at the bug pose"]
fn dump_ship_black_wall_pose_m2() {
    dump_ship_black_wall_pose_at_mip(2, 64, "m2");
}

#[test]
#[ignore = "diagnostic — builds full demo + dumps PPM at the bug pose"]
fn dump_ship_black_wall_pose_m3() {
    dump_ship_black_wall_pose_at_mip(3, 64, "m3");
}

#[test]
#[ignore = "diagnostic — builds full demo + dumps PPM at the bug pose"]
fn dump_ship_black_wall_pose() {
    dump_ship_black_wall_pose_at_mip(4, 64, "m4");
}

fn dump_ship_black_wall_pose_at_mip(mip_levels: u32, mip_scan_dist: i32, tag: &str) {
    let mut sc = crate::scene::build_demo();
    let cam = camera_for_yaw_pitch(
        [
            340.592_643_528_680_36,
            536.579_685_026_495_7,
            54.393_369_117_022_51,
        ],
        1.548_296_326_794_943_4,
        0.282_499_999_999_991_65,
    );

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
    let mut settings = OpticastSettings::for_oracle_framebuffer(W, H);
    settings.mip_levels = mip_levels;
    settings.mip_scan_dist = mip_scan_dist;
    settings.max_scan_dist = 512;
    let _ = render_scene_composed(
        &mut fb,
        &mut zb,
        W as usize,
        W,
        H,
        &mut pool,
        &mut sc.scene,
        &cam,
        &settings,
        sky,
        None,
    );
    let n_pure_black = fb.iter().filter(|&&p| (p & 0x00_ff_ff_ff) == 0).count();
    let path = format!("/tmp/ship_black_wall_pose_{tag}.ppm");
    write_ppm(&path, &fb);
    eprintln!("ship-black-wall-{tag}: pure-black pixels {n_pure_black}/{pixel_count} → {path}");
}

/// Variant 1: same pose with only the GROUND grid (no ship).
/// If the black wall disappears, the bug is in the ship grid.
#[test]
#[ignore = "diagnostic — builds ground-only at ship bug pose"]
fn dump_ship_black_wall_pose_ground_only() {
    use roxlap_scene::{GridTransform, Scene};
    let mut scene = Scene::new();
    let id = scene.add_grid(GridTransform::at(glam::DVec3::ZERO));
    let grid = scene.grid_mut(id).expect("ground grid present");
    crate::terrain::build_ground_extent(grid, 32, 32);
    // No lighting bake — the bug is geometric, not shaded.
    let cam = camera_for_yaw_pitch(
        [
            340.592_643_528_680_36,
            536.579_685_026_495_7,
            54.393_369_117_022_51,
        ],
        1.548_296_326_794_943_4,
        0.282_499_999_999_991_65,
    );

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
    let mut settings = OpticastSettings::for_oracle_framebuffer(W, H);
    settings.mip_levels = 4;
    settings.mip_scan_dist = 64;
    settings.max_scan_dist = 512;
    let _ = render_scene_composed(
        &mut fb, &mut zb, W as usize, W, H, &mut pool, &mut scene, &cam, &settings, sky, None,
    );
    let n_pure_black = fb.iter().filter(|&&p| (p & 0x00_ff_ff_ff) == 0).count();
    write_ppm("/tmp/ship_black_wall_pose_ground_only.ppm", &fb);
    eprintln!("ground-only: pure-black pixels {n_pure_black}/{pixel_count}");
}

/// Regression: OOB-XY camera at the ship-only mip-N render path
/// used to paint a 388k-pixel BLACK WALL around the saucer
/// (user-reported 2026-05-26). Root cause: `phase_remiporend`
/// reloaded `state.column` from the seed chunk's column_offsets
/// without checking `current_chunk_exists`. For OOB-XY cameras
/// the unbounded `wrapping_add_signed` of `ixy_sptr_col_idx`
/// across OOB chunks would eventually land in a DEEPER mip's
/// sub-table (= read 4-byte voxel records past the actual slab,
/// producing RGB=0 voxel records the bedrock-z guard couldn't
/// catch because the z byte didn't match `0xff >> gmipcnt`).
/// Fix: gate the column reload on `current_chunk_exists`.
#[test]
#[ignore = "expensive: builds the 4×6 ship + lighting bake + 6 mips (~1 s)"]
fn dump_ship_black_wall_pose_ship_only_mips() {
    use roxlap_scene::{GridTransform, Scene};
    let mut scene = Scene::new();
    let id = scene.add_grid(GridTransform::at(glam::DVec3::new(0.0, 500.0, -100.0)));
    let grid = scene.grid_mut(id).expect("ship grid present");
    crate::ship::build_ship(grid);
    crate::scene::bake_lightmode_1_pub(&mut scene);
    let cam = camera_for_yaw_pitch(
        [
            340.592_643_528_680_36,
            536.579_685_026_495_7,
            54.393_369_117_022_51,
        ],
        1.548_296_326_794_943_4,
        0.282_499_999_999_991_65,
    );

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
    let mut settings = OpticastSettings::for_oracle_framebuffer(W, H);
    settings.mip_levels = 4;
    settings.mip_scan_dist = 64;
    settings.max_scan_dist = 512;
    let _ = render_scene_composed(
        &mut fb, &mut zb, W as usize, W, H, &mut pool, &mut scene, &cam, &settings, sky, None,
    );
    let n_pure_black = fb.iter().filter(|&&p| (p & 0x00_ff_ff_ff) == 0).count();
    write_ppm("/tmp/ship_black_wall_pose_ship_only_mips.ppm", &fb);
    eprintln!("ship-only-mips: pure-black pixels {n_pure_black}/{pixel_count}");
    // Pre-fix: 388_130 pure-black pixels at this pose. The ship hull
    // is gray, accent orange, bridge blue — no legitimate pure-RGB-0
    // pixels exist; anything > a handful is a regression.
    assert!(
        n_pure_black < 100,
        "ship-only-mips at OOB-XY camera pose painted {n_pure_black} pure-black pixels — phase_remiporend column-reload regression (was 388k pre-fix)"
    );
}

/// Variant 2: same pose with only the SHIP grid (no ground).
/// Isolates the ship-grid render path.
#[test]
#[ignore = "diagnostic — builds ship-only at ship bug pose"]
fn dump_ship_black_wall_pose_ship_only() {
    use roxlap_scene::{GridTransform, Scene};
    let mut scene = Scene::new();
    let id = scene.add_grid(GridTransform::at(glam::DVec3::new(0.0, 500.0, -100.0)));
    let grid = scene.grid_mut(id).expect("ship grid present");
    crate::ship::build_ship(grid);
    let cam = camera_for_yaw_pitch(
        [
            340.592_643_528_680_36,
            536.579_685_026_495_7,
            54.393_369_117_022_51,
        ],
        1.548_296_326_794_943_4,
        0.282_499_999_999_991_65,
    );

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
    let mut settings = OpticastSettings::for_oracle_framebuffer(W, H);
    settings.mip_levels = 4;
    settings.mip_scan_dist = 64;
    settings.max_scan_dist = 512;
    let _ = render_scene_composed(
        &mut fb, &mut zb, W as usize, W, H, &mut pool, &mut scene, &cam, &settings, sky, None,
    );
    let n_pure_black = fb.iter().filter(|&&p| (p & 0x00_ff_ff_ff) == 0).count();
    write_ppm("/tmp/ship_black_wall_pose_ship_only.ppm", &fb);
    eprintln!("ship-only: pure-black pixels {n_pure_black}/{pixel_count}");
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

/// User-reported S4B.5 bedrock-mip-leak pose
/// (capture 2026-05-12, second report).
/// Looking +y near horizontal at the saucer floating above terrain.
/// Multi-mip renders a large BLACK region under the ship where the
/// terrain should be. Hypothesis: the ship chunk's mip-N bedrock
/// placeholder voxels (z=255) get colour-averaged into the mip-N
/// table and rendered as black voxels that `treat_z_max_as_air`
/// doesn't suppress (since it only checks z==255 of mip-0, not the
/// halved z=127 / 63 / 31 of mip-N).
const SHIP_BEDROCK_POS: [f64; 3] = [
    114.033_161_208_257_45,
    -39.266_620_412_447_05,
    51.049_406_147_596_84,
];
const SHIP_BEDROCK_YAW: f64 = 1.890_796_326_794_889_7;
const SHIP_BEDROCK_PITCH: f64 = 0.162_500_000_000_000_1;

#[test]
#[ignore = "expensive: builds full demo; dumps PPM for visual inspection of bedrock-mip-leak"]
fn dump_bedrock_mip_leak_pose() {
    use roxlap_scene::CHUNK_SIZE_XY;

    let mut sc = crate::scene::build_demo();
    let cam = camera_for_yaw_pitch(SHIP_BEDROCK_POS, SHIP_BEDROCK_YAW, SHIP_BEDROCK_PITCH);

    let mut engine = Engine::new();
    engine.set_fog(engine.sky_color(), 512);
    let mut pool = ScratchPool::new_parallel(W, H, 32 * CHUNK_SIZE_XY, 4);
    let sky = engine.sky_color();
    let sky_col_i = i32::from_ne_bytes(sky.to_ne_bytes());
    pool.set_skycast(sky_col_i, 0);
    let fog_col_i = i32::from_ne_bytes(engine.fog_color().to_ne_bytes());
    pool.set_fog(fog_col_i, engine.fog_max_scan_dist());
    pool.set_treat_z_max_as_air(true);

    let pixel_count = (W as usize) * (H as usize);
    let mut settings = OpticastSettings::for_oracle_framebuffer(W, H);
    settings.max_scan_dist = 512;

    let mut fb_base = vec![sky; pixel_count];
    let mut zb_base = vec![f32::INFINITY; pixel_count];
    settings.mip_levels = 1;
    settings.mip_scan_dist = 4;
    let _ = render_scene_composed(
        &mut fb_base,
        &mut zb_base,
        W as usize,
        W,
        H,
        &mut pool,
        &mut sc.scene,
        &cam,
        &settings,
        sky,
        None,
    );
    let base_non_sky = fb_base.iter().filter(|&&p| p != sky).count();
    let black: u32 = 0xff_00_00_00;
    let base_black = fb_base.iter().filter(|&&p| p == black).count();
    write_ppm("/tmp/scene-demo-ship-pose-baseline.ppm", &fb_base);
    eprintln!(
        "single-mip ship pose: non-sky {base_non_sky}, exact-black {base_black} / {pixel_count}"
    );

    let mut fb_mips = vec![sky; pixel_count];
    let mut zb_mips = vec![f32::INFINITY; pixel_count];
    settings.mip_levels = 4;
    settings.mip_scan_dist = 64;
    let _ = render_scene_composed(
        &mut fb_mips,
        &mut zb_mips,
        W as usize,
        W,
        H,
        &mut pool,
        &mut sc.scene,
        &cam,
        &settings,
        sky,
        None,
    );
    let mips_non_sky = fb_mips.iter().filter(|&&p| p != sky).count();
    let mips_black = fb_mips.iter().filter(|&&p| p == black).count();
    write_ppm("/tmp/scene-demo-ship-pose-mips.ppm", &fb_mips);
    eprintln!(
        "multi-mip ship pose:  non-sky {mips_non_sky}, exact-black {mips_black} / {pixel_count}"
    );

    assert!(base_non_sky > 0);
    assert!(mips_non_sky > 0);
    // Regression: multi-mip mustn't introduce a big black region that
    // wasn't in the single-mip baseline.
    let extra_black = mips_black.saturating_sub(base_black);
    let extra_pct = 100.0 * extra_black as f64 / pixel_count as f64;
    assert!(
        extra_pct < 1.0,
        "multi-mip leaks {extra_black} extra exact-black pixels ({extra_pct:.2}%) vs baseline ({base_black}) — bedrock-mip-leak?"
    );
}

/// User-reported S4B.5 multi-chunk + multi-mip artifact pose
/// (capture 2026-05-12, see `roxlap-scene-capture.txt`).
/// Pre-fix: phantom green strips at chunk boundaries because the
/// mip-N column-step pinned to the camera's chunk and wrapped its
/// data across neighbouring chunks. Post-fix: `cx_mip`/`cy_mip`
/// track mip-N voxel coords and trigger proper chunk-XY swaps in
/// the mip-N branch of `phase_after_delete_kept_presync`.
///
/// Renders the full 32×32 ground + ship at multi-mip msd=64 and
/// dumps PPM to /tmp/. Manual visual inspection — assertion is
/// only "non-zero non-sky pixels", correctness is eyeballed.
#[test]
#[ignore = "expensive: builds full 32×32 demo (~3-5 s); use --ignored to dump capture pose"]
fn dump_chunk_tearing_capture_pose() {
    use roxlap_scene::CHUNK_SIZE_XY;

    const CAP_POS: [f64; 3] = [
        -474.287_937_724_851_3,
        -464.324_076_691_379_5,
        -57.843_548_692_727_964,
    ];
    const CAP_YAW: f64 = -5.694_203_673_205_082;
    const CAP_PITCH: f64 = 1.282_499_999_999_981_8;

    let mut sc = crate::scene::build_demo();
    let cam = camera_for_yaw_pitch(CAP_POS, CAP_YAW, CAP_PITCH);
    eprintln!("capture cam right: {:?}", cam.right);
    eprintln!("capture cam down: {:?}", cam.down);
    eprintln!("capture cam forward: {:?}", cam.forward);

    let mut engine = Engine::new();
    engine.set_fog(engine.sky_color(), 512);
    let mut pool = ScratchPool::new_parallel(W, H, 32 * CHUNK_SIZE_XY, 4);
    let sky = engine.sky_color();
    let sky_col_i = i32::from_ne_bytes(sky.to_ne_bytes());
    pool.set_skycast(sky_col_i, 0);
    let fog_col_i = i32::from_ne_bytes(engine.fog_color().to_ne_bytes());
    pool.set_fog(fog_col_i, engine.fog_max_scan_dist());
    pool.set_treat_z_max_as_air(true);

    let pixel_count = (W as usize) * (H as usize);
    let mut fb = vec![sky; pixel_count];
    let mut zb = vec![f32::INFINITY; pixel_count];
    let mut settings = OpticastSettings::for_oracle_framebuffer(W, H);
    settings.max_scan_dist = 512;
    // First render with mips on (the case the user reported broken).
    settings.mip_levels = 4;
    settings.mip_scan_dist = 64;
    let _ = render_scene_composed(
        &mut fb,
        &mut zb,
        W as usize,
        W,
        H,
        &mut pool,
        &mut sc.scene,
        &cam,
        &settings,
        sky,
        None,
    );
    let non_sky_mips = fb.iter().filter(|&&p| p != sky).count();
    write_ppm("/tmp/scene-demo-capture-pose-mips.ppm", &fb);
    eprintln!(
        "wrote /tmp/scene-demo-capture-pose-mips.ppm — mips=4 msd=64 non-sky {non_sky_mips}/{pixel_count} ({:.1}%)",
        100.0 * non_sky_mips as f64 / pixel_count as f64
    );

    // Second render without mips (baseline / ground truth).
    let mut fb2 = vec![sky; pixel_count];
    let mut zb2 = vec![f32::INFINITY; pixel_count];
    settings.mip_levels = 1;
    settings.mip_scan_dist = 4;
    let _ = render_scene_composed(
        &mut fb2,
        &mut zb2,
        W as usize,
        W,
        H,
        &mut pool,
        &mut sc.scene,
        &cam,
        &settings,
        sky,
        None,
    );
    let non_sky_base = fb2.iter().filter(|&&p| p != sky).count();
    write_ppm("/tmp/scene-demo-capture-pose-baseline.ppm", &fb2);
    eprintln!(
        "wrote /tmp/scene-demo-capture-pose-baseline.ppm — single-mip non-sky {non_sky_base}/{pixel_count} ({:.1}%)",
        100.0 * non_sky_base as f64 / pixel_count as f64
    );

    // Also render at SPAWN pose (single-mip + multi-mip) — a known-
    // working pose. Compare counts to verify multi-mip doesn't drift
    // wildly from single-mip there.
    let mut fb_s_base = vec![sky; pixel_count];
    let mut zb_s_base = vec![f32::INFINITY; pixel_count];
    let mut fb_s_mips = vec![sky; pixel_count];
    let mut zb_s_mips = vec![f32::INFINITY; pixel_count];
    let spawn_cam = sc.camera;
    settings.mip_levels = 1;
    settings.mip_scan_dist = 4;
    let _ = render_scene_composed(
        &mut fb_s_base,
        &mut zb_s_base,
        W as usize,
        W,
        H,
        &mut pool,
        &mut sc.scene,
        &spawn_cam,
        &settings,
        sky,
        None,
    );
    settings.mip_levels = 4;
    settings.mip_scan_dist = 64;
    let _ = render_scene_composed(
        &mut fb_s_mips,
        &mut zb_s_mips,
        W as usize,
        W,
        H,
        &mut pool,
        &mut sc.scene,
        &spawn_cam,
        &settings,
        sky,
        None,
    );
    let spawn_base = fb_s_base.iter().filter(|&&p| p != sky).count();
    let spawn_mips = fb_s_mips.iter().filter(|&&p| p != sky).count();
    eprintln!("spawn pose: single-mip non-sky {spawn_base}/{pixel_count}");
    eprintln!("spawn pose: multi-mip non-sky {spawn_mips}/{pixel_count}");
    write_ppm("/tmp/scene-demo-spawn-pose-baseline.ppm", &fb_s_base);
    write_ppm("/tmp/scene-demo-spawn-pose-mips.ppm", &fb_s_mips);

    assert!(non_sky_mips > 0, "capture pose rendered all-sky");
    assert!(spawn_base > 0, "spawn pose single-mip rendered all-sky");
    assert!(spawn_mips > 0, "spawn pose multi-mip rendered all-sky");
    // Multi-mip typically renders MORE non-sky pixels than single-mip
    // because mip-N rays reach farther into terrain without fog cutoff.
    // We only sanity-check the non-zero pixel count + visually inspect
    // the PPMs to confirm no chunk-edge tearing returns.
    assert!(
        spawn_mips >= spawn_base,
        "spawn-pose multi-mip rendered FEWER pixels ({spawn_mips}) than single-mip ({spawn_base}) — chunk-tearing regression?"
    );
}

/// User-reported S4B.5 green-beam artifact at deep scan_dist
/// (capture 2026-05-12). At pose (370.70, 47.60, 138.17) with
/// scan_dist > 700, four thin green columns rise from terrain into
/// the sky along the world-X / world-Y axes. Dumps PPMs at three
/// scan distances so the threshold is visible across them.
const BEAM_POS: [f64; 3] = [
    370.703_458_908_366_4,
    47.604_355_604_949_27,
    138.166_228_693_474_72,
];
const BEAM_YAW: f64 = 2.355_796_326_794_885_6;
const BEAM_PITCH: f64 = -0.490_000_000_000_002_6;

#[test]
#[ignore = "expensive: builds full demo; dumps green-beam-artifact PPMs"]
fn dump_green_beam_pose() {
    use roxlap_core::sky::Sky;
    use roxlap_scene::CHUNK_SIZE_XY;

    fn blue_sky() -> Sky {
        Sky::blue_gradient()
    }

    let mut sc = crate::scene::build_demo();
    let cam = camera_for_yaw_pitch(BEAM_POS, BEAM_YAW, BEAM_PITCH);
    let engine = Engine::new();
    let sky_tex = blue_sky();
    let mut pool = ScratchPool::new_parallel(W, H, 32 * CHUNK_SIZE_XY, 4);
    let sky = engine.sky_color();
    let sky_col_i = i32::from_ne_bytes(sky.to_ne_bytes());
    pool.set_skycast(sky_col_i, 0);
    pool.set_treat_z_max_as_air(true);

    let pixel_count = (W as usize) * (H as usize);
    let mut settings = OpticastSettings::for_oracle_framebuffer(W, H);
    settings.mip_levels = 6;
    settings.mip_scan_dist = 64;

    settings.max_scan_dist = 2047;
    settings.mip_levels = 6;
    settings.mip_scan_dist = 64;

    // (a) ground-only + (b) ship-only at the full config.
    let mut ground_only = Scene::new();
    let id_g = ground_only.add_grid(GridTransform::at(DVec3::ZERO));
    terrain::build_ground(ground_only.grid_mut(id_g).expect("ground"));
    crate::scene::bake_lightmode_1_pub(&mut ground_only);
    let mut fb_g = vec![sky; pixel_count];
    let mut zb_g = vec![f32::INFINITY; pixel_count];
    let _ = render_scene_composed(
        &mut fb_g,
        &mut zb_g,
        W as usize,
        W,
        H,
        &mut pool,
        &mut ground_only,
        &cam,
        &settings,
        sky,
        Some(&sky_tex),
    );
    let g_grass = fb_g
        .iter()
        .filter(|&&p| {
            let r = ((p >> 16) & 0xff) as i32;
            let g = ((p >> 8) & 0xff) as i32;
            let b = (p & 0xff) as i32;
            g > r + 30 && g > b + 30 && (r + g + b) > 100
        })
        .count();
    write_ppm("/tmp/scene-demo-beam-ground-only.ppm", &fb_g);
    eprintln!("beam pose ground-only: grass {g_grass}");

    let mut ship_only = Scene::new();
    let id_s = ship_only.add_grid(GridTransform::at(DVec3::new(0.0, 500.0, -100.0)));
    crate::ship::build_ship(ship_only.grid_mut(id_s).expect("ship"));
    crate::scene::bake_lightmode_1_pub(&mut ship_only);
    let mut fb_s = vec![sky; pixel_count];
    let mut zb_s = vec![f32::INFINITY; pixel_count];
    let _ = render_scene_composed(
        &mut fb_s,
        &mut zb_s,
        W as usize,
        W,
        H,
        &mut pool,
        &mut ship_only,
        &cam,
        &settings,
        sky,
        Some(&sky_tex),
    );
    let s_grass = fb_s
        .iter()
        .filter(|&&p| {
            let r = ((p >> 16) & 0xff) as i32;
            let g = ((p >> 8) & 0xff) as i32;
            let b = (p & 0xff) as i32;
            g > r + 30 && g > b + 30 && (r + g + b) > 100
        })
        .count();
    write_ppm("/tmp/scene-demo-beam-ship-only.ppm", &fb_s);
    eprintln!("beam pose ship-only:   grass {s_grass}");

    settings.mip_levels = 6;
    settings.mip_scan_dist = 64;
    for msd in [256_i32, 384, 448, 512, 576, 640, 700, 1024] {
        // `msd` is the SCAN_DIST sweep here, not mip_scan_dist. Local
        // rename to keep the inner loop body terse.
        settings.max_scan_dist = msd;
        let mut fb = vec![sky; pixel_count];
        let mut zb = vec![f32::INFINITY; pixel_count];
        let _ = render_scene_composed(
            &mut fb,
            &mut zb,
            W as usize,
            W,
            H,
            &mut pool,
            &mut sc.scene,
            &cam,
            &settings,
            sky,
            Some(&sky_tex),
        );
        // Count pixels with the grass-green signature
        // (R+G+B in [200, 380] AND G > R+30 AND G > B+30) — the
        // beam pixels are pure-ish grass green so this is a cheap
        // proxy for "beam visible".
        let mut grass = 0usize;
        for &px in &fb {
            let r = ((px >> 16) & 0xff) as i32;
            let g = ((px >> 8) & 0xff) as i32;
            let b = (px & 0xff) as i32;
            if g > r + 30 && g > b + 30 && (r + g + b) > 100 {
                grass += 1;
            }
        }
        let path = format!("/tmp/scene-demo-beam-scan{msd}.ppm");
        write_ppm(&path, &fb);
        eprintln!("ml=6 scan={msd}: grass {grass} / {pixel_count}, wrote {path}");
    }
}

/// Same as `dump_green_beam_pose_diff` but at the demo's SPAWN
/// pose — used to verify a candidate fix doesn't over-cull
/// legitimate terrain. Spawn pose looks at lots of close + far
/// terrain so it stress-tests "is the guard culling real pixels?"
#[test]
#[ignore = "expensive: builds full demo; reports beam pixel coords at spawn"]
fn dump_spawn_pose_diff() {
    use roxlap_scene::CHUNK_SIZE_XY;

    let mut sc = crate::scene::build_demo();
    let cam = sc.camera;
    let engine = Engine::new();
    let mut pool = ScratchPool::new_parallel(W, H, 32 * CHUNK_SIZE_XY, 4);
    let sky = engine.sky_color();
    let sky_col_i = i32::from_ne_bytes(sky.to_ne_bytes());
    pool.set_skycast(sky_col_i, 0);
    pool.set_treat_z_max_as_air(true);

    let pixel_count = (W as usize) * (H as usize);
    let mut settings = OpticastSettings::for_oracle_framebuffer(W, H);
    settings.max_scan_dist = 512;

    let mut fb_z = vec![sky; pixel_count];
    let mut zb_z = vec![f32::INFINITY; pixel_count];
    settings.mip_levels = 1;
    settings.mip_scan_dist = 4;
    let _ = render_scene_composed(
        &mut fb_z,
        &mut zb_z,
        W as usize,
        W,
        H,
        &mut pool,
        &mut sc.scene,
        &cam,
        &settings,
        sky,
        None,
    );

    let mut fb_m = vec![sky; pixel_count];
    let mut zb_m = vec![f32::INFINITY; pixel_count];
    settings.mip_levels = 4;
    settings.mip_scan_dist = 64;
    let _ = render_scene_composed(
        &mut fb_m,
        &mut zb_m,
        W as usize,
        W,
        H,
        &mut pool,
        &mut sc.scene,
        &cam,
        &settings,
        sky,
        None,
    );

    let is_grass = |p: u32| -> bool {
        let r = ((p >> 16) & 0xff) as i32;
        let g = ((p >> 8) & 0xff) as i32;
        let b = (p & 0xff) as i32;
        g > r + 30 && g > b + 30 && (r + g + b) > 100
    };

    let mut beam = 0usize;
    let mut lost_terrain = 0usize;
    let mut total_non_sky_ml1 = 0usize;
    let mut total_non_sky_ml4 = 0usize;
    for i in 0..pixel_count {
        if fb_z[i] != sky {
            total_non_sky_ml1 += 1;
        }
        if fb_m[i] != sky {
            total_non_sky_ml4 += 1;
        }
        if fb_z[i] == sky && is_grass(fb_m[i]) {
            beam += 1;
        }
        if is_grass(fb_z[i]) && fb_m[i] == sky {
            lost_terrain += 1;
        }
    }
    eprintln!("SPAWN BEAM PIXELS (sky in ml=1, grass in ml=4): {beam}");
    eprintln!("SPAWN LOST TERRAIN (grass in ml=1, sky in ml=4): {lost_terrain}");
    eprintln!("SPAWN non-sky: ml=1 {total_non_sky_ml1}, ml=4 {total_non_sky_ml4}");
}

/// Precise beam-pixel finder: render the SAME scene + camera with
/// `mip_levels=1` (baseline = no transitions) and `mip_levels=6
/// mip_scan_dist=64` (live config). A pixel that is SKY in the
/// baseline frame but has terrain colour in the multi-mip frame is
/// by definition a beam (multi-mip painted terrain where none should
/// be). Reports a precise count + bounding box + sample coords.
///
/// Catches beams in any screen quadrant — the previous "grass"
/// metric counted both legitimate far-terrain (correct) and beams
/// (bug) without distinguishing.
#[test]
#[ignore = "expensive: builds full demo; reports beam pixel coords"]
fn dump_green_beam_pose_diff() {
    use roxlap_scene::CHUNK_SIZE_XY;

    let mut sc = crate::scene::build_demo();
    let cam = camera_for_yaw_pitch(BEAM_POS, BEAM_YAW, BEAM_PITCH);
    let engine = Engine::new();
    let mut pool = ScratchPool::new_parallel(W, H, 32 * CHUNK_SIZE_XY, 4);
    let sky = engine.sky_color();
    let sky_col_i = i32::from_ne_bytes(sky.to_ne_bytes());
    pool.set_skycast(sky_col_i, 0);
    pool.set_treat_z_max_as_air(true);

    let pixel_count = (W as usize) * (H as usize);
    let mut settings = OpticastSettings::for_oracle_framebuffer(W, H);
    let scan = std::env::var("BEAM_SCAN_DIST")
        .ok()
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(1024);
    let msd_env = std::env::var("BEAM_MSD")
        .ok()
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(64);
    settings.max_scan_dist = scan;

    // (Z) baseline: mip_levels=1 — no transitions.
    let mut fb_z = vec![sky; pixel_count];
    let mut zb_z = vec![f32::INFINITY; pixel_count];
    settings.mip_levels = 1;
    settings.mip_scan_dist = 4;
    let _ = render_scene_composed(
        &mut fb_z,
        &mut zb_z,
        W as usize,
        W,
        H,
        &mut pool,
        &mut sc.scene,
        &cam,
        &settings,
        sky,
        None,
    );
    write_ppm("/tmp/scene-demo-beam-diff-ml1.ppm", &fb_z);

    // (M) multi-mip: live demo config.
    let mut fb_m = vec![sky; pixel_count];
    let mut zb_m = vec![f32::INFINITY; pixel_count];
    settings.mip_levels = 6;
    settings.mip_scan_dist = msd_env;
    let _ = render_scene_composed(
        &mut fb_m,
        &mut zb_m,
        W as usize,
        W,
        H,
        &mut pool,
        &mut sc.scene,
        &cam,
        &settings,
        sky,
        None,
    );
    write_ppm("/tmp/scene-demo-beam-diff-ml6.ppm", &fb_m);
    eprintln!("config: scan_dist={scan} mip_scan_dist={msd_env}");

    // Diff: pixels SKY in (Z) but grass-green in (M).
    let is_grass = |p: u32| -> bool {
        let r = ((p >> 16) & 0xff) as i32;
        let g = ((p >> 8) & 0xff) as i32;
        let b = (p & 0xff) as i32;
        g > r + 30 && g > b + 30 && (r + g + b) > 100
    };

    let mut beam_pixels = Vec::new();
    for y in 0..H {
        for x in 0..W {
            let i = (y * W + x) as usize;
            if fb_z[i] == sky && is_grass(fb_m[i]) {
                beam_pixels.push((x, y));
            }
        }
    }
    // ALSO count "lost terrain" pixels: pixels that are terrain in
    // ml=1 but SKY in ml=6 — these would be legitimate terrain
    // pixels that a too-aggressive guard culled.
    let mut lost_terrain = 0usize;
    let is_grass = |p: u32| -> bool {
        let r = ((p >> 16) & 0xff) as i32;
        let g = ((p >> 8) & 0xff) as i32;
        let b = (p & 0xff) as i32;
        g > r + 30 && g > b + 30 && (r + g + b) > 100
    };
    let mut total_grass_ml1 = 0usize;
    let mut total_grass_ml6 = 0usize;
    for i in 0..pixel_count {
        if is_grass(fb_z[i]) {
            total_grass_ml1 += 1;
        }
        if is_grass(fb_m[i]) {
            total_grass_ml6 += 1;
        }
        if is_grass(fb_z[i]) && fb_m[i] == sky {
            lost_terrain += 1;
        }
    }
    eprintln!(
        "BEAM PIXELS (sky in ml=1, grass in ml=6): {} / {pixel_count}",
        beam_pixels.len()
    );
    eprintln!("LOST TERRAIN (grass in ml=1, sky in ml=6): {lost_terrain} / {pixel_count}");
    eprintln!("total grass ml=1: {total_grass_ml1}; total grass ml=6: {total_grass_ml6}");
    if !beam_pixels.is_empty() {
        let xmin = beam_pixels.iter().map(|&(x, _)| x).min().unwrap_or(0);
        let xmax = beam_pixels.iter().map(|&(x, _)| x).max().unwrap_or(0);
        let ymin = beam_pixels.iter().map(|&(_, y)| y).min().unwrap_or(0);
        let ymax = beam_pixels.iter().map(|&(_, y)| y).max().unwrap_or(0);
        eprintln!("  bbox: x=[{xmin}..{xmax}], y=[{ymin}..{ymax}]");
        // Histogram of x-coordinates (= which screen column the beam
        // is in). Beams are vertical → cluster around specific x.
        let mut xhist = vec![0usize; W as usize];
        for (x, _) in &beam_pixels {
            xhist[*x as usize] += 1;
        }
        // Print x columns with >=5 beam pixels — these are the
        // columns the beam runs through.
        let mut beam_cols: Vec<(u32, usize)> = (0..W as usize)
            .filter(|i| xhist[*i] >= 5)
            .map(|i| (i as u32, xhist[i]))
            .collect();
        beam_cols.sort_by_key(|&(_, c)| std::cmp::Reverse(c));
        eprintln!("  top-10 beam columns (x, count):");
        for (x, c) in beam_cols.iter().take(10) {
            eprintln!("    x={x} count={c}");
        }
        // First 3 beam pixels (sample).
        for (x, y) in beam_pixels.iter().take(3) {
            eprintln!("  sample: ({x}, {y})");
        }
    }
}

/// Visual check: render the demo's spawn pose with the live
/// demo's settings (fog OFF, checkerboard sky ON, multi-mip ON) so
/// the multi-mip LOD bands are visible against a patterned sky.
/// Dumps PPM only; no assertion beyond non-empty.
#[test]
#[ignore = "expensive: builds full demo; dumps showcase PPM"]
fn dump_showcase_pose_with_skybox() {
    use roxlap_core::sky::Sky;
    use roxlap_scene::CHUNK_SIZE_XY;

    // Checkerboard sky helper — duplicated from main.rs because the
    // bin's helper isn't reachable from the test crate.
    fn checker_sky() -> Sky {
        const W: u32 = 64;
        const H: u32 = 256;
        const TILE_X: u32 = 8;
        const TILE_Y: u32 = 16;
        let mut pixels = Vec::with_capacity((W * H) as usize);
        for y in 0..H {
            for x in 0..W {
                let (rb, gb, bb): (u32, u32, u32) = match (y * 4) / H {
                    0 => (0xe0, 0x40, 0x40),
                    1 => (0x40, 0xe0, 0x40),
                    2 => (0x40, 0x40, 0xe0),
                    _ => (0xe0, 0xd0, 0x40),
                };
                let dark = ((x / TILE_X) + (y / TILE_Y)) & 1 == 1;
                let (r, g, b) = if dark {
                    (rb / 4, gb / 4, bb / 4)
                } else {
                    (rb, gb, bb)
                };
                let (r, g, b) = if x == 0 {
                    (0xff, 0xff, 0xff)
                } else if x == W - 1 {
                    (0, 0, 0)
                } else {
                    (r, g, b)
                };
                #[allow(clippy::cast_possible_wrap)]
                let px = ((0x80u32 << 24) | (r << 16) | (g << 8) | b) as i32;
                pixels.push(px);
            }
        }
        Sky::from_pixels(pixels, W, H)
    }

    let mut sc = crate::scene::build_demo();
    let cam = sc.camera;
    let engine = Engine::new();
    let sky_tex = checker_sky();
    let mut pool = ScratchPool::new_parallel(W, H, 32 * CHUNK_SIZE_XY, 4);
    let sky = engine.sky_color();
    let sky_col_i = i32::from_ne_bytes(sky.to_ne_bytes());
    pool.set_skycast(sky_col_i, 0);
    // No fog.
    pool.set_treat_z_max_as_air(true);

    let pixel_count = (W as usize) * (H as usize);
    let mut fb = vec![sky; pixel_count];
    let mut zb = vec![f32::INFINITY; pixel_count];
    let mut settings = OpticastSettings::for_oracle_framebuffer(W, H);
    settings.max_scan_dist = 512;
    settings.mip_levels = 4;
    settings.mip_scan_dist = 64;
    let _ = render_scene_composed(
        &mut fb,
        &mut zb,
        W as usize,
        W,
        H,
        &mut pool,
        &mut sc.scene,
        &cam,
        &settings,
        sky,
        Some(&sky_tex),
    );
    let non_sky = fb.iter().filter(|&&p| p != sky).count();
    eprintln!("showcase pose: non-sky pixels {non_sky}/{pixel_count}");
    write_ppm("/tmp/scene-demo-showcase.ppm", &fb);
    assert!(non_sky > 0);
}

/// User-reported S4B.5 thin-black-ring artifact pose under the
/// saucer (capture 2026-05-12, third report). The user noted these
/// only appear under the ship — terrain rendered alone is clean.
/// Hypothesis: the ship-grid's mip-N data has dark voxels somewhere
/// (averaged from the saucer edge meeting air?) that win the
/// z-buffer composition against the ground-grid's terrain pixels.
///
/// Renders three frames: (1) ship grid alone at mip-N; (2) ground
/// grid alone at mip-N; (3) full composed scene. Compares
/// composed-mips with composed-single-mip exact-black drift.
const RING_POS: [f64; 3] = [
    65.350_478_161_762_95,
    100.183_671_604_502_34,
    43.180_526_662_698_51,
];
const RING_YAW: f64 = 1.715_796_326_794_896_6;
const RING_PITCH: f64 = 0.879_999_999_999_993_2;

#[test]
#[ignore = "expensive: builds full demo; dumps PPMs for thin-black-ring inspection"]
fn dump_ring_artifact_pose() {
    use roxlap_scene::CHUNK_SIZE_XY;

    let mut sc = crate::scene::build_demo();
    let cam = camera_for_yaw_pitch(RING_POS, RING_YAW, RING_PITCH);

    let mut engine = Engine::new();
    engine.set_fog(engine.sky_color(), 512);
    let mut pool = ScratchPool::new_parallel(W, H, 32 * CHUNK_SIZE_XY, 4);
    let sky = engine.sky_color();
    let sky_col_i = i32::from_ne_bytes(sky.to_ne_bytes());
    pool.set_skycast(sky_col_i, 0);
    let fog_col_i = i32::from_ne_bytes(engine.fog_color().to_ne_bytes());
    pool.set_fog(fog_col_i, engine.fog_max_scan_dist());
    pool.set_treat_z_max_as_air(true);

    let pixel_count = (W as usize) * (H as usize);
    let mut settings = OpticastSettings::for_oracle_framebuffer(W, H);
    settings.max_scan_dist = 512;

    // (1) Composed scene, single-mip
    let mut fb_base = vec![sky; pixel_count];
    let mut zb_base = vec![f32::INFINITY; pixel_count];
    settings.mip_levels = 1;
    settings.mip_scan_dist = 4;
    let _ = render_scene_composed(
        &mut fb_base,
        &mut zb_base,
        W as usize,
        W,
        H,
        &mut pool,
        &mut sc.scene,
        &cam,
        &settings,
        sky,
        None,
    );
    let base_non_sky = fb_base.iter().filter(|&&p| p != sky).count();
    write_ppm("/tmp/scene-demo-ring-baseline.ppm", &fb_base);
    eprintln!("ring pose single-mip composed: non-sky {base_non_sky}");

    // (2) Composed scene, multi-mip
    let mut fb_mips = vec![sky; pixel_count];
    let mut zb_mips = vec![f32::INFINITY; pixel_count];
    settings.mip_levels = 4;
    settings.mip_scan_dist = 64;
    let _ = render_scene_composed(
        &mut fb_mips,
        &mut zb_mips,
        W as usize,
        W,
        H,
        &mut pool,
        &mut sc.scene,
        &cam,
        &settings,
        sky,
        None,
    );
    let mips_non_sky = fb_mips.iter().filter(|&&p| p != sky).count();
    write_ppm("/tmp/scene-demo-ring-mips.ppm", &fb_mips);
    eprintln!("ring pose multi-mip composed: non-sky {mips_non_sky}");

    // Count "very dark" pixels (R+G+B < 60) — the artifact is dark
    // gray/black rings.
    let dark_count = |fb: &[u32]| -> usize {
        fb.iter()
            .filter(|&&p| {
                let r = (p >> 16) & 0xff;
                let g = (p >> 8) & 0xff;
                let b = p & 0xff;
                r + g + b < 60
            })
            .count()
    };
    let base_dark = dark_count(&fb_base);
    let mips_dark = dark_count(&fb_mips);
    eprintln!("ring pose dark pixels (r+g+b<60): single-mip={base_dark}, multi-mip={mips_dark}");

    assert!(mips_non_sky > 0);

    // (3+4) Isolate by grid: render ground-only and ship-only scenes
    // at multi-mip to determine which grid contributes the dark rings.
    let mut ground_only = Scene::new();
    let id_g = ground_only.add_grid(GridTransform::at(DVec3::ZERO));
    terrain::build_ground(ground_only.grid_mut(id_g).expect("ground"));
    // Apply lighting + mips just like build_demo does:
    crate::scene::bake_lightmode_1_pub(&mut ground_only);

    let mut ship_only = Scene::new();
    let id_s = ship_only.add_grid(GridTransform::at(DVec3::new(0.0, 500.0, -100.0)));
    crate::ship::build_ship(ship_only.grid_mut(id_s).expect("ship"));
    crate::scene::bake_lightmode_1_pub(&mut ship_only);

    let mut fb_g = vec![sky; pixel_count];
    let mut zb_g = vec![f32::INFINITY; pixel_count];
    settings.mip_levels = 4;
    settings.mip_scan_dist = 64;
    let _ = render_scene_composed(
        &mut fb_g,
        &mut zb_g,
        W as usize,
        W,
        H,
        &mut pool,
        &mut ground_only,
        &cam,
        &settings,
        sky,
        None,
    );
    let g_non_sky = fb_g.iter().filter(|&&p| p != sky).count();
    write_ppm("/tmp/scene-demo-ring-ground-only.ppm", &fb_g);

    let mut fb_s = vec![sky; pixel_count];
    let mut zb_s = vec![f32::INFINITY; pixel_count];
    let _ = render_scene_composed(
        &mut fb_s,
        &mut zb_s,
        W as usize,
        W,
        H,
        &mut pool,
        &mut ship_only,
        &cam,
        &settings,
        sky,
        None,
    );
    let s_non_sky = fb_s.iter().filter(|&&p| p != sky).count();
    write_ppm("/tmp/scene-demo-ring-ship-only.ppm", &fb_s);
    eprintln!("ground-only mips: {g_non_sky} non-sky; ship-only mips: {s_non_sky} non-sky");

    // Regression: multi-mip's dark-pixel count should be close to
    // single-mip's. Big increase = ring artifact returned.
    let extra_dark = mips_dark.saturating_sub(base_dark);
    let extra_pct = 100.0 * extra_dark as f64 / pixel_count as f64;
    assert!(
        extra_pct < 1.0,
        "multi-mip introduces {extra_dark} extra dark pixels ({extra_pct:.2}%) — ring artifact?"
    );
}
