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
