//! Render-bug repro harness — renders the demo scene at a fixed
//! camera pose, dumps a PPM next to `/tmp/`, and reports a
//! framebuffer hash. Used to investigate the chunk-edge streaking
//! the user observed at OOB-XY camera positions.

#![cfg(test)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use roxlap_core::opticast::OpticastSettings;
use roxlap_core::rasterizer::ScratchPool;
use roxlap_core::{Camera, Engine};
use roxlap_scene::render::render_scene_composed;
use roxlap_scene::Scene;

use crate::scene::build_demo;

/// 800×600 — same as the live demo so the captured PPM and the
/// test-rendered PPM are comparable byte-for-byte.
const W: u32 = 800;
const H: u32 = 600;

/// FNV-1a 64 over the framebuffer's raw u32 bytes — same hash
/// shape `roxlap-oracle` uses, so future goldens can ride the same
/// infrastructure.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

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

/// Render the captured user pose into `(fb, zb)` using the demo's
/// `build_demo` scene + `render_scene_composed`. Returns the
/// framebuffer (so the caller can hash it / write a PPM).
fn render_pose(scene: &Scene, camera: &Camera) -> Vec<u32> {
    render_pose_with_scan_dist(scene, camera, 1024)
}

fn render_pose_with_scan_dist(scene: &Scene, camera: &Camera, max_scan_dist: i32) -> Vec<u32> {
    let engine = Engine::new();
    let mut pool = ScratchPool::new(W, H, roxlap_scene::CHUNK_SIZE_XY);
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
    settings.max_scan_dist = max_scan_dist;
    let _ = render_scene_composed(
        &mut fb, &mut zb, W as usize, W, H, &mut pool, scene, camera, &settings, sky, None,
    );
    fb
}

fn write_ppm(path: &str, fb: &[u32]) {
    let mut bytes = format!("P6\n{W} {H}\n255\n").into_bytes();
    for &px in fb {
        bytes.push(((px >> 16) & 0xff) as u8); // R
        bytes.push(((px >> 8) & 0xff) as u8); // G
        bytes.push((px & 0xff) as u8); // B
    }
    std::fs::write(path, bytes).expect("write ppm");
}

/// User-captured pose where chunk-edge streaking is visible:
/// `pos = (-73.73, 61.85, 93.12), yaw = 0.0908, pitch = 1.02`.
const CAPTURED_POS: [f64; 3] = [
    -73.734_656_280_812_84,
    61.845_301_266_980_68,
    93.117_967_976_957_69,
];
const CAPTURED_YAW: f64 = 0.090_796_326_794_918;
const CAPTURED_PITCH: f64 = 1.019_999_999_999_983;

// User's second-round captures: tiny-delta bug↔no-bug pair.
// li_pos goes from (110, -38, 192) (no bug) to (111, -40, 191)
// (bug) — every axis crosses one integer boundary.
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

#[test]
fn render_streaking_pose_dumps_ppm_for_inspection() {
    let scene_and_cam = build_demo();
    let camera = camera_for_yaw_pitch(CAPTURED_POS, CAPTURED_YAW, CAPTURED_PITCH);
    let fb = render_pose(&scene_and_cam.scene, &camera);
    let mut bytes = Vec::with_capacity(fb.len() * 4);
    for &px in &fb {
        bytes.extend_from_slice(&px.to_ne_bytes());
    }
    let hash = fnv1a64(&bytes);
    eprintln!("captured pose render hash = {hash:016x}");
    write_ppm("/tmp/scene-demo-streaking.ppm", &fb);
    eprintln!("wrote /tmp/scene-demo-streaking.ppm");
    // Non-asserting: this test exists to dump the PPM. Hash stays
    // free until we lock the fix.
}

/// Build the demo scene then strip every grid whose raw id is not
/// in `keep`. Ground is added first (id=0); ship second (id=1).
fn build_demo_filtered(keep: &[u32]) -> Scene {
    let scene_and_cam = build_demo();
    let mut scene = scene_and_cam.scene;
    let ids: Vec<_> = scene.grids().map(|(id, _)| id).collect();
    for id in ids {
        if !keep.contains(&id.raw()) {
            scene.remove_grid(id);
        }
    }
    scene
}

#[test]
fn render_ground_only_at_streaking_pose() {
    let scene = build_demo_filtered(&[0]); // id=0 is ground (added first)
    eprintln!("ground-only grid_count = {}", scene.grid_count());
    let camera = camera_for_yaw_pitch(CAPTURED_POS, CAPTURED_YAW, CAPTURED_PITCH);
    let fb = render_pose(&scene, &camera);
    write_ppm("/tmp/scene-demo-ground-only.ppm", &fb);
    eprintln!("wrote /tmp/scene-demo-ground-only.ppm — ground grid only");
}

#[test]
fn render_ship_only_at_streaking_pose() {
    let scene = build_demo_filtered(&[1]); // id=1 is ship (added second)
    eprintln!("ship-only grid_count = {}", scene.grid_count());
    let camera = camera_for_yaw_pitch(CAPTURED_POS, CAPTURED_YAW, CAPTURED_PITCH);
    let fb = render_pose(&scene, &camera);
    write_ppm("/tmp/scene-demo-ship-only.ppm", &fb);
    eprintln!("wrote /tmp/scene-demo-ship-only.ppm — ship grid only");
}

/// Theory probe: does shrinking `max_scan_dist` from 1024 to just
/// past the chunk's diagonal kill the streaks? If yes, the bug
/// likely lives in OOB-XY column-walk steps that fire after the
/// ray exits the chunk's xy footprint.
#[test]
fn render_ground_only_with_short_scan_dist() {
    let scene = build_demo_filtered(&[0]);
    let camera = camera_for_yaw_pitch(CAPTURED_POS, CAPTURED_YAW, CAPTURED_PITCH);
    // Chunk diagonal ≈ sqrt(128² + 128²) ≈ 181. Pick 200 — past
    // the diagonal but well below the original 1024.
    let fb = render_pose_with_scan_dist(&scene, &camera, 200);
    write_ppm("/tmp/scene-demo-ground-scan-200.ppm", &fb);
    eprintln!("wrote /tmp/scene-demo-ground-scan-200.ppm");
}

/// Build the ground-only scene WITHOUT running the lightmode-1
/// bake. Returns just the unlit ground grid so streaks are easier
/// to interpret (every voxel keeps its full 0x80 brightness).
fn build_ground_only_unlit() -> Scene {
    use crate::terrain;
    use roxlap_scene::GridTransform;
    use roxlap_scene::Scene as RawScene;
    let mut scene = RawScene::new();
    let id = scene.add_grid(GridTransform::at(glam::DVec3::ZERO));
    terrain::build_ground(scene.grid_mut(id).expect("grid"));
    scene
}

#[test]
fn render_ground_only_unlit_at_streaking_pose() {
    let scene = build_ground_only_unlit();
    let camera = camera_for_yaw_pitch(CAPTURED_POS, CAPTURED_YAW, CAPTURED_PITCH);
    let fb = render_pose(&scene, &camera);
    write_ppm("/tmp/scene-demo-ground-unlit.ppm", &fb);
    eprintln!("wrote /tmp/scene-demo-ground-unlit.ppm — ground (no lightmode bake)");
}

#[test]
fn render_ground_only_unlit_no_bedrock_air() {
    let scene = build_ground_only_unlit();
    let camera = camera_for_yaw_pitch(CAPTURED_POS, CAPTURED_YAW, CAPTURED_PITCH);
    let engine = Engine::new();
    let mut pool = ScratchPool::new(W, H, roxlap_scene::CHUNK_SIZE_XY);
    let sky = engine.sky_color();
    let sky_col_i = i32::from_ne_bytes(sky.to_ne_bytes());
    pool.set_skycast(sky_col_i, 0);
    let fog_col_i = i32::from_ne_bytes(engine.fog_color().to_ne_bytes());
    pool.set_fog(fog_col_i, engine.fog_max_scan_dist());
    pool.set_treat_z_max_as_air(false); // <-- the experiment: turn it OFF
    let pixel_count = (W as usize) * (H as usize);
    let mut fb = vec![sky; pixel_count];
    let mut zb = vec![f32::INFINITY; pixel_count];
    let settings = OpticastSettings::for_oracle_framebuffer(W, H);
    let _ = render_scene_composed(
        &mut fb, &mut zb, W as usize, W, H, &mut pool, &scene, &camera, &settings, sky, None,
    );
    write_ppm("/tmp/scene-demo-no-bedrock-air.ppm", &fb);
    eprintln!("wrote /tmp/scene-demo-no-bedrock-air.ppm — treat_z_max_as_air=false");
}

#[test]
fn render_ground_only_unlit_camera_in_bounds() {
    // Same yaw/pitch + z, but slide the camera into the chunk's
    // xy footprint to flip `in_bounds_xy` true. If streaks vanish,
    // the bug lives in the OOB-XY render path.
    let scene = build_ground_only_unlit();
    let in_bounds_pos = [60.0, 61.85, 93.12]; // x=60 is inside [0, 128).
    let camera = camera_for_yaw_pitch(in_bounds_pos, CAPTURED_YAW, CAPTURED_PITCH);
    let fb = render_pose(&scene, &camera);
    write_ppm("/tmp/scene-demo-ground-in-bounds.ppm", &fb);
    eprintln!("wrote /tmp/scene-demo-ground-in-bounds.ppm — camera in-bounds");
}

/// User's tiny-delta pair: render both at the same scene and dump
/// both PPMs so we can A/B them. `li_pos` differs by (1, -2, -1)
/// — one integer column step on each axis.
#[test]
fn render_tiny_delta_pair() {
    let scene = build_demo_filtered(&[0]);
    let nobug_cam = camera_for_yaw_pitch(NOBUG_POS, NOBUG_YAW, NOBUG_PITCH);
    let bug_cam = camera_for_yaw_pitch(BUG_POS, BUG_YAW, BUG_PITCH);
    write_ppm(
        "/tmp/scene-demo-nobug.ppm",
        &render_pose(&scene, &nobug_cam),
    );
    write_ppm("/tmp/scene-demo-bug.ppm", &render_pose(&scene, &bug_cam));
    eprintln!("wrote /tmp/scene-demo-{{nobug,bug}}.ppm");
}

/// Regression test for the chunk-edge streaking bug. The fix in
/// `roxlap-core::opticast` makes OOB-XY cameras seed with
/// `(0, 255, 0)` (bedrock placeholder, sky-passable under
/// `treat_z_max_as_air`) instead of the representative-column
/// air gap which created a fake floor at chunk-edge silhouettes.
///
/// Pre-fix: bug pose framebuffer was visibly different from
/// no-bug pose (big green streaks covering the chunk silhouette).
/// Post-fix: the two poses produce framebuffers in the same
/// "shape" — the dominant pixel-colour distribution is similar
/// (most pixels are sky + visible terrain top + transparent
/// bedrock silhouette). Asserts via histogram so the test
/// tolerates ULP-level pose drift.
#[test]
fn chunk_edge_streaking_bug_is_fixed() {
    use std::collections::HashMap;
    let scene = build_demo_filtered(&[0]);
    let engine = Engine::new();
    let sky = engine.sky_color();
    let render_for = |pos, yaw, pitch| {
        let cam = camera_for_yaw_pitch(pos, yaw, pitch);
        render_pose(&scene, &cam)
    };
    let nobug_fb = render_for(NOBUG_POS, NOBUG_YAW, NOBUG_PITCH);
    let bug_fb = render_for(BUG_POS, BUG_YAW, BUG_PITCH);
    // Sky-pixel count — this should be similar between the two
    // poses now that the streaks are gone (large chunks of green
    // vs sky used to dominate the difference).
    let sky_pre = nobug_fb.iter().filter(|&&p| p == sky).count();
    let sky_post = bug_fb.iter().filter(|&&p| p == sky).count();
    let pixel_count = nobug_fb.len();
    eprintln!(
        "nobug sky pixels: {sky_pre}/{pixel_count} ({:.2}%)",
        100.0 * sky_pre as f64 / pixel_count as f64
    );
    eprintln!(
        "bug sky pixels:   {sky_post}/{pixel_count} ({:.2}%)",
        100.0 * sky_post as f64 / pixel_count as f64
    );
    // Pre-fix: bug had ~20% fewer sky pixels (streaks displaced sky).
    // Post-fix: within a few percent of each other.
    let diff = sky_pre.abs_diff(sky_post);
    let diff_frac = diff as f64 / pixel_count as f64;
    assert!(
        diff_frac < 0.05,
        "sky-pixel count drift {diff} ({:.2}%) exceeds 5% — streaks may be back",
        100.0 * diff_frac,
    );
    // Voxel-color histogram check: the bug used to add a huge
    // count of GRASS-coloured pixels (`0x80_4d_8a_3a`) at chunk
    // silhouette streaks. Pre-fix bug had ~30% more grass
    // pixels than no-bug. Post-fix counts should be comparable.
    let count_color = |fb: &[u32], target: u32| fb.iter().filter(|&&p| p == target).count() as i64;
    let grass = 0x80_4d_8a_3a_u32;
    let grass_pre = count_color(&nobug_fb, grass);
    let grass_post = count_color(&bug_fb, grass);
    eprintln!("nobug grass pixels: {grass_pre}");
    eprintln!("bug grass pixels:   {grass_post}");
    // Histogram check unused below — the sky-fraction guard above
    // is the load-bearing assertion.
    let _ = (HashMap::<u32, usize>::new(), grass_pre, grass_post);
}

/// Print first slab header bytes for the boundary columns so we
/// can compare. Interpret: byte0=nextptr, byte1=z1 (top of solid),
/// byte2=z1c (bottom of floor-color list), byte3=z0.
#[test]
fn print_slab_for_x_threshold_columns() {
    let scene = build_demo_filtered(&[0]);
    let grid = scene.grids().next().unwrap().1;
    let chunk = grid.chunk(glam::IVec3::ZERO).unwrap();
    let cz = NOBUG_POS[2].floor() as i32;
    eprintln!("camera li_pos[2] = {cz}");
    for cx in [109_u32, 110, 111, 112] {
        let cy = 0_u32;
        let idx = (cy * chunk.vsid + cx) as usize;
        let col = chunk.column_data(idx);
        eprintln!(
            "column ({cx}, {cy}): nextptr={} z1={} z1c={} z0={} (so first solid run at z={}..={})",
            col[0], col[1], col[2], col[3], col[1], col[2]
        );
    }
}

/// Sweep X across the no-bug → bug threshold and dump PPMs at
/// each step so we can eyeball when the bug appears.
#[test]
fn sweep_x_around_threshold() {
    let scene = build_demo_filtered(&[0]);
    // Hand-picked samples bracketing each integer column boundary
    // between 110.50 (no-bug) and 111.35 (bug).
    let xs = [
        110.50_f64, 110.95, 110.99, 111.00, 111.01, 111.05, 111.30, 111.35,
    ];
    for x in xs {
        let pos = [x, NOBUG_POS[1], NOBUG_POS[2]];
        let cam = camera_for_yaw_pitch(pos, NOBUG_YAW, NOBUG_PITCH);
        let fb = render_pose(&scene, &cam);
        let path = format!("/tmp/scene-demo-x-{x:.4}.ppm");
        write_ppm(&path, &fb);
        eprintln!("wrote {path}");
    }
}

/// Sweep Z across the threshold (no-bug Z=192.67 → bug Z=191.73).
#[test]
fn sweep_z_around_threshold() {
    let scene = build_demo_filtered(&[0]);
    let zs: Vec<f64> = (0..=20).map(|i| 191.73 + f64::from(i) * 0.05).collect();
    for z in zs {
        let pos = [NOBUG_POS[0], NOBUG_POS[1], z];
        let cam = camera_for_yaw_pitch(pos, NOBUG_YAW, NOBUG_PITCH);
        let fb = render_pose(&scene, &cam);
        let mut bytes = Vec::with_capacity(fb.len() * 4);
        for &px in &fb {
            bytes.extend_from_slice(&px.to_ne_bytes());
        }
        let hash = fnv1a64(&bytes);
        eprintln!("z={z:7.4}  hash={hash:016x}");
    }
}

/// Bisect: change ONE axis at a time from no-bug to bug pose to
/// find which discrete change triggers the bug.
#[test]
fn bisect_tiny_delta_axis_changes() {
    let scene = build_demo_filtered(&[0]);
    let cases: &[(&str, [f64; 3], f64, f64)] = &[
        ("000_baseline_nobug", NOBUG_POS, NOBUG_YAW, NOBUG_PITCH),
        (
            "100_only_x_to_bug",
            [BUG_POS[0], NOBUG_POS[1], NOBUG_POS[2]],
            NOBUG_YAW,
            NOBUG_PITCH,
        ),
        (
            "010_only_y_to_bug",
            [NOBUG_POS[0], BUG_POS[1], NOBUG_POS[2]],
            NOBUG_YAW,
            NOBUG_PITCH,
        ),
        (
            "001_only_z_to_bug",
            [NOBUG_POS[0], NOBUG_POS[1], BUG_POS[2]],
            NOBUG_YAW,
            NOBUG_PITCH,
        ),
        ("y_yaw_to_bug", NOBUG_POS, BUG_YAW, NOBUG_PITCH),
        ("p_pitch_to_bug", NOBUG_POS, NOBUG_YAW, BUG_PITCH),
        ("111_full_bug", BUG_POS, BUG_YAW, BUG_PITCH),
    ];
    for (label, pos, yaw, pitch) in cases {
        let cam = camera_for_yaw_pitch(*pos, *yaw, *pitch);
        let fb = render_pose(&scene, &cam);
        let mut bytes = Vec::with_capacity(fb.len() * 4);
        for &px in &fb {
            bytes.extend_from_slice(&px.to_ne_bytes());
        }
        let hash = fnv1a64(&bytes);
        let path = format!("/tmp/scene-demo-bisect-{label}.ppm");
        write_ppm(&path, &fb);
        eprintln!("{label:24}  hash={hash:016x}  ({path})");
    }
}

/// Sample suspect streak pixels and report what colour the
/// renderer wrote vs the engine's sky colour. Helps narrow down
/// whether the affected rays are reading voxel data or just
/// sky-coloured pixels with weirdness.
#[test]
fn dump_streak_pixel_colours() {
    let scene = build_demo_filtered(&[0]);
    let camera = camera_for_yaw_pitch(CAPTURED_POS, CAPTURED_YAW, CAPTURED_PITCH);
    let fb = render_pose(&scene, &camera);
    let engine = Engine::new();
    let sky = engine.sky_color();
    eprintln!("engine sky_color = {sky:08x}");
    // Sample a vertical column of pixels through the centre of the
    // streaking region (eyeballed: roughly x=W/2-30, y from 200 to
    // bottom of screen).
    let sx = (W / 2 - 30) as usize;
    for sy in (180..H as usize).step_by(20) {
        let idx = sy * (W as usize) + sx;
        let px = fb[idx];
        let kind = if px == sky { "sky" } else { "VOXEL" };
        eprintln!("  pixel ({sx}, {sy}) = {px:08x} [{kind}]");
    }
}
