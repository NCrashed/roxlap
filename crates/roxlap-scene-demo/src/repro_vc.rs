//! VC.0 — Virtual Column rewrite pre-flight harness.
//!
//! Pins the render output across the VC stage so VC.1..VC.4
//! refactors stay byte-identical, and locks the user-reported
//! z = -19.44 bug into a test that fails when VC.5's virtual
//! stitched-chunk column lands.
//!
//! See `memory/project_vc_scope.md` and
//! `memory/project_s4b_6_chz_multi_research.md` for the full
//! scope brief and prior-attempt log.
//!
//! Mechanics:
//! - The 0.3.0 demo mitigation
//!   ([`crate::terrain::HillsChunkGenerator::should_generate`]) hides
//!   the bug from the live binary by declining `chunk_idx.z != 0`.
//!   For the bug-fires test we wrap it with [`BugFireHillsGenerator`]
//!   that returns `true` for every chz — the streaming pump then
//!   materialises chz = -1 placeholder bedrock chunks alongside the
//!   chz = 0 hills, recreating the user's exact pre-mitigation grid
//!   topology.
//! - The user's pose `(3.78, -159.66, -19.44) yaw=1.5033 pitch=1.3575`
//!   sits in chz = -1 (world z < 0). Camera-XY column's `seed_chz`
//!   walks down to chz = 0 (hills); column-step then reads the
//!   chz = -1 placeholder at every other XY → mid-render handoff
//!   bumps `chunk_world_z_base` by 256 → distant hill voxels render
//!   one chunk too deep in world-z.
//!
//! VC.5 makes [`vc5_bug_fires_at_user_z_neg_19_pose`] pass; flip its
//! assertion direction at that point. Until then it fails on
//! current master as the brief sanctions.

#![cfg(test)]
#![allow(clippy::cast_precision_loss)]

use std::sync::Arc;

use glam::{DVec3, IVec3};
use roxlap_core::opticast::OpticastSettings;
use roxlap_core::rasterizer::ScratchPool;
use roxlap_core::{Camera, Engine};
use roxlap_formats::vxl::Vxl;
use roxlap_scene::render::render_scene_composed;
use roxlap_scene::ChunkGenerator;

use crate::scene::build_demo;
use crate::terrain::HillsChunkGenerator;

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

/// Bypass wrapper around [`HillsChunkGenerator`] that ignores the
/// `chunk_idx.z != 0` filter from the 0.3.0 demo mitigation
/// ([[chunk-generator-should-generate]]). `generate` still returns
/// empty bedrock for chz != 0 (the inner generator's behaviour),
/// so chz = -1 / chz = 1 chunks materialise as the placeholder
/// bedrock that triggers the S4B.6.j cross-chunk look-down path.
#[derive(Debug, Clone, Copy)]
struct BugFireHillsGenerator(HillsChunkGenerator);

impl ChunkGenerator for BugFireHillsGenerator {
    fn generate(&self, chunk_idx: IVec3) -> Vxl {
        self.0.generate(chunk_idx)
    }
    fn should_generate(&self, _chunk_idx: IVec3) -> bool {
        true
    }
}

/// Render at the given pose. Returns `(framebuffer, zbuffer)`.
fn render_at_pose(
    sc: &mut crate::scene::SceneAndCamera,
    pose: [f64; 3],
    yaw: f64,
    pitch: f64,
) -> (Vec<u32>, Vec<f32>) {
    let cam = camera_for_yaw_pitch(pose, yaw, pitch);
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
    (fb, zb)
}

/// FNV-1a 64-bit over the framebuffer's packed-u32 bytes. Same
/// hash as `roxlap_oracle::fnv1a64` — gives us a stable pin
/// across runs without pulling in the oracle crate.
fn fnv1a64_fb(fb: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &px in fb {
        for &b in &px.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
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

/// Build the streaming-hills demo, override the ground generator
/// with [`BugFireHillsGenerator`] (so chz = -1 placeholders
/// materialise), and sync-pump at `cam_world_pos`.
fn build_streaming_demo_with_bug_bypass(cam_world_pos: DVec3) -> crate::scene::SceneAndCamera {
    let mut sc = build_demo();
    let ground_id = sc
        .scene
        .grids()
        .map(|(id, _)| id)
        .min_by_key(|id| id.raw())
        .expect("ground grid registered");
    let g = sc.scene.grid_mut(ground_id).expect("ground grid present");
    g.set_generator(Some(Arc::new(BugFireHillsGenerator(HillsChunkGenerator))));
    sc.scene.pump_streaming_sync(cam_world_pos);
    sc
}

/// User's z = -19.44 capture pose. Camera world z < 0 → chz = -1
/// for `CHUNK_SIZE_Z = 256`.
const USER_Z_NEG_19_POS: [f64; 3] = [3.78, -159.66, -19.44];
const USER_Z_NEG_19_YAW: f64 = 1.5033;
const USER_Z_NEG_19_PITCH: f64 = 1.3575;

/// VC.0 pin: at the user's z = -19.44 pose with chz = -1 placeholder
/// chunks present, the bug pushes distant-XY hills 256 voxels too
/// low. Test counts green-dominant pixels at z-buffer depth > 250
/// (= where hills would land after the +256 shift, given camera at
/// z = -19.44 and a 77.7° downward pitch).
///
/// VC.0..VC.4: this fails the `< 30_000` upper bound IFF the
/// virtual-column rewrite isn't done yet but happens to render the
/// shifted hills inside the max_scan_dist envelope. On current
/// master at this pose, the +256 shift puts the distant hills at
/// world z ≈ 256..306 → ray distance ≈ 250..310 from camera; they
/// DO render and DO pass the `depth > 250` filter. So this test
/// currently asserts the bugged distant-hill count > 5000 to LOCK
/// IN the bug. Once VC.5 fixes the world-z, distant hills move to
/// world z ≈ 0..50 (depth ≈ 130..150) → fall OUT of the
/// `depth > 250` bucket → count drops to ~0 → assertion fails →
/// signals VC.5 is done.
///
/// FLIP at VC.5: change `> 5_000` to `< 1_000`. Update the docstring
/// to describe the post-fix expectation.
#[test]
fn vc5_bug_fires_at_user_z_neg_19_pose() {
    let cam_world_pos = DVec3::new(
        USER_Z_NEG_19_POS[0],
        USER_Z_NEG_19_POS[1],
        USER_Z_NEG_19_POS[2],
    );
    let mut sc = build_streaming_demo_with_bug_bypass(cam_world_pos);

    // Confirm both chz = -1 placeholders and chz = 0 hills made it
    // into the grid — otherwise the bypass didn't fire and the
    // test loses its meaning.
    let ground_id = sc
        .scene
        .grids()
        .map(|(id, _)| id)
        .min_by_key(|id| id.raw())
        .expect("ground grid registered");
    let grid = sc.scene.grid(ground_id).expect("ground grid");
    let chz_negative = grid.chunks.keys().filter(|i| i.z < 0).count();
    let chz_zero = grid.chunks.keys().filter(|i| i.z == 0).count();
    eprintln!(
        "vc5-bug-fires: pumped {} chunks ({} at chz<0, {} at chz=0)",
        grid.chunk_count(),
        chz_negative,
        chz_zero,
    );
    assert!(
        chz_negative > 0,
        "BugFireHillsGenerator bypass didn't materialise chz<0 placeholders — bug topology missing"
    );
    assert!(
        chz_zero > 0,
        "streaming pump didn't materialise chz=0 hills at the user's pose"
    );

    let (fb, zb) = render_at_pose(
        &mut sc,
        USER_Z_NEG_19_POS,
        USER_Z_NEG_19_YAW,
        USER_Z_NEG_19_PITCH,
    );

    // Count green-dominant pixels in two depth buckets. With the
    // bug: distant-XY hills get shifted to world z ≈ 256..306 →
    // depth ≈ 250..310. Without the bug: distant-XY hills sit near
    // world z ≈ 0..50 → depth ≈ 130..150.
    let pixel_count = (W as usize) * (H as usize);
    let mut near_hill_pixels = 0usize;
    let mut far_hill_pixels = 0usize;
    for (&p, &d) in fb.iter().zip(zb.iter()) {
        let r = (p >> 16) & 0xff;
        let g = (p >> 8) & 0xff;
        let b = p & 0xff;
        let is_green = g > r && g > b && g > 50;
        if !is_green {
            continue;
        }
        if d < 200.0 {
            near_hill_pixels += 1;
        } else if d > 250.0 {
            far_hill_pixels += 1;
        }
    }
    write_ppm("/tmp/vc5_bug_fires_z_neg_19.ppm", &fb);
    eprintln!(
        "vc5-bug-fires hill pixels: near (d<200)={near_hill_pixels} far (d>250)={far_hill_pixels} \
         of {pixel_count} (PPM at /tmp/vc5_bug_fires_z_neg_19.ppm)"
    );

    // VC.0 pin (BUG ACTIVE): the +256 shift moves distant-XY hills
    // into the d > 250 bucket; expect ample far-bucket hits.
    assert!(
        far_hill_pixels > 5_000,
        "VC.0 pin: expected bug-shifted distant hill pixels (depth > 250) > 5000; got {far_hill_pixels}. \
         If this is BELOW the bound, the +256 chunk_world_z_base shift may have been fixed — flip the \
         assertion to `< 1_000` (and document it in project_vc_0_landed.md or the substage memo)."
    );
}

/// Baseline render hash for the user's z = -19.44 pose (bug-fires
/// configuration). VC.1..VC.4 must keep this byte-identical
/// (those substages are internal refactors). VC.5 deliberately
/// busts it — update the pin to the post-fix hash at that point.
#[test]
fn vc0_baseline_hash_user_z_neg_19_pose_bug_active() {
    let cam_world_pos = DVec3::new(
        USER_Z_NEG_19_POS[0],
        USER_Z_NEG_19_POS[1],
        USER_Z_NEG_19_POS[2],
    );
    let mut sc = build_streaming_demo_with_bug_bypass(cam_world_pos);
    let (fb, _zb) = render_at_pose(
        &mut sc,
        USER_Z_NEG_19_POS,
        USER_Z_NEG_19_YAW,
        USER_Z_NEG_19_PITCH,
    );
    let hash = fnv1a64_fb(&fb);
    eprintln!("vc0-baseline user_z_neg_19 (bug-active): {hash:#018x}");
    // Pinned by VC.0. Substages VC.1..VC.4 must keep this stable;
    // VC.5 deliberately changes it. If a non-VC.5 substage breaks
    // this, audit the substage's edits against `column` / `slab_buf`
    // / `chunk_world_z_base` arithmetic.
    assert_eq!(
        hash, VC0_BASELINE_USER_Z_NEG_19_BUG_ACTIVE,
        "VC.0 baseline hash drift at user_z_neg_19 (bug-active)"
    );
}

/// Pinned on `master` at commit 28a104d (2026-05-31). See
/// [`vc0_baseline_hash_user_z_neg_19_pose_bug_active`].
const VC0_BASELINE_USER_Z_NEG_19_BUG_ACTIVE: u64 = 0x0fbe_1121_6bc6_691a;

/// Diagnostic dump for the user's z = -19.44 pose under the
/// in-binary (mitigated) generator. Renders WITHOUT the
/// `BugFireHillsGenerator` bypass — so this matches what the user
/// sees in the live demo. Pure smoke test; asserts that the
/// streaming pump materialises something + the render produces
/// some non-sky pixels.
#[test]
fn vc0_user_z_neg_19_pose_demo_smoke() {
    let cam_world_pos = DVec3::new(
        USER_Z_NEG_19_POS[0],
        USER_Z_NEG_19_POS[1],
        USER_Z_NEG_19_POS[2],
    );
    let mut sc = build_demo();
    sc.scene.pump_streaming_sync(cam_world_pos);
    let (fb, _zb) = render_at_pose(
        &mut sc,
        USER_Z_NEG_19_POS,
        USER_Z_NEG_19_YAW,
        USER_Z_NEG_19_PITCH,
    );
    let engine = Engine::new();
    let sky = engine.sky_color();
    let non_sky = fb.iter().filter(|&&p| p != sky).count();
    write_ppm("/tmp/vc0_user_z_neg_19_demo.ppm", &fb);
    eprintln!(
        "vc0 demo (mitigated) at user_z_neg_19: {non_sky} non-sky pixels / {} (PPM at /tmp/vc0_user_z_neg_19_demo.ppm)",
        (W as usize) * (H as usize),
    );
    assert!(
        non_sky > 0,
        "user_z_neg_19 demo render produced zero non-sky pixels — pump/render regression"
    );
}
