//! VC.6.0 — Pre-flight repro for the mip-N multi-chz column-step
//! bug.
//!
//! VC.5 (`memory/project_vc_5_landed.md`) fixed the mip-0
//! multi-chunk column-step at `grouscan.rs::phase_after_delete_kept_presync`
//! by routing the install through `build_owned_column_multi_chz`.
//! The corresponding **mip-N** column-step branch
//! (`grouscan.rs:1673..1751`) still uses single-chz
//! `install_owned_column` — same shape as pre-VC.5 mip-0. The
//! brief (`memory/project_vc_6_scope.md`) calls VC.6.0 the
//! critical-path investigation: build a multi-chz grid with real
//! content at every chz, find a pose that exposes the gap, and
//! land a regression test. If no reachable pose fires the bug,
//! VC.6 closes as dead-code and no engine fix ships.
//!
//! ## Fixture
//!
//! 4×4×3 chunk grid centred on the origin (vsid_z = 768):
//!   * chz=0 + chz=1 — materialised but empty (bedrock placeholder
//!     columns only). With `treat_z_max_as_air = true` the
//!     rasterizer's `drawfwall` / `drawflor` bedrock-as-air bypass
//!     skips these, exposing chz=2 beneath when (and only when)
//!     the column-step install stitches across chz layers.
//!   * chz=2 — solid red floor at world z = [600, 650] across the
//!     full XY footprint.
//!
//! ## Pose
//!
//! Camera at world `(0, 0, 50)` (chz=0 air) looking +y forward+
//! down at pitch ≈ 60°. Rays walk forward+down through chz=0 →
//! chz=1 → chz=2 across multiple chunk-XY crossings before
//! hitting the red floor at z=600.
//!
//! Render settings: `mip_levels = 4`, `mip_scan_dist = 64`,
//! `max_scan_dist = 1024`. The mip-N column-step branch fires
//! past the first transition; rays that have crossed at least
//! one chunk-XY boundary at mip ≥ 1 read the new chunk's mip-N
//! sub-table via `install_owned_column`.
//!
//! ## Expected pre-fix behaviour
//!
//! With the bug active, the mip-N column-step at distant XY
//! reads `chunk_at_xyz([new_xy, current_chunk_z = seed_chz = 0])`
//! → chz=0's mip-N sub-table → placeholder bedrock → distant rays
//! draw sky. Red floor visible only inside the camera's own
//! chunk-XY footprint (where the seed install's multi-chz stitch
//! covers all chz) and at near range before mip transitions fire.
//! Far-bucket red pixel count is **low** relative to the near
//! bucket.
//!
//! ## Post-fix behaviour (VC.6.2 target)
//!
//! Routing the mip-N column-step through a multi-chz builder
//! (analogous to `build_owned_column_multi_chz` but stepping
//! `mip_base_offsets[gmipcnt]` per chunk) lets distant rays at
//! distant XY see chz=2's mip-N floor. Far-bucket red count grows
//! substantially.

#![cfg(test)]
#![allow(clippy::cast_precision_loss)]

use glam::{DVec3, IVec3};
use roxlap_core::opticast::OpticastSettings;
use roxlap_core::rasterizer::ScratchPool;
use roxlap_core::{Camera, Engine};
use roxlap_scene::render::render_scene_composed;
use roxlap_scene::{GridTransform, Scene, CHUNK_SIZE_XY, CHUNK_SIZE_Z};

const W: u32 = 800;
const H: u32 = 600;
/// roxlap colour bytes are `0xAA_RR_GG_BB` (top byte is the
/// lightmode-1 brightness alpha; renderer reads it as opacity-
/// flagged when bit 7 is set). `0x80` keeps the voxel opaque
/// without baking lighting.
const GROUND_COLOR: u32 = 0x80_ff_00_00;
const CHUNKS_X: i32 = 4;
const CHUNKS_Y: i32 = 4;

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

/// FNV-1a over the framebuffer bytes — same shape as
/// `roxlap_oracle::fnv1a64`. Used as a diagnostic pin so a future
/// VC.6.2 fix change is visible across renders.
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
        bytes.push(((px >> 16) & 0xff) as u8);
        bytes.push(((px >> 8) & 0xff) as u8);
        bytes.push((px & 0xff) as u8);
    }
    std::fs::write(path, bytes).expect("write ppm");
}

/// Build a 4×4×3 chunk scene: placeholder bedrock at chz=0 +
/// chz=1, solid red floor at chz=2 across the XY extent.
fn build_3chz_grid_with_floor_at_chz2() -> Scene {
    let mut scene = Scene::new();
    let grid_id = scene.add_grid(GridTransform::at(DVec3::ZERO));
    let grid = scene.grid_mut(grid_id).expect("ground grid present");
    let half_x = CHUNKS_X / 2;
    let half_y = CHUNKS_Y / 2;
    let cs_xy = CHUNK_SIZE_XY as i32;
    let _cs_z = CHUNK_SIZE_Z as i32;

    // Materialise empty bedrock-placeholder chunks at chz=0 and
    // chz=1. `chunk_xyz_backing`'s bbox then spans chz=0..2 (origin
    // 0, chunks_z 3), so the rasterizer sees a stacked grid with
    // both `chunks_z > 1` and real content at a chz != seed_chz.
    for chy in -half_y..(CHUNKS_Y - half_y) {
        for chx in -half_x..(CHUNKS_X - half_x) {
            let _ = grid.ensure_chunk(IVec3::new(chx, chy, 0));
            let _ = grid.ensure_chunk(IVec3::new(chx, chy, 1));
        }
    }

    // Solid red floor across the full XY footprint at world z =
    // [600, 650]. Lands inside chz=2 (which covers voxel z =
    // [512, 768)) at chunk-local z = [88, 138]. `set_rect`
    // materialises chunks at chz=2 on demand — together with the
    // chz=0/1 ensure-chunk loop above the grid carries all
    // chunks_x × chunks_y × chunks_z = 48 slots.
    grid.set_rect(
        IVec3::new(-half_x * cs_xy, -half_y * cs_xy, 600),
        IVec3::new(
            (CHUNKS_X - half_x) * cs_xy - 1,
            (CHUNKS_Y - half_y) * cs_xy - 1,
            649,
        ),
        Some(GROUND_COLOR),
    );

    // 6-mip ladder matches the live demo's bake-pass depth (see
    // `scene::bake_lightmode_1`). The mip-N column-step branch
    // only fires for `mip_levels >= 2`, but generating beyond 1
    // covers any future bump to `OpticastSettings::mip_levels`.
    for chunk in grid.chunks.values_mut() {
        chunk.generate_mips(6);
    }
    scene
}

fn render_repro(scene: &mut Scene, cam: &Camera) -> (Vec<u32>, Vec<f32>) {
    let engine = Engine::new();
    let mut pool = ScratchPool::new(W, H, 32 * CHUNK_SIZE_XY);
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
    settings.max_scan_dist = 1024;

    let _ = render_scene_composed(
        &mut fb, &mut zb, W as usize, W, H, &mut pool, scene, cam, &settings, sky, None,
    );
    (fb, zb)
}

/// Count red-dominant pixels split by screen-Y position.
///
/// The floor sits below the camera at world z=600..650. With a
/// pitch=π/3 downward camera at z=50, the horizon line lies above
/// the screen midline, so the floor (post-fix) fills the bottom
/// half of the framebuffer. Today (bug active) only the camera-
/// chunk's XY footprint shows red — a small blob near the bottom
/// centre. The screen-Y split is therefore a much cleaner metric
/// than zbuffer depth: at this pose all red pixels happen to lie
/// at `d ≈ 550` (the floor's perpendicular distance) so a depth
/// bucket has zero discriminating power.
fn count_red_buckets(fb: &[u32]) -> (usize, usize, usize) {
    let mut total = 0usize;
    let mut top_half = 0usize;
    let mut bottom_half = 0usize;
    let half = (H / 2) as usize;
    for (i, &p) in fb.iter().enumerate() {
        let r = (p >> 16) & 0xff;
        let g = (p >> 8) & 0xff;
        let b = p & 0xff;
        // Strict red dominance — excludes any sky tint at the
        // horizon. The floor colour `0x80_ff_00_00` shows up as
        // `r >= 200, g + b < 30` once fog has decayed it.
        if r > 100 && r > g + 30 && r > b + 30 {
            total += 1;
            let y = i / (W as usize);
            if y < half {
                top_half += 1;
            } else {
                bottom_half += 1;
            }
        }
    }
    (total, top_half, bottom_half)
}

/// VC.6.0 primary deliverable. Verifies the mip-N multi-chz bug
/// fires reproducibly at a chosen pose by asserting that distant
/// red-floor pixels are RARE relative to near red-floor pixels
/// today.
///
/// Today (HEAD post-GY-overflow, 2026-06-01): the mip-N
/// column-step at distant XY reads chz=0's placeholder bedrock
/// → far-bucket red count is low. This assertion passes.
///
/// VC.6.2 lands the multi-chz mip-N install. Far-bucket red
/// count grows → this assertion fails. At that point flip the
/// direction (e.g. `far_red > 5_000`) to assert the FIX is in
/// place. The flip lives alongside the same code change that
/// land VC.6.2 — mirrors the [[vc-5-landed]] `vc5_bug_fires`
/// pattern.
#[test]
fn vc6_0_mip_n_multi_chz_bug_fires_at_chz0_camera_looking_down_at_chz2_floor() {
    let mut scene = build_3chz_grid_with_floor_at_chz2();

    let cam = camera_for_yaw_pitch(
        [0.0, 0.0, 50.0],
        std::f64::consts::FRAC_PI_2, // yaw=π/2 → looking +y
        std::f64::consts::FRAC_PI_3, // pitch=π/3 → 60° down
    );
    let (fb, _zb) = render_repro(&mut scene, &cam);

    let (total_red, top_red, bottom_red) = count_red_buckets(&fb);
    let hash = fnv1a64_fb(&fb);
    write_ppm("/tmp/vc6_0_mip_n_multi_chz_repro.ppm", &fb);
    eprintln!(
        "vc6.0 mip-N multi-chz repro: total_red={total_red} top_half={top_red} \
         bottom_half={bottom_red} hash={hash:#018x} (PPM at \
         /tmp/vc6_0_mip_n_multi_chz_repro.ppm)"
    );

    // Sanity: the camera-XY column has chz=2 stitched via the
    // multi-chz seed install (see `from_seed` at
    // `crates/roxlap-core/src/grouscan.rs:510`). Some red is
    // expected; if `total_red == 0` either the fixture didn't
    // build the floor or the rasterizer regressed on the seed
    // install.
    assert!(
        total_red > 5_000,
        "VC.6.0 sanity: expected total_red > 5000 (camera-XY column should \
         stitch chz=2 floor via multi-chz seed install in the camera's chunk); \
         got total_red={total_red}. Either the fixture broke or the multi-chz \
         seed install regressed."
    );

    // PRIMARY ASSERTION — current HEAD (= bug active). Bottom-half
    // red count is BOUNDED by the small camera-chunk footprint
    // (~19 k pixels at this pose). If the mip-N multi-chz fix
    // landed, distant rays would see chz=2's floor across the
    // full bottom half (~240 k pixels) — the count would jump 5-
    // 10×. VC.6.2 will flip this assertion's direction.
    assert!(
        bottom_red < 30_000,
        "VC.6.0: expected the mip-N multi-chz bug to fire (bottom_red < 30000, \
         i.e. distant XY rays draw sky instead of chz=2 floor). Got \
         bottom_red={bottom_red}, total_red={total_red}. If bottom_red is much \
         larger, the bug may NOT fire at this pose: (a) the multi-chz mip-N \
         path may already do the right thing in this topology, or (b) the \
         camera/scene needs adjusting. Per the VC.6.0 brief, if no reachable \
         pose triggers the bug, close VC.6 as dead-code."
    );

    // HASH PIN — diagnostic, not load-bearing. Re-anchor as VC.6.2
    // flips the bug. A mid-stage refactor that changes this hash
    // without obvious cause warrants investigation.
    assert_eq!(
        hash, VC6_0_BUG_ACTIVE_HASH,
        "VC.6.0 hash drift — bug-active render changed. If intentional (refactor \
         that preserves the bug behaviour), update the pin; if unintentional, \
         audit the mip-N column-step path."
    );
}

/// Pinned at VC.6.0 (2026-06-01). Render under the mip-N multi-chz
/// bug = chz=2 floor visible only inside the camera chunk's XY
/// footprint. VC.6.2 deliberately busts this when the fix lands.
const VC6_0_BUG_ACTIVE_HASH: u64 = 0x74b1_9785_9c26_911c;

/// VC.6.2 target — un-ignore when the multi-chz mip-N install
/// lands. Asserts the FIX is in place: distant-XY rays see chz=2's
/// floor across the bottom half of the screen, so `bottom_red`
/// jumps from ~19 k to ~100 k+ (the entire visible-ground area
/// fills with red). Same fixture + pose as
/// [`vc6_0_mip_n_multi_chz_bug_fires_at_chz0_camera_looking_down_at_chz2_floor`].
///
/// Mirrors [[vc-5-landed]]'s pattern: VC.0 introduced
/// `stacked_chz0_distant_mountain_visible_from_chz0_camera` as
/// `#[ignore]`, VC.5 un-ignored it once the fix flipped its
/// outcome.
#[test]
#[ignore = "VC.6.2 fix target — un-ignore when the multi-chz mip-N install \
            lands and bottom_red grows past the camera-chunk-only blob"]
fn vc6_2_fix_landed_distant_chz2_floor_visible_under_mip_n() {
    let mut scene = build_3chz_grid_with_floor_at_chz2();
    let cam = camera_for_yaw_pitch(
        [0.0, 0.0, 50.0],
        std::f64::consts::FRAC_PI_2,
        std::f64::consts::FRAC_PI_3,
    );
    let (fb, _zb) = render_repro(&mut scene, &cam);
    let (total_red, top_red, bottom_red) = count_red_buckets(&fb);
    let hash = fnv1a64_fb(&fb);
    eprintln!(
        "vc6.2 mip-N multi-chz FIX: total_red={total_red} top_half={top_red} \
         bottom_half={bottom_red} hash={hash:#018x}"
    );
    // With the fix in place, distant-XY rays read chz=2's floor
    // through the multi-chz install. Bottom half is mostly red.
    assert!(
        bottom_red > 100_000,
        "VC.6.2: expected bottom_red > 100000 with the multi-chz mip-N fix \
         in place; got bottom_red={bottom_red} (still close to the bug-active \
         baseline of ~19000). The fix at `grouscan.rs:1729-1751` may not be \
         routing through the multi-chz path."
    );
}
