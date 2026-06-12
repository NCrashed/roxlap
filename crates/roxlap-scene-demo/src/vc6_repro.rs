//! VC.6 — mip-N multi-chz column-step regression pin.
//!
//! VC.5 (`memory/project_vc_5_landed.md`) fixed the mip-0
//! multi-chunk column-step at `grouscan.rs::phase_after_delete_kept_presync`
//! by routing the install through `build_owned_column_multi_chz`.
//! VC.6.2 (`memory/project_vc_6_2_landed.md`) lands the matching
//! fix on the **mip-N** column-step branch (`grouscan.rs:1673..1751`):
//! the helper grew a `mip_level: u32` parameter (VC.6.1) and the
//! mip-N column-step now calls it with `state.gmipcnt` so distant-XY
//! rays at mip-N stitch chz layers the same way mip-0 already does.
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
//! past the first chunk-XY crossing.
//!
//! ## Pre-VC.6.2 behaviour (= the bug)
//!
//! `install_owned_column` reads `chunk_at_xyz([new_xy,
//! current_chunk_z = seed_chz = 0])` → chz=0's mip-N sub-table
//! → placeholder bedrock → distant rays draw sky. Red floor
//! visible only inside the camera's own chunk-XY footprint
//! (~19k pixels at this pose) where the seed install's multi-
//! chz stitch covers all chz.
//!
//! ## Post-VC.6.2 behaviour (= fix in place)
//!
//! `build_owned_column_multi_chz(.., mip_level = state.gmipcnt)`
//! stitches every chz at the chunk's mip-N sub-table per
//! column-step. The bottom half of the framebuffer fills with
//! red (~104k pixels) — chz=2's floor visible at distant XY.

#![cfg(test)]
#![allow(clippy::cast_precision_loss)]

use glam::{DVec3, IVec3};
use roxlap_core::opticast::OpticastSettings;
use roxlap_core::rasterizer::ScratchPool;
use roxlap_core::{Camera, Engine};
use roxlap_scene::render::render_scene_composed;
use roxlap_scene::{GridTransform, Scene, CHUNK_SIZE_XY};

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

/// VC.6.2 primary regression pin. The mip-N column-step now
/// routes through `build_owned_column_multi_chz` with `mip_level =
/// state.gmipcnt`, so distant-XY rays at mip-N read the stitched
/// chz=0..max_chz column at the chunk's mip-N sub-table. The
/// bottom half of the framebuffer fills with red (~104k pixels) —
/// a 5.5× jump from the bug-active baseline of ~19k.
///
/// Direction flipped at VC.6.2 land. Pre-VC.6.2 form asserted
/// `bottom_red < 30_000` (= bug active); the inverse was an
/// `#[ignore]`'d companion. They merged into this single test
/// once the fix landed. Mirrors [[vc-5-landed]]'s pattern.
#[test]
fn vc6_2_mip_n_multi_chz_distant_chz2_floor_visible_at_chz0_camera() {
    let mut scene = build_3chz_grid_with_floor_at_chz2();

    let cam = camera_for_yaw_pitch(
        [0.0, 0.0, 50.0],
        std::f64::consts::FRAC_PI_2, // yaw=π/2 → looking +y
        std::f64::consts::FRAC_PI_3, // pitch=π/3 → 60° down
    );
    let (fb, _zb) = render_repro(&mut scene, &cam);

    let (total_red, top_red, bottom_red) = count_red_buckets(&fb);
    let hash = fnv1a64_fb(&fb);
    write_ppm("/tmp/vc6_2_mip_n_multi_chz_fix.ppm", &fb);
    eprintln!(
        "vc6.2 mip-N multi-chz FIX: total_red={total_red} top_half={top_red} \
         bottom_half={bottom_red} hash={hash:#018x} (PPM at \
         /tmp/vc6_2_mip_n_multi_chz_fix.ppm)"
    );

    // FIX assertion — distant-XY rays at mip-N now read the
    // multi-chz column. `bottom_red` jumped from ~19k (bug
    // active) to ~104k (fix landed). A regression that re-
    // introduces single-chz mip-N install would shrink this
    // count back below the ~30k threshold.
    assert!(
        bottom_red > 100_000,
        "VC.6.2: expected bottom_red > 100000 with the multi-chz mip-N fix \
         in place; got bottom_red={bottom_red}, total_red={total_red}. If \
         bottom_red ≲ 19000 the mip-N column-step has regressed to single-chz \
         install. Audit `grouscan.rs:1729-1751`."
    );

    // HASH PIN — locks the post-fix render. Future engine
    // changes that touch the mip-N column-step or
    // `build_owned_column_multi_chz` must update this pin
    // deliberately.
    assert_eq!(
        hash, VC6_2_FIX_LANDED_HASH,
        "VC.6.2 hash drift — fix-landed render changed. If intentional \
         (e.g. a follow-up mip-N path refinement), update the pin; if \
         unintentional, audit the mip-N column-step or multi-chz install."
    );
}

/// Pinned at PRR.1 (2026-06-03). Post-fix render of the 4×4×3
/// chunk fixture at the chz=0-camera-looking-down pose.
///
/// History:
/// * `0x74b1_9785_9c26_911c` — VC.6.0 bug-active (camera-chunk-
///   only blob, ~19 k red pixels).
/// * `0x577e_6879_b86e_f758` — VC.6.2 fix landed; multi-chz
///   mip-N install at the column-step revealed the full chz=2
///   floor at distant XY (~104 k red pixels).
/// * `0xd8eb_5565_84f2_f30d` — PRR.1: also routes
///   `phase_remiporend`'s post-mip-transition reload through
///   the multi-chz install. Adds ~931 voxel-fills at the
///   camera's own chunk-XY that the single-chz reload had been
///   bypassing to sky between mip-transitions and the next
///   column-step. Subtle in-fill; the trapezoidal shape is
///   visually unchanged.
const VC6_2_FIX_LANDED_HASH: u64 = 0xd8eb_5565_84f2_f30d;
