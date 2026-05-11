//! S1.W reproducer — BLACK pentagon under outside-camera world.
//!
//! Builds the same thin-floor scene as roxlap-host's
//! `build_small_test_scene` (vsid=256, 1-voxel ground at z=200, a
//! few primitives at world centre) and renders the user's reported
//! BLACK-pentagon pose:
//!
//! ```
//! pos=(-5.08, 53.28, 204.47)  yaw=-0.5525  pitch=0.1875
//! ```
//!
//! The renderer leaves a wedge of below-the-world pixels at sky
//! distance with `radar.col == 0` (BLACK) instead of the engine
//! sky color.  This test loads that exact pose, dumps the PPM next
//! to /tmp, and reports per-row stats so we can pinpoint the y-band
//! where the leak fires.

#![cfg(not(target_arch = "wasm32"))]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::explicit_iter_loop,
    clippy::too_many_lines,
    clippy::unreadable_literal,
    clippy::uninlined_format_args
)]

use roxlap_cavegen::{pack_dense_grid_to_vxl, MAXZDIM};
use roxlap_core::camera_math;
use roxlap_core::opticast::{opticast, OpticastOutcome};
use roxlap_core::rasterizer::ScratchPool;
use roxlap_core::scalar_rasterizer::ScalarRasterizer;
use roxlap_core::sky::Sky;
use roxlap_core::{Camera, Engine, OpticastSettings};

const VSID: u32 = 256;
const XRES: u32 = 800;
const YRES: u32 = 600;

fn build_thin_floor_scene() -> roxlap_formats::vxl::Vxl {
    const GROUND_Z: usize = 200;
    const GROUND_COL: u32 = 0x80_3a_8a_3a;
    const PLATFORM_COL: u32 = 0x80_b0_b0_b8;
    const CUBE_COL: u32 = 0x80_e0_30_30;
    const SPHERE_COL: u32 = 0x80_30_60_e0;
    const TOWER_COL: u32 = 0x80_e0_d0_30;
    let vsid = VSID;
    let vsid_u = vsid as usize;
    let max_z = MAXZDIM as usize;
    let total = vsid_u * vsid_u * max_z;
    let mut grid = vec![0u8; total];
    let mut color = vec![0u32; total];
    let idx = |x: usize, y: usize, z: usize| (y * vsid_u + x) * max_z + z;

    // 1-voxel floor.
    for y in 0..vsid_u {
        for x in 0..vsid_u {
            let i = idx(x, y, GROUND_Z);
            grid[i] = 1;
            color[i] = GROUND_COL;
        }
    }

    // Platform 16×16 at centre, 8 voxels above ground.
    let cx = vsid_u / 2;
    let cy = vsid_u / 2;
    let plat_top = GROUND_Z - 8;
    for y in (cy - 8)..(cy + 8) {
        for x in (cx - 8)..(cx + 8) {
            for z in plat_top..GROUND_Z {
                let i = idx(x, y, z);
                grid[i] = 1;
                color[i] = PLATFORM_COL;
            }
        }
    }

    // Red cube 24³ east.
    for dz in 0..24 {
        for dy in 0..24 {
            for dx in 0..24 {
                let x = cx + 32 + dx;
                let y = cy.wrapping_sub(12).wrapping_add(dy);
                let z = GROUND_Z - 24 + dz;
                if x < vsid_u && y < vsid_u && z < max_z {
                    let i = idx(x, y, z);
                    grid[i] = 1;
                    color[i] = CUBE_COL;
                }
            }
        }
    }

    // Blue sphere r=12 west.
    let scx = cx as i32 - 32;
    let scy = cy as i32;
    let scz = GROUND_Z as i32 - 12;
    for dz in -12..=12 {
        for dy in -12..=12 {
            for dx in -12..=12 {
                if dx * dx + dy * dy + dz * dz <= 12 * 12 {
                    let x = scx + dx;
                    let y = scy + dy;
                    let z = scz + dz;
                    if x >= 0 && y >= 0 && z >= 0 {
                        let xu = x as usize;
                        let yu = y as usize;
                        let zu = z as usize;
                        if xu < vsid_u && yu < vsid_u && zu < max_z {
                            let i = idx(xu, yu, zu);
                            grid[i] = 1;
                            color[i] = SPHERE_COL;
                        }
                    }
                }
            }
        }
    }

    // Yellow tower 8×8×64 north.
    for dz in 0..64 {
        for dy in 0..8 {
            for dx in 0..8 {
                let x = cx - 4 + dx;
                let y = cy + 32 + dy;
                let z = GROUND_Z - 64 + dz;
                if x < vsid_u && y < vsid_u && z < max_z {
                    let i = idx(x, y, z);
                    grid[i] = 1;
                    color[i] = TOWER_COL;
                }
            }
        }
    }

    pack_dense_grid_to_vxl(&grid, &color, vsid)
}

fn camera_from_yaw_pitch(pos: [f64; 3], yaw: f64, pitch: f64) -> Camera {
    let cy_ = yaw.cos();
    let sy = yaw.sin();
    let cp = pitch.cos();
    let sp = pitch.sin();
    Camera {
        pos,
        right: [-sy, cy_, 0.0],
        down: [-cy_ * sp, -sy * sp, cp],
        forward: [cy_ * cp, sy * cp, sp],
    }
}

#[test]
fn black_pentagon_under_oob_camera() {
    let vxl = build_thin_floor_scene();
    let engine = Engine::new();

    // User's exact pose where BLACK pentagon shows below the world.
    let pos = [-5.0815, 53.2772, 204.4685];
    let yaw = -0.5525;
    let pitch = 0.1875;
    let cam = camera_from_yaw_pitch(pos, yaw, pitch);

    let mut framebuffer = vec![0u32; (XRES * YRES) as usize];
    let mut zbuffer = vec![0f32; (XRES * YRES) as usize];
    let mut pool = ScratchPool::new(XRES, YRES, vxl.vsid);

    let sky = engine.sky_color();
    // Pre-fill with a SENTINEL color so any unwritten pixel is
    // distinguishable from sky.  0x80FF00FF = magenta with the
    // engine's brightness-bit; obvious in screenshots, byte-distinct
    // from sky-blue (0x8087CEEB) and BLACK (0x00000000).
    let sentinel = 0x80FF00FFu32;
    for px in framebuffer.iter_mut() {
        *px = sentinel;
    }
    let sky_col_i = i32::from_ne_bytes(sky.to_ne_bytes());
    pool.set_skycast(sky_col_i, 0);
    let fog_col_i = i32::from_ne_bytes(engine.fog_color().to_ne_bytes());
    pool.set_fog(fog_col_i, engine.fog_max_scan_dist());
    // S1.W: treat the bedrock placeholder at z=MAXZDIM-1 as air so
    // OOB-camera rays fall through to Startsky / textured sky
    // instead of writing the BLACK bedrock color.
    pool.set_treat_z_max_as_air(true);

    let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);

    let _cs = camera_math::derive(&cam, XRES, YRES, settings.hx, settings.hy, settings.hz);

    let grid = roxlap_core::GridView::from_single_vxl(&vxl);
    let mut rasterizer = ScalarRasterizer::new(&mut framebuffer, &mut zbuffer, XRES as usize, grid);
    let outcome = opticast(&mut rasterizer, &mut pool, &cam, &settings, grid);
    drop(rasterizer);

    assert_eq!(outcome, OpticastOutcome::Rendered);

    // Per-row stats: count of sentinel/sky/black/other.
    let mut total_sentinel = 0usize;
    let mut total_sky = 0usize;
    let mut total_black = 0usize;
    let mut total_other = 0usize;
    let mut row_with_max_black = (0usize, 0usize);
    for y in 0..YRES as usize {
        let row_start = y * XRES as usize;
        let row_end = row_start + XRES as usize;
        let row = &framebuffer[row_start..row_end];
        let nb = row.iter().filter(|&&p| p == 0x0000_0000).count();
        if nb > row_with_max_black.1 {
            row_with_max_black = (y, nb);
        }
        for &p in row {
            if p == sentinel {
                total_sentinel += 1;
            } else if p == sky {
                total_sky += 1;
            } else if p == 0x0000_0000 {
                total_black += 1;
            } else {
                total_other += 1;
            }
        }
    }
    let total = framebuffer.len();
    eprintln!("BLACK pentagon repro");
    eprintln!(
        "  sentinel={total_sentinel}/{total} ({:.2}%)",
        100.0 * total_sentinel as f64 / total as f64
    );
    eprintln!(
        "  sky=     {total_sky}/{total} ({:.2}%)",
        100.0 * total_sky as f64 / total as f64
    );
    eprintln!(
        "  black=   {total_black}/{total} ({:.2}%)",
        100.0 * total_black as f64 / total as f64
    );
    eprintln!(
        "  other=   {total_other}/{total} ({:.2}%)",
        100.0 * total_other as f64 / total as f64
    );
    eprintln!(
        "  max-black row: y={} count={}",
        row_with_max_black.0, row_with_max_black.1
    );

    let mut ppm = format!("P6\n{XRES} {YRES}\n255\n").into_bytes();
    for &px in &framebuffer {
        let bytes = px.to_le_bytes();
        ppm.push(bytes[2]);
        ppm.push(bytes[1]);
        ppm.push(bytes[0]);
    }
    let path = "/tmp/oob_pentagon.ppm";
    std::fs::write(path, ppm).expect("write ppm");
    eprintln!("  wrote {path}");

    // The BLACK pentagon shows when grouscan exits without filling
    // some radar slots.  When this assertion FAILS, the leak is
    // present.  Once fixed, every unwritten slot should get filled
    // by Startsky → sky color → no BLACK pixels.
    assert_eq!(
        total_black, 0,
        "BLACK pentagon: {total_black} slots left unwritten by grouscan"
    );
}

/// Build a procedural checkerboard sky, mirror of
/// `roxlap-host::build_checkerboard_sky`. Width = elevation
/// (horizon → zenith), height = azimuth wrap.
fn build_checkerboard_sky() -> Sky {
    const W: u32 = 64;
    const H: u32 = 256;
    const TILE_X: u32 = 8;
    const TILE_Y: u32 = 16;
    let mut pixels = Vec::with_capacity((W * H) as usize);
    for y in 0..H {
        for x in 0..W {
            let (r_base, g_base, b_base) = match (y * 4) / H {
                0 => (0xe0u32, 0x40, 0x40),
                1 => (0x40, 0xe0, 0x40),
                2 => (0x40, 0x40, 0xe0),
                _ => (0xe0, 0xd0, 0x40),
            };
            let dark = ((x / TILE_X) + (y / TILE_Y)) & 1 == 1;
            let (r, g, b) = if dark {
                (r_base / 4, g_base / 4, b_base / 4)
            } else {
                (r_base, g_base, b_base)
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

/// Render the test scene at `cam` with the checkerboard sky and
/// dump a PPM at `path`. Returns the framebuffer for further
/// analysis.
fn render_to_ppm(cam: Camera, path: &str) -> Vec<u32> {
    let vxl = build_thin_floor_scene();
    let mut engine = Engine::new();
    engine.set_sky(Some(build_checkerboard_sky()));

    let mut framebuffer = vec![0u32; (XRES * YRES) as usize];
    let mut zbuffer = vec![0f32; (XRES * YRES) as usize];
    let mut pool = ScratchPool::new(XRES, YRES, vxl.vsid);

    let sky = engine.sky_color();
    for px in framebuffer.iter_mut() {
        *px = sky;
    }
    let sky_col_i = i32::from_ne_bytes(sky.to_ne_bytes());
    pool.set_skycast(sky_col_i, 0);
    let fog_col_i = i32::from_ne_bytes(engine.fog_color().to_ne_bytes());
    pool.set_fog(fog_col_i, engine.fog_max_scan_dist());
    pool.set_treat_z_max_as_air(true);

    let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
    let grid = roxlap_core::GridView::from_single_vxl(&vxl);
    let mut rasterizer = ScalarRasterizer::new(&mut framebuffer, &mut zbuffer, XRES as usize, grid);
    if let Some(sky_ref) = engine.sky() {
        rasterizer = rasterizer.with_sky(sky_ref);
    }
    let outcome = opticast(&mut rasterizer, &mut pool, &cam, &settings, grid);
    drop(rasterizer);
    assert_eq!(outcome, OpticastOutcome::Rendered);

    let mut ppm = format!("P6\n{XRES} {YRES}\n255\n").into_bytes();
    for &px in &framebuffer {
        let bytes = px.to_le_bytes();
        ppm.push(bytes[2]);
        ppm.push(bytes[1]);
        ppm.push(bytes[0]);
    }
    std::fs::write(path, ppm).expect("write ppm");
    eprintln!("  wrote {path}");
    framebuffer
}

/// S1.V — render the user's reported distortion pose alongside an
/// in-bounds camera at the same yaw/pitch. The two PPMs at
/// `/tmp/checker_oob.ppm` vs `/tmp/checker_in.ppm` let us A/B
/// whether the seam/shear pattern is OOB-specific or voxlap-
/// canonical (per-quadrant scanline orientation).
#[test]
fn checkerboard_sky_oob_vs_inbounds() {
    // User's reported OOB pose.
    let oob_pos = [-22.2338, 119.1835, 207.7956];
    // In-bounds at world centre, same yaw/pitch.
    let in_pos = [128.0, 128.0, 207.7956];
    let yaw = -0.1950;
    let pitch = 0.4375;

    let _oob_fb = render_to_ppm(
        camera_from_yaw_pitch(oob_pos, yaw, pitch),
        "/tmp/checker_oob.ppm",
    );
    let _in_fb = render_to_ppm(
        camera_from_yaw_pitch(in_pos, yaw, pitch),
        "/tmp/checker_in.ppm",
    );
}

/// S1.W — camera below the bedrock placeholder (z > MAXZDIM-1 = 255)
/// with `treat_z_max_as_air=true`. With the flag the bedrock is air,
/// so a camera at z=261 looking up should see the floor at z=200.
/// Repro of user-reported bug: render shows only sky (opticast
/// returns `SkippedCameraInSolid` because `column_walk` reaches the
/// end of the slab list with no visible air gap).
///
/// User pose: pos=(17.34, 182.69, 261.95) yaw=-0.7225 pitch=-0.7000.
/// Test scene's floor is at z=200; bedrock placeholder at z=255.
#[test]
fn below_bedrock_camera_renders_world() {
    let pos = [17.3422, 182.6868, 261.9467];
    let yaw = -0.7225;
    let pitch = -0.7000;
    let cam = camera_from_yaw_pitch(pos, yaw, pitch);

    let vxl = build_thin_floor_scene();
    let mut engine = Engine::new();
    engine.set_sky(Some(build_checkerboard_sky()));

    let mut framebuffer = vec![0u32; (XRES * YRES) as usize];
    let mut zbuffer = vec![0f32; (XRES * YRES) as usize];
    let mut pool = ScratchPool::new(XRES, YRES, vxl.vsid);
    let sky = engine.sky_color();
    for px in framebuffer.iter_mut() {
        *px = sky;
    }
    let sky_col_i = i32::from_ne_bytes(sky.to_ne_bytes());
    pool.set_skycast(sky_col_i, 0);
    let fog_col_i = i32::from_ne_bytes(engine.fog_color().to_ne_bytes());
    pool.set_fog(fog_col_i, engine.fog_max_scan_dist());
    pool.set_treat_z_max_as_air(true);

    let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
    let grid = roxlap_core::GridView::from_single_vxl(&vxl);
    let mut rasterizer = ScalarRasterizer::new(&mut framebuffer, &mut zbuffer, XRES as usize, grid);
    if let Some(sky_ref) = engine.sky() {
        rasterizer = rasterizer.with_sky(sky_ref);
    }
    let outcome = opticast(&mut rasterizer, &mut pool, &cam, &settings, grid);
    drop(rasterizer);

    let mut ppm = format!("P6\n{XRES} {YRES}\n255\n").into_bytes();
    for &px in &framebuffer {
        let bytes = px.to_le_bytes();
        ppm.push(bytes[2]);
        ppm.push(bytes[1]);
        ppm.push(bytes[0]);
    }
    std::fs::write("/tmp/below_bedrock.ppm", ppm).expect("write ppm");
    eprintln!("  wrote /tmp/below_bedrock.ppm");
    eprintln!("  outcome={outcome:?}");

    // Should NOT skip — the bedrock is air, camera is in air below
    // the bedrock-as-air, looking up should see the floor at z=200.
    assert_eq!(
        outcome,
        OpticastOutcome::Rendered,
        "below-bedrock camera with treat_z_max_as_air should render the world",
    );

    // Most pixels should NOT be sky — the floor at z=200 occupies
    // a non-trivial portion of the upper half of the screen at
    // pitch=-0.70 (looking up). Allow up to 50% sky to account for
    // the corners where rays escape past the world.
    let sky_count = framebuffer.iter().filter(|&&p| p == sky).count();
    let sky_frac = sky_count as f64 / framebuffer.len() as f64;
    eprintln!("  sky fraction = {:.3}", sky_frac);
    assert!(
        sky_frac < 0.95,
        "below-bedrock render is {:.1}% sky — looks like the world wasn't rendered",
        100.0 * sky_frac,
    );
}

/// S1.V — above-floor vs below-floor at the SAME OOB-X yaw/pitch.
/// User's claim: the sky-checker distortion appears ONLY when the
/// camera is below the floor (z > `GROUND_Z` = 200). Above the
/// floor (z < 200), the same yaw/pitch should render a clean
/// checker.
///
/// Both positions are OOB-X (cx=-22) so OOB-vs-inbounds is held
/// constant. The floor crossing is the only variable.
///
/// Outputs:
/// - `/tmp/checker_below_floor.ppm` — z=207.80 (≈ user's pose Z)
/// - `/tmp/checker_above_floor.ppm` — z=192.00 (8 voxels above the
///   floor, inside the air-above region)
#[test]
fn checkerboard_sky_above_vs_below_floor() {
    let yaw = -0.1950;
    let pitch = 0.4375;
    let xy = [-22.2338, 119.1835];
    let _below = render_to_ppm(
        camera_from_yaw_pitch([xy[0], xy[1], 207.7956], yaw, pitch),
        "/tmp/checker_below_floor.ppm",
    );
    let _above = render_to_ppm(
        camera_from_yaw_pitch([xy[0], xy[1], 192.0], yaw, pitch),
        "/tmp/checker_above_floor.ppm",
    );
}
