//! Diagnostic for `project_axis_aligned_mip_beams.md` — minimal
//! single-chunk repro of the world-axis green-beam artifact at
//! deep mip-N.
//!
//! The live demo's symptom ("vertical green columns rising into the
//! sky along ±X / ±Y at scan_dist > 384") was bisected to the
//! multi-chunk ground grid with `mip_levels=6 + mip_scan_dist=64`.
//! This test asks the prior question: does the same bug appear in
//! a single-chunk world?
//!
//! Setup:
//! - One `vsid=512` chunk (large enough for deep-mip walk-out from
//!   the camera).
//! - Rolling-hill terrain via the same heightmap formula scene-demo
//!   uses (`terrain.rs::terrain_height`).
//! - `generate_mips(6)`.
//! - Camera placed in the chunk at world (256, 256, 138) looking
//!   exactly along +X (yaw=0, pitch=0 = horizontal). The horizon
//!   sits mid-screen; pixels above the horizon (screen y < H/2) must
//!   be sky.
//! - Render with `mip_levels=6 + mip_scan_dist=64`, same config as
//!   the live demo.
//!
//! If there are non-sky pixels in the upper half of the screen
//! (specifically the horizontal +X vanishing-point column), the bug
//! is single-chunk-reproducible and we can iterate without the
//! 32×32 build cost. If the upper half is clean, the bug requires
//! the multi-chunk path.

#![cfg(test)]
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use roxlap_core::opticast::{opticast, OpticastSettings};
use roxlap_core::rasterizer::ScratchPool;
use roxlap_core::scalar_rasterizer::ScalarRasterizer;
use roxlap_core::{Camera, GridView};
use roxlap_formats::edit::{set_spans_with_colfunc, SpanOp, Vspan};
use roxlap_formats::vxl::Vxl;

const W: u32 = 800;
const H: u32 = 600;
const VSID: u32 = 512;

/// Same heightmap formula as scene-demo's terrain.rs.
fn terrain_height(world_x: i32, world_y: i32) -> i32 {
    const BASE_Z: f32 = 200.0;
    const AMPLITUDE: f32 = 18.0;
    let fx = world_x as f32;
    let fy = world_y as f32;
    let h = AMPLITUDE
        * (0.5 * ((fx * 0.07).sin() + (fy * 0.06).sin())
            + 0.3 * ((fx * 0.15 + fy * 0.11).sin())
            + 0.2 * (((fx + fy) * 0.05).cos()));
    let z = BASE_Z - h;
    (z.round() as i32).clamp(80, 254)
}

fn build_single_chunk_ground(mip_levels: u32) -> Vxl {
    let n_cols = (VSID as usize) * (VSID as usize);
    let mut data: Vec<u8> = Vec::with_capacity(n_cols * 8);
    let mut column_offset: Vec<u32> = Vec::with_capacity(n_cols + 1);
    for _ in 0..n_cols {
        column_offset.push(u32::try_from(data.len()).expect("offset fits in u32"));
        data.extend_from_slice(&[0, 0, 0, 0]); // header
        data.extend_from_slice(&[0, 0, 0, 0]); // placeholder colour
    }
    column_offset.push(u32::try_from(data.len()).expect("offset fits in u32"));
    let mut vxl = Vxl {
        vsid: VSID,
        ipo: [0.0; 3],
        ist: [1.0, 0.0, 0.0],
        ihe: [0.0, 0.0, 1.0],
        ifo: [0.0, 1.0, 0.0],
        data: data.into_boxed_slice(),
        column_offset: column_offset.into_boxed_slice(),
        mip_base_offsets: Box::new([0, n_cols + 1]),
        vbit: Box::new([]),
        vbiti: 0,
    };
    vxl.reserve_edit_capacity(64 * 1024 * 1024);
    // Carve every column to all-air first.
    let mut carve_spans: Vec<Vspan> = Vec::with_capacity(n_cols);
    for y in 0..VSID {
        for x in 0..VSID {
            carve_spans.push(Vspan {
                x,
                y,
                z0: 0,
                z1: u8::MAX,
            });
        }
    }
    roxlap_formats::edit::set_spans(&mut vxl, &carve_spans, None);
    let mut spans: Vec<Vspan> = Vec::with_capacity((VSID * VSID) as usize);
    for ly in 0..VSID as i32 {
        for lx in 0..VSID as i32 {
            let surface_z = terrain_height(lx, ly);
            spans.push(Vspan {
                x: lx as u32,
                y: ly as u32,
                z0: surface_z as u8,
                z1: 254u8,
            });
        }
    }
    let colfunc = |_x: i32, _y: i32, z: i32| -> i32 {
        // Grass green for top, dirt for second band, stone below.
        if z <= 82 {
            i32::from_ne_bytes(0x80_5a_a0_3au32.to_ne_bytes()) // grass
        } else if z <= 88 {
            i32::from_ne_bytes(0x80_4e_70_8au32.to_ne_bytes()) // dirt
        } else {
            i32::from_ne_bytes(0x80_70_70_70u32.to_ne_bytes()) // stone
        }
    };
    set_spans_with_colfunc(&mut vxl, &spans, SpanOp::Insert, colfunc);
    if mip_levels > 1 {
        vxl.generate_mips(mip_levels);
    }
    vxl
}

fn render_pose(vxl: &Vxl, camera: &Camera, mip_levels: u32, mip_scan_dist: i32) -> Vec<u32> {
    let sky_color: u32 = 0xff_87_ce_eb;
    let mut fb = vec![sky_color; (W * H) as usize];
    let mut zb = vec![f32::INFINITY; fb.len()];
    let mut pool = ScratchPool::new(W, H, VSID);
    pool.set_skycast(i32::from_ne_bytes(sky_color.to_ne_bytes()), 0);
    pool.set_treat_z_max_as_air(true);
    let grid_view = GridView::from_single_vxl(vxl);
    let mut settings = OpticastSettings::for_oracle_framebuffer(W, H);
    settings.max_scan_dist = 2047;
    settings.mip_levels = mip_levels;
    settings.mip_scan_dist = mip_scan_dist;
    let mut rasterizer = ScalarRasterizer::new(&mut fb, &mut zb, W as usize, grid_view);
    let _ = opticast(&mut rasterizer, &mut pool, camera, &settings, grid_view);
    drop(rasterizer);
    fb
}

fn write_ppm(path: &str, fb: &[u32]) {
    let mut bytes = format!("P6\n{W} {H}\n255\n").into_bytes();
    for &px in fb {
        bytes.push(((px >> 16) & 0xff) as u8);
        bytes.push(((px >> 8) & 0xff) as u8);
        bytes.push((px & 0xff) as u8);
    }
    std::fs::write(path, bytes).expect("write ppm");
}

/// Count grass-green pixels in the upper half of the frame (above
/// the horizon). For an exactly-horizontal camera looking along +X,
/// upper-half pixels should be 100% sky.
fn count_grass_upper_half(fb: &[u32]) -> usize {
    let mut n = 0;
    for y in 0..(H as usize / 2) {
        for x in 0..(W as usize) {
            let px = fb[y * W as usize + x];
            let r = ((px >> 16) & 0xff) as i32;
            let g = ((px >> 8) & 0xff) as i32;
            let b = (px & 0xff) as i32;
            if g > r + 30 && g > b + 30 && (r + g + b) > 100 {
                n += 1;
            }
        }
    }
    n
}

fn count_non_sky(fb: &[u32]) -> usize {
    let sky_color: u32 = 0xff_87_ce_eb;
    fb.iter().filter(|&&p| p != sky_color).count()
}

fn is_grass(px: u32) -> bool {
    let r = ((px >> 16) & 0xff) as i32;
    let g = ((px >> 8) & 0xff) as i32;
    let b = (px & 0xff) as i32;
    g > r + 30 && g > b + 30 && (r + g + b) > 100
}

/// Pixels that are SKY in the baseline (`ml=1`) frame but GRASS in
/// the multi-mip frame. These are the bug pixels — the multi-mip
/// renderer painted terrain where the single-mip ground truth saw
/// sky.
fn count_beam_pixels(baseline: &[u32], multi_mip: &[u32]) -> usize {
    let sky_color: u32 = 0xff_87_ce_eb;
    baseline
        .iter()
        .zip(multi_mip.iter())
        .filter(|(&b, &m)| b == sky_color && is_grass(m))
        .count()
}

/// Return all (x, y) where the bug paints sky as grass. Useful for
/// piping into a follow-up trace.
fn beam_pixel_coords(baseline: &[u32], multi_mip: &[u32]) -> Vec<(u32, u32)> {
    let sky_color: u32 = 0xff_87_ce_eb;
    let mut out = Vec::new();
    for y in 0..H {
        for x in 0..W {
            let idx = (y * W + x) as usize;
            if baseline[idx] == sky_color && is_grass(multi_mip[idx]) {
                out.push((x, y));
            }
        }
    }
    out
}

/// Render the same scene at different mip configurations and report
/// how many "grass green" pixels appear in the UPPER HALF of the
/// frame. A horizontal camera should produce 0 such pixels.
#[test]
#[ignore = "expensive: builds vsid=512 chunk + generate_mips(6); dumps PPMs"]
fn axis_aligned_single_chunk_multi_mip_should_have_no_sky_terrain() {
    // Camera at chunk center, exactly horizontal, looking +X.
    let camera = Camera {
        pos: [256.0, 256.5, 138.0],
        right: [0.0, 1.0, 0.0],   // screen +x = world +y
        down: [0.0, 0.0, 1.0],    // screen +y = world +z (down)
        forward: [1.0, 0.0, 0.0], // world +x forward
    };

    // (Z) baseline — single-mip
    let vxl_no_mips = build_single_chunk_ground(1);
    let fb_z = render_pose(&vxl_no_mips, &camera, 1, 4);
    let upper_z = count_grass_upper_half(&fb_z);
    write_ppm("/tmp/beam-single-ml1.ppm", &fb_z);
    eprintln!(
        "(Z) mip_levels=1 (baseline): upper-half grass = {upper_z}, total non-sky = {}",
        count_non_sky(&fb_z)
    );

    // (A) multi-mip + far transition
    let vxl_mips = build_single_chunk_ground(6);
    let fb_a = render_pose(&vxl_mips, &camera, 6, 1024);
    let upper_a = count_grass_upper_half(&fb_a);
    write_ppm("/tmp/beam-single-ml6-msd1024.ppm", &fb_a);
    eprintln!(
        "(A) mip_levels=6 msd=1024: upper-half grass = {upper_a}, total non-sky = {}",
        count_non_sky(&fb_a)
    );

    // (B) multi-mip + live-demo config
    let fb_b = render_pose(&vxl_mips, &camera, 6, 64);
    let upper_b = count_grass_upper_half(&fb_b);
    write_ppm("/tmp/beam-single-ml6-msd64.ppm", &fb_b);
    eprintln!(
        "(B) mip_levels=6 msd=64:   upper-half grass = {upper_b}, total non-sky = {}",
        count_non_sky(&fb_b)
    );

    // (C) multi-mip + aggressive transition
    let fb_c = render_pose(&vxl_mips, &camera, 6, 8);
    let upper_c = count_grass_upper_half(&fb_c);
    write_ppm("/tmp/beam-single-ml6-msd8.ppm", &fb_c);
    eprintln!(
        "(C) mip_levels=6 msd=8:    upper-half grass = {upper_c}, total non-sky = {}",
        count_non_sky(&fb_c)
    );

    // Diagnostic only — we don't assert. The numbers tell us whether
    // single-chunk reproduces the bug or whether multi-chunk is
    // required.
}

/// 2-chunk repro: two adjacent CHUNK_SIZE_XY (128) chunks side-by-side,
/// rolling-hill terrain in both. Axis-aligned camera looks +X from
/// chunk-0 into chunk-1. If beams appear here, the bug is in the
/// multi-chunk path. Smallest possible reproducer.
#[test]
#[ignore = "expensive: builds 2-chunk world; dumps PPMs"]
fn axis_aligned_two_chunk_should_be_clean() {
    use glam::{DVec3, IVec3};
    use roxlap_scene::render::render_scene_composed;
    use roxlap_scene::{GridTransform, Scene, CHUNK_SIZE_XY};

    let cs_xy = CHUNK_SIZE_XY as i32;

    let mut scene = Scene::new();
    let grid_id = scene.add_grid(GridTransform::at(DVec3::ZERO));
    let grid = scene.grid_mut(grid_id).expect("grid");

    // Build terrain in chunks (0, 0, 0) and (1, 0, 0).
    for chx in 0..2 {
        let chunk_origin_x = chx * cs_xy;
        let mut spans: Vec<Vspan> = Vec::with_capacity((cs_xy * cs_xy) as usize);
        for ly in 0..cs_xy {
            for lx in 0..cs_xy {
                let wx = chunk_origin_x + lx;
                let wy = ly;
                let surface_z = terrain_height(wx, wy);
                spans.push(Vspan {
                    x: lx as u32,
                    y: ly as u32,
                    z0: surface_z as u8,
                    z1: 254u8,
                });
            }
        }
        let colfunc = |_x: i32, _y: i32, z: i32| -> i32 {
            if z <= 82 {
                i32::from_ne_bytes(0x80_5a_a0_3au32.to_ne_bytes())
            } else if z <= 88 {
                i32::from_ne_bytes(0x80_4e_70_8au32.to_ne_bytes())
            } else {
                i32::from_ne_bytes(0x80_70_70_70u32.to_ne_bytes())
            }
        };
        let vxl = grid.ensure_chunk(IVec3::new(chx, 0, 0));
        set_spans_with_colfunc(vxl, &spans, SpanOp::Insert, colfunc);
    }
    // Generate mips on every populated chunk.
    let chunk_keys: Vec<IVec3> = grid.chunks.keys().copied().collect();
    for k in chunk_keys {
        if let Some(chunk) = grid.chunk_mut(k) {
            chunk.generate_mips(6);
        }
    }

    // Camera in chunk (0, 0, 0), looking +X exactly. Position the
    // camera near the chunk-0/chunk-1 boundary so the ray crosses
    // into chunk 1 right after starting the walk.
    let camera = Camera {
        pos: [16.0, 64.5, 138.0],
        right: [0.0, 1.0, 0.0],
        down: [0.0, 0.0, 1.0],
        forward: [1.0, 0.0, 0.0],
    };

    let sky_color: u32 = 0xff_87_ce_eb;
    let render = |scene: &mut Scene, mip_levels: u32, mip_scan_dist: i32| -> Vec<u32> {
        let mut fb = vec![sky_color; (W * H) as usize];
        let mut zb = vec![f32::INFINITY; fb.len()];
        let mut pool = ScratchPool::new(W, H, 2 * CHUNK_SIZE_XY);
        pool.set_skycast(i32::from_ne_bytes(sky_color.to_ne_bytes()), 0);
        pool.set_treat_z_max_as_air(true);
        let mut settings = OpticastSettings::for_oracle_framebuffer(W, H);
        settings.max_scan_dist = 2047;
        settings.mip_levels = mip_levels;
        settings.mip_scan_dist = mip_scan_dist;
        let _ = render_scene_composed(
            &mut fb, &mut zb, W as usize, W, H, &mut pool, scene, &camera, &settings, sky_color,
            None,
        );
        fb
    };

    let fb_z = render(&mut scene, 1, 4);
    let upper_z = count_grass_upper_half(&fb_z);
    write_ppm("/tmp/beam-2chunk-ml1.ppm", &fb_z);
    eprintln!(
        "(Z) 2-chunk ml=1: upper grass = {upper_z}, total non-sky = {}",
        count_non_sky(&fb_z)
    );

    for msd in [8, 64, 256, 1024] {
        let fb = render(&mut scene, 6, msd);
        let upper = count_grass_upper_half(&fb);
        write_ppm(&format!("/tmp/beam-2chunk-ml6-msd{msd}.ppm"), &fb);
        eprintln!(
            "(  ) 2-chunk ml=6 msd={msd}: upper grass = {upper}, total non-sky = {}",
            count_non_sky(&fb)
        );
    }
}

/// Variant of the 2-chunk repro that mirrors the live demo's
/// centred-grid origin. `build_ground` in scene-demo places its
/// 32 chunks at chunk indices -16..16; ChunkGrid stores them with
/// `origin_chunk_xy = [-16, -16]`. A 2-chunk centred world has
/// `origin_chunk_xy = [-1, 0]` (or similar) — the negative chunk
/// index might trigger the bug while the [0, 0]..[1, 0] form
/// doesn't.
#[test]
#[ignore = "expensive: builds 2-chunk centred world; dumps PPMs"]
fn axis_aligned_two_chunk_centred_should_be_clean() {
    use glam::{DVec3, IVec3};
    use roxlap_scene::render::render_scene_composed;
    use roxlap_scene::{GridTransform, Scene, CHUNK_SIZE_XY};

    let cs_xy = CHUNK_SIZE_XY as i32;
    let mut scene = Scene::new();
    let grid_id = scene.add_grid(GridTransform::at(DVec3::ZERO));
    let grid = scene.grid_mut(grid_id).expect("grid");

    // Build terrain in centred chunks (-1, 0, 0) and (0, 0, 0).
    for chx in -1..1 {
        let chunk_origin_x = chx * cs_xy;
        let mut spans: Vec<Vspan> = Vec::with_capacity((cs_xy * cs_xy) as usize);
        for ly in 0..cs_xy {
            for lx in 0..cs_xy {
                let wx = chunk_origin_x + lx;
                let wy = ly;
                let surface_z = terrain_height(wx, wy);
                spans.push(Vspan {
                    x: lx as u32,
                    y: ly as u32,
                    z0: surface_z as u8,
                    z1: 254u8,
                });
            }
        }
        let colfunc = |_x: i32, _y: i32, z: i32| -> i32 {
            if z <= 82 {
                i32::from_ne_bytes(0x80_5a_a0_3au32.to_ne_bytes())
            } else if z <= 88 {
                i32::from_ne_bytes(0x80_4e_70_8au32.to_ne_bytes())
            } else {
                i32::from_ne_bytes(0x80_70_70_70u32.to_ne_bytes())
            }
        };
        let vxl = grid.ensure_chunk(IVec3::new(chx, 0, 0));
        set_spans_with_colfunc(vxl, &spans, SpanOp::Insert, colfunc);
    }
    let chunk_keys: Vec<IVec3> = grid.chunks.keys().copied().collect();
    for k in chunk_keys {
        if let Some(chunk) = grid.chunk_mut(k) {
            chunk.generate_mips(6);
        }
    }

    // Camera in chunk (-1, 0, 0) at the boundary, looking +X.
    let camera = Camera {
        pos: [-112.0, 64.5, 138.0],
        right: [0.0, 1.0, 0.0],
        down: [0.0, 0.0, 1.0],
        forward: [1.0, 0.0, 0.0],
    };

    let sky_color: u32 = 0xff_87_ce_eb;
    let render = |scene: &mut Scene, mip_levels: u32, mip_scan_dist: i32| -> Vec<u32> {
        let mut fb = vec![sky_color; (W * H) as usize];
        let mut zb = vec![f32::INFINITY; fb.len()];
        let mut pool = ScratchPool::new(W, H, 2 * CHUNK_SIZE_XY);
        pool.set_skycast(i32::from_ne_bytes(sky_color.to_ne_bytes()), 0);
        pool.set_treat_z_max_as_air(true);
        let mut settings = OpticastSettings::for_oracle_framebuffer(W, H);
        settings.max_scan_dist = 2047;
        settings.mip_levels = mip_levels;
        settings.mip_scan_dist = mip_scan_dist;
        let _ = render_scene_composed(
            &mut fb, &mut zb, W as usize, W, H, &mut pool, scene, &camera, &settings, sky_color,
            None,
        );
        fb
    };

    let fb_z = render(&mut scene, 1, 4);
    let upper_z = count_grass_upper_half(&fb_z);
    write_ppm("/tmp/beam-2chunkc-ml1.ppm", &fb_z);
    eprintln!(
        "(Z) 2-chunk-centred ml=1: upper grass = {upper_z}, total non-sky = {}",
        count_non_sky(&fb_z)
    );

    for msd in [8, 64, 256, 1024] {
        let fb = render(&mut scene, 6, msd);
        let upper = count_grass_upper_half(&fb);
        write_ppm(&format!("/tmp/beam-2chunkc-ml6-msd{msd}.ppm"), &fb);
        eprintln!(
            "(  ) 2-chunk-centred ml=6 msd={msd}: upper grass = {upper}, total non-sky = {}",
            count_non_sky(&fb)
        );
    }
}

/// 4×4 centred chunks — closer to the demo's 32×32 in shape but
/// 64× smaller. If beams appear here but not in 2-chunk, the bug
/// scales with chunk count / boundary crossings.
#[test]
#[ignore = "expensive: builds 4×4 centred world; dumps PPMs"]
fn axis_aligned_4x4_centred_grid() {
    use glam::{DVec3, IVec3};
    use roxlap_scene::render::render_scene_composed;
    use roxlap_scene::{GridTransform, Scene, CHUNK_SIZE_XY};

    let cs_xy = CHUNK_SIZE_XY as i32;
    let mut scene = Scene::new();
    let grid_id = scene.add_grid(GridTransform::at(DVec3::ZERO));
    let grid = scene.grid_mut(grid_id).expect("grid");

    for chy in -2..2 {
        for chx in -2..2 {
            let chunk_origin_x = chx * cs_xy;
            let chunk_origin_y = chy * cs_xy;
            let mut spans: Vec<Vspan> = Vec::with_capacity((cs_xy * cs_xy) as usize);
            for ly in 0..cs_xy {
                for lx in 0..cs_xy {
                    let wx = chunk_origin_x + lx;
                    let wy = chunk_origin_y + ly;
                    let surface_z = terrain_height(wx, wy);
                    spans.push(Vspan {
                        x: lx as u32,
                        y: ly as u32,
                        z0: surface_z as u8,
                        z1: 254u8,
                    });
                }
            }
            let colfunc = |_x: i32, _y: i32, z: i32| -> i32 {
                if z <= 82 {
                    i32::from_ne_bytes(0x80_5a_a0_3au32.to_ne_bytes())
                } else if z <= 88 {
                    i32::from_ne_bytes(0x80_4e_70_8au32.to_ne_bytes())
                } else {
                    i32::from_ne_bytes(0x80_70_70_70u32.to_ne_bytes())
                }
            };
            let vxl = grid.ensure_chunk(IVec3::new(chx, chy, 0));
            set_spans_with_colfunc(vxl, &spans, SpanOp::Insert, colfunc);
        }
    }
    let chunk_keys: Vec<IVec3> = grid.chunks.keys().copied().collect();
    for k in chunk_keys {
        if let Some(chunk) = grid.chunk_mut(k) {
            chunk.generate_mips(6);
        }
    }

    // Mimic the user's pose orientation. 4×4 world covers world
    // [-256..256]² so position the camera proportionally.
    let yaw = 2.355_796_326_794_885_6_f64;
    let pitch = -0.490_000_000_000_002_6_f64;
    let (sy, cy_) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    let camera = Camera {
        pos: [-46.0, -200.0, 138.0],
        right: [-sy, cy_, 0.0],
        down: [-cy_ * sp, -sy * sp, cp],
        forward: [cy_ * cp, sy * cp, sp],
    };

    let sky_color: u32 = 0xff_87_ce_eb;
    let render = |scene: &mut Scene, mip_levels: u32, mip_scan_dist: i32| -> Vec<u32> {
        let mut fb = vec![sky_color; (W * H) as usize];
        let mut zb = vec![f32::INFINITY; fb.len()];
        let mut pool = ScratchPool::new(W, H, 4 * CHUNK_SIZE_XY);
        pool.set_skycast(i32::from_ne_bytes(sky_color.to_ne_bytes()), 0);
        pool.set_treat_z_max_as_air(true);
        let mut settings = OpticastSettings::for_oracle_framebuffer(W, H);
        settings.max_scan_dist = 2047;
        settings.mip_levels = mip_levels;
        settings.mip_scan_dist = mip_scan_dist;
        let _ = render_scene_composed(
            &mut fb, &mut zb, W as usize, W, H, &mut pool, scene, &camera, &settings, sky_color,
            None,
        );
        fb
    };

    let fb_z = render(&mut scene, 1, 4);
    let upper_z = count_grass_upper_half(&fb_z);
    write_ppm("/tmp/beam-4x4-ml1.ppm", &fb_z);
    eprintln!(
        "(Z) 4x4 ml=1: upper grass = {upper_z}, total non-sky = {}",
        count_non_sky(&fb_z)
    );

    for msd in [8, 64, 256, 1024] {
        let fb = render(&mut scene, 6, msd);
        let upper = count_grass_upper_half(&fb);
        let beam = count_beam_pixels(&fb_z, &fb);
        let coords = beam_pixel_coords(&fb_z, &fb);
        write_ppm(&format!("/tmp/beam-4x4-ml6-msd{msd}.ppm"), &fb);
        eprintln!(
            "(  ) 4x4 ml=6 msd={msd}: upper grass = {upper}, total non-sky = {}, BEAM PIXELS = {beam}",
            count_non_sky(&fb)
        );
        if !coords.is_empty() {
            let xs: Vec<u32> = coords.iter().map(|&(x, _)| x).collect();
            let ys: Vec<u32> = coords.iter().map(|&(_, y)| y).collect();
            eprintln!(
                "    bbox x=[{}..{}], y=[{}..{}], first pixel ({}, {})",
                xs.iter().min().unwrap_or(&0),
                xs.iter().max().unwrap_or(&0),
                ys.iter().min().unwrap_or(&0),
                ys.iter().max().unwrap_or(&0),
                coords[0].0,
                coords[0].1,
            );
        }
    }
}

/// Same camera, but pitched slightly upward — exactly the failure
/// configuration in the live demo (pitch=-0.49). Verifies the bug
/// also requires non-horizontal pitch or appears regardless of it.
#[test]
#[ignore = "expensive: same as horizontal repro but with pitch"]
fn axis_aligned_single_chunk_pitched_up() {
    // Looking +X with pitch=-0.49 rad (up). yaw=0 → forward in +x.
    let pitch = -0.49_f32;
    let (sp, cp) = pitch.sin_cos();
    let camera = Camera {
        pos: [256.0, 256.5, 138.0],
        right: [0.0, 1.0, 0.0],
        down: [-(sp as f64), 0.0, cp as f64],
        forward: [cp as f64, 0.0, sp as f64],
    };

    let vxl_mips = build_single_chunk_ground(6);
    for (label, msd) in [
        ("msd8", 8),
        ("msd64", 64),
        ("msd512", 512),
        ("msd2047", 2047),
    ] {
        let fb = render_pose(&vxl_mips, &camera, 6, msd);
        let upper = count_grass_upper_half(&fb);
        write_ppm(&format!("/tmp/beam-single-pitched-{label}.ppm"), &fb);
        eprintln!("pitched up ml=6 {label}: upper-half grass = {upper}");
    }
    // Single-mip baseline.
    let vxl_no_mips = build_single_chunk_ground(1);
    let fb_z = render_pose(&vxl_no_mips, &camera, 1, 4);
    let upper_z = count_grass_upper_half(&fb_z);
    write_ppm("/tmp/beam-single-pitched-ml1.ppm", &fb_z);
    eprintln!("pitched up ml=1 baseline: upper-half grass = {upper_z}");
}
