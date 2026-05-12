//! S4B.5: diagnostic dump for the multi-mip column-step bug
//! (fixed 2026-05-12 — see `grouscan.rs::phase_after_delete_kept_presync`).
//!
//! Hypothesis pre-fix was that `phase_remiporend`'s cf-halving was
//! the bug; the actual cause was the single-chunk column-step's
//! `cy * vsid + cx` index recompute clobbering `ixy_sptr_col_idx`
//! back into mip-0's sub-table after each step in mip-N. Fix is
//! to trust the `wrapping_add(step)` result in mip-N (gmipcnt > 0).
//!
//! Reading from low to high `mip_scan_dist` shows the expected
//! shape: at msd small enough that the 3-mip depth ladder
//! (msd → 2·msd → 4·msd) can't reach the floor, no pixels render
//! (this is correct; the budget is exhausted before geometry).
//! At higher msd, mip transitions still fire but the floor is
//! reachable at mip-1 / mip-2 and the rendered pixel count
//! climbs to the mip-0 baseline (21775 for the solid-floor fixture).

use glam::IVec3;
use roxlap_formats::vxl::slng;
use roxlap_scene::{Grid, GridTransform, CHUNK_SIZE_XY};

#[test]
#[ignore = "diagnostic dump, not a correctness check — invoke with --ignored"]
fn dump_mip_bytes_for_solid_floor_chunk() {
    // Build the same chunk the ignored test in render.rs hits.
    let mut grid = Grid::new(GridTransform::identity());
    grid.set_rect(
        IVec3::new(0, 0, 100),
        IVec3::new((CHUNK_SIZE_XY - 1) as i32, (CHUNK_SIZE_XY - 1) as i32, 254),
        Some(0x80_88_88_88),
    );
    let chunk = grid.chunks.get_mut(&IVec3::ZERO).unwrap();

    // Inspect mip-0 column at (0, 0). Should encode the all-solid
    // z=100..254 floor + bedrock placeholder z=255.
    let mip0_offsets = chunk.column_offset_for_mip(0);
    let col0_start = mip0_offsets[0] as usize;
    let col0_len = slng(&chunk.data[col0_start..]);
    eprintln!("=== MIP-0 column (0,0): {col0_len} bytes ===");
    for i in (0..col0_len).step_by(4) {
        let b = &chunk.data[col0_start + i..col0_start + i + 4];
        eprintln!(
            "  [{i:4}] nextptr/z1/z1c/dummy or RGBA: {:3} {:3} {:3} {:3}",
            b[0], b[1], b[2], b[3]
        );
    }

    // (Z) baseline render WITHOUT generate_mips. Camera at z=80
    // (20 above the floor at z=100) so bottom-of-screen rays
    // actually reach the floor inside the chunk's y-extent.
    {
        use roxlap_core::{
            opticast::{opticast, OpticastSettings},
            rasterizer::ScratchPool,
            scalar_rasterizer::ScalarRasterizer,
            Camera, GridView,
        };
        const XRES: u32 = 320;
        const YRES: u32 = 200;
        let mut fb = vec![0u32; (XRES as usize) * (YRES as usize)];
        let mut zb = vec![0.0f32; fb.len()];
        let mut pool = ScratchPool::new(XRES, YRES, CHUNK_SIZE_XY);
        let sky_color: u32 = 0xff_87_ce_eb;
        pool.set_skycast(i32::from_ne_bytes(sky_color.to_ne_bytes()), 0);
        fb.fill(sky_color);
        zb.fill(f32::INFINITY);
        let camera = Camera {
            pos: [64.0, 0.0, 80.0],
            right: [-1.0, 0.0, 0.0],
            down: [0.0, 0.0, 1.0],
            forward: [0.0, 1.0, 0.0],
        };
        let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
        let grid_view = GridView::from_single_vxl(chunk);
        let mut rasterizer = ScalarRasterizer::new(&mut fb, &mut zb, XRES as usize, grid_view);
        let _ = opticast(&mut rasterizer, &mut pool, &camera, &settings, grid_view);
        drop(rasterizer);
        let non_sky = fb.iter().filter(|&&p| p != sky_color).count();
        eprintln!("\n(Z) BEFORE generate_mips, cam z=80: {non_sky} non-sky pixels");
        // Dump PPM.
        let mut ppm = format!("P6\n{XRES} {YRES}\n255\n").into_bytes();
        for px in &fb {
            let [b, g, r, _a] = px.to_le_bytes();
            ppm.extend_from_slice(&[r, g, b]);
        }
        std::fs::write("/tmp/mip_repro_before.ppm", &ppm).expect("write ppm");
        eprintln!("    wrote /tmp/mip_repro_before.ppm");
    }

    chunk.generate_mips(3);
    eprintln!(
        "\n=== mip_count after generate_mips(3) = {} ===",
        chunk.mip_count()
    );
    eprintln!(
        "    chunk.mip_base_offsets = {:?} (len={})",
        chunk.mip_base_offsets,
        chunk.mip_base_offsets.len()
    );
    let grid_view_check = roxlap_core::GridView::from_single_vxl(&*chunk);
    eprintln!(
        "    grid_view.mip_base_offsets.len() = {}",
        grid_view_check.mip_base_offsets.len()
    );
    let _ = grid_view_check;

    // Hypothesis: mip-N halves z coords, so a camera at z=64 (above
    // the mip-0 floor z=100..254 → in air) lands at z=64 in the mip-1
    // coord system whose floor is z=50..127 → inside solid. Verify
    // with a direct render at a "safe" camera z that stays above the
    // floor in every mip.
    use roxlap_core::{
        opticast::{opticast, OpticastSettings},
        rasterizer::ScratchPool,
        scalar_rasterizer::ScalarRasterizer,
        Camera, GridView,
    };
    const XRES: u32 = 320;
    const YRES: u32 = 200;
    let mut fb = vec![0u32; (XRES as usize) * (YRES as usize)];
    let mut zb = vec![0.0f32; fb.len()];
    let mut pool = ScratchPool::new(XRES, YRES, CHUNK_SIZE_XY);
    let sky_color: u32 = 0xff_87_ce_eb;
    pool.set_skycast(i32::from_ne_bytes(sky_color.to_ne_bytes()), 0);
    fb.fill(sky_color);
    zb.fill(f32::INFINITY);
    // Camera at z=10 (above mip-N floor for N up to 3: floor halves
    // 100→50→25→12, z=10 stays above).
    let camera = Camera {
        pos: [64.0, 0.0, 80.0],
        right: [-1.0, 0.0, 0.0],
        down: [0.0, 0.0, 1.0],
        forward: [0.0, 1.0, 0.0],
    };
    eprintln!("\n>>> ABOUT TO RUN (B) MULTI-MIP <<<");
    // (B) sweep mip_scan_dist on the SINGLE-SLAB fixture too.
    for msd in [4i32, 16, 64, 128, 256, 1024] {
        let mut settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
        settings.mip_levels = 3;
        settings.mip_scan_dist = msd;
        fb.fill(sky_color);
        zb.fill(f32::INFINITY);
        let grid_view = GridView::from_single_vxl(chunk);
        let mut rasterizer = ScalarRasterizer::new(&mut fb, &mut zb, XRES as usize, grid_view);
        let _ = opticast(&mut rasterizer, &mut pool, &camera, &settings, grid_view);
        drop(rasterizer);
        let non_sky = fb.iter().filter(|&&p| p != sky_color).count();
        eprintln!("(B) single-slab mip_scan_dist={msd:5}: {non_sky:6} non-sky pixels");
    }
}

/// (F) Dump mip-5 bytes of a chunk where terrain reaches z=254
/// (one voxel below the bedrock z=255). At mip-5 both terrain top
/// and bedrock placeholder end up at the same z (7), which can
/// trip my bedrock-z-suppression check in `phase_draw_flor`.
#[test]
#[ignore = "diagnostic — invoke with --ignored"]
fn dump_terrain_reaches_z_254_mip5() {
    let mut grid = Grid::new(GridTransform::identity());
    // Terrain z=100..254 + bedrock placeholder at z=255.
    grid.set_rect(
        IVec3::new(0, 0, 100),
        IVec3::new((CHUNK_SIZE_XY - 1) as i32, (CHUNK_SIZE_XY - 1) as i32, 254),
        Some(0x80_88_88_88),
    );
    let chunk = grid.chunks.get_mut(&IVec3::ZERO).unwrap();
    chunk.generate_mips(6);

    for level in 0..6 {
        let off = chunk.column_offset_for_mip(level)[0] as usize;
        let len = slng(&chunk.data[off..]);
        eprintln!("=== MIP-{level} column (0,0): {len} bytes ===");
        for i in (0..len).step_by(4) {
            let b = &chunk.data[off + i..off + i + 4];
            eprintln!("  [{i:4}] {:3} {:3} {:3} {:3}", b[0], b[1], b[2], b[3]);
        }
    }
}

/// (G) Realistic ground-chunk-like fixture: terrain at z=200..254
/// (surface at z=200, bedrock at z=255, air above). Replicates the
/// demo's flat-terrain columns to see how mip-N encodes them.
#[test]
#[ignore = "diagnostic — invoke with --ignored"]
fn dump_realistic_ground_column_mips() {
    let mut grid = Grid::new(GridTransform::identity());
    grid.set_rect(
        IVec3::new(0, 0, 200),
        IVec3::new((CHUNK_SIZE_XY - 1) as i32, (CHUNK_SIZE_XY - 1) as i32, 254),
        Some(0x80_44_e0_44), // green-ish (matches demo grass)
    );
    let chunk = grid.chunks.get_mut(&IVec3::ZERO).unwrap();
    chunk.generate_mips(6);
    for level in 0..6 {
        let off = chunk.column_offset_for_mip(level)[0] as usize;
        let len = slng(&chunk.data[off..]);
        eprintln!("=== MIP-{level} column (0,0): {len} bytes ===");
        for i in (0..len).step_by(4) {
            let b = &chunk.data[off + i..off + i + 4];
            eprintln!("  [{i:4}] {:3} {:3} {:3} {:3}", b[0], b[1], b[2], b[3]);
        }
    }
}

/// (E) End-to-end: build a SMALL multi-chunk ship-like grid where
/// some chunks are all-air-plus-bedrock and some have a single
/// floor voxel near the top, render through it at mip-N from an
/// OOB-XY camera, and assert no dark pixels appear. This catches
/// the user-reported "thin black ring walls" mip-N artifact for
/// the ship grid's all-air perimeter chunks.
#[test]
#[ignore = "diagnostic — invoke with --ignored"]
fn all_air_chunks_mip_n_no_dark_pixels() {
    use roxlap_core::{
        opticast::{opticast, OpticastSettings},
        rasterizer::ScratchPool,
        scalar_rasterizer::ScalarRasterizer,
        Camera, ChunkGrid, GridView,
    };

    // 2×2 grid of chunks. Chunks (0,0) and (1,1) have a "ship" voxel
    // at z=50; (0,1) and (1,0) are all-air. Mirrors the saucer-edge
    // situation where most ship chunks have no ship voxels.
    let mut grid = Grid::new(GridTransform::identity());
    grid.set_voxel(IVec3::new(10, 10, 50), Some(0x80_88_88_88));
    grid.set_voxel(IVec3::new(140, 140, 50), Some(0x80_88_88_88));
    // Force materialise the other two chunks too as all-air.
    let _ = grid.ensure_chunk(IVec3::new(0, 1, 0));
    let _ = grid.ensure_chunk(IVec3::new(1, 0, 0));

    // Mip every chunk.
    let chunk_idxs: Vec<IVec3> = grid.chunks.keys().copied().collect();
    for idx in &chunk_idxs {
        let c = grid.chunks.get_mut(idx).unwrap();
        c.generate_mips(4);
    }

    // Build Approach B view. Camera OOB-XY (y < 0), looking +y at
    // the grid from outside.
    let backing = grid.chunk_xy_backing().expect("at least one chunk");
    let cg = ChunkGrid {
        chunks: &backing.chunks,
        origin_chunk_xy: backing.origin_chunk_xy,
        chunks_x: backing.chunks_x,
        chunks_y: backing.chunks_y,
    };
    let view = GridView::from_chunk_grid(&cg, CHUNK_SIZE_XY);

    const XRES: u32 = 320;
    const YRES: u32 = 240;
    let mut fb = vec![0u32; (XRES as usize) * (YRES as usize)];
    let mut zb = vec![f32::INFINITY; fb.len()];
    let mut pool = ScratchPool::new(XRES, YRES, 2 * CHUNK_SIZE_XY);
    let sky_color: u32 = 0xff_87_ce_eb;
    pool.set_skycast(i32::from_ne_bytes(sky_color.to_ne_bytes()), 0);
    pool.set_treat_z_max_as_air(true);
    fb.fill(sky_color);
    let camera = Camera {
        pos: [128.0, -50.0, 30.0],
        right: [-1.0, 0.0, 0.0],
        down: [0.0, 0.0, 1.0],
        forward: [0.0, 1.0, 0.0],
    };
    let mut settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
    settings.mip_levels = 4;
    settings.mip_scan_dist = 16;
    let mut raster = ScalarRasterizer::new(&mut fb, &mut zb, XRES as usize, view);
    let _ = opticast(&mut raster, &mut pool, &camera, &settings, view);
    drop(raster);

    let dark = fb
        .iter()
        .filter(|&&p| {
            let r = (p >> 16) & 0xff;
            let g = (p >> 8) & 0xff;
            let b = p & 0xff;
            r + g + b < 60
        })
        .count();
    let non_sky = fb.iter().filter(|&&p| p != sky_color).count();
    eprintln!("all-air mip-N render: non-sky {non_sky}, dark {dark}");

    // Dump for visual inspection.
    let mut ppm = format!("P6\n{XRES} {YRES}\n255\n").into_bytes();
    for &px in &fb {
        ppm.push(((px >> 16) & 0xff) as u8);
        ppm.push(((px >> 8) & 0xff) as u8);
        ppm.push((px & 0xff) as u8);
    }
    std::fs::write("/tmp/all_air_mip_n.ppm", &ppm).expect("write ppm");

    assert_eq!(
        dark, 0,
        "all-air multi-chunk mip-N rendered {dark} dark pixels — bedrock placeholder leaking"
    );
}

/// (D) Dump bytes of an ALL-AIR chunk (just the bedrock placeholder).
/// Used to inspect what `generate_mips` does to the bedrock-only
/// encoding the ship-grid's around-the-saucer chunks have.
#[test]
#[ignore = "diagnostic — invoke with --ignored"]
fn dump_all_air_chunk_mip_bytes() {
    let mut grid = Grid::new(GridTransform::identity());
    grid.set_voxel(IVec3::new(0, 0, 0), Some(0x80_88_88_88)); // any voxel to materialise the chunk
    grid.set_voxel(IVec3::new(0, 0, 0), None); // then unset → back to all air
    let chunk = grid.chunks.get_mut(&IVec3::ZERO).unwrap();

    let col0_start = chunk.column_offset_for_mip(0)[0] as usize;
    let col0_len = slng(&chunk.data[col0_start..]);
    eprintln!("=== ALL-AIR MIP-0 column (0,0): {col0_len} bytes ===");
    for i in (0..col0_len).step_by(4) {
        let b = &chunk.data[col0_start + i..col0_start + i + 4];
        eprintln!("  [{i:4}] {:3} {:3} {:3} {:3}", b[0], b[1], b[2], b[3]);
    }

    chunk.generate_mips(4);
    let mip1_start = chunk.column_offset_for_mip(1)[0] as usize;
    let mip1_len = slng(&chunk.data[mip1_start..]);
    eprintln!("\n=== ALL-AIR MIP-1 column (0,0): {mip1_len} bytes ===");
    for i in (0..mip1_len).step_by(4) {
        let b = &chunk.data[mip1_start + i..mip1_start + i + 4];
        eprintln!("  [{i:4}] {:3} {:3} {:3} {:3}", b[0], b[1], b[2], b[3]);
    }
    let mip2_start = chunk.column_offset_for_mip(2)[0] as usize;
    let mip2_len = slng(&chunk.data[mip2_start..]);
    eprintln!("\n=== ALL-AIR MIP-2 column (0,0): {mip2_len} bytes ===");
    for i in (0..mip2_len).step_by(4) {
        let b = &chunk.data[mip2_start + i..mip2_start + i + 4];
        eprintln!("  [{i:4}] {:3} {:3} {:3} {:3}", b[0], b[1], b[2], b[3]);
    }
}

/// (C) Same fixture but with an AIR GAP between the surface and
/// the bedrock so compilerle emits a multi-slab column. If
/// multi-mip works on this, the bug is specific to single-slab
/// columns (compilerle's "buried interior" encoding).
#[test]
#[ignore = "diagnostic — invoke with --ignored"]
fn multi_mip_renders_floor_with_air_gap_below() {
    use roxlap_core::{
        opticast::{opticast, OpticastSettings},
        rasterizer::ScratchPool,
        scalar_rasterizer::ScalarRasterizer,
        Camera, GridView,
    };

    let mut grid = Grid::new(GridTransform::identity());
    // Solid only at z=100..150 — leaves z=151..254 as AIR
    // (preserved from empty_chunk_vxl's all-air carve), z=255 is
    // the bedrock placeholder.
    grid.set_rect(
        IVec3::new(0, 0, 100),
        IVec3::new((CHUNK_SIZE_XY - 1) as i32, (CHUNK_SIZE_XY - 1) as i32, 150),
        Some(0x80_88_88_88),
    );
    let chunk = grid.chunks.get_mut(&IVec3::ZERO).unwrap();

    let col0_start = chunk.column_offset_for_mip(0)[0] as usize;
    let col0_len = slng(&chunk.data[col0_start..]);
    eprintln!("=== AIR-GAP MIP-0 column (0,0): {col0_len} bytes ===");
    for i in (0..col0_len).step_by(4) {
        let b = &chunk.data[col0_start + i..col0_start + i + 4];
        eprintln!("  [{i:4}] {:3} {:3} {:3} {:3}", b[0], b[1], b[2], b[3]);
    }

    chunk.generate_mips(3);
    let col0_m1_start = chunk.column_offset_for_mip(1)[0] as usize;
    let col0_m1_len = slng(&chunk.data[col0_m1_start..]);
    eprintln!("\n=== AIR-GAP MIP-1 column (0,0): {col0_m1_len} bytes ===");
    for i in (0..col0_m1_len).step_by(4) {
        let b = &chunk.data[col0_m1_start + i..col0_m1_start + i + 4];
        eprintln!("  [{i:4}] {:3} {:3} {:3} {:3}", b[0], b[1], b[2], b[3]);
    }

    const XRES: u32 = 320;
    const YRES: u32 = 200;
    let mut fb = vec![0u32; (XRES as usize) * (YRES as usize)];
    let mut zb = vec![0.0f32; fb.len()];
    let mut pool = ScratchPool::new(XRES, YRES, CHUNK_SIZE_XY);
    let sky_color: u32 = 0xff_87_ce_eb;
    pool.set_skycast(i32::from_ne_bytes(sky_color.to_ne_bytes()), 0);
    fb.fill(sky_color);
    zb.fill(f32::INFINITY);
    let camera = Camera {
        pos: [64.0, 0.0, 80.0],
        right: [-1.0, 0.0, 0.0],
        down: [0.0, 0.0, 1.0],
        forward: [0.0, 1.0, 0.0],
    };
    // Sweep mip_scan_dist to find the threshold where rendering starts working.
    for msd in [4i32, 16, 64, 128, 256, 1024] {
        let mut settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
        settings.mip_levels = 3;
        settings.mip_scan_dist = msd;
        fb.fill(sky_color);
        zb.fill(f32::INFINITY);
        let grid_view = GridView::from_single_vxl(chunk);
        let mut rasterizer = ScalarRasterizer::new(&mut fb, &mut zb, XRES as usize, grid_view);
        let _ = opticast(&mut rasterizer, &mut pool, &camera, &settings, grid_view);
        drop(rasterizer);
        let non_sky = fb.iter().filter(|&&p| p != sky_color).count();
        eprintln!("(C) air-gap mip_scan_dist={msd:5}: {non_sky:6} non-sky pixels");
    }

    // Dump mip-1 column at (0, 0). Should encode the same shape at
    // half resolution: z=50..127 solid (or maybe with the bedrock
    // collapsed to z=127).
    let mip1_offsets = chunk.column_offset_for_mip(1);
    eprintln!(
        "mip-1 column_offset has {} entries (vsid_1 = {} → {}² + 1)",
        mip1_offsets.len(),
        CHUNK_SIZE_XY / 2,
        (CHUNK_SIZE_XY / 2) * (CHUNK_SIZE_XY / 2)
    );
    let col0_m1_start = mip1_offsets[0] as usize;
    let col0_m1_len = slng(&chunk.data[col0_m1_start..]);
    eprintln!("=== MIP-1 column (0,0): {col0_m1_len} bytes ===");
    for i in (0..col0_m1_len).step_by(4) {
        let b = &chunk.data[col0_m1_start + i..col0_m1_start + i + 4];
        eprintln!(
            "  [{i:4}] nextptr/z1/z1c/dummy or RGBA: {:3} {:3} {:3} {:3}",
            b[0], b[1], b[2], b[3]
        );
    }
}
