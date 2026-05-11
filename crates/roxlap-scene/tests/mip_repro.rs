//! S4B.5: characterise the compilerle / generate_mips interaction
//! that makes edit-API-built chunks render all-sky when rendered
//! through mip-1+. Dump the slab bytes pre- / post-generate_mips
//! for a known-broken fixture so the diagnosis is text-greppable.

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
    drop(grid_view_check);

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
