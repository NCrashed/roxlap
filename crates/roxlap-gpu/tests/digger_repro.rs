//! Digger ghost-wall repro probe (docs/handover-stale-mips-volume-edits.md).
//!
//! Mirrors the monada digger terrain grid as faithfully as the headless
//! harness allows: vws = 16, NEGATIVE chunk indices in x and z (the
//! mirrored `(-x-1, y, -z-1)` addressing puts the terrain in chunks
//! (-1,0,-1) and (-1,0,0)), a slab spanning the chz boundary, a carved
//! shaft crossing that boundary, camera near, `mip_scan_dist = 8192`.
//!
//! The centre ray looks straight down the shaft: correct render sees the
//! shaft floor (world z 144, depth ≈ 444); a ghost wall at the chunk
//! boundary would read depth ≈ 300; a ghost at the surface ≈ 172.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names
)]

use std::sync::Mutex;

use roxlap_formats::edit::set_rect as vxl_set_rect;
use roxlap_formats::vxl::Vxl;
use roxlap_gpu::{
    decompress_chunk, Camera, GpuInitError, GpuRendererSettings, GpuSceneResident, GridUpload,
    GridWorldTransform, HeadlessGpu, HeadlessSceneRenderer, SceneUpload,
};

static GPU_TEST_LOCK: Mutex<()> = Mutex::new(());

fn try_init() -> Option<(HeadlessGpu, std::sync::MutexGuard<'static, ()>)> {
    let guard = GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    match HeadlessGpu::new_blocking(GpuRendererSettings::default()) {
        Ok(gpu) => Some((gpu, guard)),
        Err(GpuInitError::NoAdapter) => {
            eprintln!("[skip] no GPU adapter reachable");
            None
        }
        Err(e) => {
            eprintln!("[skip] GPU init failed ({e})");
            None
        }
    }
}

/// `vsid × vsid` chunk, every column solid over local `z ∈ [top, bot]`.
fn slab_chunk(vsid: u32, top: u8, bot: u8) -> Vxl {
    let n_cols = (vsid as usize) * (vsid as usize);
    let n_vox = (bot - top + 1) as usize;
    let mut data: Vec<u8> = Vec::with_capacity(n_cols * (4 + n_vox * 4));
    let mut column_offset: Vec<u32> = Vec::with_capacity(n_cols + 1);
    let bgra = [0x00u8, 0x80, 0xff, 0x80]; // orange, brightness 1.0
    for _ in 0..n_cols {
        column_offset.push(u32::try_from(data.len()).expect("offset fits"));
        data.extend_from_slice(&[0, top, bot, 0]);
        for _ in 0..n_vox {
            data.extend_from_slice(&bgra);
        }
    }
    column_offset.push(u32::try_from(data.len()).expect("offset fits"));
    Vxl {
        vsid,
        ipo: [0.0; 3],
        ist: [1.0, 0.0, 0.0],
        ihe: [0.0, 0.0, 1.0],
        ifo: [0.0, 1.0, 0.0],
        data: data.into_boxed_slice(),
        column_offset: column_offset.into_boxed_slice(),
        mip_base_offsets: Box::new([0, n_cols + 1]),
        vbit: Box::new([]),
        vbiti: 0,
    }
}

fn render_depth_at(msd: f32, z_clip: Option<i32>) -> (f32, Vec<f32>, u32, u32) {
    let Some((gpu, _lock)) = try_init() else {
        panic!("no adapter");
    };
    let vsid = 32u32;
    // Terrain slab grid z ∈ [-8, 15] (24 voxels thick, world -128..+256):
    //   chunk (-1,0,-1): local z 248..255  (grid z -8..-1)
    //   chunk (-1,0, 0): local z 0..15     (grid z 0..15)
    let mut a = slab_chunk(vsid, 248, 255);
    let mut b = slab_chunk(vsid, 0, 15);
    // Carve a 2×2 shaft at local x/y 16..17 (grid x -16..-15, y 16..17),
    // crossing the chz boundary: grid z -8..8 → floor at grid z 9.
    a.reserve_edit_capacity(64 * 1024);
    b.reserve_edit_capacity(64 * 1024);
    vxl_set_rect(&mut a, [16, 16, 248], [17, 17, 255], None);
    vxl_set_rect(&mut b, [16, 16, 0], [17, 17, 8], None);

    let up_a = decompress_chunk(&a);
    let up_b = decompress_chunk(&b);
    // Probe the mip-0 solid occupancy of the carved shaft column (16,16):
    // 8 words per column, bit z%32 of word z/32.
    let probe = |up: &roxlap_gpu::ChunkUpload, label: &str, zs: &[u32]| {
        let col = (16 + 16 * vsid) as usize;
        for &z in zs {
            let w = col * 8 + (z / 32) as usize;
            let tex = up.mips[0].occupancy[w] & (1 << (z % 32)) != 0;
            let sol = up.mips[0].solid_occupancy[w] & (1 << (z % 32)) != 0;
            eprintln!("chunk {label} local z {z}: textured={tex} solid={sol}");
        }
    };
    probe(&up_a, "A(chz=-1)", &[247, 250, 254, 255]);
    probe(&up_b, "B(chz= 0)", &[0, 8, 9, 15]);
    let grid = GridUpload {
        vsid,
        origin_chunk: [-1, 0, -1],
        chunks_dims: [1, 1, 2],
        pool_dims: [1, 1, 2],
        chunks: vec![([-1, 0, -1], up_a), ([-1, 0, 0], up_b)],
    };
    let scene = GpuSceneResident::upload(&gpu.device, &SceneUpload { grids: vec![grid] });

    let (w, h) = (64u32, 64u32);
    let renderer = HeadlessSceneRenderer::new(&gpu.device, &gpu.queue, w, h);
    // Hover above the shaft mouth (surface at world z -128), looking
    // straight DOWN. right × down = [0,0,1] = forward (RH basis).
    let cam = Camera {
        position: [-240.0, 272.0, -300.0],
        right: [1.0, 0.0, 0.0],
        down: [0.0, 1.0, 0.0],
        forward: [0.0, 0.0, 1.0],
        fov_y_rad: 60f32.to_radians(),
    };
    let xf = GridWorldTransform {
        voxel_world_size: 16.0,
        z_clip,
        ..GridWorldTransform::default()
    };
    let depth = renderer.render_depth_with_transforms(
        &gpu.device,
        &gpu.queue,
        &scene,
        &[cam],
        &[xf],
        cam.fov_y_rad,
        64,
        msd,
    );
    let centre = (h / 2 * w + w / 2) as usize;
    (depth[centre], depth, w, h)
}

fn ascii_depth(depth: &[f32], w: u32, h: u32) {
    // '.' sky/infinite, digits = depth/100 clamped 0..9
    for y in (0..h).step_by(2) {
        let mut line = String::new();
        for x in (0..w).step_by(1) {
            let d = depth[(y * w + x) as usize];
            if d.is_finite() && d < 9000.0 {
                let v = ((d / 100.0) as i32).clamp(0, 9);
                line.push(char::from_digit(v as u32, 10).unwrap());
            } else {
                line.push('.');
            }
        }
        eprintln!("{line}");
    }
}

#[test]
#[ignore = "KNOWN RED: uncarvable local-z-255 bedrock layer at chz boundaries \
            (docs/handover-stale-mips-volume-edits.md, root cause section). \
            Run with --ignored to reproduce; must pass once carve-through-floor lands."]
fn digger_shaft_msd_8192_sees_floor() {
    let (d, depth, w, h) = render_depth_at(8192.0, None);
    eprintln!("=== msd=8192, no z_clip: centre depth {d} (expect ≈444) ===");
    ascii_depth(&depth, w, h);
    assert!(
        (d - 444.0).abs() < 20.0,
        "centre ray must reach the shaft floor at ≈444, got {d} \
         (≈300 = ghost wall at the chz boundary, ≈172 = surface never carved)"
    );
}

#[test]
#[ignore = "KNOWN RED: same uncarvable z=255 layer as digger_shaft_msd_8192_sees_floor \
            (proves the ghost is threshold-independent — 64 and 8192 fail identically)."]
fn digger_shaft_msd_default64_sees_floor() {
    let (d, depth, w, h) = render_depth_at(64.0, None);
    eprintln!("=== msd=64 (RenderOptions default): centre depth {d} ===");
    ascii_depth(&depth, w, h);
    assert!(
        (d - 444.0).abs() < 20.0,
        "msd=64 with vws=16 projected-size LOD should still be mip-0 here, got {d}"
    );
}

#[test]
fn digger_shaft_msd_tiny_forces_coarse() {
    // Force mips to SEE what the ghost looks like (documentation run).
    let (d, depth, w, h) = render_depth_at(1.0, None);
    eprintln!("=== msd=1 (forced coarse): centre depth {d} ===");
    ascii_depth(&depth, w, h);
}

#[test]
fn digger_shaft_with_deck_clip_sees_floor() {
    // Deck clip like monada's deck_clip while drilling in the shaft:
    // hide everything ABOVE grid z 2 (z < 2 cut).
    let (d, depth, w, h) = render_depth_at(8192.0, Some(2));
    eprintln!("=== msd=8192, z_clip=2: centre depth {d} (expect ≈444) ===");
    ascii_depth(&depth, w, h);
    assert!(
        (d - 444.0).abs() < 20.0,
        "with deck clip the shaft floor must still read ≈444, got {d}"
    );
}
