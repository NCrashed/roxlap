//! GPU.11.0 gate — headless scene-DDA render.
//!
//! Stands up a real GPU device, uploads a one-grid scene whose every
//! column has a textured floor voxel, and renders it through the
//! `scene_dda.wgsl` compute pipeline that now carries the full mip
//! ladder per slot (GPU.11.0). The shader still marches mip-0, so a
//! correct render proves:
//!   1. `scene_dda.wgsl` compiles with the grown `GridStaticMeta`.
//!   2. The 112-byte std430 struct layout matches the Rust upload.
//!   3. The new per-slot occupancy / color_offsets *strides* still
//!      address mip-0 byte-identically (a floor voxel reads its
//!      colour back through the strided layout).
//!
//! Skips silently if no Vulkan/Metal/DX12 adapter is reachable.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::redundant_closure_for_method_calls
)]

use std::sync::Mutex;

use roxlap_formats::vxl::Vxl;
use roxlap_gpu::{
    decompress_chunk, Camera, GpuInitError, GpuRendererSettings, GpuSceneResident, GridUpload,
    HeadlessGpu, HeadlessSceneRenderer, SceneUpload,
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

/// `vsid × vsid` chunk: one textured floor voxel per column at
/// `z = 100`, colour `0x80ff_8000` (A=0x80 → brightness 1.0,
/// R=0xff, G=0x80, B=0x00). decompress_chunk builds its mip ladder.
fn floor_chunk(vsid: u32) -> Vxl {
    let n_cols = (vsid as usize) * (vsid as usize);
    let mut data: Vec<u8> = Vec::with_capacity(n_cols * 8);
    let mut column_offset: Vec<u32> = Vec::with_capacity(n_cols + 1);
    let bgra = [0x00u8, 0x80, 0xff, 0x80];
    for _ in 0..n_cols {
        column_offset.push(u32::try_from(data.len()).expect("offset fits"));
        data.extend_from_slice(&[0, 100, 100, 0]); // nextptr=0, z1=100, z1c=100, z0=0
        data.extend_from_slice(&bgra);
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

#[test]
fn scene_dda_renders_floor_through_mip_layout() {
    let Some((gpu, _lock)) = try_init() else {
        return;
    };
    eprintln!("scene_render: adapter = {}", gpu.adapter_info);

    let vsid = 64u32;
    let chunk = decompress_chunk(&floor_chunk(vsid));
    // Sanity: the mip ladder was built (GPU.11.0 plumbing).
    assert!(chunk.mips.len() >= 2, "expected a mip ladder");

    let grid = GridUpload {
        vsid,
        origin_chunk: [0, 0, 0],
        chunks_dims: [1, 1, 1],
        pool_dims: [1, 1, 1],
        chunks: vec![([0, 0, 0], chunk)],
    };
    let scene = GpuSceneResident::upload(&gpu.device, &SceneUpload { grids: vec![grid] });
    eprintln!("scene_render: resident {} bytes", scene.resident_bytes());

    let (w, h) = (64u32, 64u32);
    let renderer = HeadlessSceneRenderer::new(&gpu.device, w, h);

    // Camera at the chunk's XY centre, above the floor (small z),
    // looking straight down (+z). right × down == forward (RH).
    let cam = Camera {
        position: [vsid as f32 * 0.5, vsid as f32 * 0.5, 20.0],
        right: [1.0, 0.0, 0.0],
        down: [0.0, 1.0, 0.0],
        forward: [0.0, 0.0, 1.0],
        fov_y_rad: 30f32.to_radians(),
    };
    let fb = renderer.render(
        &gpu.device,
        &gpu.queue,
        &scene,
        &[cam],
        30f32.to_radians(),
        64,
    );
    assert_eq!(fb.len(), (w * h) as usize);

    // Centre pixel: the down-ray hits the z=100 floor voxel of the
    // centre column. Expected colour = (R,G,B) * (alpha/128) / 255
    // → (255,128,0) at brightness 1.0 → rgba8 ≈ (255,128,0).
    let centre = fb[(h / 2 * w + w / 2) as usize];
    let (r, g, b) = (centre & 0xff, (centre >> 8) & 0xff, (centre >> 16) & 0xff);
    eprintln!("scene_render: centre pixel = ({r}, {g}, {b})");
    assert!(r > 200, "floor R should be ~255, got {r}");
    assert!((100..=160).contains(&g), "floor G should be ~128, got {g}");
    assert!(b < 40, "floor B should be ~0, got {b}");

    // The floor fills the frame at this near-vertical view; assert a
    // solid majority of pixels are floor-coloured (not sky / clear),
    // proving the strided mip-0 lookup works across the whole image.
    let floor_px = fb
        .iter()
        .filter(|&&p| {
            let (r, g, b) = (p & 0xff, (p >> 8) & 0xff, (p >> 16) & 0xff);
            r > 200 && (90..=170).contains(&g) && b < 50
        })
        .count();
    let frac = floor_px as f32 / fb.len() as f32;
    eprintln!("scene_render: floor fraction = {frac:.3}");
    assert!(frac > 0.6, "expected floor to fill the view, got {frac:.3}");
}
