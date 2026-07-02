//! Perf probe: time the headless scene march at demo resolution on the
//! live adapter — no swapchain, no egui, no present. Discriminates "the
//! marcher / driver is slow" from "the surface/present path is slow"
//! (e.g. the 2026-07 nixos mesa update that deadlocked nouveau↔i915
//! PRIME explicit-sync fences: this probe showed a healthy ~10 ms march
//! while windowed frames took ~2.5 s and the drm scheduler killed the
//! channel). Prints only — never asserts a rate (CI adapters vary). Run:
//! `cargo test -p roxlap-gpu --release --test perf_probe -- --nocapture`

#![allow(clippy::cast_precision_loss)]

use roxlap_formats::vxl::Vxl;
use roxlap_gpu::{
    decompress_chunk, Camera, GpuRendererSettings, GpuSceneResident, GridUpload, HeadlessGpu,
    HeadlessSceneRenderer, SceneUpload,
};

/// One textured floor voxel per column at z = 100 (same as scene_render.rs).
fn floor_chunk(vsid: u32) -> Vxl {
    let n_cols = (vsid as usize) * (vsid as usize);
    let mut data: Vec<u8> = Vec::with_capacity(n_cols * 8);
    let mut column_offset: Vec<u32> = Vec::with_capacity(n_cols + 1);
    let bgra = [0x00u8, 0x80, 0xff, 0x80];
    for _ in 0..n_cols {
        column_offset.push(u32::try_from(data.len()).expect("offset fits"));
        data.extend_from_slice(&[0, 100, 100, 0]);
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
fn headless_march_timing() {
    let Ok(gpu) = HeadlessGpu::new_blocking(GpuRendererSettings::default()) else {
        eprintln!("no adapter — skipping");
        return;
    };
    eprintln!("perf_probe: adapter = {}", gpu.adapter_info);

    let vsid = 128u32;
    let mut chunks = Vec::new();
    for cy in 0..2i32 {
        for cx in 0..2i32 {
            chunks.push(([cx, cy, 0], decompress_chunk(&floor_chunk(vsid))));
        }
    }
    let grid = GridUpload {
        vsid,
        origin_chunk: [0, 0, 0],
        chunks_dims: [2, 2, 1],
        pool_dims: [2, 2, 1],
        chunks,
    };
    let scene = GpuSceneResident::upload(&gpu.device, &SceneUpload { grids: vec![grid] });

    let (w, h) = (860u32, 520u32);
    let renderer = HeadlessSceneRenderer::new(&gpu.device, &gpu.queue, w, h);
    let cam = Camera {
        position: [128.0, 128.0, 60.0],
        right: [1.0, 0.0, 0.0],
        down: [0.0, 0.0, 1.0],
        forward: [0.0, 1.0, 0.0],
        fov_y_rad: 60f32.to_radians(),
    };
    // Warm-up (pipeline compile) then a timed loop.
    let _ = renderer.render(
        &gpu.device,
        &gpu.queue,
        &scene,
        &[cam],
        cam.fov_y_rad,
        64,
        64.0,
    );
    let n = 20u32;
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        let _ = renderer.render(
            &gpu.device,
            &gpu.queue,
            &scene,
            &[cam],
            cam.fov_y_rad,
            64,
            64.0,
        );
    }
    let dt = t0.elapsed();
    eprintln!(
        "perf_probe: {n} x 860x520 marches in {:.1} ms ({:.2} ms/frame)",
        dt.as_secs_f64() * 1e3,
        dt.as_secs_f64() * 1e3 / f64::from(n),
    );
}
