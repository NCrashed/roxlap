//! GPU.2 integration test — synthesise a `Vxl`, decompress it CPU-
//! side, upload to a headless GPU, then read back individual voxels
//! via the `debug_read.wgsl` shader and assert they match the CPU
//! `ChunkUpload::voxel_at`. Skips silently with a stderr note if no
//! Vulkan/Metal/DX12 adapter is reachable.

#![allow(clippy::cast_precision_loss)]

use std::sync::Mutex;
use std::time::Instant;

use roxlap_formats::vxl::Vxl;
use roxlap_gpu::{
    decompress_chunk, GpuChunkResident, GpuInitError, GpuRendererSettings, HeadlessGpu, CHUNK_Z,
};

/// Serialise the GPU-touching tests. `cargo test` runs test fns on
/// separate threads by default; spinning up 4 independent wgpu
/// devices + concurrent `map_async` readback on the same physical
/// GPU is racy on some drivers (observed: Mesa NVK returns stale
/// readback bytes under concurrent device load). The upload /
/// readback logic itself is correct — it passes 100% serially —
/// so a process-wide lock around each test's device work is the
/// right fix, not a logic change.
static GPU_TEST_LOCK: Mutex<()> = Mutex::new(());

/// Build a `vsid × vsid` Vxl where every column has one textured
/// floor voxel at z=100 with colour `0x80ff_8000` (red-orange).
/// Identical in shape to `decompress::tests::fixture_one_voxel_per_column`.
fn fixture_one_voxel_per_column(vsid: u32) -> Vxl {
    let n_cols = (vsid as usize) * (vsid as usize);
    let mut data: Vec<u8> = Vec::with_capacity(n_cols * 8);
    let mut column_offset: Vec<u32> = Vec::with_capacity(n_cols + 1);
    // BGRA little-endian bytes of 0x80ff_8000 — alpha 0x80 keeps
    // the fixture visible under the alpha-zero placeholder filter.
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

/// Try to bring up a headless device, holding [`GPU_TEST_LOCK`] for
/// the duration so no two GPU tests touch the device concurrently.
/// Skips the calling test (returns `None`) when no adapter is
/// present. The returned guard must be kept alive for the whole
/// test — bind it to a named local (`let _g = ...`) at the call
/// site.
fn try_init() -> Option<(HeadlessGpu, std::sync::MutexGuard<'static, ()>)> {
    // Recover from a poisoned lock (a panicking test still releases
    // the device); the data is `()` so there's nothing to corrupt.
    let guard = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    match HeadlessGpu::new_blocking(GpuRendererSettings::default()) {
        Ok(gpu) => Some((gpu, guard)),
        Err(GpuInitError::NoAdapter) => {
            eprintln!("[skip] no GPU adapter reachable — set up Vulkan/Metal/DX12 to run");
            None
        }
        Err(e) => {
            eprintln!("[skip] GPU init failed ({e}) — driver issue");
            None
        }
    }
}

#[test]
fn round_trip_textured_voxel_matches_cpu() {
    let Some((gpu, _gpu_lock)) = try_init() else {
        return;
    };
    eprintln!("round_trip: adapter = {}", gpu.adapter_info);

    let vxl = fixture_one_voxel_per_column(4);
    let chunk = decompress_chunk(&vxl);
    let resident = GpuChunkResident::upload(&gpu.device, &chunk);
    eprintln!("resident bytes: {}", resident.resident_bytes());

    // Textured voxels — every (x, y, 100) should read 0x80ff_8000.
    for y in 0..vxl.vsid {
        for x in 0..vxl.vsid {
            let v = resident.read_voxel_blocking(&gpu.device, &gpu.queue, x, y, 100);
            assert_eq!(
                v,
                Some(0x80ff_8000),
                "GPU voxel at ({x}, {y}, 100) should be 0x80ff_8000"
            );
            // CPU mirror.
            assert_eq!(
                chunk.voxel_at(x, y, 100),
                v,
                "CPU/GPU disagree at ({x}, {y}, 100)"
            );
        }
    }
}

#[test]
fn round_trip_air_above_returns_none() {
    let Some((gpu, _gpu_lock)) = try_init() else {
        return;
    };
    let vxl = fixture_one_voxel_per_column(4);
    let chunk = decompress_chunk(&vxl);
    let resident = GpuChunkResident::upload(&gpu.device, &chunk);

    for &z in &[0u32, 1, 50, 99] {
        let v = resident.read_voxel_blocking(&gpu.device, &gpu.queue, 1, 2, z);
        assert!(
            v.is_none(),
            "GPU should report empty at (1, 2, {z}) — got {v:?}"
        );
        assert_eq!(chunk.voxel_at(1, 2, z), v);
    }
}

#[test]
fn round_trip_bedrock_below_now_returns_air() {
    // Bedrock-as-air refactor (GPU.4 prereq) — z > z1c is no
    // longer reported as solid by the GPU decompressor or shader.
    let Some((gpu, _gpu_lock)) = try_init() else {
        return;
    };
    let vxl = fixture_one_voxel_per_column(4);
    let chunk = decompress_chunk(&vxl);
    let resident = GpuChunkResident::upload(&gpu.device, &chunk);

    for &z in &[101u32, 150, CHUNK_Z - 1] {
        let v = resident.read_voxel_blocking(&gpu.device, &gpu.queue, 1, 2, z);
        assert!(
            v.is_none(),
            "bedrock at (1, 2, {z}) should be empty — got {v:?}"
        );
        assert_eq!(chunk.voxel_at(1, 2, z), v);
    }
}

#[test]
fn bench_single_chunk_upload() {
    let Some((gpu, _gpu_lock)) = try_init() else {
        return;
    };

    // Scene-demo chunk size — vsid=128 matches roxlap-scene's
    // `CHUNK_SIZE_XY`. One textured voxel + bedrock per column =
    // ~156 colours × 128² = ~10 MiB on the GPU. Real chunks are
    // denser (more textured voxels) but this single-shape upload is
    // a representative "biggest chunk we'd ever upload" lower bound.
    let vxl = fixture_one_voxel_per_column(128);
    let chunk = decompress_chunk(&vxl);

    let t0 = Instant::now();
    let resident = GpuChunkResident::upload(&gpu.device, &chunk);
    // `create_buffer_init` doesn't block on transfer completion;
    // poll to make the timing reflect the full upload.
    gpu.device.poll(wgpu::Maintain::Wait);
    let upload_dt = t0.elapsed();

    let bytes = resident.resident_bytes();
    eprintln!(
        "bench: vsid=128 upload {:.1} KiB in {:.3?} → {:.1} MiB/s",
        bytes as f64 / 1024.0,
        upload_dt,
        bytes as f64 / (1024.0 * 1024.0) / upload_dt.as_secs_f64(),
    );

    // Sanity check — the GPU resident sees the same first voxel.
    let v = resident.read_voxel_blocking(&gpu.device, &gpu.queue, 7, 13, 100);
    assert_eq!(v, Some(0x80ff_8000));
}
