//! PF.10 gate — `cull_bin_upload` caching semantics on a live device:
//! same-key repeat calls return identical results (and skip the uploads),
//! a transform update invalidates the cache, and the `colmul` buffer
//! carries the lazy identity fill until a real table is set (then the
//! packed per-visible tables). Skips silently with no adapter.

#![allow(clippy::cast_possible_truncation)]

use roxlap_formats::kv6::Kv6;
use roxlap_formats::sprite::Sprite;
use roxlap_formats::VoxColor;
use roxlap_gpu::sprite_model::ViewFrustum;
use roxlap_gpu::{
    build_sprite_model, GpuRendererSettings, HeadlessGpu, SpriteInstance, SpriteInstanceTransform,
    SpriteModelRegistry, SpriteRegistryResident,
};

fn read_u32(gpu: &HeadlessGpu, buf: &wgpu::Buffer, words: u64) -> Vec<u32> {
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sprite_cull readback"),
        size: words * 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_buffer_to_buffer(buf, 0, &staging, 0, words * 4);
    gpu.queue.submit(std::iter::once(enc.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok();
    rx.recv().unwrap().unwrap();
    let data = slice.get_mapped_range();
    bytemuck::cast_slice(&data).to_vec()
}

fn axis_instance(model: u32, pos: [f32; 3]) -> SpriteInstance {
    let sprite = Sprite::axis_aligned(Kv6::solid_cube(8, VoxColor(0x80_ff_80_40)), pos);
    SpriteInstance::new(model, SpriteInstanceTransform::from_sprite(&sprite))
}

#[test]
fn cull_cache_and_identity_colmul() {
    let Ok(gpu) = HeadlessGpu::new_blocking(GpuRendererSettings::default()) else {
        eprintln!("[skip] no GPU adapter reachable");
        return;
    };

    let mut registry = SpriteModelRegistry::new();
    let model = registry.add(build_sprite_model(&Kv6::solid_cube(
        8,
        VoxColor(0x80_ff_80_40),
    )));
    // Two instances ahead of the camera, one far off to the side (culled).
    let instances = vec![
        axis_instance(model, [0.0, 30.0, 0.0]),
        axis_instance(model, [4.0, 50.0, 0.0]),
        axis_instance(model, [10_000.0, 30.0, 0.0]),
    ];
    let mut reg = SpriteRegistryResident::upload(&gpu.device, &registry, &instances);

    let f = ViewFrustum {
        pos: [0.0, 0.0, 0.0],
        right: [1.0, 0.0, 0.0],
        down: [0.0, 0.0, 1.0],
        forward: [0.0, 1.0, 0.0],
        half_w: 1.0,
        half_h: 0.6,
        far: 1.0e9,
    };
    let r1 = reg.cull_bin_upload(&gpu.device, &gpu.queue, &f, 640, 360, 16, 4.0);
    assert_eq!(r1.0, 2, "two of three instances are in view");

    // PF.10 — identity colmul fast path: the buffer must hold the
    // repeating identity pattern (0x0100 per lane) for every visible word.
    let lane = 0x0100u64;
    let w = lane | (lane << 16) | (lane << 32) | (lane << 48);
    let (lo, hi) = ((w & 0xffff_ffff) as u32, (w >> 32) as u32);
    let colmul = read_u32(&gpu, &reg.colmul, u64::from(r1.0) * 512);
    for (i, &v) in colmul.iter().enumerate() {
        assert_eq!(v, if i & 1 == 0 { lo } else { hi }, "identity word {i}");
    }

    // Same key ⇒ cached result, no re-cull.
    let r2 = reg.cull_bin_upload(&gpu.device, &gpu.queue, &f, 640, 360, 16, 4.0);
    assert_eq!(r1, r2, "same-key call must return the cached result");

    // A transform update must invalidate: move instance 1 out of view.
    let mut moved = instances.clone();
    moved[1] = axis_instance(model, [-10_000.0, 30.0, 0.0]);
    reg.update_transforms(&moved);
    let r3 = reg.cull_bin_upload(&gpu.device, &gpu.queue, &f, 640, 360, 16, 4.0);
    assert_eq!(r3.0, 1, "moved instance must drop out of the visible set");

    // A real (non-identity) table leaves the fast path: the packed
    // per-visible tables land in the buffer.
    let mut custom = [w; 256];
    custom[0] = 0x0042_0042_0042_0042;
    let tables: Vec<[u64; 256]> = vec![custom, [w; 256], [w; 256]];
    reg.set_instance_colmul(&tables);
    let r4 = reg.cull_bin_upload(&gpu.device, &gpu.queue, &f, 640, 360, 16, 4.0);
    assert_eq!(r4.0, 1);
    let packed = read_u32(&gpu, &reg.colmul, 2);
    assert_eq!(
        (packed[0], packed[1]),
        (0x0042_0042u32, 0x0042_0042u32),
        "custom table must be packed for the visible instance",
    );
}
