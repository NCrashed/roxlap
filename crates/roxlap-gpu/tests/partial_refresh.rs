//! PF.12.c gate — `refresh_chunk_partial` on a live device: a
//! count-stable recolour partially uploaded must leave the resident's
//! occupancy + colours byte-identical to a full `refresh_chunk` of the
//! same edit, and a count-CHANGING carve must be refused (nothing
//! written). Skips silently with no adapter.

#![allow(clippy::cast_possible_truncation)]

use roxlap_formats::edit::{set_spans, Vspan};
use roxlap_formats::vxl::Vxl;
use roxlap_formats::VoxColor;
use roxlap_gpu::{
    decompress_chunk, GpuRendererSettings, GpuSceneResident, GridUpload, HeadlessGpu, SceneUpload,
};

/// One textured floor voxel per column at z = 100 (scene_render.rs's
/// shape) with an edit pool + full mip ladder.
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
    let mut vxl = Vxl {
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
    };
    vxl.reserve_edit_capacity(n_cols * 64);
    vxl.generate_mips(6);
    vxl
}

fn upload_scene(gpu: &HeadlessGpu, vxl: &Vxl) -> GpuSceneResident {
    let grid = GridUpload {
        vsid: vxl.vsid,
        origin_chunk: [0, 0, 0],
        chunks_dims: [1, 1, 1],
        pool_dims: [1, 1, 1],
        chunks: vec![([0, 0, 0], decompress_chunk(vxl))],
    };
    GpuSceneResident::upload(&gpu.device, &SceneUpload { grids: vec![grid] })
}

fn read_u32(gpu: &HeadlessGpu, buf: &wgpu::Buffer, words: u64) -> Vec<u32> {
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("partial_refresh readback"),
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

fn snapshot(gpu: &HeadlessGpu, r: &GpuSceneResident) -> Vec<Vec<u32>> {
    [
        (&r.occupancy_pages[0], r.occupancy_page_words as u64),
        (&r.all_color_offsets, 0),
        (&r.all_colors, 0),
    ]
    .iter()
    .map(|(buf, words)| {
        let w = if *words > 0 { *words } else { buf.size() / 4 };
        read_u32(gpu, buf, w)
    })
    .collect()
}

#[test]
fn partial_matches_full_and_refuses_reflow() {
    let Ok(gpu) = HeadlessGpu::new_blocking(GpuRendererSettings::default()) else {
        eprintln!("[skip] no GPU adapter reachable");
        return;
    };

    let mut vxl = floor_chunk(64);
    let mut full_res = upload_scene(&gpu, &vxl);
    let mut part_res = upload_scene(&gpu, &vxl);

    // Count-stable recolour: overwrite the floor voxel of a 6×5 patch
    // with a new colour (same single-voxel column shape), then re-mip.
    let mut spans = Vec::new();
    for y in 20..25u32 {
        for x in 30..36u32 {
            spans.push(Vspan {
                x,
                y,
                z0: 100,
                z1: 100,
            });
        }
    }
    set_spans(&mut vxl, &spans, Some(VoxColor(0x80_12_34_56)));
    vxl.remip_bbox(30, 20, 35, 24, 6);

    // Full path on resident A; partial on resident B.
    let up = decompress_chunk(&vxl);
    let outcome = full_res.refresh_chunk(&gpu.queue, 0, [0, 0, 0], &up);
    assert_eq!(outcome, roxlap_gpu::RefreshOutcome::Ok);
    assert!(
        part_res.refresh_chunk_partial(&gpu.queue, 0, [0, 0, 0], &vxl, 29, 19, 36, 25),
        "count-stable recolour must take the partial path",
    );

    let a = snapshot(&gpu, &full_res);
    let b = snapshot(&gpu, &part_res);
    assert_eq!(a[0], b[0], "occupancy pages diverge");
    assert_eq!(a[1], b[1], "color_offsets diverge");
    assert_eq!(a[2], b[2], "colors diverge");

    // Count-changing carve (removes voxels) must be refused untouched.
    let before = snapshot(&gpu, &part_res);
    let carve = [Vspan {
        x: 10,
        y: 10,
        z0: 100,
        z1: 100,
    }];
    set_spans(&mut vxl, &carve, None);
    vxl.remip_bbox(10, 10, 10, 10, 6);
    assert!(
        !part_res.refresh_chunk_partial(&gpu.queue, 0, [0, 0, 0], &vxl, 9, 9, 11, 11),
        "count change must refuse the partial path",
    );
    let after = snapshot(&gpu, &part_res);
    assert_eq!(before, after, "refused partial must write nothing");
}
