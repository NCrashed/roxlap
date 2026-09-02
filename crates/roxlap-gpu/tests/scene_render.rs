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
    decompress_chunk, Camera, GpuInitError, GpuLight, GpuRendererSettings, GpuSceneResident,
    GpuViewCutout, GridUpload, GridWorldTransform, HeadlessGpu, HeadlessSceneRenderer, SceneLights,
    SceneUpload,
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

/// `vsid × vsid` chunk: every column solid over `z ∈ [top, bot]`
/// (a wall/block facing a horizontal ray), colour `0x80ff_8000`.
fn block_chunk(vsid: u32, top: u8, bot: u8) -> Vxl {
    let n_cols = (vsid as usize) * (vsid as usize);
    let n_vox = (bot - top + 1) as usize;
    let mut data: Vec<u8> = Vec::with_capacity(n_cols * (4 + n_vox * 4));
    let mut column_offset: Vec<u32> = Vec::with_capacity(n_cols + 1);
    let bgra = [0x00u8, 0x80, 0xff, 0x80];
    for _ in 0..n_cols {
        column_offset.push(u32::try_from(data.len()).expect("offset fits"));
        data.extend_from_slice(&[0, top, bot, 0]); // nextptr=0, z1=top, z1c=bot, z0=0
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

/// [`block_chunk`] with an explicit BGRA colour (SC.4 — two colours so a
/// two-grid composite test can tell which grid won the min-t depth test).
fn block_chunk_bgra(vsid: u32, top: u8, bot: u8, bgra: [u8; 4]) -> Vxl {
    let n_cols = (vsid as usize) * (vsid as usize);
    let n_vox = (bot - top + 1) as usize;
    let mut data: Vec<u8> = Vec::with_capacity(n_cols * (4 + n_vox * 4));
    let mut column_offset: Vec<u32> = Vec::with_capacity(n_cols + 1);
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

/// Recognisably the orange block (`0x80ff_8000` at brightness 1.0 →
/// ~(255,128,0)), not the bluish sky (~(120,150,220)). Loose because
/// coarse mips average the (uniform) block colour.
fn is_block_color(p: u32) -> bool {
    let (r, g, b) = (p & 0xff, (p >> 8) & 0xff, (p >> 16) & 0xff);
    r > 180 && (80..=175).contains(&g) && b < 70
}

/// DL.3 — floor at z=100 in every column, plus a short wall standing on it
/// at `x ∈ [wx0, wx1)` (all y), solid `z ∈ [wtop, 100]` (rising `100-wtop`
/// voxels above the floor). Used to cast a sun shadow onto the floor next to
/// the wall. Colour `0x80ff_8000`.
fn floor_with_wall_chunk(vsid: u32, wx0: u32, wx1: u32, wtop: u8) -> Vxl {
    let n_cols = (vsid as usize) * (vsid as usize);
    let mut data: Vec<u8> = Vec::new();
    let mut column_offset: Vec<u32> = Vec::with_capacity(n_cols + 1);
    let bgra = [0x00u8, 0x80, 0xff, 0x80];
    for i in 0..n_cols {
        let x = (i as u32) % vsid;
        column_offset.push(u32::try_from(data.len()).expect("offset fits"));
        if x >= wx0 && x < wx1 {
            let n_vox = (100 - wtop + 1) as usize;
            data.extend_from_slice(&[0, wtop, 100, 0]); // z1=wtop..z1c=100
            for _ in 0..n_vox {
                data.extend_from_slice(&bgra);
            }
        } else {
            data.extend_from_slice(&[0, 100, 100, 0]); // floor voxel at z=100
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

/// `vsid × vsid` chunk: one textured floor voxel per column at `z =
/// surf`, with implicit voxlap **bedrock** solid below it to z=255.
/// Models a cliff/wall: only the top is coloured; the face below is
/// bedrock. Pre-fix the GPU treated bedrock as air → the face showed
/// sky. Slab `[nextptr=0, z1=surf, z1c=surf, z0=0]` + 1 colour.
fn wall_chunk(vsid: u32, surf: u8) -> Vxl {
    let n_cols = (vsid as usize) * (vsid as usize);
    let mut data: Vec<u8> = Vec::with_capacity(n_cols * 8);
    let mut column_offset: Vec<u32> = Vec::with_capacity(n_cols + 1);
    let bgra = [0x00u8, 0x80, 0xff, 0x80];
    for _ in 0..n_cols {
        column_offset.push(u32::try_from(data.len()).expect("offset fits"));
        data.extend_from_slice(&[0, surf, surf, 0]);
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
fn scene_dda_marches_coarse_mip_for_distant_chunk() {
    let Some((gpu, _lock)) = try_init() else {
        return;
    };
    eprintln!("mip_render: adapter = {}", gpu.adapter_info);

    // One solid block chunk placed FAR along +y (chunk index 4) so a
    // horizontal ray enters it at t ≈ 128 — past several octaves of
    // mip_scan_dist, forcing a deep mip. The camera sits in the empty
    // chunk (0,0,0).
    let vsid = 32u32;
    let chunk = decompress_chunk(&block_chunk(vsid, 0, 31));
    assert!(chunk.mips.len() >= 5, "need a deep ladder for mip-4");

    let grid = GridUpload {
        vsid,
        origin_chunk: [0, 0, 0],
        chunks_dims: [1, 8, 1],
        pool_dims: [1, 8, 1],
        chunks: vec![([0, 4, 0], chunk)],
    };
    let scene = GpuSceneResident::upload(&gpu.device, &SceneUpload { grids: vec![grid] });

    let (w, h) = (64u32, 64u32);
    let renderer = HeadlessSceneRenderer::new(&gpu.device, &gpu.queue, w, h);
    // Camera in the empty near chunk, looking +y at the block; z=16
    // lands inside the block's z=0..31 band. right × down == forward.
    let cam = Camera {
        position: [vsid as f32 * 0.5, 0.0, 16.0],
        right: [1.0, 0.0, 0.0],
        down: [0.0, 0.0, 1.0],
        forward: [0.0, 1.0, 0.0],
        fov_y_rad: 30f32.to_radians(),
    };
    let centre = (h / 2 * w + w / 2) as usize;

    // mip-0 baseline (LOD off): the block renders.
    let fb0 = renderer.render(
        &gpu.device,
        &gpu.queue,
        &scene,
        &[cam],
        cam.fov_y_rad,
        64,
        0.0,
    );
    assert!(
        is_block_color(fb0[centre]),
        "mip-0 centre should be the block, got {:#08x}",
        fb0[centre],
    );

    // Force a deep mip: msd=8 at t≈128 → mip-4. If mip-N occupancy /
    // colour addressing were wrong the block would vanish (sky) or
    // render a garbage colour.
    let fb4 = renderer.render(
        &gpu.device,
        &gpu.queue,
        &scene,
        &[cam],
        cam.fov_y_rad,
        64,
        8.0,
    );
    eprintln!(
        "mip_render: centre mip0={:#08x} mip4={:#08x}",
        fb0[centre], fb4[centre]
    );
    assert!(
        is_block_color(fb4[centre]),
        "coarse-mip centre should still be the block, got {:#08x}",
        fb4[centre],
    );

    // The coarse render should broadly agree with mip-0 (same block
    // fills the view) — most pixels classify the same way.
    let agree = fb0
        .iter()
        .zip(&fb4)
        .filter(|(a, b)| is_block_color(**a) == is_block_color(**b))
        .count();
    let frac = agree as f32 / fb0.len() as f32;
    eprintln!("mip_render: block/sky agreement = {frac:.3}");
    assert!(
        frac > 0.9,
        "mip-0 vs mip-4 block coverage diverged: {frac:.3}"
    );
}

#[test]
fn scene_dda_aabb_early_out_away_is_sky() {
    // GPU.13.0 — the occupied chunk-AABB early-out must terminate a
    // ray the moment it has left the box along its travel direction.
    // One block chunk at +y (chunk 4); the camera sits in the near
    // chunk (0,0,0) but looks the OTHER way (−y). Every ray starts
    // already past the AABB's near slab (p.y=0 < aabb_min.y=4 with
    // step.y<0) → instant early-out → pure sky, no block pixels.
    let Some((gpu, _lock)) = try_init() else {
        return;
    };
    let vsid = 32u32;
    let chunk = decompress_chunk(&block_chunk(vsid, 0, 31));
    let grid = GridUpload {
        vsid,
        origin_chunk: [0, 0, 0],
        chunks_dims: [1, 8, 1],
        pool_dims: [1, 8, 1],
        chunks: vec![([0, 4, 0], chunk)],
    };
    let scene = GpuSceneResident::upload(&gpu.device, &SceneUpload { grids: vec![grid] });
    // Sanity: the upload computed the occupied AABB at chunk y=4.
    assert_eq!(scene.static_meta[0].aabb_min, [0, 4, 0]);
    assert_eq!(scene.static_meta[0].aabb_max, [0, 4, 0]);

    let (w, h) = (64u32, 64u32);
    let renderer = HeadlessSceneRenderer::new(&gpu.device, &gpu.queue, w, h);
    let cam = Camera {
        position: [vsid as f32 * 0.5, 0.0, 16.0],
        right: [-1.0, 0.0, 0.0],
        down: [0.0, 0.0, 1.0],
        forward: [0.0, -1.0, 0.0], // look AWAY from the block
        fov_y_rad: 30f32.to_radians(),
    };
    let fb = renderer.render(
        &gpu.device,
        &gpu.queue,
        &scene,
        &[cam],
        cam.fov_y_rad,
        64,
        0.0,
    );
    let block_px = fb.iter().filter(|&&p| is_block_color(p)).count();
    assert_eq!(
        block_px, 0,
        "looking away from the only chunk must be all sky, got {block_px} block pixels",
    );
}

#[test]
fn scene_dda_zero_grids_renders_sky() {
    // Regression for the sprite-only / empty-scene GPU path: a scene
    // with ZERO grids must still render a valid frame — the scene pass
    // fills the flat sky everywhere (+ far depth), giving the sprite
    // pass a background to composite over. Pre-fix the render facade
    // short-circuited a grid-less scene to a bare clear and never ran
    // `render_scene`, so a sprite-only viewer (no voxel grids) showed
    // only the clear colour with the model invisible. This exercises
    // the engine half the facade fix newly relies on: grid_count == 0
    // with zero cameras renders the uniform sky without panicking.
    let Some((gpu, _lock)) = try_init() else {
        return;
    };
    let scene = GpuSceneResident::upload(&gpu.device, &SceneUpload { grids: vec![] });
    assert_eq!(
        scene.grid_count, 0,
        "empty SceneUpload → zero-grid resident"
    );

    let (w, h) = (32u32, 32u32);
    let renderer = HeadlessSceneRenderer::new(&gpu.device, &gpu.queue, w, h);
    // Zero cameras matches grid_count == 0 (render_scene asserts equal).
    // Pre-fix this path was never dispatched (the facade short-circuited
    // to a clear); the win here is it runs end-to-end without panic and
    // produces a clean, consistent background for the sprite pass.
    let fb = renderer.render(
        &gpu.device,
        &gpu.queue,
        &scene,
        &[],
        30f32.to_radians(),
        64,
        0.0,
    );
    assert_eq!(fb.len(), (w * h) as usize);

    // No grids → no per-pixel grid hits → every pixel is the flat sky
    // sample (headless default sky [120, 150, 220]). Uniform proves the
    // zero-grid path reads no garbage grid data; bluish (b > r) + not
    // black proves the sky direction is well-formed — the pre-fix
    // (0,0,1) default fed atan2(0,0) → NaN → a black sample.
    let first = fb[0];
    assert!(
        fb.iter().all(|&p| p == first),
        "zero-grid frame must be a uniform sky, got varied pixels (first={first:#08x})",
    );
    assert_ne!(
        first & 0x00ff_ffff,
        0,
        "sky must not be black, got {first:#08x}"
    );
    let (r, _g, b) = (first & 0xff, (first >> 8) & 0xff, (first >> 16) & 0xff);
    assert!(b > r, "headless sky is bluish (b>r), got {first:#08x}");
}

/// GPU side-shade (voxlap setsideshades) darkens a grid face. Camera
/// looks straight down at a floor plane; the floor is hit via a +z
/// step, so its shade comes from the `bot` lane. Rendering with
/// `bot = 64` must darken the floor ~half vs the unshaded baseline,
/// proving the scene-DDA face detection + brightness reduction work.
/// (Exact CPU parity needs visual inspection; this guards the
/// mechanism + uniform plumbing against regressions.)
#[test]
fn scene_dda_side_shades_darken_floor() {
    let Some((gpu, _lock)) = try_init() else {
        return;
    };
    let vsid = 32u32;
    let chunk = decompress_chunk(&floor_chunk(vsid)); // floor voxel at z=100
    let grid = GridUpload {
        vsid,
        origin_chunk: [0, 0, 0],
        chunks_dims: [1, 1, 1],
        pool_dims: [1, 1, 1],
        chunks: vec![([0, 0, 0], chunk)],
    };
    let scene = GpuSceneResident::upload(&gpu.device, &SceneUpload { grids: vec![grid] });

    let (w, h) = (64u32, 64u32);
    let mut renderer = HeadlessSceneRenderer::new(&gpu.device, &gpu.queue, w, h);
    // Above the floor (z=50 < 100), looking straight down (+z, voxlap
    // z-down). right × down == forward.
    let cam = Camera {
        position: [16.0, 16.0, 50.0],
        right: [1.0, 0.0, 0.0],
        down: [0.0, 1.0, 0.0],
        forward: [0.0, 0.0, 1.0],
        fov_y_rad: 60f32.to_radians(),
    };
    let centre = (h / 2 * w + w / 2) as usize;
    let lum = |p: u32| (p & 0xff) + ((p >> 8) & 0xff) + ((p >> 16) & 0xff);

    let fb0 = renderer.render(
        &gpu.device,
        &gpu.queue,
        &scene,
        &[cam],
        cam.fov_y_rad,
        64,
        0.0,
    );
    let base = fb0[centre];
    assert!(
        is_block_color(base),
        "centre should be the lit floor, got {base:#08x}",
    );

    // Darken the floor face (bot lane = side_shades[1]).
    renderer.set_side_shades([0, 64, 0, 0, 0, 0]);
    let fb1 = renderer.render(
        &gpu.device,
        &gpu.queue,
        &scene,
        &[cam],
        cam.fov_y_rad,
        64,
        0.0,
    );
    let shaded = fb1[centre];
    assert!(
        lum(shaded) < lum(base),
        "side-shade should darken the floor: {base:#08x} -> {shaded:#08x}",
    );
    // bot=64 of 128 → ~half brightness, still textured (not black sky).
    assert_ne!(
        shaded & 0x00ff_ffff,
        0,
        "half-shaded floor must not be black"
    );
}

/// FW.3 — a `fog_mask` for one 1-deck grid (slot 0, band z∈[0,255]) over
/// cells `[0, vsid)²` at origin `(0, 0)`, every cell set to `state` (2 =
/// Visible, 1 = Memory, 0 = Unseen) at `intensity` (0..=63), styled by
/// `memory_dim`. Built via the SHARED `roxlap_gpu::fow::pack_fog_mask`
/// (one header owner — the same packer the production path uses).
fn fog_mask_uniform(vsid: u32, state: u8, intensity: u8, memory_dim: f32) -> Vec<u32> {
    let byte = (state << 6) | (intensity & 63);
    let cells = vec![byte; (vsid * vsid) as usize];
    roxlap_gpu::fow::pack_fog_mask(
        0,
        [0, 0],
        vsid,
        vsid,
        1,
        &[[0, 255]],
        0,
        memory_dim,
        0.0,
        false,
        &cells,
    )
}

/// FW.3 — the GPU fog mask hides Unseen cells (renders sky) and shows
/// Visible cells (renders the floor, dim=1), mirroring the CPU verdict.
#[test]
fn scene_dda_fog_mask_hides_unseen_shows_visible() {
    let Some((gpu, _lock)) = try_init() else {
        return;
    };
    let vsid = 32u32;
    let chunk = decompress_chunk(&floor_chunk(vsid)); // floor voxel at z=100
    let grid = GridUpload {
        vsid,
        origin_chunk: [0, 0, 0],
        chunks_dims: [1, 1, 1],
        pool_dims: [1, 1, 1],
        chunks: vec![([0, 0, 0], chunk)],
    };
    let scene = GpuSceneResident::upload(&gpu.device, &SceneUpload { grids: vec![grid] });
    let (w, h) = (64u32, 64u32);
    let mut renderer = HeadlessSceneRenderer::new(&gpu.device, &gpu.queue, w, h);
    let cam = Camera {
        position: [16.0, 16.0, 50.0],
        right: [1.0, 0.0, 0.0],
        down: [0.0, 1.0, 0.0],
        forward: [0.0, 0.0, 1.0],
        fov_y_rad: 60f32.to_radians(),
    };
    let centre = (h / 2 * w + w / 2) as usize;
    let render = |r: &mut HeadlessSceneRenderer| {
        r.render(
            &gpu.device,
            &gpu.queue,
            &scene,
            &[cam],
            cam.fov_y_rad,
            64,
            0.0,
        )[centre]
    };

    // No fog → the floor draws.
    let base = render(&mut renderer);
    assert!(is_block_color(base), "baseline floor, got {base:#08x}");

    // All-Visible fog at full intensity → still the floor (dim=1 identity).
    renderer.set_fog_mask(&fog_mask_uniform(vsid, 2, 63, 1.0));
    let visible = render(&mut renderer);
    assert!(is_block_color(visible), "visible floor, got {visible:#08x}");

    // All-Unseen fog → the floor is Hidden (treated as air) → sky.
    renderer.set_fog_mask(&fog_mask_uniform(vsid, 0, 0, 1.0));
    let unseen = render(&mut renderer);
    assert!(
        !is_block_color(unseen),
        "unseen cell must render as sky, got {unseen:#08x}"
    );

    // Clearing the mask restores the floor (byte-identical to baseline).
    renderer.set_fog_mask(&[]);
    let cleared = render(&mut renderer);
    assert_eq!(cleared, base, "disabled fog is byte-identical");
}

/// A mask CELL may be wider than a grid column (`VisionConfig::
/// cell_span`): the shader shifts a voxel's XY down by the span's `log2`
/// before indexing. Two things have to hold, and a mask that merely fits
/// would pass the first alone — so this checks both: a span-4 mask a
/// quarter the width still covers the whole floor (the shift happens at
/// all), and blanking ONE cell of it blanks exactly its own block, with
/// the neighbouring block still drawn (the index lands where it should).
#[test]
fn scene_dda_fog_mask_spans_blocks_of_columns() {
    let Some((gpu, _lock)) = try_init() else {
        return;
    };
    let vsid = 32u32;
    let chunk = decompress_chunk(&floor_chunk(vsid)); // floor voxel at z=100
    let grid = GridUpload {
        vsid,
        origin_chunk: [0, 0, 0],
        chunks_dims: [1, 1, 1],
        pool_dims: [1, 1, 1],
        chunks: vec![([0, 0, 0], chunk)],
    };
    let scene = GpuSceneResident::upload(&gpu.device, &SceneUpload { grids: vec![grid] });
    let (w, h) = (64u32, 64u32);
    let mut renderer = HeadlessSceneRenderer::new(&gpu.device, &gpu.queue, w, h);
    let cam = Camera {
        position: [16.0, 16.0, 50.0],
        right: [1.0, 0.0, 0.0],
        down: [0.0, 1.0, 0.0],
        forward: [0.0, 0.0, 1.0],
        fov_y_rad: 60f32.to_radians(),
    };
    let px = |dx: u32| ((h / 2) * w + w / 2 + dx) as usize;
    let render = |r: &mut HeadlessSceneRenderer| {
        r.render(
            &gpu.device,
            &gpu.queue,
            &scene,
            &[cam],
            cam.fov_y_rad,
            64,
            0.0,
        )
    };

    // A span-4 mask over the same ground: 8x8 cells, not 32x32 columns.
    let span = 4u32;
    let cells_wide = vsid / span;
    let mask = |cells: &[u8]| {
        roxlap_gpu::fow::pack_fog_mask(
            0,
            [0, 0],
            cells_wide,
            cells_wide,
            span,
            &[[0, 255]],
            0,
            1.0,
            0.0,
            false,
            cells,
        )
    };

    let visible = vec![2u8 << 6 | 63; (cells_wide * cells_wide) as usize];
    renderer.set_fog_mask(&mask(&visible));
    let all = render(&mut renderer);
    assert!(
        is_block_color(all[px(0)]),
        "a quarter-width mask must still cover the floor — the shader \
         did not shift, got {:#08x}",
        all[px(0)],
    );

    // The camera sits over column (16, 16) — cell (4, 4) at this span —
    // and eight pixels right is about seven columns, so it lands in the
    // NEXT cell along. Blank the centre cell only.
    let mut one_out = visible.clone();
    one_out[(4 * cells_wide + 4) as usize] = 0;
    renderer.set_fog_mask(&mask(&one_out));
    let holed = render(&mut renderer);
    assert!(
        !is_block_color(holed[px(0)]),
        "the blanked block must read as sky, got {:#08x}",
        holed[px(0)],
    );
    assert!(
        is_block_color(holed[px(8)]),
        "…and only that block: its neighbour is still floor, got {:#08x}",
        holed[px(8)],
    );
}

/// FW.3 — a Memory cell at low intensity renders dimmer than the live
/// floor (the `memory_dim` taper), mirroring the CPU `fow_style`.
#[test]
fn scene_dda_fog_memory_dims_floor() {
    let Some((gpu, _lock)) = try_init() else {
        return;
    };
    let vsid = 32u32;
    let chunk = decompress_chunk(&floor_chunk(vsid));
    let grid = GridUpload {
        vsid,
        origin_chunk: [0, 0, 0],
        chunks_dims: [1, 1, 1],
        pool_dims: [1, 1, 1],
        chunks: vec![([0, 0, 0], chunk)],
    };
    let scene = GpuSceneResident::upload(&gpu.device, &SceneUpload { grids: vec![grid] });
    let (w, h) = (64u32, 64u32);
    let mut renderer = HeadlessSceneRenderer::new(&gpu.device, &gpu.queue, w, h);
    let cam = Camera {
        position: [16.0, 16.0, 50.0],
        right: [1.0, 0.0, 0.0],
        down: [0.0, 1.0, 0.0],
        forward: [0.0, 0.0, 1.0],
        fov_y_rad: 60f32.to_radians(),
    };
    let centre = (h / 2 * w + w / 2) as usize;
    let lum = |p: u32| (p & 0xff) + ((p >> 8) & 0xff) + ((p >> 16) & 0xff);
    let render = |r: &mut HeadlessSceneRenderer| {
        r.render(
            &gpu.device,
            &gpu.queue,
            &scene,
            &[cam],
            cam.fov_y_rad,
            64,
            0.0,
        )[centre]
    };

    // Visible at full intensity (dim 1).
    renderer.set_fog_mask(&fog_mask_uniform(vsid, 2, 63, 0.4));
    let visible = render(&mut renderer);
    // Memory at intensity 0 → dim = memory_dim = 0.4.
    renderer.set_fog_mask(&fog_mask_uniform(vsid, 1, 0, 0.4));
    let memory = render(&mut renderer);
    assert!(is_block_color(visible), "visible floor, got {visible:#08x}");
    assert!(
        lum(memory) < lum(visible),
        "memory floor must be dimmer: {visible:#08x} -> {memory:#08x}"
    );
    assert_ne!(memory & 0x00ff_ffff, 0, "dimmed memory is not black");
}

/// DL.1 — the directional sun (N·L diffuse) lights a grid face by its
/// facing. Camera looks straight down a floor (hit via +z step ⇒ top-face
/// normal = -z = up). A sun coming from above (to-sun = up = -z) gives
/// N·L = 1 → the floor is brighter than the baked-only baseline; a sun
/// from below (to-sun = +z) gives N·L = 0 → no sun term. Proves the
/// albedo/ambient split, face-normal, sun_dir plumbing, and `sun_flags`
/// gate end-to-end through `scene_dda.wgsl`.
#[test]
fn scene_dda_sun_lights_floor_by_facing() {
    let Some((gpu, _lock)) = try_init() else {
        return;
    };
    let vsid = 32u32;
    let chunk = decompress_chunk(&floor_chunk(vsid)); // floor voxel at z=100
    let grid = GridUpload {
        vsid,
        origin_chunk: [0, 0, 0],
        chunks_dims: [1, 1, 1],
        pool_dims: [1, 1, 1],
        chunks: vec![([0, 0, 0], chunk)],
    };
    let scene = GpuSceneResident::upload(&gpu.device, &SceneUpload { grids: vec![grid] });

    let (w, h) = (64u32, 64u32);
    let mut renderer = HeadlessSceneRenderer::new(&gpu.device, &gpu.queue, w, h);
    let cam = Camera {
        position: [16.0, 16.0, 50.0],
        right: [1.0, 0.0, 0.0],
        down: [0.0, 1.0, 0.0],
        forward: [0.0, 0.0, 1.0],
        fov_y_rad: 60f32.to_radians(),
    };
    let centre = (h / 2 * w + w / 2) as usize;
    let lum = |p: u32| (p & 0xff) + ((p >> 8) & 0xff) + ((p >> 16) & 0xff);
    let render = |r: &mut HeadlessSceneRenderer| {
        r.render(
            &gpu.device,
            &gpu.queue,
            &scene,
            &[cam],
            cam.fov_y_rad,
            64,
            0.0,
        )[centre]
    };

    // Baseline: no lights (baked-only path).
    let baked = render(&mut renderer);
    assert!(
        is_block_color(baked),
        "centre should be the floor: {baked:#08x}"
    );

    // A single white grid is identity-aligned, so grid-local == world. Sun
    // from above: to-sun direction is up (-z, voxlap z-down).
    let sun = |to_sun: [f32; 3]| SceneLights {
        enabled: true,
        grid_sun_dirs: vec![to_sun],
        sun_color: [1.0; 3],
        sun_intensity: 1.0,
        ambient: [1.0; 3],
        ..SceneLights::default()
    };

    renderer.set_scene_lights(sun([0.0, 0.0, -1.0]));
    let lit_above = render(&mut renderer);
    renderer.set_scene_lights(sun([0.0, 0.0, 1.0]));
    let lit_below = render(&mut renderer);

    assert!(
        lum(lit_above) > lum(baked),
        "sun from above must brighten the floor: baked {baked:#08x} -> {lit_above:#08x}",
    );
    assert!(
        lum(lit_above) > lum(lit_below),
        "sun facing the surface must beat a back-facing sun: {lit_above:#08x} vs {lit_below:#08x}",
    );
}

/// DL.6 — stylized cel banding terraces the sun's diffuse. Two sun
/// directions giving distinct smooth N·L (0.8 vs 0.9 on the flat floor)
/// land on the **same** band at `bands = 2` (both round to the top level),
/// so the stylized floor renders identically while the smooth floor differs.
#[test]
fn scene_dda_cel_banding_terraces_sun() {
    let Some((gpu, _lock)) = try_init() else {
        return;
    };
    let vsid = 32u32;
    let chunk = decompress_chunk(&floor_chunk(vsid));
    let grid = GridUpload {
        vsid,
        origin_chunk: [0, 0, 0],
        chunks_dims: [1, 1, 1],
        pool_dims: [1, 1, 1],
        chunks: vec![([0, 0, 0], chunk)],
    };
    let scene = GpuSceneResident::upload(&gpu.device, &SceneUpload { grids: vec![grid] });

    let (w, h) = (64u32, 64u32);
    let mut renderer = HeadlessSceneRenderer::new(&gpu.device, &gpu.queue, w, h);
    let cam = Camera {
        position: [16.0, 16.0, 50.0],
        right: [1.0, 0.0, 0.0],
        down: [0.0, 1.0, 0.0],
        forward: [0.0, 0.0, 1.0],
        fov_y_rad: 60f32.to_radians(),
    };
    let centre = (h / 2 * w + w / 2) as usize;
    let render = |r: &mut HeadlessSceneRenderer| {
        r.render(
            &gpu.device,
            &gpu.queue,
            &scene,
            &[cam],
            cam.fov_y_rad,
            64,
            0.0,
        )[centre]
    };

    // Floor top normal = up (-z); N·L = -to_sun.z. Two sun elevations:
    // ndl = 0.8 and 0.9 (distinct), both rounding to the top band at bands=2.
    let a = [0.6_f32, 0.0, -0.8]; // ndl 0.8
    let b = [0.435_89_f32, 0.0, -0.9]; // ndl 0.9
    let rig = |to_sun: [f32; 3], bands: u32| SceneLights {
        enabled: true,
        grid_sun_dirs: vec![to_sun],
        sun_color: [1.0; 3],
        sun_intensity: 1.0,
        ambient: [0.1; 3],
        style_bands: bands,
        shadow_tint: [0.1, 0.1, 0.2],
        ..SceneLights::default()
    };

    // Smooth (bands = 0): the two elevations differ.
    renderer.set_scene_lights(rig(a, 0));
    let smooth_a = render(&mut renderer);
    renderer.set_scene_lights(rig(b, 0));
    let smooth_b = render(&mut renderer);
    assert_ne!(
        smooth_a, smooth_b,
        "smooth diffuse must vary with N·L: {smooth_a:#08x} vs {smooth_b:#08x}",
    );

    // Stylized (bands = 2): both N·L round to the same band ⇒ identical.
    renderer.set_scene_lights(rig(a, 2));
    let cel_a = render(&mut renderer);
    renderer.set_scene_lights(rig(b, 2));
    let cel_b = render(&mut renderer);
    assert_eq!(
        cel_a, cel_b,
        "cel banding must terrace both N·L to one level: {cel_a:#08x} vs {cel_b:#08x}",
    );
}

/// DL.2 — point lights: N·L diffuse + distance falloff + hard radius cut.
/// Floor viewed straight down (top-face normal = up = -z). A point light
/// hovering just above the floor centre brightens it vs the baked baseline;
/// a light below the top face contributes nothing (back-facing); a distant
/// light (still above) is dimmer than a near one (falloff).
#[test]
fn scene_dda_point_light_brightens_by_distance_and_facing() {
    let Some((gpu, _lock)) = try_init() else {
        return;
    };
    let vsid = 32u32;
    let chunk = decompress_chunk(&floor_chunk(vsid)); // floor at z=100
    let grid = GridUpload {
        vsid,
        origin_chunk: [0, 0, 0],
        chunks_dims: [1, 1, 1],
        pool_dims: [1, 1, 1],
        chunks: vec![([0, 0, 0], chunk)],
    };
    let scene = GpuSceneResident::upload(&gpu.device, &SceneUpload { grids: vec![grid] });

    let (w, h) = (64u32, 64u32);
    let mut renderer = HeadlessSceneRenderer::new(&gpu.device, &gpu.queue, w, h);
    let cam = Camera {
        position: [16.0, 16.0, 50.0],
        right: [1.0, 0.0, 0.0],
        down: [0.0, 1.0, 0.0],
        forward: [0.0, 0.0, 1.0],
        fov_y_rad: 60f32.to_radians(),
    };
    let centre = (h / 2 * w + w / 2) as usize;
    let lum = |p: u32| (p & 0xff) + ((p >> 8) & 0xff) + ((p >> 16) & 0xff);
    let render = |r: &mut HeadlessSceneRenderer| {
        r.render(
            &gpu.device,
            &gpu.queue,
            &scene,
            &[cam],
            cam.fov_y_rad,
            64,
            0.0,
        )[centre]
    };

    // Baseline: no lights (ambient only via the baked path).
    let baked = render(&mut renderer);
    assert!(
        is_block_color(baked),
        "centre should be the floor: {baked:#08x}"
    );

    // Identity grid ⇒ grid-local == world. The floor centre is ~(16,16,100).
    let one_point = |pos: [f32; 3]| SceneLights {
        enabled: true,
        ambient: [1.0; 3],
        grid_point_lights: vec![vec![GpuLight {
            position: pos,
            radius: 64.0,
            color: [1.0; 3],
            intensity: 2.0,
            casts_shadow: false,
            // SL — `-1.0` outer cosine ⇒ a pure point light (no cone mask).
            spot_dir: [0.0, 0.0, 1.0],
            cos_inner: -1.0,
            cos_outer: -1.0,
        }]],
        ..SceneLights::default()
    };

    renderer.set_scene_lights(one_point([16.0, 16.0, 98.0])); // 2 above the top
    let near_above = render(&mut renderer);
    renderer.set_scene_lights(one_point([16.0, 16.0, 60.0])); // 40 above the top
    let far_above = render(&mut renderer);
    renderer.set_scene_lights(one_point([16.0, 16.0, 110.0])); // below the top face
    let below = render(&mut renderer);

    assert!(
        lum(near_above) > lum(baked),
        "a near point light must brighten the floor: baked {baked:#08x} -> {near_above:#08x}",
    );
    assert!(
        lum(near_above) > lum(far_above),
        "distance falloff: near must beat far: {near_above:#08x} vs {far_above:#08x}",
    );
    assert!(
        lum(below) <= lum(baked) + 2,
        "a back-facing point light must not light the top face: {below:#08x} vs baked {baked:#08x}",
    );
}

/// SL — spot (cone) lights. Same floor-viewed-straight-down setup as the
/// point-light test. A spot hovering just above the centre and aimed straight
/// down (on-axis, `cd == 1`, inside the inner cone) must render **identically**
/// to the equivalent point light — proving the fold passes colour/intensity
/// and that the cone factor saturates to 1 inside the inner half-angle. The
/// same spot aimed sideways puts the centre outside the cone, so it masks to
/// zero and matches the unlit baked baseline (masking + the early-out).
#[test]
fn scene_dda_spot_cone_matches_point_on_axis_and_masks_off_axis() {
    let Some((gpu, _lock)) = try_init() else {
        return;
    };
    let vsid = 32u32;
    let chunk = decompress_chunk(&floor_chunk(vsid)); // floor at z=100
    let grid = GridUpload {
        vsid,
        origin_chunk: [0, 0, 0],
        chunks_dims: [1, 1, 1],
        pool_dims: [1, 1, 1],
        chunks: vec![([0, 0, 0], chunk)],
    };
    let scene = GpuSceneResident::upload(&gpu.device, &SceneUpload { grids: vec![grid] });

    let (w, h) = (64u32, 64u32);
    let mut renderer = HeadlessSceneRenderer::new(&gpu.device, &gpu.queue, w, h);
    let cam = Camera {
        position: [16.0, 16.0, 50.0],
        right: [1.0, 0.0, 0.0],
        down: [0.0, 1.0, 0.0],
        forward: [0.0, 0.0, 1.0],
        fov_y_rad: 60f32.to_radians(),
    };
    let centre = (h / 2 * w + w / 2) as usize;
    let lum = |p: u32| (p & 0xff) + ((p >> 8) & 0xff) + ((p >> 16) & 0xff);
    let render = |r: &mut HeadlessSceneRenderer| {
        r.render(
            &gpu.device,
            &gpu.queue,
            &scene,
            &[cam],
            cam.fov_y_rad,
            64,
            0.0,
        )[centre]
    };

    // Light 2 units above the floor top (z=98 < 100), over the centre (16,16).
    let pos = [16.0, 16.0, 98.0];
    // A wide cone: the light sits close to the surface, so the centre pixel's
    // exact hit is a few degrees off the axis — well inside a 40° inner angle
    // (cone == 1), while a sideways-aimed copy leaves it ~90° off (masked).
    let cos_inner = 40f32.to_radians().cos();
    let cos_outer = 60f32.to_radians().cos();
    let light = |spot_dir: [f32; 3], ci: f32, co: f32| SceneLights {
        enabled: true,
        ambient: [1.0; 3],
        grid_point_lights: vec![vec![GpuLight {
            position: pos,
            radius: 64.0,
            color: [1.0; 3],
            intensity: 2.0,
            casts_shadow: false,
            spot_dir,
            cos_inner: ci,
            cos_outer: co,
        }]],
        ..SceneLights::default()
    };

    let baked = render(&mut renderer); // no lights set yet ⇒ ambient/baked only
                                       // A pure point light (cone disabled, cos_outer = -1).
    renderer.set_scene_lights(light([0.0, 0.0, 1.0], -1.0, -1.0));
    let point = render(&mut renderer);
    // A real cone aimed straight down (+z) at the floor ⇒ centre on-axis.
    renderer.set_scene_lights(light([0.0, 0.0, 1.0], cos_inner, cos_outer));
    let spot_on = render(&mut renderer);
    // The same cone aimed sideways ⇒ the centre is outside it.
    renderer.set_scene_lights(light([1.0, 0.0, 0.0], cos_inner, cos_outer));
    let spot_off = render(&mut renderer);

    assert!(
        lum(point) > lum(baked),
        "sanity: the point light must brighten the floor: {baked:#08x} -> {point:#08x}",
    );
    assert_eq!(
        spot_on, point,
        "on-axis spot (cd=1, inside inner cone) must equal the point light: \
         {spot_on:#08x} vs {point:#08x}",
    );
    assert_eq!(
        spot_off, baked,
        "off-cone spot must contribute nothing (masked to zero): \
         {spot_off:#08x} vs baked {baked:#08x}",
    );
}

/// DL.3 — stylized hard shadows. A short wall stands on the floor at x≈16;
/// the camera looks straight down at the floor point (14,16) — which the
/// wall does NOT block from above. An angled sun (toward +x and up) is
/// occluded by the wall on its way to that point, so with shadow-casting ON
/// the point is darker than with the same sun and shadows OFF. Proves the
/// `shadow_occluded` DDA, the normal bias, and the `casts_shadow` gate.
#[test]
fn scene_dda_sun_shadow_darkens_occluded_floor() {
    let Some((gpu, _lock)) = try_init() else {
        return;
    };
    let vsid = 32u32;
    // Wall at x ∈ [16,18), 10 voxels tall above the floor (z 90..100).
    let chunk = decompress_chunk(&floor_with_wall_chunk(vsid, 16, 18, 90));
    let grid = GridUpload {
        vsid,
        origin_chunk: [0, 0, 0],
        chunks_dims: [1, 1, 1],
        pool_dims: [1, 1, 1],
        chunks: vec![([0, 0, 0], chunk)],
    };
    let scene = GpuSceneResident::upload(&gpu.device, &SceneUpload { grids: vec![grid] });

    let (w, h) = (64u32, 64u32);
    let mut renderer = HeadlessSceneRenderer::new(&gpu.device, &gpu.queue, w, h);
    // Directly above the floor point (14,16): the centre pixel shows it, and
    // the wall at x=16 doesn't block this straight-down view.
    let cam = Camera {
        position: [14.0, 16.0, 50.0],
        right: [1.0, 0.0, 0.0],
        down: [0.0, 1.0, 0.0],
        forward: [0.0, 0.0, 1.0],
        fov_y_rad: 60f32.to_radians(),
    };
    let centre = (h / 2 * w + w / 2) as usize;
    let lum = |p: u32| (p & 0xff) + ((p >> 8) & 0xff) + ((p >> 16) & 0xff);
    let render = |r: &mut HeadlessSceneRenderer| {
        r.render(
            &gpu.device,
            &gpu.queue,
            &scene,
            &[cam],
            cam.fov_y_rad,
            64,
            0.0,
        )[centre]
    };

    // Sun toward +x and up (to-sun = normalize(1,0,-1)): it reaches (14,16)
    // only by passing through the wall at x=16 → occluded when shadows cast.
    let s = std::f32::consts::FRAC_1_SQRT_2;
    let sun = |casts_shadow: bool| SceneLights {
        enabled: true,
        grid_sun_dirs: vec![[s, 0.0, -s]],
        sun_color: [1.0; 3],
        sun_intensity: 3.0,
        sun_casts_shadow: casts_shadow,
        ambient: [0.5; 3],
        shadow_strength: 1.0, // full black shadow
        shadow_bias: 1.5,
        shadow_max_dist: 512.0,
        shadow_max_steps: 256,
        ..SceneLights::default()
    };

    renderer.set_scene_lights(sun(false));
    let lit = render(&mut renderer); // sun reaches the floor (no shadow test)
    renderer.set_scene_lights(sun(true));
    let shadowed = render(&mut renderer); // wall occludes the sun → darker

    // Both are the floor (low blue), not the bluish sky — even at intensity 3
    // the floor's blue stays 0 while sky is ~0xdc.
    let blue = |p: u32| (p >> 16) & 0xff;
    assert!(
        blue(lit) < 70 && blue(shadowed) < 70,
        "expected floor, not sky"
    );
    assert!(
        lum(shadowed) < lum(lit),
        "wall must cast a sun shadow on the floor: lit {lit:#08x} -> shadowed {shadowed:#08x}",
    );
}

/// XS.3 — cross-grid sun shadow: a wall in grid **B** (placed at a world
/// offset) shadows the floor of grid **A**. The shadow only appears if the
/// shadow ray crosses from A into B in world space, so shadows-on must be
/// darker than shadows-off. Exercises the per-grid world transform packing +
/// `shadow_occluded_world`.
#[test]
fn scene_dda_cross_grid_sun_shadow() {
    let Some((gpu, _lock)) = try_init() else {
        return;
    };
    let vsid = 32u32;
    // Grid A: plain floor at z=100, world origin (0,0,0).
    let grid_a = GridUpload {
        vsid,
        origin_chunk: [0, 0, 0],
        chunks_dims: [1, 1, 1],
        pool_dims: [1, 1, 1],
        chunks: vec![([0, 0, 0], decompress_chunk(&floor_chunk(vsid)))],
    };
    // Grid B: a wall (x∈[16,18), z 90..100) — moved to world (+2,0,0) below, so
    // its wall sits at world x∈[18,20), on the to-sun ray from A's floor point
    // (the ray reaches it at z≈96, inside the wall's z-span).
    let grid_b = GridUpload {
        vsid,
        origin_chunk: [0, 0, 0],
        chunks_dims: [1, 1, 1],
        pool_dims: [1, 1, 1],
        chunks: vec![(
            [0, 0, 0],
            decompress_chunk(&floor_with_wall_chunk(vsid, 16, 18, 90)),
        )],
    };
    let scene = GpuSceneResident::upload(
        &gpu.device,
        &SceneUpload {
            grids: vec![grid_a, grid_b],
        },
    );

    let (w, h) = (64u32, 64u32);
    let mut renderer = HeadlessSceneRenderer::new(&gpu.device, &gpu.queue, w, h);
    // Straight down over A's floor point (14,16). World camera.
    let cam = Camera {
        position: [14.0, 16.0, 50.0],
        right: [1.0, 0.0, 0.0],
        down: [0.0, 1.0, 0.0],
        forward: [0.0, 0.0, 1.0],
        fov_y_rad: 60f32.to_radians(),
    };
    // Per-grid local cameras: A is identity; B is offset by world (2,0,0).
    let cam_b = Camera {
        position: [14.0 - 2.0, 16.0, 50.0],
        ..cam
    };
    // Per-grid world transforms: A identity, B translated +2 in x.
    let xf_a = GridWorldTransform::default();
    let xf_b = GridWorldTransform {
        origin: [2.0, 0.0, 0.0],
        ..GridWorldTransform::default()
    };
    let centre = (h / 2 * w + w / 2) as usize;
    let lum = |p: u32| (p & 0xff) + ((p >> 8) & 0xff) + ((p >> 16) & 0xff);
    let render = |r: &mut HeadlessSceneRenderer| {
        r.render_with_transforms(
            &gpu.device,
            &gpu.queue,
            &scene,
            &[cam, cam_b],
            &[xf_a, xf_b],
            cam.fov_y_rad,
            64,
            0.0,
        )[centre]
    };

    // Sun toward +x and up: reaches A's floor (14,16,100) only by crossing
    // grid B's wall at world x≈24 → occluded when shadows cast.
    let s = std::f32::consts::FRAC_1_SQRT_2;
    let sun = |casts_shadow: bool| SceneLights {
        enabled: true,
        grid_sun_dirs: vec![[s, 0.0, -s], [s, 0.0, -s]], // both grids identity-rot
        sun_color: [1.0; 3],
        sun_intensity: 3.0,
        sun_casts_shadow: casts_shadow,
        ambient: [0.5; 3],
        shadow_strength: 1.0,
        shadow_bias: 1.5,
        shadow_max_dist: 512.0,
        shadow_max_steps: 256,
        ..SceneLights::default()
    };

    renderer.set_scene_lights(sun(false));
    let lit = render(&mut renderer);
    renderer.set_scene_lights(sun(true));
    let shadowed = render(&mut renderer);

    let blue = |p: u32| (p >> 16) & 0xff;
    assert!(
        blue(lit) < 70 && blue(shadowed) < 70,
        "expected A's floor, not sky: lit={lit:#08x} shadowed={shadowed:#08x}"
    );
    assert!(
        lum(shadowed) < lum(lit),
        "grid B's wall must cast a cross-grid sun shadow on grid A: lit {lit:#08x} -> shadowed {shadowed:#08x}",
    );
}

#[test]
fn scene_dda_fine_occluder_cross_grid_shadow_survives_lod() {
    // SC.4 bug: on the GPU a fine (vws < 1) grid casts NO cross-grid sun
    // shadow when LOD is on (the demo: a mini ship over a coarse planet). The
    // shadow ray marches the occluder at pick_mip(receiver world-t), and a
    // coarse mip of a small fine grid mis-resolves → the shadow vanishes.
    // Repro: an unscaled floor + a fine (vws=0.25) slab above it, rendered
    // with LOD OFF (mip_scan_dist=0) and ON (>0); the shadow must survive.
    let Some((gpu, _lock)) = try_init() else {
        return;
    };
    let vsid = 32u32;
    let grid_a = GridUpload {
        vsid,
        origin_chunk: [0, 0, 0],
        chunks_dims: [1, 1, 1],
        pool_dims: [1, 1, 1],
        chunks: vec![([0, 0, 0], decompress_chunk(&floor_chunk(vsid)))],
    };
    // Fine slab low in its chunk (local z 8..12); decompress_chunk builds its
    // mip ladder, so LOD can pick a coarse mip for the shadow march.
    let grid_b = GridUpload {
        vsid,
        origin_chunk: [0, 0, 0],
        chunks_dims: [1, 1, 1],
        pool_dims: [1, 1, 1],
        chunks: vec![([0, 0, 0], decompress_chunk(&block_chunk(vsid, 8, 12)))],
    };
    let scene = GpuSceneResident::upload(
        &gpu.device,
        &SceneUpload {
            grids: vec![grid_a, grid_b],
        },
    );

    let (w, h) = (64u32, 64u32);
    let mut renderer = HeadlessSceneRenderer::new(&gpu.device, &gpu.queue, w, h);
    let cam = Camera {
        position: [14.0, 16.0, 50.0],
        right: [1.0, 0.0, 0.0],
        down: [0.0, 1.0, 0.0],
        forward: [0.0, 0.0, 1.0],
        fov_y_rad: 60f32.to_radians(),
    };
    let b_origin = [10.0f32, 12.0, 30.0];
    let cam_b = Camera {
        position: [
            cam.position[0] - b_origin[0],
            cam.position[1] - b_origin[1],
            cam.position[2] - b_origin[2],
        ],
        ..cam
    };
    let xf_a = GridWorldTransform::default();
    let xf_b = GridWorldTransform {
        origin: b_origin,
        voxel_world_size: 0.25,
        ..GridWorldTransform::default()
    };
    let centre = (h / 2 * w + w / 2) as usize;
    let lum = |p: u32| (p & 0xff) + ((p >> 8) & 0xff) + ((p >> 16) & 0xff);
    let render = |r: &mut HeadlessSceneRenderer, msd: f32| -> u32 {
        r.render_with_transforms(
            &gpu.device,
            &gpu.queue,
            &scene,
            &[cam, cam_b],
            &[xf_a, xf_b],
            cam.fov_y_rad,
            64,
            msd,
        )[centre]
    };

    let sd = glam::Vec3::new(0.12, 0.0, -0.99).normalize();
    let sun = |casts_shadow: bool| SceneLights {
        enabled: true,
        grid_sun_dirs: vec![[sd.x, sd.y, sd.z], [sd.x, sd.y, sd.z]],
        sun_color: [1.0; 3],
        sun_intensity: 3.0,
        sun_casts_shadow: casts_shadow,
        ambient: [0.5; 3],
        shadow_strength: 1.0,
        shadow_bias: 1.5,
        shadow_max_dist: 512.0,
        shadow_max_steps: 768,
        ..SceneLights::default()
    };

    renderer.set_scene_lights(sun(false));
    let lit = render(&mut renderer, 0.0);
    renderer.set_scene_lights(sun(true));
    let shadowed_lod_off = render(&mut renderer, 0.0);
    let shadowed_lod_on = render(&mut renderer, 8.0);
    eprintln!(
        "fine occluder LOD: lit={}({}) lod_off={}({}) lod_on={}({})",
        lit,
        lum(lit),
        shadowed_lod_off,
        lum(shadowed_lod_off),
        shadowed_lod_on,
        lum(shadowed_lod_on),
    );

    // The fine occluder must cast a cross-grid shadow with LOD off…
    assert!(
        lum(shadowed_lod_off) < lum(lit),
        "fine occluder must cast a shadow (LOD off): lit {lit:#08x} -> {shadowed_lod_off:#08x}"
    );
    // …and it must NOT vanish when LOD is on (the reported bug).
    assert!(
        lum(shadowed_lod_on) < lum(lit),
        "fine occluder shadow must survive LOD (the ship-over-planet bug): \
         lit {lit:#08x} -> lod_on {shadowed_lod_on:#08x}"
    );
}

#[test]
fn scene_dda_scaled_grid_composites_by_world_depth() {
    // SC.4 — GPU per-grid voxel_world_size. Two grids at the same origin,
    // both blocks on the camera column but different scale, so the world and
    // voxel-frame depth metrics DISAGREE:
    //  - Grid A (vws 1.0): RED block at chunk y=4 → world y-near 128.
    //  - Grid B (vws 2.0): BLUE block at chunk y=3 → world y-near 3·32·2 = 192
    //    (FARTHER in world), but voxel-near 96 (< A's 128).
    // Correct (world depth): A (128) occludes B (192) → RED wins the min-t
    // composite. Without the shader's chunk_dim/vsize × vws the marcher would
    // put B at t≈96 < 128 → BLUE would win.
    let Some((gpu, _lock)) = try_init() else {
        return;
    };
    let vsid = 32u32;
    let red = [0x00u8, 0x00, 0xff, 0x80]; // BGRA → R=0xff
    let blue = [0xffu8, 0x00, 0x00, 0x80]; // BGRA → B=0xff
    let grid_a = GridUpload {
        vsid,
        origin_chunk: [0, 0, 0],
        chunks_dims: [1, 8, 1],
        pool_dims: [1, 8, 1],
        chunks: vec![(
            [0, 4, 0],
            decompress_chunk(&block_chunk_bgra(vsid, 0, 31, red)),
        )],
    };
    let grid_b = GridUpload {
        vsid,
        origin_chunk: [0, 0, 0],
        chunks_dims: [1, 8, 1],
        pool_dims: [1, 8, 1],
        chunks: vec![(
            [0, 3, 0],
            decompress_chunk(&block_chunk_bgra(vsid, 0, 31, blue)),
        )],
    };
    let scene = GpuSceneResident::upload(
        &gpu.device,
        &SceneUpload {
            grids: vec![grid_a, grid_b],
        },
    );

    let (w, h) = (64u32, 64u32);
    let renderer = HeadlessSceneRenderer::new(&gpu.device, &gpu.queue, w, h);
    // World camera at (16,0,16) looking +y. Both grids sit at origin (0,0,0),
    // so each per-grid (world-local) camera equals the world camera.
    let cam = Camera {
        position: [16.0, 0.0, 16.0],
        right: [1.0, 0.0, 0.0],
        down: [0.0, 0.0, 1.0],
        forward: [0.0, 1.0, 0.0],
        fov_y_rad: 60f32.to_radians(),
    };
    let xf_a = GridWorldTransform::default(); // vws 1.0
    let xf_b = GridWorldTransform {
        voxel_world_size: 2.0,
        ..GridWorldTransform::default()
    };
    let fb = renderer.render_with_transforms(
        &gpu.device,
        &gpu.queue,
        &scene,
        &[cam, cam],
        &[xf_a, xf_b],
        cam.fov_y_rad,
        64,
        0.0,
    );
    let centre = (h / 2 * w + w / 2) as usize;
    let p = fb[centre];
    let (r, b) = (p & 0xff, (p >> 16) & 0xff);
    // Shading scales all channels uniformly, so the R>B ratio is preserved.
    assert!(
        r > b,
        "the world-nearer vws=1 RED grid must win the min-t composite; BLUE \
         here means grid B's vws=2 depth (chunk_dim/vsize × vws) wasn't applied: {p:#08x}"
    );
}

#[test]
fn scene_dda_scaled_grid_depth_is_world() {
    // SC.4 — the GPU depth buffer stores `best_t` in WORLD units even for a
    // scaled grid, so it agrees with the CPU compose depth. This is the GPU
    // half of the CPU-vs-GPU depth parity: the CPU test
    // `sc1_scaled_grid_depth_is_world` (roxlap-scene) asserts its zbuffer at a
    // world VALUE, and this asserts the GPU depth at a world value too — both
    // world ⇒ they agree. A vws=2 grid whose block near-face is at world
    // y=128 must read depth 128, not the voxel-frame 64.
    let Some((gpu, _lock)) = try_init() else {
        return;
    };
    let vsid = 32u32;
    // Block chunk at index y=2 → voxel y 64.. → world y 128.. (× vws 2).
    let grid = GridUpload {
        vsid,
        origin_chunk: [0, 0, 0],
        chunks_dims: [1, 4, 1],
        pool_dims: [1, 4, 1],
        chunks: vec![([0, 2, 0], decompress_chunk(&block_chunk(vsid, 0, 31)))],
    };
    let scene = GpuSceneResident::upload(&gpu.device, &SceneUpload { grids: vec![grid] });

    let (w, h) = (64u32, 64u32);
    let renderer = HeadlessSceneRenderer::new(&gpu.device, &gpu.queue, w, h);
    // World camera at (32,0,32) looking +y; centre ray hits the block interior
    // (voxel x=16, z=16 under vws=2). Grid at origin ⇒ per-grid cam == world.
    let cam = Camera {
        position: [32.0, 0.0, 32.0],
        right: [1.0, 0.0, 0.0],
        down: [0.0, 0.0, 1.0],
        forward: [0.0, 1.0, 0.0],
        fov_y_rad: 60f32.to_radians(),
    };
    let xf = GridWorldTransform {
        voxel_world_size: 2.0,
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
        0.0,
    );
    let centre = (h / 2 * w + w / 2) as usize;
    let d = depth[centre];
    assert!(
        d.is_finite(),
        "centre ray must hit the scaled block, got {d}"
    );
    // Near face: chunk 2 · vsid 32 · vws 2 = world y 128; camera at y=0.
    assert!(
        (d - 128.0).abs() <= 4.0,
        "GPU depth must be WORLD (128); {d} ≈ 64 would mean the voxel-frame \
         t wasn't scaled by vws"
    );
}

#[test]
fn scene_dda_fine_scaled_grid_composites_by_world_depth() {
    // SC.4 — the vws<1 (fine grid) mirror of the coarse test, for symmetry.
    //  - Grid A (vws 1.0): RED block at chunk y=3 → world y-near 96.
    //  - Grid B (vws 0.5): BLUE block at chunk y=4 → world y-near 4·32·0.5 = 64
    //    (NEARER in world), but voxel-near 128 (> A's 96).
    // Correct (world depth): B (64) occludes A (96) → BLUE wins. Without the
    // × vws the marcher would put B at t≈128 > 96 → RED would win.
    let Some((gpu, _lock)) = try_init() else {
        return;
    };
    let vsid = 32u32;
    let red = [0x00u8, 0x00, 0xff, 0x80];
    let blue = [0xffu8, 0x00, 0x00, 0x80];
    let grid_a = GridUpload {
        vsid,
        origin_chunk: [0, 0, 0],
        chunks_dims: [1, 8, 1],
        pool_dims: [1, 8, 1],
        chunks: vec![(
            [0, 3, 0],
            decompress_chunk(&block_chunk_bgra(vsid, 0, 31, red)),
        )],
    };
    let grid_b = GridUpload {
        vsid,
        origin_chunk: [0, 0, 0],
        chunks_dims: [1, 8, 1],
        pool_dims: [1, 8, 1],
        chunks: vec![(
            [0, 4, 0],
            decompress_chunk(&block_chunk_bgra(vsid, 0, 31, blue)),
        )],
    };
    let scene = GpuSceneResident::upload(
        &gpu.device,
        &SceneUpload {
            grids: vec![grid_a, grid_b],
        },
    );

    let (w, h) = (64u32, 64u32);
    let renderer = HeadlessSceneRenderer::new(&gpu.device, &gpu.queue, w, h);
    // Camera at (8,0,8): inside both blocks' world x/z spans (grid B's
    // vws=0.5 block only reaches world 16 in x and z).
    let cam = Camera {
        position: [8.0, 0.0, 8.0],
        right: [1.0, 0.0, 0.0],
        down: [0.0, 0.0, 1.0],
        forward: [0.0, 1.0, 0.0],
        fov_y_rad: 60f32.to_radians(),
    };
    let xf_a = GridWorldTransform::default(); // vws 1.0
    let xf_b = GridWorldTransform {
        voxel_world_size: 0.5,
        ..GridWorldTransform::default()
    };
    let fb = renderer.render_with_transforms(
        &gpu.device,
        &gpu.queue,
        &scene,
        &[cam, cam],
        &[xf_a, xf_b],
        cam.fov_y_rad,
        64,
        0.0,
    );
    let centre = (h / 2 * w + w / 2) as usize;
    let p = fb[centre];
    let (r, b) = (p & 0xff, (p >> 16) & 0xff);
    assert!(
        b > r,
        "the world-nearer vws=0.5 BLUE grid must win the min-t composite; RED \
         here means grid B's vws=0.5 depth wasn't applied: {p:#08x}"
    );
}

/// `vsid × vsid` chunk: every column carries an `nvox`-tall coloured
/// slab whose top sits at `surf` — so the chunk's mip-0 colour count is
/// `vsid² · nvox`. With `vsid = 128, nvox = 8` that's 131072 colours,
/// 2× the per-chunk GPU colour stride — the dense-chunk case the cave
/// demo hits (a fully solid 128×128×256 cave chunk).
fn dense_floor_chunk(vsid: u32, surf: u8, nvox: u8) -> Vxl {
    let n_cols = (vsid as usize) * (vsid as usize);
    let bot = surf + nvox - 1;
    let mut data: Vec<u8> = Vec::with_capacity(n_cols * (4 + nvox as usize * 4));
    let mut column_offset: Vec<u32> = Vec::with_capacity(n_cols + 1);
    let bgra = [0x00u8, 0x80, 0xff, 0x80];
    for _ in 0..n_cols {
        column_offset.push(u32::try_from(data.len()).expect("offset fits"));
        data.extend_from_slice(&[0, surf, bot, 0]);
        for _ in 0..nvox {
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

/// Regression for the cave-demo "half the map is black" report: a chunk
/// whose colour count exceeds the per-chunk GPU stride must NOT have its
/// colours truncated. Columns are stored in `y·vsid + x` order, so a
/// truncated tail blacks out the high-`y` spatial half of the chunk.
///
/// A dense top-faced floor is viewed straight down; with the bug the
/// bottom half of the frame (world `y ≥ vsid/2`, the truncated columns)
/// renders black instead of the floor colour.
#[test]
fn scene_dda_dense_chunk_colours_not_truncated() {
    let Some((gpu, _lock)) = try_init() else {
        return;
    };
    let vsid = 128u32;
    let chunk = decompress_chunk(&dense_floor_chunk(vsid, 100, 8));
    // Precondition: this chunk really does exceed the per-chunk stride,
    // so the test exercises the truncation path (not a no-op).
    assert!(
        chunk.mips[0].colors.len() > roxlap_gpu::scene::COLORS_PER_CHUNK_WORDS as usize,
        "test chunk must exceed the colour stride to exercise truncation \
         ({} colours)",
        chunk.mips[0].colors.len()
    );
    let grid = GridUpload {
        vsid,
        origin_chunk: [0, 0, 0],
        chunks_dims: [1, 1, 1],
        pool_dims: [1, 1, 1],
        chunks: vec![([0, 0, 0], chunk)],
    };
    let scene = GpuSceneResident::upload(&gpu.device, &SceneUpload { grids: vec![grid] });

    let (w, h) = (64u32, 64u32);
    let renderer = HeadlessSceneRenderer::new(&gpu.device, &gpu.queue, w, h);
    // Inside the chunk, above the floor (z=2 < 100), looking straight
    // down (+z). Screen-down maps to world +y, so the high-y (truncated)
    // columns land in the bottom half of the frame.
    let cam = Camera {
        position: [vsid as f32 * 0.5, vsid as f32 * 0.5, 2.0],
        right: [1.0, 0.0, 0.0],
        down: [0.0, 1.0, 0.0],
        forward: [0.0, 0.0, 1.0],
        fov_y_rad: 60f32.to_radians(),
    };
    let fb = renderer.render(
        &gpu.device,
        &gpu.queue,
        &scene,
        &[cam],
        cam.fov_y_rad,
        64,
        0.0,
    );

    // Count floor-colour pixels in the top half (low world-y, never
    // truncated) vs the bottom half (high world-y, truncated by the bug).
    let half = (h / 2) as usize;
    let count = |rows: std::ops::Range<usize>| {
        let mut n = 0;
        for y in rows {
            for x in 0..w as usize {
                if is_block_color(fb[y * w as usize + x]) {
                    n += 1;
                }
            }
        }
        n
    };
    let top = count(0..half);
    let bottom = count(half..h as usize);
    eprintln!("dense_chunk: top-half floor px {top}, bottom-half floor px {bottom}");
    assert!(top > 0, "top half should show the floor, got {top}");
    // The bottom half is the truncated spatial half — it must still
    // render the floor, not black.
    assert!(
        bottom * 2 > top,
        "bottom (high-y) half is mostly black — colours were truncated \
         (top {top} vs bottom {bottom})",
    );
}

/// Lifting the 16-grid cap moved the per-grid cameras out of a fixed
/// `array<…, 16>` uniform and into a runtime-sized storage buffer
/// (binding 15). This guards that grid `g` marches with **its own**
/// `grid_cameras[g]`, not `cameras[0]` for every grid. Two identical
/// floor grids get OPPOSITE cameras: grid 0 looks up (away → sky), grid
/// 1 looks down (at the floor). The floor can only appear if grid 1's
/// own camera was used; a control with both cameras up must be pure sky.
#[test]
fn scene_dda_per_grid_cameras_are_independent() {
    let Some((gpu, _lock)) = try_init() else {
        return;
    };
    let vsid = 32u32;
    let mk_floor = || GridUpload {
        vsid,
        origin_chunk: [0, 0, 0],
        chunks_dims: [1, 1, 1],
        pool_dims: [1, 1, 1],
        chunks: vec![([0, 0, 0], decompress_chunk(&floor_chunk(vsid)))],
    };
    let scene = GpuSceneResident::upload(
        &gpu.device,
        &SceneUpload {
            grids: vec![mk_floor(), mk_floor()],
        },
    );
    assert_eq!(scene.grid_count, 2, "two-grid scene");

    let (w, h) = (64u32, 64u32);
    let renderer = HeadlessSceneRenderer::new(&gpu.device, &gpu.queue, w, h);
    // Above the floor (z=50 < 100). Down = +z (voxlap z-down) hits it;
    // up = −z looks at empty space → sky.
    let cam_down = Camera {
        position: [16.0, 16.0, 50.0],
        right: [1.0, 0.0, 0.0],
        down: [0.0, 1.0, 0.0],
        forward: [0.0, 0.0, 1.0],
        fov_y_rad: 60f32.to_radians(),
    };
    let cam_up = Camera {
        forward: [0.0, 0.0, -1.0],
        ..cam_down
    };
    let fov = cam_down.fov_y_rad;

    // grid 0 looks away (sky), grid 1 looks at its floor.
    let fb = renderer.render(
        &gpu.device,
        &gpu.queue,
        &scene,
        &[cam_up, cam_down],
        fov,
        64,
        0.0,
    );
    let floor_px = fb.iter().filter(|&&p| is_block_color(p)).count();
    assert!(
        floor_px > 0,
        "grid 1's floor must be visible via grid_cameras[1] — got {floor_px} floor px \
         (per-grid camera indexing broken?)",
    );

    // Control: BOTH grids look away → pure sky. If the shader read a
    // stale/shared camera this would disagree with the result above.
    let fb2 = renderer.render(
        &gpu.device,
        &gpu.queue,
        &scene,
        &[cam_up, cam_up],
        fov,
        64,
        0.0,
    );
    let floor_px2 = fb2.iter().filter(|&&p| is_block_color(p)).count();
    assert_eq!(
        floor_px2, 0,
        "both grids looking away must be all sky — got {floor_px2} floor px",
    );
}

#[test]
fn aabb_tracks_streaming_refresh_and_evict() {
    // GPU.13.0 — the early-out box is maintained live: installing a
    // chunk at a new index must GROW the AABB (so the shader never
    // skips streamed-in terrain), and evicting it must SHRINK it back.
    let Some((gpu, _lock)) = try_init() else {
        return;
    };
    let vsid = 32u32;
    let grid = GridUpload {
        vsid,
        origin_chunk: [0, 0, 0],
        chunks_dims: [1, 1, 1],
        pool_dims: [8, 8, 1], // room for streamed indices
        chunks: vec![([0, 0, 0], decompress_chunk(&block_chunk(vsid, 0, 31)))],
    };
    let mut scene = GpuSceneResident::upload(&gpu.device, &SceneUpload { grids: vec![grid] });
    assert_eq!(scene.static_meta[0].aabb_min, [0, 0, 0]);
    assert_eq!(scene.static_meta[0].aabb_max, [0, 0, 0]);

    // Stream in a far chunk at (3, 2, 0) → AABB grows to cover it.
    let far = decompress_chunk(&block_chunk(vsid, 0, 31));
    scene.refresh_chunk(&gpu.queue, 0, [3, 2, 0], &far);
    assert_eq!(scene.static_meta[0].aabb_min, [0, 0, 0]);
    assert_eq!(scene.static_meta[0].aabb_max, [3, 2, 0]);

    // Evict it → AABB shrinks back to the lone origin chunk.
    scene.evict_chunk(&gpu.queue, 0, [3, 2, 0]);
    assert_eq!(scene.static_meta[0].aabb_min, [0, 0, 0]);
    assert_eq!(scene.static_meta[0].aabb_max, [0, 0, 0]);
}

#[test]
fn scene_dda_renders_bedrock_wall_face_solid() {
    let Some((gpu, _lock)) = try_init() else {
        return;
    };
    eprintln!("wall_render: adapter = {}", gpu.adapter_info);

    // A wall chunk: textured top at z=40, bedrock 41..255 below. Place
    // it far along +y; the camera looks at its face from BELOW the
    // textured top (z=128, deep in the bedrock region) — exactly the
    // cliff-face view that pre-fix showed sky through.
    let vsid = 32u32;
    let chunk = decompress_chunk(&wall_chunk(vsid, 40));
    let grid = GridUpload {
        vsid,
        origin_chunk: [0, 0, 0],
        chunks_dims: [1, 8, 1],
        pool_dims: [1, 8, 1],
        chunks: vec![([0, 4, 0], chunk)],
    };
    let scene = GpuSceneResident::upload(&gpu.device, &SceneUpload { grids: vec![grid] });

    let (w, h) = (64u32, 64u32);
    let renderer = HeadlessSceneRenderer::new(&gpu.device, &gpu.queue, w, h);
    let cam = Camera {
        position: [vsid as f32 * 0.5, 0.0, 128.0], // z=128 = bedrock region
        right: [1.0, 0.0, 0.0],
        down: [0.0, 0.0, 1.0],
        forward: [0.0, 1.0, 0.0],
        fov_y_rad: 30f32.to_radians(),
    };
    let fb = renderer.render(
        &gpu.device,
        &gpu.queue,
        &scene,
        &[cam],
        cam.fov_y_rad,
        64,
        0.0,
    );
    let centre = fb[(h / 2 * w + w / 2) as usize];
    eprintln!("wall_render: centre pixel = {centre:#08x}");
    // The bedrock face must be SOLID and inherit the surface colour
    // (was sky before the bedrock-as-solid fix).
    assert!(
        is_block_color(centre),
        "bedrock wall face should be solid surface colour, got {centre:#08x} (sky = regression)",
    );
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
    let renderer = HeadlessSceneRenderer::new(&gpu.device, &gpu.queue, w, h);

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
        0.0, // mip_scan_dist=0 → always mip-0
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

#[test]
fn hierarchical_skip_hits_far_chunk_and_tracks_evict() {
    // GPU.13.1 — a ray crossing a long run of EMPTY chunks (stepped
    // read-free under the chunk-occupancy pyramid) must still enter
    // the far occupied chunk exactly — the integer block test cannot
    // overshoot it. Evicting the chunk must clear the pyramid (same
    // view → pure sky), and re-installing must set it again.
    let Some((gpu, _lock)) = try_init() else {
        return;
    };
    let vsid = 32u32;
    let chunk = decompress_chunk(&block_chunk(vsid, 0, 31));
    let grid = GridUpload {
        vsid,
        origin_chunk: [0, 0, 0],
        chunks_dims: [1, 8, 1],
        pool_dims: [1, 8, 1],
        chunks: vec![([0, 7, 0], chunk.clone())], // 7 empty chunks in front
    };
    let mut scene = GpuSceneResident::upload(&gpu.device, &SceneUpload { grids: vec![grid] });
    // Pool [1,8,1] → 3 pyramid levels above L0.
    assert_eq!(scene.static_meta[0].chunk_occ_levels, 3);

    let (w, h) = (64u32, 64u32);
    let renderer = HeadlessSceneRenderer::new(&gpu.device, &gpu.queue, w, h);
    let cam = Camera {
        position: [vsid as f32 * 0.5, 8.0, 16.0],
        right: [1.0, 0.0, 0.0],
        down: [0.0, 0.0, 1.0],
        forward: [0.0, 1.0, 0.0], // toward the far chunk at +y
        fov_y_rad: 30f32.to_radians(),
    };
    let render = |scene: &GpuSceneResident| {
        let fb = renderer.render(
            &gpu.device,
            &gpu.queue,
            scene,
            &[cam],
            cam.fov_y_rad,
            64,
            0.0,
        );
        fb.iter().filter(|&&p| is_block_color(p)).count()
    };

    let visible = render(&scene);
    assert!(
        visible > 0,
        "far chunk must render through the skipped empty run"
    );

    scene.evict_chunk(&gpu.queue, 0, [0, 7, 0]);
    // The maintenance path re-ORed the ancestors down to all-empty.
    assert!(
        scene.chunk_occ_pyramid_shadow()[0]
            .iter()
            .all(|lvl| lvl.iter().all(|&w| w == 0)),
        "evicting the only chunk empties every pyramid level"
    );
    assert_eq!(render(&scene), 0, "evicted chunk leaves pure sky");

    scene.refresh_chunk(&gpu.queue, 0, [0, 7, 0], &chunk);
    assert_eq!(
        render(&scene),
        visible,
        "re-installed chunk renders exactly as before"
    );
}

/// EV — headless material plumbing + GPU emissive parity: an emissive
/// terrain mapping renders full-bright through the real `scene_dda.wgsl`
/// pipeline, matching the CPU `emissive_shade` ladder exactly, and is
/// independent of both the baked byte and a hostile dynamic rig. Also
/// gates: an empty map re-renders byte-identically to the pre-material
/// baseline (the fast path stays byte-exact).
#[test]
fn scene_dda_emissive_ignores_lighting() {
    use roxlap_formats::material::{Material, MaterialTable};
    use roxlap_formats::Rgb;

    let Some((gpu, _lock)) = try_init() else {
        return;
    };
    let vsid = 32u32;
    // A **dim** floor at z=100: baked byte 0x40 ⇒ the baked path renders at
    // half albedo, so full-bright emissive is unmistakable. BGRA, R=0xff.
    let chunk = decompress_chunk(&block_chunk_bgra(vsid, 100, 100, [0x00, 0x80, 0xff, 0x40]));
    let grid = GridUpload {
        vsid,
        origin_chunk: [0, 0, 0],
        chunks_dims: [1, 1, 1],
        pool_dims: [1, 1, 1],
        chunks: vec![([0, 0, 0], chunk)],
    };
    let scene = GpuSceneResident::upload(&gpu.device, &SceneUpload { grids: vec![grid] });

    let (w, h) = (64u32, 64u32);
    let mut renderer = HeadlessSceneRenderer::new(&gpu.device, &gpu.queue, w, h);
    let cam = Camera {
        position: [16.0, 16.0, 50.0],
        right: [1.0, 0.0, 0.0],
        down: [0.0, 1.0, 0.0],
        forward: [0.0, 0.0, 1.0],
        fov_y_rad: 60f32.to_radians(),
    };
    let centre = (h / 2 * w + w / 2) as usize;
    let render = |r: &mut HeadlessSceneRenderer| {
        r.render(
            &gpu.device,
            &gpu.queue,
            &scene,
            &[cam],
            cam.fov_y_rad,
            64,
            0.0,
        )
    };
    // Readback is 0xAABBGGRR — R in the low byte.
    let rgb = |p: u32| (p & 0xff, (p >> 8) & 0xff, (p >> 16) & 0xff);

    // Baseline: the dim baked floor (≈ half of R=0xff / G=0x80).
    let baked_fb = render(&mut renderer);
    let (br, bg, _) = rgb(baked_fb[centre]);
    assert!(
        (100..=160).contains(&br) && bg < 90,
        "baked floor should be dim orange: {:#010x}",
        baked_fb[centre]
    );

    // Map the floor colour to an opaque **emissive** material: the centre
    // must hit the exact CPU ladder value — emissive_shade(0xff8000, 255)
    // = (255, 255, 0) — ignoring the dim baked byte.
    let mut table = MaterialTable::new();
    table.set(1, Material::OPAQUE.with_emissive(255));
    renderer.set_terrain_materials(&table, &[(Rgb(0x00ff_8000), 1)]);
    let glow = render(&mut renderer)[centre];
    assert_eq!(
        rgb(glow),
        (255, 255, 0),
        "emissive must ignore the baked byte and match the CPU ladder: {glow:#010x}"
    );

    // A hostile rig (zero ambient, back-facing sun) blacks out a plain
    // floor but must not touch the emissive one.
    renderer.set_scene_lights(SceneLights {
        enabled: true,
        grid_sun_dirs: vec![[0.0, 0.0, 1.0]], // to-sun below: N·L = 0 on top
        sun_color: [1.0; 3],
        sun_intensity: 1.0,
        ambient: [0.0; 3],
        ..SceneLights::default()
    });
    let glow_lit = render(&mut renderer)[centre];
    assert_eq!(
        rgb(glow_lit),
        (255, 255, 0),
        "emissive must outrank the dynamic rig: {glow_lit:#010x}"
    );
    renderer.set_terrain_materials(&table, &[]); // gate off again
    let (dr, dg, db) = rgb(render(&mut renderer)[centre]);
    assert!(
        dr == 0 && dg == 0 && db == 0,
        "zero rig must black out the plain floor: ({dr},{dg},{db})"
    );

    // Byte-exactness gate: with the rig off and an empty map the whole
    // frame is identical to the pre-material baseline.
    renderer.set_scene_lights(SceneLights::default());
    assert_eq!(
        render(&mut renderer),
        baked_fb,
        "empty material map must re-render byte-identically"
    );
}

/// CA.3 — z-graded [`block_chunk`]: solid over `z ∈ [top, bot]`, each
/// voxel's stored BLUE byte = its z (R=0xff, G=0x00, brightness 0x80 →
/// exact colour passthrough), so a render pins EXACTLY which voxel
/// layer produced a pixel.
fn graded_block_chunk(vsid: u32, top: u8, bot: u8) -> Vxl {
    let n_cols = (vsid as usize) * (vsid as usize);
    let n_vox = (bot - top + 1) as usize;
    let mut data: Vec<u8> = Vec::with_capacity(n_cols * (4 + n_vox * 4));
    let mut column_offset: Vec<u32> = Vec::with_capacity(n_cols + 1);
    for _ in 0..n_cols {
        column_offset.push(u32::try_from(data.len()).expect("offset fits"));
        data.extend_from_slice(&[0, top, bot, 0]);
        for z in top..=bot {
            data.extend_from_slice(&[z, 0x00, 0xff, 0x80]); // BGRA, B = z
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

/// CA.3 — cutaway parity gate, primary rays: the per-grid clip lane
/// hides `z < z_clip` through the real `scene_dda.wgsl` pipeline, the
/// cut face shows EXACTLY the voxel layer at the plane (matching the
/// CPU sampler's stored-colour/run-top rule — the fixture stores a
/// z-graded colour per voxel, so any off-by-one or wrong-layer fetch
/// changes the byte), a fully-clipped grid is all sky, and a `None`
/// clip re-renders byte-identically to the unclipped baseline.
#[test]
fn scene_dda_cutaway_clips_and_pins_cut_face_colour() {
    let Some((gpu, _lock)) = try_init() else {
        return;
    };
    let vsid = 32u32;
    // Solid block z ∈ [100, 140], blue byte = z.
    let chunk = decompress_chunk(&graded_block_chunk(vsid, 100, 140));
    let grid = GridUpload {
        vsid,
        origin_chunk: [0, 0, 0],
        chunks_dims: [1, 1, 1],
        pool_dims: [1, 1, 1],
        chunks: vec![([0, 0, 0], chunk)],
    };
    let scene = GpuSceneResident::upload(&gpu.device, &SceneUpload { grids: vec![grid] });

    let (w, h) = (64u32, 64u32);
    let mut renderer = HeadlessSceneRenderer::new(&gpu.device, &gpu.queue, w, h);
    let cam = Camera {
        position: [16.0, 16.0, 50.0],
        right: [1.0, 0.0, 0.0],
        down: [0.0, 1.0, 0.0],
        forward: [0.0, 0.0, 1.0],
        fov_y_rad: 60f32.to_radians(),
    };
    let centre = (h / 2 * w + w / 2) as usize;
    let render = |r: &mut HeadlessSceneRenderer, clip: Option<i32>| {
        r.render_with_transforms(
            &gpu.device,
            &gpu.queue,
            &scene,
            &[cam],
            &[GridWorldTransform {
                z_clip: clip,
                ..GridWorldTransform::default()
            }],
            cam.fov_y_rad,
            64,
            0.0,
        )
    };
    // Readback is 0xAABBGGRR — R low byte, B bits 16..23.
    let rgb = |p: u32| (p & 0xff, (p >> 8) & 0xff, (p >> 16) & 0xff);

    // Unclipped: the top surface at z=100 (blue byte 100).
    let base = render(&mut renderer, None);
    assert_eq!(
        rgb(base[centre]),
        (255, 0, 100),
        "unclipped top face must be the z=100 layer: {:#010x}",
        base[centre]
    );
    // Clip mid-run: the cut face is EXACTLY the z=120 layer.
    let cut = render(&mut renderer, Some(120));
    assert_eq!(
        rgb(cut[centre]),
        (255, 0, 120),
        "cut face must be the z=120 layer: {:#010x}",
        cut[centre]
    );
    // Pixel classification: clipping only REMOVES geometry, so every
    // block pixel of the cut render must be a block pixel of the base
    // render too (edge rays that grazed the block's side above the
    // plane legitimately become sky — the reverse never happens).
    for (i, (&b, &c)) in base.iter().zip(cut.iter()).enumerate() {
        assert!(
            (b & 0xff) > 180 || (c & 0xff) <= 180,
            "pixel {i} was sky and became block: base={b:#010x} cut={c:#010x}"
        );
    }
    // Clip just past the block's bottom: voxlap columns are BEDROCK
    // below the last slab, so the cut face is the (colourless) bedrock
    // layer at z=141 — the colour fetch falls back to the nearest
    // stored colour, the run's z=140 byte. Pins the interior-fallback
    // rule the CPU sampler uses (`surface_color_mip` run-top/bottom).
    let bedrock = render(&mut renderer, Some(141));
    assert_eq!(
        rgb(bedrock[centre]),
        (255, 0, 140),
        "bedrock cut face must fall back to the run's stored colour: {:#010x}",
        bedrock[centre]
    );
    // Clip past the chunk's full depth: nothing left — all sky.
    let gone = render(&mut renderer, Some(256));
    assert!(
        gone.iter().all(|&p| (p & 0xff) <= 180),
        "clip=256 must hide the entire chunk"
    );
    // Standing gate: clip=None is byte-identical to the baseline.
    let none_again = render(&mut renderer, None);
    assert_eq!(
        none_again, base,
        "z_clip=None must re-render byte-identically"
    );
}

/// CA.3 — the GPU clip uses the CPU's exact `ceil(z_clip / 2^mip)` (round
/// UP) formula. Two 1-voxel plates with real air between them (P at
/// z=100, Q at z=140) and `z_clip = 101`: at mip 0 plate P (z=100 < 101)
/// is hidden and the ray reaches Q; at ANY coarse mip the round-up plane
/// keeps P's straddling cell HIDDEN too (`ceil(101/2)=51 > 50`), so Q
/// wins at BOTH mips — no boundary-layer leak (the plain floor let P poke
/// through as a ring past the mip-0 radius, `GPU_ZCLIP_MIP_BUG`). The
/// plates' blue bytes (100 vs 140) name the winning layer exactly.
#[test]
fn scene_dda_cutaway_mip_formula_ceils() {
    use roxlap_formats::color::VoxColor;

    let Some((gpu, _lock)) = try_init() else {
        return;
    };
    let vsid = 32u32;
    // B = z for each plate (BGRA upload order is handled by from_dense).
    let vxl = Vxl::from_dense(vsid, |_, _, z| match z {
        100 => Some(VoxColor(0x80ff_0064)), // P: R=0xff, B=100
        140 => Some(VoxColor(0x80ff_008c)), // Q: R=0xff, B=140
        _ => None,
    });
    let grid = GridUpload {
        vsid,
        origin_chunk: [0, 0, 0],
        chunks_dims: [1, 1, 1],
        pool_dims: [1, 1, 1],
        chunks: vec![([0, 0, 0], decompress_chunk(&vxl))],
    };
    let scene = GpuSceneResident::upload(&gpu.device, &SceneUpload { grids: vec![grid] });

    let (w, h) = (64u32, 64u32);
    let renderer = HeadlessSceneRenderer::new(&gpu.device, &gpu.queue, w, h);
    let centre = (h / 2 * w + w / 2) as usize;
    let xf = GridWorldTransform {
        z_clip: Some(101),
        ..GridWorldTransform::default()
    };
    // Camera OUTSIDE the chunk (z = -200), so the chunk is entered at
    // t ≈ 200 and `mip_scan_dist` alone dictates the marched mip.
    let cam = Camera {
        position: [16.0, 16.0, -200.0],
        right: [1.0, 0.0, 0.0],
        down: [0.0, 1.0, 0.0],
        forward: [0.0, 0.0, 1.0],
        fov_y_rad: 20f32.to_radians(),
    };
    let render_at = |mip_scan_dist: f32| {
        renderer.render_with_transforms(
            &gpu.device,
            &gpu.queue,
            &scene,
            &[cam],
            &[xf],
            cam.fov_y_rad,
            64,
            mip_scan_dist,
        )[centre]
    };
    let blue = |p: u32| (p >> 16) & 0xff;
    // LOD off → mip 0 → P hidden, the ray lands on Q (blue 140).
    let p_mip0 = render_at(0.0);
    assert_eq!(
        blue(p_mip0),
        140,
        "mip 0: clip=101 must hide plate P and hit Q: {p_mip0:#010x}"
    );
    // mip ≥ 1 (t_enter ≈ 200 ≥ 2·mip_scan_dist): the round-up plane keeps
    // P's straddling cell HIDDEN (no leak), so the ray still reaches Q —
    // blue 140, the SAME as mip 0.
    let p_coarse = render_at(100.0);
    assert_eq!(
        blue(p_coarse),
        140,
        "coarse mip: ceil formula must still hide plate P and hit Q: {p_coarse:#010x}"
    );
}

/// CA.3 — cutaway shadow parity: a clipped-away wall stops casting sun
/// shadow on the floor next to it (the shadow march applies the same
/// per-grid clip as the primary rays — "world as if removed").
#[test]
fn scene_dda_cutaway_hidden_wall_stops_shadowing() {
    let Some((gpu, _lock)) = try_init() else {
        return;
    };
    let vsid = 32u32;
    // Floor at z=100 + wall x ∈ [16,18) rising z ∈ [90, 100].
    let chunk = decompress_chunk(&floor_with_wall_chunk(vsid, 16, 18, 90));
    let grid = GridUpload {
        vsid,
        origin_chunk: [0, 0, 0],
        chunks_dims: [1, 1, 1],
        pool_dims: [1, 1, 1],
        chunks: vec![([0, 0, 0], chunk)],
    };
    let scene = GpuSceneResident::upload(&gpu.device, &SceneUpload { grids: vec![grid] });

    let (w, h) = (64u32, 64u32);
    let mut renderer = HeadlessSceneRenderer::new(&gpu.device, &gpu.queue, w, h);
    let cam = Camera {
        position: [14.0, 16.0, 50.0],
        right: [1.0, 0.0, 0.0],
        down: [0.0, 1.0, 0.0],
        forward: [0.0, 0.0, 1.0],
        fov_y_rad: 60f32.to_radians(),
    };
    let centre = (h / 2 * w + w / 2) as usize;
    let lum = |p: u32| (p & 0xff) + ((p >> 8) & 0xff) + ((p >> 16) & 0xff);
    let render = |r: &mut HeadlessSceneRenderer, clip: Option<i32>| {
        r.render_with_transforms(
            &gpu.device,
            &gpu.queue,
            &scene,
            &[cam],
            &[GridWorldTransform {
                z_clip: clip,
                ..GridWorldTransform::default()
            }],
            cam.fov_y_rad,
            64,
            0.0,
        )[centre]
    };
    // Sun toward +x and up: the wall at x ∈ [16,18) occludes the floor
    // point (14,16,100) — the to-sun ray crosses it at z ≈ 96..94.
    let s = std::f32::consts::FRAC_1_SQRT_2;
    renderer.set_scene_lights(SceneLights {
        enabled: true,
        grid_sun_dirs: vec![[s, 0.0, -s]],
        sun_color: [1.0; 3],
        sun_intensity: 3.0,
        sun_casts_shadow: true,
        ambient: [0.5; 3],
        shadow_strength: 1.0,
        shadow_bias: 1.5,
        shadow_max_dist: 512.0,
        shadow_max_steps: 256,
        ..SceneLights::default()
    });
    let shadowed = render(&mut renderer, None);
    // Clip at z=100: the wall body (z 90..99) vanishes, the floor layer
    // itself (z=100) stays visible AND sun-lit.
    let unshadowed = render(&mut renderer, Some(100));
    let blue = |p: u32| (p >> 16) & 0xff;
    assert!(
        blue(shadowed) < 70 && blue(unshadowed) < 70,
        "both renders must show the floor, not sky: {shadowed:#010x} / {unshadowed:#010x}"
    );
    assert!(
        lum(unshadowed) > lum(shadowed),
        "a clipped-away wall must stop shadowing: shadowed {shadowed:#010x} -> clipped {unshadowed:#010x}",
    );
}

/// CA follow-up — cross-grid sun shadow under a TELE camera (the deck
/// view): a hull grid must darken the ground grid beside it even when
/// the receiver sits ~1150 world units from the eye. Regression for
/// the Decks report "GPU shows no hull shadow" — the CPU backend
/// renders it, so a miss here is a backend divergence.
#[test]
fn scene_dda_cross_grid_shadow_survives_tele_distance() {
    use roxlap_formats::color::VoxColor;

    let Some((gpu, _lock)) = try_init() else {
        return;
    };
    let vsid = 128u32;
    // Ground grid: plate z ∈ [420, 440] (deep in chunk z=1 → the
    // upload uses chunk (0,0,1) like the demo).
    let ground_vxl = Vxl::from_dense(vsid, |_, _, z| {
        (164..=184).contains(&z).then_some(VoxColor(0x80_4A_5E_3C))
    });
    // The demo's ground plate lives at z 420..440 = chunk 1, local 164..184.
    let ground = GridUpload {
        vsid,
        origin_chunk: [0, 0, 0],
        chunks_dims: [2, 1, 2],
        pool_dims: [2, 1, 2],
        chunks: vec![
            ([0, 0, 1], decompress_chunk(&ground_vxl)),
            ([1, 0, 1], decompress_chunk(&ground_vxl)),
        ],
    };
    // Hull grid: a HOLLOW box (roof plate + perimeter walls + deck
    // floors, like the demo shiplet) x 24..104, y 32..96, z 240..313,
    // split over stacked chunks (0,0,0) + (0,0,1), world origin z=106.
    let hull_at = |x: u32, y: u32, z: i32| -> bool {
        let inside = (24..104).contains(&x) && (32..96).contains(&y) && (240..=313).contains(&z);
        if !inside {
            return false;
        }
        let roof = z <= 241;
        let wall = x <= 25 || x >= 102 || y <= 33 || y >= 94;
        let floor = matches!(z, 259..=262 | 283..=286 | 307..=310 | 311..=313);
        roof || wall || floor
    };
    let hull_c0 = Vxl::from_dense(vsid, |x, y, z| {
        hull_at(x, y, i32::try_from(z).unwrap()).then_some(VoxColor(0x80_62_66_70))
    });
    let hull_c1 = Vxl::from_dense(vsid, |x, y, z| {
        hull_at(x, y, i32::try_from(z).unwrap() + 256).then_some(VoxColor(0x80_62_66_70))
    });
    let ship = GridUpload {
        vsid,
        origin_chunk: [0, 0, 0],
        chunks_dims: [1, 1, 2],
        pool_dims: [1, 1, 2],
        chunks: vec![
            ([0, 0, 0], decompress_chunk(&hull_c0)),
            ([0, 0, 1], decompress_chunk(&hull_c1)),
        ],
    };
    let scene = GpuSceneResident::upload(
        &gpu.device,
        &SceneUpload {
            grids: vec![ground, ship],
        },
    );

    let (w, h) = (128u32, 128u32);
    let renderer = HeadlessSceneRenderer::new(&gpu.device, &gpu.queue, w, h);
    // Tele camera: straight down from ~1150 world units, framing the
    // shadow band EAST of the hull (sun travel [0.45, 0.35, 0.82] ⇒
    // the hull's shadow falls on its +x/+y side, extending ~40 voxels
    // past the wall from the 74-voxel drop to the ground).
    let cam = Camera {
        position: [120.0, 70.0, 420.0 - 1150.0],
        right: [1.0, 0.0, 0.0],
        down: [0.0, 1.0, 0.0],
        forward: [0.0, 0.0, 1.0],
        fov_y_rad: 0.15,
    };
    let cam_ship = Camera {
        position: [cam.position[0], cam.position[1], cam.position[2] - 106.0],
        ..cam
    };
    let xf_ground = GridWorldTransform::default();
    let xf_ship = GridWorldTransform {
        origin: [0.0, 0.0, 106.0],
        ..GridWorldTransform::default()
    };
    let mut r = renderer;
    let render = |r: &mut HeadlessSceneRenderer| {
        r.render_with_transforms(
            &gpu.device,
            &gpu.queue,
            &scene,
            &[cam, cam_ship],
            &[xf_ground, xf_ship],
            cam.fov_y_rad,
            64,
            640.0, // the demo's tele LOD override
        )
    };
    let s = |casts: bool| SceneLights {
        enabled: true,
        grid_sun_dirs: vec![[-0.45, -0.35, -0.82], [-0.45, -0.35, -0.82]],
        sun_color: [1.0, 0.95, 0.85],
        sun_intensity: 1.1,
        sun_casts_shadow: casts,
        ambient: [0.4, 0.42, 0.48],
        shadow_strength: 0.8,
        shadow_bias: 1.5,
        shadow_max_dist: 200.0,
        shadow_max_steps: 768,
        ..SceneLights::default()
    };
    let lum = |p: u32| (p & 0xff) + ((p >> 8) & 0xff) + ((p >> 16) & 0xff);
    let sum = |fb: &[u32]| fb.iter().map(|&p| u64::from(lum(p))).sum::<u64>();
    r.set_scene_lights(s(false));
    let lit = render(&mut r);
    r.set_scene_lights(s(true));
    let shadowed = render(&mut r);
    // Compare total luminance over the frame: with the hull shadow
    // present a visible patch of ground darkens.
    let (l0, l1) = (sum(&lit), sum(&shadowed));
    assert!(
        l1 < l0 && (l0 - l1) * 100 > l0,
        "hull must cast a visible cross-grid shadow at tele distance: \
         lit {l0} shadowed {l1}"
    );
}

/// OC.2 — keyhole parity gate, primary rays: the uniform cone +
/// per-grid focus-plane lane cut the front wall through the real
/// `scene_dda.wgsl` pipeline exactly like the CPU keyhole — cone
/// centre revealed, outside-cone column intact, cells at/below the
/// focus plane intact — and both "off" encodings re-render
/// byte-identically.
#[test]
fn scene_dda_cutout_reveals_wall_inside_window_only() {
    use roxlap_formats::color::VoxColor;

    let Some((gpu, _lock)) = try_init() else {
        return;
    };
    let vsid = 32u32;
    // Front wall at y=8, room back wall at y=24, both spanning x/z.
    const WALL: (u32, u32, u32) = (0xC0, 0x40, 0x40);
    const BACK: (u32, u32, u32) = (0x40, 0xC0, 0x40);
    let vxl = Vxl::from_dense(vsid, |_, y, _| match y {
        8 => Some(VoxColor(0x80_C0_40_40)),
        24 => Some(VoxColor(0x80_40_C0_40)),
        _ => None,
    });
    let grid = GridUpload {
        vsid,
        origin_chunk: [0, 0, 0],
        chunks_dims: [1, 1, 1],
        pool_dims: [1, 1, 1],
        chunks: vec![([0, 0, 0], decompress_chunk(&vxl))],
    };
    let scene = GpuSceneResident::upload(&gpu.device, &SceneUpload { grids: vec![grid] });

    let (w, h) = (64u32, 64u32);
    let mut renderer = HeadlessSceneRenderer::new(&gpu.device, &gpu.queue, w, h);
    // Camera at y=2 looking +y (right × down == forward): wall at
    // world-t ≈ 6, back wall at ≈ 22, centre ray at grid z = 16.
    let cam = Camera {
        position: [16.0, 2.0, 16.0],
        right: [-1.0, 0.0, 0.0],
        down: [0.0, 0.0, 1.0],
        forward: [0.0, 1.0, 0.0],
        fov_y_rad: 60f32.to_radians(),
    };
    let centre = (h / 2 * w + w / 2) as usize;
    let outside = (h / 2 * w + 4) as usize; // 28.5 px from the centre
    let render = |r: &mut HeadlessSceneRenderer, focus_z: i32| {
        r.render_with_transforms(
            &gpu.device,
            &gpu.queue,
            &scene,
            &[cam],
            &[GridWorldTransform {
                cutout_focus_z: focus_z,
                cutout_focus_local: [16.0, 25.0, 16.0],
                ..GridWorldTransform::default()
            }],
            cam.fov_y_rad,
            64,
            0.0,
        )
    };
    let rgb = |p: u32| (p & 0xff, (p >> 8) & 0xff, (p >> 16) & 0xff);

    // Baseline: no cutout → the front wall everywhere.
    let base = render(&mut renderer, i32::MIN);
    assert_eq!(
        rgb(base[centre]),
        WALL,
        "base centre: {:#010x}",
        base[centre]
    );

    // Cone down the +y view axis, hard edge, reveal past the wall
    // (cell dist ≈ 6.5) but short of the back wall (≈ 22.5); focus
    // plane below the centre ray's grid z (16 < 100).
    renderer.set_view_cutout(Some(GpuViewCutout {
        tan_outer: 0.2,
        tan_inner: 0.2,
        margin: 2.0,
    }));
    let cut = render(&mut renderer, 100);
    assert_eq!(
        rgb(cut[centre]),
        BACK,
        "cone centre must see through the wall: {:#010x}",
        cut[centre]
    );
    assert_eq!(
        rgb(cut[outside]),
        WALL,
        "outside the cone the wall must stay: {:#010x}",
        cut[outside]
    );
    // Focus plane above the centre ray's z (16 ≮ 10): the wall stays
    // even inside the cone — the floor-in-front rule.
    let below = render(&mut renderer, 10);
    assert_eq!(
        rgb(below[centre]),
        WALL,
        "cells at/below the focus plane must stay: {:#010x}",
        below[centre]
    );
    // Off encodings: a cleared cutout AND a margin larger than the
    // scene are both byte-identical to the baseline (decision 8).
    renderer.set_view_cutout(None);
    assert_eq!(
        render(&mut renderer, i32::MIN),
        base,
        "cleared cutout must be byte-identical"
    );
    renderer.set_view_cutout(Some(GpuViewCutout {
        tan_outer: 10.0,
        tan_inner: 10.0,
        margin: 1.0e6,
    }));
    assert_eq!(
        render(&mut renderer, 100),
        base,
        "a scene-sized margin must be byte-identical"
    );
    renderer.set_view_cutout(None);
}

/// OC.2 — the GPU feather tapers the reveal distance across the cone
/// band by the SAME per-cell rule the CPU test pins (visual-pass
/// redesign — no dither): the cut edge is spatially coherent (the
/// axis row's revealed pixels form one contiguous run — no teeth),
/// deterministic frame-to-frame, and strictly narrower than the same
/// cone with a hard edge.
#[test]
fn scene_dda_cutout_feather_tapers_reveal_radially() {
    use roxlap_formats::color::VoxColor;

    let Some((gpu, _lock)) = try_init() else {
        return;
    };
    let vsid = 32u32;
    let vxl = Vxl::from_dense(vsid, |_, y, _| match y {
        8 => Some(VoxColor(0x80_C0_40_40)),
        24 => Some(VoxColor(0x80_40_C0_40)),
        _ => None,
    });
    let grid = GridUpload {
        vsid,
        origin_chunk: [0, 0, 0],
        chunks_dims: [1, 1, 1],
        pool_dims: [1, 1, 1],
        chunks: vec![([0, 0, 0], decompress_chunk(&vxl))],
    };
    let scene = GpuSceneResident::upload(&gpu.device, &SceneUpload { grids: vec![grid] });

    let (w, h) = (64u32, 64u32);
    let mut renderer = HeadlessSceneRenderer::new(&gpu.device, &gpu.queue, w, h);
    let cam = Camera {
        position: [16.0, 2.0, 16.0],
        right: [-1.0, 0.0, 0.0],
        down: [0.0, 0.0, 1.0],
        forward: [0.0, 1.0, 0.0],
        fov_y_rad: 60f32.to_radians(),
    };
    let tapered = GpuViewCutout {
        tan_outer: 0.5,
        tan_inner: 0.1,
        margin: 2.0,
    };
    let xf = [GridWorldTransform {
        cutout_focus_local: [16.0, 25.0, 16.0],
        cutout_focus_z: 100,
        ..GridWorldTransform::default()
    }];
    let render = |r: &mut HeadlessSceneRenderer| {
        r.render_with_transforms(
            &gpu.device,
            &gpu.queue,
            &scene,
            &[cam],
            &xf,
            cam.fov_y_rad,
            64,
            0.0,
        )
    };
    renderer.set_view_cutout(Some(tapered));
    let fb = render(&mut renderer);
    let fb2 = render(&mut renderer);
    assert_eq!(fb, fb2, "the feather taper must be deterministic");
    const BACK: (u32, u32, u32) = (0x40, 0xC0, 0x40);
    let rgb = |p: u32| (p & 0xff, (p >> 8) & 0xff, (p >> 16) & 0xff);
    // Per-row coherence on the axis row: BACK pixels form ONE
    // contiguous run (whole-cell classification — no teeth).
    let row = (h / 2) as usize * w as usize;
    let flags: Vec<bool> = (0..w as usize)
        .map(|px| rgb(fb[row + px]) == BACK)
        .collect();
    let first = flags.iter().position(|&b| b);
    let last = flags.iter().rposition(|&b| b);
    let (Some(first), Some(last)) = (first, last) else {
        panic!("axis row must contain revealed pixels");
    };
    assert!(
        flags[first..=last].iter().all(|&b| b),
        "revealed run must be contiguous (no teeth): {flags:?}"
    );
    assert!(
        flags.iter().any(|&b| !b),
        "the taper must keep wall pixels on the axis row"
    );
    // The taper cuts a strictly SMALLER hole than a hard edge at the
    // same outer cone.
    renderer.set_view_cutout(Some(GpuViewCutout {
        tan_inner: 0.5,
        ..tapered
    }));
    let fb_hard = render(&mut renderer);
    let count = |fb: &[u32]| fb.iter().filter(|&&p| rgb(p) == BACK).count();
    let (n_taper, n_hard) = (count(&fb), count(&fb_hard));
    assert!(
        0 < n_taper && n_taper < n_hard,
        "taper must shrink the hole: tapered {n_taper} vs hard {n_hard}"
    );
}

/// OC.2 — cut faces through the keyhole use the stored-colour /
/// run-top fallback (decision 4): cutting mid-run into a z-graded
/// block shows EXACTLY the voxel layer at the focus plane.
#[test]
fn scene_dda_cutout_cut_face_pins_layer_colour() {
    let Some((gpu, _lock)) = try_init() else {
        return;
    };
    let vsid = 32u32;
    let chunk = decompress_chunk(&graded_block_chunk(vsid, 100, 140));
    let grid = GridUpload {
        vsid,
        origin_chunk: [0, 0, 0],
        chunks_dims: [1, 1, 1],
        pool_dims: [1, 1, 1],
        chunks: vec![([0, 0, 0], chunk)],
    };
    let scene = GpuSceneResident::upload(&gpu.device, &SceneUpload { grids: vec![grid] });

    let (w, h) = (64u32, 64u32);
    let mut renderer = HeadlessSceneRenderer::new(&gpu.device, &gpu.queue, w, h);
    let cam = Camera {
        position: [16.0, 16.0, 50.0],
        right: [1.0, 0.0, 0.0],
        down: [0.0, 1.0, 0.0],
        forward: [0.0, 0.0, 1.0],
        fov_y_rad: 60f32.to_radians(),
    };
    let centre = (h / 2 * w + w / 2) as usize;
    renderer.set_view_cutout(Some(GpuViewCutout {
        tan_outer: 4.0,
        tan_inner: 4.0,
        margin: 0.0,
    }));
    let fb = renderer.render_with_transforms(
        &gpu.device,
        &gpu.queue,
        &scene,
        &[cam],
        &[GridWorldTransform {
            cutout_focus_local: [16.0, 16.0, 130.0],
            cutout_focus_z: 120,
            ..GridWorldTransform::default()
        }],
        cam.fov_y_rad,
        64,
        0.0,
    );
    let rgb = |p: u32| (p & 0xff, (p >> 8) & 0xff, (p >> 16) & 0xff);
    assert_eq!(
        rgb(fb[centre]),
        (255, 0, 120),
        "keyhole cut face must be the z=120 layer: {:#010x}",
        fb[centre]
    );
}

/// OC.2 — the GPU keyhole uses the CPU's exact `focus_z >> mip` FLOOR
/// formula (the CA.3 mip gate, cutout edition): plates P (z=100) and
/// Q (z=140) with the focus plane at 101 — mip 0 hides P and lands on
/// Q; a coarse mip floors the plane onto P's cell so P pokes through.
#[test]
fn scene_dda_cutout_mip_formula_floors() {
    use roxlap_formats::color::VoxColor;

    let Some((gpu, _lock)) = try_init() else {
        return;
    };
    let vsid = 32u32;
    let vxl = Vxl::from_dense(vsid, |_, _, z| match z {
        100 => Some(VoxColor(0x80ff_0064)), // P: R=0xff, B=100
        140 => Some(VoxColor(0x80ff_008c)), // Q: R=0xff, B=140
        _ => None,
    });
    let grid = GridUpload {
        vsid,
        origin_chunk: [0, 0, 0],
        chunks_dims: [1, 1, 1],
        pool_dims: [1, 1, 1],
        chunks: vec![([0, 0, 0], decompress_chunk(&vxl))],
    };
    let scene = GpuSceneResident::upload(&gpu.device, &SceneUpload { grids: vec![grid] });

    let (w, h) = (64u32, 64u32);
    let mut renderer = HeadlessSceneRenderer::new(&gpu.device, &gpu.queue, w, h);
    let centre = (h / 2 * w + w / 2) as usize;
    renderer.set_view_cutout(Some(GpuViewCutout {
        tan_outer: 1.0,
        tan_inner: 1.0,
        margin: 0.0,
    }));
    let xf = GridWorldTransform {
        cutout_focus_local: [16.0, 16.0, 120.0],
        cutout_focus_z: 101,
        ..GridWorldTransform::default()
    };
    // Camera OUTSIDE the chunk (t_enter ≈ 200) so `mip_scan_dist`
    // alone dictates the marched mip.
    let cam = Camera {
        position: [16.0, 16.0, -200.0],
        right: [1.0, 0.0, 0.0],
        down: [0.0, 1.0, 0.0],
        forward: [0.0, 0.0, 1.0],
        fov_y_rad: 20f32.to_radians(),
    };
    let render_at = |r: &mut HeadlessSceneRenderer, mip_scan_dist: f32| {
        r.render_with_transforms(
            &gpu.device,
            &gpu.queue,
            &scene,
            &[cam],
            &[xf],
            cam.fov_y_rad,
            64,
            mip_scan_dist,
        )[centre]
    };
    let blue = |p: u32| (p >> 16) & 0xff;
    // LOD off → mip 0 → P hidden by the keyhole, the ray lands on Q.
    let p_mip0 = render_at(&mut renderer, 0.0);
    assert_eq!(
        blue(p_mip0),
        140,
        "mip 0: focus plane 101 must hide plate P and hit Q: {p_mip0:#010x}"
    );
    // mip ≥ 1: the floored plane exposes P's cell (100 >> m == 101 >> m).
    let p_coarse = render_at(&mut renderer, 100.0);
    assert!(
        blue(p_coarse) < 120,
        "coarse mip: floor formula must expose plate P: {p_coarse:#010x}"
    );
}

/// OC.2 — the keyhole is a VIEW aid (entry-doc non-goal): a wall
/// hidden by the cutout still casts its sun shadow (the shadow march
/// never sees the cutout) — the exact opposite of the CA clip's
/// "world as if removed" gate, pinned side-by-side against it.
#[test]
fn scene_dda_cutout_hidden_wall_still_shadows() {
    let Some((gpu, _lock)) = try_init() else {
        return;
    };
    let vsid = 32u32;
    // Floor at z=100 + wall x ∈ [16,18) rising z ∈ [90, 100).
    let chunk = decompress_chunk(&floor_with_wall_chunk(vsid, 16, 18, 90));
    let grid = GridUpload {
        vsid,
        origin_chunk: [0, 0, 0],
        chunks_dims: [1, 1, 1],
        pool_dims: [1, 1, 1],
        chunks: vec![([0, 0, 0], chunk)],
    };
    let scene = GpuSceneResident::upload(&gpu.device, &SceneUpload { grids: vec![grid] });

    let (w, h) = (64u32, 64u32);
    let mut renderer = HeadlessSceneRenderer::new(&gpu.device, &gpu.queue, w, h);
    // Shallow shoulder view from −x: the wall stands BETWEEN the eye
    // and the character column at (20, 16); the centre ray crosses it
    // at z ≈ 92 and, once the keyhole melts it, lands on the floor
    // beyond at x ≈ 24 — inside the wall's sun shadow.
    let (fx, fz) = (0.747_41_f32, 0.664_36_f32); // normalize(18, 0, 16)
    let cam = Camera {
        position: [2.0, 16.0, 80.0],
        right: [fz, 0.0, -fx],
        down: [0.0, 1.0, 0.0],
        forward: [fx, 0.0, fz],
        fov_y_rad: 60f32.to_radians(),
    };
    let centre = (h / 2 * w + w / 2) as usize;
    let lum = |p: u32| (p & 0xff) + ((p >> 8) & 0xff) + ((p >> 16) & 0xff);
    let render = |r: &mut HeadlessSceneRenderer, clip: Option<i32>, focus_z: i32| {
        r.render_with_transforms(
            &gpu.device,
            &gpu.queue,
            &scene,
            &[cam],
            &[GridWorldTransform {
                z_clip: clip,
                cutout_focus_z: focus_z,
                cutout_focus_local: [20.0, 16.0, 96.0],
                ..GridWorldTransform::default()
            }],
            cam.fov_y_rad,
            64,
            0.0,
        )[centre]
    };
    // Sun toward −x and up: the wall shadows the floor strip BEYOND
    // it (x ≳ 18) — exactly what the cut ray lands on.
    let s = std::f32::consts::FRAC_1_SQRT_2;
    renderer.set_scene_lights(SceneLights {
        enabled: true,
        grid_sun_dirs: vec![[-s, 0.0, -s]],
        sun_color: [1.0; 3],
        sun_intensity: 3.0,
        sun_casts_shadow: true,
        ambient: [0.5; 3],
        shadow_strength: 1.0,
        shadow_bias: 1.5,
        shadow_max_dist: 512.0,
        shadow_max_steps: 256,
        ..SceneLights::default()
    });
    // Uncut baseline: the centre pixel is the (sun-lit) wall face.
    let base = render(&mut renderer, None, i32::MIN);
    // Keyhole around the column behind the wall: the wall melts, the
    // revealed floor beyond stays in the hidden wall's shadow.
    renderer.set_view_cutout(Some(GpuViewCutout {
        tan_outer: 10.0,
        tan_inner: 10.0,
        margin: 1.0,
    }));
    let cut = render(&mut renderer, None, 100);
    assert_ne!(cut, base, "the keyhole must melt the wall at the centre");
    // Contrast: the CA clip REMOVES the wall from the world — the same
    // floor point brightens (no occluder left to shadow it).
    renderer.set_view_cutout(None);
    let clipped = render(&mut renderer, Some(100), i32::MIN);
    assert!(
        lum(clipped) > lum(cut),
        "world-removal must unshadow what the view cutout keeps dark: \
         cut {cut:#010x} clipped {clipped:#010x}"
    );
}
