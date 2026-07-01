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
    GridUpload, GridWorldTransform, HeadlessGpu, HeadlessSceneRenderer, SceneLights, SceneUpload,
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
