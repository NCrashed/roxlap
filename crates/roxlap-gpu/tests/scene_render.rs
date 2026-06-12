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
