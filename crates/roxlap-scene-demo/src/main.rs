//! roxlap-scene-demo — interactive showcase of the scene-graph
//! engine. See `README.md` for the controls + the demo's
//! evolution roadmap as the scene-graph substages land.

mod collision;
mod kv6_sprite;
mod markers;
#[cfg(test)]
mod repro;
mod scene;
// DS.0 — demo-scene API + scenes (host wiring lands in DS.1).
mod scene_api;
mod scenes;
mod ship;
mod terrain;

use std::sync::Arc;
use std::time::Instant;

use roxlap_core::opticast::OpticastSettings;
use roxlap_core::Engine;
use roxlap_formats::character::{self, Attachment, Bone, Character, Clip, ClipData};
use roxlap_formats::kfa::{Hinge, Point3, Seq};
use roxlap_formats::kv6::Kv6;
use roxlap_formats::sprite::Sprite;
use roxlap_formats::xform::BoneXform;
use roxlap_render::{
    CharacterId, DynSpriteTransform, FrameParams, ImageFacing, ImageId, ImageSprite, KfaSprite,
    Line3, LoopMode, RenderOptions, SceneRenderer, SpriteInstanceDesc, SpriteInstanceId,
    SpriteModelId, SpriteSet, VoxelClip, VoxelFrame,
};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{DeviceEvent, DeviceId, ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

use crate::scene::{build_demo, SceneAndCamera, StreamingBakeTracker};

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;
/// Headroom for the per-frame [`ScratchPool`] sizing — `lastx`
/// inside each pool slot is sized `max(yres, vsid)`. The S4.0 demo
/// ships with a 2-chunk-wide ground (combined `vsid = 256`); pre-
/// allocating for 32×32 chunks keeps later demo expansions
/// allocation-free.
const MAX_GRID_VSID: u32 = 32 * roxlap_scene::CHUNK_SIZE_XY;

/// Initial max ray-march distance for the per-frame opticast pass.
/// User can adjust at runtime via `+` / `-` (range
/// [`SCAN_DIST_MIN`, `SCAN_DIST_MAX`]). Multi-mip absorbs the cost
/// of larger distances by transitioning distant rays to coarser
/// chunk LODs — at 384+ the mip-2 voxels dominate the budget while
/// mip-0 stays sharp near the camera.
///
/// AAMB (axis-aligned-mip-beams) was the cap rationale — kept
/// `SCAN_DIST_MAX` at 1500 to push the slider below the beam
/// threshold. The VC/CB/PRR cascade incidentally resolved the
/// beam bug (multi-chunk beam tests report 0 pixels across every
/// msd config at ml=6). Cap reverted to 1024 here as part of the
/// AAMB cleanup; the full 6-mip ladder is now safe at the
/// original config.
const SCAN_DIST_INITIAL: i32 = 384;
const SCAN_DIST_MIN: i32 = 64;
const SCAN_DIST_MAX: i32 = 1024;
const SCAN_DIST_STEP: i32 = 64;

/// Cap for `rayon`'s strip-parallel pool. Voxlap's per-strip
/// projection re-derivation adds fixed overhead that amortises
/// poorly past ~4 strips for an 800×600 frame; bench shows >4
/// threads slows down (per-strip overhead > work). Set high
/// enough to use modest multicore boost without going past the
/// efficiency knee.
const RENDER_THREADS: usize = 4;

const MOVE_SPEED: f64 = 64.0;

/// Embedded panoramic sky texture for the textured-`startsky`
/// path. Whatever PNG the user has dropped in `assets/sky.png` is
/// baked into the binary at build time. Width maps to elevation
/// (horizon → zenith), height to azimuth (wrap-around). Same asset
/// the roxlap-host demo ships.
const SKY_PNG: &[u8] = include_bytes!("../../../assets/sky.png");

/// Re-export of [`SKY_PNG`] under a stable name for the
/// `#[cfg(test)]` `repro` module to load the demo's sky panorama
/// without duplicating the bytes include.
#[cfg(test)]
pub(crate) const SKY_PNG_BYTES: &[u8] = SKY_PNG;

/// Decode a PNG byte slice into a `roxlap_core::sky::Sky`.
///
/// Voxlap's sky-mapping convention: **texture width = elevation
/// gradient (horizon → zenith)**, **texture height = azimuth wrap
/// (360° around the camera)**. Standard equirectangular panoramas
/// are usually laid out the other way (width=azimuth,
/// height=elevation), so `Sky::from_pixels` re-interprets the
/// dimensions accordingly. Mirror of roxlap-host's helper.
/// GPU.8 helper — decode `SKY_PNG` to a raw RGBA byte buffer
/// (`width * height * 4`). The GPU sky binding wants pixels in
/// equirectangular layout, which is exactly what the PNG already
/// is; the host of `load_png_sky` re-interprets the CPU side but
/// the GPU samples the original bytes directly.
pub(crate) fn load_png_sky_rgba(png_bytes: &[u8]) -> Result<(Vec<u8>, u32, u32), String> {
    let decoder = png::Decoder::new(png_bytes);
    let mut reader = decoder
        .read_info()
        .map_err(|e| format!("png header: {e}"))?;
    let info = reader.info();
    let width = info.width;
    let height = info.height;
    let (bytes_per_pixel, has_alpha) = match info.color_type {
        png::ColorType::Rgb => (3, false),
        png::ColorType::Rgba => (4, true),
        ct => return Err(format!("unsupported colour type {ct:?}; want RGB or RGBA")),
    };
    if info.bit_depth != png::BitDepth::Eight {
        return Err(format!(
            "unsupported bit depth {:?}; want 8-bit",
            info.bit_depth
        ));
    }
    let mut src = vec![0u8; reader.output_buffer_size()];
    reader
        .next_frame(&mut src)
        .map_err(|e| format!("png frame: {e}"))?;
    let mut rgba = Vec::with_capacity((width as usize) * (height as usize) * 4);
    for chunk in src.chunks_exact(bytes_per_pixel) {
        rgba.push(chunk[0]);
        rgba.push(chunk[1]);
        rgba.push(chunk[2]);
        rgba.push(if has_alpha { chunk[3] } else { 0xff });
    }
    Ok((rgba, width, height))
}

pub(crate) fn load_png_sky(png_bytes: &[u8]) -> Result<roxlap_core::sky::Sky, String> {
    let decoder = png::Decoder::new(png_bytes);
    let mut reader = decoder
        .read_info()
        .map_err(|e| format!("png header: {e}"))?;
    let info = reader.info();
    let png_w = info.width;
    let png_h = info.height;
    let bytes_per_pixel = match info.color_type {
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        ct => return Err(format!("unsupported colour type {ct:?}; want RGB or RGBA")),
    };
    if info.bit_depth != png::BitDepth::Eight {
        return Err(format!(
            "unsupported bit depth {:?}; want 8-bit",
            info.bit_depth
        ));
    }
    let mut pixel_bytes = vec![0u8; reader.output_buffer_size()];
    reader
        .next_frame(&mut pixel_bytes)
        .map_err(|e| format!("png frame: {e}"))?;
    let mut pixels = Vec::with_capacity((png_w as usize) * (png_h as usize));
    for chunk in pixel_bytes.chunks_exact(bytes_per_pixel) {
        let r = i32::from(chunk[0]);
        let g = i32::from(chunk[1]);
        let b = i32::from(chunk[2]);
        pixels.push((0x80 << 24) | (r << 16) | (g << 8) | b);
    }
    Ok(roxlap_core::sky::Sky::from_pixels(pixels, png_w, png_h))
}
/// GPU.9 — assemble the demo's KV6 sprites. Currently a single
/// `coco.kv6` placed at the same world position the throwaway
/// KV6-as-grid prototype used (~30 voxels in front of camera
/// spawn). Returns an empty `Vec` if the embedded asset fails to
/// parse, so the demo keeps booting either way.
fn build_sprites() -> Vec<Sprite> {
    match kv6_sprite::load_coco_kv6() {
        // Directly ahead of the spawn camera ([0, -120, 50] looking
        // +y) at eye level, so the splatter is exercised and FPS is
        // meaningful even with a static camera.
        Ok(kv6) => vec![Sprite::axis_aligned(kv6, [0.0, -75.0, 50.0])],
        Err(e) => {
            eprintln!("kv6_sprite: load_coco_kv6 failed ({e}); skipping sprite");
            Vec::new()
        }
    }
}

// ---- shoot-to-carve target ------------------------------------------

/// Edge length (voxels) of the procedural carve target.
const TARGET_N: u32 = 48;
/// World position the carve target floats at — straight ahead of the
/// spawn camera ([0, -120, 50] looking +y) and a bit above eye level so
/// it reads clearly against the sky.
const TARGET_WORLD: [f32; 3] = [0.0, 60.0, 95.0];
/// Radius (voxels) of the sphere each shot subtracts.
const SHOT_RADIUS: u32 = 5;
/// Surface colour of the intact blob (voxlap-packed `0x80RRGGBB`).
const TARGET_SKIN: u32 = 0x8050_70A0;

/// A procedural blob you can shoot craters into — the demo for
/// [`Sprite::carve_sphere_with_colfunc`]. Unlike a loaded `.kv6` (which
/// stores only its surface hull and so has no interior to carve), this
/// target owns a dense occupancy grid, so each hit can expose a real
/// interior wall whose colour the carve's colfunc controls.
struct CarveTarget {
    sprite: Sprite,
    /// Dense `n³` solid occupancy in kv6-local voxel coords; the carve's
    /// `solid` predicate reads this and each shot clears the carved cells.
    occ: Vec<bool>,
    n: u32,
}

impl CarveTarget {
    fn new() -> Self {
        let n = TARGET_N;
        #[allow(clippy::cast_precision_loss)]
        let c = n as f32 * 0.5;
        let r = c - 1.0;
        let inside = |x: u32, y: u32, z: u32| {
            #[allow(clippy::cast_precision_loss)]
            let (dx, dy, dz) = (x as f32 + 0.5 - c, y as f32 + 0.5 - c, z as f32 + 0.5 - c);
            dx * dx + dy * dy + dz * dz <= r * r
        };
        let kv6 = Kv6::from_fn_shaded(n, n, n, |x, y, z| inside(x, y, z).then_some(TARGET_SKIN));
        let mut occ = vec![false; (n * n * n) as usize];
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    if inside(x, y, z) {
                        occ[((z * n + y) * n + x) as usize] = true;
                    }
                }
            }
        }
        Self {
            sprite: Sprite::axis_aligned(kv6, TARGET_WORLD),
            occ,
            n,
        }
    }

    #[inline]
    fn solid(&self, x: i32, y: i32, z: i32) -> bool {
        if x < 0 || y < 0 || z < 0 {
            return false;
        }
        let (x, y, z) = (x as u32, y as u32, z as u32);
        x < self.n && y < self.n && z < self.n && self.occ[((z * self.n + y) * self.n + x) as usize]
    }

    /// March a world-space ray through the target and return the first
    /// solid voxel it hits, in kv6-local integer coords. The sprite is
    /// axis-aligned with unit basis, so world↔local is the pivot shift
    /// `local = world - p + pivot` (see `kv6_draw_prepare`).
    fn raycast(&self, origin: [f64; 3], dir: [f64; 3]) -> Option<[i32; 3]> {
        let p = self.sprite.p;
        let piv = [
            f64::from(self.sprite.kv6.xpiv),
            f64::from(self.sprite.kv6.ypiv),
            f64::from(self.sprite.kv6.zpiv),
        ];
        const STEP: f64 = 0.3;
        const T_MAX: f64 = 2000.0;
        let mut t = 0.0;
        while t < T_MAX {
            let l = [
                origin[0] + dir[0] * t - f64::from(p[0]) + piv[0],
                origin[1] + dir[1] * t - f64::from(p[1]) + piv[1],
                origin[2] + dir[2] * t - f64::from(p[2]) + piv[2],
            ];
            #[allow(clippy::cast_possible_truncation)]
            let v = [
                l[0].floor() as i32,
                l[1].floor() as i32,
                l[2].floor() as i32,
            ];
            if self.solid(v[0], v[1], v[2]) {
                return Some(v);
            }
            t += STEP;
        }
        None
    }

    /// Carve a `SHOT_RADIUS` sphere at kv6-local `centre`, painting the
    /// freshly-exposed interior with a vertical molten gradient. Returns
    /// the number of voxels removed.
    fn carve(&mut self, centre: [i32; 3]) -> u32 {
        let n = self.n;
        let r = SHOT_RADIUS as i32;
        let r_sq = r * r;
        let (cx, cy, cz) = (centre[0], centre[1], centre[2]);
        let inside = |x: i32, y: i32, z: i32| {
            let (dx, dy, dz) = (x - cx, y - cy, z - cz);
            dx * dx + dy * dy + dz * dz <= r_sq
        };

        // Molten crater: dark-red rim at the bottom → bright yellow at
        // the top, demonstrating colfunc control over the new surface.
        let crater = move |_x: i32, _y: i32, z: i32| -> u32 {
            #[allow(clippy::cast_precision_loss)]
            let up = ((z - cz + r) as f32 / (2.0 * r as f32)).clamp(0.0, 1.0);
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let red = 0xC0 + (up * 63.0) as u32;
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let grn = 0x20 + (up * 0xB0 as f32) as u32;
            0x8000_0000 | (red << 16) | (grn << 8) | 0x10
        };

        // `solid` borrows occ immutably; sprite is a disjoint field.
        {
            let occ = &self.occ;
            let solid = |x: i32, y: i32, z: i32| {
                x >= 0
                    && y >= 0
                    && z >= 0
                    && (x as u32) < n
                    && (y as u32) < n
                    && (z as u32) < n
                    && occ[((z as u32 * n + y as u32) * n + x as u32) as usize]
            };
            self.sprite
                .carve_sphere_with_colfunc(centre, SHOT_RADIUS, solid, crater);
        }

        // Mirror the carve into our occupancy so the next shot's `solid`
        // predicate is correct.
        let mut removed = 0u32;
        for z in (cz - r).max(0)..=(cz + r).min(n as i32 - 1) {
            for y in (cy - r).max(0)..=(cy + r).min(n as i32 - 1) {
                for x in (cx - r).max(0)..=(cx + r).min(n as i32 - 1) {
                    if inside(x, y, z) {
                        let idx = ((z as u32 * n + y as u32) * n + x as u32) as usize;
                        if self.occ[idx] {
                            self.occ[idx] = false;
                            removed += 1;
                        }
                    }
                }
            }
        }
        removed
    }
}

/// Author the demo's character as an `.rkc` [`Character`]: a two-bone
/// hierarchy (a static "body" + a hinged "arm", both sharing the single
/// `coco.kv6` mesh at `mesh_id` 0) carrying a baked swing clip.
///
/// The arm angle (the second per-frame value; the first is the ignored
/// root bone) swings 0 → +16000 → 0 → -16000 and loops; the trailing seq
/// entry is a `!0` jump back to entry 0 (a 2 s cycle), exercising
/// animsprite's loop-jump path.
///
/// Returns `None` if the embedded KV6 fails to parse.
fn authored_character() -> Option<Character> {
    let kv6 = kv6_sprite::load_coco_kv6().ok()?;
    // Placed beside the static sprite, at spawn eye level.
    let root_pos = [70.0, -75.0, 50.0];

    let zero = Point3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    let z_axis = Point3 {
        x: 0.0,
        y: 0.0,
        z: 1.0,
    };
    let body_hinge = Hinge {
        parent: -1,
        p: [zero, zero],
        v: [z_axis, z_axis],
        vmin: 0,
        vmax: 0,
        htype: 0,
        filler: [0; 7],
    };
    let arm_hinge = Hinge {
        parent: 0,
        // Arm-side velcro at the arm origin; body-side velcro 40 voxels
        // right of the body centre, so the arm pivots out to the side.
        p: [
            zero,
            Point3 {
                x: 40.0,
                y: 0.0,
                z: 0.0,
            },
        ],
        v: [z_axis, z_axis],
        // Free hinge → animsprite takes the shortest angular path.
        vmin: i16::MIN,
        vmax: i16::MAX,
        htype: 0,
        filler: [0; 7],
    };

    Some(Character {
        name: "coco".to_string(),
        root: root_pos,
        meshes: vec![kv6],
        bones: vec![
            Bone {
                name: "body".to_string(),
                attachments: vec![Attachment::static_mesh(0)],
                hinge: body_hinge,
            },
            Bone {
                name: "arm".to_string(),
                attachments: vec![Attachment::static_mesh(0)],
                hinge: arm_hinge,
            },
        ],
        clips: vec![Clip {
            name: "swing".to_string(),
            data: ClipData::Skeletal {
                // Per-frame, per-bone (body, arm) Q15 hinge angles about the
                // bones' shared z-axis, lifted to rotation-only `BoneXform`s
                // (behaviour-preserving — `from_hinge_angle` reproduces the
                // legacy hinge rotation exactly).
                frmval: {
                    let axis = [0.0, 0.0, 1.0];
                    let frame = |a: i16, b: i16| {
                        vec![
                            BoneXform::from_hinge_angle(axis, a),
                            BoneXform::from_hinge_angle(axis, b),
                        ]
                    };
                    vec![frame(0, 0), frame(0, 16000), frame(0, 0), frame(0, -16000)]
                },
                seq: vec![
                    Seq { tim: 0, frm: 0 },
                    Seq { tim: 500, frm: 1 },
                    Seq { tim: 1000, frm: 2 },
                    Seq { tim: 1500, frm: 3 },
                    Seq { tim: 2000, frm: !0 },
                ],
            },
        }],
        voxel_clips: Vec::new(),
        extra_chunks: Vec::new(),
    })
}

/// Build the demo's animated KFA sprite from an `.rkc` character, played
/// back per frame by [`KfaSprite::animsprite`] (voxlap's animation
/// playback) on **both** render backends (GPU: cheap per-frame transform
/// update; CPU: re-solve), so the arm sweeps ±~88° and loops.
///
/// Source selection mirrors how **monada** will load a character at
/// runtime, exercising the disk path when asked:
/// - `ROXLAP_RKC=<path>` — load + parse that `.rkc` file instead of the
///   built-in character. On a read/parse failure the demo logs and falls
///   back to the authored character so it keeps booting.
/// - `ROXLAP_RKC_DUMP=<path>` — write the authored character to `<path>`
///   (a quick way to mint a sample `.rkc` to then load via `ROXLAP_RKC`).
/// - `ROXLAP_KFA_DUMP=<path>` — write the **lossy** voxlap-toolchain
///   `.kfa` export (skeleton + clip 0 + `coco.kv6` filename).
///
/// Returns an empty `Vec` if the embedded KV6 fails to parse, so the demo
/// keeps booting.
fn build_kfa() -> Vec<KfaSprite> {
    let Some(authored) = authored_character() else {
        return Vec::new();
    };

    // Optional side exports for the toolchain / loader testing.
    if let Some(path) = std::env::var_os("ROXLAP_RKC_DUMP") {
        let bytes = character::serialize(&authored);
        match std::fs::write(&path, &bytes) {
            Ok(()) => eprintln!("wrote {} ({} bytes)", path.to_string_lossy(), bytes.len()),
            Err(e) => eprintln!(
                "ROXLAP_RKC_DUMP: failed to write {}: {e}",
                path.to_string_lossy()
            ),
        }
    }
    if let Some(path) = std::env::var_os("ROXLAP_KFA_DUMP") {
        let bytes = roxlap_formats::kfa::serialize(&authored.to_kfa(Some(0), "coco.kv6"));
        match std::fs::write(&path, &bytes) {
            Ok(()) => eprintln!("wrote {} ({} bytes)", path.to_string_lossy(), bytes.len()),
            Err(e) => eprintln!(
                "ROXLAP_KFA_DUMP: failed to write {}: {e}",
                path.to_string_lossy()
            ),
        }
    }

    // Load from disk if asked, else round-trip the authored character
    // through serialize/parse (dogfooding the container either way).
    let character = match std::env::var_os("ROXLAP_RKC") {
        Some(path) => match std::fs::read(&path).map(|b| character::parse(&b)) {
            Ok(Ok(c)) => {
                eprintln!("loaded character from {}", path.to_string_lossy());
                c
            }
            Ok(Err(e)) => {
                eprintln!(
                    "ROXLAP_RKC: parse failed for {}: {e} — using built-in character",
                    path.to_string_lossy()
                );
                authored
            }
            Err(e) => {
                eprintln!(
                    "ROXLAP_RKC: read failed for {}: {e} — using built-in character",
                    path.to_string_lossy()
                );
                authored
            }
        },
        None => {
            let bytes = character::serialize(&authored);
            character::parse(&bytes).expect("round-trip authored .rkc character")
        }
    };

    vec![character.to_kfa_sprite(Some(0))]
}

// ---- VCL.7 flame dogfood --------------------------------------------------

/// Build a [`VoxelFrame`] from a dense `fill(x, y, z) -> Option<color>`
/// closure (voxlap-packed `0x80RRGGBB`).
fn voxel_frame_from_fn(dims: [u32; 3], fill: impl Fn(u32, u32, u32) -> Option<u32>) -> VoxelFrame {
    let owpc = dims[2].div_ceil(32).max(1) as usize;
    let cols = (dims[0] * dims[1]) as usize;
    let mut occupancy = vec![0u32; cols * owpc];
    let mut color_offsets = vec![0u32; cols + 1];
    let mut colors = Vec::new();
    for y in 0..dims[1] {
        for x in 0..dims[0] {
            let col = (x + y * dims[0]) as usize;
            color_offsets[col] = colors.len() as u32;
            for z in 0..dims[2] {
                if let Some(c) = fill(x, y, z) {
                    occupancy[col * owpc + (z >> 5) as usize] |= 1u32 << (z & 31);
                    colors.push(c);
                }
            }
        }
    }
    color_offsets[cols] = colors.len() as u32;
    VoxelFrame {
        occupancy,
        colors,
        color_offsets,
    }
}

/// One procedural flame frame: a blob wider at the base (high local z),
/// narrowing + reddening toward the tip (low z), with a per-frame height
/// flicker + a cheap spatial wobble so the silhouette dances.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn flame_frame(dims: [u32; 3], fi: u32) -> VoxelFrame {
    let (mx, my, mz) = (dims[0], dims[1], dims[2]);
    let (cx, cy) = (mx as f32 * 0.5, my as f32 * 0.5);
    // Per-frame flame height (70–100% of the bbox).
    let height = mz as f32 * (0.7 + 0.3 * ((fi as f32 * 1.3).sin() * 0.5 + 0.5));
    voxel_frame_from_fn(dims, move |x, y, z| {
        // `up`: 0 at the base (z = mz-1), growing toward the tip (z = 0).
        let up = (mz - 1 - z) as f32;
        if up > height {
            return None;
        }
        let frac = up / mz as f32; // 0 base → ~1 tip
        let wobble = (((x * 7 + y * 11 + fi * 13) % 7) as f32 - 3.0) * 0.15;
        let r = (1.0 - frac) * (mx as f32 * 0.45) + wobble;
        let (dx, dy) = (x as f32 + 0.5 - cx, y as f32 + 0.5 - cy);
        if dx * dx + dy * dy > r * r {
            return None;
        }
        // Yellow at the base → orange → red at the tip.
        let t = frac.clamp(0.0, 1.0);
        let grn = (0xF0 as f32 * (1.0 - t) + 0x20 as f32 * t) as u32;
        let blu = (0x40 as f32 * (1.0 - t)) as u32;
        Some(0x8000_0000 | (0xFF << 16) | (grn << 8) | blu)
    })
}

/// A small looping procedural flame voxel clip (8 frames, ~70 ms each).
fn flame_clip() -> VoxelClip {
    const DIMS: [u32; 3] = [10, 10, 22];
    const FRAMES: u32 = 8;
    let frames: Vec<VoxelFrame> = (0..FRAMES).map(|fi| flame_frame(DIMS, fi)).collect();
    // Pivot at the base centre (high z) so the flame rises from wherever the
    // attachment places it.
    #[allow(clippy::cast_precision_loss)]
    VoxelClip::from_frames(
        DIMS,
        [5.0, 5.0, (DIMS[2] - 1) as f32],
        1.0,
        LoopMode::Loop,
        &frames,
        &[],
        70,
        2,
    )
}

/// VCL.7 dogfood — the authored coco character plus a flame voxel clip hung
/// off its swinging arm, registered via the attachment runtime
/// ([`SceneRenderer::add_character`]). Positioned beside the KFA-path coco
/// so both the legacy and new paths are visible. `None` if `coco.kv6`
/// fails to parse.
fn flame_character() -> Option<Character> {
    use roxlap_formats::xform::{BoneXform, Quat};
    let mut ch = authored_character()?;
    // Sit beside the KFA-path coco (which is at [70, -75, 50]).
    ch.root = [120.0, -75.0, 50.0];
    ch.voxel_clips = vec![flame_clip()];
    // A second attachment on the arm bone (index 1): the flame, offset out
    // toward the arm tip + a bit up (world -z). Multi-attachment + a clip
    // playing on its own clock.
    ch.bones[1].attachments.push(Attachment {
        target: roxlap_formats::character::MeshRef::Clip(0),
        local_offset: BoneXform {
            t: [40.0, 0.0, -20.0],
            r: Quat::IDENTITY,
            s: [1.0, 1.0, 1.0],
        },
        playback: roxlap_formats::character::ClipPlayback::default(),
    });
    Some(ch)
}

const FAST_MULT: f64 = 4.0;
const MOUSE_SENS: f64 = 0.0025;
/// Pitch clamped just shy of ±90° so the basis stays well-conditioned.
const PITCH_LIMIT: f64 = 88.0_f64 * std::f64::consts::PI / 180.0;

/// Pick-demo (`C`): horizontal ground reference plane the mouse cursor
/// is projected onto (world z; voxlap z is *down*, smaller = up). The
/// streaming-hills surface sits near z≈80, so the cursor floats just
/// above it. A fixed plane is the right primitive for a top-down tile
/// cursor; snapping to the actual terrain height under the ray (read a
/// voxel column) is a follow-up.
const PICK_GROUND_Z: f64 = 72.0;
/// Pick-demo top-down camera: high above the centred grid (z very
/// negative = high up), looking steeply down with a touch of
/// perspective so the scene reads as a strategy-game view.
const PICK_CAM_POS: [f64; 3] = [0.0, 0.0, -520.0];
const PICK_CAM_PITCH: f64 = 1.30; // ~74° down (z-down convention)
/// Vertical FOV the GPU marcher renders with (`FrameParams.gpu_fov_y_rad`).
/// The pick unproject must use the same value, so it lives here.
const GPU_FOV_Y_DEG: f64 = 60.0;

// Screen→world picking. The ray-per-backend unproject + depth read +
// grid resolution now live in the LIBRARY (`SceneRenderer::pixel_ray`
// / `pick`, `Scene::resolve_voxel`); the demo only keeps the
// ground-plane intersect below for the continuous hover cursor.

/// Intersect a world ray `pos + t·dir` with the horizontal plane
/// `z = ground_z`. `None` if the ray is parallel to the plane or the
/// plane lies behind the camera.
#[allow(clippy::cast_possible_truncation)]
/// Build the HUD overlay panel (a fixed top-left info box). Run inside
/// `egui::Context::run`; the renderer then composites the tessellation
/// over the frame via `SceneRenderer::paint_egui`.
fn hud_panel(
    ctx: &egui::Context,
    backend: &str,
    fps: f64,
    pos: [f64; 3],
    yaw: f64,
    pitch: f64,
    scan: i32,
) {
    egui::Area::new(egui::Id::new("roxlap-hud"))
        .fixed_pos(egui::pos2(8.0, 8.0))
        .show(ctx, |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.label(format!("roxlap-scene-demo — {backend} backend"));
                ui.label(format!("{fps:.0} FPS"));
                ui.label(format!("pos ({:.0}, {:.0}, {:.0})", pos[0], pos[1], pos[2]));
                ui.label(format!("yaw {yaw:.2}   pitch {pitch:.2}"));
                ui.label(format!("scan dist {scan}"));
                ui.separator();
                ui.label("F1: toggle HUD");
            });
        });
}

/// L3.0 occlusion gate: a depth-tested debug overlay — a floor reference
/// grid laid on the terrain surface (`z ≈ 200`), a wire box rising out of
/// the hills, and always-on-top RGB origin axes that show through the
/// terrain. The grid/box are `depth_test = true` (hills in front occlude
/// them); the axes are `depth_test = false`. Placed in front of the spawn
/// camera (`(0, -120, 50)` looking `+y`).
/// Generate a 64×64 RGBA8 reference texture for the `I`-toggle image
/// sprite: a magenta/teal checkerboard with a red top edge and a green
/// left edge, so orientation (row 0 = top, col 0 = left) and
/// perspective-correctness read at a glance. Fully opaque.
fn make_reference_image() -> (Vec<u8>, u32, u32) {
    const N: u32 = 64;
    let mut rgba = vec![0u8; (N * N * 4) as usize];
    for y in 0..N {
        for x in 0..N {
            let i = ((y * N + x) * 4) as usize;
            let checker = ((x / 8) + (y / 8)) % 2 == 0;
            let (r, g, b) = if y < 3 {
                (255, 40, 40) // red top edge
            } else if x < 3 {
                (40, 255, 40) // green left edge
            } else if checker {
                (220, 40, 200) // magenta
            } else {
                (40, 200, 200) // teal
            };
            rgba[i] = r;
            rgba[i + 1] = g;
            rgba[i + 2] = b;
            rgba[i + 3] = 255;
        }
    }
    (rgba, N, N)
}

/// The demo's `I`-toggle reference sprite (shared by the draw + pick
/// paths so they agree on placement): a 64×64 quad on the Front plane
/// (normal +Y, u=+X, v=+Z) at 1 texel = 1 voxel, depth-tested + two-sided.
fn demo_image_sprite(id: ImageId) -> ImageSprite {
    ImageSprite {
        image: id,
        origin: [-32.0, 0.0, -64.0],
        facing: ImageFacing::World {
            u: [1.0, 0.0, 0.0],
            v: [0.0, 0.0, 1.0],
        },
        size: [64.0, 64.0],
        tint: 0xFFFF_FFFF,
        alpha_cutoff: 0.0,
        depth_test: true,
        double_sided: true,
    }
}

fn debug_overlay_lines() -> Vec<Line3> {
    const GROUND_Z: f64 = 199.0; // just above the surface (smaller z = up)
    let mut lines = Vec::new();

    // Floor grid spanning x∈[-80,80], y∈[0,240], 20-voxel cells, sitting on
    // the terrain. Cyan, depth-tested so it's hidden behind hill silhouettes.
    let grid_color = 0xC0_30_C0_FF;
    let (x0, x1, y0, y1, step) = (-80, 80, 0, 240, 20);
    let mut x = x0;
    while x <= x1 {
        lines.push(Line3 {
            a: [x as f64, y0 as f64, GROUND_Z],
            b: [x as f64, y1 as f64, GROUND_Z],
            color: grid_color,
            width_px: 1.0,
            depth_test: true,
        });
        x += step;
    }
    let mut y = y0;
    while y <= y1 {
        lines.push(Line3 {
            a: [x0 as f64, y as f64, GROUND_Z],
            b: [x1 as f64, y as f64, GROUND_Z],
            color: grid_color,
            width_px: 1.0,
            depth_test: true,
        });
        y += step;
    }

    // Wire box: 40³ cube centred at (0, 120), rising from the surface
    // (z=200) up to z=160. Yellow, depth-tested.
    push_box_edges(
        &mut lines,
        [-20.0, 100.0, 160.0],
        [20.0, 140.0, 200.0],
        0xFF_FF_D0_00,
        2.0,
        true,
    );

    // Origin axes from (0, 0, 195): +X red, +Y green, +Z(down) blue.
    // Always-on-top (depth_test = false) — visible through the hills.
    let origin = [0.0, 0.0, 195.0];
    for (axis, color) in [
        ([60.0, 0.0, 0.0], 0xFF_FF_30_30u32),
        ([0.0, 60.0, 0.0], 0xFF_30_FF_30),
        ([0.0, 0.0, 60.0], 0xFF_30_30_FF),
    ] {
        lines.push(Line3 {
            a: origin,
            b: [
                origin[0] + axis[0],
                origin[1] + axis[1],
                origin[2] + axis[2],
            ],
            color,
            width_px: 3.0,
            depth_test: false,
        });
    }

    lines
}

/// Push the 12 edges of the axis-aligned box `[lo, hi]` as [`Line3`]s.
fn push_box_edges(
    out: &mut Vec<Line3>,
    lo: [f64; 3],
    hi: [f64; 3],
    color: u32,
    width_px: f32,
    depth_test: bool,
) {
    // 8 corners by bit-picking lo/hi per axis.
    let corner = |i: usize| {
        [
            if i & 1 == 0 { lo[0] } else { hi[0] },
            if i & 2 == 0 { lo[1] } else { hi[1] },
            if i & 4 == 0 { lo[2] } else { hi[2] },
        ]
    };
    // Edges connect corners differing in exactly one axis bit.
    for i in 0..8usize {
        for bit in [1usize, 2, 4] {
            let j = i | bit;
            if j != i && (i & bit) == 0 {
                out.push(Line3 {
                    a: corner(i),
                    b: corner(j),
                    color,
                    width_px,
                    depth_test,
                });
            }
        }
    }
}

fn plane_hit(pos: [f64; 3], dir: [f64; 3], ground_z: f64) -> Option<[f32; 3]> {
    if dir[2].abs() < 1e-9 {
        return None; // ray parallel to the ground plane
    }
    let t = (ground_z - pos[2]) / dir[2];
    if t <= 0.0 {
        return None; // plane is behind the camera
    }
    Some([
        (pos[0] + t * dir[0]) as f32,
        (pos[1] + t * dir[1]) as f32,
        ground_z as f32,
    ])
}

fn main() {
    let event_loop = EventLoop::new().expect("winit: EventLoop::new");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("winit: run_app");
}

#[allow(clippy::struct_excessive_bools)]
struct InputState {
    forward: bool,
    back: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
    fast: bool,
}

impl InputState {
    const fn new() -> Self {
        Self {
            forward: false,
            back: false,
            left: false,
            right: false,
            up: false,
            down: false,
            fast: false,
        }
    }
}

// --- sprite spinner (incremental add/remove demo) -----------------------

/// Number of distinct colours the spinner cycles through.
const SPINNER_COLORS: usize = 6;
/// World centre of the spinner ring (in front of the spawn camera at
/// `[0, -120, 50]` looking +y), and its radius in voxels.
// z is voxlap-down (smaller = higher); the streaming hills crest near
// z≈72, so centre the ring well above that and keep its radius small
// enough that the bottom (z = centre.z + radius) stays clear of terrain —
// otherwise the lower blocks z-fight the hills and flicker.
const SPINNER_CENTER: [f32; 3] = [0.0, -45.0, 28.0];
const SPINNER_RADIUS: f32 = 26.0;

/// The spinner's palette (voxlap-packed `0x80RRGGBB`, high bit = shaded).
const SPINNER_PALETTE: [u32; SPINNER_COLORS] = [
    0x80FF_4040, // red
    0x80FF_A030, // orange
    0x80F0_F040, // yellow
    0x8040_E060, // green
    0x8040_A0FF, // blue
    0x80C0_60FF, // violet
];

/// Build one spinner block, recoloured `col`. Deliberately **non-cubic**
/// (wide in local x) so the per-frame tumble the spinner applies is
/// visible — a rotation visibly swings the silhouette's width. Pivot is
/// centred by `from_fn_shaded`, so an instance's position places the
/// block's centre.
fn build_spinner_block(col: u32) -> Kv6 {
    let (bx, by, bz) = (14u32, 5u32, 5u32);
    Kv6::from_fn_shaded(bx, by, bz, |_, _, _| Some(col))
}

/// Where the carve target landed in a [`SpriteSet`], so the demo can map
/// [`SceneRenderer::set_sprites`]'s positional ids back to its handle.
/// (The spinner streams its own models via `add_sprite_model`, so it
/// needs no slot here.)
struct SpriteLayout {
    /// Model index of the `G`-carve target, if present.
    carve_model: Option<usize>,
}

/// One live spinner block: its streamed-in unique model + instance, the
/// fixed ring position (orbit angle), and a per-block spin offset for
/// variety.
struct SpinnerBlock {
    model: SpriteModelId,
    inst: SpriteInstanceId,
    pos: [f32; 3],
    spin_offset: f64,
}

/// A ring of coloured blocks streaming in and out while each tumbles in
/// place — the canonical dogfood of the dynamic sprite API the way a
/// physics demo (asteroids/debris) would drive it:
///
/// * `add_sprite_model` — every block is a unique procedural model
///   registered incrementally (no full `set_sprites` rebuild);
/// * `add_sprite_instance_posed` — blocks spawn already tumbling, so
///   there's no one-frame axis-aligned flash;
/// * `set_sprite_instance_transforms` — every frame all live blocks are
///   re-posed in one batched call (GPU dirty-flush coalesces it to one
///   upload);
/// * `remove_sprite_instance` + `remove_sprite_model` — the oldest block
///   leaves once the ring is full;
/// * `compact_sprite_models` — periodically reclaims the removed models'
///   GPU buffer space.
#[derive(Default)]
struct Spinner {
    /// Live blocks oldest→newest; the front is dropped when full.
    ring: std::collections::VecDeque<SpinnerBlock>,
    /// Head angle (radians), advanced one step per appended block.
    angle: f64,
    /// Seconds accumulated toward the next append.
    accum: f64,
    /// Total elapsed seconds — drives the shared tumble phase.
    clock: f64,
    /// Next palette colour to use.
    next_color: usize,
    /// Removed-model count since the last `compact_sprite_models`.
    dead_models: usize,
    /// `false` until the demo confirms the renderer's sprite layer is
    /// live (cleared whenever a `set_sprites` wipes the dynamic layer).
    enabled: bool,
}

impl Spinner {
    /// Seconds between appends (≈ append rate).
    const ADD_PERIOD: f64 = 0.18;
    /// Head-angle advance per appended block (radians). Wide spacing so
    /// the few blocks are clearly separated around the ring.
    const ANGLE_STEP: f64 = 0.6;
    /// Trail length — blocks are dropped once the ring exceeds this.
    const MAX: usize = 10;
    /// Tumble rate (radians/second) of each block about world-z.
    const SPIN_RATE: f64 = 1.4;
    /// Compact the registry after this many model removals.
    const COMPACT_EVERY: usize = 8;

    /// Enable the spinner against a freshly built sprite layer, dropping
    /// any prior handles (a `set_sprites` invalidated every dynamic model
    /// + instance). The spinner re-streams its own models from scratch.
    fn reset(&mut self) {
        self.ring.clear();
        self.angle = 0.0;
        self.accum = 0.0;
        self.clock = 0.0;
        self.next_color = 0;
        self.dead_models = 0;
        self.enabled = true;
    }

    /// Tumble pose for a block at `pos` given the shared `phase` (radians)
    /// — a rotation about world-z, so the wide block visibly swings its
    /// silhouette width.
    #[allow(clippy::cast_possible_truncation)]
    fn pose(pos: [f32; 3], phase: f64) -> DynSpriteTransform {
        let (s, c) = (phase.sin() as f32, phase.cos() as f32);
        DynSpriteTransform {
            pos,
            right: [c, s, 0.0],       // local +x in world (xy-plane rotation)
            up: [-s, c, 0.0],         // local +y in world
            forward: [0.0, 0.0, 1.0], // local +z stays world +z
        }
    }

    /// Advance by `dt` seconds: stream blocks in/out and re-pose every
    /// live block, exercising the full dynamic sprite API each frame.
    fn update(&mut self, renderer: &mut SceneRenderer, dt: f64) {
        if !self.enabled {
            return;
        }
        self.clock += dt;
        // Clamp so a long stall (e.g. window drag) doesn't burst-spawn.
        self.accum = (self.accum + dt).min(0.5);
        while self.accum >= Self::ADD_PERIOD {
            self.accum -= Self::ADD_PERIOD;
            #[allow(clippy::cast_possible_truncation)]
            let a = self.angle as f32;
            let pos = [
                SPINNER_CENTER[0] + SPINNER_RADIUS * a.cos(),
                SPINNER_CENTER[1],
                SPINNER_CENTER[2] + SPINNER_RADIUS * a.sin(),
            ];
            let col = SPINNER_PALETTE[self.next_color % SPINNER_PALETTE.len()];
            let spin_offset = self.angle; // stagger each block's phase
            self.next_color += 1;
            self.angle += Self::ANGLE_STEP;

            // Stream a fresh unique model in, spawn it pre-posed.
            let model = renderer.add_sprite_model(&build_spinner_block(col));
            let phase = self.clock * Self::SPIN_RATE + spin_offset;
            let inst = renderer.add_sprite_instance_posed(model, Self::pose(pos, phase));
            self.ring.push_back(SpinnerBlock {
                model,
                inst,
                pos,
                spin_offset,
            });

            if self.ring.len() > Self::MAX {
                if let Some(old) = self.ring.pop_front() {
                    renderer.remove_sprite_instance(old.inst);
                    renderer.remove_sprite_model(old.model);
                    self.dead_models += 1;
                }
            }
            // Periodically reclaim the removed models' GPU buffer holes.
            if self.dead_models >= Self::COMPACT_EVERY {
                renderer.compact_sprite_models();
                self.dead_models = 0;
            }
        }

        // Re-pose every live block this frame in one batched upload.
        let phase_base = self.clock * Self::SPIN_RATE;
        let updates: Vec<(SpriteInstanceId, DynSpriteTransform)> = self
            .ring
            .iter()
            .map(|b| (b.inst, Self::pose(b.pos, phase_base + b.spin_offset)))
            .collect();
        renderer.set_sprite_instance_transforms(&updates);
    }
}

#[allow(clippy::struct_excessive_bools)]
struct App {
    window: Option<Arc<Window>>,
    /// Unified CPU/GPU renderer (RF). Owns presentation, the GPU
    /// scene residency + dirty-chunk tracking, the CPU compositor +
    /// pool/z-buffer, and the sprite reps. Created in `resumed` —
    /// `ROXLAP_GPU=1` selects the GPU backend with automatic CPU
    /// fallback.
    renderer: Option<SceneRenderer>,
    engine: Engine,
    scene: SceneAndCamera,
    input: InputState,
    grabbed: bool,
    last_frame: Instant,
    /// Set by the `F` hotkey; consumed in `redraw` after the frame
    /// is composited so the captured PPM is the same pixels the
    /// user just saw.
    capture_pending: bool,
    /// Live-adjustable scan distance (voxels). `+` / `-` bump it
    /// by `SCAN_DIST_STEP`; clamped to `[SCAN_DIST_MIN, SCAN_DIST_MAX]`.
    scan_dist: i32,
    /// GPU scene-grid LOD scan distance (world units), from
    /// `ROXLAP_GPU_MIP_SCAN_DIST` (default 64). Fed to the GPU
    /// backend each frame; ignored by the CPU backend.
    gpu_mip_scan_dist: f32,
    /// GPU.13.0 — saved `(pos, yaw, pitch)` so `H` can toggle between
    /// the high-altitude top-down vantage (where the chunk-AABB
    /// early-out pays off) and the previous ground-level pose for an
    /// FPS A/B. `None` until the first `H` press.
    saved_pose: Option<([f64; 3], f64, f64)>,
    /// Pick-demo mode (`C`): top-down camera + a free OS cursor that
    /// places a sprite on a ground plane under the pointer. Prototype
    /// for screen→world picking (roxlap ships no built-in picker).
    pick_mode: bool,
    /// Last cursor position in physical pixels (window space), from
    /// `CursorMoved`; unprojected each frame while in pick mode.
    mouse_px: (f64, f64),
    /// World point under the cursor this frame (ground-plane hit).
    cursor_world: [f32; 3],
    /// Left-click-placed marker world positions.
    placed: Vec<[f32; 3]>,
    /// Recoloured `[cursor, marker]` KV6 models, built once on entering
    /// pick mode from the base sprite.
    pick_models: Vec<Sprite>,
    /// `(pos, yaw, pitch)` saved on entering pick mode, restored on
    /// exit.
    pick_saved_pose: Option<([f64; 3], f64, f64)>,
    /// Base KV6 sprite(s); `build_sprite_set` expands these into the
    /// demo's instanced field handed to the renderer at startup.
    sprites: Vec<Sprite>,
    /// Animated KFA sprite(s), registered once in `resumed` and re-posed
    /// each frame via `animsprite` + `update_kfa_poses`.
    kfa: Vec<KfaSprite>,
    /// VCL.7 dogfood — a second coco registered via the **attachment
    /// runtime** (`add_character`) with a procedural flame voxel clip
    /// hanging off its swinging arm (multi-attachment + clip playback +
    /// skeletal animation). Advanced each frame via `advance_character`.
    flame_char: Option<CharacterId>,
    /// Post-S7.6: lighting + mip bake driver for streaming grids.
    /// Runs each frame right after `pump_streaming`; bakes any
    /// newly-installed chunks (and re-bakes their 4 cardinal
    /// neighbours so chunk-edge brightness banding resolves as
    /// chunks arrive). Empty + no-op when no streaming grid exists.
    bake_tracker: StreamingBakeTracker,
    /// Title bar prefix (e.g. `"roxlap-scene-demo (GPU: …)"`).
    /// `tick_fps` composes `"<base> — NNN FPS"` on top of this
    /// every ~500ms so the user can read the live frame rate
    /// without a HUD overlay.
    title_base: String,
    fps_frames: u32,
    fps_last: Instant,
    /// HUD: egui context (host-side) + the winit↔egui input bridge,
    /// created in `resumed`. `F1` toggles `hud_on`; when on, `redraw`
    /// overlays an egui panel via `SceneRenderer::paint_egui` instead of
    /// a plain present. Demonstrates the renderer's egui seam on both
    /// the CPU (software-rasterised) and GPU (egui-wgpu) backends.
    egui_ctx: egui::Context,
    egui_state: Option<egui_winit::State>,
    hud_on: bool,
    /// `L` toggles a depth-tested debug overlay (a wire box + floor grid +
    /// always-on-top origin axes) drawn via `SceneRenderer::draw_lines` —
    /// the L3.0 occlusion gate.
    lines_on: bool,
    /// `I` toggles a world-placed 2D image sprite drawn via
    /// [`SceneRenderer::draw_images`] — a depth-tested reference quad
    /// (the demiurg editor's first consumer of the primitive). The
    /// texture is uploaded once in `resumed`; `image_id` holds its handle.
    images_on: bool,
    image_id: Option<ImageId>,
    /// Last FPS computed in `tick_fps`, shown in the HUD.
    last_fps: f64,
    /// Shoot-to-carve target: left-click (while grabbed) casts the
    /// centre-screen ray and subtracts a sphere via
    /// [`Sprite::carve_sphere_with_colfunc`].
    carve_target: CarveTarget,
    /// Stable handle for the carve target's registered model, captured
    /// from [`SceneRenderer::set_sprites`] whenever the normal sprite
    /// field is (re)registered. `fire` refreshes this model incrementally
    /// instead of re-deriving a positional index. `None` until the first
    /// registration (or when sprites are disabled).
    carve_target_id: Option<SpriteModelId>,
    /// Rotating ring of coloured spheres exercising the incremental
    /// add/remove sprite API each frame (see [`Spinner`]).
    spinner: Spinner,
}

impl App {
    fn new() -> Self {
        let mut engine = Engine::new();
        // Fog disabled — the embedded `assets/sky.png` panorama
        // provides a far-distance reference. Falls back to voxlap's
        // blue gradient if the PNG fails to decode.
        let sky = load_png_sky(SKY_PNG).unwrap_or_else(|e| {
            eprintln!("sky: PNG decode failed ({e}); falling back to blue gradient");
            roxlap_core::sky::Sky::blue_gradient()
        });
        engine.set_sky(Some(sky));

        // S7.6: `build_demo` now defaults to the streaming-hills
        // path — the ground grid is backed by
        // `HillsChunkGenerator` and pumped each frame. The
        // historical static 32×32 ground stays available via
        // `ROXLAP_STATIC=1` for repro / regression tests.
        let scene = build_demo();
        if scene.streaming_enabled {
            eprintln!(
                "streaming hills active (T prints chunks/pending, ROXLAP_STATIC=1 for static)"
            );
        }
        Self {
            window: None,
            renderer: None,
            engine,
            scene,
            input: InputState::new(),
            grabbed: false,
            last_frame: Instant::now(),
            capture_pending: false,
            scan_dist: SCAN_DIST_INITIAL,
            gpu_mip_scan_dist: 64.0,
            saved_pose: None,
            pick_mode: false,
            mouse_px: (0.0, 0.0),
            cursor_world: [0.0, 0.0, PICK_GROUND_Z as f32],
            placed: Vec::new(),
            pick_models: Vec::new(),
            pick_saved_pose: None,
            sprites: build_sprites(),
            kfa: build_kfa(),
            flame_char: None,
            bake_tracker: StreamingBakeTracker::new(),
            title_base: "roxlap-scene-demo".to_string(),
            fps_frames: 0,
            fps_last: Instant::now(),
            egui_ctx: egui::Context::default(),
            egui_state: None,
            hud_on: true,
            lines_on: true,
            images_on: false,
            image_id: None,
            last_fps: 0.0,
            carve_target: CarveTarget::new(),
            carve_target_id: None,
            spinner: Spinner::default(),
        }
    }

    /// Bump the rolling frame counter and refresh the title bar
    /// every ~500ms. Both `redraw` (CPU path) and `redraw_gpu` (GPU
    /// path) call this at the end of a successful frame.
    fn tick_fps(&mut self) {
        self.fps_frames += 1;
        let now = Instant::now();
        let dt = (now - self.fps_last).as_secs_f32();
        if dt < 0.5 {
            return;
        }
        let fps = f64::from(self.fps_frames) / f64::from(dt);
        self.last_fps = fps;
        if let Some(window) = self.window.as_ref() {
            window.set_title(&format!("{} — {:.1} FPS", self.title_base, fps));
        }
        // GPU.11.2 — `ROXLAP_FPS_LOG=1` mirrors the title-bar FPS to
        // stderr so it can be captured in a terminal alongside the
        // present-mode / pass-toggle diagnostic lines.
        if std::env::var_os("ROXLAP_FPS_LOG").is_some() {
            eprintln!("fps: {fps:.1}");
        }
        self.fps_frames = 0;
        self.fps_last = now;
    }

    /// Feed a window event to egui (passive — never consumed, so the
    /// game keeps its controls) and handle the `F1` HUD toggle.
    fn handle_hud_event(&mut self, event: &WindowEvent) {
        if let (Some(window), Some(state)) = (self.window.as_ref(), self.egui_state.as_mut()) {
            let _ = state.on_window_event(window, event);
        }
        if let WindowEvent::KeyboardInput {
            event:
                KeyEvent {
                    physical_key: PhysicalKey::Code(KeyCode::F1),
                    state: ElementState::Pressed,
                    repeat: false,
                    ..
                },
            ..
        } = event
        {
            self.hud_on = !self.hud_on;
        }
    }

    /// One frame: advance the scene from input + streaming, then hand
    /// it to the unified renderer (CPU or GPU — the demo no longer
    /// knows which). All the old per-backend glue lives in
    /// `roxlap-render` now.
    fn redraw(&mut self) {
        // Snapshot the size, then drop the window borrow so the
        // `&mut self` animation/render below doesn't collide with it.
        let size = {
            let Some(window) = self.window.as_ref() else {
                return;
            };
            window.inner_size()
        };
        if size.width == 0 || size.height == 0 {
            return;
        }

        // Advance camera + scene animation from input.
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f64();
        self.last_frame = now;
        // DEBUG: ROXLAP_AUTOFLY=1 drives the camera forward in a slow
        // turn to churn chunk streaming (repro for the fly-around crash).
        if std::env::var_os("ROXLAP_AUTOFLY").is_some() {
            self.input.forward = true;
            self.input.fast = true;
            self.scene.yaw += 0.9 * dt;
            // Oscillate pitch to fly up/down (exercise chunk z-stacking).
            self.scene.pitch = 0.45 * (self.scene.yaw * 0.7).sin();
            self.scene.refresh_camera();
        }
        self.tick_camera(dt);
        self.scene.tick_ship_spin(dt);
        // Streaming pump (no-op unless `build_streaming_demo` is
        // active) — drains arrivals, evicts past r_evict, dispatches
        // missing chunks, then bakes lighting/mips on the delta.
        if self.scene.streaming_enabled {
            self.scene
                .scene
                .pump_streaming(glam::DVec3::from_array(self.scene.cam_pos));
            self.bake_tracker.process(&mut self.scene.scene);
        }

        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };

        // Advance + re-pose the animated KFA sprite(s): `animsprite`
        // (voxlap's playback port) walks the baked curve from `dt`, then
        // the renderer re-poses the limbs — GPU does a transform-only
        // buffer update, CPU re-solves the bone transforms for drawing.
        if !self.kfa.is_empty() {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let dt_ms = (dt * 1000.0) as i32;
            for k in &mut self.kfa {
                k.animsprite(dt_ms);
            }
            renderer.update_kfa_poses(&mut self.kfa);
        }

        // VCL.7 — advance the flame character (skeletal swing + flame clip
        // playback) in one call.
        if let Some(id) = self.flame_char {
            renderer.advance_character(id, dt);
        }

        // Spinner: stream coloured spheres in/out around the ring each
        // frame via the incremental add/remove API (paused in pick mode,
        // which rebuilds the whole sprite set per frame).
        if !self.pick_mode && std::env::var_os("ROXLAP_NO_SPINNER").is_none() {
            self.spinner.update(renderer, dt);
        }

        // Per-frame opticast settings (host owns scan distance).
        let mut settings = OpticastSettings::for_oracle_framebuffer(size.width, size.height);
        settings.max_scan_dist = self.scan_dist;
        settings.mip_levels = 6;
        settings.mip_scan_dist = 64;
        #[allow(clippy::cast_sign_loss)]
        let chunks_visible = (self.scan_dist.max(1) as u32) / roxlap_scene::CHUNK_SIZE_XY + 4;

        let frame = FrameParams {
            settings: &settings,
            sky_color: self.engine.sky_color(),
            sky: self.engine.sky(),
            // DDA.9: fog is config-driven across all backends. Drive it
            // from the sky colour + the live scan distance so terrain
            // fades into the sky (matches the DDA/GPU look) and tracks
            // the +/- scan-distance keys. (An explicit `Engine::set_fog`
            // would override this.)
            fog_color: if self.engine.fog_max_scan_dist() > 0 {
                self.engine.fog_color()
            } else {
                self.engine.sky_color()
            },
            fog_max_scan_dist: if self.engine.fog_max_scan_dist() > 0 {
                self.engine.fog_max_scan_dist()
            } else {
                self.scan_dist
            },
            treat_z_max_as_air: true,
            gpu_mip_scan_dist: self.gpu_mip_scan_dist,
            gpu_max_outer_steps: chunks_visible,
            gpu_fov_y_rad: (GPU_FOV_Y_DEG as f32).to_radians(),
            draw_sprites: true,
            side_shades: self.engine.side_shades(),
        };

        // Pick mode: unproject the mouse onto the ground plane and
        // re-place the cursor (+ any dropped markers) as sprites. Field
        // accesses are disjoint from `self.renderer`, so this composes
        // with the `renderer` borrow held above.
        if self.pick_mode && self.pick_models.len() >= 2 {
            let cam = self.scene.camera;
            let (mx, my) = self.mouse_px;
            // Hover cursor: the canonical unproject (matches whichever
            // backend is active) dropped on the ground plane. Uses the
            // previous frame's projection — fine for a cursor.
            if let Some(ray) = renderer.view_ray(&cam, mx, my) {
                if let Some(w) = plane_hit(ray.origin.to_array(), ray.dir.to_array(), PICK_GROUND_Z)
                {
                    self.cursor_world = w;
                }
            }
            let mut instances = vec![SpriteInstanceDesc {
                model: 0,
                pos: self.cursor_world,
            }];
            for p in &self.placed {
                instances.push(SpriteInstanceDesc { model: 1, pos: *p });
            }
            let set = SpriteSet {
                models: self.pick_models.clone(),
                instances,
                carve_model: None,
            };
            renderer.set_sprites(&set);
        }

        if self.capture_pending {
            renderer.request_capture();
        }
        let camera = self.scene.camera; // Camera is Copy
        renderer.render(&mut self.scene.scene, &camera, &frame);

        // L3.0: depth-tested debug overlay. Lands in the framebuffer
        // after the world pass, before present/paint_egui (egui still
        // paints panels on top). The grid + box are occluded by terrain
        // they sit behind; the origin axes are always-on-top.
        if self.lines_on {
            let lines = debug_overlay_lines();
            renderer.draw_lines(&camera, &lines);
        }

        // 2D image sprite (depth-tested reference quad). Placed on the
        // Front plane (normal +Y, u=+X, v=+Z) at 1 texel = 1 voxel, so the
        // terrain occludes the parts behind it.
        if self.images_on {
            if let Some(id) = self.image_id {
                renderer.draw_images(&camera, &[demo_image_sprite(id)]);
            }
        }

        // RF: render no longer presents — finish the frame. With the HUD
        // on, overlay an egui panel via `paint_egui`; otherwise a plain
        // present. (The capture below reads the pre-HUD framebuffer.)
        let hud_ready = self.hud_on && self.window.is_some() && self.egui_state.is_some();
        if hud_ready {
            let backend_label = match renderer.backend() {
                roxlap_render::Backend::Gpu => "GPU",
                roxlap_render::Backend::Cpu => "CPU",
            };
            let fps = self.last_fps;
            let pos = self.scene.cam_pos;
            let yaw = self.scene.yaw;
            let pitch = self.scene.pitch;
            let scan = self.scan_dist;
            let window = self.window.as_ref().expect("hud_ready");
            let state = self.egui_state.as_mut().expect("hud_ready");
            let raw_input = state.take_egui_input(window);
            // egui 0.34 deprecated `Context::run` (Context closure) in
            // favour of `run_ui` (Ui closure); the begin/end-pass pair
            // keeps the Context-based `hud_panel` without the warning.
            self.egui_ctx.begin_pass(raw_input);
            hud_panel(&self.egui_ctx, backend_label, fps, pos, yaw, pitch, scan);
            let full = self.egui_ctx.end_pass();
            state.handle_platform_output(window, full.platform_output);
            let ppp = self.egui_ctx.pixels_per_point();
            let jobs = self.egui_ctx.tessellate(full.shapes, ppp);
            renderer.paint_egui(&jobs, &full.textures_delta, ppp);
        } else {
            renderer.present();
        }

        if self.capture_pending {
            self.capture_pending = false;
            if let Some((buf, w, h)) = renderer.take_capture() {
                match write_capture(
                    &buf,
                    w,
                    h,
                    &self.scene.camera,
                    self.scene.yaw,
                    self.scene.pitch,
                    self.scene.ship_angles,
                    self.scene.spin_enabled,
                ) {
                    Ok(()) => eprintln!(
                        "captured: roxlap-scene-capture.txt + .ppm (pos=({:.2}, {:.2}, {:.2}))",
                        self.scene.cam_pos[0], self.scene.cam_pos[1], self.scene.cam_pos[2],
                    ),
                    Err(e) => eprintln!("capture failed: {e}"),
                }
            } else {
                eprintln!("capture unsupported on the GPU backend");
            }
        }

        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
        self.tick_fps();
    }

    /// Expand the base KV6 sprite into the demo's instanced field: a
    /// green model + a recoloured red variant, checkerboarded across an
    /// N×N field (`ROXLAP_SPRITE_GRID`, default 16 ⇒ 256 instances).
    /// The red model is the `G`-carve target. Content only — the
    /// renderer builds the CPU draws + GPU registry from this.
    fn build_sprite_set(&self) -> Option<(SpriteSet, SpriteLayout)> {
        let mut models = Vec::new();
        let mut instances = Vec::new();
        let mut carve_model = None;

        if let Some(base) = self.sprites.first() {
            // Red variant: recolour every KV6 voxel (keep alpha, force red)
            // — applied once on the CPU so both backends agree.
            let mut red = base.clone();
            for v in &mut red.kv6.voxels {
                v.col = (v.col & 0xFF00_0000) | 0x00FF_0000;
            }
            models.push(base.clone()); // model 0: green
            models.push(red); // model 1: red (G-carve target)
            carve_model = Some(1);
            instances.extend(Self::coco_field_instances());
        }

        // Shoot-to-carve target (the dense blob): its model id is kept in
        // `carve_target_id` for the incremental shoot-to-carve refresh.
        let target_model = models.len();
        models.push(self.carve_target.sprite.clone());
        instances.push(SpriteInstanceDesc {
            model: target_model,
            pos: TARGET_WORLD,
        });

        // The spinner streams its own block models in dynamically via
        // `add_sprite_model` (no static slots here).

        if models.is_empty() {
            return None;
        }
        Some((
            SpriteSet {
                models,
                instances,
                carve_model,
            },
            SpriteLayout {
                carve_model: Some(target_model),
            },
        ))
    }

    /// Map the positional ids from a [`SceneRenderer::set_sprites`] back
    /// into the carve-target handle, and (re-)enable the spinner — a
    /// `set_sprites` wiped the dynamic layer, so it re-streams its models.
    fn map_sprite_ids(&mut self, ids: &[SpriteModelId], layout: &SpriteLayout) {
        self.carve_target_id = layout.carve_model.and_then(|i| ids.get(i).copied());
        self.spinner.reset();
    }

    /// The checkerboarded green/red `coco` field (model indices 0/1).
    fn coco_field_instances() -> Vec<SpriteInstanceDesc> {
        let n: i32 = std::env::var("ROXLAP_SPRITE_GRID")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4);
        let spacing = 40.0_f32;
        let mut instances = Vec::new();
        for iy in 0..n {
            for ix in 0..n {
                #[allow(clippy::cast_precision_loss)]
                let dx = (ix as f32 - (n as f32 - 1.0) * 0.5) * spacing;
                #[allow(clippy::cast_precision_loss)]
                let dy = (iy as f32 - (n as f32 - 1.0) * 0.5) * spacing;
                let model = usize::from((ix + iy) % 2 != 0); // 0 green, 1 red
                instances.push(SpriteInstanceDesc {
                    model,
                    pos: [dx, dy, 40.0],
                });
            }
        }
        instances
    }

    /// Cast the centre-screen ray and, if it hits the carve target,
    /// subtract a `SHOT_RADIUS` sphere (with a molten-gradient interior)
    /// and re-upload the sprite field. Bound to left-click while grabbed.
    fn fire(&mut self) {
        let Some(window) = self.window.as_ref() else {
            return;
        };
        let size = window.inner_size();
        let (cx, cy) = (f64::from(size.width) * 0.5, f64::from(size.height) * 0.5);
        let camera = self.scene.camera;
        let Some(ray) = self
            .renderer
            .as_ref()
            .and_then(|r| r.view_ray(&camera, cx, cy))
        else {
            return;
        };
        let Some(hit) = self
            .carve_target
            .raycast(ray.origin.to_array(), ray.dir.to_array())
        else {
            eprintln!("shot missed the carve target");
            return;
        };
        let removed = self.carve_target.carve(hit);
        eprintln!(
            "hit target at local ({}, {}, {}) — carved {removed} voxels",
            hit[0], hit[1], hit[2],
        );
        // GPU.12 incremental: refresh ONLY the carve target's model, not
        // the whole 256-instance sprite field. The instance set and every
        // other model are untouched. The handle came from `set_sprites`.
        if let (Some(renderer), Some(id)) = (self.renderer.as_mut(), self.carve_target_id) {
            renderer.refresh_sprite_model(id, &self.carve_target.sprite.kv6);
        }
    }

    /// Update camera position from the active input bits.
    fn tick_camera(&mut self, dt: f64) {
        let fast = if self.input.fast { FAST_MULT } else { 1.0 };
        let speed = MOVE_SPEED * fast * dt;

        let cam = &self.scene.camera;
        // Strafe / advance along camera-frame axes.
        let mut dx = 0.0;
        let mut dy = 0.0;
        let mut dz = 0.0;
        if self.input.forward {
            dx += cam.forward[0];
            dy += cam.forward[1];
            dz += cam.forward[2];
        }
        if self.input.back {
            dx -= cam.forward[0];
            dy -= cam.forward[1];
            dz -= cam.forward[2];
        }
        if self.input.right {
            // The "right" basis vector is screen-right — to strafe
            // right we move along it. Voxlap stores `right` as the
            // physical right vector (RH basis), so `+right` is correct.
            dx += cam.right[0];
            dy += cam.right[1];
            dz += cam.right[2];
        }
        if self.input.left {
            dx -= cam.right[0];
            dy -= cam.right[1];
            dz -= cam.right[2];
        }
        // World-frame vertical (Space / LShift): in voxlap z is
        // *down*, so `Space` should subtract from z (move up).
        if self.input.up {
            dz -= 1.0;
        }
        if self.input.down {
            dz += 1.0;
        }

        let mag2 = dx * dx + dy * dy + dz * dz;
        if mag2 > 0.0 {
            let inv = 1.0 / mag2.sqrt();
            let step = [dx * inv * speed, dy * inv * speed, dz * inv * speed];
            collision::slide_with_collision(&self.scene.scene, &mut self.scene.cam_pos, step);
            self.scene.refresh_camera();
        }
    }

    fn set_grab(&mut self, grab: bool) {
        let Some(window) = self.window.as_ref() else {
            return;
        };
        if grab {
            // Confined mode is enough for the scene demo; locked
            // mode requires extra winit support that's platform-
            // dependent.
            let _ = window
                .set_cursor_grab(CursorGrabMode::Confined)
                .or_else(|_| window.set_cursor_grab(CursorGrabMode::Locked));
            window.set_cursor_visible(false);
            self.grabbed = true;
        } else {
            let _ = window.set_cursor_grab(CursorGrabMode::None);
            window.set_cursor_visible(true);
            self.grabbed = false;
        }
    }

    /// Toggle the pick demo (`C`). Entering: build recoloured cursor +
    /// marker models, save the camera pose, jump to a top-down view,
    /// and release the cursor so `CursorMoved` reports absolute window
    /// positions. Leaving: restore the pose + the normal sprite field.
    fn toggle_pick_mode(&mut self) {
        self.pick_mode = !self.pick_mode;
        if self.pick_mode {
            // Cursor = cyan, placed markers = yellow. Recolour every
            // KV6 voxel (keep alpha) once, like `build_sprite_set`'s
            // red variant.
            let models = if let Some(base) = self.sprites.first() {
                let recolor = |rgb: u32| {
                    let mut s = base.clone();
                    for v in &mut s.kv6.voxels {
                        v.col = (v.col & 0xFF00_0000) | rgb;
                    }
                    s
                };
                vec![recolor(0x0000_FFFF), recolor(0x00FF_FF00)]
            } else {
                eprintln!("pick mode: no base sprite loaded — cursor disabled");
                Vec::new()
            };
            self.pick_models = models;
            self.pick_saved_pose = Some((self.scene.cam_pos, self.scene.yaw, self.scene.pitch));
            self.scene.cam_pos = PICK_CAM_POS;
            self.scene.yaw = std::f64::consts::FRAC_PI_2; // look +y
            self.scene.pitch = PICK_CAM_PITCH;
            self.scene.refresh_camera();
            self.set_grab(false);
            eprintln!(
                "pick mode ON — move the mouse to place the cursor, left-click to drop a \
                 marker, C to exit",
            );
        } else {
            if let Some((pos, yaw, pitch)) = self.pick_saved_pose.take() {
                self.scene.cam_pos = pos;
                self.scene.yaw = yaw;
                self.scene.pitch = pitch;
                self.scene.refresh_camera();
            }
            // Restore the normal sprite field (compute before borrowing
            // the renderer to keep the borrows disjoint), then remap the
            // carve-target + spinner model handles.
            if let Some((set, layout)) = self.build_sprite_set() {
                if let Some(ids) = self.renderer.as_mut().map(|r| r.set_sprites(&set)) {
                    self.map_sprite_ids(&ids, &layout);
                }
            }
            eprintln!("pick mode OFF");
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_title("roxlap-scene-demo")
            .with_inner_size(LogicalSize::new(WIDTH, HEIGHT));
        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .expect("winit: create_window"),
        );

        // RF: one unified renderer. `ROXLAP_GPU=1` selects the GPU
        // backend; roxlap-render falls back to the CPU softbuffer path
        // automatically on any WGPU init failure.
        let want_gpu = std::env::var_os("ROXLAP_GPU").is_some_and(|v| v != "0" && !v.is_empty());
        let opts = RenderOptions {
            want_gpu,
            cpu_max_grid_vsid: MAX_GRID_VSID,
            cpu_render_threads: RENDER_THREADS,
            ..RenderOptions::default()
        };
        let init_size = window.inner_size();
        let mut renderer =
            SceneRenderer::new(window.clone(), (init_size.width, init_size.height), &opts);

        self.title_base = if let Some(info) = renderer.adapter_info() {
            eprintln!("roxlap-render: GPU backend — {info}");
            format!("roxlap-scene-demo (GPU: {info})")
        } else {
            eprintln!("roxlap-render: CPU backend");
            "roxlap-scene-demo (CPU)".to_string()
        };
        window.set_title(&self.title_base);

        // Sky panorama (GPU shader sky sampling; CPU samples engine.sky()).
        match load_png_sky_rgba(SKY_PNG) {
            Ok((rgba, w, h)) => {
                renderer.set_sky_panorama(&rgba, w, h);
                eprintln!("roxlap-render: sky panorama uploaded ({w}×{h})");
            }
            Err(e) => eprintln!("roxlap-render: sky decode failed ({e})"),
        }

        // GPU scene-grid LOD scan distance (world units). Default 64
        // (matches the CPU opticast `mip_scan_dist`).
        // `ROXLAP_GPU_MIP_SCAN_DIST=0` disables LOD; crank down to force
        // coarse mips close, or up to push banding out.
        if let Some(msd) = std::env::var("ROXLAP_GPU_MIP_SCAN_DIST")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
        {
            eprintln!("roxlap-render: scene mip scan dist = {msd}");
            self.gpu_mip_scan_dist = msd;
        }

        // Sprites: the demo's instanced field (green + red models).
        // `ROXLAP_GPU_NO_SPRITES=1` skips them (FPS isolation).
        let no_sprites = std::env::var_os("ROXLAP_GPU_NO_SPRITES").is_some_and(|v| v != "0");
        if no_sprites {
            eprintln!("roxlap-render: ROXLAP_GPU_NO_SPRITES — sprites disabled");
        } else if let Some((set, layout)) = self.build_sprite_set() {
            eprintln!(
                "roxlap-render: {} sprite instances, {} models \
                 (click to grab, then left-click shoots spheres out of the blob ahead; \
                 'G' carves the red model; a colour-sphere spinner streams in/out each frame)",
                set.instances.len(),
                set.models.len(),
            );
            let ids = renderer.set_sprites(&set);
            self.map_sprite_ids(&ids, &layout);
        }

        // Animated KFA sprite (the swinging arm) — registered once;
        // re-posed each frame in `redraw`. Shares the `no_sprites` knob.
        if !no_sprites && !self.kfa.is_empty() {
            eprintln!("roxlap-render: KFA sprite registered (animsprite-driven arm)");
            renderer.set_kfa_sprites(&mut self.kfa);
        }

        // VCL.7 — a second coco via the attachment runtime, with a
        // procedural flame voxel clip on its swinging arm (multi-attachment
        // + clip playback). Must come after `set_sprites` (which resets the
        // dynamic/clip/character layers).
        if !no_sprites {
            if let Some(ch) = flame_character() {
                self.flame_char = Some(renderer.add_character(&ch, Some(0)));
                eprintln!(
                    "roxlap-render: flame character registered (add_character + clip attachment)"
                );
            }
        }

        // HUD: bridge winit events into egui. `&*window` provides the
        // display handle egui-winit needs for clipboard/IME.
        self.egui_state = Some(egui_winit::State::new(
            self.egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &*window,
            None,
            None,
            Some(2048),
        ));

        // Upload the demo reference image once (a generated checkerboard +
        // axis tint), so the `I`-toggle draw is a cheap per-frame call.
        let (rgba, iw, ih) = make_reference_image();
        self.image_id = Some(renderer.upload_image(&rgba, iw, ih));

        self.renderer = Some(renderer);
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        self.handle_hud_event(&event);
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(new_size) => {
                // GPU.1: the swapchain must follow physical resizes;
                // softbuffer resizes lazily inside `redraw`, so the
                // CPU branch needs nothing here.
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.resize(new_size.width, new_size.height);
                }
            }
            WindowEvent::RedrawRequested => self.redraw(),
            // Pick mode tracks the free cursor; FPS mode ignores absolute
            // moves (it uses `DeviceEvent::MouseMotion` deltas instead).
            WindowEvent::CursorMoved { position, .. } => {
                self.mouse_px = (position.x, position.y);
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                // When the image sprite is shown, a left click reports the
                // texel under the cursor via `pick_image` (alpha- and
                // occlusion-aware) — the demiurg eyedropper/probe path.
                if self.images_on {
                    if let (Some(renderer), Some(id)) = (self.renderer.as_ref(), self.image_id) {
                        let sprite = demo_image_sprite(id);
                        let (mx, my) = self.mouse_px;
                        match renderer.pick_image(&self.scene.camera, mx, my, &[sprite]) {
                            Some(h) => eprintln!(
                                "image hit: texel ({}, {}) uv ({:.3}, {:.3}) @ dist {:.1}",
                                h.texel.0, h.texel.1, h.uv[0], h.uv[1], h.t,
                            ),
                            None => eprintln!("image hit: none (miss / transparent / occluded)"),
                        }
                    }
                }
                if self.pick_mode {
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let (px, py) = (self.mouse_px.0 as u32, self.mouse_px.1 as u32);
                    // One library call: unproject → depth read → owning
                    // grid + grid-local voxel. Sky clicks (None) fall
                    // back to the ground-plane hover cursor.
                    let hit = self
                        .renderer
                        .as_ref()
                        .and_then(|r| r.pick(&self.scene.scene, &self.scene.camera, px, py));
                    match hit {
                        Some(h) => {
                            self.placed.push(h.world);
                            eprintln!(
                                "surface [{:.1}, {:.1}, {:.1}] → grid #{} voxel ({}, {}, {}) \
                                 ({} placed)",
                                h.world[0],
                                h.world[1],
                                h.world[2],
                                h.grid.raw(),
                                h.voxel.x,
                                h.voxel.y,
                                h.voxel.z,
                                self.placed.len(),
                            );
                        }
                        None => {
                            let world = self.cursor_world;
                            self.placed.push(world);
                            eprintln!(
                                "ground-plane [{:.1}, {:.1}, {:.1}] ({} placed)",
                                world[0],
                                world[1],
                                world[2],
                                self.placed.len(),
                            );
                        }
                    }
                } else if self.grabbed {
                    // FPS mode: a click is a shot — carve a sphere out of
                    // the target where the crosshair points.
                    self.fire();
                } else {
                    self.set_grab(true);
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state,
                        ..
                    },
                ..
            } => {
                let pressed = state == ElementState::Pressed;
                match code {
                    KeyCode::KeyW => self.input.forward = pressed,
                    KeyCode::KeyS => self.input.back = pressed,
                    KeyCode::KeyA => self.input.left = pressed,
                    KeyCode::KeyD => self.input.right = pressed,
                    KeyCode::Space => self.input.up = pressed,
                    KeyCode::ShiftLeft => self.input.down = pressed,
                    KeyCode::ControlLeft => self.input.fast = pressed,
                    KeyCode::KeyF if pressed => {
                        // Defer the actual write until after redraw
                        // composites this frame, so the captured PPM
                        // matches what's on screen.
                        self.capture_pending = true;
                    }
                    // S5.2: `R` toggles the ship grid's continuous
                    // Z-axis spin. Pressed-edge only — release is
                    // ignored so the toggle survives the key going up.
                    KeyCode::KeyR if pressed => {
                        self.scene.spin_enabled = !self.scene.spin_enabled;
                        eprintln!(
                            "ship spin = {}",
                            if self.scene.spin_enabled { "ON" } else { "OFF" }
                        );
                    }
                    // GPU.12: `G` structurally carves the next z-layer
                    // off the forked ("red") sprite model at runtime,
                    // rebuilds its LOD mips, and re-uploads — the base
                    // (non-red) instances stay intact (copy-on-modify).
                    // GPU backend only; a no-op on the CPU path.
                    KeyCode::KeyG if pressed => {
                        if let Some(renderer) = self.renderer.as_mut() {
                            let removed = renderer.carve_active_sprite();
                            if removed > 0 {
                                eprintln!("carved sprite z-layer ({removed} voxels)");
                            }
                        }
                    }
                    // S6.6: `B` toggles the marker pillars'
                    // LOD configuration between always-Near
                    // (default — full voxel, pre-S6.6 behaviour)
                    // and the tuned billboards split (closer
                    // pillars Near, farther pillars Far via S6.3's
                    // billboard impostor blit). Pressed-edge only.
                    KeyCode::KeyB if pressed => {
                        let on = self.scene.toggle_billboards_lod();
                        eprintln!("S6 billboards = {}", if on { "ON" } else { "OFF" });
                    }
                    // Pick demo: top-down camera + mouse-driven cursor
                    // placement (screen→world unproject prototype).
                    KeyCode::KeyC if pressed => self.toggle_pick_mode(),
                    // L3.0: toggle the depth-tested debug line overlay.
                    KeyCode::KeyL if pressed => {
                        self.lines_on = !self.lines_on;
                        eprintln!("debug lines = {}", self.lines_on);
                    }
                    // Toggle the world-placed 2D image sprite (the
                    // `draw_images` reference-quad primitive).
                    KeyCode::KeyI if pressed => {
                        self.images_on = !self.images_on;
                        eprintln!("image sprite = {}", self.images_on);
                    }
                    // S7.6: `T` (telemetry) prints chunk count +
                    // pending count for each streaming-enabled
                    // grid. No-op when not in streaming mode.
                    KeyCode::KeyT if pressed && self.scene.streaming_enabled => {
                        for (id, grid) in self.scene.scene.grids() {
                            eprintln!(
                                "grid #{id} chunks={chunks} pending={pending} radius={r_active:.0}/{r_evict:.0}",
                                id = id.raw(),
                                chunks = grid.chunk_count(),
                                pending = grid.pending_gen.len(),
                                r_active = grid.stream_radius.r_active,
                                r_evict = grid.stream_radius.r_evict,
                            );
                        }
                    }
                    // GPU.13.0: `H` teleports to a high-altitude
                    // top-down vantage (and back) for the chunk-AABB
                    // early-out FPS A/B. From way up, horizon + sky
                    // rays cross many empty chunks before exiting the
                    // grid's occupied box — exactly what 13.0 culls.
                    // Toggles with the pre-teleport pose so the user
                    // can compare framerate at the two poses.
                    KeyCode::KeyH if pressed => {
                        if let Some((pos, yaw, pitch)) = self.saved_pose.take() {
                            self.scene.cam_pos = pos;
                            self.scene.yaw = yaw;
                            self.scene.pitch = pitch;
                            eprintln!("camera restored to {pos:?}");
                        } else {
                            self.saved_pose =
                                Some((self.scene.cam_pos, self.scene.yaw, self.scene.pitch));
                            // High above the centred ground (z-down →
                            // very negative z = high up), looking ~40°
                            // down so terrain fills the lower frame and
                            // the horizon/sky fills the upper.
                            self.scene.cam_pos = [0.0, 0.0, -800.0];
                            self.scene.pitch = 0.7; // ~40° down (z-down)
                            eprintln!(
                                "camera → high-altitude top-down {:?} (press H again to return)",
                                self.scene.cam_pos
                            );
                        }
                        self.scene.refresh_camera();
                    }
                    // `+` / `=` (same key on US layout, with or without Shift)
                    // and the numpad `+` bump scan distance up by
                    // SCAN_DIST_STEP. Both keys handled so the
                    // shift-modifier doesn't trip the binding.
                    KeyCode::Equal | KeyCode::NumpadAdd if pressed => {
                        self.scan_dist = (self.scan_dist + SCAN_DIST_STEP).min(SCAN_DIST_MAX);
                        eprintln!("scan_dist = {}", self.scan_dist);
                    }
                    // `-` and numpad `-` bump down.
                    KeyCode::Minus | KeyCode::NumpadSubtract if pressed => {
                        self.scan_dist = (self.scan_dist - SCAN_DIST_STEP).max(SCAN_DIST_MIN);
                        eprintln!("scan_dist = {}", self.scan_dist);
                    }
                    KeyCode::Escape if pressed => {
                        if self.grabbed {
                            self.set_grab(false);
                        } else {
                            event_loop.exit();
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if !self.grabbed {
            return;
        }
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            self.scene.yaw += dx * MOUSE_SENS;
            self.scene.pitch =
                (self.scene.pitch + dy * MOUSE_SENS).clamp(-PITCH_LIMIT, PITCH_LIMIT);
            self.scene.refresh_camera();
        }
    }
}

/// Save the current framebuffer + camera state to
/// `roxlap-scene-capture.{txt,ppm}` for off-line debugging. Same
/// shape as roxlap-host's `write_capture` so the existing
/// `roxlap-oracle find-hairlines` tooling can read it (with the
/// `--ours <path>` override).
///
/// `.txt` is human-readable `key = value` lines; `.ppm` is a
/// binary P6 RGB dump. Both files overwrite on each call.
#[allow(clippy::too_many_arguments)]
fn write_capture(
    buffer: &[u32],
    width: u32,
    height: u32,
    cam: &roxlap_core::Camera,
    yaw: f64,
    pitch: f64,
    ship_angles: [f64; 3],
    spin_enabled: bool,
) -> std::io::Result<()> {
    use std::io::Write;
    let txt = format!(
        "# roxlap-scene-demo capture — generated by F hotkey\n\
         width = {width}\n\
         height = {height}\n\
         pos.x = {}\n\
         pos.y = {}\n\
         pos.z = {}\n\
         yaw = {}\n\
         pitch = {}\n\
         # Camera basis at the time of capture (host yaw/pitch convention):\n\
         right = [{}, {}, {}]\n\
         down = [{}, {}, {}]\n\
         forward = [{}, {}, {}]\n\
         # Ship grid rotation at the time of capture (S5.2). Reproduce by\n\
         # setting ship_angles[0..3] before render and setting\n\
         # spin_enabled = false (so the angles don't advance further).\n\
         ship_angle.x = {}\n\
         ship_angle.y = {}\n\
         ship_angle.z = {}\n\
         ship_spin_enabled = {}\n",
        cam.pos[0],
        cam.pos[1],
        cam.pos[2],
        yaw,
        pitch,
        cam.right[0],
        cam.right[1],
        cam.right[2],
        cam.down[0],
        cam.down[1],
        cam.down[2],
        cam.forward[0],
        cam.forward[1],
        cam.forward[2],
        ship_angles[0],
        ship_angles[1],
        ship_angles[2],
        spin_enabled,
    );
    std::fs::write("roxlap-scene-capture.txt", txt)?;

    let header = format!("P6\n{width} {height}\n255\n");
    let mut bytes = Vec::with_capacity(header.len() + (width as usize) * (height as usize) * 3);
    bytes.extend_from_slice(header.as_bytes());
    for &px in buffer {
        bytes.push(((px >> 16) & 0xff) as u8); // R
        bytes.push(((px >> 8) & 0xff) as u8); // G
        bytes.push((px & 0xff) as u8); // B
    }
    let mut f = std::fs::File::create("roxlap-scene-capture.ppm")?;
    f.write_all(&bytes)?;
    Ok(())
}

#[cfg(test)]
mod character_tests {
    use super::{authored_character, build_kfa, flame_character, flame_clip};
    use roxlap_formats::character;

    /// VCL.7 — the procedural flame clip decodes to a looping flipbook with
    /// real voxels in every frame.
    #[test]
    fn flame_clip_decodes() {
        let decoded = flame_clip().decode().expect("flame clip decodes");
        assert_eq!(decoded.frame_count(), 8);
        assert!(
            decoded.frames.iter().all(|f| !f.colors.is_empty()),
            "every flame frame has voxels"
        );
    }

    /// VCL.7 — the flame character carries the clip and a multi-attachment
    /// arm bone ([static mesh, flame clip]).
    #[test]
    fn flame_character_arm_is_multi_attachment() {
        let ch = flame_character().expect("coco.kv6 parses");
        assert_eq!(ch.voxel_clips.len(), 1);
        let arm = &ch.bones[1].attachments;
        assert_eq!(arm.len(), 2, "arm has the mesh + the flame clip");
        assert!(matches!(arm[0].target, character::MeshRef::Static(0)));
        assert!(matches!(arm[1].target, character::MeshRef::Clip(0)));
    }

    // The authored character writes to disk and reloads byte-equal — the
    // path `ROXLAP_RKC_DUMP` then `ROXLAP_RKC` exercise.
    #[test]
    fn rkc_disk_round_trip() {
        let c = authored_character().expect("coco.kv6 parses");
        let bytes = character::serialize(&c);
        let dir = std::env::temp_dir();
        let path = dir.join(format!("roxlap-demo-{}.rkc", std::process::id()));
        std::fs::write(&path, &bytes).expect("write .rkc");
        let read = std::fs::read(&path).expect("read .rkc");
        let parsed = character::parse(&read).expect("parse .rkc from disk");
        assert_eq!(
            character::serialize(&parsed),
            bytes,
            "disk round-trip byte-equal"
        );
        let sprite = parsed.to_kfa_sprite(Some(0));
        assert_eq!(sprite.limbs.len(), 2);
        std::fs::remove_file(&path).ok();
    }

    // The lossy .kfa export keeps the skeleton + the selected clip.
    #[test]
    fn kfa_export_keeps_skeleton_and_clip() {
        let c = authored_character().expect("coco.kv6 parses");
        let kfa = c.to_kfa(Some(0), "coco.kv6");
        assert_eq!(kfa.kv6_name, b"coco.kv6");
        assert_eq!(kfa.hinges.len(), 2);
        assert!(!kfa.frmval.is_empty());
        // It's a valid .kfa on disk.
        let bytes = roxlap_formats::kfa::serialize(&kfa);
        assert!(roxlap_formats::kfa::parse(&bytes).is_ok());
    }

    // The default (no env vars) build path produces one animated sprite.
    #[test]
    fn build_kfa_default_builds_one_sprite() {
        let sprites = build_kfa();
        assert_eq!(sprites.len(), 1);
    }
}

#[cfg(test)]
mod pick_tests {
    use super::plane_hit;

    // The ray→world unproject (per-backend projection) is tested in
    // roxlap-render; here we cover only the demo's ground-plane
    // intersect used for the hover cursor.

    // A straight-down ray from above the plane hits directly below the
    // camera, on the plane.
    #[test]
    fn straight_down_hits_under_camera() {
        let p = plane_hit([10.0, 20.0, -100.0], [0.0, 0.0, 1.0], 0.0).expect("hits the plane");
        assert!((p[0] - 10.0).abs() < 1e-3, "x under camera, got {}", p[0]);
        assert!((p[1] - 20.0).abs() < 1e-3, "y under camera, got {}", p[1]);
        assert!((p[2] - 0.0).abs() < 1e-3, "on the plane, got {}", p[2]);
    }

    // A slanted ray scales its lateral offset by the t to the plane:
    // dir=(2,1,1) from z=-100 to z=0 → t=100 → (+200, +100).
    #[test]
    fn slanted_ray_scales_by_t() {
        let p = plane_hit([0.0, 0.0, -100.0], [2.0, 1.0, 1.0], 0.0).expect("hits the plane");
        assert!((p[0] - 200.0).abs() < 1e-2, "x, got {}", p[0]);
        assert!((p[1] - 100.0).abs() < 1e-2, "y, got {}", p[1]);
    }

    // Parallel to the plane, or pointing away from it → None.
    #[test]
    fn degenerate_rays_are_none() {
        assert!(
            plane_hit([0.0, 0.0, -100.0], [1.0, 0.0, 0.0], 0.0).is_none(),
            "parallel to plane → None",
        );
        assert!(
            plane_hit([0.0, 0.0, -100.0], [0.0, 0.0, -1.0], 0.0).is_none(),
            "pointing away → None",
        );
    }
}
