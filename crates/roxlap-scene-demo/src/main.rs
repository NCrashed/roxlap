//! roxlap-scene-demo — interactive showcase of the scene-graph
//! engine. See `README.md` for the controls + the demo's
//! evolution roadmap as the scene-graph substages land.
//!
//! The demo is a thin host ([`host::Host`]) that drives one
//! menu-selectable [`scene_api::DemoScene`] at a time (`Tab` opens the
//! scene menu). The host owns the window, [`SceneRenderer`], egui HUD,
//! shared fly-camera, and FPS; each scene in [`scenes`] owns its world
//! content + per-scene update / input / render. This module keeps the
//! content helpers the scenes share (sprite / character / clip / picking
//! / primitive builders) plus `fn main`.

mod collision;
mod host;
mod kv6_sprite;
mod markers;
#[cfg(test)]
mod repro;
mod scene;
// DS.0–DS.1 — demo-scene API + scenes + the thin host.
mod scene_api;
mod scenes;
mod ship;
mod terrain;

use roxlap_formats::character::{self, Attachment, Bone, Character, Clip, ClipData};
use roxlap_formats::kfa::{Hinge, Point3, Seq};
use roxlap_formats::kv6::Kv6;
use roxlap_formats::sprite::Sprite;
use roxlap_formats::xform::BoneXform;
use roxlap_render::{
    DynSpriteTransform, ImageFacing, ImageId, ImageSprite, KfaSprite, Line3, LoopMode,
    SceneRenderer, SpriteInstanceId, SpriteModelId, VoxelClip, VoxelFrame,
};
use winit::event_loop::{ControlFlow, EventLoop};

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

/// Pick-demo: horizontal ground reference plane the mouse cursor
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

// Screen→world picking. The ray-per-backend unproject + depth read +
// grid resolution now live in the LIBRARY (`SceneRenderer::pixel_ray`
// / `pick`, `Scene::resolve_voxel`); the demo only keeps the
// ground-plane intersect below for the continuous hover cursor.

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
    let mut app = host::Host::new();
    event_loop.run_app(&mut app).expect("winit: run_app");
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
