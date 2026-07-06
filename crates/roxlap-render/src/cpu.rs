//! CPU backend — the `roxlap-core` per-pixel DDA renderer presented
//! via `softbuffer` (native) / a WebGL2 blit (wasm).
//!
//! Owns the software surface, the framebuffer + z-buffer, and the
//! multi-grid compositor ([`render_scene_composed`]), plus the CPU
//! sides of sprites, clips, billboards, materials, dynamic lighting
//! and the post pipeline.

#[cfg(not(target_arch = "wasm32"))]
use std::num::NonZeroU32;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;

use roxlap_core::camera_math;
use roxlap_core::dda_sprite::{
    draw_sprite_dense_shaded, ClipFlipbook, SpriteDense, SpriteOccluder, SpriteShade,
};
use roxlap_core::kfa_draw::solve_kfa_limbs;
use roxlap_core::render_sky_fill;
use roxlap_core::Camera;
use roxlap_core::{CompositeOccluder, WorldOccluder};
use roxlap_formats::kv6::Kv6;

use roxlap_formats::sprite::Sprite;
use roxlap_formats::voxel_clip::{DecodedClip, VoxelFrame};
use roxlap_scene::occluder::SceneOccluder;
use roxlap_scene::render::{
    render_scene_composed_with_materials_scratch, CpuFog, SceneRenderScratch,
};
use roxlap_scene::Scene;

#[cfg(not(target_arch = "wasm32"))]
use crate::{DynDisplay, DynWindow, HasDisplayHandle, HasWindowHandle};
use crate::{
    DynSpriteTransform, FrameParams, KfaSprite, Line3, QuadDraw, RenderOptions, Rgb, SpriteSet,
};

/// An empty (zero-voxel) KV6 — the placeholder a removed CPU model
/// template holds so its slot stays put while its geometry is freed.
fn empty_kv6() -> Kv6 {
    Kv6 {
        xsiz: 0,
        ysiz: 0,
        zsiz: 0,
        xpiv: 0.0,
        ypiv: 0.0,
        zpiv: 0.0,
        voxels: Vec::new(),
        xlen: Vec::new(),
        ylen: Vec::new(),
        palette: None,
    }
}

/// Near plane (camera-forward distance, voxel units) below which a
/// [`Line3`] endpoint is clipped — keeps the pinhole divide finite and
/// stops points behind the camera from wrapping onto the screen.
const NEAR_Z: f32 = 0.0625;

/// Depth-test slack (perpendicular distance) so a line resting on the
/// surface it traces doesn't z-fight against that surface.
const DEPTH_BIAS: f32 = 0.5;

/// Alpha-blend `rgb` (`0x__RRGGBB`) over `dst` (`0x00RRGGBB`) by `alpha`
/// (`0..=255`). Returns `0x00RRGGBB`, matching the framebuffer packing.
fn blend_rgb(dst: u32, rgb: u32, alpha: u32) -> u32 {
    if alpha >= 255 {
        return rgb & 0x00ff_ffff;
    }
    let ia = 255 - alpha;
    let r = (((rgb >> 16) & 0xff) * alpha + ((dst >> 16) & 0xff) * ia) / 255;
    let g = (((rgb >> 8) & 0xff) * alpha + ((dst >> 8) & 0xff) * ia) / 255;
    let b = ((rgb & 0xff) * alpha + (dst & 0xff) * ia) / 255;
    (r << 16) | (g << 8) | b
}

/// A retained RGBA8 image-sprite texture (straight alpha, row-major).
/// Sampled nearest-neighbour by [`CpuBackend::draw_images`].
struct CpuImage {
    rgba: Vec<u8>,
    width: u32,
    height: u32,
}

impl CpuImage {
    /// Nearest-neighbour fetch at normalised `(u, v)` (clamped to the
    /// edge) → `(r, g, b, a)` bytes.
    fn sample(&self, u: f32, v: f32) -> (u32, u32, u32, u32) {
        let w = self.width.max(1);
        let h = self.height.max(1);
        // `as i32` truncates toward zero; clamp keeps us in-bounds for
        // UVs that drift just outside [0, 1] at a quad edge.
        let tx = ((u * w as f32) as i32).clamp(0, w as i32 - 1) as u32;
        let ty = ((v * h as f32) as i32).clamp(0, h as i32 - 1) as u32;
        let idx = ((ty * w + tx) * 4) as usize;
        (
            u32::from(self.rgba[idx]),
            u32::from(self.rgba[idx + 1]),
            u32::from(self.rgba[idx + 2]),
            u32::from(self.rgba[idx + 3]),
        )
    }
}

/// A near-clipped quad/triangle vertex in camera space (`cam` =
/// `(right, down, forward)` components) carrying its texture `uv`.
#[derive(Clone, Copy)]
struct ClipVert {
    cam: [f32; 3],
    uv: [f32; 2],
}

/// A projected vertex ready for the perspective-correct raster: screen
/// `(sx, sy)`, the linear-in-screen-space `inv_w = 1/forward`, and the
/// pre-divided `u/forward`, `v/forward` (also linear in screen space).
#[derive(Clone, Copy)]
struct ScreenVert {
    sx: f32,
    sy: f32,
    inv_w: f32,
    su: f32,
    sv: f32,
}

/// Clip a convex camera-space polygon against the near plane
/// (`forward >= NEAR_Z`) with Sutherland–Hodgman, interpolating UVs at
/// each crossing. Keeps the pinhole divide finite and drops geometry
/// behind the camera. Returns `< 3` vertices when fully clipped.
fn clip_near(poly: &[ClipVert]) -> Vec<ClipVert> {
    let n = poly.len();
    let mut out: Vec<ClipVert> = Vec::with_capacity(n + 1);
    for i in 0..n {
        let cur = poly[i];
        let prev = poly[(i + n - 1) % n];
        let cur_in = cur.cam[2] >= NEAR_Z;
        let prev_in = prev.cam[2] >= NEAR_Z;
        if cur_in != prev_in {
            let t = (NEAR_Z - prev.cam[2]) / (cur.cam[2] - prev.cam[2]);
            out.push(ClipVert {
                cam: [
                    prev.cam[0] + (cur.cam[0] - prev.cam[0]) * t,
                    prev.cam[1] + (cur.cam[1] - prev.cam[1]) * t,
                    NEAR_Z,
                ],
                uv: [
                    prev.uv[0] + (cur.uv[0] - prev.uv[0]) * t,
                    prev.uv[1] + (cur.uv[1] - prev.uv[1]) * t,
                ],
            });
        }
        if cur_in {
            out.push(cur);
        }
    }
    out
}

/// Pinhole-project a near-clipped camera-space vertex to a [`ScreenVert`],
/// pre-dividing the UVs by `forward` for the perspective-correct raster.
fn project_clip(v: ClipVert, hx: f32, hy: f32, hz: f32) -> ScreenVert {
    let inv_w = 1.0 / v.cam[2];
    ScreenVert {
        sx: hx + v.cam[0] * hz * inv_w,
        sy: hy + v.cam[1] * hz * inv_w,
        inv_w,
        su: v.uv[0] * inv_w,
        sv: v.uv[1] * inv_w,
    }
}

/// Rasterise one perspective-correct textured triangle into `fb`,
/// depth-tested against `zb` (forward distance, smaller = closer). The
/// per-vertex `inv_w` / `su` / `sv` interpolate linearly in screen space;
/// the true `u, v, forward` are recovered per pixel by dividing by the
/// interpolated `inv_w`. Nearest sampling, straight-alpha `tint`, over-blend.
#[allow(clippy::too_many_arguments)]
fn fill_textured_tri(
    fb: &mut [u32],
    zb: &[f32],
    w: u32,
    h: u32,
    v0: &ScreenVert,
    v1: &ScreenVert,
    v2: &ScreenVert,
    image: &CpuImage,
    tint: u32,
    depth_test: bool,
    alpha_cutoff: f32,
) {
    // Texels with alpha below this are discarded outright (crisp edges).
    let cutoff_u8 = (alpha_cutoff.clamp(0.0, 1.0) * 255.0) as u32;
    // Signed area (== barycentric denominator); skip degenerate slivers.
    let det = (v1.sx - v0.sx) * (v2.sy - v0.sy) - (v2.sx - v0.sx) * (v1.sy - v0.sy);
    if det.abs() < 1e-6 {
        return;
    }
    let inv_det = 1.0 / det;

    let (wi, hi) = (w as i32, h as i32);
    let minx = v0.sx.min(v1.sx).min(v2.sx).floor().max(0.0) as i32;
    let maxx = v0.sx.max(v1.sx).max(v2.sx).ceil().min(wi as f32 - 1.0) as i32;
    let miny = v0.sy.min(v1.sy).min(v2.sy).floor().max(0.0) as i32;
    let maxy = v0.sy.max(v1.sy).max(v2.sy).ceil().min(hi as f32 - 1.0) as i32;
    if minx > maxx || miny > maxy {
        return;
    }

    let tint_a = (tint >> 24) & 0xff;
    let tint_r = (tint >> 16) & 0xff;
    let tint_g = (tint >> 8) & 0xff;
    let tint_b = tint & 0xff;

    for py in miny..=maxy {
        let fy = py as f32 + 0.5;
        for px in minx..=maxx {
            let fx = px as f32 + 0.5;
            // Barycentric weights (signed-area form; valid for both
            // windings since each term carries `det`'s sign).
            let b0 = ((v1.sy - v2.sy) * (fx - v2.sx) + (v2.sx - v1.sx) * (fy - v2.sy)) * inv_det;
            let b1 = ((v2.sy - v0.sy) * (fx - v2.sx) + (v0.sx - v2.sx) * (fy - v2.sy)) * inv_det;
            let b2 = 1.0 - b0 - b1;
            // Small epsilon so shared edges between the two triangles
            // don't leave a 1px gap.
            if b0 < -1e-4 || b1 < -1e-4 || b2 < -1e-4 {
                continue;
            }

            let inv_w = b0 * v0.inv_w + b1 * v1.inv_w + b2 * v2.inv_w;
            if inv_w <= 0.0 {
                continue;
            }
            let fwd = 1.0 / inv_w; // forward distance — the z-buffer metric

            let idx = (py as usize) * (w as usize) + (px as usize);
            if depth_test && fwd > zb[idx] + DEPTH_BIAS {
                continue; // occluded by nearer rendered geometry
            }

            let u = (b0 * v0.su + b1 * v1.su + b2 * v2.su) * fwd;
            let v = (b0 * v0.sv + b1 * v1.sv + b2 * v2.sv) * fwd;
            let (tr, tg, tb, ta) = image.sample(u, v);
            if ta < cutoff_u8 {
                continue; // below the alpha cutoff — discard, don't blend
            }

            // Combine texel alpha with the tint's alpha byte.
            let alpha = ta * tint_a / 255;
            if alpha == 0 {
                continue;
            }
            let rgb =
                ((tr * tint_r / 255) << 16) | ((tg * tint_g / 255) << 8) | (tb * tint_b / 255);
            fb[idx] = blend_rgb(fb[idx], rgb, alpha);
        }
    }
}

/// The CPU backend's framebuffer presenter. Native blits into a
/// `softbuffer` window surface; wasm uploads to a WebGL2 texture +
/// fullscreen quad on the canvas (no softbuffer in the browser).
#[cfg(not(target_arch = "wasm32"))]
type Presenter = softbuffer::Surface<Arc<DynDisplay>, Arc<DynWindow>>;
#[cfg(target_arch = "wasm32")]
type Presenter = crate::cpu_blit::WebGlBlit;

/// World-space view-ray direction (un-normalised) for window pixel
/// `(x, y)` under the CPU opticast projection (voxlap `setcamera`):
/// `(x − hx)·right + (y − hy)·down + hz·forward` — `camera_math`'s
/// `corn[0]` plus the per-pixel `right`/`down` steps. Standalone so
/// it's unit-testable without a window.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub(crate) fn setcamera_pixel_ray(
    right: [f64; 3],
    down: [f64; 3],
    forward: [f64; 3],
    x: f64,
    y: f64,
    hx: f32,
    hy: f32,
    hz: f32,
) -> [f64; 3] {
    let (a, b, c) = (x - f64::from(hx), y - f64::from(hy), f64::from(hz));
    [
        a * right[0] + b * down[0] + c * forward[0],
        a * right[1] + b * down[1] + c * forward[1],
        a * right[2] + b * down[2] + c * forward[2],
    ]
}

/// RP.1 — box-average one `s × s` march block into a single logical pixel.
/// `fb` is the march framebuffer (`0x00RRGGBB`, row stride `march_w`); the
/// block's top-left is `(lx·s, ly·s)`. Per-channel round-to-nearest via integer
/// `(sum + n/2) / n` (matches round-half-away-from-zero). `s == 1` is the
/// identity (returns the source pixel unchanged).
fn downfilter_pixel(fb: &[u32], march_w: usize, lx: usize, ly: usize, s: usize) -> u32 {
    let (mut ar, mut ag, mut ab) = (0u32, 0u32, 0u32);
    for j in 0..s {
        let row = (ly * s + j) * march_w;
        for i in 0..s {
            let px = fb[row + lx * s + i];
            ar += (px >> 16) & 0xff;
            ag += (px >> 8) & 0xff;
            ab += px & 0xff;
        }
    }
    let n = (s * s) as u32;
    let half = n / 2;
    (((ar + half) / n) << 16) | (((ag + half) / n) << 8) | ((ab + half) / n)
}

/// RP.2 — dither threshold in `[0, 1)` for the logical pixel `(x, y)`.
/// [`DitherMode::None`] returns `0.5` so `floor(scaled + 0.5)` is a plain
/// round-to-nearest. Bayer is the classic `4×4` ordered matrix; `BlueNoise`
/// is interleaved-gradient noise (texture-free, non-repeating).
fn dither_offset(mode: crate::DitherMode, x: usize, y: usize) -> f32 {
    match mode {
        crate::DitherMode::None => 0.5,
        crate::DitherMode::Bayer4x4 => {
            const B: [u8; 16] = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5];
            (f32::from(B[(y % 4) * 4 + (x % 4)]) + 0.5) / 16.0
        }
        crate::DitherMode::BlueNoise => {
            // Jimenez interleaved-gradient noise.
            #[allow(clippy::cast_precision_loss)]
            let f = (x as f32) * 0.067_110_56 + (y as f32) * 0.005_837_15;
            (52.982_918 * f.fract()).fract()
        }
    }
}

/// RP.2 — quantize one `0..=255` channel to `levels` evenly-spaced steps with
/// the given dither `offset` (`[0, 1)`). `levels <= 1` leaves it untouched.
fn quantize_channel(c: u32, levels: u8, offset: f32) -> u32 {
    if levels <= 1 {
        return c;
    }
    let m = f32::from(levels - 1);
    #[allow(clippy::cast_precision_loss)]
    let scaled = (c as f32 / 255.0) * m;
    let q = (scaled + offset).floor().clamp(0.0, m);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let out = (q / m * 255.0).round() as u32;
    out
}

/// RP.2 — posterize one `0x00RRGGBB` logical pixel: per-channel quantization
/// with a per-pixel dither threshold (so banding becomes a stable pattern).
fn posterize_pixel(rgb: u32, x: usize, y: usize, cfg: crate::PosterizeConfig) -> u32 {
    let off = dither_offset(cfg.dither, x, y);
    let r = quantize_channel((rgb >> 16) & 0xff, cfg.levels_r, off);
    let g = quantize_channel((rgb >> 8) & 0xff, cfg.levels_g, off);
    let b = quantize_channel(rgb & 0xff, cfg.levels_b, off);
    (r << 16) | (g << 8) | b
}

/// RP.1 — which owned buffer a present/upscale step reads from. `Frame` is the
/// march-or-logical scene framebuffer, `Resolve` the box-downfiltered logical
/// buffer (`ssaa > 1`), `Output` the native-size composited buffer.
#[derive(Clone, Copy, PartialEq)]
enum CpuSrc {
    Frame,
    Resolve,
    Output,
}

/// PF.8 — one cached dense decode plus the shape key of the kv6 it was
/// decoded from. The key (`dims + voxel count`) backstops the explicit
/// invalidations: a lingering instance whose kv6 no longer matches the
/// template (e.g. spawned before a `remove_model` tombstone) misses the
/// cache and falls back to its own inline decode instead of drawing the
/// wrong volume.
struct DenseCacheEntry {
    key: (u32, u32, u32, usize),
    dense: std::sync::Arc<SpriteDense>,
}

/// Shape key of a kv6 for [`DenseCacheEntry`] staleness checks.
fn kv6_key(k: &Kv6) -> (u32, u32, u32, usize) {
    (k.xsiz, k.ysiz, k.zsiz, k.voxels.len())
}

/// Decode a sprite's kv6 (with its material map when present) — the
/// single decode the PF.8 caches hold.
fn decode_dense(s: &Sprite) -> SpriteDense {
    if s.material_map.is_empty() {
        SpriteDense::from_kv6(&s.kv6)
    } else {
        SpriteDense::from_kv6_with_materials(&s.kv6, &s.material_map)
    }
}

/// Fetch the cached dense for model/limb slot `m` when its key matches
/// `sprite`'s kv6, else decode inline (the pre-PF.8 per-draw behaviour).
fn dense_or_decode(
    cache: &[Option<DenseCacheEntry>],
    m: usize,
    sprite: &Sprite,
) -> std::sync::Arc<SpriteDense> {
    if let Some(Some(e)) = cache.get(m) {
        if e.key == kv6_key(&sprite.kv6) {
            return e.dense.clone();
        }
    }
    std::sync::Arc::new(decode_dense(sprite))
}

pub(crate) struct CpuBackend {
    /// Framebuffer presenter — native `softbuffer` window surface, or
    /// the wasm WebGL2 canvas blitter (see [`Presenter`]). On native,
    /// `softbuffer::Context` is dropped after surface creation; the
    /// surface keeps its own clone of the type-erased `Arc<dyn …>`
    /// display/window handles so the backend stays generic-free over
    /// the host's windowing library.
    present_target: Presenter,
    /// Current **window** (native) size in physical pixels. Seeded at
    /// construction, updated by [`Self::resize`] — replaces the old
    /// per-frame `window.inner_size()` poll so the backend never
    /// touches a concrete window type. The scene marches at
    /// [`Self::logical_dims`] (≤ this under a fixed [`RenderResolution`])
    /// and is nearest-upscaled to this size at present (RP.0).
    current_dims: (u32, u32),
    /// RP.0 — logical render resolution policy. `Native` ⇒ logical == window
    /// (byte-identical to pre-RP). A `Fixed`/`Scale` value decouples the
    /// marched pixel count from the window size.
    render_res: crate::RenderResolution,
    /// RP.1 — supersampling factor. `1` = off (the raycaster marches at the
    /// logical size). `>1` marches at `logical × ssaa` and box-downfilters back
    /// to logical before the upscale — anti-aliasing the retro grid. The
    /// [`Self::framebuffer`] is sized to the march resolution (`logical × ssaa`).
    ssaa: u32,
    /// RP.1 — logical-sized buffer holding the box-downfiltered scene when
    /// `ssaa > 1`. Empty when `ssaa == 1` (the framebuffer is already logical).
    /// Also holds the posterized image (RP.2) when posterize is active.
    resolve: Vec<u32>,
    /// RP.2 — reduced-palette post applied at logical resolution in the resolve
    /// step. `None` = off (when `ssaa == 1` the framebuffer is presented as-is).
    posterize: Option<crate::PosterizeConfig>,
    /// RP.0 — native-size scratch buffer the logical scene is nearest-upscaled
    /// into before present, in the non-`Native` path. The egui overlay
    /// rasterises here (at native res) so the HUD stays crisp. Empty in the
    /// `Native` path (which presents the logical buffer directly).
    output: Vec<u32>,
    zbuffer: Vec<f32>,
    /// Framebuffer dimensions of the last `render` — the `zbuffer`
    /// stride for [`Self::pick_depth`].
    last_dims: (u32, u32),
    /// Opticast projection params `(hx, hy, hz)` of the last `render`,
    /// from its [`OpticastSettings`] — the CPU unproject for
    /// [`Self::pixel_ray`].
    last_hxyz: (f32, f32, f32),
    clear_sky: u32,
    /// Pre-built per-instance sprites (one per [`SpriteSet`] instance,
    /// model KV6 cloned once at `set_sprites`), drawn each frame after
    /// the world via `draw_sprite`.
    sprites: Vec<Sprite>,
    /// GPU.12 incremental — source [`SpriteSet::models`] index per entry
    /// in [`sprites`](Self::sprites), so
    /// [`update_sprite_model`](Self::update_sprite_model) can swap one
    /// model's `kv6` into every instance of it without a full rebuild.
    sprite_models: Vec<usize>,
    /// Model templates from the last [`SpriteSet`] (`set.models`), kept so
    /// [`Self::add_dyn_instance_posed`] can clone a model by id. The GPU backend
    /// keeps the analogous `sprite_models_tpl`.
    models: Vec<Sprite>,
    /// Dynamically added instances (see [`Self::add_dyn_instance_posed`]) — a
    /// swap-removable tail sublist drawn after the static sprites, the CPU
    /// analogue of the GPU registry's appended instances.
    dyn_sprites: Vec<Sprite>,
    /// Source model index per entry in [`dyn_sprites`](Self::dyn_sprites),
    /// so [`Self::update_sprite_model`] refreshes dynamic instances too.
    dyn_models: Vec<usize>,
    /// Decoded animated voxel clips, one cached [`ClipFlipbook`] each
    /// (VCL.3). A clip's frames are decoded once at
    /// [`Self::add_voxel_clip`]; per-frame playback is a grid select.
    clip_books: Vec<ClipFlipbook>,
    /// Posed KFA limbs (flattened across all registered KFA sprites),
    /// refreshed by [`Self::update_kfa_poses`] and drawn after the
    /// static sprites each frame via `draw_sprite`.
    kfa_limbs: Vec<Sprite>,
    /// PF.5 — the shadow-caster demotion count last warned about, so the
    /// over-cap `eprintln` fires once per change instead of every frame.
    shadow_demote_warned: usize,
    /// PF.7 — reusable composed-render scratch (temp fb/zb pair + per-grid
    /// light scratch): kills two full-frame allocations per frame.
    scene_scratch: SceneRenderScratch,
    /// PF.8 — cached dense decodes per model template (parallel to
    /// [`models`](Self::models)): the draw + shadow-occluder paths share
    /// one `Arc<SpriteDense>` per model instead of re-densifying every
    /// instance every frame. Explicitly invalidated on model changes; the
    /// stored kv6 shape key backstops any missed path (a mismatching
    /// instance falls back to an inline decode).
    model_dense: Vec<Option<DenseCacheEntry>>,
    /// PF.8 — cached dense decodes per posed KFA limb (flat, parallel to
    /// [`kfa_limbs`](Self::kfa_limbs)); limb voxels are pose-invariant.
    limb_dense: Vec<Option<DenseCacheEntry>>,
    /// `F`-capture: when set, the next frame copies its composited
    /// buffer into `captured` before presenting.
    capture_next: bool,
    captured: Option<(Vec<u32>, u32, u32)>,
    /// Owned composited frame (`0x00RRGGBB`), sized `width*height` of the
    /// last [`Self::render`]. `render` composites the scene + sprites
    /// here without touching the window; [`Self::present`] blits it into
    /// the softbuffer surface and presents, and [`Self::paint_egui`]
    /// rasterises egui over it first. Decoupling the composite from the
    /// present lets a host slot a UI pass between them.
    framebuffer: Vec<u32>,
    /// Mirror the composited scene horizontally just before display (set via
    /// [`SceneRenderer::set_flip_x`](crate::SceneRenderer::set_flip_x)). The
    /// flip is applied to the scene framebuffer *before* the egui overlay, so
    /// the 3D view un-mirrors while the UI stays upright.
    flip_x: bool,
    /// Retained image-sprite textures, indexed by [`ImageId`]. A dropped
    /// slot is `None` and may be re-used by a later `upload_image`.
    images: Vec<Option<CpuImage>>,
    /// egui atlas cache + software rasteriser (`hud` feature).
    #[cfg(feature = "hud")]
    egui_raster: crate::cpu_egui::EguiRaster,
}

impl CpuBackend {
    /// Shared construction: build the pool / z-buffer / framebuffer
    /// around an already-created `present_target` (native softbuffer
    /// surface or wasm WebGL2 blitter).
    fn assemble(present_target: Presenter, size: (u32, u32), opts: &RenderOptions) -> Self {
        let (w, h) = (size.0.max(1), size.1.max(1));
        let zbuffer = vec![f32::INFINITY; (w as usize) * (h as usize)];
        let framebuffer = vec![opts.clear_sky.0; (w as usize) * (h as usize)];

        Self {
            present_target,
            current_dims: (w, h),
            render_res: crate::RenderResolution::Native,
            ssaa: 1,
            resolve: Vec::new(),
            posterize: None,
            output: Vec::new(),
            zbuffer,
            last_dims: (w, h),
            last_hxyz: (0.0, 0.0, 0.0),
            clear_sky: opts.clear_sky.0,
            sprites: Vec::new(),
            sprite_models: Vec::new(),
            models: Vec::new(),
            dyn_sprites: Vec::new(),
            dyn_models: Vec::new(),
            clip_books: Vec::new(),
            kfa_limbs: Vec::new(),
            shadow_demote_warned: 0,
            scene_scratch: SceneRenderScratch::default(),
            model_dense: Vec::new(),
            limb_dense: Vec::new(),
            capture_next: false,
            captured: None,
            framebuffer,
            flip_x: false,
            images: Vec::new(),
            #[cfg(feature = "hud")]
            egui_raster: crate::cpu_egui::EguiRaster::default(),
        }
    }

    /// Toggle the horizontal scene flip (see [`Self::flip_x`]).
    pub(crate) fn set_flip_x(&mut self, flip: bool) {
        self.flip_x = flip;
    }

    /// Reverse each framebuffer row in place — a horizontal mirror of the
    /// composited scene. Called before display, before any egui overlay.
    fn flip_framebuffer(&mut self) {
        let w = self.last_dims.0 as usize;
        if w == 0 {
            return;
        }
        for row in self.framebuffer.chunks_mut(w) {
            row.reverse();
        }
    }

    /// Native: present into a `softbuffer` surface bound to `window`.
    /// QE.2b — fallible: a display/surface the software presenter can't
    /// bind (broken Wayland/X11 connection, unsupported handle kind)
    /// surfaces as [`RenderError::CpuSurface`] instead of a panic.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn try_new<W>(
        window: Arc<W>,
        size: (u32, u32),
        opts: &RenderOptions,
    ) -> Result<Self, crate::RenderError>
    where
        W: HasWindowHandle + HasDisplayHandle + Send + Sync + 'static,
    {
        // Erase the concrete window type behind two `Arc<dyn …>`
        // handles. `raw-window-handle` implements `HasDisplayHandle` /
        // `HasWindowHandle` for `Arc<H>` with `H: ?Sized`, and a bare
        // trait object implements its own (object-safe) trait, so both
        // erased Arcs satisfy softbuffer's bounds.
        let display: Arc<DynDisplay> = window.clone();
        let window: Arc<DynWindow> = window;
        let context = softbuffer::Context::new(display)
            .map_err(|e| crate::RenderError::CpuSurface(format!("softbuffer context: {e}")))?;
        let surface = softbuffer::Surface::new(&context, window)
            .map_err(|e| crate::RenderError::CpuSurface(format!("softbuffer surface: {e}")))?;
        Ok(Self::assemble(surface, size, opts))
    }

    /// wasm: present into a WebGL2 blitter over `canvas` (no softbuffer
    /// in the browser).
    #[cfg(target_arch = "wasm32")]
    pub(crate) fn new_from_canvas(
        canvas: web_sys::HtmlCanvasElement,
        size: (u32, u32),
        opts: &RenderOptions,
    ) -> Self {
        let (w, h) = (size.0.max(1), size.1.max(1));
        let blit = crate::cpu_blit::WebGlBlit::new(&canvas, w, h)
            .expect("roxlap-render: WebGL2 blit init");
        Self::assemble(blit, size, opts)
    }

    /// Request that the next rendered frame be captured for readback.
    pub(crate) fn request_capture(&mut self) {
        self.capture_next = true;
    }

    /// Take the most recently captured frame, if any.
    pub(crate) fn take_capture(&mut self) -> Option<(Vec<u32>, u32, u32)> {
        self.captured.take()
    }

    /// World-space view-ray direction (un-normalised) for pixel
    /// `(x, y)` under the CPU opticast projection (voxlap `setcamera`):
    /// `(x - hx)·right + (y - hy)·down + hz·forward`, using the last
    /// frame's `(hx, hy, hz)`. `None` before the first render.
    pub(crate) fn pixel_ray(&self, camera: &Camera, x: f64, y: f64) -> Option<[f64; 3]> {
        let (hx, hy, hz) = self.last_hxyz;
        if hz <= 0.0 {
            return None;
        }
        // RP.0 — `(x, y)` are window pixels; the projection (`hx,hy,hz`) was
        // derived at the logical size, so map window → logical first.
        let (lx, ly) = self.window_to_logical_f(x, y);
        Some(setcamera_pixel_ray(
            camera.right,
            camera.down,
            camera.forward,
            lx,
            ly,
            hx,
            hy,
            hz,
        ))
    }

    /// Map a window (native) pixel coordinate to the logical render-target
    /// coordinate the last frame marched at. Identity under `Native`.
    fn window_to_logical_f(&self, x: f64, y: f64) -> (f64, f64) {
        let (lw, lh) = self.last_dims;
        let (nw, nh) = self.current_dims;
        if nw == 0 || nh == 0 || (lw, lh) == (nw, nh) {
            return (x, y);
        }
        (
            x * f64::from(lw) / f64::from(nw),
            y * f64::from(lh) / f64::from(nh),
        )
    }

    /// World-t depth at window pixel `(x, y)` from the last frame's z-buffer
    /// (already in CPU memory — no readback). `None` for out-of-bounds
    /// or sky (`+INF`). See [`SceneRenderer::pick_depth`].
    pub(crate) fn pick_depth(&self, x: u32, y: u32) -> Option<f32> {
        let (lw, lh) = self.last_dims;
        let (nw, nh) = self.current_dims;
        // Map the window pixel to the logical z-buffer grid (RP.0).
        let (lx, ly) = if (lw, lh) == (nw, nh) {
            (x, y)
        } else if nw == 0 || nh == 0 {
            return None;
        } else {
            (
                (x * lw / nw).min(lw.saturating_sub(1)),
                (y * lh / nh).min(lh.saturating_sub(1)),
            )
        };
        if lx >= lw || ly >= lh {
            return None;
        }
        let t = *self.zbuffer.get((ly * lw + lx) as usize)?;
        if t.is_finite() {
            Some(t)
        } else {
            None
        }
    }

    /// Pre-build one [`Sprite`] per instance (model KV6 cloned, the
    /// instance position applied) so per-frame drawing never re-clones.
    pub(crate) fn set_sprites(&mut self, set: &SpriteSet) {
        let mut sprites = Vec::with_capacity(set.instances.len());
        let mut sprite_models = Vec::with_capacity(set.instances.len());
        for inst in &set.instances {
            if let Some(model) = set.models.get(inst.model) {
                let mut s = model.clone();
                s.p = inst.pos;
                sprites.push(s);
                sprite_models.push(inst.model);
            }
        }
        self.sprites = sprites;
        self.sprite_models = sprite_models;
        // Retain templates for dynamic adds; a new set drops old dynamics.
        self.models.clone_from(&set.models);
        self.dyn_sprites.clear();
        self.dyn_models.clear();
        // Mirror the GPU backend: drop the registered clip flipbooks too, so
        // clip indices restart at 0 on both backends (and the old volumes
        // don't leak).
        self.clip_books.clear();
        // PF.8 — a fresh model set invalidates every cached decode.
        self.model_dense.clear();
    }

    /// Append one dynamic instance of `model_index` pre-posed by `xf`;
    /// returns its dynamic-sublist index (always the new last), or
    /// `None` (appending nothing) if the model id is unknown - so the
    /// facade never books a handle for an instance that was not
    /// created (QE.3a).
    pub(crate) fn add_dyn_instance_posed(
        &mut self,
        model_index: usize,
        xf: DynSpriteTransform,
    ) -> Option<usize> {
        let idx = self.dyn_sprites.len();
        let model = self.models.get(model_index)?;
        let mut s = model.clone();
        xf.apply_to(&mut s);
        self.dyn_sprites.push(s);
        self.dyn_models.push(model_index);
        Some(idx)
    }

    /// O(1) per-frame pose update of dynamic instance `idx` (position +
    /// orientation). No-op if `idx` is out of range.
    pub(crate) fn set_dyn_instance_transform(&mut self, idx: usize, xf: DynSpriteTransform) {
        if let Some(s) = self.dyn_sprites.get_mut(idx) {
            xf.apply_to(s);
        }
    }

    /// Set dynamic instance `idx`'s voxel-material id (TV stage). No-op if
    /// `idx` is out of range.
    pub(crate) fn set_dyn_instance_material(&mut self, idx: usize, material: u8) {
        if let Some(s) = self.dyn_sprites.get_mut(idx) {
            s.material = material;
        }
    }

    /// Set dynamic instance `idx`'s per-instance alpha multiplier (TV stage,
    /// `255` = unscaled). No-op if `idx` is out of range.
    pub(crate) fn set_dyn_instance_alpha(&mut self, idx: usize, alpha_mul: u8) {
        if let Some(s) = self.dyn_sprites.get_mut(idx) {
            s.alpha_mul = alpha_mul;
        }
    }

    /// Set dynamic instance `idx`'s per-instance RGB tint (`0x00RRGGBB`, white
    /// = no-op). No-op if `idx` is out of range.
    pub(crate) fn set_dyn_instance_tint(&mut self, idx: usize, tint: u32) {
        if let Some(s) = self.dyn_sprites.get_mut(idx) {
            s.tint = tint & 0x00FF_FFFF;
        }
    }

    /// Set dynamic instance `idx`'s shadow cast/receive flags live (XS.4 /
    /// BB.3), preserving its other flag bits. No-op if `idx` is out of range.
    pub(crate) fn set_dyn_instance_shadow_flags(
        &mut self,
        idx: usize,
        casts: bool,
        receives: bool,
    ) {
        if let Some(s) = self.dyn_sprites.get_mut(idx) {
            crate::apply_shadow_flags(&mut s.flags, casts, receives);
        }
    }

    /// Set dynamic instance `idx`'s lighting mode live (BB.2b), preserving its
    /// other flag bits. No-op if `idx` is out of range.
    pub(crate) fn set_dyn_instance_lighting(&mut self, idx: usize, mode: crate::BillboardLighting) {
        if let Some(s) = self.dyn_sprites.get_mut(idx) {
            crate::apply_lighting_flags(&mut s.flags, mode);
        }
    }

    /// Register a new model template (axis-aligned, kv6 cloned once) and
    /// return its positional index. The streaming-in counterpart to
    /// [`Self::add_dyn_instance_posed`] for unique generated geometry.
    pub(crate) fn add_model(&mut self, kv6: &Kv6) -> usize {
        let idx = self.models.len();
        self.models
            .push(Sprite::axis_aligned(kv6.clone(), [0.0, 0.0, 0.0]));
        idx
    }

    /// Register a model template carrying a per-voxel material colour map
    /// (TV.3 mixed models). Instances clone it, so the map travels with each
    /// spawned instance and the per-draw decode classifies voxels by colour.
    pub(crate) fn add_model_with_materials(
        &mut self,
        kv6: &Kv6,
        material_map: &[(Rgb, u8)],
    ) -> usize {
        let idx = self.models.len();
        let mut s = Sprite::axis_aligned(kv6.clone(), [0.0, 0.0, 0.0]);
        s.material_map = material_map.to_vec();
        self.models.push(s);
        idx
    }

    /// Tombstone model template `host_idx` in place: replace it with an
    /// empty placeholder (freeing its kv6) but keep the slot, mirroring
    /// the GPU backend's in-place tombstone. Existing instances keep their
    /// own kv6 clones and draw until removed via
    /// [`Self::remove_dyn_instance`]. No-op if `host_idx` is out of range.
    pub(crate) fn remove_model(&mut self, host_idx: usize) {
        // PF.8 — drop the cached decode; lingering instances (which keep
        // their own kv6 clones) fall back to inline decodes via the shape
        // key mismatch.
        if let Some(slot) = self.model_dense.get_mut(host_idx) {
            *slot = None;
        }
        if let Some(t) = self.models.get_mut(host_idx) {
            *t = Sprite::axis_aligned(empty_kv6(), [0.0, 0.0, 0.0]);
        }
    }

    /// No reclamation step on the CPU backend — removed templates already
    /// dropped their kv6 in [`Self::remove_model`], and live instances own
    /// independent clones. Present only for facade parity with the GPU
    /// backend's buffer repack.
    #[allow(clippy::unused_self)]
    pub(crate) fn compact_models(&mut self) {}

    /// Remove the dynamic instance at `idx` by swap-remove. Returns
    /// `Some(old_last)` when a different instance was moved into `idx`, or
    /// `None` if `idx` was the last / out of range — matching the GPU
    /// backend so the facade's handle fixup is identical.
    pub(crate) fn remove_dyn_instance(&mut self, idx: usize) -> Option<usize> {
        if idx >= self.dyn_sprites.len() {
            return None;
        }
        let last = self.dyn_sprites.len() - 1;
        self.dyn_sprites.swap_remove(idx);
        self.dyn_models.swap_remove(idx);
        (idx != last).then_some(last)
    }

    /// Register an animated voxel clip (VCL.4): decode every frame into a
    /// cached [`ClipFlipbook`]. With a non-empty `material_map` (TV.3), each
    /// frame's voxels are classified into per-voxel material ids by colour —
    /// the clip analogue of [`Self::add_model_with_materials`]. An empty map
    /// is the plain all-opaque clip.
    pub(crate) fn add_voxel_clip_with_materials(
        &mut self,
        clip: &DecodedClip,
        material_map: &[(Rgb, u8)],
    ) -> usize {
        let idx = self.clip_books.len();
        self.clip_books
            .push(ClipFlipbook::from_decoded_with_materials(
                clip,
                material_map,
            ));
        idx
    }

    /// Tombstone clip `clip_idx` in place (replace its flipbook with an
    /// empty one). Existing instances of it then draw nothing; the slot is
    /// kept so other clip indices stay valid. No-op if out of range.
    pub(crate) fn remove_voxel_clip(&mut self, clip_idx: usize) {
        // The facade detaches the clip's instances in its shared
        // bookkeeping (QE.3a); this backend just empties the flipbook.
        if let Some(book) = self.clip_books.get_mut(clip_idx) {
            *book = ClipFlipbook::empty();
        }
    }

    /// Append a dynamic instance playing clip `clip_idx`, posed by `xf`,
    /// starting on frame 0. Returns its dynamic-sublist index. The
    /// `dyn_sprites` entry is a pose carrier (empty kv6); the clip's
    /// frames supply the pixels — which frame comes from the facade's
    /// [`SceneState::dyn_clip`](crate::SceneState) bookkeeping (QE.3a).
    /// `Option` for facade parity with the GPU backend, which can
    /// decline (empty clip / no registry); the CPU path always appends.
    #[allow(clippy::unnecessary_wraps)]
    pub(crate) fn add_clip_instance(
        &mut self,
        _clip_idx: usize,
        xf: DynSpriteTransform,
    ) -> Option<usize> {
        let idx = self.dyn_sprites.len();
        let mut s = Sprite::axis_aligned(empty_kv6(), [0.0, 0.0, 0.0]);
        xf.apply_to(&mut s);
        self.dyn_sprites.push(s);
        self.dyn_models.push(usize::MAX); // not a KV6 model
        Some(idx)
    }

    /// Replace one frame's cached dense grid of clip `clip_idx` (the editor's
    /// single-frame edit). Returns `false` if out of range.
    pub(crate) fn update_clip_frame(
        &mut self,
        clip_idx: usize,
        frame: usize,
        vf: &VoxelFrame,
        dims: [u32; 3],
        pivot: [f32; 3],
        material_map: &[(Rgb, u8)],
    ) -> bool {
        let dense = SpriteDense::from_voxel_frame_with_materials(vf, dims, pivot, material_map);
        self.clip_books
            .get_mut(clip_idx)
            .is_some_and(|b| b.set_frame(frame, dense))
    }

    /// GPU.12 incremental — swap the edited `kv6` into every cached
    /// instance of host model `model_index`, keeping each instance's
    /// world position. Mirrors the GPU backend's single-model update on
    /// the software path (where "rebuild" is just a kv6 clone per
    /// instance, so the win is parity rather than bandwidth). No-op if no
    /// instance references `model_index`.
    pub(crate) fn update_sprite_model(&mut self, model_index: usize, kv6: &Kv6) {
        self.update_sprite_model_with_materials(model_index, kv6, None);
    }

    /// Like [`Self::update_sprite_model`] but also overwrites the per-voxel
    /// material colour map on the model's instances/template (TV.3) — the
    /// material-aware refresh behind the streaming-clip path. `Some(map)`
    /// replaces the map (empty clears it); `None` leaves the existing map
    /// untouched (the plain `refresh_sprite_model` behaviour).
    pub(crate) fn update_sprite_model_with_materials(
        &mut self,
        model_index: usize,
        kv6: &Kv6,
        material_map: Option<&[(Rgb, u8)]>,
    ) {
        for (s, &m) in self.sprites.iter_mut().zip(&self.sprite_models) {
            if m == model_index {
                s.kv6 = kv6.clone();
                if let Some(map) = material_map {
                    s.material_map = map.to_vec();
                }
            }
        }
        // Dynamic instances of the same model refresh too.
        for (s, &m) in self.dyn_sprites.iter_mut().zip(&self.dyn_models) {
            if m == model_index {
                s.kv6 = kv6.clone();
                if let Some(map) = material_map {
                    s.material_map = map.to_vec();
                }
            }
        }
        // Keep the stored template current so future dynamic adds use it.
        if let Some(t) = self.models.get_mut(model_index) {
            t.kv6 = kv6.clone();
            if let Some(map) = material_map {
                t.material_map = map.to_vec();
            }
        }
        // PF.8 — the swapped kv6 invalidates the cached decode (explicit:
        // the shape key alone can collide across e.g. streaming-clip
        // frames with equal dims + voxel counts).
        if let Some(slot) = self.model_dense.get_mut(model_index) {
            *slot = None;
        }
    }

    /// Register KFA sprites: solve every limb's world transform and cache
    /// the posed [`Sprite`]s (full clone, including each limb's kv6 —
    /// once, at registration). PF.8 — also resets the limb dense cache.
    pub(crate) fn set_kfa_sprites(&mut self, kfas: &mut [KfaSprite]) {
        self.kfa_limbs.clear();
        self.limb_dense.clear();
        for kfa in kfas.iter_mut() {
            solve_kfa_limbs(kfa);
            self.kfa_limbs.extend(kfa.limbs.iter().cloned());
        }
    }

    /// Re-solve every KFA limb's world transform for the next
    /// [`Self::render`]. PF.8 — pose-only: limb voxel data is animation-
    /// invariant, so only the transform + display fields are copied (the
    /// old full-clone path deep-copied every limb's kv6 EVERY pose
    /// update). A limb-count mismatch (re-registration) falls back to the
    /// full rebuild.
    pub(crate) fn update_kfa_poses(&mut self, kfas: &mut [KfaSprite]) {
        let total: usize = kfas.iter().map(|k| k.limbs.len()).sum();
        if total != self.kfa_limbs.len() {
            self.set_kfa_sprites(kfas);
            return;
        }
        let mut i = 0usize;
        for kfa in kfas.iter_mut() {
            solve_kfa_limbs(kfa);
            for limb in &kfa.limbs {
                let dst = &mut self.kfa_limbs[i];
                dst.p = limb.p;
                dst.s = limb.s;
                dst.h = limb.h;
                dst.f = limb.f;
                dst.flags = limb.flags;
                dst.material = limb.material;
                dst.alpha_mul = limb.alpha_mul;
                dst.tint = limb.tint;
                i += 1;
            }
        }
    }

    /// PF.8 — refresh the dense-decode caches for everything the frame
    /// will draw: model templates referenced by static/dynamic instances,
    /// and posed KFA limbs. One decode per model/limb, reused by the draw
    /// AND the shadow-occluder build.
    fn ensure_dense_caches(&mut self) {
        if self.model_dense.len() < self.models.len() {
            self.model_dense.resize_with(self.models.len(), || None);
        }
        for &m in self.sprite_models.iter().chain(self.dyn_models.iter()) {
            let Some(slot) = self.model_dense.get_mut(m) else {
                continue; // usize::MAX clip sentinel / stale index
            };
            let Some(t) = self.models.get(m) else {
                continue;
            };
            let key = kv6_key(&t.kv6);
            if slot.as_ref().map_or(true, |e| e.key != key) {
                *slot = Some(DenseCacheEntry {
                    key,
                    dense: std::sync::Arc::new(decode_dense(t)),
                });
            }
        }
        if self.limb_dense.len() != self.kfa_limbs.len() {
            self.limb_dense = (0..self.kfa_limbs.len()).map(|_| None).collect();
        }
        for (slot, limb) in self.limb_dense.iter_mut().zip(&self.kfa_limbs) {
            let key = kv6_key(&limb.kv6);
            if slot.as_ref().map_or(true, |e| e.key != key) {
                *slot = Some(DenseCacheEntry {
                    key,
                    dense: std::sync::Arc::new(decode_dense(limb)),
                });
            }
        }
    }

    pub(crate) fn resize(&mut self, width: u32, height: u32) {
        // softbuffer + the pool resize lazily inside `render`; we just
        // record the new size the host reported (replacing the old
        // per-frame `window.inner_size()` poll). The WebGL2 blitter's
        // texture, by contrast, must be re-allocated eagerly — to the
        // **native** size, since the present surface is the window (the
        // logical→native upscale happens before the blitter sees pixels).
        self.current_dims = (width.max(1), height.max(1));
        #[cfg(target_arch = "wasm32")]
        self.present_target
            .resize(self.current_dims.0, self.current_dims.1);
    }

    /// RP.0 — set the logical render resolution policy.
    pub(crate) fn set_render_resolution(&mut self, res: crate::RenderResolution) {
        self.render_res = res;
    }

    /// RP.1 — set the supersampling factor (clamped to `1..=4`). `1` = off.
    pub(crate) fn set_ssaa(&mut self, factor: u8) {
        self.ssaa = u32::from(factor).clamp(1, 4);
    }

    /// RP.2 — set (or clear) the reduced-palette posterize post.
    pub(crate) fn set_posterize(&mut self, cfg: Option<crate::PosterizeConfig>) {
        self.posterize = cfg;
    }

    /// Whether the resolve step has work to do (SSAA downfilter or posterize).
    fn resolve_active(&self) -> bool {
        self.ssaa > 1 || self.posterize.is_some()
    }

    /// RP.1 — the resolution the raycaster actually marches at:
    /// `logical × ssaa`. The framebuffer/zbuffer are sized to this.
    pub(crate) fn render_dims(&self) -> (u32, u32) {
        let (lw, lh) = self.logical_dims();
        (lw * self.ssaa, lh * self.ssaa)
    }

    /// RP.0 — the logical (retro) grid size the scene resolves to before the
    /// nearest upscale to the window, resolved against the current window size.
    pub(crate) fn logical_dims(&self) -> (u32, u32) {
        self.render_res.logical_for(self.current_dims)
    }

    pub(crate) fn render(
        &mut self,
        scene: &mut Scene,
        camera: &Camera,
        frame: &FrameParams,
        shared: &crate::SceneState,
    ) {
        // RP.0/RP.1 — march at the *render* size (`logical × ssaa`), then
        // box-downfilter to logical (RP.1) + nearest-upscale to the window
        // (RP.0) at present. `Native` + `ssaa==1` ⇒ render == window ⇒ pre-RP
        // behaviour verbatim.
        let (width, height) = self.render_dims();
        if width == 0 || height == 0 {
            return;
        }
        let pixel_count = (width as usize) * (height as usize);
        self.last_dims = (width, height);

        // RP.0 — the host builds `OpticastSettings` for the *window*; the
        // raster extent (`xres`/`yres`) and projection must instead match the
        // logical render target the framebuffer is sized to, or the compositor
        // indexes past the (smaller) framebuffer. Rescale: recentre `hx`/`hy`,
        // scale the focal `hz` by the vertical ratio (preserving the vertical
        // FOV — the GPU pinhole does the same via `gpu_fov_y_rad`), and march
        // the full logical frame. Identity when the host already sized to
        // logical (e.g. `Native`).
        let settings = {
            let src = frame.settings;
            if (src.xres, src.yres) == (width, height) {
                *src
            } else {
                #[allow(clippy::cast_precision_loss)]
                let sx = width as f32 / (src.xres.max(1) as f32);
                #[allow(clippy::cast_precision_loss)]
                let sy = height as f32 / (src.yres.max(1) as f32);
                let mut s = *src;
                s.xres = width;
                s.yres = height;
                s.y_start = 0;
                s.y_end = height;
                s.hx = src.hx * sx;
                s.hy = src.hy * sy;
                s.hz = src.hz * sy;
                s
            }
        };
        self.last_hxyz = (settings.hx, settings.hy, settings.hz);

        // Grow the z-buffer to follow a window resize.
        if self.zbuffer.len() < pixel_count {
            self.zbuffer.resize(pixel_count, f32::INFINITY);
        }

        // PF.8 — one dense decode per model/limb, shared (`Arc`) by the
        // shadow-occluder build and the sprite draw pass below (was: every
        // caster re-densified per occluder rebuild, every instance
        // re-densified again per draw call, every frame). Refreshed before
        // the framebuffer borrow below.
        if frame.draw_sprites {
            self.ensure_dense_caches();
        }

        // Per-frame DDA fog config (engine sky/fog → renderer). Fog is
        // off when `fog_max_scan_dist <= 0`; otherwise the DDA ramps each
        // hit toward `fog_color` over that distance. `side_shades` darkens
        // each voxel face (default `[0; 6]` = no side shading).
        let fog = CpuFog {
            color: frame.fog_color.0,
            max_scan_dist: frame.fog_max_scan_dist,
            side_shades: frame.side_shades,
        };

        // Composite into the owned framebuffer (not the window) so the
        // present can be deferred — a host may paint a UI over it first.
        // `render_scene_composed` convention: caller pre-fills the
        // framebuffer with sky + the z-buffer with +INF, then it
        // z-merges every grid in.
        if self.framebuffer.len() < pixel_count {
            self.framebuffer.resize(pixel_count, self.clear_sky);
        }
        let fb = &mut self.framebuffer[..pixel_count];
        for px in fb.iter_mut() {
            *px = self.clear_sky;
        }
        for z in &mut self.zbuffer[..pixel_count] {
            *z = f32::INFINITY;
        }

        // CPU.1/CPU.2 — world-space dynamic lights for the diffuse stylized
        // path (the scene renderer transforms them per grid). `None` ⇒
        // disabled (the baked-byte fallback, byte-identical to pre-DL).
        // CPU.2 casts hard sun + point-light shadows; the per-pixel shadow
        // march reuses the render Sampler, so it's the slow fallback's
        // slowest path — but correct + on parity with the GPU look.
        let mut world_points: Vec<roxlap_core::CpuPointLight> = Vec::new();
        let cpu_lights = if let Some(rig) = frame.lights {
            let (sun, sun_dir, sun_color, sun_intensity) = match rig.sun {
                Some(s) => {
                    // Direction TO the sun = normalized −travel.
                    let d = s.direction;
                    let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                    let to = if len > 1e-6 {
                        [-d[0] / len, -d[1] / len, -d[2] / len]
                    } else {
                        [0.0; 3]
                    };
                    (true, to, s.color, s.intensity)
                }
                None => (false, [0.0; 3], [0.0; 3], 0.0),
            };
            let sun_casts = rig.sun.is_some_and(|s| s.casts_shadow);
            // CPU.2 — mirror the GPU MAX_SHADOW_CASTERS cap (the sun is the
            // first caster); demote the excess to shadowless, never silently.
            let mut budget = roxlap_gpu::MAX_SHADOW_CASTERS;
            if sun && sun_casts {
                budget = budget.saturating_sub(1);
            }
            let mut demoted = 0usize;
            // Shared greedy caster grant (points take priority over spots).
            let mut grant = |casts: bool| -> bool {
                if casts && budget > 0 {
                    budget -= 1;
                    true
                } else {
                    if casts {
                        demoted += 1;
                    }
                    false
                }
            };
            for p in rig.points {
                let allow = grant(p.casts_shadow);
                world_points.push(roxlap_core::CpuPointLight {
                    pos: p.position,
                    color: p.color,
                    intensity: p.intensity,
                    radius: p.radius,
                    casts_shadow: allow,
                    // `-1.0` outer cosine (180° cone) marks "not a spot" ⇒ the
                    // cone mask is skipped (an omnidirectional point light).
                    spot_dir: [0.0, 0.0, 1.0],
                    cos_inner: -1.0,
                    cos_outer: -1.0,
                });
            }
            // SL.2 — spots fold into the same world-space array; the cone axis
            // stays world-space here (render.rs inverse-rotates it per grid).
            for s in rig.spots {
                let allow = grant(s.casts_shadow);
                world_points.push(roxlap_core::CpuPointLight {
                    pos: s.position,
                    color: s.color,
                    intensity: s.intensity,
                    radius: s.radius,
                    casts_shadow: allow,
                    spot_dir: s.axis(),
                    cos_inner: s.cos_inner(),
                    cos_outer: s.cos_outer(),
                });
            }
            // PF.5 — warn once per change, not per frame (this runs in the
            // frame loop; an over-cap rig otherwise spams stderr at 60 Hz).
            if demoted != self.shadow_demote_warned {
                if demoted > 0 {
                    log::warn!(
                        "CPU: {demoted} shadow-casting point lights > MAX_SHADOW_CASTERS ({}); demoting the excess to shadowless",
                        roxlap_gpu::MAX_SHADOW_CASTERS
                    );
                }
                self.shadow_demote_warned = demoted;
            }
            roxlap_core::CpuLights {
                enabled: true,
                sun,
                sun_dir,
                sun_color,
                sun_intensity,
                sun_casts_shadow: sun && sun_casts,
                points: &world_points,
                ambient: rig.ambient,
                bands: rig.bands,
                shadow_tint: rig.shadow_tint,
                shadow_strength: rig.shadow_strength,
                shadow_bias: rig.shadow_bias_voxels,
                shadow_max_dist: rig.shadow_max_dist,
            }
        } else {
            roxlap_core::CpuLights::default()
        };

        // XS.2 — sprite shadows. A world-space occluder over the sprite
        // volumes (decoded dense grids + world poses): passed into the terrain
        // render so sprites **cast** hard shadows onto terrain, and reused for
        // the sprite pass below so sprites **receive** them (from terrain +
        // each other). Only built when a caster is active, to skip the decode
        // cost on the common unshadowed path.
        let shadows_active = cpu_lights.enabled
            && cpu_lights.shadow_strength > 0.0
            && (cpu_lights.sun_casts_shadow || cpu_lights.points.iter().any(|p| p.casts_shadow));
        // XS.4 — a sprite contributes to the occluder (casts a shadow) unless
        // it's invisible or flagged `NO_SHADOW_CAST`.
        let invis = roxlap_formats::sprite::SPRITE_FLAG_INVISIBLE;
        let casts = |s: &Sprite| s.flags & invis == 0 && s.casts_shadow();
        let sprite_occ = if shadows_active && frame.draw_sprites {
            let mut so = SpriteOccluder::new();
            for (s, &m) in self.sprites.iter().zip(&self.sprite_models) {
                if casts(s) {
                    so.push(dense_or_decode(&self.model_dense, m, s), s.p, s.s, s.h, s.f);
                }
            }
            for (i, s) in self.kfa_limbs.iter().enumerate() {
                if casts(s) {
                    so.push(dense_or_decode(&self.limb_dense, i, s), s.p, s.s, s.h, s.f);
                }
            }
            for (i, s) in self.dyn_sprites.iter().enumerate() {
                if !casts(s) {
                    continue;
                }
                if let Some((book, fr)) = shared.dyn_clip[i] {
                    if let Some(d) = self.clip_books.get(book).and_then(|b| b.frame_arc(fr)) {
                        so.push(d, s.p, s.s, s.h, s.f);
                    }
                } else {
                    so.push(
                        dense_or_decode(&self.model_dense, self.dyn_models[i], s),
                        s.p,
                        s.s,
                        s.h,
                        s.f,
                    );
                }
            }
            (!so.is_empty()).then_some(so)
        } else {
            None
        };

        let _ = render_scene_composed_with_materials_scratch(
            fb,
            &mut self.zbuffer[..pixel_count],
            width as usize,
            width,
            height,
            fog,
            scene,
            camera,
            &settings,
            frame.sky_color.0,
            frame.sky,
            Some(&shared.materials),
            &shared.terrain_materials,
            cpu_lights,
            sprite_occ.as_ref().map(|o| o as &dyn WorldOccluder),
            // PF.7 — persistent temp-buffer pair + per-grid scratch.
            &mut self.scene_scratch,
        );

        // Paint the panorama sky into every background pixel (z still +INF):
        // pixels outside any grid's screen rect — most of a sprite/effect-only
        // view, and the margins around a small world grid — would otherwise
        // keep the flat `clear_sky` pre-fill. Terrain hits (finite z) and
        // composited translucent pixels are left untouched. Matches the GPU's
        // full-frame sky. (`outcome` no longer gates this — a grid scene needs
        // it just as much as an empty one.)
        if let Some(sky) = frame.sky {
            let cam_state =
                camera_math::derive(camera, width, height, settings.hx, settings.hy, settings.hz);
            render_sky_fill(
                fb,
                &self.zbuffer[..pixel_count],
                width as usize,
                width,
                height,
                &cam_state,
                &settings,
                sky,
            );
        }

        // Sprites layer on top of the voxel world, z-tested against the
        // same z-buffer via the clean-room DDA sprite raycaster. Drawn
        // flat-lit; `frame.draw_sprites` is the opt-in.
        if frame.draw_sprites
            && (!self.sprites.is_empty()
                || !self.dyn_sprites.is_empty()
                || !self.kfa_limbs.is_empty())
        {
            let cam_state =
                camera_math::derive(camera, width, height, settings.hx, settings.hy, settings.hz);
            // Global voxel-material palette (TV stage). A sprite whose
            // material is opaque (the default, and all of them until a
            // translucent material is defined) takes the unchanged first-hit
            // path; only translucent sprites accumulate.
            let materials = &shared.materials;
            // XS.2 — sprites RECEIVE shadows: the scene-wide occluder = grids
            // (built now that the terrain render released the `&mut Scene`) +
            // the sprite volumes. `composite_store` backs the borrow.
            let grid_occ = shadows_active
                .then(|| SceneOccluder::build(scene))
                .filter(|o| !o.is_empty());
            let composite_store;
            let recv_occ: Option<&dyn WorldOccluder> =
                match (grid_occ.as_ref(), sprite_occ.as_ref()) {
                    (Some(g), Some(s)) => {
                        composite_store = CompositeOccluder {
                            a: g,
                            b: s as &dyn WorldOccluder,
                        };
                        Some(&composite_store)
                    }
                    (Some(g), None) => Some(g as &dyn WorldOccluder),
                    (None, Some(s)) => Some(s as &dyn WorldOccluder),
                    (None, None) => None,
                };
            let shade_of = |s: &Sprite| SpriteShade {
                materials,
                material: s.material,
                alpha_mul: s.alpha_mul,
                // Per-instance RGB tint (white ⇒ no-op).
                tint: s.tint,
                // DL.7 — world-space lights so opaque sprites/clips get the
                // same stylized lighting as the terrain.
                lights: cpu_lights,
                // XS.2/XS.4 — receive hard shadows from terrain + other
                // sprites, unless this sprite opted out (`NO_SHADOW_RECEIVE`).
                shadow: if s.receives_shadow() { recv_occ } else { None },
            };
            // Static sprites + posed KFA limbs: plain KV6 sprites. All
            // z-test against the shared buffer so order doesn't matter.
            // PF.8 — draw from the cached per-model/per-limb dense decode
            // (was: `draw_sprite_dda_shaded` re-densified per call).
            for (sprite, &m) in self.sprites.iter().zip(&self.sprite_models) {
                if sprite.flags & invis != 0 {
                    continue;
                }
                let dense = dense_or_decode(&self.model_dense, m, sprite);
                let _written = draw_sprite_dense_shaded(
                    fb,
                    &mut self.zbuffer[..pixel_count],
                    width as usize,
                    width,
                    height,
                    &cam_state,
                    &settings,
                    &dense,
                    sprite.p,
                    sprite.s,
                    sprite.h,
                    sprite.f,
                    sprite.flags,
                    Some(shade_of(sprite)),
                );
            }
            for (i, sprite) in self.kfa_limbs.iter().enumerate() {
                if sprite.flags & invis != 0 {
                    continue;
                }
                let dense = dense_or_decode(&self.limb_dense, i, sprite);
                let _written = draw_sprite_dense_shaded(
                    fb,
                    &mut self.zbuffer[..pixel_count],
                    width as usize,
                    width,
                    height,
                    &cam_state,
                    &settings,
                    &dense,
                    sprite.p,
                    sprite.s,
                    sprite.h,
                    sprite.f,
                    sprite.flags,
                    Some(shade_of(sprite)),
                );
            }
            // Dynamic instances: a KV6 sprite, or — if it carries a clip
            // association — the selected frame of an animated voxel clip
            // (VCL.4). The `dyn_sprites` entry is the pose carrier either way.
            for (i, sprite) in self.dyn_sprites.iter().enumerate() {
                let zb = &mut self.zbuffer[..pixel_count];
                let shade = shade_of(sprite);
                if let Some((book, fr)) = shared.dyn_clip[i] {
                    if let Some(b) = self.clip_books.get(book) {
                        let _written = b.draw_frame_shaded(
                            fb,
                            zb,
                            width as usize,
                            width,
                            height,
                            &cam_state,
                            &settings,
                            fr,
                            sprite.p,
                            sprite.s,
                            sprite.h,
                            sprite.f,
                            sprite.flags,
                            Some(shade),
                        );
                    }
                } else {
                    if sprite.flags & invis != 0 {
                        continue;
                    }
                    let dense = dense_or_decode(&self.model_dense, self.dyn_models[i], sprite);
                    let _written = draw_sprite_dense_shaded(
                        fb,
                        zb,
                        width as usize,
                        width,
                        height,
                        &cam_state,
                        &settings,
                        &dense,
                        sprite.p,
                        sprite.s,
                        sprite.h,
                        sprite.f,
                        sprite.flags,
                        Some(shade),
                    );
                }
            }
        }

        if self.capture_next {
            self.capture_next = false;
            self.captured = Some((fb.to_vec(), width, height));
        }
        // No present here — the host calls `present` or `paint_egui`.
    }

    /// Blit the composited [`Self::framebuffer`] into the softbuffer
    /// surface and present it. The no-UI counterpart to
    /// [`Self::paint_egui`]; both finish the frame `render` started.
    pub(crate) fn present(&mut self) {
        // Flip is applied in render (march) space before any resolve/upscale,
        // matching the pre-RP order (a box downfilter commutes with the flip).
        if self.flip_x {
            self.flip_framebuffer();
        }
        let logical = self.logical_dims();
        let native = self.current_dims;
        // RP.1 — resolve march → logical (box downfilter) when supersampling;
        // otherwise the framebuffer is already the logical image.
        let logical_src = self.resolve_scene(logical);
        if logical == native {
            // Present the logical buffer directly (Native + ssaa==1 ⇒ pre-RP).
            self.blit_and_present_from(logical_src, native);
        } else {
            // Nearest-upscale logical → native, then present.
            self.upscale_to_output(logical_src, logical, native);
            self.blit_and_present_from(CpuSrc::Output, native);
        }
    }

    /// RP.1 — box-downfilter the march-resolution [`Self::framebuffer`]
    /// (`logical × ssaa`) into the logical-size [`Self::resolve`] buffer and
    /// return [`CpuSrc::Resolve`]. When `ssaa == 1` the framebuffer is already
    /// the logical image, so this is a no-op returning [`CpuSrc::Frame`].
    fn resolve_scene(&mut self, logical: (u32, u32)) -> CpuSrc {
        // `ssaa == 1` + no posterize ⇒ the framebuffer is already the final
        // logical image (RP.0/RP.1 fast path, byte-identical).
        if !self.resolve_active() {
            return CpuSrc::Frame;
        }
        let (lw, lh) = (logical.0 as usize, logical.1 as usize);
        let s = self.ssaa as usize;
        let (mw, mh) = (lw * s, lh * s);
        let lpc = lw * lh;
        if self.framebuffer.len() < mw * mh {
            return CpuSrc::Frame; // not yet rendered at this size; nothing to do
        }
        if self.resolve.len() < lpc {
            self.resolve.resize(lpc, self.clear_sky);
        }
        let post = self.posterize;
        let Self {
            framebuffer,
            resolve,
            ..
        } = self;
        for ly in 0..lh {
            for lx in 0..lw {
                // RP.1 box downfilter (1×1 = copy when ssaa==1), then RP.2
                // posterize + dither at the logical resolution.
                let mut px = downfilter_pixel(framebuffer, mw, lx, ly, s);
                if let Some(cfg) = post {
                    px = posterize_pixel(px, lx, ly, cfg);
                }
                resolve[ly * lw + lx] = px;
            }
        }
        CpuSrc::Resolve
    }

    /// No GPU work in flight on the software backend — teardown is a plain
    /// drop. Present for facade parity (see
    /// [`SceneRenderer::wait_idle`](crate::SceneRenderer::wait_idle)).
    #[allow(clippy::unused_self)]
    pub(crate) fn wait_idle(&mut self) {}

    /// Rasterise depth-tested world-space [`Line3`] segments over the
    /// framebuffer the last [`render`](Self::render) composited. Uses that
    /// frame's pinhole projection (`last_hxyz` / `last_dims`) and z-buffer
    /// (perpendicular distance, smaller = closer, sky = `+inf`), so the
    /// rendered terrain occludes lines behind it. Call after `render`,
    /// before `present` / `paint_egui`.
    pub(crate) fn draw_lines(&mut self, camera: &Camera, lines: &[Line3]) {
        let (w, h) = self.last_dims;
        let (hx, hy, hz) = self.last_hxyz;
        if w == 0 || h == 0 || hz <= 0.0 {
            return; // nothing rendered yet — no projection to reuse
        }
        let pixel_count = (w as usize) * (h as usize);
        if self.framebuffer.len() < pixel_count || self.zbuffer.len() < pixel_count {
            return;
        }
        let cam = camera_math::derive(camera, w, h, hx, hy, hz);
        // World point → camera-relative (right, down, forward) coords.
        // The forward component is the CPU z-buffer metric (perpendicular
        // distance); right/down drive the pinhole screen projection.
        let cam_coords = |p: [f32; 3]| -> [f32; 3] {
            let d = [p[0] - cam.pos[0], p[1] - cam.pos[1], p[2] - cam.pos[2]];
            [
                cam.right[0] * d[0] + cam.right[1] * d[1] + cam.right[2] * d[2],
                cam.down[0] * d[0] + cam.down[1] * d[1] + cam.down[2] * d[2],
                cam.forward[0] * d[0] + cam.forward[1] * d[1] + cam.forward[2] * d[2],
            ]
        };

        let fb = &mut self.framebuffer[..pixel_count];
        let zb = &self.zbuffer[..pixel_count];
        let (wi, hi) = (w as i32, h as i32);

        for line in lines {
            let a = [line.a[0] as f32, line.a[1] as f32, line.a[2] as f32];
            let b = [line.b[0] as f32, line.b[1] as f32, line.b[2] as f32];
            let ca = cam_coords(a);
            let cb = cam_coords(b);

            // Near-plane clip in segment-parameter space (forward depth
            // `cz >= NEAR_Z`) so the pinhole divide stays finite and points
            // behind the camera don't wrap. Both behind → invisible.
            let (cza, czb) = (ca[2], cb[2]);
            if cza < NEAR_Z && czb < NEAR_Z {
                continue;
            }
            let (mut t0, mut t1) = (0.0f32, 1.0f32);
            let dz = czb - cza;
            if dz.abs() > f32::EPSILON {
                let t_near = (NEAR_Z - cza) / dz;
                if dz > 0.0 {
                    t0 = t0.max(t_near); // a is behind: enter at the near plane
                } else {
                    t1 = t1.min(t_near); // b is behind: leave at the near plane
                }
            }
            if t0 > t1 {
                continue;
            }
            let lerp3 = |t: f32| {
                [
                    ca[0] + (cb[0] - ca[0]) * t,
                    ca[1] + (cb[1] - ca[1]) * t,
                    ca[2] + (cb[2] - ca[2]) * t,
                ]
            };
            let p0 = lerp3(t0);
            let p1 = lerp3(t1);

            // Pinhole project; carry 1/cz for perspective-correct depth
            // (1/cz is linear in screen space, cz is not).
            let inv0 = 1.0 / p0[2];
            let inv1 = 1.0 / p1[2];
            let sx0 = hx + p0[0] * hz * inv0;
            let sy0 = hy + p0[1] * hz * inv0;
            let sx1 = hx + p1[0] * hz * inv1;
            let sy1 = hy + p1[1] * hz * inv1;

            let alpha = (line.color.0 >> 24) & 0xff;
            if alpha == 0 {
                continue; // fully transparent
            }
            let rgb = line.color.0 & 0x00ff_ffff;

            // DDA along the dominant screen axis, stamping `width_px`
            // pixels perpendicular to the segment.
            let dx = sx1 - sx0;
            let dy = sy1 - sy0;
            let steps = dx.abs().max(dy.abs()).ceil().max(1.0);
            let len = (dx * dx + dy * dy).sqrt().max(1e-6);
            let (perp_x, perp_y) = (-dy / len, dx / len);
            let half = ((line.width_px - 1.0).max(0.0) * 0.5).round() as i32;

            let nsteps = steps as i32;
            for s in 0..=nsteps {
                let t = s as f32 / steps;
                let inv_z = inv0 + (inv1 - inv0) * t;
                let depth = 1.0 / inv_z; // perpendicular distance at this pixel
                let cx = sx0 + dx * t;
                let cy = sy0 + dy * t;
                for woff in -half..=half {
                    let px = (cx + perp_x * woff as f32).round() as i32;
                    let py = (cy + perp_y * woff as f32).round() as i32;
                    if px < 0 || py < 0 || px >= wi || py >= hi {
                        continue;
                    }
                    let idx = (py as usize) * (w as usize) + (px as usize);
                    if line.depth_test && depth > zb[idx] + DEPTH_BIAS {
                        continue; // occluded by nearer rendered geometry
                    }
                    fb[idx] = blend_rgb(fb[idx], rgb, alpha);
                }
            }
        }
    }

    /// Upload (or replace) an RGBA8 image; reuses a freed slot when one
    /// exists, else appends. See [`SceneRenderer::upload_image`].
    /// Returns the SLOT the image landed in (append or reuse); the
    /// facade owns the generational handle. Input is facade-validated.
    pub(crate) fn upload_image(&mut self, rgba: &[u8], width: u32, height: u32) -> usize {
        debug_assert!(
            width > 0 && height > 0 && rgba.len() == (width as usize) * (height as usize) * 4
        );
        let img = CpuImage {
            rgba: rgba.to_vec(),
            width,
            height,
        };
        if let Some(slot) = self.images.iter().position(Option::is_none) {
            self.images[slot] = Some(img);
            slot
        } else {
            self.images.push(Some(img));
            self.images.len() - 1
        }
    }

    /// Release a previously uploaded image (the slot becomes reusable).
    pub(crate) fn drop_image(&mut self, slot: usize) {
        if let Some(s) = self.images.get_mut(slot) {
            *s = None;
        }
    }

    /// Source `(width, height)` of an uploaded image, for `pick_image`.
    pub(crate) fn image_dims(&self, slot: usize) -> Option<(u32, u32)> {
        self.images
            .get(slot)
            .and_then(Option::as_ref)
            .map(|img| (img.width, img.height))
    }

    /// Alpha byte of texel `(tx, ty)`; `0` for an unknown id / out-of-range.
    pub(crate) fn image_alpha_at(&self, slot: usize, tx: u32, ty: u32) -> u8 {
        let Some(Some(img)) = self.images.get(slot) else {
            return 0;
        };
        if tx >= img.width || ty >= img.height {
            return 0;
        }
        let idx = ((ty * img.width + tx) * 4 + 3) as usize;
        img.rgba.get(idx).copied().unwrap_or(0)
    }

    /// Project a world point to window pixels under the last frame's
    /// `setcamera` projection. See [`SceneRenderer::project_point`].
    pub(crate) fn project_point(&self, camera: &Camera, world: [f32; 3]) -> Option<(f32, f32)> {
        let (hx, hy, hz) = self.last_hxyz;
        let (w, h) = self.last_dims;
        if hz <= 0.0 || w == 0 || h == 0 {
            return None;
        }
        let cam = camera_math::derive(camera, w, h, hx, hy, hz);
        let d = [
            world[0] - cam.pos[0],
            world[1] - cam.pos[1],
            world[2] - cam.pos[2],
        ];
        let cz = cam.forward[0] * d[0] + cam.forward[1] * d[1] + cam.forward[2] * d[2];
        if cz < NEAR_Z {
            return None;
        }
        let cx = cam.right[0] * d[0] + cam.right[1] * d[1] + cam.right[2] * d[2];
        let cy = cam.down[0] * d[0] + cam.down[1] * d[1] + cam.down[2] * d[2];
        Some((hx + cx * hz / cz, hy + cy * hz / cz))
    }

    /// Rasterise world-space textured quads ([`QuadDraw`]) over the
    /// framebuffer the last [`render`](Self::render) composited, with
    /// perspective-correct UVs and the same depth buffer the world pass
    /// filled (so the terrain occludes quads behind it). Nearest-neighbour
    /// sampling, straight-alpha tint, over-blend. Call after `render`,
    /// before `present` / `paint_egui`.
    pub(crate) fn draw_images(&mut self, camera: &Camera, quads: &[QuadDraw]) {
        let (w, h) = self.last_dims;
        let (hx, hy, hz) = self.last_hxyz;
        if w == 0 || h == 0 || hz <= 0.0 {
            return; // nothing rendered yet — no projection to reuse
        }
        let pixel_count = (w as usize) * (h as usize);
        if self.framebuffer.len() < pixel_count || self.zbuffer.len() < pixel_count {
            return;
        }
        let cam = camera_math::derive(camera, w, h, hx, hy, hz);
        let cam_coords = |p: [f32; 3]| -> [f32; 3] {
            let d = [p[0] - cam.pos[0], p[1] - cam.pos[1], p[2] - cam.pos[2]];
            [
                cam.right[0] * d[0] + cam.right[1] * d[1] + cam.right[2] * d[2],
                cam.down[0] * d[0] + cam.down[1] * d[1] + cam.down[2] * d[2],
                cam.forward[0] * d[0] + cam.forward[1] * d[1] + cam.forward[2] * d[2],
            ]
        };

        let fb = &mut self.framebuffer[..pixel_count];
        let zb = &self.zbuffer[..pixel_count];

        for quad in quads {
            let Some(Some(image)) = self.images.get(quad.image) else {
                continue; // dropped or never-uploaded id
            };
            let [tl, tr, bl, br] = quad.corners;
            // Per-corner UV: TL(0,0) TR(1,0) BL(0,1) BR(1,1).
            let verts = [
                ClipVert {
                    cam: cam_coords(tl),
                    uv: [0.0, 0.0],
                },
                ClipVert {
                    cam: cam_coords(tr),
                    uv: [1.0, 0.0],
                },
                ClipVert {
                    cam: cam_coords(bl),
                    uv: [0.0, 1.0],
                },
                ClipVert {
                    cam: cam_coords(br),
                    uv: [1.0, 1.0],
                },
            ];
            // Two triangles: (TL, TR, BL) and (TR, BR, BL).
            for tri in [[0usize, 1, 2], [1, 3, 2]] {
                let poly = [verts[tri[0]], verts[tri[1]], verts[tri[2]]];
                let clipped = clip_near(&poly);
                if clipped.len() < 3 {
                    continue;
                }
                // Project once, then fan-triangulate the clipped polygon.
                let screen: Vec<ScreenVert> = clipped
                    .iter()
                    .map(|v| project_clip(*v, hx, hy, hz))
                    .collect();
                for i in 1..screen.len() - 1 {
                    fill_textured_tri(
                        fb,
                        zb,
                        w,
                        h,
                        &screen[0],
                        &screen[i],
                        &screen[i + 1],
                        image,
                        quad.tint.0,
                        quad.depth_test,
                        quad.alpha_cutoff,
                    );
                }
            }
        }
    }

    /// Borrow the owned buffer a [`CpuSrc`] names (disjoint from the rest of
    /// `self` via destructure at the call site).
    fn src_slice<'a>(
        framebuffer: &'a [u32],
        resolve: &'a [u32],
        output: &'a [u32],
        src: CpuSrc,
    ) -> &'a [u32] {
        match src {
            CpuSrc::Frame => framebuffer,
            CpuSrc::Resolve => resolve,
            CpuSrc::Output => output,
        }
    }

    /// RP.0/RP.1 — nearest-upscale the logical scene buffer `src` (`logical`
    /// dims) into the native-size [`Self::output`] (`native` dims). Each native
    /// pixel samples the logical texel at `floor(x·lw/nw, y·lh/nh)` (hard
    /// pixels — the retro look). Only used when `logical != native`.
    fn upscale_to_output(&mut self, src: CpuSrc, logical: (u32, u32), native: (u32, u32)) {
        let (lw, lh) = (logical.0 as usize, logical.1 as usize);
        let (nw, nh) = (native.0 as usize, native.1 as usize);
        let (npc, lpc) = (nw * nh, lw * lh);
        if lw == 0 || lh == 0 || nw == 0 || nh == 0 {
            return;
        }
        if self.output.len() < npc {
            self.output.resize(npc, self.clear_sky);
        }
        let Self {
            framebuffer,
            resolve,
            output,
            ..
        } = self;
        let src_buf = Self::src_slice(framebuffer, resolve, &[], src);
        if src_buf.len() < lpc {
            return;
        }
        for y in 0..nh {
            let sy = (y * lh) / nh;
            let src_row = sy * lw;
            let dst_row = y * nw;
            for x in 0..nw {
                let sx = (x * lw) / nw;
                output[dst_row + x] = src_buf[src_row + sx];
            }
        }
    }

    /// Shared tail of `present` / `paint_egui`: copy the [`CpuSrc`] buffer to
    /// the window surface at `(width, height)` and present. The destructure
    /// gives disjoint borrows of `present_target` and the source slice.
    #[cfg(not(target_arch = "wasm32"))]
    fn blit_and_present_from(&mut self, src: CpuSrc, dims: (u32, u32)) {
        let (width, height) = dims;
        let (Some(w_nz), Some(h_nz)) = (NonZeroU32::new(width), NonZeroU32::new(height)) else {
            return;
        };
        let pixel_count = (width as usize) * (height as usize);
        let Self {
            present_target,
            framebuffer,
            resolve,
            output,
            ..
        } = self;
        let src: &[u32] = Self::src_slice(framebuffer, resolve, output, src);
        if src.len() < pixel_count {
            return;
        }
        present_target
            .resize(w_nz, h_nz)
            .expect("softbuffer: resize");
        let mut buffer = present_target.buffer_mut().expect("softbuffer: buffer_mut");
        buffer[..pixel_count].copy_from_slice(&src[..pixel_count]);
        buffer.present().expect("softbuffer: present");
    }

    /// wasm counterpart: upload the source buffer to the WebGL2 texture
    /// and draw the fullscreen quad on the canvas.
    #[cfg(target_arch = "wasm32")]
    fn blit_and_present_from(&mut self, src: CpuSrc, dims: (u32, u32)) {
        let (width, height) = dims;
        let pixel_count = (width as usize) * (height as usize);
        if width == 0 || height == 0 {
            return;
        }
        self.present_target.resize(width, height);
        // Destructure so `present_target` (mut) and the source buffer
        // (shared) don't alias through `self`.
        let Self {
            present_target,
            framebuffer,
            resolve,
            output,
            ..
        } = self;
        let src: &[u32] = Self::src_slice(framebuffer, resolve, output, src);
        if src.len() < pixel_count {
            return;
        }
        present_target.present(&src[..pixel_count]);
    }

    /// Software-rasterise the egui `jobs` over the composited
    /// framebuffer, then present (`hud` feature). Replaces
    /// [`Self::present`] for the UI-overlay path.
    #[cfg(feature = "hud")]
    pub(crate) fn paint_egui(
        &mut self,
        jobs: &[egui::ClippedPrimitive],
        textures: &egui::TexturesDelta,
        pixels_per_point: f32,
    ) {
        let logical = self.logical_dims();
        let native = self.current_dims;
        let lpc = (logical.0 as usize) * (logical.1 as usize);
        // Mirror the 3D scene (in render space) before the UI is drawn over
        // it, so the egui overlay stays upright.
        if self.flip_x {
            self.flip_framebuffer();
        }
        // RP.1 — resolve march → logical (box downfilter) when supersampling.
        let logical_src = self.resolve_scene(logical);
        self.egui_raster
            .update_textures(&textures.set, &textures.free);
        if logical == native {
            // Logical == window — rasterise egui straight into the logical
            // buffer at window res (Native + ssaa==1 ⇒ pre-RP), then present.
            let Self {
                framebuffer,
                resolve,
                egui_raster,
                ..
            } = self;
            let buf: &mut [u32] = match logical_src {
                CpuSrc::Resolve => &mut resolve[..lpc],
                _ => &mut framebuffer[..lpc],
            };
            egui_raster.paint(buf, native.0, native.1, jobs, pixels_per_point);
            self.blit_and_present_from(logical_src, native);
        } else {
            // Fixed/scaled — upscale the scene first, then rasterise egui into
            // the native-size output so the HUD stays crisp (RP.0 locked #6).
            self.upscale_to_output(logical_src, logical, native);
            let npc = (native.0 as usize) * (native.1 as usize);
            self.egui_raster.paint(
                &mut self.output[..npc],
                native.0,
                native.1,
                jobs,
                pixels_per_point,
            );
            self.blit_and_present_from(CpuSrc::Output, native);
        }
    }
}

#[cfg(test)]
mod posterize_tests {
    use super::{posterize_pixel, quantize_channel};
    use crate::{DitherMode, PosterizeConfig};

    /// `levels <= 1` leaves a channel untouched (the byte-identical guard).
    #[test]
    fn levels_one_is_identity() {
        for c in [0, 1, 127, 128, 200, 255] {
            assert_eq!(quantize_channel(c, 1, 0.5), c);
            assert_eq!(quantize_channel(c, 0, 0.5), c);
        }
    }

    /// 2-level, no-dither quantization snaps to black/white at the midpoint.
    #[test]
    fn two_levels_round_to_nearest() {
        assert_eq!(quantize_channel(0, 2, 0.5), 0);
        assert_eq!(quantize_channel(127, 2, 0.5), 0);
        assert_eq!(quantize_channel(128, 2, 0.5), 255);
        assert_eq!(quantize_channel(255, 2, 0.5), 255);
    }

    /// 4 levels map to the evenly-spaced palette {0, 85, 170, 255}.
    #[test]
    fn four_levels_palette() {
        let p = |c| quantize_channel(c, 4, 0.5);
        assert_eq!(p(0), 0);
        assert_eq!(p(255), 255);
        assert_eq!(p(85), 85);
        assert_eq!(p(170), 170);
        // Every output is one of the 4 levels.
        for c in 0..=255u32 {
            assert!(matches!(p(c), 0 | 85 | 170 | 255), "c={c} → {}", p(c));
        }
    }

    /// `None` posterize round-trips the whole pixel unchanged (per channel
    /// `levels == 1`), and a uniform config touches every channel.
    #[test]
    fn posterize_pixel_per_channel() {
        let cfg = PosterizeConfig::uniform(2, DitherMode::None);
        // r=200→255, g=10→0, b=130→255.
        assert_eq!(posterize_pixel(0x00_c8_0a_82, 0, 0, cfg), 0x00_ff_00_ff);
    }

    /// Dither pushes a near-boundary value across the threshold for some
    /// pixels but not others (so a flat ramp breaks into a stable pattern).
    #[test]
    fn dither_varies_by_pixel() {
        let cfg = PosterizeConfig::uniform(2, DitherMode::Bayer4x4);
        // Mid-grey 0x80 sits right at the 2-level boundary; Bayer must yield
        // both black and white across the 4×4 tile.
        let mut blacks = 0;
        let mut whites = 0;
        for y in 0..4 {
            for x in 0..4 {
                match posterize_pixel(0x00_80_80_80, x, y, cfg) & 0xff {
                    0 => blacks += 1,
                    255 => whites += 1,
                    other => panic!("unexpected {other}"),
                }
            }
        }
        assert!(blacks > 0 && whites > 0, "blacks={blacks} whites={whites}");
    }
}

#[cfg(test)]
mod downfilter_tests {
    use super::downfilter_pixel;

    /// `ssaa == 1` is the identity — every source pixel passes through
    /// unchanged (the byte-identical guarantee for the non-SSAA path).
    #[test]
    fn ssaa1_is_identity() {
        let fb = [0x00_12_34_56, 0x00_ab_cd_ef, 0x00_00_00_00, 0x00_ff_ff_ff];
        for (i, &px) in fb.iter().enumerate() {
            assert_eq!(downfilter_pixel(&fb, 2, i % 2, i / 2, 1), px);
        }
    }

    /// A uniform `s × s` block resolves to that exact colour (no drift).
    #[test]
    fn uniform_block_is_exact() {
        let fb = vec![0x00_40_80_c0_u32; 16]; // 4×4 march
        assert_eq!(downfilter_pixel(&fb, 4, 0, 0, 2), 0x00_40_80_c0);
        assert_eq!(downfilter_pixel(&fb, 4, 1, 1, 2), 0x00_40_80_c0);
    }

    /// 2×2 average with round-to-nearest, per channel independently.
    #[test]
    fn averages_with_rounding() {
        // Red: 0,0,0,2 → 2/4 = 0.5 → 1. Green: 10,10,10,10 → 10.
        // Blue: 0,1,2,3 → 6/4 = 1.5 → 2.
        let fb = [0x00_00_0a_00, 0x00_00_0a_01, 0x00_00_0a_02, 0x00_02_0a_03];
        assert_eq!(downfilter_pixel(&fb, 2, 0, 0, 2), 0x00_01_0a_02);
    }
}

#[cfg(test)]
mod cpu_ray_tests {
    use super::setcamera_pixel_ray;

    const RIGHT: [f64; 3] = [1.0, 0.0, 0.0];
    const DOWN: [f64; 3] = [0.0, 1.0, 0.0];
    const FWD: [f64; 3] = [0.0, 0.0, 1.0]; // voxlap z-down "look down"

    // Centre pixel (hx, hy) → straight along `forward`.
    #[test]
    fn centre_pixel_is_forward() {
        let d = setcamera_pixel_ray(RIGHT, DOWN, FWD, 320.0, 240.0, 320.0, 240.0, 320.0);
        assert_eq!(d, [0.0, 0.0, 320.0]);
    }

    // Off-centre pixel tilts proportionally: (px-hx, py-hy, hz).
    #[test]
    fn offcentre_pixel_tilts_linearly() {
        let d = setcamera_pixel_ray(RIGHT, DOWN, FWD, 384.0, 272.0, 320.0, 240.0, 320.0);
        assert_eq!(d, [64.0, 32.0, 320.0]);
    }
}

#[cfg(test)]
mod image_raster_tests {
    use super::{clip_near, fill_textured_tri, ClipVert, CpuImage, ScreenVert, NEAR_Z};

    fn cv(cam: [f32; 3], uv: [f32; 2]) -> ClipVert {
        ClipVert { cam, uv }
    }

    #[test]
    fn clip_near_keeps_a_front_triangle() {
        let tri = [
            cv([0.0, 0.0, 10.0], [0.0, 0.0]),
            cv([1.0, 0.0, 10.0], [1.0, 0.0]),
            cv([0.0, 1.0, 10.0], [0.0, 1.0]),
        ];
        assert_eq!(clip_near(&tri).len(), 3, "fully in front: unchanged");
    }

    #[test]
    fn clip_near_splits_a_straddling_triangle() {
        // One vertex behind the near plane → the clipped polygon gains a
        // vertex (two edges cross the plane).
        let tri = [
            cv([0.0, 0.0, -1.0], [0.0, 0.0]), // behind
            cv([1.0, 0.0, 10.0], [1.0, 0.0]),
            cv([0.0, 1.0, 10.0], [0.0, 1.0]),
        ];
        let out = clip_near(&tri);
        assert_eq!(out.len(), 4, "one-behind triangle clips to a quad");
        for v in &out {
            assert!(v.cam[2] >= NEAR_Z - 1e-6, "no vertex behind the near plane");
        }
    }

    /// Render a screen-aligned quad (constant forward depth) over a 10×10
    /// framebuffer from a 2×2 colour image and read back corner pixels.
    fn render_quad(depth_test: bool, zb_fill: f32) -> Vec<u32> {
        render_quad_cutoff(depth_test, zb_fill, 0.0)
    }

    fn render_quad_cutoff(depth_test: bool, zb_fill: f32, alpha_cutoff: f32) -> Vec<u32> {
        // 2×2: TL red, TR green, BL blue, BR white (row-major RGBA8).
        let rgba = vec![
            255, 0, 0, 255, /* (0,0) */ 0, 255, 0, 255, /* (1,0) */
            0, 0, 255, 255, /* (0,1) */ 255, 255, 255, 255, /* (1,1) */
        ];
        let image = CpuImage {
            rgba,
            width: 2,
            height: 2,
        };
        let (w, h) = (10u32, 10u32);
        let mut fb = vec![0u32; (w * h) as usize];
        let zb = vec![zb_fill; (w * h) as usize];

        let fwd = 10.0f32;
        let iw = 1.0 / fwd;
        // Quad corners in screen space, UVs TL(0,0) TR(1,0) BL(0,1) BR(1,1).
        let sv = |sx: f32, sy: f32, u: f32, v: f32| ScreenVert {
            sx,
            sy,
            inv_w: iw,
            su: u * iw,
            sv: v * iw,
        };
        let tl = sv(0.0, 0.0, 0.0, 0.0);
        let tr = sv(10.0, 0.0, 1.0, 0.0);
        let bl = sv(0.0, 10.0, 0.0, 1.0);
        let br = sv(10.0, 10.0, 1.0, 1.0);
        for tri in [[tl, tr, bl], [tr, br, bl]] {
            fill_textured_tri(
                &mut fb,
                &zb,
                w,
                h,
                &tri[0],
                &tri[1],
                &tri[2],
                &image,
                0xFFFF_FFFF,
                depth_test,
                alpha_cutoff,
            );
        }
        fb
    }

    #[test]
    fn textured_quad_maps_uv_corners() {
        let fb = render_quad(false, f32::INFINITY);
        let at = |x: u32, y: u32| fb[(y * 10 + x) as usize];
        // Corners sample the matching texel (top-left of image = TL of quad).
        assert_eq!(at(1, 1), 0x00FF_0000, "TL → red");
        assert_eq!(at(8, 1), 0x0000_FF00, "TR → green");
        assert_eq!(at(1, 8), 0x0000_00FF, "BL → blue");
        assert_eq!(at(8, 8), 0x00FF_FFFF, "BR → white");
    }

    #[test]
    fn depth_test_occludes_quad_behind_geometry() {
        // Quad at forward distance 10; z-buffer says geometry is at 5
        // everywhere → the whole quad is occluded and nothing is written.
        let fb = render_quad(true, 5.0);
        assert!(fb.iter().all(|&p| p == 0), "occluded quad writes nothing");
    }

    #[test]
    fn depth_test_passes_when_in_front() {
        // Geometry behind the quad (z-buffer at 100) → quad draws.
        let fb = render_quad(true, 100.0);
        assert!(fb.iter().any(|&p| p != 0), "unoccluded quad draws");
    }

    /// A half-transparent (alpha 100) texel draws below its cutoff and is
    /// discarded above it.
    #[test]
    fn alpha_cutoff_discards_below_threshold() {
        let image = CpuImage {
            rgba: vec![255, 255, 255, 100], // white, alpha 100/255
            width: 1,
            height: 1,
        };
        let render = |cutoff: f32| {
            let (w, h) = (4u32, 4u32);
            let mut fb = vec![0u32; (w * h) as usize];
            let zb = vec![f32::INFINITY; (w * h) as usize];
            let iw = 0.1f32;
            let sv = |sx: f32, sy: f32, u: f32, v: f32| ScreenVert {
                sx,
                sy,
                inv_w: iw,
                su: u * iw,
                sv: v * iw,
            };
            let tl = sv(0.0, 0.0, 0.0, 0.0);
            let tr = sv(4.0, 0.0, 1.0, 0.0);
            let bl = sv(0.0, 4.0, 0.0, 1.0);
            let br = sv(4.0, 4.0, 1.0, 1.0);
            for tri in [[tl, tr, bl], [tr, br, bl]] {
                fill_textured_tri(
                    &mut fb,
                    &zb,
                    w,
                    h,
                    &tri[0],
                    &tri[1],
                    &tri[2],
                    &image,
                    0xFFFF_FFFF,
                    false,
                    cutoff,
                );
            }
            fb
        };
        // 100/255 ≈ 0.39 — below 0.3 draws, above 0.5 is discarded.
        assert!(
            render(0.3).iter().any(|&p| p != 0),
            "alpha 100 > cutoff 0.3 draws"
        );
        assert!(
            render(0.5).iter().all(|&p| p == 0),
            "alpha 100 < cutoff 0.5 discarded"
        );
    }
}

#[cfg(test)]
mod blend_tests {
    use super::blend_rgb;

    #[test]
    fn opaque_replaces_destination() {
        // alpha = 255 → source colour, ignoring the destination.
        assert_eq!(blend_rgb(0x00_12_34_56, 0xAA_BB_CC, 255), 0x00_AA_BB_CC);
    }

    #[test]
    fn zero_alpha_keeps_destination() {
        // alpha = 0 → 100% destination (the caller skips alpha==0, but
        // the blend itself must still be a no-op).
        assert_eq!(blend_rgb(0x00_12_34_56, 0xAA_BB_CC, 0), 0x00_12_34_56);
    }

    #[test]
    fn half_alpha_is_midpoint() {
        // white over black at ~50% → mid grey, per channel (255*128/255).
        let out = blend_rgb(0x00_00_00_00, 0x00_FF_FF_FF, 128);
        assert_eq!(out, 0x00_80_80_80);
    }

    #[test]
    fn result_has_no_high_byte() {
        // Output must stay 0x00RRGGBB to match the framebuffer packing.
        assert_eq!(
            blend_rgb(0x00_FF_FF_FF, 0xFF_FF_FF_FF, 200) & 0xFF00_0000,
            0
        );
    }
}
