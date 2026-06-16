//! CPU backend — `roxlap-core` opticast presented via `softbuffer`.
//!
//! RF.1: owns the software surface + the per-frame [`ScratchPool`] and
//! z-buffer, and runs the multi-grid opticast compositor
//! ([`render_scene_composed`]). Mirrors the scene-demo's old `redraw`
//! world pass. Sprites land in RF.3.

#[cfg(not(target_arch = "wasm32"))]
use std::num::NonZeroU32;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;

use roxlap_core::camera_math;
use roxlap_core::kfa_draw::solve_kfa_limbs;
use roxlap_core::rasterizer::ScratchPool;
use roxlap_core::sprite::{draw_sprite, DrawTarget};
use roxlap_core::Camera;
use roxlap_formats::kv6::Kv6;
use roxlap_formats::sprite::Sprite;
use roxlap_scene::render::render_scene_composed;
use roxlap_scene::Scene;

#[cfg(not(target_arch = "wasm32"))]
use crate::{DynDisplay, DynWindow, HasDisplayHandle, HasWindowHandle};
use crate::{FrameParams, ImageId, KfaSprite, Line3, QuadDraw, RenderOptions, SpriteSet};

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

pub(crate) struct CpuBackend {
    /// Framebuffer presenter — native `softbuffer` window surface, or
    /// the wasm WebGL2 canvas blitter (see [`Presenter`]). On native,
    /// `softbuffer::Context` is dropped after surface creation; the
    /// surface keeps its own clone of the type-erased `Arc<dyn …>`
    /// display/window handles so the backend stays generic-free over
    /// the host's windowing library.
    present_target: Presenter,
    /// Current framebuffer size in physical pixels. Seeded at
    /// construction, updated by [`Self::resize`] — replaces the old
    /// per-frame `window.inner_size()` poll so the backend never
    /// touches a concrete window type.
    current_dims: (u32, u32),
    pool: ScratchPool,
    zbuffer: Vec<f32>,
    /// Framebuffer dimensions of the last `render` — the `zbuffer`
    /// stride for [`Self::pick_depth`].
    last_dims: (u32, u32),
    /// Opticast projection params `(hx, hy, hz)` of the last `render`,
    /// from its [`OpticastSettings`] — the CPU unproject for
    /// [`Self::pixel_ray`].
    last_hxyz: (f32, f32, f32),
    /// Widest combined-grid `vsid` the pool's `lastx` is sized for;
    /// kept so a window grow can re-create the pool.
    max_grid_vsid: u32,
    n_threads: usize,
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
    /// [`Self::add_dyn_instance`] can clone a model by id. The GPU backend
    /// keeps the analogous `sprite_models_tpl`.
    models: Vec<Sprite>,
    /// Dynamically added instances (see [`Self::add_dyn_instance`]) — a
    /// swap-removable tail sublist drawn after the static sprites, the CPU
    /// analogue of the GPU registry's appended instances.
    dyn_sprites: Vec<Sprite>,
    /// Source model index per entry in [`dyn_sprites`](Self::dyn_sprites),
    /// so [`Self::update_sprite_model`] refreshes dynamic instances too.
    dyn_models: Vec<usize>,
    /// Posed KFA limbs (flattened across all registered KFA sprites),
    /// refreshed by [`Self::update_kfa_poses`] and drawn after the
    /// static sprites each frame via `draw_sprite`.
    kfa_limbs: Vec<Sprite>,
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
        let n_threads = opts
            .cpu_render_threads
            .clamp(1, rayon::current_num_threads().max(1));
        let pool = ScratchPool::new_parallel(w, h, opts.cpu_max_grid_vsid, n_threads);
        let zbuffer = vec![f32::INFINITY; (w as usize) * (h as usize)];
        let framebuffer = vec![opts.clear_sky; (w as usize) * (h as usize)];

        Self {
            present_target,
            current_dims: (w, h),
            pool,
            zbuffer,
            last_dims: (w, h),
            last_hxyz: (0.0, 0.0, 0.0),
            max_grid_vsid: opts.cpu_max_grid_vsid,
            n_threads,
            clear_sky: opts.clear_sky,
            sprites: Vec::new(),
            sprite_models: Vec::new(),
            models: Vec::new(),
            dyn_sprites: Vec::new(),
            dyn_models: Vec::new(),
            kfa_limbs: Vec::new(),
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
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn new<W>(window: Arc<W>, size: (u32, u32), opts: &RenderOptions) -> Self
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
        let context = softbuffer::Context::new(display).expect("softbuffer: Context::new");
        let surface = softbuffer::Surface::new(&context, window).expect("softbuffer: Surface::new");
        Self::assemble(surface, size, opts)
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
        Some(setcamera_pixel_ray(
            camera.right,
            camera.down,
            camera.forward,
            x,
            y,
            hx,
            hy,
            hz,
        ))
    }

    /// World-t depth at pixel `(x, y)` from the last frame's z-buffer
    /// (already in CPU memory — no readback). `None` for out-of-bounds
    /// or sky (`+INF`). See [`SceneRenderer::pick_depth`].
    pub(crate) fn pick_depth(&self, x: u32, y: u32) -> Option<f32> {
        let (w, h) = self.last_dims;
        if x >= w || y >= h {
            return None;
        }
        let t = *self.zbuffer.get((y * w + x) as usize)?;
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
    }

    /// Append one dynamic instance of `model_index` at `pos`; returns its
    /// dynamic-sublist index (always the new last). The facade wraps this
    /// in a stable handle. No-op-ish (returns the current count) if the
    /// model id is unknown.
    pub(crate) fn add_dyn_instance(&mut self, model_index: usize, pos: [f32; 3]) -> usize {
        let idx = self.dyn_sprites.len();
        if let Some(model) = self.models.get(model_index) {
            let mut s = model.clone();
            s.p = pos;
            self.dyn_sprites.push(s);
            self.dyn_models.push(model_index);
        }
        idx
    }

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

    /// GPU.12 incremental — swap the edited `kv6` into every cached
    /// instance of host model `model_index`, keeping each instance's
    /// world position. Mirrors the GPU backend's single-model update on
    /// the software path (where "rebuild" is just a kv6 clone per
    /// instance, so the win is parity rather than bandwidth). No-op if no
    /// instance references `model_index`.
    pub(crate) fn update_sprite_model(&mut self, model_index: usize, kv6: &Kv6) {
        for (s, &m) in self.sprites.iter_mut().zip(&self.sprite_models) {
            if m == model_index {
                s.kv6 = kv6.clone();
            }
        }
        // Dynamic instances of the same model refresh too.
        for (s, &m) in self.dyn_sprites.iter_mut().zip(&self.dyn_models) {
            if m == model_index {
                s.kv6 = kv6.clone();
            }
        }
        // Keep the stored template current so future dynamic adds use it.
        if let Some(t) = self.models.get_mut(model_index) {
            t.kv6 = kv6.clone();
        }
    }

    /// Register KFA sprites — for the CPU backend this is the same as a
    /// pose refresh: solve every limb's world transform from its
    /// current `kfaval[]` and cache the resulting [`Sprite`]s.
    pub(crate) fn set_kfa_sprites(&mut self, kfas: &mut [KfaSprite]) {
        self.update_kfa_poses(kfas);
    }

    /// Re-solve every KFA limb's world transform and cache the posed
    /// [`Sprite`]s for the next [`Self::render`].
    pub(crate) fn update_kfa_poses(&mut self, kfas: &mut [KfaSprite]) {
        self.kfa_limbs.clear();
        for kfa in kfas.iter_mut() {
            solve_kfa_limbs(kfa);
            self.kfa_limbs.extend(kfa.limbs.iter().cloned());
        }
    }

    pub(crate) fn resize(&mut self, width: u32, height: u32) {
        // softbuffer + the pool resize lazily inside `render`; we just
        // record the new size the host reported (replacing the old
        // per-frame `window.inner_size()` poll). The WebGL2 blitter's
        // texture, by contrast, must be re-allocated eagerly.
        self.current_dims = (width.max(1), height.max(1));
        #[cfg(target_arch = "wasm32")]
        self.present_target
            .resize(self.current_dims.0, self.current_dims.1);
    }

    pub(crate) fn render(&mut self, scene: &mut Scene, camera: &Camera, frame: &FrameParams) {
        let (width, height) = self.current_dims;
        if width == 0 || height == 0 {
            return;
        }
        let pixel_count = (width as usize) * (height as usize);
        self.last_dims = (width, height);
        self.last_hxyz = (frame.settings.hx, frame.settings.hy, frame.settings.hz);

        // Grow the z-buffer + pool to follow a window resize.
        if self.zbuffer.len() < pixel_count {
            self.zbuffer.resize(pixel_count, f32::INFINITY);
        }
        if self.pool.slot(0).uurend_half_stride < width as usize {
            self.pool =
                ScratchPool::new_parallel(width, height, self.max_grid_vsid, self.n_threads);
        }

        // Per-frame pool config (engine sky/fog → rasterizer). The
        // rasterizer takes packed colours as `i32`; reinterpret the
        // bits (not a numeric cast).
        let sky_i = i32::from_ne_bytes(frame.sky_color.to_ne_bytes());
        self.pool.set_skycast(sky_i, 0);
        let fog_i = i32::from_ne_bytes(frame.fog_color.to_ne_bytes());
        self.pool.set_fog(fog_i, frame.fog_max_scan_dist);
        self.pool.set_treat_z_max_as_air(frame.treat_z_max_as_air);
        // Per-face grid shading (voxlap setsideshades) — the grid-scan
        // analogue of sprite_lighting. Default [0;6] keeps sideshademode
        // off (byte-identical to the no-side-shade path).
        let [top, bot, left, right, up, down] = frame.side_shades;
        self.pool.set_side_shades(top, bot, left, right, up, down);

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

        let _outcome = render_scene_composed(
            fb,
            &mut self.zbuffer[..pixel_count],
            width as usize,
            width,
            height,
            &mut self.pool,
            scene,
            camera,
            frame.settings,
            frame.sky_color,
            frame.sky,
        );

        // Sprites layer on top of the heightmap world, z-tested against
        // the same z-buffer (camera-facing voxel splat). Needs the
        // host-built lighting; skipped if absent or no sprites.
        if let Some(lighting) = frame.sprite_lighting {
            if !self.sprites.is_empty()
                || !self.dyn_sprites.is_empty()
                || !self.kfa_limbs.is_empty()
            {
                let cam_state = camera_math::derive(
                    camera,
                    width,
                    height,
                    frame.settings.hx,
                    frame.settings.hy,
                    frame.settings.hz,
                );
                let mut target = DrawTarget::new(
                    fb,
                    &mut self.zbuffer[..pixel_count],
                    width as usize,
                    width,
                    height,
                );
                // Static sprites, then the posed KFA limbs (already
                // solved by `update_kfa_poses`); both z-test against the
                // shared buffer so order doesn't affect the result.
                for sprite in self
                    .sprites
                    .iter()
                    .chain(self.dyn_sprites.iter())
                    .chain(self.kfa_limbs.iter())
                {
                    let _written =
                        draw_sprite(&mut target, &cam_state, frame.settings, lighting, sprite);
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
        if self.flip_x {
            self.flip_framebuffer();
        }
        self.blit_and_present(self.last_dims);
    }

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

            let alpha = (line.color >> 24) & 0xff;
            if alpha == 0 {
                continue; // fully transparent
            }
            let rgb = line.color & 0x00ff_ffff;

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
    pub(crate) fn upload_image(&mut self, rgba: &[u8], width: u32, height: u32) -> ImageId {
        if width == 0 || height == 0 || rgba.len() != (width as usize) * (height as usize) * 4 {
            return ImageId(0); // malformed — a draw with this id is a no-op
        }
        let img = CpuImage {
            rgba: rgba.to_vec(),
            width,
            height,
        };
        if let Some(slot) = self.images.iter().position(Option::is_none) {
            self.images[slot] = Some(img);
            ImageId(slot)
        } else {
            self.images.push(Some(img));
            ImageId(self.images.len() - 1)
        }
    }

    /// Release a previously uploaded image (the slot becomes reusable).
    pub(crate) fn drop_image(&mut self, id: ImageId) {
        if let Some(slot) = self.images.get_mut(id.0) {
            *slot = None;
        }
    }

    /// Source `(width, height)` of an uploaded image, for `pick_image`.
    pub(crate) fn image_dims(&self, id: ImageId) -> Option<(u32, u32)> {
        self.images
            .get(id.0)
            .and_then(Option::as_ref)
            .map(|img| (img.width, img.height))
    }

    /// Alpha byte of texel `(tx, ty)`; `0` for an unknown id / out-of-range.
    pub(crate) fn image_alpha_at(&self, id: ImageId, tx: u32, ty: u32) -> u8 {
        let Some(Some(img)) = self.images.get(id.0) else {
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
            let Some(Some(image)) = self.images.get(quad.image.0) else {
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
                        quad.tint,
                        quad.depth_test,
                        quad.alpha_cutoff,
                    );
                }
            }
        }
    }

    /// Shared tail of `present` / `paint_egui`: copy the framebuffer to
    /// the window surface at `(width, height)` and present.
    #[cfg(not(target_arch = "wasm32"))]
    fn blit_and_present(&mut self, dims: (u32, u32)) {
        let (width, height) = dims;
        let (Some(w_nz), Some(h_nz)) = (NonZeroU32::new(width), NonZeroU32::new(height)) else {
            return;
        };
        let pixel_count = (width as usize) * (height as usize);
        if self.framebuffer.len() < pixel_count {
            return;
        }
        self.present_target
            .resize(w_nz, h_nz)
            .expect("softbuffer: resize");
        let mut buffer = self
            .present_target
            .buffer_mut()
            .expect("softbuffer: buffer_mut");
        buffer[..pixel_count].copy_from_slice(&self.framebuffer[..pixel_count]);
        buffer.present().expect("softbuffer: present");
    }

    /// wasm counterpart: upload the framebuffer to the WebGL2 texture
    /// and draw the fullscreen quad on the canvas.
    #[cfg(target_arch = "wasm32")]
    fn blit_and_present(&mut self, dims: (u32, u32)) {
        let (width, height) = dims;
        let pixel_count = (width as usize) * (height as usize);
        if width == 0 || height == 0 || self.framebuffer.len() < pixel_count {
            return;
        }
        self.present_target.resize(width, height);
        self.present_target
            .present(&self.framebuffer[..pixel_count]);
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
        let (width, height) = self.last_dims;
        let pixel_count = (width as usize) * (height as usize);
        if self.framebuffer.len() < pixel_count {
            return;
        }
        // Mirror the 3D scene before the UI is drawn over it, so the egui
        // overlay stays upright.
        if self.flip_x {
            self.flip_framebuffer();
        }
        self.egui_raster
            .update_textures(&textures.set, &textures.free);
        self.egui_raster.paint(
            &mut self.framebuffer[..pixel_count],
            width,
            height,
            jobs,
            pixels_per_point,
        );
        self.blit_and_present((width, height));
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
