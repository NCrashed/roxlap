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
use crate::{FrameParams, KfaSprite, Line3, RenderOptions, SpriteSet};

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
            kfa_limbs: Vec::new(),
            capture_next: false,
            captured: None,
            framebuffer,
            #[cfg(feature = "hud")]
            egui_raster: crate::cpu_egui::EguiRaster::default(),
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
            if !self.sprites.is_empty() || !self.kfa_limbs.is_empty() {
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
                for sprite in self.sprites.iter().chain(self.kfa_limbs.iter()) {
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
