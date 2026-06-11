//! CPU backend — `roxlap-core` opticast presented via `softbuffer`.
//!
//! RF.1: owns the software surface + the per-frame [`ScratchPool`] and
//! z-buffer, and runs the multi-grid opticast compositor
//! ([`render_scene_composed`]). Mirrors the scene-demo's old `redraw`
//! world pass. Sprites land in RF.3.

use std::num::NonZeroU32;
use std::sync::Arc;

use roxlap_core::camera_math;
use roxlap_core::rasterizer::ScratchPool;
use roxlap_core::sprite::{draw_sprite, DrawTarget};
use roxlap_core::Camera;
use roxlap_formats::sprite::Sprite;
use roxlap_scene::render::render_scene_composed;
use roxlap_scene::Scene;

use crate::{
    DynDisplay, DynWindow, FrameParams, HasDisplayHandle, HasWindowHandle, RenderOptions, SpriteSet,
};

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
    /// `softbuffer::Context` is dropped after surface creation — the
    /// surface keeps its own clone of the display handle (matches the
    /// existing scene-demo setup). The display/window handles are
    /// type-erased to `Arc<dyn …>` so the backend stays generic-free
    /// over the host's windowing library.
    surface: softbuffer::Surface<Arc<DynDisplay>, Arc<DynWindow>>,
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

        let (w, h) = (size.0.max(1), size.1.max(1));
        let n_threads = opts
            .cpu_render_threads
            .clamp(1, rayon::current_num_threads().max(1));
        let pool = ScratchPool::new_parallel(w, h, opts.cpu_max_grid_vsid, n_threads);
        let zbuffer = vec![f32::INFINITY; (w as usize) * (h as usize)];
        let framebuffer = vec![opts.clear_sky; (w as usize) * (h as usize)];

        Self {
            surface,
            current_dims: (w, h),
            pool,
            zbuffer,
            last_dims: (w, h),
            last_hxyz: (0.0, 0.0, 0.0),
            max_grid_vsid: opts.cpu_max_grid_vsid,
            n_threads,
            clear_sky: opts.clear_sky,
            sprites: Vec::new(),
            capture_next: false,
            captured: None,
            framebuffer,
            #[cfg(feature = "hud")]
            egui_raster: crate::cpu_egui::EguiRaster::default(),
        }
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
        for inst in &set.instances {
            if let Some(model) = set.models.get(inst.model) {
                let mut s = model.clone();
                s.p = inst.pos;
                sprites.push(s);
            }
        }
        self.sprites = sprites;
    }

    #[allow(clippy::unused_self)] // symmetry with GpuBackend::resize
    pub(crate) fn resize(&mut self, width: u32, height: u32) {
        // softbuffer + the pool resize lazily inside `render`; we just
        // record the new size the host reported (replacing the old
        // per-frame `window.inner_size()` poll).
        self.current_dims = (width.max(1), height.max(1));
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
            if !self.sprites.is_empty() {
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
                for sprite in &self.sprites {
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

    /// Shared tail of `present` / `paint_egui`: copy the framebuffer to
    /// the window surface at `(width, height)` and present.
    fn blit_and_present(&mut self, dims: (u32, u32)) {
        let (width, height) = dims;
        let (Some(w_nz), Some(h_nz)) = (NonZeroU32::new(width), NonZeroU32::new(height)) else {
            return;
        };
        let pixel_count = (width as usize) * (height as usize);
        if self.framebuffer.len() < pixel_count {
            return;
        }
        self.surface.resize(w_nz, h_nz).expect("softbuffer: resize");
        let mut buffer = self.surface.buffer_mut().expect("softbuffer: buffer_mut");
        buffer[..pixel_count].copy_from_slice(&self.framebuffer[..pixel_count]);
        buffer.present().expect("softbuffer: present");
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
