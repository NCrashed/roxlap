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
use winit::window::Window;

use crate::{FrameParams, RenderOptions, SpriteSet};

pub(crate) struct CpuBackend {
    window: Arc<Window>,
    /// `softbuffer::Context` is dropped after surface creation — the
    /// surface keeps its own clone of the display handle (matches the
    /// existing scene-demo setup).
    surface: softbuffer::Surface<Arc<Window>, Arc<Window>>,
    pool: ScratchPool,
    zbuffer: Vec<f32>,
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
}

impl CpuBackend {
    pub(crate) fn new(window: Arc<Window>, opts: &RenderOptions) -> Self {
        let context = softbuffer::Context::new(window.clone()).expect("softbuffer: Context::new");
        let surface =
            softbuffer::Surface::new(&context, window.clone()).expect("softbuffer: Surface::new");

        let size = window.inner_size();
        let (w, h) = (size.width.max(1), size.height.max(1));
        let n_threads = opts
            .cpu_render_threads
            .clamp(1, rayon::current_num_threads().max(1));
        let pool = ScratchPool::new_parallel(w, h, opts.cpu_max_grid_vsid, n_threads);
        let zbuffer = vec![f32::INFINITY; (w as usize) * (h as usize)];

        Self {
            window,
            surface,
            pool,
            zbuffer,
            max_grid_vsid: opts.cpu_max_grid_vsid,
            n_threads,
            clear_sky: opts.clear_sky,
            sprites: Vec::new(),
            capture_next: false,
            captured: None,
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
    pub(crate) fn resize(&mut self, _width: u32, _height: u32) {
        // softbuffer + the pool resize lazily inside `render`.
    }

    pub(crate) fn render(&mut self, scene: &mut Scene, camera: &Camera, frame: &FrameParams) {
        let size = self.window.inner_size();
        let (Some(w_nz), Some(h_nz)) = (NonZeroU32::new(size.width), NonZeroU32::new(size.height))
        else {
            return;
        };
        let (width, height) = (size.width, size.height);
        let pixel_count = (width as usize) * (height as usize);

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

        self.surface.resize(w_nz, h_nz).expect("softbuffer: resize");
        let mut buffer = self.surface.buffer_mut().expect("softbuffer: buffer_mut");

        // `render_scene_composed` convention: caller pre-fills the
        // framebuffer with sky + the z-buffer with +INF, then it
        // z-merges every grid in.
        for px in buffer.iter_mut() {
            *px = self.clear_sky;
        }
        for z in &mut self.zbuffer[..pixel_count] {
            *z = f32::INFINITY;
        }

        let _outcome = render_scene_composed(
            &mut buffer,
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
                    &mut buffer,
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
            self.captured = Some((buffer.to_vec(), width, height));
        }

        buffer.present().expect("softbuffer: present");
    }
}
