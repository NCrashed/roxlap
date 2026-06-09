//! roxlap-render — unified CPU/GPU renderer facade.
//!
//! One [`SceneRenderer`] hides the choice between the CPU opticast
//! path (`roxlap-core` / `roxlap-scene`, presented via `softbuffer`)
//! and the GPU compute-shader path (`roxlap-gpu`, presented via its
//! own wgpu surface). Construction picks the GPU backend when asked
//! and able, and **falls back to CPU automatically** when WGPU init
//! fails — so a host never has to branch on GPU availability or carry
//! the `Scene`→GPU upload/refresh/transform glue itself.
//!
//! Hosts stay thin: build a `Scene`, advance it from input, then call
//! [`SceneRenderer::render`] each frame. The facade owns the window
//! surface, the framebuffer/z-buffer (CPU) or the resident scene +
//! dirty-chunk tracking (GPU), and presentation.
//!
//! This is the RF.0 skeleton: backend selection + fallback + a
//! clear-to-sky frame. RF.1/RF.2 fill in the real CPU/GPU scene
//! render; RF.3 adds sprites; RF.4 adds framebuffer capture.

#![forbid(unsafe_code)]

mod cpu;
mod gpu;

use std::sync::Arc;

use winit::window::Window;

use roxlap_core::opticast::OpticastSettings;
use roxlap_core::sky::Sky;
use roxlap_core::sprite::SpriteLighting;
use roxlap_core::Camera;
use roxlap_scene::Scene;

pub use roxlap_formats::sprite::Sprite;
pub use roxlap_gpu::{GpuInitError, GpuRendererSettings};

use crate::cpu::CpuBackend;
use crate::gpu::GpuBackend;

/// One placed sprite instance: which [`SpriteSet::models`] entry and
/// where in the world.
pub struct SpriteInstanceDesc {
    pub model: usize,
    pub pos: [f32; 3],
}

/// Backend-agnostic sprite description. The facade builds the CPU
/// per-instance draw list and the GPU instanced registry from the
/// same data, so both backends show identical sprites. The host owns
/// content (which models, where, recolouring) — building a recoloured
/// variant is just a second [`Sprite`] model with edited `kv6.voxels`.
pub struct SpriteSet {
    /// Distinct voxel models (KV6 + base orientation). Instances index
    /// into this; their position overrides the model's.
    pub models: Vec<Sprite>,
    pub instances: Vec<SpriteInstanceDesc>,
    /// Model the [`SceneRenderer::carve_active_sprite`] hotkey edits
    /// (GPU only, mirroring the demo's `G`-carve). `None` disables it.
    pub carve_model: Option<usize>,
}

/// Per-frame inputs both backends consume. The host builds the
/// [`OpticastSettings`] (it owns scan distance etc.); the facade does
/// everything else (pool config, sky fill, render, present).
pub struct FrameParams<'a> {
    /// CPU opticast settings (scan distance, mip ladder, framebuffer
    /// geometry). Ignored by the GPU backend.
    pub settings: &'a OpticastSettings,
    /// Packed engine sky colour: the CPU sky-miss fill + skycast, and
    /// the clear colour if no scene renders.
    pub sky_color: u32,
    /// Optional sky panorama for the CPU rasterizer's sky sampling.
    pub sky: Option<&'a Sky>,
    /// CPU fog: packed colour + max scan distance (voxels). `0` scan
    /// distance disables CPU fog.
    pub fog_color: u32,
    pub fog_max_scan_dist: i32,
    /// CPU: treat z=255 as air (avoids the S1.X bedrock path for
    /// out-of-bounds cameras).
    pub treat_z_max_as_air: bool,
    /// GPU scene-grid LOD scan distance (world units); see GPU.11.1.
    /// Ignored by the CPU backend.
    pub gpu_mip_scan_dist: f32,
    /// GPU outer-DDA step budget (chunks). Ignored by the CPU backend.
    pub gpu_max_outer_steps: u32,
    /// GPU vertical field of view (radians). Ignored by the CPU
    /// backend (it derives projection from [`OpticastSettings`]).
    pub gpu_fov_y_rad: f32,
    /// CPU sprite shading (built by the host from its engine). Required
    /// for the CPU backend to draw sprites; ignored by the GPU backend
    /// (its sprite pass shades from the uploaded model colours). `None`
    /// skips CPU sprite drawing.
    pub sprite_lighting: Option<&'a SpriteLighting<'a>>,
}

/// Result of [`SceneRenderer::pick`] — a resolved screen→world voxel
/// hit. `world` is the surface point (`cam.pos + t · normalize(ray)`);
/// `grid` + `voxel` are the owning grid and its **grid-local** voxel
/// (transform-correct for rotated / translated grids).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct PickHit {
    pub world: [f32; 3],
    pub grid: roxlap_scene::GridId,
    pub voxel: glam::IVec3,
}

/// Which renderer a [`SceneRenderer`] resolved to at construction.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Backend {
    /// `roxlap-core` opticast, presented via `softbuffer`.
    Cpu,
    /// `roxlap-gpu` compute marcher, presented via wgpu.
    Gpu,
}

/// Construction-time options for [`SceneRenderer::new`].
pub struct RenderOptions {
    /// Try the GPU backend first. When `false`, or when GPU init
    /// fails, the renderer uses the CPU backend.
    pub want_gpu: bool,
    /// Settings forwarded to [`roxlap_gpu::GpuRenderer`] when the GPU
    /// backend is selected.
    pub gpu: GpuRendererSettings,
    /// Packed `0x00RRGGBB` (alpha ignored) the empty/clear frame fills
    /// with until a scene render lands. Also the CPU sky-miss colour
    /// default if a frame supplies none.
    pub clear_sky: u32,
    /// CPU [`ScratchPool`](roxlap_core::rasterizer::ScratchPool) `lastx`
    /// sizing — the largest combined grid `vsid` the CPU rasterizer
    /// will see. Pre-sizing keeps later frames allocation-free.
    pub cpu_max_grid_vsid: u32,
    /// CPU strip-parallel render thread count (capped to the rayon
    /// pool). One [`ScratchPool`](roxlap_core::rasterizer::ScratchPool)
    /// slot per thread.
    pub cpu_render_threads: usize,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            want_gpu: false,
            gpu: GpuRendererSettings::default(),
            clear_sky: 0x0099_b3d9,
            // 32 chunks × CHUNK_SIZE_XY — the scene-demo's widest
            // combined ground grid.
            cpu_max_grid_vsid: 32 * roxlap_scene::CHUNK_SIZE_XY,
            cpu_render_threads: 4,
        }
    }
}

/// Renderer-internal backend; never exposes wgpu or softbuffer types.
/// The GPU variant owns the whole wgpu device/queue/pipelines, so
/// it's boxed to keep the enum small.
enum BackendImpl {
    Cpu(CpuBackend),
    Gpu(Box<GpuBackend>),
}

/// Unified renderer over the CPU and GPU paths. See the crate docs.
pub struct SceneRenderer {
    inner: BackendImpl,
}

impl SceneRenderer {
    /// Build a renderer for `window`. Selects the GPU backend when
    /// `opts.want_gpu` and WGPU initialises; otherwise the CPU
    /// backend. **Never fails** — a missing/incompatible GPU silently
    /// yields the CPU path (the message is logged to stderr).
    #[must_use]
    pub fn new(window: Arc<Window>, opts: &RenderOptions) -> Self {
        if opts.want_gpu {
            match GpuBackend::new(window.clone(), opts) {
                Ok(g) => {
                    return Self {
                        inner: BackendImpl::Gpu(Box::new(g)),
                    };
                }
                Err(e) => {
                    eprintln!(
                        "roxlap-render: GPU init failed ({e}); falling back to the CPU renderer",
                    );
                }
            }
        }
        Self {
            inner: BackendImpl::Cpu(CpuBackend::new(window, opts)),
        }
    }

    /// Which backend was selected.
    #[must_use]
    pub fn backend(&self) -> Backend {
        match self.inner {
            BackendImpl::Cpu(_) => Backend::Cpu,
            BackendImpl::Gpu(_) => Backend::Gpu,
        }
    }

    /// The GPU adapter description when on the GPU backend, else
    /// `None`.
    #[must_use]
    pub fn adapter_info(&self) -> Option<&str> {
        match &self.inner {
            BackendImpl::Gpu(g) => Some(g.adapter_info()),
            BackendImpl::Cpu(_) => None,
        }
    }

    /// Upload an equirectangular sky panorama (RGBA8, `w×h`) for the
    /// GPU marcher's sky sampling. No-op on the CPU backend, which
    /// samples the [`Sky`] passed in each [`FrameParams`] instead.
    pub fn set_sky_panorama(&mut self, rgba: &[u8], w: u32, h: u32) {
        if let BackendImpl::Gpu(g) = &mut self.inner {
            g.set_sky_panorama(rgba, w, h);
        }
    }

    /// Follow a window resize. CPU resizes its framebuffer lazily, so
    /// this only matters to the GPU swapchain — but it's safe to call
    /// for both.
    pub fn resize(&mut self, width: u32, height: u32) {
        match &mut self.inner {
            BackendImpl::Cpu(c) => c.resize(width, height),
            BackendImpl::Gpu(g) => g.resize(width, height),
        }
    }

    /// Render `scene` from `view` with `frame` params and present to
    /// the window. The CPU backend fills sky, runs the opticast
    /// compositor, and presents via softbuffer; the GPU backend
    /// uploads/refreshes the scene and runs the compute marcher, then
    /// the sprite pass.
    pub fn render(&mut self, scene: &mut Scene, camera: &Camera, frame: &FrameParams) {
        match &mut self.inner {
            BackendImpl::Cpu(c) => c.render(scene, camera, frame),
            BackendImpl::Gpu(g) => g.render(scene, camera, frame),
        }
    }

    /// Register sprite models + instances. The CPU backend builds a
    /// per-instance draw list; the GPU backend builds an instanced
    /// model registry. Call once at setup (or again to replace).
    pub fn set_sprites(&mut self, set: &SpriteSet) {
        match &mut self.inner {
            BackendImpl::Cpu(c) => c.set_sprites(set),
            BackendImpl::Gpu(g) => g.set_sprites(set),
        }
    }

    /// Carve the next z-layer off the [`SpriteSet::carve_model`] and
    /// re-upload (the demo's `G` hotkey + GPU.12 copy-on-modify). GPU
    /// only; a no-op on the CPU backend. Returns the voxels removed.
    pub fn carve_active_sprite(&mut self) -> u32 {
        match &mut self.inner {
            BackendImpl::Cpu(_) => 0,
            BackendImpl::Gpu(g) => g.carve_active_sprite(),
        }
    }

    /// Request that the next [`render`](Self::render) capture its
    /// framebuffer for [`take_capture`](Self::take_capture). CPU only
    /// (the GPU swapchain isn't read back) — a no-op on GPU.
    pub fn request_capture(&mut self) {
        if let BackendImpl::Cpu(c) = &mut self.inner {
            c.request_capture();
        }
    }

    /// Take the most recently captured frame as packed `0x00RRGGBB`
    /// pixels + dimensions, or `None` if no capture is ready / GPU.
    pub fn take_capture(&mut self) -> Option<(Vec<u32>, u32, u32)> {
        match &mut self.inner {
            BackendImpl::Cpu(c) => c.take_capture(),
            BackendImpl::Gpu(_) => None,
        }
    }

    /// Screen→world picking input: the world-space hit distance `t` at
    /// window pixel `(x, y)` from the **last rendered frame**, or `None`
    /// for out-of-bounds pixels and sky / no-hit. The host reconstructs
    /// the world hit point as `cam.pos + t * normalize(ray_dir)`, where
    /// `ray_dir` is the same per-pixel ray the frame was rendered with
    /// (see the backend's projection).
    ///
    /// `t` is the distance to the nearest **scene-grid** surface
    /// (terrain + grids); sprites do not occlude it (the sprite pass
    /// reads depth read-only), so a cursor sprite under the pointer is
    /// transparent to the pick.
    ///
    /// Cost: the CPU backend reads its in-memory z-buffer (free); the
    /// GPU backend stages the depth buffer and blocks on a device poll
    /// (cheap at click time — do not call every frame). The GPU path
    /// only has depth when the last frame drew sprites (`write_depth`).
    #[must_use]
    pub fn pick_depth(&self, x: u32, y: u32) -> Option<f32> {
        match &self.inner {
            BackendImpl::Cpu(c) => c.pick_depth(x, y),
            BackendImpl::Gpu(g) => g.pick_depth(x, y),
        }
    }

    /// World-space view-ray direction (un-normalised) for window pixel
    /// `(x, y)`, under the projection the **last frame** rendered with.
    /// The backends differ (CPU `setcamera` vs GPU vertical-FOV
    /// pinhole), so this hides which one is active. `None` before the
    /// first frame. Intersect it with a plane for tile picking, or feed
    /// it to [`Self::pick`] for a voxel.
    #[must_use]
    pub fn pixel_ray(&self, camera: &Camera, x: f64, y: f64) -> Option<[f64; 3]> {
        match &self.inner {
            BackendImpl::Cpu(c) => c.pixel_ray(camera, x, y),
            BackendImpl::Gpu(g) => g.pixel_ray(camera, x, y),
        }
    }

    /// One-call screen→world voxel pick: unproject pixel `(x, y)` with
    /// the active backend's projection, read the last frame's depth
    /// there, reconstruct the world hit, and resolve it to the owning
    /// grid + grid-local voxel via [`Scene::resolve_voxel`]. `None` on
    /// sky / no-hit, or when no grid claims the surface.
    ///
    /// `scene` and `camera` must be the ones the last frame rendered;
    /// the projection (size + FOV / `hx,hy,hz`) is taken from that
    /// frame. Cheap on CPU (in-memory z-buffer); on GPU it stages the
    /// depth buffer (a click-time device poll — not per frame).
    #[must_use]
    pub fn pick(&self, scene: &Scene, camera: &Camera, x: u32, y: u32) -> Option<PickHit> {
        let dir = self.pixel_ray(camera, f64::from(x), f64::from(y))?;
        let t = f64::from(self.pick_depth(x, y)?);
        let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        if len < 1e-9 {
            return None;
        }
        let s = t / len; // world = cam.pos + t · (dir / |dir|)
        let world = glam::DVec3::new(
            camera.pos[0] + dir[0] * s,
            camera.pos[1] + dir[1] * s,
            camera.pos[2] + dir[2] * s,
        );
        let (grid, voxel) = scene.resolve_voxel(world, glam::DVec3::from_array(dir))?;
        #[allow(clippy::cast_possible_truncation)]
        let world_f32 = [world.x as f32, world.y as f32, world.z as f32];
        Some(PickHit {
            world: world_f32,
            grid,
            voxel,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_default_is_cpu_intent() {
        let o = RenderOptions::default();
        assert!(!o.want_gpu);
        assert_eq!(o.clear_sky & 0xFF00_0000, 0, "clear_sky is 0x00RRGGBB");
    }
}
