//! WGPU-backed compute-shader renderer scaffold for the roxlap
//! voxel engine. GPU.1 in `PORTING-GPU.md`.
//!
//! GPU.1's job: stand up the device + surface + swapchain on a
//! winit window, present a clear-to-colour frame each render call,
//! and give the host a one-call opt-in. No voxel marching yet — the
//! [`examples/probe.rs`](../examples/probe.rs) standalone holds
//! the empirical FPS baseline from GPU.0.
//!
//! Later sub-substages flesh `GpuRenderer::render` out: GPU.2
//! uploads voxel data, GPU.3 dispatches the inner-DDA compute
//! shader, GPU.4 layers in chunk skipping, GPU.5 plugs the renderer
//! into `roxlap-scene::Scene`, …
//!
//! ## Host integration shape (GPU.1)
//!
//! ```no_run
//! use std::sync::Arc;
//! use roxlap_gpu::{GpuRenderer, GpuRendererSettings};
//! # use winit::window::Window;
//! # fn pick(w: Arc<Window>) -> Option<GpuRenderer> {
//! match GpuRenderer::new_blocking(w, GpuRendererSettings::default()) {
//!     Ok(r) => Some(r),
//!     Err(e) => {
//!         eprintln!("GPU init failed: {e}; falling back to CPU");
//!         None
//!     }
//! }
//! # }
//! ```

#![allow(clippy::must_use_candidate, clippy::too_many_lines)]

pub mod camera;
pub mod decompress;
pub mod grid;
pub mod headless;
pub mod resident;
pub mod scene;
pub mod sprite_model;

pub use camera::Camera;
pub use decompress::{decompress_chunk, ChunkUpload, BEDROCK_RGB, CHUNK_Z};
pub use grid::{bounding_box_of, GpuGridResident, GridUpload};
pub use headless::HeadlessGpu;
pub use resident::GpuChunkResident;
pub use scene::{
    GpuSceneResident, GridRuntimeTransform, GridStaticMeta, RefreshOutcome, SceneUpload,
    MAX_SCENE_GRIDS,
};
pub use sprite_model::{
    build_sprite_model, SpriteInstance, SpriteInstanceTransform, SpriteModel, SpriteModelRegistry,
    SpriteRegistryResident,
};

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use winit::window::Window;

/// Caller-controllable knobs for [`GpuRenderer::new`]. Defaults
/// target "highest-performance GPU, prefer Mailbox/Immediate over
/// vsync" — i.e. the same configuration the GPU.0 probe used to
/// measure the FPS ceiling.
#[derive(Debug, Clone, Copy)]
pub struct GpuRendererSettings {
    pub power_preference: PowerPreference,
    /// Initial clear colour cycled by GPU.1's empty render path.
    /// The voxel-rendering substages overwrite this entirely.
    pub clear_colour: [f64; 3],
    /// Prefer mailbox/immediate when offered; falls back to FIFO if
    /// the surface only supports it (Wayland under Mesa often does).
    pub uncapped_present: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum PowerPreference {
    Low,
    High,
}

impl Default for GpuRendererSettings {
    fn default() -> Self {
        Self {
            power_preference: PowerPreference::High,
            clear_colour: [0.06, 0.08, 0.12],
            uncapped_present: true,
        }
    }
}

/// Errors `GpuRenderer::new` surfaces to the host. The host's
/// expected flow is "try this, fall back to the CPU path on Err".
#[derive(Debug)]
pub enum GpuInitError {
    CreateSurface(wgpu::CreateSurfaceError),
    NoAdapter,
    RequestDevice(wgpu::RequestDeviceError),
}

impl std::fmt::Display for GpuInitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CreateSurface(e) => write!(f, "create_surface failed: {e}"),
            Self::NoAdapter => write!(
                f,
                "no compatible adapter — does this system have a Vulkan/Metal/DX12 driver?"
            ),
            Self::RequestDevice(e) => write!(f, "request_device failed: {e}"),
        }
    }
}

impl std::error::Error for GpuInitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::CreateSurface(e) => Some(e),
            Self::RequestDevice(e) => Some(e),
            Self::NoAdapter => None,
        }
    }
}

impl From<wgpu::CreateSurfaceError> for GpuInitError {
    fn from(value: wgpu::CreateSurfaceError) -> Self {
        Self::CreateSurface(value)
    }
}

impl From<wgpu::RequestDeviceError> for GpuInitError {
    fn from(value: wgpu::RequestDeviceError) -> Self {
        Self::RequestDevice(value)
    }
}

/// WGPU-backed renderer. Owns the device, queue, and surface
/// bound to the host's winit window. [`Self::render`] is the GPU.1
/// clear-to-colour path; [`Self::render_chunk`] is GPU.3's
/// single-chunk DDA marcher.
pub struct GpuRenderer {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_info: String,
    clear_colour: [f64; 3],
    frame_count: u32,
    /// Lazy-built on first [`Self::render_chunk`] call; rebuilt when
    /// the swapchain resizes (storage texture must match).
    chunk_dda: Option<ChunkDdaResources>,
    /// Lazy-built on first [`Self::render_grid`] call; same resize
    /// trigger as `chunk_dda`. The two paths share the same blit
    /// pipeline structure but bind different storage layouts.
    grid_dda: Option<GridDdaResources>,
    /// Lazy-built on first [`Self::render_scene`] call. Holds the
    /// multi-grid pipeline + per-grid camera uniforms.
    scene_dda: Option<SceneDdaResources>,
    /// GPU.8 — panoramic sky texture + sampler. Created at
    /// `new` as a 1×1 mid-grey default; [`Self::set_sky_panorama`]
    /// replaces it. The scene-DDA bind group references this each
    /// frame.
    sky_texture: wgpu::Texture,
    sky_view: wgpu::TextureView,
    sky_sampler: wgpu::Sampler,
    /// GPU.8 fog state. `color` is BGRA-style premultiplied (each
    /// channel in [0, 1]); `near` is the world-t distance at which
    /// fog starts kicking in; `far` is the distance at which it's
    /// fully opaque. The shader does
    /// `mix(hit, fog, smoothstep(near, far, t))`.
    fog_color: [f32; 3],
    fog_near: f32,
    fog_far: f32,
    /// GPU.10 — sprites rendered as DDA-marched voxel models (the
    /// precise path; the GPU.9 compute splatter it replaced was
    /// retired in 10.5). Holds the concatenated model registry + the
    /// per-frame instance array; set via [`Self::set_sprite_instances`].
    sprite_registry: Option<sprite_model::SpriteRegistryResident>,
    /// Lazy-built pipeline + uniform for the model-DDA pass.
    sprite_model_dda: Option<SpriteModelDdaResources>,
    /// GPU.10.4 — LOD aggressiveness: step a sprite to the next mip
    /// once a mip-0 voxel projects below this many screen pixels.
    /// Defaults to 4.0 (the empirical sweet spot); the host can tune
    /// via [`Self::set_sprite_lod_px`].
    sprite_lod_px: f32,
    /// GPU.11.1 — scene-grid LOD scan distance (world units). A chunk
    /// entered at world-t `t` is marched at the mip level
    /// `floor(log2(max(t, msd) / msd))`, clamped to the grid's mip
    /// ladder. `0` disables LOD (always mip-0). Tunable via
    /// [`Self::set_scene_mip_scan_dist`] — the axis-aligned-mip-beams
    /// mitigation (GPU.11.2) pushes it outward if banding appears.
    scene_mip_scan_dist: f32,
}

/// Per-renderer chunk-DDA pipeline state. The compute shader writes
/// into the storage texture; a fullscreen-triangle render pass
/// nearest-neighbour blits it to the swapchain.
struct ChunkDdaResources {
    storage_size: (u32, u32),
    storage_view: wgpu::TextureView,
    uniform_buf: wgpu::Buffer,
    bgl_dda: wgpu::BindGroupLayout,
    pipeline_dda: wgpu::ComputePipeline,
    blit_bg: wgpu::BindGroup,
    pipeline_blit: wgpu::RenderPipeline,
    // wgpu BindGroups internally Arc their resources, but we keep
    // the handle so the sampler shows up in profiler dumps.
    _sampler: wgpu::Sampler,
}

struct GridDdaResources {
    storage_size: (u32, u32),
    storage_view: wgpu::TextureView,
    uniform_buf: wgpu::Buffer,
    bgl_dda: wgpu::BindGroupLayout,
    pipeline_dda: wgpu::ComputePipeline,
    blit_bg: wgpu::BindGroup,
    pipeline_blit: wgpu::RenderPipeline,
    _sampler: wgpu::Sampler,
}

struct SceneDdaResources {
    storage_size: (u32, u32),
    storage_view: wgpu::TextureView,
    uniform_buf: wgpu::Buffer,
    bgl_dda: wgpu::BindGroupLayout,
    pipeline_dda: wgpu::ComputePipeline,
    blit_bg: wgpu::BindGroup,
    pipeline_blit: wgpu::RenderPipeline,
    _sampler: wgpu::Sampler,
    /// GPU.9 — per-pixel world-t depth (f32 bits as u32), sized
    /// `width * height * 4`. The scene pass writes it when sprites
    /// are present; the sprite model-DDA pass reads + composites
    /// against it.
    depth_buffer: wgpu::Buffer,
    /// Picking — a `COPY_DST | MAP_READ` staging copy of `depth_buffer`
    /// so the host can read back the per-pixel world-t after a frame
    /// (e.g. click → which voxel). Same size as `depth_buffer`.
    depth_readback: wgpu::Buffer,
}

/// GPU.10.0 — single-sprite model-DDA pipeline: one thread per pixel
/// marches the model voxel volume and composites against the scene
/// depth buffer.
struct SpriteModelDdaResources {
    bgl: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    uniform_buf: wgpu::Buffer,
}

/// Per-frame uniform for the model-DDA pass. Mirrors `Uniform` in
/// `sprite_model_dda.wgsl` (std140). Per-model + per-instance data
/// now live in storage buffers; this holds only the camera, fog, and
/// instance count.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SpriteModelUniform {
    cam_pos: [f32; 3],
    _p0: f32,
    cam_right: [f32; 3],
    _p1: f32,
    cam_down: [f32; 3],
    _p2: f32,
    cam_forward: [f32; 3],
    _p3: f32,
    fog_color: [f32; 4],
    screen_size: [u32; 2],
    instance_count: u32,
    fog_far: f32,
    fov_y_rad: f32,
    tiles_x: u32,
    tile_size: u32,
    _p6: f32,
}

const SCENE_MAX_GRIDS: usize = MAX_SCENE_GRIDS as usize;

/// GPU.10.3 — sprite screen-tile edge in pixels for instance binning.
const SPRITE_TILE_SIZE: u32 = 16;

// The scene_dda bind group + layout wire occupancy pages 1..=3 at
// bindings 12..=14 explicitly; keep that in lockstep with the page
// count. Bump the bindings (here, in the WGSL, and in the bind
// group) if MAX_OCC_PAGES changes.
const _: () = assert!(scene::MAX_OCC_PAGES == 4);

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SceneDdaPerGridCamera {
    pos: [f32; 3],
    _pad0: f32,
    right: [f32; 3],
    _pad1: f32,
    down: [f32; 3],
    _pad2: f32,
    forward: [f32; 3],
    _pad3: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SceneDdaUniform {
    fov_y_rad: f32,
    grid_count: u32,
    max_outer_steps: u32,
    _pad0: u32,
    screen_size: [u32; 2],
    _pad1: [u32; 2],
    cameras: [SceneDdaPerGridCamera; SCENE_MAX_GRIDS],
    /// GPU.8 — `[r, g, b, fog_near]`. The `near` distance is packed
    /// into the colour's alpha channel to keep std140 alignment
    /// tidy (a bare `f32` after the `vec4` would force extra pads).
    fog_color: [f32; 4],
    fog_far: f32,
    /// GPU.9 — `1` when the sprite pass is active (scene pass then
    /// records `best_t` into the depth buffer), `0` otherwise.
    write_depth: u32,
    /// Occupancy paging: words per storage page (see
    /// `scene::split_occupancy_pages`). Only consulted by the shader
    /// when `occ_num_pages > 1`.
    occ_page_words: u32,
    /// Number of real occupancy pages (1 on multi-GiB GPUs → the
    /// shader takes a branch-free single-page read).
    occ_num_pages: u32,
    /// GPU.11.1 — scene-grid LOD scan distance (world units). A chunk
    /// entered at world-t `t` marches at mip
    /// `floor(log2(max(t, msd) / msd))`, clamped to the grid's mip
    /// count. `0` disables LOD (always mip-0).
    mip_scan_dist: f32,
    _pad2: u32,
    _pad3: u32,
    _pad4: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GridDdaUniform {
    camera_pos: [f32; 3],
    _pad0: f32,
    camera_right: [f32; 3],
    _pad1: f32,
    camera_down: [f32; 3],
    _pad2: f32,
    camera_forward: [f32; 3],
    fov_y_rad: f32,
    screen_size: [u32; 2],
    vsid: u32,
    max_outer_steps: u32,
    chunks_dims: [u32; 3],
    _pad3: u32,
    origin_chunk: [i32; 3],
    _pad4: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ChunkDdaUniform {
    camera_pos: [f32; 3],
    _pad0: f32,
    camera_right: [f32; 3],
    _pad1: f32,
    camera_down: [f32; 3],
    _pad2: f32,
    camera_forward: [f32; 3],
    fov_y_rad: f32,
    screen_size: [u32; 2],
    vsid: u32,
    max_scan_dist: u32,
}

impl GpuRenderer {
    /// Stand up the device + surface + swapchain on `window`. Async
    /// because `wgpu::Adapter`/`Device` requests are.
    ///
    /// # Errors
    /// Returns [`GpuInitError`] if surface creation, adapter
    /// selection, or device request fails. Hosts treat any error as
    /// "fall back to the CPU path".
    pub async fn new(
        window: Arc<Window>,
        settings: GpuRendererSettings,
    ) -> Result<Self, GpuInitError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let surface = instance.create_surface(window.clone())?;
        let power_preference = match settings.power_preference {
            PowerPreference::Low => wgpu::PowerPreference::LowPower,
            PowerPreference::High => wgpu::PowerPreference::HighPerformance,
        };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or(GpuInitError::NoAdapter)?;

        let info = adapter.get_info();
        let adapter_info = format!(
            "{name} ({backend:?}, {device_type:?})",
            name = info.name,
            backend = info.backend,
            device_type = info.device_type,
        );

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("roxlap-gpu device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: pick_required_limits(&adapter.limits()),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await?;

        let caps = surface.get_capabilities(&adapter);
        // Pick a NON-sRGB swapchain format. Voxlap colours are
        // already sRGB-encoded (the slab bytes are display-ready,
        // matching what the CPU softbuffer path writes straight to
        // the framebuffer with no conversion). An sRGB swapchain
        // would re-apply the gamma curve on top, producing a
        // washed-out / pastel look that diverges from the CPU
        // renderer. Falls back to `caps.formats[0]` only if every
        // offered format is sRGB.
        let surface_format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let present_mode = if settings.uncapped_present {
            pick_present_mode(&caps.present_modes)
        } else {
            wgpu::PresentMode::Fifo
        };
        // GPU.11.2 — surface the present mode: `Fifo` is vsync-capped
        // (FPS pinned to refresh rate → compute optimisations like the
        // mip LOD won't show up in the FPS counter). Mailbox/Immediate
        // are uncapped. Wayland under Mesa frequently offers only Fifo.
        eprintln!(
            "roxlap-gpu: present mode = {present_mode:?} (available: {:?})",
            caps.present_modes,
        );
        let physical = window.inner_size();
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: physical.width.max(1),
            height: physical.height.max(1),
            present_mode,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        // GPU.8 default sky: a 1×1 mid-grey texture. Hosts replace
        // it via `set_sky_panorama` with a real equirectangular
        // panorama; the default stops the shader sampling
        // uninitialised memory before that happens.
        let default_sky_pixel = [0x80u8, 0x80, 0x80, 0xff];
        let (sky_texture, sky_view) = create_sky_texture(&device, 1, 1, &default_sky_pixel);
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &sky_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &default_sky_pixel,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let sky_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("roxlap-gpu sky_sampler"),
            // Voxlap-convention panorama: u = elevation [0, 1]
            // (Repeat is a no-op since values don't go outside),
            // v = azimuth (wraps 360° — Repeat is required).
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Ok(Self {
            window,
            surface,
            surface_config,
            device,
            queue,
            adapter_info,
            clear_colour: settings.clear_colour,
            frame_count: 0,
            chunk_dda: None,
            grid_dda: None,
            scene_dda: None,
            sky_texture,
            sky_view,
            sky_sampler,
            // Fog disabled by default — voxlap's CPU rasterizer
            // also runs without fog in the scene-demo, so matching
            // it means no GPU fog out of the box. Hosts can opt in
            // via `set_fog` (e.g. for atmospheric far-LOD masking).
            fog_color: [0.66, 0.74, 0.88],
            fog_near: 0.0,
            fog_far: 1.0e30,
            sprite_registry: None,
            sprite_model_dda: None,
            // GPU.10.4 — default LOD threshold: step to a coarser mip
            // once a voxel projects below 4 px. Empirically the best
            // quality/cost tradeoff; the host can override.
            sprite_lod_px: 4.0,
            // GPU.11.1 — matches the CPU demo's mip_scan_dist=64.
            scene_mip_scan_dist: 64.0,
        })
    }

    /// Synchronous wrapper for hosts that don't have an async
    /// runtime. Internally `pollster::block_on`s [`Self::new`].
    ///
    /// # Errors
    /// See [`Self::new`].
    pub fn new_blocking(
        window: Arc<Window>,
        settings: GpuRendererSettings,
    ) -> Result<Self, GpuInitError> {
        pollster::block_on(Self::new(window, settings))
    }

    /// Human-readable adapter description — name + backend +
    /// device type. The demo host prints this in the title bar.
    pub fn adapter_info(&self) -> &str {
        &self.adapter_info
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    /// Borrow the underlying wgpu device — hosts use this to build
    /// chunk uploads (`GpuChunkResident::upload(gpu.device(), …)`).
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Borrow the wgpu queue — hosts use this for read-back paths
    /// (`GpuChunkResident::read_voxel_blocking(gpu.device(), gpu.queue(), …)`).
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// GPU.8 — upload an equirectangular panorama as the scene's
    /// sky texture. `rgba` is row-major, `width × height` pixels,
    /// 4 bytes per pixel (R, G, B, A). The shader samples it with
    /// `u = atan2(dir.x, dir.y) / (2π) + 0.5` (azimuth) and
    /// `v = acos(-dir.z) / π` (elevation), matching standard
    /// equirectangular layout (top of image = zenith for voxlap's
    /// `+z = down` basis).
    ///
    /// # Panics
    /// If `rgba.len() != (width * height * 4) as usize`.
    pub fn set_sky_panorama(&mut self, rgba: &[u8], width: u32, height: u32) {
        assert_eq!(
            rgba.len(),
            (width as usize) * (height as usize) * 4,
            "set_sky_panorama: expected w*h*4 bytes, got {}",
            rgba.len(),
        );
        let (tex, view) = create_sky_texture(&self.device, width, height, rgba);
        // Upload pixel data via `queue.write_texture` so we don't
        // have to map the buffer manually.
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        self.sky_texture = tex;
        self.sky_view = view;
    }

    /// GPU.8 — set the fog blend. `color` is per-channel [0, 1];
    /// `near`/`far` are world-space ray distances in voxel units.
    /// Hits with `t < near` show their full colour; hits with
    /// `t > far` show `color` exclusively; in between is a
    /// smoothstep blend.
    pub fn set_fog(&mut self, color: [f32; 3], near: f32, far: f32) {
        self.fog_color = color;
        self.fog_near = near;
        self.fog_far = far.max(near + 1.0);
    }

    /// Re-configure the swapchain to a new physical size. Call from
    /// `WindowEvent::Resized`. Drops the chunk-DDA storage texture
    /// so [`Self::render_chunk`] rebuilds it at the new size.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.surface_config.width = width;
        self.surface_config.height = height;
        self.surface.configure(&self.device, &self.surface_config);
        self.chunk_dda = None;
        self.grid_dda = None;
        self.scene_dda = None;
    }

    /// GPU.1 render: single render pass clearing the swapchain to a
    /// slowly drifting colour, then presenting. Voxels arrive in
    /// GPU.3+.
    pub fn render(&mut self) {
        let surf_tex = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Lost) => {
                self.surface.configure(&self.device, &self.surface_config);
                return;
            }
            Err(e) => {
                eprintln!("roxlap-gpu surface error: {e:?}");
                return;
            }
        };
        let view = surf_tex
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Slow colour drift so the user can tell the GPU path is
        // actually presenting frames vs. e.g. a frozen window.
        // Wrap at 2π/0.005 frames (~1257) so the cast stays exact.
        let phase = f64::from(self.frame_count % 1257) * 0.005;
        let [r, g, b] = self.clear_colour;
        let drift = (phase.sin() * 0.04 + 0.04).clamp(0.0, 0.1);
        let clear = wgpu::Color {
            r: (r + drift).clamp(0.0, 1.0),
            g: (g + drift * 0.5).clamp(0.0, 1.0),
            b: (b + drift * 0.25).clamp(0.0, 1.0),
            a: 1.0,
        };

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("roxlap-gpu encoder"),
            });
        {
            let _rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("roxlap-gpu clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        surf_tex.present();
        self.frame_count = self.frame_count.wrapping_add(1);
    }

    /// GPU.3 single-chunk render. Dispatches `chunk_dda.wgsl`
    /// against `resident`'s storage buffers, then blits the
    /// low-res storage texture to the swapchain. `camera.position`
    /// is in **chunk-local** voxel units (host translates from
    /// world coords). `max_scan_dist` caps the per-pixel DDA loop —
    /// scene-demo wires `+` / `-` through this each frame.
    ///
    /// # Panics
    /// Internally `expect`s the chunk-DDA resources to be built —
    /// they are constructed at the top of this function if missing.
    /// Cannot fire in normal control flow.
    pub fn render_chunk(
        &mut self,
        resident: &GpuChunkResident,
        camera: &Camera,
        max_scan_dist: u32,
    ) {
        let surf_tex = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Lost) => {
                self.surface.configure(&self.device, &self.surface_config);
                return;
            }
            Err(e) => {
                eprintln!("roxlap-gpu surface error: {e:?}");
                return;
            }
        };
        let surf_view = surf_tex
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let surface_w = self.surface_config.width;
        let surface_h = self.surface_config.height;
        let surface_format = self.surface_config.format;

        // Lazy-build chunk-DDA resources; rebuild when the swapchain
        // grew or shrank.
        let needs_build = match &self.chunk_dda {
            Some(r) => r.storage_size != (surface_w, surface_h),
            None => true,
        };
        if needs_build {
            self.chunk_dda = Some(self.build_chunk_dda(surface_w, surface_h, surface_format));
        }
        let dda = self.chunk_dda.as_ref().expect("just built");

        // Update uniforms.
        let uniform = ChunkDdaUniform {
            camera_pos: camera.position,
            _pad0: 0.0,
            camera_right: camera.right,
            _pad1: 0.0,
            camera_down: camera.down,
            _pad2: 0.0,
            camera_forward: camera.forward,
            fov_y_rad: camera.fov_y_rad,
            screen_size: [surface_w, surface_h],
            vsid: resident.vsid,
            max_scan_dist,
        };
        self.queue
            .write_buffer(&dda.uniform_buf, 0, bytemuck::bytes_of(&uniform));

        // Per-frame DDA bind group — references the chunk's buffers
        // so we rebuild every frame (the resident can change between
        // calls).
        let dda_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("roxlap-gpu chunk_dda.bg"),
            layout: &dda.bgl_dda,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: dda.uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resident.occupancy.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resident.color_offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: resident.colors.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&dda.storage_view),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("roxlap-gpu chunk encoder"),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("roxlap-gpu chunk_dda compute"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&dda.pipeline_dda);
            cpass.set_bind_group(0, &dda_bg, &[]);
            cpass.dispatch_workgroups(surface_w.div_ceil(8), surface_h.div_ceil(8), 1);
        }
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("roxlap-gpu chunk_dda blit"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surf_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&dda.pipeline_blit);
            rpass.set_bind_group(0, &dda.blit_bg, &[]);
            rpass.draw(0..3, 0..1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        surf_tex.present();
        self.frame_count = self.frame_count.wrapping_add(1);
    }

    fn build_chunk_dda(
        &self,
        width: u32,
        height: u32,
        surface_format: wgpu::TextureFormat,
    ) -> ChunkDdaResources {
        let storage_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("roxlap-gpu chunk_dda.storage"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let storage_view = storage_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let uniform_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("roxlap-gpu chunk_dda.uniform"),
            size: std::mem::size_of::<ChunkDdaUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let dda_shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("chunk_dda.wgsl"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/chunk_dda.wgsl").into()),
            });
        let bgl_dda = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("roxlap-gpu chunk_dda.bgl"),
                entries: &[
                    bgl_uniform_entry(0),
                    bgl_storage_entry(1, true),
                    bgl_storage_entry(2, true),
                    bgl_storage_entry(3, true),
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });
        let dda_pl = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("roxlap-gpu chunk_dda.layout"),
                bind_group_layouts: &[&bgl_dda],
                push_constant_ranges: &[],
            });
        let pipeline_dda = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("roxlap-gpu chunk_dda.pipeline"),
                layout: Some(&dda_pl),
                module: &dda_shader,
                entry_point: "render_chunk",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        // Fullscreen-triangle blit upscales the storage texture into
        // the swapchain. Nearest filter keeps the retro pixel look.
        let blit_shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("blit.wgsl"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/blit.wgsl").into()),
            });
        let bgl_blit = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("roxlap-gpu chunk_dda.blit_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                ],
            });
        let blit_pl = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("roxlap-gpu chunk_dda.blit_layout"),
                bind_group_layouts: &[&bgl_blit],
                push_constant_ranges: &[],
            });
        let pipeline_blit = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("roxlap-gpu chunk_dda.blit_pipeline"),
                layout: Some(&blit_pl),
                vertex: wgpu::VertexState {
                    module: &blit_shader,
                    entry_point: "vs_main",
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &blit_shader,
                    entry_point: "fs_main",
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surface_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });
        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("roxlap-gpu chunk_dda.blit_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let blit_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("roxlap-gpu chunk_dda.blit_bg"),
            layout: &bgl_blit,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&storage_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        ChunkDdaResources {
            storage_size: (width, height),
            storage_view,
            uniform_buf,
            bgl_dda,
            pipeline_dda,
            blit_bg,
            pipeline_blit,
            _sampler: sampler,
        }
    }

    /// GPU.4 render — outer DDA over chunk indices + inner DDA into
    /// non-empty chunks. `camera.position` is in **grid-local**
    /// voxel units. `max_outer_steps` caps how many chunks the
    /// outer DDA may traverse per ray (scene-demo wires `+ / -`
    /// through this).
    ///
    /// # Panics
    /// Internally `expect`s the grid-DDA resources to be built;
    /// they are constructed at the top of this function if missing.
    pub fn render_grid(&mut self, grid: &GpuGridResident, camera: &Camera, max_outer_steps: u32) {
        let surf_tex = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Lost) => {
                self.surface.configure(&self.device, &self.surface_config);
                return;
            }
            Err(e) => {
                eprintln!("roxlap-gpu surface error: {e:?}");
                return;
            }
        };
        let surf_view = surf_tex
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let surface_w = self.surface_config.width;
        let surface_h = self.surface_config.height;
        let surface_format = self.surface_config.format;

        let needs_build = match &self.grid_dda {
            Some(r) => r.storage_size != (surface_w, surface_h),
            None => true,
        };
        if needs_build {
            self.grid_dda = Some(self.build_grid_dda(surface_w, surface_h, surface_format));
        }
        let dda = self.grid_dda.as_ref().expect("just built");

        let uniform = GridDdaUniform {
            camera_pos: camera.position,
            _pad0: 0.0,
            camera_right: camera.right,
            _pad1: 0.0,
            camera_down: camera.down,
            _pad2: 0.0,
            camera_forward: camera.forward,
            fov_y_rad: camera.fov_y_rad,
            screen_size: [surface_w, surface_h],
            vsid: grid.vsid,
            max_outer_steps,
            chunks_dims: grid.chunks_dims,
            _pad3: 0,
            origin_chunk: grid.origin_chunk,
            _pad4: 0,
        };
        self.queue
            .write_buffer(&dda.uniform_buf, 0, bytemuck::bytes_of(&uniform));

        let dda_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("roxlap-gpu grid_dda.bg"),
            layout: &dda.bgl_dda,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: dda.uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grid.occupancy.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: grid.color_offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: grid.colors.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: grid.chunk_colors_base.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: grid.chunk_occupancy.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&dda.storage_view),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("roxlap-gpu grid encoder"),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("roxlap-gpu grid_dda compute"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&dda.pipeline_dda);
            cpass.set_bind_group(0, &dda_bg, &[]);
            cpass.dispatch_workgroups(surface_w.div_ceil(8), surface_h.div_ceil(8), 1);
        }
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("roxlap-gpu grid_dda blit"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surf_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&dda.pipeline_blit);
            rpass.set_bind_group(0, &dda.blit_bg, &[]);
            rpass.draw(0..3, 0..1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        surf_tex.present();
        self.frame_count = self.frame_count.wrapping_add(1);
    }

    fn build_grid_dda(
        &self,
        width: u32,
        height: u32,
        surface_format: wgpu::TextureFormat,
    ) -> GridDdaResources {
        let storage_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("roxlap-gpu grid_dda.storage"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let storage_view = storage_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let uniform_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("roxlap-gpu grid_dda.uniform"),
            size: std::mem::size_of::<GridDdaUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let dda_shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("grid_dda.wgsl"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/grid_dda.wgsl").into()),
            });
        let bgl_dda = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("roxlap-gpu grid_dda.bgl"),
                entries: &[
                    bgl_uniform_entry(0),
                    bgl_storage_entry(1, true),
                    bgl_storage_entry(2, true),
                    bgl_storage_entry(3, true),
                    bgl_storage_entry(4, true),
                    bgl_storage_entry(5, true),
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });
        let dda_pl = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("roxlap-gpu grid_dda.layout"),
                bind_group_layouts: &[&bgl_dda],
                push_constant_ranges: &[],
            });
        let pipeline_dda = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("roxlap-gpu grid_dda.pipeline"),
                layout: Some(&dda_pl),
                module: &dda_shader,
                entry_point: "render_grid",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let blit_shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("blit.wgsl"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/blit.wgsl").into()),
            });
        let bgl_blit = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("roxlap-gpu grid_dda.blit_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                ],
            });
        let blit_pl = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("roxlap-gpu grid_dda.blit_layout"),
                bind_group_layouts: &[&bgl_blit],
                push_constant_ranges: &[],
            });
        let pipeline_blit = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("roxlap-gpu grid_dda.blit_pipeline"),
                layout: Some(&blit_pl),
                vertex: wgpu::VertexState {
                    module: &blit_shader,
                    entry_point: "vs_main",
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &blit_shader,
                    entry_point: "fs_main",
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surface_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });
        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("roxlap-gpu grid_dda.blit_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let blit_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("roxlap-gpu grid_dda.blit_bg"),
            layout: &bgl_blit,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&storage_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        GridDdaResources {
            storage_size: (width, height),
            storage_view,
            uniform_buf,
            bgl_dda,
            pipeline_dda,
            blit_bg,
            pipeline_blit,
            _sampler: sampler,
        }
    }

    /// GPU.5 render — multi-grid scene marcher. `cameras[i]` is the
    /// world camera transformed into grid `i`'s local frame
    /// (caller-supplied; see scene-demo's `redraw_gpu` for the
    /// glam-based transform). `fov_y_rad` is the shared vertical
    /// FOV; `max_outer_steps` caps per-ray chunk-DDA work for each
    /// grid.
    ///
    /// # Panics
    /// If `cameras.len() != scene.grid_count` or
    /// `scene.grid_count > MAX_SCENE_GRIDS`.
    pub fn render_scene(
        &mut self,
        scene: &GpuSceneResident,
        cameras: &[Camera],
        fov_y_rad: f32,
        max_outer_steps: u32,
    ) {
        assert_eq!(
            cameras.len(),
            scene.grid_count as usize,
            "render_scene: {} cameras supplied, scene has {} grids",
            cameras.len(),
            scene.grid_count,
        );
        assert!(
            scene.grid_count as usize <= SCENE_MAX_GRIDS,
            "render_scene: scene has {} grids, shader supports {}",
            scene.grid_count,
            SCENE_MAX_GRIDS,
        );

        let surf_tex = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Lost) => {
                self.surface.configure(&self.device, &self.surface_config);
                return;
            }
            Err(e) => {
                eprintln!("roxlap-gpu surface error: {e:?}");
                return;
            }
        };
        let surf_view = surf_tex
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let surface_w = self.surface_config.width;
        let surface_h = self.surface_config.height;
        let surface_format = self.surface_config.format;

        let needs_build = match &self.scene_dda {
            Some(r) => r.storage_size != (surface_w, surface_h),
            None => true,
        };
        if needs_build {
            self.scene_dda = Some(self.build_scene_dda(surface_w, surface_h, surface_format));
        }
        // GPU.9 — materialise the sprite pipeline the first frame
        // sprites are present (before the immutable `dda` borrow).
        // GPU.10.0 — build the model-DDA pipeline the first frame a
        // sprite registry is present.
        if self.sprite_registry.is_some() && self.sprite_model_dda.is_none() {
            self.sprite_model_dda = Some(self.build_sprite_model_dda());
        }
        // GPU.10.3 — frustum-cull + screen-tile-bin the sprite instances
        // (needs &mut self for buffer growth, so before the immutable
        // scene_dda borrow). Captures (visible_count, tiles_x); None when
        // nothing is in view.
        let sprite_pass: Option<(u32, u32)> = if let Some(reg) = self.sprite_registry.as_mut() {
            if !cameras.is_empty() && reg.instance_capacity > 0 {
                let cam = &cameras[0];
                #[allow(clippy::cast_precision_loss)]
                let aspect = surface_w as f32 / surface_h as f32;
                let half_h = (fov_y_rad * 0.5).tan();
                let frustum = sprite_model::ViewFrustum {
                    pos: cam.position,
                    right: cam.right,
                    down: cam.down,
                    forward: cam.forward,
                    half_w: half_h * aspect,
                    half_h,
                    far: 1.0e9,
                };
                let (visible, tiles_x, _tiles_y) = reg.cull_bin_upload(
                    &self.device,
                    &self.queue,
                    &frustum,
                    surface_w,
                    surface_h,
                    SPRITE_TILE_SIZE,
                    self.sprite_lod_px,
                );
                (visible > 0).then_some((visible, tiles_x))
            } else {
                None
            }
        } else {
            None
        };
        let dda = self.scene_dda.as_ref().expect("just built");

        // Pack per-grid cameras.
        let mut cam_array = [SceneDdaPerGridCamera::zeroed(); SCENE_MAX_GRIDS];
        for (i, cam) in cameras.iter().enumerate() {
            cam_array[i] = SceneDdaPerGridCamera {
                pos: cam.position,
                _pad0: 0.0,
                right: cam.right,
                _pad1: 0.0,
                down: cam.down,
                _pad2: 0.0,
                forward: cam.forward,
                _pad3: 0.0,
            };
        }
        let uniform = SceneDdaUniform {
            fov_y_rad,
            grid_count: scene.grid_count,
            max_outer_steps,
            _pad0: 0,
            screen_size: [surface_w, surface_h],
            _pad1: [0; 2],
            cameras: cam_array,
            fog_color: [
                self.fog_color[0],
                self.fog_color[1],
                self.fog_color[2],
                self.fog_near,
            ],
            fog_far: self.fog_far,
            write_depth: u32::from(self.sprite_registry.is_some()),
            occ_page_words: scene.occupancy_page_words,
            occ_num_pages: scene.occupancy_num_pages,
            mip_scan_dist: self.scene_mip_scan_dist,
            _pad2: 0,
            _pad3: 0,
            _pad4: 0,
        };
        self.queue
            .write_buffer(&dda.uniform_buf, 0, bytemuck::bytes_of(&uniform));

        let dda_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("roxlap-gpu scene_dda.bg"),
            layout: &dda.bgl_dda,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: dda.uniform_buf.as_entire_binding(),
                },
                // Occupancy page 0 at binding 1; pages 1..MAX_OCC_PAGES
                // at bindings 12.. (see GPU.X occupancy paging).
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: scene.occupancy_pages[0].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: scene.all_color_offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: scene.all_colors.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: scene.all_chunk_colors_base.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: scene.all_chunk_occupancy.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: scene.grid_static_meta.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: scene.all_slot_chunk_idx.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(&dda.storage_view),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::TextureView(&self.sky_view),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::Sampler(&self.sky_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: dda.depth_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: scene.occupancy_pages[1].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: scene.occupancy_pages[2].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: scene.occupancy_pages[3].as_entire_binding(),
                },
            ],
        });

        // GPU.9 — when sprites are present, build both splatter bind
        // groups up front (the splat pass writes the key buffer; the
        // resolve pass reads keys + scene depth and writes colour).
        // GPU.10.3 — model-DDA bind group + per-frame uniform, using the
        // cull/bin results captured above. Per-model + per-instance data
        // + the tile lists live in the registry buffers.
        let sprite_model_bg = match (&self.sprite_model_dda, &self.sprite_registry, sprite_pass) {
            (Some(smd), Some(reg), Some((visible, tiles_x))) => {
                let cam = &cameras[0];
                let uni = SpriteModelUniform {
                    cam_pos: cam.position,
                    _p0: 0.0,
                    cam_right: cam.right,
                    _p1: 0.0,
                    cam_down: cam.down,
                    _p2: 0.0,
                    cam_forward: cam.forward,
                    _p3: 0.0,
                    fog_color: [
                        self.fog_color[0],
                        self.fog_color[1],
                        self.fog_color[2],
                        self.fog_near,
                    ],
                    screen_size: [surface_w, surface_h],
                    instance_count: visible,
                    fog_far: self.fog_far,
                    fov_y_rad,
                    tiles_x,
                    tile_size: SPRITE_TILE_SIZE,
                    _p6: 0.0,
                };
                self.queue
                    .write_buffer(&smd.uniform_buf, 0, bytemuck::bytes_of(&uni));
                Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("roxlap-gpu sprite_model_dda.bg"),
                    layout: &smd.bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: smd.uniform_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: reg.occupancy.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: reg.colors.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: reg.color_offsets.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: reg.model_meta.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: reg.instances.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: dda.depth_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: wgpu::BindingResource::TextureView(&dda.storage_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 8,
                            resource: reg.tile_ranges.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 9,
                            resource: reg.tile_instances.as_entire_binding(),
                        },
                    ],
                }))
            }
            _ => None,
        };

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("roxlap-gpu scene encoder"),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("roxlap-gpu scene_dda compute"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&dda.pipeline_dda);
            cpass.set_bind_group(0, &dda_bg, &[]);
            cpass.dispatch_workgroups(surface_w.div_ceil(8), surface_h.div_ceil(8), 1);
        }
        // GPU.10 — sprite model-DDA pass: one thread per pixel marches
        // the tile's instances + composites against scene depth, after
        // the scene pass wrote the depth buffer and before the blit.
        if let (Some(smd), Some(bg)) = (&self.sprite_model_dda, &sprite_model_bg) {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("roxlap-gpu sprite_model_dda"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&smd.pipeline);
            cpass.set_bind_group(0, bg, &[]);
            cpass.dispatch_workgroups(surface_w.div_ceil(8), surface_h.div_ceil(8), 1);
        }
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("roxlap-gpu scene_dda blit"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surf_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&dda.pipeline_blit);
            rpass.set_bind_group(0, &dda.blit_bg, &[]);
            rpass.draw(0..3, 0..1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        surf_tex.present();
        self.frame_count = self.frame_count.wrapping_add(1);
    }

    fn build_scene_dda(
        &self,
        width: u32,
        height: u32,
        surface_format: wgpu::TextureFormat,
    ) -> SceneDdaResources {
        let storage_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("roxlap-gpu scene_dda.storage"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let storage_view = storage_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let uniform_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("roxlap-gpu scene_dda.uniform"),
            size: std::mem::size_of::<SceneDdaUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // GPU.9 — per-pixel world-t depth (f32 bits as u32). Sized to
        // the storage texture; written by the scene pass when sprites
        // are active, read+tested by the sprite splatter.
        let depth_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("roxlap-gpu scene_dda.depth"),
            size: u64::from(width) * u64::from(height) * 4,
            // COPY_SRC so `read_depth_pixel` can stage it for picking.
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let depth_readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("roxlap-gpu scene_dda.depth_readback"),
            size: u64::from(width) * u64::from(height) * 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let dda_shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("scene_dda.wgsl"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/scene_dda.wgsl").into()),
            });
        let bgl_dda = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("roxlap-gpu scene_dda.bgl"),
                entries: &[
                    bgl_uniform_entry(0),
                    bgl_storage_entry(1, true),
                    bgl_storage_entry(2, true),
                    bgl_storage_entry(3, true),
                    bgl_storage_entry(4, true),
                    bgl_storage_entry(5, true),
                    bgl_storage_entry(6, true),
                    bgl_storage_entry(7, true),
                    wgpu::BindGroupLayoutEntry {
                        binding: 8,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // GPU.8 sky panorama + sampler.
                    wgpu::BindGroupLayoutEntry {
                        binding: 9,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 10,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // GPU.9 — read-write per-pixel depth buffer.
                    bgl_storage_entry(11, false),
                    // Occupancy pages 1..MAX_OCC_PAGES (page 0 is
                    // binding 1). Unused pages bind a dummy buffer.
                    bgl_storage_entry(12, true),
                    bgl_storage_entry(13, true),
                    bgl_storage_entry(14, true),
                ],
            });
        let dda_pl = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("roxlap-gpu scene_dda.layout"),
                bind_group_layouts: &[&bgl_dda],
                push_constant_ranges: &[],
            });
        let pipeline_dda = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("roxlap-gpu scene_dda.pipeline"),
                layout: Some(&dda_pl),
                module: &dda_shader,
                entry_point: "render_scene",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let blit_shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("blit.wgsl"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/blit.wgsl").into()),
            });
        let bgl_blit = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("roxlap-gpu scene_dda.blit_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                ],
            });
        let blit_pl = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("roxlap-gpu scene_dda.blit_layout"),
                bind_group_layouts: &[&bgl_blit],
                push_constant_ranges: &[],
            });
        let pipeline_blit = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("roxlap-gpu scene_dda.blit_pipeline"),
                layout: Some(&blit_pl),
                vertex: wgpu::VertexState {
                    module: &blit_shader,
                    entry_point: "vs_main",
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &blit_shader,
                    entry_point: "fs_main",
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surface_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });
        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("roxlap-gpu scene_dda.blit_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let blit_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("roxlap-gpu scene_dda.blit_bg"),
            layout: &bgl_blit,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&storage_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        SceneDdaResources {
            storage_size: (width, height),
            storage_view,
            uniform_buf,
            bgl_dda,
            pipeline_dda,
            blit_bg,
            pipeline_blit,
            _sampler: sampler,
            depth_buffer,
            depth_readback,
        }
    }

    /// Read back the per-pixel world-t depth at window pixel `(x, y)`
    /// from the last rendered frame, for screen→world picking. Returns
    /// the distance `t` along the (normalised) view ray to the nearest
    /// scene-grid surface, so the host reconstructs the world hit as
    /// `cam.pos + t * normalize(ray_dir)`. `None` for out-of-bounds
    /// pixels, sky / no-hit (the `T_INF` sentinel), or when no scene
    /// frame has been rendered.
    ///
    /// The depth buffer is the SCENE pass's output (terrain + grids),
    /// untouched by the sprite pass (which reads it read-only), so a
    /// cursor sprite under the pointer does not occlude the pick.
    ///
    /// Synchronous: copies the depth buffer to a mapped staging buffer
    /// and blocks on `device.poll(Wait)`. Cheap enough for click-time
    /// picks; do not call it every frame.
    ///
    /// Requires the last frame to have written depth, which happens
    /// when sprites are present (`write_depth`). The pick demo always
    /// has a cursor sprite, so this holds.
    #[must_use]
    pub fn read_depth_pixel(&self, x: u32, y: u32) -> Option<f32> {
        let dda = self.scene_dda.as_ref()?;
        let (w, h) = dda.storage_size;
        if x >= w || y >= h {
            return None;
        }
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("roxlap-gpu depth readback"),
            });
        let size = u64::from(w) * u64::from(h) * 4;
        enc.copy_buffer_to_buffer(&dda.depth_buffer, 0, &dda.depth_readback, 0, size);
        self.queue.submit(std::iter::once(enc.finish()));

        let slice = dda.depth_readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().ok()?.ok()?;

        let t = {
            let data = slice.get_mapped_range();
            let idx = ((y * w + x) * 4) as usize;
            let bytes: [u8; 4] = data[idx..idx + 4].try_into().ok()?;
            f32::from_le_bytes(bytes)
        };
        dda.depth_readback.unmap();

        // Reject sky / no-hit (T_INF == 1e30 in the shader) + non-finite.
        if !t.is_finite() || t >= 1.0e29 {
            return None;
        }
        Some(t)
    }

    /// GPU.10.1 — upload a sprite model registry + its instances for
    /// the DDA path. An empty instance slice clears all sprites.
    pub fn set_sprite_instances(
        &mut self,
        registry: &sprite_model::SpriteModelRegistry,
        instances: &[sprite_model::SpriteInstance],
    ) {
        if instances.is_empty() {
            self.sprite_registry = None;
            return;
        }
        self.sprite_registry = Some(sprite_model::SpriteRegistryResident::upload(
            &self.device,
            registry,
            instances,
        ));
    }

    /// GPU.10.4 — set the LOD pixel threshold: a sprite steps to the
    /// next mip once a mip-0 voxel would project below `px` screen
    /// pixels. `1.0` is the natural "no sub-pixel voxels" default;
    /// larger values force LOD in closer (useful for inspection).
    /// Clamped to ≥ 0.25.
    pub fn set_sprite_lod_px(&mut self, px: f32) {
        self.sprite_lod_px = px.max(0.25);
    }

    /// GPU.11.1 — set the scene-grid LOD scan distance (world units).
    /// A chunk entered at world-t `t` is marched at mip
    /// `floor(log2(max(t, msd) / msd))`, clamped to its grid's mip
    /// ladder. `0` disables LOD (always mip-0). Larger values push
    /// the coarser mips farther out — the axis-aligned-mip-beams
    /// mitigation lever (GPU.11.2). Default 64 (matches CPU
    /// `mip_scan_dist`).
    pub fn set_scene_mip_scan_dist(&mut self, dist: f32) {
        self.scene_mip_scan_dist = dist.max(0.0);
    }

    /// GPU.10.1 — build the instanced model-DDA pipeline (one thread
    /// per pixel). Lazily invoked the first frame a registry is present.
    fn build_sprite_model_dda(&self) -> SpriteModelDdaResources {
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("sprite_model_dda.wgsl"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/sprite_model_dda.wgsl").into(),
                ),
            });
        let bgl = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("roxlap-gpu sprite_model_dda.bgl"),
                entries: &[
                    bgl_uniform_entry(0),
                    bgl_storage_entry(1, true), // occupancy
                    bgl_storage_entry(2, true), // colors
                    bgl_storage_entry(3, true), // color_offsets
                    bgl_storage_entry(4, true), // model_meta
                    bgl_storage_entry(5, true), // instances
                    bgl_storage_entry(6, true), // scene depth
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    bgl_storage_entry(8, true), // tile_ranges
                    bgl_storage_entry(9, true), // tile_instances
                ],
            });
        let pl = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("roxlap-gpu sprite_model_dda.layout"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("roxlap-gpu sprite_model_dda.pipeline"),
                layout: Some(&pl),
                module: &shader,
                entry_point: "march",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let uniform_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("roxlap-gpu sprite_model_dda.uniform"),
            size: std::mem::size_of::<SpriteModelUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        SpriteModelDdaResources {
            bgl,
            pipeline,
            uniform_buf,
        }
    }
}

/// GPU.11 — headless scene-DDA renderer for tests + offline visual
/// gates. Owns the `scene_dda.wgsl` compute pipeline with no surface
/// and no blit pass; renders a [`GpuSceneResident`] to an in-memory
/// RGBA framebuffer via texture readback. The per-substage visual
/// gate (render reference scenes, diff PPMs) and the GPU.11.1 mip
/// render-diff both ride on this.
pub struct HeadlessSceneRenderer {
    width: u32,
    height: u32,
    output_tex: wgpu::Texture,
    output_view: wgpu::TextureView,
    depth_buffer: wgpu::Buffer,
    uniform_buf: wgpu::Buffer,
    _sky_texture: wgpu::Texture,
    sky_view: wgpu::TextureView,
    sky_sampler: wgpu::Sampler,
    bgl: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    readback: wgpu::Buffer,
    padded_bytes_per_row: u32,
}

impl HeadlessSceneRenderer {
    /// Build the compute pipeline + output/readback resources for a
    /// `width × height` framebuffer. Validates `scene_dda.wgsl` and
    /// the [`scene::GridStaticMeta`] std430 layout at pipeline /
    /// bind-group time.
    #[must_use]
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let output_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("roxlap-gpu headless.output"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let output_view = output_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("roxlap-gpu headless.uniform"),
            size: std::mem::size_of::<SceneDdaUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let depth_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("roxlap-gpu headless.depth"),
            size: u64::from(width) * u64::from(height) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let default_sky_pixel = [120u8, 150, 220, 255];
        let (sky_texture, sky_view) = create_sky_texture(device, 1, 1, &default_sky_pixel);
        let sky_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("roxlap-gpu headless.sky_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scene_dda.wgsl (headless)"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/scene_dda.wgsl").into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("roxlap-gpu headless.bgl"),
            entries: &[
                bgl_uniform_entry(0),
                bgl_storage_entry(1, true),
                bgl_storage_entry(2, true),
                bgl_storage_entry(3, true),
                bgl_storage_entry(4, true),
                bgl_storage_entry(5, true),
                bgl_storage_entry(6, true),
                bgl_storage_entry(7, true),
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                bgl_storage_entry(11, false),
                bgl_storage_entry(12, true),
                bgl_storage_entry(13, true),
                bgl_storage_entry(14, true),
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("roxlap-gpu headless.layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("roxlap-gpu headless.pipeline"),
            layout: Some(&pl),
            module: &shader,
            entry_point: "render_scene",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Readback buffer: row pitch must be 256-aligned for
        // copy_texture_to_buffer.
        let padded_bytes_per_row = (width * 4).div_ceil(256) * 256;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("roxlap-gpu headless.readback"),
            size: u64::from(padded_bytes_per_row) * u64::from(height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            width,
            height,
            output_tex,
            output_view,
            depth_buffer,
            uniform_buf,
            _sky_texture: sky_texture,
            sky_view,
            sky_sampler,
            bgl,
            pipeline,
            readback,
            padded_bytes_per_row,
        }
    }

    /// Render `scene` from `cameras` (one per grid) and read the
    /// framebuffer back as `width*height` packed `0xAABBGGRR` pixels
    /// (R in the low byte). Fog is disabled. `mip_scan_dist` drives
    /// the GPU.11.1 scene-grid LOD (`0` = always mip-0). Blocks on
    /// readback.
    ///
    /// # Panics
    /// If `cameras.len() != scene.grid_count`.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &GpuSceneResident,
        cameras: &[Camera],
        fov_y_rad: f32,
        max_outer_steps: u32,
        mip_scan_dist: f32,
    ) -> Vec<u32> {
        assert_eq!(
            cameras.len(),
            scene.grid_count as usize,
            "headless render: {} cameras for {} grids",
            cameras.len(),
            scene.grid_count,
        );

        let mut cam_array = [SceneDdaPerGridCamera::zeroed(); SCENE_MAX_GRIDS];
        for (i, cam) in cameras.iter().enumerate() {
            cam_array[i] = SceneDdaPerGridCamera {
                pos: cam.position,
                _pad0: 0.0,
                right: cam.right,
                _pad1: 0.0,
                down: cam.down,
                _pad2: 0.0,
                forward: cam.forward,
                _pad3: 0.0,
            };
        }
        let uniform = SceneDdaUniform {
            fov_y_rad,
            grid_count: scene.grid_count,
            max_outer_steps,
            _pad0: 0,
            screen_size: [self.width, self.height],
            _pad1: [0; 2],
            cameras: cam_array,
            // Fog off: near/far past any reachable t → factor 0.
            fog_color: [0.0, 0.0, 0.0, 1.0e29],
            fog_far: 1.0e30,
            write_depth: 0,
            occ_page_words: scene.occupancy_page_words,
            occ_num_pages: scene.occupancy_num_pages,
            mip_scan_dist,
            _pad2: 0,
            _pad3: 0,
            _pad4: 0,
        };
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniform));

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("roxlap-gpu headless.bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: scene.occupancy_pages[0].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: scene.all_color_offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: scene.all_colors.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: scene.all_chunk_colors_base.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: scene.all_chunk_occupancy.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: scene.grid_static_meta.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: scene.all_slot_chunk_idx.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(&self.output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::TextureView(&self.sky_view),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::Sampler(&self.sky_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: self.depth_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: scene.occupancy_pages[1].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: scene.occupancy_pages[2].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: scene.occupancy_pages[3].as_entire_binding(),
                },
            ],
        });

        let mut enc =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("roxlap-gpu headless.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);
        }
        enc.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.output_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &self.readback,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_bytes_per_row),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(enc.finish()));

        let slice = self.readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().expect("map_async channel").expect("map_async");

        let data = slice.get_mapped_range();
        let mut out = Vec::with_capacity((self.width * self.height) as usize);
        let pitch = self.padded_bytes_per_row as usize;
        for y in 0..self.height as usize {
            let row = &data[y * pitch..y * pitch + self.width as usize * 4];
            for px in row.chunks_exact(4) {
                out.push(
                    u32::from(px[0])
                        | (u32::from(px[1]) << 8)
                        | (u32::from(px[2]) << 16)
                        | (u32::from(px[3]) << 24),
                );
            }
        }
        drop(data);
        self.readback.unmap();
        out
    }
}

fn bgl_uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Create a fresh sky panorama texture sized `width × height` with
/// the initial pixel data uploaded via `write_texture`. Used by
/// `GpuRenderer::new` (1×1 default) and `set_sky_panorama` (host-
/// supplied panorama).
fn create_sky_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    _initial_pixels: &[u8],
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("roxlap-gpu sky_texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

/// GPU.4 needs to upload a whole grid (~hundreds of MiB) as a few
/// storage buffers. wgpu's default `max_storage_buffer_binding_size`
/// is 128 MiB, which is just enough for the demo's 32×32 ground
/// occupancy (~128 MiB) but not the colour array. We request as
/// much as the adapter is willing to give — most desktop GPUs cap
/// individual storage buffers at 2-4 GiB; iGPUs often offer the
/// full system memory.
pub(crate) fn pick_required_limits(adapter_limits: &wgpu::Limits) -> wgpu::Limits {
    wgpu::Limits {
        max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
        max_buffer_size: adapter_limits.max_buffer_size,
        // Occupancy paging adds up to MAX_OCC_PAGES-1 extra storage
        // bindings; with the scene's other buffers + the GPU.9 depth
        // buffer the scene_dda stage needs ~11. The default cap is 8.
        // Both NVK and lavapipe advertise ≫16, so request 16.
        max_storage_buffers_per_shader_stage: adapter_limits
            .max_storage_buffers_per_shader_stage
            .min(16),
        ..wgpu::Limits::default()
    }
}

fn pick_present_mode(modes: &[wgpu::PresentMode]) -> wgpu::PresentMode {
    // Prefer Mailbox > Immediate > Fifo. Fifo is the universal
    // fallback and the only one Wayland-on-Mesa always offers.
    for &m in &[wgpu::PresentMode::Mailbox, wgpu::PresentMode::Immediate] {
        if modes.contains(&m) {
            return m;
        }
    }
    wgpu::PresentMode::Fifo
}
