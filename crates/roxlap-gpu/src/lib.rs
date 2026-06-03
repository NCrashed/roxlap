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

#![allow(clippy::must_use_candidate)]

use std::sync::Arc;

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

/// GPU.1 scaffold renderer. Owns a wgpu device, queue, and surface
/// bound to the host's winit window. `render` presents a single
/// clear-to-colour frame each call.
pub struct GpuRenderer {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_info: String,
    clear_colour: [f64; 3],
    frame_count: u32,
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
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await?;

        let caps = surface.get_capabilities(&adapter);
        let surface_format = caps
            .formats
            .iter()
            .copied()
            .find(wgpu::TextureFormat::is_srgb)
            .unwrap_or(caps.formats[0]);
        let present_mode = if settings.uncapped_present {
            pick_present_mode(&caps.present_modes)
        } else {
            wgpu::PresentMode::Fifo
        };
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

        Ok(Self {
            window,
            surface,
            surface_config,
            device,
            queue,
            adapter_info,
            clear_colour: settings.clear_colour,
            frame_count: 0,
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

    /// Re-configure the swapchain to a new physical size. Call from
    /// `WindowEvent::Resized`.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.surface_config.width = width;
        self.surface_config.height = height;
        self.surface.configure(&self.device, &self.surface_config);
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
