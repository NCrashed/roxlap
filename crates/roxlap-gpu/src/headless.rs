//! GPU.2 — headless device + queue for tests and offline tools.
//!
//! Stands up a wgpu device with no surface — the surface dependency
//! is what couples `GpuRenderer` to winit. Tests that exercise the
//! decompressed-chunk upload + read-back path want neither a window
//! nor a swapchain; this module exposes that bare device.
//!
//! Same `GpuInitError` surface as `GpuRenderer::new` so callers can
//! share the fall-back-to-CPU pattern.

use crate::{GpuInitError, GpuRendererSettings, PowerPreference};

/// Bare wgpu device + queue with no surface. Suitable for
/// `wgpu::ComputePass` work that doesn't present.
pub struct HeadlessGpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_info: String,
}

impl HeadlessGpu {
    /// # Errors
    /// Returns [`GpuInitError::NoAdapter`] if no compatible adapter
    /// is present (no Vulkan/Metal/DX12 driver), or
    /// [`GpuInitError::RequestDevice`] if device creation fails.
    pub async fn new(settings: GpuRendererSettings) -> Result<Self, GpuInitError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let power_preference = match settings.power_preference {
            PowerPreference::Low => wgpu::PowerPreference::LowPower,
            PowerPreference::High => wgpu::PowerPreference::HighPerformance,
        };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference,
                compatible_surface: None,
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
                    label: Some("roxlap-gpu headless device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await?;
        Ok(Self {
            device,
            queue,
            adapter_info,
        })
    }

    /// Synchronous wrapper for tests / hosts without an async runtime.
    ///
    /// # Errors
    /// See [`Self::new`].
    pub fn new_blocking(settings: GpuRendererSettings) -> Result<Self, GpuInitError> {
        pollster::block_on(Self::new(settings))
    }
}
