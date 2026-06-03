//! GPU.0 probe — standalone WGPU compute-shader voxel marcher.
//!
//! Per `PORTING-GPU.md`, this is the empirical FPS-ceiling check
//! before committing to the rest of the stage. A hand-built 64³
//! voxel chunk is uploaded as (occupancy bitmap + per-voxel colour
//! array); one compute shader marches an Amanatides–Woo 3D DDA per
//! pixel; a fullscreen-triangle blit upscales the low-res output
//! into the swapchain at the surface's native size.
//!
//! Resolution presets (storage-texture / compute-output size, not
//! window size):
//!
//!   1 → 320×240
//!   2 → 640×480  (default)
//!   3 → 1280×720
//!
//! Override the startup resolution with `ROXLAP_GPU_RES=320x240`,
//! `=640x480`, or `=1280x720`. The window is resizable; the storage
//! texture is independent and nearest-neighbour-upscaled.
//!
//! Run with `cargo run --release -p roxlap-gpu --example probe`.
//! `--release` matters — debug builds murder DDA throughput.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::similar_names
)]

use std::sync::Arc;
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowAttributes, WindowId};

const CHUNK_SIZE: u32 = 64;
const CHUNK_VOXELS: usize = (CHUNK_SIZE as usize).pow(3);
const MAX_SCAN_DIST: u32 = 256;
const DEFAULT_W: u32 = 640;
const DEFAULT_H: u32 = 480;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    inv_view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    _pad0: f32,
    screen_size: [u32; 2],
    chunk_size: u32,
    max_scan_dist: u32,
}

#[derive(Clone, Copy)]
struct Resolution {
    w: u32,
    h: u32,
}

impl Resolution {
    fn from_env() -> Self {
        let want = std::env::var("ROXLAP_GPU_RES").unwrap_or_default();
        match want.as_str() {
            "320x240" => Self { w: 320, h: 240 },
            "1280x720" => Self { w: 1280, h: 720 },
            _ => Self {
                w: DEFAULT_W,
                h: DEFAULT_H,
            },
        }
    }
}

/// Build a 64³ chunk with a recognisable silhouette: a checkered
/// ground slab, four corner pillars, a centred sphere. Returns
/// (occupancy bitmap packed into u32s, per-voxel RGB colour array).
fn build_chunk() -> (Vec<u32>, Vec<u32>) {
    let s = CHUNK_SIZE as i32;
    let mut occ = vec![0u32; CHUNK_VOXELS.div_ceil(32)];
    let mut col = vec![0u32; CHUNK_VOXELS];

    let idx = |x: i32, y: i32, z: i32| -> usize {
        (x as usize)
            + (y as usize) * (CHUNK_SIZE as usize)
            + (z as usize) * (CHUNK_SIZE as usize) * (CHUNK_SIZE as usize)
    };
    let set = |occ: &mut [u32], col: &mut [u32], x: i32, y: i32, z: i32, rgb: u32| {
        let i = idx(x, y, z);
        occ[i >> 5] |= 1u32 << (i & 31);
        col[i] = rgb;
    };

    // Ground slab z ∈ [0, 8) with an 8-voxel checker tint.
    for z in 0..8 {
        for y in 0..s {
            for x in 0..s {
                let cb = ((x / 8) ^ (y / 8) ^ (z / 4)) & 1;
                let rgb = if cb == 0 { 0x006c_4a2b } else { 0x008b_6f3f };
                set(&mut occ, &mut col, x, y, z, rgb);
            }
        }
    }

    // Four corner pillars, blue, ~40 voxels tall, 3×3 footprint.
    for &(px, py) in &[(4, 4), (s - 7, 4), (4, s - 7), (s - 7, s - 7)] {
        for z in 0..40 {
            for dx in 0..3 {
                for dy in 0..3 {
                    set(&mut occ, &mut col, px + dx, py + dy, z, 0x003e_6ac1);
                }
            }
        }
    }

    // Centre sphere, red-orange, lifted off the floor.
    let cx = s / 2;
    let cy = s / 2;
    let cz = s / 2 + 8;
    let r = 14;
    let r2 = r * r;
    for z in (cz - r).max(0)..(cz + r).min(s) {
        for y in (cy - r).max(0)..(cy + r).min(s) {
            for x in (cx - r).max(0)..(cx + r).min(s) {
                let dx = x - cx;
                let dy = y - cy;
                let dz = z - cz;
                if dx * dx + dy * dy + dz * dz <= r2 {
                    set(&mut occ, &mut col, x, y, z, 0x00d2_553b);
                }
            }
        }
    }

    (occ, col)
}

struct Renderer {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,

    storage_view: wgpu::TextureView,
    uniforms_buf: wgpu::Buffer,
    occupancy_buf: wgpu::Buffer,
    colors_buf: wgpu::Buffer,
    compute_bgl: wgpu::BindGroupLayout,
    compute_bg: wgpu::BindGroup,
    compute_pipeline: wgpu::ComputePipeline,

    blit_bgl: wgpu::BindGroupLayout,
    blit_bg: wgpu::BindGroup,
    blit_pipeline: wgpu::RenderPipeline,
    blit_sampler: wgpu::Sampler,

    probe_res: Resolution,
    start: Instant,
    frames: u32,
    last_fps_at: Instant,
}

impl Renderer {
    async fn new(window: Arc<Window>, probe_res: Resolution) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let surface = instance
            .create_surface(window.clone())
            .expect("create_surface");
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("no compatible adapter — does this system have a Vulkan/Metal/DX12 driver?");

        eprintln!(
            "GPU.0 probe: adapter = {:?} ({:?}, {:?})",
            adapter.get_info().name,
            adapter.get_info().backend,
            adapter.get_info().device_type,
        );

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("roxlap-gpu probe device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .expect("request_device");

        // Surface config — pick an sRGB swapchain format if offered.
        let caps = surface.get_capabilities(&adapter);
        let surface_format = caps
            .formats
            .iter()
            .copied()
            .find(wgpu::TextureFormat::is_srgb)
            .unwrap_or(caps.formats[0]);
        let physical = window.inner_size();
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: physical.width.max(1),
            height: physical.height.max(1),
            present_mode: caps.present_modes[0],
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        // Storage texture for the compute shader's per-pixel output.
        let storage_view = make_storage_view(&device, probe_res);

        // Uniforms — one CameraUniform per frame.
        let uniforms_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("probe.uniforms"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Hand-built voxel chunk uploaded once at startup.
        let (occupancy, colors) = build_chunk();
        let occupancy_buf = create_storage_buffer(&device, "probe.occupancy", &occupancy);
        let colors_buf = create_storage_buffer(&device, "probe.colors", &colors);

        // Compute pipeline.
        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("probe.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/probe.wgsl").into()),
        });
        let compute_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("probe.compute_bgl"),
            entries: &[
                bgl_uniform_entry(0, wgpu::ShaderStages::COMPUTE),
                bgl_storage_buffer_entry(1, wgpu::ShaderStages::COMPUTE, true),
                bgl_storage_buffer_entry(2, wgpu::ShaderStages::COMPUTE, true),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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
        let compute_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("probe.compute_layout"),
            bind_group_layouts: &[&compute_bgl],
            push_constant_ranges: &[],
        });
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("probe.compute"),
            layout: Some(&compute_pl),
            module: &compute_shader,
            entry_point: "render_frame",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let compute_bg = make_compute_bg(
            &device,
            &compute_bgl,
            &uniforms_buf,
            &occupancy_buf,
            &colors_buf,
            &storage_view,
        );

        // Blit pipeline.
        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blit.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/blit.wgsl").into()),
        });
        let blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("probe.blit_bgl"),
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
        let blit_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("probe.blit_layout"),
            bind_group_layouts: &[&blit_bgl],
            push_constant_ranges: &[],
        });
        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("probe.blit"),
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
        let blit_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("probe.blit_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let blit_bg = make_blit_bg(&device, &blit_bgl, &storage_view, &blit_sampler);

        Self {
            window,
            surface,
            surface_config,
            device,
            queue,
            storage_view,
            uniforms_buf,
            occupancy_buf,
            colors_buf,
            compute_bgl,
            compute_bg,
            compute_pipeline,
            blit_bgl,
            blit_bg,
            blit_pipeline,
            blit_sampler,
            probe_res,
            start: Instant::now(),
            frames: 0,
            last_fps_at: Instant::now(),
        }
    }

    fn resize_surface(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 {
            return;
        }
        self.surface_config.width = w;
        self.surface_config.height = h;
        self.surface.configure(&self.device, &self.surface_config);
    }

    fn set_probe_res(&mut self, res: Resolution) {
        if res.w == self.probe_res.w && res.h == self.probe_res.h {
            return;
        }
        self.probe_res = res;
        self.storage_view = make_storage_view(&self.device, res);
        self.compute_bg = make_compute_bg(
            &self.device,
            &self.compute_bgl,
            &self.uniforms_buf,
            &self.occupancy_buf,
            &self.colors_buf,
            &self.storage_view,
        );
        self.blit_bg = make_blit_bg(
            &self.device,
            &self.blit_bgl,
            &self.storage_view,
            &self.blit_sampler,
        );
        self.frames = 0;
        self.last_fps_at = Instant::now();
    }

    fn render(&mut self) {
        // Orbit camera around the chunk centre at a fixed radius +
        // height; full revolution every ~6 s.
        let t = self.start.elapsed().as_secs_f32();
        let cs = CHUNK_SIZE as f32;
        let centre = Vec3::new(cs * 0.5, cs * 0.5, cs * 0.35);
        let radius = cs * 1.6;
        let height = cs * 0.95;
        let cam = Vec3::new(
            centre.x + radius * t.cos(),
            centre.y + radius * t.sin(),
            centre.z + height,
        );
        let view = Mat4::look_at_rh(cam, centre, Vec3::Z);
        let aspect = self.probe_res.w as f32 / self.probe_res.h as f32;
        let proj = Mat4::perspective_rh(60_f32.to_radians(), aspect, 0.5, 5000.0);
        let inv_vp = (proj * view).inverse();

        let uniforms = CameraUniform {
            inv_view_proj: inv_vp.to_cols_array_2d(),
            camera_pos: cam.into(),
            _pad0: 0.0,
            screen_size: [self.probe_res.w, self.probe_res.h],
            chunk_size: CHUNK_SIZE,
            max_scan_dist: MAX_SCAN_DIST,
        };
        self.queue
            .write_buffer(&self.uniforms_buf, 0, bytemuck::bytes_of(&uniforms));

        let surf_tex = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Lost) => {
                self.surface.configure(&self.device, &self.surface_config);
                return;
            }
            Err(e) => {
                eprintln!("surface error: {e:?}");
                return;
            }
        };
        let surf_view = surf_tex
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("probe.encoder"),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("probe.compute_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, &self.compute_bg, &[]);
            let wgx = self.probe_res.w.div_ceil(8);
            let wgy = self.probe_res.h.div_ceil(8);
            cpass.dispatch_workgroups(wgx, wgy, 1);
        }
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("probe.blit_pass"),
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
            rpass.set_pipeline(&self.blit_pipeline);
            rpass.set_bind_group(0, &self.blit_bg, &[]);
            rpass.draw(0..3, 0..1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        surf_tex.present();

        // FPS rolling-window — update title every ~500 ms.
        self.frames += 1;
        let now = Instant::now();
        let dt = (now - self.last_fps_at).as_secs_f32();
        if dt >= 0.5 {
            let fps = f64::from(self.frames) / f64::from(dt);
            self.window.set_title(&format!(
                "roxlap-gpu probe — {}x{} @ {:.1} FPS (1/2/3 = res, Esc = quit)",
                self.probe_res.w, self.probe_res.h, fps,
            ));
            eprintln!(
                "probe: {}x{} @ {:.1} FPS",
                self.probe_res.w, self.probe_res.h, fps,
            );
            self.frames = 0;
            self.last_fps_at = now;
        }
    }
}

fn make_storage_view(device: &wgpu::Device, res: Resolution) -> wgpu::TextureView {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("probe.storage"),
        size: wgpu::Extent3d {
            width: res.w,
            height: res.h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    tex.create_view(&wgpu::TextureViewDescriptor::default())
}

fn create_storage_buffer(device: &wgpu::Device, label: &str, data: &[u32]) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

fn bgl_uniform_entry(binding: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: vis,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_storage_buffer_entry(
    binding: u32,
    vis: wgpu::ShaderStages,
    read_only: bool,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: vis,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn make_compute_bg(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniforms: &wgpu::Buffer,
    occupancy: &wgpu::Buffer,
    colors: &wgpu::Buffer,
    output: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("probe.compute_bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniforms.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: occupancy.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: colors.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(output),
            },
        ],
    })
}

fn make_blit_bg(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    storage: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("probe.blit_bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(storage),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    initial_res: Resolution,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = WindowAttributes::default()
            .with_title("roxlap-gpu probe — GPU.0")
            .with_inner_size(winit::dpi::LogicalSize::new(
                self.initial_res.w,
                self.initial_res.h,
            ));
        let window = Arc::new(event_loop.create_window(attrs).expect("create_window"));
        let renderer = pollster::block_on(Renderer::new(window.clone(), self.initial_res));
        self.window = Some(window);
        self.renderer = Some(renderer);
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => renderer.resize_surface(size.width, size.height),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(code),
                        ..
                    },
                ..
            } => match code {
                KeyCode::Escape => event_loop.exit(),
                KeyCode::Digit1 => renderer.set_probe_res(Resolution { w: 320, h: 240 }),
                KeyCode::Digit2 => renderer.set_probe_res(Resolution { w: 640, h: 480 }),
                KeyCode::Digit3 => renderer.set_probe_res(Resolution { w: 1280, h: 720 }),
                _ => {}
            },
            WindowEvent::RedrawRequested => renderer.render(),
            _ => {}
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().expect("EventLoop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        window: None,
        renderer: None,
        initial_res: Resolution::from_env(),
    };
    event_loop.run_app(&mut app).expect("run_app");
}
