//! GPU.2 — chunk-resident storage buffers + debug read-back.
//!
//! Uploads a [`ChunkUpload`] as three storage buffers (occupancy,
//! per-column colour offsets, packed colour array) on a wgpu
//! device. The [`GpuChunkResident::read_voxel_blocking`] helper
//! dispatches the `debug_read.wgsl` shader to extract a single
//! voxel's colour via map-async readback, validating the round trip
//! demanded by `PORTING-GPU.md` §GPU.2.

#![allow(clippy::too_many_lines, clippy::missing_panics_doc)]

use std::num::NonZeroU64;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::decompress::{ChunkUpload, CHUNK_Z};

/// Uniform handed to `debug_read.wgsl` — the voxel coordinate to
/// probe plus the chunk extents the shader needs to index occupancy.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ProbeUniform {
    coord: [u32; 3],
    vsid: u32,
    chunk_z: u32,
    _pad: [u32; 3],
}

/// GPU-side storage for one decompressed chunk. Owns its buffers;
/// dropping releases them.
pub struct GpuChunkResident {
    pub vsid: u32,
    pub occupancy: wgpu::Buffer,
    pub color_offsets: wgpu::Buffer,
    pub colors: wgpu::Buffer,
    pub occupancy_bytes: u64,
    pub color_offsets_bytes: u64,
    pub colors_bytes: u64,

    // Debug-read scaffolding. In GPU.3+ the main render shader
    // consumes the storage buffers directly; for GPU.2 these are
    // the only consumer.
    probe_uniform: wgpu::Buffer,
    probe_output: wgpu::Buffer,
    probe_readback: wgpu::Buffer,
    probe_bg: wgpu::BindGroup,
    probe_pipeline: wgpu::ComputePipeline,
}

impl GpuChunkResident {
    /// Upload `chunk` to `device`. Single-shot allocation; no
    /// streaming machinery yet — that arrives in GPU.6 / GPU.7.
    pub fn upload(device: &wgpu::Device, chunk: &ChunkUpload) -> Self {
        let occupancy = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("roxlap-gpu chunk.occupancy"),
            contents: bytemuck::cast_slice(&chunk.occupancy),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let color_offsets = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("roxlap-gpu chunk.color_offsets"),
            contents: bytemuck::cast_slice(&chunk.color_offsets),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let colors = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("roxlap-gpu chunk.colors"),
            contents: bytemuck::cast_slice(&chunk.colors),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let occupancy_bytes = (chunk.occupancy.len() * 4) as u64;
        let color_offsets_bytes = (chunk.color_offsets.len() * 4) as u64;
        let colors_bytes = (chunk.colors.len() * 4) as u64;

        // Debug-read scaffolding ----------------------------------------------
        let probe_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("roxlap-gpu chunk.probe_uniform"),
            size: std::mem::size_of::<ProbeUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let probe_output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("roxlap-gpu chunk.probe_output"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let probe_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("roxlap-gpu chunk.probe_readback"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("debug_read.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/debug_read.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("roxlap-gpu chunk.probe_bgl"),
            entries: &[
                bgl_uniform_entry(0),
                bgl_storage_entry(1, true),
                bgl_storage_entry(2, true),
                bgl_storage_entry(3, true),
                bgl_storage_entry(4, false),
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("roxlap-gpu chunk.probe_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("roxlap-gpu chunk.probe_pipeline"),
            layout: Some(&pl),
            module: &shader,
            entry_point: "debug_read",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("roxlap-gpu chunk.probe_bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: probe_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: occupancy.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: color_offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: colors.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: probe_output.as_entire_binding(),
                },
            ],
        });

        Self {
            vsid: chunk.vsid,
            occupancy,
            color_offsets,
            colors,
            occupancy_bytes,
            color_offsets_bytes,
            colors_bytes,
            probe_uniform,
            probe_output,
            probe_readback,
            probe_bg: bg,
            probe_pipeline: pipeline,
        }
    }

    /// Total resident bytes (occupancy + offsets + colours) — for
    /// the upload-time benchmark.
    pub fn resident_bytes(&self) -> u64 {
        self.occupancy_bytes + self.color_offsets_bytes + self.colors_bytes
    }

    /// Round-trip read of a single voxel via the debug shader.
    /// Returns `Some(rgb)` for a solid voxel (textured or bedrock),
    /// `None` for empty / out-of-bounds.
    ///
    /// Blocks until the GPU finishes; not intended for the render
    /// hot path. The GPU.2 validation test is the only caller.
    pub fn read_voxel_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        x: u32,
        y: u32,
        z: u32,
    ) -> Option<u32> {
        let uniform = ProbeUniform {
            coord: [x, y, z],
            vsid: self.vsid,
            chunk_z: CHUNK_Z,
            _pad: [0; 3],
        };
        queue.write_buffer(&self.probe_uniform, 0, bytemuck::bytes_of(&uniform));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("roxlap-gpu chunk.read_voxel"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("roxlap-gpu chunk.debug_read"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.probe_pipeline);
            cpass.set_bind_group(0, &self.probe_bg, &[]);
            cpass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.probe_output, 0, &self.probe_readback, 0, 4);
        queue.submit(std::iter::once(encoder.finish()));

        // Map the readback buffer. wgpu's map_async runs the
        // callback when the device.poll(Wait) services it; pollster
        // turns the resulting future into a blocking wait.
        let slice = self.probe_readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            tx.send(res).expect("send map result");
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .expect("recv map result")
            .expect("map_async returned an error");

        let bytes = slice.get_mapped_range();
        let value = u32::from_le_bytes(bytes[..4].try_into().expect("4 bytes"));
        drop(bytes);
        self.probe_readback.unmap();

        if value == 0 {
            None
        } else {
            Some(value)
        }
    }
}

fn bgl_uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: NonZeroU64::new(std::mem::size_of::<ProbeUniform>() as u64),
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
