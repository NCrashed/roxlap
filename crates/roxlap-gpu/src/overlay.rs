//! QE.8b — deferred overlay passes, split verbatim out of `lib.rs`:
//! the world-space debug-line pass (`line.wgsl`), the textured
//! image-quad pass (`image.wgsl`) with its retained image textures,
//! and the egui HUD paint (`hud` feature). All draw `LoadOp::Load`
//! over the pending frame, between `render_scene` and `present`.

use bytemuck::{Pod, Zeroable};

use crate::GpuRenderer;

/// WGPU-backed renderer. Owns the device, queue, and surface
/// bound to the host's window. [`GpuRenderer::render`] is the GPU.1
/// clear-to-colour path; [`GpuRenderer::render_scene`] is the
/// multi-grid scene marcher.
///
/// The window is consumed only at construction — `wgpu`'s
/// `Surface<'static>` keeps its own `Arc` clone of the handle, so
/// the renderer holds no window field of its own.
/// A world-space line segment for [`GpuRenderer::draw_lines_deferred`].
/// `color` is straight RGBA in `0..=1` (the alpha drives the over-blend);
/// `width_px` is the screen-space thickness; `depth_test` occludes the
/// segment behind nearer marched geometry.
#[derive(Clone, Copy, Debug)]
pub struct GpuLine {
    /// First endpoint, world voxel units.
    pub a: [f32; 3],
    /// Second endpoint, world voxel units.
    pub b: [f32; 3],
    /// Straight (non-premultiplied) RGBA, each channel `0..=1`; alpha
    /// drives the over-blend onto the frame.
    pub color: [f32; 4],
    /// Screen-space line thickness in pixels; values below `1.0` are
    /// clamped up to 1 px.
    pub width_px: f32,
    /// `true` ⇒ fragments behind nearer marched scene geometry are
    /// discarded (with a small bias so surface-hugging lines don't
    /// z-fight); `false` ⇒ always drawn on top.
    pub depth_test: bool,
}

/// World camera basis for projecting [`GpuLine`] endpoints — the same
/// pinhole the scene-DDA pass marches with (`right`/`down`/`forward`
/// orthonormal, `pos` in world voxel units).
#[derive(Clone, Copy, Debug)]
pub struct GpuLineCamera {
    /// Eye position in world voxel units.
    pub pos: [f32; 3],
    /// Unit basis toward screen-right (must match the scene camera's
    /// right-handed `right × down == forward` basis).
    pub right: [f32; 3],
    /// Unit basis toward screen-down (+z is down in voxlap space).
    pub down: [f32; 3],
    /// Unit view direction; endpoints with a forward component below
    /// the near plane are clipped before projection.
    pub forward: [f32; 3],
}

/// Near plane (camera-forward distance) below which a [`GpuLine`] endpoint
/// is clipped, so the pinhole divide stays finite.
pub(crate) const LINE_NEAR_Z: f32 = 0.0625;
/// Depth-test slack (euclidean world distance) so a line resting on the
/// surface it traces doesn't z-fight the marched geometry.
const LINE_DEPTH_BIAS: f32 = 0.5;

/// One expanded-quad vertex (`build_line_vertices` output). `pos` is NDC;
/// `depth` is the euclidean world distance of the source endpoint (the
/// marcher's `best_t` metric); `depth_test` is `1.0`/`0.0`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LineVertex {
    pos: [f32; 2],
    depth: f32,
    depth_test: f32,
    color: [f32; 4],
}

/// `line.wgsl` / `image.wgsl` fragment uniform (std140; padded to 32 bytes
/// so the uniform's struct stride is a 16-byte multiple).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LineParams {
    /// Target (swapchain) size — the range of the fragment's `clip.xy`.
    screen_w: u32,
    screen_h: u32,
    depth_bias: f32,
    no_depth: u32,
    /// 1 when the viewport flip is on. The depth buffer is written
    /// unflipped (the blit mirrors at read time), but these passes flip the
    /// vertex NDC X, so the fragment must mirror its depth lookup to match.
    flip_x: u32,
    /// RP.0 — the **render** (logical) size the depth buffer is stored at.
    /// The fragment scales its swapchain `clip.xy` into this grid for the
    /// depth lookup. Equal to `screen_*` under `Native` (identity).
    depth_w: u32,
    depth_h: u32,
    _pad: u32,
}

/// Lazy-built debug-line pipeline (L3.2). The bind group is rebuilt each
/// draw (it references the current `scene_dda.depth_buffer`, which the
/// swapchain resize recreates); the pipeline / layout / uniform persist.
pub(crate) struct LineResources {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    uniform_buf: wgpu::Buffer,
    /// 1-word stand-in bound when no scene depth exists (sprite-only /
    /// empty scene); `no_depth = 1` keeps the shader from indexing it.
    dummy_depth: wgpu::Buffer,
}

/// Project + expand world-space [`GpuLine`]s into screen-space quad
/// vertices (6 per visible segment) for `line.wgsl`. Mirrors the
/// scene-DDA pinhole (`forward + ndc_x·half_w·right − ndc_y·half_h·down`)
/// so lines land on the marched geometry, carrying each endpoint's
/// euclidean world distance as the depth-test key (= the marcher's
/// `best_t`). Segments fully behind the near plane are dropped; the rest
/// are clipped to it.
fn build_line_vertices(
    cam: &GpuLineCamera,
    lines: &[GpuLine],
    w: u32,
    h: u32,
    fov_y: f32,
    flip_x: bool,
) -> Vec<LineVertex> {
    let aspect = w as f32 / h as f32;
    let half_h = (fov_y * 0.5).tan();
    let half_w = half_h * aspect;
    let (wf, hf) = (w as f32, h as f32);

    let cam_coords = |p: [f32; 3]| -> [f32; 3] {
        let d = [p[0] - cam.pos[0], p[1] - cam.pos[1], p[2] - cam.pos[2]];
        [
            cam.right[0] * d[0] + cam.right[1] * d[1] + cam.right[2] * d[2],
            cam.down[0] * d[0] + cam.down[1] * d[1] + cam.down[2] * d[2],
            cam.forward[0] * d[0] + cam.forward[1] * d[1] + cam.forward[2] * d[2],
        ]
    };
    // Camera-space point → (NDC xy, euclidean depth). NDC y is up (+1 top),
    // matching WebGPU clip space; depth is the marcher's world-t metric.
    let project = |q: [f32; 3]| -> ([f32; 2], f32) {
        let inv = 1.0 / q[2];
        let nx = q[0] * inv / half_w;
        let ny = -q[1] * inv / half_h;
        let depth = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt();
        ([nx, ny], depth)
    };

    let mut out = Vec::with_capacity(lines.len() * 6);
    for line in lines {
        let ca = cam_coords(line.a);
        let cb = cam_coords(line.b);
        let (cfa, cfb) = (ca[2], cb[2]);
        if cfa < LINE_NEAR_Z && cfb < LINE_NEAR_Z {
            continue;
        }
        // Near-clip in segment-parameter space on the forward component.
        let (mut t0, mut t1) = (0.0f32, 1.0f32);
        let dz = cfb - cfa;
        if dz.abs() > f32::EPSILON {
            let tn = (LINE_NEAR_Z - cfa) / dz;
            if dz > 0.0 {
                t0 = t0.max(tn);
            } else {
                t1 = t1.min(tn);
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
        let (n0, d0) = project(lerp3(t0));
        let (n1, d1) = project(lerp3(t1));

        // Expand in pixel space for a uniform screen-space thickness.
        let to_px = |n: [f32; 2]| [(n[0] * 0.5 + 0.5) * wf, (0.5 - n[1] * 0.5) * hf];
        let to_ndc = |p: [f32; 2]| [p[0] / wf * 2.0 - 1.0, 1.0 - p[1] / hf * 2.0];
        let p0 = to_px(n0);
        let p1 = to_px(n1);
        let (dx, dy) = (p1[0] - p0[0], p1[1] - p0[1]);
        let len = (dx * dx + dy * dy).sqrt().max(1e-6);
        let half = line.width_px.max(1.0) * 0.5;
        let (ex, ey) = (-dy / len * half, dx / len * half);

        let c0a = to_ndc([p0[0] + ex, p0[1] + ey]);
        let c0b = to_ndc([p0[0] - ex, p0[1] - ey]);
        let c1a = to_ndc([p1[0] + ex, p1[1] + ey]);
        let c1b = to_ndc([p1[0] - ex, p1[1] - ey]);
        let dt = if line.depth_test { 1.0 } else { 0.0 };
        // Mirror the overlay's NDC x to match the flipped scene blit.
        let vert = |pos: [f32; 2], depth: f32| LineVertex {
            pos: [if flip_x { -pos[0] } else { pos[0] }, pos[1]],
            depth,
            depth_test: dt,
            color: line.color,
        };
        // Two triangles, cull disabled so winding is irrelevant.
        out.push(vert(c0a, d0));
        out.push(vert(c0b, d0));
        out.push(vert(c1a, d1));
        out.push(vert(c1a, d1));
        out.push(vert(c0b, d0));
        out.push(vert(c1b, d1));
    }
    out
}

/// A world-space 2D image-sprite quad for [`GpuRenderer::draw_images_deferred`].
/// `corners` are the four world points `TL, TR, BL, BR` (UVs `(0,0) (1,0)
/// (0,1) (1,1)`); `image` indexes a texture uploaded via
/// [`GpuRenderer::upload_image`]; `tint` is straight RGBA in `0..=1`
/// (multiplied into every texel); `depth_test` occludes the quad behind
/// nearer marched geometry. The facade resolves orientation + back-face
/// culling, so this is pure geometry.
#[derive(Clone, Copy, Debug)]
pub struct GpuImageQuad {
    /// The four world-space corner points in `TL, TR, BL, BR` order
    /// (voxel units), mapping to UVs `(0,0) (1,0) (0,1) (1,1)`. Need
    /// not be planar-axis-aligned; the quad is split into two
    /// perspective-correct triangles.
    pub corners: [[f32; 3]; 4],
    /// Texture handle returned by [`GpuRenderer::upload_image`].
    /// Quads referencing an id that was never uploaded are skipped.
    pub image: usize,
    /// Straight RGBA multiplier `0..=1` applied to every texel
    /// (`[1.0; 4]` = untinted); the combined alpha drives the
    /// over-blend.
    pub tint: [f32; 4],
    /// `true` ⇒ fragments behind nearer marched scene geometry are
    /// discarded; `false` ⇒ always drawn on top.
    pub depth_test: bool,
    /// Texels with alpha below this (`0..=1`) are discarded in the FS.
    /// `0.0` keeps the plain over-blend.
    pub alpha_cutoff: f32,
}

/// One expanded textured-quad vertex (`build_image_vertices` output).
/// `ndc` is the projected NDC xy; `w` is the source `forward` depth, fed
/// back into a homogeneous clip position so the rasterizer interpolates
/// `uv` perspective-correctly; `depth` is the euclidean world distance
/// (the marcher's `best_t`) for the manual depth test.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ImageVertex {
    ndc: [f32; 2],
    w: f32,
    depth: f32,
    depth_test: f32,
    cutoff: f32,
    uv: [f32; 2],
    tint: [f32; 4],
}

/// Lazy-built image-sprite pipeline (mirrors [`LineResources`]). The
/// per-draw bind group adds the quad's texture + a sampler to the line
/// pass's uniform + scene-depth bindings.
pub(crate) struct ImageResources {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    uniform_buf: wgpu::Buffer,
    dummy_depth: wgpu::Buffer,
    sampler: wgpu::Sampler,
}

/// A retained image-sprite texture (uploaded via
/// [`GpuRenderer::upload_image`], referenced by [`GpuImageQuad::image`]).
pub(crate) struct ImageResident {
    view: wgpu::TextureView,
    // Held so the view stays valid + the texture shows in profiler dumps.
    _texture: wgpu::Texture,
}

/// Camera-space textured-quad vertex (near-clip working set): the
/// `(right, down, forward)` components + the texture `uv`.
#[derive(Clone, Copy)]
struct ImgClipV {
    cam: [f32; 3],
    uv: [f32; 2],
}

/// Clip a convex camera-space polygon against the near plane
/// (`forward >= LINE_NEAR_Z`), interpolating UVs at each crossing.
fn clip_near_image(poly: &[ImgClipV]) -> Vec<ImgClipV> {
    let n = poly.len();
    let mut out: Vec<ImgClipV> = Vec::with_capacity(n + 1);
    for i in 0..n {
        let cur = poly[i];
        let prev = poly[(i + n - 1) % n];
        let cur_in = cur.cam[2] >= LINE_NEAR_Z;
        let prev_in = prev.cam[2] >= LINE_NEAR_Z;
        if cur_in != prev_in {
            let t = (LINE_NEAR_Z - prev.cam[2]) / (cur.cam[2] - prev.cam[2]);
            out.push(ImgClipV {
                cam: [
                    prev.cam[0] + (cur.cam[0] - prev.cam[0]) * t,
                    prev.cam[1] + (cur.cam[1] - prev.cam[1]) * t,
                    LINE_NEAR_Z,
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

/// Project + near-clip a world-space [`GpuImageQuad`] into perspective-correct
/// textured-quad vertices for `image.wgsl`. Mirrors the scene-DDA pinhole
/// (the same one [`build_line_vertices`] uses), carrying each vertex's
/// euclidean world distance as the depth-test key. Quads fully behind the
/// near plane produce no vertices.
fn build_image_vertices(
    cam: &GpuLineCamera,
    quad: &GpuImageQuad,
    w: u32,
    h: u32,
    fov_y: f32,
    flip_x: bool,
) -> Vec<ImageVertex> {
    let aspect = w as f32 / h as f32;
    let half_h = (fov_y * 0.5).tan();
    let half_w = half_h * aspect;
    let dt = if quad.depth_test { 1.0 } else { 0.0 };

    let cam_coords = |p: [f32; 3]| -> [f32; 3] {
        let d = [p[0] - cam.pos[0], p[1] - cam.pos[1], p[2] - cam.pos[2]];
        [
            cam.right[0] * d[0] + cam.right[1] * d[1] + cam.right[2] * d[2],
            cam.down[0] * d[0] + cam.down[1] * d[1] + cam.down[2] * d[2],
            cam.forward[0] * d[0] + cam.forward[1] * d[1] + cam.forward[2] * d[2],
        ]
    };
    let project = |v: ImgClipV| -> ImageVertex {
        let (cx, cy, cz) = (v.cam[0], v.cam[1], v.cam[2]);
        let nx = cx / (cz * half_w);
        ImageVertex {
            // Mirror NDC x to match the flipped scene blit.
            ndc: [if flip_x { -nx } else { nx }, -cy / (cz * half_h)],
            w: cz,
            depth: (cx * cx + cy * cy + cz * cz).sqrt(),
            depth_test: dt,
            cutoff: quad.alpha_cutoff,
            uv: v.uv,
            tint: quad.tint,
        }
    };

    // Per-corner UV: TL(0,0) TR(1,0) BL(0,1) BR(1,1).
    let uvs = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let verts: Vec<ImgClipV> = quad
        .corners
        .iter()
        .zip(uvs)
        .map(|(c, uv)| ImgClipV {
            cam: cam_coords(*c),
            uv,
        })
        .collect();

    let mut out = Vec::with_capacity(12);
    for tri in [[0usize, 1, 2], [1, 3, 2]] {
        let poly = [verts[tri[0]], verts[tri[1]], verts[tri[2]]];
        let clipped = clip_near_image(&poly);
        if clipped.len() < 3 {
            continue;
        }
        for i in 1..clipped.len() - 1 {
            out.push(project(clipped[0]));
            out.push(project(clipped[i]));
            out.push(project(clipped[i + 1]));
        }
    }
    out
}

impl GpuRenderer {
    /// Draw depth-tested world-space [`GpuLine`]s over the pending frame
    /// (L3.2). Projects each endpoint with `cam` (the marcher's pinhole) +
    /// the last frame's FOV / surface size, expands to screen-space quads,
    /// and runs a `LoadOp::Load` pass into the pending swapchain view — so
    /// the lines land on the marched frame and a later `present` /
    /// `paint_egui` still finishes it (the pending frame is left intact).
    /// Depth-tested lines are occluded by nearer marched geometry (compared
    /// against the scene-DDA depth buffer's `best_t`); call after `render`,
    /// before `present` / `paint_egui`. No-op if no frame is pending.
    pub fn draw_lines_deferred(&mut self, cam: &GpuLineCamera, lines: &[GpuLine]) {
        if self.pending_frame.is_none() || lines.is_empty() {
            return;
        }
        let (w, h) = (self.surface_config.width, self.surface_config.height);
        // RP.0 — project with the render (logical) aspect so the lines align
        // with the upscaled scene; the depth buffer is render-sized too.
        let (rw, rh) = self.render_dims();
        let fov = self.last_fov_y_rad;
        if w == 0 || h == 0 || fov <= 0.0 {
            return; // no frame marched yet — no projection to reuse
        }
        let verts = build_line_vertices(cam, lines, rw, rh, fov, self.flip_x);
        if verts.is_empty() {
            return;
        }
        self.ensure_line_resources();
        let res = self.line_resources.as_ref().expect("just built");

        // Skip the depth test when there's no current scene depth to read —
        // either no buffer at all (sprite-only / never-rendered) or this
        // frame was a color-only clear so the buffer is stale (an empty
        // scene drawn after a grid scene). The 1-word dummy / stale buffer
        // is still bound to satisfy the layout; `no_depth = 1` keeps the
        // shader from indexing it.
        let no_depth = u32::from(self.scene_dda.is_none() || !self.dirty.scene_depth_valid);
        let params = LineParams {
            screen_w: w,
            screen_h: h,
            depth_bias: LINE_DEPTH_BIAS,
            no_depth,
            flip_x: u32::from(self.flip_x),
            depth_w: rw,
            depth_h: rh,
            _pad: 0,
        };
        self.queue
            .write_buffer(&res.uniform_buf, 0, bytemuck::bytes_of(&params));

        // PF.13 (H7-lite) — the bind group depends only on the uniform
        // buffer (stable) and the depth buffer identity, so cache it and
        // rebuild only when the depth buffer is swapped (resize / scene
        // rebuild). wgpu buffers compare by identity.
        let depth_key: Option<wgpu::Buffer> =
            self.scene_dda.as_ref().map(|dda| dda.depth_buffer.clone());
        if !matches!(&self.line_bg_cache, Some((_, key)) if *key == depth_key) {
            let depth_resource = match &self.scene_dda {
                Some(dda) => dda.depth_buffer.as_entire_binding(),
                None => res.dummy_depth.as_entire_binding(),
            };
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("roxlap-gpu line.bg"),
                layout: &res.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: res.uniform_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: depth_resource,
                    },
                ],
            });
            self.line_bg_cache = Some((bg, depth_key));
        }
        let bg = &self.line_bg_cache.as_ref().expect("just ensured").0;

        // Grow-only persistent vertex buffer (L3.3): one `write_buffer`
        // per overlay, reused across frames. Power-of-two capacity keeps
        // re-allocation rare as the segment count drifts.
        let needed = std::mem::size_of_val(verts.as_slice()) as u64;
        if self.line_vbuf_cap < needed {
            let cap = needed.next_power_of_two().max(4096);
            self.line_vbuf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("roxlap-gpu line.vbuf"),
                size: cap,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.line_vbuf_cap = cap;
        }
        let vbuf = self.line_vbuf.as_ref().expect("ensured above");
        self.queue
            .write_buffer(vbuf, 0, bytemuck::cast_slice(&verts));

        let view = &self.pending_frame.as_ref().expect("checked above").1;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("roxlap-gpu lines"),
            });
        {
            // `LoadOp::Load` keeps the marcher's frame; the lines draw over
            // it. Manual depth test in the FS (no depth-stencil attachment).
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("roxlap-gpu line paint"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&res.pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.set_vertex_buffer(0, vbuf.slice(..));
            pass.draw(0..verts.len() as u32, 0..1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        // pending_frame left intact — present/paint_egui finishes the frame.
    }

    /// Lazy-build the [`LineResources`] (`line.wgsl` pipeline + uniform +
    /// dummy depth buffer). The colour target uses the surface format with
    /// straight-alpha over-blending; no depth-stencil attachment (the depth
    /// test is manual in the fragment shader against the scene depth buffer).
    fn ensure_line_resources(&mut self) {
        if self.line_resources.is_some() {
            return;
        }
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("line.wgsl"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/line.wgsl").into()),
            });
        let bgl = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("roxlap-gpu line.bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("roxlap-gpu line.layout"),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });
        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("roxlap-gpu line.pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<LineVertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![
                            0 => Float32x2, // pos (NDC)
                            1 => Float32,   // depth
                            2 => Float32,   // depth_test
                            3 => Float32x4, // color
                        ],
                    }],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: self.surface_config.format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });
        let uniform_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("roxlap-gpu line.uniform"),
            size: std::mem::size_of::<LineParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dummy_depth = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("roxlap-gpu line.dummy_depth"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        self.line_resources = Some(LineResources {
            pipeline,
            bgl,
            uniform_buf,
            dummy_depth,
        });
    }

    /// Upload (or replace) an RGBA8 image as a sampled texture, returning
    /// a stable id for [`GpuImageQuad::image`]. `rgba` is row-major,
    /// `width * height * 4` bytes, straight (un-premultiplied) alpha.
    /// Reuses a dropped slot when one exists. Returns `0` for malformed
    /// input (an id that draws nothing).
    pub fn upload_image(&mut self, rgba: &[u8], width: u32, height: u32) -> usize {
        if width == 0 || height == 0 || rgba.len() != (width as usize) * (height as usize) * 4 {
            return 0;
        }
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("roxlap-gpu image_sprite"),
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
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba,
            wgpu::TexelCopyBufferLayout {
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
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let resident = ImageResident {
            view,
            _texture: texture,
        };
        if let Some(slot) = self.images.iter().position(Option::is_none) {
            self.images[slot] = Some(resident);
            // PF.13 (H7-lite) — the slot now holds a different texture;
            // any cached bind group for the old occupant is stale.
            self.image_bg_cache.remove(&slot);
            slot
        } else {
            self.images.push(Some(resident));
            self.images.len() - 1
        }
    }

    /// Release an image uploaded with [`Self::upload_image`] (the slot
    /// becomes reusable).
    pub fn drop_image(&mut self, id: usize) {
        if let Some(slot) = self.images.get_mut(id) {
            *slot = None;
            self.image_bg_cache.remove(&id);
        }
    }

    /// Draw world-space 2D image sprites ([`GpuImageQuad`]) over the
    /// pending frame — the textured-quad sibling of
    /// [`Self::draw_lines_deferred`]. Projects each quad with `cam` (the
    /// marcher's pinhole) + the last frame's FOV / surface size, expands +
    /// near-clips to triangles, and runs one `LoadOp::Load` pass with a
    /// draw per quad (each binds its own texture). UVs are perspective-correct;
    /// depth-tested quads are occluded by nearer marched geometry. Call
    /// after `render`, before `present` / `paint_egui`. No-op if no frame
    /// is pending.
    pub fn draw_images_deferred(&mut self, cam: &GpuLineCamera, quads: &[GpuImageQuad]) {
        if self.pending_frame.is_none() || quads.is_empty() {
            return;
        }
        let (w, h) = (self.surface_config.width, self.surface_config.height);
        // RP.0 — project with the render (logical) aspect (see
        // `draw_lines_deferred`); depth buffer is render-sized.
        let (rw, rh) = self.render_dims();
        let fov = self.last_fov_y_rad;
        if w == 0 || h == 0 || fov <= 0.0 {
            return;
        }

        // Concatenate every quad's verts into one buffer, recording each
        // quad's (range, texture) so they share a single render pass.
        let mut verts: Vec<ImageVertex> = Vec::new();
        let mut draws: Vec<(u32, u32, usize)> = Vec::new();
        for quad in quads {
            if !matches!(self.images.get(quad.image), Some(Some(_))) {
                continue; // dropped / never-uploaded id
            }
            let v = build_image_vertices(cam, quad, rw, rh, fov, self.flip_x);
            if v.is_empty() {
                continue;
            }
            let start = verts.len() as u32;
            verts.extend_from_slice(&v);
            draws.push((start, verts.len() as u32, quad.image));
        }
        if draws.is_empty() {
            return;
        }

        self.ensure_image_resources();
        // See `draw_lines_deferred`: skip depth when there's no valid
        // current-frame scene depth (none built, or a color-only clear).
        let no_depth = u32::from(self.scene_dda.is_none() || !self.dirty.scene_depth_valid);
        let params = LineParams {
            screen_w: w,
            screen_h: h,
            depth_bias: LINE_DEPTH_BIAS,
            no_depth,
            flip_x: u32::from(self.flip_x),
            depth_w: rw,
            depth_h: rh,
            _pad: 0,
        };
        {
            let res = self.image_resources.as_ref().expect("just built");
            self.queue
                .write_buffer(&res.uniform_buf, 0, bytemuck::bytes_of(&params));
        }

        // Grow-only persistent vertex buffer (mirrors the line vbuf).
        let needed = std::mem::size_of_val(verts.as_slice()) as u64;
        if self.image_vbuf_cap < needed {
            let cap = needed.next_power_of_two().max(4096);
            self.image_vbuf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("roxlap-gpu image.vbuf"),
                size: cap,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.image_vbuf_cap = cap;
        }
        let vbuf = self.image_vbuf.as_ref().expect("ensured above");
        self.queue
            .write_buffer(vbuf, 0, bytemuck::cast_slice(&verts));

        // One bind group per image id (the texture view differs per
        // image). PF.13 (H7-lite) — cached across frames keyed by image
        // id, valid while the depth buffer identity holds; a static HUD
        // costs zero bind-group creations per frame. Entries evict on
        // image drop / slot re-upload.
        let res = self.image_resources.as_ref().expect("just built");
        let depth_key: Option<wgpu::Buffer> =
            self.scene_dda.as_ref().map(|dda| dda.depth_buffer.clone());
        if self.image_bg_depth != depth_key {
            self.image_bg_cache.clear();
            self.image_bg_depth = depth_key;
        }
        let depth_resource = match &self.scene_dda {
            Some(dda) => dda.depth_buffer.as_entire_binding(),
            None => res.dummy_depth.as_entire_binding(),
        };
        for &(_, _, image_id) in &draws {
            if self.image_bg_cache.contains_key(&image_id) {
                continue;
            }
            let resident = self.images[image_id].as_ref().expect("checked present");
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("roxlap-gpu image.bg"),
                layout: &res.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: res.uniform_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: depth_resource.clone(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&resident.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&res.sampler),
                    },
                ],
            });
            self.image_bg_cache.insert(image_id, bg);
        }

        let view = &self.pending_frame.as_ref().expect("checked above").1;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("roxlap-gpu images"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("roxlap-gpu image paint"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&res.pipeline);
            pass.set_vertex_buffer(0, vbuf.slice(..));
            for &(start, end, image_id) in &draws {
                let bg = self.image_bg_cache.get(&image_id).expect("just ensured");
                pass.set_bind_group(0, bg, &[]);
                pass.draw(start..end, 0..1);
            }
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        // pending_frame left intact — present/paint_egui finishes it.
    }

    /// Lazy-build the [`ImageResources`] (`image.wgsl` pipeline + uniform +
    /// nearest sampler + dummy depth). Straight-alpha over-blend, no
    /// depth-stencil attachment (the depth test is manual in the FS).
    fn ensure_image_resources(&mut self) {
        if self.image_resources.is_some() {
            return;
        }
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("image.wgsl"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/image.wgsl").into()),
            });
        let bgl = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("roxlap-gpu image.bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });
        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("roxlap-gpu image.layout"),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });
        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("roxlap-gpu image.pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<ImageVertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![
                            0 => Float32x2, // ndc
                            1 => Float32,   // w
                            2 => Float32,   // depth
                            3 => Float32,   // depth_test
                            4 => Float32,   // cutoff
                            5 => Float32x2, // uv
                            6 => Float32x4, // tint
                        ],
                    }],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: self.surface_config.format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });
        let uniform_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("roxlap-gpu image.uniform"),
            size: std::mem::size_of::<LineParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dummy_depth = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("roxlap-gpu image.dummy_depth"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("roxlap-gpu image.sampler"),
            // Nearest + clamp: pixel-art references want crisp texels and
            // no wrap bleed at the quad edges.
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        self.image_resources = Some(ImageResources {
            pipeline,
            bgl,
            uniform_buf,
            dummy_depth,
            sampler,
        });
    }

    /// Overlay an `egui` UI on the pending frame, then present it
    /// (`hud` feature). `jobs` are the host's tessellated primitives
    /// (`egui::Context::tessellate`), `textures` the per-frame texture
    /// delta from `egui::FullOutput`, `pixels_per_point` the UI scale.
    ///
    /// Draws with `LoadOp::Load` over the marcher's frame (a separate
    /// encoder submitted after the scene's), so the UI composites on top
    /// of the world. No-op if no frame is pending.
    #[cfg(feature = "hud")]
    pub fn paint_egui(
        &mut self,
        jobs: &[egui::ClippedPrimitive],
        textures: &egui::TexturesDelta,
        pixels_per_point: f32,
    ) {
        let Some((surf_tex, surf_view)) = self.pending_frame.take() else {
            return;
        };
        let format = self.surface_config.format;
        let egui_rend = self.egui_renderer.get_or_insert_with(|| {
            egui_wgpu::Renderer::new(
                &self.device,
                format,
                egui_wgpu::RendererOptions {
                    msaa_samples: 1,
                    depth_stencil_format: None,
                    dithering: false,
                    ..Default::default()
                },
            )
        });

        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.surface_config.width, self.surface_config.height],
            pixels_per_point,
        };
        for (id, delta) in &textures.set {
            egui_rend.update_texture(&self.device, &self.queue, *id, delta);
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("roxlap-gpu egui"),
            });
        let user_bufs =
            egui_rend.update_buffers(&self.device, &self.queue, &mut encoder, jobs, &screen);
        {
            // `LoadOp::Load` keeps the marcher's frame; egui draws over it.
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("roxlap-gpu egui paint"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &surf_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                })
                // egui-wgpu 0.29 requires a `'static` pass (see its docs).
                .forget_lifetime();
            egui_rend.render(&mut pass, jobs, &screen);
        }
        for id in &textures.free {
            egui_rend.free_texture(id);
        }
        self.queue.submit(
            user_bufs
                .into_iter()
                .chain(std::iter::once(encoder.finish())),
        );
        surf_tex.present();
    }
}

#[cfg(test)]
mod tests {
    /// A 2×2 world quad centred straight ahead projects to vertices whose
    /// homogeneous `w` equals the camera-forward distance (so the shader's
    /// `clip = ndc·w` recovers perspective-correct UVs) and whose `depth`
    /// is the euclidean range. Verifies geometry without a GPU device.
    #[test]
    fn image_vertices_carry_forward_w_and_euclidean_depth() {
        let cam = crate::GpuLineCamera {
            pos: [0.0, 0.0, 0.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 1.0, 0.0],
            forward: [0.0, 0.0, 1.0],
        };
        // Quad 10 units ahead (forward = +Z), spanning x∈[-1,1], y∈[-1,1].
        let quad = crate::GpuImageQuad {
            corners: [
                [-1.0, -1.0, 10.0], // TL
                [1.0, -1.0, 10.0],  // TR
                [-1.0, 1.0, 10.0],  // BL
                [1.0, 1.0, 10.0],   // BR
            ],
            image: 0,
            tint: [1.0, 1.0, 1.0, 1.0],
            depth_test: true,
            alpha_cutoff: 0.0,
        };
        let verts = super::build_image_vertices(&cam, &quad, 800, 600, 60_f32.to_radians(), false);
        assert_eq!(verts.len(), 6, "two triangles, no near-clip");
        for v in &verts {
            assert!((v.w - 10.0).abs() < 1e-4, "w == forward distance");
            assert!(v.depth >= 10.0, "euclidean depth >= forward distance");
            assert_eq!(v.depth_test, 1.0);
        }
    }
}
