//! WGPU-backed compute-shader renderer scaffold for the roxlap
//! voxel engine. GPU.1 in `PORTING-GPU.md`.
//!
//! GPU.1's job: stand up the device + surface + swapchain on a
//! host window (any [`raw-window-handle`](raw_window_handle)
//! provider), present a clear-to-colour frame each render call,
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
//! # fn pick(w: Arc<Window>, size: (u32, u32)) -> Option<GpuRenderer> {
//! match GpuRenderer::new_blocking(w, size, GpuRendererSettings::default()) {
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
// Headless rendering is a native-only test/bench aid: it blocks on
// `pollster` + `device.poll(Wait)`, neither of which exists on wasm.
#[cfg(not(target_arch = "wasm32"))]
pub mod headless;
pub mod resident;
pub mod scene;
pub mod sprite_model;

pub use camera::Camera;
pub use decompress::{decompress_chunk, ChunkUpload, BEDROCK_RGB, CHUNK_Z};
pub use grid::{bounding_box_of, GpuGridResident, GridUpload};
#[cfg(not(target_arch = "wasm32"))]
pub use headless::HeadlessGpu;
pub use resident::GpuChunkResident;
pub use scene::{
    GpuSceneResident, GridRuntimeTransform, GridStaticMeta, RefreshOutcome, SceneUpload,
};
pub use sprite_model::{
    build_sprite_model, build_sprite_model_with_materials, sprite_model_from_clip_frame,
    sprite_model_from_clip_frame_with_materials, sprite_model_from_voxel_frame,
    sprite_model_from_voxel_frame_with_materials, SpriteInstance, SpriteInstanceTransform,
    SpriteModel, SpriteModelRegistry, SpriteRegistryResident,
};

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

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
/// bound to the host's window. [`GpuRenderer::render`] is the GPU.1
/// clear-to-colour path; [`GpuRenderer::render_chunk`] is GPU.3's
/// single-chunk DDA marcher.
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
    pub a: [f32; 3],
    pub b: [f32; 3],
    pub color: [f32; 4],
    pub width_px: f32,
    pub depth_test: bool,
}

/// World camera basis for projecting [`GpuLine`] endpoints — the same
/// pinhole the scene-DDA pass marches with (`right`/`down`/`forward`
/// orthonormal, `pos` in world voxel units).
#[derive(Clone, Copy, Debug)]
pub struct GpuLineCamera {
    pub pos: [f32; 3],
    pub right: [f32; 3],
    pub down: [f32; 3],
    pub forward: [f32; 3],
}

/// Near plane (camera-forward distance) below which a [`GpuLine`] endpoint
/// is clipped, so the pinhole divide stays finite.
const LINE_NEAR_Z: f32 = 0.0625;
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
struct LineResources {
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
    pub corners: [[f32; 3]; 4],
    pub image: usize,
    pub tint: [f32; 4],
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
struct ImageResources {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    uniform_buf: wgpu::Buffer,
    dummy_depth: wgpu::Buffer,
    sampler: wgpu::Sampler,
}

/// A retained image-sprite texture (uploaded via
/// [`GpuRenderer::upload_image`], referenced by [`GpuImageQuad::image`]).
struct ImageResident {
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

/// RP.2 — flat posterize config for the resolve pass uniform. `levels[c] <= 1`
/// leaves that channel untouched; `dither` is `0`=none, `1`=Bayer4×4,
/// `2`=blue-noise (IGN). Mirror of `roxlap_render::PosterizeConfig`.
#[derive(Clone, Copy, Debug)]
pub struct PosterizeGpu {
    pub levels: [u32; 3],
    pub dither: u32,
}

/// RP.0 — logical render resolution policy for the scene marcher, decoupled
/// from the swapchain size. Mirror of `roxlap_render::RenderResolution` (kept
/// here so `roxlap-gpu` has no upward dependency). See [`GpuRenderer::render_dims`].
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum RenderResolution {
    /// Logical == swapchain. Default; byte-identical to pre-RP rendering.
    #[default]
    Native,
    /// Fixed logical grid, nearest-upscaled to the swapchain.
    Fixed { w: u32, h: u32 },
    /// Logical = `round(swapchain * factor)`, clamped to `>= 1px`.
    Scale(f32),
}

impl RenderResolution {
    /// Resolve to concrete logical pixels given the swapchain (native) size.
    #[must_use]
    fn logical_for(self, native: (u32, u32)) -> (u32, u32) {
        let (nw, nh) = (native.0.max(1), native.1.max(1));
        match self {
            Self::Native => (nw, nh),
            Self::Fixed { w, h } => (w.max(1), h.max(1)),
            Self::Scale(f) => {
                let s = f.max(1e-3);
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let lw = ((nw as f32) * s).round() as u32;
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let lh = ((nh as f32) * s).round() as u32;
                (lw.max(1), lh.max(1))
            }
        }
    }
}

#[allow(clippy::struct_excessive_bools)] // independent per-frame flags, not a state enum
pub struct GpuRenderer {
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_info: String,
    /// Whether the adapter is a low-power device (integrated / software)
    /// rather than a discrete GPU — hosts use this to pick lighter
    /// render-resolution defaults. See [`Self::low_power`].
    low_power: bool,
    clear_colour: [f64; 3],
    frame_count: u32,
    /// Mirror the marched scene horizontally on present (the scene blit
    /// samples `width-1-x`, and line/image overlays mirror their NDC x).
    /// The egui pass is unaffected. See [`Self::set_flip_x`].
    flip_x: bool,
    /// RP.0 — logical render resolution. The scene/sprite passes march at
    /// [`Self::render_dims`] (≤ the swapchain under a fixed value) into a
    /// render-sized framebuffer + depth buffer; the blit nearest-upscales it
    /// to the swapchain. `Native` keeps `render_dims == swapchain` ⇒ the
    /// pre-RP straight blit, byte-identical.
    render_res: RenderResolution,
    /// RP.1 — supersampling factor. `1` = off (march at logical size). `>1`
    /// marches at `logical × ssaa` into the framebuffer/depth and a resolve
    /// compute pass box-downfilters back to logical before the blit.
    ssaa: u32,
    /// RP.2 — reduced-palette post applied in the resolve pass (at logical
    /// resolution). `None` = off (`levels = [1,1,1]` ⇒ the RP.1 box-avg only).
    posterize: Option<PosterizeGpu>,
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
    /// TV.6 — global voxel-material palette mirrored to the scene pass (256
    /// entries, default all-opaque), set via [`Self::set_scene_materials`].
    scene_materials: Box<[MaterialGpu; 256]>,
    /// TV.6 — terrain colour→material map (`[rgb, material_id]` rows) +
    /// whether any mapped material is translucent (the shader gate).
    scene_terrain_map: Vec<[u32; 2]>,
    scene_terrain_translucent: bool,
    /// Whether the *current* deferred frame ran a scene pass that wrote
    /// `scene_dda.depth_buffer`. [`Self::render_scene`] sets it; the
    /// color-only [`Self::render_clear_deferred`] clears it. Without this,
    /// depth-tested overlays (`draw_lines_deferred` / `draw_image`) drawn
    /// over an empty/cleared scene would test against the *previous*
    /// scene's stale depth and clip incorrectly.
    scene_depth_valid: bool,
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
    /// TV — global voxel-material palette mirrored to the sprite pass (256
    /// entries, default all-opaque), set via [`Self::set_sprite_materials`].
    /// `sprite_has_translucent` gates the shader's accumulate path.
    sprite_materials: Box<[MaterialGpu; 256]>,
    sprite_has_translucent: bool,
    /// XS.4 — whether this device grants enough storage buffers per shader
    /// stage for GPU sprite shadows (the cross-pass occupancy bindings push a
    /// pass past the baseline 16). `false` ⇒ GPU sprites render unshadowed (the
    /// pre-XS.4 path); the CPU backend always has sprite shadows. Computed once
    /// at init from the granted device limits (see
    /// [`SPRITE_SHADOW_MIN_STORAGE_BUFFERS`]).
    sprite_shadows_capable: bool,
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
    /// Per-face grid side-shades (voxlap setsideshades), packed for the
    /// scene-DDA uniform: `[0]=(top,bot,left,right)`, `[1]=(up,down,_,_)`.
    /// Each is the u8 shade intensity. `[[0;4];2]` = no shading. Set via
    /// [`Self::set_scene_side_shades`].
    scene_side_shades: [[i32; 4]; 2],
    /// DL — per-frame dynamic lights (sun + point lights), already
    /// transformed into each grid's local frame by the facade. Set via
    /// [`Self::set_scene_lights`]; [`SceneLights::default`] = no lights
    /// (the pre-DL render). Consumed by `render_scene` each frame.
    scene_lights: SceneLights,
    /// PF.5 — set when [`Self::set_scene_lights`] stores a DIFFERENT rig;
    /// `render_scene` re-packs + re-uploads the scene point lights only
    /// then (a static rig costs nothing per frame). Starts `true`.
    scene_lights_dirty: bool,
    /// PF.5 — like `scene_lights_dirty` but cleared by the SPRITE pass's
    /// world-light upload (which only runs when sprites are visible, so it
    /// needs its own flag or a lights change while no sprite is on screen
    /// would be lost).
    sprite_lights_dirty: bool,
    /// PF.5 — cached results of the last `pack_scene_lights` (they feed the
    /// per-frame uniform even on pack-skipped frames).
    lights_sun_flags: u32,
    lights_point_count: u32,
    /// PF.5 — grid count the lights were last packed for (the grid-major
    /// rows depend on it, so a grid-count change forces a re-pack).
    lights_packed_grids: u32,
    /// Vertical FOV (radians) the last `render_scene` marched with —
    /// cached so [`Self::pixel_ray`] reconstructs the matching view ray
    /// for picking. `0` until the first scene render.
    last_fov_y_rad: f32,
    /// The acquired-but-not-yet-presented swapchain frame from the most
    /// recent deferred render ([`Self::render_scene`] /
    /// [`Self::render_clear_deferred`]). [`Self::present`] shows it as
    /// is; [`Self::paint_egui`] overlays egui first. Lets a host slot a
    /// UI pass between the marcher and present. `None` between present
    /// and the next render.
    pending_frame: Option<(wgpu::SurfaceTexture, wgpu::TextureView)>,
    /// PF.4 — persistent per-frame camera/light buffers + cached scene and
    /// sprite bind groups. Lazily built on the first `render_scene`.
    frame_pack: Option<FramePackBuffers>,
    /// Lazy-built debug-line pipeline (L3.2) — built on the first
    /// [`Self::draw_lines_deferred`] call.
    line_resources: Option<LineResources>,
    /// Persistent debug-line vertex buffer (L3.3) — grown on demand and
    /// reused across frames so a per-frame overlay (hundreds of segments)
    /// costs one `write_buffer`, not a fresh allocation. `line_vbuf_cap`
    /// is its capacity in bytes.
    line_vbuf: Option<wgpu::Buffer>,
    line_vbuf_cap: u64,
    /// PF.13 (H7-lite) — cached line-overlay bind group + the scene
    /// depth buffer it was built against (`None` = the dummy depth).
    /// Rebuilt only when that identity changes (resize / scene swap)
    /// instead of every `draw_lines_deferred` call.
    line_bg_cache: Option<(wgpu::BindGroup, Option<wgpu::Buffer>)>,
    /// Lazy-built image-sprite pipeline — built on the first
    /// [`Self::draw_images_deferred`] call.
    image_resources: Option<ImageResources>,
    /// Persistent image-sprite vertex buffer, grown on demand and reused
    /// across frames (like [`Self::line_vbuf`]).
    image_vbuf: Option<wgpu::Buffer>,
    image_vbuf_cap: u64,
    /// PF.13 (H7-lite) — image-overlay bind groups keyed by image id,
    /// valid only while the depth-buffer identity in
    /// [`image_bg_depth`](Self::image_bg_depth) holds. Entries are
    /// evicted on image drop / slot re-upload; the whole map clears
    /// when the depth buffer is swapped.
    image_bg_cache: std::collections::HashMap<usize, wgpu::BindGroup>,
    image_bg_depth: Option<wgpu::Buffer>,
    /// Retained image-sprite textures, indexed by the id
    /// [`Self::upload_image`] returns. A dropped slot is `None` and is
    /// re-used by a later upload.
    images: Vec<Option<ImageResident>>,
    /// Lazy-built `egui-wgpu` paint pipeline; created on the first
    /// [`Self::paint_egui`] call (`hud` feature).
    #[cfg(feature = "hud")]
    egui_renderer: Option<egui_wgpu::Renderer>,
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
    /// RP.1 — the **march** framebuffer size (`logical × ssaa`); the scene +
    /// sprite + depth passes run at this. Used for the rebuild check.
    storage_size: (u32, u32),
    /// RP.1 — the **logical** (resolved) size: `resolve_buf` + the blit src.
    logical_size: (u32, u32),
    /// QE.7a - retained so `read_frame_pixels` (capture) can stage it;
    /// the resolve/blit bind groups hold their own references.
    resolve_buf: wgpu::Buffer,
    /// Framebuffer as a packed-`rgba8unorm` storage **buffer** (row
    /// stride = march width), written by the scene + sprite compute passes
    /// and read by the resolve pass. A buffer (not a storage texture) dodges
    /// Chrome-Dawn's tiled write-texture layout (which produced a
    /// 128×256-tiled image); linear + explicit stride is portable.
    framebuffer: wgpu::Buffer,
    uniform_buf: wgpu::Buffer,
    bgl_dda: wgpu::BindGroupLayout,
    pipeline_dda: wgpu::ComputePipeline,
    /// RP.1/RP.2 — box-downfilter + posterize compute pass
    /// (`scene_resolve.wgsl`): framebuffer(march) → resolve_buf(logical). The
    /// bind group retains the resolve buffer (not stored separately).
    pipeline_resolve: wgpu::ComputePipeline,
    resolve_bg: wgpu::BindGroup,
    /// Resolve uniform `[src w,h, dst w,h, ssaa, levels r,g,b, dither, pad×3]`.
    /// Retained so the posterize fields are re-written per frame (RP.2).
    resolve_dims: wgpu::Buffer,
    /// Blit bind group — binds `resolve_buf` (logical) + `blit_dims`.
    blit_bg: wgpu::BindGroup,
    /// PF.5 (H6) — blit variant reading `framebuffer` directly, used when
    /// the resolve pass would be an identity copy (ssaa 1, posterize off).
    blit_bg_direct: wgpu::BindGroup,
    pipeline_blit: wgpu::RenderPipeline,
    /// Blit uniform `Dims`: `[src(logical) w,h, dst(swapchain) w,h, flip_x,
    /// pad×3]`. Retained so the flip flag (offset 16) is re-written per frame.
    blit_dims: wgpu::Buffer,
    /// GPU.9 — per-pixel world-t depth (f32 bits as u32), sized
    /// `width * height * 4`. The scene pass writes it when sprites
    /// are present; the sprite model-DDA pass reads + composites
    /// against it.
    depth_buffer: wgpu::Buffer,
    /// Picking — a `COPY_DST | MAP_READ` staging copy of `depth_buffer`
    /// so the host can read back the per-pixel world-t after a frame
    /// (e.g. click → which voxel). Same size as `depth_buffer`.
    depth_readback: wgpu::Buffer,
    /// TV.6 — global voxel-material palette (256 `MaterialGpu`, binding 16),
    /// seeded from `scene_materials`, rewritten by [`GpuRenderer::set_scene_materials`].
    materials_pal_buf: wgpu::Buffer,
    /// TV.6 — terrain colour→material map (`[rgb, material_id]` rows, binding
    /// 17); ≥1 element (wgpu rejects a zero-sized storage binding).
    terrain_map_buf: wgpu::Buffer,
    /// XS.4.3 — placeholder bound at the sprite-cast bindings (19..21) on a
    /// capable device when no sprite registry exists (or this frame has no
    /// sprites). `sprite_cast_count == 0` keeps the shader from indexing it.
    /// `None` on non-capable devices (those bindings aren't in the BGL).
    sprite_cast_dummy: Option<wgpu::Buffer>,
}

/// PF.4 — persistent per-frame pack state for `render_scene`: the per-grid
/// camera + point-light storage buffers (previously `create_buffer_init`-ed
/// EVERY frame, which also forced rebuilding the 22/23-entry bind groups
/// every frame) plus the cached bind groups themselves.
///
/// Buffers are grow-only (pow2, like `line_vbuf`) with `COPY_DST`, updated
/// via `queue.write_buffer`; wgpu zero-initialises fresh buffers, so the
/// empty-scene "one zeroed element" padding of the old path is implicit.
/// The shaders only index `0..grid_count` / `0..count*grid_count`, so stale
/// bytes past the current write are never read.
///
/// Bind groups are cached against the exact resources they bound (wgpu 23+
/// resources compare by identity): any regrow, scene-resident swap,
/// `scene_dda` rebuild, sky replacement, or sprite-registry buffer growth
/// changes some handle and misses the cache — no manual event tracking.
struct FramePackBuffers {
    grid_cameras: wgpu::Buffer,
    grid_cameras_cap: u64,
    point_lights: wgpu::Buffer,
    point_lights_cap: u64,
    /// World-space lights for the sprite pass (binding 15 there).
    sprite_lights: wgpu::Buffer,
    sprite_lights_cap: u64,
    dda_bg: Option<CachedBindGroup>,
    sprite_bg: Option<CachedBindGroup>,
}

/// A cached bind group plus the exact resources it bound, in binding order.
/// Cheap to compare (identity) and to clone (refcounts).
struct CachedBindGroup {
    bufs: Vec<(u32, wgpu::Buffer)>,
    views: Vec<(u32, wgpu::TextureView)>,
    bg: wgpu::BindGroup,
}

impl FramePackBuffers {
    fn new(device: &wgpu::Device) -> Self {
        let mk = |label: &str, cap: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: cap,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        // Seed capacities: a few grids' cameras / a few dozen lights — most
        // scenes never regrow past these.
        let cam_cap = 4 * 144;
        let light_cap = 4096;
        Self {
            grid_cameras: mk("roxlap-gpu scene_dda.grid_cameras", cam_cap),
            grid_cameras_cap: cam_cap,
            point_lights: mk("roxlap-gpu scene_dda.grid_point_lights", light_cap),
            point_lights_cap: light_cap,
            sprite_lights: mk("roxlap-gpu sprite_model_dda.point_lights", light_cap),
            sprite_lights_cap: light_cap,
            dda_bg: None,
            sprite_bg: None,
        }
    }

    /// Write `bytes` into the selected persistent buffer, regrowing (pow2)
    /// when capacity is exceeded. Regrowth replaces the buffer handle, which
    /// the bind-group cache detects by identity on its next lookup.
    fn write_grow(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buf: &mut wgpu::Buffer,
        cap: &mut u64,
        label: &str,
        bytes: &[u8],
    ) {
        let needed = bytes.len() as u64;
        if needed > *cap {
            let new_cap = needed.next_power_of_two();
            *buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: new_cap,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            *cap = new_cap;
        }
        if !bytes.is_empty() {
            queue.write_buffer(buf, 0, bytes);
        }
    }

    fn write_cameras(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cams: &[SceneDdaPerGridCamera],
    ) {
        Self::write_grow(
            device,
            queue,
            &mut self.grid_cameras,
            &mut self.grid_cameras_cap,
            "roxlap-gpu scene_dda.grid_cameras",
            bytemuck::cast_slice(cams),
        );
    }

    fn write_point_lights(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        lights: &[GpuPointLight],
    ) {
        Self::write_grow(
            device,
            queue,
            &mut self.point_lights,
            &mut self.point_lights_cap,
            "roxlap-gpu scene_dda.grid_point_lights",
            bytemuck::cast_slice(lights),
        );
    }

    fn write_sprite_lights(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        lights: &[GpuPointLight],
    ) {
        Self::write_grow(
            device,
            queue,
            &mut self.sprite_lights,
            &mut self.sprite_lights_cap,
            "roxlap-gpu sprite_model_dda.point_lights",
            bytemuck::cast_slice(lights),
        );
    }
}

/// PF.4 — return the cached bind group when it bound exactly `bufs` +
/// `views` (identity compare), else build + cache a fresh one.
/// `samplers` are bound but NOT part of the key: every sampler we bind
/// (`sky_sampler`) is created once at init and never replaced
/// (`set_sky_panorama` swaps the texture + view only).
fn cached_bind_group<'a>(
    slot: &'a mut Option<CachedBindGroup>,
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    bufs: Vec<(u32, wgpu::Buffer)>,
    views: Vec<(u32, wgpu::TextureView)>,
    samplers: &[(u32, &wgpu::Sampler)],
) -> &'a wgpu::BindGroup {
    let hit = slot
        .as_ref()
        .is_some_and(|c| c.bufs == bufs && c.views == views);
    if !hit {
        let mut entries: Vec<wgpu::BindGroupEntry> = bufs
            .iter()
            .map(|(binding, b)| wgpu::BindGroupEntry {
                binding: *binding,
                resource: b.as_entire_binding(),
            })
            .collect();
        entries.extend(views.iter().map(|(binding, v)| wgpu::BindGroupEntry {
            binding: *binding,
            resource: wgpu::BindingResource::TextureView(v),
        }));
        entries.extend(samplers.iter().map(|&(binding, s)| wgpu::BindGroupEntry {
            binding,
            resource: wgpu::BindingResource::Sampler(s),
        }));
        entries.sort_by_key(|e| e.binding);
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &entries,
        });
        *slot = Some(CachedBindGroup { bufs, views, bg });
    }
    &slot.as_ref().expect("just cached").bg
}

/// GPU.10.0 — single-sprite model-DDA pipeline: one thread per pixel
/// marches the model voxel volume and composites against the scene
/// depth buffer.
struct SpriteModelDdaResources {
    bgl: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    uniform_buf: wgpu::Buffer,
    /// TV — global voxel-material palette (256 `MaterialGpu`, binding 12),
    /// seeded from the renderer's `sprite_materials` and rewritten by
    /// [`GpuRenderer::set_sprite_materials`].
    materials_buf: wgpu::Buffer,
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
    /// TV — 1 if any palette material is translucent: gates the shader's
    /// accumulate path. 0 ⇒ the unchanged nearest-hit opaque path.
    has_translucent: u32,
    // ── DL.4 — dynamic lighting for sprites (world space; all-zero ⇒
    // unchanged flat-lit sprites). No sprite shadows (deferred). ──
    /// World-space unit direction TO the sun (xyz; w unused).
    sun_dir: [f32; 4],
    /// `rgb` = sun colour, `w` = sun intensity.
    sun_color: [f32; 4],
    /// `rgb` = ambient multiplier on the sprite's albedo, `w` unused.
    ambient_color: [f32; 4],
    /// bit0 = sun enabled, bit2 = dynamic lighting active (use the lit path).
    sun_flags: u32,
    point_light_count: u32,
    _pad_dl: [u32; 2],
    // ── DL.6 — stylized sprite lighting (cel + ramp + flat per voxel) ──
    /// `rgb` = cool unlit end of the sun ramp; `w` unused.
    shadow_tint: [f32; 4],
    /// Cel band count; 0 = smooth.
    style_bands: u32,
    // ── XS.4.2 — GPU sprite-shadow (receive) params. Mirror the scene pass's
    // paging + shadow uniform fields so the sprite pass's duplicated terrain
    // occupancy march reads the exact same ABI. All zero ⇒ no sprite shadows
    // (the capability fallback / pre-XS.4 path). ──
    occ_num_pages: u32,
    occ_page_words: u32,
    grid_count: u32,
    max_outer_steps: u32,
    shadow_max_steps: u32,
    shadow_bias: f32,
    shadow_max_dist: f32,
    /// Fraction of a caster's light removed in shadow (`in_shadow = 1 - this`).
    shadow_strength: f32,
    _pad_xs: [u32; 3],
}

/// GPU.10.3 — sprite screen-tile edge in pixels for instance binning.
const SPRITE_TILE_SIZE: u32 = 16;

/// One material in the GPU sprite material palette (binding 12). Mirrors
/// `Mat` in `sprite_model_dda.wgsl` (std430, 8 bytes). TV stage.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MaterialGpu {
    /// Opacity / additive intensity, normalised to `0..=1`.
    alpha: f32,
    /// [`roxlap_formats::material::BlendMode`] discriminant.
    mode: u32,
}

/// Convert the global [`MaterialTable`](roxlap_formats::material::MaterialTable)
/// into the GPU palette + a flag of whether any material is non-opaque (the
/// shader gate — an all-opaque palette runs the unchanged first-hit path).
fn material_palette(
    table: &roxlap_formats::material::MaterialTable,
) -> (Box<[MaterialGpu; 256]>, bool) {
    let mut out = Box::new(
        [MaterialGpu {
            alpha: 1.0,
            mode: 0,
        }; 256],
    );
    let mut any_translucent = false;
    for (id, slot) in out.iter_mut().enumerate() {
        let m = table.get(id as u8);
        slot.alpha = f32::from(m.alpha) / 255.0;
        slot.mode = u32::from(m.mode.as_u8());
        if !m.is_opaque() {
            any_translucent = true;
        }
    }
    (out, any_translucent)
}

// ───────────────────────── DL — dynamic lighting ─────────────────────────
// Stage DL (GPU-only). The scene-DDA pass gains a runtime sun + point
// lights + stylized hard shadows. The host passes lights already
// transformed into each grid's local frame (mirroring the per-grid
// cameras); the shader works entirely in grid-local space. DL.0 wires the
// buffers + uniform fields + bindings; the shader receives them but does
// not yet read them (the hit-site shading lands in DL.1+).

/// Max point lights honoured per frame. Excess are dropped with a warning
/// (never silently truncated). The per-grid buffer is sized
/// `grid_count * point_count`.
pub const MAX_POINT_LIGHTS: usize = 32;
/// Max simultaneous shadow casters (the sun counts as one). Lights flagged
/// to cast beyond this are demoted to shadowless with a warning. Enforced
/// in DL.3 (shadow stage); declared here so the budget is one constant.
pub const MAX_SHADOW_CASTERS: usize = 4;

/// A point light in a grid's **local** space, as handed to
/// [`GpuRenderer::set_scene_lights`]. The facade transforms world-space
/// `roxlap_render::PointLight`s into each grid's frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuLight {
    /// Grid-local position (voxel units).
    pub position: [f32; 3],
    /// Hard cutoff distance, world/voxel units.
    pub radius: f32,
    /// Linear RGB, `0..1`.
    pub color: [f32; 3],
    pub intensity: f32,
    pub casts_shadow: bool,
    /// SL — spot (cone) axis: unit direction the light shines **along**,
    /// in the same frame as [`Self::position`] (grid-local for the scene
    /// pass, world for the sprite pass). Ignored when [`Self::cos_outer`]
    /// is `-1.0`.
    pub spot_dir: [f32; 3],
    /// SL — cosine of the inner cone half-angle (full brightness within it).
    pub cos_inner: f32,
    /// SL — cosine of the outer cone half-angle (zero past it; soft between).
    /// `-1.0` (180° cone) ⇒ a pure point light: the cone mask is skipped.
    pub cos_outer: f32,
}

/// The whole per-frame light environment, already transformed per grid.
/// `grid_sun_dirs` and `grid_point_lights` are indexed by grid (outer
/// length == `grid_count`); empty ⇒ that light type is off. Set each frame
/// via [`GpuRenderer::set_scene_lights`]; [`Default`] = no lights (the
/// pre-DL render).
#[derive(Clone, Default, PartialEq)]
pub struct SceneLights {
    /// Whether a dynamic-lighting rig is active this frame. `false` (the
    /// default) ⇒ the shader takes the unchanged baked-only path
    /// (byte-identical to pre-DL). `true` ⇒ the lit path runs (ambient
    /// term + sun + point lights), even with no sun/points set, so the
    /// `ambient` multiplier still applies.
    pub enabled: bool,
    /// Per-grid unit direction **to** the sun (grid-local). Empty ⇒ no sun.
    pub grid_sun_dirs: Vec<[f32; 3]>,
    pub sun_color: [f32; 3],
    pub sun_intensity: f32,
    pub sun_casts_shadow: bool,
    /// Per-grid point lights (grid-local). Outer len == `grid_count`; the
    /// inner len (the point count) is the same for every grid.
    pub grid_point_lights: Vec<Vec<GpuLight>>,
    /// Multiplier on the baked ambient byte.
    pub ambient: [f32; 3],
    pub shadow_strength: f32,
    pub shadow_bias: f32,
    pub shadow_max_dist: f32,
    pub shadow_max_steps: u32,
    /// DL.4 — **world-space** unit direction to the sun, for the sprite
    /// pass (sprites render in world space, not grid-local). `[0;3]` ⇒ no
    /// sun. Empty `grid_sun_dirs` and a zero `world_sun_dir` both mean
    /// "no sun" for their respective passes.
    pub world_sun_dir: [f32; 3],
    /// DL.4 — world-space point lights for the sprite pass (positions in
    /// world coords; same colour/intensity/radius as the per-grid copies).
    pub world_points: Vec<GpuLight>,
    /// DL.6 — stylized cel banding: `0` = smooth, `≥1` = quantize the
    /// diffuse to `bands + 1` levels + gradient-map the sun key.
    pub style_bands: u32,
    /// DL.6 — cool shadow/ambient tint (the stylized ramp's unlit end).
    pub shadow_tint: [f32; 3],
}

/// One point light packed for the GPU (binding 18, std430, 64 bytes —
/// four `vec4<f32>`). Mirrors `PointLight` in `scene_dda.wgsl` and
/// `sprite_model_dda.wgsl`. SL grew this 48→64 for the spot (cone) fields;
/// a `-1.0` `cos_outer` marks a pure point light (cone mask skipped).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuPointLight {
    pos: [f32; 3],
    radius: f32,
    color: [f32; 3],
    intensity: f32,
    // SL — spot (cone) fields. `cos_outer == -1.0` ⇒ omnidirectional point.
    spot_dir: [f32; 3],
    cos_outer: f32,
    cos_inner: f32,
    casts_shadow: u32,
    _pad: [u32; 2],
}

// SL — the std430 stride must stay locked to the WGSL `PointLight` (four
// `vec4<f32>`); a mismatch silently corrupts every light past the first.
const _: () = assert!(core::mem::size_of::<GpuPointLight>() == 64);

/// Build the per-grid point-light storage buffer (binding 18), grid-major:
/// grid `g`'s lights occupy `[g*count .. (g+1)*count]`. Pads to one zeroed
/// element when empty (wgpu rejects a zero-sized storage binding).
fn upload_grid_point_lights(device: &wgpu::Device, lights: &[GpuPointLight]) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    let one = [GpuPointLight::zeroed()];
    let src: &[GpuPointLight] = if lights.is_empty() { &one } else { lights };
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("roxlap-gpu scene_dda.grid_point_lights"),
        contents: bytemuck::cast_slice(src),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

/// DL — inject each grid's sun direction into `cam_vec[g].sun_dir`
/// (binding 15). PF.5 — split from [`pack_scene_lights`]: the camera
/// vector is rebuilt every frame, so the injection must run every frame,
/// while the point-light pack is gated behind the lights dirty flag.
fn inject_grid_sun_dirs(lights: &SceneLights, cam_vec: &mut [SceneDdaPerGridCamera]) {
    if lights.grid_sun_dirs.is_empty() {
        return;
    }
    for (g, cam) in cam_vec.iter_mut().enumerate() {
        let d = lights.grid_sun_dirs.get(g).copied().unwrap_or([0.0; 3]);
        cam.sun_dir = [d[0], d[1], d[2], 0.0];
    }
}

/// DL — pack `lights` for the scene-DDA pass, shared by the surface and
/// headless paths: the grid-major point-light rows (binding 18), returned
/// as `(packed_lights, sun_flags, point_count)`. PF.4 — packing only; the
/// caller uploads (the surface path into a persistent buffer, headless via
/// `create_buffer_init`). PF.5 — the surface path calls this only when the
/// lights changed (so the over-cap warnings below also fire once per
/// change, not per frame). `sun_flags`: bit0 = sun enabled, bit1 = sun
/// casts shadow, bit2 = dynamic lighting active. Over-cap point lights are
/// dropped with a warning (never silently truncated).
fn pack_scene_lights(lights: &SceneLights, grid_count: usize) -> (Vec<GpuPointLight>, u32, u32) {
    let sun_enabled = !lights.grid_sun_dirs.is_empty();
    // Point-light count per grid (same across grids); capped + warned.
    let mut point_count = lights
        .grid_point_lights
        .first()
        .map_or(0, std::vec::Vec::len);
    if point_count > MAX_POINT_LIGHTS {
        eprintln!(
            "roxlap-gpu: {point_count} point lights > MAX_POINT_LIGHTS ({MAX_POINT_LIGHTS}); dropping the excess"
        );
        point_count = MAX_POINT_LIGHTS;
    }
    // MAX_SHADOW_CASTERS cap (locked decision #5): the sun (if it casts) is
    // the first caster; keep at most MAX_SHADOW_CASTERS shadow casters total
    // and demote the rest to shadowless — never silently. The point list is
    // identical across grids (only positions differ), so decide per index
    // once from the representative (grid-0) row.
    let mut budget = MAX_SHADOW_CASTERS;
    if sun_enabled && lights.sun_casts_shadow {
        budget = budget.saturating_sub(1);
    }
    let mut allow_shadow = vec![false; point_count];
    let mut demoted = 0usize;
    if let Some(rep) = lights.grid_point_lights.first() {
        for (i, slot) in allow_shadow.iter_mut().enumerate() {
            if rep.get(i).is_some_and(|l| l.casts_shadow) {
                if budget > 0 {
                    *slot = true;
                    budget -= 1;
                } else {
                    demoted += 1;
                }
            }
        }
    }
    if demoted > 0 {
        eprintln!(
            "roxlap-gpu: {demoted} shadow-casting point lights > MAX_SHADOW_CASTERS ({MAX_SHADOW_CASTERS}); demoting the excess to shadowless"
        );
    }
    // Grid-major point-light buffer: grid g at [g*count .. (g+1)*count].
    let mut packed: Vec<GpuPointLight> = Vec::with_capacity(grid_count * point_count);
    for g in 0..grid_count {
        let row = lights.grid_point_lights.get(g);
        for (i, &allow) in allow_shadow.iter().enumerate() {
            let p = row.and_then(|r| r.get(i));
            packed.push(p.map_or(GpuPointLight::zeroed(), |l| GpuPointLight {
                pos: l.position,
                radius: l.radius,
                color: l.color,
                intensity: l.intensity,
                spot_dir: l.spot_dir,
                cos_outer: l.cos_outer,
                cos_inner: l.cos_inner,
                casts_shadow: u32::from(l.casts_shadow && allow),
                _pad: [0; 2],
            }));
        }
    }
    let sun_flags = u32::from(sun_enabled)
        | (u32::from(sun_enabled && lights.sun_casts_shadow) << 1)
        | (u32::from(lights.enabled) << 2);
    (packed, sun_flags, point_count as u32)
}

/// Build the per-grid camera storage buffer bound at `scene_dda.wgsl`
/// binding 15 (read-only). One [`SceneDdaPerGridCamera`] per grid; the
/// shader only indexes `0..grid_count`. An empty scene pads to one
/// zeroed element (wgpu rejects a zero-sized storage binding). This
/// replaces the old fixed `[…; 16]` uniform array, so a scene can hold
/// any number of grids — the only ceiling is the device's storage size.
fn upload_grid_cameras(device: &wgpu::Device, cams: &[SceneDdaPerGridCamera]) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    let one = [SceneDdaPerGridCamera::zeroed()];
    let src: &[SceneDdaPerGridCamera] = if cams.is_empty() { &one } else { cams };
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("roxlap-gpu scene_dda.grid_cameras"),
        contents: bytemuck::cast_slice(src),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

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
    /// DL — unit direction TO the sun in this grid's local frame (xyz; w
    /// unused). Packed here rather than a separate per-grid storage buffer
    /// because the device's `max_storage_buffers_per_shader_stage` (16) is
    /// already saturated. Zero ⇒ no sun (the uniform's `sun_flags` gates).
    sun_dir: [f32; 4],
    /// XS.3 — this grid's world transform, for cross-grid shadows: a shadow
    /// ray (grid-local in the grid being shaded) is lifted to world space and
    /// tested against every grid. `world_origin` (xyz) is the grid origin;
    /// `rot0/1/2` (xyz) are the local→world rotation columns (world images of
    /// grid-local axes x/y/z). Packed here for the same buffer-limit reason.
    world_origin: [f32; 4],
    rot0: [f32; 4],
    rot1: [f32; 4],
    rot2: [f32; 4],
}

impl SceneDdaPerGridCamera {
    fn from_camera(c: &Camera) -> Self {
        Self {
            pos: c.position,
            _pad0: 0.0,
            right: c.right,
            _pad1: 0.0,
            down: c.down,
            _pad2: 0.0,
            forward: c.forward,
            _pad3: 0.0,
            sun_dir: [0.0; 4],
            // Identity world transform by default; the per-grid build
            // (`grid_cameras`) overwrites it with the grid's real transform.
            world_origin: [0.0; 4],
            rot0: [1.0, 0.0, 0.0, 0.0],
            rot1: [0.0, 1.0, 0.0, 0.0],
            rot2: [0.0, 0.0, 1.0, 0.0],
        }
    }

    /// XS.3 — stamp this grid's world transform (for cross-grid shadows).
    /// `rot_cols[i]` is the world image of grid-local axis `i` (the
    /// local→world rotation's columns).
    fn set_world_transform(&mut self, t: &GridWorldTransform) {
        self.world_origin = [t.origin[0], t.origin[1], t.origin[2], 0.0];
        self.rot0 = [t.rot_cols[0][0], t.rot_cols[0][1], t.rot_cols[0][2], 0.0];
        self.rot1 = [t.rot_cols[1][0], t.rot_cols[1][1], t.rot_cols[1][2], 0.0];
        self.rot2 = [t.rot_cols[2][0], t.rot_cols[2][1], t.rot_cols[2][2], 0.0];
    }
}

/// XS.3 — a grid's world transform for cross-grid shadows: world origin +
/// the local→world rotation columns (`rot_cols[i]` = world image of grid-local
/// axis `i`). Built host-side per frame from the grid's `GridTransform` and
/// handed to `SceneRenderer::render_scene` alongside the per-grid cameras.
#[derive(Clone, Copy)]
pub struct GridWorldTransform {
    pub origin: [f32; 3],
    pub rot_cols: [[f32; 3]; 3],
}

impl Default for GridWorldTransform {
    fn default() -> Self {
        Self {
            origin: [0.0; 3],
            rot_cols: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }
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
    /// TV.6 — `1` if any terrain material is translucent (gates the
    /// accumulate path; `0` ⇒ unchanged opaque first-hit march).
    terrain_has_translucent: u32,
    /// TV.6 — number of `(rgb, material_id)` entries in the terrain map.
    terrain_map_count: u32,
    _pad4: u32,
    /// World camera used only to derive the per-pixel sky direction —
    /// always valid, so a `grid_count == 0` (sprite-only / empty) scene
    /// still paints a proper sky instead of a degenerate `(0,0,1)`
    /// (whose `atan2(0,0)` sky lookup samples black).
    sky_cam: SceneDdaPerGridCamera,
    /// Per-face side-shade intensities (voxlap setsideshades), each the
    /// u8 shade subtracted from a voxel's brightness byte at a hit.
    /// `side_shades0 = (top, bot, left, right)`,
    /// `side_shades1 = (up, down, _, _)`. All-zero = no shading.
    side_shades0: [i32; 4],
    side_shades1: [i32; 4],
    // ── DL — dynamic lighting (appended; all-zero ⇒ pre-DL render) ──
    /// `rgb` = sun colour, `w` = sun intensity.
    sun_color: [f32; 4],
    /// `rgb` = ambient multiplier on the baked byte, `w` = shadow strength.
    ambient_color: [f32; 4],
    /// Bit 0 = sun enabled, bit 1 = sun casts shadow.
    sun_flags: u32,
    /// Number of point lights per grid (rows in the binding-18 buffer).
    point_light_count: u32,
    /// Shadow-ray step budget (DL.3).
    shadow_max_steps: u32,
    _pad5: u32,
    /// Shadow-ray origin bias along the surface normal (voxel units).
    shadow_bias: f32,
    /// Sun shadow-ray length cap (world units).
    shadow_max_dist: f32,
    _pad6: [f32; 2],
    /// DL.6 — stylized ramp's cool shadow tint (rgb; w unused).
    shadow_tint: [f32; 4],
    /// DL.6 — cel band count; 0 = smooth (no banding / gradient map).
    style_bands: u32,
    /// XS.4.3 — visible sprite-instance count for the scene pass's
    /// sprite-cast shadow march (sprites cast onto terrain). `0` ⇒ no sprite
    /// casters (the loop is skipped); only consulted by the capable variant.
    sprite_cast_count: u32,
    _pad7: [u32; 2],
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
    /// `window` is any [`raw-window-handle`] provider (winit, SDL,
    /// GLFW, …) wrapped in an `Arc`; `size` is its initial physical
    /// framebuffer size in pixels — passed explicitly so the renderer
    /// stays decoupled from any one windowing library's size API.
    ///
    /// [`raw-window-handle`]: raw_window_handle
    ///
    /// # Errors
    /// Returns [`GpuInitError`] if surface creation, adapter
    /// selection, or device request fails. Hosts treat any error as
    /// "fall back to the CPU path".
    pub async fn new<W>(
        window: Arc<W>,
        size: (u32, u32),
        settings: GpuRendererSettings,
    ) -> Result<Self, GpuInitError>
    where
        W: HasWindowHandle + HasDisplayHandle + Send + Sync + 'static,
    {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let surface = instance.create_surface(window.clone())?;
        let adapter = Self::request_adapter(&instance, Some(&surface), settings).await?;
        let (device, queue) = Self::request_device(&adapter).await?;
        Ok(Self::finish_init(
            &adapter, device, queue, surface, size, settings,
        ))
    }

    /// wasm/WebGPU: build the renderer against an HTML `canvas`. No
    /// `Send + Sync` bound — wgpu's surface/device/queue are `!Send` on
    /// the `+atomics` shared-memory wasm build, and the browser host is
    /// single-threaded (`Rc<RefCell<…>>`). The native generic-`W` entry
    /// (which carries the bound) isn't reachable on wasm.
    ///
    /// Probes for an adapter **before** `create_surface`: on wasm,
    /// creating the surface calls `canvas.getContext("webgpu")`, which
    /// permanently locks the canvas's context type. If we bound it and
    /// then found no adapter, a CPU/WebGL2 fallback on the *same* canvas
    /// (the facade clones the handle, but it's the same DOM element)
    /// would fail with "no webgl2 context". Probing first leaves the
    /// canvas pristine when WebGPU is unavailable.
    ///
    /// # Errors
    /// See [`Self::new`].
    #[cfg(target_arch = "wasm32")]
    pub async fn new_from_canvas(
        canvas: web_sys::HtmlCanvasElement,
        size: (u32, u32),
        settings: GpuRendererSettings,
    ) -> Result<Self, GpuInitError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        // Probe adapter AND device before binding the canvas — both
        // `requestAdapter` and `requestDevice` can fail on wasm, and
        // `create_surface` permanently locks the canvas to a WebGPU
        // context. Creating the surface last keeps the canvas pristine
        // for the CPU/WebGL2 fallback on any GPU-init failure.
        let adapter = Self::request_adapter(&instance, None, settings).await?;
        let (device, queue) = Self::request_device(&adapter).await?;
        let surface = instance.create_surface(wgpu::SurfaceTarget::Canvas(canvas))?;
        Ok(Self::finish_init(
            &adapter, device, queue, surface, size, settings,
        ))
    }

    /// Pick a GPU adapter at the settings' power preference. `None`
    /// `compatible_surface` is used on the wasm canvas path so the probe
    /// doesn't bind the canvas's context (see [`Self::new_from_canvas`]);
    /// WebGPU exposes a single surface-independent adapter, so this is
    /// safe there.
    async fn request_adapter(
        instance: &wgpu::Instance,
        compatible_surface: Option<&wgpu::Surface<'static>>,
        settings: GpuRendererSettings,
    ) -> Result<wgpu::Adapter, GpuInitError> {
        // `ROXLAP_GPU_POWER=low|high` overrides the host's adapter
        // preference — an escape hatch for broken hybrid-GPU (PRIME)
        // driver stacks, where rendering on the display-owning iGPU
        // avoids the cross-GPU present entirely. (A nixos mesa update
        // deadlocked the nouveau↔i915 explicit-sync fences: dGPU frames
        // hit the drm job timeout and the channel was killed; `low` kept
        // the demo alive.) Init-time only — never read per frame; on
        // wasm `env::var` errs and the settings value applies.
        let power_preference = match std::env::var("ROXLAP_GPU_POWER").as_deref() {
            Ok("low") => wgpu::PowerPreference::LowPower,
            Ok("high") => wgpu::PowerPreference::HighPerformance,
            _ => match settings.power_preference {
                PowerPreference::Low => wgpu::PowerPreference::LowPower,
                PowerPreference::High => wgpu::PowerPreference::HighPerformance,
            },
        };
        instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference,
                compatible_surface,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|_| GpuInitError::NoAdapter)
    }

    /// Request the device + queue from `adapter`. Pulled out of
    /// [`Self::finish_init`] so the wasm canvas path can validate the
    /// device **before** `create_surface` binds the canvas's WebGPU
    /// context — if the device request fails (e.g. a browser that
    /// rejects a wgpu-sent limit), the canvas stays pristine for the
    /// CPU/WebGL2 fallback instead of being poisoned.
    async fn request_device(
        adapter: &wgpu::Adapter,
    ) -> Result<(wgpu::Device, wgpu::Queue), GpuInitError> {
        Ok(adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("roxlap-gpu device"),
                required_features: wgpu::Features::empty(),
                required_limits: pick_required_limits(&adapter.limits()),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .await?)
    }

    /// Shared swapchain → sky/sampler setup, run after the adapter +
    /// device + surface exist (the surface comes from a window handle on
    /// native, or an HTML canvas on wasm — created last on wasm so a
    /// failed device request never touches the canvas).
    fn finish_init(
        adapter: &wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
        surface: wgpu::Surface<'static>,
        size: (u32, u32),
        settings: GpuRendererSettings,
    ) -> Self {
        let info = adapter.get_info();
        let adapter_info = format!(
            "{name} ({backend:?}, {device_type:?})",
            name = info.name,
            backend = info.backend,
            device_type = info.device_type,
        );
        let low_power = info.device_type != wgpu::DeviceType::DiscreteGpu;

        let caps = surface.get_capabilities(adapter);
        // Pick a NON-sRGB, 8-bit swapchain format. Voxlap colours are
        // already sRGB-encoded (the slab bytes are display-ready,
        // matching what the CPU softbuffer path writes straight to the
        // framebuffer with no conversion); an sRGB swapchain would
        // re-apply the gamma curve, washing the look out. We also
        // *prefer 8-bit BGRA/RGBA* over any other non-sRGB format: some
        // adapters (e.g. NVK) advertise a 16-bit-unorm format first,
        // and wgpu 29 gates `create_view` on 16-bit-norm formats behind
        // the `TEXTURE_FORMAT_16BIT_NORM` device feature (which we don't
        // enable, to stay WebGPU-portable). Falls back to the first
        // non-sRGB format, then `caps.formats[0]`.
        let surface_format = caps
            .formats
            .iter()
            .copied()
            .find(|f| {
                matches!(
                    f,
                    wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Rgba8Unorm
                )
            })
            .or_else(|| caps.formats.iter().copied().find(|f| !f.is_srgb()))
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
        let (init_w, init_h) = size;
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: init_w.max(1),
            height: init_h.max(1),
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
            wgpu::TexelCopyTextureInfo {
                texture: &sky_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &default_sky_pixel,
            wgpu::TexelCopyBufferLayout {
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
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        // XS.4 — did the device grant enough storage buffers per stage for the
        // GPU sprite-shadow cross-pass bindings? If not, sprites render
        // unshadowed (the CPU backend still has full sprite shadows).
        let sprite_shadows_capable = device.limits().max_storage_buffers_per_shader_stage
            >= SPRITE_SHADOW_MIN_STORAGE_BUFFERS;

        Self {
            surface,
            surface_config,
            device,
            queue,
            adapter_info,
            low_power,
            clear_colour: settings.clear_colour,
            frame_count: 0,
            flip_x: false,
            render_res: RenderResolution::Native,
            ssaa: 1,
            posterize: None,
            chunk_dda: None,
            grid_dda: None,
            scene_dda: None,
            scene_materials: Box::new(
                [MaterialGpu {
                    alpha: 1.0,
                    mode: 0,
                }; 256],
            ),
            scene_terrain_map: Vec::new(),
            scene_terrain_translucent: false,
            scene_depth_valid: false,
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
            sprite_shadows_capable,
            sprite_materials: Box::new(
                [MaterialGpu {
                    alpha: 1.0,
                    mode: 0,
                }; 256],
            ),
            sprite_has_translucent: false,
            // GPU.10.4 — default LOD threshold: step to a coarser mip
            // once a voxel projects below 4 px. Empirically the best
            // quality/cost tradeoff; the host can override.
            sprite_lod_px: 4.0,
            // GPU.11.1 — matches the CPU demo's mip_scan_dist=64.
            scene_mip_scan_dist: 64.0,
            scene_side_shades: [[0; 4]; 2],
            scene_lights: SceneLights::default(),
            scene_lights_dirty: true,
            sprite_lights_dirty: true,
            lights_sun_flags: 0,
            lights_point_count: 0,
            lights_packed_grids: 0,
            last_fov_y_rad: 0.0,
            pending_frame: None,
            frame_pack: None,
            line_resources: None,
            line_vbuf: None,
            line_vbuf_cap: 0,
            line_bg_cache: None,
            image_resources: None,
            image_vbuf: None,
            image_vbuf_cap: 0,
            image_bg_cache: std::collections::HashMap::new(),
            image_bg_depth: None,
            images: Vec::new(),
            #[cfg(feature = "hud")]
            egui_renderer: None,
        }
    }

    /// Synchronous wrapper for hosts that don't have an async
    /// runtime. Internally `pollster::block_on`s [`Self::new`].
    ///
    /// # Errors
    /// See [`Self::new`].
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new_blocking<W>(
        window: Arc<W>,
        size: (u32, u32),
        settings: GpuRendererSettings,
    ) -> Result<Self, GpuInitError>
    where
        W: HasWindowHandle + HasDisplayHandle + Send + Sync + 'static,
    {
        pollster::block_on(Self::new(window, size, settings))
    }

    /// Human-readable adapter description — name + backend +
    /// device type. The demo host prints this in the title bar.
    pub fn adapter_info(&self) -> &str {
        &self.adapter_info
    }

    /// `true` when the adapter is NOT a discrete GPU (integrated,
    /// software rasterizer, virtual, unknown) — a hint that hosts
    /// should default to a lighter render resolution.
    pub fn low_power(&self) -> bool {
        self.low_power
    }

    /// Borrow the underlying wgpu device — hosts use this to build
    /// chunk uploads (`GpuChunkResident::upload(gpu.device(), …)`).
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// XS.4 — whether this device can run GPU sprite shadows (it granted
    /// enough storage buffers per shader stage for the cross-pass occupancy
    /// bindings). `false` ⇒ GPU sprites render unshadowed; the CPU backend
    /// always has sprite shadows. Lets the facade/host report the fallback.
    #[must_use]
    pub fn sprite_shadows_capable(&self) -> bool {
        self.sprite_shadows_capable
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
    /// Mirror the marched scene (and its line/image overlays) horizontally
    /// on present, leaving the egui overlay upright. See `Self::flip_x`.
    pub fn set_flip_x(&mut self, flip: bool) {
        self.flip_x = flip;
    }

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
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
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

    /// RP.0 — set the logical render resolution. Rebuilds the scene-DDA
    /// resources on the next [`Self::render_scene`] when the render size
    /// changes.
    pub fn set_render_resolution(&mut self, res: RenderResolution) {
        self.render_res = res;
    }

    /// RP.1 — set the supersampling factor (clamped to `1..=4`). `1` = off.
    pub fn set_ssaa(&mut self, factor: u8) {
        self.ssaa = u32::from(factor).clamp(1, 4);
    }

    /// RP.2 — set (or clear) the posterize post. Applied per-frame via the
    /// resolve uniform, so no pipeline rebuild is needed.
    pub fn set_posterize(&mut self, cfg: Option<PosterizeGpu>) {
        self.posterize = cfg;
    }

    /// RP.0 — the logical (retro) grid size the scene resolves to before the
    /// upscale, resolved against the swapchain size. `logical_dims ==
    /// surface_dims` under [`RenderResolution::Native`].
    #[must_use]
    pub fn logical_dims(&self) -> (u32, u32) {
        self.render_res.logical_for(self.surface_dims())
    }

    /// RP.1 — the resolution the scene/sprite passes actually march at:
    /// `logical_dims × ssaa`. The framebuffer + depth buffer are sized to this.
    #[must_use]
    pub fn render_dims(&self) -> (u32, u32) {
        let (lw, lh) = self.logical_dims();
        (lw * self.ssaa, lh * self.ssaa)
    }

    /// RP.0 — the swapchain (native window) size.
    #[must_use]
    pub fn surface_dims(&self) -> (u32, u32) {
        (self.surface_config.width, self.surface_config.height)
    }

    /// Acquire the next swapchain frame, or `None` to skip this frame.
    /// wgpu 29's `get_current_texture` returns a
    /// [`wgpu::CurrentSurfaceTexture`] status enum (was
    /// `Result<_, SurfaceError>`): an outdated/lost surface reconfigures
    /// and skips, transient statuses just skip.
    fn acquire_frame(&self) -> Option<wgpu::SurfaceTexture> {
        use wgpu::CurrentSurfaceTexture as C;
        match self.surface.get_current_texture() {
            C::Success(t) | C::Suboptimal(t) => Some(t),
            C::Outdated | C::Lost => {
                self.surface.configure(&self.device, &self.surface_config);
                None
            }
            C::Timeout | C::Occluded | C::Validation => None,
        }
    }

    /// GPU.1 render: single render pass clearing the swapchain to a
    /// slowly drifting colour, then presenting. Voxels arrive in
    /// GPU.3+.
    pub fn render(&mut self) {
        let Some(surf_tex) = self.acquire_frame() else {
            return;
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
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
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
        let Some(surf_tex) = self.acquire_frame() else {
            return;
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
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
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
                bind_group_layouts: &[Some(&bgl_dda)],
                immediate_size: 0,
            });
        let pipeline_dda = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("roxlap-gpu chunk_dda.pipeline"),
                layout: Some(&dda_pl),
                module: &dda_shader,
                entry_point: Some("render_chunk"),
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
                bind_group_layouts: &[Some(&bgl_blit)],
                immediate_size: 0,
            });
        let pipeline_blit = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("roxlap-gpu chunk_dda.blit_pipeline"),
                layout: Some(&blit_pl),
                vertex: wgpu::VertexState {
                    module: &blit_shader,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &blit_shader,
                    entry_point: Some("fs_main"),
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
                multiview_mask: None,
                cache: None,
            });
        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("roxlap-gpu chunk_dda.blit_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
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
        let Some(surf_tex) = self.acquire_frame() else {
            return;
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
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
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
                bind_group_layouts: &[Some(&bgl_dda)],
                immediate_size: 0,
            });
        let pipeline_dda = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("roxlap-gpu grid_dda.pipeline"),
                layout: Some(&dda_pl),
                module: &dda_shader,
                entry_point: Some("render_grid"),
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
                bind_group_layouts: &[Some(&bgl_blit)],
                immediate_size: 0,
            });
        let pipeline_blit = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("roxlap-gpu grid_dda.blit_pipeline"),
                layout: Some(&blit_pl),
                vertex: wgpu::VertexState {
                    module: &blit_shader,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &blit_shader,
                    entry_point: Some("fs_main"),
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
                multiview_mask: None,
                cache: None,
            });
        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("roxlap-gpu grid_dda.blit_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
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
    /// If `cameras.len() != scene.grid_count`.
    /// `cameras[i]` is grid `i`'s world camera transformed into that
    /// grid's local frame (the grid marcher works in grid-local space).
    /// `sprite_camera` is the **world** camera: instanced sprites carry
    /// world-space positions/transforms, so they must project through
    /// the untransformed world camera — not `cameras[0]`, which is only
    /// the world camera when grid 0 is at identity.
    pub fn render_scene(
        &mut self,
        scene: &GpuSceneResident,
        cameras: &[Camera],
        // XS.3 — per-grid world transforms (parallel to `cameras`) for
        // cross-grid shadows. Empty ⇒ identity (shadows stay intra-grid).
        grid_world: &[GridWorldTransform],
        sprite_camera: &Camera,
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
        self.last_fov_y_rad = fov_y_rad; // cached for pixel_ray (picking)

        // Deferred present: drop any frame a prior render left
        // un-presented (a host that skipped present/paint_egui) so we
        // never hold two outstanding swapchain textures.
        self.pending_frame = None;
        let Some(surf_tex) = self.acquire_frame() else {
            return;
        };
        let surf_view = surf_tex
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let surface_w = self.surface_config.width;
        let surface_h = self.surface_config.height;
        let surface_format = self.surface_config.format;
        // RP.0/RP.1 — the scene + sprite + depth passes march at the *render*
        // size (`logical × ssaa`); a resolve pass box-downfilters to the
        // logical grid; the blit nearest-upscales to the swapchain. The
        // framebuffer/depth/occupancy + per-pixel projection key off the render
        // (march) size. `Native` + `ssaa==1` ⇒ render == logical == surface.
        let (logical_w, logical_h) = self.logical_dims();
        let (render_w, render_h) = self.render_dims();

        let needs_build = match &self.scene_dda {
            Some(r) => {
                r.storage_size != (render_w, render_h) || r.logical_size != (logical_w, logical_h)
            }
            None => true,
        };
        if needs_build {
            self.scene_dda = Some(self.build_scene_dda(
                render_w,
                render_h,
                logical_w,
                logical_h,
                surface_w,
                surface_h,
                surface_format,
            ));
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
            if reg.instance_capacity > 0 {
                // World camera — sprite positions/transforms are world-
                // space (independent of any grid's transform).
                let cam = sprite_camera;
                // Aspect + tile binning are in render (logical) space — the
                // sprite pass writes the render-sized framebuffer/depth.
                #[allow(clippy::cast_precision_loss)]
                let aspect = render_w as f32 / render_h as f32;
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
                    render_w,
                    render_h,
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

        // Refresh the blit's flip flag each frame (offset 16, after the
        // src + dst vec2 sizes), so toggling the flip applies without a
        // resize. The src/dst sizes themselves are written at build time
        // (a render/surface size change forces a rebuild).
        self.queue.write_buffer(
            &dda.blit_dims,
            16,
            bytemuck::bytes_of(&[u32::from(self.flip_x), 0u32]),
        );
        // RP.2 — refresh the resolve pass's posterize fields each frame (offset
        // 20, after src/dst dims + ssaa). `None` ⇒ `levels = [1,1,1]`, `dither
        // = 0` ⇒ the resolve does box-downfilter only (RP.1).
        let (plevels, pdither) = match self.posterize {
            Some(p) => (p.levels, p.dither),
            None => ([1u32; 3], 0u32),
        };
        self.queue.write_buffer(
            &dda.resolve_dims,
            20,
            bytemuck::bytes_of(&[plevels[0], plevels[1], plevels[2], pdither]),
        );

        // Pack per-grid cameras into a runtime-sized storage buffer
        // (binding 15) — no fixed cap on grid count.
        let mut cam_vec: Vec<SceneDdaPerGridCamera> = cameras
            .iter()
            .map(SceneDdaPerGridCamera::from_camera)
            .collect();
        // XS.3 — stamp each grid's world transform for cross-grid shadows.
        for (c, t) in cam_vec.iter_mut().zip(grid_world.iter()) {
            c.set_world_transform(t);
        }

        // DL — pack the per-frame lights (already grid-local). The per-grid
        // sun direction rides in each `PerGridCamera.sun_dir` (binding 15);
        // point lights go in one storage buffer (binding 18). All-zero
        // ⇒ the pre-DL render. Shared with the headless path.
        // PF.4 — pack CPU-side (no clone of `scene_lights`), then write into
        // the persistent grow-only buffers instead of `create_buffer_init`-ing
        // fresh ones (which also forced a bind-group rebuild) every frame.
        if self.frame_pack.is_none() {
            self.frame_pack = Some(FramePackBuffers::new(&self.device));
        }
        let lights = &self.scene_lights;
        // Sun dirs ride in the per-frame camera vector — inject every frame.
        inject_grid_sun_dirs(lights, &mut cam_vec);
        let fp = self.frame_pack.as_mut().expect("just built");
        fp.write_cameras(&self.device, &self.queue, &cam_vec);
        // PF.5 — re-pack + re-upload the grid-major point lights only when
        // the rig changed (or the grid count did — the rows depend on it).
        if self.scene_lights_dirty || self.lights_packed_grids != scene.grid_count {
            let (packed_lights, sun_flags, point_count) =
                pack_scene_lights(lights, scene.grid_count as usize);
            fp.write_point_lights(&self.device, &self.queue, &packed_lights);
            self.lights_sun_flags = sun_flags;
            self.lights_point_count = point_count;
            self.lights_packed_grids = scene.grid_count;
            self.scene_lights_dirty = false;
        }
        let (sun_flags, point_count) = (self.lights_sun_flags, self.lights_point_count);

        let uniform = SceneDdaUniform {
            fov_y_rad,
            grid_count: scene.grid_count,
            max_outer_steps,
            _pad0: 0,
            screen_size: [render_w, render_h],
            _pad1: [0; 2],
            fog_color: [
                self.fog_color[0],
                self.fog_color[1],
                self.fog_color[2],
                self.fog_near,
            ],
            fog_far: self.fog_far,
            // L3.1: always write scene depth. Costs one storage store per
            // pixel, and the depth is needed for sprite z-test, sprite-less
            // `pick_depth`, and `draw_lines` occlusion alike.
            write_depth: 1,
            occ_page_words: scene.occupancy_page_words,
            occ_num_pages: scene.occupancy_num_pages,
            mip_scan_dist: self.scene_mip_scan_dist,
            terrain_has_translucent: u32::from(self.scene_terrain_translucent),
            terrain_map_count: self.scene_terrain_map.len() as u32,
            _pad4: 0,
            // Sky direction comes from the world (sprite) camera, so a
            // grid-less sprite-only scene still paints a real sky.
            sky_cam: SceneDdaPerGridCamera::from_camera(sprite_camera),
            side_shades0: self.scene_side_shades[0],
            side_shades1: self.scene_side_shades[1],
            sun_color: [
                lights.sun_color[0],
                lights.sun_color[1],
                lights.sun_color[2],
                lights.sun_intensity,
            ],
            ambient_color: [
                lights.ambient[0],
                lights.ambient[1],
                lights.ambient[2],
                lights.shadow_strength,
            ],
            sun_flags,
            point_light_count: point_count,
            shadow_max_steps: lights.shadow_max_steps,
            _pad5: 0,
            shadow_bias: lights.shadow_bias,
            shadow_max_dist: lights.shadow_max_dist,
            _pad6: [0.0; 2],
            shadow_tint: [
                lights.shadow_tint[0],
                lights.shadow_tint[1],
                lights.shadow_tint[2],
                0.0,
            ],
            style_bands: lights.style_bands,
            // XS.4.3 — visible sprite casters for the scene-pass cast march
            // (only when the device is sprite-shadow capable; else the cast
            // bindings/loop are absent).
            sprite_cast_count: if self.sprite_shadows_capable {
                sprite_pass.map_or(0, |(visible, _)| visible)
            } else {
                0
            },
            _pad7: [0; 2],
        };
        self.queue
            .write_buffer(&dda.uniform_buf, 0, bytemuck::bytes_of(&uniform));

        // PF.4 — cached bind group, keyed on the exact resources bound.
        // Occupancy page 0 at binding 1; pages 1..MAX_OCC_PAGES at 12..
        // (GPU.X paging). Per-grid point lights at 18 (DL); the per-grid
        // sun dir rides in PerGridCamera.sun_dir (binding 15).
        let mut dda_bufs: Vec<(u32, wgpu::Buffer)> = vec![
            (0, dda.uniform_buf.clone()),
            (1, scene.occupancy_pages[0].clone()),
            (2, scene.all_color_offsets.clone()),
            (3, scene.all_colors.clone()),
            (4, scene.all_chunk_colors_base.clone()),
            (5, scene.all_chunk_occupancy.clone()),
            (6, scene.grid_static_meta.clone()),
            (7, scene.all_slot_chunk_idx.clone()),
            (8, dda.framebuffer.clone()),
            (11, dda.depth_buffer.clone()),
            (12, scene.occupancy_pages[1].clone()),
            (13, scene.occupancy_pages[2].clone()),
            (14, scene.occupancy_pages[3].clone()),
            (15, fp.grid_cameras.clone()),
            (16, dda.materials_pal_buf.clone()),
            (17, dda.terrain_map_buf.clone()),
            (18, fp.point_lights.clone()),
        ];
        // XS.4.3 — sprite-cast bindings (19..21). On a capable device the BGL
        // has them, so bind the sprite registry when present (terrain shadow
        // rays test sprite volumes), else the dummy (sprite_cast_count == 0).
        if self.sprite_shadows_capable {
            let dummy = dda
                .sprite_cast_dummy
                .as_ref()
                .expect("capable scene_dda has a sprite-cast dummy");
            let (insts, models, occ) = match &self.sprite_registry {
                Some(reg) => (&reg.instances, &reg.model_meta, &reg.occupancy),
                None => (dummy, dummy, dummy),
            };
            dda_bufs.push((19, insts.clone()));
            dda_bufs.push((20, models.clone()));
            dda_bufs.push((21, occ.clone()));
        }
        let dda_bg = cached_bind_group(
            &mut fp.dda_bg,
            &self.device,
            "roxlap-gpu scene_dda.bg",
            &dda.bgl_dda,
            dda_bufs,
            vec![(9, self.sky_view.clone())],
            &[(10, &self.sky_sampler)],
        )
        .clone();

        // GPU.9 — when sprites are present, build both splatter bind
        // groups up front (the splat pass writes the key buffer; the
        // resolve pass reads keys + scene depth and writes colour).
        // GPU.10.3 — model-DDA bind group + per-frame uniform, using the
        // cull/bin results captured above. Per-model + per-instance data
        // + the tile lists live in the registry buffers.
        let sprite_model_bg = match (&self.sprite_model_dda, &self.sprite_registry, sprite_pass) {
            (Some(smd), Some(reg), Some((visible, tiles_x))) => {
                // World camera (see the cull pass above) — sprites
                // project through it regardless of grid 0's transform.
                let cam = sprite_camera;
                // DL.4 — world-space lights for the sprite pass (sprites are
                // world-space, not grid-local). No sprite shadows (deferred).
                let dl = &self.scene_lights;
                let sprite_sun_enabled = dl.world_sun_dir != [0.0; 3];
                let sprite_point_count = dl.world_points.len().min(MAX_POINT_LIGHTS) as u32;
                // PF.4 — persistent buffer instead of a per-frame allocation.
                // PF.5 — rebuilt + re-uploaded only when the rig changed;
                // this pass's own dirty flag (it only runs with sprites on
                // screen, so it can't ride the scene pack's flag).
                if self.sprite_lights_dirty {
                    let sprite_pts: Vec<GpuPointLight> = dl
                        .world_points
                        .iter()
                        .take(MAX_POINT_LIGHTS)
                        .map(|l| GpuPointLight {
                            pos: l.position,
                            radius: l.radius,
                            color: l.color,
                            intensity: l.intensity,
                            spot_dir: l.spot_dir,
                            cos_outer: l.cos_outer,
                            cos_inner: l.cos_inner,
                            // XS.4.2 — honour the light's caster flag so a
                            // receiving sprite is shadowed by it (capable
                            // devices).
                            casts_shadow: u32::from(l.casts_shadow),
                            _pad: [0; 2],
                        })
                        .collect();
                    fp.write_sprite_lights(&self.device, &self.queue, &sprite_pts);
                    self.sprite_lights_dirty = false;
                }
                // sun_flags bit0 = sun enabled, bit1 = sun casts shadow (XS.4.2),
                // bit2 = dynamic lighting active.
                let sprite_sun_flags = u32::from(sprite_sun_enabled)
                    | (u32::from(dl.sun_casts_shadow) << 1)
                    | (u32::from(dl.enabled) << 2);
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
                    screen_size: [render_w, render_h],
                    instance_count: visible,
                    fog_far: self.fog_far,
                    fov_y_rad,
                    tiles_x,
                    tile_size: SPRITE_TILE_SIZE,
                    has_translucent: u32::from(self.sprite_has_translucent),
                    sun_dir: [
                        dl.world_sun_dir[0],
                        dl.world_sun_dir[1],
                        dl.world_sun_dir[2],
                        0.0,
                    ],
                    sun_color: [
                        dl.sun_color[0],
                        dl.sun_color[1],
                        dl.sun_color[2],
                        dl.sun_intensity,
                    ],
                    ambient_color: [dl.ambient[0], dl.ambient[1], dl.ambient[2], 0.0],
                    sun_flags: sprite_sun_flags,
                    point_light_count: sprite_point_count,
                    _pad_dl: [0; 2],
                    shadow_tint: [dl.shadow_tint[0], dl.shadow_tint[1], dl.shadow_tint[2], 0.0],
                    style_bands: dl.style_bands,
                    // XS.4.2 — sprite-shadow (receive) ABI, mirroring the scene
                    // pass. Only consulted when the device is sprite-shadow
                    // capable (the shadowed shader variant is built); otherwise
                    // the stub `sprite_shadow_occluded` ignores them.
                    occ_num_pages: scene.occupancy_num_pages,
                    occ_page_words: scene.occupancy_page_words,
                    grid_count: scene.grid_count,
                    max_outer_steps,
                    shadow_max_steps: dl.shadow_max_steps,
                    shadow_bias: dl.shadow_bias,
                    shadow_max_dist: dl.shadow_max_dist,
                    shadow_strength: dl.shadow_strength,
                    _pad_xs: [0; 3],
                };
                self.queue
                    .write_buffer(&smd.uniform_buf, 0, bytemuck::bytes_of(&uni));
                // PF.4 — cached bind group (identity-keyed, like the scene
                // pass's). World point lights at 15 (DL.7; binding 14 univec
                // normal table dropped — face-normal lighting now).
                let mut sprite_bufs: Vec<(u32, wgpu::Buffer)> = vec![
                    (0, smd.uniform_buf.clone()),
                    (1, reg.occupancy.clone()),
                    (2, reg.colors.clone()),
                    (3, reg.color_offsets.clone()),
                    (4, reg.model_meta.clone()),
                    (5, reg.instances.clone()),
                    (6, dda.depth_buffer.clone()),
                    (7, dda.framebuffer.clone()),
                    (8, reg.tile_ranges.clone()),
                    (9, reg.tile_instances.clone()),
                    (10, reg.dirs.clone()),
                    (11, reg.colmul.clone()),
                    (12, smd.materials_buf.clone()),
                    (13, reg.materials_vox.clone()),
                    (15, fp.sprite_lights.clone()),
                ];
                // XS.4.2 — when capable, bind the terrain occupancy set (the
                // same resident buffers + the per-frame grid cameras the scene
                // pass uses) so sprite shadow rays march terrain. Must match
                // the BGL built in `build_sprite_model_dda`.
                if self.sprite_shadows_capable {
                    let terrain: [(u32, &wgpu::Buffer); 8] = [
                        (16, &scene.occupancy_pages[0]),
                        (17, &scene.occupancy_pages[1]),
                        (18, &scene.occupancy_pages[2]),
                        (19, &scene.occupancy_pages[3]),
                        (20, &scene.all_chunk_occupancy),
                        (21, &scene.all_slot_chunk_idx),
                        (22, &scene.grid_static_meta),
                        (23, &fp.grid_cameras),
                    ];
                    for (binding, buf) in terrain {
                        sprite_bufs.push((binding, buf.clone()));
                    }
                }
                Some(
                    cached_bind_group(
                        &mut fp.sprite_bg,
                        &self.device,
                        "roxlap-gpu sprite_model_dda.bg",
                        &smd.bgl,
                        sprite_bufs,
                        Vec::new(),
                        &[],
                    )
                    .clone(),
                )
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
            cpass.dispatch_workgroups(render_w.div_ceil(8), render_h.div_ceil(8), 1);
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
            cpass.dispatch_workgroups(render_w.div_ceil(8), render_h.div_ceil(8), 1);
        }
        // RP.1 — resolve pass: box-downfilter framebuffer(march) →
        // resolve_buf(logical). One thread per logical pixel.
        // PF.5 (H6) — with ssaa == 1 AND posterize off the resolve is an
        // identity copy: skip the whole full-screen pass and blit straight
        // from the framebuffer instead (byte-identical output).
        let identity_resolve =
            (render_w, render_h) == (logical_w, logical_h) && self.posterize.is_none();
        if !identity_resolve {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("roxlap-gpu scene_dda resolve"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&dda.pipeline_resolve);
            cpass.set_bind_group(0, &dda.resolve_bg, &[]);
            cpass.dispatch_workgroups(logical_w.div_ceil(8), logical_h.div_ceil(8), 1);
        }
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("roxlap-gpu scene_dda blit"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surf_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rpass.set_pipeline(&dda.pipeline_blit);
            rpass.set_bind_group(
                0,
                if identity_resolve {
                    &dda.blit_bg_direct
                } else {
                    &dda.blit_bg
                },
                &[],
            );
            rpass.draw(0..3, 0..1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        // This frame wrote `scene_dda.depth_buffer`, so depth-tested
        // overlays may test against it.
        self.scene_depth_valid = true;
        // Deferred present — the host calls `present` or `paint_egui`.
        self.pending_frame = Some((surf_tex, surf_view));
        self.frame_count = self.frame_count.wrapping_add(1);
    }

    /// Like [`Self::render`] (clear to colour) but **deferred**: stashes
    /// the frame for [`Self::present`] / [`Self::paint_egui`] instead of
    /// presenting. The facade uses this before any grid is resident so a
    /// HUD can still be painted over an empty scene.
    pub fn render_clear_deferred(&mut self) {
        // No scene pass this frame ⇒ `scene_dda.depth_buffer` (if it
        // exists from an earlier scene) is stale; depth-tested overlays
        // must not test against it.
        self.scene_depth_valid = false;
        self.pending_frame = None;
        let Some(surf_tex) = self.acquire_frame() else {
            return;
        };
        let view = surf_tex
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let [r, g, b] = self.clear_colour;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("roxlap-gpu clear (deferred)"),
            });
        {
            let _rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("roxlap-gpu clear (deferred)"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r, g, b, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        self.pending_frame = Some((surf_tex, view));
    }

    /// Present the frame stashed by the last deferred render
    /// ([`Self::render_scene`] / [`Self::render_clear_deferred`]). No-op
    /// if nothing is pending (e.g. the surface was lost mid-render).
    pub fn present(&mut self) {
        if let Some((surf_tex, _view)) = self.pending_frame.take() {
            surf_tex.present();
        }
    }

    /// Block until the GPU has drained every submitted command (queue
    /// idle), dropping any not-yet-presented swapchain frame first. Call at
    /// shutdown — before the [`GpuRenderer`] (and its window) drop — so the
    /// device is torn down with no work in flight and no half-presented
    /// frame, instead of yanking the swapchain mid-submission (which leaves
    /// the driver/compositor compositing stale buffers — the "leftover
    /// triangles / flicker after an unclean exit" symptom). No-op on wasm
    /// (`poll(Wait)` is unavailable there; the browser reclaims the device).
    pub fn wait_idle(&mut self) {
        // Release the acquired-but-unpresented frame so its swapchain image
        // isn't held across teardown.
        self.pending_frame = None;
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        }
    }

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
        let no_depth = u32::from(self.scene_dda.is_none() || !self.scene_depth_valid);
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
        let no_depth = u32::from(self.scene_dda.is_none() || !self.scene_depth_valid);
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

    /// Project a world point to window pixels under the marcher's
    /// vertical-FOV pinhole (the inverse of [`Self::pixel_ray`]), using
    /// the last-rendered frame's size + FOV. `None` before the first
    /// scene render or for a point at/behind the near plane.
    #[must_use]
    pub fn project_point(
        &self,
        cam_pos: [f32; 3],
        right: [f32; 3],
        down: [f32; 3],
        forward: [f32; 3],
        world: [f32; 3],
    ) -> Option<(f32, f32)> {
        let dda = self.scene_dda.as_ref()?;
        let (w, h) = dda.storage_size;
        if w == 0 || h == 0 || self.last_fov_y_rad <= 0.0 {
            return None;
        }
        let d = [
            world[0] - cam_pos[0],
            world[1] - cam_pos[1],
            world[2] - cam_pos[2],
        ];
        let cz = forward[0] * d[0] + forward[1] * d[1] + forward[2] * d[2];
        if cz < LINE_NEAR_Z {
            return None;
        }
        let cx = right[0] * d[0] + right[1] * d[1] + right[2] * d[2];
        let cy = down[0] * d[0] + down[1] * d[1] + down[2] * d[2];
        let half_h = (self.last_fov_y_rad * 0.5).tan();
        let half_w = half_h * (w as f32 / h as f32);
        let ndc_x = (cx / cz) / half_w;
        let ndc_y = -(cy / cz) / half_h;
        let sx = (ndc_x * 0.5 + 0.5) * w as f32;
        let sy = (0.5 - ndc_y * 0.5) * h as f32;
        Some((sx, sy))
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

    fn build_scene_dda(
        &self,
        width: u32,
        height: u32,
        logical_w: u32,
        logical_h: u32,
        surface_w: u32,
        surface_h: u32,
        surface_format: wgpu::TextureFormat,
    ) -> SceneDdaResources {
        // `width`/`height` are the **march** size (`logical × ssaa`) — the
        // scene + sprite + depth passes run at it. `logical_*` is the resolved
        // (retro) grid the resolve pass downfilters into and the blit reads.
        // `surface_*` is the swapchain the blit upscales onto. Framebuffer is a
        // packed-`rgba8unorm` storage buffer (row stride = march `width`).
        let framebuffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("roxlap-gpu scene_dda.framebuffer"),
            size: u64::from(width) * u64::from(height) * 4,
            // QE.7a - COPY_SRC so `read_frame_pixels` can stage the
            // identity-resolve path (ssaa 1, posterize off) for capture.
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // RP.1 — logical-resolution buffer the resolve pass writes; the blit
        // reads it (so the blit src is the *logical* size, not the march size).
        let resolve_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("roxlap-gpu scene_dda.resolve_buf"),
            size: u64::from(logical_w) * u64::from(logical_h) * 4,
            // QE.7a - COPY_SRC so `read_frame_pixels` can stage it (capture).
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // Resolve uniform: `[src(march) w,h, dst(logical) w,h, ssaa,
        // levels r,g,b, dither, pad×3]` (48 B). Dims+ssaa written here; the
        // posterize fields (offset 20) are re-written per frame in render_scene.
        let resolve_dims = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("roxlap-gpu scene_dda.resolve_dims"),
            size: 48,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(
            &resolve_dims,
            0,
            bytemuck::bytes_of(&[width, height, logical_w, logical_h, self.ssaa]),
        );
        // Blit uniform `Dims`: logical (src) size, swapchain (dst) size, then
        // `flip_x` + pad (RP.0 nearest upscale). The flip flag (offset 16) is
        // re-written per frame in `render_scene`; a render/surface resize
        // forces a full rebuild, so the sizes only need writing here.
        let blit_dims = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("roxlap-gpu scene_dda.blit_dims"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(
            &blit_dims,
            0,
            bytemuck::bytes_of(&[
                logical_w,
                logical_h,
                surface_w,
                surface_h,
                u32::from(self.flip_x),
                0u32,
                0u32,
                0u32,
            ]),
        );

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
        // XS.4.3 — on sprite-shadow-capable devices, splice the sprite-cast
        // snippet over the `sprites_occlude` stub (binds the sprite registry at
        // 19..21 so terrain shadow rays test sprite volumes).
        let capable = self.sprite_shadows_capable;
        let dda_shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("scene_dda.wgsl"),
                source: wgpu::ShaderSource::Wgsl(scene_shader_source(capable).into()),
            });
        let mut dda_entries = vec![
            bgl_uniform_entry(0),
            bgl_storage_entry(1, true),
            bgl_storage_entry(2, true),
            bgl_storage_entry(3, true),
            bgl_storage_entry(4, true),
            bgl_storage_entry(5, true),
            bgl_storage_entry(6, true),
            bgl_storage_entry(7, true),
            // Framebuffer storage buffer (read-write; the scene +
            // sprite passes write packed pixels into it).
            bgl_storage_entry(8, false),
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
            // Per-grid cameras (runtime-sized; one per grid).
            bgl_storage_entry(15, true),
            // TV.6 — material palette + terrain colour→material map.
            bgl_storage_entry(16, true),
            bgl_storage_entry(17, true),
            // DL — per-grid point lights (18). Sun dir rides in
            // PerGridCamera (binding 15) to stay within the 16
            // storage-buffer limit.
            bgl_storage_entry(18, true),
        ];
        if capable {
            // XS.4.3 — sprite registry for the sprite-cast shadow march.
            dda_entries.push(bgl_storage_entry(19, true)); // sprite_instances
            dda_entries.push(bgl_storage_entry(20, true)); // sprite_models
            dda_entries.push(bgl_storage_entry(21, true)); // sprite_occupancy
        }
        let bgl_dda = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("roxlap-gpu scene_dda.bgl"),
                entries: &dda_entries,
            });
        let dda_pl = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("roxlap-gpu scene_dda.layout"),
                bind_group_layouts: &[Some(&bgl_dda)],
                immediate_size: 0,
            });
        let pipeline_dda = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("roxlap-gpu scene_dda.pipeline"),
                layout: Some(&dda_pl),
                module: &dda_shader,
                entry_point: Some("render_scene"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        // RP.1 — box-downfilter resolve pass (framebuffer march → resolve_buf
        // logical). `ssaa == 1` is a 1×1 copy; the blit always reads resolve_buf.
        let resolve_shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("scene_resolve.wgsl"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/scene_resolve.wgsl").into(),
                ),
            });
        let bgl_resolve = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("roxlap-gpu scene_dda.resolve_bgl"),
                entries: &[
                    bgl_storage_entry(0, true),  // src framebuffer (read)
                    bgl_storage_entry(1, false), // dst resolve_buf (read-write)
                    bgl_uniform_entry(2),        // resolve dims
                ],
            });
        let resolve_pl = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("roxlap-gpu scene_dda.resolve_layout"),
                bind_group_layouts: &[Some(&bgl_resolve)],
                immediate_size: 0,
            });
        let pipeline_resolve =
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("roxlap-gpu scene_dda.resolve_pipeline"),
                    layout: Some(&resolve_pl),
                    module: &resolve_shader,
                    entry_point: Some("main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });
        let resolve_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("roxlap-gpu scene_dda.resolve_bg"),
            layout: &bgl_resolve,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: framebuffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resolve_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resolve_dims.as_entire_binding(),
                },
            ],
        });

        let blit_shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("scene_blit.wgsl"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/scene_blit.wgsl").into()),
            });
        let bgl_blit = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("roxlap-gpu scene_dda.blit_bgl"),
                entries: &[
                    // Framebuffer storage buffer (read-only in the blit).
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Screen-size uniform for the pixel→index math.
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let blit_pl = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("roxlap-gpu scene_dda.blit_layout"),
                bind_group_layouts: &[Some(&bgl_blit)],
                immediate_size: 0,
            });
        let pipeline_blit = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("roxlap-gpu scene_dda.blit_pipeline"),
                layout: Some(&blit_pl),
                vertex: wgpu::VertexState {
                    module: &blit_shader,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &blit_shader,
                    entry_point: Some("fs_main"),
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
                multiview_mask: None,
                cache: None,
            });
        let blit_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("roxlap-gpu scene_dda.blit_bg"),
            layout: &bgl_blit,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    // RP.1 — blit reads the logical resolve buffer.
                    resource: resolve_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: blit_dims.as_entire_binding(),
                },
            ],
        });
        // PF.5 (H6) — direct-blit variant reading the march framebuffer:
        // used when the resolve pass would be an identity copy (ssaa == 1,
        // posterize off ⇒ march size == logical size), letting render_scene
        // skip that full-screen pass entirely.
        let blit_bg_direct = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("roxlap-gpu scene_dda.blit_bg_direct"),
            layout: &bgl_blit,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: framebuffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: blit_dims.as_entire_binding(),
                },
            ],
        });

        // TV.6 — material palette + terrain map buffers, seeded from the
        // renderer's current scene-material state (so a map defined before the
        // scene pass was built still takes effect).
        let (materials_pal_buf, terrain_map_buf) = {
            use wgpu::util::DeviceExt;
            let pal = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("roxlap-gpu scene_dda.materials_pal"),
                    contents: bytemuck::cast_slice(self.scene_materials.as_slice()),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });
            // Fixed 256-row map (≤256 materials anyway) → no re-alloc when the
            // host changes the map after the scene pass is built.
            let mut rows = [[0u32; 2]; 256];
            for (slot, &row) in rows.iter_mut().zip(self.scene_terrain_map.iter()) {
                *slot = row;
            }
            let map = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("roxlap-gpu scene_dda.terrain_map"),
                    contents: bytemuck::cast_slice(&rows),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });
            (pal, map)
        };

        SceneDdaResources {
            storage_size: (width, height),
            logical_size: (logical_w, logical_h),
            framebuffer,
            resolve_buf,
            uniform_buf,
            bgl_dda,
            pipeline_dda,
            pipeline_resolve,
            resolve_bg,
            resolve_dims,
            blit_bg,
            blit_bg_direct,
            pipeline_blit,
            blit_dims,
            depth_buffer,
            depth_readback,
            materials_pal_buf,
            terrain_map_buf,
            // XS.4.3 — 80-byte dummy (≥ one Instance) for the sprite-cast
            // bindings when capable but no sprite registry is bound this frame.
            sprite_cast_dummy: capable.then(|| {
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("roxlap-gpu scene_dda.sprite_cast_dummy"),
                    size: 80,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                })
            }),
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
    ///
    /// Compiles on wasm, but the wasm facade never calls it: WebGPU's
    /// `device.poll` doesn't block for the GPU, so the blocking
    /// `recv()` here would hang the single browser thread. Picking is
    /// deferred on the wasm GPU path (the facade returns `None`).
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
        // PF.5 (H4) — copy ONLY the picked pixel's 4 bytes, not the whole
        // depth buffer (8+ MB at high res): the pick still blocks on the
        // poll below, but the copy + map are now O(1). The 4-byte offset
        // meets wgpu's copy alignment.
        let offset = (u64::from(y) * u64::from(w) + u64::from(x)) * 4;
        enc.copy_buffer_to_buffer(&dda.depth_buffer, offset, &dda.depth_readback, 0, 4);
        self.queue.submit(std::iter::once(enc.finish()));

        let slice = dda.depth_readback.slice(..4);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        rx.recv().ok()?.ok()?;

        let t = {
            let data = slice.get_mapped_range();
            let bytes: [u8; 4] = data[0..4].try_into().ok()?;
            f32::from_le_bytes(bytes)
        };
        dda.depth_readback.unmap();

        // Reject sky / no-hit (T_INF == 1e30 in the shader) + non-finite.
        if !t.is_finite() || t >= 1.0e29 {
            return None;
        }
        Some(t)
    }

    /// QE.7a — read back the last rendered frame's colour at the
    /// **logical** resolution (post-SSAA/posterize, pre-upscale) as
    /// `0x00RRGGBB` pixels — the GPU side of frame capture, closing
    /// the "screenshots impossible on the GPU backend" parity gap.
    ///
    /// Blocking (encode copy → submit → map, like
    /// [`Self::read_depth_pixel`]): a screenshot hotkey, not a
    /// per-frame path. `None` before the first scene render. Compiles
    /// on wasm but must not be called there — WebGPU's `poll` can't
    /// block, so the facade returns `None` on the wasm GPU path.
    #[must_use]
    pub fn read_frame_pixels(&self) -> Option<(Vec<u32>, u32, u32)> {
        let dda = self.scene_dda.as_ref()?;
        let (w, h) = dda.logical_size;
        if w == 0 || h == 0 {
            return None;
        }
        // Mirror `render_scene`'s identity-resolve choice: with ssaa 1
        // + posterize off the resolve pass was skipped and the march
        // framebuffer IS the logical image.
        let identity = dda.storage_size == dda.logical_size && self.posterize.is_none();
        let src = if identity {
            &dda.framebuffer
        } else {
            &dda.resolve_buf
        };
        let size = u64::from(w) * u64::from(h) * 4;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("roxlap-gpu capture staging"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("roxlap-gpu capture readback"),
            });
        enc.copy_buffer_to_buffer(src, 0, &staging, 0, size);
        self.queue.submit(std::iter::once(enc.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        rx.recv().ok()?.ok()?;

        let pixels = {
            let data = slice.get_mapped_range();
            // The shaders store `pack4x8unorm(r, g, b, a)` — r in the
            // low byte. Repack to the facade's `0x00RRGGBB`.
            data.chunks_exact(4)
                .map(|px| {
                    let v = u32::from_le_bytes([px[0], px[1], px[2], px[3]]);
                    let (r, g, b) = (v & 0xff, (v >> 8) & 0xff, (v >> 16) & 0xff);
                    (r << 16) | (g << 8) | b
                })
                .collect()
        };
        staging.unmap();
        Some((pixels, w, h))
    }

    /// World-space view-ray direction (un-normalised) for window pixel
    /// `(x, y)`, under the GPU marcher's projection — the canonical GPU
    /// unproject, mirroring `scene_dda.wgsl`'s `render_scene`
    /// (vertical-FOV pinhole). Uses the last-rendered frame's target
    /// size + FOV; `None` before the first scene render. Pair with
    /// [`Self::read_depth_pixel`] for screen→world picking.
    #[must_use]
    pub fn pixel_ray(
        &self,
        right: [f64; 3],
        down: [f64; 3],
        forward: [f64; 3],
        x: f64,
        y: f64,
    ) -> Option<[f64; 3]> {
        let dda = self.scene_dda.as_ref()?;
        let (w, h) = dda.storage_size;
        if w == 0 || h == 0 || self.last_fov_y_rad <= 0.0 {
            return None;
        }
        Some(pinhole_pixel_ray(
            right,
            down,
            forward,
            x,
            y,
            f64::from(w),
            f64::from(h),
            f64::from(self.last_fov_y_rad),
        ))
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

    /// Incrementally append sprite instances **without** rebuilding the
    /// registry — the cheap streaming-spawn path (asteroids, projectiles).
    /// Returns the index of the first appended instance (`[base, base+N)`).
    ///
    /// Every appended instance must reference a model already registered
    /// by the [`Self::set_sprite_instances`] that established residency
    /// (model volumes are not re-uploaded here — build the full
    /// `SpriteModelRegistry` up front and seed it once, then stream
    /// instances). If no registry is resident yet, this performs the
    /// initial full upload and returns `0`.
    ///
    /// Cost is amortised O(1) per instance (the GPU instance buffer grows
    /// by powers of two), versus the full volume + buffer rebuild of
    /// [`Self::set_sprite_instances`].
    pub fn append_sprite_instances(
        &mut self,
        registry: &sprite_model::SpriteModelRegistry,
        instances: &[sprite_model::SpriteInstance],
    ) -> u32 {
        match self.sprite_registry.as_mut() {
            Some(reg) => reg.append_instances(&self.device, registry, instances),
            None => {
                self.set_sprite_instances(registry, instances);
                0
            }
        }
    }

    /// Remove the sprite instance at `index` (swap-remove, O(1), no model
    /// re-upload). Returns `Some(old_last)` if a different instance was
    /// moved into `index` to fill the hole — its index changed from
    /// `old_last` to `index`, so a caller tracking instance handles must
    /// update that one. Returns `None` if `index` was the last element /
    /// out of range, or no registry is resident.
    pub fn remove_sprite_instance(&mut self, index: usize) -> Option<usize> {
        self.sprite_registry
            .as_mut()
            .and_then(|reg| reg.remove_instance(index))
    }

    /// Incrementally add a new model (its full LOD chain) to the resident
    /// sprite registry **without** re-uploading the existing models — the
    /// counterpart to [`Self::append_sprite_instances`] for streaming in
    /// new geometry (unique asteroids, generated meshes).
    ///
    /// Usage mirrors `update_sprite_model`: you own the
    /// [`SpriteModelRegistry`], append
    /// the model with [`add_lod`](sprite_model::SpriteModelRegistry::add_lod)
    /// (or `add`), then pass the returned `chain_id` here to sync that one
    /// chain to the GPU. Afterwards [`Self::append_sprite_instances`] may
    /// reference it.
    ///
    /// If no registry is resident yet, this performs the initial full
    /// upload of `registry` (all its current models, zero instances) to
    /// establish residency — so call it for your *first* model; only
    /// chains appended *after* residency exists are added incrementally.
    ///
    /// Cost is amortised O(new model voxels): the shared volume buffers
    /// carry slack and bump-append, growing (and rebuilding once from the
    /// registry) only on overflow.
    /// Flush queued `write_buffer` uploads by submitting an empty command
    /// stream. wgpu stages `write_buffer` data and flushes it on the next
    /// `Queue::submit`; calling this between batches of uploads (e.g. a
    /// flipbook's frames in [`Self::add_sprite_model`]) recycles the device
    /// staging pool so a big one-shot batch can't exhaust it (which would
    /// then crash egui-wgpu's own `write_buffer`).
    pub fn flush_writes(&self) {
        self.queue.submit(std::iter::empty::<wgpu::CommandBuffer>());
    }

    pub fn add_sprite_model(
        &mut self,
        registry: &sprite_model::SpriteModelRegistry,
        chain_id: u32,
    ) {
        match self.sprite_registry.as_mut() {
            Some(reg) => reg.add_model(&self.device, &self.queue, registry, chain_id),
            None => {
                self.sprite_registry = Some(sprite_model::SpriteRegistryResident::upload(
                    &self.device,
                    registry,
                    &[],
                ));
            }
        }
    }

    /// Remove a model (tombstone its LOD chain) from the resident sprite
    /// registry — the counterpart to [`Self::add_sprite_model`]. Frees its
    /// `colors`/`dirs` space for reuse by a later add; the smaller
    /// `occupancy`/`color_offsets` holes are reclaimed by
    /// [`Self::compact_sprite_models`]. Entry / chain ids stay stable, so
    /// other models' `chain_id`s remain valid.
    ///
    /// Instances of the removed model keep their slots but draw as nothing
    /// until the caller drops them via [`Self::remove_sprite_instance`].
    /// No-op if `chain_id` is unknown / already removed / no registry.
    pub fn remove_sprite_model(&mut self, chain_id: u32) {
        if let Some(reg) = self.sprite_registry.as_mut() {
            reg.remove_model(chain_id);
        }
    }

    /// Reclaim the holes left by [`Self::remove_sprite_model`] by rebuilding
    /// the shared volume buffers from the live models only. `registry` must
    /// be the resident one. Cost is O(live volume) — call it when
    /// [`Self::dead_sprite_model_count`] is high (e.g. exceeds the live
    /// count), not every frame. No-op if no registry is resident.
    pub fn compact_sprite_models(&mut self, registry: &sprite_model::SpriteModelRegistry) {
        if let Some(reg) = self.sprite_registry.as_mut() {
            reg.compact(&self.device, &self.queue, registry);
        }
    }

    /// Number of live (non-removed) sprite models (0 if none uploaded).
    #[must_use]
    pub fn sprite_model_count(&self) -> usize {
        self.sprite_registry
            .as_ref()
            .map_or(0, sprite_model::SpriteRegistryResident::live_model_count)
    }

    /// Number of removed-but-not-yet-compacted sprite models — the
    /// fragmentation signal for deciding when to call
    /// [`Self::compact_sprite_models`].
    #[must_use]
    pub fn dead_sprite_model_count(&self) -> usize {
        self.sprite_registry
            .as_ref()
            .map_or(0, sprite_model::SpriteRegistryResident::dead_model_count)
    }

    /// Number of resident sprite instances (0 if none uploaded).
    #[must_use]
    pub fn sprite_instance_count(&self) -> usize {
        self.sprite_registry
            .as_ref()
            .map_or(0, sprite_model::SpriteRegistryResident::instance_count)
    }

    /// Re-pose the already-resident sprite instances in place (no model
    /// volume re-upload) — the cheap per-frame path for animated KFA
    /// limbs. `instances` must match the last [`Self::set_sprite_instances`]
    /// in length + order. No-op if no sprite registry is resident.
    pub fn update_sprite_instance_transforms(
        &mut self,
        instances: &[sprite_model::SpriteInstance],
    ) {
        if let Some(reg) = self.sprite_registry.as_mut() {
            reg.update_transforms(instances);
        }
    }

    /// GPU.12 incremental — re-upload only LOD chain `chain_id`'s entries
    /// after an in-place edit of `registry` (carve / recolour), without
    /// rebuilding the whole sprite registry. `registry` must be the one
    /// last passed to [`Self::set_sprite_instances`] with chain
    /// `chain_id` already edited. No-op if no registry is resident.
    pub fn update_sprite_model(
        &mut self,
        registry: &sprite_model::SpriteModelRegistry,
        chain_id: u32,
    ) {
        if let Some(reg) = self.sprite_registry.as_mut() {
            reg.update_model(&self.device, &self.queue, registry, chain_id);
        }
    }

    /// VCL.2 — repoint sprite instance `index` at LOD chain `chain_id`
    /// (the per-frame flipbook step for animated voxel clips). `registry`
    /// is the resident one; `chain_id`'s volume must already be uploaded
    /// (e.g. a clip's frames registered via [`Self::add_sprite_model`]).
    /// CPU-side rewrite picked up by the next frame's cull — no volume
    /// re-upload. No-op if no registry is resident.
    pub fn set_sprite_instance_model(
        &mut self,
        registry: &sprite_model::SpriteModelRegistry,
        index: usize,
        chain_id: u32,
    ) {
        if let Some(reg) = self.sprite_registry.as_mut() {
            reg.set_instance_model(registry, index, chain_id);
        }
    }

    /// Set the per-instance `kv6colmul[256]` lighting tables (voxlap's
    /// `update_reflects` output, e.g. via `roxlap_core::sprite::
    /// sprite_colmul`), in the same order/length as the last
    /// [`Self::set_sprite_instances`]. The GPU sprite pass modulates each
    /// voxel by its surface normal's entry — matching the CPU rasteriser.
    /// No-op if no sprite registry is resident.
    pub fn set_sprite_instance_colmul(&mut self, tables: &[[u64; 256]]) {
        if let Some(reg) = self.sprite_registry.as_mut() {
            reg.set_instance_colmul(tables);
        }
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

    /// Set per-face grid side-shading — voxlap's
    /// `setsideshades(top, bot, left, right, up, down)`. Each value is
    /// subtracted (as a u8, matching the CPU `gcsub` high byte) from a
    /// hit voxel's brightness byte before shading, so the scene-DDA pass
    /// darkens grid faces the same way the CPU rasteriser does. `[0; 6]`
    /// disables it (the default). The hit face is taken from the DDA's
    /// last-stepped axis + ray direction.
    pub fn set_scene_side_shades(&mut self, s: [i8; 6]) {
        // Reinterpret each i8 as u8 (voxlap stamps `sxx` into gcsub's
        // high byte verbatim), then pack (top, bot, left, right) /
        // (up, down, 0, 0) for the two uniform vec4s.
        let v = |i: usize| i32::from(s[i] as u8);
        self.scene_side_shades = [[v(0), v(1), v(2), v(3)], [v(4), v(5), 0, 0]];
    }

    /// DL — set the per-frame dynamic lights (sun + point lights), already
    /// transformed into each grid's local frame. Call once per frame before
    /// [`Self::render_scene`] (the facade does this from
    /// `FrameParams::lights`). [`SceneLights::default`] clears all lights —
    /// the pre-DL render. GPU-only; the CPU backend has no analogue.
    /// PF.5 — an unchanged rig is a no-op: `render_scene` re-packs +
    /// re-uploads the light buffers only when this actually stores
    /// something different, so a static rig costs nothing per frame.
    pub fn set_scene_lights(&mut self, lights: SceneLights) {
        if self.scene_lights != lights {
            self.scene_lights = lights;
            self.scene_lights_dirty = true;
            self.sprite_lights_dirty = true;
        }
    }

    /// GPU.10.1 — build the instanced model-DDA pipeline (one thread
    /// per pixel). Lazily invoked the first frame a registry is present.
    fn build_sprite_model_dda(&self) -> SpriteModelDdaResources {
        // XS.4.2 — on sprite-shadow-capable devices, splice the terrain shadow
        // snippet over the stub (`shadow_occluded_world` becomes a real terrain
        // march; binds occupancy 16..23). Otherwise the stub keeps sprites
        // unshadowed and the BGL stays at the base 14 storage buffers.
        let capable = self.sprite_shadows_capable;
        let src = sprite_shader_source(capable);
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("sprite_model_dda.wgsl"),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
        let mut entries = vec![
            bgl_uniform_entry(0),
            bgl_storage_entry(1, true),  // occupancy
            bgl_storage_entry(2, true),  // colors
            bgl_storage_entry(3, true),  // color_offsets
            bgl_storage_entry(4, true),  // model_meta
            bgl_storage_entry(5, true),  // instances
            bgl_storage_entry(6, true),  // scene depth
            bgl_storage_entry(7, false), // framebuffer (read-write buffer)
            bgl_storage_entry(8, true),  // tile_ranges
            bgl_storage_entry(9, true),  // tile_instances
            bgl_storage_entry(10, true), // per-voxel dir
            bgl_storage_entry(11, true), // per-instance kv6colmul
            bgl_storage_entry(12, true), // TV — material palette
            bgl_storage_entry(13, true), // TV.3 — per-voxel material id
            bgl_storage_entry(15, true), // DL.7 — world point lights
        ];
        if capable {
            // XS.4.2 — terrain occupancy set for sprite RECEIVE shadows.
            entries.push(bgl_storage_entry(16, true)); // occ_page0
            entries.push(bgl_storage_entry(17, true)); // occ_page1
            entries.push(bgl_storage_entry(18, true)); // occ_page2
            entries.push(bgl_storage_entry(19, true)); // occ_page3
            entries.push(bgl_storage_entry(20, true)); // all_chunk_occupancy
            entries.push(bgl_storage_entry(21, true)); // all_slot_chunk_idx
            entries.push(bgl_storage_entry(22, true)); // grid_static_meta
            entries.push(bgl_storage_entry(23, true)); // grid_cameras
        }
        let bgl = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("roxlap-gpu sprite_model_dda.bgl"),
                entries: &entries,
            });
        let pl = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("roxlap-gpu sprite_model_dda.layout"),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("roxlap-gpu sprite_model_dda.pipeline"),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some("march"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let uniform_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("roxlap-gpu sprite_model_dda.uniform"),
            size: std::mem::size_of::<SpriteModelUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // TV — material palette, seeded from the current renderer state so a
        // table defined before the sprite pass was built still takes effect.
        let materials_buf = {
            use wgpu::util::DeviceExt;
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("roxlap-gpu sprite_model_dda.materials"),
                    contents: bytemuck::cast_slice(self.sprite_materials.as_slice()),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                })
        };
        SpriteModelDdaResources {
            bgl,
            pipeline,
            uniform_buf,
            materials_buf,
        }
    }

    /// TV — set the global voxel-material palette for the GPU sprite pass.
    /// Mirrors the renderer's [`MaterialTable`](roxlap_formats::material::MaterialTable):
    /// every sprite/clip instance's `material` id indexes it for opacity +
    /// blend mode. Cheap (2 KB); call it whenever the palette changes (or
    /// each frame). While every material is opaque the shader stays on the
    /// unchanged first-hit path.
    pub fn set_sprite_materials(&mut self, table: &roxlap_formats::material::MaterialTable) {
        let (palette, any_translucent) = material_palette(table);
        self.sprite_materials = palette;
        self.sprite_has_translucent = any_translucent;
        if let Some(smd) = &self.sprite_model_dda {
            self.queue.write_buffer(
                &smd.materials_buf,
                0,
                bytemuck::cast_slice(self.sprite_materials.as_slice()),
            );
        }
    }

    /// TV.6 — set the scene (terrain) material palette + colour→material map
    /// for the multi-grid scene pass. Matching-colour terrain voxels render
    /// translucent; an empty map / all-opaque palette renders unchanged. The
    /// map is capped at 256 rows (the fixed buffer size).
    pub fn set_scene_terrain_materials(
        &mut self,
        table: &roxlap_formats::material::MaterialTable,
        map: &[(u32, u8)],
    ) {
        let (palette, _) = material_palette(table);
        self.scene_materials = palette;
        self.scene_terrain_map = map
            .iter()
            .take(256)
            .map(|&(c, m)| [c & 0x00ff_ffff, u32::from(m)])
            .collect();
        self.scene_terrain_translucent = map.iter().any(|&(_, m)| !table.get(m).is_opaque());
        if let Some(dda) = &self.scene_dda {
            self.queue.write_buffer(
                &dda.materials_pal_buf,
                0,
                bytemuck::cast_slice(self.scene_materials.as_slice()),
            );
            if !self.scene_terrain_map.is_empty() {
                self.queue.write_buffer(
                    &dda.terrain_map_buf,
                    0,
                    bytemuck::cast_slice(&self.scene_terrain_map),
                );
            }
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
    /// Framebuffer storage buffer (packed `rgba8unorm`, tight rows) —
    /// matches the buffer-output `scene_dda.wgsl` (see its note).
    framebuffer: wgpu::Buffer,
    depth_buffer: wgpu::Buffer,
    uniform_buf: wgpu::Buffer,
    _sky_texture: wgpu::Texture,
    sky_view: wgpu::TextureView,
    sky_sampler: wgpu::Sampler,
    bgl: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    readback: wgpu::Buffer,
    /// Per-face side-shades for the gate render (default none). Packed
    /// `[(top,bot,left,right), (up,down,_,_)]`; set via
    /// [`Self::set_side_shades`].
    side_shades: [[i32; 4]; 2],
    /// DL — dynamic lights for the render (already grid-local, like the
    /// surface path). Default = none (baked-only). Set via
    /// [`Self::set_scene_lights`]; lets tests exercise the lit path.
    lights: SceneLights,
}

impl HeadlessSceneRenderer {
    /// Build the compute pipeline + output/readback resources for a
    /// `width × height` framebuffer. Validates `scene_dda.wgsl` and
    /// the [`scene::GridStaticMeta`] std430 layout at pipeline /
    /// bind-group time.
    #[must_use]
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, width: u32, height: u32) -> Self {
        let framebuffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("roxlap-gpu headless.framebuffer"),
            size: u64::from(width) * u64::from(height) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

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
        // Upload the default sky texel (create_sky_texture only allocates
        // — the texel must be written or the shader samples black, which
        // is why a grid-less headless render came back black).
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &sky_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &default_sky_pixel,
            wgpu::TexelCopyBufferLayout {
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
                // Framebuffer storage buffer (read-write).
                bgl_storage_entry(8, false),
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
                // Per-grid cameras (runtime-sized; one per grid).
                bgl_storage_entry(15, true),
                // TV.6 — material palette + terrain map (opaque dummies here).
                bgl_storage_entry(16, true),
                bgl_storage_entry(17, true),
                // DL — per-grid point lights (18). Sun dir rides in
                // PerGridCamera (binding 15).
                bgl_storage_entry(18, true),
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("roxlap-gpu headless.layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("roxlap-gpu headless.pipeline"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("render_scene"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Readback is a tight buffer-to-buffer copy (no 256-byte row
        // padding, unlike the old texture-to-buffer path).
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("roxlap-gpu headless.readback"),
            size: u64::from(width) * u64::from(height) * 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            width,
            height,
            framebuffer,
            depth_buffer,
            uniform_buf,
            _sky_texture: sky_texture,
            sky_view,
            sky_sampler,
            bgl,
            pipeline,
            readback,
            side_shades: [[0; 4]; 2],
            lights: SceneLights::default(),
        }
    }

    /// DL — set dynamic lights for subsequent [`Self::render`] calls
    /// (already in grid-local space). Lets tests exercise the lit path
    /// (sun N·L, point lights). Default = none (baked-only).
    pub fn set_scene_lights(&mut self, lights: SceneLights) {
        self.lights = lights;
    }

    /// Set per-face side-shades for subsequent [`Self::render`] calls —
    /// voxlap `setsideshades(top, bot, left, right, up, down)`, each an
    /// i8 stamped as u8 (matching the engine path). Lets the gate test
    /// the GPU side-shade darkening.
    pub fn set_side_shades(&mut self, s: [i8; 6]) {
        let v = |i: usize| i32::from(s[i] as u8);
        self.side_shades = [[v(0), v(1), v(2), v(3)], [v(4), v(5), 0, 0]];
    }

    /// Render `scene` from `cameras` (one per grid) and read the
    /// framebuffer back as `width*height` packed `0xAABBGGRR` pixels
    /// (R in the low byte). Fog is disabled. `mip_scan_dist` drives
    /// the GPU.11.1 scene-grid LOD (`0` = always mip-0). Blocks on
    /// readback.
    ///
    /// # Panics
    /// If `cameras.len() != scene.grid_count`.
    /// Headless render with identity per-grid world transforms (shadows stay
    /// intra-grid). See [`Self::render_with_transforms`] for the cross-grid
    /// (XS.3) variant.
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
        self.render_with_transforms(
            device,
            queue,
            scene,
            cameras,
            &[],
            fov_y_rad,
            max_outer_steps,
            mip_scan_dist,
        )
    }

    /// XS.3 — headless render with explicit per-grid world transforms, so the
    /// scene shader can lift a shadow ray to world space and test it against
    /// every grid (cross-grid shadows). Empty `grid_world` ⇒ identity.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn render_with_transforms(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &GpuSceneResident,
        cameras: &[Camera],
        grid_world: &[GridWorldTransform],
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

        let mut cam_vec: Vec<SceneDdaPerGridCamera> = cameras
            .iter()
            .map(SceneDdaPerGridCamera::from_camera)
            .collect();
        // XS.3 — stamp world transforms for cross-grid shadows (identity if absent).
        for (c, t) in cam_vec.iter_mut().zip(grid_world.iter()) {
            c.set_world_transform(t);
        }
        // TV.6 — opaque dummies for the material palette + terrain map
        // bindings (headless renders opaque-only: terrain_has_translucent=0).
        let (dummy_pal, dummy_map) = {
            use wgpu::util::DeviceExt;
            let pal: Vec<MaterialGpu> = vec![
                MaterialGpu {
                    alpha: 1.0,
                    mode: 0
                };
                256
            ];
            let p = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("roxlap-gpu headless.materials_pal"),
                contents: bytemuck::cast_slice(&pal),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let m = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("roxlap-gpu headless.terrain_map"),
                contents: bytemuck::cast_slice(&[[0u32; 2]]),
                usage: wgpu::BufferUsages::STORAGE,
            });
            (p, m)
        };
        // DL — pack any dynamic lights (default none ⇒ the baked-only path,
        // matching the oracle goldens). Injects sun dir into cam_vec.sun_dir
        // and builds the point-light buffer (binding 18). Shared with the
        // surface path.
        let dl = self.lights.clone();
        inject_grid_sun_dirs(&dl, &mut cam_vec);
        let (packed_lights, sun_flags, point_count) =
            pack_scene_lights(&dl, scene.grid_count as usize);
        let dummy_point_lights = upload_grid_point_lights(device, &packed_lights);
        let grid_cameras = upload_grid_cameras(device, &cam_vec);
        let uniform = SceneDdaUniform {
            fov_y_rad,
            grid_count: scene.grid_count,
            max_outer_steps,
            _pad0: 0,
            screen_size: [self.width, self.height],
            _pad1: [0; 2],
            // Fog off: near/far past any reachable t → factor 0.
            fog_color: [0.0, 0.0, 0.0, 1.0e29],
            fog_far: 1.0e30,
            write_depth: 0,
            occ_page_words: scene.occupancy_page_words,
            occ_num_pages: scene.occupancy_num_pages,
            mip_scan_dist,
            terrain_has_translucent: 0, // headless gate: opaque only
            terrain_map_count: 0,
            _pad4: 0,
            // Sky direction from the first grid camera (the world frame
            // in these tests); a default forward camera when there are
            // none (grid_count == 0) so the sky lookup stays valid.
            sky_cam: SceneDdaPerGridCamera::from_camera(&cameras.first().copied().unwrap_or(
                Camera {
                    position: [0.0; 3],
                    right: [1.0, 0.0, 0.0],
                    down: [0.0, 0.0, 1.0],
                    forward: [0.0, 1.0, 0.0],
                    fov_y_rad,
                },
            )),
            side_shades0: self.side_shades[0],
            side_shades1: self.side_shades[1],
            // DL — light parameters (default = no lights ⇒ sun_flags 0).
            sun_color: [
                dl.sun_color[0],
                dl.sun_color[1],
                dl.sun_color[2],
                dl.sun_intensity,
            ],
            ambient_color: [
                dl.ambient[0],
                dl.ambient[1],
                dl.ambient[2],
                dl.shadow_strength,
            ],
            sun_flags,
            point_light_count: point_count,
            shadow_max_steps: dl.shadow_max_steps,
            _pad5: 0,
            shadow_bias: dl.shadow_bias,
            shadow_max_dist: dl.shadow_max_dist,
            _pad6: [0.0; 2],
            shadow_tint: [dl.shadow_tint[0], dl.shadow_tint[1], dl.shadow_tint[2], 0.0],
            style_bands: dl.style_bands,
            sprite_cast_count: 0, // headless renderer has no sprite pass
            _pad7: [0; 2],
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
                    resource: self.framebuffer.as_entire_binding(),
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
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: grid_cameras.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: dummy_pal.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 17,
                    resource: dummy_map.as_entire_binding(),
                },
                // DL — dummy per-grid point lights (18). Sun dir rides in
                // PerGridCamera (binding 15).
                wgpu::BindGroupEntry {
                    binding: 18,
                    resource: dummy_point_lights.as_entire_binding(),
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
        enc.copy_buffer_to_buffer(
            &self.framebuffer,
            0,
            &self.readback,
            0,
            u64::from(self.width) * u64::from(self.height) * 4,
        );
        queue.submit(Some(enc.finish()));

        let slice = self.readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        rx.recv().expect("map_async channel").expect("map_async");

        let data = slice.get_mapped_range();
        // Tight `width*height` packed pixels — the shader's
        // `pack4x8unorm(vec4(r,g,b,a))` already yields `0xAABBGGRR`
        // little-endian, so a straight u32 read reconstructs each pixel.
        let out: Vec<u32> = data
            .chunks_exact(4)
            .map(|px| u32::from_le_bytes([px[0], px[1], px[2], px[3]]))
            .collect();
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
        // buffer the scene_dda stage needs 16. XS.4 GPU sprite shadows
        // need more (the sprite pass binds the terrain occupancy set on
        // top of its own — up to `SPRITE_SHADOW_MIN_STORAGE_BUFFERS`), so
        // request that many when the adapter offers them; capable devices
        // light up sprite shadows, others fall back (still ≥16 for the
        // base renderer). Both NVK and lavapipe advertise ≫16.
        max_storage_buffers_per_shader_stage: adapter_limits
            .max_storage_buffers_per_shader_stage
            .min(SPRITE_SHADOW_MIN_STORAGE_BUFFERS),
        ..wgpu::Limits::default()
    }
}

/// XS.4.2 — build the sprite-pass shader source. On a sprite-shadow-capable
/// device, splice `sprite_terrain_shadow.wgsl` over the `//XS4_STUB_BEGIN`..
/// `//XS4_STUB_END` block so `shadow_occluded_world` becomes the real terrain
/// march (+ the occupancy bindings 16..23); otherwise the stub keeps GPU
/// sprites unshadowed. The base file is always valid WGSL (the stub variant),
/// so `wgsl_shaders_validate` covers the fallback path.
fn sprite_shader_source(capable: bool) -> String {
    let base = include_str!("../shaders/sprite_model_dda.wgsl");
    if !capable {
        return base.to_string();
    }
    let snippet = include_str!("../shaders/sprite_terrain_shadow.wgsl");
    const BEGIN: &str = "//XS4_STUB_BEGIN";
    const END: &str = "//XS4_STUB_END";
    let (Some(b), Some(e)) = (base.find(BEGIN), base.find(END)) else {
        // Markers missing — fail loud rather than silently shipping the stub.
        panic!("sprite_model_dda.wgsl: XS4 stub markers not found");
    };
    let e_end = e + END.len();
    let mut out = String::with_capacity(base.len() + snippet.len());
    out.push_str(&base[..b]);
    out.push_str(snippet);
    out.push_str(&base[e_end..]);
    out
}

/// XS.4.3 — build the scene-pass shader source. On a sprite-shadow-capable
/// device, splice `scene_sprite_shadow.wgsl` over the `//XS4C_STUB_BEGIN`..
/// `//XS4C_STUB_END` block so `sprites_occlude` marches the sprite registry
/// (+ bindings 19..21) and terrain receives sprite-cast shadows; otherwise the
/// stub returns false. The base file is always valid WGSL (the stub variant).
fn scene_shader_source(capable: bool) -> String {
    let base = include_str!("../shaders/scene_dda.wgsl");
    if !capable {
        return base.to_string();
    }
    let snippet = include_str!("../shaders/scene_sprite_shadow.wgsl");
    const BEGIN: &str = "//XS4C_STUB_BEGIN";
    const END: &str = "//XS4C_STUB_END";
    let (Some(b), Some(e)) = (base.find(BEGIN), base.find(END)) else {
        panic!("scene_dda.wgsl: XS4C stub markers not found");
    };
    let e_end = e + END.len();
    let mut out = String::with_capacity(base.len() + snippet.len());
    out.push_str(&base[..b]);
    out.push_str(snippet);
    out.push_str(&base[e_end..]);
    out
}

/// XS.4 — storage buffers per shader stage needed for GPU sprite shadows. The
/// sprite pass binds its own 14 + the terrain occupancy set (occupancy pages
/// 0..3, chunk occupancy, slot index, grid meta, per-grid cameras) to march
/// terrain shadows. Devices granting fewer fall back to unshadowed GPU sprites.
pub(crate) const SPRITE_SHADOW_MIN_STORAGE_BUFFERS: u32 = 22;

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

/// World-space view-ray direction (un-normalised) for window pixel
/// `(x, y)` under a vertical-FOV pinhole — the projection
/// `scene_dda.wgsl`'s `render_scene` uses. Shared by
/// [`GpuRenderer::pixel_ray`]; standalone so it's unit-testable without
/// a device. `right`/`down`/`forward` are the camera basis.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn pinhole_pixel_ray(
    right: [f64; 3],
    down: [f64; 3],
    forward: [f64; 3],
    x: f64,
    y: f64,
    w: f64,
    h: f64,
    fov_y_rad: f64,
) -> [f64; 3] {
    let half_h = (fov_y_rad * 0.5).tan();
    let half_w = half_h * (w / h);
    let ndc_x = (x + 0.5) / w * 2.0 - 1.0;
    let ndc_y_top = 1.0 - (y + 0.5) / h * 2.0;
    let (kx, ky) = (ndc_x * half_w, ndc_y_top * half_h);
    [
        forward[0] + kx * right[0] - ky * down[0],
        forward[1] + kx * right[1] - ky * down[1],
        forward[2] + kx * right[2] - ky * down[2],
    ]
}

#[cfg(test)]
mod pixel_ray_tests {
    use super::pinhole_pixel_ray;

    const RIGHT: [f64; 3] = [1.0, 0.0, 0.0];
    const DOWN: [f64; 3] = [0.0, 1.0, 0.0];
    const FWD: [f64; 3] = [0.0, 0.0, 1.0]; // voxlap z-down "look down"

    // Frame centre (NDC 0,0) points straight along `forward`.
    #[test]
    fn centre_pixel_is_forward() {
        let d = pinhole_pixel_ray(
            RIGHT,
            DOWN,
            FWD,
            639.5,
            359.5,
            1280.0,
            720.0,
            60_f64.to_radians(),
        );
        assert!(
            d[0].abs() < 1e-9 && d[1].abs() < 1e-9,
            "centre ≈ forward, got {d:?}"
        );
        assert!((d[2] - 1.0).abs() < 1e-9);
    }

    // Right edge pixel tilts +right by tan(hfov/2); the lateral
    // component equals half_w = tan(fov_y/2)*aspect at the very edge.
    #[test]
    fn right_edge_tilts_by_half_w() {
        let fov = 60_f64.to_radians();
        let d = pinhole_pixel_ray(RIGHT, DOWN, FWD, 1279.5, 359.5, 1280.0, 720.0, fov);
        let half_w = (fov * 0.5).tan() * (1280.0 / 720.0);
        assert!((d[0] - half_w).abs() < 1e-6, "x={}, half_w={half_w}", d[0]);
        assert!(d[0] > 0.0, "right edge tilts +right");
    }

    /// Statically validate every WGSL shader with naga (the same
    /// front-end + validator wgpu runs at pipeline creation), so shader
    /// edits — e.g. the GPU.10 sprite lighting bindings — are caught in
    /// CI without needing a GPU device.
    #[test]
    fn wgsl_shaders_validate() {
        let shaders: &[(&str, &str)] = &[
            (
                "sprite_model_dda.wgsl",
                include_str!("../shaders/sprite_model_dda.wgsl"),
            ),
            ("scene_dda.wgsl", include_str!("../shaders/scene_dda.wgsl")),
            ("blit.wgsl", include_str!("../shaders/blit.wgsl")),
            ("chunk_dda.wgsl", include_str!("../shaders/chunk_dda.wgsl")),
            ("grid_dda.wgsl", include_str!("../shaders/grid_dda.wgsl")),
            (
                "scene_blit.wgsl",
                include_str!("../shaders/scene_blit.wgsl"),
            ),
            (
                "scene_resolve.wgsl",
                include_str!("../shaders/scene_resolve.wgsl"),
            ),
            ("line.wgsl", include_str!("../shaders/line.wgsl")),
            ("image.wgsl", include_str!("../shaders/image.wgsl")),
        ];
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        for (name, src) in shaders {
            let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| {
                panic!("{name}: WGSL parse failed:\n{}", e.emit_to_string(src))
            });
            validator
                .validate(&module)
                .unwrap_or_else(|e| panic!("{name}: WGSL validation failed: {e:?}"));
        }
        // XS.4.2 — the raw `sprite_model_dda.wgsl` above is the unshadowed STUB
        // variant; also validate the sprite-shadow-CAPABLE spliced variant (the
        // terrain-shadow snippet injected) that capable devices build.
        let capable = super::sprite_shader_source(true);
        let module = naga::front::wgsl::parse_str(&capable).unwrap_or_else(|e| {
            panic!(
                "sprite_model_dda.wgsl (capable): parse failed:\n{}",
                e.emit_to_string(&capable)
            )
        });
        validator.validate(&module).unwrap_or_else(|e| {
            panic!("sprite_model_dda.wgsl (capable): validation failed: {e:?}")
        });
        // XS.4.3 — the capable scene variant (sprite-cast snippet spliced in).
        let scene_cap = super::scene_shader_source(true);
        let module = naga::front::wgsl::parse_str(&scene_cap).unwrap_or_else(|e| {
            panic!(
                "scene_dda.wgsl (capable): parse failed:\n{}",
                e.emit_to_string(&scene_cap)
            )
        });
        validator
            .validate(&module)
            .unwrap_or_else(|e| panic!("scene_dda.wgsl (capable): validation failed: {e:?}"));
    }

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
        let verts = crate::build_image_vertices(&cam, &quad, 800, 600, 60_f32.to_radians(), false);
        assert_eq!(verts.len(), 6, "two triangles, no near-clip");
        for v in &verts {
            assert!((v.w - 10.0).abs() < 1e-4, "w == forward distance");
            assert!(v.depth >= 10.0, "euclidean depth >= forward distance");
            assert_eq!(v.depth_test, 1.0);
        }
    }
}
