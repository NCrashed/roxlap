//! QE.8b — dynamic-lighting upload, split verbatim out of `lib.rs`
//! (stages DL + SL): the [`SceneLights`] per-frame rig, the packed
//! [`GpuPointLight`] GPU mirror, the pack/upload helpers shared by
//! the surface + headless paths, and the `set_scene_lights` entry
//! points on both renderers.

use bytemuck::{Pod, Zeroable};

use crate::{GpuRenderer, HeadlessSceneRenderer, SceneDdaPerGridCamera};

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
pub(crate) struct GpuPointLight {
    pub(crate) pos: [f32; 3],
    pub(crate) radius: f32,
    pub(crate) color: [f32; 3],
    pub(crate) intensity: f32,
    // SL — spot (cone) fields. `cos_outer == -1.0` ⇒ omnidirectional point.
    pub(crate) spot_dir: [f32; 3],
    pub(crate) cos_outer: f32,
    pub(crate) cos_inner: f32,
    pub(crate) casts_shadow: u32,
    pub(crate) _pad: [u32; 2],
}

// SL — the std430 stride must stay locked to the WGSL `PointLight` (four
// `vec4<f32>`); a mismatch silently corrupts every light past the first.
const _: () = assert!(core::mem::size_of::<GpuPointLight>() == 64);

/// Build the per-grid point-light storage buffer (binding 18), grid-major:
/// grid `g`'s lights occupy `[g*count .. (g+1)*count]`. Pads to one zeroed
/// element when empty (wgpu rejects a zero-sized storage binding).
pub(crate) fn upload_grid_point_lights(
    device: &wgpu::Device,
    lights: &[GpuPointLight],
) -> wgpu::Buffer {
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
pub(crate) fn inject_grid_sun_dirs(lights: &SceneLights, cam_vec: &mut [SceneDdaPerGridCamera]) {
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
pub(crate) fn pack_scene_lights(
    lights: &SceneLights,
    grid_count: usize,
) -> (Vec<GpuPointLight>, u32, u32) {
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

impl GpuRenderer {
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
            self.dirty.mark_lights_changed();
        }
    }
}

impl HeadlessSceneRenderer {
    /// DL — set dynamic lights for subsequent [`Self::render`] calls
    /// (already in grid-local space). Lets tests exercise the lit path
    /// (sun N·L, point lights). Default = none (baked-only).
    pub fn set_scene_lights(&mut self, lights: SceneLights) {
        self.lights = lights;
    }
}
