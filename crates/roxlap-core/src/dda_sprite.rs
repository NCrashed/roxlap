//! Clean-room KV6 sprite raycaster for the DDA backend (Substage
//! DDA.8).
//!
//! Renders KV6 sprites by **per-pixel ray casting**: for every screen
//! pixel the sprite covers, transform the camera ray into the sprite's
//! local voxel space, 3D-DDA through the KV6, and depth-composite the
//! first solid voxel against the shared z-buffer. Clean-room (no voxlap
//! code), the sprite counterpart to the terrain renderer in
//! [`crate::dda`].
//!
//! **Depth parity.** Transforming the ray by the inverse sprite basis
//! leaves the ray parameter unchanged in world units — a hit at local
//! parameter `t` is at world point `cam.pos + dir·t` — so the
//! perpendicular depth is `t · (dir·forward)`, exactly the convention
//! [`crate::dda`] writes for terrain. Sprites therefore occlude and are
//! occluded by DDA terrain correctly.
//!
//! Shading reads the KV6 voxel's baked brightness byte (high byte of
//! the packed colour) via [`crate::dda::shade`] — the clean-room
//! brightness model, not voxlap's `dir`-LUT reflection shading.

use roxlap_formats::kv6::Kv6;
use roxlap_formats::material::{material_for_color, BlendMode, MaterialTable};
use roxlap_formats::sprite::{Sprite, SPRITE_FLAG_INVISIBLE, SPRITE_FLAG_NO_Z};
use roxlap_formats::voxel_clip::{DecodedClip, VoxelFrame};

use crate::camera_math::CameraState;
use crate::dda::{dda_setup, intersect_aabb, min_axis, pixel_ray, shade, shade_dynamic, CpuLights};
use crate::opticast::OpticastSettings;
use crate::raster_target::RasterTarget;

/// Near-plane parameter: voxels nearer than this (camera-forward) are
/// dropped, keeping the pinhole divide finite.
const NEAR_Z: f32 = 1.0;

/// Force a packed voxel colour to full brightness for the flat-lit
/// clean-room sprite path. KV6 / voxel-clip colours carry voxlap's
/// `dir`/shading slot in the high byte (some `0x80`, some `0x00`), not
/// the 0..128 brightness [`shade`] expects, so a raw value can render
/// black; we render every sprite voxel at its authored RGB.
#[inline]
fn full_bright(col: u32) -> u32 {
    (col & 0x00ff_ffff) | 0x8000_0000
}

/// Dense occupancy + colour grid for one sprite frame, plus its pivot —
/// the decoded form the per-pixel raycaster marches. Built once from a
/// [`Kv6`] ([`SpriteDense::from_kv6`]) or a voxel-clip [`VoxelFrame`]
/// ([`SpriteDense::from_voxel_frame`]); the latter lets an animated clip
/// cache every frame's grid up front instead of rebuilding per frame.
///
/// Both sources store only **surface** voxels (a from-air ray's first
/// hit is the visible surface), so the grid is the visible hull.
pub struct SpriteDense {
    dims: [i32; 3],
    occ: Vec<bool>,
    col: Vec<u32>,
    /// Per-voxel material id (TV stage), parallel to [`col`](Self::col) /
    /// [`occ`](Self::occ) (same dense index). **Empty** means every voxel
    /// uses the draw-time uniform material (the TV.1 path); a non-empty
    /// array gives mixed-material models (opaque frame + glass, TV.3). Only
    /// consulted on the [`draw_sprite_dense_shaded`] accumulate path.
    mat: Vec<u8>,
    pivot: [f32; 3],
}

impl SpriteDense {
    /// Decode a [`Kv6`]'s surface-voxel run tables into a dense grid.
    #[must_use]
    #[allow(clippy::cast_possible_wrap)]
    pub fn from_kv6(kv6: &Kv6) -> Self {
        let dims = [kv6.xsiz as i32, kv6.ysiz as i32, kv6.zsiz as i32];
        let n = (dims[0].max(0) * dims[1].max(0) * dims[2].max(0)) as usize;
        let mut occ = vec![false; n];
        let mut col = vec![0u32; n];
        let mut vi = 0usize;
        for x in 0..kv6.xsiz as usize {
            for y in 0..kv6.ysiz as usize {
                let cnt = usize::from(kv6.ylen[x][y]);
                for _ in 0..cnt {
                    let v = kv6.voxels[vi];
                    vi += 1;
                    let z = i32::from(v.z);
                    if z >= 0 && z < dims[2] {
                        let idx = ((x as i32 * dims[1] + y as i32) * dims[2] + z) as usize;
                        occ[idx] = true;
                        col[idx] = full_bright(v.col);
                    }
                }
            }
        }
        Self {
            dims,
            occ,
            col,
            mat: Vec::new(),
            pivot: [kv6.xpiv, kv6.ypiv, kv6.zpiv],
        }
    }

    /// Like [`from_kv6`](Self::from_kv6) but classifies each voxel into a
    /// material id by colour (TV.3 mixed models) via `material_map`
    /// (`(rgb, material_id)` pairs; see
    /// [`material_for_color`](roxlap_formats::material::material_for_color)).
    /// The resulting per-voxel `mat` array is consulted by the
    /// [`draw_sprite_dense_shaded`] accumulate path. An empty map yields the
    /// same all-opaque (uniform) result as `from_kv6`.
    #[must_use]
    #[allow(clippy::cast_possible_wrap)]
    pub fn from_kv6_with_materials(kv6: &Kv6, material_map: &[(u32, u8)]) -> Self {
        let mut dense = Self::from_kv6(kv6);
        if !material_map.is_empty() {
            let n = dense.col.len();
            let mut mat = vec![0u8; n];
            for (idx, slot) in mat.iter_mut().enumerate() {
                if dense.occ[idx] {
                    *slot = material_for_color(material_map, dense.col[idx]);
                }
            }
            dense.mat = mat;
        }
        dense
    }

    /// Decode a voxel-clip [`VoxelFrame`] (dense-column layout) into the
    /// dense grid, given the clip's `dims` + `pivot`. The frame's columns
    /// are `col = x + y*dims[0]`, each a per-column occupancy bitmask with
    /// an ascending-z colour run — walked here into the raycaster's
    /// `(x·my + y)·mz + z` grid.
    #[must_use]
    #[allow(clippy::cast_possible_wrap)]
    pub fn from_voxel_frame(frame: &VoxelFrame, dims: [u32; 3], pivot: [f32; 3]) -> Self {
        let (mx, my, mz) = (dims[0], dims[1], dims[2]);
        let owpc = mz.div_ceil(32).max(1) as usize;
        let n = (mx * my * mz) as usize;
        let mut occ = vec![false; n];
        let mut col = vec![0u32; n];
        for col_idx in 0..(mx * my) as usize {
            let x = col_idx as u32 % mx;
            let y = col_idx as u32 / mx;
            let run_start = frame.color_offsets[col_idx] as usize;
            let mut k = 0usize;
            for z in 0..mz {
                let word = frame.occupancy[col_idx * owpc + (z >> 5) as usize];
                if (word >> (z & 31)) & 1 != 0 {
                    let idx = (((x * my + y) * mz) + z) as usize;
                    occ[idx] = true;
                    col[idx] = full_bright(frame.colors[run_start + k]);
                    k += 1;
                }
            }
        }
        Self {
            dims: [mx as i32, my as i32, mz as i32],
            occ,
            col,
            mat: Vec::new(),
            pivot,
        }
    }

    /// Like [`from_voxel_frame`](Self::from_voxel_frame) but classifies each
    /// voxel into a material id by colour (TV.3 mixed models) via
    /// `material_map` — the clip analogue of
    /// [`from_kv6_with_materials`](Self::from_kv6_with_materials). An empty
    /// map yields the same all-opaque (uniform) result as `from_voxel_frame`.
    #[must_use]
    pub fn from_voxel_frame_with_materials(
        frame: &VoxelFrame,
        dims: [u32; 3],
        pivot: [f32; 3],
        material_map: &[(u32, u8)],
    ) -> Self {
        let mut dense = Self::from_voxel_frame(frame, dims, pivot);
        if !material_map.is_empty() {
            let n = dense.col.len();
            let mut mat = vec![0u8; n];
            for (idx, slot) in mat.iter_mut().enumerate() {
                if dense.occ[idx] {
                    *slot = material_for_color(material_map, dense.col[idx]);
                }
            }
            dense.mat = mat;
        }
        dense
    }

    #[inline]
    #[allow(clippy::cast_sign_loss)]
    fn idx_of(&self, c: [i32; 3]) -> usize {
        ((c[0] * self.dims[1] + c[1]) * self.dims[2] + c[2]) as usize
    }

    #[inline]
    fn at(&self, c: [i32; 3]) -> Option<u32> {
        let idx = self.idx_of(c);
        self.occ[idx].then(|| self.col[idx])
    }
}

/// Inverse of the column-matrix `[s | h | f]` (the sprite basis), or
/// `None` if degenerate. Maps a world delta into local voxel space.
fn invert_basis(s: [f32; 3], h: [f32; 3], f: [f32; 3]) -> Option<[[f32; 3]; 3]> {
    let det = s[0] * (h[1] * f[2] - f[1] * h[2]) - h[0] * (s[1] * f[2] - f[1] * s[2])
        + f[0] * (s[1] * h[2] - h[1] * s[2]);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv = 1.0 / det;
    Some([
        [
            (h[1] * f[2] - f[1] * h[2]) * inv,
            -(h[0] * f[2] - f[0] * h[2]) * inv,
            (h[0] * f[1] - f[0] * h[1]) * inv,
        ],
        [
            -(s[1] * f[2] - f[1] * s[2]) * inv,
            (s[0] * f[2] - f[0] * s[2]) * inv,
            -(s[0] * f[1] - f[0] * s[1]) * inv,
        ],
        [
            (s[1] * h[2] - h[1] * s[2]) * inv,
            -(s[0] * h[2] - h[0] * s[2]) * inv,
            (s[0] * h[1] - h[0] * s[1]) * inv,
        ],
    ])
}

#[inline]
fn mat_apply(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// Cast one ray (already in the sprite's local voxel space) into the
/// dense KV6 and return `(colour, t)` of the first solid voxel — `t` is
/// the world-units ray parameter (shared with the world ray).
#[allow(clippy::cast_possible_truncation)]
/// First solid voxel along the local ray. Returns `(color, t, normal_local,
/// cell)`: `normal_local` is the **model-local** face normal of the hit
/// (points back toward the ray; zero for the entry voxel, no face crossed) —
/// the caller rotates it to world for dynamic lighting (DL.7); `cell` is the
/// hit voxel for the flat-per-voxel world centre.
fn cast_local(
    dense: &SpriteDense,
    origin: [f32; 3],
    dir: [f32; 3],
) -> Option<(u32, f32, [f32; 3], [i32; 3])> {
    #[allow(clippy::cast_precision_loss)]
    let hi = [
        dense.dims[0] as f32,
        dense.dims[1] as f32,
        dense.dims[2] as f32,
    ];
    let (t0, t1) = intersect_aabb(origin, dir, [0.0; 3], hi)?;
    let start = t0 + 1e-4;
    let p = [
        origin[0] + dir[0] * start,
        origin[1] + dir[1] * start,
        origin[2] + dir[2] * start,
    ];
    let mut cell = [
        (p[0].floor() as i32).clamp(0, dense.dims[0] - 1),
        (p[1].floor() as i32).clamp(0, dense.dims[1] - 1),
        (p[2].floor() as i32).clamp(0, dense.dims[2] - 1),
    ];
    let (step, mut t_max, t_delta) = dda_setup(origin, dir, cell, 1.0);
    let mut t_curr = t0;
    // Face crossed to reach the current cell (model-local normal). The entry
    // voxel (solid at t0, no step yet) has none → zero normal.
    let mut normal = [0.0f32; 3];
    let max_steps = (dense.dims[0] + dense.dims[1] + dense.dims[2]) as usize + 8;
    for _ in 0..max_steps {
        if cell[0] < 0
            || cell[0] >= dense.dims[0]
            || cell[1] < 0
            || cell[1] >= dense.dims[1]
            || cell[2] < 0
            || cell[2] >= dense.dims[2]
            || t_curr > t1
        {
            return None;
        }
        if let Some(color) = dense.at(cell) {
            return Some((color, t_curr, normal, cell));
        }
        let axis = min_axis(t_max);
        t_curr = t_max[axis];
        cell[axis] += step[axis];
        t_max[axis] += t_delta[axis];
        normal = [0.0; 3];
        normal[axis] = -(step[axis] as f32);
    }
    None
}

/// Material context for a translucent sprite draw (TV stage): the global
/// [`MaterialTable`] plus this instance's uniform material id and per-frame
/// alpha multiplier. Passed (as `Some`) to [`draw_sprite_dense_shaded`] /
/// [`ClipFlipbook::draw_frame_shaded`] to enable front-to-back
/// accumulate-and-continue compositing; `None` (or an all-opaque effective
/// material) takes the existing first-hit opaque path byte-for-byte.
#[derive(Clone, Copy)]
pub struct SpriteShade<'a> {
    /// Global voxel-material palette (per-voxel id → opacity + blend mode).
    pub materials: &'a MaterialTable,
    /// Uniform material id for every voxel of this sprite whose dense
    /// per-voxel `mat` array is empty (the TV.1 whole-sprite material).
    pub material: u8,
    /// Per-instance opacity multiplier (`255` = unscaled), so an effect can
    /// fade out by cheap per-frame updates without re-uploading the volume.
    pub alpha_mul: u8,
    /// DL.7 — world-space dynamic lights. When `enabled`, the opaque hit is
    /// lit (sun + point lights + cel + ramp, flat per voxel) instead of the
    /// baked `shade`. `CpuLights::default()` (disabled) ⇒ unchanged.
    pub lights: CpuLights<'a>,
}

/// Accumulated front-to-back composite for one ray through a sprite.
struct LayerAccum {
    /// Premultiplied accumulated colour, channels in `0..=~1` (additive may
    /// exceed 1; clamped at pack time).
    rgb: [f32; 3],
    /// Remaining transmittance (starts 1.0, decays through `AlphaBlend`).
    trans: f32,
    /// The opaque/background hit that terminated the march, if any: its
    /// already-shaded packed colour + world-ray parameter `t`. `None` if the
    /// ray exited (or fully attenuated) without an opaque voxel — then the
    /// background is whatever the framebuffer already holds (terrain/sky).
    opaque: Option<(u32, f32)>,
}

/// Unpack a packed `0x..RRGGBB` colour to linear-ish `0..1` float channels
/// (RGB only; the high byte is ignored here — sprite voxels are flat-lit).
#[inline]
fn rgb_to_f32(c: u32) -> [f32; 3] {
    [
        ((c >> 16) & 0xff) as f32 / 255.0,
        ((c >> 8) & 0xff) as f32 / 255.0,
        (c & 0xff) as f32 / 255.0,
    ]
}

/// Repack `0..1` float channels (clamped) into `0x80RRGGBB` — the
/// full-brightness packing the flat-lit sprite path writes.
#[inline]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn f32_to_rgb(c: [f32; 3]) -> u32 {
    let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
    0x8000_0000 | (q(c[0]) << 16) | (q(c[1]) << 8) | q(c[2])
}

/// Cast one ray (in sprite-local voxel space) accumulating translucent
/// voxels front-to-back until an opaque voxel, transmittance exhaustion, or
/// the `max_t` cutoff (the terrain depth, so the march stops at geometry it
/// can't see past). `fwd_dot = dir·camera-forward` converts the ray
/// parameter to perpendicular depth. Returns `None` if the ray contributes
/// nothing (missed the box, or every voxel was clipped / behind terrain).
#[allow(clippy::cast_possible_truncation, clippy::too_many_arguments)]
fn cast_local_layers(
    dense: &SpriteDense,
    origin: [f32; 3],
    dir: [f32; 3],
    fwd_dot: f32,
    max_t: f32,
    shade_ctx: SpriteShade,
) -> Option<LayerAccum> {
    #[allow(clippy::cast_precision_loss)]
    let hi = [
        dense.dims[0] as f32,
        dense.dims[1] as f32,
        dense.dims[2] as f32,
    ];
    let (t0, t1) = intersect_aabb(origin, dir, [0.0; 3], hi)?;
    let start = t0 + 1e-4;
    let p = [
        origin[0] + dir[0] * start,
        origin[1] + dir[1] * start,
        origin[2] + dir[2] * start,
    ];
    let mut cell = [
        (p[0].floor() as i32).clamp(0, dense.dims[0] - 1),
        (p[1].floor() as i32).clamp(0, dense.dims[1] - 1),
        (p[2].floor() as i32).clamp(0, dense.dims[2] - 1),
    ];
    let (step, mut t_max, t_delta) = dda_setup(origin, dir, cell, 1.0);
    let mut t_curr = t0;
    let max_steps = (dense.dims[0] + dense.dims[1] + dense.dims[2]) as usize + 8;

    let mut acc = LayerAccum {
        rgb: [0.0; 3],
        trans: 1.0,
        opaque: None,
    };
    let mut touched = false;
    // Per-span compositing: a translucent voxel contributes one alpha layer
    // only when the ray *enters* a contiguous solid run (the previous cell
    // was air). Without this, a ray clipping the shared boundary between two
    // adjacent surface voxels passes through both and double-composites a thin
    // strip — the model reads as "diced" by a voxel grid. Treating each solid
    // run as one surface makes a wall contribute exactly one alpha regardless
    // of how many of its voxels the ray grazes. A run is also re-entered on a
    // material change (TV.3: two adjacent translucent materials each count),
    // and an opaque voxel stops the ray on every cell (the opaque core of a
    // mixed model).
    let mut prev_solid = false;
    let mut prev_mat = 0u8;
    // Local ray length per ray-parameter unit — converts a cell's `t` span to
    // its path length in voxel units for the `Volumetric` Beer–Lambert weight.
    let dir_len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();

    for _ in 0..max_steps {
        if cell[0] < 0
            || cell[0] >= dense.dims[0]
            || cell[1] < 0
            || cell[1] >= dense.dims[1]
            || cell[2] < 0
            || cell[2] >= dense.dims[2]
            || t_curr > t1
        {
            break;
        }
        // Stop at the terrain depth: everything past it is occluded, and the
        // already-drawn framebuffer pixel becomes the background.
        let depth = t_curr * fwd_dot;
        if depth >= max_t {
            break;
        }
        // Exit `t` of the current cell — the next boundary crossing. Its span
        // from `t_curr` is the ray's path through this cell (Volumetric).
        let exit_axis = min_axis(t_max);
        let t_exit = t_max[exit_axis];
        let idx = dense.idx_of(cell);
        let solid_here = dense.occ[idx];
        if solid_here && depth >= NEAR_Z {
            let mat_id = if dense.mat.is_empty() {
                shade_ctx.material
            } else {
                dense.mat[idx]
            };
            let m = shade_ctx.materials.get(mat_id);
            if m.is_opaque() {
                acc.opaque = Some((shade(dense.col[idx], 0), t_curr));
                touched = true;
                break;
            }
            let a = f32::from(m.alpha) / 255.0 * (f32::from(shade_ctx.alpha_mul) / 255.0);
            if m.mode == BlendMode::Volumetric {
                // Per-cell Beer–Lambert: opacity weighted by traversed length
                // (in voxel units), so a thin sliver contributes ≈0 and a
                // filled volume thickens smoothly with depth. Always occludes.
                let seg_len = (t_exit - t_curr).max(0.0) * dir_len;
                let eff_a = 1.0 - (1.0 - a).powf(seg_len);
                let lit = rgb_to_f32(shade(dense.col[idx], 0));
                acc.rgb[0] += acc.trans * eff_a * lit[0];
                acc.rgb[1] += acc.trans * eff_a * lit[1];
                acc.rgb[2] += acc.trans * eff_a * lit[2];
                acc.trans *= 1.0 - eff_a;
                touched = true;
                prev_mat = mat_id;
                if acc.trans < 1.0 / 256.0 {
                    break;
                }
            } else if !prev_solid || mat_id != prev_mat {
                // AlphaBlend / Additive: one alpha layer per solid-run entry or
                // material change (thickness-independent — shells, glass).
                let lit = rgb_to_f32(shade(dense.col[idx], 0));
                acc.rgb[0] += acc.trans * a * lit[0];
                acc.rgb[1] += acc.trans * a * lit[1];
                acc.rgb[2] += acc.trans * a * lit[2];
                if m.mode == BlendMode::AlphaBlend {
                    acc.trans *= 1.0 - a; // Additive glow does not occlude.
                }
                touched = true;
                prev_mat = mat_id;
                if acc.trans < 1.0 / 256.0 {
                    break;
                }
            }
        }
        prev_solid = solid_here;
        t_curr = t_exit;
        cell[exit_axis] += step[exit_axis];
        t_max[exit_axis] += t_delta[exit_axis];
    }

    touched.then_some(acc)
}

/// Draw one KV6 [`Sprite`] into `(fb, zb)` by per-pixel ray casting,
/// depth-compositing against whatever the terrain pass already wrote.
/// Returns the number of pixels written.
///
/// `cam` / `settings` are the **same** per-frame projection the DDA
/// terrain pass used (build via [`crate::camera_math::derive`]), so
/// sprite and terrain share one pinhole and z convention. `pitch_pixels`
/// is the framebuffer row stride. Honours `SPRITE_FLAG_INVISIBLE`
/// (skip) and `SPRITE_FLAG_NO_Z` (write without the depth test).
#[allow(
    clippy::too_many_arguments,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
#[must_use]
pub fn draw_sprite_dda(
    fb: &mut [u32],
    zb: &mut [f32],
    pitch_pixels: usize,
    width: u32,
    height: u32,
    cam: &CameraState,
    settings: &OpticastSettings,
    sprite: &Sprite,
) -> u32 {
    if sprite.flags & SPRITE_FLAG_INVISIBLE != 0 {
        return 0;
    }
    draw_sprite_dda_shaded(
        fb,
        zb,
        pitch_pixels,
        width,
        height,
        cam,
        settings,
        sprite,
        None,
    )
}

/// Draw one KV6 [`Sprite`], optionally with a translucent material (TV
/// stage) — the [`draw_sprite_dense_shaded`] counterpart of
/// [`draw_sprite_dda`]. `shade_ctx == None` (or an opaque effective
/// material) renders the sprite opaque, byte-for-byte unchanged.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn draw_sprite_dda_shaded(
    fb: &mut [u32],
    zb: &mut [f32],
    pitch_pixels: usize,
    width: u32,
    height: u32,
    cam: &CameraState,
    settings: &OpticastSettings,
    sprite: &Sprite,
    shade_ctx: Option<SpriteShade>,
) -> u32 {
    if sprite.flags & SPRITE_FLAG_INVISIBLE != 0 {
        return 0;
    }
    // Decodes the KV6 to a dense grid each call (the per-frame cost an
    // animated clip avoids via [`ClipFlipbook`]'s cached grids). A non-empty
    // `material_map` classifies voxels into per-voxel materials (TV.3 mixed
    // models); an empty one is the plain uniform-material decode.
    let dense = if sprite.material_map.is_empty() {
        SpriteDense::from_kv6(&sprite.kv6)
    } else {
        SpriteDense::from_kv6_with_materials(&sprite.kv6, &sprite.material_map)
    };
    draw_sprite_dense_shaded(
        fb,
        zb,
        pitch_pixels,
        width,
        height,
        cam,
        settings,
        &dense,
        sprite.p,
        sprite.s,
        sprite.h,
        sprite.f,
        sprite.flags,
        shade_ctx,
    )
}

/// Draw a pre-decoded [`SpriteDense`] at a world pose — the generalised
/// core of [`draw_sprite_dda`], shared by the KV6 path and animated
/// [`ClipFlipbook`] frames. `pos` is the world pivot; `s`/`h`/`f` are the
/// model→world basis columns (local +x/+y/+z); `flags` honours
/// [`SPRITE_FLAG_INVISIBLE`] / [`SPRITE_FLAG_NO_Z`]. Returns pixels written.
///
/// Fully opaque (the existing first-hit path). For translucent sprites use
/// [`draw_sprite_dense_shaded`].
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn draw_sprite_dense(
    fb: &mut [u32],
    zb: &mut [f32],
    pitch_pixels: usize,
    width: u32,
    height: u32,
    cam: &CameraState,
    settings: &OpticastSettings,
    dense: &SpriteDense,
    pos: [f32; 3],
    s: [f32; 3],
    h: [f32; 3],
    f: [f32; 3],
    flags: u32,
) -> u32 {
    draw_sprite_dense_shaded(
        fb,
        zb,
        pitch_pixels,
        width,
        height,
        cam,
        settings,
        dense,
        pos,
        s,
        h,
        f,
        flags,
        None,
    )
}

/// Draw a pre-decoded [`SpriteDense`] at a world pose, optionally with a
/// translucent material (TV stage). `shade_ctx`:
/// - `None`, or a `Some` whose effective material is opaque ⇒ the existing
///   first-hit, depth-tested opaque path, **byte-for-byte unchanged**.
/// - `Some` with a translucent uniform material (or a non-empty per-voxel
///   `mat` array) ⇒ front-to-back accumulate-and-continue: each ray marches
///   through the sprite compositing `AlphaBlend`/`Additive` layers over what
///   lies behind (terrain/sky already in the framebuffer, or an opaque voxel
///   of the model). Opaque voxels write the model surface depth; purely
///   translucent pixels composite over the framebuffer without touching the
///   z-buffer (they do not occlude). See `PORTING-TRANSPARENCY.md`.
#[allow(
    clippy::too_many_arguments,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
#[must_use]
pub fn draw_sprite_dense_shaded(
    fb: &mut [u32],
    zb: &mut [f32],
    pitch_pixels: usize,
    width: u32,
    height: u32,
    cam: &CameraState,
    settings: &OpticastSettings,
    dense: &SpriteDense,
    pos: [f32; 3],
    s: [f32; 3],
    h: [f32; 3],
    f: [f32; 3],
    flags: u32,
    shade_ctx: Option<SpriteShade>,
) -> u32 {
    if flags & SPRITE_FLAG_INVISIBLE != 0 || dense.occ.is_empty() {
        return 0;
    }
    let Some(minv) = invert_basis(s, h, f) else {
        return 0;
    };
    let pivot = dense.pivot;
    let no_z = flags & SPRITE_FLAG_NO_Z != 0;

    // Screen bounding box from the 8 corners of the local voxel box.
    let Some(rect) = project_screen_rect(dense, pos, s, h, f, cam, settings, width, height) else {
        return 0;
    };

    // Per-sprite gate: a sprite whose effective material is opaque (the
    // common case, and every sprite while no translucent material is
    // defined) takes the original loop unchanged — so the opaque world stays
    // bit-identical. Only a genuinely translucent sprite runs the accumulate
    // loop.
    let layers =
        shade_ctx.filter(|s| !dense.mat.is_empty() || !s.materials.get(s.material).is_opaque());

    debug_assert_eq!(fb.len(), zb.len());
    let target = RasterTarget::new(fb, zb);
    let mut written = 0u32;
    for py in rect.1..rect.3 {
        let row = py as usize * pitch_pixels;
        for px in rect.0..rect.2 {
            let (origin, dir) = pixel_ray(cam, settings, px, py);
            // World ray → sprite-local voxel space.
            let rel = [origin[0] - pos[0], origin[1] - pos[1], origin[2] - pos[2]];
            let ol = mat_apply(&minv, rel);
            let origin_local = [ol[0] + pivot[0], ol[1] + pivot[1], ol[2] + pivot[2]];
            let dir_local = mat_apply(&minv, dir);
            let fwd_dot =
                dir[0] * cam.forward[0] + dir[1] * cam.forward[1] + dir[2] * cam.forward[2];
            let idx = row + px as usize;

            if let Some(shade_ctx) = layers {
                // ---- translucent: accumulate front-to-back ----
                if fwd_dot <= 1e-6 {
                    continue;
                }
                // Terrain/opaque depth cutoff (perpendicular distance, the
                // same units `cast_local_layers` compares `t_curr·fwd_dot`
                // against — NOT a ray parameter, since the CPU `dir` is
                // unnormalised). SAFETY: idx in rect ⊂ (width,height).
                let max_t = if no_z {
                    f32::INFINITY
                } else {
                    unsafe { target.read_depth(idx) }
                };
                let Some(acc) =
                    cast_local_layers(dense, origin_local, dir_local, fwd_dot, max_t, shade_ctx)
                else {
                    continue;
                };
                // SAFETY: idx in bounds; single-threaded writer.
                let wrote = unsafe {
                    match acc.opaque {
                        Some((bg_color, t)) => {
                            // Opaque model surface behind the translucent
                            // layers: composite over it, write surface depth.
                            let bg = rgb_to_f32(bg_color);
                            let out = f32_to_rgb([
                                acc.rgb[0] + acc.trans * bg[0],
                                acc.rgb[1] + acc.trans * bg[1],
                                acc.rgb[2] + acc.trans * bg[2],
                            ]);
                            let depth = t * fwd_dot;
                            if no_z {
                                target.write_color(idx, out);
                                target.write_depth(idx, depth);
                                true
                            } else {
                                target.z_test_write(idx, out, depth)
                            }
                        }
                        None => {
                            // Ray exited (or fully attenuated) with no opaque
                            // model voxel: composite over the framebuffer
                            // (terrain/sky). Translucent layers do not occlude,
                            // so the z-buffer is left untouched.
                            let bg = rgb_to_f32(target.read_color(idx));
                            let out = f32_to_rgb([
                                acc.rgb[0] + acc.trans * bg[0],
                                acc.rgb[1] + acc.trans * bg[1],
                                acc.rgb[2] + acc.trans * bg[2],
                            ]);
                            target.write_color(idx, out);
                            true
                        }
                    }
                };
                written += u32::from(wrote);
            } else {
                // ---- opaque: first-hit path ----
                let Some((color, t, n_local, cell)) = cast_local(dense, origin_local, dir_local)
                else {
                    continue;
                };
                let depth = t * fwd_dot;
                if depth < NEAR_Z {
                    continue;
                }
                // DL.7 — dynamic lighting when a rig is active (sun + point
                // lights + cel + ramp, flat per voxel); else the baked `shade`
                // (byte-identical). The model-local face normal + voxel centre
                // are rotated into world space via the instance basis (s,h,f).
                let dl = shade_ctx.map_or(CpuLights::default(), |s| s.lights);
                let lit = if dl.enabled {
                    let to_world = |v: [f32; 3]| {
                        [
                            v[0] * s[0] + v[1] * h[0] + v[2] * f[0],
                            v[0] * s[1] + v[1] * h[1] + v[2] * f[1],
                            v[0] * s[2] + v[1] * h[2] + v[2] * f[2],
                        ]
                    };
                    let n_world = to_world(n_local);
                    let rel = [
                        cell[0] as f32 + 0.5 - pivot[0],
                        cell[1] as f32 + 0.5 - pivot[1],
                        cell[2] as f32 + 0.5 - pivot[2],
                    ];
                    let wc = to_world(rel);
                    let center = [pos[0] + wc[0], pos[1] + wc[1], pos[2] + wc[2]];
                    let albedo = [
                        ((color >> 16) & 0xff) as f32 / 255.0,
                        ((color >> 8) & 0xff) as f32 / 255.0,
                        (color & 0xff) as f32 / 255.0,
                    ];
                    shade_dynamic(albedo, 1.0, n_world, center, &dl)
                } else {
                    shade(color, 0)
                };
                // SAFETY: idx in-bounds for the rect within (width, height);
                // single-threaded writer.
                let wrote = unsafe {
                    if no_z {
                        target.write_color(idx, lit);
                        target.write_depth(idx, depth);
                        true
                    } else {
                        target.z_test_write(idx, lit, depth)
                    }
                };
                written += u32::from(wrote);
            }
        }
    }
    written
}

/// Project the sprite's local voxel AABB to a clamped screen rectangle
/// `(x0, y0, x1, y1)` (half-open). `None` if it can't appear; falls back
/// to the full viewport when the box straddles the near plane (rare).
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn project_screen_rect(
    dense: &SpriteDense,
    pos: [f32; 3],
    s: [f32; 3],
    h: [f32; 3],
    f: [f32; 3],
    cam: &CameraState,
    settings: &OpticastSettings,
    width: u32,
    height: u32,
) -> Option<(u32, u32, u32, u32)> {
    let (xs, ys, zs) = (
        dense.dims[0] as f32,
        dense.dims[1] as f32,
        dense.dims[2] as f32,
    );
    let (xp, yp, zp) = (dense.pivot[0], dense.pivot[1], dense.pivot[2]);
    let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    let mut all_front = true;
    for &cx in &[0.0, xs] {
        for &cy in &[0.0, ys] {
            for &cz in &[0.0, zs] {
                // Local → world via the sprite basis about the pivot.
                let lx = cx - xp;
                let ly = cy - yp;
                let lz = cz - zp;
                let world = [
                    pos[0] + lx * s[0] + ly * h[0] + lz * f[0],
                    pos[1] + lx * s[1] + ly * h[1] + lz * f[1],
                    pos[2] + lx * s[2] + ly * h[2] + lz * f[2],
                ];
                let rel = [
                    world[0] - cam.pos[0],
                    world[1] - cam.pos[1],
                    world[2] - cam.pos[2],
                ];
                let cz_cam =
                    rel[0] * cam.forward[0] + rel[1] * cam.forward[1] + rel[2] * cam.forward[2];
                if cz_cam < NEAR_Z {
                    all_front = false;
                    continue;
                }
                let cx_cam = rel[0] * cam.right[0] + rel[1] * cam.right[1] + rel[2] * cam.right[2];
                let cy_cam = rel[0] * cam.down[0] + rel[1] * cam.down[1] + rel[2] * cam.down[2];
                let sx = settings.hx + cx_cam / cz_cam * settings.hz;
                let sy = settings.hy + cy_cam / cz_cam * settings.hz;
                x0 = x0.min(sx);
                y0 = y0.min(sy);
                x1 = x1.max(sx);
                y1 = y1.max(sy);
            }
        }
    }
    let (w, h) = (width as f32, height as f32);
    let (rx0, ry0, rx1, ry1) = if all_front {
        (
            (x0 - 1.0).max(0.0),
            (y0 - 1.0).max(0.0),
            (x1 + 1.0).min(w),
            (y1 + 1.0).min(h),
        )
    } else {
        // Straddles the near plane → scan the whole viewport.
        (0.0, 0.0, w, h)
    };
    if rx0 >= rx1 || ry0 >= ry1 {
        return None;
    }
    Some((rx0 as u32, ry0 as u32, rx1.ceil() as u32, ry1.ceil() as u32))
}

/// CPU-side decoded animated voxel clip: every frame's [`SpriteDense`]
/// is cached at construction, so per-frame playback is a grid **select**
/// — not the per-frame voxel-volume decode [`draw_sprite_dda`] pays each
/// call. The CPU counterpart to the GPU flipbook (VCL.2). Build once from
/// a [`DecodedClip`], then [`draw_frame`](ClipFlipbook::draw_frame) the
/// active frame each render.
pub struct ClipFlipbook {
    frames: Vec<SpriteDense>,
}

impl ClipFlipbook {
    /// An empty flipbook (no frames) — a tombstone for a removed clip;
    /// [`draw_frame`](Self::draw_frame) always draws nothing.
    #[must_use]
    pub fn empty() -> Self {
        Self { frames: Vec::new() }
    }

    /// Decode + cache every frame of `clip` (one [`SpriteDense`] each).
    #[must_use]
    pub fn from_decoded(clip: &DecodedClip) -> Self {
        Self::from_decoded_with_materials(clip, &[])
    }

    /// Like [`from_decoded`](Self::from_decoded) but classifies every frame's
    /// voxels into per-voxel material ids by colour (TV.3 mixed models) via
    /// `material_map` — the clip analogue of
    /// [`SpriteDense::from_kv6_with_materials`]. An empty map yields the same
    /// all-opaque result as `from_decoded`.
    #[must_use]
    pub fn from_decoded_with_materials(clip: &DecodedClip, material_map: &[(u32, u8)]) -> Self {
        let frames = clip
            .frames
            .iter()
            .map(|frame| {
                SpriteDense::from_voxel_frame_with_materials(
                    frame,
                    clip.dims,
                    clip.pivot,
                    material_map,
                )
            })
            .collect();
        Self { frames }
    }

    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Borrow frame `frame`'s cached dense grid, if in range.
    #[must_use]
    pub fn frame(&self, frame: usize) -> Option<&SpriteDense> {
        self.frames.get(frame)
    }

    /// Replace one frame's cached dense grid in place — the CPU side of an
    /// editor's single-frame edit (no re-decode of the other frames).
    /// Returns `false` if `frame` is out of range.
    pub fn set_frame(&mut self, frame: usize, dense: SpriteDense) -> bool {
        match self.frames.get_mut(frame) {
            Some(slot) => {
                *slot = dense;
                true
            }
            None => false,
        }
    }

    /// Draw frame `frame` at a world pose via [`draw_sprite_dense`] —
    /// `pos` is the world pivot, `s`/`h`/`f` the model→world basis columns.
    /// Returns pixels written (0 if `frame` is out of range).
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn draw_frame(
        &self,
        fb: &mut [u32],
        zb: &mut [f32],
        pitch_pixels: usize,
        width: u32,
        height: u32,
        cam: &CameraState,
        settings: &OpticastSettings,
        frame: usize,
        pos: [f32; 3],
        s: [f32; 3],
        h: [f32; 3],
        f: [f32; 3],
        flags: u32,
    ) -> u32 {
        self.draw_frame_shaded(
            fb,
            zb,
            pitch_pixels,
            width,
            height,
            cam,
            settings,
            frame,
            pos,
            s,
            h,
            f,
            flags,
            None,
        )
    }

    /// Draw frame `frame`, optionally with a translucent material (TV stage)
    /// — the [`draw_sprite_dense_shaded`] counterpart of
    /// [`draw_frame`](Self::draw_frame). `shade_ctx == None` (or an opaque
    /// effective material) renders the frame opaque, unchanged.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn draw_frame_shaded(
        &self,
        fb: &mut [u32],
        zb: &mut [f32],
        pitch_pixels: usize,
        width: u32,
        height: u32,
        cam: &CameraState,
        settings: &OpticastSettings,
        frame: usize,
        pos: [f32; 3],
        s: [f32; 3],
        h: [f32; 3],
        f: [f32; 3],
        flags: u32,
        shade_ctx: Option<SpriteShade>,
    ) -> u32 {
        let Some(dense) = self.frames.get(frame) else {
            return 0;
        };
        draw_sprite_dense_shaded(
            fb,
            zb,
            pitch_pixels,
            width,
            height,
            cam,
            settings,
            dense,
            pos,
            s,
            h,
            f,
            flags,
            shade_ctx,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera_math;
    use crate::Camera;
    use roxlap_formats::kv6::Kv6;
    use roxlap_formats::material::{Material, MaterialTable};

    /// DL.7 — `cast_local` reports the hit's model-local face normal (used to
    /// light sprites/clips). A ray crossing air then a solid block via the
    /// z face gets a back-facing (-z) normal; the entry voxel (immediately
    /// solid) gets a zero normal.
    #[test]
    fn cast_local_reports_face_normal() {
        // Solid only at z >= 4 (air below), full in x/y.
        let kv6 = Kv6::from_fn(8, 8, 8, |_, _, z| (z >= 4).then_some(0x80_C0_40_20));
        let dense = SpriteDense::from_kv6(&kv6);
        // Ray from below, travelling +z: air (z<4) then the block's top face.
        let (_c, _t, n, cell) =
            cast_local(&dense, [4.0, 4.0, -5.0], [0.0, 0.0, 1.0]).expect("ray hits the block");
        assert_eq!(cell[2], 4, "first solid voxel is the z=4 surface");
        assert!(
            n[2] < -0.5 && n[0].abs() < 1e-6 && n[1].abs() < 1e-6,
            "z-crossing face normal points back toward the ray (-z): {n:?}",
        );
    }
    use roxlap_formats::sprite::Sprite;
    use roxlap_formats::voxel_clip::{LoopMode, VoxelClip, VoxelFrame};

    fn settings(w: u32, h: u32) -> OpticastSettings {
        OpticastSettings::for_oracle_framebuffer(w, h)
    }

    /// Camera at the origin looking down +y at a sprite ahead.
    fn cam_looking_y() -> Camera {
        Camera {
            pos: [0.0, 0.0, 0.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 0.0, 1.0],
            forward: [0.0, 1.0, 0.0],
        }
    }

    /// Build a [`VoxelFrame`] from a dense `fill(x,y,z) -> Option<color>`.
    fn clip_frame(dims: [u32; 3], fill: impl Fn(u32, u32, u32) -> Option<u32>) -> VoxelFrame {
        let owpc = dims[2].div_ceil(32).max(1) as usize;
        let cols = (dims[0] * dims[1]) as usize;
        let mut occupancy = vec![0u32; cols * owpc];
        let mut color_offsets = vec![0u32; cols + 1];
        let mut colors = Vec::new();
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                let col = (x + y * dims[0]) as usize;
                color_offsets[col] = colors.len() as u32;
                for z in 0..dims[2] {
                    if let Some(c) = fill(x, y, z) {
                        occupancy[col * owpc + (z >> 5) as usize] |= 1u32 << (z & 31);
                        colors.push(c);
                    }
                }
            }
        }
        color_offsets[cols] = colors.len() as u32;
        VoxelFrame {
            occupancy,
            colors,
            color_offsets,
        }
    }

    /// A cached [`ClipFlipbook`] draws distinct frames distinctly — the
    /// CPU flipbook select. Frame 0 fills the bottom half (red), frame 1
    /// the top half (green); rendered at the same pose they cover
    /// different screen pixels in different colours.
    #[test]
    fn clip_flipbook_frames_render_differently() {
        let dims = [8u32, 8, 8];
        let f0 = clip_frame(dims, |_x, _y, z| (z < 4).then_some(0x00FF_0000)); // red, low z
        let f1 = clip_frame(dims, |_x, _y, z| (z >= 4).then_some(0x0000_FF00)); // green, high z
        let clip = VoxelClip::from_frames(
            dims,
            [4.0, 4.0, 4.0],
            1.0,
            LoopMode::Loop,
            &[f0, f1],
            &[],
            33,
            0,
        );
        let decoded = clip.decode().expect("decode");
        let book = ClipFlipbook::from_decoded(&decoded);
        assert_eq!(book.frame_count(), 2);
        assert!(book.frame(0).is_some() && book.frame(2).is_none());

        let (w, h) = (64u32, 64u32);
        let n = (w * h) as usize;
        let cam = cam_looking_y();
        let cs = camera_math::derive(&cam, w, h, 32.0, 32.0, 32.0);
        let cfg = settings(w, h);
        let pose = [0.0, 40.0, 0.0];
        let (s, hh, f) = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);

        let render = |frame: usize| -> Vec<u32> {
            let mut fb = vec![0u32; n];
            let mut zb = vec![f32::INFINITY; n];
            let wrote = book.draw_frame(
                &mut fb, &mut zb, w as usize, w, h, &cs, &cfg, frame, pose, s, hh, f, 0,
            );
            assert!(wrote > 0, "frame {frame} should draw some pixels");
            fb
        };
        let fb0 = render(0);
        let fb1 = render(1);
        assert_ne!(fb0, fb1, "distinct frames must render distinct pixels");
        // Each frame shows its own channel: red present in frame 0, green
        // present in frame 1.
        assert!(fb0.iter().any(|&p| (p & 0x00FF_0000) != 0));
        assert!(fb1.iter().any(|&p| (p & 0x0000_FF00) != 0));
        // Out-of-range frame draws nothing.
        let mut fb = vec![0u32; n];
        let mut zb = vec![f32::INFINITY; n];
        assert_eq!(
            book.draw_frame(&mut fb, &mut zb, w as usize, w, h, &cs, &cfg, 9, pose, s, hh, f, 0),
            0
        );
    }

    #[test]
    fn clip_flipbook_set_frame_replaces_one_frame() {
        // The single-frame edit primitive: replace frame 0's dense with
        // frame 1's content, in place. Out-of-range → false.
        let dims = [8u32, 8, 8];
        let f0 = clip_frame(dims, |_, _, z| (z < 4).then_some(0x00FF_0000)); // red
        let f1 = clip_frame(dims, |_, _, z| (z >= 4).then_some(0x0000_FF00)); // green
        let clip =
            VoxelClip::from_frames(dims, [4.0; 3], 1.0, LoopMode::Loop, &[f0, f1], &[], 33, 0);
        let decoded = clip.decode().unwrap();
        let mut book = ClipFlipbook::from_decoded(&decoded);

        let (w, h) = (64u32, 64u32);
        let n = (w * h) as usize;
        let cs = camera_math::derive(&cam_looking_y(), w, h, 32.0, 32.0, 32.0);
        let cfg = settings(w, h);
        let render0 = |b: &ClipFlipbook| -> Vec<u32> {
            let mut fb = vec![0u32; n];
            let mut zb = vec![f32::INFINITY; n];
            let _ = b.draw_frame(
                &mut fb,
                &mut zb,
                w as usize,
                w,
                h,
                &cs,
                &cfg,
                0,
                [0.0, 40.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                0,
            );
            fb
        };

        let before = render0(&book);
        assert!(
            before.iter().any(|&p| (p & 0x00FF_0000) != 0),
            "frame 0 is red"
        );

        // Replace frame 0 with frame 1's dense.
        let replacement = SpriteDense::from_voxel_frame(&decoded.frames[1], dims, decoded.pivot);
        assert!(book.set_frame(0, replacement));
        let extra = SpriteDense::from_voxel_frame(&decoded.frames[1], dims, decoded.pivot);
        assert!(!book.set_frame(9, extra), "out-of-range set_frame is false");

        let after = render0(&book);
        assert!(
            after.iter().any(|&p| (p & 0x0000_FF00) != 0),
            "frame 0 now green"
        );
        assert_ne!(before, after);
    }

    /// A solid cube sprite in front of the camera is drawn, with the
    /// cube colour (shaded) and a sensible centre depth.
    #[test]
    fn cube_sprite_renders() {
        let kv6 = Kv6::solid_cube(8, 0x80_C0_40_20);
        let sprite = Sprite::axis_aligned(kv6, [0.0, 40.0, 0.0]);
        let (w, h) = (64u32, 64u32);
        let n = (w * h) as usize;
        let mut fb = vec![0u32; n];
        let mut zb = vec![f32::INFINITY; n];
        let cam = cam_looking_y();
        let cs = camera_math::derive(&cam, w, h, 32.0, 32.0, 32.0);
        let wrote = draw_sprite_dda(
            &mut fb,
            &mut zb,
            w as usize,
            w,
            h,
            &cs,
            &settings(w, h),
            &sprite,
        );

        assert!(wrote > 20, "cube should cover many pixels (got {wrote})");
        let centre = (h / 2 * w + w / 2) as usize;
        assert_eq!(
            fb[centre] & 0x00ff_ffff,
            0x00_C0_40_20,
            "got {:08x}",
            fb[centre]
        );
        // Pivot at world y=40, cube spans y in [36,44] → near face ~36.
        assert!(
            (zb[centre] - 36.0).abs() < 3.0,
            "centre depth {} not ≈ 36",
            zb[centre]
        );
    }

    /// A KV6 whose voxel colours store a `0x00` high byte (voxlap's
    /// unused `dir` slot, e.g. `sprite_meltsphere.kv6`) must still
    /// render its authored RGB, not black — the brightness byte is
    /// normalised to full on decode.
    #[test]
    fn zero_high_byte_sprite_not_black() {
        let kv6 = Kv6::solid_cube(8, 0x00_C0_40_20);
        let sprite = Sprite::axis_aligned(kv6, [0.0, 40.0, 0.0]);
        let (w, h) = (64u32, 64u32);
        let n = (w * h) as usize;
        let mut fb = vec![0u32; n];
        let mut zb = vec![f32::INFINITY; n];
        let cam = cam_looking_y();
        let cs = camera_math::derive(&cam, w, h, 32.0, 32.0, 32.0);
        let wrote = draw_sprite_dda(
            &mut fb,
            &mut zb,
            w as usize,
            w,
            h,
            &cs,
            &settings(w, h),
            &sprite,
        );
        assert!(wrote > 20, "cube should cover many pixels (got {wrote})");
        let centre = (h / 2 * w + w / 2) as usize;
        assert_eq!(
            fb[centre] & 0x00ff_ffff,
            0x00_C0_40_20,
            "zero-high-byte sprite rendered as {:08x} (black bug)",
            fb[centre]
        );
    }

    /// A sprite occludes / is occluded by the z-buffer: a nearer
    /// pre-filled depth blocks the sprite; a farther one lets it win.
    #[test]
    fn sprite_respects_zbuffer() {
        let kv6 = Kv6::solid_cube(8, 0x80_FF_FF_FF);
        let sprite = Sprite::axis_aligned(kv6, [0.0, 40.0, 0.0]);
        let (w, h) = (32u32, 32u32);
        let n = (w * h) as usize;
        let cam = cam_looking_y();
        let cs = camera_math::derive(&cam, w, h, 16.0, 16.0, 16.0);
        let centre = (h / 2 * w + w / 2) as usize;

        // Terrain in front (depth 10 < ~36) → sprite blocked at centre.
        let mut fb = vec![0u32; n];
        let mut zb = vec![f32::INFINITY; n];
        fb[centre] = 0x80_11_22_33;
        zb[centre] = 10.0;
        let _ = draw_sprite_dda(
            &mut fb,
            &mut zb,
            w as usize,
            w,
            h,
            &cs,
            &settings(w, h),
            &sprite,
        );
        assert_eq!(
            fb[centre], 0x80_11_22_33,
            "near terrain must occlude sprite"
        );

        // Terrain behind (depth 100) → sprite wins.
        let mut fb2 = vec![0u32; n];
        let mut zb2 = vec![f32::INFINITY; n];
        fb2[centre] = 0x80_11_22_33;
        zb2[centre] = 100.0;
        let _ = draw_sprite_dda(
            &mut fb2,
            &mut zb2,
            w as usize,
            w,
            h,
            &cs,
            &settings(w, h),
            &sprite,
        );
        assert_ne!(fb2[centre], 0x80_11_22_33, "sprite must beat far terrain");
        assert!(zb2[centre] < 100.0, "sprite depth must replace terrain's");
    }

    /// The covered screen rect (min/max px,py) of whatever the sprite
    /// painted — used to compare an axis-aligned vs a rotated pose.
    fn covered_rect(fb: &[u32], w: u32, h: u32) -> (u32, u32, u32, u32) {
        let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0u32, 0u32);
        for py in 0..h {
            for px in 0..w {
                if fb[(py * w + px) as usize] & 0x00ff_ffff != 0 {
                    x0 = x0.min(px);
                    y0 = y0.min(py);
                    x1 = x1.max(px);
                    y1 = y1.max(py);
                }
            }
        }
        (x0, y0, x1, y1)
    }

    /// A non-cube box drawn axis-aligned vs. drawn with a per-instance
    /// transform that swaps its long axis onto the screen's other axis
    /// flips the silhouette's aspect ratio. Pins that the `s/h/f` basis
    /// (the path `DynSpriteTransform` feeds) actually reorients the model.
    #[test]
    fn posed_basis_reorients_silhouette() {
        // Wide-in-local-x, short-in-local-z box → appears wide on screen
        // (screen-x = world-x via `right`, screen-y = world-z via `down`).
        let kv6 = Kv6::solid_box(16, 4, 4, 0x80_C0_40_20);
        let (w, h) = (64u32, 64u32);
        let n = (w * h) as usize;
        let cam = cam_looking_y();
        let cs = camera_math::derive(&cam, w, h, 32.0, 32.0, 32.0);

        // Axis-aligned: wide silhouette.
        let aa = Sprite::axis_aligned(kv6.clone(), [0.0, 40.0, 0.0]);
        let mut fb = vec![0u32; n];
        let mut zb = vec![f32::INFINITY; n];
        let _ = draw_sprite_dda(
            &mut fb,
            &mut zb,
            w as usize,
            w,
            h,
            &cs,
            &settings(w, h),
            &aa,
        );
        let (ax0, ay0, ax1, ay1) = covered_rect(&fb, w, h);
        let aa_wide = (ax1 - ax0) as i32 - (ay1 - ay0) as i32;
        assert!(
            aa_wide > 4,
            "axis-aligned box should be wider than tall (got w-h={aa_wide})"
        );

        // Posed: map local +x onto world +z and local +z onto world +x
        // (det = -1 ≠ 0). Same box now reads tall on screen.
        let mut posed = aa.clone();
        posed.s = [0.0, 0.0, 1.0]; // local +x ↦ world +z (screen down)
        posed.h = [0.0, 1.0, 0.0]; // local +y ↦ world +y (depth)
        posed.f = [1.0, 0.0, 0.0]; // local +z ↦ world +x (screen right)
        let mut fb2 = vec![0u32; n];
        let mut zb2 = vec![f32::INFINITY; n];
        let _ = draw_sprite_dda(
            &mut fb2,
            &mut zb2,
            w as usize,
            w,
            h,
            &cs,
            &settings(w, h),
            &posed,
        );
        let (bx0, by0, bx1, by1) = covered_rect(&fb2, w, h);
        let posed_tall = (by1 - by0) as i32 - (bx1 - bx0) as i32;
        assert!(
            posed_tall > 4,
            "posed box should be taller than wide (got h-w={posed_tall})"
        );
    }

    /// A degenerate (singular) basis — `det == 0` — makes the sprite
    /// silently skip rather than panic (the `DynSpriteTransform` guard).
    #[test]
    fn degenerate_basis_draws_nothing() {
        let kv6 = Kv6::solid_cube(8, 0x80_FF_FF_FF);
        let mut sprite = Sprite::axis_aligned(kv6, [0.0, 40.0, 0.0]);
        sprite.f = sprite.s; // two equal columns → det 0
        let (w, h) = (32u32, 32u32);
        let n = (w * h) as usize;
        let mut fb = vec![0u32; n];
        let mut zb = vec![f32::INFINITY; n];
        let cam = cam_looking_y();
        let cs = camera_math::derive(&cam, w, h, 16.0, 16.0, 16.0);
        let wrote = draw_sprite_dda(
            &mut fb,
            &mut zb,
            w as usize,
            w,
            h,
            &cs,
            &settings(w, h),
            &sprite,
        );
        assert_eq!(wrote, 0, "singular basis must skip, not panic");
    }

    /// An invisible sprite draws nothing.
    #[test]
    fn invisible_sprite_skipped() {
        let kv6 = Kv6::solid_cube(8, 0x80_FF_FF_FF);
        let mut sprite = Sprite::axis_aligned(kv6, [0.0, 40.0, 0.0]);
        sprite.flags |= roxlap_formats::sprite::SPRITE_FLAG_INVISIBLE;
        let (w, h) = (32u32, 32u32);
        let n = (w * h) as usize;
        let mut fb = vec![0u32; n];
        let mut zb = vec![f32::INFINITY; n];
        let cam = cam_looking_y();
        let cs = camera_math::derive(&cam, w, h, 16.0, 16.0, 16.0);
        let wrote = draw_sprite_dda(
            &mut fb,
            &mut zb,
            w as usize,
            w,
            h,
            &cs,
            &settings(w, h),
            &sprite,
        );
        assert_eq!(wrote, 0);
    }

    // ---------- TV.1a: translucent accumulate-and-continue path ----------

    /// Draw a uniform-material 8³ cube (RGB `0xC0_40_20`) at world y=40 over
    /// a `bg`-filled framebuffer with z-buffer `zb_v`, using palette id 1 =
    /// `mat`. Returns `(centre_pixel, full_framebuffer)`.
    fn draw_cube_shaded(mat: Material, alpha_mul: u8, bg: u32, zb_v: f32) -> (u32, Vec<u32>) {
        let mut table = MaterialTable::new();
        table.set(1, mat);
        let dense = SpriteDense::from_kv6(&Kv6::solid_cube(8, 0x80_C0_40_20));
        let (w, h) = (64u32, 64u32);
        let n = (w * h) as usize;
        let mut fb = vec![bg; n];
        let mut zb = vec![zb_v; n];
        let cs = camera_math::derive(&cam_looking_y(), w, h, 32.0, 32.0, 32.0);
        let sh = SpriteShade {
            materials: &table,
            lights: CpuLights::default(),
            material: 1,
            alpha_mul,
        };
        let _ = draw_sprite_dense_shaded(
            &mut fb,
            &mut zb,
            w as usize,
            w,
            h,
            &cs,
            &settings(w, h),
            &dense,
            [0.0, 40.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            0,
            Some(sh),
        );
        (fb[(h / 2 * w + w / 2) as usize], fb)
    }

    /// An additive sprite over a dark background brightens it (glow) — and
    /// never darkens any channel below the background.
    #[test]
    fn additive_sprite_brightens_background() {
        let bg = 0x80_20_20_20;
        let (centre, _) = draw_cube_shaded(Material::additive(255), 255, bg, f32::INFINITY);
        let (cr, cg, cb) = ((centre >> 16) & 0xff, (centre >> 8) & 0xff, centre & 0xff);
        assert!(
            cr > 0x20 && cg > 0x20 && cb >= 0x20,
            "centre {centre:08x} should be brighter than bg"
        );
        // Red channel (sprite 0xC0) lifts the most.
        assert!(
            cr >= cg && cr >= cb,
            "additive of a red-dominant cube stays red-dominant"
        );
    }

    /// An alpha-blend sprite composites *between* the background and its own
    /// colour — neither equal to the bare background nor the opaque colour.
    #[test]
    fn alpha_blend_sprite_between_bg_and_color() {
        let bg = 0x80_20_20_20;
        let (centre, _) = draw_cube_shaded(Material::alpha_blend(128), 255, bg, f32::INFINITY);
        let cr = (centre >> 16) & 0xff;
        assert!(
            cr > 0x20,
            "blended red must rise above bg 0x20 (got {cr:02x})"
        );
        assert!(
            cr < 0xC0,
            "blended red must stay below opaque 0xC0 (got {cr:02x})"
        );
        // Distinct from both endpoints.
        assert_ne!(centre & 0x00ff_ffff, bg & 0x00ff_ffff);
        assert_ne!(centre & 0x00ff_ffff, 0x00_C0_40_20);
    }

    /// The per-instance `alpha_mul` scales opacity: a lower multiplier keeps
    /// more of the background (less of the sprite colour).
    #[test]
    fn alpha_mul_scales_opacity() {
        let bg = 0x80_20_20_20;
        let (full, _) = draw_cube_shaded(Material::alpha_blend(255), 255, bg, f32::INFINITY);
        let (faded, _) = draw_cube_shaded(Material::alpha_blend(255), 64, bg, f32::INFINITY);
        let r_full = (full >> 16) & 0xff;
        let r_faded = (faded >> 16) & 0xff;
        // Both lift red above bg, but the faded one stays closer to bg.
        assert!(
            r_full > r_faded,
            "alpha_mul=255 ({r_full:02x}) more opaque than 64 ({r_faded:02x})"
        );
        assert!(r_faded > 0x20, "even faded lifts above bg");
    }

    /// A `SpriteShade` whose effective material is **opaque** (id 0) renders
    /// byte-for-byte identically to the plain opaque path — the per-sprite
    /// gate that keeps the opaque world unchanged.
    #[test]
    fn opaque_shade_ctx_matches_plain_path() {
        let table = MaterialTable::new();
        let dense = SpriteDense::from_kv6(&Kv6::solid_cube(8, 0x80_C0_40_20));
        let (w, h) = (64u32, 64u32);
        let n = (w * h) as usize;
        let cs = camera_math::derive(&cam_looking_y(), w, h, 32.0, 32.0, 32.0);
        let pose = (
            [0.0, 40.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        );

        let mut fb_plain = vec![0u32; n];
        let mut zb_plain = vec![f32::INFINITY; n];
        let _ = draw_sprite_dense(
            &mut fb_plain,
            &mut zb_plain,
            w as usize,
            w,
            h,
            &cs,
            &settings(w, h),
            &dense,
            pose.0,
            pose.1,
            pose.2,
            pose.3,
            0,
        );

        let mut fb_sh = vec![0u32; n];
        let mut zb_sh = vec![f32::INFINITY; n];
        let sh = SpriteShade {
            materials: &table,
            lights: CpuLights::default(),
            material: 0, // opaque
            alpha_mul: 255,
        };
        let _ = draw_sprite_dense_shaded(
            &mut fb_sh,
            &mut zb_sh,
            w as usize,
            w,
            h,
            &cs,
            &settings(w, h),
            &dense,
            pose.0,
            pose.1,
            pose.2,
            pose.3,
            0,
            Some(sh),
        );

        assert_eq!(
            fb_plain, fb_sh,
            "opaque shade-ctx must match the plain path bit-for-bit"
        );
        assert_eq!(zb_plain, zb_sh, "opaque shade-ctx z-buffer must match too");
    }

    /// A translucent (additive) sprite behind nearer terrain is occluded:
    /// the front depth (~36) is past the z-buffer cutoff (5), so the march
    /// stops before contributing and the background pixel is untouched.
    #[test]
    fn translucent_sprite_occluded_by_near_terrain() {
        let bg = 0x80_20_20_20;
        let (centre, _) = draw_cube_shaded(Material::additive(255), 255, bg, 5.0);
        assert_eq!(
            centre, bg,
            "near terrain (z=5) must occlude the sprite at y≈36"
        );
    }

    /// Per-span compositing: a translucent voxel contributes one alpha layer
    /// per contiguous solid run, so a 2-voxel-thick slab composites the same
    /// as a 1-voxel-thick one (adjacent voxels are not double-counted). This
    /// is the fix for the voxel-grid striping where a ray clipping a shared
    /// voxel boundary passed through two cells of one wall.
    #[test]
    fn per_span_thickness_independent() {
        fn centre(ysiz: u32) -> u32 {
            let mut table = MaterialTable::new();
            table.set(1, Material::alpha_blend(128));
            let dense = SpriteDense::from_kv6(&Kv6::solid_box(8, ysiz, 8, 0x80_C0_40_20));
            let (w, h) = (64u32, 64u32);
            let n = (w * h) as usize;
            let mut fb = vec![0x80_10_10_10u32; n];
            let mut zb = vec![f32::INFINITY; n];
            let cs = camera_math::derive(&cam_looking_y(), w, h, 32.0, 32.0, 32.0);
            let sh = SpriteShade {
                materials: &table,
                lights: CpuLights::default(),
                material: 1,
                alpha_mul: 255,
            };
            let _ = draw_sprite_dense_shaded(
                &mut fb,
                &mut zb,
                w as usize,
                w,
                h,
                &cs,
                &settings(w, h),
                &dense,
                [0.0, 40.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                0,
                Some(sh),
            );
            fb[(h / 2 * w + w / 2) as usize] & 0x00ff_ffff
        }
        // A 2-deep box is solid through (surface-only of a 2-thick box is both
        // y-layers); per-span treats the straight-through ray's two adjacent
        // voxels as one surface → identical to the 1-deep slab.
        assert_eq!(
            centre(1),
            centre(2),
            "per-span: a 2-thick slab must match a 1-thick one (no double-count)"
        );
    }

    /// Volumetric (Beer–Lambert) is the thickness-*dependent* counterpart of
    /// per-span: a deeper **filled** volume absorbs more, so its centre pixel
    /// sits closer to the volume colour (less background shows through) than a
    /// shallow one — the opposite of `per_span_thickness_independent`.
    #[test]
    fn volumetric_thickness_deepens_opacity() {
        // Centre-pixel red channel of a filled red box `depth` voxels deep,
        // Volumetric material, over a dark background.
        fn red_at(depth: u32) -> u32 {
            let mut table = MaterialTable::new();
            table.set(1, Material::volumetric(128));
            // FILLED box (every cell solid, interior kept) so the ray actually
            // traverses `depth` absorbing voxels. `from_fn` would cull the
            // interior to a hollow shell (front+back faces only) — no genuine
            // depth accumulation — so use `from_fn_keep_interior`.
            let kv6 =
                Kv6::from_fn_keep_interior(8, depth, 8, |_, _, _| Some(0x80_C0_20_20), |_| true);
            let dense = SpriteDense::from_kv6(&kv6);
            let (w, h) = (64u32, 64u32);
            let n = (w * h) as usize;
            let mut fb = vec![0x80_10_10_10u32; n];
            let mut zb = vec![f32::INFINITY; n];
            let cs = camera_math::derive(&cam_looking_y(), w, h, 32.0, 32.0, 32.0);
            let sh = SpriteShade {
                materials: &table,
                lights: CpuLights::default(),
                material: 1,
                alpha_mul: 255,
            };
            let _ = draw_sprite_dense_shaded(
                &mut fb,
                &mut zb,
                w as usize,
                w,
                h,
                &cs,
                &settings(w, h),
                &dense,
                [0.0, 40.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                0,
                Some(sh),
            );
            (fb[(h / 2 * w + w / 2) as usize] >> 16) & 0xff
        }
        let shallow = red_at(1);
        let deep = red_at(12);
        // Both lift red above the 0x10 background; the deeper volume absorbs
        // more of its own colour in, so its red is higher (more opaque).
        assert!(
            shallow > 0x10,
            "even a 1-deep volume tints (got {shallow:02x})"
        );
        assert!(
            deep > shallow,
            "deeper Volumetric volume is more opaque: deep {deep:02x} > shallow {shallow:02x}"
        );
    }

    /// The demo scenario: an **opaque** backdrop sprite drawn first, then a
    /// **translucent** sprite in front of it sharing the buffer. The glass
    /// must composite over the backdrop colour (tint it), not leave it
    /// unchanged. Pins the CPU opaque-then-translucent interaction.
    #[test]
    fn translucent_sprite_tints_opaque_sprite_behind() {
        let mut table = MaterialTable::new();
        table.set(1, Material::alpha_blend(128));
        let (w, h) = (64u32, 64u32);
        let n = (w * h) as usize;
        let mut fb = vec![0x80_10_20_40u32; n]; // flat sky
        let mut zb = vec![f32::INFINITY; n];
        let cs = camera_math::derive(&cam_looking_y(), w, h, 32.0, 32.0, 32.0);
        let cfg = settings(w, h);
        let id = [1.0, 0.0, 0.0];
        let up = [0.0, 1.0, 0.0];
        let fw = [0.0, 0.0, 1.0];
        let centre = (h / 2 * w + w / 2) as usize;

        // Opaque red backdrop (material 0), far.
        let backdrop = SpriteDense::from_kv6(&Kv6::solid_cube(12, 0x80_FF_00_00));
        let sh_op = SpriteShade {
            materials: &table,
            lights: CpuLights::default(),
            material: 0,
            alpha_mul: 255,
        };
        let _ = draw_sprite_dense_shaded(
            &mut fb,
            &mut zb,
            w as usize,
            w,
            h,
            &cs,
            &cfg,
            &backdrop,
            [0.0, 80.0, 0.0],
            id,
            up,
            fw,
            0,
            Some(sh_op),
        );
        let after_backdrop = fb[centre];
        assert_eq!(
            after_backdrop & 0x00ff_ffff,
            0x00FF_0000,
            "backdrop red must be drawn first"
        );

        // Cyan glass (material 1), nearer + overlapping.
        let glass = SpriteDense::from_kv6(&Kv6::solid_cube(12, 0x80_00_FF_FF));
        let sh_gl = SpriteShade {
            materials: &table,
            lights: CpuLights::default(),
            material: 1,
            alpha_mul: 255,
        };
        let wrote = draw_sprite_dense_shaded(
            &mut fb,
            &mut zb,
            w as usize,
            w,
            h,
            &cs,
            &cfg,
            &glass,
            [0.0, 40.0, 0.0],
            id,
            up,
            fw,
            0,
            Some(sh_gl),
        );
        let _ = wrote;
        let after_glass = fb[centre];
        assert_ne!(
            after_glass, after_backdrop,
            "glass must tint the backdrop (composite over it)"
        );
        // Cyan over red: red channel drops, blue/green rise.
        assert!(
            (after_glass >> 16) & 0xff < 0xFF,
            "glass should reduce the backdrop's red (got {after_glass:08x})"
        );
    }

    /// TV.3: `from_kv6_with_materials` classifies voxels into per-voxel
    /// material ids by colour — mapped colour → its id, unmapped → 0.
    #[test]
    fn from_kv6_with_materials_classifies_by_color() {
        let col = 0x80_AA_BB_CC;
        let kv6 = Kv6::solid_cube(6, col);
        let dense = SpriteDense::from_kv6_with_materials(&kv6, &[(0x00AA_BBCC, 2)]);
        assert_eq!(
            dense.mat.len(),
            dense.col.len(),
            "per-voxel mat array sized"
        );
        let mut solids = 0;
        for idx in 0..dense.occ.len() {
            if dense.occ[idx] {
                assert_eq!(dense.mat[idx], 2, "mapped colour → material 2");
                solids += 1;
            }
        }
        assert!(solids > 0, "cube has solid voxels");
        // A map that doesn't include the cube's colour → all opaque (0).
        let dense0 = SpriteDense::from_kv6_with_materials(&kv6, &[(0x0012_3456, 5)]);
        assert!(
            dense0.mat.iter().all(|&m| m == 0),
            "unmapped colour → material 0"
        );
    }

    /// TV.3: a model whose every voxel maps to the *same* material id renders
    /// identically to drawing it with that material as the instance's uniform
    /// material — the per-voxel path reduces to the uniform path when
    /// homogeneous (and overrides the instance's `material`).
    #[test]
    fn per_voxel_material_matches_uniform_when_homogeneous() {
        let mut table = MaterialTable::new();
        table.set(1, Material::alpha_blend(120));
        let col = 0x80_30_A0_F0;
        let kv6 = Kv6::solid_cube(10, col);
        let (w, h) = (64u32, 64u32);
        let n = (w * h) as usize;
        let cs = camera_math::derive(&cam_looking_y(), w, h, 32.0, 32.0, 32.0);
        let cfg = settings(w, h);
        let (pos, s, hh, f) = (
            [0.0, 40.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        );
        let render = |dense: &SpriteDense, material: u8| -> Vec<u32> {
            let mut fb = vec![0x80_10_10_10u32; n];
            let mut zb = vec![f32::INFINITY; n];
            let sh = SpriteShade {
                materials: &table,
                lights: CpuLights::default(),
                material,
                alpha_mul: 255,
            };
            let _ = draw_sprite_dense_shaded(
                &mut fb,
                &mut zb,
                w as usize,
                w,
                h,
                &cs,
                &cfg,
                dense,
                pos,
                s,
                hh,
                f,
                0,
                Some(sh),
            );
            fb
        };
        // Per-voxel: every voxel → material 1; instance's uniform material is 0
        // (opaque) but the per-voxel id overrides it.
        let pv = render(
            &SpriteDense::from_kv6_with_materials(&kv6, &[(col & 0xff_ffff, 1)]),
            0,
        );
        // Uniform: no per-voxel data, instance material 1.
        let un = render(&SpriteDense::from_kv6(&kv6), 1);
        assert_eq!(pv, un, "homogeneous per-voxel material == uniform material");
        // And it's actually translucent (differs from the bare background).
        let centre = (h / 2 * w + w / 2) as usize;
        assert_ne!(pv[centre] & 0x00ff_ffff, 0x0010_1010, "translucent, not bg");
    }

    /// TV.3 (clip wiring): a [`ClipFlipbook`] built with a colour→material map
    /// carries per-voxel materials on every cached frame (the clip analogue of
    /// `from_kv6_with_materials`); an empty map leaves them all-opaque, so the
    /// flipbook is byte-identical to `from_decoded`.
    #[test]
    fn clip_flipbook_with_materials_classifies_every_frame() {
        let dims = [6u32, 6, 6];
        let glass = 0x00AA_BBCC;
        let glass_lit = 0x80AA_BBCC;
        // Two distinct frames, both filled with the glass colour.
        let f0 = clip_frame(dims, |_x, _y, z| (z < 3).then_some(glass_lit));
        let f1 = clip_frame(dims, |_x, _y, z| (z >= 3).then_some(glass_lit));
        let clip = VoxelClip::from_frames(
            dims,
            [3.0, 3.0, 3.0],
            1.0,
            LoopMode::Loop,
            &[f0, f1],
            &[],
            33,
            0,
        );
        let decoded = clip.decode().expect("decode");

        let book = ClipFlipbook::from_decoded_with_materials(&decoded, &[(glass, 2)]);
        assert_eq!(book.frame_count(), 2);
        for fr in 0..2 {
            let dense = book.frame(fr).expect("frame in range");
            assert_eq!(dense.mat.len(), dense.col.len(), "frame {fr} mat sized");
            let mut solids = 0;
            for idx in 0..dense.occ.len() {
                if dense.occ[idx] {
                    assert_eq!(dense.mat[idx], 2, "frame {fr}: glass → material 2");
                    solids += 1;
                }
            }
            assert!(solids > 0, "frame {fr} has solid voxels");
        }

        // An empty map ⇒ no per-voxel materials, identical to `from_decoded`.
        let plain = ClipFlipbook::from_decoded(&decoded);
        let plain_mat = ClipFlipbook::from_decoded_with_materials(&decoded, &[]);
        for fr in 0..2 {
            assert!(plain.frame(fr).unwrap().mat.is_empty());
            assert!(plain_mat.frame(fr).unwrap().mat.is_empty());
        }
    }
}
