//! Per-pixel 3D-DDA + brickmap CPU renderer (Substage DDA).
//!
//! This is the clean-room replacement for the voxlap-derived
//! column-coherent opticast pipeline (`opticast` + `grouscan` +
//! `scan_loops`). Every pixel casts one independent ray, so none of
//! the column/row-coherence stitching artifacts of the 2.5D voxlap
//! renderer can occur (silhouette notch, floor hairlines, axis-aligned
//! mip beams, cross-chunk virtual-column complexity). See
//! `PORTING-DDA.md` for the full stage plan.
//!
//! **Stage status — DDA.6 (per-grid distance mip) + DDA.7 (tile
//! parallelism).** Each pixel casts one ray over the grid's full voxel
//! box ([`GridView::voxel_bounds`], spanning every chunk in XY **and**
//! Z) via a 3D-DDA (Amanatides–Woo). A uniform render mip (chosen per
//! grid by LOD distance, clamped by [`effective_mip`] to a level every
//! chunk has built) coarsens the cell size to `2^mip` mip-0 voxels and
//! samples mip-`mip` data — the ray stays in mip-0 units so depth and
//! fog are exact. [`BrickMaps`] (one occupancy map per populated chunk,
//! at the render mip) are built once per frame and shared immutably; a
//! [`Sampler`] resolves each cell to its chunk
//! ([`GridView::chunk_at_xyz`]) and brick-gates the
//! [`GridView::surface_color_mip`] slab walk, caching the current chunk
//! so air costs an O(1) bit test. [`render_dda_parallel`] splits the
//! frame into disjoint rayon bands — bit-identical to sequential since
//! pixels are independent. Hits are shaded by baked brightness
//! ([`shade`]) + [`DdaEnv::side_shades`] face tint, fogged toward
//! [`DdaEnv::fog_color`] ([`apply_fog`]); misses sample the
//! [`DdaEnv::sky`] panorama ([`sample_sky`]) or keep the solid pre-fill.
//!
//! Buffer conventions match the rest of the engine so this backend is
//! colour is packed `0x80RRGGBB`; depth is perpendicular distance from
//! the camera with **smaller = closer** (so the scene compositor's
//! min-z merge works directly on the z-buffer this writes).

use std::collections::HashMap;

use rayon::prelude::*;

use crate::camera_math::{self, CameraState};
use crate::grid_view::GridView;
use crate::opticast::OpticastSettings;
use crate::raster_target::RasterTarget;
use crate::sky::Sky;
use crate::Camera;
use roxlap_formats::material::{material_for_color, Material, MaterialTable};

/// Per-frame environment for DDA shading (Substage DDA.5): a textured
/// sky panorama, distance fog, and per-face side shading.
///
/// [`DdaEnv::default`] disables all three — flat baked-brightness hits
/// and a caller-pre-filled solid sky — so the brickmap/dense equivalence
/// tests run against an unchanged pipeline.
#[derive(Clone, Copy)]
pub struct DdaEnv<'a> {
    /// Textured sky sampled per-ray-direction on a miss. `None` leaves
    /// the destination untouched (caller's solid sky pre-fill shows).
    pub sky: Option<&'a Sky>,
    /// Fog target colour (`0x__RRGGBB`); hits blend toward it with
    /// distance. Typically the sky colour so terrain fades into the sky.
    pub fog_color: u32,
    /// Depth at which fog is fully opaque. `<= 0` disables fog.
    pub fog_max_dist: f32,
    /// Per-face brightness reduction `[x-, x+, y-, y+, z-, z+]`, applied
    /// to the hit face (voxlap `setsideshades`). All-zero = off.
    pub side_shades: [i8; 6],
    /// TV: global voxel-material palette (id → opacity + blend mode). `None`
    /// keeps terrain fully opaque (the first-hit path, bit-identical).
    pub materials: Option<&'a MaterialTable>,
    /// TV: terrain colour→material map (`(rgb, material_id)`). A hit voxel's
    /// colour is looked up here for its material. **Empty** (the default) ⇒
    /// every voxel is opaque, so the march returns the first hit unchanged.
    pub terrain_materials: &'a [(u32, u8)],
}

impl Default for DdaEnv<'_> {
    fn default() -> Self {
        Self {
            sky: None,
            fog_color: 0,
            fog_max_dist: 0.0,
            side_shades: [0; 6],
            materials: None,
            terrain_materials: &[],
        }
    }
}

/// Per-pixel output target for the DDA renderer.
///
/// Abstracts "where does a ray hit go" so the traversal core stays
/// free of framebuffer mechanics. The production impl is
/// [`RasterSink`] (raw fb/zb pointers); tests use a recording sink.
/// Only *hits* are reported — misses (sky) leave the destination
/// untouched, matching the caller-pre-fills-sky convention.
pub trait PixelSink {
    /// Record a ray hit at framebuffer index `idx` (`py * pitch + px`)
    /// with packed ARGB `color` and perpendicular `dist` (smaller =
    /// closer).
    fn put(&mut self, idx: usize, color: u32, dist: f32);
}

/// [`PixelSink`] over a borrowed `(framebuffer, zbuffer)` pair.
///
/// Wraps a [`RasterTarget`] so the DDA path writes through the same
/// raw-pointer mechanism the scalar rasterizer uses — which keeps the
/// door open for the same strip/tile-disjoint parallel writes in
/// DDA.7.
pub struct RasterSink<'a> {
    target: RasterTarget<'a>,
    len: usize,
}

impl<'a> RasterSink<'a> {
    /// Build a sink from exclusive framebuffer + zbuffer borrows.
    /// Both slices must have the same length (the pixel count).
    #[must_use]
    pub fn new(framebuffer: &'a mut [u32], zbuffer: &'a mut [f32]) -> Self {
        debug_assert_eq!(framebuffer.len(), zbuffer.len());
        let len = framebuffer.len();
        Self {
            target: RasterTarget::new(framebuffer, zbuffer),
            len,
        }
    }
}

impl PixelSink for RasterSink<'_> {
    fn put(&mut self, idx: usize, color: u32, dist: f32) {
        if idx < self.len {
            // SAFETY: bounds checked above; single-threaded writer in
            // DDA.0 so the disjoint-write invariant holds trivially.
            unsafe {
                self.target.write_color(idx, color);
                self.target.write_depth(idx, dist);
            }
        }
    }
}

/// A resolved ray hit: surface colour + perpendicular distance.
#[derive(Debug, Clone, Copy)]
struct Hit {
    color: u32,
    dist: f32,
}

/// Test-only per-thread traversal counters for the perf bench.
#[cfg(test)]
pub(crate) mod prof {
    use std::cell::Cell;
    thread_local! {
        pub static CELLS: Cell<u64> = const { Cell::new(0) };
        pub static BRICKS: Cell<u64> = const { Cell::new(0) };
        pub static SURF: Cell<u64> = const { Cell::new(0) };
    }
    pub fn reset() {
        CELLS.with(|x| x.set(0));
        BRICKS.with(|x| x.set(0));
        SURF.with(|x| x.set(0));
    }
    pub fn read() -> (u64, u64, u64) {
        (
            CELLS.with(Cell::get),
            BRICKS.with(Cell::get),
            SURF.with(Cell::get),
        )
    }
}

/// Apply the voxel's baked directional brightness (Substage DDA.5).
///
/// Voxlap (and the GPU marcher, `grid_dda.wgsl`) store per-voxel
/// brightness in the colour's high byte on a `0..128` scale — `0x80`
/// is full brightness — written by `Grid::bake_lightmode` (estnorm
/// directional shading). The shaded channel is `c · a / 128`, so the
/// DDA matches the GPU look; an unbaked / full-bright voxel (`a =
/// 0x80`) passes through unchanged. Output alpha is normalised to
/// `0x80` (the standard "lit" flag; the present blit ignores it).
///
/// The renderer only *reads* the baked byte — it computes no normals
/// itself, so per-impact relight is free (re-bake the chunk and the
/// byte updates). The estnorm bake that produces the byte is the
/// voxlap-derived piece slated for a clean-room rewrite in DDA.10.
///
/// `bright_sub` is the per-face `side_shades` reduction (DDA.5): voxlap
/// subtracts it from the brightness byte before the multiply, so a
/// shaded face is uniformly darker. `0` = no side shading.
#[inline]
pub(crate) fn shade(color: u32, bright_sub: u32) -> u32 {
    let a = ((color >> 24) & 0xff).saturating_sub(bright_sub);
    let ch = |shift: u32| -> u32 { ((((color >> shift) & 0xff) * a) >> 7).min(255) };
    0x8000_0000 | (ch(16) << 16) | (ch(8) << 8) | ch(0)
}

/// Blend `color` toward `env.fog_color` by perpendicular `depth`
/// (linear, fully fogged at `env.fog_max_dist`). No-op when fog is
/// disabled (`fog_max_dist <= 0`).
#[inline]
fn apply_fog(color: u32, depth: f32, env: &DdaEnv<'_>) -> u32 {
    if env.fog_max_dist <= 0.0 {
        return color;
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let f = ((depth / env.fog_max_dist).clamp(0.0, 1.0) * 256.0) as u32; // 0..256
    let g = 256 - f;
    let fog = env.fog_color;
    let mix = |shift: u32| -> u32 {
        let src = (color >> shift) & 0xff;
        let dst = (fog >> shift) & 0xff;
        ((src * g + dst * f) >> 8).min(255)
    };
    0x8000_0000 | (mix(16) << 16) | (mix(8) << 8) | mix(0)
}

/// TV: resolve a terrain voxel's [`Material`] from its colour via the env's
/// colour→material map + palette. Returns [`Material::OPAQUE`] when no
/// material table is set, the map is empty, or the colour is unmapped — so
/// the march stays on the opaque first-hit path.
#[inline]
fn terrain_material(env: &DdaEnv<'_>, color: u32) -> Material {
    match env.materials {
        Some(table) if !env.terrain_materials.is_empty() => {
            table.get(material_for_color(env.terrain_materials, color))
        }
        _ => Material::OPAQUE,
    }
}

/// Composite premultiplied `accum` (+ remaining `trans`) over a packed
/// background colour → packed `0x80RRGGBB`.
#[inline]
fn composite_over(accum: [f32; 3], trans: f32, bg: u32) -> u32 {
    let b = rgb_to_f32(bg);
    f32_to_rgb([
        accum[0] + trans * b[0],
        accum[1] + trans * b[1],
        accum[2] + trans * b[2],
    ])
}

/// Finalize a translucent terrain ray that exited the grid (sky). Returns
/// `None` when nothing was accumulated (the opaque first-hit path — the
/// caller's sky handling stands, bit-identical), else the accumulated
/// layers composited over the sky at `dist`.
#[inline]
fn finalize_exit(
    touched: bool,
    accum: [f32; 3],
    trans: f32,
    env: &DdaEnv<'_>,
    dir: [f32; 3],
    dist: f32,
) -> Option<Hit> {
    if !touched {
        return None;
    }
    let bg = match env.sky {
        Some(s) => sample_sky(s, dir),
        None => 0x8000_0000 | (env.fog_color & 0x00ff_ffff),
    };
    Some(Hit {
        color: composite_over(accum, trans, bg),
        dist,
    })
}

/// Unpack `0x__RRGGBB` to `0..1` float channels (RGB; the high byte is
/// dropped — it has already been folded into the colour by `shade`/`fog`).
#[inline]
#[allow(clippy::cast_precision_loss)]
fn rgb_to_f32(c: u32) -> [f32; 3] {
    [
        ((c >> 16) & 0xff) as f32 / 255.0,
        ((c >> 8) & 0xff) as f32 / 255.0,
        (c & 0xff) as f32 / 255.0,
    ]
}

/// Repack `0..1` float channels (clamped) into `0x80RRGGBB`.
#[inline]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn f32_to_rgb(c: [f32; 3]) -> u32 {
    let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
    0x8000_0000 | (q(c[0]) << 16) | (q(c[1]) << 8) | q(c[2])
}

/// Sample the sky panorama in ray direction `dir` (need not be
/// normalised), returning a packed `0x80RRGGBB` colour.
///
/// Clean-room equirectangular mapping (not voxlap's `lng`/`lat` asm
/// search): the texture's x axis is elevation (`asin` of the vertical
/// component), the y axis is azimuth (`atan2` around the vertical). A
/// `ysiz == 1` panorama (e.g. [`Sky::blue_gradient`]) is a pure
/// horizon→zenith gradient.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn sample_sky(sky: &Sky, dir: [f32; 3]) -> u32 {
    let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
    if len < 1e-9 {
        return 0x8000_0000;
    }
    let d = [dir[0] / len, dir[1] / len, dir[2] / len];
    let xsiz_full = sky.lat.len().max(1) as i32; // original column count
    let pi = std::f32::consts::PI;
    // Elevation → x, matching the GPU `sky_color` (scene_dda.wgsl): z is
    // down, so `acos(-z)` is 0 at the zenith (looking up) and π at the nadir
    // (looking down); `/π` puts the zenith at x=0 and the nadir at x=xsiz.
    let elev01 = (-d[2]).clamp(-1.0, 1.0).acos() / pi; // 0 (up) .. 1 (down)
    let x = (elev01 * xsiz_full as f32) as i32;
    let x = x.clamp(0, xsiz_full - 1);
    // Azimuth → y (wrapped).
    let y = if sky.ysiz <= 1 {
        0
    } else {
        let az = d[1].atan2(d[0]); // -pi..pi
        let yf = ((az / (pi * 2.0)) + 0.5) * sky.ysiz as f32;
        (yf as i32).rem_euclid(sky.ysiz)
    };
    let idx = (y * xsiz_full + x) as usize;
    let px = sky.pixels.get(idx).copied().unwrap_or(0) as u32;
    0x8000_0000 | (px & 0x00ff_ffff)
}

/// Fill `fb` with the panorama [`Sky`] sampled per pixel — the background for
/// a **gridless** scene. The per-grid DDA samples the sky for its miss rays,
/// but a scene with no grids (a sprite/effect-only view) casts no rays, so it
/// would otherwise show a flat clear colour. This mirrors that miss-ray
/// sample for every pixel. The z-buffer is left untouched (sky is infinitely
/// far, so sprites always pass the depth test). `cam`/`settings` are the same
/// per-frame projection the sprite pass uses.
#[allow(clippy::cast_possible_truncation)]
pub fn render_sky_fill(
    fb: &mut [u32],
    pitch_pixels: usize,
    width: u32,
    height: u32,
    cam: &CameraState,
    settings: &OpticastSettings,
    sky: &Sky,
) {
    for py in 0..height {
        let row = py as usize * pitch_pixels;
        for px in 0..width {
            let (_origin, dir) = pixel_ray(cam, settings, px, py);
            fb[row + px as usize] = sample_sky(sky, dir);
        }
    }
}

/// World-space ray for screen pixel `(px, py)` under opticast's
/// pinhole: origin is the camera position, direction is
/// `(px - hx)·right + (py - hy)·down + hz·forward`.
///
/// This is the exact ray `camera_math::derive` bakes into its corner
/// vectors (`corn[0]` is `pixel (0, 0)`'s direction), so the DDA
/// renderer samples the same rays the voxlap path's frustum is built
/// from. The direction is **not** normalised — callers that need a
/// unit ray (and a true Euclidean distance) normalise themselves;
/// DDA.1 will track perpendicular distance via the forward-projection
/// instead, matching the engine's z-buffer convention.
#[must_use]
pub fn pixel_ray(
    cs: &CameraState,
    settings: &OpticastSettings,
    px: u32,
    py: u32,
) -> ([f32; 3], [f32; 3]) {
    // u32 → f32 is exact for any realistic screen coordinate.
    #[allow(clippy::cast_precision_loss)]
    let sx = px as f32 - settings.hx;
    #[allow(clippy::cast_precision_loss)]
    let sy = py as f32 - settings.hy;
    let dir = [
        sx * cs.right[0] + sy * cs.down[0] + settings.hz * cs.forward[0],
        sx * cs.right[1] + sy * cs.down[1] + settings.hz * cs.forward[1],
        sx * cs.right[2] + sy * cs.down[2] + settings.hz * cs.forward[2],
    ];
    (cs.pos, dir)
}

/// Ray ↔ axis-aligned box `[lo, hi]` slab test. Returns the
/// `(t_enter, t_exit)` parameter interval along `dir` (already clamped
/// so `t_enter >= 0`, i.e. a camera inside the box starts at `t = 0`),
/// or `None` if the ray misses the box. `dir` need not be normalised —
/// `t` is in units of `|dir|`.
pub(crate) fn intersect_aabb(
    o: [f32; 3],
    dir: [f32; 3],
    lo: [f32; 3],
    hi: [f32; 3],
) -> Option<(f32, f32)> {
    let mut t0 = 0.0f32;
    let mut t1 = f32::INFINITY;
    for a in 0..3 {
        if dir[a].abs() < 1e-9 {
            // Ray parallel to this slab — must already be inside it.
            if o[a] < lo[a] || o[a] > hi[a] {
                return None;
            }
        } else {
            let inv = 1.0 / dir[a];
            let mut ta = (lo[a] - o[a]) * inv;
            let mut tb = (hi[a] - o[a]) * inv;
            if ta > tb {
                core::mem::swap(&mut ta, &mut tb);
            }
            t0 = t0.max(ta);
            t1 = t1.min(tb);
            if t0 > t1 {
                return None;
            }
        }
    }
    Some((t0, t1))
}

/// Brick edge length in voxels — one occupancy bit per `BRICK³` block.
const BRICK: i32 = 8;

/// Per-chunk brick occupancy map for two-level DDA empty-space skip
/// (Substage DDA.3).
///
/// One bit per `BRICK³` block of the active chunk, set iff any voxel in
/// the block is solid. The ray steps the coarse brick grid (8× longer
/// strides) and only descends into a per-voxel walk inside occupied
/// bricks, so a ray through open air crosses ~`length / 8` empty bricks
/// instead of `length` air voxels — each of which would otherwise walk
/// the column slab chain via `surface_color`.
///
/// Built per frame from a [`GridView`] in [`render_dda`]. A persistent
/// per-chunk cache with edit-driven invalidation (locked decision #2 in
/// `PORTING-DDA.md`) is a later perf refinement.
#[derive(Debug)]
pub(crate) struct BrickMap {
    /// Brick counts along x / y / z (one entry per `BRICK³` cells).
    nb: [i32; 3],
    /// Brick occupancy bitset; brick `(bx, by, bz)` is bit
    /// `(bz * nb[1] + by) * nb[0] + bx`.
    bits: Vec<u64>,
    /// Super-brick counts (one entry per `BRICK³` *bricks* = `SUPER³`
    /// cells), `ceil(nb / BRICK)`.
    ns: [i32; 3],
    /// Super-brick occupancy (DDA.7 perf): a coarse level so a ray
    /// through open air above the terrain skips `SUPER` cells per outer
    /// step instead of `BRICK`. A super-brick is set iff any child brick
    /// is set.
    super_bits: Vec<u64>,
}

/// Super-brick edge in cells (`BRICK` bricks per axis).
const SUPER: i32 = BRICK * BRICK;

impl BrickMap {
    /// Scan every mip-`mip` column of `grid`, building brick + super-
    /// brick occupancy. `mip` must be `< grid.mip_count()`.
    #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
    fn build(grid: &GridView<'_>, mip: u32) -> Self {
        let vsid_m = (grid.vsid >> mip).max(1) as i32;
        let z_m = (crate::grid_view::CHUNK_SIZE_Z >> mip).max(1) as i32;
        let nb = [
            (vsid_m + BRICK - 1) / BRICK,
            (vsid_m + BRICK - 1) / BRICK,
            (z_m + BRICK - 1) / BRICK,
        ];
        let ns = [
            (nb[0] + BRICK - 1) / BRICK,
            (nb[1] + BRICK - 1) / BRICK,
            (nb[2] + BRICK - 1) / BRICK,
        ];
        let count = (nb[0] * nb[1] * nb[2]) as usize;
        let scount = (ns[0] * ns[1] * ns[2]) as usize;
        let mut bits = vec![0u64; count.div_ceil(64)];
        let mut super_bits = vec![0u64; scount.div_ceil(64)];
        for y in 0..vsid_m {
            for x in 0..vsid_m {
                let (bx, by) = (x / BRICK, y / BRICK);
                grid.for_each_run_mip(x as u32, y as u32, mip, |top, bot| {
                    for bz in (top / BRICK)..=((bot - 1) / BRICK) {
                        let idx = ((bz * nb[1] + by) * nb[0] + bx) as usize;
                        bits[idx / 64] |= 1u64 << (idx % 64);
                        let sidx =
                            (((bz / BRICK) * ns[1] + by / BRICK) * ns[0] + bx / BRICK) as usize;
                        super_bits[sidx / 64] |= 1u64 << (sidx % 64);
                    }
                });
            }
        }
        Self {
            nb,
            bits,
            ns,
            super_bits,
        }
    }

    /// Whether brick `b` is in range and holds any solid voxel.
    #[inline]
    #[allow(clippy::cast_sign_loss)]
    fn occupied(&self, b: [i32; 3]) -> bool {
        if b[0] < 0
            || b[0] >= self.nb[0]
            || b[1] < 0
            || b[1] >= self.nb[1]
            || b[2] < 0
            || b[2] >= self.nb[2]
        {
            return false;
        }
        let idx = ((b[2] * self.nb[1] + b[1]) * self.nb[0] + b[0]) as usize;
        (self.bits[idx / 64] >> (idx % 64)) & 1 != 0
    }

    /// Whether super-brick `s` is in range and holds any solid voxel.
    #[inline]
    #[allow(clippy::cast_sign_loss)]
    fn occupied_super(&self, s: [i32; 3]) -> bool {
        if s[0] < 0
            || s[0] >= self.ns[0]
            || s[1] < 0
            || s[1] >= self.ns[1]
            || s[2] < 0
            || s[2] >= self.ns[2]
        {
            return false;
        }
        let idx = ((s[2] * self.ns[1] + s[1]) * self.ns[0] + s[0]) as usize;
        (self.super_bits[idx / 64] >> (idx % 64)) & 1 != 0
    }
}

/// Per-axis 3D-DDA stepping state for a cell size of `cell` voxels.
/// `t_max[a]` is the ray parameter at which the next `a`-boundary is
/// crossed; `t_delta[a]` is the parameter increment per cell. An
/// axis-parallel component gets `t_max = t_delta = +inf` so it's never
/// chosen as the stepping axis.
pub(crate) fn dda_setup(
    origin: [f32; 3],
    dir: [f32; 3],
    cell: [i32; 3],
    cell_size: f32,
) -> ([i32; 3], [f32; 3], [f32; 3]) {
    let mut step = [0i32; 3];
    let mut t_max = [f32::INFINITY; 3];
    let mut t_delta = [f32::INFINITY; 3];
    for a in 0..3 {
        if dir[a] > 1e-9 {
            step[a] = 1;
            #[allow(clippy::cast_precision_loss)]
            let boundary = (cell[a] + 1) as f32 * cell_size;
            t_max[a] = (boundary - origin[a]) / dir[a];
            t_delta[a] = cell_size / dir[a];
        } else if dir[a] < -1e-9 {
            step[a] = -1;
            #[allow(clippy::cast_precision_loss)]
            let boundary = cell[a] as f32 * cell_size;
            t_max[a] = (boundary - origin[a]) / dir[a];
            t_delta[a] = -cell_size / dir[a];
        }
    }
    (step, t_max, t_delta)
}

/// Index of the axis with the smallest `t_max` (the next boundary the
/// ray crosses).
#[inline]
pub(crate) fn min_axis(t_max: [f32; 3]) -> usize {
    if t_max[0] <= t_max[1] && t_max[0] <= t_max[2] {
        0
    } else if t_max[1] <= t_max[2] {
        1
    } else {
        2
    }
}

/// Persistent, cross-frame brick occupancy cache (Substage DDA.7
/// perf). Keyed by `(chunk x, y, z, mip)` with the chunk's edit
/// `version`; an entry is reused until its chunk's version changes, so a
/// static / streamed-once world pays **zero** brick-build cost after the
/// first frame (the per-frame rebuild was the dominant DDA cost).
///
/// Owned by the caller across frames (the scene's `Grid`), populated
/// single-threaded via [`Self::ensure`], then borrowed immutably by the
/// parallel render bands.
#[derive(Debug, Default)]
pub struct BrickCache {
    maps: HashMap<(i32, i32, i32, u32), (u64, BrickMap)>,
}

impl BrickCache {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Ensure a current mip-`mip` brick map exists for `chunk` (built
    /// from `view`); rebuilds only when the cached `version` differs.
    pub fn ensure(&mut self, chunk: [i32; 3], mip: u32, version: u64, view: &GridView<'_>) {
        let key = (chunk[0], chunk[1], chunk[2], mip);
        let stale = self.maps.get(&key).map_or(true, |(v, _)| *v != version);
        if stale {
            self.maps.insert(key, (version, BrickMap::build(view, mip)));
        }
    }

    #[inline]
    fn get(&self, chunk: [i32; 3], mip: u32) -> Option<&BrickMap> {
        self.maps
            .get(&(chunk[0], chunk[1], chunk[2], mip))
            .map(|(_, m)| m)
    }

    /// Drop cached entries whose chunk fails `keep` — bounds memory as
    /// streaming evicts chunks. Called once per frame by the scene.
    pub fn retain_chunks(&mut self, keep: impl Fn([i32; 3]) -> bool) {
        self.maps.retain(|k, _| keep([k.0, k.1, k.2]));
    }
}

/// Build a throwaway [`BrickCache`] covering every populated chunk of
/// `grid` at the effective mip — for the sequential [`render_dda`] /
/// tests, where no persistent cache is threaded in. Returns
/// `(cache, effective_mip)`.
#[allow(clippy::cast_possible_wrap)]
fn local_cache(grid: &GridView<'_>, requested_mip: u32) -> (BrickCache, u32) {
    let mip = effective_mip(grid, requested_mip);
    let mut cache = BrickCache::new();
    if let Some(cg) = grid.chunk_grid {
        for dz in 0..cg.chunks_z as i32 {
            for dy in 0..cg.chunks_y as i32 {
                for dx in 0..cg.chunks_x as i32 {
                    let slot = ((dz * cg.chunks_y as i32 + dy) * cg.chunks_x as i32 + dx) as usize;
                    if let Some(Some(view)) = cg.chunks.get(slot) {
                        let ch = [
                            cg.origin_chunk_xy[0] + dx,
                            cg.origin_chunk_xy[1] + dy,
                            cg.origin_chunk_z + dz,
                        ];
                        cache.ensure(ch, mip, 0, view);
                    }
                }
            }
        }
    } else {
        cache.ensure([0, 0, 0], mip, 0, grid);
    }
    (cache, mip)
}

/// Clamp a requested render mip to one every populated chunk actually
/// has built — so the uniform-mip traversal never under-samples a chunk
/// that lacks the requested level (which would punch holes). `0` short-
/// circuits (always available).
#[must_use]
pub fn effective_mip(grid: &GridView<'_>, requested: u32) -> u32 {
    if requested == 0 {
        return 0;
    }
    let mut m = requested;
    if let Some(cg) = grid.chunk_grid {
        for c in cg.chunks.iter().flatten() {
            m = m.min(c.mip_count().saturating_sub(1));
        }
    } else {
        m = m.min(grid.mip_count().saturating_sub(1));
    }
    m
}

/// Cross-chunk voxel sampler (Substage DDA.4 / DDA.7).
///
/// Resolves a grid-local voxel coordinate to the chunk that owns it
/// (via [`GridView::chunk_at_xyz`]) and answers the DDA's per-voxel hit
/// query — brick-gated [`GridView::surface_color`]. It borrows the
/// shared immutable [`BrickMaps`] and caches the **current chunk**
/// (`cur_*`: view + brick-map reference): a ray usually stays in one
/// chunk for many voxels, so the per-voxel cost is a single index
/// compare + an O(1) brick bit test — no hashing, no mutation. Holding
/// only shared borrows, a `Sampler` is cheap to spin up per render band.
///
/// Single-chunk grids are the degenerate case: every voxel maps to
/// chunk `[0, 0, 0]` (= the view itself).
struct Sampler<'a> {
    grid: GridView<'a>,
    bricks: &'a BrickCache,
    /// Effective render mip (DDA.6). Traversal cells are mip-`mip`
    /// cells; sampling reads mip-`mip` data.
    mip: u32,
    /// Chunk size in mip-`mip` cells is a power of two; store it as
    /// `log2` (shift) + `size - 1` (mask) so [`Self::locate`] splits a
    /// cell into `(chunk, in-chunk)` with a shift + an `&` per axis
    /// instead of a signed `div_euclid` — the dominant per-cell cost.
    /// Arithmetic `>>` floors toward -∞ (= `div_euclid` for a positive
    /// power-of-two divisor) and `& mask` gives the non-negative
    /// remainder (= `rem_euclid`) even for negative cells (two's
    /// complement), so results are identical to the division form.
    xy_shift: u32,
    xy_mask: i32,
    z_shift: u32,
    z_mask: i32,
    cur_ch: [i32; 3],
    cur_view: Option<GridView<'a>>,
    cur_brick: Option<&'a BrickMap>,
    has_cur: bool,
}

impl<'a> Sampler<'a> {
    fn new(grid: GridView<'a>, bricks: &'a BrickCache, mip: u32) -> Self {
        let cs_xy = (grid.chunk_size_xy >> mip).max(1);
        let cs_z = (crate::grid_view::CHUNK_SIZE_Z >> mip).max(1);
        debug_assert!(
            cs_xy.is_power_of_two() && cs_z.is_power_of_two(),
            "chunk dims must be powers of two for the shift/mask split"
        );
        #[allow(clippy::cast_possible_wrap)]
        Self {
            grid,
            bricks,
            mip,
            xy_shift: cs_xy.trailing_zeros(),
            xy_mask: cs_xy as i32 - 1,
            z_shift: cs_z.trailing_zeros(),
            z_mask: cs_z as i32 - 1,
            cur_ch: [0; 3],
            cur_view: None,
            cur_brick: None,
            has_cur: false,
        }
    }

    /// Refresh the current-chunk cache (view + brick map) for `ch`.
    fn select_chunk(&mut self, ch: [i32; 3]) {
        if self.has_cur && self.cur_ch == ch {
            return;
        }
        self.cur_view = self.grid.chunk_at_xyz(ch);
        self.cur_brick = self.bricks.get(ch, self.mip);
        self.cur_ch = ch;
        self.has_cur = true;
    }

    /// Split a grid-local **mip-`mip` cell** index into `(chunk index,
    /// in-chunk mip-cell)` via shift + mask (see field docs). Chunk
    /// indices are mip-independent; only the per-chunk resolution
    /// shrinks with mip.
    #[allow(clippy::cast_sign_loss)]
    fn locate(&self, c: [i32; 3]) -> ([i32; 3], [u32; 3]) {
        let ch = [
            c[0] >> self.xy_shift,
            c[1] >> self.xy_shift,
            c[2] >> self.z_shift,
        ];
        let loc = [
            (c[0] & self.xy_mask) as u32,
            (c[1] & self.xy_mask) as u32,
            (c[2] & self.z_mask) as u32,
        ];
        (ch, loc)
    }

    /// Hit colour for grid-local mip-cell `c`, or `None` for air / empty
    /// chunk / uncoloured bedrock. Brick-gated so air inside a populated
    /// chunk costs only a bit test, not a slab walk.
    #[allow(clippy::cast_possible_wrap)]
    fn hit(&mut self, c: [i32; 3]) -> Option<u32> {
        #[cfg(test)]
        prof::SURF.with(|x| x.set(x.get() + 1));
        let (ch, loc) = self.locate(c);
        self.select_chunk(ch);
        let occupied = self.cur_brick.is_some_and(|bm| {
            bm.occupied([
                loc[0] as i32 / BRICK,
                loc[1] as i32 / BRICK,
                loc[2] as i32 / BRICK,
            ])
        });
        if !occupied {
            return None;
        }
        self.cur_view?
            .surface_color_mip(loc[0], loc[1], loc[2], self.mip)
    }

    /// Chunk size in mip-cells along XY / Z (always a power of two).
    #[inline]
    fn cells_per_chunk_xy(&self) -> i32 {
        1 << self.xy_shift
    }
    #[inline]
    fn cells_per_chunk_z(&self) -> i32 {
        1 << self.z_shift
    }

    /// Whether the brick at brick-index `brick` (in `BRICK`-mip-cell
    /// units) holds any solid voxel. Used by the outer brick-DDA to skip
    /// empty space `BRICK` cells at a time. Assumes bricks nest within
    /// chunks (caller gates on [`Self::cells_per_chunk_xy`]`>= BRICK`).
    #[allow(clippy::cast_sign_loss)]
    fn brick_occupied(&mut self, brick: [i32; 3]) -> bool {
        // First mip-cell of the brick (BRICK = 8 → `<< 3`).
        let c0 = [brick[0] << 3, brick[1] << 3, brick[2] << 3];
        let ch = [
            c0[0] >> self.xy_shift,
            c0[1] >> self.xy_shift,
            c0[2] >> self.z_shift,
        ];
        self.select_chunk(ch);
        self.cur_brick.is_some_and(|bm| {
            bm.occupied([
                (c0[0] & self.xy_mask) >> 3,
                (c0[1] & self.xy_mask) >> 3,
                (c0[2] & self.z_mask) >> 3,
            ])
        })
    }

    /// Whether the super-brick at super-index `s` (in `SUPER`-mip-cell
    /// units) holds any solid voxel. Outer-most empty-space skip (steps
    /// `SUPER` cells). Assumes super-bricks nest in chunks (caller gates
    /// on `cells_per_chunk >= SUPER`).
    #[allow(clippy::cast_sign_loss)]
    fn super_occupied(&mut self, s: [i32; 3]) -> bool {
        // First mip-cell of the super-brick (SUPER = 64 → `<< 6`).
        let c0 = [s[0] << 6, s[1] << 6, s[2] << 6];
        let ch = [
            c0[0] >> self.xy_shift,
            c0[1] >> self.xy_shift,
            c0[2] >> self.z_shift,
        ];
        self.select_chunk(ch);
        self.cur_brick.is_some_and(|bm| {
            bm.occupied_super([
                (c0[0] & self.xy_mask) >> 6,
                (c0[1] & self.xy_mask) >> 6,
                (c0[2] & self.z_mask) >> 6,
            ])
        })
    }
}

/// Walk mip-cells along the ray within `[lo_c, hi_c)` and return the
/// first solid hit, with leak-free empty-space skipping (DDA.7 redux).
///
/// **Why one continuous DDA, not nested level-walks.** The previous
/// design ran an outer brick/super DDA that *jumped* whole bricks and
/// only descended into occupied ones. Stepping a coarse cell at a time
/// lets the ray slip diagonally **past an occupied coarse cell it only
/// touches at a shared edge/corner** — a leak that showed as bright
/// sky seams across thin diagonal walls (the cave-demo report). Here a
/// *single* cell-granularity DDA carries the exact `(cellc, t_max)`
/// state for the whole ray; it only ever **fast-forwards across an
/// empty super-brick / brick**, where skipping cannot miss anything.
/// The exit axis lands on the integer box-boundary cell (no float
/// re-floor on the critical axis), so the entry cell of the next —
/// possibly occupied — box is always visited densely. Result: hits are
/// bit-identical to the dense per-cell reference, with the empty-space
/// speed-up retained.
///
/// `cell_size` is the mip-cell edge in mip-0 voxels (`1 << mip`);
/// `fwd_dot = dir·forward` → perpendicular depth.
#[allow(
    clippy::too_many_arguments,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn cell_walk_skip(
    origin: [f32; 3],
    dir: [f32; 3],
    fwd_dot: f32,
    sampler: &mut Sampler<'_>,
    lo_c: [i32; 3],
    hi_c: [i32; 3],
    cell_size: f32,
    t_enter: f32,
    t_exit: f32,
    max_dist: f32,
    env: &DdaEnv<'_>,
) -> Option<Hit> {
    let has_super = sampler.cells_per_chunk_xy() >= SUPER && sampler.cells_per_chunk_z() >= SUPER;
    let has_brick = sampler.cells_per_chunk_xy() >= BRICK && sampler.cells_per_chunk_z() >= BRICK;

    let start = t_enter + 1e-4;
    let p = [
        origin[0] + dir[0] * start,
        origin[1] + dir[1] * start,
        origin[2] + dir[2] * start,
    ];
    let mut cellc = [
        ((p[0] / cell_size).floor() as i32).clamp(lo_c[0], hi_c[0] - 1),
        ((p[1] / cell_size).floor() as i32).clamp(lo_c[1], hi_c[1] - 1),
        ((p[2] / cell_size).floor() as i32).clamp(lo_c[2], hi_c[2] - 1),
    ];
    let (step, mut t_max, t_delta) = dda_setup(origin, dir, cellc, cell_size);
    // Reciprocal direction → the per-skip box-boundary t and the t_max
    // refresh use multiplies instead of divisions (the dominant skip
    // cost). `0.0` where `step == 0` (that axis' t_max stays +∞).
    let inv = [
        if step[0] != 0 { 1.0 / dir[0] } else { 0.0 },
        if step[1] != 0 { 1.0 / dir[1] } else { 0.0 },
        if step[2] != 0 { 1.0 / dir[2] } else { 0.0 },
    ];
    let mut t_curr = t_enter;
    let mut last_axis = 3usize;

    // TV: front-to-back translucent accumulation. While no translucent voxel
    // is hit (`touched` stays false) every return is unchanged — the opaque
    // world renders bit-identically. `prev_*` drive per-span compositing (one
    // alpha layer per contiguous solid run or material change).
    let mut accum = [0.0f32; 3];
    let mut trans = 1.0f32;
    let mut touched = false;
    let mut prev_solid = false;
    let mut prev_mat = 0u8;

    // Each iteration either advances ≥1 cell (dense) or ≥1 box (skip),
    // so the total cell span bounds the loop.
    let span = (hi_c[0] - lo_c[0]) + (hi_c[1] - lo_c[1]) + (hi_c[2] - lo_c[2]);
    let max_steps = span.max(0) as usize + 16;
    for _ in 0..max_steps {
        if cellc[0] < lo_c[0]
            || cellc[0] >= hi_c[0]
            || cellc[1] < lo_c[1]
            || cellc[1] >= hi_c[1]
            || cellc[2] < lo_c[2]
            || cellc[2] >= hi_c[2]
        {
            return finalize_exit(touched, accum, trans, env, dir, max_dist);
        }
        let depth = t_curr * fwd_dot;
        if depth > max_dist || t_curr > t_exit {
            return finalize_exit(touched, accum, trans, env, dir, max_dist);
        }
        // Fog is fully opaque at `fog_max_dist`: nothing beyond is
        // visible, so stop the ray there and return the fog colour
        // rather than traversing (and skip/step-counting) to the far box
        // wall. Both correct and the dominant perf win for foggy worlds —
        // it caps every ray's length at the fog distance.
        if env.fog_max_dist > 0.0 && depth >= env.fog_max_dist {
            let fog = 0x8000_0000 | (env.fog_color & 0x00ff_ffff);
            let color = if touched {
                composite_over(accum, trans, fog)
            } else {
                fog
            };
            return Some(Hit {
                color,
                dist: env.fog_max_dist,
            });
        }

        // Empty-space skip: a whole empty super-brick, else an empty
        // brick. Skipping only empty boxes can never miss a surface.
        let skip_shift = if has_super
            && !sampler.super_occupied([cellc[0] >> 6, cellc[1] >> 6, cellc[2] >> 6])
        {
            Some(6u32)
        } else if has_brick
            && !sampler.brick_occupied([cellc[0] >> 3, cellc[1] >> 3, cellc[2] >> 3])
        {
            Some(3u32)
        } else {
            None
        };
        if let Some(sh) = skip_shift {
            #[cfg(test)]
            prof::BRICKS.with(|x| x.set(x.get() + 1));
            // Nearest box boundary along the ray (in cell units).
            let mut best_t = f32::INFINITY;
            let mut best_axis = 3usize;
            let mut plane = [0i32; 3];
            for a in 0..3 {
                if step[a] == 0 {
                    continue;
                }
                let idx = cellc[a] >> sh;
                plane[a] = if step[a] > 0 {
                    (idx + 1) << sh
                } else {
                    idx << sh
                };
                let tb = (plane[a] as f32 * cell_size - origin[a]) * inv[a];
                if tb < best_t {
                    best_t = tb;
                    best_axis = a;
                }
            }
            if best_axis == 3 {
                return finalize_exit(touched, accum, trans, env, dir, max_dist);
            }
            // Land just across the boundary; pin the exit axis to the
            // integer boundary cell so float error can't skip the next
            // box's entry cell. Other axes haven't crossed their box
            // boundary (best_t is the min), so the point's floor is safe.
            let pb = [
                origin[0] + dir[0] * (best_t + 1e-4),
                origin[1] + dir[1] * (best_t + 1e-4),
                origin[2] + dir[2] * (best_t + 1e-4),
            ];
            let mut nc = [
                (pb[0] / cell_size).floor() as i32,
                (pb[1] / cell_size).floor() as i32,
                (pb[2] / cell_size).floor() as i32,
            ];
            nc[best_axis] = if step[best_axis] > 0 {
                plane[best_axis]
            } else {
                plane[best_axis] - 1
            };
            // The skip crossed a box boundary; if that takes the ray out
            // of the grid box it has exited (sky) — return rather than
            // clamping back in-bounds, which would spin at the edge.
            if nc[0] < lo_c[0]
                || nc[0] >= hi_c[0]
                || nc[1] < lo_c[1]
                || nc[1] >= hi_c[1]
                || nc[2] < lo_c[2]
                || nc[2] >= hi_c[2]
            {
                return finalize_exit(touched, accum, trans, env, dir, max_dist);
            }
            cellc = nc;
            // Refresh t_max for the new cell (dir unchanged → t_delta and
            // step constant; axes with step==0 keep their +∞).
            for a in 0..3 {
                if step[a] > 0 {
                    t_max[a] = ((cellc[a] + 1) as f32 * cell_size - origin[a]) * inv[a];
                } else if step[a] < 0 {
                    t_max[a] = (cellc[a] as f32 * cell_size - origin[a]) * inv[a];
                }
            }
            t_curr = best_t.max(t_curr);
            last_axis = best_axis;
            prev_solid = false; // skipped empty space → next hit starts a run
            continue;
        }

        // Occupied brick: dense per-cell surface test.
        #[cfg(test)]
        prof::CELLS.with(|x| x.set(x.get() + 1));
        if let Some(color) = sampler.hit(cellc) {
            let bright_sub = side_shade_sub(env, last_axis, step);
            let lit = apply_fog(shade(color, bright_sub), depth.max(0.0), env);
            let m = terrain_material(env, color);
            if m.is_opaque() {
                // Opaque surface: the background. Return the first hit verbatim
                // when nothing translucent preceded it (bit-identical), else
                // composite the accumulated layers over it.
                let color = if touched {
                    composite_over(accum, trans, lit)
                } else {
                    lit
                };
                return Some(Hit {
                    color,
                    dist: depth.max(0.0),
                });
            }
            // Translucent: one alpha layer per solid-run entry or material
            // change (per-span — avoids the voxel-grid striping through a
            // thick glass/water slab).
            let mat_id = material_for_color(env.terrain_materials, color);
            if !prev_solid || mat_id != prev_mat {
                let a = f32::from(m.alpha) / 255.0;
                let c = rgb_to_f32(lit);
                accum[0] += trans * a * c[0];
                accum[1] += trans * a * c[1];
                accum[2] += trans * a * c[2];
                if !matches!(m.mode, roxlap_formats::material::BlendMode::Additive) {
                    trans *= 1.0 - a; // AlphaBlend occludes; Additive does not.
                }
                touched = true;
                prev_mat = mat_id;
                if trans < 1.0 / 256.0 {
                    return Some(Hit {
                        color: f32_to_rgb(accum),
                        dist: depth.max(0.0),
                    });
                }
            }
            prev_solid = true;
        } else {
            prev_solid = false;
        }
        let axis = min_axis(t_max);
        last_axis = axis;
        t_curr = t_max[axis];
        cellc[axis] += step[axis];
        t_max[axis] += t_delta[axis];
    }
    None
}

/// Per-face brightness reduction for the hit face. `axis` is the axis
/// the ray crossed to enter the hit voxel (`3` = entry voxel, no face);
/// `step[axis]` gives the crossing direction. Maps to the
/// `[x-, x+, y-, y+, z-, z+]` `side_shades` entry of the face the ray
/// looks at (a `+step` crossing enters through the low / `-` face).
#[inline]
fn side_shade_sub(env: &DdaEnv<'_>, axis: usize, step: [i32; 3]) -> u32 {
    if axis >= 3 {
        return 0;
    }
    let face = axis * 2 + usize::from(step[axis] < 0);
    env.side_shades[face].max(0) as u32
}

/// Cast one ray into the grid and return the first solid hit.
///
/// **DDA.4:** cross-chunk per-pixel 3D-DDA over the grid's full voxel
/// box ([`GridView::voxel_bounds`], spanning every chunk in XY **and**
/// Z). The [`Sampler`] resolves each stepped voxel to its chunk and
/// brick-gates the slab walk. Cross-chunk look-down (the case the
/// voxlap renderer needed the whole virtual-column stack for) falls out
/// of the box simply spanning `chunks_z` along Z.
fn cast_ray(
    origin: [f32; 3],
    dir: [f32; 3],
    forward: [f32; 3],
    sampler: &mut Sampler<'_>,
    settings: &OpticastSettings,
    env: &DdaEnv<'_>,
) -> Option<Hit> {
    let (lo_i, hi_i) = sampler.grid.voxel_bounds();
    #[allow(clippy::cast_precision_loss)]
    let lo_f = [lo_i[0] as f32, lo_i[1] as f32, lo_i[2] as f32];
    #[allow(clippy::cast_precision_loss)]
    let hi_f = [hi_i[0] as f32, hi_i[1] as f32, hi_i[2] as f32];
    let (t_enter, t_exit) = intersect_aabb(origin, dir, lo_f, hi_f)?;
    let fwd_dot = dir[0] * forward[0] + dir[1] * forward[1] + dir[2] * forward[2];
    #[allow(clippy::cast_precision_loss)]
    let max_dist = settings.max_scan_dist.max(1) as f32;
    let cell = 1i32 << sampler.mip;
    let cell_size = cell as f32;
    let lo_c = [
        lo_i[0].div_euclid(cell),
        lo_i[1].div_euclid(cell),
        lo_i[2].div_euclid(cell),
    ];
    let hi_c = [
        hi_i[0].div_euclid(cell),
        hi_i[1].div_euclid(cell),
        hi_i[2].div_euclid(cell),
    ];
    cell_walk_skip(
        origin, dir, fwd_dot, sampler, lo_c, hi_c, cell_size, t_enter, t_exit, max_dist, env,
    )
}

/// Render one grid into `sink` with per-pixel 3D-DDA.
///
/// `camera` is the grid-local pose, `settings`
/// ([`OpticastSettings`]) carries the projection + viewport (including
/// the `y_start..y_end` strip bound), and `grid` is the per-frame
/// [`GridView`] borrow. `pitch_pixels` is the framebuffer
/// row stride in pixels (matches `ScalarRasterizer::new`'s argument).
///
/// On a miss, a textured sky ([`DdaEnv::sky`]) is sampled per ray
/// direction and written at `+inf` depth; with no textured sky the miss
/// writes nothing, so the caller's solid sky pre-fill shows (the
/// `render_scene_composed` path pre-fills it).
pub fn render_dda(
    camera: &Camera,
    settings: &OpticastSettings,
    grid: GridView<'_>,
    pitch_pixels: usize,
    env: &DdaEnv<'_>,
    mip: u32,
    sink: &mut impl PixelSink,
) {
    let cs = camera_math::derive(
        camera,
        settings.xres,
        settings.yres,
        settings.hx,
        settings.hy,
        settings.hz,
    );

    // Sequential path builds a throwaway per-call cache (tests / single
    // grid). The parallel path takes a persistent cross-frame cache.
    let (cache, mip) = local_cache(&grid, mip);
    let mut sampler = Sampler::new(grid, &cache, mip);

    for py in settings.y_start..settings.y_end {
        let row = py as usize * pitch_pixels;
        for px in 0..settings.xres {
            if let Some((color, dist)) = pixel_result(&cs, settings, &mut sampler, env, px, py) {
                sink.put(row + px as usize, color, dist);
            }
        }
    }
}

/// Resolve one pixel: a shaded + fogged hit colour, a sampled textured
/// sky on a miss, or `None` (miss with no textured sky → caller's
/// pre-fill stands). Shared by the sequential ([`render_dda`]) and
/// parallel ([`render_dda_parallel`]) drivers.
#[inline]
fn pixel_result(
    cs: &CameraState,
    settings: &OpticastSettings,
    sampler: &mut Sampler<'_>,
    env: &DdaEnv<'_>,
    px: u32,
    py: u32,
) -> Option<(u32, f32)> {
    let (origin, dir) = pixel_ray(cs, settings, px, py);
    if let Some(hit) = cast_ray(origin, dir, cs.forward, sampler, settings, env) {
        Some((hit.color, hit.dist))
    } else {
        env.sky.map(|sky| (sample_sky(sky, dir), f32::INFINITY))
    }
}

/// Tile-parallel [`render_dda`] writing straight into `(fb, zb)`.
///
/// DDA pixels are independent, so the framebuffer splits into disjoint
/// horizontal bands rendered concurrently (rayon) — **bit-identical**
/// to the sequential render regardless of thread count, unlike voxlap's
/// per-strip discretisation. Each band spins up its own lightweight
/// [`Sampler`] over the shared, immutable `cache`.
///
/// `cache` must already hold current brick maps for every chunk at
/// `mip` (populate via [`BrickCache::ensure`]); `mip` is the effective
/// render mip ([`effective_mip`]). `(fb, zb)` use the standard
/// conventions (`0x80RRGGBB`; z = perp distance, smaller = closer); a
/// miss writes nothing unless [`DdaEnv::sky`] is set. `pitch_pixels` is
/// the row stride.
#[allow(clippy::cast_possible_truncation, clippy::too_many_arguments)]
pub fn render_dda_parallel(
    camera: &Camera,
    settings: &OpticastSettings,
    grid: GridView<'_>,
    fb: &mut [u32],
    zb: &mut [f32],
    pitch_pixels: usize,
    env: &DdaEnv<'_>,
    cache: &BrickCache,
    mip: u32,
) {
    debug_assert_eq!(fb.len(), zb.len());
    let (y0, y1) = (settings.y_start, settings.y_end);
    if y1 <= y0 {
        return;
    }
    let cs = camera_math::derive(
        camera,
        settings.xres,
        settings.yres,
        settings.hx,
        settings.hy,
        settings.hz,
    );
    let target = RasterTarget::new(fb, zb);

    // Split the y-range into ~one band per worker thread.
    let nthreads = rayon::current_num_threads().max(1);
    let rows = (y1 - y0) as usize;
    let band = rows.div_ceil(nthreads).max(1) as u32;
    let bands: Vec<(u32, u32)> = (y0..y1)
        .step_by(band as usize)
        .map(|s| (s, (s + band).min(y1)))
        .collect();

    bands.par_iter().for_each(|&(by0, by1)| {
        let mut sampler = Sampler::new(grid, cache, mip);
        for py in by0..by1 {
            let row = py as usize * pitch_pixels;
            for px in 0..settings.xres {
                if let Some((color, dist)) = pixel_result(&cs, settings, &mut sampler, env, px, py)
                {
                    let idx = row + px as usize;
                    // SAFETY: bands cover disjoint row ranges, so writes
                    // never alias across threads; `idx` is in-bounds for
                    // a `pitch * height`-sized buffer.
                    unsafe {
                        target.write_color(idx, color);
                        target.write_depth(idx, dist);
                    }
                }
            }
        }
    });
}

/// Dense per-voxel reference cast for a **single-chunk** grid: walks
/// every voxel of `[0, vsid)² × [0, CHUNK_SIZE_Z)` calling
/// [`GridView::surface_color`] directly — no brick gate, no chunk
/// resolution. The equivalence oracle the brickmap + sampler
/// [`cast_ray`] is checked against in tests.
#[cfg(test)]
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn cast_ray_reference(
    origin: [f32; 3],
    dir: [f32; 3],
    forward: [f32; 3],
    grid: &GridView<'_>,
    settings: &OpticastSettings,
) -> Option<Hit> {
    let nx = grid.vsid as f32;
    let nz = f32::from(u16::try_from(crate::grid_view::CHUNK_SIZE_Z).unwrap_or(256));
    #[allow(clippy::cast_possible_wrap)]
    let n_i = [
        grid.vsid as i32,
        grid.vsid as i32,
        crate::grid_view::CHUNK_SIZE_Z as i32,
    ];
    let (t_enter, t_exit) = intersect_aabb(origin, dir, [0.0; 3], [nx, nx, nz])?;
    let fwd_dot = dir[0] * forward[0] + dir[1] * forward[1] + dir[2] * forward[2];
    let max_dist = settings.max_scan_dist.max(1) as f32;

    let start = t_enter + 1e-4;
    let p = [
        origin[0] + dir[0] * start,
        origin[1] + dir[1] * start,
        origin[2] + dir[2] * start,
    ];
    let mut voxel = [
        (p[0].floor() as i32).clamp(0, n_i[0] - 1),
        (p[1].floor() as i32).clamp(0, n_i[1] - 1),
        (p[2].floor() as i32).clamp(0, n_i[2] - 1),
    ];
    let (step, mut t_max, t_delta) = dda_setup(origin, dir, voxel, 1.0);
    let mut t_curr = t_enter;
    let max_steps = (n_i[0] + n_i[1] + n_i[2]) as usize + 8;
    for _ in 0..max_steps {
        if voxel[0] < 0
            || voxel[0] >= n_i[0]
            || voxel[1] < 0
            || voxel[1] >= n_i[1]
            || voxel[2] < 0
            || voxel[2] >= n_i[2]
        {
            return None;
        }
        let depth = t_curr * fwd_dot;
        if depth > max_dist || t_curr > t_exit {
            return None;
        }
        #[allow(clippy::cast_sign_loss)]
        if let Some(color) = grid.surface_color(voxel[0] as u32, voxel[1] as u32, voxel[2] as u32) {
            return Some(Hit {
                color: shade(color, 0),
                dist: depth.max(0.0),
            });
        }
        let axis = min_axis(t_max);
        t_curr = t_max[axis];
        voxel[axis] += step[axis];
        t_max[axis] += t_delta[axis];
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Recording sink: collects `(idx, color, dist)` puts for tests.
    #[derive(Default)]
    struct Recorder {
        puts: Vec<(usize, u32, f32)>,
    }
    impl PixelSink for Recorder {
        fn put(&mut self, idx: usize, color: u32, dist: f32) {
            self.puts.push((idx, color, dist));
        }
    }

    fn oracle_camera() -> Camera {
        // Identity-basis camera at origin: ray math is integer-exact.
        Camera {
            pos: [0.0, 0.0, 0.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 0.0, 1.0],
            forward: [0.0, 1.0, 0.0],
        }
    }

    /// Render `grid` from `camera` into a `w × h` framebuffer and
    /// return the per-pixel hit mask (`true` where a ray hit a voxel).
    fn render_mask(grid: GridView<'_>, camera: &Camera, w: u32, h: u32) -> Vec<bool> {
        let n = (w as usize) * (h as usize);
        let mut fb = vec![0u32; n]; // sky sentinel = 0
        let mut zb = vec![f32::INFINITY; n];
        let settings = OpticastSettings::for_oracle_framebuffer(w, h);
        {
            let mut sink = RasterSink::new(&mut fb, &mut zb);
            render_dda(
                camera,
                &settings,
                grid,
                w as usize,
                &DdaEnv::default(),
                0,
                &mut sink,
            );
        }
        fb.iter().map(|&c| c != 0).collect()
    }

    /// A silhouette is "row-convex" if every framebuffer row's hit
    /// pixels form a single contiguous run (no interior gap). The
    /// voxlap silhouette notch is exactly such an interior gap, so this
    /// is the headline DDA.1 acceptance check.
    fn rows_have_no_holes(mask: &[bool], w: u32, h: u32) -> bool {
        let w = w as usize;
        for y in 0..h as usize {
            let row = &mask[y * w..(y + 1) * w];
            let first = row.iter().position(|&b| b);
            let last = row.iter().rposition(|&b| b);
            if let (Some(f), Some(l)) = (first, last) {
                if row[f..=l].iter().any(|&b| !b) {
                    return false;
                }
            }
        }
        true
    }

    /// Same contiguity check down each column.
    fn cols_have_no_holes(mask: &[bool], w: u32, h: u32) -> bool {
        let w = w as usize;
        let h = h as usize;
        for x in 0..w {
            let col: Vec<bool> = (0..h).map(|y| mask[y * w + x]).collect();
            let first = col.iter().position(|&b| b);
            let last = col.iter().rposition(|&b| b);
            if let (Some(f), Some(l)) = (first, last) {
                if col[f..=l].iter().any(|&b| !b) {
                    return false;
                }
            }
        }
        true
    }

    /// The principal-point pixel `(hx, hy)` looks straight down the
    /// forward axis, scaled by `hz`.
    #[test]
    fn center_pixel_ray_is_forward() {
        let settings = OpticastSettings::for_oracle_framebuffer(640, 480);
        let cs = camera_math::derive(&oracle_camera(), 640, 480, 320.0, 240.0, 320.0);
        // hx = hy = 320 / 240 → use the exact principal point.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let (origin, dir) = pixel_ray(&cs, &settings, settings.hx as u32, settings.hy as u32);
        assert_eq!(origin, [0.0, 0.0, 0.0]);
        // hz·forward = 320·[0,1,0].
        assert_eq!(
            dir.map(f32::to_bits),
            [0.0f32, 320.0, 0.0].map(f32::to_bits)
        );
    }

    /// Pixel `(0, 0)`'s ray equals `camera_math`'s `corn[0]` — proving
    /// the DDA renderer samples the same rays the voxlap frustum is
    /// built from.
    #[test]
    fn corner_pixel_ray_matches_camera_corn0() {
        let settings = OpticastSettings::for_oracle_framebuffer(640, 480);
        let cs = camera_math::derive(&oracle_camera(), 640, 480, 320.0, 240.0, 320.0);
        let (_origin, dir) = pixel_ray(&cs, &settings, 0, 0);
        assert_eq!(dir.map(f32::to_bits), cs.corn[0].map(f32::to_bits));
    }

    /// The renderer's independent slab decoder
    /// ([`GridView::voxel_color`]) must agree with the reference
    /// [`roxlap_formats::vxl::Vxl::voxel_color`] for every cell —
    /// including a column with an air gap, which exercises the
    /// ceiling-colour-list branch.
    #[test]
    fn gridview_voxel_color_matches_reference() {
        // Two solid runs per column separated by air → ceiling list.
        let vxl = roxlap_formats::vxl::Vxl::from_dense(8, |x, _, z| {
            let lo = (10..=12).contains(&z);
            let hi = (40..=42).contains(&z);
            (lo || hi).then_some(0x80_10_20_30 + x)
        });
        let grid = GridView::from_single_vxl(&vxl);
        for x in 0..8 {
            for y in 0..8 {
                for z in 0..64 {
                    assert_eq!(
                        grid.voxel_color(x, y, z),
                        vxl.voxel_color(x, y, z),
                        "mismatch at ({x},{y},{z})"
                    );
                }
            }
        }
    }

    /// An all-air grid produces no hits (every ray misses).
    #[test]
    fn empty_grid_no_hits() {
        let vxl = roxlap_formats::vxl::Vxl::empty(64);
        let grid = GridView::from_single_vxl(&vxl);
        let settings = OpticastSettings::for_oracle_framebuffer(64, 48);
        let mut rec = Recorder::default();
        render_dda(
            &oracle_camera(),
            &settings,
            grid,
            64,
            &DdaEnv::default(),
            0,
            &mut rec,
        );
        assert!(rec.puts.is_empty(), "all-air grid must produce no hits");
    }

    /// Camera above a solid floor, looking straight down: every ray
    /// hits, the recovered colour is the floor colour, and the centre
    /// pixel's depth ≈ the camera's height above the floor.
    #[test]
    fn floor_seen_from_above() {
        const FLOOR_Z: u32 = 40;
        const FLOOR_COL: u32 = 0x80_30_60_90;
        let vxl =
            roxlap_formats::vxl::Vxl::from_dense(32, |_, _, z| (z >= FLOOR_Z).then_some(FLOOR_COL));
        let grid = GridView::from_single_vxl(&vxl);

        // Eye above the floor (z is down), looking down (+z).
        let cam = Camera {
            pos: [16.0, 16.0, 10.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 1.0, 0.0],
            forward: [0.0, 0.0, 1.0],
        };
        let settings = OpticastSettings::for_oracle_framebuffer(48, 48);
        let mut rec = Recorder::default();
        render_dda(&cam, &settings, grid, 48, &DdaEnv::default(), 0, &mut rec);

        assert!(!rec.puts.is_empty(), "floor must be visible");
        // Centre pixel looks straight down → depth ≈ FLOOR_Z - eye_z.
        let centre = 24usize * 48 + 24;
        let hit = rec
            .puts
            .iter()
            .find(|(idx, _, _)| *idx == centre)
            .expect("centre ray must hit the floor");
        assert_eq!(hit.1 & 0x00ff_ffff, FLOOR_COL & 0x00ff_ffff);
        let expected = (FLOOR_Z as f32) - 10.0;
        assert!(
            (hit.2 - expected).abs() < 1.5,
            "centre depth {} not ≈ {}",
            hit.2,
            expected
        );
    }

    /// DDA.2: a camera looking at the horizon splits the frame into
    /// sky (upward rays miss → no write) and floor (downward rays hit).
    /// The top of the frame must be mostly sky, the bottom mostly
    /// floor.
    #[test]
    fn horizon_splits_sky_and_floor() {
        const FLOOR_Z: u32 = 40;
        let vxl = roxlap_formats::vxl::Vxl::from_dense(64, |_, _, z| {
            (z >= FLOOR_Z).then_some(0x80_44_66_88)
        });
        let grid = GridView::from_single_vxl(&vxl);

        // At z=30 (above the z=40 floor), looking +y horizontally,
        // down = +z. Upward rays (low py) escape through the box top
        // (z=0) → sky; downward rays (high py) strike the floor.
        let cam = Camera {
            pos: [32.0, 4.0, 30.0],
            right: [-1.0, 0.0, 0.0],
            down: [0.0, 0.0, 1.0],
            forward: [0.0, 1.0, 0.0],
        };
        let (w, h) = (64u32, 64u32);
        let mask = render_mask(grid, &cam, w, h);

        let count_band = |y0: usize, y1: usize| -> usize {
            (y0 * w as usize..y1 * w as usize)
                .filter(|&i| mask[i])
                .count()
        };
        let top = count_band(0, h as usize / 4);
        let bottom = count_band(3 * h as usize / 4, h as usize);
        assert!(mask.iter().any(|&b| b), "floor must be visible");
        assert!(mask.iter().any(|&b| !b), "sky must be visible");
        assert!(
            bottom > top,
            "bottom band ({bottom}) should hit more floor than top band ({top})"
        );
    }

    /// Render `grid` from `camera` with the dense reference cast (no
    /// brickmap), returning `(colour, depth)` buffers.
    fn render_reference(
        grid: GridView<'_>,
        camera: &Camera,
        w: u32,
        h: u32,
    ) -> (Vec<u32>, Vec<f32>) {
        let n = (w as usize) * (h as usize);
        let mut fb = vec![0u32; n];
        let mut zb = vec![f32::INFINITY; n];
        let settings = OpticastSettings::for_oracle_framebuffer(w, h);
        let cs = camera_math::derive(camera, w, h, settings.hx, settings.hy, settings.hz);
        for py in 0..h {
            for px in 0..w {
                let (o, d) = pixel_ray(&cs, &settings, px, py);
                if let Some(hit) = cast_ray_reference(o, d, cs.forward, &grid, &settings) {
                    let i = (py * w + px) as usize;
                    fb[i] = hit.color;
                    zb[i] = hit.dist;
                }
            }
        }
        (fb, zb)
    }

    /// Render `grid` from `camera` via the production brickmap path.
    fn render_brickmap(
        grid: GridView<'_>,
        camera: &Camera,
        w: u32,
        h: u32,
    ) -> (Vec<u32>, Vec<f32>) {
        render_brickmap_env(grid, camera, w, h, &DdaEnv::default())
    }

    /// As [`render_brickmap`] but with an explicit [`DdaEnv`] (fog /
    /// textured sky / side shades).
    fn render_brickmap_env(
        grid: GridView<'_>,
        camera: &Camera,
        w: u32,
        h: u32,
        env: &DdaEnv<'_>,
    ) -> (Vec<u32>, Vec<f32>) {
        let n = (w as usize) * (h as usize);
        let mut fb = vec![0u32; n];
        let mut zb = vec![f32::INFINITY; n];
        let settings = OpticastSettings::for_oracle_framebuffer(w, h);
        {
            let mut sink = RasterSink::new(&mut fb, &mut zb);
            render_dda(camera, &settings, grid, w as usize, env, 0, &mut sink);
        }
        (fb, zb)
    }

    /// Regression for the cave-demo "bright sky seams" report: the
    /// empty-space-skip walk must not leak past an occupied box the ray
    /// only grazes at a shared edge/corner. A 1-voxel-thick diagonal
    /// wall (`x+y==64`, voxels edge-connected) with air on both sides is
    /// the canonical case. The production skip walk must hit exactly the
    /// same pixels as the dense per-cell reference — zero divergence.
    #[test]
    fn no_sky_leak_through_diagonal_wall() {
        let vxl = roxlap_formats::vxl::Vxl::from_dense(64, |x, y, z| {
            ((x + y == 64) && (2..62).contains(&z)).then_some(0x80_40_80_60)
        });
        let grid = GridView::from_single_vxl(&vxl);
        let (w, h) = (160u32, 160u32);
        let c = [10.0, 10.0, 32.0];
        let poses = [
            Camera::from_yaw_pitch(c, 0.785, 0.0),
            Camera::from_yaw_pitch(c, 0.6, 0.1),
            Camera::from_yaw_pitch(c, 0.95, -0.1),
            Camera::from_yaw_pitch(c, 0.785, 0.3),
            Camera::from_yaw_pitch(c, 0.5, 0.0),
        ];
        for (i, cam) in poses.iter().enumerate() {
            let (fb_b, _) = render_brickmap(grid, cam, w, h);
            let (fb_r, _) = render_reference(grid, cam, w, h);
            let leak = (0..(w * h) as usize)
                .filter(|&k| (fb_b[k] != 0) != (fb_r[k] != 0))
                .count();
            assert_eq!(leak, 0, "pose {i}: {leak} px diverge from dense reference");
        }
    }

    /// TV terrain transparency: a glass-coloured voxel slab in front of an
    /// opaque floor. With no terrain material map the glass is an opaque first
    /// hit; with the map it becomes translucent and the floor tints through.
    #[test]
    fn terrain_glass_tints_floor_behind() {
        let glass = 0x80_40_C0_E0; // cyan
        let floor = 0x80_C0_40_40; // red
        let vxl = roxlap_formats::vxl::Vxl::from_dense(16, |_, _, z| {
            if z == 4 {
                Some(glass)
            } else if z >= 10 {
                Some(floor)
            } else {
                None
            }
        });
        let grid = GridView::from_single_vxl(&vxl);
        // Camera above the grid looking straight down (+z), centred.
        let cam = Camera {
            pos: [8.0, 8.0, 0.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 1.0, 0.0],
            forward: [0.0, 0.0, 1.0],
        };
        let (w, h) = (32u32, 32u32);
        let centre = (h / 2 * w + w / 2) as usize;

        // Opaque: the glass voxel stops the ray (no terrain materials).
        let (fb_op, _) = render_brickmap(grid, &cam, w, h);
        assert_eq!(
            fb_op[centre] & 0x00ff_ffff,
            0x0040_C0E0,
            "opaque glass first-hit"
        );

        // Translucent: glass colour → material 1 (alpha-blend).
        let mut table = MaterialTable::new();
        table.set(1, Material::alpha_blend(128));
        let env = DdaEnv {
            materials: Some(&table),
            terrain_materials: &[(glass & 0x00ff_ffff, 1)],
            ..DdaEnv::default()
        };
        let (fb_tr, _) = render_brickmap_env(grid, &cam, w, h, &env);
        assert_ne!(
            fb_tr[centre], fb_op[centre],
            "glass should composite over the floor, not stay opaque"
        );
        let r_op = (fb_op[centre] >> 16) & 0xff; // glass red ≈ 0x40
        let r_tr = (fb_tr[centre] >> 16) & 0xff; // + floor red bleeds in
        assert!(
            r_tr > r_op,
            "floor red tints through the glass (op={r_op:02x} tr={r_tr:02x})"
        );
    }

    /// DDA.5: distance fog blends a hit toward the fog colour. A far
    /// floor pixel is closer to the fog colour than a near one.
    #[test]
    fn distance_fog_blends_toward_fog_color() {
        let vxl =
            roxlap_formats::vxl::Vxl::from_dense(64, |_, _, z| (z >= 40).then_some(0x80_FF_FF_FF));
        let grid = GridView::from_single_vxl(&vxl);
        let cam = Camera {
            pos: [32.0, 2.0, 38.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 0.0, 1.0],
            forward: [0.0, 1.0, 0.0],
        };
        let env = DdaEnv {
            sky: None,
            fog_color: 0x00_00_00_00, // black fog → distance darkens
            fog_max_dist: 64.0,
            side_shades: [0; 6],
            materials: None,
            terrain_materials: &[],
        };
        let (w, h) = (64u32, 64u32);
        let (fog, _) = render_brickmap_env(grid, &cam, w, h, &env);
        let (nofog, zb) = render_brickmap(grid, &cam, w, h);
        let (idx, depth) = zb.iter().enumerate().filter(|(_, z)| z.is_finite()).fold(
            (0usize, 0.0f32),
            |acc, (i, &z)| {
                if z > acc.1 {
                    (i, z)
                } else {
                    acc
                }
            },
        );
        assert!(depth > 20.0, "need a deep pixel to test fog (got {depth})");
        let lum = |c: u32| (c & 0xff) + ((c >> 8) & 0xff) + ((c >> 16) & 0xff);
        assert!(
            lum(fog[idx]) < lum(nofog[idx]),
            "fogged pixel {:08x} not darker than {:08x}",
            fog[idx],
            nofog[idx]
        );
    }

    /// DDA.5: with a textured sky, miss pixels are filled from the sky
    /// panorama (direction-dependent) instead of left at the pre-fill.
    #[test]
    fn textured_sky_fills_misses() {
        let sky = crate::sky::Sky::blue_gradient();
        let vxl = roxlap_formats::vxl::Vxl::empty(32); // all air → all miss
        let grid = GridView::from_single_vxl(&vxl);
        let env = DdaEnv {
            sky: Some(&sky),
            fog_color: 0,
            fog_max_dist: 0.0,
            side_shades: [0; 6],
            materials: None,
            terrain_materials: &[],
        };
        let cam = Camera::from_yaw_pitch([16.0, 16.0, 128.0], 0.3, -0.4);
        let (w, h) = (48u32, 48u32);
        let (fb, _) = render_brickmap_env(grid, &cam, w, h, &env);
        assert!(fb.iter().all(|&c| c >> 24 == 0x80), "all misses sky-filled");
        let top = fb[0];
        let bottom = fb[(h - 1) as usize * w as usize];
        assert_ne!(top, bottom, "sky gradient should vary with elevation");
    }

    /// Sky elevation orientation matches the GPU `sky_color` (acos(-z)/π):
    /// looking **up** (−z) samples panorama column 0 (zenith), looking
    /// **down** (+z) samples the last column (nadir). Regression for the
    /// CPU up/down inversion.
    #[test]
    fn sky_elevation_zenith_at_column_zero() {
        let mut pixels = vec![0i32; 8];
        pixels[0] = 0x0011_1111; // zenith marker
        pixels[7] = 0x0099_9999; // nadir marker
        let sky = crate::sky::Sky::from_pixels(pixels, 8, 1);
        let up = sample_sky(&sky, [0.0, 0.0, -1.0]); // −z is up
        let down = sample_sky(&sky, [0.0, 0.0, 1.0]); // +z is down
        assert_eq!(
            up & 0x00ff_ffff,
            0x0011_1111,
            "looking up → column 0 (zenith)"
        );
        assert_eq!(
            down & 0x00ff_ffff,
            0x0099_9999,
            "looking down → last column (nadir)"
        );
    }

    /// `render_sky_fill` paints the panorama for a **gridless** view — the
    /// same per-pixel sky sample the miss-ray path uses, with no grid present
    /// (the CPU empty-scene background, matching the GPU).
    #[test]
    fn sky_fill_paints_panorama_gridless() {
        let sky = crate::sky::Sky::blue_gradient();
        let cam = Camera::from_yaw_pitch([0.0, 0.0, 0.0], 0.3, -0.4);
        let (w, h) = (48u32, 48u32);
        let cs = crate::camera_math::derive(&cam, w, h, 24.0, 24.0, 24.0);
        let settings = crate::opticast::OpticastSettings::for_oracle_framebuffer(w, h);
        let mut fb = vec![0u32; (w * h) as usize];
        render_sky_fill(&mut fb, w as usize, w, h, &cs, &settings, &sky);
        assert!(
            fb.iter().all(|&c| c >> 24 == 0x80),
            "every pixel sky-filled with the brightness byte set"
        );
        let top = fb[0];
        let bottom = fb[(h - 1) as usize * w as usize];
        assert_ne!(top, bottom, "sky gradient should vary with elevation");
    }

    /// DDA.5: side shading darkens the hit face by its `side_shades`
    /// entry. A top-facing floor (ray crosses +z to enter) gets the
    /// `z-` face reduction (index 4).
    #[test]
    fn side_shades_darken_hit_face() {
        let vxl =
            roxlap_formats::vxl::Vxl::from_dense(16, |_, _, z| (z >= 8).then_some(0x80_FF_FF_FF));
        let grid = GridView::from_single_vxl(&vxl);
        let cam = Camera {
            pos: [8.0, 8.0, 2.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 1.0, 0.0],
            forward: [0.0, 0.0, 1.0],
        };
        let centre = 16 * 32 + 16;
        let (plain, _) = render_brickmap(grid, &cam, 32, 32);
        let env = DdaEnv {
            sky: None,
            fog_color: 0,
            fog_max_dist: 0.0,
            side_shades: [0, 0, 0, 0, 0x40, 0],
            materials: None,
            terrain_materials: &[],
        };
        let (shaded, _) = render_brickmap_env(grid, &cam, 32, 32, &env);
        let lum = |c: u32| (c & 0xff) + ((c >> 8) & 0xff) + ((c >> 16) & 0xff);
        assert!(
            lum(shaded[centre]) < lum(plain[centre]),
            "side-shaded face {:08x} not darker than {:08x}",
            shaded[centre],
            plain[centre]
        );
    }

    /// The two-level brick-skip cast closely approximates the dense
    /// per-voxel reference. The outer brick DDA re-seeds the inner cell
    /// walk at each occupied brick, so a few silhouette-boundary pixels
    /// jitter by one voxel (different hit cell → different colour/depth)
    /// — visually invisible, and the gain is ~`BRICK`× fewer air steps.
    /// Assert the divergence is tiny: coverage (hit/sky mask) is nearly
    /// identical and only a small fraction of pixels differ. (The
    /// thread-invariance guarantee is the separate, exact
    /// `parallel_matches_sequential`.)
    #[test]
    fn brickmap_approximates_dense_reference() {
        // Rolling heightmap + a floating block (air above and below).
        let vxl = roxlap_formats::vxl::Vxl::from_dense(64, |x, y, z| {
            let surf = 30 + ((x / 5 + y / 7) % 11);
            let ground = z >= surf;
            let block = (20..=24).contains(&z) && (10..20).contains(&x) && (40..50).contains(&y);
            (ground || block).then_some(0x80_30_50_70 + (x ^ y) % 0x40)
        });
        let grid = GridView::from_single_vxl(&vxl);

        let (w, h) = (80u32, 80u32);
        let poses = [
            Camera::orbit(0.6, 0.5, 90.0, [32.0, 32.0, 40.0]),
            Camera::orbit(2.1, 0.2, 70.0, [32.0, 32.0, 35.0]),
            Camera::orbit(-1.0, 0.9, 120.0, [32.0, 32.0, 45.0]),
        ];
        let n = (w * h) as usize;
        for (i, cam) in poses.iter().enumerate() {
            let (fb_b, zb_b) = render_brickmap(grid, cam, w, h);
            let (fb_r, _zb_r) = render_reference(grid, cam, w, h);
            // Coverage (hit vs sky) must match almost exactly.
            let cov_b = fb_b.iter().filter(|&&c| c != 0).count();
            let cov_r = fb_r.iter().filter(|&&c| c != 0).count();
            assert!(cov_b > 200, "pose {i} rendered ~empty (cov {cov_b})");
            let cov_diff = cov_b.abs_diff(cov_r);
            assert!(
                cov_diff * 100 <= n, // < 1 % of pixels flip hit↔sky
                "pose {i} coverage diverged: brick {cov_b} vs dense {cov_r}"
            );
            // Colour diffs (boundary-voxel jitter) must be a small slice.
            let diffs = fb_b.iter().zip(&fb_r).filter(|(a, b)| a != b).count();
            assert!(
                diffs * 100 <= n * 3, // < 3 % of pixels differ
                "pose {i} too many pixel diffs vs dense: {diffs}/{n}"
            );
            // Depth must be sane (finite where hit), not wildly off.
            for k in 0..n {
                if fb_b[k] != 0 {
                    assert!(zb_b[k].is_finite(), "pose {i} px {k} non-finite depth");
                }
            }
        }
    }

    /// DDA.5: a voxel's baked brightness byte darkens its colour. A
    /// half-bright voxel (`a = 0x40`) renders at roughly half RGB; a
    /// full-bright one (`a = 0x80`) is unchanged.
    #[test]
    fn baked_brightness_darkens_color() {
        // Half brightness: alpha 0x40 (64/128). White RGB → ~mid grey.
        let dim =
            roxlap_formats::vxl::Vxl::from_dense(16, |_, _, z| (z >= 8).then_some(0x40_FF_FF_FF));
        let grid = GridView::from_single_vxl(&dim);
        let cam = Camera {
            pos: [8.0, 8.0, 2.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 1.0, 0.0],
            forward: [0.0, 0.0, 1.0],
        };
        let (fb, _) = render_brickmap(grid, &cam, 32, 32);
        let centre = 16 * 32 + 16;
        // 0xFF * 64 >> 7 = 127 per channel; alpha normalised to 0x80.
        assert_eq!(fb[centre], 0x80_7F_7F_7F, "got {:08x}", fb[centre]);

        // Full brightness passes RGB through unchanged.
        let full =
            roxlap_formats::vxl::Vxl::from_dense(16, |_, _, z| (z >= 8).then_some(0x80_FF_FF_FF));
        let gridf = GridView::from_single_vxl(&full);
        let (fbf, _) = render_brickmap(gridf, &cam, 32, 32);
        assert_eq!(fbf[centre], 0x80_FF_FF_FF, "got {:08x}", fbf[centre]);
    }

    /// DDA.4 headline gate: cross-chunk look-down. A camera in an
    /// all-air upper chunk (chz=0) looking straight down must see the
    /// floor in the *lower* stacked chunk (chz=1), through the chunk-Z
    /// boundary. This is exactly the case the voxlap renderer needed the
    /// whole virtual-column stack (S4B.6.j / VC) for; the DDA gets it
    /// for free from the outer box spanning `chunks_z`.
    #[test]
    fn cross_chunk_lookdown_sees_lower_stacked_floor() {
        const FLOOR_LOCAL_Z: u32 = 40;
        const FLOOR_COL: u32 = 0x80_22_88_44;
        let upper = roxlap_formats::vxl::Vxl::empty(32); // all air + bedrock
        let lower = roxlap_formats::vxl::Vxl::from_dense(32, |_, _, z| {
            (z >= FLOOR_LOCAL_Z).then_some(FLOOR_COL)
        });
        let v_up = GridView::from_single_vxl(&upper);
        let v_lo = GridView::from_single_vxl(&lower);
        // Z-stack: index (dz*chunks_y+dy)*chunks_x+dx → [upper, lower].
        let chunks = [Some(v_up), Some(v_lo)];
        let cg = crate::ChunkGrid {
            chunks: &chunks,
            origin_chunk_xy: [0, 0],
            origin_chunk_z: 0,
            chunks_x: 1,
            chunks_y: 1,
            chunks_z: 2,
        };
        let grid = GridView::from_chunk_grid(&cg, 32);

        // Camera in the upper chunk (world z=100), looking straight down.
        let cam = Camera {
            pos: [16.0, 16.0, 100.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 1.0, 0.0],
            forward: [0.0, 0.0, 1.0],
        };
        let (w, h) = (48u32, 48u32);
        let (fb, zb) = render_brickmap(grid, &cam, w, h);
        let centre = 24 * 48 + 24;
        assert!(
            fb[centre] & 0x00ff_ffff == FLOOR_COL & 0x00ff_ffff,
            "centre ray must reach the lower-chunk floor (got {:08x})",
            fb[centre]
        );
        // Floor world-z = 256 + 40 = 296; camera z = 100 → depth ≈ 196.
        let expected = 296.0 - 100.0;
        assert!(
            (zb[centre] - expected).abs() < 2.0,
            "look-down depth {} not ≈ {expected}",
            zb[centre]
        );
    }

    /// DDA.4: a floor spanning two side-by-side chunks (chunks_x=2)
    /// renders continuously across the chunk-XY seam — hits on both
    /// sides, no gap column.
    #[test]
    fn cross_chunk_xy_floor_is_seamless() {
        let mk = || {
            roxlap_formats::vxl::Vxl::from_dense(32, |_, _, z| (z >= 20).then_some(0x80_50_50_50))
        };
        let (c0, c1) = (mk(), mk());
        let v0 = GridView::from_single_vxl(&c0);
        let v1 = GridView::from_single_vxl(&c1);
        let chunks = [Some(v0), Some(v1)];
        let cg = crate::ChunkGrid {
            chunks: &chunks,
            origin_chunk_xy: [0, 0],
            origin_chunk_z: 0,
            chunks_x: 2,
            chunks_y: 1,
            chunks_z: 1,
        };
        let grid = GridView::from_chunk_grid(&cg, 32);

        // High above the seam (x=32), looking straight down.
        let cam = Camera {
            pos: [32.0, 16.0, 4.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 1.0, 0.0],
            forward: [0.0, 0.0, 1.0],
        };
        let (w, h) = (64u32, 64u32);
        let mask = render_mask(grid, &cam, w, h);
        // Both the left chunk (screen left) and right chunk (screen
        // right) must show floor on the centre row.
        let row = (h / 2) as usize * w as usize;
        let left = (0..w as usize / 2).filter(|&x| mask[row + x]).count();
        let right = (w as usize / 2..w as usize)
            .filter(|&x| mask[row + x])
            .count();
        assert!(
            left > 5 && right > 5,
            "seam not continuous: left={left} right={right}"
        );
    }

    /// Render `grid` from `camera` at render `mip` and return the hit
    /// mask.
    fn render_mask_mip(grid: GridView<'_>, camera: &Camera, w: u32, h: u32, mip: u32) -> Vec<bool> {
        let n = (w as usize) * (h as usize);
        let mut fb = vec![0u32; n];
        let mut zb = vec![f32::INFINITY; n];
        let settings = OpticastSettings::for_oracle_framebuffer(w, h);
        {
            let mut sink = RasterSink::new(&mut fb, &mut zb);
            render_dda(
                camera,
                &settings,
                grid,
                w as usize,
                &DdaEnv::default(),
                mip,
                &mut sink,
            );
        }
        fb.iter().map(|&c| c != 0).collect()
    }

    /// DDA.6: rendering a mip-built grid at a coarse mip stays complete
    /// (hole-free silhouette) with roughly the same screen coverage as
    /// mip 0 — LOD coarsens detail, it doesn't punch holes or shrink the
    /// shape. (DDA has no axis-aligned mip beam — the artifact is
    /// structurally impossible with honest per-cell traversal.)
    #[test]
    fn mip_render_is_coarse_but_complete() {
        let mut vxl = roxlap_formats::vxl::Vxl::from_dense(64, |x, y, z| {
            let surf = 24 + ((x / 3 + y / 5) % 17);
            (z >= surf).then_some(0x80_50_70_90)
        });
        vxl.generate_mips(4);
        assert!(vxl.mip_count() >= 3, "need mips built for this test");
        let grid = GridView::from_single_vxl(&vxl);
        let (w, h) = (96u32, 96u32);
        let cam = Camera::orbit(0.7, 0.6, 110.0, [32.0, 32.0, 36.0]);

        let m0 = render_mask_mip(grid, &cam, w, h, 0);
        let m2 = render_mask_mip(grid, &cam, w, h, 2);

        let c0 = m0.iter().filter(|&&b| b).count();
        let c2 = m2.iter().filter(|&&b| b).count();
        assert!(c0 > 200 && c2 > 200, "both mips visible (c0={c0} c2={c2})");
        // Coverage within ~30 % — a coarse-mip silhouette closely tracks
        // the fine one (LOD coarsens detail, it doesn't lose the shape).
        // (Terrain silhouettes are non-convex — sky shows through
        // valleys — so a hole-free invariant doesn't apply here; that's
        // the convex single-voxel test's job.)
        let ratio = c2 as f32 / c0 as f32;
        assert!(
            (0.7..1.4).contains(&ratio),
            "mip-2 coverage {c2} vs mip-0 {c0} (ratio {ratio:.2}) diverged"
        );
    }

    /// Headless perf bench (run: `cargo test -p roxlap-core --release
    /// dda::tests::bench_terrain -- --ignored --nocapture`). Single-
    /// thread `render_dda` over a hilly chunk at a horizon pose; prints
    /// ms/frame + per-frame traversal counters (cells / bricks /
    /// surface_color calls) to locate the bottleneck.
    #[test]
    #[ignore = "perf benchmark — run explicitly with --ignored"]
    fn bench_terrain() {
        use std::time::Instant;
        // Multi-chunk grid like the demo: NC×NC chunks of 128, hills.
        const NC: i32 = 6;
        let cs = crate::grid_view::CHUNK_SIZE_Z; // 256, but vsid is 128
        let _ = cs;
        let mut vxls: Vec<roxlap_formats::vxl::Vxl> = Vec::new();
        for cy in 0..NC {
            for cx in 0..NC {
                let (ox, oy) = (cx * 128, cy * 128);
                let mut v = roxlap_formats::vxl::Vxl::from_dense(128, |x, y, z| {
                    let (gx, gy) = (ox + x as i32, oy + y as i32);
                    let surf = 90 + ((gx / 7 + gy / 9).rem_euclid(40)) + ((gx / 23).rem_euclid(20));
                    (z as i32 >= surf).then_some(0x80_50_70_90 + (x ^ y) % 0x30)
                });
                v.generate_mips(4);
                vxls.push(v);
            }
        }
        let views: Vec<Option<GridView>> = vxls
            .iter()
            .map(|v| Some(GridView::from_single_vxl(v)))
            .collect();
        let cg = crate::ChunkGrid {
            chunks: &views,
            origin_chunk_xy: [0, 0],
            origin_chunk_z: 0,
            chunks_x: NC as u32,
            chunks_y: NC as u32,
            chunks_z: 1,
        };
        let grid = GridView::from_chunk_grid(&cg, 128);

        let (w, h) = (960u32, 600u32);
        let mut settings = OpticastSettings::for_oracle_framebuffer(w, h);
        settings.max_scan_dist = 512;
        let n = (w * h) as usize;
        let mut fb = vec![0u32; n];
        let mut zb = vec![f32::INFINITY; n];
        let centre = [f64::from(NC * 128) / 2.0, f64::from(NC * 128) / 2.0, 60.0];

        // Two poses: eye-level toward horizon (long rays) + looking down
        // at nearby terrain (short rays, demo-typical).
        let poses = [
            (
                "horizon",
                Camera::from_yaw_pitch([20.0, 20.0, 40.0], 0.6, 0.15),
            ),
            ("down", Camera::orbit(0.7, 1.0, 130.0, centre)),
        ];
        for (name, cam) in poses {
            {
                let mut sink = RasterSink::new(&mut fb, &mut zb);
                prof::reset();
                render_dda(
                    &cam,
                    &settings,
                    grid,
                    w as usize,
                    &DdaEnv::default(),
                    0,
                    &mut sink,
                );
            }
            let (cells, bricks, surf) = prof::read();
            let iters = 6;
            let t0 = Instant::now();
            for _ in 0..iters {
                let mut sink = RasterSink::new(&mut fb, &mut zb);
                render_dda(
                    &cam,
                    &settings,
                    grid,
                    w as usize,
                    &DdaEnv::default(),
                    0,
                    &mut sink,
                );
            }
            let ms = t0.elapsed().as_secs_f64() * 1000.0 / f64::from(iters);
            let hits = fb.iter().filter(|&&c| c != 0).count();
            eprintln!(
                "[{name}] {w}x{h} 1-thread: {ms:.1} ms | hits={hits}/{n} | per-px: cells={:.1} bricks={:.1} surf={:.1}",
                cells as f64 / n as f64,
                bricks as f64 / n as f64,
                surf as f64 / n as f64,
            );
        }
    }

    /// DDA.7: the tile-parallel driver is bit-identical to the
    /// sequential one — DDA pixels are independent, so banding can't
    /// change a pixel.
    #[test]
    fn parallel_matches_sequential() {
        let vxl = roxlap_formats::vxl::Vxl::from_dense(64, |x, y, z| {
            let surf = 28 + ((x / 4 + y / 6) % 13);
            (z >= surf).then_some(0x80_40_60_80 + (x ^ y) % 0x30)
        });
        let grid = GridView::from_single_vxl(&vxl);
        let (w, h) = (96u32, 96u32);
        let cam = Camera::orbit(0.8, 0.55, 100.0, [32.0, 32.0, 40.0]);
        let env = DdaEnv {
            sky: None,
            fog_color: 0x00_20_30_40,
            fog_max_dist: 120.0,
            side_shades: [0, 0, 0, 0, 0x30, 0x10],
            materials: None,
            terrain_materials: &[],
        };

        let (seq_fb, seq_zb) = render_brickmap_env(grid, &cam, w, h, &env);

        let n = (w * h) as usize;
        let mut par_fb = vec![0u32; n];
        let mut par_zb = vec![f32::INFINITY; n];
        let settings = OpticastSettings::for_oracle_framebuffer(w, h);
        let (cache, mip) = local_cache(&grid, 0);
        render_dda_parallel(
            &cam,
            &settings,
            grid,
            &mut par_fb,
            &mut par_zb,
            w as usize,
            &env,
            &cache,
            mip,
        );
        assert!(par_fb == seq_fb, "parallel colour differs from sequential");
        assert!(
            par_zb
                .iter()
                .zip(&seq_zb)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "parallel depth differs from sequential"
        );
    }

    /// DDA.2 correctness: a heightmap column's interior is solid even
    /// though voxlap only stores a colour for its surface. `voxel_color`
    /// returns `None` for an interior voxel, but `surface_color` must
    /// return the run's surface colour — otherwise oblique rays striking
    /// a cliff *side* would pass straight through (see-through terrain).
    #[test]
    fn cliff_side_is_solid_not_see_through() {
        const TOP_Z: u32 = 50;
        const COL: u32 = 0x80_77_88_99;
        let vxl = roxlap_formats::vxl::Vxl::from_dense(8, |_, _, z| (z >= TOP_Z).then_some(COL));
        let grid = GridView::from_single_vxl(&vxl);

        // Surface voxel: coloured directly.
        assert_eq!(grid.voxel_color(4, 4, TOP_Z), Some(COL));
        // Interior voxel: voxlap stores no colour …
        assert_eq!(grid.voxel_color(4, 4, 150), None);
        // … but it is solid, and surface_color bleeds the run-top colour
        // down the cliff face → a real hit, not see-through.
        assert_eq!(grid.surface_color(4, 4, 150), Some(COL));
        // Bedrock-style air above the surface stays air.
        assert_eq!(grid.surface_color(4, 4, 10), None);
    }

    /// DDA.2: a camera embedded in solid material hits its own voxel
    /// immediately — every ray reports a hit (no skip / no garbage).
    #[test]
    fn camera_inside_solid_hits_everywhere() {
        let vxl = roxlap_formats::vxl::Vxl::from_dense(16, |_, _, _| Some(0x80_55_55_55));
        let grid = GridView::from_single_vxl(&vxl);
        let cam = Camera {
            pos: [8.0, 8.0, 128.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 1.0, 0.0],
            forward: [0.0, 0.0, 1.0],
        };
        let (w, h) = (32u32, 32u32);
        let mask = render_mask(grid, &cam, w, h);
        assert!(
            mask.iter().all(|&b| b),
            "every ray must hit when the camera is inside solid"
        );
    }

    /// Headline DDA.1 gate: a single solid voxel viewed obliquely
    /// projects to a convex silhouette with **no interior holes** —
    /// the artifact class (`tiny_grid_1x1x1` silhouette notch) the
    /// voxlap renderer cannot avoid. DDA casts independent per-pixel
    /// rays, so the silhouette is hole-free by construction.
    #[test]
    fn single_voxel_silhouette_has_no_notch() {
        const C: u32 = 0x80_FF_80_40;
        let vxl = roxlap_formats::vxl::Vxl::from_dense(16, |x, y, z| {
            (x == 8 && y == 8 && z == 8).then_some(C)
        });
        let grid = GridView::from_single_vxl(&vxl);

        // Orbit the voxel centre obliquely so all three faces show and
        // the silhouette is a sizeable hexagon (dist 4 → ~12 px wide).
        let cam = Camera::orbit(0.7, 0.6, 4.0, [8.5, 8.5, 8.5]);
        let (w, h) = (96u32, 96u32);
        let mask = render_mask(grid, &cam, w, h);

        let hits = mask.iter().filter(|&&b| b).count();
        assert!(
            hits > 30,
            "silhouette too small to be meaningful: {hits} px"
        );
        assert!(
            rows_have_no_holes(&mask, w, h),
            "row-interior gap in single-voxel silhouette (notch)"
        );
        assert!(
            cols_have_no_holes(&mask, w, h),
            "column-interior gap in single-voxel silhouette (notch)"
        );
    }
}
