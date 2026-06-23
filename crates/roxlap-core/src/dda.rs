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
//! **Stage status — DDA.5 (baked brightness shading).** Each pixel
//! casts one ray and walks unit voxels over the grid's full voxel box
//! ([`GridView::voxel_bounds`], spanning every chunk in XY **and** Z)
//! via a 3D-DDA (Amanatides–Woo). A [`Sampler`] resolves each voxel to
//! its chunk ([`GridView::chunk_at_xyz`]) and brick-gates the
//! [`GridView::surface_color`] slab walk. The hit colour is shaded by
//! the voxel's baked directional brightness ([`shade`]) — matching the
//! GPU marcher — so lit scenes render correctly and editor relight is
//! free. Misses leave the destination untouched, so the caller's sky
//! pre-fill shows through. Remaining DDA.5 polish: distance fog,
//! textured sky panorama, `side_shades` face tint.
//!
//! Buffer conventions match the rest of the engine so this backend is
//! a drop-in for `opticast`: colour is packed `0x80RRGGBB`; depth is
//! perpendicular distance from the camera with **smaller = closer**
//! (so [`crate::scalar_rasterizer`]'s `compose_into` min-z merge works
//! unchanged).

use std::collections::HashMap;

use crate::camera_math::{self, CameraState};
use crate::grid_view::GridView;
use crate::opticast::OpticastSettings;
use crate::scalar_rasterizer::RasterTarget;
use crate::Camera;

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
#[inline]
fn shade(color: u32) -> u32 {
    let a = (color >> 24) & 0xff;
    let ch = |shift: u32| -> u32 { ((((color >> shift) & 0xff) * a) >> 7).min(255) };
    0x8000_0000 | (ch(16) << 16) | (ch(8) << 8) | ch(0)
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
fn intersect_aabb(o: [f32; 3], dir: [f32; 3], lo: [f32; 3], hi: [f32; 3]) -> Option<(f32, f32)> {
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
pub(crate) struct BrickMap {
    /// Brick counts along x / y / z.
    nb: [i32; 3],
    /// Occupancy bitset; brick `(bx, by, bz)` is bit
    /// `(bz * nb[1] + by) * nb[0] + bx`.
    bits: Vec<u64>,
}

impl BrickMap {
    /// Scan every column of `grid` once, marking the brick of each
    /// solid run. `O(vsid² · slabs)` — amortised across all pixels of
    /// the frame.
    #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
    fn build(grid: &GridView<'_>) -> Self {
        let vsid = grid.vsid as i32;
        let nb = [
            (vsid + BRICK - 1) / BRICK,
            (vsid + BRICK - 1) / BRICK,
            (crate::grid_view::CHUNK_SIZE_Z as i32 + BRICK - 1) / BRICK,
        ];
        let count = (nb[0] * nb[1] * nb[2]) as usize;
        let mut bits = vec![0u64; count.div_ceil(64)];
        for y in 0..vsid {
            for x in 0..vsid {
                let (bx, by) = (x / BRICK, y / BRICK);
                grid.for_each_run(x as u32, y as u32, |top, bot| {
                    let bz0 = top / BRICK;
                    let bz1 = (bot - 1) / BRICK;
                    for bz in bz0..=bz1 {
                        let idx = ((bz * nb[1] + by) * nb[0] + bx) as usize;
                        bits[idx / 64] |= 1u64 << (idx % 64);
                    }
                });
            }
        }
        Self { nb, bits }
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
}

/// Per-axis 3D-DDA stepping state for a cell size of `cell` voxels.
/// `t_max[a]` is the ray parameter at which the next `a`-boundary is
/// crossed; `t_delta[a]` is the parameter increment per cell. An
/// axis-parallel component gets `t_max = t_delta = +inf` so it's never
/// chosen as the stepping axis.
fn dda_setup(
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
fn min_axis(t_max: [f32; 3]) -> usize {
    if t_max[0] <= t_max[1] && t_max[0] <= t_max[2] {
        0
    } else if t_max[1] <= t_max[2] {
        1
    } else {
        2
    }
}

/// Cross-chunk voxel sampler (Substage DDA.4).
///
/// Resolves a grid-local voxel coordinate to the chunk that owns it
/// (via [`GridView::chunk_at_xyz`]) and answers the DDA's per-voxel hit
/// query — brick-gated [`GridView::surface_color`]. Two caches keep the
/// hot loop cheap:
///
/// * the **current chunk** (`cur_*`) — a ray usually stays in one chunk
///   for many voxels, so chunk resolution is a single compare;
/// * a **per-frame brick cache** (`bricks`) keyed by chunk index, built
///   lazily on first touch and reused across every ray of the frame.
///
/// Single-chunk grids are the degenerate case: every voxel maps to
/// chunk `[0, 0, 0]` (= the view itself), so the path is identical to
/// the DDA.3 single-chunk render.
struct Sampler<'a> {
    grid: GridView<'a>,
    cs_xy: i32,
    cs_z: i32,
    cur_ch: [i32; 3],
    cur_view: Option<GridView<'a>>,
    has_cur: bool,
    bricks: HashMap<[i32; 3], BrickMap>,
}

impl<'a> Sampler<'a> {
    fn new(grid: GridView<'a>) -> Self {
        #[allow(clippy::cast_possible_wrap)]
        let cs_xy = grid.chunk_size_xy as i32;
        #[allow(clippy::cast_possible_wrap)]
        let cs_z = crate::grid_view::CHUNK_SIZE_Z as i32;
        Self {
            grid,
            cs_xy,
            cs_z,
            cur_ch: [0; 3],
            cur_view: None,
            has_cur: false,
            bricks: HashMap::new(),
        }
    }

    /// Resolve a chunk index to its view, caching the last lookup.
    fn chunk_view(&mut self, ch: [i32; 3]) -> Option<GridView<'a>> {
        if self.has_cur && self.cur_ch == ch {
            return self.cur_view;
        }
        let v = self.grid.chunk_at_xyz(ch);
        self.cur_ch = ch;
        self.cur_view = v;
        self.has_cur = true;
        v
    }

    /// Split a grid-local voxel into `(chunk index, in-chunk local)`.
    #[allow(clippy::cast_sign_loss)]
    fn locate(&self, g: [i32; 3]) -> ([i32; 3], [u32; 3]) {
        let ch = [
            g[0].div_euclid(self.cs_xy),
            g[1].div_euclid(self.cs_xy),
            g[2].div_euclid(self.cs_z),
        ];
        let loc = [
            g[0].rem_euclid(self.cs_xy) as u32,
            g[1].rem_euclid(self.cs_xy) as u32,
            g[2].rem_euclid(self.cs_z) as u32,
        ];
        (ch, loc)
    }

    /// Hit colour for grid-local voxel `g`, or `None` for air / empty
    /// chunk / uncoloured bedrock. Brick-gated so air inside a populated
    /// chunk costs only a bit test, not a slab walk.
    #[allow(clippy::cast_possible_wrap)]
    fn hit(&mut self, g: [i32; 3]) -> Option<u32> {
        let (ch, loc) = self.locate(g);
        let view = self.chunk_view(ch)?;
        let occupied = {
            let bricks = &mut self.bricks;
            let bm = bricks.entry(ch).or_insert_with(|| BrickMap::build(&view));
            bm.occupied([
                loc[0] as i32 / BRICK,
                loc[1] as i32 / BRICK,
                loc[2] as i32 / BRICK,
            ])
        };
        if !occupied {
            return None;
        }
        view.surface_color(loc[0], loc[1], loc[2])
    }
}

/// Walk unit voxels along the ray within the half-open grid-local voxel
/// box `[lo, hi)`, between ray parameters `t_lo` (box entry) and `t_hi`
/// (box exit). Returns the first solid + renderable cell (via
/// [`Sampler::hit`]) as a [`Hit`], or `None` if the ray leaves the
/// box / t-range / `max_dist` first. `fwd_dot = dir·forward` converts a
/// ray parameter to perpendicular depth.
///
/// The sampler resolves each voxel's chunk and brick-gates the slab
/// walk, so air — whether in an empty chunk or an empty brick — is
/// crossed with cheap integer DDA stepping only.
#[allow(
    clippy::too_many_arguments,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn voxel_walk(
    origin: [f32; 3],
    dir: [f32; 3],
    fwd_dot: f32,
    sampler: &mut Sampler<'_>,
    lo: [i32; 3],
    hi: [i32; 3],
    t_lo: f32,
    t_hi: f32,
    max_dist: f32,
) -> Option<Hit> {
    let start = t_lo + 1e-4;
    let p = [
        origin[0] + dir[0] * start,
        origin[1] + dir[1] * start,
        origin[2] + dir[2] * start,
    ];
    let mut voxel = [
        (p[0].floor() as i32).clamp(lo[0], hi[0] - 1),
        (p[1].floor() as i32).clamp(lo[1], hi[1] - 1),
        (p[2].floor() as i32).clamp(lo[2], hi[2] - 1),
    ];
    let (step, mut t_max, t_delta) = dda_setup(origin, dir, voxel, 1.0);
    let mut t_curr = t_lo;
    let max_steps = ((hi[0] - lo[0]) + (hi[1] - lo[1]) + (hi[2] - lo[2])) as usize + 8;
    for _ in 0..max_steps {
        if voxel[0] < lo[0]
            || voxel[0] >= hi[0]
            || voxel[1] < lo[1]
            || voxel[1] >= hi[1]
            || voxel[2] < lo[2]
            || voxel[2] >= hi[2]
        {
            return None;
        }
        let depth = t_curr * fwd_dot;
        if depth > max_dist || t_curr > t_hi {
            return None;
        }
        if let Some(color) = sampler.hit(voxel) {
            return Some(Hit {
                color: shade(color),
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
    voxel_walk(
        origin, dir, fwd_dot, sampler, lo_i, hi_i, t_enter, t_exit, max_dist,
    )
}

/// Render one grid into `sink` with per-pixel 3D-DDA.
///
/// Mirrors [`crate::opticast::opticast`]'s contract: `camera` is the
/// grid-local pose, `settings` carries the projection + viewport
/// (including the `y_start..y_end` strip bound), and `grid` is the
/// per-frame [`GridView`] borrow. `pitch_pixels` is the framebuffer
/// row stride in pixels (matches `ScalarRasterizer::new`'s argument).
///
/// Misses write nothing, so the caller must pre-fill the framebuffer
/// with sky (the `render_scene_composed` path already does).
pub fn render_dda(
    camera: &Camera,
    settings: &OpticastSettings,
    grid: GridView<'_>,
    pitch_pixels: usize,
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

    // One sampler per frame: caches resolved chunks + per-chunk brick
    // maps, reused across every ray.
    let mut sampler = Sampler::new(grid);

    for py in settings.y_start..settings.y_end {
        let row = py as usize * pitch_pixels;
        for px in 0..settings.xres {
            let (origin, dir) = pixel_ray(&cs, settings, px, py);
            if let Some(hit) = cast_ray(origin, dir, cs.forward, &mut sampler, settings) {
                sink.put(row + px as usize, hit.color, hit.dist);
            }
        }
    }
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
                color: shade(color),
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
            render_dda(camera, &settings, grid, w as usize, &mut sink);
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
        render_dda(&oracle_camera(), &settings, grid, 64, &mut rec);
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
        render_dda(&cam, &settings, grid, 48, &mut rec);

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
        let n = (w as usize) * (h as usize);
        let mut fb = vec![0u32; n];
        let mut zb = vec![f32::INFINITY; n];
        let settings = OpticastSettings::for_oracle_framebuffer(w, h);
        {
            let mut sink = RasterSink::new(&mut fb, &mut zb);
            render_dda(camera, &settings, grid, w as usize, &mut sink);
        }
        (fb, zb)
    }

    /// The brickmap two-level cast must be bit-identical to the dense
    /// per-voxel reference — empty-space skip is a speed optimisation,
    /// not a result change. Exercised over varied terrain (hills + a
    /// floating block + an air gap) from several oblique poses.
    #[test]
    fn brickmap_matches_dense_reference() {
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
        for (i, cam) in poses.iter().enumerate() {
            let (fb_b, zb_b) = render_brickmap(grid, cam, w, h);
            let (fb_r, zb_r) = render_reference(grid, cam, w, h);
            assert!(fb_b == fb_r, "colour mismatch vs reference at pose {i}");
            assert!(
                zb_b.iter()
                    .zip(&zb_r)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "depth mismatch vs reference at pose {i}"
            );
            // Sanity: the scene is actually visible.
            assert!(fb_b.iter().any(|&c| c != 0), "pose {i} rendered empty");
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
