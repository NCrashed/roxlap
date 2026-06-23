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
//! **Stage status — DDA.1 (single-chunk dense traversal).** Each pixel
//! casts one ray and walks unit voxels of the active chunk via a 3D-DDA
//! (Amanatides–Woo), sampling [`GridView::voxel_color`] until the first
//! textured cell. Flat voxel colour, no brickmap, no cross-chunk step,
//! no shading yet (DDA.3 / DDA.4 / DDA.5). The framebuffer keeps
//! whatever the caller pre-filled (sky) wherever a ray misses.
//!
//! Buffer conventions match the rest of the engine so this backend is
//! a drop-in for `opticast`: colour is packed `0x80RRGGBB`; depth is
//! perpendicular distance from the camera with **smaller = closer**
//! (so [`crate::scalar_rasterizer`]'s `compose_into` min-z merge works
//! unchanged).

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

/// Cast one ray into the grid and return the first solid hit.
///
/// **DDA.1:** single-chunk dense 3D-DDA (Amanatides–Woo "A Fast Voxel
/// Traversal Algorithm", 1987) over the active chunk's `[0, vsid) ×
/// [0, vsid) × [0, CHUNK_SIZE_Z)` voxel box. Walks unit voxels along
/// the ray, sampling [`GridView::voxel_color`] at each; the first
/// textured cell is the hit. No brickmap (DDA.3), no cross-chunk step
/// (DDA.4), no shading (DDA.5) yet — flat voxel colour.
///
/// `forward` is the camera forward axis; the returned [`Hit::dist`] is
/// the hit's perpendicular (forward-projected) distance, matching the
/// engine's z-buffer convention (smaller = closer).
fn cast_ray(
    origin: [f32; 3],
    dir: [f32; 3],
    forward: [f32; 3],
    grid: &GridView<'_>,
    settings: &OpticastSettings,
) -> Option<Hit> {
    // u32 → f32 / i32 are exact for realistic chunk dimensions.
    #[allow(clippy::cast_precision_loss)]
    let nx = grid.vsid as f32;
    #[allow(clippy::cast_precision_loss)]
    let nz = f32::from(u16::try_from(crate::grid_view::CHUNK_SIZE_Z).unwrap_or(256));
    let n = [nx, nx, nz];
    let n_i = [
        grid.vsid as i32,
        grid.vsid as i32,
        crate::grid_view::CHUNK_SIZE_Z as i32,
    ];

    let (t_enter, t_exit) = intersect_aabb(origin, dir, [0.0; 3], n)?;

    // Entry point, nudged a hair past the face so floor() lands inside
    // the first cell rather than on the boundary.
    let eps = (t_exit - t_enter) * 1e-4 + 1e-4;
    let p = [
        origin[0] + dir[0] * (t_enter + eps),
        origin[1] + dir[1] * (t_enter + eps),
        origin[2] + dir[2] * (t_enter + eps),
    ];

    // Current voxel, clamped into range against floating-point slop on
    // the entry face.
    #[allow(clippy::cast_possible_truncation)]
    let mut voxel = [
        (p[0].floor() as i32).clamp(0, n_i[0] - 1),
        (p[1].floor() as i32).clamp(0, n_i[1] - 1),
        (p[2].floor() as i32).clamp(0, n_i[2] - 1),
    ];

    // Per-axis step direction, the t to the first crossing, and the t
    // increment per voxel. Axis-parallel rays get `t_max = +inf` so
    // they're never chosen as the stepping axis.
    let mut step = [0i32; 3];
    let mut t_max = [f32::INFINITY; 3];
    let mut t_delta = [f32::INFINITY; 3];
    for a in 0..3 {
        if dir[a] > 1e-9 {
            step[a] = 1;
            #[allow(clippy::cast_precision_loss)]
            let boundary = (voxel[a] + 1) as f32;
            t_max[a] = (boundary - origin[a]) / dir[a];
            t_delta[a] = 1.0 / dir[a];
        } else if dir[a] < -1e-9 {
            step[a] = -1;
            #[allow(clippy::cast_precision_loss)]
            let boundary = voxel[a] as f32;
            t_max[a] = (boundary - origin[a]) / dir[a];
            t_delta[a] = -1.0 / dir[a];
        }
    }

    #[allow(clippy::cast_precision_loss)]
    let max_dist = settings.max_scan_dist.max(1) as f32;
    let mut t_curr = t_enter;
    // Hard iteration cap: a ray can cross at most this many voxel
    // boundaries inside the box. Guards against floating-point stalls.
    let max_steps = (n_i[0] + n_i[1] + n_i[2]) as usize + 8;

    for _ in 0..max_steps {
        if voxel[0] < 0
            || voxel[0] >= n_i[0]
            || voxel[1] < 0
            || voxel[1] >= n_i[1]
            || voxel[2] < 0
            || voxel[2] >= n_i[2]
        {
            return None; // walked out of the chunk
        }
        // `t_curr` is the entry distance of `voxel`; bound the scan by
        // forward-projected distance (the z-buffer metric).
        let depth = t_curr * (dir[0] * forward[0] + dir[1] * forward[1] + dir[2] * forward[2]);
        if depth > max_dist {
            return None;
        }
        #[allow(clippy::cast_sign_loss)]
        if let Some(color) = grid.voxel_color(voxel[0] as u32, voxel[1] as u32, voxel[2] as u32) {
            return Some(Hit {
                color,
                dist: depth.max(0.0),
            });
        }

        // Advance to the next voxel across the nearest axis boundary.
        let axis = if t_max[0] <= t_max[1] && t_max[0] <= t_max[2] {
            0
        } else if t_max[1] <= t_max[2] {
            1
        } else {
            2
        };
        t_curr = t_max[axis];
        voxel[axis] += step[axis];
        t_max[axis] += t_delta[axis];
    }
    None
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

    for py in settings.y_start..settings.y_end {
        let row = py as usize * pitch_pixels;
        for px in 0..settings.xres {
            let (origin, dir) = pixel_ray(&cs, settings, px, py);
            if let Some(hit) = cast_ray(origin, dir, cs.forward, &grid, settings) {
                sink.put(row + px as usize, hit.color, hit.dist);
            }
        }
    }
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
