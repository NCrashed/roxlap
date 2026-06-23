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
//! **Stage status — DDA.0 (scaffold).** This module currently wires
//! the seam only: the pixel loop, the camera→ray projection, and the
//! [`PixelSink`] output abstraction. [`cast_ray`] is a stub that finds
//! no geometry, so [`render_dda`] writes nothing — the framebuffer
//! keeps whatever the caller pre-filled (sky). DDA.1 replaces the stub
//! with the single-chunk dense 3D-DDA traversal.
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

/// Cast one ray into the grid and return the first solid hit.
///
/// **DDA.0 stub:** always returns `None`. DDA.1 implements the
/// single-chunk dense 3D-DDA (Amanatides–Woo) here; DDA.3 adds the
/// brickmap empty-space skip; DDA.4 adds the cross-chunk outer step
/// via [`GridView::chunk_at_xyz`].
#[allow(clippy::unnecessary_wraps)]
fn cast_ray(
    _origin: [f32; 3],
    _dir: [f32; 3],
    _grid: &GridView<'_>,
    _settings: &OpticastSettings,
) -> Option<Hit> {
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
            if let Some(hit) = cast_ray(origin, dir, &grid, settings) {
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

    /// DDA.0 stub finds no geometry → no puts, framebuffer untouched.
    #[test]
    fn render_dda_stub_writes_nothing() {
        let vxl = roxlap_formats::vxl::Vxl::empty(64);
        let grid = GridView::from_single_vxl(&vxl);
        let settings = OpticastSettings::for_oracle_framebuffer(64, 48);
        let mut rec = Recorder::default();
        render_dda(&oracle_camera(), &settings, grid, 64, &mut rec);
        assert!(rec.puts.is_empty(), "DDA.0 stub must not write any pixels");
    }
}
