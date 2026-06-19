//! Billboard impostor cache — S6.2 of `PORTING-SCENE.md` § S6.
//!
//! At `Lod::Far` distance the per-grid rasterizer is too expensive
//! and produces sub-pixel detail no one will ever see. The
//! billboard cache replaces it with N pre-rendered orthographic-ish
//! snapshots from canonical viewpoints arranged on a sphere; the
//! S6.3 blit path picks the snapshot whose direction most closely
//! matches the current camera and stamps it into the framebuffer
//! as a screen-aligned quad with per-pixel depth.
//!
//! S6.2 lands the cache infrastructure ONLY — types + viewpoint
//! generation + lazy population API + edit-time invalidation. The
//! blit path (consuming the snapshots) is S6.3.
//!
//! ## Viewpoint set
//!
//! [`canonical_viewpoints`] returns 26 unit vectors covering the
//! sphere octants:
//!
//! - **6 face viewpoints**: `±x`, `±y`, `±z` — axis-aligned.
//! - **12 edge viewpoints**: e.g. `(1, 1, 0) / √2` — between two
//!   adjacent faces.
//! - **8 corner viewpoints**: `(1, 1, 1) / √3` — diagonal.
//!
//! The set is hand-tuned to PORTING-SCENE.md § S6's "26 is a
//! reasonable starting point" recommendation; can grow to a
//! denser Fibonacci sphere later if angular gaps prove visible.
//!
//! ## Snapshot camera
//!
//! For each unit viewpoint `v`, the snapshot camera lives in
//! **grid-local space** at:
//! ```text
//! pos     = grid_center_local + v * D
//! forward = -v
//! ```
//! where `D = 8 * bounding_radius` (well past the Far threshold).
//! The basis right / down is constructed via Gram-Schmidt from an
//! arbitrary "up reference" — `(0, 0, 1)` unless the viewpoint is
//! near-parallel, in which case `(1, 0, 0)`. Final basis satisfies
//! voxlap's right-handed convention `right × down == forward`.
//!
//! The projection is perspective with a large focal length so it
//! approximates orthographic: `hz = N * D / (2 * R)`. At `D = 8R`
//! this is `hz = 4N` — a very narrow FOV, rays diverge by at most
//! `atan(1/8) ≈ 7°` across the framebuffer, which is "ortho
//! enough" for impostor purposes.
//!
//! S6.2's `mip_levels = 1` keeps the snapshot rendering simple.
//! Multi-mip and bigger resolutions are a S6.4 polish concern.

use glam::{DVec3, IVec3};
use roxlap_core::opticast::{opticast, OpticastOutcome, OpticastSettings};
use roxlap_core::rasterizer::ScratchPool;
use roxlap_core::scalar_rasterizer::ScalarRasterizer;
use roxlap_core::Camera;

use crate::{Grid, CHUNK_SIZE_XY, CHUNK_SIZE_Z};

/// Per-grid bounding metadata in grid-local space — computed once
/// per render dispatch by [`grid_local_centre_and_radius`].
/// Exposed so S6.3's `render_scene_composed` Far arm can share the
/// centre/radius pair across the cache lookup AND the blit
/// projection without recomputing.
#[derive(Debug, Clone, Copy)]
pub struct GridLocalBounds {
    pub centre: DVec3,
    pub radius: f64,
}

/// See [`super::grid_local_centre_and_radius`] for the internal
/// version. The pub re-export keeps render.rs out of the
/// internal-helper namespace.
#[must_use]
pub fn grid_bounds(grid: &Grid) -> GridLocalBounds {
    let (centre, radius) = grid_local_centre_and_radius(grid);
    GridLocalBounds { centre, radius }
}

/// Distance multiple of `bounding_radius` at which the snapshot
/// camera is placed. 8× is "narrow enough FOV to look orthographic
/// without busting numerical precision". Documented above; not a
/// runtime knob in S6.2 (S6.4 polish can expose it).
const CAMERA_DISTANCE_FACTOR: f64 = 8.0;

/// Default per-snapshot framebuffer resolution. 128 × 128 keeps
/// memory budget per grid at `26 × 128² × (4 colour + 4 depth) =
/// ~3.4 MB` — acceptable for a handful of ships in a scene.
/// Configurable via [`BillboardCache::with_resolution`].
pub const DEFAULT_RESOLUTION: u32 = 128;

/// Sentinel colour stamped into a snapshot's framebuffer wherever
/// opticast would have drawn sky. The S6.3 blit detects sentinel
/// pixels and skips them so the snapshot's sky shows through to
/// whatever else is in the shared `(fb, zb)` (sky panorama, ground
/// from another grid, etc.).
///
/// Picked as `0x00_00_00_00` because:
/// - Alpha byte `0x00` is unused by voxlap's lit-voxel path
///   (`set_side_shades` produces values in `[0x40, 0x80]`), so a
///   real hit never accidentally collides with the sentinel.
/// - All-zero is trivially fast to check and to bulk-fill.
pub const SKY_SENTINEL: u32 = 0x00_00_00_00;

/// One pre-rendered orthographic-ish view of a grid from a fixed
/// direction. The depth buffer carries grid-local Euclidean
/// distance (voxlap's z-buffer convention: smaller = closer); S6.3
/// uses it to z-compose the billboard with other grids' rendered
/// pixels.
#[derive(Debug, Clone)]
pub struct BillboardSnapshot {
    /// Unit vector from the grid centre TO the viewpoint, in
    /// grid-local space. The snapshot camera was at
    /// `centre + view_dir * D` looking back.
    pub view_dir: DVec3,
    /// Snapshot framebuffer width and height in pixels (square in
    /// S6.2 but the struct supports rectangular for future work).
    pub width: u32,
    /// See [`Self::width`].
    pub height: u32,
    /// RGBA framebuffer. Sky pixels carry the build's `sky_color`.
    pub color: Vec<u32>,
    /// Per-pixel grid-local distance (voxlap's z-buffer convention:
    /// smaller = closer). Sky pixels carry [`f32::INFINITY`].
    pub depth: Vec<f32>,
}

/// Per-grid lazy cache of 26 [`BillboardSnapshot`]s indexed by the
/// 26 [`canonical_viewpoints`] directions.
///
/// Construction modes:
/// - [`Self::new_empty`]: allocate an empty cache, populate later
///   via [`Self::build`].
/// - [`Self::build`]: render all 26 snapshots in one call. Use this
///   from the render-time lazy path: when a grid first lands on
///   `Lod::Far`, the S6.3 dispatch checks
///   [`Grid::billboards`]; if `None`, calls `build` and stores.
#[derive(Debug, Clone)]
pub struct BillboardCache {
    /// Snapshot resolution in pixels (square). Pinned at build
    /// time; rebuilds construct a fresh `BillboardCache` rather
    /// than resizing in place.
    pub resolution: u32,
    /// 26 snapshots, indexed in the same order as
    /// [`canonical_viewpoints`]. Empty (`Vec::new`) iff this cache
    /// is uninitialised.
    pub snapshots: Vec<BillboardSnapshot>,
}

impl BillboardCache {
    /// Allocate an empty cache. `snapshots` is empty; future
    /// [`Self::build`] populates it. Cheap — no allocations beyond
    /// the empty `Vec` header.
    #[must_use]
    pub fn new_empty(resolution: u32) -> Self {
        Self {
            resolution,
            snapshots: Vec::new(),
        }
    }

    /// Number of snapshots populated. `0` for an empty cache,
    /// `26` after [`Self::build`].
    #[must_use]
    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    /// `true` iff this cache has not yet been populated (no
    /// snapshots stored).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }

    /// Render all 26 viewpoint snapshots of `grid` into a fresh
    /// cache.
    ///
    /// Cost: `O(26 × resolution² × grid_render_cost)`. For
    /// `resolution = 128` and a small ship grid this is roughly
    /// equivalent to a single full-frame render. Intended to be
    /// called once per grid-edit cycle (caches are invalidated on
    /// edits via [`Grid::set_voxel`] et al.).
    ///
    /// An empty grid (no populated chunks) yields 26 all-sky
    /// snapshots. The cache is still populated so the S6.3 blit
    /// path doesn't keep retrying — a Far-tier empty grid's blit
    /// is then a no-op (every pixel skipped via [`SKY_SENTINEL`]).
    ///
    /// **Sky-pixel detection**: the pool's skycast colour is set
    /// to [`SKY_SENTINEL`] before each snapshot render so opticast
    /// writes the sentinel (not a real sky colour) into pixels
    /// rays missed. Post-render, every sentinel pixel's depth is
    /// reset to [`f32::INFINITY`] (opticast writes a finite "sky
    /// distance" for these pixels, which would otherwise leak
    /// through the blit's depth check).
    #[must_use]
    pub fn build(grid: &Grid, resolution: u32) -> Self {
        let viewpoints = canonical_viewpoints();
        let mut snapshots = Vec::with_capacity(viewpoints.len());

        // Grid centre + bounding radius in grid-local space.
        let (centre, radius) = grid_local_centre_and_radius(grid);
        // Camera distance + ray budget. `R_floor` prevents
        // degenerate budgets for empty grids.
        let r = radius.max(1.0);
        let d = CAMERA_DISTANCE_FACTOR * r;
        // Scan budget covers camera-to-far-side + a little slack
        // for foreshortened rays past the centre.
        let max_scan_dist = ((d + r) * 1.25).ceil().max(64.0) as i32;

        // One ScratchPool shared across all 26 renders. Sized so
        // the per-strip uurend stride fits the snapshot width.
        let pool_vsid = CHUNK_SIZE_XY.max(resolution).max(64);
        let mut pool = ScratchPool::new(resolution, resolution, pool_vsid);
        // Skycast colour = SKY_SENTINEL so opticast stamps the
        // sentinel into sky pixels (instead of a real colour the
        // blit can't reliably tell apart from a voxel hit).
        let sentinel_i = i32::from_ne_bytes(SKY_SENTINEL.to_ne_bytes());
        pool.set_skycast(sentinel_i, 0);
        pool.set_treat_z_max_as_air(true);

        for view_dir in viewpoints {
            let camera = snapshot_camera(view_dir, centre, d);
            let mut color = vec![SKY_SENTINEL; (resolution as usize) * (resolution as usize)];
            let mut depth = vec![f32::INFINITY; color.len()];

            // Empty grid → render all-sky and move on (no chunks
            // means `chunk_xyz_backing` returns None).
            let outcome = if let Some(backing) = grid.chunk_xyz_backing() {
                let cg = roxlap_core::ChunkGrid {
                    chunks: &backing.chunks,
                    origin_chunk_xy: backing.origin_chunk_xy,
                    origin_chunk_z: backing.origin_chunk_z,
                    chunks_x: backing.chunks_x,
                    chunks_y: backing.chunks_y,
                    chunks_z: backing.chunks_z,
                };
                let grid_view = roxlap_core::GridView::from_chunk_grid(&cg, CHUNK_SIZE_XY);
                let settings = snapshot_settings(resolution, d, r, max_scan_dist);
                let mut rasterizer =
                    ScalarRasterizer::new(&mut color, &mut depth, resolution as usize, grid_view);
                opticast(&mut rasterizer, &mut pool, &camera, &settings, grid_view)
            } else {
                // Empty grid — buffers stay at SKY_SENTINEL / INFINITY.
                OpticastOutcome::Rendered
            };
            // `Rendered` and `SkippedCameraInSolid` both keep the
            // buffers — the latter means the camera was inside
            // solid material (impossible for our outside-the-grid
            // camera position), in which case we get sky too.
            let _ = outcome;

            // Sentinel post-process: opticast writes a finite
            // depth (`gxmax` / `max_scan_dist`) for sky pixels via
            // `phase_startsky`. Reset those to INFINITY so the
            // blit's `is_infinite()` belt-and-braces check still
            // works alongside the colour-sentinel check.
            for (px, z) in color.iter().zip(depth.iter_mut()) {
                if *px == SKY_SENTINEL {
                    *z = f32::INFINITY;
                }
            }

            snapshots.push(BillboardSnapshot {
                view_dir,
                width: resolution,
                height: resolution,
                color,
                depth,
            });
        }

        Self {
            resolution,
            snapshots,
        }
    }

    /// Pick the snapshot whose `view_dir` is closest to `query`
    /// (largest dot product). Returns `None` iff the cache is
    /// empty.
    ///
    /// `query` is the unit vector from the grid centre to the
    /// current camera position in grid-local space — the same
    /// frame the snapshot `view_dir`s live in. Caller is
    /// responsible for normalisation.
    ///
    /// Tie-breaking is "first in viewpoint order"; with the 26
    /// canonical viewpoints, ties only happen for query directions
    /// exactly equidistant between two viewpoints, which is a
    /// measure-zero set under f64.
    #[must_use]
    pub fn pick_nearest(&self, query: DVec3) -> Option<&BillboardSnapshot> {
        if self.snapshots.is_empty() {
            return None;
        }
        let mut best_idx = 0usize;
        let mut best_dot = self.snapshots[0].view_dir.dot(query);
        for (i, snap) in self.snapshots.iter().enumerate().skip(1) {
            let d = snap.view_dir.dot(query);
            if d > best_dot {
                best_dot = d;
                best_idx = i;
            }
        }
        Some(&self.snapshots[best_idx])
    }
}

/// 26 unit vectors covering the cube's face / edge / corner
/// directions on the unit sphere.
///
/// Order is stable: 6 face → 12 edge → 8 corner. Indices are
/// implementation details — call [`BillboardCache::pick_nearest`]
/// for the query-driven lookup rather than indexing directly.
///
/// All vectors are unit length to within f64 precision (face
/// vectors are exact; edge / corner vectors come from
/// `normalize()` so they carry the platform's sqrt rounding —
/// typically 1 ULP).
#[must_use]
pub fn canonical_viewpoints() -> Vec<DVec3> {
    let mut out = Vec::with_capacity(26);

    // 6 face directions — axis-aligned, exact unit length.
    for &axis in &[
        DVec3::X,
        DVec3::NEG_X,
        DVec3::Y,
        DVec3::NEG_Y,
        DVec3::Z,
        DVec3::NEG_Z,
    ] {
        out.push(axis);
    }

    // 12 edge directions — (±1, ±1, 0), (±1, 0, ±1), (0, ±1, ±1)
    // normalised to unit length (1/√2 in each non-zero component).
    let signs = [-1.0_f64, 1.0_f64];
    for &sa in &signs {
        for &sb in &signs {
            out.push(DVec3::new(sa, sb, 0.0).normalize());
            out.push(DVec3::new(sa, 0.0, sb).normalize());
            out.push(DVec3::new(0.0, sa, sb).normalize());
        }
    }

    // 8 corner directions — (±1, ±1, ±1) normalised (1/√3 each).
    for &sx in &signs {
        for &sy in &signs {
            for &sz in &signs {
                out.push(DVec3::new(sx, sy, sz).normalize());
            }
        }
    }

    debug_assert_eq!(out.len(), 26);
    out
}

/// Grid centre + bounding-sphere radius, in grid-local voxel
/// coordinates. The centre is the AABB midpoint of the populated
/// chunks (NOT the bounding-sphere centre, which would require a
/// Welzl pass and isn't worth it for 26 fixed viewpoints).
///
/// Empty grid → centre is `(0, 0, 0)` and radius `0.0`. The
/// snapshot camera still renders to all-sky in this case (no
/// chunks → opticast skips the dispatch).
fn grid_local_centre_and_radius(grid: &Grid) -> (DVec3, f64) {
    if grid.chunks.is_empty() {
        return (DVec3::ZERO, 0.0);
    }
    let mut lo = IVec3::splat(i32::MAX);
    let mut hi = IVec3::splat(i32::MIN);
    for &idx in grid.chunks.keys() {
        lo = lo.min(idx);
        hi = hi.max(idx);
    }
    let sx = f64::from(CHUNK_SIZE_XY);
    let sz = f64::from(CHUNK_SIZE_Z);
    let lo_v = DVec3::new(
        f64::from(lo.x) * sx,
        f64::from(lo.y) * sx,
        f64::from(lo.z) * sz,
    );
    let hi_v = DVec3::new(
        f64::from(hi.x + 1) * sx,
        f64::from(hi.y + 1) * sx,
        f64::from(hi.z + 1) * sz,
    );
    let centre = (lo_v + hi_v) * 0.5;
    let half_extent = (hi_v - lo_v) * 0.5;
    let radius = half_extent.length();
    (centre, radius)
}

/// Build the snapshot camera for one viewpoint.
///
/// Positions the camera at `centre + view_dir * d` in grid-local
/// space, looking back at the grid centre with a right-handed
/// basis (`right × down == forward` per voxlap convention).
fn snapshot_camera(view_dir: DVec3, centre: DVec3, d: f64) -> Camera {
    let pos = centre + view_dir * d;
    let forward = -view_dir;
    // Pick an "up reference" not parallel to forward.
    //
    // Voxlap is z-down (positive Z = downward in world space), so
    // the world's "up" direction is **negative** Z. Using
    // `DVec3::NEG_Z` here gives `down = forward × right` that
    // points toward voxlap's +Z (= screen down) — i.e. the
    // snapshot is rendered right-side-up.
    //
    // Pre-2026-05-28 bug: this was `DVec3::Z`, producing
    // `down = (0,0,-1)` for face viewpoints. The rasterizer
    // rendered each snapshot upside-down; the blit then stamped
    // the inverted image at the projected grid centre, making
    // pillars appear at the wrong vertical position (visually:
    // "billboards are taller than the voxel model").
    //
    // Falls back to +X when the viewpoint is near-±Z (forward
    // nearly parallel to the up reference).
    let up_ref = if forward.z.abs() < 0.99 {
        DVec3::NEG_Z
    } else {
        DVec3::X
    };
    let right = forward.cross(up_ref).normalize();
    // down = forward × right gives right × down == forward (voxlap
    // RH convention). Verified by cross-product handedness.
    let down = forward.cross(right);
    Camera {
        pos: pos.to_array(),
        right: right.to_array(),
        down: down.to_array(),
        forward: forward.to_array(),
    }
}

/// Build the [`OpticastSettings`] for one snapshot render.
///
/// `mip_levels = 1` keeps mip-0 only — the snapshot resolution is
/// already coarse, deep mips would over-blur. `mip_scan_dist` is
/// the floor (4). `max_scan_dist` is sized to cover camera-to-
/// far-side of the grid plus a 25 % slack for foreshortened rays
/// past the centre.
///
/// Projection: orthographic-ish perspective with `hz = N * D /
/// (2 * R)`. The image scale at the grid centre lands the
/// bounding sphere exactly at the framebuffer edges; rays
/// diverge by `atan(R/D)` across the framebuffer (~7° at the
/// default `D = 8R`).
fn snapshot_settings(resolution: u32, d: f64, r: f64, max_scan_dist: i32) -> OpticastSettings {
    let n = f64::from(resolution);
    let half_n = (n * 0.5) as f32;
    // hz = (N * D) / (2 * R). For D = 8R this is 4N — narrow FOV,
    // near-orthographic. f64 → f32 cast: f32 mantissa handles
    // values up to ~16M, comfortably above realistic 4N for the
    // 128×128 default.
    #[allow(clippy::cast_possible_truncation)]
    let hz = ((n * d) / (2.0 * r)) as f32;
    OpticastSettings {
        xres: resolution,
        yres: resolution,
        y_start: 0,
        y_end: resolution,
        hx: half_n,
        hy: half_n,
        hz,
        anginc: 1.0,
        mip_levels: 1,
        mip_scan_dist: 4,
        max_scan_dist,
    }
}

/// Blit one [`BillboardSnapshot`] into a `(fb, zb)` pair as a
/// camera-aligned 2D quad — the S6.3 Far-tier render path.
///
/// Projection:
/// - The grid's world centre projects to a screen pixel via the
///   runtime camera's basis + `settings.{hx, hy, hz}`. If the
///   centre is at or behind the camera (`depth <= 0`), this is a
///   no-op.
/// - The grid's bounding sphere (`radius` in grid-local voxel
///   units, identical to world units because rotations preserve
///   distance) projects to a screen-pixel half-extent
///   `pixel_radius = radius * hz / depth`. The blit covers the
///   square `[cx - pixel_radius, cx + pixel_radius]² ×
///   [cy - pixel_radius, cy + pixel_radius]` around the projected
///   centre.
/// - Source-to-destination sampling is nearest-neighbour. The
///   snapshot's resolution is fixed at build time; under-sampling
///   when far away (snapshot_pixels >> screen_pixels) is fine for
///   impostors. Over-sampling when close (screen_pixels >>
///   snapshot_pixels) causes blocky scaling but Far tier is by
///   definition past the radius-based threshold so screen_pixels
///   should stay modest.
///
/// Depth: every non-sky destination pixel gets the same `z` value
/// — the camera-to-grid-centre distance. This treats the impostor
/// as a flat disk at the grid centre. Adequate for S6.3 minimum-
/// viable; S6.4 polish can use per-pixel depth from
/// `snapshot.depth` to recover internal shape.
///
/// Sky pixels: detected via `snapshot.depth[..].is_infinite()`.
/// Skipped entirely (the destination pixel + zbuffer are
/// untouched), so the underlying sky / other grids show through.
///
/// Compose: min-z merge against the existing `zb`. Closer-than-
/// existing wins. Sky pixels in the snapshot don't even attempt
/// the merge, so distant grids never wipe out a near grid's
/// rendered pixel.
#[allow(clippy::too_many_arguments)]
pub fn billboard_blit_into(
    fb: &mut [u32],
    zb: &mut [f32],
    pitch_pixels: usize,
    width: u32,
    height: u32,
    snapshot: &BillboardSnapshot,
    grid_world_centre: DVec3,
    grid_world_radius: f64,
    camera: &Camera,
    settings: &OpticastSettings,
) {
    // Camera basis in world space.
    let cam_pos = DVec3::from_array(camera.pos);
    let forward = DVec3::from_array(camera.forward);
    let right = DVec3::from_array(camera.right);
    let down = DVec3::from_array(camera.down);
    // Vector from camera to grid centre.
    let to_centre = grid_world_centre - cam_pos;
    // Depth along the camera forward axis. Negative = behind
    // camera; skip (the perspective project would invert sign).
    let depth = to_centre.dot(forward);
    if depth <= 0.0 || !depth.is_finite() {
        return;
    }
    // Off-axis offsets along right / down.
    let x_off = to_centre.dot(right);
    let y_off = to_centre.dot(down);
    // Perspective scale factor at the grid centre's depth.
    let scale = f64::from(settings.hz) / depth;
    let cx = f64::from(settings.hx) + x_off * scale;
    let cy = f64::from(settings.hy) + y_off * scale;
    // Half-extent of the impostor quad in destination pixels.
    let pixel_radius_f = grid_world_radius * scale;
    if !pixel_radius_f.is_finite() || pixel_radius_f < 1.0 {
        // Sub-pixel impostor — invisible at this resolution.
        return;
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let pixel_radius = pixel_radius_f.ceil() as i32;
    let dst_size = pixel_radius * 2;
    if dst_size <= 0 {
        return;
    }
    // Source dims.
    let src_w = snapshot.width as i32;
    let src_h = snapshot.height as i32;
    if src_w <= 0 || src_h <= 0 {
        return;
    }
    // Quad's top-left destination corner. Clamped to screen at
    // sampling time so partially-off-screen quads still render
    // their visible portion.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let dst_left = (cx - pixel_radius_f) as i32;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let dst_top = (cy - pixel_radius_f) as i32;
    // Z value used for every non-sky billboard pixel.
    #[allow(clippy::cast_possible_truncation)]
    let z = depth as f32;

    let w_i = width as i32;
    let h_i = height as i32;
    for dy in 0..dst_size {
        let screen_y = dst_top + dy;
        if screen_y < 0 || screen_y >= h_i {
            continue;
        }
        // Nearest-neighbour source y.
        let sy = (dy * src_h) / dst_size;
        let row_src_base = (sy as usize) * (src_w as usize);
        let row_dst_base = (screen_y as usize) * pitch_pixels;
        for dx in 0..dst_size {
            let screen_x = dst_left + dx;
            if screen_x < 0 || screen_x >= w_i {
                continue;
            }
            let sx = (dx * src_w) / dst_size;
            let src_idx = row_src_base + sx as usize;
            // Sky pixels: skip. Two redundant signals — colour
            // matches [`SKY_SENTINEL`] (opticast's skycast write)
            // OR depth is infinite (post-build patch + empty-grid
            // init). Either alone is sufficient; both are checked
            // so a future refactor of build's init can drop one
            // without breaking the blit.
            if snapshot.color[src_idx] == SKY_SENTINEL || snapshot.depth[src_idx].is_infinite() {
                continue;
            }
            let dst_idx = row_dst_base + screen_x as usize;
            if z < zb[dst_idx] {
                fb[dst_idx] = snapshot.color[src_idx];
                zb[dst_idx] = z;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GridTransform;

    // Sky pixels in built snapshots are tagged with SKY_SENTINEL
    // (not a caller-supplied colour). The test asserts use this
    // directly rather than re-exposing a separate constant.

    #[test]
    fn canonical_viewpoints_has_26() {
        let v = canonical_viewpoints();
        assert_eq!(v.len(), 26);
    }

    #[test]
    fn canonical_viewpoints_all_unit_length() {
        for (i, d) in canonical_viewpoints().iter().enumerate() {
            let len = d.length();
            assert!(
                (len - 1.0).abs() < 1e-12,
                "viewpoint {i}: {d:?} length={len}",
            );
        }
    }

    #[test]
    fn canonical_viewpoints_all_distinct() {
        let v = canonical_viewpoints();
        for i in 0..v.len() {
            for j in (i + 1)..v.len() {
                let same = (v[i] - v[j]).length() < 1e-9;
                assert!(!same, "viewpoint {i} and {j} are equal: {:?}", v[i]);
            }
        }
    }

    #[test]
    fn canonical_viewpoints_cover_all_octants() {
        // Among the 8 corner viewpoints, all eight (±1, ±1, ±1) sign
        // combos should appear. Octant signature = (sign_x, sign_y, sign_z).
        let mut octants_seen = std::collections::HashSet::new();
        for v in canonical_viewpoints() {
            let sig = (
                v.x.partial_cmp(&0.0).unwrap(),
                v.y.partial_cmp(&0.0).unwrap(),
                v.z.partial_cmp(&0.0).unwrap(),
            );
            // Only collect strictly-positive-or-strictly-negative axes
            // (no zeros) — those identify the 8 corner octants.
            use std::cmp::Ordering::*;
            if !matches!(sig.0, Equal) && !matches!(sig.1, Equal) && !matches!(sig.2, Equal) {
                octants_seen.insert(sig);
            }
        }
        assert_eq!(octants_seen.len(), 8);
    }

    fn build_small_grid() -> Grid {
        // Single-chunk grid with a recognisable shape — 16-voxel
        // box at chunk-local (50, 50, 50)..(65, 65, 65). Enough
        // content that every viewpoint sees non-sky pixels.
        let mut g = Grid::new(GridTransform::identity());
        g.set_rect(
            IVec3::new(40, 40, 40),
            IVec3::new(80, 80, 80),
            Some(0x80_22_aa_22),
        );
        g
    }

    #[test]
    fn build_populates_26_snapshots() {
        let grid = build_small_grid();
        let cache = BillboardCache::build(&grid, 32);
        assert_eq!(cache.resolution, 32);
        assert_eq!(cache.len(), 26);
        for (i, snap) in cache.snapshots.iter().enumerate() {
            assert_eq!(snap.width, 32);
            assert_eq!(snap.height, 32);
            assert_eq!(snap.color.len(), 32 * 32);
            assert_eq!(snap.depth.len(), 32 * 32);
            // Each snapshot's view_dir must match the canonical
            // viewpoint at the same index.
            let expected = canonical_viewpoints()[i];
            assert!(
                (snap.view_dir - expected).length() < 1e-12,
                "snapshot {i} view_dir mismatch",
            );
        }
    }

    #[test]
    fn build_renders_some_non_sky_pixels_per_viewpoint() {
        // Every viewpoint should hit the box. We don't pin pixel
        // counts (mip-0 + 32×32 res + 8R distance produces 5-50
        // hit pixels depending on viewpoint), just that every
        // viewpoint produces at least ONE non-sky pixel — i.e.
        // the snapshot camera correctly framed the grid.
        let grid = build_small_grid();
        let cache = BillboardCache::build(&grid, 32);
        for (i, snap) in cache.snapshots.iter().enumerate() {
            let non_sky = snap.color.iter().filter(|&&p| p != SKY_SENTINEL).count();
            assert!(
                non_sky > 0,
                "snapshot {i} (view_dir={:?}) rendered all-sky",
                snap.view_dir,
            );
        }
    }

    #[test]
    fn build_empty_grid_yields_26_all_sky_snapshots() {
        let grid = Grid::new(GridTransform::identity());
        let cache = BillboardCache::build(&grid, 16);
        assert_eq!(cache.len(), 26);
        for (i, snap) in cache.snapshots.iter().enumerate() {
            for &px in &snap.color {
                assert_eq!(
                    px, SKY_SENTINEL,
                    "empty grid snapshot {i} produced non-sky pixel {px:#010x}",
                );
            }
            for &z in &snap.depth {
                assert!(z.is_infinite(), "empty grid snapshot {i} depth not INF");
            }
        }
    }

    #[test]
    fn pick_nearest_returns_face_viewpoint_for_axis_query() {
        let grid = build_small_grid();
        let cache = BillboardCache::build(&grid, 16);
        // Query along +x: nearest viewpoint should be +x.
        let snap = cache.pick_nearest(DVec3::X).expect("non-empty cache");
        assert!(
            (snap.view_dir - DVec3::X).length() < 1e-12,
            "+x query picked {:?}",
            snap.view_dir,
        );
        // Query along -z: nearest viewpoint should be -z.
        let snap = cache.pick_nearest(DVec3::NEG_Z).expect("non-empty cache");
        assert!(
            (snap.view_dir - DVec3::NEG_Z).length() < 1e-12,
            "-z query picked {:?}",
            snap.view_dir,
        );
    }

    #[test]
    fn pick_nearest_routes_oblique_to_a_corner_viewpoint() {
        // Query along (1, 1, 1) / √3 lands exactly on the (+, +, +)
        // corner viewpoint; pick_nearest must return that.
        let grid = build_small_grid();
        let cache = BillboardCache::build(&grid, 16);
        let query = DVec3::new(1.0, 1.0, 1.0).normalize();
        let snap = cache.pick_nearest(query).expect("non-empty cache");
        assert!(
            (snap.view_dir - query).length() < 1e-9,
            "diagonal query picked {:?}",
            snap.view_dir,
        );
    }

    #[test]
    fn pick_nearest_returns_none_for_empty_cache() {
        let cache = BillboardCache::new_empty(32);
        assert!(cache.is_empty());
        assert!(cache.pick_nearest(DVec3::X).is_none());
    }

    #[test]
    fn new_empty_allocates_no_snapshots() {
        let cache = BillboardCache::new_empty(64);
        assert_eq!(cache.resolution, 64);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }
}
