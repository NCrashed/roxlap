//! roxlap scene-graph layer — many independent chunked voxel
//! grids in a single 3D scene.
//!
//! See `PORTING-SCENE.md` at the workspace root for the substage
//! roadmap. This crate is the layer **above** voxlap's per-chunk
//! renderer (`roxlap-core`): a [`Scene`] holds a sparse set of
//! [`Grid`]s, each with its own f64 world position + arbitrary 3D
//! rotation. Future stages will add per-grid raycast composition
//! (S3), cross-chunk gline within a grid (S4), per-grid rotation
//! (S5), far-LOD billboards / planet proxies (S6), and streaming +
//! procedural generation (S7).
//!
//! S2.0 lands the **type skeleton + grid registration only**.
//! S2.1 adds the [`addr`] module — world ↔ grid-local ↔ chunk +
//! voxel-in-chunk decomposition, the canonical f64↔i32 boundary
//! helper called out by risk R5 in `PORTING-SCENE.md`. S2.2 adds
//! the [`chunks`] module (sparse storage with on-demand chunk
//! allocation) and the [`Grid`] edit API ([`Grid::set_voxel`],
//! [`Grid::set_rect`], [`Grid::set_sphere`]) which decompose
//! multi-chunk operations and delegate to
//! [`roxlap_formats::edit`]. S2.3 adds the [`snapshot`] module —
//! a serde-friendly view of the scene that round-trips through
//! `Serialize` + `Deserialize` (chunks encode via
//! [`roxlap_formats::vxl::serialize`] / [`parse`]). Rendering
//! composition is still owed (S3+).
//!
//! [`parse`]: roxlap_formats::vxl::parse

pub mod addr;
pub mod chunks;
pub mod edit;
pub mod lod;
pub mod render;
pub mod snapshot;

use std::collections::HashMap;

use glam::{DQuat, DVec3, IVec3, UVec3};
use roxlap_formats::vxl::Vxl;
use serde::{Deserialize, Serialize};

pub use addr::{grid_local_to_world, voxel_global, voxel_split, world_to_grid_local, GridLocalPos};
pub use lod::{select_lod, Lod, LodThresholds};

/// XY size of one chunk in voxels. The plan locks 128 — keeps
/// chunks compact (~2 MB worst-case dense-slab footprint inside
/// each `Vxl`) and divides cleanly into voxlap's 2048 reference
/// world size.
pub const CHUNK_SIZE_XY: u32 = 128;

/// Z size of one chunk in voxels. Locked at 256 to preserve
/// voxlap's existing slab byte format unchanged inside each chunk
/// — the per-chunk renderer doesn't need to know it's living
/// inside a scene-graph.
pub const CHUNK_SIZE_Z: u32 = 256;

/// Stable identifier for a grid registered in a [`Scene`]. Issued
/// by [`Scene::add_grid`]; persists across edits but a removed
/// grid's id is not reissued.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct GridId(u32);

impl GridId {
    /// The integer wire form. Useful for serde / debug output.
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

/// f64 world placement of one grid: position + orientation.
///
/// `origin` is the grid's local-space origin in world coords —
/// chunk `(0, 0, 0)`'s `(0, 0, 0)` voxel maps to
/// `origin + rotation * vec3(0, 0, 0)` (i.e. just `origin`).
/// Voxel size is fixed at 1 world unit / voxel for v1.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GridTransform {
    pub origin: DVec3,
    pub rotation: DQuat,
}

impl GridTransform {
    /// Identity transform at world origin. Useful as a default for
    /// the first grid added to an otherwise empty scene.
    #[must_use]
    pub fn identity() -> Self {
        Self {
            origin: DVec3::ZERO,
            rotation: DQuat::IDENTITY,
        }
    }

    /// Axis-aligned grid placed at `origin` with no rotation.
    #[must_use]
    pub fn at(origin: DVec3) -> Self {
        Self {
            origin,
            rotation: DQuat::IDENTITY,
        }
    }
}

impl Default for GridTransform {
    fn default() -> Self {
        Self::identity()
    }
}

/// Address of one voxel inside a scene: which grid it belongs to,
/// which chunk within that grid, and the voxel's offset inside
/// that chunk.
///
/// `chunk` is signed (`IVec3`) because chunks are centred on the
/// grid's local origin and may extend in either direction. `voxel`
/// is unsigned and must satisfy
/// `(voxel.x, voxel.y) < CHUNK_SIZE_XY` and `voxel.z < CHUNK_SIZE_Z`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GridAddr {
    pub grid: GridId,
    pub chunk: IVec3,
    pub voxel: UVec3,
}

/// One independent voxel grid in a scene. Holds its world placement
/// and a sparse map of populated chunks. Empty chunk slots are
/// implicit air and skipped during rendering / raycasts.
///
/// Each chunk is internally a [`Vxl`] with `vsid = CHUNK_SIZE_XY`
/// — the existing per-chunk renderer (opticast + grouscan +
/// sprites + lighting in `roxlap-core`) runs on each chunk
/// unchanged. Vertical worlds are built by stacking chunks along
/// grid-local `+z`.
#[derive(Debug)]
pub struct Grid {
    /// World placement (origin + rotation).
    pub transform: GridTransform,
    /// Sparse chunk storage keyed by `(chx, chy, chz)` chunk
    /// coordinates. A missing entry means the chunk is fully air.
    pub chunks: HashMap<IVec3, Vxl>,
    /// Whether sky pixels rendered for this grid should be
    /// composited into the final framebuffer. `true` is the
    /// historical "grid owns its own sky" behaviour: ray misses
    /// inside this grid's frustum paint sky_color into the temp
    /// buffer. Set `false` for grids that are a foreground object
    /// (e.g. a ship) — the sky is owned by a single "world" grid
    /// (the ground) and other grids should not contribute sky
    /// pixels, otherwise their grid-local-frame sky lookup
    /// rotates with the grid and visibly fights the world's sky
    /// during compose. See [`crate::render::render_scene_composed`]
    /// for the masking implementation.
    pub render_sky: bool,
    /// Override [`roxlap_core::opticast::OpticastSettings::mip_levels`]
    /// for this grid. `None` ⇒ use the caller's value. `Some(n)`
    /// ⇒ cap at `n` (clamped to `[1, settings.mip_levels]`). Use
    /// to disable multi-mip on a per-grid basis — small grids
    /// (rotating ships, billboards) don't benefit from deep mips
    /// and CAN trigger the
    /// `[[project_axis_aligned_mip_beams]]`-style cf-cancellation
    /// artifact when near-axis-aligned rays hit the rotated grid.
    /// `Some(1)` = mip-0 only, byte-stable to single-mip.
    pub mip_levels_override: Option<u32>,
    /// World-distance thresholds for per-grid LOD tier selection
    /// (S6.0). Defaults to [`LodThresholds::always_near`], so a
    /// freshly-constructed grid always renders at full voxel (the
    /// S5-and-earlier byte-stable behaviour). S6.1 plugs `Mid` into
    /// the existing multi-mip path; S6.3 plugs `Far` into the
    /// billboard impostor cache. See [`crate::lod`].
    pub lod_thresholds: LodThresholds,
}

impl Grid {
    /// New empty grid at the given transform — no chunks populated,
    /// `render_sky = true`, LOD thresholds default to
    /// [`LodThresholds::always_near`].
    #[must_use]
    pub fn new(transform: GridTransform) -> Self {
        Self {
            transform,
            chunks: HashMap::new(),
            render_sky: true,
            mip_levels_override: None,
            lod_thresholds: LodThresholds::always_near(),
        }
    }

    /// Bounding-sphere radius of the populated chunk set in
    /// grid-local space.
    ///
    /// Walks the sparse chunk map once, computes the chunk-index
    /// AABB, converts to voxel-space half-extent, returns its
    /// Euclidean length. Empty grid → `0.0`.
    ///
    /// Conservative — bounds the full chunk volume, not just its
    /// populated voxels (a chunk containing one voxel still
    /// contributes `CHUNK_SIZE_XY × CHUNK_SIZE_XY × CHUNK_SIZE_Z`
    /// to the bbox). For LOD picking that's fine: an over-bound
    /// sphere errs on the side of `Near`.
    ///
    /// Cost: `O(chunks.len())`; recomputed on every call. Callers
    /// who need this every frame should memoize at the
    /// [`Scene`]-level cache (added when S6.2 needs it).
    #[must_use]
    pub fn bounding_radius(&self) -> f64 {
        if self.chunks.is_empty() {
            return 0.0;
        }
        let mut min = IVec3::splat(i32::MAX);
        let mut max = IVec3::splat(i32::MIN);
        for &idx in self.chunks.keys() {
            min = min.min(idx);
            max = max.max(idx);
        }
        // Chunk-index bbox → voxel-space half-extent. `+1` on max
        // converts inclusive chunk index to exclusive voxel upper
        // bound (chunk `idx` covers voxels `[idx*size, (idx+1)*size)`).
        let sx = f64::from(CHUNK_SIZE_XY);
        let sz = f64::from(CHUNK_SIZE_Z);
        let lo = DVec3::new(
            f64::from(min.x) * sx,
            f64::from(min.y) * sx,
            f64::from(min.z) * sz,
        );
        let hi = DVec3::new(
            f64::from(max.x + 1) * sx,
            f64::from(max.y + 1) * sx,
            f64::from(max.z + 1) * sz,
        );
        let half_extent = (hi - lo) * 0.5;
        half_extent.length()
    }

    /// Pick this grid's LOD tier for the given world-space camera
    /// position. Convenience wrapper around [`crate::select_lod`]
    /// that pulls [`Self::lod_thresholds`] from the grid.
    #[must_use]
    pub fn select_lod(&self, camera_world_pos: DVec3) -> Lod {
        select_lod(camera_world_pos, &self.transform, self.lod_thresholds)
    }
}

/// Top-level scene container. Holds a flat collection of grids
/// keyed by [`GridId`].
///
/// S2.0 only exposes registration / removal / lookup. Address math
/// helpers (S2.x), edit API (S2.x), and rendering composition (S3)
/// land in later sub-substages.
#[derive(Debug, Default)]
pub struct Scene {
    grids: HashMap<GridId, Grid>,
    next_grid_id: u32,
}

impl Scene {
    /// New empty scene — no grids.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of grids currently registered.
    #[must_use]
    pub fn grid_count(&self) -> usize {
        self.grids.len()
    }

    /// Register a new grid. Returns its fresh, unique [`GridId`].
    pub fn add_grid(&mut self, transform: GridTransform) -> GridId {
        let id = GridId(self.next_grid_id);
        self.next_grid_id += 1;
        self.grids.insert(id, Grid::new(transform));
        id
    }

    /// Remove a grid by id. Returns the removed [`Grid`] (so the
    /// caller can reclaim its chunks) or `None` if the id wasn't
    /// registered. Removed ids are not reissued.
    pub fn remove_grid(&mut self, id: GridId) -> Option<Grid> {
        self.grids.remove(&id)
    }

    /// Borrow a registered grid.
    #[must_use]
    pub fn grid(&self, id: GridId) -> Option<&Grid> {
        self.grids.get(&id)
    }

    /// Mutably borrow a registered grid.
    pub fn grid_mut(&mut self, id: GridId) -> Option<&mut Grid> {
        self.grids.get_mut(&id)
    }

    /// Iterator over all `(id, grid)` pairs in registration order
    /// is **not** guaranteed — the underlying map is a `HashMap`.
    /// Callers that need a stable order must sort by [`GridId`].
    pub fn grids(&self) -> impl Iterator<Item = (GridId, &Grid)> {
        self.grids.iter().map(|(id, g)| (*id, g))
    }

    /// Mutable iterator over all `(id, grid)` pairs. Yield order
    /// is not guaranteed (HashMap-backed).
    pub fn grids_mut(&mut self) -> impl Iterator<Item = (GridId, &mut Grid)> {
        self.grids.iter_mut().map(|(id, g)| (*id, g))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_scene_has_no_grids() {
        let scene = Scene::new();
        assert_eq!(scene.grid_count(), 0);
        assert!(scene.grids().next().is_none());
    }

    #[test]
    fn add_grid_returns_fresh_ids() {
        let mut scene = Scene::new();
        let a = scene.add_grid(GridTransform::identity());
        let b = scene.add_grid(GridTransform::at(DVec3::new(100.0, 0.0, 0.0)));
        assert_ne!(a, b);
        assert_eq!(a.raw(), 0);
        assert_eq!(b.raw(), 1);
        assert_eq!(scene.grid_count(), 2);
    }

    #[test]
    fn grid_lookup_round_trips() {
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::at(DVec3::new(10.0, 20.0, 30.0)));
        let g = scene.grid(id).expect("grid registered");
        assert_eq!(g.transform.origin, DVec3::new(10.0, 20.0, 30.0));
        assert_eq!(g.transform.rotation, DQuat::IDENTITY);
        assert!(g.chunks.is_empty());
    }

    #[test]
    fn remove_grid_drops_it_from_scene() {
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::identity());
        let removed = scene.remove_grid(id);
        assert!(removed.is_some());
        assert_eq!(scene.grid_count(), 0);
        assert!(scene.grid(id).is_none());
        // Re-adding does NOT reuse the dropped id.
        let id2 = scene.add_grid(GridTransform::identity());
        assert_ne!(id, id2);
        assert_eq!(id2.raw(), 1);
    }

    #[test]
    fn remove_unknown_grid_is_none() {
        let mut scene = Scene::new();
        let bogus = GridId(999);
        assert!(scene.remove_grid(bogus).is_none());
    }

    #[test]
    fn grid_mut_can_modify_transform() {
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::identity());
        scene.grid_mut(id).unwrap().transform.origin = DVec3::new(1.0, 2.0, 3.0);
        assert_eq!(
            scene.grid(id).unwrap().transform.origin,
            DVec3::new(1.0, 2.0, 3.0)
        );
    }

    #[test]
    fn chunk_size_constants_match_plan() {
        // Plan locks these values; bumping either breaks the slab
        // byte format (Z) or the worst-case chunk footprint budget
        // (XY). Pin them so a future refactor that drifts them
        // shows up in CI.
        assert_eq!(CHUNK_SIZE_XY, 128);
        assert_eq!(CHUNK_SIZE_Z, 256);
    }

    // ---- S6.0: bounding_radius + Grid::select_lod ----

    #[test]
    fn new_grid_defaults_to_always_near_lod() {
        // Byte-identity contract for the staged S6 rollout: a
        // grid built through `new` must never trigger the Mid/Far
        // branches by accident, even when bounding_radius would
        // imply otherwise.
        let g = Grid::new(GridTransform::identity());
        assert_eq!(g.lod_thresholds.r_near, f64::INFINITY);
        assert_eq!(g.lod_thresholds.r_mid, f64::INFINITY);
        assert_eq!(g.select_lod(DVec3::new(1e9, 0.0, 0.0)), Lod::Near);
    }

    #[test]
    fn bounding_radius_empty_grid_is_zero() {
        let g = Grid::new(GridTransform::identity());
        assert_eq!(g.bounding_radius(), 0.0);
    }

    #[test]
    fn bounding_radius_single_chunk_at_origin() {
        // One chunk at (0, 0, 0): bbox is [0, 128) × [0, 128) × [0, 256).
        // Half-extent = (64, 64, 128); length = sqrt(64² + 64² + 128²)
        //   = sqrt(4096 + 4096 + 16384) = sqrt(24576) ≈ 156.7747...
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::identity());
        let g = scene.grid_mut(id).unwrap();
        // Populate chunk (0, 0, 0) via the edit API.
        g.set_voxel(IVec3::new(0, 0, 0), Some(0x80_88_88_88));
        let r = g.bounding_radius();
        let expected = ((64.0_f64).powi(2) * 2.0 + (128.0_f64).powi(2)).sqrt();
        assert!(
            (r - expected).abs() < 1e-9,
            "bounding_radius={r} expected={expected}"
        );
    }

    #[test]
    fn bounding_radius_grows_with_chunk_extent() {
        // Two chunks at (0,0,0) and (3,0,0): x extent is 4 chunks =
        // 512 voxels; y/z are 1 chunk each. Half-extent = (256, 64, 128);
        // length = sqrt(256² + 64² + 128²) = sqrt(65536+4096+16384)
        //        = sqrt(86016) ≈ 293.2848.
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::identity());
        let g = scene.grid_mut(id).unwrap();
        // Stamp one voxel in chunk (0,0,0).
        g.set_voxel(IVec3::new(0, 0, 0), Some(0x80_88_88_88));
        // Stamp one voxel in chunk (3,0,0): grid-local x = 3*128 = 384.
        g.set_voxel(IVec3::new(384, 0, 0), Some(0x80_88_88_88));
        assert_eq!(g.chunks.len(), 2);
        let r = g.bounding_radius();
        let expected = (256.0_f64.powi(2) + 64.0_f64.powi(2) + 128.0_f64.powi(2)).sqrt();
        assert!(
            (r - expected).abs() < 1e-9,
            "bounding_radius={r} expected={expected}"
        );
    }

    #[test]
    fn grid_select_lod_respects_lod_thresholds_field() {
        // Set a non-default threshold and verify the helper picks
        // the right tier for known distances.
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::at(DVec3::new(100.0, 0.0, 0.0)));
        let g = scene.grid_mut(id).unwrap();
        g.lod_thresholds = LodThresholds {
            r_near: 50.0,
            r_mid: 200.0,
            ..LodThresholds::always_near()
        };
        // Camera 25 units from grid origin → Near.
        assert_eq!(g.select_lod(DVec3::new(125.0, 0.0, 0.0)), Lod::Near);
        // 100 units → Mid.
        assert_eq!(g.select_lod(DVec3::new(200.0, 0.0, 0.0)), Lod::Mid);
        // 500 units → Far.
        assert_eq!(g.select_lod(DVec3::new(600.0, 0.0, 0.0)), Lod::Far);
    }
}
