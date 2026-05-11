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
pub mod combined;
pub mod edit;
pub mod render;
pub mod snapshot;

use std::collections::HashMap;

use glam::{DQuat, DVec3, IVec3, UVec3};
use roxlap_formats::vxl::Vxl;
use serde::{Deserialize, Serialize};

pub use addr::{grid_local_to_world, voxel_global, voxel_split, world_to_grid_local, GridLocalPos};
pub use combined::CombinedGridView;

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
    /// Cached "combined" virtual-world view (S4.0 Approach C).
    /// Built lazily on the first [`Grid::combined_world`] call after
    /// any chunk edit; the edit API and [`Grid::ensure_chunk`]
    /// invalidate. External mutators of [`Grid::chunks`] must call
    /// [`Grid::invalidate_combined`] explicitly when their pass ends.
    pub(crate) cached_combined: Option<CombinedGridView>,
}

impl Grid {
    /// New empty grid at the given transform — no chunks populated.
    #[must_use]
    pub fn new(transform: GridTransform) -> Self {
        Self {
            transform,
            chunks: HashMap::new(),
            cached_combined: None,
        }
    }

    /// Get-or-build the cached [`CombinedGridView`]. Rebuild cost
    /// is `O(virtual_vsid² + total slab bytes)`; subsequent calls
    /// are `O(1)` until an edit invalidates.
    ///
    /// # Panics
    ///
    /// Cannot panic in practice — the cache is always populated by
    /// the time the unwrap runs. Defensive `expect` rather than a
    /// silent `unwrap_or_default`.
    pub fn combined_world(&mut self) -> &CombinedGridView {
        if self.cached_combined.is_none() {
            self.cached_combined = Some(CombinedGridView::build(&self.chunks));
        }
        // Cache populated above; unwrap is infallible here.
        self.cached_combined
            .as_ref()
            .expect("cached_combined populated")
    }

    /// Mutable counterpart to [`Grid::combined_world`]. Intended for
    /// **alpha-byte-only mutations** that preserve every column's
    /// slab byte LENGTH (e.g. lightmode bakes that only rewrite the
    /// brightness channel of existing colour records). After the
    /// mutation pass, callers should invoke
    /// [`Grid::sync_combined_to_chunks`] to propagate the changes
    /// back into [`Grid::chunks`] so a later cache invalidation
    /// doesn't drop them.
    ///
    /// Mutations that change slab structure (insert/remove slabs,
    /// change z bounds) must NOT go through this path —
    /// [`Grid::sync_combined_to_chunks`] assumes per-column byte
    /// lengths match the source chunks (debug-asserted).
    ///
    /// # Panics
    ///
    /// Cannot panic in practice — same invariant as
    /// [`Grid::combined_world`].
    pub fn combined_world_mut(&mut self) -> &mut CombinedGridView {
        if self.cached_combined.is_none() {
            self.cached_combined = Some(CombinedGridView::build(&self.chunks));
        }
        self.cached_combined
            .as_mut()
            .expect("cached_combined populated")
    }

    /// Copy per-column slab bytes from the cached combined view
    /// back into [`Grid::chunks`]. Each column's byte range
    /// (per-column `slng` length) must equal between combined view
    /// and source chunk — only alpha-byte-only mutations on the
    /// combined view (e.g. lightmode bakes) meet that invariant.
    ///
    /// No-op if the combined view hasn't been built yet.
    ///
    /// # Panics
    ///
    /// Debug builds panic if a column's combined-view byte length
    /// doesn't match its source-chunk byte length — that's the
    /// invariant violation noted above.
    pub fn sync_combined_to_chunks(&mut self) {
        let Some(combined) = self.cached_combined.as_ref() else {
            return;
        };
        let cs_xy = CHUNK_SIZE_XY;
        let vsid = combined.vsid;
        for (chunk_idx, vxl) in &mut self.chunks {
            if chunk_idx.z != 0 {
                // S4.0 scope: only chx/chy chunks at chz=0 are
                // represented in the combined view. Skip others.
                continue;
            }
            #[allow(clippy::cast_sign_loss)]
            let chx = chunk_idx.x as u32;
            #[allow(clippy::cast_sign_loss)]
            let chy = chunk_idx.y as u32;
            let chunk_origin_x = chx * cs_xy;
            let chunk_origin_y = chy * cs_xy;
            for ly in 0..cs_xy {
                for lx in 0..cs_xy {
                    let local_idx = (ly * cs_xy + lx) as usize;
                    let vx = chunk_origin_x + lx;
                    let vy = chunk_origin_y + ly;
                    let v_idx = (vy * vsid + vx) as usize;

                    // Combined view is built contiguously (no
                    // voxalloc-scatter), so column N's bytes occupy
                    // exactly `[column_offset[N], column_offset[N+1])`.
                    // No `slng()` walk needed.
                    let combined_start = combined.column_offset[v_idx] as usize;
                    let combined_end = combined.column_offset[v_idx + 1] as usize;
                    let combined_len = combined_end - combined_start;

                    let chunk_start = vxl.column_offset[local_idx] as usize;
                    debug_assert_eq!(
                        roxlap_formats::vxl::slng(&vxl.data[chunk_start..]),
                        combined_len,
                        "combined-view column ({vx}, {vy}) length {combined_len} != chunk ({chx}, {chy}) local ({lx}, {ly}) slng length — sync_combined_to_chunks requires byte-length-preserving mutations"
                    );
                    vxl.data[chunk_start..chunk_start + combined_len]
                        .copy_from_slice(&combined.data[combined_start..combined_end]);
                }
            }
        }
    }

    /// Mark the cached combined view stale so the next
    /// [`Grid::combined_world`] call rebuilds it. Called automatically
    /// by the edit API and [`Grid::ensure_chunk`]; external code that
    /// mutates [`Grid::chunks`] (e.g. calling
    /// [`roxlap_formats::edit`] primitives directly on a borrowed
    /// `&mut Vxl`, or running a per-chunk lighting bake) must call
    /// this once their mutation pass finishes.
    pub fn invalidate_combined(&mut self) {
        self.cached_combined = None;
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

    /// Mutable iterator over all `(id, grid)` pairs. Yield order is
    /// not guaranteed (HashMap-backed). Used by S4.0+
    /// [`render::render_scene_composed`] so the per-grid combined
    /// cache can be populated lazily during the render pass.
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

    /// `sync_combined_to_chunks` propagates an alpha-byte-only
    /// edit on the combined view back into source chunks. Stand-in
    /// for the lightmode bake use case at much smaller scale.
    #[test]
    fn sync_combined_to_chunks_propagates_alpha_edit() {
        use roxlap_formats::vxl::slng;
        let mut grid = Grid::new(GridTransform::identity());
        grid.set_voxel(IVec3::new(5, 6, 100), Some(0x80_aa_bb_cc));
        // Materialise + tweak the alpha byte of the surface voxel
        // through the combined view.
        {
            let combined = grid.combined_world_mut();
            let v_idx = (6_u32 * combined.vsid + 5) as usize;
            let start = combined.column_offset[v_idx] as usize;
            let len = slng(&combined.data[start..]);
            // Slab layout: [nextptr, z1, z1c, z0, b, g, r, alpha].
            // The alpha byte is at the END of the colour record.
            // For a single solid voxel column the record is the
            // last 4 bytes.
            let alpha_idx = start + len - 1;
            combined.data[alpha_idx] = 0xff;
        }
        grid.sync_combined_to_chunks();
        // Source chunk now carries the modified alpha byte.
        let chunk = grid.chunk(IVec3::ZERO).unwrap();
        let local_idx = (6 * CHUNK_SIZE_XY + 5) as usize;
        let chunk_start = chunk.column_offset[local_idx] as usize;
        let chunk_len = slng(&chunk.data[chunk_start..]);
        let chunk_alpha_idx = chunk_start + chunk_len - 1;
        assert_eq!(chunk.data[chunk_alpha_idx], 0xff);
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
}
