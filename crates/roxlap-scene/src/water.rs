//! WT.0 — water volumes (`PORTING-WATER.md`): the *physics*
//! representation of water.
//!
//! Why not the CC.4 passable-veto: the veto classifies by voxel
//! colour, and `.vxl` slab interiors are colourless by format
//! (`Cube::UnexposedSolid`) — a filled pool has a core the veto never
//! sees, so colour-keyed water works only ~2 voxels deep. A
//! [`WaterVolume`] is instead declared *beside* the voxels: an
//! axis-aligned grid-local box that answers "is this point in water,
//! and how deep?" independently of what the slabs store. The visuals
//! stay voxels (fill the same region with a
//! [`BlendMode::Volumetric`](roxlap_formats::material::BlendMode)
//! material); the veto stays useful for thin pass-through curtains.
//!
//! Conventions (chapter 2: **+z is DOWN**):
//! - `lo`/`hi` are **inclusive** voxel corners; voxel `(x, y, z)`
//!   occupies the continuous cube `[x, x+1)³`, so the volume spans
//!   `[lo, hi + 1)` in continuous grid-local coordinates.
//! - The **surface is the volume's top face** — the plane
//!   `z = lo.z` (smaller z is higher).
//! - Volumes are grid-local: a scaled grid's world waterline moves
//!   with `voxel_world_size`, and a rotated grid's surface tilts with
//!   it. Water is *usually* authored on identity/yaw-only grids so
//!   the surface stays world-horizontal; the queries are exact under
//!   any transform regardless.
//!
//! Authoring state only, like [`crate::BakeLight`]s: editing the list
//! re-renders nothing by itself (the *visual* water is voxels you
//! edit separately). Persisted in snapshots since wire **v3**.

use glam::{DVec3, IVec3};
use serde::{Deserialize, Serialize};

use crate::{GridId, Scene};

/// An axis-aligned box of water in grid-local voxel coordinates,
/// inclusive on both corners. See the module doc for the coordinate
/// conventions (surface = `lo.z`, z-down).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WaterVolume {
    /// Minimum corner (inclusive). `lo.z` is the **surface** layer.
    pub lo: IVec3,
    /// Maximum corner (inclusive). `hi.z` is the bottom layer.
    pub hi: IVec3,
}

impl WaterVolume {
    /// A volume spanning the two corners in any order (canonicalised
    /// so `lo <= hi` per component — `PartialEq` compares fields, so
    /// prefer this over a raw struct literal).
    #[must_use]
    pub fn new(a: IVec3, b: IVec3) -> Self {
        Self {
            lo: a.min(b),
            hi: a.max(b),
        }
    }

    /// Depth of a continuous grid-local point below this volume's
    /// surface, in **voxel units**: `Some(distance below the min-z
    /// face)` when the point is inside the water, `None` outside.
    /// Zero exactly at the surface plane; positive going down
    /// (z-down).
    ///
    /// Corner order is normalised HERE, not assumed: the fields are
    /// pub (and easy to swap under z-down), so a hand-built
    /// `WaterVolume { lo: big, hi: small }` queries identically to
    /// its [`Self::new`]-canonicalised form — and identically before
    /// and after a snapshot round-trip.
    #[must_use]
    pub fn depth_local(&self, p: DVec3) -> Option<f64> {
        let lo = self.lo.min(self.hi).as_dvec3();
        let hi = self.lo.max(self.hi).as_dvec3() + DVec3::ONE; // inclusive → half-open
        let inside =
            p.x >= lo.x && p.x < hi.x && p.y >= lo.y && p.y < hi.y && p.z >= lo.z && p.z < hi.z;
        inside.then_some(p.z - lo.z)
    }
}

impl crate::Grid {
    /// Register a water volume spanning the two grid-local voxel
    /// corners (any order). Convenience over pushing onto
    /// [`water_volumes`](crate::Grid::water_volumes) directly — this
    /// normalises the corners.
    pub fn add_water_volume(&mut self, a: IVec3, b: IVec3) {
        self.water_volumes.push(WaterVolume::new(a, b));
    }

    /// Depth of a continuous **grid-local** point below the water
    /// surface, in voxel units — the deepest submersion across this
    /// grid's volumes (overlapping volumes: the highest surface
    /// wins). `None` when the point is in no volume.
    #[must_use]
    pub fn water_depth_local(&self, p: DVec3) -> Option<f64> {
        self.water_volumes
            .iter()
            .filter_map(|v| v.depth_local(p))
            .max_by(f64::total_cmp)
    }
}

impl Scene {
    /// Depth of a **world**-space point below a water surface, in
    /// **world units**, with the grid that owns the deepest water at
    /// that point. `None` when no grid's water contains the point.
    ///
    /// The world↔grid boundary rule (SC) applies: the point is
    /// converted into each grid's local voxel frame (un-rotate,
    /// un-translate, `/ voxel_world_size`), the depth is measured
    /// along the grid's **local** z (world-vertical for the usual
    /// identity/yaw-only water grids), and converted back to world
    /// units (`× voxel_world_size`).
    ///
    /// Deterministic: on an exact depth tie between grids (two
    /// identity grids sharing a waterline at their seam), the
    /// smallest [`GridId`] wins — [`Scene::grids`] iterates a
    /// `HashMap`, so without the explicit tie-break the winner would
    /// flip between runs (and WT.2/WT.3 hang per-grid state off this
    /// id).
    #[must_use]
    pub fn water_depth_at(&self, world: DVec3) -> Option<(GridId, f64)> {
        self.query_grids()
            .filter(|(_, grid)| !grid.water_volumes.is_empty())
            .filter_map(|(id, grid)| {
                let local = crate::streaming::world_to_grid_local_pos(world, &grid.transform);
                let d_local = grid.water_depth_local(local)?;
                Some((id, d_local * grid.transform.voxel_world_size))
            })
            .max_by(|a, b| a.1.total_cmp(&b.1).then_with(|| b.0.cmp(&a.0)))
    }

    /// `true` when the world-space point is inside any water volume.
    /// Short-circuits on the first hit — the query the swim state
    /// (WT.1) runs per actor per frame; use
    /// [`Self::water_depth_at`] only when the depth is needed.
    #[must_use]
    pub fn in_water(&self, world: DVec3) -> bool {
        self.query_grids()
            .filter(|(_, grid)| !grid.water_volumes.is_empty())
            .any(|(_, grid)| {
                let local = crate::streaming::world_to_grid_local_pos(world, &grid.transform);
                grid.water_depth_local(local).is_some()
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GridTransform;
    use glam::DQuat;

    // ---- WaterVolume geometry ----

    #[test]
    fn corners_normalise_and_span_is_inclusive() {
        let v = WaterVolume::new(IVec3::new(9, 9, 9), IVec3::new(2, 3, 4));
        assert_eq!(v.lo, IVec3::new(2, 3, 4));
        assert_eq!(v.hi, IVec3::new(9, 9, 9));
        // Inclusive hi: the far corner voxel's interior is inside...
        assert!(v.depth_local(DVec3::new(9.5, 9.5, 9.5)).is_some());
        // ...and the half-open continuous edge just past it is not.
        assert!(v.depth_local(DVec3::new(10.0, 9.5, 9.5)).is_none());
    }

    #[test]
    fn depth_is_zero_at_surface_and_grows_downward() {
        // z-down: the surface is the MIN-z face.
        let v = WaterVolume::new(IVec3::new(0, 0, 10), IVec3::new(7, 7, 19));
        assert_eq!(v.depth_local(DVec3::new(4.0, 4.0, 10.0)), Some(0.0));
        assert_eq!(v.depth_local(DVec3::new(4.0, 4.0, 15.5)), Some(5.5));
        // Above the surface (smaller z) is dry.
        assert!(v.depth_local(DVec3::new(4.0, 4.0, 9.9)).is_none());
        // Below the bottom is outside the volume, not "very deep".
        assert!(v.depth_local(DVec3::new(4.0, 4.0, 20.0)).is_none());
    }

    // ---- Grid / Scene queries ----

    #[test]
    fn grid_takes_deepest_of_overlapping_volumes() {
        let mut g = crate::Grid::new(GridTransform::identity());
        // Two overlapping pools; the one with the HIGHER surface
        // (smaller lo.z) reads deeper at a shared point.
        g.add_water_volume(IVec3::new(0, 0, 10), IVec3::new(9, 9, 19));
        g.add_water_volume(IVec3::new(0, 0, 14), IVec3::new(9, 9, 19));
        assert_eq!(g.water_depth_local(DVec3::new(5.0, 5.0, 16.0)), Some(6.0));
    }

    #[test]
    fn scene_world_query_identity_grid() {
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::at(DVec3::new(100.0, 0.0, 0.0)));
        scene
            .grid_mut(id)
            .unwrap()
            .add_water_volume(IVec3::new(0, 0, 50), IVec3::new(31, 31, 63));
        // World x=110 → local x=10; world z=55 → 5 under the surface.
        assert_eq!(
            scene.water_depth_at(DVec3::new(110.0, 5.0, 55.0)),
            Some((id, 5.0))
        );
        assert!(scene.water_depth_at(DVec3::new(110.0, 5.0, 49.0)).is_none());
        assert!(!scene.in_water(DVec3::new(0.0, 0.0, 55.0))); // outside grid
    }

    #[test]
    fn scene_world_query_scales_with_vws() {
        // SC discipline: a vws=0.5 grid's water is half as deep in
        // world units, and the world waterline sits at origin.z +
        // lo.z * vws.
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::at_scale(DVec3::ZERO, 0.5));
        scene
            .grid_mut(id)
            .unwrap()
            .add_water_volume(IVec3::new(0, 0, 20), IVec3::new(15, 15, 39));
        // World z=10 is the surface (20 voxels × 0.5); 4 world units
        // down = 8 voxels down.
        assert_eq!(
            scene.water_depth_at(DVec3::new(3.0, 3.0, 14.0)),
            Some((id, 4.0))
        );
        assert!(scene.water_depth_at(DVec3::new(3.0, 3.0, 9.9)).is_none());
    }

    #[test]
    fn scene_world_query_rotated_grid() {
        // 90° yaw about z: world (x, y) → local (y, -x); z unchanged,
        // so the surface stays world-horizontal under yaw.
        let mut scene = Scene::new();
        let rot = DQuat::from_rotation_z(std::f64::consts::FRAC_PI_2);
        let id = scene.add_grid(GridTransform {
            origin: DVec3::ZERO,
            rotation: rot,
            ..GridTransform::identity()
        });
        scene
            .grid_mut(id)
            .unwrap()
            .add_water_volume(IVec3::new(0, 0, 8), IVec3::new(9, 19, 15));
        // Local target (5, 10, 12) → world = rot * local.
        let world = rot * DVec3::new(5.0, 10.0, 12.0);
        assert_eq!(scene.water_depth_at(world), Some((id, 4.0)));
        // A point that would be inside pre-rotation but isn't post.
        assert!(scene.water_depth_at(DVec3::new(5.0, 10.0, 12.0)).is_none());
    }

    #[test]
    fn swapped_corners_query_like_canonical_ones() {
        // The fields are pub and easy to swap under z-down; a
        // hand-built inverted volume must behave exactly like its
        // canonical form — live AND across a snapshot round-trip
        // (depth_local normalises, nothing else has to).
        let swapped = WaterVolume {
            lo: IVec3::new(7, 7, 19),
            hi: IVec3::new(0, 0, 10),
        };
        let canonical = WaterVolume::new(IVec3::new(0, 0, 10), IVec3::new(7, 7, 19));
        for p in [
            DVec3::new(4.0, 4.0, 10.0),
            DVec3::new(4.0, 4.0, 15.5),
            DVec3::new(4.0, 4.0, 9.9),
            DVec3::new(8.5, 4.0, 15.0),
        ] {
            assert_eq!(swapped.depth_local(p), canonical.depth_local(p));
        }
    }

    #[test]
    fn equal_depth_tie_breaks_to_smallest_grid_id() {
        // Two identity grids sharing a waterline: the returned owner
        // must be deterministic (grids() iterates a HashMap).
        let mut scene = Scene::new();
        let a = scene.add_grid(GridTransform::identity());
        let b = scene.add_grid(GridTransform::identity());
        for id in [a, b] {
            scene
                .grid_mut(id)
                .unwrap()
                .add_water_volume(IVec3::new(0, 0, 50), IVec3::new(31, 31, 63));
        }
        let winner = scene.water_depth_at(DVec3::new(5.0, 5.0, 55.0));
        assert_eq!(winner, Some((a.min(b), 5.0)), "smallest GridId wins ties");
    }

    #[test]
    fn no_volumes_means_dry_everywhere() {
        let mut scene = Scene::new();
        scene.add_grid(GridTransform::identity());
        assert!(scene.water_depth_at(DVec3::new(1.0, 2.0, 3.0)).is_none());
        assert!(!scene.in_water(DVec3::splat(50.0)));
    }
}
