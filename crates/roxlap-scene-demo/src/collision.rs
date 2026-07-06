//! Camera-vs-voxel collision for the scene demo.
//!
//! S3.x: walks every grid in the scene, transforms the world-space
//! query into each grid's local frame (axis-aligned only — no
//! rotation yet, that's S5), then asks `roxlap-core::world_query::getcube`
//! whether any voxel inside a `±PLAYER_RADIUS` cube around the
//! position is solid. Out-of-grid positions count as air so the
//! camera can fly past a grid's footprint.
//!
//! S5 evolution: replace the per-grid translate with the full
//! `world_to_grid_local` (the helper already exists in
//! `roxlap_scene::addr`); rotation just composes onto the
//! translate. The cube-of-voxels probe stays the same.

use glam::IVec3;
use roxlap_core::world_query::{getcube, Cube};
use roxlap_scene::{voxel_split, Scene, CHUNK_SIZE_Z};

/// Camera "skin" radius in voxel units. A move is blocked when any
/// voxel inside the cube `pos ± PLAYER_RADIUS` is solid; matches
/// the cave-demo / roxlap-host tuning.
pub const PLAYER_RADIUS: f64 = 0.3;

/// `true` if any voxel of any grid in `scene` overlaps the camera's
/// `±PLAYER_RADIUS` cube around `world_pos`.
///
/// Skips the bedrock placeholder at chunk-local `z = CHUNK_SIZE_Z - 1`
/// to match the renderer pool's `treat_z_max_as_air = true` setting:
/// voxlap auto-maintains a solid voxel at z=255 in every column, but
/// the demo renders it as transparent (so the camera can fly under
/// floating chunks without hitting an invisible wall). Without this
/// skip, getcube reports it as `Cube::Color` and collision blocks at
/// world z=155 (= ship grid's z=255) — the user-reported "invisible
/// border" between world z<155 and z>155 inside the ship's xy
/// footprint. For ground-style dense terrain the player is blocked
/// well before reaching z=255 anyway (visible stone fills z=85..254),
/// so skipping z=255 doesn't regress that case.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::similar_names
)]
pub fn is_blocked_in_scene(scene: &Scene, world_pos: [f64; 3]) -> bool {
    for (_id, grid) in scene.grids() {
        // S3.x..S4.x: axis-aligned grids only — translate world → grid-local.
        let lx_centre = world_pos[0] - grid.transform.origin.x;
        let ly_centre = world_pos[1] - grid.transform.origin.y;
        let lz_centre = world_pos[2] - grid.transform.origin.z;
        let lo_x = (lx_centre - PLAYER_RADIUS).floor() as i32;
        let hi_x = (lx_centre + PLAYER_RADIUS).floor() as i32;
        let lo_y = (ly_centre - PLAYER_RADIUS).floor() as i32;
        let hi_y = (ly_centre + PLAYER_RADIUS).floor() as i32;
        let lo_z = (lz_centre - PLAYER_RADIUS).floor() as i32;
        let hi_z = (lz_centre + PLAYER_RADIUS).floor() as i32;

        // S4.0: per-voxel chunk lookup. The probe cube's voxels may
        // straddle chunk boundaries, so we split each grid-local
        // voxel into (chunk_idx, in_chunk_voxel) and route the
        // getcube query to the right chunk's slab data. Implicit-
        // air chunks count as air; bedrock placeholders are skipped
        // per the fn-level docs.
        let cs_z_solid = (CHUNK_SIZE_Z as i32) - 1;
        for vz in lo_z..=hi_z {
            for vy in lo_y..=hi_y {
                for vx in lo_x..=hi_x {
                    let (chunk_idx, in_chunk) = voxel_split(IVec3::new(vx, vy, vz));
                    if (in_chunk.z as i32) >= cs_z_solid {
                        // Bedrock placeholder — treat as air.
                        continue;
                    }
                    let Some(chunk) = grid.chunk(chunk_idx) else {
                        // Implicit-air chunk; nothing to collide with.
                        continue;
                    };
                    // `Cube::Color` = visible surface voxel,
                    // `Cube::UnexposedSolid` = hidden interior of a
                    // solid run; both block. `Cube::Air` does not.
                    if !matches!(
                        getcube(
                            &chunk.data,
                            &chunk.column_offset,
                            chunk.vsid,
                            in_chunk.x as i32,
                            in_chunk.y as i32,
                            in_chunk.z as i32,
                        ),
                        Cube::Air
                    ) {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Apply `step` to `pos` per-axis with collision rejection. Each
/// axis is tested independently so the camera slides along walls
/// instead of jamming when one component would collide. If `pos`
/// already overlaps solid material (e.g. spawned next to a voxel,
/// or lighting bake re-classified a column), all axes move freely
/// — the user can escape rather than getting permanently stuck.
pub fn slide_with_collision(scene: &Scene, pos: &mut [f64; 3], step: [f64; 3]) {
    let already_stuck = is_blocked_in_scene(scene, *pos);
    for axis in 0..3 {
        let mut candidate = *pos;
        candidate[axis] += step[axis];
        if already_stuck || !is_blocked_in_scene(scene, candidate) {
            pos[axis] = candidate[axis];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;
    use roxlap_scene::VoxColor;
    use roxlap_scene::{GridTransform, Scene};

    /// Single floating voxel at chunk-local (10, 10, 50) in a grid
    /// at world (0, 0, -100). Used to probe the area around an
    /// isolated voxel.
    fn floating_chunk_scene() -> Scene {
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::at(DVec3::new(0.0, 0.0, -100.0)));
        let grid = scene.grid_mut(id).expect("grid present");
        grid.set_voxel(IVec3::new(10, 10, 50), Some(VoxColor(0x80_aa_bb_cc)));
        scene
    }

    #[test]
    fn visible_voxel_blocks() {
        let scene = floating_chunk_scene();
        // World (10, 10, -50) → chunk-local (10, 10, 50). That's the
        // inserted voxel; should block.
        assert!(is_blocked_in_scene(&scene, [10.0, 10.0, -50.0]));
    }

    #[test]
    fn out_of_chunk_position_is_air() {
        let scene = floating_chunk_scene();
        // Far outside any chunk's xy footprint.
        assert!(!is_blocked_in_scene(&scene, [-500.0, -500.0, -50.0]));
    }

    #[test]
    fn below_isolated_floating_voxel_is_air() {
        // For a sparse column with one solid voxel + bedrock,
        // getcube past the deepest visible slab returns Cube::Air
        // (it walks past the slab and finds the air pocket below).
        // Confirms collision handles "below a sparse voxel" as
        // air, not UnexposedSolid.
        let scene = floating_chunk_scene();
        // World (10, 10, 0) → chunk-local (10, 10, 100). Below the
        // single voxel at z=50, above bedrock at z=255.
        assert!(!is_blocked_in_scene(&scene, [10.0, 10.0, 0.0]));
    }

    fn build_user_repro_scene() -> Scene {
        use crate::{ship, terrain};
        let mut scene = Scene::new();
        let ground = scene.add_grid(GridTransform::at(DVec3::new(0.0, 0.0, 0.0)));
        terrain::build_ground(scene.grid_mut(ground).unwrap());
        let ship = scene.add_grid(GridTransform::at(DVec3::new(0.0, 0.0, -100.0)));
        ship::build_ship(scene.grid_mut(ship).unwrap());
        scene
    }

    /// Pin the saucer body collision: voxlap's slab format reports
    /// the *interior* of a solid run (between top floor-colour list
    /// and bottom ceiling-colour list) as `Cube::UnexposedSolid` —
    /// not `Color`. The collision check must treat that as solid
    /// (not air), or the camera flies through saucer interiors.
    #[test]
    fn saucer_body_interior_blocks() {
        let scene = build_user_repro_scene();
        // World z=-36 → ship-local z=64 = saucer centre. The voxel
        // there is in the slab interior; getcube returns
        // UnexposedSolid (not Color), but the user can't fly
        // through visible saucer body.
        assert!(
            is_blocked_in_scene(&scene, [64.0, 64.0, -36.0]),
            "saucer body interior must block (UnexposedSolid is solid material)"
        );
    }

    /// Pin the user-reported "invisible border" fix. F-key capture
    /// at world (79.33, 48.57, 154.06) — the camera was hitting an
    /// invisible wall at world z=155 (= ship grid's chunk-local
    /// z=255 = bedrock placeholder). The renderer hides the
    /// bedrock via `treat_z_max_as_air=true`; collision now does
    /// the same so the player can pass through.
    #[test]
    fn ship_bedrock_placeholder_does_not_block_collision() {
        let scene = build_user_repro_scene();
        // Captured pos itself — should already be unblocked (it's
        // just above bedrock at world z=155).
        assert!(!is_blocked_in_scene(&scene, [79.33, 48.57, 154.06]));
        // Across the bedrock plane: world z=156 was blocked before
        // the fix, must be air now.
        assert!(
            !is_blocked_in_scene(&scene, [79.33, 48.57, 156.0]),
            "world z=156 (just past ship bedrock) must be air"
        );
        // And further past — well below the ship grid's chunk
        // (ship-local z=257, outside chunk → my code skips, always air).
        assert!(!is_blocked_in_scene(&scene, [79.33, 48.57, 200.0]));
    }
}
