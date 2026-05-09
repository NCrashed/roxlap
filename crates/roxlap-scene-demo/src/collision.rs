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
use roxlap_scene::{Scene, CHUNK_SIZE_Z};

/// Camera "skin" radius in voxel units. A move is blocked when any
/// voxel inside the cube `pos ± PLAYER_RADIUS` is solid; matches
/// the cave-demo / roxlap-host tuning.
pub const PLAYER_RADIUS: f64 = 0.3;

/// `true` if any voxel of any grid in `scene` overlaps the camera's
/// `±PLAYER_RADIUS` cube around `world_pos`.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::similar_names
)]
pub fn is_blocked_in_scene(scene: &Scene, world_pos: [f64; 3]) -> bool {
    for (_id, grid) in scene.grids() {
        // S3.x: axis-aligned grids only — translate world → grid-local.
        let lx_centre = world_pos[0] - grid.transform.origin.x;
        let ly_centre = world_pos[1] - grid.transform.origin.y;
        let lz_centre = world_pos[2] - grid.transform.origin.z;
        let lo_x = (lx_centre - PLAYER_RADIUS).floor() as i32;
        let hi_x = (lx_centre + PLAYER_RADIUS).floor() as i32;
        let lo_y = (ly_centre - PLAYER_RADIUS).floor() as i32;
        let hi_y = (ly_centre + PLAYER_RADIUS).floor() as i32;
        let lo_z = (lz_centre - PLAYER_RADIUS).floor() as i32;
        let hi_z = (lz_centre + PLAYER_RADIUS).floor() as i32;

        // S3.x: single-chunk grids only — only chunk (0, 0, 0) is
        // queried. S4 will lift this to a sparse-map walk over the
        // chunk indices the cube intersects.
        let Some(chunk) = grid.chunk(IVec3::ZERO) else {
            continue;
        };
        let cs_xy = chunk.vsid as i32;
        let cs_z = CHUNK_SIZE_Z as i32;
        for vz in lo_z..=hi_z {
            for vy in lo_y..=hi_y {
                for vx in lo_x..=hi_x {
                    if vx < 0 || vy < 0 || vz < 0 || vx >= cs_xy || vy >= cs_xy || vz >= cs_z {
                        // Outside this chunk — keep walking (other
                        // grids may still overlap).
                        continue;
                    }
                    if !matches!(
                        getcube(&chunk.data, &chunk.column_offset, chunk.vsid, vx, vy, vz),
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
