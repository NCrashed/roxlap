//! Procedural ground / planet surface.
//!
//! S4.0: emits a `GROUND_CHUNKS_X × GROUND_CHUNKS_Y` lattice of
//! chunks into the given grid. The S3.x single-chunk loop became
//! the inner loop of a chunk-lattice walk; the per-column
//! heightmap function takes grid-local world coordinates so it
//! lands different terrain in each chunk without re-tiling.
//!
//! At S4.0 the lattice is locked to 2×1 chunks (`vsid = 256`) so
//! the seam at grid-local x=128 exercises the cross-chunk gline
//! path without the 4096²-column allocation cost of the planned
//! 32×32 final demo. Bumping to 32×32 lands at S4.1 once S4.0
//! has settled.
//!
//! Material palette:
//! - `grass` for the topmost voxel of each column when the local
//!   slope is gentle.
//! - `stone` for the topmost voxel when the slope is steep.
//! - `dirt` for a band immediately below grass.
//! - `stone` everywhere below the dirt band.

// Voxel coords + heightmap values stay in `[0, CHUNK_SIZE_*]` so
// every i32 ↔ f32 ↔ usize cast in this module is safe.
// `world_x_extent` / `world_y_extent` differ by one letter; the
// pair name follows from the X/Y axis split, which is more
// readable than `extent_along_x` / `extent_along_y`.
#![allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::similar_names
)]

use glam::IVec3;
use roxlap_scene::{Grid, CHUNK_SIZE_XY, CHUNK_SIZE_Z};

/// Voxlap-packed colour: `(brightness << 24) | (R << 16) | (G << 8) | B`.
/// `0x80` brightness is voxlap's neutral / unlit baseline.
const GRASS: u32 = 0x80_4d_8a_3a; // mossy green
const DIRT: u32 = 0x80_6b_4a_28; // earthy brown
const STONE: u32 = 0x80_7a_7a_82; // cool gray

/// Voxels of dirt sitting between grass and stone.
const DIRT_BAND_THICKNESS: i32 = 4;

/// Slope (in voxels) above which the topmost voxel turns to
/// stone — exposes the steeper hillsides as rock.
const STONE_SLOPE_THRESHOLD: i32 = 3;

/// Ground footprint in chunks along grid-local +x.
pub const GROUND_CHUNKS_X: i32 = 2;
/// Ground footprint in chunks along grid-local +y.
pub const GROUND_CHUNKS_Y: i32 = 1;

/// Build the ground terrain into a `GROUND_CHUNKS_X × GROUND_CHUNKS_Y`
/// chunk lattice. Voxels are inserted via [`Grid::set_voxel`]
/// against grid-local coordinates; the multi-chunk decomposition
/// in [`crate::edit`] (re-exposed via the same `set_voxel` entry
/// point) routes each voxel to its chunk.
///
/// `terrain_height` takes grid-local world coordinates, so the
/// terrain stays continuous across chunk boundaries — there's no
/// per-chunk re-tiling artefact at the seams.
pub fn build_ground(grid: &mut Grid) {
    let cs_xy = CHUNK_SIZE_XY as i32;
    let world_x_extent = GROUND_CHUNKS_X * cs_xy;
    let world_y_extent = GROUND_CHUNKS_Y * cs_xy;

    // Pre-compute the heightmap so we can look up neighbour heights
    // for slope detection without recomputing. Indexed by grid-local
    // world (x, y).
    let n_cells = (world_x_extent * world_y_extent) as usize;
    let mut heights = vec![0i32; n_cells];
    for wy in 0..world_y_extent {
        for wx in 0..world_x_extent {
            heights[(wy * world_x_extent + wx) as usize] = terrain_height(wx, wy);
        }
    }
    let h_at = |x: i32, y: i32| -> i32 {
        let xc = x.clamp(0, world_x_extent - 1);
        let yc = y.clamp(0, world_y_extent - 1);
        heights[(yc * world_x_extent + xc) as usize]
    };

    let z_max = (CHUNK_SIZE_Z as i32) - 1; // bedrock placeholder z

    for wy in 0..world_y_extent {
        for wx in 0..world_x_extent {
            let surface_z = h_at(wx, wy);
            // Local slope: max |dh| over the 4-neighbourhood.
            let slope = [
                (h_at(wx - 1, wy) - surface_z).abs(),
                (h_at(wx + 1, wy) - surface_z).abs(),
                (h_at(wx, wy - 1) - surface_z).abs(),
                (h_at(wx, wy + 1) - surface_z).abs(),
            ]
            .into_iter()
            .max()
            .unwrap_or(0);
            let top_is_stone = slope >= STONE_SLOPE_THRESHOLD;
            let top_color = if top_is_stone { STONE } else { GRASS };

            // Surface voxel.
            grid.set_voxel(IVec3::new(wx, wy, surface_z), Some(top_color));
            // Dirt band — only when the surface is grass; under
            // stone hillsides, stone goes all the way down.
            for dz in 1..=DIRT_BAND_THICKNESS {
                let z = surface_z + dz;
                if z >= z_max {
                    break;
                }
                let band_color = if top_is_stone { STONE } else { DIRT };
                grid.set_voxel(IVec3::new(wx, wy, z), Some(band_color));
            }
            // Stone fill from the bottom of the dirt band down to
            // (but not including) bedrock at z = z_max.
            let stone_top = surface_z + DIRT_BAND_THICKNESS + 1;
            for z in stone_top..z_max {
                grid.set_voxel(IVec3::new(wx, wy, z), Some(STONE));
            }
        }
    }
}

/// Heightmap function. Voxlap z-down: a *smaller* `z` is
/// closer to the sky, so "hills" are columns with smaller surface
/// `z` than valleys.
///
/// Sum of three sine waves at different frequencies + a low base
/// height. Deterministic and cheap.
fn terrain_height(world_x: i32, world_y: i32) -> i32 {
    // Base surface at z=200 (deep enough that hills can rise to
    // ~z=160 and still leave a sky band above z=0..150 for the
    // ship to fly through).
    const BASE_Z: f32 = 200.0;
    const AMPLITUDE: f32 = 18.0;
    let fx = world_x as f32;
    let fy = world_y as f32;
    let h = AMPLITUDE
        * (0.5 * ((fx * 0.07).sin() + (fy * 0.06).sin())
            + 0.3 * ((fx * 0.15 + fy * 0.11).sin())
            + 0.2 * (((fx + fy) * 0.05).cos()));
    // Smaller z = higher peak; subtract h so hills rise toward z=0.
    let z = BASE_Z - h;
    (z.round() as i32).clamp(80, (CHUNK_SIZE_Z as i32) - 6)
}
