//! Procedural ground / planet surface.
//!
//! S3.x: emits a single-chunk heightmap (128×128 cells) into the
//! given grid's chunk `(0, 0, 0)`. The heightmap is the sum of a
//! few sine waves — cheap, deterministic, no external noise dep,
//! visibly hilly. Voxlap z-down convention: a smaller `z` value
//! is closer to the sky; the surface lives at the heightmap's
//! `z` and solid voxels extend toward `z = 255` (bedrock).
//!
//! Material palette:
//! - `grass` for the topmost voxel of each column when the local
//!   slope is gentle.
//! - `stone` for the topmost voxel when the slope is steep.
//! - `dirt` for a band immediately below grass.
//! - `stone` everywhere below the dirt band.
//!
//! S4 evolution: replace the single-chunk loop with iteration
//! over `n_chunks_xy² × n_chunks_z` chunks; the per-column
//! heightmap function works at world coordinates so it doesn't
//! care which chunk it lands in.

// Voxel coords + heightmap values stay in `[0, CHUNK_SIZE_*]` so
// every i32 ↔ f32 ↔ usize cast in this module is safe.
#![allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
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

/// Build the ground terrain into the grid's `(0, 0, 0)` chunk.
///
/// Generates a deterministic hilly heightmap, then walks columns
/// to write grass / dirt / stone bands. Calls `Grid::set_voxel`
/// per voxel — slow-ish for a 128×128×~30 voxel terrain but
/// runs once at startup.
///
/// **S4 hook:** replace the `cx, cy in 0..CHUNK_SIZE_XY` loops
/// below with iteration over a 32×32 chunk lattice; the
/// heightmap function `terrain_height(world_x, world_y)` already
/// works at world coordinates.
pub fn build_ground(grid: &mut Grid) {
    let cs_xy = CHUNK_SIZE_XY as i32;
    // Pre-compute the heightmap so we can look up neighbour heights
    // for slope detection without recomputing.
    let mut heights = vec![0i32; (cs_xy * cs_xy) as usize];
    for cy in 0..cs_xy {
        for cx in 0..cs_xy {
            heights[(cy * cs_xy + cx) as usize] = terrain_height(cx, cy);
        }
    }
    let h_at = |x: i32, y: i32| -> i32 {
        let xc = x.clamp(0, cs_xy - 1);
        let yc = y.clamp(0, cs_xy - 1);
        heights[(yc * cs_xy + xc) as usize]
    };

    let z_max = (CHUNK_SIZE_Z as i32) - 1; // bedrock placeholder z

    for cy in 0..cs_xy {
        for cx in 0..cs_xy {
            let surface_z = h_at(cx, cy);
            // Local slope: max |dh| over the 4-neighbourhood.
            let slope = [
                (h_at(cx - 1, cy) - surface_z).abs(),
                (h_at(cx + 1, cy) - surface_z).abs(),
                (h_at(cx, cy - 1) - surface_z).abs(),
                (h_at(cx, cy + 1) - surface_z).abs(),
            ]
            .into_iter()
            .max()
            .unwrap_or(0);
            let top_is_stone = slope >= STONE_SLOPE_THRESHOLD;
            let top_color = if top_is_stone { STONE } else { GRASS };

            // Surface voxel.
            grid.set_voxel(IVec3::new(cx, cy, surface_z), Some(top_color));
            // Dirt band — only when the surface is grass; under
            // stone hillsides, stone goes all the way down.
            for dz in 1..=DIRT_BAND_THICKNESS {
                let z = surface_z + dz;
                if z >= z_max {
                    break;
                }
                let band_color = if top_is_stone { STONE } else { DIRT };
                grid.set_voxel(IVec3::new(cx, cy, z), Some(band_color));
            }
            // Stone fill from the bottom of the dirt band down to
            // (but not including) bedrock at z = z_max.
            let stone_top = surface_z + DIRT_BAND_THICKNESS + 1;
            for z in stone_top..z_max {
                grid.set_voxel(IVec3::new(cx, cy, z), Some(STONE));
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
