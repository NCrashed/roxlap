//! Procedural ground / planet surface.
//!
//! S4.1: emits a `GROUND_CHUNKS_X × GROUND_CHUNKS_Y` lattice of
//! chunks into the given grid. The terrain builder skips
//! `Grid::set_voxel`'s per-voxel scum2 path and goes straight to
//! `roxlap_formats::edit::set_spans` per chunk — at 32×32 chunks
//! the per-voxel path was ~3 billion `set_cube` calls; batched
//! per-chunk spans land in seconds.
//!
//! S4.1 ships with the full 32×32 ground (combined `vsid = 4096`,
//! 16M virtual columns) **centred on the grid origin** — chunks
//! span grid-local chunk-XY `[-16..16) × [-16..16)`. The world
//! feels like it surrounds the player rather than extending only
//! in `+x / +y`. S4.0's 2×1 micro-bench has been retired.
//!
//! Material palette:
//! - `grass` for the topmost voxel of each column when the local
//!   slope is gentle.
//! - `stone` for the topmost voxel when the slope is steep.
//! - `dirt` for a band immediately below grass.
//! - `stone` everywhere below the dirt band.

// Voxel coords + heightmap values stay in `[0, CHUNK_SIZE_*]` so
// every i32 ↔ f32 ↔ usize cast in this module is safe.
#![allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::similar_names
)]

use glam::IVec3;
use roxlap_formats::edit::{set_spans, Vspan};
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
pub const GROUND_CHUNKS_X: i32 = 32;
/// Ground footprint in chunks along grid-local +y.
pub const GROUND_CHUNKS_Y: i32 = 32;

/// Build the ground terrain into the `GROUND_CHUNKS_X × GROUND_CHUNKS_Y`
/// chunk lattice, centred on the grid origin — chunk indices span
/// `[-GROUND_CHUNKS_X/2..GROUND_CHUNKS_X/2) × [-GROUND_CHUNKS_Y/2..GROUND_CHUNKS_Y/2)`.
/// Equivalent to `build_ground_extent(grid, GROUND_CHUNKS_X, GROUND_CHUNKS_Y)`.
pub fn build_ground(grid: &mut Grid) {
    build_ground_extent(grid, GROUND_CHUNKS_X, GROUND_CHUNKS_Y);
}

/// Build a `chunks_x × chunks_y` ground lattice into `grid`,
/// centred on the grid origin: chunk-XY indices span
/// `[-chunks_x/2..chunks_x/2) × [-chunks_y/2..chunks_y/2)` so the
/// player at grid-local `(0, 0)` sits in the middle of the world.
///
/// Useful for tests that need a smaller terrain than the demo's
/// 32×32 default — building the full lattice takes ~2-3 seconds
/// and dominates test wall-time.
///
/// Implementation:
/// 1. Pre-compute the heightmap over the full grid-local world
///    extent so neighbour-height lookups for slope detection are
///    O(1).
/// 2. For each chunk in the lattice, materialise it via
///    [`Grid::ensure_chunk`] and stage three [`Vspan`] lists in
///    chunk-local coords (grass / dirt / stone). One
///    `set_spans` call per colour per chunk replaces ~150
///    `set_voxel` calls per column.
///
/// `terrain_height` takes grid-local world coordinates, so the
/// terrain stays continuous across chunk boundaries — there's no
/// per-chunk re-tiling artefact at the seams (though per-chunk
/// lighting bake currently produces a brightness seam; see
/// `project_chunk_edge_lighting_seam.md`).
pub fn build_ground_extent(grid: &mut Grid, chunks_x: i32, chunks_y: i32) {
    let cs_xy = CHUNK_SIZE_XY as i32;
    // Half-extents in chunks; world spans
    // `[-half_chunks_*, half_chunks_*)` along each axis. For an
    // odd `chunks_x`/`chunks_y` the world slightly favours `+`.
    let half_chunks_x = chunks_x / 2;
    let half_chunks_y = chunks_y / 2;
    let world_x_lo = -half_chunks_x * cs_xy;
    let world_x_hi = (chunks_x - half_chunks_x) * cs_xy;
    let world_y_lo = -half_chunks_y * cs_xy;
    let world_y_hi = (chunks_y - half_chunks_y) * cs_xy;
    let world_x_extent = world_x_hi - world_x_lo;
    let world_y_extent = world_y_hi - world_y_lo;

    // Pre-compute the heightmap so we can look up neighbour heights
    // for slope detection without recomputing. `heights` is indexed
    // by (wy - world_y_lo) * world_x_extent + (wx - world_x_lo).
    let n_cells = (world_x_extent * world_y_extent) as usize;
    let mut heights = vec![0i32; n_cells];
    for wy in world_y_lo..world_y_hi {
        for wx in world_x_lo..world_x_hi {
            let idx = ((wy - world_y_lo) * world_x_extent + (wx - world_x_lo)) as usize;
            heights[idx] = terrain_height(wx, wy);
        }
    }
    let h_at = |x: i32, y: i32| -> i32 {
        let xc = x.clamp(world_x_lo, world_x_hi - 1);
        let yc = y.clamp(world_y_lo, world_y_hi - 1);
        let idx = ((yc - world_y_lo) * world_x_extent + (xc - world_x_lo)) as usize;
        heights[idx]
    };

    let z_max = (CHUNK_SIZE_Z as i32) - 1; // bedrock placeholder z

    // Per-chunk pass over the centred chunk lattice.
    for chy in -half_chunks_y..(chunks_y - half_chunks_y) {
        for chx in -half_chunks_x..(chunks_x - half_chunks_x) {
            let chunk_origin_x = chx * cs_xy;
            let chunk_origin_y = chy * cs_xy;

            // Three Vspan lists, accumulated in (y, x) order to
            // satisfy set_spans's "sorted ascending by (y, x)
            // then by z0" contract. Each (x, y) column contributes
            // at most one span per colour, so within-column sort
            // is trivially satisfied.
            let mut grass_spans: Vec<Vspan> = Vec::new();
            let mut dirt_spans: Vec<Vspan> = Vec::new();
            let mut stone_spans: Vec<Vspan> = Vec::new();

            for ly in 0..cs_xy {
                for lx in 0..cs_xy {
                    let wx = chunk_origin_x + lx;
                    let wy = chunk_origin_y + ly;
                    let surface_z = h_at(wx, wy);
                    // Local slope: max |dh| over the 4-neighbourhood.
                    // `h_at` clamps at the grid edge so neighbours
                    // outside the lattice get the edge's height (no
                    // artificial cliff at the grid boundary).
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

                    // Vspan x/y are chunk-local u32 (set_spans skips
                    // out-of-bounds silently otherwise). Both fit
                    // u32 trivially since 0..128.
                    let sx = lx as u32;
                    let sy = ly as u32;

                    if top_is_stone {
                        // Single stone span covers surface → above-
                        // bedrock. Saves 3 spans per column on the
                        // steep tiles.
                        stone_spans.push(Vspan {
                            x: sx,
                            y: sy,
                            z0: surface_z as u8,
                            z1: (z_max - 1) as u8,
                        });
                    } else {
                        // Grass: just the surface voxel.
                        grass_spans.push(Vspan {
                            x: sx,
                            y: sy,
                            z0: surface_z as u8,
                            z1: surface_z as u8,
                        });
                        // Dirt band, clamped to stay above bedrock.
                        let dirt_top = surface_z + 1;
                        let dirt_bot = (surface_z + DIRT_BAND_THICKNESS).min(z_max - 1);
                        if dirt_bot >= dirt_top {
                            dirt_spans.push(Vspan {
                                x: sx,
                                y: sy,
                                z0: dirt_top as u8,
                                z1: dirt_bot as u8,
                            });
                        }
                        // Stone fill from below dirt down to just
                        // above bedrock.
                        let stone_top = surface_z + DIRT_BAND_THICKNESS + 1;
                        if stone_top < z_max {
                            stone_spans.push(Vspan {
                                x: sx,
                                y: sy,
                                z0: stone_top as u8,
                                z1: (z_max - 1) as u8,
                            });
                        }
                    }
                }
            }

            let vxl = grid.ensure_chunk(IVec3::new(chx, chy, 0));
            // One ScumCtx batch per colour band. Empty inputs are a
            // no-op so we don't bother filtering chunks where every
            // column happens to be stone-only.
            set_spans(vxl, &grass_spans, Some(GRASS));
            set_spans(vxl, &dirt_spans, Some(DIRT));
            set_spans(vxl, &stone_spans, Some(STONE));
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
