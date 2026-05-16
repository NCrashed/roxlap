//! Procedural ground / planet surface.
//!
//! S4.1: emits a `GROUND_CHUNKS_X × GROUND_CHUNKS_Y` lattice of
//! chunks into the given grid. The terrain builder uses
//! `roxlap_formats::edit::set_spans_with_colfunc` to insert each
//! column's full z stack in **one** call with a colour-by-z
//! closure — that compounds correctly into the column, while
//! three back-to-back single-colour `set_spans` calls hit voxlap's
//! `insslab` last-slab merge edge case and silently drop dirt /
//! stone bands.
//!
//! S4.1 ships with the full 32×32 ground (combined `vsid = 4096`,
//! 16M virtual columns) centred on the grid origin — chunks
//! span grid-local chunk-XY `[-16..16) × [-16..16)`.
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
use roxlap_formats::edit::{set_spans_with_colfunc, SpanOp, Vspan};
use roxlap_scene::{Grid, CHUNK_SIZE_XY, CHUNK_SIZE_Z};

/// Per-column metadata the colfunc closure consults: each
/// column's surface z and whether the top is stone (steep slope)
/// or grass.
#[derive(Clone, Copy)]
struct ColMeta {
    surface_z: i32,
    top_is_stone: bool,
}

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
/// 2. For each chunk, walk its 128² columns in `(ly, lx)` order
///    and emit ONE `Vspan` per column covering `[surface_z,
///    z_max-1]`. Capture per-column `(surface_z, top_is_stone)`
///    metadata so the colfunc closure picks grass / dirt / stone
///    per voxel z.
/// 3. One `set_spans_with_colfunc` call per chunk inserts the
///    full vertical stack with the right colours.
///
/// `terrain_height` takes grid-local world coordinates, so the
/// terrain stays continuous across chunk boundaries — there's no
/// per-chunk re-tiling artefact at the seams. Per-chunk lighting
/// bake uses the combined-view path to avoid the brightness-jump
/// seam (`project_chunk_edge_lighting_seam.md`).
pub fn build_ground_extent(grid: &mut Grid, chunks_x: i32, chunks_y: i32) {
    build_ground_extent_at_chz(grid, chunks_x, chunks_y, 0);
}

/// `chunks_z=2` showcase variant: builds an all-air chz=0 layer + the
/// `build_ground_extent` terrain in chz=1. The camera sits in chz=0
/// air-gap and uses S4B.6.e's cross-chunk look-down to see chz=1's
/// floor below. Materialises chz=0 chunks as empty so
/// `Grid::chunk_xyz_backing` enumerates the full stack.
pub fn build_ground_stacked(grid: &mut Grid) {
    build_ground_extent_at_chz(grid, GROUND_CHUNKS_X, GROUND_CHUNKS_Y, 1);
}

/// Per-chunk-z variant. `ground_chz=0` matches `build_ground_extent`
/// byte-for-byte (no extra chunks materialised). `ground_chz>=1`
/// materialises empty chunks for `0..ground_chz` so the chz=0..N
/// layers above the terrain are walkable by the camera but contain
/// no voxels.
pub fn build_ground_extent_at_chz(grid: &mut Grid, chunks_x: i32, chunks_y: i32, ground_chz: i32) {
    let cs_xy = CHUNK_SIZE_XY as i32;
    let half_chunks_x = chunks_x / 2;
    let half_chunks_y = chunks_y / 2;
    let world_x_lo = -half_chunks_x * cs_xy;
    let world_x_hi = (chunks_x - half_chunks_x) * cs_xy;
    let world_y_lo = -half_chunks_y * cs_xy;
    let world_y_hi = (chunks_y - half_chunks_y) * cs_xy;
    let world_x_extent = world_x_hi - world_x_lo;
    let world_y_extent = world_y_hi - world_y_lo;

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

    for chy in -half_chunks_y..(chunks_y - half_chunks_y) {
        for chx in -half_chunks_x..(chunks_x - half_chunks_x) {
            let chunk_origin_x = chx * cs_xy;
            let chunk_origin_y = chy * cs_xy;

            // Stage spans + per-column metadata in (ly, lx) order
            // — matches `set_spans`'s sort contract.
            let mut col_meta: Vec<ColMeta> = Vec::with_capacity((cs_xy * cs_xy) as usize);
            let mut spans: Vec<Vspan> = Vec::with_capacity((cs_xy * cs_xy) as usize);
            for ly in 0..cs_xy {
                for lx in 0..cs_xy {
                    let wx = chunk_origin_x + lx;
                    let wy = chunk_origin_y + ly;
                    let surface_z = h_at(wx, wy);
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
                    col_meta.push(ColMeta {
                        surface_z,
                        top_is_stone,
                    });
                    spans.push(Vspan {
                        x: lx as u32,
                        y: ly as u32,
                        z0: surface_z as u8,
                        z1: (z_max - 1) as u8,
                    });
                }
            }

            // Colfunc picks grass / dirt / stone by relative z from
            // the column's surface. `x` / `y` are chunk-local voxel
            // coords (Vspan's u32 fields cast to i32 inside set_spans).
            let colfunc = move |x: i32, y: i32, z: i32| -> i32 {
                let lx = x.clamp(0, cs_xy - 1) as usize;
                let ly = y.clamp(0, cs_xy - 1) as usize;
                let meta = col_meta[ly * (cs_xy as usize) + lx];
                let dz = z - meta.surface_z;
                let colour_u32 = if meta.top_is_stone {
                    STONE
                } else if dz == 0 {
                    GRASS
                } else if dz <= DIRT_BAND_THICKNESS {
                    DIRT
                } else {
                    STONE
                };
                colour_u32 as i32
            };

            // Materialise empty chunks for the air layers above the
            // terrain (chz=0..ground_chz). These columns are
            // bedrock-only — S4B.6.e's cross-chunk look-down walks
            // through them at seed time.
            for chz in 0..ground_chz {
                grid.ensure_chunk(IVec3::new(chx, chy, chz));
            }
            let vxl = grid.ensure_chunk(IVec3::new(chx, chy, ground_chz));
            set_spans_with_colfunc(vxl, &spans, SpanOp::Insert, colfunc);
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
