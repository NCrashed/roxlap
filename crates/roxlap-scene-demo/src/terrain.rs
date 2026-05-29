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
use roxlap_formats::vxl::Vxl;
use roxlap_scene::{ChunkGenerator, Grid, GridTransform, CHUNK_SIZE_XY, CHUNK_SIZE_Z};

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
/// `build_ground_extent` terrain in chz=1, then adds a handful of
/// tall stone mountains that BREACH the chunk boundary (base on the
/// hills at world z≈336, peak at world z≈100 = ~236 voxels tall,
/// spanning chz=1's lower half + most of chz=0). The mountains
/// exercise S4B.6.h mid-render chunk-Z handoff: rays from the chz=0
/// camera that miss a mountain peak walk into chz=1's hills via
/// handoff, while rays that hit a peak read its chz=0 voxels
/// directly.
pub fn build_ground_stacked(grid: &mut Grid) {
    build_ground_extent_at_chz(grid, GROUND_CHUNKS_X, GROUND_CHUNKS_Y, 1);
    add_tall_mountains(grid);
}

/// Stone colour for the tall stacked-demo mountains.
const MOUNTAIN_STONE: u32 = 0x80_8a_82_7a;

/// Place a handful of stepped-cone mountains spanning chz=0 + chz=1.
/// Bases at world z=336 (= the top of chz=1's hills), peaks at
/// world z=100 (= well inside chz=0). Each "step" is a square ring
/// of voxels stamped via `Grid::set_rect`, which handles the
/// chunk-XY + chunk-Z multi-chunk decomposition automatically.
///
/// **Bedrock-preserving placement.** Each chunk's column needs its
/// bedrock placeholder at `chunk_local_z = 255` (= the slab-list
/// terminator with `z1 == 0xff`) intact so the rasterizer's
/// bedrock-as-air check (= S4B.6.h mid-render handoff trigger)
/// fires at the chz boundary. If a mountain step's z range
/// includes world z=255 (chz=0's bedrock) or world z=511 (chz=1's
/// bedrock), set_rect would overwrite the placeholder, the column
/// would just end with mountain solid + no sentinel, and handoff
/// would never fire — the mountain's lower half would render as a
/// "floating top" with no chz+1 continuation. To prevent that,
/// stamp each step as up to TWO sub-rects that skip world z=255
/// and z=511 explicitly.
fn add_tall_mountains(grid: &mut Grid) {
    // Hand-placed XY locations — visible from the stacked-demo
    // spawn pose (looking +y from `(0, -120, 200)`).
    let centres = [(0i32, 200i32), (-180, 380), (220, 320)];
    const BASE_Z: i32 = 336; // on top of hills (world z)
    const PEAK_Z: i32 = 100; // breaches into chz=0
    const BASE_HALF: i32 = 40; // mountain half-width at the base
    const STEPS: i32 = 24; // discrete elevation rings
    let z_thickness = ((BASE_Z - PEAK_Z) / STEPS).max(1);

    for (cx, cy) in centres {
        for step in 0..STEPS {
            let frac = step as f32 / (STEPS - 1).max(1) as f32;
            // Linear interp z_top from BASE_Z down to PEAK_Z. (Voxlap
            // z-down: smaller z = closer to sky = mountain peak.)
            let z_top = BASE_Z - ((BASE_Z - PEAK_Z) as f32 * frac).round() as i32;
            // Radius shrinks linearly from BASE_HALF to 1.
            let half = (BASE_HALF as f32 * (1.0 - frac) + 1.0).round() as i32;
            let z_lo = z_top - z_thickness;
            let z_hi_excl = z_top + 1;
            stamp_mountain_step_preserving_bedrock(grid, cx, cy, half, z_lo, z_hi_excl);
        }
    }
}

/// Issue a `Grid::set_rect` for one mountain step, splitting at
/// per-chunk bedrock z values so the chunks' bedrock placeholders
/// at `chunk_local_z = 255` (= world z `chz*256 + 255`) stay
/// intact. Callers must pass `z_hi_excl > z_lo`.
fn stamp_mountain_step_preserving_bedrock(
    grid: &mut Grid,
    cx: i32,
    cy: i32,
    half: i32,
    z_lo: i32,
    z_hi_excl: i32,
) {
    let cs_z = CHUNK_SIZE_Z as i32;
    let mut z = z_lo;
    while z < z_hi_excl {
        // Find the next bedrock z >= current z (world z = N*cs_z - 1
        // for any positive N).
        let chunk_of_z = z.div_euclid(cs_z);
        let chunk_bedrock_z = chunk_of_z * cs_z + cs_z - 1;
        let segment_end = z_hi_excl.min(chunk_bedrock_z);
        if segment_end > z {
            grid.set_rect(
                IVec3::new(cx - half, cy - half, z),
                IVec3::new(cx + half + 1, cy + half + 1, segment_end),
                Some(MOUNTAIN_STONE),
            );
        }
        // Skip the bedrock voxel (= the placeholder z) and continue
        // from the next chunk's first voxel.
        z = chunk_bedrock_z + 1;
    }
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
                                           // S4B.6.k: don't cap the hill 2 short of bedrock — the previous
                                           // attempt (`z_max - 2`) caused a regression at the usual
                                           // downward-looking render path where drawcwall on the new
                                           // bedrock slab rendered a spurious black region at world z=510
                                           // in chz=1. The merged hill+bedrock slab structure is fine
                                           // because drawflor / drawfwall's own bedrock-z byte check
                                           // (`column[vptr+1] == 0xff >> mip`) treats the bedrock implicit
                                           // at z=255 as air without needing a separate slab.
    let z_hill_max = z_max - 1;

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
                        z1: z_hill_max as u8,
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

// =====================================================================
// S7.6: HillsChunkGenerator — streaming variant of `build_ground`.
// =====================================================================
//
// Wraps the same `terrain_height` heightmap + grass/dirt/stone palette
// as `build_ground_extent`, but emits one chunk at a time so a Grid
// can register it via `set_generator` and let `Scene::pump_streaming`
// stream chunks in / out as the camera moves. The result: visibly
// infinite green hills. Each `generate` call also runs lightmode-1
// directional shading + 4-level mips so streamed chunks render
// identically to the statically-built lattice.
//
// `chunk_idx.z != 0` returns a bedrock-only chunk — the terrain layer
// lives in `chz = 0` only; higher / lower chz layers are implicit air
// (the renderer's `treat_z_max_as_air` handles the bedrock sentinel).

/// Procedural-generation hook for the demo's green hills. Hand to
/// `Grid::set_generator` to make the ground stream as the camera
/// moves.
#[derive(Debug, Clone, Copy)]
pub struct HillsChunkGenerator;

impl ChunkGenerator for HillsChunkGenerator {
    fn generate(&self, chunk_idx: IVec3) -> Vxl {
        if chunk_idx.z != 0 {
            return empty_air_chunk();
        }
        let mut vxl = empty_air_chunk();
        stamp_hills_into(&mut vxl, chunk_idx);
        bake_chunk_lighting(&mut vxl);
        // 4 mip levels matches `OpticastSettings::mip_levels = 4` in
        // `main.rs` — past mip-3 the demo doesn't read further mips.
        vxl.generate_mips(4);
        vxl
    }
}

/// Produce a fresh bedrock-only chunk-sized Vxl by detaching the
/// chunk that [`Grid::ensure_chunk`] would create. Avoids reaching
/// into `roxlap-scene`'s private `chunks::empty_chunk_vxl` helper
/// while still using the canonical empty-chunk shape.
fn empty_air_chunk() -> Vxl {
    let mut g = Grid::new(GridTransform::identity());
    g.ensure_chunk(IVec3::ZERO);
    g.chunks.remove(&IVec3::ZERO).expect("just inserted")
}

/// Build the hills surface inside `vxl` for the chunk at
/// `chunk_idx`. Same `terrain_height` heightmap + grass/dirt/stone
/// palette as [`build_ground_extent_at_chz`], with the
/// neighbour-slope sample reaching one voxel into adjacent chunks
/// via `terrain_height` itself (the heightmap is a pure function
/// of world coords, so this stays seamless across chunk
/// boundaries).
fn stamp_hills_into(vxl: &mut Vxl, chunk_idx: IVec3) {
    let cs_xy = CHUNK_SIZE_XY as i32;
    let chunk_origin_x = chunk_idx.x * cs_xy;
    let chunk_origin_y = chunk_idx.y * cs_xy;
    let z_max = (CHUNK_SIZE_Z as i32) - 1;
    let z_hill_max = z_max - 1;

    let mut col_meta: Vec<ColMeta> = Vec::with_capacity((cs_xy * cs_xy) as usize);
    let mut spans: Vec<Vspan> = Vec::with_capacity((cs_xy * cs_xy) as usize);
    for ly in 0..cs_xy {
        for lx in 0..cs_xy {
            let wx = chunk_origin_x + lx;
            let wy = chunk_origin_y + ly;
            let surface_z = terrain_height(wx, wy);
            let slope = [
                (terrain_height(wx - 1, wy) - surface_z).abs(),
                (terrain_height(wx + 1, wy) - surface_z).abs(),
                (terrain_height(wx, wy - 1) - surface_z).abs(),
                (terrain_height(wx, wy + 1) - surface_z).abs(),
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
                z1: z_hill_max as u8,
            });
        }
    }
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
    set_spans_with_colfunc(vxl, &spans, SpanOp::Insert, colfunc);
}

/// Bake lightmode-1 directional shading into the chunk's alpha bytes.
///
/// The estnorm padding (`±ESTNORMRAD` voxels past each chunk face)
/// can't see the neighbour chunks here — they may not be loaded
/// yet at stream-in time. Reader returns `None` past the chunk's
/// own column range, which the bake treats as full air. Result:
/// edge columns get a slight brightness shift compared to the
/// scene-wide bake in `bake_lightmode_1`. Visible as a faint chunk
/// outline at low sun angles; acceptable for a v1 streaming demo
/// (full continuity needs the [[s4b-4-b-landed]] cross-chunk
/// `Grid::chunk` reader, which requires Grid context the generator
/// doesn't have).
fn bake_chunk_lighting(vxl: &mut Vxl) {
    let cs_xy = CHUNK_SIZE_XY as i32;
    let cs_z = CHUNK_SIZE_Z as i32;
    let cache = {
        let vxl_ref: &Vxl = &*vxl;
        let reader = |px: i32, py: i32| -> Option<&[u8]> {
            if px < 0 || px >= cs_xy || py < 0 || py >= cs_xy {
                return None;
            }
            let col_idx = (py as u32) * CHUNK_SIZE_XY + (px as u32);
            let off = vxl_ref.column_offset[col_idx as usize] as usize;
            Some(&vxl_ref.data[off..])
        };
        roxlap_core::EstNormCache::build_with_reader(reader, 0, 0, cs_xy, cs_xy)
    };
    roxlap_core::apply_lighting_with_cache(
        &mut vxl.data,
        &vxl.column_offset,
        CHUNK_SIZE_XY,
        0,
        0,
        0,
        cs_xy,
        cs_xy,
        cs_z,
        &cache,
        1, // LIGHTMODE
        &[],
    );
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
