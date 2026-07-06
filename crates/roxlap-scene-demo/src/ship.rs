//! Multi-chunk voxel ship.
//!
//! S4.2: 4×6×1 chunk lattice centred on the ship grid origin —
//! chunk indices `chx ∈ [-2, 2), chy ∈ [-3, 3), chz = 0` →
//! grid-local voxel extent `[-256..256) × [-384..384) × [0..256)`.
//! Saucer body centred at grid-local `(0, 0, BODY_Z)` with
//! `BODY_Z = 64` so when paired with the demo's ship grid origin
//! at world `(0, 500, -100)` the body floats at world z = -36
//! (same altitude as the S3.x/S4.1 single-chunk version).
//!
//! Geometry is emitted per chunk via batched
//! [`roxlap_formats::edit::set_spans`] — same pattern as
//! `terrain.rs`. The naive per-voxel `Grid::set_voxel` path would
//! be ~4 M edit calls at this scale (≈10-15 s startup); the
//! batched path drops to ~24 `set_spans` calls per colour
//! (negligible).
//!
//! S5 evolution: leave the geometry alone and let
//! `GridTransform::rotation` carry the 45°/45°/45° pitch/yaw/roll.

// Hull / bridge constants are tiny; every cast in this file is
// exact at the input ranges in use.
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
use roxlap_render::VoxColor;
use roxlap_scene::{Grid, CHUNK_SIZE_XY};

/// Voxlap-packed colours. `0x80` is voxlap's neutral brightness.
const HULL: VoxColor = VoxColor(0x80_8a_8a_8a); // gunmetal
const ACCENT: VoxColor = VoxColor(0x80_d0_60_30); // warning-stripe orange
const BRIDGE: VoxColor = VoxColor(0x80_30_60_e0); // cool blue

/// Hull ellipsoid half-extents in grid-local voxels.
const HULL_RX: i32 = 220;
const HULL_RY: i32 = 350;
const HULL_RZ: i32 = 12;

/// Z (grid-local) of the saucer equator — same as the S3.x ship
/// so altitude is preserved when paired with the demo's ship
/// grid origin.
const BODY_Z: i32 = 64;

/// Bridge cube half-extent (XY) and height (Z).
const BRIDGE_HALF_XY: i32 = 32;
const BRIDGE_HEIGHT: i32 = 24;

/// Ship footprint in chunks along grid-local x.
pub const SHIP_CHUNKS_X: i32 = 4;
/// Ship footprint in chunks along grid-local y.
pub const SHIP_CHUNKS_Y: i32 = 6;

/// Build the multi-chunk ship into `grid`.
///
/// Implementation:
/// 1. For each chunk in the centred `SHIP_CHUNKS_X × SHIP_CHUNKS_Y`
///    lattice, walk the chunk's local `(lx, ly)` grid in (y, x)
///    order — `set_spans`'s required sort.
/// 2. For each column, compute the ellipsoid `(gx/rx)² + (gy/ry)²`
///    discriminant. If `<= 1`, the column intersects the hull;
///    emit one or two hull `Vspan`s plus (optionally) an accent
///    `Vspan` at the equator stripe.
/// 3. Call `set_spans` once per colour per chunk.
/// 4. Bridge cube via `Grid::set_rect` — small enough to be
///    decomposed across at most 4 chunks naturally.
pub fn build_ship(grid: &mut Grid) {
    let cs_xy = CHUNK_SIZE_XY as i32;
    let half_x = SHIP_CHUNKS_X / 2;
    let half_y = SHIP_CHUNKS_Y / 2;

    // r_xy threshold at which the dz=±1 voxels still satisfy
    // voxlap's `r2 > 0.9` accent-stripe predicate. Lower r_xy
    // columns get hull-only output; higher r_xy columns split
    // into hull + accent + hull bands.
    let accent_threshold = 0.9_f32 - (1.0 / HULL_RZ as f32).powi(2);

    for chy in -half_y..(SHIP_CHUNKS_Y - half_y) {
        for chx in -half_x..(SHIP_CHUNKS_X - half_x) {
            let chunk_origin_x = chx * cs_xy;
            let chunk_origin_y = chy * cs_xy;
            let mut hull_spans: Vec<Vspan> = Vec::new();
            let mut accent_spans: Vec<Vspan> = Vec::new();

            for ly in 0..cs_xy {
                for lx in 0..cs_xy {
                    let gx = chunk_origin_x + lx;
                    let gy = chunk_origin_y + ly;
                    let nx = gx as f32 / HULL_RX as f32;
                    let ny = gy as f32 / HULL_RY as f32;
                    let r_xy = nx * nx + ny * ny;
                    if r_xy > 1.0 {
                        continue;
                    }
                    let dz_max = (HULL_RZ as f32) * (1.0 - r_xy).sqrt();
                    let dz_max_i = dz_max.floor() as i32;
                    let z_top = BODY_Z - dz_max_i;
                    let z_bot = BODY_Z + dz_max_i;
                    let sx = lx as u32;
                    let sy = ly as u32;

                    if r_xy > accent_threshold {
                        // Column intersects the equator stripe.
                        // Order: hull-above (smaller z), accent
                        // (middle z), hull-below (larger z) — keeps
                        // the per-(lx, ly) Vspan list ascending in
                        // z0 as required by set_spans's contract.
                        let acc_top = z_top.max(BODY_Z - 1);
                        let acc_bot = z_bot.min(BODY_Z + 1);
                        if z_top < acc_top {
                            hull_spans.push(Vspan {
                                x: sx,
                                y: sy,
                                z0: z_top as u8,
                                z1: (acc_top - 1) as u8,
                            });
                        }
                        if acc_top <= acc_bot {
                            accent_spans.push(Vspan {
                                x: sx,
                                y: sy,
                                z0: acc_top as u8,
                                z1: acc_bot as u8,
                            });
                        }
                        if acc_bot < z_bot {
                            hull_spans.push(Vspan {
                                x: sx,
                                y: sy,
                                z0: (acc_bot + 1) as u8,
                                z1: z_bot as u8,
                            });
                        }
                    } else {
                        hull_spans.push(Vspan {
                            x: sx,
                            y: sy,
                            z0: z_top as u8,
                            z1: z_bot as u8,
                        });
                    }
                }
            }

            // Skip empty-bbox chunks (no ellipsoid intersection) so
            // we don't materialise unused chunks via ensure_chunk —
            // they stay as implicit-air entries in the combined view.
            if hull_spans.is_empty() && accent_spans.is_empty() {
                continue;
            }
            let vxl = grid.ensure_chunk(IVec3::new(chx, chy, 0));
            set_spans(vxl, &hull_spans, Some(HULL));
            set_spans(vxl, &accent_spans, Some(ACCENT));
        }
    }

    // Bridge cube on top (smaller z = "above" in voxlap-z-down).
    // BRIDGE_HALF_XY = 32 → cube spans the 4 chunks adjacent to
    // the grid origin; Grid::set_rect handles the per-chunk
    // decomposition.
    let bridge_top_z = BODY_Z - HULL_RZ - BRIDGE_HEIGHT;
    let bridge_bot_z = BODY_Z - HULL_RZ - 1;
    grid.set_rect(
        IVec3::new(-BRIDGE_HALF_XY, -BRIDGE_HALF_XY, bridge_top_z),
        IVec3::new(BRIDGE_HALF_XY, BRIDGE_HALF_XY, bridge_bot_z),
        Some(BRIDGE),
    );
}
