//! Sky-borne voxel ship.
//!
//! S3.x: axis-aligned, single-chunk ship hull built into chunk
//! `(0, 0, 0)`. The hull is a flattened ellipsoid (saucer shape)
//! plus a cube "command bridge" on top — recognisable enough that
//! the user can tell at a glance the renderer is compositing two
//! independent grids rather than rendering one merged volume.
//!
//! S4 evolution: extend to 4×6×1 chunks of more detailed
//! geometry (hull keel, deck, wing struts).
//! S5 evolution: leave geometry alone and let `GridTransform`'s
//! `rotation` field carry the 45°/45°/45° pitch/yaw/roll.

// Hull / bridge extents are tiny constants; every cast in this
// file is exact at the input ranges in use.
#![allow(clippy::cast_precision_loss)]

use glam::IVec3;
use roxlap_scene::Grid;

/// Voxlap-packed colours. `0x80` is voxlap's neutral brightness.
const HULL: u32 = 0x80_8a_8a_8a; // gunmetal
const ACCENT: u32 = 0x80_d0_60_30; // warning-stripe orange
const BRIDGE: u32 = 0x80_30_60_e0; // cool blue

/// Hull ellipsoid half-extents (chunk-local voxels).
const HULL_RX: i32 = 50;
const HULL_RY: i32 = 30;
const HULL_RZ: i32 = 8;

/// Bridge cube extents.
const BRIDGE_HALF_XY: i32 = 8;
const BRIDGE_HEIGHT: i32 = 10;

/// Centre of the ship inside its chunk. Offsets the geometry
/// inside chunk (0,0,0) so neighbouring "implicit-air" chunks
/// don't show clipping at the chunk boundaries.
const CENTRE: IVec3 = IVec3::new(64, 64, 64);

/// Build the ship into the grid's `(0, 0, 0)` chunk.
pub fn build_ship(grid: &mut Grid) {
    // Hull: flattened ellipsoid. Walk the bounding box and test
    // `(dx/rx)² + (dy/ry)² + (dz/rz)² <= 1`.
    for dz in -HULL_RZ..=HULL_RZ {
        for dy in -HULL_RY..=HULL_RY {
            for dx in -HULL_RX..=HULL_RX {
                let nx = dx as f32 / HULL_RX as f32;
                let ny = dy as f32 / HULL_RY as f32;
                let nz = dz as f32 / HULL_RZ as f32;
                let r2 = nx * nx + ny * ny + nz * nz;
                if r2 > 1.0 {
                    continue;
                }
                // Accent stripe around the equator (nz close to 0).
                let color = if dz.abs() <= 1 && r2 > 0.9 {
                    ACCENT
                } else {
                    HULL
                };
                grid.set_voxel(CENTRE + IVec3::new(dx, dy, dz), Some(color));
            }
        }
    }

    // Bridge cube on top (smaller z = "above" in voxlap-z-down).
    let bridge_top_z = CENTRE.z - HULL_RZ - BRIDGE_HEIGHT;
    let bridge_bot_z = CENTRE.z - HULL_RZ - 1;
    grid.set_rect(
        IVec3::new(
            CENTRE.x - BRIDGE_HALF_XY,
            CENTRE.y - BRIDGE_HALF_XY,
            bridge_top_z,
        ),
        IVec3::new(
            CENTRE.x + BRIDGE_HALF_XY,
            CENTRE.y + BRIDGE_HALF_XY,
            bridge_bot_z,
        ),
        Some(BRIDGE),
    );
}
