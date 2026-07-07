//! Address math for the world ↔ grid-local ↔ chunk + voxel
//! coordinate spaces.
//!
//! Three spaces:
//!
//! 1. **World** (f64). Universe-level positions; one [`DVec3`]
//!    per point.
//! 2. **Grid-local** (f64). Position in a grid's local frame
//!    (origin + rotation undone, then divided by the grid's
//!    `voxel_world_size` — SC). Integer voxel coordinate `v`
//!    covers grid-local space `[v, v+1)` on each axis; one voxel
//!    spans `voxel_world_size` world units (default `1.0` = 1:1).
//! 3. **Chunk + voxel-in-chunk** (i32 + u32). A grid-local voxel
//!    coordinate `v: IVec3` decomposes into a chunk index
//!    `c: IVec3` and a voxel offset `u: UVec3` within that
//!    chunk. Chunks are XY = [`CHUNK_SIZE_XY`], Z =
//!    [`CHUNK_SIZE_Z`], so `u.x, u.y < CHUNK_SIZE_XY` and
//!    `u.z < CHUNK_SIZE_Z`.
//!
//! Negative voxel coords decompose with [`i32::div_euclid`] /
//! [`i32::rem_euclid`] semantics: voxel `-1` lives in chunk `-1`
//! at position `(CHUNK_SIZE - 1)`. This matches the natural
//! "voxel slots tile the integer line, chunks tile groups of
//! slots" intuition; using truncating division would put voxel
//! `-1` in chunk `0` at position `-1`, splitting the chunk-0 /
//! chunk-(-1) boundary inconsistently.
//!
//! All conversions go through these helpers — risk R5 in
//! `PORTING-SCENE.md` calls out the f64↔i32 boundary as a common
//! off-by-one source, so concentrating the casts here lets the
//! property tests pin them.

// `CHUNK_SIZE_XY` (128) and `CHUNK_SIZE_Z` (256) both fit in
// i32::MAX/2 with room to spare, so all `as i32` casts in this
// module are exact. Same for `voxel_in_chunk: UVec3` components,
// which are bounded by those constants and only cast for
// arithmetic on signed `IVec3`.
#![allow(clippy::cast_possible_wrap)]

use glam::{DVec3, IVec3, UVec3, Vec3};

use crate::{GridTransform, CHUNK_SIZE_XY, CHUNK_SIZE_Z};

/// Decomposition of a grid-local position into discrete chunk +
/// voxel + sub-voxel coordinates.
///
/// `chunk + voxel` reconstructs the integer voxel coordinate via
/// [`voxel_global`]; adding `fract` (range `[0, 1)` per axis) and
/// the rotated origin gets back to the original world position
/// via [`grid_local_to_world`]. `fract` is f32 because per-chunk
/// ray math is f32 throughout — keeping the boundary cast here
/// means downstream code doesn't repeat it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridLocalPos {
    /// Chunk index in grid-local space. Signed because chunks
    /// can extend in either direction from the grid origin.
    pub chunk: IVec3,
    /// Voxel offset within `chunk`. `voxel.x, voxel.y` are in
    /// `[0, CHUNK_SIZE_XY)`; `voxel.z` is in `[0, CHUNK_SIZE_Z)`.
    pub voxel: UVec3,
    /// Sub-voxel position within the voxel cell, `[0, 1)` per
    /// axis. Cast to f32 because per-chunk ray math is f32.
    pub fract: Vec3,
}

/// Per-axis chunk size as an [`IVec3`]. Used as the divisor for
/// `div_euclid` / `rem_euclid` when splitting a grid-local voxel
/// coordinate into chunk + voxel-in-chunk.
#[inline]
fn chunk_size_ivec3() -> IVec3 {
    // `as i32` is exact: both constants are well under `i32::MAX`.
    #[allow(clippy::cast_possible_wrap)]
    IVec3::new(
        CHUNK_SIZE_XY as i32,
        CHUNK_SIZE_XY as i32,
        CHUNK_SIZE_Z as i32,
    )
}

/// Split a grid-local voxel coordinate into `(chunk, voxel-in-chunk)`.
///
/// Uses [`IVec3::div_euclid`] / [`IVec3::rem_euclid`] so negative
/// voxel coordinates round toward `-∞`: voxel `-1` decomposes to
/// `(chunk = -1, voxel = CHUNK_SIZE - 1)`, not `(0, -1)`. The
/// returned `voxel-in-chunk` is always non-negative, so casting
/// each component `as u32` is safe.
#[must_use]
pub fn voxel_split(voxel: IVec3) -> (IVec3, UVec3) {
    let cs = chunk_size_ivec3();
    let chunk = voxel.div_euclid(cs);
    let in_chunk_i = voxel.rem_euclid(cs);
    // rem_euclid postcondition: each component in [0, divisor).
    // Cast is safe.
    #[allow(clippy::cast_sign_loss)]
    let in_chunk = UVec3::new(
        in_chunk_i.x as u32,
        in_chunk_i.y as u32,
        in_chunk_i.z as u32,
    );
    (chunk, in_chunk)
}

/// Inverse of [`voxel_split`]: combine a chunk index and a
/// voxel-in-chunk offset back into a grid-local voxel coordinate.
///
/// Caller is responsible for the `voxel_in_chunk` invariant
/// (`x, y < CHUNK_SIZE_XY` and `z < CHUNK_SIZE_Z`); a stray
/// out-of-range value just shifts the result by a chunk's worth.
/// Debug builds panic via the [`debug_assert!`]s.
#[must_use]
pub fn voxel_global(chunk: IVec3, voxel_in_chunk: UVec3) -> IVec3 {
    debug_assert!(voxel_in_chunk.x < CHUNK_SIZE_XY, "voxel.x out of range");
    debug_assert!(voxel_in_chunk.y < CHUNK_SIZE_XY, "voxel.y out of range");
    debug_assert!(voxel_in_chunk.z < CHUNK_SIZE_Z, "voxel.z out of range");
    let cs = chunk_size_ivec3();
    // `as i32` is safe: voxel_in_chunk components are < CHUNK_SIZE_*,
    // which fit comfortably in i32.
    #[allow(clippy::cast_possible_wrap)]
    let in_chunk_i = IVec3::new(
        voxel_in_chunk.x as i32,
        voxel_in_chunk.y as i32,
        voxel_in_chunk.z as i32,
    );
    chunk * cs + in_chunk_i
}

/// Project a world-space position into the grid-local frame and
/// decompose into chunk + voxel + sub-voxel.
///
/// Steps:
/// 1. Translate by `-transform.origin`.
/// 2. Rotate by `transform.rotation.inverse()` (back to grid-local
///    axes). Identity rotation (axis-aligned grid) collapses this
///    to a no-op.
/// 3. Floor each component to the integer voxel; keep the
///    remainder as the sub-voxel fractional position (cast to
///    f32 at the boundary).
/// 4. Split the integer voxel into chunk + voxel-in-chunk via
///    [`voxel_split`].
///
/// The full pipeline is the canonical world↔grid handoff — risk
/// R5 in `PORTING-SCENE.md`. Round-tripping with
/// [`grid_local_to_world`] reconstructs the original world point
/// up to f32 precision in the fractional component (sub-millimetre
/// at typical voxel scales).
#[must_use]
pub fn world_to_grid_local(world_pos: DVec3, transform: &GridTransform) -> GridLocalPos {
    // SC — un-rotate, un-translate, then divide by the grid's world
    // units per voxel to land in voxel coordinates (`1.0` = the classic
    // 1:1, byte-identical to the pre-SC path).
    let local_d = (transform.rotation.inverse() * (world_pos - transform.origin))
        / transform.voxel_world_size;
    let voxel_d = local_d.floor();
    // After `.floor()` the components are integer-valued; truncating
    // to i32 is equivalent to flooring. Out-of-range coords saturate,
    // which is acceptable behaviour for a degenerate input — the
    // caller never sees a wrap. f32 fractional cast is lossy by
    // design; sub-mm precision at 1-unit voxel scale.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss
    )]
    let voxel = IVec3::new(voxel_d.x as i32, voxel_d.y as i32, voxel_d.z as i32);
    #[allow(clippy::cast_possible_truncation)]
    let fract = (local_d - voxel_d).as_vec3();
    let (chunk, in_chunk) = voxel_split(voxel);
    GridLocalPos {
        chunk,
        voxel: in_chunk,
        fract,
    }
}

/// Inverse of [`world_to_grid_local`]: reconstruct the world-space
/// position of a grid-local chunk + voxel + sub-voxel.
///
/// Round-trips with [`world_to_grid_local`] up to f32 precision in
/// the fractional component (the `fract: Vec3` cast is the lossy
/// step).
#[must_use]
pub fn grid_local_to_world(
    chunk: IVec3,
    voxel_in_chunk: UVec3,
    fract: Vec3,
    transform: &GridTransform,
) -> DVec3 {
    let voxel = voxel_global(chunk, voxel_in_chunk);
    // SC — voxel coordinates → world: scale by the grid's world units
    // per voxel BEFORE rotating + translating.
    let local = (voxel.as_dvec3() + fract.as_dvec3()) * transform.voxel_world_size;
    transform.origin + transform.rotation * local
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DQuat;

    // ---- voxel_split / voxel_global round-trip ----

    #[test]
    fn voxel_split_origin() {
        let (c, v) = voxel_split(IVec3::ZERO);
        assert_eq!(c, IVec3::ZERO);
        assert_eq!(v, UVec3::ZERO);
    }

    #[test]
    fn voxel_split_at_chunk_boundary_positive() {
        // (CHUNK_SIZE_XY, 0, 0) is the first voxel of chunk (1,0,0).
        let (c, v) = voxel_split(IVec3::new(CHUNK_SIZE_XY as i32, 0, 0));
        assert_eq!(c, IVec3::new(1, 0, 0));
        assert_eq!(v, UVec3::new(0, 0, 0));
    }

    #[test]
    fn voxel_split_at_chunk_boundary_minus_one() {
        // (-1, 0, 0) is the last voxel of chunk (-1, 0, 0).
        let (c, v) = voxel_split(IVec3::new(-1, 0, 0));
        assert_eq!(c, IVec3::new(-1, 0, 0));
        assert_eq!(v, UVec3::new(CHUNK_SIZE_XY - 1, 0, 0));
    }

    #[test]
    fn voxel_split_z_axis_uses_z_chunk_size() {
        // z = 256 should fall into chunk (0, 0, 1) at z-offset 0,
        // not into a 128-strided chunk like XY.
        let (c, v) = voxel_split(IVec3::new(0, 0, CHUNK_SIZE_Z as i32));
        assert_eq!(c, IVec3::new(0, 0, 1));
        assert_eq!(v, UVec3::new(0, 0, 0));
    }

    #[test]
    fn voxel_global_inverts_voxel_split() {
        // Sample chunks/voxels covering positive, negative,
        // boundary, and interior cases. Each should round-trip.
        let cases = [
            IVec3::ZERO,
            IVec3::new(1, 1, 1),
            IVec3::new(-1, 0, 0),
            IVec3::new(0, -1, 0),
            IVec3::new(0, 0, -1),
            IVec3::new(CHUNK_SIZE_XY as i32, 0, 0),
            IVec3::new(0, CHUNK_SIZE_XY as i32, 0),
            IVec3::new(0, 0, CHUNK_SIZE_Z as i32),
            IVec3::new(-(CHUNK_SIZE_XY as i32) - 5, 7, 33),
            IVec3::new(127, 128, 256),
            IVec3::new(1_000_000, -1_000_000, 500),
        ];
        for v in cases {
            let (c, in_chunk) = voxel_split(v);
            assert_eq!(
                voxel_global(c, in_chunk),
                v,
                "round trip failed for v={v:?} → (c={c:?}, in_chunk={in_chunk:?})"
            );
        }
    }

    #[test]
    fn voxel_split_in_chunk_always_in_range() {
        // Brute-force a 3-chunk-wide range around the origin so we
        // hit positive, negative, and boundary voxels on all axes.
        for vx in -200i32..200 {
            for vy in -200i32..200 {
                for vz in -300i32..300 {
                    let (_, u) = voxel_split(IVec3::new(vx, vy, vz));
                    assert!(u.x < CHUNK_SIZE_XY, "x={} out of range", u.x);
                    assert!(u.y < CHUNK_SIZE_XY, "y={} out of range", u.y);
                    assert!(u.z < CHUNK_SIZE_Z, "z={} out of range", u.z);
                }
            }
        }
    }

    // ---- world_to_grid_local: identity transform ----

    #[test]
    fn world_to_local_identity_at_origin() {
        let t = GridTransform::identity();
        let p = world_to_grid_local(DVec3::ZERO, &t);
        assert_eq!(p.chunk, IVec3::ZERO);
        assert_eq!(p.voxel, UVec3::ZERO);
        assert!(p.fract.abs_diff_eq(Vec3::ZERO, 1e-6));
    }

    #[test]
    fn world_to_local_identity_at_voxel_centre() {
        // (1.5, 2.5, 3.5) is the centre of voxel (1, 2, 3) — chunk
        // (0, 0, 0), voxel-in-chunk (1, 2, 3), fractional (.5, .5, .5).
        let t = GridTransform::identity();
        let p = world_to_grid_local(DVec3::new(1.5, 2.5, 3.5), &t);
        assert_eq!(p.chunk, IVec3::ZERO);
        assert_eq!(p.voxel, UVec3::new(1, 2, 3));
        assert!(p.fract.abs_diff_eq(Vec3::splat(0.5), 1e-6));
    }

    #[test]
    fn world_to_local_negative_world_pos() {
        // (-0.5, 0, 0) sits in voxel (-1, 0, 0) = chunk (-1, 0, 0)
        // at position CHUNK_SIZE_XY - 1, fractional .5.
        let t = GridTransform::identity();
        let p = world_to_grid_local(DVec3::new(-0.5, 0.0, 0.0), &t);
        assert_eq!(p.chunk, IVec3::new(-1, 0, 0));
        assert_eq!(p.voxel, UVec3::new(CHUNK_SIZE_XY - 1, 0, 0));
        assert!(p.fract.abs_diff_eq(Vec3::new(0.5, 0.0, 0.0), 1e-6));
    }

    #[test]
    fn world_to_local_at_chunk_boundary() {
        // World x = 128.0 == CHUNK_SIZE_XY: starts a new chunk.
        let t = GridTransform::identity();
        let p = world_to_grid_local(DVec3::new(f64::from(CHUNK_SIZE_XY), 0.0, 0.0), &t);
        assert_eq!(p.chunk, IVec3::new(1, 0, 0));
        assert_eq!(p.voxel, UVec3::ZERO);
        assert!(p.fract.abs_diff_eq(Vec3::ZERO, 1e-6));
    }

    // ---- world_to_local: translated grid ----

    #[test]
    fn translation_offsets_world_position() {
        // Grid placed at world (1000, 2000, 3000); world point
        // (1000.5, 2000.5, 3000.5) is grid-local (0.5, 0.5, 0.5)
        // → voxel (0, 0, 0), fractional (0.5, 0.5, 0.5).
        let t = GridTransform::at(DVec3::new(1000.0, 2000.0, 3000.0));
        let p = world_to_grid_local(DVec3::new(1000.5, 2000.5, 3000.5), &t);
        assert_eq!(p.chunk, IVec3::ZERO);
        assert_eq!(p.voxel, UVec3::ZERO);
        assert!(p.fract.abs_diff_eq(Vec3::splat(0.5), 1e-6));
    }

    // ---- world_to_local: rotated grid ----

    #[test]
    fn rotation_90_z_swaps_x_and_y() {
        // 90° rotation about +z maps grid-local +x to world +y.
        // World point (0, 5, 0) is grid-local (5, 0, 0): rotation
        // inverse takes world +y back to grid-local +x.
        let t = GridTransform {
            origin: DVec3::ZERO,
            rotation: DQuat::from_rotation_z(std::f64::consts::FRAC_PI_2),
            voxel_world_size: 1.0,
        };
        let p = world_to_grid_local(DVec3::new(0.0, 5.5, 0.0), &t);
        assert_eq!(p.chunk, IVec3::ZERO);
        assert_eq!(p.voxel, UVec3::new(5, 0, 0));
        // Quat math has tiny rounding error — allow 1e-6.
        assert!(
            p.fract.abs_diff_eq(Vec3::new(0.5, 0.0, 0.0), 1e-5),
            "fract={:?} expected ~(0.5, 0, 0)",
            p.fract
        );
    }

    // ---- grid_local_to_world: round-trip ----

    #[test]
    fn world_local_world_round_trip_identity() {
        let t = GridTransform::identity();
        let world = DVec3::new(12.25, -7.75, 200.5);
        let p = world_to_grid_local(world, &t);
        let back = grid_local_to_world(p.chunk, p.voxel, p.fract, &t);
        assert!(
            back.abs_diff_eq(world, 1e-5),
            "back={back:?} world={world:?}"
        );
    }

    #[test]
    fn world_local_world_round_trip_translated() {
        let t = GridTransform::at(DVec3::new(500.0, -250.0, 100.0));
        let world = DVec3::new(512.25, -260.5, 109.75);
        let p = world_to_grid_local(world, &t);
        let back = grid_local_to_world(p.chunk, p.voxel, p.fract, &t);
        assert!(
            back.abs_diff_eq(world, 1e-5),
            "back={back:?} world={world:?}"
        );
    }

    #[test]
    fn world_local_world_round_trip_rotated() {
        let t = GridTransform {
            origin: DVec3::new(10.0, 20.0, 30.0),
            rotation: DQuat::from_rotation_z(0.5).normalize(),
            voxel_world_size: 1.0,
        };
        // Sample several points to exercise the rotation math.
        let samples = [
            DVec3::new(11.5, 22.5, 33.5),
            DVec3::new(10.0, 20.0, 30.0),
            DVec3::new(9.0, 19.0, 29.0),
        ];
        for world in samples {
            let p = world_to_grid_local(world, &t);
            let back = grid_local_to_world(p.chunk, p.voxel, p.fract, &t);
            assert!(
                back.abs_diff_eq(world, 1e-5),
                "back={back:?} world={world:?}"
            );
        }
    }

    // ---- S5.1: parameterised round-trip over multiple rotations ----

    /// Sweep a grid of rotations × world positions and confirm the
    /// `world_to_grid_local` ∘ `grid_local_to_world` round-trip
    /// closes within f32-fract precision (cast from f64 → f32 in
    /// [`GridLocalPos::fract`] is the lossy step). Per PORTING-SCENE.md
    /// risk R5, this is the canonical f64↔i32 boundary check.
    ///
    /// Tolerance is `1e-5` — empirically matches the worst-case
    /// f32 precision of a `fract` cast at unit voxel scale (≈ 1.2e-7
    /// per component, amplified slightly by the round-trip's
    /// re-rotation).
    #[test]
    fn world_local_world_round_trip_rotation_sweep() {
        // Rotations: identity (sanity), three axis-aligned 90°s, a
        // shallow tilt, a 45° composite, and a fully arbitrary
        // axis/angle. Cover identity → near-singular → arbitrary.
        let rotations = [
            ("identity", DQuat::IDENTITY),
            (
                "90deg-z",
                DQuat::from_rotation_z(std::f64::consts::FRAC_PI_2),
            ),
            (
                "90deg-y",
                DQuat::from_rotation_y(std::f64::consts::FRAC_PI_2),
            ),
            (
                "90deg-x",
                DQuat::from_rotation_x(std::f64::consts::FRAC_PI_2),
            ),
            ("180deg-z exact", DQuat::from_xyzw(0.0, 0.0, 1.0, 0.0)),
            (
                "45deg-z",
                DQuat::from_rotation_z(std::f64::consts::FRAC_PI_4),
            ),
            (
                "tilted 0.7rad",
                DQuat::from_axis_angle(DVec3::new(0.3, 0.8, 0.5).normalize(), 0.7),
            ),
            (
                "yaw+pitch+roll composite",
                DQuat::from_rotation_y(0.4)
                    * DQuat::from_rotation_x(0.3)
                    * DQuat::from_rotation_z(0.2),
            ),
        ];
        // World positions: origin, positive interior, negative
        // quadrant, near chunk boundaries (positive + negative), and
        // far-from-origin so the f64 → f32 fract cast loses no
        // additional precision.
        let world_positions = [
            DVec3::ZERO,
            DVec3::new(1.5, 2.5, 3.5),
            DVec3::new(-1.5, -2.5, -3.5),
            DVec3::new(f64::from(CHUNK_SIZE_XY) - 0.01, 0.5, 0.5),
            DVec3::new(-f64::from(CHUNK_SIZE_XY) - 0.01, 0.5, 0.5),
            DVec3::new(500.25, -250.75, 100.125),
        ];
        // Grid origins: at world origin (canonical case) and a
        // non-trivial offset to verify the translation interacts
        // correctly with the rotation.
        let grid_origins = [DVec3::ZERO, DVec3::new(1000.0, -500.0, 200.0)];

        for (rot_name, rotation) in rotations {
            for grid_origin in grid_origins {
                let t = GridTransform {
                    origin: grid_origin,
                    rotation,
                    voxel_world_size: 1.0,
                };
                for world in world_positions {
                    let p = world_to_grid_local(world, &t);
                    let back = grid_local_to_world(p.chunk, p.voxel, p.fract, &t);
                    assert!(
                        back.abs_diff_eq(world, 1e-5),
                        "rotation={rot_name} origin={grid_origin:?} world={world:?} back={back:?}"
                    );
                }
            }
        }
    }

    // ---- SC: per-grid voxel scale ----

    #[test]
    fn scaled_grid_maps_world_to_smaller_voxel_index() {
        // voxel_world_size 2.0 ⇒ each voxel is 2 world units, so world
        // (10.5, 4.5, 6.5) lands in voxel (5, 2, 3).
        let t = GridTransform::at_scale(DVec3::ZERO, 2.0);
        let p = world_to_grid_local(DVec3::new(10.5, 4.5, 6.5), &t);
        assert_eq!(p.chunk, IVec3::ZERO);
        assert_eq!(p.voxel, UVec3::new(5, 2, 3));
        // 0.5 world into a 2-unit voxel = 0.25 of the cell.
        assert!(
            p.fract.abs_diff_eq(Vec3::splat(0.25), 1e-6),
            "fract={:?}",
            p.fract
        );
    }

    #[test]
    fn scaled_grid_round_trips() {
        // world → local → world with a scaled + translated + rotated
        // grid must reconstruct the original world point.
        for vws in [0.25, 1.0, 2.0, 4.0] {
            let t = GridTransform {
                origin: DVec3::new(100.0, -50.0, 30.0),
                rotation: DQuat::from_rotation_z(0.6).normalize(),
                voxel_world_size: vws,
            };
            for world in [
                DVec3::new(101.5, -48.25, 33.75),
                DVec3::new(100.0, -50.0, 30.0),
                DVec3::new(140.0, -20.0, 60.0),
            ] {
                let p = world_to_grid_local(world, &t);
                let back = grid_local_to_world(p.chunk, p.voxel, p.fract, &t);
                assert!(
                    back.abs_diff_eq(world, 1e-4),
                    "vws={vws} world={world:?} back={back:?}"
                );
            }
        }
    }

    /// Spot-check that a non-identity rotation actually places the
    /// voxel decomposition in a different chunk than the identity
    /// case would — guards against a regression where rotation is
    /// silently dropped (e.g., a missed `transform.rotation` field
    /// read). For 90°-Z about the world origin, world point
    /// `(0, 5, 0)` lives in grid-local chunk (0, 0, 0) voxel
    /// `(5, 0, 0)`, NOT the `(0, 5, 0)` it would map to under
    /// identity.
    #[test]
    fn rotated_world_point_lands_in_rotated_voxel() {
        let t = GridTransform {
            origin: DVec3::ZERO,
            rotation: DQuat::from_rotation_z(std::f64::consts::FRAC_PI_2),
            voxel_world_size: 1.0,
        };
        let p_rotated = world_to_grid_local(DVec3::new(0.0, 5.5, 0.0), &t);
        let p_identity = world_to_grid_local(DVec3::new(0.0, 5.5, 0.0), &GridTransform::identity());
        assert_ne!(
            p_rotated.voxel, p_identity.voxel,
            "rotated voxel ({:?}) coincidentally equals identity voxel ({:?}) — rotation may have been dropped",
            p_rotated.voxel,
            p_identity.voxel,
        );
        assert_eq!(p_rotated.voxel, UVec3::new(5, 0, 0));
        assert_eq!(p_identity.voxel, UVec3::new(0, 5, 0));
    }
}
