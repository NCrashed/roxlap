//! Collision query layer (stage CC.0 — see
//! `docs/porting/PORTING-CONTROLLER.md`).
//!
//! World-space box-vs-voxel overlap tests over a whole [`Scene`],
//! promoted from the three demo copies of the fly-camera collision
//! hack (scene-demo `collision.rs`, cave-demo, cave-web) so the CC
//! character controller — and the demos themselves — share one
//! implementation with the demos' hard-won lessons pinned as unit
//! tests here:
//!
//! - Solidity comes from [`roxlap_core::world_query::getcube`]:
//!   [`Cube::Color`](roxlap_core::world_query::Cube::Color) *and*
//!   [`Cube::UnexposedSolid`](roxlap_core::world_query::Cube::UnexposedSolid)
//!   block (slab interiors are solid material),
//!   [`Cube::Air`](roxlap_core::world_query::Cube::Air) does not.
//! - The voxlap bedrock placeholder at chunk-local
//!   `z = CHUNK_SIZE_Z - 1` is a *policy*, not a fact —
//!   [`Solidity::bedrock_blocks`], default `false` to match the
//!   demos' `treat_z_max_as_air` rendering (else an invisible wall
//!   appears at every grid's bottom plane).
//! - Positions outside every grid's footprint are air, so a body can
//!   move past a grid without hitting invisible bounds.
//!
//! Grid placement: **axis-aligned grids are probed cell-exactly**
//! (the box's floor-range of voxel cells); **rotated grids are
//! probed conservatively** — the world box's 8 corners are
//! transformed into grid-local space and their local AABB is probed,
//! which blocks slightly early near rotated geometry but never
//! leaks. An exact OBB-vs-voxel test is out of scope for a
//! controller-grade query.

use glam::{DQuat, DVec3, IVec3};
use roxlap_core::world_query::{getcube, Cube};

use crate::{voxel_split, Grid, Scene, CHUNK_SIZE_Z};

/// What counts as solid for a collision probe.
#[derive(Debug, Clone, Copy, Default)]
pub struct Solidity {
    /// Does the voxlap bedrock placeholder (chunk-local
    /// `z = CHUNK_SIZE_Z - 1`) block? Default `false`, matching the
    /// demos' `treat_z_max_as_air` rendering. Set `true` for worlds
    /// whose bottom plane is genuinely solid ground **rendered as
    /// such** — collision and rendering must agree, or the player
    /// either hits an invisible wall or falls through a visible
    /// floor.
    pub bedrock_blocks: bool,
    /// The material hook (CC.4): a colour veto for *visible* voxels.
    /// `Some(f)` makes a voxel whose colour `f` approves passable —
    /// water, foliage, ladders — while everything else still blocks;
    /// `None` (default) blocks on any voxel. A plain `fn` (not a
    /// closure) so `Solidity` stays `Copy`; key it the way material
    /// maps key their colours (compare `.rgb_part()`, ignore the
    /// brightness byte — lighting bakes rewrite it).
    ///
    /// Limitation, inherent to the voxlap slab format: hidden run
    /// interiors ([`Cube::UnexposedSolid`]) store no colour and
    /// always block. Rule of thumb: a
    /// pass-through wall/curtain works up to **2 voxels thick**
    /// (every voxel still has an exposed face and carries its
    /// colour); at 3+ a colourless core appears and blocks. Filled
    /// pools are wade-through only near their visible surface. Also
    /// keep such geometry at least 1 voxel away from chunk-local
    /// x/y = 0 and the chunk's far edge: the slab encoder treats
    /// out-of-chunk neighbours as solid, so edge voxels lose their
    /// side colours (they render, but classify as UnexposedSolid —
    /// both facts are pinned in this module's tests).
    pub passable: Option<fn(crate::VoxColor) -> bool>,
}

impl Solidity {
    /// Does this probe result block, under the current policy?
    fn cube_blocks(self, cube: Cube) -> bool {
        match cube {
            Cube::Air => false,
            Cube::UnexposedSolid => true,
            Cube::Color(c) => !self
                .passable
                .is_some_and(|passes| passes(crate::VoxColor(c))),
        }
    }
}

/// `true` if the axis-aligned world-space box `[min, max]` overlaps
/// any solid voxel of any grid in `scene`.
///
/// `min`/`max` are corner positions with `min[i] <= max[i]`; a face
/// exactly on a voxel-cell boundary counts as overlapping the cell
/// on both sides (conservative, matching the demos' `±radius`
/// probes).
#[must_use]
pub fn box_overlaps_solid(scene: &Scene, min: DVec3, max: DVec3, solidity: Solidity) -> bool {
    scene
        .grids()
        .any(|(_id, grid)| grid_box_overlaps_solid(grid, min, max, solidity))
}

/// Point form of [`box_overlaps_solid`]: `true` if the voxel cell
/// containing the world-space point `p` is solid in any grid.
#[must_use]
pub fn point_overlaps_solid(scene: &Scene, p: DVec3, solidity: Solidity) -> bool {
    box_overlaps_solid(scene, p, p, solidity)
}

/// Single-grid form of [`box_overlaps_solid`] — the building block
/// the scene-level test folds over, public for hosts that manage
/// their own grid (the cave demos collide against one grid only).
#[must_use]
pub fn grid_box_overlaps_solid(grid: &Grid, min: DVec3, max: DVec3, solidity: Solidity) -> bool {
    // World box → grid-local box.
    let (lmin, lmax) = if grid.transform.rotation == DQuat::IDENTITY {
        // Axis-aligned: translate — the probe below is cell-exact.
        (min - grid.transform.origin, max - grid.transform.origin)
    } else {
        // Rotated: local AABB of the 8 transformed corners —
        // conservative (a fat OBB approximation), never leaky.
        let inv = grid.transform.rotation.inverse();
        let mut lmin = DVec3::INFINITY;
        let mut lmax = DVec3::NEG_INFINITY;
        for corner in 0..8 {
            let world = DVec3::new(
                if corner & 1 == 0 { min.x } else { max.x },
                if corner & 2 == 0 { min.y } else { max.y },
                if corner & 4 == 0 { min.z } else { max.z },
            );
            let local = inv * (world - grid.transform.origin);
            lmin = lmin.min(local);
            lmax = lmax.max(local);
        }
        (lmin, lmax)
    };

    // Every voxel cell the local box touches. `floor` of both ends:
    // a face exactly on a cell boundary includes the far cell.
    #[allow(clippy::cast_possible_truncation)]
    let (lo, hi) = (lmin.floor().as_ivec3(), lmax.floor().as_ivec3());

    let bedrock_z = CHUNK_SIZE_Z - 1;
    for vz in lo.z..=hi.z {
        for vy in lo.y..=hi.y {
            for vx in lo.x..=hi.x {
                let (chunk_idx, in_chunk) = voxel_split(IVec3::new(vx, vy, vz));
                if !solidity.bedrock_blocks && in_chunk.z == bedrock_z {
                    // Bedrock placeholder treated as air (fn docs).
                    continue;
                }
                let Some(chunk) = grid.chunk(chunk_idx) else {
                    // Implicit-air chunk.
                    continue;
                };
                // rem_euclid postcondition: components in [0, chunk
                // size), far below i32::MAX.
                #[allow(clippy::cast_possible_wrap)]
                let cube = getcube(
                    &chunk.data,
                    &chunk.column_offset,
                    chunk.vsid,
                    in_chunk.x as i32,
                    in_chunk.y as i32,
                    in_chunk.z as i32,
                );
                if solidity.cube_blocks(cube) {
                    return true;
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GridTransform, VoxColor};

    const AIR_BEDROCK: Solidity = Solidity {
        bedrock_blocks: false,
        passable: None,
    };

    /// Single floating voxel at grid-local (10, 10, 50) in a grid at
    /// world (0, 0, -100) — the scene-demo `collision.rs` fixture,
    /// ported with its tests (CC.0 regression net).
    fn floating_voxel_scene() -> Scene {
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::at(DVec3::new(0.0, 0.0, -100.0)));
        let grid = scene.grid_mut(id).expect("grid present");
        grid.set_voxel(IVec3::new(10, 10, 50), Some(VoxColor(0x80_aa_bb_cc)));
        scene
    }

    fn cube_probe(scene: &Scene, world: [f64; 3], r: f64) -> bool {
        let p = DVec3::from(world);
        box_overlaps_solid(scene, p - r, p + r, AIR_BEDROCK)
    }

    #[test]
    fn visible_voxel_blocks() {
        let scene = floating_voxel_scene();
        // World (10, 10, -50) → grid-local (10, 10, 50): the voxel.
        assert!(cube_probe(&scene, [10.0, 10.0, -50.0], 0.3));
    }

    #[test]
    fn out_of_grid_position_is_air() {
        let scene = floating_voxel_scene();
        assert!(!cube_probe(&scene, [-500.0, -500.0, -50.0], 0.3));
    }

    #[test]
    fn below_isolated_floating_voxel_is_air() {
        // Sparse column: getcube past the deepest visible slab walks
        // to the air pocket below — NOT UnexposedSolid.
        let scene = floating_voxel_scene();
        assert!(!cube_probe(&scene, [10.0, 10.0, 0.0], 0.3));
    }

    #[test]
    fn slab_interior_unexposed_solid_blocks() {
        // A thick set_rect slab: its hidden interior reports
        // Cube::UnexposedSolid, which must block (the scene-demo
        // saucer-interior lesson — treating it as air lets the body
        // inside solid material).
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::identity());
        let grid = scene.grid_mut(id).expect("grid present");
        grid.set_rect(
            IVec3::new(0, 0, 50),
            IVec3::new(20, 20, 80),
            Some(VoxColor(0x80_66_77_88)),
        );
        assert!(cube_probe(&scene, [10.0, 10.0, 65.0], 0.3));
    }

    #[test]
    fn bedrock_placeholder_is_policy() {
        // Any edit materialises a chunk whose columns keep the voxlap
        // bedrock placeholder at chunk-local z = CHUNK_SIZE_Z - 1.
        // Probe that plane far from the real voxel: air by default
        // (the scene-demo invisible-wall fix), solid on request.
        let scene = floating_voxel_scene();
        let at_bedrock = [100.0, 100.0, -100.0 + f64::from(CHUNK_SIZE_Z) - 0.5];
        assert!(!cube_probe(&scene, at_bedrock, 0.3));
        let p = DVec3::from(at_bedrock);
        assert!(box_overlaps_solid(
            &scene,
            p - 0.3,
            p + 0.3,
            Solidity {
                bedrock_blocks: true,
                passable: None,
            },
        ));
    }

    #[test]
    fn rotated_grid_blocks_conservatively() {
        // Grid rotated 45° about z, one voxel at local (10, 10, 50).
        // Probing the voxel's world-space centre must block; probing
        // far away must not. (Exactness near rotated geometry is not
        // promised — the corner-AABB is conservative.)
        let mut scene = Scene::new();
        let rot = DQuat::from_rotation_z(std::f64::consts::FRAC_PI_4);
        let id = scene.add_grid(GridTransform {
            origin: DVec3::new(200.0, 0.0, 0.0),
            rotation: rot,
            voxel_world_size: 1.0,
        });
        let grid = scene.grid_mut(id).expect("grid present");
        grid.set_voxel(IVec3::new(10, 10, 50), Some(VoxColor(0x80_11_22_33)));

        let centre_world = DVec3::new(200.0, 0.0, 0.0) + rot * DVec3::new(10.5, 10.5, 50.5);
        assert!(box_overlaps_solid(
            &scene,
            centre_world - 0.1,
            centre_world + 0.1,
            AIR_BEDROCK,
        ));
        let far = centre_world + DVec3::new(40.0, 40.0, 0.0);
        assert!(!box_overlaps_solid(
            &scene,
            far - 0.1,
            far + 0.1,
            AIR_BEDROCK
        ));
    }

    /// CC.4 — the material hook. Water-coloured voxels pass, any
    /// other colour still blocks, and hidden run interiors block
    /// regardless (the slab format stores no colour for them).
    #[test]
    fn passable_colour_veto() {
        const WATER: VoxColor = VoxColor(0x80_20_60_c0);
        const GLASS: VoxColor = VoxColor(0x80_50_c0_e0);
        fn water_passes(c: VoxColor) -> bool {
            c.rgb_part() == WATER.rgb_part()
        }
        let veto = Solidity {
            bedrock_blocks: false,
            passable: Some(water_passes),
        };

        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::identity());
        let grid = scene.grid_mut(id).expect("grid present");
        // A water curtain and a glass wall, 1 voxel thin (fully
        // visible), plus a THICK water block whose interior is
        // UnexposedSolid.
        grid.set_rect(IVec3::new(10, 0, 40), IVec3::new(10, 20, 60), Some(WATER));
        grid.set_rect(IVec3::new(20, 0, 40), IVec3::new(20, 20, 60), Some(GLASS));
        grid.set_rect(IVec3::new(30, 0, 40), IVec3::new(40, 20, 60), Some(WATER));

        let probe = |x: f64, z: f64, s: Solidity| {
            let p = DVec3::new(x, 10.0, z);
            box_overlaps_solid(&scene, p - 0.3, p + 0.3, s)
        };
        // Thin water: blocked without the veto, passable with it.
        assert!(probe(10.5, 50.0, AIR_BEDROCK));
        assert!(!probe(10.5, 50.0, veto));
        // Glass: blocked either way — the veto is colour-selective.
        assert!(probe(20.5, 50.0, veto));
        // Thick water body: surface layer passes…
        assert!(!probe(30.5, 50.0, veto));
        // …but the hidden interior has no colour and still blocks
        // (brightness byte differences must not matter either — the
        // veto compares rgb_part, and bakes rewrite the high byte).
        assert!(probe(35.5, 50.0, veto));
    }

    #[test]
    fn point_probe_matches_degenerate_box() {
        let scene = floating_voxel_scene();
        assert!(point_overlaps_solid(
            &scene,
            DVec3::new(10.5, 10.5, -49.5),
            AIR_BEDROCK,
        ));
        assert!(!point_overlaps_solid(
            &scene,
            DVec3::new(10.5, 10.5, -55.5),
            AIR_BEDROCK,
        ));
    }

    /// The passable-veto geometry rule of thumb, pinned: a wall 1 or
    /// 2 voxels thick is fully side-exposed (every voxel carries its
    /// colour — the veto sees it all), while 3+ has a hidden core of
    /// colourless `UnexposedSolid` that blocks regardless of the
    /// veto. Build pass-through curtains at most 2 voxels thick.
    #[test]
    fn passable_walls_must_be_at_most_two_voxels_thick() {
        const WATER: VoxColor = VoxColor(0x80_30_60_c8);
        fn water_passes(c: VoxColor) -> bool {
            c.rgb_part() == WATER.rgb_part()
        }
        let veto = Solidity {
            bedrock_blocks: false,
            passable: Some(water_passes),
        };
        let crossing_blocked = |thick: i32| {
            let mut scene = Scene::new();
            let id = scene.add_grid(GridTransform::identity());
            let g = scene.grid_mut(id).expect("grid");
            g.set_rect(
                IVec3::new(0, 10, 0),
                IVec3::new(50, 10 + thick - 1, 50),
                Some(WATER),
            );
            // A CharacterBody-sized box mid-wall, mid-height.
            let p = DVec3::new(25.0, 10.0 + f64::from(thick) * 0.5, 25.0);
            box_overlaps_solid(
                &scene,
                p - DVec3::new(0.4, 0.4, 0.9),
                p + DVec3::new(0.4, 0.4, 0.9),
                veto,
            )
        };
        assert!(!crossing_blocked(1));
        assert!(!crossing_blocked(2));
        assert!(crossing_blocked(3), "3-thick has a hidden core");
        assert!(crossing_blocked(4));

        // Second format fact: a wall hugging grid-local y = 0 (the
        // chunk boundary) blocks even at 2 voxels thick — the slab
        // encoder treats the out-of-chunk neighbour as solid, so the
        // edge layer's side colours are never stored (it RENDERS,
        // but classifies as UnexposedSolid). Keep pass-through
        // geometry at least 1 voxel inside the chunk.
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::identity());
        let g = scene.grid_mut(id).expect("grid");
        g.set_rect(IVec3::new(0, 0, 0), IVec3::new(50, 1, 50), Some(WATER));
        let p = DVec3::new(25.0, 1.0, 25.0);
        assert!(
            box_overlaps_solid(
                &scene,
                p - DVec3::new(0.4, 0.4, 0.9),
                p + DVec3::new(0.4, 0.4, 0.9),
                veto,
            ),
            "chunk-edge wall: edge layer has no stored colour"
        );
    }
}
