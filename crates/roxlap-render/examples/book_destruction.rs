//! Companion example for the book's "Destruction" chapter
//! (`docs/book/src/destruction.md`) — the chapter pulls its snippets
//! from here via `// ANCHOR:` markers, so everything it shows compiles.
//!
//! This exercises the crumble **simulation** only (no window, no
//! renderer): it builds a stone cantilever, shoots the beam off at its
//! root, and prints what the destruction pipeline does — the islands
//! detection finds, the fragments they fracture into, the fall, and
//! the burst sites the shatter scatters on impact. Run it anywhere:
//!
//! ```sh
//! cargo run -p roxlap-render --example book_destruction
//! ```
//!
//! Keep the anchors when editing; `docs/book/check-anchors.sh` (run by
//! the CI `book` job) goes red if one disappears.

use glam::IVec3;
use roxlap_render::DebrisSystem;
use roxlap_scene::islands::{detect_islands, FracturePattern, DEFAULT_ISLAND_BUDGET};
use roxlap_scene::{BakeMode, GridTransform, Scene, VoxColor};

const STONE: VoxColor = VoxColor(0x80_8a_7a_5a);
const CRYSTAL: VoxColor = VoxColor(0x80_50_c8_e8);
const CRYSTAL_MATERIAL: u8 = 1;

fn main() {
    // ANCHOR: scene
    // A stone cantilever over a floor: a pillar anchored to the floor
    // (and through it to the format's bedrock at the column bottoms —
    // the support the detector floods toward), a beam sticking out
    // sideways, and a crystal cluster grown near the beam's tip.
    // z is DOWN: the floor is the high-z slab, the rock rises toward
    // smaller z.
    let mut scene = Scene::new();
    let grid = scene.add_grid(GridTransform::identity());
    let g = scene.grid_mut(grid).expect("grid just added");
    g.set_rect(IVec3::new(0, 0, 200), IVec3::new(63, 63, 255), Some(STONE));
    g.set_rect(
        IVec3::new(30, 30, 160),
        IVec3::new(31, 31, 199),
        Some(STONE),
    );
    g.set_rect(
        IVec3::new(32, 30, 158),
        IVec3::new(48, 31, 159),
        Some(STONE),
    );
    g.set_rect(
        IVec3::new(44, 30, 154),
        IVec3::new(47, 31, 157),
        Some(CRYSTAL),
    );
    g.bake(BakeMode::Directional);
    // ANCHOR_END: scene

    // ANCHOR: detect
    // Shoot the beam off at its root, then ask what came loose: a
    // budgeted span-BFS floods every region the carve touched;
    // whatever cannot reach bedrock-anchored ground (and stays under
    // the budget) comes back as an `Island` — its voxels, colours and
    // bounds. The pillar survives (still anchored); the beam's tip
    // and its crystal do not.
    let g = scene.grid_mut(grid).expect("grid");
    g.set_sphere(IVec3::new(33, 30, 158), 4, None);
    let islands = detect_islands(
        g,
        IVec3::new(29, 26, 154), // the carve's bbox, any corner order
        IVec3::new(37, 34, 162),
        DEFAULT_ISLAND_BUDGET,
    );
    for isl in &islands {
        println!(
            "island: {} voxels, bbox {:?}..{:?}",
            isl.voxels.len(),
            isl.bbox.0,
            isl.bbox.1
        );
    }
    // ANCHOR_END: detect

    // ANCHOR: spawn
    // Hand each island to the debris system. `spawn_island` extracts
    // the voxels from the grid (one carve + one incremental re-bake;
    // re-mip stays the caller's job, like any edit), splits them per
    // the fracture tables — rock into rounded Voronoi lumps, crystal
    // into sharp plates that keep their emissive material — and
    // registers every fragment as a falling body at the exact world
    // pose of the voxels it replaced.
    let mut debris = DebrisSystem::new();
    debris.set_fracture_patterns(
        &[(CRYSTAL.rgb_part(), CRYSTAL_MATERIAL)],
        &[
            (0, FracturePattern::Chunks { cell: 6 }),
            (CRYSTAL_MATERIAL, FracturePattern::Shards { plates: 3 }),
        ],
    );
    for isl in islands {
        debris.spawn_island(&mut scene, grid, isl, BakeMode::Directional);
    }
    println!("falling fragments: {}", debris.debris_count());
    // ANCHOR_END: spawn

    // ANCHOR: tick
    // Per frame a windowed host calls `debris.tick(renderer, &scene,
    // dt)` and shatters each drained impact into a colour-true
    // particle burst (`ParticleSystem::voxel_debris(&hit.burst_sites(),
    // …)`). The windowless half is `update` + `drain_impacts`: each
    // landed fragment reports where it hit, how fast, and the
    // world-space burst sites — one per voxel, in the voxel's own
    // colour — a particle system scatters.
    for _ in 0..600 {
        debris.update(&scene, 1.0 / 60.0);
        for hit in debris.drain_impacts() {
            println!(
                "impact at z={:.1} @ {:.1} u/s -> {} burst sites",
                hit.pos.z,
                hit.speed,
                hit.burst_sites().len(),
            );
        }
    }
    assert_eq!(debris.debris_count(), 0, "everything landed");
    // ANCHOR_END: tick
}
