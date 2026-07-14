//! Companion example for the book's "The scene graph" chapter — the
//! character-controller section pulls its snippets from here via
//! `// ANCHOR:` markers, so everything it shows compiles and its
//! assertions actually ran. Headless: no window, no renderer.
//!
//! ```sh
//! cargo run -p roxlap-scene --example book_controller
//! ```
//!
//! Keep the anchors when editing; `docs/book/check-anchors.sh` (run by
//! the CI `book` job) goes red if one disappears.

use glam::{DVec3, IVec3};
use roxlap_scene::{
    CharacterBody, CharacterDef, GridTransform, MoveMode, Scene, Solidity, VoxColor, WalkInput,
};

// ANCHOR: veto
/// Colour veto for the collision probe: voxels this colour are
/// passable (water, foliage, ladders); everything else blocks. A
/// plain `fn`, keyed like a material map — compare `.rgb_part()`,
/// the brightness byte belongs to the lighting bake.
const WATER: VoxColor = VoxColor::rgb(0x30, 0x60, 0xc8);
fn water_passes(c: VoxColor) -> bool {
    c.rgb_part() == WATER.rgb_part()
}
// ANCHOR_END: veto

fn build_world() -> Scene {
    let mut scene = Scene::new();
    let id = scene.add_grid(GridTransform::identity());
    let g = scene.grid_mut(id).expect("grid");
    // Ground slab: surface plane at z = 100 (+z is down, so the
    // slab fills downward).
    g.set_rect(
        IVec3::new(20, 20, 100),
        IVec3::new(140, 140, 120),
        Some(VoxColor::rgb(0x4d, 0x8a, 0x3a)),
    );
    // A 1-voxel ledge to step onto, and a wall to slide along.
    g.set_rect(
        IVec3::new(90, 20, 99),
        IVec3::new(140, 140, 99),
        Some(VoxColor::rgb(0x77, 0x66, 0x55)),
    );
    g.set_rect(
        IVec3::new(60, 70, 60),
        IVec3::new(62, 140, 99),
        Some(VoxColor::rgb(0x88, 0x44, 0x44)),
    );
    // A water curtain the veto lets the body wade through. Two
    // format facts (both pinned as engine tests): pass-through
    // geometry works up to 2 voxels thick — thicker walls grow a
    // colourless UnexposedSolid core no veto can classify — and
    // must sit ≥ 1 voxel inside the chunk (edge voxels lose their
    // side colours).
    g.set_rect(IVec3::new(40, 40, 80), IVec3::new(41, 60, 99), Some(WATER));
    scene
}

fn main() {
    let scene = build_world();

    // ANCHOR: setup
    // A walking body: a feet-positioned collision box, f64 world.
    // Distances are voxels, times seconds; +z is DOWN, so gravity is
    // positive and a jump impulse negative.
    let mut body = CharacterBody::new(CharacterDef {
        radius: 0.4,      // xy half-extent of the box
        height: 1.8,      // feet → head (toward -z)
        eye_height: 1.62, // feet → camera anchor
        step_up: 1.05,    // auto-step ledges up to 1 voxel
        solidity: Solidity {
            bedrock_blocks: false, // match your renderer's policy
            passable: Some(water_passes),
        },
        ..CharacterDef::default()
    });
    body.teleport(DVec3::new(50.0, 50.0, 95.0)); // FEET position
                                                 // ANCHOR_END: setup

    // ANCHOR: frame
    // Per frame: build a wish direction from input (unit-length or
    // zero — walk() clamps), call walk() EVERY frame (a zero wish is
    // what stops the body), then anchor the camera at the eye.
    let dt = 1.0 / 60.0;
    for frame in 0..480 {
        let input = WalkInput {
            wish: DVec3::new(1.0, 0.2, 0.0), // toward the ledge
            jump: frame == 120,              // one hop on the way
            ..WalkInput::default()           // sink (WT.1) + future fields
        };
        body.walk(&scene, dt, input);
    }
    let eye = body.eye_pos(); // = Camera::from_yaw_pitch(eye.into(), yaw, pitch)
                              // ANCHOR_END: frame

    // The trajectory the walk above promises, pinned:
    assert!(body.on_ground(), "settled after the hop");
    // It fell to the floor plane, then stepped up the 1-voxel ledge
    // (surface z = 99) while walking +x.
    assert!(body.pos().x > 90.0, "reached the ledge region");
    assert!(
        (body.pos().z - 99.0).abs() < 0.01,
        "standing on the ledge, feet at {}",
        body.pos().z
    );
    assert!(eye.z < body.pos().z, "the eye is above the feet (-z is up)");

    // Fly mode is the demos' free camera: no gravity, full-3D wish,
    // instant start/stop, still sliding along walls.
    body.set_mode(MoveMode::Fly);
    body.walk(&scene, dt, WalkInput::default());
    assert_eq!(body.vel(), DVec3::ZERO, "fly idles in place");

    // The water curtain from `build_world` is passable: probe the
    // veto directly (the same query walk() uses).
    let wading = roxlap_scene::box_overlaps_solid(
        &scene,
        DVec3::new(40.2, 50.0, 90.0),
        DVec3::new(40.8, 50.6, 91.8),
        body.def().solidity,
    );
    assert!(!wading, "water is passable under the veto");

    println!("book_controller: all controller assertions hold");
}
