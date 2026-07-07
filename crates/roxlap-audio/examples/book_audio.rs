//! Companion example for the book's "Audio" chapter
//! (`docs/book/src/audio.md`) — the chapter pulls its snippets from
//! here via `// ANCHOR:` markers, so everything it shows compiles.
//!
//! This exercises the acoustics **core** only (no `kira` feature, no
//! audio device): it builds a tiny voxel scene and prints the occlusion
//! and reverb parameters roxlap-audio computes for it, so you can see
//! the numbers a playback backend would apply. Run it anywhere:
//!
//! ```sh
//! cargo run -p roxlap-audio --example book_audio
//! ```
//!
//! Keep the anchors when editing; `docs/book/check-anchors.sh` (run by
//! the CI `book` job) goes red if one disappears.

use glam::{DVec3, IVec3};
use roxlap_audio::{
    source_acoustics, AcousticsConfig, CavityConfig, CavityEstimator, SourceAcoustics,
};
use roxlap_scene::{GridTransform, Scene, VoxColor};

const STONE: VoxColor = VoxColor(0x80_66_77_88);

fn main() {
    // ANCHOR: scene
    // A source in a sealed stone room, a listener out in the open, and
    // a 3-voxel wall between them (with a doorway punched to one side),
    // all standing on a stone floor (z is DOWN, so "the ground" is the
    // high-z slab everything rests on).
    let mut scene = Scene::new();
    let id = scene.add_grid(GridTransform::identity());
    let grid = scene.grid_mut(id).expect("grid just added");
    grid.set_rect(
        IVec3::new(0, 0, 150),
        IVec3::new(127, 127, 170),
        Some(STONE),
    ); // floor
    grid.set_rect(IVec3::new(6, 40, 96), IVec3::new(40, 88, 150), Some(STONE));
    grid.set_rect(IVec3::new(10, 44, 100), IVec3::new(36, 84, 146), None);
    grid.set_rect(IVec3::new(60, 40, 96), IVec3::new(62, 88, 150), Some(STONE));
    grid.set_rect(IVec3::new(60, 62, 100), IVec3::new(62, 66, 112), None); // doorway

    let source = DVec3::new(22.0, 64.0, 120.0); // inside the room
                                                // ANCHOR_END: scene

    // ANCHOR: occlusion
    // Occlusion is a fan of rays from the source to the listener, each
    // accumulating the SOLID THICKNESS it crosses; the average becomes a
    // transmission that drives a muffling lowpass + a volume drop.
    let cfg = AcousticsConfig::default();
    for (label, listener) in [
        ("line of sight", DVec3::new(30.0, 64.0, 120.0)),
        ("through the wall", DVec3::new(80.0, 64.0, 120.0)),
        ("past the doorway", DVec3::new(80.0, 64.0, 106.0)),
    ] {
        let a: SourceAcoustics = source_acoustics(&scene, source, listener, &cfg);
        println!(
            "{label:>16}: transmission {:.2}  cutoff {:>6.0} Hz  gain {:+.1} dB",
            a.transmission, a.lowpass_cutoff_hz, a.gain_db,
        );
    }
    // ANCHOR_END: occlusion

    // ANCHOR: cavity
    // Reverb comes from a fan of rays around the LISTENER: the mean
    // enclosed free path reads as room size (feedback), the fraction
    // escaping to open sky as dryness. A `CavityEstimator` smooths the
    // raw probe over time — call it at ~2 Hz, not every frame.
    let mut cavity = CavityEstimator::new(CavityConfig::default());
    for (label, listener) in [
        ("in the room", DVec3::new(22.0, 64.0, 120.0)),
        ("out in the open", DVec3::new(90.0, 64.0, 120.0)),
    ] {
        // The first update seeds from the raw probe (no fade-in); in a
        // game you'd feed each result to the reverb with a ~1 s tween.
        cavity.reset();
        let env = cavity.update(&scene, listener);
        println!(
            "{label:>16}: openness {:.2}  reverb feedback {:.2}  mix {:.2}",
            env.openness, env.reverb_feedback, env.reverb_mix,
        );
    }
    // ANCHOR_END: cavity
}
