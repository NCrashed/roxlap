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
    doppler_factor, source_acoustics, AcousticsConfig, CavityConfig, CavityEstimator,
    SourceAcoustics, DEFAULT_SPEED_OF_SOUND,
};
use roxlap_scene::{GridTransform, Scene, VoxColor};

const STONE: VoxColor = VoxColor(0x80_66_77_88);
/// AU2 — the material-aware sections map this colour to material 7.
const GLASS: VoxColor = VoxColor(0x80_46_c4_e2);
const GLASS_ID: u8 = 7;

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

    // ANCHOR: materials
    // AU2 — materials change what the rays hear. The colour→material
    // map is the SAME one the renderer's `set_terrain_materials` (and
    // the debris system's fracture tables) take; `absorption` weighs a
    // wall's effective thickness, `damping_override` recolours the
    // reverb of the walls around the listener. Everything is PAINTED,
    // never carved — a carve leaves colour-less faces the classifier
    // can't read.
    let grid = scene.grid_mut(id).expect("grid");
    // A 3-voxel glass pane…
    grid.set_rect(
        IVec3::new(76, 20, 100),
        IVec3::new(78, 40, 140),
        Some(GLASS),
    );
    // …and a small sealed glass booth (six painted slabs).
    let (bl, bh) = (IVec3::new(98, 22, 118), IVec3::new(106, 30, 130));
    for ax in 0..3 {
        for side in 0..2 {
            let mut slo = bl - IVec3::splat(2);
            let mut shi = bh + IVec3::splat(2);
            if side == 0 {
                shi[ax] = bl[ax] - 1;
            } else {
                slo[ax] = bh[ax] + 1;
            }
            grid.set_rect(slo, shi, Some(GLASS));
        }
    }

    // The same pane, heard with and without the glass tables: 3 voxels
    // of wall count as ~1 voxel of muffling at 0.35×.
    let by_pane = DVec3::new(70.0, 30.0, 120.0);
    let behind_pane = DVec3::new(85.0, 30.0, 120.0);
    let mcfg = AcousticsConfig {
        material_map: vec![(GLASS.rgb_part(), GLASS_ID)],
        absorption: vec![(GLASS_ID, 0.35)],
        ..AcousticsConfig::default()
    };
    let as_stone = source_acoustics(&scene, by_pane, behind_pane, &cfg);
    let as_glass = source_acoustics(&scene, by_pane, behind_pane, &mcfg);
    println!(
        "same pane, stone vs glass tables: transmission {:.2} -> {:.2}",
        as_stone.transmission, as_glass.transmission,
    );

    // Reverb character: from inside the booth, every wall hit
    // classifies to the override's 0.1 damping — a glass chamber
    // rings brighter than the stone default (0.4).
    let mut glassy = CavityEstimator::new(CavityConfig {
        material_map: vec![(GLASS.rgb_part(), GLASS_ID)],
        damping_override: vec![(GLASS_ID, 0.1)],
        ..CavityConfig::default()
    });
    let env = glassy.update(&scene, DVec3::new(102.0, 26.0, 124.0));
    println!(
        "inside the glass booth: damping {:.2} (stone default 0.40)",
        env.reverb_damping,
    );
    // ANCHOR_END: materials

    // ANCHOR: doppler
    // AU2 — Doppler is pure math over positions and velocities: feed
    // the factor to `AudioOut::set_source_pitch` (the kira backend
    // tweens the playback rate). Apply it to LOOPS — bending a
    // one-shot mid-envelope reads as a glitch, not physics.
    let listener = DVec3::new(80.0, 64.0, 120.0);
    for (label, vel) in [
        ("flying toward the source", DVec3::new(-40.0, 0.0, 0.0)),
        ("flying away", DVec3::new(40.0, 0.0, 0.0)),
        ("standing still", DVec3::ZERO),
    ] {
        let f = doppler_factor(source, DVec3::ZERO, listener, vel, DEFAULT_SPEED_OF_SOUND);
        println!("{label:>26}: pitch x{f:.2}");
    }
    // ANCHOR_END: doppler
}
