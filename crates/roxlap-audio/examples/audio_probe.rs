//! AU.2 — maintainer's listening probe (needs a real audio device):
//!
//! ```sh
//! cargo run -p roxlap-audio --features kira --example audio_probe
//! ```
//!
//! Builds a scene with a wall between a fixed source and a listener that
//! walks from behind the wall, through a doorway, into an open room, so
//! you can *hear* the occlusion lowpass lift and the reverb change. No
//! window, no render — just the acoustics core driving the kira backend.
//! Ctrl-C to quit.

use std::thread::sleep;
use std::time::Duration;

use glam::{DQuat, DVec3, IVec3};
use roxlap_audio::{
    source_acoustics, AcousticsConfig, AudioOut, CavityConfig, CavityEstimator, KiraAudio,
};
use roxlap_scene::{GridTransform, Scene, VoxColor};

const STONE: VoxColor = VoxColor(0x80_66_77_88);

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A sealed room on the left (the source sits inside), a 2-voxel wall
    // with a doorway, and open floor on the right (the listener's path).
    let mut scene = Scene::new();
    let id = scene.add_grid(GridTransform::identity());
    let g = scene.grid_mut(id).expect("grid");
    // Room shell around the source at x≈22.
    g.set_rect(IVec3::new(6, 40, 96), IVec3::new(40, 88, 150), Some(STONE));
    g.set_rect(IVec3::new(10, 44, 100), IVec3::new(36, 84, 146), None);
    // Dividing wall at x=40 with a doorway.
    g.set_rect(IVec3::new(40, 40, 96), IVec3::new(41, 88, 150), Some(STONE));
    g.set_rect(IVec3::new(40, 62, 100), IVec3::new(41, 66, 112), None);
    // A floor to the right so the open area isn't a bottomless void.
    g.set_rect(
        IVec3::new(42, 20, 150),
        IVec3::new(120, 110, 160),
        Some(STONE),
    );
    g.bake(roxlap_scene::BakeMode::Directional);

    let mut audio = KiraAudio::new()?;
    let hum = audio.register(&roxlap_audio::synth::hum(44_100));
    let source = DVec3::new(22.0, 64.0, 120.0);
    let src_id = audio.play_loop(hum, source, None).expect("a free voice");

    let acfg = AcousticsConfig::default();
    let ccfg = CavityConfig::default();
    let mut cavity = CavityEstimator::new(ccfg);

    // Listener path: deep in the sealed room → through the doorway →
    // out into the open floor.
    let path = [
        DVec3::new(30.0, 64.0, 120.0),  // inside the room, near the source
        DVec3::new(40.5, 64.0, 106.0),  // in the doorway
        DVec3::new(60.0, 64.0, 140.0),  // just outside
        DVec3::new(100.0, 64.0, 155.0), // open floor, far
    ];
    println!(
        "roxlap audio_probe: hum source in a sealed room; listener walks out. Ctrl-C to quit."
    );
    let mut leg = 0;
    loop {
        let a = path[leg % path.len()];
        let b = path[(leg + 1) % path.len()];
        for i in 0..30 {
            let f = f64::from(i) / 30.0;
            let listener = a.lerp(b, f);
            audio.set_listener(listener, DQuat::IDENTITY);
            let src = source_acoustics(&scene, source, listener, &acfg);
            audio.apply_source(src_id, &src);
            let env = cavity.update(&scene, listener);
            audio.apply_listener(&env);
            println!(
                "  listener {:>6.1?}  transmission {:.2}  cutoff {:>6.0}Hz  reverb mix {:.2} fb {:.2}",
                listener.to_array(),
                src.transmission,
                src.lowpass_cutoff_hz,
                env.reverb_mix,
                env.reverb_feedback,
            );
            sleep(Duration::from_millis(120));
        }
        leg += 1;
    }
}
