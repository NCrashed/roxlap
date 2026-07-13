//! PW.0 — voxel-aware audio for the browser cave demo (feature
//! `audio`, `PORTING-PLATFORM.md`).
//!
//! A trimmed port of the native cave demo's `DemoAudio`: a plasma-shot
//! transient at the muzzle and an impact boom at every carve — each
//! occlusion-shaded through the rock between it and the listener — plus
//! the cavity reverb that swells in big chambers and dries in tunnels.
//! No crystal hums (and therefore no Doppler loops): the web cave has
//! no crystals — that half follows if the crystals are ever ported.
//!
//! kira 0.12 runs on wasm out of the box (its manifest switches cpal to
//! the `wasm-bindgen` WebAudio backend). The ONE web-specific rule:
//! **construct this only inside a user-gesture handler** (the first
//! pointer-lock click / first touch) — an `AudioContext` created before
//! a gesture is suspended by the browser autoplay policy and, on some
//! browsers, never recovers. `crate::lib` wires that.

use glam::{DMat3, DQuat, DVec3, IVec3};
use roxlap_audio::{
    source_acoustics, synth, AcousticsConfig, AudioOut, CavityConfig, CavityEstimator, KiraAudio,
    SoundKey,
};
use roxlap_core::Camera;
use roxlap_scene::{GridId, Scene};

const SAMPLE_RATE: u32 = 44_100;
/// Cavity/reverb re-estimate rate (Hz) — the native demo's cadence.
const CAVITY_HZ: f64 = 2.0;
/// Cap booms started in a single frame (multi-impact carve bursts).
const MAX_BOOMS_PER_FRAME: usize = 2;

/// The browser demo's audio system. Built lazily on the first user
/// gesture; `None` from [`WebAudio::new`] when the device/context is
/// unavailable — the demo then simply stays silent.
pub struct WebAudio {
    audio: KiraAudio,
    shot: SoundKey,
    impact: SoundKey,
    acfg: AcousticsConfig,
    cavity: CavityEstimator,
    cavity_timer: f64,
}

impl WebAudio {
    /// Open the WebAudio backend and register the synthesized sounds.
    /// MUST be called from inside a user-gesture event handler (see
    /// the module doc).
    pub fn new() -> Option<Self> {
        let mut audio = match KiraAudio::new() {
            Ok(a) => a,
            Err(e) => {
                web_sys::console::warn_1(&format!("roxlap-cave-web: audio disabled ({e})").into());
                return None;
            }
        };
        let shot = audio.register(&synth::shot(SAMPLE_RATE));
        let impact = audio.register(&synth::impact(SAMPLE_RATE));
        Some(Self {
            audio,
            shot,
            impact,
            acfg: AcousticsConfig::default(),
            cavity: CavityEstimator::new(CavityConfig::default()),
            cavity_timer: f64::MAX, // first tick estimates immediately
        })
    }

    /// A gunshot at the muzzle, occlusion-shaded at spawn (a one-shot's
    /// envelope outruns any tween — it must START muffled).
    pub fn fire(&mut self, muzzle: [f64; 3], scene: &Scene, listener: DVec3) {
        let at = DVec3::from(muzzle);
        let a = source_acoustics(scene, at, listener, &self.acfg);
        self.audio.play(self.shot, at, Some(&a));
    }

    /// Impact booms at the carve voxels (capped per frame), each shaded
    /// at spawn for the rock between it and the listener.
    pub fn impacts(&mut self, hits: &[IVec3], scene: &Scene, listener: DVec3) {
        for hit in hits.iter().take(MAX_BOOMS_PER_FRAME) {
            let at = DVec3::new(
                f64::from(hit.x) + 0.5,
                f64::from(hit.y) + 0.5,
                f64::from(hit.z) + 0.5,
            );
            let a = source_acoustics(scene, at, listener, &self.acfg);
            self.audio.play(self.impact, at, Some(&a));
        }
    }

    /// Clear the reverb history — call on a world regenerate (preset /
    /// reseed), where the cave changes wholesale under the listener.
    pub fn reset(&mut self) {
        self.cavity.reset();
        self.cavity_timer = f64::MAX;
    }

    /// Per-frame update: listener pose every frame, the reverb
    /// environment at [`CAVITY_HZ`] (skipped while the camera is
    /// buried — a clipped eye would collapse the reverb to a sealed
    /// box). The cave grid is identity-transform, so world coords ==
    /// grid-local (same assumption as the native demo).
    pub fn tick(&mut self, dt: f64, scene: &Scene, grid_id: GridId, cam: &Camera) {
        let listener = DVec3::from(cam.pos);
        self.audio.set_listener(listener, listener_orientation(cam));

        let Some(grid) = scene.grid(grid_id) else {
            return;
        };
        self.cavity_timer += dt;
        if self.cavity_timer >= 1.0 / CAVITY_HZ {
            self.cavity_timer = 0.0;
            #[allow(clippy::cast_possible_truncation)]
            let eye = IVec3::new(
                listener.x.floor() as i32,
                listener.y.floor() as i32,
                listener.z.floor() as i32,
            );
            if !grid.voxel_solid(eye) {
                let env = self.cavity.update(scene, listener);
                self.audio.apply_listener(&env);
            }
        }
    }
}

/// kira wants the listener as a quaternion; build it from the camera
/// basis (right, up = −down, back = −forward) — the native demo's
/// convention.
fn listener_orientation(cam: &Camera) -> DQuat {
    let right = DVec3::from(cam.right);
    let down = DVec3::from(cam.down);
    let forward = DVec3::from(cam.forward);
    DQuat::from_mat3(&DMat3::from_cols(right, -down, -forward))
}
