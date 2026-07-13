//! PW.0 — voxel-aware audio for the browser cave demo (feature
//! `audio`, `PORTING-PLATFORM.md`).
//!
//! The native cave demo's `DemoAudio`, ported whole (PW.0b): a
//! plasma-shot transient at the muzzle, an impact boom at every carve
//! and island landing, a looping hum at every glowing crystal — all
//! muffled by the rock between them and the listener — plus the cavity
//! reverb that swells in big chambers and dries in tunnels, and AU2
//! Doppler bending the hums as the listener flies past.
//!
//! Voice budget (per the AU.2 review): the crystal hums are
//! **distance-culled** to the nearest [`MAX_HUMS`] (within
//! [`HUM_ENTER`], held to [`HUM_EXIT`]), started/stopped as the
//! listener moves, so a cave full of crystals never starves the
//! one-shot voices. `select_near` is byte-identical to the native
//! demo's, which unit-tests it (this crate is wasm-only, so the tests
//! live there).
//!
//! kira 0.12 runs on wasm out of the box (its manifest switches cpal to
//! the `wasm-bindgen` WebAudio backend). The ONE web-specific rule:
//! **construct this only inside a user-gesture handler** (the first
//! pointer-lock click / first touch) — an `AudioContext` created before
//! a gesture is suspended by the browser autoplay policy and, on some
//! browsers, never recovers. `crate::lib` wires that.

use std::collections::{HashMap, HashSet};

use glam::{DMat3, DQuat, DVec3, IVec3};
use roxlap_audio::{
    doppler_factor, source_acoustics, synth, AcousticsConfig, AudioOut, CavityConfig,
    CavityEstimator, KiraAudio, SoundKey, SourceId, DEFAULT_SPEED_OF_SOUND,
};
use roxlap_core::Camera;
use roxlap_scene::{GridId, Scene};

const SAMPLE_RATE: u32 = 44_100;
/// Crystal hums audible at once — the nearest this many. Well under
/// the 24-voice pool, leaving headroom for shots and booms.
const MAX_HUMS: usize = 8;
/// Hysteresis for the near-set (world units): a crystal ENTERS the
/// audible set within `HUM_ENTER` and only LEAVES past `HUM_EXIT`, so
/// a crystal hovering at the boundary doesn't flicker its voice on
/// camera jitter.
const HUM_ENTER: f64 = 80.0;
const HUM_EXIT: f64 = 92.0;
/// Cap-membership hysteresis (world units): an already-active hum
/// sorts this much nearer than its true distance, so it isn't
/// displaced from the `MAX_HUMS` cap by a marginally-closer newcomer
/// each recompute.
const HUM_CAP_HYSTERESIS: f64 = 6.0;
/// Cavity/reverb re-estimate rate (Hz) — the native demo's cadence.
const CAVITY_HZ: f64 = 2.0;
/// Near-set recompute rate (Hz) — throttled off the per-frame path so
/// a jittering camera doesn't thrash `play_loop`/`stop`.
const NEAR_HZ: f64 = 5.0;
/// Per-source occlusion re-evaluation rate (Hz) for the active hums.
const HUM_ACOUSTICS_HZ: f64 = 4.0;
/// Cap booms started in a single frame (multi-impact carve bursts).
const MAX_BOOMS_PER_FRAME: usize = 2;

/// The browser demo's audio system. Built lazily on the first user
/// gesture; `None` from [`WebAudio::new`] when the device/context is
/// unavailable — the demo then simply stays silent.
pub struct WebAudio {
    audio: KiraAudio,
    shot: SoundKey,
    impact: SoundKey,
    hum: SoundKey,
    acfg: AcousticsConfig,
    cavity: CavityEstimator,
    /// Crystal (bake-light) index → its live looping hum voice.
    hums: HashMap<usize, SourceId>,
    cavity_timer: f64,
    hum_timer: f64,
    near_timer: f64,
    /// AU2.2 — last tick's listener position, for the Doppler velocity
    /// estimate (`None` right after construction / [`Self::reset`]).
    prev_listener: Option<DVec3>,
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
        let hum = audio.register(&synth::hum(SAMPLE_RATE));
        Some(Self {
            audio,
            shot,
            impact,
            hum,
            acfg: AcousticsConfig::default(),
            cavity: CavityEstimator::new(CavityConfig::default()),
            hums: HashMap::new(),
            cavity_timer: f64::MAX, // first tick estimates immediately
            hum_timer: 0.0,
            near_timer: 0.0,
            prev_listener: None,
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
            let at = voxel_centre(*hit);
            let a = source_acoustics(scene, at, listener, &self.acfg);
            self.audio.play(self.impact, at, Some(&a));
        }
    }

    /// Stop every crystal hum and clear the reverb history — call on a
    /// world regenerate (preset / reseed), where the crystal set
    /// changes wholesale under the listener.
    pub fn reset(&mut self) {
        for (_, id) in self.hums.drain() {
            self.audio.stop(id);
        }
        self.cavity.reset();
        self.cavity_timer = f64::MAX;
        self.hum_timer = 0.0;
        self.near_timer = 0.0;
        // AU2.2 — a regen teleports the camera; forget the old position
        // so the first post-reset tick reads as at-rest, not warp-speed.
        self.prev_listener = None;
    }

    /// Per-frame update: listener pose (every frame), reverb
    /// environment (throttled, skipped while the camera is buried — a
    /// clipped eye would collapse the reverb to a sealed box), the
    /// distance-culled crystal hums + their occlusion, and Doppler
    /// from the listener's velocity. The cave grid is
    /// identity-transform, so world coords == grid-local (same
    /// assumption as the native demo).
    pub fn tick(&mut self, dt: f64, scene: &Scene, grid_id: GridId, cam: &Camera) {
        let listener = DVec3::from(cam.pos);
        self.audio.set_listener(listener, listener_orientation(cam));

        let Some(grid) = scene.grid(grid_id) else {
            return;
        };

        self.cavity_timer += dt;
        if self.cavity_timer >= 1.0 / CAVITY_HZ {
            self.cavity_timer = 0.0;
            if !grid.voxel_solid(floor_ivec(listener)) {
                let env = self.cavity.update(scene, listener);
                self.audio.apply_listener(&env);
            }
        }

        // Recompute the near-set (throttled + hysteretic) and start/
        // stop hums under the diff.
        self.near_timer += dt;
        if self.near_timer >= 1.0 / NEAR_HZ {
            self.near_timer = 0.0;
            let positions: Vec<DVec3> = grid.bake_lights.iter().map(|l| l.pos.as_dvec3()).collect();
            let active: HashSet<usize> = self.hums.keys().copied().collect();
            let want = select_near(&positions, listener, &active);

            let stale: Vec<usize> = active.difference(&want).copied().collect();
            for i in stale {
                if let Some(id) = self.hums.remove(&i) {
                    self.audio.stop(id);
                }
            }
            for &i in &want {
                if !self.hums.contains_key(&i) {
                    let pos = positions[i];
                    // Start already muffled for the rock in the way.
                    let a = source_acoustics(scene, pos, listener, &self.acfg);
                    if let Some(id) = self.audio.play_loop(self.hum, pos, Some(&a)) {
                        self.hums.insert(i, id);
                    }
                }
            }
        }

        // Occlusion for the active hums at HUM_ACOUSTICS_HZ.
        self.hum_timer += dt;
        if self.hum_timer >= 1.0 / HUM_ACOUSTICS_HZ {
            self.hum_timer = 0.0;
            for (&i, &id) in &self.hums {
                let pos = grid.bake_lights[i].pos.as_dvec3();
                let a = source_acoustics(scene, pos, listener, &self.acfg);
                self.audio.apply_source(id, &a);
            }
        }

        // AU2.2 — Doppler on the hums, every frame (pure math over a
        // handful of sources; kira's ~120 ms tween smooths the ramp):
        // crystals are static, so only the listener's velocity bends
        // the pitch — fly past a crystal and its hum bends down.
        let vel = self.listener_velocity(listener, dt);
        for (&i, &id) in &self.hums {
            let pos = grid.bake_lights[i].pos.as_dvec3();
            let f = doppler_factor(pos, DVec3::ZERO, listener, vel, DEFAULT_SPEED_OF_SOUND);
            self.audio.set_source_pitch(id, f);
        }
    }

    /// AU2.2 — the listener's velocity from consecutive tick positions.
    /// Teleport-guarded: a jump reading over 200 u/s (regen, spawn
    /// warp) is treated as rest rather than a one-frame pitch spike.
    fn listener_velocity(&mut self, listener: DVec3, dt: f64) -> DVec3 {
        let prev = self.prev_listener.replace(listener);
        match prev {
            Some(p) if dt > 1e-4 => {
                let v = (listener - p) / dt;
                if v.length() > 200.0 {
                    DVec3::ZERO
                } else {
                    v
                }
            }
            _ => DVec3::ZERO,
        }
    }
}

/// The nearest [`MAX_HUMS`] crystals that should be audible, with
/// hysteresis on both the radius (enter within [`HUM_ENTER`], leave
/// past [`HUM_EXIT`]) and the cap (an already-`active` crystal sorts
/// [`HUM_CAP_HYSTERESIS`] nearer so it isn't displaced from the cap by
/// a marginally-closer newcomer). Identical to the native demo's,
/// where it is unit-tested.
fn select_near(positions: &[DVec3], listener: DVec3, active: &HashSet<usize>) -> HashSet<usize> {
    let mut cands: Vec<(usize, f64)> = positions
        .iter()
        .enumerate()
        .filter_map(|(i, &p)| {
            let d = (p - listener).length();
            let is_active = active.contains(&i);
            let limit = if is_active { HUM_EXIT } else { HUM_ENTER };
            if d > limit {
                return None;
            }
            // Bias active hums nearer for the cap-membership sort.
            let eff = if is_active { d - HUM_CAP_HYSTERESIS } else { d };
            Some((i, eff))
        })
        .collect();
    cands.sort_by(|a, b| a.1.total_cmp(&b.1));
    cands.truncate(MAX_HUMS);
    cands.into_iter().map(|(i, _)| i).collect()
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

fn voxel_centre(v: IVec3) -> DVec3 {
    DVec3::new(
        f64::from(v.x) + 0.5,
        f64::from(v.y) + 0.5,
        f64::from(v.z) + 0.5,
    )
}

#[allow(clippy::cast_possible_truncation)]
fn floor_ivec(p: DVec3) -> IVec3 {
    IVec3::new(p.x.floor() as i32, p.y.floor() as i32, p.z.floor() as i32)
}
