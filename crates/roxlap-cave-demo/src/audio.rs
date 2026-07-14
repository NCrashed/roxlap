//! AU.3 — voxel-aware audio for the cave demo (feature `audio`).
//!
//! Wires the `roxlap-audio` acoustics core + kira backend to the demo's
//! events: a plasma-shot transient at the muzzle, an impact boom at each
//! carve, and a looping hum at every glowing crystal — all muffled by
//! the rock between them and the listener, with reverb that follows the
//! cavern around the camera.
//!
//! Voice budget (per the AU.2 review): the crystal hums are
//! **distance-culled** to the nearest [`MAX_HUMS`] (within [`HUM_ENTER`],
//! held to [`HUM_EXIT`]), started/stopped as the listener moves, so a
//! cave full of crystals never starves the one-shot voices
//! (gunshots/booms).

use std::collections::{HashMap, HashSet};

use glam::{DMat3, DQuat, DVec3, IVec3};
use roxlap_audio::{
    doppler_factor, source_acoustics, synth, AcousticsConfig, AudioOut, CavityConfig,
    CavityEstimator, KiraAudio, SoundKey, SourceId, DEFAULT_SPEED_OF_SOUND,
};
use roxlap_core::Camera;
use roxlap_scene::{GridId, Scene};

const SAMPLE_RATE: u32 = 44_100;
/// Crystal hums audible at once — the nearest this many. Well under the
/// 24-voice pool, leaving headroom for shots and booms.
const MAX_HUMS: usize = 8;
/// Hysteresis for the near-set (world units): a crystal ENTERS the
/// audible set within `HUM_ENTER` and only LEAVES past `HUM_EXIT`, so a
/// crystal hovering at the boundary doesn't flicker its voice on camera
/// jitter.
const HUM_ENTER: f64 = 80.0;
const HUM_EXIT: f64 = 92.0;
/// Cap-membership hysteresis (world units): an already-active hum sorts
/// this much nearer than its true distance, so it isn't displaced from
/// the `MAX_HUMS` cap by a marginally-closer newcomer each recompute.
const HUM_CAP_HYSTERESIS: f64 = 6.0;
/// Cavity/reverb re-estimate rate (Hz) — the environment eases slowly,
/// no need to probe every frame.
const CAVITY_HZ: f64 = 2.0;
/// Near-set recompute rate (Hz) — throttled off the per-frame path so a
/// jittering camera doesn't thrash `play_loop`/`stop`.
const NEAR_HZ: f64 = 5.0;
/// Per-source occlusion re-evaluation rate (Hz) for the active hums.
const HUM_ACOUSTICS_HZ: f64 = 4.0;
/// Cap booms started in a single frame so a shotgun-into-a-wall burst
/// of impacts doesn't machine-gun the voice pool.
const MAX_BOOMS_PER_FRAME: usize = 2;
/// WT.4 — sound through water: half a voxel of effective rock per
/// voxel of water — muffled but audible (rock keeps the implicit 1.0).
const WATER_ABSORPTION: f32 = 0.5;

/// The demo's audio system. `None`-friendly: [`DemoAudio::new`] returns
/// `None` when no audio device is available (headless CI, no ALSA), and
/// the demo simply runs silent.
pub struct DemoAudio {
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

impl DemoAudio {
    /// Open the audio device and register the synthesized sounds, or
    /// `None` if no device is available (the demo then runs silent).
    pub fn new() -> Option<Self> {
        // WT.4 — opt into the WT.3 listener lowpass (a master-track
        // filter, construction-time-only in kira): the flooded cave
        // dunks the ear.
        let mut audio = match KiraAudio::with_options(roxlap_audio::DEFAULT_POOL, true) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("roxlap-cave-demo: audio disabled ({e})");
                return None;
            }
        };
        let shot = audio.register(&synth::shot(SAMPLE_RATE));
        let impact = audio.register(&synth::impact(SAMPLE_RATE));
        let hum = audio.register(&synth::hum(SAMPLE_RATE));
        // WT.4 — water muffles less than rock: a boom heard through
        // the pool reads distant, not sealed away (AU2 per-material
        // absorption; rock keeps the implicit 1.0). The colour map is
        // THE shared one (crystals riding along classify as material
        // 1 with no absorption entry = the default 1.0 — a no-op).
        let acfg = AcousticsConfig {
            material_map: crate::material_map().to_vec(),
            absorption: vec![(crate::WATER_MATERIAL_ID, WATER_ABSORPTION)],
            ..AcousticsConfig::default()
        };
        Some(Self {
            audio,
            shot,
            impact,
            hum,
            acfg,
            cavity: CavityEstimator::new(CavityConfig::default()),
            hums: HashMap::new(),
            cavity_timer: 0.0,
            hum_timer: 0.0,
            near_timer: 0.0,
            prev_listener: None,
        })
    }

    /// A gunshot at the muzzle, occlusion-shaded for the current
    /// listener. Shaded at spawn (`play`'s instant `initial`), not
    /// tracked — a one-shot's envelope outruns a 120 ms occlusion ramp,
    /// so it must START muffled.
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

    /// WT.4 — the underwater ear: drive the WT.3 listener lowpass from
    /// the submersion signal, every frame. Cheap above water — the
    /// kira backend dedups an unchanged target (no tween restart).
    pub fn set_submerged(&mut self, under: bool) {
        self.audio.set_listener_lowpass(if under {
            roxlap_audio::LISTENER_LOWPASS_SUBMERGED_HZ
        } else {
            roxlap_audio::LISTENER_LOWPASS_OPEN_HZ
        });
    }

    /// Stop every crystal hum and clear the reverb history — call on a
    /// world regenerate (F/R), where the crystal set changes wholesale.
    pub fn reset(&mut self) {
        for (_, id) in self.hums.drain() {
            self.audio.stop(id);
        }
        self.cavity.reset();
        self.cavity_timer = 0.0;
        self.hum_timer = 0.0;
        self.near_timer = 0.0;
        // AU2.2 — a regen teleports the camera; forget the old position
        // so the first post-reset tick reads as at-rest, not warp-speed.
        self.prev_listener = None;
    }

    /// Per-frame update: listener pose (every frame), reverb environment
    /// (throttled, skipped while the camera is buried), and the
    /// distance-culled crystal hums + their occlusion.
    ///
    /// **Assumes the cave grid is identity-transform** (it is — one
    /// chunk at the origin, `main.rs`), so world coords == grid-local:
    /// `listener` and `bake_lights[i].pos` are used directly as both. A
    /// transformed grid would need a `world_to_grid_local` rebase for
    /// the `voxel_solid` guard and the acoustic queries.
    pub fn tick(
        &mut self,
        dt: f64,
        scene: &Scene,
        grid_id: GridId,
        listener: DVec3,
        orientation: DQuat,
    ) {
        self.audio.set_listener(listener, orientation);

        let Some(grid) = scene.grid(grid_id) else {
            return;
        };

        // Reverb environment at CAVITY_HZ. Guard: a camera clipped into
        // rock sees every ray hit at t≈0 and would collapse the reverb
        // to a tiny sealed box — skip the update and hold the last one.
        // (Only the eye-centre voxel is tested; a near-clip with the eye
        // just outside a wall but rays into rock can still collapse it —
        // acceptable for the demo.)
        self.cavity_timer += dt;
        if self.cavity_timer >= 1.0 / CAVITY_HZ {
            self.cavity_timer = 0.0;
            if !grid.voxel_solid(floor_ivec(listener)) {
                let env = self.cavity.update(scene, listener);
                self.audio.apply_listener(&env);
            }
        }

        // Recompute the near-set (throttled + hysteretic) and start/stop
        // hums under the diff.
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
        // Per-frame is deliberate (positions already update per frame;
        // ≤ MAX_HUMS playback-rate commands/frame) — if the kira
        // command queue ever shows in a profile, throttle HERE first.
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
    /// The threshold sits well above the demo's fast-fly (~4× walk);
    /// a future movement mode faster than 200 u/s would silently mute
    /// its own Doppler — raise the guard alongside it.
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
/// hysteresis on both the radius (enter within [`HUM_ENTER`], leave past
/// [`HUM_EXIT`]) and the cap (an already-`active` crystal sorts
/// [`HUM_CAP_HYSTERESIS`] nearer so it isn't displaced from the cap by a
/// marginally-closer newcomer). Pure — unit-tested without a device.
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

/// Listener orientation from the camera basis. The listener's local
/// axes (kira convention: +X right, +Y up, −Z forward) map to the
/// camera's world basis; positions are ALSO passed in roxlap world
/// coords, so kira reads both in its own frame but consistently —
/// azimuthal panning follows head-turn correctly. `det = +1` because
/// negating two columns of the right-handed `(right, down, forward)`
/// voxlap basis preserves orientation.
pub fn listener_orientation(cam: &Camera) -> DQuat {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn xs(coords: &[f64]) -> Vec<DVec3> {
        coords.iter().map(|&x| DVec3::new(x, 0.0, 0.0)).collect()
    }

    #[test]
    fn caps_to_nearest_max() {
        // 10 crystals all within HUM_ENTER; only the nearest MAX_HUMS win.
        let pos = xs(&[5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0]);
        let sel = select_near(&pos, DVec3::ZERO, &HashSet::new());
        assert_eq!(sel.len(), MAX_HUMS);
        // The two farthest (indices 8, 9) are excluded.
        assert!(!sel.contains(&8) && !sel.contains(&9));
        assert!(sel.contains(&0) && sel.contains(&7));
    }

    #[test]
    fn radius_hysteresis_holds_a_boundary_crystal() {
        let pos = xs(&[85.0]); // between HUM_ENTER (80) and HUM_EXIT (92)
                               // Inactive at 85 ⇒ NOT admitted (past the 80 enter radius).
        assert!(select_near(&pos, DVec3::ZERO, &HashSet::new()).is_empty());
        // Already active at 85 ⇒ HELD (within the 92 exit radius).
        let active: HashSet<usize> = [0].into_iter().collect();
        assert!(select_near(&pos, DVec3::ZERO, &active).contains(&0));
        // Only past HUM_EXIT does it finally drop.
        let far = xs(&[95.0]);
        assert!(select_near(&far, DVec3::ZERO, &active).is_empty());
    }

    #[test]
    fn cap_hysteresis_keeps_active_over_marginal_newcomer() {
        // Cap of MAX_HUMS filled: an active hum at the cap edge must not
        // be evicted by a newcomer only slightly closer.
        // MAX_HUMS active crystals at 10,20,...; plus a newcomer 1 unit
        // nearer than the farthest active one.
        let mut coords: Vec<f64> = (0..MAX_HUMS).map(|i| 10.0 + i as f64 * 5.0).collect();
        let farthest_active_d = *coords.last().unwrap();
        coords.push(farthest_active_d - 1.0); // newcomer, marginally closer
        let pos = xs(&coords);
        let active: HashSet<usize> = (0..MAX_HUMS).collect();
        let sel = select_near(&pos, DVec3::ZERO, &active);
        assert_eq!(sel.len(), MAX_HUMS);
        // The newcomer (last index) does NOT displace the active edge hum.
        assert!(!sel.contains(&MAX_HUMS), "marginal newcomer must not evict");
        assert!(sel.contains(&(MAX_HUMS - 1)), "active edge hum held");
    }

    #[test]
    fn empty_and_far_select_nothing() {
        assert!(select_near(&[], DVec3::ZERO, &HashSet::new()).is_empty());
        let pos = xs(&[500.0, 1000.0]);
        assert!(select_near(&pos, DVec3::ZERO, &HashSet::new()).is_empty());
    }
}
