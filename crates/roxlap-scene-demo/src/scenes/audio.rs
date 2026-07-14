//! The **Audio** scene (macro-stage AU2, `PORTING-AUDIO.md`) — the
//! book's `book_audio` numbers made walkable: a sealed stone room with
//! a doorway, an open field, and a big **glass** cavern, one looping
//! source per zone, with the live acoustic parameters in the HUD.
//!
//! The numbers come from the **pure** acoustics core, so this tab
//! exists in every build — walk between the zones and watch the
//! occlusion transmission, the reverb environment (the glass cavern's
//! per-material damping reads lower = brighter than stone, AU2.1) and
//! the Doppler factors bend as you fly. Build with
//! `--features audio` to also *hear* it (kira playback, the cave
//! demo's convention); without the feature the scene runs silent.
//!
//! The per-material tables here are ACTIVE (crystal colour → material
//! 7: absorption 0.35 + damping 0.1) — this scene is also the AU2
//! profiling harness owed by the entry doc: the HUD's `acoustics:` line
//! is the measured cost of the last full parameter recompute.

use std::time::Instant;

use glam::DVec3;
use glam::IVec3;
use roxlap_audio::{
    doppler_factor, source_acoustics, AcousticsConfig, CavityConfig, CavityEstimator,
    ListenerAcoustics, SourceAcoustics, DEFAULT_SPEED_OF_SOUND,
};
#[cfg(feature = "audio")]
use roxlap_audio::{synth, AudioOut, KiraAudio, SourceId};
use roxlap_render::{Material, VoxColor};
use roxlap_scene::{
    CharacterBody, CharacterDef, GridTransform, MoveMode, Scene, Solidity, WalkInput,
};

use crate::scene_api::{frame_params, opticast_settings, CameraPose, DemoScene, SceneCtx};

const STONE: VoxColor = VoxColor(0x80_6a_60_52);
const GLASS: VoxColor = VoxColor(0x80_46_c4_e2);
/// Palette id of the glass material (0 is the reserved opaque stone).
const GLASS_ID: u8 = 7;

/// Refresh rates, mirroring the cave demo's cadence: the environment
/// eases slowly, per-source occlusion a bit faster, Doppler per frame.
const CAVITY_HZ: f64 = 2.0;
const SOURCE_HZ: f64 = 4.0;

/// The three looping sources: sealed room, open field, glass cavern.
const SOURCES: [([f64; 3], &str); 3] = [
    ([16.5, 32.5, 195.5], "room"),
    ([44.5, 32.5, 195.5], "open"),
    ([78.5, 32.5, 189.5], "cavern"),
];

/// Camera start — shared by [`DemoScene::start_pose`] and the playback
/// spawn shading (the hums are born pre-shaded for THIS pose; one
/// definition so they can't drift apart).
const START_POS: [f64; 3] = [44.0, 8.0, 192.0];

pub struct AudioScene {
    scene: Scene,
    body: CharacterBody,
    last_eye: [f64; 3],
    /// Occlusion config with ACTIVE material tables (AU2.0).
    acfg: AcousticsConfig,
    /// Reverb estimator with ACTIVE damping tables (AU2.1).
    cavity: CavityEstimator,
    src: [SourceAcoustics; 3],
    env: Option<ListenerAcoustics>,
    doppler: [f32; 3],
    /// Last tick's eye position, for the Doppler velocity estimate.
    prev_eye: Option<DVec3>,
    listener_speed: f64,
    /// Measured cost of the last acoustics recompute (µs) — the AU2
    /// profiling readout (tables active). Diagnostic, deliberately
    /// coarse: 4 Hz ticks time sources-only, every other one adds the
    /// cavity probe, so the readout alternates between two magnitudes.
    acoustics_us: f64,
    cavity_timer: f64,
    source_timer: f64,
    #[cfg(feature = "audio")]
    playback: Option<Playback>,
}

/// Audible half (feature `audio`): a kira device + one looping hum per
/// zone. Dropping it tears the audio thread down (scene switch).
#[cfg(feature = "audio")]
struct Playback {
    audio: KiraAudio,
    hums: [SourceId; 3],
}

impl AudioScene {
    #[must_use]
    pub fn new() -> Self {
        let mut body = CharacterBody::new(CharacterDef {
            fly_speed: crate::scene_api::MOVE_SPEED,
            solidity: Solidity {
                bedrock_blocks: false,
                ..Solidity::default()
            },
            ..CharacterDef::default()
        });
        body.set_mode(MoveMode::Fly);
        let map = vec![(GLASS.rgb_part(), GLASS_ID)];
        let acfg = AcousticsConfig {
            material_map: map.clone(),
            absorption: vec![(GLASS_ID, 0.35)],
            ..AcousticsConfig::default()
        };
        let cavity = CavityEstimator::new(CavityConfig {
            material_map: map,
            damping_override: vec![(GLASS_ID, 0.1)],
            ..CavityConfig::default()
        });
        let clear = SourceAcoustics::clear(&acfg);
        Self {
            scene: build_world(),
            body,
            last_eye: [f64::NAN; 3],
            src: [clear; 3],
            env: None,
            doppler: [1.0; 3],
            prev_eye: None,
            listener_speed: 0.0,
            acoustics_us: 0.0,
            cavity_timer: f64::MAX, // force an immediate first update
            source_timer: f64::MAX,
            acfg,
            cavity,
            #[cfg(feature = "audio")]
            playback: None,
        }
    }
}

/// Floor + a sealed stone room (doorway on +x) + a big glass cavern
/// (entrance on −x). Everything is built from PAINTED slabs, never
/// carved — a carve leaves zero-RGB faces the material classification
/// couldn't read (the AU2.1 lesson).
fn build_world() -> Scene {
    let mut scene = Scene::new();
    let id = scene.add_grid(GridTransform::identity());
    let g = scene.grid_mut(id).expect("grid just added");

    // Ground slab under everything (z is down; the floor top is 200).
    g.set_rect(IVec3::new(0, 0, 200), IVec3::new(95, 63, 206), Some(STONE));

    // --- Sealed stone room, interior x 10..22, y 26..38, z 184..199 ---
    let (rl, rh) = (IVec3::new(10, 26, 184), IVec3::new(22, 38, 199));
    // Ceiling.
    g.set_rect(
        IVec3::new(rl.x - 2, rl.y - 2, rl.z - 2),
        IVec3::new(rh.x + 2, rh.y + 2, rl.z - 1),
        Some(STONE),
    );
    // −x / +y / −y walls, full.
    g.set_rect(
        IVec3::new(rl.x - 2, rl.y - 2, rl.z),
        IVec3::new(rl.x - 1, rh.y + 2, rh.z),
        Some(STONE),
    );
    g.set_rect(
        IVec3::new(rl.x - 2, rl.y - 2, rl.z),
        IVec3::new(rh.x + 2, rl.y - 1, rh.z),
        Some(STONE),
    );
    g.set_rect(
        IVec3::new(rl.x - 2, rh.y + 1, rl.z),
        IVec3::new(rh.x + 2, rh.y + 2, rh.z),
        Some(STONE),
    );
    // +x wall with a doorway (y 30..34, floor-height): three segments
    // AROUND the gap instead of carving it out afterwards.
    g.set_rect(
        IVec3::new(rh.x + 1, rl.y - 2, rl.z),
        IVec3::new(rh.x + 2, 29, rh.z),
        Some(STONE),
    );
    g.set_rect(
        IVec3::new(rh.x + 1, 35, rl.z),
        IVec3::new(rh.x + 2, rh.y + 2, rh.z),
        Some(STONE),
    );
    g.set_rect(
        IVec3::new(rh.x + 1, 30, rl.z),
        IVec3::new(rh.x + 2, 34, 191),
        Some(STONE),
    ); // lintel above the doorway

    // --- Glass cavern, interior x 62..90, y 14..50, z 172..199 ---
    let (cl, ch) = (IVec3::new(62, 14, 172), IVec3::new(90, 50, 199));
    g.set_rect(
        IVec3::new(cl.x - 2, cl.y - 2, cl.z - 2),
        IVec3::new(ch.x + 2, ch.y + 2, cl.z - 1),
        Some(GLASS),
    );
    g.set_rect(
        IVec3::new(ch.x + 1, cl.y - 2, cl.z),
        IVec3::new(ch.x + 2, ch.y + 2, ch.z),
        Some(GLASS),
    );
    g.set_rect(
        IVec3::new(cl.x - 2, cl.y - 2, cl.z),
        IVec3::new(ch.x + 2, cl.y - 1, ch.z),
        Some(GLASS),
    );
    g.set_rect(
        IVec3::new(cl.x - 2, ch.y + 1, cl.z),
        IVec3::new(ch.x + 2, ch.y + 2, ch.z),
        Some(GLASS),
    );
    // −x wall with the entrance (y 28..36, floor-height).
    g.set_rect(
        IVec3::new(cl.x - 2, cl.y - 2, cl.z),
        IVec3::new(cl.x - 1, 27, ch.z),
        Some(GLASS),
    );
    g.set_rect(
        IVec3::new(cl.x - 2, 37, cl.z),
        IVec3::new(cl.x - 1, ch.y + 2, ch.z),
        Some(GLASS),
    );
    g.set_rect(
        IVec3::new(cl.x - 2, 28, cl.z),
        IVec3::new(cl.x - 1, 36, 189),
        Some(GLASS),
    );

    g.bake(roxlap_scene::BakeMode::Directional);
    scene
}

impl DemoScene for AudioScene {
    fn name(&self) -> &'static str {
        "Audio"
    }

    fn controls(&self) -> &'static str {
        "WASD+mouse fly — the HUD numbers are live: duck into the room, \
         fly through the glass cavern (build with --features audio to hear it)"
    }

    fn enter(&mut self, ctx: &mut SceneCtx) {
        // The SAME colour→material map drives rendering: the cavern
        // renders as actual glass.
        ctx.renderer
            .define_material(GLASS_ID, Material::alpha_blend(150));
        ctx.renderer
            .set_terrain_materials(&[(GLASS.rgb_part(), GLASS_ID)]);
        self.prev_eye = None;
        // Force an immediate recompute on (re-)entry so the HUD never
        // shows the previous visit's numbers for the first half-second.
        self.cavity_timer = f64::MAX;
        self.source_timer = f64::MAX;

        #[cfg(feature = "audio")]
        {
            self.playback = Playback::start(&self.scene, &self.acfg);
        }
    }

    fn start_pose(&self) -> CameraPose {
        CameraPose {
            pos: START_POS,
            yaw: std::f64::consts::FRAC_PI_2, // facing +y, all three zones ahead
            pitch: 0.0,
        }
    }

    fn update(&mut self, ctx: &mut SceneCtx, dt: f64) {
        // Collision fly (the transparency scene's pattern).
        if ctx.cam.pos != self.last_eye {
            self.body.teleport(
                DVec3::from(ctx.cam.pos) + DVec3::new(0.0, 0.0, self.body.def().eye_height),
            );
        }
        self.body.def_mut().fly_speed = crate::scene_api::MOVE_SPEED
            * if ctx.input.fast {
                crate::scene_api::FAST_MULT
            } else {
                1.0
            };
        let wish = DVec3::from(ctx.cam.wish_dir(ctx.input));
        self.body.walk(
            &self.scene,
            dt,
            WalkInput {
                wish,
                ..WalkInput::default()
            },
        );
        ctx.cam.pos = self.body.eye_pos().into();
        self.last_eye = ctx.cam.pos;

        let listener = DVec3::from(ctx.cam.pos);

        // Listener velocity for Doppler (teleport-guarded, per frame).
        let vel = match self.prev_eye.replace(listener) {
            Some(p) if dt > 1e-4 => {
                let v = (listener - p) / dt;
                if v.length() > 200.0 {
                    DVec3::ZERO
                } else {
                    v
                }
            }
            _ => DVec3::ZERO,
        };
        self.listener_speed = vel.length();
        for (i, (pos, _)) in SOURCES.iter().enumerate() {
            self.doppler[i] = doppler_factor(
                DVec3::from(*pos),
                DVec3::ZERO,
                listener,
                vel,
                DEFAULT_SPEED_OF_SOUND,
            );
        }

        // Throttled parameter recomputes — timed, with the material
        // tables ACTIVE: this line is the AU2 profiling readout.
        self.cavity_timer += dt;
        self.source_timer += dt;
        let run_sources = self.source_timer >= 1.0 / SOURCE_HZ;
        let run_cavity = self.cavity_timer >= 1.0 / CAVITY_HZ;
        if run_sources || run_cavity {
            let t0 = Instant::now();
            if run_sources {
                self.source_timer = 0.0;
                for (i, (pos, _)) in SOURCES.iter().enumerate() {
                    self.src[i] =
                        source_acoustics(&self.scene, DVec3::from(*pos), listener, &self.acfg);
                }
            }
            if run_cavity {
                self.cavity_timer = 0.0;
                self.env = Some(self.cavity.update(&self.scene, listener));
            }
            self.acoustics_us = t0.elapsed().as_secs_f64() * 1e6;
        }

        // Audible half (feature `audio`): mirror the numbers into kira.
        #[cfg(feature = "audio")]
        if let Some(p) = self.playback.as_mut() {
            p.audio
                .set_listener(listener, listener_orientation(&ctx.cam.camera()));
            for (i, &id) in p.hums.iter().enumerate() {
                if run_sources {
                    p.audio.apply_source(id, &self.src[i]);
                }
                p.audio.set_source_pitch(id, self.doppler[i]);
            }
            if run_cavity {
                if let Some(env) = &self.env {
                    p.audio.apply_listener(env);
                }
            }
        }

        ctx.renderer.tick(&ctx.cam.camera(), dt);
    }

    fn render(&mut self, ctx: &mut SceneCtx) {
        let settings = opticast_settings(ctx.size, ctx.scan_dist);
        let frame = frame_params(ctx.engine, &settings, ctx.scan_dist);
        let camera = ctx.cam.camera();
        ctx.renderer.render(&mut self.scene, &camera, &frame);
    }

    fn exit(&mut self, _ctx: &mut SceneCtx) {
        #[cfg(feature = "audio")]
        {
            self.playback = None; // drops kira: audio thread torn down
        }
    }

    fn hud_lines(&self) -> Vec<String> {
        let mut out: Vec<String> = SOURCES
            .iter()
            .enumerate()
            .map(|(i, (_, label))| {
                let a = &self.src[i];
                format!(
                    "{label}: T={:.2} gain={:+.1}dB cutoff={:.1}kHz doppler={:.2}",
                    a.transmission,
                    a.gain_db,
                    a.lowpass_cutoff_hz / 1000.0,
                    self.doppler[i],
                )
            })
            .collect();
        if let Some(e) = &self.env {
            out.push(format!(
                "listener: open={:.2} feedback={:.2} damp={:.2} mix={:.2} (glass damps 0.10)",
                e.openness, e.reverb_feedback, e.reverb_damping, e.reverb_mix
            ));
        }
        out.push(format!(
            "acoustics: {:.0} µs/update (material tables ACTIVE) | v={:.0} u/s",
            self.acoustics_us, self.listener_speed
        ));
        #[cfg(not(feature = "audio"))]
        out.push("silent build — `--features audio` to hear it".to_string());
        out
    }
}

/// kira wants the listener as a quaternion; build it from the camera
/// basis (right, up = −down, back = −forward) — the cave demo's
/// convention.
#[cfg(feature = "audio")]
fn listener_orientation(cam: &roxlap_core::Camera) -> glam::DQuat {
    let right = DVec3::from(cam.right);
    let down = DVec3::from(cam.down);
    let forward = DVec3::from(cam.forward);
    glam::DQuat::from_mat3(&glam::DMat3::from_cols(right, -down, -forward))
}

#[cfg(feature = "audio")]
impl Playback {
    /// Open the default device and start the three zone hums, each
    /// born already shaded for the rock between it and the start pose.
    /// `None` (silent scene) when no device is available.
    fn start(scene: &Scene, acfg: &AcousticsConfig) -> Option<Self> {
        let mut audio = match KiraAudio::new() {
            Ok(a) => a,
            Err(e) => {
                eprintln!("roxlap-scene-demo: audio disabled ({e})");
                return None;
            }
        };
        let hum = audio.register(&synth::hum(44_100));
        let start = DVec3::from(START_POS);
        let mut hums = Vec::with_capacity(3);
        for (pos, _) in SOURCES {
            let p = DVec3::from(pos);
            let a = source_acoustics(scene, p, start, acfg);
            hums.push(audio.play_loop(hum, p, Some(&a))?);
        }
        let hums: [SourceId; 3] = hums.try_into().ok()?;
        Some(Self { audio, hums })
    }
}
