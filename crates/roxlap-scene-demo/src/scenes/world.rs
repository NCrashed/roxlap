//! The **World** scene (DS.1): streaming hills terrain + a rotating ship,
//! moved with the engine character controller (CC.3). Showcases the
//! scene-graph, chunk streaming, multi-grid composition, and LOD
//! billboards.
//!
//! Controls: WASD+mouse fly (collision) · `G` walk mode (gravity,
//! Space jumps, 1-voxel step-up) · `R` ship spin · `B` LOD billboards
//! · `T` streaming telemetry.

use glam::DVec3;
use roxlap_scene::{CharacterBody, CharacterDef, MoveMode, WalkInput};
use winit::keyboard::KeyCode;

use crate::scene::{build_demo, SceneAndCamera, StreamingBakeTracker};
use crate::scene_api::{
    frame_params, opticast_settings, CameraPose, DemoScene, SceneCtx, SceneInput, FAST_MULT,
    MOVE_SPEED,
};

/// Spawn pose `build_demo` places the camera at (looking +y).
const SPAWN_POS: [f64; 3] = [0.0, -120.0, 50.0];

/// Grounded speed — terrain-scale rather than the 64 v/s fly rate.
const WALK_SPEED: f64 = 12.0;

pub struct WorldScene {
    world: SceneAndCamera,
    bake: StreamingBakeTracker,
    /// CC.3 — the engine character controller drives movement; the
    /// camera rides at its eye. `Fly` (default) is the classic
    /// fly-with-collision; `G` drops into `Walk` (gravity + Space
    /// jumps + step-up).
    body: CharacterBody,
    /// Eye position the camera held after our last `update`. If the
    /// host moved the camera behind our back (scene switch, capture
    /// pose), we re-teleport the body instead of dragging it across
    /// the world.
    last_eye: [f64; 3],
}

impl WorldScene {
    #[must_use]
    pub fn new() -> Self {
        let world = build_demo();
        if world.streaming_enabled {
            eprintln!("streaming hills active (T: chunks/pending, ROXLAP_STATIC=1 for static)");
        }
        let mut body = CharacterBody::new(CharacterDef {
            walk_speed: WALK_SPEED,
            fly_speed: MOVE_SPEED,
            ..CharacterDef::default()
        });
        body.set_mode(MoveMode::Fly);
        Self {
            world,
            bake: StreamingBakeTracker::new(),
            body,
            last_eye: [f64::NAN; 3],
        }
    }

    /// Feet position for a camera (eye) position.
    fn feet_for_eye(&self, eye: [f64; 3]) -> DVec3 {
        DVec3::from(eye) + DVec3::new(0.0, 0.0, self.body.def().eye_height)
    }
}

impl DemoScene for WorldScene {
    fn name(&self) -> &'static str {
        "World"
    }

    fn controls(&self) -> &'static str {
        "WASD+mouse fly · G: walk/fly · R: ship spin · B: LOD billboards · T: streaming · H: top-down"
    }

    fn start_pose(&self) -> CameraPose {
        CameraPose {
            pos: SPAWN_POS,
            yaw: std::f64::consts::FRAC_PI_2,
            pitch: 0.0,
        }
    }

    fn enter(&mut self, _ctx: &mut SceneCtx) {
        // The world is built in `new`; nothing to register on the renderer.
    }

    fn update(&mut self, ctx: &mut SceneCtx, dt: f64) {
        // CC.3 — the CharacterBody moves, the camera rides its eye.
        if ctx.cam.pos != self.last_eye {
            // Host moved the camera (scene switch / capture pose):
            // re-place the body there rather than dragging it over.
            self.body.teleport(self.feet_for_eye(ctx.cam.pos));
        }
        // Shift boost tunes the def live (def_mut is the sprint API).
        self.body.def_mut().fly_speed = MOVE_SPEED * if ctx.input.fast { FAST_MULT } else { 1.0 };
        self.body.def_mut().walk_speed = WALK_SPEED * if ctx.input.fast { 2.0 } else { 1.0 };
        let wish = DVec3::from(ctx.cam.wish_dir(ctx.input));
        self.body.walk(
            &self.world.scene,
            dt,
            WalkInput {
                wish,
                jump: ctx.input.up,
            },
        );
        ctx.cam.pos = self.body.eye_pos().into();
        self.last_eye = ctx.cam.pos;

        self.world.tick_ship_spin(dt);
        if self.world.streaming_enabled {
            self.world
                .scene
                .pump_streaming(glam::DVec3::from_array(ctx.cam.pos));
            self.bake.process(&mut self.world.scene);
        }
    }

    fn on_input(&mut self, _ctx: &mut SceneCtx, ev: &SceneInput) {
        let SceneInput::Key {
            code,
            pressed: true,
        } = ev
        else {
            return;
        };
        match code {
            KeyCode::KeyG => {
                let next = match self.body.mode() {
                    MoveMode::Walk => MoveMode::Fly,
                    _ => MoveMode::Walk,
                };
                self.body.set_mode(next);
                eprintln!(
                    "move mode = {}",
                    match next {
                        MoveMode::Walk => "WALK (Space jumps, G back to fly)",
                        _ => "FLY",
                    }
                );
            }
            KeyCode::KeyR => {
                self.world.spin_enabled = !self.world.spin_enabled;
                eprintln!(
                    "ship spin = {}",
                    if self.world.spin_enabled { "ON" } else { "OFF" }
                );
            }
            KeyCode::KeyB => {
                let on = self.world.toggle_billboards_lod();
                eprintln!("S6 billboards = {}", if on { "ON" } else { "OFF" });
            }
            KeyCode::KeyT if self.world.streaming_enabled => {
                for (id, grid) in self.world.scene.grids() {
                    eprintln!(
                        "grid #{id} chunks={chunks} pending={pending} radius={r_active:.0}/{r_evict:.0}",
                        id = id.raw(),
                        chunks = grid.chunk_count(),
                        pending = grid.pending_gen.len(),
                        r_active = grid.stream_radius.r_active,
                        r_evict = grid.stream_radius.r_evict,
                    );
                }
            }
            _ => {}
        }
    }

    fn render(&mut self, ctx: &mut SceneCtx) {
        let settings = opticast_settings(ctx.size, ctx.scan_dist);
        let frame = frame_params(ctx.engine, &settings, ctx.scan_dist);
        let camera = ctx.cam.camera();
        ctx.renderer.render(&mut self.world.scene, &camera, &frame);
    }
}
