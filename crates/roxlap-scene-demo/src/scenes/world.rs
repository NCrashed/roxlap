//! The **World** scene (DS.1): streaming hills terrain + a rotating ship,
//! moved with the engine character controller (CC.3). Showcases the
//! scene-graph, chunk streaming, multi-grid composition, and LOD
//! billboards.
//!
//! Controls: WASD+mouse fly (collision) · `G` walk mode (gravity,
//! Space jumps, 1-voxel step-up) · `R` ship spin · `B` LOD billboards
//! · `T` streaming telemetry.

use glam::DVec3;
use roxlap_render::{
    ActorState, BillboardActorDef, BillboardActorId, BillboardLighting, BillboardMode, Kv6,
    LoopMode, ShadowFlags, VoxColor, VoxelClip,
};
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

/// Third-person figure: 24 slab voxels tall, drawn at
/// `BillboardActorDef::scale` world units per voxel — 0.72 world
/// units, deliberately smaller than the 1.8 collision body so the
/// figure reads as a marker, not a giant.
const FIGURE_H_VOX: u32 = 24;
const FIGURE_W_VOX: u32 = 12;
const FIGURE_SCALE: f32 = 0.03;

/// Camera boom length in third person (clamped by a raycast so the
/// camera never sits inside a hill).
const BOOM_DIST: f64 = 8.0;

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
    /// CC.4 — third-person: a synthetic billboard-actor figure posed
    /// at the body, camera boomed back along −forward. `None` =
    /// first person.
    tp_actor: Option<BillboardActorId>,
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
            tp_actor: None,
        }
    }

    /// Feet position for a camera (eye) position.
    fn feet_for_eye(&self, eye: [f64; 3]) -> DVec3 {
        DVec3::from(eye) + DVec3::new(0.0, 0.0, self.body.def().eye_height)
    }

    /// One frame of the synthetic third-person figure: a flat 2-voxel
    /// slab silhouette — head, torso, and two legs whose spread is
    /// the walk phase.
    fn figure_frame(leg_spread: u32) -> Kv6 {
        const SKIN: VoxColor = VoxColor(0x80_d8_a8_78);
        const TUNIC: VoxColor = VoxColor(0x80_30_60_a0);
        const BOOTS: VoxColor = VoxColor(0x80_50_40_30);
        let cx = FIGURE_W_VOX / 2; // 6
        Kv6::from_fn(FIGURE_W_VOX, 2, FIGURE_H_VOX, |x, _y, z| {
            // Slab convention (formats/src/slab.rs): z = 0 is the
            // BOTTOM of the rendered card — flip so the crown zones
            // below land at the top.
            let z = FIGURE_H_VOX - 1 - z;
            match z {
                0..=5 => {
                    // Head: a 4-wide block centred on the body.
                    (x + 2 >= cx && x < cx + 2).then_some(SKIN)
                }
                6..=15 => {
                    // Torso + arms: 8 wide.
                    (x + 4 >= cx && x < cx + 4).then_some(TUNIC)
                }
                _ => {
                    // Legs: two 2-wide columns, `leg_spread` voxels
                    // out from centre.
                    let left = cx - 2 - leg_spread;
                    let right = cx + leg_spread;
                    ((x >= left && x < left + 2) || (x >= right && x < right + 2)).then_some(BOOTS)
                }
            }
        })
    }

    /// Register the figure's walk (2 frames) + idle (1 frame) states
    /// and spawn the actor at the body.
    fn spawn_tp_actor(&mut self, ctx: &mut SceneCtx) -> Option<BillboardActorId> {
        let clip = |frames: &[Kv6]| {
            VoxelClip::from_kv6_frames(frames, 1.0, LoopMode::Loop, &[], 220, 1)
                .expect("figure frames are non-empty + same dims")
        };
        let walk = ctx.renderer.add_voxel_clip(
            &clip(&[Self::figure_frame(3), Self::figure_frame(0)])
                .decode()
                .expect("clip decodes"),
        );
        let idle = ctx.renderer.add_voxel_clip(
            &clip(&[Self::figure_frame(1)])
                .decode()
                .expect("clip decodes"),
        );
        let def = BillboardActorDef {
            states: vec![
                ActorState {
                    name: "walk".to_owned(),
                    dirs: vec![walk],
                },
                ActorState {
                    name: "idle".to_owned(),
                    dirs: vec![idle],
                },
            ],
            mode: BillboardMode::Cylindrical,
            lighting: BillboardLighting::FaceNormal,
            speed: 1.0,
            scale: FIGURE_SCALE,
            shadows: ShadowFlags::default(),
        };
        ctx.renderer
            .add_billboard_actor(def, self.actor_pos(), ctx.cam.yaw)
    }

    /// The actor's anchor: clip pivot is the volume CENTRE, so half
    /// the figure's height above (−z of) the feet.
    #[allow(clippy::cast_possible_truncation)]
    fn actor_pos(&self) -> [f32; 3] {
        let feet = self.body.pos();
        let half_h = f64::from(FIGURE_H_VOX) * f64::from(FIGURE_SCALE) * 0.5;
        [feet.x as f32, feet.y as f32, (feet.z - half_h) as f32]
    }
}

impl DemoScene for WorldScene {
    fn name(&self) -> &'static str {
        "World"
    }

    fn controls(&self) -> &'static str {
        "WASD+mouse fly · G: walk/fly · C: third person · R: ship spin · B: LOD · T: streaming"
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

        // CC.4 — third person: pose the figure at the body, animate
        // it, and boom the camera back along −forward (raycast-
        // clamped so it never sits inside a hill).
        if let Some(actor) = self.tp_actor {
            ctx.renderer
                .set_actor_transform(actor, self.actor_pos(), ctx.cam.yaw);
            let moving = self.body.vel().truncate().length() > 0.5;
            ctx.renderer
                .set_actor_state(actor, if moving { "walk" } else { "idle" });

            // Boom BEFORE tick: the billboard must face the real
            // (boomed) camera. Ticking from the eye — which sits in
            // the actor's own xy column — fed the cylindrical facing
            // a degenerate view axis every frame.
            let eye = self.body.eye_pos();
            let back = -DVec3::from(ctx.cam.camera().forward);
            let boom = self
                .world
                .scene
                .raycast(eye, back, BOOM_DIST)
                .map_or(BOOM_DIST, |hit| (hit.t - 0.5).max(0.5));
            ctx.cam.pos = (eye + back * boom).into();
            ctx.renderer.tick(&ctx.cam.camera(), dt);
        }
        self.last_eye = ctx.cam.pos;

        self.world.tick_ship_spin(dt);
        if self.world.streaming_enabled {
            self.world
                .scene
                .pump_streaming(glam::DVec3::from_array(ctx.cam.pos));
            self.bake.process(&mut self.world.scene);
        }
    }

    fn on_input(&mut self, ctx: &mut SceneCtx, ev: &SceneInput) {
        let SceneInput::Key {
            code,
            pressed: true,
        } = ev
        else {
            return;
        };
        match code {
            KeyCode::KeyC => {
                if let Some(actor) = self.tp_actor.take() {
                    ctx.renderer.remove_billboard_actor(actor);
                    // Camera snaps back to the eye next update (the
                    // body stays authoritative through last_eye).
                    ctx.cam.pos = self.body.eye_pos().into();
                    self.last_eye = ctx.cam.pos;
                    eprintln!("view = FIRST person");
                } else {
                    self.tp_actor = self.spawn_tp_actor(ctx);
                    eprintln!("view = THIRD person (C back to first)");
                }
            }
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
