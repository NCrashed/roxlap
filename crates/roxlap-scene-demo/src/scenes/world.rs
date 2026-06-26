//! The **World** scene (DS.1): streaming hills terrain + a rotating ship,
//! flown with collision. Showcases the scene-graph, chunk streaming,
//! multi-grid composition, and LOD billboards.
//!
//! Controls: WASD+mouse fly (collision) · `R` ship spin · `B` LOD
//! billboards · `T` streaming telemetry.

use winit::keyboard::KeyCode;

use crate::collision;
use crate::scene::{build_demo, SceneAndCamera, StreamingBakeTracker};
use crate::scene_api::{
    frame_params, opticast_settings, CameraPose, DemoScene, SceneCtx, SceneInput,
};

/// Spawn pose `build_demo` places the camera at (looking +y).
const SPAWN_POS: [f64; 3] = [0.0, -120.0, 50.0];

pub struct WorldScene {
    world: SceneAndCamera,
    bake: StreamingBakeTracker,
}

impl WorldScene {
    #[must_use]
    pub fn new() -> Self {
        let world = build_demo();
        if world.streaming_enabled {
            eprintln!("streaming hills active (T: chunks/pending, ROXLAP_STATIC=1 for static)");
        }
        Self {
            world,
            bake: StreamingBakeTracker::new(),
        }
    }
}

impl DemoScene for WorldScene {
    fn name(&self) -> &'static str {
        "World"
    }

    fn controls(&self) -> &'static str {
        "WASD+mouse fly · R: ship spin · B: LOD billboards · T: streaming · H: top-down"
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
        // Collision-checked fly: propose a step from held keys, slide it
        // against the terrain.
        let step = ctx.cam.fly_delta(ctx.input, dt);
        if step != [0.0; 3] {
            collision::slide_with_collision(&self.world.scene, &mut ctx.cam.pos, step);
        }
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
