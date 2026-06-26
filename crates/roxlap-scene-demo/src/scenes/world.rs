//! The **World** scene (DS.1): streaming hills terrain + a rotating ship,
//! flown with collision. Showcases the scene-graph, chunk streaming,
//! multi-grid composition, and LOD billboards.
//!
//! Controls: WASD+mouse fly (collision) · `R` ship spin · `B` LOD
//! billboards · `T` streaming telemetry · `H` high-altitude vantage.

use roxlap_core::opticast::OpticastSettings;
use roxlap_render::FrameParams;
use roxlap_scene::CHUNK_SIZE_XY;
use winit::keyboard::KeyCode;

use crate::collision;
use crate::scene::{build_demo, SceneAndCamera, StreamingBakeTracker};
use crate::scene_api::{CameraPose, DemoScene, SceneCtx, SceneInput};

/// Spawn pose `build_demo` places the camera at (looking +y).
const SPAWN_POS: [f64; 3] = [0.0, -120.0, 50.0];

pub struct WorldScene {
    world: SceneAndCamera,
    bake: StreamingBakeTracker,
    /// Saved pose for the `H` high-altitude A/B toggle.
    saved_pose: Option<CameraPose>,
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
            saved_pose: None,
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

    fn on_input(&mut self, ctx: &mut SceneCtx, ev: &SceneInput) {
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
            KeyCode::KeyH => {
                if let Some(p) = self.saved_pose.take() {
                    ctx.cam.pos = p.pos;
                    ctx.cam.yaw = p.yaw;
                    ctx.cam.pitch = p.pitch;
                    eprintln!("camera restored to {:?}", p.pos);
                } else {
                    self.saved_pose = Some(CameraPose {
                        pos: ctx.cam.pos,
                        yaw: ctx.cam.yaw,
                        pitch: ctx.cam.pitch,
                    });
                    // High above the centred ground (z-down → very negative
                    // z = high up), looking ~40° down.
                    ctx.cam.pos = [0.0, 0.0, -800.0];
                    ctx.cam.pitch = 0.7;
                    eprintln!("camera → high-altitude top-down (H again to return)");
                }
            }
            _ => {}
        }
    }

    fn render(&mut self, ctx: &mut SceneCtx) {
        let mut settings = OpticastSettings::for_oracle_framebuffer(ctx.size.0, ctx.size.1);
        settings.max_scan_dist = ctx.scan_dist;
        settings.mip_levels = 6;
        settings.mip_scan_dist = 64;
        #[allow(clippy::cast_sign_loss)]
        let chunks_visible = (ctx.scan_dist.max(1) as u32) / CHUNK_SIZE_XY + 4;
        let frame = FrameParams {
            settings: &settings,
            sky_color: ctx.engine.sky_color(),
            sky: ctx.engine.sky(),
            fog_color: ctx.engine.sky_color(),
            fog_max_scan_dist: ctx.scan_dist,
            treat_z_max_as_air: true,
            gpu_mip_scan_dist: 64.0,
            gpu_max_outer_steps: chunks_visible,
            gpu_fov_y_rad: 60.0_f32.to_radians(),
            draw_sprites: true,
            side_shades: ctx.engine.side_shades(),
        };
        let camera = ctx.cam.camera();
        ctx.renderer.render(&mut self.world.scene, &camera, &frame);
    }
}
