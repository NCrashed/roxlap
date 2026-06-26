//! The **Picking** scene (DS.4): a top-down view over the streaming hills
//! with a mouse-driven cursor that snaps to a ground plane, and left-click
//! to drop a marker at the picked surface voxel (or the ground-plane
//! fallback). Showcases `view_ray` / `pick` (screen→world).
//!
//! Controls: move the mouse to place the cursor · left-click drops a
//! marker · WASD pans the top-down camera.

use roxlap_core::opticast::OpticastSettings;
use roxlap_render::{FrameParams, Sprite, SpriteInstanceDesc, SpriteSet};
use roxlap_scene::CHUNK_SIZE_XY;
use winit::event::MouseButton;

use crate::scene::{build_demo, SceneAndCamera, StreamingBakeTracker};
use crate::scene_api::{CameraPose, DemoScene, SceneCtx, SceneInput};
use crate::{build_sprites, plane_hit, PICK_CAM_PITCH, PICK_CAM_POS, PICK_GROUND_Z};

pub struct PickingScene {
    /// Streaming hills to pick against.
    world: SceneAndCamera,
    bake: StreamingBakeTracker,
    /// `[cursor (cyan), marker (yellow)]`, recoloured from the base coco.
    pick_models: Vec<Sprite>,
    /// Ground-plane point under the cursor this frame.
    cursor_world: [f32; 3],
    /// Dropped marker positions.
    placed: Vec<[f32; 3]>,
    /// Last cursor position (physical px).
    mouse_px: (f64, f64),
}

impl PickingScene {
    #[must_use]
    pub fn new() -> Self {
        let base = build_sprites();
        let pick_models = base.first().map_or_else(Vec::new, |base| {
            let recolor = |rgb: u32| {
                let mut s = base.clone();
                for v in &mut s.kv6.voxels {
                    v.col = (v.col & 0xFF00_0000) | rgb;
                }
                s
            };
            vec![recolor(0x0000_FFFF), recolor(0x00FF_FF00)] // cyan cursor, yellow marker
        });
        Self {
            world: build_demo(),
            bake: StreamingBakeTracker::new(),
            pick_models,
            cursor_world: [0.0, 0.0, PICK_GROUND_Z as f32],
            placed: Vec::new(),
            mouse_px: (0.0, 0.0),
        }
    }
}

impl DemoScene for PickingScene {
    fn name(&self) -> &'static str {
        "Picking"
    }

    fn controls(&self) -> &'static str {
        "move mouse to place cursor · left-click drops a marker · WASD pans"
    }

    fn start_pose(&self) -> CameraPose {
        CameraPose {
            pos: PICK_CAM_POS,
            yaw: std::f64::consts::FRAC_PI_2,
            pitch: PICK_CAM_PITCH,
        }
    }

    fn enter(&mut self, _ctx: &mut SceneCtx) {
        if self.pick_models.is_empty() {
            eprintln!("Picking: no base sprite loaded — cursor disabled");
        }
    }

    fn update(&mut self, ctx: &mut SceneCtx, dt: f64) {
        ctx.cam.fly_free(ctx.input, dt);
        if self.world.streaming_enabled {
            self.world
                .scene
                .pump_streaming(glam::DVec3::from_array(ctx.cam.pos));
            self.bake.process(&mut self.world.scene);
        }
        if self.pick_models.len() < 2 {
            return;
        }
        // Hover cursor: unproject the mouse onto the ground plane.
        let cam = ctx.cam.camera();
        if let Some(ray) = ctx
            .renderer
            .view_ray(&cam, self.mouse_px.0, self.mouse_px.1)
        {
            if let Some(w) = plane_hit(ray.origin.to_array(), ray.dir.to_array(), PICK_GROUND_Z) {
                self.cursor_world = w;
            }
        }
        // Rebuild the cursor + marker sprite field (cursor follows the mouse).
        let mut instances = vec![SpriteInstanceDesc {
            model: 0,
            pos: self.cursor_world,
        }];
        for p in &self.placed {
            instances.push(SpriteInstanceDesc { model: 1, pos: *p });
        }
        ctx.renderer.set_sprites(&SpriteSet {
            models: self.pick_models.clone(),
            instances,
            carve_model: None,
        });
    }

    fn on_input(&mut self, ctx: &mut SceneCtx, ev: &SceneInput) {
        match ev {
            SceneInput::CursorMoved { x, y } => self.mouse_px = (*x, *y),
            SceneInput::Mouse {
                button: MouseButton::Left,
                pressed: true,
            } => {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let (px, py) = (self.mouse_px.0 as u32, self.mouse_px.1 as u32);
                let cam = ctx.cam.camera();
                // One library call: unproject → depth read → owning grid +
                // grid-local voxel. Sky clicks fall back to the ground plane.
                match ctx.renderer.pick(&self.world.scene, &cam, px, py) {
                    Some(h) => {
                        self.placed.push(h.world);
                        eprintln!(
                            "surface [{:.1}, {:.1}, {:.1}] → grid #{} voxel ({}, {}, {}) ({} placed)",
                            h.world[0], h.world[1], h.world[2], h.grid.raw(),
                            h.voxel.x, h.voxel.y, h.voxel.z, self.placed.len(),
                        );
                    }
                    None => {
                        let w = self.cursor_world;
                        self.placed.push(w);
                        eprintln!(
                            "ground-plane [{:.1}, {:.1}, {:.1}] ({} placed)",
                            w[0],
                            w[1],
                            w[2],
                            self.placed.len(),
                        );
                    }
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

    fn hud_lines(&self) -> Vec<String> {
        vec![format!("markers placed: {}", self.placed.len())]
    }
}
