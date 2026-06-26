//! The `Empty` placeholder scene (DS.0): an empty world that renders just
//! the sky. Proves the [`DemoScene`] contract is implementable + drives
//! the host scaffold until the real scenes land (DS.1+).

use roxlap_core::opticast::OpticastSettings;
use roxlap_render::FrameParams;
use roxlap_scene::{Scene, CHUNK_SIZE_XY};

use crate::scene_api::{CameraPose, DemoScene, SceneCtx};

/// An empty scene — no grids, no sprites; the renderer fills the sky.
pub struct EmptyScene {
    scene: Scene,
}

impl EmptyScene {
    #[must_use]
    pub fn new() -> Self {
        Self {
            scene: Scene::new(),
        }
    }
}

impl DemoScene for EmptyScene {
    fn name(&self) -> &'static str {
        "Empty"
    }

    fn controls(&self) -> &'static str {
        "WASD + mouse to fly · Tab: scenes · F1: HUD"
    }

    fn enter(&mut self, _ctx: &mut SceneCtx) {}

    fn start_pose(&self) -> CameraPose {
        CameraPose {
            pos: [0.0, -120.0, 50.0],
            yaw: std::f64::consts::FRAC_PI_2, // look +y
            pitch: 0.0,
        }
    }

    fn update(&mut self, ctx: &mut SceneCtx, dt: f64) {
        ctx.cam.fly_free(ctx.input, dt);
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
        ctx.renderer.render(&mut self.scene, &camera, &frame);
    }
}
