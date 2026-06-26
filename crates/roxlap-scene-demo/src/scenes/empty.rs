//! The `Empty` placeholder scene (DS.0): an empty world that renders just
//! the sky. Proves the [`DemoScene`] contract is implementable + drives
//! the host scaffold until the real scenes land (DS.1+).

use roxlap_scene::Scene;

use crate::scene_api::{frame_params, opticast_settings, CameraPose, DemoScene, SceneCtx};

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
        let settings = opticast_settings(ctx.size, ctx.scan_dist);
        let frame = frame_params(ctx.engine, &settings, ctx.scan_dist);
        let camera = ctx.cam.camera();
        ctx.renderer.render(&mut self.scene, &camera, &frame);
    }
}
