//! The **Primitives** scene (DS.4): depth-tested debug line gizmos (a wire
//! grid + box occluded by nothing here, plus always-on-top origin axes) and
//! a 2D reference image quad, drawn over an empty world. Left-click runs the
//! `pick_image` eyedropper, reporting the texel under the cursor.
//!
//! Showcases `draw_lines` / `draw_images` / `upload_image` / `pick_image`.

use roxlap_render::ImageId;
use roxlap_scene::Scene;
use winit::event::MouseButton;

use crate::scene_api::{
    frame_params, opticast_settings, CameraPose, DemoScene, SceneCtx, SceneInput,
};
use crate::{debug_overlay_lines, demo_image_sprite, make_reference_image};

pub struct PrimitivesScene {
    /// Empty world — the gizmos + image draw on top of the sky pass.
    scene: Scene,
    /// Handle of the uploaded reference texture.
    image_id: Option<ImageId>,
    /// Last cursor position (physical px), for the eyedropper.
    mouse_px: (f64, f64),
}

impl PrimitivesScene {
    #[must_use]
    pub fn new() -> Self {
        Self {
            scene: Scene::new(),
            image_id: None,
            mouse_px: (0.0, 0.0),
        }
    }
}

impl DemoScene for PrimitivesScene {
    fn name(&self) -> &'static str {
        "Primitives"
    }

    fn controls(&self) -> &'static str {
        "WASD+mouse fly · left-click runs the image eyedropper (pick_image)"
    }

    fn start_pose(&self) -> CameraPose {
        CameraPose {
            pos: [0.0, -120.0, 50.0],
            yaw: std::f64::consts::FRAC_PI_2,
            pitch: 0.0,
        }
    }

    fn enter(&mut self, ctx: &mut SceneCtx) {
        if self.image_id.is_none() {
            let (rgba, iw, ih) = make_reference_image();
            self.image_id = Some(ctx.renderer.upload_image(&rgba, iw, ih));
            eprintln!("Primitives: reference image uploaded ({iw}x{ih}); debug gizmos active");
        }
    }

    fn update(&mut self, ctx: &mut SceneCtx, dt: f64) {
        ctx.cam.fly_free(ctx.input, dt);
    }

    fn on_input(&mut self, ctx: &mut SceneCtx, ev: &SceneInput) {
        match ev {
            SceneInput::CursorMoved { x, y } => self.mouse_px = (*x, *y),
            SceneInput::Mouse {
                button: MouseButton::Left,
                pressed: true,
            } => {
                let Some(id) = self.image_id else { return };
                let cam = ctx.cam.camera();
                let (mx, my) = self.mouse_px;
                // Alpha- and occlusion-aware texel probe — the eyedropper path.
                match ctx
                    .renderer
                    .pick_image(&cam, mx, my, &[demo_image_sprite(id)])
                {
                    Some(h) => eprintln!(
                        "image hit: texel ({}, {}) uv ({:.3}, {:.3}) @ dist {:.1}",
                        h.texel.0, h.texel.1, h.uv[0], h.uv[1], h.t,
                    ),
                    None => eprintln!("image hit: none (miss / transparent / occluded)"),
                }
            }
            _ => {}
        }
    }

    fn render(&mut self, ctx: &mut SceneCtx) {
        let settings = opticast_settings(ctx.size, ctx.scan_dist);
        let frame = frame_params(ctx.engine, &settings, ctx.scan_dist);
        let camera = ctx.cam.camera();
        ctx.renderer.render(&mut self.scene, &camera, &frame);

        // Overlay the depth-tested gizmos + reference quad after the world
        // pass, before the host's egui/present.
        ctx.renderer.draw_lines(&camera, &debug_overlay_lines());
        if let Some(id) = self.image_id {
            ctx.renderer.draw_images(&camera, &[demo_image_sprite(id)]);
        }
    }
}
