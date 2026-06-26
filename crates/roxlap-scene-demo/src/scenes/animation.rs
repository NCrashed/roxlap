//! The **Animation** scene (DS.3): the KFA swinging-arm character + a
//! second `coco` driven by the attachment runtime with a procedural flame
//! voxel clip on its arm. Showcases KFA skeletal animation, RKC v3, voxel
//! clips, and the attachment runtime, side by side.
//!
//! The `.rkc`/`.kfa` dump tooling (`ROXLAP_RKC` / `ROXLAP_RKC_DUMP` /
//! `ROXLAP_KFA_DUMP`) runs when the scene is constructed (via `build_kfa`).

use roxlap_render::{CharacterId, KfaSprite};
use roxlap_scene::Scene;

use crate::scene_api::{frame_params, opticast_settings, CameraPose, DemoScene, SceneCtx};
use crate::{build_kfa, flame_character};

pub struct AnimationScene {
    /// Empty world — the limbs + character draw via the sprite pass.
    scene: Scene,
    /// The animsprite-driven swinging arm (KFA path), re-posed each frame.
    kfa: Vec<KfaSprite>,
    /// The flame coco (attachment runtime), advanced each frame.
    flame_char: Option<CharacterId>,
}

impl AnimationScene {
    #[must_use]
    pub fn new() -> Self {
        Self {
            scene: Scene::new(),
            // `build_kfa` also runs the optional `.rkc`/`.kfa` dump tooling.
            kfa: build_kfa(),
            flame_char: None,
        }
    }
}

impl DemoScene for AnimationScene {
    fn name(&self) -> &'static str {
        "Animation"
    }

    fn controls(&self) -> &'static str {
        "WASD+mouse fly — KFA arm + a flame-clip character animate on their own"
    }

    fn start_pose(&self) -> CameraPose {
        CameraPose {
            pos: [0.0, -120.0, 50.0],
            yaw: std::f64::consts::FRAC_PI_2,
            pitch: 0.0,
        }
    }

    fn enter(&mut self, ctx: &mut SceneCtx) {
        // The host reset the content layers before `enter`; register the KFA
        // limbs first, then the flame character (which appends its own
        // models + clips).
        if !self.kfa.is_empty() {
            ctx.renderer.set_kfa_sprites(&mut self.kfa);
            eprintln!("Animation: KFA sprite registered (animsprite-driven arm)");
        }
        if let Some(ch) = flame_character() {
            self.flame_char = Some(ctx.renderer.add_character(&ch, Some(0)));
            eprintln!("Animation: flame character registered (add_character + clip attachment)");
        }
    }

    fn update(&mut self, ctx: &mut SceneCtx, dt: f64) {
        ctx.cam.fly_free(ctx.input, dt);
        if !self.kfa.is_empty() {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let dt_ms = (dt * 1000.0) as i32;
            for k in &mut self.kfa {
                k.animsprite(dt_ms);
            }
            ctx.renderer.update_kfa_poses(&mut self.kfa);
        }
        if let Some(id) = self.flame_char {
            ctx.renderer.advance_character(id, dt);
        }
    }

    fn render(&mut self, ctx: &mut SceneCtx) {
        let settings = opticast_settings(ctx.size, ctx.scan_dist);
        let frame = frame_params(ctx.engine, &settings, ctx.scan_dist);
        let camera = ctx.cam.camera();
        ctx.renderer.render(&mut self.scene, &camera, &frame);
    }
}
