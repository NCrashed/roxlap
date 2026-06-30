//! The **Sprites** scene (DS.2): a checkerboarded green/red `coco` field, a
//! shoot-to-carve dense blob, and a streaming "spinner" ring. Showcases
//! sprite models, `set_sprites`, the dynamic add/remove API, and sprite
//! carving. The sprites float in an empty world (sky background).
//!
//! Controls: click to grab, then left-click shoots a sphere out of the
//! blob ahead · `G` structurally carves the red model's next z-layer.

use roxlap_render::{Sprite, SpriteInstanceDesc, SpriteInstanceId, SpriteModelId, SpriteSet};
use roxlap_scene::Scene;
use winit::event::MouseButton;
use winit::keyboard::KeyCode;

use crate::scene_api::{
    frame_params, opticast_settings, CameraPose, DemoScene, SceneCtx, SceneInput,
};
use crate::{build_sprites, CarveTarget, Spinner, TARGET_WORLD};

pub struct SpritesScene {
    /// Empty world — the sprites are drawn by the renderer's sprite pass.
    scene: Scene,
    /// Base `coco` sprite(s) the green/red field expands from.
    base: Vec<Sprite>,
    /// The dense blob shot-to-carve subtracts spheres from.
    carve_target: CarveTarget,
    /// Handle of the carve-target's registered model (for the incremental
    /// shoot-to-carve refresh).
    carve_id: Option<SpriteModelId>,
    /// Streaming sprite ring exercising the incremental add/remove API.
    spinner: Spinner,
    /// A row of dynamic instances of the **same** green `coco` model, each
    /// re-tinted to a cycling rainbow hue every frame — the per-instance tint
    /// showcase (`set_sprite_instance_tint`).
    tinted: Vec<SpriteInstanceId>,
    /// Seconds elapsed, driving the tint cycle.
    tint_clock: f64,
}

/// HSV (`s = v = 1`) → packed `0x00RRGGBB`, `h` in `0..1`. A tiny rainbow
/// helper for the tint showcase.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn hue_rgb(h: f32) -> u32 {
    let h6 = (h.rem_euclid(1.0)) * 6.0;
    let i = h6.floor() as i32;
    let f = h6 - i as f32;
    let q = ((1.0 - f) * 255.0) as u32;
    let t = (f * 255.0) as u32;
    let (r, g, b) = match i.rem_euclid(6) {
        0 => (255, t, 0),
        1 => (q, 255, 0),
        2 => (0, 255, t),
        3 => (0, q, 255),
        4 => (t, 0, 255),
        _ => (255, 0, q),
    };
    (r << 16) | (g << 8) | b
}

impl SpritesScene {
    #[must_use]
    pub fn new() -> Self {
        Self {
            scene: Scene::new(),
            base: build_sprites(),
            carve_target: CarveTarget::new(),
            carve_id: None,
            spinner: Spinner::default(),
            tinted: Vec::new(),
            tint_clock: 0.0,
        }
    }

    /// The checkerboarded green/red `coco` field (model indices 0/1).
    fn coco_field() -> Vec<SpriteInstanceDesc> {
        let n: i32 = std::env::var("ROXLAP_SPRITE_GRID")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4);
        let spacing = 40.0_f32;
        let mut instances = Vec::new();
        for iy in 0..n {
            for ix in 0..n {
                #[allow(clippy::cast_precision_loss)]
                let dx = (ix as f32 - (n as f32 - 1.0) * 0.5) * spacing;
                #[allow(clippy::cast_precision_loss)]
                let dy = (iy as f32 - (n as f32 - 1.0) * 0.5) * spacing;
                let model = usize::from((ix + iy) % 2 != 0); // 0 green, 1 red
                instances.push(SpriteInstanceDesc {
                    model,
                    pos: [dx, dy, 40.0],
                });
            }
        }
        instances
    }

    /// Build the sprite set (green + red `coco` field + the carve blob).
    /// Returns the set and the blob's model index (→ `carve_id`).
    fn build_set(&self) -> Option<(SpriteSet, usize)> {
        let mut models = Vec::new();
        let mut instances = Vec::new();
        let mut carve_model = None;

        if let Some(base) = self.base.first() {
            // Red variant: recolour every KV6 voxel (keep alpha), once.
            let mut red = base.clone();
            for v in &mut red.kv6.voxels {
                v.col = (v.col & 0xFF00_0000) | 0x00FF_0000;
            }
            models.push(base.clone()); // model 0: green
            models.push(red); // model 1: red — the `G`-carve target
            carve_model = Some(1);
            instances.extend(Self::coco_field());
        }

        // The shoot-to-carve dense blob.
        let target_model = models.len();
        models.push(self.carve_target.sprite.clone());
        instances.push(SpriteInstanceDesc {
            model: target_model,
            pos: TARGET_WORLD,
        });

        if models.is_empty() {
            return None;
        }
        Some((
            SpriteSet {
                models,
                instances,
                carve_model,
            },
            target_model,
        ))
    }

    /// Cast the centre-screen ray; if it hits the carve blob, subtract a
    /// sphere and re-upload only that model.
    fn fire(&mut self, ctx: &mut SceneCtx) {
        let (cx, cy) = (f64::from(ctx.size.0) * 0.5, f64::from(ctx.size.1) * 0.5);
        let cam = ctx.cam.camera();
        let Some(ray) = ctx.renderer.view_ray(&cam, cx, cy) else {
            return;
        };
        let Some(hit) = self
            .carve_target
            .raycast(ray.origin.to_array(), ray.dir.to_array())
        else {
            eprintln!("shot missed the carve target");
            return;
        };
        let removed = self.carve_target.carve(hit);
        eprintln!(
            "hit target at local ({}, {}, {}) — carved {removed} voxels",
            hit[0], hit[1], hit[2],
        );
        if let Some(id) = self.carve_id {
            ctx.renderer
                .refresh_sprite_model(id, &self.carve_target.sprite.kv6);
        }
    }
}

impl DemoScene for SpritesScene {
    fn name(&self) -> &'static str {
        "Sprites"
    }

    fn controls(&self) -> &'static str {
        "click to grab · left-click shoots the blob · G: carve the red coco · rainbow row = per-instance tint"
    }

    fn start_pose(&self) -> CameraPose {
        CameraPose {
            pos: [0.0, -120.0, 50.0],
            yaw: std::f64::consts::FRAC_PI_2,
            pitch: 0.0,
        }
    }

    fn enter(&mut self, ctx: &mut SceneCtx) {
        if let Some((set, target_model)) = self.build_set() {
            eprintln!(
                "Sprites: {} instances, {} models (left-click shoots the blob; G carves the red coco; \
                 a colour-sphere spinner streams in/out)",
                set.instances.len(),
                set.models.len(),
            );
            let ids = ctx.renderer.set_sprites(&set);
            self.carve_id = ids.get(target_model).copied();
            self.spinner.reset();

            // Per-instance tint showcase: a row of dynamic instances of the
            // SAME green `coco` model (id 0), each recoloured every frame via
            // `set_sprite_instance_tint` — no extra models, just a tint.
            self.tinted.clear();
            self.tint_clock = 0.0;
            if let Some(&green) = ids.first() {
                let n = 6i32;
                for i in 0..n {
                    #[allow(clippy::cast_precision_loss)]
                    let x = (i as f32 - (n as f32 - 1.0) * 0.5) * 44.0;
                    // A row right in front of the start camera (looking +y),
                    // near eye level, ahead of the green/red field.
                    let id = ctx.renderer.add_sprite_instance(green, [x, -55.0, 45.0]);
                    self.tinted.push(id);
                }
            }
        }
    }

    fn update(&mut self, ctx: &mut SceneCtx, dt: f64) {
        ctx.cam.fly_free(ctx.input, dt);
        self.spinner.update(ctx.renderer, dt);

        // Cycle each tinted instance through the rainbow (staggered by index),
        // proving per-instance `set_sprite_instance_tint` on one shared model.
        self.tint_clock += dt;
        let n = self.tinted.len().max(1) as f64;
        for (i, &id) in self.tinted.iter().enumerate() {
            #[allow(clippy::cast_possible_truncation)]
            let hue = (self.tint_clock * 0.25 + i as f64 / n) as f32;
            ctx.renderer.set_sprite_instance_tint(id, hue_rgb(hue));
        }
    }

    fn on_input(&mut self, ctx: &mut SceneCtx, ev: &SceneInput) {
        match ev {
            SceneInput::Mouse {
                button: MouseButton::Left,
                pressed: true,
            } => self.fire(ctx),
            SceneInput::Key {
                code: KeyCode::KeyG,
                pressed: true,
            } => {
                let removed = ctx.renderer.carve_active_sprite();
                if removed > 0 {
                    eprintln!("carved sprite z-layer ({removed} voxels)");
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
    }
}
