//! The **Transparency** scene (TV stage): translucent voxel sprites over
//! opaque ones, showcasing the per-pixel front-to-back compositing the DDA
//! renderer enables — `AlphaBlend` glass/smoke and `Additive` glow.
//!
//! Layout (camera looks +y): an opaque brick backdrop with a cyan **glass
//! pane** in front of it (the backdrop tints through), an **additive glow**
//! aura off to the left (a spell-like emissive that brightens whatever is
//! behind it), and a grey **smoke puff** whose per-instance `alpha_mul`
//! pulses each frame — fading without re-uploading its volume.
//!
//! Each effect is one [`Material`] in the renderer's global palette
//! (`define_material`) referenced by an instance via
//! `set_sprite_instance_material`. See `PORTING-TRANSPARENCY.md`.

use roxlap_render::{DynSpriteTransform, Kv6, Material, SceneRenderer, SpriteInstanceId};
use roxlap_scene::Scene;

use crate::scene_api::{frame_params, opticast_settings, CameraPose, DemoScene, SceneCtx};

/// Palette ids (id 0 is the reserved opaque material).
const MAT_GLASS: u8 = 1;
const MAT_GLOW: u8 = 2;
const MAT_SMOKE: u8 = 3;

pub struct TransparencyScene {
    /// Empty world — every surface here is a sprite (sky background).
    scene: Scene,
    /// The smoke instance whose `alpha_mul` pulses each frame.
    smoke: Option<SpriteInstanceId>,
    /// Seconds since `enter`, driving the smoke pulse.
    clock: f64,
}

impl TransparencyScene {
    #[must_use]
    pub fn new() -> Self {
        Self {
            scene: Scene::new(),
            smoke: None,
            clock: 0.0,
        }
    }

    /// An axis-aligned pose at `pos` (identity model→world basis).
    fn pose(pos: [f32; 3]) -> DynSpriteTransform {
        DynSpriteTransform {
            pos,
            right: [1.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
            forward: [0.0, 0.0, 1.0],
        }
    }

    /// Register one model + a posed instance, returning the instance id.
    fn spawn(r: &mut SceneRenderer, kv6: &Kv6, pos: [f32; 3]) -> SpriteInstanceId {
        let model = r.add_sprite_model(kv6);
        r.add_sprite_instance_posed(model, Self::pose(pos))
    }

    /// A **mixed-material** window panel (TV.3): an opaque brown frame around
    /// translucent cyan glass — built as one colour-coded model. Thin in y so
    /// it faces the camera. The glass colour maps to `MAT_GLASS`; the frame
    /// colour isn't in the map, so it stays opaque (material 0).
    fn build_window() -> Kv6 {
        const FRAME: u32 = 0x80_6A_4A_2A; // opaque brown
        Kv6::from_fn(30, 3, 30, |x, _y, z| {
            let edge = x < 4 || x >= 26 || z < 4 || z >= 26;
            Some(if edge { FRAME } else { GLASS_RGB })
        })
    }
}

/// The glass voxel colour shared by the window's glass + its material map.
const GLASS_RGB: u32 = 0x80_50_C0_E0;

impl DemoScene for TransparencyScene {
    fn name(&self) -> &'static str {
        "Transparency"
    }

    fn controls(&self) -> &'static str {
        "WASD+mouse fly — glass (alpha) tints the backdrop · glow (additive) brightens · smoke pulses"
    }

    fn start_pose(&self) -> CameraPose {
        CameraPose {
            pos: [0.0, -120.0, 40.0],
            yaw: std::f64::consts::FRAC_PI_2,
            pitch: 0.0,
        }
    }

    fn enter(&mut self, ctx: &mut SceneCtx) {
        let r = &mut *ctx.renderer;

        // Define the global material palette: a translucent cyan glass, an
        // additive glow, and a semi-transparent smoke.
        r.define_material(MAT_GLASS, Material::alpha_blend(110));
        r.define_material(MAT_GLOW, Material::additive(200));
        r.define_material(MAT_SMOKE, Material::alpha_blend(150));

        // Opaque brick backdrop (reference surface the glass tints).
        let _backdrop = Self::spawn(r, &Kv6::solid_cube(22, 0x80_B0_50_40), [40.0, 90.0, 40.0]);

        // Cyan glass pane in front of the backdrop (wide in x, tall in z,
        // thin in y — it faces the camera). AlphaBlend over the backdrop.
        let glass = Self::spawn(
            r,
            &Kv6::solid_box(28, 3, 28, 0x80_50_C0_E0),
            [40.0, 55.0, 38.0],
        );
        r.set_sprite_instance_material(glass, MAT_GLASS);

        // Additive glow aura off to the left — brightens the sky / anything
        // behind it, like a spell or muzzle flash.
        let glow = Self::spawn(r, &Kv6::solid_cube(18, 0x80_FF_A0_30), [-45.0, 70.0, 42.0]);
        r.set_sprite_instance_material(glow, MAT_GLOW);

        // Grey smoke puff in the centre, its opacity pulsing each frame.
        let smoke = Self::spawn(r, &Kv6::solid_cube(22, 0x80_C8_C8_C8), [-2.0, 60.0, 30.0]);
        r.set_sprite_instance_material(smoke, MAT_SMOKE);
        self.smoke = Some(smoke);
        self.clock = 0.0;

        // Mixed-material window (TV.3): one model, an opaque frame around
        // translucent glass — the glass colour maps to MAT_GLASS, the frame
        // stays opaque (per-voxel materials, no per-instance material needed).
        let window = r.add_sprite_model_with_materials(
            &Self::build_window(),
            &[(GLASS_RGB & 0x00ff_ffff, MAT_GLASS)],
        );
        let _window_inst = r.add_sprite_instance_posed(window, Self::pose([-50.0, 95.0, 40.0]));

        eprintln!(
            "Transparency: glass (alpha id {MAT_GLASS}) over an opaque backdrop, \
             additive glow (id {MAT_GLOW}), pulsing smoke (id {MAT_SMOKE})"
        );
    }

    fn update(&mut self, ctx: &mut SceneCtx, dt: f64) {
        ctx.cam.fly_free(ctx.input, dt);
        self.clock += dt;
        // Pulse the smoke opacity in [40, 255] so it visibly thins + thickens
        // — a per-instance alpha_mul update, no volume re-upload.
        if let Some(id) = self.smoke {
            let t = 0.5 + 0.5 * (self.clock * 1.6).sin(); // 0..1
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let a = (40.0 + t * 215.0) as u8;
            ctx.renderer.set_sprite_instance_alpha(id, a);
        }
    }

    fn render(&mut self, ctx: &mut SceneCtx) {
        let settings = opticast_settings(ctx.size, ctx.scan_dist);
        let frame = frame_params(ctx.engine, &settings, ctx.scan_dist);
        let camera = ctx.cam.camera();
        ctx.renderer.render(&mut self.scene, &camera, &frame);
    }

    fn hud_lines(&self) -> Vec<String> {
        vec![
            "glass: AlphaBlend over opaque backdrop".to_string(),
            "glow: Additive (order-independent)".to_string(),
            "smoke: pulsing per-instance alpha_mul".to_string(),
        ]
    }
}
