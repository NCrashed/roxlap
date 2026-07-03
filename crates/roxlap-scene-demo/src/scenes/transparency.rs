//! The **Transparency** scene (TV stage): translucent voxel sprites over
//! opaque ones, showcasing the per-pixel front-to-back compositing the DDA
//! renderer enables — `AlphaBlend` glass/smoke, `Additive` glow, and
//! `Volumetric` (Beer–Lambert) fog.
//!
//! Layout (camera looks +y): an opaque brick backdrop with a cyan **glass
//! pane** in front of it (the backdrop tints through), an **additive glow**
//! aura off to the left (a spell-like emissive that brightens whatever is
//! behind it), and a grey **smoke puff** whose per-instance `alpha_mul`
//! pulses each frame — fading without re-uploading its volume.
//!
//! There is also a static mixed-material **window** (opaque frame + glass),
//! a **pulsing glass orb clip** — an animated `.rvc` whose voxels carry
//! per-voxel materials (`add_voxel_clip_with_materials`), the animated
//! analogue of the window — and a **filled volumetric fog cloud** whose core
//! reads denser than its rim (opacity ∝ ray path length, unlike the
//! thickness-independent alpha-blend puff).
//!
//! Each effect is one [`Material`] in the renderer's global palette
//! (`define_material`) referenced by an instance via
//! `set_sprite_instance_material` (whole-instance), a colour→material map
//! (`add_sprite_model_with_materials` / `add_voxel_clip_with_materials`,
//! per voxel), or `set_terrain_materials` (grid). See `PORTING-TRANSPARENCY.md`.

use glam::{DVec3, IVec3};
use roxlap_render::{
    DynSpriteTransform, Kv6, LoopMode, Material, SceneRenderer, SpriteInstanceId, VoxelClip,
};
use roxlap_scene::{GridTransform, Scene};

use crate::scene_api::{frame_params, opticast_settings, CameraPose, DemoScene, SceneCtx};

/// Palette ids (id 0 is the reserved opaque material).
const MAT_GLASS: u8 = 1;
const MAT_GLOW: u8 = 2;
const MAT_SMOKE: u8 = 3;
const MAT_FOG: u8 = 4;

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
            scene: Self::build_terrain(),
            smoke: None,
            clock: 0.0,
        }
    }

    /// A world grid behind the sprite cluster (TV.5/TV.6 terrain
    /// transparency): a big opaque red wall with a glass wall standing in
    /// front of it. The glass colour maps to `MAT_GLASS` via
    /// `set_terrain_materials`, so the red wall tints through it. Grid origin
    /// is world `(−25, 130, 10)`; grid-local `(x, y, z)` → world
    /// `(x−25, 130+y, 10+z)` (z is down).
    fn build_terrain() -> Scene {
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::at(DVec3::new(-25.0, 130.0, 10.0)));
        let g = scene.grid_mut(id).expect("terrain grid present");
        // Opaque red wall (far, y-local 30..33), spanning x 0..50, z 0..50.
        g.set_rect(
            IVec3::new(0, 30, 0),
            IVec3::new(50, 33, 50),
            Some(0x80_B0_50_40),
        );
        // Glass wall in front of it (y-local 0..3) — same span.
        g.set_rect(IVec3::new(0, 0, 0), IVec3::new(50, 3, 50), Some(GLASS_RGB));
        scene
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
            .expect("model just registered")
    }

    /// A **mixed-material** window panel (TV.3): an opaque brown frame around
    /// translucent cyan glass — built as one colour-coded model. Thin in y so
    /// it faces the camera. The glass colour maps to `MAT_GLASS`; the frame
    /// colour isn't in the map, so it stays opaque (material 0).
    fn build_window() -> Kv6 {
        const FRAME: u32 = 0x80_6A_4A_2A; // opaque brown
        Kv6::from_fn(30, 3, 30, |x, _y, z| {
            let edge = !(4..26).contains(&x) || !(4..26).contains(&z);
            Some(if edge { FRAME } else { GLASS_RGB })
        })
    }

    /// A **mixed-material animated clip** (TV.3 + the clip wiring): a small
    /// orb that pulses between a tight and a loose radius across four frames,
    /// every voxel in the glass colour. Registered via
    /// `add_voxel_clip_with_materials` so the glass colour classifies into
    /// `MAT_GLASS` per voxel — the animated, per-voxel-translucent analogue of
    /// the static mixed-material window. Returns a looping `.rvc` clip.
    fn build_glass_orb_clip() -> VoxelClip {
        const DIM: u32 = 16;
        let cx = (DIM as f32 - 1.0) * 0.5;
        // Four radii pulsing small→large→small (looped), each a sphere of the
        // glass colour.
        let radii = [4.0f32, 5.5, 7.0, 5.5];
        let frames: Vec<Kv6> = radii
            .iter()
            .map(|&r| {
                let r2 = r * r;
                Kv6::from_fn(DIM, DIM, DIM, |x, y, z| {
                    let (dx, dy, dz) = (x as f32 - cx, y as f32 - cx, z as f32 - cx);
                    (dx * dx + dy * dy + dz * dz <= r2).then_some(GLASS_RGB)
                })
            })
            .collect();
        VoxelClip::from_kv6_frames(&frames, 1.0, LoopMode::Loop, &[], 120, 1)
            .expect("glass orb clip frames are non-empty + same dims")
    }

    /// A **filled** smoke cloud for the `Volumetric` (Beer–Lambert) material:
    /// a solid grey sphere (not a hollow shell), so the ray traverses many
    /// absorbing voxels and the centre reads denser than the rim — opacity
    /// grows with path length, the thickness-aware counterpart of the flat
    /// `MAT_SMOKE` puff. Mapped to `MAT_FOG` per voxel.
    ///
    /// Built with [`Kv6::from_fn_keep_interior`] so the sphere's **interior**
    /// voxels survive — the default surface-only `from_fn` would leave a hollow
    /// shell and the ray would only graze the front + back faces, killing the
    /// depth-accumulation effect. The `keep_interior` predicate keeps the fog
    /// colour (all of it here); an opaque colour would still be culled.
    fn build_fog_cloud() -> Kv6 {
        const DIM: u32 = 30;
        let cx = (DIM as f32 - 1.0) * 0.5;
        let r2 = (DIM as f32 * 0.48).powi(2);
        Kv6::from_fn_keep_interior(
            DIM,
            DIM,
            DIM,
            |x, y, z| {
                let (dx, dy, dz) = (x as f32 - cx, y as f32 - cx, z as f32 - cx);
                (dx * dx + dy * dy + dz * dz <= r2).then_some(FOG_RGB)
            },
            |c| c == FOG_RGB, // translucent fog → keep interior; opaque → cull
        )
    }
}

/// The smoke voxel colour shared by the volumetric cloud + its material map.
const FOG_RGB: u32 = 0x80_A0_A0_A8;

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
        // Beer–Lambert fog: low per-voxel absorption, so a filled volume reads
        // dense at its core and thin at its rim (thickness-aware).
        r.define_material(MAT_FOG, Material::volumetric(28));

        // TV.5/TV.6 — the world grid's glass-coloured voxels render as the
        // glass material (the opaque red wall behind tints through).
        r.set_terrain_materials(&[(GLASS_RGB & 0x00ff_ffff, MAT_GLASS)]);

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

        // Mixed-material *animated* clip: a pulsing glass orb whose voxels
        // classify into MAT_GLASS per voxel (the clip analogue of the window).
        // Auto-plays on its own clock — `advance_voxel_clips` ticks it.
        let orb_clip = r.add_voxel_clip_with_materials(
            &Self::build_glass_orb_clip()
                .decode()
                .expect("glass orb clip decodes"),
            &[(GLASS_RGB & 0x00ff_ffff, MAT_GLASS)],
        );
        let _orb_inst =
            r.add_clip_instance_playing(orb_clip, Self::pose([45.0, 70.0, 40.0]), 1.0, 0);

        // Volumetric (Beer–Lambert) fog cloud: a *filled* grey sphere mapped
        // per voxel to MAT_FOG, so its core reads denser than its rim — the
        // thickness-aware counterpart of the flat alpha-blend smoke puff.
        let fog = r.add_sprite_model_with_materials(
            &Self::build_fog_cloud(),
            &[(FOG_RGB & 0x00ff_ffff, MAT_FOG)],
        );
        let _fog_inst = r.add_sprite_instance_posed(fog, Self::pose([95.0, 85.0, 36.0]));

        eprintln!(
            "Transparency: glass (alpha id {MAT_GLASS}) over an opaque backdrop, \
             additive glow (id {MAT_GLOW}), pulsing smoke (id {MAT_SMOKE}), \
             volumetric fog cloud (id {MAT_FOG})"
        );
    }

    fn update(&mut self, ctx: &mut SceneCtx, dt: f64) {
        ctx.cam.fly_free(ctx.input, dt);
        // One tick drives the auto-playing glass-orb clip (QE.1b).
        ctx.renderer.tick(&ctx.cam.camera(), dt);
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
            "orb: animated clip, per-voxel glass material".to_string(),
            "fog: Volumetric (Beer-Lambert, thickness-aware)".to_string(),
        ]
    }
}
