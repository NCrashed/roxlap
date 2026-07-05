//! The **Particles** scene (PS.4): the stage-PS `ParticleSystem` live —
//! a bouncing fountain, a rising smoke column, and left-click
//! explosions that carve the floor and burst sparks + tumbling debris.
//!
//! Everything here is the host-side pattern the particle system is
//! designed around: one `ParticleSystem` owned by the scene, emitters
//! built from `ParticleEmitterDef`, and a single
//! `tick_with_scene(renderer, dt, &scene)` per frame (simulation +
//! voxel collision + facade sync). Axes reminder: +z is DOWN — the
//! fountain fires and smoke rises with **negative** z velocity.

use glam::{DVec3, IVec3};
use roxlap_render::{
    BillboardLighting, CollisionMode, ConeDef, EmitterShape, Kv6, Material, ParticleEmitterDef,
    ParticleSystem, SpawnMode, SpriteModelId, VelocityDef,
};
use roxlap_scene::{GridTransform, Scene};
use winit::event::MouseButton;

use crate::scene_api::{
    frame_params, opticast_settings, CameraPose, DemoScene, SceneCtx, SceneInput,
};

/// Palette ids (0 is the reserved opaque material).
const MAT_SMOKE: u8 = 1;
const MAT_SPARK: u8 = 2;

/// Deterministic effects: same seed every `enter`.
const SEED: u64 = 0x00EF_FEC7;

pub struct ParticlesScene {
    /// A flat arena slab the effects live on (and get carved out of).
    scene: Scene,
    particles: ParticleSystem,
    /// `[spark, puff, debris]` model handles from this activation.
    spark: Option<SpriteModelId>,
    puff: Option<SpriteModelId>,
    debris: Option<SpriteModelId>,
    /// Last cursor position (physical px) for click picking.
    mouse_px: (f64, f64),
    explosions: u64,
}

impl ParticlesScene {
    #[must_use]
    pub fn new() -> Self {
        Self {
            scene: Self::build_arena(),
            particles: ParticleSystem::new(SEED),
            spark: None,
            puff: None,
            debris: None,
            mouse_px: (0.0, 0.0),
            explosions: 0,
        }
    }

    /// A 128×128 floor slab: grid origin world `(−64, 30, 60)`, solid
    /// for local z 0..6 ⇒ the walkable surface is world z = 60 and the
    /// slab is thick enough that explosion craters stay inside it.
    fn build_arena() -> Scene {
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::at(DVec3::new(-64.0, 30.0, 60.0)));
        let g = scene.grid_mut(id).expect("arena grid present");
        g.set_rect(
            IVec3::new(0, 0, 0),
            IVec3::new(128, 128, 6),
            Some(0x80_58_6E_46), // mossy ground
        );
        // A lighter landing pad under the fountain so the bounce reads.
        g.set_rect(
            IVec3::new(52, 48, 0),
            IVec3::new(76, 72, 0),
            Some(0x80_8A_8A_7A),
        );
        scene
    }

    /// White models — per-instance tint does all the colouring, so the
    /// three effects share plain geometry.
    fn spark_kv6() -> Kv6 {
        Kv6::solid_cube(2, 0x80_FF_FF_FF)
    }

    fn debris_kv6() -> Kv6 {
        Kv6::solid_cube(3, 0x80_FF_FF_FF)
    }

    /// A rough ball for smoke puffs (surface shell is fine for
    /// alpha-blend; only Beer–Lambert volumetrics need interiors).
    fn puff_kv6() -> Kv6 {
        Kv6::from_fn(7, 7, 7, |x, y, z| {
            let d = |v: u32| v as f32 - 3.0;
            let r2 = d(x).powi(2) + d(y).powi(2) + d(z).powi(2);
            (r2 <= 3.4 * 3.4).then_some(0x80_FF_FF_FF)
        })
    }

    /// One click: carve a crater at the picked voxel and burst sparks
    /// and debris from the surface point. Emitters are created, burst,
    /// and removed immediately — retire-drain keeps them alive until
    /// their last particle dies.
    fn explode(&mut self, world: [f32; 3], grid: roxlap_scene::GridId, voxel: IVec3) {
        if let Some(g) = self.scene.grid_mut(grid) {
            g.set_sphere(voxel, 4, None);
        }
        // Lift the burst origin half a voxel off the surface so the
        // first collision sample doesn't sit inside the old floor.
        let origin = [world[0], world[1], world[2] - 0.5];
        if let Some(spark) = self.spark {
            let em = self.particles.add_emitter(ParticleEmitterDef {
                pos: origin,
                spawn: SpawnMode::Burst(50),
                lifetime: 0.5..1.1,
                velocity: VelocityDef {
                    spread: 24.0,
                    ..VelocityDef::default()
                },
                collision: CollisionMode::Kill,
                scale: 0.6,
                fade_out_frac: 0.5,
                tint: 0x00FF_C840,
                tint_end: Some(0x00FF_3000),
                material: MAT_SPARK,
                lighting: BillboardLighting::FullBright,
                ..ParticleEmitterDef::new(spark)
            });
            self.particles.remove_emitter(em);
        }
        if let Some(debris) = self.debris {
            let em = self.particles.add_emitter(ParticleEmitterDef {
                pos: origin,
                spawn: SpawnMode::Burst(24),
                lifetime: 1.4..2.4,
                velocity: VelocityDef {
                    spread: 5.0,
                    cone: Some(ConeDef {
                        axis: [0.0, 0.0, -1.0], // up and out
                        half_angle_deg: 55.0,
                        speed: 8.0..18.0,
                    }),
                    ..VelocityDef::default()
                },
                collision: CollisionMode::Bounce { restitution: 0.35 },
                spin: -7.0..7.0,
                scale: 0.9,
                scale_end: Some(0.4),
                fade_out_frac: 0.2,
                tint: 0x0058_6E46, // the floor's colour flying off
                ..ParticleEmitterDef::new(debris)
            });
            self.particles.remove_emitter(em);
        }
        self.explosions += 1;
    }
}

impl DemoScene for ParticlesScene {
    fn name(&self) -> &'static str {
        "Particles"
    }

    fn controls(&self) -> &'static str {
        "left-click carves + explodes · fountain bounces, smoke rises · WASD+mouse fly"
    }

    fn start_pose(&self) -> CameraPose {
        CameraPose {
            pos: [0.0, -10.0, 35.0],
            yaw: std::f64::consts::FRAC_PI_2,
            pitch: 0.35, // look slightly down at the arena
        }
    }

    fn enter(&mut self, ctx: &mut SceneCtx) {
        let r = &mut *ctx.renderer;
        r.define_material(MAT_SMOKE, Material::alpha_blend(110));
        r.define_material(MAT_SPARK, Material::additive(220));

        // Fresh models each activation (the host reset the registry) —
        // and a fresh system: the old one holds handles from the
        // previous epoch, which would only rack up stale-model kills.
        self.spark = Some(r.add_sprite_model(&Self::spark_kv6()));
        self.puff = Some(r.add_sprite_model(&Self::puff_kv6()));
        self.debris = Some(r.add_sprite_model(&Self::debris_kv6()));
        self.scene = Self::build_arena(); // un-carve previous visits
        self.particles = ParticleSystem::new(SEED);

        // Fountain: a tight upward cone that falls back and bounces
        // off the landing pad.
        if let Some(spark) = self.spark {
            self.particles.add_emitter(ParticleEmitterDef {
                pos: [0.0, 90.0, 59.0],
                spawn: SpawnMode::Rate(140.0),
                lifetime: 2.2..3.2,
                velocity: VelocityDef {
                    cone: Some(ConeDef {
                        axis: [0.0, 0.0, -1.0], // straight up
                        half_angle_deg: 11.0,
                        speed: 26.0..34.0,
                    }),
                    ..VelocityDef::default()
                },
                collision: CollisionMode::Bounce { restitution: 0.45 },
                scale: 0.7,
                fade_out_frac: 0.3,
                tint: 0x0060_A8FF, // water blue
                lighting: BillboardLighting::FullBright,
                ..ParticleEmitterDef::new(spark)
            });
        }

        // Smoke column: buoyant, spinning, growing, condensing in and
        // thinning out.
        if let Some(puff) = self.puff {
            self.particles.add_emitter(ParticleEmitterDef {
                pos: [-38.0, 110.0, 58.0],
                shape: EmitterShape::Sphere { radius: 2.0 },
                spawn: SpawnMode::Rate(16.0),
                lifetime: 3.5..5.0,
                velocity: VelocityDef {
                    base: [0.0, 0.0, -7.0], // rises: -z is up
                    spread: 1.0,
                    ..VelocityDef::default()
                },
                gravity: [1.2, 0.0, -1.5], // buoyant, drifting east
                drag: 0.9,
                spin: -0.6..0.6,
                scale: 0.8,
                scale_end: Some(2.8),
                fade_in_frac: 0.25,
                fade_out_frac: 0.45,
                tint: 0x00B8_B8B8,
                tint_end: Some(0x0050_5050),
                material: MAT_SMOKE,
                lighting: BillboardLighting::AmbientOnly,
                ..ParticleEmitterDef::new(puff)
            });
        }
    }

    fn update(&mut self, ctx: &mut SceneCtx, dt: f64) {
        ctx.cam.fly_free(ctx.input, dt);
        // The whole per-frame particle protocol: simulate, collide
        // against the arena, mirror into sprite instances.
        self.particles
            .tick_with_scene(ctx.renderer, dt, &self.scene);
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
                if let Some(h) = ctx.renderer.pick(&self.scene, &cam, px, py) {
                    self.explode(h.world, h.grid, h.voxel);
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

    fn hud_lines(&self) -> Vec<String> {
        vec![
            format!(
                "particles: {} live · {} spawn-dropped",
                self.particles.particle_count(),
                self.particles.dropped_spawns(),
            ),
            format!("explosions: {}", self.explosions),
        ]
    }
}
