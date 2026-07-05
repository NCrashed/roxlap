//! The **Particles** scene (PS.4): the stage-PS `ParticleSystem` live —
//! a translucent water fountain over a water pool, a rising smoke
//! column, and crosshair-aimed left-click explosions that carve the
//! floor and burst sparks + tumbling debris.
//!
//! Everything here is the host-side pattern the particle system is
//! designed around: one `ParticleSystem` owned by the scene, emitters
//! built from `ParticleEmitterDef`, and a single
//! `tick_with_scene(renderer, dt, &scene)` per frame (simulation +
//! voxel collision + facade sync). The scene is lit by a runtime
//! [`LightRig`] (warm sun with shadows + accent point lights +
//! fading orange explosion flashes) over a baked ambient byte, and
//! aiming is FPS-style: the click picks through the **screen-centre
//! crosshair** (drawn as world-space overlay [`Line3`] ticks), not a
//! mouse cursor. Axes reminder: +z is DOWN — the fountain fires and
//! smoke rises with **negative** z velocity.

use glam::{DVec3, IVec3};
use roxlap_core::Camera;
use roxlap_render::{
    BillboardLighting, CollisionMode, ConeDef, DirectionalLight, EmitterShape, Kv6, LightRig,
    Line3, Material, ParticleEmitterDef, ParticleSystem, PointLight, SpawnMode, SpriteModelId,
    VelocityDef,
};
use roxlap_scene::{GridTransform, Scene};
use winit::event::MouseButton;

use crate::scene_api::{
    frame_params, opticast_settings, CameraPose, DemoScene, SceneCtx, SceneInput,
};

/// Palette ids (0 is the reserved opaque material).
const MAT_SMOKE: u8 = 1;
const MAT_SPARK: u8 = 2;
const MAT_WATER: u8 = 3;

/// Water voxel colour — unique in the arena, so the terrain-material
/// map (`WATER_RGB → MAT_WATER`) makes exactly the pool translucent.
const WATER_RGB: u32 = 0x80_2E_6E_C8;

/// Deterministic effects: same seed every `enter`.
const SEED: u64 = 0x00EF_FEC7;

/// Explosion light-flash lifetime, seconds.
const FLASH_SECS: f64 = 0.45;

pub struct ParticlesScene {
    /// A flat arena slab the effects live on (and get carved out of).
    scene: Scene,
    particles: ParticleSystem,
    /// `[spark, puff, debris]` model handles from this activation.
    spark: Option<SpriteModelId>,
    puff: Option<SpriteModelId>,
    debris: Option<SpriteModelId>,
    /// Explosion light flashes: `(world pos, seconds remaining)`.
    flashes: Vec<([f32; 3], f64)>,
    /// Per-frame point-light scratch (accents + flashes) the render's
    /// [`LightRig`] borrows.
    points: Vec<PointLight>,
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
            flashes: Vec::new(),
            points: Vec::new(),
            explosions: 0,
        }
    }

    /// A 128×128 floor slab: grid origin world `(−64, 30, 60)`, solid
    /// for local z 0..6 ⇒ the walkable surface is world z = 60 and the
    /// slab is thick enough that explosion craters stay inside it.
    ///
    /// Recolouring an already-solid region is a no-op (`set_rect`
    /// merges spans but keeps existing colours), so the pad and the
    /// pool **carve first, then insert** their own colour.
    fn build_arena() -> Scene {
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::at(DVec3::new(-64.0, 30.0, 60.0)));
        let g = scene.grid_mut(id).expect("arena grid present");
        g.set_rect(
            IVec3::new(0, 0, 0),
            IVec3::new(128, 128, 6),
            Some(0x80_58_6E_46), // mossy ground
        );
        // A lighter landing-pad rim under the fountain so the bounce
        // reads (carve + reinsert = recolour).
        let (pad_lo, pad_hi) = (IVec3::new(52, 48, 0), IVec3::new(76, 72, 0));
        g.set_rect(pad_lo, pad_hi, None);
        g.set_rect(pad_lo, pad_hi, Some(0x80_8A_8A_7A));
        // Translucent water pool inside the rim: `WATER_RGB` maps to
        // `MAT_WATER` via `set_terrain_materials` in `enter`, so these
        // grid voxels render alpha-blended — the slab shows through.
        let (pool_lo, pool_hi) = (IVec3::new(56, 52, 0), IVec3::new(72, 68, 1));
        g.set_rect(pool_lo, pool_hi, None);
        g.set_rect(pool_lo, pool_hi, Some(WATER_RGB));
        // Baked byte = the ambient/AO channel under the runtime rig.
        g.bake_lightmode(1);
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

    /// One click: `carve_debris` (PS.5) samples the floor's voxel
    /// colours, carves the crater, and bursts them back as tumbling
    /// tinted debris — plus a hand-rolled spark flash on top.
    fn explode(&mut self, world: [f32; 3], grid: roxlap_scene::GridId, voxel: IVec3) {
        if let Some(debris) = self.debris {
            self.particles.carve_debris(
                &mut self.scene,
                grid,
                voxel,
                4,
                8.0..16.0, // radial kick away from the crater
                &ParticleEmitterDef {
                    lifetime: 1.4..2.4,
                    collision: CollisionMode::Bounce { restitution: 0.35 },
                    spin: -7.0..7.0,
                    scale: 0.9,
                    scale_end: Some(0.4),
                    fade_out_frac: 0.2,
                    ..ParticleEmitterDef::new(debris)
                },
            );
        }
        // Spark flash: a transient burst emitter, lifted half a voxel
        // off the surface so the first collision sample doesn't sit
        // inside the (now carved) floor.
        let origin = [world[0], world[1], world[2] - 0.5];
        if let Some(spark) = self.spark {
            let em = self.particles.add_emitter(ParticleEmitterDef {
                pos: origin,
                spawn: SpawnMode::Burst(24),
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
        // A brief orange light flash at the blast (fades in `update`,
        // rendered as an extra rig point light).
        self.flashes
            .push(([world[0], world[1], world[2] - 2.0], FLASH_SECS));
        self.explosions += 1;
    }
}

impl DemoScene for ParticlesScene {
    fn name(&self) -> &'static str {
        "Particles"
    }

    fn controls(&self) -> &'static str {
        "left-click explodes at the crosshair · water fountain + pool · WASD+mouse fly"
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
        r.define_material(MAT_WATER, Material::alpha_blend(150));
        // The pool's grid voxels render translucent (TV.5/TV.6 terrain
        // transparency) — same material the fountain droplets use.
        r.set_terrain_materials(&[(WATER_RGB & 0x00ff_ffff, MAT_WATER)]);

        // Fresh models each activation (the host reset the registry) —
        // and a fresh system: the old one holds handles from the
        // previous epoch, which would only rack up stale-model kills.
        self.spark = Some(r.add_sprite_model(&Self::spark_kv6()));
        self.puff = Some(r.add_sprite_model(&Self::puff_kv6()));
        self.debris = Some(r.add_sprite_model(&Self::debris_kv6()));
        self.scene = Self::build_arena(); // un-carve previous visits
        self.particles = ParticleSystem::new(SEED);
        // Halved explosion load: a radius-4 crater samples ~70 voxels,
        // which crowded the frame together with the spark flash.
        self.particles.set_carve_debris_cap(32);
        self.flashes.clear();

        // Water fountain: a tight upward cone of translucent droplets
        // (MAT_WATER) that fall back and bounce off the pool surface.
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
                material: MAT_WATER,
                // Default FaceNormal lighting: droplets pick up the
                // sun, the cool pool fill and explosion flashes.
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
                // WorldUp: stable sun shading that doesn't swim as the
                // camera orbits, and the warm accent light tints it.
                lighting: BillboardLighting::WorldUp,
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
        // Explosion flashes burn down.
        for f in &mut self.flashes {
            f.1 -= dt;
        }
        self.flashes.retain(|f| f.1 > 0.0);
    }

    fn on_input(&mut self, ctx: &mut SceneCtx, ev: &SceneInput) {
        // FPS-style aiming: the click picks through the screen centre
        // (where the crosshair is), not a mouse cursor — mouse-look
        // already owns the pointer.
        if let SceneInput::Mouse {
            button: MouseButton::Left,
            pressed: true,
        } = ev
        {
            let (px, py) = (ctx.size.0 / 2, ctx.size.1 / 2);
            let cam = ctx.cam.camera();
            if let Some(h) = ctx.renderer.pick(&self.scene, &cam, px, py) {
                self.explode(h.world, h.grid, h.voxel);
            }
        }
    }

    fn render(&mut self, ctx: &mut SceneCtx) {
        let settings = opticast_settings(ctx.size, ctx.scan_dist);
        let mut frame = frame_params(ctx.engine, &settings, ctx.scan_dist);

        // Runtime lighting: two fixed accents + the explosion flashes.
        self.points.clear();
        self.points.push(PointLight {
            position: [0.0, 90.0, 48.0], // cool fill over the pool
            color: [0.35, 0.55, 1.0],
            intensity: 2.5,
            radius: 70.0,
            casts_shadow: false,
        });
        self.points.push(PointLight {
            position: [-38.0, 110.0, 46.0], // warm accent by the smoke
            color: [1.0, 0.6, 0.3],
            intensity: 2.0,
            radius: 60.0,
            casts_shadow: false,
        });
        for &(pos, ttl) in &self.flashes {
            #[allow(clippy::cast_possible_truncation)]
            let k = (ttl / FLASH_SECS) as f32;
            self.points.push(PointLight {
                position: pos,
                color: [1.0, 0.55, 0.2],
                intensity: 7.0 * k, // fades out with the flash
                radius: 48.0,
                casts_shadow: false,
            });
        }
        frame.lights = Some(LightRig {
            sun: Some(DirectionalLight {
                direction: [0.45, 0.3, 0.65], // travels down-ish: sun up
                color: [1.0, 0.95, 0.85],
                intensity: 1.15,
                casts_shadow: true,
            }),
            points: &self.points,
            spots: &[],
            ambient: [0.42, 0.44, 0.5],
            shadow_strength: 0.7,
            shadow_bias_voxels: 1.5,
            shadow_max_dist: 256.0,
            bands: 0,
            shadow_tint: [0.18, 0.2, 0.3],
        });

        let camera = ctx.cam.camera();
        ctx.renderer.render(&mut self.scene, &camera, &frame);
        // Crosshair over the frame — always on top (no depth test).
        ctx.renderer.draw_lines(&camera, &crosshair_lines(&camera));
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

/// Four crosshair ticks around the screen centre, as world-space
/// overlay lines a fixed distance in front of the camera (drawn with
/// `depth_test: false`, so nothing occludes them). World-space +
/// unit-basis camera ⇒ the on-screen size is FOV-stable and the same
/// on both backends — no egui needed.
fn crosshair_lines(cam: &Camera) -> Vec<Line3> {
    const DIST: f64 = 2.0; // ahead of the camera
    const GAP: f64 = 0.016; // half-gap around the exact centre
    const ARM: f64 = 0.05; // tick length
    let centre = [
        cam.pos[0] + cam.forward[0] * DIST,
        cam.pos[1] + cam.forward[1] * DIST,
        cam.pos[2] + cam.forward[2] * DIST,
    ];
    let mut out = Vec::with_capacity(4);
    for axis in [cam.right, cam.down] {
        for sign in [-1.0, 1.0] {
            let at = |k: f64| {
                [
                    centre[0] + axis[0] * sign * k,
                    centre[1] + axis[1] * sign * k,
                    centre[2] + axis[2] * sign * k,
                ]
            };
            out.push(Line3 {
                a: at(GAP),
                b: at(GAP + ARM),
                color: 0xE0_FF_FF_FF, // near-opaque white
                width_px: 2.0,
                depth_test: false,
            });
        }
    }
    out
}
