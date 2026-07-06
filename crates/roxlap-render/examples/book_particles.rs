//! Companion example for the book's "Particles" chapter
//! (`docs/book/src/particles.md`) — the chapter pulls its snippets
//! from here via `// ANCHOR:` markers, so everything it shows compiles.
//!
//! A water fountain bouncing off the floor, a buoyant smoke column,
//! and a scripted explosion every four seconds that carves a crater
//! and bursts the floor's own voxels back as tumbling debris:
//!
//! ```sh
//! cargo run --release -p roxlap-render --example book_particles
//! ROXLAP_GPU=0 cargo run --release -p roxlap-render --example book_particles  # force CPU
//! ```
//!
//! Keep the anchors when editing; `docs/book/check-anchors.sh` (run by
//! the CI `book` job) goes red if one disappears.

use std::sync::Arc;
use std::time::Instant;

use glam::{DVec3, IVec3};
use roxlap_core::opticast::OpticastSettings;
use roxlap_core::Camera;
use roxlap_render::{
    BackendPreference, BillboardLighting, CollisionMode, ConeDef, EmitterShape, FrameParams, Kv6,
    Material, ParticleEmitterDef, ParticleSystem, RenderOptions, Rgb, SceneRenderer, SpawnMode,
    SpriteModelId, VelocityDef, VoxColor,
};
use roxlap_scene::{GridId, GridTransform, Scene};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

const GROUND: VoxColor = VoxColor(0x80_58_6e_46);
const SKY: Rgb = Rgb(0x00_8f_bc_d4);

/// Palette ids (0 is the reserved opaque material).
const MAT_SMOKE: u8 = 1;
const MAT_SPARK: u8 = 2;
const MAT_WATER: u8 = 3;

/// Deterministic effects: the same seed reproduces the same run.
const SEED: u64 = 0x00ef_fec7;

/// The arena: a thick floor slab (craters must stay inside it).
/// Grid origin world (−64, −64, 200) ⇒ the walkable top is z = 200.
fn build_arena() -> (Scene, GridId) {
    let mut scene = Scene::new();
    let id = scene.add_grid(GridTransform::at(DVec3::new(-64.0, -64.0, 200.0)));
    let g = scene.grid_mut(id).expect("arena grid present");
    g.set_rect(IVec3::new(0, 0, 0), IVec3::new(128, 128, 8), Some(GROUND));
    g.bake(roxlap_scene::BakeMode::Directional);
    (scene, id)
}

/// White models — the emitter's per-particle tint does the colouring,
/// so all effects share plain geometry.
fn spark_kv6() -> Kv6 {
    Kv6::solid_cube(2, VoxColor(0x80_ff_ff_ff))
}

fn debris_kv6() -> Kv6 {
    Kv6::solid_cube(3, VoxColor(0x80_ff_ff_ff))
}

/// A rough ball for smoke puffs (a surface shell is fine for
/// alpha-blend; only Beer–Lambert volumetrics need interiors).
fn puff_kv6() -> Kv6 {
    Kv6::from_fn(7, 7, 7, |x, y, z| {
        let d = |v: u32| v as f32 - 3.0;
        let r2 = d(x).powi(2) + d(y).powi(2) + d(z).powi(2);
        (r2 <= 3.4 * 3.4).then_some(VoxColor(0x80_ff_ff_ff))
    })
}

/// `renderer` before `window` so it drops first.
#[derive(Default)]
struct App {
    renderer: Option<SceneRenderer>,
    window: Option<Arc<Window>>,
    scene: Option<Scene>,
    grid: Option<GridId>,
    particles: Option<ParticleSystem>,
    spark: Option<SpriteModelId>,
    debris: Option<SpriteModelId>,
    started: Option<Instant>,
    last_frame: Option<Instant>,
    /// Seconds until the next scripted explosion; crater index.
    fuse: f64,
    craters: i32,
}

impl App {
    // ANCHOR: explosion
    /// One explosion: `carve_debris` samples the floor's own voxel
    /// colours, carves the crater out of the grid, and bursts the
    /// removed voxels back as tumbling, bouncing, tinted debris — the
    /// world visibly becomes the particles. A transient spark burst
    /// goes on top (`Burst` spawns on add; removing the emitter lets
    /// the sparks live out their lifetimes).
    fn explode(&mut self, world: [f32; 3], voxel: IVec3) {
        let (Some(particles), Some(scene), Some(grid)) =
            (self.particles.as_mut(), self.scene.as_mut(), self.grid)
        else {
            return;
        };
        if let Some(debris) = self.debris {
            particles.carve_debris(
                scene,
                grid,
                voxel,
                4,         // crater radius, voxels
                8.0..16.0, // radial kick away from the crater centre
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
        if let Some(spark) = self.spark {
            let em = particles.add_emitter(ParticleEmitterDef {
                pos: [world[0], world[1], world[2] - 0.5],
                spawn: SpawnMode::Burst(24),
                lifetime: 0.5..1.1,
                velocity: VelocityDef {
                    spread: 24.0,
                    ..VelocityDef::default()
                },
                collision: CollisionMode::Kill,
                scale: 0.6,
                fade_out_frac: 0.5,
                tint: Rgb(0x00ff_c840),           // white-hot…
                tint_end: Some(Rgb(0x00ff_3000)), // …to ember red
                material: MAT_SPARK,
                lighting: BillboardLighting::FullBright,
                ..ParticleEmitterDef::new(spark)
            });
            particles.remove_emitter(em);
        }
    }
    // ANCHOR_END: explosion
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("roxlap book_particles"))
                .expect("create window"),
        );
        let size = window.inner_size();
        let backend = if std::env::var_os("ROXLAP_GPU").map_or(true, |v| v != "0") {
            BackendPreference::PreferGpu
        } else {
            BackendPreference::Cpu
        };
        let opts = RenderOptions {
            backend,
            clear_sky: SKY,
            ..RenderOptions::default()
        };
        let mut renderer = SceneRenderer::new(window.clone(), (size.width, size.height), &opts);

        renderer.define_material(MAT_SMOKE, Material::alpha_blend(110));
        renderer.define_material(MAT_SPARK, Material::additive(220));
        renderer.define_material(MAT_WATER, Material::alpha_blend(150));

        // ANCHOR: system
        // One system per scene, seeded for reproducible effects. The
        // particles instantiate ordinary kv6 sprite models.
        let mut particles = ParticleSystem::new(SEED);
        let spark = renderer.add_sprite_model(&spark_kv6());
        let puff = renderer.add_sprite_model(&puff_kv6());
        let debris = renderer.add_sprite_model(&debris_kv6());

        // A water fountain: a tight upward cone of translucent
        // droplets that fall back and bounce. +z is down, so the
        // fountain fires along −z and gravity (+z) pulls it back.
        particles.add_emitter(ParticleEmitterDef {
            pos: [0.0, 0.0, 199.0],
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
            tint: Rgb(0x0060_a8ff),
            material: MAT_WATER,
            ..ParticleEmitterDef::new(spark)
        });

        // A smoke column: buoyant (negative-z "gravity"), dragged,
        // spinning, growing while it condenses in and thins out.
        particles.add_emitter(ParticleEmitterDef {
            pos: [-40.0, 20.0, 198.0],
            shape: EmitterShape::Sphere { radius: 2.0 },
            spawn: SpawnMode::Rate(16.0),
            lifetime: 3.5..5.0,
            velocity: VelocityDef {
                base: [0.0, 0.0, -7.0], // rises: −z is up
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
            tint: Rgb(0x00b8_b8b8),
            tint_end: Some(Rgb(0x0050_5050)),
            material: MAT_SMOKE,
            // WorldUp: stable shading that doesn't swim as the camera
            // orbits (FaceNormal is the default; FullBright for glows).
            lighting: BillboardLighting::WorldUp,
            ..ParticleEmitterDef::new(puff)
        });
        // ANCHOR_END: system

        let (scene, grid) = build_arena();
        self.renderer = Some(renderer);
        self.window = Some(window);
        self.scene = Some(scene);
        self.grid = Some(grid);
        self.particles = Some(particles);
        self.spark = Some(spark);
        self.debris = Some(debris);
        self.started = Some(Instant::now());
        self.last_frame = Some(Instant::now());
        self.fuse = 4.0;
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        if self.renderer.is_none() {
            return;
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(r) = self.renderer.as_mut() {
                    r.resize(size.width.max(1), size.height.max(1));
                }
            }
            WindowEvent::RedrawRequested => {
                let t = self.started.map_or(0.0, |s| s.elapsed().as_secs_f64());
                let now = Instant::now();
                let dt = self
                    .last_frame
                    .map_or(0.0, |l| (now - l).as_secs_f64())
                    .min(0.1);
                self.last_frame = Some(now);

                // A scripted explosion every 4 s, marching across the
                // floor so each crater carves fresh voxels.
                self.fuse -= dt;
                if self.fuse <= 0.0 {
                    self.fuse = 4.0;
                    let wx = -30 + 12 * (self.craters % 6);
                    self.craters += 1;
                    #[allow(clippy::cast_precision_loss)]
                    self.explode(
                        [wx as f32, 30.0, 200.0],
                        IVec3::new(wx + 64, 94, 0), // grid-local: world − origin
                    );
                }

                let (Some(renderer), Some(scene), Some(particles)) = (
                    self.renderer.as_mut(),
                    self.scene.as_mut(),
                    self.particles.as_mut(),
                ) else {
                    return;
                };
                let camera = Camera::orbit(t * 0.15, 0.4, 170.0, [0.0, 0.0, 185.0]);

                // ANCHOR: tick
                // The whole per-frame particle protocol is one call:
                // simulate, collide against the scene's voxels, and
                // mirror live particles into sprite instances.
                particles.tick_with_scene(renderer, dt, scene);
                // ANCHOR_END: tick

                let window = self.window.as_ref().expect("window outlives renderer");
                let size = window.inner_size();
                let settings =
                    OpticastSettings::for_oracle_framebuffer(size.width.max(1), size.height.max(1));
                let mut frame = FrameParams::new(&settings);
                frame.sky_color = SKY;
                frame.fog_color = SKY;
                renderer.render(scene, &camera, &frame);
                renderer.present();
                window.request_redraw();
            }
            _ => {}
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.wait_idle();
        }
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let event_loop = EventLoop::new().expect("create event loop");
    event_loop
        .run_app(&mut App::default())
        .expect("run event loop");
}
