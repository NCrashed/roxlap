//! Companion example for the book's "Lighting & materials" chapter
//! (`docs/book/src/lighting.md`) — the chapter pulls its snippets from
//! here via `// ANCHOR:` markers, so everything it shows compiles.
//!
//! A stone courtyard under a sweeping sun with hard shadows, two
//! orbiting coloured point lights, a spot cone on the monument, a
//! glass terrain wall, and a filled volumetric fog cloud:
//!
//! ```sh
//! cargo run --release -p roxlap-render --example book_lighting
//! ROXLAP_GPU=0 cargo run --release -p roxlap-render --example book_lighting  # force CPU
//! ```
//!
//! Keep the anchors when editing; `docs/book/check-anchors.sh` (run by
//! the CI `book` job) goes red if one disappears.

use std::sync::Arc;
use std::time::Instant;

use glam::{DVec3, IVec3};
use roxlap_core::opticast::OpticastSettings;
use roxlap_core::{AoParams, Camera};
use roxlap_render::{
    BackendPreference, DirectionalLight, DynSpriteTransform, FrameParams, Kv6, LightRig, Material,
    PointLight, RenderOptions, Rgb, SceneRenderer, SpotLight, VoxColor,
};
use roxlap_scene::{BakeMode, GridTransform, Scene};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

const GRASS: VoxColor = VoxColor(0x80_4d_8a_3a);
const STONE: VoxColor = VoxColor(0x80_8a_8a_92);
const MONUMENT: VoxColor = VoxColor(0x80_b0_60_48);
/// Voxel colours the material map classifies (low 24 bits matter).
const GLASS_RGB: VoxColor = VoxColor(0x80_50_c0_e0);
const FOG_RGB: VoxColor = VoxColor(0x80_a0_a0_a8);
const CRYSTAL_RGB: VoxColor = VoxColor(0x80_40_e8_ff);
const SKY: Rgb = Rgb(0x00_8f_bc_d4);

/// Material palette ids (id 0 is the reserved opaque material).
const MAT_GLASS: u8 = 1;
const MAT_FOG: u8 = 2;
const MAT_CRYSTAL: u8 = 3;

// ANCHOR: bake
/// A courtyard: floor, four pillars, a central monument and a glass
/// wall — then bake ambient occlusion into the brightness byte.
fn build_scene() -> Scene {
    let mut scene = Scene::new();
    let id = scene.add_grid(GridTransform::at(DVec3::ZERO));
    let grid = scene.grid_mut(id).expect("grid just added");
    grid.set_rect(
        IVec3::new(-96, -96, 210),
        IVec3::new(95, 95, 254),
        Some(GRASS),
    );
    for (px, py) in [(-60, -60), (40, -60), (-60, 40), (40, 40)] {
        grid.set_rect(
            IVec3::new(px, py, 165),
            IVec3::new(px + 12, py + 12, 209),
            Some(STONE),
        );
    }
    grid.set_rect(
        IVec3::new(-10, -10, 150),
        IVec3::new(10, 10, 209),
        Some(MONUMENT),
    );
    // A glass wall (the colour is mapped to a translucent material at
    // renderer setup — the terrain itself just stores the colour).
    grid.set_rect(
        IVec3::new(-70, -2, 170),
        IVec3::new(-30, 1, 209),
        Some(GLASS_RGB),
    );
    // A crystal capping the monument — the colour maps to an EMISSIVE
    // material, so it glows through every lighting mode below.
    grid.set_sphere(IVec3::new(0, 0, 147), 4, Some(CRYSTAL_RGB));

    // Bake ambient occlusion into every voxel's brightness byte:
    // crevices, pillar bases and inner corners darken. The runtime
    // lights below read this byte as their ambient/AO fill. Re-bake
    // after bulk edits (for small runtime carves use `bake_bbox` —
    // it re-bakes just the hole).
    grid.bake(BakeMode::AmbientOcclusion(AoParams {
        strength: 0.85, // fraction of ambient removed in a crevice
        radius: 1,      // contact reach in voxels (1 = tight edges)
        ..AoParams::default()
    }));
    scene
}
// ANCHOR_END: bake

// ANCHOR: volumetric
/// A **filled** fog ball for the volumetric material. The default
/// `Kv6::from_fn` culls interior voxels (a sprite normally only needs
/// its shell) — but Beer–Lambert opacity accumulates along the ray's
/// path *through* the volume, so a hollow shell would read as two thin
/// films. `from_fn_keep_interior` keeps the inside; its second closure
/// says which colours to keep (translucent ones — opaque interiors
/// would still be dead weight).
fn fog_cloud() -> Kv6 {
    const DIM: u32 = 28;
    let c = (DIM as f32 - 1.0) * 0.5;
    let r2 = (DIM as f32 * 0.48).powi(2);
    Kv6::from_fn_keep_interior(
        DIM,
        DIM,
        DIM,
        |x, y, z| {
            let (dx, dy, dz) = (x as f32 - c, y as f32 - c, z as f32 - c);
            (dx * dx + dy * dy + dz * dz <= r2).then_some(FOG_RGB)
        },
        |color| color == FOG_RGB,
    )
}
// ANCHOR_END: volumetric

/// `renderer` before `window` so it drops first.
#[derive(Default)]
struct App {
    renderer: Option<SceneRenderer>,
    window: Option<Arc<Window>>,
    scene: Option<Scene>,
    started: Option<Instant>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("roxlap book_lighting"))
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

        // ANCHOR: materials
        // The global material palette: 256 slots, id 0 permanently
        // opaque. A material is opacity + blend mode; voxels reference
        // it by id.
        renderer.define_material(MAT_GLASS, Material::alpha_blend(110));
        renderer.define_material(MAT_FOG, Material::volumetric(28));
        // Emissive (EV): the crystal renders at over-bright albedo —
        // immune to the bake, the runtime rig, shadows and side shades
        // (only fog still applies). `with_emissive` composes with any
        // blend mode; `Material::glow(e)` is the opaque shorthand.
        renderer.define_material(MAT_CRYSTAL, Material::alpha_blend(180).with_emissive(255));
        // Terrain: map voxel colours (low 24 bits) → material ids. Every
        // grid voxel in the glass colour now composites translucently.
        renderer.set_terrain_materials(&[
            (GLASS_RGB.rgb_part(), MAT_GLASS),
            (CRYSTAL_RGB.rgb_part(), MAT_CRYSTAL),
        ]);
        // ANCHOR_END: materials

        // The fog cloud: per-voxel colour → material classification at
        // model registration, then a posed instance above the courtyard.
        let cloud = renderer
            .add_sprite_model_with_materials(&fog_cloud(), &[(FOG_RGB.rgb_part(), MAT_FOG)]);
        renderer
            .add_sprite_instance_posed(
                cloud,
                DynSpriteTransform {
                    pos: [60.0, 50.0, 175.0],
                    ..DynSpriteTransform::default()
                },
            )
            .expect("model just registered");

        self.renderer = Some(renderer);
        self.window = Some(window);
        self.scene = Some(build_scene());
        self.started = Some(Instant::now());
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let (Some(renderer), Some(scene)) = (self.renderer.as_mut(), self.scene.as_mut()) else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => renderer.resize(size.width.max(1), size.height.max(1)),
            WindowEvent::RedrawRequested => {
                let t = self.started.map_or(0.0, |s| s.elapsed().as_secs_f64());
                let camera = Camera::orbit(t * 0.15, 0.35, 240.0, [0.0, 0.0, 185.0]);
                let window = self.window.as_ref().expect("window outlives renderer");
                let size = window.inner_size();
                let settings =
                    OpticastSettings::for_oracle_framebuffer(size.width.max(1), size.height.max(1));
                let mut frame = FrameParams::new(&settings);
                frame.sky_color = SKY;
                frame.fog_color = SKY;

                // ANCHOR: light_rig
                // Lighting is per-frame: build a LightRig and hand it to
                // FrameParams — there are no stateful light setters.
                // The sun sweeps overhead (+z is down, so a positive z
                // component keeps it above the horizon).
                let a = t * 0.4;
                let sun = DirectionalLight {
                    direction: [0.7 * a.cos() as f32, 0.7 * a.sin() as f32, 0.55],
                    color: [1.0, 0.93, 0.82], // warm daylight
                    intensity: 1.25,
                    casts_shadow: true,
                };
                // Two orbiting coloured fills; only one casts shadows
                // (shadow casters share a small per-frame budget —
                // excess casters are demoted with a log warning).
                let b = t * 0.8;
                let points = [
                    PointLight {
                        position: [70.0 * b.cos() as f32, 70.0 * b.sin() as f32, 195.0],
                        color: [1.0, 0.25, 0.2],
                        intensity: 4.0,
                        radius: 80.0,
                        casts_shadow: true,
                    },
                    PointLight {
                        position: [-70.0 * b.cos() as f32, -70.0 * b.sin() as f32, 195.0],
                        color: [0.25, 0.45, 1.0],
                        intensity: 4.0,
                        radius: 80.0,
                        casts_shadow: false,
                    },
                ];
                // A spot cone pinning the monument: position + axis +
                // inner/outer half-angles (soft edge between them).
                let spots = [SpotLight {
                    position: [0.0, 0.0, 110.0],
                    direction: [0.0, 0.0, 1.0], // straight down
                    color: [1.0, 0.9, 0.6],
                    intensity: 6.0,
                    radius: 120.0,
                    inner_angle_deg: 12.0,
                    outer_angle_deg: 22.0,
                    casts_shadow: false,
                }];
                frame.lights = Some(LightRig {
                    sun: Some(sun),
                    points: &points,
                    spots: &spots,
                    // Dim the baked ambient so the runtime lights read
                    // as the key; the baked byte is the ambient/AO fill.
                    ambient: [0.32, 0.34, 0.4],
                    shadow_strength: 0.85,
                    shadow_bias_voxels: 1.5,
                    shadow_max_dist: 256.0,
                    // Stylized cel look: quantize diffuse into 6 bands
                    // and ramp shadows toward a cool tint instead of
                    // black. `bands: 0` = smooth diffuse.
                    bands: 6,
                    shadow_tint: [0.16, 0.2, 0.34],
                });
                // ANCHOR_END: light_rig

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
