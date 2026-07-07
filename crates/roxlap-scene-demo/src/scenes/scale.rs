//! The **Scale** scene (SC stage): per-grid `voxel_world_size`. A coarse
//! **planet** grid (`voxel_world_size = 4.0` — big chunky voxels) and a fine
//! **ship** grid (`voxel_world_size = 0.25` — small smooth voxels) share one
//! scene at their true *relative* sizes: the ship is a detailed little craft
//! hovering over the blocky planet, a 16× voxel-size ratio between the two
//! grids.
//!
//! The ship (fine caster) drops a **cross-grid** hard sun shadow down onto
//! the planet (coarse receiver) — the SC.2 / SC.4 showcase: the shadow ray
//! is scaled into each grid's own frame, so a fine grid correctly shadows a
//! coarse one. It runs on **both** backends (CPU compose + GPU compute).
//!
//! The ship bobs vertically so the cross-grid shadow visibly sweeps across
//! the planet surface. Controls: WASD+mouse fly · `P` pause the bob · `K`
//! toggle sun shadows.

use std::f64::consts::FRAC_PI_2;

use glam::{DVec3, IVec3};
use roxlap_render::{DirectionalLight, LightRig, VoxColor};
use roxlap_scene::{Grid, GridId, GridTransform, Scene};
use winit::keyboard::KeyCode;

use crate::scene_api::{
    frame_params, opticast_settings, CameraPose, DemoScene, SceneCtx, SceneInput,
};

// Planet (coarse) palette — voxlap-packed `0x80_RR_GG_BB`.
const GRASS: VoxColor = VoxColor(0x80_4d_8a_3a);
const ROCK: VoxColor = VoxColor(0x80_8a_82_78);
// Ship (fine) palette.
const HULL: VoxColor = VoxColor(0x80_b8_bc_c8);
const CANOPY: VoxColor = VoxColor(0x80_5a_c8_e0);
const ENGINE: VoxColor = VoxColor(0x80_e0_7a_30);

/// Planet grid world origin (unscaled — vws multiplies grid-local coords).
const PLANET_ORIGIN: DVec3 = DVec3::new(0.0, 0.0, 0.0);
/// Ship grid resting world origin. voxlap z is DOWN, so a smaller z is
/// *higher*: the planet floor sits at world z ≈ 240, the ship hovers above
/// it at z ≈ 176.
const SHIP_BASE: DVec3 = DVec3::new(84.0, 78.0, 176.0);

pub struct ScaleScene {
    scene: Scene,
    /// The fine ship grid — its world origin bobs each frame.
    ship: GridId,
    /// Seconds since `enter`, driving the ship bob.
    clock: f64,
    paused: bool,
    sun_shadows: bool,
}

impl ScaleScene {
    #[must_use]
    pub fn new() -> Self {
        let (scene, ship) = Self::build_scene();
        Self {
            scene,
            ship,
            clock: 0.0,
            paused: false,
            sun_shadows: true,
        }
    }

    /// A coarse planet (vws 4.0) + a fine ship (vws 0.25) in one scene.
    fn build_scene() -> (Scene, GridId) {
        let mut scene = Scene::new();

        // Coarse planet — big chunky voxels. A 48×48 floor slab + two hills.
        let planet = scene.add_grid(GridTransform::at_scale(PLANET_ORIGIN, 4.0));
        {
            let g = scene.grid_mut(planet).expect("planet grid");
            g.set_rect(IVec3::new(0, 0, 60), IVec3::new(48, 48, 63), Some(GRASS));
            g.set_sphere(IVec3::new(14, 16, 62), 10, Some(ROCK));
            g.set_sphere(IVec3::new(34, 30, 62), 8, Some(ROCK));
        }

        // Fine ship — small smooth voxels; hovers above the planet floor.
        let ship = scene.add_grid(GridTransform::at_scale(SHIP_BASE, 0.25));
        {
            let g = scene.grid_mut(ship).expect("ship grid");
            build_ship(g);
        }

        (scene, ship)
    }
}

/// A rounded pod craft in the fine grid's local voxels (centred ≈ (30,30,34)).
/// At `voxel_world_size = 0.25` its ~64-voxel span is ~16 world units — a
/// small detailed craft next to the 192-world-unit planet.
fn build_ship(g: &mut Grid) {
    // Body — two overlapping spheres (pod + nose).
    g.set_sphere(IVec3::new(30, 30, 34), 18, Some(HULL));
    g.set_sphere(IVec3::new(30, 46, 34), 12, Some(HULL));
    // Canopy dome on top (voxlap z down → a smaller z is up).
    g.set_sphere(IVec3::new(30, 38, 22), 7, Some(CANOPY));
    // Two engine pods on the flanks.
    g.set_sphere(IVec3::new(11, 24, 38), 6, Some(ENGINE));
    g.set_sphere(IVec3::new(49, 24, 38), 6, Some(ENGINE));
}

impl DemoScene for ScaleScene {
    fn name(&self) -> &'static str {
        "Scale"
    }

    fn controls(&self) -> &'static str {
        "WASD+mouse fly · P: pause ship bob · K: toggle sun shadows (cross-grid)"
    }

    fn start_pose(&self) -> CameraPose {
        CameraPose {
            // Above + in front, looking +y and tilted down at the ship + its
            // shadow on the planet floor beyond.
            pos: [88.0, 8.0, 148.0],
            yaw: FRAC_PI_2,
            pitch: 0.36,
        }
    }

    fn enter(&mut self, _ctx: &mut SceneCtx) {
        eprintln!(
            "Scale: planet voxel_world_size=4.0 (chunky) + ship 0.25 (fine), a 16× ratio; \
             the fine ship casts a cross-grid sun shadow onto the coarse planet"
        );
    }

    fn update(&mut self, ctx: &mut SceneCtx, dt: f64) {
        ctx.cam.fly_free(ctx.input, dt);
        ctx.renderer.tick(&ctx.cam.camera(), dt);
        if !self.paused {
            self.clock += dt;
        }
        // Gentle vertical bob — sweeps the ship's cross-grid shadow across the
        // planet floor. Moving a grid is just updating its world origin.
        let bob = (self.clock * 0.8).sin() * 8.0;
        if let Some(g) = self.scene.grid_mut(self.ship) {
            g.transform.origin = SHIP_BASE + DVec3::new(0.0, 0.0, bob);
        }
    }

    fn on_input(&mut self, _ctx: &mut SceneCtx, ev: &SceneInput) {
        let SceneInput::Key {
            code,
            pressed: true,
        } = ev
        else {
            return;
        };
        match code {
            KeyCode::KeyP => {
                self.paused = !self.paused;
                eprintln!("ship bob {}", if self.paused { "paused" } else { "on" });
            }
            KeyCode::KeyK => {
                self.sun_shadows = !self.sun_shadows;
                eprintln!(
                    "sun shadows = {}",
                    if self.sun_shadows { "ON" } else { "OFF" }
                );
            }
            _ => {}
        }
    }

    fn render(&mut self, ctx: &mut SceneCtx) {
        let settings = opticast_settings(ctx.size, ctx.scan_dist);
        let mut frame = frame_params(ctx.engine, &settings, ctx.scan_dist);
        // Sun angled down-and-sideways so the ship's shadow lands on the floor
        // beside its footprint (voxlap +z is down → a positive z is overhead).
        let d = DVec3::new(0.35, 0.25, 0.9).normalize();
        let sun = DirectionalLight {
            direction: [d.x as f32, d.y as f32, d.z as f32],
            color: [1.0, 0.95, 0.85],
            intensity: 1.3,
            casts_shadow: self.sun_shadows,
        };
        frame.lights = Some(LightRig {
            sun: Some(sun),
            points: &[],
            spots: &[],
            ambient: [0.4, 0.42, 0.48],
            shadow_strength: 0.8,
            shadow_bias_voxels: 1.5,
            // World-distance cap (SC.2: world-uniform across grids).
            shadow_max_dist: 512.0,
            bands: 0,
            shadow_tint: [0.2, 0.24, 0.34],
        });
        let camera = ctx.cam.camera();
        ctx.renderer.render(&mut self.scene, &camera, &frame);
    }

    fn hud_lines(&self) -> Vec<String> {
        vec![
            "planet voxel_world_size 4.0 (chunky) · ship 0.25 (fine) — 16× ratio".to_string(),
            format!(
                "ship bob {} (P) · sun shadows {} (K) — cross-grid, both backends",
                if self.paused { "paused" } else { "on" },
                if self.sun_shadows { "on" } else { "off" },
            ),
        ]
    }
}
