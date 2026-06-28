//! The **Lighting** scene (DL stage): runtime dynamic lighting — a sweeping
//! coloured **sun** that casts hard shadows, plus three orbiting coloured
//! **point lights** (two shadow-casting, one not). GPU-only; the CPU
//! backend renders the baked-ambient fallback.
//!
//! Layout (camera looks +y): a grass floor with four stone pillars and a
//! central monument. The sun rotates overhead so the pillars' shadows sweep
//! across the floor; the point lights orbit just above it, pooling coloured
//! light that the shadow-casters occlude behind the pillars.
//!
//! The baked brightness byte acts as a dim **ambient** fill (locked
//! decision #2); the sun + point lights are the runtime key/fill, composed
//! in the GPU scene-DDA shader (`shade_lit` + `shadow_occluded`). See
//! `PORTING-DYNLIGHT.md`.
//!
//! Controls: WASD+mouse fly · `P` pause the sun · `K` toggle sun shadows ·
//! `L` toggle point lights.

use glam::{DVec3, IVec3};
use roxlap_render::{DirectionalLight, LightRig, PointLight};
use roxlap_scene::{GridTransform, Scene};
use winit::keyboard::KeyCode;

use crate::scene_api::{
    frame_params, opticast_settings, CameraPose, DemoScene, SceneCtx, SceneInput,
};

// Voxlap-packed `0x80_RR_GG_BB` (high byte = full ambient brightness).
const GRASS: u32 = 0x80_4d_8a_3a;
const STONE: u32 = 0x80_8a_8a_92;
const MONUMENT: u32 = 0x80_b0_60_48;

/// Grid origin in world space; grid-local `(x,y,z)` → world `origin + xyz`.
const GRID_ORIGIN: DVec3 = DVec3::new(-48.0, 30.0, 0.0);
/// Floor top is at grid-local z=60 (voxlap z is down, so this is the lowest
/// the camera stands above). World centre of the floor ≈ (0, 78, 60).
const FLOOR_TOP_Z: i32 = 60;

// Four independent demo toggles (sun pause / sun shadows / point lights /
// stylized) — each a distinct on/off control, not a state better modelled
// as an enum.
#[allow(clippy::struct_excessive_bools)]
pub struct LightingScene {
    scene: Scene,
    /// Seconds since `enter`, driving the sun sweep + point-light orbit.
    clock: f64,
    paused: bool,
    sun_shadows: bool,
    points_on: bool,
    /// DL.6 — stylized (cel + hue-ramp) vs smooth diffuse. `J` toggles.
    stylized: bool,
    /// The point lights, rebuilt each frame (orbit) — borrowed into the
    /// per-frame [`LightRig`] at render time.
    points: Vec<PointLight>,
}

impl LightingScene {
    #[must_use]
    pub fn new() -> Self {
        Self {
            scene: Self::build_terrain(),
            clock: 0.0,
            paused: false,
            sun_shadows: true,
            points_on: true,
            stylized: true,
            points: Vec::new(),
        }
    }

    /// A grass floor with four stone pillars + a central monument — enough
    /// vertical geometry for the sun's shadows to sweep across the ground.
    fn build_terrain() -> Scene {
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::at(GRID_ORIGIN));
        let g = scene.grid_mut(id).expect("lighting grid present");
        // Floor slab (top face up at z=60), 96×96.
        g.set_rect(
            IVec3::new(0, 0, FLOOR_TOP_Z),
            IVec3::new(96, 96, FLOOR_TOP_Z + 3),
            Some(GRASS),
        );
        // Four pillars rising 30 voxels off the floor (z 30..60).
        for (px, py) in [(20, 20), (66, 20), (20, 66), (66, 66)] {
            g.set_rect(
                IVec3::new(px, py, 30),
                IVec3::new(px + 10, py + 10, FLOOR_TOP_Z),
                Some(STONE),
            );
        }
        // Central monument (taller, z 18..60).
        g.set_rect(
            IVec3::new(40, 40, 18),
            IVec3::new(56, 56, FLOOR_TOP_Z),
            Some(MONUMENT),
        );
        scene
    }

    /// World centre the point lights orbit around (just above the floor).
    fn orbit_centre() -> DVec3 {
        GRID_ORIGIN + DVec3::new(48.0, 48.0, f64::from(FLOOR_TOP_Z) - 12.0)
    }

    /// The sun's world **travel** direction (from sky toward ground): a
    /// fixed downward tilt with the azimuth sweeping with the clock, so
    /// shadows rotate across the floor.
    fn sun_direction(&self) -> [f32; 3] {
        let a = self.clock * 0.4;
        // +z is down (voxlap), so a positive z keeps the sun above the
        // horizon; the xy part rotates the shadow azimuth.
        let v = DVec3::new(0.7 * a.cos(), 0.7 * a.sin(), 0.55).normalize();
        [v.x as f32, v.y as f32, v.z as f32]
    }

    /// Rebuild the three orbiting coloured point lights for the current
    /// clock. Two cast shadows (red, blue); the green one does not — the
    /// "a few with shadows, the rest without" the feature targets.
    fn rebuild_points(&mut self) {
        self.points.clear();
        if !self.points_on {
            return;
        }
        let c = Self::orbit_centre();
        let specs = [
            ([1.0_f32, 0.25, 0.2], 0.0_f64, true), // red, shadows
            ([0.25, 0.45, 1.0], 2.094, true),      // blue, shadows (+120°)
            ([0.3, 1.0, 0.35], 4.189, false),      // green, no shadows (+240°)
        ];
        for (color, phase, casts_shadow) in specs {
            let a = self.clock * 0.8 + phase;
            let pos = c + DVec3::new(38.0 * a.cos(), 38.0 * a.sin(), 0.0);
            self.points.push(PointLight {
                position: [pos.x as f32, pos.y as f32, pos.z as f32],
                color,
                intensity: 4.0,
                radius: 80.0,
                casts_shadow,
            });
        }
    }
}

impl DemoScene for LightingScene {
    fn name(&self) -> &'static str {
        "Lighting"
    }

    fn controls(&self) -> &'static str {
        "WASD+mouse fly · P: pause sun · K: sun shadows · L: point lights · J: stylized/smooth (GPU only)"
    }

    fn start_pose(&self) -> CameraPose {
        CameraPose {
            // Above + in front of the floor, looking +y and tilted down.
            pos: [0.0, -10.0, 22.0],
            yaw: std::f64::consts::FRAC_PI_2,
            pitch: 0.42,
        }
    }

    fn enter(&mut self, _ctx: &mut SceneCtx) {
        self.rebuild_points();
    }

    fn update(&mut self, ctx: &mut SceneCtx, dt: f64) {
        // Shared free-fly camera (WASD + Space/Shift); no collision here.
        ctx.cam.fly_free(ctx.input, dt);
        if !self.paused {
            self.clock += dt;
        }
        self.rebuild_points();
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
                eprintln!("sun {}", if self.paused { "paused" } else { "sweeping" });
            }
            KeyCode::KeyK => {
                self.sun_shadows = !self.sun_shadows;
                eprintln!(
                    "sun shadows = {}",
                    if self.sun_shadows { "ON" } else { "OFF" }
                );
            }
            KeyCode::KeyL => {
                self.points_on = !self.points_on;
                eprintln!(
                    "point lights = {}",
                    if self.points_on { "ON" } else { "OFF" }
                );
            }
            KeyCode::KeyJ => {
                self.stylized = !self.stylized;
                eprintln!(
                    "lighting = {}",
                    if self.stylized { "stylized (cel+ramp)" } else { "smooth" }
                );
            }
            _ => {}
        }
    }

    fn render(&mut self, ctx: &mut SceneCtx) {
        let settings = opticast_settings(ctx.size, ctx.scan_dist);
        let mut frame = frame_params(ctx.engine, &settings, ctx.scan_dist);
        let sun = DirectionalLight {
            direction: self.sun_direction(),
            color: [1.0, 0.93, 0.82], // warm daylight
            intensity: 1.25,
            casts_shadow: self.sun_shadows,
        };
        frame.lights = Some(LightRig {
            sun: Some(sun),
            points: &self.points,
            // Dim ambient so the runtime sun/points read as the key light
            // (the baked byte is the ambient/AO channel, locked decision #2).
            ambient: [0.32, 0.34, 0.4],
            shadow_strength: 0.85,
            shadow_bias_voxels: 1.5,
            shadow_max_dist: 256.0,
            // DL.6 — stylized cel + hue-shifted ramp (J toggles vs smooth).
            // Cool shadow tint → warm sun, terraced in 4 bands.
            bands: if self.stylized { 4 } else { 0 },
            shadow_tint: [0.16, 0.2, 0.34],
        });
        let camera = ctx.cam.camera();
        ctx.renderer.render(&mut self.scene, &camera, &frame);
    }

    fn hud_lines(&self) -> Vec<String> {
        vec![
            format!(
                "sun {} · shadows {} · points {} · {}",
                if self.paused { "paused" } else { "sweeping" },
                if self.sun_shadows { "on" } else { "off" },
                if self.points_on { "on" } else { "off" },
                if self.stylized { "stylized" } else { "smooth" },
            ),
            "GPU-only dynamic lighting (CPU shows baked ambient)".to_string(),
        ]
    }
}
