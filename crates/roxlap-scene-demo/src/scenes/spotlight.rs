//! The **Spotlight** scene (SL stage): a focused showcase of the spot (cone)
//! light on its own. A near-dark room lit by a **single** spotlight — no sun,
//! no point lights — so the cone is unmistakable: a crisp circular pool on the
//! floor, a soft edge, and hard shadows where pillars block the beam.
//!
//! Two modes:
//!   * **Searchlight** (default) — mounted high above the pillar cluster,
//!     aimed down; `O` sweeps its axis so the pool circles the floor.
//!   * **Flashlight** (`F`) — the spot rides the camera (position + forward),
//!     so flying around sweeps the cone over the geometry like a torch.
//!
//! The cone half-angle is live-adjustable (`[` / `]`), which is the clearest
//! way to *see* it is a cone. Shadows toggle with `K`. Runs on both backends.
//!
//! Controls: WASD+mouse fly · `F` flashlight/searchlight · `O` sweep ·
//! `[` / `]` narrow/widen the cone · `K` shadows · `P` pause the sweep.

use glam::{DVec3, IVec3};
use roxlap_render::VoxColor;
use roxlap_render::{LightRig, SpotLight};
use roxlap_scene::{GridTransform, Scene};
use winit::keyboard::KeyCode;

use crate::scene_api::{
    frame_params, opticast_settings, CameraPose, DemoScene, SceneCtx, SceneInput,
};

// Voxlap-packed `0x80_RR_GG_BB` (high byte = full ambient brightness).
const FLOOR: VoxColor = VoxColor(0x80_6a_6a_74);
const PILLAR: VoxColor = VoxColor(0x80_9a_92_86);

/// Grid origin in world space; grid-local `(x,y,z)` → world `origin + xyz`.
const GRID_ORIGIN: DVec3 = DVec3::new(-48.0, 30.0, 0.0);
/// Floor top at grid-local z=60 (voxlap z is down). World floor centre ≈
/// `(0, 78, 60)`; the pillar cluster sits around it.
const FLOOR_TOP_Z: i32 = 60;

/// Baked-AO depth (a touch of contact darkening under the pillars).
const AO_STRENGTH: f32 = 0.7;
const AO_RADIUS: i32 = 1;

/// A dark room so anything outside the cone reads as near-black.
const DARK_SKY: u32 = 0x0a_0a_14;
const AMBIENT: [f32; 3] = [0.10, 0.10, 0.15];

// Four independent demo toggles (pause / flashlight / sweep / shadows) —
// each a distinct on/off control, not a state better modelled as an enum.
#[allow(clippy::struct_excessive_bools)]
pub struct SpotlightScene {
    scene: Scene,
    /// Seconds since `enter`, driving the searchlight sweep.
    clock: f64,
    paused: bool,
    /// `F`: the spot rides the camera (flashlight) vs a mounted searchlight.
    flashlight: bool,
    /// `O`: sweep the mounted searchlight's axis (ignored in flashlight mode).
    sweep: bool,
    shadows: bool,
    /// Outer cone half-angle in degrees (`[` / `]`); the inner is 8° tighter.
    cone_deg: f32,
    /// The one spot, rebuilt each frame — borrowed into the [`LightRig`].
    spots: Vec<SpotLight>,
}

impl SpotlightScene {
    #[must_use]
    pub fn new() -> Self {
        Self {
            scene: Self::build_terrain(),
            clock: 0.0,
            paused: false,
            flashlight: false,
            sweep: true,
            shadows: true,
            cone_deg: 22.0,
            spots: Vec::new(),
        }
    }

    /// A floor with a cluster of pillars of mixed heights — enough vertical
    /// geometry for the cone to pool on the ground and throw crisp shadows.
    fn build_terrain() -> Scene {
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::at(GRID_ORIGIN));
        let g = scene.grid_mut(id).expect("spotlight grid present");
        // Floor slab (top face up at z=60), 96×96.
        g.set_rect(
            IVec3::new(0, 0, FLOOR_TOP_Z),
            IVec3::new(96, 96, FLOOR_TOP_Z + 3),
            Some(FLOOR),
        );
        // A 3×3 cluster of square pillars of varying heights around the centre.
        let heights = [26, 16, 30, 14, 34, 20, 28, 18, 24];
        for (i, &h) in heights.iter().enumerate() {
            let (cx, cy) = (i % 3, i / 3);
            let px = 30 + cx as i32 * 18;
            let py = 30 + cy as i32 * 18;
            g.set_rect(
                IVec3::new(px, py, FLOOR_TOP_Z - h),
                IVec3::new(px + 8, py + 8, FLOOR_TOP_Z),
                Some(PILLAR),
            );
        }
        crate::scene::bake_ao_pub(&mut scene, AO_STRENGTH, AO_RADIUS);
        scene
    }

    /// World centre of the pillar cluster (just above the floor).
    fn cluster_centre() -> DVec3 {
        GRID_ORIGIN + DVec3::new(48.0, 48.0, f64::from(FLOOR_TOP_Z))
    }

    /// The two cone half-angles: outer = `cone_deg`, inner 8° tighter (a soft
    /// edge), inner floored at 2° so it stays a valid cone.
    fn angles(&self) -> (f32, f32) {
        (((self.cone_deg - 8.0).max(2.0)), self.cone_deg)
    }

    /// Rebuild the single spot for the current frame + mode. In flashlight
    /// mode it sits at the camera aiming along its forward; otherwise it is a
    /// mounted searchlight whose axis optionally sweeps.
    fn rebuild_spot(&mut self, ctx: &SceneCtx) {
        self.spots.clear();
        let (inner, outer) = self.angles();
        let (position, direction) = if self.flashlight {
            // Ride the camera: pos + forward (voxlap-basis camera, f64→f32).
            let cam = ctx.cam.camera();
            (cam.pos.map(|v| v as f32), cam.forward.map(|v| v as f32))
        } else {
            // Mounted ~45 voxels above the cluster (voxlap +z is down).
            let c = Self::cluster_centre();
            let pos = c + DVec3::new(0.0, 0.0, -45.0);
            // Aim down (+z); when sweeping, rotate a lateral tilt in.
            let a = if self.sweep { self.clock * 0.5 } else { 0.0 };
            let dir = DVec3::new(0.45 * a.cos(), 0.45 * a.sin(), 1.0).normalize();
            (
                [pos.x as f32, pos.y as f32, pos.z as f32],
                [dir.x as f32, dir.y as f32, dir.z as f32],
            )
        };
        self.spots.push(SpotLight {
            position,
            direction,
            color: [1.0, 0.96, 0.86], // warm white
            intensity: 6.0,
            radius: 180.0,
            inner_angle_deg: inner,
            outer_angle_deg: outer,
            casts_shadow: self.shadows,
        });
    }
}

impl Default for SpotlightScene {
    fn default() -> Self {
        Self::new()
    }
}

impl DemoScene for SpotlightScene {
    fn name(&self) -> &'static str {
        "Spotlight"
    }

    fn controls(&self) -> &'static str {
        "F flashlight · O sweep · [ ] cone · K shadows · P pause"
    }

    fn start_pose(&self) -> CameraPose {
        CameraPose {
            // In front of + above the cluster, looking +y and tilted down.
            pos: [0.0, 6.0, 26.0],
            yaw: std::f64::consts::FRAC_PI_2,
            pitch: 0.5,
        }
    }

    fn enter(&mut self, ctx: &mut SceneCtx) {
        // A dark room so the cone is the only meaningful light.
        ctx.engine.set_sky(None);
        ctx.engine.set_sky_color(DARK_SKY);
        self.rebuild_spot(ctx);
    }

    fn update(&mut self, ctx: &mut SceneCtx, dt: f64) {
        ctx.cam.fly_free(ctx.input, dt);
        if !self.paused {
            self.clock += dt;
        }
        self.rebuild_spot(ctx);
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
            KeyCode::KeyF => {
                self.flashlight = !self.flashlight;
                eprintln!(
                    "mode = {}",
                    if self.flashlight {
                        "flashlight"
                    } else {
                        "searchlight"
                    }
                );
            }
            KeyCode::KeyO => {
                self.sweep = !self.sweep;
                eprintln!("sweep = {}", if self.sweep { "ON" } else { "OFF" });
            }
            KeyCode::KeyK => {
                self.shadows = !self.shadows;
                eprintln!("shadows = {}", if self.shadows { "ON" } else { "OFF" });
            }
            KeyCode::BracketRight => {
                self.cone_deg = (self.cone_deg + 2.0).min(60.0);
                eprintln!("cone half-angle = {:.0}°", self.cone_deg);
            }
            KeyCode::BracketLeft => {
                self.cone_deg = (self.cone_deg - 2.0).max(6.0);
                eprintln!("cone half-angle = {:.0}°", self.cone_deg);
            }
            KeyCode::KeyP => {
                self.paused = !self.paused;
                eprintln!("sweep {}", if self.paused { "paused" } else { "running" });
            }
            _ => {}
        }
    }

    fn render(&mut self, ctx: &mut SceneCtx) {
        let settings = opticast_settings(ctx.size, ctx.scan_dist);
        let mut frame = frame_params(ctx.engine, &settings, ctx.scan_dist);
        frame.lights = Some(LightRig {
            sun: None,
            points: &[],
            spots: &self.spots,
            // Near-dark fill so the cone reads; smooth (bands 0) keeps the
            // soft cone edge + pool gradient rather than terracing it.
            ambient: AMBIENT,
            shadow_strength: 0.9,
            shadow_bias_voxels: 1.5,
            shadow_max_dist: 256.0,
            bands: 0,
            shadow_tint: [0.04, 0.05, 0.09],
        });
        let camera = ctx.cam.camera();
        ctx.renderer.render(&mut self.scene, &camera, &frame);
    }

    fn hud_lines(&self) -> Vec<String> {
        let (inner, outer) = self.angles();
        vec![
            format!(
                "{} · sweep {} · shadows {} · cone {:.0}°/{:.0}°",
                if self.flashlight {
                    "flashlight"
                } else {
                    "searchlight"
                },
                if self.sweep { "on" } else { "off" },
                if self.shadows { "on" } else { "off" },
                inner,
                outer,
            ),
            "single spot (cone) light — no sun, no points".to_string(),
        ]
    }
}
