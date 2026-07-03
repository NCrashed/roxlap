//! The **Lighting** scene (DL stage): runtime dynamic lighting — a sweeping
//! coloured **sun** that casts hard shadows, plus three orbiting coloured
//! **point lights** (two shadow-casting, one not). The diffuse stylized
//! lighting (sun + points + cel + ramp, flat per voxel) **and** the hard
//! voxel shadows run on **both** backends — GPU per-pixel (`shade_lit` +
//! `shadow_occluded`), CPU per-(voxel,face) via the render Sampler (CPU.1 +
//! CPU.2). The CPU shadow march is the slow fallback's slowest path but is
//! on visual parity with the GPU.
//!
//! Layout (camera looks +y): a grass floor with four stone pillars and a
//! central monument. The sun rotates overhead so the pillars' shadows sweep
//! across the floor; the point lights orbit just above it, pooling coloured
//! light that the shadow-casters occlude behind the pillars.
//!
//! The baked brightness byte acts as a dim **ambient** fill (locked
//! decision #2); the sun + point lights are the runtime key/fill. See
//! `PORTING-DYNLIGHT.md`.
//!
//! Controls: WASD+mouse fly · `P` pause the sun · `K` toggle sun shadows ·
//! `L` toggle point lights.

use glam::{DVec3, IVec3};
use roxlap_render::{
    DirectionalLight, DynSpriteTransform, Kv6, LightRig, LoopMode, PointLight, VoxelClip,
};
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

/// AO.2 — baked-AO depth (`N`/`M` retune live) + a fixed tight 1-voxel reach.
const AO_STRENGTH_DEFAULT: f32 = 0.85;
const AO_RADIUS: i32 = 1;

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
    /// DL.6 — stylized (cel + hue-ramp + flat per-voxel) vs smooth. `J`.
    stylized: bool,
    /// DL.6 — cel band count when stylized; `[` / `]` adjust (clamped 2..16).
    bands: u32,
    /// AO.2 — baked AO depth; `N`/`M` retune (re-bakes the scene).
    ao_strength: f32,
    /// The point lights, rebuilt each frame (orbit) — borrowed into the
    /// per-frame [`LightRig`] at render time.
    points: Vec<PointLight>,
}

impl LightingScene {
    #[must_use]
    pub fn new() -> Self {
        Self {
            scene: Self::build_terrain(AO_STRENGTH_DEFAULT),
            clock: 0.0,
            paused: false,
            sun_shadows: true,
            points_on: true,
            stylized: true,
            bands: 6,
            ao_strength: AO_STRENGTH_DEFAULT,
            points: Vec::new(),
        }
    }

    /// AO.2 — re-bake the scene's ambient occlusion at the current strength
    /// (cheap: one 96×96 chunk). Called when `N`/`M` retune it.
    fn rebake_ao(&mut self) {
        crate::scene::bake_ao_pub(&mut self.scene, self.ao_strength, AO_RADIUS);
    }

    /// A grass floor with four stone pillars + a central monument — enough
    /// vertical geometry for the sun's shadows to sweep across the ground.
    fn build_terrain(ao_strength: f32) -> Scene {
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
        // AO stage — bake ambient occlusion into the brightness byte, which
        // the stylized dynamic lighting reads as its ambient/AO fill: pillar
        // bases, the monument's foot, and inner corners darken even where no
        // sun/point light directly shades them. `strength` (N/M) sets depth.
        crate::scene::bake_ao_pub(&mut scene, ao_strength, AO_RADIUS);
        scene
    }

    /// World centre the point lights orbit around (just above the floor).
    fn orbit_centre() -> DVec3 {
        GRID_ORIGIN + DVec3::new(48.0, 48.0, f64::from(FLOOR_TOP_Z) - 12.0)
    }

    /// A solid voxel sphere — a curved surface shows the cel ramp + the
    /// coloured point-light pools far better than the flat floor.
    fn sphere_kv6(dim: u32, color: u32) -> Kv6 {
        let c = (dim as f32 - 1.0) * 0.5;
        let r2 = (dim as f32 * 0.46).powi(2);
        Kv6::from_fn(dim, dim, dim, |x, y, z| {
            let (dx, dy, dz) = (x as f32 - c, y as f32 - c, z as f32 - c);
            (dx * dx + dy * dy + dz * dz <= r2).then_some(color)
        })
    }

    /// An axis-aligned sprite pose at a world position (identity basis).
    fn pose(pos: [f32; 3]) -> DynSpriteTransform {
        DynSpriteTransform {
            pos,
            right: [1.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
            forward: [0.0, 0.0, 1.0],
        }
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
        "WASD fly · P: pause sun · K: shadows · L: points · J: stylized · [ ]: bands · N/M: AO depth"
    }

    fn start_pose(&self) -> CameraPose {
        CameraPose {
            // Above + in front of the floor, looking +y and tilted down.
            pos: [0.0, -10.0, 22.0],
            yaw: std::f64::consts::FRAC_PI_2,
            pitch: 0.42,
        }
    }

    fn enter(&mut self, ctx: &mut SceneCtx) {
        self.rebuild_points();
        // Showcase that the stylized lighting also covers sprites + clips
        // (DL.4/DL.6): a static voxel sphere and a pulsing voxel clip standing
        // on the floor, lit by the same sun + orbiting point lights as the
        // terrain. Curved surfaces read the cel ramp + coloured pools clearly.
        let c = Self::orbit_centre();
        let ball = ctx
            .renderer
            .add_sprite_model(&Self::sphere_kv6(18, 0x80_c8_c8_cc));
        ctx.renderer.add_sprite_instance_posed(
            ball,
            Self::pose([(c.x - 22.0) as f32, c.y as f32, (c.z - 6.0) as f32]),
        );
        // A pulsing-radius sphere clip (.rvc path) — same shader as sprites,
        // so it gets the stylized lighting too.
        let frames: Vec<Kv6> = [5.0_f32, 7.0, 9.0, 7.0]
            .iter()
            .map(|&r| {
                let dim = 20u32;
                let cc = (dim as f32 - 1.0) * 0.5;
                let r2 = r * r;
                Kv6::from_fn(dim, dim, dim, |x, y, z| {
                    let (dx, dy, dz) = (x as f32 - cc, y as f32 - cc, z as f32 - cc);
                    (dx * dx + dy * dy + dz * dz <= r2).then_some(0x80_d0_a0_50)
                })
            })
            .collect();
        if let Ok(clip) = VoxelClip::from_kv6_frames(&frames, 1.0, LoopMode::Loop, &[], 150, 1) {
            if let Ok(decoded) = clip.decode() {
                let id = ctx.renderer.add_voxel_clip(&decoded);
                ctx.renderer.add_clip_instance_playing(
                    id,
                    Self::pose([(c.x + 22.0) as f32, c.y as f32, (c.z - 8.0) as f32]),
                    1.0,
                    0,
                );
            }
        }
    }

    fn update(&mut self, ctx: &mut SceneCtx, dt: f64) {
        // Shared free-fly camera (WASD + Space/Shift); no collision here.
        ctx.cam.fly_free(ctx.input, dt);
        ctx.renderer.tick(&ctx.cam.camera(), dt);
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
                    if self.stylized {
                        "stylized (cel+ramp)"
                    } else {
                        "smooth"
                    }
                );
            }
            KeyCode::BracketRight => {
                self.bands = (self.bands + 1).min(16);
                eprintln!("cel bands = {}", self.bands);
            }
            KeyCode::BracketLeft => {
                self.bands = self.bands.saturating_sub(1).max(2);
                eprintln!("cel bands = {}", self.bands);
            }
            KeyCode::KeyM => {
                self.ao_strength = (self.ao_strength + 0.1).min(1.0);
                self.rebake_ao();
                eprintln!("AO strength = {:.2}", self.ao_strength);
            }
            KeyCode::KeyN => {
                self.ao_strength = (self.ao_strength - 0.1).max(0.0);
                self.rebake_ao();
                eprintln!("AO strength = {:.2}", self.ao_strength);
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
            spots: &[],
            // Dim ambient so the runtime sun/points read as the key light
            // (the baked byte is the ambient/AO channel, locked decision #2).
            ambient: [0.32, 0.34, 0.4],
            shadow_strength: 0.85,
            shadow_bias_voxels: 1.5,
            shadow_max_dist: 256.0,
            // DL.6 — stylized cel + hue-shifted ramp + flat per-voxel (J
            // toggles vs smooth; [ / ] change the band count). Cool shadow
            // tint → warm sun.
            bands: if self.stylized { self.bands } else { 0 },
            shadow_tint: [0.16, 0.2, 0.34],
        });
        let camera = ctx.cam.camera();
        ctx.renderer.render(&mut self.scene, &camera, &frame);
    }

    fn hud_lines(&self) -> Vec<String> {
        vec![
            format!(
                "sun {} · shadows {} · points {} · {}{}",
                if self.paused { "paused" } else { "sweeping" },
                if self.sun_shadows { "on" } else { "off" },
                if self.points_on { "on" } else { "off" },
                if self.stylized { "stylized" } else { "smooth" },
                if self.stylized {
                    format!(" ({} bands)", self.bands)
                } else {
                    String::new()
                },
            ),
            format!("baked AO depth {:.2} (N/M)", self.ao_strength),
            "dynamic lighting + shadows on both backends".to_string(),
        ]
    }
}
