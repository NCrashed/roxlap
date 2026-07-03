//! The **Doom** scene (stage BB): Doom/Build-style **GIF billboard sprites**
//! — flat, camera-facing animated cutouts that are real lighting citizens
//! (they cast + receive the dynamic sun's shadows). Every sprite here is
//! synthesised as an animated **GIF** at startup and fed back through the
//! `roxlap-render` GIF importer (`gif_import`), so the scene dogfoods the
//! full `GIF → VoxelClip → flat voxel slab` path without shipping a binary
//! asset.
//!
//! Showcases:
//! - **`BillboardActor`** (BB.4) — an 8-directional walking monster: the
//!   renderer picks the directional clip from the view angle (orbit it, or
//!   turn it with `Q`/`E`, and the shown rotation changes — each direction
//!   has a distinct hue marker), and plays a named state (`walk`/`idle`,
//!   `Space` toggles). It casts a shadow on the floor and is shadowed by the
//!   pillars.
//! - A **1-directional animated billboard actor** — a flickering flame that
//!   does not cast a shadow (per-actor `casts_shadow = false`, BB.3).
//! - A **standalone billboard** (BB.2) — a static signpost oriented by
//!   `face_billboards_to`, whose **lighting mode** (BB.2b: `FaceNormal` /
//!   `WorldUp` / `AmbientOnly` / `FullBright`) cycles with `L` (the flame is
//!   `FullBright` — emissive).
//!
//! Controls: WASD+mouse fly · `Q`/`E` turn the monster · `Space` walk/idle ·
//! `K` toggle sun shadows · `L` cycle the signpost's lighting mode.

use glam::{DVec3, IVec3};
use roxlap_render::{
    gif_import::{voxel_clip_from_gif, GifImportOpts},
    ActorState, BillboardActorDef, BillboardActorId, BillboardLighting, BillboardMode,
    DirectionalLight, LightRig, SpriteInstanceId, VoxelClipId,
};
use roxlap_scene::{GridTransform, Scene};
use winit::keyboard::KeyCode;

use crate::scene_api::{
    frame_params, opticast_settings, CameraPose, DemoScene, SceneCtx, SceneInput,
};

const GRASS: u32 = 0x80_4d_8a_3a;
const STONE: u32 = 0x80_8a_8a_92;

/// Grid origin in world space (grid-local `xyz` → world `origin + xyz`).
const GRID_ORIGIN: DVec3 = DVec3::new(-40.0, 30.0, 0.0);
/// Floor top is at grid-local z=60 (voxlap z is down). World floor top z=60.
const FLOOR_TOP_Z: i32 = 60;

/// Monster sprite size in pixels (= voxels per slab face).
const FIG_W: usize = 16;
const FIG_H: usize = 22;
const FLAME_W: usize = 10;
const FLAME_H: usize = 14;
const NDIR: usize = 8;

pub struct DoomScene {
    scene: Scene,
    clock: f64,
    sun_shadows: bool,
    monster: Option<BillboardActorId>,
    monster_pos: [f32; 3],
    monster_yaw: f64,
    walking: bool,
    standalone: Option<SpriteInstanceId>,
    /// BB.2b — the standalone billboard's lighting mode, cycled by `L`.
    standalone_light: BillboardLighting,
    /// BB.2b — the monster actor's lighting mode, cycled by `O` (dogfoods
    /// the runtime `set_actor_lighting`).
    monster_light: BillboardLighting,
}

impl DoomScene {
    #[must_use]
    pub fn new() -> Self {
        Self {
            scene: Self::build_terrain(),
            clock: 0.0,
            sun_shadows: true,
            monster: None,
            monster_pos: [0.0, 70.0, f32::from(FLOOR_TOP_Z as u8)],
            monster_yaw: -std::f64::consts::FRAC_PI_2, // face the start camera
            walking: true,
            standalone: None,
            standalone_light: BillboardLighting::FaceNormal,
            monster_light: BillboardLighting::FaceNormal,
        }
    }

    /// A grass floor with two stone pillars — vertical geometry for the sun
    /// to cast pillar shadows onto the billboards, and for the billboards to
    /// cast their own shadows onto the floor.
    fn build_terrain() -> Scene {
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::at(GRID_ORIGIN));
        let g = scene.grid_mut(id).expect("doom grid present");
        g.set_rect(
            IVec3::new(0, 0, FLOOR_TOP_Z),
            IVec3::new(80, 100, FLOOR_TOP_Z + 3),
            Some(GRASS),
        );
        for (px, py) in [(14, 58), (54, 58)] {
            g.set_rect(
                IVec3::new(px, py, 28),
                IVec3::new(px + 10, py + 10, FLOOR_TOP_Z),
                Some(STONE),
            );
        }
        crate::scene::bake_ao_pub(&mut scene, 0.8, 1);
        scene
    }

    /// The sun's world travel direction (sky → ground), azimuth sweeping with
    /// the clock so the billboard shadows rotate across the floor.
    fn sun_direction(&self) -> [f32; 3] {
        let a = self.clock * 0.35;
        // +z is down (voxlap), so a positive z keeps the sun above the horizon.
        let v = DVec3::new(0.65 * a.cos(), 0.65 * a.sin(), 0.6).normalize();
        [v.x as f32, v.y as f32, v.z as f32]
    }

    /// Register an animated GIF (built from RGBA `frames`) as a voxel clip,
    /// dogfooding the full encode → import → voxelize path. Panics on a
    /// malformed synthetic GIF (a demo-build bug, not user input).
    fn clip_from_frames(
        ctx: &mut SceneCtx,
        w: usize,
        h: usize,
        frames: &[Vec<u8>],
        delay_cs: u16,
    ) -> VoxelClipId {
        let bytes = encode_gif(w as u16, h as u16, frames, delay_cs);
        let opts = GifImportOpts {
            thickness: 2, // a little depth so the cast shadow has a body
            ..Default::default()
        };
        let clip = voxel_clip_from_gif(&bytes, &opts).expect("synthetic GIF decodes");
        let decoded = clip.decode().expect("clip decodes");
        ctx.renderer.add_voxel_clip(&decoded)
    }
}

impl DemoScene for DoomScene {
    fn name(&self) -> &'static str {
        "Doom"
    }

    fn controls(&self) -> &'static str {
        "WASD fly · Q/E turn monster · Space walk/idle · K sun shadows · L signpost light · O monster light"
    }

    fn start_pose(&self) -> CameraPose {
        CameraPose {
            pos: [0.0, 28.0, 46.0],
            yaw: std::f64::consts::FRAC_PI_2, // look +y toward the monster
            pitch: 0.3,
        }
    }

    fn enter(&mut self, ctx: &mut SceneCtx) {
        // 8-directional walk + idle clips. Each direction gets a distinct hue
        // marker so the directional selection is unmistakable as you orbit.
        let mut walk = Vec::with_capacity(NDIR);
        let mut idle = Vec::with_capacity(NDIR);
        for d in 0..NDIR {
            let wf = vec![figure_rgba(d, NDIR, 0), figure_rgba(d, NDIR, 1)];
            walk.push(Self::clip_from_frames(ctx, FIG_W, FIG_H, &wf, 14));
            idle.push(Self::clip_from_frames(
                ctx,
                FIG_W,
                FIG_H,
                &[figure_rgba(d, NDIR, 0)],
                20,
            ));
        }
        let def = BillboardActorDef {
            states: vec![
                ActorState {
                    name: "walk",
                    dirs: walk,
                },
                ActorState {
                    name: "idle",
                    dirs: idle,
                },
            ],
            mode: BillboardMode::Cylindrical,
            lighting: BillboardLighting::FaceNormal,
            speed_q8: 256,
            casts_shadow: true,
            receives_shadow: true,
        };
        self.monster = ctx
            .renderer
            .add_billboard_actor(def, self.monster_pos, self.monster_yaw);

        // A flickering flame: a 1-directional, multi-frame billboard actor
        // that does NOT cast a shadow (a flat glow shouldn't).
        let flame_frames: Vec<Vec<u8>> = (0..4).map(flame_rgba).collect();
        let flame = Self::clip_from_frames(ctx, FLAME_W, FLAME_H, &flame_frames, 8);
        let fire_def = BillboardActorDef {
            states: vec![ActorState {
                name: "burn",
                dirs: vec![flame],
            }],
            mode: BillboardMode::Cylindrical,
            // A flame is emissive — full-bright so it glows instead of being
            // dimmed by the scene's ambient/sun.
            lighting: BillboardLighting::FullBright,
            speed_q8: 256,
            casts_shadow: false,
            receives_shadow: false,
        };
        ctx.renderer
            .add_billboard_actor(fire_def, [22.0, 86.0, f32::from(FLOOR_TOP_Z as u8)], 0.0);

        // A standalone billboard (BB.2 path): a static signpost oriented by
        // face_billboards_to each frame.
        let totem = Self::clip_from_frames(ctx, FIG_W, FIG_H, &[totem_rgba()], 100);
        self.standalone = ctx.renderer.add_billboard_instance(
            totem,
            [-24.0, 84.0, f32::from(FLOOR_TOP_Z as u8)],
            BillboardMode::Cylindrical,
        );
    }

    fn update(&mut self, ctx: &mut SceneCtx, dt: f64) {
        ctx.cam.fly_free(ctx.input, dt);
        self.clock += dt;
        // One tick drives the directional actors (pick dir-clip +
        // animate + face camera) and orients the standalone billboard.
        ctx.renderer.tick(&ctx.cam.camera(), dt);
    }

    fn on_input(&mut self, ctx: &mut SceneCtx, ev: &SceneInput) {
        let SceneInput::Key {
            code,
            pressed: true,
        } = ev
        else {
            return;
        };
        let Some(monster) = self.monster else { return };
        match code {
            KeyCode::KeyK => {
                self.sun_shadows = !self.sun_shadows;
                eprintln!(
                    "sun shadows = {}",
                    if self.sun_shadows { "ON" } else { "OFF" }
                );
            }
            KeyCode::KeyQ | KeyCode::KeyE => {
                let step = if matches!(code, KeyCode::KeyQ) {
                    -0.39
                } else {
                    0.39
                };
                self.monster_yaw += step;
                ctx.renderer
                    .set_actor_transform(monster, self.monster_pos, self.monster_yaw);
            }
            KeyCode::Space => {
                self.walking = !self.walking;
                ctx.renderer
                    .set_actor_state(monster, if self.walking { "walk" } else { "idle" });
            }
            KeyCode::KeyL => {
                // BB.2b — cycle the standalone billboard's lighting mode.
                self.standalone_light = cycle_lighting(self.standalone_light);
                if let Some(s) = self.standalone {
                    ctx.renderer
                        .set_sprite_instance_lighting(s, self.standalone_light);
                }
                eprintln!("signpost lighting = {:?}", self.standalone_light);
            }
            KeyCode::KeyO => {
                // BB.2b — cycle the monster actor's lighting via set_actor_lighting.
                self.monster_light = cycle_lighting(self.monster_light);
                ctx.renderer.set_actor_lighting(monster, self.monster_light);
                eprintln!("monster lighting = {:?}", self.monster_light);
            }
            _ => {}
        }
    }

    fn render(&mut self, ctx: &mut SceneCtx) {
        let settings = opticast_settings(ctx.size, ctx.scan_dist);
        let mut frame = frame_params(ctx.engine, &settings, ctx.scan_dist);
        let sun = DirectionalLight {
            direction: self.sun_direction(),
            color: [1.0, 0.92, 0.8],
            intensity: 1.3,
            casts_shadow: self.sun_shadows,
        };
        frame.lights = Some(LightRig {
            sun: Some(sun),
            points: &[],
            spots: &[],
            ambient: [0.34, 0.36, 0.42],
            shadow_strength: 0.85,
            shadow_bias_voxels: 1.5,
            shadow_max_dist: 256.0,
            bands: 6,
            shadow_tint: [0.16, 0.2, 0.34],
        });
        let camera = ctx.cam.camera();
        ctx.renderer.render(&mut self.scene, &camera, &frame);
    }

    fn hud_lines(&self) -> Vec<String> {
        vec![
            format!(
                "monster: {} · sun shadows {}",
                if self.walking { "walk" } else { "idle" },
                if self.sun_shadows { "on" } else { "off" },
            ),
            "GIF billboards: import → flat voxel slab, cast + receive shadows".to_string(),
            format!(
                "lighting — signpost {:?} (L) · monster {:?} (O)",
                self.standalone_light, self.monster_light
            ),
        ]
    }
}

/// Cycle a [`BillboardLighting`] mode FaceNormal → WorldUp → AmbientOnly →
/// FullBright → … (BB.2b demo toggles).
fn cycle_lighting(m: BillboardLighting) -> BillboardLighting {
    match m {
        BillboardLighting::FaceNormal => BillboardLighting::WorldUp,
        BillboardLighting::WorldUp => BillboardLighting::AmbientOnly,
        BillboardLighting::AmbientOnly => BillboardLighting::FullBright,
        BillboardLighting::FullBright => BillboardLighting::FaceNormal,
    }
}

// ---- procedural sprite art (synthesised as GIFs, then imported) ----------

/// HSV → RGB (each 0..=1 in, bytes out).
#[allow(
    clippy::many_single_char_names,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn hsv(h: f32, s: f32, v: f32) -> [u8; 3] {
    let h6 = (h.fract() + 1.0).fract() * 6.0;
    let i = h6.floor() as i32;
    let f = h6 - i as f32;
    let (p, q, t) = (v * (1.0 - s), v * (1.0 - s * f), v * (1.0 - s * (1.0 - f)));
    let (r, g, b) = match i.rem_euclid(6) {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
}

/// A simple humanoid viewed from direction `dir` (of `ndir`): grey body, a
/// per-direction hue marker that rotates around the head, and a 2-pose walk
/// cycle (`pose` 0 = legs together, 1 = legs apart).
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
fn figure_rgba(dir: usize, ndir: usize, pose: usize) -> Vec<u8> {
    let (w, h) = (FIG_W, FIG_H);
    let mut px = vec![0u8; w * h * 4];
    let mut put = |x: i32, y: i32, c: [u8; 4]| {
        if x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h {
            let i = ((y as usize) * w + x as usize) * 4;
            px[i..i + 4].copy_from_slice(&c);
        }
    };
    let body = [110u8, 110, 122, 255];
    let limb = [80u8, 80, 92, 255];
    let skin = [222u8, 196, 168, 255];
    for y in 2..7 {
        for x in 6..10 {
            put(x, y, skin); // head
        }
    }
    for y in 7..15 {
        for x in 5..11 {
            put(x, y, body); // torso
        }
    }
    for y in 8..14 {
        put(4, y, limb); // arms
        put(11, y, limb);
    }
    let (lx, rx) = if pose == 1 { (4, 11) } else { (6, 9) };
    for y in 15..21 {
        put(lx, y, limb);
        put(lx + 1, y, limb);
        put(rx, y, limb);
        put(rx + 1, y, limb);
    }
    // Direction marker: a 2×2 hue block on the head rim at the view angle.
    let m = hsv(dir as f32 / ndir as f32, 0.95, 1.0);
    let ang = (dir as f32 / ndir as f32) * std::f32::consts::TAU;
    let (mx, my) = (
        8 + (3.0 * ang.cos()).round() as i32,
        4 + (3.0 * ang.sin()).round() as i32,
    );
    for dy in 0..2 {
        for dx in 0..2 {
            put(mx + dx, my + dy, [m[0], m[1], m[2], 255]);
        }
    }
    px
}

/// A flickering teardrop flame (frame `f` of the loop).
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn flame_rgba(f: usize) -> Vec<u8> {
    let (w, h) = (FLAME_W, FLAME_H);
    let mut px = vec![0u8; w * h * 4];
    let mut put = |x: i32, y: i32, c: [u8; 4]| {
        if x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h {
            let i = ((y as usize) * w + x as usize) * 4;
            px[i..i + 4].copy_from_slice(&c);
        }
    };
    let top = 1 + (f % 2) as i32; // flicker the tip height
    let cx = w as i32 / 2;
    for y in top..h as i32 {
        let prog = (y - top) as f32 / (h as i32 - top) as f32;
        let half = (0.8 + prog * 3.6) as i32;
        let wobble = i32::from((y + f as i32) % 2 != 0);
        for x in (cx - half)..=(cx + half) {
            let edge = ((x - cx).abs() as f32) / (half as f32 + 0.01);
            let c = if edge < 0.4 {
                [255, 242, 150, 255]
            } else if edge < 0.8 {
                [255, 158, 42, 255]
            } else {
                [208, 70, 20, 255]
            };
            put(x + wobble, y, c);
        }
    }
    px
}

/// A static signpost (single-frame), for the standalone-billboard path.
#[allow(clippy::cast_possible_wrap)]
fn totem_rgba() -> Vec<u8> {
    let (w, h) = (FIG_W, FIG_H);
    let mut px = vec![0u8; w * h * 4];
    let mut put = |x: i32, y: i32, c: [u8; 4]| {
        if x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h {
            let i = ((y as usize) * w + x as usize) * 4;
            px[i..i + 4].copy_from_slice(&c);
        }
    };
    for y in 6..h as i32 {
        for x in 7..9 {
            put(x, y, [120, 80, 40, 255]); // post
        }
    }
    for y in 2..7 {
        for x in 3..13 {
            put(x, y, [200, 180, 60, 255]); // board
        }
    }
    for y in 3..6 {
        put(7, y, [60, 40, 20, 255]); // a mark
        put(8, y, [60, 40, 20, 255]);
    }
    px
}

/// Encode RGBA `frames` (`w×h`, alpha 0 = transparent) into in-memory GIF
/// bytes — the demo's synthetic billboard assets.
fn encode_gif(w: u16, h: u16, frames: &[Vec<u8>], delay_cs: u16) -> Vec<u8> {
    let mut out = Vec::new();
    {
        let mut enc = gif::Encoder::new(&mut out, w, h, &[]).expect("gif encoder");
        enc.set_repeat(gif::Repeat::Infinite).ok();
        for frame in frames {
            let mut buf = frame.clone();
            let mut fr = gif::Frame::from_rgba(w, h, &mut buf);
            fr.delay = delay_cs;
            enc.write_frame(&fr).expect("gif frame");
        }
    }
    out
}
