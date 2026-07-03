//! Demo-scene API (stage DS) — the contract every selectable showcase
//! scene implements, plus the host context it's handed each frame.
//!
//! The host (`main.rs`) owns the window, [`SceneRenderer`], egui, FPS, the
//! shared fly-camera + mouse-look, and the scene menu; a [`DemoScene`]
//! owns its world content + per-scene update / input / render / overlays.
//! See `PORTING-DEMO-SCENES.md`.
//!
//! The trait + context are defined here; the host and the concrete scenes
//! consume them.

use std::f64::consts::PI;

use roxlap_core::opticast::OpticastSettings;
use roxlap_core::{Camera, Engine};
use roxlap_render::{FrameParams, SceneRenderer};

use winit::event::MouseButton;
use winit::keyboard::KeyCode;

/// Per-frame opticast settings every scene shares: the full 6-mip ladder
/// with the host's `+`/`-` ray-march distance.
#[must_use]
pub fn opticast_settings(size: (u32, u32), scan_dist: i32) -> OpticastSettings {
    let mut s = OpticastSettings::for_oracle_framebuffer(size.0, size.1);
    s.max_scan_dist = scan_dist;
    s.mip_levels = 6;
    s.mip_scan_dist = 64;
    s
}

/// Per-frame [`FrameParams`] every scene shares. Sky/fog/shading come
/// from the host `engine`. QE.2 — both backends project from
/// `settings`, and the GPU mip-scan distance / step budget moved to
/// [`roxlap_render::RenderOptions`] / derivation (the
/// `ROXLAP_GPU_MIP_SCAN_DIST` env override still works — the facade
/// reads it at construction).
#[must_use]
pub fn frame_params<'a>(
    engine: &'a Engine,
    settings: &'a OpticastSettings,
    scan_dist: i32,
) -> FrameParams<'a> {
    let mut frame = FrameParams::new(settings);
    frame.sky_color = engine.sky_color();
    frame.sky = engine.sky();
    frame.fog_color = engine.sky_color();
    frame.fog_max_scan_dist = scan_dist;
    frame.side_shades = engine.side_shades();
    frame
}

/// Fly speed (voxels/sec) and look sensitivity shared by every scene.
const MOVE_SPEED: f64 = 64.0;
const FAST_MULT: f64 = 4.0;
const MOUSE_SENS: f64 = 0.0025;
/// Pitch clamped just shy of ±90° so the camera basis stays conditioned.
const PITCH_LIMIT: f64 = 88.0 * PI / 180.0;

/// Held movement-key state, polled by [`CameraRig::fly_delta`]. The host
/// updates the flags from key events; scenes read them.
#[derive(Default, Clone, Copy)]
#[allow(clippy::struct_excessive_bools)]
pub struct InputState {
    pub forward: bool,
    pub back: bool,
    pub left: bool,
    pub right: bool,
    pub up: bool,
    pub down: bool,
    pub fast: bool,
}

/// The shared fly-camera: position + yaw/pitch. The host applies mouse-look
/// ([`apply_look`](Self::apply_look)); each scene's `update` applies
/// movement ([`fly_delta`](Self::fly_delta)) — free or collision-checked as
/// the scene sees fit.
#[derive(Clone, Copy, Debug)]
pub struct CameraRig {
    pub pos: [f64; 3],
    pub yaw: f64,
    pub pitch: f64,
}

impl CameraRig {
    #[must_use]
    pub fn from_pose(p: CameraPose) -> Self {
        Self {
            pos: p.pos,
            yaw: p.yaw,
            pitch: p.pitch,
        }
    }

    /// Build the voxlap-basis [`Camera`] for the current pose.
    #[must_use]
    pub fn camera(&self) -> Camera {
        crate::scene::camera_for_yaw_pitch(self.pos, self.yaw, self.pitch)
    }

    /// Apply a mouse-look delta (pixels) to yaw/pitch.
    pub fn apply_look(&mut self, dx: f64, dy: f64) {
        self.yaw += dx * MOUSE_SENS;
        self.pitch = (self.pitch + dy * MOUSE_SENS).clamp(-PITCH_LIMIT, PITCH_LIMIT);
    }

    /// The world-space movement delta this frame from the held keys —
    /// forward/back along the camera's forward, left/right along its right,
    /// up/down along world ∓z (voxlap z-down). Diagonals are normalised so
    /// two-key combos don't move √2× faster. The scene applies the delta
    /// (directly, or after a collision slide).
    #[must_use]
    pub fn fly_delta(&self, input: InputState, dt: f64) -> [f64; 3] {
        let cam = self.camera();
        let mut d = [0.0f64; 3];
        let mut add = |v: [f64; 3], s: f64| {
            d[0] += v[0] * s;
            d[1] += v[1] * s;
            d[2] += v[2] * s;
        };
        if input.forward {
            add(cam.forward, 1.0);
        }
        if input.back {
            add(cam.forward, -1.0);
        }
        if input.right {
            add(cam.right, 1.0);
        }
        if input.left {
            add(cam.right, -1.0);
        }
        if input.up {
            d[2] -= 1.0; // -z is up
        }
        if input.down {
            d[2] += 1.0;
        }
        let mag2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        if mag2 <= 1e-9 {
            return [0.0; 3];
        }
        let speed = MOVE_SPEED * if input.fast { FAST_MULT } else { 1.0 };
        let k = speed * dt / mag2.sqrt();
        [d[0] * k, d[1] * k, d[2] * k]
    }

    /// `pos + fly_delta` — the free-fly convenience for scenes that don't
    /// collision-check.
    pub fn fly_free(&mut self, input: InputState, dt: f64) {
        let d = self.fly_delta(input, dt);
        self.pos = [self.pos[0] + d[0], self.pos[1] + d[1], self.pos[2] + d[2]];
    }
}

/// The camera pose a scene starts at (set by the host on `enter`).
#[derive(Clone, Copy, Debug)]
pub struct CameraPose {
    pub pos: [f64; 3],
    pub yaw: f64,
    pub pitch: f64,
}

/// A one-shot, scene-local input event. The host consumes the universal
/// inputs first (movement keys, mouse-look, `Tab` menu, `F1` HUD, `Esc`)
/// and forwards the rest here.
#[derive(Clone, Copy, Debug)]
pub enum SceneInput {
    Key { code: KeyCode, pressed: bool },
    Mouse { button: MouseButton, pressed: bool },
    CursorMoved { x: f64, y: f64 },
}

/// What the host lends a [`DemoScene`] each frame: the renderer, the shared
/// camera rig, held input, framebuffer size, the engine (sky/fog), and the
/// host-owned scan distance.
pub struct SceneCtx<'a> {
    pub renderer: &'a mut SceneRenderer,
    pub cam: &'a mut CameraRig,
    pub input: InputState,
    pub size: (u32, u32),
    pub engine: &'a mut Engine,
    /// Host-owned ray-march distance (the `+`/`-` keys); scenes that don't
    /// care ignore it.
    pub scan_dist: i32,
}

/// One selectable demo scene. The host calls `enter` when it becomes
/// active (just after resetting the renderer's content layers via an empty
/// `set_sprites`), then `update` + `render` each frame, `on_input` for
/// scene-local events, and `exit` on switch-away.
pub trait DemoScene {
    /// Menu label.
    fn name(&self) -> &'static str;
    /// One-line controls help, shown in the HUD.
    fn controls(&self) -> &'static str;

    /// Become active: build the world + register sprites/clips/characters.
    fn enter(&mut self, ctx: &mut SceneCtx);
    /// Where the camera starts (the host sets the rig to this on `enter`).
    fn start_pose(&self) -> CameraPose;

    /// Per-frame: apply movement, advance animation/streaming, tick clocks.
    fn update(&mut self, ctx: &mut SceneCtx, dt: f64);

    /// Render this scene's world (`renderer.render` + post overlays). Does
    /// **not** present — the host finishes the frame with egui/present.
    fn render(&mut self, ctx: &mut SceneCtx);

    /// A scene-local input event (universal keys already handled).
    fn on_input(&mut self, _ctx: &mut SceneCtx, _ev: &SceneInput) {}

    /// Drop scene-specific non-renderer state. The host resets the
    /// renderer's content layers itself on switch.
    fn exit(&mut self, _ctx: &mut SceneCtx) {}

    /// Extra HUD lines (live state) under the host's FPS/pos/backend block.
    fn hud_lines(&self) -> Vec<String> {
        Vec::new()
    }
}
