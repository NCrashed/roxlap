//! roxlap-cave-demo — procedural-cave showcase.
//!
//! Generates a [`BlueCaveGenerator`] world on startup, opens a
//! winit + softbuffer window, and renders the cave via the roxlap
//! engine's `opticast` rasterizer. Fly through with WASD + mouse-
//! look; click in the window to grab the cursor.
//!
//! Controls:
//! - Click in the window → grab cursor (mouse-look active).
//! - `W`/`A`/`S`/`D` → forward / strafe-left / back / strafe-right.
//! - `Space` → up (world `-z`); `LShift` → down (world `+z`).
//! - Hold `LCtrl` for fast-fly (≈4× speed).
//! - `LMB` (while grabbed) → fire a plasma bullet that flies along
//!   camera-forward and carves a sphere into the world on impact.
//! - `F` → toggle blue ↔ mag cave preset (regenerates the world).
//! - `R` → regenerate the world with the next seed (preset preserved).
//! - `Esc` → release cursor (or exit if already released).
//! - Window close → exit.
//!
//! Movement is collision-checked: the camera slides along walls
//! instead of clipping through them.

use std::num::NonZeroU32;
use std::rc::Rc;
use std::time::Instant;

use roxlap_cavegen::{BlueCaveGenerator, CaveParams, Generator, MagCaveGenerator, MAXZDIM};
use roxlap_core::opticast;
use roxlap_core::rasterizer::ScratchPool;
use roxlap_core::scalar_rasterizer::ScalarRasterizer;
use roxlap_core::update_lighting;
use roxlap_core::world_query::{getcube, Cube};
use roxlap_core::Camera;
use roxlap_core::Engine;
use roxlap_core::OpticastSettings;
use roxlap_formats::edit::{set_sphere_with_colfunc, SpanOp};
use roxlap_formats::vxl;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{DeviceEvent, DeviceId, ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

/// World dimension. 128 keeps cave-gen at startup under ~2 s on a
/// modern laptop while still showing a recognisable cave network.
const VSID: u32 = 128;

/// Walking speed (voxels / second).
const MOVE_SPEED: f64 = 32.0;
/// Multiplier applied while `LCtrl` is held.
const FAST_MULT: f64 = 4.0;
/// Mouse sensitivity (radians per pixel of cursor delta).
const MOUSE_SENS: f64 = 0.0025;
/// Pitch is clamped just shy of ±90° to keep the basis well-conditioned.
const PITCH_LIMIT: f64 = 88.0_f64 * std::f64::consts::PI / 180.0;

/// Maximum bullet flight distance, in voxel units. Bullets that
/// exceed this without hitting are despawned silently.
const BULLET_MAX_DIST: f64 = 96.0;

/// Bullet velocity, in voxels / second. Tuned so a bullet fired
/// across an open chamber takes ~1 s to land — visible "plasma
/// bolt" arc, not an instant ray.
const BULLET_VEL: f64 = 60.0;

/// Sphere carve radius applied at bullet impact, in voxels.
const FIRE_RADIUS: u32 = 4;

/// Initial pixel radius of the bullet's screen-space billboard.
/// The bullet shrinks linearly with travelled distance so the
/// player gets a depth cue even when firing along the camera's
/// view axis (where parallax doesn't help).
const BULLET_RADIUS_PX_MAX: i32 = 12;

/// Bullet despawns when its scaled radius drops below this.
const BULLET_RADIUS_PX_MIN: i32 = 1;

/// Bullet colour (softbuffer u32 = `0x00_RR_GG_BB`). Bright
/// pink core + softer halo; visible against both blue and mag
/// cave palettes.
const BULLET_COLOR_CORE: u32 = 0x00FF_4080;
const BULLET_COLOR_HALO: u32 = 0x00FF_A0C0;

/// Voxlap colour stamped on the inner walls of the carved crater
/// (the voxels that were buried before the carve and are now newly
/// exposed). Charred dark grey with a faint orange tint reads as
/// "scorched rock" against both blue and mag cave palettes.
/// Encoded as `(brightness << 24) | (R << 16) | (G << 8) | B` per
/// voxlap convention; brightness `0x80` is voxlap's neutral.
#[allow(clippy::cast_possible_wrap)]
const CARVE_COLOR: i32 = 0x8050_3018u32 as i32;

/// Voxlap colour used when carving the spawn bubble — neutral mid-
/// grey that doesn't betray the carve's source as much as the
/// scorched-amber `CARVE_COLOR`.
#[allow(clippy::cast_possible_wrap)]
const SPAWN_BUBBLE_COLOR: i32 = 0x8060_6068u32 as i32;

/// Radius of the carve performed at world centre so the camera
/// always spawns inside an open pocket (cave-gen otherwise leaves
/// the centre randomly air or solid).
const SPAWN_BUBBLE_RADIUS: u32 = 6;

/// Voxlap lightmode driving `update_lighting`'s per-voxel
/// brightness bake.
///
/// - `0` → no bake (default; scene reads as flat colour scaled
///   only by side-shading).
/// - `1` → directional sun-style bake: every visible voxel gets
///   `(tp.y * 0.5 + tp.z) * 64 + 103.5` from its surface normal,
///   clamped to `[0, 255]`. Smooth gradient across the whole
///   cave, baseline brightness ~104 (close to voxlap's neutral
///   128). No `LightSrc` consulted.
/// - `2` → per-light Lambertian point-source bake. The contribution
///   per light is `g · h · sc` where `g = 1/d³ − 1/r³`, so the
///   cube falloff drops to ~0 within ~64 voxels even at large
///   `sc`. Suitable for tight rooms with dense light placement;
///   not great for a 128³ cave from a handful of point lights.
///   Empirically tested at sc=5000 — still went black at ~64
///   voxels from any light.
///
/// Mode 1 reads better in this demo (smooth, every wall lit) and
/// `relight_bbox` still re-bakes carved regions correctly because
/// `update_lighting` recomputes surface normals after the edit.
const LIGHTMODE: u32 = 1;

/// Fog colour (BGR low-24-bit; brightness bit MUST be 0 per
/// `project_oracle_fog_disabled.md` — `set_fogcol(BR(rgb))` with
/// the brightness bit set silently disables fog because
/// `vx5.fogcol < 0` short-circuits opticast's fog-enabled path).
const FOG_COLOR: u32 = 0x0090_98B0;

/// Fog "max scan distance" in voxels. At this distance pixels
/// blend fully to `FOG_COLOR`. 128 voxels at vsid=128 is dense
/// enough to dim distant cave walls without obscuring nearby ones.
const FOG_MAX_SCAN_DIST: i32 = 128;

/// Effective camera "skin" radius in voxel units. Movement is
/// blocked when any voxel intersected by a ±`PLAYER_RADIUS` cube
/// around the proposed new position is solid — this keeps the
/// camera off walls instead of letting it touch them at sub-pixel
/// distance. Small enough that the camera fits through the
/// `SPAWN_BUBBLE_RADIUS = 6` carved bubble + bullet-impact craters.
const PLAYER_RADIUS: f64 = 0.3;

/// In-flight plasma bullet. Travels in a straight line until it hits
/// a solid voxel (carved into a sphere via [`set_sphere_with_colfunc`])
/// or exits the world / max-flight envelope.
#[derive(Debug, Clone, Copy)]
struct Bullet {
    pos: [f64; 3],
    vel: [f64; 3],
    /// Distance travelled so far; bullet despawns past `BULLET_MAX_DIST`.
    travelled: f64,
}

/// Which cave-gen preset is currently active. F toggles between them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Preset {
    Blue,
    Mag,
}

impl Preset {
    fn name(self) -> &'static str {
        match self {
            Self::Blue => "blue",
            Self::Mag => "mag",
        }
    }

    fn next(self) -> Self {
        match self {
            Self::Blue => Self::Mag,
            Self::Mag => Self::Blue,
        }
    }

    fn default_params(self) -> CaveParams {
        match self {
            Self::Blue => BlueCaveGenerator::default_params(),
            Self::Mag => MagCaveGenerator::default_params(),
        }
    }

    /// Generate a Vxl with this preset's pipeline + the given seed.
    fn generate(self, seed: u64) -> roxlap_formats::vxl::Vxl {
        let mut params = self.default_params();
        params.seed = seed;
        match self {
            Self::Blue => BlueCaveGenerator.generate(&params, VSID),
            Self::Mag => MagCaveGenerator.generate(&params, VSID),
        }
    }
}

/// Movement key flags packed into a single byte. Bit layout matches
/// the order of [`KeyCode`] queries in the input handler.
#[derive(Default, Clone, Copy)]
struct KeyState(u8);

impl KeyState {
    const FWD: u8 = 1 << 0;
    const BACK: u8 = 1 << 1;
    const LEFT: u8 = 1 << 2;
    const RIGHT: u8 = 1 << 3;
    const UP: u8 = 1 << 4;
    const DOWN: u8 = 1 << 5;
    const FAST: u8 = 1 << 6;

    fn set(&mut self, mask: u8, on: bool) {
        if on {
            self.0 |= mask;
        } else {
            self.0 &= !mask;
        }
    }
    fn has(self, mask: u8) -> bool {
        self.0 & mask != 0
    }
}

struct App {
    window: Option<Rc<Window>>,
    surface: Option<softbuffer::Surface<Rc<Window>, Rc<Window>>>,
    engine: Engine,
    zbuffer: Vec<f32>,
    pool: ScratchPool,
    vxl: vxl::Vxl,
    cam_pos: [f64; 3],
    yaw: f64,
    pitch: f64,
    keys: KeyState,
    grabbed: bool,
    last_tick: Option<Instant>,
    /// Active cave-gen preset; toggled by `F`.
    preset: Preset,
    /// Current world seed; bumped by `R` (preset preserved).
    seed: u64,
    /// Bullets currently in flight. Spawned on LMB while grabbed,
    /// integrated each frame, removed on impact / out-of-bounds /
    /// past `BULLET_MAX_DIST`.
    bullets: Vec<Bullet>,
}

impl App {
    fn new() -> Self {
        let preset = Preset::Blue;
        let seed = preset.default_params().seed;
        let mut vxl = build_world(preset, seed);

        // Carve a guaranteed-open spawn bubble at world centre.
        // Cave-gen otherwise leaves the centre randomly air or
        // solid, so the camera frequently spawns inside a wall.
        let centre = spawn_centre();
        set_sphere_with_colfunc(
            &mut vxl,
            centre,
            SPAWN_BUBBLE_RADIUS,
            SpanOp::Carve,
            |_, _, _| SPAWN_BUBBLE_COLOR,
        );

        let mut engine = Engine::new();
        // Fog: low-24-bit colour (no brightness bit — see
        // `project_oracle_fog_disabled.md`).
        engine.set_fog(FOG_COLOR, FOG_MAX_SCAN_DIST);

        // Directional sun-style lighting (lightmode 1). No
        // `LightSrc` needed — the bake reads only the voxel's
        // surface normal. Initial whole-world bake at startup;
        // bullet impacts re-bake just the local edit region in
        // `step_bullets` so dynamic carves stay correctly lit.
        engine.set_lightmode(LIGHTMODE);
        relight_world(&mut vxl, &engine);

        let pool = ScratchPool::new(WIDTH, HEIGHT, VSID);

        // Spawn the camera at the carved bubble's centre.
        let cam_pos = [
            f64::from(VSID) * 0.5,
            f64::from(VSID) * 0.5,
            f64::from(MAXZDIM) * 0.5,
        ];

        Self {
            window: None,
            surface: None,
            engine,
            zbuffer: Vec::new(),
            pool,
            vxl,
            cam_pos,
            yaw: 0.0,
            pitch: 0.0,
            keys: KeyState::default(),
            grabbed: false,
            last_tick: None,
            preset,
            seed,
            bullets: Vec::new(),
        }
    }

    /// Rebuild the world from `self.preset` + `self.seed`. Resets
    /// in-flight bullets (the old voxels they were aimed at no
    /// longer exist) and re-carves the spawn bubble at world
    /// centre + teleports the camera back to it.
    fn regenerate(&mut self) {
        let mut vxl = build_world(self.preset, self.seed);
        let centre = spawn_centre();
        set_sphere_with_colfunc(
            &mut vxl,
            centre,
            SPAWN_BUBBLE_RADIUS,
            SpanOp::Carve,
            |_, _, _| SPAWN_BUBBLE_COLOR,
        );
        // Re-bake the static spawn-bubble light over the fresh
        // world. The light's position is pinned to the spawn
        // bubble centre so the world-space coords don't change;
        // only the voxel data does.
        relight_world(&mut vxl, &self.engine);
        self.vxl = vxl;
        self.cam_pos = [
            f64::from(VSID) * 0.5,
            f64::from(VSID) * 0.5,
            f64::from(MAXZDIM) * 0.5,
        ];
        self.bullets.clear();
    }

    /// Build a [`Camera`] from the current `cam_pos` + `yaw` + `pitch`.
    fn camera(&self) -> Camera {
        // Voxlap's right-handed basis with z growing downward.
        // Per `feedback_voxlap_basis_chirality.md`: the camera's
        // down vector must satisfy `right × down == forward`. The
        // cyclic relation is `forward × right = down` (NOT `right
        // × forward`, which gives -down and silently inverts the
        // image vertically + flips yaw direction). My initial port
        // got the cross order wrong, hence the "bottom is top"
        // bug.
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        let forward = [cy * cp, sy * cp, sp];
        let right = [-sy, cy, 0.0];
        // down = forward × right
        let down = [
            forward[1] * right[2] - forward[2] * right[1],
            forward[2] * right[0] - forward[0] * right[2],
            forward[0] * right[1] - forward[1] * right[0],
        ];
        Camera {
            pos: self.cam_pos,
            right,
            down,
            forward,
        }
    }

    /// Advance camera position by `dt` seconds based on which movement
    /// keys are currently held.
    #[allow(clippy::needless_range_loop)]
    fn integrate(&mut self, dt: f64) {
        if dt <= 0.0 {
            return;
        }
        let mut speed = MOVE_SPEED;
        if self.keys.has(KeyState::FAST) {
            speed *= FAST_MULT;
        }
        let cam = self.camera();
        let mut delta = [0.0f64; 3];
        if self.keys.has(KeyState::FWD) {
            for i in 0..3 {
                delta[i] += cam.forward[i];
            }
        }
        if self.keys.has(KeyState::BACK) {
            for i in 0..3 {
                delta[i] -= cam.forward[i];
            }
        }
        if self.keys.has(KeyState::RIGHT) {
            for i in 0..3 {
                delta[i] += cam.right[i];
            }
        }
        if self.keys.has(KeyState::LEFT) {
            for i in 0..3 {
                delta[i] -= cam.right[i];
            }
        }
        if self.keys.has(KeyState::UP) {
            // `-z` is up in voxlap convention.
            delta[2] -= 1.0;
        }
        if self.keys.has(KeyState::DOWN) {
            delta[2] += 1.0;
        }
        // Normalise diagonal motion so two-key combos don't move √2× faster.
        let mag = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
        if mag <= 1e-6 {
            return;
        }
        let step = [
            delta[0] / mag * speed * dt,
            delta[1] / mag * speed * dt,
            delta[2] / mag * speed * dt,
        ];

        // Per-axis collision: try each axis independently so the
        // camera slides along walls instead of jamming when one
        // component of movement collides. If the camera is already
        // inside solid (e.g., regen edge case), skip the block test
        // for that axis so we can still escape.
        let already_stuck = is_blocked(&self.vxl, self.cam_pos);
        for axis in 0..3 {
            let mut candidate = self.cam_pos;
            candidate[axis] += step[axis];
            if already_stuck || !is_blocked(&self.vxl, candidate) {
                self.cam_pos[axis] = candidate[axis];
            }
        }
    }

    /// LMB fire — spawn a plasma bullet that flies along the
    /// camera's forward axis at `BULLET_VEL` voxels/sec. The bullet
    /// is integrated each frame and carves a sphere at impact.
    fn fire(&mut self) {
        let cam = self.camera();
        // Spawn slightly ahead of the camera to avoid "shooting
        // yourself in the chin" if the player is brushing a wall.
        let pos = [
            cam.pos[0] + cam.forward[0] * 0.5,
            cam.pos[1] + cam.forward[1] * 0.5,
            cam.pos[2] + cam.forward[2] * 0.5,
        ];
        let vel = [
            cam.forward[0] * BULLET_VEL,
            cam.forward[1] * BULLET_VEL,
            cam.forward[2] * BULLET_VEL,
        ];
        self.bullets.push(Bullet {
            pos,
            vel,
            travelled: 0.0,
        });
    }

    /// Integrate bullet positions, check collision, and apply impact
    /// carves. Removes bullets that hit, exit world bounds, or fly
    /// past `BULLET_MAX_DIST`.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn step_bullets(&mut self, dt: f64) {
        if dt <= 0.0 {
            return;
        }
        let vsid = self.vxl.vsid;
        // Accumulate impact carves; apply after the iteration so we
        // don't borrow self.vxl mutably while iterating self.bullets.
        let mut impacts: Vec<[i32; 3]> = Vec::new();
        self.bullets.retain_mut(|b| {
            let dx = b.vel[0] * dt;
            let dy = b.vel[1] * dt;
            let dz = b.vel[2] * dt;
            b.pos[0] += dx;
            b.pos[1] += dy;
            b.pos[2] += dz;
            b.travelled += (dx * dx + dy * dy + dz * dz).sqrt();
            // Bullet shrinks linearly with travelled distance; despawn
            // when its on-screen radius would fall below the minimum.
            if bullet_radius_px(b) < BULLET_RADIUS_PX_MIN {
                return false;
            }
            let vx = b.pos[0].floor() as i32;
            let vy = b.pos[1].floor() as i32;
            let vz = b.pos[2].floor() as i32;
            // Out of world.
            if vx < 0
                || vy < 0
                || (vx as u32) >= vsid
                || (vy as u32) >= vsid
                || !(0..MAXZDIM).contains(&vz)
            {
                return false;
            }
            // Solid hit?
            if !matches!(
                getcube(&self.vxl.data, &self.vxl.column_offset, vsid, vx, vy, vz),
                Cube::Air
            ) {
                impacts.push([vx, vy, vz]);
                return false;
            }
            true
        });
        for hit in impacts {
            // Carve a sphere; newly-exposed voxels (= the inner
            // crater walls that were previously buried solid) take
            // CARVE_COLOR. Without an explicit colfunc, set_sphere's
            // default returns 0 (black) which makes craters look
            // like missing data.
            set_sphere_with_colfunc(
                &mut self.vxl,
                hit,
                FIRE_RADIUS,
                SpanOp::Carve,
                |_x, _y, _z| CARVE_COLOR,
            );
            // Local relight: voxlap's per-voxel intensity depends
            // on surface normal + line-of-sight to each light. A
            // carve changes both for voxels around the impact, so
            // re-bake just the carve's bbox rather than the whole
            // world. update_lighting pads the bbox by ESTNORMRAD
            // internally, so we only need the geometric edit
            // bounds here.
            relight_bbox(&mut self.vxl, &self.engine, hit, FIRE_RADIUS);
        }
    }

    fn redraw(&mut self) {
        let Some(window) = self.window.as_ref() else {
            return;
        };
        let size = window.inner_size();
        let (Some(w_nz), Some(h_nz)) = (NonZeroU32::new(size.width), NonZeroU32::new(size.height))
        else {
            return;
        };

        // Advance camera by real wall-clock dt — clamped so a long
        // stall doesn't teleport the camera on the next frame.
        let now = Instant::now();
        let dt = self
            .last_tick
            .map_or(0.0, |t| (now - t).as_secs_f64().min(0.1));
        self.last_tick = Some(now);
        self.integrate(dt);
        self.step_bullets(dt);

        // Resize zbuffer + scratch if window changed.
        let pixel_count = (size.width as usize) * (size.height as usize);
        if self.zbuffer.len() < pixel_count {
            self.zbuffer.resize(pixel_count, 0.0);
        }
        if self.pool.slot(0).uurend_half_stride < size.width as usize {
            self.pool = ScratchPool::new(size.width, size.height, self.vxl.vsid);
        }

        let sky_col_i = i32::from_ne_bytes(self.engine.sky_color().to_ne_bytes());
        self.pool.set_skycast(sky_col_i, 0);
        let fog_col_i = i32::from_ne_bytes(self.engine.fog_color().to_ne_bytes());
        self.pool
            .set_fog(fog_col_i, self.engine.fog_max_scan_dist());
        let s = self.engine.side_shades();
        self.pool
            .set_side_shades(s[0], s[1], s[2], s[3], s[4], s[5]);

        let cam = self.camera();
        let sky = self.engine.sky_color();
        let settings = OpticastSettings::for_oracle_framebuffer(size.width, size.height);
        let pitch_pixels = size.width as usize;

        let Some(surface) = self.surface.as_mut() else {
            return;
        };
        surface.resize(w_nz, h_nz).expect("softbuffer: resize");
        let mut buffer = surface.buffer_mut().expect("softbuffer: buffer_mut");
        for px in buffer.iter_mut() {
            *px = sky;
        }

        {
            let mut rasterizer = ScalarRasterizer::new(
                &mut buffer,
                &mut self.zbuffer,
                pitch_pixels,
                &self.vxl.data,
                &self.vxl.column_offset,
                &self.vxl.mip_base_offsets,
                self.vxl.vsid,
            );
            let _ = opticast(
                &mut rasterizer,
                &mut self.pool,
                &cam,
                &settings,
                self.vxl.vsid,
                &self.vxl.data,
                &self.vxl.column_offset,
            );
        }

        // Plasma-bullet billboards on top of the rasterized scene.
        // Each bullet projects to a screen pixel; if its view-space
        // depth is closer than the zbuffer there, draw a small filled
        // disc of plasma colour.
        for bullet in &self.bullets {
            draw_bullet(
                &mut buffer,
                &self.zbuffer,
                size.width,
                size.height,
                &cam,
                &settings,
                bullet,
            );
        }

        buffer.present().expect("softbuffer: present");
    }

    fn set_grabbed(&mut self, grabbed: bool) {
        let Some(window) = self.window.as_ref() else {
            return;
        };
        if grabbed {
            // Linux+X11 only supports Confined; Wayland+macOS only
            // Locked. Try Locked first, fall back to Confined.
            let r = window
                .set_cursor_grab(CursorGrabMode::Locked)
                .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));
            if r.is_ok() {
                window.set_cursor_visible(false);
                self.grabbed = true;
            }
        } else {
            let _ = window.set_cursor_grab(CursorGrabMode::None);
            window.set_cursor_visible(true);
            self.grabbed = false;
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_title("roxlap-cave-demo")
            .with_inner_size(LogicalSize::new(WIDTH, HEIGHT));
        let window = event_loop.create_window(attrs).expect("create window");
        let window = Rc::new(window);
        let context = softbuffer::Context::new(window.clone()).expect("softbuffer: Context::new");
        let surface =
            softbuffer::Surface::new(&context, window.clone()).expect("softbuffer: Surface::new");
        self.window = Some(window);
        self.surface = Some(surface);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                self.redraw();
                if let Some(w) = self.window.as_ref() {
                    w.request_redraw();
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                if self.grabbed {
                    self.fire();
                } else {
                    self.set_grabbed(true);
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state,
                        ..
                    },
                ..
            } => {
                let pressed = state == ElementState::Pressed;
                match code {
                    KeyCode::KeyW => self.keys.set(KeyState::FWD, pressed),
                    KeyCode::KeyS => self.keys.set(KeyState::BACK, pressed),
                    KeyCode::KeyA => self.keys.set(KeyState::LEFT, pressed),
                    KeyCode::KeyD => self.keys.set(KeyState::RIGHT, pressed),
                    KeyCode::Space => self.keys.set(KeyState::UP, pressed),
                    KeyCode::ShiftLeft => self.keys.set(KeyState::DOWN, pressed),
                    KeyCode::ControlLeft => self.keys.set(KeyState::FAST, pressed),
                    KeyCode::Escape if pressed => {
                        if self.grabbed {
                            self.set_grabbed(false);
                        } else {
                            event_loop.exit();
                        }
                    }
                    KeyCode::KeyF if pressed => {
                        self.preset = self.preset.next();
                        eprintln!("cave-demo: switched to preset = {}", self.preset.name());
                        self.regenerate();
                    }
                    KeyCode::KeyR if pressed => {
                        self.seed = self.seed.wrapping_add(1);
                        eprintln!(
                            "cave-demo: regenerating with preset = {}, seed = {}",
                            self.preset.name(),
                            self.seed
                        );
                        self.regenerate();
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _id: DeviceId, event: DeviceEvent) {
        if !self.grabbed {
            return;
        }
        if let DeviceEvent::MouseMotion { delta } = event {
            self.yaw += delta.0 * MOUSE_SENS;
            self.pitch = (self.pitch + delta.1 * MOUSE_SENS).clamp(-PITCH_LIMIT, PITCH_LIMIT);
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(w) = self.window.as_ref() {
            w.request_redraw();
        }
    }
}

/// Test whether any voxel intersected by a ±[`PLAYER_RADIUS`] cube
/// around `pos` is solid (or out-of-world). Out-of-bounds in any
/// axis counts as solid so the camera can't fly off the edge of
/// the cave-gen volume.
///
/// Cheap: at `PLAYER_RADIUS = 0.3` the cube spans at most 1 voxel
/// per axis (typically), so this fans out to 1-8 [`getcube`] calls.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
fn is_blocked(vxl: &vxl::Vxl, pos: [f64; 3]) -> bool {
    let lo_x = (pos[0] - PLAYER_RADIUS).floor() as i32;
    let hi_x = (pos[0] + PLAYER_RADIUS).floor() as i32;
    let lo_y = (pos[1] - PLAYER_RADIUS).floor() as i32;
    let hi_y = (pos[1] + PLAYER_RADIUS).floor() as i32;
    let lo_z = (pos[2] - PLAYER_RADIUS).floor() as i32;
    let hi_z = (pos[2] + PLAYER_RADIUS).floor() as i32;
    let vsid = vxl.vsid as i32;
    for vz in lo_z..=hi_z {
        for vy in lo_y..=hi_y {
            for vx in lo_x..=hi_x {
                if vx < 0 || vy < 0 || vz < 0 || vx >= vsid || vy >= vsid || vz >= MAXZDIM {
                    return true;
                }
                if !matches!(
                    getcube(&vxl.data, &vxl.column_offset, vxl.vsid, vx, vy, vz),
                    Cube::Air
                ) {
                    return true;
                }
            }
        }
    }
    false
}

/// World-centre spawn point in integer voxel coords.
#[allow(clippy::cast_possible_wrap)]
fn spawn_centre() -> [i32; 3] {
    [(VSID / 2) as i32, (VSID / 2) as i32, MAXZDIM / 2]
}

/// Rebake voxel intensities for the entire world under the
/// engine's current `lightmode` + `lights`. Run once at
/// startup + after `regenerate`; per-frame relight is too
/// expensive (~ms per call at VSID=128 / MAXZDIM=256).
fn relight_world(vxl: &mut vxl::Vxl, engine: &Engine) {
    #[allow(clippy::cast_possible_wrap)]
    update_lighting(
        &mut vxl.data,
        &vxl.column_offset,
        vxl.vsid,
        0,
        0,
        0,
        vxl.vsid as i32,
        vxl.vsid as i32,
        MAXZDIM,
        engine.lightmode(),
        engine.lights(),
    );
}

/// Rebake voxel intensities in just the bounding box around a
/// sphere edit `[centre - radius, centre + radius + 1]`,
/// clamped to world bounds. `update_lighting` itself pads by
/// `ESTNORMRAD` internally so estnorm has enough neighbourhood
/// — we only need the geometric edit extent here.
#[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
fn relight_bbox(vxl: &mut vxl::Vxl, engine: &Engine, centre: [i32; 3], radius: u32) {
    let r = radius as i32;
    let vsid_i = vxl.vsid as i32;
    let x0 = (centre[0] - r).max(0);
    let y0 = (centre[1] - r).max(0);
    let z0 = (centre[2] - r).max(0);
    let x1 = (centre[0] + r + 1).min(vsid_i);
    let y1 = (centre[1] + r + 1).min(vsid_i);
    let z1 = (centre[2] + r + 1).min(MAXZDIM);
    if x0 >= x1 || y0 >= y1 || z0 >= z1 {
        return;
    }
    update_lighting(
        &mut vxl.data,
        &vxl.column_offset,
        vxl.vsid,
        x0,
        y0,
        z0,
        x1,
        y1,
        z1,
        engine.lightmode(),
        engine.lights(),
    );
}

/// Scaled bullet billboard radius — shrinks linearly from
/// `BULLET_RADIUS_PX_MAX` at spawn to 0 at `BULLET_MAX_DIST`. The
/// shrink gives the player a depth cue when firing along the view
/// axis (where parallax doesn't help).
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn bullet_radius_px(bullet: &Bullet) -> i32 {
    let t = (bullet.travelled / BULLET_MAX_DIST).clamp(0.0, 1.0);
    let r = (1.0 - t) * f64::from(BULLET_RADIUS_PX_MAX);
    r.round() as i32
}

/// Generate a fresh world for the given preset + seed and reserve
/// edit headroom so runtime [`set_sphere_with_colfunc`] carves work.
fn build_world(preset: Preset, seed: u64) -> vxl::Vxl {
    eprintln!(
        "cave-demo: generating {} world (vsid={VSID}, seed={seed})…",
        preset.name()
    );
    let t0 = Instant::now();
    let mut vxl = preset.generate(seed);
    // 4 MB headroom covers ~thousands of carve impacts before the
    // slab pool fragments enough to overflow.
    vxl.reserve_edit_capacity(4 * 1024 * 1024);
    eprintln!(
        "cave-demo: world generated in {:.2}s",
        t0.elapsed().as_secs_f32()
    );
    vxl
}

/// Project `bullet`'s world position to screen, depth-test against
/// the framebuffer's zbuffer, and write a small filled-disc plasma
/// billboard if visible.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::too_many_arguments
)]
fn draw_bullet(
    buffer: &mut [u32],
    zbuffer: &[f32],
    width: u32,
    height: u32,
    cam: &Camera,
    settings: &OpticastSettings,
    bullet: &Bullet,
) {
    // World → camera-relative.
    let rel = [
        bullet.pos[0] - cam.pos[0],
        bullet.pos[1] - cam.pos[1],
        bullet.pos[2] - cam.pos[2],
    ];
    // View basis projection.
    let view_x = rel[0] * cam.right[0] + rel[1] * cam.right[1] + rel[2] * cam.right[2];
    let view_y = rel[0] * cam.down[0] + rel[1] * cam.down[1] + rel[2] * cam.down[2];
    let view_z = rel[0] * cam.forward[0] + rel[1] * cam.forward[1] + rel[2] * cam.forward[2];
    if view_z < 0.5 {
        // Behind / inside the camera.
        return;
    }
    let inv_z = 1.0 / view_z;
    let su = f64::from(settings.hx) + view_x * inv_z * f64::from(settings.hz);
    let sv = f64::from(settings.hy) + view_y * inv_z * f64::from(settings.hz);
    let cx = su.round() as i32;
    let cy = sv.round() as i32;
    if cx < 0 || cy < 0 || cx >= width as i32 || cy >= height as i32 {
        return;
    }
    // Z test against the centre pixel — close enough for a small
    // billboard. Skip the bullet if there's solid in front.
    let zb_idx = (cy as usize) * (width as usize) + (cx as usize);
    if zbuffer[zb_idx] < view_z as f32 {
        return;
    }
    // Filled disc with halo, shrinking with bullet travel so
    // along-axis depth is readable. r_inner is the current scaled
    // radius; r_outer adds a one-pixel halo.
    let r_inner = bullet_radius_px(bullet);
    let r_outer = r_inner + 1;
    let r_outer_sq = r_outer * r_outer;
    let r_inner_sq = r_inner * r_inner;
    for dy in -r_outer..=r_outer {
        for dx in -r_outer..=r_outer {
            let d_sq = dx * dx + dy * dy;
            if d_sq > r_outer_sq {
                continue;
            }
            let px = cx + dx;
            let py = cy + dy;
            if px < 0 || py < 0 || px >= width as i32 || py >= height as i32 {
                continue;
            }
            let idx = (py as usize) * (width as usize) + (px as usize);
            buffer[idx] = if d_sq <= r_inner_sq {
                BULLET_COLOR_CORE
            } else {
                BULLET_COLOR_HALO
            };
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app)?;
    Ok(())
}
