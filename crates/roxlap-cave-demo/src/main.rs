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
//! CD.8.3 will add fog + collision detection.

use std::num::NonZeroU32;
use std::rc::Rc;
use std::time::Instant;

use roxlap_cavegen::{BlueCaveGenerator, CaveParams, Generator, MagCaveGenerator, MAXZDIM};
use roxlap_core::opticast;
use roxlap_core::rasterizer::ScanScratch;
use roxlap_core::scalar_rasterizer::ScalarRasterizer;
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

/// Pixel radius of the bullet's screen-space billboard.
const BULLET_RADIUS_PX: i32 = 3;

/// Bullet colour (softbuffer u32 = `0x00_RR_GG_BB`). Bright
/// magenta-cyan plasma; visible against both blue and mag caves.
const BULLET_COLOR_CORE: u32 = 0x00FF_FFFF;
const BULLET_COLOR_HALO: u32 = 0x00C0_C0FF;

/// Voxlap colour stamped on the inner walls of the carved crater
/// (the voxels that were buried before the carve and are now newly
/// exposed). Charred dark grey with a faint orange tint reads as
/// "scorched rock" against both blue and mag cave palettes.
/// Encoded as `(brightness << 24) | (R << 16) | (G << 8) | B` per
/// voxlap convention; brightness `0x80` is voxlap's neutral.
#[allow(clippy::cast_possible_wrap)]
const CARVE_COLOR: i32 = 0x8050_3018u32 as i32;

/// In-flight plasma bullet. Travels in a straight line until it hits
/// a solid voxel (carved into a sphere via [`set_sphere`]) or exits
/// the world / max-flight envelope.
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
    scratch: ScanScratch,
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
        let vxl = build_world(preset, seed);

        let engine = Engine::new();
        let scratch = ScanScratch::new_for_size(WIDTH, HEIGHT, VSID);

        // Spawn the camera at world centre, midway up. The cave-gen
        // produces solid + air mixed throughout, so the camera might
        // start inside solid — that's fine for v1; CD.8.3 adds
        // collision + a "find the largest air pocket" spawn search.
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
            scratch,
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
    /// longer exist).
    fn regenerate(&mut self) {
        self.vxl = build_world(self.preset, self.seed);
        self.bullets.clear();
    }

    /// Build a [`Camera`] from the current `cam_pos` + `yaw` + `pitch`.
    fn camera(&self) -> Camera {
        // Voxlap's right-handed basis with z growing downward:
        //   forward = (cos(yaw) cos(pitch), sin(yaw) cos(pitch), sin(pitch))
        //   right = (-sin(yaw), cos(yaw), 0)         (= ist)
        //   down  = right × forward                  (= ihe)
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        let forward = [cy * cp, sy * cp, sp];
        let right = [-sy, cy, 0.0];
        // down = right × forward
        let down = [
            right[1] * forward[2] - right[2] * forward[1],
            right[2] * forward[0] - right[0] * forward[2],
            right[0] * forward[1] - right[1] * forward[0],
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
        if mag > 1e-6 {
            for i in 0..3 {
                self.cam_pos[i] += delta[i] / mag * speed * dt;
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
            if b.travelled > BULLET_MAX_DIST {
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
        if self.scratch.uurend_half_stride < size.width as usize {
            self.scratch = ScanScratch::new_for_size(size.width, size.height, self.vxl.vsid);
        }

        let sky_col_i = i32::from_ne_bytes(self.engine.sky_color().to_ne_bytes());
        self.scratch.set_skycast(sky_col_i, 0);
        let fog_col_i = i32::from_ne_bytes(self.engine.fog_color().to_ne_bytes());
        self.scratch
            .set_fog(fog_col_i, self.engine.fog_max_scan_dist());
        let s = self.engine.side_shades();
        self.scratch
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
                &mut self.scratch,
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

/// Generate a fresh world for the given preset + seed and reserve
/// edit headroom so runtime [`set_sphere`] carves work.
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
    // Filled disc with halo. Outer ring (radius BULLET_RADIUS_PX +
    // 1) gets a softer plasma colour; inner core gets the bright
    // one.
    let r_outer = BULLET_RADIUS_PX + 1;
    let r_outer_sq = r_outer * r_outer;
    let r_inner_sq = BULLET_RADIUS_PX * BULLET_RADIUS_PX;
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
