//! roxlap-cave-demo — procedural-cave showcase.
//!
//! Generates a [`BlueCaveGenerator`] world on startup, opens a
//! winit + softbuffer window, and renders the cave via the roxlap
//! engine's `opticast` rasterizer. Fly through with WASD + mouse-
//! look; click in the window to grab the cursor.
//!
//! Controls (CD.8.0 — minimal viable build):
//! - Click in the window → grab cursor (mouse-look active).
//! - `W`/`A`/`S`/`D` → forward / strafe-left / back / strafe-right.
//! - `Space` → up (world `-z`); `LShift` → down (world `+z`).
//! - Hold `LCtrl` for fast-fly (≈4× speed).
//! - `Esc` → release cursor (or exit if already released).
//! - Window close → exit.
//!
//! CD.8.1+ will wire LMB sphere carve, F preset toggle, R
//! regenerate, and fog. This commit lands the minimum-viable
//! cave-on-startup pipeline.

use std::num::NonZeroU32;
use std::rc::Rc;
use std::time::Instant;

use roxlap_cavegen::{BlueCaveGenerator, Generator, MAXZDIM};
use roxlap_core::opticast;
use roxlap_core::rasterizer::ScanScratch;
use roxlap_core::scalar_rasterizer::ScalarRasterizer;
use roxlap_core::world_query::{getcube, Cube};
use roxlap_core::Camera;
use roxlap_core::Engine;
use roxlap_core::OpticastSettings;
use roxlap_formats::edit::set_sphere;
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

/// Maximum LMB-fire ray distance, in voxel units. ~64 voxels is
/// more than enough for in-cave shooting; rays that travel further
/// without hitting solid usually mean the player is shooting into
/// open space and nothing happens.
const FIRE_MAX_DIST: f64 = 64.0;

/// Sphere carve radius for LMB fire, in voxels.
const FIRE_RADIUS: u32 = 4;

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
}

impl App {
    fn new() -> Self {
        // Generate the cave world. ~1-2 s at VSID=128.
        eprintln!("cave-demo: generating BlueCaveGenerator world (vsid={VSID})…");
        let t0 = Instant::now();
        let mut vxl = BlueCaveGenerator.generate(&BlueCaveGenerator::default_params(), VSID);
        // Reserve headroom for runtime edits (LMB sphere carves). At
        // VSID=128 the cave's column data is ~1-2 MB; 4 MB headroom
        // covers ~thousands of carve impacts before the slab pool
        // fragments enough to overflow.
        vxl.reserve_edit_capacity(4 * 1024 * 1024);
        eprintln!(
            "cave-demo: world generated in {:.2}s",
            t0.elapsed().as_secs_f32()
        );

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
        }
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

    /// LMB fire — cast a ray from the camera along forward, find the
    /// first solid voxel within `FIRE_MAX_DIST`, and carve a sphere
    /// at that voxel. Silently no-ops if the ray exits the world or
    /// hits nothing.
    fn fire(&mut self) {
        let cam = self.camera();
        let Some(hit) = cast_ray(&self.vxl, cam.pos, cam.forward, FIRE_MAX_DIST) else {
            return;
        };
        set_sphere(&mut self.vxl, hit, FIRE_RADIUS, None);
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

/// 3D voxel-grid ray traversal (Amanatides + Woo DDA).
///
/// Walks from `origin` along `dir` (unnormalised; magnitudes
/// don't affect the voxel sequence) one voxel-boundary crossing
/// at a time, calling [`getcube`] to test each voxel. Returns the
/// integer coords of the first non-air voxel within `max_dist`
/// voxel units, or `None` if the ray exits the search volume
/// without hitting anything.
///
/// Skips the voxel containing `origin` — otherwise firing from
/// inside an air pocket whose neighboring voxel is solid would
/// trivially hit the *adjacent* voxel rather than the surface
/// the player aimed at.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn cast_ray(vxl: &vxl::Vxl, origin: [f64; 3], dir: [f64; 3], max_dist: f64) -> Option<[i32; 3]> {
    let mut x = origin[0].floor() as i32;
    let mut y = origin[1].floor() as i32;
    let mut z = origin[2].floor() as i32;

    let step_x: i32 = if dir[0] >= 0.0 { 1 } else { -1 };
    let step_y: i32 = if dir[1] >= 0.0 { 1 } else { -1 };
    let step_z: i32 = if dir[2] >= 0.0 { 1 } else { -1 };

    let inv = |d: f64| {
        if d.abs() > 1e-9 {
            1.0 / d.abs()
        } else {
            f64::INFINITY
        }
    };
    let t_delta = [inv(dir[0]), inv(dir[1]), inv(dir[2])];

    // Initial t to the next voxel boundary along each axis.
    let frac = |o: f64, step: i32| -> f64 {
        let f = o - o.floor();
        if step > 0 {
            1.0 - f
        } else if f == 0.0 {
            // On a boundary, stepping negatively immediately
            // re-enters the previous voxel.
            1.0
        } else {
            f
        }
    };
    let mut t_max = [
        frac(origin[0], step_x) * t_delta[0],
        frac(origin[1], step_y) * t_delta[1],
        frac(origin[2], step_z) * t_delta[2],
    ];

    // Skip the origin voxel — first iteration steps once before the test.
    let max_iters = (max_dist as i32).saturating_mul(3) + 8;
    for _ in 0..max_iters {
        // Step in axis with smallest t_max.
        let t = if t_max[0] < t_max[1] && t_max[0] < t_max[2] {
            x += step_x;
            let crossed = t_max[0];
            t_max[0] += t_delta[0];
            crossed
        } else if t_max[1] < t_max[2] {
            y += step_y;
            let crossed = t_max[1];
            t_max[1] += t_delta[1];
            crossed
        } else {
            z += step_z;
            let crossed = t_max[2];
            t_max[2] += t_delta[2];
            crossed
        };
        if t > max_dist {
            return None;
        }
        if !(0..MAXZDIM).contains(&z) {
            return None;
        }
        match getcube(&vxl.data, &vxl.column_offset, vxl.vsid, x, y, z) {
            Cube::Air => {}
            _ => return Some([x, y, z]),
        }
    }
    None
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app)?;
    Ok(())
}
