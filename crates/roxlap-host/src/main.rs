//! roxlap-host — winit + softbuffer demo host.
//!
//! Opens a window, loads a real `.vxl` world (oracle.vxl.gz from the
//! workspace assets), and on every `RedrawRequested` event runs
//! `opticast` with the `ScalarRasterizer`.
//!
//! Controls:
//! - Click in the window → grab + hide cursor (mouse-look active).
//! - `W` / `A` / `S` / `D` → forward / strafe-left / back / strafe-right.
//! - `Space` → up (world `-z`); `LShift` → down (world `+z`).
//! - Hold `LCtrl` for fast-fly (≈4× speed).
//! - Mouse motion → yaw + pitch while grabbed.
//! - `Esc` → release cursor (or exit if already released).
//! - Window close → exit.

use std::io::Read;
use std::num::NonZeroU32;
use std::rc::Rc;
use std::time::Instant;

use flate2::read::GzDecoder;
use roxlap_core::opticast;
use roxlap_core::rasterizer::ScanScratch;
use roxlap_core::scalar_rasterizer::ScalarRasterizer;
use roxlap_core::Camera;
use roxlap_core::Engine;
use roxlap_core::OpticastSettings;
use roxlap_formats::vxl;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{DeviceEvent, DeviceId, ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

/// Walking speed (voxels / second). Oracle world is 1024-VSID with
/// terrain at z≈128, so 64 vox/sec crosses a typical scene in seconds.
const MOVE_SPEED: f64 = 64.0;
/// Multiplier applied while `LCtrl` is held.
const FAST_MULT: f64 = 4.0;
/// Mouse sensitivity (radians per pixel of cursor delta).
const MOUSE_SENS: f64 = 0.0025;
/// Pitch is clamped just shy of ±90° to keep the basis well-conditioned
/// (a perfectly straight-up/down camera collapses `right × forward`).
const PITCH_LIMIT: f64 = 88.0_f64 * std::f64::consts::PI / 180.0;

/// Embedded gzipped oracle world. Same fixture roxlap-formats uses
/// in its parser tests — no extra disk I/O at startup.
const ORACLE_VXL_GZ: &[u8] = include_bytes!("../../../assets/oracle.vxl.gz");

/// Tracks which movement keys are currently pressed. Polled each
/// frame to integrate position; we don't act on the press/release
/// edge directly because that would tie movement rate to key-repeat.
/// Backed by a bitfield to keep `clippy::struct_excessive_bools` happy.
#[derive(Default, Clone, Copy)]
struct KeyState(u8);

impl KeyState {
    const FORWARD: u8 = 1 << 0;
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
    /// Window handle. Wrapped in `Rc` because softbuffer's `Context`
    /// and `Surface` each take a clone — both need the same handle
    /// type, so we share.
    window: Option<Rc<Window>>,
    surface: Option<softbuffer::Surface<Rc<Window>, Rc<Window>>>,
    engine: Engine,
    /// f32 z-buffer, allocated lazily / re-sized on first redraw and
    /// resized on window-resize.
    zbuffer: Vec<f32>,
    /// `ScanScratch` (radar / angstart / lastx / uurend), reused across
    /// frames. Sized at app construction for the initial window
    /// resolution; resized on window-resize.
    scratch: ScanScratch,
    /// World loaded from `oracle.vxl.gz`. `vxl.data` is the flat
    /// slab buffer, `vxl.column_offset` the per-column byte offsets;
    /// `vxl.vsid` the world dimension; `vxl.ipo`/`ist`/`ihe`/`ifo`
    /// the saved camera.
    vxl: vxl::Vxl,
    /// Camera position in voxel-world units.
    cam_pos: [f64; 3],
    /// Yaw — rotation around the world +z (down) axis. 0 looks +y.
    yaw: f64,
    /// Pitch — rotation around the camera's right axis. 0 = level;
    /// +π/2 = straight down. Clamped to `±PITCH_LIMIT`.
    pitch: f64,
    keys: KeyState,
    /// True while the cursor is grabbed and mouse-look is active.
    grabbed: bool,
    /// `last_tick` is `None` until the first redraw, then advanced
    /// every frame so the camera integrator sees a real dt.
    last_tick: Option<Instant>,
}

impl App {
    fn camera(&self) -> Camera {
        // Voxlap's basis: +z is "down" into the map. Yaw rotates
        // around +z; pitch is rotation about the camera-relative
        // right axis. The forward / right / down vectors below are
        // the standard yaw-then-pitch composition.
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        let forward = [sy * cp, cy * cp, sp];
        let right = [cy, -sy, 0.0];
        // down = right × forward (right-handed).
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

    fn integrate(&mut self, dt: f64) {
        let cam = self.camera();
        let fast = if self.keys.has(KeyState::FAST) {
            FAST_MULT
        } else {
            1.0
        };
        let speed = MOVE_SPEED * fast * dt;
        let mut dx = 0.0;
        let mut dy = 0.0;
        let mut dz = 0.0;
        if self.keys.has(KeyState::FORWARD) {
            dx += cam.forward[0];
            dy += cam.forward[1];
            dz += cam.forward[2];
        }
        if self.keys.has(KeyState::BACK) {
            dx -= cam.forward[0];
            dy -= cam.forward[1];
            dz -= cam.forward[2];
        }
        if self.keys.has(KeyState::RIGHT) {
            dx += cam.right[0];
            dy += cam.right[1];
            dz += cam.right[2];
        }
        if self.keys.has(KeyState::LEFT) {
            dx -= cam.right[0];
            dy -= cam.right[1];
            dz -= cam.right[2];
        }
        if self.keys.has(KeyState::DOWN) {
            // World-down (+z), independent of camera pitch.
            dz += 1.0;
        }
        if self.keys.has(KeyState::UP) {
            dz -= 1.0;
        }
        self.cam_pos[0] += dx * speed;
        self.cam_pos[1] += dy * speed;
        self.cam_pos[2] += dz * speed;
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
        // stall (e.g. window drag) doesn't teleport the camera on
        // the next frame.
        let now = Instant::now();
        let dt = self
            .last_tick
            .map_or(0.0, |t| (now - t).as_secs_f64().min(0.1));
        self.last_tick = Some(now);
        self.integrate(dt);

        // Make sure the zbuffer + scratch fit this frame's
        // resolution. Cheap when unchanged.
        let pixel_count = (size.width as usize) * (size.height as usize);
        if self.zbuffer.len() < pixel_count {
            self.zbuffer.resize(pixel_count, 0.0);
        }
        if self.scratch.uurend_half_stride < size.width as usize {
            self.scratch = ScanScratch::new_for_size(size.width, size.height, self.vxl.vsid);
        }

        // Wire engine sky colour onto scratch so grouscan's startsky
        // has the right (col, dist) for any radar slot it drains.
        // The `dist` seeded here is overwritten per-ray by gline
        // based on the frustum-edge clip outcome.
        // u32 → i32 reinterpret (preserves bits; `cast_signed` is
        // 1.87+, beyond the workspace MSRV).
        let sky_col_i = i32::from_ne_bytes(self.engine.sky_color().to_ne_bytes());
        self.scratch.set_skycast(sky_col_i, 0);

        // Engine fog → ScanScratch foglut. Rebuilds the 2048-entry
        // table only when fog params change.
        let fog_col_i = i32::from_ne_bytes(self.engine.fog_color().to_ne_bytes());
        self.scratch
            .set_fog(fog_col_i, self.engine.fog_max_scan_dist());

        let cam = self.camera();
        let sky = self.engine.sky_color();
        let settings = OpticastSettings::for_oracle_framebuffer(size.width, size.height);
        let pitch_pixels = size.width as usize;

        let Some(surface) = self.surface.as_mut() else {
            return;
        };
        surface.resize(w_nz, h_nz).expect("softbuffer: resize");
        let mut buffer = surface.buffer_mut().expect("softbuffer: buffer_mut");
        // Pre-fill with sky so any pixel opticast leaves untouched
        // reads as sky.
        for px in buffer.iter_mut() {
            *px = sky;
        }

        // Scope the rasterizer so its &mut buffer borrow ends before
        // we present the buffer.
        {
            let mut rasterizer = ScalarRasterizer::new(
                &mut buffer,
                &mut self.zbuffer,
                pitch_pixels,
                &self.vxl.data,
                &self.vxl.column_offset,
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
            // Locked. Try Locked first, fall back to Confined — same
            // pattern winit's own docs recommend.
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
            .with_title("roxlap — oracle.vxl")
            .with_inner_size(LogicalSize::new(f64::from(WIDTH), f64::from(HEIGHT)));
        let window = Rc::new(
            event_loop
                .create_window(attrs)
                .expect("winit: create_window"),
        );
        let context = softbuffer::Context::new(window.clone()).expect("softbuffer: Context::new");
        let surface =
            softbuffer::Surface::new(&context, window.clone()).expect("softbuffer: Surface::new");
        self.window = Some(window);
        self.surface = Some(surface);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key,
                        state,
                        repeat,
                        ..
                    },
                ..
            } => {
                if repeat {
                    return;
                }
                let pressed = state == ElementState::Pressed;
                let PhysicalKey::Code(code) = physical_key else {
                    return;
                };
                match code {
                    KeyCode::Escape => {
                        if pressed {
                            if self.grabbed {
                                self.set_grabbed(false);
                            } else {
                                event_loop.exit();
                            }
                        }
                    }
                    KeyCode::KeyW => self.keys.set(KeyState::FORWARD, pressed),
                    KeyCode::KeyS => self.keys.set(KeyState::BACK, pressed),
                    KeyCode::KeyA => self.keys.set(KeyState::LEFT, pressed),
                    KeyCode::KeyD => self.keys.set(KeyState::RIGHT, pressed),
                    KeyCode::Space => self.keys.set(KeyState::UP, pressed),
                    KeyCode::ShiftLeft | KeyCode::ShiftRight => {
                        self.keys.set(KeyState::DOWN, pressed);
                    }
                    KeyCode::ControlLeft | KeyCode::ControlRight => {
                        self.keys.set(KeyState::FAST, pressed);
                    }
                    _ => {}
                }
            }

            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                if !self.grabbed {
                    self.set_grabbed(true);
                }
            }

            WindowEvent::Focused(false) => {
                // Drop any held keys when the window loses focus, so
                // we don't keep moving while the user is in another app.
                self.keys = KeyState::default();
                if self.grabbed {
                    self.set_grabbed(false);
                }
            }

            WindowEvent::RedrawRequested => {
                self.redraw();
            }

            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            if self.grabbed {
                self.yaw += dx * MOUSE_SENS;
                self.pitch = (self.pitch + dy * MOUSE_SENS).clamp(-PITCH_LIMIT, PITCH_LIMIT);
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // ControlFlow::Poll drives a continuous redraw loop so the
        // camera integrator runs every wakeup.
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }
}

/// Decompress + parse the embedded oracle world. Errors are fatal:
/// a malformed asset means the binary is broken, not a runtime
/// recoverable state.
fn load_oracle_vxl() -> vxl::Vxl {
    let mut decoder = GzDecoder::new(ORACLE_VXL_GZ);
    let mut bytes = Vec::with_capacity(40 * 1024 * 1024);
    decoder
        .read_to_end(&mut bytes)
        .expect("gunzip oracle.vxl.gz");
    vxl::parse(&bytes).expect("parse oracle.vxl")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let vxl_world = load_oracle_vxl();
    let initial_scratch = ScanScratch::new_for_size(WIDTH, HEIGHT, vxl_world.vsid);
    let cam_pos = vxl_world.ipo;

    let mut app = App {
        window: None,
        surface: None,
        engine: Engine::new(),
        zbuffer: Vec::new(),
        scratch: initial_scratch,
        vxl: vxl_world,
        cam_pos,
        yaw: 0.0,
        pitch: 0.0,
        keys: KeyState::default(),
        grabbed: false,
        last_tick: None,
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}
