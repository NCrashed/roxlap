//! roxlap-scene-demo — interactive showcase of the scene-graph
//! engine. See `README.md` for the controls + the demo's
//! evolution roadmap as the scene-graph substages land.

mod scene;
mod ship;
mod terrain;

use std::num::NonZeroU32;
use std::rc::Rc;
use std::time::Instant;

use roxlap_core::opticast::OpticastSettings;
use roxlap_core::rasterizer::ScratchPool;
use roxlap_core::Engine;
use roxlap_scene::render::render_scene_composed;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{DeviceEvent, DeviceId, ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

use crate::scene::{build_demo, SceneAndCamera};

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;
const MOVE_SPEED: f64 = 64.0;
const FAST_MULT: f64 = 4.0;
const MOUSE_SENS: f64 = 0.0025;
/// Pitch clamped just shy of ±90° so the basis stays well-conditioned.
const PITCH_LIMIT: f64 = 88.0_f64 * std::f64::consts::PI / 180.0;

fn main() {
    let event_loop = EventLoop::new().expect("winit: EventLoop::new");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("winit: run_app");
}

#[allow(clippy::struct_excessive_bools)]
struct InputState {
    forward: bool,
    back: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
    fast: bool,
}

impl InputState {
    const fn new() -> Self {
        Self {
            forward: false,
            back: false,
            left: false,
            right: false,
            up: false,
            down: false,
            fast: false,
        }
    }
}

#[allow(clippy::struct_excessive_bools)]
struct App {
    window: Option<Rc<Window>>,
    surface: Option<softbuffer::Surface<Rc<Window>, Rc<Window>>>,
    engine: Engine,
    scene: SceneAndCamera,
    zbuffer: Vec<f32>,
    pool: ScratchPool,
    input: InputState,
    grabbed: bool,
    last_frame: Instant,
}

impl App {
    fn new() -> Self {
        let scene = build_demo();
        let engine = Engine::new();
        let pool = ScratchPool::new(WIDTH, HEIGHT, roxlap_scene::CHUNK_SIZE_XY);
        Self {
            window: None,
            surface: None,
            engine,
            scene,
            zbuffer: vec![f32::INFINITY; (WIDTH * HEIGHT) as usize],
            pool,
            input: InputState::new(),
            grabbed: false,
            last_frame: Instant::now(),
        }
    }

    fn redraw(&mut self) {
        // Take the size up front so the `&self.window` borrow doesn't
        // collide with later `&mut self` operations.
        let (size, w_nz, h_nz) = {
            let Some(window) = self.window.as_ref() else {
                return;
            };
            let size = window.inner_size();
            let Some(w_nz) = NonZeroU32::new(size.width) else {
                return;
            };
            let Some(h_nz) = NonZeroU32::new(size.height) else {
                return;
            };
            (size, w_nz, h_nz)
        };

        // Step the camera from the input state.
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f64();
        self.last_frame = now;
        self.tick_camera(dt);

        // Resize ScratchPool / zbuffer when the window grew.
        let pixel_count = (size.width as usize) * (size.height as usize);
        if self.zbuffer.len() < pixel_count {
            self.zbuffer.resize(pixel_count, f32::INFINITY);
        }
        if self.pool.slot(0).uurend_half_stride < size.width as usize {
            self.pool = ScratchPool::new(size.width, size.height, roxlap_scene::CHUNK_SIZE_XY);
        }

        // Pool config — sky + fog colour. `treat_z_max_as_air` lets
        // the ship grid render correctly even though the camera is
        // above the bedrock placeholder of its (sparse) chunk
        // lattice; without it OOB-z cameras hit the S1.X bedrock
        // path.
        let sky = self.engine.sky_color();
        let sky_col_i = i32::from_ne_bytes(sky.to_ne_bytes());
        self.pool.set_skycast(sky_col_i, 0);
        let fog_col_i = i32::from_ne_bytes(self.engine.fog_color().to_ne_bytes());
        self.pool
            .set_fog(fog_col_i, self.engine.fog_max_scan_dist());
        self.pool.set_treat_z_max_as_air(true);

        let settings = OpticastSettings::for_oracle_framebuffer(size.width, size.height);
        let pitch_pixels = size.width as usize;

        let Some(surface) = self.surface.as_mut() else {
            return;
        };
        surface.resize(w_nz, h_nz).expect("softbuffer: resize");
        let mut buffer = surface.buffer_mut().expect("softbuffer: buffer_mut");

        // `render_scene_composed`'s convention: caller pre-fills fb
        // with sky and zb with INFINITY; the helper allocates per-grid
        // temp buffers and z-merges into the shared output.
        for px in buffer.iter_mut() {
            *px = sky;
        }
        for z in &mut self.zbuffer[..pixel_count] {
            *z = f32::INFINITY;
        }

        let _outcome = render_scene_composed(
            &mut buffer,
            &mut self.zbuffer[..pixel_count],
            pitch_pixels,
            size.width,
            size.height,
            &mut self.pool,
            &self.scene.scene,
            &self.scene.camera,
            &settings,
            sky,
            None,
        );

        buffer.present().expect("softbuffer: present");
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }

    /// Update camera position from the active input bits.
    fn tick_camera(&mut self, dt: f64) {
        let fast = if self.input.fast { FAST_MULT } else { 1.0 };
        let speed = MOVE_SPEED * fast * dt;

        let cam = &self.scene.camera;
        // Strafe / advance along camera-frame axes.
        let mut dx = 0.0;
        let mut dy = 0.0;
        let mut dz = 0.0;
        if self.input.forward {
            dx += cam.forward[0];
            dy += cam.forward[1];
            dz += cam.forward[2];
        }
        if self.input.back {
            dx -= cam.forward[0];
            dy -= cam.forward[1];
            dz -= cam.forward[2];
        }
        if self.input.right {
            // The "right" basis vector is screen-right — to strafe
            // right we move along it. Voxlap stores `right` as the
            // physical right vector (RH basis), so `+right` is correct.
            dx += cam.right[0];
            dy += cam.right[1];
            dz += cam.right[2];
        }
        if self.input.left {
            dx -= cam.right[0];
            dy -= cam.right[1];
            dz -= cam.right[2];
        }
        // World-frame vertical (Space / LShift): in voxlap z is
        // *down*, so `Space` should subtract from z (move up).
        if self.input.up {
            dz -= 1.0;
        }
        if self.input.down {
            dz += 1.0;
        }

        let mag2 = dx * dx + dy * dy + dz * dz;
        if mag2 > 0.0 {
            let inv = 1.0 / mag2.sqrt();
            self.scene.cam_pos[0] += dx * inv * speed;
            self.scene.cam_pos[1] += dy * inv * speed;
            self.scene.cam_pos[2] += dz * inv * speed;
            self.scene.refresh_camera();
        }
    }

    fn set_grab(&mut self, grab: bool) {
        let Some(window) = self.window.as_ref() else {
            return;
        };
        if grab {
            // Confined mode is enough for the scene demo; locked
            // mode requires extra winit support that's platform-
            // dependent.
            let _ = window
                .set_cursor_grab(CursorGrabMode::Confined)
                .or_else(|_| window.set_cursor_grab(CursorGrabMode::Locked));
            window.set_cursor_visible(false);
            self.grabbed = true;
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
            .with_title("roxlap-scene-demo")
            .with_inner_size(LogicalSize::new(WIDTH, HEIGHT));
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
            WindowEvent::RedrawRequested => self.redraw(),
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => self.set_grab(true),
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
                    KeyCode::KeyW => self.input.forward = pressed,
                    KeyCode::KeyS => self.input.back = pressed,
                    KeyCode::KeyA => self.input.left = pressed,
                    KeyCode::KeyD => self.input.right = pressed,
                    KeyCode::Space => self.input.up = pressed,
                    KeyCode::ShiftLeft => self.input.down = pressed,
                    KeyCode::ControlLeft => self.input.fast = pressed,
                    KeyCode::Escape if pressed => {
                        if self.grabbed {
                            self.set_grab(false);
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

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if !self.grabbed {
            return;
        }
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            self.scene.yaw += dx * MOUSE_SENS;
            self.scene.pitch =
                (self.scene.pitch + dy * MOUSE_SENS).clamp(-PITCH_LIMIT, PITCH_LIMIT);
            self.scene.refresh_camera();
        }
    }
}
