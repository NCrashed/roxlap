//! Hello-world entry point for new developers.
//!
//! Opens a window, builds a tiny voxel world (flat ground +
//! 10×10×10 cube), bakes directional sun-style lighting, and runs
//! the engine's opticast rasterizer every frame against a WASD +
//! mouse-look camera.
//!
//! What this file shows — read top-to-bottom to learn how the
//! engine is assembled into an app:
//!
//!   1. **World construction.** `pack_dense_grid_to_vxl` folds a
//!      dense `(y, x, z)` mask + colour grid into voxlap's slab
//!      format. We seed a one-voxel-thick ground plane this way.
//!   2. **Runtime editing.** `Vxl::reserve_edit_capacity` enables
//!      the edit API; `roxlap_formats::edit::set_rect` then carves
//!      a 10×10×10 cube on top of the ground in a single call.
//!   3. **Engine setup.** `Engine::new` for sky / fog / lights /
//!      sprite material; `engine.set_lightmode(1)` selects voxlap's
//!      directional bake (`(n.y · 0.5 + n.z) · 64 + 103.5` per
//!      surface voxel — no `LightSrc` needed).
//!   4. **Lighting bake.** `update_lighting` walks the voxel grid
//!      once at startup and writes per-voxel brightness bytes that
//!      the rasterizer reads later. Bake region is the full world.
//!   5. **Per-frame render.** `ScratchPool::new` allocates the
//!      reusable scan scratch (radar / angstart / lastx / uurend);
//!      each frame we configure it (sky cast, side shades), build
//!      a `ScalarRasterizer` over the framebuffer + zbuffer, and
//!      call `opticast`. softbuffer's surface owns the actual
//!      window pixel buffer; we render directly into it.
//!   6. **Camera + input.** We hold yaw / pitch + a packed key
//!      bitfield; every frame we integrate movement and build a
//!      right-handed `Camera` (right × down = forward) the engine
//!      consumes.
//!
//! Run from the workspace root:
//!
//! ```sh
//! cargo run --release --example hello -p roxlap-core
//! ```
//!
//! Controls:
//! - **LMB**: grab the cursor (mouse-look active).
//! - **WASD**: forward / strafe-left / back / strafe-right.
//! - **Space / `LShift`**: up / down (world-z, not view-relative).
//! - **`LCtrl`**: hold for 4× fly speed.
//! - **Esc**: release the cursor (or exit if already released).
//!
//! The single-character locals (`r`, `g`, `b`, `s`, `c`, `w`, `h`)
//! are conventional in rendering code; relax the pedantic name
//! lint just for this file.
#![allow(clippy::many_single_char_names)]
// Voxel coords routinely round-trip through f64 ↔ i32 and u32 ↔ usize.
// Domain stays inside f64's exact-integer range, so the precision /
// truncation lints here are noise rather than signal.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]

use std::num::NonZeroU32;
use std::rc::Rc;
use std::time::Instant;

use roxlap_cavegen::{pack_dense_grid_to_vxl, MAXZDIM};
use roxlap_core::opticast;
use roxlap_core::rasterizer::ScratchPool;
use roxlap_core::scalar_rasterizer::ScalarRasterizer;
use roxlap_core::{update_lighting, Camera, Engine, GridView, OpticastSettings};
use roxlap_formats::edit::set_rect;
use roxlap_formats::vxl::Vxl;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{DeviceEvent, DeviceId, ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

/// World footprint in voxels along X and Y. Voxlap requires a power
/// of two; 32 keeps the bake fast and leaves room around the cube.
const VSID: u32 = 32;

/// Z-coord of the (one-voxel-thick) ground plane. Voxlap is **z-down**:
/// small z is up, large z is down. `200` puts the floor near the
/// bottom of the voxlap z-range with ~200 voxels of empty air above
/// for the camera and the cube.
const GROUND_Z: i32 = 200;

/// Edge length of the demo cube, in voxels.
const CUBE_EDGE: i32 = 10;

/// Voxlap colour packing: `(brightness << 24) | (R << 16) | (G << 8) | B`.
/// `0x80` brightness is voxlap's neutral; the `update_lighting` bake
/// overwrites it with directional shading.
const GROUND_COL: u32 = 0x80_5a_a0_5a; // mossy green
const CUBE_COL: u32 = 0x80_c0_60_30; // warm orange

/// Walking speed, in voxels per second.
const MOVE_SPEED: f64 = 16.0;
/// Multiplier applied while `LCtrl` is held.
const FAST_MULT: f64 = 4.0;
/// Mouse sensitivity, in radians per pixel of cursor delta.
const MOUSE_SENS: f64 = 0.0025;
/// Pitch clamp — just shy of ±90° so the camera basis stays
/// well-conditioned (a straight-up view collapses `right × forward`).
const PITCH_LIMIT: f64 = 88.0_f64 * std::f64::consts::PI / 180.0;

/// Build the demo world: a flat ground plane plus a `CUBE_EDGE`³
/// cube of solid voxels sitting on top of it.
///
/// The ground is created by [`pack_dense_grid_to_vxl`] (the fast
/// dense-grid → slab packer from `roxlap-cavegen`); the cube is
/// added afterwards via the runtime edit API ([`set_rect`]) to show
/// both world-construction paths in one place.
fn build_world() -> Vxl {
    let vsid_u = VSID as usize;
    let maxz_u = MAXZDIM as usize;
    let cells = vsid_u * vsid_u * maxz_u;

    // Dense grid: 1-byte solid/air mask + matching u32 colour, stored
    // in `(y, x, z)` order (the layout `pack_dense_grid_to_vxl`
    // expects). Start every voxel as air, then stamp the ground row.
    let mut mask = vec![0u8; cells];
    let mut colour = vec![0u32; cells];
    let idx = |x: usize, y: usize, z: usize| -> usize { (y * vsid_u + x) * maxz_u + z };
    for y in 0..vsid_u {
        for x in 0..vsid_u {
            let i = idx(x, y, GROUND_Z as usize);
            mask[i] = 1;
            colour[i] = GROUND_COL;
        }
    }
    let mut world = pack_dense_grid_to_vxl(&mask, &colour, VSID);

    // Reserve a slab pool large enough for the cube edit. The edit
    // API allocates new slab records into this pool; without it
    // `set_rect` panics. 64 KB is overkill for one 10³ cube but fits
    // comfortably and gives downstream tinkering some headroom.
    world.reserve_edit_capacity(64 * 1024);

    // Place the cube centred on the world's XY footprint, sitting
    // directly on the ground (top of cube `CUBE_EDGE` voxels above
    // GROUND_Z, i.e. lower z because voxlap is z-down).
    let cx = (VSID as i32) / 2;
    let cy = (VSID as i32) / 2;
    let half = CUBE_EDGE / 2;
    let lo = [cx - half, cy - half, GROUND_Z - CUBE_EDGE];
    let hi = [cx + half - 1, cy + half - 1, GROUND_Z - 1];
    set_rect(&mut world, lo, hi, Some(CUBE_COL));

    world
}

/// Camera state owned by the example. Yaw / pitch are stored as
/// `f64` so the integrator can run at any frame rate without losing
/// fractional voxels of motion.
struct Cam {
    pos: [f64; 3],
    yaw: f64,
    pitch: f64,
}

impl Cam {
    /// Compose voxlap's right-handed yaw / pitch basis:
    ///
    /// - `right × down = forward` (chirality the engine's frustum
    ///   math assumes — flip and sprites + side shades silently
    ///   render upside-down).
    /// - `yaw = 0` looks `+x`; positive `yaw` rotates toward `+y`.
    /// - `pitch = 0` is level; positive `pitch` tilts the view down.
    fn camera(&self) -> Camera {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        let forward = [cy * cp, sy * cp, sp];
        let right = [-sy, cy, 0.0];
        let down = [
            forward[1] * right[2] - forward[2] * right[1],
            forward[2] * right[0] - forward[0] * right[2],
            forward[0] * right[1] - forward[1] * right[0],
        ];
        Camera {
            pos: self.pos,
            right,
            down,
            forward,
        }
    }
}

/// Movement key bitfield. Polled each frame so the integrator runs
/// at the redraw rate, not the OS key-repeat rate.
#[derive(Default, Clone, Copy)]
struct Keys(u8);

impl Keys {
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
    /// Engine carries the immutable per-frame state the rasterizer
    /// reads: sky colour, fog, side shades, lighting mode. No
    /// `Vxl` lives inside it — the world is passed in via
    /// `GridView` each frame.
    engine: Engine,
    /// Voxel world. Mutated only by world-edit calls (none after
    /// startup in this example).
    world: Vxl,
    /// Per-pixel depth, reused across frames. Resized on the first
    /// redraw and on every window-resize.
    zbuffer: Vec<f32>,
    /// Reusable scan scratch (`radar`, `angstart`, `lastx`,
    /// `uurend`). Sized once at startup for the initial framebuffer
    /// and resized only when the window grows past it.
    pool: ScratchPool,
    cam: Cam,
    keys: Keys,
    grabbed: bool,
    last_tick: Option<Instant>,
}

impl App {
    fn new() -> Self {
        let mut world = build_world();

        // Engine setup. The defaults are conservative (sky-blue, no
        // fog, no side shading). We override:
        //
        // - `set_side_shades`: voxlap's per-face darkening so the
        //   cube's faces read as visibly distinct (otherwise every
        //   face shares the same flat colour and the cube looks
        //   like a single-coloured blob).
        // - `set_lightmode(1)`: directional sun-style bake. The
        //   bake pass below stamps per-voxel brightness into the
        //   slab bytes; `lightmode = 1` ignores lights and uses
        //   only each voxel's surface normal.
        let mut engine = Engine::new();
        engine.set_side_shades(15, 15, 15, 15, 15, 15);
        engine.set_lightmode(1);

        // Run the directional bake over the whole world once at
        // startup. With ~32×32×256 voxels it finishes in
        // milliseconds; larger worlds would want a smaller bbox or
        // a background thread. The bake walks each voxel column,
        // computes a surface normal from neighbouring voxels, and
        // writes brightness into the slab byte stream — the
        // rasterizer then reads those bytes verbatim per voxel.
        update_lighting(
            &mut world.data,
            &world.column_offset,
            world.vsid,
            0,
            0,
            0,
            world.vsid as i32,
            world.vsid as i32,
            MAXZDIM,
            engine.lightmode(),
            engine.lights(),
        );

        // Camera spawn: ~16 voxels in front of the cube (along -x),
        // raised slightly above the cube's top so we look down at
        // it. Yaw 0 looks +x toward the cube.
        let cx = f64::from(VSID) * 0.5;
        let cy = f64::from(VSID) * 0.5;
        let cz = f64::from(GROUND_Z) - f64::from(CUBE_EDGE) - 6.0;
        let cam = Cam {
            pos: [cx - 16.0, cy, cz],
            yaw: 0.0,
            pitch: 0.15, // tilt down a touch to put the cube in frame
        };

        let pool = ScratchPool::new(WIDTH, HEIGHT, world.vsid);

        Self {
            window: None,
            surface: None,
            engine,
            world,
            zbuffer: Vec::new(),
            pool,
            cam,
            keys: Keys::default(),
            grabbed: false,
            last_tick: None,
        }
    }

    /// Advance camera position by `dt` seconds based on which keys
    /// are currently held. Diagonal motion is normalised so two-key
    /// combos don't move √2× faster.
    fn integrate(&mut self, dt: f64) {
        if dt <= 0.0 {
            return;
        }
        let speed = MOVE_SPEED
            * if self.keys.has(Keys::FAST) {
                FAST_MULT
            } else {
                1.0
            };
        let cam = self.cam.camera();
        let mut d = [0.0; 3];
        let add = |d: &mut [f64; 3], v: [f64; 3], s: f64| {
            for (di, vi) in d.iter_mut().zip(v.iter()) {
                *di += vi * s;
            }
        };
        if self.keys.has(Keys::FWD) {
            add(&mut d, cam.forward, 1.0);
        }
        if self.keys.has(Keys::BACK) {
            add(&mut d, cam.forward, -1.0);
        }
        if self.keys.has(Keys::RIGHT) {
            add(&mut d, cam.right, 1.0);
        }
        if self.keys.has(Keys::LEFT) {
            add(&mut d, cam.right, -1.0);
        }
        // Vertical is world-z (independent of pitch) so Space always
        // means "up" no matter where you're looking.
        if self.keys.has(Keys::UP) {
            d[2] -= 1.0;
        }
        if self.keys.has(Keys::DOWN) {
            d[2] += 1.0;
        }
        let mag = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        if mag <= 1e-6 {
            return;
        }
        let s = speed * dt / mag;
        self.cam.pos[0] += d[0] * s;
        self.cam.pos[1] += d[1] * s;
        self.cam.pos[2] += d[2] * s;
    }

    /// Render one frame. The whole engine pipeline lives here:
    ///
    ///   1. Resize zbuffer + scratch if the window grew.
    ///   2. Push engine state (sky cast, fog, side shades) onto the
    ///      pool — these are per-frame settings opticast reads.
    ///   3. Pre-fill the framebuffer with the sky colour so any
    ///      pixel the rasterizer doesn't touch reads as sky.
    ///   4. Build a `GridView` over the world and a
    ///      `ScalarRasterizer` over the framebuffer + zbuffer.
    ///   5. Call `opticast`. That's it — the rasterizer writes
    ///      directly into softbuffer's pixel buffer.
    fn redraw(&mut self) {
        let Some(window) = self.window.as_ref() else {
            return;
        };
        let size = window.inner_size();
        let (Some(w_nz), Some(h_nz)) = (NonZeroU32::new(size.width), NonZeroU32::new(size.height))
        else {
            return;
        };

        // Tick the camera by real wall-clock dt, clamped so a long
        // stall (window drag, debugger pause) doesn't teleport.
        let now = Instant::now();
        let dt = self
            .last_tick
            .map_or(0.0, |t| (now - t).as_secs_f64().min(0.1));
        self.last_tick = Some(now);
        self.integrate(dt);

        // Resize the per-frame scratch if needed.
        let pixel_count = (size.width as usize) * (size.height as usize);
        if self.zbuffer.len() < pixel_count {
            self.zbuffer.resize(pixel_count, 0.0);
        }
        if self.pool.slot(0).uurend_half_stride < size.width as usize {
            self.pool = ScratchPool::new(size.width, size.height, self.world.vsid);
        }

        // Push engine state onto the scratch slot. `sky_color` is
        // stored as a `u32` for ergonomic ARGB packing but the pool
        // wants the same bits as `i32`; the cast just reinterprets.
        let sky_col_i = i32::from_ne_bytes(self.engine.sky_color().to_ne_bytes());
        self.pool.set_skycast(sky_col_i, 0);
        let s = self.engine.side_shades();
        self.pool
            .set_side_shades(s[0], s[1], s[2], s[3], s[4], s[5]);

        let cam = self.cam.camera();
        let sky = self.engine.sky_color();
        let settings = OpticastSettings::for_oracle_framebuffer(size.width, size.height);
        let pitch_pixels = size.width as usize;

        let Some(surface) = self.surface.as_mut() else {
            return;
        };
        surface.resize(w_nz, h_nz).expect("softbuffer: resize");
        let mut buffer = surface.buffer_mut().expect("softbuffer: buffer_mut");
        // Fill with sky so any pixel opticast leaves untouched (e.g.
        // rays that never hit a voxel) shows as sky-blue.
        for px in buffer.iter_mut() {
            *px = sky;
        }

        // Scope the rasterizer so its `&mut buffer` borrow ends
        // before `buffer.present()`.
        {
            let grid = GridView::from_single_vxl(&self.world);
            let mut rasterizer =
                ScalarRasterizer::new(&mut buffer, &mut self.zbuffer, pitch_pixels, grid);
            let _ = opticast(&mut rasterizer, &mut self.pool, &cam, &settings, grid);
        }

        buffer.present().expect("softbuffer: present");
    }

    fn set_grabbed(&mut self, grabbed: bool) {
        let Some(window) = self.window.as_ref() else {
            return;
        };
        if grabbed {
            // Linux+X11 only supports Confined; Wayland + macOS only
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
            .with_title("roxlap — hello")
            .with_inner_size(LogicalSize::new(f64::from(WIDTH), f64::from(HEIGHT)));
        let window = Rc::new(event_loop.create_window(attrs).expect("create window"));
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
            } if !self.grabbed => {
                self.set_grabbed(true);
            }

            WindowEvent::Focused(false) => {
                // Drop held keys + release cursor when the window
                // loses focus so we don't drift while in another app.
                self.keys = Keys::default();
                if self.grabbed {
                    self.set_grabbed(false);
                }
            }

            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
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
                match code {
                    KeyCode::Escape if pressed => {
                        if self.grabbed {
                            self.set_grabbed(false);
                        } else {
                            event_loop.exit();
                        }
                    }
                    KeyCode::KeyW => self.keys.set(Keys::FWD, pressed),
                    KeyCode::KeyS => self.keys.set(Keys::BACK, pressed),
                    KeyCode::KeyA => self.keys.set(Keys::LEFT, pressed),
                    KeyCode::KeyD => self.keys.set(Keys::RIGHT, pressed),
                    KeyCode::Space => self.keys.set(Keys::UP, pressed),
                    KeyCode::ShiftLeft | KeyCode::ShiftRight => self.keys.set(Keys::DOWN, pressed),
                    KeyCode::ControlLeft | KeyCode::ControlRight => {
                        self.keys.set(Keys::FAST, pressed);
                    }
                    _ => {}
                }
            }

            _ => {}
        }
    }

    fn device_event(&mut self, _el: &ActiveEventLoop, _id: DeviceId, event: DeviceEvent) {
        if !self.grabbed {
            return;
        }
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            self.cam.yaw += dx * MOUSE_SENS;
            self.cam.pitch = (self.cam.pitch + dy * MOUSE_SENS).clamp(-PITCH_LIMIT, PITCH_LIMIT);
        }
    }

    fn about_to_wait(&mut self, _el: &ActiveEventLoop) {
        // `ControlFlow::Poll` + an explicit redraw request gives us
        // a continuous render loop (rather than only redrawing on
        // OS-driven events).
        if let Some(w) = self.window.as_ref() {
            w.request_redraw();
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
