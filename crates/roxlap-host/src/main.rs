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
//! - `F` → capture current camera state + frame to
//!   `roxlap-capture.{txt,ppm}` for off-line repro (e.g. via
//!   `roxlap-oracle find-hairlines`).
//! - `Esc` → release cursor (or exit if already released).
//! - Window close → exit.

use std::io::Read;
use std::num::NonZeroU32;
use std::rc::Rc;
use std::time::Instant;

use flate2::read::GzDecoder;
use roxlap_core::camera_math;
use roxlap_core::opticast;
use roxlap_core::rasterizer::ScanScratch;
use roxlap_core::scalar_rasterizer::ScalarRasterizer;
use roxlap_core::sprite::{draw_sprite, DrawTarget, Sprite, SpriteLighting};
use roxlap_core::Camera;
use roxlap_core::Engine;
use roxlap_core::LightSrc;
use roxlap_core::OpticastSettings;
use roxlap_formats::{kv6, vxl};
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

/// Embedded coco kv6 sprite (Voxlap's iconic logo). Demoes the
/// rotated `drawboundcubesse` path: the host spins the basis about
/// the world z-axis once per ~12 seconds.
const COCO_KV6: &[u8] = include_bytes!("../../../assets/coco.kv6");

/// Embedded meltsphere kv6 sprite (401 voxels carved from the
/// oracle world via R6.0d's `meltsphere`; same fixture the
/// `sprite_*` oracle poses use). Demoes the axis-aligned
/// `drawboundcubesse` path.
const SPRITE_MELTSPHERE_KV6: &[u8] =
    include_bytes!("../../roxlap-core/tests/fixtures/sprite_meltsphere.kv6");

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
    /// Yaw — rotation around the world +z (down) axis. 0 looks +x
    /// (voxlap's canonical heading).
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
    /// Set by the `F` hotkey; the next redraw dumps the current
    /// camera state + framebuffer to `roxlap-capture.{txt,ppm}`
    /// then clears the flag. Lets the user freeze a repro for
    /// rendering bugs that surface at runtime.
    capture_pending: bool,
    /// Demo sprites plumbed through the R6.4 `draw_sprite` path.
    /// Slot 0 is the meltsphere fixture (axis-aligned); slot 1 is
    /// the coco logo, whose basis the redraw loop spins about z so
    /// the rotated `drawboundcubesse` path is exercised.
    sprites: Vec<Sprite>,
    /// Wall-clock baseline for sprite-rotation animation. Set once
    /// at app construction; the redraw loop reads `elapsed()` every
    /// frame to derive the coco's spin angle.
    spawn_time: Instant,
}

impl App {
    fn camera(&self) -> Camera {
        // Voxlap's standard yaw/pitch composition (mirror of
        // tests/oracle/oracle.c::set_camera_yaw_pitch). +z is
        // "down" into the map; yaw=0 looks +x; positive pitch
        // tilts the view downward. The basis is RIGHT-HANDED
        // (right × down = forward), which the engine's frustum-
        // normal cross product (`ginor[i] = gcorn[i] × gcorn[i+1]`)
        // assumes — get the chirality wrong and `kv6_draw_prepare`'s
        // bound-cube cull rejects every sprite as "outside the
        // frustum".
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        let right = [-sy, cy, 0.0];
        let down = [-cy * sp, -sy * sp, cp];
        let forward = [cy * cp, sy * cp, sp];
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

    #[allow(clippy::too_many_lines)] // straight-line per-frame work; splitting hurts readability
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

        // Engine side-shades → ScanScratch gcsub. Default is
        // `[0; 6]` (no shading); the host bumps it to a moderate
        // value at startup so faces facing each direction read
        // visibly different.
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

        // R6.4 sprite render: cull + setup + per-voxel rasterizer
        // writes pixels + zbuffer. Scoped after opticast so each
        // sprite layers on top of the world (and on top of any
        // earlier sprites in the `sprites` list).
        let cam_state = camera_math::derive(
            &cam,
            size.width,
            size.height,
            settings.hx,
            settings.hy,
            settings.hz,
        );

        // Spin slot 1 (coco) about world z. ~12-second period (≈30°/s)
        // — slow enough to read individual voxel faces, fast enough
        // to confirm the renderer is alive.
        if self.sprites.len() > 1 {
            let theta = self.spawn_time.elapsed().as_secs_f32() * 0.5;
            let (s, c) = theta.sin_cos();
            self.sprites[1].s = [c, s, 0.0];
            self.sprites[1].h = [-s, c, 0.0];
            self.sprites[1].f = [0.0, 0.0, 1.0];
        }

        {
            // Debug: ROXLAP_HOST_SPRITE_NO_Z=1 wipes the zbuffer
            // back to +∞ before sprite render. Sprites then draw
            // unconditionally on top of opticast output. Lets us
            // distinguish "sprite z-test is rejecting" from
            // "sprite geometry is broken".
            if std::env::var("ROXLAP_HOST_SPRITE_NO_Z").is_ok() {
                for z in &mut self.zbuffer {
                    *z = f32::INFINITY;
                }
            }

            let mut target = DrawTarget {
                framebuffer: &mut buffer,
                zbuffer: &mut self.zbuffer,
                pitch_pixels,
                width: size.width,
                height: size.height,
            };
            // Snapshot the engine's lighting state once per frame.
            // Cheap to build (it's just three field reads + a slice
            // borrow) and lets sprite shading respond to runtime
            // setter calls (e.g. moving the demo torch).
            let lighting = SpriteLighting::from_engine(&self.engine);

            // Debug: count pixels written per sprite. Gated on an
            // env var so the noise stays out of normal interactive
            // runs. Set ROXLAP_HOST_SPRITE_DEBUG=1 to see counts.
            let debug = std::env::var("ROXLAP_HOST_SPRITE_DEBUG").is_ok();
            for (i, sprite) in self.sprites.iter().enumerate() {
                let written = draw_sprite(&mut target, &cam_state, &settings, &lighting, sprite);
                if debug {
                    eprintln!(
                        "sprite[{i}]: pos=({:.1}, {:.1}, {:.1}) basis_s=({:.2},{:.2},{:.2}) → wrote {} pixels",
                        sprite.p[0], sprite.p[1], sprite.p[2],
                        sprite.s[0], sprite.s[1], sprite.s[2],
                        written
                    );
                }
            }
        }

        if self.capture_pending {
            self.capture_pending = false;
            if let Err(e) =
                write_capture(&buffer, size.width, size.height, &cam, self.yaw, self.pitch)
            {
                eprintln!("capture: failed to write: {e}");
            } else {
                eprintln!(
                    "capture: roxlap-capture.txt + .ppm (pos=[{:.4}, {:.4}, {:.4}], yaw={:.6}, pitch={:.6}, size={}x{})",
                    self.cam_pos[0], self.cam_pos[1], self.cam_pos[2], self.yaw, self.pitch, size.width, size.height,
                );
            }
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
                    KeyCode::Escape if pressed => {
                        if self.grabbed {
                            self.set_grabbed(false);
                        } else {
                            event_loop.exit();
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
                    KeyCode::KeyF if pressed => {
                        self.capture_pending = true;
                    }
                    _ => {}
                }
            }

            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } if !self.grabbed => {
                self.set_grabbed(true);
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

/// Dump the current frame + camera pose to disk. Two files written
/// to the working directory:
///
/// - `roxlap-capture.txt` — the camera state in a format
///   `roxlap-oracle find-hairlines` can read (one `key = value`
///   per line, free-form).
/// - `roxlap-capture.ppm` — P6 binary RGB framebuffer (XRGB8888 in
///   memory; we drop the high byte / brightness channel).
///
/// Used by the `F` hotkey to freeze runtime repro state for off-line
/// debugging (e.g. flickering-hairline artifacts).
fn write_capture(
    buffer: &[u32],
    width: u32,
    height: u32,
    cam: &Camera,
    yaw: f64,
    pitch: f64,
) -> std::io::Result<()> {
    use std::io::Write;
    let txt = format!(
        "# roxlap-host capture — generated by F hotkey\n\
         width = {width}\n\
         height = {height}\n\
         pos.x = {}\n\
         pos.y = {}\n\
         pos.z = {}\n\
         yaw = {}\n\
         pitch = {}\n\
         # Camera basis at the time of capture (host yaw/pitch convention):\n\
         right = [{}, {}, {}]\n\
         down = [{}, {}, {}]\n\
         forward = [{}, {}, {}]\n",
        cam.pos[0],
        cam.pos[1],
        cam.pos[2],
        yaw,
        pitch,
        cam.right[0],
        cam.right[1],
        cam.right[2],
        cam.down[0],
        cam.down[1],
        cam.down[2],
        cam.forward[0],
        cam.forward[1],
        cam.forward[2],
    );
    std::fs::write("roxlap-capture.txt", txt)?;

    let header = format!("P6\n{width} {height}\n255\n");
    let mut bytes = Vec::with_capacity(header.len() + (width as usize) * (height as usize) * 3);
    bytes.extend_from_slice(header.as_bytes());
    for &px in buffer {
        bytes.push(((px >> 16) & 0xff) as u8);
        bytes.push(((px >> 8) & 0xff) as u8);
        bytes.push((px & 0xff) as u8);
    }
    let mut f = std::fs::File::create("roxlap-capture.ppm")?;
    f.write_all(&bytes)?;
    Ok(())
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

    // Voxlap's classic per-side darkening (top, bot, left, right,
    // up, down) — moderate values that read as visible directional
    // shading without going so dark the floor crushes to black.
    // The oracle uses (0,…,0) so this only affects the interactive
    // host; goldens stay bit-exact.
    let mut engine = Engine::new();
    engine.set_side_shades(15, 15, 15, 15, 15, 15);

    // Demo sprites positioned in front of the spawn camera (which
    // looks +x at yaw=0 with voxlap's RH basis). Slot 0 = meltsphere
    // (axis-aligned), 12 voxels left of forward. Slot 1 = coco
    // (rotated about world z by the redraw loop), 12 voxels right.
    // Together they exercise both the axis-aligned and rotated
    // `drawboundcubesse` paths.
    let meltsphere_kv6 =
        kv6::parse(SPRITE_MELTSPHERE_KV6).expect("parse sprite_meltsphere.kv6 fixture");
    let coco_kv6 = kv6::parse(COCO_KV6).expect("parse coco.kv6");
    // World coords are in [0, VSID = 2048]; safely fit f32 exactly.
    #[allow(clippy::cast_possible_truncation)]
    let cam_f32 = [cam_pos[0] as f32, cam_pos[1] as f32, cam_pos[2] as f32];
    // Forward (yaw=0) = +x; right = +y. So +24 forward = +x, ±12
    // right/left = ±y.
    let meltsphere_sprite = Sprite::axis_aligned(
        meltsphere_kv6,
        [cam_f32[0] + 24.0, cam_f32[1] - 12.0, cam_f32[2]],
    );
    let coco_sprite =
        Sprite::axis_aligned(coco_kv6, [cam_f32[0] + 24.0, cam_f32[1] + 12.0, cam_f32[2]]);

    // Demo lighting: lightmode=2 + one bright point light parked
    // exactly between the two sprites (each 12 voxels lateral
    // from this point), at sprite-z. Both sprites get strong
    // directional shading — the face turned toward the light at
    // 0x80 (full ambient) brightness, the back face deeply
    // shadowed. The voxlap lighting falloff `(1/d³ - 1/r³) * sc *
    // 16` is a steep inverse-cube curve, so generous `sc` is
    // needed to push shadow contrast up; `r2 = 60²` cuts the light
    // off cleanly past 60 voxels.
    engine.set_lightmode(2);
    engine.add_light(LightSrc {
        pos: [cam_f32[0] + 24.0, cam_f32[1], cam_f32[2]],
        r2: 60.0 * 60.0,
        sc: 8192.0,
    });

    let mut app = App {
        window: None,
        surface: None,
        engine,
        zbuffer: Vec::new(),
        scratch: initial_scratch,
        vxl: vxl_world,
        cam_pos,
        yaw: 0.0,
        pitch: 0.0,
        keys: KeyState::default(),
        grabbed: false,
        last_tick: None,
        capture_pending: false,
        sprites: vec![meltsphere_sprite, coco_sprite],
        spawn_time: Instant::now(),
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}
