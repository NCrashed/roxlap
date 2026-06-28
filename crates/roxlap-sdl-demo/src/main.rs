//! SDL2 host demo for the roxlap scene-graph engine.
//!
//! Its whole reason to exist: prove that [`roxlap_render::SceneRenderer`]
//! is decoupled from any one windowing library. The renderer binds to
//! whatever [`HasWindowHandle`] / [`HasDisplayHandle`] provider the host
//! passes — winit (`roxlap-scene-demo`), SDL2 (this crate), GLFW, a
//! custom surface, … The engine sees only the two `raw-window-handle`
//! traits plus an explicit pixel size.
//!
//! What this binary does:
//!   * stands up an SDL2 window (no SDL renderer — softbuffer/wgpu own
//!     presentation),
//!   * wraps its raw handles in a tiny `Send + Sync` adapter (SDL
//!     windows are `!Send`/`!Sync`; wgpu's `Surface<'static>` needs
//!     `Send + Sync`),
//!   * builds a trivial two-shape scene,
//!   * runs a WASD + mouse-look fly-camera loop, handing each frame to
//!     `SceneRenderer::render`.
//!
//! `ROXLAP_GPU=1` asks for the wgpu backend; it falls back to the CPU
//! softbuffer path automatically if GPU init fails.

use std::sync::Arc;
use std::time::Instant;

use glam::DVec3;
use raw_window_handle::{
    DisplayHandle, HandleError, HasDisplayHandle, HasWindowHandle, RawDisplayHandle,
    RawWindowHandle, WindowHandle,
};
use roxlap_core::opticast::OpticastSettings;
use roxlap_core::Camera;
use roxlap_render::{FrameParams, RenderOptions, SceneRenderer};
use roxlap_scene::{GridTransform, Scene};

use sdl2::event::{Event, WindowEvent};
use sdl2::keyboard::Scancode;

const WIDTH: u32 = 960;
const HEIGHT: u32 = 600;

/// Voxel colours are voxlap-format `0x80_RR_GG_BB` — high byte is the
/// flat shading intensity, low 24 bits are RGB (matches the
/// `roxlap-scene-demo` terrain palette).
const GRASS: u32 = 0x80_4d_8a_3a;
const STONE: u32 = 0x80_7a_7a_82;
const DOME: u32 = 0x80_40_60_c0;

/// Sky colour (`0x00_RR_GG_BB`, the framebuffer's native packing).
/// Both backends honour this: the CPU paints it flat, and the GPU
/// backend mirrors it onto its sky texture (since 0.7 — no manual
/// `set_sky_panorama` needed).
const SKY: u32 = 0x00_8f_bc_d4;

/// Mouse-look sensitivity (radians per pixel of relative motion).
const LOOK_SENS: f64 = 0.0025;
/// Fly speed in voxels per second.
const MOVE_SPEED: f64 = 90.0;

/// `Send + Sync` raw-handle adapter around an SDL2 window.
///
/// SDL's `Window` implements `HasWindowHandle`/`HasDisplayHandle`, but
/// it is `!Send`/`!Sync` (an SDL window is bound to the thread that
/// created it). wgpu's `Surface<'static>` — which `roxlap-render`'s GPU
/// backend stores — requires `Send + Sync`, so we cannot hand it the
/// SDL window directly.
///
/// Instead we snapshot the two raw handles (plain pointers / X11 IDs,
/// `Copy` and `'static`) once at startup and re-lend them through the
/// `Has*Handle` traits. The adapter owns no SDL state, so making it
/// `Send + Sync` is sound **as long as the backing SDL window outlives
/// it** — which `main` guarantees by keeping the `Window` alive for the
/// whole program and dropping the renderer first.
struct SdlWindowHandle {
    window: RawWindowHandle,
    display: RawDisplayHandle,
}

// SAFETY: the adapter holds only `Copy` raw handles, no SDL-owned
// resources. It is only ever touched from the main thread; the raw
// handles stay valid because `main` keeps the SDL `Window` alive past
// the renderer (and thus past every clone of this adapter).
unsafe impl Send for SdlWindowHandle {}
unsafe impl Sync for SdlWindowHandle {}

impl HasWindowHandle for SdlWindowHandle {
    fn window_handle(&self) -> Result<WindowHandle<'_>, HandleError> {
        // SAFETY: `self.window` was produced by SDL's own
        // `HasWindowHandle` impl and remains valid for as long as the
        // backing window lives (see the type's invariant above).
        Ok(unsafe { WindowHandle::borrow_raw(self.window) })
    }
}

impl HasDisplayHandle for SdlWindowHandle {
    fn display_handle(&self) -> Result<DisplayHandle<'_>, HandleError> {
        // SAFETY: as `window_handle`, for the display connection.
        Ok(unsafe { DisplayHandle::borrow_raw(self.display) })
    }
}

/// Build a `Camera` (voxlap basis) from a yaw/pitch fly-camera pose.
/// Mirrors `roxlap-scene-demo`'s helper: +z is down, yaw rotates in the
/// xy plane, positive pitch looks downward.
fn camera_for_yaw_pitch(pos: [f64; 3], yaw: f64, pitch: f64) -> Camera {
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    Camera {
        pos,
        right: [-sy, cy, 0.0],
        down: [-cy * sp, -sy * sp, cp],
        forward: [cy * cp, sy * cp, sp],
    }
}

/// Fly-camera pose advanced from input each frame.
struct FlyCam {
    pos: [f64; 3],
    yaw: f64,
    pitch: f64,
}

impl FlyCam {
    fn camera(&self) -> Camera {
        camera_for_yaw_pitch(self.pos, self.yaw, self.pitch)
    }

    /// Apply mouse-look deltas (relative-mouse motion, in pixels).
    fn look(&mut self, dx: i32, dy: i32) {
        self.yaw += f64::from(dx) * LOOK_SENS;
        // +z is down, so mouse-up (negative dy) pitches the view up.
        self.pitch = (self.pitch + f64::from(dy) * LOOK_SENS).clamp(-1.45, 1.45);
    }

    /// Integrate WASD + Space/Shift movement from the live keyboard
    /// state. Horizontal motion follows the camera's yaw; Space/Shift
    /// fly straight up/down (-z is up).
    fn fly(&mut self, ks: &sdl2::keyboard::KeyboardState, dt: f64) {
        let cam = self.camera();
        let fwd = DVec3::new(cam.forward[0], cam.forward[1], 0.0).normalize_or_zero();
        let right = DVec3::new(cam.right[0], cam.right[1], 0.0).normalize_or_zero();
        let mut step = DVec3::ZERO;
        if ks.is_scancode_pressed(Scancode::W) {
            step += fwd;
        }
        if ks.is_scancode_pressed(Scancode::S) {
            step -= fwd;
        }
        if ks.is_scancode_pressed(Scancode::D) {
            step += right;
        }
        if ks.is_scancode_pressed(Scancode::A) {
            step -= right;
        }
        if ks.is_scancode_pressed(Scancode::Space) {
            step -= DVec3::Z;
        }
        if ks.is_scancode_pressed(Scancode::LShift) {
            step += DVec3::Z;
        }
        let step = step.normalize_or_zero() * MOVE_SPEED * dt;
        self.pos[0] += step.x;
        self.pos[1] += step.y;
        self.pos[2] += step.z;
    }
}

/// Build the per-frame [`FrameParams`] and hand the scene to the
/// renderer. `settings` is borrowed by `FrameParams`, so it lives in
/// the caller's stack frame and is threaded in here.
fn render_frame(
    renderer: &mut SceneRenderer,
    scene: &mut Scene,
    camera: &Camera,
    settings: &OpticastSettings,
) {
    let frame = FrameParams {
        settings,
        sky_color: SKY,
        sky: None,
        fog_color: SKY,
        fog_max_scan_dist: settings.max_scan_dist,
        treat_z_max_as_air: true,
        gpu_mip_scan_dist: 64.0,
        gpu_max_outer_steps: 64,
        gpu_fov_y_rad: 60.0_f32.to_radians(),
        draw_sprites: false,
        side_shades: [0; 6],
    };
    renderer.render(scene, camera, &frame);
    // render() composites but doesn't present — finish the frame.
    renderer.present();
}

/// A minimal scene: a flat grass plain spanning ~3×3 chunks, a stone
/// monument, and a blue dome — enough to see the opticast renderer
/// working without pulling in the full scene-demo's terrain/streaming.
fn build_scene() -> Scene {
    use glam::IVec3;

    let mut scene = Scene::new();
    let ground = scene.add_grid(GridTransform::at(DVec3::ZERO));
    let g = scene.grid_mut(ground).expect("ground grid present");

    // Ground slab. Grid-local voxel coords; +z is down, so the surface
    // (z = 210) sits below the camera and the slab fills down toward the
    // bedrock placeholder at z = 255.
    g.set_rect(
        IVec3::new(-192, -192, 210),
        IVec3::new(191, 191, 254),
        Some(GRASS),
    );
    // Stone monument poking up out of the plain.
    g.set_rect(
        IVec3::new(40, 40, 150),
        IVec3::new(75, 95, 210),
        Some(STONE),
    );
    // Blue dome bump.
    g.set_sphere(IVec3::new(-30, 70, 205), 34, Some(DOME));

    scene
}

fn main() -> Result<(), String> {
    let sdl = sdl2::init()?;
    let video = sdl.video()?;

    // A plain window — no SDL renderer. softbuffer (CPU) or wgpu (GPU)
    // owns the surface via the raw handle.
    let window = video
        .window("roxlap-sdl-demo", WIDTH, HEIGHT)
        .position_centered()
        .resizable()
        .build()
        .map_err(|e| e.to_string())?;

    // Snapshot the raw handles once and wrap them Send + Sync. `window`
    // stays alive in this scope (declared before `renderer`, so it
    // drops *after* it) — keeping the handles valid for the surface.
    let handle = Arc::new(SdlWindowHandle {
        window: window.window_handle().map_err(|e| e.to_string())?.as_raw(),
        display: window.display_handle().map_err(|e| e.to_string())?.as_raw(),
    });

    let want_gpu = std::env::var_os("ROXLAP_GPU").is_some_and(|v| v != "0" && !v.is_empty());
    let opts = RenderOptions {
        want_gpu,
        clear_sky: SKY,
        ..RenderOptions::default()
    };
    let (init_w, init_h) = window.size();
    let mut renderer = SceneRenderer::new(handle, (init_w, init_h), &opts);
    eprintln!(
        "roxlap-sdl-demo: backend = {:?}{}",
        renderer.backend(),
        renderer
            .adapter_info()
            .map(|i| format!(" ({i})"))
            .unwrap_or_default(),
    );

    let mut scene = build_scene();

    // Fly-camera pose: above the plain (smaller z = higher), looking +y
    // and pitched down so the ground fills the view.
    let mut cam = FlyCam {
        pos: [0.0, -160.0, 120.0],
        yaw: std::f64::consts::FRAC_PI_2, // +y
        pitch: 0.28,                      // look down
    };

    // Relative mouse mode = FPS-style look (cursor hidden + locked).
    sdl.mouse().set_relative_mouse_mode(true);

    let mut event_pump = sdl.event_pump()?;
    let mut last = Instant::now();

    'run: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    scancode: Some(Scancode::Escape),
                    ..
                } => break 'run,
                Event::Window {
                    win_event: WindowEvent::Resized(w, h),
                    ..
                } => {
                    #[allow(clippy::cast_sign_loss)]
                    renderer.resize(w.max(1) as u32, h.max(1) as u32);
                }
                Event::MouseMotion { xrel, yrel, .. } => cam.look(xrel, yrel),
                _ => {}
            }
        }

        let now = Instant::now();
        let dt = (now - last).as_secs_f64();
        last = now;
        cam.fly(&event_pump.keyboard_state(), dt);

        let (w, h) = window.size();
        if w == 0 || h == 0 {
            continue;
        }
        let settings = OpticastSettings::for_oracle_framebuffer(w, h);
        render_frame(&mut renderer, &mut scene, &cam.camera(), &settings);
    }

    // Clean teardown: drain in-flight GPU work + release any acquired frame
    // before the device/surface drop, so quitting never yanks the swapchain
    // mid-submission (the leftover-triangles/flicker symptom of an unclean
    // exit). `renderer` then drops here, before `window` — the surface
    // releases the raw handles while the SDL window is still alive.
    renderer.wait_idle();
    Ok(())
}
