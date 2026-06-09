//! roxlap-scene-demo — interactive showcase of the scene-graph
//! engine. See `README.md` for the controls + the demo's
//! evolution roadmap as the scene-graph substages land.

mod collision;
mod kv6_sprite;
mod markers;
#[cfg(test)]
mod repro;
#[cfg(test)]
mod repro_vc;
mod scene;
mod ship;
mod terrain;
#[cfg(test)]
mod vc6_repro;

use std::sync::Arc;
use std::time::Instant;

use roxlap_core::opticast::OpticastSettings;
use roxlap_core::sprite::SpriteLighting;
use roxlap_core::Engine;
use roxlap_formats::sprite::Sprite;
use roxlap_render::{FrameParams, RenderOptions, SceneRenderer, SpriteInstanceDesc, SpriteSet};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{DeviceEvent, DeviceId, ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

use crate::scene::{build_demo, SceneAndCamera, StreamingBakeTracker};

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;
/// Headroom for the per-frame [`ScratchPool`] sizing — `lastx`
/// inside each pool slot is sized `max(yres, vsid)`. The S4.0 demo
/// ships with a 2-chunk-wide ground (combined `vsid = 256`); pre-
/// allocating for 32×32 chunks keeps later demo expansions
/// allocation-free.
const MAX_GRID_VSID: u32 = 32 * roxlap_scene::CHUNK_SIZE_XY;

/// Initial max ray-march distance for the per-frame opticast pass.
/// User can adjust at runtime via `+` / `-` (range
/// [`SCAN_DIST_MIN`, `SCAN_DIST_MAX`]). Multi-mip absorbs the cost
/// of larger distances by transitioning distant rays to coarser
/// chunk LODs — at 384+ the mip-2 voxels dominate the budget while
/// mip-0 stays sharp near the camera.
///
/// AAMB (axis-aligned-mip-beams) was the cap rationale — kept
/// `SCAN_DIST_MAX` at 1500 to push the slider below the beam
/// threshold. The VC/CB/PRR cascade incidentally resolved the
/// beam bug (multi-chunk beam tests report 0 pixels across every
/// msd config at ml=6). Cap reverted to 1024 here as part of the
/// AAMB cleanup; the full 6-mip ladder is now safe at the
/// original config.
const SCAN_DIST_INITIAL: i32 = 384;
const SCAN_DIST_MIN: i32 = 64;
const SCAN_DIST_MAX: i32 = 1024;
const SCAN_DIST_STEP: i32 = 64;

/// Cap for `rayon`'s strip-parallel pool. Voxlap's per-strip
/// projection re-derivation adds fixed overhead that amortises
/// poorly past ~4 strips for an 800×600 frame; bench shows >4
/// threads slows down (per-strip overhead > work). Set high
/// enough to use modest multicore boost without going past the
/// efficiency knee.
const RENDER_THREADS: usize = 4;

const MOVE_SPEED: f64 = 64.0;

/// Embedded panoramic sky texture for the textured-`startsky`
/// path. Whatever PNG the user has dropped in `assets/sky.png` is
/// baked into the binary at build time. Width maps to elevation
/// (horizon → zenith), height to azimuth (wrap-around). Same asset
/// the roxlap-host demo ships.
const SKY_PNG: &[u8] = include_bytes!("../../../assets/sky.png");

/// Re-export of [`SKY_PNG`] under a stable name for the
/// `#[cfg(test)]` `repro` module to load the demo's sky panorama
/// without duplicating the bytes include.
#[cfg(test)]
pub(crate) const SKY_PNG_BYTES: &[u8] = SKY_PNG;

/// Decode a PNG byte slice into a `roxlap_core::sky::Sky`.
///
/// Voxlap's sky-mapping convention: **texture width = elevation
/// gradient (horizon → zenith)**, **texture height = azimuth wrap
/// (360° around the camera)**. Standard equirectangular panoramas
/// are usually laid out the other way (width=azimuth,
/// height=elevation), so `Sky::from_pixels` re-interprets the
/// dimensions accordingly. Mirror of roxlap-host's helper.
/// GPU.8 helper — decode `SKY_PNG` to a raw RGBA byte buffer
/// (`width * height * 4`). The GPU sky binding wants pixels in
/// equirectangular layout, which is exactly what the PNG already
/// is; the host of `load_png_sky` re-interprets the CPU side but
/// the GPU samples the original bytes directly.
pub(crate) fn load_png_sky_rgba(png_bytes: &[u8]) -> Result<(Vec<u8>, u32, u32), String> {
    let decoder = png::Decoder::new(png_bytes);
    let mut reader = decoder
        .read_info()
        .map_err(|e| format!("png header: {e}"))?;
    let info = reader.info();
    let width = info.width;
    let height = info.height;
    let (bytes_per_pixel, has_alpha) = match info.color_type {
        png::ColorType::Rgb => (3, false),
        png::ColorType::Rgba => (4, true),
        ct => return Err(format!("unsupported colour type {ct:?}; want RGB or RGBA")),
    };
    if info.bit_depth != png::BitDepth::Eight {
        return Err(format!(
            "unsupported bit depth {:?}; want 8-bit",
            info.bit_depth
        ));
    }
    let mut src = vec![0u8; reader.output_buffer_size()];
    reader
        .next_frame(&mut src)
        .map_err(|e| format!("png frame: {e}"))?;
    let mut rgba = Vec::with_capacity((width as usize) * (height as usize) * 4);
    for chunk in src.chunks_exact(bytes_per_pixel) {
        rgba.push(chunk[0]);
        rgba.push(chunk[1]);
        rgba.push(chunk[2]);
        rgba.push(if has_alpha { chunk[3] } else { 0xff });
    }
    Ok((rgba, width, height))
}

pub(crate) fn load_png_sky(png_bytes: &[u8]) -> Result<roxlap_core::sky::Sky, String> {
    let decoder = png::Decoder::new(png_bytes);
    let mut reader = decoder
        .read_info()
        .map_err(|e| format!("png header: {e}"))?;
    let info = reader.info();
    let png_w = info.width;
    let png_h = info.height;
    let bytes_per_pixel = match info.color_type {
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        ct => return Err(format!("unsupported colour type {ct:?}; want RGB or RGBA")),
    };
    if info.bit_depth != png::BitDepth::Eight {
        return Err(format!(
            "unsupported bit depth {:?}; want 8-bit",
            info.bit_depth
        ));
    }
    let mut pixel_bytes = vec![0u8; reader.output_buffer_size()];
    reader
        .next_frame(&mut pixel_bytes)
        .map_err(|e| format!("png frame: {e}"))?;
    let mut pixels = Vec::with_capacity((png_w as usize) * (png_h as usize));
    for chunk in pixel_bytes.chunks_exact(bytes_per_pixel) {
        let r = i32::from(chunk[0]);
        let g = i32::from(chunk[1]);
        let b = i32::from(chunk[2]);
        pixels.push((0x80 << 24) | (r << 16) | (g << 8) | b);
    }
    Ok(roxlap_core::sky::Sky::from_pixels(pixels, png_w, png_h))
}
/// GPU.9 — assemble the demo's KV6 sprites. Currently a single
/// `coco.kv6` placed at the same world position the throwaway
/// KV6-as-grid prototype used (~30 voxels in front of camera
/// spawn). Returns an empty `Vec` if the embedded asset fails to
/// parse, so the demo keeps booting either way.
fn build_sprites() -> Vec<Sprite> {
    match kv6_sprite::load_coco_kv6() {
        // Directly ahead of the spawn camera ([0, -120, 50] looking
        // +y) at eye level, so the splatter is exercised and FPS is
        // meaningful even with a static camera.
        Ok(kv6) => vec![Sprite::axis_aligned(kv6, [0.0, -75.0, 50.0])],
        Err(e) => {
            eprintln!("kv6_sprite: load_coco_kv6 failed ({e}); skipping sprite");
            Vec::new()
        }
    }
}

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
    window: Option<Arc<Window>>,
    /// Unified CPU/GPU renderer (RF). Owns presentation, the GPU
    /// scene residency + dirty-chunk tracking, the CPU compositor +
    /// pool/z-buffer, and the sprite reps. Created in `resumed` —
    /// `ROXLAP_GPU=1` selects the GPU backend with automatic CPU
    /// fallback.
    renderer: Option<SceneRenderer>,
    engine: Engine,
    scene: SceneAndCamera,
    input: InputState,
    grabbed: bool,
    last_frame: Instant,
    /// Set by the `F` hotkey; consumed in `redraw` after the frame
    /// is composited so the captured PPM is the same pixels the
    /// user just saw.
    capture_pending: bool,
    /// Live-adjustable scan distance (voxels). `+` / `-` bump it
    /// by `SCAN_DIST_STEP`; clamped to `[SCAN_DIST_MIN, SCAN_DIST_MAX]`.
    scan_dist: i32,
    /// GPU scene-grid LOD scan distance (world units), from
    /// `ROXLAP_GPU_MIP_SCAN_DIST` (default 64). Fed to the GPU
    /// backend each frame; ignored by the CPU backend.
    gpu_mip_scan_dist: f32,
    /// Base KV6 sprite(s); `build_sprite_set` expands these into the
    /// demo's instanced field handed to the renderer at startup.
    sprites: Vec<Sprite>,
    /// Post-S7.6: lighting + mip bake driver for streaming grids.
    /// Runs each frame right after `pump_streaming`; bakes any
    /// newly-installed chunks (and re-bakes their 4 cardinal
    /// neighbours so chunk-edge brightness banding resolves as
    /// chunks arrive). Empty + no-op when no streaming grid exists.
    bake_tracker: StreamingBakeTracker,
    /// Title bar prefix (e.g. `"roxlap-scene-demo (GPU: …)"`).
    /// `tick_fps` composes `"<base> — NNN FPS"` on top of this
    /// every ~500ms so the user can read the live frame rate
    /// without a HUD overlay.
    title_base: String,
    fps_frames: u32,
    fps_last: Instant,
}

impl App {
    fn new() -> Self {
        let mut engine = Engine::new();
        // Fog disabled — the embedded `assets/sky.png` panorama
        // provides a far-distance reference. Falls back to voxlap's
        // blue gradient if the PNG fails to decode.
        let sky = load_png_sky(SKY_PNG).unwrap_or_else(|e| {
            eprintln!("sky: PNG decode failed ({e}); falling back to blue gradient");
            roxlap_core::sky::Sky::blue_gradient()
        });
        engine.set_sky(Some(sky));

        // S7.6: `build_demo` now defaults to the streaming-hills
        // path — the ground grid is backed by
        // `HillsChunkGenerator` and pumped each frame. The
        // historical static 32×32 ground stays available via
        // `ROXLAP_STATIC=1` for repro / regression tests.
        let scene = build_demo();
        if scene.streaming_enabled {
            eprintln!(
                "streaming hills active (T prints chunks/pending, ROXLAP_STATIC=1 for static)"
            );
        }
        Self {
            window: None,
            renderer: None,
            engine,
            scene,
            input: InputState::new(),
            grabbed: false,
            last_frame: Instant::now(),
            capture_pending: false,
            scan_dist: SCAN_DIST_INITIAL,
            gpu_mip_scan_dist: 64.0,
            sprites: build_sprites(),
            bake_tracker: StreamingBakeTracker::new(),
            title_base: "roxlap-scene-demo".to_string(),
            fps_frames: 0,
            fps_last: Instant::now(),
        }
    }

    /// Bump the rolling frame counter and refresh the title bar
    /// every ~500ms. Both `redraw` (CPU path) and `redraw_gpu` (GPU
    /// path) call this at the end of a successful frame.
    fn tick_fps(&mut self) {
        self.fps_frames += 1;
        let now = Instant::now();
        let dt = (now - self.fps_last).as_secs_f32();
        if dt < 0.5 {
            return;
        }
        let fps = f64::from(self.fps_frames) / f64::from(dt);
        if let Some(window) = self.window.as_ref() {
            window.set_title(&format!("{} — {:.1} FPS", self.title_base, fps));
        }
        // GPU.11.2 — `ROXLAP_FPS_LOG=1` mirrors the title-bar FPS to
        // stderr so it can be captured in a terminal alongside the
        // present-mode / pass-toggle diagnostic lines.
        if std::env::var_os("ROXLAP_FPS_LOG").is_some() {
            eprintln!("fps: {fps:.1}");
        }
        self.fps_frames = 0;
        self.fps_last = now;
    }

    /// One frame: advance the scene from input + streaming, then hand
    /// it to the unified renderer (CPU or GPU — the demo no longer
    /// knows which). All the old per-backend glue lives in
    /// `roxlap-render` now.
    fn redraw(&mut self) {
        // Snapshot the size, then drop the window borrow so the
        // `&mut self` animation/render below doesn't collide with it.
        let size = {
            let Some(window) = self.window.as_ref() else {
                return;
            };
            window.inner_size()
        };
        if size.width == 0 || size.height == 0 {
            return;
        }

        // Advance camera + scene animation from input.
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f64();
        self.last_frame = now;
        self.tick_camera(dt);
        self.scene.tick_ship_spin(dt);
        // Streaming pump (no-op unless `build_streaming_demo` is
        // active) — drains arrivals, evicts past r_evict, dispatches
        // missing chunks, then bakes lighting/mips on the delta.
        if self.scene.streaming_enabled {
            self.scene
                .scene
                .pump_streaming(glam::DVec3::from_array(self.scene.cam_pos));
            self.bake_tracker.process(&mut self.scene.scene);
        }

        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };

        // Per-frame opticast settings (host owns scan distance).
        let mut settings = OpticastSettings::for_oracle_framebuffer(size.width, size.height);
        settings.max_scan_dist = self.scan_dist;
        settings.mip_levels = 6;
        settings.mip_scan_dist = 64;
        #[allow(clippy::cast_sign_loss)]
        let chunks_visible = (self.scan_dist.max(1) as u32) / roxlap_scene::CHUNK_SIZE_XY + 4;

        let lighting = SpriteLighting::from_engine(&self.engine);
        let frame = FrameParams {
            settings: &settings,
            sky_color: self.engine.sky_color(),
            sky: self.engine.sky(),
            fog_color: self.engine.fog_color(),
            fog_max_scan_dist: self.engine.fog_max_scan_dist(),
            treat_z_max_as_air: true,
            gpu_mip_scan_dist: self.gpu_mip_scan_dist,
            gpu_max_outer_steps: chunks_visible,
            gpu_fov_y_rad: 60_f32.to_radians(),
            sprite_lighting: Some(&lighting),
        };

        if self.capture_pending {
            renderer.request_capture();
        }
        let camera = self.scene.camera; // Camera is Copy
        renderer.render(&mut self.scene.scene, &camera, &frame);

        if self.capture_pending {
            self.capture_pending = false;
            if let Some((buf, w, h)) = renderer.take_capture() {
                match write_capture(
                    &buf,
                    w,
                    h,
                    &self.scene.camera,
                    self.scene.yaw,
                    self.scene.pitch,
                    self.scene.ship_angles,
                    self.scene.spin_enabled,
                ) {
                    Ok(()) => eprintln!(
                        "captured: roxlap-scene-capture.txt + .ppm (pos=({:.2}, {:.2}, {:.2}))",
                        self.scene.cam_pos[0], self.scene.cam_pos[1], self.scene.cam_pos[2],
                    ),
                    Err(e) => eprintln!("capture failed: {e}"),
                }
            } else {
                eprintln!("capture unsupported on the GPU backend");
            }
        }

        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
        self.tick_fps();
    }

    /// Expand the base KV6 sprite into the demo's instanced field: a
    /// green model + a recoloured red variant, checkerboarded across an
    /// N×N field (`ROXLAP_SPRITE_GRID`, default 16 ⇒ 256 instances).
    /// The red model is the `G`-carve target. Content only — the
    /// renderer builds the CPU draws + GPU registry from this.
    fn build_sprite_set(&self) -> Option<SpriteSet> {
        let base = self.sprites.first()?;
        // Red variant: recolour every KV6 voxel (keep alpha, force red)
        // — applied once on the CPU so both backends agree.
        let mut red = base.clone();
        for v in &mut red.kv6.voxels {
            v.col = (v.col & 0xFF00_0000) | 0x00FF_0000;
        }
        let models = vec![base.clone(), red];

        let n: i32 = std::env::var("ROXLAP_SPRITE_GRID")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(16);
        let spacing = 40.0_f32;
        let mut instances = Vec::new();
        for iy in 0..n {
            for ix in 0..n {
                #[allow(clippy::cast_precision_loss)]
                let dx = (ix as f32 - (n as f32 - 1.0) * 0.5) * spacing;
                #[allow(clippy::cast_precision_loss)]
                let dy = (iy as f32 - (n as f32 - 1.0) * 0.5) * spacing;
                let model = usize::from((ix + iy) % 2 != 0); // 0 green, 1 red
                instances.push(SpriteInstanceDesc {
                    model,
                    pos: [dx, dy, 40.0],
                });
            }
        }
        Some(SpriteSet {
            models,
            instances,
            carve_model: Some(1),
        })
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
            let step = [dx * inv * speed, dy * inv * speed, dz * inv * speed];
            collision::slide_with_collision(&self.scene.scene, &mut self.scene.cam_pos, step);
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
        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .expect("winit: create_window"),
        );

        // RF: one unified renderer. `ROXLAP_GPU=1` selects the GPU
        // backend; roxlap-render falls back to the CPU softbuffer path
        // automatically on any WGPU init failure.
        let want_gpu = std::env::var_os("ROXLAP_GPU").is_some_and(|v| v != "0" && !v.is_empty());
        let opts = RenderOptions {
            want_gpu,
            cpu_max_grid_vsid: MAX_GRID_VSID,
            cpu_render_threads: RENDER_THREADS,
            ..RenderOptions::default()
        };
        let mut renderer = SceneRenderer::new(window.clone(), &opts);

        self.title_base = if let Some(info) = renderer.adapter_info() {
            eprintln!("roxlap-render: GPU backend — {info}");
            format!("roxlap-scene-demo (GPU: {info})")
        } else {
            eprintln!("roxlap-render: CPU backend");
            "roxlap-scene-demo (CPU)".to_string()
        };
        window.set_title(&self.title_base);

        // Sky panorama (GPU shader sky sampling; CPU samples engine.sky()).
        match load_png_sky_rgba(SKY_PNG) {
            Ok((rgba, w, h)) => {
                renderer.set_sky_panorama(&rgba, w, h);
                eprintln!("roxlap-render: sky panorama uploaded ({w}×{h})");
            }
            Err(e) => eprintln!("roxlap-render: sky decode failed ({e})"),
        }

        // GPU scene-grid LOD scan distance (world units). Default 64
        // (matches the CPU opticast `mip_scan_dist`).
        // `ROXLAP_GPU_MIP_SCAN_DIST=0` disables LOD; crank down to force
        // coarse mips close, or up to push banding out.
        if let Some(msd) = std::env::var("ROXLAP_GPU_MIP_SCAN_DIST")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
        {
            eprintln!("roxlap-render: scene mip scan dist = {msd}");
            self.gpu_mip_scan_dist = msd;
        }

        // Sprites: the demo's instanced field (green + red models).
        // `ROXLAP_GPU_NO_SPRITES=1` skips them (FPS isolation).
        let no_sprites = std::env::var_os("ROXLAP_GPU_NO_SPRITES").is_some_and(|v| v != "0");
        if no_sprites {
            eprintln!("roxlap-render: ROXLAP_GPU_NO_SPRITES — sprites disabled");
        } else if let Some(set) = self.build_sprite_set() {
            eprintln!(
                "roxlap-render: {} sprite instances, {} models ('G' carves the red model)",
                set.instances.len(),
                set.models.len(),
            );
            renderer.set_sprites(&set);
        }

        self.renderer = Some(renderer);
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(new_size) => {
                // GPU.1: the swapchain must follow physical resizes;
                // softbuffer resizes lazily inside `redraw`, so the
                // CPU branch needs nothing here.
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.resize(new_size.width, new_size.height);
                }
            }
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
                    KeyCode::KeyF if pressed => {
                        // Defer the actual write until after redraw
                        // composites this frame, so the captured PPM
                        // matches what's on screen.
                        self.capture_pending = true;
                    }
                    // S5.2: `R` toggles the ship grid's continuous
                    // Z-axis spin. Pressed-edge only — release is
                    // ignored so the toggle survives the key going up.
                    KeyCode::KeyR if pressed => {
                        self.scene.spin_enabled = !self.scene.spin_enabled;
                        eprintln!(
                            "ship spin = {}",
                            if self.scene.spin_enabled { "ON" } else { "OFF" }
                        );
                    }
                    // GPU.12: `G` structurally carves the next z-layer
                    // off the forked ("red") sprite model at runtime,
                    // rebuilds its LOD mips, and re-uploads — the base
                    // (non-red) instances stay intact (copy-on-modify).
                    // GPU backend only; a no-op on the CPU path.
                    KeyCode::KeyG if pressed => {
                        if let Some(renderer) = self.renderer.as_mut() {
                            let removed = renderer.carve_active_sprite();
                            if removed > 0 {
                                eprintln!("carved sprite z-layer ({removed} voxels)");
                            }
                        }
                    }
                    // S6.6: `B` toggles the marker pillars'
                    // LOD configuration between always-Near
                    // (default — full voxel, pre-S6.6 behaviour)
                    // and the tuned billboards split (closer
                    // pillars Near, farther pillars Far via S6.3's
                    // billboard impostor blit). Pressed-edge only.
                    KeyCode::KeyB if pressed => {
                        let on = self.scene.toggle_billboards_lod();
                        eprintln!("S6 billboards = {}", if on { "ON" } else { "OFF" });
                    }
                    // S7.6: `T` (telemetry) prints chunk count +
                    // pending count for each streaming-enabled
                    // grid. No-op when not in streaming mode.
                    KeyCode::KeyT if pressed && self.scene.streaming_enabled => {
                        for (id, grid) in self.scene.scene.grids() {
                            eprintln!(
                                "grid #{id} chunks={chunks} pending={pending} radius={r_active:.0}/{r_evict:.0}",
                                id = id.raw(),
                                chunks = grid.chunk_count(),
                                pending = grid.pending_gen.len(),
                                r_active = grid.stream_radius.r_active,
                                r_evict = grid.stream_radius.r_evict,
                            );
                        }
                    }
                    // `+` / `=` (same key on US layout, with or without Shift)
                    // and the numpad `+` bump scan distance up by
                    // SCAN_DIST_STEP. Both keys handled so the
                    // shift-modifier doesn't trip the binding.
                    KeyCode::Equal | KeyCode::NumpadAdd if pressed => {
                        self.scan_dist = (self.scan_dist + SCAN_DIST_STEP).min(SCAN_DIST_MAX);
                        eprintln!("scan_dist = {}", self.scan_dist);
                    }
                    // `-` and numpad `-` bump down.
                    KeyCode::Minus | KeyCode::NumpadSubtract if pressed => {
                        self.scan_dist = (self.scan_dist - SCAN_DIST_STEP).max(SCAN_DIST_MIN);
                        eprintln!("scan_dist = {}", self.scan_dist);
                    }
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

/// Save the current framebuffer + camera state to
/// `roxlap-scene-capture.{txt,ppm}` for off-line debugging. Same
/// shape as roxlap-host's `write_capture` so the existing
/// `roxlap-oracle find-hairlines` tooling can read it (with the
/// `--ours <path>` override).
///
/// `.txt` is human-readable `key = value` lines; `.ppm` is a
/// binary P6 RGB dump. Both files overwrite on each call.
#[allow(clippy::too_many_arguments)]
fn write_capture(
    buffer: &[u32],
    width: u32,
    height: u32,
    cam: &roxlap_core::Camera,
    yaw: f64,
    pitch: f64,
    ship_angles: [f64; 3],
    spin_enabled: bool,
) -> std::io::Result<()> {
    use std::io::Write;
    let txt = format!(
        "# roxlap-scene-demo capture — generated by F hotkey\n\
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
         forward = [{}, {}, {}]\n\
         # Ship grid rotation at the time of capture (S5.2). Reproduce by\n\
         # setting ship_angles[0..3] before render and setting\n\
         # spin_enabled = false (so the angles don't advance further).\n\
         ship_angle.x = {}\n\
         ship_angle.y = {}\n\
         ship_angle.z = {}\n\
         ship_spin_enabled = {}\n",
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
        ship_angles[0],
        ship_angles[1],
        ship_angles[2],
        spin_enabled,
    );
    std::fs::write("roxlap-scene-capture.txt", txt)?;

    let header = format!("P6\n{width} {height}\n255\n");
    let mut bytes = Vec::with_capacity(header.len() + (width as usize) * (height as usize) * 3);
    bytes.extend_from_slice(header.as_bytes());
    for &px in buffer {
        bytes.push(((px >> 16) & 0xff) as u8); // R
        bytes.push(((px >> 8) & 0xff) as u8); // G
        bytes.push((px & 0xff) as u8); // B
    }
    let mut f = std::fs::File::create("roxlap-scene-capture.ppm")?;
    f.write_all(&bytes)?;
    Ok(())
}
