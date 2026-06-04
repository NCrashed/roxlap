//! roxlap-scene-demo — interactive showcase of the scene-graph
//! engine. See `README.md` for the controls + the demo's
//! evolution roadmap as the scene-graph substages land.

mod collision;
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

use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Instant;

use glam::IVec3;
use roxlap_core::opticast::OpticastSettings;
use roxlap_core::rasterizer::ScratchPool;
use roxlap_core::Engine;
use roxlap_gpu::{GpuRenderer, GpuRendererSettings};
use roxlap_scene::render::render_scene_composed;
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
/// SCAN_DIST_MAX at 1500 to push the slider below the beam
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
    /// CPU compositor — populated unless the GPU path won the
    /// startup race (`ROXLAP_GPU=1` + a working WGPU adapter).
    surface: Option<softbuffer::Surface<Arc<Window>, Arc<Window>>>,
    /// GPU.1 renderer — `Some` iff the demo started with
    /// `ROXLAP_GPU=1` and WGPU init succeeded. Mutually exclusive
    /// with `surface`.
    gpu: Option<GpuRenderer>,
    /// GPU.3 chunk resident — uploaded once at startup from chunk
    /// `(0, 0, 0)` of the first grid that has one. `None` means
    /// either the GPU path isn't active or no chunk was
    /// materialised in time (streaming hills startup race) — the
    /// GPU branch falls back to clear-to-colour.
    gpu_chunk: Option<roxlap_gpu::GpuChunkResident>,
    /// Grid-local world voxel origin of the uploaded chunk (= chunk
    /// index × chunk size). Frame-time camera position is translated
    /// by this before being passed to `render_chunk`.
    gpu_chunk_origin_world: [f64; 3],
    engine: Engine,
    scene: SceneAndCamera,
    zbuffer: Vec<f32>,
    pool: ScratchPool,
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
    /// Post-S7.6: lighting + mip bake driver for streaming grids.
    /// Runs each frame right after `pump_streaming`; bakes any
    /// newly-installed chunks (and re-bakes their 4 cardinal
    /// neighbours so chunk-edge brightness banding resolves as
    /// chunks arrive). Empty + no-op when no streaming grid exists.
    bake_tracker: StreamingBakeTracker,
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
        // One slot per render thread. Strip-parallel rendering
        // (R12.3.1) splits each frame's y-range across the slots;
        // RENDER_THREADS caps the count below the efficiency knee.
        let n_threads = rayon::current_num_threads().clamp(1, RENDER_THREADS);
        let pool = ScratchPool::new_parallel(WIDTH, HEIGHT, MAX_GRID_VSID, n_threads);
        Self {
            window: None,
            surface: None,
            gpu: None,
            gpu_chunk: None,
            gpu_chunk_origin_world: [0.0, 0.0, 0.0],
            engine,
            scene,
            zbuffer: vec![f32::INFINITY; (WIDTH * HEIGHT) as usize],
            pool,
            input: InputState::new(),
            grabbed: false,
            last_frame: Instant::now(),
            capture_pending: false,
            scan_dist: SCAN_DIST_INITIAL,
            bake_tracker: StreamingBakeTracker::new(),
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

        // Step the camera + scene animations from input state.
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f64();
        self.last_frame = now;
        self.tick_camera(dt);
        // S5.2: advance the ship grid's rotation when the `R`
        // toggle is on. No-op otherwise.
        self.scene.tick_ship_spin(dt);
        // S7.6: streaming pump — runs only when
        // `ROXLAP_STREAM=1` activated `build_streaming_demo`.
        // Drains any chunk-results that arrived since the last
        // frame, evicts chunks past r_evict, dispatches missing
        // chunks within r_active onto the background pool. Cheap
        // when the camera is idle (drain pass short-circuits when
        // the inbox is empty).
        if self.scene.streaming_enabled {
            self.scene
                .scene
                .pump_streaming(glam::DVec3::from_array(self.scene.cam_pos));
            // Post-pump bake — see `StreamingBakeTracker` docs.
            // Resolves chunk-edge brightness banding by running
            // lightmode-1 + generate_mips on freshly-installed
            // chunks (and their cardinal neighbours) with a
            // neighbour-aware estnorm reader. Runs on the main
            // thread for `&mut Grid` access; bounded work since it
            // only touches the delta since last frame.
            self.bake_tracker.process(&mut self.scene.scene);
        }

        // GPU.1 short-circuit — see `redraw_gpu`.
        if self.gpu.is_some() {
            self.redraw_gpu();
            return;
        }

        // Resize ScratchPool / zbuffer when the window grew.
        let pixel_count = (size.width as usize) * (size.height as usize);
        if self.zbuffer.len() < pixel_count {
            self.zbuffer.resize(pixel_count, f32::INFINITY);
        }
        if self.pool.slot(0).uurend_half_stride < size.width as usize {
            let n_threads = self.pool.n_threads().max(1);
            self.pool =
                ScratchPool::new_parallel(size.width, size.height, MAX_GRID_VSID, n_threads);
        }

        let mut settings = OpticastSettings::for_oracle_framebuffer(size.width, size.height);
        settings.max_scan_dist = self.scan_dist;
        // S4B.5: per-chunk mips generated in scene::bake_lightmode_1.
        // AAMB: reverted from (4, 128) — the VC/CB/PRR cascade fixed
        // the beam bug; full 6-mip ladder at msd=64 now safe.
        settings.mip_levels = 6;
        settings.mip_scan_dist = 64;

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
            &mut self.scene.scene,
            &self.scene.camera,
            &settings,
            sky,
            self.engine.sky(),
        );

        if self.capture_pending {
            self.capture_pending = false;
            // Snapshot the buffer + camera state to disk so an
            // off-line tool (or another roxlap-scene-demo run with a
            // matching scene) can reproduce the exact pose.
            match write_capture(
                &buffer,
                size.width,
                size.height,
                &self.scene.camera,
                self.scene.yaw,
                self.scene.pitch,
                self.scene.ship_angles,
                self.scene.spin_enabled,
            ) {
                Ok(()) => eprintln!(
                    "captured: roxlap-scene-capture.txt + .ppm (pos=({:.2}, {:.2}, {:.2}) yaw={:.4} pitch={:.4} ship=[{:.3}, {:.3}, {:.3}])",
                    self.scene.cam_pos[0],
                    self.scene.cam_pos[1],
                    self.scene.cam_pos[2],
                    self.scene.yaw,
                    self.scene.pitch,
                    self.scene.ship_angles[0],
                    self.scene.ship_angles[1],
                    self.scene.ship_angles[2],
                ),
                Err(e) => eprintln!("capture failed: {e}"),
            }
        }

        buffer.present().expect("softbuffer: present");
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }

    /// GPU.3 substitute for the softbuffer path: marches the
    /// uploaded chunk via the GPU renderer when one is resident,
    /// else falls back to GPU.1's clear-to-colour so the user still
    /// sees a window. Re-arms the redraw loop.
    fn redraw_gpu(&mut self) {
        let Some(gpu) = self.gpu.as_mut() else {
            return;
        };
        if let Some(resident) = &self.gpu_chunk {
            let cam = &self.scene.camera;
            let cam_local = [
                (self.scene.cam_pos[0] - self.gpu_chunk_origin_world[0]) as f32,
                (self.scene.cam_pos[1] - self.gpu_chunk_origin_world[1]) as f32,
                (self.scene.cam_pos[2] - self.gpu_chunk_origin_world[2]) as f32,
            ];
            let camera = roxlap_gpu::Camera {
                position: cam_local,
                right: [
                    cam.right[0] as f32,
                    cam.right[1] as f32,
                    cam.right[2] as f32,
                ],
                down: [cam.down[0] as f32, cam.down[1] as f32, cam.down[2] as f32],
                forward: [
                    cam.forward[0] as f32,
                    cam.forward[1] as f32,
                    cam.forward[2] as f32,
                ],
                fov_y_rad: 60_f32.to_radians(),
            };
            let max_scan = u32::try_from(self.scan_dist.max(1)).unwrap_or(u32::MAX);
            gpu.render_chunk(resident, &camera, max_scan);
        } else {
            gpu.render();
        }
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }

    /// At GPU startup, find a materialised chunk to upload. Under
    /// streaming hills the ground (grid 0) starts empty, so we
    /// pump the streamer once around the camera spawn pose to
    /// force-load chunk (0, 0, 0). Then drive the per-frame bake
    /// tracker so the chunk's lightmode-1 alphas are written
    /// *before* we read its slab — otherwise the GPU sees the
    /// generator's flat alphas and the render is uniformly bright.
    /// Falls through any grid in order if (0, 0, 0) still isn't
    /// there. With `ROXLAP_STATIC=1` `build_demo` already baked
    /// every chunk; the pump + tracker are no-ops.
    fn upload_first_chunk(&mut self, gpu: &GpuRenderer) {
        if self.scene.streaming_enabled {
            self.scene
                .scene
                .pump_streaming_sync(glam::DVec3::from_array(self.scene.cam_pos));
            self.bake_tracker.process(&mut self.scene.scene);
        }

        // Walk grids in id order — `scene.grids()` iterates a
        // HashMap so its order is unspecified. The ground is grid
        // 0 by construction; we want it whenever it's present.
        let mut grids_by_id: Vec<_> = self.scene.scene.grids().collect();
        grids_by_id.sort_by_key(|(gid, _)| gid.raw());
        for (gid, grid) in grids_by_id {
            if let Some(vxl) = grid.chunk(IVec3::ZERO) {
                let upload = roxlap_gpu::decompress_chunk(vxl);
                let resident = roxlap_gpu::GpuChunkResident::upload(gpu.device(), &upload);
                // For now: grid-origin-only support. GPU.5 will
                // factor in `GridTransform`. The chunk occupies
                // grid-local voxel-coords [0, 128) × [0, 128) ×
                // [0, 256); under the identity transform that's
                // also its world placement.
                self.gpu_chunk_origin_world = [0.0; 3];
                eprintln!(
                    "GPU.3: uploaded chunk (0, 0, 0) of grid {} — {} KiB resident",
                    gid.raw(),
                    resident.resident_bytes() / 1024,
                );
                self.gpu_chunk = Some(resident);
                return;
            }
        }
        eprintln!("GPU.3: no chunk at (0, 0, 0) in any grid — falling back to clear-to-colour.");
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

        // GPU.1: opt-in via `ROXLAP_GPU=1`. Falls back to the CPU
        // softbuffer path on any WGPU init failure so a missing
        // driver doesn't bring the demo down.
        let want_gpu = std::env::var_os("ROXLAP_GPU").is_some_and(|v| v != "0" && !v.is_empty());
        if want_gpu {
            match GpuRenderer::new_blocking(window.clone(), GpuRendererSettings::default()) {
                Ok(gpu) => {
                    eprintln!("roxlap-gpu: {}", gpu.adapter_info());
                    window.set_title(&format!("roxlap-scene-demo (GPU: {})", gpu.adapter_info(),));
                    self.upload_first_chunk(&gpu);
                    self.gpu = Some(gpu);
                }
                Err(e) => {
                    eprintln!("roxlap-gpu init failed ({e}); falling back to softbuffer");
                }
            }
        }

        if self.gpu.is_none() {
            let context =
                softbuffer::Context::new(window.clone()).expect("softbuffer: Context::new");
            let surface = softbuffer::Surface::new(&context, window.clone())
                .expect("softbuffer: Surface::new");
            self.surface = Some(surface);
        }

        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(new_size) => {
                // GPU.1: the swapchain must follow physical resizes;
                // softbuffer resizes lazily inside `redraw`, so the
                // CPU branch needs nothing here.
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.resize(new_size.width, new_size.height);
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
