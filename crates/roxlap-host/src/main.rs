//! roxlap-host — the legacy engine-feature demo binary (winit +
//! softbuffer). Despite the name this is **not** a host-integration
//! helper crate: to embed roxlap in your own window / event loop,
//! depend on `roxlap-render` and start from its `quickstart` example.
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
//!   `roxlap-capture.{txt,ppm}` for off-line repro.
//! - `L` → toggle the demo point light on/off (lightmode 2 ↔ 0)
//!   for an A/B comparison of sprite shading. Off → uniform
//!   ambient; on → directional shadowing from the demo torch.
//! - `Esc` → release cursor (or exit if already released).
//! - Window close → exit.

use std::io::Read;
use std::num::NonZeroU32;
use std::rc::Rc;
use std::time::Instant;

use flate2::read::GzDecoder;
use roxlap_core::camera_math;
use roxlap_core::dda::{render_dda_parallel, BrickCache, DdaEnv};
use roxlap_core::dda_sprite::draw_sprite_dda;
use roxlap_core::kfa_draw::solve_kfa_limbs;
use roxlap_core::world_query::{getcube, Cube};
use roxlap_core::Camera;
use roxlap_core::Engine;
use roxlap_core::OpticastSettings;
use roxlap_formats::kfa::{Hinge, KfaSprite, Point3};
use roxlap_formats::sprite::Sprite;
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

/// Voxlap's hard z-axis ceiling. Worlds live in `[0, MAXZDIM)` along
/// the `+z` axis (which is *down* in voxlap's convention).
const MAXZDIM: i32 = 256;

/// Effective camera "skin" radius in voxel units. Movement is
/// blocked when any voxel intersected by a ±`PLAYER_RADIUS` cube
/// around the proposed new position is solid — keeps the camera
/// off walls instead of letting it touch / clip them. 0.3 matches
/// the cave demo's tuning; comfortable through corridors but
/// tight enough that the player feels obstacle contact.
const PLAYER_RADIUS: f64 = 0.3;

/// Embedded gzipped oracle world. Same fixture roxlap-formats uses
/// in its parser tests — no extra disk I/O at startup.
const ORACLE_VXL_GZ: &[u8] = include_bytes!("../../../assets/oracle.vxl.gz");

/// S1.X: when `true`, replace the 2048-VSID oracle world with a
/// small hand-built test scene — a flat ground plane with a
/// starting platform and a handful of colour-coded primitives
/// (cube, sphere, tower). No walls, no roof — flyable from
/// outside in any direction for the scene-engine work
/// (Substage S in PORTING-SCENE.md). Default `false` keeps the
/// existing oracle demo bit-stable; flip to `true` to probe
/// outside-camera rendering.
const USE_SMALL_TEST_SCENE: bool = true;

/// Edge dimension of the test scene. Wider than the used z
/// extent (~64 vox) so the world is visibly flat — a useful
/// probe for outside-camera flying both vertically (up through
/// the open top) and horizontally (off the XY edge).
const SMALL_VSID: u32 = 256;

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

/// Embedded panoramic sky texture for the textured-`startsky`
/// path. Whatever PNG the user has dropped in `assets/sky.png` is
/// baked into the binary at build time. Width maps to elevation
/// (horizon → zenith), height to azimuth (wrap-around).
const SKY_PNG: &[u8] = include_bytes!("../../../assets/sky.png");

/// Decode a PNG byte slice into a `roxlap_core::sky::Sky`.
///
/// Voxlap's sky-mapping convention: **texture width = elevation
/// gradient (horizon → zenith)**, **texture height = azimuth wrap
/// (360° around the camera)**. Standard equirectangular
/// panoramas are usually laid out the other way (width=azimuth,
/// height=elevation), so this function **transposes** the
/// decoded pixels: `transposed.width = png.height`,
/// `transposed.height = png.width`. The result is a tall, thin
/// texture matching voxlap's expectation.
///
/// On any decode failure the host falls back to
/// [`roxlap_core::sky::Sky::blue_gradient`] so the demo always
/// renders something — the error message lands on stderr.
fn load_png_sky(png_bytes: &[u8]) -> Result<roxlap_core::sky::Sky, String> {
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
    if pixel_bytes.len() != (png_w as usize) * (png_h as usize) * bytes_per_pixel {
        return Err(format!(
            "decoded byte count {} != {}*{}*{}",
            pixel_bytes.len(),
            png_w,
            png_h,
            bytes_per_pixel
        ));
    }

    let mut pixels = Vec::with_capacity((png_w as usize) * (png_h as usize));
    for chunk in pixel_bytes.chunks_exact(bytes_per_pixel) {
        let r = i32::from(chunk[0]);
        let g = i32::from(chunk[1]);
        let b = i32::from(chunk[2]);
        pixels.push((0x80 << 24) | (r << 16) | (g << 8) | b);
    }
    Ok(roxlap_core::sky::Sky::from_pixels(pixels, png_w, png_h))
}

/// Build a procedural checkerboard sky for visual-distortion
/// debugging. Voxlap's mapping: width = elevation (horizon →
/// zenith), height = azimuth (360° wrap-around).
///
/// Layout (64 elevation × 256 azimuth):
/// - **azimuth quadrants** are color-coded so you can tell which
///   way the camera faces just from a sky pixel:
///     - `0°..90°`  (the +X-facing quadrant) → **red** base
///     - `90°..180°` (+Y) → **green** base
///     - `180°..270°` (-X) → **blue** base
///     - `270°..360°` (-Y) → **yellow** base
/// - **tiles** are 8 elevation × 16 azimuth pixels; alternating
///   tiles use a darkened version of the base color so tile
///   boundaries are obvious.
/// - **horizon stripe** (elevation=0): a single white row makes
///   the horizon line trivial to spot.
/// - **zenith stripe** (elevation=63): black, same purpose for
///   straight-up.
///
/// Each pixel is voxlap's BGRA-with-brightness-bit packing
/// (`0x80 << 24 | r<<16 | g<<8 | b`).
fn build_checkerboard_sky() -> roxlap_core::sky::Sky {
    const W: u32 = 64; // elevation
    const H: u32 = 256; // azimuth
    const TILE_X: u32 = 8;
    const TILE_Y: u32 = 16;
    // Pixel order matches `Sky::from_pixels`: row-major in
    // (azimuth y, elevation x) — outer y, inner x. The loop layout
    // mirrors `load_png_sky`'s PNG-row iteration (PNG height = voxlap
    // azimuth, PNG width = voxlap elevation).
    let mut pixels = Vec::with_capacity((W * H) as usize);
    for y in 0..H {
        for x in 0..W {
            let (r_base, g_base, b_base) = match (y * 4) / H {
                0 => (0xe0u32, 0x40, 0x40), // +X = red
                1 => (0x40, 0xe0, 0x40),    // +Y = green
                2 => (0x40, 0x40, 0xe0),    // -X = blue
                _ => (0xe0, 0xd0, 0x40),    // -Y = yellow
            };
            let tile_x = x / TILE_X;
            let tile_y = y / TILE_Y;
            let dark = (tile_x + tile_y) & 1 == 1;
            let (r, g, b) = if dark {
                (r_base / 4, g_base / 4, b_base / 4)
            } else {
                (r_base, g_base, b_base)
            };
            // Horizon stripe: bright white at x=0 so you can tell
            // up from down on a stranded checker. Zenith stripe:
            // pure black at x=W-1 (straight-up). The interior is
            // checkerboard-as-above.
            let (r, g, b) = if x == 0 {
                (0xffu32, 0xff, 0xff)
            } else if x == W - 1 {
                (0u32, 0, 0)
            } else {
                (r, g, b)
            };
            #[allow(clippy::cast_possible_wrap)]
            let px = ((0x80u32 << 24) | (r << 16) | (g << 8) | b) as i32;
            pixels.push(px);
        }
    }
    roxlap_core::sky::Sky::from_pixels(pixels, W, H)
}

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

#[allow(clippy::struct_excessive_bools)]
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
    /// DDA brick-occupancy cache for the single world chunk. Built once
    /// (occupancy is independent of the lighting bake, which only edits
    /// colours), reused every frame.
    bricks: BrickCache,
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
    /// Procedurally-built 2-bone KFA demo. Bone 0 (root) is a
    /// meltsphere body; bone 1 is a coco arm hinged 15 voxels to
    /// the body's right via a z-axis rotation joint. The redraw
    /// loop drives `kfaval[1]` from `spawn_time.elapsed()` so the
    /// arm spins continuously around the body.
    kfa_demo: KfaSprite,
    /// Demo light parameters — kept around so the `L` hotkey can
    /// re-add the same light after a previous toggle cleared it.
    /// The toggle flips the engine between `lightmode=2` + this
    /// light (visible directional shading) and `lightmode=0` (no
    /// lighting at all → sprites render at full ambient via the
    /// nolighta path), giving a clear A/B visual comparison.
    /// Snapshot of the pristine (unbaked) `vxl.data` taken at
    /// startup. The `L` toggle restores from this when switching
    /// the light off so the world voxel intensities revert to
    /// their pre-bake state. ~30 MB on the oracle world — fine
    /// for an interactive demo.
    pristine_world: Box<[u8]>,
    /// Cached snapshot of the post-bake world. Built lazily on the
    /// first `L` ON press; subsequent toggles just memcpy between
    /// `pristine_world` and `baked_world` (instant) instead of
    /// re-running `update_lighting` (~3 seconds for the bake region
    /// below). Doubles the world memory footprint (~60 MB total)
    /// but makes the toggle responsive after the first press.
    baked_world: Option<Box<[u8]>>,
    /// World-space bounding box the bake covers. Voxlap C's
    /// `diag_down_lit` oracle pose bakes a 448×448 playable area;
    /// here we go larger (1024×1024 around the spawn) so all the
    /// scene the user is likely to fly through gets shaded —
    /// otherwise distant features like the red pillar render
    /// against unbaked default brightness while nearby ones look
    /// lit. Stored as `[x0, y0, z0, x1, y1, z1]`.
    bake_bbox: [i32; 6],
    /// Tracks the toggle so press handling stays idempotent across
    /// rapid presses.
    light_on: bool,
    /// `Y` hotkey: when `true`, the engine's sky is the procedural
    /// checkerboard built by [`build_checkerboard_sky`]; when
    /// `false`, the embedded `assets/sky.png` panorama. Used to
    /// debug visual distortions under OOB-camera views — the
    /// regular pattern of the checkerboard makes warping in the
    /// sky-sphere lookup obvious at a glance.
    use_checker_sky: bool,
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
        Camera::from_yaw_pitch(self.cam_pos, self.yaw, self.pitch)
    }

    fn toggle_sky(&mut self) {
        self.use_checker_sky = !self.use_checker_sky;
        if self.use_checker_sky {
            self.engine.set_sky(Some(build_checkerboard_sky()));
            eprintln!("sky: CHECKERBOARD (64×256, color-coded by azimuth)");
        } else {
            // Re-decode the PNG. Falls back to blue gradient if the
            // PNG is missing/corrupt — same behaviour as startup.
            let sky = match load_png_sky(SKY_PNG) {
                Ok(sky) => sky,
                Err(_) => roxlap_core::sky::Sky::blue_gradient(),
            };
            self.engine.set_sky(Some(sky));
            eprintln!("sky: PNG (assets/sky.png)");
        }
    }

    fn toggle_light(&mut self) {
        self.light_on = !self.light_on;
        if self.light_on {
            // Lightmode 1 (directional sun-style) — every voxel
            // gets `(tp.y * 0.5 + tp.z) * 64 + 103.5` from its
            // surface normal. No `LightSrc` consulted; mode 1
            // ignores `engine.lights()`. Smooth gradient across
            // the whole oracle world rather than mode 2's tightly
            // localised point lights.
            self.engine.set_lightmode(1);
            self.swap_in_baked_world();
            eprintln!("light: ON  (lightmode=1, world baked)");
        } else {
            self.engine.set_lightmode(0);
            self.swap_in_pristine_world();
            eprintln!("light: OFF (lightmode=0, world unbaked)");
        }
    }

    /// Make `vxl.data` show the lit world. First call runs the
    /// `update_lighting` bake (slow — typically a few seconds for
    /// the 1024×1024 demo region) and caches the result so future
    /// toggles are an instant memcpy.
    fn swap_in_baked_world(&mut self) {
        if self.baked_world.is_none() {
            eprintln!(
                "  baking world (lightmode={}, {} light(s), bbox=[{}..{}, {}..{}, {}..{}]) — first toggle, ~few seconds…",
                self.engine.lightmode(),
                self.engine.lights().len(),
                self.bake_bbox[0],
                self.bake_bbox[3],
                self.bake_bbox[1],
                self.bake_bbox[4],
                self.bake_bbox[2],
                self.bake_bbox[5],
            );
            // Always start the bake from pristine so multiple bakes
            // stay deterministic regardless of toggle history.
            self.vxl.data.copy_from_slice(&self.pristine_world);
            let bbox = self.bake_bbox;
            let started = Instant::now();
            roxlap_core::update_lighting(
                &mut self.vxl.data,
                &self.vxl.column_offset,
                self.vxl.vsid,
                bbox[0],
                bbox[1],
                bbox[2],
                bbox[3],
                bbox[4],
                bbox[5],
                self.engine.lightmode(),
                self.engine.lights(),
            );
            eprintln!("  bake done in {:?}", started.elapsed());
            // Snapshot the post-bake state so subsequent ON
            // toggles are an O(memcpy) restore.
            self.baked_world = Some(self.vxl.data.clone());
        } else if let Some(baked) = &self.baked_world {
            self.vxl.data.copy_from_slice(baked);
        }
    }

    /// Restore the original (pre-bake) brightness bytes by copying
    /// back the snapshot `pristine_world` taken at startup. Always
    /// wholesale — a `[u8]` `copy_from_slice` on the oracle world
    /// is well below interactive-feeling latency.
    fn swap_in_pristine_world(&mut self) {
        debug_assert_eq!(self.vxl.data.len(), self.pristine_world.len());
        self.vxl.data.copy_from_slice(&self.pristine_world);
    }

    fn integrate(&mut self, dt: f64) {
        if dt <= 0.0 {
            return;
        }
        let cam = self.camera();
        let fast = if self.keys.has(KeyState::FAST) {
            FAST_MULT
        } else {
            1.0
        };
        let speed = MOVE_SPEED * fast;
        let mut delta = [0.0f64; 3];
        if self.keys.has(KeyState::FORWARD) {
            for (d, &c) in delta.iter_mut().zip(cam.forward.iter()) {
                *d += c;
            }
        }
        if self.keys.has(KeyState::BACK) {
            for (d, &c) in delta.iter_mut().zip(cam.forward.iter()) {
                *d -= c;
            }
        }
        if self.keys.has(KeyState::RIGHT) {
            for (d, &c) in delta.iter_mut().zip(cam.right.iter()) {
                *d += c;
            }
        }
        if self.keys.has(KeyState::LEFT) {
            for (d, &c) in delta.iter_mut().zip(cam.right.iter()) {
                *d -= c;
            }
        }
        if self.keys.has(KeyState::DOWN) {
            // World-down (+z), independent of camera pitch.
            delta[2] += 1.0;
        }
        if self.keys.has(KeyState::UP) {
            delta[2] -= 1.0;
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

        // Per-axis collision-checked move: try each axis
        // independently so the camera slides along walls instead of
        // jamming when one component would collide. If we're already
        // inside solid (e.g., the camera spawned next to a sprite or
        // a baked world's brightness change re-classified a voxel),
        // skip the test for that axis so we can still escape.
        let already_stuck = is_blocked(&self.vxl, self.cam_pos);
        for axis in 0..3 {
            let mut candidate = self.cam_pos;
            candidate[axis] += step[axis];
            if already_stuck || !is_blocked(&self.vxl, candidate) {
                self.cam_pos[axis] = candidate[axis];
            }
        }
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
        let cam = self.camera();
        let sky = self.engine.sky_color();
        let settings = OpticastSettings::for_oracle_framebuffer(size.width, size.height);
        let pitch_pixels = size.width as usize;

        // DDA shading environment from the engine: textured sky panorama
        // (if any), distance fog, and per-face side shading. The bedrock
        // placeholder reads as transparent in the DDA sampler, so no
        // `treat_z_max_as_air` flag is needed (OOB rays fall through to
        // sky on their own).
        #[allow(clippy::cast_precision_loss)]
        let env = DdaEnv {
            sky: self.engine.sky(),
            fog_color: self.engine.fog_color(),
            fog_max_dist: self.engine.fog_max_scan_dist().max(0) as f32,
            side_shades: self.engine.side_shades(),
            materials: None,
            terrain_materials: &[],
            lights: roxlap_core::CpuLights::default(),
            world_shadow: None,
            // CA.0 — the host's single-chunk world has no deck clip.
            z_clip: None,
            // OC.0 — nor a view cutout (first-person oracle host).
            cutout: None,
            fow: None,
        };

        // The world is one chunk; its brick occupancy is independent of
        // the lighting bake (colours only), so build once and reuse.
        let grid = roxlap_core::GridView::from_single_vxl(&self.vxl);
        self.bricks.ensure([0, 0, 0], 0, 0, &grid);

        let Some(surface) = self.surface.as_mut() else {
            return;
        };
        surface.resize(w_nz, h_nz).expect("softbuffer: resize");
        let mut buffer = surface.buffer_mut().expect("softbuffer: buffer_mut");
        // Pre-fill sky + reset the z-buffer; the DDA leaves a miss
        // untouched (sky) and writes hits with perpendicular depth.
        for px in buffer.iter_mut() {
            *px = sky;
        }
        for z in &mut self.zbuffer[..pixel_count] {
            *z = f32::INFINITY;
        }

        // Scope the buffer borrow so it ends before present.
        {
            render_dda_parallel(
                &cam,
                &settings,
                grid,
                &mut buffer,
                &mut self.zbuffer[..pixel_count],
                pitch_pixels,
                &env,
                &self.bricks,
                0,
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
            // Debug: ROXLAP_HOST_SPRITE_NO_Z=1 wipes the zbuffer back to
            // +∞ before sprite render, so sprites draw unconditionally on
            // top of the terrain — separates "z-test rejecting" from
            // "geometry broken".
            if std::env::var("ROXLAP_HOST_SPRITE_NO_Z").is_ok() {
                for z in &mut self.zbuffer[..pixel_count] {
                    *z = f32::INFINITY;
                }
            }
            let debug = std::env::var("ROXLAP_HOST_SPRITE_DEBUG").is_ok();

            // DDA.8 clean-room sprite raycaster — composites against the
            // shared z-buffer just like the terrain pass.
            for (i, sprite) in self.sprites.iter().enumerate() {
                let written = draw_sprite_dda(
                    &mut buffer,
                    &mut self.zbuffer[..pixel_count],
                    pitch_pixels,
                    size.width,
                    size.height,
                    &cam_state,
                    &settings,
                    sprite,
                );
                if debug {
                    eprintln!(
                        "sprite[{i}]: pos=({:.1}, {:.1}, {:.1}) → wrote {written} pixels",
                        sprite.p[0], sprite.p[1], sprite.p[2],
                    );
                }
            }

            // Animate the KFA demo: spin bone 1 around its hinge axis
            // once per ~4 seconds (Q15 angle; 16384 ticks/sec).
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let spin = (self.spawn_time.elapsed().as_secs_f32() * 16384.0) as i32;
            let spin_angle = (spin & 0xffff) as i16;
            let ax = self.kfa_demo.hinges[1].v[0];
            self.kfa_demo.kfaval[1] =
                roxlap_formats::xform::BoneXform::from_hinge_angle([ax.x, ax.y, ax.z], spin_angle);
            // Pose the bone hierarchy into world-space limb sprites, then
            // draw each limb as an ordinary DDA sprite (KFA = posed KV6s).
            solve_kfa_limbs(&mut self.kfa_demo);
            let mut kfa_written = 0u32;
            for limb in &self.kfa_demo.limbs {
                kfa_written += draw_sprite_dda(
                    &mut buffer,
                    &mut self.zbuffer[..pixel_count],
                    pitch_pixels,
                    size.width,
                    size.height,
                    &cam_state,
                    &settings,
                    limb,
                );
            }
            if debug {
                eprintln!(
                    "kfa_demo: pos=({:.1}, {:.1}, {:.1}) spin={spin_angle} → wrote {kfa_written} pixels",
                    self.kfa_demo.p[0], self.kfa_demo.p[1], self.kfa_demo.p[2],
                );
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
                        // S1.Y debugging: print camera pose so any
                        // screenshot taken via F can be paired with
                        // the exact camera state that produced it.
                        let vsid = self.vxl.vsid;
                        let in_xy = self.cam_pos[0] >= 0.0
                            && self.cam_pos[1] >= 0.0
                            && self.cam_pos[0] < f64::from(vsid)
                            && self.cam_pos[1] < f64::from(vsid);
                        eprintln!(
                            "F: vsid={vsid}  pos=({:.2}, {:.2}, {:.2})  yaw={:.4}  pitch={:.4}  in_xy={in_xy}",
                            self.cam_pos[0],
                            self.cam_pos[1],
                            self.cam_pos[2],
                            self.yaw,
                            self.pitch,
                        );
                    }
                    KeyCode::KeyL if pressed => {
                        self.toggle_light();
                    }
                    KeyCode::KeyY if pressed => {
                        // Toggle between the embedded `assets/sky.png`
                        // and a procedural checkerboard sky for OOB-
                        // distortion debugging.
                        self.toggle_sky();
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

/// S1.X: hand-built flyable test scene.
///
/// Layout (voxlap z-down: z=0 is up, z=255 is down):
/// - Flat ground plane at z = `GROUND_Z` (one voxel thick).
/// - Starting platform at world centre, raised 8 voxels above
///   ground, 16×16 voxel footprint, light grey.
/// - Red cube 24³ east of platform.
/// - Blue sphere radius 12 west of platform.
/// - Yellow tower 8×8×64 north of platform.
/// - Everything else is air. No walls, no roof.
///
/// Camera spawns 1 voxel above the platform centre, looking +x
/// (toward the red cube). Fly any direction to leave the world
/// boundary and see the scene from outside.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss
)]
fn build_small_test_scene() -> vxl::Vxl {
    use roxlap_cavegen::{pack_dense_grid_to_vxl, MAXZDIM};

    /// Top of the ground plane. Anything at z ≥ this is solid
    /// ground; anything above is air.
    const GROUND_Z: usize = 200;
    // Voxlap colour packing: (brightness << 24) | (R << 16) | (G << 8) | B.
    // 0x80 brightness = engine-default "neutral" lighting amplitude.
    const GROUND_COL: u32 = 0x80_3a_8a_3a; // mossy green
    const PLATFORM_COL: u32 = 0x80_b0_b0_b8; // light grey
    const CUBE_COL: u32 = 0x80_e0_30_30; // red
    const SPHERE_COL: u32 = 0x80_30_60_e0; // blue
    const TOWER_COL: u32 = 0x80_e0_d0_30; // yellow
    const PLATFORM_HALF: usize = 8;
    const PLATFORM_HEIGHT: usize = 8;

    let vsid = SMALL_VSID;
    let vsid_u = vsid as usize;
    let max_z = MAXZDIM as usize;
    let total = vsid_u * vsid_u * max_z;

    let mut grid = vec![0u8; total];
    let mut color = vec![0u32; total];

    // (y, x, z) order per pack_dense_grid_to_vxl's contract.
    let idx = |x: usize, y: usize, z: usize| -> usize { (y * vsid_u + x) * max_z + z };

    // 1. Flat ground plane covering the whole XY footprint, ONE
    //    voxel thick — anything below `GROUND_Z` is empty so rays
    //    looking up through the floor from outside the world render
    //    as sky (S1.Z negative-index walk + Startsky drain). A
    //    thicker ground would visually appear as an "infinite slab"
    //    extending downward when viewed from outside, which is
    //    geometrically correct but not what the scene-engine probe
    //    wants.
    for y in 0..vsid_u {
        for x in 0..vsid_u {
            let i = idx(x, y, GROUND_Z);
            grid[i] = 1;
            color[i] = GROUND_COL;
        }
    }

    // 2. Starting platform: 16×16 footprint at world centre,
    //    raised 8 voxels above ground (so platform top is at
    //    z = GROUND_Z - 8).
    let cx = vsid_u / 2;
    let cy = vsid_u / 2;
    let plat_top_z = GROUND_Z - PLATFORM_HEIGHT;
    for y in (cy - PLATFORM_HALF)..(cy + PLATFORM_HALF) {
        for x in (cx - PLATFORM_HALF)..(cx + PLATFORM_HALF) {
            for z in plat_top_z..GROUND_Z {
                let i = idx(x, y, z);
                grid[i] = 1;
                color[i] = PLATFORM_COL;
            }
        }
    }

    // 3. Red cube 24³, 32 voxels east of platform centre, sitting
    //    on the ground.
    fill_box(
        &mut grid,
        &mut color,
        vsid_u,
        max_z,
        cx + 32,
        cy - 12,
        GROUND_Z - 24,
        24,
        24,
        24,
        CUBE_COL,
    );

    // 4. Blue sphere radius 12, 32 voxels west of platform centre,
    //    centred 12 voxels above ground.
    fill_sphere(
        &mut grid,
        &mut color,
        vsid_u,
        max_z,
        (cx as i32 - 32, cy as i32, GROUND_Z as i32 - 12),
        12,
        SPHERE_COL,
    );

    // 5. Yellow tower 8×8 footprint, 64 voxels tall, 32 voxels
    //    north of platform.
    fill_box(
        &mut grid,
        &mut color,
        vsid_u,
        max_z,
        cx - 4,
        cy + 32,
        GROUND_Z - 64,
        8,
        8,
        64,
        TOWER_COL,
    );

    let mut vxl = pack_dense_grid_to_vxl(&grid, &color, vsid);
    // Spawn 2 voxels above the platform top, looking +x toward
    // the red cube.
    vxl.ipo = [cx as f64, cy as f64, (plat_top_z - 2) as f64];
    vxl.ist = [0.0, 1.0, 0.0]; // right (+y)
    vxl.ihe = [0.0, 0.0, 1.0]; // down (+z, voxlap z-down)
    vxl.ifo = [1.0, 0.0, 0.0]; // forward (+x)
    vxl
}

/// Fill an axis-aligned box `[ox, ox+sx) × [oy, oy+sy) × [oz, oz+sz)`
/// in the dense grid, clamping at world bounds. Helper for
/// [`build_small_test_scene`].
#[allow(clippy::too_many_arguments)]
fn fill_box(
    grid: &mut [u8],
    color: &mut [u32],
    vsid_u: usize,
    max_z: usize,
    ox: usize,
    oy: usize,
    oz: usize,
    sx: usize,
    sy: usize,
    sz: usize,
    col: u32,
) {
    let xe = (ox + sx).min(vsid_u);
    let ye = (oy + sy).min(vsid_u);
    let ze = (oz + sz).min(max_z);
    for y in oy..ye {
        for x in ox..xe {
            for z in oz..ze {
                let i = (y * vsid_u + x) * max_z + z;
                grid[i] = 1;
                color[i] = col;
            }
        }
    }
}

/// Fill a sphere centred at `(cx, cy, cz)` with radius `r` voxels
/// in the dense grid, clamping at world bounds.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::many_single_char_names
)]
fn fill_sphere(
    grid: &mut [u8],
    color: &mut [u32],
    vsid_u: usize,
    max_z: usize,
    centre: (i32, i32, i32),
    r: i32,
    col: u32,
) {
    let (cx, cy, cz) = centre;
    let r2 = r * r;
    for dz in -r..=r {
        for dy in -r..=r {
            for dx in -r..=r {
                if dx * dx + dy * dy + dz * dz > r2 {
                    continue;
                }
                let x = cx + dx;
                let y = cy + dy;
                let z = cz + dz;
                if x < 0 || y < 0 || z < 0 {
                    continue;
                }
                let (xu, yu, zu) = (x as usize, y as usize, z as usize);
                if xu >= vsid_u || yu >= vsid_u || zu >= max_z {
                    continue;
                }
                let i = (yu * vsid_u + xu) * max_z + zu;
                grid[i] = 1;
                color[i] = col;
            }
        }
    }
}

/// Test whether any voxel intersected by a ±[`PLAYER_RADIUS`] cube
/// around `pos` is solid. Cheap: at `PLAYER_RADIUS = 0.3` the cube
/// spans at most 1 voxel per axis (typically), so this fans out to
/// 1-8 [`getcube`] calls. Mirror of `roxlap-cave-demo::is_blocked`.
///
/// S1.X: out-of-world positions count as AIR (not solid) so the
/// camera can fly past the world's XY/Z bounds. The original
/// behaviour (out-of-world == solid) was to keep the camera
/// inside-grid for voxlap C parity; the scene-engine work needs
/// outside-camera as a first-class flight regime.
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
                    // Out-of-world is air — camera can fly outside.
                    continue;
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

#[allow(clippy::too_many_lines)] // straight-line setup; splitting hurts readability
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let vxl_world = if USE_SMALL_TEST_SCENE {
        build_small_test_scene()
    } else {
        load_oracle_vxl()
    };
    eprintln!(
        "loaded world: vsid={}, ipo=({:.1}, {:.1}, {:.1})",
        vxl_world.vsid, vxl_world.ipo[0], vxl_world.ipo[1], vxl_world.ipo[2],
    );
    let cam_pos = vxl_world.ipo;

    // Voxlap's classic per-side darkening (top, bot, left, right,
    // up, down) — moderate values that read as visible directional
    // shading without going so dark the floor crushes to black.
    // The oracle uses (0,…,0) so this only affects the interactive
    // host; goldens stay bit-exact.
    let mut engine = Engine::new();
    engine.set_side_shades(15, 15, 15, 15, 15, 15);

    // Load `assets/sky.png` as the panoramic sky texture. PNG
    // width maps to elevation (horizon → zenith); height wraps
    // around the camera as azimuth. On decode failure, fall back
    // to voxlap's "BLUE" gradient so the demo still has something
    // to render.
    let sky = match load_png_sky(SKY_PNG) {
        Ok(sky) => {
            eprintln!(
                "loaded sky.png: {}×{}",
                sky.xsiz + 1, // sky.xsiz is the post-decrement value
                sky.ysiz,
            );
            sky
        }
        Err(e) => {
            eprintln!("sky.png decode failed: {e} — falling back to blue gradient");
            roxlap_core::sky::Sky::blue_gradient()
        }
    };
    engine.set_sky(Some(sky));

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

    // Lightmode 1 (directional sun-style) is the default — every
    // voxel gets `(tp.y * 0.5 + tp.z) * 64 + 103.5` from its
    // surface normal. The bake runs once at startup, after world
    // load, in `App::new`'s caller path via `App::start_lit`. The
    // L hotkey toggles back to unlit (lightmode 0) for visual
    // comparison; the post-bake snapshot is reused on toggle so
    // re-enabling lighting is an O(memcpy) restore.

    // Procedural 2-bone KFA: meltsphere body + coco arm hinged
    // 15 voxels to the body's right via a z-axis rotation joint.
    // Placed at cam + [+30, 0, -20] — above the camera spawn line
    // (voxlap's +z = down, so -20 is 20 voxels up). The redraw
    // loop spins kfaval[1] over time.
    let kfa_root_pos = [cam_f32[0] + 30.0, cam_f32[1], cam_f32[2] - 20.0];
    let kfa_body = Sprite::axis_aligned(
        kv6::parse(SPRITE_MELTSPHERE_KV6).expect("parse meltsphere for kfa body"),
        kfa_root_pos,
    );
    let kfa_arm = Sprite::axis_aligned(
        kv6::parse(COCO_KV6).expect("parse coco for kfa arm"),
        kfa_root_pos, // overwritten by setlimb on first frame
    );
    let zero = Point3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    let z_axis = Point3 {
        x: 0.0,
        y: 0.0,
        z: 1.0,
    };
    let body_hinge = Hinge {
        parent: -1,
        p: [zero, zero],
        v: [z_axis, z_axis],
        vmin: 0,
        vmax: 0,
        htype: 0,
        filler: [0; 7],
    };
    let arm_hinge = Hinge {
        parent: 0,
        // Child anchor (arm-side velcro) at arm origin.
        // Parent anchor (body-side velcro) 15 voxels right of body
        // centre — the arm "attaches" at body.x+15.
        p: [
            zero,
            Point3 {
                x: 15.0,
                y: 0.0,
                z: 0.0,
            },
        ],
        v: [z_axis, z_axis],
        vmin: i16::MIN,
        vmax: i16::MAX,
        htype: 0,
        filler: [0; 7],
    };
    let kfa_demo = KfaSprite::new(
        vec![kfa_body, kfa_arm],
        vec![body_hinge, arm_hinge],
        kfa_root_pos,
    );

    // Bake region: a 1024×1024 area centred on the spawn camera,
    // covering the full voxlap z range. Big enough that distant
    // scene features (e.g. the red pillar) fall inside; the demo
    // light only affects voxels within `sqrt(r2)` (= 60 voxels) of
    // its position, so the rest of the bake region just gets the
    // base directional shading (`(tp.y*0.5 + tp.z)*16 + 47.5`),
    // matching voxlap C `diag_down_lit`'s look.
    #[allow(clippy::cast_possible_wrap)]
    let vsid_i = vxl_world.vsid as i32;
    #[allow(clippy::cast_possible_truncation)]
    let cx = cam_pos[0] as i32;
    #[allow(clippy::cast_possible_truncation)]
    let cy = cam_pos[1] as i32;
    let bake_half: i32 = 512;
    let bake_bbox: [i32; 6] = [
        (cx - bake_half).max(0),
        (cy - bake_half).max(0),
        0,
        (cx + bake_half).min(vsid_i),
        (cy + bake_half).min(vsid_i),
        256,
    ];

    // Snapshot the pristine world so the `L` toggle can restore
    // the un-lit state on demand. The post-bake snapshot is built
    // lazily on first toggle (see `App::swap_in_baked_world`).
    let pristine_world = vxl_world.data.clone();

    let mut app = App {
        window: None,
        surface: None,
        engine,
        zbuffer: Vec::new(),
        bricks: BrickCache::new(),
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
        kfa_demo,
        pristine_world,
        baked_world: None,
        bake_bbox,
        // R-host: lightmode 1 active at startup. `App::new`'s
        // direct callers want this on; the L hotkey toggles
        // back to unlit (`lightmode = 0`) for visual comparison.
        // We initialise `light_on = false` so the toggle path
        // performs the first-time bake + snapshot, then flips
        // the flag — same code path as a runtime L press.
        light_on: false,
        use_checker_sky: false,
    };
    eprintln!("baking initial lightmode-1 directional bake — first call is a few seconds…");
    app.toggle_light();
    event_loop.run_app(&mut app)?;
    Ok(())
}
