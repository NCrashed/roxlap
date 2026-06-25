//! roxlap-cave-demo — procedural-cave showcase.
//!
//! Generates a [`BlueCaveGenerator`] world on startup, opens a winit
//! window, and renders the cave through the unified
//! [`roxlap_render::SceneRenderer`] facade — the same CPU (software
//! 3D-DDA) and GPU (wgpu) backends the scene-demo uses. Fly through with
//! WASD + mouse-look; click in the window to grab the cursor.
//!
//! The cave world is exactly one scene chunk (`128 × 128 × 256` =
//! `CHUNK_SIZE_XY × CHUNK_SIZE_XY × CHUNK_SIZE_Z`), so it maps to a
//! single-grid, single-chunk [`roxlap_scene::Scene`] at chunk `(0, 0,
//! 0)` with an identity transform — grid-local voxel coordinates equal
//! world coordinates.
//!
//! Run on the GPU backend with `ROXLAP_GPU=1` (falls back to the CPU
//! backend automatically if wgpu init fails).
//!
//! Controls:
//! - Click in the window → grab cursor (mouse-look active).
//! - `W`/`A`/`S`/`D` → forward / strafe-left / back / strafe-right.
//! - `Space` → up (world `-z`); `LShift` → down (world `+z`).
//! - Hold `LCtrl` for fast-fly (≈4× speed).
//! - `LMB` (while grabbed) → fire a plasma bullet (a glowing voxel
//!   sphere sprite) that flies along camera-forward and carves a sphere
//!   into the world on impact.
//! - `F` → toggle blue ↔ mag cave preset (regenerates the world).
//! - `R` → regenerate the world with the next seed (preset preserved).
//! - `Esc` → release cursor (or exit if already released).
//! - Window close → exit.
//!
//! Movement is collision-checked: the camera slides along walls instead
//! of clipping through them.

use std::sync::Arc;
use std::time::Instant;

use glam::IVec3;
use roxlap_cavegen::{BlueCaveGenerator, CaveParams, MagCaveGenerator, MAXZDIM};
use roxlap_core::world_query::{getcube, Cube};
use roxlap_core::Camera;
use roxlap_core::Engine;
use roxlap_core::OpticastSettings;
use roxlap_formats::kv6::Kv6;
use roxlap_formats::vxl::Vxl;
use roxlap_render::{
    DynSpriteTransform, FrameParams, RenderOptions, SceneRenderer, SpriteInstanceId, SpriteModelId,
};
use roxlap_scene::cavegen::CaveChunkGenerator;
use roxlap_scene::{ChunkGenerator, GridId, GridTransform, Scene, SpanOp, CHUNK_SIZE_XY};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{DeviceEvent, DeviceId, ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

/// World dimension. The cave is one scene chunk, so the lateral size is
/// fixed at [`CHUNK_SIZE_XY`]; [`MAXZDIM`] (= `CHUNK_SIZE_Z`) is the
/// vertical extent.
const VSID: u32 = CHUNK_SIZE_XY;

/// Walking speed (voxels / second).
const MOVE_SPEED: f64 = 32.0;
/// Multiplier applied while `LCtrl` is held.
const FAST_MULT: f64 = 4.0;
/// Mouse sensitivity (radians per pixel of cursor delta).
const MOUSE_SENS: f64 = 0.0025;
/// Pitch is clamped just shy of ±90° to keep the basis well-conditioned.
const PITCH_LIMIT: f64 = 88.0_f64 * std::f64::consts::PI / 180.0;

/// Maximum bullet flight distance, in voxel units. Bullets that exceed
/// this without hitting are despawned silently.
const BULLET_MAX_DIST: f64 = 96.0;

/// Bullet velocity, in voxels / second. Tuned so a bullet fired across
/// an open chamber takes ~1 s to land — visible "plasma bolt" arc, not
/// an instant ray.
const BULLET_VEL: f64 = 60.0;

/// Sphere carve radius applied at bullet impact, in voxels.
const FIRE_RADIUS: u32 = 4;

/// Radius (voxels) of the glowing sphere kv6 each bullet renders as. A
/// voxel-accurate sprite, so it occludes against the cave + scales with
/// perspective instead of the old fixed-pixel screen-space disc.
const BULLET_SPHERE_RADIUS: u32 = 3;

/// Bullet sprite colour (voxlap-packed `0x80RRGGBB`, high bit = shaded):
/// bright plasma pink, visible against both the blue and mag cave
/// palettes.
const BULLET_COLOR: u32 = 0x80FF_4080;

/// Voxlap colour stamped on the inner walls of the carved crater (the
/// voxels that were buried before the carve and are now newly exposed).
/// Charred dark grey with a faint orange tint reads as "scorched rock"
/// against both cave palettes. Encoded as `(brightness << 24) | (R <<
/// 16) | (G << 8) | B` per voxlap convention; brightness `0x80` is
/// voxlap's neutral.
#[allow(clippy::cast_possible_wrap)]
const CARVE_COLOR: i32 = 0x8050_3018u32 as i32;

/// Voxlap colour used when carving the spawn bubble — neutral mid-grey
/// that doesn't betray the carve's source as much as the scorched-amber
/// `CARVE_COLOR`.
#[allow(clippy::cast_possible_wrap)]
const SPAWN_BUBBLE_COLOR: i32 = 0x8060_6068u32 as i32;

/// Radius of the carve performed at world centre so the camera always
/// spawns inside an open pocket (cave-gen otherwise leaves the centre
/// randomly air or solid).
const SPAWN_BUBBLE_RADIUS: u32 = 6;

/// Voxlap lightmode driving [`roxlap_scene::Grid::bake_lightmode`]'s
/// per-voxel brightness bake. `1` = directional sun-style bake (every
/// visible voxel shaded from its surface normal), the look both the cave
/// and scene demos use.
const LIGHTMODE: u32 = 1;

/// Fog colour (RGB low-24-bit). The renderer blends each pixel toward
/// this colour by `depth / fog_max_dist`.
const FOG_COLOR: u32 = 0x0090_98B0;

/// Fog "max scan distance" in voxels. At this distance pixels blend
/// fully to `FOG_COLOR`. 128 voxels at vsid=128 is dense enough to dim
/// distant cave walls without obscuring nearby ones.
const FOG_MAX_SCAN_DIST: i32 = 128;

/// Edit headroom reserved on the cave chunk so runtime carves (spawn
/// bubble + thousands of bullet impacts) don't overflow the slab pool.
const EDIT_HEADROOM_BYTES: usize = 4 * 1024 * 1024;

/// Effective camera "skin" radius in voxel units. Movement is blocked
/// when any voxel intersected by a ±`PLAYER_RADIUS` cube around the
/// proposed new position is solid — this keeps the camera off walls
/// instead of letting it touch them at sub-pixel distance. Small enough
/// that the camera fits through the `SPAWN_BUBBLE_RADIUS = 6` carved
/// bubble + bullet-impact craters.
const PLAYER_RADIUS: f64 = 0.3;

/// In-flight plasma bullet. Travels in a straight line until it hits a
/// solid voxel (carved into a sphere) or exits the world / max-flight
/// envelope. Rendered as a glowing voxel-sphere sprite instance.
#[derive(Debug, Clone, Copy)]
struct Bullet {
    pos: [f64; 3],
    vel: [f64; 3],
    /// Distance travelled so far; bullet despawns past `BULLET_MAX_DIST`.
    travelled: f64,
    /// The sprite instance rendering this bullet; removed on despawn.
    inst: SpriteInstanceId,
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
}

/// Movement key flags packed into a single byte. Bit layout matches the
/// order of [`KeyCode`] queries in the input handler.
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
    window: Option<Arc<Window>>,
    /// Unified CPU/GPU renderer. Owns presentation, the brick cache, the
    /// z-buffer, and the sprite reps. Created in `resumed` —
    /// `ROXLAP_GPU=1` selects the GPU backend with automatic CPU
    /// fallback.
    renderer: Option<SceneRenderer>,
    engine: Engine,
    /// Single-grid, single-chunk scene holding the cave at chunk
    /// `(0, 0, 0)`.
    scene: Scene,
    grid_id: GridId,
    /// Glowing-sphere model every bullet instances. Registered once in
    /// `resumed` (needs the renderer); `None` until then.
    bullet_model: Option<SpriteModelId>,
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
    /// integrated each frame, removed on impact / out-of-bounds / past
    /// `BULLET_MAX_DIST`.
    bullets: Vec<Bullet>,
}

impl App {
    fn new() -> Self {
        let preset = Preset::Blue;
        let seed = preset.default_params().seed;
        let (scene, grid_id) = build_cave_scene(preset, seed);

        let mut engine = Engine::new();
        // Fog: low-24-bit colour (no brightness bit — see
        // `project_oracle_fog_disabled.md`).
        engine.set_fog(FOG_COLOR, FOG_MAX_SCAN_DIST);

        // Spawn the camera at the carved spawn bubble's centre. Identity
        // grid transform ⇒ world coords equal grid-local voxel coords.
        let cam_pos = [
            f64::from(VSID) * 0.5,
            f64::from(VSID) * 0.5,
            f64::from(MAXZDIM) * 0.5,
        ];

        Self {
            window: None,
            renderer: None,
            engine,
            scene,
            grid_id,
            bullet_model: None,
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

    /// Rebuild chunk `(0, 0, 0)` from `self.preset` + `self.seed`. Drops
    /// in-flight bullets (the voxels they were aimed at no longer exist),
    /// re-carves the spawn bubble, re-bakes lighting, and teleports the
    /// camera back to world centre.
    fn regenerate(&mut self) {
        // Remove every live bullet sprite instance, then forget them.
        if let Some(renderer) = self.renderer.as_mut() {
            for b in &self.bullets {
                renderer.remove_sprite_instance(b.inst);
            }
        }
        self.bullets.clear();

        install_cave_chunk(&mut self.scene, self.grid_id, self.preset, self.seed);

        self.cam_pos = [
            f64::from(VSID) * 0.5,
            f64::from(VSID) * 0.5,
            f64::from(MAXZDIM) * 0.5,
        ];
    }

    /// Build a [`Camera`] from the current `cam_pos` + `yaw` + `pitch`.
    fn camera(&self) -> Camera {
        // Voxlap's right-handed basis with z growing downward. Per
        // `feedback_voxlap_basis_chirality.md`: the camera's down vector
        // must satisfy `right × down == forward`. The cyclic relation is
        // `forward × right = down`.
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        let forward = [cy * cp, sy * cp, sp];
        let right = [-sy, cy, 0.0];
        // down = forward × right
        let down = [
            forward[1] * right[2] - forward[2] * right[1],
            forward[2] * right[0] - forward[0] * right[2],
            forward[0] * right[1] - forward[1] * right[0],
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
        if mag <= 1e-6 {
            return;
        }
        let step = [
            delta[0] / mag * speed * dt,
            delta[1] / mag * speed * dt,
            delta[2] / mag * speed * dt,
        ];

        // Per-axis collision against the cave chunk: try each axis
        // independently so the camera slides along walls instead of
        // jamming when one component collides. If already inside solid
        // (e.g. a regen edge case), skip the block test for that axis so
        // we can still escape.
        let chunk = self
            .scene
            .grid(self.grid_id)
            .and_then(|g| g.chunk(IVec3::ZERO));
        let already_stuck = chunk.is_some_and(|c| is_blocked(c, self.cam_pos));
        for axis in 0..3 {
            let mut candidate = self.cam_pos;
            candidate[axis] += step[axis];
            let blocked = chunk.is_some_and(|c| is_blocked(c, candidate));
            if already_stuck || !blocked {
                self.cam_pos[axis] = candidate[axis];
            }
        }
    }

    /// LMB fire — spawn a plasma bullet that flies along the camera's
    /// forward axis at `BULLET_VEL` voxels/sec. The bullet renders as a
    /// glowing voxel-sphere sprite and carves a sphere at impact.
    fn fire(&mut self) {
        let Some(model) = self.bullet_model else {
            return;
        };
        let cam = self.camera();
        // Spawn slightly ahead of the camera to avoid "shooting yourself
        // in the chin" if the player is brushing a wall.
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
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };
        let inst = renderer.add_sprite_instance_posed(model, bullet_pose(pos));
        self.bullets.push(Bullet {
            pos,
            vel,
            travelled: 0.0,
            inst,
        });
    }

    /// Integrate bullet positions, check collision, apply impact carves,
    /// and re-pose the surviving bullet sprites. Removes bullets that
    /// hit, exit world bounds, or fly past `BULLET_MAX_DIST`.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn step_bullets(&mut self, dt: f64) {
        if dt <= 0.0 {
            return;
        }
        let grid_id = self.grid_id;
        // Accumulate impacts + despawned instances; apply after the
        // iteration so we don't borrow the scene / renderer mutably while
        // reading the chunk for collision.
        let mut impacts: Vec<IVec3> = Vec::new();
        let mut despawned: Vec<SpriteInstanceId> = Vec::new();
        {
            let chunk = self.scene.grid(grid_id).and_then(|g| g.chunk(IVec3::ZERO));
            self.bullets.retain_mut(|b| {
                let dx = b.vel[0] * dt;
                let dy = b.vel[1] * dt;
                let dz = b.vel[2] * dt;
                b.pos[0] += dx;
                b.pos[1] += dy;
                b.pos[2] += dz;
                b.travelled += (dx * dx + dy * dy + dz * dz).sqrt();
                // Despawn once the bullet outruns its flight envelope.
                if b.travelled >= BULLET_MAX_DIST {
                    despawned.push(b.inst);
                    return false;
                }
                let vx = b.pos[0].floor() as i32;
                let vy = b.pos[1].floor() as i32;
                let vz = b.pos[2].floor() as i32;
                // Out of world.
                if vx < 0
                    || vy < 0
                    || (vx as u32) >= VSID
                    || (vy as u32) >= VSID
                    || !(0..MAXZDIM).contains(&vz)
                {
                    despawned.push(b.inst);
                    return false;
                }
                // Solid hit?
                let solid = chunk.is_some_and(|c| {
                    !matches!(
                        getcube(&c.data, &c.column_offset, c.vsid, vx, vy, vz),
                        Cube::Air
                    )
                });
                if solid {
                    impacts.push(IVec3::new(vx, vy, vz));
                    despawned.push(b.inst);
                    return false;
                }
                true
            });
        }

        // Apply impact carves + relight (CR.2: through the grid).
        for hit in impacts {
            carve_sphere(&mut self.scene, grid_id, hit, FIRE_RADIUS, CARVE_COLOR);
        }

        // Drop the despawned sprite instances, then re-pose the
        // survivors (one batched upload).
        if let Some(renderer) = self.renderer.as_mut() {
            for id in despawned {
                renderer.remove_sprite_instance(id);
            }
            let updates: Vec<(SpriteInstanceId, DynSpriteTransform)> = self
                .bullets
                .iter()
                .map(|b| (b.inst, bullet_pose(b.pos)))
                .collect();
            renderer.set_sprite_instance_transforms(&updates);
        }
    }

    fn redraw(&mut self) {
        let Some(window) = self.window.as_ref() else {
            return;
        };
        let size = window.inner_size();
        if size.width == 0 || size.height == 0 {
            return;
        }

        // Advance camera by real wall-clock dt — clamped so a long stall
        // doesn't teleport the camera on the next frame.
        let now = Instant::now();
        let dt = self
            .last_tick
            .map_or(0.0, |t| (now - t).as_secs_f64().min(0.1));
        self.last_tick = Some(now);
        self.integrate(dt);
        self.step_bullets(dt);

        let cam = self.camera();
        let settings = OpticastSettings::for_oracle_framebuffer(size.width, size.height);
        #[allow(clippy::cast_sign_loss)]
        let chunks_visible = (settings.max_scan_dist.max(1) as u32) / CHUNK_SIZE_XY + 4;

        let frame = FrameParams {
            settings: &settings,
            sky_color: self.engine.sky_color(),
            sky: self.engine.sky(),
            fog_color: self.engine.fog_color(),
            fog_max_scan_dist: self.engine.fog_max_scan_dist(),
            treat_z_max_as_air: true,
            gpu_mip_scan_dist: 64.0,
            gpu_max_outer_steps: chunks_visible,
            gpu_fov_y_rad: 60.0_f32.to_radians(),
            draw_sprites: true,
            side_shades: self.engine.side_shades(),
        };

        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };
        renderer.render(&mut self.scene, &cam, &frame);
        renderer.present();
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
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));

        // `ROXLAP_GPU=1` selects the GPU backend; roxlap-render falls
        // back to the CPU path automatically on any wgpu init failure.
        let want_gpu = std::env::var_os("ROXLAP_GPU").is_some_and(|v| v != "0" && !v.is_empty());
        let opts = RenderOptions {
            want_gpu,
            ..RenderOptions::default()
        };
        let init_size = window.inner_size();
        let mut renderer =
            SceneRenderer::new(window.clone(), (init_size.width, init_size.height), &opts);
        if let Some(info) = renderer.adapter_info() {
            eprintln!("roxlap-render: GPU backend — {info}");
        } else {
            eprintln!("roxlap-render: CPU backend");
        }

        // Register the glowing-sphere bullet model once.
        self.bullet_model = Some(renderer.add_sprite_model(&build_bullet_kv6()));

        self.window = Some(window);
        self.renderer = Some(renderer);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(new_size) => {
                if let Some(renderer) = self.renderer.as_mut() {
                    if new_size.width > 0 && new_size.height > 0 {
                        renderer.resize(new_size.width, new_size.height);
                    }
                }
            }
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

/// Build the single-grid, single-chunk cave [`Scene`] for `preset` +
/// `seed`: an identity-transform grid whose chunk `(0, 0, 0)` is
/// materialised from a [`CaveChunkGenerator`], spawn-bubble carved, and
/// lighting-baked. Returns the scene + its grid id.
fn build_cave_scene(preset: Preset, seed: u64) -> (Scene, GridId) {
    let mut scene = Scene::new();
    let grid_id = scene.add_grid(GridTransform::identity());
    install_cave_chunk(&mut scene, grid_id, preset, seed);
    (scene, grid_id)
}

/// (Re)materialise chunk `(0, 0, 0)` of `grid_id` from `preset` + `seed`:
/// attach the matching cave generator, replace the chunk's content
/// (reserving edit headroom for carves), carve the spawn bubble, and
/// bake directional lighting. Used by both [`build_cave_scene`] and
/// `App::regenerate` — `ensure_chunk_generated` is a no-op on an
/// existing chunk, so the content is replaced in place + the version
/// bumped (CR.0 / CR.5).
fn install_cave_chunk(scene: &mut Scene, grid_id: GridId, preset: Preset, seed: u64) {
    let generator = make_generator(preset, seed);
    // Generate the chunk up front (so we can replace existing content),
    // then keep the generator attached for parity with the streaming API.
    let chunk = generator.generate(IVec3::ZERO);
    let grid = scene.grid_mut(grid_id).expect("cave grid registered");
    grid.set_generator(Some(generator));

    let slot = grid.ensure_chunk(IVec3::ZERO);
    *slot = chunk;
    slot.reserve_edit_capacity(EDIT_HEADROOM_BYTES);
    grid.bump_chunk_version(IVec3::ZERO);

    // Carve a guaranteed-open spawn bubble at world centre so the camera
    // never spawns buried (set_sphere_with_colfunc bumps the version).
    grid.set_sphere_with_colfunc(
        spawn_centre(),
        SPAWN_BUBBLE_RADIUS,
        SpanOp::Carve,
        |_, _, _| SPAWN_BUBBLE_COLOR,
    );

    // Directional sun-style bake over the whole (single) chunk, then a
    // final version bump so the GPU backend re-uploads the baked
    // brightness bytes.
    grid.bake_lightmode(LIGHTMODE);
    grid.bump_chunk_version(IVec3::ZERO);
}

/// Carve a `radius` sphere of `color` at grid-local `centre`, then
/// re-bake lighting over the chunk + bump its version so the GPU
/// re-uploads. The world is one chunk, so a whole-chunk bake is cheap
/// enough and keeps dynamic carves correctly lit (CR.2).
fn carve_sphere(scene: &mut Scene, grid_id: GridId, centre: IVec3, radius: u32, color: i32) {
    let Some(grid) = scene.grid_mut(grid_id) else {
        return;
    };
    // Newly-exposed crater walls (previously buried solid) take `color`.
    grid.set_sphere_with_colfunc(centre, radius, SpanOp::Carve, |_x, _y, _z| color);
    grid.bake_lightmode(LIGHTMODE);
    grid.bump_chunk_version(IVec3::ZERO);
}

/// Construct the cave generator for `preset` with `seed` as the base
/// seed, type-erased to [`ChunkGenerator`] so both presets share a slot.
fn make_generator(preset: Preset, seed: u64) -> Arc<dyn ChunkGenerator> {
    let mut params = preset.default_params();
    params.seed = seed;
    match preset {
        Preset::Blue => Arc::new(CaveChunkGenerator::new(BlueCaveGenerator, params)),
        Preset::Mag => Arc::new(CaveChunkGenerator::new(MagCaveGenerator, params)),
    }
}

/// World-centre spawn point in integer voxel coords (grid-local =
/// world for the identity-transform cave grid).
#[allow(clippy::cast_possible_wrap)]
fn spawn_centre() -> IVec3 {
    IVec3::new((VSID / 2) as i32, (VSID / 2) as i32, MAXZDIM / 2)
}

/// Build the glowing voxel sphere every bullet renders as: a solid
/// `BULLET_SPHERE_RADIUS`-radius ball of plasma-pink voxels with a
/// centred pivot (so an instance's position places the ball's centre).
#[allow(clippy::cast_precision_loss)]
fn build_bullet_kv6() -> Kv6 {
    let n = BULLET_SPHERE_RADIUS * 2 + 1;
    let c = n as f32 * 0.5;
    let r = BULLET_SPHERE_RADIUS as f32 + 0.5;
    Kv6::from_fn_shaded(n, n, n, |x, y, z| {
        let dx = x as f32 + 0.5 - c;
        let dy = y as f32 + 0.5 - c;
        let dz = z as f32 + 0.5 - c;
        (dx * dx + dy * dy + dz * dz <= r * r).then_some(BULLET_COLOR)
    })
}

/// Identity-orientation sprite pose at world `pos`. The sphere is
/// rotationally symmetric, so no per-bullet orientation is needed.
#[allow(clippy::cast_possible_truncation)]
fn bullet_pose(pos: [f64; 3]) -> DynSpriteTransform {
    DynSpriteTransform {
        pos: [pos[0] as f32, pos[1] as f32, pos[2] as f32],
        ..DynSpriteTransform::default()
    }
}

/// Test whether any voxel intersected by a ±[`PLAYER_RADIUS`] cube
/// around `pos` is solid (or out-of-world). Out-of-bounds in any axis
/// counts as solid so the camera can't fly off the edge of the cave
/// volume.
///
/// Cheap: at `PLAYER_RADIUS = 0.3` the cube spans at most 1 voxel per
/// axis (typically), so this fans out to 1-8 [`getcube`] calls against
/// the cave chunk.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
fn is_blocked(chunk: &Vxl, pos: [f64; 3]) -> bool {
    let lo_x = (pos[0] - PLAYER_RADIUS).floor() as i32;
    let hi_x = (pos[0] + PLAYER_RADIUS).floor() as i32;
    let lo_y = (pos[1] - PLAYER_RADIUS).floor() as i32;
    let hi_y = (pos[1] + PLAYER_RADIUS).floor() as i32;
    let lo_z = (pos[2] - PLAYER_RADIUS).floor() as i32;
    let hi_z = (pos[2] + PLAYER_RADIUS).floor() as i32;
    let vsid = VSID as i32;
    for vz in lo_z..=hi_z {
        for vy in lo_y..=hi_y {
            for vx in lo_x..=hi_x {
                if vx < 0 || vy < 0 || vz < 0 || vx >= vsid || vy >= vsid || vz >= MAXZDIM {
                    return true;
                }
                if !matches!(
                    getcube(&chunk.data, &chunk.column_offset, chunk.vsid, vx, vy, vz),
                    Cube::Air
                ) {
                    return true;
                }
            }
        }
    }
    false
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `build_cave_scene` materialises chunk `(0, 0, 0)` and the spawn
    /// bubble leaves the camera spawn voxel as air.
    #[test]
    fn build_cave_scene_materialises_chunk_and_spawn_bubble() {
        let (scene, grid_id) = build_cave_scene(Preset::Blue, 1234);
        let grid = scene.grid(grid_id).expect("grid registered");
        let chunk = grid.chunk(IVec3::ZERO).expect("chunk (0,0,0) materialised");

        // The spawn centre voxel must be air (carved bubble).
        let c = spawn_centre();
        assert!(
            matches!(
                getcube(&chunk.data, &chunk.column_offset, chunk.vsid, c.x, c.y, c.z),
                Cube::Air
            ),
            "spawn centre should be carved open"
        );
        // The chunk is the single-chunk lateral size.
        assert_eq!(chunk.vsid, VSID);
    }

    /// Toggling the preset / bumping the seed re-materialises the chunk
    /// in place and keeps the spawn bubble open.
    #[test]
    fn install_cave_chunk_replaces_in_place() {
        let (mut scene, grid_id) = build_cave_scene(Preset::Blue, 7);
        let v0 = scene.grid(grid_id).unwrap().chunk_version(IVec3::ZERO);
        install_cave_chunk(&mut scene, grid_id, Preset::Mag, 99);
        let grid = scene.grid(grid_id).unwrap();
        let v1 = grid.chunk_version(IVec3::ZERO);
        assert!(v1 > v0, "version must advance after a regenerate");
        let chunk = grid.chunk(IVec3::ZERO).expect("chunk still present");
        let c = spawn_centre();
        assert!(matches!(
            getcube(&chunk.data, &chunk.column_offset, chunk.vsid, c.x, c.y, c.z),
            Cube::Air
        ));
    }
}
