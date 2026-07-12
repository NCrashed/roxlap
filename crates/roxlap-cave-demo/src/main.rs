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
//!   into the world on impact. DT.4 — rock the carve disconnects from
//!   the cave **crumbles**: it detaches as a falling sprite and bursts
//!   into colour-true debris (with an occlusion-shaded boom) where it
//!   lands. By design, a detached region bigger than the detection
//!   budget (~4096 voxels) stays put — shooting the single support out
//!   from under a whole gallery will NOT drop the gallery.
//!   `ROXLAP_NO_CRUMBLE=1` restores plain carves (the `ROXLAP_GPU`
//!   convention: unset, empty or `0` keep crumble ON; any other value
//!   disables it). Known v1 seams: falling debris survives an `R`/`F`
//!   regeneration (it keeps falling in the new world), and a fallen
//!   island's crystal voxels render opaque — the island model carries
//!   no material map, so the glow returns with DT.5's per-material
//!   shards.
//! - `F` → toggle blue ↔ mag cave preset (regenerates the world).
//! - `R` → regenerate the world with the next seed (preset preserved).
//! - `Esc` → release cursor (or exit if already released).
//! - Window close → exit.
//!
//! Movement is collision-checked: the camera slides along walls instead
//! of clipping through them.

use roxlap_render::Rgb;
use roxlap_render::VoxColor;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

#[cfg(feature = "audio")]
mod audio;

use glam::{IVec3, Vec3};
use roxlap_cavegen::{BlueCaveGenerator, CaveParams, MagCaveGenerator, MAXZDIM};
use roxlap_core::update_lighting;
use roxlap_core::world_query::{getcube, Cube};
use roxlap_core::Camera;
use roxlap_core::Engine;
use roxlap_core::LightSrc;
use roxlap_core::OpticastSettings;
use roxlap_formats::edit::set_sphere_with_colfunc;
use roxlap_formats::kv6::Kv6;
use roxlap_formats::vxl::Vxl;
use roxlap_render::{
    BackendPreference, CollisionMode, DebrisSystem, DynSpriteTransform, FrameParams, Material,
    ParticleEmitterDef, ParticleSystem, RenderOptions, SceneRenderer, SpriteInstanceId,
    SpriteModelId,
};
use roxlap_scene::cavegen::CaveChunkGenerator;
use roxlap_scene::islands::{detect_islands, FracturePattern, Island, DEFAULT_ISLAND_BUDGET};
use roxlap_scene::{
    BakeLight, BakeMode, CharacterBody, CharacterDef, ChunkGenerator, Grid, GridId, GridTransform,
    MoveMode, Scene, Solidity, SpanOp, WalkInput, CHUNK_SIZE_XY,
};
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

/// DT.4 — flood budget for the worker's post-carve island detection:
/// a disconnected region bigger than this stays put (assumed
/// supported), the engine default. `ROXLAP_NO_CRUMBLE=1` disables
/// detection entirely.
const CRUMBLE_BUDGET: usize = DEFAULT_ISLAND_BUDGET;

/// DT.4 — shatters between sprite-model-pool compactions (removed
/// island models are tombstoned until compacted).
const CRUMBLE_COMPACT_EVERY: u32 = 32;

/// Radius (voxels) of the glowing sphere kv6 each bullet renders as. A
/// voxel-accurate sprite, so it occludes against the cave + scales with
/// perspective instead of the old fixed-pixel screen-space disc.
const BULLET_SPHERE_RADIUS: u32 = 3;

/// Bullet sprite colour (voxlap-packed `0x80RRGGBB`, high bit = shaded):
/// bright plasma pink, visible against both the blue and mag cave
/// palettes.
const BULLET_COLOR: VoxColor = VoxColor(0x80FF_4080);

/// Voxlap colour stamped on the inner walls of the carved crater (the
/// voxels that were buried before the carve and are now newly exposed).
/// Charred dark grey with a faint orange tint reads as "scorched rock"
/// against both cave palettes. Encoded as `(brightness << 24) | (R <<
/// 16) | (G << 8) | B` per voxlap convention; brightness `0x80` is
/// voxlap's neutral.
#[allow(clippy::cast_possible_wrap)]
const CARVE_COLOR: VoxColor = VoxColor(0x8050_3018);

/// Voxlap colour used when carving the spawn bubble — neutral mid-grey
/// that doesn't betray the carve's source as much as the scorched-amber
/// `CARVE_COLOR`.
#[allow(clippy::cast_possible_wrap)]
const SPAWN_BUBBLE_COLOR: VoxColor = VoxColor(0x8060_6068);

/// Radius of the carve performed at world centre so the camera always
/// spawns inside an open pocket (cave-gen otherwise leaves the centre
/// randomly air or solid).
const SPAWN_BUBBLE_RADIUS: u32 = 6;

/// Voxlap lightmode driving the carve worker's [`relight_bbox`] (the
/// raw [`update_lighting`] path). EV.4 — `2` = point-light bake: the
/// dim base + the crystal [`BakeLight`] pools, matching the grid-side
/// `BakeMode::PointLights` full bake so an incremental carve relight
/// reproduces the same bytes.
const LIGHTMODE: u32 = 2;

// ── EV.4 — glowing crystals ─────────────────────────────────────────
//
// Crystal clusters are planted on cavity walls at world (re)build. The
// crystal COLOUR doubles as the material key: the renderer's terrain
// material map routes exactly these RGBs to the translucent emissive
// crystal material, and the bake routes the matching `BakeLight`s'
// glow pools into the surrounding brightness bytes. The generator's
// noise-shaded palette never produces these exact RGBs, so nothing
// else in the cave picks the material up by accident.

/// Crystal voxel colour for the Blue preset — icy cyan.
const CRYSTAL_COLOR_BLUE: VoxColor = VoxColor(0x8040_E8FF);
/// Crystal voxel colour for the Mag preset — warm amber (contrast
/// against the magenta rock).
const CRYSTAL_COLOR_MAG: VoxColor = VoxColor(0x80FF_B040);
/// Material palette slot for the crystal (translucent + emissive).
const CRYSTAL_MATERIAL_ID: u8 = 1;
/// Crystal clusters planted per world (re)build.
const CRYSTAL_COUNT: usize = 16;
/// Crystal blob radius in voxels ([`roxlap_scene::Grid::set_sphere`]).
/// DT.5 bumped 2 → 3 (~33 → ~123 voxels): a `Shards { plates: 3 }`
/// split needs enough voxels per plate for the fracture structure to
/// read when a crystal crumbles (visual-pass note).
const CRYSTAL_RADIUS: u32 = 3;
/// Hard cutoff of a crystal's baked glow pool, voxels.
const CRYSTAL_LIGHT_RADIUS: f32 = 32.0;
/// Voxlap brightness scale of a crystal's glow (byte gain ≈
/// `strength / d²` — saturates the wall it sits on, fades out by
/// ~20 voxels; see [`BakeLight::strength`]).
const CRYSTAL_LIGHT_STRENGTH: f32 = 6000.0;

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

/// GPU mip-ladder depth to keep baked into the cave chunk. Mirrors the
/// GPU backend's `GPU_MAX_MIPS` (`roxlap-gpu`): the facade re-decompresses
/// an edited chunk each frame, and a chunk that already carries this many
/// mips takes the cheap read path (~6 ms) instead of regenerating the
/// whole ladder (~45 ms — the per-impact hitch this exists to avoid). The
/// background carve worker keeps the ladder fresh after every carve.
const GPU_MIP_LEVELS: u32 = 6;

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
    /// Unified CPU/GPU renderer. Owns presentation, the brick cache, the
    /// z-buffer, and the sprite reps. Created in `resumed` —
    /// `ROXLAP_GPU=1` selects the GPU backend with automatic CPU
    /// fallback.
    ///
    /// Declared **before** `window`: the renderer owns the wgpu
    /// surface/device, which must drop before the window they were created
    /// from. Rust drops fields top-to-bottom, so this order is the correct
    /// teardown even on the panic-unwind path (where `exiting` never runs).
    renderer: Option<SceneRenderer>,
    window: Option<Arc<Window>>,
    engine: Engine,
    /// Single-grid, single-chunk scene holding the cave at chunk
    /// `(0, 0, 0)`.
    scene: Scene,
    grid_id: GridId,
    /// Glowing-sphere model every bullet instances. Registered once in
    /// `resumed` (needs the renderer); `None` until then.
    bullet_model: Option<SpriteModelId>,
    cam_pos: [f64; 3],
    /// CC.3 — the engine controller (fly mode, a PLAYER_RADIUS cube
    /// body) replaces the demo's per-axis slide copy; `cam_pos` stays
    /// the outward-facing eye, synced after each walk.
    body: CharacterBody,
    /// `cam_pos` as of our last sync — a mismatch means regen/spawn
    /// moved the camera and the body must re-teleport.
    last_eye: [f64; 3],
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
    /// Background carve pipeline — applies bullet-impact carves +
    /// relight + mip rebuild off the main thread, swapping the result in
    /// when ready so impacts don't hitch the frame.
    carve: CarveWorker,
    /// DT.4 — falling voxel islands the worker's detection detaches;
    /// ticked each frame, drained impacts shatter into `particles`.
    debris: DebrisSystem,
    /// DT.4 — shatter debris particles (colour-true bursts from a
    /// landed island's own voxels).
    particles: ParticleSystem,
    /// One-voxel cube model every shatter particle instances.
    /// Registered in `resumed` alongside the bullet model.
    debris_model: Option<SpriteModelId>,
    /// Shatters since the last sprite-model-pool compaction.
    shatters_since_compact: u32,
    /// AU.3 — voxel-aware audio (feature `audio`). `None` when no audio
    /// device is available; the demo then runs silent.
    #[cfg(feature = "audio")]
    audio: Option<audio::DemoAudio>,
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

        // CC.3: a cube body the size of the old ±PLAYER_RADIUS probe
        // (eye at its centre), flying, with the cave's solid-bedrock
        // floor honoured (this world renders its bottom as rock).
        let mut body = CharacterBody::new(CharacterDef {
            radius: PLAYER_RADIUS,
            height: 2.0 * PLAYER_RADIUS,
            eye_height: PLAYER_RADIUS,
            fly_speed: MOVE_SPEED,
            solidity: Solidity {
                bedrock_blocks: true,
                ..Solidity::default()
            },
            ..CharacterDef::default()
        });
        body.set_mode(MoveMode::Fly);

        Self {
            renderer: None,
            window: None,
            engine,
            scene,
            grid_id,
            bullet_model: None,
            cam_pos,
            body,
            last_eye: [f64::NAN; 3],
            yaw: 0.0,
            pitch: 0.0,
            keys: KeyState::default(),
            grabbed: false,
            last_tick: None,
            preset,
            seed,
            bullets: Vec::new(),
            carve: CarveWorker::new(),
            debris: DebrisSystem::new(),
            particles: ParticleSystem::new(0xCA5E),
            debris_model: None,
            shatters_since_compact: 0,
            #[cfg(feature = "audio")]
            audio: audio::DemoAudio::new(),
        }
    }

    // AU.3 — audio hooks. Gated so the whole subsystem compiles out
    // (and the demo ships silent) without the `audio` feature; the
    // call sites below stay clean either way.
    #[cfg(feature = "audio")]
    fn audio_fire(&mut self, muzzle: [f64; 3]) {
        if let Some(audio) = self.audio.as_mut() {
            audio.fire(muzzle, &self.scene, glam::DVec3::from(self.cam_pos));
        }
    }
    #[cfg(feature = "audio")]
    fn audio_impacts(&mut self, hits: &[IVec3]) {
        if let Some(audio) = self.audio.as_mut() {
            audio.impacts(hits, &self.scene, glam::DVec3::from(self.cam_pos));
        }
    }
    #[cfg(feature = "audio")]
    fn audio_tick(&mut self, dt: f64) {
        if self.audio.is_none() {
            return; // silent: skip the per-frame camera/quat build
        }
        let orientation = audio::listener_orientation(&self.camera());
        let listener = glam::DVec3::from(self.cam_pos);
        let grid_id = self.grid_id;
        if let Some(audio) = self.audio.as_mut() {
            audio.tick(dt, &self.scene, grid_id, listener, orientation);
        }
    }
    #[cfg(feature = "audio")]
    fn audio_reset(&mut self) {
        if let Some(audio) = self.audio.as_mut() {
            audio.reset();
        }
    }

    #[cfg(not(feature = "audio"))]
    #[allow(clippy::unused_self)]
    fn audio_fire(&mut self, _muzzle: [f64; 3]) {}
    #[cfg(not(feature = "audio"))]
    #[allow(clippy::unused_self)]
    fn audio_impacts(&mut self, _hits: &[IVec3]) {}
    #[cfg(not(feature = "audio"))]
    #[allow(clippy::unused_self)]
    fn audio_tick(&mut self, _dt: f64) {}
    #[cfg(not(feature = "audio"))]
    #[allow(clippy::unused_self)]
    fn audio_reset(&mut self) {}

    /// Clean GPU teardown: drain in-flight work, then drop the renderer
    /// (wgpu device/queue/surface) before the window. Dropping the surface
    /// with the queue idle and no acquired frame, while the window still
    /// exists, is what keeps an exit from leaving the driver/compositor
    /// showing stale buffers. Idempotent.
    fn teardown(&mut self) {
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.wait_idle();
        }
        self.renderer = None;
        self.window = None;
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

        // Drop any in-flight background carve so its result (carved into
        // the OLD world) can't clobber the fresh chunk.
        self.carve.invalidate();

        // The crystal set is rebuilt below (indices change meaning), so
        // stop every hum and clear the reverb history first.
        self.audio_reset();

        install_cave_chunk(&mut self.scene, self.grid_id, self.preset, self.seed);

        self.cam_pos = [
            f64::from(VSID) * 0.5,
            f64::from(VSID) * 0.5,
            f64::from(MAXZDIM) * 0.5,
        ];
    }

    /// Build a [`Camera`] from the current `cam_pos` + `yaw` + `pitch`.
    fn camera(&self) -> Camera {
        // The canonical right-handed voxlap basis (`right × down ==
        // forward`) — always via `Camera::from_yaw_pitch`, never
        // hand-rolled (a chirality mistake silently culls sprites).
        Camera::from_yaw_pitch(self.cam_pos, self.yaw, self.pitch)
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
        // Normalise diagonal motion so two-key combos don't move √2×
        // faster. NO early return on idle: walk() must run every
        // frame so the wish-zero target stops the body (an early
        // return here left stale velocity — "every key moves like W"
        // right after flying forward).
        let mag = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
        let wish = if mag <= 1e-6 {
            glam::DVec3::ZERO
        } else {
            glam::DVec3::new(delta[0] / mag, delta[1] / mag, delta[2] / mag)
        };

        // CC.3: the engine controller slides the cube body; regen or
        // spawn moves `cam_pos` behind our back, so re-teleport when
        // it no longer matches our last sync.
        if self.cam_pos != self.last_eye {
            self.body.teleport(
                glam::DVec3::from(self.cam_pos) + glam::DVec3::new(0.0, 0.0, PLAYER_RADIUS),
            );
        }
        self.body.def_mut().fly_speed = speed;
        self.body
            .walk(&self.scene, dt, WalkInput { wish, jump: false });

        // The old probe treated out-of-world cells as SOLID so the
        // camera couldn't leave the cave volume; the engine treats
        // out-of-grid as air, so the world bounds become an explicit
        // clamp on the feet (velocity survives — sliding along the
        // boundary stays fast).
        let mut feet = self.body.pos();
        let hi_xy = f64::from(VSID) - PLAYER_RADIUS - 1e-3;
        let hi_z = f64::from(MAXZDIM) - 1e-3;
        feet.x = feet.x.clamp(PLAYER_RADIUS, hi_xy);
        feet.y = feet.y.clamp(PLAYER_RADIUS, hi_xy);
        feet.z = feet.z.clamp(2.0 * PLAYER_RADIUS, hi_z);
        if feet != self.body.pos() {
            self.body.set_pos(feet);
        }

        self.cam_pos = self.body.eye_pos().into();
        self.last_eye = self.cam_pos;
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
        // The bullet model is registered at startup, so the spawn
        // cannot see a stale handle.
        let inst = renderer
            .add_sprite_instance_posed(model, bullet_pose(pos))
            .expect("bullet model registered");
        self.bullets.push(Bullet {
            pos,
            vel,
            travelled: 0.0,
            inst,
        });
        self.audio_fire(pos);
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

        // Impact booms before the vec is consumed by the carve worker.
        self.audio_impacts(&impacts);

        // Hand impact carves to the background worker; it carves +
        // relights + re-mips a chunk clone off-thread, and `pump_carves`
        // swaps the result in when ready (no per-impact frame hitch).
        self.carve.enqueue(impacts);

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

    /// Swap in any finished background carve: poll the worker, and if a
    /// carved chunk is ready, replace chunk `(0, 0, 0)` with it + bump
    /// the version so the renderer re-uploads (cheap mip read-path).
    fn pump_carves(&mut self) {
        let Some(grid) = self.scene.grid(self.grid_id) else {
            return;
        };
        let Some(current) = grid.chunk(IVec3::ZERO) else {
            return;
        };
        // EV.4 — the crystal lights ride along so the worker's relight
        // reproduces the grid bake's glow pools (grid-local == chunk-
        // local for the single (0,0,0) cave chunk).
        let lights: Vec<LightSrc> = grid
            .bake_lights
            .iter()
            .map(|l| LightSrc {
                pos: l.pos.to_array(),
                r2: l.radius * l.radius,
                sc: l.strength,
            })
            .collect();
        let Some((new_chunk, islands)) = self.carve.pump(current, &lights) else {
            return;
        };
        if let Some(grid) = self.scene.grid_mut(self.grid_id) {
            *grid.ensure_chunk(IVec3::ZERO) = new_chunk;
            grid.bump_chunk_version(IVec3::ZERO);
        }
        // DT.4 — spawn each detached island as a falling sprite. The
        // worker already extracted them from the chunk we just swapped
        // in (and remipped the extraction bbox — hazard 3b), so the
        // spawn's own extract is a no-op carve + a cheap re-bake, and
        // its spawn-inside-geometry check runs against the real grid.
        for isl in islands {
            self.debris
                .spawn_island(&mut self.scene, self.grid_id, isl, BakeMode::PointLights);
        }
    }

    /// DT.4 — advance the falling islands, shatter the landed ones
    /// into colour-true debris particles (the DT.3 audio half lands
    /// here too: every impact booms through the same occlusion-shaded
    /// path as a bullet hit), and periodically compact the
    /// sprite-model pool (island models are tombstoned on removal).
    fn tick_crumble(&mut self, dt: f64) {
        let mut booms: Vec<IVec3> = Vec::new();
        if let Some(renderer) = self.renderer.as_mut() {
            self.debris.tick(renderer, &self.scene, dt);
            if let Some(model) = self.debris_model {
                for hit in self.debris.drain_impacts() {
                    let from = [hit.pos.x as f32, hit.pos.y as f32, hit.pos.z as f32];
                    // Harder landings kick the shards faster.
                    let kick = hit.speed as f32 * 0.25;
                    self.particles.voxel_debris(
                        &hit.burst_sites(),
                        from,
                        (2.0 + 0.5 * kick)..(5.0 + kick),
                        &ParticleEmitterDef {
                            lifetime: 0.6..1.4,
                            drag: 0.4,
                            collision: CollisionMode::Bounce { restitution: 0.35 },
                            fade_out_frac: 0.4,
                            scale_end: Some(0.4),
                            ..ParticleEmitterDef::new(model)
                        },
                    );
                    // Identity grid: world == grid-local voxel coords.
                    booms.push(IVec3::new(
                        hit.pos.x.floor() as i32,
                        hit.pos.y.floor() as i32,
                        hit.pos.z.floor() as i32,
                    ));
                    self.shatters_since_compact += 1;
                }
            }
            self.particles.tick_with_scene(renderer, dt, &self.scene);
            if self.shatters_since_compact >= CRUMBLE_COMPACT_EVERY {
                renderer.compact_sprite_models();
                self.shatters_since_compact = 0;
            }
        }
        if !booms.is_empty() {
            self.audio_impacts(&booms);
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
        self.pump_carves();
        self.tick_crumble(dt);
        self.audio_tick(dt);

        let cam = self.camera();
        let settings = OpticastSettings::for_oracle_framebuffer(size.width, size.height);

        // QE.2 — `FrameParams::new` + overrides; both backends project
        // from `settings` (one FOV, one derived scan budget).
        let mut frame = FrameParams::new(&settings);
        frame.sky_color = Rgb(self.engine.sky_color());
        frame.sky = self.engine.sky();
        frame.fog_color = Rgb(self.engine.fog_color());
        frame.fog_max_scan_dist = self.engine.fog_max_scan_dist();
        frame.side_shades = self.engine.side_shades();

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
        let backend = if std::env::var_os("ROXLAP_GPU").is_some_and(|v| v != "0" && !v.is_empty()) {
            BackendPreference::PreferGpu
        } else {
            BackendPreference::Cpu
        };
        let opts = RenderOptions {
            backend,
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
        // DT.4 — one white voxel; every shatter particle instances it,
        // tinted with its source voxel's colour.
        self.debris_model =
            Some(renderer.add_sprite_model(&Kv6::solid_cube(1, VoxColor(0x80FF_FFFF))));

        // EV.4 — the crystal material: translucent (the rock ghosts
        // through the gem) AND emissive (renders over-bright, immune to
        // the bake/shading). Both preset colours route to the same slot
        // so an F-toggle needs no material churn.
        renderer.define_material(
            CRYSTAL_MATERIAL_ID,
            Material::alpha_blend(180).with_emissive(255),
        );
        renderer.set_terrain_materials(&[
            (CRYSTAL_COLOR_BLUE.rgb_part(), CRYSTAL_MATERIAL_ID),
            (CRYSTAL_COLOR_MAG.rgb_part(), CRYSTAL_MATERIAL_ID),
        ]);
        // DT.5 — per-material crumble: rock (material 0) breaks into
        // rounded Voronoi lumps, crystal into sharp plates. The same
        // colour map switches island models to material-mapped
        // registration, so a falling crystal shard keeps its
        // translucent+emissive material and glows on the way down.
        self.debris.set_fracture_patterns(
            &[
                (CRYSTAL_COLOR_BLUE.rgb_part(), CRYSTAL_MATERIAL_ID),
                (CRYSTAL_COLOR_MAG.rgb_part(), CRYSTAL_MATERIAL_ID),
            ],
            &[
                (0, FracturePattern::Chunks { cell: 6 }),
                (CRYSTAL_MATERIAL_ID, FracturePattern::Shards { plates: 3 }),
            ],
        );

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

    /// Graceful shutdown — drain the GPU and drop the renderer (wgpu
    /// device/queue/surface) before the window, so an exit never tears the
    /// swapchain down mid-frame (the leftover-triangles/flicker symptom).
    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        self.teardown();
    }

    /// Same clean teardown when the platform suspends us; `resumed` rebuilds.
    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        self.teardown();
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

    // ANCHOR: crystal_bake
    // EV.4 — plant the glowing crystals (voxels + their bake lights)
    // BEFORE the bake so the first bake already writes their pools.
    plant_crystals(grid, preset, seed);

    // EV.4 — point-light bake: the dim voxlap lightmode-2 base plus a
    // glow pool around every crystal. The gloom is deliberate — the
    // cave now reads by crystal light.
    grid.bake(roxlap_scene::BakeMode::PointLights);
    // ANCHOR_END: crystal_bake

    // Build the GPU mip ladder up front so the renderer's first upload —
    // and every per-impact re-upload — takes the cheap mip read-path. The
    // background carve worker rebuilds it after each carve. (Whole-chunk
    // mip build ~30 ms; acceptable at load / on a deliberate F/R regen.)
    if let Some(vxl) = grid.chunk_mut(IVec3::ZERO) {
        vxl.generate_mips(GPU_MIP_LEVELS);
    }

    // Final version bump so the renderer re-uploads the baked brightness
    // + fresh mips.
    grid.bump_chunk_version(IVec3::ZERO);
}

/// EV.4 — plant [`CRYSTAL_COUNT`] glowing crystal clusters on cavity
/// walls: deterministic (seed + preset) rejection sampling picks an air
/// voxel, marches along a random axis to the nearest wall within reach,
/// and grows a small crystal blob into it, registering a [`BakeLight`]
/// floating in the air just off the surface. One extra crystal always
/// lands on the spawn bubble's floor so the player never spawns in
/// pitch black. Replaces (clears) any previous build's lights.
fn plant_crystals(grid: &mut Grid, preset: Preset, seed: u64) {
    grid.bake_lights.clear();
    let color = match preset {
        Preset::Blue => CRYSTAL_COLOR_BLUE,
        Preset::Mag => CRYSTAL_COLOR_MAG,
    };
    // xorshift64* — deterministic per (seed, preset), decoupled from the
    // generator's own noise seeding.
    let mut state = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(match preset {
            Preset::Blue => 0xB1,
            Preset::Mag => 0x4A,
        })
        | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    };
    // The six axis directions a crystal can grow along (into the wall).
    const DIRS: [IVec3; 6] = [
        IVec3::new(1, 0, 0),
        IVec3::new(-1, 0, 0),
        IVec3::new(0, 1, 0),
        IVec3::new(0, -1, 0),
        IVec3::new(0, 0, 1),
        IVec3::new(0, 0, -1),
    ];

    // One guaranteed crystal on the spawn bubble's floor.
    let plant_at = |grid: &mut Grid, air: IVec3, dir: IVec3| {
        // March from the air voxel to the wall (≤ 14 voxels away).
        for k in 1..=14 {
            let p = air + dir * k;
            if !(0..VSID as i32).contains(&p.x)
                || !(0..VSID as i32).contains(&p.y)
                || !(8..MAXZDIM - 1).contains(&p.z)
            {
                return false;
            }
            if grid.voxel_solid(p) {
                // ANCHOR: bake_light
                grid.set_sphere(p, CRYSTAL_RADIUS, Some(color));
                // The light floats in the air a few voxels off the
                // surface so the pool spreads over the wall around it.
                let lp = air + dir * (k - 3).max(0);
                grid.bake_lights.push(BakeLight {
                    pos: glam::Vec3::new(lp.x as f32, lp.y as f32, lp.z as f32),
                    radius: CRYSTAL_LIGHT_RADIUS,
                    strength: CRYSTAL_LIGHT_STRENGTH,
                });
                // ANCHOR_END: bake_light
                return true;
            }
        }
        false
    };
    plant_at(grid, spawn_centre(), IVec3::new(0, 0, 1));

    let mut planted = 1; // the spawn crystal
    let mut attempts = 0;
    while planted < CRYSTAL_COUNT && attempts < 800 {
        attempts += 1;
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let cand = IVec3::new(
            (8 + next() % u64::from(VSID - 16)) as i32,
            (8 + next() % u64::from(VSID - 16)) as i32,
            (24 + next() % 200) as i32,
        );
        if grid.voxel_solid(cand) {
            continue; // need an air pocket to glow into
        }
        // Spread the crystals out: reject candidates inside another
        // light's pool core.
        let too_close = grid.bake_lights.iter().any(|l| {
            let d = glam::Vec3::new(cand.x as f32, cand.y as f32, cand.z as f32) - l.pos;
            d.length_squared() < 24.0 * 24.0
        });
        if too_close {
            continue;
        }
        let dir = DIRS[(next() % 6) as usize];
        if plant_at(grid, cand, dir) {
            planted += 1;
        }
    }
}

/// A batch of bullet-impact carve centres to apply to a cloned cave
/// chunk on the worker thread. `epoch` tags the world generation the
/// clone came from, so a result that lands after a regenerate (F/R) is
/// dropped instead of clobbering the fresh world.
struct CarveJob {
    chunk: Vxl,
    impacts: Vec<IVec3>,
    epoch: u64,
    /// EV.4 — the crystal bake lights (chunk-local == grid-local: the
    /// cave is chunk `(0,0,0)`), so the relight writes the same bytes
    /// the grid-side `BakeMode::PointLights` bake would.
    lights: Vec<LightSrc>,
}

/// A finished carve: the carved + relit + re-mipped chunk, ready to swap
/// into the grid, plus the floating islands the batch disconnected
/// (DT.4 — already extracted from `chunk` and covered by its remip;
/// the main thread spawns them as falling debris).
struct CarveDone {
    chunk: Vxl,
    epoch: u64,
    islands: Vec<Island>,
}

/// Background carve pipeline. The heavy per-impact work — carve, local
/// relight, and (the expensive part, ~45 ms on the GPU path) rebuilding
/// the mip ladder — runs on a worker thread against a **clone** of the
/// cave chunk, so the main thread never stalls. The main thread keeps
/// rendering the un-carved chunk and **swaps** the finished one in when
/// it arrives (double-buffer swap), bumping the chunk version so the
/// renderer re-uploads it — which is now the cheap mip read-path because
/// the worker kept the ladder fresh.
struct CarveWorker {
    job_tx: Sender<CarveJob>,
    done_rx: Receiver<CarveDone>,
    /// A job is in flight on the worker thread.
    busy: bool,
    /// Impact centres queued since the last dispatch (and while busy).
    pending: Vec<IVec3>,
    /// World generation; bumped on regenerate so stale results are dropped.
    epoch: u64,
}

impl CarveWorker {
    fn new() -> Self {
        let (job_tx, job_rx) = std::sync::mpsc::channel::<CarveJob>();
        let (done_tx, done_rx) = std::sync::mpsc::channel::<CarveDone>();
        thread::Builder::new()
            .name("cave-carve".to_string())
            .spawn(move || carve_worker_loop(&job_rx, &done_tx))
            .expect("spawn cave carve worker");
        Self {
            job_tx,
            done_rx,
            busy: false,
            pending: Vec::new(),
            epoch: 0,
        }
    }

    /// Queue bullet-impact centres for background carving.
    fn enqueue(&mut self, impacts: impl IntoIterator<Item = IVec3>) {
        self.pending.extend(impacts);
    }

    /// Drop any in-flight job (the world was regenerated). Its result
    /// carries the old `epoch` and is discarded on arrival.
    fn invalidate(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        self.pending.clear();
        self.busy = false;
    }

    /// Collect a finished carve (if any) and dispatch the next batch.
    /// `current` is the grid's live chunk — the base the next job carves
    /// from (or the just-finished result, so queued-while-busy impacts
    /// stack correctly). Returns the chunk to swap into the grid, if one
    /// is ready this frame.
    fn pump(&mut self, current: &Vxl, lights: &[LightSrc]) -> Option<(Vxl, Vec<Island>)> {
        let mut ready: Option<(Vxl, Vec<Island>)> = None;
        while let Ok(done) = self.done_rx.try_recv() {
            self.busy = false;
            // Keep only a result from the current world generation.
            // Islands accumulate across drained results (a superseding
            // chunk was carved from its predecessor, post-extraction,
            // so they never duplicate).
            if done.epoch == self.epoch {
                let mut islands = done.islands;
                if let Some((_, mut prev)) = ready.take() {
                    prev.append(&mut islands);
                    islands = prev;
                }
                ready = Some((done.chunk, islands));
            }
        }
        if !self.busy && !self.pending.is_empty() {
            // Carve from the freshest state: the result we're about to
            // swap in, else the live grid chunk.
            let base = ready.as_ref().map_or(current, |(c, _)| c).clone();
            let impacts = std::mem::take(&mut self.pending);
            if self
                .job_tx
                .send(CarveJob {
                    chunk: base,
                    impacts,
                    epoch: self.epoch,
                    lights: lights.to_vec(),
                })
                .is_ok()
            {
                self.busy = true;
            }
        }
        ready
    }
}

/// Worker-thread loop: apply each job's carves + local relight to its
/// cloned chunk, rebuild the mip ladder once for the batch, and send it
/// back. Exits when the main thread drops the job sender.
fn carve_worker_loop(job_rx: &Receiver<CarveJob>, done_tx: &Sender<CarveDone>) {
    // DT.4 — `ROXLAP_NO_CRUMBLE=1` skips island detection entirely
    // (the escape hatch back to plain carves).
    let crumble = !std::env::var_os("ROXLAP_NO_CRUMBLE").is_some_and(|v| v != "0" && !v.is_empty());
    while let Ok(CarveJob {
        mut chunk,
        impacts,
        epoch,
        lights,
    }) = job_rx.recv()
    {
        // PF.12 — the batch's edit extent: carve spheres plus the
        // relight's internal ±ESTNORMRAD brightness writes. Feeds the
        // incremental `remip_bbox` below.
        let mut lo = IVec3::splat(i32::MAX);
        let mut hi = IVec3::splat(i32::MIN);
        for &hit in &impacts {
            // Newly-exposed crater walls (previously buried solid) take
            // CARVE_COLOR; a plain carve would leave them black.
            set_sphere_with_colfunc(
                &mut chunk,
                hit.into(),
                FIRE_RADIUS,
                SpanOp::Carve,
                |_, _, _| CARVE_COLOR,
            );
            relight_bbox(&mut chunk, hit, FIRE_RADIUS, &lights);
            let pad = FIRE_RADIUS as i32 + roxlap_core::ESTNORMRAD;
            lo = lo.min(hit - IVec3::splat(pad));
            hi = hi.max(hit + IVec3::splat(pad));
        }
        // DT.4 — floating-island crumble: detect the regions this batch
        // disconnected and extract them here, in the worker chunk, so
        // the remip below covers the extraction too (hazard 3b in
        // PORTING-DESTRUCTION.md — extraction on the main thread would
        // leave mip-N drawing the island beyond mip_scan_dist).
        // Chunk-local detection is exact for the single-(0,0,0) cave.
        // The chunk rides through a temporary identity Grid so the
        // grid-level detect/extract primitives apply; the crystal
        // lights become its bake_lights, so `BakeMode::PointLights`
        // writes the same bytes `relight_bbox` would (EV.4).
        let mut islands: Vec<Island> = Vec::new();
        if crumble && lo.x <= hi.x {
            let mut tmp = Grid::new(GridTransform::identity());
            tmp.bake_lights = lights
                .iter()
                .map(|l| BakeLight {
                    pos: Vec3::from(l.pos),
                    radius: l.r2.sqrt(),
                    strength: l.sc,
                })
                .collect();
            tmp.chunks.insert(IVec3::ZERO, chunk);
            let r = IVec3::splat(FIRE_RADIUS as i32);
            for &hit in &impacts {
                for isl in detect_islands(&tmp, hit - r, hit + r, CRUMBLE_BUDGET) {
                    // Extract immediately: a later impact's flood then
                    // cannot re-find (duplicate) this island.
                    isl.extract(&mut tmp, BakeMode::PointLights);
                    let pad = IVec3::splat(roxlap_core::ESTNORMRAD);
                    lo = lo.min(isl.bbox.0 - pad);
                    hi = hi.max(isl.bbox.1 + pad);
                    islands.push(isl);
                }
            }
            chunk = tmp
                .chunks
                .remove(&IVec3::ZERO)
                .expect("chunk moved in above");
        }
        // PF.12 — incremental re-mip: byte-identical to the old full
        // `generate_mips(GPU_MIP_LEVELS)` but only the edited columns'
        // mip data is recomputed (clean columns memcpy their old bytes).
        // This was the ~45 ms per-batch cost that forced the whole
        // worker-thread + whole-chunk-clone design.
        if lo.x <= hi.x {
            chunk.remip_bbox(lo.x, lo.y, hi.x, hi.y, GPU_MIP_LEVELS);
        }
        if done_tx
            .send(CarveDone {
                chunk,
                epoch,
                islands,
            })
            .is_err()
        {
            break; // main thread gone
        }
    }
}

/// Re-bake lighting over just the bounding box of a `radius` sphere
/// edit at `centre`, clamped to chunk bounds. `update_lighting` pads by
/// `ESTNORMRAD` internally, so only the geometric edit extent is needed
/// here — a ~0.04 ms relight vs a ~4 ms whole-chunk bake. The cave is a
/// single chunk with no neighbours, so the world-bounds clamp gives the
/// same result as the grid's neighbour-aware bake. EV.4 — `lights` are
/// the crystal [`BakeLight`]s as chunk-local [`LightSrc`], so a carve
/// inside a glow pool keeps its glow.
fn relight_bbox(chunk: &mut Vxl, centre: IVec3, radius: u32, lights: &[LightSrc]) {
    let r = radius as i32;
    let x0 = (centre.x - r).max(0);
    let y0 = (centre.y - r).max(0);
    let z0 = (centre.z - r).max(0);
    let x1 = (centre.x + r + 1).min(VSID as i32);
    let y1 = (centre.y + r + 1).min(VSID as i32);
    let z1 = (centre.z + r + 1).min(MAXZDIM);
    if x0 >= x1 || y0 >= y1 || z0 >= z1 {
        return;
    }
    update_lighting(
        &mut chunk.data,
        &chunk.column_offset,
        chunk.vsid,
        x0,
        y0,
        z0,
        x1,
        y1,
        z1,
        LIGHTMODE,
        lights,
    );
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // QE.2b - surface the facade's log-facade warnings (GPU-init
    // fallback) on stderr; RUST_LOG overrides.
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// DT.4 — the carve worker detects and extracts the islands its
    /// batch disconnects: a beam severed by the carve sphere comes
    /// back in `CarveDone::islands`, its voxels already air in the
    /// returned (remipped) chunk, the supported root untouched.
    #[test]
    fn carve_worker_detects_and_extracts_islands() {
        let mut g = Grid::new(GridTransform::identity());
        // Pillar to bedrock + a long beam; the r=4 carve at x=16
        // removes x ∈ [12, 20], leaving the tip x ∈ [21, 30] afloat.
        g.set_rect(
            IVec3::new(10, 10, 100),
            IVec3::new(10, 10, 255),
            Some(VoxColor(0x80B0_8040)),
        );
        g.set_rect(
            IVec3::new(11, 10, 100),
            IVec3::new(30, 10, 100),
            Some(VoxColor(0x80B0_8040)),
        );
        let chunk = g.chunks.remove(&IVec3::ZERO).expect("chunk built");

        let (job_tx, job_rx) = std::sync::mpsc::channel::<CarveJob>();
        let (done_tx, done_rx) = std::sync::mpsc::channel::<CarveDone>();
        thread::spawn(move || carve_worker_loop(&job_rx, &done_tx));
        job_tx
            .send(CarveJob {
                chunk,
                impacts: vec![IVec3::new(16, 10, 100)],
                epoch: 0,
                lights: Vec::new(),
            })
            .expect("worker alive");
        let done = done_rx
            .recv_timeout(std::time::Duration::from_secs(30))
            .expect("worker answers");
        drop(job_tx);

        assert_eq!(done.islands.len(), 1, "the severed tip is one island");
        let isl = &done.islands[0];
        assert_eq!(isl.voxels.len(), 10, "x ∈ [21, 30] at z=100");

        let mut tmp = Grid::new(GridTransform::identity());
        tmp.chunks.insert(IVec3::ZERO, done.chunk);
        for &(v, _) in &isl.voxels {
            assert!(!tmp.voxel_solid(v), "{v} extracted in the worker");
        }
        assert!(
            tmp.voxel_solid(IVec3::new(11, 10, 100)),
            "the supported beam root stays"
        );
    }

    /// Two impacts in one batch whose detection windows both see the
    /// same severed piece yield exactly ONE island — the
    /// immediate-extract invariant (each island is carved out of the
    /// worker chunk before the next impact's flood runs), which is
    /// what prevents duplicate falling sprites when shots queue up.
    /// Breaks silently if extraction ever moves after the detect loop.
    #[test]
    fn carve_worker_batch_dedups_islands() {
        let mut g = Grid::new(GridTransform::identity());
        // Two supported pillars bridged by a beam; the two r=4 carves
        // at x=15 and x=26 remove x ∈ [11, 19] ∪ [22, 30], leaving the
        // 2-voxel mid-piece x ∈ [20, 21] afloat — inside BOTH impacts'
        // padded detection windows (±FIRE_RADIUS ± 1).
        for x in [10, 31] {
            g.set_rect(
                IVec3::new(x, 10, 100),
                IVec3::new(x, 10, 255),
                Some(VoxColor(0x80B0_8040)),
            );
        }
        g.set_rect(
            IVec3::new(11, 10, 100),
            IVec3::new(30, 10, 100),
            Some(VoxColor(0x80B0_8040)),
        );
        let chunk = g.chunks.remove(&IVec3::ZERO).expect("chunk built");

        let (job_tx, job_rx) = std::sync::mpsc::channel::<CarveJob>();
        let (done_tx, done_rx) = std::sync::mpsc::channel::<CarveDone>();
        thread::spawn(move || carve_worker_loop(&job_rx, &done_tx));
        job_tx
            .send(CarveJob {
                chunk,
                impacts: vec![IVec3::new(15, 10, 100), IVec3::new(26, 10, 100)],
                epoch: 0,
                lights: Vec::new(),
            })
            .expect("worker alive");
        let done = done_rx
            .recv_timeout(std::time::Duration::from_secs(30))
            .expect("worker answers");
        drop(job_tx);

        assert_eq!(
            done.islands.len(),
            1,
            "one severed piece, one island — no duplicate from the second window"
        );
        assert_eq!(done.islands[0].voxels.len(), 2, "x ∈ [20, 21] at z=100");
    }

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

    /// EV.4 — crystals are planted with their bake lights: every light
    /// has a crystal-coloured voxel within reach, the spawn bubble gets
    /// its guaranteed crystal, and the presets colour-key differently.
    #[test]
    fn crystals_planted_and_lit() {
        for (preset, color) in [
            (Preset::Blue, CRYSTAL_COLOR_BLUE),
            (Preset::Mag, CRYSTAL_COLOR_MAG),
        ] {
            let (scene, grid_id) = build_cave_scene(preset, 1234);
            let grid = scene.grid(grid_id).expect("grid");
            assert!(
                grid.bake_lights.len() >= 2,
                "{preset:?}: expected the spawn crystal + at least one more, got {}",
                grid.bake_lights.len()
            );
            // Every light sits just off its crystal: a crystal-coloured
            // voxel must exist within a small cube around it.
            for (i, l) in grid.bake_lights.iter().enumerate() {
                #[allow(clippy::cast_possible_truncation)]
                let lp = IVec3::new(l.pos.x as i32, l.pos.y as i32, l.pos.z as i32);
                let mut found = false;
                'scan: for dz in -6..=6 {
                    for dy in -6..=6 {
                        for dx in -6..=6 {
                            let p = lp + IVec3::new(dx, dy, dz);
                            if grid
                                .voxel_color(p)
                                .is_some_and(|c| c.0 & 0x00ff_ffff == color.0 & 0x00ff_ffff)
                            {
                                found = true;
                                break 'scan;
                            }
                        }
                    }
                }
                assert!(
                    found,
                    "{preset:?}: light {i} at {lp:?} has no crystal voxel nearby"
                );
            }
        }
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
