//! roxlap-cave-web — procedural cave demo on wasm32 + canvas.
//!
//! GW.3: rendered through the `roxlap-render` [`SceneRenderer`](roxlap_render::SceneRenderer) facade
//! — the WebGPU compute marcher when the browser has WebGPU, else the
//! CPU DDA path presented via the facade's WebGL2 blit. The cave
//! is generated into a single-chunk `roxlap_scene::Scene` grid;
//! flying, per-voxel collision, and runtime carving all run against
//! the scene. Plasma bullets are facade **sprites** (small glowing
//! voxel spheres); on impact they carve a crater into the scene grid
//! with a local lightmode-1 re-bake, which the facade re-uploads to
//! the GPU via its per-chunk dirty tracking. `F` cycles the preset,
//! `R` reseeds — both regenerate the cave in place.

#![cfg(target_arch = "wasm32")]
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::cell::RefCell;
use std::rc::Rc;

use glam::{DVec3, IVec3};
use roxlap_cavegen::{BlueCaveGenerator, CaveParams, Generator, MagCaveGenerator, MAXZDIM};
use roxlap_core::{Camera, Engine, OpticastSettings};
use roxlap_formats::kv6::Kv6;
use roxlap_formats::sprite::{Sprite, SPRITE_FLAG_NO_SHADING};
use roxlap_formats::vxl;
use roxlap_render::{
    Backend, BackendPreference, FrameParams, RenderOptions, SceneRenderer, SpriteInstanceDesc,
    SpriteSet,
};
use roxlap_scene::{
    CharacterBody, CharacterDef, GridId, GridTransform, MoveMode, Scene, Solidity, WalkInput,
};
use wasm_bindgen::prelude::*;
use web_sys::{HtmlCanvasElement, KeyboardEvent, MouseEvent};

// ----- World / camera tuning (mirrors roxlap-cave-demo) ----------------------

const XRES: u32 = 640;
const YRES: u32 = 512;
const VSID: u32 = 128;

const MOVE_SPEED: f64 = 32.0;
const FAST_MULT: f64 = 4.0;
const MOUSE_SENSITIVITY: f64 = 0.0025;
const PITCH_LIMIT: f64 = std::f64::consts::FRAC_PI_2 - 0.05;
const PLAYER_RADIUS: f64 = 0.3;

const BULLET_MAX_DIST: f64 = 96.0;
const BULLET_VEL: f64 = 60.0;
const FIRE_RADIUS: u32 = 4;
/// Bullet sprite half-extent in voxels (a small glowing sphere).
const BULLET_SPRITE_RADIUS: u32 = 2;
const BULLET_COLOR_CORE: u32 = 0x00FF_4080;

const CARVE_COLOR: u32 = 0x8050_3018;
const SPAWN_BUBBLE_COLOR: u32 = 0x8060_6068;
const SPAWN_BUBBLE_RADIUS: u32 = 6;

const LIGHTMODE: u32 = 1;

const FOG_COLOR: u32 = 0x0090_98B0;
const FOG_MAX_SCAN_DIST: i32 = 128;
/// GPU marcher vertical field-of-view, degrees → radians at use.
const GPU_FOV_Y_DEG: f32 = 70.0;

// ----- Types ----------------------------------------------------------------

type RafCell = Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>>;

#[derive(Debug, Clone, Copy)]
struct Bullet {
    pos: [f64; 3],
    vel: [f64; 3],
    travelled: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Preset {
    Blue,
    Mag,
}

impl Preset {
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
    /// Generate the cave as a single `vsid = VSID` voxel chunk — the
    /// cavegen output is exactly one `roxlap_scene` chunk
    /// (`CHUNK_SIZE_XY = VSID`, `CHUNK_SIZE_Z = MAXZDIM`).
    fn generate(self, seed: u64) -> vxl::Vxl {
        let mut params = self.default_params();
        params.seed = seed;
        match self {
            Self::Blue => BlueCaveGenerator.generate(&params, VSID),
            Self::Mag => MagCaveGenerator.generate(&params, VSID),
        }
    }
}

struct State {
    engine: Engine,
    /// The voxel world (one identity-transform grid, one chunk).
    scene: Scene,
    grid: GridId,
    /// Unified CPU/GPU renderer over the canvas.
    renderer: SceneRenderer,
    /// Glowing bullet sprite model, reused for every live bullet
    /// instance via the facade's [`SpriteSet`].
    bullet_model: Sprite,
    cam_pos: [f64; 3],
    /// CC.3 — the engine controller (fly-mode PLAYER_RADIUS cube)
    /// replaces the wasm copy of the per-axis slide; `cam_pos` stays
    /// the outward-facing eye, synced after each walk.
    body: CharacterBody,
    /// `cam_pos` at our last sync; a mismatch (reseed/preset regen
    /// respawn) re-teleports the body.
    last_eye: [f64; 3],
    yaw: f64,
    pitch: f64,
    input: Input,
    last_frame_ms: f64,
    bullets: Vec<Bullet>,
    preset: Preset,
    seed: u64,
    /// R10.X.4: per-frame multi-touch state. Empty on desktop;
    /// 1-2 entries while a phone player holds the canvas.
    touches: Vec<ActiveTouch>,
}

/// R10.X.4: multi-touch tracking. Each entry covers one
/// finger; `id` is `Touch.identifier`. A finger that touches
/// down in one zone stays in that zone for its lifetime.
#[derive(Debug, Clone, Copy)]
struct ActiveTouch {
    id: i32,
    zone: TouchZone,
    last: (f64, f64),
    origin: (f64, f64),
    /// `performance.now()` at touchstart — used to classify a
    /// short Look-zone touch as a tap (= fire bullet).
    started_ms: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TouchZone {
    /// Left half: virtual joystick → movement.
    Joy,
    /// Right half: drag → yaw/pitch; quick tap → fire bullet.
    Look,
}

#[derive(Default)]
#[allow(clippy::struct_excessive_bools)]
struct Input {
    forward: bool,
    backward: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
    fast: bool,
    /// Pointer-lock-driven yaw/pitch deltas, accumulated since
    /// the last frame integration.
    dyaw: f64,
    dpitch: f64,
    /// R10.X.4: virtual-joystick deflection in `[-1, 1]`. `None`
    /// when no finger is on the joystick zone.
    joy: Option<(f64, f64)>,
    /// R10.X.4: tap-to-fire flag — touchend on the look zone
    /// after a short hold sets this `true`; the next frame's
    /// `step_bullets` consumes it (drains to `false`).
    tap_fire: bool,
}

// ----- World gen + lighting --------------------------------------------------

#[allow(clippy::cast_possible_wrap)]
fn spawn_centre() -> IVec3 {
    IVec3::new((VSID / 2) as i32, (VSID / 2) as i32, MAXZDIM / 2)
}

/// (Re)generate the cave into the grid's single chunk in place, carve
/// the spawn bubble, and bake lightmode-1. Editing in place (rather
/// than building a fresh `Scene`) keeps the same `GridId` so the
/// facade's GPU residency tracker re-uploads the changed chunk instead
/// of going stale.
fn regen_cave(grid: &mut roxlap_scene::Grid, preset: Preset, seed: u64) {
    let vxl = preset.generate(seed);
    *grid.ensure_chunk(IVec3::ZERO) = vxl;
    let c = spawn_centre();
    grid.set_sphere(c, SPAWN_BUBBLE_RADIUS, Some(SPAWN_BUBBLE_COLOR));
    grid.bake(roxlap_scene::BakeMode::Directional);
    // `set_sphere` bumped the version; bump once more so a re-gen with
    // an identical spawn-bubble edit still differs from the tracker.
    grid.bump_chunk_version(IVec3::ZERO);
}

/// Build the bullet sprite model: a small solid glowing sphere. The
/// `NO_SHADING` flag makes it emissive (uniform bright colour) so it
/// reads as plasma rather than a lit rock.
fn build_bullet_model() -> Sprite {
    let d = BULLET_SPRITE_RADIUS * 2 + 1;
    let r = BULLET_SPRITE_RADIUS as i32;
    let kv6 = Kv6::from_fn(d, d, d, |x, y, z| {
        let dx = x as i32 - r;
        let dy = y as i32 - r;
        let dz = z as i32 - r;
        if dx * dx + dy * dy + dz * dz <= r * r {
            Some(BULLET_COLOR_CORE)
        } else {
            None
        }
    });
    let mut sprite = Sprite::axis_aligned(kv6, [0.0, 0.0, 0.0]);
    sprite.flags = SPRITE_FLAG_NO_SHADING;
    sprite
}

// ----- Camera + collision ---------------------------------------------------

fn cam_from_yaw_pitch(pos: [f64; 3], yaw: f64, pitch: f64) -> Camera {
    Camera::from_yaw_pitch(pos, yaw, pitch)
}

fn dt_seconds(prev_ms: f64, now_ms: f64) -> f64 {
    let dt_ms = (now_ms - prev_ms).clamp(0.0, 100.0);
    dt_ms / 1000.0
}

fn integrate_input(state: &mut State, dt: f64) {
    state.yaw += state.input.dyaw * MOUSE_SENSITIVITY;
    state.pitch =
        (state.pitch + state.input.dpitch * MOUSE_SENSITIVITY).clamp(-PITCH_LIMIT, PITCH_LIMIT);
    state.input.dyaw = 0.0;
    state.input.dpitch = 0.0;

    let cam = cam_from_yaw_pitch(state.cam_pos, state.yaw, state.pitch);
    let mut delta = [0.0f64; 3];
    if state.input.forward {
        for (d, &c) in delta.iter_mut().zip(cam.forward.iter()) {
            *d += c;
        }
    }
    if state.input.backward {
        for (d, &c) in delta.iter_mut().zip(cam.forward.iter()) {
            *d -= c;
        }
    }
    if state.input.right {
        for (d, &c) in delta.iter_mut().zip(cam.right.iter()) {
            *d += c;
        }
    }
    if state.input.left {
        for (d, &c) in delta.iter_mut().zip(cam.right.iter()) {
            *d -= c;
        }
    }
    if state.input.up {
        delta[2] -= 1.0;
    }
    if state.input.down {
        delta[2] += 1.0;
    }
    // R10.X.4: virtual-joystick deflection on the canvas's
    // left half. jy points "up" → forward; jx points right →
    // strafe right. We add the unnormalised contribution so a
    // half-stick deflection moves at half MOVE_SPEED.
    if let Some((jx, jy)) = state.input.joy {
        for (d, &c) in delta.iter_mut().zip(cam.forward.iter()) {
            *d += c * (-jy);
        }
        for (d, &c) in delta.iter_mut().zip(cam.right.iter()) {
            *d += c * jx;
        }
    }
    let speed = if state.input.fast {
        MOVE_SPEED * FAST_MULT
    } else {
        MOVE_SPEED
    };
    // CC.3: the engine controller slides the cube body. The wish is
    // passed UNnormalised (walk clamps length to 1), which finally
    // honours the joystick comment above: half-stick = half speed.
    // NO early return on idle — walk() must run every frame so the
    // wish-zero target stops the body (stale velocity otherwise).
    let wish = glam::DVec3::new(delta[0], delta[1], delta[2]);
    if state.cam_pos != state.last_eye {
        state
            .body
            .teleport(glam::DVec3::from(state.cam_pos) + glam::DVec3::new(0.0, 0.0, PLAYER_RADIUS));
    }
    state.body.def_mut().fly_speed = speed;
    state
        .body
        .walk(&state.scene, dt, WalkInput { wish, jump: false });

    // Out-of-world was SOLID in the old probe; the engine says air,
    // so the cave bounds become an explicit feet clamp (velocity
    // survives — boundary sliding stays fast).
    let mut feet = state.body.pos();
    let hi_xy = f64::from(VSID) - PLAYER_RADIUS - 1e-3;
    let hi_z = f64::from(MAXZDIM) - 1e-3;
    feet.x = feet.x.clamp(PLAYER_RADIUS, hi_xy);
    feet.y = feet.y.clamp(PLAYER_RADIUS, hi_xy);
    feet.z = feet.z.clamp(2.0 * PLAYER_RADIUS, hi_z);
    if feet != state.body.pos() {
        state.body.set_pos(feet);
    }

    state.cam_pos = state.body.eye_pos().into();
    state.last_eye = state.cam_pos;
}

// ----- Bullets --------------------------------------------------------------

fn fire_bullet(state: &mut State) {
    let cam = cam_from_yaw_pitch(state.cam_pos, state.yaw, state.pitch);
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
    state.bullets.push(Bullet {
        pos,
        vel,
        travelled: 0.0,
    });
}

/// Advance bullets, carve craters on impact, and re-bake the cave's
/// lighting once if anything was carved. Returns `true` if the grid
/// was edited (so the caller knows the facade will re-upload).
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn step_bullets(state: &mut State, dt: f64) -> bool {
    if dt <= 0.0 {
        return false;
    }
    let grid = state.scene.grid(state.grid).expect("cave grid present");
    let vsid = VSID as i32;
    let mut impacts: Vec<IVec3> = Vec::new();
    state.bullets.retain_mut(|b| {
        let dx = b.vel[0] * dt;
        let dy = b.vel[1] * dt;
        let dz = b.vel[2] * dt;
        b.pos[0] += dx;
        b.pos[1] += dy;
        b.pos[2] += dz;
        b.travelled += (dx * dx + dy * dy + dz * dz).sqrt();
        if b.travelled > BULLET_MAX_DIST {
            return false;
        }
        let vx = b.pos[0].floor() as i32;
        let vy = b.pos[1].floor() as i32;
        let vz = b.pos[2].floor() as i32;
        if vx < 0 || vy < 0 || vx >= vsid || vy >= vsid || !(0..MAXZDIM).contains(&vz) {
            return false;
        }
        if grid.voxel_solid(IVec3::new(vx, vy, vz)) {
            impacts.push(IVec3::new(vx, vy, vz));
            return false;
        }
        true
    });
    if impacts.is_empty() {
        return false;
    }
    let grid = state.scene.grid_mut(state.grid).expect("cave grid present");
    for hit in impacts {
        grid.set_sphere(hit, FIRE_RADIUS, Some(CARVE_COLOR));
    }
    // Re-bake the (single) chunk's lighting once after all craters so
    // estnorm shading follows the new cavity walls; the carve already
    // bumped the chunk version, so the facade re-uploads this frame.
    grid.bake(roxlap_scene::BakeMode::Directional);
    true
}

// ----- Render ---------------------------------------------------------------

/// Rebuild the per-frame sprite set from the live bullets and march +
/// present the scene through the facade.
fn render(state: &mut State) {
    // Bullets → facade sprite instances (all share the one model).
    let instances: Vec<SpriteInstanceDesc> = state
        .bullets
        .iter()
        .map(|b| SpriteInstanceDesc {
            model: 0,
            pos: [b.pos[0] as f32, b.pos[1] as f32, b.pos[2] as f32],
        })
        .collect();
    let set = SpriteSet {
        models: vec![state.bullet_model.clone()],
        instances,
        carve_model: None,
    };
    state.renderer.set_sprites(&set);

    // `frame` borrows `state.engine` immutably; the render call below
    // mutably borrows the disjoint `state.scene` + `state.renderer`
    // fields, so NLL lets them coexist.
    let cam = cam_from_yaw_pitch(state.cam_pos, state.yaw, state.pitch);
    let mut settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
    settings.max_scan_dist = MAXZDIM;
    // QE.2 — `FrameParams::new` + overrides; the GPU projection derives
    // from `settings`, so the deliberate 70° FOV is set there (for
    // both backends).
    let settings = settings.with_fov_y(GPU_FOV_Y_DEG.to_radians());
    let mut frame = FrameParams::new(&settings);
    frame.sky_color = state.engine.sky_color();
    frame.sky = state.engine.sky();
    frame.fog_color = state.engine.fog_color();
    frame.fog_max_scan_dist = state.engine.fog_max_scan_dist();
    state.renderer.render(&mut state.scene, &cam, &frame);
    state.renderer.present();
}

fn frame_tick(state_rc: &Rc<RefCell<State>>, _perf: &web_sys::Performance, now_ms: f64) {
    let mut state = state_rc.borrow_mut();
    let dt = dt_seconds(state.last_frame_ms, now_ms);
    state.last_frame_ms = now_ms;

    integrate_input(&mut state, dt);
    // R10.X.4: tap-to-fire — touchend on the look zone after a
    // short hold sets `tap_fire`; consume here so a tap can't
    // double-fire across frames.
    if state.input.tap_fire {
        state.input.tap_fire = false;
        fire_bullet(&mut state);
    }
    let _carved = step_bullets(&mut state, dt);
    render(&mut state);
}

// ----- Regenerate -----------------------------------------------------------

fn regenerate(state: &mut State) {
    let preset = state.preset;
    let seed = state.seed;
    let grid = state.scene.grid_mut(state.grid).expect("cave grid present");
    regen_cave(grid, preset, seed);
    state.cam_pos = [
        f64::from(VSID) * 0.5,
        f64::from(VSID) * 0.5,
        f64::from(MAXZDIM) * 0.5,
    ];
    state.bullets.clear();
}

// ----- Init -----------------------------------------------------------------

// R10.X.2: re-export `wasm_bindgen_rayon::init_thread_pool` so the
// generator macro hooks up the JS-side worker module.
pub use wasm_bindgen_rayon::init_thread_pool;

/// `#[wasm_bindgen(start)]` auto-runs after trunk's `init()`. We
/// schedule an async task that spins up the rayon thread pool
/// (Promise-returning `init_thread_pool` awaited via JsFuture)
/// before running the demo. Same pattern as `roxlap-web`'s
/// `auto_start`.
#[wasm_bindgen(start)]
pub fn auto_start() {
    console_error_panic_hook::set_once();
    let n_threads = navigator_hardware_concurrency();
    web_sys::console::log_1(
        &format!("roxlap-cave-web: spinning up {n_threads} rayon worker(s)…").into(),
    );
    wasm_bindgen_futures::spawn_local(async move {
        let promise = init_thread_pool(n_threads);
        if let Err(e) = wasm_bindgen_futures::JsFuture::from(promise).await {
            web_sys::console::error_2(&"roxlap-cave-web: initThreadPool failed".into(), &e);
            return;
        }
        if let Err(e) = start().await {
            web_sys::console::error_2(&"roxlap-cave-web: start() failed".into(), &e);
        }
    });
}

fn navigator_hardware_concurrency() -> usize {
    web_sys::window()
        .as_ref()
        .map(web_sys::Window::navigator)
        .map(|n| n.hardware_concurrency() as usize)
        .unwrap_or(4)
        .clamp(1, 16)
}

/// Demo init — runs after the rayon thread pool is ready. Async
/// because the facade's GPU backend awaits WebGPU through the event
/// loop.
///
/// # Errors
/// Returns a JS-bridged error if the DOM doesn't have the expected
/// `<canvas id="roxlap-canvas">`.
async fn start() -> Result<(), JsValue> {
    let window = web_sys::window().ok_or_else(|| JsValue::from_str("no window"))?;
    let document = window
        .document()
        .ok_or_else(|| JsValue::from_str("no document"))?;
    let canvas: HtmlCanvasElement = document
        .get_element_by_id("roxlap-canvas")
        .ok_or_else(|| JsValue::from_str("no #roxlap-canvas element"))?
        .dyn_into::<HtmlCanvasElement>()
        .map_err(|_| JsValue::from_str("#roxlap-canvas is not a <canvas>"))?;
    canvas.set_width(XRES);
    canvas.set_height(YRES);

    let perf = window.performance();
    let mut engine = Engine::new();
    engine.set_fog(FOG_COLOR, FOG_MAX_SCAN_DIST);
    engine.set_lightmode(LIGHTMODE);

    let preset = Preset::Blue;
    let seed = preset.default_params().seed;
    let t_gen_start = perf.as_ref().map_or(0.0, web_sys::Performance::now);
    let mut scene = Scene::new();
    let grid_id = scene.add_grid(GridTransform::at(DVec3::ZERO));
    regen_cave(
        scene.grid_mut(grid_id).expect("cave grid present"),
        preset,
        seed,
    );
    let t_gen_end = perf.as_ref().map_or(0.0, web_sys::Performance::now);

    let opts = RenderOptions {
        backend: BackendPreference::PreferGpu,
        ..RenderOptions::default()
    };
    let renderer = SceneRenderer::new_from_canvas_async(canvas.clone(), (XRES, YRES), &opts).await;
    let backend = match renderer.backend() {
        Backend::Gpu => "WebGPU",
        Backend::Cpu => "CPU (WebGL2 present)",
    };

    let now_ms = perf.as_ref().map_or(0.0, web_sys::Performance::now);
    let state = State {
        engine,
        scene,
        grid: grid_id,
        renderer,
        bullet_model: build_bullet_model(),
        cam_pos: [
            f64::from(VSID) * 0.5,
            f64::from(VSID) * 0.5,
            f64::from(MAXZDIM) * 0.5,
        ],
        body: {
            let mut body = CharacterBody::new(CharacterDef {
                radius: PLAYER_RADIUS,
                height: 2.0 * PLAYER_RADIUS,
                eye_height: PLAYER_RADIUS,
                fly_speed: MOVE_SPEED,
                solidity: Solidity {
                    bedrock_blocks: true,
                },
                ..CharacterDef::default()
            });
            body.set_mode(MoveMode::Fly);
            body
        },
        last_eye: [f64::NAN; 3],
        yaw: 0.0,
        pitch: 0.0,
        input: Input::default(),
        last_frame_ms: now_ms,
        bullets: Vec::new(),
        preset,
        seed,
        touches: Vec::new(),
    };
    let state = Rc::new(RefCell::new(state));

    web_sys::console::log_1(
        &format!(
            "roxlap-cave-web: cave-gen + bake {:.0} ms — renderer = {backend}{} — controls: WASD move, Space/Shift up/down, Ctrl fast, click canvas to look around, click again to fire, F preset, R reseed",
            t_gen_end - t_gen_start,
            state
                .borrow()
                .renderer
                .adapter_info()
                .map(|a| format!(" [{a}]"))
                .unwrap_or_default(),
        )
        .into(),
    );

    install_input_handlers(&document, &canvas, &state)?;
    spawn_raf_loop(&window, &state);
    Ok(())
}

fn install_input_handlers(
    document: &web_sys::Document,
    canvas: &HtmlCanvasElement,
    state: &Rc<RefCell<State>>,
) -> Result<(), JsValue> {
    // Keyboard
    let key_state = state.clone();
    let on_keydown = Closure::<dyn FnMut(KeyboardEvent)>::new(move |ev: KeyboardEvent| {
        if let Ok(mut s) = key_state.try_borrow_mut() {
            let code = ev.code();
            if set_key(&mut s.input, &code, true) {
                ev.prevent_default();
                return;
            }
            // Single-press actions.
            match code.as_str() {
                "KeyF" => {
                    s.preset = s.preset.next();
                    regenerate(&mut s);
                    ev.prevent_default();
                }
                "KeyR" => {
                    s.seed = s.seed.wrapping_add(1);
                    regenerate(&mut s);
                    ev.prevent_default();
                }
                _ => {}
            }
        }
    });
    document.add_event_listener_with_callback("keydown", on_keydown.as_ref().unchecked_ref())?;
    on_keydown.forget();

    let key_state = state.clone();
    let on_keyup = Closure::<dyn FnMut(KeyboardEvent)>::new(move |ev: KeyboardEvent| {
        if let Ok(mut s) = key_state.try_borrow_mut() {
            if set_key(&mut s.input, &ev.code(), false) {
                ev.prevent_default();
            }
        }
    });
    document.add_event_listener_with_callback("keyup", on_keyup.as_ref().unchecked_ref())?;
    on_keyup.forget();

    // Click — first time grabs pointer lock, subsequent clicks fire bullets.
    let click_state = state.clone();
    let canvas_for_click = canvas.clone();
    let on_canvas_click = Closure::<dyn FnMut(MouseEvent)>::new(move |_ev: MouseEvent| {
        let Some(doc) = canvas_for_click.owner_document() else {
            return;
        };
        let locked = doc.pointer_lock_element().as_ref() == Some(canvas_for_click.unchecked_ref());
        if locked {
            if let Ok(mut s) = click_state.try_borrow_mut() {
                fire_bullet(&mut s);
            }
        } else {
            canvas_for_click.request_pointer_lock();
        }
    });
    canvas.add_event_listener_with_callback("click", on_canvas_click.as_ref().unchecked_ref())?;
    on_canvas_click.forget();

    // Mouse-move (yaw/pitch only while pointer-locked)
    let mouse_state = state.clone();
    let canvas_for_check = canvas.clone();
    let on_mousemove = Closure::<dyn FnMut(MouseEvent)>::new(move |ev: MouseEvent| {
        let Some(doc) = canvas_for_check.owner_document() else {
            return;
        };
        if doc.pointer_lock_element().as_ref() != Some(canvas_for_check.unchecked_ref()) {
            return;
        }
        if let Ok(mut s) = mouse_state.try_borrow_mut() {
            s.input.dyaw += f64::from(ev.movement_x());
            s.input.dpitch += f64::from(ev.movement_y());
        }
    });
    document
        .add_event_listener_with_callback("mousemove", on_mousemove.as_ref().unchecked_ref())?;
    on_mousemove.forget();

    install_touch_handlers(canvas, state)?;
    install_button_handlers(document, state)?;
    Ok(())
}

/// R10.X.4: virtual-joystick deadzone in canvas pixels.
const JOY_RADIUS: f64 = 60.0;

/// R10.X.4: a Look-zone touch that ends within this many ms
/// without dragging more than `TAP_MAX_DRAG_PX` is treated as
/// a tap → fire bullet.
const TAP_MAX_DURATION_MS: f64 = 250.0;
const TAP_MAX_DRAG_PX: f64 = 16.0;

#[allow(clippy::too_many_lines)] // straight-line touch wiring; splitting hurts readability
fn install_touch_handlers(
    canvas: &HtmlCanvasElement,
    state: &Rc<RefCell<State>>,
) -> Result<(), JsValue> {
    use web_sys::TouchEvent;

    let canvas_ref = canvas.clone();
    let state_for_start = state.clone();
    let on_start = Closure::<dyn FnMut(TouchEvent)>::new(move |ev: TouchEvent| {
        ev.prevent_default();
        let rect = canvas_ref.get_bounding_client_rect();
        let scale_x = f64::from(canvas_ref.width()) / rect.width();
        let scale_y = f64::from(canvas_ref.height()) / rect.height();
        let half_w = f64::from(canvas_ref.width()) * 0.5;
        let now_ms = now_perf();
        let Ok(mut s) = state_for_start.try_borrow_mut() else {
            return;
        };
        let changed = ev.changed_touches();
        for i in 0..changed.length() {
            let Some(t) = changed.get(i) else { continue };
            let cx = (f64::from(t.client_x()) - rect.left()) * scale_x;
            let cy = (f64::from(t.client_y()) - rect.top()) * scale_y;
            let zone = if cx < half_w {
                TouchZone::Joy
            } else {
                TouchZone::Look
            };
            s.touches.push(ActiveTouch {
                id: t.identifier(),
                zone,
                last: (cx, cy),
                origin: (cx, cy),
                started_ms: now_ms,
            });
            if zone == TouchZone::Joy {
                s.input.joy = Some((0.0, 0.0));
            }
        }
    });
    canvas.add_event_listener_with_callback("touchstart", on_start.as_ref().unchecked_ref())?;
    on_start.forget();

    let canvas_ref = canvas.clone();
    let state_for_move = state.clone();
    let on_move = Closure::<dyn FnMut(TouchEvent)>::new(move |ev: TouchEvent| {
        ev.prevent_default();
        let rect = canvas_ref.get_bounding_client_rect();
        let scale_x = f64::from(canvas_ref.width()) / rect.width();
        let scale_y = f64::from(canvas_ref.height()) / rect.height();
        let Ok(mut s) = state_for_move.try_borrow_mut() else {
            return;
        };
        let changed = ev.changed_touches();
        for i in 0..changed.length() {
            let Some(t) = changed.get(i) else { continue };
            let id = t.identifier();
            let cx = (f64::from(t.client_x()) - rect.left()) * scale_x;
            let cy = (f64::from(t.client_y()) - rect.top()) * scale_y;
            let Some(active) = s.touches.iter_mut().find(|a| a.id == id) else {
                continue;
            };
            let (last_x, last_y) = active.last;
            let (origin_x, origin_y) = active.origin;
            active.last = (cx, cy);
            match active.zone {
                TouchZone::Joy => {
                    let jx = ((cx - origin_x) / JOY_RADIUS).clamp(-1.0, 1.0);
                    let jy = ((cy - origin_y) / JOY_RADIUS).clamp(-1.0, 1.0);
                    s.input.joy = Some((jx, jy));
                }
                TouchZone::Look => {
                    s.input.dyaw += cx - last_x;
                    s.input.dpitch += cy - last_y;
                }
            }
        }
    });
    canvas.add_event_listener_with_callback("touchmove", on_move.as_ref().unchecked_ref())?;
    on_move.forget();

    let state_for_end = state.clone();
    let on_end = Closure::<dyn FnMut(TouchEvent)>::new(move |ev: TouchEvent| {
        ev.prevent_default();
        let now_ms = now_perf();
        let Ok(mut s) = state_for_end.try_borrow_mut() else {
            return;
        };
        let changed = ev.changed_touches();
        for i in 0..changed.length() {
            let Some(t) = changed.get(i) else { continue };
            let id = t.identifier();
            // Tap-to-fire: a Look-zone touch that ended quickly
            // and didn't drag far is a tap.
            if let Some(active) = s.touches.iter().find(|a| a.id == id).copied() {
                if active.zone == TouchZone::Look {
                    let duration = now_ms - active.started_ms;
                    let (lx, ly) = active.last;
                    let (ox, oy) = active.origin;
                    let drag = ((lx - ox).powi(2) + (ly - oy).powi(2)).sqrt();
                    if duration <= TAP_MAX_DURATION_MS && drag <= TAP_MAX_DRAG_PX {
                        s.input.tap_fire = true;
                    }
                }
            }
            s.touches.retain(|a| a.id != id);
        }
        if !s.touches.iter().any(|a| a.zone == TouchZone::Joy) {
            s.input.joy = None;
        }
    });
    canvas.add_event_listener_with_callback("touchend", on_end.as_ref().unchecked_ref())?;
    canvas.add_event_listener_with_callback("touchcancel", on_end.as_ref().unchecked_ref())?;
    on_end.forget();

    Ok(())
}

/// `performance.now()` lookup; returns `0.0` if `Performance`
/// isn't available (vanishingly rare in modern browsers).
fn now_perf() -> f64 {
    web_sys::window()
        .and_then(|w| w.performance())
        .map_or(0.0, |p| p.now())
}

/// On-screen action buttons for mobile — `#fire-btn` mirrors
/// `tap_fire`, `#preset-btn` cycles preset (mirror of F),
/// `#seed-btn` advances seed (mirror of R). Buttons live in
/// `index.html`; here we wire the click handlers.
fn install_button_handlers(
    document: &web_sys::Document,
    state: &Rc<RefCell<State>>,
) -> Result<(), JsValue> {
    let bind = |id: &str, action: fn(&mut State)| -> Result<(), JsValue> {
        let Some(el) = document.get_element_by_id(id) else {
            return Ok(()); // button missing in HTML — silent no-op
        };
        let target = el.dyn_into::<web_sys::HtmlElement>()?;
        let state_for_btn = state.clone();
        let on_click =
            Closure::<dyn FnMut(web_sys::MouseEvent)>::new(move |_ev: web_sys::MouseEvent| {
                if let Ok(mut s) = state_for_btn.try_borrow_mut() {
                    action(&mut s);
                }
            });
        target.add_event_listener_with_callback("click", on_click.as_ref().unchecked_ref())?;
        on_click.forget();
        Ok(())
    };
    bind("fire-btn", |s| {
        s.input.tap_fire = true;
    })?;
    bind("preset-btn", |s| {
        s.preset = s.preset.next();
        regenerate(s);
    })?;
    bind("seed-btn", |s| {
        s.seed = s.seed.wrapping_add(1);
        regenerate(s);
    })?;
    Ok(())
}

fn set_key(input: &mut Input, code: &str, down: bool) -> bool {
    match code {
        "KeyW" | "ArrowUp" => {
            input.forward = down;
            true
        }
        "KeyS" | "ArrowDown" => {
            input.backward = down;
            true
        }
        "KeyA" | "ArrowLeft" => {
            input.left = down;
            true
        }
        "KeyD" | "ArrowRight" => {
            input.right = down;
            true
        }
        "Space" => {
            input.up = down;
            true
        }
        "ShiftLeft" | "ShiftRight" => {
            input.down = down;
            true
        }
        "ControlLeft" | "ControlRight" => {
            input.fast = down;
            true
        }
        _ => false,
    }
}

fn spawn_raf_loop(window: &web_sys::Window, state: &Rc<RefCell<State>>) {
    let f: RafCell = Rc::new(RefCell::new(None));
    let g = f.clone();
    let state_for_raf = state.clone();
    let window_for_raf = window.clone();
    let perf = window
        .performance()
        .expect("performance API on Window — required for RAF + bench timing");
    let mut frame_count: u32 = 0;
    let mut log_accum_ms: f64 = 0.0;
    let mut log_accum_frames: u32 = 0;

    *g.borrow_mut() = Some(Closure::<dyn FnMut(f64)>::new(move |now_ms: f64| {
        let t_frame_start = now_ms;
        frame_tick(&state_for_raf, &perf, now_ms);
        let t_frame_end = perf.now();
        let frame_ms = t_frame_end - t_frame_start;
        log_accum_ms += frame_ms;
        log_accum_frames += 1;
        frame_count += 1;
        if log_accum_frames >= 60 {
            let mean_ms = log_accum_ms / f64::from(log_accum_frames);
            web_sys::console::log_1(
                &format!(
                    "roxlap-cave-web: frame {frame_count} | mean {mean_ms:.1} ms over last {log_accum_frames}"
                )
                .into(),
            );
            log_accum_ms = 0.0;
            log_accum_frames = 0;
        }
        request_animation_frame(&window_for_raf, f.borrow().as_ref().unwrap());
    }));

    request_animation_frame(window, g.borrow().as_ref().unwrap());
}

fn request_animation_frame(window: &web_sys::Window, f: &Closure<dyn FnMut(f64)>) {
    let _ = window.request_animation_frame(f.as_ref().unchecked_ref());
}
