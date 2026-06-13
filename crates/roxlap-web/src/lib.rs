//! roxlap-web — wasm32 + canvas demo of the renderer.
//!
//! `start()` wires up the canvas + input handlers, parses the
//! embedded oracle world, and kicks off a `requestAnimationFrame`
//! loop. Each frame integrates WASD/Space/Shift movement and the
//! pointer-lock-driven mouse-look, then re-renders + blits.
//!
//! Frame timings are logged to the browser console — `parse`,
//! `render`, and the per-frame `ms` are visible from the
//! devtools.

#![cfg(target_arch = "wasm32")]
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::cell::RefCell;
use std::rc::Rc;

use roxlap_core::{Camera, Engine, OpticastSettings};
use roxlap_render::{Backend, FrameParams, RenderOptions, SceneRenderer};
use roxlap_scene::Scene;
use wasm_bindgen::prelude::*;
use web_sys::{HtmlCanvasElement, KeyboardEvent, MouseEvent};

const XRES: u32 = 640;
const YRES: u32 = 480;

/// GPU marcher vertical field-of-view, degrees → radians at use.
const GPU_FOV_Y_DEG: f32 = 60.0;
/// World render distance (voxels) for both backends.
const SCAN_DIST: i32 = 768;

/// Movement speed in voxlap world-space units per second. Tuned so
/// strafing across the oracle scene feels natural without
/// overshooting the 2048-unit world bounds in a few seconds.
const MOVE_SPEED: f64 = 120.0;
/// Pointer-lock yaw/pitch sensitivity. Pixels of mouse delta map
/// to this many radians; ~0.0025 keeps a 90° turn at ~600 px of
/// mouse travel — a standard FPS feel.
const MOUSE_SENSITIVITY: f64 = 0.0025;
/// Hard cap on pitch so the camera never inverts. Voxlap's
/// scan-loops degenerate for `|pitch| > π/2` (the basis row swap
/// flips the quadrant ordering).
const PITCH_LIMIT: f64 = std::f64::consts::FRAC_PI_2 - 0.05;

/// Heap cell holding the requestAnimationFrame closure. The
/// closure captures a clone of this `Rc` so each invocation can
/// schedule the next frame — the canonical wasm-bindgen RAF
/// pattern.
type RafCell = Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>>;

/// Per-frame, mutable engine state. Kept in a single `Rc<RefCell>`
/// so the RAF closure + input handlers can all reach it.
///
/// GW.2: the demo now renders through the `roxlap-render`
/// [`SceneRenderer`] facade — WebGPU compute marcher when the browser
/// has WebGPU, else the CPU opticast path presented via WebGL2 (the
/// facade owns both). The host keeps only the `Scene` + camera +
/// engine sky/fog; the renderer owns the framebuffer + presentation.
struct State {
    /// Sky + fog source for the per-frame [`FrameParams`].
    engine: Engine,
    /// The voxel world the renderer marches each frame.
    scene: Scene,
    /// Unified CPU/GPU renderer over the canvas.
    renderer: SceneRenderer,
    cam_pos: [f64; 3],
    yaw: f64,
    pitch: f64,
    input: Input,
    last_frame_ms: f64,
    /// R10.X.4: per-frame multi-touch state. Empty most of the
    /// time on desktop; one or two entries while a phone player
    /// holds the canvas.
    touches: Vec<ActiveTouch>,
    /// R10.5: in-flight bench session. `None` when idle; the
    /// 'B' keybind seeds this with a fresh `Bench` and the RAF
    /// loop fills `samples_ms` until it hits `target_frames`,
    /// at which point stats are dumped to the console and the
    /// field is cleared back to `None`.
    bench: Option<Bench>,
}

/// In-flight bench session — captures per-frame `render+pack+
/// blit` cost so the user can compare wasm vs native at the
/// same workload.
struct Bench {
    target_frames: u32,
    samples_ms: Vec<f64>,
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
    /// Accumulated mouse delta since the last frame. Cleared after
    /// every integration. Avoids sampling rate-limiting the camera
    /// when multiple `mousemove` events fire between frames.
    dyaw: f64,
    dpitch: f64,
    /// R10.X.4: virtual-joystick state from the touch input on
    /// the canvas's left half. `Some((dx, dy))` where the
    /// components are in `[-1, 1]` — magnitude scales movement
    /// speed, direction picks WASD-equivalent axes. `None` when
    /// no finger is on the joystick zone.
    joy: Option<(f64, f64)>,
}

/// Active multi-touch tracking — one entry per finger currently
/// touching the canvas. `id` is `Touch.identifier`; `zone`
/// records which half the finger started in (the gesture stays
/// in that zone for its lifetime even if the finger drags into
/// the other half).
#[derive(Debug, Clone, Copy)]
struct ActiveTouch {
    id: i32,
    zone: TouchZone,
    /// Origin in canvas pixel coords (joystick zone) or last
    /// position (look zone). For look it's updated each move so
    /// per-frame deltas are computed against the prior event.
    last: (f64, f64),
    origin: (f64, f64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TouchZone {
    /// Left half: virtual joystick → movement.
    Joy,
    /// Right half: drag → yaw/pitch.
    Look,
}

/// Build a Voxlap-style RH basis from yaw + pitch — same shape as
/// `oracle.c:set_camera_yaw_pitch`. `right × down = forward`
/// keeps the frustum cull on the correct side of every plane.
fn cam_from_yaw_pitch(pos: [f64; 3], yaw: f64, pitch: f64) -> Camera {
    let cy = yaw.cos();
    let sy = yaw.sin();
    let cp = pitch.cos();
    let sp = pitch.sin();
    Camera {
        pos,
        right: [-sy, cy, 0.0],
        down: [-cy * sp, -sy * sp, cp],
        forward: [cy * cp, sy * cp, sp],
    }
}

// Voxlap-packed colours: `(brightness << 24) | (R << 16) | (G << 8) | B`,
// `0x80` brightness = the neutral lit baseline.
const GRASS: u32 = 0x80_4d_8a_3a; // mossy green hilltops
const DIRT: u32 = 0x80_6b_4a_28; // earthy brown
const STONE: u32 = 0x80_7a_7a_82; // cool grey rock

/// Build the demo world: one ground grid of coarse terraced hills
/// plus a few landmark boulders + a pillar so motion reads clearly.
/// Kept compact — a ~512×512 footprint at 8-voxel column resolution
/// (≈4 k bulk `set_rect` calls) — so wasm startup stays well under a
/// second even on a phone. Voxlap is z-down: smaller z is *higher*.
fn build_scene() -> Scene {
    use glam::{DVec3, IVec3};
    use roxlap_scene::GridTransform;

    let mut scene = Scene::new();
    let ground = scene.add_grid(GridTransform::at(DVec3::ZERO));
    let g = scene.grid_mut(ground).expect("ground grid present");

    const EXTENT: i32 = 512; // world footprint along grid-local x and y
    const STEP: i32 = 8; // terrace column size in voxels
    const FLOOR_Z: i32 = 254; // solid fill bottom (just above bedrock)

    for ly in (0..EXTENT).step_by(STEP as usize) {
        for lx in (0..EXTENT).step_by(STEP as usize) {
            // Smooth rolling heightfield oscillating around z≈150.
            let fx = lx as f32 * 0.018;
            let fy = ly as f32 * 0.018;
            let h =
                150.0 - 26.0 * (fx.sin() + (fy * 0.9).cos()) - 12.0 * (fx * 0.5 + fy * 0.7).sin();
            let surface_z = h.round() as i32;
            // High terraces (low z) expose rock; lower ground is grass.
            let col = if surface_z < 126 { STONE } else { GRASS };
            g.set_rect(
                IVec3::new(lx, ly, surface_z),
                IVec3::new(lx + STEP - 1, ly + STEP - 1, FLOOR_Z),
                Some(col),
            );
        }
    }

    // Landmarks.
    g.set_sphere(IVec3::new(120, 160, 150), 22, Some(STONE));
    g.set_sphere(IVec3::new(360, 300, 140), 30, Some(DIRT));
    g.set_rect(
        IVec3::new(250, 250, 70),
        IVec3::new(270, 270, 170),
        Some(STONE),
    );

    scene
}

/// Convert `now_ms` from `performance.now()` into the seconds-of-
/// dt since the last frame, with a 100 ms cap so a tab background
/// pause doesn't slingshot the camera the next frame.
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

    let cy = state.yaw.cos();
    let sy = state.yaw.sin();
    // Move along the camera's *horizontal* forward + right axes —
    // ignore pitch so looking up/down doesn't drag you off the
    // ground. Voxlap's z-axis is up, x/y are the floor plane.
    let mut dx = 0.0;
    let mut dy = 0.0;
    if state.input.forward {
        dx += cy;
        dy += sy;
    }
    if state.input.backward {
        dx -= cy;
        dy -= sy;
    }
    if state.input.right {
        dx += -sy;
        dy += cy;
    }
    if state.input.left {
        dx -= -sy;
        dy -= cy;
    }
    // R10.X.4: virtual-joystick contribution. The touch handler
    // fills `state.input.joy = Some((jx, jy))` with components in
    // `[-1, 1]`; here we add it to the keyboard axes' (dx, dy)
    // *before* normalising so a half-stick deflection moves at
    // half MOVE_SPEED. Joystick y maps to forward (negative on
    // most coordinate systems → forward), x to strafe.
    if let Some((jx, jy)) = state.input.joy {
        // jy is +ve when the finger is pulled toward the bottom
        // of the screen; we treat that as "backward". Same as
        // a flight-stick pull.
        dx += cy * (-jy) + (-sy) * jx;
        dy += sy * (-jy) + cy * jx;
    }
    let mag = (dx * dx + dy * dy).sqrt();
    if mag > 0.0 {
        let scale = MOVE_SPEED * dt / mag.max(1.0);
        state.cam_pos[0] += dx * scale;
        state.cam_pos[1] += dy * scale;
    }
    if state.input.up {
        state.cam_pos[2] -= MOVE_SPEED * dt; // -z is up in voxlap
    }
    if state.input.down {
        state.cam_pos[2] += MOVE_SPEED * dt;
    }
}

fn frame_tick(state_rc: &Rc<RefCell<State>>, perf: &web_sys::Performance, now_ms: f64) {
    let mut state = state_rc.borrow_mut();
    let dt = dt_seconds(state.last_frame_ms, now_ms);
    state.last_frame_ms = now_ms;

    let frame_start_ms = perf.now();

    integrate_input(&mut state, dt);

    // GW.2: march the world + present through the `roxlap-render`
    // facade — WebGPU compute marcher if available, else CPU opticast
    // presented via the facade's own WebGL2 blit. Disjoint `State`
    // fields are split-borrowed so the immutable engine sky/fog read
    // coexists with the mutable scene + renderer.
    {
        let State {
            engine,
            scene,
            renderer,
            cam_pos,
            yaw,
            pitch,
            ..
        } = &mut *state;
        let cam = cam_from_yaw_pitch(*cam_pos, *yaw, *pitch);
        let mut settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
        settings.max_scan_dist = SCAN_DIST;
        settings.mip_levels = 4;
        settings.mip_scan_dist = 64;
        let chunks_visible = (SCAN_DIST.max(1) as u32) / roxlap_scene::CHUNK_SIZE_XY + 4;
        let frame = FrameParams {
            settings: &settings,
            sky_color: engine.sky_color(),
            sky: engine.sky(),
            fog_color: engine.fog_color(),
            fog_max_scan_dist: engine.fog_max_scan_dist(),
            treat_z_max_as_air: true,
            gpu_mip_scan_dist: 64.0,
            gpu_max_outer_steps: chunks_visible,
            gpu_fov_y_rad: GPU_FOV_Y_DEG.to_radians(),
            sprite_lighting: None,
            side_shades: [0; 6],
        };
        renderer.render(scene, &cam, &frame);
        renderer.present();
    }

    // R10.5: if a bench session is in flight, record the work
    // we just did and check whether we've hit the target frame
    // count. The first frame after press is excluded from the
    // sample (RAF cadence + input-handler cold path inflate it).
    let frame_ms = perf.now() - frame_start_ms;
    if let Some(bench) = state.bench.as_mut() {
        bench.samples_ms.push(frame_ms);
        if bench.samples_ms.len() as u32 >= bench.target_frames {
            let report = report_bench(&bench.samples_ms);
            web_sys::console::log_1(&report.into());
            state.bench = None;
        }
    }
}

/// Format min / p50 / mean / p99 / max + fps over `samples_ms`.
/// Mirrors the native `cmd_bench` output shape so devtools-vs-CLI
/// comparison is one-line readable.
fn report_bench(samples_ms: &[f64]) -> String {
    if samples_ms.is_empty() {
        return "roxlap-web bench: no samples".to_string();
    }
    let mut sorted = samples_ms.to_vec();
    // f64 has no Ord; partial_cmp is total under the
    // assumption no NaN slips in (performance.now() doesn't
    // emit NaN).
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let min = sorted[0];
    let max = sorted[n - 1];
    let p50 = sorted[n / 2];
    let p99 = sorted[(n.saturating_sub(1) * 99) / 100];
    let mean = sorted.iter().sum::<f64>() / n as f64;
    let fps = 1000.0 / mean;
    format!(
        "roxlap-web bench: {n} frames | min {min:.2} | p50 {p50:.2} | mean {mean:.2} | p99 {p99:.2} | max {max:.2} ms — {fps:.0} fps"
    )
}

fn request_animation_frame(window: &web_sys::Window, f: &Closure<dyn FnMut(f64)>) {
    let _ = window.request_animation_frame(f.as_ref().unchecked_ref());
}

// R10.X.2: re-export `wasm_bindgen_rayon::init_thread_pool` so the
// macro that generates the JS-side `initThreadPool` shim hooks up
// the worker module. We never call it from JS; instead we await
// it from Rust below.
pub use wasm_bindgen_rayon::init_thread_pool;

/// `#[wasm_bindgen(start)]` auto-runs once trunk's loader-shim
/// `init()` resolves. We schedule an async task that spins up the
/// rayon thread pool (`init_thread_pool(N)` returns a JS
/// `Promise`; we await it via `JsFuture`) and only then runs the
/// real demo init. Doing the dance Rust-side keeps the trunk
/// auto-import path intact — no custom JS bootstrap file.
#[wasm_bindgen(start)]
pub fn auto_start() {
    console_error_panic_hook::set_once();
    let n_threads = navigator_hardware_concurrency();
    web_sys::console::log_1(
        &format!("roxlap-web: spinning up {n_threads} rayon worker(s)…").into(),
    );
    wasm_bindgen_futures::spawn_local(async move {
        let promise = init_thread_pool(n_threads);
        if let Err(e) = wasm_bindgen_futures::JsFuture::from(promise).await {
            web_sys::console::error_2(&"roxlap-web: initThreadPool failed".into(), &e);
            return;
        }
        if let Err(e) = start().await {
            web_sys::console::error_2(&"roxlap-web: start() failed".into(), &e);
        }
    });
}

/// `navigator.hardwareConcurrency` clamped to `[1, 16]` — caps
/// thread spam on hyper-threaded servers / dev machines while
/// staying useful on phones (typically 4–8 cores).
fn navigator_hardware_concurrency() -> usize {
    web_sys::window()
        .as_ref()
        .map(web_sys::Window::navigator)
        .map(|n| n.hardware_concurrency() as usize)
        .unwrap_or(4)
        .clamp(1, 16)
}

/// Demo init body — runs after the rayon thread pool is ready. Async
/// because the `roxlap-render` GPU backend awaits wgpu's WebGPU
/// adapter/device through the browser event loop.
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
    let t_build_start = perf.as_ref().map_or(0.0, web_sys::Performance::now);
    let scene = build_scene();
    let t_build_end = perf.as_ref().map_or(0.0, web_sys::Performance::now);

    // Prefer the WebGPU compute marcher; the facade falls back to the
    // CPU opticast path (presented via WebGL2) when WebGPU is absent.
    let opts = RenderOptions {
        want_gpu: true,
        cpu_max_grid_vsid: 8 * roxlap_scene::CHUNK_SIZE_XY,
        cpu_render_threads: navigator_hardware_concurrency(),
        ..RenderOptions::default()
    };
    let renderer = SceneRenderer::new_from_canvas_async(canvas.clone(), (XRES, YRES), &opts).await;

    let backend = match renderer.backend() {
        Backend::Gpu => "WebGPU",
        Backend::Cpu => "CPU (WebGL2 present)",
    };
    web_sys::console::log_1(
        &format!(
            "roxlap-web: built scene in {:.1} ms — renderer = {backend}{} — controls: WASD move, Space/Shift up/down, click canvas to look around, B to bench",
            t_build_end - t_build_start,
            renderer
                .adapter_info()
                .map(|a| format!(" [{a}]"))
                .unwrap_or_default(),
        )
        .into(),
    );

    let now_ms = perf.as_ref().map_or(0.0, web_sys::Performance::now);
    let state = State {
        engine: Engine::new(),
        scene,
        renderer,
        cam_pos: [256.0, -30.0, 90.0],
        yaw: std::f64::consts::FRAC_PI_2,
        pitch: 0.2,
        input: Input::default(),
        last_frame_ms: now_ms,
        touches: Vec::new(),
        bench: None,
    };
    let state = Rc::new(RefCell::new(state));

    install_input_handlers(&document, &canvas, &state)?;
    spawn_raf_loop(&window, &state);

    Ok(())
}

fn install_input_handlers(
    document: &web_sys::Document,
    canvas: &HtmlCanvasElement,
    state: &Rc<RefCell<State>>,
) -> Result<(), JsValue> {
    // ----- keyboard -----
    let key_state = state.clone();
    let on_keydown = Closure::<dyn FnMut(KeyboardEvent)>::new(move |ev: KeyboardEvent| {
        if let Ok(mut s) = key_state.try_borrow_mut() {
            if set_key(&mut s.input, &ev.code(), true) {
                ev.prevent_default();
                return;
            }
            // R10.5: 'B' starts a 300-frame bench. Render keeps
            // running; the RAF loop accumulates per-frame ms and
            // dumps stats to console once the target is hit.
            // Re-pressing while a session is active is a no-op so
            // we don't lose in-flight samples.
            if ev.code() == "KeyB" && s.bench.is_none() {
                s.bench = Some(Bench {
                    target_frames: 300,
                    samples_ms: Vec::with_capacity(300),
                });
                web_sys::console::log_1(&"roxlap-web bench: starting 300-frame timing run".into());
                ev.prevent_default();
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

    // ----- mouse-look (pointer lock) -----
    // Click the canvas to grab the mouse; mousemove deltas only
    // count while pointerLockElement matches the canvas. Avoids
    // snapping the camera when the user moves the cursor over the
    // page incidentally.
    let canvas_ref = canvas.clone();
    let on_canvas_click = Closure::<dyn FnMut(MouseEvent)>::new(move |_ev: MouseEvent| {
        canvas_ref.request_pointer_lock();
    });
    canvas.add_event_listener_with_callback("click", on_canvas_click.as_ref().unchecked_ref())?;
    on_canvas_click.forget();

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
    Ok(())
}

/// Joystick deadzone in canvas pixels — finger drag past this
/// is full-stick deflection. ~half a virtual joystick radius
/// on a 360-px-tall phone canvas.
const JOY_RADIUS: f64 = 60.0;

/// R10.X.4: touchstart / touchmove / touchend handlers driving
/// the virtual-joystick + look-drag scheme. Left half of canvas
/// = movement joystick, right half = camera look. A finger that
/// touches down in one zone stays in that zone for its lifetime.
fn install_touch_handlers(
    canvas: &HtmlCanvasElement,
    state: &Rc<RefCell<State>>,
) -> Result<(), JsValue> {
    use web_sys::TouchEvent;

    // touchstart: assign each new touch to a zone based on x-
    // coordinate, store origin + last positions.
    let canvas_ref = canvas.clone();
    let state_for_start = state.clone();
    let on_start = Closure::<dyn FnMut(TouchEvent)>::new(move |ev: TouchEvent| {
        ev.prevent_default();
        let rect = canvas_ref.get_bounding_client_rect();
        let scale_x = f64::from(canvas_ref.width()) / rect.width();
        let scale_y = f64::from(canvas_ref.height()) / rect.height();
        let half_w = f64::from(canvas_ref.width()) * 0.5;
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
            });
            if zone == TouchZone::Joy {
                s.input.joy = Some((0.0, 0.0));
            }
        }
    });
    canvas.add_event_listener_with_callback("touchstart", on_start.as_ref().unchecked_ref())?;
    on_start.forget();

    // touchmove: update joystick deflection (joy zone) or
    // accumulate yaw/pitch deltas (look zone).
    let canvas_ref = canvas.clone();
    let state_for_move = state.clone();
    let on_move = Closure::<dyn FnMut(TouchEvent)>::new(move |ev: TouchEvent| {
        ev.prevent_default();
        let rect = canvas_ref.get_bounding_client_rect();
        let scale_x = f64::from(canvas_ref.width()) / rect.width();
        let scale_y = f64::from(canvas_ref.height()) / rect.height();
        let half_w = f64::from(canvas_ref.width()) * 0.5;
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
            // The `half_w` boundary check would let a finger
            // *cross* into the other zone; we don't want that
            // (the gesture stays in its initial zone). half_w
            // is unused here but kept for future zone-edge
            // tweaking.
            let _ = half_w;
        }
    });
    canvas.add_event_listener_with_callback("touchmove", on_move.as_ref().unchecked_ref())?;
    on_move.forget();

    // touchend / touchcancel: drop the matching touches; if any
    // joystick touch ended, clear `input.joy` so movement stops.
    let state_for_end = state.clone();
    let on_end = Closure::<dyn FnMut(TouchEvent)>::new(move |ev: TouchEvent| {
        ev.prevent_default();
        let Ok(mut s) = state_for_end.try_borrow_mut() else {
            return;
        };
        let changed = ev.changed_touches();
        for i in 0..changed.length() {
            let Some(t) = changed.get(i) else { continue };
            let id = t.identifier();
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

/// Map a `KeyboardEvent.code` string onto the booleans of `Input`.
/// Returns `true` if the event was a movement key (so the caller
/// can `preventDefault` to suppress page scroll on Space). Uses
/// `code` not `key` so layout doesn't change which physical
/// position binds — `KeyW` is always W on the keyboard regardless
/// of locale.
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
        _ => false,
    }
}

fn spawn_raf_loop(window: &web_sys::Window, state: &Rc<RefCell<State>>) {
    // Classic wasm-bindgen RAF-self-rearm dance: the closure holds
    // an `Rc` to itself so each invocation can schedule the next
    // frame. `f` is the heap cell, `g` is its clone the closure
    // captures.
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
                    "roxlap-web: frame {frame_count} | mean {mean_ms:.1} ms over last {log_accum_frames}"
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
