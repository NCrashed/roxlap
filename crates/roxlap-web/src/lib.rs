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
use std::io::Read;
use std::rc::Rc;

use flate2::read::GzDecoder;
use roxlap_core::opticast::opticast;
use roxlap_core::rasterizer::ScratchPool;
use roxlap_core::scalar_rasterizer::ScalarRasterizer;
use roxlap_core::{Camera, Engine, OpticastSettings};
use roxlap_formats::vxl;
use wasm_bindgen::prelude::*;
use wasm_bindgen::Clamped;
use web_sys::{HtmlCanvasElement, ImageData, KeyboardEvent, MouseEvent};

const XRES: u32 = 640;
const YRES: u32 = 480;

/// Embedded gzipped oracle world. ~207 KB on disk; ~37 MB in
/// memory after gunzip + parse.
const ORACLE_VXL_GZ: &[u8] = include_bytes!("../../../assets/oracle.vxl.gz");

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
struct State {
    engine: Engine,
    world: vxl::Vxl,
    pool: ScratchPool,
    fb: Vec<u32>,
    zb: Vec<f32>,
    rgba: Vec<u8>,
    cam_pos: [f64; 3],
    yaw: f64,
    pitch: f64,
    input: Input,
    last_frame_ms: f64,
    ctx: web_sys::CanvasRenderingContext2d,
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

fn render_frame(state: &mut State) {
    let sky = state.engine.sky_color();
    state.fb.fill(sky);
    for z in &mut state.zb {
        *z = 0.0;
    }

    let sky_i = i32::from_ne_bytes(sky.to_ne_bytes());
    state.pool.set_skycast(sky_i, 0);
    let fog_i = i32::from_ne_bytes(state.engine.fog_color().to_ne_bytes());
    state.pool.set_fog(fog_i, state.engine.fog_max_scan_dist());

    let cam = cam_from_yaw_pitch(state.cam_pos, state.yaw, state.pitch);
    let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
    let mut rasterizer = ScalarRasterizer::new(
        &mut state.fb,
        &mut state.zb,
        XRES as usize,
        &state.world.data,
        &state.world.column_offset,
        &state.world.mip_base_offsets,
        state.world.vsid,
    );
    let _ = opticast(
        &mut rasterizer,
        &mut state.pool,
        &cam,
        &settings,
        state.world.vsid,
        &state.world.data,
        &state.world.column_offset,
    );
}

fn pack_rgba(framebuffer: &[u32], out: &mut [u8]) {
    debug_assert_eq!(out.len(), framebuffer.len() * 4);
    for (i, &px) in framebuffer.iter().enumerate() {
        let r = ((px >> 16) & 0xff) as u8;
        let g = ((px >> 8) & 0xff) as u8;
        let b = (px & 0xff) as u8;
        out[i * 4] = r;
        out[i * 4 + 1] = g;
        out[i * 4 + 2] = b;
        out[i * 4 + 3] = 0xff;
    }
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
    let mag = (dx * dx + dy * dy).sqrt();
    if mag > 0.0 {
        let scale = MOVE_SPEED * dt / mag;
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

fn frame_tick(state_rc: &Rc<RefCell<State>>, now_ms: f64) {
    let mut state = state_rc.borrow_mut();
    let dt = dt_seconds(state.last_frame_ms, now_ms);
    state.last_frame_ms = now_ms;

    integrate_input(&mut state, dt);
    render_frame(&mut state);

    let fb_ptr = state.fb.as_ptr();
    let fb_len = state.fb.len();
    // SAFETY: `pack_rgba` only reads `fb`. Splitting the borrow
    // through a raw pointer avoids two `&mut state` borrows for
    // `fb` + `rgba` at once.
    let fb_slice = unsafe { std::slice::from_raw_parts(fb_ptr, fb_len) };
    pack_rgba(fb_slice, &mut state.rgba);

    // ImageData.new + putImageData together are ~2 ms at 640×480
    // — small relative to the 10–20 ms render cost. Could be
    // hoisted by reusing one ImageData across frames in R10.X.
    if let Ok(image_data) =
        ImageData::new_with_u8_clamped_array_and_sh(Clamped(&state.rgba), XRES, YRES)
    {
        let _ = state.ctx.put_image_data(&image_data, 0.0, 0.0);
    }
}

fn request_animation_frame(window: &web_sys::Window, f: &Closure<dyn FnMut(f64)>) {
    let _ = window.request_animation_frame(f.as_ref().unchecked_ref());
}

/// wasm-bindgen `start` hook — invoked by the JS shim as soon as
/// the module finishes loading. Sets up the canvas + input
/// listeners + RAF loop, then returns. Returning `Result<(),
/// JsValue>` lets early-init failures bubble up as JS exceptions
/// visible in the browser devtools console.
///
/// # Errors
/// Returns a JS-bridged error if the DOM doesn't have the
/// expected `<canvas id="roxlap-canvas">`, if a 2D rendering
/// context can't be acquired, or if the embedded
/// `oracle.vxl.gz` fails to decompress / parse.
#[wasm_bindgen(start)]
pub fn start() -> Result<(), JsValue> {
    console_error_panic_hook::set_once();

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
    let ctx = canvas
        .get_context("2d")?
        .ok_or_else(|| JsValue::from_str("no 2d context"))?
        .dyn_into::<web_sys::CanvasRenderingContext2d>()
        .map_err(|_| JsValue::from_str("got the wrong context type"))?;

    let perf = window.performance();
    let t_parse_start = perf.as_ref().map_or(0.0, web_sys::Performance::now);
    let mut bytes = Vec::with_capacity(40 * 1024 * 1024);
    GzDecoder::new(ORACLE_VXL_GZ)
        .read_to_end(&mut bytes)
        .map_err(|e| JsValue::from_str(&format!("gunzip oracle.vxl.gz: {e}")))?;
    let world =
        vxl::parse(&bytes).map_err(|e| JsValue::from_str(&format!("parse oracle.vxl: {e:?}")))?;
    let t_parse_end = perf.as_ref().map_or(0.0, web_sys::Performance::now);

    let engine = Engine::new();
    let pool = ScratchPool::new(XRES, YRES, world.vsid);
    let fb = vec![0u32; (XRES * YRES) as usize];
    let zb = vec![0f32; (XRES * YRES) as usize];
    let rgba = vec![0u8; (XRES * YRES * 4) as usize];

    let now_ms = perf.as_ref().map_or(0.0, web_sys::Performance::now);
    let state = State {
        engine,
        world,
        pool,
        fb,
        zb,
        rgba,
        cam_pos: [1024.0, 1024.0, 128.0],
        yaw: std::f64::consts::FRAC_PI_2,
        pitch: 0.0,
        input: Input::default(),
        last_frame_ms: now_ms,
        ctx,
    };
    let state = Rc::new(RefCell::new(state));

    web_sys::console::log_1(
        &format!(
            "roxlap-web: parsed oracle.vxl in {:.1} ms — controls: WASD move, Space/Shift up/down, click canvas to look around",
            t_parse_end - t_parse_start,
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
    // ----- keyboard -----
    let key_state = state.clone();
    let on_keydown = Closure::<dyn FnMut(KeyboardEvent)>::new(move |ev: KeyboardEvent| {
        if let Ok(mut s) = key_state.try_borrow_mut() {
            if set_key(&mut s.input, &ev.code(), true) {
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
    let mut frame_count: u32 = 0;
    let mut log_accum_ms: f64 = 0.0;
    let mut log_accum_frames: u32 = 0;

    *g.borrow_mut() = Some(Closure::<dyn FnMut(f64)>::new(move |now_ms: f64| {
        let t_frame_start = now_ms;
        frame_tick(&state_for_raf, now_ms);
        let t_frame_end = window_for_raf.performance().map_or(now_ms, |p| p.now());
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
