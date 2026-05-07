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
use web_sys::{HtmlCanvasElement, KeyboardEvent, MouseEvent, WebGl2RenderingContext as Gl};

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
    cam_pos: [f64; 3],
    yaw: f64,
    pitch: f64,
    input: Input,
    last_frame_ms: f64,
    blit: WebGlBlit,
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

/// R10.X.3: GPU-side framebuffer presenter. The voxlap renderer
/// produces a `Vec<u32>` in BGRA byte order
/// (`(brightness << 24) | (R << 16) | (G << 8) | B`); the
/// fragment shader swizzles `.bgr` and forces alpha = 1.0 so
/// the canvas compositor doesn't darken pixels via voxlap's
/// brightness byte. One texture, one shader program, one
/// vertex array — all created at init and reused per frame.
struct WebGlBlit {
    gl: Gl,
    texture: web_sys::WebGlTexture,
    /// Width × height of the bound texture, in voxlap-renderer
    /// pixels. The render targets a fixed 640×480; if the
    /// canvas's CSS-driven viewport is larger or smaller, the
    /// fullscreen quad scales (the texture is sampled with
    /// linear filtering, so resizing reads cleanly).
    width: u32,
    height: u32,
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

// ----- WebGL blit -----------------------------------------------------------

/// Vertex shader for the fullscreen-quad presenter. Two
/// triangles covering NDC `[-1, 1]²`; per-vertex `a_uv`
/// supplies the texture coordinates (flipped vertically so
/// voxlap's top-down framebuffer maps to screen-top correctly
/// — texture coord (0, 0) is bottom-left in WebGL).
const QUAD_VS: &str = "#version 300 es
in vec2 a_pos;
in vec2 a_uv;
out vec2 v_uv;
void main() {
    v_uv = a_uv;
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
";

/// Fragment shader: samples the framebuffer texture and
/// swizzles `.bgr` so voxlap's `(B, G, R, A)` byte order
/// (uploaded as `RGBA8`) reads back as the correct RGB. Alpha
/// is forced to `1.0` so the canvas compositor doesn't darken
/// pixels via voxlap's brightness byte (which lives in `.a`
/// after the upload).
const QUAD_FS: &str = "#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_tex;
out vec4 frag;
void main() {
    vec4 c = texture(u_tex, v_uv);
    frag = vec4(c.bgr, 1.0);
}
";

// WebGL's API takes `i32` for sizes / locations and `u32` /
// signed mixed bit-flags for enum constants — the cast warnings
// fire at every call site. The values are bounded by canvas
// pixel dimensions and never wrap in practice.
#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
impl WebGlBlit {
    /// Build a blit context for `canvas`. Allocates the texture,
    /// compiles the shader program, uploads a single fullscreen
    /// quad VAO. All resources are bound once and reused per
    /// frame; `present` is the only per-frame call.
    fn new(canvas: &HtmlCanvasElement, width: u32, height: u32) -> Result<Self, JsValue> {
        let gl: Gl = canvas
            .get_context("webgl2")?
            .ok_or_else(|| JsValue::from_str("no webgl2 context"))?
            .dyn_into::<Gl>()
            .map_err(|_| JsValue::from_str("got the wrong webgl context type"))?;

        let program = compile_program(&gl, QUAD_VS, QUAD_FS)?;
        gl.use_program(Some(&program));

        // Fullscreen quad: pos.xy + uv.xy interleaved.
        // Voxlap is top-down; WebGL UV (0,0) is bottom-left, so we
        // flip the V coord to get the engine's first row at the top.
        #[rustfmt::skip]
        let quad: [f32; 16] = [
            -1.0, -1.0, 0.0, 1.0,
             1.0, -1.0, 1.0, 1.0,
            -1.0,  1.0, 0.0, 0.0,
             1.0,  1.0, 1.0, 0.0,
        ];
        let vao = gl
            .create_vertex_array()
            .ok_or_else(|| JsValue::from_str("create_vertex_array failed"))?;
        gl.bind_vertex_array(Some(&vao));

        let vbo = gl
            .create_buffer()
            .ok_or_else(|| JsValue::from_str("create_buffer failed"))?;
        gl.bind_buffer(Gl::ARRAY_BUFFER, Some(&vbo));
        unsafe {
            // SAFETY: js_sys::Float32Array::view borrows the wasm
            // linear-memory backing of `quad` for the duration of
            // the call — must not allocate / grow memory while the
            // view is live. `bufferData_with_array_buffer_view`
            // copies into the GPU before returning.
            let view = js_sys::Float32Array::view(&quad);
            gl.buffer_data_with_array_buffer_view(Gl::ARRAY_BUFFER, &view, Gl::STATIC_DRAW);
        }
        let stride = (4 * std::mem::size_of::<f32>()) as i32;
        let pos_loc = gl.get_attrib_location(&program, "a_pos");
        let uv_loc = gl.get_attrib_location(&program, "a_uv");
        if pos_loc < 0 || uv_loc < 0 {
            return Err(JsValue::from_str("attribute lookup failed"));
        }
        let pos_loc_u = pos_loc as u32;
        let uv_loc_u = uv_loc as u32;
        gl.enable_vertex_attrib_array(pos_loc_u);
        gl.vertex_attrib_pointer_with_i32(pos_loc_u, 2, Gl::FLOAT, false, stride, 0);
        gl.enable_vertex_attrib_array(uv_loc_u);
        gl.vertex_attrib_pointer_with_i32(
            uv_loc_u,
            2,
            Gl::FLOAT,
            false,
            stride,
            (2 * std::mem::size_of::<f32>()) as i32,
        );

        // RGBA8 texture sized to the engine framebuffer. Linear
        // filtering keeps it readable when the canvas's CSS-driven
        // viewport scales it up or down.
        let texture = gl
            .create_texture()
            .ok_or_else(|| JsValue::from_str("create_texture failed"))?;
        gl.active_texture(Gl::TEXTURE0);
        gl.bind_texture(Gl::TEXTURE_2D, Some(&texture));
        gl.tex_parameteri(Gl::TEXTURE_2D, Gl::TEXTURE_MIN_FILTER, Gl::LINEAR as i32);
        gl.tex_parameteri(Gl::TEXTURE_2D, Gl::TEXTURE_MAG_FILTER, Gl::LINEAR as i32);
        gl.tex_parameteri(Gl::TEXTURE_2D, Gl::TEXTURE_WRAP_S, Gl::CLAMP_TO_EDGE as i32);
        gl.tex_parameteri(Gl::TEXTURE_2D, Gl::TEXTURE_WRAP_T, Gl::CLAMP_TO_EDGE as i32);
        // Allocate storage with `tex_image_2d` once (null source);
        // per-frame uploads via `tex_sub_image_2d` are cheaper than
        // re-allocating each frame.
        gl.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_u8_array(
            Gl::TEXTURE_2D,
            0,
            Gl::RGBA8 as i32,
            width as i32,
            height as i32,
            0,
            Gl::RGBA,
            Gl::UNSIGNED_BYTE,
            None,
        )?;

        let u_tex = gl.get_uniform_location(&program, "u_tex");
        gl.uniform1i(u_tex.as_ref(), 0);

        gl.viewport(0, 0, width as i32, height as i32);

        Ok(Self {
            gl,
            texture,
            width,
            height,
        })
    }

    /// Upload `framebuffer` (XRES × YRES `u32` ARGB voxlap pixels)
    /// to the bound texture, then draw the fullscreen quad. `fb`
    /// is treated as raw bytes — the frag shader does the BGRA →
    /// RGBA swizzle on the GPU, so no CPU-side pack step.
    fn present(&self, framebuffer: &[u32]) -> Result<(), JsValue> {
        debug_assert_eq!(framebuffer.len(), (self.width * self.height) as usize);
        let bytes: &[u8] = unsafe {
            // SAFETY: `Vec<u32>` is `Vec<u8>` + alignment guarantee.
            // Reading 4× the u32 byte count as a u8 slice is
            // well-defined (no `T: Send`-style invariants).
            std::slice::from_raw_parts(
                framebuffer.as_ptr().cast::<u8>(),
                std::mem::size_of_val(framebuffer),
            )
        };
        self.gl.bind_texture(Gl::TEXTURE_2D, Some(&self.texture));
        self.gl
            .tex_sub_image_2d_with_i32_and_i32_and_u32_and_type_and_opt_u8_array(
                Gl::TEXTURE_2D,
                0,
                0,
                0,
                self.width as i32,
                self.height as i32,
                Gl::RGBA,
                Gl::UNSIGNED_BYTE,
                Some(bytes),
            )?;
        self.gl.draw_arrays(Gl::TRIANGLE_STRIP, 0, 4);
        Ok(())
    }
}

fn compile_shader(gl: &Gl, kind: u32, src: &str) -> Result<web_sys::WebGlShader, JsValue> {
    let shader = gl
        .create_shader(kind)
        .ok_or_else(|| JsValue::from_str("create_shader failed"))?;
    gl.shader_source(&shader, src);
    gl.compile_shader(&shader);
    if !gl
        .get_shader_parameter(&shader, Gl::COMPILE_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        let log = gl
            .get_shader_info_log(&shader)
            .unwrap_or_else(|| "?".into());
        return Err(JsValue::from_str(&format!("shader compile: {log}")));
    }
    Ok(shader)
}

fn compile_program(gl: &Gl, vs: &str, fs: &str) -> Result<web_sys::WebGlProgram, JsValue> {
    let vs = compile_shader(gl, Gl::VERTEX_SHADER, vs)?;
    let fs = compile_shader(gl, Gl::FRAGMENT_SHADER, fs)?;
    let program = gl
        .create_program()
        .ok_or_else(|| JsValue::from_str("create_program failed"))?;
    gl.attach_shader(&program, &vs);
    gl.attach_shader(&program, &fs);
    gl.link_program(&program);
    if !gl
        .get_program_parameter(&program, Gl::LINK_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        let log = gl
            .get_program_info_log(&program)
            .unwrap_or_else(|| "?".into());
        return Err(JsValue::from_str(&format!("program link: {log}")));
    }
    Ok(program)
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
    render_frame(&mut state);

    // R10.X.3: GPU-side blit. `WebGlBlit::present` uploads
    // `state.fb` (voxlap u32 BGRA) to the bound texture and
    // draws the cached fullscreen quad. The frag shader does
    // the BGRA→RGBA swizzle on GPU; alpha is forced to 1.0
    // so voxlap's brightness byte doesn't darken via canvas
    // compositing. ~1-2 ms/frame faster than the previous 2D
    // `pack_rgba + putImageData` path because we drop the CPU
    // byte-swap.
    let _ = state.blit.present(&state.fb);

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
    let blit = WebGlBlit::new(&canvas, XRES, YRES)?;

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

    let now_ms = perf.as_ref().map_or(0.0, web_sys::Performance::now);
    let state = State {
        engine,
        world,
        pool,
        fb,
        zb,
        cam_pos: [1024.0, 1024.0, 128.0],
        yaw: std::f64::consts::FRAC_PI_2,
        pitch: 0.0,
        input: Input::default(),
        last_frame_ms: now_ms,
        blit,
        touches: Vec::new(),
        bench: None,
    };
    let state = Rc::new(RefCell::new(state));

    web_sys::console::log_1(
        &format!(
            "roxlap-web: parsed oracle.vxl in {:.1} ms — controls: WASD move, Space/Shift up/down, click canvas to look around, B to bench",
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
