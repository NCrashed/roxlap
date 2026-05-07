//! roxlap-cave-web — procedural cave demo on wasm32 + canvas.
//!
//! Combines the engine + cave-gen pipeline from
//! `roxlap-cave-demo` with the canvas / `requestAnimationFrame`
//! / pointer-lock scaffolding from `roxlap-web`. The whole
//! pipeline (Worley + Perlin cave gen, voxlap renderer, sphere
//! carve, lightmode-1 bake) runs in the browser; the player
//! flies through with WASD + mouse-look, fires plasma bullets
//! that carve craters with local relight on impact, and can
//! regenerate the cave via `F` (preset) or `R` (seed).

#![cfg(target_arch = "wasm32")]
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::cell::RefCell;
use std::rc::Rc;

use roxlap_cavegen::{BlueCaveGenerator, CaveParams, Generator, MagCaveGenerator, MAXZDIM};
use roxlap_core::opticast::opticast;
use roxlap_core::rasterizer::ScratchPool;
use roxlap_core::scalar_rasterizer::ScalarRasterizer;
use roxlap_core::update_lighting;
use roxlap_core::world_query::{getcube, Cube};
use roxlap_core::{Camera, Engine, OpticastSettings};
use roxlap_formats::edit::{set_sphere_with_colfunc, SpanOp};
use roxlap_formats::vxl;
use wasm_bindgen::prelude::*;
use web_sys::{HtmlCanvasElement, KeyboardEvent, MouseEvent, WebGl2RenderingContext as Gl};

// ----- World / camera tuning (mirrors roxlap-cave-demo) ----------------------

const XRES: u32 = 640;
const YRES: u32 = 480;
const VSID: u32 = 128;

const MOVE_SPEED: f64 = 32.0;
const FAST_MULT: f64 = 4.0;
const MOUSE_SENSITIVITY: f64 = 0.0025;
const PITCH_LIMIT: f64 = std::f64::consts::FRAC_PI_2 - 0.05;
const PLAYER_RADIUS: f64 = 0.3;

const BULLET_MAX_DIST: f64 = 96.0;
const BULLET_VEL: f64 = 60.0;
const FIRE_RADIUS: u32 = 4;
const BULLET_RADIUS_PX_MAX: i32 = 12;
const BULLET_RADIUS_PX_MIN: i32 = 1;
const BULLET_COLOR_CORE: u32 = 0x00FF_4080;
const BULLET_COLOR_HALO: u32 = 0x00FF_A0C0;

#[allow(clippy::cast_possible_wrap)]
const CARVE_COLOR: i32 = 0x8050_3018u32 as i32;
#[allow(clippy::cast_possible_wrap)]
const SPAWN_BUBBLE_COLOR: i32 = 0x8060_6068u32 as i32;
const SPAWN_BUBBLE_RADIUS: u32 = 6;

const LIGHTMODE: u32 = 1;

const FOG_COLOR: u32 = 0x0090_98B0;
const FOG_MAX_SCAN_DIST: i32 = 128;

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
    vxl: vxl::Vxl,
    pool: ScratchPool,
    fb: Vec<u32>,
    zb: Vec<f32>,
    cam_pos: [f64; 3],
    yaw: f64,
    pitch: f64,
    input: Input,
    last_frame_ms: f64,
    blit: WebGlBlit,
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

/// R10.X.3: GPU-side framebuffer presenter — same shape as
/// `roxlap-web::WebGlBlit`. See that crate for the full design
/// notes; in short, the frag shader swizzles voxlap's BGRA byte
/// order to RGBA on the GPU and forces alpha = 1.0, dropping
/// the per-frame `pack_rgba` step.
struct WebGlBlit {
    gl: Gl,
    texture: web_sys::WebGlTexture,
    width: u32,
    height: u32,
}

const QUAD_VS: &str = "#version 300 es
in vec2 a_pos;
in vec2 a_uv;
out vec2 v_uv;
void main() {
    v_uv = a_uv;
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
";

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

#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
impl WebGlBlit {
    fn new(canvas: &HtmlCanvasElement, width: u32, height: u32) -> Result<Self, JsValue> {
        let gl: Gl = canvas
            .get_context("webgl2")?
            .ok_or_else(|| JsValue::from_str("no webgl2 context"))?
            .dyn_into::<Gl>()
            .map_err(|_| JsValue::from_str("got the wrong webgl context type"))?;
        let program = compile_program(&gl, QUAD_VS, QUAD_FS)?;
        gl.use_program(Some(&program));
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
            // SAFETY: js_sys::Float32Array::view borrows wasm
            // linear-memory backing of `quad`; bufferData copies
            // into the GPU before the view escapes.
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

        let texture = gl
            .create_texture()
            .ok_or_else(|| JsValue::from_str("create_texture failed"))?;
        gl.active_texture(Gl::TEXTURE0);
        gl.bind_texture(Gl::TEXTURE_2D, Some(&texture));
        gl.tex_parameteri(Gl::TEXTURE_2D, Gl::TEXTURE_MIN_FILTER, Gl::LINEAR as i32);
        gl.tex_parameteri(Gl::TEXTURE_2D, Gl::TEXTURE_MAG_FILTER, Gl::LINEAR as i32);
        gl.tex_parameteri(Gl::TEXTURE_2D, Gl::TEXTURE_WRAP_S, Gl::CLAMP_TO_EDGE as i32);
        gl.tex_parameteri(Gl::TEXTURE_2D, Gl::TEXTURE_WRAP_T, Gl::CLAMP_TO_EDGE as i32);
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

    fn present(&self, framebuffer: &[u32]) -> Result<(), JsValue> {
        debug_assert_eq!(framebuffer.len(), (self.width * self.height) as usize);
        let bytes: &[u8] = unsafe {
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
fn spawn_centre() -> [i32; 3] {
    [(VSID / 2) as i32, (VSID / 2) as i32, MAXZDIM / 2]
}

fn build_world(preset: Preset, seed: u64) -> vxl::Vxl {
    let mut vxl = preset.generate(seed);
    let centre = spawn_centre();
    set_sphere_with_colfunc(
        &mut vxl,
        centre,
        SPAWN_BUBBLE_RADIUS,
        SpanOp::Carve,
        |_, _, _| SPAWN_BUBBLE_COLOR,
    );
    vxl
}

fn relight_world(vxl: &mut vxl::Vxl, engine: &Engine) {
    #[allow(clippy::cast_possible_wrap)]
    update_lighting(
        &mut vxl.data,
        &vxl.column_offset,
        vxl.vsid,
        0,
        0,
        0,
        vxl.vsid as i32,
        vxl.vsid as i32,
        MAXZDIM,
        engine.lightmode(),
        engine.lights(),
    );
}

#[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
fn relight_bbox(vxl: &mut vxl::Vxl, engine: &Engine, centre: [i32; 3], radius: u32) {
    let r = radius as i32;
    let vsid_i = vxl.vsid as i32;
    let x0 = (centre[0] - r).max(0);
    let y0 = (centre[1] - r).max(0);
    let z0 = (centre[2] - r).max(0);
    let x1 = (centre[0] + r + 1).min(vsid_i);
    let y1 = (centre[1] + r + 1).min(vsid_i);
    let z1 = (centre[2] + r + 1).min(MAXZDIM);
    if x0 >= x1 || y0 >= y1 || z0 >= z1 {
        return;
    }
    update_lighting(
        &mut vxl.data,
        &vxl.column_offset,
        vxl.vsid,
        x0,
        y0,
        z0,
        x1,
        y1,
        z1,
        engine.lightmode(),
        engine.lights(),
    );
}

// ----- Camera + collision ---------------------------------------------------

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
                    return true;
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
    let mag = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
    if mag <= 1e-6 {
        return;
    }
    let speed = if state.input.fast {
        MOVE_SPEED * FAST_MULT
    } else {
        MOVE_SPEED
    };
    let step = [
        delta[0] / mag * speed * dt,
        delta[1] / mag * speed * dt,
        delta[2] / mag * speed * dt,
    ];
    let already_stuck = is_blocked(&state.vxl, state.cam_pos);
    for axis in 0..3 {
        let mut candidate = state.cam_pos;
        candidate[axis] += step[axis];
        if already_stuck || !is_blocked(&state.vxl, candidate) {
            state.cam_pos[axis] = candidate[axis];
        }
    }
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

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn step_bullets(state: &mut State, dt: f64) {
    if dt <= 0.0 {
        return;
    }
    let vsid = state.vxl.vsid;
    let mut impacts: Vec<[i32; 3]> = Vec::new();
    state.bullets.retain_mut(|b| {
        let dx = b.vel[0] * dt;
        let dy = b.vel[1] * dt;
        let dz = b.vel[2] * dt;
        b.pos[0] += dx;
        b.pos[1] += dy;
        b.pos[2] += dz;
        b.travelled += (dx * dx + dy * dy + dz * dz).sqrt();
        if bullet_radius_px(b) < BULLET_RADIUS_PX_MIN {
            return false;
        }
        let vx = b.pos[0].floor() as i32;
        let vy = b.pos[1].floor() as i32;
        let vz = b.pos[2].floor() as i32;
        if vx < 0
            || vy < 0
            || (vx as u32) >= vsid
            || (vy as u32) >= vsid
            || !(0..MAXZDIM).contains(&vz)
        {
            return false;
        }
        if !matches!(
            getcube(&state.vxl.data, &state.vxl.column_offset, vsid, vx, vy, vz),
            Cube::Air
        ) {
            impacts.push([vx, vy, vz]);
            return false;
        }
        true
    });
    for hit in impacts {
        set_sphere_with_colfunc(
            &mut state.vxl,
            hit,
            FIRE_RADIUS,
            SpanOp::Carve,
            |_x, _y, _z| CARVE_COLOR,
        );
        relight_bbox(&mut state.vxl, &state.engine, hit, FIRE_RADIUS);
    }
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn bullet_radius_px(bullet: &Bullet) -> i32 {
    let t = (bullet.travelled / BULLET_MAX_DIST).clamp(0.0, 1.0);
    let r = (1.0 - t) * f64::from(BULLET_RADIUS_PX_MAX);
    r.round() as i32
}

// ----- Render ---------------------------------------------------------------

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
        &state.vxl.data,
        &state.vxl.column_offset,
        &state.vxl.mip_base_offsets,
        state.vxl.vsid,
    );
    let _ = opticast(
        &mut rasterizer,
        &mut state.pool,
        &cam,
        &settings,
        state.vxl.vsid,
        &state.vxl.data,
        &state.vxl.column_offset,
    );
    drop(rasterizer);

    // Bullet billboards on top of the rasterized scene.
    let cam_for_bullet = cam_from_yaw_pitch(state.cam_pos, state.yaw, state.pitch);
    let settings_for_bullet = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
    for bullet in &state.bullets {
        draw_bullet(
            &mut state.fb,
            &state.zb,
            XRES,
            YRES,
            &cam_for_bullet,
            &settings_for_bullet,
            bullet,
        );
    }
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::similar_names
)]
fn draw_bullet(
    fb: &mut [u32],
    zb: &[f32],
    width: u32,
    height: u32,
    cam: &Camera,
    settings: &OpticastSettings,
    bullet: &Bullet,
) {
    // World-space → camera-space
    let dx = bullet.pos[0] - cam.pos[0];
    let dy = bullet.pos[1] - cam.pos[1];
    let dz = bullet.pos[2] - cam.pos[2];
    // Voxlap basis projection: x_cam = right · delta, y_cam = down ·
    // delta, z_cam = forward · delta. Cull anything behind the eye.
    let zc = cam.forward[0] * dx + cam.forward[1] * dy + cam.forward[2] * dz;
    if zc <= 1e-3 {
        return;
    }
    let xc = cam.right[0] * dx + cam.right[1] * dy + cam.right[2] * dz;
    let yc = cam.down[0] * dx + cam.down[1] * dy + cam.down[2] * dz;
    let inv_z = 1.0 / zc;
    let cx = f64::from(settings.hx) + xc * inv_z * f64::from(settings.hz);
    let cy = f64::from(settings.hy) + yc * inv_z * f64::from(settings.hz);
    let r_px = bullet_radius_px(bullet);
    if r_px < 1 {
        return;
    }
    let r2 = r_px * r_px;
    let r_halo2 = (r_px - 1).max(0).pow(2);
    let cx_i = cx.round() as i32;
    let cy_i = cy.round() as i32;
    let z_value = zc as f32;
    let w = width as i32;
    let h = height as i32;
    for py in (cy_i - r_px)..=(cy_i + r_px) {
        if py < 0 || py >= h {
            continue;
        }
        for px in (cx_i - r_px)..=(cx_i + r_px) {
            if px < 0 || px >= w {
                continue;
            }
            let ddx = px - cx_i;
            let ddy = py - cy_i;
            let d2 = ddx * ddx + ddy * ddy;
            if d2 > r2 {
                continue;
            }
            let idx = (py as usize) * (width as usize) + px as usize;
            // z-test: bullet is in front of the world voxel here.
            if zb[idx] > 0.0 && zb[idx] < z_value {
                continue;
            }
            fb[idx] = if d2 <= r_halo2 {
                BULLET_COLOR_CORE
            } else {
                BULLET_COLOR_HALO
            };
        }
    }
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
    step_bullets(&mut state, dt);
    render_frame(&mut state);

    // R10.X.3: GPU-side blit — texture upload + cached
    // fullscreen quad. Frag shader does the BGRA→RGBA swizzle
    // and forces alpha = 1.0 so voxlap's brightness byte
    // doesn't darken via canvas compositing. ~1-2 ms/frame
    // faster than the prior 2D `pack_rgba + putImageData`
    // path.
    let _ = state.blit.present(&state.fb);
}

// ----- Regenerate -----------------------------------------------------------

fn regenerate(state: &mut State) {
    state.vxl = build_world(state.preset, state.seed);
    relight_world(&mut state.vxl, &state.engine);
    state.cam_pos = [
        f64::from(VSID) * 0.5,
        f64::from(VSID) * 0.5,
        f64::from(MAXZDIM) * 0.5,
    ];
    state.bullets.clear();
    state.pool = ScratchPool::new(XRES, YRES, state.vxl.vsid);
}

// ----- Init -----------------------------------------------------------------

/// wasm-bindgen `start` hook — invoked by the JS shim once the
/// module finishes loading. Generates a cave, bakes lighting,
/// hooks input, kicks off the RAF loop.
///
/// # Errors
/// Returns a JS-bridged error if the DOM doesn't have the
/// expected `<canvas id="roxlap-canvas">`, or if a 2D rendering
/// context can't be acquired.
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
    let preset = Preset::Blue;
    let seed = preset.default_params().seed;
    let t_gen_start = perf.as_ref().map_or(0.0, web_sys::Performance::now);
    let mut vxl = build_world(preset, seed);
    let t_gen_end = perf.as_ref().map_or(0.0, web_sys::Performance::now);

    let mut engine = Engine::new();
    engine.set_fog(FOG_COLOR, FOG_MAX_SCAN_DIST);
    engine.set_lightmode(LIGHTMODE);
    let t_bake_start = perf.as_ref().map_or(0.0, web_sys::Performance::now);
    relight_world(&mut vxl, &engine);
    let t_bake_end = perf.as_ref().map_or(0.0, web_sys::Performance::now);

    let pool = ScratchPool::new(XRES, YRES, vxl.vsid);
    let fb = vec![0u32; (XRES * YRES) as usize];
    let zb = vec![0f32; (XRES * YRES) as usize];

    let now_ms = perf.as_ref().map_or(0.0, web_sys::Performance::now);
    let state = State {
        engine,
        vxl,
        pool,
        fb,
        zb,
        cam_pos: [
            f64::from(VSID) * 0.5,
            f64::from(VSID) * 0.5,
            f64::from(MAXZDIM) * 0.5,
        ],
        yaw: 0.0,
        pitch: 0.0,
        input: Input::default(),
        last_frame_ms: now_ms,
        blit,
        bullets: Vec::new(),
        preset,
        seed,
        touches: Vec::new(),
    };
    let state = Rc::new(RefCell::new(state));

    web_sys::console::log_1(
        &format!(
            "roxlap-cave-web: cave-gen {:.0} ms, lightmode-1 bake {:.0} ms — controls: WASD move, Space/Shift up/down, Shift+/Ctrl fast, click canvas to look around, click again to fire, F preset, R reseed",
            t_gen_end - t_gen_start,
            t_bake_end - t_bake_start,
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
