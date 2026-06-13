//! WebGL2 framebuffer presenter for the CPU backend on wasm.
//!
//! On native the CPU backend blits its composited `0x00RRGGBB`
//! framebuffer into a `softbuffer` surface. The browser has no
//! softbuffer, so the wasm CPU fallback uploads the framebuffer to an
//! RGBA8 texture and draws a fullscreen quad — the GPU does the
//! `BGRA → RGBA` swizzle in the fragment shader, so there's no
//! per-frame CPU pack step. Ported from the original direct-opticast
//! `roxlap-web` demo (whose presenter this replaces) and given a
//! `resize` so it tracks `SceneRenderer::resize`.

use wasm_bindgen::{JsCast, JsValue};
use web_sys::{HtmlCanvasElement, WebGl2RenderingContext as Gl, WebGlTexture};

/// Fullscreen-quad vertex shader. Two triangles covering NDC
/// `[-1, 1]²`; `a_uv` is flipped vertically so the engine's top row
/// (voxlap is top-down) maps to screen-top (WebGL UV origin is
/// bottom-left).
const QUAD_VS: &str = "#version 300 es
in vec2 a_pos;
in vec2 a_uv;
out vec2 v_uv;
void main() {
    v_uv = a_uv;
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
";

/// Fragment shader: samples the framebuffer texture and swizzles
/// `.bgr` so the little-endian `0x00RRGGBB` framebuffer (bytes
/// `[B, G, R, 0]`, uploaded as `RGBA8`) reads back as the correct RGB.
/// Alpha is forced to `1.0` so the canvas compositor never darkens
/// pixels via the (zero / brightness) high byte.
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

/// Owns the WebGL2 context + the single reused texture/quad. One per
/// canvas; `present` is the only per-frame call.
pub(crate) struct WebGlBlit {
    gl: Gl,
    texture: WebGlTexture,
    width: u32,
    height: u32,
}

// WebGL's API takes `i32` for sizes / locations and `u32` enum
// constants — the cast lints fire at every call site. Values are
// bounded by canvas pixel dimensions and never wrap in practice.
#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
impl WebGlBlit {
    /// Build a blit context for `canvas`: compile the program, upload
    /// the fullscreen-quad VAO, allocate the `width × height` RGBA8
    /// texture. All resources are bound once and reused per frame.
    pub(crate) fn new(
        canvas: &HtmlCanvasElement,
        width: u32,
        height: u32,
    ) -> Result<Self, JsValue> {
        let gl: Gl = canvas
            .get_context("webgl2")?
            .ok_or_else(|| JsValue::from_str("no webgl2 context"))?
            .dyn_into::<Gl>()
            .map_err(|_| JsValue::from_str("got the wrong webgl context type"))?;

        let program = compile_program(&gl, QUAD_VS, QUAD_FS)?;
        gl.use_program(Some(&program));

        // Fullscreen quad: pos.xy + uv.xy interleaved; V flipped.
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
        // `cast_slice` reinterprets the f32 quad as bytes (no unsafe —
        // the facade is `#![forbid(unsafe_code)]`); `buffer_data` copies
        // into the GPU before returning.
        gl.buffer_data_with_u8_array(
            Gl::ARRAY_BUFFER,
            bytemuck::cast_slice(&quad),
            Gl::STATIC_DRAW,
        );
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

        let u_tex = gl.get_uniform_location(&program, "u_tex");
        gl.uniform1i(u_tex.as_ref(), 0);

        let mut blit = Self {
            gl,
            texture,
            width: 0,
            height: 0,
        };
        blit.resize(width, height);
        Ok(blit)
    }

    /// Re-allocate the texture + viewport to `width × height`. Cheap
    /// no-op when the size is unchanged. Called from
    /// `SceneRenderer::resize` (the native softbuffer surface resizes
    /// lazily; the GL texture must be re-allocated explicitly).
    pub(crate) fn resize(&mut self, width: u32, height: u32) {
        let (width, height) = (width.max(1), height.max(1));
        if (width, height) == (self.width, self.height) {
            return;
        }
        self.width = width;
        self.height = height;
        self.gl.bind_texture(Gl::TEXTURE_2D, Some(&self.texture));
        // Allocate storage once (null source); per-frame uploads use
        // `tex_sub_image_2d`, cheaper than re-allocating each frame.
        let _ = self
            .gl
            .tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_u8_array(
                Gl::TEXTURE_2D,
                0,
                Gl::RGBA8 as i32,
                width as i32,
                height as i32,
                0,
                Gl::RGBA,
                Gl::UNSIGNED_BYTE,
                None,
            );
        self.gl.viewport(0, 0, width as i32, height as i32);
    }

    /// Upload `framebuffer` (`width × height` `0x00RRGGBB` pixels) to
    /// the texture, then draw the fullscreen quad. `framebuffer` is
    /// reinterpreted as raw bytes; the frag shader does the swizzle.
    pub(crate) fn present(&self, framebuffer: &[u32]) {
        let expected = (self.width * self.height) as usize;
        if framebuffer.len() < expected {
            return;
        }
        // Reinterpret the `0x00RRGGBB` pixels as bytes (`[B, G, R, 0]`
        // little-endian); the frag shader does the swizzle. `cast_slice`
        // keeps this unsafe-free (the facade forbids `unsafe`).
        let bytes: &[u8] = bytemuck::cast_slice(&framebuffer[..expected]);
        self.gl.bind_texture(Gl::TEXTURE_2D, Some(&self.texture));
        let _ = self
            .gl
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
            );
        self.gl.draw_arrays(Gl::TRIANGLE_STRIP, 0, 4);
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
