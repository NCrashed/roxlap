//! Software rasteriser for egui tessellation — the CPU backend's `hud`
//! overlay path (no GPU). egui hands us textured triangles
//! ([`ClippedPrimitive`] meshes); we sample the font/image atlas,
//! modulate by the per-vertex colour, and src-over the result into the
//! softbuffer framebuffer.
//!
//! Scope: `Primitive::Mesh` only — `Primitive::Callback` (custom GPU
//! paints) is ignored, which is fine for a HUD. Compositing is done in
//! sRGB space directly (egui's wgpu shader works in linear); the gamma
//! shortcut is visually fine for UI text/panels and keeps the inner
//! loop integer-cheap.

// A software rasteriser is wall-to-wall float↔int pixel/coord casts;
// the truncation/sign/precision are all intentional and bounded.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_lossless
)]

use std::collections::HashMap;

use egui::epaint::{ClippedPrimitive, ImageDelta, Primitive, Vertex};
use egui::{Color32, ImageData, TextureId};

/// One cached egui texture: premultiplied sRGBA pixels, row-major
/// (top-to-bottom), `w × h`.
struct Tex {
    w: usize,
    h: usize,
    px: Vec<Color32>,
}

/// Atlas cache + rasteriser. Holds the textures egui uploads via the
/// per-frame [`egui::TexturesDelta`]; [`Self::paint`] draws a frame's
/// primitives into a `0x00RRGGBB` framebuffer.
#[derive(Default)]
pub(crate) struct EguiRaster {
    textures: HashMap<TextureId, Tex>,
}

impl EguiRaster {
    /// Apply a frame's texture delta: full uploads, partial patches, and
    /// frees. Call before [`Self::paint`].
    pub(crate) fn update_textures(&mut self, set: &[(TextureId, ImageDelta)], free: &[TextureId]) {
        for (id, delta) in set {
            let [pw, ph] = delta.image.size();
            // egui 0.34 merged font atlases into `Color` (the `Font`
            // variant is gone); pixels are premultiplied sRGBA.
            let ImageData::Color(img) = &delta.image;
            let patch: Vec<Color32> = img.pixels.clone();
            match delta.pos {
                // Whole-texture (re)upload.
                None => {
                    self.textures.insert(
                        *id,
                        Tex {
                            w: pw,
                            h: ph,
                            px: patch,
                        },
                    );
                }
                // Patch a sub-region of an existing texture.
                Some([ox, oy]) => {
                    if let Some(tex) = self.textures.get_mut(id) {
                        for row in 0..ph {
                            let dy = oy + row;
                            if dy >= tex.h {
                                break;
                            }
                            for col in 0..pw {
                                let dx = ox + col;
                                if dx >= tex.w {
                                    continue;
                                }
                                tex.px[dy * tex.w + dx] = patch[row * pw + col];
                            }
                        }
                    }
                }
            }
        }
        for id in free {
            self.textures.remove(id);
        }
    }

    /// Rasterise `jobs` into `fb` (`width × height`, `0x00RRGGBB`).
    /// `pixels_per_point` scales egui's logical points to physical
    /// pixels.
    pub(crate) fn paint(
        &self,
        fb: &mut [u32],
        width: u32,
        height: u32,
        jobs: &[ClippedPrimitive],
        pixels_per_point: f32,
    ) {
        let wf = width as f32;
        let hf = height as f32;
        for cp in jobs {
            let Primitive::Mesh(mesh) = &cp.primitive else {
                continue; // Callback primitives are GPU-only; skip.
            };
            let Some(tex) = self.textures.get(&mesh.texture_id) else {
                continue;
            };
            // Clip rect → physical pixels, intersected with the frame.
            let clip = ClipRect {
                min_x: (cp.clip_rect.min.x * pixels_per_point).floor().max(0.0) as i32,
                min_y: (cp.clip_rect.min.y * pixels_per_point).floor().max(0.0) as i32,
                max_x: (cp.clip_rect.max.x * pixels_per_point).ceil().min(wf) as i32,
                max_y: (cp.clip_rect.max.y * pixels_per_point).ceil().min(hf) as i32,
            };
            if clip.max_x <= clip.min_x || clip.max_y <= clip.min_y {
                continue;
            }
            for tri in mesh.indices.chunks_exact(3) {
                let v0 = &mesh.vertices[tri[0] as usize];
                let v1 = &mesh.vertices[tri[1] as usize];
                let v2 = &mesh.vertices[tri[2] as usize];
                raster_tri(fb, width, height, tex, &clip, v0, v1, v2, pixels_per_point);
            }
        }
    }
}

struct ClipRect {
    min_x: i32,
    min_y: i32,
    max_x: i32,
    max_y: i32,
}

/// 2D edge function: signed area of the triangle `(a, b, p)` × 2.
#[inline]
fn edge(ax: f32, ay: f32, bx: f32, by: f32, px: f32, py: f32) -> f32 {
    (bx - ax) * (py - ay) - (by - ay) * (px - ax)
}

#[allow(clippy::too_many_arguments)]
fn raster_tri(
    fb: &mut [u32],
    width: u32,
    height: u32,
    tex: &Tex,
    clip: &ClipRect,
    v0: &Vertex,
    v1: &Vertex,
    v2: &Vertex,
    ppp: f32,
) {
    // Vertex positions in physical pixels.
    let (x0, y0) = (v0.pos.x * ppp, v0.pos.y * ppp);
    let (x1, y1) = (v1.pos.x * ppp, v1.pos.y * ppp);
    let (x2, y2) = (v2.pos.x * ppp, v2.pos.y * ppp);

    let area = edge(x0, y0, x1, y1, x2, y2);
    if area == 0.0 {
        return;
    }
    let inv_area = 1.0 / area;

    // Triangle bounding box, clamped to clip ∩ framebuffer.
    let min_x = x0.min(x1).min(x2).floor() as i32;
    let max_x = x0.max(x1).max(x2).ceil() as i32;
    let min_y = y0.min(y1).min(y2).floor() as i32;
    let max_y = y0.max(y1).max(y2).ceil() as i32;
    let lo_x = min_x.max(clip.min_x).max(0);
    let hi_x = max_x.min(clip.max_x).min(width as i32);
    let lo_y = min_y.max(clip.min_y).max(0);
    let hi_y = max_y.min(clip.max_y).min(height as i32);
    if lo_x >= hi_x || lo_y >= hi_y {
        return;
    }

    let c0 = v0.color.to_array();
    let c1 = v1.color.to_array();
    let c2 = v2.color.to_array();
    let tw = tex.w as f32;
    let th = tex.h as f32;

    for y in lo_y..hi_y {
        let py = y as f32 + 0.5;
        let row = (y as usize) * (width as usize);
        for x in lo_x..hi_x {
            let px = x as f32 + 0.5;
            // Barycentric weights (sub-areas / total area). Sign-agnostic
            // inside test — egui doesn't fix its winding order.
            let w0 = edge(x1, y1, x2, y2, px, py) * inv_area;
            let w1 = edge(x2, y2, x0, y0, px, py) * inv_area;
            let w2 = edge(x0, y0, x1, y1, px, py) * inv_area;
            if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                continue;
            }

            // Interpolate UV + per-vertex colour.
            let u = w0 * v0.uv.x + w1 * v1.uv.x + w2 * v2.uv.x;
            let v = w0 * v0.uv.y + w1 * v1.uv.y + w2 * v2.uv.y;
            let vr = w0 * c0[0] as f32 + w1 * c1[0] as f32 + w2 * c2[0] as f32;
            let vg = w0 * c0[1] as f32 + w1 * c1[1] as f32 + w2 * c2[1] as f32;
            let vb = w0 * c0[2] as f32 + w1 * c1[2] as f32 + w2 * c2[2] as f32;
            let va = w0 * c0[3] as f32 + w1 * c1[3] as f32 + w2 * c2[3] as f32;

            // Nearest-sample the atlas.
            let tx = (u * tw).clamp(0.0, tw - 1.0) as usize;
            let ty = (v * th).clamp(0.0, th - 1.0) as usize;
            let texel = tex.px[ty * tex.w + tx].to_array();

            // Modulate: vertex_color × texel (both premultiplied sRGBA).
            let sr = vr * texel[0] as f32 * (1.0 / 255.0);
            let sg = vg * texel[1] as f32 * (1.0 / 255.0);
            let sb = vb * texel[2] as f32 * (1.0 / 255.0);
            let sa = va * texel[3] as f32 * (1.0 / 255.0);
            if sa <= 0.0 {
                continue;
            }

            // Premultiplied src-over onto the opaque scene pixel.
            let inv = 1.0 - sa * (1.0 / 255.0);
            let idx = row + x as usize;
            let dst = fb[idx];
            let dr = ((dst >> 16) & 0xff) as f32;
            let dg = ((dst >> 8) & 0xff) as f32;
            let db = (dst & 0xff) as f32;
            let or = (sr + dr * inv).clamp(0.0, 255.0) as u32;
            let og = (sg + dg * inv).clamp(0.0, 255.0) as u32;
            let ob = (sb + db * inv).clamp(0.0, 255.0) as u32;
            fb[idx] = (or << 16) | (og << 8) | ob;
        }
    }
}
