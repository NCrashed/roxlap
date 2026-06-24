//! Clean-room KV6 sprite raycaster for the DDA backend (Substage
//! DDA.8).
//!
//! Renders KV6 sprites by **per-pixel ray casting**: for every screen
//! pixel the sprite covers, transform the camera ray into the sprite's
//! local voxel space, 3D-DDA through the KV6, and depth-composite the
//! first solid voxel against the shared z-buffer. Clean-room (no voxlap
//! code), the sprite counterpart to the terrain renderer in
//! [`crate::dda`].
//!
//! **Depth parity.** Transforming the ray by the inverse sprite basis
//! leaves the ray parameter unchanged in world units — a hit at local
//! parameter `t` is at world point `cam.pos + dir·t` — so the
//! perpendicular depth is `t · (dir·forward)`, exactly the convention
//! [`crate::dda`] writes for terrain. Sprites therefore occlude and are
//! occluded by DDA terrain correctly.
//!
//! Shading reads the KV6 voxel's baked brightness byte (high byte of
//! the packed colour) via [`crate::dda::shade`] — the clean-room
//! brightness model, not voxlap's `dir`-LUT reflection shading.

use roxlap_formats::kv6::Kv6;
use roxlap_formats::sprite::{Sprite, SPRITE_FLAG_INVISIBLE, SPRITE_FLAG_NO_Z};

use crate::camera_math::CameraState;
use crate::dda::{dda_setup, intersect_aabb, min_axis, pixel_ray, shade};
use crate::opticast::OpticastSettings;
use crate::raster_target::RasterTarget;

/// Near-plane parameter: voxels nearer than this (camera-forward) are
/// dropped, keeping the pinhole divide finite.
const NEAR_Z: f32 = 1.0;

/// Dense occupancy + colour grid decoded from a KV6's surface-voxel run
/// tables. KV6 stores only surface voxels (no interior), which is
/// exactly what a from-air ray hits, so first-occupied is the visible
/// surface.
struct Kv6Dense {
    dims: [i32; 3],
    occ: Vec<bool>,
    col: Vec<u32>,
}

impl Kv6Dense {
    #[allow(clippy::cast_possible_wrap)]
    fn build(kv6: &Kv6) -> Self {
        let dims = [kv6.xsiz as i32, kv6.ysiz as i32, kv6.zsiz as i32];
        let n = (dims[0].max(0) * dims[1].max(0) * dims[2].max(0)) as usize;
        let mut occ = vec![false; n];
        let mut col = vec![0u32; n];
        let mut vi = 0usize;
        for x in 0..kv6.xsiz as usize {
            for y in 0..kv6.ysiz as usize {
                let cnt = usize::from(kv6.ylen[x][y]);
                for _ in 0..cnt {
                    let v = kv6.voxels[vi];
                    vi += 1;
                    let z = i32::from(v.z);
                    if z >= 0 && z < dims[2] {
                        let idx = ((x as i32 * dims[1] + y as i32) * dims[2] + z) as usize;
                        occ[idx] = true;
                        // KV6's colour high byte is voxlap's `dir`/shading
                        // slot, NOT the 0..128 brightness `shade` expects:
                        // some models store `0x80` (coco), others `0x00`
                        // (meltsphere → all-black under `shade`). Force full
                        // brightness so both render at their authored RGB
                        // (sprites are flat-lit in the clean-room path; the
                        // voxlap `dir`-LUT reflection shading is not ported).
                        col[idx] = (v.col & 0x00ff_ffff) | 0x8000_0000;
                    }
                }
            }
        }
        Self { dims, occ, col }
    }

    #[inline]
    #[allow(clippy::cast_sign_loss)]
    fn at(&self, c: [i32; 3]) -> Option<u32> {
        let idx = ((c[0] * self.dims[1] + c[1]) * self.dims[2] + c[2]) as usize;
        self.occ[idx].then(|| self.col[idx])
    }
}

/// Inverse of the column-matrix `[s | h | f]` (the sprite basis), or
/// `None` if degenerate. Maps a world delta into local voxel space.
fn invert_basis(s: [f32; 3], h: [f32; 3], f: [f32; 3]) -> Option<[[f32; 3]; 3]> {
    let det = s[0] * (h[1] * f[2] - f[1] * h[2]) - h[0] * (s[1] * f[2] - f[1] * s[2])
        + f[0] * (s[1] * h[2] - h[1] * s[2]);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv = 1.0 / det;
    Some([
        [
            (h[1] * f[2] - f[1] * h[2]) * inv,
            -(h[0] * f[2] - f[0] * h[2]) * inv,
            (h[0] * f[1] - f[0] * h[1]) * inv,
        ],
        [
            -(s[1] * f[2] - f[1] * s[2]) * inv,
            (s[0] * f[2] - f[0] * s[2]) * inv,
            -(s[0] * f[1] - f[0] * s[1]) * inv,
        ],
        [
            (s[1] * h[2] - h[1] * s[2]) * inv,
            -(s[0] * h[2] - h[0] * s[2]) * inv,
            (s[0] * h[1] - h[0] * s[1]) * inv,
        ],
    ])
}

#[inline]
fn mat_apply(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// Cast one ray (already in the sprite's local voxel space) into the
/// dense KV6 and return `(colour, t)` of the first solid voxel — `t` is
/// the world-units ray parameter (shared with the world ray).
#[allow(clippy::cast_possible_truncation)]
fn cast_local(dense: &Kv6Dense, origin: [f32; 3], dir: [f32; 3]) -> Option<(u32, f32)> {
    #[allow(clippy::cast_precision_loss)]
    let hi = [
        dense.dims[0] as f32,
        dense.dims[1] as f32,
        dense.dims[2] as f32,
    ];
    let (t0, t1) = intersect_aabb(origin, dir, [0.0; 3], hi)?;
    let start = t0 + 1e-4;
    let p = [
        origin[0] + dir[0] * start,
        origin[1] + dir[1] * start,
        origin[2] + dir[2] * start,
    ];
    let mut cell = [
        (p[0].floor() as i32).clamp(0, dense.dims[0] - 1),
        (p[1].floor() as i32).clamp(0, dense.dims[1] - 1),
        (p[2].floor() as i32).clamp(0, dense.dims[2] - 1),
    ];
    let (step, mut t_max, t_delta) = dda_setup(origin, dir, cell, 1.0);
    let mut t_curr = t0;
    let max_steps = (dense.dims[0] + dense.dims[1] + dense.dims[2]) as usize + 8;
    for _ in 0..max_steps {
        if cell[0] < 0
            || cell[0] >= dense.dims[0]
            || cell[1] < 0
            || cell[1] >= dense.dims[1]
            || cell[2] < 0
            || cell[2] >= dense.dims[2]
            || t_curr > t1
        {
            return None;
        }
        if let Some(color) = dense.at(cell) {
            return Some((color, t_curr));
        }
        let axis = min_axis(t_max);
        t_curr = t_max[axis];
        cell[axis] += step[axis];
        t_max[axis] += t_delta[axis];
    }
    None
}

/// Draw one KV6 [`Sprite`] into `(fb, zb)` by per-pixel ray casting,
/// depth-compositing against whatever the terrain pass already wrote.
/// Returns the number of pixels written.
///
/// `cam` / `settings` are the **same** per-frame projection the DDA
/// terrain pass used (build via [`crate::camera_math::derive`]), so
/// sprite and terrain share one pinhole and z convention. `pitch_pixels`
/// is the framebuffer row stride. Honours `SPRITE_FLAG_INVISIBLE`
/// (skip) and `SPRITE_FLAG_NO_Z` (write without the depth test).
#[allow(
    clippy::too_many_arguments,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
#[must_use]
pub fn draw_sprite_dda(
    fb: &mut [u32],
    zb: &mut [f32],
    pitch_pixels: usize,
    width: u32,
    height: u32,
    cam: &CameraState,
    settings: &OpticastSettings,
    sprite: &Sprite,
) -> u32 {
    if sprite.flags & SPRITE_FLAG_INVISIBLE != 0 {
        return 0;
    }
    let dense = Kv6Dense::build(&sprite.kv6);
    if dense.occ.is_empty() {
        return 0;
    }
    let Some(minv) = invert_basis(sprite.s, sprite.h, sprite.f) else {
        return 0;
    };
    let pivot = [sprite.kv6.xpiv, sprite.kv6.ypiv, sprite.kv6.zpiv];
    let no_z = sprite.flags & SPRITE_FLAG_NO_Z != 0;

    // Screen bounding box from the 8 corners of the local voxel box.
    let Some(rect) = project_screen_rect(sprite, cam, settings, width, height) else {
        return 0;
    };

    debug_assert_eq!(fb.len(), zb.len());
    let target = RasterTarget::new(fb, zb);
    let mut written = 0u32;
    for py in rect.1..rect.3 {
        let row = py as usize * pitch_pixels;
        for px in rect.0..rect.2 {
            let (origin, dir) = pixel_ray(cam, settings, px, py);
            // World ray → sprite-local voxel space.
            let rel = [
                origin[0] - sprite.p[0],
                origin[1] - sprite.p[1],
                origin[2] - sprite.p[2],
            ];
            let ol = mat_apply(&minv, rel);
            let origin_local = [ol[0] + pivot[0], ol[1] + pivot[1], ol[2] + pivot[2]];
            let dir_local = mat_apply(&minv, dir);
            let Some((color, t)) = cast_local(&dense, origin_local, dir_local) else {
                continue;
            };
            let fwd_dot =
                dir[0] * cam.forward[0] + dir[1] * cam.forward[1] + dir[2] * cam.forward[2];
            let depth = t * fwd_dot;
            if depth < NEAR_Z {
                continue;
            }
            let lit = shade(color, 0);
            let idx = row + px as usize;
            // SAFETY: idx in-bounds for the rect within (width, height);
            // single-threaded writer.
            let wrote = unsafe {
                if no_z {
                    target.write_color(idx, lit);
                    target.write_depth(idx, depth);
                    true
                } else {
                    target.z_test_write(idx, lit, depth)
                }
            };
            written += u32::from(wrote);
        }
    }
    written
}

/// Project the sprite's local voxel AABB to a clamped screen rectangle
/// `(x0, y0, x1, y1)` (half-open). `None` if it can't appear; falls back
/// to the full viewport when the box straddles the near plane (rare).
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn project_screen_rect(
    sprite: &Sprite,
    cam: &CameraState,
    settings: &OpticastSettings,
    width: u32,
    height: u32,
) -> Option<(u32, u32, u32, u32)> {
    let kv6 = &sprite.kv6;
    let (xs, ys, zs) = (kv6.xsiz as f32, kv6.ysiz as f32, kv6.zsiz as f32);
    let (xp, yp, zp) = (kv6.xpiv, kv6.ypiv, kv6.zpiv);
    let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    let mut all_front = true;
    for &cx in &[0.0, xs] {
        for &cy in &[0.0, ys] {
            for &cz in &[0.0, zs] {
                // Local → world via the sprite basis about the pivot.
                let lx = cx - xp;
                let ly = cy - yp;
                let lz = cz - zp;
                let world = [
                    sprite.p[0] + lx * sprite.s[0] + ly * sprite.h[0] + lz * sprite.f[0],
                    sprite.p[1] + lx * sprite.s[1] + ly * sprite.h[1] + lz * sprite.f[1],
                    sprite.p[2] + lx * sprite.s[2] + ly * sprite.h[2] + lz * sprite.f[2],
                ];
                let rel = [
                    world[0] - cam.pos[0],
                    world[1] - cam.pos[1],
                    world[2] - cam.pos[2],
                ];
                let cz_cam =
                    rel[0] * cam.forward[0] + rel[1] * cam.forward[1] + rel[2] * cam.forward[2];
                if cz_cam < NEAR_Z {
                    all_front = false;
                    continue;
                }
                let cx_cam = rel[0] * cam.right[0] + rel[1] * cam.right[1] + rel[2] * cam.right[2];
                let cy_cam = rel[0] * cam.down[0] + rel[1] * cam.down[1] + rel[2] * cam.down[2];
                let sx = settings.hx + cx_cam / cz_cam * settings.hz;
                let sy = settings.hy + cy_cam / cz_cam * settings.hz;
                x0 = x0.min(sx);
                y0 = y0.min(sy);
                x1 = x1.max(sx);
                y1 = y1.max(sy);
            }
        }
    }
    let (w, h) = (width as f32, height as f32);
    let (rx0, ry0, rx1, ry1) = if all_front {
        (
            (x0 - 1.0).max(0.0),
            (y0 - 1.0).max(0.0),
            (x1 + 1.0).min(w),
            (y1 + 1.0).min(h),
        )
    } else {
        // Straddles the near plane → scan the whole viewport.
        (0.0, 0.0, w, h)
    };
    if rx0 >= rx1 || ry0 >= ry1 {
        return None;
    }
    Some((rx0 as u32, ry0 as u32, rx1.ceil() as u32, ry1.ceil() as u32))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera_math;
    use crate::Camera;
    use roxlap_formats::kv6::Kv6;
    use roxlap_formats::sprite::Sprite;

    fn settings(w: u32, h: u32) -> OpticastSettings {
        OpticastSettings::for_oracle_framebuffer(w, h)
    }

    /// Camera at the origin looking down +y at a sprite ahead.
    fn cam_looking_y() -> Camera {
        Camera {
            pos: [0.0, 0.0, 0.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 0.0, 1.0],
            forward: [0.0, 1.0, 0.0],
        }
    }

    /// A solid cube sprite in front of the camera is drawn, with the
    /// cube colour (shaded) and a sensible centre depth.
    #[test]
    fn cube_sprite_renders() {
        let kv6 = Kv6::solid_cube(8, 0x80_C0_40_20);
        let sprite = Sprite::axis_aligned(kv6, [0.0, 40.0, 0.0]);
        let (w, h) = (64u32, 64u32);
        let n = (w * h) as usize;
        let mut fb = vec![0u32; n];
        let mut zb = vec![f32::INFINITY; n];
        let cam = cam_looking_y();
        let cs = camera_math::derive(&cam, w, h, 32.0, 32.0, 32.0);
        let wrote = draw_sprite_dda(
            &mut fb,
            &mut zb,
            w as usize,
            w,
            h,
            &cs,
            &settings(w, h),
            &sprite,
        );

        assert!(wrote > 20, "cube should cover many pixels (got {wrote})");
        let centre = (h / 2 * w + w / 2) as usize;
        assert_eq!(
            fb[centre] & 0x00ff_ffff,
            0x00_C0_40_20,
            "got {:08x}",
            fb[centre]
        );
        // Pivot at world y=40, cube spans y in [36,44] → near face ~36.
        assert!(
            (zb[centre] - 36.0).abs() < 3.0,
            "centre depth {} not ≈ 36",
            zb[centre]
        );
    }

    /// A KV6 whose voxel colours store a `0x00` high byte (voxlap's
    /// unused `dir` slot, e.g. `sprite_meltsphere.kv6`) must still
    /// render its authored RGB, not black — the brightness byte is
    /// normalised to full on decode.
    #[test]
    fn zero_high_byte_sprite_not_black() {
        let kv6 = Kv6::solid_cube(8, 0x00_C0_40_20);
        let sprite = Sprite::axis_aligned(kv6, [0.0, 40.0, 0.0]);
        let (w, h) = (64u32, 64u32);
        let n = (w * h) as usize;
        let mut fb = vec![0u32; n];
        let mut zb = vec![f32::INFINITY; n];
        let cam = cam_looking_y();
        let cs = camera_math::derive(&cam, w, h, 32.0, 32.0, 32.0);
        let wrote = draw_sprite_dda(
            &mut fb,
            &mut zb,
            w as usize,
            w,
            h,
            &cs,
            &settings(w, h),
            &sprite,
        );
        assert!(wrote > 20, "cube should cover many pixels (got {wrote})");
        let centre = (h / 2 * w + w / 2) as usize;
        assert_eq!(
            fb[centre] & 0x00ff_ffff,
            0x00_C0_40_20,
            "zero-high-byte sprite rendered as {:08x} (black bug)",
            fb[centre]
        );
    }

    /// A sprite occludes / is occluded by the z-buffer: a nearer
    /// pre-filled depth blocks the sprite; a farther one lets it win.
    #[test]
    fn sprite_respects_zbuffer() {
        let kv6 = Kv6::solid_cube(8, 0x80_FF_FF_FF);
        let sprite = Sprite::axis_aligned(kv6, [0.0, 40.0, 0.0]);
        let (w, h) = (32u32, 32u32);
        let n = (w * h) as usize;
        let cam = cam_looking_y();
        let cs = camera_math::derive(&cam, w, h, 16.0, 16.0, 16.0);
        let centre = (h / 2 * w + w / 2) as usize;

        // Terrain in front (depth 10 < ~36) → sprite blocked at centre.
        let mut fb = vec![0u32; n];
        let mut zb = vec![f32::INFINITY; n];
        fb[centre] = 0x80_11_22_33;
        zb[centre] = 10.0;
        let _ = draw_sprite_dda(
            &mut fb,
            &mut zb,
            w as usize,
            w,
            h,
            &cs,
            &settings(w, h),
            &sprite,
        );
        assert_eq!(
            fb[centre], 0x80_11_22_33,
            "near terrain must occlude sprite"
        );

        // Terrain behind (depth 100) → sprite wins.
        let mut fb2 = vec![0u32; n];
        let mut zb2 = vec![f32::INFINITY; n];
        fb2[centre] = 0x80_11_22_33;
        zb2[centre] = 100.0;
        let _ = draw_sprite_dda(
            &mut fb2,
            &mut zb2,
            w as usize,
            w,
            h,
            &cs,
            &settings(w, h),
            &sprite,
        );
        assert_ne!(fb2[centre], 0x80_11_22_33, "sprite must beat far terrain");
        assert!(zb2[centre] < 100.0, "sprite depth must replace terrain's");
    }

    /// An invisible sprite draws nothing.
    #[test]
    fn invisible_sprite_skipped() {
        let kv6 = Kv6::solid_cube(8, 0x80_FF_FF_FF);
        let mut sprite = Sprite::axis_aligned(kv6, [0.0, 40.0, 0.0]);
        sprite.flags |= roxlap_formats::sprite::SPRITE_FLAG_INVISIBLE;
        let (w, h) = (32u32, 32u32);
        let n = (w * h) as usize;
        let mut fb = vec![0u32; n];
        let mut zb = vec![f32::INFINITY; n];
        let cam = cam_looking_y();
        let cs = camera_math::derive(&cam, w, h, 16.0, 16.0, 16.0);
        let wrote = draw_sprite_dda(
            &mut fb,
            &mut zb,
            w as usize,
            w,
            h,
            &cs,
            &settings(w, h),
            &sprite,
        );
        assert_eq!(wrote, 0);
    }
}
