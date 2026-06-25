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
use roxlap_formats::voxel_clip::{DecodedClip, VoxelFrame};

use crate::camera_math::CameraState;
use crate::dda::{dda_setup, intersect_aabb, min_axis, pixel_ray, shade};
use crate::opticast::OpticastSettings;
use crate::raster_target::RasterTarget;

/// Near-plane parameter: voxels nearer than this (camera-forward) are
/// dropped, keeping the pinhole divide finite.
const NEAR_Z: f32 = 1.0;

/// Force a packed voxel colour to full brightness for the flat-lit
/// clean-room sprite path. KV6 / voxel-clip colours carry voxlap's
/// `dir`/shading slot in the high byte (some `0x80`, some `0x00`), not
/// the 0..128 brightness [`shade`] expects, so a raw value can render
/// black; we render every sprite voxel at its authored RGB.
#[inline]
fn full_bright(col: u32) -> u32 {
    (col & 0x00ff_ffff) | 0x8000_0000
}

/// Dense occupancy + colour grid for one sprite frame, plus its pivot —
/// the decoded form the per-pixel raycaster marches. Built once from a
/// [`Kv6`] ([`SpriteDense::from_kv6`]) or a voxel-clip [`VoxelFrame`]
/// ([`SpriteDense::from_voxel_frame`]); the latter lets an animated clip
/// cache every frame's grid up front instead of rebuilding per frame.
///
/// Both sources store only **surface** voxels (a from-air ray's first
/// hit is the visible surface), so the grid is the visible hull.
pub struct SpriteDense {
    dims: [i32; 3],
    occ: Vec<bool>,
    col: Vec<u32>,
    pivot: [f32; 3],
}

impl SpriteDense {
    /// Decode a [`Kv6`]'s surface-voxel run tables into a dense grid.
    #[must_use]
    #[allow(clippy::cast_possible_wrap)]
    pub fn from_kv6(kv6: &Kv6) -> Self {
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
                        col[idx] = full_bright(v.col);
                    }
                }
            }
        }
        Self {
            dims,
            occ,
            col,
            pivot: [kv6.xpiv, kv6.ypiv, kv6.zpiv],
        }
    }

    /// Decode a voxel-clip [`VoxelFrame`] (dense-column layout) into the
    /// dense grid, given the clip's `dims` + `pivot`. The frame's columns
    /// are `col = x + y*dims[0]`, each a per-column occupancy bitmask with
    /// an ascending-z colour run — walked here into the raycaster's
    /// `(x·my + y)·mz + z` grid.
    #[must_use]
    #[allow(clippy::cast_possible_wrap)]
    pub fn from_voxel_frame(frame: &VoxelFrame, dims: [u32; 3], pivot: [f32; 3]) -> Self {
        let (mx, my, mz) = (dims[0], dims[1], dims[2]);
        let owpc = mz.div_ceil(32).max(1) as usize;
        let n = (mx * my * mz) as usize;
        let mut occ = vec![false; n];
        let mut col = vec![0u32; n];
        for col_idx in 0..(mx * my) as usize {
            let x = col_idx as u32 % mx;
            let y = col_idx as u32 / mx;
            let run_start = frame.color_offsets[col_idx] as usize;
            let mut k = 0usize;
            for z in 0..mz {
                let word = frame.occupancy[col_idx * owpc + (z >> 5) as usize];
                if (word >> (z & 31)) & 1 != 0 {
                    let idx = (((x * my + y) * mz) + z) as usize;
                    occ[idx] = true;
                    col[idx] = full_bright(frame.colors[run_start + k]);
                    k += 1;
                }
            }
        }
        Self {
            dims: [mx as i32, my as i32, mz as i32],
            occ,
            col,
            pivot,
        }
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
fn cast_local(dense: &SpriteDense, origin: [f32; 3], dir: [f32; 3]) -> Option<(u32, f32)> {
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
    // Decodes the KV6 to a dense grid each call (the per-frame cost an
    // animated clip avoids via [`ClipFlipbook`]'s cached grids).
    let dense = SpriteDense::from_kv6(&sprite.kv6);
    draw_sprite_dense(
        fb,
        zb,
        pitch_pixels,
        width,
        height,
        cam,
        settings,
        &dense,
        sprite.p,
        sprite.s,
        sprite.h,
        sprite.f,
        sprite.flags,
    )
}

/// Draw a pre-decoded [`SpriteDense`] at a world pose — the generalised
/// core of [`draw_sprite_dda`], shared by the KV6 path and animated
/// [`ClipFlipbook`] frames. `pos` is the world pivot; `s`/`h`/`f` are the
/// model→world basis columns (local +x/+y/+z); `flags` honours
/// [`SPRITE_FLAG_INVISIBLE`] / [`SPRITE_FLAG_NO_Z`]. Returns pixels written.
#[allow(
    clippy::too_many_arguments,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
#[must_use]
pub fn draw_sprite_dense(
    fb: &mut [u32],
    zb: &mut [f32],
    pitch_pixels: usize,
    width: u32,
    height: u32,
    cam: &CameraState,
    settings: &OpticastSettings,
    dense: &SpriteDense,
    pos: [f32; 3],
    s: [f32; 3],
    h: [f32; 3],
    f: [f32; 3],
    flags: u32,
) -> u32 {
    if flags & SPRITE_FLAG_INVISIBLE != 0 || dense.occ.is_empty() {
        return 0;
    }
    let Some(minv) = invert_basis(s, h, f) else {
        return 0;
    };
    let pivot = dense.pivot;
    let no_z = flags & SPRITE_FLAG_NO_Z != 0;

    // Screen bounding box from the 8 corners of the local voxel box.
    let Some(rect) = project_screen_rect(dense, pos, s, h, f, cam, settings, width, height) else {
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
            let rel = [origin[0] - pos[0], origin[1] - pos[1], origin[2] - pos[2]];
            let ol = mat_apply(&minv, rel);
            let origin_local = [ol[0] + pivot[0], ol[1] + pivot[1], ol[2] + pivot[2]];
            let dir_local = mat_apply(&minv, dir);
            let Some((color, t)) = cast_local(dense, origin_local, dir_local) else {
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
    dense: &SpriteDense,
    pos: [f32; 3],
    s: [f32; 3],
    h: [f32; 3],
    f: [f32; 3],
    cam: &CameraState,
    settings: &OpticastSettings,
    width: u32,
    height: u32,
) -> Option<(u32, u32, u32, u32)> {
    let (xs, ys, zs) = (
        dense.dims[0] as f32,
        dense.dims[1] as f32,
        dense.dims[2] as f32,
    );
    let (xp, yp, zp) = (dense.pivot[0], dense.pivot[1], dense.pivot[2]);
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
                    pos[0] + lx * s[0] + ly * h[0] + lz * f[0],
                    pos[1] + lx * s[1] + ly * h[1] + lz * f[1],
                    pos[2] + lx * s[2] + ly * h[2] + lz * f[2],
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

/// CPU-side decoded animated voxel clip: every frame's [`SpriteDense`]
/// is cached at construction, so per-frame playback is a grid **select**
/// — not the per-frame voxel-volume decode [`draw_sprite_dda`] pays each
/// call. The CPU counterpart to the GPU flipbook (VCL.2). Build once from
/// a [`DecodedClip`], then [`draw_frame`](ClipFlipbook::draw_frame) the
/// active frame each render.
pub struct ClipFlipbook {
    frames: Vec<SpriteDense>,
}

impl ClipFlipbook {
    /// Decode + cache every frame of `clip` (one [`SpriteDense`] each).
    #[must_use]
    pub fn from_decoded(clip: &DecodedClip) -> Self {
        let frames = clip
            .frames
            .iter()
            .map(|frame| SpriteDense::from_voxel_frame(frame, clip.dims, clip.pivot))
            .collect();
        Self { frames }
    }

    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Borrow frame `frame`'s cached dense grid, if in range.
    #[must_use]
    pub fn frame(&self, frame: usize) -> Option<&SpriteDense> {
        self.frames.get(frame)
    }

    /// Draw frame `frame` at a world pose via [`draw_sprite_dense`] —
    /// `pos` is the world pivot, `s`/`h`/`f` the model→world basis columns.
    /// Returns pixels written (0 if `frame` is out of range).
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn draw_frame(
        &self,
        fb: &mut [u32],
        zb: &mut [f32],
        pitch_pixels: usize,
        width: u32,
        height: u32,
        cam: &CameraState,
        settings: &OpticastSettings,
        frame: usize,
        pos: [f32; 3],
        s: [f32; 3],
        h: [f32; 3],
        f: [f32; 3],
        flags: u32,
    ) -> u32 {
        let Some(dense) = self.frames.get(frame) else {
            return 0;
        };
        draw_sprite_dense(
            fb,
            zb,
            pitch_pixels,
            width,
            height,
            cam,
            settings,
            dense,
            pos,
            s,
            h,
            f,
            flags,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera_math;
    use crate::Camera;
    use roxlap_formats::kv6::Kv6;
    use roxlap_formats::sprite::Sprite;
    use roxlap_formats::voxel_clip::{LoopMode, VoxelClip, VoxelFrame};

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

    /// Build a [`VoxelFrame`] from a dense `fill(x,y,z) -> Option<color>`.
    fn clip_frame(dims: [u32; 3], fill: impl Fn(u32, u32, u32) -> Option<u32>) -> VoxelFrame {
        let owpc = dims[2].div_ceil(32).max(1) as usize;
        let cols = (dims[0] * dims[1]) as usize;
        let mut occupancy = vec![0u32; cols * owpc];
        let mut color_offsets = vec![0u32; cols + 1];
        let mut colors = Vec::new();
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                let col = (x + y * dims[0]) as usize;
                color_offsets[col] = colors.len() as u32;
                for z in 0..dims[2] {
                    if let Some(c) = fill(x, y, z) {
                        occupancy[col * owpc + (z >> 5) as usize] |= 1u32 << (z & 31);
                        colors.push(c);
                    }
                }
            }
        }
        color_offsets[cols] = colors.len() as u32;
        VoxelFrame {
            occupancy,
            colors,
            color_offsets,
        }
    }

    /// A cached [`ClipFlipbook`] draws distinct frames distinctly — the
    /// CPU flipbook select. Frame 0 fills the bottom half (red), frame 1
    /// the top half (green); rendered at the same pose they cover
    /// different screen pixels in different colours.
    #[test]
    fn clip_flipbook_frames_render_differently() {
        let dims = [8u32, 8, 8];
        let f0 = clip_frame(dims, |_x, _y, z| (z < 4).then_some(0x00FF_0000)); // red, low z
        let f1 = clip_frame(dims, |_x, _y, z| (z >= 4).then_some(0x0000_FF00)); // green, high z
        let clip = VoxelClip::from_frames(
            dims,
            [4.0, 4.0, 4.0],
            1.0,
            LoopMode::Loop,
            &[f0, f1],
            &[],
            33,
            0,
        );
        let decoded = clip.decode().expect("decode");
        let book = ClipFlipbook::from_decoded(&decoded);
        assert_eq!(book.frame_count(), 2);
        assert!(book.frame(0).is_some() && book.frame(2).is_none());

        let (w, h) = (64u32, 64u32);
        let n = (w * h) as usize;
        let cam = cam_looking_y();
        let cs = camera_math::derive(&cam, w, h, 32.0, 32.0, 32.0);
        let cfg = settings(w, h);
        let pose = [0.0, 40.0, 0.0];
        let (s, hh, f) = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);

        let render = |frame: usize| -> Vec<u32> {
            let mut fb = vec![0u32; n];
            let mut zb = vec![f32::INFINITY; n];
            let wrote = book.draw_frame(
                &mut fb, &mut zb, w as usize, w, h, &cs, &cfg, frame, pose, s, hh, f, 0,
            );
            assert!(wrote > 0, "frame {frame} should draw some pixels");
            fb
        };
        let fb0 = render(0);
        let fb1 = render(1);
        assert_ne!(fb0, fb1, "distinct frames must render distinct pixels");
        // Each frame shows its own channel: red present in frame 0, green
        // present in frame 1.
        assert!(fb0.iter().any(|&p| (p & 0x00FF_0000) != 0));
        assert!(fb1.iter().any(|&p| (p & 0x0000_FF00) != 0));
        // Out-of-range frame draws nothing.
        let mut fb = vec![0u32; n];
        let mut zb = vec![f32::INFINITY; n];
        assert_eq!(
            book.draw_frame(&mut fb, &mut zb, w as usize, w, h, &cs, &cfg, 9, pose, s, hh, f, 0),
            0
        );
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

    /// The covered screen rect (min/max px,py) of whatever the sprite
    /// painted — used to compare an axis-aligned vs a rotated pose.
    fn covered_rect(fb: &[u32], w: u32, h: u32) -> (u32, u32, u32, u32) {
        let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0u32, 0u32);
        for py in 0..h {
            for px in 0..w {
                if fb[(py * w + px) as usize] & 0x00ff_ffff != 0 {
                    x0 = x0.min(px);
                    y0 = y0.min(py);
                    x1 = x1.max(px);
                    y1 = y1.max(py);
                }
            }
        }
        (x0, y0, x1, y1)
    }

    /// A non-cube box drawn axis-aligned vs. drawn with a per-instance
    /// transform that swaps its long axis onto the screen's other axis
    /// flips the silhouette's aspect ratio. Pins that the `s/h/f` basis
    /// (the path `DynSpriteTransform` feeds) actually reorients the model.
    #[test]
    fn posed_basis_reorients_silhouette() {
        // Wide-in-local-x, short-in-local-z box → appears wide on screen
        // (screen-x = world-x via `right`, screen-y = world-z via `down`).
        let kv6 = Kv6::solid_box(16, 4, 4, 0x80_C0_40_20);
        let (w, h) = (64u32, 64u32);
        let n = (w * h) as usize;
        let cam = cam_looking_y();
        let cs = camera_math::derive(&cam, w, h, 32.0, 32.0, 32.0);

        // Axis-aligned: wide silhouette.
        let aa = Sprite::axis_aligned(kv6.clone(), [0.0, 40.0, 0.0]);
        let mut fb = vec![0u32; n];
        let mut zb = vec![f32::INFINITY; n];
        let _ = draw_sprite_dda(
            &mut fb,
            &mut zb,
            w as usize,
            w,
            h,
            &cs,
            &settings(w, h),
            &aa,
        );
        let (ax0, ay0, ax1, ay1) = covered_rect(&fb, w, h);
        let aa_wide = (ax1 - ax0) as i32 - (ay1 - ay0) as i32;
        assert!(
            aa_wide > 4,
            "axis-aligned box should be wider than tall (got w-h={aa_wide})"
        );

        // Posed: map local +x onto world +z and local +z onto world +x
        // (det = -1 ≠ 0). Same box now reads tall on screen.
        let mut posed = aa.clone();
        posed.s = [0.0, 0.0, 1.0]; // local +x ↦ world +z (screen down)
        posed.h = [0.0, 1.0, 0.0]; // local +y ↦ world +y (depth)
        posed.f = [1.0, 0.0, 0.0]; // local +z ↦ world +x (screen right)
        let mut fb2 = vec![0u32; n];
        let mut zb2 = vec![f32::INFINITY; n];
        let _ = draw_sprite_dda(
            &mut fb2,
            &mut zb2,
            w as usize,
            w,
            h,
            &cs,
            &settings(w, h),
            &posed,
        );
        let (bx0, by0, bx1, by1) = covered_rect(&fb2, w, h);
        let posed_tall = (by1 - by0) as i32 - (bx1 - bx0) as i32;
        assert!(
            posed_tall > 4,
            "posed box should be taller than wide (got h-w={posed_tall})"
        );
    }

    /// A degenerate (singular) basis — `det == 0` — makes the sprite
    /// silently skip rather than panic (the `DynSpriteTransform` guard).
    #[test]
    fn degenerate_basis_draws_nothing() {
        let kv6 = Kv6::solid_cube(8, 0x80_FF_FF_FF);
        let mut sprite = Sprite::axis_aligned(kv6, [0.0, 40.0, 0.0]);
        sprite.f = sprite.s; // two equal columns → det 0
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
        assert_eq!(wrote, 0, "singular basis must skip, not panic");
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
