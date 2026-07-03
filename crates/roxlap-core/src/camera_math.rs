//! Per-frame camera state for the pinhole projection the renderer uses.
//!
//! A [`Camera`] carries an `f64` world position and an orthonormal
//! `right` / `down` / `forward` basis. Rendering happens in `f32`, so
//! [`derive`](fn@derive) narrows the basis once per frame and precomputes the four
//! view-frustum corner ray directions.
//!
//! ## Pinhole model
//!
//! A pixel `(px, py)` maps to a ray direction
//!
//! ```text
//! dir = (px - hx)·right + (py - hy)·down + hz·forward
//! ```
//!
//! where `(hx, hy)` is the projection centre in pixels and `hz` the
//! focal length. `(hx, hy, hz) = (w/2, h/2, w/2)` gives a 90° horizontal
//! field of view with square pixels. The same expression evaluated at
//! the four screen corners yields [`CameraState::corn`].

use crate::Camera;

/// Per-frame `f32` camera state derived from a [`Camera`] + the screen
/// projection parameters.
#[derive(Debug, Clone, Copy)]
pub struct CameraState {
    /// Camera position in world voxel units (narrowed from `f64`).
    pub pos: [f32; 3],
    /// Orthonormal basis — screen `+x`, screen `+y`, and view direction.
    pub right: [f32; 3],
    pub down: [f32; 3],
    pub forward: [f32; 3],
    /// View-frustum corner ray directions, in screen order: top-left,
    /// top-right, bottom-right, bottom-left. `corn[0]` is the direction
    /// of pixel `(0, 0)`.
    pub corn: [[f32; 3]; 4],
}

/// Derive the per-frame [`CameraState`] for an `xres × yres` framebuffer
/// with projection centre `(hx, hy)` and focal length `hz`.
//
// `f64 → f32` narrows the basis to the render precision; `u32 → f32` for
// the framebuffer dimensions is exact for any realistic screen (≤ 16M,
// within f32's 24-bit mantissa). Both are intentional.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_lossless
)]
#[must_use]
pub fn derive(camera: &Camera, xres: u32, yres: u32, hx: f32, hy: f32, hz: f32) -> CameraState {
    let pos = camera.pos.map(|v| v as f32);
    let right = camera.right.map(|v| v as f32);
    let down = camera.down.map(|v| v as f32);
    let forward = camera.forward.map(|v| v as f32);

    // Ray direction for screen pixel (sx, sy), relative to the
    // projection centre: sx·right + sy·down + hz·forward.
    let ray = |sx: f32, sy: f32| {
        [
            sx * right[0] + sy * down[0] + hz * forward[0],
            sx * right[1] + sy * down[1] + hz * forward[1],
            sx * right[2] + sy * down[2] + hz * forward[2],
        ]
    };

    let (w, h) = (xres as f32, yres as f32);
    // Corners in screen-pixel coords offset by the projection centre:
    // (0,0), (w,0), (w,h), (0,h).
    let corn = [
        ray(-hx, -hy),
        ray(w - hx, -hy),
        ray(w - hx, h - hy),
        ray(-hx, h - hy),
    ];

    CameraState {
        pos,
        right,
        down,
        forward,
        corn,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bit-pattern compare for `[f32; 3]` — the test inputs are
    /// integer-valued so results are bit-stable, but `clippy::float_cmp`
    /// still objects to `==`.
    fn bits3(a: [f32; 3]) -> [u32; 3] {
        a.map(f32::to_bits)
    }

    fn identity_cam() -> Camera {
        Camera {
            pos: [0.0, 0.0, 0.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 0.0, 1.0],
            forward: [0.0, 1.0, 0.0],
        }
    }

    #[test]
    fn identity_camera_basis_and_corners() {
        let s = derive(&identity_cam(), 640, 480, 320.0, 240.0, 320.0);
        assert_eq!(bits3(s.right), bits3([1.0, 0.0, 0.0]));
        assert_eq!(bits3(s.down), bits3([0.0, 0.0, 1.0]));
        assert_eq!(bits3(s.forward), bits3([0.0, 1.0, 0.0]));
        // corn[0] = -320·right - 240·down + 320·forward = [-320, 320, -240]
        assert_eq!(bits3(s.corn[0]), bits3([-320.0, 320.0, -240.0]));
        // corn[1] = +640 on right from corn[0] = [320, 320, -240]
        assert_eq!(bits3(s.corn[1]), bits3([320.0, 320.0, -240.0]));
        // corn[2] = +480 on down from corn[1] = [320, 320, 240]
        assert_eq!(bits3(s.corn[2]), bits3([320.0, 320.0, 240.0]));
        // corn[3] = +480 on down from corn[0] = [-320, 320, 240]
        assert_eq!(bits3(s.corn[3]), bits3([-320.0, 320.0, 240.0]));
    }

    #[test]
    fn yawed_camera_corner_propagates() {
        // Yaw 90°: right = +y, forward = -x.
        let cam = Camera {
            pos: [0.0, 0.0, 0.0],
            right: [0.0, 1.0, 0.0],
            down: [0.0, 0.0, 1.0],
            forward: [-1.0, 0.0, 0.0],
        };
        let s = derive(&cam, 640, 480, 320.0, 240.0, 320.0);
        // corn[0] = -320·[0,1,0] - 240·[0,0,1] + 320·[-1,0,0] = [-320, -320, -240]
        assert_eq!(bits3(s.corn[0]), bits3([-320.0, -320.0, -240.0]));
    }

    #[test]
    fn position_is_narrowed_through() {
        let cam = Camera {
            pos: [10.5, 20.25, 30.0],
            ..identity_cam()
        };
        let s = derive(&cam, 64, 64, 32.0, 32.0, 32.0);
        assert_eq!(bits3(s.pos), bits3([10.5, 20.25, 30.0]));
    }
}
