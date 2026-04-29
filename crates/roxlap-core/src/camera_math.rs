//! Per-frame camera-derived state — port of voxlaptest's `setcamera`
//! (`voxlap5.c:2246`).
//!
//! Inputs:
//! - [`Camera`] with `f64` position and `f64` orthonormal right / down
//!   / forward basis (matching `dpoint3d` in voxlap5.h).
//! - `xres` / `yres`: framebuffer dimensions in pixels.
//! - `hx` / `hy` / `hz`: screen-projection parameters voxlap calls
//!   `dahx` / `dahy` / `dahz`. The oracle and most demos use
//!   `(width/2, height/2, width/2)` — that's a 90° horizontal FOV with
//!   square pixels.
//!
//! Outputs (all `f32` — voxlap narrows once at the start of every
//! frame and does the rest of the per-frame math at single precision):
//! - `pos` / `right` / `down` / `forward`: copies of the camera basis.
//! - `xs` / `ys` / `zs`: the transposed basis (column-vectors of the
//!   3×3 rotation), useful for projecting world deltas into camera-
//!   relative `(right, down, forward)` coordinates.
//! - `add`: `-pos · {right, down, forward}` — the translation half of
//!   the world → camera transform.
//! - `corn[4]`: the four corners of the view frustum as direction
//!   vectors from the camera. `corn[0]` = top-left, `corn[1]` =
//!   top-right, `corn[2]` = bottom-right, `corn[3]` = bottom-left.
//! - `nor[4]`: cross-products of consecutive corner pairs, giving
//!   inward-facing frustum-edge normals (`nor[i] = corn[i] ×
//!   corn[(i+1) % 4]`). The grouscan ray-stepper uses these to clip
//!   columns against the visible rectangle.

use crate::Camera;

/// Per-frame state derived from a [`Camera`] + screen parameters.
#[derive(Debug, Clone, Copy)]
pub struct CameraState {
    /// Camera position in world voxel units, narrowed from `f64`.
    pub pos: [f32; 3],
    /// Camera basis — `right` / `down` / `forward`.
    pub right: [f32; 3],
    pub down: [f32; 3],
    pub forward: [f32; 3],
    /// Transposed basis: row-major if camera basis was column-major.
    /// `xs.0 = right.x`, `xs.1 = down.x`, `xs.2 = forward.x`, etc.
    pub xs: [f32; 3],
    pub ys: [f32; 3],
    pub zs: [f32; 3],
    /// `add[k] = -dot(pos, basis[k])`, the translation half of the
    /// world → camera-relative-coords transform.
    pub add: [f32; 3],
    /// Four-corner view-frustum direction vectors.
    pub corn: [[f32; 3]; 4],
    /// Frustum edge normals: `nor[i] = corn[i] × corn[(i + 1) % 4]`.
    pub nor: [[f32; 3]; 4],
}

/// Derive the per-frame [`CameraState`].
///
/// Mirrors voxlaptest's `setcamera(ipo, ist, ihe, ifo, dahx, dahy, dahz)`
/// (`voxlap5.c:2246`) bit-exactly, including the `f64 → f32` narrowing
/// at the call boundary.
//
// clippy::cast_possible_truncation fires on `f64 as f32` even though
// voxlap's setcamera narrows in the same direction, and on
// `u32 as f32` for the framebuffer dimensions (xres/yres are
// realistically far below f32's 24-bit mantissa cap of ~16M, so no
// precision is lost in any practical screen size). Both casts are
// load-bearing for matching voxlap's per-frame f32 math; allowing
// here is the right call.
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

    let xs = [right[0], down[0], forward[0]];
    let ys = [right[1], down[1], forward[1]];
    let zs = [right[2], down[2], forward[2]];

    let add = [
        -(pos[0] * right[0] + pos[1] * right[1] + pos[2] * right[2]),
        -(pos[0] * down[0] + pos[1] * down[1] + pos[2] * down[2]),
        -(pos[0] * forward[0] + pos[1] * forward[1] + pos[2] * forward[2]),
    ];

    let xres_f = xres as f32;
    let yres_f = yres as f32;

    // gcorn[0] = -hx·right − hy·down + hz·forward
    let c0 = [
        -hx * right[0] - hy * down[0] + hz * forward[0],
        -hx * right[1] - hy * down[1] + hz * forward[1],
        -hx * right[2] - hy * down[2] + hz * forward[2],
    ];
    // gcorn[1] = xres·right + gcorn[0]
    let c1 = [
        xres_f * right[0] + c0[0],
        xres_f * right[1] + c0[1],
        xres_f * right[2] + c0[2],
    ];
    // gcorn[2] = yres·down + gcorn[1]
    let c2 = [
        yres_f * down[0] + c1[0],
        yres_f * down[1] + c1[1],
        yres_f * down[2] + c1[2],
    ];
    // gcorn[3] = yres·down + gcorn[0]
    let c3 = [
        yres_f * down[0] + c0[0],
        yres_f * down[1] + c0[1],
        yres_f * down[2] + c0[2],
    ];
    let corn = [c0, c1, c2, c3];

    // nor[i] = corn[i] × corn[(i + 1) % 4]  — inward-facing edge normals.
    let mut nor = [[0.0f32; 3]; 4];
    for i in 0..4 {
        let a = corn[i];
        let b = corn[(i + 1) % 4];
        nor[i] = [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ];
    }

    CameraState {
        pos,
        right,
        down,
        forward,
        xs,
        ys,
        zs,
        add,
        corn,
        nor,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bit-pattern compare for `[f32; 3]`. The reference values in
    /// these tests are integer-valued floats produced from
    /// integer-valued inputs through addition / multiplication, so
    /// they are bit-equal across compilers — but `clippy::float_cmp`
    /// still rightly objects to `==`. Compare bit patterns instead.
    fn bits3(a: [f32; 3]) -> [u32; 3] {
        a.map(f32::to_bits)
    }

    /// Identity-basis camera at the origin → all derivations are
    /// integer-exact and easy to verify by hand.
    #[test]
    fn identity_camera_origin_basic_derivations() {
        let cam = Camera {
            pos: [0.0, 0.0, 0.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 0.0, 1.0],
            forward: [0.0, 1.0, 0.0],
        };
        let s = derive(&cam, 640, 480, 320.0, 240.0, 320.0);

        // Basis copied through unchanged.
        assert_eq!(bits3(s.right), bits3([1.0, 0.0, 0.0]));
        assert_eq!(bits3(s.down), bits3([0.0, 0.0, 1.0]));
        assert_eq!(bits3(s.forward), bits3([0.0, 1.0, 0.0]));

        // Transpose — pull the X / Y / Z component out of each basis vector.
        assert_eq!(bits3(s.xs), bits3([1.0, 0.0, 0.0]));
        assert_eq!(bits3(s.ys), bits3([0.0, 0.0, 1.0]));
        assert_eq!(bits3(s.zs), bits3([0.0, 1.0, 0.0]));

        // pos at origin → translation is zero. The actual computation
        // is `-(0 + 0 + 0) = -0.0` in IEEE-754 (negating any zero
        // yields the negative-signed zero); voxlap's setcamera does
        // the same, so the bit-pattern is `-0.0` not `+0.0`.
        assert_eq!(bits3(s.add), bits3([-0.0, -0.0, -0.0]));

        // gcorn[0] = -320·right - 240·down + 320·forward
        //          = -320·[1,0,0] - 240·[0,0,1] + 320·[0,1,0]
        //          = [-320, 320, -240]
        assert_eq!(bits3(s.corn[0]), bits3([-320.0, 320.0, -240.0]));
        // gcorn[1] = +xres·right + gcorn[0] = [320, 320, -240]
        assert_eq!(bits3(s.corn[1]), bits3([320.0, 320.0, -240.0]));
        // gcorn[2] = +yres·down + gcorn[1] = [320, 320, 240]
        assert_eq!(bits3(s.corn[2]), bits3([320.0, 320.0, 240.0]));
        // gcorn[3] = +yres·down + gcorn[0] = [-320, 320, 240]
        assert_eq!(bits3(s.corn[3]), bits3([-320.0, 320.0, 240.0]));
    }

    #[test]
    fn identity_camera_frustum_edge_normals() {
        let cam = Camera {
            pos: [0.0, 0.0, 0.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 0.0, 1.0],
            forward: [0.0, 1.0, 0.0],
        };
        let s = derive(&cam, 640, 480, 320.0, 240.0, 320.0);

        // nor[0] = corn[0] × corn[1]
        //        = [-320, 320, -240] × [320, 320, -240]
        //        = (320*-240 - -240*320, -240*320 - -320*-240, -320*320 - 320*320)
        //        = (-76800 + 76800, -76800 - 76800, -102400 - 102400)
        //        = (0, -153600, -204800)
        assert_eq!(bits3(s.nor[0]), bits3([0.0, -153_600.0, -204_800.0]));
        // nor[2] = corn[2] × corn[3]: opposite quadrant, opposite sign
        // on the y-axis = [320, 320, 240] × [-320, 320, 240]
        //        = (320*240 - 240*320, 240*-320 - 320*240, 320*320 - 320*-320)
        //        = (76800 - 76800, -76800 - 76800, 102400 + 102400)
        //        = (0, -153600, 204800)
        assert_eq!(bits3(s.nor[2]), bits3([0.0, -153_600.0, 204_800.0]));
    }

    #[test]
    fn translated_camera_add_is_negative_dot() {
        let cam = Camera {
            pos: [10.0, 20.0, 30.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 0.0, 1.0],
            forward: [0.0, 1.0, 0.0],
        };
        let s = derive(&cam, 640, 480, 320.0, 240.0, 320.0);

        // add[0] = -dot(pos, right) = -10
        // add[1] = -dot(pos, down)  = -30
        // add[2] = -dot(pos, forward) = -20
        assert_eq!(bits3(s.add), bits3([-10.0, -30.0, -20.0]));
        // corn[0] is independent of pos.
        assert_eq!(bits3(s.corn[0]), bits3([-320.0, 320.0, -240.0]));
    }

    #[test]
    fn yawed_camera_basis_propagates() {
        // Yaw 90° clockwise: right was +x, now becomes +y.
        // forward was +y, now becomes -x.
        let cam = Camera {
            pos: [0.0, 0.0, 0.0],
            right: [0.0, 1.0, 0.0],
            down: [0.0, 0.0, 1.0],
            forward: [-1.0, 0.0, 0.0],
        };
        let s = derive(&cam, 640, 480, 320.0, 240.0, 320.0);

        // corn[0] = -hx·right - hy·down + hz·forward
        //         = -320·[0,1,0] - 240·[0,0,1] + 320·[-1,0,0]
        //         = [-320, -320, -240]
        assert_eq!(bits3(s.corn[0]), bits3([-320.0, -320.0, -240.0]));
        // xs / ys / zs reflect the new basis layout.
        assert_eq!(bits3(s.xs), bits3([0.0, 0.0, -1.0])); // right.x, down.x, forward.x
        assert_eq!(bits3(s.ys), bits3([1.0, 0.0, 0.0]));
        assert_eq!(bits3(s.zs), bits3([0.0, 1.0, 0.0]));
    }
}
