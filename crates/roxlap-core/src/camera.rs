//! Camera state — position + orthonormal basis.
//!
//! Mirrors voxlaptest's `setcamera(ipo, ist, ihe, ifo, ...)` ABI: a
//! starting point and three orthonormal axes (right, down, forward).
//! The convention matches the voxlap C engine so a `.vxl` file's
//! camera vectors load directly.

/// Camera state. All vectors are in voxel-world units (1 unit = 1
/// voxel); the basis is right-handed with `down` aligned to +z (i.e.
/// z grows downward into the map, matching voxlap's coordinate
/// system).
///
/// # Examples
///
/// ```
/// use roxlap_core::Camera;
///
/// let cam = Camera::default();
/// assert_eq!(cam.pos, [1024.0, 1024.0, 128.0]);
/// assert_eq!(cam.forward, [0.0, 1.0, 0.0]); // looking +y (north)
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera {
    /// Camera position (`ipo` / `dpoint3d` in voxlaptest).
    pub pos: [f64; 3],
    /// Right vector (`ist`).
    pub right: [f64; 3],
    /// Down vector (`ihe`).
    pub down: [f64; 3],
    /// Forward vector (`ifo`).
    pub forward: [f64; 3],
}

impl Default for Camera {
    /// Centred at the middle of a 2048-VSID map, looking +y, level.
    /// Matches the placeholder vectors voxlaptest's oracle writes
    /// into the `.vxl` header (see `tests/oracle/oracle.c`).
    ///
    /// **Caution:** this placeholder basis is *left-handed*
    /// (`right × down = -forward`). It is fine for the world raycaster,
    /// but the sprite frustum cull requires the canonical right-handed
    /// chirality (`right × down = +forward`) and will reject every
    /// sprite under this basis. For an interactive camera build the
    /// basis with [`Camera::from_yaw_pitch`] / [`Camera::orbit`] /
    /// [`Camera::look_at`] instead of starting from `default()` and
    /// rotating it. See the [crate-level handedness
    /// notes](crate#world-handedness-and-the-horizontal-mirror).
    fn default() -> Self {
        Self {
            pos: [1024.0, 1024.0, 128.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 0.0, 1.0],
            forward: [0.0, 1.0, 0.0],
        }
    }
}

impl Camera {
    /// Build a camera from a position plus `yaw` / `pitch` (radians),
    /// with no roll.
    ///
    /// This is the **canonical** voxlap-convention constructor: it
    /// reproduces `oracle.c::set_camera_yaw_pitch` bit-for-bit, so the
    /// frustum cull (which requires `right × down = +forward`) accepts
    /// sprites and the render matches the bit-exact oracle goldens.
    /// Any project that hand-rolls `right = [-sin yaw, cos yaw, 0]`
    /// should call this instead — that hand-rolled form is exactly
    /// this basis, and copying it by hand is the usual source of
    /// chirality mistakes.
    ///
    /// `yaw` sweeps the world's horizontal (x/y) plane: `yaw = 0` looks
    /// down `+x`, increasing `yaw` turns toward `+y`. `pitch` tilts
    /// toward `+z` (down): `pitch = 0` is level, positive pitch aims
    /// downward.
    ///
    /// See the [crate-level handedness
    /// notes](crate#world-handedness-and-the-horizontal-mirror) for why
    /// the rendered image is horizontally mirrored versus a real camera
    /// and what to do if you need an un-mirrored world.
    ///
    /// # Examples
    /// ```
    /// use roxlap_core::Camera;
    ///
    /// let cam = Camera::from_yaw_pitch([1024.0, 1024.0, 128.0], 0.0, 0.0);
    /// // yaw = 0, pitch = 0 → looking down +x.
    /// assert_eq!(cam.forward, [1.0, 0.0, 0.0]);
    /// assert_eq!(cam.right, [0.0, 1.0, 0.0]);
    /// assert_eq!(cam.down, [0.0, 0.0, 1.0]);
    ///
    /// // Canonical chirality: right × down == +forward. The sprite
    /// // frustum cull depends on this; `Camera::default`'s placeholder
    /// // basis does *not* satisfy it.
    /// let cross = [
    ///     cam.right[1] * cam.down[2] - cam.right[2] * cam.down[1],
    ///     cam.right[2] * cam.down[0] - cam.right[0] * cam.down[2],
    ///     cam.right[0] * cam.down[1] - cam.right[1] * cam.down[0],
    /// ];
    /// assert_eq!(cross, cam.forward);
    /// ```
    #[must_use]
    pub fn from_yaw_pitch(pos: [f64; 3], yaw: f64, pitch: f64) -> Self {
        let (sy, cy) = yaw.sin_cos();
        let (sp, cp) = pitch.sin_cos();
        Self {
            pos,
            right: [-sy, cy, 0.0],
            down: [-cy * sp, -sy * sp, cp],
            forward: [cy * cp, sy * cp, sp],
        }
    }

    /// Orbit camera: look *at* `center` from `dist` voxels away, at the
    /// given `yaw` / `pitch`. Heading conventions match
    /// [`Camera::from_yaw_pitch`]; the position is placed `dist` behind
    /// the forward axis so `center` sits on the view ray.
    ///
    /// # Examples
    /// ```
    /// use roxlap_core::Camera;
    ///
    /// // yaw = 0 → forward = +x, so the eye sits at center − dist·(+x).
    /// let cam = Camera::orbit(0.0, 0.0, 100.0, [1024.0, 1024.0, 128.0]);
    /// assert_eq!(cam.forward, [1.0, 0.0, 0.0]);
    /// assert_eq!(cam.pos, [924.0, 1024.0, 128.0]);
    /// ```
    #[must_use]
    pub fn orbit(yaw: f64, pitch: f64, dist: f64, center: [f64; 3]) -> Self {
        let mut cam = Self::from_yaw_pitch([0.0; 3], yaw, pitch);
        cam.pos = [
            center[0] - cam.forward[0] * dist,
            center[1] - cam.forward[1] * dist,
            center[2] - cam.forward[2] * dist,
        ];
        cam
    }

    /// Look from `eye` toward `target`, with no roll (the `right` axis
    /// stays in the world's horizontal plane). Produces the same
    /// canonical chirality as [`Camera::from_yaw_pitch`] by recovering
    /// yaw / pitch from the look direction.
    ///
    /// If `eye == target` (or the look direction is purely vertical),
    /// yaw collapses to `0`; the resulting basis is still orthonormal
    /// and correctly handed.
    ///
    /// # Examples
    /// ```
    /// use roxlap_core::Camera;
    ///
    /// // Looking from x=924 toward x=1024 (i.e. down +x).
    /// let cam = Camera::look_at([924.0, 1024.0, 128.0], [1024.0, 1024.0, 128.0]);
    /// assert_eq!(cam.forward, [1.0, 0.0, 0.0]);
    /// assert_eq!(cam.right, [0.0, 1.0, 0.0]);
    /// ```
    #[must_use]
    pub fn look_at(eye: [f64; 3], target: [f64; 3]) -> Self {
        let fx = target[0] - eye[0];
        let fy = target[1] - eye[1];
        let fz = target[2] - eye[2];
        let yaw = fy.atan2(fx);
        let horiz = (fx * fx + fy * fy).sqrt();
        let pitch = fz.atan2(horiz);
        Self::from_yaw_pitch(eye, yaw, pitch)
    }
}
