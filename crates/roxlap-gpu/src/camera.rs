//! World-space camera state passed to [`crate::GpuRenderer::render_scene`].
//!
//! Mirrors `roxlap-core::Camera`'s shape (position + orthonormal
//! right / down / forward basis) but in `f32` since the GPU does its
//! ray math at single precision. Host bridges by casting f64 → f32
//! after computing chunk-local coordinates.

/// World-space camera state in the voxlap convention (Z = down).
///
/// `right`, `down`, `forward` form a right-handed orthonormal basis;
/// `position` is in voxel-world units. `fov_y_rad` is the vertical
/// field-of-view in radians — voxlap's default is roughly 60°
/// (`std::f32::consts::FRAC_PI_3`).
#[derive(Debug, Clone, Copy)]
pub struct Camera {
    /// Eye position in world voxel units (1 voxel = 1 world unit).
    pub position: [f32; 3],
    /// Unit basis vector toward screen-right. Must satisfy
    /// `right × down == forward` (right-handed) or frustum culling
    /// silently rejects sprites.
    pub right: [f32; 3],
    /// Unit basis vector toward screen-down. In the voxlap convention
    /// +z points down (z = 0 sky, z = 255 bedrock), so a level camera
    /// has `down = [0, 0, 1]`.
    pub down: [f32; 3],
    /// Unit view direction (into the screen); the third leg of the
    /// right-handed `right`/`down`/`forward` basis.
    pub forward: [f32; 3],
    /// Vertical field of view in radians, `(0, π)`. Default is
    /// `FRAC_PI_3` (≈60°, voxlap's default); the horizontal FOV
    /// follows from the render aspect ratio.
    pub fov_y_rad: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 0.0, 1.0],
            forward: [0.0, 1.0, 0.0],
            fov_y_rad: std::f32::consts::FRAC_PI_3,
        }
    }
}
