//! World-space camera state passed to [`crate::GpuRenderer::render_chunk`].
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
    pub position: [f32; 3],
    pub right: [f32; 3],
    pub down: [f32; 3],
    pub forward: [f32; 3],
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
