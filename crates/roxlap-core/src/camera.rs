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
    fn default() -> Self {
        Self {
            pos: [1024.0, 1024.0, 128.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 0.0, 1.0],
            forward: [0.0, 1.0, 0.0],
        }
    }
}
