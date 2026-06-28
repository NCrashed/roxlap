//! Dynamic lighting (stage DL) — runtime sun + point lights + stylized
//! voxel shadows. **GPU-only**: the CPU rasterizer ignores everything
//! here (it keeps multiplying the baked ambient byte). See
//! `PORTING-DYNLIGHT.md`.
//!
//! Lighting is per-frame: a host builds a [`LightRig`] each frame and
//! hands it to the renderer via [`crate::FrameParams::lights`]. There are
//! deliberately no stateful lighting setters on [`crate::SceneRenderer`]
//! (mirroring how sky / fog / side-shades already flow through
//! [`crate::FrameParams`]). `lights: None` ⇒ exactly the pre-DL render.

/// The sun — a single directional light for the whole scene. World space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DirectionalLight {
    /// Direction the light **travels** (from the sun toward the scene),
    /// world space. Need not be normalized — the backend normalizes. The
    /// N·L term uses the negation (the direction *to* the sun).
    pub direction: [f32; 3],
    /// Linear RGB, `0..1` (may exceed 1 for extra punch; the output is
    /// clamped). `[1.0; 3]` is neutral white.
    pub color: [f32; 3],
    /// Scalar brightness multiplier on `color`.
    pub intensity: f32,
    /// Whether this light casts stylized hard shadows (DL.3). The sun is
    /// the natural first shadow caster.
    pub casts_shadow: bool,
}

impl Default for DirectionalLight {
    fn default() -> Self {
        Self {
            direction: [0.0, 0.0, 1.0],
            color: [1.0; 3],
            intensity: 1.0,
            casts_shadow: false,
        }
    }
}

/// A colored point light. World space, with a hard radius cutoff (it
/// contributes nothing beyond `radius`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointLight {
    /// World-space position (voxel units).
    pub position: [f32; 3],
    /// Linear RGB, `0..1`.
    pub color: [f32; 3],
    /// Scalar brightness multiplier on `color`.
    pub intensity: f32,
    /// Hard cutoff distance in world units; past it the light is zero.
    pub radius: f32,
    /// Whether this light casts stylized hard shadows (DL.3). Only the
    /// first few flagged lights actually cast (see the renderer's
    /// shadow-caster cap); the rest are silently demoted to shadowless
    /// with a log warning — never truncated.
    pub casts_shadow: bool,
}

impl Default for PointLight {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            color: [1.0; 3],
            intensity: 1.0,
            radius: 32.0,
            casts_shadow: false,
        }
    }
}

/// The whole per-frame light environment, borrowed into
/// [`crate::FrameParams`]. GPU-only.
#[derive(Clone, Copy, Debug)]
pub struct LightRig<'a> {
    /// The sun. `None` ⇒ no directional light this frame.
    pub sun: Option<DirectionalLight>,
    /// Point lights. The backend caps the count (excess dropped with a
    /// log warning), so this slice may be longer than the GPU honours.
    pub points: &'a [PointLight],
    /// Multiplier applied to the baked ambient byte — the global ambient
    /// level / tint. `[1.0; 3]` uses the baked byte as-is.
    pub ambient: [f32; 3],
    /// Stylized-shadow strength `0..1`: the fraction of a caster's light
    /// removed in shadow (`1.0` = full black, `0.0` = no visible shadow).
    pub shadow_strength: f32,
    /// Shadow-ray origin bias along the surface normal, in voxel units —
    /// kills self-shadow acne. ~1.5 is a good default.
    pub shadow_bias_voxels: f32,
    /// Sun shadow-ray length cap, world units (point-light shadow rays
    /// stop at the light instead).
    pub shadow_max_dist: f32,
}

impl Default for LightRig<'_> {
    fn default() -> Self {
        Self {
            sun: None,
            points: &[],
            ambient: [1.0; 3],
            shadow_strength: 0.7,
            shadow_bias_voxels: 1.5,
            shadow_max_dist: 512.0,
        }
    }
}
