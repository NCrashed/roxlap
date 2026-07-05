//! Dynamic lighting (stages DL + SL) — runtime sun + point + spot
//! lights with stylized voxel shadows, on **both** backends (the GPU
//! shaders since DL, the CPU renderer since CPU.1/CPU.2). See
//! `PORTING-DYNLIGHT.md` + `PORTING-SPOTLIGHT.md`.
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

/// A spot (cone) light — a [`PointLight`] with a direction and a soft
/// angular cutoff, so it lights only a cone. World space. Internally folded
/// into the point-light path (a point light is the 180°-cone degenerate), so
/// spots share the point-light count + shadow-caster budgets.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpotLight {
    /// World-space position (voxel units).
    pub position: [f32; 3],
    /// Cone **axis** — the direction the light shines **along** (the way the
    /// light travels). Need not be normalized; the backend normalizes.
    pub direction: [f32; 3],
    /// Linear RGB, `0..1`.
    pub color: [f32; 3],
    /// Scalar brightness multiplier on `color`.
    pub intensity: f32,
    /// Hard cutoff distance in world units; past it the light is zero.
    pub radius: f32,
    /// Inner half-angle in **degrees**: within it the cone is at full
    /// brightness. Clamped to `0..=outer_angle_deg`.
    pub inner_angle_deg: f32,
    /// Outer half-angle in **degrees**: past it the cone is zero; between the
    /// two it soft-falls off (`smoothstep`). `outer == inner` ⇒ a hard edge;
    /// `>= 180` ⇒ omnidirectional (a plain point light). Clamped to `0..=180`.
    pub outer_angle_deg: f32,
    /// Whether this spot casts a stylized hard shadow (shares the
    /// shadow-caster cap with the sun + point lights; excess demoted with a
    /// warning, never truncated).
    pub casts_shadow: bool,
}

impl Default for SpotLight {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            direction: [0.0, 0.0, 1.0],
            color: [1.0; 3],
            intensity: 1.0,
            radius: 32.0,
            inner_angle_deg: 20.0,
            outer_angle_deg: 30.0,
            casts_shadow: false,
        }
    }
}

impl SpotLight {
    /// The clamped `(inner, outer)` half-angles in degrees (`0 <= inner <=
    /// outer <= 180`). Shared by both backends so the cone is identical.
    fn clamped_angles(&self) -> (f32, f32) {
        let outer = self.outer_angle_deg.clamp(0.0, 180.0);
        let inner = self.inner_angle_deg.clamp(0.0, outer);
        (inner, outer)
    }

    /// The normalized world-space cone axis (travel direction). `[0,0,1]`
    /// if `direction` is degenerate.
    pub(crate) fn axis(&self) -> [f32; 3] {
        let d = self.direction;
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        if len > 1e-6 {
            [d[0] / len, d[1] / len, d[2] / len]
        } else {
            [0.0, 0.0, 1.0]
        }
    }

    /// Cosine of the clamped inner half-angle (`>= cos_outer`).
    pub(crate) fn cos_inner(&self) -> f32 {
        self.clamped_angles().0.to_radians().cos()
    }

    /// Cosine of the clamped outer half-angle. `<= -0.999` (a `>= ~177°`
    /// cone) makes the shader/CPU skip the cone mask — an omnidirectional
    /// point light.
    pub(crate) fn cos_outer(&self) -> f32 {
        self.clamped_angles().1.to_radians().cos()
    }
}

/// The whole per-frame light environment, borrowed into
/// [`crate::FrameParams`]. Applied by **both** backends (GPU since DL,
/// CPU since CPU.1/CPU.2) — see the module docs.
#[derive(Clone, Copy, Debug)]
pub struct LightRig<'a> {
    /// The sun. `None` ⇒ no directional light this frame.
    pub sun: Option<DirectionalLight>,
    /// Point lights. The backend caps the count (excess dropped with a
    /// log warning), so this slice may be longer than the GPU honours.
    pub points: &'a [PointLight],
    /// Spot (cone) lights. Folded into the point-light path, so they share
    /// the point-light count + shadow-caster budgets (points take priority).
    pub spots: &'a [SpotLight],
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
    /// DL.6 — **stylized lighting**. `0` ⇒ smooth (physically-ish) diffuse
    /// (the default). `≥1` ⇒ retro cel look: the sun's key term and each
    /// point light's diffuse factor quantize to `bands + 1` discrete levels
    /// (terraced light instead of a smooth gradient), and the banded sun key
    /// drives a **gradient map** from [`shadow_tint`](Self::shadow_tint)
    /// (cool, unlit) to the sun's colour (warm, lit) — hue-shifted shadows
    /// rather than plain darkening. Avoids the "generic Phong" read that
    /// flattens the voxel/retro identity.
    pub bands: u32,
    /// DL.6 — the cool shadow/ambient tint the stylized ramp starts from
    /// (the unlit end). Multiplied by the baked ambient/AO byte. Ignored
    /// when `bands == 0` (then [`ambient`](Self::ambient) is used instead).
    pub shadow_tint: [f32; 3],
}

impl Default for LightRig<'_> {
    fn default() -> Self {
        Self {
            sun: None,
            points: &[],
            spots: &[],
            ambient: [1.0; 3],
            shadow_strength: 0.7,
            shadow_bias_voxels: 1.5,
            shadow_max_dist: 512.0,
            bands: 0,
            shadow_tint: [0.12, 0.14, 0.2],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spot_angles_convert_and_clamp() {
        // cos is decreasing in angle ⇒ inner (smaller angle) has the larger
        // cosine; the two bracket the smoothstep the backends run.
        let s = SpotLight {
            inner_angle_deg: 20.0,
            outer_angle_deg: 30.0,
            ..SpotLight::default()
        };
        assert!(s.cos_inner() > s.cos_outer());
        assert!((s.cos_inner() - 20.0f32.to_radians().cos()).abs() < 1e-6);

        // inner > outer is clamped to outer ⇒ a hard edge (equal cosines).
        let hard = SpotLight {
            inner_angle_deg: 40.0,
            outer_angle_deg: 25.0,
            ..SpotLight::default()
        };
        assert!((hard.cos_inner() - hard.cos_outer()).abs() < 1e-6);

        // A 180° outer cone reads as an omnidirectional point (mask skipped:
        // the backends treat `cos_outer <= -0.999` as "not a spot").
        let wide = SpotLight {
            outer_angle_deg: 180.0,
            ..SpotLight::default()
        };
        assert!(wide.cos_outer() <= -0.999);
    }

    #[test]
    fn spot_axis_normalizes() {
        let s = SpotLight {
            direction: [0.0, 0.0, 5.0],
            ..SpotLight::default()
        };
        assert_eq!(s.axis(), [0.0, 0.0, 1.0]);
        // Degenerate direction falls back to a defined axis.
        let z = SpotLight {
            direction: [0.0; 3],
            ..SpotLight::default()
        };
        assert_eq!(z.axis(), [0.0, 0.0, 1.0]);
    }
}
