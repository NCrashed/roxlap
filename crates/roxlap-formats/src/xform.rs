//! TRS bone transform + quaternion, for rich skeletal animation.
//!
//! A [`BoneXform`] is the per-bone, per-keyframe **local** transform the KFA
//! solver applies through a hinge: a translation, a quaternion rotation, and a
//! (non-uniform) scale. The legacy single-axis hinge angle is the special case
//! [`BoneXform::from_hinge_angle`] (translation 0, scale 1, rotation about the
//! hinge axis) — see `roxlap_core::kfa_draw::setlimb`.

/// A quaternion `(x, y, z, w)`, scalar `w` last. Rotations should be unit, but
/// the type itself doesn't enforce it ([`Quat::normalize`] / [`Quat::nlerp`]
/// renormalize).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quat {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Quat {
    /// The identity rotation.
    pub const IDENTITY: Quat = Quat {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 1.0,
    };

    /// Rotation by `rad` radians about `axis` (right-handed). A near-zero axis
    /// yields the identity.
    #[must_use]
    pub fn from_axis_angle(axis: [f32; 3], rad: f32) -> Quat {
        let len = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        if len < 1e-12 {
            return Quat::IDENTITY;
        }
        let (s, c) = (rad * 0.5).sin_cos();
        let k = s / len;
        Quat {
            x: axis[0] * k,
            y: axis[1] * k,
            z: axis[2] * k,
            w: c,
        }
    }

    /// Renormalize to unit length (identity if degenerate).
    #[must_use]
    pub fn normalize(self) -> Quat {
        let n = (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt();
        if n < 1e-12 {
            return Quat::IDENTITY;
        }
        Quat {
            x: self.x / n,
            y: self.y / n,
            z: self.z / n,
            w: self.w / n,
        }
    }

    /// Rotate vector `v` by this (assumed unit) quaternion.
    #[must_use]
    pub fn rotate(self, v: [f32; 3]) -> [f32; 3] {
        // v' = v + 2w·(q×v) + 2·q×(q×v)
        let (qx, qy, qz, qw) = (self.x, self.y, self.z, self.w);
        let tx = 2.0 * (qy * v[2] - qz * v[1]);
        let ty = 2.0 * (qz * v[0] - qx * v[2]);
        let tz = 2.0 * (qx * v[1] - qy * v[0]);
        [
            v[0] + qw * tx + (qy * tz - qz * ty),
            v[1] + qw * ty + (qz * tx - qx * tz),
            v[2] + qw * tz + (qx * ty - qy * tx),
        ]
    }

    /// Normalized lerp toward `b` by `t` along the shortest arc — a cheap
    /// slerp stand-in for blending keyframes (the engine lerps frame values).
    #[must_use]
    pub fn nlerp(self, b: Quat, t: f32) -> Quat {
        // Shortest arc: flip `b` if the quaternions point apart.
        let dot = self.x * b.x + self.y * b.y + self.z * b.z + self.w * b.w;
        let b = if dot < 0.0 {
            Quat {
                x: -b.x,
                y: -b.y,
                z: -b.z,
                w: -b.w,
            }
        } else {
            b
        };
        Quat {
            x: self.x + (b.x - self.x) * t,
            y: self.y + (b.y - self.y) * t,
            z: self.z + (b.z - self.z) * t,
            w: self.w + (b.w - self.w) * t,
        }
        .normalize()
    }

    /// Build a rotation from intrinsic ZYX Euler angles (radians): roll about
    /// X, then pitch about Y, then yaw about Z (`R = Rz·Ry·Rx`). The inverse of
    /// [`Self::to_euler`]. For free 3-DOF authoring from three numeric fields.
    #[must_use]
    pub fn from_euler(roll_x: f32, pitch_y: f32, yaw_z: f32) -> Quat {
        Quat::from_axis_angle([0.0, 0.0, 1.0], yaw_z)
            * Quat::from_axis_angle([0.0, 1.0, 0.0], pitch_y)
            * Quat::from_axis_angle([1.0, 0.0, 0.0], roll_x)
    }

    /// Decompose to intrinsic ZYX Euler angles `[roll_x, pitch_y, yaw_z]`
    /// (radians) — the inverse of [`Self::from_euler`]. At a pitch of ±90° the
    /// X/Z split is ambiguous (gimbal lock); the result still rebuilds the same
    /// rotation.
    #[must_use]
    pub fn to_euler(self) -> [f32; 3] {
        let Quat { x, y, z, w } = self;
        let roll = (2.0 * (w * x + y * z)).atan2(1.0 - 2.0 * (x * x + y * y));
        let sin_pitch = (2.0 * (w * y - z * x)).clamp(-1.0, 1.0);
        let pitch = sin_pitch.asin();
        let yaw = (2.0 * (w * z + x * y)).atan2(1.0 - 2.0 * (y * y + z * z));
        [roll, pitch, yaw]
    }
}

/// Hamilton product `self * rhs` (apply `rhs` first, then `self`).
impl core::ops::Mul for Quat {
    type Output = Quat;
    fn mul(self, r: Quat) -> Quat {
        Quat {
            w: self.w * r.w - self.x * r.x - self.y * r.y - self.z * r.z,
            x: self.w * r.x + self.x * r.w + self.y * r.z - self.z * r.y,
            y: self.w * r.y - self.x * r.z + self.y * r.w + self.z * r.x,
            z: self.w * r.z + self.x * r.y - self.y * r.x + self.z * r.w,
        }
    }
}

/// A bone's local transform for one keyframe: translation, rotation, scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoneXform {
    /// Translation, added to the hinge's child-side anchor (velcro-frame local).
    pub t: [f32; 3],
    /// Rotation of the bone about its joint.
    pub r: Quat,
    /// Non-uniform scale along the bone's local axes (x, y, z).
    pub s: [f32; 3],
}

impl BoneXform {
    /// No translation, no rotation, unit scale.
    pub const IDENTITY: BoneXform = BoneXform {
        t: [0.0, 0.0, 0.0],
        r: Quat::IDENTITY,
        s: [1.0, 1.0, 1.0],
    };

    /// The legacy single-axis hinge pose: rotate about `axis` by the Q15 angle
    /// `val` (full circle = 65536), no translation, unit scale. The sign
    /// reproduces voxlap's `setlimb` (which rotates the velcro frame by the
    /// `[[cos,-sin],[sin,cos]]` matrix — a `-θ` rotation about the axis in
    /// quaternion terms).
    #[must_use]
    pub fn from_hinge_angle(axis: [f32; 3], val: i16) -> BoneXform {
        let rad = f32::from(val) * (core::f32::consts::PI * 2.0 / 65536.0);
        BoneXform {
            t: [0.0, 0.0, 0.0],
            r: Quat::from_axis_angle(axis, -rad),
            s: [1.0, 1.0, 1.0],
        }
    }

    /// Recover the legacy Q15 hinge angle about `axis` — the inverse of
    /// [`Self::from_hinge_angle`]. Projects the rotation onto `axis`, so it's
    /// exact for a rotation-only xform about that axis and an approximation for
    /// a free rotation (used to bridge to the i16 storage that doesn't yet hold
    /// translation / scale / off-axis rotation).
    #[must_use]
    pub fn hinge_angle(self, axis: [f32; 3]) -> i16 {
        let len = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        if len < 1e-12 {
            return 0;
        }
        let s = (self.r.x * axis[0] + self.r.y * axis[1] + self.r.z * axis[2]) / len;
        let phi = 2.0 * s.atan2(self.r.w); // signed rotation about `axis`
        let val = -phi * (65536.0 / (core::f32::consts::PI * 2.0));
        // Wrap into i16 range the same way voxlap's `short` assignment does.
        val.round() as i32 as i16
    }

    /// Blend toward `b` by `t ∈ [0, 1]`: lerp translation / scale, nlerp
    /// rotation. Used to interpolate keyframes during playback.
    #[must_use]
    pub fn blend(self, b: BoneXform, t: f32) -> BoneXform {
        let lerp3 = |x: [f32; 3], y: [f32; 3]| {
            [
                x[0] + (y[0] - x[0]) * t,
                x[1] + (y[1] - x[1]) * t,
                x[2] + (y[2] - x[2]) * t,
            ]
        };
        BoneXform {
            t: lerp3(self.t, b.t),
            r: self.r.nlerp(b.r, t),
            s: lerp3(self.s, b.s),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: [f32; 3], b: [f32; 3]) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() < 1e-5)
    }

    #[test]
    fn axis_angle_rotates_in_the_right_hand_sense() {
        // +90° about +z sends +x to +y.
        let q = Quat::from_axis_angle([0.0, 0.0, 1.0], core::f32::consts::FRAC_PI_2);
        assert!(approx(q.rotate([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0]));
        // The axis itself is invariant.
        assert!(approx(q.rotate([0.0, 0.0, 1.0]), [0.0, 0.0, 1.0]));
    }

    #[test]
    fn identity_and_normalize() {
        assert!(approx(
            Quat::IDENTITY.rotate([3.0, -2.0, 5.0]),
            [3.0, -2.0, 5.0]
        ));
        let n = Quat {
            x: 0.0,
            y: 0.0,
            z: 2.0,
            w: 0.0,
        }
        .normalize();
        assert!((n.x * n.x + n.y * n.y + n.z * n.z + n.w * n.w - 1.0).abs() < 1e-6);
    }

    #[test]
    fn nlerp_hits_the_endpoints() {
        let a = Quat::IDENTITY;
        let b = Quat::from_axis_angle([0.0, 1.0, 0.0], 1.0);
        assert!((a.nlerp(b, 0.0).w - a.w).abs() < 1e-6);
        let end = a.nlerp(b, 1.0);
        assert!((end.w - b.w).abs() < 1e-6 && (end.y - b.y).abs() < 1e-6);
    }

    #[test]
    fn hinge_angle_inverts_from_hinge_angle() {
        let axis = [0.0, 0.0, 1.0];
        for val in [0i16, 1000, 16384, -16384, 30000, -30000] {
            let got = BoneXform::from_hinge_angle(axis, val).hinge_angle(axis);
            assert!(
                (i32::from(got) - i32::from(val)).abs() <= 1,
                "{val} -> {got}"
            );
        }
    }

    #[test]
    fn euler_round_trips_away_from_gimbal_lock() {
        // A generic orientation (pitch well clear of ±90°) round-trips.
        let e = [0.3_f32, -0.7, 1.1];
        let got = Quat::from_euler(e[0], e[1], e[2]).to_euler();
        assert!(
            (0..3).all(|i| (got[i] - e[i]).abs() < 1e-4),
            "{got:?} vs {e:?}"
        );
        // from_euler about a single axis matches a plain axis-angle.
        let qz = Quat::from_euler(0.0, 0.0, core::f32::consts::FRAC_PI_2);
        assert!(approx(qz.rotate([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0]));
    }

    #[test]
    fn mul_composes_rotations() {
        // Two +45° about z = one +90° about z.
        let h = Quat::from_axis_angle([0.0, 0.0, 1.0], core::f32::consts::FRAC_PI_4);
        let full = h * h;
        assert!(approx(full.rotate([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0]));
    }
}
