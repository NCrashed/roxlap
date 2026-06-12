//! KFA (animated kv6) sprite renderer — voxlap's `kfadraw`
//! (`voxlap5.c:9759`) plus the bone-transform helpers
//! `genperp` (voxlap5.c:9546), `mat0` (voxlap5.c:9568), and
//! `setlimb` (voxlap5.c:9643).
//!
//! A KFA sprite is a hierarchy of bones; each bone carries an
//! optional [`Sprite`] (= one kv6 limb) plus a hinge tying it to
//! its parent. Per frame, the user updates `kfaval[i]` (a 16-bit
//! angle) for each animated bone; this module walks the hinge
//! tree in topological order and computes each limb's world
//! transform from the parent's, then dispatches the existing
//! [`crate::sprite::draw_sprite`] per limb to rasterise the kv6
//! data.
//!
//! Voxlap's animation-curve playback (`animsprite` + `seq[]` +
//! `frmval[]` interpolation) lives in
//! [`roxlap_formats::kfa::KfaSprite::animsprite`] — call it to
//! advance `kfaval[]` from a baked curve, or poke `kfaval[]`
//! directly for procedural animation, then render here. The bone
//! posing this module computes is also exposed standalone as
//! [`solve_kfa_limbs`] for non-CPU backends (the GPU sprite pass).
//!
//! No oracle pose exercises KFA, so this module's correctness
//! gate is "looks right + tests verify the bone math". We can
//! tighten validation when a real `.kfa` asset lands.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::missing_panics_doc,
    clippy::float_cmp,
    clippy::useless_vec
)]

use roxlap_formats::kfa::{Hinge, KfaSprite, Point3};
use roxlap_formats::sprite::Sprite;

use crate::camera_math::CameraState;
use crate::opticast::OpticastSettings;
use crate::sprite::{draw_sprite, mat2, DrawTarget, SpriteLighting};

/// Voxlap's `genperp` — given a non-zero axis vector `a`, build
/// two orthonormal vectors `b`, `c` such that `(a, b, c)` form a
/// right-handed orthonormal basis with `a` along its primary axis.
/// Mirror of voxlap5.c:9546-9561.
///
/// If `a` is zero, returns `([0; 3], [0; 3])` (matches voxlap's
/// degenerate-input zeroing).
fn genperp(a: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    if a == [0.0, 0.0, 0.0] {
        return ([0.0; 3], [0.0; 3]);
    }
    // Pick the smallest-magnitude axis to zero out in `b`, so the
    // remaining two components dominate and the cross-product
    // `c = a × b` stays well-conditioned.
    let ax = a[0].abs();
    let ay = a[1].abs();
    let az = a[2].abs();
    let b = if ax < ay && ax < az {
        let t = 1.0 / (a[1] * a[1] + a[2] * a[2]).sqrt();
        [0.0, a[2] * t, -a[1] * t]
    } else if ay < az {
        let t = 1.0 / (a[0] * a[0] + a[2] * a[2]).sqrt();
        [-a[2] * t, 0.0, a[0] * t]
    } else {
        let t = 1.0 / (a[0] * a[0] + a[1] * a[1]).sqrt();
        [a[1] * t, -a[0] * t, 0.0]
    };
    let c = [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
    (b, c)
}

/// Voxlap's `mat0` (`voxlap5.c:9568`) — given `B` and `C` such
/// that `A * B = C`, find `A`. Returns `(a_s, a_h, a_f, a_o)`.
///
/// Used by `setlimb` to find the rotation matrix that maps the
/// parent's hinge frame to the child's hinge frame.
fn mat0(
    b_s: [f32; 3],
    b_h: [f32; 3],
    b_f: [f32; 3],
    b_o: [f32; 3],
    c_s: [f32; 3],
    c_h: [f32; 3],
    c_f: [f32; 3],
    c_o: [f32; 3],
) -> ([f32; 3], [f32; 3], [f32; 3], [f32; 3]) {
    // A's columns = C's columns expressed in B's frame.
    // Voxlap evaluates as `bs.row * c.col + bh.row * c.col + bg.row * c.col`
    // for each (row, output-col) pair.
    let ts = [
        b_s[0] * c_s[0] + b_h[0] * c_h[0] + b_f[0] * c_f[0],
        b_s[0] * c_s[1] + b_h[0] * c_h[1] + b_f[0] * c_f[1],
        b_s[0] * c_s[2] + b_h[0] * c_h[2] + b_f[0] * c_f[2],
    ];
    let th = [
        b_s[1] * c_s[0] + b_h[1] * c_h[0] + b_f[1] * c_f[0],
        b_s[1] * c_s[1] + b_h[1] * c_h[1] + b_f[1] * c_f[1],
        b_s[1] * c_s[2] + b_h[1] * c_h[2] + b_f[1] * c_f[2],
    ];
    let tf = [
        b_s[2] * c_s[0] + b_h[2] * c_h[0] + b_f[2] * c_f[0],
        b_s[2] * c_s[1] + b_h[2] * c_h[1] + b_f[2] * c_f[1],
        b_s[2] * c_s[2] + b_h[2] * c_h[2] + b_f[2] * c_f[2],
    ];
    let to = [
        c_o[0] - b_o[0] * ts[0] - b_o[1] * th[0] - b_o[2] * tf[0],
        c_o[1] - b_o[0] * ts[1] - b_o[1] * th[1] - b_o[2] * tf[1],
        c_o[2] - b_o[0] * ts[2] - b_o[1] * th[2] - b_o[2] * tf[2],
    ];
    (ts, th, tf, to)
}

#[inline]
fn pt(p: Point3) -> [f32; 3] {
    [p.x, p.y, p.z]
}

/// Convert voxlap's i16 hinge angle (full circle = 65536) to
/// `(cos, sin)` floats. Voxlap C uses `ucossin(((int32_t)val)<<16,
/// ...)`, a precomputed 256-entry table. Our port uses libm
/// `cos/sin` — no oracle pose exercises this code so the small
/// precision difference vs voxlap C's table is acceptable.
fn cossin_q15(val: i16) -> [f32; 2] {
    let ang = (i32::from(val) as f32) * (std::f32::consts::PI * 2.0 / 65536.0);
    [ang.cos(), ang.sin()]
}

/// Voxlap's `setlimb` (`voxlap5.c:9643`) — compute child limb
/// `i`'s world transform from parent limb `p`'s world transform
/// via the hinge connecting them.
///
/// Math:
/// 1. Build the child-side velcro frame from the hinge: `qp =
///    hinge.p[0]`, `(qs, qh, qf) = (hinge.v[0], genperp(hinge.v[0]))`.
/// 2. Apply the hinge transform — for `htype == 0` (rotate around
///    `qs` by `val` angle), rotate `(qh, qf)` in their plane.
/// 3. Build the parent-side velcro frame: `pp = hinge.p[1]`,
///    `(ps, ph, pf) = (hinge.v[1], genperp(hinge.v[1]))`.
/// 4. `mat0`: find `R` such that `R * (ps, ph, pf, pp) = (qs, qh,
///    qf, qp)` — `R` is the limb's hinge rotation in parent
///    coords.
/// 5. `mat2`: `child_world = parent_world * R`. The limb's `(s,
///    h, f, p)` transform updates in place.
fn setlimb(limbs: &mut [Sprite], hinges: &[Hinge], i: usize, parent: usize, htype: u8, val: i16) {
    let hinge = &hinges[i];

    // Step 1: child-side velcro frame.
    let qp = pt(hinge.p[0]);
    let mut qs = pt(hinge.v[0]);
    let (mut qh, mut qf) = genperp(qs);

    // Step 2: apply hinge transform.
    if htype == 0 {
        // Rotate (qh, qf) around qs by `val` (Q15 angle).
        let r = cossin_q15(val);
        let ph = qh;
        let pf = qf;
        qh = [
            ph[0] * r[0] - pf[0] * r[1],
            ph[1] * r[0] - pf[1] * r[1],
            ph[2] * r[0] - pf[2] * r[1],
        ];
        qf = [
            ph[0] * r[1] + pf[0] * r[0],
            ph[1] * r[1] + pf[1] * r[0],
            ph[2] * r[1] + pf[2] * r[0],
        ];
    } else {
        // Voxlap's only documented htype is 0; non-0 hits a MSVC
        // `__assume(0)`. Treat as no rotation.
        // (Suppress the unused `qs` warning if we add types later.)
        let _ = (&mut qs, &mut qh, &mut qf);
    }

    // Step 3: parent-side velcro frame.
    let pp = pt(hinge.p[1]);
    let ps = pt(hinge.v[1]);
    let (ph, pf) = genperp(ps);

    // Step 4: mat0 — find R such that R * (ps,ph,pf,pp) = (qs,qh,qf,qp).
    // R reuses the (qs, qh, qf, qp) variables in voxlap's C code.
    let (rs, rh, rf, ro) = mat0(ps, ph, pf, pp, qs, qh, qf, qp);

    // Step 5: mat2 — child_world = parent_world * R.
    let parent_s = limbs[parent].s;
    let parent_h = limbs[parent].h;
    let parent_f = limbs[parent].f;
    let parent_p = limbs[parent].p;
    let (cs, ch, cf, co) = mat2(parent_s, parent_h, parent_f, parent_p, rs, rh, rf, ro);
    let child = &mut limbs[i];
    child.s = cs;
    child.h = ch;
    child.f = cf;
    child.p = co;
}

/// Render an animated KFA sprite — voxlap's `kfadraw`
/// (voxlap5.c:9759). Walks the bone tree in topological order
/// (parents first), computes each limb's world transform from
/// the parent's via the per-limb `setlimb` walk, then dispatches
/// [`crate::sprite::draw_sprite`] per limb to rasterise its kv6.
///
/// Returns the total number of pixels written across all limbs.
pub fn draw_kfa_sprite(
    target: &mut DrawTarget<'_>,
    cam: &CameraState,
    settings: &OpticastSettings,
    lighting: &SpriteLighting<'_>,
    kfa: &mut KfaSprite,
) -> u32 {
    // Pose first, then rasterise. Voxlap interleaves the two in one
    // descending loop, but a limb's transform depends only on its
    // (already-posed) parent — never on drawing — so a full solve pass
    // followed by a full draw pass is identical, and lets non-CPU
    // backends reuse `solve_kfa_limbs` verbatim.
    solve_kfa_limbs(kfa);
    let n = kfa.hinge_sort.len();
    let mut total: u32 = 0;
    for k in (0..n).rev() {
        let j = kfa.hinge_sort[k];
        total += draw_sprite(target, cam, settings, lighting, &kfa.limbs[j]);
    }
    total
}

/// Pose every limb of a KFA sprite — the bone-transform half of
/// [`draw_kfa_sprite`], without rasterising. Walks the hinge tree in
/// topological order (parents first) and writes each limb's world
/// `(s, h, f, p)` from its parent's via the per-limb `setlimb` math,
/// reading the current [`KfaSprite::kfaval`] angles. Mirror of the
/// transform portion of voxlap's `kfadraw` (voxlap5.c:9759).
///
/// Split out so non-CPU backends (e.g. the GPU instanced-sprite pass)
/// can run the exact same posing and then consume `kfa.limbs[*]`
/// transforms however they need. The host typically calls
/// [`KfaSprite::animsprite`](roxlap_formats::kfa::KfaSprite::animsprite)
/// to advance `kfaval[]` first, then this to resolve world transforms.
pub fn solve_kfa_limbs(kfa: &mut KfaSprite) {
    // Voxlap iterates `for i = numhin-1; i >= 0; i--`; sort_hinges
    // puts parents at high indices, so descending iteration walks
    // parents first.
    let n = kfa.hinge_sort.len();
    for k in (0..n).rev() {
        let j = kfa.hinge_sort[k];
        let parent = kfa.hinges[j].parent;
        if parent >= 0 {
            // Child bone: derive transform from parent.
            let htype = kfa.hinges[j].htype;
            let val = kfa.kfaval[j];
            setlimb(&mut kfa.limbs, &kfa.hinges, j, parent as usize, htype, val);
        } else {
            // Root bone: copy world basis from KfaSprite + apply
            // hinge.p[0] as the velcro offset (voxlap5.c:9772-9782).
            let s = kfa.s;
            let h = kfa.h;
            let f = kfa.f;
            let p_world = kfa.p;
            let tp = pt(kfa.hinges[j].p[0]);
            let limb = &mut kfa.limbs[j];
            limb.s = s;
            limb.h = h;
            limb.f = f;
            limb.p = [
                p_world[0] - tp[0] * s[0] - tp[1] * h[0] - tp[2] * f[0],
                p_world[1] - tp[0] * s[1] - tp[1] * h[1] - tp[2] * f[1],
                p_world[2] - tp[0] * s[2] - tp[1] * h[2] - tp[2] * f[2],
            ];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// genperp produces an orthonormal basis with the input axis
    /// as the dominant direction.
    #[test]
    fn genperp_orthonormal() {
        let a = [1.0_f32, 0.0, 0.0];
        let (b, c) = genperp(a);
        // b · a = 0
        assert!((a[0] * b[0] + a[1] * b[1] + a[2] * b[2]).abs() < 1e-6);
        // c · a = 0
        assert!((a[0] * c[0] + a[1] * c[1] + a[2] * c[2]).abs() < 1e-6);
        // b · c = 0
        assert!((b[0] * c[0] + b[1] * c[1] + b[2] * c[2]).abs() < 1e-6);
        // |b| ≈ 1
        let lb = b[0] * b[0] + b[1] * b[1] + b[2] * b[2];
        assert!((lb - 1.0).abs() < 1e-5, "|b|² = {lb}");
        // |c| ≈ 1
        let lc = c[0] * c[0] + c[1] * c[1] + c[2] * c[2];
        assert!((lc - 1.0).abs() < 1e-5, "|c|² = {lc}");
    }

    /// genperp of zero vector returns zero vectors.
    #[test]
    fn genperp_zero() {
        let (b, c) = genperp([0.0, 0.0, 0.0]);
        assert_eq!(b, [0.0, 0.0, 0.0]);
        assert_eq!(c, [0.0, 0.0, 0.0]);
    }
}
