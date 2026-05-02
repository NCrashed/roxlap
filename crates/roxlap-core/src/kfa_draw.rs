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
//! Voxlap's full animation system (`animsprite` + `seq[]` +
//! `frmval[]` interpolation) is **not** ported here — the host
//! drives `kfaval[]` directly each frame, which is enough for
//! procedural animation. Adding `animsprite` later would let
//! `.kfa` files with baked frame curves play back unchanged.
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

use roxlap_formats::kfa::{Hinge, Point3};

use crate::camera_math::CameraState;
use crate::opticast::OpticastSettings;
use crate::sprite::{draw_sprite, mat2, DrawTarget, Sprite, SpriteLighting};

/// Voxlap's `genperp` — given a non-zero axis vector `a`, build
/// two orthonormal vectors `b`, `c` such that `(a, b, c)` form a
/// right-handed orthonormal basis with `a` along its primary axis.
/// Mirror of voxlap5.c:9546-9561.
///
/// If `a` is zero, returns `([0; 3], [0; 3])` (matches voxlap's
/// degenerate-input zeroing).
#[must_use]
pub fn genperp(a: [f32; 3]) -> ([f32; 3], [f32; 3]) {
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

/// Build the hinge-sort order — voxlap's `kfasorthinge`
/// (voxlap5.c:9427-9450). The result is an array of hinge
/// indices ordered such that **walking from index `n-1` down to
/// 0** visits parents before children (= valid topological
/// order for the chain of `setlimb` calls in `kfa_draw`).
///
/// Voxlap mutates the hinges in place during sort and restores
/// them; our port produces the same `hsort` array without
/// touching the hinges.
#[must_use]
pub fn sort_hinges(hinges: &[Hinge]) -> Vec<usize> {
    let n = hinges.len();
    let mut hsort = vec![0usize; n];
    // First pass: roots at the end, non-roots at the start.
    let mut head = 0usize;
    let mut tail = n;
    for i in (0..n).rev() {
        if hinges[i].parent < 0 {
            tail -= 1;
            hsort[tail] = i;
        } else {
            hsort[head] = i;
            head += 1;
        }
    }

    // `solved[h]` = true once hinge h's parent has been settled
    // into the "tail" half. Voxlap encodes this in-place by
    // flipping the parent field to -2-parent; we use a side
    // bitmap to leave the input immutable.
    let mut solved = vec![false; n];
    for i in (tail..n).rev() {
        solved[hsort[i]] = true;
    }

    // Iterative pass: pick non-root entries in head whose parent
    // is already solved; move them to the tail.
    let mut idx = head; // idx walks the head [0..head) backward
    while tail > 0 {
        if idx == 0 {
            idx = head;
        }
        idx -= 1;
        let j = hsort[idx];
        let parent = hinges[j].parent;
        if parent < 0 {
            // Already in the tail (shouldn't happen since the
            // first pass sorted these out).
            continue;
        }
        if solved[parent as usize] {
            solved[j] = true;
            tail -= 1;
            hsort[idx] = hsort[tail];
            hsort[tail] = j;
            head -= 1;
        }
        if head == 0 {
            break;
        }
    }
    hsort
}

/// One animated KFA sprite — bones + hinges + per-bone live
/// animation values. The host owns one of these per animated
/// model, updates `kfaval[]` over time, and passes it to
/// [`draw_kfa_sprite`] each frame.
#[derive(Clone)]
pub struct KfaSprite {
    /// One [`Sprite`] per bone. Limb `i`'s `(s, h, f, p)` is
    /// computed per frame in [`draw_kfa_sprite`] from the parent's
    /// transform + hinge math; the `kv6` field holds the bone's
    /// kv6 mesh and never changes.
    pub limbs: Vec<Sprite>,
    /// Bone hierarchy. Mirror of voxlap's `kfatype.hinge[]`.
    pub hinges: Vec<Hinge>,
    /// Topological sort of bone indices — populated once at
    /// construction, used by [`draw_kfa_sprite`]'s render loop.
    pub hinge_sort: Vec<usize>,
    /// Per-bone animation value. Voxlap's `vx5.kfaval[]`. Q15
    /// angle (full circle = 65536). Host updates per frame.
    pub kfaval: Vec<i16>,
    /// World-space anchor of the root limb's `hinge.p[0]`. The
    /// root limb is positioned so `hinge.p[0]` lands at this
    /// point given the world basis below.
    pub p: [f32; 3],
    /// World-space basis for the root limb. Mirror of
    /// `vx5sprite.{s, h, f}` for the root.
    pub s: [f32; 3],
    pub h: [f32; 3],
    pub f: [f32; 3],
}

impl KfaSprite {
    /// Build a KFA sprite from a list of `(Sprite, Hinge)` bones.
    /// `limbs.len()` must equal `hinges.len()`. The first bone with
    /// `parent < 0` is the root.
    ///
    /// `kfaval` is initialised to all zeros; the host should set
    /// per-bone angles before / between [`draw_kfa_sprite`] calls.
    #[must_use]
    pub fn new(limbs: Vec<Sprite>, hinges: Vec<Hinge>, root_pos: [f32; 3]) -> Self {
        assert_eq!(
            limbs.len(),
            hinges.len(),
            "limbs ({}) and hinges ({}) length mismatch",
            limbs.len(),
            hinges.len()
        );
        let n = hinges.len();
        let hinge_sort = sort_hinges(&hinges);
        Self {
            limbs,
            hinges,
            hinge_sort,
            kfaval: vec![0i16; n],
            p: root_pos,
            s: [1.0, 0.0, 0.0],
            h: [0.0, 1.0, 0.0],
            f: [0.0, 0.0, 1.0],
        }
    }
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
    // Voxlap iterates `for i = numhin-1; i >= 0; i--`; sort_hinges
    // puts parents at high indices, so descending iteration walks
    // parents first.
    let n = kfa.hinge_sort.len();
    let mut total: u32 = 0;
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
        // Render the limb's kv6 mesh.
        total += draw_sprite(target, cam, settings, lighting, &kfa.limbs[j]);
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use roxlap_formats::kfa::Point3;

    fn axis(x: f32, y: f32, z: f32) -> Point3 {
        Point3 { x, y, z }
    }

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

    /// sort_hinges puts roots at high indices and children at low.
    /// 3-bone chain: root → child1 → child2.
    #[test]
    fn sort_hinges_three_bone_chain() {
        let h = |parent: i32| Hinge {
            parent,
            p: [axis(0.0, 0.0, 0.0); 2],
            v: [axis(1.0, 0.0, 0.0); 2],
            vmin: 0,
            vmax: 0,
            htype: 0,
            filler: [0; 7],
        };
        // hinge[0] = root, hinge[1] child of 0, hinge[2] child of 1.
        let hinges = vec![h(-1), h(0), h(1)];
        let sort = sort_hinges(&hinges);
        // High index = root → 0 must be at index 2 (= hinges.len()-1).
        // Walking sort backward should visit: root (0), child of root
        // (1), child of 1 (2).
        // The exact ordering of the lower indices depends on voxlap's
        // sweep, but we can verify "walking sort[i] for i=n-1..=0 gives
        // an order where every bone's parent appears earlier".
        let mut seen = vec![false; 3];
        for k in (0..3).rev() {
            let j = sort[k];
            seen[j] = true;
            let p = hinges[j].parent;
            if p >= 0 {
                assert!(
                    seen[p as usize],
                    "bone {j}'s parent {p} not yet visited at descent step k={k}"
                );
            }
        }
    }
}
