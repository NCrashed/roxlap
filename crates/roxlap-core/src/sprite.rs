//! KV6 sprite type + the `draw_sprite` dispatcher.
//!
//! Mirror of voxlap's `vx5sprite` (voxlap5.h:63-79) plus the
//! `drawsprite` entry point (voxlap5.c:9818). For R6.1 the
//! dispatcher is a stub — just enough API surface for the host to
//! plumb a sprite reference through. R6.2-R6.4 fill in the actual
//! kv6 frustum-cull + per-voxel rasterization behind it.
//!
//! Voxlap's vx5sprite is a 64-byte struct:
//!
//! ```text
//! point3d p;       // position
//! int32_t flags;   // bit 0: 0=normal shading
//!                  // bit 1: 0=kv6data, 1=kfatype  (oracle uses 0)
//!                  // bit 2: 0=normal, 1=invisible
//! point3d s;       // x-basis (kv6data.xsiz direction)
//! kv6data *voxnum; // (or kfatype *kfaptr if flag bit 1 set)
//! point3d h;       // y-basis
//! int32_t kfatim;
//! point3d f;       // z-basis
//! int32_t okfatim;
//! ```
//!
//! For R6 we only handle kv6 sprites with `flags = 0` (the four
//! oracle sprite poses all use this). KFA animation + the no-z and
//! invisible flags are deferred.

// The kv6draw port is pointer-arithmetic-heavy; the casts mirror C's
// implicit i32/u32/usize narrowings. Loop bounds are clamped via
// `lbound` so sign-loss / wrap is guarded at the type-system edge.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::similar_names
)]

use roxlap_formats::kv6::{Kv6, Voxel};

use crate::camera_math::CameraState;
use crate::fixed::ftol;

/// Voxlap's `vx5.kv6mipfactor` default (`voxlap5.c:12335`). Threshold
/// distance (in voxlap's "ftol-of-forward-projected" estimate units)
/// above which kv6draw walks the lowermip chain. Roxlap doesn't yet
/// model the lowermip chain in `roxlap-formats::Kv6`, so the mip
/// descent loop in [`kv6_draw_prepare`] is structurally faithful but
/// effectively a no-op until that lands.
pub const KV6_MIPFACTOR_DEFAULT: i32 = 128;

/// Voxlap's sprite-flags bit 0: disable normal-based face shading.
pub const SPRITE_FLAG_NO_SHADING: u32 = 1 << 0;
/// Voxlap's sprite-flags bit 1: voxnum points at a `kfatype`
/// (animated). When clear (default), points at a `kv6data`.
pub const SPRITE_FLAG_KFA: u32 = 1 << 1;
/// Voxlap's sprite-flags bit 2: skip rendering entirely.
pub const SPRITE_FLAG_INVISIBLE: u32 = 1 << 2;
/// Voxlap's sprite-flags bit 3: render without z-buffer test.
pub const SPRITE_FLAG_NO_Z: u32 = 1 << 3;

/// A KV6 voxel sprite positioned in world space.
///
/// Mirror of voxlap's `vx5sprite` for the kv6 case (`flags &
/// SPRITE_FLAG_KFA == 0`). Owns its `Kv6` by value — the sprite
/// data the engine needs to access during rendering. `p` / `s` /
/// `h` / `f` are voxlap's per-axis world-space basis: `s` is
/// the `kv6.xsiz` direction, `h` the `ysiz` direction, `f` the
/// `zsiz` direction. For an axis-aligned sprite, `s = [1,0,0]`,
/// `h = [0,1,0]`, `f = [0,0,1]`.
#[derive(Debug, Clone)]
pub struct Sprite {
    /// Voxel data + bounding-box pivots. Built either by
    /// [`crate::meltsphere::meltsphere`] (extracted from a world)
    /// or loaded from a `.kv6` file via `roxlap_formats::kv6::parse`.
    pub kv6: Kv6,
    /// World-space position of the sprite's pivot (xpiv, ypiv,
    /// zpiv inside the kv6 maps to this point).
    pub p: [f32; 3],
    /// World-space basis vector for the kv6's local +x. Length
    /// scales the sprite along that axis (typically `1.0` for
    /// unit-scale).
    pub s: [f32; 3],
    /// World-space basis vector for the kv6's local +y.
    pub h: [f32; 3],
    /// World-space basis vector for the kv6's local +z.
    pub f: [f32; 3],
    /// Voxlap-style flags bitfield. See `SPRITE_FLAG_*` constants.
    pub flags: u32,
}

impl Sprite {
    /// Convenience constructor for an axis-aligned sprite at
    /// world position `pos`. Basis is identity, flags = 0
    /// (kv6 + normal shading + visible + z-tested).
    #[must_use]
    pub fn axis_aligned(kv6: Kv6, pos: [f32; 3]) -> Self {
        Self {
            kv6,
            p: pos,
            s: [1.0, 0.0, 0.0],
            h: [0.0, 1.0, 0.0],
            f: [0.0, 0.0, 1.0],
            flags: 0,
        }
    }
}

/// Post-cull state derived from a sprite + camera pair — what the
/// per-voxel iteration in R6.3+ needs to start its setup. Borrows
/// the mip-selected kv6 from the sprite.
///
/// Voxlap doesn't materialise this struct (it operates on local
/// variables inside `kv6draw`); roxlap factors the cull out so it's
/// independently testable without staging the rest of the
/// rasterizer.
#[derive(Debug, Clone)]
#[allow(dead_code)] // R6.3+ will read these fields.
pub(crate) struct Kv6DrawSetup<'a> {
    /// Mip-selected kv6. For the base-mip case (always, today),
    /// this is just `&sprite.kv6`.
    pub kv: &'a Kv6,
    /// Mip-scaled basis vectors. For the base mip these equal
    /// `sprite.s/h/f`; if a future lowermip walk runs, each is
    /// scaled by `2^mip`.
    pub ts: [f32; 3],
    pub th: [f32; 3],
    pub tf: [f32; 3],
    /// 0 for the base mip; reserved for lowermip support.
    pub mip: u32,
}

/// Mip-LOD descent + 4-plane frustum cull, mirror of voxlap5.c:8832-
/// 8875. Returns `None` if the sprite's bound cube is fully behind
/// any of the four view-frustum edge planes (`CameraState::nor`),
/// `Some(setup)` otherwise with the post-cull state R6.3 needs.
///
/// # Cull math
///
/// The bound cube has centre `npos` (in camera-relative coords) and
/// three half-extent vectors `nstr`, `nhei`, `nfor` (each = the
/// kv6-axis basis vector scaled by the corresponding half-extent).
/// For each frustum-edge normal `n`, voxlap tests:
///
/// ```text
/// |nstr · n| + |nhei · n| + |nfor · n| + npos · n < 0
/// ```
///
/// — i.e. the cube's closest-point projection onto `n` is still
/// behind the plane. Any plane satisfying this culls the sprite.
pub(crate) fn kv6_draw_prepare<'a>(
    sprite: &'a Sprite,
    cam: &CameraState,
) -> Option<Kv6DrawSetup<'a>> {
    let kv = &sprite.kv6;

    // Voxlap's quick-and-dirty distance estimate (voxlap5.c:8835):
    //   y = ftol((spr->p - gipos) · gifor)
    // Used by the lowermip descent loop. Roxlap-formats `Kv6` doesn't
    // model lowermip yet, so the loop never runs and this value is
    // unused — computed for symmetry with voxlap and to lock the
    // path for a future mip-chain port.
    let dx = sprite.p[0] - cam.pos[0];
    let dy = sprite.p[1] - cam.pos[1];
    let dz = sprite.p[2] - cam.pos[2];
    let dist_estimate = ftol(dx * cam.forward[0] + dy * cam.forward[1] + dz * cam.forward[2]);
    let _ = (dist_estimate, KV6_MIPFACTOR_DEFAULT);
    let mip = 0u32;
    let ts = sprite.s;
    let th = sprite.h;
    let tf = sprite.f;

    // Bound-cube centre + half-extents in camera-relative coords.
    // (voxlap5.c:8852-8860; tp is centre offset from pivot, tp2 is
    // axis half-extent.) kv->xsiz/ysiz/zsiz fit f32 exactly for
    // any realistic kv6 (≤ 256³ per the file format limit).
    #[allow(clippy::cast_precision_loss)]
    let half_x = kv.xsiz as f32 * 0.5;
    #[allow(clippy::cast_precision_loss)]
    let half_y = kv.ysiz as f32 * 0.5;
    #[allow(clippy::cast_precision_loss)]
    let half_z = kv.zsiz as f32 * 0.5;
    let off_x = half_x - kv.xpiv;
    let off_y = half_y - kv.ypiv;
    let off_z = half_z - kv.zpiv;
    let npos = [
        off_x * ts[0] + off_y * th[0] + off_z * tf[0] + dx,
        off_x * ts[1] + off_y * th[1] + off_z * tf[1] + dy,
        off_x * ts[2] + off_y * th[2] + off_z * tf[2] + dz,
    ];
    let nstr = [ts[0] * half_x, ts[1] * half_x, ts[2] * half_x];
    let nhei = [th[0] * half_y, th[1] * half_y, th[2] * half_y];
    let nfor = [tf[0] * half_z, tf[1] * half_z, tf[2] * half_z];

    // 4-plane cull (voxlap5.c:8861-8875, walked z=3..0).
    for n in &cam.nor {
        let proj_str = (nstr[0] * n[0] + nstr[1] * n[1] + nstr[2] * n[2]).abs();
        let proj_hei = (nhei[0] * n[0] + nhei[1] * n[1] + nhei[2] * n[2]).abs();
        let proj_for = (nfor[0] * n[0] + nfor[1] * n[1] + nfor[2] * n[2]).abs();
        let proj_pos = npos[0] * n[0] + npos[1] * n[1] + npos[2] * n[2];
        if proj_str + proj_hei + proj_for + proj_pos < 0.0 {
            return None;
        }
    }

    Some(Kv6DrawSetup {
        kv,
        ts,
        th,
        tf,
        mip,
    })
}

/// 3×3 + translation matrix multiply, port of voxlap's `mat2`
/// (voxlap5.c:9619). Composes camera transform `(a_s, a_h, a_f, a_o)`
/// with sprite basis `(b_s, b_h, b_f, b_o)` into camera-relative
/// sprite basis `(c_s, c_h, c_f, c_o)`.
///
/// `c_s = a_s * b_s.x + a_h * b_s.y + a_f * b_s.z`, similarly for
/// `c_h` / `c_f`. `c_o = same form on b_o + a_o`.
#[allow(clippy::too_many_arguments)]
fn mat2(
    a_s: [f32; 3],
    a_h: [f32; 3],
    a_f: [f32; 3],
    a_o: [f32; 3],
    b_s: [f32; 3],
    b_h: [f32; 3],
    b_f: [f32; 3],
    b_o: [f32; 3],
) -> ([f32; 3], [f32; 3], [f32; 3], [f32; 3]) {
    let c_s = [
        a_s[0] * b_s[0] + a_h[0] * b_s[1] + a_f[0] * b_s[2],
        a_s[1] * b_s[0] + a_h[1] * b_s[1] + a_f[1] * b_s[2],
        a_s[2] * b_s[0] + a_h[2] * b_s[1] + a_f[2] * b_s[2],
    ];
    let c_h = [
        a_s[0] * b_h[0] + a_h[0] * b_h[1] + a_f[0] * b_h[2],
        a_s[1] * b_h[0] + a_h[1] * b_h[1] + a_f[1] * b_h[2],
        a_s[2] * b_h[0] + a_h[2] * b_h[1] + a_f[2] * b_h[2],
    ];
    let c_f = [
        a_s[0] * b_f[0] + a_h[0] * b_f[1] + a_f[0] * b_f[2],
        a_s[1] * b_f[0] + a_h[1] * b_f[1] + a_f[1] * b_f[2],
        a_s[2] * b_f[0] + a_h[2] * b_f[1] + a_f[2] * b_f[2],
    ];
    let c_o = [
        a_s[0] * b_o[0] + a_h[0] * b_o[1] + a_f[0] * b_o[2] + a_o[0],
        a_s[1] * b_o[0] + a_h[1] * b_o[1] + a_f[1] * b_o[2] + a_o[1],
        a_s[2] * b_o[0] + a_h[2] * b_o[1] + a_f[2] * b_o[2] + a_o[2],
    ];
    (c_s, c_h, c_f, c_o)
}

/// Voxlap's `lbound(a, b, c)` (voxlap5.c:406): clamp `a` into the
/// inclusive range `[b, c]`. `c` must be `>= b`.
#[inline]
fn lbound(a: i32, b: i32, c: i32) -> i32 {
    a.clamp(b, c)
}

/// State derived from `Kv6DrawSetup` + `CameraState` that the
/// per-voxel iteration consumes. Voxlap holds these on the stack
/// inside `kv6draw`; roxlap factors them out so the iteration loop
/// can be tested independently.
#[derive(Debug, Clone)]
#[allow(dead_code)] // R6.4+ reads scisdist / qsum0 / cadd / etc; for R6.3 only inx/iny/inz/nxplane* matter.
pub(crate) struct Kv6IterState<'a> {
    pub kv: &'a Kv6,
    /// Camera origin expressed in kv6-local voxel coordinates,
    /// clamped to `[-1, kv.xsiz]` etc. by voxlap's `lbound`. Splits
    /// the voxel grid into the 4 + 1 quadrants the iteration walks
    /// in different orders so that for each (x, y) column the inner
    /// z-loop visits voxels closer to the camera first (= correct
    /// painter's-style ordering for the rasterizer in R6.4).
    pub inx: i32,
    pub iny: i32,
    pub inz: i32,
    /// `vx5.xplanemin` / `vx5.xplanemax` mirror — voxlap defaults
    /// to `[0, INT_MAX]` (no x-clipping). Roxlap doesn't yet expose
    /// a public knob for these; pinning to the defaults matches the
    /// oracle and any caller that doesn't care.
    pub nxplanemin: i32,
    pub nxplanemax: i32,
}

/// Mat2 transform + Cramer's-rule split point. Mirror of
/// voxlap5.c:8915-8940. Takes `sprite.p` separately because the
/// `Kv6DrawSetup` only carries the post-mip-LOD + cull state, not
/// the sprite reference.
fn kv6_compute_iter_state<'a>(
    setup: &Kv6DrawSetup<'a>,
    sprite_pos: [f32; 3],
    cam: &CameraState,
) -> Kv6IterState<'a> {
    let kv = setup.kv;

    // Transform sprite basis from world to camera-relative
    // screen-axis coords (voxlap5.c:8916). `(gixs, giys, gizs)` is
    // the transposed camera basis; `giadd` is the translation half.
    let (nstr, nhei, nfor, mut npos) = mat2(
        cam.xs, cam.ys, cam.zs, cam.add, setup.ts, setup.th, setup.tf, sprite_pos,
    );

    // Shift `npos` so it points at the kv6 origin (corner [0,0,0])
    // rather than the pivot point — Cramer's rule below solves for
    // the camera origin in kv6-local voxel coords, which only makes
    // sense relative to the corner. (voxlap5.c:8917-8919)
    npos[0] -= kv.xpiv * nstr[0] + kv.ypiv * nhei[0] + kv.zpiv * nfor[0];
    npos[1] -= kv.xpiv * nstr[1] + kv.ypiv * nhei[1] + kv.zpiv * nfor[1];
    npos[2] -= kv.xpiv * nstr[2] + kv.ypiv * nhei[2] + kv.zpiv * nfor[2];

    // Cramer's rule for `nstr * X + nhei * Y + nfor * Z + npos = 0`.
    // (voxlap5.c:8923-8936)
    let tp = [
        nhei[1] * nfor[2] - nfor[1] * nhei[2],
        nfor[1] * nstr[2] - nstr[1] * nfor[2],
        nstr[1] * nhei[2] - nhei[1] * nstr[2],
    ];
    let det = nstr[0] * tp[0] + nhei[0] * tp[1] + nfor[0] * tp[2];
    // Float-bit comparison against zero: matches voxlap's
    // `if (f != 0)` and dodges clippy::float_cmp.
    let (raw_inx, raw_iny, raw_inz) = if det.to_bits() & 0x7fff_ffff != 0 {
        let f_inv = -1.0 / det;
        let tp2 = [
            npos[1] * nfor[2] - nfor[1] * npos[2],
            nhei[1] * npos[2] - npos[1] * nhei[2],
            npos[1] * nstr[2] - nstr[1] * npos[2],
        ];
        (
            ftol((npos[0] * tp[0] - nhei[0] * tp2[0] - nfor[0] * tp2[1]) * f_inv),
            ftol((npos[0] * tp[1] + nstr[0] * tp2[0] - nfor[0] * tp2[2]) * f_inv),
            ftol((npos[0] * tp[2] + nstr[0] * tp2[1] + nhei[0] * tp2[2]) * f_inv),
        )
    } else {
        (-1, -1, -1)
    };

    let xsiz_i = kv.xsiz as i32;
    let ysiz_i = kv.ysiz as i32;
    let zsiz_i = kv.zsiz as i32;
    let inx = lbound(raw_inx, -1, xsiz_i);
    let iny = lbound(raw_iny, -1, ysiz_i);
    let inz = lbound(raw_inz, -1, zsiz_i);

    Kv6IterState {
        kv,
        inx,
        iny,
        inz,
        // Voxlap default `vx5.xplanemin = 0`, `xplanemax = 0x7fffffff`.
        nxplanemin: 0,
        nxplanemax: i32::MAX,
    }
}

/// One iteration of voxlap's `DRAWBOUNDCUBELINE` macro
/// (voxlap5.c:8809-8812). Walks the voxel range `[range_start,
/// range_end)` (which is one (x, y) column's voxels) in three
/// phases:
///
/// 1. Forward through voxels with `z < inz`, calling
///    `callback(voxel, base_mask | 0x20)`.
/// 2. Backward through voxels with `z > inz`, calling
///    `callback(voxel, base_mask | 0x10)`.
/// 3. If a single voxel remains with `z == inz`, call
///    `callback(voxel, base_mask | 0x00)`.
///
/// This visits every voxel in the column exactly once. The mask
/// bits encode which faces of the voxel-cube are exposed to the
/// camera; R6.4's `drawboundcubesse` will project them.
fn draw_boundcube_line<F: FnMut(&Voxel, u32)>(
    voxels: &[Voxel],
    range_start: usize,
    range_end: usize,
    inz: i32,
    base_mask: u32,
    callback: &mut F,
) {
    if range_end <= range_start {
        return;
    }
    let mut v0 = range_start;
    let mut v1_excl = range_end;

    // Phase 1: forward while voxels[v0].z < inz.
    while v0 < v1_excl && i32::from(voxels[v0].z) < inz {
        callback(&voxels[v0], base_mask | 0x20);
        v0 += 1;
    }
    // Phase 2: backward while voxels[v1_excl - 1].z > inz.
    while v0 < v1_excl && i32::from(voxels[v1_excl - 1].z) > inz {
        callback(&voxels[v1_excl - 1], base_mask | 0x10);
        v1_excl -= 1;
    }
    // Phase 3: single voxel left with z == inz.
    if v0 + 1 == v1_excl {
        callback(&voxels[v0], base_mask);
    }
}

/// 9-arm per-(x, y) column iteration walking the kv6's voxel
/// grid in painter's-back-to-front order around the camera-split
/// point (`inx`, `iny`, `inz`). Mirror of voxlap5.c:8982-9062.
///
/// Each (x, y) column is visited exactly once, so the total
/// number of `callback` fires equals `kv.numvoxs` (the sum of all
/// `xlen[x]`).
#[allow(clippy::too_many_lines)]
pub(crate) fn kv6_iterate<F: FnMut(&Voxel, u32)>(state: &Kv6IterState<'_>, mut callback: F) {
    let kv = state.kv;
    let xsiz = kv.xsiz as i32;
    let ysiz = kv.ysiz as i32;
    let inx = state.inx;
    let iny = state.iny;
    let inz = state.inz;
    let nxplanemin = state.nxplanemin;
    let nxplanemax = state.nxplanemax;

    let mut xv: usize = 0;

    // First half: x = 0..inx. Top-half quadrants (masks 0xa, 0x6, 0x2).
    let mut x: i32 = 0;
    while x < inx {
        let xu = x as usize;
        let xlen = kv.xlen[xu] as usize;
        if x < nxplanemin || x >= nxplanemax {
            xv += xlen;
            x += 1;
            continue;
        }
        let yv_initial = xv + xlen;

        // Forward y: 0..iny  -> mask 0xa.
        let mut xv_local = xv;
        let mut y: i32 = 0;
        while y < iny {
            let yu = y as usize;
            let len = kv.ylen[xu][yu] as usize;
            let v0 = xv_local;
            xv_local += len;
            draw_boundcube_line(&kv.voxels, v0, xv_local, inz, 0xa, &mut callback);
            y += 1;
        }

        // Reverse y: ysiz-1..iny  -> mask 0x6.
        let mut yv_local = yv_initial;
        let mut y = ysiz - 1;
        while y > iny {
            let yu = y as usize;
            let len = kv.ylen[xu][yu] as usize;
            let v1_excl = yv_local;
            yv_local -= len;
            draw_boundcube_line(&kv.voxels, yv_local, v1_excl, inz, 0x6, &mut callback);
            y -= 1;
        }

        // Edge y == iny  -> mask 0x2.
        if iny >= 0 && (iny as u32) < kv.ysiz {
            let yu = iny as usize;
            let len = kv.ylen[xu][yu] as usize;
            let v1_excl = yv_local;
            yv_local -= len;
            draw_boundcube_line(&kv.voxels, yv_local, v1_excl, inz, 0x2, &mut callback);
        }

        xv += xlen;
        x += 1;
    }

    // Second half: x = xsiz-1..inx (reverse). Bot-half quadrants
    // (masks 0x5, 0x9, 0x1).
    let mut xv2: usize = kv.voxels.len();
    let mut x = xsiz - 1;
    while x > inx {
        let xu = x as usize;
        let xlen = kv.xlen[xu] as usize;
        if x < nxplanemin || x >= nxplanemax {
            xv2 -= xlen;
            x -= 1;
            continue;
        }
        let yv_initial = xv2 - xlen;

        // Reverse y: ysiz-1..iny  -> mask 0x5.
        let mut xv_local = xv2;
        let mut y = ysiz - 1;
        while y > iny {
            let yu = y as usize;
            let len = kv.ylen[xu][yu] as usize;
            let v1_excl = xv_local;
            xv_local -= len;
            draw_boundcube_line(&kv.voxels, xv_local, v1_excl, inz, 0x5, &mut callback);
            y -= 1;
        }

        // Forward y: 0..iny  -> mask 0x9.
        let mut yv_local = yv_initial;
        let mut y: i32 = 0;
        while y < iny {
            let yu = y as usize;
            let len = kv.ylen[xu][yu] as usize;
            let v0 = yv_local;
            yv_local += len;
            draw_boundcube_line(&kv.voxels, v0, yv_local, inz, 0x9, &mut callback);
            y += 1;
        }

        // Edge y == iny  -> mask 0x1.
        if iny >= 0 && (iny as u32) < kv.ysiz {
            let yu = iny as usize;
            let len = kv.ylen[xu][yu] as usize;
            let v0 = yv_local;
            yv_local += len;
            draw_boundcube_line(&kv.voxels, v0, yv_local, inz, 0x1, &mut callback);
        }

        xv2 -= xlen;
        x -= 1;
    }

    // Edge x == inx (middle column). Masks 0x4, 0x8, 0x0.
    if inx >= 0 && (inx as u32) < kv.xsiz {
        let xu = inx as usize;
        if inx < nxplanemin || inx >= nxplanemax {
            return;
        }
        let xlen = kv.xlen[xu] as usize;
        let yv_initial = xv2 - xlen;

        // Reverse y -> mask 0x4.
        let mut xv_local = xv2;
        let mut y = ysiz - 1;
        while y > iny {
            let yu = y as usize;
            let len = kv.ylen[xu][yu] as usize;
            let v1_excl = xv_local;
            xv_local -= len;
            draw_boundcube_line(&kv.voxels, xv_local, v1_excl, inz, 0x4, &mut callback);
            y -= 1;
        }

        // Forward y -> mask 0x8.
        let mut yv_local = yv_initial;
        let mut y: i32 = 0;
        while y < iny {
            let yu = y as usize;
            let len = kv.ylen[xu][yu] as usize;
            let v0 = yv_local;
            yv_local += len;
            draw_boundcube_line(&kv.voxels, v0, yv_local, inz, 0x8, &mut callback);
            y += 1;
        }

        // Edge y == iny -> mask 0x0.
        if iny >= 0 && (iny as u32) < kv.ysiz {
            let yu = iny as usize;
            let len = kv.ylen[xu][yu] as usize;
            let v0 = yv_local;
            yv_local += len;
            draw_boundcube_line(&kv.voxels, v0, yv_local, inz, 0x0, &mut callback);
        }
    }
}

/// Draw a sprite into the engine's framebuffer + z-buffer.
///
/// Top-level dispatcher mirroring voxlap5.c:9818-9828:
/// - Skips on `flags & INVISIBLE`.
/// - Picks `kv6draw` vs `kv6draw_noz` based on `flags & NO_Z`.
/// - Picks the kv6 vs kfa path based on `flags & KFA`.
///
/// **R6.3 status**: dispatcher runs cull → mat2 + Cramer's split
/// point → 9-arm per-voxel iteration with a no-op callback. No
/// pixels written yet — R6.4 plugs the rasterizer in. Returns
/// `false` for "culled" and "would render" alike.
#[must_use]
pub fn draw_sprite(cam: &CameraState, sprite: &Sprite) -> bool {
    if sprite.flags & SPRITE_FLAG_INVISIBLE != 0 {
        return false;
    }
    if sprite.flags & SPRITE_FLAG_KFA != 0 {
        // KFA animation path is out of scope for R6 (no oracle
        // pose exercises it). Mirror voxlap's silent dispatch but
        // without rendering anything.
        return false;
    }
    let Some(setup) = kv6_draw_prepare(sprite, cam) else {
        return false;
    };
    let state = kv6_compute_iter_state(&setup, sprite.p, cam);
    // R6.3 stub callback: no rendering. R6.4 will swap this out
    // for the real `drawboundcubesse` per-voxel rasterizer.
    kv6_iterate(&state, |_voxel, _mask| {});
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera_math;
    use crate::Camera;
    use roxlap_formats::kv6::Kv6;

    fn empty_kv6() -> Kv6 {
        Kv6 {
            xsiz: 1,
            ysiz: 1,
            zsiz: 1,
            xpiv: 0.5,
            ypiv: 0.5,
            zpiv: 0.5,
            voxels: Vec::new(),
            xlen: vec![0],
            ylen: vec![vec![0]],
            palette: None,
        }
    }

    /// 17×17×17 kv6 with pivot at the centre — same dimensions as
    /// the meltsphere oracle sprite so the cull test exercises a
    /// realistic bound cube rather than a 1-voxel point.
    fn cube_kv6() -> Kv6 {
        Kv6 {
            xsiz: 17,
            ysiz: 17,
            zsiz: 17,
            xpiv: 8.5,
            ypiv: 8.5,
            zpiv: 8.5,
            voxels: Vec::new(),
            xlen: vec![0; 17],
            ylen: vec![vec![0; 17]; 17],
            palette: None,
        }
    }

    /// `CameraState` matching the oracle's `sprite_front` pose:
    /// pos=(1020,1050,175), yaw=0, pitch=0 → forward = +x.
    fn oracle_sprite_front_camera() -> camera_math::CameraState {
        let camera = Camera {
            pos: [1020.0, 1050.0, 175.0],
            // From oracle.c set_camera_yaw_pitch with yaw=0, pitch=0:
            //   ifor = [1, 0, 0], istr = [0, 1, 0], ihei = [0, 0, 1].
            right: [0.0, 1.0, 0.0],
            down: [0.0, 0.0, 1.0],
            forward: [1.0, 0.0, 0.0],
        };
        camera_math::derive(&camera, 640, 480, 320.0, 240.0, 320.0)
    }

    #[test]
    fn axis_aligned_sets_identity_basis() {
        // Compare bit patterns: these are integer-valued floats so
        // bit-equality is well-defined and dodges clippy::float_cmp.
        let bits = |a: [f32; 3]| a.map(f32::to_bits);
        let s = Sprite::axis_aligned(empty_kv6(), [10.0, 20.0, 30.0]);
        assert_eq!(bits(s.p), bits([10.0, 20.0, 30.0]));
        assert_eq!(bits(s.s), bits([1.0, 0.0, 0.0]));
        assert_eq!(bits(s.h), bits([0.0, 1.0, 0.0]));
        assert_eq!(bits(s.f), bits([0.0, 0.0, 1.0]));
        assert_eq!(s.flags, 0);
    }

    #[test]
    fn invisible_flag_skips_dispatch() {
        let cam = oracle_sprite_front_camera();
        let mut s = Sprite::axis_aligned(cube_kv6(), [1050.0, 1050.0, 175.0]);
        s.flags = SPRITE_FLAG_INVISIBLE;
        assert!(!draw_sprite(&cam, &s));
    }

    #[test]
    fn kfa_flag_skips_dispatch() {
        let cam = oracle_sprite_front_camera();
        let mut s = Sprite::axis_aligned(cube_kv6(), [1050.0, 1050.0, 175.0]);
        s.flags = SPRITE_FLAG_KFA;
        assert!(!draw_sprite(&cam, &s));
    }

    #[test]
    fn cull_keeps_oracle_sprite_in_front_of_camera() {
        // Oracle's `sprite_front` pose: camera at (1020,1050,175)
        // looking +x; sprite at (1050,1050,175). Sprite is 30
        // units forward, on-axis — clearly inside the frustum.
        let cam = oracle_sprite_front_camera();
        let s = Sprite::axis_aligned(cube_kv6(), [1050.0, 1050.0, 175.0]);
        assert!(
            kv6_draw_prepare(&s, &cam).is_some(),
            "front-of-camera sprite must NOT be culled"
        );
    }

    #[test]
    fn cull_removes_sprite_far_behind_camera() {
        // Same camera; sprite far in the -forward direction
        // (= behind the camera).
        let cam = oracle_sprite_front_camera();
        let s = Sprite::axis_aligned(cube_kv6(), [1020.0 - 500.0, 1050.0, 175.0]);
        assert!(
            kv6_draw_prepare(&s, &cam).is_none(),
            "behind-camera sprite must be culled"
        );
    }

    #[test]
    fn cull_removes_sprite_far_to_the_right() {
        // Camera looks +x; sprite far in the +y direction (right
        // axis), far enough that the bound cube is fully outside
        // the right-edge frustum plane.
        let cam = oracle_sprite_front_camera();
        // 30 units forward, 200 units right — well outside the 90°
        // FOV's right edge.
        let s = Sprite::axis_aligned(cube_kv6(), [1050.0, 1050.0 + 200.0, 175.0]);
        assert!(
            kv6_draw_prepare(&s, &cam).is_none(),
            "far-right sprite must be culled"
        );
    }

    #[test]
    fn cull_keeps_sprite_at_camera_position() {
        // Sprite centred on the camera — bound cube straddles the
        // camera, so by definition it's not fully outside any
        // frustum plane and must NOT be culled.
        let cam = oracle_sprite_front_camera();
        let s = Sprite::axis_aligned(cube_kv6(), cam.pos);
        assert!(
            kv6_draw_prepare(&s, &cam).is_some(),
            "sprite at camera position must not be culled"
        );
    }

    #[test]
    fn iterate_visits_each_voxel_exactly_once() {
        // Build a synthetic 3×3×3 kv6 with one voxel per (x, y)
        // column at z = x + y mod 3. Then iterate and check
        // (a) total callback fires == 27 = numvoxs, and (b) every
        // voxel index 0..27 was visited exactly once.
        let xsiz: u32 = 3;
        let ysiz: u32 = 3;
        let zsiz: u32 = 3;
        let mut voxels = Vec::new();
        let mut xlen = vec![0u32; xsiz as usize];
        let mut ylen = vec![vec![0u16; ysiz as usize]; xsiz as usize];
        for x in 0..xsiz {
            for y in 0..ysiz {
                let z = ((x + y) % 3) as u16;
                voxels.push(Voxel {
                    col: 0x0080_0000,
                    z,
                    vis: 63,
                    dir: 0,
                });
                xlen[x as usize] += 1;
                ylen[x as usize][y as usize] = 1;
            }
        }
        let kv = Kv6 {
            xsiz,
            ysiz,
            zsiz,
            xpiv: 1.5,
            ypiv: 1.5,
            zpiv: 1.5,
            voxels,
            xlen,
            ylen,
            palette: None,
        };
        let setup = Kv6DrawSetup {
            kv: &kv,
            ts: [1.0, 0.0, 0.0],
            th: [0.0, 1.0, 0.0],
            tf: [0.0, 0.0, 1.0],
            mip: 0,
        };
        let cam = oracle_sprite_front_camera();
        let state = kv6_compute_iter_state(&setup, [1050.0, 1050.0, 175.0], &cam);

        // Every voxel index must fire exactly once. We use a
        // by-pointer identity check via .as_ptr() offsets.
        let voxels_ptr = kv.voxels.as_ptr();
        let mut visited = vec![0u32; kv.voxels.len()];
        let mut total: u32 = 0;
        kv6_iterate(&state, |v, _mask| {
            // SAFETY: callback receives a borrow of an entry of
            // `kv.voxels`; computing the offset is well-defined.
            let idx = unsafe { std::ptr::from_ref::<Voxel>(v).offset_from(voxels_ptr) } as usize;
            visited[idx] += 1;
            total += 1;
        });
        assert_eq!(total as usize, kv.voxels.len(), "total callback fires");
        for (i, &n) in visited.iter().enumerate() {
            assert_eq!(n, 1, "voxel {i} visited {n} times (want 1)");
        }
    }

    #[test]
    fn iterate_meltsphere_oracle_visits_each_voxel_once() {
        // Load the dumped voxlap-C meltsphere fixture (R6.0e) and
        // run the iteration against the oracle's sprite_front
        // camera + sprite pose. Expected: every voxel hit exactly
        // once, total fires == kv.voxels.len() (= 401).
        const SPRITE_KV6: &[u8] = include_bytes!("../tests/fixtures/sprite_meltsphere.kv6");
        let kv = roxlap_formats::kv6::parse(SPRITE_KV6).expect("parse fixture");
        assert_eq!(kv.voxels.len(), 401, "fixture voxel count");

        let sprite = Sprite::axis_aligned(kv, [1050.0, 1050.0, 175.0]);
        let cam = oracle_sprite_front_camera();
        let setup = kv6_draw_prepare(&sprite, &cam).expect("oracle sprite must pass cull");
        let state = kv6_compute_iter_state(&setup, sprite.p, &cam);

        let voxels_ptr = sprite.kv6.voxels.as_ptr();
        let mut visited = vec![0u32; sprite.kv6.voxels.len()];
        let mut total: u32 = 0;
        kv6_iterate(&state, |v, _mask| {
            let idx = unsafe { std::ptr::from_ref::<Voxel>(v).offset_from(voxels_ptr) } as usize;
            visited[idx] += 1;
            total += 1;
        });
        assert_eq!(total, 401);
        let max = visited.iter().copied().max().unwrap();
        let min = visited.iter().copied().min().unwrap();
        assert_eq!(max, 1, "no voxel may be visited twice");
        assert_eq!(min, 1, "no voxel may be skipped");
    }

    #[test]
    fn iterate_no_voxels_when_culled() {
        // Sprite far behind camera → cull. draw_sprite never
        // reaches kv6_iterate, so no callback fires.
        let cam = oracle_sprite_front_camera();
        let s = Sprite::axis_aligned(cube_kv6(), [1020.0 - 500.0, 1050.0, 175.0]);
        // Cull catches it before iteration.
        assert!(kv6_draw_prepare(&s, &cam).is_none());
    }

    #[test]
    fn r63_dispatcher_returns_false_for_in_frustum_sprite() {
        // R6.3 stub: cull + iteration run, but no pixels are
        // written by the no-op callback. R6.4 will flip this.
        let cam = oracle_sprite_front_camera();
        let s = Sprite::axis_aligned(cube_kv6(), [1050.0, 1050.0, 175.0]);
        assert!(!draw_sprite(&cam, &s));
    }
}
