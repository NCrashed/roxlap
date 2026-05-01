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
// kv.{xsiz,ysiz,zsiz} are u32 with realistic max ≤ 256 (file format
// limit) — well within f32's 24-bit mantissa.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::similar_names
)]

use roxlap_formats::kv6::{Kv6, Voxel};

use crate::camera_math::CameraState;
use crate::fixed::ftol;
use crate::opticast::OpticastSettings;

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
#[allow(dead_code)] // R6.4+ reads scisdist / qsum0 / cadd / etc.
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

/// Full per-frame rasterizer state for one sprite — what
/// `drawboundcubesse` reads via voxlap's globals.
///
/// Built by [`kv6_compute_full_state`] from the post-cull
/// `Kv6DrawSetup` + the camera's projection params. Mirror of the
/// voxlap5.c:8915-8973 setup block.
#[derive(Debug, Clone)]
#[allow(dead_code)] // R6.4b/c/d will read these.
pub(crate) struct Kv6FullState<'a> {
    pub iter: Kv6IterState<'a>,
    /// 8 cube-vertex offsets, gihz-scaled. `cadd4[k]` for `k = 0..7`
    /// is the offset of cube vertex `k` from the voxel origin, where
    /// bit 0 = +x, bit 1 = +z (post-swap == old +z), bit 2 = +y
    /// (post-swap == old -y). `cadd4[0]` is `[0; 4]`. Lane 3 of
    /// each entry duplicates lane 2 (z) — voxlap's SSE convenience.
    pub cadd4: [[f32; 4]; 8],
    /// Per-z step table: `ztab4_per_z[z] = z * cadd4[2]`. Length =
    /// `kv.zsiz`. Indexed by `v.z` in `drawboundcubesse`.
    pub ztab4_per_z: Vec<[f32; 4]>,
    /// Initial r1 — the x=0 column base after voxlap's "ANNOYING
    /// HACK" pre-decrement. = `(npos*gihz with z2=npos.z) -
    /// cadd4[4]`. Iterates by `cadd4[1]` per x and (via r0) by
    /// `cadd4[4]` per y.
    pub r1_initial: [f32; 4],
    /// `r2 = -ysiz * cadd4[4]`. Used to reset r0 between forward-y
    /// and reverse-y phases inside one x column.
    pub r2: [f32; 4],
    /// Near-plane scissor distance (camera-space Z).
    /// `voxlap5.c:8953-8956` — equals the negative sum of any
    /// negative components of post-swap `nstr.z` / `nhei.z` /
    /// `nfor.z`. `0.0` if all three are non-negative.
    pub scisdist: f32,
    /// Viewport-clip biases (voxlap5.c:8947-8948).
    /// `qsum0[2] = qsum0[0] = 0x7fff - (xres - hx as i32)`,
    /// `qsum0[3] = qsum0[1] = 0x7fff - (yres - hy as i32)`.
    pub qsum0: [i16; 4],
}

#[inline]
fn vec4_add(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
}

#[inline]
fn vec4_sub(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
}

#[inline]
fn vec4_scale(a: [f32; 4], s: f32) -> [f32; 4] {
    [a[0] * s, a[1] * s, a[2] * s, a[3] * s]
}

/// Full setup: mat2 + Cramer's + nfor↔nhei swap + cadd4/ztab4/r1/r2/
/// scisdist/qsum0 init. Mirror of voxlap5.c:8915-8973.
fn kv6_compute_full_state<'a>(
    setup: &Kv6DrawSetup<'a>,
    sprite_pos: [f32; 3],
    cam: &CameraState,
    settings: &OpticastSettings,
) -> Kv6FullState<'a> {
    let kv = setup.kv;

    // Transform sprite basis from world to camera-relative
    // screen-axis coords (voxlap5.c:8916). `(gixs, giys, gizs)` is
    // the transposed camera basis; `giadd` is the translation half.
    let (nstr, mut nhei, mut nfor, mut npos) = mat2(
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
    let iter = Kv6IterState {
        kv,
        inx: lbound(raw_inx, -1, xsiz_i),
        iny: lbound(raw_iny, -1, ysiz_i),
        inz: lbound(raw_inz, -1, zsiz_i),
        // Voxlap default `vx5.xplanemin = 0`, `xplanemax = 0x7fffffff`.
        nxplanemin: 0,
        nxplanemax: i32::MAX,
    };

    // Swap `nhei` ↔ `nfor` with sign flip on the new `nfor`
    // (voxlap5.c:8942-8944). Equivalent to a 90° rotation that lines
    // the basis up with cadd4's bit-encoded vertex offsets:
    //   cadd4[1] = +x  (post-swap nstr direction)
    //   cadd4[2] = +z  (post-swap nhei direction == original +z)
    //   cadd4[4] = +y  (post-swap nfor direction == original -y)
    // After this point `nfor` / `nhei` carry the post-swap values.
    let swap_x = nhei[0];
    nhei[0] = nfor[0];
    nfor[0] = -swap_x;
    let swap_y = nhei[1];
    nhei[1] = nfor[1];
    nfor[1] = -swap_y;
    let swap_z = nhei[2];
    nhei[2] = nfor[2];
    nfor[2] = -swap_z;

    // qsum0 (voxlap5.c:8947-8948). The `0x7fff - (xres - hx)`
    // form sets the bias such that adding it to a screen-space
    // bound makes the bound saturate-positive when it lands
    // inside the viewport.
    let xres_i = settings.xres as i32;
    let yres_i = settings.yres as i32;
    let hx_i = ftol(settings.hx);
    let hy_i = ftol(settings.hy);
    let qsum0_x = (0x7fff - (xres_i - hx_i)) as i16;
    let qsum0_y = (0x7fff - (yres_i - hy_i)) as i16;
    let qsum0 = [qsum0_x, qsum0_y, qsum0_x, qsum0_y];

    // scisdist (voxlap5.c:8953-8956). Voxlap's `*(int32_t *)&f < 0`
    // bit-trick: a positive-finite float has bit-pattern >= 0;
    // only *negative* floats land < 0 as signed int. So this loop
    // sums the absolute value of any negative-z post-swap basis
    // component into a near-plane bias.
    let mut scisdist = 0.0f32;
    if (nstr[2].to_bits() as i32) < 0 {
        scisdist -= nstr[2];
    }
    if (nhei[2].to_bits() as i32) < 0 {
        scisdist -= nhei[2];
    }
    if (nfor[2].to_bits() as i32) < 0 {
        scisdist -= nfor[2];
    }

    // cadd4 step table (voxlap5.c:8958-8961). cadd4[1/2/4] are the
    // three primary axis steps (x / z / y, post-swap); cadd4[3/5/6/7]
    // are bit-OR sums (3 = 1+2, 5 = 1+4, 6 = 2+4, 7 = 3+4).
    let gihz = settings.hz;
    let cadd1 = [nstr[0] * gihz, nstr[1] * gihz, nstr[2], nstr[2]];
    let cadd2 = [nhei[0] * gihz, nhei[1] * gihz, nhei[2], nhei[2]];
    let cadd4_axis = [nfor[0] * gihz, nfor[1] * gihz, nfor[2], nfor[2]];
    let cadd3 = vec4_add(cadd1, cadd2);
    let cadd5 = vec4_add(cadd1, cadd4_axis);
    let cadd6 = vec4_add(cadd2, cadd4_axis);
    let cadd7 = vec4_add(cadd3, cadd4_axis);
    let cadd4 = [
        [0.0; 4], cadd1, cadd2, cadd3, cadd4_axis, cadd5, cadd6, cadd7,
    ];

    // ztab4 per-z step table (voxlap5.c:8973). ztab4[z] = z * cadd4[2]
    // built incrementally by addps so per-step rounding matches.
    let zsiz = kv.zsiz as usize;
    let mut ztab4_per_z = Vec::with_capacity(zsiz);
    if zsiz > 0 {
        ztab4_per_z.push([0.0f32; 4]);
        for i in 1..zsiz {
            let prev = ztab4_per_z[i - 1];
            ztab4_per_z.push(vec4_add(prev, cadd4[2]));
        }
    }

    // r1 init (voxlap5.c:8961, 8976). Post-mat2 npos becomes the
    // raw column-base; gihz-scale x/y; z lane keeps unscaled npos.z;
    // z2 lane (lane 3) duplicates z. Then "ANNOYING HACK"
    // pre-decrement by cadd4[4].
    let r1_pre = [npos[0] * gihz, npos[1] * gihz, npos[2], npos[2]];
    let r1_initial = vec4_sub(r1_pre, cadd4[4]);

    // r2 = -ysiz * cadd4[4] (voxlap5.c:8974). intss + mulps in voxlap.
    let r2 = vec4_scale(cadd4[4], -(ysiz_i as f32));

    Kv6FullState {
        iter,
        cadd4,
        ztab4_per_z,
        r1_initial,
        r2,
        scisdist,
        qsum0,
    }
}

/// Per-voxel rasterizer entry.
///
/// R6.4a stops after origin compute + scissor; later substages
/// add vertex projection (R6.4b), screen-AABB + clip (R6.4c),
/// fill loop (R6.4d), and colour modulation + `mm5` cross-call
/// tail (R6.4e).
///
/// Mirror of voxlap5.c:8179-8192 (the early-out + origin + scissor
/// portion of `drawboundcubesse`).
///
/// Returns `true` if the voxel passed the early-out + scissor —
/// i.e. R6.4b would proceed to project vertices. For R6.4a the
/// return value is the scissor-pass signal that tests assert on.
#[allow(clippy::trivially_copy_pass_by_ref)] // hot loop; matches voxlap's pointer-passed v.
pub(crate) fn drawboundcubesse(
    v: &Voxel,
    mask: u32,
    state: &Kv6FullState<'_>,
    r0: [f32; 4],
) -> bool {
    let effmask = mask & u32::from(v.vis);
    if effmask == 0 {
        return false;
    }
    let z = v.z as usize;
    let zstep = state.ztab4_per_z[z];
    let origin = vec4_add(r0, zstep);
    if origin[2] < state.scisdist {
        return false;
    }
    // R6.4b will continue from here.
    let _ = effmask;
    true
}

/// One iteration of voxlap's `DRAWBOUNDCUBELINE` macro
/// (voxlap5.c:8809-8812). Walks the voxel range `[range_start,
/// range_end)` (one (x, y) column's voxels) in three phases:
///
/// 1. Forward through voxels with `z < inz`, calling
///    `callback(voxel, base_mask | 0x20, r0)`.
/// 2. Backward through voxels with `z > inz`, calling
///    `callback(voxel, base_mask | 0x10, r0)`.
/// 3. If a single voxel remains with `z == inz`, call
///    `callback(voxel, base_mask | 0x00, r0)`.
///
/// Each (x, y) column is visited exactly once. `r0` is the screen-
/// space origin for *this* column — voxlap stores it as
/// `ztab4[MAXZSIZ]` and `drawboundcubesse` reads it via that index.
fn draw_boundcube_line<F: FnMut(&Voxel, u32, [f32; 4])>(
    voxels: &[Voxel],
    range_start: usize,
    range_end: usize,
    inz: i32,
    base_mask: u32,
    r0: [f32; 4],
    callback: &mut F,
) {
    if range_end <= range_start {
        return;
    }
    let mut v0 = range_start;
    let mut v1_excl = range_end;

    // Phase 1: forward while voxels[v0].z < inz.
    while v0 < v1_excl && i32::from(voxels[v0].z) < inz {
        callback(&voxels[v0], base_mask | 0x20, r0);
        v0 += 1;
    }
    // Phase 2: backward while voxels[v1_excl - 1].z > inz.
    while v0 < v1_excl && i32::from(voxels[v1_excl - 1].z) > inz {
        callback(&voxels[v1_excl - 1], base_mask | 0x10, r0);
        v1_excl -= 1;
    }
    // Phase 3: single voxel left with z == inz.
    if v0 + 1 == v1_excl {
        callback(&voxels[v0], base_mask, r0);
    }
}

/// 9-arm per-(x, y) column iteration walking the kv6's voxel
/// grid in painter's-back-to-front order around the camera-split
/// point (`inx`, `iny`, `inz`). Mirror of voxlap5.c:8982-9062.
///
/// Tracks `r1` (current x-column base) and `r0` (current (x, y)
/// origin) the same way voxlap mutates them with addps/subps,
/// passing `r0` to each per-voxel callback. `r0` evolves as
/// `r0[x][y] = r1_initial + x * cadd4[1] - y * cadd4[4]` (with
/// the floating-point operations applied in voxlap's order so the
/// per-step rounding matches bit-for-bit).
///
/// Each (x, y) column is visited exactly once.
#[allow(clippy::too_many_lines)]
pub(crate) fn kv6_iterate<F: FnMut(&Voxel, u32, [f32; 4])>(
    state: &Kv6FullState<'_>,
    mut callback: F,
) {
    let kv = state.iter.kv;
    let xsiz = kv.xsiz as i32;
    let ysiz = kv.ysiz as i32;
    let inx = state.iter.inx;
    let iny = state.iter.iny;
    let inz = state.iter.inz;
    let nxplanemin = state.iter.nxplanemin;
    let nxplanemax = state.iter.nxplanemax;
    let cadd1 = state.cadd4[1];
    let cadd_y = state.cadd4[4];
    let r2 = state.r2;

    let mut xv: usize = 0;
    let mut r1 = state.r1_initial;

    // First half: x = 0..inx. Top-half quadrants (masks 0xa, 0x6, 0x2).
    let mut x: i32 = 0;
    while x < inx {
        let xu = x as usize;
        let xlen = kv.xlen[xu] as usize;
        if x < nxplanemin || x >= nxplanemax {
            xv += xlen;
            r1 = vec4_add(r1, cadd1);
            x += 1;
            continue;
        }
        let yv_initial = xv + xlen;
        let mut r0 = r1; // movps r0, r1

        // Forward y: 0..iny  -> mask 0xa.
        let mut xv_local = xv;
        let mut y: i32 = 0;
        while y < iny {
            let yu = y as usize;
            let len = kv.ylen[xu][yu] as usize;
            let v0 = xv_local;
            xv_local += len;
            draw_boundcube_line(&kv.voxels, v0, xv_local, inz, 0xa, r0, &mut callback);
            r0 = vec4_sub(r0, cadd_y); // r0 -= cadd4[4]
            y += 1;
        }

        // Setup for reverse y: r0 = r1 + r2 (= base + (-ysiz)*cadd4[4]),
        // then r1 += cadd4[1] for the next x column.
        let mut yv_local = yv_initial;
        r0 = vec4_add(r1, r2);
        r1 = vec4_add(r1, cadd1);

        // Reverse y: ysiz-1..iny  -> mask 0x6.
        let mut y = ysiz - 1;
        while y > iny {
            r0 = vec4_add(r0, cadd_y); // r0 += cadd4[4]
            let yu = y as usize;
            let len = kv.ylen[xu][yu] as usize;
            let v1_excl = yv_local;
            yv_local -= len;
            draw_boundcube_line(&kv.voxels, yv_local, v1_excl, inz, 0x6, r0, &mut callback);
            y -= 1;
        }

        // Edge y == iny  -> mask 0x2.
        if iny >= 0 && (iny as u32) < kv.ysiz {
            r0 = vec4_add(r0, cadd_y);
            let yu = iny as usize;
            let len = kv.ylen[xu][yu] as usize;
            let v1_excl = yv_local;
            yv_local -= len;
            draw_boundcube_line(&kv.voxels, yv_local, v1_excl, inz, 0x2, r0, &mut callback);
        }

        xv += xlen;
        x += 1;
    }

    // Setup for second half (voxlap5.c:9011): jump r1 to past-end.
    // r1 += (xsiz - x) * cadd4[1]  with x = post-first-half value.
    let dx_remain = (xsiz - x) as f32;
    r1 = vec4_add(r1, vec4_scale(cadd1, dx_remain));

    // Second half: x = xsiz-1..inx (reverse). Bot-half quadrants
    // (masks 0x5, 0x9, 0x1).
    let mut xv2: usize = kv.voxels.len();
    let mut x = xsiz - 1;
    while x > inx {
        let xu = x as usize;
        let xlen = kv.xlen[xu] as usize;
        if x < nxplanemin || x >= nxplanemax {
            xv2 -= xlen;
            r1 = vec4_sub(r1, cadd1);
            x -= 1;
            continue;
        }
        let yv_initial = xv2 - xlen;
        // Voxlap order: r1 -= cadd1 first, then r0 = r1 + r2.
        r1 = vec4_sub(r1, cadd1);
        let mut r0 = vec4_add(r1, r2);

        // Reverse y: ysiz-1..iny  -> mask 0x5.
        let mut xv_local = xv2;
        let mut y = ysiz - 1;
        while y > iny {
            r0 = vec4_add(r0, cadd_y);
            let yu = y as usize;
            let len = kv.ylen[xu][yu] as usize;
            let v1_excl = xv_local;
            xv_local -= len;
            draw_boundcube_line(&kv.voxels, xv_local, v1_excl, inz, 0x5, r0, &mut callback);
            y -= 1;
        }

        // After reverse y: r0 = r1 (movps r0, r1).
        let mut yv_local = yv_initial;
        r0 = r1;

        // Forward y: 0..iny  -> mask 0x9.
        let mut y: i32 = 0;
        while y < iny {
            let yu = y as usize;
            let len = kv.ylen[xu][yu] as usize;
            let v0 = yv_local;
            yv_local += len;
            draw_boundcube_line(&kv.voxels, v0, yv_local, inz, 0x9, r0, &mut callback);
            r0 = vec4_sub(r0, cadd_y);
            y += 1;
        }

        // Edge y == iny  -> mask 0x1.
        if iny >= 0 && (iny as u32) < kv.ysiz {
            let yu = iny as usize;
            let len = kv.ylen[xu][yu] as usize;
            let v0 = yv_local;
            yv_local += len;
            draw_boundcube_line(&kv.voxels, v0, yv_local, inz, 0x1, r0, &mut callback);
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
        r1 = vec4_sub(r1, cadd1);
        let mut r0 = vec4_add(r1, r2);

        // Reverse y -> mask 0x4.
        let mut xv_local = xv2;
        let mut y = ysiz - 1;
        while y > iny {
            r0 = vec4_add(r0, cadd_y);
            let yu = y as usize;
            let len = kv.ylen[xu][yu] as usize;
            let v1_excl = xv_local;
            xv_local -= len;
            draw_boundcube_line(&kv.voxels, xv_local, v1_excl, inz, 0x4, r0, &mut callback);
            y -= 1;
        }

        // After reverse y: r0 = r1.
        let mut yv_local = yv_initial;
        r0 = r1;

        // Forward y -> mask 0x8.
        let mut y: i32 = 0;
        while y < iny {
            let yu = y as usize;
            let len = kv.ylen[xu][yu] as usize;
            let v0 = yv_local;
            yv_local += len;
            draw_boundcube_line(&kv.voxels, v0, yv_local, inz, 0x8, r0, &mut callback);
            r0 = vec4_sub(r0, cadd_y);
            y += 1;
        }

        // Edge y == iny -> mask 0x0.
        if iny >= 0 && (iny as u32) < kv.ysiz {
            let yu = iny as usize;
            let len = kv.ylen[xu][yu] as usize;
            let v0 = yv_local;
            yv_local += len;
            draw_boundcube_line(&kv.voxels, v0, yv_local, inz, 0x0, r0, &mut callback);
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
/// **R6.4a status**: dispatcher runs cull → setup math (mat2 +
/// Cramer's + nfor↔nhei swap + cadd4/ztab4/scisdist/qsum0) → 9-arm
/// per-voxel iteration with the per-voxel callback running
/// [`drawboundcubesse`] through scissor. No pixels written yet —
/// R6.4b adds vertex projection, R6.4c the screen-AABB, R6.4d the
/// fill loop. Returns `false` for "culled" and "would render"
/// alike until R6.4d ships.
#[must_use]
pub fn draw_sprite(cam: &CameraState, settings: &OpticastSettings, sprite: &Sprite) -> bool {
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
    let state = kv6_compute_full_state(&setup, sprite.p, cam, settings);
    kv6_iterate(&state, |voxel, mask, r0| {
        let _ = drawboundcubesse(voxel, mask, &state, r0);
    });
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

    fn oracle_settings() -> OpticastSettings {
        OpticastSettings::for_oracle_framebuffer(640, 480)
    }

    /// Bit-pattern compare for two `[f32; 4]` vectors. The setup
    /// math produces these via deterministic IEEE-754 ops, so
    /// bit-equality is well-defined and dodges `clippy::float_cmp`.
    fn bits4(a: [f32; 4]) -> [u32; 4] {
        a.map(f32::to_bits)
    }

    /// Bytes of the dumped C-oracle meltsphere sprite — used by all
    /// the kv6-load tests below. Module-scope `const` keeps clippy's
    /// `items_after_statements` happy.
    const SPRITE_MELTSPHERE_KV6: &[u8] = include_bytes!("../tests/fixtures/sprite_meltsphere.kv6");

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
        assert!(!draw_sprite(&cam, &oracle_settings(), &s));
    }

    #[test]
    fn kfa_flag_skips_dispatch() {
        let cam = oracle_sprite_front_camera();
        let mut s = Sprite::axis_aligned(cube_kv6(), [1050.0, 1050.0, 175.0]);
        s.flags = SPRITE_FLAG_KFA;
        assert!(!draw_sprite(&cam, &oracle_settings(), &s));
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
        let state =
            kv6_compute_full_state(&setup, [1050.0, 1050.0, 175.0], &cam, &oracle_settings());

        // Every voxel index must fire exactly once. We use a
        // by-pointer identity check via .as_ptr() offsets.
        let voxels_ptr = kv.voxels.as_ptr();
        let mut visited = vec![0u32; kv.voxels.len()];
        let mut total: u32 = 0;
        kv6_iterate(&state, |v, _mask, _r0| {
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
        let kv = roxlap_formats::kv6::parse(SPRITE_MELTSPHERE_KV6).expect("parse fixture");
        assert_eq!(kv.voxels.len(), 401, "fixture voxel count");

        let sprite = Sprite::axis_aligned(kv, [1050.0, 1050.0, 175.0]);
        let cam = oracle_sprite_front_camera();
        let setup = kv6_draw_prepare(&sprite, &cam).expect("oracle sprite must pass cull");
        let state = kv6_compute_full_state(&setup, sprite.p, &cam, &oracle_settings());

        let voxels_ptr = sprite.kv6.voxels.as_ptr();
        let mut visited = vec![0u32; sprite.kv6.voxels.len()];
        let mut total: u32 = 0;
        kv6_iterate(&state, |v, _mask, _r0| {
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
    fn full_state_basic_invariants() {
        // For the oracle sprite_front pose, sanity-check the setup
        // values: ztab4_per_z[0] is zero, ztab4_per_z[k] - ztab4_per_z[k-1]
        // equals cadd4[2], cadd4[3] = cadd4[1] + cadd4[2], cadd4[7] is
        // the 7-bit-OR sum, and r1_initial = (npos*gihz with z2=npos.z)
        // - cadd4[4].
        let kv = roxlap_formats::kv6::parse(SPRITE_MELTSPHERE_KV6).expect("parse fixture");
        let sprite = Sprite::axis_aligned(kv, [1050.0, 1050.0, 175.0]);
        let cam = oracle_sprite_front_camera();
        let setup = kv6_draw_prepare(&sprite, &cam).expect("cull pass");
        let state = kv6_compute_full_state(&setup, sprite.p, &cam, &oracle_settings());

        // ztab4_per_z[0] = [0; 4].
        assert_eq!(bits4(state.ztab4_per_z[0]), bits4([0.0; 4]));

        // For each subsequent z, ztab4_per_z[z] = ztab4_per_z[z-1] + cadd4[2].
        for z in 1..state.ztab4_per_z.len() {
            let want = vec4_add(state.ztab4_per_z[z - 1], state.cadd4[2]);
            assert_eq!(bits4(state.ztab4_per_z[z]), bits4(want), "ztab4_per_z[{z}]");
        }

        // cadd4[3] = cadd4[1] + cadd4[2]; cadd4[5] = cadd4[1] + cadd4[4];
        // cadd4[6] = cadd4[2] + cadd4[4]; cadd4[7] = cadd4[3] + cadd4[4].
        assert_eq!(
            bits4(state.cadd4[3]),
            bits4(vec4_add(state.cadd4[1], state.cadd4[2]))
        );
        assert_eq!(
            bits4(state.cadd4[5]),
            bits4(vec4_add(state.cadd4[1], state.cadd4[4]))
        );
        assert_eq!(
            bits4(state.cadd4[6]),
            bits4(vec4_add(state.cadd4[2], state.cadd4[4]))
        );
        assert_eq!(
            bits4(state.cadd4[7]),
            bits4(vec4_add(state.cadd4[3], state.cadd4[4]))
        );
        assert_eq!(bits4(state.cadd4[0]), bits4([0.0; 4]));

        // r2 = -ysiz * cadd4[4].
        let want_r2 = vec4_scale(state.cadd4[4], -(state.iter.kv.ysiz as f32));
        assert_eq!(bits4(state.r2), bits4(want_r2));
    }

    #[test]
    fn drawboundcubesse_culls_invisible_face_mask() {
        // Synthetic voxel with vis=0 must short-circuit the
        // early-out and not consume the scissor branch.
        let v = Voxel {
            col: 0,
            z: 0,
            vis: 0,
            dir: 0,
        };
        let kv = roxlap_formats::kv6::parse(SPRITE_MELTSPHERE_KV6).expect("parse fixture");
        let sprite = Sprite::axis_aligned(kv, [1050.0, 1050.0, 175.0]);
        let cam = oracle_sprite_front_camera();
        let setup = kv6_draw_prepare(&sprite, &cam).expect("cull pass");
        let state = kv6_compute_full_state(&setup, sprite.p, &cam, &oracle_settings());
        assert!(!drawboundcubesse(
            &v,
            0xff,
            &state,
            [0.0, 0.0, 100.0, 100.0]
        ));
    }

    #[test]
    fn drawboundcubesse_culls_voxel_behind_near_plane() {
        // Force scisdist > 0 by passing an r0 with very small
        // origin.z. Only triggers if scisdist > origin.z; for the
        // oracle sprite_front pose `scisdist` is some small
        // positive number (sum of any negative post-swap basis-z
        // components), so a r0 with z = -1 will cull.
        let v = Voxel {
            col: 0xff,
            z: 0,
            vis: 0xff,
            dir: 0,
        };
        let kv = roxlap_formats::kv6::parse(SPRITE_MELTSPHERE_KV6).expect("parse fixture");
        let sprite = Sprite::axis_aligned(kv, [1050.0, 1050.0, 175.0]);
        let cam = oracle_sprite_front_camera();
        let setup = kv6_draw_prepare(&sprite, &cam).expect("cull pass");
        let state = kv6_compute_full_state(&setup, sprite.p, &cam, &oracle_settings());
        // r0.z = -1000 makes origin.z = -1000 + ztab4_per_z[0].z = -1000.
        // scisdist >= 0; -1000 < scisdist → cull.
        let r0 = [0.0, 0.0, -1000.0, -1000.0];
        assert!(!drawboundcubesse(&v, 0xff, &state, r0));
    }

    #[test]
    fn drawboundcubesse_passes_scissor_for_in_frustum_voxel() {
        // Use the iteration's actual r0 — drawboundcubesse should
        // pass scissor for at least one voxel of the meltsphere
        // sprite. Counts how many voxels pass.
        let kv = roxlap_formats::kv6::parse(SPRITE_MELTSPHERE_KV6).expect("parse fixture");
        let sprite = Sprite::axis_aligned(kv, [1050.0, 1050.0, 175.0]);
        let cam = oracle_sprite_front_camera();
        let setup = kv6_draw_prepare(&sprite, &cam).expect("cull pass");
        let state = kv6_compute_full_state(&setup, sprite.p, &cam, &oracle_settings());

        let mut pass = 0u32;
        kv6_iterate(&state, |v, mask, r0| {
            if drawboundcubesse(v, mask, &state, r0) {
                pass += 1;
            }
        });
        // At least one voxel must pass; full-rasterizer (R6.4d)
        // bit-exact validation is the next gate.
        assert!(pass > 0, "expected at least one voxel to pass scissor");
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
        assert!(!draw_sprite(&cam, &oracle_settings(), &s));
    }
}
