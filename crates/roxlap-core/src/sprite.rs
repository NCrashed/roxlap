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
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::cast_ptr_alignment, // _mm_loadl_epi64 / _mm_storeu_si128 are intentionally unaligned
    clippy::doc_markdown,
    clippy::no_effect_underscore_binding, // SSE intrinsic side-effect-only stores
    clippy::no_effect, // the discarded pmaddwd intermediate
    clippy::ref_as_ptr,
    clippy::float_cmp_const,
    clippy::float_cmp,
)]

use roxlap_formats::kv6::{Kv6, Voxel};

use crate::camera_math::CameraState;
use crate::engine::{Engine, LightSrc, DEFAULT_KV6COL};
use crate::equivec::iunivec;
use crate::fixed::ftol;
use crate::opticast::OpticastSettings;
use crate::ptfaces16::PTFACES16;

/// Voxlap's `MAXLIGHTS` cap (`voxlap5.c`). Used to size the
/// ambient-plus-N-lights `lightlist` scratch in `update_reflects`'s
/// lightmode≥2 branch.
const MAX_LIGHTS: usize = 16;

/// Voxlap's `vx5.kv6mipfactor` default (`voxlap5.c:12335`). Threshold
/// distance (in voxlap's "ftol-of-forward-projected" estimate units)
/// above which kv6draw walks the lowermip chain. Roxlap doesn't yet
/// model the lowermip chain in `roxlap-formats::Kv6`, so the mip
/// descent loop in `kv6_draw_prepare` is structurally faithful but
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
pub(crate) fn mat2(
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
/// voxlap5.c:8915-8973 setup block + the qsum1/qbplbpp framebuffer
/// state from `voxsetframebuffer` (voxlap5.c:11119-11122) +
/// kv6colmul/kv6coladd from `updatereflects` (voxlap5.c:8466).
#[derive(Debug, Clone)]
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
    /// Viewport-clip floor (voxlap5.c:11120). After the saturated
    /// add of `qsum0`, the screen-AABB is clamped from below by
    /// `qsum1[0/2] = 0x7fff - xres`, `qsum1[1/3] = 0x7fff - yres`.
    pub qsum1: [i16; 4],
    /// Framebuffer pixel-stride packed for `pmaddwd` (voxlap5.c:11121).
    /// `qbplbpp[0/2] = 4` (bytes per pixel), `qbplbpp[1/3] = pitch_bytes`.
    pub qbplbpp: [i16; 4],
    /// Per-direction colour modulation table built by
    /// [`update_reflects`]. Indexed by `v.dir` (256 entries). Each
    /// entry packs four `u16` modulation factors (one per byte
    /// channel) used by `_mm_mulhi_epu16` against the unpacked
    /// voxel colour.
    pub kv6colmul: Box<[u64; 256]>,
    /// Fog bias added after the colour modulate. Zero when fog is
    /// disabled (the oracle case).
    pub kv6coladd: u64,
}

/// Borrowed framebuffer + zbuffer the per-voxel rasterizer fills.
///
/// Mirrors voxlap's `kv6frameplace` + `zbuffermem` but in
/// row-major-pixel form rather than byte-pointer form. `width` /
/// `height` must match the `OpticastSettings.xres` / `yres` used
/// when the per-frame `Kv6FullState` was built — the bounds derived from
/// `qsum0` / `qsum1` assume that geometry.
pub struct DrawTarget<'a> {
    /// Packed BGRA pixels, row-major, length `pitch_pixels * height`.
    pub framebuffer: &'a mut [u32],
    /// Per-pixel z values, same layout as `framebuffer`.
    pub zbuffer: &'a mut [f32],
    /// Row stride in pixels (= `framebuffer.len() / height`).
    pub pitch_pixels: usize,
    pub width: u32,
    pub height: u32,
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

/// Sprite lighting + colour state — the subset of voxlap's
/// `vx5` global that `updatereflects` reads. Built once per
/// frame from [`Engine`] state and passed to [`draw_sprite`].
///
/// All fields mirror voxlap names:
/// - `kv6col` ↔ `vx5.kv6col`
/// - `lightmode` ↔ `vx5.lightmode`
/// - `lights` ↔ `vx5.lightsrc[0..vx5.numlights]`
///
/// The `vx5.fogcol`/`ofogdist` fog plumbing is deferred — sprite
/// fog stays off for now, matching the oracle path
/// (`vx5.fogcol < 0` ⇒ `ofogdist == -1` in voxlap C, no fog).
#[derive(Debug, Clone, Copy)]
pub struct SpriteLighting<'a> {
    /// Material colour. R==G==B triggers the cheaper nolighta path
    /// in `update_reflects`; arbitrary RGB takes the per-channel
    /// nolightb path; lightmode≥2 ignores the R==G==B fast path
    /// and always does per-channel modulation.
    pub kv6col: u32,
    /// `0` / `1` → directional surface tint (lightmode<2 paths).
    /// `2` → per-light shadow-side modulation against `lights`.
    pub lightmode: u32,
    /// Active point lights — voxlap's `vx5.lightsrc[..vx5.numlights]`.
    /// Empty for lightmode<2; populated for lightmode≥2.
    pub lights: &'a [LightSrc],
}

impl<'a> SpriteLighting<'a> {
    /// Snapshot the lighting + colour subset of an [`Engine`].
    /// Use this once per frame in the host so the sprite render
    /// reflects engine setters made between frames.
    #[must_use]
    pub fn from_engine(engine: &'a Engine) -> Self {
        Self {
            kv6col: engine.kv6col(),
            lightmode: engine.lightmode(),
            lights: engine.lights(),
        }
    }
}

impl SpriteLighting<'static> {
    /// Default oracle config — grey `kv6col`, lightmode 0, no
    /// lights. Used by `roxlap-oracle` so the four sprite golden
    /// hashes stay byte-stable: this is the exact state voxlap C's
    /// oracle has when it calls `drawsprite`.
    #[must_use]
    pub fn default_oracle() -> Self {
        Self {
            kv6col: DEFAULT_KV6COL,
            lightmode: 0,
            lights: &[],
        }
    }
}

/// Builds `kv6colmul[256]` + `kv6coladd[0]` from the engine's
/// sprite lighting state. Mirror of voxlap's `updatereflects`
/// (`voxlap5.c:8466-8750`).
///
/// Branches:
/// - `lightmode < 2` + R==G==B `kv6col` → nolighta (cheap
///   single-multiplier path, voxlap5.c:8553-8584).
/// - `lightmode < 2` + arbitrary `kv6col` → nolightb (per-channel
///   path, voxlap5.c:8587-8629).
/// - `lightmode >= 2` → per-light shadow-side modulation
///   (voxlap5.c:8631-8750), iterating the active `lights`.
///
/// `flags & 1` (disable shading) and the active-fog path remain
/// deferred — neither is exercised by the oracle's four sprite
/// poses, and adding them is a follow-up that doesn't change the
/// already-frozen hashes.
///
fn update_reflects(sprite: &Sprite, lighting: &SpriteLighting<'_>) -> (Box<[u64; 256]>, u64) {
    // Sprite fog plumbing is a follow-up — `vx5.fogcol < 0` (voxlap
    // C oracle's set_fogcol(BR(...)) state) means ofogdist stays -1,
    // fogmul = 0, kv6coladd[0] = 0. We pin to that here.
    let fogmul_lo: u32 = 0;
    let kv6coladd: u64 = 0;

    let kv6col = lighting.kv6col;

    // g = ((fogmul & 32767) ^ 32767) * (16*8/65536). With fogmul=0:
    //   g = 32767 * (128/65536) ≈ 63.998.
    let g_pre = ((((fogmul_lo & 0x7fff) ^ 0x7fff) as i32) as f32) * (16.0 * 8.0 / 65536.0);

    let mut kv6colmul = Box::new([0u64; 256]);

    if lighting.lightmode < 2 {
        // (voxlap5.c:8538-8543) fx=fy=fz=1.0; tp = sum of basis vectors.
        let tp_x = sprite.s[0] + sprite.h[0] + sprite.f[0];
        let tp_y = sprite.s[1] + sprite.h[1] + sprite.f[1];
        let tp_z = sprite.s[2] + sprite.h[2] + sprite.f[2];

        let f0 = 64.0_f32 / (tp_x * tp_x + tp_y * tp_y + tp_z * tp_z).sqrt();

        // R==G==B test: ((kv6col & 0xffff) << 8) ^ (kv6col & 0xffff00)
        //   == 0  iff  R == G and G == B.
        let lo16 = kv6col & 0xffff;
        let mid24 = kv6col & 0x00ff_ff00;
        let is_grey = ((lo16 << 8) ^ mid24) == 0;

        if is_grey {
            // Nolighta path (voxlap5.c:8553-8584): grey kv6col absorbs
            // into a single multiplier per direction.
            let g = g_pre * (((kv6col & 0xff) as f32) / 256.0);
            let f = f0 * g;

            let l0 = (tp_x * f) as i16; // (short)(...) is C truncating cast
            let l1 = (tp_y * f) as i16;
            let l2 = (tp_z * f) as i16;
            let l3 = (g * 128.0) as i16;

            let iu = iunivec();
            for k in 0..256 {
                let w = dot_iunivec_i16x4(iu[k], [l0, l1, l2, l3]);
                let w64 = u64::from(w);
                kv6colmul[k] = w64 | (w64 << 16) | (w64 << 32) | (w64 << 48);
            }
        } else {
            // Nolightb path (voxlap5.c:8587-8629). Per-channel
            // modulation factor M_k = (kv6col_byte_k << 8) → mulhi_pu16
            // by the per-direction dot. Same dot derivation as nolighta.
            let f = f0 * g_pre;

            let l0 = (tp_x * f) as i16;
            let l1 = (tp_y * f) as i16;
            let l2 = (tp_z * f) as i16;
            let l3 = (g_pre * 128.0) as i16;

            let m = kv6col_channel_mods(kv6col);

            let iu = iunivec();
            for k in 0..256 {
                let w = dot_iunivec_i16x4(iu[k], [l0, l1, l2, l3]);
                kv6colmul[k] = pack_modulated_word(w, m);
            }
        }
    } else {
        // Lightmode≥2 path (voxlap5.c:8631-8750): per-sprite point
        // lighting from `lighting.lights`. Each light projects onto
        // the sprite's normalised basis; per-direction kv6colmul[i]
        // starts from a synthetic ambient slot and subtracts shadow
        // contributions from each light's "negative" lanes.
        let m = kv6col_channel_mods(kv6col);
        build_kv6colmul_lightmode2(sprite, lighting.lights, &mut kv6colmul, fogmul_lo, m);
    }

    (kv6colmul, kv6coladd)
}

/// Voxlap's `pmaddwd(iunivec[k], lightlist) summed across two
/// dword lanes mod 2^32, take high 16` reduction. Returns the
/// `u16` modulation factor before any per-channel packing.
#[inline]
fn dot_iunivec_i16x4(u: [i16; 4], l: [i16; 4]) -> u16 {
    let u0 = i32::from(u[0]);
    let u1 = i32::from(u[1]);
    let u2 = i32::from(u[2]);
    let u3 = i32::from(u[3]);
    let lo = (u0.wrapping_mul(l[0].into())) as u32;
    let lo = lo.wrapping_add((u1.wrapping_mul(l[1].into())) as u32);
    let hi = (u2.wrapping_mul(l[2].into())) as u32;
    let hi = hi.wrapping_add((u3.wrapping_mul(l[3].into())) as u32);
    ((lo.wrapping_add(hi)) >> 16) as u16
}

/// `(kv6col_byte_k << 8)` per channel — the four `M_k` factors the
/// nolightb / lightmode≥2 paths multiply against the per-direction
/// dot via `pmulhuw`.
#[inline]
fn kv6col_channel_mods(kv6col: u32) -> [u16; 4] {
    [
        ((kv6col & 0xff) << 8) as u16,
        (((kv6col >> 8) & 0xff) << 8) as u16,
        (((kv6col >> 16) & 0xff) << 8) as u16,
        (((kv6col >> 24) & 0xff) << 8) as u16,
    ]
}

/// Pack one direction's `kv6colmul[k]` u64: per-channel
/// `(W * M_c) >> 16` words concatenated.
#[inline]
fn pack_modulated_word(w_dot: u16, m: [u16; 4]) -> u64 {
    let w = u32::from(w_dot);
    let w0 = ((w * u32::from(m[0])) >> 16) as u16;
    let w1 = ((w * u32::from(m[1])) >> 16) as u16;
    let w2 = ((w * u32::from(m[2])) >> 16) as u16;
    let w3 = ((w * u32::from(m[3])) >> 16) as u16;
    u64::from(w0) | (u64::from(w1) << 16) | (u64::from(w2) << 32) | (u64::from(w3) << 48)
}

/// Lightmode≥2 path body — voxlap5.c:8631-8750. Builds the full
/// `kv6colmul[256]` from the active light list.
///
/// Steps:
/// 1. Normalise each sprite-basis axis (`sprs`/`sprh`/`sprf`).
/// 2. For each light within `r2` of the sprite, compute its
///    intensity falloff `h` and project the world-space delta onto
///    the normalised sprite basis → store in `lightlist[k]`.
/// 3. Append a synthetic ambient slot (voxlap's hardcoded
///    `(fx, fy, fz) = (0, 0.5, 1.0)` direction) at
///    `lightlist[lightcnt]`.
/// 4. For each direction `idx ∈ 0..256`:
///    - `base = ambient_slot · iunivec[idx]` (treated as one u32).
///    - For each real light `k`: compute `dot = light_k ·
///      iunivec[idx]`, split into low/high i16 lanes (asm-faithful
///      "16-bits-is-ugly-but-ok-here" quirk); subtract the negative
///      lanes from `base` (= shadow side of the surface).
///    - `W = base >> 16`, then per-channel modulate against `M_c`
///      and pack into `kv6colmul[idx]`.
fn build_kv6colmul_lightmode2(
    sprite: &Sprite,
    lights: &[LightSrc],
    kv6colmul: &mut [u64; 256],
    fogmul_lo: u32,
    m: [u16; 4],
) {
    // (voxlap5.c:8638-8643) Normalise sprite basis. WARNING from
    // voxlap: only correct for orthonormal sprite-bases; non-
    // orthogonal bases (e.g. shears) drift. The four oracle sprite
    // poses are all orthonormal so this matches voxlap's behaviour.
    let sprs = normalise(sprite.s);
    let sprh = normalise(sprite.h);
    let sprf = normalise(sprite.f);

    // hh = ((fogmul & 32767) ^ 32767) / 65536 * 2 (voxlap5.c:8645).
    // With fogmul=0 → hh = 32767 / 65536 * 2 ≈ 1.0. This is a
    // distinct scaling from `g_pre` (= same numerator * 128/65536
    // for the lightmode<2 path) — they differ by a factor of 64.
    // An earlier port mistakenly derived hh from g_pre / 128 = 0.5,
    // giving sprites half the intended ambient brightness.
    let hh_initial = ((((fogmul_lo & 0x7fff) ^ 0x7fff) as i32) as f32) * (2.0 / 65536.0);

    // Project each in-range light onto the sprite basis.
    let mut lightlist: [[i16; 4]; MAX_LIGHTS + 1] = [[0; 4]; MAX_LIGHTS + 1];
    let mut lightcnt: usize = 0;
    for light in lights.iter().rev() {
        if lightcnt >= MAX_LIGHTS {
            break;
        }
        let fx = light.pos[0] - sprite.p[0];
        let fy = light.pos[1] - sprite.p[1];
        let fz = light.pos[2] - sprite.p[2];
        let gg = fx * fx + fy * fy + fz * fz;
        let ff = light.r2;
        // Voxlap's `*(int32_t *)&gg < *(int32_t *)&ff` is a bit-
        // pattern compare. For non-negative finite floats the bit
        // order matches the magnitude order, so `gg < ff` is
        // equivalent (and safer in the presence of NaN: NaN !< x
        // for any x, matching voxlap's float-bit-cast trick).
        if gg >= ff || gg <= 0.0 {
            continue;
        }
        let f = ff.sqrt();
        let g = gg.sqrt();
        // h = (f*ff - g*gg) / (f*ff*g*gg) * sc * 16
        let mut h = (f * ff - g * gg) / (f * ff * g * gg) * light.sc * 16.0;
        if g * h > 4096.0 {
            h = 4096.0 / g; // saturation clip
        }
        h *= hh_initial;
        let l0 = (fx * sprs[0] + fy * sprs[1] + fz * sprs[2]) * h;
        let l1 = (fx * sprh[0] + fy * sprh[1] + fz * sprh[2]) * h;
        let l2 = (fx * sprf[0] + fy * sprf[1] + fz * sprf[2]) * h;
        lightlist[lightcnt] = [l0 as i16, l1 as i16, l2 as i16, 0];
        lightcnt += 1;
    }

    // Synthetic ambient slot: voxlap's hardcoded direction
    // (fx, fy, fz) = (0, 0.5, 1.0) projected onto the sprite basis,
    // scaled by `hh * 16*16*8/2 = hh * 1024`. The lane-3 bias is
    // `hh * 48 / 16 = hh * 3`.
    let amb_fx = 0.0_f32;
    let amb_fy = 0.5_f32;
    let amb_fz = 1.0_f32;
    let hh = hh_initial * (16.0 * 16.0 * 8.0 / 2.0);
    let al0 = (sprs[0] * amb_fx + sprs[1] * amb_fy + sprs[2] * amb_fz) * hh;
    let al1 = (sprh[0] * amb_fx + sprh[1] * amb_fy + sprh[2] * amb_fz) * hh;
    let al2 = (sprf[0] * amb_fx + sprf[1] * amb_fy + sprf[2] * amb_fz) * hh;
    let al3 = hh * (48.0 / 16.0);
    lightlist[lightcnt] = [al0 as i16, al1 as i16, al2 as i16, al3 as i16];

    let iu = iunivec();
    for idx in 0..256 {
        let u = iu[idx];
        // Ambient base = lightlist[lightcnt] · iunivec[idx], in u32
        // wrapping arithmetic (asm summed the pmaddwd dword lanes
        // mod 2^32).
        let u0 = i32::from(u[0]);
        let u1 = i32::from(u[1]);
        let u2 = i32::from(u[2]);
        let u3 = i32::from(u[3]);
        let amb = lightlist[lightcnt];
        let base_lo = (u0.wrapping_mul(i32::from(amb[0]))) as u32;
        let base_lo = base_lo.wrapping_add((u1.wrapping_mul(i32::from(amb[1]))) as u32);
        let base_hi = (u2.wrapping_mul(i32::from(amb[2]))) as u32;
        let base_hi = base_hi.wrapping_add((u3.wrapping_mul(i32::from(amb[3]))) as u32);
        let mut base = base_lo.wrapping_add(base_hi);

        // For each real light, compute dot, then subtract its
        // "negative" half-lanes from `base` (= shadow side).
        for k in (0..lightcnt).rev() {
            let l = lightlist[k];
            let klo = (u0.wrapping_mul(i32::from(l[0]))) as u32;
            let klo = klo.wrapping_add((u1.wrapping_mul(i32::from(l[1]))) as u32);
            let khi = (u2.wrapping_mul(i32::from(l[2]))) as u32;
            let khi = khi.wrapping_add((u3.wrapping_mul(i32::from(l[3]))) as u32);
            let dot = klo.wrapping_add(khi);
            // Voxlap quirk: 32-bit dot but pminsw is per-i16 lane.
            // Light magnitudes stay clamped enough that the
            // mixed-lane behaviour is benign — port faithfully.
            let lo16 = (dot & 0xffff) as i16;
            let hi16 = ((dot >> 16) & 0xffff) as i16;
            let lo16c: u16 = if lo16 < 0 { lo16 as u16 } else { 0 };
            let hi16c: u16 = if hi16 < 0 { hi16 as u16 } else { 0 };
            let sub = (u32::from(hi16c) << 16) | u32::from(lo16c);
            base = base.wrapping_sub(sub);
        }

        let w_dot = (base >> 16) as u16;
        kv6colmul[idx] = pack_modulated_word(w_dot, m);
    }
}

/// Normalise a 3-vector. Returns the unit-length version; if
/// the input is zero-length, returns the input unchanged (avoids
/// NaN propagation — voxlap's `1.0 / sqrt(...)` would NaN out for
/// a zero basis axis but the C code never gets passed one).
#[inline]
fn normalise(v: [f32; 3]) -> [f32; 3] {
    let len_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if len_sq <= 0.0 {
        return v;
    }
    let inv = 1.0 / len_sq.sqrt();
    [v[0] * inv, v[1] * inv, v[2] * inv]
}

/// Full setup: mat2 + Cramer's + nfor↔nhei swap + cadd4/ztab4/r1/r2/
/// scisdist/qsum0 init. Mirror of voxlap5.c:8915-8973.
pub(crate) fn kv6_compute_full_state<'a>(
    setup: &Kv6DrawSetup<'a>,
    sprite: &Sprite,
    lighting: &SpriteLighting<'_>,
    cam: &CameraState,
    settings: &OpticastSettings,
    fb_width: u32,
    fb_height: u32,
    fb_pitch_pixels: usize,
) -> Kv6FullState<'a> {
    let sprite_pos = sprite.p;
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

    // qsum1 + qbplbpp from voxsetframebuffer (voxlap5.c:11119-11122).
    // The framebuffer geometry is independent of the camera projection
    // — these are derived from `(width, height, pitch_bytes)`.
    let pitch_bytes = (fb_pitch_pixels as i32).saturating_mul(4);
    let qsum1_x = 0x7fff_i32 - fb_width as i32;
    let qsum1_y = 0x7fff_i32 - fb_height as i32;
    let qsum1 = [
        qsum1_x as i16,
        qsum1_y as i16,
        qsum1_x as i16,
        qsum1_y as i16,
    ];
    let qbplbpp = [4i16, pitch_bytes as i16, 4, pitch_bytes as i16];

    let (kv6colmul, kv6coladd) = update_reflects(sprite, lighting);

    Kv6FullState {
        iter,
        cadd4,
        ztab4_per_z,
        r1_initial,
        r2,
        scisdist,
        qsum0,
        qsum1,
        qbplbpp,
        kv6colmul,
        kv6coladd,
    }
}

/// Per-voxel rasterizer (R6.4 complete).
///
/// Mirror of `voxlap5.c:8179-8320` (`drawboundcubesse`). For each
/// voxel:
/// 1. `effmask = mask & v.vis` early-out.
/// 2. `origin = r0 + ztab4_per_z[v.z]`; scissor on `origin.z`.
/// 3. Look up `ptfaces16[effmask]` — `face[0]` = 4 or 6 vertex
///    count, `face[1..7]` = byte offsets into `caddasm` (the
///    `cadd4[8]` array, each entry 16 bytes).
/// 4. For each vertex pair (a, b), compute the projected screen
///    coords as `(cadd4[a] + origin).xy / (cadd4[a] + origin).z`
///    via `_mm_rcp_ps`.
/// 5. Pack the 4 (or 6) projected vertices to int16, min/max-reduce
///    to a single screen-AABB, viewport-clip via `qsum0` /
///    `qsum1`, and early-out on degenerate rect.
/// 6. Compute the per-voxel colour via the `mm5` cross-call tail +
///    `kv6colmul[v.dir]` + `kv6coladd[0]` modulation.
/// 7. Fill the screen rectangle with z-test + framebuffer write.
///
/// Returns the number of pixels actually written (z-test passing).
/// Tests use this as a sanity gate; production callers ignore it.
///
/// `mm5_tail` is voxlap's static cross-call register tail
/// (voxlap5.c:8170-8177). It carries one byte of contribution from
/// the previous voxel's colour into the current; bit-equality with
/// the asm requires preserving it across calls within one sprite.
///
/// Currently x86_64-only — relies on `_mm_rcp_ps` for bit-equality
/// with voxlap C. NEON / wasm ports will need their own goldens
/// (see `PORTING-RUST.md` R9 / R10).
#[cfg(target_arch = "x86_64")]
#[allow(clippy::trivially_copy_pass_by_ref)] // hot loop; matches voxlap's pointer-passed v.
pub(crate) fn drawboundcubesse(
    v: &Voxel,
    mask: u32,
    state: &Kv6FullState<'_>,
    r0: [f32; 4],
    mm5_tail: &mut u32,
    target: &mut DrawTarget<'_>,
) -> u32 {
    use core::arch::x86_64::{
        __m128, __m128i, _mm_add_epi16, _mm_add_ps, _mm_adds_epi16, _mm_cvtsi128_si32,
        _mm_cvtsi32_si128, _mm_cvttps_epi32, _mm_loadl_epi64, _mm_loadu_ps, _mm_madd_epi16,
        _mm_max_epi16, _mm_min_epi16, _mm_movehl_ps, _mm_movelh_ps, _mm_mul_ps, _mm_mulhi_epu16,
        _mm_packs_epi32, _mm_packus_epi16, _mm_rcp_ps, _mm_setzero_si128, _mm_shufflelo_epi16,
        _mm_storeu_ps, _mm_storeu_si128, _mm_subs_epu16, _mm_unpackhi_epi64, _mm_unpacklo_epi32,
        _mm_unpacklo_epi8,
    };

    let effmask = (mask & u32::from(v.vis)) as usize;
    if effmask == 0 || effmask >= PTFACES16.len() {
        return 0;
    }
    let face = PTFACES16[effmask];
    if face[0] == 0 {
        return 0;
    }

    // origin = r0 + ztab4_per_z[v.z] (4 f32 lanes, [x*hz, y*hz, z, z]).
    let z_idx = v.z as usize;
    if z_idx >= state.ztab4_per_z.len() {
        return 0;
    }
    let ztep = state.ztab4_per_z[z_idx];
    // SAFETY: `_mm_loadu_ps` reads 16 unaligned bytes from a 4-f32
    // array (which is 16 bytes); subsequent intrinsics are SSE2
    // baseline on x86_64.
    unsafe {
        let r0_v = _mm_loadu_ps(r0.as_ptr());
        let ztep_v = _mm_loadu_ps(ztep.as_ptr());
        let origin_v: __m128 = _mm_add_ps(r0_v, ztep_v);
        let mut origin_arr = [0.0f32; 4];
        _mm_storeu_ps(origin_arr.as_mut_ptr(), origin_v);
        if origin_arr[2] < state.scisdist {
            return 0;
        }

        // Project vertex pair (a, b). Returns __m128 with lanes:
        //   [b.x_proj, b.y_proj, a.x_proj, a.y_proj]
        // The byte offsets in face[k] index `caddasm` (= bytes into a
        // [point4d; 8] = [[f32; 4]; 8]); divide by 16 (= sizeof point4d)
        // to land back at the cadd4 index.
        let project = |off_a: u8, off_b: u8| -> __m128 {
            let a = state.cadd4[(off_a >> 4) as usize];
            let b = state.cadd4[(off_b >> 4) as usize];
            let wva = _mm_add_ps(_mm_loadu_ps(a.as_ptr()), origin_v);
            let wvb = _mm_add_ps(_mm_loadu_ps(b.as_ptr()), origin_v);
            let wv0 = _mm_movehl_ps(wva, wvb); // [b.z, b.z, a.z, a.z]
            let wv1 = _mm_movelh_ps(wvb, wva); // [b.x, b.y, a.x, a.y]
            let wv0_inv = _mm_rcp_ps(wv0);
            _mm_mul_ps(wv0_inv, wv1)
        };

        let pair01 = project(face[1], face[2]);
        let pair23 = project(face[3], face[4]);

        // Convert to int32 (truncate-toward-zero), pack to int16.
        // pack01_int16 lanes 0..3 = [v1x, v1y, v0x, v0y]
        // pack01_int16 lanes 4..7 = [v3x, v3y, v2x, v2y]
        let p01_i32 = _mm_cvttps_epi32(pair01);
        let p23_i32 = _mm_cvttps_epi32(pair23);
        let pack_lo = _mm_packs_epi32(p01_i32, p23_i32);
        let pack01 = pack_lo;
        let pack23 = _mm_unpackhi_epi64(pack_lo, _mm_setzero_si128());
        let mut mm_min = _mm_min_epi16(pack01, pack23);
        let mut mm_max = _mm_max_epi16(pack01, pack23);

        if face[0] != 4 {
            let pair45 = project(face[5], face[6]);
            let p45_i32 = _mm_cvttps_epi32(pair45);
            let pack45 = _mm_packs_epi32(p45_i32, _mm_setzero_si128());
            mm_min = _mm_min_epi16(mm_min, pack45);
            mm_max = _mm_max_epi16(mm_max, pack45);
        }

        // shufflelo(_, 0x0e) brings high half (lanes 2..3) into low
        // half so min/max collapses across all 4 (or 6) vertices.
        let mm_min_hi = _mm_shufflelo_epi16(mm_min, 0x0e);
        let mm_max_hi = _mm_shufflelo_epi16(mm_max, 0x0e);
        let mm_min_red = _mm_min_epi16(mm_min, mm_min_hi);
        let mm_max_red = _mm_max_epi16(mm_max, mm_max_hi);

        // bounds = unpacklo(mm_min, mm_max) lanes 0..3 (i16)
        //        = [min_x, max_x, min_y, max_y]  ?
        // Actually: _mm_unpacklo_epi32 interleaves 32-bit lanes.
        // Low 32 of mm_min = (mm_min[0], mm_min[1]) i.e. (min_x, min_y).
        // Low 32 of mm_max similarly. After unpacklo_epi32:
        //   lanes_32[0] = mm_min low32, lanes_32[1] = mm_max low32
        //   → 4 i16: [min_x, min_y, max_x, max_y]
        let bounds = _mm_unpacklo_epi32(mm_min_red, mm_max_red);

        // Apply qsum0 (saturated add) + qsum1 (max-floor). Both are
        // 8-byte values loaded into the low 64 bits of __m128i.
        let qsum0_v = _mm_loadl_epi64(state.qsum0.as_ptr().cast::<__m128i>());
        let qsum1_v = _mm_loadl_epi64(state.qsum1.as_ptr().cast::<__m128i>());
        let bounds = _mm_adds_epi16(bounds, qsum0_v);
        let bounds = _mm_max_epi16(bounds, qsum1_v);

        // dxdy = subs_epu16(bounds_hi, bounds) — saturating unsigned
        // subtract, with bounds_hi being lanes [2,3,2,3] of bounds.
        let bounds_hi = _mm_shufflelo_epi16(bounds, 0xee);
        let dxdy = _mm_subs_epu16(bounds_hi, bounds);
        let dxdy_low = _mm_cvtsi128_si32(dxdy) as u32;
        let dx = (dxdy_low & 0xffff) as i32;
        if dx == 0 {
            return 0;
        }
        let dy = ((dxdy_low >> 16) as i32) - 1;
        if dy < 0 {
            return 0;
        }

        // Recover pixel coords from bounds + qsum1. Bounds[0/1] are
        // currently in the saturated [0x7fff - res, 0x7fff] range;
        // pixel = bounds - qsum1.
        let mut bounds_arr = [0i16; 8];
        _mm_storeu_si128(bounds_arr.as_mut_ptr().cast::<__m128i>(), bounds);
        let pixel_min_x = i32::from(bounds_arr[0]) - i32::from(state.qsum1[0]);
        let pixel_min_y = i32::from(bounds_arr[1]) - i32::from(state.qsum1[1]);

        // pmaddwd is consumed for completeness so the asm-equivalent
        // pixel-byte-offset is computable; not strictly needed since
        // we index directly via (pixel_min_x, pixel_min_y).
        let qbplbpp_v = _mm_loadl_epi64(state.qbplbpp.as_ptr().cast::<__m128i>());
        let _ = _mm_madd_epi16(bounds, qbplbpp_v);

        // Colour modulation with mm5 cross-call tail.
        let tail_in = *mm5_tail;
        let mm5 = _mm_cvtsi32_si128(tail_in as i32);
        let col_v = _mm_cvtsi32_si128(v.col as i32);
        let mm5 = _mm_unpacklo_epi8(mm5, col_v);
        let kvm = state.kv6colmul[v.dir as usize];
        let kvm_v = _mm_loadl_epi64(std::ptr::addr_of!(kvm).cast::<__m128i>());
        let mm5 = _mm_mulhi_epu16(mm5, kvm_v);
        let kva_v = _mm_loadl_epi64(std::ptr::addr_of!(state.kv6coladd).cast::<__m128i>());
        let mm5 = _mm_add_epi16(mm5, kva_v);
        let mm5 = _mm_packus_epi16(mm5, mm5);
        let color = _mm_cvtsi128_si32(mm5) as u32;
        *mm5_tail = color;

        // Fill rectangle [pixel_min_x .. +dx) × [pixel_min_y .. +dy+1).
        // The qsum0/qsum1 clip + saturating sub guarantee the rect
        // sits inside the framebuffer, so no per-pixel bounds check
        // needed beyond Rust's slice indexing.
        let z_val = origin_arr[2];
        let pitch = target.pitch_pixels;
        let x0 = pixel_min_x as usize;
        let x_end = x0 + dx as usize;
        let mut written: u32 = 0;
        for row in 0..=(dy as usize) {
            let y = pixel_min_y as usize + row;
            let row_start = y * pitch;
            for x in x0..x_end {
                let idx = row_start + x;
                if z_val < target.zbuffer[idx] {
                    target.zbuffer[idx] = z_val;
                    target.framebuffer[idx] = color;
                    written += 1;
                }
            }
        }
        written
    }
}

/// Non-x86_64 fallback. Same signature, no rendering. R9 / R10 will
/// add NEON / wasm ports with their own goldens.
#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::trivially_copy_pass_by_ref)]
pub(crate) fn drawboundcubesse(
    _v: &Voxel,
    _mask: u32,
    _state: &Kv6FullState<'_>,
    _r0: [f32; 4],
    _mm5_tail: &mut u32,
    _target: &mut DrawTarget<'_>,
) -> u32 {
    0
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

/// Draw a sprite into a framebuffer + z-buffer.
///
/// Top-level dispatcher mirroring voxlap5.c:9818-9828:
/// - Skips on `flags & INVISIBLE`.
/// - Skips on `flags & KFA` (animation path; out of scope for R6).
/// - Skips on `flags & NO_Z` (handled by `drawboundcubenozsse`,
///   not yet ported — the four oracle sprite poses all use z-tested
///   rendering).
///
/// Otherwise: cull → setup math → 9-arm per-voxel iteration →
/// per-voxel rasterize via the R6.4 `drawboundcubesse` port.
///
/// Returns the total number of pixels written across all voxels of
/// the sprite (== sum of z-test passes). Zero means the sprite
/// produced no visible pixels (culled, fully behind near plane, or
/// totally occluded).
pub fn draw_sprite(
    target: &mut DrawTarget<'_>,
    cam: &CameraState,
    settings: &OpticastSettings,
    lighting: &SpriteLighting<'_>,
    sprite: &Sprite,
) -> u32 {
    if sprite.flags & SPRITE_FLAG_INVISIBLE != 0 {
        return 0;
    }
    if sprite.flags & SPRITE_FLAG_KFA != 0 {
        return 0;
    }
    if sprite.flags & SPRITE_FLAG_NO_Z != 0 {
        // drawboundcubenozsse port deferred; oracle doesn't exercise it.
        return 0;
    }
    let Some(setup) = kv6_draw_prepare(sprite, cam) else {
        return 0;
    };
    let state = kv6_compute_full_state(
        &setup,
        sprite,
        lighting,
        cam,
        settings,
        target.width,
        target.height,
        target.pitch_pixels,
    );
    let mut mm5_tail: u32 = 0;
    let mut total_written: u32 = 0;
    kv6_iterate(&state, |voxel, mask, r0| {
        total_written += drawboundcubesse(voxel, mask, &state, r0, &mut mm5_tail, target);
    });
    total_written
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

    /// Test-only ergonomic shim: build a Kv6FullState with the
    /// oracle 640×480 framebuffer geometry. Mirrors the
    /// pre-R6.4 signature so tests don't have to spell out
    /// width/height/pitch every time.
    fn compute_state_for_test<'a>(
        setup: &Kv6DrawSetup<'a>,
        sprite: &Sprite,
        cam: &camera_math::CameraState,
    ) -> Kv6FullState<'a> {
        let lighting = SpriteLighting::default_oracle();
        kv6_compute_full_state(
            setup,
            sprite,
            &lighting,
            cam,
            &oracle_settings(),
            640,
            480,
            640,
        )
    }

    /// Allocate a 640×480 framebuffer + zbuffer (zbuffer pre-filled
    /// with f32::INFINITY so any voxel passes the z-test on first
    /// write).
    fn alloc_target() -> (Vec<u32>, Vec<f32>) {
        let pixels = 640usize * 480usize;
        (vec![0u32; pixels], vec![f32::INFINITY; pixels])
    }

    fn make_target<'a>(fb: &'a mut [u32], zb: &'a mut [f32]) -> DrawTarget<'a> {
        DrawTarget {
            framebuffer: fb,
            zbuffer: zb,
            pitch_pixels: 640,
            width: 640,
            height: 480,
        }
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
        let (mut fb, mut zb) = alloc_target();
        let mut target = make_target(&mut fb, &mut zb);
        let lighting = SpriteLighting::default_oracle();
        assert_eq!(
            draw_sprite(&mut target, &cam, &oracle_settings(), &lighting, &s),
            0
        );
    }

    #[test]
    fn kfa_flag_skips_dispatch() {
        let cam = oracle_sprite_front_camera();
        let mut s = Sprite::axis_aligned(cube_kv6(), [1050.0, 1050.0, 175.0]);
        s.flags = SPRITE_FLAG_KFA;
        let (mut fb, mut zb) = alloc_target();
        let mut target = make_target(&mut fb, &mut zb);
        let lighting = SpriteLighting::default_oracle();
        assert_eq!(
            draw_sprite(&mut target, &cam, &oracle_settings(), &lighting, &s),
            0
        );
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
        let synth_sprite = Sprite::axis_aligned(empty_kv6(), [1050.0, 1050.0, 175.0]);
        let state = compute_state_for_test(&setup, &synth_sprite, &cam);

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
        let state = compute_state_for_test(&setup, &sprite, &cam);

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
        let state = compute_state_for_test(&setup, &sprite, &cam);

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
        let state = compute_state_for_test(&setup, &sprite, &cam);
        let (mut fb, mut zb) = alloc_target();
        let mut target = make_target(&mut fb, &mut zb);
        let mut tail = 0u32;
        assert_eq!(
            drawboundcubesse(
                &v,
                0xff,
                &state,
                [0.0, 0.0, 100.0, 100.0],
                &mut tail,
                &mut target,
            ),
            0
        );
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
        let state = compute_state_for_test(&setup, &sprite, &cam);
        // r0.z = -1000 makes origin.z = -1000 + ztab4_per_z[0].z = -1000.
        // scisdist >= 0; -1000 < scisdist → cull.
        let r0 = [0.0, 0.0, -1000.0, -1000.0];
        let (mut fb, mut zb) = alloc_target();
        let mut target = make_target(&mut fb, &mut zb);
        let mut tail = 0u32;
        assert_eq!(
            drawboundcubesse(&v, 0xff, &state, r0, &mut tail, &mut target),
            0
        );
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
    fn draw_sprite_writes_pixels_for_oracle_meltsphere() {
        // R6.4 end-to-end: load the meltsphere fixture, run
        // draw_sprite at the sprite_front pose. Expect a non-zero
        // pixel count and at least one non-zero framebuffer entry.
        let kv = roxlap_formats::kv6::parse(SPRITE_MELTSPHERE_KV6).expect("parse fixture");
        let sprite = Sprite::axis_aligned(kv, [1050.0, 1050.0, 175.0]);
        let cam = oracle_sprite_front_camera();
        let (mut fb, mut zb) = alloc_target();
        let mut target = make_target(&mut fb, &mut zb);
        let lighting = SpriteLighting::default_oracle();
        let written = draw_sprite(&mut target, &cam, &oracle_settings(), &lighting, &sprite);
        assert!(written > 0, "expected some pixels to be written");
        assert!(
            fb.iter().any(|&p| p != 0),
            "expected at least one non-zero framebuffer entry"
        );
        // Z-buffer must have shrunk somewhere from f32::INFINITY.
        assert!(
            zb.iter().any(|&z| z.is_finite()),
            "expected at least one finite zbuffer entry"
        );
    }

    #[test]
    fn draw_sprite_returns_zero_for_culled_sprite() {
        let cam = oracle_sprite_front_camera();
        let s = Sprite::axis_aligned(cube_kv6(), [1020.0 - 500.0, 1050.0, 175.0]);
        let (mut fb, mut zb) = alloc_target();
        let mut target = make_target(&mut fb, &mut zb);
        let lighting = SpriteLighting::default_oracle();
        assert_eq!(
            draw_sprite(&mut target, &cam, &oracle_settings(), &lighting, &s),
            0
        );
        assert!(fb.iter().all(|&p| p == 0));
    }

    /// `update_reflects` for the oracle sprite_front pose hits the
    /// nolighta path (R==G==B kv6col, no fog, lightmode<2). All
    /// kv6colmul[k] entries must repeat one u16 modulation factor
    /// across all 4 lanes.
    #[test]
    fn update_reflects_nolighta_lanes_match() {
        let s = Sprite::axis_aligned(empty_kv6(), [1050.0, 1050.0, 175.0]);
        let lighting = SpriteLighting::default_oracle();
        let (cm, ca) = update_reflects(&s, &lighting);
        assert_eq!(ca, 0, "kv6coladd must be zero (no fog)");
        for (k, e) in cm.iter().enumerate() {
            let l0 = (e & 0xffff) as u16;
            let l1 = ((e >> 16) & 0xffff) as u16;
            let l2 = ((e >> 32) & 0xffff) as u16;
            let l3 = ((e >> 48) & 0xffff) as u16;
            assert_eq!(l0, l1, "kv6colmul[{k}] lane0 != lane1");
            assert_eq!(l0, l2, "kv6colmul[{k}] lane0 != lane2");
            assert_eq!(l0, l3, "kv6colmul[{k}] lane0 != lane3");
        }
    }

    /// Non-grey kv6col forces the nolightb path. Lanes 0..3 of each
    /// `kv6colmul[k]` come from per-channel modulators built from
    /// the kv6col bytes — they should NOT all match unless the
    /// channels themselves match.
    #[test]
    fn update_reflects_nolightb_lanes_diverge_for_tinted_kv6col() {
        let s = Sprite::axis_aligned(empty_kv6(), [1050.0, 1050.0, 175.0]);
        let lighting = SpriteLighting {
            kv6col: 0x0040_8040, // R != G != B
            lightmode: 0,
            lights: &[],
        };
        let (cm, _) = update_reflects(&s, &lighting);
        // Find any direction where the dot is non-zero (most are
        // non-zero); that direction's lanes must vary by channel.
        let mut saw_divergence = false;
        for e in cm.iter() {
            let l0 = (e & 0xffff) as u16;
            let l1 = ((e >> 16) & 0xffff) as u16;
            let l2 = ((e >> 32) & 0xffff) as u16;
            if l0 != l1 || l0 != l2 {
                saw_divergence = true;
                break;
            }
        }
        assert!(
            saw_divergence,
            "non-grey kv6col must produce per-channel divergence in some kv6colmul slot"
        );
    }

    /// Lightmode-2 with one point light + grey kv6col still
    /// produces R==G==B lanes (because the per-channel modulators
    /// are all 0x80<<8 = 0x8000). It must produce a non-uniform
    /// kv6colmul (some directions face the light, others away),
    /// which differs from lightmode<2 where every direction has the
    /// same dot magnitude regardless of position.
    #[test]
    fn update_reflects_lightmode2_produces_directional_shading() {
        let s = Sprite::axis_aligned(empty_kv6(), [100.0, 100.0, 100.0]);
        let lights = [LightSrc {
            pos: [110.0, 100.0, 100.0],
            r2: 100.0,
            sc: 16.0,
        }];
        let lighting = SpriteLighting {
            kv6col: DEFAULT_KV6COL,
            lightmode: 2,
            lights: &lights,
        };
        let (cm, _) = update_reflects(&s, &lighting);
        // Some directions must darken (shadow side) while others
        // brighten (light side) — the spread between min and max
        // tells us shading is happening.
        let mut min_w = u16::MAX;
        let mut max_w = 0u16;
        for e in cm.iter() {
            let l0 = (e & 0xffff) as u16;
            min_w = min_w.min(l0);
            max_w = max_w.max(l0);
        }
        assert!(
            max_w > min_w + 16,
            "lightmode-2 should produce directional shading: min={min_w} max={max_w}"
        );
    }

    /// Lightmode-2 with no lights → ambient-only. Should still
    /// produce some non-zero kv6colmul (the synthetic ambient slot
    /// is non-trivial).
    #[test]
    fn update_reflects_lightmode2_no_lights_falls_back_to_ambient() {
        let s = Sprite::axis_aligned(empty_kv6(), [100.0, 100.0, 100.0]);
        let lighting = SpriteLighting {
            kv6col: DEFAULT_KV6COL,
            lightmode: 2,
            lights: &[],
        };
        let (cm, _) = update_reflects(&s, &lighting);
        let any_nonzero = cm.iter().any(|&e| e != 0);
        assert!(
            any_nonzero,
            "lightmode-2 with no lights should still emit ambient shading"
        );
    }
}
