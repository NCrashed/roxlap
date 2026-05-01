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

use roxlap_formats::kv6::Kv6;

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

/// Draw a sprite into the engine's framebuffer + z-buffer.
///
/// Top-level dispatcher mirroring voxlap5.c:9818-9828:
/// - Skips on `flags & INVISIBLE`.
/// - Picks `kv6draw` vs `kv6draw_noz` based on `flags & NO_Z`.
/// - Picks the kv6 vs kfa path based on `flags & KFA`.
///
/// **R6.2 status**: dispatcher runs the mip-LOD pick + 4-plane
/// frustum cull and returns `false` either way (no pixels written
/// yet — that's R6.4). Callers can rely on the same `false`
/// behaviour for both "culled" and "would render".
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
    let Some(_setup) = kv6_draw_prepare(sprite, cam) else {
        return false;
    };
    // R6.3+ will plug the per-voxel iteration in here. For R6.2
    // cull is the deliverable.
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
    fn r62_dispatcher_returns_false_for_in_frustum_sprite() {
        // R6.2 stub: even a sprite that passes the cull doesn't
        // render anything yet. R6.4 will flip this to true.
        let cam = oracle_sprite_front_camera();
        let s = Sprite::axis_aligned(cube_kv6(), [1050.0, 1050.0, 175.0]);
        assert!(!draw_sprite(&cam, &s));
    }
}
