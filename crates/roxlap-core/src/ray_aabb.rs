//! S1.1 — ray ↔ axis-aligned bounding box clip helper.
//!
//! Standard slab-method intersection. Used by the S1 outside-camera
//! dispatch (`opticast.rs`) to find the t-parameter at which an
//! outside-the-grid camera ray first enters the world's column AABB,
//! so gline can be initialised at that face hit instead of at the
//! camera origin (which sits past `vsid`).
//!
//! Will also be reused by S5 (per-grid rotation: each grid's
//! grid-local ray gets clipped against the grid's local AABB before
//! the cross-chunk DDA starts) and S6 (LOD selection: distance to
//! the AABB is part of the near/mid/far classifier).

/// Intersect a ray `origin + t * dir` against the axis-aligned box
/// `[aabb_min, aabb_max]` and return `Some((t_enter, t_exit))` if
/// the ray hits the box at any t, else `None`.
///
/// Semantics:
/// - `t_enter <= t_exit` always when `Some`.
/// - Origin **inside** the box → `t_enter <= 0.0 <= t_exit`. Caller
///   uses `t_enter <= 0` as the "camera is already inside, take the
///   normal path" signal.
/// - Origin **on a face** → `t_enter == 0` if dir points inward,
///   else `t_exit == 0` (degenerate graze — caller decides).
/// - Ray parallel to a slab while origin is outside that slab →
///   `None`. Ray parallel and origin inside that slab → that axis
///   doesn't constrain; other axes still do.
/// - Ray points entirely away from a hit-able box (`t_exit < 0`) →
///   `None`.
///
/// Robust against `dir == 0` per axis (no NaN from `0.0 / 0.0`).
/// `dir` does not need to be normalised; `t_enter / t_exit` are in
/// the same units as `1.0 / |dir|` so a unit-length dir gives a
/// `t` that's a world-space distance.
#[must_use]
pub fn clip_ray_to_aabb(
    origin: [f32; 3],
    dir: [f32; 3],
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
) -> Option<(f32, f32)> {
    let mut t_enter: f32 = f32::NEG_INFINITY;
    let mut t_exit: f32 = f32::INFINITY;
    for axis in 0..3 {
        let o = origin[axis];
        let d = dir[axis];
        let lo = aabb_min[axis];
        let hi = aabb_max[axis];
        if d == 0.0 {
            if o < lo || o > hi {
                return None;
            }
            continue;
        }
        let inv = 1.0 / d;
        let (t1, t2) = {
            let a = (lo - o) * inv;
            let b = (hi - o) * inv;
            if a <= b {
                (a, b)
            } else {
                (b, a)
            }
        };
        if t1 > t_enter {
            t_enter = t1;
        }
        if t2 < t_exit {
            t_exit = t2;
        }
        if t_enter > t_exit {
            return None;
        }
    }
    if t_exit < 0.0 {
        return None;
    }
    Some((t_enter, t_exit))
}

/// 2D slice of [`clip_ray_to_aabb`] in the XY plane only — what S1.2
/// actually needs at the dispatch level. Voxlap's column grid is
/// indexed by `(cx, cy)`; Z is handled inside each column's slab
/// data, so the per-frame outside check only cares about whether a
/// ray ever enters the XY footprint of the grid.
#[must_use]
pub fn clip_ray_to_xy_aabb(
    origin_xy: [f32; 2],
    dir_xy: [f32; 2],
    aabb_min_xy: [f32; 2],
    aabb_max_xy: [f32; 2],
) -> Option<(f32, f32)> {
    clip_ray_to_aabb(
        [origin_xy[0], origin_xy[1], 0.0],
        [dir_xy[0], dir_xy[1], 0.0],
        [aabb_min_xy[0], aabb_min_xy[1], -1.0],
        [aabb_max_xy[0], aabb_max_xy[1], 1.0],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const UNIT_BOX_MIN: [f32; 3] = [0.0, 0.0, 0.0];
    const UNIT_BOX_MAX: [f32; 3] = [1.0, 1.0, 1.0];

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn direct_hit_through_x_face() {
        let r = clip_ray_to_aabb(
            [-5.0, 0.5, 0.5],
            [1.0, 0.0, 0.0],
            UNIT_BOX_MIN,
            UNIT_BOX_MAX,
        )
        .expect("expected hit");
        assert!(approx_eq(r.0, 5.0), "t_enter={}", r.0);
        assert!(approx_eq(r.1, 6.0), "t_exit={}", r.1);
    }

    #[test]
    fn parallel_outside_misses() {
        let r = clip_ray_to_aabb(
            [-5.0, 5.0, 0.5],
            [1.0, 0.0, 0.0],
            UNIT_BOX_MIN,
            UNIT_BOX_MAX,
        );
        assert!(r.is_none());
    }

    #[test]
    fn origin_inside_yields_negative_t_enter() {
        let r = clip_ray_to_aabb([0.5, 0.5, 0.5], [1.0, 0.0, 0.0], UNIT_BOX_MIN, UNIT_BOX_MAX)
            .expect("expected hit");
        assert!(approx_eq(r.0, -0.5), "t_enter={}", r.0);
        assert!(approx_eq(r.1, 0.5), "t_exit={}", r.1);
        assert!(r.0 <= 0.0);
    }

    #[test]
    fn parallel_inside_does_not_constrain() {
        let r = clip_ray_to_aabb([0.5, 0.5, 0.5], [0.0, 0.0, 1.0], UNIT_BOX_MIN, UNIT_BOX_MAX)
            .expect("expected hit");
        assert!(approx_eq(r.0, -0.5));
        assert!(approx_eq(r.1, 0.5));
    }

    #[test]
    fn ray_pointed_away_from_box_misses() {
        let r = clip_ray_to_aabb(
            [-1.0, 0.5, 0.5],
            [-1.0, 0.0, 0.0],
            UNIT_BOX_MIN,
            UNIT_BOX_MAX,
        );
        assert!(r.is_none(), "got {r:?}");
    }

    #[test]
    fn corner_grazing_hit_is_a_hit() {
        let r = clip_ray_to_aabb(
            [-1.0, -1.0, 0.5],
            [1.0, 1.0, 0.0],
            UNIT_BOX_MIN,
            UNIT_BOX_MAX,
        )
        .expect("corner graze should still be a hit");
        assert!(approx_eq(r.0, 1.0));
        assert!(approx_eq(r.1, 2.0));
    }

    #[test]
    fn corner_grazing_miss_by_epsilon() {
        let r = clip_ray_to_aabb(
            [-1.0, -1.001, 0.5],
            [1.0, 1.0, 0.0],
            UNIT_BOX_MIN,
            UNIT_BOX_MAX,
        );
        // Still a hit (very thin sliver at y=0). The "miss by
        // epsilon" property only fires when the slabs reverse —
        // confirm that doesn't happen here.
        assert!(r.is_some());
    }

    #[test]
    fn entry_face_origin_returns_zero_t_enter() {
        let r = clip_ray_to_aabb([0.0, 0.5, 0.5], [1.0, 0.0, 0.0], UNIT_BOX_MIN, UNIT_BOX_MAX)
            .expect("expected hit");
        assert!(approx_eq(r.0, 0.0));
        assert!(approx_eq(r.1, 1.0));
    }

    #[test]
    fn exit_face_origin_zero_length_interval() {
        let r = clip_ray_to_aabb([1.0, 0.5, 0.5], [1.0, 0.0, 0.0], UNIT_BOX_MIN, UNIT_BOX_MAX)
            .expect("graze of exit face is technically a hit");
        assert!(approx_eq(r.0, -1.0));
        assert!(approx_eq(r.1, 0.0));
    }

    #[test]
    fn xy_helper_matches_3d_for_xy_ray() {
        let three_d = clip_ray_to_aabb(
            [-3.0, 0.5, 0.5],
            [1.0, 0.5, 0.0],
            [0.0, 0.0, -1.0],
            [10.0, 10.0, 1.0],
        )
        .expect("3D hit");
        let two_d =
            clip_ray_to_xy_aabb([-3.0, 0.5], [1.0, 0.5], [0.0, 0.0], [10.0, 10.0]).expect("2D hit");
        assert!(approx_eq(three_d.0, two_d.0));
        assert!(approx_eq(three_d.1, two_d.1));
    }

    #[test]
    fn outside_orbit_camera_into_world() {
        // S1's actual use case: camera at (vsid + 256, vsid/2)
        // looking back along -X at the world AABB [0, vsid]^2.
        let vsid = 2048.0;
        let origin_xy = [vsid + 256.0, vsid * 0.5];
        let dir_xy = [-1.0, 0.0];
        let r = clip_ray_to_xy_aabb(origin_xy, dir_xy, [0.0, 0.0], [vsid, vsid])
            .expect("outside_orbit center ray must hit world AABB");
        assert!(approx_eq(r.0, 256.0), "t_enter={}", r.0);
        assert!(approx_eq(r.1, 256.0 + vsid), "t_exit={}", r.1);
    }

    #[test]
    fn nan_dir_zero_origin_outside_misses() {
        // dir=0 on x with origin past the +x face → miss without
        // generating any NaNs.
        let r = clip_ray_to_aabb(
            [10.0, 0.5, 0.5],
            [0.0, 1.0, 0.0],
            UNIT_BOX_MIN,
            UNIT_BOX_MAX,
        );
        assert!(r.is_none());
    }
}
