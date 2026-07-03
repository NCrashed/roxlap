//! Per-grid LOD tier selection — S6.0 of `PORTING-SCENE.md` § S6.
//!
//! S6 introduces three discrete render tiers per grid:
//!
//! - [`Lod::Near`]: full voxel raycast (the existing S1..S5 path).
//! - [`Lod::Mid`]: voxel raycast at the grid's coarser mip level.
//!   Wires through the R4.5 multi-mip infrastructure via
//!   [`crate::Grid::mip_levels_override`] in S6.1.
//! - [`Lod::Far`]: pre-rendered orthographic billboard blit. Lands
//!   in S6.2 (impostor cache) + S6.3 (blit path).
//!
//! S6.0 lands only the **picker infrastructure**: an enum, a
//! threshold pair on every grid, and a `select_lod` helper. The
//! render path computes the LOD per grid each frame but always
//! dispatches the existing `Near` code, so a workspace at S6.0 is
//! byte-identical to one at the end of S5 — assuming the default
//! [`LodThresholds::always_near`] (which it is, courtesy of
//! [`Default`]). Tests pin both the picker's tier dispatch and the
//! framebuffer invariance.
//!
//! Distance metric is **world-space centre-to-centre**:
//! `(camera_pos - grid.transform.origin).length()`. The grid's
//! bounding sphere (radius via [`crate::Grid::bounding_radius`])
//! is *not* subtracted from the metric — thresholds are expressed
//! directly in world distance for predictability. The
//! [`LodThresholds::from_radius`] convenience produces the
//! PORTING-SCENE.md § S6 derived defaults
//! (`r_near = R`, `r_mid = 10 * R`).

use glam::DVec3;

use crate::GridTransform;

/// Discrete LOD tier per [PORTING-SCENE.md] § S6.
///
/// Picker output of [`select_lod`]; consumed by
/// [`crate::render::render_scene_composed`] (S6.1+) to choose
/// between full voxel, low-mip voxel, and billboard impostor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Lod {
    /// Full voxel raycast through the cross-chunk gline path.
    /// Default for every grid pre-S6.1.
    Near,
    /// Voxel raycast at the grid's coarser mip level — reuses the
    /// R4.5 multi-mip infrastructure. Wired in S6.1.
    Mid,
    /// Pre-rendered orthographic billboard blit. The voxel
    /// rasterizer is bypassed entirely. Wired in S6.3.
    Far,
}

/// Per-grid LOD picker configuration: world-distance thresholds
/// for tier dispatch + optional Mid-tier render overrides.
///
/// Tier dispatch (centre-to-centre distance `d`):
/// - `d <= r_near` → [`Lod::Near`]
/// - `r_near < d <= r_mid` → [`Lod::Mid`]
/// - `d > r_mid` → [`Lod::Far`]
///
/// All thresholds default to [`f64::INFINITY`] via [`Default`] /
/// [`Self::always_near`], so a freshly-constructed [`crate::Grid`]
/// always lands on [`Lod::Near`] — the S5-and-earlier byte-stable
/// behaviour. Callers that want real LOD opt in by writing a
/// non-default value into [`crate::Grid::lod_thresholds`].
///
/// `NaN` thresholds are treated as "always [`Lod::Far`]" because
/// every `d <= NaN` comparison is `false`. No assert — callers
/// shouldn't be passing `NaN` and we don't want runtime cost in a
/// per-frame per-grid hot path.
///
/// ## S6.1 — Mid-tier mip overrides
///
/// When the picker returns [`Lod::Mid`], [`Self::mid_mip_levels`]
/// and [`Self::mid_mip_scan_dist`] (if `Some`) override the
/// corresponding [`roxlap_core::opticast::OpticastSettings`] fields
/// for that grid's render. The intent: force coarser-mip rendering
/// at Mid distance to recover performance, using the existing R4.5
/// multi-mip infrastructure with no new rasterizer code.
///
/// Semantics:
/// - `mid_mip_levels = Some(n)` — clamp `OpticastSettings.mip_levels`
///   to `n` for this grid. `n` is then further clamped to
///   `[1, settings.mip_levels]` at the call site.
/// - `mid_mip_scan_dist = Some(d)` — set `OpticastSettings.mip_scan_dist`
///   to `min(settings.mip_scan_dist, d)`. The renderer floors
///   `mip_scan_dist` at 4 internally; smaller values transition to
///   coarser mips closer to the camera, biasing the whole frame
///   toward higher mips.
/// - Both `None` ⇒ Mid path renders identically to Near (graceful
///   degrade — callers can opt into the Mid plumbing without
///   committing to a mip override).
/// - [`crate::Grid::mip_levels_override`] continues to apply on
///   top as a global per-grid cap regardless of tier (the ship
///   anti-beam workaround is preserved at all LOD tiers).
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LodThresholds {
    /// Maximum world-distance at which the grid renders at
    /// [`Lod::Near`]. Grids closer than this are full voxel.
    pub r_near: f64,
    /// Maximum world-distance at which the grid renders at
    /// [`Lod::Mid`]. Beyond `r_mid` the grid uses [`Lod::Far`].
    /// Must satisfy `r_mid >= r_near` for monotonic tier dispatch;
    /// not enforced (an inverted pair just means the [`Lod::Mid`]
    /// band is empty).
    pub r_mid: f64,
    /// S6.1 — `OpticastSettings.mip_levels` override applied only
    /// when the picker returns [`Lod::Mid`]. `None` ⇒ Mid uses the
    /// caller's `settings.mip_levels` unchanged (graceful degrade
    /// to Near-equivalent behaviour). See struct doc for semantics.
    pub mid_mip_levels: Option<u32>,
    /// S6.1 — `OpticastSettings.mip_scan_dist` override applied
    /// only when the picker returns [`Lod::Mid`]. `None` ⇒ Mid uses
    /// the caller's value unchanged. Smaller values bias the grid
    /// toward coarser mips earlier in the ray walk (floor of 4
    /// inside the renderer).
    pub mid_mip_scan_dist: Option<i32>,
}

impl LodThresholds {
    /// Always-`Near` thresholds. Both distance fields set to
    /// [`f64::INFINITY`]; the picker can never enter the Mid/Far
    /// branches. Mid-tier mip overrides set to `None` (irrelevant
    /// since Mid is never selected). Use as the byte-identical
    /// default during the S6.0..S6.3 staged rollout.
    #[must_use]
    pub const fn always_near() -> Self {
        Self {
            r_near: f64::INFINITY,
            r_mid: f64::INFINITY,
            mid_mip_levels: None,
            mid_mip_scan_dist: None,
        }
    }

    /// Derived distance thresholds from the grid's bounding-sphere
    /// radius (PORTING-SCENE.md § S6):
    ///
    /// - `r_near = bounding_radius` — Near while the camera is
    ///   inside the bounding sphere.
    /// - `r_mid = 10 * bounding_radius` — Mid up to ~10× radius,
    ///   Far beyond.
    /// - `mid_mip_levels` / `mid_mip_scan_dist` ⇒ `None` (Mid
    ///   degrades to Near; opt in via
    ///   [`Self::from_radius_with_mid_mip`]).
    ///
    /// A `0.0` (or negative) bounding radius collapses both
    /// thresholds to zero; the picker returns [`Lod::Far`] for any
    /// non-zero distance. That's correct: an empty grid has no
    /// near range.
    #[must_use]
    pub fn from_radius(bounding_radius: f64) -> Self {
        Self {
            r_near: bounding_radius,
            r_mid: 10.0 * bounding_radius,
            mid_mip_levels: None,
            mid_mip_scan_dist: None,
        }
    }

    /// [`Self::from_radius`] + an explicit Mid-tier mip override
    /// pair. Convenience for S6.1 consumers that want Mid LOD wired
    /// without hand-constructing the struct.
    ///
    /// Typical values for a `mip_levels = 4, mip_scan_dist = 128`
    /// world: `mid_mip_levels = 4, mid_mip_scan_dist = 16`. The
    /// reduced scan distance biases the Mid grid into coarser mips
    /// across the whole frame.
    #[must_use]
    pub fn from_radius_with_mid_mip(
        bounding_radius: f64,
        mid_mip_levels: u32,
        mid_mip_scan_dist: i32,
    ) -> Self {
        Self {
            r_near: bounding_radius,
            r_mid: 10.0 * bounding_radius,
            mid_mip_levels: Some(mid_mip_levels),
            mid_mip_scan_dist: Some(mid_mip_scan_dist),
        }
    }
}

impl Default for LodThresholds {
    fn default() -> Self {
        Self::always_near()
    }
}

/// Pick the LOD tier for a grid given the world-space camera
/// position. Distance metric is centre-to-centre Euclidean —
/// `(camera_world_pos - transform.origin).length()`.
///
/// Branchless monotone three-way dispatch on the two thresholds.
/// Called once per grid per frame; cheap.
#[must_use]
pub fn select_lod(
    camera_world_pos: DVec3,
    transform: &GridTransform,
    thresholds: LodThresholds,
) -> Lod {
    let distance = (camera_world_pos - transform.origin).length();
    if distance <= thresholds.r_near {
        Lod::Near
    } else if distance <= thresholds.r_mid {
        Lod::Mid
    } else {
        Lod::Far
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::GridTransform;

    fn at_origin() -> GridTransform {
        GridTransform::at(DVec3::ZERO)
    }

    #[test]
    fn default_thresholds_are_always_near() {
        let t = LodThresholds::default();
        assert_eq!(t.r_near, f64::INFINITY);
        assert_eq!(t.r_mid, f64::INFINITY);
    }

    #[test]
    fn always_near_dispatches_near_at_any_distance() {
        // Even at "very far" distances the default thresholds keep
        // the picker pinned to Near. This is the byte-identity
        // contract for the staged S6 rollout.
        let t = LodThresholds::always_near();
        let xform = at_origin();
        for &d in &[0.0, 100.0, 1_000.0, 1e6, 1e15] {
            assert_eq!(
                select_lod(DVec3::new(d, 0.0, 0.0), &xform, t),
                Lod::Near,
                "expected Near at d={d}"
            );
        }
    }

    #[test]
    fn from_radius_picks_near_inside_mid_band_far_outside() {
        // bounding_radius = 100 → r_near = 100, r_mid = 1000.
        let t = LodThresholds::from_radius(100.0);
        let xform = at_origin();
        let pick = |d: f64| select_lod(DVec3::new(d, 0.0, 0.0), &xform, t);
        // Strictly inside the Near sphere.
        assert_eq!(pick(50.0), Lod::Near);
        // Exactly on the Near boundary — inclusive in Near.
        assert_eq!(pick(100.0), Lod::Near);
        // Just past Near → Mid.
        assert_eq!(pick(100.000_001), Lod::Mid);
        // Inside Mid band.
        assert_eq!(pick(500.0), Lod::Mid);
        // Exactly on Mid boundary — inclusive in Mid.
        assert_eq!(pick(1000.0), Lod::Mid);
        // Past Mid → Far.
        assert_eq!(pick(1_000.000_001), Lod::Far);
        assert_eq!(pick(1e6), Lod::Far);
    }

    #[test]
    fn distance_is_centre_to_centre_in_world_space() {
        // Grid at world (100, 200, 300); camera at (100, 200, 350)
        // is 50 units from the grid origin (z delta only).
        let t = LodThresholds {
            r_near: 49.0,
            r_mid: 51.0,
            ..LodThresholds::always_near()
        };
        let xform = GridTransform::at(DVec3::new(100.0, 200.0, 300.0));
        let cam = DVec3::new(100.0, 200.0, 350.0);
        // Inside the Mid band (49 < 50 < 51).
        assert_eq!(select_lod(cam, &xform, t), Lod::Mid);
    }

    #[test]
    fn rotation_does_not_affect_distance_metric() {
        // The picker keys off `transform.origin` only; rotation is
        // ignored. A non-identity rotation must give the same tier.
        use glam::DQuat;
        let t = LodThresholds::from_radius(10.0);
        let cam = DVec3::new(15.0, 0.0, 0.0);
        let xform_id = GridTransform::identity();
        let xform_rot = GridTransform {
            origin: DVec3::ZERO,
            rotation: DQuat::from_rotation_z(std::f64::consts::FRAC_PI_3),
        };
        assert_eq!(select_lod(cam, &xform_id, t), Lod::Mid);
        assert_eq!(select_lod(cam, &xform_rot, t), Lod::Mid);
    }

    #[test]
    fn zero_radius_collapses_to_far_for_any_nonzero_distance() {
        // An empty grid (bounding_radius = 0) yields
        // r_near = r_mid = 0 — the only `Near` distance is 0.
        let t = LodThresholds::from_radius(0.0);
        let xform = at_origin();
        assert_eq!(select_lod(DVec3::ZERO, &xform, t), Lod::Near);
        assert_eq!(select_lod(DVec3::new(0.5, 0.0, 0.0), &xform, t), Lod::Far);
    }
}
