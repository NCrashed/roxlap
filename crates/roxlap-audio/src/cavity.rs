//! AU.1 — cavity estimation: reverb parameters from the space around
//! the listener.
//!
//! A fixed golden-spiral fan of rays probes the free path in every
//! direction ([`probe_cavity`], pure and deterministic); the mean free
//! path reads as room size and the fraction of rays escaping the probe
//! radius as openness. [`CavityEstimator`] wraps the probe with the
//! heavy `(3·old + new) / 4`-style smoothing the prior art ships
//! (Sound Filters, Teardown): the environment estimate converges over
//! seconds and players read the lag as natural room acoustics, not as
//! a parameter glitch.
//!
//! Cadence: the host calls [`CavityEstimator::update`] at a low rate
//! (~2 Hz per the entry doc) — NOT per frame. Smoothing strength is
//! per *update*, so a faster cadence converges faster.

use glam::DVec3;
use roxlap_scene::Scene;

/// Tuning knobs for the cavity probe + the reverb mapping.
#[derive(Debug, Clone, PartialEq)]
pub struct CavityConfig {
    /// Rays in the golden-spiral fan. Default `32` (the Sound Physics
    /// bounce budget). Clamped to at least 4.
    pub rays: u32,
    /// Probe radius, world/voxel units: rays that reach it without a
    /// hit count as *escaped* (open sky / huge space). Also the free
    /// path a miss contributes to the mean. Deliberately SEPARATE from
    /// the occlusion budget (`AcousticsConfig::max_distance`, default
    /// 128): occlusion reaches as far as sources are audible, the
    /// reverb probe only needs the local room.
    pub max_ray_dist: f64,
    /// Reverb feedback (decay) in the smallest enclosed space.
    pub min_feedback: f32,
    /// Reverb feedback when the **enclosed** free path (walls actually
    /// hit — escaped rays don't inflate room size) reaches
    /// [`max_ray_dist`](Self::max_ray_dist).
    pub max_feedback: f32,
    /// Wet mix in a fully enclosed space.
    pub max_wet: f32,
    /// Openness at (and beyond) which the wet mix reaches zero. A
    /// listener standing on open ground always has ~half the fan in
    /// the ground, so raw openness tops out near `0.5` outdoors — this
    /// threshold makes that read as *fully dry* instead of half-wet.
    /// Default `0.5`; a sealed cavern (openness `0`) is unaffected.
    pub outdoor_openness: f32,
    /// Reverb damping (high-frequency loss), passed through to the
    /// backend unmodified. Per-material damping is future work.
    pub damping: f32,
    /// Fraction of the OLD smoothed value kept per
    /// [`CavityEstimator::update`] — `0.75` is the prior art's
    /// `(3·old + new) / 4`. `0` disables smoothing.
    pub smoothing: f32,
}

impl Default for CavityConfig {
    fn default() -> Self {
        Self {
            rays: 32,
            max_ray_dist: 64.0,
            min_feedback: 0.35,
            max_feedback: 0.88,
            max_wet: 0.5,
            outdoor_openness: 0.5,
            damping: 0.4,
            smoothing: 0.75,
        }
    }
}

/// Raw (unsmoothed) result of one [`probe_cavity`] fan.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CavityProbe {
    /// Mean free path across the WHOLE fan, world/voxel units — escaped
    /// rays contribute [`CavityConfig::max_ray_dist`]. Diagnostic; the
    /// reverb mapping does not use it (an open sky would masquerade as
    /// a huge room).
    pub mean_free_path: f32,
    /// Mean free path across the rays that actually HIT a wall — the
    /// room-size signal driving reverb feedback. Escaped rays don't
    /// inflate it: on open ground only the short ground hits count, so
    /// the "room" reads tiny. [`CavityConfig::max_ray_dist`] when no
    /// ray hit anything (unbounded space; the wet mix is zero there
    /// anyway).
    pub enclosed_free_path: f32,
    /// Fraction of rays that escaped the probe radius, `0..=1` —
    /// `0` = sealed cavern, `~0.5` = standing on open ground.
    pub openness: f32,
}

/// Smoothed reverb parameters for the listener's surroundings,
/// produced by [`CavityEstimator::update`]. Feed the reverb fields to
/// the backend with long tweens (~1 s).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ListenerAcoustics {
    /// Smoothed mean free path (room size), world/voxel units.
    pub mean_free_path: f32,
    /// Smoothed openness, `0..=1`.
    pub openness: f32,
    /// Reverb feedback (decay): bigger enclosed space ⇒ longer tail.
    pub reverb_feedback: f32,
    /// Reverb damping, straight from [`CavityConfig::damping`].
    pub reverb_damping: f32,
    /// Reverb wet mix, `0..=1`: scaled down by openness so outdoor
    /// sound stays dry.
    pub reverb_mix: f32,
}

/// Probe the space around `listener`: cast the golden-spiral fan and
/// reduce it to a [`CavityProbe`]. Pure and deterministic — the fan is
/// a fixed direction set, so tests pin exact values and the estimator
/// can round-robin freely.
///
/// Two caveats:
/// - A listener **inside solid** (camera clipped into rock) sees every
///   ray hit at `t ≈ 0` and reads as a tiny sealed box (max wet, min
///   feedback). Hosts whose listener can clip should guard or nudge
///   the probe point out of solid first.
/// - Rays go through `Scene::raycast`'s own march, NOT the
///   chunk-borrow-cached thickness march `path_thickness` uses — when
///   profiling, the fan's cost profile is `raycast`'s.
#[must_use]
pub fn probe_cavity(scene: &Scene, listener: DVec3, cfg: &CavityConfig) -> CavityProbe {
    let rays = cfg.rays.max(4);
    let mut path_sum = 0.0f64;
    let mut hit_sum = 0.0f64;
    let mut escaped = 0u32;
    for k in 0..rays {
        let dir = golden_spiral_dir(k, rays);
        match scene.raycast(listener, dir, cfg.max_ray_dist) {
            Some(hit) => {
                path_sum += hit.t;
                hit_sum += hit.t;
            }
            None => {
                path_sum += cfg.max_ray_dist;
                escaped += 1;
            }
        }
    }
    let hits = rays - escaped;
    #[allow(clippy::cast_possible_truncation)]
    CavityProbe {
        mean_free_path: (path_sum / f64::from(rays)) as f32,
        enclosed_free_path: if hits == 0 {
            cfg.max_ray_dist as f32
        } else {
            (hit_sum / f64::from(hits)) as f32
        },
        openness: escaped as f32 / rays as f32,
    }
}

/// Direction `k` of an `n`-point golden-spiral sphere covering — the
/// standard even covering of the unit sphere (unit length by
/// construction).
fn golden_spiral_dir(k: u32, n: u32) -> DVec3 {
    const GOLDEN_ANGLE: f64 = 2.399_963_229_728_653; // π(3 − √5)
    let z = 1.0 - 2.0 * (f64::from(k) + 0.5) / f64::from(n);
    let r = (1.0 - z * z).sqrt();
    let phi = GOLDEN_ANGLE * f64::from(k);
    DVec3::new(r * phi.cos(), r * phi.sin(), z)
}

/// Stateful cavity estimator: [`probe_cavity`] plus the prior-art
/// exponential smoothing, mapped to reverb parameters. One per
/// listener; call [`update`](Self::update) at ~2 Hz and
/// [`reset`](Self::reset) on teleports.
#[derive(Debug, Clone)]
pub struct CavityEstimator {
    cfg: CavityConfig,
    smoothed: Option<CavityProbe>,
}

impl CavityEstimator {
    /// A fresh estimator; the first [`update`](Self::update) seeds the
    /// smoothed state from the raw probe (no fade-in from a wrong
    /// initial guess).
    #[must_use]
    pub fn new(cfg: CavityConfig) -> Self {
        Self {
            cfg,
            smoothed: None,
        }
    }

    /// The configuration this estimator was built with.
    #[must_use]
    pub fn config(&self) -> &CavityConfig {
        &self.cfg
    }

    /// Drop the smoothed state — the next [`update`](Self::update)
    /// re-seeds from the raw probe. Call on listener teleports so a
    /// cavern's tail doesn't follow the player through a menu warp.
    pub fn reset(&mut self) {
        self.smoothed = None;
    }

    /// Probe, smooth, and map to reverb parameters.
    pub fn update(&mut self, scene: &Scene, listener: DVec3) -> ListenerAcoustics {
        let raw = probe_cavity(scene, listener, &self.cfg);
        let s = self.cfg.smoothing.clamp(0.0, 1.0);
        let sm = match self.smoothed {
            None => raw,
            Some(prev) => CavityProbe {
                mean_free_path: prev.mean_free_path * s + raw.mean_free_path * (1.0 - s),
                enclosed_free_path: prev.enclosed_free_path * s
                    + raw.enclosed_free_path * (1.0 - s),
                openness: prev.openness * s + raw.openness * (1.0 - s),
            },
        };
        self.smoothed = Some(sm);
        self.params(sm)
    }

    /// Map a (smoothed) probe to reverb parameters — split out so the
    /// mapping itself is testable without a scene.
    ///
    /// Room size comes from the **enclosed** free path only (escaped
    /// rays must not read open sky as a huge hall), and the wet mix
    /// hits zero at [`CavityConfig::outdoor_openness`] — a listener on
    /// open ground (raw openness ~0.5, the ground owns the lower
    /// hemisphere) is fully dry, not half-wet.
    #[allow(clippy::cast_possible_truncation)]
    fn params(&self, p: CavityProbe) -> ListenerAcoustics {
        let size_norm =
            (f64::from(p.enclosed_free_path) / self.cfg.max_ray_dist).clamp(0.0, 1.0) as f32;
        let outdoor = self.cfg.outdoor_openness.max(1e-6);
        let enclosure = ((outdoor - p.openness) / outdoor).clamp(0.0, 1.0);
        ListenerAcoustics {
            mean_free_path: p.mean_free_path,
            openness: p.openness,
            reverb_feedback: self.cfg.min_feedback
                + (self.cfg.max_feedback - self.cfg.min_feedback) * size_norm,
            reverb_damping: self.cfg.damping,
            reverb_mix: self.cfg.max_wet * enclosure,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::IVec3;
    use roxlap_scene::{GridTransform, VoxColor};

    const STONE: VoxColor = VoxColor(0x80_66_77_88);

    /// A hollow cubic room of inner side `side`, centred at `c` (cell
    /// coords), inside one identity grid; returns the scene + the room
    /// centre as a world point.
    fn room(side: i32, c: IVec3) -> (Scene, DVec3) {
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::identity());
        let g = scene.grid_mut(id).expect("grid just added");
        let h = side / 2;
        let shell = 2;
        g.set_rect(
            c - IVec3::splat(h + shell),
            c + IVec3::splat(h + shell),
            Some(STONE),
        );
        g.set_rect(c - IVec3::splat(h), c + IVec3::splat(h), None);
        let centre = DVec3::new(
            f64::from(c.x) + 0.5,
            f64::from(c.y) + 0.5,
            f64::from(c.z) + 0.5,
        );
        (scene, centre)
    }

    #[test]
    fn empty_scene_is_fully_open() {
        let scene = Scene::new();
        let cfg = CavityConfig::default();
        let p = probe_cavity(&scene, DVec3::new(64.0, 64.0, 128.0), &cfg);
        assert_eq!(p.openness, 1.0);
        #[allow(clippy::cast_possible_truncation)]
        let cap = cfg.max_ray_dist as f32;
        assert!((p.mean_free_path - cap).abs() < 1e-4);
        assert!((p.enclosed_free_path - cap).abs() < 1e-4, "no-hit sentinel");
        // Mapping: unbounded space is bone dry regardless of feedback.
        let a = CavityEstimator::new(cfg).params(p);
        assert_eq!(a.reverb_mix, 0.0);
    }

    #[test]
    fn bigger_room_reads_bigger_and_both_are_sealed() {
        let cfg = CavityConfig::default();
        let (small, sc) = room(8, IVec3::new(64, 64, 128));
        let (big, bc) = room(40, IVec3::new(64, 64, 128));
        let ps = probe_cavity(&small, sc, &cfg);
        let pb = probe_cavity(&big, bc, &cfg);
        assert_eq!(ps.openness, 0.0, "small room is sealed");
        assert_eq!(pb.openness, 0.0, "big room is sealed");
        assert!(
            pb.mean_free_path > ps.mean_free_path * 2.0,
            "room size ordering: big {} vs small {}",
            pb.mean_free_path,
            ps.mean_free_path
        );

        // Mapping: bigger sealed room ⇒ longer decay; both fully wet.
        let est = CavityEstimator::new(cfg.clone());
        let a_small = est.params(ps);
        let a_big = est.params(pb);
        assert!(a_big.reverb_feedback > a_small.reverb_feedback);
        assert!((a_small.reverb_mix - cfg.max_wet).abs() < 1e-6);
        assert!(a_small.reverb_feedback >= cfg.min_feedback);
        assert!(a_big.reverb_feedback <= cfg.max_feedback);
    }

    #[test]
    fn open_ground_is_much_drier_than_a_cavern() {
        let cfg = CavityConfig::default();
        // Flat ground: solid below z = 200, listener standing on it.
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::identity());
        let g = scene.grid_mut(id).expect("grid");
        g.set_rect(
            IVec3::new(0, 0, 200),
            IVec3::new(255, 255, 220),
            Some(STONE),
        );
        let outdoors = probe_cavity(&scene, DVec3::new(128.0, 128.0, 198.0), &cfg);
        assert!(
            outdoors.openness > 0.4,
            "standing on open ground, roughly the upper hemisphere escapes: {}",
            outdoors.openness
        );
        // The room-size signal must NOT be inflated by the sky: only
        // the short ground hits count.
        assert!(
            outdoors.enclosed_free_path < outdoors.mean_free_path * 0.6,
            "enclosed path ({}) must ignore escaped rays (mean {})",
            outdoors.enclosed_free_path,
            outdoors.mean_free_path
        );

        let (cavern, cc) = room(40, IVec3::new(64, 64, 128));
        let inside = probe_cavity(&cavern, cc, &cfg);
        let est = CavityEstimator::new(cfg.clone());
        let dry = est.params(outdoors);
        let wet = est.params(inside);
        // ABSOLUTE bounds — a relative comparison would let "audibly
        // reverberant outdoors" hide behind an even wetter cavern.
        assert!(
            dry.reverb_mix < 0.05,
            "open ground must be bone dry, got mix {}",
            dry.reverb_mix
        );
        assert!(
            dry.reverb_feedback < cfg.min_feedback + 0.15,
            "open ground must not read as a big hall: feedback {}",
            dry.reverb_feedback
        );
        assert!(
            wet.reverb_mix > cfg.max_wet * 0.8,
            "sealed cavern stays wet: {}",
            wet.reverb_mix
        );
    }

    #[test]
    fn smoothing_seeds_raw_then_converges_monotonically() {
        let cfg = CavityConfig::default();
        let (small, sc) = room(8, IVec3::new(64, 64, 128));
        let (big, bc) = room(40, IVec3::new(64, 64, 128));

        let mut est = CavityEstimator::new(cfg.clone());
        // First update seeds from raw — no fade-in from zero.
        let first = est.update(&big, bc);
        let raw_big = probe_cavity(&big, bc, &cfg);
        assert!((first.mean_free_path - raw_big.mean_free_path).abs() < 1e-6);

        // "Teleport" into the small room WITHOUT reset: the estimate
        // must walk down monotonically toward the small-room value.
        let raw_small = probe_cavity(&small, sc, &cfg);
        let mut prev = first.mean_free_path;
        for step in 0..12 {
            let a = est.update(&small, sc);
            assert!(
                a.mean_free_path < prev,
                "step {step}: estimate must decrease ({} vs {prev})",
                a.mean_free_path
            );
            prev = a.mean_free_path;
        }
        assert!(
            (prev - raw_small.mean_free_path).abs() < 2.0,
            "after 12 updates the estimate ({prev}) nears the raw small-room value ({})",
            raw_small.mean_free_path
        );

        // reset() re-seeds instantly.
        est.reset();
        let after = est.update(&big, bc);
        assert!((after.mean_free_path - raw_big.mean_free_path).abs() < 1e-6);
    }

    #[test]
    fn deterministic() {
        let cfg = CavityConfig::default();
        let (scene, c) = room(16, IVec3::new(64, 64, 128));
        assert_eq!(probe_cavity(&scene, c, &cfg), probe_cavity(&scene, c, &cfg));
    }
}
