//! Voxel-aware acoustics for the roxlap engine (stage AU —
//! `docs/porting/PORTING-AUDIO.md`).
//!
//! This crate computes **acoustic parameters** from the voxel world; it
//! deliberately owns no audio device and spawns no audio thread. The
//! host feeds the numbers into whatever playback stack it uses (the
//! optional `kira` backend lands in AU.2): per sound source an
//! occlusion-driven gain + lowpass cutoff + reverb send, and (AU.1) per
//! listener a cavity-driven reverb feedback/mix.
//!
//! The occlusion model follows the convergent games recipe (Sound
//! Physics, Teardown, Overwatch — see the entry doc): a small fan of
//! rays from the source to the listener, each accumulating the **solid
//! path length** it crosses (voxel units, exact segment lengths from a
//! DDA march — not a boolean hit test), averaged into a `0..=1`
//! *transmission*. Thickness matters: one voxel of wall muffles less
//! than five, and a doorway passes some rays untouched, so sound leaks
//! around corners instead of gating on/off.
//!
//! Everything here is pure and deterministic — the jitter fan is a
//! fixed ring, not RNG — so tests pin exact values and hosts can call
//! it from any thread.

use glam::{DVec3, IVec3};
use roxlap_scene::Scene;

mod backend;
mod cavity;
pub mod synth;

#[cfg(feature = "kira")]
mod kira_out;

pub use backend::{AudioOut, SoundKey, SourceId, SourcePool};
pub use cavity::{probe_cavity, CavityConfig, CavityEstimator, CavityProbe, ListenerAcoustics};
pub use synth::SoundBuffer;

#[cfg(feature = "kira")]
pub use kira_out::KiraAudio;

/// Tuning knobs for [`source_acoustics`]. The defaults are the cave-demo
/// tuning; every field is plain data so hosts can persist or lerp them.
#[derive(Debug, Clone, PartialEq)]
pub struct AcousticsConfig {
    /// Jittered rays per source→listener query, **including** the direct
    /// ray. `1` = direct only; the default `9` (direct + an 8-point ring)
    /// matches the Sound Physics budget and gives soft doorway
    /// transitions. Clamped to at least 1.
    pub rays: u32,
    /// Radius (world/voxel units) of the jitter ring around the source,
    /// perpendicular to the source→listener axis. Roughly "how big is
    /// the emitter": bigger radii leak more around thin edges.
    pub jitter_radius: f64,
    /// Per-voxel absorption: a ray crossing `t` voxels of solid keeps
    /// `exp(-absorption · t)` of its energy. `0.25` ⇒ ~78% through a
    /// 1-voxel wall, ~29% through 5.
    pub absorption_per_voxel: f64,
    /// Lowpass cutoff at full transmission (unoccluded), Hz.
    pub open_cutoff_hz: f32,
    /// Lowpass cutoff at zero transmission (fully buried), Hz.
    pub occluded_cutoff_hz: f32,
    /// Gain applied at zero transmission, decibels (negative). Scales
    /// linearly with `1 - transmission`; distance attenuation is NOT
    /// included (that is the spatial backend's job).
    pub max_occlusion_db: f32,
    /// Fraction of the occlusion gain also applied to the reverb send —
    /// a buried source still excites the room, just less. `0.5` halves
    /// the dB loss on the send relative to the dry path.
    pub send_occlusion_frac: f32,
    /// Hard cap on the marched source→listener distance (world units);
    /// beyond it no rays are cast and the source reports
    /// [`SourceAcoustics::clear`] — distance attenuation is the spatial
    /// backend's job, and a far-away source in open air must NOT come
    /// back muffled. Bounds the DDA cost.
    pub max_distance: f64,
}

impl Default for AcousticsConfig {
    fn default() -> Self {
        Self {
            rays: 9,
            jitter_radius: 1.5,
            absorption_per_voxel: 0.25,
            open_cutoff_hz: 20_000.0,
            occluded_cutoff_hz: 800.0,
            max_occlusion_db: -24.0,
            send_occlusion_frac: 0.5,
            max_distance: 128.0,
        }
    }
}

/// Occlusion-driven playback parameters for one source→listener pair,
/// produced by [`source_acoustics`]. Feed them to the playback layer
/// with short tweens (~120 ms) — never as raw jumps.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SourceAcoustics {
    /// Average transmitted energy across the ray fan, `0..=1`
    /// (`1` = clear line of sound, `0` = buried).
    pub transmission: f32,
    /// Occlusion gain, decibels (`0` when clear, down to
    /// [`AcousticsConfig::max_occlusion_db`]). Excludes distance
    /// attenuation.
    pub gain_db: f32,
    /// Lowpass cutoff for the muffling filter, Hz (log-interpolated
    /// between the config's occluded and open cutoffs).
    pub lowpass_cutoff_hz: f32,
    /// Gain for this source's reverb send, decibels.
    pub reverb_send_db: f32,
}

impl SourceAcoustics {
    /// The parameters of a fully unoccluded source under `cfg`.
    #[must_use]
    pub fn clear(cfg: &AcousticsConfig) -> Self {
        Self {
            transmission: 1.0,
            gain_db: 0.0,
            lowpass_cutoff_hz: cfg.open_cutoff_hz,
            reverb_send_db: 0.0,
        }
    }
}

/// Total **solid path length** (world/voxel units) the straight segment
/// `a → b` crosses, summed over every grid in the scene. This is the
/// thickness building block under [`source_acoustics`]: exact per-cell
/// segment lengths from a voxel DDA against [`roxlap_scene::Grid::voxel_solid`]
/// (a diagonal crossing of a 1-voxel wall correctly reports ~√2).
///
/// Grid transforms are honoured the same way `Scene::raycast` does it:
/// the segment is rebased into each grid's local frame (rotation is an
/// isometry, so lengths stay world-true). Cost is O(segment length in
/// voxels) per grid — audio queries are short and infrequent, so no
/// chunk-level skip is attempted (noted future optimisation).
#[must_use]
pub fn path_thickness(scene: &Scene, a: DVec3, b: DVec3) -> f64 {
    let mut total = 0.0;
    for (_, grid) in scene.grids() {
        let inv = grid.transform.rotation.inverse();
        let la = inv * (a - grid.transform.origin);
        let lb = inv * (b - grid.transform.origin);
        total += grid_thickness(grid, la, lb);
    }
    total
}

/// Solid path length of the grid-local segment `a → b` through one
/// grid: an Amanatides–Woo cell walk accumulating the in-cell segment
/// length wherever the voxel is solid. Consecutive steps share a
/// one-entry chunk borrow (one HashMap probe per crossed chunk, not
/// per voxel — the [`roxlap_scene::Grid::chunk_voxel_solid`] pattern).
///
/// Caveat: voxlap's bedrock placeholder plane (chunk-local z = 255 of
/// otherwise-empty chunks) reads as solid, so a segment grazing the
/// world bottom accumulates it as real thickness. Keep acoustic
/// endpoints above the bedrock plane (every demo does).
fn grid_thickness(grid: &roxlap_scene::Grid, a: DVec3, b: DVec3) -> f64 {
    let seg = b - a;
    let len = seg.length();
    if len < 1e-9 {
        return 0.0;
    }
    let dir = seg / len;

    #[allow(clippy::cast_possible_truncation)]
    let mut cell = IVec3::new(a.x.floor() as i32, a.y.floor() as i32, a.z.floor() as i32);

    // Per-axis DDA state: t at the next cell boundary and t per cell.
    let mut t_max = [f64::INFINITY; 3];
    let mut t_delta = [f64::INFINITY; 3];
    let mut step = [0i32; 3];
    for ax in 0..3 {
        let d = dir[ax];
        if d.abs() < 1e-12 {
            continue;
        }
        step[ax] = if d > 0.0 { 1 } else { -1 };
        t_delta[ax] = 1.0 / d.abs();
        let cell_f = f64::from(cell[ax]);
        let next_boundary = if d > 0.0 { cell_f + 1.0 } else { cell_f };
        t_max[ax] = (next_boundary - a[ax]) / d;
    }

    let mut thickness = 0.0;
    let mut t_prev = 0.0;
    // One-entry chunk cache: consecutive DDA cells overwhelmingly stay
    // in the same 128×128×256 chunk, so the HashMap probe amortises to
    // one per crossed chunk.
    let mut cached_idx = IVec3::MAX;
    // Inferred `Option<&Vxl>` (roxlap-scene doesn't re-export the type).
    let mut cached_chunk = None;
    // Step budget: a segment of length L crosses at most ~L+1 boundaries
    // per axis. The +8 absorbs boundary-epsilon jitter.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let max_steps = (len * 3.0) as u32 + 8;
    for _ in 0..max_steps {
        let ax = min_axis(&t_max);
        let t_next = t_max[ax].min(len);
        let (chunk_idx, in_chunk) = roxlap_scene::voxel_split(cell);
        if chunk_idx != cached_idx {
            cached_idx = chunk_idx;
            cached_chunk = grid.chunk(chunk_idx);
        }
        let solid =
            cached_chunk.is_some_and(|vxl| roxlap_scene::Grid::chunk_voxel_solid(vxl, in_chunk));
        if solid {
            thickness += t_next - t_prev;
        }
        if t_max[ax] >= len {
            break;
        }
        t_prev = t_next;
        cell[ax] += step[ax];
        t_max[ax] += t_delta[ax];
    }
    thickness
}

/// Index of the smallest component (the axis the DDA crosses next).
fn min_axis(t: &[f64; 3]) -> usize {
    if t[0] <= t[1] && t[0] <= t[2] {
        0
    } else if t[1] <= t[2] {
        1
    } else {
        2
    }
}

/// Occlusion parameters for one source heard from `listener`: a fan of
/// [`AcousticsConfig::rays`] thickness rays (the direct segment plus a
/// fixed jitter ring around the **source**, perpendicular to the
/// source→listener axis), each converted to transmitted energy
/// `exp(-absorption · thickness)` and averaged.
///
/// Deterministic — the ring is fixed 45° spokes, no RNG — and pure, so
/// hosts may round-robin sources across ticks freely (the entry doc's
/// ~4 Hz per source cadence).
#[must_use]
pub fn source_acoustics(
    scene: &Scene,
    source: DVec3,
    listener: DVec3,
    cfg: &AcousticsConfig,
) -> SourceAcoustics {
    let seg = listener - source;
    let dist = seg.length();
    // Beyond the march budget (or on top of the listener): report
    // CLEAR, not buried — the backend's distance attenuation owns far
    // sources, and clamping to "muffled" would put a step function at
    // the boundary (a far source in open air is quiet, not lowpassed).
    if dist > cfg.max_distance || dist < 1e-9 {
        return SourceAcoustics::clear(cfg);
    }
    let dir = seg / dist;

    // A stable perpendicular basis for the jitter ring.
    let helper = if dir.x.abs() < 0.9 {
        DVec3::X
    } else {
        DVec3::Y
    };
    let u = dir.cross(helper).normalize();
    let v = dir.cross(u);

    let rays = cfg.rays.max(1);
    let mut energy = 0.0f64;
    for i in 0..rays {
        let origin = if i == 0 {
            source
        } else {
            // Fixed ring: spoke k of (rays - 1) around the source.
            let ang = std::f64::consts::TAU * f64::from(i - 1) / f64::from(rays - 1);
            source + (u * ang.cos() + v * ang.sin()) * cfg.jitter_radius
        };
        let t = path_thickness(scene, origin, listener);
        energy += (-cfg.absorption_per_voxel * t).exp();
    }
    #[allow(clippy::cast_possible_truncation)]
    let transmission = (energy / f64::from(rays)).clamp(0.0, 1.0) as f32;

    // Log-space cutoff interpolation: T = 1 ⇒ open, T = 0 ⇒ occluded.
    let ratio = cfg.open_cutoff_hz / cfg.occluded_cutoff_hz;
    let lowpass_cutoff_hz = cfg.occluded_cutoff_hz * ratio.powf(transmission);
    let gain_db = cfg.max_occlusion_db * (1.0 - transmission);
    SourceAcoustics {
        transmission,
        gain_db,
        lowpass_cutoff_hz,
        reverb_send_db: gain_db * cfg.send_occlusion_frac,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use roxlap_scene::{GridTransform, VoxColor};

    const STONE: VoxColor = VoxColor(0x80_66_77_88);

    /// A scene with one identity grid; `build` shapes it.
    fn scene_with(build: impl FnOnce(&mut roxlap_scene::Grid)) -> Scene {
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::identity());
        build(scene.grid_mut(id).expect("grid just added"));
        scene
    }

    /// A y/z-spanning wall of `thick` voxels starting at `x0`, with a
    /// generous extent so jitter rays can't slip around the edges.
    fn wall(grid: &mut roxlap_scene::Grid, x0: i32, thick: i32) {
        grid.set_rect(
            IVec3::new(x0, 0, 100),
            IVec3::new(x0 + thick - 1, 127, 180),
            Some(STONE),
        );
    }

    #[test]
    fn open_air_is_clear() {
        let scene = scene_with(|_| {});
        let cfg = AcousticsConfig::default();
        let a = source_acoustics(
            &scene,
            DVec3::new(10.0, 40.0, 140.0),
            DVec3::new(60.0, 40.0, 140.0),
            &cfg,
        );
        assert_eq!(a, SourceAcoustics::clear(&cfg));
    }

    #[test]
    fn thickness_is_exact_for_perpendicular_and_diagonal_crossings() {
        let scene = scene_with(|g| wall(g, 30, 3));
        // Perpendicular: exactly the wall's 3 voxels.
        let t = path_thickness(
            &scene,
            DVec3::new(10.0, 40.5, 140.5),
            DVec3::new(60.0, 40.5, 140.5),
        );
        assert!((t - 3.0).abs() < 1e-6, "perpendicular thickness: {t}");
        // 45° in the xy plane: path length through 3 voxels of x-extent
        // is 3·√2.
        let t = path_thickness(
            &scene,
            DVec3::new(10.0, 20.5, 140.5),
            DVec3::new(60.0, 70.5, 140.5),
        );
        assert!((t - 3.0 * 2.0f64.sqrt()).abs() < 1e-6, "diagonal: {t}");
    }

    #[test]
    fn thicker_walls_muffle_monotonically() {
        let cfg = AcousticsConfig::default();
        let src = DVec3::new(10.0, 64.0, 140.5);
        let dst = DVec3::new(60.0, 64.0, 140.5);
        let mut prev = 1.0f32;
        for thick in [1, 2, 5] {
            let scene = scene_with(|g| wall(g, 30, thick));
            let a = source_acoustics(&scene, src, dst, &cfg);
            assert!(
                a.transmission < prev,
                "{thick}-voxel wall must transmit less than the previous ({} vs {prev})",
                a.transmission
            );
            assert!(a.gain_db < 0.0 && a.gain_db >= cfg.max_occlusion_db);
            assert!(a.lowpass_cutoff_hz < cfg.open_cutoff_hz);
            assert!(a.lowpass_cutoff_hz > cfg.occluded_cutoff_hz);
            prev = a.transmission;
        }
    }

    #[test]
    fn doorway_leaks_more_than_sealed_wall() {
        let src = DVec3::new(10.0, 64.0, 140.5);
        let dst = DVec3::new(60.0, 64.0, 140.5);
        let cfg = AcousticsConfig::default();

        let sealed = scene_with(|g| wall(g, 30, 2));
        let sealed_a = source_acoustics(&sealed, src, dst, &cfg);

        // Same wall with a slot punched just off the direct line. The
        // ring converges toward the listener, so at the wall plane
        // (40% of the way) the ±1.5 jitter has shrunk to ±0.9 voxels:
        // the slot at cell y=63 (z 139..141) is inside that reach for
        // the three −y spokes while the direct ray still hits rock.
        let doorway = scene_with(|g| {
            wall(g, 30, 2);
            g.set_rect(IVec3::new(30, 63, 139), IVec3::new(31, 63, 141), None);
        });
        let door_a = source_acoustics(&doorway, src, dst, &cfg);
        assert!(
            door_a.transmission > sealed_a.transmission,
            "doorway must leak: {} vs sealed {}",
            door_a.transmission,
            sealed_a.transmission
        );
        assert!(
            door_a.transmission < 1.0,
            "off-axis doorway is not fully clear"
        );
    }

    #[test]
    fn cross_chunk_wall_counts_once() {
        // Wall straddling the x = 128 chunk seam: thickness must not
        // double-count or drop the seam cell.
        let scene = scene_with(|g| {
            g.set_rect(
                IVec3::new(126, 0, 100),
                IVec3::new(129, 255, 180),
                Some(STONE),
            );
        });
        let t = path_thickness(
            &scene,
            DVec3::new(100.0, 64.5, 140.5),
            DVec3::new(160.0, 64.5, 140.5),
        );
        assert!((t - 4.0).abs() < 1e-6, "seam-straddling wall: {t}");
    }

    #[test]
    fn cross_grid_thickness_sums() {
        // Two grids, one wall each, both crossed by the same segment.
        let mut scene = Scene::new();
        let id_a = scene.add_grid(GridTransform::identity());
        wall(scene.grid_mut(id_a).expect("grid a"), 20, 2);
        let id_b = scene.add_grid(GridTransform::at(DVec3::new(20.0, 0.0, 0.0)));
        wall(scene.grid_mut(id_b).expect("grid b"), 20, 2); // world x = 40..42
        let t = path_thickness(
            &scene,
            DVec3::new(10.0, 64.5, 140.5),
            DVec3::new(60.0, 64.5, 140.5),
        );
        assert!((t - 4.0).abs() < 1e-6, "two 2-voxel walls: {t}");
    }

    #[test]
    fn beyond_max_distance_reports_clear_not_buried() {
        // Distance attenuation belongs to the spatial backend: past the
        // march budget a source must NOT come back muffled (that would
        // put a lowpass step function at the boundary).
        let scene = scene_with(|g| wall(g, 30, 5));
        let cfg = AcousticsConfig::default();
        let a = source_acoustics(
            &scene,
            DVec3::new(10.0, 64.0, 140.5),
            DVec3::new(10.0 + cfg.max_distance + 10.0, 64.0, 140.5),
            &cfg,
        );
        assert_eq!(a, SourceAcoustics::clear(&cfg));
    }

    #[test]
    fn rotated_grid_thickness_is_world_true() {
        use glam::{DQuat, DVec3};
        // A grid rotated 90° about world Z, placed so its local wall
        // (3 voxels along local x) lands across the world-Y axis: the
        // rebased segment must report the same world-true 3 voxels.
        let mut scene = Scene::new();
        let rot = DQuat::from_rotation_z(std::f64::consts::FRAC_PI_2);
        let id = scene.add_grid(GridTransform {
            origin: DVec3::new(0.0, 0.0, 0.0),
            rotation: rot,
        });
        wall(scene.grid_mut(id).expect("rotated grid"), 30, 3);
        // Local +x maps to world +y: the local x = 30..33 wall lies at
        // world y = 30..33 (local y spans map to world -x). Probe along
        // world +y through it; local y = 40 ⇒ world x = -40.
        let t = path_thickness(
            &scene,
            DVec3::new(-40.5, 10.0, 140.5),
            DVec3::new(-40.5, 60.0, 140.5),
        );
        assert!(
            (t - 3.0).abs() < 1e-6,
            "rotated-grid wall must stay 3 world voxels thick: {t}"
        );
        // Control: the same segment misses the wall in an identity grid.
        let control = scene_with(|g| wall(g, 30, 3));
        let t = path_thickness(
            &control,
            DVec3::new(-40.5, 10.0, 140.5),
            DVec3::new(-40.5, 60.0, 140.5),
        );
        assert!(t.abs() < 1e-9, "identity control must miss: {t}");
    }

    #[test]
    fn bedrock_plane_counts_as_solid_documented() {
        // Voxlap's bedrock placeholder (chunk-local z = 255 of an
        // otherwise-empty chunk) reads as solid — documented caveat on
        // `grid_thickness`: keep acoustic endpoints above the world
        // bottom. This test pins the current behaviour.
        let scene = scene_with(|g| {
            g.ensure_chunk(IVec3::ZERO);
        });
        let t = path_thickness(
            &scene,
            DVec3::new(10.0, 64.0, 255.5),
            DVec3::new(20.0, 64.0, 255.5),
        );
        assert!(t > 9.0, "bedrock plane accumulates thickness: {t}");
    }

    #[test]
    fn deterministic() {
        let scene = scene_with(|g| wall(g, 30, 2));
        let cfg = AcousticsConfig::default();
        let src = DVec3::new(10.0, 64.0, 140.5);
        let dst = DVec3::new(60.0, 64.0, 140.5);
        let a = source_acoustics(&scene, src, dst, &cfg);
        let b = source_acoustics(&scene, src, dst, &cfg);
        assert_eq!(a, b);
    }
}
