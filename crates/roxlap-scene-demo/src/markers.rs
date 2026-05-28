//! S6.6 — LOD validation markers.
//!
//! Adds 5 small "totem pole" grids spaced along world `+y` so the
//! camera (spawned at `y = -120`) sees a row of pillars at
//! increasing distance. Each pillar is a single-chunk grid with a
//! striped colour pattern that makes mip-blur / billboard pixelation
//! visually obvious.
//!
//! Distance layout (chosen so that at the spawn camera pose, 1
//! pillar is Near and 4 are Far — exercising both render paths but
//! biasing toward the cheaper billboard blit so the demo stays
//! interactive):
//!
//! | Marker | World y | Spawn distance | Default tier |
//! |--------|---------|----------------|--------------|
//! | 0      | 100     | ~297           | Near         |
//! | 1      | 500     | ~652           | Far          |
//! | 2      | 900     | ~1040          | Far          |
//! | 3      | 1300    | ~1440          | Far          |
//! | 4      | 1700    | ~1838          | Far          |
//!
//! Default — LOD off: every marker grid uses
//! [`LodThresholds::always_near`], so the row of pillars looks
//! identical to a row of regular grids (full voxel raycast). This
//! preserves the pre-S6 demo behaviour.
//!
//! `B` hotkey — LOD on: every marker grid's thresholds switch to
//! `r_near = 400, r_mid = 400` (no Mid band — just Near vs Far).
//! The closer pillars stay Near (full voxel); the farther pillars
//! flip to [`Lod::Far`] and render via S6.3's billboard impostor
//! blit. The Mid tier is deliberately skipped here because the
//! `set_rect + generate_mips` chunk currently hits the known
//! all-sky-at-distant-mip bug
//! ([[project_mip_attempt]]); when that's fixed, the demo's
//! thresholds can grow a real Mid band.

use glam::{DVec3, IVec3};
use roxlap_scene::{Grid, GridId, GridTransform, LodThresholds, Scene};

/// World-y of the first (closest) marker pillar. Picked so the
/// pillar appears in front of the camera at spawn pose but past
/// the ship grid's tail (saucer origin at world `(0, 500, -100)`).
pub const MARKER_FIRST_Y: f64 = 100.0;

/// World-y spacing between consecutive marker pillars. 400-voxel
/// gaps spread 5 pillars across `y ∈ [100, 1700]`, sized so that
/// most of the pillars are past the default `r_near = 400`
/// threshold at the spawn pose (cheap Far-tier billboards) while
/// the closest pillar stays Near for visual reference.
pub const MARKER_Y_SPACING: f64 = 400.0;

/// World-x offset of every marker pillar from the spawn-camera
/// line of sight. Picked so the pillars sit slightly off to the
/// camera's right (camera looks `+y` from `x = 0`); also keeps
/// the pillars clear of the ship grid's footprint at
/// `(0, 500, -100)`.
pub const MARKER_X: f64 = 200.0;

/// World-z of every marker pillar's origin — same as the spawn
/// camera so the pillars line up at horizon height in the default
/// view.
pub const MARKER_Z: f64 = 50.0;

/// Number of marker pillars created. PORTING-SCENE.md § S6's
/// validation gate calls for "10 small grids at varying distances",
/// but 10 markers all in Near tier (close to spawn) dropped the
/// demo to ~4 FPS because each Near render does a full opticast
/// pass + per-grid temp buffer fill + compose merge. 5 is enough
/// for tier variety while keeping the demo interactive; the
/// distance layout (see module doc) biases toward Far so most
/// markers exercise the cheap billboard blit path.
pub const NUM_MARKERS: usize = 5;

/// `r_near` threshold used by the `B`-hotkey LOD configuration.
/// Picked so the nearest 3-4 pillars stay Near at the spawn camera
/// pose; pillars beyond this distance flip to Far. Note: the
/// distance metric is centre-to-centre Euclidean, not perpendicular
/// to the camera forward, so the actual Near/Far boundary as the
/// user moves the camera depends on lateral offset too.
pub const LOD_R_NEAR: f64 = 400.0;

/// `r_mid` threshold for the LOD configuration. Set equal to
/// `LOD_R_NEAR` to collapse the Mid band — the demo isolates the
/// Near vs Far behaviour. See module doc for the rationale.
pub const LOD_R_MID: f64 = 400.0;

/// Palette of 5 (base, stripe) colour pairs — one per marker. ARGB
/// with the alpha-bit pattern voxlap expects (`0x80` brightness +
/// `0x00` flags). Designed to be distinguishable from the ground +
/// ship + sky.
const PALETTE: [(u32, u32); NUM_MARKERS] = [
    (0x80_aa_22_22, 0x80_ee_ee_ee), // red    + white
    (0x80_22_aa_22, 0x80_22_22_88), // green  + blue
    (0x80_22_22_aa, 0x80_ee_ee_22), // blue   + yellow
    (0x80_aa_aa_22, 0x80_aa_22_aa), // yellow + magenta
    (0x80_aa_22_aa, 0x80_22_aa_aa), // magenta + cyan
];

/// Build a single marker pillar inside `grid`. The pillar lives in
/// chunk-local `(49..78, 49..78, 78..177)` — a 30×30 base 100-tall
/// column **centred on the chunk centre `(64, 64, 128)`** so the
/// `BillboardCache::build` bounding-sphere centre (which uses the
/// chunk-index bbox, not the populated voxels) matches the pillar's
/// actual centre. Without this centring, the S6.3 blit positions
/// the billboard quad at the chunk centre while the pillar content
/// projects from an offset position — making impostors appear
/// floating above/below the real voxel position when toggled.
///
/// Stripes alternate between `base` and `stripe` every 8 voxels
/// (4-on, 4-off).
///
/// **No `generate_mips` call.** A `set_rect`-built chunk with this
/// shape hits the [[mip_attempt]] bug — `Vxl::generate_mips` reads
/// past the slab buffer after the cumulative stripes grow it past
/// the initial capacity. The demo's `B` toggle skips Mid (Near vs
/// Far only — see module doc), so the multi-mip ladder is dead
/// weight here anyway. The S6.3 billboard cache builds its
/// snapshots via opticast at `mip_levels = 1`, so the missing
/// ladder doesn't affect the Far-tier render either.
fn build_one_marker(grid: &mut Grid, base: u32, stripe: u32) {
    // Solid base — 30×30×100, centred on chunk centre (64, 64, 128).
    grid.set_rect(IVec3::new(49, 49, 78), IVec3::new(78, 78, 177), Some(base));
    // 4-voxel stripes every 8 voxels (4-on, 4-off).
    for z in (78..177).step_by(8) {
        grid.set_rect(
            IVec3::new(49, 49, z),
            IVec3::new(78, 78, z + 3),
            Some(stripe),
        );
    }
}

/// Add 10 marker grids to `scene` and return their [`GridId`]s in
/// distance order (closest first). All marker grids start with
/// [`LodThresholds::always_near`] so they render at full voxel
/// detail; toggle to LOD mode via [`set_billboards_lod`].
#[must_use]
pub fn build_markers(scene: &mut Scene) -> Vec<GridId> {
    let mut ids = Vec::with_capacity(NUM_MARKERS);
    for (i, &(base, stripe)) in PALETTE.iter().enumerate() {
        #[allow(clippy::cast_precision_loss)]
        let y = MARKER_FIRST_Y + (i as f64) * MARKER_Y_SPACING;
        let origin = DVec3::new(MARKER_X, y, MARKER_Z);
        let id = scene.add_grid(GridTransform::at(origin));
        let grid = scene.grid_mut(id).expect("marker grid present");
        build_one_marker(grid, base, stripe);
        // Default to always-Near so the demo's pre-S6.6 behaviour
        // (all markers full voxel) is preserved until `B` toggles
        // LOD mode on.
        grid.lod_thresholds = LodThresholds::always_near();
        ids.push(id);
    }
    ids
}

/// Toggle every marker grid's [`LodThresholds`] between
/// `always_near` (`on = false`) and the tuned LOD config
/// (`on = true`, `r_near = r_mid = LOD_R_NEAR`).
///
/// Called by the `B` hotkey. Has no effect on the ground or ship
/// grids — those keep their existing thresholds. Marker grids'
/// billboard caches are NOT pre-invalidated; `render_scene_composed`'s
/// Far branch handles lazy population on first Far-tier render.
pub fn set_billboards_lod(scene: &mut Scene, marker_ids: &[GridId], on: bool) {
    let thresholds = if on {
        LodThresholds {
            r_near: LOD_R_NEAR,
            r_mid: LOD_R_MID,
            mid_mip_levels: None,
            mid_mip_scan_dist: None,
        }
    } else {
        LodThresholds::always_near()
    };
    for &id in marker_ids {
        if let Some(grid) = scene.grid_mut(id) {
            grid.lod_thresholds = thresholds;
        }
    }
}
