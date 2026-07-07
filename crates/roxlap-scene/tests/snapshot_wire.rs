//! QE.5b — snapshot **wire-format** compatibility gate.
//!
//! `tests/fixtures/snapshot_v{1,2}.rxs` are checked-in
//! [`Scene::save_snapshot`] blobs (magic + version + bincode payload)
//! frozen at each wire version. [`v1_fixture_loads`] / [`v2_fixture_loads`]
//! must keep passing on every future engine version — that is the
//! backward-compatibility promise the envelope exists for. If the payload
//! shape ever has to change, bump `roxlap_scene::snapshot::SNAPSHOT_VERSION`,
//! add a version-N shadow shape to `load_snapshot`, and leave BOTH fixtures
//! untouched (SC.snap did exactly this for v1→v2 + `voxel_world_size`).
//!
//! To regenerate the **v2** fixture after deliberately changing its
//! reference scene (NOT after a wire-format change — see above):
//! `cargo test -p roxlap-scene --test snapshot_wire -- --ignored`. The v1
//! fixture is intentionally frozen (current `save_snapshot` emits v2).

use glam::{DVec3, IVec3};
use roxlap_scene::snapshot::SnapshotLoadError;
use roxlap_scene::VoxColor;
use roxlap_scene::{GridTransform, LodThresholds, Scene, StreamRadius};

const FIXTURE: &[u8] = include_bytes!("fixtures/snapshot_v1.rxs");
/// SC.snap — a v2 blob with a scaled grid (see [`build_scaled_scene`]).
const FIXTURE_V2: &[u8] = include_bytes!("fixtures/snapshot_v2.rxs");

/// The scene the fixture encodes: two grids — a fully-configured
/// named "terrain" grid with edits (versions non-zero) spanning a
/// negative chunk index, and a default-config anonymous grid.
fn build_reference_scene() -> Scene {
    let mut scene = Scene::new();
    let id = scene.add_grid(GridTransform::at(DVec3::new(10.0, -20.0, 0.0)));
    {
        let g = scene.grid_mut(id).expect("grid just added");
        g.name = Some("terrain".to_owned());
        g.render_sky = false;
        g.mip_levels_override = Some(2);
        g.lod_thresholds = LodThresholds {
            r_near: 100.0,
            r_mid: 400.0,
            mid_mip_levels: Some(3),
            mid_mip_scan_dist: Some(32),
        };
        g.stream_radius = StreamRadius::new(128.0, 256.0);
        g.set_voxel(IVec3::new(3, 4, 100), Some(VoxColor(0x8010_2030)));
        g.set_voxel(IVec3::new(-5, 7, 90), Some(VoxColor(0x8040_5060)));
    }
    let id2 = scene.add_grid(GridTransform::identity());
    scene
        .grid_mut(id2)
        .expect("grid just added")
        .set_voxel(IVec3::new(0, 0, 128), Some(VoxColor(0x8077_8899)));
    scene
}

/// Assert `scene` matches [`build_reference_scene`]'s observable
/// state — shared by the fixture-load and round-trip tests.
fn assert_reference_scene(scene: &Scene) {
    let mut grids: Vec<_> = scene.grids().collect();
    grids.sort_by_key(|(id, _)| *id);
    assert_eq!(grids.len(), 2);

    let (_, terrain) = grids[0];
    assert_eq!(terrain.name.as_deref(), Some("terrain"));
    assert_eq!(terrain.transform.origin, DVec3::new(10.0, -20.0, 0.0));
    assert!(!terrain.render_sky);
    assert_eq!(terrain.mip_levels_override, Some(2));
    assert_eq!(terrain.lod_thresholds.r_near, 100.0);
    assert_eq!(terrain.lod_thresholds.mid_mip_scan_dist, Some(32));
    assert_eq!(terrain.stream_radius, StreamRadius::new(128.0, 256.0));
    assert!(terrain.voxel_solid(IVec3::new(3, 4, 100)));
    assert!(terrain.voxel_solid(IVec3::new(-5, 7, 90)));
    assert!(terrain.chunk_version(IVec3::new(0, 0, 0)) > 0);
    assert!(terrain.chunk_version(IVec3::new(-1, 0, 0)) > 0);

    let (_, plain) = grids[1];
    assert_eq!(plain.name, None);
    assert!(plain.render_sky);
    assert_eq!(plain.stream_radius, StreamRadius::DISABLED);
    assert!(plain.voxel_solid(IVec3::new(0, 0, 128)));
}

#[test]
fn v1_fixture_loads() {
    let scene =
        Scene::load_snapshot(FIXTURE).expect("checked-in v1 fixture must stay loadable forever");
    assert_reference_scene(&scene);
}

#[test]
fn save_load_round_trip_preserves_grid_config() {
    let bytes = build_reference_scene().save_snapshot();
    let scene = Scene::load_snapshot(&bytes).expect("round trip");
    assert_reference_scene(&scene);
}

#[test]
fn rejects_bad_magic_truncation_and_future_versions() {
    assert!(matches!(
        Scene::load_snapshot(b"XXsomething else entirely"),
        Err(SnapshotLoadError::BadMagic)
    ));
    assert!(matches!(
        Scene::load_snapshot(b"RX"),
        Err(SnapshotLoadError::BadMagic)
    ));
    // A future version must refuse loudly, never misparse.
    let mut future = build_reference_scene().save_snapshot();
    future[4..8].copy_from_slice(&99u32.to_le_bytes());
    assert!(matches!(
        Scene::load_snapshot(&future),
        Err(SnapshotLoadError::UnsupportedVersion(99))
    ));
    // Truncated payload → Decode, not a panic.
    let bytes = build_reference_scene().save_snapshot();
    assert!(matches!(
        Scene::load_snapshot(&bytes[..bytes.len() / 2]),
        Err(SnapshotLoadError::Decode(_))
    ));
}

/// SC.snap — the v2 reference scene: one grid with a non-1
/// `voxel_world_size` (the capability v2 adds) plus a couple of edits, so
/// the v2 fixture proves persisted scale survives.
fn build_scaled_scene() -> Scene {
    let mut scene = Scene::new();
    let id = scene.add_grid(GridTransform::at_scale(DVec3::new(5.0, -3.0, 0.0), 0.25));
    let g = scene.grid_mut(id).expect("grid just added");
    g.name = Some("ship".to_owned());
    g.set_voxel(IVec3::new(2, 3, 100), Some(VoxColor(0x8012_3456)));
    scene
}

/// Assert `scene` matches [`build_scaled_scene`] — the persisted scale is
/// the point.
fn assert_scaled_scene(scene: &Scene) {
    let (_, g) = scene.grids().next().expect("one grid");
    assert_eq!(g.transform.voxel_world_size, 0.25, "v2 must persist scale");
    assert_eq!(g.transform.origin, DVec3::new(5.0, -3.0, 0.0));
    assert_eq!(g.name.as_deref(), Some("ship"));
    assert!(g.voxel_solid(IVec3::new(2, 3, 100)));
}

/// SC.snap — a v2 blob (with a scaled grid) stays loadable, the same
/// forever-promise as [`v1_fixture_loads`].
#[test]
fn v2_fixture_loads() {
    let scene =
        Scene::load_snapshot(FIXTURE_V2).expect("checked-in v2 fixture must stay loadable forever");
    assert_scaled_scene(&scene);
}

/// Regenerate the checked-in **v2** fixture. Only run deliberately after a
/// deliberate reference-scene change (NOT after a wire-format change — bump
/// SNAPSHOT_VERSION and add a shadow shape instead):
/// `cargo test -p roxlap-scene --test snapshot_wire -- --ignored`
///
/// The v1 fixture is intentionally NOT regenerable here: current
/// `save_snapshot` emits v2, so the frozen `snapshot_v1.rxs` (the
/// backward-compat gate) is left untouched by design.
#[test]
#[ignore = "writes tests/fixtures/snapshot_v2.rxs; run manually"]
fn regenerate_v2_fixture() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/snapshot_v2.rxs"
    );
    std::fs::write(path, build_scaled_scene().save_snapshot()).expect("write fixture");
}
