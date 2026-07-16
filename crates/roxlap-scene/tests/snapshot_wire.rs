//! QE.5b — snapshot **wire-format** compatibility gate.
//!
//! `tests/fixtures/snapshot_v{1,2,3,4}.rxs` are checked-in
//! [`Scene::save_snapshot`] blobs (magic + version + bincode payload)
//! frozen at each wire version. The `v*_fixture_loads` tests must keep
//! passing on every future engine version — that is the
//! backward-compatibility promise the envelope exists for. If the payload
//! shape ever has to change, bump `roxlap_scene::snapshot::SNAPSHOT_VERSION`,
//! add a version-N shadow shape to `load_snapshot`, and leave the older
//! fixtures untouched (SC.snap did this for v1→v2 + `voxel_world_size`;
//! WT.0 for v2→v3 + `water_volumes`; CA.0 for v3→v4 + `z_clip`).
//!
//! To regenerate the **v4** fixture after deliberately changing its
//! reference scene (NOT after a wire-format change — see above):
//! `cargo test -p roxlap-scene --test snapshot_wire -- --ignored`. The
//! v1/v2/v3 fixtures are intentionally frozen (current `save_snapshot`
//! emits v4).

use glam::{DVec3, IVec3};
use roxlap_scene::snapshot::SnapshotLoadError;
use roxlap_scene::VoxColor;
use roxlap_scene::{GridTransform, LodThresholds, Scene, StreamRadius, WaterVolume};

const FIXTURE: &[u8] = include_bytes!("fixtures/snapshot_v1.rxs");
/// SC.snap — a v2 blob with a scaled grid (see [`assert_scaled_scene`]).
const FIXTURE_V2: &[u8] = include_bytes!("fixtures/snapshot_v2.rxs");
/// WT.0 — a v3 blob with water volumes (see [`assert_water_scene`]).
const FIXTURE_V3: &[u8] = include_bytes!("fixtures/snapshot_v3.rxs");
/// CA.0 — a v4 blob with a cutaway clip (see [`build_clipped_scene`]).
const FIXTURE_V4: &[u8] = include_bytes!("fixtures/snapshot_v4.rxs");

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
    // WT.0 — a pre-water save restores dry.
    for (_, g) in scene.grids() {
        assert!(g.water_volumes.is_empty(), "v1 predates water volumes");
    }
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

/// Assert `scene` matches the frozen v2 fixture's reference scene — one
/// grid named "ship" at `voxel_world_size = 0.25` (the capability v2
/// added) with one edit. The builder that generated the fixture was
/// removed when v3 froze v2 (a frozen fixture needs no regeneration
/// path); the persisted scale is the point.
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
    // WT.0 — a pre-water save restores dry.
    let (_, g) = scene.grids().next().expect("one grid");
    assert!(g.water_volumes.is_empty(), "v2 predates water volumes");
}

/// Assert `scene` matches the frozen v3 fixture's reference scene — a
/// scaled grid named "flooded" with two water volumes (the capability
/// v3 added). The builder that generated the fixture was removed when
/// v4 froze v3 (a frozen fixture needs no regeneration path); the
/// persisted water is the point.
fn assert_water_scene(scene: &Scene) {
    let (_, g) = scene.grids().next().expect("one grid");
    assert_eq!(g.transform.voxel_world_size, 0.5);
    assert_eq!(g.name.as_deref(), Some("flooded"));
    assert_eq!(
        g.water_volumes,
        vec![
            WaterVolume::new(IVec3::new(0, 0, 100), IVec3::new(31, 31, 127)),
            WaterVolume::new(IVec3::new(40, 40, 110), IVec3::new(50, 50, 115)),
        ],
        "v3 must persist water volumes"
    );
}

/// WT.0 — a v3 blob (with water volumes) stays loadable, the same
/// forever-promise as the v1/v2 fixtures.
#[test]
fn v3_fixture_loads() {
    let scene =
        Scene::load_snapshot(FIXTURE_V3).expect("checked-in v3 fixture must stay loadable forever");
    assert_water_scene(&scene);
    // CA.0 — a pre-cutaway save restores unclipped.
    let (_, g) = scene.grids().next().expect("one grid");
    assert_eq!(g.z_clip, None, "v3 predates the cutaway clip");
}

/// CA.0 — the v4 reference scene: a grid with water volumes AND a
/// cutaway clip (the capability v4 adds), so the v4 fixture proves the
/// persisted clip survives alongside the v3-era fields. Deliberately
/// CHUNK-FREE, same rationale as the frozen v3 fixture (chunk decoding
/// is already gated by v1/v2; v4 gates only what v4 added).
fn build_clipped_scene() -> Scene {
    let mut scene = Scene::new();
    let id = scene.add_grid(GridTransform::at_scale(DVec3::new(1.0, 2.0, 0.0), 0.5));
    let g = scene.grid_mut(id).expect("grid just added");
    g.name = Some("shiplet".to_owned());
    g.add_water_volume(IVec3::new(0, 0, 100), IVec3::new(31, 31, 127));
    g.z_clip = Some(-96);
    scene
}

/// Assert `scene` matches [`build_clipped_scene`] — the persisted clip is
/// the point (a NEGATIVE value on purpose: stacked-chz grids clip above
/// chz 0, and a sign bug would slip through a positive-only fixture).
fn assert_clipped_scene(scene: &Scene) {
    let (_, g) = scene.grids().next().expect("one grid");
    assert_eq!(g.name.as_deref(), Some("shiplet"));
    assert_eq!(g.z_clip, Some(-96), "v4 must persist the cutaway clip");
    assert_eq!(
        g.water_volumes,
        vec![WaterVolume::new(
            IVec3::new(0, 0, 100),
            IVec3::new(31, 31, 127)
        )],
        "v3-era fields must survive the v4 shape"
    );
}

/// CA.0 — a v4 blob (with a cutaway clip) stays loadable, the same
/// forever-promise as the v1/v2/v3 fixtures.
#[test]
fn v4_fixture_loads() {
    let scene =
        Scene::load_snapshot(FIXTURE_V4).expect("checked-in v4 fixture must stay loadable forever");
    assert_clipped_scene(&scene);
}

/// CA.0 — the cutaway clip round-trips through the CURRENT wire format
/// (independent of the fixture, so a green run proves both directions).
#[test]
fn z_clip_round_trip() {
    let bytes = build_clipped_scene().save_snapshot();
    let scene = Scene::load_snapshot(&bytes).expect("round trip");
    assert_clipped_scene(&scene);
}

/// Regenerate the checked-in **v4** fixture. Only run deliberately after a
/// deliberate reference-scene change (NOT after a wire-format change — bump
/// SNAPSHOT_VERSION and add a shadow shape instead):
/// `cargo test -p roxlap-scene --test snapshot_wire -- --ignored`
///
/// The v1/v2/v3 fixtures are intentionally NOT regenerable here: current
/// `save_snapshot` emits v4, so the frozen `snapshot_v1.rxs` /
/// `snapshot_v2.rxs` / `snapshot_v3.rxs` (the backward-compat gates) are
/// left untouched by design.
#[test]
#[ignore = "writes tests/fixtures/snapshot_v4.rxs; run manually"]
fn regenerate_v4_fixture() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/snapshot_v4.rxs"
    );
    std::fs::write(path, build_clipped_scene().save_snapshot()).expect("write fixture");
}
