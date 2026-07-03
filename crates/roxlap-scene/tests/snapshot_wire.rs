//! QE.5b — snapshot **wire-format** compatibility gate.
//!
//! `tests/fixtures/snapshot_v1.rxs` is a checked-in
//! [`Scene::save_snapshot`] blob (magic + version 1 + bincode
//! payload) frozen when the envelope landed. [`v1_fixture_loads`]
//! must keep passing on every future engine version — that is the
//! backward-compatibility promise the envelope exists for. If the
//! payload shape ever has to change, bump
//! `roxlap_scene::snapshot::SNAPSHOT_VERSION`, teach `load_snapshot`
//! to migrate version 1, and leave this fixture untouched.
//!
//! To regenerate after deliberately changing the reference scene
//! (NOT after changing the wire format — see above):
//! `cargo test -p roxlap-scene --test snapshot_wire -- --ignored`

use glam::{DVec3, IVec3};
use roxlap_scene::snapshot::SnapshotLoadError;
use roxlap_scene::{GridTransform, LodThresholds, Scene, StreamRadius};

const FIXTURE: &[u8] = include_bytes!("fixtures/snapshot_v1.rxs");

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
        g.set_voxel(IVec3::new(3, 4, 100), Some(0x8010_2030));
        g.set_voxel(IVec3::new(-5, 7, 90), Some(0x8040_5060));
    }
    let id2 = scene.add_grid(GridTransform::identity());
    scene
        .grid_mut(id2)
        .expect("grid just added")
        .set_voxel(IVec3::new(0, 0, 128), Some(0x8077_8899));
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

/// Regenerate the checked-in fixture from the reference scene. Only
/// run deliberately (see the module docs):
/// `cargo test -p roxlap-scene --test snapshot_wire -- --ignored`
#[test]
#[ignore = "writes tests/fixtures/snapshot_v1.rxs; run manually"]
fn regenerate_fixture() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/snapshot_v1.rxs"
    );
    std::fs::write(path, build_reference_scene().save_snapshot()).expect("write fixture");
}
