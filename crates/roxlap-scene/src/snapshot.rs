//! Serde-friendly snapshot of a [`Scene`].
//!
//! The live [`Scene`] holds [`Vxl`] chunks with allocator state
//! ([`Vxl::vbit`] / [`Vxl::vbiti`]) that doesn't round-trip
//! cleanly through serde — the post-edit slab pool is interior
//! mutable and changes shape under voxalloc-driven scatter.
//! [`SceneSnapshot`] is a flattened view: per-chunk bytes
//! produced by [`roxlap_formats::vxl::serialize`], plus the grid
//! transforms / ids the runtime tracks. It's a value type with
//! plain serde derives — bincode / postcard / json / cbor all
//! work.
//!
//! Use [`Scene::to_snapshot`] / [`Scene::from_snapshot`] to round
//! trip. The deserialised scene is editable: each chunk goes back
//! through [`Vxl::reserve_edit_capacity`] so subsequent
//! `Grid::set_*` calls don't panic.
//!
//! [`Scene`]: crate::Scene
//! [`Vxl`]: roxlap_formats::vxl::Vxl
//! [`Vxl::vbit`]: roxlap_formats::vxl::Vxl::vbit
//! [`Vxl::vbiti`]: roxlap_formats::vxl::Vxl::vbiti
//! [`Vxl::reserve_edit_capacity`]: roxlap_formats::vxl::Vxl::reserve_edit_capacity

use glam::IVec3;
use roxlap_formats::vxl::{self, ParseError, Vxl};
use serde::{Deserialize, Serialize};

use crate::{Grid, GridId, GridTransform, Scene};

/// Re-encode a [`Vxl`]'s mip-0 columns into a contiguous bytes
/// blob that round-trips through [`vxl::parse`].
///
/// [`vxl::serialize`] writes the live `vxl.data` array verbatim,
/// which breaks the round-trip after edits: post-`voxalloc`
/// scatter, columns may live in the edit pool past the
/// originally-contiguous prefix, and `vxl::parse` walks columns
/// linearly from offset 0. This helper builds a temporary
/// contiguous [`Vxl`] (column index order) and serialises that —
/// the result is a layout `vxl::parse` accepts even after
/// arbitrary `set_voxel` / `set_rect` / `set_sphere` edits.
///
/// Mip-1+ data isn't preserved (the renderer rebuilds it on
/// demand). Snapshots are pre-mip; the receiver calls
/// [`Vxl::generate_mips`] if it wants them.
fn compact_serialize_chunk(vxl: &Vxl) -> Vec<u8> {
    let n_cols = (vxl.vsid as usize) * (vxl.vsid as usize);
    let mut data: Vec<u8> = Vec::new();
    let mut column_offset: Vec<u32> = Vec::with_capacity(n_cols + 1);
    for i in 0..n_cols {
        column_offset.push(u32::try_from(data.len()).expect("offset fits in u32"));
        data.extend_from_slice(vxl.column_data(i));
    }
    column_offset.push(u32::try_from(data.len()).expect("offset fits in u32"));

    let compact = Vxl {
        vsid: vxl.vsid,
        ipo: vxl.ipo,
        ist: vxl.ist,
        ihe: vxl.ihe,
        ifo: vxl.ifo,
        data: data.into_boxed_slice(),
        column_offset: column_offset.into_boxed_slice(),
        mip_base_offsets: Box::new([0, n_cols + 1]),
        vbit: Box::new([]),
        vbiti: 0,
    };
    vxl::serialize(&compact)
}

/// Bytes of edit-pool headroom re-applied per chunk during
/// [`Scene::from_snapshot`]. Matches the value chunk creation uses
/// in [`crate::chunks`] so a snapshot round-trip leaves chunks
/// equally edit-ready as freshly-created ones.
const RESTORE_EDIT_HEADROOM_PER_COLUMN: usize = 256;

/// Top-level scene snapshot — full state needed to reconstruct a
/// [`Scene`] via [`Scene::from_snapshot`].
///
/// Grids serialised as a `Vec<(GridId, GridSnapshot)>` rather than
/// a `HashMap` so the wire form is independent of `HashMap`'s
/// non-deterministic iteration order — the same scene snapshot
/// twice in a row produces byte-identical output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneSnapshot {
    /// Next id [`Scene::add_grid`] will hand out. Preserved across
    /// snapshot round-trips so removed-id non-reuse holds.
    pub next_grid_id: u32,
    /// All registered grids paired with their ids.
    pub grids: Vec<(GridId, GridSnapshot)>,
}

/// One grid's snapshot: transform + flattened chunks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridSnapshot {
    pub transform: GridTransform,
    /// Chunks as `(chunk_idx, vxl_bytes)`. `vxl_bytes` is
    /// [`roxlap_formats::vxl::serialize`] output — re-parseable
    /// via [`roxlap_formats::vxl::parse`].
    pub chunks: Vec<(IVec3, Vec<u8>)>,
    /// S7.2: per-chunk edit version counters, sorted by chunk
    /// index. Chunks absent from this list restore as version 0 —
    /// the same as a freshly generated or pre-S7.2 snapshot. The
    /// `#[serde(default)]` attribute lets a pre-S7.2 snapshot
    /// deserialise into this newer struct shape cleanly.
    #[serde(default)]
    pub chunk_versions: Vec<(IVec3, u64)>,
}

/// Errors from [`Scene::from_snapshot`]. Wraps the per-chunk
/// [`ParseError`] with a tag identifying which grid + chunk
/// failed.
#[derive(Debug)]
pub enum FromSnapshotError {
    /// One chunk's bytes failed to round-trip through
    /// [`roxlap_formats::vxl::parse`].
    ChunkParse {
        grid: GridId,
        chunk: IVec3,
        source: ParseError,
    },
}

impl std::fmt::Display for FromSnapshotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ChunkParse {
                grid,
                chunk,
                source,
            } => {
                write!(
                    f,
                    "scene snapshot: grid {} chunk {chunk:?} parse failed: {source:?}",
                    grid.raw()
                )
            }
        }
    }
}

impl std::error::Error for FromSnapshotError {}

impl Scene {
    /// Capture the scene's full state as a serde-friendly value.
    /// Each chunk is encoded via
    /// [`roxlap_formats::vxl::serialize`]; the rest is plain field
    /// data.
    ///
    /// Grid iteration order in the produced snapshot is sorted by
    /// [`GridId`] so two snapshots of the same scene produce
    /// byte-identical output (the live `HashMap` iteration order
    /// would be non-deterministic).
    #[must_use]
    pub fn to_snapshot(&self) -> SceneSnapshot {
        let mut grid_ids: Vec<GridId> = self.grids.keys().copied().collect();
        grid_ids.sort_unstable();

        let mut grids = Vec::with_capacity(grid_ids.len());
        for id in grid_ids {
            let grid = &self.grids[&id];
            let mut chunk_addrs: Vec<IVec3> = grid.chunks.keys().copied().collect();
            chunk_addrs.sort_unstable_by_key(|a| (a.x, a.y, a.z));
            let chunks = chunk_addrs
                .into_iter()
                .map(|addr| (addr, compact_serialize_chunk(&grid.chunks[&addr])))
                .collect();
            // S7.2: emit chunk_versions sorted by chunk idx so the
            // snapshot bytes stay deterministic (HashMap iter order
            // is not). Zero entries are dropped on the assumption
            // "missing == 0" — the snapshot stays compact for grids
            // whose live counters are dense in 1+.
            let mut version_addrs: Vec<IVec3> = grid
                .chunk_versions
                .iter()
                .filter_map(|(a, v)| if *v != 0 { Some(*a) } else { None })
                .collect();
            version_addrs.sort_unstable_by_key(|a| (a.x, a.y, a.z));
            let chunk_versions = version_addrs
                .into_iter()
                .map(|addr| (addr, grid.chunk_versions[&addr]))
                .collect();
            grids.push((
                id,
                GridSnapshot {
                    transform: grid.transform,
                    chunks,
                    chunk_versions,
                },
            ));
        }
        SceneSnapshot {
            next_grid_id: self.next_grid_id,
            grids,
        }
    }

    /// Restore a [`Scene`] from a snapshot. Each chunk's bytes are
    /// re-parsed via [`roxlap_formats::vxl::parse`] and re-armed
    /// for edits via [`roxlap_formats::vxl::Vxl::reserve_edit_capacity`].
    ///
    /// # Errors
    ///
    /// Returns [`FromSnapshotError::ChunkParse`] tagged with the
    /// owning grid + chunk index if any chunk's bytes fail to
    /// parse. The partial scene is dropped — restoration is
    /// all-or-nothing.
    pub fn from_snapshot(snap: &SceneSnapshot) -> Result<Self, FromSnapshotError> {
        let mut scene = Self::new();
        scene.next_grid_id = snap.next_grid_id;
        for (id, gsnap) in &snap.grids {
            let mut grid = Grid::new(gsnap.transform);
            for (addr, bytes) in &gsnap.chunks {
                let mut vxl =
                    vxl::parse(bytes).map_err(|source| FromSnapshotError::ChunkParse {
                        grid: *id,
                        chunk: *addr,
                        source,
                    })?;
                let n_cols = (vxl.vsid as usize) * (vxl.vsid as usize);
                vxl.reserve_edit_capacity(n_cols * RESTORE_EDIT_HEADROOM_PER_COLUMN);
                grid.chunks.insert(*addr, vxl);
                grid.mutations = grid.mutations.wrapping_add(1);
            }
            // S7.2: restore per-chunk versions. Pre-S7.2 snapshots
            // carry an empty Vec (via #[serde(default)]) → no
            // bumps applied, every chunk reads as version 0.
            for (addr, ver) in &gsnap.chunk_versions {
                grid.chunk_versions.insert(*addr, *ver);
            }
            scene.grids.insert(*id, grid);
        }
        Ok(scene)
    }
}

#[cfg(test)]
#[allow(clippy::cast_possible_wrap, clippy::type_complexity)]
mod tests {
    use super::*;
    use crate::chunks::tests::voxel_is_solid;
    use crate::CHUNK_SIZE_XY;
    use glam::DVec3;

    impl GridId {
        pub(crate) fn from_raw_for_test(raw: u32) -> Self {
            Self(raw)
        }
    }

    /// A 2-grid scene with ~100 chunks total — the validation
    /// criterion in PORTING-SCENE.md S2. Builds a deterministic
    /// pattern (one voxel per chunk, colour derived from chunk
    /// index) so the round-trip can verify each chunk byte-by-byte
    /// without relying on edit ordering.
    fn build_two_grid_scene() -> (Scene, Vec<(GridId, IVec3, u32, u32, u32, u32)>) {
        // Returns (scene, expected_voxels) where each expected entry
        // is (grid, chunk_idx, voxel_x, voxel_y, voxel_z, color) for
        // post-restore verification.
        let mut scene = Scene::new();
        let g0 = scene.add_grid(GridTransform::at(DVec3::new(0.0, 0.0, 0.0)));
        let g1 = scene.add_grid(GridTransform::at(DVec3::new(1000.0, 0.0, 0.0)));
        let mut expected = Vec::new();
        // Grid 0: 5×5×2 = 50 chunks across (chx, chy, chz) ∈
        // ([0..5], [0..5], [0..2]). One voxel per chunk at local
        // (5, 6, 7) with chunk-derived colour.
        for chz in 0..2 {
            for chy in 0..5 {
                for chx in 0..5 {
                    let chunk_idx = IVec3::new(chx, chy, chz);
                    #[allow(clippy::cast_sign_loss)]
                    let color =
                        0x80_00_00_00 | ((chx as u32) << 16) | ((chy as u32) << 8) | (chz as u32);
                    let global_voxel = chunk_idx
                        * IVec3::new(
                            CHUNK_SIZE_XY as i32,
                            CHUNK_SIZE_XY as i32,
                            crate::CHUNK_SIZE_Z as i32,
                        )
                        + IVec3::new(5, 6, 7);
                    scene
                        .grid_mut(g0)
                        .unwrap()
                        .set_voxel(global_voxel, Some(color));
                    expected.push((g0, chunk_idx, 5, 6, 7, color));
                }
            }
        }
        // Grid 1: 5×5×2 = 50 chunks, similar pattern but offset
        // colour space + different voxel coord.
        for chz in 0..2 {
            for chy in 0..5 {
                for chx in 0..5 {
                    let chunk_idx = IVec3::new(chx, chy, chz);
                    #[allow(clippy::cast_sign_loss)]
                    let color =
                        0x80_ff_00_00 | ((chx as u32) << 16) | ((chy as u32) << 8) | (chz as u32);
                    let global_voxel = chunk_idx
                        * IVec3::new(
                            CHUNK_SIZE_XY as i32,
                            CHUNK_SIZE_XY as i32,
                            crate::CHUNK_SIZE_Z as i32,
                        )
                        + IVec3::new(10, 11, 12);
                    scene
                        .grid_mut(g1)
                        .unwrap()
                        .set_voxel(global_voxel, Some(color));
                    expected.push((g1, chunk_idx, 10, 11, 12, color));
                }
            }
        }
        (scene, expected)
    }

    fn assert_voxels_match(scene: &Scene, expected: &[(GridId, IVec3, u32, u32, u32, u32)]) {
        for &(grid_id, chunk_idx, vx, vy, vz, _color) in expected {
            let grid = scene.grid(grid_id).expect("grid present");
            let chunk = grid.chunk(chunk_idx).expect("chunk present");
            assert!(
                voxel_is_solid(chunk, vx, vy, vz),
                "voxel ({vx},{vy},{vz}) in grid={} chunk={chunk_idx:?} not solid post-restore",
                grid_id.raw()
            );
        }
    }

    #[test]
    fn snapshot_round_trip_preserves_two_grid_100_chunk_scene() {
        let (scene, expected) = build_two_grid_scene();
        assert_eq!(scene.grid_count(), 2);
        let total_chunks: usize = scene.grids().map(|(_, g)| g.chunks.len()).sum();
        assert_eq!(total_chunks, 100, "test setup should produce 100 chunks");

        // Round-trip via in-memory bincode.
        let snap = scene.to_snapshot();
        let bytes = bincode::serialize(&snap).expect("bincode serialize");
        let snap_back: SceneSnapshot = bincode::deserialize(&bytes).expect("bincode deserialize");
        let restored = Scene::from_snapshot(&snap_back).expect("restore");

        // Same shape.
        assert_eq!(restored.grid_count(), 2);
        let total_restored: usize = restored.grids().map(|(_, g)| g.chunks.len()).sum();
        assert_eq!(total_restored, 100);

        // Same voxels.
        assert_voxels_match(&restored, &expected);
    }

    #[test]
    fn snapshot_preserves_next_grid_id_and_transforms() {
        let mut scene = Scene::new();
        let g0 = scene.add_grid(GridTransform::at(DVec3::new(10.0, 20.0, 30.0)));
        let _g1 = scene.add_grid(GridTransform::at(DVec3::new(40.0, 50.0, 60.0)));
        scene.remove_grid(g0); // bumps the gap
        let _g2 = scene.add_grid(GridTransform::at(DVec3::new(70.0, 80.0, 90.0)));
        // next_grid_id should be 3 now (g0=0, g1=1, g2=2).
        let snap = scene.to_snapshot();
        assert_eq!(snap.next_grid_id, 3);

        let restored = Scene::from_snapshot(&snap).expect("restore");
        assert_eq!(restored.grid_count(), 2);
        // A new grid added to the restored scene should get id 3,
        // not reuse the dropped id 0.
        let mut restored_mut = restored;
        let new_id = restored_mut.add_grid(GridTransform::identity());
        assert_eq!(new_id.raw(), 3);
    }

    #[test]
    fn restored_scene_is_editable() {
        // The "/ mutate" half of "round-trip serialize / deserialize
        // / mutate" — verify that a restored scene's chunks have
        // edit capacity reserved so subsequent `set_voxel` doesn't
        // panic.
        let (scene, _) = build_two_grid_scene();
        let snap = scene.to_snapshot();
        let mut restored = Scene::from_snapshot(&snap).expect("restore");

        let g0 = GridId::from_raw_for_test(0);
        let new_voxel = IVec3::new(50, 51, 52);
        restored
            .grid_mut(g0)
            .expect("grid 0 present")
            .set_voxel(new_voxel, Some(0x80_de_ad_be));
        let chunk = restored
            .grid(g0)
            .unwrap()
            .chunk(IVec3::ZERO)
            .expect("chunk created");
        assert!(voxel_is_solid(chunk, 50, 51, 52));
    }

    // ---- S7.2: chunk_versions round-trip ----

    #[test]
    fn snapshot_round_trip_preserves_chunk_versions() {
        // Build a scene whose chunk_versions are non-trivial (multi-
        // edit on the same chunk + edits across multiple chunks),
        // round-trip, and verify every version survives.
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::identity());
        let g = scene.grid_mut(id).unwrap();
        // Three edits on chunk (0,0,0) → version 3.
        g.set_voxel(IVec3::new(0, 0, 0), Some(0x80_aa_bb_cc));
        g.set_voxel(IVec3::new(1, 1, 1), Some(0x80_dd_ee_ff));
        g.set_voxel(IVec3::new(2, 2, 2), Some(0x80_11_22_33));
        // One edit on chunk (1,0,0) → version 1.
        g.set_voxel(IVec3::new(128, 0, 0), Some(0x80_44_55_66));
        assert_eq!(g.chunk_version(IVec3::ZERO), 3);
        assert_eq!(g.chunk_version(IVec3::new(1, 0, 0)), 1);

        let snap = scene.to_snapshot();
        let bytes = bincode::serialize(&snap).expect("bincode serialize");
        let snap_back: SceneSnapshot = bincode::deserialize(&bytes).expect("bincode deserialize");
        let restored = Scene::from_snapshot(&snap_back).expect("restore");

        let g = restored.grid(id).expect("grid present");
        assert_eq!(g.chunk_version(IVec3::ZERO), 3);
        assert_eq!(g.chunk_version(IVec3::new(1, 0, 0)), 1);
        assert_eq!(g.chunk_versions.len(), 2);
    }

    #[test]
    fn snapshot_chunk_versions_zero_entries_are_dropped_from_wire() {
        // Implementation detail worth pinning: we don't waste bytes
        // on chunks whose live counter is 0 (== absent semantically).
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::identity());
        let g = scene.grid_mut(id).unwrap();
        // Manually inject a zero entry — we don't have a public API
        // to do this; reach into chunk_versions to verify the
        // serialise-side filter behaves.
        g.chunk_versions.insert(IVec3::ZERO, 0);
        let snap = scene.to_snapshot();
        let g_snap = &snap.grids[0].1;
        assert!(g_snap.chunk_versions.is_empty(), "zero entries dropped");
    }

    #[test]
    fn snapshot_is_deterministic() {
        let (scene, _) = build_two_grid_scene();
        let s1 = bincode::serialize(&scene.to_snapshot()).unwrap();
        let s2 = bincode::serialize(&scene.to_snapshot()).unwrap();
        assert_eq!(s1, s2, "snapshot bytes should be deterministic");
    }
}
