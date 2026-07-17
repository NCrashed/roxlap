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

use crate::{Grid, GridId, GridTransform, LodThresholds, Scene, StreamRadius};

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

/// One grid's snapshot: transform + flattened chunks + the grid's
/// runtime configuration (QE.5b — pre-QE.5 snapshots restored every
/// config field to its default, so a loaded save silently lost
/// `render_sky` / LOD / streaming settings).
///
/// The `generator` and `store` hooks are host code and cannot be
/// serialised — rebind them after [`Scene::load_snapshot`], keyed on
/// [`name`](Self::name).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridSnapshot {
    /// The grid's world placement (position + rotation).
    pub transform: GridTransform,
    /// Chunks as `(chunk_idx, vxl_bytes)`. `vxl_bytes` is
    /// [`roxlap_formats::vxl::serialize`] output — re-parseable
    /// via [`roxlap_formats::vxl::parse`].
    pub chunks: Vec<(IVec3, Vec<u8>)>,
    /// S7.2: per-chunk edit version counters, sorted by chunk
    /// index. Chunks absent from this list restore as version 0 —
    /// the same as a freshly generated or pre-S7.2 snapshot.
    /// (`#[serde(default)]` covers self-describing formats; the
    /// wire-format compatibility story is the QE.5b envelope
    /// version in [`Scene::save_snapshot`].)
    #[serde(default)]
    pub chunk_versions: Vec<(IVec3, u64)>,
    /// QE.5b — the grid's host-assigned rebind tag ([`crate::Grid::name`]).
    #[serde(default)]
    pub name: Option<String>,
    /// QE.5b — [`crate::Grid::render_sky`].
    #[serde(default = "default_render_sky")]
    pub render_sky: bool,
    /// QE.5b — [`crate::Grid::mip_levels_override`].
    #[serde(default)]
    pub mip_levels_override: Option<u32>,
    /// QE.5b — [`crate::Grid::lod_thresholds`].
    #[serde(default = "LodThresholds::always_near")]
    pub lod_thresholds: LodThresholds,
    /// QE.5b — [`crate::Grid::stream_radius`].
    #[serde(default)]
    pub stream_radius: StreamRadius,
    /// SC.snap (v2) — the grid's [`GridTransform::voxel_world_size`]. Stored
    /// as a sibling field (not inside the transform, whose wire form stays
    /// frozen with `#[serde(skip)]`). The `#[serde(default)]` covers
    /// self-describing formats; the bincode positional gap between v1 and v2
    /// is handled by the private `GridSnapshotV1` shadow shape + the version
    /// dispatch in [`Scene::load_snapshot`]. v1 blobs restore this at `1.0`.
    #[serde(default = "one_f64")]
    pub voxel_world_size: f64,
    /// WT.0 (v3) — the grid's [`crate::WaterVolume`]s. Same
    /// trailing-field discipline as `voxel_world_size`: v1/v2 blobs
    /// decode through their frozen shadow shapes and restore with no
    /// water.
    #[serde(default)]
    pub water_volumes: Vec<crate::WaterVolume>,
    /// CA.0 (v4) — the grid's cutaway clip ([`crate::Grid::z_clip`]).
    /// Same trailing-field discipline: v1..v3 blobs decode through
    /// their frozen shadow shapes and restore unclipped (`None`).
    #[serde(default)]
    pub z_clip: Option<i32>,
}

/// `voxel_world_size`'s default (`1.0`) for `#[serde(default)]` on
/// self-describing formats — an unscaled grid.
fn one_f64() -> f64 {
    1.0
}

/// SC.snap — the **frozen v1** wire shape of [`SceneSnapshot`], used only to
/// deserialize version-1 blobs (which predate the persisted scale). bincode
/// is positional, so this must mirror v1's fields exactly — it has NO
/// `voxel_world_size` (grids restore at `1.0` via [`From`]).
#[derive(Deserialize)]
struct SceneSnapshotV1 {
    next_grid_id: u32,
    grids: Vec<(GridId, GridSnapshotV1)>,
}

/// SC.snap — the frozen v1 [`GridSnapshot`] shape (no `voxel_world_size`).
/// The field list + `#[serde(default)]`s must byte-match what v1
/// `save_snapshot` wrote (the checked-in fixture is the gate).
#[derive(Deserialize)]
struct GridSnapshotV1 {
    transform: GridTransform,
    chunks: Vec<(IVec3, Vec<u8>)>,
    #[serde(default)]
    chunk_versions: Vec<(IVec3, u64)>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default = "default_render_sky")]
    render_sky: bool,
    #[serde(default)]
    mip_levels_override: Option<u32>,
    #[serde(default = "LodThresholds::always_near")]
    lod_thresholds: LodThresholds,
    #[serde(default)]
    stream_radius: StreamRadius,
}

impl From<SceneSnapshotV1> for SceneSnapshot {
    fn from(v1: SceneSnapshotV1) -> Self {
        Self {
            next_grid_id: v1.next_grid_id,
            grids: v1
                .grids
                .into_iter()
                // Chain through the frozen v2 + v3 shapes (see the
                // GridSnapshotV1 → V2 → V3 impls).
                .map(|(id, g)| (id, GridSnapshotV3::from(GridSnapshotV2::from(g)).into()))
                .collect(),
        }
    }
}

impl From<GridSnapshotV1> for GridSnapshotV2 {
    // FROZEN (WT.0): v1 → v2 adds only the persisted scale. Version
    // bumps chain shadow shapes (V1 → V2 → … → live), so old links
    // like this one never change again — only the final
    // shadow-to-live impl is rewritten per bump.
    fn from(g: GridSnapshotV1) -> Self {
        Self {
            transform: g.transform,
            chunks: g.chunks,
            chunk_versions: g.chunk_versions,
            name: g.name,
            render_sky: g.render_sky,
            mip_levels_override: g.mip_levels_override,
            lod_thresholds: g.lod_thresholds,
            stream_radius: g.stream_radius,
            // v1 predates persisted scale — an unscaled grid.
            voxel_world_size: 1.0,
        }
    }
}

/// WT.0 — the **frozen v2** wire shape of [`SceneSnapshot`], used only to
/// deserialize version-2 blobs (which predate water volumes). Same
/// positional-bincode rationale as [`SceneSnapshotV1`].
#[derive(Deserialize)]
struct SceneSnapshotV2 {
    next_grid_id: u32,
    grids: Vec<(GridId, GridSnapshotV2)>,
}

/// WT.0 — the frozen v2 [`GridSnapshot`] shape (no `water_volumes`).
/// The field list must byte-match what v2 `save_snapshot` wrote (the
/// checked-in v2 fixture is the gate).
#[derive(Deserialize)]
struct GridSnapshotV2 {
    transform: GridTransform,
    chunks: Vec<(IVec3, Vec<u8>)>,
    #[serde(default)]
    chunk_versions: Vec<(IVec3, u64)>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default = "default_render_sky")]
    render_sky: bool,
    #[serde(default)]
    mip_levels_override: Option<u32>,
    #[serde(default = "LodThresholds::always_near")]
    lod_thresholds: LodThresholds,
    #[serde(default)]
    stream_radius: StreamRadius,
    #[serde(default = "one_f64")]
    voxel_world_size: f64,
}

impl From<SceneSnapshotV2> for SceneSnapshot {
    fn from(v2: SceneSnapshotV2) -> Self {
        Self {
            next_grid_id: v2.next_grid_id,
            grids: v2
                .grids
                .into_iter()
                .map(|(id, g)| (id, GridSnapshotV3::from(g).into()))
                .collect(),
        }
    }
}

impl From<GridSnapshotV2> for GridSnapshotV3 {
    // FROZEN (CA.0): v2 → v3 adds only the water volumes. Version
    // bumps chain shadow shapes (V1 → V2 → … → live), so old links
    // like this one never change again — only the final
    // shadow-to-live impl is rewritten per bump.
    fn from(g: GridSnapshotV2) -> Self {
        Self {
            transform: g.transform,
            chunks: g.chunks,
            chunk_versions: g.chunk_versions,
            name: g.name,
            render_sky: g.render_sky,
            mip_levels_override: g.mip_levels_override,
            lod_thresholds: g.lod_thresholds,
            stream_radius: g.stream_radius,
            voxel_world_size: g.voxel_world_size,
            // v2 predates water volumes — a dry grid.
            water_volumes: Vec::new(),
        }
    }
}

/// CA.0 — the **frozen v3** wire shape of [`SceneSnapshot`], used only to
/// deserialize version-3 blobs (which predate the cutaway clip). Same
/// positional-bincode rationale as [`SceneSnapshotV1`].
#[derive(Deserialize)]
struct SceneSnapshotV3 {
    next_grid_id: u32,
    grids: Vec<(GridId, GridSnapshotV3)>,
}

/// CA.0 — the frozen v3 [`GridSnapshot`] shape (no `z_clip`). The field
/// list must byte-match what v3 `save_snapshot` wrote (the checked-in
/// v3 fixture is the gate).
#[derive(Deserialize)]
struct GridSnapshotV3 {
    transform: GridTransform,
    chunks: Vec<(IVec3, Vec<u8>)>,
    #[serde(default)]
    chunk_versions: Vec<(IVec3, u64)>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default = "default_render_sky")]
    render_sky: bool,
    #[serde(default)]
    mip_levels_override: Option<u32>,
    #[serde(default = "LodThresholds::always_near")]
    lod_thresholds: LodThresholds,
    #[serde(default)]
    stream_radius: StreamRadius,
    #[serde(default = "one_f64")]
    voxel_world_size: f64,
    #[serde(default)]
    water_volumes: Vec<crate::WaterVolume>,
}

impl From<SceneSnapshotV3> for SceneSnapshot {
    fn from(v3: SceneSnapshotV3) -> Self {
        Self {
            next_grid_id: v3.next_grid_id,
            grids: v3.grids.into_iter().map(|(id, g)| (id, g.into())).collect(),
        }
    }
}

impl From<GridSnapshotV3> for GridSnapshot {
    fn from(g: GridSnapshotV3) -> Self {
        Self {
            transform: g.transform,
            chunks: g.chunks,
            chunk_versions: g.chunk_versions,
            name: g.name,
            render_sky: g.render_sky,
            mip_levels_override: g.mip_levels_override,
            lod_thresholds: g.lod_thresholds,
            stream_radius: g.stream_radius,
            voxel_world_size: g.voxel_world_size,
            water_volumes: g.water_volumes,
            // v3 predates the cutaway clip — an unclipped grid.
            z_clip: None,
        }
    }
}

/// `render_sky`'s [`crate::Grid::new`] default, for
/// `#[serde(default)]` on self-describing formats.
fn default_render_sky() -> bool {
    true
}

/// Errors from [`Scene::from_snapshot`]. Wraps the per-chunk
/// [`ParseError`] with a tag identifying which grid + chunk
/// failed.
#[derive(Debug)]
pub enum FromSnapshotError {
    /// One chunk's bytes failed to round-trip through
    /// [`roxlap_formats::vxl::parse`].
    ChunkParse {
        /// The grid whose chunk failed to parse.
        grid: GridId,
        /// The failing chunk's index within `grid`.
        chunk: IVec3,
        /// The underlying `.vxl` parse failure.
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
        // FW.1 — a presentation-only twin is DERIVED visual state
        // (rebuilt from the real grid + a `FogOfWar`), not authoritative
        // scene content: exclude it (`query_grids` shares the predicate
        // with the rest of the crate). This keeps the wire format
        // unchanged (no new fields) — after a load the host re-arms
        // fog-of-war, which re-creates the twin. The real grid's
        // `render_excluded` flag is likewise transient and defaults off
        // on load (fog is off until re-armed — [`crate::FowTwin::sync`]
        // signals a lost twin so a host can detect the re-arm).
        let mut grid_ids: Vec<GridId> = self.query_grids().map(|(id, _)| id).collect();
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
                    name: grid.name.clone(),
                    render_sky: grid.render_sky,
                    mip_levels_override: grid.mip_levels_override,
                    lod_thresholds: grid.lod_thresholds,
                    stream_radius: grid.stream_radius,
                    // SC.snap — persist the grid's scale (transform's own wire
                    // form omits it via #[serde(skip)]).
                    voxel_world_size: grid.transform.voxel_world_size,
                    // WT.0 (v3) — persist the water volumes verbatim
                    // (host-authored Vec order is already deterministic).
                    water_volumes: grid.water_volumes.clone(),
                    // CA.0 (v4) — persist the cutaway clip.
                    z_clip: grid.z_clip,
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
            // SC.snap — `transform`'s wire form omits `voxel_world_size`
            // (`#[serde(skip)]`); the persisted value rides in the sibling
            // `gsnap.voxel_world_size`. Fold it back into the transform.
            // Guard untrusted bytes to a sane range: not just ≤0 / non-finite
            // (which break the marcher — GPU `chunk_dim = 0` → NaN) but also
            // subnormal-near-zero (1e-300 > 0 yet `vsid · 1e-300 ≈ 0`) and
            // absurdly-huge finite values. `[1e-6, 1e6]` spans any sane
            // planet↔grain ratio with margin; out-of-range → fall back to 1.0.
            let mut transform = gsnap.transform;
            let vws = gsnap.voxel_world_size;
            transform.voxel_world_size = if vws.is_finite() && (1e-6..=1e6).contains(&vws) {
                vws
            } else {
                log::warn!(
                    "load_snapshot: grid {id:?} has out-of-range \
                         voxel_world_size {vws} — restoring at 1.0"
                );
                1.0
            };
            let mut grid = Grid::new(transform);
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
                grid.note_chunk_set_changed();
            }
            // S7.2: restore per-chunk versions. Pre-S7.2 snapshots
            // carry an empty Vec (via #[serde(default)]) → no
            // bumps applied, every chunk reads as version 0.
            for (addr, ver) in &gsnap.chunk_versions {
                grid.restore_chunk_version(*addr, *ver);
            }
            // QE.5b — restore the grid's runtime configuration (the
            // `generator` / `store` hooks are host code: rebind them
            // after loading, keyed on `name`).
            grid.name.clone_from(&gsnap.name);
            grid.render_sky = gsnap.render_sky;
            grid.mip_levels_override = gsnap.mip_levels_override;
            grid.lod_thresholds = gsnap.lod_thresholds;
            grid.stream_radius = gsnap.stream_radius;
            // WT.0 — restore the water volumes VERBATIM. No corner
            // re-normalisation: `WaterVolume::depth_local` accepts
            // either corner order, so normalising here would make a
            // volume behave differently before and after a round-trip
            // (a live scene must equal its restored self).
            grid.water_volumes.clone_from(&gsnap.water_volumes);
            // CA.0 — restore the cutaway clip (pre-v4 saves carry
            // `None` via the shadow-shape chain → unclipped).
            grid.z_clip = gsnap.z_clip;
            scene.grids.insert(*id, grid);
        }
        Ok(scene)
    }
}

/// The QE.5b wire envelope's magic — identifies a byte blob as a
/// roxlap scene snapshot before any deserialisation runs.
pub const SNAPSHOT_MAGIC: [u8; 4] = *b"RXSS";

/// Current snapshot wire version. Bumped whenever the
/// [`SceneSnapshot`] payload shape changes in a way `#[serde(default)]`
/// trailing fields can't cover; [`Scene::load_snapshot`] dispatches on
/// it, so an old save either loads correctly or fails with
/// [`SnapshotLoadError::UnsupportedVersion`] — never a silent misparse
/// (the pre-QE.5 failure mode of bare positional bincode).
///
/// CA.0 — **v4** appends per-grid `z_clip` (v3 = WT.0's
/// `water_volumes`; v2 = SC.snap's `voxel_world_size`; v1 = the
/// original shape). bincode is strictly positional, so every
/// trailing-field addition bumps the version and freezes the previous
/// shape as a private shadow struct: [`Scene::load_snapshot`]
/// dispatches — v4 blobs decode [`GridSnapshot`] directly; v3/v2/v1
/// blobs decode `GridSnapshotV3` / `GridSnapshotV2` / `GridSnapshotV1`
/// and restore unclipped (and, for v2/v1, with no water; for v1,
/// `voxel_world_size = 1.0`). `GridTransform`'s own wire form stays
/// frozen (`#[serde(skip)]`) in ALL versions, so the forever-loadable
/// v1/v2/v3 fixtures keep loading unchanged.
pub const SNAPSHOT_VERSION: u32 = 4;

/// Errors from [`Scene::load_snapshot`].
#[derive(Debug)]
pub enum SnapshotLoadError {
    /// The bytes don't start with [`SNAPSHOT_MAGIC`] — not a roxlap
    /// snapshot (or a pre-QE.5 bare-bincode save; see the
    /// [`Scene::load_snapshot`] migration note).
    BadMagic,
    /// A snapshot from a newer (or unknown) wire version.
    UnsupportedVersion(u32),
    /// The payload failed to decode (truncated / corrupted bytes).
    Decode(String),
    /// The payload decoded but a chunk failed to restore.
    Restore(FromSnapshotError),
}

impl std::fmt::Display for SnapshotLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadMagic => write!(f, "not a roxlap scene snapshot (bad magic)"),
            Self::UnsupportedVersion(v) => {
                write!(
                    f,
                    "unsupported snapshot version {v} (this build reads <= {SNAPSHOT_VERSION})"
                )
            }
            Self::Decode(msg) => write!(f, "snapshot payload decode failed: {msg}"),
            Self::Restore(e) => write!(f, "snapshot restore failed: {e}"),
        }
    }
}

impl std::error::Error for SnapshotLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Restore(e) => Some(e),
            _ => None,
        }
    }
}

impl Scene {
    /// Serialise the scene to the versioned snapshot **wire format**
    /// (QE.5b): [`SNAPSHOT_MAGIC`] + little-endian
    /// [`SNAPSHOT_VERSION`] + a bincode [`SceneSnapshot`] payload.
    /// This is the save-file API — a blob written today stays
    /// loadable by future engine versions ([`Self::load_snapshot`]
    /// dispatches on the version), which bare serde values can't
    /// promise under positional codecs.
    ///
    /// Hosts that want a different payload codec (JSON for debugging,
    /// postcard for size) can still serialise
    /// [`Self::to_snapshot`]'s value themselves — and then own their
    /// format's evolution story.
    #[must_use]
    pub fn save_snapshot(&self) -> Vec<u8> {
        // SC.snap — v2 persists per-grid `voxel_world_size` (see
        // [`SNAPSHOT_VERSION`]); no scale is lost across a round-trip.
        let mut out = Vec::with_capacity(64);
        out.extend_from_slice(&SNAPSHOT_MAGIC);
        out.extend_from_slice(&SNAPSHOT_VERSION.to_le_bytes());
        bincode::serialize_into(&mut out, &self.to_snapshot())
            .expect("bincode into Vec<u8> cannot fail");
        out
    }

    /// Load a scene from [`Self::save_snapshot`] bytes, dispatching
    /// on the embedded wire version.
    ///
    /// Migration note: saves written **before** QE.5b (bare bincode
    /// of a [`SceneSnapshot`], no envelope) fail with
    /// [`SnapshotLoadError::BadMagic`]; decode those with the engine
    /// version that wrote them and re-save.
    ///
    /// # Errors
    /// [`SnapshotLoadError`] — bad magic, unknown version, corrupted
    /// payload, or a failing chunk restore.
    pub fn load_snapshot(bytes: &[u8]) -> Result<Self, SnapshotLoadError> {
        let (Some(magic), Some(version)) = (bytes.get(..4), bytes.get(4..8)) else {
            return Err(SnapshotLoadError::BadMagic);
        };
        if magic != SNAPSHOT_MAGIC {
            return Err(SnapshotLoadError::BadMagic);
        }
        let version = u32::from_le_bytes(version.try_into().expect("4-byte slice"));
        let payload = &bytes[8..];
        // Dispatch on the wire version: older blobs decode their frozen
        // shadow shapes (v1: no scale, no water → 1.0 + dry; v2 (SC.snap):
        // no water → dry; v3 (WT.0): no clip → unclipped); v4 blobs decode
        // `SceneSnapshot` directly. All funnel through `from_snapshot`.
        let snap: SceneSnapshot = match version {
            1 => bincode::deserialize::<SceneSnapshotV1>(payload)
                .map_err(|e| SnapshotLoadError::Decode(e.to_string()))?
                .into(),
            2 => bincode::deserialize::<SceneSnapshotV2>(payload)
                .map_err(|e| SnapshotLoadError::Decode(e.to_string()))?
                .into(),
            3 => bincode::deserialize::<SceneSnapshotV3>(payload)
                .map_err(|e| SnapshotLoadError::Decode(e.to_string()))?
                .into(),
            4 => bincode::deserialize(payload)
                .map_err(|e| SnapshotLoadError::Decode(e.to_string()))?,
            v => return Err(SnapshotLoadError::UnsupportedVersion(v)),
        };
        Self::from_snapshot(&snap).map_err(SnapshotLoadError::Restore)
    }
}

#[cfg(test)]
#[allow(clippy::cast_possible_wrap, clippy::type_complexity)]
mod tests {
    use super::*;
    use crate::chunks::tests::voxel_is_solid;
    use crate::CHUNK_SIZE_XY;
    use glam::DVec3;
    use roxlap_formats::color::VoxColor;

    impl GridId {
        pub(crate) fn from_raw_for_test(raw: u32) -> Self {
            Self(raw)
        }
    }

    #[test]
    fn sc_snap_scaled_grid_survives_round_trip() {
        // SC.snap — v2 persists per-grid voxel_world_size. A vws=0.25 grid must
        // restore at 0.25, not the pre-SC.snap 1.0 (transform's own wire form
        // still omits it via #[serde(skip)] — the value rides the sibling
        // GridSnapshot field).
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::at_scale(DVec3::new(5.0, -3.0, 0.0), 0.25));
        scene
            .grid_mut(id)
            .unwrap()
            .set_voxel(IVec3::new(1, 2, 100), Some(VoxColor(0x8011_2233)));

        let bytes = scene.save_snapshot();
        // LITERAL on purpose (WT.0 review): this is the one assert that
        // pins the envelope version as a NUMBER. Comparing against
        // SNAPSHOT_VERSION would be a tautology (save_snapshot writes
        // that same constant) and an accidental bump would sail through
        // green while old engines stop reading new saves. Bumping the
        // version deliberately? Update this literal alongside the new
        // shadow shape + fixture.
        assert_eq!(&bytes[4..8], &4u32.to_le_bytes(), "expected v4 wire");
        let restored = Scene::load_snapshot(&bytes).expect("round trip");

        let (_, g) = restored.grids().next().expect("one grid");
        assert!(
            (g.transform.voxel_world_size - 0.25).abs() < 1e-12,
            "scale must survive: got {}",
            g.transform.voxel_world_size
        );
        assert_eq!(g.transform.origin, DVec3::new(5.0, -3.0, 0.0));
        assert!(g.voxel_solid(IVec3::new(1, 2, 100)));
    }

    #[test]
    fn sc_snap_out_of_range_scale_restores_at_one() {
        // The load-side guard rejects corrupt/absurd persisted scale — not
        // just ≤0 / non-finite but subnormal-near-zero and huge finite values,
        // any of which collapse the marcher's chunk_dim toward 0 (→ NaN on
        // GPU). All fall back to 1.0.
        for bad in [0.0, -2.0, f64::NAN, f64::INFINITY, 1e-300, 1e300] {
            let snap = SceneSnapshot {
                next_grid_id: 1,
                grids: vec![(
                    GridId::from_raw_for_test(0),
                    GridSnapshot {
                        transform: GridTransform::identity(),
                        chunks: vec![],
                        chunk_versions: vec![],
                        name: None,
                        render_sky: true,
                        mip_levels_override: None,
                        lod_thresholds: LodThresholds::always_near(),
                        stream_radius: StreamRadius::default(),
                        voxel_world_size: bad,
                        water_volumes: vec![],
                        z_clip: None,
                    },
                )],
            };
            let scene = Scene::from_snapshot(&snap).expect("restore");
            let (_, g) = scene.grids().next().expect("one grid");
            assert_eq!(
                g.transform.voxel_world_size, 1.0,
                "out-of-range vws {bad} must restore at 1.0"
            );
        }
        // A sane extreme within [1e-6, 1e6] is preserved verbatim.
        let ok = SceneSnapshot {
            next_grid_id: 1,
            grids: vec![(
                GridId::from_raw_for_test(0),
                GridSnapshot {
                    transform: GridTransform::identity(),
                    chunks: vec![],
                    chunk_versions: vec![],
                    name: None,
                    render_sky: true,
                    mip_levels_override: None,
                    lod_thresholds: LodThresholds::always_near(),
                    stream_radius: StreamRadius::default(),
                    voxel_world_size: 1000.0,
                    water_volumes: vec![],
                    z_clip: None,
                },
            )],
        };
        let scene = Scene::from_snapshot(&ok).expect("restore");
        assert_eq!(
            scene.grids().next().unwrap().1.transform.voxel_world_size,
            1000.0
        );
    }

    /// A 2-grid scene with ~100 chunks total — the validation
    /// criterion in PORTING-SCENE.md S2. Builds a deterministic
    /// pattern (one voxel per chunk, colour derived from chunk
    /// index) so the round-trip can verify each chunk byte-by-byte
    /// without relying on edit ordering.
    fn build_two_grid_scene() -> (Scene, Vec<(GridId, IVec3, u32, u32, u32, VoxColor)>) {
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
                    let color = VoxColor(
                        0x80_00_00_00 | ((chx as u32) << 16) | ((chy as u32) << 8) | (chz as u32),
                    );
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
                    let color = VoxColor(
                        0x80_ff_00_00 | ((chx as u32) << 16) | ((chy as u32) << 8) | (chz as u32),
                    );
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

    fn assert_voxels_match(scene: &Scene, expected: &[(GridId, IVec3, u32, u32, u32, VoxColor)]) {
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
            .set_voxel(new_voxel, Some(VoxColor(0x80_de_ad_be)));
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
        g.set_voxel(IVec3::new(0, 0, 0), Some(VoxColor(0x80_aa_bb_cc)));
        g.set_voxel(IVec3::new(1, 1, 1), Some(VoxColor(0x80_dd_ee_ff)));
        g.set_voxel(IVec3::new(2, 2, 2), Some(VoxColor(0x80_11_22_33)));
        // One edit on chunk (1,0,0) → version 1.
        g.set_voxel(IVec3::new(128, 0, 0), Some(VoxColor(0x80_44_55_66)));
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
