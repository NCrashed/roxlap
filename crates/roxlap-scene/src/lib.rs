//! roxlap scene-graph layer — many independent chunked voxel
//! grids in a single 3D scene.
//!
//! New to roxlap? **[The roxlap book](https://ncrashed.github.io/roxlap/)**
//! is the guide — its scene-graph chapter walks this crate end to end;
//! this page is the API reference.
//!
//! This crate is the layer **above** the per-chunk renderer
//! (`roxlap-core`): a [`Scene`] holds a sparse set of [`Grid`]s, each
//! with its own f64 world position + arbitrary 3D rotation
//! ([`GridTransform`]). Around that core sit:
//!
//! - [`addr`] — world ↔ grid-local ↔ chunk + voxel-in-chunk
//!   decomposition, the canonical f64↔i32 boundary helpers;
//! - [`chunks`] — sparse chunk storage with on-demand materialisation,
//!   plus the [`Grid`] edit API ([`Grid::set_voxel`],
//!   [`Grid::set_rect`], [`Grid::set_sphere`], …) which decomposes
//!   multi-chunk operations and delegates to [`roxlap_formats::edit`];
//! - [`render`] — multi-grid raycast composition for the CPU renderer;
//! - [`snapshot`] — save/load: the versioned wire format
//!   ([`Scene::save_snapshot`] / [`Scene::load_snapshot`], QE.5b)
//!   plus the underlying serde-friendly [`snapshot::SceneSnapshot`]
//!   value (chunks encode via [`roxlap_formats::vxl::serialize`] /
//!   [`parse`]);
//! - [`streaming`] — chunk streaming + procedural generation
//!   ([`streaming::ChunkGenerator`], radius-driven install/evict,
//!   async generation on a rayon pool) and persistence for edited
//!   chunks ([`streaming::ChunkStore`], QE.5a);
//! - [`lod`] / [`billboard`] / [`occluder`] — far-LOD billboards and
//!   render culling helpers;
//! - world queries — [`Scene::raycast`], [`Scene::resolve_voxel`],
//!   [`Grid::voxel_solid`] / [`Grid::voxel_color`].
//!
//! `docs/porting/PORTING-SCENE.md` in the repository records the
//! original substage roadmap (S1..S7, all landed).
//!
//! [`parse`]: roxlap_formats::vxl::parse

pub mod addr;
pub mod billboard;
pub mod cavegen;
/// Character controller (stage CC) — a walking body over the scene;
/// see `docs/porting/PORTING-CONTROLLER.md`.
pub mod character;
pub mod chunks;
/// Collision query layer (stage CC) — box-vs-voxel overlap over a
/// scene; see `docs/porting/PORTING-CONTROLLER.md`.
pub mod collide;
pub mod edit;
pub mod lod;
pub mod occluder;
pub mod render;
pub mod snapshot;
pub mod streaming;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use glam::{DQuat, DVec3, IVec3, UVec3};
use roxlap_formats::vxl::Vxl;
use serde::{Deserialize, Serialize};

pub use addr::{grid_local_to_world, voxel_global, voxel_split, world_to_grid_local, GridLocalPos};
pub use billboard::{canonical_viewpoints, BillboardCache, BillboardSnapshot};
pub use character::{CharacterBody, CharacterDef, MoveMode, WalkInput};
pub use chunks::{BakeLight, BakeMode};
pub use collide::{box_overlaps_solid, grid_box_overlaps_solid, point_overlaps_solid, Solidity};
pub use edit::SpanOp;
pub use lod::{select_lod, Lod, LodThresholds};
pub use roxlap_core::AoParams;
pub use roxlap_formats::color::{OverlayColor, Rgb, VoxColor};
pub use streaming::{ChunkGenerator, ChunkStore, StreamRadius};

/// XY size of one chunk in voxels. The plan locks 128 — keeps
/// chunks compact (~2 MB worst-case dense-slab footprint inside
/// each `Vxl`) and divides cleanly into voxlap's 2048 reference
/// world size.
pub const CHUNK_SIZE_XY: u32 = 128;

/// Z size of one chunk in voxels. Locked at 256 to preserve
/// voxlap's existing slab byte format unchanged inside each chunk
/// — the per-chunk renderer doesn't need to know it's living
/// inside a scene-graph.
pub const CHUNK_SIZE_Z: u32 = 256;

/// Stable identifier for a grid registered in a [`Scene`]. Issued
/// by [`Scene::add_grid`]; persists across edits but a removed
/// grid's id is not reissued.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct GridId(u32);

impl GridId {
    /// The integer wire form. Useful for serde / debug output.
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

/// A solid-voxel hit from [`Scene::raycast`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayHit {
    /// The grid the ray hit.
    pub grid: GridId,
    /// Grid-local integer voxel coordinate of the hit cell.
    pub voxel: IVec3,
    /// World-space hit point (`origin + t · normalize(dir)`).
    pub world: DVec3,
    /// World distance from the ray origin to the hit.
    pub t: f64,
    /// Packed colour of the hit voxel, or `None` if it's an untextured
    /// (bedrock / interior) cell. See [`Grid::voxel_color`].
    pub color: Option<VoxColor>,
}

/// Ray/AABB slab intersection in f64 (`[lo, hi)` box). Returns the
/// entry/exit ray parameters, or `None` on a miss. PF.6 helper for the
/// raycast pre-clip + absent-chunk exit.
fn ray_box(o: DVec3, d: DVec3, blo: DVec3, bhi: DVec3) -> Option<(f64, f64)> {
    let mut tmin = f64::NEG_INFINITY;
    let mut tmax = f64::INFINITY;
    for a in 0..3 {
        if d[a].abs() < 1e-12 {
            if o[a] < blo[a] || o[a] > bhi[a] {
                return None;
            }
        } else {
            let inv = 1.0 / d[a];
            let (t0, t1) = ((blo[a] - o[a]) * inv, (bhi[a] - o[a]) * inv);
            tmin = tmin.max(t0.min(t1));
            tmax = tmax.min(t0.max(t1));
            if tmin > tmax {
                return None;
            }
        }
    }
    Some((tmin, tmax))
}

/// The populated chunk extent of `grid` as a grid-local voxel-space AABB
/// `[lo, hi)`, or `None` for an empty grid. O(chunks) HashMap key walk —
/// fine for per-call raycast use.
fn grid_voxel_aabb_f64(grid: &Grid) -> Option<(DVec3, DVec3)> {
    let mut min = IVec3::splat(i32::MAX);
    let mut max = IVec3::splat(i32::MIN);
    let mut any = false;
    for idx in grid.chunks.keys() {
        any = true;
        min = min.min(*idx);
        max = max.max(*idx);
    }
    if !any {
        return None;
    }
    let (cs_xy, cs_z) = (
        i64::from(CHUNK_SIZE_XY as i32),
        i64::from(CHUNK_SIZE_Z as i32),
    );
    #[allow(clippy::cast_precision_loss)]
    let v = |i: IVec3, add: i64| {
        DVec3::new(
            ((i64::from(i.x) + add) * cs_xy) as f64,
            ((i64::from(i.y) + add) * cs_xy) as f64,
            ((i64::from(i.z) + add) * cs_z) as f64,
        )
    };
    Some((v(min, 0), v(max, 1)))
}

/// Voxel DDA (Amanatides-Woo) in a grid's local space. `lo` / `ld` are
/// the ray origin + unit direction already transformed into grid-local
/// coords. Returns the first [`Grid::voxel_solid`] cell and its world-
/// equal distance `t`, or `None` past `max_t`.
///
/// PF.6 — three upgrades over the naive from-origin march:
/// - the ray is pre-clipped to the grid's populated chunk AABB (a miss
///   costs one slab test; a distant grid is marched AT the box, not from
///   the origin);
/// - chunk lookups go through the chunk-cached [`SolidSampler`]-style
///   probe (one HashMap hit per chunk crossing) and the solid test walks
///   the slab chain in place (no per-step allocation);
/// - a voxel in an absent (all-air) chunk fast-forwards the march to
///   that chunk's exit face instead of stepping its up-to-128 voxels.
#[allow(clippy::cast_possible_truncation)]
fn voxel_dda(grid: &Grid, lo: DVec3, ld: DVec3, max_t: f64) -> Option<(IVec3, f64)> {
    // Clip to the populated chunk AABB: everything outside is air.
    let (blo, bhi) = grid_voxel_aabb_f64(grid)?;
    let (t0, t1) = ray_box(lo, ld, blo, bhi)?;
    let t_enter = t0.max(0.0);
    let t_exit = t1.min(max_t);
    if t_enter > t_exit {
        return None;
    }

    let step = IVec3::new(
        i32::from(ld.x > 0.0) - i32::from(ld.x < 0.0),
        i32::from(ld.y > 0.0) - i32::from(ld.y < 0.0),
        i32::from(ld.z > 0.0) - i32::from(ld.z < 0.0),
    );
    // Distance to advance one whole voxel along each axis (∞ if parallel).
    let inv_abs = |d: f64| {
        if d == 0.0 {
            f64::INFINITY
        } else {
            (1.0 / d).abs()
        }
    };
    let t_delta = DVec3::new(inv_abs(ld.x), inv_abs(ld.y), inv_abs(ld.z));
    // Absolute-`t` of the next voxel boundary from cell `p`, per axis
    // (also the re-seed after an absent-chunk jump).
    let seed_t_max = |p: IVec3| -> DVec3 {
        let axis = |pa: i32, oa: f64, da: f64| -> f64 {
            if da > 0.0 {
                (f64::from(pa) + 1.0 - oa) / da
            } else if da < 0.0 {
                (f64::from(pa) - oa) / da
            } else {
                f64::INFINITY
            }
        };
        DVec3::new(
            axis(p.x, lo.x, ld.x),
            axis(p.y, lo.y, ld.y),
            axis(p.z, lo.z, ld.z),
        )
    };

    // Start at the AABB entry (t stays measured from the true origin;
    // t_enter == 0 ⇒ the original from-origin start, bit-identical).
    let start = lo + ld * t_enter;
    let mut p = IVec3::new(
        start.x.floor() as i32,
        start.y.floor() as i32,
        start.z.floor() as i32,
    );
    let mut t_max = seed_t_max(p);
    let mut t_curr = t_enter;
    let mut sampler = grid.solid_sampler();

    #[allow(clippy::cast_sign_loss)]
    let max_steps = (max_t * 3.0) as u64 + 8;
    let (cs_xy, cs_z) = (
        f64::from(CHUNK_SIZE_XY as i32),
        f64::from(CHUNK_SIZE_Z as i32),
    );
    for _ in 0..max_steps {
        let (chunk_idx, in_chunk) = voxel_split(p);
        if let Some(vxl) = sampler.chunk_at(chunk_idx) {
            if chunks::vxl_voxel_solid(vxl, in_chunk.x, in_chunk.y, in_chunk.z) {
                return Some((p, t_curr));
            }
            // Advance across the nearest voxel boundary.
            let t = if t_max.x <= t_max.y && t_max.x <= t_max.z {
                p.x += step.x;
                let t = t_max.x;
                t_max.x += t_delta.x;
                t
            } else if t_max.y <= t_max.z {
                p.y += step.y;
                let t = t_max.y;
                t_max.y += t_delta.y;
                t
            } else {
                p.z += step.z;
                let t = t_max.z;
                t_max.z += t_delta.z;
                t
            };
            if t > t_exit {
                return None;
            }
            t_curr = t;
        } else {
            // Absent chunk ⇒ guaranteed air: jump to its exit face.
            let clo = DVec3::new(
                f64::from(chunk_idx.x) * cs_xy,
                f64::from(chunk_idx.y) * cs_xy,
                f64::from(chunk_idx.z) * cs_z,
            );
            let chi = clo + DVec3::new(cs_xy, cs_xy, cs_z);
            let exit = match ray_box(lo, ld, clo, chi) {
                // Nudge past the face so the floor lands in the next chunk.
                Some((_, t1)) => t1.max(t_curr) + 1e-4,
                None => return None, // degenerate (shouldn't happen: p is inside)
            };
            if exit > t_exit {
                return None;
            }
            let q = lo + ld * exit;
            p = IVec3::new(q.x.floor() as i32, q.y.floor() as i32, q.z.floor() as i32);
            t_max = seed_t_max(p);
            t_curr = exit;
        }
    }
    None
}

/// f64 world placement of one grid: position + orientation.
///
/// `origin` is the grid's local-space origin in world coords —
/// chunk `(0, 0, 0)`'s `(0, 0, 0)` voxel maps to
/// `origin + rotation * vec3(0, 0, 0)` (i.e. just `origin`).
/// Voxel size is fixed at 1 world unit / voxel for v1.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GridTransform {
    /// The grid's local-space origin, in world coordinates.
    pub origin: DVec3,
    /// The grid's orientation about `origin`.
    pub rotation: DQuat,
}

impl GridTransform {
    /// Identity transform at world origin. Useful as a default for
    /// the first grid added to an otherwise empty scene.
    #[must_use]
    pub fn identity() -> Self {
        Self {
            origin: DVec3::ZERO,
            rotation: DQuat::IDENTITY,
        }
    }

    /// Axis-aligned grid placed at `origin` with no rotation.
    #[must_use]
    pub fn at(origin: DVec3) -> Self {
        Self {
            origin,
            rotation: DQuat::IDENTITY,
        }
    }
}

impl Default for GridTransform {
    fn default() -> Self {
        Self::identity()
    }
}

/// Address of one voxel inside a scene: which grid it belongs to,
/// which chunk within that grid, and the voxel's offset inside
/// that chunk.
///
/// `chunk` is signed (`IVec3`) because chunks are centred on the
/// grid's local origin and may extend in either direction. `voxel`
/// is unsigned and must satisfy
/// `(voxel.x, voxel.y) < CHUNK_SIZE_XY` and `voxel.z < CHUNK_SIZE_Z`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GridAddr {
    /// The owning grid.
    pub grid: GridId,
    /// Signed chunk index within the grid.
    pub chunk: IVec3,
    /// Voxel offset inside `chunk` (see the bounds above).
    pub voxel: UVec3,
}

/// One independent voxel grid in a scene. Holds its world placement
/// and a sparse map of populated chunks. Empty chunk slots are
/// implicit air and skipped during rendering / raycasts.
///
/// Each chunk is internally a [`Vxl`] with `vsid = CHUNK_SIZE_XY`
/// — the existing per-chunk renderer (opticast + grouscan +
/// sprites + lighting in `roxlap-core`) runs on each chunk
/// unchanged. Vertical worlds are built by stacking chunks along
/// grid-local `+z`.
#[derive(Debug)]
pub struct Grid {
    /// World placement (origin + rotation).
    pub transform: GridTransform,
    /// Sparse chunk storage keyed by `(chx, chy, chz)` chunk
    /// coordinates. A missing entry means the chunk is fully air.
    pub chunks: HashMap<IVec3, Vxl>,
    /// Whether sky pixels rendered for this grid should be
    /// composited into the final framebuffer. `true` is the
    /// historical "grid owns its own sky" behaviour: ray misses
    /// inside this grid's frustum paint sky_color into the temp
    /// buffer. Set `false` for grids that are a foreground object
    /// (e.g. a ship) — the sky is owned by a single "world" grid
    /// (the ground) and other grids should not contribute sky
    /// pixels, otherwise their grid-local-frame sky lookup
    /// rotates with the grid and visibly fights the world's sky
    /// during compose. See [`crate::render::render_scene_composed`]
    /// for the masking implementation.
    pub render_sky: bool,
    /// Override [`roxlap_core::opticast::OpticastSettings::mip_levels`]
    /// for this grid. `None` ⇒ use the caller's value. `Some(n)`
    /// ⇒ cap at `n` (clamped to `[1, settings.mip_levels]`). Use
    /// to disable multi-mip on a per-grid basis — small grids
    /// (rotating ships, billboards) don't benefit from deep mips
    /// and CAN trigger the
    /// `[[project_axis_aligned_mip_beams]]`-style cf-cancellation
    /// artifact when near-axis-aligned rays hit the rotated grid.
    /// `Some(1)` = mip-0 only, byte-stable to single-mip.
    pub mip_levels_override: Option<u32>,
    /// World-distance thresholds for per-grid LOD tier selection
    /// (S6.0). Defaults to [`LodThresholds::always_near`], so a
    /// freshly-constructed grid always renders at full voxel (the
    /// S5-and-earlier byte-stable behaviour). S6.1 plugs `Mid` into
    /// the existing multi-mip path; S6.3 plugs `Far` into the
    /// billboard impostor cache. See [`crate::lod`].
    pub lod_thresholds: LodThresholds,
    /// Lazy [`BillboardCache`] for the `Lod::Far` tier (S6.2).
    /// `None` until the first time S6.3's render dispatch needs
    /// it; populated then via [`BillboardCache::build`] and
    /// cleared by edits ([`Self::set_voxel`] / [`Self::set_rect`]
    /// / [`Self::set_sphere`]) to force a rebuild on next Far use.
    /// Callers may also force-invalidate via direct assignment.
    pub billboards: Option<BillboardCache>,
    /// Optional procedural generator (S7.0). When set,
    /// [`Self::ensure_chunk_generated`] uses it to materialise
    /// chunks that are still absent from [`Self::chunks`].
    ///
    /// Streaming layers (S7.1+) walk the active radius around the
    /// camera and call `ensure_chunk_generated` for missing chunks;
    /// later stages dispatch this onto a background rayon pool. The
    /// trait bound is `Send + Sync` (needed for S7.3 async
    /// dispatch) + `Debug` (needed so [`Grid`] keeps deriving
    /// `Debug`).
    ///
    /// `None` is the default — a grid without a generator behaves
    /// exactly like the pre-S7 grids: absent chunks stay absent.
    ///
    /// `Arc` (not `Box`) so S7.3's async dispatch can clone the
    /// generator into background rayon tasks without moving it out
    /// of the grid. Trait bound `Send + Sync` (required at S7.0)
    /// already makes `Arc<dyn ChunkGenerator>` `Send + Sync`.
    pub generator: Option<Arc<dyn ChunkGenerator>>,
    /// QE.5b - optional host-assigned tag, carried through snapshots.
    /// Grid ids are runtime-opaque, so a save/load cycle gives hosts
    /// nothing to rebind their own per-grid data (generators, stores,
    /// gameplay state) against; a stable name closes that gap. Not
    /// interpreted by the engine.
    pub name: Option<String>,
    /// QE.5a - optional persistence for edited streamed chunks: the
    /// eviction pass hands every `chunk_version != 0` chunk to
    /// [`ChunkStore::store`] before dropping it, and stream-in asks
    /// [`ChunkStore::load`] before running the generator. `None` (the
    /// default) keeps the pre-QE.5 behaviour: evicting an edited
    /// chunk discards the edits.
    pub store: Option<Arc<dyn ChunkStore>>,
    /// Streaming activity / eviction radii used by
    /// [`Scene::pump_streaming_sync`] (S7.1). Defaults to
    /// [`StreamRadius::DISABLED`] so existing grids see no change
    /// in behaviour until the caller opts in.
    pub stream_radius: StreamRadius,
    /// EV.3 — baked point lights ([`BakeLight`], grid-local voxel
    /// coords) consumed by [`BakeMode::PointLights`]: [`Grid::bake`]
    /// and [`Grid::bake_bbox`] write each light's Lambertian pool
    /// into the brightness bytes, so incremental carve relights keep
    /// their glow. Authoring state only — editing this list does
    /// **not** rebake by itself (call [`Grid::bake`] after) and it is
    /// not carried through snapshots (the baked bytes are; re-set the
    /// list after a load if you keep editing). Ignored by the other
    /// bake modes and by the dynamic `LightRig`.
    pub bake_lights: Vec<BakeLight>,
    /// Per-chunk edit version counter (S7.2). Each user edit
    /// through [`Self::set_voxel`] / [`Self::set_rect`] /
    /// [`Self::set_sphere`] bumps the counter for every chunk it
    /// actually wrote to. [`Self::ensure_chunk_generated`] does
    /// NOT bump — a freshly generated chunk has no edits and
    /// reads as version 0.
    ///
    /// Wired up here so the S7.3 async dispatch can detect "an
    /// edit happened while a chunk was being generated in the
    /// background" and discard the now-stale result: each
    /// background task captures the dispatch-time version and
    /// only installs its result iff the current version still
    /// matches.
    ///
    /// Missing entries read as `0` via [`Self::chunk_version`].
    /// Evictions in [`Scene::pump_streaming_sync`] drop the
    /// corresponding entry so the map stays bounded.
    ///
    /// QE.3b — private: every mutation goes through
    /// [`Self::bump_chunk_version`] / [`Self::bump_chunk_version_bbox`]
    /// / the crate-internal tracking helpers, so the
    /// version/extent/counter triple can never desync. Read via
    /// [`Self::chunk_version`] / [`Self::chunk_versions`].
    chunk_versions: HashMap<IVec3, u64>,
    /// In-flight background generation tasks (S7.3).
    ///
    /// Populated by [`Scene::pump_streaming`] when it dispatches a
    /// generator call onto the streaming rayon pool, drained when
    /// the corresponding `ChunkResult` is received and processed
    /// (either installed or discarded). The set is consulted to
    /// avoid re-dispatching the same chunk while a previous task
    /// is still running.
    ///
    /// Stays empty when only the synchronous
    /// [`Scene::pump_streaming_sync`] is used — that path generates
    /// inline on the calling thread.
    pub pending_gen: HashSet<IVec3>,
    /// Cross-frame DDA brick-occupancy cache (Substage DDA.7 perf).
    /// Keyed by `(chunk, mip)` + the chunk's edit version, so a static
    /// chunk's brick map is built once and reused every frame. Skipped
    /// entirely on the voxlap render path. Not serialised.
    pub dda_brick_cache: roxlap_core::BrickCache,
    /// PF.12 — per-chunk change extent accumulated since a consumer
    /// last [`Self::take_chunk_dirty`]'d it: the partial-refresh /
    /// incremental-remip companion to [`Self::chunk_versions`]. Bounded
    /// by the chunk count (one merged entry per chunk). Not serialised.
    chunk_dirty: HashMap<IVec3, DirtyExtent>,
    /// PF.13 (H9) — monotonic counter bumped by EVERY chunk-set /
    /// chunk-content mutation (edits, installs, evictions). Per-frame
    /// consumers (the GPU dirty poll, the brick-cache sweep) compare it
    /// against their last-seen value and skip their O(all chunks) scans
    /// outright on quiet frames. Not serialised.
    mutations: u64,
    /// PF.13 (H9) — `(mutation counter, requested mip, effective mip)`
    /// of the last [`Self::ensure_dda_bricks`] sweep; a matching pair
    /// skips the whole per-chunk ensure + retain pass.
    last_bricks: Option<(u64, u32, u32)>,
}

/// PF.12 — how much of a chunk changed since a consumer last synced it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirtyExtent {
    /// Unknown / whole-chunk change (installs, wholesale replaces,
    /// extent-less [`Grid::bump_chunk_version`] calls).
    Full,
    /// Inclusive CHUNK-LOCAL voxel bbox covering every change.
    Bbox(IVec3, IVec3),
}

impl Grid {
    /// New empty grid at the given transform — no chunks populated,
    /// `render_sky = true`, LOD thresholds default to
    /// [`LodThresholds::always_near`], no billboard cache.
    #[must_use]
    pub fn new(transform: GridTransform) -> Self {
        Self {
            transform,
            chunks: HashMap::new(),
            render_sky: true,
            mip_levels_override: None,
            lod_thresholds: LodThresholds::always_near(),
            billboards: None,
            generator: None,
            name: None,
            store: None,
            stream_radius: StreamRadius::DISABLED,
            bake_lights: Vec::new(),
            chunk_versions: HashMap::new(),
            pending_gen: HashSet::new(),
            dda_brick_cache: roxlap_core::BrickCache::new(),
            chunk_dirty: HashMap::new(),
            mutations: 0,
            last_bricks: None,
        }
    }

    /// Ensure the DDA brick cache holds current mip-`requested_mip`
    /// occupancy maps for every populated chunk, rebuilding only chunks
    /// whose edit version changed (Substage DDA.7). Clamps the mip to a
    /// level every chunk has built (so coarse rendering never holes) and
    /// returns that effective mip. Evicts cache entries for chunks no
    /// longer present. Call once per frame before the DDA render.
    pub fn ensure_dda_bricks(&mut self, requested_mip: u32) -> u32 {
        // Split-borrow disjoint fields so the cache mutates while the
        // chunks + versions are read.
        // PF.13 (H9) — quiet frame at the same requested mip ⇒ nothing
        // in the cache can be stale: skip the whole O(chunks) sweep.
        if let Some((counter, req, eff)) = self.last_bricks {
            if counter == self.mutations && req == requested_mip {
                return eff;
            }
        }
        let mutations = self.mutations;
        let Self {
            chunks,
            chunk_versions,
            dda_brick_cache,
            ..
        } = self;
        // Effective uniform mip: min built mip across chunks, capped.
        let mut mip = requested_mip;
        if requested_mip > 0 {
            for vxl in chunks.values() {
                mip = mip.min(vxl.mip_count().saturating_sub(1));
            }
        }
        for (idx, vxl) in chunks.iter() {
            let version = chunk_versions.get(idx).copied().unwrap_or(0);
            let view = roxlap_core::GridView::from_single_vxl(vxl);
            dda_brick_cache.ensure([idx.x, idx.y, idx.z], mip, version, &view);
        }
        dda_brick_cache.retain_chunks(|c| chunks.contains_key(&IVec3::new(c[0], c[1], c[2])));
        self.last_bricks = Some((mutations, requested_mip, mip));
        mip
    }

    /// Current per-chunk edit version (S7.2). Returns `0` for any
    /// chunk that hasn't been edited yet (including absent chunks
    /// and chunks materialised only via
    /// [`Self::ensure_chunk_generated`]).
    ///
    /// Used by S7.3's async generation dispatch to detect "edit
    /// happened while we were generating" — the dispatcher
    /// snapshots this value, the background task carries it, and
    /// the result is discarded on install if the live counter has
    /// since moved.
    #[must_use]
    pub fn chunk_version(&self, chunk_idx: IVec3) -> u64 {
        self.chunk_versions.get(&chunk_idx).copied().unwrap_or(0)
    }

    /// Bump the edit version of `chunk_idx` (S7.2). Saturating add
    /// at `u64::MAX` — a chunk would need 10^11 edits per second
    /// for ~5 years to wrap, so saturation is a defensive cap, not
    /// a realistic concern.
    ///
    /// Called by the edit API ([`Self::set_voxel`] /
    /// [`Self::set_rect`] / [`Self::set_sphere`]) after a chunk
    /// has actually been written to. Pure no-op edit paths
    /// (carving from an air chunk that doesn't exist yet) skip
    /// the bump.
    ///
    /// Exposed as `pub` (vs the historical `pub(crate)`) so hosts
    /// that mutate `grid.chunks` directly — e.g.
    /// `roxlap-scene-demo`'s `StreamingBakeTracker` writing
    /// lightmode-1 alphas via `apply_lighting_with_cache` — can
    /// signal "this chunk's slab changed" to downstream consumers
    /// like the GPU dirty-chunk poller.
    pub fn bump_chunk_version(&mut self, chunk_idx: IVec3) {
        let entry = self.chunk_versions.entry(chunk_idx).or_insert(0);
        *entry = entry.saturating_add(1);
        // PF.12 — no extent information ⇒ the whole chunk must be
        // treated as changed by partial-refresh consumers.
        self.chunk_dirty.insert(chunk_idx, DirtyExtent::Full);
        self.mutations = self.mutations.wrapping_add(1);
    }

    /// PF.13 (H9) — the grid's monotonic mutation counter: bumped by
    /// every chunk edit, install, and eviction. Per-frame consumers
    /// snapshot it to skip their whole-grid scans on quiet frames.
    #[must_use]
    pub fn mutation_counter(&self) -> u64 {
        self.mutations
    }

    /// PF.12 — [`Self::bump_chunk_version`] with the edit's CHUNK-LOCAL
    /// voxel extent (inclusive), so partial-refresh consumers (the GPU
    /// facade) and incremental re-mip know how little actually changed.
    /// Extents accumulate (bbox union) until a consumer
    /// [`Self::take_chunk_dirty`]s them; an extent-less bump upgrades
    /// the entry to [`DirtyExtent::Full`].
    pub fn bump_chunk_version_bbox(&mut self, chunk_idx: IVec3, lo: IVec3, hi: IVec3) {
        let entry = self.chunk_versions.entry(chunk_idx).or_insert(0);
        *entry = entry.saturating_add(1);
        let merged = match self.chunk_dirty.get(&chunk_idx) {
            Some(DirtyExtent::Full) => DirtyExtent::Full,
            Some(DirtyExtent::Bbox(l, h)) => DirtyExtent::Bbox(l.min(lo), h.max(hi)),
            None => DirtyExtent::Bbox(lo, hi),
        };
        self.chunk_dirty.insert(chunk_idx, merged);
        self.mutations = self.mutations.wrapping_add(1);
    }

    /// PF.12 — take (and clear) the extent accumulated for `chunk_idx`
    /// since the last take. `None` ⇒ no recorded change (a consumer that
    /// still observed a version bump should treat that as
    /// [`DirtyExtent::Full`] — e.g. a change recorded before the
    /// consumer first synced the chunk).
    pub fn take_chunk_dirty(&mut self, chunk_idx: IVec3) -> Option<DirtyExtent> {
        self.chunk_dirty.remove(&chunk_idx)
    }

    /// The full per-chunk edit-version map (QE.3b — the read half of
    /// the previously `pub` field). Consumers seeding a sync tracker
    /// iterate this; per-chunk reads go through
    /// [`Self::chunk_version`].
    #[must_use]
    pub fn chunk_versions(&self) -> &HashMap<IVec3, u64> {
        &self.chunk_versions
    }

    /// QE.3b — record a chunk-**set** mutation (materialise / install /
    /// evict) on the PF.13 quiet-frame counter. The single entry point
    /// for the counter besides the version bumps above; per-frame
    /// consumers compare [`Self::mutation_counter`] snapshots.
    pub(crate) fn note_chunk_set_changed(&mut self) {
        self.mutations = self.mutations.wrapping_add(1);
    }

    /// QE.3b — drop `chunk_idx`'s per-chunk tracking on eviction, so
    /// both maps stay bounded by the live chunk count. (Also clears any
    /// accumulated [`DirtyExtent`] — pre-QE.3b that entry leaked until
    /// a consumer happened to take it.) A future re-stream of the same
    /// index restarts at version 0.
    pub(crate) fn forget_chunk_tracking(&mut self, chunk_idx: IVec3) {
        self.chunk_versions.remove(&chunk_idx);
        self.chunk_dirty.remove(&chunk_idx);
    }

    /// QE.3b — seat a restored per-chunk version verbatim (the
    /// [`snapshot`] load path; not an edit, so no extent / counter
    /// side-effects).
    pub(crate) fn restore_chunk_version(&mut self, chunk_idx: IVec3, version: u64) {
        self.chunk_versions.insert(chunk_idx, version);
    }

    /// Attach (or detach) the procedural generator used by
    /// [`Self::ensure_chunk_generated`] (S7.0).
    ///
    /// Pass `Some(Arc::new(generator))` to enable on-demand chunk
    /// generation; pass `None` to revert to the "absent stays
    /// absent" behaviour. Replacing an existing generator drops the
    /// previous `Arc` clone without touching already-materialised
    /// chunks. Any background tasks dispatched by a prior
    /// [`Scene::pump_streaming`] hold their own clones of the old
    /// generator and finish naturally.
    pub fn set_generator(&mut self, generator: Option<Arc<dyn ChunkGenerator>>) {
        self.generator = generator;
    }

    /// Attach (or detach) the persistence store for edited streamed
    /// chunks (QE.5a; see [`ChunkStore`]). Without one, evicting an
    /// edited chunk **discards the edits** — the pre-QE.5 default.
    pub fn set_chunk_store(&mut self, store: Option<Arc<dyn ChunkStore>>) {
        self.store = store;
    }

    /// Materialise the chunk at `chunk_idx` by running [`Self::generator`]
    /// if (a) the chunk is not already present and (b) a generator
    /// is attached. Returns `true` iff a chunk was newly generated.
    ///
    /// No-ops in all other cases:
    /// - chunk already present (caller edits / a previous
    ///   `ensure_chunk_generated` call already populated it),
    /// - no generator attached (the chunk stays implicit-air per
    ///   the existing convention — does NOT fall through to
    ///   [`Self::ensure_chunk`]'s empty-chunk constructor).
    ///
    /// This is the synchronous S7.0 path; [`Scene::pump_streaming`]
    /// is the async counterpart (generation + store loads on a
    /// dedicated rayon pool).
    ///
    /// QE.5a: with a [`ChunkStore`] attached, a stored chunk is
    /// installed (with its persisted edit version) **before** the
    /// generator is consulted — including for indices the generator's
    /// [`ChunkGenerator::should_generate`] declines, since edits can
    /// materialise chunks the generator never would.
    pub fn ensure_chunk_generated(&mut self, chunk_idx: IVec3) -> bool {
        if self.chunks.contains_key(&chunk_idx) {
            return false;
        }
        // QE.5a — persisted edits win over regeneration.
        if let Some((vxl, version)) = self.store.as_ref().and_then(|store| store.load(chunk_idx)) {
            self.chunks.insert(chunk_idx, vxl);
            self.restore_chunk_version(chunk_idx, version);
            self.note_chunk_set_changed();
            self.billboards = None; // same invalidation as below
            return true;
        }
        let Some(generator) = self.generator.as_ref() else {
            return false;
        };
        // S7.6+: generator may decline specific indices (e.g. a
        // single-z-layer generator skipping placeholder bedrock
        // chunks at chz != 0). Respect the filter so we don't
        // materialise an unwanted chunk.
        if !generator.should_generate(chunk_idx) {
            return false;
        }
        let chunk = generator.generate(chunk_idx);
        self.chunks.insert(chunk_idx, chunk);
        self.note_chunk_set_changed();
        // S7.4: a fresh chunk grows the populated AABB → the
        // bounding sphere shifts/expands → existing impostor
        // projections become wrong. Match the eviction (S7.1) +
        // edit (S6.2) invalidation contract and drop the cache.
        // Next Far-tier render rebuilds lazily.
        self.billboards = None;
        true
    }

    /// Bounding-sphere radius of the populated chunk set in
    /// grid-local space.
    ///
    /// Walks the sparse chunk map once, computes the chunk-index
    /// AABB, converts to voxel-space half-extent, returns its
    /// Euclidean length. Empty grid → `0.0`.
    ///
    /// Conservative — bounds the full chunk volume, not just its
    /// populated voxels (a chunk containing one voxel still
    /// contributes `CHUNK_SIZE_XY × CHUNK_SIZE_XY × CHUNK_SIZE_Z`
    /// to the bbox). For LOD picking that's fine: an over-bound
    /// sphere errs on the side of `Near`.
    ///
    /// Cost: `O(chunks.len())`; recomputed on every call. Callers
    /// who need this every frame should memoize at the
    /// [`Scene`]-level cache (added when S6.2 needs it).
    #[must_use]
    pub fn bounding_radius(&self) -> f64 {
        if self.chunks.is_empty() {
            return 0.0;
        }
        let mut min = IVec3::splat(i32::MAX);
        let mut max = IVec3::splat(i32::MIN);
        for &idx in self.chunks.keys() {
            min = min.min(idx);
            max = max.max(idx);
        }
        // Chunk-index bbox → voxel-space half-extent. `+1` on max
        // converts inclusive chunk index to exclusive voxel upper
        // bound (chunk `idx` covers voxels `[idx*size, (idx+1)*size)`).
        let sx = f64::from(CHUNK_SIZE_XY);
        let sz = f64::from(CHUNK_SIZE_Z);
        let lo = DVec3::new(
            f64::from(min.x) * sx,
            f64::from(min.y) * sx,
            f64::from(min.z) * sz,
        );
        let hi = DVec3::new(
            f64::from(max.x + 1) * sx,
            f64::from(max.y + 1) * sx,
            f64::from(max.z + 1) * sz,
        );
        let half_extent = (hi - lo) * 0.5;
        half_extent.length()
    }

    /// Pick this grid's LOD tier for the given world-space camera
    /// position. Convenience wrapper around [`crate::select_lod`]
    /// that pulls [`Self::lod_thresholds`] from the grid.
    #[must_use]
    pub fn select_lod(&self, camera_world_pos: DVec3) -> Lod {
        select_lod(camera_world_pos, &self.transform, self.lod_thresholds)
    }
}

/// Top-level scene container. Holds a flat collection of grids
/// keyed by [`GridId`].
///
/// S2.0 only exposes registration / removal / lookup. Address math
/// helpers (S2.x), edit API (S2.x), and rendering composition (S3)
/// land in later sub-substages.
#[derive(Debug, Default)]
pub struct Scene {
    grids: HashMap<GridId, Grid>,
    next_grid_id: u32,
    /// S7.3: per-scene streaming pool + result channel. Stored on
    /// the `Scene` so `pump_streaming` can dispatch background
    /// tasks and drain results across pump calls. `cfg`-gated out
    /// on wasm32 where `pump_streaming` short-circuits to
    /// `pump_streaming_sync` (no rayon pool there).
    #[cfg(not(target_arch = "wasm32"))]
    streaming: streaming::StreamingState,
}

impl Scene {
    /// New empty scene — no grids.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of grids currently registered.
    #[must_use]
    pub fn grid_count(&self) -> usize {
        self.grids.len()
    }

    /// Register a new grid. Returns its fresh, unique [`GridId`].
    pub fn add_grid(&mut self, transform: GridTransform) -> GridId {
        let id = GridId(self.next_grid_id);
        self.next_grid_id += 1;
        self.grids.insert(id, Grid::new(transform));
        id
    }

    /// Remove a grid by id. Returns the removed [`Grid`] (so the
    /// caller can reclaim its chunks) or `None` if the id wasn't
    /// registered. Removed ids are not reissued.
    pub fn remove_grid(&mut self, id: GridId) -> Option<Grid> {
        self.grids.remove(&id)
    }

    /// Borrow a registered grid.
    #[must_use]
    pub fn grid(&self, id: GridId) -> Option<&Grid> {
        self.grids.get(&id)
    }

    /// Mutably borrow a registered grid.
    pub fn grid_mut(&mut self, id: GridId) -> Option<&mut Grid> {
        self.grids.get_mut(&id)
    }

    /// Iterator over all `(id, grid)` pairs in registration order
    /// is **not** guaranteed — the underlying map is a `HashMap`.
    /// Callers that need a stable order must sort by [`GridId`].
    pub fn grids(&self) -> impl Iterator<Item = (GridId, &Grid)> {
        self.grids.iter().map(|(id, g)| (*id, g))
    }

    /// Mutable iterator over all `(id, grid)` pairs. Yield order
    /// is not guaranteed (HashMap-backed).
    pub fn grids_mut(&mut self) -> impl Iterator<Item = (GridId, &mut Grid)> {
        self.grids.iter_mut().map(|(id, g)| (*id, g))
    }

    /// Resolve a world-space surface hit to the owning grid + its
    /// grid-local voxel — the picking back half. `ray_dir` is the view
    /// direction the hit was found along (need not be normalised); the
    /// point is nudged half a voxel along it, past the surface and into
    /// the solid cell, before each grid's [`Grid::voxel_solid`] test.
    /// Returns the first grid that is solid there (transform-correct
    /// for rotated/translated grids), or `None` if none claims it.
    ///
    /// Backend-agnostic: pair with a renderer depth read to turn a
    /// click into a voxel — `world = cam.pos + t · normalize(ray_dir)`,
    /// then `resolve_voxel(world, ray_dir)`. `roxlap-render`'s
    /// `SceneRenderer::pick` wires exactly that.
    #[must_use]
    pub fn resolve_voxel(&self, world: DVec3, ray_dir: DVec3) -> Option<(GridId, IVec3)> {
        let len = ray_dir.length();
        if len < 1e-9 {
            return None;
        }
        let inside = world + ray_dir * (0.5 / len); // half a voxel inward
        for (id, grid) in self.grids() {
            let glp = addr::world_to_grid_local(inside, &grid.transform);
            let v = addr::voxel_global(glp.chunk, glp.voxel);
            if grid.voxel_solid(v) {
                return Some((id, v));
            }
        }
        None
    }

    /// Cast a world-space ray and return the nearest solid voxel hit
    /// across all grids, or `None` if nothing solid lies within
    /// `max_dist`. Renderer-independent (no depth buffer, no camera) —
    /// the primitive for line-of-sight, projectiles, AI probing, and
    /// off-screen / backend-agnostic picking.
    ///
    /// `dir` need not be normalised. Each grid's ray is transformed
    /// into the grid's local frame (so rotated / translated grids are
    /// handled exactly) and marched with a voxel DDA against
    /// [`Grid::voxel_solid`]; the closest hit by world distance `t`
    /// wins. The step budget is bounded by `max_dist`, so empty space
    /// is safe but not free — a chunk-level skip is a future
    /// optimisation if hot.
    #[must_use]
    pub fn raycast(&self, origin: DVec3, dir: DVec3, max_dist: f64) -> Option<RayHit> {
        let len = dir.length();
        if len < 1e-12 || max_dist <= 0.0 {
            return None;
        }
        let dn = dir / len; // unit world direction → t is world distance
        let mut best: Option<RayHit> = None;
        for (id, grid) in self.grids() {
            // World ray → grid-local: undo translation + rotation. The
            // inverse rotation preserves length, so `t` stays in world
            // units and is comparable across grids.
            let inv = grid.transform.rotation.inverse();
            let lo = inv * (origin - grid.transform.origin);
            let ld = inv * dn;
            if let Some((voxel, t)) = voxel_dda(grid, lo, ld, max_dist) {
                if best.as_ref().is_none_or(|b| t < b.t) {
                    best = Some(RayHit {
                        grid: id,
                        voxel,
                        world: origin + dn * t,
                        t,
                        color: grid.voxel_color(voxel),
                    });
                }
            }
        }
        best
    }

    /// Configure the number of worker threads in the dedicated
    /// streaming pool (S7.3).
    ///
    /// Lazily applied — the pool itself is constructed on the first
    /// [`Self::pump_streaming`] call. If the pool was already built
    /// (i.e. a previous `pump_streaming` already dispatched at
    /// least one task), it gets dropped and rebuilt. Dropping the
    /// old pool blocks until all of its in-flight tasks finish
    /// (rayon's contract); any results those tasks sent are still
    /// drained by the next `pump_streaming` because the channel
    /// survives the rebuild.
    ///
    /// The streaming pool is separate from rayon's global pool
    /// (which R12 multicore rendering uses), so chunk generation
    /// doesn't compete with render threads. Sensible values are 1
    /// to ~4 — generation work is CPU-bound but should leave most
    /// of the box for everything else.
    ///
    /// On wasm32 this is a no-op (no rayon pool available);
    /// `pump_streaming` runs synchronously there.
    ///
    /// # Panics
    /// Panics on native if `n == 0` (zero-thread pools are not
    /// supported; the scene crate's S7.1 helper already disallows
    /// the equivalent for `StreamRadius::r_active < 0`).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn set_streaming_threads(&mut self, n: usize) {
        self.streaming.set_thread_count(n);
    }

    /// wasm32 no-op companion of [`Self::set_streaming_threads`].
    /// Lets cross-target code call this unconditionally.
    #[cfg(target_arch = "wasm32")]
    pub fn set_streaming_threads(&mut self, _n: usize) {
        // No streaming pool on wasm32 — see `pump_streaming` docs.
    }

    /// Asynchronous streaming pump (S7.3).
    ///
    /// On native, dispatches missing-chunk generations onto a
    /// dedicated rayon pool, drains any results that arrived since
    /// the last pump, runs the eviction pass, and tracks in-flight
    /// tasks in each grid's [`Grid::pending_gen`] set. The drain
    /// uses the per-chunk version counter from S7.2 to discard
    /// results whose chunk was edited mid-generation.
    ///
    /// On wasm32 this short-circuits to [`Self::pump_streaming_sync`]
    /// — no thread pool is available there, but the same per-grid
    /// stream-in / evict semantics apply.
    ///
    /// Call once per frame from the render thread. Cheap when
    /// nothing changed (early-exit on disabled grids, try_recv
    /// loops empty fast).
    pub fn pump_streaming(&mut self, camera_world_pos: DVec3) {
        #[cfg(target_arch = "wasm32")]
        {
            self.pump_streaming_sync(camera_world_pos);
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.pump_streaming_native(camera_world_pos);
        }
    }

    /// Native implementation of [`Self::pump_streaming`].
    #[cfg(not(target_arch = "wasm32"))]
    fn pump_streaming_native(&mut self, camera_world_pos: DVec3) {
        // 1. Drain inbox — install fresh results, drop stale.
        while let Ok(result) = self.streaming.rx.try_recv() {
            let Some(grid) = self.grids.get_mut(&result.grid_id) else {
                // Grid was removed while a generation task was
                // in-flight. Drop silently.
                continue;
            };
            // Clearing pending_gen here both for "result delivered"
            // and "we shouldn't try to re-dispatch this chunk just
            // because it's missing".
            let was_pending = grid.pending_gen.remove(&result.chunk_idx);
            if !was_pending {
                // Either the chunk was evicted (pending cleared in
                // the eviction pass below in some prior call), or a
                // duplicate result for an already-handled chunk.
                continue;
            }
            if grid.chunks.contains_key(&result.chunk_idx) {
                // Some other path (e.g. `ensure_chunk_generated`
                // sync helper, or a manual edit's `ensure_chunk`)
                // already populated the slot. Don't overwrite.
                continue;
            }
            if grid.chunk_version(result.chunk_idx) != result.version_at_dispatch {
                // S7.2 stale-result discard: chunk was edited mid-
                // generation.
                continue;
            }
            let Some(vxl) = result.vxl else {
                // QE.5a — nothing to install (store miss + generator
                // declined); the pending entry is already cleared.
                continue;
            };
            grid.chunks.insert(result.chunk_idx, vxl);
            if let Some(version) = result.restored_version {
                // QE.5a — a store-restored chunk keeps its persisted
                // edit version (consumers see it as edited content).
                grid.restore_chunk_version(result.chunk_idx, version);
            }
            grid.note_chunk_set_changed();
            // S7.4: same invalidation contract as the sync
            // `ensure_chunk_generated` path — installing a new
            // chunk can grow the bounding sphere, so the
            // billboard impostor cache must be rebuilt on next
            // Far entry. Lazy: only one cache wipe per drain
            // batch, the Far render rebuilds afterwards.
            grid.billboards = None;
        }

        // 2. Per-grid: eviction first, then dispatch. Doing evict
        //    before dispatch means a chunk that's just left
        //    r_active doesn't get re-dispatched on the same pump.
        self.streaming.ensure_pool();
        // Disjoint sub-field borrows: pool/tx via `&self.streaming.*`,
        // grids via `&mut self.grids`. Hold both at once.
        let pool: &rayon::ThreadPool = self.streaming.pool.as_ref().expect("ensure_pool just ran");
        let tx_template = &self.streaming.tx;
        for (grid_id, grid) in &mut self.grids {
            evict_grid_chunks(grid, camera_world_pos);
            dispatch_grid_async(*grid_id, grid, camera_world_pos, pool, tx_template);
        }
    }

    /// Synchronous streaming pump (S7.1).
    ///
    /// For each grid with a non-[`StreamRadius::DISABLED`] policy:
    /// 1. Project the world-space camera into grid-local coords
    ///    (inverse rotation + origin subtract).
    /// 2. Stream in any chunk whose AABB-to-camera distance is
    ///    `<= r_active`, calling [`Grid::ensure_chunk_generated`].
    ///    No-ops gracefully if the grid has no generator attached
    ///    (so callers can use the eviction half of streaming on a
    ///    purely-edited grid).
    /// 3. Evict any chunk whose AABB-to-camera distance exceeds
    ///    `r_evict` from the grid's chunk map. Eviction also
    ///    clears the cached [`BillboardCache`] (the bounding sphere
    ///    may shrink, invalidating impostor projections; the next
    ///    Far-tier render rebuilds lazily).
    ///
    /// Both passes use the f64 grid-local position so rotation +
    /// non-axis-aligned grids stream and evict correctly. The
    /// generate path is blocking — S7.3 will move it to a
    /// background rayon pool with `pump_streaming` (non-blocking).
    /// Callers that want the async variant in S7.0/S7.1 stages
    /// should keep `r_active` small.
    pub fn pump_streaming_sync(&mut self, camera_world_pos: DVec3) {
        for grid in self.grids.values_mut() {
            pump_grid_streaming_sync(grid, camera_world_pos);
        }
    }
}

/// S7.1 helper — drives one grid's synchronous streaming pass.
/// Stream-in pass uses [`Grid::ensure_chunk_generated`] (blocking
/// inline generation); eviction pass shared with the S7.3 async
/// path through [`evict_grid_chunks`].
fn pump_grid_streaming_sync(grid: &mut Grid, camera_world_pos: DVec3) {
    let radius = grid.stream_radius;
    if radius.is_disabled() {
        return;
    }
    let cam_local = streaming::world_to_grid_local_pos(camera_world_pos, &grid.transform);

    // --- Pass 1: stream in active chunks (sync) ---------------
    // QE.5a — a ChunkStore alone (no generator) still restores
    // persisted edited chunks.
    if radius.r_active > 0.0 && (grid.generator.is_some() || grid.store.is_some()) {
        for_each_chunk_in_radius(cam_local, radius.r_active, |idx| {
            grid.ensure_chunk_generated(idx);
        });
    }

    // --- Pass 2: evict chunks past r_evict --------------------
    evict_grid_chunks_with_cam(grid, cam_local);
}

/// Eviction pass shared by [`pump_grid_streaming_sync`] and the
/// S7.3 async path. Public-ish so the async driver can call it
/// before dispatching to avoid generating chunks that are about
/// to be evicted. `cfg`-gated to native: on wasm32 the only
/// caller (`pump_streaming_native`) doesn't compile, so this
/// helper would warn as dead code.
#[cfg(not(target_arch = "wasm32"))]
fn evict_grid_chunks(grid: &mut Grid, camera_world_pos: DVec3) {
    let radius = grid.stream_radius;
    if radius.is_disabled() {
        return;
    }
    let cam_local = streaming::world_to_grid_local_pos(camera_world_pos, &grid.transform);
    evict_grid_chunks_with_cam(grid, cam_local);
}

/// Eviction inner — assumes `cam_local` is already computed (the
/// dispatcher and sync pump both have it on hand).
fn evict_grid_chunks_with_cam(grid: &mut Grid, cam_local: DVec3) {
    let radius = grid.stream_radius;
    if !radius.r_evict.is_finite() {
        return;
    }
    let r_sq = radius.r_evict * radius.r_evict;
    let to_evict: Vec<IVec3> = grid
        .chunks
        .keys()
        .filter(|&&idx| streaming::chunk_aabb_dist_sq(cam_local, idx) > r_sq)
        .copied()
        .collect();
    // S7.3: also evict pending in-flight tasks past r_evict so the
    // drain pass doesn't install a chunk that's no longer wanted.
    // We don't have a way to cancel the rayon task, but we can
    // drop the pending_gen entry so the result is dropped on
    // arrival.
    let to_evict_pending: Vec<IVec3> = grid
        .pending_gen
        .iter()
        .filter(|&&idx| streaming::chunk_aabb_dist_sq(cam_local, idx) > r_sq)
        .copied()
        .collect();
    if to_evict.is_empty() && to_evict_pending.is_empty() {
        return;
    }
    for idx in &to_evict {
        // QE.5a — an edited chunk (version != 0) is handed to the
        // ChunkStore before it drops, so the edits survive evict +
        // re-stream. Pristine generator output (version 0) is
        // regenerable and skips the store.
        if let Some(store) = grid.store.as_ref() {
            let version = grid.chunk_version(*idx);
            if version != 0 {
                if let Some(vxl) = grid.chunks.get(idx) {
                    store.store(*idx, vxl, version);
                }
            }
        }
        grid.chunks.remove(idx);
        grid.note_chunk_set_changed();
        // S7.2/QE.3b: drop the chunk's version + dirty-extent tracking
        // so both maps stay bounded. A future re-stream of the same idx
        // restarts at 0 — that's fine because any in-flight gen-result
        // tagged with the pre-eviction version is unreachable (no chunk
        // to install onto) and gets discarded by the new "version
        // still 0" check anyway. (A stored edited chunk re-enters with
        // its persisted version via the QE.5a load path instead.)
        grid.forget_chunk_tracking(*idx);
        // S7.3: drop pending entry for the same chunk too. If a
        // background task is still running, its result will be
        // dropped on arrival (was_pending = false).
        grid.pending_gen.remove(idx);
    }
    for idx in &to_evict_pending {
        grid.pending_gen.remove(idx);
    }
    if !to_evict.is_empty() {
        // Bounding sphere can shrink → impostor projections would
        // be wrong on next Far render. Clear lazily; the next
        // Far-tier pass repopulates via BillboardCache::build.
        grid.billboards = None;
    }
}

/// Walk every chunk index whose AABB falls within `r_active` of
/// `cam_local` and invoke `f` on it. Shared between the S7.1 sync
/// stream-in and the S7.3 async dispatch.
fn for_each_chunk_in_radius<F>(cam_local: DVec3, r_active: f64, mut f: F)
where
    F: FnMut(IVec3),
{
    let r_sq = r_active * r_active;
    let sxy = f64::from(CHUNK_SIZE_XY);
    let sz = f64::from(CHUNK_SIZE_Z);
    // Half-extent in chunk units; ceil to be conservative so any
    // chunk whose AABB clips the radius gets considered. `+1`
    // covers the half-open chunk-AABB upper edge plus the case
    // where the camera sits exactly on a chunk boundary and the
    // closest chunk is one index off.
    #[allow(clippy::cast_possible_truncation)]
    let r_chunks_xy = (r_active / sxy).ceil() as i32 + 1;
    #[allow(clippy::cast_possible_truncation)]
    let r_chunks_z = (r_active / sz).ceil() as i32 + 1;
    #[allow(clippy::cast_possible_truncation)]
    let cx_chunk = (cam_local.x / sxy).floor() as i32;
    #[allow(clippy::cast_possible_truncation)]
    let cy_chunk = (cam_local.y / sxy).floor() as i32;
    #[allow(clippy::cast_possible_truncation)]
    let cz_chunk = (cam_local.z / sz).floor() as i32;
    for chz in (cz_chunk - r_chunks_z)..=(cz_chunk + r_chunks_z) {
        for chy in (cy_chunk - r_chunks_xy)..=(cy_chunk + r_chunks_xy) {
            for chx in (cx_chunk - r_chunks_xy)..=(cx_chunk + r_chunks_xy) {
                let idx = IVec3::new(chx, chy, chz);
                if streaming::chunk_aabb_dist_sq(cam_local, idx) <= r_sq {
                    f(idx);
                }
            }
        }
    }
}

/// S7.3 async dispatch — schedule generation for every chunk in
/// `r_active` that's not already present and not already in
/// flight. Each dispatch clones the grid's generator `Arc` and a
/// sender clone, then spawns the closure on the streaming rayon
/// pool. The closure does the generate + send; the main thread
/// drains results on the next pump.
#[cfg(not(target_arch = "wasm32"))]
fn dispatch_grid_async(
    grid_id: GridId,
    grid: &mut Grid,
    camera_world_pos: DVec3,
    pool: &rayon::ThreadPool,
    tx: &crossbeam_channel::Sender<streaming::ChunkResult>,
) {
    let radius = grid.stream_radius;
    if radius.is_disabled() || radius.r_active <= 0.0 {
        return;
    }
    // QE.5a — a ChunkStore alone (no generator) still restores
    // persisted edited chunks on the background pool.
    let generator = grid.generator.as_ref().map(Arc::clone);
    let store = grid.store.as_ref().map(Arc::clone);
    if generator.is_none() && store.is_none() {
        return;
    }
    let cam_local = streaming::world_to_grid_local_pos(camera_world_pos, &grid.transform);
    for_each_chunk_in_radius(cam_local, radius.r_active, |idx| {
        if grid.chunks.contains_key(&idx) {
            return; // already present
        }
        if grid.pending_gen.contains(&idx) {
            return; // already in flight
        }
        // S7.6+: respect the generator's per-chunk filter — same
        // contract as `Grid::ensure_chunk_generated` (sync helper).
        // Lets a generator decline to materialise specific indices
        // (e.g. `HillsChunkGenerator` skipping placeholder bedrock
        // chunks at chz != 0 so the camera-above-grid path doesn't
        // create chz < 0 entries that would shift `origin_chunk_z`
        // and trigger the S4B.6.j cross-chunk look-down bug).
        // QE.5a — with a store attached the chunk is still
        // dispatched: a persisted edit wins over the decline (edits
        // can materialise chunks the generator never would); the
        // background task falls back to "nothing" on a store miss.
        let declined = !generator.as_ref().is_some_and(|g| g.should_generate(idx));
        if declined && store.is_none() {
            return;
        }
        grid.pending_gen.insert(idx);
        let version_at_dispatch = grid.chunk_version(idx);
        let tx_clone = tx.clone();
        let gen_clone = generator.clone();
        let store_clone = store.clone();
        pool.spawn(move || {
            // QE.5a — persisted edits win over regeneration; the
            // store load runs here on the pool so blocking IO never
            // stalls the render thread.
            let (vxl, restored_version) = match store_clone.as_ref().and_then(|s| s.load(idx)) {
                Some((vxl, version)) => (Some(vxl), Some(version)),
                None if declined => (None, None),
                None => (gen_clone.map(|g| g.generate(idx)), None),
            };
            // Send is non-blocking on unbounded channel; if the
            // receiver was dropped (Scene drop), the send fails
            // silently — that's fine.
            let _ = tx_clone.send(streaming::ChunkResult {
                grid_id,
                chunk_idx: idx,
                version_at_dispatch,
                vxl,
                restored_version,
            });
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_scene_has_no_grids() {
        let scene = Scene::new();
        assert_eq!(scene.grid_count(), 0);
        assert!(scene.grids().next().is_none());
    }

    #[test]
    fn raycast_hits_axis_aligned_voxel() {
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::identity());
        scene
            .grid_mut(id)
            .unwrap()
            .set_voxel(IVec3::new(5, 5, 10), Some(VoxColor(0x80_aa_bb_cc)));

        // Straight down the +z column through (5,5): hits z=10 at t≈10.
        let hit = scene
            .raycast(DVec3::new(5.5, 5.5, 0.0), DVec3::new(0.0, 0.0, 1.0), 64.0)
            .expect("ray hits the voxel");
        assert_eq!(hit.grid, id);
        assert_eq!(hit.voxel, IVec3::new(5, 5, 10));
        assert!((hit.t - 10.0).abs() < 1e-6, "t≈10, got {}", hit.t);
        assert!(hit.color.is_some(), "textured voxel has a colour");

        // A column with no voxel misses.
        assert!(
            scene
                .raycast(DVec3::new(0.5, 0.5, 0.0), DVec3::new(0.0, 0.0, 1.0), 64.0)
                .is_none(),
            "empty column → no hit",
        );
    }

    #[test]
    fn raycast_respects_grid_transform() {
        // A translated grid: the hit voxel is reported in GRID-LOCAL
        // coords, and the world hit point is back in world space — so a
        // host gets the true voxel regardless of where the grid sits.
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::at(DVec3::new(100.0, 0.0, 0.0)));
        scene
            .grid_mut(id)
            .unwrap()
            .set_voxel(IVec3::new(5, 5, 10), Some(VoxColor(0x80_11_22_33)));

        let hit = scene
            .raycast(DVec3::new(105.5, 5.5, 0.0), DVec3::new(0.0, 0.0, 1.0), 64.0)
            .expect("ray hits the translated voxel");
        assert_eq!(hit.voxel, IVec3::new(5, 5, 10), "grid-local voxel");
        assert!((hit.world.x - 105.5).abs() < 1e-6, "world x preserved");
        assert!((hit.t - 10.0).abs() < 1e-6, "t≈10, got {}", hit.t);
    }

    #[test]
    fn raycast_picks_nearest_grid() {
        // Two grids with a voxel each along the same world column; the
        // raycast must return the closer one.
        let mut scene = Scene::new();
        let near = scene.add_grid(GridTransform::identity());
        let far = scene.add_grid(GridTransform::identity());
        scene
            .grid_mut(near)
            .unwrap()
            .set_voxel(IVec3::new(1, 1, 20), Some(VoxColor(0x80_00_ff_00)));
        scene
            .grid_mut(far)
            .unwrap()
            .set_voxel(IVec3::new(1, 1, 40), Some(VoxColor(0x80_ff_00_00)));

        let hit = scene
            .raycast(DVec3::new(1.5, 1.5, 0.0), DVec3::new(0.0, 0.0, 1.0), 64.0)
            .expect("hits the nearer voxel");
        assert_eq!(hit.grid, near);
        assert_eq!(hit.voxel, IVec3::new(1, 1, 20));
    }

    #[test]
    fn add_grid_returns_fresh_ids() {
        let mut scene = Scene::new();
        let a = scene.add_grid(GridTransform::identity());
        let b = scene.add_grid(GridTransform::at(DVec3::new(100.0, 0.0, 0.0)));
        assert_ne!(a, b);
        assert_eq!(a.raw(), 0);
        assert_eq!(b.raw(), 1);
        assert_eq!(scene.grid_count(), 2);
    }

    #[test]
    fn grid_lookup_round_trips() {
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::at(DVec3::new(10.0, 20.0, 30.0)));
        let g = scene.grid(id).expect("grid registered");
        assert_eq!(g.transform.origin, DVec3::new(10.0, 20.0, 30.0));
        assert_eq!(g.transform.rotation, DQuat::IDENTITY);
        assert!(g.chunks.is_empty());
    }

    #[test]
    fn remove_grid_drops_it_from_scene() {
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::identity());
        let removed = scene.remove_grid(id);
        assert!(removed.is_some());
        assert_eq!(scene.grid_count(), 0);
        assert!(scene.grid(id).is_none());
        // Re-adding does NOT reuse the dropped id.
        let id2 = scene.add_grid(GridTransform::identity());
        assert_ne!(id, id2);
        assert_eq!(id2.raw(), 1);
    }

    #[test]
    fn remove_unknown_grid_is_none() {
        let mut scene = Scene::new();
        let bogus = GridId(999);
        assert!(scene.remove_grid(bogus).is_none());
    }

    #[test]
    fn grid_mut_can_modify_transform() {
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::identity());
        scene.grid_mut(id).unwrap().transform.origin = DVec3::new(1.0, 2.0, 3.0);
        assert_eq!(
            scene.grid(id).unwrap().transform.origin,
            DVec3::new(1.0, 2.0, 3.0)
        );
    }

    #[test]
    fn chunk_size_constants_match_plan() {
        // Plan locks these values; bumping either breaks the slab
        // byte format (Z) or the worst-case chunk footprint budget
        // (XY). Pin them so a future refactor that drifts them
        // shows up in CI.
        assert_eq!(CHUNK_SIZE_XY, 128);
        assert_eq!(CHUNK_SIZE_Z, 256);
    }

    // ---- S6.0: bounding_radius + Grid::select_lod ----

    #[test]
    fn new_grid_defaults_to_always_near_lod() {
        // Byte-identity contract for the staged S6 rollout: a
        // grid built through `new` must never trigger the Mid/Far
        // branches by accident, even when bounding_radius would
        // imply otherwise.
        let g = Grid::new(GridTransform::identity());
        assert_eq!(g.lod_thresholds.r_near, f64::INFINITY);
        assert_eq!(g.lod_thresholds.r_mid, f64::INFINITY);
        assert_eq!(g.select_lod(DVec3::new(1e9, 0.0, 0.0)), Lod::Near);
    }

    #[test]
    fn bounding_radius_empty_grid_is_zero() {
        let g = Grid::new(GridTransform::identity());
        assert_eq!(g.bounding_radius(), 0.0);
    }

    #[test]
    fn bounding_radius_single_chunk_at_origin() {
        // One chunk at (0, 0, 0): bbox is [0, 128) × [0, 128) × [0, 256).
        // Half-extent = (64, 64, 128); length = sqrt(64² + 64² + 128²)
        //   = sqrt(4096 + 4096 + 16384) = sqrt(24576) ≈ 156.7747...
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::identity());
        let g = scene.grid_mut(id).unwrap();
        // Populate chunk (0, 0, 0) via the edit API.
        g.set_voxel(IVec3::new(0, 0, 0), Some(VoxColor(0x80_88_88_88)));
        let r = g.bounding_radius();
        let expected = ((64.0_f64).powi(2) * 2.0 + (128.0_f64).powi(2)).sqrt();
        assert!(
            (r - expected).abs() < 1e-9,
            "bounding_radius={r} expected={expected}"
        );
    }

    #[test]
    fn bounding_radius_grows_with_chunk_extent() {
        // Two chunks at (0,0,0) and (3,0,0): x extent is 4 chunks =
        // 512 voxels; y/z are 1 chunk each. Half-extent = (256, 64, 128);
        // length = sqrt(256² + 64² + 128²) = sqrt(65536+4096+16384)
        //        = sqrt(86016) ≈ 293.2848.
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::identity());
        let g = scene.grid_mut(id).unwrap();
        // Stamp one voxel in chunk (0,0,0).
        g.set_voxel(IVec3::new(0, 0, 0), Some(VoxColor(0x80_88_88_88)));
        // Stamp one voxel in chunk (3,0,0): grid-local x = 3*128 = 384.
        g.set_voxel(IVec3::new(384, 0, 0), Some(VoxColor(0x80_88_88_88)));
        assert_eq!(g.chunks.len(), 2);
        let r = g.bounding_radius();
        let expected = (256.0_f64.powi(2) + 64.0_f64.powi(2) + 128.0_f64.powi(2)).sqrt();
        assert!(
            (r - expected).abs() < 1e-9,
            "bounding_radius={r} expected={expected}"
        );
    }

    #[test]
    fn grid_select_lod_respects_lod_thresholds_field() {
        // Set a non-default threshold and verify the helper picks
        // the right tier for known distances.
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::at(DVec3::new(100.0, 0.0, 0.0)));
        let g = scene.grid_mut(id).unwrap();
        g.lod_thresholds = LodThresholds {
            r_near: 50.0,
            r_mid: 200.0,
            ..LodThresholds::always_near()
        };
        // Camera 25 units from grid origin → Near.
        assert_eq!(g.select_lod(DVec3::new(125.0, 0.0, 0.0)), Lod::Near);
        // 100 units → Mid.
        assert_eq!(g.select_lod(DVec3::new(200.0, 0.0, 0.0)), Lod::Mid);
        // 500 units → Far.
        assert_eq!(g.select_lod(DVec3::new(600.0, 0.0, 0.0)), Lod::Far);
    }
}
