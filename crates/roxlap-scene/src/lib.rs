//! roxlap scene-graph layer — many independent chunked voxel
//! grids in a single 3D scene.
//!
//! See `PORTING-SCENE.md` at the workspace root for the substage
//! roadmap. This crate is the layer **above** voxlap's per-chunk
//! renderer (`roxlap-core`): a [`Scene`] holds a sparse set of
//! [`Grid`]s, each with its own f64 world position + arbitrary 3D
//! rotation. Future stages will add per-grid raycast composition
//! (S3), cross-chunk gline within a grid (S4), per-grid rotation
//! (S5), far-LOD billboards / planet proxies (S6), and streaming +
//! procedural generation (S7).
//!
//! S2.0 lands the **type skeleton + grid registration only**.
//! S2.1 adds the [`addr`] module — world ↔ grid-local ↔ chunk +
//! voxel-in-chunk decomposition, the canonical f64↔i32 boundary
//! helper called out by risk R5 in `PORTING-SCENE.md`. S2.2 adds
//! the [`chunks`] module (sparse storage with on-demand chunk
//! allocation) and the [`Grid`] edit API ([`Grid::set_voxel`],
//! [`Grid::set_rect`], [`Grid::set_sphere`]) which decompose
//! multi-chunk operations and delegate to
//! [`roxlap_formats::edit`]. S2.3 adds the [`snapshot`] module —
//! a serde-friendly view of the scene that round-trips through
//! `Serialize` + `Deserialize` (chunks encode via
//! [`roxlap_formats::vxl::serialize`] / [`parse`]). Rendering
//! composition is still owed (S3+).
//!
//! [`parse`]: roxlap_formats::vxl::parse

pub mod addr;
pub mod billboard;
pub mod cavegen;
pub mod chunks;
pub mod edit;
pub mod lod;
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
pub use lod::{select_lod, Lod, LodThresholds};
pub use streaming::{ChunkGenerator, StreamRadius};

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

/// f64 world placement of one grid: position + orientation.
///
/// `origin` is the grid's local-space origin in world coords —
/// chunk `(0, 0, 0)`'s `(0, 0, 0)` voxel maps to
/// `origin + rotation * vec3(0, 0, 0)` (i.e. just `origin`).
/// Voxel size is fixed at 1 world unit / voxel for v1.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GridTransform {
    pub origin: DVec3,
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
    pub grid: GridId,
    pub chunk: IVec3,
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
    /// Streaming activity / eviction radii used by
    /// [`Scene::pump_streaming_sync`] (S7.1). Defaults to
    /// [`StreamRadius::DISABLED`] so existing grids see no change
    /// in behaviour until the caller opts in.
    pub stream_radius: StreamRadius,
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
    pub chunk_versions: HashMap<IVec3, u64>,
    /// In-flight background generation tasks (S7.3).
    ///
    /// Populated by [`Scene::pump_streaming`] when it dispatches a
    /// generator call onto the streaming rayon pool, drained when
    /// the corresponding [`ChunkResult`] is received and processed
    /// (either installed or discarded). The set is consulted to
    /// avoid re-dispatching the same chunk while a previous task
    /// is still running.
    ///
    /// Stays empty when only the synchronous
    /// [`Scene::pump_streaming_sync`] is used — that path generates
    /// inline on the calling thread.
    ///
    /// [`ChunkResult`]: streaming::ChunkResult
    pub pending_gen: HashSet<IVec3>,
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
            stream_radius: StreamRadius::DISABLED,
            chunk_versions: HashMap::new(),
            pending_gen: HashSet::new(),
        }
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
    /// This is the synchronous S7.0 path. S7.3 will add an async
    /// counterpart that dispatches the generator call to a
    /// dedicated rayon pool and installs the result on the next
    /// `pump_streaming` call.
    pub fn ensure_chunk_generated(&mut self, chunk_idx: IVec3) -> bool {
        if self.chunks.contains_key(&chunk_idx) {
            return false;
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
            grid.chunks.insert(result.chunk_idx, result.vxl);
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
    /// Both passes use the f64 grid-local position so rotation
    /// + non-axis-aligned grids stream and evict correctly. The
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
    if radius.r_active > 0.0 && grid.generator.is_some() {
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
        grid.chunks.remove(idx);
        // S7.2: keep chunk_versions in sync with chunks so the
        // map stays bounded. A future re-stream of the same idx
        // restarts at 0 — that's fine because any in-flight
        // gen-result tagged with the pre-eviction version is
        // unreachable (no chunk to install onto) and gets
        // discarded by the new "version still 0" check anyway.
        grid.chunk_versions.remove(idx);
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
    let Some(generator) = grid.generator.as_ref().map(Arc::clone) else {
        return;
    };
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
        if !generator.should_generate(idx) {
            return;
        }
        grid.pending_gen.insert(idx);
        let version_at_dispatch = grid.chunk_version(idx);
        let tx_clone = tx.clone();
        let gen_clone = Arc::clone(&generator);
        pool.spawn(move || {
            let vxl = gen_clone.generate(idx);
            // Send is non-blocking on unbounded channel; if the
            // receiver was dropped (Scene drop), the send fails
            // silently — that's fine.
            let _ = tx_clone.send(streaming::ChunkResult {
                grid_id,
                chunk_idx: idx,
                version_at_dispatch,
                vxl,
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
        g.set_voxel(IVec3::new(0, 0, 0), Some(0x80_88_88_88));
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
        g.set_voxel(IVec3::new(0, 0, 0), Some(0x80_88_88_88));
        // Stamp one voxel in chunk (3,0,0): grid-local x = 3*128 = 384.
        g.set_voxel(IVec3::new(384, 0, 0), Some(0x80_88_88_88));
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
