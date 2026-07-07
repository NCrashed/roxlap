//! Streaming + procedural-generation hooks.
//!
//! S7 of the scene-graph port (see `PORTING-SCENE.md` § S7). This
//! module lands incrementally:
//!
//! - **S7.0** (this commit): the [`ChunkGenerator`] trait and the
//!   synchronous [`Grid::ensure_chunk_generated`](crate::Grid::ensure_chunk_generated) helper. Generators
//!   are plain `Box<dyn ChunkGenerator>` — no rayon, no channels,
//!   no async dispatch yet.
//! - S7.1: per-grid `StreamRadius { r_active, r_evict }` policy and
//!   `Scene::pump_streaming_sync(camera)`.
//! - S7.2: per-chunk version counter for the edit-vs-generate race.
//! - S7.3: async dispatch through a dedicated rayon pool +
//!   `crossbeam_channel`.
//! - S7.4: render integration (pending-chunk reads, billboard cache
//!   invalidation on stream-in).
//! - S7.5: `roxlap-cavegen` adapter as the first concrete generator.
//! - S7.6: streaming demo.
//!
//! The `Send + Sync` bound on [`ChunkGenerator`] is needed by S7.3
//! but is cheap to require now — generators are typically stateless
//! noise configs that already satisfy it.

use std::fmt;

use glam::{DVec3, IVec3};
use roxlap_formats::vxl::Vxl;

use crate::{CHUNK_SIZE_XY, CHUNK_SIZE_Z};

/// Pluggable per-chunk procedural generator.
///
/// `Grid` instances optionally carry a `Box<dyn ChunkGenerator>`.
/// When the streaming layer (or a direct
/// [`Grid::ensure_chunk_generated`](crate::Grid::ensure_chunk_generated)
/// call) needs a chunk that is not yet materialised, it asks the
/// generator to produce one. The returned [`Vxl`] is moved into the
/// grid's sparse chunk map at the requested index.
///
/// Generators are expected to be deterministic functions of
/// `chunk_idx` plus their own configuration: calling `generate` with
/// the same index twice should return equivalent chunks. This is
/// what makes "evict + re-stream" sound under [`crate::Grid`]'s
/// no-persistence default (see S7 scope brief, decision 5).
///
/// `Send + Sync` is required so S7.3 can dispatch generation onto a
/// background rayon pool without per-call locking. `Debug` is
/// required so [`crate::Grid`] can derive `Debug` while holding a
/// `Box<dyn ChunkGenerator>`.
pub trait ChunkGenerator: fmt::Debug + Send + Sync {
    /// Produce the chunk at `chunk_idx`. Implementations should not
    /// allocate or touch any state outside their own configuration
    /// — running this from a background thread must be safe.
    fn generate(&self, chunk_idx: IVec3) -> Vxl;

    /// Per-chunk filter consulted by [`crate::Scene::pump_streaming`]
    /// (+ the synchronous [`crate::Grid::ensure_chunk_generated`]
    /// helper) before dispatching `generate`. Returning `false`
    /// skips the chunk entirely — it never enters the grid's chunk
    /// map and `origin_chunk_z` (etc.) reflect only the indices the
    /// generator actually materialises.
    ///
    /// Used to avoid creating placeholder bedrock-only chunks for
    /// layers the generator has no real content for (e.g. the
    /// streaming-hills demo's `HillsChunkGenerator` declines
    /// `chunk_idx.z != 0` so the camera can fly above the world
    /// without triggering the S4B.6.j cross-chunk look-down
    /// limitation).
    ///
    /// Default returns `true` — pre-fix behaviour, every dispatched
    /// chunk gets generated.
    fn should_generate(&self, _chunk_idx: IVec3) -> bool {
        true
    }
}

/// Pluggable persistence for **edited** streamed chunks (QE.5a).
///
/// The [`ChunkGenerator`] contract makes evict + re-stream sound for
/// *pristine* chunks (generation is deterministic), but an edited
/// chunk (`chunk_version != 0` — the player dug, built, or a bake
/// wrote into it) would silently revert to generator output when the
/// camera walks away and comes back. A `ChunkStore` closes that hole:
///
/// - **Eviction** ([`crate::Scene::pump_streaming`] /
///   [`pump_streaming_sync`](crate::Scene::pump_streaming_sync)):
///   every evicted chunk whose version is non-zero is handed to
///   [`store`](Self::store) *before* it is dropped.
/// - **Stream-in**: before running the generator for a missing chunk,
///   the streaming layer asks [`load`](Self::load); a hit installs
///   the stored chunk (with its persisted edit version) and the
///   generator is not called. A stored chunk wins even where
///   [`ChunkGenerator::should_generate`] declines the index (edits
///   can materialise chunks the generator never would).
///
/// Implementations own the actual storage — an in-memory map, a
/// save-file region, a database. `load` runs on the streaming pool's
/// background threads under [`crate::Scene::pump_streaming`] (so
/// blocking IO is fine there) but **inline** under the synchronous
/// paths ([`pump_streaming_sync`](crate::Scene::pump_streaming_sync)
/// / [`Grid::ensure_chunk_generated`](crate::Grid::ensure_chunk_generated));
/// `store` always runs inline during the eviction pass — keep it
/// cheap (e.g. queue bytes for a writer thread).
///
/// Note [`crate::Scene::to_snapshot`] serialises only *materialised*
/// chunks — chunks living solely in the store are the host's to save
/// alongside the snapshot.
pub trait ChunkStore: fmt::Debug + Send + Sync {
    /// Persist an edited chunk that is about to be evicted. `version`
    /// is its [`crate::Grid::chunk_version`] (always non-zero here).
    fn store(&self, chunk_idx: IVec3, vxl: &Vxl, version: u64);

    /// Load a previously stored chunk. `Some((vxl, version))` installs
    /// it (restoring `version` as the chunk's edit version) instead of
    /// generating; `None` falls through to the generator.
    fn load(&self, chunk_idx: IVec3) -> Option<(Vxl, u64)>;
}

/// Per-grid streaming activity / eviction radii (S7.1).
///
/// Both values are in **grid-local voxel units** — the same scale
/// as a `GridLocalPos::voxel` coordinate. The math falls out
/// cleanly from there: chunks span fixed integer voxel extents and
/// the camera's grid-local position is also expressed in voxels.
///
/// Semantics inside [`crate::Scene::pump_streaming_sync`]:
///
/// - A chunk whose AABB-to-camera distance is `≤ r_active` MUST be
///   loaded; if absent + a generator is attached, it gets streamed
///   in via [`crate::Grid::ensure_chunk_generated`].
/// - A chunk whose AABB-to-camera distance is `> r_evict` is
///   dropped from the chunk map.
/// - Chunks in the hysteresis band `(r_active, r_evict]` are
///   neither streamed in nor evicted — they're left as-is. The
///   gap prevents a camera oscillating near a boundary from
///   thrashing generation + eviction.
///
/// The [`Default`] / [`Self::DISABLED`] value is `r_active = 0`,
/// `r_evict = ∞`: [`crate::Scene::pump_streaming_sync`] is a no-op.
/// Existing grids keep the pre-S7 "absent stays absent, present
/// stays present" behaviour until a caller opts in.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct StreamRadius {
    /// Chunks closer than this (grid-local voxels, AABB distance)
    /// are streamed in.
    pub r_active: f64,
    /// Chunks farther than this (grid-local voxels, AABB distance)
    /// are evicted. Must be `≥ r_active`.
    pub r_evict: f64,
}

impl StreamRadius {
    /// `r_active = 0`, `r_evict = ∞` — `pump_streaming_sync`
    /// never streams a chunk in or evicts one. The default for
    /// pre-S7.1 grids.
    pub const DISABLED: Self = Self {
        r_active: 0.0,
        r_evict: f64::INFINITY,
    };

    /// New radius pair. Requires `r_evict >= r_active` so the
    /// hysteresis band is well-formed (or empty when `==`).
    ///
    /// # Panics
    ///
    /// Panics if `r_evict < r_active`, if `r_active` is `NaN` or
    /// negative, or if `r_evict` is negative. NaN and negative
    /// radii are policy bugs — failing loud at construction beats
    /// silently degenerating chunk-AABB tests later.
    #[must_use]
    pub fn new(r_active: f64, r_evict: f64) -> Self {
        assert!(
            r_evict >= r_active,
            "StreamRadius: r_evict ({r_evict}) must be >= r_active ({r_active})"
        );
        assert!(
            r_active.is_finite() && r_active >= 0.0,
            "StreamRadius: r_active must be finite and >= 0, got {r_active}"
        );
        assert!(
            r_evict >= 0.0,
            "StreamRadius: r_evict must be >= 0, got {r_evict}"
        );
        Self { r_active, r_evict }
    }

    /// `true` for the [`Self::DISABLED`] sentinel pair. Lets
    /// `pump_streaming_sync` skip the per-grid pass cheaply when
    /// streaming is off.
    #[must_use]
    pub fn is_disabled(self) -> bool {
        self.r_active == 0.0 && self.r_evict == f64::INFINITY
    }
}

impl Default for StreamRadius {
    fn default() -> Self {
        Self::DISABLED
    }
}

/// Squared distance from `p_local` (grid-local f64) to the AABB of
/// the chunk at `chunk_idx`, also in grid-local voxel units.
///
/// The chunk at `(chx, chy, chz)` covers voxel-space
/// `[chx*XY, (chx+1)*XY) × [chy*XY, (chy+1)*XY) × [chz*Z, (chz+1)*Z)`.
/// Standard "clamp point to AABB then subtract" gives the closest
/// point on the box; squared length avoids a sqrt per chunk in the
/// streaming inner loop.
///
/// Returns `0.0` if `p_local` is inside the chunk.
#[must_use]
pub(crate) fn chunk_aabb_dist_sq(p_local: DVec3, chunk_idx: IVec3) -> f64 {
    let sxy = f64::from(CHUNK_SIZE_XY);
    let sz = f64::from(CHUNK_SIZE_Z);
    let lo = DVec3::new(
        f64::from(chunk_idx.x) * sxy,
        f64::from(chunk_idx.y) * sxy,
        f64::from(chunk_idx.z) * sz,
    );
    let hi = DVec3::new(lo.x + sxy, lo.y + sxy, lo.z + sz);
    let dx = (lo.x - p_local.x).max(0.0).max(p_local.x - hi.x);
    let dy = (lo.y - p_local.y).max(0.0).max(p_local.y - hi.y);
    let dz = (lo.z - p_local.z).max(0.0).max(p_local.z - hi.z);
    dx * dx + dy * dy + dz * dz
}

/// World-to-grid-local f64 transform — the same inverse-rotation
/// path as [`crate::addr::world_to_grid_local`], but skipping the
/// chunk + voxel + fract decomposition. Used by
/// [`crate::Scene::pump_streaming_sync`] which only needs the
/// continuous grid-local position to test chunk-AABB distances.
#[must_use]
pub(crate) fn world_to_grid_local_pos(world_pos: DVec3, transform: &crate::GridTransform) -> DVec3 {
    transform.rotation.inverse() * (world_pos - transform.origin)
}

// ===========================================================
// S7.3 async dispatch — native only.
// ===========================================================
//
// On wasm32 the streaming pool / channel infrastructure is
// cfg'd out and `Scene::pump_streaming` short-circuits to
// `pump_streaming_sync` — there's no rayon `ThreadPool::build`
// on `wasm32-unknown-unknown` without the wasm-bindgen-rayon
// adapter, which is a binary-side concern that the scene crate
// doesn't pull in.

#[cfg(not(target_arch = "wasm32"))]
pub(crate) use native::{ChunkResult, StreamingState};

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use super::*;
    use crate::GridId;

    /// One generator result carried back over the channel from a
    /// rayon worker to the main thread (S7.3). Kept `pub(crate)`
    /// — callers don't construct or inspect these directly.
    pub(crate) struct ChunkResult {
        pub grid_id: GridId,
        pub chunk_idx: IVec3,
        /// `Grid::chunk_version(chunk_idx)` at dispatch time. The
        /// main thread compares against the live counter on
        /// install; mismatch ⇒ an edit happened mid-generation,
        /// discard the result.
        pub version_at_dispatch: u64,
        /// The produced chunk, or `None` when the task had nothing
        /// to install (QE.5a: store miss + generator declined the
        /// index) — the drain then just clears the pending entry.
        pub vxl: Option<Vxl>,
        /// QE.5a — set when `vxl` came from the [`ChunkStore`]: the
        /// persisted edit version to restore on install.
        pub restored_version: Option<u64>,
    }

    /// Per-scene streaming state: dedicated rayon `ThreadPool` plus
    /// the `crossbeam_channel` inbox the background tasks send
    /// results into. One `StreamingState` lives on each
    /// [`crate::Scene`].
    ///
    /// The pool is built lazily on first dispatch — a scene that
    /// only ever uses [`crate::Scene::pump_streaming_sync`] pays
    /// no thread-pool overhead.
    pub(crate) struct StreamingState {
        pub thread_count: usize,
        pub pool: Option<rayon::ThreadPool>,
        pub tx: crossbeam_channel::Sender<ChunkResult>,
        pub rx: crossbeam_channel::Receiver<ChunkResult>,
    }

    impl Default for StreamingState {
        fn default() -> Self {
            // Unbounded so a slow drain (e.g. a frame stall in
            // pump_streaming) doesn't block rayon workers on send.
            // Inbox lifetime is bounded by pending_gen — there
            // can't be more in-flight messages than there are
            // chunks in pending_gen sets across all grids.
            let (tx, rx) = crossbeam_channel::unbounded();
            Self {
                thread_count: 2,
                pool: None,
                tx,
                rx,
            }
        }
    }

    impl std::fmt::Debug for StreamingState {
        // Intentionally elides the channel + pool internals — they
        // print noisily and add nothing over the summary booleans.
        // `finish_non_exhaustive` signals that omission to clippy.
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("StreamingState")
                .field("thread_count", &self.thread_count)
                .field("pool_built", &self.pool.is_some())
                .finish_non_exhaustive()
        }
    }

    impl StreamingState {
        /// Lazily construct the pool with the current
        /// `thread_count`. Idempotent — second call is a no-op
        /// when the pool already exists.
        ///
        /// # Panics
        /// Panics if rayon fails to build the pool (typically OS
        /// thread-allocation failure).
        pub fn ensure_pool(&mut self) {
            if self.pool.is_none() {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(self.thread_count)
                    .thread_name(|i| format!("roxlap-stream-{i}"))
                    .build()
                    .expect("rayon ThreadPoolBuilder");
                self.pool = Some(pool);
            }
        }

        /// Update the desired thread count. If the pool was already
        /// built, drops the old pool (which blocks until in-flight
        /// tasks finish — rayon's standard contract) and rebuilds
        /// with the new count on the next [`Self::ensure_pool`]
        /// call. The channel survives the rebuild so any results
        /// the old pool's tasks managed to send before drop are
        /// still drained by the next pump.
        ///
        /// No-op when the new count matches the current one.
        ///
        /// # Panics
        /// Panics if `n == 0`.
        pub fn set_thread_count(&mut self, n: usize) {
            assert!(n > 0, "streaming thread count must be >= 1");
            if self.thread_count == n {
                return;
            }
            self.thread_count = n;
            self.pool = None;
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::chunks::tests::voxel_is_solid;
    use crate::{Grid, GridTransform, Scene, CHUNK_SIZE_XY, CHUNK_SIZE_Z};
    use glam::DQuat;
    use roxlap_formats::color::VoxColor;
    use roxlap_formats::edit::{set_spans, Vspan};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// Test-only generator that stamps a chunk-idx-derived solid pad
    /// (one voxel at local origin) into an otherwise air chunk, and
    /// counts how many times `generate` was called.
    ///
    /// The count lets us assert idempotency: `ensure_chunk_generated`
    /// must not invoke the generator a second time once the chunk is
    /// materialised.
    #[derive(Debug)]
    pub(crate) struct StubGenerator {
        pub call_count: Arc<AtomicUsize>,
    }

    impl StubGenerator {
        pub(crate) fn new() -> Self {
            Self {
                call_count: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    impl ChunkGenerator for StubGenerator {
        fn generate(&self, chunk_idx: IVec3) -> Vxl {
            self.call_count.fetch_add(1, Ordering::Relaxed);
            // Build a fresh all-air chunk by stamping one voxel via
            // the same path as `chunks::empty_chunk_vxl`, then
            // carving everything except `(0, 0, chunk_idx.x as u8)`
            // — gives us a chunk-idx-distinguishable signature
            // without duplicating the empty-chunk builder.
            let mut g = Grid::new(GridTransform::identity());
            let mark_z = (chunk_idx.x.rem_euclid(200) as u32) % CHUNK_SIZE_Z;
            // ensure_chunk creates a stock all-air chunk; we then
            // stamp one voxel and detach the chunk.
            g.ensure_chunk(IVec3::ZERO);
            let vxl = g.chunks.remove(&IVec3::ZERO).expect("just inserted");
            let mut vxl = vxl;
            // Stamp one voxel at (0, 0, mark_z) so each chunk has a
            // unique geometric fingerprint.
            set_spans(
                &mut vxl,
                &[Vspan {
                    x: 0,
                    y: 0,
                    z0: u8::try_from(mark_z).unwrap_or(0),
                    z1: u8::try_from(mark_z).unwrap_or(0),
                }],
                Some(VoxColor(0x80_aa_bb_cc)),
            );
            vxl
        }
    }

    #[test]
    fn stub_generator_emits_distinguishable_chunks() {
        // Direct sanity check on the generator before we test the
        // helper. Two different chunk indices must produce
        // distinguishable voxel content.
        let gen = StubGenerator::new();
        let a = gen.generate(IVec3::new(0, 0, 0));
        let b = gen.generate(IVec3::new(7, 0, 0));
        assert_eq!(a.vsid, CHUNK_SIZE_XY);
        assert_eq!(b.vsid, CHUNK_SIZE_XY);
        assert!(voxel_is_solid(&a, 0, 0, 0), "chunk_idx.x=0 marks z=0");
        assert!(voxel_is_solid(&b, 0, 0, 7), "chunk_idx.x=7 marks z=7");
        assert_eq!(gen.call_count.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn ensure_chunk_generated_populates_via_generator() {
        let mut g = Grid::new(GridTransform::identity());
        let gen = StubGenerator::new();
        let counter = Arc::clone(&gen.call_count);
        g.set_generator(Some(Arc::new(gen)));

        assert_eq!(g.chunk_count(), 0);
        let idx = IVec3::new(3, 0, 0);
        let produced = g.ensure_chunk_generated(idx);
        assert!(
            produced,
            "ensure_chunk_generated returns true when it generates"
        );
        assert_eq!(g.chunk_count(), 1);
        let chunk = g.chunk(idx).expect("chunk now present");
        assert!(
            voxel_is_solid(chunk, 0, 0, 3),
            "stub generator's mark voxel for chunk_idx.x=3 missing"
        );
        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn ensure_chunk_generated_is_idempotent() {
        // Re-calling on an already-materialised chunk must not invoke
        // the generator again — the chunk's existing content stays,
        // and the call count stays at 1.
        let mut g = Grid::new(GridTransform::identity());
        let gen = StubGenerator::new();
        let counter = Arc::clone(&gen.call_count);
        g.set_generator(Some(Arc::new(gen)));

        let idx = IVec3::new(5, -2, 0);
        assert!(g.ensure_chunk_generated(idx));
        assert!(!g.ensure_chunk_generated(idx), "second call no-ops");
        assert!(!g.ensure_chunk_generated(idx), "third call still no-ops");
        assert_eq!(g.chunk_count(), 1);
        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn ensure_chunk_generated_without_generator_is_noop() {
        // A grid with no generator must leave a missing chunk
        // missing — no implicit empty-chunk allocation, since that
        // would conflict with the "implicit air" interpretation of
        // absent chunk-map entries.
        let mut g = Grid::new(GridTransform::identity());
        let idx = IVec3::new(0, 0, 0);
        assert!(g.generator.is_none());
        let produced = g.ensure_chunk_generated(idx);
        assert!(!produced, "no generator → no chunk generated");
        assert_eq!(g.chunk_count(), 0);
        assert!(g.chunk(idx).is_none());
    }

    #[test]
    fn ensure_chunk_generated_on_already_present_chunk_skips_generator() {
        // If the chunk was created via the edit API (ensure_chunk +
        // set_voxel) before the generator was attached, a later
        // ensure_chunk_generated call must not overwrite it with
        // procedurally-generated content.
        let mut g = Grid::new(GridTransform::identity());
        let idx = IVec3::new(0, 0, 0);
        // Stamp a manual voxel at chunk-local (10, 10, 10).
        g.set_voxel(IVec3::new(10, 10, 10), Some(VoxColor(0x80_11_22_33)));
        assert_eq!(g.chunk_count(), 1);

        let gen = StubGenerator::new();
        let counter = Arc::clone(&gen.call_count);
        g.set_generator(Some(Arc::new(gen)));

        let produced = g.ensure_chunk_generated(idx);
        assert!(!produced, "existing chunk not regenerated");
        assert_eq!(counter.load(Ordering::Relaxed), 0);
        // Manual voxel still there; stub's signature voxel absent.
        let chunk = g.chunk(idx).expect("manual chunk present");
        assert!(voxel_is_solid(chunk, 10, 10, 10), "manual voxel survived");
        assert!(
            !voxel_is_solid(chunk, 0, 0, 0),
            "generator's mark voxel must NOT appear"
        );
    }

    // ---- S7.1: StreamRadius + Scene::pump_streaming_sync ----

    #[test]
    fn stream_radius_disabled_is_truly_zero_infty() {
        let r = StreamRadius::DISABLED;
        assert_eq!(r.r_active, 0.0);
        assert!(r.r_evict.is_infinite() && r.r_evict.is_sign_positive());
        assert!(r.is_disabled());
        assert!(StreamRadius::default().is_disabled());
    }

    #[test]
    fn stream_radius_new_rejects_evict_below_active() {
        // Guard against accidental "r_evict < r_active" configs that
        // would evict eagerly + re-stream the same chunk every pump.
        let result = std::panic::catch_unwind(|| StreamRadius::new(200.0, 100.0));
        assert!(result.is_err(), "r_evict < r_active must panic");
    }

    #[test]
    fn chunk_aabb_dist_sq_inside_chunk_is_zero() {
        // Camera at (10, 20, 30) — well inside chunk (0, 0, 0)
        // which covers x,y in [0, 128) and z in [0, 256).
        let d = chunk_aabb_dist_sq(DVec3::new(10.0, 20.0, 30.0), IVec3::new(0, 0, 0));
        assert_eq!(d, 0.0);
    }

    #[test]
    fn chunk_aabb_dist_sq_axis_aligned() {
        // Camera at (0, 0, 0). Chunk (1, 0, 0) starts at x=128; nearest
        // point on its AABB is (128, 0, 0); squared distance 128² = 16384.
        let d = chunk_aabb_dist_sq(DVec3::ZERO, IVec3::new(1, 0, 0));
        let expected = 128.0_f64.powi(2);
        assert!((d - expected).abs() < 1e-9, "got {d}, want {expected}");
        // Chunk (0, 0, 1) — nearest face at z=256.
        let d = chunk_aabb_dist_sq(DVec3::ZERO, IVec3::new(0, 0, 1));
        let expected = 256.0_f64.powi(2);
        assert!((d - expected).abs() < 1e-9, "got {d}, want {expected}");
    }

    #[test]
    fn pump_streaming_sync_with_disabled_radius_is_noop() {
        // Disabled (default) → never generates, never evicts. This is
        // the byte-stability guarantee for any pre-S7.1 caller.
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::identity());
        let gen = StubGenerator::new();
        let counter = Arc::clone(&gen.call_count);
        scene
            .grid_mut(id)
            .unwrap()
            .set_generator(Some(Arc::new(gen)));
        // Stamp a far-away chunk that would be evicted under any
        // finite r_evict.
        scene
            .grid_mut(id)
            .unwrap()
            .set_voxel(IVec3::new(10_000, 0, 0), Some(VoxColor(0x80_11_22_33)));
        let baseline_chunks = scene.grid(id).unwrap().chunk_count();
        scene.pump_streaming_sync(DVec3::ZERO);
        assert_eq!(scene.grid(id).unwrap().chunk_count(), baseline_chunks);
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn pump_streaming_sync_streams_in_chunks_within_r_active() {
        // r_active = 200 voxels covers chunk (0,0,0) (origin) plus the
        // ring of XY neighbours whose nearest face lies within 200 of
        // origin. Chunks at chx ±1 are 128 voxels away (within); chunks
        // at chx ±2 are 256 voxels away (just outside).
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::identity());
        let gen = StubGenerator::new();
        let counter = Arc::clone(&gen.call_count);
        let g = scene.grid_mut(id).unwrap();
        g.set_generator(Some(Arc::new(gen)));
        g.stream_radius = StreamRadius::new(200.0, 400.0);
        scene.pump_streaming_sync(DVec3::ZERO);

        // Camera at world origin → grid-local (0, 0, 0). chz coverage:
        // chunk (0,0,0) Z-AABB is [0, 256); chunk (0,0,-1) Z-AABB is
        // [-256, 0). Both touch the camera point → both must stream
        // in. (0,0,1) is at z=256, more than 200 away → must NOT.
        let g = scene.grid(id).unwrap();
        let must_have = [
            IVec3::new(0, 0, 0),
            IVec3::new(1, 0, 0),
            IVec3::new(-1, 0, 0),
            IVec3::new(0, 1, 0),
            IVec3::new(0, -1, 0),
            IVec3::new(0, 0, -1),
        ];
        for idx in must_have {
            assert!(
                g.chunks.contains_key(&idx),
                "chunk {idx:?} missing from streamed set"
            );
        }
        // Diagonals at chunk (1,1,0): AABB nearest = (128,128,0);
        // dist = sqrt(128² + 128²) ≈ 181.0 < 200 → must be streamed.
        assert!(g.chunks.contains_key(&IVec3::new(1, 1, 0)));
        // Chunk (2, 0, 0): nearest face at x=256, dist=256 > 200.
        assert!(!g.chunks.contains_key(&IVec3::new(2, 0, 0)));
        // Camera chz coverage: (0,0,1) at z=256 > 200 → out.
        assert!(!g.chunks.contains_key(&IVec3::new(0, 0, 1)));

        // Counter equals number of streamed chunks.
        let streamed = g.chunk_count();
        assert_eq!(counter.load(Ordering::Relaxed), streamed);
    }

    #[test]
    fn pump_streaming_sync_idempotent_under_stationary_camera() {
        // Second pump at the same position must NOT regenerate any
        // already-loaded chunk — counter stays at the post-first-pump
        // value.
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::identity());
        let gen = StubGenerator::new();
        let counter = Arc::clone(&gen.call_count);
        let g = scene.grid_mut(id).unwrap();
        g.set_generator(Some(Arc::new(gen)));
        g.stream_radius = StreamRadius::new(180.0, 400.0);

        scene.pump_streaming_sync(DVec3::ZERO);
        let after_first = counter.load(Ordering::Relaxed);
        scene.pump_streaming_sync(DVec3::ZERO);
        let after_second = counter.load(Ordering::Relaxed);
        assert_eq!(after_first, after_second, "second pump regenerated chunks");
    }

    #[test]
    fn pump_streaming_sync_evicts_chunks_beyond_r_evict() {
        // Stream chunks within r_active=200; then move the camera far
        // away and verify r_evict trims the now-distant set.
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::identity());
        let gen = StubGenerator::new();
        let g = scene.grid_mut(id).unwrap();
        g.set_generator(Some(Arc::new(gen)));
        g.stream_radius = StreamRadius::new(200.0, 400.0);
        scene.pump_streaming_sync(DVec3::ZERO);
        let initial = scene.grid(id).unwrap().chunk_count();
        assert!(initial > 0, "expected chunks streamed in around origin");

        // Teleport camera ~10_000 voxels along +x; every chunk near
        // the old origin is now > r_evict (400) away.
        scene.pump_streaming_sync(DVec3::new(10_000.0, 0.0, 0.0));
        let g = scene.grid(id).unwrap();
        for idx in [
            IVec3::new(0, 0, 0),
            IVec3::new(1, 0, 0),
            IVec3::new(-1, 0, 0),
        ] {
            assert!(
                !g.chunks.contains_key(&idx),
                "chunk {idx:?} survived eviction after far teleport"
            );
        }
        // New chunks around the new camera position must exist.
        // Camera at x=10_000 → chunk chx = 10_000 / 128 ≈ 78.
        let cam_chx = 10_000_i32 / i32::try_from(CHUNK_SIZE_XY).unwrap();
        assert!(g.chunks.contains_key(&IVec3::new(cam_chx, 0, 0)));
    }

    #[test]
    fn pump_streaming_sync_hysteresis_band_retains_chunks() {
        // r_active = 200, r_evict = 600. A chunk that's currently
        // present and lives in the band (200 < d <= 600) is neither
        // streamed in (it's already present) nor evicted (within
        // r_evict). Move just past r_active and check the chunks that
        // were inside the now-shrunk active set stay loaded.
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::identity());
        let gen = StubGenerator::new();
        let g = scene.grid_mut(id).unwrap();
        g.set_generator(Some(Arc::new(gen)));
        g.stream_radius = StreamRadius::new(200.0, 600.0);
        scene.pump_streaming_sync(DVec3::ZERO);
        let g = scene.grid(id).unwrap();
        // A chunk at (1, 0, 0) is 128 voxels away — well inside
        // r_active. After bumping the camera to (300, 0, 0), nearest
        // face of (1, 0, 0) is x=256 → dist = max(0, 300-256) = 44 <
        // r_active, still inside r_active so it would stream in again
        // anyway. We need a chunk we KNOW will fall in the
        // hysteresis band. (-2, 0, 0): AABB nearest face x=-128;
        // initial dist at cam (0,0,0) = 128 (inside r_active). After
        // pump #1, present. After cam → (300, 0, 0): dist = 300 - (-128)
        // = 428 → in band (200, 600]. Must stay.
        let band_idx = IVec3::new(-2, 0, 0);
        // First, confirm (-2, 0, 0) was actually streamed in (its dist
        // from origin is min(128, 256) = 128 along x → 128 < 200).
        assert!(
            g.chunks.contains_key(&band_idx),
            "(-2, 0, 0) should be streamed at origin"
        );

        scene.pump_streaming_sync(DVec3::new(300.0, 0.0, 0.0));
        let g = scene.grid(id).unwrap();
        assert!(
            g.chunks.contains_key(&band_idx),
            "(-2, 0, 0) should remain in the hysteresis band"
        );
    }

    #[test]
    fn pump_streaming_sync_with_no_generator_does_not_panic() {
        // r_active > 0 but no generator: pump must skip the stream-in
        // pass cleanly and still run the evict pass. Use a manually-
        // edited chunk that's far away to verify eviction still works.
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::identity());
        let g = scene.grid_mut(id).unwrap();
        g.stream_radius = StreamRadius::new(200.0, 400.0);
        // Manual chunk at (50, 0, 0): grid-local x=50*128 = 6400, far
        // outside r_evict from origin.
        g.set_voxel(IVec3::new(50 * 128, 0, 0), Some(VoxColor(0x80_aa_bb_cc)));
        assert_eq!(scene.grid(id).unwrap().chunk_count(), 1);

        scene.pump_streaming_sync(DVec3::ZERO);
        let g = scene.grid(id).unwrap();
        // Stream-in pass was a no-op (no generator); evict pass
        // dropped the far chunk.
        assert_eq!(g.chunk_count(), 0);
    }

    #[test]
    fn pump_streaming_sync_respects_grid_rotation() {
        // Place a grid rotated 180° around Z. World camera at
        // (+10, 0, 0) maps to grid-local (-10, 0, 0). The streamed
        // chunk set must reflect that — chunk (-1, 0, 0) must be
        // present (grid-local x=-10 falls inside (-128, 0]).
        let transform = GridTransform {
            origin: DVec3::ZERO,
            rotation: DQuat::from_axis_angle(DVec3::Z, std::f64::consts::PI),
        };
        let mut scene = Scene::new();
        let id = scene.add_grid(transform);
        let gen = StubGenerator::new();
        let g = scene.grid_mut(id).unwrap();
        g.set_generator(Some(Arc::new(gen)));
        g.stream_radius = StreamRadius::new(50.0, 200.0);

        // World camera at (+10, 0, 0). After inverse 180°-Z, grid-local
        // is (-10, 0, 0).
        scene.pump_streaming_sync(DVec3::new(10.0, 0.0, 0.0));
        let g = scene.grid(id).unwrap();
        // Camera grid-local x = -10 → camera chunk chx = floor(-10/128) = -1.
        // Chunk (-1, 0, 0) AABB nearest x lies in [-128, 0); camera in
        // it → dist 0 → must be streamed.
        assert!(
            g.chunks.contains_key(&IVec3::new(-1, 0, 0)),
            "rotation not applied — camera should map to chunk (-1, 0, 0)"
        );
        // Chunk (0, 0, 0) starts at x=0; camera at x=-10 → dist=10 <
        // r_active → also streamed.
        assert!(g.chunks.contains_key(&IVec3::new(0, 0, 0)));
    }

    // ---- S7.2: chunk_version interplay with streaming ----

    #[test]
    fn ensure_chunk_generated_does_not_bump_version() {
        // A freshly-generated chunk has no user edits → version 0.
        let mut g = Grid::new(GridTransform::identity());
        let gen = StubGenerator::new();
        g.set_generator(Some(Arc::new(gen)));
        let idx = IVec3::new(2, 3, 0);
        assert!(g.ensure_chunk_generated(idx));
        assert_eq!(g.chunk_version(idx), 0);
        // chunk_versions doesn't grow either.
        assert!(g.chunk_versions.is_empty());
    }

    #[test]
    fn ensure_chunk_generated_then_edit_starts_at_version_one() {
        // Generator install + a single edit → version 1 (not 2).
        let mut g = Grid::new(GridTransform::identity());
        let gen = StubGenerator::new();
        g.set_generator(Some(Arc::new(gen)));
        let idx = IVec3::ZERO;
        g.ensure_chunk_generated(idx);
        g.set_voxel(IVec3::new(10, 10, 10), Some(VoxColor(0x80_aa_bb_cc)));
        assert_eq!(g.chunk_version(idx), 1);
    }

    #[test]
    fn pump_streaming_sync_eviction_drops_chunk_version_entry() {
        // Edit a chunk (version bumps to 1), then evict it via the
        // streaming pump. The chunk_versions map entry must also be
        // dropped so the map stays bounded.
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::identity());
        let g = scene.grid_mut(id).unwrap();
        g.set_voxel(IVec3::new(0, 0, 0), Some(VoxColor(0x80_aa_bb_cc)));
        assert_eq!(g.chunk_version(IVec3::ZERO), 1);
        g.stream_radius = StreamRadius::new(10.0, 50.0);

        scene.pump_streaming_sync(DVec3::new(10_000.0, 0.0, 0.0));
        let g = scene.grid(id).unwrap();
        assert_eq!(g.chunk_count(), 0, "chunk should have been evicted");
        assert_eq!(
            g.chunk_version(IVec3::ZERO),
            0,
            "version entry should be cleared on eviction"
        );
        assert!(g.chunk_versions.is_empty(), "map should be empty");
    }

    // ---- S7.3: async pump_streaming ----

    /// Test-only generator that pauses inside `generate` until the
    /// test explicitly releases it. Two-channel design:
    ///
    /// - `arrival_tx`: each task signals "I'm running" with its
    ///   chunk_idx before it blocks. Lets the test wait for the
    ///   dispatcher to have actually scheduled work without
    ///   sleeping.
    /// - `release_rx`: each task blocks on `recv()` here until the
    ///   test sends a `()` (or drops the matching `release_tx`,
    ///   which unblocks all pending tasks via `Err` return — the
    ///   safety net so a panicking test doesn't deadlock on
    ///   `Scene::drop` waiting for blocked tasks to finish).
    #[derive(Debug)]
    #[cfg(not(target_arch = "wasm32"))]
    struct BlockingGenerator {
        arrival_tx: crossbeam_channel::Sender<IVec3>,
        release_rx: crossbeam_channel::Receiver<()>,
        call_count: Arc<AtomicUsize>,
    }

    #[cfg(not(target_arch = "wasm32"))]
    impl ChunkGenerator for BlockingGenerator {
        fn generate(&self, chunk_idx: IVec3) -> Vxl {
            self.call_count.fetch_add(1, Ordering::Relaxed);
            let _ = self.arrival_tx.send(chunk_idx);
            // recv returns Err if the matching Sender was dropped
            // (e.g. test panicked / ended); still produce a chunk
            // so the rayon pool's drop doesn't hang.
            let _ = self.release_rx.recv();
            StubGenerator::new().generate(chunk_idx)
        }
    }

    /// Convenience: pump_streaming in a spin-loop until `grid`'s
    /// `pending_gen` is empty, releasing all gates as we go.
    /// Panics on a 5-second timeout — generation tasks shouldn't
    /// take that long even on the slowest CI.
    #[cfg(not(target_arch = "wasm32"))]
    fn pump_until_idle(
        scene: &mut Scene,
        cam: DVec3,
        grid_id: crate::GridId,
        release_tx: Option<&crossbeam_channel::Sender<()>>,
    ) {
        use std::time::{Duration, Instant};
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            scene.pump_streaming(cam);
            let idle = scene.grid(grid_id).is_none_or(|g| g.pending_gen.is_empty());
            if idle {
                return;
            }
            if Instant::now() > deadline {
                panic!("pump_until_idle: timeout with pending tasks");
            }
            // Release any blocked tasks so they can complete.
            if let Some(tx) = release_tx {
                let _ = tx.try_send(());
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    // ---- S7.4: stream-in clears billboards cache ----

    #[test]
    fn ensure_chunk_generated_invalidates_billboard_cache() {
        // Sync stream-in: a populated cache must be cleared when
        // a generator installs a new chunk — the bounding sphere
        // may have grown.
        let mut g = Grid::new(GridTransform::identity());
        let gen = StubGenerator::new();
        g.set_generator(Some(Arc::new(gen)));
        g.billboards = Some(crate::BillboardCache::new_empty(32));

        let installed = g.ensure_chunk_generated(IVec3::new(2, 0, 0));
        assert!(installed, "generator should have installed the chunk");
        assert!(
            g.billboards.is_none(),
            "ensure_chunk_generated must clear billboards on install"
        );
    }

    #[test]
    fn ensure_chunk_generated_noop_preserves_billboard_cache() {
        // No-install paths (no generator, already-present) must
        // NOT clear the cache — there was no bounding-sphere
        // change to invalidate.
        let mut g = Grid::new(GridTransform::identity());
        g.set_voxel(IVec3::new(0, 0, 0), Some(VoxColor(0x80_aa_bb_cc)));
        g.billboards = Some(crate::BillboardCache::new_empty(32));
        // No generator → no install → cache stays.
        let installed = g.ensure_chunk_generated(IVec3::new(5, 5, 0));
        assert!(!installed);
        assert!(
            g.billboards.is_some(),
            "no-generator no-op must not clear billboards"
        );
        // Already-present chunk → no install → cache stays.
        let installed = g.ensure_chunk_generated(IVec3::ZERO);
        assert!(!installed);
        assert!(
            g.billboards.is_some(),
            "already-present chunk must not clear billboards"
        );
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn pump_streaming_async_install_invalidates_billboard_cache() {
        // Async path: pump_streaming installs chunks via the drain;
        // each install must clear the cache so the next Far render
        // rebuilds with the new chunk set.
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::identity());
        let gen = StubGenerator::new();
        let g = scene.grid_mut(id).unwrap();
        g.set_generator(Some(Arc::new(gen)));
        g.stream_radius = StreamRadius::new(10.0, 200.0);
        // Stamp a sentinel cache before pumping.
        g.billboards = Some(crate::BillboardCache::new_empty(32));
        let cam = DVec3::new(64.0, 64.0, 128.0);

        pump_until_idle(&mut scene, cam, id, None);

        let g = scene.grid(id).unwrap();
        assert!(g.chunks.contains_key(&IVec3::ZERO), "chunk installed");
        assert!(
            g.billboards.is_none(),
            "async install must clear billboards"
        );
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn pump_streaming_no_install_preserves_billboard_cache() {
        // Pump with no missing chunks (all already present) must
        // NOT clear billboards. Set up: stream chunks in sync,
        // populate cache, pump again with same camera + same
        // r_active → drain has nothing → cache survives.
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::identity());
        let g = scene.grid_mut(id).unwrap();
        let gen = StubGenerator::new();
        g.set_generator(Some(Arc::new(gen)));
        g.stream_radius = StreamRadius::new(10.0, 200.0);
        let cam = DVec3::new(64.0, 64.0, 128.0);
        // First pump: streams in chunk(s).
        pump_until_idle(&mut scene, cam, id, None);
        // Stamp cache.
        scene.grid_mut(id).unwrap().billboards = Some(crate::BillboardCache::new_empty(32));
        // Second pump at same camera: no new chunks installed.
        scene.pump_streaming(cam);
        let g = scene.grid(id).unwrap();
        assert!(
            g.billboards.is_some(),
            "pump with no install should not clear billboards"
        );
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn pump_streaming_dispatches_and_installs_via_async_path() {
        // Happy path: set up a fast (non-blocking) generator,
        // configure r_active, call pump_streaming, spin until
        // idle, verify chunks installed.
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::identity());
        let gen = StubGenerator::new();
        let counter = Arc::clone(&gen.call_count);
        let g = scene.grid_mut(id).unwrap();
        g.set_generator(Some(Arc::new(gen)));
        // Camera inside chunk (0,0,0) far from edges so only one
        // chunk hits the r_active=10 ball.
        g.stream_radius = StreamRadius::new(10.0, 200.0);
        let cam = DVec3::new(64.0, 64.0, 128.0);

        pump_until_idle(&mut scene, cam, id, None);

        let g = scene.grid(id).unwrap();
        assert!(g.chunks.contains_key(&IVec3::ZERO), "chunk installed");
        assert_eq!(counter.load(Ordering::Relaxed), 1, "generator called once");
        assert!(g.pending_gen.is_empty(), "no leftover pending");
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn pump_streaming_tracks_in_flight_chunks_in_pending_gen() {
        // Verify pending_gen reflects in-flight async tasks: after
        // dispatch, pending_gen contains the chunk; after release
        // + drain, it's empty.
        let (arrival_tx, arrival_rx) = crossbeam_channel::unbounded();
        let (release_tx, release_rx) = crossbeam_channel::unbounded();
        let counter = Arc::new(AtomicUsize::new(0));
        let gen = BlockingGenerator {
            arrival_tx,
            release_rx,
            call_count: Arc::clone(&counter),
        };

        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::identity());
        let g = scene.grid_mut(id).unwrap();
        g.set_generator(Some(Arc::new(gen)));
        g.stream_radius = StreamRadius::new(10.0, 200.0);
        let cam = DVec3::new(64.0, 64.0, 128.0);

        scene.pump_streaming(cam);

        // Wait for the task to actually start.
        let arrived = arrival_rx
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("task didn't start");
        assert_eq!(arrived, IVec3::ZERO);

        // Right now the task is blocked inside `generate`.
        // pending_gen must reflect that.
        assert!(scene.grid(id).unwrap().pending_gen.contains(&IVec3::ZERO));
        assert!(scene.grid(id).unwrap().chunks.is_empty());

        // Release and drain.
        release_tx.send(()).unwrap();
        pump_until_idle(&mut scene, cam, id, Some(&release_tx));

        let g = scene.grid(id).unwrap();
        assert!(g.chunks.contains_key(&IVec3::ZERO));
        assert!(!g.pending_gen.contains(&IVec3::ZERO));
        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn pump_streaming_does_not_redispatch_in_flight_chunks() {
        // While chunk X is in pending_gen, repeated pump calls
        // must NOT enqueue another generate for X. Verified by
        // call_count staying at 1 across multiple pumps.
        let (arrival_tx, arrival_rx) = crossbeam_channel::unbounded();
        let (release_tx, release_rx) = crossbeam_channel::unbounded();
        let counter = Arc::new(AtomicUsize::new(0));
        let gen = BlockingGenerator {
            arrival_tx,
            release_rx,
            call_count: Arc::clone(&counter),
        };

        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::identity());
        let g = scene.grid_mut(id).unwrap();
        g.set_generator(Some(Arc::new(gen)));
        g.stream_radius = StreamRadius::new(10.0, 200.0);
        let cam = DVec3::new(64.0, 64.0, 128.0);

        scene.pump_streaming(cam);
        let _ = arrival_rx
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("task didn't start");

        // Pump several more times while task is blocked.
        for _ in 0..5 {
            scene.pump_streaming(cam);
        }
        assert_eq!(
            counter.load(Ordering::Relaxed),
            1,
            "in-flight chunk re-dispatched"
        );

        release_tx.send(()).unwrap();
        pump_until_idle(&mut scene, cam, id, Some(&release_tx));
        // Final post-drain assertion: still just one generate call.
        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn pump_streaming_discards_stale_result_when_chunk_edited_during_gen() {
        // Race: dispatch a chunk; while task is blocked, edit the
        // chunk via set_voxel (creates a real chunk + bumps
        // version to 1). Release. The result arrives with
        // version_at_dispatch=0 vs current=1 → must discard. The
        // chunk keeps the user edit; doesn't get overwritten by
        // generator output.
        let (arrival_tx, arrival_rx) = crossbeam_channel::unbounded();
        let (release_tx, release_rx) = crossbeam_channel::unbounded();
        let counter = Arc::new(AtomicUsize::new(0));
        let gen = BlockingGenerator {
            arrival_tx,
            release_rx,
            call_count: Arc::clone(&counter),
        };

        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::identity());
        let g = scene.grid_mut(id).unwrap();
        g.set_generator(Some(Arc::new(gen)));
        g.stream_radius = StreamRadius::new(10.0, 200.0);
        let cam = DVec3::new(64.0, 64.0, 128.0);

        scene.pump_streaming(cam);
        let _ = arrival_rx
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("task didn't start");

        // Edit while the task is blocked.
        let g = scene.grid_mut(id).unwrap();
        // A user voxel at (10, 11, 12) inside chunk (0,0,0).
        g.set_voxel(IVec3::new(10, 11, 12), Some(VoxColor(0x80_de_ad_be)));
        assert_eq!(g.chunk_version(IVec3::ZERO), 1);
        let chunk = g.chunk(IVec3::ZERO).unwrap();
        assert!(voxel_is_solid(chunk, 10, 11, 12));
        // Stub's signature voxel for chunk_idx.x=0 lives at
        // (0, 0, 0). After the user edit, before release, that
        // voxel is NOT solid (manual edit only set (10,11,12)).
        assert!(!voxel_is_solid(chunk, 0, 0, 0));

        release_tx.send(()).unwrap();
        pump_until_idle(&mut scene, cam, id, Some(&release_tx));

        // Chunk has the user voxel and NOT the generator signature.
        let g = scene.grid(id).unwrap();
        let chunk = g.chunk(IVec3::ZERO).unwrap();
        assert!(voxel_is_solid(chunk, 10, 11, 12), "user edit survived");
        assert!(
            !voxel_is_solid(chunk, 0, 0, 0),
            "stale generator output must not have overwritten the chunk"
        );
        // Generator ran exactly once before we discarded its result.
        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn pump_streaming_eviction_drops_pending_gen_entry() {
        // Dispatch a chunk; while task is blocked, move the camera
        // far enough that the chunk is past r_evict. After the
        // next pump, the chunk's pending_gen entry must be gone
        // (the eviction half of pump removes it). When the task
        // finally completes, the drain discards the result via
        // "was_pending = false".
        let (arrival_tx, arrival_rx) = crossbeam_channel::unbounded();
        let (release_tx, release_rx) = crossbeam_channel::unbounded();
        let counter = Arc::new(AtomicUsize::new(0));
        let gen = BlockingGenerator {
            arrival_tx,
            release_rx,
            call_count: Arc::clone(&counter),
        };

        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::identity());
        let g = scene.grid_mut(id).unwrap();
        g.set_generator(Some(Arc::new(gen)));
        g.stream_radius = StreamRadius::new(10.0, 50.0);
        let near_cam = DVec3::new(64.0, 64.0, 128.0);
        scene.pump_streaming(near_cam);
        let _ = arrival_rx
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("task didn't start");
        assert!(scene.grid(id).unwrap().pending_gen.contains(&IVec3::ZERO));

        // Teleport the camera 10_000 voxels along +x. Chunk
        // (0,0,0)'s nearest face at x=128 is now ~9872 away —
        // well past r_evict.
        let far_cam = DVec3::new(10_000.0, 64.0, 128.0);
        scene.pump_streaming(far_cam);
        assert!(
            !scene.grid(id).unwrap().pending_gen.contains(&IVec3::ZERO),
            "eviction should have cleared the pending entry"
        );

        // Now release the blocked task. Its result arrives with
        // was_pending = false → silently dropped.
        release_tx.send(()).unwrap();
        // Drain at the far camera; chunk (0,0,0) is not in
        // r_active there, so no re-dispatch.
        pump_until_idle(&mut scene, far_cam, id, Some(&release_tx));
        let g = scene.grid(id).unwrap();
        assert!(
            !g.chunks.contains_key(&IVec3::ZERO),
            "evicted chunk must not be re-installed by the stale result"
        );
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn pump_streaming_with_disabled_radius_is_noop() {
        // Like the sync pump's disabled-noop test, but going
        // through the async path. No dispatch, no drain, no
        // panic.
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::identity());
        let gen = StubGenerator::new();
        let counter = Arc::clone(&gen.call_count);
        scene
            .grid_mut(id)
            .unwrap()
            .set_generator(Some(Arc::new(gen)));
        // stream_radius defaults to DISABLED.
        scene.pump_streaming(DVec3::ZERO);
        let g = scene.grid(id).unwrap();
        assert!(g.chunks.is_empty());
        assert!(g.pending_gen.is_empty());
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn set_streaming_threads_zero_panics() {
        let mut scene = Scene::new();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            scene.set_streaming_threads(0);
        }));
        assert!(result.is_err(), "zero threads must panic");
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn set_streaming_threads_lazily_applied_before_first_pump() {
        // Set thread count before any pump → pool is built with
        // the new count on next pump. Verified by a successful
        // round-trip with thread_count = 1.
        let mut scene = Scene::new();
        scene.set_streaming_threads(1);
        let id = scene.add_grid(GridTransform::identity());
        let gen = StubGenerator::new();
        let g = scene.grid_mut(id).unwrap();
        g.set_generator(Some(Arc::new(gen)));
        g.stream_radius = StreamRadius::new(10.0, 200.0);
        let cam = DVec3::new(64.0, 64.0, 128.0);
        pump_until_idle(&mut scene, cam, id, None);
        assert!(scene.grid(id).unwrap().chunks.contains_key(&IVec3::ZERO));
    }

    #[test]
    fn pump_streaming_sync_eviction_clears_billboard_cache() {
        // S7.4 will hand off invalidation more carefully; for S7.1
        // we just pin that eviction nukes the cache so a future Far
        // render rebuilds it. (Cache stays untouched when nothing
        // gets evicted.)
        use crate::BillboardCache;
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::identity());
        let g = scene.grid_mut(id).unwrap();
        // Seed a single chunk to evict and a placeholder cache.
        g.set_voxel(IVec3::new(0, 0, 0), Some(VoxColor(0x80_aa_bb_cc)));
        g.billboards = Some(BillboardCache::new_empty(64));
        g.stream_radius = StreamRadius::new(10.0, 50.0);

        // Camera far enough that the chunk's nearest face > r_evict.
        scene.pump_streaming_sync(DVec3::new(10_000.0, 0.0, 0.0));
        let g = scene.grid(id).unwrap();
        assert_eq!(g.chunk_count(), 0, "chunk should have been evicted");
        assert!(g.billboards.is_none(), "billboard cache should be cleared");
    }

    // ---- QE.5a: ChunkStore persistence across evict + re-stream ----

    /// In-memory [`ChunkStore`] for the persistence tests.
    #[derive(Debug, Default)]
    struct MemStore {
        slots: std::sync::Mutex<std::collections::HashMap<IVec3, (Vxl, u64)>>,
    }

    impl ChunkStore for MemStore {
        fn store(&self, chunk_idx: IVec3, vxl: &Vxl, version: u64) {
            self.slots
                .lock()
                .unwrap()
                .insert(chunk_idx, (vxl.clone(), version));
        }
        fn load(&self, chunk_idx: IVec3) -> Option<(Vxl, u64)> {
            self.slots
                .lock()
                .unwrap()
                .get(&chunk_idx)
                .map(|(vxl, version)| (vxl.clone(), *version))
        }
    }

    #[test]
    fn chunk_store_round_trips_edits_across_evict_and_restream() {
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::identity());
        let store = Arc::new(MemStore::default());
        {
            let g = scene.grid_mut(id).unwrap();
            g.set_generator(Some(Arc::new(StubGenerator::new())));
            g.set_chunk_store(Some(store.clone()));
            g.stream_radius = StreamRadius::new(50.0, 200.0);
        }

        // Stream in around the origin, then edit chunk (0, 0, 0).
        scene.pump_streaming_sync(DVec3::ZERO);
        let edited_voxel = IVec3::new(1, 1, 40);
        let edited_version = {
            let g = scene.grid_mut(id).unwrap();
            assert!(g.chunks.contains_key(&IVec3::ZERO), "chunk streamed in");
            g.set_voxel(edited_voxel, Some(VoxColor(0x8011_2233)));
            assert!(g.voxel_solid(edited_voxel));
            let v = g.chunk_version(IVec3::ZERO);
            assert!(v > 0, "edit bumps the version");
            v
        };

        // Walk far away: the edited chunk must land in the store; a
        // pristine (version 0) neighbour must NOT be stored.
        scene.pump_streaming_sync(DVec3::new(10_000.0, 10_000.0, 0.0));
        assert!(
            !scene.grid(id).unwrap().chunks.contains_key(&IVec3::ZERO),
            "edited chunk evicted"
        );
        let stored = store.load(IVec3::ZERO).expect("edited chunk persisted");
        assert_eq!(stored.1, edited_version, "persisted edit version");
        assert!(
            store.load(IVec3::new(1, 0, 0)).is_none(),
            "pristine generator chunks are regenerable and skip the store"
        );

        // Walk back: the chunk restores from the store — edits intact,
        // version restored (not regenerated at version 0).
        scene.pump_streaming_sync(DVec3::ZERO);
        let g = scene.grid(id).unwrap();
        assert!(
            g.voxel_solid(edited_voxel),
            "player edit survived evict + re-stream"
        );
        assert_eq!(
            g.chunk_version(IVec3::ZERO),
            edited_version,
            "persisted edit version restored"
        );
    }

    #[test]
    fn chunk_store_alone_restores_without_generator() {
        // A grid with ONLY a store (no generator) still restores
        // persisted chunks — edits can outlive their generator.
        let store = Arc::new(MemStore::default());
        {
            // Seed the store from a throwaway grid.
            let mut g = Grid::new(GridTransform::identity());
            g.set_voxel(IVec3::new(2, 2, 50), Some(VoxColor(0x8044_5566)));
            let vxl = g.chunks.get(&IVec3::ZERO).expect("materialised");
            store.store(IVec3::ZERO, vxl, 7);
        }

        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::identity());
        {
            let g = scene.grid_mut(id).unwrap();
            g.set_chunk_store(Some(store));
            g.stream_radius = StreamRadius::new(50.0, 200.0);
        }
        scene.pump_streaming_sync(DVec3::ZERO);
        let g = scene.grid(id).unwrap();
        assert!(g.voxel_solid(IVec3::new(2, 2, 50)));
        assert_eq!(g.chunk_version(IVec3::ZERO), 7);
    }
}
