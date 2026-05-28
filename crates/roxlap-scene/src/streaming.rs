//! Streaming + procedural-generation hooks.
//!
//! S7 of the scene-graph port (see `PORTING-SCENE.md` § S7). This
//! module lands incrementally:
//!
//! - **S7.0** (this commit): the [`ChunkGenerator`] trait and the
//!   synchronous [`Grid::ensure_chunk_generated`] helper. Generators
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

use glam::IVec3;
use roxlap_formats::vxl::Vxl;

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
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::chunks::tests::voxel_is_solid;
    use crate::{Grid, GridTransform, CHUNK_SIZE_XY, CHUNK_SIZE_Z};
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
                Some(0x80_aa_bb_cc),
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
        g.set_generator(Some(Box::new(gen)));

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
        g.set_generator(Some(Box::new(gen)));

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
        g.set_voxel(IVec3::new(10, 10, 10), Some(0x80_11_22_33));
        assert_eq!(g.chunk_count(), 1);

        let gen = StubGenerator::new();
        let counter = Arc::clone(&gen.call_count);
        g.set_generator(Some(Box::new(gen)));

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
}
