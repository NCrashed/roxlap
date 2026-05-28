//! Adapter wrapping `roxlap-cavegen`'s [`Generator`] presets as a
//! [`crate::ChunkGenerator`] (S7.5).
//!
//! The cavegen crate generates one big `vsid × vsid × MAXZDIM`
//! [`Vxl`] from a [`CaveParams`] seed. The scene-graph layer wants
//! per-chunk generation: one [`Vxl`] of size
//! `CHUNK_SIZE_XY × CHUNK_SIZE_XY × CHUNK_SIZE_Z` per `(chx, chy,
//! chz)`. [`CaveChunkGenerator`] bridges the two by:
//!
//! 1. **Per-chunk seed derivation** (FNV-1a hash of base_seed +
//!    chunk_idx.xy) — each chunk gets its own Worley cell layout.
//! 2. **Single-z-slab cave** — caves live in `chz = 0` (world z in
//!    `[0, 256)`). Other `chz` layers return an empty bedrock-only
//!    [`Vxl`] (= implicit air).
//!
//! ## Known limitation: chunk-boundary seams
//!
//! Because each chunk gets an independent seed pool, the Worley
//! cells don't line up across chunk boundaries — a corridor that
//! cuts through chunk `(0, 0)` stops at the boundary of `(1, 0)`,
//! where a fresh cave begins. The result still streams cleanly +
//! looks "cave-like" everywhere, but the seams are visible.
//!
//! S7.5.b will refactor cavegen to support neighbour-aware seed
//! pools (each chunk's classification reads seeds from a 3×3 pool
//! of neighbours), erasing the seams. Deferred per [[s7-scope]]
//! risk R-S7.4's "start with (a), refactor if needed" guidance.
//!
//! [`Vxl`]: roxlap_formats::vxl::Vxl
//! [`Generator`]: roxlap_cavegen::Generator
//! [`CaveParams`]: roxlap_cavegen::CaveParams

use std::fmt;

use glam::IVec3;
use roxlap_cavegen::{CaveParams, Generator};
use roxlap_formats::vxl::Vxl;

use crate::chunks::empty_chunk_vxl;
use crate::{ChunkGenerator, CHUNK_SIZE_XY};

/// Wrap a [`Generator`] (typically
/// [`roxlap_cavegen::BlueCaveGenerator`] or
/// [`roxlap_cavegen::MagCaveGenerator`]) as a
/// [`ChunkGenerator`]. Each `(chx, chy)` gets a chunk-derived
/// seed; `chz != 0` is implicit air.
///
/// The inner generator is consumed at construction; the wrapper
/// owns it for the lifetime of the adapter. Generators are
/// expected to be cheap to clone (the cavegen presets are unit
/// structs, so this is free).
pub struct CaveChunkGenerator<G> {
    inner: G,
    base_params: CaveParams,
}

impl<G> fmt::Debug for CaveChunkGenerator<G>
where
    G: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CaveChunkGenerator")
            .field("inner", &self.inner)
            .field("base_seed", &self.base_params.seed)
            .field("seed_count", &self.base_params.seed_count)
            .finish()
    }
}

impl<G> CaveChunkGenerator<G> {
    /// New adapter wrapping `inner` with `base_params` as the
    /// template for per-chunk parameter derivation.
    ///
    /// The `base_params.seed` is XOR'd with the chunk index via
    /// FNV-1a to produce the per-chunk seed; other fields
    /// (`seed_count`, `air_ratio`, `anisotropy`, etc.) are passed
    /// through unchanged.
    pub fn new(inner: G, base_params: CaveParams) -> Self {
        Self { inner, base_params }
    }
}

/// Derive a deterministic per-chunk seed from a base seed + chunk
/// index. FNV-1a 64-bit so the hash is well-distributed even for
/// small chunk-index deltas (chunk `(0, 0)` vs `(1, 0)` produce
/// completely different output).
///
/// Only `chunk_idx.x` and `chunk_idx.y` participate — `chz` is
/// gated upstream (only `chz == 0` reaches the cave path).
#[must_use]
pub(crate) fn derive_chunk_seed(base_seed: u64, chunk_idx: IVec3) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut h = FNV_OFFSET ^ base_seed;
    for byte in chunk_idx.x.to_le_bytes() {
        h ^= u64::from(byte);
        h = h.wrapping_mul(FNV_PRIME);
    }
    for byte in chunk_idx.y.to_le_bytes() {
        h ^= u64::from(byte);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

impl<G> ChunkGenerator for CaveChunkGenerator<G>
where
    G: Generator<Params = CaveParams> + fmt::Debug + Send + Sync + 'static,
{
    fn generate(&self, chunk_idx: IVec3) -> Vxl {
        if chunk_idx.z != 0 {
            // Cave lives in the chz = 0 z-slab; other layers are
            // implicit air (just the bedrock placeholder, which
            // the renderer treats as air via `treat_z_max_as_air`).
            return empty_chunk_vxl();
        }
        let params = CaveParams {
            seed: derive_chunk_seed(self.base_params.seed, chunk_idx),
            ..self.base_params
        };
        self.inner.generate(&params, CHUNK_SIZE_XY)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunks::tests::voxel_is_solid;
    use crate::{
        Grid, GridTransform, Scene, StreamRadius, CHUNK_SIZE_XY as CXY, CHUNK_SIZE_Z as CZ,
    };
    use glam::DVec3;
    use roxlap_cavegen::{BlueCaveGenerator, MagCaveGenerator};
    use std::sync::Arc;

    #[test]
    fn derive_chunk_seed_distinct_for_neighbours() {
        let s0 = derive_chunk_seed(7, IVec3::new(0, 0, 0));
        let sx = derive_chunk_seed(7, IVec3::new(1, 0, 0));
        let sy = derive_chunk_seed(7, IVec3::new(0, 1, 0));
        let snx = derive_chunk_seed(7, IVec3::new(-1, 0, 0));
        // All four must be pairwise different — a small chunk-idx
        // delta with a 64-bit hash collides at probability ~2^-64.
        assert_ne!(s0, sx);
        assert_ne!(s0, sy);
        assert_ne!(s0, snx);
        assert_ne!(sx, sy);
    }

    #[test]
    fn derive_chunk_seed_ignores_z() {
        // chz isn't part of the cave layer; same xy + different z
        // must produce the same hash (chz != 0 is short-circuited
        // before we hit this hash anyway, but contract is explicit).
        let a = derive_chunk_seed(7, IVec3::new(3, -2, 0));
        let b = derive_chunk_seed(7, IVec3::new(3, -2, 5));
        assert_eq!(a, b);
    }

    #[test]
    fn adapter_returns_chunk_sized_vxl() {
        let gen = CaveChunkGenerator::new(
            BlueCaveGenerator,
            CaveParams {
                seed_count: 16,
                ..BlueCaveGenerator::default_params()
            },
        );
        let vxl = gen.generate(IVec3::ZERO);
        assert_eq!(vxl.vsid, CXY, "adapter must return chunk-VSID output");
    }

    #[test]
    fn adapter_is_deterministic_per_chunk_idx() {
        let mk = || {
            CaveChunkGenerator::new(
                BlueCaveGenerator,
                CaveParams {
                    seed_count: 16,
                    ..BlueCaveGenerator::default_params()
                },
            )
        };
        let g1 = mk();
        let g2 = mk();
        let a = g1.generate(IVec3::new(3, -2, 0));
        let b = g2.generate(IVec3::new(3, -2, 0));
        // Same chunk_idx + same base_params → byte-identical Vxl
        // (cavegen's docs guarantee this).
        assert_eq!(a.column_offset.as_ref(), b.column_offset.as_ref());
        assert_eq!(a.data.as_ref(), b.data.as_ref());
    }

    #[test]
    fn adapter_different_chunks_yield_different_output() {
        let gen = CaveChunkGenerator::new(
            BlueCaveGenerator,
            CaveParams {
                seed_count: 16,
                ..BlueCaveGenerator::default_params()
            },
        );
        let a = gen.generate(IVec3::new(0, 0, 0));
        let b = gen.generate(IVec3::new(1, 0, 0));
        // Different chunk seed → at least some columns differ.
        let mut differing = 0;
        for col in 0..(CXY * CXY) {
            if a.column_data(col as usize) != b.column_data(col as usize) {
                differing += 1;
            }
        }
        assert!(
            differing > 0,
            "adjacent chunks should produce differing column data"
        );
    }

    #[test]
    fn adapter_chz_nonzero_returns_implicit_air() {
        // chz != 0 → bedrock-only chunk (the standard scene
        // empty-chunk shape). Verify by checking a sample of
        // mid-z voxels are air; bedrock at z=255 is solid.
        let gen = CaveChunkGenerator::new(BlueCaveGenerator, BlueCaveGenerator::default_params());
        let vxl = gen.generate(IVec3::new(0, 0, 1));
        for &(x, y, z) in &[(0u32, 0u32, 0u32), (50, 60, 100), (CXY - 1, CXY - 1, 200)] {
            assert!(
                !voxel_is_solid(&vxl, x, y, z),
                "({x},{y},{z}) should be air for chz=1"
            );
        }
        // Bedrock placeholder still present.
        assert!(voxel_is_solid(&vxl, 0, 0, CZ - 1));

        // Negative chz also implicit-air.
        let vxl_neg = gen.generate(IVec3::new(0, 0, -3));
        assert!(!voxel_is_solid(&vxl_neg, 50, 60, 100));
    }

    #[test]
    fn adapter_chunks_have_mixed_air_and_solid_in_cave_layer() {
        // chz=0 must produce real cave content — both air and
        // solid voxels, not pathological all-one-or-the-other.
        let gen = CaveChunkGenerator::new(
            BlueCaveGenerator,
            CaveParams {
                seed_count: 32,
                ..BlueCaveGenerator::default_params()
            },
        );
        let vxl = gen.generate(IVec3::ZERO);
        let mut any_air = false;
        let mut any_solid_above_bedrock = false;
        for y in (0..CXY).step_by(16) {
            for x in (0..CXY).step_by(16) {
                for z in (0..(CZ - 1)).step_by(16) {
                    if voxel_is_solid(&vxl, x, y, z) {
                        any_solid_above_bedrock = true;
                    } else {
                        any_air = true;
                    }
                }
            }
        }
        assert!(any_air, "cave should contain air voxels");
        assert!(
            any_solid_above_bedrock,
            "cave should contain solid voxels above bedrock"
        );
    }

    #[test]
    fn adapter_works_with_mag_preset() {
        // Sanity that the generic wrapper accepts MagCaveGenerator
        // identically — type-level check + a deterministic round-trip.
        let gen = CaveChunkGenerator::new(
            MagCaveGenerator,
            CaveParams {
                seed_count: 16,
                ..MagCaveGenerator::default_params()
            },
        );
        let a = gen.generate(IVec3::ZERO);
        let b = gen.generate(IVec3::ZERO);
        assert_eq!(a.column_offset.as_ref(), b.column_offset.as_ref());
        assert_eq!(a.data.as_ref(), b.data.as_ref());
    }

    #[test]
    fn adapter_integrates_with_pump_streaming_sync() {
        // End-to-end: register an adapter on a Grid, set
        // stream_radius, call pump_streaming_sync, verify chunks
        // installed + their content is cavegen-shaped (mixed
        // air/solid in chz=0; implicit-air in chz=1).
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::identity());
        let adapter = CaveChunkGenerator::new(
            BlueCaveGenerator,
            CaveParams {
                seed_count: 16,
                ..BlueCaveGenerator::default_params()
            },
        );
        let g: &mut Grid = scene.grid_mut(id).unwrap();
        g.set_generator(Some(Arc::new(adapter)));
        // r_active = 150 from camera at (64, 64, 100) → covers chunk
        // (0, 0, 0) only on chz=0 (z=100 → chz=0; chz=1 face at z=256,
        // dist=156 > 150; chz=-1 face at z=0... camera is at z=100,
        // inside chz=0). So just one chunk streams.
        g.stream_radius = StreamRadius::new(150.0, 300.0);
        scene.pump_streaming_sync(DVec3::new(64.0, 64.0, 100.0));

        let g = scene.grid(id).unwrap();
        let vxl = g
            .chunk(IVec3::ZERO)
            .expect("chunk (0,0,0) should have streamed");
        // Quick sanity: the streamed chunk has both air and solid.
        let mut any_air = false;
        let mut any_solid = false;
        for &(x, y, z) in &[
            (40_u32, 40, 50),
            (80, 80, 100),
            (20, 90, 150),
            (100, 30, 200),
        ] {
            if voxel_is_solid(vxl, x, y, z) {
                any_solid = true;
            } else {
                any_air = true;
            }
        }
        assert!(any_air, "streamed cave should have air voxels");
        assert!(any_solid, "streamed cave should have solid voxels");
    }
}
