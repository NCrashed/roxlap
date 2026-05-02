//! Procedural cave generation for the roxlap voxel engine.
//!
//! Builds voxlap-format `Vxl` worlds from a Worley-noise + Perlin-
//! overlay classification scheme. The crate ships two presets
//! ([`BlueCaveGenerator`], [`MagCaveGenerator`]) tuned to match
//! Ken Silverman + Tom Dobrowolski's 2003 "Justfly" reference
//! screenshots (`caveblue2m.jpg`, `cavemag3m.jpg`).
//!
//! # Pipeline
//!
//! 1. [`Generator::generate`] consumes [`CaveParams`] (seed +
//!    shape knobs) and produces a [`Vxl`].
//! 2. Internally, generators sample a dense `(VSID × VSID × MAXZDIM)`
//!    voxel mask + colour grid via Worley distance classification +
//!    Perlin overlay.
//! 3. [`pack_dense_grid_to_vxl`] folds the dense grid into voxlap's
//!    slab-RLE column format — much faster than feeding voxels
//!    through the [`set_spans`] edit pipeline for whole-world
//!    creation.
//!
//! [`set_spans`]: ../roxlap_formats/edit/fn.set_spans.html

pub use roxlap_formats::vxl::Vxl;

mod pack;
mod perlin;
mod presets;
mod rng;
mod worley;

pub use pack::{pack_dense_grid_to_vxl, MAXZDIM};
pub use perlin::PerlinNoise3D;
pub use presets::{BlueCaveGenerator, MagCaveGenerator};
pub use worley::{
    classify_voxel, classify_voxel_with_perlin, place_seeds, worley_classify_grid, Seed,
};

/// Parameters for the procedural cave generators.
///
/// All fields are public so callers can override per-preset defaults.
/// Generators are expected to fall back to their preset-specific
/// defaults when the user constructs a `CaveParams` via
/// [`CaveParams::default`] — that returns the "blue cave" baseline
/// from `caveblue2m.jpg`. Other presets (e.g.,
/// [`MagCaveGenerator`]) supply their own defaults via factory
/// constructors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CaveParams {
    /// RNG seed. Same seed + same params + same code → byte-stable
    /// output. Promotes determinism across runs and platforms.
    pub seed: u64,
    /// Number of Worley seed points placed in the volume. Controls
    /// the cave's chamber-vs-corridor texture.
    pub seed_count: usize,
    /// Fraction of seeds tagged "air" (the rest are "solid").
    /// Higher values produce more open caves.
    pub air_ratio: f32,
    /// Per-seed distance scaling factor — voxlap `CaveTex`'s "shape"
    /// parameter. 1.0 produces isotropic Worley cells; > 1.0
    /// elongates them along an arbitrary axis.
    pub anisotropy: f32,
    /// Number of Perlin-noise octaves overlaid on the air/solid
    /// boundary to break up Worley's clean facets.
    pub perlin_octaves: u32,
    /// Magnitude of the Perlin overlay (added to `d_a` in the
    /// classification step). Larger values give more organic
    /// surfaces; 0 disables overlay.
    pub perlin_amplitude: f32,
}

impl Default for CaveParams {
    /// "Blue cave" baseline matching `caveblue2m.jpg`.
    fn default() -> Self {
        Self {
            seed: 7,
            seed_count: 128,
            air_ratio: 0.5,
            anisotropy: 1.0,
            perlin_octaves: 3,
            perlin_amplitude: 0.15,
        }
    }
}

/// Procedural world generator. Implementors produce a [`Vxl`] from
/// [`CaveParams`] (or their own typed parameter struct).
///
/// `generate` is expected to be deterministic in its parameters: the
/// same `params + vsid` MUST produce a byte-stable [`Vxl`] across
/// runs (within the same toolchain — like the rasterizer's tests,
/// cross-CPU FP determinism is best-effort).
pub trait Generator {
    /// Parameter type. Most generators alias this to [`CaveParams`].
    type Params;

    /// Build a `vsid × vsid × MAXZDIM` voxel world per `params`.
    /// `vsid` MUST be a power of two and `>= 4` (smaller worlds
    /// don't have room for the central air carve).
    fn generate(&self, params: &Self::Params, vsid: u32) -> Vxl;
}

// `BlueCaveGenerator` (CD.6) and `MagCaveGenerator` (CD.7) live in
// `presets.rs`; re-exported above.
