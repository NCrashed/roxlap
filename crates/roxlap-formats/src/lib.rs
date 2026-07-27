//! Voxlap on-disk formats and data manipulation.
//!
//! New to roxlap? **[The roxlap book](https://ncrashed.github.io/roxlap/)**
//! is the guide — its asset-pipeline chapter tours every format here;
//! this page is the API reference.
//!
//! Parsers for `.vxl` (heightmap + slab voxel columns), `.kv6` / `.kvx`
//! (sprite voxel data), and `.kfa` (kv6 transform / animation). Lands
//! across the R2.* sub-substages of `PORTING-RUST.md`:
//!
//! - R2.1: `.kvx`
//! - R2.2: `.kv6`
//! - R2.3: `.vxl`
//! - R2.4: `.kfa`
//!
//! [`edit`] hosts voxel-edit primitives (delslab/insslab/expandrle/
//! compilerle/`ScumCtx`) and high-level wrappers (`set_spans`,
//! `set_cube`, `set_sphere`, `set_rect`). They live with the data
//! they manipulate; rendering stays in `roxlap-core`.

mod bytes;

/// Rigged-character container (`.rkc`) — meshes + skeleton + clips, the
/// on-disk form of a complete animated voxel character. Built on
/// [`kfa`] / [`kv6`] / [`sprite`].
pub mod character;
pub mod color;
pub mod edit;
/// Voxlap's `univec[256]` surface-normal direction table + the
/// `normal → dir` quantiser ([`equivec::nearest_dir`]). Lives here (not
/// roxlap-core) so [`kv6`] model builders can fill per-voxel `dir`
/// without a circular dependency; roxlap-core re-exports it.
pub mod equivec;
/// CT.3 — shared driver for the `edit` fuzz target; also exercised by
/// the stable-CI seed tests. Not part of the public API surface.
#[doc(hidden)]
pub mod fuzz_driver;
/// Animated-GIF → [`voxel_clip::VoxelClip`] importer for Doom-style
/// billboard sprites (stage BB). Feature-gated behind `gif`; see
/// [`gif_import`] + `PORTING-BILLBOARD.md`.
#[cfg(feature = "gif")]
pub mod gif_import;
pub mod kfa;
pub mod kv6;
pub mod kvx;
/// Voxel materials — per-voxel opacity + blend mode (alpha / additive) for
/// transparent voxels (smoke, glass, water, spell glows). See [`material`]
/// + `PORTING-TRANSPARENCY.md`.
pub mod material;
pub mod palette;
/// PNG-sequence / APNG → [`voxel_clip::VoxelClip`] importer for billboard
/// sprites (stage BB). Feature-gated behind `png`; see [`png_import`].
#[cfg(feature = "png")]
pub mod png_import;
/// Shared image → flat-voxel-slab voxelization for the billboard importers
/// (BB). Internal; the `gif`/`png` decoders feed composited RGBA frames in.
#[cfg(any(feature = "gif", feature = "png"))]
mod slab;
pub mod sprite;
/// MagicaVoxel `.vox` importer (QE.6a) — models + palette → [`kv6::Kv6`]
/// sprite models. The bridge from the industry-standard voxel editor.
pub mod vox;
/// Animated voxel-sprite clips (`.rvc`) — keyframe + diff "GIF/MP4 for
/// voxel models" for effects (flame, spells). Frames use the GPU sprite
/// model's dense-column layout; see [`voxel_clip`] + `PORTING-VOXEL-CLIP.md`.
pub mod voxel_clip;
pub mod vxl;
pub mod xform;

pub use color::{OverlayColor, Rgb, VoxColor};
pub use material::{material_for_color, BlendMode, Material, MaterialTable};
pub use palette::Rgb6;
