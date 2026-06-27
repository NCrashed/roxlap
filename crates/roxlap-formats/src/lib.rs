//! Voxlap on-disk formats and data manipulation.
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
pub mod edit;
/// Voxlap's `univec[256]` surface-normal direction table + the
/// `normal → dir` quantiser ([`equivec::nearest_dir`]). Lives here (not
/// roxlap-core) so [`kv6`] model builders can fill per-voxel `dir`
/// without a circular dependency; roxlap-core re-exports it.
pub mod equivec;
pub mod kfa;
pub mod kv6;
pub mod kvx;
/// Voxel materials — per-voxel opacity + blend mode (alpha / additive) for
/// transparent voxels (smoke, glass, water, spell glows). See [`material`]
/// + `PORTING-TRANSPARENCY.md`.
pub mod material;
pub mod palette;
pub mod sprite;
/// Animated voxel-sprite clips (`.rvc`) — keyframe + diff "GIF/MP4 for
/// voxel models" for effects (flame, spells). Frames use the GPU sprite
/// model's dense-column layout; see [`voxel_clip`] + `PORTING-VOXEL-CLIP.md`.
pub mod voxel_clip;
pub mod vxl;
pub mod xform;

pub use material::{material_for_color, BlendMode, Material, MaterialTable};
pub use palette::Rgb6;
