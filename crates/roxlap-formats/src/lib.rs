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
pub mod palette;
pub mod sprite;
pub mod vxl;
pub mod xform;

pub use palette::Rgb6;
