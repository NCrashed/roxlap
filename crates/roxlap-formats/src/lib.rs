//! Voxlap on-disk format parsers.
//!
//! Parsers for `.vxl` (heightmap + slab voxel columns), `.kv6` / `.kvx`
//! (sprite voxel data), and `.kfa` (kv6 transform / animation). Lands
//! across the R2.* sub-substages of `PORTING-RUST.md`:
//!
//! - R2.1: `.kvx` (this commit)
//! - R2.2: `.kv6`
//! - R2.3: `.vxl`
//! - R2.4: `.kfa`

pub mod kvx;
