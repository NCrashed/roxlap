//! GPU.9 — KV6 voxel sprite loader. Returns the parsed `Kv6` so the
//! caller wraps it into a `roxlap_formats::sprite::Sprite` for the
//! sprite-splatter renderer (`roxlap_core::sprite::draw_sprite`).
//!
//! voxlap's traditional KV6 rendering is camera-facing voxel
//! splatter (one small quad per voxel, drawn with the world
//! z-buffer); doing it through the heightmap-based opticast loses
//! that aesthetic. The Grid-based prototype from the previous
//! commit has been reverted in favour of the proper splatter path.

#![allow(dead_code)]

use roxlap_formats::kv6::{parse, Kv6, ParseError};

/// `assets/coco.kv6` baked into the binary at build time.
pub const COCO_KV6_BYTES: &[u8] = include_bytes!("../../../assets/coco.kv6");

/// Convenience: decode the embedded `coco.kv6` once at startup.
///
/// # Errors
/// Mirrors [`roxlap_formats::kv6::parse`]'s [`ParseError`].
pub fn load_coco_kv6() -> Result<Kv6, ParseError> {
    parse(COCO_KV6_BYTES)
}
