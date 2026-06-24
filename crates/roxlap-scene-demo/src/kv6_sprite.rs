//! GPU.9 — KV6 voxel sprite loader. Returns the parsed `Kv6` so the
//! caller wraps it into a `roxlap_formats::sprite::Sprite`, which the
//! renderer draws via the clean-room DDA sprite raycaster
//! (`roxlap_core::dda_sprite::draw_sprite_dda`) — a per-pixel ray cast
//! through the KV6 that depth-composites against the shared z-buffer.

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
