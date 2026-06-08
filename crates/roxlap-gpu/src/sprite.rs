//! GPU.9 — KV6 sprite voxel splatter.
//!
//! Each `Sprite` (from `roxlap_formats::sprite`) is decomposed into
//! a flat list of world-space voxels (one per kv6 voxel) packed
//! into a storage buffer. The sprite-DDA compute shader projects
//! each voxel to screen and depth-tests against the scene depth
//! buffer; winners write their colour to the scene's colour
//! storage texture.
//!
//! `Sprite::p` is the pivot's world position; `(xpiv, ypiv, zpiv)`
//! of the kv6 maps to that point. Each voxel (vx, vy, vz) becomes
//! `p + (vx - xpiv) * s + (vy - ypiv) * h + (vz - zpiv) * f` in
//! world space. Done on CPU once per `set_sprites` call so the GPU
//! buffer is camera-independent.

#![allow(clippy::cast_precision_loss, clippy::missing_panics_doc)]

use bytemuck::{Pod, Zeroable};
use roxlap_formats::sprite::Sprite;

/// One voxel as it lives on the GPU: world-space position, colour,
/// and the voxel's world-space edge length (the splatter sizes its
/// screen square from this so scaled sprites cover their gaps).
///
/// Layout: WGSL aligns the leading `vec3<f32>` to 16 bytes, so the
/// struct's align is 16 and its array stride rounds up to 32. The
/// Rust struct is laid out `repr(C)` to exactly 32 bytes
/// (`world_pos`@0, `color`@12, `world_size`@16, `_pad`@20) so the
/// field offsets and stride match WGSL's `SpriteVoxel`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct SpriteVoxel {
    pub world_pos: [f32; 3],
    pub color: u32,
    /// World-space edge length of this voxel — `max(|s|, |h|, |f|)`
    /// of the sprite's basis. 1.0 for a unit `axis_aligned` sprite.
    pub world_size: f32,
    _pad: [f32; 3],
}

/// Decompose `sprite` into world-space voxels. Walks the KV6's
/// `xlen` / `ylen` counters (same shape as the CPU sprite renderer).
#[must_use]
pub fn sprite_voxels_world_space(sprite: &Sprite) -> Vec<SpriteVoxel> {
    let kv6 = &sprite.kv6;
    let mut out = Vec::with_capacity(kv6.voxels.len());
    let mut voxel_iter = kv6.voxels.iter();
    // World edge length of one voxel = the largest basis-vector
    // length. Neighbours are spaced by the basis vector along each
    // axis, so the splat must cover the longest spacing to stay gap-
    // free. Uniform for the whole sprite; stored per voxel so the
    // buffer can mix sprites of different scales.
    let vlen = |v: [f32; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let world_size = vlen(sprite.s).max(vlen(sprite.h)).max(vlen(sprite.f));
    for x in 0..kv6.xsiz {
        for y in 0..kv6.ysiz {
            let count = kv6.ylen[x as usize][y as usize];
            for _ in 0..count {
                let v = voxel_iter.next().expect("KV6 ylen / voxels.len mismatch");
                let rel = (
                    (x as f32) - kv6.xpiv,
                    (y as f32) - kv6.ypiv,
                    (f32::from(v.z)) - kv6.zpiv,
                );
                let world_pos = [
                    sprite.p[0] + rel.0 * sprite.s[0] + rel.1 * sprite.h[0] + rel.2 * sprite.f[0],
                    sprite.p[1] + rel.0 * sprite.s[1] + rel.1 * sprite.h[1] + rel.2 * sprite.f[1],
                    sprite.p[2] + rel.0 * sprite.s[2] + rel.1 * sprite.h[2] + rel.2 * sprite.f[2],
                ];
                out.push(SpriteVoxel {
                    world_pos,
                    color: v.col,
                    world_size,
                    _pad: [0.0; 3],
                });
            }
        }
    }
    out
}

/// Pack many sprites into one flat voxel buffer ready for upload.
#[must_use]
pub fn flatten_sprites(sprites: &[Sprite]) -> Vec<SpriteVoxel> {
    let mut out = Vec::new();
    for sprite in sprites {
        out.extend_from_slice(&sprite_voxels_world_space(sprite));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use roxlap_formats::kv6::{Kv6, Voxel};

    /// A 2×1×1 kv6 with one voxel per column at z=0, pivot at the
    /// origin. Voxel (0,0) is `col_a`, voxel (1,0) is `col_b`.
    fn two_voxel_kv6(col_a: u32, col_b: u32) -> Kv6 {
        let mk = |col| Voxel {
            col,
            z: 0,
            vis: 0,
            dir: 0,
        };
        Kv6 {
            xsiz: 2,
            ysiz: 1,
            zsiz: 1,
            xpiv: 0.0,
            ypiv: 0.0,
            zpiv: 0.0,
            voxels: vec![mk(col_a), mk(col_b)],
            xlen: vec![1, 1],
            ylen: vec![vec![1], vec![1]],
            palette: None,
        }
    }

    #[test]
    fn axis_aligned_voxels_map_to_world_grid() {
        let sprite =
            Sprite::axis_aligned(two_voxel_kv6(0x80ff_0000, 0x8000_ff00), [10.0, 20.0, 30.0]);
        let out = sprite_voxels_world_space(&sprite);
        assert_eq!(out.len(), 2);
        // (x - xpiv) * s with s = +x unit basis ⇒ voxel x maps to
        // world x; y/z unchanged at the pivot.
        assert_eq!(out[0].world_pos, [10.0, 20.0, 30.0]);
        assert_eq!(out[0].color, 0x80ff_0000);
        assert_eq!(out[1].world_pos, [11.0, 20.0, 30.0]);
        assert_eq!(out[1].color, 0x8000_ff00);
    }

    #[test]
    fn flatten_concatenates_in_sprite_order() {
        let a = Sprite::axis_aligned(two_voxel_kv6(1, 2), [0.0, 0.0, 0.0]);
        let b = Sprite::axis_aligned(two_voxel_kv6(3, 4), [100.0, 0.0, 0.0]);
        let out = flatten_sprites(&[a, b]);
        let colors: Vec<u32> = out.iter().map(|v| v.color).collect();
        assert_eq!(colors, vec![1, 2, 3, 4]);
        assert_eq!(out[2].world_pos, [100.0, 0.0, 0.0]);
    }

    #[test]
    fn empty_input_yields_no_voxels() {
        assert!(flatten_sprites(&[]).is_empty());
    }

    #[test]
    fn gpu_voxel_struct_is_32_bytes() {
        // WGSL aligns the leading vec3<f32> to 16, rounding the
        // array stride to 32; the Rust struct must match exactly.
        assert_eq!(std::mem::size_of::<SpriteVoxel>(), 32);
    }

    #[test]
    fn world_size_is_max_basis_length() {
        // axis_aligned ⇒ unit basis ⇒ world_size 1.0.
        let unit = Sprite::axis_aligned(two_voxel_kv6(1, 2), [0.0, 0.0, 0.0]);
        assert!((sprite_voxels_world_space(&unit)[0].world_size - 1.0).abs() < 1e-6);
        // Scale the basis: s=3x, h=2y, f=1z ⇒ world_size = max = 3.
        let mut scaled = Sprite::axis_aligned(two_voxel_kv6(1, 2), [0.0, 0.0, 0.0]);
        scaled.s = [3.0, 0.0, 0.0];
        scaled.h = [0.0, 2.0, 0.0];
        scaled.f = [0.0, 0.0, 1.0];
        assert!((sprite_voxels_world_space(&scaled)[0].world_size - 3.0).abs() < 1e-6);
    }
}
