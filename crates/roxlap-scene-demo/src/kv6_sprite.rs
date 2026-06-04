//! GPU.9 — load + place a `.kv6` voxel sprite into the scene as a
//! regular `Grid`. The GPU renderer's existing per-grid path
//! handles upload + rendering with no special-case shader; the
//! KV6's small voxel set just sits in a single chunk of a tiny
//! grid.
//!
//! Pivot / KFA animation are GPU.9 follow-ups — for now the
//! sprite is static and placed at its `(0, 0, 0)` corner = the
//! grid's `GridTransform::origin`.
//!
//! `assets/coco.kv6` is voxlap's small mascot sprite (9×11×9
//! voxels, 148 textured). At ~3 ms of `Grid::set_voxel` calls it
//! barely registers at startup.

#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::similar_names
)]

use glam::{DVec3, IVec3};
use roxlap_formats::kv6::Kv6;
use roxlap_scene::{GridId, GridTransform, Scene};

/// `assets/coco.kv6` baked into the binary at build time. The demo
/// places one instance just in front of the camera spawn.
pub const COCO_KV6_BYTES: &[u8] = include_bytes!("../../../assets/coco.kv6");

/// Add a new grid to `scene` filled with `kv6`'s voxels. Voxel
/// `(x, y, z)` in the KV6's local frame becomes voxel `(x, y, z)`
/// in the grid's chunk at index `(0, 0, 0)`. The grid is placed
/// at `origin` in world space.
///
/// # Panics
/// If the KV6 declares more voxels than `xlen` / `ylen` account
/// for (a malformed file). Both numbers come from the same parsed
/// `Kv6` so this shouldn't fire in practice.
pub fn add_kv6_grid(scene: &mut Scene, kv6: &Kv6, origin: DVec3) -> GridId {
    let gid = scene.add_grid(GridTransform::at(origin));
    let grid = scene.grid_mut(gid).expect("just added");

    // Walk voxels in column-major (x, then y) order, using the
    // `ylen` counts to slice the flat `voxels` Vec into columns.
    let mut voxel_iter = kv6.voxels.iter();
    for x in 0..kv6.xsiz as i32 {
        for y in 0..kv6.ysiz as i32 {
            let count = kv6.ylen[x as usize][y as usize] as usize;
            for _ in 0..count {
                let v = voxel_iter
                    .next()
                    .expect("KV6 ylen accounting matches voxels.len()");
                grid.set_voxel(IVec3::new(x, y, i32::from(v.z)), Some(v.col));
            }
        }
    }
    gid
}
