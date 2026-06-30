//! Shared image → flat-voxel-**slab** voxelization for the billboard-sprite
//! importers (stage BB). The axis convention + the [`VoxelFrame`] layout live
//! here; the per-format decoders ([`crate::gif_import`], [`crate::png_import`])
//! composite RGBA frames and feed them in. Always compiled (no image deps);
//! the decoders behind their `gif` / `png` features call into it.
//!
//! ## Axis convention
//!
//! Image **column → local +x** (right), image **row → local +z** (up; the top
//! image row maps to the highest `z`), and **local +y → depth** (the
//! `thickness` axis, the slab's normal). Pixel colours are packed voxlap-style
//! `0x80RRGGBB`; a pure-black pixel is nudged to `0x010101` because `rgb == 0`
//! is the engine's air/empty sentinel.

use crate::voxel_clip::{occ_words_per_col, LoopMode, VoxelClip, VoxelFrame};

/// Where the clip's pivot sits in the slab (the point that maps to the
/// instance's world position). `z = 0` is the bottom of the image.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Pivot {
    /// Bottom-centre — feet on the ground (the Doom-sprite default).
    BottomCenter,
    /// Geometric centre of the slab.
    Center,
    /// An explicit pivot in voxel units.
    Custom([f32; 3]),
}

impl Pivot {
    #[allow(clippy::cast_precision_loss)]
    pub(crate) fn resolve(self, dims: [u32; 3]) -> [f32; 3] {
        let (w, t, h) = (dims[0] as f32, dims[1] as f32, dims[2] as f32);
        match self {
            Self::BottomCenter => [w * 0.5, t * 0.5, 0.0],
            Self::Center => [w * 0.5, t * 0.5, h * 0.5],
            Self::Custom(p) => p,
        }
    }
}

/// Voxelize one composited RGBA canvas (`lw × lh`, row-major, 4 bytes/pixel)
/// into a flat `[lw, thickness, lh]` [`VoxelFrame`]. A pixel becomes a voxel
/// iff its alpha `>= alpha_cutoff` (cutout — 1-bit transparency; partial alpha
/// is not preserved, a `VoxelClip` is RGB-only). Column order is x-fastest then
/// depth (`col = x + depth·lw`), z ascending within a column.
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn voxelize_rgba(
    canvas: &[u8],
    lw: usize,
    lh: usize,
    thickness: u32,
    alpha_cutoff: u8,
) -> VoxelFrame {
    let dims = [lw as u32, thickness, lh as u32];
    let owpc = occ_words_per_col(dims) as usize;

    // Per image column x: ascending-z (z, colour) runs. The same content is
    // reused for every depth layer (a straight extrusion along +y).
    let mut img_col: Vec<Vec<(u32, u32)>> = vec![Vec::new(); lw];
    for row in 0..lh {
        let z = (lh - 1 - row) as u32; // top image row → highest z
        for (x, run) in img_col.iter_mut().enumerate() {
            let i = (row * lw + x) * 4;
            if canvas[i + 3] < alpha_cutoff {
                continue; // transparent ⇒ cutout
            }
            let mut rgb = (u32::from(canvas[i]) << 16)
                | (u32::from(canvas[i + 1]) << 8)
                | u32::from(canvas[i + 2]);
            if rgb == 0 {
                rgb = 0x0001_0101; // avoid the air/empty sentinel (rgb == 0)
            }
            run.push((z, 0x8000_0000 | rgb));
        }
    }
    for run in &mut img_col {
        run.sort_unstable_by_key(|&(z, _)| z);
    }

    let cols = lw * thickness as usize;
    let mut occupancy = vec![0u32; cols * owpc];
    let mut colors: Vec<u32> = Vec::new();
    let mut color_offsets: Vec<u32> = Vec::with_capacity(cols + 1);
    color_offsets.push(0);
    for depth in 0..thickness as usize {
        for (x, run) in img_col.iter().enumerate() {
            let col = x + depth * lw;
            for &(z, c) in run {
                let zi = z as usize;
                occupancy[col * owpc + zi / 32] |= 1u32 << (zi % 32);
                colors.push(c);
            }
            color_offsets.push(colors.len() as u32);
        }
    }

    VoxelFrame {
        occupancy,
        colors,
        color_offsets,
    }
}

/// Assemble already-voxelized slab `frames` + per-frame `durations` into a
/// [`VoxelClip`] (the I/P codec via [`VoxelClip::from_frames_auto`]). `pivot`
/// is resolved against `dims`.
pub(crate) fn assemble_clip(
    dims: [u32; 3],
    pivot: Pivot,
    voxel_world_size: f32,
    loop_mode: LoopMode,
    frames: &[VoxelFrame],
    durations: &[u32],
    default_frame_ms: u32,
    keyframe_gap: u32,
) -> VoxelClip {
    VoxelClip::from_frames_auto(
        dims,
        pivot.resolve(dims),
        voxel_world_size,
        loop_mode,
        frames,
        durations,
        default_frame_ms,
        keyframe_gap,
    )
}
