//! Import an animated GIF into a [`VoxelClip`] of flat, camera-facing
//! voxel **slabs** — the authoring bridge for Doom/Build-style billboard
//! sprites (stage **BB**; see `PORTING-BILLBOARD.md`). Feature-gated behind
//! the `gif` feature.
//!
//! Each GIF frame becomes one [`VoxelFrame`] slab of dims
//! `[W, thickness, H]`: every non-transparent pixel becomes a voxel, every
//! transparent pixel is air (a 1-bit cutout — GIF carries only a single
//! transparent palette index). The slab is 1 voxel thick by default
//! (`thickness`), so a camera-facing instance reads as flat pixel art while
//! still being a real voxel volume that casts + receives shadows.
//!
//! ## Axis convention
//!
//! Image **column → local +x** (right), image **row → local +z** (up; the
//! top image row maps to the highest `z`), and **local +y → depth** (the
//! `thickness` axis, the slab's normal). A `Cylindrical` billboard then
//! orients `forward = +y` toward the camera, `up = +z = world up`. Pixel
//! colours are packed voxlap-style `0x80RRGGBB` (neutral brightness byte);
//! a pure-black pixel is nudged to `0x010101` because `rgb == 0` is the
//! engine's air/empty sentinel.
//!
//! ## What this does NOT do
//!
//! Material assignment (e.g. an additive glow for fire) is **not** part of
//! the importer: a `VoxelClip` carries geometry + colour only. A whole-clip
//! glow is a per-instance concern — define an additive `Material` and set it
//! on the clip instance (`set_sprite_instance_material`) — so the importer
//! stays a pure geometry bridge.

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
    fn resolve(self, dims: [u32; 3]) -> [f32; 3] {
        let (w, t, h) = (dims[0] as f32, dims[1] as f32, dims[2] as f32);
        match self {
            Self::BottomCenter => [w * 0.5, t * 0.5, 0.0],
            Self::Center => [w * 0.5, t * 0.5, h * 0.5],
            Self::Custom(p) => p,
        }
    }
}

/// Options for [`voxel_clip_from_gif`].
#[derive(Debug, Clone)]
pub struct GifImportOpts {
    /// World size of one pixel-voxel (forwarded to the clip).
    pub voxel_world_size: f32,
    /// Slab depth in voxels (the silhouette is extruded along +y). `1` is a
    /// flat card; `> 1` gives a thicker shadow body. Clamped to `>= 1`.
    pub thickness: u32,
    /// Where the pivot sits (maps to the instance's world position).
    pub pivot: Pivot,
    /// Playback loop mode for the produced clip (GIFs loop by default).
    pub loop_mode: LoopMode,
    /// Frame duration (ms) used when a GIF frame's delay is `0`.
    pub default_frame_ms: u32,
    /// `max_keyframe_gap` forwarded to [`VoxelClip::from_frames_auto`].
    pub keyframe_gap: u32,
    /// Reject GIFs larger than this slab bounding box (no silent downscale).
    /// `None` ⇒ no limit.
    pub max_dims: Option<[u32; 3]>,
}

impl Default for GifImportOpts {
    fn default() -> Self {
        Self {
            voxel_world_size: 1.0,
            thickness: 1,
            pivot: Pivot::BottomCenter,
            loop_mode: LoopMode::Loop,
            default_frame_ms: 100,
            keyframe_gap: 8,
            max_dims: None,
        }
    }
}

/// Failure modes of [`voxel_clip_from_gif`].
#[derive(Debug)]
pub enum GifImportError {
    /// The `gif` decoder rejected the input.
    Decode(String),
    /// No frames decoded (empty / zero-sized GIF).
    Empty,
    /// The slab bounding box exceeds [`GifImportOpts::max_dims`].
    TooLarge { dims: [u32; 3], max: [u32; 3] },
}

impl core::fmt::Display for GifImportError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Decode(e) => write!(f, "GIF decode failed: {e}"),
            Self::Empty => write!(f, "GIF has no frames"),
            Self::TooLarge { dims, max } => {
                write!(f, "GIF slab {dims:?} exceeds max_dims {max:?}")
            }
        }
    }
}

impl std::error::Error for GifImportError {}

/// Decode an animated GIF into a [`VoxelClip`] of flat voxel slabs.
///
/// Per-frame GIF delays (centiseconds) become clip durations (ms; a `0`
/// delay falls back to [`GifImportOpts::default_frame_ms`]). Frame
/// disposal methods (`Background` / `Previous`) are honoured so partial
/// frames composite correctly onto a persistent canvas before voxelization.
///
/// # Errors
/// [`GifImportError::Decode`] on a malformed GIF, [`GifImportError::Empty`]
/// if no frames decode, or [`GifImportError::TooLarge`] if the slab exceeds
/// `opts.max_dims`.
#[allow(clippy::cast_possible_truncation, clippy::cast_lossless)]
pub fn voxel_clip_from_gif(
    bytes: &[u8],
    opts: &GifImportOpts,
) -> Result<VoxelClip, GifImportError> {
    let thickness = opts.thickness.max(1);

    let mut decoder = {
        let mut dopts = gif::DecodeOptions::new();
        dopts.set_color_output(gif::ColorOutput::RGBA);
        dopts
            .read_info(bytes)
            .map_err(|e| GifImportError::Decode(e.to_string()))?
    };

    let w = decoder.width() as u32;
    let h = decoder.height() as u32;
    if w == 0 || h == 0 {
        return Err(GifImportError::Empty);
    }

    let dims = [w, thickness, h];
    if let Some(max) = opts.max_dims {
        if dims[0] > max[0] || dims[1] > max[1] || dims[2] > max[2] {
            return Err(GifImportError::TooLarge { dims, max });
        }
    }

    // Composite each GIF frame onto a persistent RGBA canvas honouring the
    // logical screen + disposal, then voxelize the composited canvas.
    let (lw, lh) = (w as usize, h as usize);
    let mut canvas = vec![0u8; lw * lh * 4]; // RGBA, starts fully transparent
    let mut frames: Vec<VoxelFrame> = Vec::new();
    let mut durations: Vec<u32> = Vec::new();

    while let Some(frame) = decoder
        .read_next_frame()
        .map_err(|e| GifImportError::Decode(e.to_string()))?
    {
        // Snapshot for `Previous` disposal (restore the pre-draw canvas).
        let restore =
            matches!(frame.dispose, gif::DisposalMethod::Previous).then(|| canvas.clone());

        // Draw the frame's sub-rect over the canvas. GIF transparency is
        // 1-bit (alpha 0 ⇒ leave the canvas pixel), so only opaque pixels
        // overwrite — exactly an "over" with binary alpha.
        let (fl, ft) = (frame.left as usize, frame.top as usize);
        let (fw, fh) = (frame.width as usize, frame.height as usize);
        for ry in 0..fh {
            let cy = ft + ry;
            if cy >= lh {
                break;
            }
            for rx in 0..fw {
                let cx = fl + rx;
                if cx >= lw {
                    break;
                }
                let si = (ry * fw + rx) * 4;
                if frame.buffer[si + 3] != 0 {
                    let di = (cy * lw + cx) * 4;
                    canvas[di..di + 4].copy_from_slice(&frame.buffer[si..si + 4]);
                }
            }
        }

        frames.push(voxelize_canvas(&canvas, lw, lh, thickness));
        let delay_ms = (frame.delay as u32).saturating_mul(10);
        durations.push(if delay_ms == 0 {
            opts.default_frame_ms
        } else {
            delay_ms
        });

        // Apply this frame's disposal for the next iteration.
        match frame.dispose {
            gif::DisposalMethod::Background => {
                for ry in 0..fh {
                    let cy = ft + ry;
                    if cy >= lh {
                        break;
                    }
                    for rx in 0..fw {
                        let cx = fl + rx;
                        if cx >= lw {
                            break;
                        }
                        let di = (cy * lw + cx) * 4;
                        canvas[di..di + 4].copy_from_slice(&[0, 0, 0, 0]);
                    }
                }
            }
            gif::DisposalMethod::Previous => {
                if let Some(prev) = restore {
                    canvas = prev;
                }
            }
            gif::DisposalMethod::Any | gif::DisposalMethod::Keep => {}
        }
    }

    if frames.is_empty() {
        return Err(GifImportError::Empty);
    }

    let pivot = opts.pivot.resolve(dims);
    Ok(VoxelClip::from_frames_auto(
        dims,
        pivot,
        opts.voxel_world_size,
        opts.loop_mode,
        &frames,
        &durations,
        opts.default_frame_ms,
        opts.keyframe_gap,
    ))
}

/// Voxelize one composited RGBA canvas (`lw × lh`) into a flat
/// `[lw, thickness, lh]` [`VoxelFrame`]. Column order is x-fastest then
/// depth (`col = x + depth·lw`), z ascending within a column.
#[allow(clippy::cast_possible_truncation)]
fn voxelize_canvas(canvas: &[u8], lw: usize, lh: usize, thickness: u32) -> VoxelFrame {
    let dims = [lw as u32, thickness, lh as u32];
    let owpc = occ_words_per_col(dims) as usize;

    // Per image column x: ascending-z (z, colour) runs. The same content is
    // reused for every depth layer (a straight extrusion along +y).
    let mut img_col: Vec<Vec<(u32, u32)>> = vec![Vec::new(); lw];
    for row in 0..lh {
        let z = (lh - 1 - row) as u32; // top image row → highest z
        for (x, run) in img_col.iter_mut().enumerate() {
            let i = (row * lw + x) * 4;
            if canvas[i + 3] == 0 {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Encode a sequence of RGBA frames into in-memory GIF bytes for tests.
    /// `frames` are `(rgba, delay_cs)`; each is `w×h` RGBA (alpha 0 = transparent).
    fn encode_gif(w: u16, h: u16, frames: &[(Vec<u8>, u16)]) -> Vec<u8> {
        let mut out = Vec::new();
        {
            let mut enc = gif::Encoder::new(&mut out, w, h, &[]).unwrap();
            enc.set_repeat(gif::Repeat::Infinite).unwrap();
            for (rgba, delay) in frames {
                let mut buf = rgba.clone();
                let mut frame = gif::Frame::from_rgba(w, h, &mut buf);
                frame.delay = *delay;
                enc.write_frame(&frame).unwrap();
            }
        }
        out
    }

    /// Total solid voxels in a decoded clip frame (= colour-run length).
    fn solid(frame: &VoxelFrame) -> usize {
        frame.colors.len()
    }

    fn px(r: u8, g: u8, b: u8, a: u8) -> [u8; 4] {
        [r, g, b, a]
    }

    fn make_rgba(w: usize, h: usize, pixels: &[[u8; 4]]) -> Vec<u8> {
        assert_eq!(pixels.len(), w * h);
        pixels.iter().flat_map(|p| p.iter().copied()).collect()
    }

    #[test]
    fn imports_frames_dims_durations_and_cutout() {
        // 2×2. Frame 0: top-left red opaque, rest transparent (1 voxel).
        // Frame 1: top row green opaque, bottom transparent (2 voxels).
        let t = px(0, 0, 0, 0);
        let f0 = make_rgba(2, 2, &[px(255, 0, 0, 255), t, t, t]);
        let f1 = make_rgba(2, 2, &[px(0, 255, 0, 255), px(0, 255, 0, 255), t, t]);
        let bytes = encode_gif(2, 2, &[(f0, 5), (f1, 10)]);

        let clip = voxel_clip_from_gif(&bytes, &GifImportOpts::default()).unwrap();
        assert_eq!(clip.dims, [2, 1, 2]);
        let dec = clip.decode().unwrap();
        assert_eq!(dec.frame_count(), 2);
        assert_eq!(dec.durations, vec![50, 100]); // 5cs→50ms, 10cs→100ms
        assert_eq!(solid(&dec.frames[0]), 1);
        assert_eq!(solid(&dec.frames[1]), 2);
        // Colour is packed 0x80RRGGBB with the neutral brightness byte.
        assert_eq!(dec.frames[0].colors[0] >> 24, 0x80);
        assert_eq!(dec.frames[0].colors[0] & 0x00ff_ffff, 0x00ff_0000); // red
    }

    #[test]
    fn thickness_extrudes_voxel_count() {
        let f0 = make_rgba(2, 1, &[px(10, 20, 30, 255), px(40, 50, 60, 255)]);
        let bytes = encode_gif(2, 1, &[(f0, 5)]);
        let opts = GifImportOpts {
            thickness: 3,
            ..Default::default()
        };
        let clip = voxel_clip_from_gif(&bytes, &opts).unwrap();
        assert_eq!(clip.dims, [2, 3, 1]);
        let dec = clip.decode().unwrap();
        // 2 opaque pixels × 3 depth layers.
        assert_eq!(solid(&dec.frames[0]), 6);
    }

    #[test]
    fn black_pixel_is_not_swallowed_by_air_sentinel() {
        let f0 = make_rgba(1, 1, &[px(0, 0, 0, 255)]);
        let bytes = encode_gif(1, 1, &[(f0, 5)]);
        let clip = voxel_clip_from_gif(&bytes, &GifImportOpts::default()).unwrap();
        let dec = clip.decode().unwrap();
        assert_eq!(solid(&dec.frames[0]), 1);
        assert_ne!(dec.frames[0].colors[0] & 0x00ff_ffff, 0); // nudged off sentinel
    }

    #[test]
    fn rejects_oversize() {
        let f0 = make_rgba(2, 2, &[px(1, 2, 3, 255); 4]);
        let bytes = encode_gif(2, 2, &[(f0, 5)]);
        let opts = GifImportOpts {
            max_dims: Some([1, 1, 1]),
            ..Default::default()
        };
        assert!(matches!(
            voxel_clip_from_gif(&bytes, &opts),
            Err(GifImportError::TooLarge { .. })
        ));
    }

    #[test]
    fn pivot_bottom_center_sits_at_feet() {
        let f0 = make_rgba(2, 4, &[px(9, 9, 9, 255); 8]);
        let bytes = encode_gif(2, 4, &[(f0, 5)]);
        let clip = voxel_clip_from_gif(&bytes, &GifImportOpts::default()).unwrap();
        assert_eq!(clip.pivot, [1.0, 0.5, 0.0]); // [w/2, thickness/2, 0]
    }
}
