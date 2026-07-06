//! Import PNG sequences and animated PNG (APNG) into a [`VoxelClip`] of flat,
//! camera-facing voxel **slabs** — the truecolor counterpart of
//! [`crate::gif_import`] for Doom/Build-style billboard sprites (stage BB).
//! Feature-gated behind the `png` feature.
//!
//! Two entry points:
//! - [`voxel_clip_from_png_frames`] — N independent PNG files, each a full
//!   frame (a "frame_000.png … frame_023.png" export), all the same size.
//! - [`voxel_clip_from_apng`] — a single animated PNG (frames + per-frame
//!   delays + APNG dispose/blend compositing). A plain (non-animated) PNG is a
//!   1-frame clip.
//!
//! PNG carries real 8-bit alpha, but a [`VoxelClip`] is RGB-only, so alpha is
//! resolved as a **cutout** at [`PngImportOpts::alpha_cutoff`] (a pixel becomes
//! a voxel iff `alpha >= cutoff`). Per-pixel translucency is not preserved; for
//! a uniform fade use a translucent instance material + `alpha_mul`. See
//! [`crate::slab`] for the axis convention.

use crate::slab::{self, Pivot};
use crate::voxel_clip::{LoopMode, VoxelClip};

/// Options for the PNG importers.
#[derive(Debug, Clone)]
pub struct PngImportOpts {
    /// World size of one pixel-voxel.
    pub voxel_world_size: f32,
    /// Slab depth in voxels (extruded along +y). Clamped to `>= 1`.
    pub thickness: u32,
    /// Where the pivot sits (maps to the instance's world position).
    pub pivot: Pivot,
    /// Produced clip's loop mode.
    pub loop_mode: LoopMode,
    /// Frame duration (ms) for the sequence path / when an APNG delay is 0.
    pub default_frame_ms: u32,
    /// `max_keyframe_gap` forwarded to [`VoxelClip::from_frames_auto`].
    pub keyframe_gap: u32,
    /// Reject slabs larger than this bounding box (no silent downscale).
    pub max_dims: Option<[u32; 3]>,
    /// Minimum alpha (0..=255) for a pixel to become a voxel (cutout).
    /// Default 128.
    pub alpha_cutoff: u8,
}

impl Default for PngImportOpts {
    fn default() -> Self {
        Self {
            voxel_world_size: 1.0,
            thickness: 1,
            pivot: Pivot::BottomCenter,
            loop_mode: LoopMode::Loop,
            default_frame_ms: 100,
            keyframe_gap: 8,
            // QE.6b - default cap (mirrors GifImportOpts): opt out with
            // None for truly unbounded imports.
            max_dims: Some([4096, 4096, 4096]),
            alpha_cutoff: 128,
        }
    }
}

/// Failure modes of the PNG importers.
#[derive(Debug)]
pub enum PngImportError {
    /// The `png` decoder rejected the input (or an unsupported pixel format).
    Decode(String),
    /// No frames.
    Empty,
    /// Frames in a sequence disagree on size.
    SizeMismatch { expected: [u32; 2], got: [u32; 2] },
    /// `durations_ms` length didn't match the frame count.
    DurationsLen,
    /// The slab bounding box exceeds [`PngImportOpts::max_dims`].
    TooLarge { dims: [u32; 3], max: [u32; 3] },
}

impl core::fmt::Display for PngImportError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Decode(e) => write!(f, "PNG decode failed: {e}"),
            Self::Empty => write!(f, "no PNG frames"),
            Self::SizeMismatch { expected, got } => {
                write!(f, "PNG frame size {got:?} != expected {expected:?}")
            }
            Self::DurationsLen => write!(f, "durations length must match the frame count"),
            Self::TooLarge { dims, max } => {
                write!(f, "PNG slab {dims:?} exceeds max_dims {max:?}")
            }
        }
    }
}

impl std::error::Error for PngImportError {}

/// Decode one PNG image to RGBA8 + its `(width, height)`. `EXPAND` promotes
/// palette / sub-8-bit grayscale / `tRNS` so we only have to widen the four
/// 8-bit colour types to RGBA. 16-bit inputs are rejected (rare for sprites).
fn decode_rgba(bytes: &[u8]) -> Result<(Vec<u8>, u32, u32), String> {
    let mut dec = png::Decoder::new(bytes);
    dec.set_transformations(png::Transformations::EXPAND);
    let mut reader = dec.read_info().map_err(|e| e.to_string())?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).map_err(|e| e.to_string())?;
    if info.bit_depth != png::BitDepth::Eight {
        return Err(format!(
            "unsupported bit depth {:?} (use 8-bit)",
            info.bit_depth
        ));
    }
    let rgba = widen_to_rgba(&buf[..info.buffer_size()], info.color_type)?;
    Ok((rgba, info.width, info.height))
}

/// Widen an 8-bit decoded scanline buffer of `ct` to RGBA8.
fn widen_to_rgba(src: &[u8], ct: png::ColorType) -> Result<Vec<u8>, String> {
    let rgba = match ct {
        png::ColorType::Rgba => src.to_vec(),
        png::ColorType::Rgb => src
            .chunks_exact(3)
            .flat_map(|c| [c[0], c[1], c[2], 255])
            .collect(),
        png::ColorType::GrayscaleAlpha => src
            .chunks_exact(2)
            .flat_map(|c| [c[0], c[0], c[0], c[1]])
            .collect(),
        png::ColorType::Grayscale => src
            .chunks_exact(1)
            .flat_map(|c| [c[0], c[0], c[0], 255])
            .collect(),
        png::ColorType::Indexed => {
            return Err("indexed PNG not expanded — re-export as RGBA".into())
        }
    };
    Ok(rgba)
}

/// Import a sequence of independent PNG frames (all the same size) into a clip.
/// `durations_ms` is either empty (uniform [`PngImportOpts::default_frame_ms`])
/// or one entry per frame.
///
/// # Errors
/// [`PngImportError::Empty`] / [`PngImportError::DurationsLen`] /
/// [`PngImportError::SizeMismatch`] / [`PngImportError::Decode`] /
/// [`PngImportError::TooLarge`].
pub fn voxel_clip_from_png_frames(
    frames: &[&[u8]],
    durations_ms: &[u32],
    opts: &PngImportOpts,
) -> Result<VoxelClip, PngImportError> {
    if frames.is_empty() {
        return Err(PngImportError::Empty);
    }
    if !durations_ms.is_empty() && durations_ms.len() != frames.len() {
        return Err(PngImportError::DurationsLen);
    }
    let thickness = opts.thickness.max(1);
    let mut size: Option<(u32, u32)> = None;
    let mut vframes = Vec::with_capacity(frames.len());
    for &png in frames {
        let (rgba, w, h) = decode_rgba(png).map_err(PngImportError::Decode)?;
        match size {
            None => {
                let dims = [w, thickness, h];
                if let Some(max) = opts.max_dims {
                    if dims[0] > max[0] || dims[1] > max[1] || dims[2] > max[2] {
                        return Err(PngImportError::TooLarge { dims, max });
                    }
                }
                size = Some((w, h));
            }
            Some((ew, eh)) if (w, h) != (ew, eh) => {
                return Err(PngImportError::SizeMismatch {
                    expected: [ew, eh],
                    got: [w, h],
                });
            }
            Some(_) => {}
        }
        vframes.push(slab::voxelize_rgba(
            &rgba,
            w as usize,
            h as usize,
            thickness,
            opts.alpha_cutoff,
        ));
    }
    let (w, h) = size.unwrap();
    Ok(slab::assemble_clip(
        [w, thickness, h],
        opts.pivot,
        opts.voxel_world_size,
        opts.loop_mode,
        &vframes,
        durations_ms,
        opts.default_frame_ms,
        opts.keyframe_gap,
    ))
}

/// Import an animated PNG (APNG) into a clip — frames + per-frame delays, with
/// APNG dispose/blend compositing onto a persistent canvas (so sub-region
/// frames compose correctly). A non-animated PNG yields a 1-frame clip.
///
/// # Errors
/// As [`voxel_clip_from_png_frames`] (minus the sequence-only variants).
#[allow(clippy::cast_possible_truncation)]
pub fn voxel_clip_from_apng(
    bytes: &[u8],
    opts: &PngImportOpts,
) -> Result<VoxelClip, PngImportError> {
    let thickness = opts.thickness.max(1);
    let mut dec = png::Decoder::new(bytes);
    dec.set_transformations(png::Transformations::EXPAND);
    let mut reader = dec
        .read_info()
        .map_err(|e| PngImportError::Decode(e.to_string()))?;

    let (cw, ch) = (reader.info().width as usize, reader.info().height as usize);
    if cw == 0 || ch == 0 {
        return Err(PngImportError::Empty);
    }
    let dims = [cw as u32, thickness, ch as u32];
    if let Some(max) = opts.max_dims {
        if dims[0] > max[0] || dims[1] > max[1] || dims[2] > max[2] {
            return Err(PngImportError::TooLarge { dims, max });
        }
    }
    let num_frames = reader
        .info()
        .animation_control()
        .map_or(1, |a| a.num_frames)
        .max(1) as usize;

    let mut canvas = vec![0u8; cw * ch * 4]; // RGBA, starts transparent
    let mut vframes = Vec::with_capacity(num_frames);
    let mut durations = Vec::with_capacity(num_frames);

    for _ in 0..num_frames {
        let mut buf = vec![0u8; reader.output_buffer_size()];
        let info = reader
            .next_frame(&mut buf)
            .map_err(|e| PngImportError::Decode(e.to_string()))?;
        if info.bit_depth != png::BitDepth::Eight {
            return Err(PngImportError::Decode(format!(
                "unsupported bit depth {:?} (use 8-bit)",
                info.bit_depth
            )));
        }
        let frame_rgba = widen_to_rgba(&buf[..info.buffer_size()], info.color_type)
            .map_err(PngImportError::Decode)?;

        // Per-frame control (sub-region + delay + dispose/blend). Absent for a
        // plain PNG ⇒ a full-canvas Source frame with the default delay.
        let fc = reader.info().frame_control();
        let (fw, fh, fx, fy) = fc.map_or((cw, ch, 0usize, 0usize), |c| {
            (
                c.width as usize,
                c.height as usize,
                c.x_offset as usize,
                c.y_offset as usize,
            )
        });
        let blend = fc.map_or(png::BlendOp::Source, |c| c.blend_op);
        let dispose = fc.map_or(png::DisposeOp::None, |c| c.dispose_op);

        let restore = matches!(dispose, png::DisposeOp::Previous).then(|| canvas.clone());
        composite(&mut canvas, cw, ch, &frame_rgba, fw, fh, fx, fy, blend);

        vframes.push(slab::voxelize_rgba(
            &canvas,
            cw,
            ch,
            thickness,
            opts.alpha_cutoff,
        ));

        let ms = fc.map_or(0u32, |c| {
            let den = if c.delay_den == 0 {
                100
            } else {
                u32::from(c.delay_den)
            };
            u32::from(c.delay_num) * 1000 / den
        });
        durations.push(if ms == 0 { opts.default_frame_ms } else { ms });

        match dispose {
            png::DisposeOp::None => {}
            png::DisposeOp::Background => clear_rect(&mut canvas, cw, ch, fw, fh, fx, fy),
            png::DisposeOp::Previous => {
                if let Some(prev) = restore {
                    canvas = prev;
                }
            }
        }
    }

    if vframes.is_empty() {
        return Err(PngImportError::Empty);
    }
    Ok(slab::assemble_clip(
        dims,
        opts.pivot,
        opts.voxel_world_size,
        opts.loop_mode,
        &vframes,
        &durations,
        opts.default_frame_ms,
        opts.keyframe_gap,
    ))
}

/// Composite an APNG frame's `fw×fh` RGBA sub-region onto `canvas` at
/// `(fx, fy)` with the frame's blend op (`Source` replaces, `Over` alpha-blends).
#[allow(clippy::too_many_arguments)]
fn composite(
    canvas: &mut [u8],
    cw: usize,
    ch: usize,
    frame: &[u8],
    fw: usize,
    fh: usize,
    fx: usize,
    fy: usize,
    blend: png::BlendOp,
) {
    for ry in 0..fh {
        let cy = fy + ry;
        if cy >= ch {
            break;
        }
        for rx in 0..fw {
            let cx = fx + rx;
            if cx >= cw {
                break;
            }
            let si = (ry * fw + rx) * 4;
            let di = (cy * cw + cx) * 4;
            match blend {
                png::BlendOp::Source => canvas[di..di + 4].copy_from_slice(&frame[si..si + 4]),
                png::BlendOp::Over => {
                    let sa = u32::from(frame[si + 3]);
                    if sa == 255 {
                        canvas[di..di + 4].copy_from_slice(&frame[si..si + 4]);
                    } else if sa > 0 {
                        for k in 0..3 {
                            let s = u32::from(frame[si + k]);
                            let d = u32::from(canvas[di + k]);
                            #[allow(clippy::cast_possible_truncation)]
                            {
                                canvas[di + k] = ((s * sa + d * (255 - sa)) / 255) as u8;
                            }
                        }
                        let da = u32::from(canvas[di + 3]);
                        #[allow(clippy::cast_possible_truncation)]
                        {
                            canvas[di + 3] = (sa + da * (255 - sa) / 255).min(255) as u8;
                        }
                    }
                }
            }
        }
    }
}

/// Clear an `fw×fh` sub-region of `canvas` at `(fx, fy)` to transparent.
fn clear_rect(canvas: &mut [u8], cw: usize, ch: usize, fw: usize, fh: usize, fx: usize, fy: usize) {
    for ry in 0..fh {
        let cy = fy + ry;
        if cy >= ch {
            break;
        }
        for rx in 0..fw {
            let cx = fx + rx;
            if cx >= cw {
                break;
            }
            let di = (cy * cw + cx) * 4;
            canvas[di..di + 4].copy_from_slice(&[0, 0, 0, 0]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::color::VoxColor;

    fn px(r: u8, g: u8, b: u8, a: u8) -> [u8; 4] {
        [r, g, b, a]
    }
    fn make_rgba(w: usize, h: usize, pixels: &[[u8; 4]]) -> Vec<u8> {
        assert_eq!(pixels.len(), w * h);
        pixels.iter().flat_map(|p| p.iter().copied()).collect()
    }
    fn encode_png(w: u32, h: u32, rgba: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        {
            let mut enc = png::Encoder::new(&mut out, w, h);
            enc.set_color(png::ColorType::Rgba);
            enc.set_depth(png::BitDepth::Eight);
            let mut writer = enc.write_header().unwrap();
            writer.write_image_data(rgba).unwrap();
        }
        out
    }
    fn solid(frame: &crate::voxel_clip::VoxelFrame) -> usize {
        frame.colors.len()
    }

    #[test]
    fn png_sequence_imports_frames_dims_and_cutout() {
        let t = px(0, 0, 0, 0);
        let f0 = make_rgba(2, 2, &[px(255, 0, 0, 255), t, t, t]); // 1 opaque
        let f1 = make_rgba(2, 2, &[px(0, 255, 0, 255), px(0, 255, 0, 255), t, t]); // 2 opaque
        let p0 = encode_png(2, 2, &f0);
        let p1 = encode_png(2, 2, &f1);
        let clip =
            voxel_clip_from_png_frames(&[&p0, &p1], &[60, 120], &PngImportOpts::default()).unwrap();
        assert_eq!(clip.dims, [2, 1, 2]);
        let dec = clip.decode().unwrap();
        assert_eq!(dec.frame_count(), 2);
        assert_eq!(dec.durations, vec![60, 120]);
        assert_eq!(solid(&dec.frames[0]), 1);
        assert_eq!(solid(&dec.frames[1]), 2);
        assert_eq!(dec.frames[0].colors[0] & 0x00ff_ffff, 0x00ff_0000); // truecolor red
    }

    #[test]
    fn png_alpha_cutoff_thresholds_voxels() {
        // One pixel at alpha 100.
        let img = make_rgba(1, 1, &[px(200, 200, 200, 100)]);
        let p = encode_png(1, 1, &img);
        // Default cutoff 128 ⇒ dropped (air).
        let dec = voxel_clip_from_png_frames(&[&p], &[], &PngImportOpts::default())
            .unwrap()
            .decode()
            .unwrap();
        assert_eq!(solid(&dec.frames[0]), 0);
        // Cutoff 50 ⇒ kept.
        let opts = PngImportOpts {
            alpha_cutoff: 50,
            ..Default::default()
        };
        let dec = voxel_clip_from_png_frames(&[&p], &[], &opts)
            .unwrap()
            .decode()
            .unwrap();
        assert_eq!(solid(&dec.frames[0]), 1);
    }

    #[test]
    fn png_sequence_size_mismatch_is_rejected() {
        let a = encode_png(2, 2, &make_rgba(2, 2, &[px(1, 2, 3, 255); 4]));
        let b = encode_png(3, 1, &make_rgba(3, 1, &[px(1, 2, 3, 255); 3]));
        assert!(matches!(
            voxel_clip_from_png_frames(&[&a, &b], &[], &PngImportOpts::default()),
            Err(PngImportError::SizeMismatch { .. })
        ));
    }

    #[test]
    fn apng_imports_animated_frames() {
        // 2-frame full-canvas APNG (the encoder's default Source/None frames).
        let f0 = make_rgba(2, 1, &[px(255, 0, 0, 255), px(0, 0, 0, 0)]);
        let f1 = make_rgba(2, 1, &[px(0, 255, 0, 255), px(0, 0, 255, 255)]);
        let mut out = Vec::new();
        {
            let mut enc = png::Encoder::new(&mut out, 2, 1);
            enc.set_color(png::ColorType::Rgba);
            enc.set_depth(png::BitDepth::Eight);
            enc.set_animated(2, 0).unwrap();
            enc.set_frame_delay(5, 100).unwrap(); // 50 ms
            let mut writer = enc.write_header().unwrap();
            writer.write_image_data(&f0).unwrap();
            writer.write_image_data(&f1).unwrap();
        }
        let dec = voxel_clip_from_apng(&out, &PngImportOpts::default())
            .unwrap()
            .decode()
            .unwrap();
        assert_eq!(dec.frame_count(), 2);
        assert_eq!(dec.dims, [2, 1, 1]);
        assert_eq!(solid(&dec.frames[0]), 1); // one opaque pixel
        assert_eq!(solid(&dec.frames[1]), 2);
        assert!(dec.durations.iter().all(|&d| d == 50));
    }
}
