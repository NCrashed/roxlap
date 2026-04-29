//! Scalar `Rasterizer` implementation — port of voxlaptest's 4.7.5
//! scalar `hrendzsse` / `vrendzsse` fallbacks (`voxlap5.c:1947` /
//! `:2003` post-4.8.4 line numbers). Writes one `u32` ARGB pixel +
//! one `f32` z-buffer entry per screen position from radar entries
//! the (still-stubbed) `gline` will produce in R4.3.
//!
//! Per-pixel math (the SSE-batched form lives behind R5; this is the
//! pre-batch shape):
//!
//! ```text
//! col = scratch.angstart[plc >> 16] + j     // signed offset into radar
//! framebuffer[pixel] = radar[col].col       // packed ARGB
//! z = radar[col].dist / sqrt(dirx² + diry²) // f32 z-buffer entry
//! dirx += strx; diry += stry; plc += incr
//! ```
//!
//! The vertical scan additionally mutates `scratch.uurend[sx] +=
//! scratch.uurend[sx + half_stride]` per pixel — that's why the
//! Rasterizer trait now hands `&mut ScanScratch` to `vrend` /
//! `hrend`. `gline` is a TODO stub; without it the angstart entries
//! it would write are zero, so a freshly allocated radar yields all-
//! zero pixels. Tests pre-fill the radar manually to verify the
//! scanline rasterizer end of the pipeline.

// Module-wide cast allows: the per-pixel arithmetic constantly
// crosses signed/unsigned and i32/usize boundaries (loop counters,
// signed offsets into radar, framebuffer indices). Annotating each
// site individually buries the per-pixel logic in lint suppressions;
// the cast-correctness invariants hold by construction.
#![allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::similar_names
)]

use crate::rasterizer::{Rasterizer, ScanScratch};
use crate::ray_step::RayStep;
use crate::scan_loops::ScanContext;

/// Scalar rasterizer that writes pixels and a z-buffer entry per
/// screen position.
///
/// Borrows the framebuffer + zbuffer for the duration of one
/// `opticast` call; SDL hosts allocate these once and reuse across
/// frames, see `roxlap-host`.
pub struct ScalarRasterizer<'a> {
    framebuffer: &'a mut [u32],
    zbuffer: &'a mut [f32],
    /// Row stride in `u32` / `f32` elements (== framebuffer width
    /// for tightly-packed buffers; SDL streaming textures may add
    /// trailing padding).
    pitch_pixels: usize,
    /// Cached per-frame ray-step coefficients (`optistr*` /
    /// `optihei*` / `optiadd*` in voxlap globals). Stamped on each
    /// `frame_setup` call.
    ray_step: RayStep,
}

impl<'a> ScalarRasterizer<'a> {
    /// Create a rasterizer that will write into the supplied
    /// framebuffer + zbuffer pair. `pitch_pixels` must satisfy
    /// `pitch_pixels * height ≤ framebuffer.len()` for the height
    /// the engine renders into; the `frame_setup` hook does not
    /// validate sizes (it has no height to check against).
    ///
    /// `ray_step` is initialised to a zero placeholder; the real
    /// values get stamped on the first [`Rasterizer::frame_setup`]
    /// call before any per-pixel work runs.
    #[must_use]
    pub fn new(framebuffer: &'a mut [u32], zbuffer: &'a mut [f32], pitch_pixels: usize) -> Self {
        Self {
            framebuffer,
            zbuffer,
            pitch_pixels,
            ray_step: zero_ray_step(),
        }
    }
}

fn zero_ray_step() -> RayStep {
    RayStep {
        strx: 0.0,
        stry: 0.0,
        heix: 0.0,
        heiy: 0.0,
        addx: 0.0,
        addy: 0.0,
        cx16: 0,
        cy16: 0,
    }
}

impl Rasterizer for ScalarRasterizer<'_> {
    fn frame_setup(&mut self, ctx: &ScanContext<'_>) {
        // RayStep is Copy — cache per-frame so the per-pixel math in
        // hrend / vrend doesn't pay the indirect-borrow cost.
        self.ray_step = *ctx.rs;
    }

    fn gline(
        &mut self,
        scratch: &mut ScanScratch,
        length: u32,
        _x0: f32,
        _y0: f32,
        _x1: f32,
        _y1: f32,
    ) {
        // R4.3a placeholder: write `length + 1` magenta CastDat slots
        // starting at scratch.gscanptr (= what voxlap's gline would
        // populate). The four-quadrant scan loops advance gscanptr
        // by length + 1 after this call, so the writes here line up
        // with where angstart[ray] points. R4.3b onwards replaces
        // this with the real grouscan voxel-column raycaster.
        //
        // Magenta = 0x80FF00FF (Voxlap brightness bit + RGB ff00ff)
        // is intentionally a colour the scene's actual palette does
        // not produce, so the placeholder is unmistakeable when it
        // shows up on screen.
        let placeholder = crate::rasterizer::CastDat {
            col: 0x80ff_00ff_u32 as i32,
            dist: 1024,
        };
        let start = scratch.gscanptr;
        let end = start
            .saturating_add(length as usize + 1)
            .min(scratch.radar.len());
        for slot in start..end {
            scratch.radar[slot] = placeholder;
        }
    }

    fn hrend(
        &mut self,
        scratch: &mut ScanScratch,
        sx: i32,
        sy: i32,
        p1: i32,
        plc: i32,
        incr: i32,
        j: i32,
    ) {
        let rs = self.ray_step;
        // Per-frame setup gives strx/stry/heix/heiy/addx/addy; per-
        // pixel direction = strx*sx + heix*sy + addx, advancing by
        // strx in the inner loop.
        #[allow(clippy::cast_precision_loss)]
        let mut dirx = rs.strx * sx as f32 + rs.heix * sy as f32 + rs.addx;
        #[allow(clippy::cast_precision_loss)]
        let mut diry = rs.stry * sx as f32 + rs.heiy * sy as f32 + rs.addy;
        let row_start = sy as usize * self.pitch_pixels;

        let mut plc_local = plc;
        let mut x = sx;
        while x < p1 {
            // ray index = signed shift right (voxlap's `plc >> 16`).
            let ray_idx = (plc_local >> 16) as usize;
            let cd_offset = scratch.angstart[ray_idx] + j as isize;
            let cd = scratch.radar[cd_offset as usize];

            let pixel_idx = row_start + x as usize;
            self.framebuffer[pixel_idx] = cd.col as u32;
            #[allow(clippy::cast_precision_loss)]
            let z = cd.dist as f32 / (dirx * dirx + diry * diry).sqrt();
            self.zbuffer[pixel_idx] = z;

            dirx += rs.strx;
            diry += rs.stry;
            plc_local = plc_local.wrapping_add(incr);
            x += 1;
        }
    }

    fn vrend(
        &mut self,
        scratch: &mut ScanScratch,
        sx: i32,
        sy: i32,
        p1: i32,
        iplc: i32,
        iinc: i32,
    ) {
        let rs = self.ray_step;
        #[allow(clippy::cast_precision_loss)]
        let mut dirx = rs.strx * sx as f32 + rs.heix * sy as f32 + rs.addx;
        #[allow(clippy::cast_precision_loss)]
        let mut diry = rs.stry * sx as f32 + rs.heiy * sy as f32 + rs.addy;
        let row_start = sy as usize * self.pitch_pixels;
        let half_stride = scratch.uurend_half_stride;

        let mut iplc_local = iplc;
        let mut x = sx;
        while x < p1 {
            // Vertical scan reads the per-column ray index from
            // uurend[sx] (>>16 to drop the fractional bits).
            let xu = x as usize;
            let ray_idx = (scratch.uurend[xu] >> 16) as usize;
            let cd_offset = scratch.angstart[ray_idx] + iplc_local as isize;
            let cd = scratch.radar[cd_offset as usize];

            let pixel_idx = row_start + xu;
            self.framebuffer[pixel_idx] = cd.col as u32;
            #[allow(clippy::cast_precision_loss)]
            let z = cd.dist as f32 / (dirx * dirx + diry * diry).sqrt();
            self.zbuffer[pixel_idx] = z;

            dirx += rs.strx;
            diry += rs.stry;
            // Advance per-column ray index. uurend[x] persists
            // across vrend calls — this state is what couples
            // consecutive scanlines through the same column.
            scratch.uurend[xu] = scratch.uurend[xu].wrapping_add(scratch.uurend[xu + half_stride]);
            x += 1;
            iplc_local = iplc_local.wrapping_add(iinc);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rasterizer::CastDat;

    /// Build a `ScanContext` reference triple where rs uses a known
    /// strx so that pixel-spaced direction math is predictable.
    /// Returns owned `ProjectionRect`, `RayStep`, `OpticastPrelude`
    /// the caller can then assemble into a `ScanContext`.
    fn dummy_per_frame() -> (
        crate::projection::ProjectionRect,
        crate::ray_step::RayStep,
        crate::opticast_prelude::OpticastPrelude,
    ) {
        // Build a real CameraState + projection so the supporting
        // borrows have proper lifetimes; values aren't load-bearing
        // for these unit tests because we test scalar fill behaviour.
        let cam = crate::Camera {
            pos: [0.0, 0.0, 0.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 1.0, 0.0],
            forward: [0.0, 0.0, 1.0],
        };
        let cs = crate::camera_math::derive(&cam, 64, 64, 32.0, 32.0, 32.0);
        let proj = crate::projection::derive_projection(&cs, 64, 64, 32.0, 32.0, 32.0, 1);
        let rs = crate::ray_step::derive_ray_step(&cs, proj.cx, proj.cy, 32.0);
        let prelude = crate::opticast_prelude::derive_prelude(&cs, 2048, 1, 4, 1024);
        (proj, rs, prelude)
    }

    #[test]
    fn frame_setup_caches_ray_step() {
        let mut fb = vec![0u32; 64 * 64];
        let mut zb = vec![0.0f32; 64 * 64];
        let mut r = ScalarRasterizer::new(&mut fb, &mut zb, 64);
        let (proj, rs, prelude) = dummy_per_frame();
        let ctx = ScanContext {
            proj: &proj,
            rs: &rs,
            prelude: &prelude,
            xres: 64,
            yres: 64,
            anginc: 1,
        };
        r.frame_setup(&ctx);
        assert_eq!(r.ray_step.strx.to_bits(), rs.strx.to_bits());
        assert_eq!(r.ray_step.stry.to_bits(), rs.stry.to_bits());
        assert_eq!(r.ray_step.cx16, rs.cx16);
        assert_eq!(r.ray_step.cy16, rs.cy16);
    }

    #[test]
    fn hrend_writes_pixel_per_column_from_radar() {
        // Pre-fill scratch.radar with a recognizable colour gradient
        // and pre-set scratch.angstart so hrend resolves to the
        // expected radar slot per pixel. Then call hrend manually
        // and verify the framebuffer received the colours.
        let mut fb = vec![0u32; 64 * 64];
        let mut zb = vec![0.0f32; 64 * 64];
        let mut r = ScalarRasterizer::new(&mut fb, &mut zb, 64);
        let (proj, rs, prelude) = dummy_per_frame();
        let ctx = ScanContext {
            proj: &proj,
            rs: &rs,
            prelude: &prelude,
            xres: 64,
            yres: 64,
            anginc: 1,
        };
        r.frame_setup(&ctx);

        let mut scratch = ScanScratch::new_for_size(64, 64, 64);
        // Synthetic radar: 16 castdat entries with col = 0xAA...,
        // dist = 1 (zbuffer math will divide by sqrt of dir² so we
        // just pick a stable distance).
        for (i, slot) in scratch.radar.iter_mut().enumerate().take(16) {
            slot.col = 0x8000_0000_u32 as i32 | i as i32;
            slot.dist = 1024;
        }
        // angstart[0..4] all point at radar[0]; with j=0..3 and a
        // plc that increments by 0 (incr=0 holds plc>>16 at 0), the
        // pixel-by-pixel index lands at slots 0, 1, 2, 3 — i.e. j
        // selects the column.
        scratch.angstart[0] = 0;

        // Render a span of 4 columns starting at sx=10, sy=5, with j
        // varying via the scan: but voxlap's hrend uses a single j
        // for the whole span (per-ray castdat column is fixed). We
        // pick j=2 and verify framebuffer[row*pitch + sx..sx+4] all
        // equal radar[0+2].col = 0x80000002.
        r.hrend(&mut scratch, 10, 5, 14, 0, 0, 2);

        let row_off = 5 * 64;
        for x in 10..14 {
            let want = 0x8000_0000_u32 | 2;
            assert_eq!(
                fb[row_off + x],
                want,
                "fb[5][{x}] = {:#010x}, expected {:#010x}",
                fb[row_off + x],
                want,
            );
        }
        // Pixels outside the rendered span are untouched.
        assert_eq!(fb[row_off + 9], 0);
        assert_eq!(fb[row_off + 14], 0);
    }

    #[test]
    fn end_to_end_opticast_paints_magenta_through_placeholder_gline() {
        // Smoke test: with a valid above-the-slab camera and the
        // ScalarRasterizer's R4.3a placeholder gline, full opticast
        // should fill some framebuffer pixels with magenta.
        use crate::opticast as opticast_fn;
        use crate::OpticastSettings;

        let mut fb = vec![0u32; 640 * 480];
        let mut zb = vec![0.0f32; 640 * 480];
        let mut scratch = ScanScratch::new_for_size(640, 480, 2048);
        let mut rasterizer = ScalarRasterizer::new(&mut fb, &mut zb, 640);

        let cam = crate::Camera {
            pos: [1024.0, 1024.0, 128.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 1.0, 0.0],
            forward: [0.0, 0.0, 1.0],
        };
        let settings = OpticastSettings::for_oracle_framebuffer(640, 480);
        // Single solid slab at z = 200..254. cz = 128 < 200 →
        // air-above-the-slab, opticast renders.
        let column = vec![0u8, 200, 254, 0];

        let outcome = opticast_fn(
            &mut rasterizer,
            &mut scratch,
            &cam,
            &settings,
            2048,
            &column,
        );
        assert_eq!(outcome, crate::OpticastOutcome::Rendered);

        let magenta = 0x80ff_00ff_u32;
        let count = fb.iter().filter(|&&p| p == magenta).count();
        assert!(
            count > 0,
            "expected ≥ 1 magenta pixel, got 0 (gline placeholder \
             didn't make it through hrend/vrend)",
        );
    }

    #[test]
    fn vrend_advances_uurend_per_pixel() {
        // Verify the uurend[sx] += uurend[sx+half_stride] mutation
        // happens once per pixel.
        let mut fb = vec![0u32; 64 * 64];
        let mut zb = vec![0.0f32; 64 * 64];
        let mut r = ScalarRasterizer::new(&mut fb, &mut zb, 64);
        let (proj, rs, prelude) = dummy_per_frame();
        let ctx = ScanContext {
            proj: &proj,
            rs: &rs,
            prelude: &prelude,
            xres: 64,
            yres: 64,
            anginc: 1,
        };
        r.frame_setup(&ctx);

        let mut scratch = ScanScratch::new_for_size(64, 64, 64);
        scratch.radar[0] = CastDat {
            col: 0x8033_4455_u32 as i32,
            dist: 1024,
        };
        // angstart[0] = 0 → all rays read radar[0 + iplc].
        scratch.angstart[0] = 0;
        // Pre-set uurend so ray_idx = uurend[sx] >> 16 = 0 for all
        // four columns (we want them to all hit angstart[0]).
        let half = scratch.uurend_half_stride;
        for sx in 10..14 {
            scratch.uurend[sx] = 0;
            scratch.uurend[sx + half] = 1; // delta added per pixel
        }

        r.vrend(&mut scratch, 10, 5, 14, 0, 0);

        // Each column's uurend should have advanced by the delta.
        for sx in 10..14 {
            assert_eq!(scratch.uurend[sx], 1, "uurend[{sx}] not advanced");
        }
    }
}
