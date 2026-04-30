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

use crate::camera_math::CameraState;
use crate::gline::derive_gline_frustum;
use crate::grouscan::{grouscan_run, CfType, GrouscanInputs, CF_SEED_INDEX};
use crate::opticast::camera_column_slice;
use crate::opticast_prelude::{OpticastPrelude, PREC};
use crate::rasterizer::{Rasterizer, ScanScratch};
use crate::ray_step::RayStep;
use crate::scan_loops::ScanContext;

/// Voxlap's `gcsub[9]` per-side shading table, default zeroed
/// (no shading subtraction → raw voxel colour passes through).
/// The real engine populates this from `vx5.sideshademode` and
/// camera state; for R4.3a-rewire-3b we just need a valid borrow
/// to thread into `GrouscanInputs.gcsub`.
const DEFAULT_GCSUB: [i64; 9] = [0; 9];

/// Per-channel fog blend — voxlap5.c:2052-2056 (and the matching
/// hrend / vrend scalar tail). `col` is the source ARGB voxel
/// colour; `dist` is the radar slot's depth (PREC-scaled, so
/// `>> 20` gives the integer cell distance index into `foglut`).
/// `foglut` empty short-circuits to "no fog" (returns `col`
/// unchanged); OOB indices saturate to 32767 (full fog) since
/// `set_fog` pads the table that way.
//
// The C form is one branchless expression per channel; we keep
// it as-is for legibility against the source. `as i32` casts
// match the C int32 arithmetic.
#[allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::many_single_char_names
)]
fn fog_blend(col: i32, dist: i32, foglut: &[i32], fog_col: i32) -> i32 {
    if foglut.is_empty() {
        return col;
    }
    let idx = (dist >> 20) as usize;
    let l = foglut.get(idx).copied().unwrap_or(32767) & 32767;
    let k = col;
    let fc = fog_col;
    let r = (((fc & 255) - (k & 255)) * l) >> 15;
    let g = (((((fc >> 8) & 255) - ((k >> 8) & 255)) * l) >> 15) << 8;
    let b = (((((fc >> 16) & 255) - ((k >> 16) & 255)) * l) >> 15) << 16;
    r + g + b + k
}

/// Per-frame state cached on first `frame_setup` call. Owned here
/// (vs. borrowed from `ScanContext`) because gline needs to read
/// it across many calls without re-borrowing each time. The
/// `prelude` clone copies one `Vec<i32>` (the `y_lookup` mip
/// table) per frame — cheap.
struct FrameCache {
    ray_step: RayStep,
    camera_state: CameraState,
    prelude: OpticastPrelude,
    gstartz0: i32,
    gstartz1: i32,
    /// Voxlap's `v - *ixy_sptr_col` — byte offset within the
    /// camera's column to the slab whose top bounds the air gap
    /// from below. `0` ⇒ column-top.
    vptr_offset: usize,
}

/// Scalar rasterizer that writes pixels and a z-buffer entry per
/// screen position.
///
/// Borrows the framebuffer + zbuffer for the duration of one
/// `opticast` call; SDL hosts allocate these once and reuse across
/// frames, see `roxlap-host`.
//
// `slab_buf` / `column_offsets` / `vsid` are R4.3a-rewire-2
// scaffolding — the real `gline` (R4.3a-rewire-3) needs them to
// call `grouscan_run` per ray. The current placeholder gline
// doesn't read them yet, hence the dead_code allow.
#[allow(dead_code)]
pub struct ScalarRasterizer<'a> {
    framebuffer: &'a mut [u32],
    zbuffer: &'a mut [f32],
    /// Row stride in `u32` / `f32` elements (== framebuffer width
    /// for tightly-packed buffers; SDL streaming textures may add
    /// trailing padding).
    pitch_pixels: usize,
    /// World-level flat slab buffer (voxlap's malloc'd column
    /// data). Re-borrowed from opticast's caller for the lifetime
    /// of the rasterizer.
    slab_buf: &'a [u8],
    /// Per-column byte offsets into [`Self::slab_buf`].
    /// `column_offsets[i]..column_offsets[i + 1]` is column `i`'s
    /// slab list. Length is `vsid² + 1`.
    column_offsets: &'a [u32],
    /// World dimension. Combined with the prelude's `column_index`
    /// and the column-step path in grouscan, this is what lets the
    /// real gline walk the per-ray voxel-column traversal.
    vsid: u32,
    /// Per-frame state cache. `None` until the first `frame_setup`
    /// call; gline panics if invoked before that.
    frame: Option<FrameCache>,
}

impl<'a> ScalarRasterizer<'a> {
    /// Create a rasterizer that will write into the supplied
    /// framebuffer + zbuffer pair. `pitch_pixels` must satisfy
    /// `pitch_pixels * height ≤ framebuffer.len()` for the height
    /// the engine renders into; the `frame_setup` hook does not
    /// validate sizes (it has no height to check against).
    ///
    /// `slab_buf` / `column_offsets` / `vsid` describe the world
    /// the renderer reads from. Same shape as opticast's world
    /// arguments — pass the same values.
    ///
    /// `ray_step` is initialised to a zero placeholder; the real
    /// values get stamped on the first [`Rasterizer::frame_setup`]
    /// call before any per-pixel work runs.
    #[must_use]
    pub fn new(
        framebuffer: &'a mut [u32],
        zbuffer: &'a mut [f32],
        pitch_pixels: usize,
        slab_buf: &'a [u8],
        column_offsets: &'a [u32],
        vsid: u32,
    ) -> Self {
        Self {
            framebuffer,
            zbuffer,
            pitch_pixels,
            slab_buf,
            column_offsets,
            vsid,
            frame: None,
        }
    }
}

impl Rasterizer for ScalarRasterizer<'_> {
    fn frame_setup(&mut self, ctx: &ScanContext<'_>) {
        // Cache everything per-frame so gline doesn't re-borrow on
        // every call. Prelude is cloned (one Vec<i32> alloc per
        // frame for y_lookup; small).
        self.frame = Some(FrameCache {
            ray_step: *ctx.rs,
            camera_state: *ctx.camera_state,
            prelude: ctx.prelude.clone(),
            gstartz0: ctx.camera_gstartz0,
            gstartz1: ctx.camera_gstartz1,
            vptr_offset: ctx.camera_vptr_offset,
        });
    }

    fn gline(
        &mut self,
        scratch: &mut ScanScratch,
        length: u32,
        x0: f32,
        y0: f32,
        x1: f32,
        y1: f32,
    ) {
        // Voxlap's per-scanline ray-cast: derive the frustum, seed
        // cf[128], stamp scratch globals, call grouscan. Mirror of
        // voxlap5.c:gline (1146..1235) sans the unported pieces
        // (sideshademode, sky-radar bookkeeping — both R4.4 work).
        let cache = self
            .frame
            .as_ref()
            .expect("gline called before frame_setup");
        let leng = length as i32;

        // 1. Project per-ray frustum (vd0/vd1/vz0/vx1/vy1/vz1 +
        //    gixy/gpz/gdz). voxlap5.c:1153-1175.
        let f = derive_gline_frustum(
            &cache.camera_state,
            &cache.prelude,
            self.vsid,
            length,
            x0,
            y0,
            x1,
            y1,
        );

        // 2. Stamp ray-step globals onto scratch.
        scratch.gixy = f.gixy;
        scratch.gpz = f.gpz;
        scratch.gdz = f.gdz;

        // 3. cmprecip[leng] = 1/leng (voxlap precomputed table).
        //    gi0 / gi1 are per-pixel ray-step coefficients in
        //    Q12.20 (= PREC); cx0/cy0/cx1/cy1 are the cf[128] seed
        //    endpoints. voxlap5.c:1179-1190.
        // Voxlap precomputes a `cmprecip[leng]` table; we just
        // compute on the fly. The `as f32` casts here lose
        // precision for very large leng (> 2²³), but realistic
        // scanline lengths (a few thousand) are well below that.
        #[allow(clippy::cast_precision_loss)]
        let cmprecip = if leng > 0 { 1.0 / (leng as f32) } else { 0.0 };
        #[allow(clippy::cast_precision_loss)]
        let cmpprec = PREC as f32;
        let (gi0, gi1, cx0, cy0) = if cache.prelude.forward_z_sign < 0 {
            (
                ((f.vd1 - f.vd0) * cmprecip).round_ties_even() as i32,
                ((f.vz1 - f.vz0) * cmprecip).round_ties_even() as i32,
                (f.vd0 * cmpprec).round_ties_even() as i32,
                (f.vz0 * cmpprec).round_ties_even() as i32,
            )
        } else {
            (
                ((f.vd0 - f.vd1) * cmprecip).round_ties_even() as i32,
                ((f.vz0 - f.vz1) * cmprecip).round_ties_even() as i32,
                (f.vd1 * cmpprec).round_ties_even() as i32,
                (f.vz1 * cmpprec).round_ties_even() as i32,
            )
        };
        let cx1 = leng.wrapping_mul(gi0).wrapping_add(cx0);
        let cy1 = leng.wrapping_mul(gi1).wrapping_add(cy0);

        scratch.gi0 = gi0;
        scratch.gi1 = gi1;

        // 4. Seed cf[128] with the radar range + air-gap z-bounds +
        //    Q12.20 ray endpoints. voxlap5.c:1176-1190.
        let gscanptr_isize = scratch.gscanptr as isize;
        scratch.cf[CF_SEED_INDEX] = CfType {
            i0: gscanptr_isize,
            i1: gscanptr_isize + leng as isize,
            z0: cache.gstartz0,
            z1: cache.gstartz1,
            cx0,
            cy0,
            cx1,
            cy1,
        };

        // 5. gxmax = min(gmaxscandist, frustum-edge clip per axis).
        //    voxlap5.c:1192-1228. Unsigned compare — voxlap's `q`
        //    is a uint64_t product that may exceed gmaxscandist or
        //    wrap negative.
        let mut gxmax = cache.prelude.max_scan_dist;
        let li_pos = cache.prelude.li_pos;
        let vsid_signed = self.vsid as i32;
        let j0 = if f.gixy[0] < 0 {
            li_pos[0]
        } else {
            vsid_signed - 1 - li_pos[0]
        };
        let q0 = (i64::from(f.gdz[0]).wrapping_mul(i64::from(j0)))
            .wrapping_add(i64::from(f.gpz[0] as u32));
        if (q0 as u64) < u64::from(gxmax as u32) {
            gxmax = q0 as i32;
        }
        let j1 = if f.gixy[1] < 0 {
            li_pos[1]
        } else {
            vsid_signed - 1 - li_pos[1]
        };
        let q1 = (i64::from(f.gdz[1]).wrapping_mul(i64::from(j1)))
            .wrapping_add(i64::from(f.gpz[1] as u32));
        if (q1 as u64) < u64::from(gxmax as u32) {
            gxmax = q1 as i32;
        }
        scratch.gxmax = gxmax;

        // 6. Build inputs and call grouscan_run. The starting
        //    column is the camera's column (column_index from the
        //    prelude); the slab walker handles the rest.
        let column = camera_column_slice(
            self.slab_buf,
            self.column_offsets,
            cache.prelude.column_index,
        )
        .unwrap_or(&[]);
        let inputs = GrouscanInputs {
            column,
            gylookup: &cache.prelude.y_lookup,
            gcsub: &DEFAULT_GCSUB,
            slab_buf: self.slab_buf,
            column_offsets: self.column_offsets,
        };
        let _ = grouscan_run(
            scratch,
            &inputs,
            cache.vptr_offset,
            cache.prelude.column_index as usize,
            cache.prelude.x_mip,
            1, // gmipnum — single-mip until R4.5 lands the full
               // remiporend body and a real multi-mip world.
        );

        // 7. Advance gscanptr for the next gline call. Voxlap
        //    increments by leng + 1 (the radar slot just past the
        //    end of this ray's range).
        scratch.gscanptr = scratch.gscanptr.saturating_add(leng as usize + 1);
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
        let rs = self
            .frame
            .as_ref()
            .map(|f| f.ray_step)
            .expect("hrend/vrend called before frame_setup");
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

        // R5.1: SSE2 4-pixel batch via `_mm_rsqrt_ps` — port of
        // voxlaptest's `hrendzsse` (voxlap5.c:1947). 12-bit
        // approximation, no Newton refine, matching the
        // historical asm. The tail (0..3 leftover pixels)
        // continues with the bit-exact scalar form below; the
        // batch's z lanes will not match scalar 1/sqrt exactly,
        // mirroring voxlap. SSE2 is x86_64 baseline so no
        // runtime CPU-feature check is needed.
        //
        // `cast_ptr_alignment` is suppressed because we use
        // `_mm_storeu_si128` / `_mm_storeu_ps` — the `u`-suffix
        // variants explicitly support unaligned addresses, so a
        // u32 pointer cast to `*mut __m128i` is sound.
        #[cfg(target_arch = "x86_64")]
        #[allow(clippy::cast_ptr_alignment)]
        unsafe {
            use core::arch::x86_64::{
                __m128i, _mm_add_ps, _mm_cvtepi32_ps, _mm_cvtss_f32, _mm_mul_ps, _mm_rsqrt_ps,
                _mm_set1_ps, _mm_setr_epi32, _mm_setr_ps, _mm_storeu_ps, _mm_storeu_si128,
            };
            let strx = rs.strx;
            let stry = rs.stry;
            let vstrx4 = _mm_set1_ps(strx * 4.0);
            let vstry4 = _mm_set1_ps(stry * 4.0);
            let mut vdx = _mm_setr_ps(dirx, dirx + strx, dirx + 2.0 * strx, dirx + 3.0 * strx);
            let mut vdy = _mm_setr_ps(diry, diry + stry, diry + 2.0 * stry, diry + 3.0 * stry);
            while p1 - x >= 4 {
                // Gather 4 castdat hits — one per ray index.
                let mut col = [0i32; 4];
                let mut dst = [0i32; 4];
                for k in 0..4 {
                    let ray_idx = (plc_local >> 16) as usize;
                    let cd_offset = scratch.angstart[ray_idx] + j as isize;
                    let cd = scratch.radar[cd_offset as usize];
                    col[k] = cd.col;
                    dst[k] = cd.dist;
                    plc_local = plc_local.wrapping_add(incr);
                }
                // R5.2: per-pixel fog blend (voxlap's `hrendzfogsse`).
                // No-op when foglut is empty. Voxlap's MMX path used
                // pmulhw with foglut as 4 packed int16 lanes; we
                // mirror the scalar fallback the goldens use, which
                // applies a single `l = foglut[..] & 32767` factor
                // per pixel (one `l` per ray, all 3 channels).
                if !scratch.foglut.is_empty() {
                    for k in 0..4 {
                        col[k] = fog_blend(col[k], dst[k], &scratch.foglut, scratch.fog_col);
                    }
                }
                let vcol = _mm_setr_epi32(col[0], col[1], col[2], col[3]);
                let vdsi = _mm_setr_epi32(dst[0], dst[1], dst[2], dst[3]);
                let vdst = _mm_cvtepi32_ps(vdsi);
                let vsqr = _mm_add_ps(_mm_mul_ps(vdx, vdx), _mm_mul_ps(vdy, vdy));
                let vinv = _mm_rsqrt_ps(vsqr);
                let vz = _mm_mul_ps(vdst, vinv);

                let pixel_idx = row_start + x as usize;
                _mm_storeu_si128(
                    self.framebuffer
                        .as_mut_ptr()
                        .add(pixel_idx)
                        .cast::<__m128i>(),
                    vcol,
                );
                _mm_storeu_ps(self.zbuffer.as_mut_ptr().add(pixel_idx), vz);

                vdx = _mm_add_ps(vdx, vstrx4);
                vdy = _mm_add_ps(vdy, vstry4);
                x += 4;
            }
            // Bring scalar dirx/diry up to where the batch left
            // off — first lane of the post-step vdx/vdy.
            dirx = _mm_cvtss_f32(vdx);
            diry = _mm_cvtss_f32(vdy);
        }

        // Scalar tail — handles 0..3 leftover pixels on x86_64
        // and the full body on other targets.
        while x < p1 {
            // ray index = signed shift right (voxlap's `plc >> 16`).
            let ray_idx = (plc_local >> 16) as usize;
            let cd_offset = scratch.angstart[ray_idx] + j as isize;
            let cd = scratch.radar[cd_offset as usize];
            let col = fog_blend(cd.col, cd.dist, &scratch.foglut, scratch.fog_col);

            let pixel_idx = row_start + x as usize;
            self.framebuffer[pixel_idx] = col as u32;
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
        let rs = self
            .frame
            .as_ref()
            .map(|f| f.ray_step)
            .expect("hrend/vrend called before frame_setup");
        #[allow(clippy::cast_precision_loss)]
        let mut dirx = rs.strx * sx as f32 + rs.heix * sy as f32 + rs.addx;
        #[allow(clippy::cast_precision_loss)]
        let mut diry = rs.stry * sx as f32 + rs.heiy * sy as f32 + rs.addy;
        let row_start = sy as usize * self.pitch_pixels;
        let half_stride = scratch.uurend_half_stride;

        let mut iplc_local = iplc;
        let mut x = sx;

        // R5.3: SSE2 4-pixel batch — port of voxlaptest's
        // `vrendzsse` (voxlap5.c:2083). The per-column
        // `uurend[sx] += uurend[sx + half_stride]` update is
        // parallel-safe: uurend[sx + half_stride..] is read-only
        // here, and uurend[sx..+3] are four distinct lanes.
        // Read OLD u/d values, do the SSE z math, then write
        // back four NEW u values. Plus fog blend (R5.2-style)
        // when foglut is non-empty.
        #[cfg(target_arch = "x86_64")]
        #[allow(clippy::cast_ptr_alignment)]
        unsafe {
            use core::arch::x86_64::{
                __m128i, _mm_add_ps, _mm_cvtepi32_ps, _mm_cvtss_f32, _mm_mul_ps, _mm_rsqrt_ps,
                _mm_set1_ps, _mm_setr_epi32, _mm_setr_ps, _mm_storeu_ps, _mm_storeu_si128,
            };
            let strx = rs.strx;
            let stry = rs.stry;
            let vstrx4 = _mm_set1_ps(strx * 4.0);
            let vstry4 = _mm_set1_ps(stry * 4.0);
            let mut vdx = _mm_setr_ps(dirx, dirx + strx, dirx + 2.0 * strx, dirx + 3.0 * strx);
            let mut vdy = _mm_setr_ps(diry, diry + stry, diry + 2.0 * stry, diry + 3.0 * stry);
            while p1 - x >= 4 {
                let xu = x as usize;
                // Read 4 OLD uurend pairs (u, d). u = current ray
                // index for column; d = per-pixel delta.
                let mut u = [0i32; 4];
                let mut d = [0i32; 4];
                for k in 0..4 {
                    u[k] = scratch.uurend[xu + k];
                    d[k] = scratch.uurend[xu + k + half_stride];
                }
                // Gather 4 castdat hits.
                let mut col = [0i32; 4];
                let mut dst = [0i32; 4];
                for k in 0..4 {
                    let ray_idx = (u[k] >> 16) as usize;
                    let iplc_k = iplc_local.wrapping_add(iinc.wrapping_mul(k as i32));
                    let cd_offset = scratch.angstart[ray_idx] + iplc_k as isize;
                    let cd = scratch.radar[cd_offset as usize];
                    col[k] = cd.col;
                    dst[k] = cd.dist;
                }
                if !scratch.foglut.is_empty() {
                    for k in 0..4 {
                        col[k] = fog_blend(col[k], dst[k], &scratch.foglut, scratch.fog_col);
                    }
                }
                let vcol = _mm_setr_epi32(col[0], col[1], col[2], col[3]);
                let vdsi = _mm_setr_epi32(dst[0], dst[1], dst[2], dst[3]);
                let vdst = _mm_cvtepi32_ps(vdsi);
                let vsqr = _mm_add_ps(_mm_mul_ps(vdx, vdx), _mm_mul_ps(vdy, vdy));
                let vinv = _mm_rsqrt_ps(vsqr);
                let vz = _mm_mul_ps(vdst, vinv);

                let pixel_idx = row_start + xu;
                _mm_storeu_si128(
                    self.framebuffer
                        .as_mut_ptr()
                        .add(pixel_idx)
                        .cast::<__m128i>(),
                    vcol,
                );
                _mm_storeu_ps(self.zbuffer.as_mut_ptr().add(pixel_idx), vz);

                // Write back NEW uurend values — u + d per lane.
                for k in 0..4 {
                    scratch.uurend[xu + k] = u[k].wrapping_add(d[k]);
                }

                vdx = _mm_add_ps(vdx, vstrx4);
                vdy = _mm_add_ps(vdy, vstry4);
                iplc_local = iplc_local.wrapping_add(iinc.wrapping_mul(4));
                x += 4;
            }
            dirx = _mm_cvtss_f32(vdx);
            diry = _mm_cvtss_f32(vdy);
        }

        // Scalar tail — handles 0..3 leftover pixels on x86_64
        // and the full body on other targets.
        while x < p1 {
            // Vertical scan reads the per-column ray index from
            // uurend[sx] (>>16 to drop the fractional bits).
            let xu = x as usize;
            let ray_idx = (scratch.uurend[xu] >> 16) as usize;
            let cd_offset = scratch.angstart[ray_idx] + iplc_local as isize;
            let cd = scratch.radar[cd_offset as usize];
            let col = fog_blend(cd.col, cd.dist, &scratch.foglut, scratch.fog_col);

            let pixel_idx = row_start + xu;
            self.framebuffer[pixel_idx] = col as u32;
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

    /// Build owned per-frame state so tests can assemble a
    /// `ScanContext` with proper-lifetime borrows. Values aren't
    /// load-bearing for the scalar-fill behaviour tests; the real
    /// `gline` cares about them, hence `camera_state` joining the
    /// tuple.
    fn dummy_per_frame() -> (
        crate::camera_math::CameraState,
        crate::projection::ProjectionRect,
        crate::ray_step::RayStep,
        crate::opticast_prelude::OpticastPrelude,
    ) {
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
        (cs, proj, rs, prelude)
    }

    #[test]
    fn frame_setup_caches_ray_step() {
        let mut fb = vec![0u32; 64 * 64];
        let mut zb = vec![0.0f32; 64 * 64];
        let mut r = ScalarRasterizer::new(&mut fb, &mut zb, 64, &[], &[], 64);
        let (cs, proj, rs, prelude) = dummy_per_frame();
        let ctx = ScanContext {
            proj: &proj,
            rs: &rs,
            prelude: &prelude,
            xres: 64,
            yres: 64,
            anginc: 1,
            camera_state: &cs,
            camera_gstartz0: 0,
            camera_gstartz1: 0,
            camera_vptr_offset: 0,
        };
        r.frame_setup(&ctx);
        let cached_rs = r.frame.as_ref().expect("frame populated").ray_step;
        assert_eq!(cached_rs.strx.to_bits(), rs.strx.to_bits());
        assert_eq!(cached_rs.stry.to_bits(), rs.stry.to_bits());
        assert_eq!(cached_rs.cx16, rs.cx16);
        assert_eq!(cached_rs.cy16, rs.cy16);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn hrend_sse_batch_writes_4_pixel_block() {
        // R5.1 smoke test: hrend's SSE batch fires for span len ≥
        // 4. Pre-fill radar with 4 distinct ARGB values, run hrend
        // over [10..14], assert the framebuffer carries the colour
        // bits (z lanes use rsqrtps approximation so are intent-
        // ionally not bit-checked).
        let mut fb = vec![0u32; 64 * 64];
        let mut zb = vec![0.0f32; 64 * 64];
        let mut r = ScalarRasterizer::new(&mut fb, &mut zb, 64, &[], &[], 64);
        let (cs, proj, rs, prelude) = dummy_per_frame();
        let ctx = ScanContext {
            proj: &proj,
            rs: &rs,
            prelude: &prelude,
            xres: 64,
            yres: 64,
            anginc: 1,
            camera_state: &cs,
            camera_gstartz0: 0,
            camera_gstartz1: 0,
            camera_vptr_offset: 0,
        };
        r.frame_setup(&ctx);

        let mut scratch = ScanScratch::new_for_size(64, 64, 64);
        // 4 colour records so the batch reads each lane.
        for (i, slot) in scratch.radar.iter_mut().enumerate().take(4) {
            slot.col = 0x8000_0000_u32 as i32 | i as i32;
            slot.dist = 1024;
        }
        // angstart[i] = i so ray_idx i (= plc>>16) resolves to
        // radar[i + j] = radar[i] with j=0.
        for k in 0..4 {
            scratch.angstart[k] = k as isize;
        }

        // sx=10, p1=14, j=0, plc=0, incr=1<<16 → plc>>16 steps
        // 0,1,2,3 over the four pixels → angstart[0..4].
        r.hrend(&mut scratch, 10, 5, 14, 0, 1 << 16, 0);

        let row_off = 5 * 64;
        for k in 0..4 {
            let want = 0x8000_0000_u32 | k as u32;
            assert_eq!(
                fb[row_off + 10 + k],
                want,
                "fb[5][{}] = {:#010x}, expected {:#010x}",
                10 + k,
                fb[row_off + 10 + k],
                want,
            );
            // z lane non-zero (rsqrtps produced something).
            // Bit-compare to dodge clippy::float_cmp; we just want
            // to confirm the slot was written, not its precise value.
            assert_ne!(zb[row_off + 10 + k].to_bits(), 0u32);
        }
    }

    #[test]
    fn fog_blend_disabled_returns_col_unchanged() {
        let foglut: Vec<i32> = Vec::new();
        let col = 0x0080_C040;
        assert_eq!(fog_blend(col, 0x1234_5678, &foglut, 0xFF_FFFF), col);
    }

    #[test]
    fn fog_blend_full_fog_returns_fog_col_per_channel() {
        // l = 32767 → ((fog - col) * 32767) >> 15 = fog - col → final
        // is col + (fog - col) = fog (per channel; alpha untouched).
        let foglut = vec![32767; 2048];
        let col = 0x80_AA_BB_CC_u32 as i32;
        let fog = 0x00_11_22_33_i32;
        let blended = fog_blend(col, 0, &foglut, fog) as u32;
        // Low 24 bits = fog colour; alpha (high byte) survives from col.
        assert_eq!(blended & 0x00FF_FFFF, fog as u32 & 0x00FF_FFFF);
        assert_eq!(blended & 0xFF00_0000, col as u32 & 0xFF00_0000);
    }

    #[test]
    fn set_fog_zero_distance_clears_table() {
        let mut s = ScanScratch::new_for_size(64, 64, 64);
        s.set_fog(0x1234_5678, 100);
        assert!(!s.foglut.is_empty());
        s.set_fog(0, 0);
        assert!(s.foglut.is_empty());
    }

    #[test]
    fn set_fog_table_starts_at_zero_and_climbs() {
        let mut s = ScanScratch::new_for_size(64, 64, 64);
        s.set_fog(0xFF, 1024);
        // First entry: acc = 0 → hi16 = 0.
        assert_eq!(s.foglut[0], 0);
        // Last entry near the overflow boundary should be near 32767
        // (the saturate-fill value); voxlap's exact step makes
        // foglut[2047] either ~32766 (last walked entry) or 32767
        // (post-overflow padding) depending on max_scan_dist
        // divisibility.
        assert!(
            s.foglut[2047] > 30_000,
            "tail entry too low: {}",
            s.foglut[2047]
        );
    }

    #[test]
    fn hrend_writes_pixel_per_column_from_radar() {
        // Pre-fill scratch.radar with a recognizable colour gradient
        // and pre-set scratch.angstart so hrend resolves to the
        // expected radar slot per pixel. Then call hrend manually
        // and verify the framebuffer received the colours.
        let mut fb = vec![0u32; 64 * 64];
        let mut zb = vec![0.0f32; 64 * 64];
        let mut r = ScalarRasterizer::new(&mut fb, &mut zb, 64, &[], &[], 64);
        let (cs, proj, rs, prelude) = dummy_per_frame();
        let ctx = ScanContext {
            proj: &proj,
            rs: &rs,
            prelude: &prelude,
            xres: 64,
            yres: 64,
            anginc: 1,
            camera_state: &cs,
            camera_gstartz0: 0,
            camera_gstartz1: 0,
            camera_vptr_offset: 0,
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
    fn end_to_end_opticast_runs_through_real_gline() {
        // Smoke test: with a valid above-the-slab camera and the
        // real R4.3a-rewire-3b gline, full opticast should run
        // without panicking and return `Rendered`. The synthetic
        // single-slab world has no voxel colour bytes so grouscan's
        // fill loops bail to startsky, which writes the configured
        // skycast into the radar — verify by setting a recognizable
        // skycast and asserting some pixels carry it.
        use crate::opticast as opticast_fn;
        use crate::OpticastSettings;

        let mut fb = vec![0u32; 640 * 480];
        let mut zb = vec![0.0f32; 640 * 480];
        let mut scratch = ScanScratch::new_for_size(640, 480, 2048);
        // Recognizable sky colour — pixels filled by startsky's
        // solid-fill branch (the path the empty-colour-byte slab
        // ends up routing through) carry this.
        let sky_col = 0x80AB_CDEF_u32 as i32;
        scratch.set_skycast(sky_col, 0x7FFF_FFFF);

        // Single solid slab at z = 200..254. cz = 128 < 200 →
        // air-above-the-slab, opticast renders. Synthetic world:
        // only the camera's column (1024 * 2048 + 1024) holds the
        // slab; every other column is empty. Build world before
        // the rasterizer so it can borrow `&column` / `&column_offsets`.
        let column = vec![0u8, 200, 254, 0];
        let cam_idx = 1024usize * 2048 + 1024;
        let mut column_offsets = vec![0u32; 2048 * 2048 + 1];
        let column_len_u32 = u32::try_from(column.len()).expect("column fits u32");
        for offset in &mut column_offsets[(cam_idx + 1)..] {
            *offset = column_len_u32;
        }

        let mut rasterizer =
            ScalarRasterizer::new(&mut fb, &mut zb, 640, &column, &column_offsets, 2048);

        let cam = crate::Camera {
            pos: [1024.0, 1024.0, 128.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 1.0, 0.0],
            forward: [0.0, 0.0, 1.0],
        };
        let settings = OpticastSettings::for_oracle_framebuffer(640, 480);

        let outcome = opticast_fn(
            &mut rasterizer,
            &mut scratch,
            &cam,
            &settings,
            2048,
            &column,
            &column_offsets,
        );
        assert_eq!(outcome, crate::OpticastOutcome::Rendered);

        // Wiring smoke test — gline → derive_gline_frustum →
        // grouscan_run chain executes for every ray without
        // panicking, opticast returns Rendered. The synthetic
        // single-slab world has no colour bytes (header only) so
        // grouscan's drawflor fill bails on the bounds check and
        // routes to predeletez → deletez → Done before reaching
        // startsky; the radar stays at default zeros, the
        // framebuffer ends up sky-blue (the host pre-fill). What
        // matters is that nothing crashed — the per-ray gline
        // arithmetic + cf[128] seeding + grouscan dispatch all
        // hold up under live ray geometry. Once R4.3a-rewire-4
        // loads a real `.vxl` with colour bytes, grouscan's fill
        // loops will write recognisable voxel colours and a
        // colour-presence assertion replaces this comment.
        let _ = sky_col; // suppress unused-let warning — kept as
                         // scaffolding for the future assertion.
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn vrend_sse_batch_writes_4_pixel_block() {
        // R5.3 smoke test: vrend's SSE batch fires for span len ≥
        // 4. Pre-fill 4 distinct radar entries, set angstart so
        // each lane's uurend[sx]>>16 indexes a different ray, run
        // vrend over [10..14], assert each column got the right
        // colour and uurend advanced.
        let mut fb = vec![0u32; 64 * 64];
        let mut zb = vec![0.0f32; 64 * 64];
        let mut r = ScalarRasterizer::new(&mut fb, &mut zb, 64, &[], &[], 64);
        let (cs, proj, rs, prelude) = dummy_per_frame();
        let ctx = ScanContext {
            proj: &proj,
            rs: &rs,
            prelude: &prelude,
            xres: 64,
            yres: 64,
            anginc: 1,
            camera_state: &cs,
            camera_gstartz0: 0,
            camera_gstartz1: 0,
            camera_vptr_offset: 0,
        };
        r.frame_setup(&ctx);

        let mut scratch = ScanScratch::new_for_size(64, 64, 64);
        for k in 0..4 {
            scratch.radar[k] = CastDat {
                col: 0x8000_0000_u32 as i32 | k as i32,
                dist: 1024,
            };
            scratch.angstart[k] = k as isize;
        }
        // uurend[sx + k] >> 16 = k → ray_idx k → angstart[k] = k
        // → radar[k]. delta = 5 (so post-batch uurend[sx + k] =
        // (k << 16) + 5).
        let half = scratch.uurend_half_stride;
        for k in 0..4 {
            scratch.uurend[10 + k] = (k as i32) << 16;
            scratch.uurend[10 + k + half] = 5;
        }

        r.vrend(&mut scratch, 10, 5, 14, 0, 0);

        let row_off = 5 * 64;
        for k in 0..4 {
            let want = 0x8000_0000_u32 | k as u32;
            assert_eq!(fb[row_off + 10 + k], want, "fb col[{}]", 10 + k);
            assert!(zb[row_off + 10 + k].to_bits() != 0, "z[{}]", 10 + k);
            // Post-batch uurend = old_u + delta.
            assert_eq!(
                scratch.uurend[10 + k],
                ((k as i32) << 16) + 5,
                "uurend[{}]",
                10 + k
            );
        }
    }

    #[test]
    fn vrend_advances_uurend_per_pixel() {
        // Verify the uurend[sx] += uurend[sx+half_stride] mutation
        // happens once per pixel.
        let mut fb = vec![0u32; 64 * 64];
        let mut zb = vec![0.0f32; 64 * 64];
        let mut r = ScalarRasterizer::new(&mut fb, &mut zb, 64, &[], &[], 64);
        let (cs, proj, rs, prelude) = dummy_per_frame();
        let ctx = ScanContext {
            proj: &proj,
            rs: &rs,
            prelude: &prelude,
            xres: 64,
            yres: 64,
            anginc: 1,
            camera_state: &cs,
            camera_gstartz0: 0,
            camera_gstartz1: 0,
            camera_vptr_offset: 0,
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
