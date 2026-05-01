//! Voxlap's 2D textured-quad blit primitive — `drawtile`
//! (`voxlap5.c:6954`). Used for HUD overlays, weapon sprites,
//! and the oracle's `tile_*` validation poses.
//!
//! Three rendering paths fork on `(black ^ white) & 0xff000000`
//! (the two endpoints' alpha bytes) and on the zoom factors:
//!
//! 1. **Ignore-alpha, 0.5× zoom** (`xz == yz == 32768`): each
//!    output pixel is the byte-wise rounded average of a 2×2
//!    source block. Voxlap's MMX `pavgb`-chain fast path.
//! 2. **Ignore-alpha, arbitrary zoom**: nearest-neighbour
//!    texture stretch — output pixel = source[uu, vv] with
//!    Q16.16 fixed-point u/v.
//! 3. **Alpha modulate + blend**: per-channel `(W - B)/256 + B`
//!    modulation between the `black` / `white` endpoint colours,
//!    then alpha-blended onto the destination via the modulated
//!    alpha byte. Includes voxlap's transparent-skip /
//!    opaque-passthrough trichotomy.
//!
//! Tile pixels are voxlap's BGRA `i32` layout (low byte = blue,
//! top byte = brightness/alpha) — same convention as the rest of
//! the engine. The destination framebuffer is row-major `u32`
//! with `pitch_pixels` stride.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::doc_markdown,
    clippy::many_single_char_names
)]

use crate::sprite::DrawTarget;

/// Voxlap's `mulshr16(a, d) = (a * d) >> 16` with i64 intermediate
/// to avoid signed-overflow on the multiply.
#[inline]
fn mulshr16(a: i32, d: i32) -> i32 {
    ((i64::from(a) * i64::from(d)) >> 16) as i32
}

/// Voxlap's `shldiv16(a, b) = ((a << 16) / b)` — Q16.16 reciprocal-
/// like helper for converting screen extents to tile-space steps.
#[inline]
fn shldiv16(a: i32, b: i32) -> i32 {
    ((i64::from(a) << 16) / i64::from(b)) as i32
}

/// Render one screen-space tile blit. Mirror of voxlap5.c:6954-7082.
///
/// Parameters mirror voxlap's call signature:
/// - `target`: framebuffer + dimensions. Z-buffer is unused.
/// - `tile_pixels`: source pixels, row-major BGRA `i32`. Length
///   must accommodate `(ty - 1) * (tile_pitch_bytes / 4) + tx`.
/// - `tile_pitch_bytes`: byte stride between source rows
///   (voxlap's `tp`). Typically `tx * 4`.
/// - `(tx, ty)`: source tile dimensions in pixels.
/// - `(tcx, tcy)`: tile centre in source-pixel Q16.16
///   coordinates. Voxlap uses this to anchor the tile so the
///   centre lands at `(sx, sy)` regardless of zoom.
/// - `(sx, sy)`: screen-space anchor in Q16.16. The tile centre
///   `(tcx, tcy)` ends up at this screen position.
/// - `(xz, yz)`: per-axis zoom in Q16.16. `65536` = 1×;
///   `32768` = 0.5× (triggers the 2×2-average fast path);
///   anything else takes the texture-stretch path.
/// - `black`, `white`: alpha-modulation endpoints. If the alpha
///   bytes are equal, the alpha path is skipped (the colour
///   modulation would be a constant tint applied to every pixel
///   and voxlap special-cases it as "no alpha"). Otherwise:
///   each source pixel's bytes get linearly remapped from
///   `[0, 255]` to `[black_byte, white_byte]`, then the modulated
///   alpha byte drives an alpha-blend onto the framebuffer.
pub fn drawtile(
    target: &mut DrawTarget<'_>,
    tile_pixels: &[i32],
    tile_pitch_bytes: i32,
    tx: i32,
    ty: i32,
    tcx: i32,
    tcy: i32,
    sx: i32,
    sy: i32,
    xz: i32,
    yz: i32,
    black: i32,
    white: i32,
) {
    if tile_pixels.is_empty() || xz == 0 || yz == 0 {
        return;
    }

    // Voxlap5.c:6962-6967 — derive screen + tile clip + per-pixel
    // step coefficients in Q16.16.
    let sx0 = sx.wrapping_sub(mulshr16(tcx, xz));
    let sx1 = sx0.wrapping_add(xz.wrapping_mul(tx));
    let sy0 = sy.wrapping_sub(mulshr16(tcy, yz));
    let sy1 = sy0.wrapping_add(yz.wrapping_mul(ty));

    let xres = target.width as i32;
    let yres = target.height as i32;
    let x0 = ((sx0 + 65535) >> 16).max(0);
    let x1 = ((sx1 + 65535) >> 16).min(xres);
    let y0 = ((sy0 + 65535) >> 16).max(0);
    let y1 = ((sy1 + 65535) >> 16).min(yres);
    if x0 >= x1 || y0 >= y1 {
        return;
    }

    let ui = shldiv16(65536, xz);
    let u = mulshr16(-sx0, ui);
    let vi = shldiv16(65536, yz);
    let v = mulshr16(-sy0, vi);

    let pitch_pixels = target.pitch_pixels;
    let tile_pitch_pixels = (tile_pitch_bytes >> 2) as usize;

    if (black ^ white) & 0x00ff_0000_u32.wrapping_shl(8) as i32 == 0
        && (black ^ white) & (0xff_u32 << 24) as i32 == 0
    {
        // Voxlap's `if (!((black^white)&0xff000000))` — alpha bytes
        // match → no-alpha branch. The literal `0xff000000` is i32
        // negative; using the unsigned form above sidesteps clippy
        // overflow warnings and is bit-equivalent.
        if xz == 32768 && yz == 32768 {
            // ---------------------------------------------------------
            // Path 1: 0.5× zoom, 2×2-average. Voxlap5.c:6970-7000.
            // ---------------------------------------------------------
            for y in y0..y1 {
                let vv = y.wrapping_mul(vi).wrapping_add(v);
                let row_pixel = (y as usize) * pitch_pixels;
                // Source-tile starting pixel for this row's first
                // 2×2 block.
                let plc_pixel = (((x0.wrapping_mul(ui).wrapping_add(u)) >> 16) as usize)
                    + ((vv >> 16) as usize) * tile_pitch_pixels;
                for x in x0..x1 {
                    let k = (x - x0) as usize;
                    // Each output pixel: 2x2 source block.
                    let ta = tile_pixels[plc_pixel + k * 2] as u32;
                    let tb = tile_pixels[plc_pixel + k * 2 + 1] as u32;
                    let ba = tile_pixels[plc_pixel + k * 2 + tile_pitch_pixels] as u32;
                    let bb = tile_pixels[plc_pixel + k * 2 + tile_pitch_pixels + 1] as u32;
                    let mut out: u32 = 0;
                    for b in 0..4u32 {
                        let va = (ta >> (b * 8)) & 0xff;
                        let vb = (tb >> (b * 8)) & 0xff;
                        let va_avg = (va + vb + 1) >> 1;
                        let v2a = (ba >> (b * 8)) & 0xff;
                        let v2b = (bb >> (b * 8)) & 0xff;
                        let vb_avg = (v2a + v2b + 1) >> 1;
                        let avg2 = (va_avg + vb_avg + 1) >> 1;
                        out |= avg2 << (b * 8);
                    }
                    target.framebuffer[row_pixel + x as usize] = out;
                }
            }
        } else {
            // ---------------------------------------------------------
            // Path 2: arbitrary zoom, nearest-neighbour stretch.
            // Voxlap5.c:7002-7012.
            // ---------------------------------------------------------
            let plc = x0.wrapping_mul(ui).wrapping_add(u);
            for y in y0..y1 {
                let vv = y.wrapping_mul(vi).wrapping_add(v);
                let row_pixel = (y as usize) * pitch_pixels;
                let j_pixel = ((vv >> 16) as usize) * tile_pitch_pixels;
                let mut uu = plc;
                for x in x0..x1 {
                    let src = tile_pixels[j_pixel + ((uu >> 16) as usize)];
                    target.framebuffer[row_pixel + x as usize] = src as u32;
                    uu = uu.wrapping_add(ui);
                }
            }
        }
    } else {
        // -----------------------------------------------------------
        // Path 3: alpha modulate + blend. Voxlap5.c:7014-7081.
        // -----------------------------------------------------------
        // Per-channel scale = (white_byte - black_byte) << 4. Voxlap
        // bumps a ±255 difference to ±256 so the pmulhw-equivalent
        // multiply produces the unbiased "scale by full byte range"
        // result without losing 1 LSB.
        let mut bw_scale = [0i16; 4];
        let mut bk = [0i32; 4];
        for b in 0..4usize {
            let bl = (black >> (b * 8)) & 0xff;
            let wh = (white >> (b * 8)) & 0xff;
            let mut diff = wh - bl;
            if diff == 255 {
                diff = 256;
            } else if diff == -255 {
                diff = -256;
            }
            bw_scale[b] = (diff << 4) as i16;
            bk[b] = bl;
        }

        for y in y0..y1 {
            let vv = y.wrapping_mul(vi).wrapping_add(v);
            let row_pixel = (y as usize) * pitch_pixels;
            let j_pixel = ((vv >> 16) as usize) * tile_pitch_pixels;
            let mut uu = x0.wrapping_mul(ui).wrapping_add(u);
            for x in x0..x1 {
                let src = tile_pixels[j_pixel + ((uu >> 16) as usize)];
                uu = uu.wrapping_add(ui);

                // Per-channel modulate: byte ↦ ((byte<<4)*scale)>>16
                // + black_byte, then per-channel saturate to [0, 255].
                let mut mod_word = [0i16; 4];
                let mut isat: u32 = 0;
                for (b, mw) in mod_word.iter_mut().enumerate() {
                    let byte = (src >> (b * 8)) & 0xff;
                    let prod = ((byte << 4) * i32::from(bw_scale[b])) >> 16;
                    let m = (prod + bk[b]) as i16;
                    *mw = m;
                    let r = i32::from(m).clamp(0, 255);
                    isat |= ((r as u32) & 0xff) << (b * 8);
                }
                let i = isat as i32;

                // Voxlap's "transparent-skip / opaque-passthrough"
                // hack (voxlap5.c:7056-7060). For i in
                // `[-0x1000000, +0x1000000)`: skip pixel (= alpha
                // ≈ 0). When i < 0 in this range (sign-extension of
                // 0xff alpha), write the pixel as-is.
                if (i.wrapping_add(0x0100_0000) as u32) < 0x0200_0000 {
                    if i < 0 {
                        target.framebuffer[row_pixel + x as usize] = i as u32;
                    }
                    continue;
                }

                // Alpha blend: dst.byte = clamp((mod-dst)*alpha/256
                // + dst, 0, 255). Voxlap's psubw / psllw 4 / pshufw
                // alpha / pmulhw / paddw / packuswb.
                let dst = target.framebuffer[row_pixel + x as usize] & 0x00ff_ffff;
                let alpha_shifted = i32::from(mod_word[3]) << 4;
                let mut blended: u32 = 0;
                for (b, &mw) in mod_word.iter().enumerate() {
                    let screen_byte = ((dst >> (b * 8)) & 0xff) as i32;
                    let delta = i32::from(mw) - screen_byte;
                    let scaled = ((delta << 4) * alpha_shifted) >> 16;
                    let r = (scaled + screen_byte).clamp(0, 255);
                    blended |= (r as u32) << (b * 8);
                }
                target.framebuffer[row_pixel + x as usize] = blended;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Allocate a framebuffer pre-filled with `fill_col` and a
    /// matching dummy zbuffer for `DrawTarget`.
    fn alloc_fb(w: u32, h: u32, fill_col: u32) -> (Vec<u32>, Vec<f32>) {
        let n = (w * h) as usize;
        (vec![fill_col; n], vec![f32::INFINITY; n])
    }

    fn make_target<'a>(fb: &'a mut [u32], zb: &'a mut [f32], w: u32, h: u32) -> DrawTarget<'a> {
        DrawTarget {
            framebuffer: fb,
            zbuffer: zb,
            pitch_pixels: w as usize,
            width: w,
            height: h,
        }
    }

    /// 1× zoom, no-alpha path: a 4×4 tile stamped at screen
    /// centre should reproduce the tile exactly within the
    /// destination region.
    #[test]
    fn one_x_no_alpha_copies_tile_pixels_unchanged() {
        let tile: Vec<i32> = (0..16).map(|i| 0x80_000000_u32 as i32 + i).collect();
        let (mut fb, mut zb) = alloc_fb(16, 16, 0);
        let mut target = make_target(&mut fb, &mut zb, 16, 16);
        // Tile centre = (2, 2); screen anchor = (8, 8). The 4×4
        // tile lands at screen [6..10) × [6..10).
        drawtile(
            &mut target,
            &tile,
            4 * 4,
            4,
            4,
            2 << 16,
            2 << 16,
            8 << 16,
            8 << 16,
            1 << 16,
            1 << 16,
            0,
            0,
        );
        // Spot-check: tile[5] (= row 1, col 1) lands at screen
        // (6+1, 6+1) = (7, 7).
        assert_eq!(fb[7 * 16 + 7], 0x80_000005);
        // Tile pixel at (3, 3) → screen (9, 9).
        assert_eq!(fb[9 * 16 + 9], 0x80_00000f);
    }

    /// 0.5× zoom, no-alpha path: a 4×4 tile blits as a 2×2
    /// region of byte-wise-averaged pixels.
    #[test]
    fn half_zoom_averages_2x2_blocks() {
        // All-white tile: every output pixel should also be white
        // (averaging white with white gives white).
        let tile: Vec<i32> = vec![0x80_ffffff_u32 as i32; 16];
        let (mut fb, mut zb) = alloc_fb(16, 16, 0);
        let mut target = make_target(&mut fb, &mut zb, 16, 16);
        drawtile(
            &mut target,
            &tile,
            4 * 4,
            4,
            4,
            2 << 16,
            2 << 16,
            8 << 16,
            8 << 16,
            32768,
            32768,
            0,
            0,
        );
        // Output region for 4×4 tile @ 0.5× = 2×2 pixels at
        // screen (7, 7) ish. Find any non-zero pixel.
        let touched: Vec<u32> = fb.iter().copied().filter(|&p| p != 0).collect();
        assert!(!touched.is_empty(), "blit produced no pixels");
        for p in touched {
            assert_eq!(p, 0x80_ffffff, "averaged white tile must stay white");
        }
    }

    /// Out-of-bounds blit should be a no-op (no panic / no write
    /// past the framebuffer).
    #[test]
    fn fully_offscreen_is_noop() {
        let tile: Vec<i32> = vec![0x80_aabbcc_u32 as i32; 16];
        let (mut fb, mut zb) = alloc_fb(16, 16, 0xdead_beef);
        let mut target = make_target(&mut fb, &mut zb, 16, 16);
        // Anchor far off-screen → blit clipped to zero pixels.
        drawtile(
            &mut target,
            &tile,
            16,
            4,
            4,
            2 << 16,
            2 << 16,
            10000 << 16,
            10000 << 16,
            1 << 16,
            1 << 16,
            0,
            0,
        );
        assert!(fb.iter().all(|&p| p == 0xdead_beef));
    }
}
