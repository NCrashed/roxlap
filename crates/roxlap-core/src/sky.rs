//! Sky-texture resource for the textured-sky path.
//!
//! Holds the equirectangular sky panorama (`pixels` + `xsiz`/`ysiz`)
//! the renderer samples on a ray miss (see
//! [`crate::dda`]'s `sample_sky`). Built once when a host calls
//! `Engine::set_sky`. The `lng[]` / `lat[]` / `bpl` / `lng_mul` fields
//! are a carryover from voxlap's per-ray longitude/latitude search
//! tables; the DDA renderer instead samples directly via `asin`/`atan2`
//! and does not need them.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::similar_names
)]

/// Sky texture + precomputed angle-lookup tables.
///
/// `pixels` are voxlap-style packed BGRA `i32`s (low byte = blue,
/// high byte = brightness/alpha) — same layout as voxel records.
/// `xsiz` × `ysiz` is the texture's pixel extent.
#[derive(Debug, Clone)]
pub struct Sky {
    /// Texture pixels, row-major, length `xsiz * ysiz`.
    pub pixels: Vec<i32>,
    /// Texel columns. Stored as the **post-decrement** value (= one
    /// less than the physical column count), matching voxlap's
    /// `skyxsiz` global state after `loadsky`. The lookup table
    /// [`Self::lat`] still has `xsiz + 1` entries, with `lat[0] = 0`
    /// as the asm-search lower-bound sentinel.
    pub xsiz: i32,
    /// Texel rows.
    pub ysiz: i32,
    /// Bytes per row (= `(xsiz + 1) * 4`, matching voxlap's
    /// `skybpl`).
    pub bpl: i32,
    /// Per-row `(cos, sin)` of the longitude angle. Length
    /// `ysiz`. Voxlap: `skylng[y].x = cos(y·2π/ysiz + π)`,
    /// `skylng[y].y = sin(...)`.
    pub lng: Vec<[f32; 2]>,
    /// Per-column packed `(xoff << 16) | (-yoff & 0xffff)`, both
    /// 16-bit. Length `xsiz + 1` (= original column count).
    /// Voxlap: `lat[x] = (xoff<<16) | ((-yoff) & 0xffff)` where
    /// `xoff = cos(((2x - xsiz)·π/(2·xsiz))·32767)` and
    /// `yoff = sin(...)`. `lat[0] = 0` is voxlap's "make sure
    /// assembly index never goes < 0" hack.
    pub lat: Vec<i32>,
    /// `ysiz / (2π)` — converts a longitude angle in radians into
    /// a row index. Used by `gline`'s first-ray atan2 path.
    pub lng_mul: f32,
}

impl Sky {
    /// Build a [`Sky`] from a row-major BGRA pixel grid. Computes
    /// the `lng` / `lat` lookup tables; mirror of voxlap5.c:3946-
    /// 3970 (the post-`kpzload` table init in `loadsky`).
    ///
    /// `pixels.len()` must equal `original_xsiz * ysiz`.
    /// `original_xsiz` is the **pre-decrement** column count
    /// (what voxlap reads from the loaded texture before stamping
    /// `skyxsiz--`). The returned [`Sky`] holds `xsiz =
    /// original_xsiz - 1`.
    ///
    /// # Panics
    ///
    /// Panics if `pixels.len() != original_xsiz * ysiz` or if
    /// `original_xsiz < 2` or `ysiz < 1`.
    #[must_use]
    pub fn from_pixels(pixels: Vec<i32>, original_xsiz: u32, ysiz: u32) -> Self {
        assert!(
            original_xsiz >= 2 && ysiz >= 1,
            "sky texture must be ≥ 2 wide and ≥ 1 tall (got {original_xsiz}×{ysiz})"
        );
        assert_eq!(
            pixels.len(),
            (original_xsiz as usize) * (ysiz as usize),
            "sky pixel count {} != xsiz*ysiz = {}",
            pixels.len(),
            (original_xsiz as usize) * (ysiz as usize)
        );

        let bpl = (original_xsiz as i32) * 4;
        let ysiz_i = ysiz as i32;
        let original_xsiz_i = original_xsiz as i32;

        // skylng — voxlap5.c:3946-3954.
        let mut lng = vec![[0.0_f32; 2]; ysiz as usize];
        let f = std::f32::consts::PI * 2.0 / (ysiz as f32);
        for y in 0..ysiz {
            let a = (y as f32) * f + std::f32::consts::PI;
            lng[y as usize] = [a.cos(), a.sin()];
        }
        // Voxlap's "make sure those while loops in gline() don't
        // lock up when ysiz==1" hack.
        if ysiz == 1 {
            lng[0] = [0.0, 0.0];
        }
        let lng_mul = (ysiz as f32) / (std::f32::consts::PI * 2.0);

        // skylat — voxlap5.c:3956-3967. lat[] has `original_xsiz`
        // entries; lat[0] = 0 is the lower-bound sentinel.
        let mut lat = vec![0i32; original_xsiz as usize];
        let f = std::f32::consts::PI * 0.5 / (original_xsiz as f32);
        for x in (1..original_xsiz_i).rev() {
            let ang = ((x << 1) - original_xsiz_i) as f32 * f;
            // Voxlap uses `ftol(cos(ang)*32767.0, ...)` which is
            // banker's-rounding via `lrintf`. f32::round() is
            // half-away-from-zero; the difference at our 32767
            // scale is at most ±1 for a handful of half-integer
            // edge cases. Use `lrintf` semantics via the existing
            // `fixed::ftol` helper for byte-stability.
            let xoff = crate::fixed::ftol(ang.cos() * 32767.0);
            let yoff = crate::fixed::ftol(ang.sin() * 32767.0);
            lat[x as usize] = (xoff << 16) | ((-yoff) & 0xffff);
        }
        // lat[0] = 0 (already the default vec! init).

        Self {
            pixels,
            xsiz: original_xsiz_i - 1,
            ysiz: ysiz_i,
            bpl,
            lng,
            lat,
            lng_mul,
        }
    }

    /// Voxlap's "BLUE" fallback sky (voxlap5.c:3920-3944). A
    /// 512×1 horizon-gradient texture: dark blue at the horizon
    /// fading up to lighter blue, then to a pale top. Useful as a
    /// default when no `.png` is loaded.
    #[must_use]
    pub fn blue_gradient() -> Self {
        const SKYXSIZ: i32 = 512;
        const SKYYSIZ: i32 = 1;
        let mut pixels = vec![0i32; (SKYXSIZ * SKYYSIZ) as usize];
        let y = SKYXSIZ * SKYXSIZ;
        // Lower half: gradient from horizon (x=0) to mid (x=256).
        for x in 0..=(SKYXSIZ >> 1) {
            let r = ((x * 1081 - SKYXSIZ * 252) * x) / y + 35;
            let g = ((x * 950 - SKYXSIZ * 198) * x) / y + 53;
            let b = ((x * 439 - SKYXSIZ * 21) * x) / y + 98;
            pixels[x as usize] = (r << 16) | (g << 8) | b;
        }
        pixels[(SKYXSIZ - 1) as usize] = 0x0050_903c;
        let mid = SKYXSIZ >> 1;
        let r_mid = (pixels[mid as usize] >> 16) & 0xff;
        let g_mid = (pixels[mid as usize] >> 8) & 0xff;
        let b_mid = pixels[mid as usize] & 0xff;
        // Upper half: linear interpolation from mid colour to
        // (0x50, 0x90, 0x3c).
        for x in (mid + 1)..SKYXSIZ {
            let denom = SKYXSIZ - 1 - mid;
            let r = ((0x50 - r_mid) * (x - mid)) / denom + r_mid;
            let g = ((0x90 - g_mid) * (x - mid)) / denom + g_mid;
            let b = ((0x3c - b_mid) * (x - mid)) / denom + b_mid;
            pixels[x as usize] = (r << 16) | (g << 8) | b;
        }
        Self::from_pixels(pixels, SKYXSIZ as u32, SKYYSIZ as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: blue gradient builds without panicking, has the
    /// expected pixel count, and lat/lng tables are consistent.
    #[test]
    fn blue_gradient_builds() {
        let s = Sky::blue_gradient();
        assert_eq!(s.pixels.len(), 512);
        assert_eq!(s.xsiz, 511); // post-decrement
        assert_eq!(s.ysiz, 1);
        assert_eq!(s.bpl, 512 * 4);
        assert_eq!(s.lng.len(), 1);
        assert_eq!(s.lat.len(), 512);
        // lat[0] is the asm-sentinel zero.
        assert_eq!(s.lat[0], 0);
        // lng[0] is forced to (0, 0) for the ysiz==1 hack.
        assert_eq!(s.lng[0][0].to_bits(), 0u32);
        assert_eq!(s.lng[0][1].to_bits(), 0u32);
    }

    /// `lat[x]`'s low and high 16 bits should encode int16
    /// `(-yoff, xoff)` such that `xoff² + yoff² ≈ 32767²`
    /// (= unit vector scaled by 32767).
    #[test]
    fn lat_entries_are_unit_vectors() {
        let s = Sky::blue_gradient();
        // Skip lat[0] (the sentinel) and check a few mid entries.
        for x in [1, 50, 100, 256, 400, 510] {
            let entry = s.lat[x];
            let neg_yoff = (entry & 0xffff) as i16 as i32;
            let xoff = ((entry >> 16) & 0xffff) as i16 as i32;
            let yoff = -neg_yoff;
            let len2 = xoff * xoff + yoff * yoff;
            // Should be ~ 32767² = 1_073_676_289 within rounding
            // (a few thousand units of slack for the f32→i16
            // truncation).
            assert!(
                (len2 - 32767 * 32767).abs() < 200_000,
                "lat[{x}] = ({xoff}, {yoff}); len² = {len2}, expected ~{}",
                32767 * 32767
            );
        }
    }

    /// A 4×4 procedural texture should round-trip xsiz=3, ysiz=4
    /// + build correctly-sized tables.
    #[test]
    fn from_pixels_4x4() {
        let pixels: Vec<i32> = (0..16).collect();
        let s = Sky::from_pixels(pixels, 4, 4);
        assert_eq!(s.xsiz, 3);
        assert_eq!(s.ysiz, 4);
        assert_eq!(s.bpl, 16);
        assert_eq!(s.lng.len(), 4);
        assert_eq!(s.lat.len(), 4);
        // pixel[3] (last column of row 0) is preserved.
        assert_eq!(s.pixels[3], 3);
    }
}
