//! Preset cave generators matching Ken + Tom's reference
//! screenshots from the 2003 "Justfly" demo.
//!
//! Each preset hard-codes a colour palette + per-voxel intensity
//! variation (via a dedicated Perlin sampler) on top of the shared
//! [`crate::worley_classify_grid`] cave-shape pipeline. Param
//! defaults differ per preset (see [`BlueCaveGenerator::default_params`]
//! and CD.7's [`MagCaveGenerator`] equivalent) so the same `seed`
//! produces visually distinct caves between presets.

use crate::pack::pack_dense_grid_to_vxl;
use crate::perlin::PerlinNoise3D;
use crate::worley::worley_classify_grid;
use crate::{CaveParams, Generator, Vxl, MAXZDIM};

/// Frequency of the colour-Perlin sampler in voxel units. Lower
/// values give larger colour patches.
const COLOR_PERLIN_FREQUENCY: f32 = 1.0 / 8.0;

/// Sub-seed offset applied to `params.seed` for the colour Perlin
/// sampler so its permutation table is decorrelated from the
/// cave-shape seed stream.
const COLOR_SEED_OFFSET: u64 = 0xDEAD_BEEF_CAFE_F00D;

/// Blue-cave preset matching Ken's `caveblue2m.jpg`.
///
/// Stone-grey base, mossy green near the top (sky-facing), dim
/// orange near the floor. Per-voxel intensity wobbles ±20% via a
/// dedicated colour Perlin sampler.
#[derive(Debug, Default, Clone, Copy)]
pub struct BlueCaveGenerator;

impl BlueCaveGenerator {
    /// Default cave parameters tuned to match `caveblue2m.jpg` —
    /// `seed = 7` was Justfly's `run3.bat` setup.
    #[must_use]
    pub fn default_params() -> CaveParams {
        CaveParams {
            seed: 7,
            seed_count: 128,
            air_ratio: 0.5,
            anisotropy: 1.0,
            perlin_octaves: 3,
            perlin_amplitude: 0.15,
        }
    }
}

impl Generator for BlueCaveGenerator {
    type Params = CaveParams;

    fn generate(&self, params: &Self::Params, vsid: u32) -> Vxl {
        let grid = worley_classify_grid(params, vsid);
        let color = build_blue_color_grid(params, vsid, &grid);
        pack_dense_grid_to_vxl(&grid, &color, vsid)
    }
}

/// Build the per-voxel colour grid for the blue preset. Only solid
/// voxels (`grid[i] != 0`) get a meaningful colour; air voxels are
/// left as 0.
#[allow(clippy::cast_precision_loss, clippy::cast_sign_loss)]
fn build_blue_color_grid(params: &CaveParams, vsid: u32, grid: &[u8]) -> Vec<u32> {
    let perlin = PerlinNoise3D::new(params.seed.wrapping_add(COLOR_SEED_OFFSET));
    let vsid_u = vsid as usize;
    let maxzdim_u = MAXZDIM as usize;
    let mut color = vec![0u32; grid.len()];
    for y in 0..vsid {
        for x in 0..vsid {
            for z in 0..MAXZDIM {
                let idx = (y as usize * vsid_u + x as usize) * maxzdim_u + z as usize;
                if grid[idx] != 0 {
                    color[idx] = blue_cave_color(x, y, z, &perlin);
                }
            }
        }
    }
    color
}

/// Stone grey at mid depth, mossy green at the top, dim orange at
/// the floor. Per-voxel intensity perturbed ±20% via colour Perlin.
#[allow(clippy::cast_precision_loss)]
fn blue_cave_color(x: u32, y: u32, z: i32, perlin: &PerlinNoise3D) -> u32 {
    /// Voxlap 32-bit colour encoding: `(brightness << 24) | (R << 16) | (G << 8) | B`.
    /// Brightness `0x80` is voxlap's "neutral" — matches the engine's
    /// default lighting amplitude.
    const BASE: u32 = 0x80_70_78_80; // stone grey
    const UPPER: u32 = 0x80_60_80_60; // mossy green
    const LOWER: u32 = 0x80_60_40_30; // dim orange
    const INTENSITY_AMPLITUDE: f32 = 0.20;

    let z_norm = (z as f32) / (MAXZDIM as f32);
    let base = if z_norm < 0.5 {
        // Top half: blend from upper (z=0) to base (z=MAXZDIM/2).
        lerp_rgb(UPPER, BASE, z_norm * 2.0)
    } else {
        // Bottom half: blend from base to lower (z=MAXZDIM-1).
        lerp_rgb(BASE, LOWER, (z_norm - 0.5) * 2.0)
    };
    let perlin_val = perlin.sample(
        (x as f32) * COLOR_PERLIN_FREQUENCY,
        (y as f32) * COLOR_PERLIN_FREQUENCY,
        (z as f32) * COLOR_PERLIN_FREQUENCY,
    );
    let intensity = 1.0 + INTENSITY_AMPLITUDE * perlin_val;
    apply_intensity(base, intensity)
}

/// Linearly interpolate per-channel between two voxlap-format
/// colours. Brightness byte is taken from `a` (the "from" colour).
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::many_single_char_names
)]
fn lerp_rgb(a: u32, b: u32, t: f32) -> u32 {
    let (ar, ag, ab) = unpack_rgb(a);
    let (br, bg, bb) = unpack_rgb(b);
    let brightness = (a >> 24) & 0xff;
    let r = (f32::from(ar) + (f32::from(br) - f32::from(ar)) * t).round() as u32;
    let g = (f32::from(ag) + (f32::from(bg) - f32::from(ag)) * t).round() as u32;
    let blu = (f32::from(ab) + (f32::from(bb) - f32::from(ab)) * t).round() as u32;
    (brightness << 24) | (r << 16) | (g << 8) | blu
}

/// Multiply the RGB channels of a voxlap-format colour by `factor`,
/// clamping to `0..=255`. Brightness byte is preserved.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn apply_intensity(color: u32, factor: f32) -> u32 {
    let (r, g, b) = unpack_rgb(color);
    let brightness = (color >> 24) & 0xff;
    let scaled = |c: u8| (f32::from(c) * factor).clamp(0.0, 255.0).round() as u32;
    (brightness << 24) | (scaled(r) << 16) | (scaled(g) << 8) | scaled(b)
}

#[inline]
fn unpack_rgb(color: u32) -> (u8, u8, u8) {
    #[allow(clippy::cast_possible_truncation)]
    let r = ((color >> 16) & 0xff) as u8;
    #[allow(clippy::cast_possible_truncation)]
    let g = ((color >> 8) & 0xff) as u8;
    #[allow(clippy::cast_possible_truncation)]
    let b = (color & 0xff) as u8;
    (r, g, b)
}

#[cfg(test)]
#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
mod tests {
    use super::*;

    #[test]
    fn blue_default_params_match_plan() {
        let p = BlueCaveGenerator::default_params();
        assert_eq!(p.seed, 7);
        assert_eq!(p.seed_count, 128);
        assert!((p.air_ratio - 0.5).abs() < 1e-6);
        assert!((p.anisotropy - 1.0).abs() < 1e-6);
        assert_eq!(p.perlin_octaves, 3);
        assert!((p.perlin_amplitude - 0.15).abs() < 1e-6);
    }

    #[test]
    fn blue_generate_byte_stable_in_seed() {
        // Cheap-VSID world; same seed → byte-equal Vxl.
        let p = CaveParams {
            seed_count: 16,
            ..BlueCaveGenerator::default_params()
        };
        let a = BlueCaveGenerator.generate(&p, 16);
        let b = BlueCaveGenerator.generate(&p, 16);
        assert_eq!(a.vsid, b.vsid);
        assert_eq!(a.column_offset.as_ref(), b.column_offset.as_ref());
        assert_eq!(a.data.as_ref(), b.data.as_ref());
    }

    #[test]
    fn blue_generate_yields_mixed_air_and_solid() {
        // Cave should have both air and solid (not pathological all-
        // air or all-solid).
        let p = CaveParams {
            seed_count: 16,
            ..BlueCaveGenerator::default_params()
        };
        let vxl = BlueCaveGenerator.generate(&p, 16);
        // Sample a few columns; each should expandrle to a non-trivial
        // b2 (at least one air gap somewhere).
        let mut total_runs = 0;
        for idx in 0..(16 * 16) {
            let mut b2 = vec![0i32; 256];
            roxlap_formats::edit::expandrle(vxl.column_data(idx), &mut b2);
            let mut i = 0;
            while b2[i + 1] < MAXZDIM {
                i += 2;
            }
            // i+2 entries ≥ sentinel; (i+2)/2 = number of solid runs.
            total_runs += (i + 2) / 2;
        }
        // 16x16 = 256 columns. Even pathological "every column is one
        // run" would give 256. Cave with carved air gaps should have
        // strictly more.
        assert!(
            total_runs > 256,
            "expected multi-run columns from cave gen; got {total_runs} total runs"
        );
    }

    #[test]
    fn lerp_rgb_endpoints_match() {
        let a = 0x80_aa_bb_cc;
        let b = 0x80_11_22_33;
        assert_eq!(lerp_rgb(a, b, 0.0), a);
        assert_eq!(lerp_rgb(a, b, 1.0), b & 0x00_ff_ff_ff | (a & 0xff00_0000));
    }

    #[test]
    fn lerp_rgb_midpoint() {
        // Halfway between (R=0, G=0, B=0) and (R=200, G=100, B=50)
        // → (R=100, G=50, B=25). Brightness from a.
        let a = 0x8000_0000u32;
        let b = 0x40c8_6432u32; // brightness ignored for b, RGB = (200, 100, 50)
        let mid = lerp_rgb(a, b, 0.5);
        let (r, g, blu) = unpack_rgb(mid);
        assert_eq!(r, 100, "red midpoint");
        assert_eq!(g, 50, "green midpoint");
        assert_eq!(blu, 25, "blue midpoint");
        // Brightness stays at 0x80.
        assert_eq!((mid >> 24) & 0xff, 0x80);
    }

    #[test]
    fn apply_intensity_clamps_to_255() {
        // Intensity > 1 saturates channels at 255.
        let c = 0x80_80_80_80; // brightness=0x80, RGB=(0x80,0x80,0x80)
        let scaled = apply_intensity(c, 2.5);
        let (r, g, b) = unpack_rgb(scaled);
        assert_eq!(r, 255, "red clamped");
        assert_eq!(g, 255, "green clamped");
        assert_eq!(b, 255, "blue clamped");
    }

    #[test]
    fn apply_intensity_preserves_brightness_byte() {
        let c = 0x80_80_80_80;
        let scaled = apply_intensity(c, 0.5);
        assert_eq!((scaled >> 24) & 0xff, 0x80, "brightness preserved");
    }

    #[test]
    fn blue_cave_color_top_skews_green() {
        // At z=0 (sky-facing top), colour blends fully toward UPPER
        // (mossy green). G channel should dominate over R, B.
        let perlin = PerlinNoise3D::new(0);
        // Sample ignoring perlin perturbation: zero-out perlin by
        // making coords land on integer grid (Perlin is exactly 0
        // there).
        let c = blue_cave_color(0, 0, 0, &perlin);
        let (r, g, b) = unpack_rgb(c);
        // UPPER = 0x80_60_80_60 → R=0x60, G=0x80, B=0x60.
        // At z=0 the lerp gives exactly UPPER.
        assert_eq!(r, 0x60);
        assert_eq!(g, 0x80);
        assert_eq!(b, 0x60);
    }

    #[test]
    fn blue_cave_color_bottom_skews_orange() {
        // At z=MAXZDIM-1 (floor), colour blends fully toward LOWER
        // (orange). R should dominate.
        let perlin = PerlinNoise3D::new(0);
        // z=MAXZDIM-1 means z_norm ≈ 1, lerp(BASE, LOWER, 1) = LOWER.
        // But Perlin at x=0,y=0,z=255 might not be exactly 0 — use
        // an integer grid coord that's safe.
        let c = blue_cave_color(0, 0, MAXZDIM - 1, &perlin);
        let (r, g, b) = unpack_rgb(c);
        // LOWER = 0x80_60_40_30 → R=0x60, G=0x40, B=0x30.
        // The Perlin perturbation at integer points is ~0 so colour
        // should be exactly LOWER (modulo intensity float math).
        // Allow ±2 per channel for f32 rounding noise.
        assert!(
            (i32::from(r) - 0x60).abs() <= 2,
            "R close to 0x60: got {r:#04x}"
        );
        assert!(
            (i32::from(g) - 0x40).abs() <= 2,
            "G close to 0x40: got {g:#04x}"
        );
        assert!(
            (i32::from(b) - 0x30).abs() <= 2,
            "B close to 0x30: got {b:#04x}"
        );
    }
}
