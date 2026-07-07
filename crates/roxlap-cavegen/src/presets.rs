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
use crate::worley::{anisotropic_dist_sq, place_seeds, worley_classify_grid, Seed};
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

// ====================================================================
// MagCaveGenerator (CD.7) — magenta base + yellow-green edge highlight.
// ====================================================================

/// Magenta-cave preset matching Ken's `cavemag3m.jpg`.
///
/// Magenta base across the cave surfaces; voxels whose distance to
/// the nearest air seed roughly equals the distance to the nearest
/// solid seed (i.e. right on the Worley boundary) get a yellow-green
/// edge highlight blended in. Per-voxel intensity wobbles ±25% via
/// a dedicated colour Perlin sampler.
#[derive(Debug, Default, Clone, Copy)]
pub struct MagCaveGenerator;

impl MagCaveGenerator {
    /// Default cave parameters tuned to match `cavemag3m.jpg`.
    /// Higher anisotropy (1.5) + lower air ratio (0.4) + extra
    /// Perlin octaves give the magenta variant more sinuous,
    /// densely-walled corridors than the blue variant.
    #[must_use]
    pub fn default_params() -> CaveParams {
        CaveParams {
            seed: 7,
            seed_count: 128,
            air_ratio: 0.4,
            anisotropy: 1.5,
            perlin_octaves: 4,
            perlin_amplitude: 0.25,
        }
    }
}

impl Generator for MagCaveGenerator {
    type Params = CaveParams;

    fn generate(&self, params: &Self::Params, vsid: u32) -> Vxl {
        // Mag's colour function needs the Worley seeds (the edge
        // highlight is keyed on `|d_a - d_s|`), so we place seeds
        // here and reuse them for both classify + colour.
        let shape_seeds = place_seeds(params, vsid);
        let grid = worley_classify_grid(params, vsid);
        let color = build_mag_color_grid(params, vsid, &grid, &shape_seeds);
        pack_dense_grid_to_vxl(&grid, &color, vsid)
    }
}

/// Build the per-voxel colour grid for the mag preset. Reuses the
/// shape-pass seeds for the edge-highlight distance check, plus a
/// dedicated Perlin sampler for intensity wobble.
#[allow(clippy::cast_precision_loss, clippy::cast_sign_loss)]
fn build_mag_color_grid(params: &CaveParams, vsid: u32, grid: &[u8], seeds: &[Seed]) -> Vec<u32> {
    let perlin = PerlinNoise3D::new(params.seed.wrapping_add(COLOR_SEED_OFFSET));
    let vsid_u = vsid as usize;
    let maxzdim_u = MAXZDIM as usize;
    let mut color = vec![0u32; grid.len()];
    for y in 0..vsid {
        for x in 0..vsid {
            for z in 0..MAXZDIM {
                let idx = (y as usize * vsid_u + x as usize) * maxzdim_u + z as usize;
                if grid[idx] != 0 {
                    color[idx] = mag_cave_color(x, y, z, &perlin, seeds, params.anisotropy);
                }
            }
        }
    }
    color
}

/// Magenta base; voxels close to the Worley boundary
/// (`|d_a − d_s|` small) blend toward yellow-green. Per-voxel
/// intensity perturbed ±25% via colour Perlin.
#[allow(clippy::cast_precision_loss)]
fn mag_cave_color(
    x: u32,
    y: u32,
    z: i32,
    perlin: &PerlinNoise3D,
    seeds: &[Seed],
    anisotropy: f32,
) -> u32 {
    const MAGENTA: u32 = 0x80_a0_40_a0;
    const YELLOW_GREEN: u32 = 0x80_b0_b0_20;
    /// Edge-highlight reach in voxel-distance units. Voxels with
    /// `|d_a − d_s|` ≤ this get full highlight; voxels well past
    /// it get pure magenta. Tuned for `seed_count = 128` at
    /// `vsid = 256` (typical `d_a ≈ 16`).
    const EDGE_THRESHOLD: f32 = 4.0;
    const INTENSITY_AMPLITUDE: f32 = 0.25;

    let p = [x as f32, y as f32, z as f32];
    let mut d_air_sq = f32::INFINITY;
    let mut d_solid_sq = f32::INFINITY;
    for seed in seeds {
        let d_sq = anisotropic_dist_sq(p, seed.pos, anisotropy);
        if seed.is_air {
            if d_sq < d_air_sq {
                d_air_sq = d_sq;
            }
        } else if d_sq < d_solid_sq {
            d_solid_sq = d_sq;
        }
    }
    let d_air = d_air_sq.sqrt();
    let d_solid = d_solid_sq.sqrt();
    let edge_factor = (1.0 - (d_air - d_solid).abs() / EDGE_THRESHOLD).clamp(0.0, 1.0);
    let base = lerp_rgb(MAGENTA, YELLOW_GREEN, edge_factor);

    let perlin_val = perlin.sample(
        (x as f32) * COLOR_PERLIN_FREQUENCY,
        (y as f32) * COLOR_PERLIN_FREQUENCY,
        (z as f32) * COLOR_PERLIN_FREQUENCY,
    );
    let intensity = 1.0 + INTENSITY_AMPLITUDE * perlin_val;
    apply_intensity(base, intensity)
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
            total_runs += usize::midpoint(i, 2);
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

    // ---- MagCaveGenerator (CD.7) -------------------------------------

    #[test]
    fn mag_default_params_match_plan() {
        let p = MagCaveGenerator::default_params();
        assert_eq!(p.seed_count, 128);
        assert!((p.air_ratio - 0.4).abs() < 1e-6, "air_ratio");
        assert!((p.anisotropy - 1.5).abs() < 1e-6, "anisotropy");
        assert_eq!(p.perlin_octaves, 4);
        assert!((p.perlin_amplitude - 0.25).abs() < 1e-6, "amplitude");
    }

    #[test]
    fn mag_generate_byte_stable_in_seed() {
        let p = CaveParams {
            seed_count: 16,
            ..MagCaveGenerator::default_params()
        };
        let a = MagCaveGenerator.generate(&p, 16);
        let b = MagCaveGenerator.generate(&p, 16);
        assert_eq!(a.column_offset.as_ref(), b.column_offset.as_ref());
        assert_eq!(a.data.as_ref(), b.data.as_ref());
    }

    #[test]
    fn mag_generate_yields_mixed_air_and_solid() {
        let p = CaveParams {
            seed_count: 16,
            ..MagCaveGenerator::default_params()
        };
        let vxl = MagCaveGenerator.generate(&p, 16);
        let mut total_runs = 0;
        for idx in 0..(16 * 16) {
            let mut b2 = vec![0i32; 256];
            roxlap_formats::edit::expandrle(vxl.column_data(idx), &mut b2);
            let mut i = 0;
            while b2[i + 1] < MAXZDIM {
                i += 2;
            }
            total_runs += usize::midpoint(i, 2);
        }
        assert!(
            total_runs > 256,
            "expected multi-run columns from cave gen; got {total_runs} total runs"
        );
    }

    #[test]
    fn mag_far_from_boundary_skews_magenta() {
        // Voxel deep inside a solid region (close to a solid seed,
        // far from any air seed) → |d_a - d_s| large → no edge
        // highlight → magenta base.
        // Use synthetic seeds: solid at origin, air far away.
        let seeds = vec![
            Seed {
                pos: [0.0, 0.0, 0.0],
                is_air: false,
            },
            Seed {
                pos: [100.0, 100.0, 100.0],
                is_air: true,
            },
        ];
        let perlin = PerlinNoise3D::new(0);
        // Voxel at integer coords (Perlin == 0 there) close to the
        // solid seed → magenta survives.
        let c = mag_cave_color(1, 0, 0, &perlin, &seeds, 1.0);
        let (r, g, b) = unpack_rgb(c);
        // MAGENTA = 0x80_a0_40_a0 → R=0xa0, G=0x40, B=0xa0.
        // Allow ±2 per channel for f32 round.
        assert!(
            (i32::from(r) - 0xa0).abs() <= 2,
            "R magenta-ish: got {r:#04x}"
        );
        assert!(
            (i32::from(g) - 0x40).abs() <= 2,
            "G magenta-ish: got {g:#04x}"
        );
        assert!(
            (i32::from(b) - 0xa0).abs() <= 2,
            "B magenta-ish: got {b:#04x}"
        );
    }

    #[test]
    fn mag_at_boundary_skews_yellow_green() {
        // Voxel equidistant from one air + one solid seed → full
        // edge-highlight → yellow-green.
        let seeds = vec![
            Seed {
                pos: [0.0, 0.0, 0.0],
                is_air: false,
            },
            Seed {
                pos: [2.0, 0.0, 0.0],
                is_air: true,
            },
        ];
        let perlin = PerlinNoise3D::new(0);
        // Voxel at (1, 0, 0): equidistant from both seeds (d=1.0 each
        // → |d_a - d_s| = 0 → full highlight).
        let c = mag_cave_color(1, 0, 0, &perlin, &seeds, 1.0);
        let (r, g, b) = unpack_rgb(c);
        // YELLOW_GREEN = 0x80_b0_b0_20 → R=0xb0, G=0xb0, B=0x20.
        assert!(
            (i32::from(r) - 0xb0).abs() <= 2,
            "R yellow-green-ish: got {r:#04x}"
        );
        assert!(
            (i32::from(g) - 0xb0).abs() <= 2,
            "G yellow-green-ish: got {g:#04x}"
        );
        assert!(
            (i32::from(b) - 0x20).abs() <= 2,
            "B yellow-green-ish: got {b:#04x}"
        );
    }

    #[test]
    fn mag_and_blue_diverge_in_byte_output() {
        // Same seed + vsid; the two presets produce different Vxls.
        // (Mag uses different defaults for air_ratio / anisotropy /
        // perlin_octaves / amplitude, so cave shape differs even
        // before colour.)
        let p = CaveParams {
            seed_count: 16,
            ..BlueCaveGenerator::default_params()
        };
        let blue = BlueCaveGenerator.generate(&p, 16);
        let q = CaveParams {
            seed_count: 16,
            ..MagCaveGenerator::default_params()
        };
        let mag = MagCaveGenerator.generate(&q, 16);
        // Shape differs: their grid structure should disagree on at
        // least some columns.
        let mut differing = 0;
        for idx in 0..(16 * 16) {
            if blue.column_data(idx) != mag.column_data(idx) {
                differing += 1;
            }
        }
        assert!(
            differing > 0,
            "Blue and Mag presets should produce different output"
        );
    }
}
