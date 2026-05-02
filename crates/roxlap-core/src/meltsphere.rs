//! Sphere-region voxel extraction (`meltsphere`) and its support
//! tables.
//!
//! Port of voxlap's `meltsphere` (voxlap5.c:10222-10344). Walks an
//! AABB-bounded sphere of voxels in the world and packages the
//! visible (border) voxels into a fresh `kv6` sprite.
//!
//! Supporting helpers:
//! - `lightvox` — alpha-byte face shader (voxlap5.c:623-632).
//! - [`PowerTables`] + [`PowerTables::build_tempfloatbuf`] — the
//!   factr / logint / tempfloatbuf machinery (voxlap5.c:118-120
//!   statics, :12224-12236 init, :10240-10252 per-call build).

// Entire module is a port of voxlap C bit-twiddle; the casts
// implicit in the source map cleanly to Rust's `as` casts.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]

use roxlap_formats::kv6::{Kv6, Voxel};

use crate::world_query::{getcube, Cube};

/// Voxlap's `SETSPHMAXRAD` (voxlap5.c:117). Upper bound on
/// `hitrad`; the fast-pow tables are sized to this.
pub(crate) const SETSPHMAXRAD: usize = 256;

/// Voxlap's `MAXZDIM` (voxlap5.h:10) — voxlap stores z as a single
/// byte, so the world is at most 256 voxels tall.
pub(crate) const MAXZDIM: i32 = 256;

/// Apply alpha-byte face shading to a packed voxlap colour.
///
/// Port of `lightvox` (voxlap5.c:623-632). The high byte of `i` is
/// treated as a brightness multiplier (`0x80` is neutral); each RGB
/// channel is multiplied by it, shifted right by 7, and clamped to
/// 255. The returned colour has its alpha byte cleared.
///
/// Voxlap uses this to bake alpha-byte intensity into the per-voxel
/// colour stored in a kv6 sprite: meltsphere copies the world's
/// `BR(rgb)`-style packed colour through `lightvox` so the resulting
/// `Voxel::col` is plain `0x00rrggbb`.
#[must_use]
pub(crate) fn lightvox(i: u32) -> u32 {
    let b = i >> 24;
    let r = ((((i >> 16) & 0xff) * b) >> 7).min(255);
    let g = ((((i >> 8) & 0xff) * b) >> 7).min(255);
    let bl = (((i & 0xff) * b) >> 7).min(255);
    (r << 16) | (g << 8) | bl
}

/// Tables that voxlap precomputes once in `initvoxlap` and reuses
/// across every `meltsphere` / `setsphere` call.
///
/// Both tables are indexed by an integer `z ∈ [0, SETSPHMAXRAD)`:
///
/// - `factr[z]` is voxlap's prime-decomposition cache. If `z` is
///   prime, `factr[z][0] == 0`. If `z` is composite,
///   `factr[z][0] * factr[z][1] == z` and `factr[z][0]` is the
///   smallest prime divisor of `z`. `factr[2][0]` is forced to 0.
/// - `logint[z] = ln(z)` (natural log) for `z >= 1`; `logint[0]`
///   is unused.
///
/// Used by [`PowerTables::build_tempfloatbuf`] to compute
/// `i^curpow` cheaply: prime indices go through `exp(ln(i) *
/// curpow)`, composite indices fold into a multiplication of two
/// already-computed entries (avoiding `pow` per index).
#[derive(Debug, Clone)]
pub struct PowerTables {
    pub factr: [[u32; 2]; SETSPHMAXRAD],
    pub logint: [f64; SETSPHMAXRAD],
}

impl Default for PowerTables {
    fn default() -> Self {
        Self::new()
    }
}

impl PowerTables {
    /// Mirror of voxlap's `initvoxlap` factr-/logint-init block
    /// (voxlap5.c:12224-12236). Sieves the prime decomposition and
    /// fills `logint[i] = ln(i)`.
    #[must_use]
    pub fn new() -> Self {
        let mut factr = [[0u32; 2]; SETSPHMAXRAD];

        // Voxlap's hand-written prime sieve. `i` tracks the largest
        // prime ≤ √z; `j` is the next perfect square at which `i`
        // increments by 2. `k` is the previous prime — `factr[k][1]`
        // is rewritten on every iteration so that, by the time the
        // loop has crossed the next prime, `factr[k][1]` holds that
        // next prime (used to walk the prime list during composite
        // testing).
        factr[2][0] = 0;
        let mut i: u32 = 1;
        let mut j: u32 = 9;
        let mut k: usize = 0;
        let mut z: u32 = 3;
        while (z as usize) < SETSPHMAXRAD {
            if z == j {
                j += (i << 2) + 12;
                i += 2;
            }
            factr[z as usize][0] = 0;
            factr[k][1] = z;
            // Walk primes ≤ i looking for a divisor of z.
            let mut zz: u32 = 3;
            while zz <= i {
                if z % zz == 0 {
                    factr[z as usize][0] = zz;
                    factr[z as usize][1] = z / zz;
                    break;
                }
                zz = factr[zz as usize][1];
            }
            if factr[z as usize][0] == 0 {
                k = z as usize;
            }
            // Even number z + 1 is always 2 × ((z+1)/2).
            if (z as usize) + 1 < SETSPHMAXRAD {
                factr[(z as usize) + 1][0] = (z + 1) >> 1;
                factr[(z as usize) + 1][1] = 2;
            }
            z += 2;
        }

        let mut logint = [0.0f64; SETSPHMAXRAD];
        // logint[0] stays 0.0 (unused; voxlap leaves it uninitialised
        // but the meltsphere loop starts at i=2 after special-casing
        // tempfloatbuf[1]=1.0).
        for (zz, slot) in logint.iter_mut().enumerate().skip(1) {
            *slot = f64::ln(zz as f64);
        }

        Self { factr, logint }
    }

    /// Build a per-call `tempfloatbuf` such that
    /// `tempfloatbuf[i] ≈ i.powf(curpow)` for `i ∈ [0, hitrad]`.
    /// `hitrad + 1` is filled with the IEEE-754 max-finite-float
    /// sentinel (`0x7f7fffff`) so the int-bit-pattern comparisons
    /// in meltsphere terminate cleanly at the table edge.
    ///
    /// Port of voxlap5.c:10240-10252. `hitrad` is clamped to
    /// `SETSPHMAXRAD - 2`.
    #[must_use]
    pub fn build_tempfloatbuf(&self, hitrad: i32, curpow: f32) -> [f32; SETSPHMAXRAD] {
        let mut buf = [0.0f32; SETSPHMAXRAD];
        let hitrad_clamped = (hitrad.max(0) as usize).min(SETSPHMAXRAD - 2);

        buf[0] = 0.0;
        if hitrad_clamped >= 1 {
            buf[1] = 1.0;
        }
        // Voxlap mixes f64 / f32 here: logint[i] is f64, curpow is
        // f32 promoted to f64 for the multiply, exp is f64,
        // assignment to tempfloatbuf truncates to f32.
        let curpow_d = f64::from(curpow);
        for i in 2..=hitrad_clamped {
            if self.factr[i][0] == 0 {
                // Prime: tempfloatbuf[i] = exp(log(i) * curpow).
                buf[i] = (self.logint[i] * curpow_d).exp() as f32;
            } else {
                // Composite: factor[a] * factor[b] where a*b == i.
                let a = self.factr[i][0] as usize;
                let b = self.factr[i][1] as usize;
                buf[i] = buf[a] * buf[b];
            }
        }
        // Sentinel: 0x7f7fffff (= f32::MAX bit pattern) lives at
        // hitrad + 1 so the binary-search loops in meltsphere
        // hit a "guaranteed > anything finite" boundary.
        buf[hitrad_clamped + 1] = f32::from_bits(0x7f7f_ffff);
        buf
    }
}

/// Output of [`meltsphere`] — a freshly-built kv6 sprite plus the
/// computed sprite centroid and total solid-voxel count.
#[derive(Debug, Clone)]
pub struct MeltsphereOutput {
    /// Freshly-built kv6 with one record per visible (color) voxel
    /// inside the sphere. `xpiv`/`ypiv`/`zpiv` are set to the
    /// centroid offset within the kv6's local AABB. `vis = 63`
    /// (all 6 faces) and `dir = 0` for every voxel — voxlap's
    /// meltsphere comment flags both as "FIX THIS!!!" so we mirror.
    pub kv6: Kv6,
    /// Centroid in world coordinates. The caller typically
    /// overwrites this with a placement position; voxlap stores it
    /// in `vx5sprite.p` then expects callers to relocate.
    pub p: [f32; 3],
    /// Centroid weight = total solid voxels in the sphere region
    /// (both `Color` and `UnexposedSolid`). Voxlap's return value.
    pub cw: u32,
}

/// Region-grow a sphere of voxels out of the world into a kv6
/// sprite.
///
/// Port of voxlap's `meltsphere` (voxlap5.c:10222-10344). Two-pass:
/// pass 1 counts voxels and tallies the centroid, pass 2 emits
/// voxel records, `xlen` and `ylen` slice counts.
///
/// Returns `None` if the sphere AABB is empty (the hit point is
/// off-map), if the sphere is fully clipped against the world
/// bounds, or if no visible (color) voxels are found.
///
/// # Faithful-port notes
///
/// - `(i==1) && (1)` in voxlap is a placeholder (commented "FIX
///   THIS!!!") that always evaluates to true — meaning unexposed
///   solid voxels are *not* emitted into the kv6 but *do* count
///   toward the centroid via `cw`. We mirror.
/// - Voxlap's loop uses `for(z=z0; z<z1; z++)` with `z1 =
///   min(hit->z+sq+1, ze)`. Because `ze` is voxlap's *inclusive*
///   max bound, the loop excludes `z = ze` and the topmost voxel
///   that the sphere should touch on the +z axis. We mirror this
///   off-by-one — diverging would change the sphere's voxel set
///   and break byte-equality against the C oracle's sprite.
/// - `vis = 63` (all faces visible) and `dir = 0` (normal index 0)
///   per voxel, matching voxlap's stub fields.
/// - Centroid math uses voxlap's `f = 1.0 / (float)cw; p = base +
///   (float)cx * f` exactly (1.0 promoted to double for the divide,
///   truncated to f32, then float-multiplied by the f32-cast
///   centroid sum).
#[allow(clippy::too_many_lines, clippy::similar_names)]
#[must_use]
pub fn meltsphere(
    slab_buf: &[u8],
    column_offsets: &[u32],
    vsid: u32,
    hit: [i32; 3],
    hitrad: i32,
    curpow: f32,
    tables: &PowerTables,
) -> Option<MeltsphereOutput> {
    let vsid_i = vsid as i32;

    // AABB clamp (voxlap5.c:10233-10236).
    let xs = (hit[0] - hitrad).max(0);
    let xe = (hit[0] + hitrad).min(vsid_i - 1);
    let ys = (hit[1] - hitrad).max(0);
    let ye = (hit[1] + hitrad).min(vsid_i - 1);
    let zs = (hit[2] - hitrad).max(0);
    let ze = (hit[2] + hitrad).min(MAXZDIM - 1);
    if xs > xe || ys > ye || zs > ze {
        return None;
    }

    // Clamp hitrad to fit the power-table span (voxlap5.c:10238).
    let hitrad_clamped = if hitrad >= (SETSPHMAXRAD as i32) - 1 {
        (SETSPHMAXRAD as i32) - 2
    } else {
        hitrad
    };
    let buf = tables.build_tempfloatbuf(hitrad_clamped, curpow);
    let h = hitrad_clamped as usize;

    // -- Pass 1 (voxlap5.c:10254-10283) — count voxels + centroid.
    // Voxlap uses int32_t for the centroid sums with implicit
    // overflow. Wrapping_add matches that without UB on the rare
    // huge-radius path.
    let mut cx: i32 = 0;
    let mut cy: i32 = 0;
    let mut cz: i32 = 0;
    let mut cw: i32 = 0;
    let mut numvoxs: u32 = 0;
    let mut sq: usize = 0;

    for x in xs..=xe {
        let dx = (x - hit[0]).unsigned_abs() as usize;
        let ff = buf[h] - buf[dx];
        for y in ys..=ye {
            let dy = (y - hit[1]).unsigned_abs() as usize;
            let f = ff - buf[dy];
            // *(int32_t *)&f > 0 with positive-finite buf entries
            // matches `f > 0.0f` exactly; any negative result of the
            // subtractions above flips the sign bit and tests false.
            if f > 0.0 {
                while buf[sq] < f {
                    sq += 1;
                }
                while sq > 0 && buf[sq] >= f {
                    sq -= 1;
                }
                let sq_i = sq as i32;
                let z0 = (hit[2] - sq_i).max(zs);
                // Voxlap's `z1 = min(hit->z+sq+1, ze)`; the loop
                // `z<z1` excludes `z=ze` even though ze is inclusive.
                // Faithful port — see function-level note above.
                let z1 = (hit[2] + sq_i + 1).min(ze);
                for z in z0..z1 {
                    match getcube(slab_buf, column_offsets, vsid, x, y, z) {
                        Cube::Air => {}
                        Cube::UnexposedSolid => {
                            cx = cx.wrapping_add(x.wrapping_sub(hit[0]));
                            cy = cy.wrapping_add(y.wrapping_sub(hit[1]));
                            cz = cz.wrapping_add(z.wrapping_sub(hit[2]));
                            cw = cw.wrapping_add(1);
                            // Don't emit (i==1 path).
                        }
                        Cube::Color(_) => {
                            cx = cx.wrapping_add(x.wrapping_sub(hit[0]));
                            cy = cy.wrapping_add(y.wrapping_sub(hit[1]));
                            cz = cz.wrapping_add(z.wrapping_sub(hit[2]));
                            cw = cw.wrapping_add(1);
                            numvoxs = numvoxs.wrapping_add(1);
                        }
                    }
                }
            }
        }
    }

    if numvoxs == 0 {
        return None;
    }

    // Centroid (voxlap5.c:10286-10289). Match voxlap's mixed
    // double/float math: 1.0 (double) / cw (float-promoted-to-double)
    // → double, cast to f32; then per-axis: f32 base + f32 sum * f32
    // reciprocal.
    let f_inv = (1.0f64 / f64::from(cw)) as f32;
    let p = [
        hit[0] as f32 + cx as f32 * f_inv,
        hit[1] as f32 + cy as f32 * f_inv,
        hit[2] as f32 + cz as f32 * f_inv,
    ];

    let xsiz = (xe - xs + 1) as u32;
    let ysiz = (ye - ys + 1) as u32;
    let zsiz = (ze - zs + 1) as u32;
    let xpiv = p[0] - xs as f32;
    let ypiv = p[1] - ys as f32;
    let zpiv = p[2] - zs as f32;

    // -- Pass 2 (voxlap5.c:10313-10343) — emit voxel records,
    // xlen, ylen. Same iteration as pass 1 (sq reset to 0 to match).
    let mut voxels: Vec<Voxel> = Vec::with_capacity(numvoxs as usize);
    let mut xlen: Vec<u32> = Vec::with_capacity(xsiz as usize);
    let mut ylen_flat: Vec<u16> = Vec::with_capacity((xsiz as usize) * (ysiz as usize));

    let mut o_x_voxs: u32 = 0;
    let mut o_y_voxs: u32 = 0;
    let mut sq: usize = 0;

    for x in xs..=xe {
        let dx = (x - hit[0]).unsigned_abs() as usize;
        let ff = buf[h] - buf[dx];
        for y in ys..=ye {
            let dy = (y - hit[1]).unsigned_abs() as usize;
            let f = ff - buf[dy];
            if f > 0.0 {
                while buf[sq] < f {
                    sq += 1;
                }
                while sq > 0 && buf[sq] >= f {
                    sq -= 1;
                }
                let sq_i = sq as i32;
                let z0 = (hit[2] - sq_i).max(zs);
                let z1 = (hit[2] + sq_i + 1).min(ze);
                for z in z0..z1 {
                    if let Cube::Color(c) = getcube(slab_buf, column_offsets, vsid, x, y, z) {
                        voxels.push(Voxel {
                            col: lightvox(c),
                            z: (z - zs) as u16,
                            vis: 63,
                            dir: 0,
                        });
                    }
                }
            }
            let cur = voxels.len() as u32;
            ylen_flat.push((cur - o_y_voxs) as u16);
            o_y_voxs = cur;
        }
        let cur = voxels.len() as u32;
        xlen.push(cur - o_x_voxs);
        o_x_voxs = cur;
    }

    // Convert flat ylen into the nested Vec<Vec<u16>> shape that
    // roxlap-formats::Kv6 expects (outer length xsiz, inner ysiz).
    let ylen: Vec<Vec<u16>> = ylen_flat
        .chunks_exact(ysiz as usize)
        .map(<[u16]>::to_vec)
        .collect();

    let kv6 = Kv6 {
        xsiz,
        ysiz,
        zsiz,
        xpiv,
        ypiv,
        zpiv,
        voxels,
        xlen,
        ylen,
        palette: None,
    };

    Some(MeltsphereOutput {
        kv6,
        p,
        cw: cw as u32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- lightvox -------------------------------------------------------

    #[test]
    fn lightvox_neutral_brightness_passes_rgb_through() {
        // Alpha 0x80 = 128, multiplier (x * 128) >> 7 = x.
        assert_eq!(lightvox(0x80ff_4030), 0x00ff_4030);
        assert_eq!(lightvox(0x80ff_ffff), 0x00ff_ffff);
        assert_eq!(lightvox(0x8000_0000), 0x0000_0000);
    }

    #[test]
    fn lightvox_zero_alpha_blackens() {
        assert_eq!(lightvox(0x00ff_ffff), 0);
        assert_eq!(lightvox(0x0080_4020), 0);
    }

    #[test]
    fn lightvox_clamps_at_255() {
        // Alpha 0xff = 255; multiplier (0xff * 255) >> 7 = 510 → clamp 255.
        assert_eq!(lightvox(0xffff_ffff), 0x00ff_ffff);
        // Alpha 0xc0 = 192; (0x80 * 192) >> 7 = 0xc0 = 192. Not clamped.
        assert_eq!(lightvox(0xc080_8080), 0x00c0_c0c0);
        // Alpha 0xc0 with 0xff channel: (0xff * 192) >> 7 = 382 → clamp 255.
        assert_eq!(lightvox(0xc0ff_4030), 0x00ff_6048);
    }

    #[test]
    fn lightvox_half_brightness() {
        // Alpha 0x40 = 64; (x * 64) >> 7 = x / 2.
        assert_eq!(lightvox(0x4080_8080), 0x0040_4040);
    }

    // --- factr / prime sieve --------------------------------------------

    #[test]
    fn factr_known_primes_have_zero_factor() {
        let pt = PowerTables::new();
        // factr[2][0] is force-zero in init.
        for &p in &[
            2u32, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 251,
        ] {
            assert_eq!(
                pt.factr[p as usize][0], 0,
                "prime {p} should have factr[{p}][0] == 0"
            );
        }
    }

    #[test]
    fn factr_composites_decompose_to_their_factors() {
        let pt = PowerTables::new();
        // Spot-check: each is `(z, expected_a, expected_b)` with a*b == z.
        let cases = [
            (4u32, 2u32, 2u32),
            (6, 3, 2),
            (8, 4, 2),
            (9, 3, 3),
            (10, 5, 2),
            (12, 6, 2),
            (15, 3, 5),
            (21, 3, 7),
            (25, 5, 5),
            (27, 3, 9),
            (35, 5, 7),
            (49, 7, 7),
            (121, 11, 11),
            (169, 13, 13),
            // 255 = 3 × 5 × 17; smallest-prime-first → 3 × 85.
            (255, 3, 85),
        ];
        for (z, a, b) in cases {
            assert_eq!(
                pt.factr[z as usize],
                [a, b],
                "factr[{z}] should be [{a}, {b}]"
            );
        }
    }

    #[test]
    fn factr_invariant_holds_for_all_composites() {
        // Voxlap's invariant: if factr[z][0] != 0 then a*b == z.
        let pt = PowerTables::new();
        for z in 2..SETSPHMAXRAD as u32 {
            let a = pt.factr[z as usize][0];
            if a != 0 {
                let b = pt.factr[z as usize][1];
                assert_eq!(a * b, z, "factr[{z}] = [{a}, {b}], product {}", a * b);
            }
        }
    }

    // --- logint ----------------------------------------------------------

    #[test]
    fn logint_matches_natural_log() {
        let pt = PowerTables::new();
        // Compare bit patterns: these are exact-equal floats produced
        // by the same `ln` call, so any drift is a real bug.
        assert_eq!(pt.logint[1].to_bits(), 0.0f64.to_bits());
        assert_eq!(pt.logint[10].to_bits(), (10.0f64).ln().to_bits());
        assert_eq!(pt.logint[100].to_bits(), (100.0f64).ln().to_bits());
        assert_eq!(pt.logint[255].to_bits(), (255.0f64).ln().to_bits());
    }

    // --- tempfloatbuf ---------------------------------------------------

    #[test]
    fn tempfloatbuf_curpow_two_approximates_squares() {
        let pt = PowerTables::new();
        let buf = pt.build_tempfloatbuf(64, 2.0);
        // tempfloatbuf[i] should be very close to i² for curpow=2.
        // Not exactly equal because the prime path goes through
        // exp(ln*2), which carries ULP rounding. Tolerance: 1 ULP
        // of i² in f32, or just relative 1e-6.
        for i in 0..=64u32 {
            let want = (i * i) as f32;
            let got = buf[i as usize];
            let rel = ((got - want) / want.max(1.0)).abs();
            assert!(rel < 1e-5, "tempfloatbuf[{i}] = {got}, want {want}");
        }
    }

    #[test]
    fn tempfloatbuf_zero_and_one_are_exact() {
        let pt = PowerTables::new();
        let buf = pt.build_tempfloatbuf(8, 2.0);
        // Voxlap hard-codes tempfloatbuf[0]=0, tempfloatbuf[1]=1
        // before the prime-walk loop.
        assert_eq!(buf[0].to_bits(), 0.0f32.to_bits());
        assert_eq!(buf[1].to_bits(), 1.0f32.to_bits());
    }

    #[test]
    fn tempfloatbuf_sentinel_is_max_finite_float() {
        let pt = PowerTables::new();
        let buf = pt.build_tempfloatbuf(8, 2.0);
        assert_eq!(buf[9].to_bits(), 0x7f7f_ffff);
    }

    #[test]
    fn tempfloatbuf_clamps_huge_hitrad() {
        let pt = PowerTables::new();
        // hitrad past SETSPHMAXRAD-2 should clamp; the function must
        // not panic and the sentinel should sit at index 255.
        let buf = pt.build_tempfloatbuf(10_000, 2.0);
        assert_eq!(buf[SETSPHMAXRAD - 1].to_bits(), 0x7f7f_ffff);
    }

    #[test]
    fn tempfloatbuf_curpow_three_matches_cubes() {
        let pt = PowerTables::new();
        let buf = pt.build_tempfloatbuf(20, 3.0);
        for i in 0..=20u32 {
            let want = (i as f32).powi(3);
            let got = buf[i as usize];
            let rel = ((got - want) / want.max(1.0)).abs();
            assert!(rel < 1e-4, "tempfloatbuf[{i}] = {got}, want {want}");
        }
    }

    // --- meltsphere -----------------------------------------------------

    /// Build a synthetic 16×16 world. Every column is the minimal
    /// "deep solid only" form (one floor voxel at z=255). Then any
    /// `(x, y, z)` overrides in `paints` add a single visible floor
    /// voxel at that z with the given color via a two-slab column.
    fn synth_world(paints: &[(i32, i32, i32, u32)]) -> (Vec<u8>, Vec<u32>) {
        const VSID: u32 = 16;
        // Empty column: single slab [0, 255, 255, 0] + 1 floor color
        // = 8 bytes per empty column.
        let empty_col: [u8; 8] = [0, 255, 255, 0, 0xff, 0xff, 0xff, 0x80];
        let n_cols = (VSID * VSID) as usize;
        let mut data: Vec<u8> = Vec::new();
        let mut offsets: Vec<u32> = Vec::with_capacity(n_cols + 1);

        // Resolve paints to per-column override.
        let mut overrides: std::collections::HashMap<usize, (i32, u32)> =
            std::collections::HashMap::new();
        for &(x, y, z, c) in paints {
            let idx = y as usize * VSID as usize + x as usize;
            overrides.insert(idx, (z, c));
        }

        for i in 0..n_cols {
            offsets.push(u32::try_from(data.len()).unwrap());
            if let Some(&(z, c)) = overrides.get(&i) {
                // Two slabs:
                //   slab 0 [nextptr=2, z1=z, z1c=z, dummy=0] + 1 floor (4 bytes)
                //   slab 1 [nextptr=0, z1=255, z1c=255, z0=z+1] + 1 floor (4 bytes)
                let z_u8 = z as u8;
                data.extend_from_slice(&[2, z_u8, z_u8, 0]);
                data.extend_from_slice(&c.to_le_bytes());
                data.extend_from_slice(&[0, 255, 255, z_u8 + 1]);
                data.extend_from_slice(&[0xff, 0xff, 0xff, 0x80]);
            } else {
                data.extend_from_slice(&empty_col);
            }
        }
        offsets.push(u32::try_from(data.len()).unwrap());
        (data, offsets)
    }

    #[test]
    fn meltsphere_empty_region_returns_none() {
        // No paints: the entire 16×16 world is one-voxel-deep solid
        // at z=255. Sphere at (8, 8, 4) with radius 2 finds no color
        // voxels in the 5×5×5 AABB (z=2..=6) — only unexposed solid
        // and air. So numvoxs == 0 → None.
        let (buf, off) = synth_world(&[]);
        let pt = PowerTables::new();
        let r = meltsphere(&buf, &off, 16, [8, 8, 4], 2, 2.0, &pt);
        assert!(r.is_none(), "expected None, got {r:?}");
    }

    #[test]
    fn meltsphere_single_voxel_at_center_extracts_one() {
        let (buf, off) = synth_world(&[(8, 8, 4, 0x8011_2233)]);
        let pt = PowerTables::new();
        let out = meltsphere(&buf, &off, 16, [8, 8, 4], 2, 2.0, &pt)
            .expect("sphere should hit the painted voxel");
        // Exactly one color voxel emitted.
        assert_eq!(out.kv6.voxels.len(), 1);
        let v = out.kv6.voxels[0];
        // lightvox(0x8011_2233): alpha=0x80=128, channels pass through.
        assert_eq!(v.col, 0x0011_2233);
        assert_eq!(v.vis, 63);
        assert_eq!(v.dir, 0);
        // Sphere AABB: x ∈ [6,10], y ∈ [6,10], z ∈ [2,6]. zs=2 so
        // the painted voxel at z=4 lands at local z = 4-2 = 2.
        assert_eq!(v.z, 2);
        // kv6 dimensions = AABB clamp to map.
        assert_eq!(out.kv6.xsiz, 5);
        assert_eq!(out.kv6.ysiz, 5);
        assert_eq!(out.kv6.zsiz, 5);
        // numvoxs is the sum of xlen.
        let xsum: u32 = out.kv6.xlen.iter().sum();
        assert_eq!(xsum, 1);
        // ylen rows match xsiz.
        assert_eq!(out.kv6.ylen.len(), 5);
        for row in &out.kv6.ylen {
            assert_eq!(row.len(), 5);
        }
    }

    #[test]
    fn meltsphere_returns_none_when_aabb_off_map() {
        // hit way outside the 16×16 world → AABB clamps to empty.
        let (buf, off) = synth_world(&[]);
        let pt = PowerTables::new();
        // hit at x = -100: xs = max(-102, 0) = 0; xe = min(-98, 15)
        // = -98. xs > xe → None.
        let r = meltsphere(&buf, &off, 16, [-100, 8, 4], 2, 2.0, &pt);
        assert!(r.is_none());
    }

    #[test]
    fn meltsphere_centroid_is_on_painted_voxel() {
        // Single voxel at hit location → centroid == hit location.
        let (buf, off) = synth_world(&[(8, 8, 4, 0x8011_2233)]);
        let pt = PowerTables::new();
        let out = meltsphere(&buf, &off, 16, [8, 8, 4], 2, 2.0, &pt).unwrap();
        // cx = cy = cz = 0 (the only voxel is at offset (0, 0, 0)
        // from hit). centroid = hit + 0/cw = hit.
        assert_eq!(out.p[0].to_bits(), 8.0f32.to_bits());
        assert_eq!(out.p[1].to_bits(), 8.0f32.to_bits());
        assert_eq!(out.p[2].to_bits(), 4.0f32.to_bits());
        assert_eq!(out.cw, 1);
    }

    #[test]
    fn meltsphere_skips_unexposed_solid_but_counts_for_centroid() {
        // Place two color voxels at (8,8,4) and (9,8,4) and let the
        // empty columns' z=255 deep-solid voxels NOT overlap the
        // sphere (sphere stays well above z=10). Then cw should
        // equal numvoxs since no UnexposedSolid voxels are in range.
        // (This tests centroid math is consistent with cw=numvoxs.)
        let (buf, off) = synth_world(&[(8, 8, 4, 0x8000_ff00), (9, 8, 4, 0x8000_00ff)]);
        let pt = PowerTables::new();
        let out = meltsphere(&buf, &off, 16, [8, 8, 4], 2, 2.0, &pt).unwrap();
        assert_eq!(out.kv6.voxels.len(), 2);
        assert_eq!(out.cw, 2);
        // cx = (8-8) + (9-8) = 1; cy = 0; cz = 0.
        // p.x = 8 + 1 * (1/2) = 8.5
        let expected_px = 8.0f32 + 1.0 * (1.0 / 2.0f64) as f32;
        assert_eq!(out.p[0].to_bits(), expected_px.to_bits());
    }

    #[test]
    fn meltsphere_xlen_ylen_partition_voxels() {
        // Three painted voxels on different (x, y) columns. Verify
        // xlen sums to numvoxs and ylen rows sum to xlen.
        let paints = [
            (8, 8, 4, 0x8011_2233),
            (9, 8, 4, 0x8044_5566),
            (8, 9, 4, 0x8077_8899),
        ];
        let (buf, off) = synth_world(&paints);
        let pt = PowerTables::new();
        let out = meltsphere(&buf, &off, 16, [8, 8, 4], 2, 2.0, &pt).unwrap();
        assert_eq!(out.kv6.voxels.len(), 3);
        let xsum: u32 = out.kv6.xlen.iter().sum();
        assert_eq!(xsum, 3);
        for (x_i, ylen_row) in out.kv6.ylen.iter().enumerate() {
            let row_sum: u32 = ylen_row.iter().map(|&v| u32::from(v)).sum();
            assert_eq!(
                row_sum, out.kv6.xlen[x_i],
                "ylen row {x_i} sum {row_sum} != xlen[{x_i}] {}",
                out.kv6.xlen[x_i]
            );
        }
    }
}
