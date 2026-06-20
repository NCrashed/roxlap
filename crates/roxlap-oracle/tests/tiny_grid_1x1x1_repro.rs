//! TODO(cpu-renderer): THIS TEST IS EXPECTED TO FAIL (red).
//! It pins a known, unfixed bug in the CPU grouscan marcher and is left
//! failing on purpose as a living TODO until that fix lands in a later
//! session. Do NOT `#[ignore]` it -- the red is the reminder. When the
//! marcher is fixed it should go green with no other change; if you need
//! CI green before then, fix the marcher, do not silence the test.
//!
//! Repro: a single voxel (demiurg's 1x1x1 model) renders a *non-convex*
//! silhouette at certain camera angles — the CPU marcher punches sky
//! holes into what must be a solid cube outline. Reported via demiurg's
//! `--shot` on a 1x1x1 `.demiurg` (voxel at the corner column, z=0),
//! viewed from below at `yaw=-1.043, pitch=-0.782`.
//!
//! Despite the "1x1x1 / outside the grid" framing, the bug is NOT
//! edge-column-specific: rendering the same lone voxel deep in the
//! interior of a large grid produces a byte-identical (equally broken)
//! silhouette (`diff=0` below). It is a single-voxel / thin-geometry
//! artifact — it only *shows* on a lone voxel, because on a solid model
//! the broken outline of one voxel is hidden among its neighbours
//! ("не были видны на больших моделях").
//!
//! Voxlap C has no reference for this, so the invariant is geometric: a
//! cube projects to a convex polygon, so the pixel silhouette must have
//! no internal sky gap in any row or column. `span_gaps` counts the
//! violations; a correct renderer yields zero.

#![cfg(not(target_arch = "wasm32"))]
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use roxlap_core::camera_math;
use roxlap_core::opticast::{opticast, OpticastOutcome};
use roxlap_core::rasterizer::ScratchPool;
use roxlap_core::scalar_rasterizer::ScalarRasterizer;
use roxlap_core::{Camera, Engine, GridView, OpticastSettings};
use roxlap_formats::vxl::Vxl;
use roxlap_oracle::{XRES, YRES};

/// Z of the lone voxel (voxlap z-down). demiurg places a model's z=0
/// voxel at the very top of the world (z=0), so the corner voxel is on
/// the boundary in z as well as x/y — exactly the "outside the grid"
/// regime where the bug lives.
const VOX_Z: u32 = 0;

/// A rendered silhouette: the set of non-sky pixels, normalised so the
/// bounding box starts at (0, 0). Two renders of the same voxel from
/// the same relative camera must yield equal masks.
struct Silhouette {
    /// `(dx, dy)` of every world pixel relative to the bbox top-left.
    mask: std::collections::BTreeSet<(u32, u32)>,
    w: u32,
    h: u32,
    count: usize,
}

/// Render a single voxel at grid column `(vx, vy)` (z = `VOX_Z`) in a
/// `vsid`-sized world, from a three-quarter view at the given yaw /
/// pitch / dist aimed at the voxel centre. Returns the normalised
/// silhouette (and dumps a PPM to `ppm_path` for eyeballing).
fn render_voxel(
    vsid: u32,
    vx: u32,
    vy: u32,
    look_off: [f64; 3],
    yaw: f64,
    pitch: f64,
    dist: f64,
    ppm_path: &str,
) -> Silhouette {
    let vxl = Vxl::from_dense(vsid, |x, y, z| {
        if x == vx && y == vy && z == VOX_Z {
            Some(0x8040_80c0) // voxlap-packed 0x80RRGGBB, a lit blue-ish voxel
        } else {
            None
        }
    });
    assert_eq!(vxl.vsid, vsid);

    let engine = Engine::new();
    // Aim at `look_off` measured from the voxel's min corner, so the same
    // relative viewpoint is used for the corner and interior voxels.
    let center = [
        f64::from(vx) + look_off[0],
        f64::from(vy) + look_off[1],
        f64::from(VOX_Z) + look_off[2],
    ];
    let (cy, sy) = (yaw.cos(), yaw.sin());
    let (cp, sp) = (pitch.cos(), pitch.sin());
    let forward = [cy * cp, sy * cp, sp];
    let pos = [
        center[0] - dist * forward[0],
        center[1] - dist * forward[1],
        center[2] - dist * forward[2],
    ];
    let cam = Camera {
        pos,
        right: [-sy, cy, 0.0],
        down: [-cy * sp, -sy * sp, cp],
        forward,
    };

    let mut framebuffer = vec![0u32; (XRES * YRES) as usize];
    let mut zbuffer = vec![0f32; (XRES * YRES) as usize];
    let mut pool = ScratchPool::new(XRES, YRES, vxl.vsid);

    let sky = engine.sky_color();
    for px in framebuffer.iter_mut() {
        *px = sky;
    }
    pool.set_skycast(i32::from_ne_bytes(sky.to_ne_bytes()), 0);
    pool.set_fog(
        i32::from_ne_bytes(engine.fog_color().to_ne_bytes()),
        engine.fog_max_scan_dist(),
    );
    // Match demiurg's voxel render: the top of the world (z=255) is air,
    // not bedrock — relevant for a voxel sitting on the z boundary.
    pool.set_treat_z_max_as_air(true);

    let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
    let _cs = camera_math::derive(&cam, XRES, YRES, settings.hx, settings.hy, settings.hz);

    let grid = GridView::from_single_vxl(&vxl);
    let mut rasterizer = ScalarRasterizer::new(&mut framebuffer, &mut zbuffer, XRES as usize, grid);
    let outcome = opticast(&mut rasterizer, &mut pool, &cam, &settings, grid);
    drop(rasterizer);
    assert_eq!(
        outcome,
        OpticastOutcome::Rendered,
        "vsid={vsid} must render"
    );

    // World-pixel bounding box.
    let (mut x0, mut y0, mut x1, mut y1) = (u32::MAX, u32::MAX, 0u32, 0u32);
    let mut count = 0usize;
    for y in 0..YRES {
        for x in 0..XRES {
            if framebuffer[(y * XRES + x) as usize] != sky {
                count += 1;
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x);
                y1 = y1.max(y);
            }
        }
    }

    // Dump a PPM (voxlap framebuffer is BGRA-ish).
    let mut ppm = format!("P6\n{XRES} {YRES}\n255\n").into_bytes();
    for &px in &framebuffer {
        let b = px.to_le_bytes();
        ppm.push(b[2]);
        ppm.push(b[1]);
        ppm.push(b[0]);
    }
    std::fs::write(ppm_path, ppm).expect("write ppm");

    let mut mask = std::collections::BTreeSet::new();
    if count > 0 {
        for y in y0..=y1 {
            for x in x0..=x1 {
                if framebuffer[(y * XRES + x) as usize] != sky {
                    mask.insert((x - x0, y - y0));
                }
            }
        }
    }
    let (w, h) = if count > 0 {
        (x1 - x0 + 1, y1 - y0 + 1)
    } else {
        (0, 0)
    };
    Silhouette { mask, w, h, count }
}

/// Count rows and columns whose world pixels are not a single contiguous
/// run (i.e. there is a sky gap between two world pixels). A single
/// voxel's silhouette is a convex polygon, so a correct render has zero of
/// each — any gap is a hole/notch artifact.
fn span_gaps(s: &Silhouette) -> (u32, u32) {
    let mut row_gaps = 0;
    for y in 0..s.h {
        let xs: Vec<u32> = (0..s.w).filter(|&x| s.mask.contains(&(x, y))).collect();
        if let (Some(&lo), Some(&hi)) = (xs.first(), xs.last()) {
            if (hi - lo + 1) as usize != xs.len() {
                row_gaps += 1;
            }
        }
    }
    let mut col_gaps = 0;
    for x in 0..s.w {
        let ys: Vec<u32> = (0..s.h).filter(|&y| s.mask.contains(&(x, y))).collect();
        if let (Some(&lo), Some(&hi)) = (ys.first(), ys.last()) {
            if (hi - lo + 1) as usize != ys.len() {
                col_gaps += 1;
            }
        }
    }
    (row_gaps, col_gaps)
}

#[test]
fn single_voxel_silhouette_is_convex() {
    // Sweep a handful of views; the steep from-below ones are where the
    // demiurg screenshot showed the serrated bottom-face silhouette.
    // (name, look_off-from-voxel-corner, yaw, pitch, dist). The
    // `single_demiurg` view is the exact camera a user reported the broken
    // (non-convex) cube silhouette at: a 1x1x1 model (voxel in the corner
    // column, z=0), viewed from below.
    let views: &[(&str, [f64; 3], f64, f64, f64)] = &[
        (
            "single_demiurg",
            [0.5044, 0.2619, 0.0277],
            -1.043_44,
            -0.781_99,
            8.0,
        ),
        (
            "from_below",
            [0.5, 0.5, 0.5],
            std::f64::consts::FRAC_PI_4,
            -1.2,
            24.0,
        ),
        (
            "from_below_steep",
            [0.5, 0.5, 0.5],
            std::f64::consts::FRAC_PI_4,
            -1.45,
            24.0,
        ),
        ("low_angle", [0.5, 0.5, 0.5], 0.3, -0.25, 24.0),
    ];

    const VSID: u32 = 64;
    let mut failures = Vec::new();
    for &(name, off, yaw, pitch, dist) in views {
        // A single voxel is a cube — its silhouette is a convex polygon, so
        // it must have no internal sky gaps. Render it both in the corner
        // column (0, 0) and deep in the interior (32, 32): the bug is NOT
        // edge-specific (the two are byte-identical — `diff` below), it just
        // only *shows* on a lone voxel; on a solid model the broken outline
        // is hidden among neighbouring voxels.
        let corner = render_voxel(
            VSID,
            0,
            0,
            off,
            yaw,
            pitch,
            dist,
            &format!("/tmp/tiny_1x1x1_{name}_corner.ppm"),
        );
        let interior = render_voxel(
            VSID,
            32,
            32,
            off,
            yaw,
            pitch,
            dist,
            &format!("/tmp/tiny_1x1x1_{name}_interior.ppm"),
        );

        let diff = corner.mask.symmetric_difference(&interior.mask).count();
        let (row_gaps, col_gaps) = span_gaps(&corner);
        eprintln!(
            "[{name}] {}px {}x{} | corner-vs-interior diff={diff}px | row_gaps={row_gaps} col_gaps={col_gaps}",
            corner.count, corner.w, corner.h
        );
        if row_gaps != 0 || col_gaps != 0 {
            failures.push(format!(
                "{name} (yaw={yaw} pitch={pitch}): single-voxel silhouette has \
                 {row_gaps} row + {col_gaps} col sky-gaps — a cube must be convex \
                 (see /tmp/tiny_1x1x1_{name}_corner.ppm)"
            ));
        }
    }

    // TODO(cpu-renderer): expected-red. Goes green when the grouscan
    // thin-geometry miss is fixed; do not silence it before then.
    assert!(
        failures.is_empty(),
        "TODO(cpu-renderer): single-voxel silhouette is not convex \
         (known grouscan marcher artifact, fix pending):\n  {}",
        failures.join("\n  ")
    );
}
