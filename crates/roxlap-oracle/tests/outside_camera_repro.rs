//! S1 closeout — `outside_orbit` golden for the camera-outside-grid path.
//!
//! Pre-S1 this test was non-asserting (printed hash + sky/world
//! split). After S1.0..S1.Z + S1.V + S1.X landed, the outside-XY
//! render path is stable, so this freezes the framebuffer FNV-1a64
//! as a roxlap-only golden — voxlap C can't render this pose at
//! all, so there's no upstream reference to diff against. Per-arch
//! values follow the same `(x86_64, aarch64)` split as
//! `tests/golden-hashes*.txt`; aarch64 row is filled in when first
//! run on that arch (same flow as R9.4).

#![cfg(not(target_arch = "wasm32"))]
#![allow(clippy::cast_precision_loss, clippy::explicit_iter_loop)]

use roxlap_core::camera_math;
use roxlap_core::opticast::{opticast, OpticastOutcome};
use roxlap_core::rasterizer::ScratchPool;
use roxlap_core::scalar_rasterizer::ScalarRasterizer;
use roxlap_core::{Camera, Engine, OpticastSettings};
use roxlap_oracle::{fnv1a64, load_oracle_vxl, XRES, YRES};

/// Frozen `outside_orbit` framebuffer hash. `0` means "no golden
/// pinned for this arch yet" — the test will print the rendered
/// hash so the next run on that arch can paste it back here.
const OUTSIDE_ORBIT_GOLDEN: u64 = if cfg!(target_arch = "x86_64") {
    0xaa2b_4263_e714_667b
} else {
    0
};

#[test]
fn outside_orbit_matches_golden() {
    let vxl = load_oracle_vxl();
    let engine = Engine::new();

    // Camera high above the world, looking back-and-down. Above
    // the terrain top so rays enter through the X face high in
    // voxlap-z (small cz = sky) and pierce DOWN into terrain
    // colours. This is the canonical "orbit" viewpoint S1 wants.
    let pos = [
        f64::from(vxl.vsid) + 512.0,
        f64::from(vxl.vsid) / 2.0,
        -300.0,
    ];
    let yaw: f64 = std::f64::consts::PI;
    let pitch: f64 = std::f64::consts::FRAC_PI_4;

    let cy = yaw.cos();
    let sy = yaw.sin();
    let cp = pitch.cos();
    let sp = pitch.sin();
    let cam = Camera {
        pos,
        right: [-sy, cy, 0.0],
        down: [-cy * sp, -sy * sp, cp],
        forward: [cy * cp, sy * cp, sp],
    };

    let mut framebuffer = vec![0u32; (XRES * YRES) as usize];
    let mut zbuffer = vec![0f32; (XRES * YRES) as usize];
    let mut pool = ScratchPool::new(XRES, YRES, vxl.vsid);

    let sky = engine.sky_color();
    for px in framebuffer.iter_mut() {
        *px = sky;
    }
    let sky_col_i = i32::from_ne_bytes(sky.to_ne_bytes());
    pool.set_skycast(sky_col_i, 0);
    let fog_col_i = i32::from_ne_bytes(engine.fog_color().to_ne_bytes());
    pool.set_fog(fog_col_i, engine.fog_max_scan_dist());

    let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
    let pitch_pixels = XRES as usize;

    let _cs = camera_math::derive(&cam, XRES, YRES, settings.hx, settings.hy, settings.hz);

    let mut rasterizer = ScalarRasterizer::new(
        &mut framebuffer,
        &mut zbuffer,
        pitch_pixels,
        &vxl.data,
        &vxl.column_offset,
        &vxl.mip_base_offsets,
        vxl.vsid,
    );
    let outcome = opticast(
        &mut rasterizer,
        &mut pool,
        &cam,
        &settings,
        vxl.vsid,
        &vxl.data,
        &vxl.column_offset,
    );
    drop(rasterizer);

    // S1.Z: outside-XY camera now goes through the standard
    // inside-camera path with a signed-coords negative-index walk.
    // Expect Rendered.
    assert_eq!(outcome, OpticastOutcome::Rendered);

    let mut bytes = Vec::with_capacity(framebuffer.len() * 4);
    for &px in framebuffer.iter() {
        bytes.extend_from_slice(&px.to_ne_bytes());
    }
    let hash = fnv1a64(&bytes);

    let sky_pixels = framebuffer.iter().filter(|&&p| p == sky).count();
    let total = framebuffer.len();
    let world_pixels = total - sky_pixels;

    eprintln!("outside_orbit hash={hash:016x}");
    eprintln!(
        "  sky={sky_pixels}/{total} ({:.1}%)  world={world_pixels} ({:.1}%)",
        100.0 * sky_pixels as f64 / total as f64,
        100.0 * world_pixels as f64 / total as f64,
    );

    // Dump a PPM next to the test binary so we can eyeball the
    // result. Path is dynamic ($CARGO_TARGET_DIR or default) but
    // for a test it's enough to land it in /tmp.
    let mut ppm = format!("P6\n{XRES} {YRES}\n255\n").into_bytes();
    for &px in &framebuffer {
        let bytes = px.to_le_bytes();
        // voxlap framebuffer is BGRA-ish; PPM wants RGB.
        ppm.push(bytes[2]);
        ppm.push(bytes[1]);
        ppm.push(bytes[0]);
    }
    let path = "/tmp/outside_orbit.ppm";
    std::fs::write(path, ppm).expect("write ppm");
    eprintln!("  wrote {path}");

    if OUTSIDE_ORBIT_GOLDEN == 0 {
        eprintln!(
            "  no frozen golden for this arch yet — paste {hash:016x} into \
             OUTSIDE_ORBIT_GOLDEN's matching cfg arm to freeze."
        );
    } else {
        assert_eq!(
            hash, OUTSIDE_ORBIT_GOLDEN,
            "outside_orbit hash drifted: got {hash:016x}, want {OUTSIDE_ORBIT_GOLDEN:016x}",
        );
    }
}
