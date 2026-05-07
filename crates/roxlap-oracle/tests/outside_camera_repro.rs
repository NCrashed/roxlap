//! S1.0 reproducer — camera placed outside the world's X/Y bounds.
//!
//! Today (pre-S1) voxlap's opticast assumes the camera sits inside
//! the column grid. With camera at `(vsid + 256, vsid/2, 128)` the
//! initial column index in `opticast_prelude::derive_prelude`
//! silently overflows (the f32→i32→u32 cast wraps to a huge value),
//! so the renderer either accesses garbage column data or, when
//! `camera_column_slice` early-outs, the rest of the rasterizer
//! sees uninitialised state.
//!
//! The test is intentionally non-asserting: it prints the output
//! hash + a "sky vs world" pixel split so we can see at a glance
//! whether the failure today is panic / all-sky / corrupt.
//! After S1.2 + S1.3 land, the same harness becomes the validation
//! for the fix.

#![cfg(not(target_arch = "wasm32"))]

use roxlap_core::camera_math;
use roxlap_core::opticast::{opticast, OpticastOutcome};
use roxlap_core::rasterizer::ScratchPool;
use roxlap_core::scalar_rasterizer::ScalarRasterizer;
use roxlap_core::{Camera, Engine, OpticastSettings};
use roxlap_oracle::{fnv1a64, load_oracle_vxl, XRES, YRES};

#[test]
fn outside_camera_renders_without_panic() {
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

    assert_eq!(outcome, OpticastOutcome::OutsideCamera);

    let mut bytes = Vec::with_capacity(framebuffer.len() * 4);
    for &px in framebuffer.iter() {
        bytes.extend_from_slice(&px.to_ne_bytes());
    }
    let hash = fnv1a64(&bytes);

    let sky_pixels = framebuffer.iter().filter(|&&p| p == sky).count();
    let total = framebuffer.len();
    let world_pixels = total - sky_pixels;

    eprintln!("outside_orbit (pre-S1) hash={hash:016x}");
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
}
