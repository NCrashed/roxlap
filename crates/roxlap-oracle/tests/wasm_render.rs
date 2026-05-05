//! R10.1 — wasm32 scalar render baseline.
//!
//! A wasm-bindgen-test that allocates a 640×480 framebuffer + zbuffer,
//! parses the embedded oracle world, runs `opticast` for the `north`
//! pose, FNV-1a64-hashes the framebuffer bytes, and prints the hash
//! to Node's console. Runs the render twice to verify the wasm
//! scalar path is deterministic (no goldens yet — those land in
//! R10.4).
//!
//! On native (non-wasm32) targets the whole file is cfg-gated out;
//! `cargo test` over native builds skips it without trying to pull
//! the wasm-bindgen-test deps.

#![cfg(target_arch = "wasm32")]

use flate2::read::GzDecoder;
use roxlap_core::opticast::opticast;
use roxlap_core::rasterizer::ScratchPool;
use roxlap_core::scalar_rasterizer::ScalarRasterizer;
use roxlap_core::{Camera, Engine, OpticastSettings};
use roxlap_formats::vxl;
use std::io::Read;
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_node_experimental);

const XRES: u32 = 640;
const YRES: u32 = 480;

// Embedded oracle world — same gzipped fixture the native bin uses.
// `include_bytes!` keeps the wasm self-contained (no fetch / async
// asset loading needed for v1; that's R10.X).
const ORACLE_VXL_GZ: &[u8] = include_bytes!("../../../assets/oracle.vxl.gz");

fn fnv1a64(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// `north` pose from `oracle.c`: yaw = π/2, pitch = 0 at world
/// (1024, 1024, 128). Hand-rolled basis matches voxlap's
/// `set_camera_yaw_pitch` exactly (right × down = forward, RH).
fn camera_north() -> Camera {
    Camera {
        pos: [1024.0, 1024.0, 128.0],
        right: [-1.0, 0.0, 0.0],
        down: [0.0, 0.0, 1.0],
        forward: [0.0, 1.0, 0.0],
    }
}

fn render_north(world: &vxl::Vxl, engine: &Engine) -> u64 {
    let sky = engine.sky_color();
    let mut fb = vec![sky; (XRES * YRES) as usize];
    let mut zb = vec![0f32; (XRES * YRES) as usize];

    let mut pool = ScratchPool::new(XRES, YRES, world.vsid);
    let sky_i = i32::from_ne_bytes(sky.to_ne_bytes());
    pool.set_skycast(sky_i, 0);
    let fog_i = i32::from_ne_bytes(engine.fog_color().to_ne_bytes());
    pool.set_fog(fog_i, engine.fog_max_scan_dist());

    let cam = camera_north();
    let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
    {
        let mut rasterizer = ScalarRasterizer::new(
            &mut fb,
            &mut zb,
            XRES as usize,
            &world.data,
            &world.column_offset,
            &world.mip_base_offsets,
            world.vsid,
        );
        let _ = opticast(
            &mut rasterizer,
            &mut pool,
            &cam,
            &settings,
            world.vsid,
            &world.data,
            &world.column_offset,
        );
    }

    let mut bytes = Vec::with_capacity(fb.len() * 4);
    for &px in fb.iter() {
        bytes.extend_from_slice(&px.to_ne_bytes());
    }
    fnv1a64(&bytes)
}

#[wasm_bindgen_test]
fn north_pose_hash_is_stable() {
    console_error_panic_hook::set_once();

    let mut bytes = Vec::with_capacity(40 * 1024 * 1024);
    GzDecoder::new(ORACLE_VXL_GZ)
        .read_to_end(&mut bytes)
        .expect("gunzip oracle.vxl.gz");
    let world = vxl::parse(&bytes).expect("parse oracle.vxl");
    let engine = Engine::new();

    let h1 = render_north(&world, &engine);
    let h2 = render_north(&world, &engine);

    web_sys_console_log(&format!("R10.1 north pose fb fnv1a64 = {h1:016x}"));
    assert_eq!(h1, h2, "wasm scalar render produced non-deterministic hash");
}

// Tiny inline shim so we don't need the full `web-sys` crate just
// to print one string. wasm-bindgen-test pipes Node's `console.log`
// to its captured-output channel; this is the canonical way to
// surface diagnostic strings from a wasm test.
#[wasm_bindgen::prelude::wasm_bindgen]
extern "C" {
    #[wasm_bindgen::prelude::wasm_bindgen(js_namespace = console, js_name = log)]
    fn console_log(s: &str);
}
fn web_sys_console_log(s: &str) {
    console_log(s);
    let _ = JsValue::from_str(s);
}
