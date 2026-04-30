//! roxlap-oracle — cross-engine render-hash oracle.
//!
//! R8 deliverable. Mirrors voxlaptest's `tests/oracle/oracle.c`:
//! loads the same `oracle.vxl.gz` fixture, renders the 4
//! opticast-only poses (`north`, `east`, `diag_down`,
//! `high_down`), FNV-1a-64 hashes each framebuffer, writes
//! `roxlap-hashes.txt` in the same `name  hex_hash\n` format
//! voxlap's `hashes.txt` uses.
//!
//! Sprite (`sprite_*`), lighting (`*_lit`) and tile (`tile_*`)
//! poses from the C oracle are skipped — those exercise R6
//! (sprites), engine lighting, and a tile-blit primitive that
//! roxlap doesn't yet implement.
//!
//! ## Hash equivalence with voxlap
//!
//! The hashes will NOT match voxlaptest's `golden-hashes.txt`
//! initially. Known divergences carried from R4.4 / R5:
//! - `gcsub` is hard-coded zero (no sideshademode), so voxel
//!   side-shading isn't applied.
//! - Mip transition (`remiporend` full body) is unported (R4.5).
//! - Textured sky branch in `startsky` is unported (R4.4 part 2).
//! - rsqrtps is a 12-bit approximation; voxlap C's matching
//!   path uses the same instruction so the SSE z-batch should
//!   converge bit-equally once the upstream divergences close.
//! - Floating-point round-trips between f64 (camera basis) and
//!   f32 (frustum math) introduce 1-ULP differences in some
//!   places; voxlap's C path is mostly f32 throughout.
//!
//! Use `cargo run -p roxlap-oracle -- diff` to compare the
//! freshly-produced `roxlap-hashes.txt` against
//! `voxlaptest/tests/oracle/golden-hashes.txt` (path passed via
//! `--golden`). The diff tool lists per-pose match / mismatch
//! so progress on closing the divergences shows up as more
//! matches.

use std::fs;
use std::io::{Read, Write};

use flate2::read::GzDecoder;
use roxlap_core::opticast;
use roxlap_core::rasterizer::ScanScratch;
use roxlap_core::scalar_rasterizer::ScalarRasterizer;
use roxlap_core::Camera;
use roxlap_core::Engine;
use roxlap_core::OpticastSettings;
use roxlap_formats::vxl;

const XRES: u32 = 640;
const YRES: u32 = 480;
const ORACLE_VXL_GZ: &[u8] = include_bytes!("../../../assets/oracle.vxl.gz");

/// One render pose. Mirror of voxlaptest's `struct pose`, minus
/// the sprite / lit / tile fields (skipped here).
#[derive(Debug, Clone, Copy)]
struct Pose {
    name: &'static str,
    px: f64,
    py: f64,
    pz: f64,
    yaw: f64,
    pitch: f64,
}

/// The 4 opticast-only poses. Same names + positions as
/// voxlaptest's oracle so the diff line-by-line comparison
/// works.
const POSES: &[Pose] = &[
    Pose {
        name: "north",
        px: 1024.0,
        py: 1024.0,
        pz: 128.0,
        yaw: std::f64::consts::FRAC_PI_2,
        pitch: 0.0,
    },
    Pose {
        name: "east",
        px: 1024.0,
        py: 1024.0,
        pz: 128.0,
        yaw: 0.0,
        pitch: 0.0,
    },
    Pose {
        name: "diag_down",
        px: 1000.0,
        py: 1000.0,
        pz: 110.0,
        yaw: std::f64::consts::FRAC_PI_4,
        pitch: 0.4,
    },
    Pose {
        name: "high_down",
        px: 1024.0,
        py: 1024.0,
        pz: 90.0,
        yaw: std::f64::consts::FRAC_PI_2,
        pitch: 0.7,
    },
];

/// FNV-1a 64-bit, byte-for-byte the same as voxlaptest's
/// `tests/oracle/oracle.c` `fnv1a64`. Comparisons against
/// voxlaptest's `golden-hashes.txt` need this exact constant
/// pair (offset basis 0xcbf29ce484222325, prime
/// 0x100000001b3).
fn fnv1a64(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Build a camera basis from yaw + pitch — mirror of voxlaptest's
/// `set_camera_yaw_pitch` (oracle.c:259), minus the
/// `dorthonormalize` call. The constructed basis is already
/// orthonormal for any yaw/pitch combo; voxlap's defensive
/// re-orthonormalize is a no-op here.
fn camera_for_pose(pose: &Pose) -> Camera {
    let cy = pose.yaw.cos();
    let sy = pose.yaw.sin();
    let cp = pose.pitch.cos();
    let sp = pose.pitch.sin();
    Camera {
        pos: [pose.px, pose.py, pose.pz],
        right: [-sy, cy, 0.0],
        down: [-cy * sp, -sy * sp, cp],
        forward: [cy * cp, sy * cp, sp],
    }
}

/// Decompress + parse the embedded oracle world.
fn load_oracle_vxl() -> vxl::Vxl {
    let mut decoder = GzDecoder::new(ORACLE_VXL_GZ);
    let mut bytes = Vec::with_capacity(40 * 1024 * 1024);
    decoder
        .read_to_end(&mut bytes)
        .expect("gunzip oracle.vxl.gz");
    vxl::parse(&bytes).expect("parse oracle.vxl")
}

/// Render one pose into `framebuffer` (XRES × YRES, row-major
/// `u32`). Returns the FNV-1a 64-bit hash of the framebuffer
/// bytes — same semantic as voxlap's `fnv1a64(g_fb,
/// sizeof(g_fb))`.
fn render_pose(
    engine: &Engine,
    vxl: &vxl::Vxl,
    pose: &Pose,
    framebuffer: &mut [u32],
    zbuffer: &mut [f32],
    scratch: &mut ScanScratch,
) -> u64 {
    // Pre-fill with sky-blue, matching voxlaptest's
    // `BR(0x87ceeb) = 0x8087ceeb` per-frame fill.
    let sky = engine.sky_color();
    for px in framebuffer.iter_mut() {
        *px = sky;
    }

    // Wire engine sky / fog onto scratch (same shape as the host).
    let sky_col_i = i32::from_ne_bytes(engine.sky_color().to_ne_bytes());
    scratch.set_skycast(sky_col_i, 0);
    let fog_col_i = i32::from_ne_bytes(engine.fog_color().to_ne_bytes());
    scratch.set_fog(fog_col_i, engine.fog_max_scan_dist());

    let cam = camera_for_pose(pose);
    let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
    let pitch_pixels = XRES as usize;

    {
        let mut rasterizer = ScalarRasterizer::new(
            framebuffer,
            zbuffer,
            pitch_pixels,
            &vxl.data,
            &vxl.column_offset,
            vxl.vsid,
        );
        let _ = opticast(
            &mut rasterizer,
            scratch,
            &cam,
            &settings,
            vxl.vsid,
            &vxl.data,
            &vxl.column_offset,
        );
    }

    // FNV-1a over the framebuffer's raw bytes — same shape as
    // voxlap's `fnv1a64(g_fb, sizeof(g_fb))`. Cast u32 → bytes
    // via to_ne_bytes; voxlap stores the framebuffer as int32_t
    // host-endian so this matches when both engines run on the
    // same endian (LE in practice — both x86 + aarch64).
    let mut bytes = Vec::with_capacity(framebuffer.len() * 4);
    for &px in framebuffer.iter() {
        bytes.extend_from_slice(&px.to_ne_bytes());
    }
    fnv1a64(&bytes)
}

/// Format hashes lines as `name  hex_hash\n` — same shape as
/// voxlap's `fprintf(hf, "%s  %016llx\n", ...)`.
fn format_hashes(rows: &[(&str, u64)]) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    for (name, hash) in rows {
        let _ = writeln!(out, "{name}  {hash:016x}");
    }
    out
}

/// Render every pose and write `roxlap-hashes.txt` next to the
/// invocation cwd.
fn cmd_render() -> std::io::Result<()> {
    let engine = Engine::new();
    let vxl_world = load_oracle_vxl();

    let pixel_count = (XRES as usize) * (YRES as usize);
    let mut framebuffer = vec![0u32; pixel_count];
    let mut zbuffer = vec![0.0f32; pixel_count];
    let mut scratch = ScanScratch::new_for_size(XRES, YRES, vxl_world.vsid);

    let mut rows: Vec<(&str, u64)> = Vec::with_capacity(POSES.len());
    for pose in POSES {
        let hash = render_pose(
            &engine,
            &vxl_world,
            pose,
            &mut framebuffer,
            &mut zbuffer,
            &mut scratch,
        );
        println!("{:<14}  {:016x}", pose.name, hash);
        rows.push((pose.name, hash));
    }

    let body = format_hashes(&rows);
    fs::write("roxlap-hashes.txt", &body)?;
    println!("\nwrote roxlap-hashes.txt ({} bytes)", body.len());
    Ok(())
}

/// Diff `roxlap-hashes.txt` against a `golden-hashes.txt` from
/// voxlaptest. Lists per-pose match / mismatch / missing.
/// Exits non-zero on any mismatch so CI can gate on it.
fn cmd_diff(roxlap_path: &str, golden_path: &str) -> std::io::Result<i32> {
    let roxlap = fs::read_to_string(roxlap_path)?;
    let golden = fs::read_to_string(golden_path)?;
    let parse = |s: &str| -> Vec<(String, String)> {
        s.lines()
            .filter_map(|line| {
                let mut it = line.split_whitespace();
                let name = it.next()?.to_string();
                let hash = it.next()?.to_string();
                Some((name, hash))
            })
            .collect()
    };
    let roxlap_rows = parse(&roxlap);
    let golden_rows = parse(&golden);

    let mut mismatches = 0;
    let mut matches = 0;
    let mut missing = 0;
    for (rname, rhash) in &roxlap_rows {
        if let Some((_, ghash)) = golden_rows.iter().find(|(g, _)| g == rname) {
            if rhash == ghash {
                println!("MATCH    {rname}  {rhash}");
                matches += 1;
            } else {
                println!("MISMATCH {rname}  rox={rhash}  c={ghash}");
                mismatches += 1;
            }
        } else {
            println!("MISSING  {rname} (not in golden)");
            missing += 1;
        }
    }
    println!(
        "\n{} match, {} mismatch, {} missing-from-golden ({} total roxlap rows)",
        matches,
        mismatches,
        missing,
        roxlap_rows.len()
    );
    Ok(i32::from(mismatches != 0))
}

fn print_help() {
    let _ = writeln!(
        std::io::stderr(),
        "roxlap-oracle — render-hash oracle for the roxlap engine.\n\
         \n\
         Usage:\n\
             roxlap-oracle           render every pose, write roxlap-hashes.txt\n\
             roxlap-oracle render    same as above\n\
             roxlap-oracle diff [--golden PATH] [--ours PATH]\n\
                                      diff roxlap-hashes.txt against voxlaptest's\n\
                                      golden-hashes.txt; exits non-zero on any mismatch.\n\
                                      Defaults: --ours=roxlap-hashes.txt,\n\
                                      --golden=../voxlaptest/tests/oracle/golden-hashes.txt"
    );
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map_or("render", String::as_str);
    match cmd {
        "render" => cmd_render(),
        "diff" => {
            let mut ours = "roxlap-hashes.txt".to_string();
            let mut golden = "../voxlaptest/tests/oracle/golden-hashes.txt".to_string();
            let mut i = 1;
            while i < args.len() {
                match args[i].as_str() {
                    "--ours" => {
                        ours = args.get(i + 1).cloned().ok_or_else(|| {
                            std::io::Error::new(
                                std::io::ErrorKind::InvalidInput,
                                "--ours expects a path",
                            )
                        })?;
                        i += 2;
                    }
                    "--golden" => {
                        golden = args.get(i + 1).cloned().ok_or_else(|| {
                            std::io::Error::new(
                                std::io::ErrorKind::InvalidInput,
                                "--golden expects a path",
                            )
                        })?;
                        i += 2;
                    }
                    other => {
                        let _ = writeln!(std::io::stderr(), "unknown diff arg: {other}");
                        print_help();
                        std::process::exit(2);
                    }
                }
            }
            let exit = cmd_diff(&ours, &golden)?;
            std::process::exit(exit);
        }
        "--help" | "-h" => {
            print_help();
            Ok(())
        }
        other => {
            let _ = writeln!(std::io::stderr(), "unknown subcommand: {other}");
            print_help();
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv1a64_known_strings() {
        // Canonical FNV-1a-64 test vectors that pin the constants
        // (offset basis 0xcbf29ce484222325, prime 0x100000001b3) —
        // same shape as voxlaptest's oracle.c `fnv1a64`. Both
        // outputs come from the published FNV reference.
        assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
    }

    #[test]
    fn fnv1a64_is_deterministic_and_changes_with_input() {
        let h1 = fnv1a64(b"hello");
        let h2 = fnv1a64(b"hello");
        let h3 = fnv1a64(b"hellp"); // last byte differs
        assert_eq!(h1, h2, "same input must hash to same value");
        assert_ne!(h1, h3, "different input must hash differently");
    }

    #[test]
    fn camera_north_pose_basis_is_orthonormal() {
        let cam = camera_for_pose(&POSES[0]); // north
        let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        // Each axis unit-length.
        assert!((dot(cam.right, cam.right) - 1.0).abs() < 1e-12);
        assert!((dot(cam.down, cam.down) - 1.0).abs() < 1e-12);
        assert!((dot(cam.forward, cam.forward) - 1.0).abs() < 1e-12);
        // Pairwise orthogonal.
        assert!(dot(cam.right, cam.down).abs() < 1e-12);
        assert!(dot(cam.right, cam.forward).abs() < 1e-12);
        assert!(dot(cam.down, cam.forward).abs() < 1e-12);
    }

    #[test]
    fn format_hashes_matches_voxlap_format() {
        // voxlap's fprintf format: `%s  %016llx\n`. Two-space
        // separator, 16-hex-digit zero-padded.
        let body = format_hashes(&[("north", 0x326a_7c41_c3cc_659d)]);
        assert_eq!(body, "north  326a7c41c3cc659d\n");
    }

    #[test]
    fn render_produces_stable_hash_per_pose() {
        // Smoke test: rendering the same pose twice gives the same
        // hash (no nondeterminism).
        let engine = Engine::new();
        let vxl_world = load_oracle_vxl();
        let pixel_count = (XRES as usize) * (YRES as usize);
        let mut fb = vec![0u32; pixel_count];
        let mut zb = vec![0.0f32; pixel_count];
        let mut scratch = ScanScratch::new_for_size(XRES, YRES, vxl_world.vsid);
        let h1 = render_pose(
            &engine,
            &vxl_world,
            &POSES[0],
            &mut fb,
            &mut zb,
            &mut scratch,
        );
        let h2 = render_pose(
            &engine,
            &vxl_world,
            &POSES[0],
            &mut fb,
            &mut zb,
            &mut scratch,
        );
        assert_eq!(h1, h2);
    }
}
