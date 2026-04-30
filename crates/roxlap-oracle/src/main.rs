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
//! freshly-produced `roxlap-hashes.txt` against the in-tree
//! `tests/golden-hashes.txt` (currently the 4 opticast-only
//! poses roxlap renders bit-exact). Override `--golden` to
//! compare against voxlaptest's full 12-pose
//! `tests/oracle/golden-hashes.txt` when both repos sit side by
//! side locally. The diff tool lists per-pose match / mismatch
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

/// Dump framebuffer as a P6 PPM (binary RGB). Standalone format
/// — no external image library — that GIMP / feh / `xdg-open`
/// open natively. Voxlap's framebuffer is `int32_t` packed as
/// `(brightness << 24) | (R << 16) | (G << 8) | B`; we extract
/// the RGB bytes and discard the brightness/alpha.
///
/// Useful for side-by-side visual diff against
/// `voxlaptest/build-clang/oracle-run/<pose>.png`. Gated behind
/// the `ROXLAP_ORACLE_PPM` env var so the default render run
/// stays disk-quiet.
fn write_ppm(path: &str, framebuffer: &[u32], width: u32, height: u32) -> std::io::Result<()> {
    let header = format!("P6\n{width} {height}\n255\n");
    let mut bytes = Vec::with_capacity(header.len() + framebuffer.len() * 3);
    bytes.extend_from_slice(header.as_bytes());
    for &px in framebuffer {
        bytes.push(((px >> 16) & 0xff) as u8); // R
        bytes.push(((px >> 8) & 0xff) as u8); // G
        bytes.push((px & 0xff) as u8); // B
    }
    fs::write(path, bytes)?;
    Ok(())
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
    let mut engine = Engine::new();
    // Mirror voxlap C oracle.c:117 + :110 — `set_fogcol(0x87ceeb)` +
    // `setMaxScanDist(1024)`. Without this the fog falloff table
    // (`scratch.foglut`) stays empty and `fog_blend` short-circuits
    // to a no-op, leaving the floor flat instead of gradient-shaded
    // toward sky as distance grows.
    engine.set_fog(0x0087_ceeb, 1024);
    let engine = engine; // re-freeze after one-shot setup
    let vxl_world = load_oracle_vxl();

    let pixel_count = (XRES as usize) * (YRES as usize);
    let mut framebuffer = vec![0u32; pixel_count];
    let mut zbuffer = vec![0.0f32; pixel_count];
    let mut scratch = ScanScratch::new_for_size(XRES, YRES, vxl_world.vsid);

    let mut rows: Vec<(&str, u64)> = Vec::with_capacity(POSES.len());
    let dump_ppm = std::env::var("ROXLAP_ORACLE_PPM").is_ok();
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
        if dump_ppm {
            write_ppm(&format!("{}.ppm", pose.name), &framebuffer, XRES, YRES)?;
        }
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

/// `debug-gline <pose>` — print the gline frustum values for a
/// canonical center bottom-quadrant scanline. Lets the next debug
/// session diff these against voxlap C's `gline` debug prints
/// for the same input. Pin the divergence cause for the
/// `north`/`east` identical-hash bug without trial-and-erroring
/// blind.
#[allow(clippy::unnecessary_wraps)]
fn cmd_debug_gline(pose_name: &str) -> std::io::Result<()> {
    use roxlap_core::camera_math;
    use roxlap_core::gline::derive_gline_frustum;
    use roxlap_core::opticast_prelude;

    let Some(pose) = POSES.iter().find(|p| p.name == pose_name) else {
        let _ = writeln!(std::io::stderr(), "unknown pose: {pose_name}");
        let names: Vec<&str> = POSES.iter().map(|p| p.name).collect();
        let _ = writeln!(std::io::stderr(), "known: {}", names.join(", "));
        std::process::exit(2);
    };
    let cam = camera_for_pose(pose);
    println!("=== pose: {} ===", pose.name);
    println!(
        "camera.pos     = [{}, {}, {}]",
        cam.pos[0], cam.pos[1], cam.pos[2]
    );
    println!(
        "camera.right   = [{}, {}, {}]",
        cam.right[0], cam.right[1], cam.right[2]
    );
    println!(
        "camera.down    = [{}, {}, {}]",
        cam.down[0], cam.down[1], cam.down[2]
    );
    println!(
        "camera.forward = [{}, {}, {}]",
        cam.forward[0], cam.forward[1], cam.forward[2]
    );

    let cs = camera_math::derive(&cam, XRES, YRES, 320.0, 240.0, 320.0);
    println!("\nCameraState:");
    println!("  right  = {:?}", cs.right);
    println!("  down   = {:?}", cs.down);
    println!("  fwd    = {:?}", cs.forward);
    println!("  add    = {:?}", cs.add);
    println!("  corn[0]= {:?}", cs.corn[0]);
    println!("  corn[1]= {:?}", cs.corn[1]);
    println!("  corn[2]= {:?}", cs.corn[2]);
    println!("  corn[3]= {:?}", cs.corn[3]);

    let prelude = opticast_prelude::derive_prelude(&cs, 2048, 1, 4, 1024);
    println!("\nOpticastPrelude:");
    println!("  forward_z_sign = {}", prelude.forward_z_sign);
    println!("  li_pos         = {:?}", prelude.li_pos);
    println!("  column_index   = {}", prelude.column_index);
    println!("  pos_xfrac      = {:?}", prelude.pos_xfrac);
    println!("  pos_yfrac      = {:?}", prelude.pos_yfrac);
    println!(
        "  pos_z          = {} (= 0x{:08x})",
        prelude.pos_z, prelude.pos_z
    );
    println!(
        "  x_mip          = {} (= 0x{:08x})",
        prelude.x_mip, prelude.x_mip
    );
    println!(
        "  max_scan_dist  = {} (= 0x{:08x})",
        prelude.max_scan_dist, prelude.max_scan_dist
    );

    // Canonical center bottom-quadrant ray from voxlap's scan
    // loop: x0 = cx (= 320), y0 = cy (= 240), x1 = cx, y1 = wy1
    // (= 480). For voxlap C debug-print parity, instrument
    // voxlap5.c:gline at line ~1146 with the same inputs and dump
    // the same fields.
    let leng = 240; // |y1 - y0| = 480 - 240
    let f = derive_gline_frustum(&cs, &prelude, 2048, leng, 320.0, 240.0, 320.0, 480.0);
    println!("\ngline frustum (x0=320, y0=240, x1=320, y1=480, leng={leng}):");
    println!("  vd0 = {} (post-rescale)", f.vd0);
    println!("  vd1 = {} (= f = sqrt(vx1²+vy1²))", f.vd1);
    println!("  vz0 = {}", f.vz0);
    println!("  vx1 = {}", f.vx1);
    println!("  vy1 = {}", f.vy1);
    println!("  vz1 = {}", f.vz1);
    println!("  gixy = {:?}", f.gixy);
    println!("  gpz  = {:?}", f.gpz);
    println!("  gdz  = {:?}", f.gdz);
    println!(
        "\nKey check: vd0 == vd1? {} (vd0-vd1 = {})",
        f.vd0.to_bits() == f.vd1.to_bits(),
        f.vd0 - f.vd1
    );
    #[allow(clippy::cast_precision_loss)]
    let leng_f = leng as f32;
    println!(
        "  → gi0 = (vd0 - vd1) / leng = {} (zero ⇒ drawflor exits on first cross-sign test)",
        (f.vd0 - f.vd1) / leng_f
    );
    Ok(())
}

fn print_help() {
    let _ = writeln!(
        std::io::stderr(),
        "roxlap-oracle — render-hash oracle for the roxlap engine.\n\
         \n\
         Usage:\n\
             roxlap-oracle              render every pose, write roxlap-hashes.txt\n\
             roxlap-oracle render       same as above\n\
             roxlap-oracle diff [--golden PATH] [--ours PATH]\n\
                                         diff roxlap-hashes.txt against the in-tree\n\
                                         golden-hashes.txt; exits non-zero on any mismatch.\n\
                                         Defaults: --ours=roxlap-hashes.txt,\n\
                                         --golden=tests/golden-hashes.txt\n\
             roxlap-oracle debug-gline POSE\n\
                                         dump camera basis + prelude + gline frustum\n\
                                         values for one named pose's center bottom-\n\
                                         quadrant scanline. For diff against voxlap C\n\
                                         debug prints. Known poses: north, east,\n\
                                         diag_down, high_down.\n\
         \n\
         Env:\n\
             ROXLAP_ORACLE_PPM=1        also dump <pose>.ppm framebuffers (P6 RGB)\n\
                                         alongside roxlap-hashes.txt"
    );
}

/// Captured runtime state from the host's F-key capture. Exactly
/// the fields `crates/roxlap-host/src/main.rs::write_capture` writes.
struct Capture {
    width: u32,
    height: u32,
    pos: [f64; 3],
    yaw: f64,
    pitch: f64,
}

/// Parse a `key = value` capture file. Tolerates `# comment` lines.
fn parse_capture(text: &str) -> Capture {
    let mut width = 800u32;
    let mut height = 600u32;
    let mut pos = [1024.0_f64, 1024.0, 100.0];
    let mut yaw = 0.0_f64;
    let mut pitch = 0.0_f64;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((k, v)) = line.split_once('=') else {
            continue;
        };
        let k = k.trim();
        let v = v.trim();
        match k {
            "width" => {
                if let Ok(n) = v.parse() {
                    width = n;
                }
            }
            "height" => {
                if let Ok(n) = v.parse() {
                    height = n;
                }
            }
            "pos.x" => {
                if let Ok(n) = v.parse() {
                    pos[0] = n;
                }
            }
            "pos.y" => {
                if let Ok(n) = v.parse() {
                    pos[1] = n;
                }
            }
            "pos.z" => {
                if let Ok(n) = v.parse() {
                    pos[2] = n;
                }
            }
            "yaw" => {
                if let Ok(n) = v.parse() {
                    yaw = n;
                }
            }
            "pitch" => {
                if let Ok(n) = v.parse() {
                    pitch = n;
                }
            }
            _ => {}
        }
    }
    Capture {
        width,
        height,
        pos,
        yaw,
        pitch,
    }
}

/// `find-hairlines [path-to-capture.txt]` — debug aid for the
/// "vertical sky-coloured hairline on the floor" artifact. Loads
/// the camera state captured by the host's `F` hotkey (default:
/// `./roxlap-capture.txt`), re-renders into a sentinel-initialised
/// framebuffer at the captured resolution, and scans for sandwich
/// patterns where sky / unwritten pixels sit between floor pixels.
//
// Suppress dead-code lints — this is diagnostic-only, called via
// the CLI dispatch, not by the test suite.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::too_many_lines
)]
fn cmd_find_hairlines(capture_path: &str) -> std::io::Result<()> {
    let txt = fs::read_to_string(capture_path)?;
    let cap = parse_capture(&txt);
    let Capture {
        width: hx,
        height: hy,
        pos,
        yaw,
        pitch,
    } = cap;

    let mut engine = Engine::new();
    // ROXLAP_TAG_SKY: paint sky bright green so pixels that hrend
    // writes from a startsky-drained radar slot are unambiguously
    // distinguishable from sky-colour reads via fog. Off by default;
    // set the env var when isolating a hairline source.
    if std::env::var("ROXLAP_TAG_SKY").is_ok() {
        engine.set_sky_color(0x80_00_FF_00);
    }
    if std::env::var("ROXLAP_FOG").is_ok() {
        engine.set_fog(0x00_87_ce_eb, 1024);
    }
    let vxl_world = load_oracle_vxl();

    let pixel_count = (hx as usize) * (hy as usize);
    let mut framebuffer = vec![0u32; pixel_count];
    let mut zbuffer = vec![0.0f32; pixel_count];
    let mut scratch = ScanScratch::new_for_size(hx, hy, vxl_world.vsid);
    // Host yaw/pitch convention (main.rs::App::camera): yaw=0 looks
    // +y; right = [cos, -sin, 0]; forward = [sin*cos, cos*cos, sin].
    let cyaw = yaw.cos();
    let syaw = yaw.sin();
    let cp = pitch.cos();
    let sp = pitch.sin();
    let host_cam = Camera {
        pos,
        right: [cyaw, -syaw, 0.0],
        down: [-syaw * sp, -cyaw * sp, cp],
        forward: [syaw * cp, cyaw * cp, sp],
    };

    // Pre-fill with a SENTINEL value (≠ sky and ≠ any voxel colour)
    // so we can distinguish (a) "pixel never written by opticast"
    // [stays sentinel] from (b) "pixel written but with sky colour"
    // [becomes sky], which is the actual hairline symptom.
    let sentinel: u32 = 0x0000_0001;
    framebuffer.fill(sentinel);
    let sky_col_i = i32::from_ne_bytes(engine.sky_color().to_ne_bytes());
    scratch.set_skycast(sky_col_i, 0);
    let fog_col_i = i32::from_ne_bytes(engine.fog_color().to_ne_bytes());
    scratch.set_fog(fog_col_i, engine.fog_max_scan_dist());

    let settings = OpticastSettings::for_oracle_framebuffer(hx, hy);
    {
        let mut rasterizer = ScalarRasterizer::new(
            &mut framebuffer,
            &mut zbuffer,
            hx as usize,
            &vxl_world.data,
            &vxl_world.column_offset,
            vxl_world.vsid,
        );
        let _ = opticast(
            &mut rasterizer,
            &mut scratch,
            &host_cam,
            &settings,
            vxl_world.vsid,
            &vxl_world.data,
            &vxl_world.column_offset,
        );
    }

    // sky here = the sentinel green we set above; it's what
    // startsky drains into radar slots.
    let sky = engine.sky_color();
    // True hairline = a run of sky/sentinel pixels that has a
    // FLOOR-coloured pixel both ABOVE and BELOW it in the same
    // column. That excludes the actual horizon-meets-sky region
    // and edge artifacts (where sky reaches the screen border).
    let is_floor = |p: u32| p != sky && p != sentinel;
    let mut hairline_columns = Vec::<(u32, u32, u32, u32)>::new(); // (sx, sy_top, run_len, is_sky)
    for sx in 0..hx {
        // Walk top-down looking for [floor, sky-run, floor] sandwiches.
        let col_idx = |sy: u32| (sy as usize) * (hx as usize) + (sx as usize);
        let mut sy = 0u32;
        while sy < hy {
            // Skip until we find a floor pixel — required for the
            // top side of the sandwich.
            while sy < hy && !is_floor(framebuffer[col_idx(sy)]) {
                sy += 1;
            }
            if sy >= hy {
                break;
            }
            // Skip floor pixels.
            while sy < hy && is_floor(framebuffer[col_idx(sy)]) {
                sy += 1;
            }
            if sy >= hy {
                break;
            }
            // Sky/sentinel run; record its top + length.
            let top = sy;
            let mut run_is_sky = 0u32;
            while sy < hy && !is_floor(framebuffer[col_idx(sy)]) {
                if framebuffer[col_idx(sy)] == sky {
                    run_is_sky = 1;
                }
                sy += 1;
            }
            let run = sy - top;
            // Sandwich check: must be followed by another floor pixel
            // (sy < hy here means a floor pixel exists below).
            if sy < hy && run >= 2 {
                hairline_columns.push((sx, top, run, run_is_sky));
            }
        }
    }

    println!(
        "pose: pos=({}, {}, {}) yaw={yaw:.4} pitch={pitch:.4} size={hx}x{hy}\n\
         {} sandwich-shaped hairlines (sky/sentinel between floor pixels, run ≥ 2)",
        pos[0],
        pos[1],
        pos[2],
        hairline_columns.len(),
    );
    for (sx, sy_top, run, is_sky) in hairline_columns.iter().take(40) {
        let kind = if *is_sky == 1 { "sky" } else { "unwritten" };
        println!("  sx={sx} sy_top={sy_top} run={run} ({kind})");
    }

    // Highlight the offending pixels in the PPM dump.
    let mut viz = framebuffer.clone();
    for px in &mut viz {
        if *px == sentinel {
            *px = 0xFFFF_00FF; // magenta = unwritten
        }
    }
    write_ppm("find-hairlines.ppm", &viz, hx, hy)?;
    println!("\nwrote find-hairlines.ppm (unwritten → magenta; sky stays sky)");
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map_or("render", String::as_str);
    match cmd {
        "render" => cmd_render(),
        "diff" => {
            let mut ours = "roxlap-hashes.txt".to_string();
            // Default golden lives in roxlap (`tests/golden-hashes.txt`) so
            // CI is self-contained. Override with `--golden` to compare
            // against voxlaptest's full 12-pose `golden-hashes.txt` when
            // those repos sit side by side locally.
            let mut golden = "tests/golden-hashes.txt".to_string();
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
        "debug-gline" => {
            let pose_name = args.get(1).cloned().unwrap_or_else(|| "north".to_string());
            cmd_debug_gline(&pose_name)
        }
        "find-hairlines" => {
            // Default: ./roxlap-capture.txt — what the host writes
            // when the user presses F. CLI override lets you point
            // at a different captured pose.
            let path = args
                .get(1)
                .cloned()
                .unwrap_or_else(|| "roxlap-capture.txt".to_string());
            cmd_find_hairlines(&path)
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
