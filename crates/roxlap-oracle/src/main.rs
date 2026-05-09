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
use std::io::Write;

use roxlap_core::camera_math;
use roxlap_core::opticast;
use roxlap_core::rasterizer::ScratchPool;
use roxlap_core::scalar_rasterizer::ScalarRasterizer;
use roxlap_core::sprite::{draw_sprite, draw_sprites_parallel, DrawTarget, SpriteLighting};
use roxlap_core::Camera;
use roxlap_core::Engine;
use roxlap_core::OpticastSettings;
use roxlap_formats::kv6;
use roxlap_formats::sprite::Sprite;

// R10.4 refactor — render-pose helpers (POSES table, fnv1a64,
// camera_for_pose, render_pose, lighting bake, etc.) live in
// `lib.rs` so the wasm-bindgen-test goldens suite shares the
// same render path.
use roxlap_oracle::{
    camera_for_pose, load_oracle_vxl, render_pose, Pose, POSES, SPRITE_MELTSPHERE_KV6, XRES, YRES,
};

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

/// `cmd_render` options parsed off the CLI. `ppm_dir = Some(dir)`
/// means "dump `<pose>.ppm` into `dir`" (creating it if missing).
/// `threads >= 4` switches the rasterizer onto opticast's parallel
/// branch (R12.2.1); `1` (default) is the single-threaded path the
/// goldens are frozen against.
struct RenderOpts {
    ppm_dir: Option<std::path::PathBuf>,
    threads: usize,
}

/// Default output directory for `--ppm` without an explicit path.
/// Lives next to `roxlap-hashes.txt`; one PPM per pose dropped in.
const DEFAULT_PPM_DIR: &str = "roxlap-oracle-out";

/// `cmd_bench` options parsed off the CLI. `pose_name = None`
/// benchmarks every pose; `Some(name)` runs that one pose only.
/// `threads >= 4` opts into opticast's parallel branch (R12.2.1).
struct BenchOpts {
    pose_name: Option<String>,
    iters: usize,
    warmup: usize,
    threads: usize,
}

/// Microbenchmark the render pipeline over the oracle poses.
/// Reports per-pose min / p50 / mean / p99 / max ms and a derived
/// FPS, plus a grand total when running the full set. Uses the same
/// `render_pose` path the oracle and CI exercise — sticky lighting
/// bake on the first `pose.lit` entry, fresh framebuffer + zbuffer
/// per run.
///
/// Defaults: `iters = 30`, `warmup = 5`. The warmup smooths over
/// CPU frequency ramp / cache-warming on the first few iterations;
/// the measure phase records per-iter `Instant::elapsed`.
#[allow(clippy::too_many_lines, clippy::unnecessary_wraps)]
fn cmd_bench(opts: &BenchOpts) -> std::io::Result<()> {
    let mut engine = Engine::new();
    engine.set_fog(0x0087_ceeb, 1024);
    let mut vxl_world = load_oracle_vxl();

    let pixel_count = (XRES as usize) * (YRES as usize);
    let mut framebuffer = vec![0u32; pixel_count];
    let mut zbuffer = vec![0.0f32; pixel_count];
    let mut pool = ScratchPool::new_parallel(XRES, YRES, vxl_world.vsid, opts.threads);

    let target_poses: Vec<&Pose> = match opts.pose_name.as_deref() {
        None | Some("all") => POSES.iter().collect(),
        Some(name) => {
            if let Some(p) = POSES.iter().find(|p| p.name == name) {
                vec![p]
            } else {
                let _ = writeln!(std::io::stderr(), "unknown pose: {name}");
                let names: Vec<&str> = POSES.iter().map(|p| p.name).collect();
                let _ = writeln!(
                    std::io::stderr(),
                    "available: {} (or `all`)",
                    names.join(", ")
                );
                std::process::exit(2);
            }
        }
    };

    println!(
        "Bench: {} iter{} per pose, {} warmup, {}×{} framebuffer, {} thread{}",
        opts.iters,
        if opts.iters == 1 { "" } else { "s" },
        opts.warmup,
        XRES,
        YRES,
        opts.threads,
        if opts.threads == 1 { "" } else { "s" },
    );
    println!();
    println!(
        "{:<14}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
        "pose", "min ms", "p50 ms", "mean ms", "p99 ms", "max ms", "fps"
    );

    let mut lighting_baked = false;
    let mut grand_total = std::time::Duration::ZERO;
    let mut grand_iters = 0usize;

    for pose in target_poses {
        if pose.lit && !lighting_baked {
            engine.set_lightmode(2);
            engine.add_light(roxlap_core::LightSrc {
                pos: [1100.0, 1100.0, 70.0],
                r2: 600.0 * 600.0,
                sc: 1.0,
            });
            roxlap_core::update_lighting(
                &mut vxl_world.data,
                &vxl_world.column_offset,
                vxl_world.vsid,
                800,
                800,
                0,
                1248,
                1248,
                200,
                engine.lightmode(),
                engine.lights(),
            );
            lighting_baked = true;
        }

        for _ in 0..opts.warmup {
            let _ = render_pose(
                &engine,
                &vxl_world,
                pose,
                &mut framebuffer,
                &mut zbuffer,
                &mut pool,
            );
        }

        let mut samples: Vec<std::time::Duration> = Vec::with_capacity(opts.iters);
        for _ in 0..opts.iters {
            let t0 = std::time::Instant::now();
            let _ = render_pose(
                &engine,
                &vxl_world,
                pose,
                &mut framebuffer,
                &mut zbuffer,
                &mut pool,
            );
            samples.push(t0.elapsed());
        }

        samples.sort_unstable();
        let min = samples[0];
        let max = samples[samples.len() - 1];
        let p50 = samples[samples.len() / 2];
        // p99 with len < 100 falls back to max — that's fine; the
        // signal is that large-N runs see real percentile data.
        let p99_idx = samples
            .len()
            .saturating_sub(1)
            .min(samples.len() * 99 / 100);
        let p99 = samples[p99_idx];
        let total: std::time::Duration = samples.iter().sum();
        // `as u32` cast: opts.iters is u32-bounded by the CLI parser.
        #[allow(clippy::cast_possible_truncation)]
        let mean = total / opts.iters as u32;
        let fps = 1.0 / mean.as_secs_f64();

        println!(
            "{:<14}  {:>8.3}  {:>8.3}  {:>8.3}  {:>8.3}  {:>8.3}  {:>8.1}",
            pose.name,
            min.as_secs_f64() * 1000.0,
            p50.as_secs_f64() * 1000.0,
            mean.as_secs_f64() * 1000.0,
            p99.as_secs_f64() * 1000.0,
            max.as_secs_f64() * 1000.0,
            fps
        );

        grand_total += total;
        grand_iters += opts.iters;
    }

    if grand_iters > 1 {
        #[allow(clippy::cast_possible_truncation)]
        let grand_mean = grand_total / grand_iters as u32;
        println!();
        println!(
            "Total: {:.3} ms across {} renders, mean {:.3} ms/frame, {:.1} fps",
            grand_total.as_secs_f64() * 1000.0,
            grand_iters,
            grand_mean.as_secs_f64() * 1000.0,
            1.0 / grand_mean.as_secs_f64()
        );
    }

    Ok(())
}

/// Microbenchmark `update_lighting` over the oracle bake region —
/// the same `(800..1248) × (800..1248) × (0..200)` box `cmd_bench`
/// uses for the lit poses. Reports min / p50 / mean / p99 / max
/// across `iters` repetitions.
///
/// Defaults: `iters = 20`, `warmup = 3`. Rebakes are idempotent
/// (same lights → same alpha bytes), so repeated calls are safe.
/// Speedup follows `RAYON_NUM_THREADS` — set `=1` to measure the
/// sequential baseline, leave unset to use rayon's default pool
/// (= `num_cpus`). The R12.4.1 parallel path is always engaged;
/// there is no `--threads` flag because `update_lighting`'s pool
/// is rayon's global pool, not [`ScratchPool`].
#[allow(
    clippy::unnecessary_wraps,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn cmd_bench_lighting(iters: usize, warmup: usize) -> std::io::Result<()> {
    let mut engine = Engine::new();
    let mut vxl_world = load_oracle_vxl();

    engine.set_lightmode(2);
    engine.add_light(roxlap_core::LightSrc {
        pos: [1100.0, 1100.0, 70.0],
        r2: 600.0 * 600.0,
        sc: 1.0,
    });

    let bake_region = ((800, 800, 0), (1248, 1248, 200));
    println!(
        "bench-lighting: {} iter{} ({} warmup), bake region ({:?}..{:?}), \
         RAYON_NUM_THREADS={}",
        iters,
        if iters == 1 { "" } else { "s" },
        warmup,
        bake_region.0,
        bake_region.1,
        std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "<default>".to_string()),
    );

    for _ in 0..warmup {
        roxlap_core::update_lighting(
            &mut vxl_world.data,
            &vxl_world.column_offset,
            vxl_world.vsid,
            bake_region.0 .0,
            bake_region.0 .1,
            bake_region.0 .2,
            bake_region.1 .0,
            bake_region.1 .1,
            bake_region.1 .2,
            engine.lightmode(),
            engine.lights(),
        );
    }

    let mut samples: Vec<std::time::Duration> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = std::time::Instant::now();
        roxlap_core::update_lighting(
            &mut vxl_world.data,
            &vxl_world.column_offset,
            vxl_world.vsid,
            bake_region.0 .0,
            bake_region.0 .1,
            bake_region.0 .2,
            bake_region.1 .0,
            bake_region.1 .1,
            bake_region.1 .2,
            engine.lightmode(),
            engine.lights(),
        );
        samples.push(t0.elapsed());
    }
    samples.sort_unstable();

    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    let mean = (samples
        .iter()
        .map(std::time::Duration::as_secs_f64)
        .sum::<f64>()
        / samples.len() as f64)
        * 1000.0;
    let min = samples[0].as_secs_f64() * 1000.0;
    let p50 = samples[samples.len() / 2].as_secs_f64() * 1000.0;
    let p99_idx = ((samples.len() as f64) * 0.99) as usize;
    let p99 = samples[p99_idx.min(samples.len() - 1)].as_secs_f64() * 1000.0;
    let max = samples[samples.len() - 1].as_secs_f64() * 1000.0;

    println!();
    println!(
        "{:<10}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
        "metric", "min ms", "p50 ms", "mean ms", "p99 ms", "max ms"
    );
    println!(
        "{:<10}  {:>8.3}  {:>8.3}  {:>8.3}  {:>8.3}  {:>8.3}",
        "bake", min, p50, mean, p99, max
    );

    Ok(())
}

/// Microbenchmark `draw_sprites_parallel` over a synthetic
/// `n_sprites`-sprite scene. The sprites are meltsphere kv6
/// instances scattered on a grid around the camera so each one
/// projects to a distinct screen rect (no z-test contention,
/// reproducible byte output regardless of thread scheduling).
///
/// Compares:
///  - **sequential**: a manual `for sprite in &sprites { draw_sprite(...) }` loop.
///  - **parallel**:   `draw_sprites_parallel(&sprites)` (rayon `par_iter`).
///
/// Reports min / p50 / mean / p99 / max for each variant. Speedup
/// follows `RAYON_NUM_THREADS`; set `=1` for the sequential
/// baseline of the parallel call (which lets us isolate rayon
/// overhead from the actual parallel speedup).
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::unnecessary_wraps,
    clippy::too_many_lines
)]
fn cmd_bench_sprites(n_sprites: u32, iters: usize, warmup: usize) -> std::io::Result<()> {
    let mut engine = Engine::new();
    let vxl_world = load_oracle_vxl();
    engine.set_fog(0x0087_ceeb, 1024);

    let pixel_count = (XRES as usize) * (YRES as usize);
    let mut framebuffer = vec![0u32; pixel_count];
    let mut zbuffer = vec![0.0f32; pixel_count];
    let mut pool = ScratchPool::new(XRES, YRES, vxl_world.vsid);

    // Build the synthetic scene: one camera looking +y, sprites
    // scattered on a grid of meltsphere instances at z=170 (just
    // above the playable plane).
    // Right-handed basis: right × down == forward. Looking +y with
    // world +z down, right is -x. (LH basis triggers silent sprite
    // cull in kv6_draw_prepare — see feedback_voxlap_basis_chirality.)
    let cam = Camera {
        pos: [1024.0, 980.0, 175.0],
        right: [-1.0, 0.0, 0.0],
        down: [0.0, 0.0, 1.0],
        forward: [0.0, 1.0, 0.0],
    };
    let cam_state = camera_math::derive(
        &cam,
        XRES,
        YRES,
        XRES as f32 / 2.0,
        YRES as f32 / 2.0,
        XRES as f32 / 2.0,
    );
    let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
    let lighting = SpriteLighting::default_oracle();

    let kv = kv6::parse(SPRITE_MELTSPHERE_KV6).expect("parse meltsphere fixture");
    let n_grid = (n_sprites as f64).sqrt().ceil() as u32;
    let cell = 12.0_f32;
    let origin_x = cam.pos[0] as f32 - (n_grid as f32 * cell) / 2.0;
    let origin_y = cam.pos[1] as f32 + 30.0;
    let mut sprites: Vec<Sprite> = Vec::with_capacity(n_sprites as usize);
    for i in 0..n_sprites {
        let gx = i % n_grid;
        let gy = i / n_grid;
        let p = [
            origin_x + (gx as f32) * cell,
            origin_y + (gy as f32) * cell,
            175.0,
        ];
        sprites.push(Sprite::axis_aligned(kv.clone(), p));
    }

    println!(
        "bench-sprites: {n_sprites} sprites, {iters} iter{} ({warmup} warmup), \
         RAYON_NUM_THREADS={}",
        if iters == 1 { "" } else { "s" },
        std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "<default>".to_string()),
    );

    // First, render world once so the framebuffer / zbuffer are
    // populated to a representative state (sprite z-test reads
    // existing zbuffer values).
    let pitch_pixels = XRES as usize;
    let sky = engine.sky_color();
    framebuffer.fill(sky);
    zbuffer.fill(f32::INFINITY);
    {
        let mut rasterizer = ScalarRasterizer::new(
            &mut framebuffer,
            &mut zbuffer,
            pitch_pixels,
            &vxl_world.data,
            &vxl_world.column_offset,
            &vxl_world.mip_base_offsets,
            vxl_world.vsid,
        );
        let _ = opticast(
            &mut rasterizer,
            &mut pool,
            &cam,
            &settings,
            vxl_world.vsid,
            &vxl_world.data,
            &vxl_world.column_offset,
        );
    }
    // Snapshot the post-world fb / zb so each sprite-bench run
    // starts from the same state.
    let fb_snapshot = framebuffer.clone();
    let zb_snapshot = zbuffer.clone();

    let mut bench_loop = |label: &str, run: &dyn Fn(&mut [u32], &mut [f32])| {
        for _ in 0..warmup {
            framebuffer.copy_from_slice(&fb_snapshot);
            zbuffer.copy_from_slice(&zb_snapshot);
            run(&mut framebuffer, &mut zbuffer);
        }
        let mut samples: Vec<std::time::Duration> = Vec::with_capacity(iters);
        for _ in 0..iters {
            framebuffer.copy_from_slice(&fb_snapshot);
            zbuffer.copy_from_slice(&zb_snapshot);
            let t0 = std::time::Instant::now();
            run(&mut framebuffer, &mut zbuffer);
            samples.push(t0.elapsed());
        }
        samples.sort_unstable();
        let mean_ms = (samples
            .iter()
            .map(std::time::Duration::as_secs_f64)
            .sum::<f64>()
            / samples.len() as f64)
            * 1000.0;
        let min = samples[0].as_secs_f64() * 1000.0;
        let p50 = samples[samples.len() / 2].as_secs_f64() * 1000.0;
        let p99_idx = ((samples.len() as f64) * 0.99) as usize;
        let p99 = samples[p99_idx.min(samples.len() - 1)].as_secs_f64() * 1000.0;
        let max = samples[samples.len() - 1].as_secs_f64() * 1000.0;
        println!("{label:<14}  {min:>8.3}  {p50:>8.3}  {mean_ms:>8.3}  {p99:>8.3}  {max:>8.3}");
    };

    println!();
    println!(
        "{:<14}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
        "variant", "min ms", "p50 ms", "mean ms", "p99 ms", "max ms"
    );
    bench_loop("sequential", &|fb, zb| {
        let mut target = DrawTarget::new(fb, zb, pitch_pixels, XRES, YRES);
        for sprite in &sprites {
            let _ = draw_sprite(&mut target, &cam_state, &settings, &lighting, sprite);
        }
    });
    bench_loop("parallel", &|fb, zb| {
        let target = DrawTarget::new(fb, zb, pitch_pixels, XRES, YRES);
        let _ = draw_sprites_parallel(target, &cam_state, &settings, &lighting, &sprites);
    });

    Ok(())
}

/// Render every pose and write `roxlap-hashes.txt` next to the
/// invocation cwd. With `--ppm[=DIR]` (or `ROXLAP_ORACLE_PPM=1`),
/// also dumps a P6 PPM per pose into `DIR` (default
/// `roxlap-oracle-out/`) for visual inspection of the rendered
/// framebuffer — useful for eyeballing whether a hash mismatch is
/// a real visual difference or sub-pixel drift.
fn cmd_render(opts: &RenderOpts) -> std::io::Result<()> {
    let mut engine = Engine::new();
    // Mirror voxlap C oracle.c:117 + :110 — `set_fogcol(0x87ceeb)` +
    // `setMaxScanDist(1024)`. Without this the fog falloff table
    // (`scratch.foglut`) stays empty and `fog_blend` short-circuits
    // to a no-op, leaving the floor flat instead of gradient-shaded
    // toward sky as distance grows.
    engine.set_fog(0x0087_ceeb, 1024);
    let mut vxl_world = load_oracle_vxl();

    let pixel_count = (XRES as usize) * (YRES as usize);
    let mut framebuffer = vec![0u32; pixel_count];
    let mut zbuffer = vec![0.0f32; pixel_count];
    let mut pool = ScratchPool::new_parallel(XRES, YRES, vxl_world.vsid, opts.threads);

    if let Some(dir) = &opts.ppm_dir {
        fs::create_dir_all(dir)?;
    }

    // Mirror voxlap C oracle's `lighting_baked` flag: the bake is
    // sticky — once applied, every later pose sees the lit world.
    let mut lighting_baked = false;
    let mut rows: Vec<(&str, u64)> = Vec::with_capacity(POSES.len());
    for pose in POSES {
        if pose.lit && !lighting_baked {
            // Match oracle.c:438-443: setLightingMode(2) +
            // add_light(1100, 1100, 70, 600, 1.0) +
            // updatelighting(800, 800, 0, 1248, 1248, 200).
            engine.set_lightmode(2);
            engine.add_light(roxlap_core::LightSrc {
                pos: [1100.0, 1100.0, 70.0],
                r2: 600.0 * 600.0,
                sc: 1.0,
            });
            roxlap_core::update_lighting(
                &mut vxl_world.data,
                &vxl_world.column_offset,
                vxl_world.vsid,
                800,
                800,
                0,
                1248,
                1248,
                200,
                engine.lightmode(),
                engine.lights(),
            );
            lighting_baked = true;
        }
        let hash = render_pose(
            &engine,
            &vxl_world,
            pose,
            &mut framebuffer,
            &mut zbuffer,
            &mut pool,
        );
        println!("{:<14}  {:016x}", pose.name, hash);
        if let Some(dir) = &opts.ppm_dir {
            let path = dir.join(format!("{}.ppm", pose.name));
            write_ppm(
                path.to_str().unwrap_or("<bad utf-8>"),
                &framebuffer,
                XRES,
                YRES,
            )?;
            println!("  wrote {}", path.display());
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
    let f = derive_gline_frustum(
        &cs,
        prelude.pos_xfrac,
        prelude.pos_yfrac,
        2048,
        leng,
        320.0,
        240.0,
        320.0,
        480.0,
    );
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
             roxlap-oracle [--ppm[=DIR]]\n\
                                         render every pose, write roxlap-hashes.txt;\n\
                                         with --ppm also drop a P6 PPM per pose into\n\
                                         DIR (default: roxlap-oracle-out/) for visual\n\
                                         inspection of hash-mismatched poses.\n\
             roxlap-oracle render [--ppm[=DIR]] [--threads N]\n\
                                         same as above. --threads >= 4 opts\n\
                                         into the per-quadrant parallel render\n\
                                         (R12.2.1); default 1 keeps the single-\n\
                                         threaded path the goldens are frozen\n\
                                         against.\n\
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
             roxlap-oracle bench [--pose NAME|all] [--iters N] [--warmup N] [--threads N]\n\
                                         microbenchmark render_pose; reports per-pose\n\
                                         min/p50/mean/p99/max ms + FPS, plus a grand\n\
                                         total when running `all`. Defaults:\n\
                                         --pose all, --iters 30, --warmup 5,\n\
                                         --threads 1. Pass --threads 4 to drive\n\
                                         opticast's parallel branch.\n\
             roxlap-oracle bench-lighting [--iters N] [--warmup N]\n\
                                         microbenchmark update_lighting on the\n\
                                         oracle bake region (448 × 448 × 200).\n\
                                         Speedup follows RAYON_NUM_THREADS\n\
                                         (set =1 to measure sequential).\n\
             roxlap-oracle bench-sprites [--sprites N] [--iters N] [--warmup N]\n\
                                         microbenchmark sequential vs parallel\n\
                                         sprite drawing on a synthetic grid scene.\n\
                                         Defaults: --sprites 64, --iters 50,\n\
                                         --warmup 5. Speedup follows\n\
                                         RAYON_NUM_THREADS.\n\
         \n\
         Env:\n\
             ROXLAP_ORACLE_PPM=1        legacy alias for --ppm (dumps to cwd)\n\
             ROXLAP_ORACLE_PPM=DIR      legacy alias for --ppm=DIR"
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
    // Mirror roxlap-host's shading so find-hairlines's rendered
    // output matches what the user sees in the interactive view.
    // Override with `ROXLAP_NO_SHADES=1` to revert to the oracle's
    // (0,…,0) baseline.
    if std::env::var("ROXLAP_NO_SHADES").is_err() {
        engine.set_side_shades(15, 15, 15, 15, 15, 15);
    }
    let vxl_world = load_oracle_vxl();

    let pixel_count = (hx as usize) * (hy as usize);
    let mut framebuffer = vec![0u32; pixel_count];
    let mut zbuffer = vec![0.0f32; pixel_count];
    let mut pool = ScratchPool::new(hx, hy, vxl_world.vsid);
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
    pool.set_skycast(sky_col_i, 0);
    let fog_col_i = i32::from_ne_bytes(engine.fog_color().to_ne_bytes());
    pool.set_fog(fog_col_i, engine.fog_max_scan_dist());
    let s = engine.side_shades();
    pool.set_side_shades(s[0], s[1], s[2], s[3], s[4], s[5]);

    let settings = OpticastSettings::for_oracle_framebuffer(hx, hy);
    {
        let mut rasterizer = ScalarRasterizer::new(
            &mut framebuffer,
            &mut zbuffer,
            hx as usize,
            &vxl_world.data,
            &vxl_world.column_offset,
            &vxl_world.mip_base_offsets,
            vxl_world.vsid,
        );
        let _ = opticast(
            &mut rasterizer,
            &mut pool,
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
/// Parse a `render` subcommand's flags. Recognises
/// `--ppm[=DIR]` / `--ppm DIR` and the `ROXLAP_ORACLE_PPM` env var
/// (legacy path: `=1` → cwd, `=DIR` → that dir).
fn parse_render_opts(args: &[String]) -> std::io::Result<RenderOpts> {
    let mut ppm_dir: Option<std::path::PathBuf> = None;
    let mut threads: usize = 1;

    // Env-var fallback (back-compat with the original gating).
    if let Ok(v) = std::env::var("ROXLAP_ORACLE_PPM") {
        ppm_dir = Some(if v == "1" || v.is_empty() {
            std::path::PathBuf::from(".")
        } else {
            std::path::PathBuf::from(v)
        });
    }

    // CLI flags override env var.
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        if a == "--ppm" {
            // Look at the next arg: if it's another flag or end, default-dir.
            let next_is_path = args.get(i + 1).is_some_and(|n| !n.starts_with("--"));
            if next_is_path {
                ppm_dir = Some(std::path::PathBuf::from(&args[i + 1]));
                i += 2;
            } else {
                ppm_dir = Some(std::path::PathBuf::from(DEFAULT_PPM_DIR));
                i += 1;
            }
        } else if let Some(rest) = a.strip_prefix("--ppm=") {
            ppm_dir = Some(std::path::PathBuf::from(rest));
            i += 1;
        } else if a == "--threads" {
            let raw = args.get(i + 1).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "--threads expects an integer",
                )
            })?;
            threads = raw.parse().map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("--threads: not an integer ({raw})"),
                )
            })?;
            i += 2;
        } else if let Some(rest) = a.strip_prefix("--threads=") {
            threads = rest.parse().map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("--threads: not an integer ({rest})"),
                )
            })?;
            i += 1;
        } else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("unknown render arg: {a}"),
            ));
        }
    }
    Ok(RenderOpts { ppm_dir, threads })
}

#[allow(clippy::too_many_lines)]
fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    // Default subcommand is `render`. Treating a leading `--`-flag
    // as "this is render with flags" lets `roxlap-oracle --ppm`
    // work without spelling out `render`.
    let cmd = args
        .first()
        .filter(|s| !s.starts_with("--"))
        .map_or("render", String::as_str);
    match cmd {
        "render" => {
            // Skip the subcommand if present (`render --ppm`); the no-arg
            // form (`roxlap-oracle --ppm`) treats the first arg as a flag.
            let flag_start = usize::from(args.first().map(String::as_str) == Some("render"));
            let opts = parse_render_opts(&args[flag_start..])?;
            cmd_render(&opts)
        }
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
        "bench-sprites" => {
            let mut n_sprites: u32 = 64;
            let mut iters: usize = 50;
            let mut warmup: usize = 5;
            let mut i = 1;
            while i < args.len() {
                match args[i].as_str() {
                    "--sprites" => {
                        let raw = args.get(i + 1).cloned().ok_or_else(|| {
                            std::io::Error::new(
                                std::io::ErrorKind::InvalidInput,
                                "--sprites expects an integer",
                            )
                        })?;
                        n_sprites = raw.parse().map_err(|_| {
                            std::io::Error::new(
                                std::io::ErrorKind::InvalidInput,
                                format!("--sprites: not an integer ({raw})"),
                            )
                        })?;
                        i += 2;
                    }
                    "--iters" => {
                        let raw = args.get(i + 1).cloned().ok_or_else(|| {
                            std::io::Error::new(
                                std::io::ErrorKind::InvalidInput,
                                "--iters expects an integer",
                            )
                        })?;
                        iters = raw.parse().map_err(|_| {
                            std::io::Error::new(
                                std::io::ErrorKind::InvalidInput,
                                format!("--iters: not an integer ({raw})"),
                            )
                        })?;
                        i += 2;
                    }
                    "--warmup" => {
                        let raw = args.get(i + 1).cloned().ok_or_else(|| {
                            std::io::Error::new(
                                std::io::ErrorKind::InvalidInput,
                                "--warmup expects an integer",
                            )
                        })?;
                        warmup = raw.parse().map_err(|_| {
                            std::io::Error::new(
                                std::io::ErrorKind::InvalidInput,
                                format!("--warmup: not an integer ({raw})"),
                            )
                        })?;
                        i += 2;
                    }
                    other => {
                        let _ = writeln!(std::io::stderr(), "unknown bench-sprites arg: {other}");
                        print_help();
                        std::process::exit(2);
                    }
                }
            }
            cmd_bench_sprites(n_sprites, iters, warmup)
        }
        "bench-lighting" => {
            let mut iters: usize = 20;
            let mut warmup: usize = 3;
            let mut i = 1;
            while i < args.len() {
                match args[i].as_str() {
                    "--iters" => {
                        let raw = args.get(i + 1).cloned().ok_or_else(|| {
                            std::io::Error::new(
                                std::io::ErrorKind::InvalidInput,
                                "--iters expects an integer",
                            )
                        })?;
                        iters = raw.parse().map_err(|_| {
                            std::io::Error::new(
                                std::io::ErrorKind::InvalidInput,
                                format!("--iters: not an integer ({raw})"),
                            )
                        })?;
                        i += 2;
                    }
                    "--warmup" => {
                        let raw = args.get(i + 1).cloned().ok_or_else(|| {
                            std::io::Error::new(
                                std::io::ErrorKind::InvalidInput,
                                "--warmup expects an integer",
                            )
                        })?;
                        warmup = raw.parse().map_err(|_| {
                            std::io::Error::new(
                                std::io::ErrorKind::InvalidInput,
                                format!("--warmup: not an integer ({raw})"),
                            )
                        })?;
                        i += 2;
                    }
                    other => {
                        let _ = writeln!(std::io::stderr(), "unknown bench-lighting arg: {other}");
                        print_help();
                        std::process::exit(2);
                    }
                }
            }
            cmd_bench_lighting(iters, warmup)
        }
        "bench" => {
            let mut opts = BenchOpts {
                pose_name: None,
                iters: 30,
                warmup: 5,
                threads: 1,
            };
            let mut i = 1;
            while i < args.len() {
                match args[i].as_str() {
                    "--pose" => {
                        opts.pose_name = Some(args.get(i + 1).cloned().ok_or_else(|| {
                            std::io::Error::new(
                                std::io::ErrorKind::InvalidInput,
                                "--pose expects a name (or `all`)",
                            )
                        })?);
                        i += 2;
                    }
                    "--iters" => {
                        let raw = args.get(i + 1).cloned().ok_or_else(|| {
                            std::io::Error::new(
                                std::io::ErrorKind::InvalidInput,
                                "--iters expects an integer",
                            )
                        })?;
                        opts.iters = raw.parse().map_err(|_| {
                            std::io::Error::new(
                                std::io::ErrorKind::InvalidInput,
                                format!("--iters: not an integer ({raw})"),
                            )
                        })?;
                        i += 2;
                    }
                    "--warmup" => {
                        let raw = args.get(i + 1).cloned().ok_or_else(|| {
                            std::io::Error::new(
                                std::io::ErrorKind::InvalidInput,
                                "--warmup expects an integer",
                            )
                        })?;
                        opts.warmup = raw.parse().map_err(|_| {
                            std::io::Error::new(
                                std::io::ErrorKind::InvalidInput,
                                format!("--warmup: not an integer ({raw})"),
                            )
                        })?;
                        i += 2;
                    }
                    "--threads" => {
                        let raw = args.get(i + 1).cloned().ok_or_else(|| {
                            std::io::Error::new(
                                std::io::ErrorKind::InvalidInput,
                                "--threads expects an integer",
                            )
                        })?;
                        opts.threads = raw.parse().map_err(|_| {
                            std::io::Error::new(
                                std::io::ErrorKind::InvalidInput,
                                format!("--threads: not an integer ({raw})"),
                            )
                        })?;
                        i += 2;
                    }
                    other => {
                        let _ = writeln!(std::io::stderr(), "unknown bench arg: {other}");
                        print_help();
                        std::process::exit(2);
                    }
                }
            }
            if opts.iters == 0 {
                let _ = writeln!(std::io::stderr(), "--iters must be >= 1");
                std::process::exit(2);
            }
            cmd_bench(&opts)
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
    // Test-only imports — the parent module's `use` list is
    // production-only after the R10.4 lib refactor.
    use roxlap_oracle::{fnv1a64, POSES};

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
        let mut pool = ScratchPool::new(XRES, YRES, vxl_world.vsid);
        let h1 = render_pose(&engine, &vxl_world, &POSES[0], &mut fb, &mut zb, &mut pool);
        let h2 = render_pose(&engine, &vxl_world, &POSES[0], &mut fb, &mut zb, &mut pool);
        assert_eq!(h1, h2);
    }
}
