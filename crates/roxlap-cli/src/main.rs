//! roxlap-cli — asset tool for the roxlap voxel engine.
//!
//! Three subcommands:
//!
//! - `info <file>` — identify + summarise any roxlap-readable asset
//!   (`.vox`, `.kv6`, `.kvx`, `.vxl`, `.kfa`, `.rvc`, `.rkc`, scene
//!   snapshots). Detection is magic-based where the format has one,
//!   extension-based for the two that don't (`.kvx`, `.vxl`).
//! - `vox2kv6 <in.vox> <out.kv6>` — MagicaVoxel → kv6 sprite model(s).
//!   A multi-model file writes `out_0.kv6`, `out_1.kv6`, ….
//! - `vox2rvc <in.vox> <out.rvc> [frame_ms]` — MagicaVoxel models as
//!   the frames of one animated voxel clip (all models must share one
//!   size; default 80 ms per frame, looping).
//! - `kv62vox <in.kv6> <out.vox>` — the reverse trip: a kv6 sprite
//!   back into a MagicaVoxel file (exact for ≤ 255 distinct colours,
//!   palette-quantised beyond; the kv6 pivot is lost — `.vox` has no
//!   pivot).
//! - `gif2rvc <in.gif> <out.rvc> [thickness]` — animated GIF →
//!   billboard voxel clip (Doom-style flat slabs; default thickness 1).
//! - `png2rvc <in.png> <out.rvc> [thickness]` — APNG (or a single
//!   still PNG) → billboard voxel clip. For a numbered frame
//!   sequence, pass the frames then the output:
//!   `png2rvc <f0.png> <f1.png> … <out.rvc>` (80 ms per frame).
//! - `chunks <snapshot> <grid>` — list a snapshot grid's resident
//!   chunks with their edit versions (`grid` = the raw id `info`
//!   prints).
//! - `extract <snapshot> <grid> <chx,chy,chz> <out.(vxl|kv6|vox)>` —
//!   pull one chunk out of a scene snapshot: as a raw `.vxl` world,
//!   a `.kv6` model, or a MagicaVoxel `.vox` (open a player-carved
//!   chunk in the editor). The voxlap bedrock placeholder plane
//!   (chunk-local z = 255) is dropped from `.kv6`/`.vox` — it's
//!   format bookkeeping, not content — and kept in `.vxl`.
//!
//! Exit codes: `0` ok, `1` operation failed, `2` usage error.

use std::path::Path;
use std::process::ExitCode;

use roxlap_formats::voxel_clip::{LoopMode, VoxelClip};
use roxlap_formats::{character, kfa, kv6, kvx, vox, vxl};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let str_args: Vec<&str> = args.iter().map(String::as_str).collect();
    match str_args.as_slice() {
        ["info", path] => run(info(Path::new(path))),
        ["vox2kv6", input, output] => run(vox2kv6(Path::new(input), Path::new(output))),
        ["vox2rvc", input, output] => run(vox2rvc(Path::new(input), Path::new(output), 80)),
        ["vox2rvc", input, output, ms] => match ms.parse::<u32>() {
            Ok(ms) if ms > 0 => run(vox2rvc(Path::new(input), Path::new(output), ms)),
            _ => usage(&format!("frame_ms must be a positive integer, got {ms:?}")),
        },
        ["kv62vox", input, output] => run(kv62vox(Path::new(input), Path::new(output))),
        ["gif2rvc", input, output] => run(gif2rvc(Path::new(input), Path::new(output), 1)),
        ["gif2rvc", input, output, th] => match th.parse::<u32>() {
            Ok(th) if th > 0 => run(gif2rvc(Path::new(input), Path::new(output), th)),
            _ => usage(&format!("thickness must be a positive integer, got {th:?}")),
        },
        ["png2rvc", input, output] => run(png2rvc_apng(Path::new(input), Path::new(output), 1)),
        ["png2rvc", input, output, th] if th.parse::<u32>().is_ok_and(|t| t > 0) => {
            let th = th.parse::<u32>().expect("guard checked");
            run(png2rvc_apng(Path::new(input), Path::new(output), th))
        }
        ["png2rvc", parts @ ..] if parts.len() > 2 => {
            let frames: Vec<&Path> = parts[..parts.len() - 1].iter().map(Path::new).collect();
            run(png2rvc_frames(&frames, Path::new(parts[parts.len() - 1])))
        }
        ["chunks", snap, grid] => match grid.parse::<u32>() {
            Ok(grid) => run(list_chunks(Path::new(snap), grid)),
            Err(_) => usage(&format!("grid must be the raw integer id, got {grid:?}")),
        },
        ["extract", snap, grid, chunk, out] => {
            match (grid.parse::<u32>(), parse_chunk_idx(chunk)) {
                (Ok(grid), Some(idx)) => run(extract(Path::new(snap), grid, idx, Path::new(out))),
                (Err(_), _) => usage(&format!("grid must be the raw integer id, got {grid:?}")),
                (_, None) => usage(&format!(
                    "chunk must be three comma-separated integers (chx,chy,chz), got {chunk:?}"
                )),
            }
        }
        _ => usage("expected a subcommand"),
    }
}

fn run(result: Result<(), String>) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("roxlap-cli: {e}");
            ExitCode::FAILURE
        }
    }
}

fn usage(problem: &str) -> ExitCode {
    eprintln!(
        "roxlap-cli: {problem}\n\n\
         Usage:\n  \
         roxlap-cli info <file>\n  \
         roxlap-cli vox2kv6 <in.vox> <out.kv6>\n  \
         roxlap-cli vox2rvc <in.vox> <out.rvc> [frame_ms]\n  \
         roxlap-cli kv62vox <in.kv6> <out.vox>\n  \
         roxlap-cli gif2rvc <in.gif> <out.rvc> [thickness]\n  \
         roxlap-cli png2rvc <in.(a)png> <out.rvc> [thickness]\n  \
         roxlap-cli png2rvc <frame0.png> <frame1.png> … <out.rvc>\n  \
         roxlap-cli chunks <snapshot> <grid>\n  \
         roxlap-cli extract <snapshot> <grid> <chx,chy,chz> <out.(vxl|kv6|vox)>"
    );
    ExitCode::from(2)
}

fn read(path: &Path) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|e| format!("reading {}: {e}", path.display()))
}

fn write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    std::fs::write(path, bytes).map_err(|e| format!("writing {}: {e}", path.display()))?;
    println!("wrote {} ({} bytes)", path.display(), bytes.len());
    Ok(())
}

// ---- info ---------------------------------------------------------------

/// What the magic bytes (or, failing that, the extension) say a file is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Vox,
    Kv6,
    Kvx,
    Vxl,
    Kfa,
    Rvc,
    Rkc,
    Snapshot,
}

/// Sniff the format. Magic bytes win (`.vox`/`.kv6`/`.kfa`/`.rvc`/
/// `.rkc`/snapshots all carry one); `.kvx` and `.vxl` have none, so
/// they fall back to the file extension.
fn sniff(path: &Path, bytes: &[u8]) -> Option<Kind> {
    if bytes.len() >= 4 {
        match &bytes[..4] {
            b"VOX " => return Some(Kind::Vox),
            b"Kvxl" => return Some(Kind::Kv6),
            b"Kwlk" => return Some(Kind::Kfa),
            b"RVCL" => return Some(Kind::Rvc),
            b"RKCH" => return Some(Kind::Rkc),
            m if *m == roxlap_scene::snapshot::SNAPSHOT_MAGIC => return Some(Kind::Snapshot),
            _ => {}
        }
    }
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("kvx") => Some(Kind::Kvx),
        Some("vxl") => Some(Kind::Vxl),
        _ => None,
    }
}

fn info(path: &Path) -> Result<(), String> {
    let bytes = read(path)?;
    let kind = sniff(path, &bytes).ok_or_else(|| {
        format!(
            "{}: unrecognised format (no known magic; .kvx/.vxl are \
             detected by extension)",
            path.display()
        )
    })?;
    print!(
        "{}",
        summarize(kind, &bytes).map_err(|e| format!("{}: {e}", path.display()))?
    );
    Ok(())
}

/// Parse `bytes` as `kind` and render the human summary. Separated
/// from [`info`] so tests can drive it without touching the fs.
fn summarize(kind: Kind, bytes: &[u8]) -> Result<String, String> {
    use std::fmt::Write as _;
    let mut out = String::new();
    let w = &mut out;
    match kind {
        Kind::Vox => {
            let f = vox::parse(bytes).map_err(|e| e.to_string())?;
            let _ = writeln!(w, "MagicaVoxel .vox: {} model(s)", f.models.len());
            for (i, m) in f.models.iter().enumerate() {
                let _ = writeln!(
                    w,
                    "  model {i}: {}x{}x{}, {} voxels",
                    m.size_x,
                    m.size_y,
                    m.size_z,
                    m.voxels.len()
                );
            }
        }
        Kind::Kv6 => {
            let m = kv6::parse(bytes).map_err(|e| e.to_string())?;
            let _ = writeln!(
                w,
                "kv6 sprite model: {}x{}x{}, {} voxels, pivot ({}, {}, {})",
                m.xsiz,
                m.ysiz,
                m.zsiz,
                m.voxels.len(),
                m.xpiv,
                m.ypiv,
                m.zpiv
            );
        }
        Kind::Kvx => {
            let m = kvx::parse(bytes).map_err(|e| e.to_string())?;
            let _ = writeln!(w, "kvx (Build) model: {}x{}x{}", m.xsiz, m.ysiz, m.zsiz);
        }
        Kind::Vxl => {
            let m = vxl::parse(bytes).map_err(|e| e.to_string())?;
            let _ = writeln!(
                w,
                "vxl world: {0}x{0} columns, camera at ({1:.1}, {2:.1}, {3:.1})",
                m.vsid, m.ipo[0], m.ipo[1], m.ipo[2]
            );
        }
        Kind::Kfa => {
            let m = kfa::parse(bytes).map_err(|e| e.to_string())?;
            let _ = writeln!(
                w,
                "kfa animation rig: for {:?}, {} hinge(s), {} frame(s), {} sequence entrie(s)",
                String::from_utf8_lossy(&m.kv6_name),
                m.hinges.len(),
                m.frmval.len(),
                m.seq.len()
            );
        }
        Kind::Rvc => {
            // `voxel_clip::ParseError` is Debug-only (no Display).
            let c = VoxelClip::parse(bytes).map_err(|e| format!("{e:?}"))?;
            let total_ms: u32 = if c.durations.is_empty() {
                c.default_frame_ms * u32::try_from(c.frames.len()).unwrap_or(0)
            } else {
                c.durations.iter().sum()
            };
            let _ = writeln!(
                w,
                "rvc voxel clip: {}x{}x{}, {} frame(s), {} ms total, {:?}",
                c.dims[0],
                c.dims[1],
                c.dims[2],
                c.frames.len(),
                total_ms,
                c.loop_mode
            );
        }
        Kind::Rkc => {
            let c = character::parse(bytes).map_err(|e| e.to_string())?;
            let _ = writeln!(
                w,
                "rkc character {:?}: {} mesh(es), {} bone(s), {} clip(s), {} voxel clip(s)",
                c.name,
                c.meshes.len(),
                c.bones.len(),
                c.clips.len(),
                c.voxel_clips.len()
            );
        }
        Kind::Snapshot => {
            let scene = roxlap_scene::Scene::load_snapshot(bytes).map_err(|e| format!("{e:?}"))?;
            let _ = writeln!(w, "scene snapshot: {} grid(s)", scene.grid_count());
            for (id, g) in scene.grids() {
                let edited = g.chunk_versions().values().filter(|&&v| v != 0).count();
                let _ = writeln!(
                    w,
                    "  grid #{}: name {:?}, {} chunk(s) ({} edited), origin ({:.1}, {:.1}, {:.1})",
                    id.raw(),
                    g.name.as_deref().unwrap_or("-"),
                    g.chunk_count(),
                    edited,
                    g.transform.origin.x,
                    g.transform.origin.y,
                    g.transform.origin.z
                );
            }
        }
    }
    Ok(out)
}

// ---- conversions ---------------------------------------------------------

fn parse_vox_models(bytes: &[u8]) -> Result<Vec<roxlap_formats::kv6::Kv6>, String> {
    let f = vox::parse(bytes).map_err(|e| e.to_string())?;
    let models = f.to_kv6_models();
    if models.is_empty() {
        return Err("the .vox file contains no models".to_string());
    }
    Ok(models)
}

fn vox2kv6(input: &Path, output: &Path) -> Result<(), String> {
    let models = parse_vox_models(&read(input)?)?;
    if models.len() == 1 {
        return write(output, &kv6::serialize(&models[0]));
    }
    // Multi-model: out.kv6 → out_0.kv6, out_1.kv6, …
    let stem = output.file_stem().and_then(|s| s.to_str()).unwrap_or("out");
    let ext = output.extension().and_then(|e| e.to_str()).unwrap_or("kv6");
    for (i, m) in models.iter().enumerate() {
        let path = output.with_file_name(format!("{stem}_{i}.{ext}"));
        write(&path, &kv6::serialize(m))?;
    }
    Ok(())
}

fn vox2rvc(input: &Path, output: &Path, frame_ms: u32) -> Result<(), String> {
    let clip = vox_to_clip(&read(input)?, frame_ms)?;
    write(output, &clip.serialize())
}

/// The conversion core, fs-free for tests: every `.vox` model becomes
/// one clip frame (auto keyframe/delta encoding, looping).
fn vox_to_clip(bytes: &[u8], frame_ms: u32) -> Result<VoxelClip, String> {
    let models = parse_vox_models(bytes)?;
    VoxelClip::from_kv6_frames_auto(&models, 1.0, LoopMode::Loop, &[], frame_ms, 0)
        .map_err(|e| format!("building the clip: {e:?} (all models must share one size)"))
}

fn kv62vox(input: &Path, output: &Path) -> Result<(), String> {
    let kv6 = kv6::parse(&read(input)?).map_err(|e| e.to_string())?;
    let file = vox::VoxFile::from_kv6(&kv6);
    write(output, &vox::serialize(&file))
}

fn gif2rvc(input: &Path, output: &Path, thickness: u32) -> Result<(), String> {
    use roxlap_formats::gif_import::{voxel_clip_from_gif, GifImportOpts};
    let opts = GifImportOpts {
        thickness,
        ..GifImportOpts::default()
    };
    let clip = voxel_clip_from_gif(&read(input)?, &opts).map_err(|e| e.to_string())?;
    write(output, &clip.serialize())
}

fn png2rvc_apng(input: &Path, output: &Path, thickness: u32) -> Result<(), String> {
    use roxlap_formats::png_import::{voxel_clip_from_apng, PngImportOpts};
    let opts = PngImportOpts {
        thickness,
        ..PngImportOpts::default()
    };
    let clip = voxel_clip_from_apng(&read(input)?, &opts).map_err(|e| e.to_string())?;
    write(output, &clip.serialize())
}

fn png2rvc_frames(inputs: &[&Path], output: &Path) -> Result<(), String> {
    use roxlap_formats::png_import::{voxel_clip_from_png_frames, PngImportOpts};
    let bytes: Vec<Vec<u8>> = inputs.iter().map(|p| read(p)).collect::<Result<_, _>>()?;
    let frames: Vec<&[u8]> = bytes.iter().map(Vec::as_slice).collect();
    // Uniform default timing; per-frame delays belong to the APNG form.
    let clip = voxel_clip_from_png_frames(&frames, &[], &PngImportOpts::default())
        .map_err(|e| e.to_string())?;
    write(output, &clip.serialize())
}

// ---- snapshot chunk extraction ---------------------------------------------

/// `"chx,chy,chz"` → chunk index.
fn parse_chunk_idx(s: &str) -> Option<glam::IVec3> {
    let parts: Vec<i32> = s
        .split(',')
        .map(str::trim)
        .map(str::parse)
        .collect::<Result<_, _>>()
        .ok()?;
    (parts.len() == 3).then(|| glam::IVec3::new(parts[0], parts[1], parts[2]))
}

fn load_scene(path: &Path) -> Result<roxlap_scene::Scene, String> {
    roxlap_scene::Scene::load_snapshot(&read(path)?)
        .map_err(|e| format!("{}: not a readable scene snapshot: {e:?}", path.display()))
}

/// Find a grid by the raw id `info` prints.
fn grid_by_raw(scene: &roxlap_scene::Scene, raw: u32) -> Result<&roxlap_scene::Grid, String> {
    scene
        .grids()
        .find(|(id, _)| id.raw() == raw)
        .map(|(_, g)| g)
        .ok_or_else(|| {
            let have: Vec<String> = scene.grids().map(|(id, _)| id.raw().to_string()).collect();
            format!("no grid {raw} in the snapshot (grids: {})", have.join(", "))
        })
}

fn list_chunks(snap: &Path, grid_raw: u32) -> Result<(), String> {
    let scene = load_scene(snap)?;
    let grid = grid_by_raw(&scene, grid_raw)?;
    let mut idxs: Vec<(glam::IVec3, u64)> = grid
        .chunk_versions()
        .iter()
        .map(|(&i, &v)| (i, v))
        .collect();
    idxs.sort_by_key(|(i, _)| (i.z, i.y, i.x));
    println!(
        "grid {grid_raw} ({}): {} chunk(s)",
        grid.name.as_deref().unwrap_or("-"),
        idxs.len()
    );
    for (i, v) in idxs {
        let state = if v == 0 {
            "pristine".to_string()
        } else {
            format!("edited v{v}")
        };
        println!("  ({}, {}, {}) {state}", i.x, i.y, i.z);
    }
    Ok(())
}

/// The chunk as a kv6 model: every voxel of the chunk except the
/// voxlap bedrock placeholder plane (chunk-local `z = 255` — format
/// bookkeeping, not content; the demos render it as air too).
fn chunk_to_kv6(chunk: &roxlap_formats::vxl::Vxl) -> roxlap_formats::kv6::Kv6 {
    let vsid = chunk.vsid;
    roxlap_formats::kv6::Kv6::from_fn(vsid, vsid, 255, |x, y, z| chunk.voxel_color(x, y, z))
}

fn extract(snap: &Path, grid_raw: u32, idx: glam::IVec3, out: &Path) -> Result<(), String> {
    let scene = load_scene(snap)?;
    let bytes = extract_chunk_bytes(&scene, grid_raw, idx, out)?;
    write(out, &bytes)
}

/// The conversion core, fs-free for tests: chunk → bytes in the
/// format `out`'s extension names.
fn extract_chunk_bytes(
    scene: &roxlap_scene::Scene,
    grid_raw: u32,
    idx: glam::IVec3,
    out: &Path,
) -> Result<Vec<u8>, String> {
    let grid = grid_by_raw(scene, grid_raw)?;
    let chunk = grid.chunk(idx).ok_or_else(|| {
        format!(
            "grid {grid_raw} has no chunk ({}, {}, {}) — `roxlap-cli chunks` lists them",
            idx.x, idx.y, idx.z
        )
    })?;
    match out
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("vxl") => Ok(vxl::serialize(chunk)),
        Some("kv6") => Ok(kv6::serialize(&chunk_to_kv6(chunk))),
        Some("vox") => Ok(vox::serialize(&vox::VoxFile::from_kv6(&chunk_to_kv6(
            chunk,
        )))),
        other => Err(format!(
            "unsupported output extension {other:?} (use .vxl, .kv6 or .vox)"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal two-model `.vox` (3x3x3 each) built in memory — the
    /// same synthesis approach as `roxlap-formats`' `book_assets`
    /// example.
    fn test_vox(models: usize) -> Vec<u8> {
        let mut v = Vec::new();
        let chunk = |v: &mut Vec<u8>, id: &[u8; 4], content: &[u8]| {
            v.extend_from_slice(id);
            v.extend_from_slice(&u32::try_from(content.len()).expect("small").to_le_bytes());
            v.extend_from_slice(&0u32.to_le_bytes());
            v.extend_from_slice(content);
        };
        v.extend_from_slice(b"VOX ");
        v.extend_from_slice(&150u32.to_le_bytes());
        let size = [3u32, 3, 3].map(u32::to_le_bytes).concat();
        let mut xyzi = 1u32.to_le_bytes().to_vec();
        xyzi.extend_from_slice(&[1, 1, 1, 7]);
        let per_model = 12 + size.len() + 12 + xyzi.len();
        v.extend_from_slice(b"MAIN");
        v.extend_from_slice(&0u32.to_le_bytes());
        v.extend_from_slice(
            &u32::try_from(per_model * models)
                .expect("small")
                .to_le_bytes(),
        );
        for _ in 0..models {
            chunk(&mut v, b"SIZE", &size);
            chunk(&mut v, b"XYZI", &xyzi);
        }
        v
    }

    #[test]
    fn sniffs_by_magic_and_extension() {
        let vox = test_vox(1);
        assert_eq!(sniff(Path::new("noext"), &vox), Some(Kind::Vox));
        let kv6 = kv6::serialize(&parse_vox_models(&vox).expect("one model")[0]);
        assert_eq!(sniff(Path::new("m.bin"), &kv6), Some(Kind::Kv6));
        // No magic ⇒ extension decides; unknown ⇒ None.
        assert_eq!(sniff(Path::new("w.vxl"), &[0; 8]), Some(Kind::Vxl));
        assert_eq!(sniff(Path::new("w.dat"), &[0; 8]), None);
    }

    #[test]
    fn vox_summary_counts_models() {
        let s = summarize(Kind::Vox, &test_vox(2)).expect("valid vox");
        assert!(s.contains("2 model(s)"), "got: {s}");
        assert!(s.contains("3x3x3, 1 voxels"), "got: {s}");
    }

    #[test]
    fn vox_to_clip_uses_models_as_frames() {
        let clip = vox_to_clip(&test_vox(3), 120).expect("uniform dims");
        assert_eq!(clip.frames.len(), 3);
        assert_eq!(clip.dims, [3, 3, 3]);
        // Round-trip through the .rvc wire format.
        let reparsed = VoxelClip::parse(&clip.serialize()).expect("self-authored rvc");
        assert_eq!(reparsed.decode().expect("decodes").frame_count(), 3);
        // And the summary agrees.
        let s = summarize(Kind::Rvc, &clip.serialize()).expect("valid rvc");
        assert!(s.contains("3 frame(s), 360 ms total"), "got: {s}");
    }

    #[test]
    fn kv6_to_vox_export_chain() {
        // vox → kv6 → vox → summary: the reverse trip stays parseable
        // and keeps the voxel count.
        let kv6 = &parse_vox_models(&test_vox(1)).expect("one model")[0];
        let bytes = vox::serialize(&vox::VoxFile::from_kv6(kv6));
        assert_eq!(sniff(Path::new("x"), &bytes), Some(Kind::Vox));
        let s = summarize(Kind::Vox, &bytes).expect("re-exported vox parses");
        assert!(s.contains("1 model(s)"), "got: {s}");
        assert!(s.contains("3x3x3, 1 voxels"), "got: {s}");
    }

    /// Snapshot → chunk → every export format round-trips: `.vxl`
    /// reparses as the same chunk, `.kv6`/`.vox` carry the edited
    /// voxels (minus the bedrock placeholder plane, dropped by
    /// design).
    /// CT.8 — extracting a carved-through chunk: empty-sentinel and
    /// air-terminal columns export cleanly in every format (no phantom
    /// bedrock voxels reappear on the way out).
    #[test]
    fn extract_carved_through_chunk() {
        use glam::{DVec3, IVec3};
        let mut scene = roxlap_scene::Scene::new();
        let id = scene.add_grid(roxlap_scene::GridTransform::at(DVec3::ZERO));
        let g = scene.grid_mut(id).expect("just added");
        g.set_rect(
            IVec3::new(0, 0, 250),
            IVec3::new(3, 3, 255),
            Some(roxlap_scene::VoxColor(0x80_aa_bb_00)),
        );
        // One column emptied entirely, one dug through from z=252.
        g.set_rect(IVec3::new(1, 1, 0), IVec3::new(1, 1, 255), None);
        g.set_rect(IVec3::new(2, 2, 252), IVec3::new(2, 2, 255), None);
        let raw = id.raw();
        let idx = IVec3::ZERO;

        let vxl_bytes = extract_chunk_bytes(&scene, raw, idx, Path::new("c.vxl")).expect("vxl");
        let reparsed = vxl::parse(&vxl_bytes).expect("carved vxl parses");
        assert_eq!(reparsed.voxel_color(1, 1, 255), None, "emptied column");
        assert_eq!(reparsed.voxel_color(2, 2, 255), None, "dug-through floor");
        assert!(reparsed.voxel_color(2, 2, 251).is_some(), "survivor kept");

        let kv6_bytes = extract_chunk_bytes(&scene, raw, idx, Path::new("c.kv6")).expect("kv6");
        let model = kv6::parse(&kv6_bytes).expect("carved kv6 parses");
        // Walk the column-major voxel stream via the ylen tables: the
        // emptied column (1,1) exports zero voxels, the dug-through
        // column (2,2) nothing at z >= 252.
        let mut cursor = 0usize;
        for (x, row) in model.ylen.iter().enumerate() {
            for (y, &n) in row.iter().enumerate() {
                let col = &model.voxels[cursor..cursor + n as usize];
                cursor += n as usize;
                if (x, y) == (1, 1) {
                    assert!(col.is_empty(), "emptied column must export nothing");
                }
                if (x, y) == (2, 2) {
                    assert!(
                        col.iter().all(|v| v.z < 252),
                        "dug-through column must export nothing below the carve"
                    );
                }
            }
        }

        let vox_bytes = extract_chunk_bytes(&scene, raw, idx, Path::new("c.vox")).expect("vox");
        vox::parse(&vox_bytes).expect("carved vox parses");
    }

    #[test]
    fn extract_chunk_all_formats() {
        use glam::{DVec3, IVec3};
        let mut scene = roxlap_scene::Scene::new();
        let id = scene.add_grid(roxlap_scene::GridTransform::at(DVec3::ZERO));
        let g = scene.grid_mut(id).expect("just added");
        g.set_voxel(
            IVec3::new(3, 4, 100),
            Some(roxlap_scene::VoxColor(0x80_aa_bb_cc)),
        );
        g.set_voxel(
            IVec3::new(3, 5, 100),
            Some(roxlap_scene::VoxColor(0x80_11_22_33)),
        );
        let raw = id.raw();
        let idx = IVec3::ZERO;

        // .vxl: byte round-trip through the raw chunk serializer.
        let vxl_bytes = extract_chunk_bytes(&scene, raw, idx, Path::new("c.vxl")).expect("vxl");
        let reparsed = vxl::parse(&vxl_bytes).expect("self-authored vxl");
        assert_eq!(reparsed.vsid, 128);
        assert_eq!(
            reparsed.voxel_color(3, 4, 100),
            Some(roxlap_scene::VoxColor(0x80_aa_bb_cc))
        );

        // .kv6: the two edited voxels, WITHOUT the bedrock plane.
        let kv6_bytes = extract_chunk_bytes(&scene, raw, idx, Path::new("c.kv6")).expect("kv6");
        let model = kv6::parse(&kv6_bytes).expect("self-authored kv6");
        assert_eq!((model.xsiz, model.ysiz, model.zsiz), (128, 128, 255));
        assert_eq!(model.voxels.len(), 2, "bedrock plane dropped");

        // .vox: same voxel count through the MagicaVoxel exporter.
        let vox_bytes = extract_chunk_bytes(&scene, raw, idx, Path::new("c.vox")).expect("vox");
        let f = vox::parse(&vox_bytes).expect("self-authored vox");
        assert_eq!(f.models[0].voxels.len(), 2);

        // Error paths name the problem.
        assert!(extract_chunk_bytes(&scene, raw, idx, Path::new("c.png"))
            .unwrap_err()
            .contains("unsupported output extension"));
        assert!(
            extract_chunk_bytes(&scene, raw, IVec3::new(9, 9, 0), Path::new("c.vxl"))
                .unwrap_err()
                .contains("has no chunk")
        );
        assert!(extract_chunk_bytes(&scene, 77, idx, Path::new("c.vxl"))
            .unwrap_err()
            .contains("no grid 77"));
        // And the arg parser.
        assert_eq!(parse_chunk_idx("1,-2, 3"), Some(IVec3::new(1, -2, 3)));
        assert_eq!(parse_chunk_idx("1,2"), None);
        assert_eq!(parse_chunk_idx("a,b,c"), None);
    }

    /// Unique per-test scratch dir under the system temp root (kept
    /// tiny; the OS reaps it).
    fn scratch(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir()
            .join(format!("roxlap-cli-tests-{}", std::process::id()))
            .join(name);
        std::fs::create_dir_all(&d).expect("create scratch dir");
        d
    }

    /// Encode RGBA frames as in-memory GIF bytes (mirror of the
    /// `roxlap-formats` gif_import test helper).
    fn encode_gif(w: u16, h: u16, frames: &[(Vec<u8>, u16)]) -> Vec<u8> {
        let mut out = Vec::new();
        {
            let mut enc = gif::Encoder::new(&mut out, w, h, &[]).expect("gif encoder");
            enc.set_repeat(gif::Repeat::Infinite).expect("repeat");
            for (rgba, delay) in frames {
                let mut buf = rgba.clone();
                let mut frame = gif::Frame::from_rgba(w, h, &mut buf);
                frame.delay = *delay;
                enc.write_frame(&frame).expect("write frame");
            }
        }
        out
    }

    /// Encode one RGBA image as in-memory PNG bytes.
    fn encode_png(w: u32, h: u32, rgba: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        {
            let mut enc = png::Encoder::new(&mut out, w, h);
            enc.set_color(png::ColorType::Rgba);
            enc.set_depth(png::BitDepth::Eight);
            let mut writer = enc.write_header().expect("png header");
            writer.write_image_data(rgba).expect("png data");
        }
        out
    }

    /// `vox2kv6`'s file contract: one model writes `output` verbatim,
    /// a multi-model file fans out to `stem_0.ext`, `stem_1.ext`, …
    /// (and does NOT write the plain `output` path).
    #[test]
    fn vox2kv6_single_and_multi_model_files() {
        let d = scratch("vox2kv6");
        let single = d.join("one.vox");
        std::fs::write(&single, test_vox(1)).expect("write fixture");
        let out = d.join("out.kv6");
        vox2kv6(&single, &out).expect("single-model conversion");
        kv6::parse(&std::fs::read(&out).expect("read out.kv6")).expect("out.kv6 parses");

        let multi = d.join("two.vox");
        std::fs::write(&multi, test_vox(2)).expect("write fixture");
        let out2 = d.join("mm.kv6");
        vox2kv6(&multi, &out2).expect("multi-model conversion");
        assert!(!out2.exists(), "multi-model must write numbered files only");
        for i in 0..2 {
            let p = d.join(format!("mm_{i}.kv6"));
            kv6::parse(&std::fs::read(&p).unwrap_or_else(|e| panic!("read {p:?}: {e}")))
                .expect("numbered kv6 parses");
        }
    }

    /// `vox2rvc` + `kv62vox` end-to-end through the filesystem: the
    /// written `.rvc` / `.vox` reparse with the expected content.
    #[test]
    fn vox2rvc_and_kv62vox_file_roundtrip() {
        let d = scratch("file-roundtrip");
        let vox_in = d.join("anim.vox");
        std::fs::write(&vox_in, test_vox(3)).expect("write fixture");
        let rvc_out = d.join("anim.rvc");
        vox2rvc(&vox_in, &rvc_out, 120).expect("vox2rvc");
        let clip = VoxelClip::parse(&std::fs::read(&rvc_out).expect("read rvc"))
            .expect("written rvc parses");
        assert_eq!(clip.frames.len(), 3);
        assert_eq!(clip.decode().expect("decodes").durations, vec![120; 3]);

        let kv6_in = d.join("model.kv6");
        let model = &parse_vox_models(&test_vox(1)).expect("one model")[0];
        std::fs::write(&kv6_in, kv6::serialize(model)).expect("write fixture");
        let vox_out = d.join("model.vox");
        kv62vox(&kv6_in, &vox_out).expect("kv62vox");
        let f = vox::parse(&std::fs::read(&vox_out).expect("read vox")).expect("written vox");
        assert_eq!(f.models[0].voxels.len(), 1);
    }

    /// `gif2rvc` honours the thickness argument and the GIF's frame
    /// timing (centiseconds → ms) through the written `.rvc`.
    #[test]
    fn gif2rvc_thickness_and_timing() {
        let d = scratch("gif2rvc");
        let t = [0u8, 0, 0, 0];
        let opaque = [255u8, 0, 0, 255];
        let f0: Vec<u8> = [opaque, t, t, t].concat();
        let f1: Vec<u8> = [opaque, opaque, t, t].concat();
        let gif_in = d.join("anim.gif");
        std::fs::write(&gif_in, encode_gif(2, 2, &[(f0, 5), (f1, 10)])).expect("write gif");
        let out = d.join("anim.rvc");
        gif2rvc(&gif_in, &out, 3).expect("gif2rvc");
        let clip = VoxelClip::parse(&std::fs::read(&out).expect("read rvc")).expect("parses");
        assert_eq!(clip.dims, [2, 3, 2], "thickness extrudes the slab");
        let dec = clip.decode().expect("decodes");
        assert_eq!(dec.frame_count(), 2);
        assert_eq!(dec.durations, vec![50, 100]);
        assert_eq!(dec.frames[0].colors.len(), 3, "1 opaque px × 3 layers");
    }

    /// `png2rvc` both ways: a numbered frame sequence and a single
    /// still (the APNG path's degenerate case).
    #[test]
    fn png2rvc_sequence_and_still() {
        let d = scratch("png2rvc");
        let t = [0u8, 0, 0, 0];
        let opaque = [0u8, 255, 0, 255];
        let p0 = d.join("f0.png");
        let p1 = d.join("f1.png");
        std::fs::write(&p0, encode_png(2, 2, &[opaque, t, t, t].concat())).expect("write f0");
        std::fs::write(&p1, encode_png(2, 2, &[opaque, opaque, t, t].concat())).expect("f1");
        let out = d.join("seq.rvc");
        png2rvc_frames(&[&p0, &p1], &out).expect("png2rvc frames");
        let clip = VoxelClip::parse(&std::fs::read(&out).expect("read rvc")).expect("parses");
        assert_eq!(clip.decode().expect("decodes").frame_count(), 2);

        let still_out = d.join("still.rvc");
        png2rvc_apng(&p0, &still_out, 2).expect("png2rvc still");
        let clip = VoxelClip::parse(&std::fs::read(&still_out).expect("read rvc")).expect("ok");
        assert_eq!(clip.dims, [2, 2, 2], "thickness 2 slab");
        assert_eq!(clip.decode().expect("decodes").frame_count(), 1);
    }

    /// `info` succeeds on a real asset and reports unreadable /
    /// unrecognised inputs as errors that name the problem.
    #[test]
    fn info_reads_assets_and_reports_errors() {
        let d = scratch("info");
        let vox_in = d.join("m.vox");
        std::fs::write(&vox_in, test_vox(1)).expect("write fixture");
        info(&vox_in).expect("info on a valid .vox");

        let missing = info(Path::new("definitely/not/here.vox")).expect_err("missing file");
        assert!(
            missing.contains("not/here.vox"),
            "names the path: {missing}"
        );

        let junk = d.join("junk.dat");
        std::fs::write(&junk, [0u8; 16]).expect("write junk");
        let unrecognised = info(&junk).expect_err("unrecognised bytes");
        assert!(
            unrecognised.contains("unrecognised") || unrecognised.contains("unknown"),
            "explains the failure: {unrecognised}"
        );
    }

    #[test]
    fn snapshot_summary_lists_grids() {
        use glam::{DVec3, IVec3};
        let mut scene = roxlap_scene::Scene::new();
        let id = scene.add_grid(roxlap_scene::GridTransform::at(DVec3::new(4.0, 0.0, 0.0)));
        let g = scene.grid_mut(id).expect("just added");
        g.name = Some("ground".to_string());
        g.set_voxel(
            IVec3::new(0, 0, 200),
            Some(roxlap_scene::VoxColor(0x80_40_40_40)),
        );
        let s = summarize(Kind::Snapshot, &scene.save_snapshot()).expect("valid snapshot");
        assert!(s.contains("1 grid(s)"), "got: {s}");
        assert!(s.contains("\"ground\", 1 chunk(s) (1 edited)"), "got: {s}");
    }
}
