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
         roxlap-cli vox2rvc <in.vox> <out.rvc> [frame_ms]"
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
