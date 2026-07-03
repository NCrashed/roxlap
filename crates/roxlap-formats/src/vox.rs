//! MagicaVoxel `.vox` importer (QE.6a) — the industry-standard voxel
//! authoring format, hand-parsed in the house style (shared
//! `Cursor`, per-format `ParseError`, no external decoder deps).
//!
//! Scope: the chunks every `.vox` build writes — `SIZE` + `XYZI`
//! model pairs and the `RGBA` palette — which covers plain models
//! and multi-model files. The scene graph (`nTRN`/`nGRP`/`nSHP`
//! transforms), materials, and cameras are **skipped**: a multi-model
//! file yields its models in file order without world placement (the
//! host positions them). Files without an `RGBA` chunk fall back to
//! the official default palette.
//!
//! Coordinate mapping in [`VoxModel::to_kv6`](crate::vox::VoxModel::to_kv6): MagicaVoxel is
//! z-**up**, voxlap/roxlap is z-**down**, so `(x, y, z)` maps to
//! `(x, y, zsiz - 1 - z)` — a model right-side-up in the editor is
//! right-side-up in the engine. Colours become voxlap-packed
//! `0x80_RR_GG_BB` (neutral brightness; alpha is dropped — pair the
//! import with a colour→material map for translucency, exactly like
//! the GIF/PNG importers).
//!
//! ```no_run
//! use roxlap_formats::vox;
//! let bytes = std::fs::read("castle.vox").unwrap();
//! let file = vox::parse(&bytes).unwrap();
//! let models = file.to_kv6_models(); // ready for add_sprite_model
//! ```

use std::fmt;

use crate::bytes::{Cursor, OutOfBounds};
use crate::kv6::Kv6;

/// Per-axis model size cap, from the format spec (MagicaVoxel's own
/// editor limit). Also bounds [`VoxModel::to_kv6`]'s dense scratch
/// grid to 256³ = 16 MiB — a crafted `SIZE` chunk can't allocation-
/// bomb the importer (QE.6b).
pub const MAX_MODEL_DIM: u32 = 256;

/// Errors from [`parse`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseError {
    /// File too small for the 8-byte `VOX ` header.
    TooSmall { got: usize },
    /// First 4 bytes are not the `"VOX "` magic.
    BadMagic { got: [u8; 4] },
    /// A read of `need` bytes at offset `at` would run past the end
    /// of the buffer.
    Truncated { at: usize, need: usize },
    /// A chunk header declares more content/children bytes than the
    /// file has left.
    BadChunkSize { id: [u8; 4], at: usize },
    /// A `SIZE` chunk with a zero or > [`MAX_MODEL_DIM`] dimension.
    BadModelSize { x: u32, y: u32, z: u32 },
    /// An `XYZI` chunk with no preceding unpaired `SIZE` chunk.
    OrphanXyzi { at: usize },
    /// An `XYZI` voxel count inconsistent with the chunk's size.
    BadVoxelCount { declared: u32, room_for: u32 },
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::TooSmall { got } => write!(f, "vox: {got} bytes is too small"),
            Self::BadMagic { got } => write!(f, "vox: bad magic {got:?}"),
            Self::Truncated { at, need } => {
                write!(f, "vox: truncated (need {need} bytes at {at})")
            }
            Self::BadChunkSize { id, at } => {
                write!(f, "vox: chunk {id:?} at {at} overruns the file")
            }
            Self::BadModelSize { x, y, z } => {
                write!(f, "vox: model size {x}x{y}x{z} outside 1..={MAX_MODEL_DIM}")
            }
            Self::OrphanXyzi { at } => write!(f, "vox: XYZI at {at} without a SIZE"),
            Self::BadVoxelCount { declared, room_for } => {
                write!(
                    f,
                    "vox: XYZI declares {declared} voxels, room for {room_for}"
                )
            }
        }
    }
}

impl std::error::Error for ParseError {}

impl From<OutOfBounds> for ParseError {
    fn from(e: OutOfBounds) -> Self {
        Self::Truncated {
            at: e.at,
            need: e.need,
        }
    }
}

/// One model: its size plus the sparse voxel list, exactly as stored
/// (MagicaVoxel coordinates, palette indices).
#[derive(Debug, Clone)]
pub struct VoxModel {
    /// Model extent along MagicaVoxel x (right).
    pub size_x: u32,
    /// Model extent along MagicaVoxel y (into the screen).
    pub size_y: u32,
    /// Model extent along MagicaVoxel z (**up** — flipped by
    /// [`Self::to_kv6`]).
    pub size_z: u32,
    /// Voxels as `(x, y, z, colour_index)`. Indices are 1..=255 into
    /// [`VoxFile::palette`]; out-of-size voxels are dropped at parse.
    pub voxels: Vec<(u8, u8, u8, u8)>,
}

impl VoxModel {
    /// Convert to a [`Kv6`] sprite model: z-up → z-down flip, palette
    /// colours → voxlap-packed `0x80_RR_GG_BB` (see the module docs).
    /// Surface-only like every KV6 (interiors culled).
    #[must_use]
    pub fn to_kv6(&self, palette: &[u32; 256]) -> Kv6 {
        // Dense colour-index scratch, bounded by MAX_MODEL_DIM³.
        let (sx, sy, sz) = (
            self.size_x as usize,
            self.size_y as usize,
            self.size_z as usize,
        );
        let mut grid = vec![0u8; sx * sy * sz];
        for &(x, y, z, i) in &self.voxels {
            let (x, y, z) = (x as usize, y as usize, z as usize);
            if x < sx && y < sy && z < sz && i != 0 {
                grid[(z * sy + y) * sx + x] = i;
            }
        }
        Kv6::from_fn(self.size_x, self.size_y, self.size_z, |x, y, z| {
            // z-up (MagicaVoxel) → z-down (voxlap).
            let mz = (self.size_z - 1 - z) as usize;
            let idx = grid[(mz * sy + y as usize) * sx + x as usize];
            if idx == 0 {
                return None;
            }
            let abgr = palette[idx as usize];
            let (r, g, b) = (abgr & 0xff, (abgr >> 8) & 0xff, (abgr >> 16) & 0xff);
            Some(0x8000_0000 | (r << 16) | (g << 8) | b)
        })
    }
}

/// A parsed `.vox` file: every model plus the palette (the file's
/// `RGBA` chunk, or the official default when absent). Palette slot 0
/// is "empty" and never referenced by a stored voxel.
#[derive(Debug, Clone)]
pub struct VoxFile {
    /// Models in file order (multi-model files: the scene-graph
    /// placement is not parsed — see the module docs).
    pub models: Vec<VoxModel>,
    /// 256 ABGR entries (`0xAABBGGRR`, the format's own packing).
    pub palette: [u32; 256],
}

impl VoxFile {
    /// Convert every model via [`VoxModel::to_kv6`] with this file's
    /// palette.
    #[must_use]
    pub fn to_kv6_models(&self) -> Vec<Kv6> {
        self.models
            .iter()
            .map(|m| m.to_kv6(&self.palette))
            .collect()
    }
}

/// Parse a `.vox` byte buffer. See the module docs for scope.
///
/// # Errors
/// [`ParseError`] on truncation, bad magic, overrunning chunk sizes,
/// out-of-spec model dimensions, or inconsistent voxel counts —
/// malformed input can never panic, hang, or allocation-bomb (QE.6b).
pub fn parse(bytes: &[u8]) -> Result<VoxFile, ParseError> {
    if bytes.len() < 8 {
        return Err(ParseError::TooSmall { got: bytes.len() });
    }
    let mut c = Cursor::new(bytes);
    let magic = c.read_bytes(4)?;
    if magic != b"VOX " {
        return Err(ParseError::BadMagic {
            got: [magic[0], magic[1], magic[2], magic[3]],
        });
    }
    let _version = c.read_u32()?; // 150 / 200; chunk-tagged, no dispatch needed

    let mut models: Vec<VoxModel> = Vec::new();
    let mut palette = DEFAULT_PALETTE;
    let mut pending_size: Option<(u32, u32, u32)> = None;

    // Chunk walk. The MAIN wrapper carries everything in its children;
    // walking the byte stream linearly and skipping unknown chunks by
    // their declared size handles that without recursion (child chunks
    // simply follow their parent's 12-byte header).
    while c.remaining() >= 12 {
        let at = c.pos;
        let id_bytes = c.read_bytes(4)?;
        let id = [id_bytes[0], id_bytes[1], id_bytes[2], id_bytes[3]];
        let content = c.read_u32()? as usize;
        let _children = c.read_u32()? as usize;
        if content > c.remaining() {
            return Err(ParseError::BadChunkSize { id, at });
        }
        match &id {
            b"SIZE" if content >= 12 => {
                let mut sc = Cursor::new(c.read_bytes(content)?);
                let (x, y, z) = (sc.read_u32()?, sc.read_u32()?, sc.read_u32()?);
                if x == 0
                    || y == 0
                    || z == 0
                    || x > MAX_MODEL_DIM
                    || y > MAX_MODEL_DIM
                    || z > MAX_MODEL_DIM
                {
                    return Err(ParseError::BadModelSize { x, y, z });
                }
                pending_size = Some((x, y, z));
            }
            b"XYZI" if content >= 4 => {
                let Some((sx, sy, sz)) = pending_size.take() else {
                    return Err(ParseError::OrphanXyzi { at });
                };
                let mut xc = Cursor::new(c.read_bytes(content)?);
                let declared = xc.read_u32()?;
                // Every voxel is 4 bytes; the count must fit the chunk
                // (QE.6b — a crafted count can't drive a huge
                // with_capacity).
                let room_for = u32::try_from((content - 4) / 4).unwrap_or(u32::MAX);
                if declared > room_for {
                    return Err(ParseError::BadVoxelCount { declared, room_for });
                }
                let mut voxels = Vec::with_capacity(declared as usize);
                for _ in 0..declared {
                    let v = xc.read_bytes(4)?;
                    // Out-of-size voxels are dropped (defensive; the
                    // editor never writes them).
                    if u32::from(v[0]) < sx && u32::from(v[1]) < sy && u32::from(v[2]) < sz {
                        voxels.push((v[0], v[1], v[2], v[3]));
                    }
                }
                models.push(VoxModel {
                    size_x: sx,
                    size_y: sy,
                    size_z: sz,
                    voxels,
                });
            }
            b"RGBA" if content >= 256 * 4 => {
                let mut pc = Cursor::new(c.read_bytes(content)?);
                // File entry i maps to palette index i+1 (index 0 is
                // "empty"); entry 255 is unreachable and dropped.
                for i in 0..255 {
                    palette[i + 1] = pc.read_u32()?;
                }
            }
            _ => {
                // Unknown / undersized chunk: skip its content (its
                // children follow linearly and get walked normally).
                let _ = c.read_bytes(content)?;
            }
        }
    }
    Ok(VoxFile { models, palette })
}

/// The official default palette (ABGR), applied when a file has no
/// `RGBA` chunk. Verbatim from the format spec
/// (`ephtracy/voxel-model`, `MagicaVoxel-file-format-vox.txt`).
// Verbatim spec table - separators would only obscure the diff against
// the upstream source.
#[allow(clippy::unreadable_literal)]
#[rustfmt::skip]
pub const DEFAULT_PALETTE: [u32; 256] = [
    0x00000000, 0xffffffff, 0xffccffff, 0xff99ffff, 0xff66ffff, 0xff33ffff, 0xff00ffff, 0xffffccff,
    0xffccccff, 0xff99ccff, 0xff66ccff, 0xff33ccff, 0xff00ccff, 0xffff99ff, 0xffcc99ff, 0xff9999ff,
    0xff6699ff, 0xff3399ff, 0xff0099ff, 0xffff66ff, 0xffcc66ff, 0xff9966ff, 0xff6666ff, 0xff3366ff,
    0xff0066ff, 0xffff33ff, 0xffcc33ff, 0xff9933ff, 0xff6633ff, 0xff3333ff, 0xff0033ff, 0xffff00ff,
    0xffcc00ff, 0xff9900ff, 0xff6600ff, 0xff3300ff, 0xff0000ff, 0xffffffcc, 0xffccffcc, 0xff99ffcc,
    0xff66ffcc, 0xff33ffcc, 0xff00ffcc, 0xffffcccc, 0xffcccccc, 0xff99cccc, 0xff66cccc, 0xff33cccc,
    0xff00cccc, 0xffff99cc, 0xffcc99cc, 0xff9999cc, 0xff6699cc, 0xff3399cc, 0xff0099cc, 0xffff66cc,
    0xffcc66cc, 0xff9966cc, 0xff6666cc, 0xff3366cc, 0xff0066cc, 0xffff33cc, 0xffcc33cc, 0xff9933cc,
    0xff6633cc, 0xff3333cc, 0xff0033cc, 0xffff00cc, 0xffcc00cc, 0xff9900cc, 0xff6600cc, 0xff3300cc,
    0xff0000cc, 0xffffff99, 0xffccff99, 0xff99ff99, 0xff66ff99, 0xff33ff99, 0xff00ff99, 0xffffcc99,
    0xffcccc99, 0xff99cc99, 0xff66cc99, 0xff33cc99, 0xff00cc99, 0xffff9999, 0xffcc9999, 0xff999999,
    0xff669999, 0xff339999, 0xff009999, 0xffff6699, 0xffcc6699, 0xff996699, 0xff666699, 0xff336699,
    0xff006699, 0xffff3399, 0xffcc3399, 0xff993399, 0xff663399, 0xff333399, 0xff003399, 0xffff0099,
    0xffcc0099, 0xff990099, 0xff660099, 0xff330099, 0xff000099, 0xffffff66, 0xffccff66, 0xff99ff66,
    0xff66ff66, 0xff33ff66, 0xff00ff66, 0xffffcc66, 0xffcccc66, 0xff99cc66, 0xff66cc66, 0xff33cc66,
    0xff00cc66, 0xffff9966, 0xffcc9966, 0xff999966, 0xff669966, 0xff339966, 0xff009966, 0xffff6666,
    0xffcc6666, 0xff996666, 0xff666666, 0xff336666, 0xff006666, 0xffff3366, 0xffcc3366, 0xff993366,
    0xff663366, 0xff333366, 0xff003366, 0xffff0066, 0xffcc0066, 0xff990066, 0xff660066, 0xff330066,
    0xff000066, 0xffffff33, 0xffccff33, 0xff99ff33, 0xff66ff33, 0xff33ff33, 0xff00ff33, 0xffffcc33,
    0xffcccc33, 0xff99cc33, 0xff66cc33, 0xff33cc33, 0xff00cc33, 0xffff9933, 0xffcc9933, 0xff999933,
    0xff669933, 0xff339933, 0xff009933, 0xffff6633, 0xffcc6633, 0xff996633, 0xff666633, 0xff336633,
    0xff006633, 0xffff3333, 0xffcc3333, 0xff993333, 0xff663333, 0xff333333, 0xff003333, 0xffff0033,
    0xffcc0033, 0xff990033, 0xff660033, 0xff330033, 0xff000033, 0xffffff00, 0xffccff00, 0xff99ff00,
    0xff66ff00, 0xff33ff00, 0xff00ff00, 0xffffcc00, 0xffcccc00, 0xff99cc00, 0xff66cc00, 0xff33cc00,
    0xff00cc00, 0xffff9900, 0xffcc9900, 0xff999900, 0xff669900, 0xff339900, 0xff009900, 0xffff6600,
    0xffcc6600, 0xff996600, 0xff666600, 0xff336600, 0xff006600, 0xffff3300, 0xffcc3300, 0xff993300,
    0xff663300, 0xff333300, 0xff003300, 0xffff0000, 0xffcc0000, 0xff990000, 0xff660000, 0xff330000,
    0xff0000ee, 0xff0000dd, 0xff0000bb, 0xff0000aa, 0xff000088, 0xff000077, 0xff000055, 0xff000044,
    0xff000022, 0xff000011, 0xff00ee00, 0xff00dd00, 0xff00bb00, 0xff00aa00, 0xff008800, 0xff007700,
    0xff005500, 0xff004400, 0xff002200, 0xff001100, 0xffee0000, 0xffdd0000, 0xffbb0000, 0xffaa0000,
    0xff880000, 0xff770000, 0xff550000, 0xff440000, 0xff220000, 0xff110000, 0xffeeeeee, 0xffdddddd,
    0xffbbbbbb, 0xffaaaaaa, 0xff888888, 0xff777777, 0xff555555, 0xff444444, 0xff222222, 0xff111111,
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal `.vox` writer for the tests: header + MAIN + the given
    /// raw chunks (each already including its 12-byte chunk header).
    fn vox_bytes(chunks: &[Vec<u8>]) -> Vec<u8> {
        let children: usize = chunks.iter().map(Vec::len).sum();
        let mut out = Vec::new();
        out.extend_from_slice(b"VOX ");
        out.extend_from_slice(&150u32.to_le_bytes());
        out.extend_from_slice(b"MAIN");
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&u32::try_from(children).unwrap().to_le_bytes());
        for c in chunks {
            out.extend_from_slice(c);
        }
        out
    }

    fn chunk(id: [u8; 4], content: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&id);
        out.extend_from_slice(&u32::try_from(content.len()).unwrap().to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(content);
        out
    }

    fn size_chunk(x: u32, y: u32, z: u32) -> Vec<u8> {
        let mut c = Vec::new();
        c.extend_from_slice(&x.to_le_bytes());
        c.extend_from_slice(&y.to_le_bytes());
        c.extend_from_slice(&z.to_le_bytes());
        chunk(*b"SIZE", &c)
    }

    fn xyzi_chunk(voxels: &[(u8, u8, u8, u8)]) -> Vec<u8> {
        let mut c = Vec::new();
        c.extend_from_slice(&u32::try_from(voxels.len()).unwrap().to_le_bytes());
        for &(x, y, z, i) in voxels {
            c.extend_from_slice(&[x, y, z, i]);
        }
        chunk(*b"XYZI", &c)
    }

    #[test]
    fn parses_model_palette_and_converts_to_kv6() {
        // A 2x1x3 model: one voxel at MV (0, 0, 0) [bottom] and one at
        // (1, 0, 2) [top], with a custom palette making index 1 pure
        // red and index 2 pure green (ABGR entries).
        let mut rgba = Vec::new();
        rgba.extend_from_slice(&0xff00_00ffu32.to_le_bytes()); // file[0] → idx 1: red
        rgba.extend_from_slice(&0xff00_ff00u32.to_le_bytes()); // file[1] → idx 2: green
        rgba.extend_from_slice(&vec![0u8; 254 * 4]);
        let bytes = vox_bytes(&[
            size_chunk(2, 1, 3),
            xyzi_chunk(&[(0, 0, 0, 1), (1, 0, 2, 2)]),
            chunk(*b"RGBA", &rgba),
        ]);

        let file = parse(&bytes).expect("parses");
        assert_eq!(file.models.len(), 1);
        assert_eq!(file.palette[1], 0xff00_00ff);
        let kv6 = &file.to_kv6_models()[0];
        assert_eq!((kv6.xsiz, kv6.ysiz, kv6.zsiz), (2, 1, 3));
        assert_eq!(kv6.voxels.len(), 2);
        // z-up → z-down: MV z=0 (bottom) lands at kv6 z=2; MV z=2 at 0.
        let mut cols: Vec<(u32, u16)> = kv6.voxels.iter().map(|v| (v.col, v.z)).collect();
        cols.sort_unstable_by_key(|&(_, z)| z);
        assert_eq!(cols[0], (0x8000_ff00, 0), "green voxel on top");
        assert_eq!(cols[1], (0x80ff_0000, 2), "red voxel at the bottom");
    }

    #[test]
    fn missing_rgba_falls_back_to_default_palette() {
        let bytes = vox_bytes(&[size_chunk(1, 1, 1), xyzi_chunk(&[(0, 0, 0, 1)])]);
        let file = parse(&bytes).expect("parses");
        assert_eq!(file.palette, DEFAULT_PALETTE);
        // Default index 1 is white → 0x80ffffff packed.
        assert_eq!(file.to_kv6_models()[0].voxels[0].col, 0x80ff_ffff);
    }

    #[test]
    fn multi_model_files_yield_models_in_order() {
        let bytes = vox_bytes(&[
            size_chunk(1, 1, 1),
            xyzi_chunk(&[(0, 0, 0, 1)]),
            size_chunk(2, 2, 2),
            xyzi_chunk(&[(1, 1, 1, 3)]),
        ]);
        let file = parse(&bytes).expect("parses");
        assert_eq!(file.models.len(), 2);
        assert_eq!(file.models[1].size_x, 2);
    }

    #[test]
    fn unknown_chunks_are_skipped() {
        let bytes = vox_bytes(&[
            chunk(*b"nTRN", &[0u8; 28]),
            size_chunk(1, 1, 1),
            xyzi_chunk(&[(0, 0, 0, 1)]),
            chunk(*b"MATL", &[1, 2, 3]),
        ]);
        assert_eq!(parse(&bytes).expect("parses").models.len(), 1);
    }

    // ---- QE.6b adversarial inputs: error, never panic/hang/bomb ----

    #[test]
    fn rejects_bad_magic_and_truncation() {
        assert!(matches!(parse(b"VOX"), Err(ParseError::TooSmall { .. })));
        assert!(matches!(
            parse(b"XVOX....."),
            Err(ParseError::BadMagic { .. })
        ));
        // Truncated mid-chunk-header is fine (trailing <12 bytes are
        // ignored); truncated CONTENT must error.
        let mut bytes = vox_bytes(&[size_chunk(1, 1, 1)]);
        bytes.truncate(bytes.len() - 4);
        assert!(matches!(
            parse(&bytes),
            Err(ParseError::BadChunkSize { .. } | ParseError::Truncated { .. })
        ));
    }

    #[test]
    fn rejects_huge_declared_sizes_without_allocating() {
        // SIZE outside the spec cap → error, no 4-billion³ scratch grid.
        let bytes = vox_bytes(&[size_chunk(u32::MAX, 1, 1), xyzi_chunk(&[(0, 0, 0, 1)])]);
        assert!(matches!(
            parse(&bytes),
            Err(ParseError::BadModelSize { .. })
        ));
        // XYZI declaring u32::MAX voxels in a 8-byte chunk → error, no
        // 16 GiB with_capacity.
        let mut c = Vec::new();
        c.extend_from_slice(&u32::MAX.to_le_bytes());
        c.extend_from_slice(&[0, 0, 0, 1]);
        let bytes = vox_bytes(&[size_chunk(1, 1, 1), chunk(*b"XYZI", &c)]);
        assert!(matches!(
            parse(&bytes),
            Err(ParseError::BadVoxelCount { .. })
        ));
    }

    #[test]
    fn rejects_orphan_xyzi_and_overrunning_chunk() {
        let bytes = vox_bytes(&[xyzi_chunk(&[(0, 0, 0, 1)])]);
        assert!(matches!(parse(&bytes), Err(ParseError::OrphanXyzi { .. })));

        // A chunk claiming more content than the file holds.
        let mut lying = Vec::new();
        lying.extend_from_slice(b"SIZE");
        lying.extend_from_slice(&u32::MAX.to_le_bytes());
        lying.extend_from_slice(&0u32.to_le_bytes());
        let bytes = vox_bytes(&[lying]);
        assert!(matches!(
            parse(&bytes),
            Err(ParseError::BadChunkSize { .. })
        ));
    }

    #[test]
    fn out_of_size_voxels_are_dropped_not_oob() {
        let bytes = vox_bytes(&[size_chunk(1, 1, 1), xyzi_chunk(&[(5, 5, 5, 1)])]);
        let file = parse(&bytes).expect("parses");
        assert!(file.models[0].voxels.is_empty());
        assert!(file.to_kv6_models()[0].voxels.is_empty());
    }
}
