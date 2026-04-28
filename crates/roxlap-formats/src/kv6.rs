//! `.kv6` voxel-sprite format (Voxlap voxel sprites).
//!
//! Reference: voxlaptest's `loadkv6` in `voxlap/voxlap5.c`. File layout
//! (all multi-byte fields are little-endian):
//!
//! ```text
//! offset  size                            description
//! 0x00    4 bytes                         "Kvxl" magic
//! 0x04    u32                             xsiz
//! 0x08    u32                             ysiz
//! 0x0c    u32                             zsiz
//! 0x10    f32                             xpiv (pivot, voxel units)
//! 0x14    f32                             ypiv
//! 0x18    f32                             zpiv
//! 0x1c    u32                             numvoxs
//! 0x20    numvoxs × Voxel                 voxel records (8 bytes each)
//! ...     u32 × xsiz                      xlen — voxels per x slice
//! ...     u16 × xsiz × ysiz               ylen — voxels per (x, y) column
//! ```
//!
//! Optional trailer (present in files produced by SLAB6 and similar
//! tools; absent if the file ends after `ylen`):
//!
//! ```text
//! ...     4 bytes                         "SPal" magic
//! ...     256 × [r6 g6 b6]                palette (each component 0..=63)
//! ```
//!
//! voxlaptest's loader ignores the trailer (per-voxel `Voxel::col`
//! already carries the rendered colour); we still parse and round-trip
//! it so byte equality holds.

use core::fmt;

use crate::bytes::{Cursor, OutOfBounds};
use crate::Rgb6;

/// One voxel record (`kv6voxtype` in voxlaptest).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Voxel {
    /// Voxlap-style packed colour: `0x80RRGGBB` (alpha is the
    /// brightness flag).
    pub col: u32,
    /// z coordinate in voxel units.
    pub z: u16,
    /// Visibility-flags byte. Bits encode which of the six cube faces
    /// of this voxel are exposed.
    pub vis: u8,
    /// Index into the 256-entry surface-normal lookup table.
    pub dir: u8,
}

/// Parsed `.kv6` model. Round-trips byte-equally via [`parse`] +
/// [`serialize`].
#[derive(Debug, Clone)]
pub struct Kv6 {
    pub xsiz: u32,
    pub ysiz: u32,
    pub zsiz: u32,
    pub xpiv: f32,
    pub ypiv: f32,
    pub zpiv: f32,
    /// Voxel records in file order (`numvoxs == voxels.len() as u32`).
    pub voxels: Vec<Voxel>,
    /// `xlen[x]` is the number of voxels in the x-th slice.
    /// `xlen.len() == xsiz`. `xlen.iter().sum() == numvoxs`.
    pub xlen: Vec<u32>,
    /// `ylen[x][y]` is the number of voxels in column (x, y).
    /// Outer length `xsiz`, inner `ysiz`.
    pub ylen: Vec<Vec<u16>>,
    /// Optional trailing 256-entry palette (`"SPal"` section).
    pub palette: Option<[Rgb6; 256]>,
}

/// Errors returned by [`parse`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    /// File too small to contain even the 32-byte header.
    TooSmall { got: usize },
    /// First 4 bytes are not the `"Kvxl"` magic.
    BadMagic { got: [u8; 4] },
    /// A read of `need` bytes at offset `at` would run past the end of
    /// the buffer.
    Truncated { at: usize, need: usize },
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::TooSmall { got } => write!(
                f,
                "kv6 file too small ({got} bytes; need at least 32 byte header)"
            ),
            Self::BadMagic { got } => write!(
                f,
                "kv6 bad magic: got [{:#04x},{:#04x},{:#04x},{:#04x}], expected b\"Kvxl\"",
                got[0], got[1], got[2], got[3]
            ),
            Self::Truncated { at, need } => {
                write!(f, "kv6 truncated: need {need} bytes at offset {at}")
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

const HEADER_LEN: usize = 32;
const MAGIC: &[u8; 4] = b"Kvxl";
const PALETTE_MAGIC: &[u8; 4] = b"SPal";
const PALETTE_LEN: usize = 768;

/// Parse a `.kv6` file's bytes into a [`Kv6`].
///
/// # Errors
///
/// Returns [`ParseError`] if `bytes` is shorter than the 32-byte
/// header, if the `"Kvxl"` magic does not match, or if a sequential
/// read for any of the voxel / xlen / ylen / palette regions runs past
/// EOF.
pub fn parse(bytes: &[u8]) -> Result<Kv6, ParseError> {
    if bytes.len() < HEADER_LEN {
        return Err(ParseError::TooSmall { got: bytes.len() });
    }

    let mut cur = Cursor::new(bytes);
    let magic = cur.read_bytes(4)?;
    if magic != MAGIC {
        return Err(ParseError::BadMagic {
            got: [magic[0], magic[1], magic[2], magic[3]],
        });
    }
    let xsiz = cur.read_u32()?;
    let ysiz = cur.read_u32()?;
    let zsiz = cur.read_u32()?;
    let xpiv = cur.read_f32()?;
    let ypiv = cur.read_f32()?;
    let zpiv = cur.read_f32()?;
    let numvoxs = cur.read_u32()?;

    let mut voxels = Vec::with_capacity(numvoxs as usize);
    for _ in 0..numvoxs {
        let col = cur.read_u32()?;
        let z = cur.read_u16()?;
        let vis = cur.read_u8()?;
        let dir = cur.read_u8()?;
        voxels.push(Voxel { col, z, vis, dir });
    }

    let mut xlen = Vec::with_capacity(xsiz as usize);
    for _ in 0..xsiz {
        xlen.push(cur.read_u32()?);
    }

    let mut ylen = Vec::with_capacity(xsiz as usize);
    for _ in 0..xsiz {
        let mut row = Vec::with_capacity(ysiz as usize);
        for _ in 0..ysiz {
            row.push(cur.read_u16()?);
        }
        ylen.push(row);
    }

    // Optional "SPal" + 768-byte palette trailer.
    let palette =
        if cur.remaining() >= 4 + PALETTE_LEN && cur.peek(4) == Some(PALETTE_MAGIC.as_slice()) {
            cur.read_bytes(4)?;
            let mut pal = [Rgb6::default(); 256];
            for entry in &mut pal {
                entry.r = cur.read_u8()?;
                entry.g = cur.read_u8()?;
                entry.b = cur.read_u8()?;
            }
            Some(pal)
        } else {
            None
        };

    Ok(Kv6 {
        xsiz,
        ysiz,
        zsiz,
        xpiv,
        ypiv,
        zpiv,
        voxels,
        xlen,
        ylen,
        palette,
    })
}

/// Serialise a [`Kv6`] back to bytes. The output round-trips byte-
/// equally with the input that produced this `Kv6` via [`parse`],
/// including the optional `"SPal"` palette trailer.
///
/// # Panics
///
/// Panics if `kv6.voxels.len()` does not fit in a `u32` (the on-disk
/// `numvoxs` field is a `u32`). `Kv6` values produced by [`parse`]
/// always satisfy this.
#[must_use]
pub fn serialize(kv6: &Kv6) -> Vec<u8> {
    let pal_bytes = if kv6.palette.is_some() {
        4 + PALETTE_LEN
    } else {
        0
    };
    let body_bytes = kv6.voxels.len() * 8
        + kv6.xlen.len() * 4
        + kv6.ylen.iter().map(|row| row.len() * 2).sum::<usize>();
    let mut out = Vec::with_capacity(HEADER_LEN + body_bytes + pal_bytes);

    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&kv6.xsiz.to_le_bytes());
    out.extend_from_slice(&kv6.ysiz.to_le_bytes());
    out.extend_from_slice(&kv6.zsiz.to_le_bytes());
    out.extend_from_slice(&kv6.xpiv.to_le_bytes());
    out.extend_from_slice(&kv6.ypiv.to_le_bytes());
    out.extend_from_slice(&kv6.zpiv.to_le_bytes());
    let numvoxs =
        u32::try_from(kv6.voxels.len()).expect("kv6 numvoxs must fit in u32 (file format limit)");
    out.extend_from_slice(&numvoxs.to_le_bytes());

    for v in &kv6.voxels {
        out.extend_from_slice(&v.col.to_le_bytes());
        out.extend_from_slice(&v.z.to_le_bytes());
        out.push(v.vis);
        out.push(v.dir);
    }
    for v in &kv6.xlen {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for row in &kv6.ylen {
        for v in row {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    if let Some(pal) = &kv6.palette {
        out.extend_from_slice(PALETTE_MAGIC);
        for e in pal {
            out.push(e.r);
            out.push(e.g);
            out.push(e.b);
        }
    }

    out
}

// --- tests --------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// `assets/coco.kv6`, produced from `coco.kvx` via SLAB6.
    const COCO_KV6: &[u8] = include_bytes!("../../../assets/coco.kv6");

    #[test]
    fn parse_coco_header() {
        let kv6 = parse(COCO_KV6).expect("parse coco.kv6");
        assert_eq!(kv6.xsiz, 9);
        assert_eq!(kv6.ysiz, 11);
        assert_eq!(kv6.zsiz, 9);
        // Pivots are stored as f32 in kv6 (vs 8.8 fixed in kvx).
        assert!((kv6.xpiv - 2.0).abs() < f32::EPSILON);
        assert!((kv6.ypiv - 3.0).abs() < f32::EPSILON);
        assert!((kv6.zpiv - 9.0).abs() < f32::EPSILON);
        assert_eq!(kv6.voxels.len(), 148);
    }

    #[test]
    fn coco_voxel_counts_consistent() {
        let kv6 = parse(COCO_KV6).expect("parse coco.kv6");
        assert_eq!(kv6.xlen.len(), kv6.xsiz as usize);
        assert_eq!(kv6.ylen.len(), kv6.xsiz as usize);
        for row in &kv6.ylen {
            assert_eq!(row.len(), kv6.ysiz as usize);
        }
        let xlen_sum: u64 = kv6.xlen.iter().map(|&n| u64::from(n)).sum();
        let ylen_sum: u64 = kv6
            .ylen
            .iter()
            .flat_map(|row| row.iter().map(|&n| u64::from(n)))
            .sum();
        let nv = kv6.voxels.len() as u64;
        assert_eq!(xlen_sum, nv);
        assert_eq!(ylen_sum, nv);
    }

    #[test]
    fn coco_palette_present_and_matches_kvx() {
        let kv6 = parse(COCO_KV6).expect("parse coco.kv6");
        let pal = kv6.palette.as_ref().expect("SPal trailer present");
        // First palette entry from the hex dump matches coco.kvx's first.
        assert_eq!((pal[0].r, pal[0].g, pal[0].b), (0x3f, 0x19, 0x19));
    }

    #[test]
    fn coco_first_voxel_packed_colour() {
        let kv6 = parse(COCO_KV6).expect("parse coco.kv6");
        // From the hex dump: 60 a4 fc 80 → little-endian u32 0x80fca460.
        // High bit 0x80000000 is the brightness flag the engine sets on
        // every coloured voxel.
        let v0 = kv6.voxels[0];
        assert_eq!(v0.col, 0x80fc_a460);
        assert_eq!(v0.col & 0x8000_0000, 0x8000_0000);
    }

    #[test]
    fn coco_roundtrips_byte_equal() {
        let kv6 = parse(COCO_KV6).expect("parse coco.kv6");
        let out = serialize(&kv6);
        assert_eq!(out.len(), COCO_KV6.len(), "length differs");
        assert_eq!(out.as_slice(), COCO_KV6, "byte content differs");
    }

    #[test]
    fn parse_truncated_header_fails() {
        let r = parse(&[0u8; 16]);
        assert!(matches!(r, Err(ParseError::TooSmall { .. })));
    }

    #[test]
    fn parse_bad_magic_fails() {
        let mut bad = COCO_KV6.to_vec();
        bad[0] = b'X';
        let r = parse(&bad);
        assert!(matches!(r, Err(ParseError::BadMagic { .. })));
    }
}
