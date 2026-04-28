//! `.kvx` voxel-sprite format (Build-engine voxel sprites).
//!
//! Reference: voxlaptest's `setkvx` in `voxlap/voxlap5.c`. File layout
//! (all multi-byte fields are little-endian):
//!
//! ```text
//! offset  size                        description
//! 0x00    u32                         numbytes (header + offsets + slabs, excluding palette)
//! 0x04    u32                         xsiz
//! 0x08    u32                         ysiz
//! 0x0c    u32                         zsiz
//! 0x10    u32                         xpivot (8.8 fixed-point voxel units)
//! 0x14    u32                         ypivot
//! 0x18    u32                         zpivot
//! 0x1c    u32 × (xsiz+1)              xoffset table
//! ...     u16 × xsiz × (ysiz+1)       xyoffset table
//! ...     variable                    slab data (per (x, y) column)
//! end-768 256 × [r6 g6 b6]            palette (each component 0..=63)
//! ```
//!
//! Slab data per (x, y) column is a sequence of `[ztop:u8, zleng:u8,
//! vis:u8, colors:[u8; zleng]]` records; the byte length of column
//! (x, y)'s slab list is `xyoffset[x][y+1] - xyoffset[x][y]`.
//!
//! This module preserves `xoffset` and `xyoffset` verbatim so a parsed
//! `Kvx` round-trips byte-equally. Synthesising a fresh `Kvx` from
//! voxel data (without reusing existing offset tables) is left for a
//! future stage and is out of R2.1's scope.

use core::fmt;

use crate::bytes::{Cursor, OutOfBounds};
use crate::Rgb6;

/// One run of consecutive voxels at a fixed (x, y) column.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Slab {
    /// Top z coordinate of the run (the "ztop" byte in the file format).
    pub ztop: u8,
    /// Visibility-flags byte; bits encode which of the six cube faces of
    /// the run are exposed to air. Bit 4 (`0x10`) is the "back-face
    /// suppress" flag voxlaptest's `setkvx` reads at offset +2 of each
    /// slab.
    pub vis: u8,
    /// Per-voxel palette indices. `colors.len()` is the run length
    /// ("zleng" in the file format).
    pub colors: Vec<u8>,
}

/// Parsed `.kvx` model. Round-trips byte-equally via [`parse`] +
/// [`serialize`].
#[derive(Debug, Clone)]
pub struct Kvx {
    pub xsiz: u32,
    pub ysiz: u32,
    pub zsiz: u32,
    /// Pivot point in 8.8 fixed-point voxel units (i.e. divide by 256
    /// to get fractional voxels).
    pub xpivot: u32,
    pub ypivot: u32,
    pub zpivot: u32,
    /// xoffset table, length `xsiz + 1`. Stored verbatim from the file;
    /// not interpreted by this crate beyond round-trip.
    pub xoffset: Vec<u32>,
    /// xyoffset table, dimensions `[xsiz][ysiz + 1]`. Slab list byte
    /// length for column (x, y) is `xyoffset[x][y+1] - xyoffset[x][y]`.
    pub xyoffset: Vec<Vec<u16>>,
    /// Slab lists per column. Outer index is x in `0..xsiz`, inner is y
    /// in `0..ysiz`. Inner-most `Vec<Slab>` is the column's slab list,
    /// in file order.
    pub columns: Vec<Vec<Vec<Slab>>>,
    /// 256-entry palette. Indexed by `Slab::colors[i]`.
    pub palette: [Rgb6; 256],
}

/// Errors returned by [`parse`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    /// File too small to contain even the 28-byte header + 768-byte
    /// palette.
    TooSmall { got: usize },
    /// A read of `need` bytes at offset `at` would run past the end of
    /// the buffer.
    Truncated { at: usize, need: usize },
    /// xyoffset values for column `x` are non-monotonic (would imply a
    /// negative slab list length).
    NonMonotonicOffsets { x: u32, y: u32 },
    /// A slab record's declared length runs past the end of its
    /// column's slab list.
    SlabOverrun { x: u32, y: u32 },
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::TooSmall { got } => write!(
                f,
                "kvx file too small ({got} bytes; need at least 28 byte header + 768 byte palette)"
            ),
            Self::Truncated { at, need } => {
                write!(f, "kvx truncated: need {need} bytes at offset {at}")
            }
            Self::NonMonotonicOffsets { x, y } => write!(
                f,
                "kvx column (x={x}, y={y}): xyoffset table is non-monotonic"
            ),
            Self::SlabOverrun { x, y } => write!(
                f,
                "kvx column (x={x}, y={y}): slab declared zleng overruns column byte budget"
            ),
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

const HEADER_LEN: usize = 28;
const PALETTE_LEN: usize = 768;

/// Parse a `.kvx` file's bytes into a [`Kvx`].
///
/// # Errors
///
/// Returns [`ParseError`] if `bytes` is too short for the fixed header
/// or palette, if a sequential read would run past EOF, or if a slab
/// list's recorded `xyoffset` differences are non-monotonic or imply a
/// declared zleng that overruns the column's byte budget.
pub fn parse(bytes: &[u8]) -> Result<Kvx, ParseError> {
    if bytes.len() < HEADER_LEN + PALETTE_LEN {
        return Err(ParseError::TooSmall { got: bytes.len() });
    }

    let mut cur = Cursor::new(bytes);
    let _numbytes = cur.read_u32()?;
    let xsiz = cur.read_u32()?;
    let ysiz = cur.read_u32()?;
    let zsiz = cur.read_u32()?;
    let xpivot = cur.read_u32()?;
    let ypivot = cur.read_u32()?;
    let zpivot = cur.read_u32()?;

    let xoff_len = xsiz as usize + 1;
    let mut xoffset = Vec::with_capacity(xoff_len);
    for _ in 0..xoff_len {
        xoffset.push(cur.read_u32()?);
    }

    let yoff_len = ysiz as usize + 1;
    let mut xyoffset = Vec::with_capacity(xsiz as usize);
    for _ in 0..xsiz {
        let mut row = Vec::with_capacity(yoff_len);
        for _ in 0..yoff_len {
            row.push(cur.read_u16()?);
        }
        xyoffset.push(row);
    }

    // Slab data spans from `cur.pos` to `bytes.len() - PALETTE_LEN`.
    let slab_region_start = cur.pos;
    let slab_region_end = bytes
        .len()
        .checked_sub(PALETTE_LEN)
        .ok_or(ParseError::TooSmall { got: bytes.len() })?;

    let mut columns = Vec::with_capacity(xsiz as usize);
    let mut slabs_cur = Cursor::new(&bytes[..slab_region_end]);
    slabs_cur.pos = slab_region_start;

    for x in 0..xsiz {
        let mut col = Vec::with_capacity(ysiz as usize);
        for y in 0..ysiz {
            let lo = u32::from(xyoffset[x as usize][y as usize]);
            let hi = u32::from(xyoffset[x as usize][y as usize + 1]);
            if hi < lo {
                return Err(ParseError::NonMonotonicOffsets { x, y });
            }
            let nbytes = (hi - lo) as usize;
            let mut budget = nbytes;
            let mut slabs = Vec::new();
            while budget > 0 {
                if budget < 3 {
                    return Err(ParseError::SlabOverrun { x, y });
                }
                let ztop = slabs_cur.read_u8()?;
                let zleng = slabs_cur.read_u8()? as usize;
                let vis = slabs_cur.read_u8()?;
                budget -= 3;
                if zleng > budget {
                    return Err(ParseError::SlabOverrun { x, y });
                }
                let mut colors = vec![0u8; zleng];
                for c in &mut colors {
                    *c = slabs_cur.read_u8()?;
                }
                budget -= zleng;
                slabs.push(Slab { ztop, vis, colors });
            }
            col.push(slabs);
        }
        columns.push(col);
    }

    // Palette is the last 768 bytes.
    let mut palette = [Rgb6 { r: 0, g: 0, b: 0 }; 256];
    let pal = &bytes[bytes.len() - PALETTE_LEN..];
    for (i, e) in palette.iter_mut().enumerate() {
        e.r = pal[i * 3];
        e.g = pal[i * 3 + 1];
        e.b = pal[i * 3 + 2];
    }

    Ok(Kvx {
        xsiz,
        ysiz,
        zsiz,
        xpivot,
        ypivot,
        zpivot,
        xoffset,
        xyoffset,
        columns,
        palette,
    })
}

/// Serialise a [`Kvx`] back to bytes. The output round-trips byte-
/// equally with the input that produced this `Kvx` via [`parse`].
///
/// # Panics
///
/// Panics if the encoded file size exceeds `u32::MAX` (a `.kvx` file
/// larger than 4 GiB cannot represent its `numbytes` header field) or
/// if any [`Slab::colors`] has length > 255 (the on-disk `zleng` field
/// is a single byte). Both are file-format limits, not runtime errors;
/// `Kvx` values produced by [`parse`] always satisfy them.
#[must_use]
pub fn serialize(kvx: &Kvx) -> Vec<u8> {
    // Compute slab data size first so we can fill in `numbytes`.
    let slab_bytes_total: usize = kvx
        .columns
        .iter()
        .flatten()
        .flatten()
        .map(|s| 3 + s.colors.len())
        .sum();

    let offset_table_bytes =
        kvx.xoffset.len() * 4 + kvx.xyoffset.iter().map(|row| row.len() * 2).sum::<usize>();
    let numbytes = HEADER_LEN - 4 + offset_table_bytes + slab_bytes_total;
    let numbytes_u32 = u32::try_from(numbytes).expect("kvx file >= 4 GiB is not representable");

    let mut out =
        Vec::with_capacity(HEADER_LEN + offset_table_bytes + slab_bytes_total + PALETTE_LEN);

    out.extend_from_slice(&numbytes_u32.to_le_bytes());
    out.extend_from_slice(&kvx.xsiz.to_le_bytes());
    out.extend_from_slice(&kvx.ysiz.to_le_bytes());
    out.extend_from_slice(&kvx.zsiz.to_le_bytes());
    out.extend_from_slice(&kvx.xpivot.to_le_bytes());
    out.extend_from_slice(&kvx.ypivot.to_le_bytes());
    out.extend_from_slice(&kvx.zpivot.to_le_bytes());
    for v in &kvx.xoffset {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for row in &kvx.xyoffset {
        for v in row {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    for col in &kvx.columns {
        for slabs in col {
            for s in slabs {
                // zleng (run length) is a 1-byte field, so colors.len()
                // must fit in a u8. Slabs from `parse` always satisfy
                // this; user-constructed slabs that don't are a bug.
                let zleng = u8::try_from(s.colors.len())
                    .expect("kvx slab zleng must fit in u8 (file format limit)");
                out.push(s.ztop);
                out.push(zleng);
                out.push(s.vis);
                out.extend_from_slice(&s.colors);
            }
        }
    }
    for e in &kvx.palette {
        out.push(e.r);
        out.push(e.g);
        out.push(e.b);
    }

    out
}

// --- tests --------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// `assets/coco.kvx`, the same fixture `voxlaptest`'s oracle loads
    /// for its `sprite_coco` pose.
    const COCO_KVX: &[u8] = include_bytes!("../../../assets/coco.kvx");

    #[test]
    fn parse_coco_header() {
        let kvx = parse(COCO_KVX).expect("parse coco.kvx");
        assert_eq!(kvx.xsiz, 9);
        assert_eq!(kvx.ysiz, 11);
        assert_eq!(kvx.zsiz, 9);
        assert_eq!(kvx.xpivot, 0x200);
        assert_eq!(kvx.ypivot, 0x300);
        assert_eq!(kvx.zpivot, 0x900);
        assert_eq!(kvx.columns.len(), 9);
        assert_eq!(kvx.columns[0].len(), 11);
    }

    #[test]
    fn coco_palette_widening() {
        let kvx = parse(COCO_KVX).expect("parse coco.kvx");
        // First palette entry from the hex dump is r=0x3f g=0x19 b=0x19.
        let p0 = kvx.palette[0];
        assert_eq!((p0.r, p0.g, p0.b), (0x3f, 0x19, 0x19));
        // Voxlap-style packing matches setkvx's longpal[i] for the same
        // bytes: ((0x3f << 18) | (0x19 << 10) | (0x19 << 2)) | 0x80000000.
        let want = 0x8000_0000u32 | (0x3fu32 << 18) | (0x19u32 << 10) | (0x19u32 << 2);
        assert_eq!(p0.to_voxlap_argb(), want);
    }

    #[test]
    fn coco_roundtrips_byte_equal() {
        let kvx = parse(COCO_KVX).expect("parse coco.kvx");
        let out = serialize(&kvx);
        assert_eq!(out.len(), COCO_KVX.len(), "length differs");
        assert_eq!(out.as_slice(), COCO_KVX, "byte content differs");
    }

    #[test]
    fn parse_truncated_fails() {
        let r = parse(&[0u8; 16]);
        assert!(matches!(r, Err(ParseError::TooSmall { .. })));
    }
}
