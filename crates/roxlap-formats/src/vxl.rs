//! `.vxl` voxel-map format (Voxlap world / heightmap + slab columns).
//!
//! Reference: voxlaptest's `loadvxl` / `savevxl` (`voxlap5.c:3828` /
//! `:3887`) and the `vbuf` slab layout comment at
//! `voxlap5.c:75`. File layout (all multi-byte fields are little-
//! endian):
//!
//! ```text
//! offset  size                            description
//! 0x00    u32                             magic = 0x09072000
//! 0x04    u32                             xdim (must equal ydim — VSID)
//! 0x08    u32                             ydim
//! 0x0c    24 bytes                        ipo: starting position (3 × f64)
//! 0x24    24 bytes                        ist: right vector       (3 × f64)
//! 0x3c    24 bytes                        ihe: down vector        (3 × f64)
//! 0x54    24 bytes                        ifo: forward vector     (3 × f64)
//! 0x6c    variable                        column slab data (`vsid * vsid` columns)
//! ```
//!
//! Each column's slab data is a chain of slabs:
//!
//! ```text
//! slab header (4 bytes):
//!     byte 0  nextptr  — offset to next slab in dwords (== 0 for last slab)
//!     byte 1  z1       — top z of floor-colour list
//!     byte 2  z1c      — bottom z of floor-colour list MINUS 1
//!     byte 3  z0       — ceiling z (additional slabs); dummy in the first
//! followed by (per-voxel) 4-byte BGRA colour records.
//! ```
//!
//! Walker semantics, copied from `loadvxl`:
//!
//! - Non-last slab: total bytes = `nextptr * 4`. Advance by `nextptr * 4`.
//! - Last slab (`nextptr == 0`): total bytes = `4 + (z1c - z1 + 1) * 4`
//!   (header + floor colours; no ceiling colours).
//!
//! This module preserves column slab bytes verbatim in [`Vxl::data`] and
//! exposes a per-column byte-offset table in [`Vxl::column_offset`].
//! Iterating individual slabs (interpreting ceiling vs floor colour
//! lists) is left for a follow-up — the world fixture is large enough
//! that the test workload favours a flat byte representation, and the
//! engine port (R4) walks the bytes directly anyway.

use core::fmt;

use crate::bytes::{Cursor, OutOfBounds};

const MAGIC: u32 = 0x0907_2000;
const HEADER_LEN: usize = 4 + 4 + 4 + 4 * 24;

/// Parsed `.vxl` map. Round-trips byte-equally via [`parse`] +
/// [`serialize`].
#[derive(Debug, Clone)]
pub struct Vxl {
    /// Square map dimension. xdim and ydim are both equal to this in
    /// the file format (the loader rejects non-square maps).
    pub vsid: u32,
    /// Starting camera position (`dpoint3d`).
    pub ipo: [f64; 3],
    /// Right vector (`dpoint3d`).
    pub ist: [f64; 3],
    /// Down vector (`dpoint3d`).
    pub ihe: [f64; 3],
    /// Forward vector (`dpoint3d`).
    pub ifo: [f64; 3],
    /// Concatenated raw slab data for all `vsid * vsid` columns.
    pub data: Box<[u8]>,
    /// `column_offset[i]..column_offset[i + 1]` is the byte range
    /// inside [`Vxl::data`] that holds column `i`'s slab list.
    /// `column_offset.len() == vsid * vsid + 1`; the final entry is
    /// `data.len()`.
    pub column_offset: Box<[u32]>,
}

impl Vxl {
    /// Raw slab bytes for column `idx` (`idx < vsid * vsid`).
    #[must_use]
    pub fn column_data(&self, idx: usize) -> &[u8] {
        let start = self.column_offset[idx] as usize;
        let end = self.column_offset[idx + 1] as usize;
        &self.data[start..end]
    }
}

/// Errors returned by [`parse`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    /// File too small to even contain the 108-byte header.
    TooSmall { got: usize },
    /// Magic bytes are not `0x09072000`.
    BadMagic { got: u32 },
    /// xdim and ydim disagree (file format requires square maps).
    NonSquareVsid { x: u32, y: u32 },
    /// A read of `need` bytes at offset `at` would run past EOF.
    Truncated { at: usize, need: usize },
    /// While walking column `idx`'s slab chain, the cursor at offset
    /// `at` would have run past the end of the column data region.
    BadColumn { idx: u32, at: usize },
    /// File total size > `u32::MAX`. The internal `column_offset`
    /// table uses `u32` because realistic maps fit comfortably.
    FileTooLarge { got: usize },
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::TooSmall { got } => write!(
                f,
                "vxl file too small ({got} bytes; need at least 108 byte header)"
            ),
            Self::BadMagic { got } => {
                write!(f, "vxl bad magic: got {got:#010x}, expected 0x09072000")
            }
            Self::NonSquareVsid { x, y } => write!(
                f,
                "vxl non-square dimensions: xdim={x}, ydim={y} (must be equal)"
            ),
            Self::Truncated { at, need } => {
                write!(f, "vxl truncated: need {need} bytes at offset {at}")
            }
            Self::BadColumn { idx, at } => write!(
                f,
                "vxl column {idx}: slab walker overran data region at offset {at}"
            ),
            Self::FileTooLarge { got } => write!(
                f,
                "vxl file size {got} exceeds {} bytes that this parser handles",
                u32::MAX
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

/// Parse a `.vxl` file's bytes into a [`Vxl`].
///
/// # Errors
///
/// Returns [`ParseError`] if `bytes` is shorter than the 108-byte
/// header, if the magic mismatches, if xdim ≠ ydim, if the file size
/// exceeds `u32::MAX` bytes, if a header field would run past EOF, or
/// if any column's slab chain runs off the end of the data region.
///
/// # Panics
///
/// Cannot panic on valid input: `pos` is bounded by `data.len()` which
/// the [`ParseError::FileTooLarge`] gate at the top of the function
/// proves fits in `u32`. The internal `expect` calls would only fire
/// on a logic bug in the walker.
pub fn parse(bytes: &[u8]) -> Result<Vxl, ParseError> {
    if bytes.len() < HEADER_LEN {
        return Err(ParseError::TooSmall { got: bytes.len() });
    }
    if u32::try_from(bytes.len()).is_err() {
        return Err(ParseError::FileTooLarge { got: bytes.len() });
    }

    let mut cur = Cursor::new(bytes);
    let magic = cur.read_u32()?;
    if magic != MAGIC {
        return Err(ParseError::BadMagic { got: magic });
    }
    let xdim = cur.read_u32()?;
    let ydim = cur.read_u32()?;
    if xdim != ydim {
        return Err(ParseError::NonSquareVsid { x: xdim, y: ydim });
    }
    let vsid = xdim;

    let ipo = read_dpoint3d(&mut cur)?;
    let ist = read_dpoint3d(&mut cur)?;
    let ihe = read_dpoint3d(&mut cur)?;
    let ifo = read_dpoint3d(&mut cur)?;

    // Column data begins immediately after the header and runs to EOF.
    let data_start = cur.pos;
    let data: Box<[u8]> = bytes[data_start..].to_vec().into_boxed_slice();

    let n_cols = (vsid as usize) * (vsid as usize);
    let mut column_offset = Vec::with_capacity(n_cols + 1);
    let mut pos = 0usize;
    for i in 0..n_cols {
        column_offset.push(u32::try_from(pos).expect("data offset within u32"));
        loop {
            if pos + 4 > data.len() {
                return Err(ParseError::BadColumn {
                    idx: u32::try_from(i).unwrap_or(u32::MAX),
                    at: pos,
                });
            }
            let nextptr = data[pos];
            if nextptr == 0 {
                // Last slab. Length = 4 + n_floor * 4 where
                //   n_floor = max(0, z1c - z1 + 1).
                let z1 = data[pos + 1];
                let z1c = data[pos + 2];
                // n_floor = max(0, z1c - z1 + 1). Promote to i32 to
                // avoid u8 underflow when z1c == z1 - 1 (the "no floor
                // colours" case voxlap allows).
                let n_floor_signed = i32::from(z1c) - i32::from(z1) + 1;
                let n_floor = usize::try_from(n_floor_signed.max(0))
                    .expect("n_floor non-negative after .max(0)");
                let last_size = 4 + n_floor * 4;
                if pos + last_size > data.len() {
                    return Err(ParseError::BadColumn {
                        idx: u32::try_from(i).unwrap_or(u32::MAX),
                        at: pos,
                    });
                }
                pos += last_size;
                break;
            }
            let advance = usize::from(nextptr) * 4;
            // Guard against `nextptr * 4 < 4` which would loop forever.
            if advance < 4 {
                return Err(ParseError::BadColumn {
                    idx: u32::try_from(i).unwrap_or(u32::MAX),
                    at: pos,
                });
            }
            pos += advance;
        }
    }
    column_offset.push(u32::try_from(pos).expect("data offset within u32"));

    Ok(Vxl {
        vsid,
        ipo,
        ist,
        ihe,
        ifo,
        data,
        column_offset: column_offset.into_boxed_slice(),
    })
}

/// Serialise a [`Vxl`] back to bytes. Round-trips byte-equally with
/// the input that produced this `Vxl` via [`parse`].
#[must_use]
pub fn serialize(vxl: &Vxl) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER_LEN + vxl.data.len());
    out.extend_from_slice(&MAGIC.to_le_bytes());
    out.extend_from_slice(&vxl.vsid.to_le_bytes());
    out.extend_from_slice(&vxl.vsid.to_le_bytes());
    write_dpoint3d(&mut out, &vxl.ipo);
    write_dpoint3d(&mut out, &vxl.ist);
    write_dpoint3d(&mut out, &vxl.ihe);
    write_dpoint3d(&mut out, &vxl.ifo);
    out.extend_from_slice(&vxl.data);
    out
}

fn read_dpoint3d(cur: &mut Cursor<'_>) -> Result<[f64; 3], OutOfBounds> {
    let mut out = [0.0f64; 3];
    for v in &mut out {
        let buf = cur.read_bytes(8)?;
        *v = f64::from_bits(u64::from_le_bytes([
            buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
        ]));
    }
    Ok(out)
}

fn write_dpoint3d(out: &mut Vec<u8>, p: &[f64; 3]) {
    for v in p {
        out.extend_from_slice(&v.to_bits().to_le_bytes());
    }
}

// --- tests --------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::io::Read;

    use flate2::read::GzDecoder;

    use super::*;

    /// Gzipped `oracle.vxl` produced by voxlaptest's oracle binary
    /// (run with `ROXLAP_SAVE_VXL=oracle.vxl`). At VSID = 2048 the raw
    /// file is ~37 MB but compresses to ~200 KB thanks to the largely
    /// uniform solid block surrounding the 448³ playable carve.
    const ORACLE_VXL_GZ: &[u8] = include_bytes!("../../../assets/oracle.vxl.gz");

    fn decode_oracle() -> Vec<u8> {
        let mut decoder = GzDecoder::new(ORACLE_VXL_GZ);
        let mut out = Vec::with_capacity(40 * 1024 * 1024);
        decoder.read_to_end(&mut out).expect("ungzip oracle.vxl.gz");
        out
    }

    #[test]
    fn parse_oracle_header() {
        let bytes = decode_oracle();
        let vxl = parse(&bytes).expect("parse oracle.vxl");
        // voxlaptest's fork uses VSID = 2048.
        assert_eq!(vxl.vsid, 2048);
        // The placeholder camera vectors written by oracle.c match the
        // values we set in tests/oracle/oracle.c's savevxl call. Compare
        // bit patterns to dodge clippy::float_cmp — these are exact
        // integer-valued doubles so the comparison is well-defined.
        let bits = |a: [f64; 3]| a.map(f64::to_bits);
        assert_eq!(bits(vxl.ipo), bits([1024.0, 1024.0, 128.0]));
        assert_eq!(bits(vxl.ist), bits([1.0, 0.0, 0.0]));
        assert_eq!(bits(vxl.ihe), bits([0.0, 0.0, 1.0]));
        assert_eq!(bits(vxl.ifo), bits([0.0, 1.0, 0.0]));
        // 2048 * 2048 = 4_194_304 columns; column_offset has one extra entry.
        assert_eq!(vxl.column_offset.len(), 4_194_304 + 1);
    }

    #[test]
    fn oracle_columns_partition_data_exactly() {
        let bytes = decode_oracle();
        let vxl = parse(&bytes).expect("parse oracle.vxl");
        // First column starts at offset 0, last sentinel equals data.len().
        assert_eq!(vxl.column_offset[0], 0);
        assert_eq!(
            vxl.column_offset[vxl.column_offset.len() - 1] as usize,
            vxl.data.len()
        );
        // Every column has at least one 4-byte slab header.
        let n_cols = (vxl.vsid as usize) * (vxl.vsid as usize);
        let min_col_len = (0..n_cols)
            .map(|i| vxl.column_data(i).len())
            .min()
            .expect("at least one column");
        assert!(min_col_len >= 4);
    }

    #[test]
    fn oracle_solid_corner_column_has_minimal_slab() {
        let bytes = decode_oracle();
        let vxl = parse(&bytes).expect("parse oracle.vxl");
        // The carve in tests/oracle/oracle.c is at x=800..1248, y=800..1248.
        // Column (0, 0) is well outside that range — solid block all the
        // way down, so its slab list should be the minimal one-slab form.
        let col = vxl.column_data(0);
        // Last slab nextptr == 0 must occur somewhere; the simplest valid
        // column is exactly one slab (header only or header + a few colours).
        // We assert column length is small (≤ 32 bytes — much less than a
        // carved column would be).
        assert!(
            col.len() <= 32,
            "solid corner column should be tiny; got {} bytes",
            col.len()
        );
    }

    #[test]
    fn oracle_roundtrips_byte_equal() {
        let bytes = decode_oracle();
        let vxl = parse(&bytes).expect("parse oracle.vxl");
        let out = serialize(&vxl);
        assert_eq!(out.len(), bytes.len(), "length differs");
        assert_eq!(out, bytes, "byte content differs");
    }

    #[test]
    fn parse_truncated_header_fails() {
        let r = parse(&[0u8; 32]);
        assert!(matches!(r, Err(ParseError::TooSmall { .. })));
    }

    #[test]
    fn parse_bad_magic_fails() {
        let mut bad = decode_oracle();
        bad[0] ^= 0xff;
        let r = parse(&bad);
        assert!(matches!(r, Err(ParseError::BadMagic { .. })));
    }
}
