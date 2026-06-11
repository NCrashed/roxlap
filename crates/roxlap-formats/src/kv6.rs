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

impl Kv6 {
    /// Build a `Kv6` procedurally from a dense occupancy + colour
    /// closure: `fill(x, y, z)` returns `Some(col)` for a solid voxel,
    /// `None` for air. `col` is voxlap-packed `0x80RRGGBB` — the high
    /// byte is **brightness**, not alpha, so `0x00…` renders black; use
    /// `0x80…` for a flat-lit mid value.
    ///
    /// Only **surface** voxels are emitted (a voxel with at least one
    /// of its six neighbours air or out of bounds), matching how a
    /// `.kv6` stores a hull and how [`crate::sprite::Sprite`] expects to
    /// be drawn; fully-enclosed interior voxels are skipped. Emitted
    /// voxels get `vis = 63` (all faces) and `dir = 0`, mirroring
    /// `roxlap_core::meltsphere`'s flat output — adequate for procedural
    /// models that don't need per-face normals. The pivot is the
    /// geometric centre.
    ///
    /// Voxels are emitted in the canonical x-major, then y, then
    /// ascending-z order the format requires, with matching `xlen` /
    /// `ylen` run tables.
    // Dimensions are bounded by realistic model sizes: column/x counts
    // fit u16/u32, sizes fit f32 exactly, and the closure-local i64
    // neighbour coords are range-checked before the u32 cast.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    #[must_use]
    pub fn from_fn<F: Fn(u32, u32, u32) -> Option<u32>>(
        xsiz: u32,
        ysiz: u32,
        zsiz: u32,
        fill: F,
    ) -> Kv6 {
        let occupied = |x: i64, y: i64, z: i64| -> bool {
            x >= 0
                && y >= 0
                && z >= 0
                && (x as u32) < xsiz
                && (y as u32) < ysiz
                && (z as u32) < zsiz
                && fill(x as u32, y as u32, z as u32).is_some()
        };

        let mut voxels: Vec<Voxel> = Vec::new();
        let mut xlen: Vec<u32> = Vec::with_capacity(xsiz as usize);
        let mut ylen: Vec<Vec<u16>> = Vec::with_capacity(xsiz as usize);

        for x in 0..xsiz {
            let mut col_counts: Vec<u16> = Vec::with_capacity(ysiz as usize);
            for y in 0..ysiz {
                let before = voxels.len();
                for z in 0..zsiz {
                    let Some(col) = fill(x, y, z) else { continue };
                    let (xi, yi, zi) = (i64::from(x), i64::from(y), i64::from(z));
                    let exposed = !occupied(xi - 1, yi, zi)
                        || !occupied(xi + 1, yi, zi)
                        || !occupied(xi, yi - 1, zi)
                        || !occupied(xi, yi + 1, zi)
                        || !occupied(xi, yi, zi - 1)
                        || !occupied(xi, yi, zi + 1);
                    if exposed {
                        voxels.push(Voxel {
                            col,
                            z: z as u16,
                            vis: 63,
                            dir: 0,
                        });
                    }
                }
                col_counts.push((voxels.len() - before) as u16);
            }
            xlen.push(col_counts.iter().map(|&c| u32::from(c)).sum());
            ylen.push(col_counts);
        }

        Kv6 {
            xsiz,
            ysiz,
            zsiz,
            xpiv: xsiz as f32 * 0.5,
            ypiv: ysiz as f32 * 0.5,
            zpiv: zsiz as f32 * 0.5,
            voxels,
            xlen,
            ylen,
            palette: None,
        }
    }

    /// A solid axis-aligned box of a single colour (voxlap-packed
    /// `0x80RRGGBB`). Convenience over [`Kv6::from_fn`].
    #[must_use]
    pub fn solid_box(xsiz: u32, ysiz: u32, zsiz: u32, col: u32) -> Kv6 {
        Kv6::from_fn(xsiz, ysiz, zsiz, |_, _, _| Some(col))
    }

    /// A solid `n³` cube of a single colour.
    #[must_use]
    pub fn solid_cube(n: u32, col: u32) -> Kv6 {
        Kv6::solid_box(n, n, n, col)
    }
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
///
/// # Examples
///
/// Round-trip a synthetic empty kv6 through [`serialize`] + [`parse`]:
///
/// ```
/// use roxlap_formats::kv6::{self, Kv6};
///
/// let original = Kv6 {
///     xsiz: 1, ysiz: 1, zsiz: 1,
///     xpiv: 0.5, ypiv: 0.5, zpiv: 0.5,
///     voxels: vec![],
///     xlen: vec![0],
///     ylen: vec![vec![0]],
///     palette: None,
/// };
/// let bytes = kv6::serialize(&original);
/// let parsed = kv6::parse(&bytes).unwrap();
/// assert_eq!(parsed.xsiz, original.xsiz);
/// assert_eq!(parsed.voxels.len(), 0);
/// ```
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
    fn solid_cube_builder_is_surface_only_and_consistent() {
        let cube = Kv6::solid_cube(4, 0x8012_3456);
        assert_eq!((cube.xsiz, cube.ysiz, cube.zsiz), (4, 4, 4));
        // Pivot at the geometric centre.
        assert!((cube.xpiv - 2.0).abs() < f32::EPSILON);

        // Surface-only: a solid 4³ has 64 voxels, minus the 2³ interior
        // shell core (all six neighbours occupied) = 56 emitted.
        assert_eq!(cube.voxels.len(), 64 - 8);
        assert!(cube
            .voxels
            .iter()
            .all(|v| v.vis == 63 && v.col == 0x8012_3456));

        // Run tables match the format contract.
        assert_eq!(cube.xlen.len(), 4);
        assert_eq!(cube.ylen.len(), 4);
        assert!(cube.ylen.iter().all(|row| row.len() == 4));
        let xlen_sum: usize = cube.xlen.iter().map(|&n| n as usize).sum();
        let ylen_sum: usize = cube
            .ylen
            .iter()
            .flat_map(|r| r.iter())
            .map(|&n| n as usize)
            .sum();
        assert_eq!(xlen_sum, cube.voxels.len());
        assert_eq!(ylen_sum, cube.voxels.len());
    }

    #[test]
    fn built_cube_round_trips_through_serialize_parse() {
        let cube = Kv6::solid_cube(5, 0x80AB_CDEF);
        let bytes = serialize(&cube);
        let back = parse(&bytes).expect("parse built cube");
        assert_eq!(back.xsiz, cube.xsiz);
        assert_eq!(back.voxels.len(), cube.voxels.len());
        assert_eq!(
            serialize(&back),
            bytes,
            "serialize is stable across round-trip"
        );
    }

    #[test]
    fn from_fn_skips_air_and_keeps_z_order() {
        // A single occupied column at (0,0,*): two voxels (z=0,1), both
        // surface; ordered ascending z.
        let kv6 = Kv6::from_fn(1, 1, 2, |_, _, _| Some(0x8000_FF00));
        assert_eq!(kv6.voxels.len(), 2);
        assert_eq!(kv6.voxels[0].z, 0);
        assert_eq!(kv6.voxels[1].z, 1);
        assert_eq!(kv6.xlen, vec![2]);
        assert_eq!(kv6.ylen, vec![vec![2]]);
    }

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
