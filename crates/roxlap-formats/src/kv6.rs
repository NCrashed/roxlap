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
use std::collections::HashMap;

use crate::bytes::{Cursor, OutOfBounds};
use crate::color::VoxColor;
use crate::Rgb6;

// Voxlap kv6 `vis` face bits. These must match the `mask` the CPU
// sprite rasteriser ANDs `vis` with (`roxlap_core::sprite::kv6_iterate`
// / `draw_boundcube_line`), which is the same convention an authored
// `.kv6`'s `vis` uses. Derived from that mask construction and
// calibrated against `coco.kv6` (see the `coco_vis_*` tests):
//   x±/y± from the quadrant masks; z from the per-column z-run phases
//   (`z < inz` ⇒ −z face uses 0x20; `z > inz` ⇒ +z face uses 0x10).
const VIS_NEG_X: u8 = 0x01;
const VIS_POS_X: u8 = 0x02;
const VIS_NEG_Y: u8 = 0x04;
const VIS_POS_Y: u8 = 0x08;
// z bits calibrated against coco.kv6: 0x10 is the -z face, 0x20 the +z
// face (the naive draw-order reading was reversed; see the test
// `coco_vis_z_order_matches_authored`).
const VIS_POS_Z: u8 = 0x20;
const VIS_NEG_Z: u8 = 0x10;

/// Per-voxel `(vis, dir)` for a surface voxel at local `(x, y, z)`,
/// given an occupancy predicate `occ` (out-of-range ⇒ air). `vis` is
/// the exposed-face bitmask; `dir` is the nearest voxlap direction
/// ([`crate::equivec::nearest_dir`]) to the outward surface normal,
/// estimated as the gradient of occupancy over the 3³ neighbourhood
/// (summing the offsets to *empty* cells points away from the solid).
pub(crate) fn compute_vis_dir(
    occ: &impl Fn(i64, i64, i64) -> bool,
    x: i64,
    y: i64,
    z: i64,
) -> (u8, u8) {
    let mut vis = 0u8;
    if !occ(x - 1, y, z) {
        vis |= VIS_NEG_X;
    }
    if !occ(x + 1, y, z) {
        vis |= VIS_POS_X;
    }
    if !occ(x, y - 1, z) {
        vis |= VIS_NEG_Y;
    }
    if !occ(x, y + 1, z) {
        vis |= VIS_POS_Y;
    }
    if !occ(x, y, z - 1) {
        vis |= VIS_NEG_Z;
    }
    if !occ(x, y, z + 1) {
        vis |= VIS_POS_Z;
    }

    let mut n = [0.0f32; 3];
    for dz in -1..=1 {
        for dy in -1..=1 {
            for dx in -1..=1 {
                if (dx | dy | dz) != 0 && !occ(x + dx, y + dy, z + dz) {
                    n[0] += dx as f32;
                    n[1] += dy as f32;
                    n[2] += dz as f32;
                }
            }
        }
    }
    (vis, crate::equivec::nearest_dir(n))
}

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
    /// Model extent along x, in voxels (`kv6data.xsiz`).
    pub xsiz: u32,
    /// Model extent along y, in voxels (`kv6data.ysiz`).
    pub ysiz: u32,
    /// Model extent along z, in voxels (`kv6data.zsiz`). Sprites keep
    /// voxlap's z-down convention: z=0 is the model's top.
    pub zsiz: u32,
    /// Pivot x in fractional voxel units from the model's minimum
    /// corner (`kv6data.xpiv`); the point placement/rotation happens
    /// about.
    pub xpiv: f32,
    /// Pivot y, same units/origin as [`xpiv`](Self::xpiv).
    pub ypiv: f32,
    /// Pivot z, same units/origin as [`xpiv`](Self::xpiv).
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
    #[must_use]
    pub fn from_fn<F: Fn(u32, u32, u32) -> Option<VoxColor>>(
        xsiz: u32,
        ysiz: u32,
        zsiz: u32,
        fill: F,
    ) -> Kv6 {
        Self::build_inner(xsiz, ysiz, zsiz, fill, false, |_| false)
    }

    /// Like [`Kv6::from_fn`], but keeps **interior** (fully-enclosed) voxels
    /// whose colour `keep_interior` returns `true` for, instead of culling
    /// them as the surface-only default does.
    ///
    /// `from_fn` stores only voxels with at least one exposed face — correct
    /// for opaque models (you never see an enclosed voxel) and the reason a
    /// solid cube is a hollow shell. But a **`BlendMode::Volumetric`**
    /// (Beer–Lambert) volume needs its interior: the per-cell absorption that
    /// makes a filled cloud read denser at its core than its rim only works if
    /// the ray actually traverses the inner voxels (`crate::material`). This
    /// variant keeps an interior voxel when `keep_interior(colour)` is true —
    /// pass a predicate that matches your translucent/volumetric colours, so
    /// opaque interiors are still dropped (the storage win) while translucent
    /// bodies stay solid through. The kept interiors are flat (`vis = 63`,
    /// `dir = 0`); translucent voxels render flat-lit anyway.
    #[must_use]
    pub fn from_fn_keep_interior<F, G>(
        xsiz: u32,
        ysiz: u32,
        zsiz: u32,
        fill: F,
        keep_interior: G,
    ) -> Kv6
    where
        F: Fn(u32, u32, u32) -> Option<VoxColor>,
        G: Fn(VoxColor) -> bool,
    {
        Self::build_inner(xsiz, ysiz, zsiz, fill, false, keep_interior)
    }

    /// Like [`Kv6::from_fn`], but fills **real** per-voxel surface
    /// normals ([`Voxel::dir`]) and face visibility ([`Voxel::vis`])
    /// instead of the flat `dir = 0`, `vis = 63`. The CPU sprite
    /// rasteriser shades each voxel by `dir` (`kv6colmul[dir]`), so a
    /// `from_fn`-built model shades flat while a `from_fn_shaded` one
    /// gets proper directional gradient shading — the difference an
    /// authored `.kv6` shows.
    ///
    /// `dir` is the nearest voxlap direction
    /// ([`crate::equivec::nearest_dir`]) to the voxel's outward surface
    /// normal, estimated as the occupancy gradient over the 3³
    /// neighbourhood (pointing toward empty space). `vis` is the bitmask
    /// of the six exposed faces.
    #[must_use]
    pub fn from_fn_shaded<F: Fn(u32, u32, u32) -> Option<VoxColor>>(
        xsiz: u32,
        ysiz: u32,
        zsiz: u32,
        fill: F,
    ) -> Kv6 {
        Self::build_inner(xsiz, ysiz, zsiz, fill, true, |_| false)
    }

    // Dimensions are bounded by realistic model sizes: column/x counts
    // fit u16/u32, sizes fit f32 exactly, and the closure-local i64
    // neighbour coords are range-checked before the u32 cast.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    fn build_inner<F, G>(
        xsiz: u32,
        ysiz: u32,
        zsiz: u32,
        fill: F,
        shaded: bool,
        keep_interior: G,
    ) -> Kv6
    where
        F: Fn(u32, u32, u32) -> Option<VoxColor>,
        G: Fn(VoxColor) -> bool,
    {
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
                    if exposed || keep_interior(col) {
                        let (vis, dir) = if shaded {
                            compute_vis_dir(&occupied, xi, yi, zi)
                        } else {
                            (63, 0)
                        };
                        voxels.push(Voxel {
                            col: col.0,
                            z: z as u16,
                            vis,
                            dir,
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

    /// Recompute every stored voxel's [`Voxel::vis`] + [`Voxel::dir`]
    /// from `occupied` (a predicate over the **full** solid in this
    /// kv6's local coordinates; out-of-range / air ⇒ `false`). Use this
    /// after editing a model's voxels to refresh its shading + face
    /// visibility — the editor counterpart to building with
    /// [`Kv6::from_fn_shaded`]. Geometry (positions, run tables) is left
    /// untouched; only `vis`/`dir` change.
    #[allow(clippy::cast_possible_wrap)]
    pub fn recompute_surface(&mut self, occupied: impl Fn(i32, i32, i32) -> bool) {
        let xsiz = self.xsiz;
        let ysiz = self.ysiz;
        let zsiz = self.zsiz;
        let occ = |x: i64, y: i64, z: i64| -> bool {
            x >= 0
                && y >= 0
                && z >= 0
                && (x as u32) < xsiz
                && (y as u32) < ysiz
                && (z as u32) < zsiz
                && occupied(x as i32, y as i32, z as i32)
        };
        let mut vi = 0usize;
        for x in 0..xsiz as usize {
            for y in 0..ysiz as usize {
                let len = self.ylen[x][y] as usize;
                for _ in 0..len {
                    let z = i64::from(self.voxels[vi].z);
                    let (vis, dir) = compute_vis_dir(&occ, x as i64, y as i64, z);
                    self.voxels[vi].vis = vis;
                    self.voxels[vi].dir = dir;
                    vi += 1;
                }
            }
        }
    }

    /// Map of every stored surface voxel's local `(x, y, z)` to its
    /// colour, decoded from the run tables. Used by
    /// [`Kv6::carve_sphere_with_colfunc`] to keep surviving surface
    /// voxels at their authored colour while the cut repaints only the
    /// freshly-exposed ones.
    fn surface_color_map(&self) -> HashMap<(u32, u32, u32), u32> {
        let mut map = HashMap::with_capacity(self.voxels.len());
        let mut vi = 0usize;
        for x in 0..self.xsiz as usize {
            for y in 0..self.ysiz as usize {
                let len = self.ylen[x][y] as usize;
                for _ in 0..len {
                    let v = self.voxels[vi];
                    #[allow(clippy::cast_lossless)]
                    map.insert((x as u32, y as u32, u32::from(v.z)), v.col);
                    vi += 1;
                }
            }
        }
        map
    }

    /// Carve a sphere out of this model and control the colour of the
    /// interior the cut exposes — the sprite counterpart of
    /// [`roxlap_scene::Grid::set_sphere_with_colfunc`] /
    /// [`crate::edit::set_sphere_with_colfunc`].
    ///
    /// **Why a `solid` predicate is required.** A `.kv6` stores only
    /// its *surface* hull — fully-enclosed interior voxels are not
    /// recorded (see [`Kv6::from_fn`]). A carve must therefore know the
    /// model's *full* occupancy to expose meaningful interior walls,
    /// which the data alone can't provide. The caller supplies it via
    /// `solid(x, y, z) -> bool` in kv6-local voxel coords (e.g. the
    /// same predicate used to build the model with
    /// [`Kv6::from_fn_shaded`]). `solid` must report `true` for at
    /// least every stored surface voxel.
    ///
    /// Behaviour:
    /// - Voxels inside the sphere (`dx²+dy²+dz² <= r²`, matching
    ///   [`crate::edit::set_sphere`]) become air.
    /// - Voxels the cut newly exposes get their colour from
    ///   `colfunc(x, y, z)` (kv6-local coords, voxlap-packed
    ///   `0x80RRGGBB`). Pass `|_, _, _| col` for a flat crater colour.
    /// - Voxels that were already on the surface keep their stored
    ///   colour.
    ///
    /// `centre` / `radius` are in kv6-local voxel units. Dimensions,
    /// pivot, and palette are preserved; the model is re-extracted with
    /// real per-voxel normals + face visibility (as
    /// [`Kv6::from_fn_shaded`]).
    ///
    /// [`roxlap_scene::Grid::set_sphere_with_colfunc`]: https://docs.rs/roxlap-scene
    pub fn carve_sphere_with_colfunc<S, C>(
        &mut self,
        centre: [i32; 3],
        radius: u32,
        solid: S,
        colfunc: C,
    ) where
        S: Fn(i32, i32, i32) -> bool,
        C: Fn(i32, i32, i32) -> VoxColor,
    {
        let orig = self.surface_color_map();
        // Preserve identity fields the rebuild would otherwise reset.
        let (xpiv, ypiv, zpiv) = (self.xpiv, self.ypiv, self.zpiv);
        let palette = self.palette;

        #[allow(clippy::cast_possible_wrap)]
        let r = radius as i32;
        let r_sq = r * r;
        let (cx, cy, cz) = (centre[0], centre[1], centre[2]);
        let inside = |x: i32, y: i32, z: i32| {
            let (dx, dy, dz) = (x - cx, y - cy, z - cz);
            dx * dx + dy * dy + dz * dz <= r_sq
        };

        let rebuilt = Kv6::from_fn_shaded(self.xsiz, self.ysiz, self.zsiz, |x, y, z| {
            #[allow(clippy::cast_possible_wrap)]
            let (xi, yi, zi) = (x as i32, y as i32, z as i32);
            if inside(xi, yi, zi) || !solid(xi, yi, zi) {
                return None;
            }
            // Surviving surface voxels keep their authored colour;
            // anything else solid here is freshly exposed → colfunc.
            Some(
                orig.get(&(x, y, z))
                    .copied()
                    .map_or_else(|| colfunc(xi, yi, zi), VoxColor),
            )
        });

        self.voxels = rebuilt.voxels;
        self.xlen = rebuilt.xlen;
        self.ylen = rebuilt.ylen;
        self.xpiv = xpiv;
        self.ypiv = ypiv;
        self.zpiv = zpiv;
        self.palette = palette;
    }

    /// A solid axis-aligned box of a single colour (voxlap-packed
    /// `0x80RRGGBB`). Convenience over [`Kv6::from_fn`].
    #[must_use]
    pub fn solid_box(xsiz: u32, ysiz: u32, zsiz: u32, col: VoxColor) -> Kv6 {
        Kv6::from_fn(xsiz, ysiz, zsiz, |_, _, _| Some(col))
    }

    /// A solid `n³` cube of a single colour.
    #[must_use]
    pub fn solid_cube(n: u32, col: VoxColor) -> Kv6 {
        Kv6::solid_box(n, n, n, col)
    }
}

/// Errors returned by [`parse`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    /// File too small to contain even the 32-byte header.
    TooSmall {
        /// Actual file size in bytes.
        got: usize,
    },
    /// First 4 bytes are not the `"Kvxl"` magic.
    BadMagic {
        /// The 4 bytes actually found.
        got: [u8; 4],
    },
    /// A read of `need` bytes at offset `at` would run past the end of
    /// the buffer.
    Truncated {
        /// Byte offset of the failed read.
        at: usize,
        /// Number of bytes the read required.
        need: usize,
    },
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

    // QE.6b — every capacity is clamped by the bytes actually left
    // (8 B/voxel, 4 B/xlen, 2 B/ylen), so a crafted count errors as
    // Truncated instead of allocation-bombing first.
    let mut voxels = Vec::with_capacity(cur.clamped_capacity(numvoxs as usize, 8));
    for _ in 0..numvoxs {
        let col = cur.read_u32()?;
        let z = cur.read_u16()?;
        let vis = cur.read_u8()?;
        let dir = cur.read_u8()?;
        voxels.push(Voxel { col, z, vis, dir });
    }

    let mut xlen = Vec::with_capacity(cur.clamped_capacity(xsiz as usize, 4));
    for _ in 0..xsiz {
        xlen.push(cur.read_u32()?);
    }

    let row_bytes = (ysiz as usize).saturating_mul(2);
    let mut ylen = Vec::with_capacity(cur.clamped_capacity(xsiz as usize, row_bytes));
    for _ in 0..xsiz {
        let mut row = Vec::with_capacity(cur.clamped_capacity(ysiz as usize, 2));
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
        let cube = Kv6::solid_cube(4, VoxColor(0x8012_3456));
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

    /// `from_fn_keep_interior` retains enclosed voxels whose colour the
    /// predicate accepts — the storage policy for `BlendMode::Volumetric`
    /// bodies (keep translucent interiors, still cull opaque ones).
    #[test]
    fn from_fn_keep_interior_retains_matching_interiors() {
        let col = VoxColor(0x8012_3456);
        // All-solid 4³: from_fn culls the 2³ interior (56 voxels); keeping the
        // interior gives the full 64.
        let shell = Kv6::from_fn(4, 4, 4, |_, _, _| Some(col));
        assert_eq!(shell.voxels.len(), 64 - 8, "from_fn is surface-only");

        let filled = Kv6::from_fn_keep_interior(4, 4, 4, |_, _, _| Some(col), |c| c == col);
        assert_eq!(filled.voxels.len(), 64, "keep_interior retains all 64");
        // The dead-centre voxel (1,1,1) is fully enclosed — absent in the
        // shell, present when kept.
        assert_eq!(color_at(&shell, 1, 1, 1), None);
        assert_eq!(color_at(&filled, 1, 1, 1), Some(col));

        // A predicate that rejects the colour culls interiors as usual.
        let culled = Kv6::from_fn_keep_interior(4, 4, 4, |_, _, _| Some(col), |_| false);
        assert_eq!(
            culled.voxels.len(),
            64 - 8,
            "predicate=false ⇒ surface-only"
        );
    }

    /// Decode the colour stored at local `(tx, ty, tz)`, or `None` if
    /// no surface voxel sits there.
    fn color_at(kv6: &Kv6, tx: u32, ty: u32, tz: u32) -> Option<VoxColor> {
        let mut vi = 0usize;
        for x in 0..kv6.xsiz {
            for y in 0..kv6.ysiz {
                let len = kv6.ylen[x as usize][y as usize] as usize;
                for _ in 0..len {
                    let v = kv6.voxels[vi];
                    if x == tx && y == ty && u32::from(v.z) == tz {
                        return Some(VoxColor(v.col));
                    }
                    vi += 1;
                }
            }
        }
        None
    }

    #[test]
    fn carve_sphere_exposes_interior_with_colfunc() {
        const BASE: VoxColor = VoxColor(0x8011_2233);
        // A full solid 16³ cube — its occupancy predicate is "all".
        let mut cube = Kv6::from_fn_shaded(16, 16, 16, |_, _, _| Some(BASE));
        // Custom pivot to verify carve preserves it.
        cube.xpiv = 1.0;
        cube.ypiv = 2.0;
        cube.zpiv = 3.0;

        // colfunc encodes kv6-local coords into the low 24 bits so we
        // can assert the closure sees local (not some shifted) coords.
        let encode = |x: i32, y: i32, z: i32| VoxColor(((x << 16) | (y << 8) | z) as u32);
        cube.carve_sphere_with_colfunc([8, 8, 8], 4, |_, _, _| true, encode);

        // Sphere centre removed.
        assert_eq!(color_at(&cube, 8, 8, 8), None);
        // (8,8,3) sits just below the carved range (its +z neighbour
        // (8,8,4) is carved): solid, freshly exposed → colfunc colour
        // at its LOCAL coords.
        assert_eq!(color_at(&cube, 8, 8, 3), Some(encode(8, 8, 3)));
        // An original face voxel far from the cut keeps its colour.
        assert_eq!(color_at(&cube, 0, 8, 8), Some(BASE));

        // Pivot preserved across the rebuild.
        assert!((cube.xpiv - 1.0).abs() < f32::EPSILON);
        assert!((cube.ypiv - 2.0).abs() < f32::EPSILON);
        assert!((cube.zpiv - 3.0).abs() < f32::EPSILON);

        // Run tables stay consistent with the voxel list.
        let xlen_sum: usize = cube.xlen.iter().map(|&n| n as usize).sum();
        assert_eq!(xlen_sum, cube.voxels.len());
    }

    #[test]
    fn carve_sphere_respects_caller_solid_predicate() {
        const BASE: VoxColor = VoxColor(0x80AA_BBCC);
        // Build from a half-solid predicate (only x < 8 solid), and
        // pass the SAME predicate as `solid`. A carve centred in the
        // solid half must not resurrect the air half.
        let solid = |x: i32, _y: i32, _z: i32| (0..8).contains(&x);
        #[allow(clippy::cast_sign_loss)]
        let mut m =
            Kv6::from_fn_shaded(16, 16, 16, |x, _, _| solid(x as i32, 0, 0).then_some(BASE));
        m.carve_sphere_with_colfunc([4, 8, 8], 3, solid, |_, _, _| VoxColor(0x8000_FF00));
        // Air half stays air (never solid → never emitted).
        assert_eq!(color_at(&m, 12, 8, 8), None);
        // Carved centre gone.
        assert_eq!(color_at(&m, 4, 8, 8), None);
    }

    #[test]
    fn built_cube_round_trips_through_serialize_parse() {
        let cube = Kv6::solid_cube(5, VoxColor(0x80AB_CDEF));
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
        let kv6 = Kv6::from_fn(1, 1, 2, |_, _, _| Some(VoxColor(0x8000_FF00)));
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
    fn from_fn_shaded_keeps_from_fn_geometry() {
        // Shading must not change which voxels are emitted or the run
        // tables — only vis/dir. (A hollow shell: surface of a 5³ cube.)
        let fill = |x: u32, y: u32, z: u32| {
            let on_face = x == 0 || x == 4 || y == 0 || y == 4 || z == 0 || z == 4;
            on_face.then_some(VoxColor(0x80_44_55_66))
        };
        let flat = Kv6::from_fn(5, 5, 5, fill);
        let shaded = Kv6::from_fn_shaded(5, 5, 5, fill);
        assert_eq!(flat.voxels.len(), shaded.voxels.len());
        assert_eq!(flat.xlen, shaded.xlen);
        assert_eq!(flat.ylen, shaded.ylen);
        for (f, s) in flat.voxels.iter().zip(&shaded.voxels) {
            assert_eq!((f.col, f.z), (s.col, s.z));
        }
        // And shading actually varied dir (not all 0 like from_fn).
        assert!(
            shaded.voxels.iter().any(|v| v.dir != 0),
            "from_fn_shaded left every dir flat"
        );
        assert!(flat.voxels.iter().all(|v| v.dir == 0 && v.vis == 63));
    }

    #[test]
    fn from_fn_shaded_column_z_faces() {
        // A 1×1×2 stack: lower voxel's +z face and upper's -z face are
        // internal (the two touch); the four side faces + the outer z
        // face are exposed. Validates the VIS_*_Z constants' internal
        // consistency against the neighbour checks.
        let kv = Kv6::from_fn_shaded(1, 1, 2, |_, _, _| Some(VoxColor(0x80_80_80_80)));
        assert_eq!(kv.voxels.len(), 2);
        let (lower, upper) = (&kv.voxels[0], &kv.voxels[1]); // ascending z
        assert_eq!(lower.z, 0);
        assert_eq!(upper.z, 1);
        assert_eq!(lower.vis & VIS_POS_Z, 0, "lower +z should be internal");
        assert_eq!(lower.vis & VIS_NEG_Z, VIS_NEG_Z, "lower -z exposed");
        assert_eq!(upper.vis & VIS_NEG_Z, 0, "upper -z should be internal");
        assert_eq!(upper.vis & VIS_POS_Z, VIS_POS_Z, "upper +z exposed");
        // All four side faces exposed on both.
        let sides = VIS_NEG_X | VIS_POS_X | VIS_NEG_Y | VIS_POS_Y;
        assert_eq!(lower.vis & sides, sides);
        assert_eq!(upper.vis & sides, sides);
    }

    /// CALIBRATION: confirm every `vis` face bit matches voxlap's
    /// authored convention, against `coco.kv6`. When two voxels are
    /// stored adjacent along an axis, the face between them is internal,
    /// so the corresponding bit must be clear in the authored `vis` —
    /// interior-independent (the shared face is internal regardless of
    /// any unstored solid), so it pins all six bits without needing
    /// coco's full solid. (We don't assert the converse: a missing
    /// stored neighbour may still be solid interior, leaving the bit
    /// legitimately clear.)
    #[test]
    fn coco_vis_matches_authored_all_faces() {
        use std::collections::HashMap;
        let kv6 = parse(COCO_KV6).expect("parse coco.kv6");
        let mut pos: HashMap<(u32, u32, u32), u8> = HashMap::new();
        let mut vi = 0usize;
        for x in 0..kv6.xsiz {
            for y in 0..kv6.ysiz {
                let len = kv6.ylen[x as usize][y as usize] as usize;
                for _ in 0..len {
                    pos.insert((x, y, u32::from(kv6.voxels[vi].z)), kv6.voxels[vi].vis);
                    vi += 1;
                }
            }
        }
        let mut checked = 0u32;
        for (&(x, y, z), &vis) in &pos {
            let mut chk = |present: bool, bit: u8, face: &str| {
                if present {
                    assert_eq!(
                        vis & bit,
                        0,
                        "coco ({x},{y},{z}): {face} internal but bit set"
                    );
                    checked += 1;
                }
            };
            chk(pos.contains_key(&(x + 1, y, z)), VIS_POS_X, "+x");
            chk(x > 0 && pos.contains_key(&(x - 1, y, z)), VIS_NEG_X, "-x");
            chk(pos.contains_key(&(x, y + 1, z)), VIS_POS_Y, "+y");
            chk(y > 0 && pos.contains_key(&(x, y - 1, z)), VIS_NEG_Y, "-y");
            chk(pos.contains_key(&(x, y, z + 1)), VIS_POS_Z, "+z");
            chk(z > 0 && pos.contains_key(&(x, y, z - 1)), VIS_NEG_Z, "-z");
        }
        assert!(
            checked > 100,
            "expected many adjacent faces in coco, got {checked}"
        );
    }

    #[test]
    fn recompute_surface_matches_from_fn_shaded() {
        // recompute_surface on a flat-built model must reproduce exactly
        // what from_fn_shaded would have emitted (same vis + dir).
        let fill = |x: u32, y: u32, z: u32| {
            let cx = x as f32 - 4.0;
            let cy = y as f32 - 4.0;
            let cz = z as f32 - 4.0;
            (cx * cx + cy * cy + cz * cz <= 16.0).then_some(VoxColor(0x80_30_60_90))
        };
        let shaded = Kv6::from_fn_shaded(9, 9, 9, fill);
        let mut edited = Kv6::from_fn(9, 9, 9, fill); // flat vis/dir
        edited.recompute_surface(|x, y, z| {
            x >= 0 && y >= 0 && z >= 0 && fill(x as u32, y as u32, z as u32).is_some()
        });
        assert_eq!(edited.voxels.len(), shaded.voxels.len());
        for (e, s) in edited.voxels.iter().zip(&shaded.voxels) {
            assert_eq!((e.vis, e.dir), (s.vis, s.dir), "voxel z={}", e.z);
        }
    }

    #[test]
    fn from_fn_shaded_slab_top_normal_points_up() {
        use crate::equivec::univec;
        // A solid slab filling z in [2,9]; the z=2 surface (smallest z =
        // "up" in voxlap z-down) faces empty above, so its outward normal
        // points toward -z. Check an interior-of-face voxel.
        let kv = Kv6::from_fn_shaded(8, 8, 12, |_, _, z| {
            (2..=9).contains(&z).then_some(VoxColor(0x80_aa_aa_aa))
        });
        let v = kv
            .voxels
            .iter()
            .enumerate()
            .find_map(|(i, v)| {
                // recover (x,y) for voxel i
                let mut acc = 0usize;
                for x in 0..kv.xsiz as usize {
                    for y in 0..kv.ysiz as usize {
                        let len = kv.ylen[x][y] as usize;
                        if i < acc + len {
                            return (x == 4 && y == 4 && v.z == 2).then_some(*v);
                        }
                        acc += len;
                    }
                }
                None
            })
            .expect("centre top-face voxel present");
        let n = univec()[v.dir as usize];
        assert!(
            n[2] < -0.5,
            "top-face normal should point -z (up), got {n:?}"
        );
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

    /// QE.6b — a 36-byte file claiming `u32::MAX` voxels (≈32 GiB of
    /// records) must fail as `Truncated`, not allocation-bomb the
    /// process on `Vec::with_capacity` first.
    #[test]
    fn parse_survives_absurd_numvoxs_without_alloc_bomb() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"Kvxl");
        for dim in [1u32, 1, 1] {
            bytes.extend_from_slice(&dim.to_le_bytes());
        }
        for piv in [0f32, 0.0, 0.0] {
            bytes.extend_from_slice(&piv.to_le_bytes());
        }
        bytes.extend_from_slice(&u32::MAX.to_le_bytes()); // numvoxs
        assert!(matches!(parse(&bytes), Err(ParseError::Truncated { .. })));
    }
}
