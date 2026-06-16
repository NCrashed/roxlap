//! `.rkc` rigged-character container — the on-disk form of a complete
//! animated voxel character (meshes + skeleton + clips).
//!
//! Where [`kfa`](crate::kfa) stores only a skeleton, one clip, and a
//! *single* kv6 filename, this container stores a whole character: N
//! per-bone KV6 meshes, the hinge skeleton, and any number of named
//! animation clips. It is the build target of the **demiurg** editor and
//! the load source for **monada** at runtime; both sides go through
//! [`parse`] / [`serialize`], and [`Character::to_kfa_sprite`] turns a
//! parsed character into a renderable [`KfaSprite`].
//!
//! # Layout
//!
//! Little-endian throughout (matching [`kv6`](crate::kv6) /
//! [`kfa`](crate::kfa)). A fixed header, then a flat list of **chunks**;
//! a reader indexes by `tag` and skips unknown tags via `len`. This chunk
//! list is the whole forward-compatibility story — any future content
//! type is a new tag (top-level chunk), a new clip `kind`, or a new
//! `mesh_kind`.
//!
//! ```text
//! File:
//!     magic   [u8;4]   = b"RKCH"
//!     version u16      = 1
//!     chunks  [Chunk]  until EOF
//!
//! Chunk:
//!     tag     [u8;4]
//!     len     u32              # payload byte length
//!     payload [u8; len]
//! ```
//!
//! The canonical writer emits `META`, `MSHS`, `BONS`, `CLPS` in that
//! order, followed by any preserved unknown chunks; a reader must accept
//! any order and ignore unknown tags. See the module-level handoff doc
//! (`docs/handoff-character-container.md`) for the full byte spec and the
//! forward-compat rationale.

use core::fmt;

use crate::bytes::{Cursor, OutOfBounds};
use crate::kfa::{Hinge, KfaSprite, Point3, Seq};
use crate::kv6::{self, Kv6};
use crate::sprite::Sprite;

const MAGIC: [u8; 4] = *b"RKCH";
const VERSION: u16 = 1;

const TAG_META: [u8; 4] = *b"META";
const TAG_MSHS: [u8; 4] = *b"MSHS";
const TAG_BONS: [u8; 4] = *b"BONS";
const TAG_CLPS: [u8; 4] = *b"CLPS";

const HINGE_SIZE: usize = 64;

/// `mesh_kind` discriminant for a static KV6 mesh (index into `MSHS`).
/// The only kind any current build can render. Future kinds (e.g. a
/// voxel-video source) get their own value and are added as new
/// [`MeshRef`] variants.
const MESH_KIND_STATIC: u16 = 0;

/// `kind` discriminant for a skeletal animation clip. Future clip types
/// (e.g. voxel-video) get their own value; an old reader keeps them as
/// [`ClipData::Unknown`].
const CLIP_KIND_SKELETAL: u16 = 0;

/// A parsed rigged-character container.
///
/// Index conventions that the byte layout depends on:
/// - `meshes` is indexed by `mesh_id` ([`MeshRef::Static`]).
/// - `bones` index *is* the canonical bone index — the column index of
///   every clip's `frmval` and of [`KfaSprite::kfaval`], and the target
///   of every [`Hinge::parent`].
#[derive(Debug, Clone)]
pub struct Character {
    /// Human-facing name (`META`). May be empty.
    pub name: String,
    /// World placement passed to [`KfaSprite::new`] (`META`).
    pub root: [f32; 3],
    /// Bone meshes (`MSHS`), indexed by `mesh_id`.
    pub meshes: Vec<Kv6>,
    /// The skeleton (`BONS`); index = canonical bone index.
    pub bones: Vec<Bone>,
    /// Named animation clips (`CLPS`). May be empty (a posable rig with
    /// no baked animation).
    pub clips: Vec<Clip>,
    /// Unknown top-level chunks preserved verbatim so re-saving with an
    /// older build doesn't strip newer data. Re-emitted after the known
    /// chunks in encounter order. Empty for canonically-written files.
    pub extra_chunks: Vec<([u8; 4], Vec<u8>)>,
}

/// One bone: a name, a typed mesh reference, and its hinge.
#[derive(Debug, Clone)]
pub struct Bone {
    pub name: String,
    /// Typed reference to the bone's mesh source (see [`MeshRef`]).
    pub mesh: MeshRef,
    /// Packed hinge, reused from [`kfa`](crate::kfa). `hinge.parent`
    /// references bone indices in [`Character::bones`]; `-1` = root.
    pub hinge: Hinge,
}

/// Typed reference to a bone's mesh source. The on-disk `mesh_kind`
/// discriminant has one defined value today; new variants map to new
/// discriminants without changing the `Bone` layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshRef {
    /// `mesh_kind 0` — index into [`Character::meshes`].
    Static(usize),
    // future: Video(usize), … — a new variant, a new mesh_kind on disk
}

/// One named animation clip.
#[derive(Debug, Clone)]
pub struct Clip {
    pub name: String,
    pub data: ClipData,
}

/// A clip's payload, discriminated by the on-disk `kind`.
#[derive(Debug, Clone, PartialEq)]
pub enum ClipData {
    /// `kind 0` — the `frmval` + `seq` pair, byte-identical to
    /// [`Kfa::frmval`](crate::kfa::Kfa::frmval) /
    /// [`Kfa::seq`](crate::kfa::Kfa::seq). `frmval[frame][bone]`, Q15
    /// angles; the inner length equals [`Character::bones`]`.len()`.
    Skeletal {
        frmval: Vec<Vec<i16>>,
        seq: Vec<Seq>,
    },
    /// A clip `kind` this build doesn't model — preserved verbatim so it
    /// survives a load/save cycle.
    Unknown { kind: u16, bytes: Vec<u8> },
}

/// Errors returned by [`parse`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    /// First 4 bytes are not the `b"RKCH"` magic.
    BadMagic { got: [u8; 4] },
    /// `version` field is not one this build understands.
    UnsupportedVersion(u16),
    /// A sequential read ran past EOF.
    Truncated { at: usize, need: usize },
    /// A bone references a `mesh_kind` this build can't render. Hard
    /// error — you can't render what you can't decode.
    UnsupportedMeshKind(u16),
    /// A `Skeletal` clip's `numhin` does not equal `bones.len()`.
    ClipBoneCountMismatch,
    /// A required chunk (`META` / `MSHS` / `BONS`) was absent.
    MissingChunk([u8; 4]),
    /// An embedded `MSHS` mesh blob failed to parse as a KV6.
    BadMesh(kv6::ParseError),
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadMagic { got } => {
                write!(
                    f,
                    "character bad magic: got {got:02x?}, expected {MAGIC:02x?}"
                )
            }
            Self::UnsupportedVersion(v) => {
                write!(
                    f,
                    "character unsupported version {v} (this build reads {VERSION})"
                )
            }
            Self::Truncated { at, need } => {
                write!(f, "character truncated: need {need} bytes at offset {at}")
            }
            Self::UnsupportedMeshKind(k) => {
                write!(f, "character bone references unsupported mesh_kind {k}")
            }
            Self::ClipBoneCountMismatch => {
                write!(f, "character skeletal clip numhin != bones.len()")
            }
            Self::MissingChunk(tag) => {
                write!(
                    f,
                    "character missing required chunk {:?}",
                    String::from_utf8_lossy(tag)
                )
            }
            Self::BadMesh(e) => write!(f, "character embedded kv6 mesh: {e}"),
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

/// Parse an `.rkc` file's bytes into a [`Character`].
///
/// Accepts chunks in any order and skips unknown top-level tags (kept in
/// [`Character::extra_chunks`]). `META` / `MSHS` / `BONS` are required.
///
/// # Errors
///
/// Returns [`ParseError`] on a bad magic / unsupported version, a read
/// past EOF, an unsupported `mesh_kind`, a skeletal clip whose bone count
/// disagrees with `BONS`, a missing required chunk, or a malformed
/// embedded KV6 mesh. Note: `Hinge::htype` is *not* validated here — the
/// renderer only implements `htype == 0` (others are treated as no
/// rotation), so non-zero values load but won't animate.
pub fn parse(bytes: &[u8]) -> Result<Character, ParseError> {
    let mut cur = Cursor::new(bytes);
    let magic = cur.read_bytes(4)?;
    if magic != MAGIC {
        return Err(ParseError::BadMagic {
            got: [magic[0], magic[1], magic[2], magic[3]],
        });
    }
    let version = cur.read_u16()?;
    if version != VERSION {
        return Err(ParseError::UnsupportedVersion(version));
    }

    // Pass 1: split the chunk list. Keep the raw payload of each known
    // chunk and stash unknown chunks for verbatim re-emit. Last
    // occurrence of a known tag wins (the canonical writer emits each
    // once).
    let mut meta: Option<&[u8]> = None;
    let mut mshs: Option<&[u8]> = None;
    let mut bons: Option<&[u8]> = None;
    let mut clps: Option<&[u8]> = None;
    let mut extra_chunks = Vec::new();
    while cur.remaining() > 0 {
        let tag_buf = cur.read_bytes(4)?;
        let tag = [tag_buf[0], tag_buf[1], tag_buf[2], tag_buf[3]];
        let len = cur.read_u32()? as usize;
        let payload = cur.read_bytes(len)?;
        match tag {
            TAG_META => meta = Some(payload),
            TAG_MSHS => mshs = Some(payload),
            TAG_BONS => bons = Some(payload),
            TAG_CLPS => clps = Some(payload),
            _ => extra_chunks.push((tag, payload.to_vec())),
        }
    }

    let meta = meta.ok_or(ParseError::MissingChunk(TAG_META))?;
    let mshs = mshs.ok_or(ParseError::MissingChunk(TAG_MSHS))?;
    let bons = bons.ok_or(ParseError::MissingChunk(TAG_BONS))?;

    let (name, root) = parse_meta(meta)?;
    let meshes = parse_mshs(mshs)?;
    let bones = parse_bons(bons)?;
    let clips = match clps {
        Some(p) => parse_clps(p, bones.len())?,
        None => Vec::new(),
    };

    Ok(Character {
        name,
        root,
        meshes,
        bones,
        clips,
        extra_chunks,
    })
}

/// Serialise a [`Character`] back to bytes. Round-trips byte-equally
/// with the canonical output that produced this `Character` via
/// [`parse`].
///
/// # Panics
///
/// Panics if any count (mesh / bone / clip / frame / seq) or byte length
/// does not fit in a `u32`, or if a `Skeletal` clip's `frmval` is not
/// rectangular with row length `bones.len()`. `Character` values produced
/// by [`parse`] always satisfy these.
#[must_use]
pub fn serialize(c: &Character) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());

    write_chunk(&mut out, TAG_META, |b| write_meta(b, c));
    write_chunk(&mut out, TAG_MSHS, |b| write_mshs(b, c));
    write_chunk(&mut out, TAG_BONS, |b| write_bons(b, c));
    write_chunk(&mut out, TAG_CLPS, |b| write_clps(b, c));

    for (tag, payload) in &c.extra_chunks {
        write_chunk(&mut out, *tag, |b| b.extend_from_slice(payload));
    }

    out
}

impl Character {
    /// Build a renderable [`KfaSprite`]. `clip` selects a `Skeletal`
    /// clip to bake in via [`KfaSprite::set_animation`]; `None` leaves
    /// the sprite in its rest pose for the host to drive `kfaval`
    /// directly. A non-`Skeletal` (`Unknown`) clip selection is ignored
    /// (no animation attached).
    ///
    /// Meshes are cloned per call. Editors that re-pose every frame
    /// should build the sprite once and reuse it.
    ///
    /// # Panics
    ///
    /// Panics if `clip` is out of range, or if a bone's `MeshRef::Static`
    /// index is out of range for [`Character::meshes`]. `Character`
    /// values from [`parse`] keep mesh indices in range.
    #[must_use]
    pub fn to_kfa_sprite(&self, clip: Option<usize>) -> KfaSprite {
        let limbs = self
            .bones
            .iter()
            .map(|b| match b.mesh {
                MeshRef::Static(i) => Sprite::axis_aligned(self.meshes[i].clone(), self.root),
            })
            .collect();
        let hinges = self.bones.iter().map(|b| b.hinge).collect();
        let mut k = KfaSprite::new(limbs, hinges, self.root);
        if let Some(ci) = clip {
            if let ClipData::Skeletal { frmval, seq } = &self.clips[ci].data {
                k.set_animation(frmval.clone(), seq.clone());
            }
        }
        k
    }
}

// --- chunk payload readers ----------------------------------------------

fn parse_meta(payload: &[u8]) -> Result<(String, [f32; 3]), ParseError> {
    let mut cur = Cursor::new(payload);
    let name_len = cur.read_u16()? as usize;
    let name = String::from_utf8_lossy(cur.read_bytes(name_len)?).into_owned();
    let root = [cur.read_f32()?, cur.read_f32()?, cur.read_f32()?];
    Ok((name, root))
}

fn parse_mshs(payload: &[u8]) -> Result<Vec<Kv6>, ParseError> {
    let mut cur = Cursor::new(payload);
    let count = cur.read_u32()? as usize;
    let mut meshes = Vec::with_capacity(count);
    for _ in 0..count {
        let blob_len = cur.read_u32()? as usize;
        let blob = cur.read_bytes(blob_len)?;
        meshes.push(kv6::parse(blob).map_err(ParseError::BadMesh)?);
    }
    Ok(meshes)
}

fn parse_bons(payload: &[u8]) -> Result<Vec<Bone>, ParseError> {
    let mut cur = Cursor::new(payload);
    let count = cur.read_u32()? as usize;
    let mut bones = Vec::with_capacity(count);
    for _ in 0..count {
        let name_len = cur.read_u16()? as usize;
        let name = String::from_utf8_lossy(cur.read_bytes(name_len)?).into_owned();
        let mesh_kind = cur.read_u16()?;
        let mesh_id = cur.read_u32()? as usize;
        let hinge = read_hinge(&mut cur)?;
        let mesh = match mesh_kind {
            MESH_KIND_STATIC => MeshRef::Static(mesh_id),
            other => return Err(ParseError::UnsupportedMeshKind(other)),
        };
        bones.push(Bone { name, mesh, hinge });
    }
    Ok(bones)
}

fn parse_clps(payload: &[u8], numbone: usize) -> Result<Vec<Clip>, ParseError> {
    let mut cur = Cursor::new(payload);
    let count = cur.read_u32()? as usize;
    let mut clips = Vec::with_capacity(count);
    for _ in 0..count {
        let name_len = cur.read_u16()? as usize;
        let name = String::from_utf8_lossy(cur.read_bytes(name_len)?).into_owned();
        let kind = cur.read_u16()?;
        let payload_len = cur.read_u32()? as usize;
        let body = cur.read_bytes(payload_len)?;
        let data = if kind == CLIP_KIND_SKELETAL {
            parse_skeletal(body, numbone)?
        } else {
            ClipData::Unknown {
                kind,
                bytes: body.to_vec(),
            }
        };
        clips.push(Clip { name, data });
    }
    Ok(clips)
}

fn parse_skeletal(body: &[u8], numbone: usize) -> Result<ClipData, ParseError> {
    let mut cur = Cursor::new(body);
    let numfrm = cur.read_u32()? as usize;
    let numhin = cur.read_u32()? as usize;
    if numhin != numbone {
        return Err(ParseError::ClipBoneCountMismatch);
    }
    let mut frmval = Vec::with_capacity(numfrm);
    for _ in 0..numfrm {
        let mut row = Vec::with_capacity(numhin);
        for _ in 0..numhin {
            row.push(cur.read_i16()?);
        }
        frmval.push(row);
    }
    let seqcount = cur.read_u32()? as usize;
    let mut seq = Vec::with_capacity(seqcount);
    for _ in 0..seqcount {
        let tim = cur.read_i32()?;
        let frm = cur.read_i32()?;
        seq.push(Seq { tim, frm });
    }
    Ok(ClipData::Skeletal { frmval, seq })
}

fn read_hinge(cur: &mut Cursor<'_>) -> Result<Hinge, OutOfBounds> {
    let parent = cur.read_i32()?;
    let p0 = read_point3(cur)?;
    let p1 = read_point3(cur)?;
    let v0 = read_point3(cur)?;
    let v1 = read_point3(cur)?;
    let vmin = cur.read_i16()?;
    let vmax = cur.read_i16()?;
    let htype = cur.read_u8()?;
    let filler_buf = cur.read_bytes(7)?;
    let mut filler = [0u8; 7];
    filler.copy_from_slice(filler_buf);
    Ok(Hinge {
        parent,
        p: [p0, p1],
        v: [v0, v1],
        vmin,
        vmax,
        htype,
        filler,
    })
}

fn read_point3(cur: &mut Cursor<'_>) -> Result<Point3, OutOfBounds> {
    Ok(Point3 {
        x: cur.read_f32()?,
        y: cur.read_f32()?,
        z: cur.read_f32()?,
    })
}

// --- chunk payload writers ----------------------------------------------

/// Emit `tag` + a `u32` length + the payload produced by `body`.
fn write_chunk(out: &mut Vec<u8>, tag: [u8; 4], body: impl FnOnce(&mut Vec<u8>)) {
    out.extend_from_slice(&tag);
    let len_pos = out.len();
    out.extend_from_slice(&0u32.to_le_bytes()); // placeholder
    let start = out.len();
    body(out);
    let len = u32::try_from(out.len() - start).expect("chunk payload length must fit in u32");
    out[len_pos..len_pos + 4].copy_from_slice(&len.to_le_bytes());
}

fn write_meta(out: &mut Vec<u8>, c: &Character) {
    let name = c.name.as_bytes();
    let name_len = u16::try_from(name.len()).expect("character name length must fit in u16");
    out.extend_from_slice(&name_len.to_le_bytes());
    out.extend_from_slice(name);
    for v in c.root {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

fn write_mshs(out: &mut Vec<u8>, c: &Character) {
    let count = u32::try_from(c.meshes.len()).expect("mesh count must fit in u32");
    out.extend_from_slice(&count.to_le_bytes());
    for mesh in &c.meshes {
        let blob = kv6::serialize(mesh);
        let blob_len = u32::try_from(blob.len()).expect("kv6 blob length must fit in u32");
        out.extend_from_slice(&blob_len.to_le_bytes());
        out.extend_from_slice(&blob);
    }
}

fn write_bons(out: &mut Vec<u8>, c: &Character) {
    let count = u32::try_from(c.bones.len()).expect("bone count must fit in u32");
    out.extend_from_slice(&count.to_le_bytes());
    for bone in &c.bones {
        let name = bone.name.as_bytes();
        let name_len = u16::try_from(name.len()).expect("bone name length must fit in u16");
        out.extend_from_slice(&name_len.to_le_bytes());
        out.extend_from_slice(name);
        let (kind, mesh_id) = match bone.mesh {
            MeshRef::Static(i) => (
                MESH_KIND_STATIC,
                u32::try_from(i).expect("mesh_id must fit in u32"),
            ),
        };
        out.extend_from_slice(&kind.to_le_bytes());
        out.extend_from_slice(&mesh_id.to_le_bytes());
        write_hinge(out, &bone.hinge);
    }
}

fn write_clps(out: &mut Vec<u8>, c: &Character) {
    let count = u32::try_from(c.clips.len()).expect("clip count must fit in u32");
    out.extend_from_slice(&count.to_le_bytes());
    for clip in &c.clips {
        let name = clip.name.as_bytes();
        let name_len = u16::try_from(name.len()).expect("clip name length must fit in u16");
        out.extend_from_slice(&name_len.to_le_bytes());
        out.extend_from_slice(name);
        match &clip.data {
            ClipData::Skeletal { frmval, seq } => {
                out.extend_from_slice(&CLIP_KIND_SKELETAL.to_le_bytes());
                write_chunk_body(out, |b| write_skeletal(b, frmval, seq));
            }
            ClipData::Unknown { kind, bytes } => {
                out.extend_from_slice(&kind.to_le_bytes());
                write_chunk_body(out, |b| b.extend_from_slice(bytes));
            }
        }
    }
}

/// Emit a `u32` length + the payload produced by `body` (the clip
/// `payload_len` + `payload` pair). Shares the back-patch trick with
/// [`write_chunk`] but without a leading tag.
fn write_chunk_body(out: &mut Vec<u8>, body: impl FnOnce(&mut Vec<u8>)) {
    let len_pos = out.len();
    out.extend_from_slice(&0u32.to_le_bytes());
    let start = out.len();
    body(out);
    let len = u32::try_from(out.len() - start).expect("clip payload length must fit in u32");
    out[len_pos..len_pos + 4].copy_from_slice(&len.to_le_bytes());
}

fn write_skeletal(out: &mut Vec<u8>, frmval: &[Vec<i16>], seq: &[Seq]) {
    let numhin = frmval.first().map_or(0, Vec::len);
    for (i, row) in frmval.iter().enumerate() {
        assert!(
            row.len() == numhin,
            "skeletal frmval[{i}].len() = {} != numhin {numhin}",
            row.len(),
        );
    }
    let numfrm = u32::try_from(frmval.len()).expect("numfrm must fit in u32");
    let numhin_u32 = u32::try_from(numhin).expect("numhin must fit in u32");
    out.extend_from_slice(&numfrm.to_le_bytes());
    out.extend_from_slice(&numhin_u32.to_le_bytes());
    for row in frmval {
        for v in row {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    let seqcount = u32::try_from(seq.len()).expect("seqcount must fit in u32");
    out.extend_from_slice(&seqcount.to_le_bytes());
    for s in seq {
        out.extend_from_slice(&s.tim.to_le_bytes());
        out.extend_from_slice(&s.frm.to_le_bytes());
    }
}

fn write_hinge(out: &mut Vec<u8>, h: &Hinge) {
    out.extend_from_slice(&h.parent.to_le_bytes());
    for p in h.p {
        write_point3(out, p);
    }
    for v in h.v {
        write_point3(out, v);
    }
    out.extend_from_slice(&h.vmin.to_le_bytes());
    out.extend_from_slice(&h.vmax.to_le_bytes());
    out.push(h.htype);
    out.extend_from_slice(&h.filler);
}

fn write_point3(out: &mut Vec<u8>, p: Point3) {
    out.extend_from_slice(&p.x.to_le_bytes());
    out.extend_from_slice(&p.y.to_le_bytes());
    out.extend_from_slice(&p.z.to_le_bytes());
}

// Keep the on-disk hinge width pinned to the shared 64-byte layout.
const _: () = assert!(HINGE_SIZE == 64);

// --- tests --------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_kv6(fill: u32) -> Kv6 {
        // A 1×1×1 model with a single voxel — enough to round-trip.
        Kv6 {
            xsiz: 1,
            ysiz: 1,
            zsiz: 1,
            xpiv: 0.5,
            ypiv: 0.5,
            zpiv: 0.5,
            voxels: vec![kv6::Voxel {
                col: fill,
                z: 0,
                vis: 0x3f,
                dir: 0,
            }],
            xlen: vec![1],
            ylen: vec![vec![1]],
            palette: None,
        }
    }

    fn hinge(parent: i32) -> Hinge {
        let zero = Point3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        let axis = Point3 {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        };
        Hinge {
            parent,
            p: [zero, zero],
            v: [axis, axis],
            vmin: 0,
            vmax: 0,
            htype: 0,
            filler: [0; 7],
        }
    }

    fn synthetic_character() -> Character {
        Character {
            name: "anasaur".to_string(),
            root: [70.0, -75.0, 50.0],
            meshes: vec![unit_kv6(0x00ff_8040), unit_kv6(0x0010_2030)],
            bones: vec![
                Bone {
                    name: "body".to_string(),
                    mesh: MeshRef::Static(0),
                    hinge: hinge(-1),
                },
                Bone {
                    name: "arm".to_string(),
                    mesh: MeshRef::Static(1),
                    hinge: hinge(0),
                },
            ],
            clips: vec![Clip {
                name: "wave".to_string(),
                data: ClipData::Skeletal {
                    frmval: vec![vec![0, 0], vec![0, 16000], vec![0, 0], vec![0, -16000]],
                    seq: vec![
                        Seq { tim: 0, frm: 0 },
                        Seq { tim: 500, frm: 1 },
                        Seq { tim: 1000, frm: 2 },
                        Seq { tim: 1500, frm: 3 },
                        Seq { tim: 2000, frm: !0 },
                    ],
                },
            }],
            extra_chunks: Vec::new(),
        }
    }

    #[test]
    fn roundtrips_byte_equal() {
        let c = synthetic_character();
        let bytes = serialize(&c);
        let parsed = parse(&bytes).expect("parse synthetic");
        let bytes2 = serialize(&parsed);
        assert_eq!(bytes, bytes2, "byte-level round-trip failed");
        assert_eq!(parsed.name, c.name);
        assert_eq!(parsed.root, c.root);
        assert_eq!(parsed.meshes.len(), c.meshes.len());
        assert_eq!(parsed.bones.len(), c.bones.len());
        assert_eq!(parsed.bones[1].name, "arm");
        assert_eq!(parsed.bones[1].mesh, MeshRef::Static(1));
        assert_eq!(parsed.bones[1].hinge.parent, 0);
        assert_eq!(parsed.clips.len(), 1);
        assert_eq!(parsed.clips[0].data, c.clips[0].data);
    }

    #[test]
    fn to_kfa_sprite_builds_renderable() {
        let c = synthetic_character();
        let mut k = c.to_kfa_sprite(Some(0));
        assert_eq!(k.limbs.len(), 2);
        assert_eq!(k.hinges.len(), 2);
        assert_eq!(k.p, c.root);
        // Baked clip advances the child bone away from rest.
        k.animsprite(500);
        assert_ne!(k.kfaval[1], 0, "baked clip should move the arm bone");

        // Rest pose: no curve attached → animsprite is a no-op.
        let mut rest = c.to_kfa_sprite(None);
        rest.animsprite(500);
        assert_eq!(rest.kfaval[1], 0);
    }

    #[test]
    fn clips_may_be_empty() {
        let mut c = synthetic_character();
        c.clips.clear();
        let bytes = serialize(&c);
        let parsed = parse(&bytes).expect("parse");
        assert!(parsed.clips.is_empty());
        // Posable rig still builds a sprite.
        let k = parsed.to_kfa_sprite(None);
        assert_eq!(k.limbs.len(), 2);
    }

    #[test]
    fn unknown_top_level_chunk_is_skipped_and_preserved() {
        let mut bytes = serialize(&synthetic_character());
        // Append a ZZZZ chunk with a 3-byte payload.
        bytes.extend_from_slice(b"ZZZZ");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&[1, 2, 3]);
        let parsed = parse(&bytes).expect("parse with unknown chunk");
        assert_eq!(parsed.bones.len(), 2, "known chunks still parse");
        assert_eq!(parsed.extra_chunks, vec![(*b"ZZZZ", vec![1u8, 2, 3])]);
        // Re-emit preserves it.
        let bytes2 = serialize(&parsed);
        assert_eq!(
            bytes2, bytes,
            "unknown chunk preserved byte-equal on re-save"
        );
    }

    #[test]
    fn unknown_clip_kind_preserved() {
        let mut c = synthetic_character();
        c.clips.push(Clip {
            name: "mystery".to_string(),
            data: ClipData::Unknown {
                kind: 7,
                bytes: vec![9, 8, 7, 6],
            },
        });
        let bytes = serialize(&c);
        let parsed = parse(&bytes).expect("parse");
        // The skeletal clip still loads and plays.
        assert!(matches!(parsed.clips[0].data, ClipData::Skeletal { .. }));
        assert_eq!(
            parsed.clips[1].data,
            ClipData::Unknown {
                kind: 7,
                bytes: vec![9, 8, 7, 6]
            }
        );
        // And re-serialises byte-equal.
        assert_eq!(serialize(&parsed), bytes);
    }

    #[test]
    fn bad_magic_errors() {
        let mut bytes = serialize(&synthetic_character());
        bytes[0] ^= 0xff;
        assert!(matches!(parse(&bytes), Err(ParseError::BadMagic { .. })));
    }

    #[test]
    fn version_mismatch_errors() {
        let mut bytes = serialize(&synthetic_character());
        // version is the 2 bytes right after the 4-byte magic.
        bytes[4] = 0xff;
        bytes[5] = 0xff;
        assert!(matches!(
            parse(&bytes),
            Err(ParseError::UnsupportedVersion(0xffff))
        ));
    }

    #[test]
    fn truncated_errors() {
        let bytes = serialize(&synthetic_character());
        assert!(matches!(
            parse(&bytes[..bytes.len() - 4]),
            Err(ParseError::Truncated { .. })
        ));
    }

    #[test]
    fn missing_required_chunk_errors() {
        // A minimal valid header with no chunks at all.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&MAGIC);
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        assert!(matches!(
            parse(&bytes),
            Err(ParseError::MissingChunk(TAG_META))
        ));
    }

    #[test]
    fn unsupported_mesh_kind_errors() {
        let mut bytes = serialize(&synthetic_character());
        // Flip the first bone's mesh_kind to 1. The BONS payload begins:
        //   tag(4) len(4) count(4) name_len(2) name("body"=4) kind(2)...
        // Locate the BONS tag and step into its payload.
        let pos = find_tag(&bytes, *b"BONS");
        // payload: +8 to skip tag+len, +4 count, +2 name_len, +4 "body"
        let kind_off = pos + 8 + 4 + 2 + 4;
        bytes[kind_off] = 1;
        bytes[kind_off + 1] = 0;
        assert!(matches!(
            parse(&bytes),
            Err(ParseError::UnsupportedMeshKind(1))
        ));
    }

    #[test]
    fn clip_bone_count_mismatch_errors() {
        // Build a character whose skeletal clip has the wrong bone count
        // by hand-serialising bones=1 but clip numhin=2.
        let mut c = synthetic_character();
        c.bones.pop(); // now 1 bone
        c.bones[0].mesh = MeshRef::Static(0);
        // frmval rows still have width 2 → numhin 2 != bones 1.
        let bytes = serialize(&c);
        assert!(matches!(
            parse(&bytes),
            Err(ParseError::ClipBoneCountMismatch)
        ));
    }

    #[test]
    fn bad_embedded_mesh_errors() {
        let mut bytes = serialize(&synthetic_character());
        // Corrupt the first embedded kv6's magic. MSHS payload begins:
        //   tag(4) len(4) count(4) blob_len(4) <kv6 magic 4 bytes>
        let pos = find_tag(&bytes, *b"MSHS");
        let kv6_magic_off = pos + 8 + 4 + 4;
        bytes[kv6_magic_off] ^= 0xff;
        assert!(matches!(parse(&bytes), Err(ParseError::BadMesh(_))));
    }

    /// Find the byte offset of a top-level chunk `tag` in a serialised
    /// character (walks the chunk list from the 6-byte header).
    fn find_tag(bytes: &[u8], tag: [u8; 4]) -> usize {
        let mut pos = 6; // magic(4) + version(2)
        while pos + 8 <= bytes.len() {
            let here = &bytes[pos..pos + 4];
            let len = u32::from_le_bytes([
                bytes[pos + 4],
                bytes[pos + 5],
                bytes[pos + 6],
                bytes[pos + 7],
            ]) as usize;
            if here == tag {
                return pos;
            }
            pos += 8 + len;
        }
        panic!("tag {tag:?} not found");
    }
}
