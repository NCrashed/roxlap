//! `.rkc` rigged-character container — the on-disk form of a complete
//! animated voxel character (meshes + skeleton + clips).
//!
//! Where [`kfa`](crate::kfa) stores only a skeleton, one clip, and a
//! *single* kv6 filename, this container stores a whole character: N
//! KV6 meshes, the hinge skeleton, any number of named animation clips,
//! and (v3) embedded animated voxel clips. Each bone carries a **list of
//! attachments** — static meshes and/or animated clips, each with its own
//! local offset — so e.g. a flame can hang off a hand. It is the build
//! target of the **demiurg** editor and
//! the load source for **monada** at runtime; both sides go through
//! [`parse`](crate::character::parse) / [`serialize`](crate::character::serialize),
//! and [`Character::to_kfa_sprite`](crate::character::Character::to_kfa_sprite) turns a
//! parsed character into a renderable [`KfaSprite`](crate::kfa::KfaSprite).
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
//!     version u16      = 3
//!     chunks  [Chunk]  until EOF
//!
//! Chunk:
//!     tag     [u8;4]
//!     len     u32              # payload byte length
//!     payload [u8; len]
//! ```
//!
//! The canonical writer emits `META`, `MSHS`, `BONS`, `CLPS`, `VCLP` in
//! that order, followed by any preserved unknown chunks; a reader must
//! accept any order and ignore unknown tags. See the module-level handoff doc
//! (`docs/handoff-character-container.md`) for the full byte spec and the
//! forward-compat rationale.

use core::fmt;

use crate::bytes::{Cursor, OutOfBounds};
use crate::kfa::{Hinge, Kfa, KfaSprite, Point3, Seq};
use crate::kv6::{self, Kv6};
use crate::sprite::Sprite;
use crate::voxel_clip::{self, VoxelClip};
use crate::xform::{BoneXform, Quat};

const MAGIC: [u8; 4] = *b"RKCH";
// v3 (VCL.5): a bone carries a *list* of attachments (each a static KV6
// mesh or an animated voxel clip, with a local offset + playback), and the
// character gains a `VCLP` chunk of embedded clips. v2 (single mesh per
// bone) and v1 (i16 frmval) files are rejected — regenerate them.
const VERSION: u16 = 3;

const TAG_META: [u8; 4] = *b"META";
const TAG_MSHS: [u8; 4] = *b"MSHS";
const TAG_BONS: [u8; 4] = *b"BONS";
const TAG_CLPS: [u8; 4] = *b"CLPS";
/// Embedded animated voxel clips (VCL.5), referenced by [`MeshRef::Clip`].
const TAG_VCLP: [u8; 4] = *b"VCLP";

const HINGE_SIZE: usize = 64;

/// `mesh_kind` discriminant for a static KV6 mesh (index into `MSHS`).
const MESH_KIND_STATIC: u16 = 0;
/// `mesh_kind` discriminant for an animated voxel clip (index into
/// [`Character::voxel_clips`]) — VCL.5.
const MESH_KIND_CLIP: u16 = 1;

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
    /// Embedded animated voxel clips (`VCLP`, VCL.5), indexed by
    /// [`MeshRef::Clip`]. May be empty.
    pub voxel_clips: Vec<VoxelClip>,
    /// Unknown top-level chunks preserved verbatim so re-saving with an
    /// older build doesn't strip newer data. Re-emitted after the known
    /// chunks in encounter order. Empty for canonically-written files.
    pub extra_chunks: Vec<([u8; 4], Vec<u8>)>,
}

/// One bone: a name, a list of attachments, and its hinge (VCL.5). A bone
/// may carry several attachments (static meshes and/or animated clips),
/// each positioned by its own `local_offset` relative to the bone — so a
/// flame can hang off a hand alongside the hand mesh. v2 had exactly one
/// mesh per bone; that is now a single [`MeshRef::Static`] attachment at
/// the identity offset.
#[derive(Debug, Clone)]
pub struct Bone {
    /// Human-facing bone name (e.g. `"hand_l"`). May be empty; not
    /// required to be unique.
    pub name: String,
    /// What this bone draws (see [`Attachment`]). May be empty (a pure
    /// transform bone).
    pub attachments: Vec<Attachment>,
    /// Packed hinge, reused from [`kfa`](crate::kfa). `hinge.parent`
    /// references bone indices in [`Character::bones`]; `-1` = root.
    pub hinge: Hinge,
}

/// One thing a bone draws: a mesh/clip reference, a local offset placing
/// it on the bone, and (for clips) playback parameters (VCL.5).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Attachment {
    /// The drawn source — a static KV6 mesh or an animated voxel clip.
    pub target: MeshRef,
    /// Transform applied **on top of** the bone's solved world transform,
    /// positioning/orienting/scaling this attachment relative to the bone.
    /// Identity = sit at the bone origin.
    pub local_offset: BoneXform,
    /// Playback parameters; ignored for a [`MeshRef::Static`] target.
    pub playback: ClipPlayback,
}

impl Attachment {
    /// A static mesh attachment at the identity offset — the v2-equivalent
    /// "one mesh per bone".
    #[must_use]
    pub fn static_mesh(mesh_id: usize) -> Self {
        Self {
            target: MeshRef::Static(mesh_id),
            local_offset: BoneXform::IDENTITY,
            playback: ClipPlayback::default(),
        }
    }

    /// An animated-clip attachment at the identity offset, default playback.
    #[must_use]
    pub fn clip(clip_id: usize) -> Self {
        Self {
            target: MeshRef::Clip(clip_id),
            local_offset: BoneXform::IDENTITY,
            playback: ClipPlayback::default(),
        }
    }
}

/// Playback parameters for a [`MeshRef::Clip`] attachment. The clip's own
/// [`LoopMode`](crate::voxel_clip::LoopMode) governs looping; these only
/// control rate + phase, so the same clip can run fast on one attachment
/// and slow/offset on another.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClipPlayback {
    /// Playback speed as Q8 fixed point (`256` = 1.0×). Drives how fast the
    /// clip clock advances.
    pub speed_q8: i32,
    /// Initial clock offset (ms) so several instances of one clip don't
    /// play in lockstep.
    pub start_phase_ms: u32,
}

impl Default for ClipPlayback {
    fn default() -> Self {
        Self {
            speed_q8: 256,
            start_phase_ms: 0,
        }
    }
}

/// Typed reference to what a bone attachment draws. The on-disk
/// `mesh_kind` discriminant selects the variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshRef {
    /// `mesh_kind 0` — index into [`Character::meshes`].
    Static(usize),
    /// `mesh_kind 1` — index into [`Character::voxel_clips`] (VCL.5).
    Clip(usize),
}

/// One named animation clip.
#[derive(Debug, Clone)]
pub struct Clip {
    /// Clip name the host plays it by (e.g. `"walk"`, `"idle"`). May
    /// be empty; not required to be unique.
    pub name: String,
    /// The keyframe payload (or an unknown-kind blob kept for
    /// round-trip) — see [`ClipData`].
    pub data: ClipData,
}

/// A clip's payload, discriminated by the on-disk `kind`.
#[derive(Debug, Clone, PartialEq)]
pub enum ClipData {
    /// `kind 0` — the `frmval` + `seq` pair. `frmval[frame][bone]` is the
    /// bone's local [`BoneXform`] (translation, quaternion rotation, scale);
    /// the inner length equals [`Character::bones`]`.len()`. `seq` matches
    /// [`Kfa::seq`](crate::kfa::Kfa::seq).
    Skeletal {
        /// Keyframe table: `frmval[frame][bone]` local TRS; every inner
        /// row has exactly `Character::bones.len()` entries.
        frmval: Vec<Vec<BoneXform>>,
        /// Playback sequence — `(tim, frm)` entries, same semantics as
        /// [`Kfa::seq`](crate::kfa::Kfa::seq).
        seq: Vec<Seq>,
    },
    /// A clip `kind` this build doesn't model — preserved verbatim so it
    /// survives a load/save cycle.
    Unknown {
        /// The on-disk `kind` discriminant that wasn't recognised.
        kind: u16,
        /// The clip's raw payload bytes, kept for byte-exact re-save.
        bytes: Vec<u8>,
    },
}

/// Errors returned by [`parse`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    /// First 4 bytes are not the `b"RKCH"` magic.
    BadMagic {
        /// The 4 bytes actually found.
        got: [u8; 4],
    },
    /// `version` field is not one this build understands.
    UnsupportedVersion(u16),
    /// A sequential read ran past EOF.
    Truncated {
        /// Byte offset of the failed read.
        at: usize,
        /// Number of bytes the read required.
        need: usize,
    },
    /// A bone references a `mesh_kind` this build can't render. Hard
    /// error — you can't render what you can't decode.
    UnsupportedMeshKind(u16),
    /// A `Skeletal` clip's `numhin` does not equal `bones.len()`.
    ClipBoneCountMismatch,
    /// A required chunk (`META` / `MSHS` / `BONS`) was absent.
    MissingChunk([u8; 4]),
    /// An embedded `MSHS` mesh blob failed to parse as a KV6.
    BadMesh(kv6::ParseError),
    /// An embedded `VCLP` clip blob failed to parse as a voxel clip (VCL.5).
    BadClip(voxel_clip::ParseError),
    /// QE.6b - an attachment references a mesh / voxel-clip index past
    /// the corresponding table. Pre-QE.6 this surfaced as a documented
    /// (but wrong: "parse keeps indices in range") panic later, in
    /// `to_kfa_sprite` / the attachment runtime.
    BadAttachmentIndex {
        /// The owning bone.
        bone: usize,
        /// The out-of-range index.
        index: usize,
        /// The referenced table's length (meshes or voxel clips).
        table_len: usize,
    },
    /// QE.6b - a bone hinge whose `parent` is neither `-1` nor a valid
    /// bone index, or whose parent links form a cycle (previously an
    /// OOB panic / infinite loop in the skeleton solve).
    BadSkeleton {
        /// The offending bone.
        bone: usize,
    },
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
            Self::BadClip(e) => write!(f, "character embedded voxel clip: {e:?}"),
            Self::BadAttachmentIndex {
                bone,
                index,
                table_len,
            } => write!(
                f,
                "character bone {bone}: attachment index {index} past table of {table_len}"
            ),
            Self::BadSkeleton { bone } => {
                write!(f, "character bone {bone}: parent out of range or cyclic")
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
    let mut vclp: Option<&[u8]> = None;
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
            TAG_VCLP => vclp = Some(payload),
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
    let voxel_clips = match vclp {
        Some(p) => parse_vclp(p)?,
        None => Vec::new(),
    };

    // QE.6b - cross-reference validation, so the doc contract
    // "`Character` values from `parse` keep indices in range" is
    // actually true (pre-QE.6 it was claimed but never checked:
    // `to_kfa_sprite` / the attachment runtime panicked instead).
    for (bi, bone) in bones.iter().enumerate() {
        for att in &bone.attachments {
            let (index, table_len) = match att.target {
                MeshRef::Static(i) => (i, meshes.len()),
                MeshRef::Clip(i) => (i, voxel_clips.len()),
            };
            if index >= table_len {
                return Err(ParseError::BadAttachmentIndex {
                    bone: bi,
                    index,
                    table_len,
                });
            }
        }
        // Parent range; cycles checked below over the whole set.
        if bone.hinge.parent != -1
            && usize::try_from(bone.hinge.parent).map_or(true, |p| p >= bones.len())
        {
            return Err(ParseError::BadSkeleton { bone: bi });
        }
    }
    for start in 0..bones.len() {
        let mut cur_parent = bones[start].hinge.parent;
        let mut hops = 0usize;
        while cur_parent != -1 {
            hops += 1;
            if hops > bones.len() {
                return Err(ParseError::BadSkeleton { bone: start });
            }
            cur_parent = bones[usize::try_from(cur_parent).expect("validated non-negative")]
                .hinge
                .parent;
        }
    }

    Ok(Character {
        name,
        root,
        meshes,
        bones,
        clips,
        voxel_clips,
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
    write_chunk(&mut out, TAG_VCLP, |b| write_vclp(b, c));

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
        // The hinge solver needs exactly one limb per bone, so this legacy
        // path draws each bone's **first `Static` attachment** (the
        // v2-equivalent single mesh), at the bone's transform. Extra
        // attachments, per-attachment `local_offset`s, and `Clip`
        // attachments are handled by the richer attachment runtime (VCL.6),
        // not by `KfaSprite`. A bone with no static mesh gets an empty limb
        // (draws nothing) to keep the 1:1 bone↔limb mapping.
        let limbs = self
            .bones
            .iter()
            .map(|b| {
                let kv6 = b
                    .attachments
                    .iter()
                    .find_map(|a| match a.target {
                        MeshRef::Static(i) => Some(self.meshes[i].clone()),
                        MeshRef::Clip(_) => None,
                    })
                    .unwrap_or_else(|| Kv6::from_fn(1, 1, 1, |_, _, _| None));
                Sprite::axis_aligned(kv6, self.root)
            })
            .collect();
        let hinges = self.bones.iter().map(|b| b.hinge).collect();
        let mut k = KfaSprite::new(limbs, hinges, self.root);
        if let Some(ci) = clip {
            if let ClipData::Skeletal { frmval, seq } = &self.clips[ci].data {
                // Clip frames are already per-bone TRS — hand them straight to
                // the poser.
                k.set_animation(frmval.clone(), seq.clone());
            }
        }
        k
    }

    /// Export a **lossy** voxlap-toolchain [`Kfa`] (`.kfa`): the skeleton
    /// plus one clip, referencing a single kv6 by filename.
    ///
    /// voxlap's `.kfa` is fundamentally narrower than this container — it
    /// stores the hinge skeleton and one animation, but points at just
    /// *one* kv6 file (voxlap rigs a single mesh with a per-voxel limb
    /// index, which roxlap deliberately doesn't model). So this export
    /// drops, by design:
    /// - every embedded [`Character::meshes`] mesh — only `kv6_name` (the
    ///   filename voxlap should load) is written. Export the bone meshes
    ///   separately via [`kv6::serialize`] if a tool needs them.
    /// - every clip except `clip`.
    ///
    /// `clip` selects the [`ClipData::Skeletal`] clip whose `frmval` /
    /// `seq` to bake in. `None`, an out-of-range index, or a
    /// non-`Skeletal` clip yields an empty animation table (a posable rig
    /// with no baked motion). Serialise the result with
    /// [`kfa::serialize`](crate::kfa::serialize).
    #[must_use]
    pub fn to_kfa(&self, clip: Option<usize>, kv6_name: impl Into<Vec<u8>>) -> Kfa {
        let hinges = self.bones.iter().map(|b| b.hinge).collect();
        // The `.kfa` file stores one Q15 angle per bone; collapse each TRS to
        // its rotation about the bone's hinge axis (translation / scale /
        // off-axis rotation are dropped — this export is documented as lossy).
        let (frmval, seq) = match clip.and_then(|ci| self.clips.get(ci)) {
            Some(Clip {
                data: ClipData::Skeletal { frmval, seq },
                ..
            }) => {
                let angles = frmval
                    .iter()
                    .map(|row| {
                        row.iter()
                            .enumerate()
                            .map(|(bone, x)| {
                                let v = self.bones[bone].hinge.v[0];
                                x.hinge_angle([v.x, v.y, v.z])
                            })
                            .collect()
                    })
                    .collect();
                (angles, seq.clone())
            }
            _ => (Vec::new(), Vec::new()),
        };
        Kfa {
            kv6_name: kv6_name.into(),
            hinges,
            frmval,
            seq,
        }
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
    // QE.6b - capacity clamped by remaining bytes (>= 4 B length per
    // mesh blob); a lying count fails as Truncated, not an alloc bomb.
    let mut meshes = Vec::with_capacity(cur.clamped_capacity(count, 4));
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
    // QE.6b - >= 70 B per bone (name len + attach count + 64 B hinge).
    let mut bones = Vec::with_capacity(cur.clamped_capacity(count, 70));
    for _ in 0..count {
        let name_len = cur.read_u16()? as usize;
        let name = String::from_utf8_lossy(cur.read_bytes(name_len)?).into_owned();
        let attach_count = cur.read_u32()? as usize;
        // QE.6b - an attachment record is 54 bytes.
        let mut attachments = Vec::with_capacity(cur.clamped_capacity(attach_count, 54));
        for _ in 0..attach_count {
            attachments.push(read_attachment(&mut cur)?);
        }
        let hinge = read_hinge(&mut cur)?;
        bones.push(Bone {
            name,
            attachments,
            hinge,
        });
    }
    Ok(bones)
}

/// One attachment: `mesh_kind u16`, `index u32`, `local_offset` (40-byte
/// `BoneXform`), then playback (`speed_q8 i32`, `start_phase_ms u32`).
fn read_attachment(cur: &mut Cursor<'_>) -> Result<Attachment, ParseError> {
    let mesh_kind = cur.read_u16()?;
    let index = cur.read_u32()? as usize;
    let target = match mesh_kind {
        MESH_KIND_STATIC => MeshRef::Static(index),
        MESH_KIND_CLIP => MeshRef::Clip(index),
        other => return Err(ParseError::UnsupportedMeshKind(other)),
    };
    let local_offset = read_bonexform(cur)?;
    let speed_q8 = cur.read_i32()?;
    let start_phase_ms = cur.read_u32()?;
    Ok(Attachment {
        target,
        local_offset,
        playback: ClipPlayback {
            speed_q8,
            start_phase_ms,
        },
    })
}

fn parse_vclp(payload: &[u8]) -> Result<Vec<VoxelClip>, ParseError> {
    let mut cur = Cursor::new(payload);
    let count = cur.read_u32()? as usize;
    // QE.6b - >= 4 B length prefix per clip blob.
    let mut clips = Vec::with_capacity(cur.clamped_capacity(count, 4));
    for _ in 0..count {
        let blob_len = cur.read_u32()? as usize;
        let blob = cur.read_bytes(blob_len)?;
        clips.push(VoxelClip::parse(blob).map_err(ParseError::BadClip)?);
    }
    Ok(clips)
}

fn parse_clps(payload: &[u8], numbone: usize) -> Result<Vec<Clip>, ParseError> {
    let mut cur = Cursor::new(payload);
    let count = cur.read_u32()? as usize;
    // QE.6b - >= 8 B per named clip record.
    let mut clips = Vec::with_capacity(cur.clamped_capacity(count, 8));
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
    // QE.6b - a BoneXform row is 40 B per bone.
    let mut frmval = Vec::with_capacity(cur.clamped_capacity(numfrm, numhin.saturating_mul(40)));
    for _ in 0..numfrm {
        let mut row = Vec::with_capacity(cur.clamped_capacity(numhin, 40));
        for _ in 0..numhin {
            row.push(read_bonexform(&mut cur)?);
        }
        frmval.push(row);
    }
    let seqcount = cur.read_u32()? as usize;
    let mut seq = Vec::with_capacity(cur.clamped_capacity(seqcount, 8));
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

/// One per-bone keyframe transform: translation (3), rotation quaternion
/// (x, y, z, w), scale (3) — ten little-endian `f32`s, 40 bytes.
fn read_bonexform(cur: &mut Cursor<'_>) -> Result<BoneXform, OutOfBounds> {
    let t = [cur.read_f32()?, cur.read_f32()?, cur.read_f32()?];
    let r = Quat {
        x: cur.read_f32()?,
        y: cur.read_f32()?,
        z: cur.read_f32()?,
        w: cur.read_f32()?,
    };
    let s = [cur.read_f32()?, cur.read_f32()?, cur.read_f32()?];
    Ok(BoneXform { t, r, s })
}

fn write_bonexform(out: &mut Vec<u8>, x: &BoneXform) {
    for v in [
        x.t[0], x.t[1], x.t[2], x.r.x, x.r.y, x.r.z, x.r.w, x.s[0], x.s[1], x.s[2],
    ] {
        out.extend_from_slice(&v.to_le_bytes());
    }
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
        let attach_count =
            u32::try_from(bone.attachments.len()).expect("attachment count must fit in u32");
        out.extend_from_slice(&attach_count.to_le_bytes());
        for a in &bone.attachments {
            write_attachment(out, a);
        }
        write_hinge(out, &bone.hinge);
    }
}

fn write_attachment(out: &mut Vec<u8>, a: &Attachment) {
    let (kind, index) = match a.target {
        MeshRef::Static(i) => (MESH_KIND_STATIC, i),
        MeshRef::Clip(i) => (MESH_KIND_CLIP, i),
    };
    out.extend_from_slice(&kind.to_le_bytes());
    let idx = u32::try_from(index).expect("attachment index must fit in u32");
    out.extend_from_slice(&idx.to_le_bytes());
    write_bonexform(out, &a.local_offset);
    out.extend_from_slice(&a.playback.speed_q8.to_le_bytes());
    out.extend_from_slice(&a.playback.start_phase_ms.to_le_bytes());
}

fn write_vclp(out: &mut Vec<u8>, c: &Character) {
    let count = u32::try_from(c.voxel_clips.len()).expect("voxel clip count must fit in u32");
    out.extend_from_slice(&count.to_le_bytes());
    for clip in &c.voxel_clips {
        let blob = clip.serialize();
        let blob_len = u32::try_from(blob.len()).expect("voxel clip blob length must fit in u32");
        out.extend_from_slice(&blob_len.to_le_bytes());
        out.extend_from_slice(&blob);
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

fn write_skeletal(out: &mut Vec<u8>, frmval: &[Vec<BoneXform>], seq: &[Seq]) {
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
            write_bonexform(out, v);
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
                    attachments: vec![Attachment::static_mesh(0)],
                    hinge: hinge(-1),
                },
                Bone {
                    name: "arm".to_string(),
                    attachments: vec![Attachment::static_mesh(1)],
                    hinge: hinge(0),
                },
            ],
            clips: vec![Clip {
                name: "wave".to_string(),
                // Bones rotate about +z; build rotation-only TRS frames from
                // the wave's Q15 angles.
                data: ClipData::Skeletal {
                    frmval: [[0i16, 0], [0, 16000], [0, 0], [0, -16000]]
                        .iter()
                        .map(|[r, a]| {
                            let z = [0.0, 0.0, 1.0];
                            vec![
                                BoneXform::from_hinge_angle(z, *r),
                                BoneXform::from_hinge_angle(z, *a),
                            ]
                        })
                        .collect(),
                    seq: vec![
                        Seq { tim: 0, frm: 0 },
                        Seq { tim: 500, frm: 1 },
                        Seq { tim: 1000, frm: 2 },
                        Seq { tim: 1500, frm: 3 },
                        Seq { tim: 2000, frm: !0 },
                    ],
                },
            }],
            voxel_clips: Vec::new(),
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
        assert_eq!(parsed.bones[1].attachments[0].target, MeshRef::Static(1));
        assert_eq!(parsed.bones[1].hinge.parent, 0);
        assert_eq!(parsed.clips.len(), 1);
        assert_eq!(parsed.clips[0].data, c.clips[0].data);
        assert!(parsed.voxel_clips.is_empty());
    }

    #[test]
    fn roundtrips_with_clips_and_multi_attachment() {
        use crate::voxel_clip::{LoopMode, VoxelClip, VoxelFrame};
        // A tiny 1-frame clip (1×1×4, one voxel at z=0).
        let frame = VoxelFrame {
            occupancy: vec![1u32],
            colors: vec![0x8011_2233],
            color_offsets: vec![0, 1],
        };
        let clip = VoxelClip::from_frames(
            [1, 1, 4],
            [0.5, 0.5, 2.0],
            1.0,
            LoopMode::Loop,
            &[frame],
            &[],
            33,
            0,
        );

        let mut c = synthetic_character();
        c.voxel_clips = vec![clip];
        // Give the body bone a SECOND attachment: the clip, at a non-identity
        // offset + non-default playback — so a flame hangs off it.
        c.bones[0].attachments.push(Attachment {
            target: MeshRef::Clip(0),
            local_offset: BoneXform {
                t: [1.0, 2.0, 3.0],
                r: Quat::IDENTITY,
                s: [1.0, 1.0, 1.0],
            },
            playback: ClipPlayback {
                speed_q8: 512,
                start_phase_ms: 100,
            },
        });

        let bytes = serialize(&c);
        let parsed = parse(&bytes).expect("parse v3 with clips");
        assert_eq!(serialize(&parsed), bytes, "byte round-trip");

        assert_eq!(parsed.voxel_clips.len(), 1);
        assert_eq!(parsed.voxel_clips[0], c.voxel_clips[0]);

        // body bone: static mesh 0 + clip 0 (multi-attachment).
        let body = &parsed.bones[0].attachments;
        assert_eq!(body.len(), 2);
        assert_eq!(body[0].target, MeshRef::Static(0));
        assert_eq!(body[1].target, MeshRef::Clip(0));
        assert_eq!(body[1].local_offset.t, [1.0, 2.0, 3.0]);
        assert_eq!(
            body[1].playback,
            ClipPlayback {
                speed_q8: 512,
                start_phase_ms: 100,
            }
        );

        // to_kfa_sprite keeps one limb per bone (first static attachment;
        // the clip is the attachment runtime's job).
        let k = parsed.to_kfa_sprite(None);
        assert_eq!(k.limbs.len(), parsed.bones.len());
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
        assert_ne!(
            k.kfaval[1],
            crate::xform::BoneXform::IDENTITY,
            "baked clip should move the arm bone"
        );

        // Rest pose: no curve attached → animsprite is a no-op.
        let mut rest = c.to_kfa_sprite(None);
        rest.animsprite(500);
        assert_eq!(rest.kfaval[1], crate::xform::BoneXform::IDENTITY);
    }

    #[test]
    fn to_kfa_export_is_lossy_but_valid() {
        let c = synthetic_character();
        let kfa = c.to_kfa(Some(0), "coco.kv6");
        // Skeleton + the selected clip survive; the filename is set.
        assert_eq!(kfa.kv6_name, b"coco.kv6");
        assert_eq!(kfa.hinges.len(), 2);
        assert_eq!(kfa.hinges[1].parent, 0);
        if let ClipData::Skeletal { frmval, seq } = &c.clips[0].data {
            // The .kfa export collapses each TRS to its hinge angle about +z;
            // assert it recovers the angles the frames were built from.
            let z = [0.0, 0.0, 1.0];
            let expected: Vec<Vec<i16>> = frmval
                .iter()
                .map(|row| row.iter().map(|x| x.hinge_angle(z)).collect())
                .collect();
            assert_eq!(kfa.frmval, expected);
            assert_eq!(&kfa.seq, seq);
        } else {
            panic!("clip 0 should be skeletal");
        }
        // The embedded meshes are dropped (lossy) — only the name remains.
        // The result is a well-formed .kfa: serialise + re-parse round-trips.
        let bytes = crate::kfa::serialize(&kfa);
        let reparsed = crate::kfa::parse(&bytes).expect("export round-trips through kfa");
        assert_eq!(reparsed.kv6_name, kfa.kv6_name);
        assert_eq!(reparsed.frmval, kfa.frmval);
        assert_eq!(reparsed.seq, kfa.seq);
    }

    #[test]
    fn to_kfa_without_clip_is_posable() {
        let c = synthetic_character();
        // No clip / out-of-range / Unknown → empty animation table.
        let kfa = c.to_kfa(None, b"x.kv6".to_vec());
        assert_eq!(kfa.hinges.len(), 2);
        assert!(kfa.frmval.is_empty());
        assert!(kfa.seq.is_empty());
        // Still a valid .kfa.
        let bytes = crate::kfa::serialize(&kfa);
        assert!(crate::kfa::parse(&bytes).is_ok());
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
        // Flip the first attachment's mesh_kind to an undefined value (2;
        // 0 = Static, 1 = Clip are both valid). The BONS payload begins:
        //   tag(4) len(4) count(4) name_len(2) name("body"=4) attach_count(4)
        //   kind(2)...
        let pos = find_tag(&bytes, *b"BONS");
        let kind_off = pos + 8 + 4 + 2 + 4 + 4;
        bytes[kind_off] = 2;
        bytes[kind_off + 1] = 0;
        assert!(matches!(
            parse(&bytes),
            Err(ParseError::UnsupportedMeshKind(2))
        ));
    }

    #[test]
    fn clip_bone_count_mismatch_errors() {
        // Build a character whose skeletal clip has the wrong bone count
        // by hand-serialising bones=1 but clip numhin=2.
        let mut c = synthetic_character();
        c.bones.pop(); // now 1 bone
        c.bones[0].attachments = vec![Attachment::static_mesh(0)];
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

    // ---- QE.6b adversarial characters: error, never panic/hang ----

    #[test]
    fn parse_rejects_out_of_range_attachment_index() {
        // Pre-QE.6b: parse accepted this and to_kfa_sprite / the
        // attachment runtime panicked out-of-bounds later.
        let mut c = synthetic_character();
        c.bones[1].attachments = vec![Attachment::static_mesh(99)];
        let bytes = serialize(&c);
        assert!(matches!(
            parse(&bytes),
            Err(ParseError::BadAttachmentIndex {
                bone: 1,
                index: 99,
                table_len: 2,
            })
        ));
    }

    #[test]
    fn parse_rejects_bad_or_cyclic_bone_skeleton() {
        // Out-of-range parent (pre-QE.6b: OOB panic in the solve).
        let mut c = synthetic_character();
        c.bones[1].hinge.parent = 42;
        assert!(matches!(
            parse(&serialize(&c)),
            Err(ParseError::BadSkeleton { bone: 1 })
        ));
        // Cyclic parents (pre-QE.6b: the loader hung forever).
        let mut c = synthetic_character();
        c.bones[0].hinge.parent = 1;
        assert!(matches!(
            parse(&serialize(&c)),
            Err(ParseError::BadSkeleton { .. })
        ));
    }
}
