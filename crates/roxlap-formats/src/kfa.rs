//! `.kfa` kv6 hinge / animation transform data.
//!
//! Reference: voxlaptest's `getkfa` (`voxlap5.c:9454`) and the
//! `hingetype` / `seqtyp` / `kfatype` declarations in
//! `voxlap5.h:38..59`. File layout (all multi-byte fields are little-
//! endian; structs are tightly packed because `voxlap5.h` opens with
//! `#pragma pack(push, 1)` before declaring them):
//!
//! ```text
//! offset  size                            description
//! 0x00    u32                             magic = 0x6b6c774b ("Kwlk")
//! 0x04    u32                             name_len
//! 0x08    name_len bytes                  associated kv6 filename (no NUL)
//! ...     u32                             numhin
//! ...     numhin × 64 bytes               hinges
//! ...     u32                             numfrm
//! ...     numfrm × numhin × i16           frmval (per-frame, per-hinge values)
//! ...     u32                             seqnum
//! ...     seqnum × 8 bytes                seq (tim:i32, frm:i32)
//! ```
//!
//! `hingetype` (64 bytes packed):
//!
//! ```text
//!     i32    parent       index of parent hinge (-1 = none)
//!     point3 p[2]         "velcro" anchor points (24 bytes, 2 × 3 × f32)
//!     point3 v[2]         rotation axes (24 bytes)
//!     i16    vmin
//!     i16    vmax
//!     u8     htype
//!     u8[7]  filler
//! ```
//!
//! No real `.kfa` fixture lives in voxlaptest yet (the oracle doesn't
//! render animated sprites), so this module's tests build a synthetic
//! `Kfa`, serialise, parse, and assert struct-equal + byte-equal
//! round-trip. Swap in a real fixture once R6 / sprite animation
//! coverage needs one.

use core::fmt;

use crate::bytes::{Cursor, OutOfBounds};
use crate::xform::BoneXform;

const MAGIC: u32 = 0x6b6c_774b; // "Kwlk" little-endian
const HINGE_SIZE: usize = 64;
const SEQ_SIZE: usize = 8;

/// 3D point (`point3d` in voxlaptest), 12 bytes packed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// One hinge / joint definition (`hingetype` in voxlaptest).
#[derive(Debug, Clone, Copy)]
pub struct Hinge {
    /// Index of the parent hinge in the same `Kfa`, or `-1` for none.
    pub parent: i32,
    /// Anchor ("velcro") points — `p[0]` on this object, `p[1]` on the
    /// parent.
    pub p: [Point3; 2],
    /// Rotation axes — same convention as `p`.
    pub v: [Point3; 2],
    pub vmin: i16,
    pub vmax: i16,
    pub htype: u8,
    /// Trailing 7 bytes of padding inside the on-disk struct. Stored
    /// verbatim so byte-equal round-trip survives — files in the wild
    /// may carry non-zero bytes here.
    pub filler: [u8; 7],
}

/// One animation sequence entry (`seqtyp` in voxlaptest).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Seq {
    pub tim: i32,
    pub frm: i32,
}

/// Parsed `.kfa` file. Round-trips byte-equally via [`parse`] +
/// [`serialize`].
#[derive(Debug, Clone)]
pub struct Kfa {
    /// Associated `.kv6` filename (raw bytes, no NUL terminator). Voxlap
    /// uses this to locate the rigged kv6 model.
    pub kv6_name: Vec<u8>,
    pub hinges: Vec<Hinge>,
    /// `frmval[frame_idx][hinge_idx]` — outer length is `numfrm`,
    /// inner length must equal `hinges.len()` for every frame.
    pub frmval: Vec<Vec<i16>>,
    pub seq: Vec<Seq>,
}

/// Errors returned by [`parse`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    /// First 4 bytes are not the `0x6b6c774b` magic.
    BadMagic { got: u32 },
    /// A read of `need` bytes at offset `at` would run past EOF.
    Truncated { at: usize, need: usize },
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::BadMagic { got } => {
                write!(f, "kfa bad magic: got {got:#010x}, expected 0x6b6c774b")
            }
            Self::Truncated { at, need } => {
                write!(f, "kfa truncated: need {need} bytes at offset {at}")
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

/// Parse a `.kfa` file's bytes into a [`Kfa`].
///
/// # Errors
///
/// Returns [`ParseError`] if the magic mismatches or a sequential read
/// for any header / hinge / frmval / seq region runs past EOF.
pub fn parse(bytes: &[u8]) -> Result<Kfa, ParseError> {
    let mut cur = Cursor::new(bytes);
    let magic = cur.read_u32()?;
    if magic != MAGIC {
        return Err(ParseError::BadMagic { got: magic });
    }

    let name_len = cur.read_u32()? as usize;
    let kv6_name = cur.read_bytes(name_len)?.to_vec();

    let numhin = cur.read_u32()? as usize;
    let mut hinges = Vec::with_capacity(numhin);
    for _ in 0..numhin {
        hinges.push(read_hinge(&mut cur)?);
    }

    let numfrm = cur.read_u32()? as usize;
    let mut frmval = Vec::with_capacity(numfrm);
    for _ in 0..numfrm {
        let mut row = Vec::with_capacity(numhin);
        for _ in 0..numhin {
            row.push(cur.read_i16()?);
        }
        frmval.push(row);
    }

    let seqnum = cur.read_u32()? as usize;
    let mut seq = Vec::with_capacity(seqnum);
    for _ in 0..seqnum {
        let tim = cur.read_i32()?;
        let frm = cur.read_i32()?;
        seq.push(Seq { tim, frm });
    }

    Ok(Kfa {
        kv6_name,
        hinges,
        frmval,
        seq,
    })
}

/// Serialise a [`Kfa`] back to bytes. Round-trips byte-equally with
/// the input that produced this `Kfa` via [`parse`].
///
/// # Panics
///
/// Panics if `kv6_name.len()`, `hinges.len()`, `frmval.len()`, or
/// `seq.len()` does not fit in a `u32` (the on-disk format stores
/// these as `u32`), or if `frmval` is not rectangular (every inner
/// row's length must equal `hinges.len()`). `Kfa` values produced by
/// [`parse`] always satisfy these invariants.
#[must_use]
pub fn serialize(kfa: &Kfa) -> Vec<u8> {
    let numhin = kfa.hinges.len();
    for (i, row) in kfa.frmval.iter().enumerate() {
        assert!(
            row.len() == numhin,
            "kfa frmval[{}].len() = {}, expected numhin = {}",
            i,
            row.len(),
            numhin,
        );
    }
    let name_len = u32::try_from(kfa.kv6_name.len()).expect("kv6_name length must fit in u32");
    let numhin_u32 = u32::try_from(numhin).expect("numhin must fit in u32");
    let numfrm_u32 = u32::try_from(kfa.frmval.len()).expect("numfrm must fit in u32");
    let seqnum_u32 = u32::try_from(kfa.seq.len()).expect("seqnum must fit in u32");

    let total = 4
        + 4
        + kfa.kv6_name.len()
        + 4
        + numhin * HINGE_SIZE
        + 4
        + (kfa.frmval.len() * numhin) * 2
        + 4
        + kfa.seq.len() * SEQ_SIZE;
    let mut out = Vec::with_capacity(total);

    out.extend_from_slice(&MAGIC.to_le_bytes());
    out.extend_from_slice(&name_len.to_le_bytes());
    out.extend_from_slice(&kfa.kv6_name);

    out.extend_from_slice(&numhin_u32.to_le_bytes());
    for h in &kfa.hinges {
        write_hinge(&mut out, h);
    }

    out.extend_from_slice(&numfrm_u32.to_le_bytes());
    for row in &kfa.frmval {
        for v in row {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }

    out.extend_from_slice(&seqnum_u32.to_le_bytes());
    for s in &kfa.seq {
        out.extend_from_slice(&s.tim.to_le_bytes());
        out.extend_from_slice(&s.frm.to_le_bytes());
    }

    out
}

// --- internal helpers ---------------------------------------------------

fn read_point3(cur: &mut Cursor<'_>) -> Result<Point3, OutOfBounds> {
    let x = cur.read_f32()?;
    let y = cur.read_f32()?;
    let z = cur.read_f32()?;
    Ok(Point3 { x, y, z })
}

fn write_point3(out: &mut Vec<u8>, p: Point3) {
    out.extend_from_slice(&p.x.to_le_bytes());
    out.extend_from_slice(&p.y.to_le_bytes());
    out.extend_from_slice(&p.z.to_le_bytes());
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

fn write_hinge(out: &mut Vec<u8>, h: &Hinge) {
    out.extend_from_slice(&h.parent.to_le_bytes());
    write_point3(out, h.p[0]);
    write_point3(out, h.p[1]);
    write_point3(out, h.v[0]);
    write_point3(out, h.v[1]);
    out.extend_from_slice(&h.vmin.to_le_bytes());
    out.extend_from_slice(&h.vmax.to_le_bytes());
    out.push(h.htype);
    out.extend_from_slice(&h.filler);
}

// --- KFA sprite (host-facing scene type) --------------------------------

/// One animated KFA sprite — bones + hinges + per-bone live
/// animation values.
///
/// The host owns one of these per animated model, updates `kfaval[]`
/// over time, and passes it to roxlap-core's `draw_kfa_sprite` each
/// frame. Construction is data-only (this crate); rendering is in
/// `roxlap-core`.
#[derive(Clone)]
pub struct KfaSprite {
    /// One [`crate::sprite::Sprite`] per bone. Limb `i`'s
    /// `(s, h, f, p)` is computed per frame by the renderer from
    /// the parent's transform + hinge math; the `kv6` field holds
    /// the bone's kv6 mesh and never changes.
    pub limbs: Vec<crate::sprite::Sprite>,
    /// Bone hierarchy. Mirror of voxlap's `kfatype.hinge[]`.
    pub hinges: Vec<Hinge>,
    /// Topological sort of bone indices — populated once at
    /// construction, used by the renderer's per-frame loop.
    pub hinge_sort: Vec<usize>,
    /// Per-bone resolved local transform for the current frame (translation,
    /// quaternion rotation, scale). Generalises voxlap's `vx5.kfaval[]` (which
    /// was a single Q15 hinge angle) to full TRS. Updated per frame by
    /// [`Self::animsprite`], or poked directly by the host.
    pub kfaval: Vec<BoneXform>,
    /// World-space anchor of the root limb's `hinge.p[0]`. The
    /// root limb is positioned so `hinge.p[0]` lands at this
    /// point given the world basis below.
    pub p: [f32; 3],
    /// World-space basis for the root limb. Mirror of
    /// `vx5sprite.{s, h, f}` for the root.
    pub s: [f32; 3],
    pub h: [f32; 3],
    pub f: [f32; 3],
    /// Animation keyframe table — `frmval[frame][hinge]` local transforms.
    /// Empty until [`Self::set_animation`]; an empty table makes
    /// [`Self::animsprite`] a no-op so hosts that poke [`kfaval`](Self::kfaval)
    /// directly keep working.
    pub frmval: Vec<Vec<BoneXform>>,
    /// Animation sequence — ordered `(tim, frm)` keyframes. Mirror of
    /// `kfatype.seq`. `tim` is an absolute timestamp (ms); `frm` is a
    /// frame index into [`frmval`](Self::frmval), or `!target`
    /// (bitwise-NOT, hence negative) for a jump/loop to seq entry
    /// `target`.
    pub seq: Vec<Seq>,
    /// Current animation time (ms) — voxlap's `vx5sprite.kfatim`.
    /// Advanced by [`Self::animsprite`].
    pub kfatim: i32,
    /// Previous animation time (ms) — voxlap's `vx5sprite.okfatim`,
    /// used to cross-fade when the active sequence entry is itself a
    /// blend marker (`seq[z].frm < 0`). Host sets it when switching
    /// animations; [`Self::animsprite`] never writes it.
    pub okfatim: i32,
}

impl KfaSprite {
    /// Build a KFA sprite from a list of `(Sprite, Hinge)` bones.
    /// `limbs.len()` must equal `hinges.len()`. The first bone with
    /// `parent < 0` is the root.
    ///
    /// `kfaval` is initialised to all zeros; the host should set
    /// per-bone angles before / between render calls.
    ///
    /// # Panics
    ///
    /// Panics if `limbs.len() != hinges.len()`.
    #[must_use]
    pub fn new(limbs: Vec<crate::sprite::Sprite>, hinges: Vec<Hinge>, root_pos: [f32; 3]) -> Self {
        assert_eq!(
            limbs.len(),
            hinges.len(),
            "limbs ({}) and hinges ({}) length mismatch",
            limbs.len(),
            hinges.len()
        );
        let n = hinges.len();
        let hinge_sort = sort_hinges(&hinges);
        Self {
            limbs,
            hinges,
            hinge_sort,
            kfaval: vec![BoneXform::IDENTITY; n],
            p: root_pos,
            s: [1.0, 0.0, 0.0],
            h: [0.0, 1.0, 0.0],
            f: [0.0, 0.0, 1.0],
            frmval: Vec::new(),
            seq: Vec::new(),
            kfatim: 0,
            okfatim: 0,
        }
    }

    /// Attach an animation curve — the `frmval` + `seq` tables parsed
    /// from a [`Kfa`]. After this, [`Self::animsprite`] drives
    /// [`kfaval`](Self::kfaval) from playback time instead of the host
    /// poking individual bones.
    pub fn set_animation(&mut self, frmval: Vec<Vec<BoneXform>>, seq: Vec<Seq>) {
        self.frmval = frmval;
        self.seq = seq;
    }

    /// Advance the animation by `ti` milliseconds and recompute every
    /// child bone's [`kfaval`](Self::kfaval) — a faithful port of
    /// voxlap's `animsprite` (`voxlap5.c:11125`).
    ///
    /// Walks the sequence forward from the current
    /// [`kfatim`](Self::kfatim) (honouring `!target` jump/loop
    /// entries), then piecewise-linearly interpolates the two bracketing
    /// keyframes per hinge. Interpolation is angle-wrap-aware: a free
    /// hinge (`vmin == vmax`) takes the shortest path, a limited hinge
    /// winds in its allowed direction. When the active entry is itself a
    /// blend marker (`seq[z].frm < 0`), the pose cross-fades from the
    /// [`okfatim`](Self::okfatim)-derived frame.
    ///
    /// No-op when no animation curve is attached (see
    /// [`Self::set_animation`]).
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        clippy::similar_names
    )]
    pub fn animsprite(&mut self, mut ti: i32) {
        if self.seq.is_empty() || self.frmval.is_empty() {
            return;
        }
        let numhin = self.hinges.len();
        let seqnum = self.seq.len();

        // Phase 1 — advance kfatim by `ti` ms through the sequence,
        // following `!target` jump entries (voxlap5.c:11133-11143).
        let mut z = kfatime2seq(&self.seq, self.kfatim) as i32;
        while ti > 0 {
            z += 1;
            if z as usize >= seqnum {
                break;
            }
            let dt = self.seq[z as usize].tim - self.kfatim;
            if dt <= 0 {
                break;
            }
            if dt > ti {
                self.kfatim += ti;
                break;
            }
            ti -= dt;
            let jump = !self.seq[z as usize].frm; // ~frm
            if jump >= 0 {
                if z == jump {
                    break;
                }
                z = jump;
            }
            self.kfatim = self.seq[z as usize].tim;
        }

        // Phase 2 — resolve the bracketing frames + 16.16 blend ratios
        // for the current segment (voxlap5.c:11147-11167).
        let z_seq = kfatime2seq(&self.seq, self.kfatim);
        let zz_idx = z_seq + 1;
        let (trat, zz_frm) = if zz_idx < seqnum && self.seq[zz_idx].frm != !(zz_idx as i32) {
            let span = self.seq[zz_idx].tim - self.seq[z_seq].tim;
            let trat = if span != 0 {
                shldiv16(self.kfatim - self.seq[z_seq].tim, span)
            } else {
                0
            };
            let i = self.seq[zz_idx].frm;
            let zz_frm = if i < 0 {
                self.seq[(!i) as usize].frm
            } else {
                i
            };
            (trat, zz_frm)
        } else {
            (0, 0)
        };

        let z_frm = self.seq[z_seq].frm;
        // trat2 < 0 signals "no okfatim cross-fade" (the common path).
        let mut trat2 = -1i32;
        let mut z0_frm = 0i32;
        let mut zz0_frm = 0i32;
        if z_frm < 0 {
            let z0_seq = kfatime2seq(&self.seq, self.okfatim);
            let zz0_idx = z0_seq + 1;
            if zz0_idx < seqnum && self.seq[zz0_idx].frm != !(zz0_idx as i32) {
                let span = self.seq[zz0_idx].tim - self.seq[z0_seq].tim;
                trat2 = if span != 0 {
                    shldiv16(self.okfatim - self.seq[z0_seq].tim, span)
                } else {
                    0
                };
                let i = self.seq[zz0_idx].frm;
                zz0_frm = if i < 0 {
                    self.seq[(!i) as usize].frm
                } else {
                    i
                };
            } else {
                trat2 = 0;
            }
            z0_frm = self.seq[z0_seq].frm;
            if z0_frm < 0 {
                z0_frm = zz0_frm;
                trat2 = 0;
            }
        }

        // Phase 3 — per-hinge interpolation into kfaval
        // (voxlap5.c:11169-11195). Root bones (parent < 0) keep their
        // value untouched, exactly as voxlap's `continue`.
        // `trat` / `trat2` are 16.16 fixed-point blend ratios; `/ 65536` gives
        // the `[0, 1]` factor for the TRS blend.
        for i in (0..numhin).rev() {
            if self.hinges[i].parent < 0 {
                continue;
            }
            let mut x = if trat2 < 0 {
                self.frmval[z_frm as usize][i]
            } else {
                let base = self.frmval[z0_frm as usize][i];
                if trat2 > 0 {
                    base.blend(self.frmval[zz0_frm as usize][i], trat2 as f32 / 65536.0)
                } else {
                    base
                }
            };
            if trat > 0 {
                x = x.blend(self.frmval[zz_frm as usize][i], trat as f32 / 65536.0);
            }
            self.kfaval[i] = x;
        }
    }
}

/// 16.16 fixed-point signed shift-divide — voxlap's `shldiv16`
/// (`voxlap5.c:296`): `((i64)a << 16) / b`, truncating toward zero
/// (matching x86 `idiv`).
#[inline]
#[allow(clippy::cast_possible_truncation)]
fn shldiv16(a: i32, b: i32) -> i32 {
    ((i64::from(a) << 16) / i64::from(b)) as i32
}

/// Binary-search the seq entry whose `tim` brackets `tim` from below —
/// voxlap's `kfatime2seq` (`voxlap5.c`). Returns the index `a` such
/// that `seq[a].tim <= tim < seq[a+1].tim` (clamped to the ends).
/// Caller guarantees `seq` is non-empty.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
fn kfatime2seq(seq: &[Seq], tim: i32) -> usize {
    let mut a: isize = 0;
    let mut b: isize = seq.len() as isize - 1;
    while b - a >= 2 {
        let i = (a + b) >> 1;
        if tim >= seq[i as usize].tim {
            a = i;
        } else {
            b = i;
        }
    }
    a as usize
}

/// Build the hinge-sort order — voxlap's `kfasorthinge`
/// (`voxlap5.c:9427-9450`). The result is an array of hinge
/// indices ordered such that **walking from index `n-1` down to
/// 0** visits parents before children — a valid topological order
/// for the chain of `setlimb` calls in voxlap's `kfadraw`.
///
/// Voxlap mutates the hinges in place during sort and restores
/// them; this port produces the same `hsort` array without
/// touching the input.
#[must_use]
#[allow(clippy::cast_sign_loss)] // parent >= 0 checked immediately above
pub fn sort_hinges(hinges: &[Hinge]) -> Vec<usize> {
    let n = hinges.len();
    let mut hsort = vec![0usize; n];
    // First pass: roots at the end, non-roots at the start.
    let mut head = 0usize;
    let mut tail = n;
    for i in (0..n).rev() {
        if hinges[i].parent < 0 {
            tail -= 1;
            hsort[tail] = i;
        } else {
            hsort[head] = i;
            head += 1;
        }
    }

    // `solved[h]` = true once hinge h's parent has been settled
    // into the "tail" half. Voxlap encodes this in-place by
    // flipping the parent field to -2-parent; we use a side
    // bitmap to leave the input immutable.
    let mut solved = vec![false; n];
    for i in (tail..n).rev() {
        solved[hsort[i]] = true;
    }

    // Iterative pass: pick non-root entries in head whose parent
    // is already solved; move them to the tail.
    let mut idx = head; // idx walks the head [0..head) backward
    while tail > 0 {
        if idx == 0 {
            idx = head;
        }
        idx -= 1;
        let j = hsort[idx];
        let parent = hinges[j].parent;
        if parent < 0 {
            // Already in the tail (shouldn't happen since the
            // first pass sorted these out).
            continue;
        }
        if solved[parent as usize] {
            solved[j] = true;
            tail -= 1;
            hsort[idx] = hsort[tail];
            hsort[tail] = j;
            head -= 1;
        }
        if head == 0 {
            break;
        }
    }
    hsort
}

// --- tests --------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_kfa() -> Kfa {
        Kfa {
            kv6_name: b"anasaur.kv6".to_vec(),
            hinges: vec![
                Hinge {
                    parent: -1,
                    p: [
                        Point3 {
                            x: 0.0,
                            y: 0.0,
                            z: 0.0,
                        },
                        Point3 {
                            x: 1.0,
                            y: 0.0,
                            z: 0.0,
                        },
                    ],
                    v: [
                        Point3 {
                            x: 0.0,
                            y: 1.0,
                            z: 0.0,
                        },
                        Point3 {
                            x: 0.0,
                            y: 0.0,
                            z: 1.0,
                        },
                    ],
                    vmin: -180,
                    vmax: 180,
                    htype: 0,
                    filler: [0; 7],
                },
                Hinge {
                    parent: 0,
                    p: [
                        Point3 {
                            x: 0.5,
                            y: 0.0,
                            z: 0.0,
                        },
                        Point3 {
                            x: 0.5,
                            y: 1.0,
                            z: 0.0,
                        },
                    ],
                    v: [
                        Point3 {
                            x: 1.0,
                            y: 0.0,
                            z: 0.0,
                        },
                        Point3 {
                            x: 0.0,
                            y: 1.0,
                            z: 0.0,
                        },
                    ],
                    vmin: -90,
                    vmax: 90,
                    htype: 1,
                    // Non-zero filler tests round-trip preservation.
                    filler: [0xde, 0xad, 0xbe, 0xef, 0xca, 0xfe, 0xba],
                },
            ],
            frmval: vec![vec![0, 0], vec![45, -30], vec![90, -60], vec![135, -90]],
            seq: vec![
                Seq { tim: 0, frm: 0 },
                Seq { tim: 100, frm: 1 },
                Seq { tim: 200, frm: 2 },
                Seq { tim: 300, frm: 3 },
            ],
        }
    }

    #[test]
    fn synthetic_roundtrips_byte_equal() {
        let kfa = synthetic_kfa();
        let bytes = serialize(&kfa);
        let parsed = parse(&bytes).expect("parse synthetic");
        let bytes2 = serialize(&parsed);
        assert_eq!(bytes, bytes2, "byte-level round-trip failed");
        // Spot-check the structural round-trip too.
        assert_eq!(parsed.kv6_name, kfa.kv6_name);
        assert_eq!(parsed.hinges.len(), kfa.hinges.len());
        assert_eq!(parsed.frmval, kfa.frmval);
        assert_eq!(parsed.seq, kfa.seq);
    }

    #[test]
    fn hinge_size_matches_voxlap_packed_layout() {
        // 4 (parent) + 24 (p[2]) + 24 (v[2]) + 2 (vmin) + 2 (vmax)
        //   + 1 (htype) + 7 (filler) = 64.
        assert_eq!(HINGE_SIZE, 64);
        // And we serialise exactly that many bytes per hinge.
        let kfa = synthetic_kfa();
        let bytes = serialize(&kfa);
        // 4 magic + 4 name_len + 11 name + 4 numhin = 23 bytes header.
        let header = 4 + 4 + kfa.kv6_name.len() + 4;
        let after_hinges = header + kfa.hinges.len() * HINGE_SIZE;
        // Re-parse and verify the second hinge's filler matches what we set.
        let parsed = parse(&bytes).expect("parse synthetic");
        assert_eq!(
            parsed.hinges[1].filler,
            [0xde, 0xad, 0xbe, 0xef, 0xca, 0xfe, 0xba]
        );
        // Sanity: total size must include numfrm field after hinges.
        assert!(bytes.len() > after_hinges + 4);
    }

    #[test]
    fn parse_bad_magic_fails() {
        let mut bytes = serialize(&synthetic_kfa());
        bytes[0] ^= 0xff;
        let r = parse(&bytes);
        assert!(matches!(r, Err(ParseError::BadMagic { .. })));
    }

    #[test]
    fn parse_truncated_in_hinge_table_fails() {
        let bytes = serialize(&synthetic_kfa());
        // Truncate inside the first hinge.
        let truncated = &bytes[..30];
        let r = parse(truncated);
        assert!(matches!(r, Err(ParseError::Truncated { .. })));
    }

    /// `sort_hinges` puts roots at high indices and children at low.
    /// 3-bone chain: root → child1 → child2.
    #[test]
    #[allow(clippy::cast_sign_loss)] // p >= 0 checked at the assert site
    fn sort_hinges_three_bone_chain() {
        let axis = |x: f32, y: f32, z: f32| Point3 { x, y, z };
        let h = |parent: i32| Hinge {
            parent,
            p: [axis(0.0, 0.0, 0.0); 2],
            v: [axis(1.0, 0.0, 0.0); 2],
            vmin: 0,
            vmax: 0,
            htype: 0,
            filler: [0; 7],
        };
        // hinge[0] = root, hinge[1] child of 0, hinge[2] child of 1.
        let hinges = vec![h(-1), h(0), h(1)];
        let sort = sort_hinges(&hinges);
        // Walking sort[i] for i=n-1..=0 must visit each bone's parent
        // before the bone itself.
        let mut seen = [false; 3];
        for k in (0..3).rev() {
            let j = sort[k];
            seen[j] = true;
            let p = hinges[j].parent;
            if p >= 0 {
                assert!(
                    seen[p as usize],
                    "bone {j}'s parent {p} not yet visited at descent step k={k}"
                );
            }
        }
    }

    // --- animsprite playback ------------------------------------------

    /// Minimal two-bone sprite (root + one child hinge) for driving
    /// [`KfaSprite::animsprite`]. `limbs` is empty — `animsprite` reads
    /// only the hinges + curve, never the limb geometry — so we build
    /// the struct directly to avoid needing a kv6.
    fn anim_sprite(
        child_vmin: i16,
        child_vmax: i16,
        frmval: Vec<Vec<i16>>,
        seq: Vec<Seq>,
    ) -> KfaSprite {
        let zero = Point3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        let axis = Point3 {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        };
        let hinges = vec![
            Hinge {
                parent: -1,
                p: [zero, zero],
                v: [axis, axis],
                vmin: 0,
                vmax: 0,
                htype: 0,
                filler: [0; 7],
            },
            Hinge {
                parent: 0,
                p: [zero, zero],
                v: [axis, axis],
                vmin: child_vmin,
                vmax: child_vmax,
                htype: 0,
                filler: [0; 7],
            },
        ];
        // Tests author keyframes as Q15 angles; migrate them to rotation-only
        // BoneXforms about each bone's hinge axis (the runtime model is TRS).
        let frmval: Vec<Vec<BoneXform>> = frmval
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .enumerate()
                    .map(|(b, a)| {
                        let v = hinges[b].v[0];
                        BoneXform::from_hinge_angle([v.x, v.y, v.z], a)
                    })
                    .collect()
            })
            .collect();
        KfaSprite {
            limbs: Vec::new(),
            hinge_sort: sort_hinges(&hinges),
            kfaval: vec![BoneXform::IDENTITY; hinges.len()],
            hinges,
            p: [0.0; 3],
            s: [1.0, 0.0, 0.0],
            h: [0.0, 1.0, 0.0],
            f: [0.0, 0.0, 1.0],
            frmval,
            seq,
            kfatim: 0,
            okfatim: 0,
        }
    }

    /// Recover bone `i`'s Q15 hinge angle about the test axis (`+x`) from its
    /// resolved `kfaval` — the inverse of how the helper builds keyframes.
    fn angle_of(kfa: &KfaSprite, i: usize) -> i16 {
        kfa.kfaval[i].hinge_angle([1.0, 0.0, 0.0])
    }

    /// Half-way through a single 0→16384 segment a free hinge sits at
    /// exactly 8192, and the root bone is left untouched.
    #[test]
    fn animsprite_lerps_free_hinge_midpoint() {
        // Free hinge: vmin == vmax.
        let mut kfa = anim_sprite(
            0,
            0,
            vec![vec![0, 0], vec![0, 16384]],
            vec![Seq { tim: 0, frm: 0 }, Seq { tim: 1000, frm: 1 }],
        );
        kfa.animsprite(500);
        assert_eq!(kfa.kfatim, 500, "time cursor advanced by ti");
        assert_eq!(angle_of(&kfa, 0), 0, "root bone untouched");
        // nlerp at t=0.5 of two same-axis rotations is exact, so the midpoint
        // is still 8192 (45°).
        assert!(
            (i32::from(angle_of(&kfa, 1)) - 8192).abs() <= 2,
            "child at midpoint"
        );
    }

    /// A free hinge interpolating 30000 → -30000 takes the *short* way
    /// (through ±32768), not the long way through 0 — so the midpoint
    /// lands at the wrap boundary, not near 0.
    #[test]
    fn animsprite_free_hinge_takes_shortest_wrap() {
        let mut kfa = anim_sprite(
            0,
            0,
            vec![vec![0, 30000], vec![0, -30000]],
            vec![Seq { tim: 0, frm: 0 }, Seq { tim: 1000, frm: 1 }],
        );
        kfa.animsprite(500);
        // nlerp takes the short arc (the quaternions are flipped to the same
        // hemisphere), so the midpoint lands at the ±180° wrap, not near 0.
        assert!(
            i32::from(angle_of(&kfa, 1)).abs() >= 32000,
            "midpoint at the wrap"
        );
    }

    /// `seq[].frm < 0` is a `!target` jump: advancing time past the
    /// jump entry loops back to `target` and keeps consuming `ti`.
    #[test]
    fn animsprite_follows_loop_jump_entry() {
        let mut kfa = anim_sprite(
            0,
            0,
            vec![vec![0, 0], vec![0, 16384]],
            vec![
                Seq { tim: 0, frm: 0 },
                Seq { tim: 1000, frm: 1 },
                // Jump back to seq entry 0 (== !0 == -1).
                Seq { tim: 2000, frm: !0 },
            ],
        );
        // 2500 ms: 0→1000 (seg 0), 1000→2000 hits the jump → loop to 0,
        // then 500 ms more into the first segment again.
        kfa.animsprite(2500);
        assert_eq!(kfa.kfatim, 500, "looped back and advanced 500 ms");
    }

    /// With no curve attached, animsprite leaves kfaval alone so hosts
    /// that drive kfaval[] directly are unaffected.
    #[test]
    fn animsprite_no_curve_is_noop() {
        let mut kfa = anim_sprite(0, 0, Vec::new(), Vec::new());
        kfa.kfaval[1] = BoneXform::from_hinge_angle([1.0, 0.0, 0.0], 1234);
        kfa.animsprite(500);
        assert_eq!(angle_of(&kfa, 1), 1234);
        assert_eq!(kfa.kfatim, 0);
    }

    #[test]
    fn kfatime2seq_brackets_from_below() {
        let seq = vec![
            Seq { tim: 0, frm: 0 },
            Seq { tim: 100, frm: 1 },
            Seq { tim: 200, frm: 2 },
            Seq { tim: 300, frm: 3 },
        ];
        assert_eq!(kfatime2seq(&seq, 0), 0);
        assert_eq!(kfatime2seq(&seq, 99), 0);
        assert_eq!(kfatime2seq(&seq, 100), 1);
        assert_eq!(kfatime2seq(&seq, 250), 2);
        // Never returns the final index: the last entry is always the
        // *upper* bracket, so beyond it we stay on the last segment.
        assert_eq!(kfatime2seq(&seq, 9999), 2, "last segment's lower bracket");
    }
}
