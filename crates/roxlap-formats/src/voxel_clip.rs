//! Animated voxel-sprite clips (`.rvc`) — a "GIF/MP4 for voxel models".
//!
//! A [`VoxelClip`] is a fixed-bounding-box sequence of voxel frames,
//! encoded as **keyframes + inter-frame diffs** (like video I/P frames),
//! for effects such as flame, spells, and muzzle flashes. See
//! `PORTING-VOXEL-CLIP.md` for the full design (stage VCL).
//!
//! ## Frame representation
//!
//! A frame is stored in the same **dense-column layout** the GPU sprite
//! model uses ([`roxlap-gpu`'s `SpriteModel`]): a per-`(x, y)`-column
//! occupancy bitmask plus per-column ascending-z colour runs. Columns
//! are indexed `col = x + y * dims[0]`; a column's occupancy is
//! [`occ_words_per_col`](VoxelClip::occ_words_per_col) u32 words, bit
//! `z & 31` of word `z >> 5`. This makes GPU upload a field move (no
//! bucket-sort) and makes diffs clean (per column). Surface-normal
//! `dir` indices are **recomputed at [`decode`](VoxelClip::decode)** from
//! the reconstructed occupancy, so the on-disk codec carries only
//! occupancy + colour.
//!
//! ## On-disk format (`.rvc`)
//!
//! ```text
//! magic   b"RVCL"
//! version u16 = 1
//! chunks  [tag(4) | len(u32) | payload]  until EOF; unknown tags preserved
//!   META : dims[3] u32, pivot[3] f32, voxel_world_size f32,
//!          loop_mode u8, default_frame_ms u32, frame_count u32
//!   FRMS : per frame: kind u8 {Key=0, Delta=1}; Key = full frame
//!          (occupancy + color_offsets + colors, each u32-len-prefixed);
//!          Delta = changed_count u32 + per changed column
//!          (col u32, occ_words_per_col × u32, color_run len+u32s)
//!   TIME : optional per-frame durations (frame_count × u32 ms)
//! ```
//!
//! v1 is plain (no varints / no deflate) — those are reserved as
//! backward-compatible follow-ups (a per-chunk "compressed" flag).

use crate::bytes::{Cursor, OutOfBounds};
use crate::kv6::compute_vis_dir;

const MAGIC: [u8; 4] = *b"RVCL";
const VERSION: u16 = 1;

const TAG_META: [u8; 4] = *b"META";
const TAG_FRMS: [u8; 4] = *b"FRMS";
const TAG_TIME: [u8; 4] = *b"TIME";

const FRAME_KIND_KEY: u8 = 0;
const FRAME_KIND_DELTA: u8 = 1;

/// How playback advances past the last frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopMode {
    /// Wrap back to frame 0 (the default for ambient effects).
    Loop,
    /// Hold the last frame (one-shot, e.g. a spell impact).
    Once,
    /// Bounce 0→N→0 (ping-pong).
    PingPong,
}

impl LoopMode {
    fn to_u8(self) -> u8 {
        match self {
            Self::Loop => 0,
            Self::Once => 1,
            Self::PingPong => 2,
        }
    }
    fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Loop),
            1 => Some(Self::Once),
            2 => Some(Self::PingPong),
            _ => None,
        }
    }
}

/// One fully-reconstructed frame in the dense-column layout. Field shapes
/// are validated against the clip's `dims` by [`VoxelFrame::validate`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoxelFrame {
    /// Per-column occupancy bitmask, `dims[0] * dims[1] *
    /// occ_words_per_col` words. Bit `z & 31` of word
    /// `col * occ_words_per_col + (z >> 5)` is set iff voxel `(x, y, z)`
    /// is solid, where `col = x + y * dims[0]`.
    pub occupancy: Vec<u32>,
    /// Voxel colours (voxlap-packed `0x80RRGGBB`), ascending z within
    /// each column, columns in `col` order.
    pub colors: Vec<u32>,
    /// Prefix sums: `color_offsets[col]` is the first colour index of
    /// column `col`; length `dims[0] * dims[1] + 1`.
    pub color_offsets: Vec<u32>,
}

impl VoxelFrame {
    /// Check the field shapes + per-column occupancy/colour agreement for
    /// the given clip `dims`.
    ///
    /// # Errors
    /// Returns the offending [`FrameError`] (wrong array length, broken
    /// prefix-sum bounds/monotonicity, or a column whose occupancy
    /// popcount disagrees with its colour-run length).
    pub fn validate(&self, dims: [u32; 3]) -> Result<(), FrameError> {
        let cols = (dims[0] as usize) * (dims[1] as usize);
        let owpc = occ_words_per_col(dims) as usize;
        if self.occupancy.len() != cols * owpc {
            return Err(FrameError::OccupancyLen);
        }
        if self.color_offsets.len() != cols + 1 {
            return Err(FrameError::OffsetsLen);
        }
        if self.color_offsets[0] != 0 || *self.color_offsets.last().unwrap() as usize != self.colors.len() {
            return Err(FrameError::OffsetsBounds);
        }
        for col in 0..cols {
            let lo = self.color_offsets[col];
            let hi = self.color_offsets[col + 1];
            if hi < lo {
                return Err(FrameError::OffsetsMonotonic);
            }
            let run = (hi - lo) as usize;
            let mut popcount = 0usize;
            for w in 0..owpc {
                popcount += self.occupancy[col * owpc + w].count_ones() as usize;
            }
            if popcount != run {
                return Err(FrameError::OccupancyColorMismatch(col));
            }
        }
        Ok(())
    }

    /// The colours of column `col` (`colors[color_offsets[col]..[col+1]]`).
    fn column_colors(&self, col: usize) -> &[u32] {
        &self.colors[self.color_offsets[col] as usize..self.color_offsets[col + 1] as usize]
    }

    /// The occupancy words of column `col`.
    fn column_occ(&self, col: usize, owpc: usize) -> &[u32] {
        &self.occupancy[col * owpc..(col + 1) * owpc]
    }
}

/// A per-column overwrite in a delta (P-) frame: the column's new
/// occupancy words + new ascending-z colour run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnDelta {
    pub col: u32,
    pub occ: Vec<u32>,
    pub colors: Vec<u32>,
}

/// One frame as stored in a [`VoxelClip`]: a full keyframe or a diff
/// against the previous reconstructed frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncodedFrame {
    /// Full frame (I-frame).
    Key(VoxelFrame),
    /// Sparse list of changed columns relative to the previous frame.
    Delta(Vec<ColumnDelta>),
}

/// On-disk animated voxel clip. Construct via [`VoxelClip::from_frames`]
/// (the encoder) or [`VoxelClip::parse`]; expand to a runtime flipbook
/// via [`VoxelClip::decode`].
#[derive(Debug, Clone, PartialEq)]
pub struct VoxelClip {
    pub dims: [u32; 3],
    pub pivot: [f32; 3],
    pub voxel_world_size: f32,
    pub loop_mode: LoopMode,
    /// Frame duration used when `durations` is empty.
    pub default_frame_ms: u32,
    /// I/P frame stream. The first frame must be a `Key`.
    pub frames: Vec<EncodedFrame>,
    /// Per-frame durations (ms); empty ⇒ uniform `default_frame_ms`.
    pub durations: Vec<u32>,
    /// Unknown top-level chunks, preserved verbatim for forward-compat.
    pub extra_chunks: Vec<([u8; 4], Vec<u8>)>,
}

/// A decoded clip: every frame expanded to a full [`VoxelFrame`] plus its
/// recomputed `dirs` (parallel to `frames[i].colors`) and resolved
/// durations. The runtime flipbook.
#[derive(Debug, Clone)]
pub struct DecodedClip {
    pub dims: [u32; 3],
    pub pivot: [f32; 3],
    pub voxel_world_size: f32,
    pub occ_words_per_col: u32,
    pub loop_mode: LoopMode,
    pub frames: Vec<VoxelFrame>,
    /// Per-frame surface-normal LUT indices, parallel to
    /// `frames[i].colors`.
    pub dirs: Vec<Vec<u32>>,
    pub durations: Vec<u32>,
}

impl DecodedClip {
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Total loop length in ms (sum of frame durations).
    #[must_use]
    pub fn total_ms(&self) -> u32 {
        self.durations.iter().copied().sum()
    }
}

/// u32 occupancy words per `(x, y)` column for a clip of `dims`.
#[must_use]
pub fn occ_words_per_col(dims: [u32; 3]) -> u32 {
    dims[2].div_ceil(32).max(1)
}

impl VoxelClip {
    /// u32 occupancy words per column for this clip.
    #[must_use]
    pub fn occ_words_per_col(&self) -> u32 {
        occ_words_per_col(self.dims)
    }

    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Encode a sequence of full frames into a clip. Frame 0 (and every
    /// `keyframe_interval`-th frame) is stored as a keyframe; the rest are
    /// diffed against the previous frame. `keyframe_interval == 0` ⇒ only
    /// frame 0 is a keyframe (smallest, but no mid-stream seek points).
    ///
    /// `durations` is per-frame ms; pass an empty slice for uniform
    /// `default_frame_ms`.
    ///
    /// # Panics
    /// If any frame fails [`VoxelFrame::validate`] for `dims`, or
    /// `durations` is non-empty but not `frames.len()` long.
    #[must_use]
    pub fn from_frames(
        dims: [u32; 3],
        pivot: [f32; 3],
        voxel_world_size: f32,
        loop_mode: LoopMode,
        frames: &[VoxelFrame],
        durations: &[u32],
        default_frame_ms: u32,
        keyframe_interval: u32,
    ) -> Self {
        for (i, f) in frames.iter().enumerate() {
            f.validate(dims)
                .unwrap_or_else(|e| panic!("frame {i} invalid: {e:?}"));
        }
        assert!(
            durations.is_empty() || durations.len() == frames.len(),
            "durations must be empty or one per frame",
        );
        let owpc = occ_words_per_col(dims) as usize;
        let cols = (dims[0] as usize) * (dims[1] as usize);

        let mut encoded = Vec::with_capacity(frames.len());
        for (i, frame) in frames.iter().enumerate() {
            let is_key = i == 0 || (keyframe_interval != 0 && (i as u32) % keyframe_interval == 0);
            if is_key {
                encoded.push(EncodedFrame::Key(frame.clone()));
            } else {
                let prev = &frames[i - 1];
                let mut changed = Vec::new();
                for col in 0..cols {
                    if prev.column_occ(col, owpc) != frame.column_occ(col, owpc)
                        || prev.column_colors(col) != frame.column_colors(col)
                    {
                        changed.push(ColumnDelta {
                            col: col as u32,
                            occ: frame.column_occ(col, owpc).to_vec(),
                            colors: frame.column_colors(col).to_vec(),
                        });
                    }
                }
                encoded.push(EncodedFrame::Delta(changed));
            }
        }

        Self {
            dims,
            pivot,
            voxel_world_size,
            loop_mode,
            default_frame_ms,
            frames: encoded,
            durations: durations.to_vec(),
            extra_chunks: Vec::new(),
        }
    }

    /// Expand the I/P stream to full frames, compute per-frame `dirs`, and
    /// resolve durations — the runtime flipbook.
    ///
    /// # Errors
    /// [`DecodeError::DeltaBeforeKey`] if the stream doesn't start with a
    /// keyframe; [`DecodeError::ColumnOutOfRange`] if a delta names a
    /// column outside `dims`; [`DecodeError::Frame`] if a reconstructed
    /// frame fails validation.
    pub fn decode(&self) -> Result<DecodedClip, DecodeError> {
        let owpc = occ_words_per_col(self.dims) as usize;
        let cols = (self.dims[0] as usize) * (self.dims[1] as usize);

        // Reconstruct incrementally via a per-column working set so a
        // delta is an O(changed columns) overwrite, not a flat-array splice.
        let mut work_occ = vec![0u32; cols * owpc];
        let mut work_cols: Vec<Vec<u32>> = vec![Vec::new(); cols];
        let mut frames: Vec<VoxelFrame> = Vec::with_capacity(self.frames.len());
        let mut started = false;

        for ef in &self.frames {
            match ef {
                EncodedFrame::Key(frame) => {
                    frame.validate(self.dims).map_err(DecodeError::Frame)?;
                    work_occ.copy_from_slice(&frame.occupancy);
                    for (col, wc) in work_cols.iter_mut().enumerate() {
                        wc.clear();
                        wc.extend_from_slice(frame.column_colors(col));
                    }
                    started = true;
                }
                EncodedFrame::Delta(changed) => {
                    if !started {
                        return Err(DecodeError::DeltaBeforeKey);
                    }
                    for d in changed {
                        let col = d.col as usize;
                        if col >= cols || d.occ.len() != owpc {
                            return Err(DecodeError::ColumnOutOfRange(d.col));
                        }
                        work_occ[col * owpc..(col + 1) * owpc].copy_from_slice(&d.occ);
                        work_cols[col].clear();
                        work_cols[col].extend_from_slice(&d.colors);
                    }
                }
            }
            frames.push(flatten(&work_occ, &work_cols, cols));
        }

        // Per-frame dirs from the reconstructed occupancy.
        let mut dirs = Vec::with_capacity(frames.len());
        for f in &frames {
            f.validate(self.dims).map_err(DecodeError::Frame)?;
            dirs.push(frame_dirs(f, self.dims, owpc));
        }

        let durations = if self.durations.is_empty() {
            vec![self.default_frame_ms; frames.len()]
        } else {
            self.durations.clone()
        };

        Ok(DecodedClip {
            dims: self.dims,
            pivot: self.pivot,
            voxel_world_size: self.voxel_world_size,
            occ_words_per_col: owpc as u32,
            loop_mode: self.loop_mode,
            frames,
            dirs,
            durations,
        })
    }

    /// Serialise to the `.rvc` byte form. Round-trips byte-equally with
    /// [`VoxelClip::parse`].
    #[must_use]
    pub fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());

        write_chunk(&mut out, TAG_META, |b| {
            for v in self.dims {
                b.extend_from_slice(&v.to_le_bytes());
            }
            for v in self.pivot {
                b.extend_from_slice(&v.to_le_bytes());
            }
            b.extend_from_slice(&self.voxel_world_size.to_le_bytes());
            b.push(self.loop_mode.to_u8());
            b.extend_from_slice(&self.default_frame_ms.to_le_bytes());
            let fc = u32::try_from(self.frames.len()).expect("frame count fits u32");
            b.extend_from_slice(&fc.to_le_bytes());
        });

        write_chunk(&mut out, TAG_FRMS, |b| {
            for ef in &self.frames {
                match ef {
                    EncodedFrame::Key(frame) => {
                        b.push(FRAME_KIND_KEY);
                        write_u32_vec(b, &frame.occupancy);
                        write_u32_vec(b, &frame.color_offsets);
                        write_u32_vec(b, &frame.colors);
                    }
                    EncodedFrame::Delta(changed) => {
                        b.push(FRAME_KIND_DELTA);
                        let n = u32::try_from(changed.len()).expect("changed count fits u32");
                        b.extend_from_slice(&n.to_le_bytes());
                        for d in changed {
                            b.extend_from_slice(&d.col.to_le_bytes());
                            // occ is a fixed occ_words_per_col count → no length prefix.
                            for w in &d.occ {
                                b.extend_from_slice(&w.to_le_bytes());
                            }
                            write_u32_vec(b, &d.colors);
                        }
                    }
                }
            }
        });

        if !self.durations.is_empty() {
            write_chunk(&mut out, TAG_TIME, |b| {
                for d in &self.durations {
                    b.extend_from_slice(&d.to_le_bytes());
                }
            });
        }

        for (tag, payload) in &self.extra_chunks {
            write_chunk(&mut out, *tag, |b| b.extend_from_slice(payload));
        }

        out
    }

    /// Parse the `.rvc` byte form.
    ///
    /// # Errors
    /// [`ParseError`] for a bad magic / unsupported version / truncation /
    /// missing required chunk / malformed frame stream.
    pub fn parse(bytes: &[u8]) -> Result<VoxelClip, ParseError> {
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

        let mut meta: Option<&[u8]> = None;
        let mut frms: Option<&[u8]> = None;
        let mut time: Option<&[u8]> = None;
        let mut extra_chunks = Vec::new();
        while cur.remaining() > 0 {
            let tag_buf = cur.read_bytes(4)?;
            let tag = [tag_buf[0], tag_buf[1], tag_buf[2], tag_buf[3]];
            let len = cur.read_u32()? as usize;
            let payload = cur.read_bytes(len)?;
            match tag {
                TAG_META => meta = Some(payload),
                TAG_FRMS => frms = Some(payload),
                TAG_TIME => time = Some(payload),
                _ => extra_chunks.push((tag, payload.to_vec())),
            }
        }

        let meta = meta.ok_or(ParseError::MissingChunk(TAG_META))?;
        let frms = frms.ok_or(ParseError::MissingChunk(TAG_FRMS))?;

        let (dims, pivot, voxel_world_size, loop_mode, default_frame_ms, frame_count) =
            parse_meta(meta)?;
        let frames = parse_frms(frms, dims, frame_count)?;
        let durations = match time {
            Some(p) => parse_time(p, frame_count)?,
            None => Vec::new(),
        };

        Ok(VoxelClip {
            dims,
            pivot,
            voxel_world_size,
            loop_mode,
            default_frame_ms,
            frames,
            durations,
            extra_chunks,
        })
    }
}

// ---- decode helpers ------------------------------------------------------

/// Flatten per-column working state into a [`VoxelFrame`].
fn flatten(occ: &[u32], cols_colors: &[Vec<u32>], cols: usize) -> VoxelFrame {
    let mut color_offsets = Vec::with_capacity(cols + 1);
    let mut colors = Vec::new();
    for run in cols_colors {
        color_offsets.push(colors.len() as u32);
        colors.extend_from_slice(run);
    }
    color_offsets.push(colors.len() as u32);
    VoxelFrame {
        occupancy: occ.to_vec(),
        colors,
        color_offsets,
    }
}

/// Per-voxel `dir` (surface-normal LUT index) for every voxel of `frame`,
/// ascending-z within each column — parallel to `frame.colors`.
fn frame_dirs(frame: &VoxelFrame, dims: [u32; 3], owpc: usize) -> Vec<u32> {
    let (mx, my, mz) = (dims[0] as i64, dims[1] as i64, dims[2] as i64);
    let solid = |x: i64, y: i64, z: i64| -> bool {
        if x < 0 || y < 0 || z < 0 || x >= mx || y >= my || z >= mz {
            return false;
        }
        let col = (x + y * mx) as usize;
        let word = frame.occupancy[col * owpc + (z >> 5) as usize];
        (word >> (z & 31)) & 1 != 0
    };
    let mut dirs = Vec::with_capacity(frame.colors.len());
    for y in 0..my {
        for x in 0..mx {
            let col = (x + y * mx) as usize;
            // Walk set bits ascending z to match the colour run order.
            for z in 0..mz {
                let word = frame.occupancy[col * owpc + (z >> 5) as usize];
                if (word >> (z & 31)) & 1 != 0 {
                    let (_vis, dir) = compute_vis_dir(&solid, x, y, z);
                    dirs.push(u32::from(dir));
                }
            }
        }
    }
    dirs
}

// ---- serialize / parse helpers ------------------------------------------

fn write_chunk(out: &mut Vec<u8>, tag: [u8; 4], body: impl FnOnce(&mut Vec<u8>)) {
    out.extend_from_slice(&tag);
    let len_pos = out.len();
    out.extend_from_slice(&0u32.to_le_bytes());
    let start = out.len();
    body(out);
    let len = u32::try_from(out.len() - start).expect("chunk payload length fits u32");
    out[len_pos..len_pos + 4].copy_from_slice(&len.to_le_bytes());
}

/// Length-prefixed (`u32`) array of `u32`s.
fn write_u32_vec(out: &mut Vec<u8>, v: &[u32]) {
    let n = u32::try_from(v.len()).expect("u32 array length fits u32");
    out.extend_from_slice(&n.to_le_bytes());
    for w in v {
        out.extend_from_slice(&w.to_le_bytes());
    }
}

fn read_u32_vec(cur: &mut Cursor) -> Result<Vec<u32>, ParseError> {
    let n = cur.read_u32()? as usize;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        v.push(cur.read_u32()?);
    }
    Ok(v)
}

#[allow(clippy::type_complexity)]
fn parse_meta(payload: &[u8]) -> Result<([u32; 3], [f32; 3], f32, LoopMode, u32, u32), ParseError> {
    let mut cur = Cursor::new(payload);
    let dims = [cur.read_u32()?, cur.read_u32()?, cur.read_u32()?];
    let pivot = [cur.read_f32()?, cur.read_f32()?, cur.read_f32()?];
    let voxel_world_size = cur.read_f32()?;
    let loop_mode = LoopMode::from_u8(cur.read_u8()?).ok_or(ParseError::BadLoopMode)?;
    let default_frame_ms = cur.read_u32()?;
    let frame_count = cur.read_u32()?;
    Ok((dims, pivot, voxel_world_size, loop_mode, default_frame_ms, frame_count))
}

fn parse_frms(
    payload: &[u8],
    dims: [u32; 3],
    frame_count: u32,
) -> Result<Vec<EncodedFrame>, ParseError> {
    let owpc = occ_words_per_col(dims) as usize;
    let mut cur = Cursor::new(payload);
    let mut frames = Vec::with_capacity(frame_count as usize);
    for _ in 0..frame_count {
        let kind = cur.read_u8()?;
        match kind {
            FRAME_KIND_KEY => {
                let occupancy = read_u32_vec(&mut cur)?;
                let color_offsets = read_u32_vec(&mut cur)?;
                let colors = read_u32_vec(&mut cur)?;
                frames.push(EncodedFrame::Key(VoxelFrame {
                    occupancy,
                    colors,
                    color_offsets,
                }));
            }
            FRAME_KIND_DELTA => {
                let n = cur.read_u32()? as usize;
                let mut changed = Vec::with_capacity(n);
                for _ in 0..n {
                    let col = cur.read_u32()?;
                    let mut occ = Vec::with_capacity(owpc);
                    for _ in 0..owpc {
                        occ.push(cur.read_u32()?);
                    }
                    let colors = read_u32_vec(&mut cur)?;
                    changed.push(ColumnDelta { col, occ, colors });
                }
                frames.push(EncodedFrame::Delta(changed));
            }
            other => return Err(ParseError::BadFrameKind(other)),
        }
    }
    Ok(frames)
}

fn parse_time(payload: &[u8], frame_count: u32) -> Result<Vec<u32>, ParseError> {
    let mut cur = Cursor::new(payload);
    let mut durations = Vec::with_capacity(frame_count as usize);
    for _ in 0..frame_count {
        durations.push(cur.read_u32()?);
    }
    Ok(durations)
}

// ---- errors --------------------------------------------------------------

/// Why a [`VoxelFrame`] failed validation against a clip's `dims`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameError {
    OccupancyLen,
    OffsetsLen,
    OffsetsBounds,
    OffsetsMonotonic,
    /// Column index whose occupancy popcount ≠ its colour-run length.
    OccupancyColorMismatch(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseError {
    BadMagic { got: [u8; 4] },
    UnsupportedVersion(u16),
    Truncated,
    MissingChunk([u8; 4]),
    BadLoopMode,
    BadFrameKind(u8),
}

impl From<OutOfBounds> for ParseError {
    fn from(_: OutOfBounds) -> Self {
        ParseError::Truncated
    }
}

/// Why [`VoxelClip::decode`] failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    /// The frame stream began with a delta frame.
    DeltaBeforeKey,
    /// A delta named a column ≥ `dims[0]*dims[1]` or wrong occ length.
    ColumnOutOfRange(u32),
    /// A reconstructed frame failed validation.
    Frame(FrameError),
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a full frame from a dense `solid(x,y,z) -> Option<color>`
    /// closure (the authoring shape demiurg / the encoder will use).
    fn frame_from_fn(
        dims: [u32; 3],
        fill: impl Fn(u32, u32, u32) -> Option<u32>,
    ) -> VoxelFrame {
        let owpc = occ_words_per_col(dims) as usize;
        let cols = (dims[0] as usize) * (dims[1] as usize);
        let mut occupancy = vec![0u32; cols * owpc];
        let mut color_offsets = vec![0u32; cols + 1];
        let mut colors = Vec::new();
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                let col = (x + y * dims[0]) as usize;
                color_offsets[col] = colors.len() as u32;
                for z in 0..dims[2] {
                    if let Some(c) = fill(x, y, z) {
                        occupancy[col * owpc + (z >> 5) as usize] |= 1u32 << (z & 31);
                        colors.push(c);
                    }
                }
            }
        }
        color_offsets[cols] = colors.len() as u32;
        VoxelFrame {
            occupancy,
            colors,
            color_offsets,
        }
    }

    /// A small flame-ish clip: a flickering blob whose top voxel toggles
    /// per frame (so most columns are static, a few change — the diff
    /// codec's target).
    fn flame_clip(dims: [u32; 3], n_frames: u32, keyframe_interval: u32) -> (VoxelClip, Vec<VoxelFrame>) {
        let frames: Vec<VoxelFrame> = (0..n_frames)
            .map(|fi| {
                frame_from_fn(dims, |x, y, z| {
                    let cx = dims[0] / 2;
                    let cy = dims[1] / 2;
                    // a static stem in the centre column
                    let stem = x == cx && y == cy && z < dims[2] - 2;
                    // a flickering tip whose height depends on the frame
                    let tip = x == cx && y == cy && z == dims[2] - 2 - (fi % 2);
                    if stem || tip {
                        Some(0x80FF_8000 | (fi & 0xF)) // vary low bits per frame
                    } else {
                        None
                    }
                })
            })
            .collect();
        let clip = VoxelClip::from_frames(
            dims,
            [dims[0] as f32 * 0.5, dims[1] as f32 * 0.5, dims[2] as f32 * 0.5],
            1.0,
            LoopMode::Loop,
            &frames,
            &[],
            33,
            keyframe_interval,
        );
        (clip, frames)
    }

    #[test]
    fn occ_words_per_col_matches_sprite_model() {
        assert_eq!(occ_words_per_col([8, 8, 1]), 1);
        assert_eq!(occ_words_per_col([8, 8, 32]), 1);
        assert_eq!(occ_words_per_col([8, 8, 33]), 2);
        assert_eq!(occ_words_per_col([8, 8, 256]), 8);
    }

    #[test]
    fn frame_validate_catches_mismatch() {
        let dims = [4, 4, 8];
        let mut f = frame_from_fn(dims, |x, y, z| (x == 0 && y == 0 && z < 3).then_some(0x8000_00FF));
        assert!(f.validate(dims).is_ok());
        // Corrupt column 0: clear one occupancy bit but keep its colour run
        // (popcount 2 ≠ run 3).
        f.occupancy[0] &= !1u32;
        assert!(matches!(
            f.validate(dims),
            Err(FrameError::OccupancyColorMismatch(0))
        ));
    }

    #[test]
    fn decode_reconstructs_every_frame() {
        let dims = [9, 9, 40];
        let (clip, original) = flame_clip(dims, 8, 4);
        let decoded = clip.decode().expect("decode");
        assert_eq!(decoded.frame_count(), original.len());
        for (i, (got, want)) in decoded.frames.iter().zip(&original).enumerate() {
            assert_eq!(got, want, "frame {i} mismatch");
            // dirs are parallel to colours.
            assert_eq!(decoded.dirs[i].len(), got.colors.len(), "frame {i} dirs len");
        }
    }

    #[test]
    fn diff_frames_are_smaller_than_keyframes() {
        let dims = [9, 9, 40];
        let (clip, _) = flame_clip(dims, 8, 0); // only frame 0 is a key
        let keys = clip
            .frames
            .iter()
            .filter(|f| matches!(f, EncodedFrame::Key(_)))
            .count();
        assert_eq!(keys, 1, "keyframe_interval=0 ⇒ exactly one keyframe");
        // Every non-key frame touches only a handful of columns (the tip),
        // far fewer than the dims[0]*dims[1] columns a keyframe rewrites.
        for f in &clip.frames {
            if let EncodedFrame::Delta(changed) = f {
                assert!(
                    changed.len() < (dims[0] * dims[1]) as usize,
                    "delta should be sparse, got {} columns",
                    changed.len()
                );
            }
        }
    }

    #[test]
    fn serialize_parse_round_trips() {
        let dims = [9, 9, 40];
        let (clip, _) = flame_clip(dims, 8, 4);
        let bytes = clip.serialize();
        let parsed = VoxelClip::parse(&bytes).expect("parse");
        assert_eq!(parsed, clip);
        // Re-serialise is byte-identical.
        assert_eq!(parsed.serialize(), bytes);
        // And it still decodes to the same frames.
        let a = clip.decode().expect("decode a");
        let b = parsed.decode().expect("decode b");
        assert_eq!(a.frames, b.frames);
    }

    #[test]
    fn durations_default_when_time_chunk_absent() {
        let dims = [4, 4, 8];
        let (clip, _) = flame_clip(dims, 4, 2);
        assert!(clip.durations.is_empty());
        let decoded = clip.decode().expect("decode");
        assert_eq!(decoded.durations, vec![33; 4]);
        assert_eq!(decoded.total_ms(), 33 * 4);
    }

    #[test]
    fn explicit_durations_round_trip() {
        let dims = [4, 4, 8];
        let frames: Vec<VoxelFrame> = (0..3)
            .map(|fi| frame_from_fn(dims, move |x, y, z| (x == 0 && y == 0 && z == fi).then_some(0x8011_2233)))
            .collect();
        let clip = VoxelClip::from_frames(
            dims,
            [0.0; 3],
            1.0,
            LoopMode::Once,
            &frames,
            &[10, 20, 30],
            33,
            0,
        );
        let parsed = VoxelClip::parse(&clip.serialize()).expect("parse");
        assert_eq!(parsed.durations, vec![10, 20, 30]);
        assert_eq!(parsed.decode().unwrap().durations, vec![10, 20, 30]);
        assert_eq!(parsed.loop_mode, LoopMode::Once);
    }

    #[test]
    fn unknown_chunks_preserved() {
        let dims = [4, 4, 8];
        let (mut clip, _) = flame_clip(dims, 2, 0);
        clip.extra_chunks.push((*b"XTRA", vec![1, 2, 3, 4, 5]));
        let parsed = VoxelClip::parse(&clip.serialize()).expect("parse");
        assert_eq!(parsed.extra_chunks, vec![(*b"XTRA", vec![1, 2, 3, 4, 5])]);
    }

    #[test]
    fn bad_magic_and_version_rejected() {
        let dims = [4, 4, 8];
        let (clip, _) = flame_clip(dims, 2, 0);
        let mut bytes = clip.serialize();
        let good = bytes.clone();
        bytes[0] = b'X';
        assert!(matches!(VoxelClip::parse(&bytes), Err(ParseError::BadMagic { .. })));
        let mut v = good.clone();
        v[4] = 9; // version low byte
        assert!(matches!(
            VoxelClip::parse(&v),
            Err(ParseError::UnsupportedVersion(_))
        ));
    }

    #[test]
    fn delta_before_key_rejected() {
        let dims = [4, 4, 8];
        let clip = VoxelClip {
            dims,
            pivot: [0.0; 3],
            voxel_world_size: 1.0,
            loop_mode: LoopMode::Loop,
            default_frame_ms: 33,
            frames: vec![EncodedFrame::Delta(Vec::new())],
            durations: Vec::new(),
            extra_chunks: Vec::new(),
        };
        assert!(matches!(clip.decode(), Err(DecodeError::DeltaBeforeKey)));
    }
}
