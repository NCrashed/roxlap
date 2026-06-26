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
//! version u16 = 2
//! chunks  [tag(4) | flags(u8) | len(u32) | payload]  until EOF; unknown
//!         tags preserved. flags bit0 = payload is raw-deflated, stored as
//!         raw_len(u32) | deflate_bytes (and `len` counts that). Each chunk
//!         is deflated only when it shrinks; small ones stay raw.
//!   META : dims[3] u32, pivot[3] f32, voxel_world_size f32,
//!          loop_mode u8, default_frame_ms u32, frame_count u32
//!   FRMS : per frame: kind u8 {Key=0, Delta=1}; Key = full frame
//!          (occupancy + color_offsets + colors, each u32-len-prefixed);
//!          Delta = changed_count u32 + per changed column
//!          (col u32, occ_words_per_col × u32, color_run len+u32s)
//!   TIME : optional per-frame durations (frame_count × u32 ms)
//! ```
//!
//! Compression is per-chunk deflate (`miniz_oxide`): the occupancy
//! bitmasks + colour runs compress well, while `META` / small chunks stay
//! raw. **v1** (no `flags` byte, every payload raw) still parses.

use crate::bytes::{Cursor, OutOfBounds};
use crate::kv6::{compute_vis_dir, Kv6, Voxel};

const MAGIC: [u8; 4] = *b"RVCL";
/// Current on-disk version. v2 adds a per-chunk `flags` byte (deflate).
const VERSION: u16 = 2;
/// v1 had no per-chunk `flags` byte and stored every payload raw; still
/// readable.
const VERSION_LEGACY: u16 = 1;

const TAG_META: [u8; 4] = *b"META";
const TAG_FRMS: [u8; 4] = *b"FRMS";
const TAG_TIME: [u8; 4] = *b"TIME";

/// Chunk `flags` bit: the payload is raw-deflated (`raw_len(u32) | data`).
const CHUNK_FLAG_DEFLATED: u8 = 0x01;
/// miniz_oxide deflate level for `.rvc` writes (clips are written once,
/// read often — favour ratio over encode speed, but level 10 is overkill).
const DEFLATE_LEVEL: u8 = 8;

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
    /// Build one dense-column [`VoxelFrame`] from a `.kv6` model — the
    /// authoring bridge from a voxel sprite to a clip frame. The frame's
    /// dims are the kv6's `[xsiz, ysiz, zsiz]`; the kv6's pivot +
    /// voxel-world-size travel at the clip level (see
    /// [`VoxelClip::from_kv6_frames`]).
    ///
    /// `.kv6` already stores surface voxels per `(x, y)` column in
    /// ascending z — the very layout a frame wants — so this is a re-index
    /// from the kv6's x-major column order (`x · ysiz + y`) to the frame's
    /// `col = x + y · xsiz`, packing each column's z's into the occupancy
    /// bitmask. Each column is sorted by z so the colour run is ascending
    /// even if the source isn't strictly ordered; voxels with `z >= zsiz`
    /// are dropped (defensive against malformed input).
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn from_kv6(kv6: &Kv6) -> Self {
        let dims = [kv6.xsiz, kv6.ysiz, kv6.zsiz];
        let (nx, ny) = (dims[0] as usize, dims[1] as usize);
        let cols = nx * ny;
        let owpc = occ_words_per_col(dims) as usize;
        let zmax = dims[2];

        // Bucket the kv6's flat voxel stream into the frame's column index.
        let mut per_col: Vec<Vec<(u16, u32)>> = vec![Vec::new(); cols];
        let mut vi = 0usize;
        for x in 0..nx {
            for (y, &cnt) in kv6.ylen[x].iter().enumerate() {
                let count = cnt as usize;
                let col = x + y * nx; // frame ordering (x-fastest)
                for v in &kv6.voxels[vi..vi + count] {
                    if u32::from(v.z) < zmax {
                        per_col[col].push((v.z, v.col));
                    }
                }
                vi += count;
            }
        }

        let mut occupancy = vec![0u32; cols * owpc];
        let mut colors = Vec::new();
        let mut color_offsets = Vec::with_capacity(cols + 1);
        color_offsets.push(0u32);
        for (col, run) in per_col.iter_mut().enumerate() {
            run.sort_by_key(|&(z, _)| z);
            for &(z, c) in run.iter() {
                let zi = z as usize;
                occupancy[col * owpc + zi / 32] |= 1u32 << (zi % 32);
                colors.push(c);
            }
            color_offsets.push(colors.len() as u32);
        }

        Self {
            occupancy,
            colors,
            color_offsets,
        }
    }

    /// Inverse of [`from_kv6`](Self::from_kv6): materialise this frame as a
    /// flat-lit `.kv6` model (every voxel `vis = 63`, `dir = 0` — voxel
    /// clips render full-bright, so per-face normals are unused). `dims` is
    /// the clip bounding box, `pivot` becomes the kv6 pivot. Lets a single
    /// streaming-clip frame drive `add_sprite_model` / `refresh_sprite_model`
    /// (one model re-uploaded per frame) instead of an N-frame flipbook.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn to_kv6(&self, dims: [u32; 3], pivot: [f32; 3]) -> Kv6 {
        let (nx, ny) = (dims[0] as usize, dims[1] as usize);
        let owpc = occ_words_per_col(dims) as usize;
        let mut voxels = Vec::new();
        let mut xlen = Vec::with_capacity(nx);
        let mut ylen = Vec::with_capacity(nx);

        // `.kv6` walks x-major, then y, then ascending z; the frame's column
        // index is x-fastest (`col = x + y·nx`), so re-index back.
        for x in 0..nx {
            let mut col_counts: Vec<u16> = Vec::with_capacity(ny);
            let mut xcount = 0u32;
            for y in 0..ny {
                let col = x + y * nx;
                let run = self.column_colors(col);
                let occ = &self.occupancy[col * owpc..(col + 1) * owpc];
                let before = voxels.len();
                let mut ci = 0usize;
                for z in 0..dims[2] {
                    if (occ[(z >> 5) as usize] >> (z & 31)) & 1 != 0 {
                        voxels.push(Voxel {
                            col: run[ci],
                            z: z as u16,
                            vis: 63,
                            dir: 0,
                        });
                        ci += 1;
                    }
                }
                let c = (voxels.len() - before) as u16;
                col_counts.push(c);
                xcount += u32::from(c);
            }
            xlen.push(xcount);
            ylen.push(col_counts);
        }

        Kv6 {
            xsiz: dims[0],
            ysiz: dims[1],
            zsiz: dims[2],
            xpiv: pivot[0],
            ypiv: pivot[1],
            zpiv: pivot[2],
            voxels,
            xlen,
            ylen,
            palette: None,
        }
    }

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
        if self.color_offsets[0] != 0
            || *self.color_offsets.last().unwrap() as usize != self.colors.len()
        {
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

    /// The frame index to show after `elapsed_ms` of playback, honouring
    /// the clip's [`LoopMode`] and per-frame durations. Pure — the host
    /// (or the facade's clip-instance clocks) drives `set_clip_instance_frame`
    /// from this. Empty clip ⇒ `0`.
    ///
    /// - [`LoopMode::Loop`]: wraps modulo the total length.
    /// - [`LoopMode::Once`]: holds the last frame past the end.
    /// - [`LoopMode::PingPong`]: bounces `0→N-1→0` over `2·total`.
    #[must_use]
    pub fn frame_at(&self, elapsed_ms: u32) -> usize {
        frame_at(&self.durations, self.loop_mode, elapsed_ms)
    }
}

/// The frame index to show after `elapsed_ms` of playback, given per-frame
/// `durations` (ms) + a [`LoopMode`] — the pure playback math behind
/// [`DecodedClip::frame_at`], usable on its own so a per-instance clock
/// (e.g. a character clip attachment, VCL.6) can resolve a frame without
/// holding the whole [`DecodedClip`]. Empty / single-frame ⇒ `0`.
///
/// - [`LoopMode::Loop`]: wraps modulo the total length.
/// - [`LoopMode::Once`]: holds the last frame past the end.
/// - [`LoopMode::PingPong`]: bounces `0→N-1→0` over `2·total`.
#[must_use]
pub fn frame_at(durations: &[u32], loop_mode: LoopMode, elapsed_ms: u32) -> usize {
    let n = durations.len();
    if n <= 1 {
        return 0;
    }
    let total: u32 = durations.iter().copied().sum();
    if total == 0 {
        return 0;
    }
    // Position within one forward pass (after applying the loop mode).
    let t = match loop_mode {
        LoopMode::Loop => elapsed_ms % total,
        LoopMode::Once => elapsed_ms.min(total - 1),
        LoopMode::PingPong => {
            let p = elapsed_ms % (2 * total);
            if p < total {
                p
            } else {
                // Mirror the second half back: 2·total-1 → ~0.
                2 * total - 1 - p
            }
        }
    };
    // Walk the duration prefix sums to find the frame holding `t`.
    let mut acc = 0u32;
    for (i, &d) in durations.iter().enumerate() {
        acc += d;
        if t < acc {
            return i;
        }
    }
    n - 1
}

/// u32 occupancy words per `(x, y)` column for a clip of `dims`.
#[must_use]
pub fn occ_words_per_col(dims: [u32; 3]) -> u32 {
    dims[2].div_ceil(32).max(1)
}

/// A seekable, **O(1-frame)-memory** cursor over a [`VoxelClip`]'s I/P
/// stream — the streaming alternative to [`DecodedClip`], which
/// materialises *every* frame (and which the GPU/CPU flipbook then holds N
/// volumes for). For a huge clip this keeps one reconstructed frame plus
/// the compact encoded stream instead of N full frames.
///
/// Seeking to a frame replays deltas from the nearest preceding keyframe;
/// stepping forward from the current frame is incremental. Drive it from
/// [`frame_at`] like the flipbook, then rebuild a single sprite model from
/// [`current_frame`](Self::current_frame) (+ [`current_dirs`](Self::current_dirs)
/// for the GPU) each time the frame changes — e.g. via
/// `roxlap_core::SpriteDense::from_voxel_frame` or
/// `SceneRenderer::refresh_sprite_model`.
#[derive(Debug, Clone)]
pub struct StreamingClip {
    dims: [u32; 3],
    pivot: [f32; 3],
    voxel_world_size: f32,
    loop_mode: LoopMode,
    owpc: usize,
    cols: usize,
    /// Owned copy of the encoded I/P stream (the compact representation).
    frames: Vec<EncodedFrame>,
    durations: Vec<u32>,
    /// Ascending indices of the keyframes in `frames` (the seek points).
    keyframes: Vec<usize>,
    // --- cursor state ---
    work_occ: Vec<u32>,
    work_cols: Vec<Vec<u32>>,
    /// Frame index currently reconstructed in the working set.
    current: usize,
    cur_frame: VoxelFrame,
    cur_dirs: Vec<u32>,
}

impl StreamingClip {
    /// Build a streaming cursor over `clip` and reconstruct frame 0.
    ///
    /// # Errors
    /// [`DecodeError::DeltaBeforeKey`] if the stream is empty or doesn't
    /// start with a keyframe; otherwise the same per-frame errors as
    /// [`VoxelClip::decode`] (surfaced lazily while seeking).
    pub fn new(clip: &VoxelClip) -> Result<Self, DecodeError> {
        if !matches!(clip.frames.first(), Some(EncodedFrame::Key(_))) {
            return Err(DecodeError::DeltaBeforeKey);
        }
        let owpc = occ_words_per_col(clip.dims) as usize;
        let cols = (clip.dims[0] as usize) * (clip.dims[1] as usize);
        let keyframes = clip
            .frames
            .iter()
            .enumerate()
            .filter_map(|(i, f)| matches!(f, EncodedFrame::Key(_)).then_some(i))
            .collect();
        let durations = if clip.durations.is_empty() {
            vec![clip.default_frame_ms; clip.frames.len()]
        } else {
            clip.durations.clone()
        };
        let mut s = Self {
            dims: clip.dims,
            pivot: clip.pivot,
            voxel_world_size: clip.voxel_world_size,
            loop_mode: clip.loop_mode,
            owpc,
            cols,
            frames: clip.frames.clone(),
            durations,
            keyframes,
            work_occ: vec![0u32; cols * owpc],
            work_cols: vec![Vec::new(); cols],
            current: 0,
            cur_frame: VoxelFrame {
                occupancy: Vec::new(),
                colors: Vec::new(),
                color_offsets: Vec::new(),
            },
            cur_dirs: Vec::new(),
        };
        s.reconstruct(0)?;
        Ok(s)
    }

    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }
    #[must_use]
    pub fn dims(&self) -> [u32; 3] {
        self.dims
    }
    #[must_use]
    pub fn pivot(&self) -> [f32; 3] {
        self.pivot
    }
    #[must_use]
    pub fn voxel_world_size(&self) -> f32 {
        self.voxel_world_size
    }
    #[must_use]
    pub fn loop_mode(&self) -> LoopMode {
        self.loop_mode
    }
    #[must_use]
    pub fn durations(&self) -> &[u32] {
        &self.durations
    }
    /// Frame index currently reconstructed.
    #[must_use]
    pub fn current_index(&self) -> usize {
        self.current
    }
    /// The currently-reconstructed frame.
    #[must_use]
    pub fn current_frame(&self) -> &VoxelFrame {
        &self.cur_frame
    }
    /// Per-voxel `dir` LUT indices for the current frame, parallel to
    /// `current_frame().colors` (for GPU sprite-model upload).
    #[must_use]
    pub fn current_dirs(&self) -> &[u32] {
        &self.cur_dirs
    }

    /// Seek to `frame` (clamped to the last frame) and return the
    /// reconstructed [`VoxelFrame`]. Forward seeks step incrementally;
    /// backward / random seeks replay from the nearest preceding keyframe.
    ///
    /// # Errors
    /// Per-frame [`DecodeError`]s from a malformed stream (out-of-range
    /// delta column, invalid reconstructed frame).
    pub fn seek(&mut self, frame: usize) -> Result<&VoxelFrame, DecodeError> {
        let target = frame.min(self.frame_count() - 1);
        if target != self.current || self.cur_frame.occupancy.is_empty() {
            self.reconstruct(target)?;
        }
        Ok(&self.cur_frame)
    }

    /// Largest keyframe index `<= target` (frame 0 is always a keyframe).
    fn keyframe_at_or_before(&self, target: usize) -> usize {
        let pp = self.keyframes.partition_point(|&k| k <= target);
        self.keyframes[pp - 1]
    }

    /// Rebuild the working set + materialised frame/dirs at `target`.
    fn reconstruct(&mut self, target: usize) -> Result<(), DecodeError> {
        // Step forward from the current frame when possible; otherwise reset
        // the working set to the nearest preceding keyframe and replay.
        let start = if target > self.current && !self.cur_frame.occupancy.is_empty() {
            self.current + 1
        } else {
            let kf = self.keyframe_at_or_before(target);
            let mut started = false;
            apply_frame(
                &self.frames[kf],
                &mut self.work_occ,
                &mut self.work_cols,
                self.dims,
                self.owpc,
                self.cols,
                &mut started,
            )?;
            kf + 1
        };
        let mut started = true;
        for i in start..=target {
            // Disjoint field borrows: `frames` (read) vs the working set.
            let ef = &self.frames[i];
            apply_frame(
                ef,
                &mut self.work_occ,
                &mut self.work_cols,
                self.dims,
                self.owpc,
                self.cols,
                &mut started,
            )?;
        }
        self.current = target;
        self.cur_frame = flatten(&self.work_occ, &self.work_cols, self.cols);
        self.cur_frame
            .validate(self.dims)
            .map_err(DecodeError::Frame)?;
        self.cur_dirs = frame_dirs(&self.cur_frame, self.dims, self.owpc);
        Ok(())
    }
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

    /// Build a clip from a sequence of `.kv6` frames sharing one
    /// `[xsiz, ysiz, zsiz]` — the authoring path from animated voxel
    /// sprites to a `.rvc` clip. Each kv6 becomes a [`VoxelFrame`] via
    /// [`VoxelFrame::from_kv6`], then the lot is encoded with
    /// [`VoxelClip::from_frames`] (frame 0 + every `keyframe_interval`-th a
    /// keyframe; `0` ⇒ only frame 0). The pivot comes from the first
    /// frame's kv6; `voxel_world_size` is the render scale (`.kv6` carries
    /// none). `durations` is per-frame ms (empty ⇒ uniform
    /// `default_frame_ms`).
    ///
    /// # Errors
    /// [`Kv6ImportError::Empty`] if `frames` is empty;
    /// [`Kv6ImportError::DimsMismatch`] if any frame's dims differ from the
    /// first (clips are fixed-bbox).
    pub fn from_kv6_frames(
        frames: &[Kv6],
        voxel_world_size: f32,
        loop_mode: LoopMode,
        durations: &[u32],
        default_frame_ms: u32,
        keyframe_interval: u32,
    ) -> Result<Self, Kv6ImportError> {
        let Some(first) = frames.first() else {
            return Err(Kv6ImportError::Empty);
        };
        let dims = [first.xsiz, first.ysiz, first.zsiz];
        for (i, k) in frames.iter().enumerate() {
            let d = [k.xsiz, k.ysiz, k.zsiz];
            if d != dims {
                return Err(Kv6ImportError::DimsMismatch {
                    frame: i,
                    dims: d,
                    expected: dims,
                });
            }
        }
        let pivot = [first.xpiv, first.ypiv, first.zpiv];
        let vframes: Vec<VoxelFrame> = frames.iter().map(VoxelFrame::from_kv6).collect();
        Ok(Self::from_frames(
            dims,
            pivot,
            voxel_world_size,
            loop_mode,
            &vframes,
            durations,
            default_frame_ms,
            keyframe_interval,
        ))
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
            apply_frame(
                ef,
                &mut work_occ,
                &mut work_cols,
                self.dims,
                owpc,
                cols,
                &mut started,
            )?;
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
        if version != VERSION && version != VERSION_LEGACY {
            return Err(ParseError::UnsupportedVersion(version));
        }
        // v2 prefixes each chunk payload with a `flags` byte; v1 doesn't.
        let has_flags = version >= VERSION;

        let mut meta: Option<Vec<u8>> = None;
        let mut frms: Option<Vec<u8>> = None;
        let mut time: Option<Vec<u8>> = None;
        let mut extra_chunks = Vec::new();
        while cur.remaining() > 0 {
            let tag_buf = cur.read_bytes(4)?;
            let tag = [tag_buf[0], tag_buf[1], tag_buf[2], tag_buf[3]];
            let flags = if has_flags { cur.read_u8()? } else { 0 };
            let len = cur.read_u32()? as usize;
            let stored = cur.read_bytes(len)?;
            let payload = if flags & CHUNK_FLAG_DEFLATED != 0 {
                inflate_chunk(stored)?
            } else {
                stored.to_vec()
            };
            match tag {
                TAG_META => meta = Some(payload),
                TAG_FRMS => frms = Some(payload),
                TAG_TIME => time = Some(payload),
                _ => extra_chunks.push((tag, payload)),
            }
        }

        let meta = meta.ok_or(ParseError::MissingChunk(TAG_META))?;
        let frms = frms.ok_or(ParseError::MissingChunk(TAG_FRMS))?;

        let (dims, pivot, voxel_world_size, loop_mode, default_frame_ms, frame_count) =
            parse_meta(&meta)?;
        let frames = parse_frms(&frms, dims, frame_count)?;
        let durations = match time {
            Some(p) => parse_time(&p, frame_count)?,
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

/// Apply one I/P frame to the per-column working set (`work_occ` +
/// `work_cols`): a keyframe overwrites the whole set, a delta rewrites only
/// its changed columns. Shared by [`VoxelClip::decode`] and
/// [`StreamingClip`]. `started` guards against a leading delta.
fn apply_frame(
    ef: &EncodedFrame,
    work_occ: &mut [u32],
    work_cols: &mut [Vec<u32>],
    dims: [u32; 3],
    owpc: usize,
    cols: usize,
    started: &mut bool,
) -> Result<(), DecodeError> {
    match ef {
        EncodedFrame::Key(frame) => {
            frame.validate(dims).map_err(DecodeError::Frame)?;
            work_occ.copy_from_slice(&frame.occupancy);
            for (col, wc) in work_cols.iter_mut().enumerate() {
                wc.clear();
                wc.extend_from_slice(frame.column_colors(col));
            }
            *started = true;
        }
        EncodedFrame::Delta(changed) => {
            if !*started {
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
    Ok(())
}

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

/// Write a v2 chunk: `tag(4) | flags(u8) | len(u32) | payload`. The body is
/// built into a scratch buffer, deflated, and stored compressed
/// (`CHUNK_FLAG_DEFLATED`, payload = `raw_len(u32) | deflate_bytes`) only
/// when that actually shrinks it — small/incompressible chunks stay raw.
fn write_chunk(out: &mut Vec<u8>, tag: [u8; 4], body: impl FnOnce(&mut Vec<u8>)) {
    let mut raw = Vec::new();
    body(&mut raw);
    out.extend_from_slice(&tag);

    let deflated = miniz_oxide::deflate::compress_to_vec(&raw, DEFLATE_LEVEL);
    // `+4` accounts for the raw-length prefix a deflated payload carries.
    if deflated.len() + 4 < raw.len() {
        out.push(CHUNK_FLAG_DEFLATED);
        let len = u32::try_from(deflated.len() + 4).expect("chunk length fits u32");
        out.extend_from_slice(&len.to_le_bytes());
        let raw_len = u32::try_from(raw.len()).expect("raw length fits u32");
        out.extend_from_slice(&raw_len.to_le_bytes());
        out.extend_from_slice(&deflated);
    } else {
        out.push(0);
        let len = u32::try_from(raw.len()).expect("chunk length fits u32");
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&raw);
    }
}

/// Inflate a `CHUNK_FLAG_DEFLATED` payload (`raw_len(u32) | deflate_bytes`).
/// The stored `raw_len` bounds the output (decompression-bomb guard) and is
/// checked against the actual inflated length.
fn inflate_chunk(payload: &[u8]) -> Result<Vec<u8>, ParseError> {
    if payload.len() < 4 {
        return Err(ParseError::BadDeflate);
    }
    let raw_len = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;
    let out = miniz_oxide::inflate::decompress_to_vec_with_limit(&payload[4..], raw_len)
        .map_err(|_| ParseError::BadDeflate)?;
    if out.len() != raw_len {
        return Err(ParseError::BadDeflate);
    }
    Ok(out)
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
    Ok((
        dims,
        pivot,
        voxel_world_size,
        loop_mode,
        default_frame_ms,
        frame_count,
    ))
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

/// Why [`VoxelClip::from_kv6_frames`] could not build a clip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kv6ImportError {
    /// No frames were supplied.
    Empty,
    /// A frame's dims differ from the first frame's (clips are fixed-bbox).
    DimsMismatch {
        frame: usize,
        dims: [u32; 3],
        expected: [u32; 3],
    },
}

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
    BadMagic {
        got: [u8; 4],
    },
    UnsupportedVersion(u16),
    Truncated,
    MissingChunk([u8; 4]),
    BadLoopMode,
    BadFrameKind(u8),
    /// A `CHUNK_FLAG_DEFLATED` payload failed to inflate, or its inflated
    /// length disagreed with the stored `raw_len`.
    BadDeflate,
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
    fn frame_from_fn(dims: [u32; 3], fill: impl Fn(u32, u32, u32) -> Option<u32>) -> VoxelFrame {
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
    fn flame_clip(
        dims: [u32; 3],
        n_frames: u32,
        keyframe_interval: u32,
    ) -> (VoxelClip, Vec<VoxelFrame>) {
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
            [
                dims[0] as f32 * 0.5,
                dims[1] as f32 * 0.5,
                dims[2] as f32 * 0.5,
            ],
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
        let mut f = frame_from_fn(dims, |x, y, z| {
            (x == 0 && y == 0 && z < 3).then_some(0x8000_00FF)
        });
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
            assert_eq!(
                decoded.dirs[i].len(),
                got.colors.len(),
                "frame {i} dirs len"
            );
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
            .map(|fi| {
                frame_from_fn(dims, move |x, y, z| {
                    (x == 0 && y == 0 && z == fi).then_some(0x8011_2233)
                })
            })
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
        assert!(matches!(
            VoxelClip::parse(&bytes),
            Err(ParseError::BadMagic { .. })
        ));
        let mut v = good.clone();
        v[4] = 9; // version low byte
        assert!(matches!(
            VoxelClip::parse(&v),
            Err(ParseError::UnsupportedVersion(_))
        ));
    }

    #[test]
    fn frame_at_honours_loop_modes() {
        // 3 frames, 10 ms each (total 30).
        let dims = [4, 4, 8];
        let frames: Vec<VoxelFrame> = (0..3)
            .map(|fi| {
                frame_from_fn(dims, move |x, y, z| {
                    (x == 0 && y == 0 && z == fi).then_some(0x8011_2233)
                })
            })
            .collect();
        let mk = |mode| {
            VoxelClip::from_frames(dims, [0.0; 3], 1.0, mode, &frames, &[10, 10, 10], 33, 0)
                .decode()
                .unwrap()
        };

        let loop_c = mk(LoopMode::Loop);
        assert_eq!(loop_c.frame_at(0), 0);
        assert_eq!(loop_c.frame_at(9), 0);
        assert_eq!(loop_c.frame_at(10), 1);
        assert_eq!(loop_c.frame_at(25), 2);
        assert_eq!(loop_c.frame_at(30), 0, "wraps at total");
        assert_eq!(loop_c.frame_at(45), 1);

        let once = mk(LoopMode::Once);
        assert_eq!(once.frame_at(25), 2);
        assert_eq!(once.frame_at(1000), 2, "holds the last frame");

        let ping = mk(LoopMode::PingPong);
        assert_eq!(ping.frame_at(5), 0);
        assert_eq!(ping.frame_at(25), 2);
        assert_eq!(ping.frame_at(35), 2, "mirror: 35→ frame 2");
        assert_eq!(ping.frame_at(55), 0, "mirror back to 0 near 2·total");
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

    // ---- VCL.1: .kv6 → VoxelFrame / VoxelClip import ----------------------

    /// A fill whose every solid voxel is isolated (no 6-neighbour solid),
    /// so `Kv6::from_fn` (surface-only) keeps all of them — letting the
    /// import be compared against the all-voxels `frame_from_fn` reference.
    /// Spaced on even coords; the colour encodes `(x, y, z)`.
    fn isolated_fill(x: u32, y: u32, z: u32) -> Option<u32> {
        (x % 2 == 0 && y % 2 == 0 && z % 2 == 0).then(|| 0x8000_0000 | (x << 16) | (y << 8) | z)
    }

    #[test]
    fn from_kv6_matches_dense_reference() {
        // Non-square xy exercises the x-major→x-fastest re-index; z = 41
        // (> 32) exercises the 2-word occupancy column.
        let dims = [3u32, 2, 41];
        let kv6 = Kv6::from_fn(dims[0], dims[1], dims[2], isolated_fill);
        let imported = VoxelFrame::from_kv6(&kv6);
        let expected = frame_from_fn(dims, isolated_fill);
        assert_eq!(imported, expected);
        imported.validate(dims).expect("imported frame is valid");
    }

    #[test]
    fn from_kv6_packs_z_across_word_boundary() {
        // A single 1×1 column with voxels straddling the 32-bit word split.
        let kv6 = Kv6::from_fn(1, 1, 41, |_, _, z| match z {
            0 => Some(0x80FF_0000),
            5 => Some(0x8000_FF00),
            33 => Some(0x8000_00FF),
            40 => Some(0x80FF_FF00),
            _ => None,
        });
        let f = VoxelFrame::from_kv6(&kv6);
        // owpc = 2; word0 bits 0,5; word1 bits 1 (=33-32), 8 (=40-32).
        assert_eq!(f.occupancy, vec![(1 << 0) | (1 << 5), (1 << 1) | (1 << 8)]);
        // Colours ascending z.
        assert_eq!(
            f.colors,
            vec![0x80FF_0000, 0x8000_FF00, 0x8000_00FF, 0x80FF_FF00]
        );
        assert_eq!(f.color_offsets, vec![0, 4]);
        f.validate([1, 1, 41]).expect("valid");
    }

    #[test]
    fn from_kv6_frames_round_trips_through_clip() {
        let dims = [2u32, 2, 3];
        // Two full xy layers at different z's — every voxel surface-exposed.
        let ka = Kv6::from_fn(dims[0], dims[1], dims[2], |_, _, z| {
            (z == 0).then_some(0x80FF_0000)
        });
        let kb = Kv6::from_fn(dims[0], dims[1], dims[2], |_, _, z| {
            (z == 2).then_some(0x8000_FF00)
        });
        let clip = VoxelClip::from_kv6_frames(
            &[ka.clone(), kb.clone()],
            2.0,
            LoopMode::Loop,
            &[100, 200],
            0,
            0,
        )
        .expect("import");
        assert_eq!(clip.dims, dims);
        assert_eq!(clip.voxel_world_size, 2.0);
        assert_eq!(clip.pivot, [ka.xpiv, ka.ypiv, ka.zpiv]);
        assert_eq!(clip.durations, vec![100, 200]);

        let decoded = clip.decode().expect("decode");
        assert_eq!(decoded.frames.len(), 2);
        assert_eq!(decoded.frames[0], VoxelFrame::from_kv6(&ka));
        assert_eq!(decoded.frames[1], VoxelFrame::from_kv6(&kb));
    }

    #[test]
    fn from_kv6_frames_rejects_empty() {
        let err = VoxelClip::from_kv6_frames(&[], 1.0, LoopMode::Loop, &[], 50, 0)
            .expect_err("empty must fail");
        assert_eq!(err, Kv6ImportError::Empty);
    }

    #[test]
    fn from_kv6_frames_rejects_dims_mismatch() {
        let ka = Kv6::from_fn(2, 2, 2, |_, _, z| (z == 0).then_some(0x80FF_FFFF));
        let kb = Kv6::from_fn(3, 2, 2, |_, _, z| (z == 0).then_some(0x80FF_FFFF));
        let err = VoxelClip::from_kv6_frames(&[ka, kb], 1.0, LoopMode::Loop, &[], 50, 0)
            .expect_err("mismatch must fail");
        assert_eq!(
            err,
            Kv6ImportError::DimsMismatch {
                frame: 1,
                dims: [3, 2, 2],
                expected: [2, 2, 2],
            }
        );
    }

    #[test]
    fn to_kv6_inverts_from_kv6() {
        // Solid below a per-column threshold (interior + surface voxels), a
        // z-run crossing the 32-bit word boundary, distinct colours.
        let dims = [3u32, 2, 40];
        let frame = frame_from_fn(dims, |x, y, z| {
            (z <= (x + y) * 6 + 3).then_some(0x8000_0000 | (z << 8) | (x * 16 + y))
        });
        let kv6 = frame.to_kv6(dims, [1.0, 0.5, 20.0]);
        assert_eq!([kv6.xsiz, kv6.ysiz, kv6.zsiz], dims);
        assert_eq!([kv6.xpiv, kv6.ypiv, kv6.zpiv], [1.0, 0.5, 20.0]);
        // from_kv6 ∘ to_kv6 reproduces occupancy + colours exactly.
        assert_eq!(VoxelFrame::from_kv6(&kv6), frame);
    }

    // ---- compression (v2 per-chunk deflate) -------------------------------

    #[test]
    fn compressed_clip_round_trips_and_shrinks() {
        // A fully-solid frame: every occupancy word all-set, one repeated
        // colour — maximally compressible.
        let dims = [16u32, 16, 32];
        let frame = frame_from_fn(dims, |_, _, _| Some(0x80AB_CDEF));
        let clip = VoxelClip::from_frames(
            dims,
            [8.0, 8.0, 16.0],
            1.0,
            LoopMode::Loop,
            &[frame],
            &[],
            33,
            0,
        );
        let bytes = clip.serialize();
        // The colour run alone is 16·16·32·4 = 32 KiB raw; deflate of a
        // single repeated colour collapses the whole file far under that.
        let raw_colors_bytes = (dims[0] * dims[1] * dims[2]) as usize * 4;
        assert!(
            bytes.len() < raw_colors_bytes / 4,
            "expected compression: {} serialized bytes vs {raw_colors_bytes} raw colour bytes",
            bytes.len(),
        );
        // Version is 2 and round-trips through parse byte-for-byte (deflate
        // is deterministic).
        assert_eq!(&bytes[4..6], &VERSION.to_le_bytes());
        let parsed = VoxelClip::parse(&bytes).expect("parse");
        assert_eq!(parsed, clip);
        assert_eq!(parsed.serialize(), bytes);
    }

    /// Serialize a keyframe-only clip in the pre-v2 (v1) byte form: no
    /// per-chunk `flags` byte, every payload raw.
    fn serialize_v1(clip: &VoxelClip) -> Vec<u8> {
        fn chunk(out: &mut Vec<u8>, tag: &[u8; 4], payload: &[u8]) {
            out.extend_from_slice(tag);
            out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
            out.extend_from_slice(payload);
        }
        fn u32_vec(out: &mut Vec<u8>, v: &[u32]) {
            out.extend_from_slice(&(v.len() as u32).to_le_bytes());
            for w in v {
                out.extend_from_slice(&w.to_le_bytes());
            }
        }
        let mut out = Vec::new();
        out.extend_from_slice(b"RVCL");
        out.extend_from_slice(&1u16.to_le_bytes());

        let mut meta = Vec::new();
        for v in clip.dims {
            meta.extend_from_slice(&v.to_le_bytes());
        }
        for v in clip.pivot {
            meta.extend_from_slice(&v.to_le_bytes());
        }
        meta.extend_from_slice(&clip.voxel_world_size.to_le_bytes());
        meta.push(clip.loop_mode.to_u8());
        meta.extend_from_slice(&clip.default_frame_ms.to_le_bytes());
        meta.extend_from_slice(&(clip.frames.len() as u32).to_le_bytes());
        chunk(&mut out, b"META", &meta);

        let mut frms = Vec::new();
        for ef in &clip.frames {
            let EncodedFrame::Key(f) = ef else {
                panic!("serialize_v1 test helper handles keyframes only");
            };
            frms.push(FRAME_KIND_KEY);
            u32_vec(&mut frms, &f.occupancy);
            u32_vec(&mut frms, &f.color_offsets);
            u32_vec(&mut frms, &f.colors);
        }
        chunk(&mut out, b"FRMS", &frms);
        out
    }

    #[test]
    fn legacy_v1_file_still_parses() {
        let dims = [2u32, 2, 3];
        let frame = frame_from_fn(dims, |_, _, z| (z == 0).then_some(0x80FF_0000));
        let clip =
            VoxelClip::from_frames(dims, [0.0; 3], 1.0, LoopMode::Once, &[frame], &[], 50, 0);
        let v1 = serialize_v1(&clip);
        assert_eq!(&v1[4..6], &1u16.to_le_bytes(), "helper writes version 1");
        let parsed = VoxelClip::parse(&v1).expect("v1 must still parse");
        assert_eq!(parsed, clip);
    }

    #[test]
    fn bad_deflate_payload_is_rejected() {
        // v2 file whose META chunk is flagged deflated but holds garbage.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RVCL");
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.extend_from_slice(b"META");
        bytes.push(CHUNK_FLAG_DEFLATED);
        let payload = [99u8, 0, 0, 0, 0xDE, 0xAD, 0xBE, 0xEF]; // raw_len=99, junk
        bytes.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&payload);
        assert_eq!(VoxelClip::parse(&bytes), Err(ParseError::BadDeflate));
    }

    // ---- StreamingClip (O(1-frame) seekable cursor) -----------------------

    /// A 7-frame clip with a rising fill height + per-frame colour, so every
    /// frame differs from its neighbour (non-trivial deltas) and frame 6's
    /// height crosses the 32-bit occupancy word boundary. `keyframe_interval
    /// = 3` ⇒ keyframes at 0/3/6, deltas between (exercises replay).
    fn build_varied_clip() -> VoxelClip {
        let dims = [4u32, 3, 40];
        let frames: Vec<VoxelFrame> = (0..7u32)
            .map(|i| {
                let h = 5 + i * 5;
                frame_from_fn(dims, move |_x, _y, z| {
                    (z < h).then_some(0x8000_0000 | (i * 0x10))
                })
            })
            .collect();
        VoxelClip::from_frames(
            dims,
            [2.0, 1.5, 20.0],
            1.0,
            LoopMode::Loop,
            &frames,
            &[],
            33,
            3,
        )
    }

    #[test]
    fn streaming_matches_decoded_forward_and_random() {
        let clip = build_varied_clip();
        let decoded = clip.decode().expect("decode");
        let mut stream = StreamingClip::new(&clip).expect("stream");
        assert_eq!(stream.frame_count(), decoded.frames.len());
        assert_eq!(stream.dims(), decoded.dims);
        assert_eq!(stream.pivot(), decoded.pivot);

        // Sequential forward (incremental stepping).
        for (i, want) in decoded.frames.iter().enumerate() {
            let got = stream.seek(i).expect("seek").clone();
            assert_eq!(&got, want, "frame {i} (forward)");
            assert_eq!(
                stream.current_dirs(),
                decoded.dirs[i].as_slice(),
                "dirs {i}"
            );
            assert_eq!(stream.current_index(), i);
        }
        // Random + backward order (keyframe replay).
        for &i in &[6usize, 0, 4, 1, 5, 2, 3, 0, 6] {
            let got = stream.seek(i).expect("seek").clone();
            assert_eq!(&got, &decoded.frames[i], "frame {i} (random)");
            assert_eq!(stream.current_dirs(), decoded.dirs[i].as_slice());
        }
    }

    #[test]
    fn streaming_seek_clamps_past_end() {
        let clip = build_varied_clip();
        let decoded = clip.decode().unwrap();
        let mut stream = StreamingClip::new(&clip).unwrap();
        let last = decoded.frames.len() - 1;
        let got = stream.seek(999).unwrap().clone();
        assert_eq!(got, decoded.frames[last]);
        assert_eq!(stream.current_index(), last);
    }

    #[test]
    fn streaming_rejects_empty_and_delta_first() {
        let dims = [1u32, 1, 1];
        let mk = |frames: Vec<EncodedFrame>| VoxelClip {
            dims,
            pivot: [0.0; 3],
            voxel_world_size: 1.0,
            loop_mode: LoopMode::Loop,
            default_frame_ms: 1,
            frames,
            durations: Vec::new(),
            extra_chunks: Vec::new(),
        };
        assert_eq!(
            StreamingClip::new(&mk(Vec::new())).map(|_| ()),
            Err(DecodeError::DeltaBeforeKey),
        );
        assert_eq!(
            StreamingClip::new(&mk(vec![EncodedFrame::Delta(Vec::new())])).map(|_| ()),
            Err(DecodeError::DeltaBeforeKey),
        );
    }
}
