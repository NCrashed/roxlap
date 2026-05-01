//! Voxlap's `ptfaces16[43][8]` table (`voxlap5.c:8129-8142`).
//!
//! 43 entries × 8 bytes each. Indexed by `effmask = mask & v->vis`
//! (the bit-AND of the per-voxel "which faces are exposed" byte
//! with the per-iteration mask the DRAWBOUNDCUBELINE macro picks).
//!
//! Each entry's layout:
//! - byte 0: vertex count (4, 6, or 0 for "skip").
//! - bytes 1..7: byte offsets into `caddasm` (the `cadd4[8]` table)
//!   of the vertex projections to compute.
//!
//! Entries with vertex count 0 are skipped — these are face masks
//! whose voxel-cube has no visible polygon (interior voxels with
//! all six faces hidden, or impossible mask combinations).
//!
//! Port verbatim — the byte values encode voxlap's projection
//! order; even reordering "equivalent" entries will diverge from
//! voxlap C's bit-exact output.

/// `ptfaces16[effmask]` table. `effmask` is a 6-bit value (the
/// `& v->vis` mask byte limits inputs to `0..=63`); voxlap pads to
/// 43 rows because sub-mask `0` and the unused-face combinations
/// in `0x33` / `0x37` etc. fall through to skip-entries.
pub const PTFACES16: [[u8; 8]; 43] = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [4, 0, 32, 96, 64, 0, 32, 0],
    [4, 16, 80, 112, 48, 16, 80, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [4, 64, 96, 112, 80, 64, 96, 0],
    [6, 0, 32, 96, 112, 80, 64, 0],
    [6, 16, 80, 64, 96, 112, 48, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [4, 0, 16, 48, 32, 0, 16, 0],
    [6, 0, 16, 48, 32, 96, 64, 0],
    [6, 0, 16, 80, 112, 48, 32, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [4, 0, 64, 80, 16, 0, 64, 0],
    [6, 0, 32, 96, 64, 80, 16, 0],
    [6, 0, 64, 80, 112, 48, 16, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [6, 0, 64, 96, 112, 80, 16, 0],
    [6, 0, 32, 96, 112, 80, 16, 0],
    [6, 0, 64, 96, 112, 48, 16, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [6, 0, 64, 80, 16, 48, 32, 0],
    [6, 16, 48, 32, 96, 64, 80, 0],
    [6, 0, 64, 80, 112, 48, 32, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [4, 32, 48, 112, 96, 32, 48, 0],
    [6, 0, 32, 48, 112, 96, 64, 0],
    [6, 16, 80, 112, 96, 32, 48, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [6, 32, 48, 112, 80, 64, 96, 0],
    [6, 0, 32, 48, 112, 80, 64, 0],
    [6, 16, 80, 64, 96, 32, 48, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [6, 0, 16, 48, 112, 96, 32, 0],
    [6, 0, 16, 48, 112, 96, 64, 0],
    [6, 0, 16, 80, 112, 96, 32, 0],
];
