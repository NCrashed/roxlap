//! grouscan = `gline`'s per-ray voxel-column raycaster — port of
//! `voxlap5.c:grouscanasm_scalar` (~600 lines, voxlap5.c:11575).
//!
//! Substaged across R4.3c..f, mirroring voxlaptest's own grouscan
//! port (Stage 4.5b.2..6):
//!
//! - **R4.3c (this commit)**: cftype data model + `grouscan_run`
//!   prologue. Caches the cf[128] seed slot's state into local
//!   scalars, picks the leading raycast lane. The dispatch skeleton
//!   + draw-phase stubs land in R4.3d.
//! - **R4.3d**: drawcwall / drawfwall / drawceil / drawflor stubs +
//!   the prologue's `v == *ixy_sptr_col ? drawflor : drawceil`
//!   initial dispatch.
//! - **R4.3e**: findslab / slab-split / deletez column advance.
//! - **R4.3f**: remiporend (mip transition) + startsky.

/// One entry on grouscan's `cf` stack — voxlap's `cftype`
/// (`voxlap5.c:128`):
///
/// ```c
/// typedef struct {
///     castdat *i0, *i1;
///     int32_t z0, z1, cx0, cy0, cx1, cy1;
/// } cftype;
/// ```
///
/// `i0` / `i1` are pointers into the `radar` buffer; we mirror with
/// `isize` offsets to match the rest of the port (voxlap's pointer
/// arithmetic can produce values that land before `radar[0]`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CfType {
    pub i0: isize,
    pub i1: isize,
    pub z0: i32,
    pub z1: i32,
    pub cx0: i32,
    pub cy0: i32,
    pub cx1: i32,
    pub cy1: i32,
}

/// Length of the `cf` stack. Voxlap declares `int8_t cfasm[256*32]`
/// (`voxlap5.c:143`) — 8192 bytes used as `cftype[256]`. The seed
/// slot lives at index 128 and the active stack pushes upward
/// (capped at index 191 by an asm `cmp eax, _cfasm[4096]` check).
/// We mirror the full 256-slot footprint.
pub const CF_LEN: usize = 256;

/// Index of the seed slot `gline` populates before invoking
/// `grouscan_run`. Voxlap calls this `cf[128]`.
pub const CF_SEED_INDEX: usize = 128;

use crate::rasterizer::ScanScratch;

/// Per-ray inputs grouscan reads from but does not mutate. Bundled
/// to keep `grouscan_run`'s signature compact.
pub struct GrouscanInputs<'a> {
    /// The slab-list bytes of the column the ray currently sits in.
    /// Voxlap's `v` pointer indexes into this. After R4.3e2d, this
    /// is the INITIAL column (matching the seed `ixy_sptr_col_idx`);
    /// the column-step path recomputes it from `slab_buf` and
    /// `column_offsets` as the ray walks across columns.
    pub column: &'a [u8],
    /// Voxlap's `gylookoff` window into the per-frame `gylookup`
    /// table. For single-mip rendering this is just
    /// `&prelude.y_lookup[..]`; mip transitions in R4.3f6 advance
    /// the offset.
    pub gylookup: &'a [i32],
    /// Voxlap's `gcsub[9]` per-side shading table (each entry is
    /// 8 bytes viewed as four `u16` lanes — see `grouscan_shade`).
    pub gcsub: &'a [i64; 9],
    /// World-level flat slab buffer — voxlap's malloc'd column
    /// data (`vbuf` / `vbit` area). The column-step path slices
    /// this at `column_offsets[ixy_sptr_col_idx]` to refresh
    /// `state.column`.
    pub slab_buf: &'a [u8],
    /// `vsid² × u32` table of byte offsets into `slab_buf`, one
    /// per voxel column. Voxlap stores this as `unsigned char **`
    /// (`sptr`); we use `u32` offsets to keep the port pointer-
    /// free.
    pub column_offsets: &'a [u32],
}

/// All of grouscan's per-ray local state in one struct.
///
/// Voxlap's `grouscanasm_scalar` keeps these as scalars in the
/// function's stack frame, threaded through goto labels via
/// "everything is in scope" implicit dataflow. Rust's per-phase
/// functions can't share locals that way, so we put them on a
/// struct the state machine driver passes by `&mut`.
///
/// The borrows here mean a `GrouscanState` can't outlive the
/// `ScanScratch` it's reading from — that matches voxlap's design
/// where this state lives strictly within one `grouscan_run` call.
//
// dead_code allow: the per-voxel scratch fields (color, gy_raw,
// off, mm5_tail, wall_lane, ebx) are populated and consumed by the
// R4.3f3+ fill loops; in R4.3f2 they're scaffolding the next
// commits will start using.
#[allow(dead_code)]
pub(crate) struct GrouscanState<'a> {
    /// Per-frame scratch (radar, angstart, cf, gpz, gdz, gixy, gi0,
    /// gi1, gxmax, lastx, uurend).
    pub scratch: &'a mut ScanScratch,
    /// Slab bytes of the column the ray currently sits in. Mutated
    /// by R4.3e2d's column-step path (re-sliced from
    /// [`Self::slab_buf`] at the new column's offset).
    pub column: &'a [u8],
    /// `gylookoff` window into the per-frame gylookup table.
    pub gylookup: &'a [i32],
    /// Per-side shading table.
    pub gcsub: &'a [i64; 9],
    /// World-level flat slab buffer (see [`GrouscanInputs`]).
    pub slab_buf: &'a [u8],
    /// Per-column byte offsets into [`Self::slab_buf`].
    pub column_offsets: &'a [u32],

    // -------------------------------------------------------------
    // Cached prologue scalars (R4.3c). Mutated as the algorithm
    // walks; voxlap's `cf[128]` is the seed they're initialised from.
    // -------------------------------------------------------------
    pub z0: i32,
    pub z1: i32,
    pub cx0: i32,
    pub cy0: i32,
    pub cx1: i32,
    pub cy1: i32,
    /// Voxlap's "previous gx", seeded with `gpz[lane] & 0xFFFF0000`.
    pub ogx: i32,
    /// Voxlap's "current gx" — accumulates depth as columns advance.
    pub gx: i32,
    /// `min(gxmax, gxmip)` when multiple mips exist.
    pub ngxmax: i32,
    /// Leading raycast lane: `0` (x) or `1` (y).
    pub lane: usize,

    // -------------------------------------------------------------
    // Per-voxel scratch (R4.3f+ fill loops use these). All start at
    // zero on entry to `grouscan_run`.
    // -------------------------------------------------------------
    /// The per-voxel packed colour shaded by `grouscan_shade`.
    pub color: u32,
    /// Voxlap's `gy_raw` — the gylookup entry for the current voxel
    /// z, used by `grouscan_cross_sign`.
    pub gy_raw: i32,
    /// Byte offset within the current slab for the colour fetch.
    pub off: i32,
    /// `mm5_tail` — alpha-blend tail carried across `grouscan_shade`
    /// invocations within one ray.
    pub mm5_tail: u32,
    /// Which side-shading lane (`gcsub` index) the current wall fill
    /// is using. `0` or `1` for the two raycast lanes.
    pub wall_lane: usize,
    /// Radar offset of the current pixel write — voxlap's `ebx`.
    pub ebx: isize,
    /// Voxlap's `v - *ixy_sptr_col` byte offset within the current
    /// column's slab list. `0` means we're at the top of the column.
    /// Updated by R4.3e2's deletez when the algorithm walks past a
    /// slab; for R4.3f4 it stays at the initial-dispatch value.
    pub vptr_offset: usize,

    // ---------------------------------------------------------------
    // cf-stack cursors (R4.3e2a). Voxlap's `c` (current entry) and
    // `ce` (top-of-stack) are pointers into the `cf[]` array; we
    // mirror with usize indices into `scratch.cf`. Both initialise
    // to `CF_SEED_INDEX = 128` so the seed slot acts as the bottom
    // of the working stack the way voxlap's asm uses it.
    // ---------------------------------------------------------------
    /// Index of the current cf entry — voxlap's `c`.
    pub c_idx: usize,
    /// Index of the cf-stack top — voxlap's `ce`.
    pub ce_idx: usize,
    /// Index of the pre-pop sync slot — voxlap's `c_presync`. Used
    /// by deletez → afterdelete to steer the skipixy2 sync test.
    /// `usize::MAX` means "not set" (the `AfterDelete` path
    /// initialises it; `AfterDeleteKeptPresync` sets it to the
    /// freed slot's index inside deletez).
    pub c_presync_idx: usize,

    /// Voxlap's `ixy_sptr_col` cursor — index into the world's
    /// per-column slab-pointer array. Mutated by R4.3e2d's column-
    /// step path via `gixy[lane]`. `gline` seeds it before invoking
    /// `grouscan_run`; the fill-loop phases never touch it.
    pub ixy_sptr_col_idx: usize,
}

impl<'a> GrouscanState<'a> {
    /// Build a fresh state from the cf[128] seed slot. Mirrors
    /// voxlap5.c:11601-11606.
    fn from_seed(
        scratch: &'a mut ScanScratch,
        inputs: &GrouscanInputs<'a>,
        vptr_offset: usize,
        ixy_sptr_col_idx: usize,
    ) -> Self {
        let c = scratch.cf[CF_SEED_INDEX];
        Self {
            scratch,
            column: inputs.column,
            gylookup: inputs.gylookup,
            gcsub: inputs.gcsub,
            slab_buf: inputs.slab_buf,
            column_offsets: inputs.column_offsets,
            z0: c.z0,
            z1: c.z1,
            cx0: c.cx0,
            cy0: c.cy0,
            cx1: c.cx1,
            cy1: c.cy1,
            ogx: 0,
            gx: 0,
            ngxmax: 0,
            lane: 0,
            color: 0,
            gy_raw: 0,
            off: 0,
            mm5_tail: 0,
            wall_lane: 0,
            ebx: 0,
            vptr_offset,
            c_idx: CF_SEED_INDEX,
            ce_idx: CF_SEED_INDEX,
            c_presync_idx: usize::MAX,
            ixy_sptr_col_idx,
        }
    }
}

/// Voxlap's per-voxel colour-shading helper, used by every fill loop
/// in grouscan.
///
/// Originally MMX (`punpcklbw mm5, vox; psubusb; pshufw mm5, 0xff;
/// pmulhuw; psrlw 7; packuswb`); voxlaptest's scalar port at
/// `voxlap5.c:11438` is what we mirror here. Reads `*tail` (the
/// previous voxel's packed result, used by `pmulhuw`'s broadcast
/// stage), reads 8 byte-lanes of `csub_qword` for the saturated
/// subtract, and writes the new packed colour back into `*tail`
/// before returning it.
///
/// `csub_qword` is voxlap's `gcsub[lane]` — an `i64` viewed as
/// 8 bytes. The high byte (`csub_qword[7]`) is the per-side shading
/// intensity used for the broadcast; the low 4 bytes apply to the
/// per-channel saturated subtract.
//
// The byte arithmetic is voxlap's verbatim — the constant 7-bit
// right-shift, the high-half `pmulhuw` broadcast, and the saturated
// pack are all asm-defined.
#[allow(clippy::cast_possible_truncation)]
#[must_use]
pub fn grouscan_shade(vox: u32, tail: &mut u32, csub_qword: i64) -> u32 {
    let cs = csub_qword.to_le_bytes();
    let t = *tail;

    // punpcklbw mm5, vox — interleave low 4 bytes of tail and vox.
    let mut b = [
        t as u8,
        vox as u8,
        (t >> 8) as u8,
        (vox >> 8) as u8,
        (t >> 16) as u8,
        (vox >> 16) as u8,
        (t >> 24) as u8,
        (vox >> 24) as u8,
    ];

    // psubusb — saturated u8 subtract per byte against csub.
    for i in 0..8 {
        b[i] = b[i].saturating_sub(cs[i]);
    }

    // Repack to 4 u16 words.
    let mut w = [
        u16::from(b[0]) | (u16::from(b[1]) << 8),
        u16::from(b[2]) | (u16::from(b[3]) << 8),
        u16::from(b[4]) | (u16::from(b[5]) << 8),
        u16::from(b[6]) | (u16::from(b[7]) << 8),
    ];

    // pshufw 0xff broadcast w[3], pmulhuw — high-half u16×u16.
    let repl = u32::from(w[3]);
    for slot in &mut w {
        *slot = ((u32::from(*slot) * repl) >> 16) as u16;
    }

    // psrlw 7.
    for slot in &mut w {
        *slot >>= 7;
    }

    // packuswb mm5, mm5 — saturate-pack each word to u8.
    let p = w.map(|x| if x > 255 { 255 } else { x as u8 });
    let color = u32::from(p[0])
        | (u32::from(p[1]) << 8)
        | (u32::from(p[2]) << 16)
        | (u32::from(p[3]) << 24);
    *tail = color;
    color
}

/// Voxlap's cross-product sign test, used by every grouscan fill
/// loop's exit condition. Port of `voxlap5.c:11546`.
///
/// Returns `cx_hi16_signed * gy_low16_signed + cy_hi16_signed *
/// depth_hi16_signed`. The bit-level signature matters: gylookup
/// entries are populated in the asm's int16-signed format so this
/// must use signed 16-bit operands rather than (say) the 32-bit
/// `dmulrethigh` shape — the algebraic equivalence breaks under
/// the int16 sign-extensions.
//
// The `as i16` casts are intentional bit-narrowings — we want the
// low 16 bits viewed as a signed int16. clippy::cast_possible_
// truncation flags exactly that. similar_names: cx_s16 / cy_s16 are
// voxlap names; the one-letter difference is meaningful.
#[allow(clippy::cast_possible_truncation, clippy::similar_names)]
#[must_use]
pub fn grouscan_cross_sign(cx: i32, cy: i32, depth: i32, gy_raw: i32) -> i32 {
    let gy_s16 = i32::from(gy_raw as i16);
    let depth_s16 = i32::from((depth >> 16) as i16);
    let cx_s16 = i32::from((cx >> 16) as i16);
    let cy_s16 = i32::from((cy >> 16) as i16);
    cx_s16 * gy_s16 + cy_s16 * depth_s16
}

/// Snapshot of the prologue state — the local scalars voxlap caches
/// from `cf[128]` before walking the ray. Returned by
/// [`grouscan_run`] in R4.3c so the caller can verify the prologue
/// did its work; later sub-substages will keep this state internal
/// once the dispatch loop consumes it directly.
#[derive(Debug, Clone, Copy)]
pub struct GrouscanPrologue {
    pub z0: i32,
    pub z1: i32,
    pub cx0: i32,
    pub cy0: i32,
    pub cx1: i32,
    pub cy1: i32,
    /// Leading raycast lane: `0` if the next x-grid crossing is
    /// closer than the next y-grid crossing, `1` otherwise.
    pub lane: usize,
    /// `ogx` — voxlap's "previous gx", seeded with `gpz[lane] &
    /// 0xFFFF0000` (the integer part of the leading lane's depth).
    pub ogx: i32,
    /// `gx` starts at `0`; voxlap accumulates depth into it as it
    /// walks columns.
    pub gx: i32,
    /// `ngxmax` = `min(gxmax, gxmip)` when multiple mips exist; for
    /// `gmipnum == 1` this just equals `gxmax`.
    pub ngxmax: i32,
    /// Which draw phase the prologue's initial-dispatch picked. R4.3e+
    /// will branch into the corresponding fill loop; R4.3d ships
    /// only the stubs.
    pub dispatch: InitialDispatch,
}

/// Voxlap's `v == *ixy_sptr_col ? drawflor : drawceil` initial
/// dispatch (`voxlap5.c:11640-11641`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitialDispatch {
    /// Camera sits *above* the first slab in this column — render
    /// the floor of the first slab as seen from below.
    DrawFlor,
    /// Camera is in an air gap *between* slabs — render the ceiling
    /// of the slab immediately below it.
    DrawCeil,
}

/// Run grouscan for one ray.
///
/// `vptr_offset` is the byte offset within the camera-column's slab
/// list where voxlap's `gstartv` lands (`0` when the camera is in
/// the air *above* the first slab; `> 0` when in an interior air
/// gap). The C source compares `v == *ixy_sptr_col`; here we just
/// check the offset directly.
///
/// R4.3c shipped the prologue. R4.3d (this commit) adds the
/// initial dispatch and stubs the four draw phases. R4.3e+ fleshes
/// out the fill loops.
///
/// Side effects on `scratch`:
/// - `gpz[lane] += gdz[lane]` (voxlap's first column advance, baked
///   into the prologue).
//
// The full grouscan body is sub-staged across R4.3c..f; this stub
// returns the prologue snapshot so the prologue's behaviour is
// unit-testable in isolation.
#[must_use]
pub fn grouscan_run(
    scratch: &mut ScanScratch,
    inputs: &GrouscanInputs<'_>,
    vptr_offset: usize,
    ixy_sptr_col_idx: usize,
    gxmip: i32,
    gmipnum: u32,
) -> GrouscanPrologue {
    let mut state = GrouscanState::from_seed(scratch, inputs, vptr_offset, ixy_sptr_col_idx);

    // --- ngxmax = min(gxmax, gxmip) when multiple mips exist. ---
    state.ngxmax = state.scratch.gxmax;
    if gmipnum > 1 && gxmip < state.ngxmax {
        state.ngxmax = gxmip;
    }

    // --- Pick the leading raycast lane. Voxlap5.c:11621-11624. ---
    state.lane = usize::from(state.scratch.gpz[1] < state.scratch.gpz[0]);
    // ogx = gpz[lane] & 0xFFFF0000 — keep only the integer part of
    // the fixed-point depth.
    state.ogx = state.scratch.gpz[state.lane] & -0x1_0000_i32;
    state.gx = 0;
    // First column advance — voxlap's `gpz[lane] += gdz[lane]`.
    state.scratch.gpz[state.lane] =
        state.scratch.gpz[state.lane].wrapping_add(state.scratch.gdz[state.lane]);

    // --- Initial dispatch. Voxlap5.c:11640-11641. ---
    let dispatch = if state.vptr_offset == 0 {
        InitialDispatch::DrawFlor
    } else {
        InitialDispatch::DrawCeil
    };

    // Snapshot the prologue state BEFORE dispatching the state
    // machine — the returned `GrouscanPrologue` is meant to expose
    // the prologue setup, not the post-fill register state.
    let prologue = GrouscanPrologue {
        z0: state.z0,
        z1: state.z1,
        cx0: state.cx0,
        cy0: state.cy0,
        cx1: state.cx1,
        cy1: state.cy1,
        lane: state.lane,
        ogx: state.ogx,
        gx: state.gx,
        ngxmax: state.ngxmax,
        dispatch,
    };

    // --- Phase state machine. R4.3e ships the driver + stubs;
    // R4.3f+ replaces each stub with the real fill body. ---
    let entry = match dispatch {
        InitialDispatch::DrawFlor => Phase::DrawFlor,
        InitialDispatch::DrawCeil => Phase::DrawCeil,
    };
    run_phases(&mut state, entry);

    prologue
}

/// One label in voxlap's grouscan state machine. The C source uses
/// `goto` between these labels; we drive them via [`run_phases`].
///
/// Voxlap line numbers reference the same label names in
/// `voxlaptest`'s `grouscanasm_scalar` (`voxlap5.c:11643..11770`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Front-wall fill (voxlap5.c:11643).
    DrawFwall,
    /// Back-wall fill, falls through from `drawfwall` (11681).
    DrawCwall,
    /// Pre-ceiling — swaps mm6 halves before drawceil (11734).
    PreDrawCeil,
    /// Ceiling fill (11740).
    DrawCeil,
    /// Pre-floor (11761).
    PreDrawFlor,
    /// Floor fill (11765).
    DrawFlor,
    /// Pre-pop cleanup before deletez (no source label; `goto
    /// predeletez` from inside the fill loops).
    PreDeleteZ,
    /// Cf-stack pop / column advance (11967).
    DeleteZ,
    /// Post-pop cleanup. Voxlap5.c:11788. Sets `c_presync = c` and
    /// falls through to [`Phase::AfterDeleteKeptPresync`].
    AfterDelete,
    /// Voxlap5.c:11793. Decrements `c`; either jumps to
    /// [`Phase::SkipixyWithPresync`] (intra-column case) or to the
    /// column-step path. Reached directly from `deletez` when the
    /// post-pop shift fired (skipping the `c_presync = c` re-
    /// assignment).
    AfterDeleteKeptPresync,
    /// Voxlap5.c:11833. Same-column skip path: swap `ogx ↔ gx`
    /// (undoing predeletez's swap), then fall to
    /// [`Phase::SyncFromPresync`].
    SkipixyWithPresync,
    /// Voxlap5.c:11840 (`skipixy2_sync_from_presync`). Saves
    /// scalars to the `c_presync` slot and loads them from the
    /// new `c` slot. Reached from [`Phase::SkipixyWithPresync`]
    /// (intra-column) and — once R4.3e2d lands — from the
    /// column-step path when `c_presync != c`.
    SyncFromPresync,
    /// Voxlap5.c:11853. Findslab dispatch entry. Reads the new
    /// column's first slab header byte `v[0]` — `0` means the
    /// column has only one slab so jump to drawfwall; otherwise
    /// drop into [`Phase::Intoslabloop`] to walk slabs.
    Skipixy3,
    /// Voxlap5.c:11863 (`intoslabloop`). Per-slab body of the
    /// findslab walk: tests whether the current slab intersects
    /// the ray. If `test_hi <= 0` (slab intersects) falls through
    /// to drawfwall (R4.3e2e ships the single-slab case; R4.3e2f
    /// will add the two-slab cfasm split). If `test_hi > 0` (slab
    /// is still above the ray) routes to [`Phase::Findslabloop`].
    Intoslabloop,
    /// Voxlap5.c:11860 (`findslabloop`). Advances `v` by
    /// `v[0] * 4` bytes to the next slab header and re-checks
    /// `v[0]` for column-end. Routes back to
    /// [`Phase::Intoslabloop`] or out to drawfwall.
    Findslabloop,
    /// Voxlap5.c:11998. Mip-level transition. Triggered by the
    /// column step when `gpz[lane]` (unsigned) exceeds `ngxmax`.
    /// R4.3e3 ships the body; R4.3e2d stubs to [`Phase::Done`].
    Remiporend,
    /// Driver-only: no more work. Returned by the last phase.
    Done,
}

/// Drive grouscan's state machine starting at `entry`.
///
/// Each phase function reads / mutates `scratch` and returns the
/// next [`Phase`] — modelling voxlap's `goto X` jumps. R4.3e ships
/// every phase as a stub that returns [`Phase::Done`]; R4.3f+
/// replaces them with the actual fill loops.
fn run_phases(state: &mut GrouscanState<'_>, entry: Phase) {
    let mut current = entry;
    loop {
        current = match current {
            Phase::DrawFwall => phase_draw_fwall(state),
            Phase::DrawCwall => phase_draw_cwall(state),
            Phase::PreDrawCeil => phase_pre_draw_ceil(state),
            Phase::DrawCeil => phase_draw_ceil(state),
            Phase::PreDrawFlor => phase_pre_draw_flor(state),
            Phase::DrawFlor => phase_draw_flor(state),
            Phase::PreDeleteZ => phase_pre_delete_z(state),
            Phase::DeleteZ => phase_delete_z(state),
            Phase::AfterDelete => phase_after_delete(state),
            Phase::AfterDeleteKeptPresync => phase_after_delete_kept_presync(state),
            Phase::SkipixyWithPresync => phase_skipixy_with_presync(state),
            Phase::SyncFromPresync => phase_sync_from_presync(state),
            Phase::Skipixy3 => phase_skipixy3(state),
            Phase::Intoslabloop => phase_intoslabloop(state),
            Phase::Findslabloop => phase_findslabloop(state),
            Phase::Remiporend => phase_remiporend(state),
            Phase::Done => break,
        };
    }
}

// --- Per-phase functions. R4.3f+ stubs return Phase::Done; later
//     iterations replace each body with the fill / pop / mip-
//     transition logic ported from voxlap5.c:11643..11770-area. ---

/// `drawfwall` — front-wall fill (voxlap5.c:11643).
///
/// Walks `z1` upward through the slab's floor-colour list while
/// writing radar entries leftward (`ebx--`) until either the
/// cross-product sign test goes non-positive (move to next voxel
/// row) or `ebx` falls below `c->i0` (radar exhausted, jump to
/// pre-pop cleanup).
//
// Heavy cast traffic ports the asm's bit-narrowings; voxlap names
// (z0/z1, cx0/cy0/cx1/cy1) intentionally one-letter different.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::similar_names
)]
fn phase_draw_fwall(state: &mut GrouscanState<'_>) -> Phase {
    // Need at least 4 header bytes; otherwise no front wall to draw.
    if state.column.len() < 4 {
        return Phase::DrawCwall;
    }

    // Voxlap5.c:11646-11648. dv1 = v[1] = top of floor-colour list.
    let dv1 = i32::from(state.column[1]);
    if dv1 >= state.z1 {
        return Phase::DrawCwall;
    }
    // Cache c->i1 as ebx — the radar offset we walk down from.
    state.ebx = state.scratch.cf[CF_SEED_INDEX].i1;

    'outer: loop {
        // -- loop0 (voxlap5.c:11650): per voxel-row setup. --
        state.off = state.z1 - i32::from(state.column[1]);
        state.z1 -= 1;
        // Read 4-byte voxel colour at byte offset off*4 inside slab.
        let row_offset = (state.off as usize) * 4;
        if row_offset + 4 > state.column.len() {
            // Malformed slab — bail out gracefully.
            state.scratch.cf[CF_SEED_INDEX].i1 = state.ebx;
            return Phase::DrawCwall;
        }
        let vox = u32::from_le_bytes(
            state.column[row_offset..row_offset + 4]
                .try_into()
                .expect("4-byte slice"),
        );
        state.color = grouscan_shade(vox, &mut state.mm5_tail, state.gcsub[state.wall_lane]);
        // gylookup index by current (post-decrement) z1.
        let z1_idx = state.z1 as usize;
        if z1_idx >= state.gylookup.len() {
            state.scratch.cf[CF_SEED_INDEX].i1 = state.ebx;
            return Phase::DrawCwall;
        }
        state.gy_raw = state.gylookup[z1_idx];

        // -- loop1 (voxlap5.c:11659): per-pixel inner. --
        loop {
            let test = grouscan_cross_sign(state.cx1, state.cy1, state.ogx, state.gy_raw);
            if test <= 0 {
                // endloop1 (voxlap5.c:11676). Voxel row exhausted.
                if i32::from(state.column[1]) != state.z1 {
                    continue 'outer;
                }
                // c->i1 = ebx, then fall through to drawcwall.
                state.scratch.cf[CF_SEED_INDEX].i1 = state.ebx;
                return Phase::DrawCwall;
            }
            // Advance right-edge ray left.
            state.cx1 = state.cx1.wrapping_sub(state.scratch.gi0);
            state.cy1 = state.cy1.wrapping_sub(state.scratch.gi1);

            // Write pixel + depth into radar at ebx.
            let radar_idx = state.ebx as usize;
            if let Some(slot) = state.scratch.radar.get_mut(radar_idx) {
                slot.col = state.color as i32;
                slot.dist = state.ogx;
            }
            state.ebx -= 1;
            if state.ebx < state.scratch.cf[CF_SEED_INDEX].i0 {
                // Radar exhausted — jump to pre-pop cleanup.
                return Phase::PreDeleteZ;
            }
            // else: continue loop1.
        }
    }
}

/// `drawcwall` — back-wall fill (voxlap5.c:11681). Mirror of
/// drawfwall:
/// - walks `z0` *upward* through the slab (z0++ per row, vs z1-- in
///   drawfwall),
/// - writes radar entries *rightward* (`ebx++`),
/// - exits the inner loop when cross-sign goes `> 0` (drawfwall:
///   `≤ 0`),
/// - early-out branches: column-top → predrawflor, dv3 ≤ z0 →
///   predrawceil with `z0 = dv3`.
//
// Mirror of phase_draw_fwall — same structural shape with sign
// flips.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::similar_names
)]
fn phase_draw_cwall(state: &mut GrouscanState<'_>) -> Phase {
    if state.column.len() < 4 {
        return Phase::PreDrawCeil;
    }

    // Voxlap5.c:11694 — `z1 = v[1]` UNCONDITIONALLY at drawcwall
    // entry (the comment in the C source warns that drawfwall's
    // early-exit path leaves z1 stale otherwise).
    state.z1 = i32::from(state.column[1]);

    // Column-top: no back wall, jump to drawflor's prep.
    if state.vptr_offset == 0 {
        return Phase::PreDrawFlor;
    }

    // Voxlap5.c:11699-11703. v[3] = z0 of this slab (the air-ceiling
    // above it). If it's ≤ the cached z0 there's no back wall above
    // this slab to draw → set z0 = dv3, fall through to drawceil.
    let dv3 = i32::from(state.column[3]);
    if dv3 <= state.z0 {
        state.z0 = dv3;
        return Phase::PreDrawCeil;
    }

    state.ebx = state.scratch.cf[CF_SEED_INDEX].i0;

    'outer: loop {
        // -- loop2 (voxlap5.c:11706): per voxel-row setup. --
        state.off = state.z0 - i32::from(state.column[3]);
        state.z0 += 1;
        let row_offset = (state.off as usize) * 4;
        if row_offset + 4 > state.column.len() {
            state.scratch.cf[CF_SEED_INDEX].i0 = state.ebx;
            state.z0 = i32::from(state.column[3]);
            return Phase::PreDrawCeil;
        }
        let vox = u32::from_le_bytes(
            state.column[row_offset..row_offset + 4]
                .try_into()
                .expect("4-byte slice"),
        );
        state.color = grouscan_shade(vox, &mut state.mm5_tail, state.gcsub[state.wall_lane]);
        let z0_idx = state.z0 as usize;
        if z0_idx >= state.gylookup.len() {
            state.scratch.cf[CF_SEED_INDEX].i0 = state.ebx;
            state.z0 = i32::from(state.column[3]);
            return Phase::PreDrawCeil;
        }
        state.gy_raw = state.gylookup[z0_idx];

        // -- loop3 (voxlap5.c:11714): per-pixel inner. --
        loop {
            let test = grouscan_cross_sign(state.cx0, state.cy0, state.ogx, state.gy_raw);
            if test > 0 {
                // endloop3 (voxlap5.c:11728). Voxel row exhausted.
                if i32::from(state.column[3]) != state.z0 {
                    continue 'outer;
                }
                // c->i0 = ebx, z0 = v[3], fall through to drawceil.
                state.scratch.cf[CF_SEED_INDEX].i0 = state.ebx;
                state.z0 = i32::from(state.column[3]);
                return Phase::PreDrawCeil;
            }
            // Advance left-edge ray right.
            state.cx0 = state.cx0.wrapping_add(state.scratch.gi0);
            state.cy0 = state.cy0.wrapping_add(state.scratch.gi1);

            let radar_idx = state.ebx as usize;
            if let Some(slot) = state.scratch.radar.get_mut(radar_idx) {
                slot.col = state.color as i32;
                slot.dist = state.ogx;
            }
            state.ebx += 1;
            if state.ebx > state.scratch.cf[CF_SEED_INDEX].i1 {
                return Phase::PreDeleteZ;
            }
        }
    }
}

/// `predrawceil` — voxlap5.c:11734-11737. The asm's `mm6` halves
/// hold `(ogx, gx)` packed; the `pshufd 0x4e` swap before drawceil
/// exposes what was `gx` as the operand the cross-product test
/// reads as `ogx`. In our scalar port that's a plain swap of the
/// two `GrouscanState` scalars.
fn phase_pre_draw_ceil(state: &mut GrouscanState<'_>) -> Phase {
    std::mem::swap(&mut state.ogx, &mut state.gx);
    Phase::DrawCeil
}

/// `drawceil` — ceiling fill (voxlap5.c:11740). Walks `c->i0`
/// rightward (the radar cursor that drawfwall's left-edge fill
/// previously bounded), shading the previous slab's last voxel
/// (`v - 4` in voxlap's pointer layout — `column[vptr_offset - 4
/// ..vptr_offset]` here) into each radar slot.
///
/// Two exit branches:
/// - cross-sign goes `> 0` → fall through to drawflor (the
///   ceiling has been fully drawn for this column).
/// - `c->i0 > c->i1` (radar exhausted mid-fill) → predeletez.
//
// Heavy cast traffic ports the asm's bit-narrowings; voxlap names
// (z0/z1, cx0/cy0/cx1/cy1) intentionally one-letter different.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::similar_names
)]
fn phase_draw_ceil(state: &mut GrouscanState<'_>) -> Phase {
    // gy_raw = gylookoff[z0].
    let z0_idx = state.z0 as usize;
    if z0_idx >= state.gylookup.len() {
        return Phase::PreDeleteZ;
    }
    state.gy_raw = state.gylookup[z0_idx];

    // Ceiling colour = `v - 4` = previous slab's last voxel. Only
    // safe when vptr_offset >= 4; an interior slab always satisfies
    // this (drawceil isn't reachable at column-top — drawcwall
    // detects column-top first and routes to predrawflor).
    if state.vptr_offset < 4 {
        return Phase::PreDeleteZ;
    }
    let vox_off = state.vptr_offset - 4;
    if vox_off + 4 > state.column.len() {
        return Phase::PreDeleteZ;
    }
    let vox = u32::from_le_bytes(
        state.column[vox_off..vox_off + 4]
            .try_into()
            .expect("4-byte slice"),
    );

    loop {
        let test = grouscan_cross_sign(state.cx0, state.cy0, state.ogx, state.gy_raw);
        if test > 0 {
            return Phase::DrawFlor;
        }
        state.cx0 = state.cx0.wrapping_add(state.scratch.gi0);
        state.cy0 = state.cy0.wrapping_add(state.scratch.gi1);

        // Shade per-iteration: mm5_tail carries forward into the
        // pmulhuw broadcast so successive writes differ even with
        // identical `vox`.
        state.color = grouscan_shade(vox, &mut state.mm5_tail, state.gcsub[2]);

        let i0 = state.scratch.cf[CF_SEED_INDEX].i0;
        if let Some(slot) = state.scratch.radar.get_mut(i0 as usize) {
            slot.col = state.color as i32;
            slot.dist = state.ogx;
        }
        state.scratch.cf[CF_SEED_INDEX].i0 = i0 + 1;
        if state.scratch.cf[CF_SEED_INDEX].i0 > state.scratch.cf[CF_SEED_INDEX].i1 {
            return Phase::PreDeleteZ;
        }
    }
}

/// `predrawflor` — voxlap5.c:11761-11763. Mirror of predrawceil.
/// The C code swaps `ogx ↔ gx` so the subsequent drawflor cross-
/// product test reads what was `gx` as its `ogx` operand. Reached
/// from drawcwall's column-top branch (where predrawceil's swap
/// never fired, so this swap takes its place); drawceil → drawflor
/// transitions skip predrawflor and use the post-predrawceil state.
fn phase_pre_draw_flor(state: &mut GrouscanState<'_>) -> Phase {
    std::mem::swap(&mut state.ogx, &mut state.gx);
    Phase::DrawFlor
}

/// `drawflor` — floor fill (voxlap5.c:11765-11783). Mirror of
/// drawceil with three sign flips:
/// - cross-sign exits when `≤ 0` (vs `> 0` in drawceil),
/// - `c->i1` walks LEFTWARD (`c->i1 -= 1`) vs drawceil's
///   `i0 += 1`,
/// - cx1/cy1 advance with `-= gi0/gi1` (vs `+=` in drawceil),
///   and the voxel source is `v + 4` (= top of CURRENT slab)
///   instead of `v - 4` (previous slab's last voxel). gcsub lane
///   3 vs 2.
///
/// Two exits:
/// - cross-sign goes `≤ 0` → `Done` (enddrawflor → afterdelete;
///   the cf-stack pop lives in R4.3e2's deletez).
/// - `c->i1 < c->i0` (radar exhausted) → `PreDeleteZ`.
//
// Heavy cast traffic ports the asm's bit-narrowings.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::similar_names
)]
fn phase_draw_flor(state: &mut GrouscanState<'_>) -> Phase {
    // gy_raw = gylookoff[z1].
    let z1_idx = state.z1 as usize;
    if z1_idx >= state.gylookup.len() {
        return Phase::PreDeleteZ;
    }
    state.gy_raw = state.gylookup[z1_idx];

    // Floor colour = `v + 4` = first voxel byte INSIDE the current
    // slab. Always within column even at column-top (vptr_offset
    // == 0 → slab starts at column[0..4], floor voxel at column[4]).
    let vox_off = state.vptr_offset + 4;
    if vox_off + 4 > state.column.len() {
        return Phase::PreDeleteZ;
    }
    let vox = u32::from_le_bytes(
        state.column[vox_off..vox_off + 4]
            .try_into()
            .expect("4-byte slice"),
    );

    loop {
        let test = grouscan_cross_sign(state.cx1, state.cy1, state.ogx, state.gy_raw);
        if test <= 0 {
            // enddrawflor → afterdelete. Cf-stack pop lands in
            // R4.3e2; for now just terminate the state machine.
            return Phase::Done;
        }
        state.cx1 = state.cx1.wrapping_sub(state.scratch.gi0);
        state.cy1 = state.cy1.wrapping_sub(state.scratch.gi1);

        state.color = grouscan_shade(vox, &mut state.mm5_tail, state.gcsub[3]);

        let i1 = state.scratch.cf[CF_SEED_INDEX].i1;
        if let Some(slot) = state.scratch.radar.get_mut(i1 as usize) {
            slot.col = state.color as i32;
            slot.dist = state.ogx;
        }
        state.scratch.cf[CF_SEED_INDEX].i1 = i1 - 1;
        if state.scratch.cf[CF_SEED_INDEX].i1 < state.scratch.cf[CF_SEED_INDEX].i0 {
            return Phase::PreDeleteZ;
        }
    }
}

/// `predeletez` — voxlap5.c:11962-11965. Swaps `ogx ↔ gx` before
/// falling into deletez. Mirrors the `pshufd 0x4e` on mm6 in
/// the asm.
fn phase_pre_delete_z(state: &mut GrouscanState<'_>) -> Phase {
    std::mem::swap(&mut state.ogx, &mut state.gx);
    Phase::DeleteZ
}

/// `deletez` — voxlap5.c:11967-11997. Pops the cf-stack top (`ce--`).
/// If we're processing an interior entry (`c < old_ce`), shifts
/// entries `(c .. old_ce]` down by one slot to close the gap and
/// stashes the freed slot's index in `c_presync` so the post-
/// column-step skip-sync test fires (otherwise locals would never
/// reload from the now-shifted `cf[c]` memory). Falls into
/// `afterdelete` (or `afterdelete_kept_presync` when the shift
/// fired).
///
/// `if (ce <= &cf[128]) goto retsub` — when the stack drops below
/// the seed slot, the algorithm is done; we return [`Phase::Done`].
fn phase_delete_z(state: &mut GrouscanState<'_>) -> Phase {
    if state.ce_idx <= CF_SEED_INDEX {
        return Phase::Done;
    }
    let old_ce = state.ce_idx;
    state.ce_idx -= 1;
    if state.c_idx < old_ce {
        // Shift cf[c..old_ce] down by one (cf[c] = cf[c+1], …,
        // cf[old_ce-1] = cf[old_ce]).
        for p in state.c_idx..old_ce {
            state.scratch.cf[p] = state.scratch.cf[p + 1];
        }
        state.c_presync_idx = old_ce;
        return Phase::AfterDeleteKeptPresync;
    }
    Phase::AfterDelete
}

/// `afterdelete` — voxlap5.c:11788. The "no-shift" entry point
/// from deletez. Sets `c_presync = c` then falls into
/// [`Phase::AfterDeleteKeptPresync`].
fn phase_after_delete(state: &mut GrouscanState<'_>) -> Phase {
    state.c_presync_idx = state.c_idx;
    Phase::AfterDeleteKeptPresync
}

/// `afterdelete_kept_presync` — voxlap5.c:11793-11831. Decrements
/// `c`; if still in the active region (`c >= cf[128]` after the
/// decrement) routes to [`Phase::SkipixyWithPresync`]. Otherwise
/// the algorithm steps to the next voxel column: advance
/// `ixy_sptr_col_idx` by `gixy[lane]`, refresh `state.column`
/// from `slab_buf` + `column_offsets`, recompute the leading
/// raycast lane, update `gx`/`gpz`. If the new `gpz[lane]`
/// (unsigned) exceeds `ngxmax`, divert to mip transition
/// ([`Phase::Remiporend`], stubbed to [`Phase::Done`] until
/// R4.3e3). Otherwise reset `c` to the stack top (`ce`) and
/// route to [`Phase::Skipixy3`] when the post-pop slot equals
/// `c_presync`, or [`Phase::SyncFromPresync`] otherwise.
//
// Heavy bit-narrowings + unsigned-compare port; the asm uses
// `ja` (unsigned >) on the gpz overflow check.
#[allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::similar_names
)]
fn phase_after_delete_kept_presync(state: &mut GrouscanState<'_>) -> Phase {
    if state.c_idx == 0 {
        // Defensive — voxlap's `c--` would underflow; treat as
        // terminate.
        return Phase::Done;
    }
    state.c_idx -= 1;
    if state.c_idx >= CF_SEED_INDEX {
        return Phase::SkipixyWithPresync;
    }

    // --- Column step (voxlap5.c:11803-11831). ---

    // Cache OLD lane as wall_lane — voxlap's asm captures `mm4 =
    // gcsub[OLD ebp]` BEFORE recomputing the lane (v5.asm:388).
    // Subsequent drawfwall / drawcwall fills use this cached value
    // for their wall side-shade until the next column step.
    state.wall_lane = state.lane;

    // ixy_sptr_col_idx advances by gixy[lane] in element units.
    // Voxlap does the byte arithmetic directly; we keep
    // `column_offsets` in element units so a plain signed-add
    // suffices. gixy can be negative — `wrapping_add_signed` on
    // usize handles that without panicking on overflow.
    let step = state.scratch.gixy[state.lane] as isize;
    state.ixy_sptr_col_idx = state.ixy_sptr_col_idx.wrapping_add_signed(step);

    // Refresh state.column from the world buffers. If the new
    // index is out of range or the offset points past the buffer,
    // leave column unchanged — a malformed world shouldn't crash
    // us; the fill loops have bounds checks of their own.
    if let Some(&col_off) = state.column_offsets.get(state.ixy_sptr_col_idx) {
        let off = col_off as usize;
        if off <= state.slab_buf.len() {
            state.column = &state.slab_buf[off..];
        }
    }
    // Voxlap's `v = *ixy_sptr_col` resets v to the new column's
    // base — vptr_offset was relative to the OLD column's slab
    // list and is meaningless for the new column.
    state.vptr_offset = 0;

    // Recompute the leading raycast lane (the one whose next grid
    // crossing is closer).
    state.lane = usize::from(state.scratch.gpz[1] < state.scratch.gpz[0]);

    let new_gpz = state.scratch.gpz[state.lane];
    // Asm: `punpckldq mm6, mm7 + pand mmask` — gx = high half of
    // new_gpz (the integer part). Low half stays at the post-swap
    // ogx the column-step path inherited.
    state.gx = new_gpz & -0x1_0000_i32;

    // Asm: `ja remiporend`. Unsigned compare catches negative-
    // wrap of new_gpz (gpz can drift past INT32_MAX into negative
    // territory under accumulated step-additions; the unsigned
    // view rolls that into a "very large" gpz that triggers mip
    // transition rather than fooling a signed compare).
    if (new_gpz as u32) > (state.ngxmax as u32) {
        return Phase::Remiporend;
    }

    state.scratch.gpz[state.lane] =
        state.scratch.gpz[state.lane].wrapping_add(state.scratch.gdz[state.lane]);

    // c = ce — re-set current to top-of-stack.
    state.c_idx = state.ce_idx;

    if state.c_presync_idx == state.c_idx {
        Phase::Skipixy3
    } else {
        Phase::SyncFromPresync
    }
}

/// `skipixy_with_presync` — voxlap5.c:11833-11838. Same-column
/// skip path: undoes predeletez's swap (`ogx ↔ gx`), then falls
/// through to [`Phase::SyncFromPresync`]. The swap-undo only
/// fires here because the column-step path overwrites `gx` with
/// `new_gpz_masked`, making the swap meaningful for that path —
/// here we stayed in the same column so the swap was just
/// predeletez bookkeeping.
fn phase_skipixy_with_presync(state: &mut GrouscanState<'_>) -> Phase {
    std::mem::swap(&mut state.ogx, &mut state.gx);
    Phase::SyncFromPresync
}

/// `skipixy2_sync_from_presync` — voxlap5.c:11840-11849. Saves
/// current scalars to the `c_presync` slot and loads them from
/// the new `c` slot. The `i0`/`i1` radar offsets are NOT part of
/// this swap — they stay at whatever the cf entry already holds.
/// Falls through to [`Phase::Skipixy3`] (findslab).
fn phase_sync_from_presync(state: &mut GrouscanState<'_>) -> Phase {
    // Save current scalars into c_presync. Voxlap notes "c_presync
    // is c+1 here" on the same-column path; the column-step path
    // also enters this phase only when `c_presync != c` (the
    // equal case skips directly to skipixy3). Either way, the
    // save/load is on distinct slots → no read-after-write hazard.
    if state.c_presync_idx < state.scratch.cf.len() {
        let presync = &mut state.scratch.cf[state.c_presync_idx];
        presync.z0 = state.z0;
        presync.z1 = state.z1;
        presync.cx0 = state.cx0;
        presync.cy0 = state.cy0;
        presync.cx1 = state.cx1;
        presync.cy1 = state.cy1;
    }

    // Load scalars from the new c slot.
    let c = state.scratch.cf[state.c_idx];
    state.z0 = c.z0;
    state.z1 = c.z1;
    state.cx0 = c.cx0;
    state.cy0 = c.cy0;
    state.cx1 = c.cx1;
    state.cy1 = c.cy1;

    Phase::Skipixy3
}

/// `skipixy3` — voxlap5.c:11853-11858. Findslab dispatch entry.
/// Reads `v[0]` of the new column. `0` means single-slab (jump
/// straight to drawfwall); anything else falls through to
/// [`Phase::Intoslabloop`] to walk slabs.
fn phase_skipixy3(state: &mut GrouscanState<'_>) -> Phase {
    let v0 = column_byte_at(state, 0);
    if v0 == 0 {
        Phase::DrawFwall
    } else {
        Phase::Intoslabloop
    }
}

/// `intoslabloop` — voxlap5.c:11863-11876 (R4.3e2e ports the
/// single-slab dispatch; the two-slab cfasm split lands in
/// R4.3e2f).
///
/// 1. `v2 = v[2]` (slab's solid-bottom z).
/// 2. `gy_raw = gylookoff[v2 + 1]`.
/// 3. `test_hi = cross_sign(cx0, cy0, ogx, gy_raw)`. If `> 0`
///    the slab is still above the ray's frustum top — route to
///    [`Phase::Findslabloop`] to advance.
/// 4. Else (slab intersects): test the NEXT slab's
///    `next_v3 = v[v0*4 + 3]` against `cx1/cy1` to decide if
///    the cfasm needs a split. R4.3e2e takes the single-slab
///    fall-through (`DrawFwall`) regardless — the split case
///    becomes R4.3e2f. This matches voxlap's `if (test_next <=
///    0) goto drawfwall;` half exactly; the `else` branch we
///    stub by also routing to drawfwall, which gives a visual
///    artifact on multi-slab transitions but keeps the algorithm
///    terminating until R4.3e2f lands.
//
// Heavy bit-narrowings + the test_hi sign bookkeeping; voxlap
// names (cx0/cy0/cx1/cy1, v0/v2/v3) intentionally one-letter
// different.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::similar_names
)]
fn phase_intoslabloop(state: &mut GrouscanState<'_>) -> Phase {
    let v2 = i32::from(column_byte_at(state, 2));
    let gy_idx = (v2 + 1) as usize;
    if gy_idx >= state.gylookup.len() {
        // Defensive — malformed v2 puts gylookup index out of
        // range; voxlap C wouldn't bounds-check, but bailing to
        // drawfwall keeps the algorithm terminating safely.
        return Phase::DrawFwall;
    }
    state.gy_raw = state.gylookup[gy_idx];

    let test_hi = grouscan_cross_sign(state.cx0, state.cy0, state.ogx, state.gy_raw);
    if test_hi > 0 {
        // Slab still above the ray — advance to next slab.
        return Phase::Findslabloop;
    }

    // Slab intersects. Voxlap then tests the NEXT slab to decide
    // single-vs-split. R4.3e2e collapses both branches into the
    // single-slab fall-through; R4.3e2f will fork them.
    Phase::DrawFwall
}

/// `findslabloop` — voxlap5.c:11860-11862. Advance `v` by
/// `v[0] * 4` bytes to the next slab header. If the new slab's
/// `v[0]` is `0` we've hit column-end → drawfwall. Otherwise
/// fall back into [`Phase::Intoslabloop`] for the next slab
/// test.
fn phase_findslabloop(state: &mut GrouscanState<'_>) -> Phase {
    let v0 = column_byte_at(state, 0);
    if v0 == 0 {
        // Defensive — would loop forever otherwise (advancing
        // by 0). Voxlap relies on the slab walker reaching the
        // sentinel; if a corrupt column has a non-zero v[0]
        // here that's already been handled, but a 0 sneaking
        // back in is just a column-end.
        return Phase::DrawFwall;
    }
    state.vptr_offset = state.vptr_offset.saturating_add(usize::from(v0) * 4);

    let next_v0 = column_byte_at(state, 0);
    if next_v0 == 0 {
        Phase::DrawFwall
    } else {
        Phase::Intoslabloop
    }
}

/// Read `column[vptr_offset + offset]`, returning `0` (the asm's
/// natural sentinel) when out of bounds. The slab walker reads
/// individual header bytes (`v[0]`, `v[2]`, …); centralising the
/// bounds-check keeps the per-phase code readable.
fn column_byte_at(state: &GrouscanState<'_>, offset: usize) -> u8 {
    state
        .column
        .get(state.vptr_offset.saturating_add(offset))
        .copied()
        .unwrap_or(0)
}

/// `remiporend` — voxlap5.c:11998. Mip-level transition. R4.3e3
/// ships the body; R4.3e2d stubs to [`Phase::Done`].
#[allow(clippy::needless_pass_by_ref_mut)]
fn phase_remiporend(_state: &mut GrouscanState<'_>) -> Phase {
    Phase::Done
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh_scratch() -> ScanScratch {
        let mut s = ScanScratch::new_for_size(64, 64, 64);
        // Seed cf[128] with recognisable values.
        s.cf[CF_SEED_INDEX] = CfType {
            i0: 10,
            i1: 20,
            z0: 5,
            z1: 50,
            cx0: 100,
            cy0: 200,
            cx1: 300,
            cy1: 400,
        };
        s
    }

    /// Empty-ish inputs for tests that don't exercise drawfwall.
    /// `gylookup` and `column` are non-zero-sized so the slice
    /// machinery stays well-defined; values don't matter for the
    /// prologue-only tests.
    const DUMMY_GYLOOKUP: [i32; 64] = [0; 64];
    const DUMMY_GCSUB: [i64; 9] = [0; 9];
    const DUMMY_COLUMN: [u8; 4] = [0, 0, 0, 0];
    const DUMMY_SLAB_BUF: [u8; 0] = [];
    const DUMMY_COLUMN_OFFSETS: [u32; 0] = [];

    fn dummy_inputs<'a>() -> GrouscanInputs<'a> {
        GrouscanInputs {
            column: &DUMMY_COLUMN,
            gylookup: &DUMMY_GYLOOKUP,
            gcsub: &DUMMY_GCSUB,
            slab_buf: &DUMMY_SLAB_BUF,
            column_offsets: &DUMMY_COLUMN_OFFSETS,
        }
    }

    #[test]
    fn prologue_caches_cf_seed_state() {
        let mut s = fresh_scratch();
        s.gpz = [1_000, 2_000];
        s.gdz = [10, 20];
        s.gxmax = 999_999;
        let p = grouscan_run(&mut s, &dummy_inputs(), 0, 0, 50_000, 1);
        assert_eq!(p.z0, 5);
        assert_eq!(p.z1, 50);
        assert_eq!(p.cx0, 100);
        assert_eq!(p.cy0, 200);
        assert_eq!(p.cx1, 300);
        assert_eq!(p.cy1, 400);
    }

    #[test]
    fn prologue_picks_leading_lane_with_smaller_gpz() {
        // gpz[0] = 1000 < gpz[1] = 2000 → lane 0 wins.
        let mut s = fresh_scratch();
        s.gpz = [1_000, 2_000];
        s.gdz = [10, 20];
        let p = grouscan_run(&mut s, &dummy_inputs(), 0, 0, 1, 1);
        assert_eq!(p.lane, 0);

        // Reverse: gpz[1] = 500 < gpz[0] = 800 → lane 1 wins.
        let mut s = fresh_scratch();
        s.gpz = [800, 500];
        s.gdz = [10, 20];
        let p = grouscan_run(&mut s, &dummy_inputs(), 0, 0, 1, 1);
        assert_eq!(p.lane, 1);
    }

    #[test]
    fn prologue_advances_winning_lane_by_gdz() {
        // gpz[0] = 1000 wins; gdz[0] = 256. After: gpz[0] = 1256.
        let mut s = fresh_scratch();
        s.gpz = [1_000, 2_000];
        s.gdz = [256, 999];
        let _ = grouscan_run(&mut s, &dummy_inputs(), 0, 0, 1, 1);
        assert_eq!(s.gpz[0], 1_256);
        // Lane 1 is untouched.
        assert_eq!(s.gpz[1], 2_000);
    }

    #[test]
    fn prologue_ngxmax_clamps_to_gxmip_when_multimip() {
        // gxmax = 1_000_000, gxmip = 50_000. ngxmax = 50_000.
        let mut s = fresh_scratch();
        s.gpz = [1_000, 2_000];
        s.gdz = [10, 20];
        s.gxmax = 1_000_000;
        let p = grouscan_run(&mut s, &dummy_inputs(), 0, 0, 50_000, 2);
        assert_eq!(p.ngxmax, 50_000);

        // Single-mip case: ngxmax = gxmax regardless of gxmip.
        let mut s = fresh_scratch();
        s.gpz = [1_000, 2_000];
        s.gdz = [10, 20];
        s.gxmax = 1_000_000;
        let p = grouscan_run(&mut s, &dummy_inputs(), 0, 0, 50_000, 1);
        assert_eq!(p.ngxmax, 1_000_000);
    }

    #[test]
    fn shade_zero_csub_passes_voxel_through_unchanged_intensity() {
        // csub = 0 → saturating sub leaves bytes alone.
        // Run with a voxel = 0x80aabbcc, tail = 0x80112233.
        // After interleave: [33,cc, 22,bb, 11,aa, 80,80].
        // psubusb 0 → unchanged.
        // word[3] = (80 << 8) | 80 = 0x8080.
        // pmulhuw broadcast: each word * 0x8080 >> 16.
        //   w[0] = ccu(0xcc33) * 0x8080 >> 16
        // psrlw 7 + packuswb. Verify the whole pipeline runs without
        // panicking and produces a valid u32 colour. (Bit-exact tests
        // for non-trivial inputs come once we verify against the C.)
        let mut tail: u32 = 0x8011_2233;
        let _ = grouscan_shade(0x80aa_bbcc, &mut tail, 0);
        // Tail is updated in place.
        assert_ne!(tail, 0x8011_2233);
    }

    #[test]
    fn shade_max_csub_produces_zero_intensity_blackout() {
        // csub all-ones → saturating subtract drops every byte to 0;
        // word[3] = 0; pmulhuw produces 0 across; final colour = 0.
        let mut tail: u32 = 0xdead_beef;
        let out = grouscan_shade(0xffff_ffff, &mut tail, !0_i64);
        assert_eq!(out, 0);
        assert_eq!(tail, 0);
    }

    #[test]
    fn cross_sign_basic_signs() {
        // depth = 1<<16 → depth_hi16 = 1.
        // gy_raw = 0x0001 → gy_low16 = 1.
        // cx = 1<<16 → cx_hi16 = 1.
        // cy = 1<<16 → cy_hi16 = 1.
        // result = 1 * 1 + 1 * 1 = 2.
        assert_eq!(grouscan_cross_sign(1 << 16, 1 << 16, 1 << 16, 1), 2);
    }

    #[test]
    fn cross_sign_negative_high_word_uses_signed_extension() {
        // cx_hi16 = -1 (cx = -65536 = 0xFFFF_0000), gy_low16 = 1.
        // depth = cy = 0. result = -1 * 1 + 0 * 0 = -1.
        assert_eq!(grouscan_cross_sign(-(1 << 16), 0, 0, 1), -1);
    }

    #[test]
    fn cross_sign_drops_low_16_of_cx_cy() {
        // Two cx values that share the same hi16 should produce the
        // same result regardless of the low bits.
        let r1 = grouscan_cross_sign(0x0003_0000, 0, 1 << 16, 1);
        let r2 = grouscan_cross_sign(0x0003_FFFF, 0, 1 << 16, 1);
        assert_eq!(r1, r2);
    }

    /// Build a `GrouscanState` with custom column / gylookup / gcsub
    /// so drawfwall / drawcwall can be exercised directly.
    fn state_for_drawfwall<'a>(
        scratch: &'a mut ScanScratch,
        column: &'a [u8],
        gylookup: &'a [i32],
        gcsub: &'a [i64; 9],
    ) -> GrouscanState<'a> {
        let inputs = GrouscanInputs {
            column,
            gylookup,
            gcsub,
            slab_buf: &DUMMY_SLAB_BUF,
            column_offsets: &DUMMY_COLUMN_OFFSETS,
        };
        GrouscanState::from_seed(scratch, &inputs, 0, 0)
    }

    fn state_for_drawcwall<'a>(
        scratch: &'a mut ScanScratch,
        column: &'a [u8],
        gylookup: &'a [i32],
        gcsub: &'a [i64; 9],
        vptr_offset: usize,
    ) -> GrouscanState<'a> {
        let inputs = GrouscanInputs {
            column,
            gylookup,
            gcsub,
            slab_buf: &DUMMY_SLAB_BUF,
            column_offsets: &DUMMY_COLUMN_OFFSETS,
        };
        GrouscanState::from_seed(scratch, &inputs, vptr_offset, 0)
    }

    #[test]
    fn drawfwall_early_exit_when_v1_above_z1() {
        let mut s = fresh_scratch();
        s.cf[CF_SEED_INDEX].z1 = 30;
        s.cf[CF_SEED_INDEX].i0 = 0;
        s.cf[CF_SEED_INDEX].i1 = 100;
        let column = [0u8, 50, 51, 0]; // v[1] = 50 ≥ z1 = 30 → DrawCwall
        let gylookup = [0i32; 64];
        let gcsub = [0i64; 9];
        let mut state = state_for_drawfwall(&mut s, &column, &gylookup, &gcsub);
        assert_eq!(phase_draw_fwall(&mut state), Phase::DrawCwall);
        // ebx untouched (still 0, never set to c->i1).
        assert_eq!(state.ebx, 0);
    }

    #[test]
    fn drawfwall_iterates_until_z1_hits_v1() {
        // v[1] = 10, z1 = 13 → 3 voxel rows. cx1 = cy1 = 0 → cross
        // sign always 0 → ≤ 0 path → no pixels written. Loop exits
        // when z1 == v[1].
        let mut s = fresh_scratch();
        s.cf[CF_SEED_INDEX].z1 = 13;
        s.cf[CF_SEED_INDEX].i0 = 0;
        s.cf[CF_SEED_INDEX].i1 = 100;
        s.cf[CF_SEED_INDEX].cx1 = 0;
        s.cf[CF_SEED_INDEX].cy1 = 0;
        let mut column = vec![0u8, 10, 12, 0];
        column.extend_from_slice(&[
            0xaa, 0xbb, 0xcc, 0x80, 0x11, 0x22, 0x33, 0x80, 0x44, 0x55, 0x66, 0x80,
        ]);
        let gylookup = [0i32; 64];
        let gcsub = [0i64; 9];
        let mut state = state_for_drawfwall(&mut s, &column, &gylookup, &gcsub);
        assert_eq!(phase_draw_fwall(&mut state), Phase::DrawCwall);
        assert_eq!(state.z1, 10);
        assert_eq!(state.ebx, 100);
        assert_eq!(state.scratch.cf[CF_SEED_INDEX].i1, 100);
    }

    #[test]
    fn drawfwall_writes_pixel_when_cross_sign_positive() {
        // cx1 = 1<<16 (cx_hi16 = 1), gylookup[10] = 100, gi0 = 1<<16,
        // cy1 = 0, ogx = 0. test = 1*100 + 0*0 = 100 > 0 → write
        // pixel; then cx1 -= gi0 → cx_hi16 = 0 → test = 0 ≤ 0 → exit.
        let mut s = fresh_scratch();
        s.cf[CF_SEED_INDEX].z1 = 11;
        s.cf[CF_SEED_INDEX].i0 = 0;
        s.cf[CF_SEED_INDEX].i1 = 50;
        s.cf[CF_SEED_INDEX].cx1 = 1 << 16;
        s.cf[CF_SEED_INDEX].cy1 = 0;
        s.gi0 = 1 << 16;
        s.gi1 = 0;

        let mut column = vec![0u8, 10, 11, 0];
        column.extend_from_slice(&[0x00, 0x00, 0xff, 0x80]); // z=10 voxel

        let mut gylookup = [0i32; 64];
        gylookup[10] = 100;

        let gcsub = [0i64; 9];
        let mut state = state_for_drawfwall(&mut s, &column, &gylookup, &gcsub);
        state.ogx = 0;

        let next = phase_draw_fwall(&mut state);
        assert_eq!(next, Phase::DrawCwall);
        // One pixel written at radar[50] (voxlap-style ARGB; non-zero).
        assert_ne!(state.scratch.radar[50].col, 0);
        // ebx decremented from 50 to 49 by the one pixel write.
        assert_eq!(state.ebx, 49);
    }

    #[test]
    fn drawcwall_column_top_jumps_to_predrawflor() {
        let mut s = fresh_scratch();
        let column = [0u8, 10, 12, 0]; // any header; column-top so v == ixy_sptr_col
        let gylookup = [0i32; 64];
        let gcsub = [0i64; 9];
        let mut state = state_for_drawcwall(&mut s, &column, &gylookup, &gcsub, 0);
        assert_eq!(phase_draw_cwall(&mut state), Phase::PreDrawFlor);
        // z1 was unconditionally updated to v[1] = 10.
        assert_eq!(state.z1, 10);
    }

    #[test]
    fn drawcwall_dv3_le_z0_jumps_to_predrawceil() {
        // dv3 = v[3] = 5, z0 cached = 20. dv3 <= z0 → set z0 = 5,
        // return PreDrawCeil.
        let mut s = fresh_scratch();
        s.cf[CF_SEED_INDEX].z0 = 20;
        let column = [0u8, 10, 12, 5];
        let gylookup = [0i32; 64];
        let gcsub = [0i64; 9];
        let mut state = state_for_drawcwall(&mut s, &column, &gylookup, &gcsub, 32);
        assert_eq!(phase_draw_cwall(&mut state), Phase::PreDrawCeil);
        assert_eq!(state.z0, 5);
    }

    /// Build a `GrouscanState` set up for drawceil entry: a column
    /// whose `vptr_offset` points past 4 bytes of "previous slab
    /// tail" that the ceiling shade reads as `v - 4`.
    fn state_for_drawceil<'a>(
        scratch: &'a mut ScanScratch,
        column: &'a [u8],
        gylookup: &'a [i32],
        gcsub: &'a [i64; 9],
        vptr_offset: usize,
    ) -> GrouscanState<'a> {
        let inputs = GrouscanInputs {
            column,
            gylookup,
            gcsub,
            slab_buf: &DUMMY_SLAB_BUF,
            column_offsets: &DUMMY_COLUMN_OFFSETS,
        };
        GrouscanState::from_seed(scratch, &inputs, vptr_offset, 0)
    }

    #[test]
    fn predrawceil_swaps_ogx_and_gx() {
        let mut s = fresh_scratch();
        let inputs = dummy_inputs();
        let mut state = GrouscanState::from_seed(&mut s, &inputs, 0, 0);
        state.ogx = 0x1111;
        state.gx = 0x2222;
        assert_eq!(phase_pre_draw_ceil(&mut state), Phase::DrawCeil);
        assert_eq!(state.ogx, 0x2222);
        assert_eq!(state.gx, 0x1111);
    }

    #[test]
    fn drawceil_cross_sign_positive_jumps_to_drawflor() {
        // cx0 = 1<<16 (cx_hi16 = 1), gylookup[z0] = 1 → cross_sign
        // = 1*1 + 0*0 = 1 > 0 on entry → fall through to DrawFlor.
        let mut s = fresh_scratch();
        s.cf[CF_SEED_INDEX].z0 = 5;
        s.cf[CF_SEED_INDEX].cx0 = 1 << 16;
        s.cf[CF_SEED_INDEX].cy0 = 0;
        s.cf[CF_SEED_INDEX].i0 = 10;
        s.cf[CF_SEED_INDEX].i1 = 20;

        let mut column = vec![0u8; 8];
        column[0..4].copy_from_slice(&[0x44, 0x55, 0x66, 0x80]); // ceiling vox
        let mut gylookup = [0i32; 64];
        gylookup[5] = 1;
        let gcsub = [0i64; 9];

        let mut state = state_for_drawceil(&mut s, &column, &gylookup, &gcsub, 4);
        state.ogx = 0;
        assert_eq!(phase_draw_ceil(&mut state), Phase::DrawFlor);
        // No pixel written: c->i0 untouched.
        assert_eq!(state.scratch.cf[CF_SEED_INDEX].i0, 10);
    }

    #[test]
    fn drawceil_writes_pixel_then_exhausts_radar() {
        // First iteration: cross_sign = 0 (cx0 = cy0 = 0) ≤ 0 → write
        // pixel at radar[i0=20], i0 → 21. i0 > i1 = 20 → PreDeleteZ.
        let mut s = fresh_scratch();
        s.cf[CF_SEED_INDEX].z0 = 5;
        s.cf[CF_SEED_INDEX].cx0 = 0;
        s.cf[CF_SEED_INDEX].cy0 = 0;
        s.cf[CF_SEED_INDEX].i0 = 20;
        s.cf[CF_SEED_INDEX].i1 = 20;
        s.gi0 = 0;
        s.gi1 = 0;

        let mut column = vec![0u8; 8];
        column[0..4].copy_from_slice(&[0x00, 0x00, 0xff, 0x80]);
        let mut gylookup = [0i32; 64];
        gylookup[5] = 0;
        let gcsub = [0i64; 9];

        let mut state = state_for_drawceil(&mut s, &column, &gylookup, &gcsub, 4);
        state.ogx = 0;
        assert_eq!(phase_draw_ceil(&mut state), Phase::PreDeleteZ);
        assert_ne!(state.scratch.radar[20].col, 0);
        assert_eq!(state.scratch.cf[CF_SEED_INDEX].i0, 21);
    }

    #[test]
    fn drawceil_bails_when_z0_out_of_gylookup() {
        // z0 = 64, gylookup len = 64 → out-of-range, bail to PreDeleteZ.
        let mut s = fresh_scratch();
        s.cf[CF_SEED_INDEX].z0 = 64;
        let column = vec![0u8; 8];
        let gylookup = [0i32; 64];
        let gcsub = [0i64; 9];
        let mut state = state_for_drawceil(&mut s, &column, &gylookup, &gcsub, 4);
        assert_eq!(phase_draw_ceil(&mut state), Phase::PreDeleteZ);
    }

    #[test]
    fn predrawflor_swaps_ogx_and_gx() {
        let mut s = fresh_scratch();
        let inputs = dummy_inputs();
        let mut state = GrouscanState::from_seed(&mut s, &inputs, 0, 0);
        state.ogx = 0x3333;
        state.gx = 0x4444;
        assert_eq!(phase_pre_draw_flor(&mut state), Phase::DrawFlor);
        assert_eq!(state.ogx, 0x4444);
        assert_eq!(state.gx, 0x3333);
    }

    #[test]
    fn drawflor_cross_sign_non_positive_returns_done() {
        // cx1 = cy1 = 0, gylookup[z1] = 0 → cross_sign = 0 ≤ 0 on
        // entry → enddrawflor → Done. No pixel written.
        let mut s = fresh_scratch();
        s.cf[CF_SEED_INDEX].z1 = 5;
        s.cf[CF_SEED_INDEX].cx1 = 0;
        s.cf[CF_SEED_INDEX].cy1 = 0;
        s.cf[CF_SEED_INDEX].i0 = 10;
        s.cf[CF_SEED_INDEX].i1 = 20;

        // Slab header at column[0..4]; floor voxel at column[4..8].
        let mut column = vec![0u8; 8];
        column[4..8].copy_from_slice(&[0x77, 0x88, 0x99, 0x80]);
        let gylookup = [0i32; 64];
        let gcsub = [0i64; 9];

        let mut state = state_for_drawceil(&mut s, &column, &gylookup, &gcsub, 0);
        state.ogx = 0;
        assert_eq!(phase_draw_flor(&mut state), Phase::Done);
        assert_eq!(state.scratch.cf[CF_SEED_INDEX].i1, 20);
    }

    #[test]
    fn drawflor_writes_pixel_then_exhausts_radar() {
        // cx1 = 1<<16, cy1 = 0, gylookup[z1] = 1 → cross_sign = 1 > 0
        // → write pixel at radar[i1=20], i1 → 19. i0 = 20 → 19 < 20
        // trips PreDeleteZ.
        let mut s = fresh_scratch();
        s.cf[CF_SEED_INDEX].z1 = 5;
        s.cf[CF_SEED_INDEX].cx1 = 1 << 16;
        s.cf[CF_SEED_INDEX].cy1 = 0;
        s.cf[CF_SEED_INDEX].i0 = 20;
        s.cf[CF_SEED_INDEX].i1 = 20;
        s.gi0 = 0;
        s.gi1 = 0;

        let mut column = vec![0u8; 8];
        column[4..8].copy_from_slice(&[0x00, 0x00, 0xff, 0x80]);
        let mut gylookup = [0i32; 64];
        gylookup[5] = 1;
        let gcsub = [0i64; 9];

        let mut state = state_for_drawceil(&mut s, &column, &gylookup, &gcsub, 0);
        state.ogx = 0;
        assert_eq!(phase_draw_flor(&mut state), Phase::PreDeleteZ);
        assert_ne!(state.scratch.radar[20].col, 0);
        assert_eq!(state.scratch.cf[CF_SEED_INDEX].i1, 19);
    }

    #[test]
    fn drawflor_bails_when_z1_out_of_gylookup() {
        let mut s = fresh_scratch();
        s.cf[CF_SEED_INDEX].z1 = 64;
        let column = vec![0u8; 8];
        let gylookup = [0i32; 64];
        let gcsub = [0i64; 9];
        let mut state = state_for_drawceil(&mut s, &column, &gylookup, &gcsub, 0);
        assert_eq!(phase_draw_flor(&mut state), Phase::PreDeleteZ);
    }

    #[test]
    fn predeletez_swaps_ogx_and_gx() {
        let mut s = fresh_scratch();
        let inputs = dummy_inputs();
        let mut state = GrouscanState::from_seed(&mut s, &inputs, 0, 0);
        state.ogx = 0xAAAA;
        state.gx = 0xBBBB;
        assert_eq!(phase_pre_delete_z(&mut state), Phase::DeleteZ);
        assert_eq!(state.ogx, 0xBBBB);
        assert_eq!(state.gx, 0xAAAA);
    }

    #[test]
    fn deletez_at_seed_slot_returns_done() {
        // ce_idx == CF_SEED_INDEX (= 128) → `ce <= &cf[128]` →
        // retsub → Done. Initial state from from_seed already has
        // ce_idx == CF_SEED_INDEX.
        let mut s = fresh_scratch();
        let inputs = dummy_inputs();
        let mut state = GrouscanState::from_seed(&mut s, &inputs, 0, 0);
        assert_eq!(state.ce_idx, CF_SEED_INDEX);
        assert_eq!(phase_delete_z(&mut state), Phase::Done);
    }

    #[test]
    fn deletez_pops_top_when_c_equals_ce() {
        // ce above seed and c == ce → just decrement ce, no shift,
        // route to AfterDelete.
        let mut s = fresh_scratch();
        let inputs = dummy_inputs();
        let mut state = GrouscanState::from_seed(&mut s, &inputs, 0, 0);
        state.ce_idx = CF_SEED_INDEX + 2;
        state.c_idx = CF_SEED_INDEX + 2;
        assert_eq!(phase_delete_z(&mut state), Phase::AfterDelete);
        assert_eq!(state.ce_idx, CF_SEED_INDEX + 1);
        assert_eq!(state.c_idx, CF_SEED_INDEX + 2);
        assert_eq!(state.c_presync_idx, usize::MAX);
    }

    #[test]
    fn deletez_shifts_down_when_c_below_ce() {
        // ce above c → shift cf[c..old_ce] down, stash old_ce as
        // c_presync, route to AfterDeleteKeptPresync.
        let mut s = fresh_scratch();
        // Plant recognisable values in cf[129] and cf[130].
        s.cf[CF_SEED_INDEX + 1] = CfType {
            i0: 1,
            i1: 1,
            ..Default::default()
        };
        s.cf[CF_SEED_INDEX + 2] = CfType {
            i0: 2,
            i1: 2,
            ..Default::default()
        };
        let inputs = dummy_inputs();
        let mut state = GrouscanState::from_seed(&mut s, &inputs, 0, 0);
        state.ce_idx = CF_SEED_INDEX + 2;
        state.c_idx = CF_SEED_INDEX + 1;
        assert_eq!(phase_delete_z(&mut state), Phase::AfterDeleteKeptPresync);
        assert_eq!(state.ce_idx, CF_SEED_INDEX + 1);
        assert_eq!(state.c_presync_idx, CF_SEED_INDEX + 2);
        // cf[c=129] now holds what was at cf[130].
        assert_eq!(state.scratch.cf[CF_SEED_INDEX + 1].i0, 2);
    }

    #[test]
    fn from_seed_initialises_cf_indices_to_seed() {
        let mut s = fresh_scratch();
        let inputs = dummy_inputs();
        let state = GrouscanState::from_seed(&mut s, &inputs, 0, 0);
        assert_eq!(state.c_idx, CF_SEED_INDEX);
        assert_eq!(state.ce_idx, CF_SEED_INDEX);
        assert_eq!(state.c_presync_idx, usize::MAX);
    }

    #[test]
    fn afterdelete_sets_presync_and_routes_to_kept() {
        let mut s = fresh_scratch();
        let inputs = dummy_inputs();
        let mut state = GrouscanState::from_seed(&mut s, &inputs, 0, 0);
        state.c_idx = CF_SEED_INDEX + 1;
        state.c_presync_idx = usize::MAX;
        assert_eq!(
            phase_after_delete(&mut state),
            Phase::AfterDeleteKeptPresync
        );
        assert_eq!(state.c_presync_idx, CF_SEED_INDEX + 1);
    }

    #[test]
    fn afterdelete_kept_presync_routes_to_skipixy_when_c_above_seed() {
        // c at cf[129]; c-- = cf[128], which is >= cf[128] →
        // SkipixyWithPresync (intra-column case).
        let mut s = fresh_scratch();
        let inputs = dummy_inputs();
        let mut state = GrouscanState::from_seed(&mut s, &inputs, 0, 0);
        state.c_idx = CF_SEED_INDEX + 1;
        assert_eq!(
            phase_after_delete_kept_presync(&mut state),
            Phase::SkipixyWithPresync
        );
        assert_eq!(state.c_idx, CF_SEED_INDEX);
    }

    #[test]
    fn afterdelete_kept_presync_below_seed_runs_column_step() {
        // c at cf[128]; c-- = cf[127], below seed → column-step
        // fires. With all-zero gpz/gdz/gixy and empty world, the
        // step:
        //   - wall_lane = 0 (cached from old lane)
        //   - lane recompute: (0 < 0) → 0
        //   - gpz[0] = 0, ngxmax = 0 → not > ngxmax → no Remiporend
        //   - c_idx = ce_idx = CF_SEED_INDEX
        //   - c_presync_idx (= usize::MAX) != c_idx → SyncFromPresync
        let mut s = fresh_scratch();
        let inputs = dummy_inputs();
        let mut state = GrouscanState::from_seed(&mut s, &inputs, 0, 0);
        state.c_idx = CF_SEED_INDEX;
        assert_eq!(
            phase_after_delete_kept_presync(&mut state),
            Phase::SyncFromPresync
        );
        // c was reset to ce (still at seed).
        assert_eq!(state.c_idx, CF_SEED_INDEX);
    }

    /// 4×4 voxel-column world, each column starts with a bare
    /// 4-byte slab header. Used by the column-step tests to verify
    /// `state.column` re-slices correctly.
    fn build_4x4_world() -> (Vec<u8>, Vec<u32>) {
        let mut buf = Vec::with_capacity(16 * 4);
        for col in 0..16u8 {
            // Per-column header — first byte holds the column
            // index so tests can verify which column got loaded.
            buf.extend_from_slice(&[col, 10, 12, 0]);
        }
        let offsets: Vec<u32> = (0..16u32).map(|c| c * 4).collect();
        (buf, offsets)
    }

    #[test]
    fn column_step_advances_ixy_and_reslices_column() {
        let (slab_buf, column_offsets) = build_4x4_world();
        let gylookup = DUMMY_GYLOOKUP;
        let gcsub = DUMMY_GCSUB;
        let inputs = GrouscanInputs {
            column: &slab_buf[20..], // initial column = #5
            gylookup: &gylookup,
            gcsub: &gcsub,
            slab_buf: &slab_buf,
            column_offsets: &column_offsets,
        };
        let mut s = fresh_scratch();
        s.gixy = [1, 4]; // x-step = 1, y-step = 4
        let mut state = GrouscanState::from_seed(&mut s, &inputs, 0, 5);
        state.c_idx = CF_SEED_INDEX; // → c-- below seed → column step
        state.lane = 0; // step by gixy[0] = 1

        let next = phase_after_delete_kept_presync(&mut state);
        assert!(matches!(
            next,
            Phase::SyncFromPresync | Phase::Skipixy3 | Phase::Remiporend
        ));

        // Cursor advanced from 5 to 6 (= column #6).
        assert_eq!(state.ixy_sptr_col_idx, 6);
        // state.column now points at column #6's bytes — first
        // byte of the new slab header is `6`.
        assert_eq!(state.column[0], 6);
        // wall_lane captured the OLD lane (= 0) before recompute.
        assert_eq!(state.wall_lane, 0);
    }

    #[test]
    fn column_step_routes_to_remiporend_when_gpz_exceeds_ngxmax() {
        let (slab_buf, column_offsets) = build_4x4_world();
        let gylookup = DUMMY_GYLOOKUP;
        let gcsub = DUMMY_GCSUB;
        let inputs = GrouscanInputs {
            column: &slab_buf[..],
            gylookup: &gylookup,
            gcsub: &gcsub,
            slab_buf: &slab_buf,
            column_offsets: &column_offsets,
        };
        let mut s = fresh_scratch();
        // BOTH lanes must exceed ngxmax: lane recompute picks the
        // smaller gpz, so the *winning* lane is the one whose gpz
        // is also above ngxmax.
        s.gpz = [0x100, 0x200];
        let mut state = GrouscanState::from_seed(&mut s, &inputs, 0, 0);
        state.ngxmax = 0xFF;
        state.c_idx = CF_SEED_INDEX;

        assert_eq!(
            phase_after_delete_kept_presync(&mut state),
            Phase::Remiporend
        );
    }

    #[test]
    fn column_step_routes_to_skipixy3_when_presync_equals_c() {
        let (slab_buf, column_offsets) = build_4x4_world();
        let gylookup = DUMMY_GYLOOKUP;
        let gcsub = DUMMY_GCSUB;
        let inputs = GrouscanInputs {
            column: &slab_buf[..],
            gylookup: &gylookup,
            gcsub: &gcsub,
            slab_buf: &slab_buf,
            column_offsets: &column_offsets,
        };
        let mut s = fresh_scratch();
        let mut state = GrouscanState::from_seed(&mut s, &inputs, 0, 0);
        state.c_idx = CF_SEED_INDEX;
        // c is reset to ce inside column step. If presync == ce
        // already, the post-reset c equals presync → Skipixy3.
        state.ce_idx = CF_SEED_INDEX;
        state.c_presync_idx = CF_SEED_INDEX;
        assert_eq!(phase_after_delete_kept_presync(&mut state), Phase::Skipixy3);
    }

    #[test]
    fn column_step_resets_vptr_offset_to_zero() {
        let (slab_buf, column_offsets) = build_4x4_world();
        let gylookup = DUMMY_GYLOOKUP;
        let gcsub = DUMMY_GCSUB;
        let inputs = GrouscanInputs {
            column: &slab_buf[..],
            gylookup: &gylookup,
            gcsub: &gcsub,
            slab_buf: &slab_buf,
            column_offsets: &column_offsets,
        };
        let mut s = fresh_scratch();
        let mut state = GrouscanState::from_seed(&mut s, &inputs, 32, 0);
        state.c_idx = CF_SEED_INDEX;
        let _ = phase_after_delete_kept_presync(&mut state);
        assert_eq!(state.vptr_offset, 0);
    }

    #[test]
    fn skipixy3_routes_to_drawfwall_when_v0_zero() {
        let mut s = fresh_scratch();
        let column = [0u8, 10, 12, 0]; // v[0] = 0 → single-slab.
        let gylookup = [0i32; 64];
        let gcsub = [0i64; 9];
        let mut state = state_for_drawcwall(&mut s, &column, &gylookup, &gcsub, 0);
        assert_eq!(phase_skipixy3(&mut state), Phase::DrawFwall);
    }

    #[test]
    fn skipixy3_routes_to_intoslabloop_when_v0_nonzero() {
        let mut s = fresh_scratch();
        // v[0] = 2 → next slab is 8 bytes ahead → multi-slab column.
        let column = [2u8, 10, 12, 0, 0, 0, 0, 0, 0, 20, 22, 0];
        let gylookup = [0i32; 64];
        let gcsub = [0i64; 9];
        let mut state = state_for_drawcwall(&mut s, &column, &gylookup, &gcsub, 0);
        assert_eq!(phase_skipixy3(&mut state), Phase::Intoslabloop);
    }

    #[test]
    fn intoslabloop_routes_to_findslabloop_when_test_hi_positive() {
        // cx0 = 1<<16, gylookup[v[2]+1] = 1 → cross_sign = 1 > 0 →
        // slab is still above the ray → Findslabloop.
        let mut s = fresh_scratch();
        s.cf[CF_SEED_INDEX].cx0 = 1 << 16;
        let column = [2u8, 10, 12, 0, 0, 0, 0, 0, 0, 20, 22, 0];
        let mut gylookup = [0i32; 64];
        gylookup[13] = 1; // v[2]+1 = 12+1 = 13
        let gcsub = [0i64; 9];
        let mut state = state_for_drawcwall(&mut s, &column, &gylookup, &gcsub, 0);
        state.ogx = 0;
        assert_eq!(phase_intoslabloop(&mut state), Phase::Findslabloop);
    }

    #[test]
    fn intoslabloop_routes_to_drawfwall_when_test_hi_nonpositive() {
        // cx0 = cy0 = 0, gylookup all 0 → cross_sign = 0 ≤ 0 →
        // slab intersects → DrawFwall (single-slab path; R4.3e2f
        // will fork the split case).
        let mut s = fresh_scratch();
        let column = [2u8, 10, 12, 0, 0, 0, 0, 0, 0, 20, 22, 0];
        let gylookup = [0i32; 64];
        let gcsub = [0i64; 9];
        let mut state = state_for_drawcwall(&mut s, &column, &gylookup, &gcsub, 0);
        state.ogx = 0;
        assert_eq!(phase_intoslabloop(&mut state), Phase::DrawFwall);
    }

    #[test]
    fn findslabloop_advances_vptr_then_intoslabloop_when_next_nonzero() {
        // v[0] = 2 at vptr_offset 0 → advance by 8 to slab #2 at
        // offset 8. Slab #2's v[0] = 2 (also non-zero) → loop
        // back to Intoslabloop.
        let mut s = fresh_scratch();
        let column = [
            2u8, 10, 12, 0, 0, 0, 0, 0, // slab 0
            2u8, 20, 22, 0, 0, 0, 0, 0, // slab 1 (advance lands here)
            0u8, 30, 32, 0, // slab 2 (sentinel)
        ];
        let gylookup = [0i32; 64];
        let gcsub = [0i64; 9];
        let mut state = state_for_drawcwall(&mut s, &column, &gylookup, &gcsub, 0);
        assert_eq!(phase_findslabloop(&mut state), Phase::Intoslabloop);
        assert_eq!(state.vptr_offset, 8);
    }

    #[test]
    fn findslabloop_routes_to_drawfwall_when_next_v0_zero() {
        // v[0] = 1 → advance by 4 → next slab v[0] = 0 (column-end).
        let mut s = fresh_scratch();
        let column = [1u8, 10, 12, 0, 0u8, 20, 22, 0];
        let gylookup = [0i32; 64];
        let gcsub = [0i64; 9];
        let mut state = state_for_drawcwall(&mut s, &column, &gylookup, &gcsub, 0);
        assert_eq!(phase_findslabloop(&mut state), Phase::DrawFwall);
        assert_eq!(state.vptr_offset, 4);
    }

    #[test]
    fn skipixy_with_presync_swaps_ogx_and_routes_to_sync() {
        let mut s = fresh_scratch();
        let inputs = dummy_inputs();
        let mut state = GrouscanState::from_seed(&mut s, &inputs, 0, 0);
        state.ogx = 0xAAAA;
        state.gx = 0xBBBB;

        assert_eq!(
            phase_skipixy_with_presync(&mut state),
            Phase::SyncFromPresync
        );
        // Swap fired.
        assert_eq!(state.ogx, 0xBBBB);
        assert_eq!(state.gx, 0xAAAA);
    }

    #[test]
    fn sync_from_presync_saves_to_presync_and_loads_from_c() {
        // Two slots: c_presync at 130, c at 129. State scalars hold
        // "current" values (call them A); cf[129] holds different
        // values (B). After sync_from_presync:
        //   - cf[130] holds A (saved from state).
        //   - state holds B (loaded from cf[129]).
        // ogx / gx are NOT touched (that's skipixy_with_presync's
        // job; column-step path overwrites gx separately).
        let mut s = fresh_scratch();
        s.cf[CF_SEED_INDEX + 1] = CfType {
            i0: 0,
            i1: 0,
            z0: 100,
            z1: 200,
            cx0: 300,
            cy0: 400,
            cx1: 500,
            cy1: 600,
        };
        let inputs = dummy_inputs();
        let mut state = GrouscanState::from_seed(&mut s, &inputs, 0, 0);
        // Working state = "A".
        state.z0 = 1;
        state.z1 = 2;
        state.cx0 = 3;
        state.cy0 = 4;
        state.cx1 = 5;
        state.cy1 = 6;
        state.ogx = 0xAAAA;
        state.gx = 0xBBBB;
        state.c_idx = CF_SEED_INDEX + 1;
        state.c_presync_idx = CF_SEED_INDEX + 2;

        assert_eq!(phase_sync_from_presync(&mut state), Phase::Skipixy3);

        // ogx / gx untouched.
        assert_eq!(state.ogx, 0xAAAA);
        assert_eq!(state.gx, 0xBBBB);

        // c_presync got A.
        let presync = state.scratch.cf[CF_SEED_INDEX + 2];
        assert_eq!(presync.z0, 1);
        assert_eq!(presync.z1, 2);
        assert_eq!(presync.cx0, 3);
        assert_eq!(presync.cy0, 4);
        assert_eq!(presync.cx1, 5);
        assert_eq!(presync.cy1, 6);

        // state got B.
        assert_eq!(state.z0, 100);
        assert_eq!(state.z1, 200);
        assert_eq!(state.cx0, 300);
        assert_eq!(state.cy0, 400);
        assert_eq!(state.cx1, 500);
        assert_eq!(state.cy1, 600);
    }

    #[test]
    fn from_seed_carries_ixy_sptr_col_idx() {
        let mut s = fresh_scratch();
        let inputs = dummy_inputs();
        let state = GrouscanState::from_seed(&mut s, &inputs, 0, 42);
        assert_eq!(state.ixy_sptr_col_idx, 42);
    }

    #[test]
    fn dispatch_drawflor_when_camera_at_top_of_column() {
        let mut s = fresh_scratch();
        s.gpz = [1_000, 2_000];
        s.gdz = [10, 20];
        let p = grouscan_run(&mut s, &dummy_inputs(), 0, 0, 1, 1);
        assert_eq!(p.dispatch, InitialDispatch::DrawFlor);
    }

    #[test]
    fn dispatch_drawceil_when_camera_in_interior_air_gap() {
        let mut s = fresh_scratch();
        s.gpz = [1_000, 2_000];
        s.gdz = [10, 20];
        // vptr_offset > 0 → camera in interior.
        let p = grouscan_run(&mut s, &dummy_inputs(), 16, 0, 1, 1);
        assert_eq!(p.dispatch, InitialDispatch::DrawCeil);
    }

    #[test]
    fn prologue_ogx_keeps_integer_part_of_gpz() {
        // gpz[0] = 0x12345678 → ogx = 0x12340000.
        let mut s = fresh_scratch();
        s.gpz = [0x1234_5678, 0x7FFF_FFFF];
        s.gdz = [0, 0];
        let p = grouscan_run(&mut s, &dummy_inputs(), 0, 0, 1, 1);
        assert_eq!(p.lane, 0);
        // 0x1234_0000 fits i32 positively (high bit clear).
        assert_eq!(p.ogx, 0x1234_0000_i32);
        assert_eq!(p.gx, 0);
    }
}
