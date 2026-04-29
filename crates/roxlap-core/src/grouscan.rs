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

/// Length of the `cf` stack. Voxlap's asm allocates `cfasm dd 32*129`
/// (32 bytes × 129 entries); we keep the same 129-slot footprint.
pub const CF_LEN: usize = 129;

/// Index of the seed slot `gline` populates before invoking
/// `grouscan_run`. Voxlap calls this `cf[128]`.
pub const CF_SEED_INDEX: usize = 128;

use crate::rasterizer::ScanScratch;

/// Per-ray inputs grouscan reads from but does not mutate. Bundled
/// to keep `grouscan_run`'s signature compact.
pub struct GrouscanInputs<'a> {
    /// The slab-list bytes of the column the ray currently sits in.
    /// Voxlap's `v` pointer indexes into this.
    pub column: &'a [u8],
    /// Voxlap's `gylookoff` window into the per-frame `gylookup`
    /// table. For single-mip rendering this is just
    /// `&prelude.y_lookup[..]`; mip transitions in R4.3f6 advance
    /// the offset.
    pub gylookup: &'a [i32],
    /// Voxlap's `gcsub[9]` per-side shading table (each entry is
    /// 8 bytes viewed as four `u16` lanes — see `grouscan_shade`).
    pub gcsub: &'a [i64; 9],
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
    /// Slab bytes of the column the ray currently sits in.
    pub column: &'a [u8],
    /// `gylookoff` window into the per-frame gylookup table.
    pub gylookup: &'a [i32],
    /// Per-side shading table.
    pub gcsub: &'a [i64; 9],

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
}

impl<'a> GrouscanState<'a> {
    /// Build a fresh state from the cf[128] seed slot. Mirrors
    /// voxlap5.c:11601-11606.
    fn from_seed(
        scratch: &'a mut ScanScratch,
        inputs: &GrouscanInputs<'a>,
        vptr_offset: usize,
    ) -> Self {
        let c = scratch.cf[CF_SEED_INDEX];
        Self {
            scratch,
            column: inputs.column,
            gylookup: inputs.gylookup,
            gcsub: inputs.gcsub,
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
    gxmip: i32,
    gmipnum: u32,
) -> GrouscanPrologue {
    let mut state = GrouscanState::from_seed(scratch, inputs, vptr_offset);

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

    // --- Phase state machine. R4.3e ships the driver + stubs;
    // R4.3f+ replaces each stub with the real fill body. ---
    let entry = match dispatch {
        InitialDispatch::DrawFlor => Phase::DrawFlor,
        InitialDispatch::DrawCeil => Phase::DrawCeil,
    };
    run_phases(&mut state, entry);

    GrouscanPrologue {
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
    }
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

#[allow(clippy::needless_pass_by_ref_mut)]
fn phase_pre_draw_flor(_state: &mut GrouscanState<'_>) -> Phase {
    // R4.3f6: voxlap5.c:11761. Sets up drawflor.
    Phase::DrawFlor
}

#[allow(clippy::needless_pass_by_ref_mut)]
fn phase_draw_flor(_state: &mut GrouscanState<'_>) -> Phase {
    // R4.3f6: voxlap5.c:11765. Floor fill.
    Phase::Done
}

#[allow(clippy::needless_pass_by_ref_mut)]
fn phase_pre_delete_z(_state: &mut GrouscanState<'_>) -> Phase {
    // R4.3e2: pre-pop cleanup before deletez.
    Phase::DeleteZ
}

#[allow(clippy::needless_pass_by_ref_mut)]
fn phase_delete_z(_state: &mut GrouscanState<'_>) -> Phase {
    // R4.3e2: voxlap5.c:11967. Cf-stack pop / column advance.
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

    fn dummy_inputs<'a>() -> GrouscanInputs<'a> {
        GrouscanInputs {
            column: &DUMMY_COLUMN,
            gylookup: &DUMMY_GYLOOKUP,
            gcsub: &DUMMY_GCSUB,
        }
    }

    #[test]
    fn prologue_caches_cf_seed_state() {
        let mut s = fresh_scratch();
        s.gpz = [1_000, 2_000];
        s.gdz = [10, 20];
        s.gxmax = 999_999;
        let p = grouscan_run(&mut s, &dummy_inputs(), 0, 50_000, 1);
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
        let p = grouscan_run(&mut s, &dummy_inputs(), 0, 1, 1);
        assert_eq!(p.lane, 0);

        // Reverse: gpz[1] = 500 < gpz[0] = 800 → lane 1 wins.
        let mut s = fresh_scratch();
        s.gpz = [800, 500];
        s.gdz = [10, 20];
        let p = grouscan_run(&mut s, &dummy_inputs(), 0, 1, 1);
        assert_eq!(p.lane, 1);
    }

    #[test]
    fn prologue_advances_winning_lane_by_gdz() {
        // gpz[0] = 1000 wins; gdz[0] = 256. After: gpz[0] = 1256.
        let mut s = fresh_scratch();
        s.gpz = [1_000, 2_000];
        s.gdz = [256, 999];
        let _ = grouscan_run(&mut s, &dummy_inputs(), 0, 1, 1);
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
        let p = grouscan_run(&mut s, &dummy_inputs(), 0, 50_000, 2);
        assert_eq!(p.ngxmax, 50_000);

        // Single-mip case: ngxmax = gxmax regardless of gxmip.
        let mut s = fresh_scratch();
        s.gpz = [1_000, 2_000];
        s.gdz = [10, 20];
        s.gxmax = 1_000_000;
        let p = grouscan_run(&mut s, &dummy_inputs(), 0, 50_000, 1);
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
        };
        GrouscanState::from_seed(scratch, &inputs, 0)
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
        };
        GrouscanState::from_seed(scratch, &inputs, vptr_offset)
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
        };
        GrouscanState::from_seed(scratch, &inputs, vptr_offset)
    }

    #[test]
    fn predrawceil_swaps_ogx_and_gx() {
        let mut s = fresh_scratch();
        let inputs = dummy_inputs();
        let mut state = GrouscanState::from_seed(&mut s, &inputs, 0);
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
    fn dispatch_drawflor_when_camera_at_top_of_column() {
        let mut s = fresh_scratch();
        s.gpz = [1_000, 2_000];
        s.gdz = [10, 20];
        let p = grouscan_run(&mut s, &dummy_inputs(), 0, 1, 1);
        assert_eq!(p.dispatch, InitialDispatch::DrawFlor);
    }

    #[test]
    fn dispatch_drawceil_when_camera_in_interior_air_gap() {
        let mut s = fresh_scratch();
        s.gpz = [1_000, 2_000];
        s.gdz = [10, 20];
        // vptr_offset > 0 → camera in interior.
        let p = grouscan_run(&mut s, &dummy_inputs(), 16, 1, 1);
        assert_eq!(p.dispatch, InitialDispatch::DrawCeil);
    }

    #[test]
    fn prologue_ogx_keeps_integer_part_of_gpz() {
        // gpz[0] = 0x12345678 → ogx = 0x12340000.
        let mut s = fresh_scratch();
        s.gpz = [0x1234_5678, 0x7FFF_FFFF];
        s.gdz = [0, 0];
        let p = grouscan_run(&mut s, &dummy_inputs(), 0, 1, 1);
        assert_eq!(p.lane, 0);
        // 0x1234_0000 fits i32 positively (high bit clear).
        assert_eq!(p.ogx, 0x1234_0000_i32);
        assert_eq!(p.gx, 0);
    }
}
