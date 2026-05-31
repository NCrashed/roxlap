//! grouscan = `gline`'s per-ray voxel-column raycaster — port of
//! `voxlap5.c:grouscanasm_scalar` (~600 lines, voxlap5.c:11575).
//!
//! Substaged across R4.3c..f, mirroring voxlaptest's own grouscan
//! port (Stage 4.5b.2..6):
//!
//! - **R4.3c (this commit)**: cftype data model + `grouscan_run`
//!   prologue. Caches the `cf[128]` seed slot's state into local
//!   scalars, picks the leading raycast lane. The dispatch skeleton
//!   + draw-phase stubs land in R4.3d.
//! - **R4.3d**: drawcwall / drawfwall / drawceil / drawflor stubs +
//!   the prologue's `v == *ixy_sptr_col ? drawflor : drawceil`
//!   initial dispatch.
//! - **R4.3e**: findslab / slab-split / deletez column advance.
//! - **R4.3f**: remiporend (mip transition) + startsky.

// Several scratch structs preserve voxlap-C state for parity even when
// individual fields aren't yet read (e.g. SkyRef::row_stride is derived
// in from_sky but the rasterizer indexes via lat[]). Module-level
// allow keeps the parity-driven layout intact without per-field churn.
#![allow(dead_code)]

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
    /// S4B.6.l: chunk-z layer this cf entry reads voxel data from.
    /// During multi-chz rendering, each cf entry can map to a
    /// different chz layer so the rasterizer renders content from
    /// multiple layers in one pass (= cf-splitting at chz
    /// boundaries). Initialised to `state.current_chunk_z` at seed
    /// time and inherited on slab_split. The cf-pop handler
    /// (`phase_after_delete_kept_presync`) reloads
    /// `state.{column, slab_buf, column_offsets, mip_base_offsets,
    /// chunk_world_z_base}` when the new top-of-stack entry's chz
    /// differs from the popped one.
    pub chz_layer: i32,
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
use crate::sky::Sky;

/// Borrowed read-only view of an [`crate::sky::Sky`] resource —
/// the subset `phase_startsky`'s textured-fill branch reads. Built
/// from a `&Sky` once per ray (the per-ray sky-row state lives on
/// [`ScanScratch::sky_off`] / [`ScanScratch::sky_cur_lng`]).
#[derive(Clone, Copy)]
pub struct SkyRef<'a> {
    /// Pixel grid; voxlap-style packed BGRA i32. Row `y` starts at
    /// `pixels[y * row_stride]`.
    pub pixels: &'a [i32],
    /// Latitude lookup table — packed `(xoff << 16) | (-yoff &
    /// 0xffff)`. Length = `xsiz_post + 1`. `lat[0] = 0` is the
    /// asm-search lower-bound sentinel.
    pub lat: &'a [i32],
    /// Post-decrement column count (matches voxlap's `skyxsiz`
    /// after `loadsky`'s "skyxsiz--; //Hack" stamp). The latitude
    /// search starts from this index as voxlap's initial `edi`.
    pub xsiz_post: i32,
    /// Row stride in `i32` elements (= `xsiz_post + 1` = pre-
    /// decrement column count = `bpl / 4`).
    pub row_stride: i32,
}

impl<'a> SkyRef<'a> {
    /// Borrow a [`Sky`] for one rasterizer call.
    #[must_use]
    pub fn from_sky(sky: &'a Sky) -> Self {
        Self {
            pixels: &sky.pixels,
            lat: &sky.lat,
            xsiz_post: sky.xsiz,
            row_stride: sky.bpl / 4,
        }
    }
}

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
    /// Per-column byte offsets into `slab_buf`, concatenated across
    /// every built mip level. Mip-0's sub-table is the prefix
    /// (`vsid² + 1` entries) — pre-multi-mip callers passing
    /// `&vxl.column_offset` keep working. R4.5d's `phase_remiporend`
    /// will switch to indexing via `mip_base_offsets[gmipcnt + 1]`
    /// to land in mip-N+1's sub-table.
    pub column_offsets: &'a [u32],
    /// `mip_base_offsets[mip]` is the start index of mip-N's
    /// sub-table in [`Self::column_offsets`]. Length
    /// `mip_count + 1`; the trailing sentinel equals
    /// `column_offsets.len()`. Single-mip callers pass
    /// `&[0, vsid² + 1]`.
    pub mip_base_offsets: &'a [usize],
    /// World dimension at mip-0. Power of two (voxlap-canonical
    /// values are 1024 / 2048). `phase_remiporend` consumes this
    /// to derive the y-parity bit position
    /// (`log2(vsid >> gmipcnt)`) when rebasing the column index
    /// across a mip transition.
    pub vsid: u32,
    /// Optional sky texture borrow. `None` ⇒ `phase_startsky`
    /// always solid-fills with `scratch.skycast` (the existing
    /// behaviour). `Some(_)` ⇒ `phase_startsky` runs the textured
    /// path when `scratch.sky_off != 0`.
    pub sky: Option<SkyRef<'a>>,
    /// S4B.2.c.2: parent (whole-grid) [`GridView`]. For multi-chunk
    /// grids this carries the [`ChunkGrid`] backend the column-step
    /// swap consults via [`crate::grid_view::GridView::chunk_at_xy`].
    /// For single-chunk callers it's the same view the rasterizer
    /// holds (chunk_grid: None — chunk_at_xy degenerates to
    /// Some(Self) for [0, 0]).
    ///
    /// Distinct from the flat `(slab_buf, column_offsets,
    /// mip_base_offsets, vsid)` fields above: those are the
    /// **camera chunk's** per-chunk borrows, set by gline's seed
    /// path so the slab walker starts on the correct chunk.
    pub grid_view: crate::grid_view::GridView<'a>,
    /// S4B.6.b: chunk-z layer the camera sits in. Constant per
    /// ray; the column-step's chunk-XY swap uses this to query
    /// `chunk_at_xyz([new_chx, new_chy, camera_chunk_z])` instead
    /// of the legacy 2D `chunk_at_xy`. For `chunks_z == 1` grids
    /// this is `0` and the dispatch degenerates to today's path.
    ///
    pub camera_chunk_z: i32,
    /// S4B.6.c: world-z offset for the slab data the walker starts
    /// at. `world_z = column_local_z + chunk_world_z_base`. Seeded
    /// to `camera_chunk_z * chunk_size_z` so the cf seed's z0/z1
    /// are in world-z from the start. The slab walker bumps this
    /// when crossing a chunk-z boundary.
    pub chunk_world_z_base: i32,
    /// S4B.6.c: Z extent of one chunk in voxel units (= 256). Used
    /// by the slab walker's chunk-z handoff to compute the next
    /// chunk's world-z base.
    pub chunk_size_z: u32,
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
    ///
    /// VC.1: owned [`Vec<u8>`] (was `&'a [u8]`). On every column
    /// install, [`install_owned_column`] copies the slab chain's
    /// bytes from `slab_buf[off..]`. Subsequent reads (`column[i]`
    /// / `column.get(i)`) deref to the owned slice — byte-identical
    /// to the pre-VC.1 borrow path because [`slab_chain_byte_len`]
    /// bounds the copy at the chain's natural end and the rasterizer
    /// never reads past it. Decouples the lifetime of the column data
    /// from `slab_buf`, which VC.3 will use to inject world-z bytes
    /// without touching the parent borrow.
    pub column: Vec<u8>,
    /// `gylookoff` window into the per-frame gylookup table.
    pub gylookup: &'a [i32],
    /// Per-side shading table.
    pub gcsub: &'a [i64; 9],
    /// World-level flat slab buffer (see [`GrouscanInputs`]).
    pub slab_buf: &'a [u8],
    /// Per-column byte offsets into [`Self::slab_buf`], concatenated
    /// across all built mip levels (see [`GrouscanInputs`]).
    pub column_offsets: &'a [u32],
    /// Per-mip column-offset sub-table base indices (see
    /// [`GrouscanInputs`]).
    pub mip_base_offsets: &'a [usize],
    /// World dimension at mip-0 (see [`GrouscanInputs::vsid`]).
    pub vsid: u32,
    /// Sky texture borrow (see [`GrouscanInputs::sky`]).
    pub sky: Option<SkyRef<'a>>,

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
    /// S1.Z: signed column coordinates carried alongside
    /// [`Self::ixy_sptr_col_idx`] so the column-step path can
    /// classify out-of-bounds reads independently of the wrapping
    /// u32 linear index. For in-bounds camera, `(cx, cy)` track the
    /// same column the linear index points at; for outside-XY camera
    /// they go negative or `>= vsid` while the linear index wraps
    /// to whatever u32 arithmetic yields. Advanced in
    /// [`phase_after_delete_kept_presync`] by ±1 per axis based on
    /// `lane` and `sign(gixy[lane])`.
    pub cx: i32,
    pub cy: i32,
    /// Mip-N voxel coords; identical to `(cx, cy)` until the first
    /// `phase_remiporend` halves them. `cx_mip`/`cy_mip` advance by
    /// `±1` per column step in mip-N (one mip-N voxel) while `cx`/`cy`
    /// keep advancing by `±1` per step (which is geometrically wrong
    /// in mip-N — they'd represent one mip-N column-step's worth of
    /// motion, not one mip-N voxel). The column step uses `cx`/`cy`
    /// at mip-0 (so the OOB check works against `vsid_signed` in
    /// mip-0 voxel units) and `cx_mip`/`cy_mip` at mip-N (so chunk
    /// boundary detection and `correct_idx` use mip-N voxel units).
    pub cx_mip: i32,
    pub cy_mip: i32,
    /// `vsid as i32`, cached so the OOB check in the column-step
    /// path doesn't re-cast every iteration. Mip transitions rescale
    /// this — see [`phase_remiporend`].
    pub vsid_signed: i32,

    /// Voxlap's `gmipcnt` — current mip level walked. Starts at 0;
    /// incremented inside `remiporend` each time the column step's
    /// `gpz > ngxmax` overflow fires. Single-mip rendering
    /// (`gmipnum == 1`) never increments it.
    pub gmipcnt: i32,
    /// Voxlap's `gmipnum` — total mip levels available. Constant
    /// per ray; copied from `grouscan_run`'s parameter.
    pub gmipnum: u32,

    // -----------------------------------------------------------------
    // S4B.2.b — chunk-aware column-step scaffold. For today's
    // single-chunk callers (chunk_size_xy == vsid) the chunk-swap
    // branch in `phase_after_delete_kept_presync` is dead code:
    // the only chunk boundary lies at cx=vsid which the world-edge
    // OOB check already covers. S4B.2.c introduces multi-chunk
    // callers where these fields drive cross-chunk dispatch.
    // -----------------------------------------------------------------
    /// Per-frame voxel-world borrow. Carries `chunk_at_xy`, the
    /// lookup the column step uses to swap active per-chunk
    /// `(slab_buf, column_offsets)` views across chunk boundaries.
    pub grid_view: crate::grid_view::GridView<'a>,
    /// Cached `grid_view.chunk_size_xy`. Read once per column step
    /// to compute `(cx, cy) → chunk_idx` so the inner loop doesn't
    /// re-touch the GridView struct fields.
    pub chunk_size_xy: u32,
    /// `log2(chunk_size_xy)` — used to lower the multi-chunk
    /// column-step's `div_euclid` to an arithmetic shift. Always
    /// derivable in debug because `chunk_size_xy` is asserted
    /// power-of-two; in release we trust the invariant.
    pub chunk_size_xy_log2: u32,
    /// `chunk_size_xy - 1` as `i32` — bitwise mask used to derive
    /// chunk-local coords (`local_cx = cx & mask`) in lieu of
    /// `cx - chunk_idx * chunk_size`. Same power-of-two invariant.
    pub chunk_size_xy_mask: i32,
    /// XY index of the chunk the ray currently sits in. Initialised
    /// from `(cx, cy).div_euclid(chunk_size_xy)`; advanced by the
    /// column step when a step crosses a chunk boundary.
    pub current_chunk_idx_xy: [i32; 2],
    /// `false` ⇒ the current chunk is empty (or outside the grid's
    /// AABB) — the column-refresh branch sets `state.column = &[]`
    /// instead of dereferencing `column_offsets`. Tracks the result
    /// of the most recent `chunk_at_xy` call so the
    /// `(slab_buf, column_offsets)` borrows can stay pointed at the
    /// last valid chunk without aliasing.
    pub current_chunk_exists: bool,
    /// S4B.6.b: camera's chunk-z layer. Pinned for the lifetime of
    /// this ray — the column-step's chunk-XY swap calls
    /// `chunk_at_xyz([new_chx, new_chy, camera_chunk_z])` so
    /// rays in a stacked grid always swap into the camera's z
    /// layer. Vertical ray traversal (= incrementing chunk_z when
    /// the ray crosses into chunks above/below) lands in S4B.6.c.
    pub camera_chunk_z: i32,
    /// S4B.6.c: world-z offset of the chunk the slab walker is
    /// currently reading. `world_z = chunk_local_z +
    /// chunk_world_z_base`. The cf entries' `z0` / `z1` carry
    /// world-z; per-slab byte reads (`column[vptr+1]` etc.)
    /// translate via `+ chunk_world_z_base` to compare correctly.
    /// Updates when the walker crosses a chunk-z boundary.
    pub chunk_world_z_base: i32,
    /// S4B.6.c: Z extent of one chunk in voxel units (`= 256`).
    /// Used by the slab walker's chunk-z handoff to compute the
    /// next chunk's base.
    pub chunk_size_z: u32,
    /// S4B.6.c: chunk-z layer the slab walker is currently in.
    /// Initialized to `camera_chunk_z` at seed; the slab walker
    /// increments / decrements as it crosses chunk-z boundaries.
    pub current_chunk_z: i32,
}

impl<'a> GrouscanState<'a> {
    /// Build a fresh state from the cf[128] seed slot. Mirrors
    /// voxlap5.c:11601-11606.
    #[allow(clippy::too_many_arguments)]
    fn from_seed(
        scratch: &'a mut ScanScratch,
        inputs: &GrouscanInputs<'a>,
        vptr_offset: usize,
        ixy_sptr_col_idx: usize,
        cx: i32,
        cy: i32,
        gmipnum: u32,
    ) -> Self {
        let c = scratch.cf[CF_SEED_INDEX];

        // S4B.2.c.2: take the parent multi-chunk GridView from
        // GrouscanInputs (set by the rasterizer's gline seed path).
        // For single-chunk callers `inputs.grid_view.chunk_grid ==
        // None` and `chunk_size_xy == vsid`, so the column-step
        // fast path stays active and goldens are byte-identical.
        let grid_view = inputs.grid_view;
        let chunk_size_xy = grid_view.chunk_size_xy;
        // Power-of-two invariant: enables shift / mask lowering of
        // the column-step's chunk-index split. CHUNK_SIZE_XY is
        // locked to 128 by `roxlap-scene` and any test-only override
        // is expected to honour the same shape.
        debug_assert!(
            chunk_size_xy.is_power_of_two() && chunk_size_xy > 0,
            "chunk_size_xy must be a positive power of two (got {chunk_size_xy})"
        );
        let chunk_size_xy_log2 = chunk_size_xy.trailing_zeros();
        #[allow(clippy::cast_possible_wrap)]
        let chunk_size_xy_mask = (chunk_size_xy - 1) as i32;
        let current_chunk_idx_xy = [cx >> chunk_size_xy_log2, cy >> chunk_size_xy_log2];
        let camera_chunk_z = inputs.camera_chunk_z;
        let current_chunk_exists = grid_view
            .chunk_at_xyz([
                current_chunk_idx_xy[0],
                current_chunk_idx_xy[1],
                camera_chunk_z,
            ])
            .is_some();

        // VC.1: own the seed column's bytes. Walk the slab chain at
        // `inputs.column` (= the camera-XY column's slab data,
        // pre-sliced by gline's seed path from `slab_buf[col_off..]`)
        // and copy only the chain's actual length — the rasterizer
        // never reads past it, so this stays byte-identical to the
        // pre-VC.1 borrow.
        //
        // VC.3: when the grid is a multi-chunk backend AND its z
        // stack fits in u8 (= `chunks_z == 1 && origin_chunk_z == 0`,
        // the only case where the per-slab z translation is
        // guaranteed not to overflow), route the seed install
        // through `build_owned_column_multi_chz` instead. For the
        // single-iteration N = 1 case the output is byte-identical
        // to the bulk extend above; the multi-chz scaffolding is in
        // place for VC.4's z widening to unblock chunks_z > 1.
        let mut column_owned: Vec<u8> = Vec::with_capacity(inputs.column.len().min(8192));
        let use_vc3_multi_chz_seed = grid_view
            .chunk_grid
            .map_or(false, |cg| cg.chunks_z == 1 && cg.origin_chunk_z == 0);
        if use_vc3_multi_chz_seed {
            let cg = grid_view.chunk_grid.expect("guarded above");
            #[allow(clippy::cast_possible_wrap)]
            let starting_chz = cg.origin_chunk_z;
            #[allow(clippy::cast_possible_wrap)]
            let max_chz = starting_chz + cg.chunks_z as i32 - 1;
            let chunk_local_xy = [cx & chunk_size_xy_mask, cy & chunk_size_xy_mask];
            #[allow(clippy::cast_possible_wrap)]
            let chunk_size_z_signed = inputs.chunk_size_z as i32;
            build_owned_column_multi_chz(
                &mut column_owned,
                grid_view,
                current_chunk_idx_xy,
                chunk_local_xy,
                starting_chz,
                max_chz,
                chunk_size_z_signed,
            );
        } else {
            let seed_chain_len = slab_chain_byte_len(inputs.column);
            column_owned.extend_from_slice(&inputs.column[..seed_chain_len]);
        }

        Self {
            scratch,
            column: column_owned,
            gylookup: inputs.gylookup,
            gcsub: inputs.gcsub,
            slab_buf: inputs.slab_buf,
            column_offsets: inputs.column_offsets,
            mip_base_offsets: inputs.mip_base_offsets,
            vsid: inputs.vsid,
            sky: inputs.sky,
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
            cx,
            cy,
            cx_mip: cx,
            cy_mip: cy,
            #[allow(clippy::cast_possible_wrap)]
            vsid_signed: inputs.vsid as i32,
            gmipcnt: 0,
            gmipnum,
            grid_view,
            chunk_size_xy,
            chunk_size_xy_log2,
            chunk_size_xy_mask,
            current_chunk_idx_xy,
            current_chunk_exists,
            camera_chunk_z,
            chunk_world_z_base: inputs.chunk_world_z_base,
            chunk_size_z: inputs.chunk_size_z,
            current_chunk_z: camera_chunk_z,
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
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn grouscan_run(
    scratch: &mut ScanScratch,
    inputs: &GrouscanInputs<'_>,
    vptr_offset: usize,
    ixy_sptr_col_idx: usize,
    cx: i32,
    cy: i32,
    gxmip: i32,
    gmipnum: u32,
) -> GrouscanPrologue {
    let mut state = GrouscanState::from_seed(
        scratch,
        inputs,
        vptr_offset,
        ixy_sptr_col_idx,
        cx,
        cy,
        gmipnum,
    );

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
/// `goto` between these labels; we drive them via the phase driver.
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
    /// R4.3e3 ports only the early-out fast-path
    /// (`gmipcnt + 1 >= gmipnum`) → [`Phase::Startsky`]; the
    /// full mip-transition body is R4.5 work.
    Remiporend,
    /// Voxlap5.c:12120. Sky-fill primitive that drains remaining
    /// cfasm entries with sky pixels. R4.3e4 ships the body;
    /// R4.3e3 stubs to [`Phase::Done`].
    Startsky,
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
    let trace = std::env::var("ROXLAP_TRACE_PHASES").is_ok();
    let mut current = entry;
    let mut step_count = 0u32;
    loop {
        if trace {
            eprintln!(
                "  phase {step_count:4}: {current:?} c={} ce={} z0={} z1={} cx1={} cy1={} ogx={} gx={}",
                state.c_idx, state.ce_idx, state.z0, state.z1, state.cx1, state.cy1, state.ogx, state.gx,
            );
            step_count += 1;
            if step_count > 200 {
                eprintln!("  (truncated)");
                break;
            }
        }
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
            Phase::Startsky => phase_startsky(state),
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
    // Need at least 4 header bytes at vptr_offset; otherwise no
    // front wall to draw.
    if state.vptr_offset + 4 > state.column.len() {
        return Phase::DrawCwall;
    }

    // Voxlap5.c:11646-11648. dv1 = v[1] = top of floor-colour list.
    let dv1 = slab_z_at(state, state.vptr_offset, 1);

    // S4B.6.k bedrock-z guard. Mirror of `phase_draw_flor`'s check:
    // compare against the RAW z1 byte (chunk-local), not the
    // world-z value `slab_z_at` returns. For stacked grids (chz>0)
    // these diverge: at chz=1, mip=2, the z1 byte is `0xff>>2 = 63`
    // but `slab_z_at` adds `chunk_world_z_base >> mip = 64`, yielding
    // world-z 127. The previous `(dv1 as u8) == 0xff>>mip` compared
    // 127 against 63 and missed the bedrock — drawfwall then drew
    // the bedrock placeholder's (0,0,0,0) voxel into the radar,
    // visible as the triangular BLACK wedge in pose D's bottom-left
    // (user-reported 2026-05-17).
    let bedrock_z_at_mip = 0xff_u8 >> (state.gmipcnt as u32);
    if state.scratch.treat_z_max_as_air
        && state.vptr_offset + 1 < state.column.len()
        && state.column[state.vptr_offset + 1] == bedrock_z_at_mip
    {
        return Phase::DrawCwall;
    }

    if dv1 >= state.z1 {
        return Phase::DrawCwall;
    }
    // Cache c->i1 as ebx — the radar offset we walk down from.
    // Voxlap's `c` is the current cf-stack pointer (advances after
    // slab-split via `c++`), so reads/writes target `cf[c_idx]`,
    // NOT the seed slot. Using CF_SEED_INDEX here previously meant
    // post-split rays drew the wrong [i0,i1] range — visible as
    // missing yellow voxels when sphere columns triggered slab-split.
    state.ebx = state.scratch.cf[state.c_idx].i1;

    'outer: loop {
        // -- loop0 (voxlap5.c:11650): per voxel-row setup. --
        state.off = state.z1 - slab_z_at(state, state.vptr_offset, 1);
        state.z1 -= 1;
        // Read 4-byte voxel colour at byte offset off*4 inside slab.
        // off is non-negative here (loop entry guards `dv1 >= z1`),
        // so usize math is safe.
        let row_offset = state.vptr_offset + (state.off as usize) * 4;
        if row_offset + 4 > state.column.len() {
            // Malformed slab — bail out gracefully.
            state.scratch.cf[state.c_idx].i1 = state.ebx;
            state.scratch.cf[state.c_idx].cx1 = state.cx1;
            state.scratch.cf[state.c_idx].cy1 = state.cy1;
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
            state.scratch.cf[state.c_idx].i1 = state.ebx;
            state.scratch.cf[state.c_idx].cx1 = state.cx1;
            state.scratch.cf[state.c_idx].cy1 = state.cy1;
            return Phase::DrawCwall;
        }
        state.gy_raw = state.gylookup[z1_idx];

        // -- loop1 (voxlap5.c:11659): per-pixel inner. --
        loop {
            let test = grouscan_cross_sign(state.cx1, state.cy1, state.ogx, state.gy_raw);
            if test <= 0 {
                // endloop1 (voxlap5.c:11676). Voxel row exhausted.
                if slab_z_at(state, state.vptr_offset, 1) != state.z1 {
                    continue 'outer;
                }
                // c->i1 = ebx, then fall through to drawcwall.
                // S1.V: also write cx1/cy1 so cf entry stays
                // coherent with i1 (= position at post-drain i1).
                // Without this, phase_startsky_textured's
                // cx0+(i1-i0)*gi0 sx_init formula uses stale cx1
                // and the sky checker shears across cf entries
                // that drained from the back end.
                state.scratch.cf[state.c_idx].i1 = state.ebx;
                state.scratch.cf[state.c_idx].cx1 = state.cx1;
                state.scratch.cf[state.c_idx].cy1 = state.cy1;
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
            if state.ebx < state.scratch.cf[state.c_idx].i0 {
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
    // Need 4 header bytes at vptr_offset.
    if state.vptr_offset + 4 > state.column.len() {
        return Phase::PreDrawCeil;
    }

    // Voxlap5.c:11694 — `z1 = v[1]` UNCONDITIONALLY at drawcwall
    // entry (the comment in the C source warns that drawfwall's
    // early-exit path leaves z1 stale otherwise).
    state.z1 = slab_z_at(state, state.vptr_offset, 1);

    // Column-top: no back wall, jump to drawflor's prep.
    if state.vptr_offset == 0 {
        return Phase::PreDrawFlor;
    }

    // Voxlap5.c:11699-11703. v[3] = z0 of this slab (the air-ceiling
    // above it). If it's ≤ the cached z0 there's no back wall above
    // this slab to draw → set z0 = dv3, fall through to drawceil.
    let dv3 = slab_z_at(state, state.vptr_offset, 3);
    if dv3 <= state.z0 {
        state.z0 = dv3;
        return Phase::PreDrawCeil;
    }

    // c->i0 — current cf-stack pointer's i0, NOT the seed. After
    // slab-split this is cf[c_idx], which carries the post-split
    // [i0, i1] range distinct from cf[CF_SEED_INDEX].
    state.ebx = state.scratch.cf[state.c_idx].i0;

    'outer: loop {
        // -- loop2 (voxlap5.c:11706): per voxel-row setup. --
        // off is NEGATIVE here on entry (loop guard `dv3 > z0` ⇒
        // off = z0 - v[3] < 0). Voxlap reads `v[off*4]` which lands
        // BEFORE the slab header — in the previous slab's tail
        // colour bytes. Use isize math so the negative offset is
        // computed correctly relative to vptr_offset.
        state.off = state.z0 - slab_z_at(state, state.vptr_offset, 3);
        state.z0 += 1;
        let row_offset_signed = state.vptr_offset as isize + (state.off as isize) * 4;
        if row_offset_signed < 0 || (row_offset_signed as usize) + 4 > state.column.len() {
            state.scratch.cf[state.c_idx].i0 = state.ebx;
            state.scratch.cf[state.c_idx].cx0 = state.cx0;
            state.scratch.cf[state.c_idx].cy0 = state.cy0;
            state.z0 = slab_z_at(state, state.vptr_offset, 3);
            return Phase::PreDrawCeil;
        }
        let row_offset = row_offset_signed as usize;
        let vox = u32::from_le_bytes(
            state.column[row_offset..row_offset + 4]
                .try_into()
                .expect("4-byte slice"),
        );
        state.color = grouscan_shade(vox, &mut state.mm5_tail, state.gcsub[state.wall_lane]);
        let z0_idx = state.z0 as usize;
        if z0_idx >= state.gylookup.len() {
            state.scratch.cf[state.c_idx].i0 = state.ebx;
            state.scratch.cf[state.c_idx].cx0 = state.cx0;
            state.scratch.cf[state.c_idx].cy0 = state.cy0;
            state.z0 = slab_z_at(state, state.vptr_offset, 3);
            return Phase::PreDrawCeil;
        }
        state.gy_raw = state.gylookup[z0_idx];

        // -- loop3 (voxlap5.c:11714): per-pixel inner. --
        loop {
            let test = grouscan_cross_sign(state.cx0, state.cy0, state.ogx, state.gy_raw);
            if test > 0 {
                // endloop3 (voxlap5.c:11728). Voxel row exhausted.
                if slab_z_at(state, state.vptr_offset, 3) != state.z0 {
                    continue 'outer;
                }
                // c->i0 = ebx, z0 = v[3], fall through to drawceil.
                // S1.V: also write cx0/cy0 so cf entry stays
                // coherent with i0 (= position at post-drain i0).
                // Without this, phase_startsky_textured's sx_init
                // is off by drain_count*gi0 and the sky checker
                // shears under the floor's underside (visible in
                // the below-floor checker_below_floor.ppm test).
                state.scratch.cf[state.c_idx].i0 = state.ebx;
                state.scratch.cf[state.c_idx].cx0 = state.cx0;
                state.scratch.cf[state.c_idx].cy0 = state.cy0;
                state.z0 = slab_z_at(state, state.vptr_offset, 3);
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
            if state.ebx > state.scratch.cf[state.c_idx].i1 {
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
        // S1.Z: empty/OOB column path — column-step instead of
        // popping the cf stack. Same rationale as drawflor's
        // matching early-out: keeps the walk progressing through
        // void columns until either an in-bounds column produces
        // voxels or gxmax fires (Remiporend → Startsky).
        return Phase::AfterDelete;
    }
    state.gy_raw = state.gylookup[z0_idx];

    // Ceiling colour = `v - 4` = previous slab's last voxel. Only
    // safe when vptr_offset >= 4; an interior slab always satisfies
    // this (drawceil isn't reachable at column-top — drawcwall
    // detects column-top first and routes to predrawflor).
    if state.vptr_offset < 4 {
        // S1.Z: see above. For OOB camera the dispatch chain
        // drawflor (column-top) → AfterDelete → column-step →
        // Skipixy3 → DrawFwall (empty) → DrawCwall (empty) →
        // PreDrawCeil → DrawCeil routes here for any column-step
        // landing on another OOB column. Sending it to AfterDelete
        // (column-step again) keeps the cf seed alive so the walk
        // can keep progressing.
        return Phase::AfterDelete;
    }
    let vox_off = state.vptr_offset - 4;
    if vox_off + 4 > state.column.len() {
        return Phase::AfterDelete;
    }
    let vox = u32::from_le_bytes(
        state.column[vox_off..vox_off + 4]
            .try_into()
            .expect("4-byte slice"),
    );

    loop {
        let test = grouscan_cross_sign(state.cx0, state.cy0, state.ogx, state.gy_raw);
        if test > 0 {
            // S1.V: cf entry's i0 advanced via per-iter writes
            // above; sync cx0/cy0 here so they track i0.
            state.scratch.cf[state.c_idx].cx0 = state.cx0;
            state.scratch.cf[state.c_idx].cy0 = state.cy0;
            return Phase::DrawFlor;
        }
        state.cx0 = state.cx0.wrapping_add(state.scratch.gi0);
        state.cy0 = state.cy0.wrapping_add(state.scratch.gi1);

        // Shade per-iteration: mm5_tail carries forward into the
        // pmulhuw broadcast so successive writes differ even with
        // identical `vox`.
        state.color = grouscan_shade(vox, &mut state.mm5_tail, state.gcsub[2]);

        let i0 = state.scratch.cf[state.c_idx].i0;
        if let Some(slot) = state.scratch.radar.get_mut(i0 as usize) {
            slot.col = state.color as i32;
            slot.dist = state.ogx;
        }
        state.scratch.cf[state.c_idx].i0 = i0 + 1;
        if state.scratch.cf[state.c_idx].i0 > state.scratch.cf[state.c_idx].i1 {
            // drawceil exits to deletez direct (voxlap5.c:11766) —
            // NO `ogx ↔ gx` swap. Only drawfwall / drawcwall route
            // through predeletez. Routing through PreDeleteZ here
            // adds a stray swap that perturbs subsequent ogx for
            // 1-bit shading drift on sphere-edge pixels.
            // S1.V: sync cx0/cy0 to track post-drain i0.
            state.scratch.cf[state.c_idx].cx0 = state.cx0;
            state.scratch.cf[state.c_idx].cy0 = state.cy0;
            return Phase::DeleteZ;
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
    // S1.W: optional "below-the-world is air" mode. The slab's top
    // voxel sits at z = column[vptr+1] (= z1c). When that's the
    // bottom of the world (=0xff = MAXZDIM-1), this slab is voxlap's
    // canonical bedrock placeholder — `delslab` clamps every carve's
    // y1 to MAXZDIM-1 so z=255 stays solid even for fully-air
    // dense-grid columns. Bailing to AfterDelete here lets the walk
    // column-step past the bedrock until `gxmax` triggers
    // `Startsky`, which fills the radar with `skycast` (solid OR
    // textured sky depending on the SkyRef binding) — the visual
    // result the user wants when the camera flies under the world.
    //
    // Gated on `treat_z_max_as_air` so the 12 oracle goldens stay
    // byte-identical (oracle's canonical scene has its bedrock
    // hidden behind terrain that reaches z=255 inside multi-voxel
    // slabs, where z1c < 255).
    //
    // S4B.5 mip-N: `generate_mips` halves the bedrock z each level
    // (`(255+1)>>1 - 1 = 127` at mip-1, then `63`, `31`, …). The
    // single-byte mip-N column still has the bedrock as a one-voxel
    // slab at the world bottom; just compare against the mip-shifted
    // sentinel so multi-mip rendering treats sparse-chunk bedrock as
    // air the same way mip-0 does. Without this, the SHIP grid's
    // all-air chunks would render their mip-N bedrock as a black
    // floor under the saucer (user-reported 2026-05-12).
    let bedrock_z_at_mip = 0xff_u8 >> (state.gmipcnt as u32);
    if state.scratch.treat_z_max_as_air
        && state.vptr_offset + 1 < state.column.len()
        && state.column[state.vptr_offset + 1] == bedrock_z_at_mip
    {
        // S4B.6.h: mid-render chunk-Z handoff. Before falling
        // through to AfterDelete (= sky), try to swap state to the
        // chunk at `current_chunk_z + 1` at the same XY — if it
        // exists, the new chunk's column might have a real floor
        // we can draw. After a successful handoff route through
        // `Phase::Skipixy3` so the rasterizer re-walks the new
        // column's slab list from vptr=0: this lets drawcwall set
        // `state.z1` from the new slab's z-byte BEFORE drawflor's
        // gylookup projection runs (re-entering drawflor directly
        // leaves `state.z1` stuck at the previous column's slab z
        // → hill / lower-mountain pixels project to the wrong
        // screen-y on a non-vertical view).
        //
        // `try_handoff_chunk_z_down` bumps `current_chunk_z` by 1
        // per call and bails when chz exceeds the grid's z extent,
        // so back-to-back handoffs walking all-air-bedrock chunks
        // terminate after at most `chunks_z` iterations.
        //
        // Mip-0 only — `try_handoff_chunk_z_down`'s column-index
        // recompute uses the mip-0 stride; multi-mip handoff is
        // unaudited. Gate explicitly so multi-mip paths fall
        // through to the historical sky behaviour.
        if state.gmipcnt == 0 && try_handoff_chunk_z_down(state) {
            return Phase::Skipixy3;
        }
        return Phase::AfterDelete;
    }
    // gy_raw = gylookoff[z1].
    let z1_idx = state.z1 as usize;
    if z1_idx >= state.gylookup.len() {
        // S1.Z: route empty-column drawflor through AfterDelete so
        // the walk column-steps WITHOUT popping the cf stack — the
        // cf seed survives to drive subsequent columns until either
        // an in-bounds column produces voxels or gxmax fires
        // (Remiporend → Startsky). Pre-S1.Z this returned
        // PreDeleteZ → DeleteZ → Done, leaving radar slots at
        // default zero (= black) for OOB camera.
        return Phase::AfterDelete;
    }
    state.gy_raw = state.gylookup[z1_idx];

    // Floor colour = `v + 4` = first voxel byte INSIDE the current
    // slab. Always within column even at column-top (vptr_offset
    // == 0 → slab starts at column[0..4], floor voxel at column[4]).
    let vox_off = state.vptr_offset + 4;
    if vox_off + 4 > state.column.len() {
        // Same S1.Z fix as above — empty / too-short column means
        // there's no voxel data to draw; column-step rather than
        // popping the cf stack.
        return Phase::AfterDelete;
    }
    let vox = u32::from_le_bytes(
        state.column[vox_off..vox_off + 4]
            .try_into()
            .expect("4-byte slice"),
    );

    loop {
        let test = grouscan_cross_sign(state.cx1, state.cy1, state.ogx, state.gy_raw);
        if test <= 0 {
            // enddrawflor (voxlap5.c:11785) → afterdelete. Pops
            // the current cf entry; if the column gets exhausted
            // the chain reaches startsky which fills any
            // remaining radar slots with skycast (= sky colour).
            // Without this route, sky-pointing rays exit drawflor
            // on the first cross-sign test and leave radar at
            // default zeros — those screen rows render as black.
            // S1.V: sync cx1/cy1 to track post-drain i1.
            state.scratch.cf[state.c_idx].cx1 = state.cx1;
            state.scratch.cf[state.c_idx].cy1 = state.cy1;
            return Phase::AfterDelete;
        }
        state.cx1 = state.cx1.wrapping_sub(state.scratch.gi0);
        state.cy1 = state.cy1.wrapping_sub(state.scratch.gi1);

        state.color = grouscan_shade(vox, &mut state.mm5_tail, state.gcsub[3]);

        let i1 = state.scratch.cf[state.c_idx].i1;
        if let Some(slot) = state.scratch.radar.get_mut(i1 as usize) {
            slot.col = state.color as i32;
            slot.dist = state.ogx;
        }
        state.scratch.cf[state.c_idx].i1 = i1 - 1;
        if state.scratch.cf[state.c_idx].i1 < state.scratch.cf[state.c_idx].i0 {
            // drawflor exits to deletez direct (voxlap5.c:11790) —
            // NO `ogx ↔ gx` swap. See drawceil's matching note.
            // S1.V: sync cx1/cy1 to track post-drain i1.
            state.scratch.cf[state.c_idx].cx1 = state.cx1;
            state.scratch.cf[state.c_idx].cy1 = state.cy1;
            return Phase::DeleteZ;
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

    // S1.Z: advance the signed (cx, cy) cursor in lockstep. The
    // ixy_sptr_col_idx u32-wrap path silently aliases out-of-bounds
    // indices to in-range columns when the camera sits past
    // [0, vsid)²; (cx, cy) preserve the geometric truth.
    //
    // gixy[0] = ±1 (one x-column step) when lane=0; gixy[1] = ±vsid
    // (one y-column step) when lane=1. The `signum` recovers the
    // ±1 cx/cy increment. In mip-N the mip-0 `cx`/`cy` advance is
    // geometrically wrong (one column step = one mip-N voxel = 2^N
    // mip-0 voxels) but the values are unused — the mip-N branch
    // below uses `cx_mip`/`cy_mip` instead.
    if state.lane == 0 {
        state.cx += state.scratch.gixy[0].signum();
        state.cx_mip += state.scratch.gixy[0].signum();
    } else {
        state.cy += state.scratch.gixy[1].signum();
        state.cy_mip += state.scratch.gixy[1].signum();
    }

    // S4B.2.b: chunk-XY boundary detection + swap. Two code paths
    // routed by `grid_view.chunk_grid.is_none()`:
    //
    // - **Single-chunk fast path** (`chunk_grid: None` — every
    //   `from_single_vxl` / `from_parts` caller). The world-edge
    //   OOB check IS the chunk boundary, so the chunk-swap is
    //   purely additive overhead — skip it. Refresh state.column
    //   via the pre-S4B.2.b flat lookup, byte-identical to the
    //   goldens.
    // - **Multi-chunk path** (`chunk_grid: Some(&...)` — Approach B
    //   callers via `GridView::from_chunk_grid`).
    //   `cx.div_euclid(chunk_size_xy)` yields the chunk index; on
    //   boundary crossings `chunk_at_xy` swaps the active per-chunk
    //   `(slab_buf, column_offsets, mip_base_offsets, vsid)`. Empty
    //   chunks (`chunk_at_xy → None`) keep borrows pinned at the
    //   previous chunk and mark `current_chunk_exists = false` so
    //   the column refresh resolves to `&[]`.
    //
    // S4B.5 multi-mip: in mip-N (gmipcnt > 0) the column-step uses
    // `cx_mip`/`cy_mip` instead of `cx`/`cy`. `mip_base_offsets[gmipcnt]`
    // pins the chunk's mip-N sub-table; `vsid_at_mip = chunk_size_xy
    // >> gmipcnt` is the mip-N stride; chunk crossings are detected
    // when `cx_mip / chunk_size_at_mip` changes. Voxlap-C tracks the
    // equivalent state via `gxmipk`/`gymipk` masks on raw `esi`
    // (LP32-baked); our port re-derives the same indices from
    // explicit mip-N voxel coords for portability + multi-chunk.
    if state.grid_view.chunk_grid.is_none() {
        // Single-chunk: pre-S4B.2.b column refresh, with a mip-N
        // branch that derives the in-chunk mip-N column index from
        // (cx_mip, cy_mip).
        if state.gmipcnt > 0 {
            let gmip = state.gmipcnt as u32;
            let chunk_size_at_mip = (state.chunk_size_xy >> gmip) as i32;
            let mip_base = state.mip_base_offsets[state.gmipcnt as usize];
            let in_bounds = state.cx_mip >= 0
                && state.cy_mip >= 0
                && state.cx_mip < chunk_size_at_mip
                && state.cy_mip < chunk_size_at_mip;
            if in_bounds {
                #[allow(clippy::cast_sign_loss)]
                let correct_idx = mip_base
                    + ((state.cy_mip as u32) * (chunk_size_at_mip as u32) + (state.cx_mip as u32))
                        as usize;
                state.ixy_sptr_col_idx = correct_idx;
                if let Some(&col_off) = state.column_offsets.get(correct_idx) {
                    let off = col_off as usize;
                    if off <= state.slab_buf.len() {
                        install_owned_column(&mut state.column, state.slab_buf, off);
                    } else {
                        state.column.clear();
                    }
                } else {
                    state.column.clear();
                }
            } else {
                state.column.clear();
            }
        } else {
            let in_bounds = state.cx >= 0
                && state.cy >= 0
                && state.cx < state.vsid_signed
                && state.cy < state.vsid_signed;
            if in_bounds {
                #[allow(clippy::cast_sign_loss)]
                let correct_idx = (state.cy as u32)
                    .wrapping_mul(state.vsid)
                    .wrapping_add(state.cx as u32) as usize;
                state.ixy_sptr_col_idx = correct_idx;
                if let Some(&col_off) = state.column_offsets.get(correct_idx) {
                    let off = col_off as usize;
                    if off <= state.slab_buf.len() {
                        install_owned_column(&mut state.column, state.slab_buf, off);
                    }
                }
            } else {
                state.column.clear();
            }
        }
    } else {
        // Multi-chunk: detect chunk-XY boundary, swap on transition.
        //
        // Lowering: `chunk_size_xy` is a positive power of two (the
        // debug_assert in `from_seed` guards this), so the
        // chunk-index split uses arithmetic shift / bitwise mask
        // instead of `div_euclid` / `mul-sub`. Arithmetic shift on
        // a signed `i32` rounds toward negative infinity, matching
        // `div_euclid`; the `& mask` always lands in
        // `[0, chunk_size_xy)` regardless of sign, matching
        // `rem_euclid`.
        if state.gmipcnt > 0 {
            // S4B.5 mip-N path. `(cx_mip, cy_mip)` are mip-N voxel
            // coords; the chunk index is the SAME under mip-0 and
            // mip-N (chunks live at fixed world positions; mip
            // levels are just sub-tables inside the chunk).
            // `cx_mip >> log2(chunk_size_at_mip) = chunk_xy`.
            let gmip = state.gmipcnt as u32;
            let chunk_size_at_mip = (state.chunk_size_xy >> gmip) as i32;
            // chunk_size_xy is power-of-two, so chunk_size_at_mip is too.
            let chunk_size_at_mip_mask = chunk_size_at_mip - 1;
            let chunk_size_at_mip_log2 = state.chunk_size_xy_log2 - gmip;
            let new_chunk_xy = [
                state.cx_mip >> chunk_size_at_mip_log2,
                state.cy_mip >> chunk_size_at_mip_log2,
            ];
            if new_chunk_xy != state.current_chunk_idx_xy {
                state.current_chunk_idx_xy = new_chunk_xy;
                if let Some(new_chunk) = state.grid_view.chunk_at_xyz([
                    new_chunk_xy[0],
                    new_chunk_xy[1],
                    state.current_chunk_z,
                ]) {
                    state.slab_buf = new_chunk.slab_buf;
                    state.column_offsets = new_chunk.column_offsets;
                    state.mip_base_offsets = new_chunk.mip_base_offsets;
                    state.vsid = new_chunk.vsid;
                    #[allow(clippy::cast_possible_wrap)]
                    {
                        state.vsid_signed = new_chunk.vsid as i32;
                    }
                    state.current_chunk_exists = true;
                } else {
                    state.current_chunk_exists = false;
                }
            }
            // Recompute `correct_idx` from the active mip-N tables
            // regardless of `current_chunk_exists`. When the next
            // chunk doesn't exist, `mip_base_offsets` / `column_offsets`
            // are still pinned at the previous (existing) chunk's
            // tables; using THEIR `mip_base + local_y * stride +
            // local_x` keeps `ixy_sptr_col_idx` inside
            // `state.mip_base_offsets[gmipcnt]`'s sub-table so a
            // subsequent [`phase_remiporend`] subtraction can't
            // underflow. Without this fix, the wrap-add from above
            // can leave `ixy_sptr_col_idx < mip_base_offsets[gmipcnt]`
            // (= the S5.2-followup disappearing-ship panic).
            // `state.column` still drops to `&[]` when the chunk is
            // absent, so empty-chunk rays correctly draw no voxels.
            let local_cx_mip = state.cx_mip & chunk_size_at_mip_mask;
            let local_cy_mip = state.cy_mip & chunk_size_at_mip_mask;
            let mip_base = state.mip_base_offsets[state.gmipcnt as usize];
            #[allow(clippy::cast_sign_loss)]
            let correct_idx = mip_base
                + ((local_cy_mip as u32) * (chunk_size_at_mip as u32) + (local_cx_mip as u32))
                    as usize;
            state.ixy_sptr_col_idx = correct_idx;
            if state.current_chunk_exists {
                if let Some(&col_off) = state.column_offsets.get(correct_idx) {
                    let off = col_off as usize;
                    if off <= state.slab_buf.len() {
                        install_owned_column(&mut state.column, state.slab_buf, off);
                    } else {
                        state.column.clear();
                    }
                } else {
                    state.column.clear();
                }
            } else {
                state.column.clear();
            }
        } else {
            let log2 = state.chunk_size_xy_log2;
            let mask = state.chunk_size_xy_mask;
            let new_chunk_xy = [state.cx >> log2, state.cy >> log2];
            if new_chunk_xy != state.current_chunk_idx_xy {
                state.current_chunk_idx_xy = new_chunk_xy;
                if let Some(new_chunk) = state.grid_view.chunk_at_xyz([
                    new_chunk_xy[0],
                    new_chunk_xy[1],
                    state.current_chunk_z,
                ]) {
                    state.slab_buf = new_chunk.slab_buf;
                    state.column_offsets = new_chunk.column_offsets;
                    state.mip_base_offsets = new_chunk.mip_base_offsets;
                    state.vsid = new_chunk.vsid;
                    #[allow(clippy::cast_possible_wrap)]
                    {
                        state.vsid_signed = new_chunk.vsid as i32;
                    }
                    state.current_chunk_exists = true;
                } else {
                    state.current_chunk_exists = false;
                }
            }

            // Chunk-local coords via mask. `local_cx` is in
            // `[0, chunk_size_xy)` for any signed `cx`, so the prior
            // explicit range checks are redundant.
            if state.current_chunk_exists {
                let local_cx = state.cx & mask;
                let local_cy = state.cy & mask;
                #[allow(clippy::cast_sign_loss)]
                let correct_idx = (local_cy as u32)
                    .wrapping_mul(state.chunk_size_xy)
                    .wrapping_add(local_cx as u32) as usize;
                state.ixy_sptr_col_idx = correct_idx;
                if let Some(&col_off) = state.column_offsets.get(correct_idx) {
                    let off = col_off as usize;
                    if off <= state.slab_buf.len() {
                        install_owned_column(&mut state.column, state.slab_buf, off);
                    }
                }
            } else {
                state.column.clear();
            }
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

    // CF.3.C — per-column-step phantom narrowing. Apply drawfwall +
    // drawcwall-shape narrowing on the active cf entry (cf[ce_idx],
    // which the next drawfwall/cwall will load via SyncFromPresync)
    // using `virtual_ogx = state.gx` (the just-computed new column's
    // depth integer-part).
    //
    // Insight from CF.3.B diagnostic: only 0.84 % of cf entries at
    // remiporend entry trigger drawfwall narrowing. The bug pixels
    // reach the cancellation regime DURING drawfwall at the new mip,
    // NOT at remiporend. Per-column narrowing fires at every column
    // step (instead of once per mip transition), catching narrowings
    // at the moment they should fire.
    //
    // Gated behind `ROXLAP_CF_NARROW_PER_COLUMN=1` so default flow
    // stays byte-stable. mip-0 (gmipcnt == 0) is excluded — at mip-0
    // drawfwall does all narrowing natively.
    if state.gmipcnt > 0 && std::env::var_os("ROXLAP_CF_NARROW_PER_COLUMN").is_some() {
        // ROXLAP_CF_NARROW_PER_COLUMN_NO_I1=1 — skip the i1 decrement.
        // Earlier CF.3.B diagnostic showed cx1/cy1 narrowing alone is
        // observably a no-op (gi0 magnitude is sufficient to flip
        // cx_s16, but the cf entries the simulator narrows don't
        // reach the bug pixels' drawfwall fire path). Use this to
        // confirm: if NO_I1 = baseline, the i1 decrement is the
        // entire source of regression in per-column too.
        let skip_i1 = std::env::var_os("ROXLAP_CF_NARROW_PER_COLUMN_NO_I1").is_some();
        let virtual_ogx = state.gx;
        let entry = &mut state.scratch.cf[state.ce_idx];

        // drawfwall-side (i1 / cx1 / cy1).
        let z1_idx = entry.z1 as usize;
        if z1_idx < state.gylookup.len() {
            let gy_raw_fwall = state.gylookup[z1_idx];
            let test = grouscan_cross_sign(entry.cx1, entry.cy1, virtual_ogx, gy_raw_fwall);
            if test > 0 && entry.i1 > entry.i0 {
                entry.cx1 = entry.cx1.wrapping_sub(state.scratch.gi0);
                entry.cy1 = entry.cy1.wrapping_sub(state.scratch.gi1);
                if !skip_i1 {
                    entry.i1 -= 1;
                }
            }
        }

        // drawcwall-side (i0 / cx0 / cy0).
        let z0_idx = entry.z0 as usize;
        if z0_idx < state.gylookup.len() {
            let gy_raw_cwall = state.gylookup[z0_idx];
            let test = grouscan_cross_sign(entry.cx0, entry.cy0, virtual_ogx, gy_raw_cwall);
            if test <= 0 && entry.i0 < entry.i1 {
                entry.cx0 = entry.cx0.wrapping_add(state.scratch.gi0);
                entry.cy0 = entry.cy0.wrapping_add(state.scratch.gi1);
                if !skip_i1 {
                    entry.i0 += 1;
                }
            }
        }
    }

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

/// `intoslabloop` — voxlap5.c:11863-11957.
///
/// 1. `v2 = v[2]` (slab's solid-bottom z).
/// 2. `gy_raw = gylookoff[v2 + 1]`.
/// 3. `test_hi = cross_sign(cx0, cy0, ogx, gy_raw)`. If `> 0`
///    the slab is still above the ray's frustum top — route to
///    [`Phase::Findslabloop`] to advance.
/// 4. Else (slab intersects): also test the NEXT slab's
///    `next_v3 = v[v0*4 + 3]` against `cx1/cy1`.
///    - `test_next <= 0` → single-slab dispatch
///      ([`Phase::DrawFwall`]).
///    - `test_next > 0` → two-slab cfasm split via
///      [`do_slab_split`]; that helper pushes a new cf entry,
///      narrows the current one, advances `c`, then returns
///      [`Phase::DrawFwall`].
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
    // S4B.6.c: `v2` is a z-byte (z1c); translate via `slab_z_at`.
    let v2 = slab_z_at(state, state.vptr_offset, 2);
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

    // Slab intersects. Test the NEXT slab to decide
    // single-vs-split.
    let v0 = i32::from(column_byte_at(state, 0));
    // `v0` is nextptr (NOT a z value) — read raw. `next_v3` is
    // a z-byte of the NEXT slab; translate via slab_z_at.
    let next_v3 = slab_z_at(state, state.vptr_offset + (v0 as usize) * 4, 3);
    let next_gy_idx = next_v3 as usize;
    if next_gy_idx >= state.gylookup.len() {
        return Phase::DrawFwall;
    }
    state.gy_raw = state.gylookup[next_gy_idx];
    let test_next = grouscan_cross_sign(state.cx1, state.cy1, state.ogx, state.gy_raw);
    if test_next <= 0 {
        // Single-slab case — voxlap's `jle drawfwall`.
        return Phase::DrawFwall;
    }

    // Two-slab split.
    do_slab_split(state, v2, next_v3)
}

/// Two-slab cfasm split — voxlap5.c:11880-11957. Called from
/// [`phase_intoslabloop`] when both the current and the next
/// slab intersect the ray's frustum.
///
/// 1. Save current scalars to `cf[c]` (so the about-to-be-
///    duplicated entry holds the pre-split state).
/// 2. Reset `gy_raw = gylookoff[v2 + 1]` (the asm's mm3 for the
///    column search; the next-slab test above clobbered it).
/// 3. Search for the split column `col` walking from `c->i1`
///    leftward, with `cx1`/`cy1` decrementing by `gi0`/`gi1`
///    per column. Two-rate, mirroring voxlap asm's
///    `prebegsearchi16` + `begsearchi`: big-step backward by 16
///    cols until the next big step would overshoot the sign
///    transition, then single-step the 0..15 residual. Hash-
///    neutral vs the per-step form (same transition, same col),
///    purely a perf win — voxlap profiling estimated 1-15% on
///    the scanline render path of a CPU renderer.
/// 4. Stack-overflow check: voxlap caps `ce` at `cf[191]`
///    (`cmp eax, _cfasm[4096]`). Past that we bail to
///    [`Phase::Done`] (the asm's `retsub`).
/// 5. Push new entry (`ce++`), then shift entries `(c, ce]` up
///    by one slot — the C `for (p=ce; p>c; p--) *p = *(p-1)`
///    duplicates `cf[c]` into `cf[c+1]` in the final pass.
/// 6. Modify split fields:
///    - `cf[c+1].i1 = col` (narrowed right edge for the
///      BEFORE-split range).
///    - `cf[c].i0 = col + 1`, `cf[c].z0 = next_v3`, and
///      `cf[c].cx0/cy0 = cx1+gi0/cy1+gi1` (split-point ray).
/// 7. Advance `c++` into the new top slot. `cf[c]` (new) holds
///    the original via shift-copy; the locals' `cx1/cy1` carry
///    the search-end values, which is what drawfwall wants for
///    its right-edge walk.
/// 8. Restore `z0 = c->z0` (= original z0, unchanged via
///    shift-copy) and set `z1 = next_v3`. Voxlap's
///    `mov edx, eax` here is functionally non-trivial — leaving
///    `z1` stale at the pre-split value makes drawfwall iterate
///    the wrong number of times and bleeds garbage past the
///    slab's visible range (the project-memory note about the
///    oracle's `sprite_iso` / `diag_down` ball artifacts traces
///    back to this).
//
// Bit-narrowings + signed/unsigned isize math; voxlap names
// retained.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::similar_names
)]
fn do_slab_split(state: &mut GrouscanState<'_>, v2: i32, next_v3: i32) -> Phase {
    // 1. Save current scalars to cf[c] (voxlap5.c:11890-11892).
    {
        let z0 = state.z0;
        let z1 = state.z1;
        let cx0 = state.cx0;
        let cy0 = state.cy0;
        let cx1 = state.cx1;
        let cy1 = state.cy1;
        let c = &mut state.scratch.cf[state.c_idx];
        c.z0 = z0;
        c.z1 = z1;
        c.cx0 = cx0;
        c.cy0 = cy0;
        c.cx1 = cx1;
        c.cy1 = cy1;
    }

    // 2. Reset gy_raw — the next-slab test clobbered it.
    let gy_idx = (v2 + 1) as usize;
    if gy_idx >= state.gylookup.len() {
        return Phase::DrawFwall;
    }
    state.gy_raw = state.gylookup[gy_idx];

    // 3. Two-rate search for the split column. Big-step phase
    //    walks backward by 16 cols at a time (voxlap asm's
    //    `prebegsearchi16`); single-step phase finishes the 0..15
    //    residual (voxlap asm's `begsearchi`). Both phases bounded
    //    defensively against malformed fixtures — geometry
    //    guarantees the cross-sign transition exists in
    //    `[i0, i1]`, but a degenerate (cf-i1 < cf-i0) input would
    //    otherwise spin.
    let mut col = state.scratch.cf[state.c_idx].i1;
    let i0 = state.scratch.cf[state.c_idx].i0;
    let span = (col - i0).max(0) as usize;
    // Big-step (16-col) phase. Pre-compute `gi0 << 4` / `gi1 << 4`
    // once; voxlap C uses `(gi0 << 4)` per iteration. Rust's `<<`
    // on i32 with shift amount 4 is well-defined truncating-shift
    // (no overflow panic), matching voxlap's MMX `pslld mm7, 4`
    // behaviour on 32-bit lanes.
    let gi0_16 = state.scratch.gi0 << 4;
    let gi1_16 = state.scratch.gi1 << 4;
    let big_step_max = span / 16 + 1;
    for _ in 0..big_step_max {
        let cx_try = state.cx1.wrapping_sub(gi0_16);
        let cy_try = state.cy1.wrapping_sub(gi1_16);
        if grouscan_cross_sign(cx_try, cy_try, state.ogx, state.gy_raw) <= 0 {
            break;
        }
        state.cx1 = cx_try;
        state.cy1 = cy_try;
        col -= 16;
    }
    // Single-step finish — at most 16 cols residual after the
    // big-step bail-out (transition lies inside the next 16 cols).
    // Capped at 17 to absorb an off-by-one on degenerate fixtures.
    for _ in 0..=16 {
        if grouscan_cross_sign(state.cx1, state.cy1, state.ogx, state.gy_raw) <= 0 {
            break;
        }
        state.cx1 = state.cx1.wrapping_sub(state.scratch.gi0);
        state.cy1 = state.cy1.wrapping_sub(state.scratch.gi1);
        col -= 1;
    }

    // 4. Stack-overflow check — voxlap's cf[191] cap.
    if state.ce_idx >= 191 {
        return Phase::Done;
    }

    // 5. Push + shift cf entries up.
    state.ce_idx += 1;
    for p in (state.c_idx + 1..=state.ce_idx).rev() {
        state.scratch.cf[p] = state.scratch.cf[p - 1];
    }

    // 6. Modify split fields.
    state.scratch.cf[state.c_idx + 1].i1 = col;
    {
        let new_cx0 = state.cx1.wrapping_add(state.scratch.gi0);
        let new_cy0 = state.cy1.wrapping_add(state.scratch.gi1);
        let c = &mut state.scratch.cf[state.c_idx];
        c.i0 = col + 1;
        c.z0 = next_v3;
        c.cx0 = new_cx0;
        c.cy0 = new_cy0;
    }

    // 7. Advance into the new top slot.
    state.c_idx += 1;

    // 8. z0 = c->z0 (= original z0 via shift-copy), z1 = next_v3.
    state.z0 = state.scratch.cf[state.c_idx].z0;
    state.z1 = next_v3;

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

/// VC.1: walk the slab chain at `column[0..]` and return its exact
/// byte length. Matches `roxlap_formats::vxl::parse`'s canonical
/// walker (`crates/roxlap-formats/src/vxl.rs:962`):
///
/// - Non-last slab: `pos += nextptr * 4`.
/// - Last slab (`nextptr == 0`): `4 + max(0, z1c - z1 + 1) * 4`.
///
/// Defensive against truncated / malformed columns: clamps to
/// `column.len()` and bails on `nextptr * 4 < 4` (which would loop
/// forever in voxlap's walker). Returns 0 when the input doesn't
/// have the 4-byte header.
fn slab_chain_byte_len(column: &[u8]) -> usize {
    let mut pos = 0usize;
    loop {
        if pos + 4 > column.len() {
            return column.len();
        }
        let nextptr = column[pos];
        if nextptr == 0 {
            let z1 = i32::from(column[pos + 1]);
            let z1c = i32::from(column[pos + 2]);
            let n_floor_signed = z1c - z1 + 1;
            let n_floor = usize::try_from(n_floor_signed.max(0)).unwrap_or(0);
            let last_size = 4 + n_floor * 4;
            return (pos + last_size).min(column.len());
        }
        let advance = usize::from(nextptr) * 4;
        if advance < 4 {
            return column.len();
        }
        pos = pos.saturating_add(advance);
    }
}

/// VC.1: copy the slab chain rooted at `slab_buf[off..]` into
/// `target`. Computes the chain's exact byte length via
/// [`slab_chain_byte_len`] so reads stay byte-identical to the
/// pre-VC.1 `state.column = &slab_buf[off..]` slice path (the
/// rasterizer never reads past the chain anyway; the bounded copy
/// just avoids allocating + memcpying the slab_buf tail).
///
/// VC.2: delegates to [`build_owned_column_from_chain`], which
/// walks the chain slab-by-slab and copies each slab's bytes
/// individually. Output is byte-identical to the pre-VC.2 bulk
/// `extend_from_slice` path — VC.3 reuses the per-slab loop to
/// concatenate chunks at chz boundaries with z bytes translated
/// to world-z.
fn install_owned_column(target: &mut Vec<u8>, slab_buf: &[u8], off: usize) {
    target.clear();
    build_owned_column_from_chain(target, slab_buf, off);
}

/// VC.2: walk the slab chain at `slab_buf[off..]` and APPEND each
/// slab's bytes individually to `target`. The caller is responsible
/// for clearing `target` first ([`install_owned_column`] does this);
/// keeping the append semantics here lets VC.3's multi-chz path call
/// the builder N times in sequence to concatenate chains across chz
/// layers.
///
/// Mirrors voxlap's canonical chain walk
/// (`roxlap_formats::vxl::parse_columns`,
/// `crates/roxlap-formats/src/vxl.rs:962`):
///
/// - Non-last slab (`nextptr != 0`): total bytes = `nextptr * 4`,
///   advance `pos += nextptr * 4`.
/// - Last slab (`nextptr == 0`): total bytes =
///   `4 + max(0, z1c - z1 + 1) * 4`. Emit and stop.
///
/// Defensive against malformed columns (truncated tail, `nextptr * 4
/// < 4`, last-slab body extending past the slab_buf end) — emits the
/// available bytes and returns. Output is byte-identical to
/// `target.extend_from_slice(&slab_buf[off..off + slab_chain_byte_len(...)]).`
fn build_owned_column_from_chain(target: &mut Vec<u8>, slab_buf: &[u8], off: usize) {
    if off >= slab_buf.len() {
        return;
    }
    let tail = &slab_buf[off..];
    let mut pos = 0usize;
    loop {
        if pos + 4 > tail.len() {
            // Truncated header. Emit whatever's left and stop.
            target.extend_from_slice(&tail[pos..]);
            return;
        }
        let nextptr = tail[pos];
        if nextptr == 0 {
            let z1 = i32::from(tail[pos + 1]);
            let z1c = i32::from(tail[pos + 2]);
            let n_floor = usize::try_from((z1c - z1 + 1).max(0)).unwrap_or(0);
            let last_size = 4 + n_floor * 4;
            let end = (pos + last_size).min(tail.len());
            target.extend_from_slice(&tail[pos..end]);
            return;
        }
        let advance = usize::from(nextptr) * 4;
        if advance < 4 {
            // Malformed: a `nextptr * 4 < 4` advance would loop
            // forever in voxlap's walker. Treat as terminator;
            // emit the malformed slab's 4-byte header so reads at
            // `pos..pos+4` still see the original bytes.
            let end = (pos + 4).min(tail.len());
            target.extend_from_slice(&tail[pos..end]);
            return;
        }
        let next_pos = pos.saturating_add(advance).min(tail.len());
        target.extend_from_slice(&tail[pos..next_pos]);
        if next_pos >= tail.len() {
            return;
        }
        pos = next_pos;
    }
}

/// VC.3: per-slab chain walker that translates each slab's z bytes
/// (header bytes 1 = z1, 2 = z1c, 3 = z0) to world-z by adding
/// `world_z_base`. Other header / voxel-record bytes pass through
/// untouched. Append semantics (caller clears `target`).
///
/// For `world_z_base == 0` produces byte-identical output to
/// [`build_owned_column_from_chain`]; that's the case the VC.3
/// dispatch in [`GrouscanState::from_seed`] actually exercises
/// today. The translation arithmetic is in place so VC.4 can flip
/// the z bytes to u16 / i32 and unblock `chunks_z > 1`.
///
/// VC.3 constraint: panics if any translated z exceeds u8. Guarded
/// by the dispatch which only fires when `origin_chunk_z == 0` AND
/// `chunks_z == 1` (= `world_z_base == 0` for every layer the loop
/// visits).
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn build_owned_column_from_chain_translated(
    target: &mut Vec<u8>,
    slab_buf: &[u8],
    off: usize,
    world_z_base: i32,
) {
    if off >= slab_buf.len() {
        return;
    }
    let tail = &slab_buf[off..];
    let mut pos = 0usize;
    loop {
        if pos + 4 > tail.len() {
            target.extend_from_slice(&tail[pos..]);
            return;
        }
        let nextptr = tail[pos];
        let z1_raw = i32::from(tail[pos + 1]);
        let z1c_raw = i32::from(tail[pos + 2]);
        let z0_raw = i32::from(tail[pos + 3]);
        let z1_w = z1_raw + world_z_base;
        let z1c_w = z1c_raw + world_z_base;
        let z0_w = z0_raw + world_z_base;
        assert!(
            (0..256).contains(&z1_w)
                && (0..256).contains(&z1c_w)
                && (0..256).contains(&z0_w),
            "VC.3 z translation overflows u8 (base={world_z_base}, raw z1={z1_raw}, z1c={z1c_raw}, z0={z0_raw})"
        );
        // Emit translated header.
        target.push(nextptr);
        target.push(z1_w as u8);
        target.push(z1c_w as u8);
        target.push(z0_w as u8);
        if nextptr == 0 {
            // Last slab body — n_floor based on the CHUNK-LOCAL diff
            // (= identical to the world-z diff after translation).
            let n_floor = usize::try_from((z1c_raw - z1_raw + 1).max(0)).unwrap_or(0);
            let body_bytes = n_floor * 4;
            let body_end = (pos + 4 + body_bytes).min(tail.len());
            if pos + 4 < body_end {
                target.extend_from_slice(&tail[pos + 4..body_end]);
            }
            return;
        }
        let advance = usize::from(nextptr) * 4;
        if advance < 4 {
            return;
        }
        let next_pos = pos.saturating_add(advance).min(tail.len());
        // Emit voxel-record bytes between header and next slab —
        // colours pass through untouched.
        if pos + 4 < next_pos {
            target.extend_from_slice(&tail[pos + 4..next_pos]);
        }
        if next_pos >= tail.len() {
            return;
        }
        pos = next_pos;
    }
}

/// VC.3: build the camera-XY column by concatenating chz layers
/// from `starting_chz` to `max_chz` (inclusive). Each layer's slab
/// chain is z-translated by `chz * chunk_size_z` before being
/// appended. The slab walker still reads bytes positionally — but
/// for layers with non-zero `world_z_base`, those bytes now
/// represent world-z directly.
///
/// VC.3 scope: only single-chz stacks (chunks_z = 1) are wired
/// through to this function via the
/// [`GrouscanState::from_seed`] dispatch. For N = 1 the loop runs
/// once, world_z_base = 0, and the output is byte-identical to
/// VC.2's single-chunk install. The N > 1 iteration logic is
/// scaffolding for VC.4+ — the
/// [`build_owned_column_from_chain_translated`] u8-overflow
/// assertion fires before any incorrect output reaches the
/// rasterizer.
///
/// Note: chains are appended back-to-back; the chz=K last-slab's
/// `nextptr == 0` sentinel is NOT rewritten to point at chz=K+1's
/// first slab. That stitching belongs in VC.4 alongside the z
/// widening — without it, a multi-chz walker would terminate at
/// the first chz boundary.
fn build_owned_column_multi_chz(
    target: &mut Vec<u8>,
    grid_view: crate::grid_view::GridView<'_>,
    chunk_xy: [i32; 2],
    chunk_local_xy: [i32; 2],
    starting_chz: i32,
    max_chz: i32,
    chunk_size_z: i32,
) {
    target.clear();
    for chz in starting_chz..=max_chz {
        let Some(chunk) = grid_view.chunk_at_xyz([chunk_xy[0], chunk_xy[1], chz]) else {
            continue;
        };
        #[allow(clippy::cast_possible_wrap)]
        let chunk_size_xy_mask = (chunk.chunk_size_xy as i32) - 1;
        #[allow(clippy::cast_sign_loss)]
        let lx = (chunk_local_xy[0] & chunk_size_xy_mask) as u32;
        #[allow(clippy::cast_sign_loss)]
        let ly = (chunk_local_xy[1] & chunk_size_xy_mask) as u32;
        // Mip-0 sub-table only — seed-time install never reads mip-N.
        let mip_base = chunk.mip_base_offsets[0];
        let col_idx = ly.wrapping_mul(chunk.chunk_size_xy).wrapping_add(lx);
        let table_idx = mip_base + col_idx as usize;
        let Some(&col_off) = chunk.column_offsets.get(table_idx) else {
            continue;
        };
        let world_z_base = chz * chunk_size_z;
        build_owned_column_from_chain_translated(
            target,
            chunk.slab_buf,
            col_off as usize,
            world_z_base,
        );
    }
}

/// S4B.6.c: read a slab z-byte (chunk-local) and translate to
/// world-z by adding `state.chunk_world_z_base`. Use for any
/// byte in {1: z1, 2: z1c, 3: z0} where the result is interpreted
/// as a z coordinate. For nextptr (byte 0) and colour bytes use
/// raw [`column_byte_at`] / slice indexing. When
/// `chunk_world_z_base == 0` (= camera in chz=0 of a non-stacked
/// world) the addition is a no-op, byte-identical with the
/// pre-S4B.6.c bare-byte reads.
///
/// S4B.6.g (mip-N stacked fix): the slab byte at mip level N is in
/// mip-N units (`= chunk_local_z >> N`), while `chunk_world_z_base`
/// is stored in mip-0 / world-z units. Both cf entries and the
/// gylookup index for mip-N are in mip-N units (cf is halved at
/// every `phase_remiporend`), so the offset must shift right by
/// `gmipcnt` to stay in the same scale as the slab byte. Mip-0
/// callers see no change because `>> 0` is identity. The bug it
/// fixes: stacked-grid mip-N rendered the floor at world-z =
/// `byte * 2^N + chunk_world_z_base` instead of
/// `byte * 2^N + (chunk_world_z_base >> N) * 2^N` — for chz=1
/// (base=256) at mip-1 this is a 128-voxel shift toward the camera
/// = the "green wall in a circle around the camera" artifact.
#[inline]
fn slab_z_at(state: &GrouscanState<'_>, vptr_offset: usize, byte: usize) -> i32 {
    let raw = i32::from(
        state
            .column
            .get(vptr_offset.saturating_add(byte))
            .copied()
            .unwrap_or(0),
    );
    raw + (state.chunk_world_z_base >> (state.gmipcnt as u32))
}

/// S4B.6.c: try to swap the slab walker into the chunk at
/// `current_chunk_z + 1` (= one chunk DOWN in voxlap's z-down
/// convention) at the same XY. Returns `true` on success — caller
/// re-enters drawflor / drawcwall / etc. so the chain continues
/// with the new chunk's slab data + world-z base. Returns `false`
/// when no such chunk exists, leaving state unchanged.
///
/// Used by [`phase_draw_flor`]'s bedrock-as-air bypass to extend
/// across stacked chunks (e.g. camera at chz=0 above an empty
/// chunk sees terrain in chz=1's chunk).
///
/// Mip-0 only — multi-mip + chunk-Z handoff is unaudited; the
/// caller gates on `state.gmipcnt == 0`.
fn try_handoff_chunk_z_down(state: &mut GrouscanState<'_>) -> bool {
    // Only meaningful for multi-chunk grids. Single-chunk grids
    // (`chunk_grid: None`) carry one un-stacked chunk — the
    // `chunk_at_xyz` single-chunk shortcut returns Some(self) for
    // any z, which would cause infinite handoff into the same
    // chunk. Skip when there's no real chunk-grid backend.
    let cg = match state.grid_view.chunk_grid {
        Some(cg) => cg,
        None => return false,
    };
    let next_chz = state.current_chunk_z + 1;
    // Bail before querying when next_chz is past the grid's z
    // extent — same effect as `chunk_at_xyz` returning None but
    // skips a lookup.
    #[allow(clippy::cast_possible_wrap)]
    if next_chz >= cg.origin_chunk_z + cg.chunks_z as i32 {
        return false;
    }
    let new_chunk_xyz = [
        state.current_chunk_idx_xy[0],
        state.current_chunk_idx_xy[1],
        next_chz,
    ];
    let new_chunk = match state.grid_view.chunk_at_xyz(new_chunk_xyz) {
        Some(c) => c,
        None => return false,
    };
    state.slab_buf = new_chunk.slab_buf;
    state.column_offsets = new_chunk.column_offsets;
    state.mip_base_offsets = new_chunk.mip_base_offsets;
    state.vsid = new_chunk.vsid;
    #[allow(clippy::cast_possible_wrap)]
    {
        state.vsid_signed = new_chunk.vsid as i32;
    }
    state.current_chunk_exists = true;
    state.current_chunk_z = next_chz;
    #[allow(clippy::cast_possible_wrap)]
    {
        state.chunk_world_z_base += state.chunk_size_z as i32;
    }
    // Recompute column index in the new chunk. Mip-0 path:
    // `cy_local * chunk_size_xy + cx_local`. `cx` / `cy` are
    // signed world coords; `& mask` yields chunk-local for
    // power-of-two `chunk_size_xy`.
    let local_cx = state.cx & state.chunk_size_xy_mask;
    let local_cy = state.cy & state.chunk_size_xy_mask;
    #[allow(clippy::cast_sign_loss)]
    let correct_idx = (local_cy as u32)
        .wrapping_mul(state.chunk_size_xy)
        .wrapping_add(local_cx as u32) as usize;
    state.ixy_sptr_col_idx = correct_idx;
    if let Some(&col_off) = state.column_offsets.get(correct_idx) {
        let off = col_off as usize;
        if off <= state.slab_buf.len() {
            install_owned_column(&mut state.column, state.slab_buf, off);
        } else {
            state.column.clear();
        }
    } else {
        state.column.clear();
    }
    state.vptr_offset = 0;
    true
}

/// `remiporend` — voxlap5.c:11998-12118. Mip-level transition.
///
/// Coarsens the active raycast onto mip-(N+1): doubles `gdz`/`ngxmax`,
/// halves `gixy[1]` and every active `cf` entry's `z0`/`z1`, slides
/// `gylookup` to the mip-(N+1) sub-range, and rebases
/// `ixy_sptr_col_idx` into mip-(N+1)'s `column_offsets` sub-table.
///
/// Reaching this branch requires `gmipnum > 1`. The oracle uses
/// `gmipnum == 1` (no host calls [`roxlap_formats::vxl::Vxl::generate_mips`]
/// yet), so all 12 poses take the `(gmipcnt + 1) >= gmipnum`
/// early-out — the body below is byte-stable dead code at oracle
/// time.
///
/// # LP32 / LP64 sptr-stride bug audit (voxlap5.c:12017-12023)
///
/// The C body's `xor0 = (esi_rel << 29) ^ gixy[0]` and
/// `xor1 = (esi_rel << (gmipcnt + 17)) ^ gixy[1]` parity tests
/// were calibrated for `sizeof(char *) == 4`. After voxlaptest's
/// LP64 widen (`SPTR_LOG2_STRIDE = 3`, `voxlap5.c:99-104`),
/// `esi_rel = ixy_sptr_col - sptr` carries an extra factor of 2,
/// so:
///
/// | shift              | LP32 tests                    | LP64 tests                       |
/// |--------------------|-------------------------------|----------------------------------|
/// | `<<29`             | bit 0 of `column_index` (x parity) | bit −1 (always 0 → always "add") |
/// | `<<(gmipcnt + 17)` | depends on VSID + N (off-by-one even on LP32 for VSID > 1024) | even more wrong |
///
/// The C source flags this with an explicit `NOTE` and gates the
/// path on `vxlmipuse == 1` so it never burns. **Roxlap sidesteps
/// the whole issue**: `ixy_sptr_col_idx` is an *element* index
/// (no pointer-stride at all), and the body below derives parity
/// directly from `(column_index - mip_base) & (vsid_OLD - 1)` /
/// `>> log2(vsid_OLD)`. LP-independent and VSID-independent by
/// construction.
//
// Voxlap's outer check is `(uint8_t)(gmipcnt + 1) >=
// (uint8_t)gmipnum` — bytewise compare. For our port the
// natural i32/u32 widths give the same answer for any
// realistic mip count (0..32-ish); the byte-cast was an asm
// artifact.
#[allow(clippy::cast_sign_loss, clippy::cast_possible_wrap)]
fn phase_remiporend(state: &mut GrouscanState<'_>) -> Phase {
    if std::env::var("ROXLAP_TRACE_STARTSKY").is_ok() {
        eprintln!(
            "remiporend: gmipcnt={} gmipnum={} ce={} c={} gpz=[{}, {}] gxmax={} ngxmax={}",
            state.gmipcnt,
            state.gmipnum,
            state.ce_idx,
            state.c_idx,
            state.scratch.gpz[0],
            state.scratch.gpz[1],
            state.scratch.gxmax,
            state.ngxmax,
        );
    }
    if (state.gmipcnt + 1) as u32 >= state.gmipnum {
        return Phase::Startsky;
    }

    // CF.2 — cf-narrowing simulator. Gated behind `ROXLAP_CF_NARROW=1`
    // so default behaviour stays byte-stable to commit e484378 until
    // validated by CF.3. Walks each active cf entry through
    // `(2^old_mip - 1)` virtual finer-mip column steps' worth of
    // `phase_draw_fwall`-shape narrowing. Algorithmic fix for the
    // axis-aligned-mip-beams artifact at deep mip-N + near-axis rays.
    // See `crate::cf_narrow` + the `cf-narrowing-multi-session-plan`
    // memo for the full design.
    if state.gmipcnt > 0 && std::env::var_os("ROXLAP_CF_NARROW").is_some() {
        let inputs = crate::cf_narrow::CfNarrowInputs {
            gpz_at_entry: state.scratch.gpz,
            gdz_old: state.scratch.gdz,
            gi0: state.scratch.gi0,
            gi1: state.scratch.gi1,
            gylookup: state.gylookup,
            old_mip: state.gmipcnt as u32,
        };
        let lo = CF_SEED_INDEX;
        let hi = state.ce_idx + 1;
        crate::cf_narrow::cf_narrow_simulate(&mut state.scratch.cf[lo..hi], &inputs);
    }

    // Voxlap5.c:12007 — increment gmipcnt to NEW (= OLD + 1).
    let old_mip = state.gmipcnt as usize;
    state.gmipcnt += 1;
    let new_mip = state.gmipcnt as usize;

    // Column-index parity at mip-OLD. Audit at the doc comment
    // above: `ixy_sptr_col_idx` is currently in mip-OLD's
    // sub-table; subtracting `mip_base_offsets[old_mip]` gives
    // col-within-mip whose low bit is x parity and bit
    // `log2(vsid_OLD)` is y parity. This sidesteps voxlap C's
    // LP32-baked `<<29` / `<<(gmipcnt+17)` shift trick.
    let mip_old_base = state.mip_base_offsets[old_mip];
    let mip_new_base = state.mip_base_offsets[new_mip];
    let col_within_old = state.ixy_sptr_col_idx - mip_old_base;
    let vsid_old = (state.vsid >> old_mip) as usize;
    debug_assert!(vsid_old.is_power_of_two() && vsid_old > 0);
    let log2_vsid_old = vsid_old.trailing_zeros() as usize;
    let x_parity = col_within_old & 1;
    let y_parity = (col_within_old >> log2_vsid_old) & 1;

    // Voxlap5.c:12012-12037 — lane 0 (x) gpz/gdz adjust.
    // C: `xor0 = (esi_rel<<29) ^ gixy[0]`; if bit-31 zero, add gdz.
    // Bit 31 of xor0 == (col-x-parity) XOR (sign-bit of gixy[0]).
    // Trailing column ⇒ next column-step lands inside the same
    // mip-NEW super-cell, so advance gpz to the next coarser
    // grid line.
    {
        let dz = state.scratch.gdz[0];
        let trailing = (x_parity == 0) == (state.scratch.gixy[0] >= 0);
        if trailing {
            state.scratch.gpz[0] = state.scratch.gpz[0].wrapping_add(dz);
        }
        let doubled = dz.wrapping_add(dz);
        if (dz ^ doubled) < 0 {
            // Signed overflow → saturate gpz to i32::MAX, gdz to 0
            // (voxlap5.c:12030-12036).
            state.scratch.gpz[0] = i32::MAX;
            state.scratch.gdz[0] = 0;
        } else {
            state.scratch.gdz[0] = doubled;
        }
    }

    // Voxlap5.c:12043 — save z0 to the c_presync slot before the
    // halve loop runs. Voxlap relies on `c_presync` lying inside
    // `[cf[128], ce]` so the loop halves it as a side effect; the
    // explicit reload below halves it AGAIN (yes, twice — voxlap's
    // literal asm fix-up). Defensive guard handles the unset
    // (`usize::MAX`) case even though voxlap C dereferences
    // unconditionally.
    if state.c_presync_idx < state.scratch.cf.len() {
        state.scratch.cf[state.c_presync_idx].z0 = state.z0;
    }

    // Voxlap5.c:12047-12060 — lane 1 (y) gpz/gdz adjust.
    {
        let dz = state.scratch.gdz[1];
        let trailing = (y_parity == 0) == (state.scratch.gixy[1] >= 0);
        if trailing {
            state.scratch.gpz[1] = state.scratch.gpz[1].wrapping_add(dz);
        }
        let doubled = dz.wrapping_add(dz);
        if (dz ^ doubled) < 0 {
            state.scratch.gpz[1] = i32::MAX;
            state.scratch.gdz[1] = 0;
        } else {
            state.scratch.gdz[1] = doubled;
        }
    }

    // Voxlap5.c:12062-12073 — re-mask `ixy_sptr_col_idx` into
    // mip-NEW's sub-table. The audit's natural form: halve the
    // OLD-mip x/y coords, then index the NEW sub-table.
    {
        let x_old = col_within_old & (vsid_old - 1);
        let y_old = col_within_old >> log2_vsid_old;
        let x_new = x_old >> 1;
        let y_new = y_old >> 1;
        let vsid_new = vsid_old >> 1;
        state.ixy_sptr_col_idx = mip_new_base + y_new * vsid_new + x_new;
    }

    // Voxlap5.c:12076 — slide `gylookup` to mip-NEW's sub-range.
    // Each mip-N table is `((chunks_z * 512) >> N) + 4` int32 entries
    // (S4B.6.d widens from 512 to chunks_z*512 for stacked grids);
    // advance by mip-OLD's length to skip past it.
    {
        let chunks_z = state.grid_view.chunk_grid.map_or(1u32, |cg| cg.chunks_z);
        let advance = (((chunks_z * 512) >> old_mip) as usize) + 4;
        let advance = advance.min(state.gylookup.len());
        state.gylookup = &state.gylookup[advance..];
    }

    // Voxlap5.c:12079 — halve gixy[1] (signed arithmetic shift).
    state.scratch.gixy[1] >>= 1;

    // Roxlap multi-chunk + multi-mip: halve `cx_mip`/`cy_mip` so they
    // track mip-NEW voxel coords. The column-step's mip-N branch
    // advances them by `±1` per step (one mip-N voxel) and derives
    // both chunk-XY index and the in-chunk mip-N column offset from
    // them. Voxlap C doesn't carry this state — it tracks `esi` as a
    // raw pointer through `gxmipk`/`gymipk` masks (LP32-baked); our
    // port re-derives the same indices from explicit voxel coords.
    //
    // Signed arithmetic shift: `>> 1` rounds toward negative infinity,
    // mirroring two-voxel-merge geometry for both positive and
    // negative coords. (OOB-XY camera can reach mip-N with negative
    // `cx`/`cy`; the multi-mip path handles that.)
    state.cx_mip >>= 1;
    state.cy_mip >>= 1;

    // Voxlap5.c:12084-12087 — halve every active cf entry's z bounds.
    // z0 uses unsigned shift (rounds down, preserves voxlap's `shr`
    // semantics on negative z arising from underflow on air-gap
    // entries); z1 uses `(z1 + 1) >> 1` (round up).
    for idx in CF_SEED_INDEX..=state.ce_idx {
        let entry = &mut state.scratch.cf[idx];
        entry.z0 = (entry.z0 as u32 >> 1) as i32;
        entry.z1 = ((entry.z1 + 1) as u32 >> 1) as i32;
    }

    // Voxlap5.c:12089-12095 — saturating-double ngxmax (capped at
    // gxmax). Pre-doubled comparison uses unsigned semantics to
    // catch the wrapping-overflow case.
    let gxmax = state.scratch.gxmax;
    if (state.ngxmax as u32) >= (gxmax as u32) {
        return Phase::Startsky;
    }
    let dn = state.ngxmax.wrapping_add(state.ngxmax);
    state.ngxmax = if dn < 0 || dn >= gxmax { gxmax } else { dn };

    // Voxlap5.c:12101-12102 — z0/z1 reload. cf-halve already halved
    // c_presync's z0; the shift here halves it AGAIN. Voxlap's
    // literal asm fix-up.
    if state.c_presync_idx < state.scratch.cf.len() {
        state.z0 = state.scratch.cf[state.c_presync_idx].z0 >> 1;
    }
    state.z1 = ((state.z1 + 1) as u32 >> 1) as i32;

    // v5.asm `remiporend:` tail — recompute leading lane and advance
    // gpz[lane]. Critically, the asm does NOT update `mm6` here: the
    // packed `(gx, ogx)` pair was already set by the column-step's
    // `punpckldq mm6, mm7` just before the `ja remiporend` jump, and
    // remiporend preserves it through the mip transition. Our port
    // mirrors that — `state.gx` was written by
    // [`phase_after_delete_kept_presync`] right before this phase was
    // entered and must not be overwritten here, or the downstream
    // `predraw{ceil,flor}` / `predeletez` ogx↔gx swaps see the
    // post-mip lane's potentially trailing-incremented gpz instead of
    // the pre-mip new_gpz that triggered the transition. Visible as
    // faint world-axis-aligned green beams under deep mip-N — for
    // axis-aligned rays (gdz[dead lane] = 0, gpz[dead lane] = MAX) the
    // live lane's `gpz[live]` is incremented by `gdz_old` inside the
    // trailing-column branch above, which is what state.gx would have
    // been miswritten to.
    state.lane = usize::from(state.scratch.gpz[1] < state.scratch.gpz[0]);
    state.scratch.gpz[state.lane] =
        state.scratch.gpz[state.lane].wrapping_add(state.scratch.gdz[state.lane]);

    // Voxlap5.c:12112-12113 — reload `state.column` from the new
    // column. Malformed offsets fall back to an empty slice
    // (matches `camera_column_slice`'s defensive posture).
    //
    // Multi-chunk: only reload from the active chunk when one
    // exists. For OOB-XY chunks the column-step path has already
    // set `state.column = &[]`; without this guard, remiporend
    // would resurrect the column by indexing into the SEED chunk's
    // mip sub-table (whose `ixy_sptr_col_idx` is meaningless after
    // the unbounded `wrapping_add_signed` march through OOB chunks).
    // The march can land in a DEEPER mip's sub-table — read as
    // garbage RGB-0 voxels at gmipcnt-1 z values. Manifested as the
    // 388k-pixel BLACK WALL around the ship at OOB-XY camera
    // (user-reported 2026-05-26).
    if !state.current_chunk_exists {
        state.column.clear();
    } else if let Some(&col_off) = state.column_offsets.get(state.ixy_sptr_col_idx) {
        let col_off = col_off as usize;
        install_owned_column(&mut state.column, state.slab_buf, col_off);
    }

    // Voxlap5.c:12116 — reset c to top-of-stack.
    state.c_idx = state.ce_idx;

    // Voxlap5.c:12118 — `goto skipixy2_sync_from_presync`.
    Phase::SyncFromPresync
}

/// `startsky` — voxlap5.c:12120-12190. Drains every remaining
/// cfasm entry's pixel range with sky values.
///
/// Voxlap's body forks on `skyoff`:
/// - `sky_off == 0` or no [`SkyRef`] loaded: solid fill — write
///   `skycast` into each radar slot in the entry's `[i0, i1]`
///   range. This is the cheap default path used by every oracle
///   pose.
/// - `sky_off != 0` and a [`SkyRef`] is bound: per-pixel latitude
///   search into the `skylat` table, write `sky_tex[edi]` into
///   each radar slot's `col` and `skydist` into its `dist`.
//
// `p as usize` cast is intentional: `p` walks an `isize` range
// `[i0, i1]` where both ends were checked non-negative inside
// the loop above (i0 > i1 short-circuits, and CfType.i0/i1 are
// always set to non-negative values by the rest of grouscan).
#[allow(clippy::cast_sign_loss)]
fn phase_startsky(state: &mut GrouscanState<'_>) -> Phase {
    // Voxlap5.c:12125-12126. c starts at the seed slot; if the
    // stack already drained below it, retsub.
    if CF_SEED_INDEX > state.ce_idx {
        return Phase::Done;
    }

    // Branch on whether a sky texture is bound AND gline picked a
    // non-zero per-ray sky_off. Either condition false ⇒ solid
    // fill. The oracle never loads a sky; its hashes are byte-
    // stable through this dispatch.
    let textured = state.sky.is_some() && state.scratch.sky_off != 0;
    if textured {
        phase_startsky_textured(state)
    } else {
        phase_startsky_solid(state)
    }
}

/// Solid-fill branch (voxlap5.c:12128-12141). Writes
/// `state.scratch.skycast` into every remaining radar slot.
#[allow(clippy::cast_sign_loss)]
fn phase_startsky_solid(state: &mut GrouscanState<'_>) -> Phase {
    let trace = std::env::var("ROXLAP_TRACE_STARTSKY").is_ok();

    let skycast = state.scratch.skycast;
    for c_idx in CF_SEED_INDEX..=state.ce_idx {
        let i0 = state.scratch.cf[c_idx].i0;
        let i1 = state.scratch.cf[c_idx].i1;
        if i0 > i1 {
            if trace {
                eprintln!(
                    "startsky cf[{c_idx}] i0={i0} i1={i1} (empty, skip; ce={})",
                    state.ce_idx
                );
            }
            continue;
        }
        if trace {
            eprintln!(
                "startsky cf[{c_idx}] drains slots [{i0}..={i1}] ({} slots; ce={})",
                i1 - i0 + 1,
                state.ce_idx
            );
        }
        for p in i0..=i1 {
            if let Some(slot) = state.scratch.radar.get_mut(p as usize) {
                *slot = skycast;
            }
        }
    }
    Phase::Done
}

/// Textured-sky fill (voxlap5.c:12143-12188).
///
/// For each cf entry, walk pixels right-to-left. Per pixel:
/// 1. Step ray endpoint backward by `(gi0, gi1)` (the per-pixel
///    coefficient gline stamped on `scratch`).
/// 2. Latitude search: from a starting `edi = sky.xsiz_post`
///    (preserved across cf entries within one ray), walk `edi--`
///    while `(cx1 >> 16) * neg_yvi + (cy1 >> 16) * xvi < 0`. Stop
///    when the cross product flips sign — that's the texel
///    column for this pixel ray.
/// 3. Sample `sky_pixels[sky_off / 4 + edi]` into the radar
///    slot's `col`; stamp `skydist` into its `dist`.
///
/// `sky_off` is a byte offset in voxlap C (`skyoff = curlng *
/// skybpl + nskypic`); we keep it as a byte offset too and divide
/// by 4 to land at the i32 pixel index.
#[allow(clippy::cast_sign_loss, clippy::cast_possible_wrap)]
fn phase_startsky_textured(state: &mut GrouscanState<'_>) -> Phase {
    let sky = state.sky.expect("phase_startsky_textured requires SkyRef");
    let sky_off = state.scratch.sky_off;
    let skydist = state.scratch.skycast.dist;
    let gi0 = state.scratch.gi0;
    let gi1 = state.scratch.gi1;

    // sky_off is a byte offset relative to the texture's pixel
    // base; voxlap C uses `int32_t *sky_tex = (int32_t *)skyoff`
    // which means sky_off must be a multiple of 4. Convert to
    // i32-pixel index.
    let row_pixel_base = (sky_off as usize) / 4;

    // edi cursor — voxlap calls this `sky_edi`, preserved across
    // cf entries within one ray (mirrors the asm's `static`-like
    // edi register).
    let mut sky_edi: i32 = sky.xsiz_post;

    for c_idx in CF_SEED_INDEX..=state.ce_idx {
        let i0 = state.scratch.cf[c_idx].i0;
        let i1 = state.scratch.cf[c_idx].i1;
        if i0 > i1 {
            continue;
        }
        // The cf entry's stored `cx1`/`cy1` are gline's original
        // far-end values; drain operations during grouscan shrink
        // `i1` without updating them. Re-derive the position at
        // the *current* `i1` from `cx0 + (i1 - i0) * gi0`. Voxlap
        // C has this same stale-cx1 quirk (voxlap5.c:11678 writes
        // `c->i1 = ebx` without touching `c->cx1`); the textured-
        // sky distortion is hidden in voxlap when looking
        // upward (most rays hit sky directly with no drain) but
        // becomes visible at low pitch where wall-fills shrink
        // `i1` substantially.
        // Realistic radar widths fit i32 by orders of magnitude
        // (xres ≤ a few thousand); the isize→i32 narrowing is safe.
        #[allow(clippy::cast_possible_truncation)]
        let leng_remaining = (i1 - i0) as i32;
        let cx0 = state.scratch.cf[c_idx].cx0;
        let cy0 = state.scratch.cf[c_idx].cy0;
        let mut sx = cx0.wrapping_add(leng_remaining.wrapping_mul(gi0));
        let mut sy = cy0.wrapping_add(leng_remaining.wrapping_mul(gi1));
        let mut p = i1;
        loop {
            // preskysearch: step ray backward.
            sx = sx.wrapping_sub(gi0);
            sy = sy.wrapping_sub(gi1);

            // skysearch: find matching sky column.
            loop {
                if sky_edi < 0 || (sky_edi as usize) >= sky.lat.len() {
                    // Out-of-range edi shouldn't happen with a
                    // well-formed lat[], but guard so a malformed
                    // sky doesn't OOB-panic the whole render.
                    sky_edi = 0;
                    break;
                }
                let sl = sky.lat[sky_edi as usize];
                let neg_yvi = i32::from((sl & 0xffff) as i16);
                let xvi_lane = i32::from(((sl >> 16) & 0xffff) as i16);
                let test = (sx >> 16).wrapping_mul(neg_yvi) + (sy >> 16).wrapping_mul(xvi_lane);
                if test >= 0 {
                    break;
                }
                sky_edi -= 1;
            }

            let pixel_idx = row_pixel_base + sky_edi as usize;
            // S1.Z: out-of-range texel falls back to skycast.col
            // (the solid sky color), not 0 (which renders BLACK).
            // For inside camera the OOB lookup almost never fires;
            // for outside camera with steep angles + OOB walks the
            // sky_edi search can land past the panorama's pixel
            // buffer, and the previous `else { 0 }` fallback drew
            // a hard-edged black pentagon under the world (visible
            // when flying just outside the world's XY footprint and
            // looking back through the floor).
            let col = if pixel_idx < sky.pixels.len() {
                sky.pixels[pixel_idx]
            } else {
                state.scratch.skycast.col
            };
            if let Some(slot) = state.scratch.radar.get_mut(p as usize) {
                slot.col = col;
                slot.dist = skydist;
            }
            if p <= i0 {
                break;
            }
            p -= 1;
        }
    }
    Phase::Done
}

#[cfg(test)]
mod tests {
    use super::*;

    /// VC.2: builder output equals the chain-bounded bulk slice for
    /// representative slab-chain shapes — single-slab bedrock, a
    /// multi-slab column with non-last + last slabs, a chain whose
    /// last-slab body extends to slab_buf's end, and a truncated tail.
    #[test]
    fn vc2_builder_matches_bulk_slice_byte_for_byte() {
        // Helper: full chain-bounded copy via VC.1's bulk path so we
        // can diff per-shape against the per-slab builder.
        fn bulk_install(target: &mut Vec<u8>, slab_buf: &[u8], off: usize) {
            target.clear();
            if off >= slab_buf.len() {
                return;
            }
            let tail = &slab_buf[off..];
            let len = slab_chain_byte_len(tail);
            target.extend_from_slice(&tail[..len]);
        }
        fn assert_match(slab_buf: &[u8], off: usize, label: &str) {
            let mut bulk = Vec::new();
            let mut built = Vec::new();
            bulk_install(&mut bulk, slab_buf, off);
            install_owned_column(&mut built, slab_buf, off);
            assert_eq!(bulk, built, "{label}: bulk != builder output");
        }

        // Shape 1 — single-slab bedrock-only column.
        // [nextptr=0, z1=255, z1c=255, z0=0] + 1 placeholder colour.
        let single = vec![0u8, 255, 255, 0, 0xAA, 0xBB, 0xCC, 0xDD];
        assert_match(&single, 0, "single-slab bedrock");

        // Shape 2 — two-slab column. First slab: nextptr=4 (advance
        // 16 bytes), z1/z1c/z0 don't matter for the walker advance,
        // 12 bytes of voxel records. Last slab at offset 16:
        // [nextptr=0, z1=10, z1c=12, z0=0] + 3 colour records.
        let mut multi = vec![4u8, 0, 0, 0]; // first slab header
                                            // 12 voxel-record bytes (= 3 colour records); contents arbitrary.
        multi.extend_from_slice(&[
            0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE, 0x11, 0x22, 0x33, 0x44,
        ]);
        multi.extend_from_slice(&[0, 10, 12, 0]); // last slab header
        multi.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]); // 3 colour records
        assert_match(&multi, 0, "two-slab column");

        // Shape 3 — last-slab body extends past slab_buf end. Builder
        // should clamp to slab_buf.len(). Header says n_floor=3
        // (12 colour bytes) but only 4 colour bytes are present.
        let truncated_body = vec![0u8, 10, 12, 0, 1, 2, 3, 4];
        assert_match(&truncated_body, 0, "last-slab body truncated");

        // Shape 4 — empty slab_buf via off past end. Returns empty
        // target, no panic.
        let empty: Vec<u8> = vec![];
        assert_match(&empty, 0, "empty slab_buf");
        assert_match(&single, single.len() + 5, "off past end");

        // Shape 5 — malformed nextptr (= 1 → advance 4 bytes back into
        // own header, which would loop forever). Builder bails on
        // `advance < 4`; emit the malformed slab's header so the
        // walker's defensive `column.get(...).unwrap_or(0)` reads
        // produce the same bytes as the bulk path.
        let malformed = vec![1u8, 99, 100, 101];
        let mut bulk_malformed = Vec::new();
        let mut built_malformed = Vec::new();
        bulk_install(&mut bulk_malformed, &malformed, 0);
        install_owned_column(&mut built_malformed, &malformed, 0);
        // bulk = chain_len walker which bails at advance<4 returning
        // column.len() = 4 → bulk has all 4 bytes.
        assert_eq!(bulk_malformed, malformed);
        // builder: emit the 4 header bytes then return.
        assert_eq!(built_malformed, malformed);
    }

    /// VC.3: `build_owned_column_from_chain_translated` with
    /// `world_z_base == 0` produces byte-identical output to VC.2's
    /// `build_owned_column_from_chain`. Validates the no-op case
    /// the seed-install dispatch actually hits today.
    #[test]
    fn vc3_translation_with_zero_base_is_identity() {
        // Multi-slab + last-slab columns. Same fixtures as the VC.2
        // test so any divergence between the two builders pops here
        // immediately.
        let single = vec![0u8, 100, 105, 0, 0xAA, 0xBB, 0xCC, 0xDD];
        let mut multi = vec![4u8, 5, 10, 0]; // first slab: 16 bytes total
        multi.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]); // 3 colours
        multi.extend_from_slice(&[0, 50, 60, 40]); // last slab header
        multi.extend_from_slice(&[
            0x11, 0x12, 0x13, 0x14, 0x21, 0x22, 0x23, 0x24, 0x31, 0x32, 0x33, 0x34, 0x41, 0x42,
            0x43, 0x44, 0x51, 0x52, 0x53, 0x54, 0x61, 0x62, 0x63, 0x64, 0x71, 0x72, 0x73, 0x74,
            0x81, 0x82, 0x83, 0x84, 0x91, 0x92, 0x93, 0x94, 0xA1, 0xA2, 0xA3, 0xA4, 0xB1, 0xB2,
            0xB3, 0xB4,
        ]); // 11 colour records (n_floor = 60 - 50 + 1 = 11)

        for (label, slab_buf) in [("single", &single[..]), ("multi", &multi[..])] {
            let mut untranslated = Vec::new();
            let mut translated = Vec::new();
            build_owned_column_from_chain(&mut untranslated, slab_buf, 0);
            build_owned_column_from_chain_translated(&mut translated, slab_buf, 0, 0);
            assert_eq!(
                untranslated, translated,
                "{label}: world_z_base=0 must produce byte-identical output"
            );
        }
    }

    /// VC.3: positive `world_z_base` shifts every slab header's z1
    /// / z1c / z0 by that amount; other bytes (nextptr, voxel
    /// records) pass through. Body lengths (= chain end position)
    /// stay unchanged.
    #[test]
    fn vc3_translation_with_positive_base_shifts_z_bytes() {
        // Two-slab column. First slab: nextptr=4 (advance 16 bytes),
        // z1=5, z1c=10, z0=0. Last slab: nextptr=0, z1=20, z1c=22,
        // z0=18 → n_floor = 3 → 12 colour bytes.
        let mut col = vec![4u8, 5, 10, 0];
        col.extend_from_slice(&[
            0xA1, 0xA2, 0xA3, 0xA4, 0xB1, 0xB2, 0xB3, 0xB4, 0xC1, 0xC2, 0xC3, 0xC4,
        ]);
        col.extend_from_slice(&[0, 20, 22, 18]);
        col.extend_from_slice(&[
            0xD1, 0xD2, 0xD3, 0xD4, 0xE1, 0xE2, 0xE3, 0xE4, 0xF1, 0xF2, 0xF3, 0xF4,
        ]);

        let mut out = Vec::new();
        build_owned_column_from_chain_translated(&mut out, &col, 0, 100);

        // First slab header: nextptr unchanged, z1/z1c/z0 shifted by 100.
        assert_eq!(out[0], 4);
        assert_eq!(out[1], 105);
        assert_eq!(out[2], 110);
        assert_eq!(out[3], 100);
        // First slab voxel records pass through.
        assert_eq!(&out[4..16], &col[4..16]);
        // Last slab header: shifted z values.
        assert_eq!(out[16], 0);
        assert_eq!(out[17], 120);
        assert_eq!(out[18], 122);
        assert_eq!(out[19], 118);
        // Last slab voxel records pass through.
        assert_eq!(&out[20..32], &col[20..32]);
        // Total length matches input (= chain end).
        assert_eq!(out.len(), col.len());
    }

    /// VC.3: u8 overflow at the translation step panics. Guards
    /// against silently producing wrap-around z values when callers
    /// fail to honour the VC.3 dispatch constraint
    /// (`chunks_z * chunk_size_z <= 256`).
    #[test]
    #[should_panic(expected = "VC.3 z translation overflows u8")]
    fn vc3_translation_overflow_panics() {
        // z1 = 200 + world_z_base 100 = 300 → overflow.
        let col = vec![0u8, 200, 200, 0];
        let mut out = Vec::new();
        build_owned_column_from_chain_translated(&mut out, &col, 0, 100);
    }

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
            chz_layer: 0,
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
    /// Single-mip placeholder: `[0, column_offsets.len()]` is the
    /// shape post-`Vxl::parse` callers send. With an empty
    /// `DUMMY_COLUMN_OFFSETS`, this collapses to `[0, 0]`.
    const DUMMY_MIP_OFFSETS: [usize; 2] = [0, 0];

    fn dummy_inputs<'a>() -> GrouscanInputs<'a> {
        GrouscanInputs {
            column: &DUMMY_COLUMN,
            gylookup: &DUMMY_GYLOOKUP,
            gcsub: &DUMMY_GCSUB,
            slab_buf: &DUMMY_SLAB_BUF,
            column_offsets: &DUMMY_COLUMN_OFFSETS,
            mip_base_offsets: &DUMMY_MIP_OFFSETS,
            vsid: 64,
            sky: None,
            grid_view: crate::grid_view::GridView::from_parts(
                64,
                &DUMMY_SLAB_BUF,
                &DUMMY_COLUMN_OFFSETS,
                &DUMMY_MIP_OFFSETS,
            ),
            camera_chunk_z: 0,
            chunk_world_z_base: 0,
            chunk_size_z: 256,
        }
    }

    #[test]
    fn prologue_caches_cf_seed_state() {
        let mut s = fresh_scratch();
        s.gpz = [1_000, 2_000];
        s.gdz = [10, 20];
        s.gxmax = 999_999;
        let p = grouscan_run(&mut s, &dummy_inputs(), 0, 0, 0, 0, 50_000, 1);
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
        let p = grouscan_run(&mut s, &dummy_inputs(), 0, 0, 0, 0, 1, 1);
        assert_eq!(p.lane, 0);

        // Reverse: gpz[1] = 500 < gpz[0] = 800 → lane 1 wins.
        let mut s = fresh_scratch();
        s.gpz = [800, 500];
        s.gdz = [10, 20];
        let p = grouscan_run(&mut s, &dummy_inputs(), 0, 0, 0, 0, 1, 1);
        assert_eq!(p.lane, 1);
    }

    #[test]
    fn prologue_advances_winning_lane_by_gdz() {
        // gpz[0] = 1000 wins; gdz[0] = 256. After: gpz[0] = 1256.
        let mut s = fresh_scratch();
        s.gpz = [1_000, 2_000];
        s.gdz = [256, 999];
        let _ = grouscan_run(&mut s, &dummy_inputs(), 0, 0, 0, 0, 1, 1);
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
        let p = grouscan_run(&mut s, &dummy_inputs(), 0, 0, 0, 0, 50_000, 2);
        assert_eq!(p.ngxmax, 50_000);

        // Single-mip case: ngxmax = gxmax regardless of gxmip.
        let mut s = fresh_scratch();
        s.gpz = [1_000, 2_000];
        s.gdz = [10, 20];
        s.gxmax = 1_000_000;
        let p = grouscan_run(&mut s, &dummy_inputs(), 0, 0, 0, 0, 50_000, 1);
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
            mip_base_offsets: &DUMMY_MIP_OFFSETS,
            vsid: 64,
            sky: None,
            grid_view: crate::grid_view::GridView::from_parts(
                64,
                &DUMMY_SLAB_BUF,
                &DUMMY_COLUMN_OFFSETS,
                &DUMMY_MIP_OFFSETS,
            ),
            camera_chunk_z: 0,
            chunk_world_z_base: 0,
            chunk_size_z: 256,
        };
        GrouscanState::from_seed(scratch, &inputs, 0, 0, 0, 0, 1)
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
            mip_base_offsets: &DUMMY_MIP_OFFSETS,
            vsid: 64,
            sky: None,
            grid_view: crate::grid_view::GridView::from_parts(
                64,
                &DUMMY_SLAB_BUF,
                &DUMMY_COLUMN_OFFSETS,
                &DUMMY_MIP_OFFSETS,
            ),
            camera_chunk_z: 0,
            chunk_world_z_base: 0,
            chunk_size_z: 256,
        };
        GrouscanState::from_seed(scratch, &inputs, vptr_offset, 0, 0, 0, 1)
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
        // return PreDrawCeil. Slab header lives at column[32..36].
        let mut s = fresh_scratch();
        s.cf[CF_SEED_INDEX].z0 = 20;
        let mut column = vec![0u8; 32];
        // VC.1: chain walker bounds the seed-time owned-column copy
        // at the slab chain's natural end. Give byte 0 a valid
        // `nextptr = 8` (= advance 8 * 4 = 32 bytes to the next slab)
        // so the chain walker spans the artificial padding bytes
        // drawcwall's negative-offset reads land in. The header's
        // z1/z1c/z0 fields stay 0 — they don't influence this test.
        column[0] = 8;
        column.extend_from_slice(&[0, 10, 12, 5]);
        let gylookup = [0i32; 64];
        let gcsub = [0i64; 9];
        let mut state = state_for_drawcwall(&mut s, &column, &gylookup, &gcsub, 32);
        assert_eq!(phase_draw_cwall(&mut state), Phase::PreDrawCeil);
        assert_eq!(state.z0, 5);
    }

    #[test]
    fn drawcwall_inner_loop_reads_previous_slab_tail() {
        // Multi-slab column: prev slab at [0..16] with the last 4
        // bytes (column[12..16]) as the back-wall colour bytes
        // drawcwall reads via negative `off`. Current slab header
        // at [16..20] with v[3] = 0 (back wall extends from z=0
        // upward). z0 cached = 0 < dv3 = ... wait, dv3 = 0 ≤ z0
        // would early-exit. Need dv3 > z0; with z0 = 0 we need
        // dv3 > 0. Set v[3] = 2 → dv3 = 2 > z0 = 0 → enter loop.
        // off = z0 - v[3] = 0 - 2 = -2 → row_offset = vptr_offset
        // (16) + (-2)*4 = 8 → reads column[8..12] from prev slab.
        //
        // Inner loop: cx0 = 1<<16, gylookup[1] = 1 → cross_sign =
        // 1 > 0 → check v[3] != z0 (2 != 1 after z0++ → 1) → true
        // → continue 'outer (next iteration: off = 1 - 2 = -1,
        // row_offset = 16 + (-1)*4 = 12 → reads column[12..16]).
        // Then z0 = 2 = v[3] → save c->i0, return PreDrawCeil.
        let mut s = fresh_scratch();
        s.cf[CF_SEED_INDEX].z0 = 0;
        s.cf[CF_SEED_INDEX].i0 = 0;
        s.cf[CF_SEED_INDEX].i1 = 100;
        s.cf[CF_SEED_INDEX].cx0 = 1 << 16;
        s.cf[CF_SEED_INDEX].cy0 = 0;

        // 16-byte previous slab + 4-byte current header.
        let mut column = vec![0u8; 16];
        // VC.1: same as above — set nextptr=4 (advance 4*4=16 bytes)
        // so the chain walker's bounded copy spans the artificial
        // 16 bytes of "previous slab tail" the inner loop reads.
        column[0] = 4;
        column.extend_from_slice(&[0, 10, 12, 2]); // current slab v[3] = 2

        let mut gylookup = [0i32; 64];
        gylookup[1] = 1;
        gylookup[2] = 1;
        let gcsub = [0i64; 9];

        let mut state = state_for_drawcwall(&mut s, &column, &gylookup, &gcsub, 16);
        state.ogx = 0;
        assert_eq!(phase_draw_cwall(&mut state), Phase::PreDrawCeil);
        // z0 ended at v[3] = 2.
        assert_eq!(state.z0, 2);
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
            mip_base_offsets: &DUMMY_MIP_OFFSETS,
            vsid: 64,
            sky: None,
            grid_view: crate::grid_view::GridView::from_parts(
                64,
                &DUMMY_SLAB_BUF,
                &DUMMY_COLUMN_OFFSETS,
                &DUMMY_MIP_OFFSETS,
            ),
            camera_chunk_z: 0,
            chunk_world_z_base: 0,
            chunk_size_z: 256,
        };
        GrouscanState::from_seed(scratch, &inputs, vptr_offset, 0, 0, 0, 1)
    }

    #[test]
    fn predrawceil_swaps_ogx_and_gx() {
        let mut s = fresh_scratch();
        let inputs = dummy_inputs();
        let mut state = GrouscanState::from_seed(&mut s, &inputs, 0, 0, 0, 0, 1);
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
        // pixel at radar[i0=20], i0 → 21. i0 > i1 = 20 → DeleteZ
        // (voxlap5.c:11766 → `goto deletez` direct, NO ogx ↔ gx
        // swap; only drawfwall / drawcwall route through predeletez).
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
        assert_eq!(phase_draw_ceil(&mut state), Phase::DeleteZ);
        assert_ne!(state.scratch.radar[20].col, 0);
        assert_eq!(state.scratch.cf[CF_SEED_INDEX].i0, 21);
    }

    #[test]
    fn drawceil_bails_when_z0_out_of_gylookup() {
        // z0 = 64, gylookup len = 64 → out-of-range, bail to
        // AfterDelete (S1.Z: column-step instead of cf-stack pop).
        let mut s = fresh_scratch();
        s.cf[CF_SEED_INDEX].z0 = 64;
        let column = vec![0u8; 8];
        let gylookup = [0i32; 64];
        let gcsub = [0i64; 9];
        let mut state = state_for_drawceil(&mut s, &column, &gylookup, &gcsub, 4);
        assert_eq!(phase_draw_ceil(&mut state), Phase::AfterDelete);
    }

    #[test]
    fn predrawflor_swaps_ogx_and_gx() {
        let mut s = fresh_scratch();
        let inputs = dummy_inputs();
        let mut state = GrouscanState::from_seed(&mut s, &inputs, 0, 0, 0, 0, 1);
        state.ogx = 0x3333;
        state.gx = 0x4444;
        assert_eq!(phase_pre_draw_flor(&mut state), Phase::DrawFlor);
        assert_eq!(state.ogx, 0x4444);
        assert_eq!(state.gx, 0x3333);
    }

    #[test]
    fn drawflor_cross_sign_non_positive_returns_after_delete() {
        // cx1 = cy1 = 0, gylookup[z1] = 0 → cross_sign = 0 ≤ 0 on
        // entry → enddrawflor → AfterDelete. No pixel written.
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
        assert_eq!(phase_draw_flor(&mut state), Phase::AfterDelete);
        assert_eq!(state.scratch.cf[CF_SEED_INDEX].i1, 20);
    }

    #[test]
    fn drawflor_writes_pixel_then_exhausts_radar() {
        // cx1 = 1<<16, cy1 = 0, gylookup[z1] = 1 → cross_sign = 1 > 0
        // → write pixel at radar[i1=20], i1 → 19. i0 = 20 → 19 < 20
        // trips DeleteZ direct (voxlap5.c:11790; NO ogx ↔ gx swap).
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
        assert_eq!(phase_draw_flor(&mut state), Phase::DeleteZ);
        assert_ne!(state.scratch.radar[20].col, 0);
        assert_eq!(state.scratch.cf[CF_SEED_INDEX].i1, 19);
    }

    #[test]
    fn drawflor_bails_when_z1_out_of_gylookup() {
        // S1.Z: bail to AfterDelete (column-step) instead of
        // PreDeleteZ (cf-stack pop).
        let mut s = fresh_scratch();
        s.cf[CF_SEED_INDEX].z1 = 64;
        let column = vec![0u8; 8];
        let gylookup = [0i32; 64];
        let gcsub = [0i64; 9];
        let mut state = state_for_drawceil(&mut s, &column, &gylookup, &gcsub, 0);
        assert_eq!(phase_draw_flor(&mut state), Phase::AfterDelete);
    }

    #[test]
    fn predeletez_swaps_ogx_and_gx() {
        let mut s = fresh_scratch();
        let inputs = dummy_inputs();
        let mut state = GrouscanState::from_seed(&mut s, &inputs, 0, 0, 0, 0, 1);
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
        let mut state = GrouscanState::from_seed(&mut s, &inputs, 0, 0, 0, 0, 1);
        assert_eq!(state.ce_idx, CF_SEED_INDEX);
        assert_eq!(phase_delete_z(&mut state), Phase::Done);
    }

    #[test]
    fn deletez_pops_top_when_c_equals_ce() {
        // ce above seed and c == ce → just decrement ce, no shift,
        // route to AfterDelete.
        let mut s = fresh_scratch();
        let inputs = dummy_inputs();
        let mut state = GrouscanState::from_seed(&mut s, &inputs, 0, 0, 0, 0, 1);
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
        let mut state = GrouscanState::from_seed(&mut s, &inputs, 0, 0, 0, 0, 1);
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
        let state = GrouscanState::from_seed(&mut s, &inputs, 0, 0, 0, 0, 1);
        assert_eq!(state.c_idx, CF_SEED_INDEX);
        assert_eq!(state.ce_idx, CF_SEED_INDEX);
        assert_eq!(state.c_presync_idx, usize::MAX);
    }

    #[test]
    fn afterdelete_sets_presync_and_routes_to_kept() {
        let mut s = fresh_scratch();
        let inputs = dummy_inputs();
        let mut state = GrouscanState::from_seed(&mut s, &inputs, 0, 0, 0, 0, 1);
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
        let mut state = GrouscanState::from_seed(&mut s, &inputs, 0, 0, 0, 0, 1);
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
        let mut state = GrouscanState::from_seed(&mut s, &inputs, 0, 0, 0, 0, 1);
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
        let mip_base = [0usize, column_offsets.len()];
        let inputs = GrouscanInputs {
            column: &slab_buf[20..], // initial column = #5
            gylookup: &gylookup,
            gcsub: &gcsub,
            slab_buf: &slab_buf,
            column_offsets: &column_offsets,
            mip_base_offsets: &mip_base,
            vsid: 4,
            sky: None,
            grid_view: crate::grid_view::GridView::from_parts(
                4,
                &slab_buf,
                &column_offsets,
                &mip_base,
            ),
            camera_chunk_z: 0,
            chunk_world_z_base: 0,
            chunk_size_z: 256,
        };
        let mut s = fresh_scratch();
        s.gixy = [1, 4]; // x-step = 1, y-step = 4
                         // S1.Z: column-step recomputes ixy_sptr_col_idx from the signed
                         // (cx, cy) cursor (cy*vsid + cx), so the seed must match — for
                         // idx=5 in a 4×4 world, that's (cx=1, cy=1). After lane=0 step:
                         //   cx 1→2, cy=1 → recomputed idx = 1*4 + 2 = 6.
        let mut state = GrouscanState::from_seed(&mut s, &inputs, 0, 5, 1, 1, 1);
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
        let mip_base = [0usize, column_offsets.len()];
        let inputs = GrouscanInputs {
            column: &slab_buf[..],
            gylookup: &gylookup,
            gcsub: &gcsub,
            slab_buf: &slab_buf,
            column_offsets: &column_offsets,
            mip_base_offsets: &mip_base,
            vsid: 4,
            sky: None,
            grid_view: crate::grid_view::GridView::from_parts(
                4,
                &slab_buf,
                &column_offsets,
                &mip_base,
            ),
            camera_chunk_z: 0,
            chunk_world_z_base: 0,
            chunk_size_z: 256,
        };
        let mut s = fresh_scratch();
        // BOTH lanes must exceed ngxmax: lane recompute picks the
        // smaller gpz, so the *winning* lane is the one whose gpz
        // is also above ngxmax.
        s.gpz = [0x100, 0x200];
        let mut state = GrouscanState::from_seed(&mut s, &inputs, 0, 0, 0, 0, 1);
        state.ngxmax = 0xFF;
        state.c_idx = CF_SEED_INDEX;

        assert_eq!(
            phase_after_delete_kept_presync(&mut state),
            Phase::Remiporend
        );
    }

    #[test]
    fn remiporend_routes_to_startsky_when_no_more_mips() {
        // Single-mip rendering: gmipnum == 1, gmipcnt starts at 0
        // → (0 + 1) >= 1 → Startsky. This is the path the oracle
        // scenes always take.
        let mut s = fresh_scratch();
        let inputs = dummy_inputs();
        let mut state = GrouscanState::from_seed(&mut s, &inputs, 0, 0, 0, 0, 1);
        state.gmipcnt = 0;
        assert_eq!(phase_remiporend(&mut state), Phase::Startsky);
    }

    #[test]
    fn remiporend_multimip_falls_through_to_startsky() {
        // gmipnum > 1 + gmipcnt+1 < gmipnum hits the multi-mip
        // body. Until the world model carries mip-N+ column data
        // (port of voxlap's genmipvxl), the stub falls through to
        // Phase::Startsky — the visually-correct fallback that
        // fills the unrendered tail with sky rather than leaving
        // the radar uninitialized (which Phase::Done would do).
        // The audit at `phase_remiporend`'s doc comment covers
        // why the full body is gated on multi-mip column_offsets.
        let mut s = fresh_scratch();
        let inputs = dummy_inputs();
        let mut state = GrouscanState::from_seed(&mut s, &inputs, 0, 0, 0, 0, 4);
        state.gmipcnt = 0;
        assert_eq!(phase_remiporend(&mut state), Phase::Startsky);
    }

    #[test]
    fn startsky_returns_done_when_stack_below_seed() {
        let mut s = fresh_scratch();
        let inputs = dummy_inputs();
        let mut state = GrouscanState::from_seed(&mut s, &inputs, 0, 0, 0, 0, 1);
        // ce_idx below seed (defensive — unreachable in normal flow,
        // but voxlap explicitly guards `if (c > ce) goto retsub`).
        state.ce_idx = CF_SEED_INDEX - 1;
        assert_eq!(phase_startsky(&mut state), Phase::Done);
    }

    #[test]
    fn startsky_solid_fills_radar_with_skycast() {
        let mut s = fresh_scratch();
        let sky_col_bits: u32 = 0x80AB_CDEF;
        s.set_skycast(sky_col_bits.cast_signed(), 0x7FFF_FFFF);
        // Seed cf[128] with a 4-slot pixel range [10..=13].
        s.cf[CF_SEED_INDEX].i0 = 10;
        s.cf[CF_SEED_INDEX].i1 = 13;
        let inputs = dummy_inputs();
        let mut state = GrouscanState::from_seed(&mut s, &inputs, 0, 0, 0, 0, 1);
        // Single entry at the seed slot.
        state.ce_idx = CF_SEED_INDEX;

        assert_eq!(phase_startsky(&mut state), Phase::Done);

        for p in 10usize..=13 {
            assert_eq!(state.scratch.radar[p].col, sky_col_bits.cast_signed());
            assert_eq!(state.scratch.radar[p].dist, 0x7FFF_FFFF);
        }
        // Outside the range untouched (default 0).
        assert_eq!(state.scratch.radar[9].col, 0);
        assert_eq!(state.scratch.radar[14].col, 0);
    }

    #[test]
    fn startsky_walks_multiple_cf_entries() {
        let mut s = fresh_scratch();
        s.set_skycast(0x1234_5678, 0);
        s.cf[CF_SEED_INDEX].i0 = 0;
        s.cf[CF_SEED_INDEX].i1 = 1;
        s.cf[CF_SEED_INDEX + 1].i0 = 5;
        s.cf[CF_SEED_INDEX + 1].i1 = 6;
        let inputs = dummy_inputs();
        let mut state = GrouscanState::from_seed(&mut s, &inputs, 0, 0, 0, 0, 1);
        state.ce_idx = CF_SEED_INDEX + 1;

        assert_eq!(phase_startsky(&mut state), Phase::Done);

        assert_eq!(state.scratch.radar[0].col, 0x1234_5678);
        assert_eq!(state.scratch.radar[1].col, 0x1234_5678);
        // Gap [2, 4] untouched.
        assert_eq!(state.scratch.radar[2].col, 0);
        assert_eq!(state.scratch.radar[5].col, 0x1234_5678);
        assert_eq!(state.scratch.radar[6].col, 0x1234_5678);
    }

    #[test]
    fn column_step_routes_to_skipixy3_when_presync_equals_c() {
        let (slab_buf, column_offsets) = build_4x4_world();
        let gylookup = DUMMY_GYLOOKUP;
        let gcsub = DUMMY_GCSUB;
        let mip_base = [0usize, column_offsets.len()];
        let inputs = GrouscanInputs {
            column: &slab_buf[..],
            gylookup: &gylookup,
            gcsub: &gcsub,
            slab_buf: &slab_buf,
            column_offsets: &column_offsets,
            mip_base_offsets: &mip_base,
            vsid: 4,
            sky: None,
            grid_view: crate::grid_view::GridView::from_parts(
                4,
                &slab_buf,
                &column_offsets,
                &mip_base,
            ),
            camera_chunk_z: 0,
            chunk_world_z_base: 0,
            chunk_size_z: 256,
        };
        let mut s = fresh_scratch();
        let mut state = GrouscanState::from_seed(&mut s, &inputs, 0, 0, 0, 0, 1);
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
        let mip_base = [0usize, column_offsets.len()];
        let inputs = GrouscanInputs {
            column: &slab_buf[..],
            gylookup: &gylookup,
            gcsub: &gcsub,
            slab_buf: &slab_buf,
            column_offsets: &column_offsets,
            mip_base_offsets: &mip_base,
            vsid: 4,
            sky: None,
            grid_view: crate::grid_view::GridView::from_parts(
                4,
                &slab_buf,
                &column_offsets,
                &mip_base,
            ),
            camera_chunk_z: 0,
            chunk_world_z_base: 0,
            chunk_size_z: 256,
        };
        let mut s = fresh_scratch();
        let mut state = GrouscanState::from_seed(&mut s, &inputs, 32, 0, 0, 0, 1);
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
    fn intoslabloop_routes_to_drawfwall_when_test_hi_and_test_next_nonpositive() {
        // cx0 = cy0 = cx1 = cy1 = 0 → both cross_sign tests = 0 ≤ 0
        // → slab intersects (test_hi) AND next slab does not extend
        // past ray (test_next ≤ 0) → single-slab DrawFwall.
        let mut s = fresh_scratch();
        let column = [2u8, 10, 12, 0, 0, 0, 0, 0, 0, 20, 22, 0];
        let gylookup = [0i32; 64];
        let gcsub = [0i64; 9];
        let mut state = state_for_drawcwall(&mut s, &column, &gylookup, &gcsub, 0);
        state.ogx = 0;
        assert_eq!(phase_intoslabloop(&mut state), Phase::DrawFwall);
    }

    #[test]
    fn intoslabloop_pushes_split_when_test_next_positive() {
        // test_hi ≤ 0 (slab intersects) AND test_next > 0 (next slab
        // straddles the right edge) → two-slab cfasm split.
        //
        // Setup:
        //   v[0] = 2, v[2] = 12 → next_v3_offset = 2*4+3 = 11.
        //   column[11] = next_v3 = 5. gylookup[5] = 1, cx1 = 1<<16
        //   → test_next = 1 > 0.
        //   gylookup[v[2]+1] = gylookup[13] = 0 → test_hi = 0 ≤ 0.
        //
        // Verify after split:
        //   - ce_idx incremented 128 → 129
        //   - c_idx incremented 128 → 129 (advanced into new slot)
        //   - cf[128].i0 = col+1, cf[128].z0 = next_v3 = 5
        //   - cf[129] holds the original via shift-copy + i1 = col
        //   - state.z0 = original z0 (= 0, via shift-copy)
        //   - state.z1 = next_v3 = 5
        //
        let mut s = fresh_scratch();
        s.cf[CF_SEED_INDEX].i0 = 0;
        s.cf[CF_SEED_INDEX].i1 = 5;
        s.cf[CF_SEED_INDEX].cx1 = 1 << 16;
        s.cf[CF_SEED_INDEX].cy1 = 0;
        s.gi0 = 0;
        s.gi1 = 0;
        let column = [
            2u8, 10, 12, 0, 0, 0, 0, 0, // slab 0 — v[0]=2, v[2]=12, v[3]=0
            0u8, 20, 22, 5, // next slab header — v[3] = 5 (= next_v3)
        ];
        let mut gylookup = [0i32; 64];
        gylookup[5] = 1; // gylookup[next_v3]
                         // gylookup[v[2]+1] = gylookup[13] stays 0 → test_hi = 0.
        let gcsub = [0i64; 9];
        let mut state = state_for_drawcwall(&mut s, &column, &gylookup, &gcsub, 0);
        state.cx1 = 1 << 16;
        state.cy1 = 0;
        state.ogx = 0;
        state.z0 = 0;
        state.z1 = 99; // pre-split sentinel — will be overwritten

        assert_eq!(phase_intoslabloop(&mut state), Phase::DrawFwall);

        assert_eq!(state.ce_idx, CF_SEED_INDEX + 1);
        assert_eq!(state.c_idx, CF_SEED_INDEX + 1);
        assert_eq!(state.z1, 5);
        assert_eq!(state.z0, 0);
        // cf[128] (= old c, post-modification) — z0 = next_v3, i0 = col+1.
        // The search loop's gy_raw is reset to gylookup[v[2]+1] =
        // gylookup[13] = 0, so cross_sign(1<<16, 0, 0, 0) = 0 ≤ 0
        // → break on first iteration → col stays at i1 = 5.
        assert_eq!(state.scratch.cf[CF_SEED_INDEX].z0, 5);
        assert_eq!(state.scratch.cf[CF_SEED_INDEX].i0, 6);
        // cf[129].i1 = col = 5 (shift-copied original i1 was also
        // 5, then overwritten with col which is also 5 — same
        // value either way; this pins the modification fired).
        assert_eq!(state.scratch.cf[CF_SEED_INDEX + 1].i1, 5);
    }

    #[test]
    fn slab_split_returns_done_when_stack_full() {
        // ce_idx already at the cap → push fails → Done.
        let mut s = fresh_scratch();
        s.cf[CF_SEED_INDEX].i0 = 0;
        s.cf[CF_SEED_INDEX].i1 = 5;
        s.cf[CF_SEED_INDEX].cx1 = 1 << 16;
        let column = [2u8, 10, 12, 0, 0, 0, 0, 0, 0u8, 20, 22, 5];
        let mut gylookup = [0i32; 64];
        gylookup[5] = 1;
        let gcsub = [0i64; 9];
        let mut state = state_for_drawcwall(&mut s, &column, &gylookup, &gcsub, 0);
        state.cx1 = 1 << 16;
        state.ce_idx = 191; // at cap
        state.ogx = 0;
        assert_eq!(phase_intoslabloop(&mut state), Phase::Done);
        // ce_idx unchanged (push didn't fire).
        assert_eq!(state.ce_idx, 191);
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
        let mut state = GrouscanState::from_seed(&mut s, &inputs, 0, 0, 0, 0, 1);
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
            chz_layer: 0,
        };
        let inputs = dummy_inputs();
        let mut state = GrouscanState::from_seed(&mut s, &inputs, 0, 0, 0, 0, 1);
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
        let state = GrouscanState::from_seed(&mut s, &inputs, 0, 42, 0, 0, 1);
        assert_eq!(state.ixy_sptr_col_idx, 42);
    }

    #[test]
    fn dispatch_drawflor_when_camera_at_top_of_column() {
        let mut s = fresh_scratch();
        s.gpz = [1_000, 2_000];
        s.gdz = [10, 20];
        let p = grouscan_run(&mut s, &dummy_inputs(), 0, 0, 0, 0, 1, 1);
        assert_eq!(p.dispatch, InitialDispatch::DrawFlor);
    }

    #[test]
    fn dispatch_drawceil_when_camera_in_interior_air_gap() {
        let mut s = fresh_scratch();
        s.gpz = [1_000, 2_000];
        s.gdz = [10, 20];
        // vptr_offset > 0 → camera in interior.
        let p = grouscan_run(&mut s, &dummy_inputs(), 16, 0, 0, 0, 1, 1);
        assert_eq!(p.dispatch, InitialDispatch::DrawCeil);
    }

    #[test]
    fn prologue_ogx_keeps_integer_part_of_gpz() {
        // gpz[0] = 0x12345678 → ogx = 0x12340000.
        let mut s = fresh_scratch();
        s.gpz = [0x1234_5678, 0x7FFF_FFFF];
        s.gdz = [0, 0];
        let p = grouscan_run(&mut s, &dummy_inputs(), 0, 0, 0, 0, 1, 1);
        assert_eq!(p.lane, 0);
        // 0x1234_0000 fits i32 positively (high bit clear).
        assert_eq!(p.ogx, 0x1234_0000_i32);
        assert_eq!(p.gx, 0);
    }
}
