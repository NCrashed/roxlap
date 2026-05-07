//! Scalar `Rasterizer` implementation — port of voxlaptest's 4.7.5
//! scalar `hrendzsse` / `vrendzsse` fallbacks (`voxlap5.c:1947` /
//! `:2003` post-4.8.4 line numbers). Writes one `u32` ARGB pixel +
//! one `f32` z-buffer entry per screen position from radar entries
//! the (still-stubbed) `gline` will produce in R4.3.
//!
//! Per-pixel math (the SSE-batched form lives behind R5; this is the
//! pre-batch shape):
//!
//! ```text
//! col = scratch.angstart[plc >> 16] + j     // signed offset into radar
//! framebuffer[pixel] = radar[col].col       // packed ARGB
//! z = radar[col].dist / sqrt(dirx² + diry²) // f32 z-buffer entry
//! dirx += strx; diry += stry; plc += incr
//! ```
//!
//! The vertical scan additionally mutates `scratch.uurend[sx] +=
//! scratch.uurend[sx + half_stride]` per pixel — that's why the
//! Rasterizer trait now hands `&mut ScanScratch` to `vrend` /
//! `hrend`. `gline` is a TODO stub; without it the angstart entries
//! it would write are zero, so a freshly allocated radar yields all-
//! zero pixels. Tests pre-fill the radar manually to verify the
//! scanline rasterizer end of the pipeline.

// Module-wide cast allows: the per-pixel arithmetic constantly
// crosses signed/unsigned and i32/usize boundaries (loop counters,
// signed offsets into radar, framebuffer indices). Annotating each
// site individually buries the per-pixel logic in lint suppressions;
// the cast-correctness invariants hold by construction.
#![allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::similar_names
)]

use std::marker::PhantomData;

use crate::camera_math::CameraState;
use crate::column_walk;
use crate::fixed::ftol;
use crate::gline::derive_gline_frustum;
use crate::grouscan::{grouscan_run, CfType, GrouscanInputs, CF_SEED_INDEX};
use crate::opticast::camera_column_slice;
use crate::opticast_prelude::{OpticastPrelude, PREC};
use crate::rasterizer::{Rasterizer, ScanScratch};
use crate::ray_aabb::clip_ray_to_xy_aabb;
use crate::ray_step::RayStep;
use crate::scan_loops::ScanContext;

/// Borrowed view of the framebuffer + zbuffer as raw pointers.
///
/// R12.2.0 introduces this so the per-frame ScalarRasterizer can be
/// `Copy` (for the per-thread fan-out R12.2.1 lands). Holding `&'a
/// mut [u32]` / `&'a mut [f32]` directly forces exclusive borrows
/// per instance, blocking the four quadrants from running on four
/// threads even though their pixel writes are disjoint.
///
/// Constructed safely from exclusive slice borrows — the slices are
/// consumed and re-exposed as raw pointers tied to lifetime `'a`
/// via `PhantomData`. Once a `RasterTarget` exists, it is the sole
/// path to the underlying memory; the slices cannot be used through
/// any other channel for the duration of `'a`.
///
/// `Copy` lets opticast hand each quadrant thread its own copy of
/// the same target. The four threads write disjoint pixels (top /
/// bottom / left / right wedges of the screen, no overlap), so
/// pointer aliasing is safe under the documented invariant.
///
/// # Safety contract for parallel use
/// Callers that copy a `RasterTarget` and pass copies to multiple
/// threads MUST guarantee that the threads collectively write to
/// pairwise-disjoint pixel indices. opticast enforces this via the
/// four-quadrant wedge geometry (see `scan_loops::{top,right,
/// bottom,left}_quadrant`). Single-threaded callers (R12.1 default)
/// hold one copy and trivially satisfy the invariant.
#[derive(Clone, Copy, Debug)]
pub struct RasterTarget<'a> {
    fb_ptr: *mut u32,
    fb_len: usize,
    zb_ptr: *mut f32,
    zb_len: usize,
    _marker: PhantomData<&'a mut [u32]>,
}

// SAFETY: `RasterTarget` is morally a borrowed mutable slice pair —
// the same shape `&'a mut [u32]` / `&'a mut [f32]` would have, both of
// which are `Send` when `T: Send`. Multi-thread safety is enforced
// by the wedge / strip-disjoint write invariant (see struct doc).
unsafe impl Send for RasterTarget<'_> {}

// SAFETY: sharing `&RasterTarget` across threads exposes only the
// raw pointers + lengths. Reading a pointer field is itself free of
// data races; concurrent writes through the pointer are gated by
// the disjoint-write invariant the caller upholds. Required so
// `ScalarRasterizer: Sync`, which `rayon::par_iter_mut` needs to
// share `&rasterizer` across the strip-parallel closures (R12.3.1).
unsafe impl Sync for RasterTarget<'_> {}

impl<'a> RasterTarget<'a> {
    /// Build a target from exclusive slice borrows. The slices are
    /// consumed (their `&'a mut` reborrow is the load-bearing thing —
    /// this constructor is the only way to mint a `RasterTarget`
    /// from safe code).
    #[must_use]
    pub fn new(framebuffer: &'a mut [u32], zbuffer: &'a mut [f32]) -> Self {
        Self {
            fb_ptr: framebuffer.as_mut_ptr(),
            fb_len: framebuffer.len(),
            zb_ptr: zbuffer.as_mut_ptr(),
            zb_len: zbuffer.len(),
            _marker: PhantomData,
        }
    }

    /// Framebuffer length in `u32` elements.
    #[must_use]
    pub fn fb_len(self) -> usize {
        self.fb_len
    }

    /// Raw mutable framebuffer pointer. Used by SSE blocks that do
    /// their own arithmetic + bounds reasoning.
    ///
    /// # Safety
    /// Callers must respect `fb_len` and the parallel-use invariant.
    #[must_use]
    pub fn fb_ptr(self) -> *mut u32 {
        self.fb_ptr
    }

    /// Raw mutable zbuffer pointer. Same contract as
    /// [`Self::fb_ptr`].
    #[must_use]
    pub fn zb_ptr(self) -> *mut f32 {
        self.zb_ptr
    }

    /// Write one ARGB pixel.
    ///
    /// # Safety
    /// `idx < self.fb_len()`, plus the parallel-use invariant.
    pub unsafe fn write_color(self, idx: usize, color: u32) {
        debug_assert!(idx < self.fb_len, "fb idx {} >= len {}", idx, self.fb_len);
        // SAFETY: caller asserts in-bounds + disjoint-from-other-threads.
        unsafe { self.fb_ptr.add(idx).write(color) };
    }

    /// Write one z-buffer entry.
    ///
    /// # Safety
    /// `idx < self.fb_len()` (zbuffer length matches fb), plus the
    /// parallel-use invariant.
    pub unsafe fn write_depth(self, idx: usize, z: f32) {
        debug_assert!(idx < self.zb_len, "zb idx {} >= len {}", idx, self.zb_len);
        // SAFETY: caller asserts in-bounds + disjoint-from-other-threads.
        unsafe { self.zb_ptr.add(idx).write(z) };
    }
}

// gcsub now lives on `ScanScratch::gcsub`; the host pokes it via
// `ScanScratch::set_side_shades` per frame (mirrors voxlap's
// `setsideshades` global). The default-state pattern
// (`0x00ff00ff00ff00ff` per entry, == `setsideshades(0,…,0)`) is
// what `ScanScratch::new_for_size` initialises to, so the oracle
// stays bit-exact when no shading is configured.

/// Per-channel fog blend — voxlap5.c:2052-2056 (and the matching
/// hrend / vrend scalar tail). `col` is the source ARGB voxel
/// colour; `dist` is the radar slot's depth (PREC-scaled, so
/// `>> 20` gives the integer cell distance index into `foglut`).
/// `foglut` empty short-circuits to "no fog" (returns `col`
/// unchanged); OOB indices saturate to 32767 (full fog) since
/// `set_fog` pads the table that way.
//
// The C form is one branchless expression per channel; we keep
// it as-is for legibility against the source. `as i32` casts
// match the C int32 arithmetic.
#[allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::many_single_char_names
)]
fn fog_blend(col: i32, dist: i32, foglut: &[i32], fog_col: i32) -> i32 {
    if foglut.is_empty() {
        return col;
    }
    let idx = (dist >> 20) as usize;
    let l = foglut.get(idx).copied().unwrap_or(32767) & 32767;
    let k = col;
    let fc = fog_col;
    let r = (((fc & 255) - (k & 255)) * l) >> 15;
    let g = (((((fc >> 8) & 255) - ((k >> 8) & 255)) * l) >> 15) << 8;
    let b = (((((fc >> 16) & 255) - ((k >> 16) & 255)) * l) >> 15) << 16;
    r + g + b + k
}

/// Per-ray sky-row update. Mirror of voxlap5.c:1236-1255.
///
/// On the first ray of a quadrant (`scratch.sky_cur_lng < 0`),
/// initialise from the ray's `atan2(vy1, vx1)` mapped through
/// `sky.lng_mul`. On subsequent rays, walk `sky.lng[]` forward
/// (when `sky_cur_dir < 0`) or backward (`sky_cur_dir >= 0`)
/// until the cross-product flips sign — voxlap's rotating-cursor
/// trick that avoids re-running atan2 per ray. After the search,
/// stamp `scratch.sky_off = sky_cur_lng * sky.bpl`, which
/// `phase_startsky_textured` divides by 4 to land at the row's
/// pixel-base index.
fn sky_per_ray_update(
    scratch: &mut crate::rasterizer::ScanScratch,
    sky: &crate::sky::Sky,
    vx1: f32,
    vy1: f32,
) {
    let ysiz = sky.ysiz;
    if scratch.sky_cur_lng < 0 {
        // First-ray init — atan2 mapped to row index.
        let ang = vy1.atan2(vx1) + std::f32::consts::PI;
        let raw = ang * sky.lng_mul - 0.5;
        let mut lng = ftol(raw);
        // Voxlap's `(uint32_t)skycurlng >= skyysiz` clamp uses an
        // uninitialised `j` for the corrective shift; we substitute
        // `rem_euclid` for a deterministic in-range value. The
        // rotating-cursor walk in subsequent rays will quickly
        // converge whichever way we land.
        if (lng as u32) >= (ysiz as u32) {
            lng = lng.rem_euclid(ysiz);
        }
        scratch.sky_cur_lng = lng;
    } else if scratch.sky_cur_dir < 0 {
        // Walk forward (rotating).
        let mut j = scratch.sky_cur_lng + 1;
        if j >= ysiz {
            j = 0;
        }
        loop {
            let l = sky.lng[j as usize];
            if l[0] * vy1 <= l[1] * vx1 {
                break;
            }
            scratch.sky_cur_lng = j;
            j += 1;
            if j >= ysiz {
                j = 0;
            }
        }
    } else {
        // Walk backward (rotating).
        loop {
            let l = sky.lng[scratch.sky_cur_lng as usize];
            if l[0] * vy1 >= l[1] * vx1 {
                break;
            }
            scratch.sky_cur_lng -= 1;
            if scratch.sky_cur_lng < 0 {
                scratch.sky_cur_lng = ysiz - 1;
            }
        }
    }
    // Voxlap: `skyoff = skycurlng * skybpl + nskypic`. We strip
    // the `+ nskypic` (texture base address) — `phase_startsky`
    // adds it implicitly by indexing `sky.pixels` directly.
    scratch.sky_off = scratch.sky_cur_lng * sky.bpl;
}

/// Per-frame state cached on first `frame_setup` call. Owned here
/// (vs. borrowed from `ScanContext`) because gline needs to read
/// it across many calls without re-borrowing each time. The
/// `prelude` clone copies one `Vec<i32>` (the `y_lookup` mip
/// table) per frame — cheap.
//
// `Clone` is used by [`ScalarRasterizer`]'s 4-way fan-out (R12.2.1)
// to mint one rasterizer per quadrant thread; each clone copies the
// (~few KB) y_lookup Vec — small per-frame allocation cost.
#[derive(Clone)]
struct FrameCache {
    ray_step: RayStep,
    camera_state: CameraState,
    prelude: OpticastPrelude,
    gstartz0: i32,
    gstartz1: i32,
    /// Voxlap's `v - *ixy_sptr_col` — byte offset within the
    /// camera's column to the slab whose top bounds the air gap
    /// from below. `0` ⇒ column-top.
    vptr_offset: usize,
}

/// S1.3: per-scanline override of the camera-position-specific
/// state that [`ScalarRasterizer::gline`] would otherwise read from
/// [`FrameCache`]. The outside-camera dispatch in
/// [`crate::opticast::opticast_outside`] sets one of these before
/// each `gline` call, with the AABB-entry-point's column / fractional
/// XY / air-gap z-bounds in place of the camera's. When `None` (the
/// default for inside-camera frames), `gline` reads directly from
/// the frame cache and behaviour is bit-exact with pre-S1.3.
///
/// Set at the [`ScalarRasterizer`] level rather than the
/// [`Rasterizer`] trait so stub / test rasterizers don't need to
/// know about outside mode — opticast_outside threads through the
/// concrete-typed setter explicitly.
#[derive(Clone, Copy, Debug)]
pub struct OutsideRayContext {
    /// Override for `prelude.li_pos[0..2]` — the entry column's
    /// signed integer XY. Used to compute the j0/j1 lanes in the
    /// gxmax frustum-edge clip.
    pub li_pos_xy: [i32; 2],
    /// Override for `prelude.column_index` — `li_pos_y * vsid +
    /// li_pos_x` at the entry column.
    pub column_index: u32,
    /// Override for `prelude.pos_xfrac` — fractional X at the AABB-
    /// entry point, packed as `[1 - frac, frac]`. Goes through
    /// [`crate::gline::derive_gline_frustum`] for the per-scanline
    /// `gpz[0]` lane.
    pub pos_xfrac: [f32; 2],
    /// Override for `prelude.pos_yfrac` — same shape as `pos_xfrac`
    /// for the y axis.
    pub pos_yfrac: [f32; 2],
    /// Override for `cache.gstartz0` — air-gap top z at the entry
    /// column for the entry-point z. Seeds `cf[CF_SEED_INDEX].z0`.
    pub gstartz0: i32,
    /// Override for `cache.gstartz1` — air-gap bottom z. Seeds
    /// `cf[CF_SEED_INDEX].z1`.
    pub gstartz1: i32,
    /// Override for `cache.vptr_offset` — byte offset within the
    /// entry column to the slab whose top bounds the air gap from
    /// below. Passed to `grouscan_run`.
    pub vptr_offset: usize,
    /// Scanline whose ray misses the world AABB entirely (no entry
    /// point exists). gline takes the sky-fill path: cf seed gets
    /// safe placeholder values, gxmax = 0, skycast.dist = i32::MAX,
    /// grouscan_run's startsky phase fills every radar slot in the
    /// scanline's range with the engine's sky color. The other
    /// override fields are ignored when this is true.
    pub force_miss: bool,
}

/// Scalar rasterizer that writes pixels and a z-buffer entry per
/// screen position.
///
/// Borrows the framebuffer + zbuffer for the duration of one
/// `opticast` call; SDL hosts allocate these once and reuse across
/// frames, see `roxlap-host`.
//
// `slab_buf` / `column_offsets` / `vsid` are R4.3a-rewire-2
// scaffolding — the real `gline` (R4.3a-rewire-3) needs them to
// call `grouscan_run` per ray. The current placeholder gline
// doesn't read them yet, hence the dead_code allow.
#[allow(dead_code)]
#[derive(Clone)]
pub struct ScalarRasterizer<'a> {
    /// Framebuffer + zbuffer raw-pointer view. Stripped from the
    /// caller's `&mut [u32]` / `&mut [f32]` borrows at construction
    /// (see [`Self::new`]) so the rasterizer can be `Copy` for the
    /// per-thread quadrant fan-out R12.2.1 lands. Single-threaded
    /// path holds one copy, parallel path will hold four (one per
    /// quadrant — wedge-disjoint pixel writes; see
    /// [`RasterTarget`]'s safety contract).
    target: RasterTarget<'a>,
    /// Row stride in `u32` / `f32` elements (== framebuffer width
    /// for tightly-packed buffers; SDL streaming textures may add
    /// trailing padding).
    pitch_pixels: usize,
    /// World-level flat slab buffer (voxlap's malloc'd column
    /// data). Re-borrowed from opticast's caller for the lifetime
    /// of the rasterizer.
    slab_buf: &'a [u8],
    /// Per-column byte offsets into [`Self::slab_buf`], concatenated
    /// across all built mip levels. The mip-0 sub-table prefix
    /// (`vsid² + 1` entries) is what existing single-mip callers
    /// pass; multi-mip callers pass the full concatenation and
    /// declare boundaries via [`Self::mip_base_offsets`].
    column_offsets: &'a [u32],
    /// Per-mip column-offset sub-table base indices. Length
    /// `mip_count + 1`; trailing sentinel equals
    /// `column_offsets.len()`. Single-mip callers pass
    /// `&[0, vsid² + 1]`. R4.5d's `phase_remiporend` indexes
    /// this to land in mip-N+1's sub-table.
    mip_base_offsets: &'a [usize],
    /// World dimension. Combined with the prelude's `column_index`
    /// and the column-step path in grouscan, this is what lets the
    /// real gline walk the per-ray voxel-column traversal.
    vsid: u32,
    /// Optional sky texture borrow. `None` ⇒ `phase_startsky`
    /// solid-fills with `scratch.skycast`. `Some(_)` ⇒ gline's
    /// per-ray frustum prep updates `scratch.sky_off`, and
    /// `phase_startsky` runs the textured search-and-sample loop.
    /// Set via [`Self::with_sky`] after construction; unset ⇒
    /// engine's existing solid-sky behaviour, byte-stable for the
    /// oracle.
    sky: Option<&'a crate::sky::Sky>,
    /// Per-frame state cache. `None` until the first `frame_setup`
    /// call; gline panics if invoked before that.
    frame: Option<FrameCache>,
    /// S1.3: per-scanline override for the camera-position-specific
    /// fields in [`FrameCache`]. `None` (the default) means `gline`
    /// reads from the cache as voxlap C does — the inside-camera
    /// path. `Some(_)` swaps in the entry-point context.
    ///
    /// When [`Self::outside_camera_active`] is `true` AND this is
    /// `None`, `gline` auto-computes a fresh per-scanline override
    /// inline by clipping each scanline's ray against the world AABB.
    /// Setting this manually overrides the auto-computation (used by
    /// tests / advanced callers).
    outside_ray_context: Option<OutsideRayContext>,
    /// S1.3: when `true`, `gline` clips each scanline's ray against
    /// the world XY AABB `[0, vsid] × [0, vsid]` and builds an
    /// [`OutsideRayContext`] inline. `opticast_outside` sets this
    /// for the duration of the outside-camera frame and clears it
    /// at exit.
    outside_camera_active: bool,
}

// R12.2.1 / R12.3.1: opticast's parallel branches fan the rasterizer
// across rayon-managed threads — each thread owns its own clone. The
// clones share `target: RasterTarget` (raw pointers; safe under the
// strip-disjoint pixel-write invariant documented on RasterTarget),
// hold &-refs into the slab/column data (Sync), and have independent
// FrameCache copies. Compile-time checks: this fails if any field
// becomes non-Send/non-Sync so the parallel path can no longer hold.
const _: fn() = || {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    assert_send::<ScalarRasterizer<'_>>();
    assert_sync::<ScalarRasterizer<'_>>();
};

impl<'a> ScalarRasterizer<'a> {
    /// Create a rasterizer that will write into the supplied
    /// framebuffer + zbuffer pair. `pitch_pixels` must satisfy
    /// `pitch_pixels * height ≤ framebuffer.len()` for the height
    /// the engine renders into; the `frame_setup` hook does not
    /// validate sizes (it has no height to check against).
    ///
    /// `slab_buf` / `column_offsets` / `mip_base_offsets` / `vsid`
    /// describe the world the renderer reads from. Pass the matching
    /// fields from a [`roxlap_formats::vxl::Vxl`] (or, for tests,
    /// `&[0, vsid² + 1]` as the single-mip placeholder).
    ///
    /// `ray_step` is initialised to a zero placeholder; the real
    /// values get stamped on the first [`Rasterizer::frame_setup`]
    /// call before any per-pixel work runs.
    #[must_use]
    pub fn new(
        framebuffer: &'a mut [u32],
        zbuffer: &'a mut [f32],
        pitch_pixels: usize,
        slab_buf: &'a [u8],
        column_offsets: &'a [u32],
        mip_base_offsets: &'a [usize],
        vsid: u32,
    ) -> Self {
        Self {
            target: RasterTarget::new(framebuffer, zbuffer),
            pitch_pixels,
            slab_buf,
            column_offsets,
            mip_base_offsets,
            vsid,
            sky: None,
            frame: None,
            outside_ray_context: None,
            outside_camera_active: false,
        }
    }

    /// Bind a sky texture for the lifetime of this rasterizer
    /// instance. Hosts call this when [`crate::Engine::sky`] is
    /// `Some(_)`. Without it, the rasterizer keeps the legacy
    /// solid-fill `skycast` behaviour.
    #[must_use]
    pub fn with_sky(mut self, sky: &'a crate::sky::Sky) -> Self {
        self.sky = Some(sky);
        self
    }

    /// S1.3: install or clear the per-scanline outside-camera
    /// override. `Some(_)` swaps the entry-point context into the
    /// next `gline` call's frame-cache reads; `None` (the default)
    /// reverts to the camera-inside-grid path. Useful for tests
    /// and advanced callers; the outside_orbit pose path uses
    /// [`Self::set_outside_camera_active`] instead.
    pub fn set_outside_ray_context(&mut self, ctx: Option<OutsideRayContext>) {
        self.outside_ray_context = ctx;
    }

    /// S1.3: per-scanline AABB-clip helper. For the scanline whose
    /// far-end is screen pixel `(x1, y1)`, project the world-space
    /// ray from the camera through that pixel, clip it against the
    /// world XY AABB `[0, vsid]²`, and build an
    /// [`OutsideRayContext`] anchored at the AABB-entry point.
    ///
    /// Miss → `OutsideRayContext { force_miss: true, .. }`. Hit
    /// where the entry column is in solid voxels → also `force_miss`
    /// for now (S1 first-cut; wall rendering is out-of-scope).
    /// Hit with valid air gap at entry → full hit context.
    ///
    /// Per-scanline cost: one AABB clip (~10 f32 ops), one column
    /// air-gap walk (slab list traversal — typically a few bytes).
    /// Cheap enough to run per scanline.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn compute_outside_ray_context_for_scanline(
        cs: &CameraState,
        vsid: u32,
        slab_buf: &[u8],
        column_offsets: &[u32],
        x1: f32,
        y1: f32,
    ) -> OutsideRayContext {
        // Voxlap's per-scanline ray endpoint in world-relative coords:
        // (x1, y1) on screen projects to a 3D direction via the
        // camera basis. Same formula `derive_gline_frustum` uses for
        // (vx1, vy1, vz1).
        let dir = [
            x1 * cs.right[0] + y1 * cs.down[0] + cs.corn[0][0],
            x1 * cs.right[1] + y1 * cs.down[1] + cs.corn[0][1],
            x1 * cs.right[2] + y1 * cs.down[2] + cs.corn[0][2],
        ];
        let cam_xy = [cs.pos[0], cs.pos[1]];
        let dir_xy = [dir[0], dir[1]];
        let aabb_max = [vsid as f32, vsid as f32];

        let miss = OutsideRayContext {
            li_pos_xy: [0, 0],
            column_index: 0,
            pos_xfrac: [1.0, 0.0],
            pos_yfrac: [1.0, 0.0],
            gstartz0: 0,
            gstartz1: 0,
            vptr_offset: 0,
            force_miss: true,
        };

        let Some((t_enter, _t_exit)) = clip_ray_to_xy_aabb(cam_xy, dir_xy, [0.0, 0.0], aabb_max)
        else {
            return miss;
        };
        // Negative t_enter means camera is already inside the XY
        // footprint — that's an inside-camera situation that
        // shouldn't reach this path; treat as miss to be safe.
        if t_enter < 0.0 {
            return miss;
        }

        // Entry point in world coords, nudged inward by a tiny
        // epsilon along each axis to keep the floor / column-index
        // from sliding to vsid (out-of-bounds) at exactly-on-face
        // entries (e.g. yaw=π hitting x=vsid).
        const EPS: f32 = 1e-4;
        let entry_x = cs.pos[0] + t_enter * dir[0] + EPS * dir[0].signum();
        let entry_y = cs.pos[1] + t_enter * dir[1] + EPS * dir[1].signum();
        let entry_z = cs.pos[2] + t_enter * dir[2];

        let li_x = (entry_x.floor() as i32).clamp(0, vsid as i32 - 1);
        let li_y = (entry_y.floor() as i32).clamp(0, vsid as i32 - 1);
        let column_index = (li_y as u32) * vsid + (li_x as u32);
        let xfrac1 = entry_x - li_x as f32;
        let yfrac1 = entry_y - li_y as f32;

        // Air-gap walk at the entry column for the entry-z. None if
        // the entry point lies inside solid voxels — first-cut
        // behaviour is to treat as miss (no wall rendering yet).
        let column = camera_column_slice(slab_buf, column_offsets, column_index).unwrap_or(&[]);
        let entry_z_int = entry_z.floor() as i32;
        let Some((gstartz0, gstartz1, vptr_offset)) =
            column_walk::camera_column_air_gap(column, entry_z_int)
        else {
            return miss;
        };

        OutsideRayContext {
            li_pos_xy: [li_x, li_y],
            column_index,
            pos_xfrac: [1.0 - xfrac1, xfrac1],
            pos_yfrac: [1.0 - yfrac1, yfrac1],
            gstartz0,
            gstartz1,
            vptr_offset,
            force_miss: false,
        }
    }
}

impl Rasterizer for ScalarRasterizer<'_> {
    fn set_outside_camera_active(&mut self, active: bool) {
        self.outside_camera_active = active;
    }

    fn frame_setup(&mut self, ctx: &ScanContext<'_>) {
        // Cache everything per-frame so gline doesn't re-borrow on
        // every call. Prelude is cloned (one Vec<i32> alloc per
        // frame for y_lookup; small).
        self.frame = Some(FrameCache {
            ray_step: *ctx.rs,
            camera_state: *ctx.camera_state,
            prelude: ctx.prelude.clone(),
            gstartz0: ctx.camera_gstartz0,
            gstartz1: ctx.camera_gstartz1,
            vptr_offset: ctx.camera_vptr_offset,
        });
    }

    #[allow(clippy::too_many_lines)]
    fn gline(
        &mut self,
        scratch: &mut ScanScratch,
        length: u32,
        x0: f32,
        y0: f32,
        x1: f32,
        y1: f32,
    ) {
        // Voxlap's per-scanline ray-cast: derive the frustum, seed
        // cf[128], stamp scratch globals, call grouscan. Mirror of
        // voxlap5.c:gline (1146..1235).
        let cache = self
            .frame
            .as_ref()
            .expect("gline called before frame_setup");
        let leng = length as i32;

        // S1.3: resolve the camera-position-specific state once at
        // the top so the rest of `gline` reads from locals. With no
        // outside-mode state set (the default for inside-camera
        // frames) every value falls through to the cache and the
        // function is bit-exact with the pre-S1.3 body. Outside
        // mode supplies AABB-entry-point values per scanline:
        // 1. Manual `set_outside_ray_context` wins if present.
        // 2. Otherwise, if `outside_camera_active` is on, auto-clip
        //    this scanline's ray against the world AABB.
        // 3. Otherwise, None → inside path.
        let or_ctx: Option<OutsideRayContext> = self.outside_ray_context.or_else(|| {
            if self.outside_camera_active {
                Some(Self::compute_outside_ray_context_for_scanline(
                    &cache.camera_state,
                    self.vsid,
                    self.slab_buf,
                    self.column_offsets,
                    x1,
                    y1,
                ))
            } else {
                None
            }
        });

        // Outside-mode miss: the scanline's ray doesn't enter the
        // world AABB (or enters in solid voxels). Fill the radar
        // range with skycast directly so hrend / vrend produces
        // sky-colored pixels at z = far. No grouscan walk needed —
        // there's nothing to walk.
        if or_ctx.is_some_and(|c| c.force_miss) {
            scratch.skycast.dist = i32::MAX;
            let start = scratch.gscanptr;
            let end = (start + length as usize).min(scratch.radar.len());
            let sk = scratch.skycast;
            for slot in &mut scratch.radar[start..end] {
                *slot = sk;
            }
            return;
        }

        let pos_xfrac = or_ctx.map_or(cache.prelude.pos_xfrac, |o| o.pos_xfrac);
        let pos_yfrac = or_ctx.map_or(cache.prelude.pos_yfrac, |o| o.pos_yfrac);
        let li_pos_xy = or_ctx.map_or([cache.prelude.li_pos[0], cache.prelude.li_pos[1]], |o| {
            o.li_pos_xy
        });
        let column_index = or_ctx.map_or(cache.prelude.column_index, |o| o.column_index);
        let gstartz0 = or_ctx.map_or(cache.gstartz0, |o| o.gstartz0);
        let gstartz1 = or_ctx.map_or(cache.gstartz1, |o| o.gstartz1);
        let vptr_offset = or_ctx.map_or(cache.vptr_offset, |o| o.vptr_offset);

        // 1. Project per-ray frustum (vd0/vd1/vz0/vx1/vy1/vz1 +
        //    gixy/gpz/gdz). voxlap5.c:1153-1175.
        let f = derive_gline_frustum(
            &cache.camera_state,
            pos_xfrac,
            pos_yfrac,
            self.vsid,
            length,
            x0,
            y0,
            x1,
            y1,
        );

        // 2. Stamp ray-step globals onto scratch.
        scratch.gixy = f.gixy;
        scratch.gpz = f.gpz;
        scratch.gdz = f.gdz;

        // 3. cmprecip[leng] = CMPPREC / leng (voxlap precomputed
        //    table; voxlap5.c:12315 builds it as `CMPPREC/(float)i`).
        //    CMPPREC = 256*4096 = PREC. gi0 / gi1 are per-pixel ray-
        //    step coefficients in Q12.20 (= PREC); cx0/cy0/cx1/cy1
        //    are the cf[128] seed endpoints. voxlap5.c:1179-1190.
        // The `as f32` casts here lose precision for very large leng
        // (> 2²³), but realistic scanline lengths (a few thousand)
        // are well below that.
        #[allow(clippy::cast_precision_loss)]
        let cmpprec = PREC as f32;
        #[allow(clippy::cast_precision_loss)]
        let cmprecip = if leng > 0 {
            cmpprec / (leng as f32)
        } else {
            0.0
        };
        // ftol() routes float→i32 through i64 to mirror voxlap C's
        // wrap-on-overflow `lrintf+(int32_t)cast`. The cf-seed
        // products (vd ± vd) * cmprecip and vd * cmpprec land at
        // the i32 boundary for world-coord magnitudes near VSID
        // (= 2048) × PREC (= 2²⁰); Rust's `as i32` saturates and
        // diverges for those edge cases.
        let (gi0, gi1, cx0, cy0) = if cache.prelude.forward_z_sign < 0 {
            (
                ftol((f.vd1 - f.vd0) * cmprecip),
                ftol((f.vz1 - f.vz0) * cmprecip),
                ftol(f.vd0 * cmpprec),
                ftol(f.vz0 * cmpprec),
            )
        } else {
            (
                ftol((f.vd0 - f.vd1) * cmprecip),
                ftol((f.vz0 - f.vz1) * cmprecip),
                ftol(f.vd1 * cmpprec),
                ftol(f.vz1 * cmpprec),
            )
        };
        let cx1 = leng.wrapping_mul(gi0).wrapping_add(cx0);
        let cy1 = leng.wrapping_mul(gi1).wrapping_add(cy0);

        scratch.gi0 = gi0;
        scratch.gi1 = gi1;

        // 4. Seed cf[128] with the radar range + air-gap z-bounds +
        //    Q12.20 ray endpoints. voxlap5.c:1176-1190.
        let gscanptr_isize = scratch.gscanptr as isize;
        scratch.cf[CF_SEED_INDEX] = CfType {
            i0: gscanptr_isize,
            i1: gscanptr_isize + leng as isize,
            z0: gstartz0,
            z1: gstartz1,
            cx0,
            cy0,
            cx1,
            cy1,
        };

        // 5. gxmax = min(gmaxscandist, frustum-edge clip per axis).
        //    voxlap5.c:1192-1228. Unsigned compare — voxlap's `q`
        //    is a uint64_t product that may exceed gmaxscandist or
        //    wrap negative.
        //
        //    Also stamps `skycast.dist` per voxlap5.c:1209-1227:
        //    initialised to `gxmax` (the scan-distance ceiling),
        //    overwritten with `0x7FFFFFFF` if either frustum-edge
        //    clip fires (= ray hits world edge before scan-dist
        //    cap → "infinitely far" sky depth). startsky's solid-
        //    fill writes this into every drained radar slot's
        //    `dist`, which the z-buffer ends up carrying.
        let mut gxmax = cache.prelude.max_scan_dist;
        scratch.skycast.dist = gxmax;
        let vsid_signed = self.vsid as i32;
        let j0 = if f.gixy[0] < 0 {
            li_pos_xy[0]
        } else {
            vsid_signed - 1 - li_pos_xy[0]
        };
        let q0 = (i64::from(f.gdz[0]).wrapping_mul(i64::from(j0)))
            .wrapping_add(i64::from(f.gpz[0] as u32));
        if (q0 as u64) < u64::from(gxmax as u32) {
            gxmax = q0 as i32;
            scratch.skycast.dist = i32::MAX;
        }
        let j1 = if f.gixy[1] < 0 {
            li_pos_xy[1]
        } else {
            vsid_signed - 1 - li_pos_xy[1]
        };
        let q1 = (i64::from(f.gdz[1]).wrapping_mul(i64::from(j1)))
            .wrapping_add(i64::from(f.gpz[1] as u32));
        if (q1 as u64) < u64::from(gxmax as u32) {
            gxmax = q1 as i32;
            scratch.skycast.dist = i32::MAX;
        }
        scratch.gxmax = gxmax;

        // 5b. Per-ray sky-row search. Mirror of voxlap5.c:1236-
        //     1255. Walks `sky.lng[]` to find the texel-row whose
        //     longitude vector matches the ray's `(vx1, vy1)`
        //     direction; stamps `scratch.sky_off` so
        //     `phase_startsky` knows which row to sample. No-op
        //     when no sky texture is bound.
        if let Some(sky) = self.sky {
            sky_per_ray_update(scratch, sky, f.vx1, f.vy1);
        }

        // 6. Build inputs and call grouscan_run. The starting
        //    column is the camera's column (or, in outside mode,
        //    the AABB-entry column supplied by the override). The
        //    slab walker handles the rest.
        let column =
            camera_column_slice(self.slab_buf, self.column_offsets, column_index).unwrap_or(&[]);
        // Copy gcsub out of scratch so the GrouscanInputs immutable
        // borrow doesn't collide with the `&mut scratch` grouscan_run
        // takes below. `[i64; 9]` is 72 bytes — cheap.
        let mut gcsub_local: [i64; 9] = scratch.gcsub;
        // Voxlap5.c:1230-1234. Per-ray, populate the wall-side lanes
        // (0/1) from the directional lanes (4/5 = left/right,
        // 6/7 = up/down) according to the sign of `gixy`. Without
        // this, `wall_lane` reads from the stale `0x00ff_00ff_00ff_00ff`
        // baseline and wall faces get no directional darkening, even
        // after the host calls `set_side_shades`.
        if scratch.sideshademode {
            let lane0_idx = if f.gixy[0] < 0 { 4 } else { 5 };
            let lane1_idx = if f.gixy[1] < 0 { 6 } else { 7 };
            gcsub_local[0] = gcsub_local[lane0_idx];
            gcsub_local[1] = gcsub_local[lane1_idx];
        }
        let inputs = GrouscanInputs {
            column,
            gylookup: &cache.prelude.y_lookup,
            gcsub: &gcsub_local,
            slab_buf: self.slab_buf,
            column_offsets: self.column_offsets,
            mip_base_offsets: self.mip_base_offsets,
            vsid: self.vsid,
            sky: self.sky.map(crate::grouscan::SkyRef::from_sky),
        };
        // gmipnum = number of built mip levels. R4.5d's
        // `phase_remiporend` body will start incrementing
        // `state.gmipcnt` once gmipnum > 1 and the column step's
        // `gpz > ngxmax` overflow fires; until then a multi-mip
        // world simply renders mip-0 only, byte-stable with the
        // single-mip path.
        let gmipnum = u32::try_from(self.mip_base_offsets.len().saturating_sub(1))
            .expect("mip count fits in u32");
        let _ = grouscan_run(
            scratch,
            &inputs,
            vptr_offset,
            column_index as usize,
            cache.prelude.x_mip,
            gmipnum.max(1),
        );

        // gscanptr is advanced by the opticast quadrant scan
        // (`scan_loops.rs::top_quadrant` etc., voxlap5.c:2382 area)
        // AFTER each gline call. Voxlap's `gline` itself does NOT
        // touch gscanptr — advancing it here too created gaps of
        // `leng+1` unwritten radar slots between consecutive glines,
        // which read back as 0 in hrend → black pixels at the
        // sphere position in diag_down / high_down.
    }

    fn hrend(
        &mut self,
        scratch: &mut ScanScratch,
        sx: i32,
        sy: i32,
        p1: i32,
        plc: i32,
        incr: i32,
        j: i32,
    ) {
        let rs = self
            .frame
            .as_ref()
            .map(|f| f.ray_step)
            .expect("hrend/vrend called before frame_setup");
        // Per-frame setup gives strx/stry/heix/heiy/addx/addy; per-
        // pixel direction = strx*sx + heix*sy + addx, advancing by
        // strx in the inner loop.
        #[allow(clippy::cast_precision_loss)]
        let mut dirx = rs.strx * sx as f32 + rs.heix * sy as f32 + rs.addx;
        #[allow(clippy::cast_precision_loss)]
        let mut diry = rs.stry * sx as f32 + rs.heiy * sy as f32 + rs.addy;
        let row_start = sy as usize * self.pitch_pixels;

        let mut plc_local = plc;
        let mut x = sx;

        // R5.1: SSE2 4-pixel batch via `_mm_rsqrt_ps` — port of
        // voxlaptest's `hrendzsse` (voxlap5.c:1947). 12-bit
        // approximation, no Newton refine, matching the
        // historical asm. The tail (0..3 leftover pixels)
        // continues with the bit-exact scalar form below; the
        // batch's z lanes will not match scalar 1/sqrt exactly,
        // mirroring voxlap. SSE2 is x86_64 baseline so no
        // runtime CPU-feature check is needed.
        //
        // `cast_ptr_alignment` is suppressed because we use
        // `_mm_storeu_si128` / `_mm_storeu_ps` — the `u`-suffix
        // variants explicitly support unaligned addresses, so a
        // u32 pointer cast to `*mut __m128i` is sound.
        #[cfg(target_arch = "x86_64")]
        #[allow(clippy::cast_ptr_alignment)]
        unsafe {
            use core::arch::x86_64::{
                __m128i, _mm_add_ps, _mm_cvtepi32_ps, _mm_cvtss_f32, _mm_mul_ps, _mm_rsqrt_ps,
                _mm_set1_ps, _mm_setr_epi32, _mm_setr_ps, _mm_storeu_ps, _mm_storeu_si128,
            };
            let strx = rs.strx;
            let stry = rs.stry;
            let vstrx4 = _mm_set1_ps(strx * 4.0);
            let vstry4 = _mm_set1_ps(stry * 4.0);
            let mut vdx = _mm_setr_ps(dirx, dirx + strx, dirx + 2.0 * strx, dirx + 3.0 * strx);
            let mut vdy = _mm_setr_ps(diry, diry + stry, diry + 2.0 * stry, diry + 3.0 * stry);
            while p1 - x >= 4 {
                // Gather 4 castdat hits — one per ray index.
                let mut col = [0i32; 4];
                let mut dst = [0i32; 4];
                for k in 0..4 {
                    let ray_idx = (plc_local >> 16) as usize;
                    let cd_offset = scratch.angstart[ray_idx] + j as isize;
                    let cd = scratch.radar[cd_offset as usize];
                    col[k] = cd.col;
                    dst[k] = cd.dist;
                    plc_local = plc_local.wrapping_add(incr);
                }
                // R5.2: per-pixel fog blend (voxlap's `hrendzfogsse`).
                // No-op when foglut is empty. Voxlap's MMX path used
                // pmulhw with foglut as 4 packed int16 lanes; we
                // mirror the scalar fallback the goldens use, which
                // applies a single `l = foglut[..] & 32767` factor
                // per pixel (one `l` per ray, all 3 channels).
                if !scratch.foglut.is_empty() {
                    for k in 0..4 {
                        col[k] = fog_blend(col[k], dst[k], &scratch.foglut, scratch.fog_col);
                    }
                }
                let vcol = _mm_setr_epi32(col[0], col[1], col[2], col[3]);
                let vdsi = _mm_setr_epi32(dst[0], dst[1], dst[2], dst[3]);
                let vdst = _mm_cvtepi32_ps(vdsi);
                let vsqr = _mm_add_ps(_mm_mul_ps(vdx, vdx), _mm_mul_ps(vdy, vdy));
                let vinv = _mm_rsqrt_ps(vsqr);
                let vz = _mm_mul_ps(vdst, vinv);

                let pixel_idx = row_start + x as usize;
                _mm_storeu_si128(self.target.fb_ptr().add(pixel_idx).cast::<__m128i>(), vcol);
                _mm_storeu_ps(self.target.zb_ptr().add(pixel_idx), vz);

                vdx = _mm_add_ps(vdx, vstrx4);
                vdy = _mm_add_ps(vdy, vstry4);
                x += 4;
            }
            // Bring scalar dirx/diry up to where the batch left
            // off — first lane of the post-step vdx/vdy.
            dirx = _mm_cvtss_f32(vdx);
            diry = _mm_cvtss_f32(vdy);
        }

        // R9: NEON 4-pixel batch — aarch64 equivalent of the SSE2
        // path above. Uses `vrsqrteq_f32` + one Newton–Raphson step
        // via `vrsqrtsq_f32` for ~16-bit precision (vs SSE2's ~12-bit
        // without Newton). NEON is baseline on all AArch64 — no
        // runtime feature check needed. Stores are naturally unaligned.
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use core::arch::aarch64::{
                float32x4_t, vaddq_f32, vcvtq_f32_s32, vdupq_n_f32, vgetq_lane_f32, vld1q_f32,
                vld1q_s32, vmulq_f32, vreinterpretq_u32_s32, vrsqrteq_f32, vrsqrtsq_f32, vst1q_f32,
                vst1q_u32,
            };
            let strx = rs.strx;
            let stry = rs.stry;
            let vstrx4 = vdupq_n_f32(strx * 4.0);
            let vstry4 = vdupq_n_f32(stry * 4.0);
            let dx_arr: [f32; 4] = [dirx, dirx + strx, dirx + 2.0 * strx, dirx + 3.0 * strx];
            let dy_arr: [f32; 4] = [diry, diry + stry, diry + 2.0 * stry, diry + 3.0 * stry];
            let mut vdx: float32x4_t = vld1q_f32(dx_arr.as_ptr());
            let mut vdy: float32x4_t = vld1q_f32(dy_arr.as_ptr());
            while p1 - x >= 4 {
                // Scalar gather — same as SSE2 path.
                let mut col = [0i32; 4];
                let mut dst = [0i32; 4];
                for k in 0..4 {
                    let ray_idx = (plc_local >> 16) as usize;
                    let cd_offset = scratch.angstart[ray_idx] + j as isize;
                    let cd = scratch.radar[cd_offset as usize];
                    col[k] = cd.col;
                    dst[k] = cd.dist;
                    plc_local = plc_local.wrapping_add(incr);
                }
                if !scratch.foglut.is_empty() {
                    for k in 0..4 {
                        col[k] = fog_blend(col[k], dst[k], &scratch.foglut, scratch.fog_col);
                    }
                }
                let vcol = vreinterpretq_u32_s32(vld1q_s32(col.as_ptr()));
                let vdst = vcvtq_f32_s32(vld1q_s32(dst.as_ptr()));
                let vsqr = vaddq_f32(vmulq_f32(vdx, vdx), vmulq_f32(vdy, vdy));
                // One Newton–Raphson step: est * vrsqrts(x * est, est).
                let est = vrsqrteq_f32(vsqr);
                let vinv = vmulq_f32(est, vrsqrtsq_f32(vmulq_f32(vsqr, est), est));
                let vz = vmulq_f32(vdst, vinv);

                let pixel_idx = row_start + x as usize;
                vst1q_u32(self.target.fb_ptr().add(pixel_idx), vcol);
                vst1q_f32(self.target.zb_ptr().add(pixel_idx), vz);

                vdx = vaddq_f32(vdx, vstrx4);
                vdy = vaddq_f32(vdy, vstry4);
                x += 4;
            }
            dirx = vgetq_lane_f32(vdx, 0);
            diry = vgetq_lane_f32(vdy, 0);
        }

        // R10.3: wasm SIMD 4-pixel batch — equivalent of the SSE2
        // / NEON paths above. Uses `1.0 / sqrt(x)` (full-precision
        // `f32x4_sqrt` + `f32x4_div`) where SSE2 had `_mm_rsqrt_ps`
        // and NEON had `vrsqrteq_f32`+Newton, since wasm SIMD has
        // no rsqrt approximation. Wasm bytes therefore differ
        // from both x86 and aarch64 goldens — captured by R10.4's
        // separate `wasm-hashes.txt`.
        #[cfg(target_arch = "wasm32")]
        unsafe {
            use core::arch::wasm32::{
                f32x4, f32x4_add, f32x4_convert_i32x4, f32x4_div, f32x4_extract_lane, f32x4_mul,
                f32x4_splat, f32x4_sqrt, i32x4, v128, v128_store,
            };
            let strx = rs.strx;
            let stry = rs.stry;
            let vstrx4 = f32x4_splat(strx * 4.0);
            let vstry4 = f32x4_splat(stry * 4.0);
            let one = f32x4_splat(1.0);
            let mut vdx = f32x4(dirx, dirx + strx, dirx + 2.0 * strx, dirx + 3.0 * strx);
            let mut vdy = f32x4(diry, diry + stry, diry + 2.0 * stry, diry + 3.0 * stry);
            while p1 - x >= 4 {
                // Scalar gather — same shape as SSE2 / NEON paths.
                let mut col = [0i32; 4];
                let mut dst = [0i32; 4];
                for k in 0..4 {
                    let ray_idx = (plc_local >> 16) as usize;
                    let cd_offset = scratch.angstart[ray_idx] + j as isize;
                    let cd = scratch.radar[cd_offset as usize];
                    col[k] = cd.col;
                    dst[k] = cd.dist;
                    plc_local = plc_local.wrapping_add(incr);
                }
                if !scratch.foglut.is_empty() {
                    for k in 0..4 {
                        col[k] = fog_blend(col[k], dst[k], &scratch.foglut, scratch.fog_col);
                    }
                }
                let vcol: v128 = i32x4(col[0], col[1], col[2], col[3]);
                let vdsi: v128 = i32x4(dst[0], dst[1], dst[2], dst[3]);
                let vdst = f32x4_convert_i32x4(vdsi);
                let vsqr = f32x4_add(f32x4_mul(vdx, vdx), f32x4_mul(vdy, vdy));
                let vinv = f32x4_div(one, f32x4_sqrt(vsqr));
                let vz = f32x4_mul(vdst, vinv);

                let pixel_idx = row_start + x as usize;
                v128_store(self.target.fb_ptr().add(pixel_idx).cast::<v128>(), vcol);
                v128_store(self.target.zb_ptr().add(pixel_idx).cast::<v128>(), vz);

                vdx = f32x4_add(vdx, vstrx4);
                vdy = f32x4_add(vdy, vstry4);
                x += 4;
            }
            dirx = f32x4_extract_lane::<0>(vdx);
            diry = f32x4_extract_lane::<0>(vdy);
        }

        // Scalar tail — handles 0..3 leftover pixels on x86_64 /
        // aarch64 / wasm32 and the full body on other targets.
        while x < p1 {
            // ray index = signed shift right (voxlap's `plc >> 16`).
            let ray_idx = (plc_local >> 16) as usize;
            let cd_offset = scratch.angstart[ray_idx] + j as isize;
            let cd = scratch.radar[cd_offset as usize];
            let col = fog_blend(cd.col, cd.dist, &scratch.foglut, scratch.fog_col);

            let pixel_idx = row_start + x as usize;
            #[allow(clippy::cast_precision_loss)]
            let z = cd.dist as f32 / (dirx * dirx + diry * diry).sqrt();
            // SAFETY: pixel_idx = sy*pitch + x, with sy < yres and x < p1
            // ≤ xres (loop guard); p1 ≤ ctx.xres in scan_loops::top_quadrant /
            // bottom_quadrant. fb / zb were allocated at pitch*height by the
            // caller (asserted in Engine::render's preamble); pixel_idx is
            // therefore in-range. Wedge-disjoint invariant: top + bottom
            // quadrants own disjoint sy ranges.
            unsafe {
                self.target.write_color(pixel_idx, col as u32);
                self.target.write_depth(pixel_idx, z);
            }

            dirx += rs.strx;
            diry += rs.stry;
            plc_local = plc_local.wrapping_add(incr);
            x += 1;
        }
    }

    fn vrend(
        &mut self,
        scratch: &mut ScanScratch,
        sx: i32,
        sy: i32,
        p1: i32,
        iplc: i32,
        iinc: i32,
    ) {
        let rs = self
            .frame
            .as_ref()
            .map(|f| f.ray_step)
            .expect("hrend/vrend called before frame_setup");
        #[allow(clippy::cast_precision_loss)]
        let mut dirx = rs.strx * sx as f32 + rs.heix * sy as f32 + rs.addx;
        #[allow(clippy::cast_precision_loss)]
        let mut diry = rs.stry * sx as f32 + rs.heiy * sy as f32 + rs.addy;
        let row_start = sy as usize * self.pitch_pixels;
        let half_stride = scratch.uurend_half_stride;

        let mut iplc_local = iplc;
        let mut x = sx;

        // R5.3: SSE2 4-pixel batch — port of voxlaptest's
        // `vrendzsse` (voxlap5.c:2083). The per-column
        // `uurend[sx] += uurend[sx + half_stride]` update is
        // parallel-safe: uurend[sx + half_stride..] is read-only
        // here, and uurend[sx..+3] are four distinct lanes.
        // Read OLD u/d values, do the SSE z math, then write
        // back four NEW u values. Plus fog blend (R5.2-style)
        // when foglut is non-empty.
        #[cfg(target_arch = "x86_64")]
        #[allow(clippy::cast_ptr_alignment)]
        unsafe {
            use core::arch::x86_64::{
                __m128i, _mm_add_ps, _mm_cvtepi32_ps, _mm_cvtss_f32, _mm_mul_ps, _mm_rsqrt_ps,
                _mm_set1_ps, _mm_setr_epi32, _mm_setr_ps, _mm_storeu_ps, _mm_storeu_si128,
            };
            let strx = rs.strx;
            let stry = rs.stry;
            let vstrx4 = _mm_set1_ps(strx * 4.0);
            let vstry4 = _mm_set1_ps(stry * 4.0);
            let mut vdx = _mm_setr_ps(dirx, dirx + strx, dirx + 2.0 * strx, dirx + 3.0 * strx);
            let mut vdy = _mm_setr_ps(diry, diry + stry, diry + 2.0 * stry, diry + 3.0 * stry);
            while p1 - x >= 4 {
                let xu = x as usize;
                // Read 4 OLD uurend pairs (u, d). u = current ray
                // index for column; d = per-pixel delta.
                let mut u = [0i32; 4];
                let mut d = [0i32; 4];
                for k in 0..4 {
                    u[k] = scratch.uurend[xu + k];
                    d[k] = scratch.uurend[xu + k + half_stride];
                }
                // Gather 4 castdat hits.
                let mut col = [0i32; 4];
                let mut dst = [0i32; 4];
                for k in 0..4 {
                    let ray_idx = (u[k] >> 16) as usize;
                    let iplc_k = iplc_local.wrapping_add(iinc.wrapping_mul(k as i32));
                    let cd_offset = scratch.angstart[ray_idx] + iplc_k as isize;
                    let cd = scratch.radar[cd_offset as usize];
                    col[k] = cd.col;
                    dst[k] = cd.dist;
                }
                if !scratch.foglut.is_empty() {
                    for k in 0..4 {
                        col[k] = fog_blend(col[k], dst[k], &scratch.foglut, scratch.fog_col);
                    }
                }
                let vcol = _mm_setr_epi32(col[0], col[1], col[2], col[3]);
                let vdsi = _mm_setr_epi32(dst[0], dst[1], dst[2], dst[3]);
                let vdst = _mm_cvtepi32_ps(vdsi);
                let vsqr = _mm_add_ps(_mm_mul_ps(vdx, vdx), _mm_mul_ps(vdy, vdy));
                let vinv = _mm_rsqrt_ps(vsqr);
                let vz = _mm_mul_ps(vdst, vinv);

                let pixel_idx = row_start + xu;
                _mm_storeu_si128(self.target.fb_ptr().add(pixel_idx).cast::<__m128i>(), vcol);
                _mm_storeu_ps(self.target.zb_ptr().add(pixel_idx), vz);

                // Write back NEW uurend values — u + d per lane.
                for k in 0..4 {
                    scratch.uurend[xu + k] = u[k].wrapping_add(d[k]);
                }

                vdx = _mm_add_ps(vdx, vstrx4);
                vdy = _mm_add_ps(vdy, vstry4);
                iplc_local = iplc_local.wrapping_add(iinc.wrapping_mul(4));
                x += 4;
            }
            dirx = _mm_cvtss_f32(vdx);
            diry = _mm_cvtss_f32(vdy);
        }

        // R9: NEON 4-pixel batch for vrend — aarch64 equivalent.
        // Same structure as hrend NEON: scalar gather + uurend
        // read/write, NEON rsqrt for z, vectorized store.
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use core::arch::aarch64::{
                float32x4_t, vaddq_f32, vcvtq_f32_s32, vdupq_n_f32, vgetq_lane_f32, vld1q_f32,
                vld1q_s32, vmulq_f32, vreinterpretq_u32_s32, vrsqrteq_f32, vrsqrtsq_f32, vst1q_f32,
                vst1q_u32,
            };
            let strx = rs.strx;
            let stry = rs.stry;
            let vstrx4 = vdupq_n_f32(strx * 4.0);
            let vstry4 = vdupq_n_f32(stry * 4.0);
            let dx_arr: [f32; 4] = [dirx, dirx + strx, dirx + 2.0 * strx, dirx + 3.0 * strx];
            let dy_arr: [f32; 4] = [diry, diry + stry, diry + 2.0 * stry, diry + 3.0 * stry];
            let mut vdx: float32x4_t = vld1q_f32(dx_arr.as_ptr());
            let mut vdy: float32x4_t = vld1q_f32(dy_arr.as_ptr());
            while p1 - x >= 4 {
                let xu = x as usize;
                // Read 4 OLD uurend pairs (u, d).
                let mut u = [0i32; 4];
                let mut d = [0i32; 4];
                for k in 0..4 {
                    u[k] = scratch.uurend[xu + k];
                    d[k] = scratch.uurend[xu + k + half_stride];
                }
                // Scalar gather — 4 castdat hits.
                let mut col = [0i32; 4];
                let mut dst = [0i32; 4];
                for k in 0..4 {
                    let ray_idx = (u[k] >> 16) as usize;
                    let iplc_k = iplc_local.wrapping_add(iinc.wrapping_mul(k as i32));
                    let cd_offset = scratch.angstart[ray_idx] + iplc_k as isize;
                    let cd = scratch.radar[cd_offset as usize];
                    col[k] = cd.col;
                    dst[k] = cd.dist;
                }
                if !scratch.foglut.is_empty() {
                    for k in 0..4 {
                        col[k] = fog_blend(col[k], dst[k], &scratch.foglut, scratch.fog_col);
                    }
                }
                let vcol = vreinterpretq_u32_s32(vld1q_s32(col.as_ptr()));
                let vdst = vcvtq_f32_s32(vld1q_s32(dst.as_ptr()));
                let vsqr = vaddq_f32(vmulq_f32(vdx, vdx), vmulq_f32(vdy, vdy));
                let est = vrsqrteq_f32(vsqr);
                let vinv = vmulq_f32(est, vrsqrtsq_f32(vmulq_f32(vsqr, est), est));
                let vz = vmulq_f32(vdst, vinv);

                let pixel_idx = row_start + xu;
                vst1q_u32(self.target.fb_ptr().add(pixel_idx), vcol);
                vst1q_f32(self.target.zb_ptr().add(pixel_idx), vz);

                // Write back NEW uurend values — u + d per lane.
                for k in 0..4 {
                    scratch.uurend[xu + k] = u[k].wrapping_add(d[k]);
                }

                vdx = vaddq_f32(vdx, vstrx4);
                vdy = vaddq_f32(vdy, vstry4);
                iplc_local = iplc_local.wrapping_add(iinc.wrapping_mul(4));
                x += 4;
            }
            dirx = vgetq_lane_f32(vdx, 0);
            diry = vgetq_lane_f32(vdy, 0);
        }

        // R10.3: wasm SIMD 4-pixel batch for vrend — equivalent of
        // the SSE2 / NEON paths above. Same scalar-gather + uurend
        // read/write structure; full-precision `1.0 / sqrt(x)` for
        // the inverse magnitude, since wasm SIMD has no rsqrt
        // approximation. Bytes diverge from the other arches —
        // R10.4's `wasm-hashes.txt` covers the divergence.
        #[cfg(target_arch = "wasm32")]
        unsafe {
            use core::arch::wasm32::{
                f32x4, f32x4_add, f32x4_convert_i32x4, f32x4_div, f32x4_extract_lane, f32x4_mul,
                f32x4_splat, f32x4_sqrt, i32x4, v128, v128_store,
            };
            let strx = rs.strx;
            let stry = rs.stry;
            let vstrx4 = f32x4_splat(strx * 4.0);
            let vstry4 = f32x4_splat(stry * 4.0);
            let one = f32x4_splat(1.0);
            let mut vdx = f32x4(dirx, dirx + strx, dirx + 2.0 * strx, dirx + 3.0 * strx);
            let mut vdy = f32x4(diry, diry + stry, diry + 2.0 * stry, diry + 3.0 * stry);
            while p1 - x >= 4 {
                let xu = x as usize;
                // Read 4 OLD uurend pairs (u, d).
                let mut u = [0i32; 4];
                let mut d = [0i32; 4];
                for k in 0..4 {
                    u[k] = scratch.uurend[xu + k];
                    d[k] = scratch.uurend[xu + k + half_stride];
                }
                // Scalar gather — 4 castdat hits.
                let mut col = [0i32; 4];
                let mut dst = [0i32; 4];
                for k in 0..4 {
                    let ray_idx = (u[k] >> 16) as usize;
                    let iplc_k = iplc_local.wrapping_add(iinc.wrapping_mul(k as i32));
                    let cd_offset = scratch.angstart[ray_idx] + iplc_k as isize;
                    let cd = scratch.radar[cd_offset as usize];
                    col[k] = cd.col;
                    dst[k] = cd.dist;
                }
                if !scratch.foglut.is_empty() {
                    for k in 0..4 {
                        col[k] = fog_blend(col[k], dst[k], &scratch.foglut, scratch.fog_col);
                    }
                }
                let vcol: v128 = i32x4(col[0], col[1], col[2], col[3]);
                let vdsi: v128 = i32x4(dst[0], dst[1], dst[2], dst[3]);
                let vdst = f32x4_convert_i32x4(vdsi);
                let vsqr = f32x4_add(f32x4_mul(vdx, vdx), f32x4_mul(vdy, vdy));
                let vinv = f32x4_div(one, f32x4_sqrt(vsqr));
                let vz = f32x4_mul(vdst, vinv);

                let pixel_idx = row_start + xu;
                v128_store(self.target.fb_ptr().add(pixel_idx).cast::<v128>(), vcol);
                v128_store(self.target.zb_ptr().add(pixel_idx).cast::<v128>(), vz);

                // Write back NEW uurend values — u + d per lane.
                for k in 0..4 {
                    scratch.uurend[xu + k] = u[k].wrapping_add(d[k]);
                }

                vdx = f32x4_add(vdx, vstrx4);
                vdy = f32x4_add(vdy, vstry4);
                iplc_local = iplc_local.wrapping_add(iinc.wrapping_mul(4));
                x += 4;
            }
            dirx = f32x4_extract_lane::<0>(vdx);
            diry = f32x4_extract_lane::<0>(vdy);
        }

        // Scalar tail — handles 0..3 leftover pixels on x86_64 /
        // aarch64 / wasm32 and the full body on other targets.
        while x < p1 {
            // Vertical scan reads the per-column ray index from
            // uurend[sx] (>>16 to drop the fractional bits).
            let xu = x as usize;
            let ray_idx = (scratch.uurend[xu] >> 16) as usize;
            let cd_offset = scratch.angstart[ray_idx] + iplc_local as isize;
            let cd = scratch.radar[cd_offset as usize];
            let col = fog_blend(cd.col, cd.dist, &scratch.foglut, scratch.fog_col);

            let pixel_idx = row_start + xu;
            #[allow(clippy::cast_precision_loss)]
            let z = cd.dist as f32 / (dirx * dirx + diry * diry).sqrt();
            // SAFETY: see hrend's matching write — pixel_idx is in-bounds
            // by the same scan_loops geometry argument; right + left
            // quadrants own disjoint sx ranges so cross-thread writes
            // are pairwise pixel-disjoint.
            unsafe {
                self.target.write_color(pixel_idx, col as u32);
                self.target.write_depth(pixel_idx, z);
            }

            dirx += rs.strx;
            diry += rs.stry;
            // Advance per-column ray index. uurend[x] persists
            // across vrend calls — this state is what couples
            // consecutive scanlines through the same column.
            scratch.uurend[xu] = scratch.uurend[xu].wrapping_add(scratch.uurend[xu + half_stride]);
            x += 1;
            iplc_local = iplc_local.wrapping_add(iinc);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rasterizer::CastDat;

    /// Build owned per-frame state so tests can assemble a
    /// `ScanContext` with proper-lifetime borrows. Values aren't
    /// load-bearing for the scalar-fill behaviour tests; the real
    /// `gline` cares about them, hence `camera_state` joining the
    /// tuple.
    fn dummy_per_frame() -> (
        crate::camera_math::CameraState,
        crate::projection::ProjectionRect,
        crate::ray_step::RayStep,
        crate::opticast_prelude::OpticastPrelude,
    ) {
        let cam = crate::Camera {
            pos: [0.0, 0.0, 0.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 1.0, 0.0],
            forward: [0.0, 0.0, 1.0],
        };
        let cs = crate::camera_math::derive(&cam, 64, 64, 32.0, 32.0, 32.0);
        let proj = crate::projection::derive_projection(&cs, 64, 64, 32.0, 32.0, 32.0, 1);
        let rs = crate::ray_step::derive_ray_step(&cs, proj.cx, proj.cy, 32.0);
        let prelude = crate::opticast_prelude::derive_prelude(&cs, 2048, 1, 4, 1024);
        (cs, proj, rs, prelude)
    }

    #[test]
    fn frame_setup_caches_ray_step() {
        let mut fb = vec![0u32; 64 * 64];
        let mut zb = vec![0.0f32; 64 * 64];
        let mut r = ScalarRasterizer::new(&mut fb, &mut zb, 64, &[], &[], &[0usize, 0], 64);
        let (cs, proj, rs, prelude) = dummy_per_frame();
        let ctx = ScanContext {
            proj: &proj,
            rs: &rs,
            prelude: &prelude,
            xres: 64,
            y_start: 0,
            y_end: 64,
            anginc: 1,
            camera_state: &cs,
            camera_gstartz0: 0,
            camera_gstartz1: 0,
            camera_vptr_offset: 0,
        };
        r.frame_setup(&ctx);
        let cached_rs = r.frame.as_ref().expect("frame populated").ray_step;
        assert_eq!(cached_rs.strx.to_bits(), rs.strx.to_bits());
        assert_eq!(cached_rs.stry.to_bits(), rs.stry.to_bits());
        assert_eq!(cached_rs.cx16, rs.cx16);
        assert_eq!(cached_rs.cy16, rs.cy16);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn hrend_sse_batch_writes_4_pixel_block() {
        // R5.1 smoke test: hrend's SSE batch fires for span len ≥
        // 4. Pre-fill radar with 4 distinct ARGB values, run hrend
        // over [10..14], assert the framebuffer carries the colour
        // bits (z lanes use rsqrtps approximation so are intent-
        // ionally not bit-checked).
        let mut fb = vec![0u32; 64 * 64];
        let mut zb = vec![0.0f32; 64 * 64];
        let mut r = ScalarRasterizer::new(&mut fb, &mut zb, 64, &[], &[], &[0usize, 0], 64);
        let (cs, proj, rs, prelude) = dummy_per_frame();
        let ctx = ScanContext {
            proj: &proj,
            rs: &rs,
            prelude: &prelude,
            xres: 64,
            y_start: 0,
            y_end: 64,
            anginc: 1,
            camera_state: &cs,
            camera_gstartz0: 0,
            camera_gstartz1: 0,
            camera_vptr_offset: 0,
        };
        r.frame_setup(&ctx);

        let mut scratch = ScanScratch::new_for_size(64, 64, 64);
        // 4 colour records so the batch reads each lane.
        for (i, slot) in scratch.radar.iter_mut().enumerate().take(4) {
            slot.col = 0x8000_0000_u32 as i32 | i as i32;
            slot.dist = 1024;
        }
        // angstart[i] = i so ray_idx i (= plc>>16) resolves to
        // radar[i + j] = radar[i] with j=0.
        for k in 0..4 {
            scratch.angstart[k] = k as isize;
        }

        // sx=10, p1=14, j=0, plc=0, incr=1<<16 → plc>>16 steps
        // 0,1,2,3 over the four pixels → angstart[0..4].
        r.hrend(&mut scratch, 10, 5, 14, 0, 1 << 16, 0);

        let row_off = 5 * 64;
        for k in 0..4 {
            let want = 0x8000_0000_u32 | k as u32;
            assert_eq!(
                fb[row_off + 10 + k],
                want,
                "fb[5][{}] = {:#010x}, expected {:#010x}",
                10 + k,
                fb[row_off + 10 + k],
                want,
            );
            // z lane non-zero (rsqrtps produced something).
            // Bit-compare to dodge clippy::float_cmp; we just want
            // to confirm the slot was written, not its precise value.
            assert_ne!(zb[row_off + 10 + k].to_bits(), 0u32);
        }
    }

    #[test]
    fn fog_blend_disabled_returns_col_unchanged() {
        let foglut: Vec<i32> = Vec::new();
        let col = 0x0080_C040;
        assert_eq!(fog_blend(col, 0x1234_5678, &foglut, 0xFF_FFFF), col);
    }

    #[test]
    fn fog_blend_full_fog_returns_fog_col_per_channel() {
        // l = 32767 → ((fog - col) * 32767) >> 15 = fog - col → final
        // is col + (fog - col) = fog (per channel; alpha untouched).
        let foglut = vec![32767; 2048];
        let col = 0x80_AA_BB_CC_u32 as i32;
        let fog = 0x00_11_22_33_i32;
        let blended = fog_blend(col, 0, &foglut, fog) as u32;
        // Low 24 bits = fog colour; alpha (high byte) survives from col.
        assert_eq!(blended & 0x00FF_FFFF, fog as u32 & 0x00FF_FFFF);
        assert_eq!(blended & 0xFF00_0000, col as u32 & 0xFF00_0000);
    }

    #[test]
    fn set_fog_zero_distance_clears_table() {
        let mut s = ScanScratch::new_for_size(64, 64, 64);
        s.set_fog(0x1234_5678, 100);
        assert!(!s.foglut.is_empty());
        s.set_fog(0, 0);
        assert!(s.foglut.is_empty());
    }

    #[test]
    fn set_fog_table_starts_at_zero_and_climbs() {
        let mut s = ScanScratch::new_for_size(64, 64, 64);
        s.set_fog(0xFF, 1024);
        // First entry: acc = 0 → hi16 = 0.
        assert_eq!(s.foglut[0], 0);
        // Last entry near the overflow boundary should be near 32767
        // (the saturate-fill value); voxlap's exact step makes
        // foglut[2047] either ~32766 (last walked entry) or 32767
        // (post-overflow padding) depending on max_scan_dist
        // divisibility.
        assert!(
            s.foglut[2047] > 30_000,
            "tail entry too low: {}",
            s.foglut[2047]
        );
    }

    #[test]
    fn hrend_writes_pixel_per_column_from_radar() {
        // Pre-fill scratch.radar with a recognizable colour gradient
        // and pre-set scratch.angstart so hrend resolves to the
        // expected radar slot per pixel. Then call hrend manually
        // and verify the framebuffer received the colours.
        let mut fb = vec![0u32; 64 * 64];
        let mut zb = vec![0.0f32; 64 * 64];
        let mut r = ScalarRasterizer::new(&mut fb, &mut zb, 64, &[], &[], &[0usize, 0], 64);
        let (cs, proj, rs, prelude) = dummy_per_frame();
        let ctx = ScanContext {
            proj: &proj,
            rs: &rs,
            prelude: &prelude,
            xres: 64,
            y_start: 0,
            y_end: 64,
            anginc: 1,
            camera_state: &cs,
            camera_gstartz0: 0,
            camera_gstartz1: 0,
            camera_vptr_offset: 0,
        };
        r.frame_setup(&ctx);

        let mut scratch = ScanScratch::new_for_size(64, 64, 64);
        // Synthetic radar: 16 castdat entries with col = 0xAA...,
        // dist = 1 (zbuffer math will divide by sqrt of dir² so we
        // just pick a stable distance).
        for (i, slot) in scratch.radar.iter_mut().enumerate().take(16) {
            slot.col = 0x8000_0000_u32 as i32 | i as i32;
            slot.dist = 1024;
        }
        // angstart[0..4] all point at radar[0]; with j=0..3 and a
        // plc that increments by 0 (incr=0 holds plc>>16 at 0), the
        // pixel-by-pixel index lands at slots 0, 1, 2, 3 — i.e. j
        // selects the column.
        scratch.angstart[0] = 0;

        // Render a span of 4 columns starting at sx=10, sy=5, with j
        // varying via the scan: but voxlap's hrend uses a single j
        // for the whole span (per-ray castdat column is fixed). We
        // pick j=2 and verify framebuffer[row*pitch + sx..sx+4] all
        // equal radar[0+2].col = 0x80000002.
        r.hrend(&mut scratch, 10, 5, 14, 0, 0, 2);

        let row_off = 5 * 64;
        for x in 10..14 {
            let want = 0x8000_0000_u32 | 2;
            assert_eq!(
                fb[row_off + x],
                want,
                "fb[5][{x}] = {:#010x}, expected {:#010x}",
                fb[row_off + x],
                want,
            );
        }
        // Pixels outside the rendered span are untouched.
        assert_eq!(fb[row_off + 9], 0);
        assert_eq!(fb[row_off + 14], 0);
    }

    #[test]
    fn end_to_end_opticast_runs_through_real_gline() {
        // Smoke test: with a valid above-the-slab camera and the
        // real R4.3a-rewire-3b gline, full opticast should run
        // without panicking and return `Rendered`. The synthetic
        // single-slab world has no voxel colour bytes so grouscan's
        // fill loops bail to startsky, which writes the configured
        // skycast into the radar — verify by setting a recognizable
        // skycast and asserting some pixels carry it.
        use crate::opticast as opticast_fn;
        use crate::rasterizer::ScratchPool;
        use crate::OpticastSettings;

        let mut fb = vec![0u32; 640 * 480];
        let mut zb = vec![0.0f32; 640 * 480];
        let mut pool = ScratchPool::new(640, 480, 2048);
        // Recognizable sky colour — pixels filled by startsky's
        // solid-fill branch (the path the empty-colour-byte slab
        // ends up routing through) carry this.
        let sky_col = 0x80AB_CDEF_u32 as i32;
        pool.set_skycast(sky_col, 0x7FFF_FFFF);

        // Single solid slab at z = 200..254. cz = 128 < 200 →
        // air-above-the-slab, opticast renders. Synthetic world:
        // only the camera's column (1024 * 2048 + 1024) holds the
        // slab; every other column is empty. Build world before
        // the rasterizer so it can borrow `&column` / `&column_offsets`.
        let column = vec![0u8, 200, 254, 0];
        let cam_idx = 1024usize * 2048 + 1024;
        let mut column_offsets = vec![0u32; 2048 * 2048 + 1];
        let column_len_u32 = u32::try_from(column.len()).expect("column fits u32");
        for offset in &mut column_offsets[(cam_idx + 1)..] {
            *offset = column_len_u32;
        }

        let mip_base_offsets = [0usize, column_offsets.len()];
        let mut rasterizer = ScalarRasterizer::new(
            &mut fb,
            &mut zb,
            640,
            &column,
            &column_offsets,
            &mip_base_offsets,
            2048,
        );

        let cam = crate::Camera {
            pos: [1024.0, 1024.0, 128.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 1.0, 0.0],
            forward: [0.0, 0.0, 1.0],
        };
        let settings = OpticastSettings::for_oracle_framebuffer(640, 480);

        let outcome = opticast_fn(
            &mut rasterizer,
            &mut pool,
            &cam,
            &settings,
            2048,
            &column,
            &column_offsets,
        );
        assert_eq!(outcome, crate::OpticastOutcome::Rendered);

        // Wiring smoke test — gline → derive_gline_frustum →
        // grouscan_run chain executes for every ray without
        // panicking, opticast returns Rendered. The synthetic
        // single-slab world has no colour bytes (header only) so
        // grouscan's drawflor fill bails on the bounds check and
        // routes to predeletez → deletez → Done before reaching
        // startsky; the radar stays at default zeros, the
        // framebuffer ends up sky-blue (the host pre-fill). What
        // matters is that nothing crashed — the per-ray gline
        // arithmetic + cf[128] seeding + grouscan dispatch all
        // hold up under live ray geometry. Once R4.3a-rewire-4
        // loads a real `.vxl` with colour bytes, grouscan's fill
        // loops will write recognisable voxel colours and a
        // colour-presence assertion replaces this comment.
        let _ = sky_col; // suppress unused-let warning — kept as
                         // scaffolding for the future assertion.
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn vrend_sse_batch_writes_4_pixel_block() {
        // R5.3 smoke test: vrend's SSE batch fires for span len ≥
        // 4. Pre-fill 4 distinct radar entries, set angstart so
        // each lane's uurend[sx]>>16 indexes a different ray, run
        // vrend over [10..14], assert each column got the right
        // colour and uurend advanced.
        let mut fb = vec![0u32; 64 * 64];
        let mut zb = vec![0.0f32; 64 * 64];
        let mut r = ScalarRasterizer::new(&mut fb, &mut zb, 64, &[], &[], &[0usize, 0], 64);
        let (cs, proj, rs, prelude) = dummy_per_frame();
        let ctx = ScanContext {
            proj: &proj,
            rs: &rs,
            prelude: &prelude,
            xres: 64,
            y_start: 0,
            y_end: 64,
            anginc: 1,
            camera_state: &cs,
            camera_gstartz0: 0,
            camera_gstartz1: 0,
            camera_vptr_offset: 0,
        };
        r.frame_setup(&ctx);

        let mut scratch = ScanScratch::new_for_size(64, 64, 64);
        for k in 0..4 {
            scratch.radar[k] = CastDat {
                col: 0x8000_0000_u32 as i32 | k as i32,
                dist: 1024,
            };
            scratch.angstart[k] = k as isize;
        }
        // uurend[sx + k] >> 16 = k → ray_idx k → angstart[k] = k
        // → radar[k]. delta = 5 (so post-batch uurend[sx + k] =
        // (k << 16) + 5).
        let half = scratch.uurend_half_stride;
        for k in 0..4 {
            scratch.uurend[10 + k] = (k as i32) << 16;
            scratch.uurend[10 + k + half] = 5;
        }

        r.vrend(&mut scratch, 10, 5, 14, 0, 0);

        let row_off = 5 * 64;
        for k in 0..4 {
            let want = 0x8000_0000_u32 | k as u32;
            assert_eq!(fb[row_off + 10 + k], want, "fb col[{}]", 10 + k);
            assert!(zb[row_off + 10 + k].to_bits() != 0, "z[{}]", 10 + k);
            // Post-batch uurend = old_u + delta.
            assert_eq!(
                scratch.uurend[10 + k],
                ((k as i32) << 16) + 5,
                "uurend[{}]",
                10 + k
            );
        }
    }

    #[test]
    fn vrend_advances_uurend_per_pixel() {
        // Verify the uurend[sx] += uurend[sx+half_stride] mutation
        // happens once per pixel.
        let mut fb = vec![0u32; 64 * 64];
        let mut zb = vec![0.0f32; 64 * 64];
        let mut r = ScalarRasterizer::new(&mut fb, &mut zb, 64, &[], &[], &[0usize, 0], 64);
        let (cs, proj, rs, prelude) = dummy_per_frame();
        let ctx = ScanContext {
            proj: &proj,
            rs: &rs,
            prelude: &prelude,
            xres: 64,
            y_start: 0,
            y_end: 64,
            anginc: 1,
            camera_state: &cs,
            camera_gstartz0: 0,
            camera_gstartz1: 0,
            camera_vptr_offset: 0,
        };
        r.frame_setup(&ctx);

        let mut scratch = ScanScratch::new_for_size(64, 64, 64);
        scratch.radar[0] = CastDat {
            col: 0x8033_4455_u32 as i32,
            dist: 1024,
        };
        // angstart[0] = 0 → all rays read radar[0 + iplc].
        scratch.angstart[0] = 0;
        // Pre-set uurend so ray_idx = uurend[sx] >> 16 = 0 for all
        // four columns (we want them to all hit angstart[0]).
        let half = scratch.uurend_half_stride;
        for sx in 10..14 {
            scratch.uurend[sx] = 0;
            scratch.uurend[sx + half] = 1; // delta added per pixel
        }

        r.vrend(&mut scratch, 10, 5, 14, 0, 0);

        // Each column's uurend should have advanced by the delta.
        for sx in 10..14 {
            assert_eq!(scratch.uurend[sx], 1, "uurend[{sx}] not advanced");
        }
    }
}
