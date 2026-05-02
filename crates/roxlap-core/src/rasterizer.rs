//! Rasterizer trait + per-frame scan scratch — the callback surface
//! the four-quadrant scan loops dispatch into. R4.3 will provide the
//! real implementation (`grouscan` for `gline`, the 4.7-scalar /
//! 4.9-SSE rasterizers for `hrend` / `vrend`); test code can plug a
//! recording stub here and exercise the scan loops without any
//! actual world data.

/// One ray-cast hit record. Voxlap calls this `castdat`
/// (`voxlap5.c:124..127`):
///
/// ```c
/// typedef struct { int32_t col, dist; } castdat;
/// ```
///
/// `col` is a Voxlap-style packed colour (`0x80RRGGBB`); `dist` is a
/// fixed-point distance to the hit slab.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CastDat {
    pub col: i32,
    pub dist: i32,
}

/// Scratch state the scan loops share between `gline` (the ray
/// caster, R4.3) and `hrend` / `vrend` (the scanline rasterizers, in
/// roxlap-core's R5 SSE-recover companion).
///
/// In voxlap C this is several globals: a static `radar` buffer,
/// `castdat *angstart[MAXXDIM*4]` pointers, `gscanptr` cursor, and
/// the `skycurlng` / `skycurdir` sky-radar bookkeeping. The Rust
/// port keeps them on a stack-allocatable struct so each render call
/// owns its own scratch and the engine doesn't have hidden mutable
/// global state.
#[derive(Debug, Clone)]
pub struct ScanScratch {
    /// All ray-cast hit records, written by `gline` calls and read
    /// indirectly by `hrend` / `vrend` via [`Self::angstart`].
    pub radar: Vec<CastDat>,
    /// Per-ray offset into [`Self::radar`] — voxlap stores this as a
    /// `castdat*` array and computes entries via `gscanptr ± p0/p1`,
    /// which can land *before* `radar[0]` (negative offset). The
    /// scanline rasterizers add a per-pixel `plc` value on top before
    /// the actual deref, and that combination is always in-range. We
    /// keep the raw signed offset here to mirror voxlap's pointer
    /// arithmetic exactly.
    pub angstart: Vec<isize>,
    /// Cursor into [`Self::radar`] for the next-to-be-written hit
    /// record. Reset to 0 at the start of each quadrant scan.
    pub gscanptr: usize,
    /// Sky-radar bookkeeping cursor (`skycurlng` in voxlap). `-1`
    /// when no sky pixel has been emitted yet.
    pub sky_cur_lng: i32,
    /// `+1` or `-1` — the sign of `-giforzsgn` that voxlap stamps on
    /// each new quadrant entry. The scan loops will set this; for
    /// now [`ScanScratch::new_for_size`] just initialises to `0`.
    pub sky_cur_dir: i32,
    /// Voxlap's `skyoff` — current row's pixel-index offset into the
    /// sky texture (= `sky_cur_lng * (sky.xsiz + 1)`, computed in
    /// gline's per-ray frustum prep). `0` means the textured-sky
    /// path is inactive — `phase_startsky` falls back to solid-fill
    /// `skycast`. Set by gline when an [`crate::sky::Sky`] is
    /// loaded; reset to 0 each quadrant.
    pub sky_off: i32,
    /// Per-screen-row x-boundary, voxlap's
    /// `int32_t lastx[max(MAXYDIM, VSID)]`. The right / left
    /// quadrants populate this during their pass-2 column walk; the
    /// `vrend` dispatch pass then reads `lastx[sy]` per row to know
    /// where each vertical slice begins.
    pub lastx: Vec<i32>,
    /// Per-screen-column ray-index pair, voxlap's
    /// `int32_t uurendmem[MAXXDIM*2 + 9]` viewed as
    /// `[uurend[sx], uurend[sx + MAXXDIM]]`. The right / left
    /// quadrants stamp `uurend[sx] = u` and
    /// `uurend[sx + MAXXDIM] = ui` per column for the vertical
    /// rasterizer to consume.
    pub uurend: Vec<i32>,
    /// Stride between the `uurend[sx]` half and the
    /// `uurend[sx + half_stride]` half. Equals `MAXXDIM` in voxlap;
    /// our port sizes the buffer exactly to the framebuffer width
    /// rounded up.
    pub uurend_half_stride: usize,

    // ---------------------------------------------------------------
    // grouscan (R4.3c+) per-ray state. Voxlap keeps these as globals;
    // we group them on ScanScratch so each render call owns them and
    // there is no hidden mutable global state.
    // ---------------------------------------------------------------
    /// `cf[129]` — voxlap's cfasm scratch. The seed slot at index
    /// `CF_SEED_INDEX` (= 128) is filled by `gline` before each ray;
    /// grouscan pops / pushes from there.
    pub cf: Vec<crate::grouscan::CfType>,
    /// `gpz[2]` — distance to next voxel-grid line per axis,
    /// `PREC`-scaled. Set by gline per ray; grouscan walks it.
    pub gpz: [i32; 2],
    /// `gdz[2]` — per-column-step delta added to `gpz` after a
    /// column advance. Constant per ray. Set by gline.
    pub gdz: [i32; 2],
    /// `gixy[2]` — voxel-column step in the ray's direction
    /// (`±1` along x, `±vsid` along y). Set by gline.
    pub gixy: [i32; 2],
    /// `gxmax` — scan-distance ceiling, `PREC`-scaled. Set by gline
    /// per ray (clipped against viewport edges and `gmaxscandist`).
    pub gxmax: i32,
    /// `gi0` — voxlap's per-pixel x step coefficient written by
    /// gline; consumed by grouscan's column advance.
    pub gi0: i32,
    /// `gi1` — voxlap's per-pixel y step coefficient.
    pub gi1: i32,

    /// Voxlap's `skycast` — the `(col, dist)` pair grouscan's
    /// startsky writes into every remaining radar slot when the
    /// solid-sky branch fires (textured sky is R4.4 work). The
    /// engine sets it via [`Self::set_skycast`] before invoking
    /// the rasterizer; default is opaque black at far depth.
    pub skycast: CastDat,
    /// Fog colour (packed ARGB; the alpha byte isn't used by the
    /// per-channel blend). Set by [`Self::set_fog`].
    pub fog_col: i32,
    /// Fog distance falloff table. Empty = fog disabled (voxlap's
    /// `ofogdist < 0`). Otherwise `foglut[dist >> 20] & 32767`
    /// gives the per-pixel blend factor (0 = no fog applied,
    /// 32767 = full fog colour). Built by [`Self::set_fog`].
    pub foglut: Vec<i32>,
    /// Voxlap's `gcsub[9]` per-side shading table. Default pattern
    /// is `0x00ff00ff00ff00ff` per entry — that's voxlap's
    /// `setsideshades(0,0,0,0,0,0)` baseline (no per-side
    /// darkening). The high byte of entries 2..7 is the per-side
    /// intensity (top, bottom, left, right, up, down). Set via
    /// [`Self::set_side_shades`]; rasterizers read it on every gline
    /// call.
    pub gcsub: [i64; 9],
    /// Voxlap's `vx5.sideshademode` flag — derived by
    /// [`Self::set_side_shades`]: false when all six args are zero
    /// (oracle baseline, swap is dead), true otherwise. When true,
    /// the per-ray gline body picks `gcsub[0]`/`gcsub[1]` from
    /// `gcsub[4..7]` based on the sign of `gixy[0]`/`gixy[1]` so
    /// wall faces get directional darkening (voxlap5.c:1230-1234).
    pub sideshademode: bool,
}

impl ScanScratch {
    /// Allocate a scratch buffer sized for an `xres × yres`
    /// framebuffer. Voxlap's per-frame `radar` buffer is
    /// `MAXXDIM * 6 * 256` `castdat` entries (`voxlap5.c:206`-area
    /// declaration); over-provisioned by `xres * 6 * 256` here until
    /// R4.1f3 nails down the exact upper bound. `uurend` /
    /// `lastx` are sized to fit `xres` / `max(yres, vsid)` entries
    /// respectively (R4.1f4b consumers).
    #[must_use]
    pub fn new_for_size(xres: u32, yres: u32, vsid: u32) -> Self {
        let radar_cap = (xres as usize) * 6 * 256;
        let angstart_cap = (xres as usize) * 4;
        let half_stride = xres as usize;
        let lastx_cap = std::cmp::max(yres, vsid) as usize;
        Self {
            radar: vec![CastDat::default(); radar_cap],
            angstart: vec![0isize; angstart_cap],
            gscanptr: 0,
            sky_cur_lng: -1,
            sky_cur_dir: 0,
            sky_off: 0,
            lastx: vec![0i32; lastx_cap],
            uurend: vec![0i32; half_stride * 2],
            uurend_half_stride: half_stride,
            cf: vec![crate::grouscan::CfType::default(); crate::grouscan::CF_LEN],
            gpz: [0; 2],
            gdz: [0; 2],
            gixy: [0; 2],
            gxmax: 0,
            gi0: 0,
            gi1: 0,
            skycast: CastDat::default(),
            fog_col: 0,
            foglut: Vec::new(),
            gcsub: [0x00ff_00ff_00ff_00ff; 9],
            sideshademode: false,
        }
    }

    /// Engine-side setter for per-side shading intensities, mirror
    /// of voxlap's `setsideshades(top, bot, left, right, up, down)`
    /// (`voxlap5.c`). Each `i8` parameter is the high byte stamped
    /// onto `gcsub[2..7]`; the low 7 bytes keep the
    /// `0x00ff00ff00ff00ff` saturate-zero pattern. Pass `(0,…,0)` to
    /// disable shading (the default), or moderate positive values
    /// (15..31) for visible side darkening like voxlap's classic
    /// games use.
    pub fn set_side_shades(&mut self, top: i8, bot: i8, left: i8, right: i8, up: i8, down: i8) {
        // High byte of an i64 (LE) is byte 7. Voxlap writes
        // `((char *)&gcsub[k])[7] = sxx;` directly — bit-equivalent
        // to reinterpreting the i8 as u8 and stamping it into the
        // top byte. The sign-loss `as u8` is intentional (mirrors
        // the C cast).
        let pack = |intensity: i8| -> i64 {
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_wrap)]
            let high = u64::from(intensity as u8) << 56;
            #[allow(clippy::cast_possible_wrap)]
            {
                (high | 0x00ff_00ff_00ff_00ff_u64) as i64
            }
        };
        self.gcsub[0] = 0x00ff_00ff_00ff_00ff;
        self.gcsub[1] = 0x00ff_00ff_00ff_00ff;
        self.gcsub[2] = pack(top);
        self.gcsub[3] = pack(bot);
        self.gcsub[4] = pack(left);
        self.gcsub[5] = pack(right);
        self.gcsub[6] = pack(up);
        self.gcsub[7] = pack(down);
        self.gcsub[8] = 0x00ff_00ff_00ff_00ff;
        // Voxlap5.c:2535-2540: `if (!(sto|sbo|sle|sri|sup|sdo))` →
        // sideshademode = 0; else sideshademode = 1. Any non-zero
        // arg flips on the per-ray gcsub[0]/[1] swap in gline.
        self.sideshademode =
            top != 0 || bot != 0 || left != 0 || right != 0 || up != 0 || down != 0;
    }

    /// Engine-side setter for the sky `(col, dist)` pair. Engine
    /// owns `Engine::sky_color`; this is the wire it writes to so
    /// `grouscan`'s startsky has the right value at fill time.
    pub fn set_skycast(&mut self, col: i32, dist: i32) {
        self.skycast = CastDat { col, dist };
    }

    /// Engine-side setter for fog. Voxlap5.c:11151-11185.
    /// `max_scan_dist <= 0` disables fog (clears the table).
    /// Otherwise builds the 2048-entry fog falloff table:
    /// `foglut[k] = (acc >> 16) & 32767` where `acc` accumulates
    /// `step = i32::MAX / max_scan_dist` per entry. After the
    /// accumulation overflows, remaining entries pad with `32767`
    /// (full fog).
    //
    // The C version stores per-entry as a 4-lane packed `int64`
    // (`hi16` repeated four times) for the asm's MMX path. The
    // scalar fallback only reads the low 15 bits; we mirror the
    // scalar form, storing `i32` per entry.
    pub fn set_fog(&mut self, col: i32, max_scan_dist: i32) {
        if max_scan_dist <= 0 {
            self.fog_col = 0;
            self.foglut.clear();
            return;
        }
        self.fog_col = col;
        // Pad with full-fog (32767) so OOB / past-overflow entries
        // saturate at maximum fog.
        self.foglut = vec![32767; 2048];
        let step = i32::MAX / max_scan_dist;
        let mut acc: i32 = 0;
        for entry in self.foglut.iter_mut().take(2048) {
            let Some(next) = acc.checked_add(step) else {
                break;
            };
            // hi16 = (acc >> 16) treated as u16 then widened — for
            // acc in [0, i32::MAX), hi16 is in [0, 32767].
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_wrap)]
            let hi16 = ((acc as u32) >> 16) as i32;
            *entry = hi16;
            acc = next;
        }
    }

    /// Reset cursors at the start of a new quadrant scan.
    pub fn reset_for_quadrant(&mut self, sky_cur_dir: i32) {
        self.gscanptr = 0;
        self.sky_cur_lng = -1;
        self.sky_cur_dir = sky_cur_dir;
        // sky_off stays 0 until gline updates it; the textured-sky
        // path checks `sky_off != 0` to decide whether to texture-
        // fill, so resetting here keeps the first-ray-of-quadrant
        // path honest.
        self.sky_off = 0;
    }
}

/// Callback surface for the column-scan loop dispatch.
///
/// - `gline` is voxlap's `gline` (R4.3 = grouscan): casts a ray of
///   `length` cells from `(x0, y0)` to `(x1, y1)` in screen space,
///   writing hit records into `scratch.radar` starting at
///   `scratch.gscanptr`.
/// - `hrend` is the horizontal-scan rasterizer (`hrendzsse` etc.):
///   given a row `sy` and column range `sx..p1`, looks up the right
///   `angstart` entries in `scratch` and writes a band of pixels.
/// - `vrend` is the vertical-scan rasterizer (`vrendzsse` etc.).
///
/// Test code can implement a recording stub that just remembers the
/// arguments — useful for verifying the scan loops dispatch the right
/// calls without involving any rasterization.
//
// Voxlap's hrend / vrend / gline take 6-8 positional arguments each.
// Boxing them in a struct would just add noise — the names match
// voxlap's parameter names so the trait body stays one-to-one with
// the C source it's tracking.
#[allow(clippy::too_many_arguments)]
pub trait Rasterizer {
    /// Called once per frame, before the four-quadrant scan loops
    /// run, with the per-frame derived state. Concrete rasterizers
    /// override this to cache whatever they need from the
    /// projection / ray-step / prelude triple (a default-noop stub
    /// is fine for recording / counting test rasterizers).
    fn frame_setup(&mut self, _ctx: &crate::scan_loops::ScanContext<'_>) {}

    fn gline(&mut self, scratch: &mut ScanScratch, length: u32, x0: f32, y0: f32, x1: f32, y1: f32);

    fn hrend(
        &mut self,
        scratch: &mut ScanScratch,
        sx: i32,
        sy: i32,
        p1: i32,
        plc: i32,
        incr: i32,
        j: i32,
    );

    fn vrend(&mut self, scratch: &mut ScanScratch, sx: i32, sy: i32, p1: i32, iplc: i32, iinc: i32);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal Rasterizer that records every call into a flat list
    /// for the per-quadrant scan-loop tests R4.1f3+ will land.
    #[derive(Debug, Default)]
    struct RecordingRasterizer {
        events: Vec<&'static str>,
    }

    impl Rasterizer for RecordingRasterizer {
        fn gline(&mut self, _: &mut ScanScratch, _: u32, _: f32, _: f32, _: f32, _: f32) {
            self.events.push("gline");
        }
        fn hrend(&mut self, _: &mut ScanScratch, _: i32, _: i32, _: i32, _: i32, _: i32, _: i32) {
            self.events.push("hrend");
        }
        fn vrend(&mut self, _: &mut ScanScratch, _: i32, _: i32, _: i32, _: i32, _: i32) {
            self.events.push("vrend");
        }
    }

    #[test]
    fn scratch_initial_state() {
        let s = ScanScratch::new_for_size(640, 480, 2048);
        assert_eq!(s.gscanptr, 0);
        assert_eq!(s.sky_cur_lng, -1);
        assert_eq!(s.sky_cur_dir, 0);
        assert!(!s.radar.is_empty());
        assert!(!s.angstart.is_empty());
    }

    #[test]
    fn scratch_reset_for_quadrant_keeps_buffers() {
        let mut s = ScanScratch::new_for_size(640, 480, 2048);
        let radar_cap = s.radar.len();
        let angstart_cap = s.angstart.len();
        // Pretend the previous quadrant filled in some scratch.
        s.gscanptr = 12345;
        s.sky_cur_lng = 7;
        s.reset_for_quadrant(-1);
        assert_eq!(s.gscanptr, 0);
        assert_eq!(s.sky_cur_lng, -1);
        assert_eq!(s.sky_cur_dir, -1);
        // Buffers are not reallocated.
        assert_eq!(s.radar.len(), radar_cap);
        assert_eq!(s.angstart.len(), angstart_cap);
    }

    #[test]
    fn set_side_shades_zero_keeps_mode_off() {
        // Voxlap5.c:2535-2540: all-zero args ⇒ sideshademode = 0 and
        // gcsub[0]/[1] high byte zeroed. Roxlap re-stamps the whole
        // i64 with the `0x00ff_00ff_00ff_00ff` baseline so cs[7] = 0.
        let mut s = ScanScratch::new_for_size(64, 64, 64);
        s.set_side_shades(0, 0, 0, 0, 0, 0);
        assert!(!s.sideshademode);
        assert_eq!(s.gcsub[0], 0x00ff_00ff_00ff_00ff);
        assert_eq!(s.gcsub[1], 0x00ff_00ff_00ff_00ff);
    }

    #[test]
    fn set_side_shades_nonzero_flips_mode_on() {
        // Any non-zero arg ⇒ sideshademode = 1 (voxlap5.c:2540). The
        // gline body's per-ray swap reads this flag.
        let mut s = ScanScratch::new_for_size(64, 64, 64);
        s.set_side_shades(15, 15, 15, 15, 15, 15);
        assert!(s.sideshademode);
        // Lanes 4..7 carry the per-side intensity in their high byte.
        assert_eq!((s.gcsub[4] >> 56) & 0xff, 15);
        assert_eq!((s.gcsub[7] >> 56) & 0xff, 15);
        // Lanes 0/1 stay at the baseline; the per-ray swap in gline
        // populates them from 4..7 based on gixy sign.
        assert_eq!(s.gcsub[0], 0x00ff_00ff_00ff_00ff);
        assert_eq!(s.gcsub[1], 0x00ff_00ff_00ff_00ff);
    }

    #[test]
    fn set_side_shades_one_arg_nonzero_flips_mode_on() {
        // Voxlap derives the flag from `sto|sbo|sle|sri|sup|sdo`;
        // a single non-zero arg is enough.
        let mut s = ScanScratch::new_for_size(64, 64, 64);
        s.set_side_shades(0, 0, 0, 0, 0, 1);
        assert!(s.sideshademode);
    }

    #[test]
    fn rasterizer_trait_object_dispatch() {
        // Confirms the trait object surface is callable — the scan
        // loops in R4.1f3+ will hold &mut dyn Rasterizer.
        let mut rec = RecordingRasterizer::default();
        let mut scratch = ScanScratch::new_for_size(64, 64, 64);
        let r: &mut dyn Rasterizer = &mut rec;
        r.gline(&mut scratch, 4, 0.0, 0.0, 1.0, 1.0);
        r.hrend(&mut scratch, 0, 0, 10, 0, 1, 0);
        r.vrend(&mut scratch, 0, 0, 10, 0, 1);
        assert_eq!(rec.events, ["gline", "hrend", "vrend"]);
    }
}
