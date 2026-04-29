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
    /// Per-ray index into [`Self::radar`] — what ray `i` should
    /// dereference for its starting `castdat`. Voxlap stores this as
    /// a `castdat*` table; integer indices are the natural Rust
    /// translation given Rust's lack of pointer arithmetic.
    pub angstart: Vec<usize>,
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
}

impl ScanScratch {
    /// Allocate a scratch buffer sized for `xres` columns. Voxlap's
    /// per-frame `radar` buffer is `MAXXDIM * 6 * 256` `castdat`
    /// entries (`voxlap5.c:206`-area declaration); for now we
    /// over-provision by `xres * something` until R4.1f3 nails down
    /// the exact upper bound the scan loops require.
    #[must_use]
    pub fn new_for_size(xres: u32) -> Self {
        let radar_cap = (xres as usize) * 6 * 256;
        let angstart_cap = (xres as usize) * 4;
        Self {
            radar: vec![CastDat::default(); radar_cap],
            angstart: vec![0usize; angstart_cap],
            gscanptr: 0,
            sky_cur_lng: -1,
            sky_cur_dir: 0,
        }
    }

    /// Reset cursors at the start of a new quadrant scan.
    pub fn reset_for_quadrant(&mut self, sky_cur_dir: i32) {
        self.gscanptr = 0;
        self.sky_cur_lng = -1;
        self.sky_cur_dir = sky_cur_dir;
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
    fn gline(&mut self, scratch: &mut ScanScratch, length: u32, x0: f32, y0: f32, x1: f32, y1: f32);

    fn hrend(
        &mut self,
        scratch: &ScanScratch,
        sx: i32,
        sy: i32,
        p1: i32,
        plc: i32,
        incr: i32,
        j: i32,
    );

    fn vrend(&mut self, scratch: &ScanScratch, sx: i32, sy: i32, p1: i32, iplc: i32, iinc: i32);
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
        fn hrend(&mut self, _: &ScanScratch, _: i32, _: i32, _: i32, _: i32, _: i32, _: i32) {
            self.events.push("hrend");
        }
        fn vrend(&mut self, _: &ScanScratch, _: i32, _: i32, _: i32, _: i32, _: i32) {
            self.events.push("vrend");
        }
    }

    #[test]
    fn scratch_initial_state() {
        let s = ScanScratch::new_for_size(640);
        assert_eq!(s.gscanptr, 0);
        assert_eq!(s.sky_cur_lng, -1);
        assert_eq!(s.sky_cur_dir, 0);
        assert!(!s.radar.is_empty());
        assert!(!s.angstart.is_empty());
    }

    #[test]
    fn scratch_reset_for_quadrant_keeps_buffers() {
        let mut s = ScanScratch::new_for_size(640);
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
    fn rasterizer_trait_object_dispatch() {
        // Confirms the trait object surface is callable — the scan
        // loops in R4.1f3+ will hold &mut dyn Rasterizer.
        let mut rec = RecordingRasterizer::default();
        let mut scratch = ScanScratch::new_for_size(64);
        let r: &mut dyn Rasterizer = &mut rec;
        r.gline(&mut scratch, 4, 0.0, 0.0, 1.0, 1.0);
        r.hrend(&scratch, 0, 0, 10, 0, 1, 0);
        r.vrend(&scratch, 0, 0, 10, 0, 1);
        assert_eq!(rec.events, ["gline", "hrend", "vrend"]);
    }
}
