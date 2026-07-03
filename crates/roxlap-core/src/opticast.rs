//! Per-frame render settings shared by the CPU renderer.
//!
//! Historically this module also held voxlap's `opticast` orchestrator
//! (the four-quadrant 2.5D scan loops). That renderer was replaced by
//! the per-pixel 3D-DDA renderer in [`crate::dda`] and removed; what
//! remains is [`OpticastSettings`], the framebuffer + projection +
//! scan-distance bundle both the DDA renderer and the sprite raycaster
//! still consume. (The name is kept for API stability; "opticast"
//! survives only as a label for the settings struct.)

/// Per-frame settings the renderer forwards through its passes. Most
/// fields map onto a classic raycaster control: `(hx, hy, hz)` are the
/// pinhole projection centre + focal length (the voxlap `setcamera`
/// `dahx`/`dahy`/`dahz`), `mip_*` drive the distance-mip ladder, and
/// `max_scan_dist` bounds the ray.
///
/// `y_start..y_end` is the strip-render iteration bound. Default is the
/// full framebuffer (`0..yres`). Tile / strip callers set a sub-range
/// to render only that horizontal strip; the projection centre stays in
/// absolute screen coords, only the viewport edges shrink.
#[derive(Debug, Clone, Copy)]
pub struct OpticastSettings {
    pub xres: u32,
    pub yres: u32,
    /// First y-row this render call covers (inclusive). `0` for
    /// full-frame.
    pub y_start: u32,
    /// One past the last y-row (exclusive). `yres` for full-frame.
    pub y_end: u32,
    /// PF.13 (C7) — first x-column covered (inclusive). `0` for
    /// full-frame. The DDA renderer's pixels are fully independent, so
    /// an x sub-range is as safe as the y strip (the historical
    /// full-width-only constraint came from the deleted voxlap radar's
    /// column-indexed `angstart`).
    pub x_start: u32,
    /// One past the last x-column (exclusive). `xres` for full-frame.
    pub x_end: u32,
    pub hx: f32,
    pub hy: f32,
    pub hz: f32,
    pub anginc: f32,
    pub mip_levels: u32,
    pub mip_scan_dist: i32,
    pub max_scan_dist: i32,
}

impl OpticastSettings {
    /// Default settings for a `width × height` framebuffer with the
    /// convention `(hx, hy, hz) = (w/2, h/2, w/2)` and `anginc = 1`.
    /// Renders the full frame (`y_start = 0, y_end = height`).
    //
    // `width` / `height` cast to f32 is bounded by realistic screen
    // sizes (≤ 16M, well within f32's 24-bit mantissa).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn for_oracle_framebuffer(width: u32, height: u32) -> Self {
        let half_w = (width as f32) * 0.5;
        let half_h = (height as f32) * 0.5;
        Self {
            xres: width,
            yres: height,
            y_start: 0,
            y_end: height,
            x_start: 0,
            x_end: width,
            hx: half_w,
            hy: half_h,
            hz: half_w,
            anginc: 1.0,
            mip_levels: 1,
            mip_scan_dist: 4,
            max_scan_dist: 1024,
        }
    }

    /// Pick an explicit vertical field of view (radians): sets the
    /// focal length `hz = (yres/2) / tan(fov_y/2)` so both renderer
    /// backends show exactly this FOV (QE.2a — the facade derives the
    /// GPU projection from these settings too). The projection centre
    /// (`hx`, `hy`) is untouched. The default
    /// [`Self::for_oracle_framebuffer`] focal (`hz = w/2`) equals
    /// `with_fov_y(2·atan(h/w))` — ≈ 73.7° for 4:3, ≈ 58.7° for 16:9.
    #[must_use]
    pub fn with_fov_y(mut self, fov_y_rad: f32) -> Self {
        // yres → f32 is exact for realistic screen sizes.
        #[allow(clippy::cast_precision_loss)]
        let half_h = (self.yres as f32) * 0.5;
        self.hz = half_h / (fov_y_rad * 0.5).tan();
        self
    }

    /// Restrict this settings struct to the `[y_start, y_end)`
    /// horizontal strip. Used by the per-strip parallel dispatch — each
    /// strip clones the base settings and clamps the y-range. Caller is
    /// responsible for ensuring `y_start < y_end <= yres`.
    #[must_use]
    pub fn with_y_range(mut self, y_start: u32, y_end: u32) -> Self {
        self.y_start = y_start;
        self.y_end = y_end;
        self
    }

    /// PF.13 (C7) — restrict to the `[x_start, x_end)` vertical band;
    /// the per-grid screen scissor pairs this with
    /// [`Self::with_y_range`] so a small grid renders only its true
    /// screen rect instead of full-width rows. Caller ensures
    /// `x_start < x_end <= xres`.
    #[must_use]
    pub fn with_x_range(mut self, x_start: u32, x_end: u32) -> Self {
        self.x_start = x_start;
        self.x_end = x_end;
        self
    }
}
