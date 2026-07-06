//! The [`Engine`] is the public façade of `roxlap-core`. R3 ships
//! the API surface with a sky-fill stub; R4 wires in the real
//! opticast + grouscan rasterizer behind the same call signatures.

use crate::sky::Sky;
use crate::Camera;

/// Voxlap's `vx5.kv6col` default — mid-grey, equal R/G/B so the
/// sprite `update_reflects` nolighta optimisation kicks in.
pub const DEFAULT_KV6COL: u32 = 0x0080_8080;

/// One point light source for sprite (and, eventually, world) lighting.
/// Mirror of voxlap's `lightsrc_t` (`voxlap5.h`): position, squared
/// reach radius, and intensity scale. The lighting math reads `r2`
/// not `r`, matching voxlap's `vx5.lightsrc[i].r2`-keyed range
/// check.
#[derive(Debug, Clone, Copy)]
pub struct LightSrc {
    /// World-space position.
    pub pos: [f32; 3],
    /// Squared influence radius. Voxels / sprites further than
    /// `sqrt(r2)` from `pos` get no contribution.
    pub r2: f32,
    /// Intensity scale — voxlap's `lightsrc_t::sc`. Larger = brighter.
    pub sc: f32,
}

/// Voxel engine state.
#[derive(Debug, Clone)]
pub struct Engine {
    camera: Camera,
    sky_color: u32,
    fog_color: u32,
    /// Maximum distance the fog blend interpolates over (PREC-
    /// scaled cells; voxlap's `vx5.maxscandist`). `0` disables fog.
    fog_max_scan_dist: i32,
    /// Per-side darkening intensities — voxlap's
    /// `setsideshades(top, bot, left, right, up, down)`. Default is
    /// `[0; 6]` (no shading), matching the oracle. `ScanScratch`
    /// rebuilds its `gcsub` table from these per frame.
    side_shades: [i8; 6],
    /// Sprite material colour — voxlap's `vx5.kv6col`. Default
    /// `0x80_8080` (mid grey, R==G==B → triggers `update_reflects`'s
    /// nolighta fast path).
    kv6col: u32,
    /// Sprite lighting mode — voxlap's `vx5.lightmode`. 0 / 1 →
    /// directional surface tint (the cheap nolighta / nolightb
    /// path); 2 → per-light point-source modulation against
    /// [`Engine::lights`].
    lightmode: u32,
    /// Active point lights. Voxlap's `vx5.lightsrc[]`/`vx5.numlights`.
    /// Read by sprite `update_reflects` (and, when world voxel
    /// lighting lands, by `updatelighting`).
    lights: Vec<LightSrc>,
    /// Sky texture for the textured-`startsky` path. `None` ⇒
    /// `phase_startsky` solid-fills with `skycast` (cheap default;
    /// every oracle pose stays here so its hashes are byte-stable
    /// independent of any sky a host loads). `Some(sky)` ⇒ the
    /// rasterizer walks `sky.lng` per ray + samples
    /// `sky.pixels` per pixel, à la voxlap's `loadsky` path.
    sky: Option<Sky>,
}

impl Default for Engine {
    fn default() -> Self {
        Self {
            camera: Camera::default(),
            // Voxlap-style packed sky blue: brightness bit | 0x87ceeb.
            sky_color: 0x8087_ceeb,
            fog_color: 0,
            fog_max_scan_dist: 0,
            side_shades: [0; 6],
            kv6col: DEFAULT_KV6COL,
            lightmode: 0,
            lights: Vec::new(),
            sky: None,
        }
    }
}

impl Engine {
    /// Construct a new [`Engine`] with default state — voxlap-blue
    /// sky, no fog, no per-side shading, default kv6 colour, no
    /// lights, no sky texture.
    ///
    /// # Examples
    ///
    /// ```
    /// use roxlap_core::Engine;
    ///
    /// let mut engine = Engine::new();
    /// engine.set_sky_color(0x80aa_ddff);
    /// assert_eq!(engine.sky_color(), 0x80aa_ddff);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the camera pose (position + right/down/forward basis) the
    /// next [`render`](Self::render) uses.
    pub fn set_camera(&mut self, camera: Camera) {
        self.camera = camera;
    }

    /// The current camera pose (a copy — mutate via
    /// [`set_camera`](Self::set_camera)).
    #[must_use]
    pub fn camera(&self) -> Camera {
        self.camera
    }

    /// Override the sky / background colour. Bytes are `0xAARRGGBB`
    /// where `AA` is the voxlap-style "brightness" channel (`0x80` is
    /// "normal" intensity, matching the engine's other surface
    /// colours).
    pub fn set_sky_color(&mut self, color: u32) {
        self.sky_color = color;
    }

    /// The current sky / background colour, `0xAARRGGBB` with the
    /// voxlap-style brightness byte (see
    /// [`set_sky_color`](Self::set_sky_color)).
    #[must_use]
    pub fn sky_color(&self) -> u32 {
        self.sky_color
    }

    /// Configure fog. `max_scan_dist <= 0` disables fog. Otherwise
    /// pixels at the maximum scan distance blend fully to
    /// `fog_color` (low 24 bits — alpha byte ignored). Voxlap's
    /// `vx5.maxscandist`-based foglut is rebuilt downstream by
    /// `ScanScratch::set_fog`.
    pub fn set_fog(&mut self, color: u32, max_scan_dist: i32) {
        self.fog_color = color;
        self.fog_max_scan_dist = max_scan_dist.max(0);
    }

    /// The fog colour set by [`set_fog`](Self::set_fog). Only the low
    /// 24 RGB bits are blended toward; the alpha byte is ignored.
    #[must_use]
    pub fn fog_color(&self) -> u32 {
        self.fog_color
    }

    /// Distance (PREC-scaled cells, voxlap's `vx5.maxscandist`) at
    /// which fog reaches full opacity. `0` ⇒ fog disabled.
    #[must_use]
    pub fn fog_max_scan_dist(&self) -> i32 {
        self.fog_max_scan_dist
    }

    /// Voxlap's `setsideshades(top, bot, left, right, up, down)`
    /// — per-side voxel darkening intensities. Each `i8` is stamped
    /// onto the high byte of `gcsub[2..7]` (downstream by
    /// `ScanScratch::set_side_shades`). Pass `(0,…,0)` to disable
    /// (the oracle baseline); positive values like 15 / 31 give the
    /// directional darkening typical of voxlap's classic games.
    pub fn set_side_shades(&mut self, top: i8, bot: i8, left: i8, right: i8, up: i8, down: i8) {
        self.side_shades = [top, bot, left, right, up, down];
    }

    /// The per-side darkening intensities in
    /// `[top, bot, left, right, up, down]` order (see
    /// [`set_side_shades`](Self::set_side_shades)). All-zero ⇒ shading
    /// off.
    #[must_use]
    pub fn side_shades(&self) -> [i8; 6] {
        self.side_shades
    }

    /// Sprite material colour — packed BGRA bytes, voxlap's
    /// `vx5.kv6col`. R/G/B equal triggers `update_reflects`'s
    /// nolighta fast path.
    pub fn set_kv6col(&mut self, color: u32) {
        self.kv6col = color;
    }

    /// The current sprite material colour — voxlap's `vx5.kv6col`
    /// (see [`set_kv6col`](Self::set_kv6col)). Defaults to
    /// [`DEFAULT_KV6COL`].
    #[must_use]
    pub fn kv6col(&self) -> u32 {
        self.kv6col
    }

    /// Sprite lighting mode — voxlap's `vx5.lightmode`. 0 / 1 →
    /// directional tint; 2 → point-light shading from
    /// [`Engine::lights`]. Other values clamp to 2 in voxlap.
    pub fn set_lightmode(&mut self, mode: u32) {
        self.lightmode = mode;
    }

    /// The current sprite lighting mode — voxlap's `vx5.lightmode`
    /// (see [`set_lightmode`](Self::set_lightmode) for the values).
    #[must_use]
    pub fn lightmode(&self) -> u32 {
        self.lightmode
    }

    /// Append a light source. No upper bound enforced here —
    /// voxlap's `MAXLIGHTS` (16) is the practical limit, but the
    /// rendering math just iterates whatever's in the slice.
    pub fn add_light(&mut self, light: LightSrc) {
        self.lights.push(light);
    }

    /// Remove every light source — voxlap's `vx5.numlights = 0`.
    /// Typical per-frame pattern: clear, then re-[`add_light`]
    /// whatever is active this frame.
    ///
    /// [`add_light`]: Self::add_light
    pub fn clear_lights(&mut self) {
        self.lights.clear();
    }

    /// The active point lights, in [`add_light`](Self::add_light)
    /// insertion order (voxlap's `vx5.lightsrc[0..numlights]`).
    #[must_use]
    pub fn lights(&self) -> &[LightSrc] {
        &self.lights
    }

    /// Set the sky texture used by the textured-`startsky` path.
    /// `None` reverts to the cheap solid-fill default.
    pub fn set_sky(&mut self, sky: Option<Sky>) {
        self.sky = sky;
    }

    /// The current sky texture, if one is loaded (see
    /// [`set_sky`](Self::set_sky)). `None` ⇒ the solid-fill sky path.
    #[must_use]
    pub fn sky(&self) -> Option<&Sky> {
        self.sky.as_ref()
    }

    /// Render one frame into the caller-owned ARGB framebuffer.
    ///
    /// `pixels` is a row-major u32 buffer; `pitch_pixels` is the row
    /// stride in u32 elements (which equals `width` for a tightly-
    /// packed buffer, but may be larger when the host is e.g. an SDL2
    /// streaming texture with per-row padding).
    ///
    /// R3 is a stub that fills the visible region with [`sky_color`].
    /// R4 replaces this with the real raycaster.
    ///
    /// # Panics
    ///
    /// Panics if `pixels.len() < (height as usize) * (pitch_pixels as
    /// usize)` or if `width > pitch_pixels` — i.e. when the buffer
    /// would not contain `height` rows of `pitch_pixels` u32 each, or
    /// when the visible width would overflow each row.
    ///
    /// [`sky_color`]: Self::sky_color
    pub fn render(&mut self, pixels: &mut [u32], width: u32, height: u32, pitch_pixels: u32) {
        assert!(
            width <= pitch_pixels,
            "render: width {width} > pitch_pixels {pitch_pixels}"
        );
        let w = width as usize;
        let h = height as usize;
        let stride = pitch_pixels as usize;
        assert!(
            pixels.len() >= h * stride,
            "render: buffer too small ({} pixels) for {h} × {stride}",
            pixels.len(),
        );
        for y in 0..h {
            let row_start = y * stride;
            pixels[row_start..row_start + w].fill(self.sky_color);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_fills_with_sky_color() {
        let mut e = Engine::new();
        e.set_sky_color(0xdead_beef);
        let mut buf = vec![0u32; 64 * 32];
        e.render(&mut buf, 64, 32, 64);
        assert!(buf.iter().all(|&p| p == 0xdead_beef));
    }

    #[test]
    fn render_respects_pitch() {
        // Buffer wider than the visible rectangle — the trailing slack
        // per row must be left untouched.
        let mut e = Engine::new();
        e.set_sky_color(0x1234_5678);
        let stride: u32 = 80;
        let width: u32 = 64;
        let height: u32 = 32;
        let mut buf = vec![0u32; (stride as usize) * (height as usize)];
        e.render(&mut buf, width, height, stride);
        for y in 0..height as usize {
            let row = &buf[y * stride as usize..(y + 1) * stride as usize];
            assert!(row[..width as usize].iter().all(|&p| p == 0x1234_5678));
            assert!(row[width as usize..].iter().all(|&p| p == 0));
        }
    }

    #[test]
    fn fog_defaults_disabled() {
        let e = Engine::new();
        assert_eq!(e.fog_color(), 0);
        assert_eq!(e.fog_max_scan_dist(), 0);
    }

    #[test]
    fn set_fog_stores_color_and_distance() {
        let mut e = Engine::new();
        e.set_fog(0xFF_AA_BB_CC, 1024);
        assert_eq!(e.fog_color(), 0xFF_AA_BB_CC);
        assert_eq!(e.fog_max_scan_dist(), 1024);
    }

    #[test]
    fn set_fog_clamps_negative_distance_to_zero() {
        let mut e = Engine::new();
        e.set_fog(0xFF, -100);
        assert_eq!(e.fog_max_scan_dist(), 0);
    }

    #[test]
    fn camera_default_matches_oracle_placeholders() {
        // The Camera::default values must match what voxlaptest's
        // oracle.c writes into the .vxl header so a default-built
        // Engine + a freshly-loaded oracle.vxl agree on the starting
        // pose.
        let cam = Engine::new().camera();
        let bits = |a: [f64; 3]| a.map(f64::to_bits);
        assert_eq!(bits(cam.pos), bits([1024.0, 1024.0, 128.0]));
        assert_eq!(bits(cam.right), bits([1.0, 0.0, 0.0]));
        assert_eq!(bits(cam.down), bits([0.0, 0.0, 1.0]));
        assert_eq!(bits(cam.forward), bits([0.0, 1.0, 0.0]));
    }
}
