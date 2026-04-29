//! The [`Engine`] is the public façade of `roxlap-core`. R3 ships
//! the API surface with a sky-fill stub; R4 wires in the real
//! opticast + grouscan rasterizer behind the same call signatures.

use crate::Camera;

/// Voxel engine state.
#[derive(Debug, Clone)]
pub struct Engine {
    camera: Camera,
    sky_color: u32,
}

impl Default for Engine {
    fn default() -> Self {
        Self {
            camera: Camera::default(),
            // Voxlap-style packed sky blue: brightness bit | 0x87ceeb.
            sky_color: 0x8087_ceeb,
        }
    }
}

impl Engine {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_camera(&mut self, camera: Camera) {
        self.camera = camera;
    }

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

    #[must_use]
    pub fn sky_color(&self) -> u32 {
        self.sky_color
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
