//! QE.8b — blocking GPU readbacks + unproject, split verbatim out
//! of `lib.rs`: click-time depth picking, whole-frame colour capture
//! (QE.7a), and the vertical-FOV pinhole pixel→ray helper the
//! picking path shares with the facade.

use crate::GpuRenderer;

impl GpuRenderer {
    /// Read back the per-pixel world-t depth at window pixel `(x, y)`
    /// from the last rendered frame, for screen→world picking. Returns
    /// the distance `t` along the (normalised) view ray to the nearest
    /// scene-grid surface, so the host reconstructs the world hit as
    /// `cam.pos + t * normalize(ray_dir)`. `None` for out-of-bounds
    /// pixels, sky / no-hit (the `T_INF` sentinel), or when no scene
    /// frame has been rendered.
    ///
    /// The depth buffer is the SCENE pass's output (terrain + grids),
    /// untouched by the sprite pass (which reads it read-only), so a
    /// cursor sprite under the pointer does not occlude the pick.
    ///
    /// Synchronous: copies the depth buffer to a mapped staging buffer
    /// and blocks on `device.poll(Wait)`. Cheap enough for click-time
    /// picks; do not call it every frame.
    ///
    /// Requires the last frame to have written depth, which happens
    /// when sprites are present (`write_depth`). The pick demo always
    /// has a cursor sprite, so this holds.
    ///
    /// Compiles on wasm, but the wasm facade never calls it: WebGPU's
    /// `device.poll` doesn't block for the GPU, so the blocking
    /// `recv()` here would hang the single browser thread. Picking is
    /// deferred on the wasm GPU path (the facade returns `None`).
    #[must_use]
    pub fn read_depth_pixel(&self, x: u32, y: u32) -> Option<f32> {
        let dda = self.scene_dda.as_ref()?;
        let (w, h) = dda.storage_size;
        if x >= w || y >= h {
            return None;
        }
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("roxlap-gpu depth readback"),
            });
        // PF.5 (H4) — copy ONLY the picked pixel's 4 bytes, not the whole
        // depth buffer (8+ MB at high res): the pick still blocks on the
        // poll below, but the copy + map are now O(1). The 4-byte offset
        // meets wgpu's copy alignment.
        let offset = (u64::from(y) * u64::from(w) + u64::from(x)) * 4;
        enc.copy_buffer_to_buffer(&dda.depth_buffer, offset, &dda.depth_readback, 0, 4);
        self.queue.submit(std::iter::once(enc.finish()));

        let slice = dda.depth_readback.slice(..4);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        rx.recv().ok()?.ok()?;

        let t = {
            let data = slice.get_mapped_range();
            let bytes: [u8; 4] = data[0..4].try_into().ok()?;
            f32::from_le_bytes(bytes)
        };
        dda.depth_readback.unmap();

        // Reject sky / no-hit (T_INF == 1e30 in the shader) + non-finite.
        if !t.is_finite() || t >= 1.0e29 {
            return None;
        }
        Some(t)
    }

    /// QE.7a — read back the last rendered frame's colour at the
    /// **logical** resolution (post-SSAA/posterize, pre-upscale) as
    /// `0x00RRGGBB` pixels — the GPU side of frame capture, closing
    /// the "screenshots impossible on the GPU backend" parity gap.
    ///
    /// Blocking (encode copy → submit → map, like
    /// [`Self::read_depth_pixel`]): a screenshot hotkey, not a
    /// per-frame path. `None` before the first scene render. Compiles
    /// on wasm but must not be called there — WebGPU's `poll` can't
    /// block, so the facade returns `None` on the wasm GPU path.
    #[must_use]
    pub fn read_frame_pixels(&self) -> Option<(Vec<u32>, u32, u32)> {
        let dda = self.scene_dda.as_ref()?;
        let (w, h) = dda.logical_size;
        if w == 0 || h == 0 {
            return None;
        }
        // Mirror `render_scene`'s identity-resolve choice: with ssaa 1
        // + posterize off the resolve pass was skipped and the march
        // framebuffer IS the logical image.
        let identity = dda.storage_size == dda.logical_size && self.posterize.is_none();
        let src = if identity {
            &dda.framebuffer
        } else {
            &dda.resolve_buf
        };
        let size = u64::from(w) * u64::from(h) * 4;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("roxlap-gpu capture staging"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("roxlap-gpu capture readback"),
            });
        enc.copy_buffer_to_buffer(src, 0, &staging, 0, size);
        self.queue.submit(std::iter::once(enc.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        rx.recv().ok()?.ok()?;

        let pixels = {
            let data = slice.get_mapped_range();
            // The shaders store `pack4x8unorm(r, g, b, a)` — r in the
            // low byte. Repack to the facade's `0x00RRGGBB`.
            data.chunks_exact(4)
                .map(|px| {
                    let v = u32::from_le_bytes([px[0], px[1], px[2], px[3]]);
                    let (r, g, b) = (v & 0xff, (v >> 8) & 0xff, (v >> 16) & 0xff);
                    (r << 16) | (g << 8) | b
                })
                .collect()
        };
        staging.unmap();
        Some((pixels, w, h))
    }

    /// World-space view-ray direction (un-normalised) for window pixel
    /// `(x, y)`, under the GPU marcher's projection — the canonical GPU
    /// unproject, mirroring `scene_dda.wgsl`'s `render_scene`
    /// (vertical-FOV pinhole). Uses the last-rendered frame's target
    /// size + FOV; `None` before the first scene render. Pair with
    /// [`Self::read_depth_pixel`] for screen→world picking.
    #[must_use]
    pub fn pixel_ray(
        &self,
        right: [f64; 3],
        down: [f64; 3],
        forward: [f64; 3],
        x: f64,
        y: f64,
    ) -> Option<[f64; 3]> {
        let dda = self.scene_dda.as_ref()?;
        let (w, h) = dda.storage_size;
        if w == 0 || h == 0 || self.last_fov_y_rad <= 0.0 {
            return None;
        }
        Some(pinhole_pixel_ray(
            right,
            down,
            forward,
            x,
            y,
            f64::from(w),
            f64::from(h),
            f64::from(self.last_fov_y_rad),
        ))
    }
}

/// World-space view-ray direction (un-normalised) for window pixel
/// `(x, y)` under a vertical-FOV pinhole — the projection
/// `scene_dda.wgsl`'s `render_scene` uses. Shared by
/// [`GpuRenderer::pixel_ray`]; standalone so it's unit-testable without
/// a device. `right`/`down`/`forward` are the camera basis.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn pinhole_pixel_ray(
    right: [f64; 3],
    down: [f64; 3],
    forward: [f64; 3],
    x: f64,
    y: f64,
    w: f64,
    h: f64,
    fov_y_rad: f64,
) -> [f64; 3] {
    let half_h = (fov_y_rad * 0.5).tan();
    let half_w = half_h * (w / h);
    let ndc_x = (x + 0.5) / w * 2.0 - 1.0;
    let ndc_y_top = 1.0 - (y + 0.5) / h * 2.0;
    let (kx, ky) = (ndc_x * half_w, ndc_y_top * half_h);
    [
        forward[0] + kx * right[0] - ky * down[0],
        forward[1] + kx * right[1] - ky * down[1],
        forward[2] + kx * right[2] - ky * down[2],
    ]
}

#[cfg(test)]
mod pixel_ray_tests {
    use super::pinhole_pixel_ray;

    const RIGHT: [f64; 3] = [1.0, 0.0, 0.0];
    const DOWN: [f64; 3] = [0.0, 1.0, 0.0];
    const FWD: [f64; 3] = [0.0, 0.0, 1.0]; // voxlap z-down "look down"

    // Frame centre (NDC 0,0) points straight along `forward`.
    #[test]
    fn centre_pixel_is_forward() {
        let d = pinhole_pixel_ray(
            RIGHT,
            DOWN,
            FWD,
            639.5,
            359.5,
            1280.0,
            720.0,
            60_f64.to_radians(),
        );
        assert!(
            d[0].abs() < 1e-9 && d[1].abs() < 1e-9,
            "centre ≈ forward, got {d:?}"
        );
        assert!((d[2] - 1.0).abs() < 1e-9);
    }

    // Right edge pixel tilts +right by tan(hfov/2); the lateral
    // component equals half_w = tan(fov_y/2)*aspect at the very edge.
    #[test]
    fn right_edge_tilts_by_half_w() {
        let fov = 60_f64.to_radians();
        let d = pinhole_pixel_ray(RIGHT, DOWN, FWD, 1279.5, 359.5, 1280.0, 720.0, fov);
        let half_w = (fov * 0.5).tan() * (1280.0 / 720.0);
        assert!((d[0] - half_w).abs() < 1e-6, "x={}, half_w={half_w}", d[0]);
        assert!(d[0] > 0.0, "right edge tilts +right");
    }
}
