//! Render the bundled oracle world from a north-facing camera into
//! a PPM image. Showcases the minimum viable opticast pipeline.
//!
//! Run from the workspace root (so `assets/oracle.vxl.gz` resolves):
//!
//! ```sh
//! cargo run --release --example oracle_view -p roxlap-core
//! ```
//!
//! Output: `oracle_view.ppm` in the current directory.

// Single-char locals (w, h, fb, zb, b, g, r) are conventional in
// rendering example code; relax the pedantic name lint here.
#![allow(clippy::many_single_char_names)]

use std::io::{Read, Write};

use flate2::read::GzDecoder;
use roxlap_core::rasterizer::ScanScratch;
use roxlap_core::scalar_rasterizer::ScalarRasterizer;
use roxlap_core::{opticast, Camera, OpticastSettings};
use roxlap_formats::vxl;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (w, h) = (640u32, 480u32);

    // Decode the gzipped voxel world (~37 MB of slab data).
    let gz = std::fs::read("assets/oracle.vxl.gz")?;
    let mut bytes = Vec::new();
    GzDecoder::new(&gz[..]).read_to_end(&mut bytes)?;
    let world = vxl::parse(&bytes)?;

    let mut fb = vec![0u32; (w as usize) * (h as usize)];
    let mut zb = vec![0.0f32; fb.len()];
    let mut scratch = ScanScratch::new_for_size(w, h, world.vsid);

    let mut rast = ScalarRasterizer::new(
        &mut fb,
        &mut zb,
        w as usize,
        &world.data,
        &world.column_offset,
        &world.mip_base_offsets,
        world.vsid,
    );

    let _ = opticast(
        &mut rast,
        &mut scratch,
        &Camera::default(),
        &OpticastSettings::for_oracle_framebuffer(w, h),
        world.vsid,
        &world.data,
        &world.column_offset,
    );

    let mut f = std::fs::File::create("oracle_view.ppm")?;
    write!(f, "P6\n{w} {h}\n255\n")?;
    for &px in &fb {
        // Framebuffer is BGRA in little-endian u32 (matches voxlap +
        // softbuffer); to_le_bytes gives [B, G, R, A]. PPM wants RGB.
        let [b, g, r, _a] = px.to_le_bytes();
        f.write_all(&[r, g, b])?;
    }
    println!("wrote oracle_view.ppm ({w}×{h})");
    Ok(())
}
