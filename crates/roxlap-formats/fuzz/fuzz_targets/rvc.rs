//! Fuzz the `.rvc` clip parser; decode survivors (the delta-apply
//! path is where crafted frame data would bite).
#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(c) = roxlap_formats::voxel_clip::VoxelClip::parse(data) {
        let _ = c.decode();
    }
});
