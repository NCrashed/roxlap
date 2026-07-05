//! Fuzz the `.vxl` world parser; round-trip survivors.
#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(w) = roxlap_formats::vxl::parse(data) {
        let _ = roxlap_formats::vxl::serialize(&w);
    }
});
