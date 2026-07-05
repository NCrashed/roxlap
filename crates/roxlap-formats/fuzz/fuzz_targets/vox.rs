//! Fuzz the MagicaVoxel `.vox` parser: must never panic/OOM — every
//! failure is a typed `ParseError`. On success, exercise the
//! consumer path too (to_kv6_models walks the parsed structures).
#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(f) = roxlap_formats::vox::parse(data) {
        let _ = f.to_kv6_models();
    }
});
