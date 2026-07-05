//! Fuzz the `.kfa` rig parser; round-trip survivors.
#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(k) = roxlap_formats::kfa::parse(data) {
        let _ = roxlap_formats::kfa::serialize(&k);
    }
});
