//! Fuzz the `.rkc` character parser; round-trip survivors.
#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(c) = roxlap_formats::character::parse(data) {
        let _ = roxlap_formats::character::serialize(&c);
    }
});
