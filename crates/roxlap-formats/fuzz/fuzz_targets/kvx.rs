//! Fuzz the `.kvx` parser; round-trip survivors through the writer.
#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(m) = roxlap_formats::kvx::parse(data) {
        let _ = roxlap_formats::kvx::serialize(&m);
    }
});
