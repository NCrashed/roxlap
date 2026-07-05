//! Fuzz the `.kv6` parser; round-trip survivors through the writer.
#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(m) = roxlap_formats::kv6::parse(data) {
        let _ = roxlap_formats::kv6::serialize(&m);
    }
});
