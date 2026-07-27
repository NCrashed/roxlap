//! CT.3 — fuzz the edit pipeline with carve-through shapes: random
//! insert/carve rects, sphere carves and bottom-reaching carves on a
//! small world, then assert the format invariants (byte-stable
//! round-trip, sane per-column runs incl. the empty sentinel /
//! air-terminal, mip ladder builds + walks). The driver lives in the
//! library (`roxlap_formats::fuzz_driver`) so the committed seeds also
//! run as stable-CI unit tests.
#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    roxlap_formats::fuzz_driver::run_edit_ops(data);
});
