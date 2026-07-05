//! Fuzz the scene-snapshot envelope: bincode over attacker-controlled
//! bytes, then per-chunk `.vxl` parses — the most security-relevant
//! load path in the engine (save files travel between machines).
#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = roxlap_scene::Scene::load_snapshot(data);
});
