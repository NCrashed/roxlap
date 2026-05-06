//! R10.1 + R10.4 — wasm32 render goldens.
//!
//! Iterates all 12 oracle poses (the same set the native bin
//! renders), FNV-1a64-hashes each framebuffer, and asserts the
//! result against `tests/wasm-hashes.txt` line-for-line. The
//! goldens file is committed; on a fresh wasm port edit (e.g.
//! R10.3 SIMD changes) the failing test prints the new hashes
//! so an authorised refreeze is a straight `cp` of the printed
//! lines into the file.
//!
//! On native (non-wasm32) targets the file is fully cfg-gated
//! out; the native bin's `cargo run -- diff` covers the same
//! 12 poses against `tests/golden-hashes.txt`.

#![cfg(target_arch = "wasm32")]

use roxlap_oracle::{load_oracle_vxl, render_all_poses, POSES};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_node_experimental);

const WASM_GOLDENS: &str = include_str!("wasm-hashes.txt");

#[wasm_bindgen_test]
fn all_12_poses_match_wasm_goldens() {
    console_error_panic_hook::set_once();

    let mut vxl = load_oracle_vxl();
    let mut engine = roxlap_core::Engine::new();
    let rows = render_all_poses(&mut engine, &mut vxl);

    // Pretty-print the actual hashes so a refreeze is a copy-paste.
    let mut got = String::new();
    for (name, hash) in &rows {
        got.push_str(&format!("{name:<20}{hash:016x}\n"));
    }
    web_sys_console_log("R10.4 wasm pose hashes:");
    web_sys_console_log(&got);

    // Parse the goldens file the same way: name [whitespace] hex.
    let mut want: Vec<(&str, u64)> = Vec::with_capacity(POSES.len());
    for (i, line) in WASM_GOLDENS.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let mut parts = trimmed.split_ascii_whitespace();
        let name = parts.next().unwrap_or_else(|| {
            panic!("wasm-hashes.txt:{}: missing name", i + 1);
        });
        let hex = parts.next().unwrap_or_else(|| {
            panic!("wasm-hashes.txt:{}: missing hash", i + 1);
        });
        let hash = u64::from_str_radix(hex, 16).unwrap_or_else(|e| {
            panic!("wasm-hashes.txt:{}: invalid hex {hex:?}: {e}", i + 1);
        });
        want.push((name, hash));
    }

    assert_eq!(
        rows.len(),
        want.len(),
        "pose count mismatch: rendered {} poses but goldens has {} entries",
        rows.len(),
        want.len(),
    );

    let mut mismatches = 0_usize;
    for ((got_name, got_hash), (want_name, want_hash)) in rows.iter().zip(want.iter()) {
        assert_eq!(
            got_name, want_name,
            "pose name order drift: got {got_name:?}, expected {want_name:?}",
        );
        if got_hash != want_hash {
            mismatches += 1;
        }
    }
    assert!(
        mismatches == 0,
        "{} of {} wasm pose hashes differ from tests/wasm-hashes.txt — paste the table below over the file to refreeze:\n{}",
        mismatches,
        rows.len(),
        got,
    );
}

// Tiny inline shim so we don't need the full `web-sys` crate just
// to print one string. wasm-bindgen-test pipes Node's `console.log`
// to its captured-output channel.
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console, js_name = log)]
    fn console_log(s: &str);
}
fn web_sys_console_log(s: &str) {
    console_log(s);
    let _ = JsValue::from_str(s);
}
