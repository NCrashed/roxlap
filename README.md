# roxlap

A pure-Rust port of [Ken Silverman's Voxlap](http://advsys.net/ken/voxlap.htm)
voxel engine.

## Status

R4 (opticast + grouscan), R5 (x86_64 SSE batches), R6 (KV6 sprites,
sprite + world voxel lighting, textured sky), R6.6 (animated KFA
sprites), and R8 (oracle + CI) are all done. R7 was cancelled; R9
(NEON), R10 (wasm), R11 (polish + crates.io publish) are open.

The cross-engine oracle tracks 9 of 12 voxlap C poses — 5 byte-for-byte
bit-exact against voxlap C, and 4 frozen as roxlap's own goldens
(visually verified against voxlap C, sub-pixel rounding noise
documented).

```
$ cargo run -p roxlap-oracle -- diff
MATCH    north  326a7c41c3cc659d
MATCH    east  3e00f1d0d62d5be0
MATCH    diag_down  118de3c1132d0f6b
MATCH    high_down  cd1ceac6e21c55f4
MATCH    sprite_above  79b87c92dd96a59b
MATCH    sprite_front  87c7de0ddeb0f7ce        (roxlap-frozen)
MATCH    sprite_iso  9caf71069594fde6          (roxlap-frozen)
MATCH    sprite_coco  0c8f30141c9e7a4e         (roxlap-frozen)
MATCH    diag_down_lit  b536ce3fdf771b9e       (roxlap-frozen)
9 match, 0 mismatch, 0 missing-from-golden (9 total roxlap rows)
```

R8 GitHub CI runs fmt / clippy / tests / oracle-diff on every push
and fails the build if any frozen hash drifts. Expected hashes live in
[`tests/golden-hashes.txt`](tests/golden-hashes.txt); the workflow is
in [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

A separate roxlap-only integration test
(`crates/roxlap-core/tests/multi_mip.rs`) regression-tests the
multi-mip rendering path — voxlap C never exercises `vxlmipuse > 1`
so no shared golden exists, and the test pins roxlap's own FNV-1a
hashes for the single-mip baseline and the post-`generate_mips(4)`
render.

[`roxlap-host`](crates/roxlap-host) is interactive: opens a winit
window, loads `oracle.vxl.gz`, and renders with WASD + mouse-look,
animated KFA sprites, and a textured panoramic sky. Press `L` to
toggle the world-voxel lighting bake; press `F` to capture the
current camera pose + framebuffer to `roxlap-capture.{txt,ppm}` for
off-line repro of any rendering artifact. Side-shading is on by
default in the host so directional lighting reads correctly; the
oracle stays at voxlap's `setsideshades(0,…,0)` baseline so its
hashes don't shift.

### What's left

R9 (ARM NEON), R10 (wasm SIMD + browser host), and R11 (docs +
crates.io publish) are the remaining big stages. Smaller open
items — a `do_slab_split` perf restoration (1-15% scanline win),
a deferred `drawsprite` no-z path — are tracked in
[PORTING-RUST.md](PORTING-RUST.md).

The port is staged out of
[voxlaptest](https://github.com/NCrashed/voxlaptest), the modernised C
fork of Voxlap.

## Goals

- **Cross-platform from one source.** Linux, Windows, macOS (x86_64 + arm64),
  wasm, all from a single Cargo workspace. No `#ifdef _MSC_VER`, no MASM, no
  C FFI.
- **SIMD per architecture.** SSE2 on x86_64 via `core::arch::x86_64`, NEON on
  aarch64 via `core::arch::aarch64`, v128 on wasm via `core::arch::wasm32`.
  Portable scalar fallback as the correctness reference.
- **Idiomatic Rust public API.** RAII handles for the engine, voxel maps,
  lighting; iterator-style colour funcs; `Result` everywhere external I/O can
  fail; no globals leaked across the FFI boundary because there is no FFI.
- **Bit-exact correctness against voxlaptest** where the SIMD approach
  matches. Image-similarity correctness everywhere else.

## Workspace

```
crates/
├── roxlap-core/      engine: framebuffer, camera, opticast, grouscan, rasterizers
├── roxlap-formats/   .vxl / .kv6 / .kvx / .kfa parsers
├── roxlap-host/      winit + softbuffer demo binary
└── roxlap-oracle/    cross-engine render-hash oracle (writes roxlap-hashes.txt; `diff` mode compares against voxlap C goldens)
```

## Relationship to voxlaptest

[voxlaptest](https://github.com/NCrashed/voxlaptest) is the modernised C
engine this project ports *from*:

- voxlaptest is the **bit-exact reference** for terrain and sprite rasterizer
  correctness during development.
- roxlap matches its image output using the same oracle harness — 5 of 9
  tracked poses already byte-for-byte; the remaining 4 are frozen as
  roxlap's own hashes after visual verification (sub-pixel rounding
  noise from voxlap C's `_mm_rcp_ps`-based vertex projection).
- Once roxlap reaches full feature + perf parity, voxlaptest can be retired.

The two repositories move in sync: when voxlaptest gains a new oracle pose
or freezes a hash change, roxlap's matching code path lands the equivalent
update.

## Development

After cloning, point git at the tracked hooks directory:

```sh
git config core.hooksPath .githooks
```

That installs:

- **`pre-commit`** — `cargo fmt --check` across the workspace, with
  unstaged changes stashed for the check so it never fails on
  something you did not stage. Run `cargo fmt` and re-stage if it
  flags anything. Bypass with `git commit --no-verify`.
- **`commit-msg`** — strips trailing whitespace from every line of
  the commit message.

Clippy is **not** in the pre-commit hook — pedantic-level lints are
opinionated enough that they belong in CI rather than every-commit
gating, and a >2-second pre-commit hook just gets `--no-verify`'d.
Run `cargo clippy --all-targets -- -D warnings` manually before
pushing if you want the same gate locally; CI enforces it on every
push regardless (see [`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

## License

Dual-licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

The Voxlap engine algorithms and on-disk data formats this crate implements
were originally created by Ken Silverman. Voxlap's original C source is
distributed under separate terms: royalty-free for non-commercial use;
commercial use requires a license from Ken Silverman directly. roxlap is an
independent Rust port that does not contain Ken's original C source, but its
observable behaviour mirrors his engine's. If you intend to use roxlap or any
derived work commercially, contact Ken Silverman about Voxlap commercial
licensing — see [advsys.net/ken](http://advsys.net/ken/) for current contact
information.
