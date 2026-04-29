# roxlap

A pure-Rust port of [Ken Silverman's Voxlap](http://advsys.net/ken/voxlap.htm)
voxel engine.

## Status

Very early — Stage R1 (repo skeleton). The port is being staged out of
[voxlaptest](https://github.com/NCrashed/voxlaptest), the modernised C fork of
Voxlap. See `PORTING-RUST.md` for the substage roadmap and current status.

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
└── roxlap-sdl/       SDL2-hosted demo binary
```

## Relationship to voxlaptest

[voxlaptest](https://github.com/NCrashed/voxlaptest) is the modernised C
engine this project ports *from*:

- voxlaptest is the **bit-exact reference** for terrain and sprite rasterizer
  correctness during development.
- roxlap matches its image output using the same oracle harness — eventually
  re-converging on the same `golden-hashes.txt` rows once the SIMD batches in
  R5 / R7 land.
- Once roxlap reaches feature + perf parity, voxlaptest can be retired.

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
pushing if you want the same gate locally; CI (from R8 onward) will
enforce it on every push regardless.

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
