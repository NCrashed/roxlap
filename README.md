# roxlap

A pure-Rust port of [Ken Silverman's Voxlap](http://advsys.net/ken/voxlap.htm)
voxel engine — a CPU-rendered 3D voxel renderer from the Build-engine era.
Runs on Linux / macOS / Windows from one Cargo workspace, no GPU required,
no C dependency, idiomatic safe Rust with per-architecture SIMD.

![sample render from roxlap-oracle](docs/screenshot.png)

## What is Voxlap?

Voxlap is the voxel rendering engine [Ken Silverman](http://advsys.net/ken/)
wrote in the early 2000s, after the Build engine that powered *Duke Nukem
3D*. It draws volumetric voxel terrain plus animated kv6 sprites entirely
on the CPU, using Ken's classic "raycast columns + scanline fill"
algorithm — no GPU, no shaders. Cult-favourite games like
[Voxelstein 3D](https://en.wikipedia.org/wiki/Voxelstein_3D),
[Ace of Spades](https://en.wikipedia.org/wiki/Ace_of_Spades_\(video_game\)),
and Ken's own *Slab6* / *Voxed* shipped on top of it.

roxlap is that engine, reimplemented from scratch in Rust. It reads the
same `.vxl` (worlds) / `.kv6` / `.kvx` (sprite voxels) / `.kfa` (sprite
animation rigs) files Ken's engine reads, renders them with the same
algorithms, and is **bit-exact** against the reference C engine
([voxlaptest](https://github.com/NCrashed/voxlaptest)) on every test
pose where the underlying SIMD allows.

## Quick start

Try the interactive demo on a sample voxel world (no extra setup —
all assets are bundled into the binary via `include_bytes!`):

```sh
git clone https://github.com/NCrashed/roxlap
cd roxlap
cargo run --release -p roxlap-host
```

A window opens with WASD + mouse-look fly-through over a procedurally-
carved voxel landscape, with an animated KFA sprite and a textured
panoramic sky. Press `L` to toggle baked world-voxel lighting.
Press `F` to capture the current camera + framebuffer to
`roxlap-capture.{txt,ppm}` for off-line repro of any render artifact.

## Crates

| Crate | Purpose |
|-------|---------|
| [`roxlap-core`](crates/roxlap-core) | The engine: framebuffer, camera, opticast raycaster, grouscan rasterizer, sprite + sky + voxel-lighting. |
| [`roxlap-formats`](crates/roxlap-formats) | Pure parsers for voxlap's on-disk file formats — `.vxl`, `.kv6`, `.kvx`, `.kfa` — plus the `Sprite` / `KfaSprite` data types. No renderer dependency; useful standalone for asset pipelines. |
| [`roxlap-host`](crates/roxlap-host) | Interactive demo binary (winit + softbuffer). |
| [`roxlap-oracle`](crates/roxlap-oracle) | Cross-engine render-hash oracle: renders 12 fixed test poses, FNV-1a-hashes each framebuffer, diffs against voxlaptest's C goldens. CI gates on this. |

The library API surface is documented at [docs.rs/roxlap-core](https://docs.rs/roxlap-core)
and [docs.rs/roxlap-formats](https://docs.rs/roxlap-formats).

## Why roxlap?

- **Cross-platform from one source.** Linux, Windows, macOS (x86_64 +
  arm64), wasm — all from one Cargo workspace. No `#ifdef _MSC_VER`,
  no MASM, no C FFI.
- **SIMD per architecture.** SSE2 on x86_64 today; NEON on aarch64
  and v128 on wasm planned (R9 / R10). All via `core::arch::*`
  intrinsics. A portable scalar fallback exists as the correctness
  reference.
- **Idiomatic safe Rust public API.** RAII handles, `Result` at every
  external boundary, no globals leaked across an FFI seam because there
  is no FFI.
- **Bit-exact correctness against voxlaptest** where the SIMD approach
  matches; image-similarity correctness everywhere else, with frozen
  per-pose hashes pinning known sub-pixel rounding noise so any
  *unintentional* drift fails CI immediately.

## Status

The renderer is feature-complete: voxel terrain (`opticast` + `grouscan`),
animated kv6 sprites, world-voxel lighting, textured panoramic sky,
x86_64 SSE2 batches. The cross-engine oracle tracks 9 of 12 voxlap C
poses — 5 byte-for-byte bit-exact with the C reference, 4 frozen as
roxlap's own goldens after visual verification (sub-pixel rounding
noise from `_mm_rcp_ps`-based vertex projection is documented in
[PORTING-RUST.md](PORTING-RUST.md)).

```text
$ cargo run --release -p roxlap-oracle -- diff
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

Open work: ARM NEON (R9), wasm SIMD + browser host (R10), and
multicore CPU rendering (R12, ~3-5× projected). See
[PORTING-RUST.md](PORTING-RUST.md) for the full substage roadmap.

## Documentation

- API: [docs.rs/roxlap-core](https://docs.rs/roxlap-core),
  [docs.rs/roxlap-formats](https://docs.rs/roxlap-formats).
- Algorithm + porting notes: [PORTING-RUST.md](PORTING-RUST.md).
- Reference C engine this ports from:
  [voxlaptest](https://github.com/NCrashed/voxlaptest).
- Original Voxlap homepage: [advsys.net/ken/voxlap.htm](http://advsys.net/ken/voxlap.htm).

## Contributing

After cloning, point git at the tracked hooks:

```sh
git config core.hooksPath .githooks
```

Installed:
- **`pre-commit`** — `cargo fmt --check` across the workspace, with
  unstaged changes stashed for the check so it never fails on
  something you didn't stage. Bypass with `git commit --no-verify`.
- **`commit-msg`** — strips trailing whitespace from every commit
  message line.

Clippy is **not** in the pre-commit hook — pedantic lints are
opinionated enough that a >2-second pre-commit hook would just get
`--no-verify`'d. Run `cargo clippy --all-targets -- -D warnings`
manually before pushing if you want the same gate locally; CI
enforces it on every push regardless
([.github/workflows/ci.yml](.github/workflows/ci.yml)).

## License

Dual-licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

The Voxlap engine algorithms and on-disk data formats this crate
implements were originally created by Ken Silverman. Voxlap's
original C source is distributed under separate terms: royalty-free
for non-commercial use; commercial use requires a license from Ken
Silverman directly. roxlap is an independent Rust port that does not
contain Ken's original C source, but its observable behaviour mirrors
his engine's. If you intend to use roxlap or any derived work
commercially, contact Ken Silverman about Voxlap commercial
licensing — see [advsys.net/ken](http://advsys.net/ken/) for
current contact information.
