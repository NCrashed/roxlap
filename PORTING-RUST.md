# roxlap — Pure-Rust port of Ken Silverman's Voxlap

Substage roadmap and locked decisions. See [README.md](README.md) for
project intent and the relationship to
[voxlaptest](https://github.com/NCrashed/voxlaptest).

## Locked decisions

| # | Decision | Consequence |
|---|---|---|
| 1 | **Pure Rust, no C FFI.** | Eliminates voxlaptest's MASM / inline-asm / MSVC-only baggage. One toolchain (`cargo`). |
| 2 | **Targets: x86_64, aarch64, wasm32.** | SIMD via `core::arch::*` per architecture; portable scalar fallback as the correctness reference. |
| 3 | **SDL2 host as the canonical demo.** | Replaces both Voxlaptest's C# frontend and Ken's original `game.c` host. Cross-platform for free. |
| 4 | **Cargo workspace from day 1.** | `roxlap-core` (engine), `roxlap-formats` (I/O), `roxlap-host` (winit + softbuffer demo). Splittable later (e.g. `roxlap-wasm` web host in R10). |
| 5 | **Bit-exact data format compat with voxlaptest.** | `.vxl`/`.kv6`/`.kfa` produced by voxlaptest's tooling load identically in roxlap and vice versa. |
| 6 | **Algorithmic correctness validated against voxlaptest's C oracle.** | Image-similarity initially (FP non-associativity tolerable); bit-exact convergence after R5 SSE batches land. |
| 7 | **License: `MIT OR Apache-2.0`.** | Standard Rust ecosystem dual. README documents Voxlap's separate non-commercial-only license — commercial users contact Ken. |
| 8 | **No naive per-pixel raycaster shortcut in R4.** | The grouscan algorithm is the engine's defining contribution; porting it directly is the point of the project. |

## Substage roadmap

| Stage | Scope | Validation |
|---|---|---|
| **R1** | Repo skeleton: workspace, README, `PORTING-RUST.md`, license files, asset copy. | `cargo build` green, `cargo test` green (zero tests). |
| **R2** | `.vxl` / `.kv6` / `.kvx` / `.kfa` parsers in `roxlap-formats`. | Byte-equal round-trip parse + re-serialise on `assets/coco.kvx` and procedurally-built `.vxl` fixtures. Header dumps match voxlaptest's loader output. |
| **R3** | `Engine` public API (framebuffer, camera, render entry point); winit + softbuffer host opens a window and shows sky-blue fill. | `cargo run -p roxlap-host` opens a window. End-to-end toolchain works before any rasterizer code lands. |
| **R4** | Full `opticast` + `grouscan` algorithm port from `grouscanasm_scalar` (scalar Rust). | Renders the same scene voxlaptest does; image-similarity ≥ 99% per terrain pose (RMS pixel diff bounded by FP rounding). |
| **R5** | x86_64 SSE2 4-pixel rsqrtps batches in the four hot rasterizers (mirror of voxlaptest Stage 4.9). | Re-converges on voxlaptest's CI golden hashes where the rsqrtps approach is bit-equivalent. |
| **R6** | KV6 sprite renderer (scalar `drawsprite` / `drawboundcube`). | Sprite poses image-similarity ≥ 99%. |
| **R7** | KV6 sprite renderer SSE (mirror of voxlaptest's `drawboundcube_sse`). | Re-converges on voxlaptest's sprite hashes. |
| **R8** | Cross-engine oracle: roxlap-side oracle binary writing `roxlap-hashes.txt`; CI matrix that diffs against `golden-hashes.txt`. | x86_64 Linux + Windows CI green, hashes equal voxlaptest's. |
| **R9** | ARM NEON via `core::arch::aarch64`; macOS arm64 + Linux aarch64 in CI. | Own goldens (NEON ≠ x86 SSE bits); aarch64 CI green. |
| **R10** | wasm SIMD via `core::arch::wasm32`; web host (canvas + js glue) as a separate crate. | Browser perf benchmark; own wasm goldens. |
| **R11** | Polish: docs, examples, version 0.1 publish to crates.io. | Crates published; docs.rs renders. |

## Substage R4 — opticast + grouscan (the hard part)

The biggest single port and the algorithm at the heart of Voxlap.
voxlaptest's `grouscanasm_scalar` (~600 lines in `voxlap/voxlap5.c`) is the
reference implementation; the algorithm is documented in
[voxlaptest/docs/grouscan-algorithm.md](https://github.com/NCrashed/voxlaptest/blob/port/docs/grouscan-algorithm.md).

Sub-substages, each landing as its own commit and validated against
voxlaptest's image output:

| # | Scope |
|---|---|
| **R4.1a** | `setcamera` math: derive the per-frame camera-relative basis, translation `giadd`, frustum corners `gcorn[4]`, and frustum edge normals `ginor[4]` from `Camera` + screen-projection parameters. Pure value-in / value-out, bit-exact testable. |
| **R4.1b** | Opticast prelude: `gylookup` mip table, `gposxfrac`/`gposyfrac`, `gpixy` (sptr-based column-base address), `gstartv` (top-of-column slab walk). Per-frame state cache. |
| **R4.1c** | Column-scan dispatch: the four-quadrant `vline`/`hline` setup + `angstart` radar table + per-column `hrend` / `vrend` calls. Rasterizers stubbed until R4.3. |
| **R4.2** | Scalar `hrend` / `vrend` scanline rasterizers — consume radar entries, write pixels + z-buffer. First stage that puts real pixels on screen behind a stub `gline`. (Originally scoped as "drawcwall/drawfwall/drawceil/drawflor"; those turned out to be labels *inside* grouscan, so they fold into R4.3.) |
| **R4.3** | grouscan = `gline` ray-cast: full algorithmic port including the drawcwall / drawfwall / drawceil / drawflor / deletez / findslab / slab-split / mid-column-search state machine. The hard core. |
| **R4.4** | Sky-fill primitives: `startsky` integration, fog gating. |
| **R4.5** | Mip-level transition (`remiporend`): the deeper-recursion cftype refresh. |

## Substage R5 — x86 SSE batches

Mirror of voxlaptest's Stage 4.9. Same four rasterizers, same rsqrtps
approximation strategy (no Newton refinement), but using
`core::arch::x86_64::{__m128, _mm_*}` from a single source — one intrinsic
per call site as in C, not portable_simd. Sub-substages:

| # | Scope |
|---|---|
| **R5.1** | `hrend_z_sse` — horizontal-scan, no-fog, 4-pixel rsqrtps batch. |
| **R5.2** | `hrend_z_fog_sse` — horizontal-scan, fog blend (per-pixel scalar inside batch, matching voxlaptest's bit-exact path). |
| **R5.3** | `vrend_z_sse` — vertical-scan with parallel `uurend` update. |
| **R5.4** | `vrend_z_fog_sse` — vertical-scan with fog. |

## Out of scope

- Network play, multiplayer, game logic.
- Loading non-Voxlap asset formats. Stick to `.vxl` / `.kv6` / `.kvx` / `.kfa`.
- DirectX or Vulkan backends. Engine remains a software rasterizer; the host
  blits one texture per frame.
- Pre-R8 CI: until the oracle binary lands, validation is local-only.

## Sync with voxlaptest

When voxlaptest ships an oracle change (new pose, refrozen hash, fixed bug),
reflect the matching change in roxlap's oracle as soon as roxlap implements
the affected code path. Both repos move on the `port` / `master` branches
respectively; PRs in either reference the other.
