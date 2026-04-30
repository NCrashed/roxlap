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

| Stage | Scope | Status | Validation |
|---|---|---|---|
| **R1** | Repo skeleton: workspace, README, `PORTING-RUST.md`, license files, asset copy. | done | `cargo build` green, `cargo test` green (zero tests). |
| **R2** | `.vxl` / `.kv6` / `.kvx` / `.kfa` parsers in `roxlap-formats`. | done | Byte-equal round-trip parse + re-serialise on `assets/coco.kvx` and procedurally-built `.vxl` fixtures. Header dumps match voxlaptest's loader output. |
| **R3** | `Engine` public API (framebuffer, camera, render entry point); winit + softbuffer host opens a window and shows sky-blue fill. | done | `cargo run -p roxlap-host` opens a window. End-to-end toolchain works before any rasterizer code lands. |
| **R4** | Full `opticast` + `grouscan` algorithm port from `grouscanasm_scalar` (scalar Rust). | **opticast-only poses bit-exact**; deferred: R4.4 textured-sky branch, R4.5 multi-mip `remiporend`, sideshademode high-byte. | 4 of 12 oracle poses match voxlap C goldens byte-for-byte (`north`, `east`, `diag_down`, `high_down`). The other 8 poses each require a feature roxlap doesn't have yet (sprites → R6, lighting, drawtile). |
| **R5** | x86_64 SSE2 4-pixel rsqrtps batches in the four hot rasterizers (mirror of voxlaptest Stage 4.9). | scalar reference is bit-exact; SSE batches not yet ported | Re-converges on voxlaptest's CI golden hashes where the rsqrtps approach is bit-equivalent. |
| **R6** | KV6 sprite renderer (scalar `drawsprite` / `drawboundcube`). | not started | Sprite poses image-similarity ≥ 99%. |
| **R7** | KV6 sprite renderer SSE (mirror of voxlaptest's `drawboundcube_sse`). | not started | Re-converges on voxlaptest's sprite hashes. |
| **R8** | Cross-engine oracle: roxlap-side oracle binary writing `roxlap-hashes.txt`; CI matrix that diffs against `golden-hashes.txt`. | done | `cargo run -p roxlap-oracle -- diff` reports `4 match, 0 mismatch` against the in-tree `tests/golden-hashes.txt` (4 opticast-only poses, frozen against voxlap C). `.github/workflows/ci.yml` runs fmt / clippy / test / oracle-diff on every push + PR; oracle job fails on any hash mismatch so the bit-exact milestone can't regress silently. |
| **R9** | ARM NEON via `core::arch::aarch64`; macOS arm64 + Linux aarch64 in CI. | not started | Own goldens (NEON ≠ x86 SSE bits); aarch64 CI green. |
| **R10** | wasm SIMD via `core::arch::wasm32`; web host (canvas + js glue) as a separate crate. | not started | Browser perf benchmark; own wasm goldens. |
| **R11** | Polish: docs, examples, version 0.1 publish to crates.io. | not started | Crates published; docs.rs renders. |

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
| **R4.3** | grouscan = `gline` ray-cast: full algorithmic port. Sub-substaged the same way voxlaptest's grouscanasm scalar port was (`Stage 4.5b.2..6`): R4.3a magenta-placeholder gline, R4.3b gline frustum setup, R4.3c grouscan prologue + dispatch skeleton, R4.3d wall/ceil/flor fill loops, R4.3e findslab/slab-split/deletez, R4.3f remiporend + startsky. The hard core. |
| **R4.4** | Sky-fill primitives: `startsky` integration, fog gating. Solid-fill branch and fog blend done; textured-sky `skyoff != 0` path deferred until a fixture exercises it. |
| **R4.5** | Mip-level transition (`remiporend`): the deeper-recursion cftype refresh. **Audit complete; full body deferred behind a world-model dependency.** Oracle uses `gmipnum=1` so the gmipnum check short-circuits to `Phase::Startsky`. The body is gated on multi-mip column data (port of `genmipvxl`, voxlap5.c:4710+) which roxlap-formats doesn't load yet. See the audit at `crates/roxlap-core/src/grouscan.rs::phase_remiporend` — it covers (a) the LP32/LP64 sptr-stride bug in voxlap C's `<<29` / `<<(gmipcnt+17)` parity shifts, (b) why roxlap's column-index design sidesteps it, and (c) the genmipvxl + multi-mip-`column_offset` work needed before the body can land for real. |

### What "R4 effectively done" means

The 4 opticast-only oracle poses are bit-exact against voxlap C goldens
as of 2026-04-30 (commit `74172a2`). The other 8 oracle poses are not a
grouscan / opticast issue — each needs a feature that lives outside the
opticast pipeline: KV6 sprites (R6), per-voxel lighting, or `drawtile`
2D blits.

The remaining engine-correctness gaps in opticast itself are the three
deferred R4 items above (R4.4 textured sky, R4.5 multi-mip,
sideshademode high-byte). None of them break the 4 matching poses; they
matter for worlds outside the oracle's deliberately-narrow fixture.

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
- macOS / aarch64 / wasm CI — deferred to R9 / R10. Each architecture
  needs its own per-arch goldens (NEON ≠ x86 SSE bits, wasm v128 ≠
  either). The R8 CI is x86_64-linux only.

## Sync with voxlaptest

When voxlaptest ships an oracle change (new pose, refrozen hash, fixed bug),
reflect the matching change in roxlap's oracle as soon as roxlap implements
the affected code path. Both repos move on the `port` / `master` branches
respectively; PRs in either reference the other.
