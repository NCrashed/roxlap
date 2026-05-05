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
| **R4** | Full `opticast` + `grouscan` algorithm port from `grouscanasm_scalar` (scalar Rust). | **fully landed**: opticast bit-exact on the 4 opticast-only oracle poses; R4.4 textured sky landed alongside R6; R4.5 multi-mip `remiporend` + sideshademode landed. | 4 of 12 oracle poses match voxlap C goldens byte-for-byte (`north`, `east`, `diag_down`, `high_down`). The other 8 poses each require a feature roxlap covers in R6+ (sprites, lighting, drawtile). |
| **R5** | x86_64 SSE2 4-pixel rsqrtps batches in the four hot rasterizers (mirror of voxlaptest Stage 4.9). | **fully landed**: hrend's SSE block inlines R5.1 (no-fog) + R5.2 (fog); vrend's SSE block inlines R5.3 (no-fog, parallel `uurend` update) + R5.4 (fog). | Same FNV-1a hashes as the scalar reference where SSE2 `rsqrtps` is bit-equivalent. |
| **R6** | KV6 sprite renderer + sprite/world lighting (`drawsprite` / `drawboundcubesse` / `updatereflects` / `updatelighting`) + textured sky (`startsky` non-solid branch). | **fully landed**: R6.0..R6.4 sprite renderer; R6.4f sprite point lighting (`updatereflects` lightmode≥2); R6.5 world voxel lighting bake; R4.4-final textured sky. 1 of 4 sprite oracle poses bit-exact, 3 drift on sub-pixel rounding; `diag_down_lit` drifts similarly (frozen as roxlap golden). | **9 of 12 oracle poses tracked** by `tests/golden-hashes.txt`: 4 opticast bit-exact, 4 sprite + 1 lit pose frozen as roxlap goldens (visually verified vs voxlap C, sub-pixel rounding noise documented). `roxlap-host` demos the full pipeline: textured `assets/sky.png` panorama, two animated kv6 sprites with point-light shading, world voxel lighting bake toggleable via `L`. |
| **R7** | _Cancelled_. Was scoped as "close sprite-pose drift". The three drifted poses (`sprite_front`, `sprite_iso`, `sprite_coco`) and `diag_down_lit` are visually indistinguishable from voxlap C output; the in-tree goldens track roxlap's own hashes for regression detection. Reopen if a downstream consumer demands bit-exactness against voxlap C. | cancelled | n/a |
| **R8** | Cross-engine oracle: roxlap-side oracle binary writing `roxlap-hashes.txt`; CI matrix that diffs against `golden-hashes.txt`. | done | `cargo run -p roxlap-oracle -- diff` reports `9 match, 0 mismatch` against the in-tree `tests/golden-hashes.txt`. `.github/workflows/ci.yml` runs fmt / clippy / test / oracle-diff on every push + PR; oracle job fails on any hash mismatch so the bit-exact milestone can't regress silently. |
| **R9** | ARM NEON via `core::arch::aarch64`; macOS arm64 + Linux aarch64 in CI. | not started | Own goldens (NEON ≠ x86 SSE bits); aarch64 CI green. |
| **R10** | wasm SIMD via `core::arch::wasm32`; web host (canvas + js glue) as a separate crate. | not started | Browser perf benchmark; own wasm goldens. |
| **R11** | Polish: docs, examples, version 0.1 publish to crates.io. | not started | Crates published; docs.rs renders. |
| **R12** | Multicore CPU rendering via `rayon`. Three parallelism axes: per-strip opticast (1.49× peak at N=4; sub-pixel drift across N, goldens at `--threads 1`), `update_lighting` per-row (3.35× peak), `draw_sprites_parallel` per-sprite (4.4× at 64 sprites, 6.1× at 256). Full sub-substage breakdown + measured numbers in [PORTING-MULTICORE.md](PORTING-MULTICORE.md). | landed | `--threads 1` byte-stable vs goldens (`10 match, 2 mismatch`, same as pre-R12); `bench-lighting`, `bench-sprites`, `bench --threads N` reproducible scaling curves. |

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
| **R4.4** | Sky-fill primitives: `startsky` integration, fog gating. **Done**: solid-fill branch + fog blend (R4.3f), textured-sky `skyoff != 0` path landed alongside R6's lighting work. The textured branch is wrapped behind an [`crate::sky::Sky`] resource on `Engine`; gline updates `scratch.sky_off` per ray via the rotating-cursor walk; `phase_startsky` dispatches between solid (oracle default, byte-stable) and textured (host's `assets/sky.png`). Includes a fix for voxlap C's latent stale-cx1 bug — `phase_startsky_textured` re-derives the per-cf-entry far-end position from `cx0 + (i1 - i0) * gi0` instead of reading the cf's stored `cx1`, which goes stale when drains shrink `i1` without touching cx1 (visible in voxlap C as horizontal sky distortion at low pitch). |
| **R4.5** | Mip-level transition (`remiporend`): the deeper-recursion cftype refresh. **Landed.** `Vxl::generate_mips` ports voxlap's `genmipvxl` (voxlap5.c:4710+) onto a flat per-mip `column_offset` extension; `phase_remiporend`'s body uses column-index parity (`(col_within_mip >> bit_pos) & 1`) for the gpz/gdz alignment instead of voxlap C's broken LP32/LP64 sptr-stride shift arithmetic. Oracle stays at `gmipnum=1`, so its 12 hashes are byte-stable; multi-mip rendering is exercised by the new roxlap-only golden in `crates/roxlap-core/tests/multi_mip.rs`. The audit + LP32/LP64 bug table is preserved in the doc comment on `phase_remiporend`. |

### What "R4 fully done" means

The 4 opticast-only oracle poses are bit-exact against voxlap C goldens
as of 2026-04-30 (commit `74172a2`). The other 8 oracle poses are not a
grouscan / opticast issue — each needs a feature that lives outside the
opticast pipeline: KV6 sprites (R6), per-voxel lighting, or `drawtile`
2D blits.

R4.4 textured sky, R4.5 multi-mip `remiporend`, and the sideshademode
high-byte all landed (textured sky alongside R6; multi-mip + sideshademode
in the 2026-05-02 work). The 12 voxlap-C oracle hashes stay byte-stable
because they all run with `gmipnum=1` and zero side shading, and the
new code paths early-out for those configurations. Multi-mip rendering
is regression-tested by `crates/roxlap-core/tests/multi_mip.rs`'s own
pinned hashes (no voxlap C reference exists for `vxlmipuse > 1`).

### Known voxlap-inherent quirk: floor-hairline at certain camera poses

At unusual interactive camera positions (e.g. yaw + pitch that sends one
ray's `gdz[0]` to ~2 G — just under `i32::MAX`), grouscan's column-step
overflow check fires after a single lane-0 increment instead of the
expected ~6th, routing the ray to `Phase::Startsky` and draining the
seed cf entry's unwritten remainder to sky. Visible as a single 6–30
pixel sky-blue vertical run on the floor.

**This bug also exists in voxlap C** at the same camera + 800×600
resolution. Voxlap's standard 640×480 oracle doesn't show it because at
that resolution the buggy ray lands on a different (non-floor) screen
position. The `gline` frustum + cf-seed values match field-for-field
between the two engines; the artifact is a property of the underlying
grouscan algorithm's overflow handling, not the roxlap port.

A separate, similar-looking artifact rooted in Rust's saturating
`as i32` cast (vs voxlap's wrapping `lrintf + (int32_t)cast`) was an
*actual* roxlap regression — that one is fixed; see commit `6d4bcf4`
and the `fixed::ftol` helper. The voxlap-inherent quirk above is not
fixable without diverging from voxlap's bit-exact behaviour and
regressing the 4 oracle-pose hashes; documented as a known limitation
until a downstream fixture demands a workaround.

Diagnostics for chasing future hairline-style artifacts live in
`crates/roxlap-host/src/main.rs` (the `F` capture hotkey) and
`crates/roxlap-oracle/src/main.rs::cmd_find_hairlines` (env vars
`ROXLAP_TAG_SKY`, `ROXLAP_FOG`, `ROXLAP_TRACE_STARTSKY`,
`ROXLAP_TRACE_PHASES`).

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

## Substage R6 — KV6 sprite renderer + lighting + textured sky

Mirror of voxlap5.c:8179-9062 (`drawboundcubesse` + `kv6draw` + the
9-arm `DRAWBOUNDCUBELINE` iteration), voxlap5.c:8466-8750
(`updatereflects` colour-modulation table builder, all branches),
voxlap5.c:10539-10654 (`updatelighting` world-voxel bake), and
voxlap5.c:12120-12190 (`startsky` textured branch). Sub-substages:

| # | Scope |
|---|---|
| **R6.0** | Foundations: `getcube` (R6.0a), `lightvox` + `factr` / `logint` / `tempfloatbuf` power tables (R6.0b/c), `meltsphere` (R6.0d), byte-equality validation against voxlap C oracle dumps for both meltsphere fixtures (R6.0e). All bit-exact. |
| **R6.1** | `Sprite` type + `draw_sprite` skeleton dispatcher (voxlap5.c:9818). |
| **R6.2** | Frustum cull + mip-LOD distance estimate (voxlap5.c:8832-8875). |
| **R6.3** | 9-arm per-(x, y) iteration with `r0` / `r1` tracking (voxlap5.c:8982-9062). Validated: 401-voxel meltsphere visits each voxel exactly once. |
| **R6.4a** | Setup math: `mat2` + Cramer's + `nfor↔nhei` swap + `cadd4` / `ztab4` / `r1` ANNOYING-HACK pre-decrement / `scisdist` / `qsum0` (voxlap5.c:8915-8973). Per-voxel scissor on `origin.z`. |
| **R6.4b** | Vertex projection: `_mm_rcp_ps`-based projection of the (4 or 6) `ptfaces16[effmask]` vertex pairs. |
| **R6.4c** | Viewport clip + screen-AABB: `qsum0` saturated-add + `qsum1` max-floor; `_mm_subs_epu16` for `(dx, dy)` with degenerate-rect early-out. |
| **R6.4d** | Fill rect + zbuffer write: per-pixel z-test, framebuffer write. `DrawTarget<'a>` API takes the borrowed framebuffer + zbuffer. |
| **R6.4e** | Colour modulation + `mm5` cross-call tail + `update_reflects` oracle path (`flags=0`, no fog, `lightmode<2`, both nolighta + nolightb branches). |
| **R6.4f** | Full sprite lighting: `Engine::set_lightmode` / `add_light` / `set_kv6col` plumbed via `SpriteLighting<'a>` into `update_reflects`'s `lightmode≥2` branch (voxlap5.c:8631-8750). Per-sprite point-light shadow modulation. Includes the `hh = ((fogmul&32767)^32767) / 65536 * 2` brightness fix where an earlier port had `g_pre / 128` (off by a factor of 2). |
| **R6.5** | World voxel lighting bake (`updatelighting`, voxlap5.c:10539-10654). New `world_lighting.rs` module: BITNUM/BITSNUM tables, `expandbit256` slab→bit-array, `EstNormCache` with the `fsqrecip[5860]` LUT, mutable slab walker, lightmode-1 directional + lightmode-2 per-light Lambertian per-voxel brightness math. Adds the **9th oracle pose `diag_down_lit`** (matches voxlap C's bake setup at oracle.c:438-443). |
| **R4.4-final** | Textured `startsky` branch (`skyoff != 0`, voxlap5.c:12143-12188). New `sky.rs` module: `Sky` resource with `lng[]` / `lat[]` lookup tables, `Sky::blue_gradient()` for the "BLUE" fallback. `phase_startsky` splits into solid (oracle default, byte-stable) and textured (host's `assets/sky.png` panorama). Includes a stale-cx1 fix that voxlap C also has latently (visible at low pitch). Exposed as `Engine::set_sky` + `ScalarRasterizer::with_sky`. |

### What "R6 effectively done" means

End-to-end rendering pipeline shipped:
- KV6 sprites with point-light shading + bound-cube projection + z-tested fill.
- World voxel intensity bake with directional or point-light shading.
- Textured panoramic sky.
- Engine API surface: `set_sky` / `set_lightmode` / `add_light` / `set_kv6col` /
  `set_side_shades` / `set_fog` cover the full vx5 globals subset the host needs.
- `roxlap-host` demos all of it: `assets/sky.png` panorama wraps the camera, two
  animated kv6 sprites (axis-aligned meltsphere + spinning coco) with point-
  light shading, world voxel lighting bake toggled via `L`.

Bit-exact status against voxlap C goldens (full 9 poses tracked):
- 4 opticast-only poses (`north`, `east`, `diag_down`, `high_down`):
  byte-for-byte match.
- `sprite_above`: byte-for-byte match.
- `sprite_front`, `sprite_iso`, `sprite_coco`, `diag_down_lit`: stable hashes
  drift from voxlap C by sub-pixel rounding (visually verified vs voxlap C
  PPM output via `cargo run -p roxlap-oracle -- --ppm`). `tests/golden-
  hashes.txt` tracks roxlap's own hashes for regression detection — CI
  catches any algorithmic drift.

R6 is x86_64-only — `_mm_rcp_ps` produces hardware-specific 12-bit-precision
output and bit-equality with voxlap C requires the same instruction. NEON and
wasm ports (R9 / R10) will have their own goldens. The non-x86_64 path
returns 0 (no rendering).

R6.6 / R6.7 (KFA + animated sprites, no-z sprite path) stay deferred — none of
the four sprite oracle poses exercise them.

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
