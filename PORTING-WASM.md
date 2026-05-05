# roxlap — wasm SIMD + browser host (Substage R10)

Sub-substage roadmap and locked decisions for porting roxlap to
the browser via `wasm32-unknown-unknown` + `core::arch::wasm32`
SIMD intrinsics. Companion to [PORTING-RUST.md](PORTING-RUST.md)
(roadmap) and [PORTING-MULTICORE.md](PORTING-MULTICORE.md) (R12,
the most recent multi-stage doc precedent).

This document is the **start-of-stage brief**. A fresh-context
session should read this top to bottom before touching code; it
captures everything that's been decided up front so the
implementer doesn't re-litigate the easy questions.

## Status as of 2026-05-05

R12 multicore landed. R11.9 publish (0.1.0 to crates.io) is the
nominally-next stage but the user opted to do R10 first. R9
(ARM NEON) is still ahead.

Pre-R10 baseline (workspace state the implementer inherits):

- 6 crates: `roxlap-core` (engine), `roxlap-formats` (I/O + edit),
  `roxlap-cavegen` (procedural caves), `roxlap-cave-demo`
  (procedural cave game-ish demo), `roxlap-host` (engine demo,
  winit + softbuffer), `roxlap-oracle` (cross-engine hash
  oracle).
- Five SSE blocks (`#[cfg(target_arch = "x86_64")]` gated):
  - `scalar_rasterizer.rs::ScalarRasterizer::hrend` — line ~667
  - `scalar_rasterizer.rs::ScalarRasterizer::vrend` — line ~786
  - 3 SSE-test fns inside `scalar_rasterizer.rs::tests`
  - `sprite.rs::drawboundcubesse` — line ~992
- Non-x86_64 path of `drawboundcubesse` returns 0 (the
  `#[cfg(not(target_arch = "x86_64"))]` fallback is a stub).
- Hot-loop SSE intrinsics in use (~30 distinct):

  ```
  _mm_add_epi16   _mm_add_ps        _mm_adds_epi16
  _mm_cvtepi32_ps _mm_cvtsi128_si32 _mm_cvtsi32_si128
  _mm_cvtss_f32   _mm_cvttps_epi32  _mm_loadl_epi64
  _mm_loadu_ps    _mm_madd_epi16    _mm_max_epi16
  _mm_min_epi16   _mm_movehl_ps     _mm_movelh_ps
  _mm_mulhi_epu16 _mm_mul_ps        _mm_packs_epi32
  _mm_packus_epi16 _mm_rcp_ps       _mm_rcp_ss
  _mm_rsqrt_ps    _mm_rsqrt_ss      _mm_set1_ps
  _mm_setr_epi32  _mm_setr_ps       _mm_setzero_si128
  _mm_shufflelo_epi16 _mm_storeu_ps _mm_storeu_si128
  _mm_subs_epu16  _mm_unpackhi_epi64 _mm_unpacklo_epi32
  _mm_unpacklo_epi8
  ```

- Bench baseline (single-threaded x86_64 SSE, i7-12700H, 640×480):
  10.98 ms / 91 fps mean across the 12 oracle poses.
- Assets in tree:
  - `assets/oracle.vxl.gz` — 207 KB gzipped (~37 MB uncompressed)
  - `assets/sky.png` — 1.2 MB
  - `assets/coco.kv6` / `.kvx` — ~2 KB each
- Rayon is in roxlap-core's deps (R12). Will need `cfg(not(target_arch
  = "wasm32"))` gate or a workspace feature flag — bare wasm has
  no threads.

## Goal

Ship roxlap running in a modern browser (Chrome, Firefox, Safari)
via `wasm32-unknown-unknown` + WebAssembly SIMD (`simd128`). A
new `roxlap-web` crate hosts the demo; a static HTML + JS bundle
plus the wasm module is the deliverable. Browser perf benchmark
captures wasm-vs-native frame time. Oracle has a wasm goldens
file that diffs alongside the existing x86_64 goldens.

Stretch (deferred to R10.X follow-up): wasm threads via
SharedArrayBuffer + COOP/COEP headers + `wasm-bindgen-rayon` so
R12 multicore axes (per-strip, per-sprite, lighting bake) work
in-browser. Single-threaded wasm is the v1.

## Locked decisions

| # | Decision | Consequence |
|---|---|---|
| 1 | **`wasm32-unknown-unknown` target.** `wasm32-wasi` is server-side and not where browsers live; the web demo is the priority. | Browser-first dependency choices (web_sys, wasm-bindgen) over `tokio` / `wasi-bindings`. CI matrix gains one wasm32 build job. |
| 2 | **WebAssembly SIMD via `core::arch::wasm32` (`simd128`).** Required for parity with x86_64's SSE batches. Browser support is universal in Chrome 91+/Firefox 89+/Safari 16.4+ (2023+). | Five new `#[cfg(target_arch = "wasm32")]` SIMD blocks alongside the existing x86_64 ones — same shape, parallel implementation. Old browsers without simd128 fall back to the scalar tail in each rasterizer. |
| 3 | **Single-threaded wasm for v1.** Rayon doesn't compile on bare `wasm32-unknown-unknown` (no `std::thread`). Multi-threaded wasm requires `wasm-bindgen-rayon` + SharedArrayBuffer + COOP/COEP, which carries deployment complexity. | All R12 parallel paths gate to single-thread on wasm32 via cfg. `pool.n_threads() == 1` always; `update_lighting` falls back to a sequential outer loop on wasm; `draw_sprites_parallel` becomes equivalent to a `for sprite in sprites` loop. Re-enable in R10.X follow-up if there's demand. |
| 4 | **`wasm-bindgen` + `web-sys` for the JS bridge.** Industry-standard; integrates with `winit`'s wasm backend (already a transitive dep via `roxlap-host`). | One additional dep (`wasm-bindgen` + `web-sys` features) on the `roxlap-web` crate only. roxlap-core / roxlap-formats stay JS-bridge-free. |
| 5 | **Trunk as the build tool for `roxlap-web`.** `trunk serve` watches Rust + HTML and hot-reloads in the browser; closest to the cargo dev-loop ergonomics. wasm-pack is npm-package-shaped, overkill for a static demo. | One devDependency (a tool, not a Rust crate). README documents `cargo install trunk` + `trunk serve` workflow. CI uses raw `wasm-bindgen-cli` for bundle production (no trunk needed in CI). |
| 6 | **Bundle the assets via `include_bytes!`, not fetch.** `oracle.vxl.gz` is 207 KB; `coco.kv6` 2 KB; sky.png is 1.2 MB. Total bundle ~1.5 MB before wasm itself. Fetching them adds an async-init dance + CORS handling. Cleaner to embed. | wasm binary grows by ~1.5 MB (already-gzipped assets). The raw oracle.vxl.gz / kv6 / png pass through to the existing parser entry points; no async loading code. R10.X can add async asset loading if the bundle gets too big. |
| 7 | **`_mm_rsqrt_ps` and `_mm_rcp_ps` have no wasm SIMD analogue.** Wasm SIMD has `f32x4_sqrt` (full precision) but no rsqrt / rcp approximation. Use `1.0 / sqrt(x)` for rsqrt and `1.0 / x` for rcp on wasm. | **Wasm goldens differ from x86_64 goldens** (different bit pattern in the projected z and sprite vertex projection). New `wasm-hashes.txt` file, separate from the in-tree `tests/golden-hashes.txt`. CI's wasm job diffs against its own goldens. Documented as the same kind of arch-divergence as x86_64-vs-aarch64 (R9). |
| 8 | **`roxlap-web` crate is engine-demo-only (R10's deliverable).** The cave-demo on web is a stretch goal — it needs winit-wasm event handling for real-time edits + bullets. Engine-only demo is enough to prove wasm SIMD works + ships a public-facing browser version. | One new crate. Cave-demo on web tracked as a separate R10.X follow-up. |
| 9 | **Oracle wasm gate via `wasm-bindgen-test`** (headless V8/SpiderMonkey via Node, not headless browser). Renders one pose, hashes the framebuffer, compares against frozen wasm goldens. CI matrix runs alongside the x86_64 oracle. | One new dev-dependency. Avoids the heavyweight chromedriver / geckodriver setup. |

## Sub-substage roadmap

| # | Scope | Estimate | Validation |
|---|---|---|---|
| **R10.0** | Plan doc (this file) + workspace plumbing for wasm32: cfg-gate `rayon` out for wasm targets in `roxlap-core/Cargo.toml`; verify `cargo build --target wasm32-unknown-unknown -p roxlap-core -p roxlap-formats` is green WITH the rayon parallel paths cfg-stubbed. No SIMD yet — just scalar wasm builds. | 1 d | `cargo build --target wasm32-unknown-unknown --workspace --exclude roxlap-host --exclude roxlap-cave-demo` green; `cargo test --workspace` (native) still 361/361 green. |
| **R10.1** | Wasm scalar render verification: write a tiny wasm-bindgen-test that allocates a framebuffer, calls `Engine::render` over the bundled oracle world (via `include_bytes!`) for the `north` pose, FNV-hashes the output, prints the hash. Establishes the wasm scalar baseline; SIMD comes later. | 1.5 d | wasm-bindgen-test runs in node + reports a stable hash. (No goldens yet; just stable across reruns.) |
| **R10.2** | New `roxlap-web` crate: wasm-bindgen `start()` entry, `<canvas>` setup via web_sys, ImageData blit per frame, requestAnimationFrame loop, keyboard / mouse-look event handlers (winit-wasm's API or raw web_sys). Renders the oracle world with cam controls. `index.html` + trunk config in `crates/roxlap-web/`. | 2–3 d | `trunk serve` from `crates/roxlap-web/` opens a working demo on `localhost:8080`. Frame timer printed to console. |
| **R10.3** | Wasm SIMD intrinsics — five new `#[cfg(target_arch = "wasm32")]` SIMD batches, mirror of the existing x86_64 ones (hrend / vrend × no-fog / fog + drawboundcubesse). `1.0 / sqrt(x)` substitutes for `_mm_rsqrt_ps`; `1.0 / x` for `_mm_rcp_ps`. Verify each batch matches the scalar tail's output bit-for-bit on a unit fixture (where wasm SIMD's full-precision sqrt agrees with scalar `f32::sqrt` — they should). | 5–7 d | Every wasm SIMD batch test passes (output matches scalar tail). Browser bench shows ~2× speedup vs scalar wasm. |
| **R10.4** | Wasm oracle goldens. New `crates/roxlap-oracle/tests/wasm-goldens.txt`. CI matrix job: `wasm-bindgen-test` runs the 12 oracle poses, dumps hashes, diffs against the file. Frozen on first green run. | 1 d | CI's wasm-oracle job green; goldens committed; visual spot-check that the wasm renders look identical to native to the eye. |
| **R10.5** | Browser perf bench + README polish. New `roxlap-web` keyboard binding (`B`?) that runs an N-frame timing loop and dumps min/p50/mean to console. README update with wasm bundle size, install / serve instructions, and a perf note (wasm typically ~1.5–2× slower than native; same algorithm, different instruction set). | 0.5 d | Reproducible bench numbers in the browser; README documents install / serve / bench workflow. |

**Total scope**: ~2–3 weeks for R10.0–R10.5. R10.X follow-ups
(wasm threads, cave-demo on web, async asset loading) tracked as
post-0.1.0 work.

## Where wasm SIMD differs from x86_64 SSE

The five SSE blocks (~30 intrinsics) all need wasm equivalents.
The trickiest mismatches:

### Approximation instructions

| x86_64 SSE | wasm SIMD | Mitigation |
|---|---|---|
| `_mm_rsqrt_ps` (12-bit reciprocal sqrt) | none | `1.0 / f32x4_sqrt(x)` (full precision; bits differ) |
| `_mm_rcp_ps` (12-bit reciprocal) | none | `f32x4_div(splat(1.0), x)` (full precision) |

These produce **different bytes** than x86_64. Wasm goldens
diverge from x86_64 goldens. This is the same arch-divergence
that R9 NEON will face (NEON's `vrsqrteq_f32` is 8-bit
approximation, _further_ from SSE's 12-bit) — a known property of
voxlap's per-arch ports. Document, freeze, move on.

### Integer SIMD that maps cleanly

Most integer SIMD has direct wasm analogues:

| x86_64 | wasm |
|---|---|
| `_mm_add_epi16` | `i16x8_add` |
| `_mm_adds_epi16` | `i16x8_add_sat` |
| `_mm_max_epi16` | `i16x8_max` |
| `_mm_min_epi16` | `i16x8_min` |
| `_mm_subs_epu16` | `u16x8_sub_sat` |
| `_mm_packus_epi16` | `u8x16_narrow_i16x8` |
| `_mm_packs_epi32` | `i16x8_narrow_i32x4` |
| `_mm_setr_epi32` | `i32x4(a, b, c, d)` (constructor) |
| `_mm_setzero_si128` | `i32x4_splat(0)` |
| `_mm_storeu_si128` | `v128_store` (no alignment requirement) |
| `_mm_loadu_ps` / `_mm_storeu_ps` | `v128_load` / `v128_store` (loaded as `v128`, treat as `f32x4`) |
| `_mm_shufflelo_epi16` | `i16x8_shuffle::<lane indices>` |
| `_mm_unpackhi/lo_*` | `i*_shuffle` with the right lane indices |

### Multiply-related

| x86_64 | wasm | Notes |
|---|---|---|
| `_mm_mul_ps` | `f32x4_mul` | clean |
| `_mm_madd_epi16` (multiply + horizontal-add 16→32) | `i32x4_extmul_low_i16x8` + `_extmul_high_i16x8` + add | two-step |
| `_mm_mulhi_epu16` (high 16 of u16×u16) | `i32x4_extmul_low_u16x8` + shift + narrow | two-step |

### Float conversion

| x86_64 | wasm |
|---|---|
| `_mm_cvtepi32_ps` | `f32x4_convert_i32x4` |
| `_mm_cvttps_epi32` | `i32x4_trunc_sat_f32x4` |

### Move-quad / extract

| x86_64 | wasm |
|---|---|
| `_mm_movehl_ps`, `_mm_movelh_ps` | `f32x4_shuffle::<lane indices>` |
| `_mm_cvtsi32_si128` (scalar i32 → low lane) | `i32x4(x, 0, 0, 0)` |
| `_mm_cvtsi128_si32` (extract low i32) | `i32x4_extract_lane::<0>` |
| `_mm_cvtss_f32` (extract low f32) | `f32x4_extract_lane::<0>` |

## R10.0 setup details

`rayon` doesn't compile on `wasm32-unknown-unknown`. Two options:

**Option A**: Cargo target-conditional dependency.

```toml
# crates/roxlap-core/Cargo.toml
[target.'cfg(not(target_arch = "wasm32"))'.dependencies]
rayon = { workspace = true }
```

Then everywhere rayon is used, `#[cfg(not(target_arch = "wasm32"))]`
gates the parallel branch with a sequential fallback for wasm.

**Option B**: Workspace feature flag.

```toml
[features]
default = ["parallel"]
parallel = ["rayon"]
```

Wasm builds with `--no-default-features`. Slightly more
explicit but requires every consumer to know about the flag.

**Recommendation: Option A.** It's transparent — `cargo build
--target wasm32-...` just works without remembering to disable a
feature.

The cfg gates needed:

- `crates/roxlap-core/src/opticast.rs::run_strip_parallel` — the
  whole function body is wasm-incompatible (calls
  `rayon::par_iter_mut`).
- `crates/roxlap-core/src/world_lighting.rs::update_lighting` —
  the `(y0p..y1p).into_par_iter()` call.
- `crates/roxlap-core/src/sprite.rs::draw_sprites_parallel` —
  the par_iter call.

For each, the fallback is a sequential `for` loop. Trivial to
write — the parallel and sequential code paths share the same
inner body.

## R10.2 web host shape

```rust
// crates/roxlap-web/src/lib.rs (sketch)
use wasm_bindgen::prelude::*;
use web_sys::{HtmlCanvasElement, ImageData};

#[wasm_bindgen(start)]
pub fn start() -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    let canvas: HtmlCanvasElement = /* getElementById */;
    let ctx = canvas.get_context("2d")?.unwrap().dyn_into()?;

    let mut engine = Engine::new();
    let vxl = parse_oracle();              // include_bytes! + gunzip + parse
    let sky = parse_sky();                  // include_bytes! + png decode
    engine.set_sky(sky);

    let mut state = State { engine, vxl, /* fb, zb, pool, etc. */ };

    // requestAnimationFrame loop. Each call:
    //   1. integrate input (key state, mouse delta);
    //   2. opticast + sprites + drawtile into fb;
    //   3. blit fb to ImageData and putImageData onto the canvas.
    let f = Rc::new(RefCell::new(None));
    let g = f.clone();
    *g.borrow_mut() = Some(Closure::wrap(Box::new(move || {
        state.frame();
        request_animation_frame(f.borrow().as_ref().unwrap());
    }) as Box<dyn FnMut()>));
    request_animation_frame(g.borrow().as_ref().unwrap());

    Ok(())
}
```

Trunk's `index.html`:

```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>roxlap</title>
    <link data-trunk rel="rust" data-bin="roxlap-web" />
</head>
<body>
    <canvas id="roxlap-canvas" width="640" height="480"></canvas>
</body>
</html>
```

`Cargo.toml` (snippet):

```toml
[package]
name = "roxlap-web"
crate-type = ["cdylib"]   # for wasm-bindgen

[dependencies]
roxlap-core = { path = "../roxlap-core", version = "0.1" }
roxlap-formats = { path = "../roxlap-formats", version = "0.1" }
wasm-bindgen = "0.2"
js-sys = "0.3"
web-sys = { version = "0.3", features = ["Window", "Document",
    "HtmlCanvasElement", "CanvasRenderingContext2d", "ImageData",
    "KeyboardEvent", "MouseEvent", "Performance"] }
console_error_panic_hook = "0.1"

[dev-dependencies]
wasm-bindgen-test = "0.3"
```

## Memory cost

Wasm linear memory + bundled assets:

| component | size |
|---|---|
| oracle.vxl.gz (gzipped) | 207 KB |
| oracle.vxl (decoded into memory) | ~37 MB |
| sky.png (compressed) | 1.2 MB |
| sky decoded (640×320 BGRA approx) | ~800 KB |
| coco.kv6 | 2 KB |
| Framebuffer 640×480 u32 | 1.2 MB |
| Zbuffer 640×480 f32 | 1.2 MB |
| ScratchPool (1 slot at 640×480) | 7.6 MB |
| **Wasm linear memory total** | **~50 MB** |

Browser tabs comfortably allocate 100s of MB, so this is fine.
The wasm binary itself plus the gzipped assets totals ~1.5 MB of
download (most of that is sky.png). Acceptable for a demo —
fetches in under a second on broadband.

## Risks

### Performance — wasm overhead

Wasm typically runs 1.5–2× slower than equivalent native code on
the same machine. The native baseline is 10ms / frame at 640×480;
wasm should land at ~15–20 ms / frame. Above 60 fps still, but
visibly slower than the desktop demo.

Mitigation: lower the demo's default resolution to 480×320 if
needed, or cap at 30 fps explicitly. R10.5's bench step measures
exactly this delta.

### `winit`-wasm vs raw web-sys

`winit` has wasm support but it's quirky — input events come
through differently than native, and `winit`'s redraw loop on
wasm is essentially `requestAnimationFrame`-driven. Two options:

- Use `winit` wasm backend so `roxlap-web` shares input code with
  `roxlap-host`. Pro: code reuse. Con: extra deps + winit's
  wasm path is less polished than native.
- Use raw `web_sys` event handlers + `requestAnimationFrame`.
  Pro: explicit, minimal deps. Con: separate input code from
  the native host.

**Recommendation**: raw web_sys. The native `roxlap-host` is
already its own input handler; sharing wouldn't save much. wasm
gets a tighter, less-deps web bundle.

### Asset loading — `include_bytes!` blast radius

`include_bytes!("../../../assets/oracle.vxl.gz")` works fine for
207 KB. `include_bytes!("../../../assets/sky.png")` is 1.2 MB —
fine but not great. If the demo grows assets (multiple worlds,
many KFA sprites), include_bytes makes the wasm binary
proportionally larger.

Mitigation: deferred to R10.X. For v1, ship the oracle scene
(matches roxlap-host's content). Async asset fetching is a
generic pattern (`fetch` + `arrayBuffer` + `Vec<u8>`), tractable
later.

### CI complexity

The CI matrix gains a wasm target. wasm-bindgen-test in node is
self-contained — Cargo + node + cargo-binstall'd wasm-bindgen-cli
suffices. No headless browser needed.

### Wasm SIMD detection at runtime

If the user's browser doesn't support simd128 (very old browser),
the wasm module won't even load. Two options:

- Compile with `+simd128` always, accept that ancient browsers
  fail at the loader stage.
- Ship two builds: `roxlap-web-simd` (with simd128) and
  `roxlap-web-scalar` (without), JS picks at load time.

**Recommendation**: simd128-only build for v1. Browser support
is universal (94 %+ globally per caniuse). R10.X can add the
scalar build if there's demand.

## R10.4 oracle goldens shape

```sh
$ cargo test --target wasm32-unknown-unknown -p roxlap-oracle
running 1 test
test wasm_oracle_diff ... ok
$ cat crates/roxlap-oracle/tests/wasm-hashes.txt
north  <hash_a>
east   <hash_b>
... 12 entries ...
```

The hashes will differ from `tests/golden-hashes.txt` because
wasm SIMD's `1.0 / sqrt(x)` ≠ x86_64's `_mm_rsqrt_ps`. They'll
also differ from the planned R9 NEON goldens (`vrsqrteq_f32`'s
8-bit approximation differs from both).

The golden file is committed alongside `golden-hashes.txt`. CI
diffs the wasm-side hashes only.

## Bench projection

| variant | ms / frame | fps | speedup vs scalar wasm |
|---|---|---|---|
| **scalar wasm** | ~30–40 | 25–30 | 1× |
| **wasm SIMD (simd128)** | ~15–20 | 50–65 | ~2× |
| **native x86_64 SSE (R12.1 baseline)** | 10.98 | 91 | (different machine class) |

These are projections; R10.5 measures actuals. Wasm's typical
slowdown vs native is 1.5–2× on math-heavy code, less on
memory-heavy code.

## Reading list (for the implementing session)

In order:

1. **This document** — the start-of-stage brief. Top to bottom.
2. **PORTING-MULTICORE.md** — precedent for multi-stage
   planning; especially the "Locked decisions" + "Risks"
   structure. R12 also touched the same SSE blocks R10 ports;
   the patterns transfer.
3. **PORTING-RUST.md** § R5 — explains the structure of the
   four scanline rasterizers (`hrend` / `vrend` × no-fog / fog)
   and how the SSE batches relate to the scalar tail.
4. `crates/roxlap-core/src/scalar_rasterizer.rs` lines ~530–860
   — the four SSE blocks (no-fog hrend, fog hrend, no-fog vrend,
   fog vrend). Read each block + its scalar tail together; the
   wasm port mirrors this structure.
5. `crates/roxlap-core/src/sprite.rs::drawboundcubesse` —
   the fifth SSE block. Has different SIMD-shape (i16 packing,
   alpha modulation, mm5 cross-call tail) — the trickiest port
   of the five.
6. `crates/roxlap-host/src/main.rs` — input + frame-loop pattern
   the web demo mirrors with `requestAnimationFrame`.
7. [WebAssembly SIMD spec](https://webassembly.github.io/spec/core/syntax/instructions.html#vector-instructions)
   — the actual instruction set R10.3 ports to. The `core::arch::wasm32`
   docs are an exact mirror.
8. [`wasm-bindgen` book](https://rustwasm.github.io/wasm-bindgen/)
   — JS bridge mechanics; the canvas + ImageData blit lives at
   the boundary between Rust and JS.
9. [Trunk's docs](https://trunkrs.dev/) — build tool for the
   web demo crate.

## Open decisions for the implementing session

1. **Will winit-wasm be used or raw web_sys?**
   Recommendation: raw web_sys (above). Caller can override.

2. **Demo content: oracle world only, or include the cave-demo
   procedurally?**
   Cave-demo on web is a stretch. Engine demo (oracle world +
   sprites + sky) is enough for R10's deliverable.

3. **Default web demo resolution?**
   480×320 if wasm perf is too slow at 640×480. Measure in R10.3
   before committing.

4. **Asset cache strategy if `include_bytes!` becomes too big?**
   Async `fetch` per-asset, with a Promise.all init step before
   the render loop starts. Defer to R10.X.

5. **Wasm threads (R10 stretch or R10.X follow-up)?**
   Definitively R10.X — out of scope for v1. Document in this
   doc that R12's parallel paths are sequential on wasm.

## Out of scope (R10.X follow-ups)

- **Wasm threads** via `wasm-bindgen-rayon` + SharedArrayBuffer
  + COOP/COEP headers. Re-enables R12's parallel paths in
  browser. Real browser-deploy complexity (COOP/COEP needs
  server-side header config; pages.dev / netlify don't all
  support this).
- **Cave demo on web** — would showcase real-time edits +
  bullets, but doubles the input-handler / state-mgmt work.
- **Async asset loading** — fetch + decompress at startup
  instead of `include_bytes!`. Necessary if assets grow past
  ~5 MB.
- **Older-browser fallback build** without simd128. Probably
  unnecessary given current browser-support stats.
- **WebGL / WebGPU framebuffer blit** — currently using 2D canvas
  + `putImageData`. WebGL is faster for fb blits but adds JS
  surface area. R10.X if frame-blit becomes a bottleneck.
- **Mobile / touch input** — keyboard + mouse only for v1.

## How to apply

When the user says "let's do R10", the implementing session
reads this doc top to bottom, then starts at R10.0 (workspace
plumbing). Each sub-substage lands as its own commit. R10.5's
README polish closes the stage.

If R10 hits a fundamental algorithmic blocker (e.g., wasm SIMD
intrinsic that doesn't work as documented), pause and update the
"Risks" section before proceeding — the doc captures *current
understanding*, not gospel.
