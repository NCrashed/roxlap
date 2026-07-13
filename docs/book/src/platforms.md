# Platforms

One Cargo workspace covers Linux, Windows, macOS (x86_64 and arm64)
and the browser — no C dependencies, no per-platform source trees.
The per-architecture work is in SIMD kernels (SSE2 / NEON /
WebAssembly `simd128`, all via `core::arch`, with a scalar fallback
as the correctness reference) and in presentation, which is what this
chapter is about.

## Windowing: bring your own loop

`SceneRenderer::new` takes `Arc<W>` where `W: HasWindowHandle +
HasDisplayHandle + Send + Sync` — the
[raw-window-handle](https://docs.rs/raw-window-handle) traits, not any
specific windowing crate. The facade re-exports the bounds, so:

- **winit** — what the quickstart and every book example use.
- **SDL2** — the `roxlap-sdl-demo` crate is the worked example
  (including the wrapper that makes SDL's handle `Send + Sync`).
- **Anything else** (GLFW, a custom engine shell) — implement the two
  traits, pass the size explicitly, report changes via `resize`.

Presentation is backend-internal: the GPU backend owns a wgpu
swapchain; the CPU backend presents through softbuffer on native. You
never touch either.

## The browser

The same engine runs on wasm32 — both demos are live proof:

```sh
cd crates/roxlap-web        # engine demo (~360 KB wasm + 18 KB JS)
trunk serve                 # → http://localhost:8080

cd crates/roxlap-cave-web   # caves + crystals + carving + crumble
trunk serve --features audio   # optional: the full soundscape
```

Construction differs (there is no raw-window-handle in a browser):
`SceneRenderer::new_from_canvas_async(canvas, size, &opts)` — async
because the browser drives wgpu's adapter request through its event
loop; run it in a `wasm_bindgen_futures::spawn_local` task.

The backend story mirrors native, with one twist per layer:

- **GPU** = WebGPU via wgpu, where the browser supports it.
- **CPU fallback** = the same software DDA (with `simd128` batches),
  presented through a **WebGL2 blit** of the framebuffer — so even
  browsers without WebGPU get the engine. The constructor never
  fails; failures log to the console and fall through.
- `RequireGpu` degrades to `PreferGpu` (the async constructor has no
  error channel), and `supports(Feature::Capture)` is `false` —
  WebGPU cannot block on a readback.

**Threads.** CPU rendering fans out over Web Workers via
`wasm-bindgen-rayon` — a 4-core phone gets roughly 3× the
single-threaded frame rate. The catch is deployment: worker shared
memory needs `SharedArrayBuffer`, which browsers only enable under
**cross-origin isolation**. Your host must send two headers:

```text
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

Without them the thread pool silently doesn't spin up and you get the
single-threaded rate. (Building with threads also needs the nightly
toolchain pinned in the web crates — `-Z build-std` with atomics;
plain library consumers on stable are unaffected.) Per-host header
recipes, mobile touch controls, and the full setup live in
[`crates/roxlap-web/README.md`](https://github.com/NCrashed/roxlap/blob/master/crates/roxlap-web/README.md).

**Audio.** The `kira` backend runs on WebAudio unchanged — same
crate, same API as the [Audio chapter](audio.md). The one
browser-specific rule is the **autoplay policy: the gesture is the
constructor**. An `AudioContext` created outside a user-gesture
handler starts suspended and, on some browsers, never recovers — so
build the audio system *lazily inside* the first click / first touch
handler, never during init:

```rust,ignore
// In the click handler — the FIRST user gesture:
if state.audio.is_none() {
    state.audio = WebAudio::new();   // constructs KiraAudio HERE
}
```

Heavy main-thread frames can starve the browser's audio callback into
crackling; the demos accept that. If the whole demo stutters, check
the build profile first — `trunk serve` defaults to a *debug* build,
and a debug-built renderer is slow enough to read as an engine
problem (the cave demo's `Trunk.toml` sets `release = true` for this
reason).

**Picking.** `pick_depth` / `pick` read the GPU depth buffer back,
and WebGPU has no blocking readback — so the wasm GPU path answers
with **one frame of latency**: a call submits the readback for its
pixel and returns the latest *completed* result. Poll it (call again
next frame); the first call after a click usually returns `None`, and
the value that eventually arrives may belong to the previously
requested pixel. Everywhere else picking is synchronous:

| Backend            | `pick_depth` / `pick` semantics            |
|--------------------|--------------------------------------------|
| CPU (all targets)  | synchronous — in-memory z-buffer, free     |
| GPU, native        | synchronous — blocking readback, click-time cost |
| GPU, wasm          | **one-frame latency** — submit now, poll next frame |

Hosts that need a synchronous answer in the browser keep the CPU
backend, or skip the depth buffer entirely with the depth-free
`view_ray` + `Scene::raycast` composition from the
[Picking chapter](picking.md) — that path is identical on every
backend and target.

## What CI actually proves

Every push runs the full test suite on **x86_64 Linux, Apple Silicon
macOS and aarch64 Linux** — the DDA renderer's hash tests are plain
IEEE `f32` arithmetic and pass bit-identically on all three, so
"pixel-identical across architectures" is a gated claim, not an
aspiration. The wasm32 build is type- and lint-gated (`cargo clippy`
on the pinned nightly, including the browser audio stack); GPU tests
self-skip on runners without an adapter and run where one exists
(Metal on the mac runners).

## Troubleshooting

Symptoms first — each of these has burned somebody:

- **"I asked for the GPU but I'm on the CPU renderer"** — the
  fallback logs *why* through the [`log`] facade, and nothing shows
  without a logger: install one (`env_logger::init()` in the demos)
  before debugging anything else. `adapter_info()` returning `None`
  is the runtime tell.
- **No Vulkan on NixOS (and friends)** — outside a dev shell that
  provides it, wgpu needs `libvulkan.so.1` on `LD_LIBRARY_PATH`
  (match the ELF bitness). The repo's `flake.nix` wires it up;
  `nix develop` is the easy path.
- **Hybrid-GPU laptops (PRIME/offload)** — if the discrete GPU path
  misbehaves (driver/compositor sync bugs present exactly this way),
  `ROXLAP_GPU_POWER=low` steers wgpu to the integrated adapter
  without a rebuild; `high` forces discrete.
- **`cargo build` fails on `-lSDL2`** — only `roxlap-sdl-demo` links
  system SDL2, and it is excluded from the workspace's
  `default-members` for exactly this reason. Install the SDL2 dev
  package or just don't build `-p roxlap-sdl-demo`.
- **Web crates fail on stable Rust** — `roxlap-web` /
  `roxlap-cave-web` pin a nightly via `rust-toolchain.toml`
  (`-Z build-std` + atomics for the worker pool). Rustup honours the
  pin automatically inside those directories; library consumers on
  stable are unaffected.
- **Browser runs single-threaded** — the COOP/COEP headers above are
  missing. `crossOriginIsolated === false` in the console confirms
  it; the thread pool fails *silently* by design.

## Further reading

- [`PORTING-WASM.md`](https://github.com/NCrashed/roxlap/blob/master/docs/porting/PORTING-WASM.md)
  — the R10 series: SIMD batches, worker-pool threading, bundle-size
  history.
- [Rendering & backends](rendering.md) — the `BackendPreference`
  semantics this chapter's fallbacks hang off.
