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

cd crates/roxlap-cave-web   # procedural caves + carving (~130 KB wasm)
trunk serve
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
