# roxlap

An independent Rust voxel engine that reads
[Ken Silverman's Voxlap](http://advsys.net/ken/voxlap.htm) file formats,
grown into a small **voxel-scene engine**. The CPU renderer is a
clean-room per-pixel 3D-DDA over a brickmap — it runs anywhere with no
GPU and no C dependency; an **optional WGPU compute-shader renderer**
sits alongside it behind one unified facade with automatic CPU fallback.
On top is a **multi-grid scene graph** — f64 world placement +
quaternion rotation, chunk streaming + procedural generation, serde
snapshots, and real-time edit + screen→world picking APIs. One Cargo
workspace, Linux / macOS / Windows + wasm, idiomatic safe Rust with
per-architecture SIMD. Dual MIT/Apache-2.0, commercial use included.

![sample render from roxlap](https://raw.githubusercontent.com/NCrashed/roxlap/master/docs/screenshot.png)

## What is Voxlap?

Voxlap is the voxel rendering engine [Ken Silverman](http://advsys.net/ken/)
wrote in the early 2000s, after the Build engine that powered *Duke Nukem
3D*. It draws volumetric voxel terrain plus animated kv6 sprites entirely
on the CPU, using Ken's classic "raycast columns + scanline fill"
algorithm — no GPU, no shaders. Cult-favourite games like
[Voxelstein 3D](https://en.wikipedia.org/wiki/Voxelstein_3D),
[Ace of Spades](https://en.wikipedia.org/wiki/Ace_of_Spades_\(video_game\)),
and Ken's own *Slab6* / *Voxed* shipped on top of it.

roxlap reads the same `.vxl` (worlds) / `.kv6` / `.kvx` (sprite voxels)
/ `.kfa` (sprite animation rigs) files Ken's engine reads, so existing
Voxlap assets load directly. The rendering, lighting, editing and
animation code is an independent Rust implementation — it contains no
Voxlap C source. (The CPU renderer began as a faithful port of Voxlap's
column raycaster; that has since been replaced by a clean-room
per-pixel 3D-DDA, which also retires the old raycaster's silhouette and
hairline artifacts.)

It then grows past Ken's single-world engine: a multi-grid **scene graph**
(`roxlap-scene`) places many independently-rotating chunked voxel grids in
one f64 world with streaming + snapshots, and an **optional GPU compute
renderer** (`roxlap-gpu`, WGPU/WGSL) renders the same scene at much higher
frame rates — the same retro look, the CPU budget freed for game logic. A
unified `roxlap-render` facade picks CPU or GPU and falls back automatically.

## Use it in your game

Depend on the facade + the scene graph (the other crates come in
transitively):

<!-- ANCHOR: deps -->
```toml
[dependencies]
roxlap-render = "0.22"   # SceneRenderer — one renderer over CPU + GPU
roxlap-scene  = "0.22"   # Scene / Grid / edits / streaming
roxlap-core   = "0.22"   # Camera + per-frame render settings
glam          = "0.30"
```
<!-- ANCHOR_END: deps -->

This README is the pitch and the 40-line quickstart; the full guide —
concepts, scene graph, lighting, sprites, asset pipeline, tuning — is
**[the roxlap book](https://github.com/NCrashed/roxlap/tree/master/docs/book)**
(`mdbook serve docs/book` for a local copy).

The core loop is: build a [`Scene`], place voxels on a `Grid`, then
hand the scene + a `Camera` to `SceneRenderer::render` each frame —
the facade picks the GPU compute backend and falls back to the CPU
renderer automatically.

```rust,no_run
use glam::{DVec3, IVec3};
use roxlap_core::{opticast::OpticastSettings, Camera};
use roxlap_render::{BackendPreference, FrameParams, RenderOptions, SceneRenderer};
use roxlap_scene::{GridTransform, Scene};

// `window` is anything raw-window-handle: winit, SDL, GLFW, …
fn run(window: std::sync::Arc<winit::window::Window>) {
    // A one-grid world: a plain + a dome. +z points DOWN (voxlap
    // convention); colours are voxlap-packed 0x80_RR_GG_BB.
    let mut scene = Scene::new();
    let id = scene.add_grid(GridTransform::at(DVec3::ZERO));
    let grid = scene.grid_mut(id).unwrap();
    grid.set_rect(IVec3::new(-128, -128, 210), IVec3::new(127, 127, 254), Some(0x80_4d_8a_3a));
    grid.set_sphere(IVec3::new(0, 0, 205), 30, Some(0x80_40_60_c0));

    let opts = RenderOptions {
        backend: BackendPreference::PreferGpu, // GPU with CPU fallback
        ..RenderOptions::default()
    };
    let mut renderer = SceneRenderer::new(window, (960, 600), &opts);

    // Per frame:
    let camera = Camera::orbit(0.4, 0.35, 220.0, [0.0, 0.0, 195.0]);
    let settings = OpticastSettings::for_oracle_framebuffer(960, 600);
    let frame = FrameParams::new(&settings);
    renderer.render(&mut scene, &camera, &frame);
    renderer.present();
}
```

The complete runnable version (winit window + event loop + orbit
camera, ~140 lines) ships as an example:

```sh
cargo run --release -p roxlap-render --example quickstart
```

## Quick start

Try the procedural-cave demo — generates a Worley + Perlin cave
network on startup, lets you fly through it, fire plasma bullets that
carve the world in real time, and toggle between two visual presets
matching Ken's reference screenshots:

```sh
git clone https://github.com/NCrashed/roxlap
cd roxlap
cargo run --release -p roxlap-cave-demo
```

A window opens (~1-2 s startup for cave gen). Click in the window to
grab the cursor; WASD + mouse-look to fly with collision-checked
movement; `Space` / `LShift` for vertical; `LCtrl` for fast-fly;
**LMB to fire a plasma bullet** that carves a crater on impact;
`F` to toggle blue ↔ magenta cave preset (regenerates); `R` for a
new seed; `Esc` to release the cursor or exit.

For the engine-only demo (full-feature voxel world with kv6 sprites
and panoramic sky, no cave-gen / editing):

```sh
cargo run --release -p roxlap-host
```

`L` toggles baked world-voxel lighting; `F` writes
`roxlap-capture.{txt,ppm}` for off-line repro of render artifacts.

For the **scene-graph showcase** — multiple voxel grids (streaming hilly
terrain + a rotating ship) in one f64 world, on the unified renderer:

```sh
cargo run --release -p roxlap-scene-demo            # CPU softbuffer
ROXLAP_GPU=1 cargo run --release -p roxlap-scene-demo   # GPU compute path
```

WASD + mouse-look to fly; `R` spins the ship; `T` prints streaming stats;
`H` jumps to a high-altitude top-down vantage; `C` enters a top-down
**pick mode** where the cursor follows the mouse and left-click reports the
grid + voxel under the pointer (`SceneRenderer::pick` / `Scene::raycast`).

For the **browser demos** — same engine, wasm32 + WebAssembly SIMD,
running on a `<canvas>`:

```sh
# Engine demo (oracle world, ~360 KB wasm + 18 KB JS)
cd crates/roxlap-web
trunk serve         # opens http://localhost:8080

# Cave demo (procedural Worley + Perlin caves with bullets +
# carving + local relight, ~130 KB wasm)
cd crates/roxlap-cave-web
trunk serve
```

Both use WASD / arrows + click-to-mouse-look + Space / Shift for
vertical. The cave demo adds Ctrl for fast-fly, click-while-locked
to fire bullets, F to toggle blue ↔ mag preset, R for next seed.
The engine demo's `B` runs an in-browser 300-frame bench (results
in the devtools console). Mobile: drag the canvas's left half as
a virtual joystick, right half to look around (cave demo: tap to
fire). Both demos use `wasm-bindgen-rayon` to fan rayon's render
parallelism (per-strip, per-light-row, per-sprite) across Web
Workers, so a 4-core phone gets ~3× the frame rate of a single-
threaded build. `trunk build --release` produces a static `dist/`
that needs **cross-origin-isolation headers**
(`Cross-Origin-Opener-Policy: same-origin` +
`Cross-Origin-Embedder-Policy: require-corp`) on the host —
without them, `SharedArrayBuffer` is disabled and the thread pool
won't spin up. Full setup + per-host header config in
[crates/roxlap-web/README.md](https://github.com/NCrashed/roxlap/blob/master/crates/roxlap-web/README.md).

## Crates

| Crate | Purpose |
|-------|---------|
| [`roxlap-core`](https://github.com/NCrashed/roxlap/tree/master/crates/roxlap-core) | The engine: framebuffer, camera, opticast raycaster, grouscan rasterizer, sprite + sky + voxel-lighting. |
| [`roxlap-formats`](https://github.com/NCrashed/roxlap/tree/master/crates/roxlap-formats) | On-disk file format parsers (`.vxl`, `.kv6`, `.kvx`, `.kfa`) **plus the voxel-edit module** — `delslab` / `insslab` / `ScumCtx` plus high-level `set_spans` / `set_cube` / `set_sphere` / `set_rect` (with bit-exact byte equivalence to voxlap C's `setspans` validated against captured fixtures). No renderer dependency; useful standalone for level editors, asset converters, and procedural-world tools. |
| [`roxlap-cavegen`](https://github.com/NCrashed/roxlap/tree/master/crates/roxlap-cavegen) | Procedural cave generation. Worley-distance shape classification + Perlin overlay, two visual presets (`BlueCaveGenerator`, `MagCaveGenerator`) matching Ken + Tom Dobrowolski's 2003 *Justfly* demo screenshots, and a `pack_dense_grid_to_vxl` helper that folds a dense voxel mask + colour grid into voxlap's slab format. Pure-Rust (no `cmake` / C++ build deps). |
| [`roxlap-scene`](https://github.com/NCrashed/roxlap/tree/master/crates/roxlap-scene) | The scene-graph layer above the per-chunk renderer: many independently-placed chunked voxel grids in one f64 world (`GridTransform` = position + quaternion), cross-chunk raycast composition, runtime edits, serde snapshots, far-LOD billboards, chunk streaming + procedural generation (`ChunkGenerator`), and world queries — `Scene::raycast`, `resolve_voxel`, `Grid::voxel_solid` / `voxel_color`. |
| [`roxlap-gpu`](https://github.com/NCrashed/roxlap/tree/master/crates/roxlap-gpu) | Optional GPU renderer — a WGPU/WGSL compute-shader voxel marcher (two-level chunk + voxel DDA, per-chunk decompress/upload, multi-grid composition, sky + fog, edit/stream invalidation, KV6 sprite model-DDA, scene-grid mip LOD, chunk-AABB empty-space skip). Sibling to the CPU opticast, not a replacement; same retro look, much higher frame rates. |
| [`roxlap-render`](https://github.com/NCrashed/roxlap/tree/master/crates/roxlap-render) | Unified renderer facade — one `SceneRenderer` over the CPU opticast and the GPU marcher with **automatic CPU fallback**. Owns presentation, the Scene→GPU bridge, sprites + animated voxel clips, **Doom-style GIF billboard sprites** (`gif` feature: `gif_import` + `add_billboard_instance` / `BillboardActor` — camera-facing animated cutouts that cast + receive shadows), screen→world picking (`pick` / `pixel_ray` / `view_ray` / `pick_depth`), depth-tested overlay lines (`draw_lines` — editor gizmos occluded by the scene), and a **fixed-resolution post pipeline** (`set_render_resolution` renders into a fixed logical grid nearest-upscaled to the window so FPS stops tracking window size; `set_ssaa` supersamples; `set_posterize` reduced-palette + dither for the retro look). Hosts stay thin: build a `Scene`, advance it, call `render`. |
| [`roxlap-cave-demo`](https://github.com/NCrashed/roxlap/tree/master/crates/roxlap-cave-demo) | Procedural-cave showcase binary (winit + softbuffer). Cave-gen on startup, real-time edits via plasma bullets, fog, F/R preset+seed toggles. |
| [`roxlap-scene-demo`](https://github.com/NCrashed/roxlap/tree/master/crates/roxlap-scene-demo) | Scene-graph + GPU showcase binary: a menu of scenes (World, Sprites, Animation, Transparency, Lighting, **Doom** GIF billboards, Picking, …) on the unified renderer (`ROXLAP_GPU=1` selects the GPU backend). Mouse-pick mode, runtime carving, top-down vantage, and a live **Render pipeline** HUD panel (resolution / SSAA / posterize + dither). |
| [`roxlap-host`](https://github.com/NCrashed/roxlap/tree/master/crates/roxlap-host) | Engine-feature demo binary (kv6 sprites + KFA animation + panoramic sky on the bundled oracle world). Despite the name it is **not** a host-integration helper — for embedding the engine in your own window/event loop, see `roxlap-render` + the `quickstart` example. |
| [`roxlap-web`](https://github.com/NCrashed/roxlap/tree/master/crates/roxlap-web) | Engine demo for the browser (wasm32 + wasm-bindgen + canvas). Oracle world + WebAssembly SIMD batches, ~360 KB wasm bundle. Run via `trunk serve` for dev / `trunk build --release` for deploy. |
| [`roxlap-cave-web`](https://github.com/NCrashed/roxlap/tree/master/crates/roxlap-cave-web) | Cave demo for the browser — Worley + Perlin cave-gen, fly + fire + carve with local relight on impact, all on wasm32. ~130 KB wasm bundle (no embedded asset; cave is generated client-side). |

The library API surface is documented on docs.rs — start with
[docs.rs/roxlap-render](https://docs.rs/roxlap-render) (the facade your
game talks to) and [docs.rs/roxlap-scene](https://docs.rs/roxlap-scene)
(the world you build), then
[docs.rs/roxlap-core](https://docs.rs/roxlap-core) /
[docs.rs/roxlap-formats](https://docs.rs/roxlap-formats) /
[docs.rs/roxlap-gpu](https://docs.rs/roxlap-gpu) for the layers
underneath.

## Why roxlap?

- **Cross-platform from one source.** Linux, Windows, macOS (x86_64 +
  arm64), wasm — all from one Cargo workspace. No `#ifdef _MSC_VER`,
  no MASM, no C FFI.
- **SIMD per architecture.** SSE2 on x86_64, NEON on aarch64,
  WebAssembly simd128 (`f32x4_*`) on wasm32 — all via
  `core::arch::*` intrinsics. A portable scalar fallback exists as
  the correctness reference, and per-arch goldens pin each path's
  output bit-for-bit (rsqrt-approximation precision differs across
  arches by design).
- **Idiomatic safe Rust public API.** RAII handles, `Result` at every
  external boundary, no globals leaked across an FFI seam because there
  is no FFI.
- **Bit-exact correctness against voxlaptest** where the SIMD approach
  matches; image-similarity correctness everywhere else, with frozen
  per-pose hashes pinning known sub-pixel rounding noise so any
  *unintentional* drift fails CI immediately.
- **Real-time voxel editing.** Carve / fill spans, cubes, rectangles,
  and spheres at runtime via `roxlap_formats::edit::*`. The same edit
  pipeline drives the cave demo's bullet impacts and is byte-equality
  validated against voxlap C's `setspans`. Closure-based colour
  callbacks let you implement any of voxlap's `vx5.colfunc` patterns
  (constant, jittered, position-dependent, texture-mapped) without
  the global-state dance the original engine required.

- **Transparent voxels.** Alpha-blended, additive, and Beer–Lambert
  *volumetric* voxels for smoke, fire, spell auras, glass, water, and
  filled fog — on both backends. Because the per-pixel 3D-DDA renderer
  visits voxels strictly front-to-back, it composites them in order with
  no depth sort or OIT scheme. A 256-entry material palette
  (`define_material`) drives per-instance, per-voxel (mixed opaque-frame +
  glass models — for static sprites **and** animated clips), and
  world-terrain (glass walls, water) translucency; `Volumetric` weights
  opacity by the ray's path length so a filled cloud reads denser at its
  core. See `docs/porting/PORTING-TRANSPARENCY.md`.

## Status

Published on crates.io (`roxlap-core`, `-formats`, `-cavegen`, `-scene`,
`-gpu`, `-render`). The CPU renderer is feature-complete: voxel terrain
(`opticast` + `grouscan`), animated kv6 sprites, world-voxel lighting,
textured panoramic sky, per-arch SIMD (SSE2 / NEON / wasm simd128), and
rayon multicore. On top, the **scene graph** (S1–S7: multi-grid f64 world,
rotation, cross-chunk gline, far-LOD billboards, streaming + procgen) and
the **GPU compute renderer** (GPU.0–13) have landed, with a unified
CPU/GPU facade and a screen→world picking / `raycast` query API.

CPU-render correctness is pinned by in-crate test suites: `roxlap-core`'s
per-pixel DDA renderer is cross-checked against a dense per-voxel
reference walk (including a leak-free empty-space-skip regression), and
`roxlap-scene` freezes framebuffer-hash goldens for multi-grid / stacked
/ streaming poses. The GPU renderer is non-deterministic across devices
by design and is validated by a headless render-diff harness rather than
byte-goldens.

See [PORTING-RUST.md](https://github.com/NCrashed/roxlap/blob/master/docs/porting/PORTING-RUST.md)
for the CPU-port substage roadmap, plus `PORTING-SCENE.md` (scene graph) and
`PORTING-GPU.md` (GPU renderer) for the later arcs — all under
[docs/porting/](https://github.com/NCrashed/roxlap/tree/master/docs/porting).

## Multicore

CPU rendering is rayon-parallel out of the box — no API to call. The
per-pixel DDA renderer splits the frame into row strips, the
world-voxel lighting bake fans out per column batch, and the sprite
pass batches per sprite; all of it is internal to
`SceneRenderer::render` / `Grid::bake_lightmode`. Bound the pool with
the standard `RAYON_NUM_THREADS` env var. In the browser the same
parallelism runs on Web Workers via `wasm-bindgen-rayon` (see the
web-demo notes above). Design history + measurements:
[PORTING-MULTICORE.md](https://github.com/NCrashed/roxlap/blob/master/docs/porting/PORTING-MULTICORE.md)
(original strip-parallel port) and
[PORTING-PERF.md](https://github.com/NCrashed/roxlap/blob/master/docs/porting/PORTING-PERF.md)
(the PF series: parallel lighting bake, dirty-extent GPU refresh,
quiet-frame skips).

## Documentation

- **The roxlap book** — the user guide (quickstart, concepts, scene
  graph, rendering, lighting, sprites, assets, tuning):
  [docs/book](https://github.com/NCrashed/roxlap/tree/master/docs/book),
  built with `mdbook serve docs/book`.
- API: [docs.rs/roxlap-render](https://docs.rs/roxlap-render),
  [docs.rs/roxlap-scene](https://docs.rs/roxlap-scene),
  [docs.rs/roxlap-core](https://docs.rs/roxlap-core),
  [docs.rs/roxlap-formats](https://docs.rs/roxlap-formats).
- Algorithm + porting notes: [docs/porting/](https://github.com/NCrashed/roxlap/tree/master/docs/porting),
  starting from [PORTING-RUST.md](https://github.com/NCrashed/roxlap/blob/master/docs/porting/PORTING-RUST.md).
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
([.github/workflows/ci.yml](https://github.com/NCrashed/roxlap/blob/master/.github/workflows/ci.yml)).

## License

Dual-licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](https://github.com/NCrashed/roxlap/blob/master/LICENSE-APACHE))
- MIT license ([LICENSE-MIT](https://github.com/NCrashed/roxlap/blob/master/LICENSE-MIT))

at your option — including commercial use.

roxlap is an independent Rust implementation that contains none of Ken
Silverman's original Voxlap C source. Its renderer is a clean-room
per-pixel 3D-DDA over a brickmap — not Voxlap's column-coherent
raycaster — and the remaining engine math (lighting, voxel editing,
bone solving, projection) is independently implemented. The crates
interoperate with Voxlap's on-disk file formats (`.vxl`, `.kv6`,
`.kvx`, `.kfa`); file formats are not themselves subject to copyright,
and the parsers here are independent implementations written to read
those formats.

Credit where due: the `.vxl`/`.kv6`/`.kvx`/`.kfa` formats and the
original Voxlap engine that inspired this project are
[Ken Silverman's](http://advsys.net/ken/) — see
[advsys.net/ken/voxlap.htm](http://advsys.net/ken/voxlap.htm).
