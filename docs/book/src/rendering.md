# Rendering & backends

Your game draws through one type:
[`SceneRenderer`](https://docs.rs/roxlap-render), the facade over
roxlap's two renderers. You build a `Scene`, advance your game, and
call the same four or five methods every frame — which backend does
the marching is a construction-time choice, not an architectural one.

The snippets in this chapter and the next come from a runnable
example — a foggy pillar avenue rendered through the full retro
pipeline with an overlay gizmo:

```sh
cargo run --release -p roxlap-render --example book_pipeline
```

## The two backends

Both are per-pixel voxel ray-marchers with the same retro look, and
both derive their projection from the same `FrameParams`, so a scene
frames identically on either:

- **CPU** — a clean-room per-pixel 3D-DDA over a brickmap, rayon-
  parallel across row strips, presented via a software framebuffer
  (softbuffer on native, a WebGL2 blit on wasm). Zero GPU
  requirements: it runs in a VM, over remote desktop, on a machine
  with broken drivers. Design history:
  [`PORTING-DDA.md`](https://github.com/NCrashed/roxlap/blob/master/docs/porting/PORTING-DDA.md).
- **GPU** — a WGPU/WGSL compute-shader marcher (two-level chunk +
  voxel DDA with chunk-occupancy skip and distance-based mip LOD),
  presented via a wgpu swapchain. Same image, several times the frame
  rate, and the CPU budget goes back to your game. Design history:
  [`PORTING-GPU.md`](https://github.com/NCrashed/roxlap/blob/master/docs/porting/PORTING-GPU.md).

The facade keeps them in lockstep: scene edits, sprites, materials
and lighting are tracked once and pushed to whichever backend is
live.

## Choosing a backend

`RenderOptions::backend` takes a `BackendPreference`:

- `Cpu` — the software renderer, unconditionally (the default).
- `PreferGpu` — try WGPU; on failure, fall back to the CPU renderer
  with a warning through the [`log`] facade. The right choice for
  games: one binary runs everywhere.
- `RequireGpu` — GPU or an error. Use it when a silent software
  fallback would lie to you: benchmark rigs, GPU CI.

```rust,noplayground
{{#include ../../../crates/roxlap-render/examples/book_pipeline.rs:backend_select}}
```

`SceneRenderer::new` is the panicking convenience form of `try_new`
(the quickstart uses it); real games call `try_new` and show their
own error UI. Diagnostics — including *why* a machine fell back to
software rendering — go through `log`, so install a logger
(`env_logger` in the examples) or that warning is invisible.

The window parameter is anything
[raw-window-handle](https://docs.rs/raw-window-handle) in an `Arc` —
winit, SDL, GLFW, your own. On wasm, construct with
`new_from_canvas_async` over an HTML canvas instead (WebGPU, falling
back to the CPU path presented through WebGL2 — chapter 11).

## Capability parity: `supports()`

A few features exist on one backend only — sky panoramas are
GPU-only, terrain/sprite translucency is CPU-only for now, and so on.
The methods involved stay callable everywhere and degrade to
documented no-ops; `supports(Feature::..)` is the queryable form of
that parity table, so you can pick a strategy at startup instead of
discovering a no-op visually:

```rust,noplayground
{{#include ../../../crates/roxlap-render/examples/book_pipeline.rs:supports}}
```

The authoritative feature-by-feature table lives on the
[`Feature`](https://docs.rs/roxlap-render) enum's rustdoc — the book
deliberately doesn't copy it (it changes as parity gaps close).

## The frame protocol

One frame, in order:

1. **`tick(&camera, dt)`** — advances every facade-owned animation
   (clips, characters, billboard actors) in one call. Only needed
   once you use those (chapter 7).
2. **`render(&mut scene, &camera, &frame)`** — composites the scene
   into the backend's frame buffer. Does *not* present.
3. **Overlays** (optional) — `draw_lines` / `draw_images` draw into
   the composited frame, using its camera and depth buffer.
4. **Exactly one of `present()` or `paint_egui(..)`** — finishes and
   shows the frame.

At shutdown, call `wait_idle()` before the window is torn down —
otherwise the GPU backend's in-flight work can leave the compositor
showing stale buffers (the "leftover triangles on exit" symptom).

### `FrameParams`

The per-frame parameter block. It is `#[non_exhaustive]` — always
construct with `FrameParams::new(&settings)` and override fields, so
engine upgrades that add parameters don't break your build:

```rust,noplayground
{{#include ../../../crates/roxlap-render/examples/book_pipeline.rs:frame_params}}
```

What lives here: the shared `OpticastSettings` (framebuffer geometry,
projection, scan distances — both backends' field of view is derived
from it, `2·atan(yres/2 / hz)`), sky colour and optional sky, CPU fog
(colour + full-fog distance), per-face `side_shades`, the `lights`
rig (chapter 6), and the `draw_sprites` switch. Settings are cheap to
rebuild per frame from the current window size.

### Overlay lines

`draw_lines` renders world-space segments over the frame — editor
gizmos, debug paths, hover wireframes. Note the colour packing:
`Line3.color` is `0xAARRGGBB` with a real **alpha** byte, unlike voxel
colours (chapter 2). Depth-tested lines are occluded by nearer
voxels; non-depth-tested ones draw on top (hover highlights).

```rust,noplayground
{{#include ../../../crates/roxlap-render/examples/book_pipeline.rs:gizmo}}
```

```rust,noplayground
{{#include ../../../crates/roxlap-render/examples/book_pipeline.rs:overlay}}
```

For textured quads (world-fixed or billboarded) there is the
`upload_image` / `draw_images` pair — same slot in the protocol, see
the [docs.rs](https://docs.rs/roxlap-render) entries.

## Where next

- [The render pipeline](render-pipeline.md) — the fixed-resolution /
  SSAA / posterize post stack this chapter's example already enables.
- [Lighting & materials](lighting.md) — `FrameParams::lights` and the
  material palette.
- [Picking & world queries](picking.md) — `pick`, `view_ray`,
  `pick_depth` (also facade methods, same per-frame world).
