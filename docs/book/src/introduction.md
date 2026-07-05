# Introduction & quickstart

roxlap is a voxel-scene engine in pure Rust. You describe a world as
voxel grids — carve, fill, stream, and animate them at runtime — and
one renderer facade draws it either on the CPU (a per-pixel 3D-DDA
raycaster that runs anywhere, no GPU required) or on the GPU (a
WGPU compute-shader marcher, same retro look at much higher frame
rates), falling back from one to the other automatically. The engine
reads Ken Silverman's Voxlap asset formats (`.vxl` worlds, `.kv6`
sprites, `.kfa` animation rigs), so two decades of existing voxel
assets load directly.

This chapter gets a window on screen with a voxel world in it. By the
end you will have seen the whole per-frame contract — everything after
this is elaboration.

## The three crates you talk to

roxlap is a Cargo workspace of small crates, but a game depends on
three:

- [`roxlap-render`](https://docs.rs/roxlap-render) — the facade. One
  `SceneRenderer` type that owns the backend choice (CPU or GPU),
  presentation, sprites, picking, and the post pipeline. Your game
  calls this.
- [`roxlap-scene`](https://docs.rs/roxlap-scene) — the world. `Scene`
  holds many independently-placed chunked voxel `Grid`s in one f64
  world, with edits, streaming, and snapshots.
- [`roxlap-core`](https://docs.rs/roxlap-core) — the `Camera` and the
  per-frame render settings.

Everything else (`roxlap-formats`, `roxlap-gpu`, …) arrives
transitively. Add to your `Cargo.toml`:

{{#include ../../../README.md:deps}}

## A minimal application

The complete program below ships as a compile-tested example —
[`crates/roxlap-render/examples/quickstart.rs`](https://github.com/NCrashed/roxlap/blob/master/crates/roxlap-render/examples/quickstart.rs)
(~160 lines: a winit window, an event loop, and a slow orbit camera).
Run it first, then read the walkthrough:

```sh
cargo run --release -p roxlap-render --example quickstart
ROXLAP_GPU=0 cargo run --release -p roxlap-render --example quickstart  # force CPU
```

Every snippet in this section is included verbatim from that file, so
it cannot drift from what actually compiles.

### Colours

```rust,noplayground
{{#include ../../../crates/roxlap-render/examples/quickstart.rs:colors}}
```

Two conventions to absorb immediately (both inherited from Voxlap, both
covered properly in [Concepts & conventions](concepts.md)):

- **Voxel colours are packed `0x80_RR_GG_BB`.** The high byte is not
  alpha — it is the flat shading intensity, and `0x80` means "unlit
  default". Passing `0xFF_...` where a voxel colour is expected gives
  you an over-bright voxel, not an opaque one.
- The framebuffer/sky packing is plain `0x00_RR_GG_BB`.

### Building a scene

```rust,noplayground
{{#include ../../../crates/roxlap-render/examples/quickstart.rs:build_scene}}
```

A `Scene` is a set of grids; each `Grid` is an unbounded chunked voxel
volume placed in the world by a `GridTransform` (f64 position +
quaternion rotation). Here one grid at the origin gets a solid ground
slab (`set_rect`) and a dome (`set_sphere`).

The third convention, and the one that trips everyone: **+z points
DOWN**. The ground surface sits at `z = 210` and fills *downward* to
`z = 254`; the dome centre at `z = 205` is *above* the ground. Voxlap
kept screen-y and world-z aligned, and roxlap keeps Voxlap's
convention so assets and math port unchanged.

### Creating the renderer

```rust,noplayground
{{#include ../../../crates/roxlap-render/examples/quickstart.rs:init}}
```

`SceneRenderer::new` takes anything implementing raw-window-handle —
winit here, but SDL or your own windowing works the same — plus the
surface size and `RenderOptions`. `BackendPreference::PreferGpu` asks
for the GPU compute backend and **falls back to the CPU renderer
automatically** when WGPU init fails, so the same binary runs on a
machine with no usable GPU. (The renderer field is declared before the
window in the struct so it drops first — the surface must release its
window handles while the window is still alive.)

### Rendering a frame

```rust,noplayground
{{#include ../../../crates/roxlap-render/examples/quickstart.rs:render_frame}}
```

The per-frame contract:

1. Build a `Camera`. Use the constructors (`orbit`, `from_yaw_pitch`,
   `look_at`) — they produce the right-handed basis the engine's
   frustum culling expects.
2. Build `FrameParams` from `OpticastSettings` for the current surface
   size, then override what you need (sky and fog colour here).
3. `render` draws the scene into the backend's target; `present` puts
   it on screen. They are separate calls so you can draw overlays or an
   egui HUD between them — see [Rendering & backends](rendering.md).

### Teardown

```rust,noplayground
{{#include ../../../crates/roxlap-render/examples/quickstart.rs:teardown}}
```

Call `wait_idle` before the window is torn down so quitting never
yanks the swapchain out from under in-flight GPU work.

## Explore the demos

The engine's feature gallery is `roxlap-scene-demo` — eleven scenes
behind a menu (World, Sprites, Animation, Transparency, Lighting,
Spotlight, Particles, Doom, Picking, Primitives, Empty):

```sh
cargo run --release -p roxlap-scene-demo                     # CPU
ROXLAP_GPU=1 cargo run --release -p roxlap-scene-demo        # GPU backend
ROXLAP_SCENE=Lighting cargo run --release -p roxlap-scene-demo  # jump to a tab
```

The [demo tour](demo-tour.md) chapter maps each scene to the features
it showcases; each topic chapter uses its scene as the worked example.

## Where next

- [Concepts & conventions](concepts.md) — coordinate system, colour
  packing, units, camera basis: the five facts that make everything
  else make sense. Read this before writing real code.
- [The scene graph](scene-graph.md) — grids, chunks, edits, streaming.
- [Rendering & backends](rendering.md) — the facade in full.
