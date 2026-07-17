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
  rate, and the CPU budget goes back to your game. Empty space costs
  next to nothing: rays cross provably-empty regions via a per-grid
  **occupancy pyramid** (< 40 B/grid, maintained live on edits), so
  sparse worlds — a floating ship over distant terrain — don't pay
  per-chunk for the air between (measured −30% frame time on
  empty-gap-dominated views; byte-stable by construction). Design
  history:
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
back to the CPU path presented through WebGL2 — chapter 12).

## Capability parity: `supports()`

A few features exist on one backend only — sky panoramas and sprite
carving are GPU-only, free per-frame depth picks are CPU-only, and so
on. The methods involved stay callable everywhere and degrade to
documented no-ops (or documented costs); `supports(Feature::..)` is
the queryable form of that parity table, so you can pick a strategy
at startup instead of discovering a no-op visually:

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
gizmos, debug paths, hover wireframes. Note the colour type:
`Line3.color` is an `OverlayColor` — the one packing with a real
**alpha** byte (chapter 2's colour family). Depth-tested lines are occluded by nearer
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

## Cutaway deck views

For ship-interior ("deck view") rendering, every grid carries an
optional horizontal clip plane — set `Grid::z_clip` directly or via
the scene facade:

```rust,noplayground
// Hide everything above grid-local voxel z = 290 (z-down): the upper
// decks vanish, z = 290 is the first visible layer and its top face
// renders as the cut surface. `None` restores the full grid.
scene.set_grid_z_clip(ship, Some(290));
```

Semantics, on both backends identically:

- **Grid-local, absolute voxel z.** The plane lives in the grid's own
  frame (the same coordinates `set_voxel` uses, spanning stacked
  chunks), so it stays glued to a rotated or moving ship. Voxels with
  `z < z_clip` render as air; `z_clip` itself stays visible.
- **"World as if removed."** Primary rays, sun and point-light shadow
  rays (including cross-grid ones — each grid applies its *own*
  clip), sprite visibility and picking all agree. A cut-away deck
  neither renders *nor* casts shadows onto what's exposed below.
- **The cut face** reuses the run-top colour fallback — voxlap RLE
  stores no interior colours, so a cut through a solid region shows
  the run's stored surface colour with the normal top-face shading.
  No new material in v1.
- **Render-only.** Simulation, collision, audio occlusion and
  `Scene::raycast` still see the full grid: a hidden deck keeps
  blocking sound, movement and gameplay traces. Persisted in
  snapshots (wire v4; older saves load unclipped).

**Sprites — the footprint rule.** A sprite instance is hidden while
its origin, mapped into a clipped grid's frame, lands inside that
grid's materialised XY chunk footprint with local `z < z_clip`.
Instances outside the footprint are never affected — clipping the
ship's upper decks hides the crew *on* those decks, not a character
standing on the ground beside the hull. The same test is exposed as
`Scene::cutaway_hides_point(world)` for your own overlays and
effects.

**Lights are game-managed.** The engine deliberately does *not*
filter point lights above the plane — a lamp on a hidden deck would
keep lighting the exposed interior. Cull them yourself with the same
footprint rule when the cut moves:

```rust,noplayground
// The light-cull pattern (see the demo's Decks scene): keep only the
// lights the cut doesn't hide.
lit.clear();
lit.extend(
    cabin_lights
        .iter()
        .filter(|l| !scene.cutaway_hides_point(l.position.into()))
        .copied(),
);
frame.lights = Some(LightRig { points: &lit, ..rig });
```

**Picking on the cut.** GPU depth picks are clip-aware for free (they
read the clipped render's depth). For a CPU-side trace that lands on
what the cut *shows* — click-to-select on decks — use
`Scene::raycast_clipped`; keep the plain `raycast` for gameplay
line-of-sight, which must ignore the render-only clip.

**The iso look** is just a camera: a distant eye with a narrow FOV
("tele-iso", ≈ 0.15 rad) flattens perspective without any
orthographic projection. On the GPU backend raise
`set_gpu_mip_scan_dist` past the orbit distance while such a view is
active, or distance LOD will coarsen the whole scene. The scene demo
ships the full pattern as the **Decks** tab
(`ROXLAP_SCENE=Decks cargo run -p roxlap-scene-demo`, `PgUp`/`PgDn`
slides the cut).

### The keyhole cutout (third person)

The deck clip cuts *ceilings above* a grid-state plane; for
third-person play you also need to cut the *walls in front* — and
"front" rotates with the camera, so it cannot be grid state. That is
`FrameParams::view_cutout`: a camera-relative, screen-space keyhole
(the classic BG3/Divinity occlusion cutout) around a world-space
focus point, re-derived by the facade every frame:

```rust,noplayground
// Open a keyhole around the controlled character. Geometry between
// the camera and the focus, inside a `radius_px` screen circle around
// its projection and above the focus plane, renders as air.
frame.view_cutout = Some(ViewCutout {
    focus_world: chest,   // the character, world space
    radius_px: 110.0,     // logical pixels (mouse wheel it)
    feather_px: 24.0,     // radial taper band inside the radius
    margin: 1.5,          // reveal stops this short of the body
    z_bias: 6.5,          // cutting plane at the FEET (floor stays)
});
```

A cell is hidden only when **all three** hold: its **centre** lies
inside the view cone around the eye→focus axis (the cone is what
`radius_px` means — the screen circle it subtends around the
projected focus), it is closer to the eye than the nearest point of
the character COLUMN at the cell's own height (minus `margin` — so an
obstacle the character stands right behind melts down to the boots,
not just to a chest-level sphere), and its grid-local z is above the
focus plane — so the floor in front of the character stays. Classifying
whole cells (rather than gating each ray by a screen window) keeps
the cut edge cube-granular and spatially coherent, exactly like the
deck clip's edges — a per-pixel rule leaves sub-cell fragments that
read as ragged teeth wherever the boundary crosses a floor or wall at
a shallow angle. Across the feather band the reveal *distance* tapers
linearly to zero, closing the hole into a smooth funnel; the result
is deterministic (identical formula on both backends) — no dither, no
temporal shimmer. Cut faces reuse the same run-top colour fallback as
the deck clip.

**A view aid, not world removal** — the deliberate opposite of
`z_clip`'s "world as if removed": the cutout applies to **primary
rays only**. A keyhole-hidden wall keeps casting sun shadow into the
hole, keeps blocking audio and gameplay raycasts, and keeps its
collision. GPU depth picks read the cut render's depth, so clicking
*through* the keyhole selects what you see — usually exactly what a
third-person cursor wants. Per-frame view state: nothing is
persisted, and `None` (the default) is byte-identical to the plain
render.

**When to use which:** deck/iso views → `Grid::z_clip`; a follow
camera behind walls → `view_cutout`; and they compose — the natural
pattern is a deck clip that FOLLOWS the character (expose the deck
they are on) with the keyhole handling the walls in front. The demo's
**Boarding** tab (`ROXLAP_SCENE=Boarding cargo run -p
roxlap-scene-demo`) walks a character over the Decks shiplet with a
shoulder camera that never raycast-clamps its boom, and every mode on
a hotkey: `K` keyhole on/off, `V` deck-follow clip vs manual
(`PgUp`/`PgDn`), `C` shoulder vs the tele-iso deck orbit, wheel =
keyhole radius.

## Where next

- [The render pipeline](render-pipeline.md) — the fixed-resolution /
  SSAA / posterize post stack this chapter's example already enables.
- [Lighting & materials](lighting.md) — `FrameParams::lights` and the
  material palette.
- [Picking & world queries](picking.md) — `pick`, `view_ray`,
  `pick_depth` (also facade methods, same per-frame world).
