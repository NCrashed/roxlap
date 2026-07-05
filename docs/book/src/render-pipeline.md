# The render pipeline

Both roxlap backends are per-pixel marchers, so frame cost scales
with the number of pixels marched. Left alone, that couples your
frame rate to the player's window size — a 4K window costs ~9× a
720p one. The render pipeline breaks that coupling and, as a bonus,
is where the deliberate retro look lives.

The example from the previous chapter enables the whole stack:

```sh
cargo run --release -p roxlap-render --example book_pipeline
```

## The stages

Once configured, every frame flows through:

1. **March** at `logical × ssaa` pixels — the only expensive stage.
2. **SSAA resolve** — box-downfilter the supersamples back to one
   colour per logical pixel.
3. **Posterize** (optional) — dither + quantize each channel at the
   logical resolution.
4. **Upscale** — nearest-neighbour to the window. Hard pixel edges,
   no smearing.

Everything is off by default: `RenderResolution::Native`, `ssaa = 1`,
posterize `None` renders exactly the naive way (logical == window).
The knobs are independent — use fixed resolution purely for
performance without any retro styling, or posterize at native
resolution.

```rust,noplayground
{{#include ../../../crates/roxlap-render/examples/book_pipeline.rs:pipeline}}
```

## Logical resolution

`set_render_resolution` takes a `RenderResolution`:

- `Native` — logical == window (default).
- `Fixed { w, h }` — the retro pixel grid. The march cost is now a
  constant regardless of window size. A logical aspect ratio
  different from the window's stretches non-uniformly — the classic
  fixed-res behaviour, no letterboxing.
- `Scale(f)` — logical = `window × f`, aspect preserved. `Scale(0.5)`
  is "quarter the pixels, whatever the window is".

`logical_dims()` reports the resolved logical size,
`render_dims()` the actual marched size (`logical × ssaa`) — useful
for HUD readouts and for sanity-checking what you asked for.

FOV is unaffected by any of this: both backends derive it from
`FrameParams::settings`, which is resolution-independent.

## Supersampling

`set_ssaa(2)` marches a 2×2 sample grid per logical pixel and folds
it back down — anti-aliasing *within* the hard pixel grid (distant
thin geometry stops shimmering, edges settle). Cost is quadratic in
the factor (clamped to `1..=4`); at a fixed 320×180 logical grid,
`ssaa = 2` is a 640×360 march — usually still far cheaper than a
native-resolution window.

## Posterize & dither

`set_posterize(Some(PosterizeConfig { .. }))` quantizes each colour
channel to a small number of levels — the reduced-palette look.
Uniform levels via `PosterizeConfig::uniform(n, dither)`, or set
`levels_r/g/b` independently (classic hardware palettes were rarely
symmetric). Quantization alone produces hard banding on smooth
gradients (fog!), which is what the `dither` field is for:

- `DitherMode::None` — plain rounding; embrace the bands.
- `Bayer4x4` — the ordered cross-hatch pattern of 90s consoles.
- `BlueNoise` — interleaved-gradient noise; finer grain, no repeating
  grid, reads as texture rather than pattern.

Dither is indexed by the logical pixel, so each hard pixel still
resolves to exactly one colour — the upscale never blurs it.

The best way to choose values is live: the scene demo's **Render
pipeline** HUD panel (`F1`) exposes resolution / SSAA / posterize /
dither as interactive controls over any demo scene:

```sh
cargo run --release -p roxlap-scene-demo
```

## The egui HUD

An in-game UI presents through the same frame: enable the `hud`
feature on `roxlap-render`, run egui in your host (e.g. `egui` +
`egui-winit`), and finish the frame with `paint_egui` **instead of**
`present`:

```rust,ignore
let ppp = egui_ctx.pixels_per_point();
let jobs = egui_ctx.tessellate(full.shapes, ppp);
renderer.paint_egui(&jobs, &full.textures_delta, ppp);
```

The GPU backend paints via `egui-wgpu`; the CPU backend
software-rasterises the same tessellation into its framebuffer — the
HUD works even on the pure-software path. The UI draws at window
resolution, *after* the upscale: panels stay crisp over a chunky
320×180 world. A complete host (input plumbing, panels, the live
pipeline controls above) is
[`roxlap-scene-demo/src/host.rs`](https://github.com/NCrashed/roxlap/blob/master/crates/roxlap-scene-demo/src/host.rs).

## Further reading

- [`PORTING-PIPELINE.md`](https://github.com/NCrashed/roxlap/blob/master/docs/porting/PORTING-PIPELINE.md)
  — the RP-stage design history, including why TAA and texture-filtered
  upscales were rejected (they fight the crisp-retro goal).
- [Performance & tuning](tuning.md) — the other knobs that bound frame
  cost (scan distances, mip ladder, streaming radii).
