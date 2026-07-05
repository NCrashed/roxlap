# Performance & tuning

Both renderers are per-pixel ray-marchers, so nearly every knob in
this chapter bounds one of two quantities:

```text
frame cost ≈ (pixels marched) × (steps per ray)
```

Pixels are bounded by the render pipeline
([chapter 5](render-pipeline.md)); steps per ray by scan distances
and mip ladders (below); and everything else is about not doing
per-frame work you don't need.

## Bounding the ray

`OpticastSettings` (built per frame, passed via `FrameParams`) owns
the CPU marcher's budget:

- **`max_scan_dist`** — the hard ray-length cap, in voxels. The
  single biggest lever; pair it with `FrameParams::fog_max_scan_dist`
  ([chapter 4](rendering.md)) so the cutoff reads as atmosphere
  instead of a wall.
- **`mip_levels` / `mip_scan_dist`** — the CPU mip ladder: rays
  beyond `mip_scan_dist` step through progressively coarser copies of
  the world, so distance costs logarithmically instead of linearly.
  The demos run `mip_levels = 6`, `mip_scan_dist = 64`. Per-grid
  opt-out: `Grid::mip_levels_override` (small rotated object grids
  don't benefit).

The GPU marcher takes its step budget from the same `settings`
(`FrameParams::gpu_outer_steps` derives it) and has its own LOD
distance: `RenderOptions::gpu_mip_scan_dist` — rays farther than this
sample coarser chunk mips (2× frame-rate at the horizon in the GPU.11
measurements).

## Bounding the world

- **Streaming radii** ([chapter 3](scene-graph.md)): `r_active` is
  simulation/render breadth, `r_evict − r_active` is the hysteresis.
  The World demo runs (256, 384).
- **Upload budgets** (GPU backend, `RenderOptions`):
  `gpu_chunk_upload_budget` caps dirty-chunk installs per frame
  (small = bounded frame spikes, more pop-in when flying fast);
  `gpu_clip_upload_budget` does the same for clip registration.
- **Threads**: CPU rendering parallelises over rayon internally —
  bound the pool with the standard `RAYON_NUM_THREADS`. Streaming
  generation has its own pool: `Scene::set_streaming_threads(n)`.

## Lessons the engine already paid for

The PF/PR performance series distilled into host-side rules:

- **Never read env vars (or any syscall-ish config) per frame.** The
  single worst cliff in roxlap's history was an `env::var_os` inside a
  per-step loop; the recovery series it headlined took the engine-only
  frame rate from ~12 to ~47 FPS. Cache at startup; the engine does
  the same with its own overrides.
- **Batch bulk transforms** — `set_sprite_instance_transforms`
  (plural) over per-instance calls in a loop
  ([chapter 7](sprites.md)).
- **Re-light the edit, not the world** — `bake_lightmode_bbox` after
  runtime carves ([chapter 6](lighting.md)).
- **Let quiet frames stay quiet.** Change detection is version-based:
  don't touch chunks (or bump versions) you didn't modify, and the
  engine's dirty-poll and re-mip sweeps skip themselves.

## Engine environment variables

The library crates read four variables, all **init-time** overrides
of `RenderOptions` / adapter selection (never per frame — see above):

<!-- Source of truth: grep -rhoE '"ROXLAP_[A-Z_0-9]+"' crates/ --include='*.rs' | sort -u,
     keeping the ones read inside roxlap-render / roxlap-gpu (the rest are
     demo-host conveniences — see the demo tour). Re-verify at stage closes. -->

| Variable | Overrides | Meaning |
|---|---|---|
| `ROXLAP_GPU_MIP_SCAN_DIST` | `RenderOptions::gpu_mip_scan_dist` | GPU LOD distance (world units) |
| `ROXLAP_GPU_CHUNK_BUDGET` | `RenderOptions::gpu_chunk_upload_budget` | Dirty-chunk installs per frame (`0` = unbounded) |
| `ROXLAP_GPU_CLIP_BUDGET` | `RenderOptions::gpu_clip_upload_budget` | Clip-upload staging flushes (`0` = unbounded) |
| `ROXLAP_GPU_POWER` | `GpuRendererSettings::power_preference` | `low` \| `high` adapter preference — also the escape hatch when a discrete-GPU driver misbehaves |

(`ROXLAP_GPU` itself — the backend switch you've seen in every example
— is a *demo convention*, not an engine variable: the examples read it
and pick a `BackendPreference`. Your game should expose its own
setting.) A `RenderConfig` consolidation of these into one struct is
on the quality-stage backlog (QE-C6); until then the env overrides
win over the corresponding `RenderOptions` fields.

## CPU or GPU?

Feature parity is queryable — `supports()` and the [`Feature`
rustdoc table](https://docs.rs/roxlap-render)
([chapter 4](rendering.md)); the book doesn't copy it. The
*performance* character is simple: the GPU backend is several times
faster at the same pixel count and shrugs at resolution; the CPU
backend's cost tracks pixels almost linearly — so on the CPU path,
fixed logical resolution ([chapter 5](render-pipeline.md)) is not a
styling choice, it *is* the perf model. Measure with `RequireGpu` on
rigs where a silent CPU fallback would corrupt the numbers.

## Further reading

- [`PORTING-PERF.md`](https://github.com/NCrashed/roxlap/blob/master/docs/porting/PORTING-PERF.md)
  — the PF series: what got 12.8× raycasts and 23× bbox re-bakes.
- [`PORTING-QUALITY.md`](https://github.com/NCrashed/roxlap/blob/master/docs/porting/PORTING-QUALITY.md)
  — the QE backlog, including the `RenderConfig` consolidation.
