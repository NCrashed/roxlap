# roxlap — scene-graph voxel engine (Substage S)

Sub-substage roadmap and locked decisions for evolving roxlap from
"single bounded voxlap world" into a **scene-graph engine where
the universe is many independent chunked voxel grids, each with
its own arbitrary 3D transform**. Companion to
[PORTING-RUST.md](PORTING-RUST.md) (the original R1..R12 port,
landed in 0.1.0 on 2026-05-07) and
[PORTING-MULTICORE.md](PORTING-MULTICORE.md) (R12 multicore).

This is a **new engine project on top of voxlap**, not an extension
of the port. R1..R12 ported voxlap as-is; the scene work builds a
new layer above the per-chunk renderer.

This document is the **start-of-stage brief**. A fresh-context
session should read this top to bottom before touching code.

## Status as of 2026-05-07

- R1..R12 + R10.X follow-ups all landed.
- 0.1.0 + 0.1.1 published to crates.io: `roxlap-formats`,
  `roxlap-core`, `roxlap-cavegen`.
- The original voxlap port is feature-complete. This is the next
  stage of work, motivated by user's vision of a **voxel space
  game**: each ship is a small chunked voxel world; planet
  surfaces are large fixed chunked voxel worlds; the universe is
  populated with ships, planets, stations, debris, all coexisting
  in a streamed/generated 3D space.

## Goal

Render N independent chunked voxel grids in a single 3D scene,
each with its own f64 origin + arbitrary 3D rotation, with
per-pixel depth composition and far-LOD impostors so the
universe can stretch from "ship corridor at 1 m" to "planet from
1000 km away" within the same render call.

The per-grid renderer stays voxlap's opticast + grouscan +
sprite + lighting pipeline (i.e. `roxlap-core` doesn't change
shape much). The new code is the **scene-graph layer above it**,
landing in a new `roxlap-scene` crate.

End-state vision: fly a small voxel ship across a procedural
voxel solar system, dock at a station that's another grid, walk
inside while the scene continues to render the planet you came
from in the bridge window.

## Locked decisions

| # | Decision | Consequence |
|---|----------|-------------|
| 1 | **Skip MAXZDIM=512.** Vertical chunk stacking handles tall worlds for free. | Slab byte format stays unchanged. R8 oracle and 0.1.x bit-exactness remain valid. |
| 2 | **Rotation model: full arbitrary 3D.** Each grid has a `Quat` orientation, not a snap-to-90° restriction. | Per-grid raycast transforms the world ray into grid-local space; grid-local rays are arbitrary 3D directions through an axis-aligned chunk grid. The per-chunk gline already handles arbitrary 2D directions; the **outer 3D DDA across chunks** is the new code that has to handle arbitrary 3D rays. |
| 3 | **Coordinate units: f64 in world, structured per-grid.** Global universe positions are `vec3<f64>` (precision good to ~1 cm at 1000 km). Per-grid voxel addresses are `(chunk: vec3<i32>, voxel: vec3<u32>)`; the world position of a voxel is `grid.origin (vec3<f64>) + grid.rotation * (chunk * CHUNK_SIZE + voxel)`. | f32 stays inside per-chunk math (voxlap's existing rasterizer); f64 is universe-level only. Floats are cast at the world↔grid boundary. |
| 4 | **Scope: streamed / procedurally generated universe.** Real "infinite" support, not a fixed-size scene. | Chunks load on demand based on camera proximity, generate via background tasks, evict via LRU. Grids themselves can spawn / despawn at runtime. |
| 5 | **Crate boundary: new `roxlap-scene` crate.** `roxlap-core` stays focused as the per-chunk renderer (opticast + grouscan + sprites + lighting). Scene-graph types, multi-grid raycast composition, LOD selection, streaming all live in `roxlap-scene`. | Two crates ship together as a coherent stack; `roxlap-core` users who don't need scene graph can keep using just it. |
| 6 | **Chunk size: 128×128×256 voxels.** XY at 128 keeps chunks compact (~2 MB worst case dense slab); Z at 256 preserves voxlap's existing slab byte format inside each chunk (no changes to `MAXZDIM`). Tunable via constant if needed but the byte format work to change Z per chunk is large. | Each chunk is internally a `roxlap_formats::vxl::Vxl` with `vsid=128`. Existing `roxlap-core` rasterizer + edit API run on each chunk unchanged. Vertical worlds = stack chunks along grid-local Z. |
| 7 | **Voxel size: 1 world unit per voxel, fixed across all grids.** No per-grid voxel-size scaling for v1; would complicate raycast and LOD selection. | All grids share the same voxel scale; physical size of a ship is determined by voxel count, not voxel size. Visual scaling comes via grid placement / camera distance. |
| 8 | **Per-grid sprites + lighting stay grid-local.** A KV6 sprite attached to a ship moves with the ship's transform. World lighting bake is per-grid. | Lights inside ship A don't illuminate planet B (acceptable physical inaccuracy; cheaper than a global light pass). |

## Architecture sketch

```rust
// roxlap-scene crate

pub struct Scene {
    grids: HashMap<GridId, Grid>,
    next_grid_id: u32,
    // Streaming + LOD policy hooks here.
}

pub struct Grid {
    pub transform: GridTransform,
    pub chunks: HashMap<IVec3, Vxl>,          // sparse chunk storage
    pub bounds: Option<IVec3Range>,           // optional fixed bounding box
    pub lighting: GridLighting,
    pub sprites: Vec<GridSprite>,
}

pub struct GridTransform {
    pub origin: DVec3,    // f64 world position
    pub rotation: DQuat,  // f64 quaternion (matters at planet scale)
}

pub struct GridAddr {
    pub grid: GridId,
    pub chunk: IVec3,     // signed; chunks centered on grid origin
    pub voxel: UVec3,     // 0..CHUNK_SIZE_{XYZ}
}

// Voxel size constants
pub const CHUNK_SIZE_XY: u32 = 128;
pub const CHUNK_SIZE_Z:  u32 = 256;

impl Scene {
    pub fn add_grid(&mut self, transform: GridTransform) -> GridId;
    pub fn edit_voxel(&mut self, addr: GridAddr, color: Option<u32>);
    pub fn raycast(&self, ray_start: DVec3, ray_dir: DVec3) -> Option<Hit>;
    // ...
}

// Engine integration: roxlap-core::Engine::render takes a Scene
// instead of (or in addition to) a single Vxl.
```

The render dispatch is layered:

```
Engine::render(scene, camera, framebuffer)
  ├── grid_visibility_cull(scene, camera)           // frustum cull per grid AABB
  ├── grid_lod_select(scene, camera)                // near/mid/far per grid
  ├── grids_sorted_front_to_back
  └── for each visible grid:
      ├── transform world camera into grid-local frame  (DVec3+DQuat → f32 grid-local)
      ├── lod = grid_lod[grid]
      ├── match lod {
      │     Near => full_voxel_raycast(grid, grid_local_camera)  // S4: cross-chunk gline
      │     Mid  => low_mip_voxel_raycast(grid, ...)
      │     Far  => billboard_blit(grid, ...)                   // S6
      │   }
      └── composite into shared framebuffer + z-buffer
```

The "full_voxel_raycast" is the one that needs the new
cross-chunk gline (item S4). At Mid/Far it's an entirely
different code path and the per-chunk renderer is bypassed.

## Substage roadmap

| # | Scope | Estimate | Validation |
|---|-------|----------|------------|
| **S1** | Camera outside the grid (foundational unblock). gline + phase_grouscan handle ray starts outside the column grid AABB by clipping into the bounds first. New oracle pose `outside_orbit`. No scene-graph yet. | 3-5 d | Camera moved 256 voxels outside oracle world's vsid renders correct silhouette; existing 12 oracle poses unchanged. |
| **S2** | `roxlap-scene` crate skeleton: `Scene` / `Grid` / `GridTransform` / `GridAddr` types, sparse chunk storage, edit API delegating to `roxlap-formats::edit`, no rendering yet. | 1 w | Round-trip serialize / deserialize / mutate a 2-grid 100-chunk scene; unit tests for chunk ↔ voxel address math; `cargo doc` clean. |
| **S3** | Multi-grid raycast composition. `Engine::render(scene, camera, fb)` path. Per-grid sort + visibility cull. **Axis-aligned grids only** (no rotation yet); single-chunk grids only (no cross-chunk yet). Per-grid raycast uses existing `roxlap-core` opticast on each grid's single chunk; composite by z-buffer. | 1-2 w | Scene with 2 axis-aligned single-chunk grids side-by-side renders both, depth correct at occlusion seams. New visual smoke test in roxlap-scene. |
| **S4** | Cross-chunk gline within a grid. 3D DDA over the grid's chunk index space; each chunk-hit runs per-column raycast (mostly the existing gline). Empty chunks skipped via the sparse map. | 1-2 w | Grid with chunks stacked vertically (e.g. 1×1×8 chunks = 256-tall column world) renders without seams; render hash stable across chunk boundary configurations that should produce identical pixels. |
| **S5** | Per-grid arbitrary 3D rotation. World ray → grid-local ray transform via `DQuat::inverse()`. Grid-local ray runs S4's cross-chunk gline normally. May need DDA accumulator promoted to f64 to avoid drift on long oblique rays. | 2-3 w | Ship rotated 45° around long axis renders both visible faces correctly; no z-fighting at face seams; reference still shots compared against axis-aligned baseline. |
| **S6** | Far-LOD billboards + planet sphere proxies. Per-grid LOD selection (Near/Mid/Far). Billboard impostor: pre-render N orthographic snapshots from a view sphere, blit closest. Planet sphere: raymarched sphere shader with biome lookup, no voxels at planet-from-space distance. LOD-transition fade. | 2-3 w | Scene with 10 small grids at varying distances: nearest is full voxel, mid is low-mip voxel, far is billboard, all in same frame. Planet from 1000 km renders as sphere not chunked voxels. |
| **S7** | Streaming + procedural generation. Chunk-load-on-demand based on grid + camera proximity; chunk eviction via LRU. Generation hooks (cavegen-style) for procedural grids, run on rayon background tasks. Chunk-version counter for edit-vs-generation conflict. | 3-4 w | Fly across a "true infinite" procedurally-generated planet; no GC stutters; chunk popping minimal at LOD transitions; memory budget held under target. |

**Total scope**: ~3-4 months focused work — comparable to the
original R1..R12 port. Realistic split:

- **roxlap 0.2.0**: S1..S5 — full scene graph with rotation,
  fixed-size scenes, no streaming. Already a real engine.
- **roxlap 0.3.0**: S6..S7 — far LOD + streaming. The "actually
  feels like a space sim" milestone.

## Per-stage technical notes

### S1 — Camera outside the grid

Voxlap's opticast assumes the camera is inside the column grid (or
above, via the sky phase that walks down to the world top). When
the camera is outside the X/Y bounds, the gline 2D walk has no
valid starting column; rays would need to clip into the world AABB
first.

Implementation sketch:
1. Compute ray ↔ grid AABB intersection. If miss → sky / void.
2. If hit, fast-forward ray to first AABB face hit (`t_enter`).
3. From that face-hit point, run the standard gline walk. The
   only subtlety is the gline initialisation needs `t = t_enter`,
   not `t = 0`; everything else (column index, z step) is
   computed from the entry point.
4. New phase or modified `phase_grouscan` to dispatch on
   "camera-inside" vs "camera-outside".

Adds a new oracle pose `outside_orbit` with the camera at e.g.
`(vsid + 256, vsid/2, 128)` looking back at the world. Frozen as
a roxlap-only golden (no voxlap C reference; voxlap C can't do
this).

### S2 — roxlap-scene skeleton

Pure data + serde. No rendering. Useful work because:
- Locks the public API of `Scene` / `Grid` / `GridAddr`. Changing
  it later breaks downstreams.
- Forces decisions about chunk storage (sparse `HashMap<IVec3,
  Vxl>` vs dense `Vec<Option<Vxl>>` over a bounded region).
  Sparse is the right default for "grids are small ships" but
  bounded grids can opt into dense for cache locality.
- Prepares the edit API surface: `Scene::edit_sphere(addr,
  radius, color)` decomposes into per-chunk `set_sphere` calls
  via `roxlap-formats::edit` plus seam handling at chunk
  boundaries.

### S3 — Multi-grid composition

Per-grid sort + AABB frustum cull is straightforward. The
**composition step** is the interesting bit:

- Naive: render each grid to its own framebuffer + z-buffer, then
  per-pixel pick the closest. 2 layers × 480×640 framebuffer ×
  N_grids RAM cost.
- Better: render front-to-back into a shared framebuffer + z,
  early-out per pixel when fully written. Voxlap's z-buffer
  semantics need verifying but the existing rasterizer writes
  z-then-pixel which is compatible.

For S3, only axis-aligned grids are allowed. This means
"grid-local camera" is just `world_camera - grid.origin` in f64,
cast to f32 for the per-chunk renderer. No rotation math yet.

### S4 — Cross-chunk gline

This is where most of the per-chunk renderer's assumptions get
exercised. Voxlap's gline does:
1. 2D DDA over column indices `(cx, cy)`.
2. At each column, walk the column's slab list to find the first
   solid voxel along the ray's z direction.
3. Output: `(z_hit, color, normal_estimate)`.

For cross-chunk, the outer loop becomes a **3D DDA over chunk
indices `(chx, chy, chz)`**, with each chunk-hit running the
inner per-column walk (mostly the existing gline) on that
chunk's column data. Empty chunks (no entry in the sparse map)
are skipped at the outer level.

Subtleties:
- Ray entering a new chunk picks up where the previous chunk
  left off — `t` accumulates across chunk boundaries.
- Per-chunk mip levels can mismatch at chunk boundaries. For v1,
  force all chunks in a grid to the same mip level (LOD selection
  is per-grid in S6, not per-chunk).
- The 256-z slab format inside each chunk doesn't change. Stacked
  chunks' z extents are `(chz * 256)..(chz * 256 + 256)` in
  grid-local space.

### S5 — Per-grid rotation

The transform pipeline:
```
world_ray (DVec3 origin, DVec3 dir)
  → grid-local_ray = grid.transform.inverse() * world_ray
                   = grid.rotation.inverse() * (world_ray.origin - grid.origin)
                   plus rotation of dir
  → run S4's cross-chunk gline on grid-local_ray
  → hit comes back in grid-local space
  → composite hit-pos / normal back to world via grid.transform
```

Rotation is a quaternion conjugation per ray. Cheap. The
**accumulator drift** issue is what to watch: long rays through
many chunks accumulate `t` increments that may lose precision in
f32. Promote DDA accumulator to f64 if benchmarks show drift.

The rasterizer itself doesn't change — it sees an axis-aligned
grid in grid-local frame. The rotation is invisible to it.

### S6 — Far-LOD billboards + planet spheres

Three LOD tiers:

- **Near** (camera within ~grid bounding sphere): full voxel
  raycast via S4.
- **Mid** (~10× the grid's max radius away): voxel raycast at the
  grid's coarser mip level (the existing R4.5 multi-mip
  infrastructure already supports this).
- **Far** (~100× the grid's max radius): billboard impostor —
  pre-rendered orthographic snapshot from one of N viewpoints
  arranged on a Fibonacci sphere. Pick the closest viewpoint to
  the current camera direction, blit as a screen-aligned quad.
- **Planet** (massive grid seen from space): raymarched sphere
  proxy, biome lookup texture. No voxels touched. Switches to
  Far/Mid/Near as camera approaches surface.

The transition between LOD levels is the visual gotcha — hard
switches cause popping. Either do alpha cross-fade over a few
frames, or align LOD boundaries with frustum/distance such that
the visual difference is minimal.

Billboard cache is per-grid, regenerated when the grid's voxel
content changes (rare for ships, never for static planets).
Generation cost: N orthographic renders of the grid, where N is
the viewpoint sphere resolution (probably 26 = 6 axis + 12 edge
+ 8 corner is a reasonable starting point).

### S7 — Streaming + procedural generation

The streaming policy is a per-grid **chunk activity radius**: any
chunk inside `r_active` of the camera (in grid-local coordinates)
must be loaded; chunks outside `r_evict` get evicted. Hysteresis
between r_active and r_evict prevents thrash.

Procedural generation hooks: the `Grid` struct optionally carries
a generator trait. When the streaming layer needs a chunk and the
generator is set, it dispatches generation to a background rayon
task. Generated chunks are added to the sparse map atomically
(probably via `crossbeam_channel` mpsc plumbing).

Edits-during-generation conflict is handled via a per-chunk
version counter: edits bump the version; if a generation task
finishes against an out-of-date version, the result is discarded
(the edit takes precedence).

This is where the engine starts to look like a real game-engine
chunk system. Most of the wisdom from Minecraft / Space Engineers
/ etc. translates here — chunk version vectors, dirty flags, async
generation pipelines.

## Risks

### R1. Cross-chunk seam handling (S4)

Chunk boundaries are where most of the bugs live. Mip-level
mismatch (a chunk at mip 2 next to a chunk at mip 1), lighting
discontinuity (per-chunk bake doesn't know about the neighbour),
raycast ray crossing the seam mid-column. Mitigation: force same
mip level across a grid for v1; shared lighting per grid (not per
chunk); raycast walks chunk-by-chunk with explicit handoff at
boundaries.

### R2. Rotation precision (S5)

f64 quat × f32 ray accumulator drift over a 1000-voxel oblique
ray. Mitigation: f64 DDA accumulator; benchmark; if visible
artifacts appear, switch to f64 throughout the ray walk inside a
single chunk. Performance cost ~2× per ray in worst case but
voxlap's hot loops are SIMD batches that handle f64 only via
splatting (slow). Decision deferred to S5; may need profiling.

### R3. Billboard impostor authoring cost (S6)

26 orthographic renders per grid is a lot if grids change often.
A 100-ship space battle with edit-driven billboard regen could
crater frame rate. Mitigation: lazy regen (only re-render the
impostor when the grid hasn't been edited in N frames AND the
visible viewpoint changed), and only regenerate the billboard
viewpoints actually being looked at, not all 26.

### R4. Scope creep into "real engine"

S6 + S7 push toward "general voxel engine" territory; that's what
the user wants but it's also a lot of code that's no longer
related to voxlap. Mitigation: the per-grid renderer stays
voxlap; LOD billboards / planet spheres / streaming are
deliberately separate code paths in `roxlap-scene` so the voxlap
core stays focused. Resist the urge to redesign opticast for
infinite worlds — it's already great at what it does.

### R5. f32 → f64 boundary correctness

Casts at the world↔grid boundary need to be consistent. Off-by-1
voxel at chunk boundaries (because of `as i32` rounding direction
on f64 coords) is a common bug source. Mitigation: a tiny
`world_to_grid_local(world_pos: DVec3, grid: &Grid) -> (IVec3,
UVec3, Vec3<f32>)` helper with property tests; everything else
calls through it.

### R6. Per-grid lighting cost at scale

Each grid maintains its own lighting bake. A scene with 100 ships
each editing 10 voxels per second = 1000 lighting recomputes per
second. The R12 incremental lighting work helps but per-grid bake
overhead may dominate at scale. Mitigation: lazy lighting (only
recompute the dirty region of a grid; only when the grid is in
Near LOD).

### R7. World coordinate range

`f64` gives ~1 cm precision at 1000 km from origin. At 1 light-second
(~300,000 km) precision drops to ~3 m — visible jitter for
voxel-scale gameplay. Mitigation: a "floating origin" pattern —
periodically re-centre the world origin around the camera and
shift all grid origins accordingly. Standard space-sim trick.
Implement when needed (likely in S7 when streaming is real).

## Out of scope (deferred)

- **Per-grid voxel-size scaling**: all grids share 1 unit / voxel.
  Dual-scale grids (large planet + tiny detail ship) is a 0.3.X
  follow-up if needed.
- **Cross-grid lighting**: light from grid A doesn't affect grid B.
  Acceptable physical inaccuracy.
- **Cross-grid physics / collision**: scene-graph engine is
  rendering-only. Physics is a downstream concern; this work
  exposes the data structures but doesn't implement physics.
- **Network sync**: multiplayer streaming is its own beast.
- **Materials beyond color**: voxels are RGBA. PBR / per-voxel
  metal/rough is a 0.4+ direction.
- **Very large grids** (vsid much bigger than 128 per chunk):
  out-of-bound for chunk size; would need per-chunk vsid which
  voxlap C never anticipated.
- **Non-cubic chunks**: 128×128×256 only.

## How to apply (fresh-context entry)

When the user says "let's start S1", this is the entry point.
Read this doc top to bottom, then:

1. **S1**: pick a small step that forces the architecture: take
   `roxlap-host` (or write a new `roxlap-scene-demo`), move the
   camera 256 voxels outside oracle world's vsid, see what
   breaks. The renderer either segfaults (gline accesses
   negative column index) or produces a blank/wrong frame. Fix
   the gline + phase_grouscan paths first; that's S1.
2. After S1 lands, scaffold the new `roxlap-scene` crate (S2)
   with no rendering — just types + edit API.
3. Then S3 brings Engine::render(scene) online with axis-aligned
   single-chunk grids.

Each substage lands as its own commit (subcommits S1.0, S1.1, ...
within each substage following the `R*.X` convention from earlier
stages). Bench harness from R12 / R10.5 picks up scene-graph
benchmarks once Engine::render(scene) exists.

## Reading list (for the implementing session)

1. This doc, top to bottom.
2. `crates/roxlap-core/src/gline.rs` and `phase_grouscan` —
   understand where camera-inside is assumed.
3. `crates/roxlap-core/src/opticast.rs` — the per-frame entry point.
4. `crates/roxlap-formats/src/edit.rs` — the edit primitives that
   the scene-graph layer dispatches to per-chunk.
5. `crates/roxlap-cavegen/src/pack.rs` — for generation patterns
   when designing S7.
6. `crates/roxlap-core/src/world_lighting.rs` — per-chunk lighting
   bake; need to understand for S5 + R6 risk.
7. Existing literature on chunked voxel engines (Space
   Engineers' GDC talks, Minecraft chunk system, the Vintage
   Story dev blog) — for design inspiration on streaming + LOD.

## Naming + version note

Substage prefix is `S` (S1..S7) within `PORTING-SCENE.md` — same
shape as `R10.0..R10.5` within `PORTING-WASM.md`. Nested
sub-substages are `S<n>.<m>` (e.g. S4.1, S4.2).

Public version targets:
- `roxlap-core 0.2.0` + `roxlap-scene 0.1.0` after S5 lands.
- `roxlap-scene 0.2.0` after S7 lands.

`roxlap-formats` likely stays at 0.1.x — its API is stable; the
edit primitives don't need to change for scene work.
