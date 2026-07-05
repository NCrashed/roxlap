# The scene graph

A roxlap world is a [`Scene`](https://docs.rs/roxlap-scene): a sparse
set of **grids**, each an unbounded voxel volume with its own f64
world placement. This chapter covers the whole life of that world —
placing grids, editing voxels, saving, and streaming.

Every snippet comes from a runnable, assertion-checked example:

```sh
cargo run -p roxlap-scene --example book_scene_graph
```

## The mental model

Three layers, top to bottom:

- **`Scene`** — owns the grids, hands out stable `GridId`s, answers
  world-level queries (`raycast`, `resolve_voxel` — chapter 10).
- **`Grid`** — one voxel volume: a `GridTransform` (f64 origin + f64
  quaternion) plus a sparse map of chunks. A missing chunk *is* air —
  nothing is stored for empty space, and a grid has no intrinsic
  bounds.
- **Chunks** — 128×128×256 blocks in Voxlap's column-compressed slab
  format. You never address them directly: edits and queries take
  grid-local voxel coordinates and the decomposition is automatic.
  Inserting into empty space materialises the touched chunks;
  carving air is a no-op.

Grids move rigidly (translate + rotate, never scale), and each one
rotates as a whole — the classic use is a static "world" grid plus a
handful of dynamic object grids:

```rust,noplayground
{{#include ../../../crates/roxlap-scene/examples/book_scene_graph.rs:scene_grids}}
```

`render_sky = false` on object grids matters more than it looks: sky
pixels are rendered in each grid's local frame, so a rotating ship
that paints its own sky visibly fights the world's sky. One grid owns
the sky; the rest opt out.

## Editing voxels

The core edit API is three methods, one convention: `Some(colour)`
inserts solid voxels, `None` carves to air.

```rust,noplayground
{{#include ../../../crates/roxlap-scene/examples/book_scene_graph.rs:edits}}
```

- `set_voxel(pos, colour)` — one voxel.
- `set_rect(lo, hi, colour)` — an axis-aligned box, **inclusive on
  both ends**, any corner order.
- `set_sphere(centre, radius, colour)` — Euclidean ball.

All three take grid-local coordinates and may span any number of
chunks. They are also the engine's runtime carving path — the cave
demo's plasma bullets and the particle system's debris craters
(chapter 8) are `set_sphere(.., None)` at heart.

Cheap point queries pair with them:

```rust,noplayground
{{#include ../../../crates/roxlap-scene/examples/book_scene_graph.rs:queries}}
```

### The recolour gotcha

The slab format stores *surfaces*, and insertion fills *air* — so
inserting a new colour over already-solid voxels changes nothing:

```rust,noplayground
{{#include ../../../crates/roxlap-scene/examples/book_scene_graph.rs:recolour}}
```

This is not a rare corner: "paint that wall red" is exactly this
operation. Remember the idiom — **recolour = carve, then insert**.

### Colour callbacks

A carve exposes interior walls the artist never coloured; the plain
edits paint them black. The `_with_colfunc` variants
(`set_rect_with_colfunc` / `set_sphere_with_colfunc`, with
`SpanOp::Carve` or `SpanOp::Insert`) ask a closure for every touched
voxel instead — position-dependent colour, jitter, texture lookups:

```rust,noplayground
{{#include ../../../crates/roxlap-scene/examples/book_scene_graph.rs:colfunc}}
```

The closure receives **grid-local** coordinates (not chunk-local), so
a gradient stays continuous across chunk seams. This is roxlap's
answer to Voxlap's `vx5.colfunc` global — as a parameter instead of
engine state.

One more thing edits do implicitly: each write bumps the touched
chunk's version counter and dirty extent, which is how the renderers
(and the streaming persistence below) know what changed. You don't
manage that — but after bulk terrain edits you will want to re-bake
lighting (`Grid::bake`, chapter 6).

## Snapshots

`Scene::save_snapshot()` serialises the whole scene to bytes —
chunks, transforms, per-grid config (sky/LOD/streaming settings) and
edit-version counters — inside a **versioned envelope**, so a newer
engine refuses (rather than misreads) a snapshot format it doesn't
know. `Scene::load_snapshot(&bytes)` restores it:

```rust,noplayground
{{#include ../../../crates/roxlap-scene/examples/book_scene_graph.rs:snapshot}}
```

Two things a snapshot cannot carry:

- **Host hooks.** Generators and chunk stores (below) are your code —
  rebind them after loading, keyed on `Grid::name`. That is what the
  `name` field exists for.
- **Store-only chunks.** Only *materialised* chunks are serialised; a
  streamed-out edited chunk lives in your `ChunkStore`, which you
  persist alongside the snapshot.

(If you want your own on-disk format instead of the envelope,
`Scene::to_snapshot()` / `Scene::from_snapshot()` expose the plain
serde value underneath.)

## Streaming & procedural generation

For worlds bigger than memory — or generated on the fly — a grid can
stream: you attach a **`ChunkGenerator`** and a `StreamRadius`, then
pump once per frame with the camera's world position.

```rust,noplayground
{{#include ../../../crates/roxlap-scene/examples/book_scene_graph.rs:generator}}
```

The contract that makes streaming sound: `generate` is a
**deterministic** function of the chunk index (plus the generator's
own config). Evicting a pristine chunk is then lossless — walking back
regenerates the identical bytes. (A generator can also decline indices
via `should_generate` when whole layers have no content.)

```rust,noplayground
{{#include ../../../crates/roxlap-scene/examples/book_scene_graph.rs:streaming}}
```

The radii semantics: chunks with AABB distance ≤ `r_active` from the
camera (grid-local voxel units) must be loaded; chunks beyond
`r_evict` are dropped; the band between is hysteresis — untouched in
either direction — so a camera hovering at a boundary doesn't thrash
generate/evict cycles.

The example pumps `pump_streaming_sync`, which generates inline —
deterministic, good for tools and tests. Games call
**`pump_streaming`**, which dispatches generation onto a background
rayon pool and installs finished chunks on later pumps (frame-rate
stays smooth while terrain builds); `Scene::set_streaming_threads(n)`
bounds the pool.

### Persisting edits with `ChunkStore`

Determinism covers *pristine* chunks. An edited chunk — the player
dug a hole — would silently revert to generator output on
evict + re-stream. A `ChunkStore` closes that hole: eviction hands
every edited chunk (version ≠ 0) to `store`, and stream-in consults
`load` before the generator (a stored chunk always wins):

```rust,noplayground
{{#include ../../../crates/roxlap-scene/examples/book_scene_graph.rs:chunk_store}}
```

```rust,noplayground
{{#include ../../../crates/roxlap-scene/examples/book_scene_graph.rs:streaming_edit}}
```

`store` runs inline during eviction — keep it cheap (queue bytes for
a writer thread). `load` may block under `pump_streaming` (it runs on
the background pool) but runs inline under the synchronous paths.

## Further reading

- [docs.rs/roxlap-scene](https://docs.rs/roxlap-scene) — the full API,
  including the world queries deferred to chapter 10.
- [`PORTING-SCENE.md`](https://github.com/NCrashed/roxlap/blob/master/docs/porting/PORTING-SCENE.md)
  — why the scene graph is built this way (the S1–S7 design history).
