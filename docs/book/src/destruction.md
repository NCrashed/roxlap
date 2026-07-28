# Destruction

The signature voxlap moment: carve a tunnel under an overhang and the
unsupported rock does not hang there politely — it breaks off, falls,
and bursts into rubble where it lands. roxlap's destruction pipeline
reproduces that loop on top of pieces earlier chapters covered — edits,
sprites ([chapter 7](sprites.md)), particles
([chapter 8](particles.md)) — with two engine parts of its own:

- **Island detection** (`roxlap-scene`): after any carve,
  `detect_islands` finds the voxel regions that no longer connect to
  support. It floods the chunk format's RLE *runs* directly (one run is
  one search node — no dense decode), 6-connected, seeded from the
  carve's bounding box. A region that reaches a **bedrock-anchored**
  column bottom is supported; one that grows past the **budget** is
  *presumed* supported (a cheap early exit); everything else comes back
  as an `Island` — its voxels, colours and bounds.
- **`DebrisSystem`** (`roxlap-render`): the crumble loop. It extracts
  an island from its grid, optionally fractures it per material,
  registers each piece as a falling sprite at the exact world pose of
  the voxels it replaced, integrates the fall, and reports impacts for
  the shatter. Like the particle system it is host-owned and split into
  a pure simulation half (`spawn_island` / `update` — unit-testable, no
  renderer) and a facade half (`sync` — batched sprite updates).

The snippets below come from a runnable, windowless example that prints
what each stage does:

```sh
cargo run -p roxlap-render --example book_destruction
```

## Detecting what came loose

A scene with something to lose — a floor, an anchored pillar, a beam
cantilevered off it, crystal grown near the tip:

```rust,no_run,noplayground
{{#include ../../../crates/roxlap-render/examples/book_destruction.rs:scene}}
```

Carve, then ask. Detection is a separate call on purpose: hosts decide
*when* it runs (the cave demo runs it on its background carve worker,
right after the carve, on the same chunk):

```rust,no_run,noplayground
{{#include ../../../crates/roxlap-render/examples/book_destruction.rs:detect}}
```

Two properties matter in practice:

- **The budget is a design knob, not just a guard.** A detached region
  bigger than `budget` voxels stays put — `DEFAULT_ISLAND_BUDGET`
  (4096) means shooting the single support out from under a whole
  gallery will *not* drop the gallery. Raise it if your game wants
  building-sized collapses; the flood's cost is
  `O(min(region, budget))` per component, so the worst case is priced
  in advance.
- **Support means standing on the bottom — as a fact, not a format
  invariant.** A region is supported when it reaches a run that
  genuinely extends to its column's bottom (local z = 255). Since
  carve-through-floor that is a property of the column's bytes: dig
  the floor out from under a pillar and its anchor is gone — the
  hanging top comes back as an island and falls, exactly as physics
  suggests.

## Falling

```rust,no_run,noplayground
{{#include ../../../crates/roxlap-render/examples/book_destruction.rs:spawn}}
```

`spawn_island` does the irreversible part: the island's voxels leave
the grid (a coalesced carve plus one incremental re-bake — **re-mip is
your job**, exactly as for your own edits; fold the island's bbox into
whatever `remip_bbox` call your carve path already makes, or distant
mips keep drawing the rock that just fell). The sprite appears at the
island's `world_pivot`, so nothing visually jumps — the voxels become
a falling body in place.

Fracture is data, keyed by material: `Chunks { cell }` breaks matter
into rounded jittered-Voronoi lumps (stone), `Shards { plates }`
slices it with near-parallel planes (glass, crystal); unmapped
materials fall `Whole`. A mixed island splits per material group, and
with the colour→material map installed the fragment sprites register
**with** it — a crystal shard keeps its translucent, emissive material
and glows all the way down ([chapter 6](lighting.md)). Fragments get a
small outward drift (`fracture_impulse`) so a broken slab visibly
comes apart instead of falling as a stack.

Physics is deliberately voxlap-simple and deterministic: vertical
gravity with a terminal clamp, a cosmetic spin hashed from the
island's position (identical scenes crumble identically), and
collision against the scene's solid voxels with the descent marched in
substeps — a fast body cannot skip through a one-voxel shelf on a slow
frame. The AABB is the island's unrotated box; the spin never affects
collision.

## Landing and shattering

```rust,no_run,noplayground
{{#include ../../../crates/roxlap-render/examples/book_destruction.rs:tick}}
```

A windowed host wires the impact to the particle system from
[chapter 8](particles.md) — `burst_sites()` hands back one world-space
site per island voxel, in that voxel's own colour, positioned where
the body *landed*:

```rust,ignore
for hit in debris.drain_impacts() {
    particles.voxel_debris(&hit.burst_sites(), from, 4.0..9.0, &burst_def);
}
```

The burst is the same machinery `carve_debris` uses for bullet
craters, so crater debris and crumble debris look and behave like the
same rock. Feed `hit.pos` to your impact sound while you are at it —
the cave demo routes it through the same occlusion-shaded boom as a
bullet hit ([chapter 9](audio.md)).

## The cave demo's wiring

`roxlap-cave-demo` shows the full production shape ([chapter
15](demo-tour.md)): detection runs on the **background carve worker**
(same thread that already carves, relights and re-mips the chunk
clone), the extraction happens there too so the batch's re-mip covers
it, and the main thread only spawns the returned islands and ticks the
system. Rock is mapped to `Chunks`, crystals to `Shards`, and
`ROXLAP_NO_CRUMBLE=1` switches the whole thing back to plain carves.
