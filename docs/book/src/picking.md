# Picking & world queries

"What did the player click?" and "what's in front of me?" are the two
questions every interactive voxel game asks constantly. roxlap
answers them at two levels: **screen→world picking** on the renderer
(unproject a pixel) and **world queries** on the scene (march a ray,
inspect a voxel). Both resolve to the same currency: a grid id plus a
grid-local voxel coordinate — exactly what the edit API from
[chapter 3](scene-graph.md) consumes, so *pick → carve* is two calls.

The snippets come from a runnable example — an orbiting camera that
outlines the voxel under the screen centre every frame, and carves a
crater wherever you click:

```sh
cargo run --release -p roxlap-render --example book_picking
```

## Two picking paths — and when to use each

**Path 1: `view_ray` + `Scene::raycast`** — geometric. Unproject the
pixel into a world-space ray under whichever projection the last
frame used (CPU and GPU project differently; `view_ray` hides that),
then march the scene's voxels:

```rust,noplayground
{{#include ../../../crates/roxlap-render/examples/book_picking.rs:hover_raycast}}
```

No depth buffer involved, identical on both backends, cheap enough
for **every frame** — hover highlights, crosshair targeting, AI
line-of-sight. The `RayHit` gives you the grid, the grid-local voxel,
the world-space hit point, the distance `t`, and the voxel's colour.

**Path 2: `pick`** — depth-based. Read the rendered frame's z-buffer
at the pixel, reconstruct the world point, resolve it to a voxel:

```rust,noplayground
{{#include ../../../crates/roxlap-render/examples/book_picking.rs:click_pick}}
```

`pick` returns exactly what the player *sees* (it reads the frame that
was actually rendered), but its cost is backend-dependent: the CPU
backend reads an in-memory z-buffer (free), the GPU backend stages
the depth buffer and blocks on a device poll. That makes it a
**click-time** call, not a per-frame one — the
`Feature::FreePickDepth` probe from [chapter 4](rendering.md) is the
queryable form of this distinction. Two more semantics to know:
sprites don't occlude the pick (a cursor sprite under the pointer is
transparent to it), and the depth belongs to the *last rendered*
frame — pass the camera that frame used.

The lower-level pieces are exposed too: `pick_depth` (just the
distance), `pixel_ray` (just the un-normalised direction) — for tile
selection, intersect the view ray with a plane instead of the voxels.

## World queries without a screen

Both scene-level queries work headless (tools, servers, AI):

- **`Scene::raycast(origin, dir, max_dist)`** — the same march as
  path 1, minus the unprojection. Grid rotations are handled: the ray
  is transformed into each grid's local frame, and the nearest hit
  across all grids wins.
- **`Scene::resolve_voxel(world, ray_dir)`** — "which voxel is this
  world point on?": used internally by `pick` to turn a depth hit
  into a voxel address; useful whenever you already have a world
  point from other means. The `ray_dir` nudges the sample off the
  surface so a point *on* a face resolves to the solid side.
- **`Grid::voxel_solid` / `voxel_color`** — the point queries from
  [chapter 3](scene-graph.md), for when you already know the grid.

## Collision: the interim pattern

roxlap has no character controller yet — a dedicated stage (CC) is
planned. Until then the demos hand-roll a serviceable pattern:
per-axis **slide with collision** — propose a movement step, test
each axis independently with a small-radius voxel probe
(`voxel_solid` around the player's capsule points), and zero the
axes that would penetrate. The reference implementation is
[`roxlap-scene-demo/src/collision.rs`](https://github.com/NCrashed/roxlap/blob/master/crates/roxlap-scene-demo/src/collision.rs)
(~a screenful of code) — copy it, tune the radius, and expect the CC
stage to supersede it with a real controller.

## Further reading

- The **Picking** demo scene — cursor-follow pick mode over a
  multi-grid world
  (`ROXLAP_SCENE=Picking cargo run --release -p roxlap-scene-demo`).
- [docs.rs/roxlap-render](https://docs.rs/roxlap-render) (`pick`,
  `view_ray`, `pick_depth`, `pixel_ray`, `PickHit`, `Ray`) and
  [docs.rs/roxlap-scene](https://docs.rs/roxlap-scene) (`raycast`,
  `resolve_voxel`, `RayHit`).
