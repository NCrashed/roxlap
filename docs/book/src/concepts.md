# Concepts & conventions

Five conventions run through every roxlap API. They are all inherited
from Voxlap — that heritage is what lets twenty years of `.vxl` /
`.kv6` assets load unchanged — and three of them are backwards from
what most 3D engines taught you. Absorb this chapter before writing
real code; everything else in the book assumes it.

The snippets here come from a runnable, assertion-checked example:

```sh
cargo run -p roxlap-core --example book_conventions
```

## +z is down

The world axes are `x` = east, `y` = north, and `z` pointing **down**
into the map. Consequences you will hit immediately:

- "Up" is toward *smaller* z. A tower is built by decreasing z; a mine
  shaft by increasing it.
- Within a chunk, z runs `0..256`: `z = 0` is the top (sky), `z = 255`
  the bottom (bedrock). Ground surfaces typically sit somewhere in the
  200s, filling downward — that is why the quickstart's grass slab is
  `set_rect(.., z=210 ..= z=254)` with the dome *above* it at
  `z = 205`.
- Positive camera pitch aims downward:

```rust,noplayground
{{#include ../../../crates/roxlap-core/examples/book_conventions.rs:z_down}}
```

### The horizontal mirror

Physically, the triple (east, north, up) is left-handed, so a
faithfully projected view is **horizontally mirrored** relative to a
real camera at the same heading — your geometric right lands on
screen-left. This is Voxlap's native projection, reproduced
deliberately (it is baked into the bit-exact reference goldens); it is
not a bug, and most games never notice it.

If your project needs an un-mirrored world, do **not** negate the
camera's `right` vector — that breaks basis chirality (next section)
and silently culls every sprite. Fix it on the consumer side instead:
mirror one world axis when you place content, or negate your yaw
input. The full reasoning lives in the
[`roxlap-core` crate docs](https://docs.rs/roxlap-core)
("World handedness and the horizontal mirror").

## One voxel = one world unit

There is no separate voxel-size parameter: world coordinates are voxel
coordinates, scaled 1:1 (a grid's placement can translate and rotate
them, but not scale — see [the scene graph](scene-graph.md)). Grids
store voxels in chunks of **128×128×256** (`CHUNK_SIZE_XY` /
`CHUNK_SIZE_Z`); you address voxels in grid-local integer coordinates
and the chunk decomposition is invisible to you.

## Packed colours: `0x80_RR_GG_BB`

Voxel colours are a `u32` with the RGB in the low 24 bits — and the
high byte is **not alpha**. It is the per-voxel shading intensity:
`0x80` means "neutral, unlit". Passing `0xFF_...` gives you an
over-bright voxel, not an opaque one; passing `0x00_...` a black-lit
one. Lighting bakes (chapter 6) rewrite this byte per voxel — ambient
and AO live there.

```rust,noplayground
{{#include ../../../crates/roxlap-core/examples/book_conventions.rs:packed_color}}
```

One packing exception: *framebuffer* and sky colours are plain
`0x00_RR_GG_BB` — no intensity byte. The quickstart's `SKY` constant
is one of those. (Transparency is separate machinery too: alpha-,
additive- and volumetric-blended voxels go through the material
palette, chapter 6 — never through this byte.)

## The camera basis — and the chirality footgun

A [`Camera`](https://docs.rs/roxlap-core) is a position plus three
orthonormal f64 axes: `right`, `down`, `forward` (Voxlap's
`ist`/`ihe`/`ifo`). The engine requires the **right-handed** relation
`right × down = +forward`. The renderer will happily draw terrain
under a basis with the opposite chirality — but the sprite frustum
cull rejects *everything*, which presents as "my sprites disappeared"
with no error anywhere.

The rule that keeps you safe: **always build cameras with the
constructors** — `Camera::from_yaw_pitch`, `Camera::orbit`,
`Camera::look_at`. Never hand-roll the axes, and never build a camera
by rotating `Camera::default()` — the default is a `.vxl`-header
placeholder whose basis is left-handed:

```rust,noplayground
{{#include ../../../crates/roxlap-core/examples/book_conventions.rs:camera_basis}}
```

Yaw/pitch conventions: `yaw = 0` looks down `+x`, increasing yaw turns
toward `+y`; `pitch = 0` is level, positive pitch aims down.

## The precision model

Three numeric domains, each with a job:

- **f64 for the world.** Camera position/basis and every
  `GridTransform` (grid origin + rotation quaternion) are f64, so a
  huge world doesn't jitter far from the origin.
- **i32 for voxels.** Edits and queries take grid-local integer voxel
  coordinates (`glam::IVec3`).
- **f32 for sprite instances.** Per-instance sprite/clip transforms
  (chapter 7) are f32 — cheap to stream in bulk, precise enough for
  object-sized placements.

## Where these conventions come from

They are Voxlap's, kept so assets and ported math work unchanged. The
design history — what was ported faithfully, what was replaced
(the renderer itself is a clean-room per-pixel DDA, not Voxlap's
raycaster) — lives in
[`docs/porting/`](https://github.com/NCrashed/roxlap/tree/master/docs/porting),
starting from `PORTING-RUST.md` and `PORTING-SCENE.md`.
