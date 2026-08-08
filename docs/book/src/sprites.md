# Sprites & animation

Everything that moves in a roxlap world is a **sprite**: a voxel model
(`.kv6`) instanced into the scene with its own f32 transform. Sprites
are not a separate rendering world — they march through the same
per-pixel DDA, so they get the scene's lighting, stylized shadows
(cast *and* receive), and materials for free. On top of static models
sit three animation systems: flipbook **voxel clips**, skeletal
**characters**, and Doom-style **billboards**.

The snippets come from a runnable example — three gems (one plain,
one scaled, one spinning), a pulsing animated clip, and the same clip
as a camera-facing billboard:

```sh
cargo run --release -p roxlap-render --example book_sprites
```

## Models & instances

The unit of registration is a **model** (one `Kv6` volume); the unit
of placement is an **instance**. Register once, instance many:

```rust,noplayground
{{#include ../../../crates/roxlap-render/examples/book_sprites.rs:model_instances}}
```

Where models come from: paint one in
[Demiurg](https://github.com/NCrashed/demiurg) (the roxlap asset
editor) or MagicaVoxel ([chapter 11](assets.md)), parse a `.kv6` file
([`roxlap_formats::kv6`](https://docs.rs/roxlap-formats)), or build
one in code —
`Kv6::from_fn` (surface-only, as here), `solid_cube` / `solid_box`,
or `from_fn_keep_interior` for filled volumes
([chapter 6](lighting.md)).

The pose type, `DynSpriteTransform`, is a position plus a
local→world basis (all f32 — [chapter 2](concepts.md)'s precision
split). Two things the basis buys you:

- **Scale** — there is no scale field; scaling the basis vectors *is*
  the scale, and non-uniform or skewed bases work too.
- **Rotation** — any orthogonal basis. For per-frame motion, rewrite
  the transform; it's an in-place update, not a re-upload:

```rust,noplayground
{{#include ../../../crates/roxlap-render/examples/book_sprites.rs:per_frame}}
```

For bulk motion use `set_sprite_instance_transforms` (plural) — one
batched flush for hundreds of instances (the particle system in
[chapter 8](particles.md) rides exactly that path). Instances are
removed with `remove_sprite_instance`; models with
`remove_sprite_model` + `compact_sprite_models`. Per-instance
appearance knobs: `set_sprite_instance_tint` / `_alpha` /
`_material` / `_lighting` / `_shadow_flags`.

## Animated voxel clips (`.rvc`)

A **voxel clip** is the "GIF for voxels": a fixed-bbox sequence of
frames stored as keyframes + deltas, with per-frame durations and a
loop mode. Author one from kv6 frames in code (below), import a GIF
(further down), or load a `.rvc` file:

```rust,noplayground
{{#include ../../../crates/roxlap-render/examples/book_sprites.rs:clip_build}}
```

(`from_kv6_frames_auto` picks keyframe-vs-delta per frame by encoded
cost — the turnkey choice for real assets.)

Register the decoded clip and place instances — `_playing` attaches a
per-instance player that `tick` advances; frame swaps cost O(changed
frame), not O(volume):

```rust,noplayground
{{#include ../../../crates/roxlap-render/examples/book_sprites.rs:clip_instances}}
```

The player is fully scriptable: `set_clip_instance_paused` /
`_speed` / `_frame` (scrub) / `_clock_ms`, and
`set_clip_instance_clip` swaps which animation an instance plays —
that swap is how billboard actors change state below. For live
authoring there is `update_clip_frame` (rewrite one frame in place),
and for long cutscene-grade clips **streaming clips**
(`add_streaming_clip`) keep only the current frame resident instead
of the whole timeline.

## Billboards: the Doom look

A billboard is a clip instance that auto-faces the camera — the
Doom/Build sprite. Because it is still a voxel object (a flat slab of
voxels), it casts and receives real shadows and takes materials, no
special path:

```rust,noplayground
{{#include ../../../crates/roxlap-render/examples/book_sprites.rs:billboard}}
```

`Cylindrical` rotates about the vertical only (the classic look — and
its cast shadow stays a sane vertical card); `Spherical` also pitches
with the view. `tick` re-faces every billboard each frame.

**Which way is up** is the other half, and a separate knob:
`BillboardUp` (`set_billboard_up`, `BillboardActorDef::up`,
`set_actor_up`). `World` is world up — the default, and what a
Doom/Build scene wants. `Camera` takes the camera's own up, so a
**rolled** camera never leaves the art leaning; `Axis(v)` takes a
world-space axis you supply — hand over a grid's up and cards stand on
that grid's deck however it is tilted, and `Cylindrical` yaws about it
instead of the world vertical. Whichever you pick, the card turns
about its anchor, so a figure's feet stay planted. The trade is
shadows: a card that tracks the camera casts a shadow that turns as
you orbit, which is why world up stays the default.

**Importing 2D art**: with the `gif` / `png` cargo features,
`gif_import` / `png_import` turn an animated GIF or a PNG sequence
into a voxel clip — each frame a 1-voxel-thick cutout slab, palette
and timing preserved. That's the whole asset pipeline for Doom-style
monsters: draw (or rip) sprite sheets, import, place.

**Billboard actors** (`add_billboard_actor`) are the high-level layer
for such monsters: a `BillboardActorDef` holds named **states**
("walk", "attack", …), each with **N-way directional clips** (8-way
in the demo); every `tick` the renderer picks the clip from the
camera's bearing versus the actor's facing, faces it, and
advances its animation. Drive gameplay with `set_actor_state` /
`set_actor_transform` / `set_actor_tint`. An actor riding a moving
body — crew on a turning ship — wants `set_actor_pose` instead: give
it `ActorFacing::Dir(world_direction)` and `BillboardUp::Axis(deck_up)`
and both the card's vertical and its directional sector are measured
in the deck's frame, so it neither leans nor spins on the spot as the
hull turns. (Composing that facing into a world *yaw* is what does not
survive a tilted rotation: flattening and rotating do not commute.)
Lighting per actor or
instance via `BillboardLighting` — `FullBright` for pickups and
projectiles that shouldn't darken in shadow.
`BillboardActorDef::scale` sets world units per slab voxel (`1.0` =
the classic 1:1) — the knob for a giant boss or a half-height imp
from the same sheet; it composes with the clip's own
`voxel_world_size` and behaves identically on both backends. The
**Doom** demo scene is the worked example (it synthesises its GIFs
at startup, so there is no binary asset to squint at).

## Characters (`.rkc`) and KFA rigs

For articulated models roxlap has two systems, both asset-driven —
the book shows no code because the code is three calls; the substance
is in the assets:

- **`.rkc` characters** — the modern container: meshes + a bone
  skeleton + animation clips, where a bone attachment is either a
  static mesh or a voxel clip (a flickering torch in a hand). The
  whole lifecycle is four calls:
  `roxlap_formats::character::parse(&bytes)` loads the container,
  `add_character(&ch, clip)` spawns one playing the chosen clip,
  `tick` advances it every frame, and
  `set_character_world_transform` moves it. Author them in
  [Demiurg](https://github.com/NCrashed/demiurg), the roxlap asset
  editor ([chapter 11](assets.md)); the wire format is
  [`roxlap_formats::character`](https://docs.rs/roxlap-formats), and
  the **Animation** demo scene is the loading example.
- **KFA rigs** — Ken Silverman's original animated-sprite format
  (`.kfa`), supported for asset compatibility: the host owns the
  `KfaSprite`s and calls `set_kfa_sprites` / `update_kfa_poses`. See
  the `roxlap-host` demo. New projects should prefer `.rkc`.

## Further reading

- Demo scenes: **Sprites** (instancing + transforms), **Animation**
  (clips + characters), **Doom** (billboards + actors).
- [`PORTING-SPRITE-API.md`](https://github.com/NCrashed/roxlap/blob/master/docs/porting/PORTING-SPRITE-API.md),
  [`PORTING-VOXEL-CLIP.md`](https://github.com/NCrashed/roxlap/blob/master/docs/porting/PORTING-VOXEL-CLIP.md),
  [`PORTING-BILLBOARD.md`](https://github.com/NCrashed/roxlap/blob/master/docs/porting/PORTING-BILLBOARD.md)
  — the design histories (why clips are keyframe+delta, why a
  billboard is a 1-voxel slab and not a textured quad).
