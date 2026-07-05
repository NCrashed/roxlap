# The asset pipeline

Everything roxlap loads lives in
[`roxlap-formats`](https://docs.rs/roxlap-formats) — a standalone
crate with **no renderer dependency**, so the same parsers power your
game, your level editor, and your asset-conversion scripts. Every
format follows one discipline: a hand-written parser (no decoder
dependencies), a per-format `ParseError` that says exactly what broke
and where, hardening against crafted files (allocation caps, bounds
checks), and — for every format the engine can author — a symmetric
writer.

The snippets come from a headless, assertion-checked example:

```sh
cargo run -p roxlap-formats --example book_assets
```

## Authoring in MagicaVoxel: `.vox`

The industry-standard voxel editor is the intended authoring tool.
`vox::parse` reads what every MagicaVoxel build writes — `SIZE` +
`XYZI` model pairs and the `RGBA` palette (files without one get the
official default palette); multi-model files yield their models in
file order:

```rust,noplayground
{{#include ../../../crates/roxlap-formats/examples/book_assets.rs:vox}}
```

Two things the conversion handles for you:

- **The z-flip.** MagicaVoxel is z-up, roxlap is z-down; `to_kv6`
  maps `(x, y, z)` → `(x, y, zsiz−1−z)`, so right-side-up in the
  editor is right-side-up in the engine.
- **Colour packing.** Palette colours become voxlap-packed
  `0x80_RR_GG_BB` (neutral brightness). Palette *alpha* is dropped —
  translucency in roxlap is a material, not a colour channel, so pair
  the import with a colour→material map
  ([chapter 6](lighting.md)).

Scope note: the `.vox` scene graph (`nTRN`/`nGRP` transforms),
materials and cameras are skipped — models come without world
placement; your game places them.

## The Voxlap heritage formats

These are the formats Ken Silverman's tools and games produced —
supporting them is why two decades of assets load directly:

| Format | What it is | Module |
|---|---|---|
| `.kv6` | One voxel sprite model (Slab6) | `kv6` |
| `.kvx` | Build-engine voxel model (Shadow Warrior, Blood) | `kvx` |
| `.vxl` | A whole voxel world, column-compressed (Ace of Spades maps) | `vxl` |
| `.kfa` | An animation rig over a kv6 (Ken's animator) | `kfa` |

Each has `parse(bytes)` and `serialize(..)`, and the round trip is
**byte-stable** — `serialize(parse(bytes)) == bytes` — so a tool can
rewrite an asset it only meant to inspect without churning it:

```rust,noplayground
{{#include ../../../crates/roxlap-formats/examples/book_assets.rs:kv6_roundtrip}}
```

Worlds have a code-authoring path too — `Vxl::from_dense` folds any
occupancy predicate into the slab format (this is also how procedural
generators build chunks, [chapter 3](scene-graph.md)):

```rust,noplayground
{{#include ../../../crates/roxlap-formats/examples/book_assets.rs:vxl_roundtrip}}
```

## roxlap's own containers

Two formats are roxlap-native, built for what the heritage formats
can't hold:

- **`.rvc` — voxel clips** ([chapter 7](sprites.md)): fixed-bbox
  animation flipbooks, keyframes + deltas, per-frame durations:

```rust,noplayground
{{#include ../../../crates/roxlap-formats/examples/book_assets.rs:rvc_roundtrip}}
```

- **`.rkc` — characters** (`character` module): meshes + skeleton +
  animation clips in one chunked, forward-compatible container (a
  reader skips chunk types it doesn't know, so old engines open new
  files). Bone attachments reference static meshes *or* voxel clips.

## 2D art: GIF and PNG import

With the `gif` / `png` cargo features, `gif_import` / `png_import`
turn animated GIFs and PNG sequences (or APNGs) into voxel clips —
each frame a 1-voxel-thick cutout slab, transparency and frame timing
preserved. This is the Doom-style billboard pipeline from
[chapter 7](sprites.md): 2D sprite sheets in, shadow-casting voxel
objects out.

## Saves

Scene snapshots ([chapter 3](scene-graph.md)) are the save-game
format: a versioned envelope holding every grid's config plus its
chunks — each chunk encoded with the same `vxl::serialize` shown
above. There is no separate "save format" to learn; a snapshot is
made of the wire formats on this page.

## Further reading

- [docs.rs/roxlap-formats](https://docs.rs/roxlap-formats) — every
  module on this page, with per-field format documentation.
- `roxlap-formats/examples/parse_kv6.rs` — a minimal
  file-from-disk loader.
- The `edit` module — the carve/insert span machinery from
  [chapter 3](scene-graph.md) lives in this same crate, so level
  tools get it without the renderer.
