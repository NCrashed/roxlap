# Lighting & materials

roxlap lights a scene in two layers. **Baked** lighting lives in each
voxel's `VoxColor` brightness byte — remember from
[chapter 2](concepts.md): that byte is shading intensity, not alpha — and
costs nothing per frame. **Runtime** lighting is a per-frame rig of
sun + point + spot lights with stylized hard voxel shadows, composited
on top. Both layers, and the transparent-voxel materials at the end of
this chapter, run on both backends.

The snippets come from a runnable example — a courtyard under a
sweeping shadow-casting sun, orbiting coloured points, a spot cone, a
glass wall and a volumetric fog cloud:

```sh
cargo run --release -p roxlap-render --example book_lighting
```

The formula that ties the layers together:

```text
pixel = albedo × (baked byte × rig.ambient)  +  Σ direct light terms
```

The baked byte *is* the ambient/AO channel. With `lights: None` (the
default) you get exactly the classic render — baked byte only.

## Baked lighting & ambient occlusion

`Grid::bake(mode)` walks the grid once and writes shading into every
voxel's brightness byte. The two `BakeMode`s:

- **`BakeMode::Directional`** — estnorm shading, the classic Voxlap
  look (surface-normal-based sun shading, baked). Use it standalone
  when you don't run a light rig — the cave and terrain demos ship
  this.
- **`BakeMode::AmbientOcclusion(AoParams)`** — crevices, pillar bases
  and inner corners darken. This is the right bake *under* a runtime
  rig: the rig treats the byte as its ambient fill, so AO gives
  contact shading everywhere the dynamic lights don't reach.

```rust,noplayground
{{#include ../../../crates/roxlap-render/examples/book_lighting.rs:bake}}
```

The bake is neighbour-aware across chunk seams in all three axes (no
brightness discontinuities at chunk borders) and rayon-parallel.
Costs and cadence:

- Bake once after building a grid — it is a bulk operation.
- After a runtime carve, **don't** re-bake the grid: pass the edit's
  bbox to `bake_bbox` — it re-bakes a few hundred columns
  instead of whole chunks (the cave demo measured ~0.04 ms against
  4–7 ms).
- Streaming grids bake per chunk as they stream in — a scene-wide
  bake at startup would miss every chunk generated later.

## The runtime rig

Lighting is **per-frame state**: build a
[`LightRig`](https://docs.rs/roxlap-render) and set it on
`FrameParams::lights`. There are deliberately no light setters on the
renderer — lights flow the same way sky and fog already do, so
"remove the light" is just "stop passing it".

```rust,noplayground
{{#include ../../../crates/roxlap-render/examples/book_lighting.rs:light_rig}}
```

The three light types:

- **Sun** (`DirectionalLight`) — one per scene, `direction` is the way
  the light *travels*. With +z down, keep a positive z component to
  stay above the horizon.
- **Point** (`PointLight`) — world position, hard `radius` cutoff
  (zero contribution beyond it; keep radii tight, they bound the
  shading work).
- **Spot** (`SpotLight`) — a point light with a cone: `direction` is
  the axis, `inner_angle_deg`/`outer_angle_deg` are half-angles with a
  smoothstep falloff between them. Internally a spot *is* a point
  light (a point is the 180°-cone degenerate), so spots share the
  point-light count and shadow budgets.

**Shadows** are stylized hard voxel shadows, flagged per light with
`casts_shadow`. Casters are budgeted per frame: only the first few
flagged lights actually cast; the rest are demoted to shadowless with
a log warning — never dropped. The rig-level knobs:
`shadow_strength` (how dark), `shadow_bias_voxels` (~1.5 kills
self-shadow acne), `shadow_max_dist` (sun-ray length cap).

**Stylized mode**: `bands ≥ 1` quantizes the diffuse into discrete
cel levels, and the sun term drives a gradient from `shadow_tint`
(cool, unlit) to the sun colour (warm, lit) — hue-shifted shadows
instead of plain darkening, which keeps the retro identity instead of
reading as generic Phong. `bands: 0` is smooth diffuse. Compare them
live in the Lighting demo scene (`J` toggles, `[`/`]` change bands).

## Materials: transparent voxels

Both backends march rays strictly front-to-back, which makes
order-correct transparency *free* — no depth sorting, no OIT scheme:
a translucent voxel just composites over whatever the ray finds
behind it.

Materials stay out of the colour word (its high byte is taken —
that's the brightness/ambient story above). Instead the renderer owns
a **256-entry material palette**; a voxel carries a one-byte material
id. Id 0 is permanently opaque, so anything without material data
renders exactly as before. A `Material` is an opacity plus a
`BlendMode`:

- **`Opaque`** — first hit wins (the default path).
- **`AlphaBlend`** — front-to-back `over` compositing. Glass, shells,
  windows. Opacity is per *surface*, independent of thickness.
- **`Additive`** — adds light without occluding, order-independent.
  Spell glows, fire, muzzle flashes.
- **`Volumetric`** — Beer–Lambert absorption: per-cell opacity is
  `1 − (1 − α)^path_length`, so a **filled** volume reads denser
  through its core than its rim. True smoke, fog, murky water.

Materials attach at three levels:

```rust,noplayground
{{#include ../../../crates/roxlap-render/examples/book_lighting.rs:materials}}
```

- **Terrain** — `set_terrain_materials` maps voxel *colours* (low 24
  bits) to material ids: glass walls and water pools built with plain
  `set_rect` calls.
- **Whole sprite instance** — `set_sprite_instance_material(id, mat)`;
  pair with `set_sprite_instance_alpha` to pulse opacity per frame
  without touching the volume.
- **Per voxel** — `add_sprite_model_with_materials` /
  `add_voxel_clip_with_materials` take a colour→material map, so one
  model mixes opaque and translucent voxels (a window: opaque frame,
  glass panes — static or animated).

### The hollow-shell trap

`Volumetric` needs actual interior voxels to absorb through — and the
standard `Kv6::from_fn` constructor culls interiors (a normal sprite
only needs its shell). Build filled volumes with
`from_fn_keep_interior`:

```rust,noplayground
{{#include ../../../crates/roxlap-render/examples/book_lighting.rs:volumetric}}
```

If your volumetric cloud renders as two thin films (front and back
face only), this is why.

### Backend parity

Translucent compositing runs on **both** backends — CPU since the TV
stage's march rework, GPU via its own per-span accumulation paths
(sprites and terrain alike). The residual gaps are stylistic, not
structural: translucent *sprites* stay flat-lit (no side-shades, fog
or tint on the translucent layers), and compositing is per-pass — a
translucent sprite over translucent terrain resolves per pass rather
than through one unified blend. Neither has mattered in practice; if
one bites you, `supports()` ([chapter 4](rendering.md)) is still the
place a future split would surface.

## Emissive voxels & baked glow

A material can also **emit**: `Material { emissive, .. }` (or the
`Material::glow(e)` shorthand, or `.with_emissive(e)` on any blend
mode) renders the voxel at `albedo × (128 + e/2) / 128` — from 1× up
to ~2× over-bright at 255 — and skips *everything* that would darken
it: the baked brightness byte, per-face side shades, the runtime rig,
shadows, cel bands. Only fog still applies, so a distant glow fades
like the rest of the world. Emissive composes with translucency: an
`AlphaBlend` crystal glows *through* its own body. Both backends
render it identically; the terrain material map above is all the
wiring it needs (the `MAT_CRYSTAL` lines in the snippet).

Making the voxel bright is half the effect — a glow that doesn't
light its surroundings reads as a sticker. The other half is the
**point-light bake** (voxlap's lightmode 2): register the glow
sources on the grid and bake with `BakeMode::PointLights`, and each
light writes a cube-law Lambertian pool into the surrounding
brightness bytes over a deliberately *dim* directional base — light
pools reading against gloom. Baked means free at render time, on both
backends:

```rust,noplayground
{{#include ../../../crates/roxlap-scene/src/cavegen.rs:bake_light}}
```

```rust,noplayground
{{#include ../../../crates/roxlap-cave-demo/src/main.rs:crystal_bake}}
```

`Grid::bake_bbox` — the incremental relight primitive for runtime
carves — picks the grid's `bake_lights` up automatically, so shooting
a hole next to a crystal re-bakes the crater *with* its glow pool
intact. A `BakeLight`'s `strength` is on the brightness-byte scale:
the gain at distance `d` is roughly `strength / d²`, so `2000` is a
reading-torch and `8000` floods a small cavern. For a light that
moves or flickers, use a runtime `PointLight` from the rig instead —
the bake is for scenery that stays put.

The cave demo is the live showcase: crystal clusters planted on
cavity walls, each one an emissive translucent blob plus a
`BakeLight` (`cargo run --release -p roxlap-cave-demo`).

## Further reading

- The **Lighting**, **Spotlight** and **Transparency** demo scenes —
  every effect in this chapter, live with toggles
  (`ROXLAP_SCENE=Lighting cargo run --release -p roxlap-scene-demo`).
- [`PORTING-DYNLIGHT.md`](https://github.com/NCrashed/roxlap/blob/master/docs/porting/PORTING-DYNLIGHT.md),
  [`PORTING-SPOTLIGHT.md`](https://github.com/NCrashed/roxlap/blob/master/docs/porting/PORTING-SPOTLIGHT.md),
  [`PORTING-TRANSPARENCY.md`](https://github.com/NCrashed/roxlap/blob/master/docs/porting/PORTING-TRANSPARENCY.md)
  — design history: why shadows are hard-edged, why spots fold into
  the point path, why transparency needed no OIT.
