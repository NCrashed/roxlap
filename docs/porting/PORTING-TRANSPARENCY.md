# roxlap — transparent voxels: per-voxel materials + front-to-back compositing (Substage TV)

Start-of-stage brief and locked decisions for **transparent voxels** —
alpha-blended and additive voxels for effects like spell sprites, smoke,
fire, magic auras, water, and glass. Companion to
[PORTING-DDA.md](PORTING-DDA.md) (the clean-room per-pixel 3D-DDA CPU
renderer this builds on — **landed**, DDA.0–DDA.10), [PORTING-GPU.md](PORTING-GPU.md)
(the compute-shader GPU raymarcher), [PORTING-VOXEL-CLIP.md](PORTING-VOXEL-CLIP.md)
(the `.rvc` animated-voxel-clip path), and [PORTING-SPRITE-API.md](PORTING-SPRITE-API.md)
(the dynamic sprite/instance API).

This is a **start-of-stage brief**. A fresh-context session should read
it top to bottom before touching code. The stage tag is **TV**. It
targets **0.16.0 / 0.17.0**.

## Why

Every voxel roxlap renders today is **opaque**. Both backends are now
honest per-pixel 3D-DDA raymarchers (the DDA stage replaced voxlap's
column-coherent opticast), and both do **first-opaque-hit with early
return**:

- CPU terrain: `dda.rs:917-924` — `sampler.hit(cell)` returns `Some`,
  the brick-DDA loop immediately `return`s a `Hit`. No second sample.
- CPU sprites: `dda_sprite.rs` `draw_sprite_dda` — same, first solid
  voxel of the model wins.
- GPU terrain: `scene_dda.wgsl:469-481` — first solid voxel writes
  `out.color`, `out.hit=true`, `return out`. Final store hardcodes
  `pack4x8unorm(vec4(best_color, 1.0))` (`scene_dda.wgsl:573`) — alpha
  is literally `1.0`.
- GPU sprites: `sprite_model_dda.wgsl:165-173,223-226` — first hit
  overwrites the pixel, opaque.

There is **zero** multi-voxel-per-pixel blending anywhere. We want a
first-class transparency layer so demiurg can author smoke/fire/spell
auras (additive glow), and glass/water (alpha-over), and preview them
**through the engine** (editor and runtime pixel-identical, the standing
roxlap discipline).

## Key enabling facts

- **Per-pixel front-to-back DDA gives us order-correct OIT for free.**
  This is the whole reason the feature is tractable now and was not
  under voxlap's opticast. Each ray visits cells in **strict
  front-to-back order** along its own path. So "accumulate color +
  transmittance until opaque or transmittance≈0" is correct **without
  any depth sorting, OIT scheme, or per-frame back-to-front pass** — the
  thing triangle rasterizers fight hardest. We replace one `return` with
  an accumulate-and-continue loop. (The old column-coherent opticast
  could not have done this cleanly; the DDA migration unlocked it.)

- **The high byte is NOT free.** Colors are voxlap-packed `0x80RRGGBB`
  where the top byte is **lightmode-1 directional brightness** (0x80 =
  neutral), consumed by `shade()` (CPU `dda.rs:179-183`; GPU
  `scene_dda.wgsl:266-267`) and overwritten by the lighting bake
  (`world_lighting.rs:728`). Confirmed across `palette.rs:22`,
  `kv6.rs:143-145`, `vxl.rs:124-127`, `voxel_clip.rs:108`. **Alpha must
  come from a separate channel, not the high byte.** Pure-black RGB
  (`rgb & 0x00ff_ffff == 0`) is already the air/empty sentinel
  (`grid_view.rs:394`), so it stays reserved.

- **Sprites/clips already carry parallel per-voxel arrays.** `SpriteModel`
  (`sprite_model.rs:32`) and `VoxelFrame` (`voxel_clip.rs:102`) store
  `colors` + `dirs` as arrays parallel to the occupancy popcount-rank
  (`color_offsets` prefix sums). A per-voxel **material id** array slots
  in **exactly** like `dirs` — same indexing, same upload path. This is
  the clean seam the design hangs on.

- **The compositing primitive already exists in 2D.** The line/image
  overlays use `wgpu::BlendState::ALPHA_BLENDING` (`lib.rs:2475,2806`),
  and `paint_egui` blends over the deferred frame with `LoadOp::Load`
  (`lib.rs:2899`). `apply_fog` is already a `mix()` toward a color
  (`dda.rs:189-203`, `scene_dda.wgsl:347-351`) — the exact shape of a
  transmittance blend, a ready code template.

- **Default-opaque is bit-identical.** A model/grid with no material
  data resolves to material id 0 = `Opaque`, which takes the existing
  first-hit-return path verbatim. So the entire opaque world stays
  byte-for-byte unchanged through every TV stage — the regression anchor.

## Locked decisions

Taken with the engine author 2026-06-27:

1. **Unified material model from day one** (not scalar-alpha-first). A
   per-voxel **material id** (`u8`) indexes a **global material palette**
   of `Material { alpha, mode, … }` — one 256-entry table the renderer
   owns, shared by every model and grid (refined from a per-model table
   during TV.0: a global palette is simpler, shareable — all water is one
   material — and uploads once; per-file tables for portable assets can
   come later). Covers mixed models (opaque bottle + translucent potion;
   window frame + glass) and is the future seam for emissive/tint/
   refraction. Material id 0 is permanently `Material::OPAQUE` and cannot
   be redefined (the back-compat default every material-free voxel
   resolves to). Landed in `roxlap-formats::material`
   (`BlendMode`/`Material`/`MaterialTable`); facade
   `SceneRenderer::define_material` / `material`.

2. **Both blend modes from day one.** `BlendMode { Opaque, AlphaBlend,
   Additive }` designed and shaded together. `Additive` (commutative,
   order-independent, no transmittance bookkeeping) is the simplest and
   highest-value for spells/fire/auras; `AlphaBlend` (front-to-back
   `over`) is glass/smoke/water. Both ride the same march.

3. **Scope includes terrain.** Transparent voxels live both in
   **sprites/clips** (the easy seam, lands first) **and** in **grid
   terrain** (Vxl chunks — glass walls, water bodies; the heavy tail,
   lands last). Terrain needs a new per-surface-voxel material byte
   buffer in the `.vxl` path (it has no spare field today).

4. **One shared compositing core.** A single
   `composite_layer(&mut accum, &mut trans, color, alpha, mode)` helper
   in `roxlap-core` drives **both** the terrain march (`dda.rs`) and the
   sprite march (`dda_sprite.rs`), so CPU stays internally consistent;
   the WGSL mirrors it line-for-line for parity.

5. **Per-instance alpha multiplier.** Instances carry an `alpha_mul: u8`
   (default 255) scaling the material alpha, so a spell can **fade out**
   or smoke **dissipate** by cheap per-frame instance updates — no
   volume re-upload (mirrors the clip-player cheap-transform discipline).

6. **Pragmatic cross-pass ordering for v1.** Full unified OIT across the
   *separate* terrain and sprite passes is out of scope (different accel
   structures). v1 composites **within** each pass front-to-back, writes
   the **opaque/background depth** to the z-buffer, and runs the sprite
   pass over the framebuffer. Known v1 limitation: a transparent sprite
   *between* glass and the wall behind it composites over the
   already-baked glass instead of under it. Documented, accepted.

7. **No new crate.** `Material`/`BlendMode` + the material arrays land in
   `roxlap-formats`; compositing core + both marches in `roxlap-core`;
   GPU buffers/shaders in `roxlap-gpu`; the authoring API on the
   `roxlap-render` facade (mirrored by hand into `cpu.rs` + `gpu.rs` —
   the facade is duck-typed, no backend trait).

## The data model — `Material` + per-voxel ids

```rust
// roxlap-formats/src/material.rs  (new)

#[repr(u8)]
pub enum BlendMode {
    Opaque = 0,      // existing first-hit path; ignores alpha
    AlphaBlend = 1,  // front-to-back `over`; uses alpha as opacity
    Additive = 2,    // commutative glow; uses alpha as intensity scale
}

pub struct Material {
    pub alpha: u8,       // 0..=255 (255 = fully opaque / full intensity)
    pub mode: BlendMode,
    // reserved for later TV / 0.x: tint: u32, emissive: u8, ...
}
```

- **Per-model / per-grid material table:** `Vec<Material>` (≤256 entries,
  so an id fits a `u8`). Index 0 is forced to `Material { alpha: 255,
  mode: Opaque }`.
- **Per-voxel material id:** a `Vec<u8>` **parallel to `colors`/`dirs`**
  (same popcount-rank indexing via `color_offsets`). Added to:
  - `SpriteModel` (`sprite_model.rs:32`) — `materials: Vec<u8>`.
  - `VoxelFrame` (`voxel_clip.rs:102`) / `DecodedClip` (`:345`,
    alongside the parallel `dirs`).
  - the `Vxl` terrain slab path (a parallel per-surface-voxel byte
    buffer mirroring the color word's indexing — see TV.4).
- **Empty / default:** no material array ⇒ every voxel id = 0 ⇒ `Opaque`
  ⇒ existing path, bit-identical.

### Compositing math (front-to-back, premultiplied)

The shared core, applied as the ray visits each transparent voxel in
front-to-back order. `accum` is premultiplied RGB, `trans` is remaining
transmittance (starts 1.0):

```
let a = material.alpha * instance.alpha_mul   // both normalized to 0..1
let c = shade(color, face)                     // existing brightness + side-shade
match mode {
    AlphaBlend => { accum += trans * a * c; trans *= (1 - a); }
    Additive   => { accum += trans * a * c; /* glow does not occlude */ }
}
if trans < 1/256 { stop }                       // transmittance early-out
```

On the **opaque/background hit** (or sky tail): `accum += trans * bg`,
write `accum` as the final pixel color and the **opaque hit's depth**
(not the first transparent hit) to the z-buffer; stop. Fog is applied to
the composite by the background depth in v1 (per-layer fog is a TV.3
polish item). Cap transparent layers per ray (e.g. 128) and `log()` on
clamp — no silent truncation (the standing "no silent caps" rule).

### Per-span compositing (landed TV.1/TV.2)

The accumulate above is gated **per solid span**: a translucent voxel
composites one alpha layer only when the ray *enters* a contiguous solid
run (the previous cell was air); the interior of the run is skipped
(opaque voxels still stop the ray on every cell, so a mixed run's opaque
core is never skipped). Without this, a ray clipping the shared boundary
between two adjacent surface voxels passes through both and double-
composites a thin strip — the model reads as **diced by a voxel grid**
(`solid_cube`/`solid_box` are surface-only hollow shells, so this is very
visible on a translucent box face). Per-span makes a wall contribute
exactly one alpha regardless of how many of its voxels the ray grazes;
**thickness no longer affects opacity** (a surface model, ideal for
glass/shells). Implemented identically in `cast_local_layers` (CPU) and
`march_instance_layers` (GPU); pinned by `per_span_thickness_independent`.

**`BlendMode::Volumetric`** (Beer–Lambert) is the thickness-aware mode for true
smoke/fog on *filled* volumes: it weights each voxel's opacity by the ray's
segment length through it — per-cell effective opacity `1 - (1-a)^seg_len`
(`seg_len` = traversed length in voxel units), so a boundary sliver contributes
≈0 (no grid) while opacity grows smoothly with depth. Unlike `AlphaBlend` it is
**per-cell, not per-span** (every traversed cell accumulates), and it occludes.
**Landed** (post-stage follow-up) on both backends + both passes — CPU
`cast_local_layers`/terrain `cell_walk_skip` compute `seg_len` from the cell's
`t` span × ray length (÷ voxel size for terrain); GPU `march_instance_layers` /
`march_grid` mirror it. Pinned by `volumetric_thickness_deepens_opacity`
(sprite) + `terrain_volumetric_thickness_deepens_opacity`. Per-span `AlphaBlend`
stays the default for shell-based effects (glass/water surfaces).

**Interior retention (kv6).** Volumetric needs the volume's *interior* voxels —
but kv6 surface extraction (`Kv6::from_fn`, and authored `.kv6` assets) culls
every enclosed voxel, so a "filled" model is really a hollow shell and a
Volumetric ray would graze only its front + back faces (no depth accumulation).
Fixed for the authoring path by `Kv6::from_fn_keep_interior(.., keep_interior:
Fn(u32)->bool)`: it retains an enclosed voxel when its colour matches the
predicate, so translucent/volumetric bodies stay solid through while opaque
interiors are still dropped (the storage win). Terrain `.vxl` is unaffected — it
stores solid *runs*, so interior cells are already traversable; the gap was
kv6-only (sprites/clips). The demo fog cloud uses this; authored volumetric
`.kv6`/`.rvc` assets must likewise be exported with their translucent interiors
intact (a future `.kv6`/`.vxl` exporter concern, not a runtime one).

## Renderer changes

**CPU.** Replace the first-hit `return` at `dda.rs:917-924` (terrain) and
the analogous return in `dda_sprite.rs` with the accumulate-and-continue
core (decision #4). The hit path already has `shade()` (`dda.rs:179-183`)
and `apply_fog()` (`:189-203`); reuse them per layer / at finalize. Depth
write at the put sites (`dda.rs:1117-1120` / `:122-125`) moves to the
opaque/background hit only. `surface_color_mip` (`grid_view.rs:360-364`)
gains a parallel material lookup for terrain (TV.4).

**GPU.** Mirror in `scene_dda.wgsl` (terrain, the `:469-481` return →
accumulate; final pack `:573` emits the composite; depth `:574-577` from
the opaque hit) and `sprite_model_dda.wgsl` (`:165-173,223-226`). New
bind-group entries: a per-model/grid material table buffer + the
per-voxel material-id buffer (parallel to the existing colors/dirs
buffers; same upload as `dirs`). Per-instance `alpha_mul` rides the
existing instance record (`SpriteInstance`, `sprite_model.rs:669`).

**Parity.** CPU is pinned first, GPU matched (the standing discipline).
Watch integer-vs-float rounding in the `a * c` blend — pick one canonical
rounding and match both backends (cf. the historic `_mm_rcp_ps`
sub-pixel divergence pain). `Additive` is the easy parity (commutative,
no `trans`); pin it first.

## Authoring API (facade)

The facade (`roxlap-render/src/lib.rs:963`) is duck-typed — each new
method is added in `lib.rs` and mirrored by hand in `cpu.rs` + `gpu.rs`.

- `add_sprite_model` (`lib.rs:1564`) / `add_voxel_clip` (`lib.rs:1666`)
  gain an optional `materials: &[Material]` + per-voxel ids (default:
  none ⇒ all-opaque). Mirror in `cpu.rs:549/591`, `gpu.rs:346/416`.
- A posed-add / instance path carries `alpha_mul` (extend
  `DynSpriteTransform`, `lib.rs:531`, or a posed-add variant), drivable
  per frame via `set_sprite_instance_*` for fade animation.
- Terrain editing (`set_voxel`/`set_rect`/`set_sphere` in `roxlap-scene`
  /`roxlap-formats::edit`) gains an optional material id (TV.4).

## Sub-substage roadmap

| Stage | Scope | Gate |
|---|---|---|
| **TV.0** ✅ | `BlendMode`/`Material`/`MaterialTable` (global 256-entry palette, id 0 locked `Opaque`) in `roxlap-formats::material`; held authoritatively on both backends; facade `define_material`/`material`. Per-voxel `materials` arrays + GPU upload deferred to TV.1/TV.2 (land with the march that consumes them, keeping each stage symmetric). No render change. | **Done** — builds + clippy clean; formats unit tests; opaque path untouched (table inert until a translucent material is defined). |
| **TV.1** ✅ | **CPU** sprite/clip transparency: `dda_sprite.rs` accumulate-and-continue march (`cast_local_layers` + `draw_sprite_dense_shaded`/`draw_sprite_dda_shaded`/`draw_frame_shaded`); `Additive` + `AlphaBlend`; `SpriteShade {materials, material, alpha_mul}` ctx; per-sprite opaque gate keeps the opaque path byte-identical; `SpriteDense.mat` per-voxel array (empty in TV.1, used in TV.3). Plumbed: `Sprite.material`/`alpha_mul`, CPU draw sites, facade `set_sprite_instance_material`/`set_sprite_instance_alpha` (GPU retains on `sprite_basis` for TV.2). | **Done** — 5 march unit tests (additive/alpha/alpha_mul/opaque-bit-identical/terrain-occlusion); workspace clippy+test green. Visual goldens deferred to the TV.7 demo. |
| **TV.2** ✅ | **GPU** sprite/clip transparency: `sprite_model_dda.wgsl` gains a material palette (binding 12) + per-instance `material`/`alpha_mul` on the `Instance` (std430 64→80); `has_translucent` uniform gates a two-sweep path (nearest-opaque → background, then `march_instance_layers` translucent-over in tile order) — opaque path byte-identical when the flag is off. `GpuRenderer::set_sprite_materials` uploads the palette; the per-instance setters mark dirty + re-upload. | **Done (code)** — builds + clippy clean; **naga WGSL validation passes** + real GPU device tests green. **Visual correctness on a GPU is user-verified** (no display in CI). Headless wgpu *does* run here (revises the old "no headless wgpu" note) — a full sprite-pass pixel test is a good follow-up. |
| **TV.3** ✅ | Per-voxel **mixed-material** models on both backends (opaque frame + glass). Authoring = colour→material map (`add_sprite_model_with_materials`); `material_for_color`; CPU `from_kv6_with_materials` + per-span material-change in `cast_local_layers`; GPU per-voxel `materials_vox` buffer (binding 13) + `ModelMeta.has_vox_materials` + two-sweep handles mixed (own opaque voxels back its layers); demo window. **Deferred** (would change sprite shading semantics): `side_shades`/lighting + per-layer fog + tint on translucent — sprites stay flat-lit/unfogged as before. | **Done** — CPU mixed tests (classify + homogeneous==uniform); GPU naga + device tests; workspace clippy + 23 binaries green. GPU visual = user-verified. |
| **TV.4+5** ✅ | **CPU terrain transparency via colour→material LUT** (decision: LUT over Vxl per-voxel storage — zero format/edit/mip/serde change, consistent with sprites). `DdaEnv.materials` + `terrain_materials`; `cell_walk_skip` accumulates front-to-back (per-hit `terrain_material` lookup, per-span by run-entry/material-change, opaque/sky/fog finalize); **empty map ⇒ first-hit verbatim, bit-identical**. Facade `set_terrain_materials`; CPU backend + `render_scene_composed_with_materials`. | **Done** — `terrain_glass_tints_floor_behind` test; all DDA goldens unchanged; workspace clippy + 23 binaries green. |
| **TV.6** ✅ | **GPU** terrain transparency: `scene_dda.wgsl` `march_grid` accumulates front-to-back (gated on `terrain_has_translucent`; off ⇒ byte-identical first-hit), `voxel_packed_in` + `terrain_material_id` colour→material lookup, opaque/sky finalize, per-span. Material-palette (binding 16) + terrain colour→material map (binding 17, fixed 256 rows) buffers on `SceneDdaResources` + the headless renderer; `GpuRenderer::set_scene_terrain_materials`; render-layer pushes each frame. Demo grid (glass wall + opaque wall) for both backends. | **Done (code)** — builds + clippy clean; naga + GPU device tests pass (pipeline/bind-group validated on a real device). **Visual GPU = user-verified.** |
| **TV.7** ✅ | Demo "Transparency" scene (glass pane, additive glow, pulsing smoke, mixed opaque-frame+glass window, world glass wall); README transparency feature; CHANGELOG `[0.16.0]`; workspace + internal-dep version bump 0.15→0.16. | **Done** — build + clippy clean, 23 binaries green, Cargo.lock at 0.16.0. Tag/publish left to the maintainer. |

**Stage TV complete.** All sub-substages landed; CPU fully tested, GPU device-tested + visually verified by the author on both backends.

**Post-stage follow-up (clip per-voxel materials).** TV.3 landed per-voxel mixed materials for static sprites (`add_sprite_model_with_materials`) but left voxel clips (`.rvc`) at the whole-instance uniform material only — `sprite_model_from_voxel_frame` / `SpriteDense::from_voxel_frame` hard-coded an empty per-voxel `materials` array. Closed by `add_voxel_clip_with_materials` (facade + CPU + GPU): a colour→material map classifies every decoded clip frame's voxels, the clip analogue of `add_sprite_model_with_materials`. Built on the existing per-voxel composite path (CPU `cast_local_layers` gate on a non-empty `mat`; GPU `has_vox_materials`), so no march change. An empty map stays byte-identical to `add_voxel_clip`. Dogfooded by a pulsing glass-orb clip in the Transparency demo.

The same colour→material map is carried through the two edit/stream paths that rebuild a clip frame's geometry, so neither drops per-voxel materials: (1) **in-place single-frame edit** — `update_clip_frame` re-classifies the edited frame from the map retained in `ClipMeta`; (2) **streaming clips** — `add_streaming_clip_with_materials` stores the map in `StreamingClipState` and `set_streaming_clip_frame` re-applies it via the new `refresh_sprite_model_with_materials` on every per-frame model re-upload. Backends gained `update_clip_frame(..., material_map)` and `update_sprite_model_with_materials` (using `sprite_model_from_voxel_frame_with_materials` / `build_sprite_model_with_materials`).

Deferred (future): per-file material tables for portable `.rvc`/`.kv6` assets; unified cross-pass/cross-grid OIT (v1 composites within each pass/grid). (The Beer–Lambert `Volumetric` BlendMode, previously deferred here, has landed — see the per-span section above.)

## Risks

- **R1 — cross-pass ordering (decision #6).** Terrain and sprites are
  separate passes; v1 cannot perfectly order a transparent sprite that
  sits *inside* transparent terrain. Mitigation: composite within each
  pass, opaque-depth write, document. A unified per-ray terrain+sprite
  march is the "perfect OIT" future, explicitly out of TV scope.

- **R2 — CPU/GPU blend rounding divergence.** `AlphaBlend` does
  `trans*a*c` accumulation; integer vs float rounding can drift sub-pixel
  across backends (history: `_mm_rcp_ps` divergence, oracle bit-gate
  abandonment). Mitigation: pin one canonical rounding; land `Additive`
  (exact) first; treat parity as image-regression with tolerance, not a
  bit gate.

- **R3 — perf on heavy translucency.** A ray grazing a large smoke/water
  volume visits many voxels with no early-out until `trans` decays.
  Mitigation: transmittance early-out (`trans < 1/256`), per-ray layer
  cap with `log()`, preserved brick empty-skip. Snapshot perf vs the
  `project_s4b_baselines`-style baseline before/after TV.5.

- **R4 — terrain slab storage (TV.4).** The `.vxl` RLE slab format has no
  spare per-voxel field; a parallel material buffer must mirror a
  non-trivial surface-voxel indexing and survive edits, mips, and serde.
  This is the heaviest sub-stage and may itself split; it is
  deferrable behind TV.0–TV.3 (effects ship first) if scope pressure
  hits — sprites/clips already cover smoke/spells/auras and
  glass/water-as-models.

- **R5 — depth semantics for downstream passes.** Writing opaque/sky
  depth (not first-transparent) is required so sprites and picking
  (`world_query`) occlude against solid geometry. Verify picking +
  any depth consumers against transparent terrain.

## Forward-compat: dynamic lighting

TV is deliberately **orthogonal to whatever lighting model comes next**
(static bake stays, or a future dynamic-light stage replaces it). Two
design choices guarantee this:

1. **Alpha/material lives in its own per-voxel channel, never in the
   high byte.** The `0x80RRGGBB` high byte today is a *cache of the
   static lightmode-1 bake* (`world_lighting.rs:728` writes it,
   `shade()` `dda.rs:179-183` consumes it) — it is **not** a fundamental
   channel. Dynamic lighting does not need it either: per-pixel light is
   computed from **albedo** (low 24 bits) + a **surface normal** (CPU:
   central-difference gradient at the DDA hit; sprites: the `dirs` LUT
   index, both already separate from the high byte) + scene light data
   (new runtime state). So the high byte's fate — kept as a baked
   ambient/GI term (hybrid lighting), or freed and pinned to neutral
   `0x80` (pure dynamic) — is a **lighting-stage decision** that TV does
   not constrain and that does not touch the transparency data model.

2. **`shade()` is the seam; transparency sits in front of it.** The
   compositing core does `accum += trans * a * shade(color, face)` and
   treats `shade()` as a black box. *How* `shade()` derives light
   (reading the baked byte vs. evaluating dynamic lamps against the
   normal) lives entirely behind that call. A future lighting stage
   swaps the body of `shade()` / the bake without revisiting any TV
   compositing code.

Consequence and convergence: the **`material-id → Material` table is the
natural future home for the PBR material descriptor.** `Material` carries
`{alpha, mode}` now and grows `{roughness, metalness, emissive, tint,
…}` when dynamic lighting lands — the same per-voxel channel that serves
transparency becomes the material input to the lighting model. The high
byte carries none of it. On-disk: even a freed high byte keeps writing
neutral `0x80`, so `.vxl`/`.kv6` stay format-compatible with no version
bump (the `rgb & 0x00ff_ffff == 0` empty sentinel is independent of the
high byte and is unaffected either way).

## Validation (every sub-substage)

- `cargo test` across the workspace stays green; the **opaque-unchanged**
  golden is the headline regression gate at every stage (default material
  id 0 ⇒ existing path).
- New transparency visual goldens frozen per stage (image hash), CPU
  pinned then GPU matched within tolerance.
- Per-mode acceptance repros: an **additive** glow over terrain
  (occluded correctly), an **alpha-over** glass pane (background tints
  through), a **mixed-material** model (opaque + translucent in one
  volume), and (TV.5+) a **terrain glass wall + water pool**.
- Perf tracked from TV.5 against a pre-stage baseline snapshot.
- GPU + interactive paths are dogfooded in the TV.7 demo scene (note:
  headless CI has no display — GPU visual checks are manual, per the
  standing demo-scene caveat).
