# roxlap — emissive voxels + glowing cave crystals (Stage EV)

Entry doc written 2026-07-07 at workspace 0.24.0, right after the
engine-state audit. This is the **entry doc** for the emissive-voxel
stage — tag **EV**. A fresh-context session should read it top to
bottom before touching code.

## Status — EV.0..5 ALL LANDED 2026-07-07; owed items CLOSED 2026-07-12

The visual eyeball pass on the crystals passed (2026-07-07). The two
carried items landed 2026-07-12:

- **Sprite emissive LANDED** — both sprite paths now branch on the
  material's emissive, mirroring the terrain hit order (emissive
  outranks the dynamic rig and the baked shade; the per-instance tint
  still applies; per-voxel TV.3 material ids honoured). CPU:
  `dda_sprite.rs` opaque first-hit + `shade_layer` (translucent
  layers). GPU: `sprite_model_dda.wgsl` `march_instance` (the palette
  fetch is gated per hit on a new `has_emissive` uniform, repurposed
  pad — an emissive-free palette never touches the palette in the
  opaque marcher) + `march_instance_layers` (`mm` already fetched).
  Host: `material_palette` also returns `any_emissive`;
  `set_sprite_materials` stamps the gate; the facade's existing
  `materials_dirty` sync carries it — no facade change. Tests:
  `sprite_emissive_ignores_lighting` (value == `emissive_shade`
  exactly — CPU sprite/terrain parity by construction),
  `sprite_emissive_glows_through_alpha_blend`.
- **Headless GPU emissive test LANDED** —
  `HeadlessSceneRenderer::set_terrain_materials` (mirror of the
  surface path's `set_scene_terrain_materials`, same gate logic)
  replaces the hardcoded opaque dummies; `terrain_has_translucent` /
  `terrain_map_count` now come from the plumbed state. Gate test
  `scene_dda_emissive_ignores_lighting` (scene_render.rs): the GPU
  emissive branch matches the CPU ladder **exactly** ((255,255,0) for
  0xff8000 @ e=255), ignores a dim baked byte and a zero-ambient rig,
  and an empty map re-renders byte-identically to the pre-material
  baseline.

Remaining accepted simplification:

- Carving *through* a crystal keeps its light (documented in the demo).

- EV.0 — LANDED 2026-07-07: `Material.emissive: u8` +
  `Material::glow` / `with_emissive` + `MaterialTable::any_emissive`;
  all constructors stay non-emissive; unit tests. Source-breaking
  (struct literal init) — no in-repo literals existed outside
  `material.rs`.
- EV.1 — LANDED 2026-07-07: CPU hit path hoists the per-hit material
  lookup above shading (`dda.rs`); `emissive_shade()` =
  `(c · (128 + (e >> 1))) >> 7` per channel; branch outranks the
  dynamic rig and the baked byte; fog unchanged. Unit test
  `terrain_emissive_ignores_lighting` (baked byte / side shades / rig
  invariance); all 121 core tests green (goldens intact).
- EV.2 — LANDED 2026-07-07: `MaterialGpu` grew
  `emissive: f32` + pad (16-byte stride, BOTH palettes — scene
  binding 16 + sprite binding 12 structs updated in lockstep);
  `material_palette()` pre-scales the factor host-side; scene gate
  `scene_terrain_translucent` now ORs emissive mappings; scene WGSL
  hit site mirrors the CPU order (material first, emissive branch,
  compositing reuses the fetched `mm`). Sprite pass carries the field
  but does NOT render emissive (terrain-only, decision 7). **Owed:**
  a headless GPU emissive test once the headless harness grows
  material plumbing (today it hardcodes `terrain_has_translucent: 0`);
  until then parity rides on the demo visual pass (EV.4).
- EV.3 — LANDED 2026-07-07: `BakeMode::PointLights` (lightmode 2) +
  `BakeLight {pos, radius, strength}` + public `Grid::bake_lights`
  field (chose a public field over a setter, matching `chunks` /
  `stream_radius` house style); per-chunk rebase + sphere-vs-AABB cull
  in `chunk_bake_lights`; both `bake` and `bake_bbox` consume it.
  Tests: pool brightens floor / dim base vs Directional;
  bbox-vs-full byte-identity WITH lights. Not serialized in
  snapshots (documented on the field).
- EV.4 — LANDED 2026-07-07: `plant_crystals` (deterministic xorshift
  rejection sampling: air voxel → march ≤14 voxels along a random
  axis to a wall → crystal blob + `BakeLight` floating 3 voxels off
  the surface; ≥24-voxel spacing; one guaranteed spawn-bubble
  crystal); `LIGHTMODE` 1→2; `CarveJob` carries the lights so the
  worker's `relight_bbox` keeps glow pools; crystal material =
  `alpha_blend(180).with_emissive(255)`, colour-keyed per preset
  (cyan/amber). Test `crystals_planted_and_lit` (both presets).
- EV.5 — LANDED 2026-07-07: Lighting-chapter section "Emissive voxels
  & baked glow" (anchored snippets: `book_lighting:materials` grew
  the crystal; cave-demo `bake_light` + `crystal_bake` anchors);
  CHANGELOG under [Unreleased]; check-anchors green.

## Goal

Two user-visible things:

1. **Emissive voxels** — a voxel whose material glows: it renders at
   full (or over-bright) albedo intensity regardless of the baked
   brightness byte, per-face side shades, the dynamic light rig, and
   shadows. Orthogonal to blend mode, so a translucent crystal can
   glow through `AlphaBlend` compositing.
2. **Glowing crystals in `roxlap-cave-demo`** — the generator plants
   crystal clusters in cavities; the crystals are emissive *and* cast
   baked point light onto the surrounding cave (voxlap lightmode-2
   point-light bake), surviving incremental relight after carves.

## Locked design decisions

1. **`Material` grows `emissive: u8`** (`0` = none, `255` = max).
   Orthogonal to `mode`/`alpha` — an `Opaque` material can glow, an
   `AlphaBlend` one can glow through its translucency. The material
   palette is renderer-owned and never serialized (checked: no
   `Material` on the wire in `.rvc`/`.rkc`/snapshots), so this is a
   **source-level breaking change only** → next cut is a minor
   (0.25.0), consistent with house policy. Constructor sites updated
   in-repo; `Material::OPAQUE` keeps `emissive: 0`.
2. **The bake stays emissive-ignorant.** The render hit path computes
   an emissive voxel's colour purely as
   `albedo × ((128 + (emissive >> 1)) / 128)` — i.e. 1.0× at
   `emissive = 0`…`1` up to ~2.0× at 255, matching the existing
   `byte/128` brightness convention (`world_lighting.rs`: neutral
   `0x80`). The baked byte of an emissive voxel is simply **unused**,
   so `update_lighting`/`apply_lighting_with_cache` signatures and
   golden bytes stay untouched. No format or bake versioning.
3. **Emissive skips ALL shading**: no per-face `side_shade_sub` /
   `face_shade`, no N·L, no shadow test, no AO fill, no cel bands.
   Fog **still applies** (a distant glow fades like everything else;
   keeps CPU/GPU parity trivial — both apply fog after the shade).
4. **Byte-exactness gate**: with no emissive material defined (and for
   every colour not in the terrain material map) both backends take
   bit-identical existing paths. The CPU opaque fast path is gated on
   the material lookup that already happens per hit (PF.7/C4 — one
   colour→material scan, `dda.rs:1657`); the branch only reorders
   *when* the lookup result is consulted, not whether it happens.
   `MaterialTable::all_opaque()` gates the translucency fast path
   today; emissive needs its own "any emissive defined" check so an
   Opaque+emissive palette doesn't silently stay on the pre-material
   path (audit every `all_opaque()` call site).
5. **Light onto surroundings = voxlap lightmode 2**, not the dynamic
   rig. `compute_brightness` (`world_lighting.rs:979`) already
   implements the point-light bake: dim directional base
   (`×16 + 47.5`) + per-light cube-law falloff with hard radius
   cutoff. New `BakeMode::PointLights` maps to lightmode 2;
   `Grid::set_bake_lights(&[BakeLight])` stores grid-local lights on
   the grid so `bake()` **and** `bake_bbox()` (the carve-relight
   primitive) pick them up without threading lights through every
   call site. `BakeLight { pos: Vec3 (grid-local voxel coords),
   radius: f32, strength: f32 }` → per-chunk `LightSrc` translation +
   chunk-AABB×radius culling inside the bake.
6. **Dynamic `LightRig` is untouched** — crystals light the cave via
   the bake; a host that wants flicker can additionally register a
   dynamic `PointLight` (demo won't, to keep the carve worker
   simple). The dim lightmode-2 base *replaces* the demo's current
   `BakeMode::Directional` — the cave gets darker overall, which is
   the look we want (glow pools in gloom).
7. **Sprite/clip paths inherit for free where the same `Material`
   flows** (`dda_sprite.rs` materials, GPU sprite palette binding 12
   — `MaterialGpu` mirrors there too). Verify while implementing but
   don't scope-creep: terrain crystals are the deliverable; sprite
   emissive is accepted if it falls out, deferred if it fights back.

## Substages

- **EV.0 — `Material.emissive` (roxlap-formats).** Field + `Default`
  + `Material::glow(e)` / `with_emissive(e)` helpers + on-wire
  no-change assertion in docs; `MaterialTable::any_emissive()`.
  In-repo struct-literal construction sites updated. Unit tests.
- **EV.1 — CPU render path (roxlap-core, roxlap-render).** Reorder
  the per-hit material lookup ahead of shading in `dda.rs`
  (~line 1610–1663); `emissive_shade()`; branch before
  `shade_lit_cpu`/`shade`; translucent compositing consumes the
  emissive `lit` unchanged. Facade: `define_material` invalidation
  already marks `materials_dirty`. Tests: emissive voxel colour
  independent of sun direction/side shades/rig; over-bright clamp;
  empty-map render byte-identical (existing golden harness).
- **EV.2 — GPU parity (roxlap-gpu).** `MaterialGpu` gains
  `emissive: f32` (+pad → 16-byte stride) in BOTH palettes (scene
  binding 16, sprite binding 12); WGSL `Mat` struct + emissive branch
  in the scene shader hit path (skip `face_shade`/`shade_lit`) and the
  compositing loop; sprite shader same treatment if the material
  plumb is already there (decision 7). Headless CPU-vs-GPU diff
  harness run (`HeadlessSceneRenderer`).
- **EV.3 — bake lights (roxlap-scene, roxlap-core untouched).**
  `BakeMode::PointLights` (lightmode 2), `BakeLight`,
  `Grid::set_bake_lights`/`bake_lights()`; per-chunk translation +
  culling in `bake_u32`/`bake_bbox_u32`. Tests: byte brightens near a
  light, dark base far away; `bake_bbox` over a carve near a light
  matches full re-bake bytes in the touched region; no lights ⇒
  lightmode-2 base only.
- **EV.4 — cave-demo crystals (roxlap-cave-demo).** Generator plants
  crystal clusters on cavity walls (colour-coded per preset — e.g.
  cyan for Blue, magenta for Mag); demo defines
  `AlphaBlend+emissive` material + terrain colour→material map;
  registers one `BakeLight` per cluster; switches bake to
  `PointLights`; the background carve worker's `relight_bbox` passes
  the crystal lights (it calls `roxlap_core::update_lighting` on a
  cloned chunk — give it the chunk-local light list). Carving through
  a crystal: keep the light (accepted simplification; note in demo
  README). Runs on both backends.
- **EV.5 — docs.** Lighting-chapter section (emissive materials +
  bake lights + crystal recipe), CHANGELOG entry, this doc's status
  updated, memory note.

## Hazards

1. **`all_opaque()` fast-path gates** — an emissive-only palette must
   leave the fast path or crystals silently render unlit. Grep every
   call site (facade `materials_dirty` consumers, GPU
   `any_translucent` flags, CPU `env.materials` gating).
2. **WGSL struct stride** — `Mat` today is 8 bytes
   (`alpha: f32, mode: u32`); growing it changes the buffer stride for
   bindings 12 + 16. Host-side `MaterialGpu` (`#[repr(C)]`) and WGSL
   must move in lockstep; stale-stride symptoms are garbage
   alpha/mode on every material.
3. **Chunk-local light coords** — the per-chunk bake runs in
   chunk-local space (`apply_lighting_with_cache` with 0-offsets);
   a grid-local `BakeLight` must be rebased per chunk (subtract
   `chunk_idx * CHUNK_SIZE`), and `bake_bbox`'s multi-chunk loop
   rebases per chunk too. An off-by-one-chunk light shows as a bright
   pool displaced by 128 voxels.
4. **Carve worker clone** — the demo's background carve relights a
   *cloned* `Vxl` via raw `roxlap_core::update_lighting`; it bypasses
   `Grid::bake_bbox` and therefore the grid-stored lights. The demo
   must pass its own chunk-local light list there (EV.4), or carve
   flashes will erase crystal glow pools.
5. **Demo brightness regression** — switching the cave from
   `Directional` (×64 + 103.5) to lightmode 2 (×16 + 47.5) darkens
   everything. Deliberate, but tune crystal density/strength so the
   demo stays navigable; keep the muzzle-flash/carve colours legible.
