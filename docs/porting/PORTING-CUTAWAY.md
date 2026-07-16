# roxlap — cutaway deck rendering (Stage CA)

Entry doc written 2026-07-16 at workspace 0.29.0.
This is the **entry doc** for the cutaway-rendering stage — tag **CA**.

## Status — CA.0..CA.6 ALL LANDED 2026-07-16 (visual pass by user pending)

- **CA.0** — `Grid::z_clip` + `Scene::set_grid_z_clip`, snapshot wire
  v4 (frozen v3 shadow shape, v1–v3 fixtures green), `DdaEnv.z_clip`,
  GPU `_pad3` lane → `z_clip_bits` (bitcast i32, `i32::MIN` =
  disabled) + `GridWorldTransform.z_clip`. Full suite byte-identical.
- **CA.1** — gate in `Sampler::hit` on the grid-local ABSOLUTE mip-cell
  z (`c[2] < z_clip >> mip`, arithmetic shift). 5 tests: 3-deck golden,
  run-top cut-face colour, stacked-chz absolute-z, mip floor-formula
  pin, disabled byte-identity.
- **CA.2** — single-grid shadows came free via the shared `Sampler`;
  cross-grid = per-grid AABB-floor clamp in `SceneOccluder::build`
  (fully hidden grid drops out entirely). Tests: hidden-roof
  byte-equals-no-roof (render level), per-grid clip independence
  (occluder level).
- **CA.3** — WGSL clip in `march_grid` + `shadow_occluded` (same
  `z_clip >> mip`); 3 headless gates incl. exact cut-face colour, the
  bedrock-fallback colour pin, and a floor-vs-ceil mip discriminator
  (two plates + odd plane).
- **CA.4** — footprint rule: `Grid::cutaway_hides_point` /
  `Scene::cutaway_hides_point` (single source of truth), applied in the
  CPU sprite pass (draw + occluder) and the GPU cull
  (`SpriteCutawayClip`, part of the PF.10 cull key; hidden sprites also
  stop casting — the GPU sprite-shadow pass marches the culled visible
  set). `Scene::raycast_clipped` for click-to-select; GPU depth picks
  clip-aware for free.
- **CA.5** — "Decks" demo tab (3-deck shiplet, decks span the chz
  boundary at z=256; flooded bilge per hazard 6; per-deck actors +
  cabin lights with the `cutaway_hides_point` light cull;
  `ROXLAP_DECK` capture hook) + `CameraRig::tele_iso` +
  `opticast_settings_fov`. NEW facade API
  `SceneRenderer::set_gpu_mip_scan_dist` — the tele distance otherwise
  puts the GPU at deep mips (found in the CA.5 visual check; CPU
  unaffected, its LOD is per-grid). Self-checked on both backends via
  ROXLAP_CAPTURE; the formal user visual pass is still owed.
- **CA.6** — book Rendering chapter "Cutaway deck views" + demo-tour
  row + CHANGELOG (next cut minor: additive, plus
  `GridWorldTransform` grew a public field).

Hazard 4 (occupancy not clip-aware — wasted steps through hidden
decks) accepted for v1, not measured as a blocker in CA.5.

## Goal

Per-grid horizontal clip plane for isometric "deck view" rendering (SS13 /
ship-interior style): all voxels of a grid **above** a chosen z (z-down, so
`z < z_clip`) are treated as air by the renderer, exposing the interior from
a high tele-perspective camera. The cut cross-section renders as a lit top
face. Both backends, hash-gated, zero cost / byte-identical when disabled.

Non-goal: this is a *render* feature only. Simulation, collision, audio
occlusion and networking-facing state are untouched — a hidden deck still
blocks sound and pathing.

## Locked design decisions

1. **Per-grid, grid-local, absolute voxel z.** `Grid::z_clip: Option<i32>`
   in grid-local absolute voxel z (same space as `voxel_bounds()`, spanning
   stacked chz chunks). Semantics: hide `z < z_clip`; `z_clip` itself is the
   first visible layer, its top face is the cut surface. No world-space
   plane — decks are grid-aligned and the clip must stay glued to a rotated
   ship (S5). Precedent for per-grid render config: `Grid::mip_levels_override`
   (`crates/roxlap-scene/src/lib.rs:448`, snapshot handling QE.5b).
2. **Clip means "world as if removed".** Applies to primary rays, sun/world
   shadow rays, dynamic-light occlusion marches, sprite instances (footprint
   rule, see CA.4) and picking. Cross-grid shadow rays apply the *target*
   grid's own clip. It does NOT apply to roxlap-audio — the game's stealth
   layer must keep hearing through real geometry.
3. **Cut-face colour = existing run-top fallback.** Voxlap RLE stores no
   interior colours; `GridView::surface_color_mip`
   (`crates/roxlap-core/src/grid_view.rs:357`) already falls back to the
   run's top-surface colour for interior hits. The cut face reuses that plus
   the normal top-face `side_shade`. No new material, no dedicated cut tint
   in v1 (deferred).
4. **Lights are game-managed.** The engine does not filter point lights above
   the clip; a lamp on a hidden deck will light the exposed interior. The
   game/demo culls lights by z when setting the clip (pattern shown in CA.5).
5. **Projection stays perspective.** Isometric look = high camera + narrow
   FOV ("tele-iso") preset; no orthographic ray generation this stage. Both
   backends already take arbitrary camera bases, so this is demo-side only.
6. **Disabled ⇒ byte-identical.** `z_clip = None` must reproduce every
   existing golden hash on both backends. This is the CA.0 gate and stays a
   standing invariant for every substage.

## Substages

- **CA.0 — plumbing (no-op).** `Grid::z_clip: Option<i32>` + facade setter
  `set_grid_z_clip(grid_id, Option<i32>)`; snapshot round-trip with
  backwards-compatible default (follow `mip_levels_override` QE.5b pattern,
  `crates/roxlap-scene/src/snapshot.rs:122`). Flow the value into `DdaEnv`
  (`crates/roxlap-core/src/dda.rs:48`) and into a spare pad lane of
  `SceneDdaPerGridCamera` (`crates/roxlap-gpu/src/lib.rs:857`, re-uploaded
  per frame; sentinel `i32::MIN` bitcast = disabled) without reading it.
  Gate: full hash suite byte-identical.
- **CA.1 — CPU primary rays.** Gate in `Sampler::hit`
  (`crates/roxlap-core/src/dda.rs:1216`): absolute mip-z =
  `chunk_z * (CHUNK_SIZE_Z >> mip) + loc_z`; cell is air when
  `abs_z < (z_clip >> mip)`. NOT `loc[2]` alone — stacked chz grids need the
  absolute value. Cut colour comes free via decision 3. Tests: golden hash on
  a 3-deck fixture with the clip at a deck boundary; a stacked-chz fixture
  with the clip inside chz=1; a mip-N render pinning the `>> mip` formula;
  clip=None byte-identical.
- **CA.2 — CPU shadows + secondary rays.** `WorldShadowCtx` and dynamic-light
  occlusion marches apply the same per-grid clip (decision 2), including the
  cross-grid path (each tested grid uses its own clip). Test: interior under
  a removed deck is sun-lit, not shadowed by hidden geometry.
- **CA.3 — GPU parity.** WGSL: read the per-grid clip lane in `march_grid()`
  (`crates/roxlap-gpu/shaders/scene_dda.wgsl`) after the chunk-occupancy
  gate, before the colour fetch; same `>> mip` formula as CA.1; shadow path
  identically. Verify the shader's interior-hit colour fallback matches the
  CPU run-top rule (pin with an exact-colour spot check on a cut face).
  Gate: headless CPU-vs-GPU test in `crates/roxlap-gpu/tests/scene_render.rs`
  following the emissive-gate pattern (pixel-classification agreement + exact
  cut-face colours).
- **CA.4 — sprites + picking.** Footprint rule: hide a sprite instance whose
  origin, mapped into a clipped grid's frame, lands inside that grid's XY
  footprint with local `z < z_clip`; instances outside the footprint are
  never affected by that grid's clip. Apply in the CPU sprite pass
  (`crates/roxlap-core/src/dda_sprite.rs`) and in the GPU cull/`CullInstance`
  build. GPU picking is free (reads the clipped render's depth,
  `crates/roxlap-gpu/src/pending_pick.rs`); add a clip-aware variant of the
  facade CPU raycast for click-to-select on decks. Tests: instance above /
  below the plane; above the plane but outside the footprint stays visible.
- **CA.5 — demo + tele-iso preset.** New scene-demo tab "Decks": 3-deck
  shiplet (stacked chz), tele-iso `CameraRig` preset (narrow `fov_y_rad`
  ≈ 0.15, orbit around focus), PgUp/PgDn deck slider driving
  `set_grid_z_clip`, an actor + a point light per deck with the light-cull
  pattern from decision 4. Gate: visual pass by user.
- **CA.6 — book + CHANGELOG.** Rendering chapter gains a Cutaway section
  (API + footprint rule + light-cull pattern); `check-anchors.sh` green;
  CHANGELOG entry. Next cut is minor (additive API).

## Hazards

1. **Colourless RLE interiors.** Mitigated by the existing
   `surface_color_mip` fallback (decision 3); the GPU shader must mirror it —
   pinned by the CA.3 exact-colour check.
2. **Stacked-chz absolute z.** Testing in-chunk `loc[2]` alone silently works
   on 1-chunk fixtures and breaks on real multi-deck ships — that's why CA.1
   requires a stacked-chz fixture.
3. **Mip rounding bleed.** At mip m the clip rounds to `z_clip >> m`; a
   coarse cell straddling the plane can reveal up to `2^m − 1` voxels above
   it at distance. Accepted; CPU and GPU must use the *identical* formula or
   the CA.3 parity gate fails.
4. **Occupancy/bricks ignore z.** BrickMap and chunk-occupancy are not
   clip-aware, so rays march through hidden decks before per-voxel rejection
   — wasted steps, no correctness issue. Measure in CA.5; a per-grid
   clip-aware occupancy variant is a possible follow-up, not v1.
5. **Light leaks from hidden decks.** By decision 4 the engine does not
   filter lights; forgetting the game-side cull reads as a bug. The demo and
   book must show the pattern.
6. **Water interaction.** Flooded-deck shell voxels (WT SHELL design) live in
   the slabs, so they clip automatically; full-screen tint and listener
   lowpass are camera/listener-gated and a tele-iso camera is never
   submerged. Verify once with a flooded-deck fixture in CA.5.
7. **Snapshot compatibility.** Old snapshots must load with `z_clip = None`
   (serde default / version handling per the QE.5b precedent).

## Code map (as of 2026-07-16)

CPU — `crates/roxlap-core`:
- `src/dda.rs:1216` `Sampler::hit` — the single per-cell solidity choke point (CA.1)
- `src/dda.rs:48` `DdaEnv` — per-grid shading env, carries the clip to the marcher (CA.0)
- `src/grid_view.rs:239` `column_slab_mip` — all column reads converge here
- `src/grid_view.rs:357` `surface_color_mip` — run-top colour fallback (cut face)
- `src/dda_sprite.rs` — CPU sprite instance pass (CA.4)

GPU — `crates/roxlap-gpu`:
- `shaders/scene_dda.wgsl` `march_grid()` — per-cell gate + shadow path (CA.3)
- `src/lib.rs:857` `SceneDdaPerGridCamera` — per-grid per-frame upload; spare pad lane for the clip (CA.0)
- `src/pending_pick.rs` — async picking, clip-free by construction
- `tests/scene_render.rs` — headless CPU-vs-GPU gate pattern (CA.3)

Facade/scene — `crates/roxlap-scene`:
- `src/lib.rs:448` `Grid::mip_levels_override` — per-grid config + snapshot precedent
- `src/render.rs:378` `render_scene` — CPU entry, flows Grid → DdaEnv
- `src/snapshot.rs:122` — QE.5b snapshot-compat pattern

Demo — `crates/roxlap-scene-demo`:
- `src/host.rs:186` `CameraRig` — tele-iso preset (CA.5)
- `src/scenes/` — `DemoScene` tab pattern for "Decks"
