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

### Post-landing review fixes (user code review, same day)

1. `sprite_terrain_shadow.wgsl` (the spliced sprite-receives-terrain
   shadow marcher) gained the same clip gate — sprites on an exposed
   deck are no longer shadowed by the cut-away hull on the GPU.
2. Far-tier billboards: `BillboardCache::build` renders with the
   grid's clip, records `built_z_clip`; the Far dispatch rebuilds on
   mismatch (self-heals direct `z_clip` writes) and `set_grid_z_clip`
   drops the cache eagerly. Test pins clipped/unclipped snapshots.
3. GPU sprite footprint now comes from the HOST scene's materialised
   chunks (`GridWorldTransform.cutaway_footprint`, fed from
   `Grid::cutaway_volume`) — not the GPU-resident slot AABB, which is
   a moving subset on streaming grids (backend disagreement).
4. The per-grid camera clip lane is a REAL `i32` (both WGSL mirrors)
   — the old f32 bit-carrier made every clip in `1..=0x7fffff` a
   subnormal, which WGSL permits drivers to flush to zero on load
   (silent value-dependent GPU-only "clip off").
5. Decks demo: flood water placed before the crates so the bilge
   crate survives as a submerged solid.
6. Decks demo: `saved_mip_scan` is an `Option` — `0.0` (LOD off) is a
   legal value to restore on exit.
7. CHANGELOG names the full literal-breaking surface (`DdaEnv.z_clip`,
   `GridWorldTransform.z_clip`/`.cutaway_footprint`,
   `BillboardCache.built_z_clip`).
8. CPU sprite pass hoists `Grid::cutaway_volume` once per frame (no
   chunk-map walk / rotation inverse per sprite); `CutawayVolume` is
   public for host-side culls.
9. `CullKey` is `Copy` again — the clip set folds to an FNV-1a `u64`
   fingerprint instead of a per-frame `Vec`.
10. `opticast_settings_fov` deleted in favour of the existing
    `OpticastSettings::with_fov_y`.

NOT separately gated: the GPU sprite-shadow clip (fix 1) has no
headless test — the sprite-shadow splice needs a 22-storage-buffer
device; covered by the Decks visual pass instead.

### Visual-pass round 2 (user report: CPU seams + GPU 11-18 FPS)

Both engine-level, both pre-existing issues the tele-iso deck view was
the first to surface:

1. **CPU seam grid on distant mip-0 geometry** — the DDA empty-space
   skip landed rays by re-flooring `origin + dir·t`; at t ≈ 1150 the
   f32 cancellation noise (≈ 2·ulp(t) ≈ 2.4e-4) exceeds the fixed 1e-4
   nudge → one-cell sideways landings → dark seam lines + sky pinholes
   on brick boundaries. Fix: landings advance each axis by its exact
   crossing COUNT from t-differences (`cell_walk_skip`, the
   `SamplerShadow` mirror, and `occluder.rs`). One stability hash
   re-frozen; no-seam + skip-vs-dense structural gates green.
2. **GPU FPS** — root cause: at tele t_base the whole frame marches
   mip 0 (primary AND shadows; the shadow origin-chunk mip is
   `pick_mip(t_base + 0)`, so no demo-side mip_scan tuning can coarsen
   it), and the GPU marches had NO brick-level empty-space skip —
   hundreds of per-voxel air steps per pixel/shadow-ray. Fix: an
   **empty-block skip** in `march_grid` + `shadow_occluded` +
   `sprite_terrain_shadow` using the coarse SOLID mips already
   resident per slot as brick maps; safety pinned by
   `solid_mips_are_child_supersets` (a clear parent bit ⇒ all children
   clear). Decks GPU: 12 → ~30 FPS (NVK, 860×520), shadows intact.
   Demo also trimmed: `TELE_SCAN_DIST` 4096→2048, `shadow_max_dist`
   512→200, `TELE_MIP_SCAN` = 640.
3. **Bonus find while comparing backends** — the CPU cross-grid
   occluder counted the invisible zero-RGB bedrock PLACEHOLDER (bottom
   voxel of never-written columns) as solid: a stacked-chunk grid cast
   a chunk-wide phantom shadow plate (CPU only; the GPU decompressor
   drops placeholders). Fixed by making the occluder render-solid
   (`surface_color_mip` rule); regression tests both backends
   (`placeholder_bedrock_does_not_occlude`,
   `scene_dda_cross_grid_shadow_survives_tele_distance`). CPU and GPU
   shadow masks now agree exactly on the Decks fixture.

### GPU perf round (user target: 80 FPS; landed at ~127)

Breakdown-driven (A/B FPS at 860×520 NVK: no-lights 81 → shadows were
~20 ms of the 34 ms frame):

1. **Two-tier block skip** — the 8³ skip alone left 25-57 iterations
   per shadow ray crossing chunk air; the coarsest solid mip (32³ at
   mip 0) joins as a super tier, CPU brick/super style. 29 → 46 FPS.
2. **Content-box fast-forward** (march_grid + shadow_occluded +
   sprite_terrain_shadow) — one slab test rejects or advances every
   ray to the grid's content box instead of chunk-stepping the ~1150
   world units of approach void per grid per pixel. 46 → 61 FPS.
3. **Voxel-granular Z bounds** — new `GridStaticMeta.vox_z_lo/hi`
   (solid-bitmap extents, live on refresh/evict/partial-refresh; the
   cutaway clip narrows the top for free) shrink the box from the
   512-voxel chunk stack to the ~74-voxel hull and CAP rays at the box
   exit. 61 → **~127 FPS**; World scene ~366 FPS. Demo knob:
   `shadow_max_dist` 200 → 110 (real occluder reach ≈ 90).
4. **Root-caused a THIRD silent layout bug** while wiring `vox_z`: WGSL
   packs bare `vec3<i32>` members 12-byte-tight, so every
   `GridStaticMeta` tail field was read 4 bytes early (stride still
   matched) — the GPU.13.1 occupancy pyramid's level count read
   `mip_off[3]` = 0 and the pyramid NEVER FIRED. Fixed with
   `@size(16)` on the aabb pair + an offset-pinning test; the sprite
   terrain-shadow mirror was additionally a truncated 144-byte struct
   (garbage meta for g ≥ 1) — full tail now declared.

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
