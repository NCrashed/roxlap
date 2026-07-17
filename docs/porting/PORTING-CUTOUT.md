# roxlap — occlusion cutout / keyhole (Stage OC)

Entry doc written 2026-07-17 at workspace 0.29.0 + uncommitted CA tail.
This is the **entry doc** for the occlusion-cutout stage — tag **OC**.

## Status — OC.0–OC.3 + OC.5 LANDED (2026-07-17); OC.4 deferred

- **OC.0** — `ViewCutout` on `FrameParams`, facade derivation on both
  backends, `DdaEnv::cutout` (CPU) + uniform cutout block / per-grid
  `rot0.w` focus-plane lane (GPU), all unread; full suite green.
- **OC.1** — CPU keyhole: `cutout_reveal_t` window test in
  `pixel_result` (shared `bayer_threshold`), two-compare hide rule in
  `cell_walk_skip`'s hit branch, `focus_z >> mip`; 6 unit gates + the
  scene-level vws-scaling (hazard 3) and Far-billboard (hazard 8) pins.
- **OC.2** — GPU parity: same formulas + Bayer constants in
  `scene_dda.wgsl`; `HeadlessSceneRenderer::set_view_cutout`; 5
  headless gates mirroring OC.1 (incl. exact cut-face colours, mip
  floor, hidden-wall-still-shadows vs the CA clip side-by-side).
- **OC.3** — "Boarding" tab: shared `build_ship_world(with_stairs)`
  builder (Decks keeps the sealed hull), CC `CharacterBody` at ship
  scale, shoulder boom with NO raycast clamp, wheel = radius
  (`SceneInput::Wheel` plumbed), PgUp/PgDn deck clip composed. Both
  backends smoke-captured; **user visual pass pending**.
- **OC.4** — ghost mode: **no-go for v1 / deferred** (own spike later;
  v1 hides).
- **OC.5** — book "keyhole cutout" subsection + Boarding tour row +
  CHANGELOG (next cut minor).
- **Visual pass round 1 (2026-07-17)** — two findings, both fixed:
  1. **Decision 5 SUPERSEDED**: the Bayer-dithered binary edge read as
     a high-contrast noise mosaic on every surface the cut boundary
     crossed (adjacent pixels alternating wall/room). Replaced by a
     **radial reveal taper** — the keyhole closes into a smooth
     deterministic funnel; no dither anywhere in the cutout path.
     Feather now means the taper band INSIDE the radius.
  2. **Bilge headroom**: with the 8-deep flood, a swimming 12-voxel
     body floated with its head ~2 voxels under the old ceiling and
     the keyhole view into it read as "just the seam". The bilge is
     now 8 voxels taller (`FLOOR_TOPS[2]` 310 → 318, `SHIP_ORIGIN_Z`
     106 → 98, stairs 4-voxel steps so both drops divide evenly), and
     the demo sizes the default keyhole radius to the logical frame.
- **Visual pass round 2 (2026-07-17)** — teeth on shallow surfaces +
  the CA/OC composition:
  1. **Decision 1 REVISED — per-CELL cone, not per-pixel window**: any
     per-RAY gate (dithered or tapered) leaves cells straddling the
     boundary rendered for the outside pixels only — sub-cell
     fragments reading as ragged teeth where the cut crosses floors /
     walls at shallow angles. The hide rule now classifies whole cells
     by their CENTRES against the tapered eye→focus CONE
     (`tan_outer`/`tan_inner` derived from the pixel radius and the
     frame's focal): cube-granular coherent edges, exactly the CA
     clip's edge quality. Wire shape: `CpuCutout` carries grid-local
     `focus_local` + tans + voxel-unit `t_reveal` (LINEAR `/vws` now —
     Euclidean cell distances, not opticast ray depth); the GPU
     uniform carries `(t_reveal, tan_outer, tan_inner)` + the WORLD
     focus (each march converts it through its grid transform); the
     per-pixel kernel test is GONE on both backends. Focus-plane z
     lane unchanged (`rot0.w`, now `i32::MIN` sentinel = the one-
     compare disabled gate).
  2. **Boarding = the composition demo**: the deck clip now FOLLOWS
     the character (`deck_for_body` → `DECK_CLIPS[d+1]`; on a lower
     deck the hull around the keyhole read as the roof — very odd),
     with hotkeys: `K` keyhole, `V` follow/manual (PgUp/PgDn manual),
     `C` shoulder ↔ tele-iso deck orbit (the Decks preset following
     the walker). The Decks tab stays as the CA gallery fixture.
- **Visual pass round 4 (2026-07-17)** — column-hugging reveal:
  standing right behind a body-height pillar left its bottom half
  standing (the fixed eye-distance sphere through the chest: an
  elevated eye is FARTHER from below-chest cells in front than from
  the chest itself, and the old margin 6 held a shell around the
  focus). The reveal surface now hugs the character COLUMN: per cell,
  the reference is the nearest column point at the cell's own height
  (the focus-plane z = feet, mirrored around the focus = head), and
  the cell hides while its eye distance is under the reference's
  minus a small `margin` (wire: `t_reveal` replaced by `margin`; GPU
  enable rides `cutout_a.w`). Pinned by
  `cutout_pillar_in_front_cuts_down_to_the_feet` (mind the run-top
  fallback when building such fixtures: a pillar RESTING on the floor
  merges runs and the cut face under it legitimately shows the pillar
  colour — float it one voxel).
- **Review pass (2026-07-17)** — nine findings, all landed: stairs
  off-by-one (`0..n` built a degenerate inverted last step that
  `set_rect` normalised into a floor sill — `0..n-1`);
  `ViewCutout::new` re-tuned to the post-r3 defaults (feet-plane
  convention); Boarding's wheel radius survives tab switches
  (first-enter latch); doc rot purged (`FrameParams` rustdoc + locked
  decisions 1/5 rewritten as-built); the cone math now runs only for
  OCCUPIED cells (both backends — it previously priced every marched
  air cell above the plane); the GPU kernel receives the per-grid
  local focus pre-converted (`PerGridCamera.cutout_focus_local`, 160
  B — WGSL mirrors in scene_dda + sprite_terrain_shadow must stay in
  lockstep) instead of converting per ray; the world→grid conversion
  (`cutout_grid_local` in roxlap-scene) and the radius→angle shape
  (`cutout_cone_tans` + `CUTOUT_NEAR_Z` in roxlap-render) are
  single-sourced for both backends; and the 17-argument positional
  wrapper ladder is frozen behind `ComposedFrameParams` +
  `render_scene_composed_frame` (`render_scene_composed_with_cutout`
  removed before ever shipping).
- **Visual pass round 3 (2026-07-17)** — two Boarding tunes:
  1. **Deck-follow switches by the HEAD, not the feet**: a feet-based
     switch fired on the top stair steps while the figure's anchor
     (volume centre) still sat above the new clip plane — the CA
     sprite footprint rule hid it and the player vanished
     mid-staircase. The head threshold lands strictly between the
     4-voxel step levels, so no flicker while climbing.
  2. **Hazard 7's z-bias tune inverted**: the head-level focus plane
     left waist-high wall stumps in front of the character. `Z_BIAS`
     is now +6.5 (chest + 6.5 → the plane sits AT the feet after the
     conversion floor()): front walls cut down to the boots, the
     floor plate stays.

## Goal

A camera-relative "keyhole" through front geometry for third-person
play: walls between the camera and a **focus point** (the controlled
character) become air inside a screen-space circular window around the
focus, its rim a smooth deterministic funnel — the classic
BG3/Divinity occlusion cutout, complementing stage CA's deck clip (CA
cuts *ceilings above*; OC cuts *walls in front*, and "front" rotates
with the camera, so it cannot be a grid-state plane). Both backends,
hash-gated, zero cost / byte-identical when disabled.

Non-goal: this is a **view aid**, not world removal. Unlike CA's
"world as if removed", the cut wall keeps casting shadows, keeps
blocking audio/pathing/gameplay raycasts, and keeps its collision.

## Locked design decisions

(Decisions 1 and 5 are the AS-BUILT versions — the entry-doc originals
(per-pixel screen window + `t_reveal` eye-sphere + Bayer feather) were
revised at visual-pass rounds 2 and 4; the status log above records
the why.)

1. **The keyhole model — per-CELL cone + character column** (revised
   at rounds 2/4). A cell is hidden iff ALL of:
   - its CENTRE lies inside the view cone around the eye→focus axis
     (`tan_outer`/`tan_inner` — what `radius_px`/`feather_px` subtend
     under the frame's focal; per-RAY windows leave sub-cell fragments
     that read as teeth on shallow surfaces);
   - its eye distance is under the nearest CHARACTER-COLUMN point's at
     the cell's own height minus `margin` (column = focus xy, z from
     the focus plane (feet) to its mirror above the focus (head); a
     fixed eye-sphere through the chest left waist-high stumps of
     obstacles the character stood right behind);
   - its grid-local z is **above the focus plane**
     (`z < focus_z_local + z_bias`, z-down) — walls cut, the floor in
     front of the character stays. Per-grid plane derived exactly like
     CA's clip conversions (world focus → grid frame; the `>> mip`
     floor formula).
   The z compare gates FIRST (i32::MIN sentinel = one never-true
   compare when off) and the cone math runs only for OCCUPIED cells —
   the occupancy/brick/block skips stay untouched, the CA.1/CA.3
   pattern.
2. **Primary rays only.** Shadow marches, `SceneOccluder`, audio,
   collision, `Scene::raycast`/`raycast_clipped` are untouched. GPU
   depth picking follows the cutout for free (it reads the cut
   render's depth — the CA.4 argument verbatim); that IS the desired
   click-through-the-keyhole behaviour.
3. **Per-frame facade state, not grid state.** The API is
   `FrameParams::view_cutout: Option<ViewCutout>` (`focus_world`,
   `radius_px`, `feather_px`, `margin`, `z_bias`) — it follows the
   camera every frame, so it is transient render state: **no snapshot
   change, no wire bump** (contrast CA.0's v4). The facade derives per
   frame: the cone tangents from the pixel radius under the frame's
   own projection (resolution/SSAA-invariant), and each grid's local
   focus + focus-plane z.
4. **Cut-face colour = run-top fallback.** Same as CA decision 3 —
   already proven on both backends, nothing new to build.
5. **Deterministic feather = radial reveal taper** (revised at rounds
   1/2). Across the `tan_inner`..`tan_outer` band the reveal distance
   scales linearly to zero, closing the keyhole into a smooth funnel —
   NO dither anywhere in the cutout path (the original Bayer edge read
   as a noise mosaic on every surface the boundary crossed). Identical
   formula on both backends; backends need not be byte-equal at the
   rim (parity gates stay classification-based away from it), but each
   must be deterministic frame-to-frame.
6. **v1 hides (air).** Ghost rendering (cut voxels at fixed alpha via
   the TV front-to-back accumulation, which both backends share) is
   explicitly OC.4 — a stretch with its own go/no-go, not v1 scope.
7. **Sprites unaffected.** The character sprite must stay visible
   through the hole (it does automatically: sprites z-test against the
   CUT depth). Sprite instances that ARE front-wall dressing can be
   hidden by a later opt-in flag — footnote, not v1.
8. **Disabled ⇒ byte-identical.** `view_cutout = None` reproduces
   every existing golden hash on both backends (the OC.0 gate and the
   standing invariant; the per-grid focus-plane lane's `i32::MIN`
   sentinel = off, so the hot loop pays one never-true compare).

## Substages

- **OC.0 — plumbing (no-op).** `ViewCutout` on `FrameParams`
  (`roxlap-render/src/lib.rs`, next to `lights`/`tint`); facade
  derivation (screen centre at render res, `t_reveal`, per-grid
  focus-plane z) flowing into a `DdaEnv` field (CPU) and a
  `SceneDdaUniform` block + per-grid camera lane (GPU) — all unread.
  Gate: full hash suite byte-identical.
- **OC.1 — CPU keyhole.** Window test per pixel in `pixel_result`
  (`dda.rs`) → `Option<f32>` reveal-t into `cast_ray`/
  `cell_walk_skip`'s hit branch; hide rule per decision 1 (mind the
  SC `vws` scaling of `t_reveal`, hazard 3, and the mip `>> m` on the
  focus-plane z — CA.1's exact formula); Bayer feather. Tests: golden
  wall-between-camera-and-focus fixture (wall revealed, room behind
  visible), outside-window column intact, below-focus floor intact,
  stacked-chz fixture, disabled byte-identity.
- **OC.2 — GPU parity.** Same formulas + the same Bayer constants in
  the scene kernel (window test once per pixel, reveal params into
  `march_grid`'s hit branch; translucent runs reset like the clip
  does). Headless gates in `roxlap-gpu/tests/scene_render.rs`
  mirroring OC.1's fixtures (classification agreement outside the
  feather band + exact cut-face colours) — reuse the CA.3 harness
  patterns wholesale.
- **OC.3 — demo: third-person "Boarding" tab.** The Decks shiplet +
  a CC `CharacterBody` (`roxlap-scene/src/character.rs:198`,
  `walk`/`WalkInput`; auto-swims in the flooded bilge for free) walked
  in third person: follow camera (shoulder orbit, mouse look), the
  cutout keyed to the character, PgUp/PgDn deck clip COMBINED with the
  keyhole (the two features compose — that composition is the demo's
  point), mouse wheel = radius. Gate: user visual pass.
- **OC.4 — (stretch) ghost mode.** Cut cells accumulate at a fixed
  ghost alpha through the TV translucent path instead of vanishing.
  Both backends share the accumulation machinery, but the perf and
  look need a spike first — explicit go/no-go, allowed to defer to a
  later stage without blocking OC.5.
- **OC.5 — book + CHANGELOG.** Rendering chapter's cutaway section
  grows a "Keyhole" subsection (when to use which: CA = decks, OC =
  third person; the composition pattern); demo tour row; CHANGELOG.
  Next cut minor (additive `FrameParams` field — non_exhaustive, so
  not even literal-breaking).

## Hazards

1. **Render-res vs window-res coordinates.** The RP fixed-res pipeline
   marches at `render_dims`, not the window size; the facade must
   project the focus with the same `OpticastSettings` the frame uses
   (`hx/hy/hz`), or the circle drifts under SSAA/fixed-res. Derive
   inside the render call from the frame's own settings — never accept
   screen-space pixels from the host.
2. **CPU scissor rects.** The per-grid CPU render restricts to screen
   rects; the window test must use the GLOBAL `(px, py)` (available in
   `pixel_result`) — not rect-local coordinates.
3. **`t_reveal` scaling per grid.** CPU per-grid rays measure t in
   voxel units under `vws` scaling (`scale_world_dist_f32` precedent
   in `render.rs`); GPU marches world-local t directly. Same-value
   parity needs the CPU conversion — pin with a scaled-grid test
   (`vws = 0.25` fixture, SC precedent).
4. **Focus-plane z at mip N.** Same `>> mip` floor formula as CA
   (hazard 3 there); coarse-mip bleed accepted, formulas must match
   across backends — pin like `cutaway_clip_mip_formula_floors`.
5. **Skips are not cutout-aware.** Occupancy/brick/super/block skips
   march through the to-be-cut wall and reject per cell — wasted steps
   bounded by wall thickness (a few cells); accepted, mirrors CA
   hazard 4 (whose block-skip resolution already bounds the damage).
6. **Feather + posterize interaction.** The RP posterize/dither runs
   in resolve AFTER the march; the keyhole's Bayer runs in the march.
   Two dithers can moiré — keep the feather band narrow (default ~6
   px) and test with posterize on in OC.3.
7. **Character halfway through a doorway.** The focus plane z-bias
   must keep the doorway lintel cut while the character's own floor
   stays; tune `z_bias` in OC.3 (start: +2 voxels above the head).
8. **Far-tier billboards.** Impostor t is always ≥ `t_reveal` for any
   sane camera (Far means far) — naturally unaffected; assert once in
   an OC.1 test rather than plumbing anything.

## Code map (as of 2026-07-17, incl. the uncommitted CA tail)

CPU — `crates/roxlap-core`:
- `src/dda.rs` `pixel_result` — per-pixel entry; the window test + reveal param live here (OC.1)
- `src/dda.rs` `cast_ray` / `cell_walk_skip` hit branch — per-cell hide rule next to the `z_clip` gate
- `src/dda.rs` `DdaEnv` — carries the derived cutout (OC.0), `z_clip` precedent
- `src/grid_view.rs:357` `surface_color_mip` — cut-face colour fallback (decision 4, nothing to do)

GPU — `crates/roxlap-gpu`:
- `shaders/scene_dda.wgsl` render kernel — once-per-pixel window test; `march_grid` hit branch — per-cell rule next to the `z_clip_mip` compare
- `src/lib.rs` `SceneDdaUniform` — cutout block (screen centre, radius, feather, t_reveal); per-grid focus-plane z rides `SceneDdaPerGridCamera` (a spare lane next to `z_clip`)
- `tests/scene_render.rs` — CA.3 harness patterns to clone (graded fixtures, exact-colour pins)

Facade — `crates/roxlap-render`:
- `src/lib.rs` `FrameParams` (~:971 `lights`, `tint`) — `view_cutout` field; `project_point`-style math for the screen centre at render res
- `src/cpu.rs` / `src/gpu.rs` render paths — per-frame derivation + per-grid transforms (CA's `grid_local_lights` / `cutaway_footprint` precedents)

Demo — `crates/roxlap-scene-demo`:
- `src/scenes/decks.rs` — the shiplet to reuse for "Boarding" (OC.3)
- `crates/roxlap-scene/src/character.rs:198` `CharacterBody` + `WalkInput` — the third-person body (CC stage, auto-swim from WT)
- `src/scene_api.rs` `CameraRig` — follow-camera helper joins `tele_iso`
