# PORTING-WATER.md — water + swimming (Stage WT)

Entry doc written 2026-07-13, right after the PW platform wave closed
(workspace 0.28.0 + unreleased PW work). This is the **entry doc** for
the water stage — tag **WT**. A fresh-context session should read it
top to bottom before touching code. Recon: one thorough sweep
2026-07-13 (CC.4 hook, CharacterBody internals, Volumetric paths,
audio absorption, frame/post hooks — findings below are verified
file:line facts, not guesses).

## Status — OPEN (scope locked by user 2026-07-13)

- WT.0 — LANDED 2026-07-13: `roxlap-scene/src/water.rs` —
  `WaterVolume { lo, hi }` (inclusive grid-local voxel corners,
  normalised in `new`; continuous span `[lo, hi+1)`; surface plane
  `z = lo.z`), `Grid::water_volumes` (pub Vec, `bake_lights`-style) +
  `add_water_volume` + `water_depth_local` (deepest across
  overlapping volumes), `Scene::water_depth_at(world) ->
  Option<(GridId, f64)>` in WORLD units (via
  `streaming::world_to_grid_local_pos`, depth × vws on the way out) +
  `in_water`. Snapshot wire v2 → **v3**: trailing
  `GridSnapshot.water_volumes` (corners re-normalised on load —
  untrusted bytes), frozen `GridSnapshotV2`/`SceneSnapshotV2` shadow
  shapes, `snapshot_v3.rxs` fixture generated + checked in, v1/v2
  fixtures untouched and asserted to restore DRY. 9 water unit tests
  (surface/bottom half-open bounds, overlap-max, identity + vws=0.5 +
  90°-yaw world queries, dry-scene, swapped-corner equivalence,
  equal-depth tie-break) + wire tests 6/6. Scene tests, clippy, fmt,
  workspace rustdoc green; no downstream `GridSnapshot` literals
  existed.
  **Review round (user, 2026-07-13) — 8 findings closed:** corner
  normalisation moved INTO `depth_local` (pub fields made an
  inverted volume dry live but wet after load — live must equal
  restored; load now restores verbatim); `water_depth_at` rewritten
  as one `max_by` with an explicit smallest-`GridId` tie-break
  (HashMap iteration made equal-depth winners non-deterministic —
  WT.2/3 hang state off the id); the envelope-version assert pins
  the LITERAL `3` again (comparing against `SNAPSHOT_VERSION` was a
  tautology); the v3 fixture slimmed 131 KB → 212 bytes (a stray
  `set_voxel` froze a whole chunk into a forever-fixture; chunk
  decoding is v1/v2's job); shadow `From`s chained V1 → V2 → live
  (old links frozen forever; only the last is rewritten per bump);
  `in_water` short-circuits (the per-actor-per-frame query);
  decision 1 + the substage spec de-rotted (point queries only,
  fraction derives in WT.1 — no box query exists).
- WT.1 — CharacterBody swimming: NOT STARTED
- WT.2 — underwater tint (render): NOT STARTED
- WT.3 — underwater audio: NOT STARTED
- WT.4 — cave-demo flood (native): NOT STARTED
- WT.5 — web parity + docs + close: NOT STARTED

## Goal

Swimmable water: the cave demo's lower chambers flood, the player
wades in, floats, dives for crystals, and the world responds — the
frame tints blue-green under the surface, sounds go muffled, bullets
still carve above the waterline. Native and web (cave-web has full
parity since PW.0b; water keeps it).

## Audit facts the design leans on (verified 2026-07-13)

- **The CC.4 passable-veto cannot carry deep water.** `Solidity {
  passable: Option<fn(VoxColor) -> bool> }` (collide.rs:38-82) works
  on `Cube::Color` hits only; slab interiors are `Cube::UnexposedSolid`
  — colourless BY FORMAT (.vxl stores surface colours) — and always
  block. A passable water body works to ~2 voxels of thickness; a
  filled pool has a colourless core the veto never sees. This is the
  load-bearing fact behind decision 1.
- **Volumetric rendering is ready.** `BlendMode::Volumetric`
  (material.rs:141) does per-cell Beer–Lambert on BOTH backends
  (dda.rs:1726-1745; scene_dda.wgsl:1030-1046): `eff_a = 1-(1-a)^seg_len`.
  Terrain .vxl solid runs traverse correctly (interior-retention is a
  KV6-sprite concern only). Known costs, both deferred with owners:
  per-cell `pow()` (PERF G8: precompute `log2(1-a)`, use `exp2`) and
  per-cell `shade_lit` (G3). Fine for a demo pool; measure in WT.4.
- **Gravity lives in one line.** `walk_grounded()` integrates
  `vel.z += gravity*dt` capped at `max_fall_speed`
  (character.rs:315); MoveMode = Walk/Fly/Noclip; z is DOWN (gravity
  positive, jump impulse negative).
- **Audio is one config entry away from "muffled through water"** —
  `AcousticsConfig.material_map` + `absorption` (AU2.0) already
  weight per-material thickness. What does NOT exist: a listener-side
  global lowpass (each source has its own occlusion lowpass;
  `apply_listener` only drives the reverb) — decision 3 adds it.
- **No full-screen tint exists.** FrameParams has fog/sky colours
  (distance-based); the only post is posterize (RP.2), applied at the
  logical resolution in the resolve step on both backends — the
  natural place to hang a tint (decision 3).
- **The demo water fake** (Particles scene) is a 1-2 voxel AlphaBlend
  shell — exactly the thing WT replaces with real volumetric fill +
  volume physics.
- **SC interplay**: grids carry `voxel_world_size`; water volumes are
  declared in grid-local voxels, so every world-side query must scale
  by vws (the same boundary rule as everything since SC).

## Locked design decisions (user, 2026-07-13)

1. **Physics water = `WaterVolume` list on the Grid** (grid-local
   voxel AABBs; the surface is the volume's TOP face — min z, z-down).
   API is **point queries only**: `Grid::add_water_volume` (+ the pub
   Vec for removal/edits, `bake_lights`-style) and
   `Scene::water_depth_at(world) -> Option<(GridId, f64)>` /
   `in_water`. The body's **submerged fraction is NOT a volume
   query** — WT.1 derives it from the centre-line point depth vs the
   body height (`clamp(depth_at(feet) / height)`), which is exact for
   the world-horizontal surfaces water is authored with; do not go
   hunting for a box-overlap query, it does not exist. Deterministic
   (equal-depth ties break to the smallest `GridId`), cheap (few
   AABBs), persists in the snapshot (version bump v2 → v3, SC.snap
   pattern: sibling field, old fixtures still load, missing field =
   no water). The CC.4 veto stays for what it is good at: thin
   pass-through curtains. Visuals are separate: the host fills the
   same region with Volumetric voxels (no new render representation).
2. **Swimming engages automatically by submersion.** No new
   MoveMode. A Walk-mode body whose submerged fraction crosses a
   threshold (with hysteresis) enters the swim state: gravity blends
   toward buoyancy (net upward near full submersion), vertical drag,
   slower accel; `jump` = swim up, a new `WalkInput.sink` = swim
   down; `jump` at the surface breaches into a normal jump.
   `CharacterBody::is_swimming()` for hosts. Fly/Noclip ignore water
   (the cave demo grows a mode toggle in WT.4 so flying explorers
   can opt into wading).
3. **Small engine hooks for the underwater feel** (both reusable
   beyond water):
   - Render: `FrameParams`-driven full-screen tint (colour +
     strength), applied in the resolve step next to posterize on
     BOTH backends. `None` = byte-identical output (the usual gate).
   - Audio: `ListenerAcoustics` grows a listener lowpass field
     (default = no-op); the kira backend puts one filter on the
     master path and tweens it. Water material gets an `absorption`
     entry in the demo config (~0.5 — muffled but audible).
4. **Demo target = flood the cave demo** (native WT.4, web WT.5 —
   parity held): a waterline through the lower chambers, volumetric
   fill + matching WaterVolume, splash particles on entry/exit
   (`voxel_debris` burst at the crossing point), tint + lowpass below
   the surface. Crystals under water = the dive incentive.
5. **Byte-identity gates everywhere**: no water volumes, tint `None`,
   lowpass default ⇒ bit-identical frames + sound on both backends.
   The swim state must not perturb a dry body's walk integration.

## Substages

- **WT.0 — water volumes (roxlap-scene).** `WaterVolume { lo: IVec3,
  hi: IVec3 }` (grid-local, inclusive, surface = lo.z); Grid API +
  point-depth queries (world↔local boundary with vws; see decision 1
  for what deliberately does NOT exist); snapshot v3 (fixture +
  v1/v2-loads); unit tests incl. scaled + rotated grids.
- **WT.1 — swimming (CharacterBody).** Submersion sampling against
  the scene's water volumes; swim state with hysteresis; CharacterDef
  grows `buoyancy`, `swim_speed`, `swim_accel`, `water_drag`,
  `submerge_threshold` (all defaulted so existing defs compile);
  `WalkInput.sink`; breach-jump at the surface. Tests: float to rest
  at the surface, dive/rise controls, walk-out on a shore slope, dry
  path byte-identical, threshold hysteresis (no mode flicker at the
  waterline), scaled-grid water.
- **WT.2 — underwater tint (roxlap-render).** FrameParams tint
  (colour + strength 0..1) composited in the resolve step (CPU +
  scene_resolve.wgsl), orthogonal to posterize; `None` byte-identical
  (pinned); headless GPU/CPU parity test.
- **WT.3 — underwater audio (roxlap-audio).** `ListenerAcoustics`
  lowpass field + kira master filter (default no-op, pinned);
  DemoAudio-side: submerged listener drives the lowpass + water
  material absorption entry. Unit test on the pure side; listening
  pass owed at WT.4.
- **WT.4 — cave-demo flood (native).** Waterline plane in the lower
  cave: volumetric water fill (define material, colour → material
  map — the SAME map pattern as crystals) + `WaterVolume`; V-key
  fly ⇄ walk toggle so swimming is reachable; splash particles at
  surface crossings; tint + lowpass wired to submersion; bullets
  vs water documented (v1: carving below the waterline leaves the
  volume as-is — an air pocket LOOKS dry but still swims; known v1
  seam). Visual + listening pass owed.
- **WT.5 — web parity + docs + close.** cave-web gets the same flood
  (PW.0b parity held; watch CPU-fallback perf with volumetric fill —
  the fog wall at 128 caps ray length, measure before tuning); book
  (scene-graph: water volumes + swimming; demo-tour rows; audio +
  platforms cross-refs), CHANGELOG, status, memory.

## Hazards

1. **z-DOWN sign traps.** Buoyancy is a NEGATIVE-z acceleration; the
   surface is the volume's MIN z. Every review of WT.1 should read
   the signs twice (the controller chapter's "compiles fine and runs
   upside down" warning applies verbatim).
2. **vws boundary.** Water volumes are grid-local; a scaled grid's
   world waterline moves with `voxel_world_size`. Pin with a scaled
   test in WT.0 AND WT.1 (SC discipline).
3. **Volumetric cost in deep water.** Per-cell `pow()` + per-cell
   lighting through many water cells per pixel. The cave fog wall
   caps ray length; if WT.4/WT.5 profiling still hurts, pull PERF G8
   (`exp2` rewrite) into the stage rather than shipping a slow demo.
4. **Carve-vs-water incoherence** (decision 4's v1 seam): shooting
   below the waterline makes an air pocket that still swims.
   Document; a real water-settling sim is explicitly out of scope.
5. **Snapshot compatibility.** v3 must load v2 fixtures (SC.snap
   precedent has the shadow-struct recipe); add the reverse-rot
   fixture test BEFORE wiring demos.
6. **Waterline flicker.** Submersion threshold without hysteresis
   oscillates at the surface (bobbing body + discrete sampling);
   decision 2 mandates hysteresis — test it explicitly.
7. **Web CPU fallback.** 320×256 + volumetric fill may still dip —
   the trunk-profile lesson says measure release builds only.
