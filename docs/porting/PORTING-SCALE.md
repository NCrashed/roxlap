# roxlap — per-grid voxel scale (Stage SC)

Entry doc written 2026-07-07 at workspace 0.25.0, right after the AU
audio stage + 0.25.0 publish. This is the **entry doc** for the
per-grid voxel-scale stage — tag **SC**. A fresh-context session should
read it top to bottom before touching code. Recon: two very-thorough
sweeps (CPU/scene + GPU), both 2026-07-07.

## Status — OPEN (scope locked by user 2026-07-07)

- SC.0 — LANDED 2026-07-07: `GridTransform::voxel_world_size: f64`
  (default 1.0) + `at_scale()`; `addr::world_to_grid_local` (`/vws`) /
  `grid_local_to_world` (`*vws`); `Scene::raycast` t-invariant fix
  (`lo/vws`, `max_dist/vws`, `t_local*vws → t_world` — voxel_dda
  unchanged, marches a unit local dir). 4 tests: scaled voxel mapping,
  scaled+rotated round-trip (vws 0.25..4), raycast returns world t,
  cross-scale nearest-hit by WORLD distance. All 207 scene lib +
  workspace tests green ⇒ vws=1.0 byte-identical.
  **Snapshot persistence DEFERRED**: bincode is strictly positional
  (a missing trailing field is a decode error, NOT a serde default —
  the checked-in v1 fixture proves it), so `voxel_world_size` is
  `#[serde(skip)]` on `GridTransform` (wire form frozen, v1 fixture
  still loads) and NOT yet stored in `GridSnapshot`. A scaled grid
  restores at 1.0. Persisting it needs a real snapshot version-2
  migration (a v1-shadow deserialize) — its own SC substage.
- SC.1 — pending (CPU render: camera + lights scaling).
- SC.2 — pending (CPU cross-grid shadows: WorldShadowCtx + occluder).
- SC.3 — pending (collision + streaming + LOD thresholds).
- SC.4 — pending (GPU: scale in `world_origin.w`, shaders, sprite shadows).
- SC.5 — pending (demo + book chapter + CHANGELOG). **GATE: a
  user-facing scaled scene (the planet+ship demo) MUST NOT ship before
  SC.snap** — until scale persists, saving such a scene silently
  restores flat at 1.0. `save_snapshot` `log::warn!`s on a non-1.0
  grid as a stopgap, but the demo shouldn't advertise a feature that
  doesn't survive a save. Order: SC.snap before SC.5's scaled demo.

## Goal

Let each grid carry its own **world units per voxel**, so a coarse
planet grid (`voxel_world_size = 4.0`, big voxels) and a finely detailed
ship grid (`voxel_world_size = 0.25`, small voxels) coexist in one
scene at the right relative sizes. Today all grids are locked to 1
voxel = 1 world unit (PORTING-SCENE.md row 7: deferred for v1 because it
"would complicate raycast and LOD selection" — both now scoped and
tractable). The 0.3.X candidate the scene doc named.

## The one big idea (both recon reports converge on it)

**Scale is applied ONLY at the world↔grid-local transform boundary;
every marcher, sampler and bake below that boundary is unchanged.**
- world → grid-local: divide by `voxel_world_size` (after un-rotate,
  un-translate).
- grid-local → world: multiply by `voxel_world_size` (before rotate,
  translate).
- The **`t`-invariant fix**: raycast + shadow rays march in grid-local
  voxel units, but their `t` must stay **world-comparable** across
  grids. So on entry scale the ray dir + `max_t` into local units, and
  on exit convert `t_local * voxel_world_size → t_world`.

**Working precedent already in-tree:** sprites/clips ALREADY have exactly
this — `SpriteDense::voxel_world_size` (roxlap-core/src/dda_sprite.rs),
with CPU/GPU parity (the "vws parity" shipped in 0.24.0), and
`SpriteOccluder` already marches *scaled* sprite volumes for shadows.
The GPU sprite-cast shadow shader (`scene_sprite_shadow.wgsl`) already
does the `/s` on the ray. Per-grid scale is **generalising this proven
pattern from sprite volumes to terrain grids** — not greenfield.

## Locked design decisions

1. **Field: `GridTransform::voxel_world_size: f64`** (NOT `scale` — match
   the sprite/clip API name for consistency; `1.0` = today's behaviour).
   `world = origin + rotation · (local · voxel_world_size)`. Add a
   `GridTransform::at_scale(origin, vws)` helper; keep `identity()`/`at()`
   defaulting to `1.0`.
2. **Uniform scalar only** — one `f64`, not per-axis. Anisotropic scale
   would break the "rotation preserves ray length" invariant far more
   deeply (the whole t-conversion assumes an isotropic factor) and
   matches the scalar sprite `voxel_world_size`. Per-axis is explicitly
   out of scope.
3. **Byte-identity gate at `voxel_world_size == 1.0`.** Every substage
   must leave the `1.0` path bit-for-bit identical to today (the usual
   roxlap golden-hash discipline). The scale factor enters as `* 1.0` /
   `/ 1.0`, a no-op at the default. This is the primary regression
   guard on both backends.
4. **GPU storage: reuse `world_origin.w`** (the unused `.w` slot of the
   `PerGridCamera` at binding 15). ZERO new bindings — the 16-buffer
   scene pass is saturated, so this is load-bearing. `set_world_transform`
   stamps `voxel_world_size` into `world_origin.w`; the rotation columns
   stay unit vectors (do NOT fold scale into `rot0/1/2` — the shadow
   unpack needs them orthonormal).
5. **`t` stays world across grids** — the cross-grid min-t composite
   (`compose_into` / the GPU min-t merge) needs NO change: with the
   entry/exit conversion, every grid's reported `t` is already in world
   units. (Confirmed by both reports.)
6. **StreamRadius stays in grid-local voxels** (documented as such);
   only the `world_to_grid_local_pos` conversion divides by
   `voxel_world_size`. Note in docs that a grid's WORLD streaming radius
   is `r_voxels · voxel_world_size`.
7. **Volumetric (Beer–Lambert) thickness** divides per-cell path length
   by `vsize` today; at scale it must divide by `vsize · voxel_world_size`
   (else a scaled grid's fog density shifts). CPU `cast_ray` +
   GPU `scene_dda.wgsl` line ~1001. Pin with a test.

## The 1:1 hardcode map (from recon — file:line)

**CPU / scene (16 sites; scale at the boundary):**
- `GridTransform` — roxlap-scene/src/lib.rs (add `voxel_world_size`).
- `addr.rs`: `world_to_grid_local` (~146, `/vws`), `grid_local_to_world`
  (~177, `*vws`), `voxel_global`/`voxel_split` (voxel-index, unaffected).
- `lib.rs` `Scene::raycast` (~946–973) — the t-invariant fix (scale ray
  + max_t; `t_local*vws → t_world`). Comment at 954–956 is the hazard.
- `collide.rs` `grid_box_overlaps_solid` (~109) — box `/vws`.
- `render.rs` `world_camera_to_grid_local` (~101) — pos + basis `/vws`;
  `grid_local_lights` (~128) — light positions `/vws`.
- `lod.rs` `from_radius` (~151) — voxel-radius threshold `*vws`;
  `select_lod` (~196) already world-space, unchanged.
- `streaming.rs` `world_to_grid_local_pos` (~234) — `/vws`.
- `dda.rs` `WorldShadowCtx` (~386, orthonormal comment 394–396) +
  `occluder.rs` `occluded_in_grid` (~139, "rotation is rigid" comment
  140) — scale ray dir + `max_t` by `1/vws`, same t-invariant hazard.
- `render.rs` `compose_into` (~375) — **NO change** (t already world).

**GPU (scale in `world_origin.w`, ~10 edits):**
- `lib.rs` `GridWorldTransform` (+`voxel_world_size`), `set_world_transform`
  (stamp to `world_origin.w`); `SceneDdaPerGridCamera` layout unchanged
  (144 B, reuse the pad).
- `roxlap-render/src/gpu.rs` `grid_local_camera` (~1674) — divide
  pos/right/down/forward by vws.
- `scene_dda.wgsl`: `grid_local_to_world`/`world_to_grid_local`/
  `world_dir_to_grid_local` (603–623, ×/÷ vws); `march_grid` chunk_dim
  `*vws`; the ray-terminating world thresholds (scan cutoff + opaque fog)
  divide by **vws²** to match CPU SC.1 (opticast/shader depth is
  `world/vws²` under the /vws basis — vws², NOT vws); `shadow_occluded`
  same; volumetric `/(vsize·vws)`. (`mip_scan_dist` is an LOD-picker
  input, deferred like the CPU side — SC.3.)
- `sprite_terrain_shadow.wgsl` transform fns + chunk_dim (sprites RECEIVE
  scaled-grid shadows). `scene_sprite_shadow.wgsl` sprite CAST already
  handles `m.voxel_world_size` — unchanged.

## Substages

- **SC.snap — snapshot scale persistence** (added mid-SC.0; deferrable,
  do before shipping SC to users who save scaled scenes). bincode
  positional format can't gain a trailing field without breaking the
  forever-loadable v1 fixture, so: version the envelope to 2, keep a
  `GridSnapshotV1`/`GridTransformV1` shadow shape for v1 deserialize
  (→ vws 1.0), and store `voxel_world_size` on v2. Until this lands,
  `GridTransform::voxel_world_size` is `#[serde(skip)]` (frozen wire).
- **SC.0 — field + transform boundary + raycast t-fix.**
  `GridTransform::voxel_world_size` + `at_scale`; `addr` transforms;
  `Scene::raycast` t-conversion. Tests: round-trip
  `world→local→world` at vws≠1; raycast into a scaled grid returns a
  world-correct `t`; a two-grid scene (vws 1.0 + 2.0) raycast picks the
  nearer WORLD hit regardless of voxel-local distance; vws=1.0
  byte-identical to today.
- **SC.1 — LANDED 2026-07-07: CPU render (camera + lights + depth).**
  `world_camera_to_grid_local` divides the whole pinhole basis + pos by
  vws (opticast marches the world ray in voxel space);
  `grid_local_lights` divides point positions AND radius by vws (falloff
  is evaluated in the voxel frame); cross-grid depth composited world-
  correctly via a new `scale_depth_rect` × **vws²** (opticast's
  `depth = t·(dir·forward)` shrinks by vws² under the /vws basis — the
  factor is vws², NOT vws; a test with disagreeing voxel/world metrics
  pinned it after the vws attempt drew the wrong grid). Both the direct
  and scissored render paths corrected; all guarded on vws≠1. Tests:
  `sc1_scaled_grid_composites_by_world_depth` (order — world-nearer
  unscaled grid wins over a voxel-nearer scaled one) and
  `sc1_scaled_grid_depth_is_world` (value — a scaled grid rendered alone
  has zb ≈ world depth 114, pinning the vws² factor exactly: 28.5/57/228
  for no-scale/×vws/×vws³ all fail). 209 scene tests green, vws=1.0
  byte-identical.
  **Ray-terminating world thresholds scaled /vws² (SC.1, geometry).**
  opticast writes `depth = world/vws²`; the thresholds it compares
  against that depth to *stop the ray* — the scan cutoff `max_scan_dist`
  (`depth > max_dist`) and the opaque-fog distance `fog_max_dist`
  (`depth >= fog_max_dist`) — are world distances, so each is divided by
  vws² (`scale_scan_dist_i32` / `scale_world_dist_f32`) for the ray to
  reach the intended world range. Without this a fine grid (vws<1) has
  its visible terrain clipped to `range·vws²` (6 % at vws=0.25) — a
  geometry bug, not cosmetic. The world-space grid *distance cull* keeps
  the unscaled `max_scan_dist` (it's already a world compare). Both
  render paths.
  **Light-radius semantics (SC.1).** A *dynamic* point light's radius is
  WORLD (divided by vws → constant world reach across scales). A
  *baked* `BakeLight.radius` stays VOXEL (bake runs in the grid-local
  frame → world reach = `radius·vws`). Parallel to the StreamRadius
  voxel-vs-world note (hazard 5); revisit if bakes gain world radii.
  **NOT scaled (optional future perf, resolved in SC.3):** `mip_scan_dist`
  is a *scene-LOD-picker* input, not compared against the depth buffer
  inside the ray, and is dead config for the DDA backend. A fine grid's
  voxels project smaller and *could* take a coarser mip sooner, but a
  finer-than-needed mip is never wrong — see the SC.3 status, which
  confirms this is an optional optimization, not a scale bug. Shadow
  distances under scale = SC.2.
- **SC.2 — LANDED 2026-07-07: CPU cross-grid shadows.** Two scale factors,
  handled on the two sides of the world-space shadow test:
  - **Caster** (grid being shaded): `WorldShadowCtx` gains
    `voxel_world_size`; `WorldShadow::occluded` (dda.rs) lifts the
    grid-local voxel shadow ray to world by scaling the local **position
    AND direction** by vws (`max_t` is a caster-voxel distance → world
    segment covers `t·vws`).
  - **Occluder** (each grid tested): `GridOcc` gains `vws`;
    `occluded_in_grid` (occluder.rs) divides the inverse-rotated world ray
    (`o` and `d`) by vws to reach that grid's voxel frame. Dividing both
    preserves `t`, so `max_t` still clips correctly.
  A grid shadowing **itself** is scale-invariant (the two cancel), and
  `vws==1.0` everywhere is byte-identical to XS.1. Tests
  `sc2_scaled_grid_casts_world_correct_shadow` (a vws=2 occluder block
  filling the same WORLD box as an unscaled block casts a matching shadow
  within ~30%; negative-verified — drops to zero shadow without the
  occluder-side /vws) and `sc2_sun_shadow_cap_is_world_uniform`. Both
  `occluded_in_grid` and its dense test oracle scaled.
  **Review fixes (3):**
  1. **`shadow_max_dist` is WORLD-uniform**, not a voxel cap.
     `grid_local_lights` divides it by vws (parallel to point-radius/vws),
     so the sun shadow ray reaches the same world distance on every grid —
     a fine flying grid (vws<1) gets full cross-grid shadow reach instead
     of `shadow_max_dist·vws`. (Point-light shadows already march to the
     light's true `dist`, so unaffected.) Correct for both `WorldShadow`
     (×vws lift) and single-grid `SamplerShadow` (voxel march).
  2. **Sprite casters verified.** A sprite's shade builds world-space
     sample points + normals via its basis `s/h/f`, which is pre-scaled by
     the sprite's own `voxel_world_size` (`bb_scale3`, and the
     `voxel_world_size_matches_scaled_basis` pixel-identity test). So the
     shadow rays it emits are already world — the identity `WorldShadowCtx`
     (vws=1) is correct; no sprite-vs-grid vws double-apply (hazard 4).
  3. **Lift precision (hazard 2).** `WorldOccluder::occluded_world`'s
     `origin` and `WorldShadowCtx::origin` are now **f64** so a scaled
     grid's large world coordinates survive the lift (occluders are already
     f64 internally; sprite origin subtraction now in f64 too, avoiding
     cancellation). Narrows to f32 only at the voxel-frame DDA entry. `dir`
     stays f32. Does NOT solve floating-origin (still deferred) — just stops
     discarding f64 precision at an internal boundary.
  210+ scene + core shadow tests green, vws=1.0 byte-identical.
- **SC.3 — LANDED 2026-07-07: collision + streaming + LOD + render cull.**
  Every world↔grid-local boundary that treated 1 world unit = 1 voxel now
  divides/multiplies by vws:
  - `collide.rs grid_box_overlaps_solid` — the world box → grid-local box
    divides by vws (both the axis-aligned and rotated-OBB branches), so a
    scaled grid collides at its true WORLD footprint. Test
    `sc3_scaled_grid_collides_at_world_position`.
  - `streaming.rs world_to_grid_local_pos` — divides by vws (mirrors
    `addr::world_to_grid_local`), so the camera position is voxel-space to
    compare against chunk-AABB distances. StreamRadius stays grid-local
    voxels (world reach = `r·vws`). Test `sc3_world_to_grid_local_pos_scales_by_vws`.
  - `lib.rs Grid::bounding_radius` — now returns **world** units (voxel
    half-extent × vws), pairing with the world-distance LOD thresholds so
    `select_lod` picks the right tier for a scaled grid. Test
    `sc3_bounding_radius_is_world_scaled`. `from_radius` unchanged (its
    input is now a world radius; only test-called today).
  - `render.rs` per-frame **distance cull + screen-rect projection +
    billboard blit + light-reach cull** — new `grid_world_bounds` helper
    folds vws into BOTH the sphere centre (×vws, then rotate+translate) and
    radius (×vws). Previously used `billboard::grid_bounds` (grid-local)
    directly in world comparisons → a scaled grid would be mis-culled /
    mis-projected.
  All 60 collide/stream/lod/render-cull tests green, vws=1.0 byte-identical.

  **Scale-correctness now closed across the CPU pipeline** — the two
  correctness fixes landed alongside SC.3 in review rounds are DONE, not
  open: (i) the **ray-terminating thresholds** (`max_scan_dist`,
  `fog_max_dist`) divide by **vws²** so a fine grid (vws<1) isn't clipped to
  `range·vws²` — see the SC.1 status; the ONE vws<1 render test exercising
  this clip is `sc3_fine_grid_renders_beyond_unscaled_range` (vws=0.5 box
  past `max_scan_dist·vws²`, negative-verified it clips to sky without the
  fix). (ii) the **sun `shadow_max_dist`** is WORLD-uniform (`grid_local_
  lights ÷vws`) — see the SC.2 status; pinned by `sc2_sun_shadow_cap_is_
  world_uniform` (vws=0.5 → 80, vws=4 → 10 voxels, all = the same world
  reach).

  **Intentional inconsistency (do not "fix" back):** *shadow reach* is now
  WORLD-uniform (a shadow is a "does X occlude Y" world question), but
  `BakeLight.radius` and `StreamRadius` stay **VOXEL** (authoring/perf
  knobs whose world reach is `value·vws`). Different by design — spelled out
  so a later refactor doesn't unify them.

  **Perf note (finding — no re-clip):** the primary `cast_ray` step budget
  is bounded by the grid's **voxel-AABB span**, not `max_scan_dist`, so the
  ÷vws² reach extension never hits a hard cap that would silently re-clip a
  fine grid. But a fine grid genuinely costs **~vws²-more marcher steps per
  ray** for the same world reach (a vws=0.25 grid marches ~16× the voxels) —
  inherent to having more voxels along the ray. (The *shadow* march keeps a
  `SHADOW_MAX_STEPS = 1024` degenerate-ray backstop; only reachable at
  extreme vws + large shadow distance.)

  **Deferred (perf, not correctness):** vws-aware *projected-size* mip LOD.
  The DDA backend picks the per-grid mip from `select_lod` (now world-
  correct) + the Mid-tier config; `mip_scan_dist` is dead config for it. A
  fine grid's voxels project smaller so they *could* take a coarser mip
  sooner (a perf win) — but a finer-than-needed mip is never wrong, so this
  is an optional future optimization, not a scale bug.
- **SC.4 — LANDED 2026-07-07: GPU parity.** Turned out **much simpler than
  the CPU** (and than this recon predicted), because the GPU marcher uses a
  **normalized `ray_dir`**, so `t` is already a world distance along a unit
  ray and the marcher runs in a grid-local frame where 1 voxel = 1 unit.
  The whole implementation is: **scale `chunk_dim` and `vsize` by vws** in
  the two marchers (`march_grid` + `shadow_occluded` in scene_dda.wgsl, and
  the mip-0 marcher in sprite_terrain_shadow.wgsl). That makes the entire
  march run in **world-local units** — so automatically, with NO extra
  edits:
  - `t` / `best_t` stay world → cross-grid min-t compositing is correct;
  - voxel indexing (`entry_in_chunk / vsize`) still yields correct indices;
  - the volumetric `seg_len = t_span / vsize` stays a voxel count;
  - `shadow_max_dist` (a world uniform) is **world-uniform by
    construction** — no per-grid ÷vws needed (unlike the CPU, whose shadow
    marcher was voxel-frame);
  - point-light positions (already `grid_local_point` = rotate+translate,
    world-local) and radii (world) are already right.
  So `grid_local_camera`, the shader transform helpers
  (`grid_local_to_world` / `world_to_grid_local`), and the shadow cap are
  **unchanged** — the recon's "divide camera pos by vws / scale seg_len by
  vsize·vws / scale transform helpers" would have been wrong for a
  normalized-ray world-frame marcher. Host plumbing: `GridWorldTransform`
  gains `voxel_world_size`; `SceneDdaPerGridCamera::set_world_transform`
  stamps it into the spare `world_origin.w` (default 1.0 so ×vws is always
  identity, never 0); `roxlap-render/gpu.rs grid_world_transforms` fills it
  from `grid.transform.voxel_world_size`. Test
  `scene_dda_scaled_grid_composites_by_world_depth` (a vws=2 blue grid
  world-FARTHER than a vws=1 red grid but voxel-NEARER — red wins the min-t
  composite; negative-verified BLUE wins with the shader vws forced to 1).
  47 gpu + 17 scene-render tests green, vws=1.0 byte-identical.
  **Deferred:** a full headless CPU-vs-GPU pixel diff at vws≠1. Note the
  depth buffers do NOT differ by vws²: the CPU composite depth is WORLD
  (`scale_depth_rect` already multiplies the written `world/vws²` by vws²)
  and the GPU's `best_t` is world by construction — so both are world and
  should agree. The real blocker is the same one at vws==1: **colour float
  precision** between the CPU and GPU shading paths (why the existing GPU
  tests use *loose* colour classifiers), which is not vws-specific. So a
  vws≠1 **depth** parity diff (both buffers world) is actually feasible and
  would be the strongest SC.4 regression — worth adding when the harness
  gains a depth readback. The composite test already pins world-t ordering.
- **SC.5 — demo + docs.** Scene-demo: give the World scene's ship a
  different `voxel_world_size` (or a new "Scale" tab: a big coarse
  planet + a tiny detailed ship), eyeball both backends. Book: a short
  "Grid scale" section in the Scene-graph chapter (world units per
  voxel; the t-invariant note). CHANGELOG + this doc's status.

## Hazards

1. **The t-invariant (THE central risk).** Raycast + both shadow paths
   assume grid-local `t` == world `t`. Miss one conversion and cross-grid
   depth compositing / shadow lengths go subtly wrong (a scaled grid
   renders in front of / behind where it should, or its shadow is the
   wrong length). Grep every `max_t`, every `t`-return, every cross-grid
   `t` compare. The `1.0` byte-identity gate catches *regressions* but
   NOT a wrong scale factor — write explicit vws≠1 assertions.
2. **f32 precision at extreme ratios.** A planet at vws 4.0 next to a
   ship at 0.25 is a 16× ratio; the GPU camera is f32 and large world
   coords already stress it. Note the "floating origin" mitigation
   (PORTING-SCENE.md R7) as the escape hatch for very large worlds;
   don't solve it in SC, just document the limit.
3. **Volumetric fog density** shifts if the `/(vsize·vws)` isn't applied
   (decision 7). Easy to miss; pin with a test.
4. **Sprite `voxel_world_size` vs grid `voxel_world_size` are separate.**
   A sprite placed in a scaled grid composes both? No — sprites are
   world-posed independently (their vws is the sprite's own). Don't
   double-apply. Verify the sprite pass reads the SPRITE's vws, not the
   grid's.
5. **StreamRadius semantics** — is the user's radius grid-local voxels
   or world units? Locked (decision 6): grid-local voxels; document the
   world = voxels·vws relationship so a scaled grid doesn't silently
   stream a 4× larger world area than the number suggests.
6. **Inter-grid interactions with the destruction wishlist**
   ([[destruction-wishlist]]): falling fine-voxel islands landing on
   coarse terrain will exercise different-scale collision. SC keeps each
   grid's queries in its own local frame at its own vws, so the boundary
   is clean — but note it for when destruction lands.
