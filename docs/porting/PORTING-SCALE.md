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
- SC.5 — pending (demo + book chapter + CHANGELOG).

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
  `*vws` + `mip_scan_dist /vws`; `shadow_occluded` same; volumetric
  `/(vsize·vws)`.
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
- **SC.1 — CPU render (camera + lights).** `world_camera_to_grid_local`
  + `grid_local_lights` scaling. A scaled grid renders at the right
  world footprint; lights land correctly. vws=1.0 golden-hash unchanged.
- **SC.2 — CPU cross-grid shadows.** `WorldShadowCtx` + `occluded_in_grid`
  scale (ray + max_t). Test: a scaled grid casts a world-correct shadow
  onto another grid (extend `cross_grid_sun_shadow_darkens_other_grid`
  with a vws≠1 caster grid). vws=1.0 byte-identical.
- **SC.3 — collision + streaming + LOD.** `grid_box_overlaps_solid`,
  `world_to_grid_local_pos`, `lod::from_radius`. Tests: box overlap
  hits the right world region in a scaled grid; LOD tier flips at the
  scaled world distance.
- **SC.4 — GPU parity.** `world_origin.w` scale, `grid_local_camera`,
  the `scene_dda.wgsl` edits, `sprite_terrain_shadow.wgsl`, volumetric.
  Headless CPU-vs-GPU diff at vws≠1 (extend the HeadlessSceneRenderer
  harness with a scaled grid). vws=1.0 headless byte-identical.
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
