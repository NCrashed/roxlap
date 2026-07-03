# roxlap-cave-demo → SceneRenderer facade migration

Start-of-stage brief and locked decisions for **porting
`roxlap-cave-demo` off the low-level `roxlap_core::dda` renderer + its
hand-rolled `softbuffer` loop onto the `roxlap-render` `SceneRenderer`
facade** — so the cave demo runs the same CPU/GPU backends as
`roxlap-scene-demo`, gaining a GPU path (`ROXLAP_GPU=1`) for free.
Companion to [PORTING-SCENE.md](PORTING-SCENE.md) (the scene layer),
[PORTING-GPU.md](PORTING-GPU.md) (the GPU backend), and
[PORTING-DDA.md](PORTING-DDA.md) (the clean-room renderer the demo
currently calls directly).

This is a **start-of-stage brief**. A fresh-context session should read
it top to bottom before touching code. **Demo-only — no library
change.**

## Why

`roxlap-cave-demo` is the last interactive demo still bypassing the
facade: it calls `roxlap_core::dda::render_dda_parallel` straight into a
`softbuffer::Surface` it owns, manages its own z-buffer + sky prefill,
and hand-draws plasma bullets into the pixel buffer. It is therefore
**CPU-only** and duplicates render-loop plumbing that
`SceneRenderer` already owns. `roxlap-scene-demo` is the model: a thin
host that hands the facade a `Scene` + `Camera` + `FrameParams` and lets
it pick the backend.

## Key enabling fact

The cave world is **`VSID = 128` × `MAXZDIM = 256`**, which equals
**exactly one scene chunk** (`CHUNK_SIZE_XY = 128`, `CHUNK_SIZE_Z =
256`). So the cave maps to a **single-grid, single-chunk `Scene`** at
chunk `(0, 0, 0)` — no chunk decomposition, no streaming radius, no
multi-grid composition. And `roxlap-scene` already ships
`CaveChunkGenerator<G>` (`crates/roxlap-scene/src/cavegen.rs:52`)
wrapping the Blue/Mag presets as a `ChunkGenerator`, so the content path
is already validated (S7.5).

## Locked decisions

Taken with the engine author 2026-06-25:

1. **Cave content = `CaveChunkGenerator` + `ensure_chunk_generated`.**
   The grid carries a `CaveChunkGenerator<BlueCaveGenerator|MagCaveGenerator>`;
   chunk `(0, 0, 0)` is materialised synchronously via
   `Grid::ensure_chunk_generated((0,0,0))`. This is the validated
   install path (correct `mip_base_offsets` / `vbit` / edit-capacity).
   **Cosmetic consequence:** `CaveChunkGenerator` derives the per-chunk
   seed as `FNV(base_seed, (0,0,0))` (`cavegen.rs:91`), so the cave is
   *equivalent but not byte-identical* to today's `Preset::generate(seed)`
   (which uses the seed directly). Acceptable for a demo; `R` bumps the
   base seed for a fresh cave.
2. **Bullets = voxel sphere sprites** via the new dynamic sprite API
   (the one [PORTING-SPRITE-API.md](PORTING-SPRITE-API.md) just landed).
   One glowing-sphere kv6 model registered once with `add_sprite_model`;
   each in-flight bullet is an instance (`add_sprite_instance_posed`),
   moved per frame with `set_sprite_instance_transforms`, dropped on
   impact with `remove_sprite_instance`. Voxel-accurate occlusion + a
   second live dogfood of the sprite lifecycle. (Rejected: image-sprite
   billboards — simpler but less faithful; `draw_lines` — wrong look.)
3. **Single PR**, demo-only, no library signature changes.

## Code map (as of 2026-06-25)

Demo — `crates/roxlap-cave-demo/src/main.rs`:
- `struct App { window, surface, engine, zbuffer, bricks, world_version,
  vxl, cam_pos, yaw, pitch, keys, grabbed, last_tick, preset, seed,
  bullets }` (`:224`).
- `App::new` (`:254`) / `regenerate` (`:315`): `build_world` + spawn-bubble
  carve + `relight_world`.
- `camera()` (`:341`), `integrate` (`:371`, collision), `step_bullets`,
  `redraw` (`:531`): builds `GridView::from_single_vxl` (`:575`),
  `bricks.ensure` (`:576`), `render_dda_parallel` (`:593`), then
  `draw_bullet` discs (`:610`).
- `Preset::generate(seed) -> vxl::Vxl` (`:188`) via `BlueCaveGenerator` /
  `MagCaveGenerator`.

Facade — `crates/roxlap-render/src/lib.rs`:
- `SceneRenderer::{new (:577), render (:816), present (:1041),
  resize (:799), set_sky_panorama (:790)}`.
- bullets: `add_sprite_model (:1206)`, `add_sprite_instance_posed (:1151)`,
  `set_sprite_instance_transforms`, `remove_sprite_instance (:1162)`.
- picking (if needed): `view_ray (:1405)`, `pick_depth (:1373)`.
- `FrameParams { settings, sky_color, sky, fog_color, fog_max_scan_dist,
  treat_z_max_as_air, gpu_mip_scan_dist, gpu_max_outer_steps,
  gpu_fov_y_rad, draw_sprites, side_shades }` (`:175`).

Scene — `crates/roxlap-scene/src/`:
- `Scene::{new, add_grid, grid, grid_mut, raycast (:664)}` (`lib.rs:569`).
- `Grid::{new, set_generator (:451), ensure_chunk_generated (:470),
  ensure_chunk, chunk (chunks.rs:143), bump_chunk_version (:436),
  bake_lightmode (chunks.rs:162)}`.
- `Grid::set_sphere_with_colfunc` (`edit.rs:180`).
- `CaveChunkGenerator::new(inner, base_params)` (`cavegen.rs:78`).

Reference host — `crates/roxlap-scene-demo/src/main.rs` (the existing
facade demo: window → `SceneRenderer::new`, per-frame `render` +
`present`, `FrameParams` assembly, streaming pump). Mirror its shape.

## Target architecture

| Concern | Today | After |
|---|---|---|
| World | `vxl::Vxl` + `Engine` | `Scene`, 1 grid, identity `GridTransform`, chunk `(0,0,0)` |
| Content | `Preset::generate(seed)` | `CaveChunkGenerator` + `ensure_chunk_generated((0,0,0))` |
| Present | own `softbuffer::Surface` | `SceneRenderer::{new, render, present}` |
| Render | `render_dda_parallel` | `SceneRenderer::render(&mut scene, &cam, &frame)` |
| Sky / fog / side-shades | `DdaEnv` + manual prefill | `FrameParams` fields |
| Lighting bake | `update_lighting(vxl, engine)` | `grid.bake_lightmode(LIGHTMODE)` |
| Carve | `edit::set_sphere_with_colfunc(&mut vxl, …)` | `grid.set_sphere_with_colfunc` + `bump_chunk_version` + re-bake |
| Bullets | hand-drawn discs in the buffer | voxel sphere sprites (new dynamic sprite API) |
| Collision | `getcube(&vxl, …)` | `getcube` on `grid.chunk((0,0,0))` (or `Scene::resolve_voxel`) |
| Camera | `roxlap_core::Camera` | unchanged |
| Brick cache / z-buffer | demo-owned | facade-owned (deleted) |

**Cargo.toml:** drop `softbuffer`; add `roxlap-render` + `roxlap-scene`.
Keep `roxlap-cavegen`, `winit`, `roxlap-core` (Camera + `getcube`).

## Sub-stages (CR.0 – CR.6)

### CR.0 — scaffold the Scene world (no render swap yet)
Add `roxlap-render` + `roxlap-scene` deps. Build a `Scene` with one grid
(identity `GridTransform`); attach a `CaveChunkGenerator` for the active
preset/seed; `ensure_chunk_generated((0,0,0))`; carve the spawn bubble
via `grid.set_sphere_with_colfunc`; `grid.bake_lightmode(LIGHTMODE)`.
Keep the old render path running so the world is verifiable in isolation
first.

### CR.1 — swap the render loop
Replace `softbuffer` + `render_dda_parallel` with
`SceneRenderer::new(window, size, &RenderOptions { want_gpu, .. })`,
then per frame `render(&mut scene, &cam, &frame)` + `present()`. Map
`engine`/`DdaEnv` → `FrameParams` (`fog_color`, `fog_max_scan_dist`,
`side_shades`, `sky_color`, `sky: engine.sky()`, `treat_z_max_as_air =
true`, `draw_sprites = true`, the `gpu_*` knobs from the scene-demo
defaults). Delete the manual sky prefill + demo z-buffer + `BrickCache`
+ `world_version` (facade owns brick caching). Handle `resize` via
`SceneRenderer::resize`. **Checkpoint:** cave renders on CPU; now
`ROXLAP_GPU=1` runs it on the GPU backend too.

### CR.2 — carve + relight through the grid
Spawn-bubble + bullet-impact carves go to `grid.set_sphere_with_colfunc`;
after each, `grid.bump_chunk_version((0,0,0))` + `grid.bake_lightmode`
(whole chunk — there's only one). Facade GPU backend re-uploads on the
version bump; CPU re-reads the grid each frame.

### CR.3 — collision against the grid
Point camera collision at `grid.chunk((0,0,0))` via `getcube`, or switch
to `Scene::resolve_voxel`. Behaviour-preserving (`PLAYER_RADIUS` slide).

### CR.4 — bullets as voxel sphere sprites
Build one small glowing-sphere kv6 (plasma-pink, e.g. `Kv6::from_fn_shaded`)
→ `add_sprite_model` once → store the `SpriteModelId`. On fire,
`add_sprite_instance_posed(model, pose@muzzle)` → keep the
`SpriteInstanceId` on the `Bullet`. Each frame, after integrating
positions, `set_sprite_instance_transforms(&[(id, pose)])` for all live
bullets. On impact / out-of-bounds / past `BULLET_MAX_DIST`,
`remove_sprite_instance(id)`. (Optional: `compact_sprite_models` is a
no-op here since the single model is never removed.) Delete `draw_bullet`.

### CR.5 — regenerate (F / R) on the Scene
`regenerate()` swaps the grid's generator (`set_generator(Some(Arc::new(
CaveChunkGenerator::new(preset_gen, params_with_seed))))`), forces chunk
`(0,0,0)` to rebuild (clear + `ensure_chunk_generated`, or
`bump_chunk_version`), re-carves the spawn bubble, re-bakes, removes all
bullet instances + clears `bullets`, teleports the camera. `F` toggles
preset; `R` bumps `base_params.seed`.

### CR.6 — polish + docs
Update the module docstring/controls header; `Cargo.toml` dep swap;
CHANGELOG note (demo-only — under the next `[Unreleased]`); mention
`ROXLAP_GPU=1` in the demo README. Run on both backends.

## Tests
- Demo binaries have no unit-test gate today; validate by **running**
  both backends (CPU + `ROXLAP_GPU=1`): fly, fire (sphere sprites occlude
  + carve), `F`/`R` regenerate, resize. Keep the existing crate building
  in CI (`cargo build -p roxlap-cave-demo`).
- If any helper is extracted (e.g. a `build_cave_scene(preset, seed) ->
  Scene`), add a small unit test that the chunk `(0,0,0)` materialises +
  the spawn bubble is air at the camera spawn voxel.

## Risks / watch-items
- **R1 generator seed cosmetics (CR.0):** `CaveChunkGenerator`'s per-chunk
  FNV seed ⇒ the cave differs from today's direct-seed output. Expected;
  documented above.
- **R2 lighting parity (CR.2):** per-chunk `bake_lightmode` vs whole-world
  `update_lighting` — should match for a single chunk; eyeball it.
- **R3 bullet feel (CR.4):** a voxel sphere reads differently from the
  old 2-colour screen-space disc (no fixed-px halo; perspective scales
  it). Acceptable + a sprite-API dogfood.
- **R4 CPU perf:** the facade's `render_scene_composed` adds a thin
  multi-grid-compositor layer vs the direct single-chunk
  `render_dda_parallel`; negligible for one grid, and the GPU path more
  than offsets it.
- **R5 regenerate path (CR.5):** make sure a generator swap actually
  re-materialises chunk `(0,0,0)` (the chunk already exists, so a bump /
  explicit clear is needed — `ensure_chunk_generated` is a no-op on an
  existing chunk).
- **No library change** — purely a demo rewrite. ~1–2 days.

## Commit sequencing (one PR)
1. CR.0 — deps + `build_cave_scene` (generator + spawn bubble + bake),
   old render path still live.
2. CR.1 — `SceneRenderer` render/present swap; delete softbuffer +
   demo z-buffer + brick cache.
3. CR.2 — carve + relight through the grid.
4. CR.3 — collision against the grid chunk.
5. CR.4 — voxel-sphere bullets via the dynamic sprite API.
6. CR.5 + CR.6 — F/R regenerate on the Scene; docstring/README/CHANGELOG.
