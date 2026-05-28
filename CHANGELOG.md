# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-05-27

Scene-graph release: many independent chunked voxel grids, each with
f64 world position and `Quat` rotation. Substages S1..S5 of
[`PORTING-SCENE.md`](PORTING-SCENE.md).

### Added

#### `roxlap-scene` (new crate)

- New crate. Scene-graph layer above the per-grid voxel renderer.
  Many `Grid` objects, each a sparse `IVec3 → Chunk` map; each
  grid carries a `GridTransform { position: DVec3, rotation:
  DQuat, scale: f64 }` so worlds can be unbounded in size and
  rotated to arbitrary orientation.
- **Address math** (`roxlap_scene::addr`):
  `world_to_grid_local` / `voxel_split` / `voxel_global` /
  `grid_local_to_world`. Handles negative / boundary / rotated
  cases (15 property tests).
- **Edit API** (`roxlap_scene::edit`): `Scene::set_voxel` /
  `set_rect` / `set_sphere`. Multi-chunk decomposition delegates
  per-chunk writes to `roxlap_formats::edit`. Empty chunks keep
  voxlap's bedrock placeholder at z=255.
- **Snapshot + serde** (`roxlap_scene::snapshot`):
  `SceneSnapshot` / `GridSnapshot`, `Scene::to_snapshot` /
  `from_snapshot`. 100-chunk round-trip via bincode validated.
  `compact_serialize_chunk` rebuilds in column-index order to work
  around `vxl::serialize`'s post-edit non-round-trip.
- **Multi-grid render** (`roxlap_scene::render`):
  `render_scene` (last-grid-wins single-buffer) +
  `render_scene_composed` (per-grid temp buffers + min-z merge).
  Translates the world camera into grid-local space per grid and
  dispatches to `roxlap_core::opticast`.
- **Cross-chunk gline** (S4): 3D DDA over chunk index space
  inside a single grid. New `GridView` / `ChunkGrid` types in
  `roxlap_core` wrap a multi-chunk sparse map; `opticast` walks
  chunk-by-chunk with explicit handoff at boundaries (mid-render
  and at seed-time). Power-of-two chunk_size_xy → arithmetic-shift
  fast path (+22% FPS vs. the pre-S4 single-chunk view).
- **Per-grid rotation** (S5): `world_camera_to_grid_local` inverts
  the grid quat to bring camera basis + position into grid-local
  space; the rasterizer sees an axis-aligned grid and is unaware
  of rotation. Identity-rotation is byte-identical to S4 output;
  180°-Z is bit-exact via exact-arithmetic quat. Per-grid
  lighting bake stays in grid-local space — a rotating ship is
  "lit by a sun that turns with it".

#### `roxlap-core`

- **`GridView<'a>`**: new four-field (vsid, slab_buf,
  column_offsets, mip_base_offsets) borrow-style abstraction
  threaded through every `opticast` / `ScalarRasterizer::new`
  callsite. Replaces ad-hoc `&Vxl` + `usize` argument tuples.
- **`ChunkGrid<'a>`**: sparse multi-chunk view; `GridView`
  carries an optional pointer to a `ChunkGrid` and dispatches
  `chunk_at_xy` / `chunk_at_xyz` via the table for multi-chunk,
  falls back to `Some(Self)`-for-`[0,0,0]` for single-chunk.
- **Cross-chunk rasterizer state**: `GrouscanState` carries
  `current_chunk_idx_xy` / `current_chunk_z` / `current_chunk_exists`;
  column-step routes by `chunk_size_xy == vsid → flat fast path`
  vs `chunk_size_xy < vsid → chunk-swap via chunk_at_xy`. Single-
  chunk path byte-identical to 0.1.x.
- **Z-stacked chunks**: `ChunkGrid` grows `chunks_z` /
  `origin_chunk_z` / `chunk_size_z`. `gylookup` table widened to
  `((chunks_z * 512) >> mip) + 4`. Mid-render handoff for
  look-down across stacked chunks. Slab byte reads operate in
  world-z (= chunk-local + `camera_chunk_z * 256`).
- `Grid::mip_levels_override` field: per-grid clamp on mip
  levels — sidesteps the axis-aligned-mip-N artifact for small
  rotating grids by forcing mip-0.
- Engine bug fixes that surfaced during S4/S5 integration:
  - **OOB-XY chunk-edge streaking** — opticast now seeds OOB-XY
    columns from the `(0, 255, 0)` bedrock placeholder rather
    than a synthesised cf entry. (`outside_orbit` golden refrozen.)
  - **Below-floor sky distortion** — `phase_draw_{cwall,fwall,
    ceil,flor}` drains now write `cx0/cy0/cx1/cy1` alongside
    `i0/i1`, fixing the sky lookup offset.
  - **Below-bedrock all-sky** — camera at z > 255 with
    `treat_z_max_as_air` now synthesises a bedrock air gap
    instead of returning `None`.
  - **Ship-grid mip-N black wall** — `phase_remiporend` reloads
    `state.column` only when `current_chunk_exists`, avoiding an
    OOB march into the wrong sub-table.
  - **Camera above grid skipped** — `chunk_at_xyz` clamps the
    effective chz to `origin_chunk_z` for unstacked grids so a
    camera at world z < 0 renders correctly.
  - **Three rotated-grid camera-placement bugs** — grouscan mip-N
    column-step underflow, chz clamp for camera-below-grid,
    cz_local "below the column" branch.
  - **Mip-N slab_z_at scale fix** — `slab_z_at` shifts the chunk
    world-z base by `>> gmipcnt` so mip-N reads in stacked
    chunks land at the correct world-z.
  - **Pose-D `phase_draw_fwall` bedrock guard** — read
    `column[vptr+1]` as a raw byte rather than casting via
    `slab_z_at` (mirrors drawflor's fix).
- 1 added oracle pose (`outside_orbit`, S1, roxlap-only golden).
  Existing 12 voxlap-comparable oracle hashes byte-identical to
  0.1.x.

#### Public-API breakage

- Everything that took `&Vxl` + raw `vsid` / `slab_buf` /
  `column_offsets` arguments to opticast / grouscan now takes
  a single `GridView<'a>`. ~18 call sites migrated internally;
  downstream callers of `opticast` need the same.
- `roxlap_scene` is new — no migration burden, but pulls in
  `glam` as a public dep.
- `Grid::combined_world` and friends (an Approach-C scaffold
  that briefly existed during S4) — DELETED in S4B.4.b in favour
  of `Grid::chunk(idx)` reader + per-chunk bake closure. Never
  shipped in 0.1.x.

#### Not (yet) included

- Per-grid LOD tier selection / billboards / planet sphere proxies
  (S6 — coming in 0.3.0).
- Streaming + procedural generation (S7 — coming in 0.3.0).
- chz multi-rendering: rasterizer state densely coupled to a
  single column, so rendering content from two stacked chunks
  through one column is unsupported. Scene-demo poses A/B/C/D
  unaffected; full investigation log in memory.

[0.2.0]: https://github.com/NCrashed/roxlap/releases/tag/v0.2.0

## [0.1.1] — 2026-05-07

- docs.rs metadata patch. No code changes.

## [0.1.0] — 2026-05-07

Initial public release of the roxlap workspace.

### Added

#### `roxlap-formats`

- `.vxl` (heightmap voxel world) parser + serialiser, including the
  multi-mip extension (`generate_mips`) and per-mip column-offset
  tables.
- `.kv6` (voxel sprite) parser + serialiser, including the optional
  `"SPal"` palette trailer.
- `.kvx` (legacy voxel sprite) parser + serialiser.
- `.kfa` (kv6 animation rig) parser + serialiser, plus the
  `KfaSprite` host-facing scene type and the `sort_hinges`
  topological-order helper.
- `Sprite` data type — kv6 + world-space pose + flag bitfield —
  with the `SPRITE_FLAG_*` constants.
- All parsers round-trip byte-equally on every fixture they accept.
- **Voxel-edit module** (`roxlap_formats::edit`):
  - Slab-pool allocator on `Vxl` — `vbit` / `vbiti` fields plus
    `reserve_edit_capacity` / `voxalloc` / `voxdealloc`. Port of
    voxlap's `vbit` bitmap free-list (`voxlap5.c:822/841`).
  - Low-level b2 z-range buffer ops `delslab` / `insslab` (port of
    voxlap5.c:4231/4259) plus the slab encode / decode helpers
    `expandrle` / `compilerle` (voxlap5.c:4131/4154) and the
    `slng` slab-chain length walker (voxlap5.c:814).
  - `ScumCtx` — column-edit batch context with the rolling 3-row
    radar-buffer cache (voxlap5.c:4431/4544/4507's
    `scum2_line` / `scum2_finish` / `scum2`). Closure-based
    `with_column` API skips the redundant `expandrle` when
    successive edits land on the same column.
  - High-level region wrappers `set_spans` / `set_cube` /
    `set_sphere` / `set_rect` plus `*_with_colfunc` variants that
    accept any `FnMut(i32, i32, i32) -> i32` colour callback —
    closure captures replace voxlap's `vx5.colfunc` /
    `vx5.curcol` global-state dance.
  - Byte-equality validated against voxlap C: a `setspans` carve
    on a 5×5 column patch produces output byte-identical to the
    voxlap-C reference (fixture captured from voxlaptest's new
    `edit_fixture` harness binary).

#### `roxlap-core`

- Pure-Rust port of voxlap's `opticast` raycaster and `grouscan`
  per-ray voxel-column rasterizer (R4.x).
- Multi-mip rendering via per-mip column-offset tables (R4.5).
- Textured panoramic sky via the `Sky` type and `phase_startsky`
  textured-fill branch (R4.4).
- Per-side wall-face shading (`set_side_shades`, sideshademode swap)
  matching voxlap's `setsideshades` ABI.
- x86_64 SSE2 batches for the four scanline rasterizer paths
  (`hrend` / `vrend` / `hrendzfog` / `vrendzfog`).
- KV6 sprite rendering: 4-plane frustum cull, per-voxel rasterizer,
  9-arm slab walk, alpha-byte face shading, and lightmode-2 point-
  light shading via `update_reflects` (R6.0–R6.5).
- KFA-animated sprites with bone hierarchy, hinge math, and the
  `kfadraw` per-frame transform pipeline (R6.6).
- World voxel lighting (`update_lighting`) — voxlap's
  `updatelighting` baked-intensity pass (R6.5).
- 2D textured-quad blit (`drawtile`) covering voxlap's three quality
  modes.
- High-level `Engine` + `Camera` types with idiomatic Rust
  constructors and getters; `OpticastSettings::for_oracle_framebuffer`
  convenience builder.
- **Multicore CPU rendering** (R12) — three rayon-backed
  parallelism axes:
  - **Per-strip opticast** via `ScratchPool::new_parallel(.., n)`:
    splits the framebuffer into `n` row strips, each running an
    independent opticast pass over its strip's y-range. New
    `OpticastSettings::y_start` / `y_end` clip the projection
    + scan-loop iteration. Peaks at ~1.5× on 4 strips for the
    oracle pose suite (i7-12700H); per-strip ray-fan
    discretisation drifts sub-pixel across `n`, so goldens are
    frozen at `n=1` (single-strip = full frame, byte-stable).
  - **`update_lighting` parallel bake**: outer y-loop is
    `rayon::par_iter`, per-column writes via raw-pointer view
    under the voxalloc disjoint-byte-range invariant. ~3.4×
    speedup on the oracle bake region — turns dynamic /
    per-edit relighting from "pause-the-game" to interactive.
    Bit-identical to sequential.
  - **`draw_sprites_parallel`**: new entry point that par_iters
    over `&[Sprite]` with z-test arbitrating concurrent fb / zb
    writes. `DrawTarget` refactored to `Copy + Send + Sync` raw-
    pointer view. ~4–6× speedup on synthetic many-sprite
    scenes; sprite oracle goldens unchanged (existing 2-sprite
    poses are non-overlapping). Tied-z races on overlapping
    sprites are non-deterministic but visually identical.
  - Three new oracle bench subcommands report scaling curves:
    `bench --threads N`, `bench-lighting`, `bench-sprites`.
  - Full design + measured numbers in
    [`PORTING-MULTICORE.md`](PORTING-MULTICORE.md).

#### `roxlap-host`

- Interactive demo binary (`cargo run -p roxlap-host`) — winit +
  softbuffer window with WASD + mouse-look fly-through over the
  bundled oracle voxel world.
- Animated KFA sprite + procedural rotation demo.
- Textured panoramic sky from the bundled `assets/sky.png`.
- Frame-capture key (`F` writes `roxlap-capture.{txt,ppm}` for
  off-line repro of any rendering artifact).
- World-voxel-lighting toggle (`L`).

#### `roxlap-cavegen`

- New crate. Procedural cave generation; depends only on
  `roxlap-formats` (no renderer).
- `Generator` trait + `CaveParams` struct (seed + 5 shape knobs).
- `BlueCaveGenerator` and `MagCaveGenerator` presets — visual /
  parameter defaults tuned to match Ken + Tom Dobrowolski's 2003
  *Justfly* demo screenshots (`caveblue2m.jpg`,
  `cavemag3m.jpg`).
- Hand-rolled classic 3D Perlin noise + fBm sum (`PerlinNoise3D`)
  and Worley-distance classification + dense-grid emit
  (`worley_classify_grid`, `place_seeds`, `classify_voxel`,
  `classify_voxel_with_perlin`). In-house deterministic
  `SplitMix64` PRNG; no `cmake` / C++ build deps for downstream
  users.
- `pack_dense_grid_to_vxl` — folds a `(VSID × VSID × MAXZDIM)`
  voxel mask + colour grid into voxlap's slab format via the
  `roxlap-formats::edit` pipeline.

#### `roxlap-cave-demo`

- New bin crate. Procedural-cave showcase
  (`cargo run -p roxlap-cave-demo`).
- Cave-gen on startup (`BlueCaveGenerator`, vsid = 128, ~1-2 s).
- WASD + mouse-look fly through with per-axis collision-checked
  movement (camera slides along walls instead of clipping).
- Plasma-bullet projectiles: LMB spawns a hot-pink bullet that
  flies along camera-forward, shrinks with distance for
  along-axis depth perception, and carves a sphere on impact.
- `F` toggles blue ↔ mag cave preset (regenerates the world);
  `R` regenerates with the next seed (preset preserved).
- Fog enabled (low-24-bit colour, no brightness bit per the
  voxlap-fork's `set_fogcol` brightness-bit gotcha).
- Spawn-bubble carve at world centre on init + on every
  regenerate, so the camera never starts inside a wall.

#### `roxlap-oracle`

- Cross-engine render-hash oracle (R8): renders 12 fixed test poses,
  FNV-1a-64 hashes each framebuffer, diffs against
  `tests/golden-hashes.txt`. CI (`.github/workflows/ci.yml`) gates
  every push.
- 5 of 12 oracle poses bit-exact with voxlaptest's C engine output;
  the remaining 7 frozen as roxlap goldens after visual verification
  (sub-pixel rounding noise from `_mm_rcp_ps`'s 12-bit approximation
  varies across CPU vendors).
- `cargo run -p roxlap-oracle -- diff` and the lower-level
  `cmd_debug_gline` subcommand for porting / debugging workflows.

### Documentation

- README rewrite: pitch shape, screenshot, quick-start, crate table,
  links to docs.rs.
- Per-crate Cargo metadata for crates.io discovery (keywords,
  categories, documentation).
- This CHANGELOG.

### Out of scope (for 0.1.0)

- ARM NEON port (R9): scalar fallback used on aarch64.
- wasm32 SIMD + browser host (R10): scalar fallback used on wasm32.
- Voxlap's animation-curve playback (`animsprite` + per-frame
  interpolation): the host drives `kfaval[]` directly.
- Sprite no-z (`SPRITE_FLAG_NO_Z`) overlay rendering: data type
  defines the flag, renderer skips it.
- Multi-mip + voxel-lighting integration with `roxlap-formats::edit`
  ops: edits operate on mip-0 only; the cave demo runs single-mip
  with no baked lighting.
- Spatial-bucketing acceleration for `roxlap-cavegen`'s Worley
  classify: brute-force `O(vsid² × MAXZDIM × seed_count)` today
  (~2 s at vsid=128, longer at vsid=256).
- A third cave-demo preset matching Ken's `caverock1m.jpg`
  rocky-cave reference: requires a sphere-stack generator,
  different algorithm than Worley.
- Full byte-fixture coverage of the edit ops against voxlap C
  (only one `setspans` fixture today; `setcube` / `setsphere` /
  `setrect` rely on round-trip self-consistency tests).

[0.1.1]: https://github.com/NCrashed/roxlap/releases/tag/v0.1.1
[0.1.0]: https://github.com/NCrashed/roxlap/releases/tag/v0.1.0
