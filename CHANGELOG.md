# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

#### `roxlap-core` — PRR (phase_remiporend multi-chz reload)

- **`phase_remiporend`'s post-mip-transition column reload**
  switched from single-chz install to the same multi-chz
  builder VC.6.2 already routes the column-step through.
  Closes the last VC.6 follow-up from 0.4.1's
  Known-limits list. At the camera's own chunk-XY past a mip
  transition, multi-chz scenes (`chunks_z > 1` + content at
  `chz != seed_chz`) need the stitched virtual column or the
  intermediate drawing phases (drawfwall / drawcwall / drawflor)
  read the camera-chunk's placeholder bedrock and bypass to
  sky before the next column-step's multi-chz install fires.
- VC.6.0 fixture (4×4×3 chunk grid, chz=0 camera looking
  down at chz=2 floor) reveals the effect: `total_red`
  104 511 → 105 442 (+931 voxel in-fills, +0.9 %). Visual
  shape unchanged (the trapezoidal floor projection); the
  fix in-fills voxels that were previously sky-bypassed.
- VC.6.2 regression pin re-anchored to
  `0xd8eb_5565_84f2_f30d` (was `0x577e_6879_b86e_f758`).
- User's `z=-19.44` hills demo still byte-stable at VC.5
  baseline `0x15e3_21a1_012a_6109` — the steep pitch
  terminates rays in the camera-chunk; mip transitions
  don't fire there.

## [0.4.1] — 2026-06-01

Mip-N multi-chz column-step fix + column-borrow perf cleanup.
Closes the engine-level mip-N gap that 0.4.0's VC.5 left open
(distant-XY rays at mip-N now stitch chz layers the same way
mip-0 already did), and reverts the structural per-column-step
memcpy cost the VC arc accidentally introduced at 0.2.0. No
public-API breakage vs 0.4.0.

### Performance

#### `roxlap-core` — CB (column borrow)

- **`state.column` switched to `ColumnSource<'a>` enum** —
  `Borrowed(&'a [u8])` for single-chz / N=1 installs (zero
  allocation, zero memcpy per column-step) and
  `Owned(Vec<u8>)` for multi-chz stitched chains. `Deref<Target
  = [u8]>` keeps every `state.column[i]` / `.len()` / `.get(i)`
  call site working unchanged.
- Reverts the structural perf cost the VC arc accidentally
  introduced at 0.2.0 (when `state.column` flipped from `&'a
  [u8]` to `Vec<u8>` to support multi-chz stitching). For
  chunks_z = 1 grids — including the demo's ground (vsid =
  4096) and ship (vsid = 768) — every column-step install now
  borrows back into the chunk's `slab_buf` instead of copying
  the chain bytes into an owned Vec.
- Install sites updated: seed install (`from_seed`), mip-0
  multi-chunk column-step's N=1 fast path, mip-N multi-chunk
  column-step (VC.6.2's swap), single-chunk single-chz column-
  step (both mip-0 and mip-N), and `phase_remiporend`'s
  mid-ray mip-transition reload. Multi-chz stitched installs
  stay on the `Owned` Vec path unchanged.
- Bench gain (4-thread, ROXLAP_STATIC=1, vsid=4096 ground +
  vsid=768 ship, max_scan_dist=512, mip_levels=4): engine-only
  best 52.1 FPS, mean ~46 FPS (vs 0.4.0 baseline 42.9 FPS = ~
  +10-20%; bench has ±5 FPS run-to-run noise). Lower than the
  ~25% structural-gap estimate in [[perf-recovery-landed]] —
  the Deref-through-enum match adds per-read overhead that
  partly offsets the install-time savings. Architectural
  cleanup stands regardless: `install_owned_column` is no
  longer called in production (kept as a chain-walker reference
  for tests).
- Byte-stable across the entire test corpus: VC.5 baseline
  hash `0x15e3_21a1_012a_6109`, VC.6.2 fix hash
  `0x577e_6879_b86e_f758`, oracle 10 MATCH + 2 pre-existing
  CPU divergence all unchanged.

### Fixed

#### `roxlap-core` — VC.6 (mip-N multi-chz column-step)

- **Mip-N multi-chz column-step install** (VC.6.0..VC.6.2).
  Closes the mip-N gap left open by 0.4.0's VC.5 (mip-0 multi-
  chz). `build_owned_column_multi_chz` and `emit_chunk_chain`
  both grew a `mip_level: u32` parameter. The mip-N column-step
  branch at `grouscan.rs::phase_after_delete_kept_presync` now
  calls the multi-chz builder with `mip_level = state.gmipcnt`,
  so distant-XY rays at mip-N stitch chz layers the same way
  mip-0 already did. Intermediate bedrock placeholders are
  stripped at the scaled sentinel `0xff >> mip_level`.

  Visible improvement (synthetic 4×4×3 grid, chz=0 camera looking
  down at a chz=2 floor, `mip_levels=4 + mip_scan_dist=64`):

  | Metric         | Pre-VC.6.2 | Post-VC.6.2 | Δ     |
  |----------------|-----------|-------------|-------|
  | bottom_half red pixels | 19 087 | 104 511 | 5.5× |

  Render shape: small camera-chunk-only red blob → full
  trapezoidal floor at perspective. New regression pin:
  `roxlap_scene_demo::vc6_repro::vc6_2_mip_n_multi_chz_*` at
  hash `0x577e_6879_b86e_f758`.

  User's `z=-19.44` hills demo byte-stable at VC.5 baseline
  (`0x15e3_21a1_012a_6109`) — the steep pitch terminates rays
  inside the camera's own chunk-XY footprint, so the mip-N
  column-step branch never fires at distant XY for that pose.
  New scenes with shallower-pitch cameras over stacked content
  benefit.

### Removed

- **`try_handoff_chunk_z_down`** removed (VC.6.3). The
  rasterizer's mid-render chunk-Z handoff was the
  pre-VC.5 mechanism for stacked-grid rendering; VC.5's mip-0 +
  VC.6.2's mip-N multi-chz install supersede it (every chz is
  pre-stitched into one virtual column at install time, with
  intermediate bedrocks stripped). Audit confirmed the helper
  was dead in all three reachable configurations (mip-0 multi-
  chunk, single-chunk, mip-N gated out). `phase_draw_flor`'s
  bedrock-as-air bypass now falls through to `Phase::AfterDelete`
  unconditionally when the sentinel is hit. Byte-stable: every
  existing render hash unchanged (VC.5 baseline + VC.6.2 fix
  hash both verified post-removal).

### Known limits

- **`phase_remiporend` single-chz reload** at mip transition
  mid-render. The VC.6.0 fixture pose doesn't expose this
  because the chz=2 floor is hit after rays have already
  crossed chunk-XY (column-step's multi-chz path took over).
  Other topologies — content at `chz != seed_chz` visible at
  the camera's own chunk-XY past a mip transition — could
  surface it. Deferred until a real scene demonstrates the
  gap.
- **Edge tearing at deep mip-N** at the back of the VC.6.0
  fixture render. Same area-of-code as the open
  `axis-aligned-mip-beams` artifact (cf-narrowing at remiporend).
- `opticast` `gylookup` overflow at `chunks_z ≥ 4` per grid —
  unchanged from 0.4.0.

## [0.4.0] — 2026-06-01

Virtual-Column-rewrite + perf-recovery release. Closes the S4B.6.j
cross-chunk look-down rendering limitation properly (no longer
mitigated; the live demo materialises the full chunk-z stack), and
restores demo perf to ~3× the pre-VC.5 floor at the spawn pose.

### Fixed

#### `roxlap-core` — Virtual Column rewrite (VC.0..VC.7)

- **S4B.6.j cross-chunk look-down rendering** is fixed at the
  engine level. The rasterizer's column-step now stitches a
  multi-chz virtual column at every chunk-XY crossing — distant
  XY columns see the correct world-z slabs instead of falling
  into the previous handoff's broken `+ chunk_size_z` arithmetic.
  `roxlap_scene::render`'s `stacked_chz0_distant_mountain_visible
  _from_chz0_camera` test (the sanctioned VC.0 fail pin) is now
  GREEN.
- New per-slab chain builders + multi-chz install path:
  `build_owned_column_from_chain`, `build_owned_column_multi_chz`,
  `emit_chunk_chain`. The seed-time path and the per-column-step
  install both route through these so distant columns inherit the
  same multi-chz stitching the camera column gets.
- New parallel `column_z_base: Vec<i32>` translation table —
  per-slab world-z lookup decoupled from the chunk-local slab
  bytes. Lets the rasterizer keep voxlap's u8 slab format while
  the engine reasons about world-z natively.
- Camera-above-stacked-grid: `camera_chunk_air_gap` clamped to
  `origin_chunk_z`; `gline` matches. Cameras above the grid no
  longer collapse to chz=0.

#### `roxlap-scene-demo` — VC.7 cleanup

- Reverted the `HillsChunkGenerator::should_generate` mitigation
  from 0.3.0. The live demo now materialises the full chz stack
  and exercises the engine-level multi-chz path. Repurposed
  `camera_above_hills_only_streams_chz0_chunks` →
  `camera_above_hills_streams_full_chz_stack` as a regression test
  for VC.5's behaviour.

### Performance

- **PR.1 — env-var caching** (`roxlap-core`): the 7
  `std::env::var{,_os}` reads in `grouscan.rs` + `cf_narrow.rs`
  (`ROXLAP_TRACE_PHASES`, `ROXLAP_TRACE_STARTSKY`,
  `ROXLAP_CF_NARROW`, `ROXLAP_CF_NARROW_PER_COLUMN`,
  `ROXLAP_CF_NARROW_PER_COLUMN_NO_I1`, `ROXLAP_CF_NARROW_NOP`)
  are now read exactly once per process via module-private
  `LazyLock<bool>` caches instead of per-call. `run_phases` alone
  fires per-ray (~480K calls / frame at vsid=4096), so the
  previous getenv chain accounted for ~29% of render-frame CPU
  time. Diagnostic flags are not expected to change at runtime,
  so behaviour is preserved.
- **PR.2 — per-grid bounding-sphere distance cull**
  (`roxlap-scene`): `render_scene_composed`'s Near/Mid arm now
  skips the per-grid opticast pass when the grid's bounding
  sphere is entirely beyond `OpticastSettings::max_scan_dist`.
  Each opticast walks ~width\*height rays even when none reach a
  voxel, so far-away grids (marker pillars, distant pickups)
  otherwise paid ~9 ms each per frame. Safe: no ray can reach a
  grid whose closest sphere point is past the scan distance.
- **PR.3 — single-chunk fast path**
  (`roxlap-scene::render_scene_composed`): grids that are exactly
  1 chunk at index `(0, 0, 0)` reuse the
  `GridView::from_single_vxl` view that `chunk_xyz_backing`
  already populated. The rasterizer then takes the single-chunk
  branch in `phase_after_delete_kept_presync` — no per-column-
  step `chunk_at_xyz` / IVec2 equality / `Option::is_some`.
  Bench-neutral in the scene-demo's spawn pose (only one
  qualifying grid is in range; the rest are culled by PR.2),
  architectural prep for callers adding small single-chunk grids.

Combined bench at the demo's spawn pose (4-thread, vsid=4096
ground + vsid=768 ship, max_scan_dist=512, mip_levels=4):

| Config                    | 0.3.0 | 0.4.0 | Δ     |
|---------------------------|-------|-------|-------|
| Live demo (markers on)    | 11.7  | 35.3  | +202% |
| Engine only (no markers)  | ~17   | 46.5  | +173% |

### Added

- `ROXLAP_NO_MARKERS=1` env knob in `roxlap-scene-demo` —
  bench-only switch to skip the 5 marker pillars when measuring
  the core renderer's cost in isolation. Live demo defaults
  unchanged.

### Known limits

- **VC.6 — mip-N multi-chz** is deferred. The mip-N column-step
  still uses single-chz install; demo poses A/B/C/D + the ship
  don't exercise multi-mip multi-chz, so no active regression.
- **opticast `gylookup` overflow** at `chunks_z ≥ 4` per grid
  (surfaced at S7.4, still deferred).
- Per-strip parallel scheduler tops out at 4 worker threads.

## [0.3.0] — 2026-05-31

LOD-and-streaming release: per-grid Far-tier billboard impostors,
mid-tier mip overrides, and an end-to-end streaming + procedural
generation pipeline. Closes the S6 + S7 macro-stages of
[`PORTING-SCENE.md`](PORTING-SCENE.md).

### Added

#### `roxlap-scene` — S6 (Far-LOD billboards)

- **Per-grid LOD picker** (`roxlap_scene::lod`): new
  `LodThresholds { r_near, r_mid, mid_mip_levels, mid_mip_scan_dist }`
  + `Lod::{Near, Mid, Far}` enum + `select_lod(camera_world_pos,
  transform, thresholds)`. Wired into `Grid::lod_thresholds` and
  consulted by `render_scene_composed`. Default
  `LodThresholds::always_near` is byte-identical with the pre-S6
  path.
- **Mid-tier mip overrides** (`Grid::lod_thresholds.mid_mip_*`):
  per-grid Mid-tier override for `OpticastSettings::mip_levels` /
  `mip_scan_dist`. Falls back to caller's settings when both
  fields are `None`. Plays nicely with the existing
  `Grid::mip_levels_override` cap.
- **Billboard impostor cache** (`roxlap_scene::billboard`): new
  `BillboardCache` + `BillboardSnapshot` with 26 canonical
  viewpoints (6 face + 12 edge + 8 corner). Rendered via opticast
  at `D = 8 × bounding_radius` near-orthographic camera against
  the runtime sky. Lazy: built on first Far-tier entry per grid,
  cleared by edits + eviction + stream-in (S7.4).
- **Far-tier blit** (`roxlap_scene::render::billboard_blit_into`):
  walks `BillboardCache::pick_nearest`, projects the grid centre
  via the camera basis, stamps the impostor's RGBA pixels with a
  constant z. Skips by the sky-sentinel (`0x00_00_00_00`) so
  background pixels in the impostor don't write to the
  framebuffer.

#### `roxlap-scene` — S7 (streaming + procedural generation)

- **`ChunkGenerator` trait** (`roxlap_scene::streaming`):
  `Debug + Send + Sync` pluggable per-chunk generator with
  `generate(chunk_idx) -> Vxl` and a `should_generate(chunk_idx)
  -> bool` filter (default `true`). `Grid::generator:
  Option<Arc<dyn ChunkGenerator>>` carries the generator;
  `Grid::ensure_chunk_generated` is the synchronous helper.
- **`StreamRadius { r_active, r_evict }`**: per-grid streaming
  policy in grid-local voxel units. `DISABLED` sentinel (`r_active
  = 0`, `r_evict = ∞`) is the default — pre-S7.1 grids keep their
  "absent stays absent" semantics. `new()` panics on `r_evict <
  r_active`, NaN, or negative.
- **Per-chunk version counter**
  (`Grid::chunk_versions: HashMap<IVec3, u64>`): edits bump;
  `ensure_chunk_generated` does NOT. Survives the
  `SceneSnapshot` round-trip via `#[serde(default)]` so pre-S7.2
  snapshots deserialise cleanly.
- **`Scene::pump_streaming_sync(camera_world_pos)`** (S7.1) +
  **`Scene::pump_streaming(camera_world_pos)`** (S7.3, async):
  per-frame drain + evict + dispatch. Async path uses a dedicated
  `rayon::ThreadPool` + `crossbeam_channel` inbox so chunk
  generation doesn't compete with R12's render pool. Race
  detection: each ChunkResult carries `version_at_dispatch`,
  installation gated on `chunk_version(idx) ==
  version_at_dispatch && !chunks.contains_key(idx)`. Eviction
  also drops `pending_gen` entries so stale results past
  `r_evict` are discarded. `Scene::set_streaming_threads(n)`
  reconfigures the pool (drops + rebuilds; channel survives).
- **Stream-in invalidates billboards** (S7.4): both
  `ensure_chunk_generated` and the async drain clear
  `Grid::billboards` after install so the impostor cache rebuilds
  with the new bounding sphere.
- **`CaveChunkGenerator<G>`** (`roxlap_scene::cavegen`): generic
  adapter wrapping any `roxlap_cavegen::Generator<Params =
  CaveParams>` (works with `BlueCaveGenerator`,
  `MagCaveGenerator`). `chunk_idx.z != 0` returns a bedrock-only
  chunk; `chz = 0` derives a per-chunk seed via FNV-1a of `(base_seed,
  chunk_idx.x, chunk_idx.y)` and calls the inner preset at
  `vsid = CHUNK_SIZE_XY`. Visible chunk-boundary seams documented
  as a v1 limitation; continuous-cave deferred.

#### `roxlap-scene-demo` (default mode now streaming)

- **Streaming-hills demo by default**: `build_demo` attaches a
  `HillsChunkGenerator` to the ground grid with
  `StreamRadius::new(256.0, 384.0)`. Chunks visibly load + unload
  as the camera moves. `ROXLAP_STATIC=1` restores the
  historical 32×32 statically-built ground for regression /
  visual-A-B work. `T` prints `chunks=N pending=N
  radius=A/E` per streaming grid.
- **`StreamingBakeTracker`**: per-frame lighting + mip bake
  driver. Bakes newly-installed chunks + their four cardinal
  neighbours via a `Grid::chunk`-resolving `EstNormCache` reader,
  so chunk-edge brightness banding resolves as chunks settle
  around the camera. Bake-on-stream-in replaces the previous
  in-isolation bake inside `HillsChunkGenerator`, which had no
  neighbour context and produced visible seams.

#### `roxlap-formats`

- `Vxl::reset_to_single_mip` (called from `generate_mips`) now
  walks columns + `slng` to recompute the actual end of mip-0
  data instead of trusting the chunk-creation-time sentinel. Fixes
  an OOB panic on the **second** `generate_mips` call against any
  chunk that had been edited (= had `voxalloc`-driven column
  scatter past the original sentinel). Surfaced by S7.6's
  streaming bake tracker; pre-existing in `roxlap-formats`.

### Public-API breakage

- `roxlap_scene::Grid::generator` field type:
  `Option<Box<dyn ChunkGenerator>>` → `Option<Arc<dyn ChunkGenerator>>`.
  Required so S7.3's async dispatch can clone the generator into
  background rayon tasks. Callers that constructed with
  `Box::new(...)` should switch to `Arc::new(...)`.
- `ChunkGenerator` gained `should_generate(&self, _idx: IVec3) ->
  bool` (default `true`). Existing implementations keep current
  semantics; opt in by overriding.

### Known limits (still open)

- **S4B.6.j cross-chunk look-down rendering** at non-camera XY
  columns still hits the `chunk_world_z_base` mismatch on
  column-step + handoff (rendered surface appears 256 voxels
  below expected). The fix lives in a "virtual stitched-column"
  rasterizer rewrite estimated at 2-4 weeks
  (`memory/project_s4b_6_chz_multi_research.md`); deferred to
  post-0.3.0. Mitigated in the streaming-hills demo by
  `HillsChunkGenerator::should_generate` declining `chz != 0`
  so the camera-above-grid path never materialises the
  placeholder chunks that trigger the bug.
- **`opticast_prelude::derive_prelude`'s `gylookup`** multiplies
  `(chunks_z * 512) * PREC` and overflows `i32` at `chunks_z ≥
  4`. Streaming demo uses `r_active = 256` to stay at `chunks_z =
  3`. Proper fix (wrapping_mul or i64 indexing) deferred.
- **Edits across eviction**: a user edit to a streamed chunk is
  lost when the chunk is evicted + re-streamed. Persistent
  dirty-flag handling is deferred to 0.3.x.
- **CaveChunkGenerator boundary seams**: per-chunk independent
  Worley seeds. Continuous-cave neighbour-aware pool deferred to
  0.3.x.

[0.3.0]: https://github.com/NCrashed/roxlap/releases/tag/v0.3.0

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
