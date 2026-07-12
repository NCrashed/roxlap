# PORTING-DESTRUCTION.md — voxel destruction stage (DT)

Entry doc written 2026-07-12 at workspace 0.26.0+ (post-SC, post-EV
tails). This is the **entry doc** for the destruction stage — tag
**DT**. A fresh-context session should read it top to bottom before
touching code.

The marquee voxlap feature: carve a tunnel under an overhang and the
unsupported rock **crumbles** — detaches, falls, and shatters into
debris. Scoped with the user 2026-07-12: demo = **cave-demo** (shots
already carve), islands fall **vertically with a light cosmetic
spin**, landing **always shatters into particles** (no re-stamping
into the grid), per-material fracture patterns land **late in this
stage** (DT.5) after the plain crumble loop works.

## Status

- DT.0 — LANDED 2026-07-12: `roxlap-scene/src/islands.rs` —
  `detect_islands(grid, carve_lo, carve_hi, budget)` → `Vec<Island>`
  (+ `DEFAULT_ISLAND_BUDGET = 4096`), re-exported at the crate root.
  Budgeted span-BFS exactly as locked (decision 2): runs decode
  straight off the slab chain (mirror of the `vxl_voxel_solid` walk),
  chz stacks merge at chunk borders, support = **every** chunk's
  final run, unconditionally. The first cut gated the anchor on "no
  chunk below" — a maintainer review caught that as a phantom-island
  bug on chz stacks: every materialised chunk's local z=255 is
  format-pinned placeholder bedrock (`delslab` clamps below
  `MAXZDIM`, uncarvable ⇒ a component containing it can never fall),
  and the ungated version let an upper chunk's 128×128 bedrock sheet
  flood into components (silent budget-suppression with the default
  budget, an unextractable mega-island with a raised one). Pinned by
  `stacked_chz_bedrock_is_anchored` (pillar through the chz border,
  cut below it). Perf shape that landed: per-call **arena** (one
  shared `runs` + parallel `visited: Vec<u32>` keyed by `(start,
  len)` ranges — no per-column allocations), a multiply-mix
  column-key hasher (SipHash dominated the probe-heavy plate case),
  and a one-entry `Grid::chunk` cache. Probe (`--ignored`, release):
  supported-exit on a 100×100 plate **~0.3 ms** (gate < 0.5 ms),
  63-voxel beam detach **~14 µs**. Tests: beam tip / arch both-legs /
  cross-chunk-border / stacked-chz bedrock anchor /
  budget-exceeded / carve-in-air / two-islands-one-carve (keystone) /
  dense-oracle property test (exact agreement on a randomised scene)
  — 9 total, all green; clippy + fmt clean. Known-fine notes: the
  neighbour scan is linear over a column's runs (binary search over
  the z-sorted list is the fix if riddled cavern columns ever show up
  in the probe), and `Flood::new` does one pass over `Grid::chunks`
  keys (scales with world size, not carve size — microseconds at
  current grid sizes).
- DT.1 — LANDED 2026-07-12: three `Island` methods in `islands.rs`.
  `extract(grid, bake)` — one `set_rect(None)` per contiguous column
  segment (the voxel list is run-major, coalescing is a single pass)
  + one `bake_bbox` over the island bbox; edits + bake bump chunk
  versions. `to_kv6()` — `Kv6::from_fn` over the bbox-relative voxel
  set, colours normalised to full-bright `0x80RRGGBB` (sprite paths
  light the model themselves; a baked byte would double-darken);
  pivot = bbox centre per `from_fn`. `world_origin(&GridTransform)` /
  `world_pivot(...)` — bbox-min corner / bbox centre through
  `grid_local_to_world` (`voxel_world_size` honoured, SC) —
  `world_pivot` is the spawn pose that puts the model exactly over
  the voxels it replaced. Tests (4): extract leaves air + support
  stays + re-detect finds nothing + version bumped; vertical-run
  extraction across the chz border; KV6 dims/count/colours for a
  thin beam; world anchors under `at_scale(…, 2.0)`. 229 scene lib
  tests green, clippy + fmt clean. Maintainer-review follow-ups
  (2026-07-12): extract's doc states the no-remip contract explicitly
  (see hazard 3b — DT.4 obligation); empty-`Island` extract is an
  early-return no-op; the brightness-normalisation branch of
  `to_kv6` is pinned by a dim-byte (0x40) voxel in the colour test.
  Known-fine: a horizontal plate extracts as N single-voxel
  `set_rect`s (µs at budget sizes; batch `set_spans` (S4.1) is the
  fix if DT.2 profiling ever shows it).
- DT.2 — LANDED 2026-07-12: `roxlap-render/src/debris.rs` —
  `DebrisSystem` (+ `DebrisImpact`), re-exported at the crate root.
  Shaped exactly like `ParticleSystem`: `spawn_island(scene, grid,
  island, bake)` + `update(scene, dt)` are pure simulation
  (spawn extracts via `Island::extract` — **mips stay the caller's
  obligation**, hazard 3b — and registers the body at
  `world_pivot`); `sync(renderer)` mirrors bodies through a
  `DebrisFacade` seam (the `ParticleFacade` pattern grown by the
  model half): pre-posed spawn (no first-frame flash), one batched
  transform write per frame, despawn + model removal on impact;
  `tick` = update + sync; `drain_impacts()` hands DT.3 each landed
  island's voxels + world pos + speed. Physics as locked: gravity
  22 u/s² (+z, particle-matched), terminal clamp 60 u/s, cosmetic
  yaw hashed from the island position (stateless — replays
  bit-identical); collision = unrotated AABB (grid-vws-scaled, shrunk
  by an anti-flush epsilon so resting/flush contact never re-fires)
  via `box_overlaps_solid`, contact binary-searched to the voxel
  plane. Sprite pose uses the particle scaled-basis convention
  (yaw about world-z × vws — chirality preserved); grid rotation is
  NOT applied to the falling pose (v1, axis-aligned grids; noted in
  the module doc). Tests (5): monotonic fall + terminal clamp;
  lands flush on a z=140 plate (bottom within 5e-3) + exactly one
  impact + island voxels ride along; spawn extracts from the grid;
  empty/stale spawns refused; full mock-facade lifecycle (model +
  pre-posed spawn at pivot → per-frame batches → despawn + model
  removal, nothing leaked). Maintainer-review fixes (2026-07-12):
  **(a) tunneling** — collision was endpoint-only, so a frame's
  displacement past the overlap window (stock tuning at 30 FPS sits
  exactly on it; hitches/raised terminal blow past it) skipped
  one-voxel shelves; `update` now marches the displacement in
  substeps of half the window (`half.z + vws/2`) — pinned by a
  deterministic 8-units-per-frame vs 2-unit-window test. **(b) spawn
  inside geometry** (island bbox wraps a surviving support) is now an
  explicit policy — immediate zero-speed impact, never a body — not
  a degenerate-search accident; pinned + documented on
  `spawn_island`. **(c)** two-body test covers the mid-iteration
  `swap_remove` path (batches 2 → 1 → 0); binary search 32 → 16
  iterations; per-sync move batch reuses a scratch buffer. 84 render
  lib tests green, clippy + fmt clean.
- DT.3 — impact shatter + audio: NOT STARTED
- DT.4 — cave-demo crumble: NOT STARTED
- DT.5 — per-material fracture patterns: NOT STARTED
- DT.6 — docs: NOT STARTED

## Goal

Three user-visible things:

1. **Floating-island crumble** — after any carve, voxel regions no
   longer connected to support detach, fall under gravity as sprites,
   and burst into colour-true debris particles on impact (with the
   muffled boom the audio stage already provides).
2. **A reusable `DebrisSystem`** (roxlap-render, sibling of
   `ParticleSystem`) — hosts get the whole loop with three calls:
   detect → spawn → tick.
3. **Per-material fracture patterns** (DT.5) — a detached region
   splits into fragments per its material: STONE into small rounded
   chunks (jittered Voronoi cells), GLASS/crystal into sharp planar
   shards. Data-driven per material id.

## Audit facts the design leans on (verified 2026-07-12)

- **No connectivity code exists anywhere** in the workspace — DT.0 is
  a clean-room algorithm, not a port.
- Chunk storage is RLE slab columns (`Vxl.data` + `column_offset`);
  spans decode as `[top, bot)` runs
  (`roxlap-formats/src/edit.rs:4-14`); the alloc-free per-voxel test
  is `vxl_voxel_solid` (`roxlap-scene/src/chunks.rs:101-124`);
  `expandrle` (`edit.rs:188-202`) is the full-column reference.
  Chunks live in `Grid::chunks: HashMap<IVec3, Vxl>` (missing = air);
  bedrock is implicit at the column bottom (z = 255, z-down world).
- Edits: `set_sphere_with_colfunc(…, SpanOp::Carve, …)` +
  `bump_chunk_version_bbox` + `bake_bbox` (~0.04 ms) + `remip_bbox`
  are the carve primitives; cave-demo runs them on a **cloned chunk
  in a background worker** (`CarveJob`, main.rs:1079-1224) and swaps
  the result in on the main thread.
- Islands-as-sprites is fully plumbed: `Kv6::from_fn` /
  `from_fn_shaded` build a model from a closure at runtime
  (`roxlap-formats/src/kv6.rs:173-233`), the facade has
  `add_sprite_model` / `add_sprite_instance_posed` /
  `set_sprite_instance_transform` (O(1), batched) /
  `remove_sprite_model` + `compact_sprite_models`
  (`roxlap-render/src/lib.rs:2525-2755`).
- Collision primitive: `box_overlaps_solid(scene, min, max, solidity)`
  (`roxlap-scene/src/collide.rs:92`) — an AABB-vs-scene query, exactly
  what a falling island needs. No sprite physics exists; cave-demo
  bullets integrate `pos += vel·dt` by hand — the pattern to lift.
- Debris burst: `ParticleSystem::carve_debris`
  (`roxlap-render/src/particles.rs:820-920`) already samples voxel
  colours, applies a radial kick, stride-samples to a cap, and spawns
  a transient emitter. DT.3 factors its burst half out so an island's
  voxel list (already in hand) can shatter without a carve.
- Audio hook: `DemoAudio::impacts(hits, scene, listener)`
  (cave-demo/src/audio.rs:108-114) — occlusion-shaded one-shot booms,
  capped per frame. Crumble impacts reuse it as-is.

## Locked design decisions

1. **Island = sprite, not grid.** A detached region becomes a KV6
   sprite model + one posed instance (`add_sprite_instance_posed` —
   no axis-aligned flash). Grids stay heavyweight world containers;
   islands are transient. Consequence: the wishlist's "inter-grid
   collision" is **descoped** — a falling island collides
   sprite-vs-scene via `box_overlaps_solid`, which covers the whole
   demo loop. (`Kv6::from_fn` keeps surface voxels only — fine: the
   interior is invisible in flight, and the shatter samples the
   host-side voxel list, not the model.)
2. **Detection = budgeted span-BFS, support = bedrock or budget.**
   `detect_islands(grid, carve_lo, carve_hi, budget)` runs on the
   RLE spans directly (a run is one BFS node — chunk-format-native,
   no dense decode): seeds are the solid runs 6-adjacent to the
   carved bbox; each seed component flood-fills with 6-connectivity;
   a component that **touches a bedrock-anchored run** (the final run
   of ANY materialised chunk's column — its local z=255 voxel is
   format-pinned and uncarvable, so the component can never fall;
   this holds on chz stacks too, see the DT.0 status note) or
   **exceeds `budget` voxels** is supported (abort early, cheap);
   only components that exhaust under budget without support are
   islands. Cost per carve ≈ O(Σ min(component, budget)). False
   negatives (a giant detached slab > budget keeps floating) are
   accepted and tunable — voxlap behaved the same way.
   Default `budget = 4096` voxels.
3. **Detection lives in roxlap-scene** (`islands.rs`), pure and
   synchronous over `&Grid`:
   `Island { voxels: Vec<(IVec3, VoxColor)>, bbox: (IVec3, IVec3) }`.
   The cave-demo wires it into its existing carve **worker** (the
   worker owns the post-carve chunk clone — detection there is
   race-free and off the main thread); `CarveDone` grows an
   `islands` field, extraction/spawn happen on the main thread.
4. **DebrisSystem owns the loop** (roxlap-render/src/debris.rs,
   host-owned like `ParticleSystem`): `spawn_island(renderer, island,
   grid_transform)` extracts (batch-carves the island's voxels from
   the grid, one `bake_bbox` + version bump), builds the model, and
   registers a body; `tick(renderer, scene, dt)` integrates and
   returns impact events. Physics: semi-implicit Euler, gravity +z
   (z-down world, same convention as particles' `[0,0,22]` default),
   **vertical fall + cosmetic slow yaw spin**; collision tests the
   island's **unrotated** AABB via `box_overlaps_solid` (the spin is
   visual only — small angles, never part of the collision shape).
5. **Impact always shatters** (user decision): on contact the island
   despawns (instance + model removed; `compact_sprite_models`
   periodically) and its voxel list bursts into particles via a new
   `ParticleSystem::voxel_debris(voxels_world, outward_from, outward,
   def)` — the burst half of `carve_debris` factored out (colour-true
   tint, radial kick, stride-sample cap). `carve_debris` is
   reimplemented on top of it — behaviour byte-compatible.
6. **Fracture patterns split islands, they don't reshape carves**
   (DT.5). `FracturePattern` is data on the material table's id
   (side table in DebrisSystem, NOT a `Material` field — the render
   palette stays render-only): `Chunks { cell }` partitions a
   detached voxel set by jittered-Voronoi seeds (stone: small round
   fragments), `Shards { max_slabs }` by 2-3 random parallel-ish
   planes (glass: sharp angular plates), `Whole` (default) keeps one
   island. Each fragment becomes its own falling sprite with a small
   outward impulse. Deterministic (PCG32 seeded from carve position,
   like particles).
7. **Support anchor v1 = bedrock contact.** Grids without bedrock
   columns (a ship in flight) get no crumble this stage — a future
   `SupportPolicy` can add "anchored chunk" flags. The cave grid (the
   demo target) is bedrock-floored everywhere.
8. **Perf gates, house-style:** an `#[ignore]`d timing probe
   (mirroring the 10k-particle probe) measures `detect_islands` on a
   worst-case supported carve (budget-exit path) and a typical
   overhang detach; target: supported-exit < 0.5 ms, detach ∝ island
   size. Detection must never run on the render thread in the demo.

## Substages

- **DT.0 — island detection core (roxlap-scene/islands.rs).**
  Span-BFS with budget + bedrock support; 6-connectivity; crosses
  chunk borders (HashMap lookup, missing chunk = air, chz stacks
  honoured). Tests: pillar cut → island; arch cut at one leg →
  supported; arch cut at both legs → island; island spanning a chunk
  border; budget-exceeded slab stays supported; carve that touches
  nothing solid → empty. Property test: detected island voxels are
  exactly the solid voxels of the component (compare vs a naive
  dense flood-fill oracle on small grids).
- **DT.1 — extraction → sprite.** `Island::extract(grid)` batch-
  removes the voxels (span-aware, one `bake_bbox` + one version-bump
  bbox); `island_model(&island) -> Kv6` via `from_fn` over a sparse
  set; world pose from bbox-min through the grid transform
  (`voxel_world_size` honoured — SC). Test: extract leaves air +
  a re-run of detect finds nothing; model dims == bbox dims.
- **DT.2 — DebrisSystem: falling + collision.** Body integration
  (gravity, terminal-velocity clamp, cosmetic yaw), unrotated-AABB
  ground test via `box_overlaps_solid`, impact events drained by the
  host; instance pose sync batched. Tests: island falls in empty air
  monotonically; lands exactly on a floor (AABB flush, no
  penetration); event fired once.
- **DT.3 — impact shatter + audio.** `voxel_debris` factored out of
  `carve_debris` (byte-compatible reimplementation + regression
  test); impact event → burst from the island's own colours + model
  removal + periodic compact; demo will feed the event position to
  the existing `impacts()` audio hook. Tests: burst tint histogram
  matches island colours; models/instances fully reclaimed after
  shatter (no leak across 100 cycles).
- **DT.4 — cave-demo crumble.** Worker detects after each carve
  batch (budget-capped), main thread spawns via DebrisSystem; boom on
  impact through the existing audio path; `ROXLAP_NO_CRUMBLE=1`
  escape hatch. Visual eyeball pass owed to the user: shoot out an
  overhang, watch it drop and burst; both backends.
- **DT.5 — per-material fracture patterns.** `FracturePattern` table
  + Voronoi/planar partitioners (pure functions over a voxel set,
  seeded); cave-demo maps crystal material → `Shards`, rock →
  `Chunks`; crystals shatter into glinting plates (they are
  AlphaBlend+emissive — the shard sprites inherit the material and
  glow as they fall, sprite emissive landed 2026-07-12). Tests:
  partition is a disjoint cover; determinism; shard planarity metric.
- **DT.6 — docs.** Book: new "Destruction" section (Lighting-chapter
  style: runnable device-free example printing detected islands +
  demo-tour paragraph); CHANGELOG; this doc's status; memory note.

## Hazards

1. **Detection cost on big carves.** Every carve seeds a BFS; a
   mountain-side carve must exit on budget fast. The span
   representation is the defence (one node per run, not per voxel) —
   keep the budget check per-run-accumulated, not per-voxel.
2. **Cross-chunk islands.** An overhang can straddle chunk borders
   (and chz stacks). BFS must treat `chunks: HashMap` lookups as the
   neighbour source, never a single-chunk view. The cave-demo worker
   only owns ONE cloned chunk today — detection there is
   chunk-local; islands that cross the border would be missed. v1
   accepts chunk-local detection **in the demo** (single-chunk cave),
   but the roxlap-scene API is grid-wide and tested cross-chunk.
3. **GPU re-upload spikes.** Island extraction is a second edit right
   after the carve — keep both inside the same dirty bbox where
   possible so the partial-refresh path (PF.12) uploads once.
3b. **Stale mips after extraction (maintainer review, DT.1).**
   `Island::extract` does NOT remip — the workspace-wide edit
   contract (`bake_bbox` documents the same; there is no Grid-level
   remip primitive). Beyond `mip_scan_dist` (default 64) both
   renderers keep drawing the extracted island from mip-N while its
   sprite twin falls next to it — and up close everything looks
   correct, so an eyeball pass won't catch it. **DT.4 must fold the
   extraction bbox into the carve worker's existing `remip_bbox`
   call** (cave-demo main.rs, the same one its carves use).
4. **Sprite model/instance leaks.** Every island is a model; shatter
   must remove both handles, and `compact_sprite_models` must run
   periodically (models are tombstoned, not freed). The 100-cycle
   leak test in DT.3 gates this.
5. **z-convention traps.** World is z-down (gravity is +z; bedrock at
   z=255). Every "bottom face" / "support" comparison must use the
   voxlap convention — the particles' `gravity: [0,0,22]` default is
   the reference.
6. **Budget false-negatives look like bugs.** A huge slab that stays
   floating after its support is shot out is *by design* (budget
   exit). Document in the demo README and make the budget a
   DebrisSystem knob so the demo can raise it.
7. **Emissive shards & bake light.** Carving through a crystal keeps
   its `BakeLight` (accepted EV simplification) — a crystal that
   crumbles away leaves its glow pool behind. Same acceptance here;
   note in the demo.
