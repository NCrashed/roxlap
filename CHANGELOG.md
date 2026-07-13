# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **wasm GPU depth-picking** (PW.1): `SceneRenderer::pick_depth` /
  `pick` now work on the browser GPU path with **one-frame latency**
  (WebGPU has no blocking readback): a call submits the readback for
  its pixel and returns the latest *completed* pick — poll again next
  frame. Clicks arriving while a readback is mapping coalesce away
  (the newest pixel wins). Backed by the new
  `roxlap_gpu::GpuRenderer::read_depth_pixel_async` and a
  unit-tested `PendingPick` state machine; the native blocking path
  is untouched, and the CPU backend stays synchronous everywhere.
  The cave web demo grew a `P` pick probe that logs the crosshair's
  world voxel to the console.

- `roxlap-scene`: `cavegen::plant_crystals` + `cavegen::CrystalParams`
  — the cave demo's EV.4 crystal planting (wall-march, glow
  `BakeLight`s, deterministic rejection sampling), extracted so the
  native and web demos grow identical crystals from the same code.
  The native demo now delegates to it.
- `roxlap-cave-web` (unpublished demo): full parity with the native
  cave demo — glowing crystals (translucent + emissive material,
  point-light bake), floating-island crumble with landing shatter
  (synchronous in-frame: no carve worker on the web), bullets moved to
  the dynamic sprite API, and — with `--features audio` —
  distance-culled crystal hums with Doppler.

### Docs

- Book: the Platforms chapter covers browser audio (the
  gesture-is-the-constructor rule), the per-backend picking-semantics
  table (wasm GPU = one-frame poll), and what the CI matrix proves;
  the Picking chapter's pre-`CharacterBody` collision advice replaced
  with a pointer to the engine controller.

### CI

- **Architecture matrix** (PW.2): new `test-macos` (Apple Silicon) and
  `test-linux-arm` (ubuntu-24.04-arm) jobs run the full test suite;
  the wasm job upgraded from `check` to `clippy` and now also gates
  the browser audio stack (`roxlap-cave-web --features audio` +
  `roxlap-audio --features kira` on wasm32). Three wasm-only pedantic
  lints that had accumulated unseen are fixed.

### Fixed

- `roxlap-cave-web`: the spawn "bubble" **inserted** a solid painted
  ball at world centre instead of carving one (the player started
  buried); carves after each impact re-baked the whole chunk
  `Directional` (which would have erased crystal glow pools) and never
  rebuilt the mip ladder — now an incremental `bake_bbox` +
  `remip_bbox` over the edit extent, like the native demo.

## [0.28.0] — 2026-07-13

### Added: per-material acoustics + Doppler (macro-stage AU2)

- **Materials change what the rays hear.** The acoustics configs take
  the same colour→material map the renderer and the debris system use
  — "this colour is crystal" is declared once and drives rendering,
  destruction and sound together:
  - `AcousticsConfig` grew `material_map` + `absorption` (material →
    effective-thickness multiplier): one voxel of `0.35` glass muffles
    like a third of a voxel of stone, via the new
    `path_thickness_weighted` (interior cells inherit the last surface
    colour along the ray — deep interiors read as rock).
  - `CavityConfig` grew `material_map` + `damping_override`: the
    cavity probe classifies every wall hit and averages per-material
    reverb damping — a glass cavern rings brighter than a stone one.
  - Empty tables (the default) keep the exact pre-AU2 arithmetic —
    bit-identical, pinned in tests. Cost with active tables measured
    at +0.8% per source query (release probe).
- **Doppler**: pure `doppler_factor` (clamped to an octave each way,
  exactly `1.0` at rest, monotone through supersonic edge cases) +
  `AudioOut::set_source_pitch` (default no-op; the kira backend tweens
  the playback rate). `DEFAULT_SPEED_OF_SOUND = 90` world units/s —
  game-tuned so demo speeds audibly bend. The cave demo's crystal hums
  now bend as you fly past.
- **The scene gallery grew an Audio tab**: a walkable stone room /
  open field / glass cavern with every acoustic parameter live in the
  HUD — occlusion per source, the reverb environment (watch the
  damping drop inside the glass cavern), Doppler factors, and the
  measured per-update acoustics cost. The numbers come from the pure
  core, so the tab works in every build; the scene demo's new `audio`
  feature adds the three zone hums.
- Book: the Audio chapter grew **"Materials: glass sounds thin"** and
  **"Doppler"** (the device-free `book_audio` example walks both).
- Public config structs grew fields (`AcousticsConfig`, `CavityConfig`,
  `CavityProbe`) — **breaking** for exhaustive struct literals /
  destructuring (constructors and `..Default::default()` unaffected);
  hence the minor bump.

### Fixed: GPU sprite-registry offset desync after remove + growth

- A `remove_sprite_model` tombstone followed by an `add_sprite_model`
  that overflowed the shared occupancy/color-offsets buffer left every
  live model **behind the hole** reading its volume at a stale offset:
  the grow path rebuilt the buffer tightly (tombstoned entries
  contribute nothing) but never rewrote the surviving entries' meta
  offsets — permanent shifted-occupancy corruption ("black stripe"
  planes), repaired per-model only when an `update_model` happened to
  rewrite that model at the stale offset. Hosts calling
  `compact_sprite_models` periodically repaired it after the fact,
  which is why the artefact surfaced downstream only after a host
  stopped compacting. The overflow path now rebuilds through the
  compactor, which recomputes the offsets it uploads (and reclaiming
  the holes can absorb the growth outright). Root-caused with a precise
  report from the **roxlap-game-demo** author — thank you. Pinned by a
  readback regression test (remove → grow → every live entry verified
  at its meta offset).

## [0.27.0] — 2026-07-13

### Added: voxel destruction — floating-island crumble (DT stage)

- The signature voxlap feature: carve away an overhang's support and
  the disconnected rock **crumbles** — it detaches, falls, and bursts
  into colour-true debris where it lands. Three engine pieces:
  - **`roxlap_scene::islands`** — `detect_islands(grid, carve_lo,
    carve_hi, budget)` finds the voxel regions a carve disconnected
    from support: a budgeted breadth-first flood over the chunk
    format's RLE **runs** (one run = one search node, no dense
    decode), 6-connected, cross-chunk and chz-stack aware. Support =
    a bedrock-anchored column bottom (format-pinned, uncarvable);
    regions past the budget stay put by design. Worst-case
    supported-exit on a 100×100 plate ≈ 0.3 ms. `Island` carries the
    voxels + colours; `extract` removes it from the grid (coalesced
    span carves + one incremental re-bake; re-mip stays the caller's
    obligation like any edit), `to_kv6`/`world_pivot` turn it into a
    sprite model posed exactly over the voxels it replaced, and
    `split(pattern, seed)` fractures it — `Chunks { cell }` into
    rounded jittered-Voronoi lumps, `Shards { plates }` into sharp
    near-parallel plates — deterministically, cost scaling with the
    island, not its bounding box.
  - **`roxlap_render::DebrisSystem`** — the crumble loop, shaped like
    `ParticleSystem` (pure `spawn_island`/`update` over the `Scene`,
    batched `sync` into dynamic sprite instances, `tick` per frame).
    Gravity + terminal clamp, a deterministic cosmetic spin, and
    tunnel-proof collision (the descent marches in half-window
    substeps, then binary-searches flush contact — a fast body cannot
    skip a one-voxel shelf on a slow frame). `set_fracture_patterns`
    installs the per-material side table; fragments spawn with an
    outward drift, and with the colour→material map installed the
    fragment models register with materials — a fallen crystal keeps
    its translucent+emissive look and **glows on the way down**.
    `drain_impacts()` reports each landing with its world-space
    `burst_sites()` for the shatter.
  - **`ParticleSystem::voxel_debris`** — the burst half of
    `carve_debris`, factored out (bit-compatibly — same-seed spawn
    sequences are identical) so a landed island's own voxels shatter
    into the same colour-true debris a bullet crater throws.
- The **cave demo** crumbles end-to-end: island detection and
  extraction run on the existing background carve worker (covered by
  the same incremental re-mip), the main thread spawns the falling
  sprites, shatters landings into particles, and routes every impact
  through the same occlusion-shaded boom as a bullet hit. Rock breaks
  into lumps (`Chunks`), crystals into glowing plates (`Shards`);
  `ROXLAP_NO_CRUMBLE=1` restores plain carves.
- Documented in the book's new **Destruction** chapter (a runnable
  windowless `book_destruction` example walks a cantilevered beam
  through detect → fracture → fall → shatter and prints each stage).

### Added: sprite emissive + headless GPU emissive gate (EV owed items)

- **Sprite emissive**: a sprite/clip voxel whose material glows now
  renders full-bright on **both** backends (previously terrain-only) —
  the emissive branch outranks the dynamic rig and the baked shade in
  the CPU sprite raycaster (opaque first-hit + translucent layers) and
  in `sprite_model_dda.wgsl` (`march_instance` +
  `march_instance_layers`), honouring per-voxel material ids and the
  per-instance tint. The GPU opaque marcher's palette fetch is gated on
  a new `has_emissive` uniform flag (a repurposed pad — no ABI growth),
  so an emissive-free palette renders byte-identically to before. No
  facade API change: the existing material sync carries the gate.
- **Headless GPU emissive test**: `HeadlessSceneRenderer` grew
  `set_terrain_materials` (the headless mirror of
  `set_scene_terrain_materials`), unblocking CI coverage of the GPU
  material branch — the new gate test proves the GPU emissive path
  matches the CPU `emissive_shade` ladder exactly and that an empty
  material map still re-renders byte-identically.

### Added / Fixed: per-grid scale follow-ups

- **Audio on scale**: `roxlap-audio`'s occlusion path-thickness now honours a
  grid's `voxel_world_size`, so a coarse grid's thicker voxels muffle sound
  proportionally more (previously identity-only).
- **Projected-size mip LOD** (GPU): a fine grid's small voxels now take a
  coarser mip sooner and a coarse grid's big voxels stay fine longer —
  matched on-screen detail, a perf win for scaled scenes.
- **Coarse-mip shadow acne**: a scaled/distant surface no longer
  self-shadows its own coarse cell (the thin dark "shell" on a chunky
  planet's hills), without over-biasing past real occluders.

## [0.26.0] — 2026-07-08

### Added: per-grid voxel scale (SC stage)

- Each grid now carries a **`voxel_world_size`** — world units per voxel,
  set via `GridTransform::at_scale`. A coarse **planet** grid (`4.0`, big
  chunky voxels) and a fine **ship** grid (`0.25`, small smooth voxels)
  coexist in one scene at their true relative sizes (a 16× ratio) without
  either changing its voxel budget. Scale is applied **only at the
  world↔grid-local boundary**, so every marcher, sampler, edit and bake
  below it is unchanged; `voxel_world_size = 1.0` (the default) is
  byte-identical to before on **both** backends.
- Threaded through the whole pipeline: `Scene::raycast` marches voxels but
  returns a **world** `t`; the CPU compose + GPU compute renderers (per-grid
  camera, cross-grid depth composite, ray-terminating scan/fog cutoffs);
  **cross-grid hard shadows** — a fine grid correctly shadows a coarse one,
  with a world-uniform sun-shadow reach; and collision, streaming radii, LOD
  tiers and the per-frame distance cull all resolve in world units.
- Snapshots **persist scale** (wire version 2); older v1 saves load as
  unscaled `1.0`, and the checked-in v1 fixture stays loadable forever.
- New scene-demo **Scale** tab (a planet + a ship casting a cross-grid sun
  shadow, on both backends) and a book **Grid scale** section.

### Breaking

- `GridTransform` and `roxlap-gpu`'s `GridWorldTransform` gain a
  `voxel_world_size` field — struct literals must add it (use
  `GridTransform::at`/`at_scale` or `..Default::default()`).
- `roxlap_core::WorldOccluder::occluded_world`'s `origin` is now `[f64; 3]`
  (was `[f32; 3]`), so a scaled grid's large world coordinates survive the
  cross-grid shadow lift.

### Fixed

- GPU cross-grid shadows on scaled grids: the shadow-ray bias now scales
  with `voxel_world_size` (a big-voxel grid no longer self-shadows its own
  surface — the "shell" on a coarse planet), and the GPU shadow step budget
  is raised so a ray crossing a fine grid's chunk reaches the occluder (a
  fine ship now casts onto a coarse planet).

## [0.25.0] — 2026-07-07

### Added: voxel-aware acoustics — `roxlap-audio` (AU stage)

- New crate `roxlap-audio`: sound that knows about the voxels. The
  acoustics **core** is pure parameter computation over a `Scene` (no
  audio device, no threads, fully unit-tested): `source_acoustics`
  casts a 9-ray fan from each source to the listener accumulating
  exact solid path *thickness* (a doorway leaks, five voxels of rock
  muffle more than one) into an occlusion gain + lowpass cutoff +
  reverb send; `CavityEstimator` casts a 32-ray golden-spiral fan
  from the listener (room size from the enclosed free path, openness
  from sky escape) into smoothed reverb feedback/mix — a cavern
  rings, open ground is dry.
- Playback is opt-in behind the `kira` feature (native; kira 0.12):
  `KiraAudio` wires one shared reverb send + a pool of spatial voices,
  each with a per-source lowpass and reverb send, applying the core's
  parameters with tweens. Off by default — a plain build/test never
  pulls in an audio backend. Demo sounds are synthesized (no binary
  assets); an `audio_probe` example walks a listener out of a sealed
  room through a doorway for a live listen.
- The **cave demo** grew optional audio (`--features audio`): plasma
  shots at the muzzle, impact booms at each carve, and a looping hum
  at every glowing crystal — all muffled by the rock between them and
  you, with reverb that swells in caverns and dries in the open. The
  crystal hums are distance-culled to the nearest few so a
  crystal-rich cave never starves the shot/boom voices. Off by
  default; the demo ships and builds silent without it.
- Documented in the book's new **Audio** chapter (occlusion + cavity
  reverb walked through a runnable device-free `book_audio` example
  that prints the computed parameters, plus the playback boundary and
  the cave-demo showcase).

### Changed: MSRV corrected to 1.92 (was a stale 1.77)

- The declared `rust-version = "1.77"` had silently rotted: egui
  0.34.x requires 1.92 (edition2024), image/vello 1.88 — the very
  first run of the new CI msrv gate caught it. The workspace now
  declares **1.92**, validated three ways kept in lockstep: the CI
  `msrv` job, `flake.nix`'s pinned `msrvToolchain`, and the new
  `msrv-check` dev-shell command that runs the CI check locally.

### Changed: release-gate quality pass (tests, CI, book)

- `roxlap-cli` test coverage 6 → 11: every subcommand's file pipeline
  now round-trips in tests (multi-model `vox2kv6` numbered-output
  contract, `vox2rvc`/`kv62vox` through the filesystem, `gif2rvc`
  thickness + centisecond timing, `png2rvc` sequence + still,
  `info` success/error reporting), with in-memory GIF/PNG fixtures.
- CI grew three gates: **MSRV** (`cargo check` on the declared 1.77 —
  previously unvalidated), **docs** (`cargo doc --no-deps` with
  `-D warnings`, catching broken intra-doc links before docs.rs
  does), and **smoke-fuzz** (all 8 `roxlap-formats` fuzz targets,
  30 s each — corpus replay + short random walk on every PR;
  previously local-only). It earned its keep immediately, hardening
  the `.kvx` parser against two crafted-header bugs a hostile file
  could trigger: an OOM from a dimension-sized `Vec::with_capacity`
  before any bounds check, and an integer overflow in the very budget
  check that fixes it (`xsiz·ysiz·2` past `u64` — now computed in
  `u128`). Both are bounded before allocation and pinned by tests.
- Book caught up to 0.24.0: the GPU occupancy pyramid in the
  Rendering chapter, `BillboardActorDef::scale` + the `.rkc`
  lifecycle calls in Sprites, and a **Troubleshooting** section in
  Platforms (logger-first diagnosis, NixOS `libvulkan`,
  `ROXLAP_GPU_POWER` for hybrid GPUs, SDL2 linking, web nightly pin,
  silent thread-pool failure). `PORTING-TRANSPARENCY.md` now points
  to the book as the canonical user-facing doc.

### Added: emissive voxels + glowing cave crystals (EV stage)

- `Material` grew an `emissive: u8` field (**breaking** for struct
  literals; every constructor defaults it to `0`): a non-zero value
  renders the voxel at `albedo × ((128 + (emissive >> 1)) / 128)` —
  from 1× up to ~2× over-bright at 255 — skipping the baked
  brightness byte, per-face side shades, the dynamic light rig,
  shadows and cel bands (fog still applies). Orthogonal to the blend
  mode: `Material::alpha_blend(a).with_emissive(e)` is a translucent
  gem, `Material::glow(e)` the opaque shorthand;
  `MaterialTable::any_emissive` is the new gate. Identical on both
  backends (the CPU hit path hoists its one-per-hit material lookup
  above shading; the GPU palettes grew to a 16-byte stride and the
  scene shader mirrors the CPU branch order). With no emissive
  material defined every path is bit-identical to before. Terrain
  only for now — the sprite palette carries the field but the sprite
  pass doesn't render it yet (see `PORTING-EMISSIVE.md`).
- `BakeMode::PointLights` + `BakeLight` + `Grid::bake_lights` (EV.3):
  voxlap's lightmode-2 point-light bake — a dim directional base plus
  a cube-law Lambertian pool around every registered light, written
  into the brightness bytes (free at render time, both backends).
  `Grid::bake_bbox` picks the grid's lights up automatically, so
  incremental carve relights keep their glow pools (byte-identical to
  a full re-bake, tested).
- The cave demo grew **glowing crystals**: clusters planted on cavity
  walls per preset (icy cyan / warm amber), each an emissive
  translucent blob plus a `BakeLight`; the whole cave switched from
  the flat directional bake to `PointLights` gloom, and the
  background carve worker now relights craters with the crystal
  lights riding along. One crystal is guaranteed at the spawn bubble.
- Book: "Emissive voxels & baked glow" section in the Lighting
  chapter, with anchored snippets from `book_lighting` (which gained
  an emissive crystal on its monument) and the cave demo's crystal
  recipe.

## [0.24.0] — 2026-07-07

### Added: roxlap-cli snapshot chunk extraction

- `roxlap-cli chunks <snapshot> <grid>` lists a snapshot grid's
  resident chunks with their edit versions, and
  `extract <snapshot> <grid> <chx,chy,chz> <out.(vxl|kv6|vox)>` pulls
  one chunk out — as a raw `.vxl` world, a `.kv6` model, or a
  MagicaVoxel `.vox` (a player-carved chunk opens straight in the
  editor). The voxlap bedrock placeholder plane (chunk-local
  z = 255) is dropped from `.kv6`/`.vox` — bookkeeping, not
  content — and kept in byte-faithful `.vxl`.

### Added: GPU.13.1 — hierarchical empty-space skip (chunk-occupancy pyramid)

- The outer scene DDA (and the sun-shadow march) now climbs a tiny
  per-grid occupancy pyramid (levels of 2^L slot-blocks over the
  modular pool, < 40 B/grid) when it meets an empty chunk, and
  crosses the whole provably-empty block with read-free incremental
  steps — one `max_outer_steps` budget unit per block instead of two
  storage reads per chunk. Occupied chunks are still entered through
  bit-identical t sums, so render output is unchanged by
  construction. Maintained live on refresh/evict (ancestor re-OR).
- Measured (NVK, 960×540): rays crossing a 15-chunk empty gap into a
  wall go 1.86 → 1.30 ms/frame (−30%); dense or sky-dominated views
  are neutral within noise. The lever GPU.13.0 deferred — it pays in
  large sparse worlds where empty space sits INSIDE the occupied
  AABB.

### Fixed: billboard actors froze when the camera entered their column

- `billboard_transform` returned `None` for a degenerate view axis
  (camera exactly overhead / at the actor's xy column), and the tick
  dropped the WHOLE transform — including the position update — so
  the actor froze at its last pose. It now falls back to a fixed
  facing (the card is edge-on at that instant anyway) and the
  position always lands. Found via the World third-person view,
  whose camera ticked from the eye — inside the actor's own column —
  every frame.

### Changed: `BillboardActorDef::scale` (breaking)

- New field: world units per slab voxel (`1.0` = the classic 1:1),
  applied to the instance basis every tick — identical on both
  backends. Composes with the clip's `voxel_world_size`.
  Construction sites add `scale: 1.0`.

### Fixed: CPU backend now honours clip `voxel_world_size`

- The GPU clip volumes always scaled voxels by the clip's
  `voxel_world_size`; the CPU sprite pipeline silently rendered
  clips 1:1 — a scaled clip drew at different sizes per backend.
  `SpriteDense` now carries the scale and the draw + shadow-occluder
  entries apply it (`basis · ((v − pivot) · vws)` refactored as a
  basis scale, so `1.0` stays byte-identical — golden hashes held).
  Parity pinned by a test: `vws = 2` at a unit basis renders pixel-
  identically to `vws = 1` at a doubled basis, and the occluder
  agrees.
- Demo: the World third-person figure is upright (slab z = 0 is the
  image BOTTOM — the figure was authored head-at-0), draws at 0.03
  scale (0.72 units — a marker, not a giant), and the camera boom now
  applies BEFORE the actor tick so the billboard faces the real
  boomed camera.

## [0.23.0] — 2026-07-06

### Added: character controller — stage CC (CC.0–CC.5)

A walking body over `Scene`, engine-owned and headless (design
history: `docs/porting/PORTING-CONTROLLER.md`; book: "The scene
graph" → *Walking on the world*, with the runnable
`book_controller` example):

- `roxlap_scene::collide` — the query layer: `box_overlaps_solid` /
  `point_overlaps_solid` / single-grid `grid_box_overlaps_solid`.
  Axis-aligned grids probe cell-exactly, rotated grids conservatively
  (corner-AABB). `Solidity` policy: `bedrock_blocks` (match your
  renderer or get invisible walls / fall-through floors) and
  `passable` — a colour veto (`fn(VoxColor) -> bool`) that lets
  approved *visible* voxels pass (water, foliage). Two format facts,
  pinned by tests: pass-through walls work up to 2 voxels thick, and
  must sit ≥ 1 voxel inside the chunk (edge voxels lose their side
  colours).
- `roxlap_scene::character` — `CharacterBody` / `CharacterDef` /
  `WalkInput` / `MoveMode::{Walk, Fly, Noclip}`: substepped per-axis
  move-and-slide (flush contact, no tunneling at any speed), gravity
  with a `max_fall_speed` terminal clamp, press-buffered +
  coyote-timed jumps, auto step-up onto ledges, `on_ground()` /
  `hit_head()`, the demos' stuck-escape rule, `def_mut()` for runtime
  tuning (sprint) and `set_pos()` (reposition keeping velocity).
  Deterministic: same scene + same inputs = bit-identical trajectory.
  Fly mode is the demos' camera: full-3D wish, instant start/stop
  (`fly_accel`), sliding collision; call `walk()` every frame — the
  zero wish is what stops the body.
- Demos: the three copy-pasted fly-collision hacks (scene-demo,
  cave-demo, cave-web) are DELETED — every host moves on
  `CharacterBody`. World gains `G` (walk mode: gravity, Space jumps,
  step-up) and `C` (third person: a synthetic billboard-actor figure
  at the body, raycast-clamped camera boom); the Transparency scene's
  front wall is now half glass / half water and flies with collision —
  the water half lets you through.
- Perf (release, `stress_probe`): a wall-hugging walker costs
  ~0.7 µs/frame; a 100-NPC crowd ~65 µs/frame total. No caching
  needed.
- Fixed along the way (wasm blind spot): the web crates are wasm-only
  (`#![cfg]`), so native CI never type-checked them — the colour
  newtypes had silently broken both, and `roxlap-scene` didn't build
  on plain wasm32 at all (unconditional rayon in cfg'd-out code).
  Both fixed; CI gained a stable-wasm32 `cargo check` job for the web
  crates.

### Added: `.vox` export + CLI importer subcommands

- `roxlap_formats::vox` learned to WRITE: `VoxFile::from_kv6` +
  `vox::serialize` — kv6 sprites export to MagicaVoxel files (z-down
  flips back to z-up; exact for ≤ 255 distinct colours, 6-bit-bucket
  frequency quantisation beyond; the kv6 pivot is lost — `.vox` has
  no pivot). Round-trip pinned by tests.
- `roxlap-cli` grew three subcommands: `kv62vox <in.kv6> <out.vox>`,
  `gif2rvc <in.gif> <out.rvc> [thickness]` and
  `png2rvc <in.(a)png> <out.rvc> [thickness]` (or
  `png2rvc <f0.png> <f1.png> … <out.rvc>` for numbered sequences) —
  the BB-stage billboard importers, now reachable without writing
  Rust.

### Changed: env overrides consolidated (QE-C6)

- All `ROXLAP_*` environment overrides the library recognises are now
  read in ONE place, once, at `SceneRenderer` construction — the
  variable table lives on `RenderOptions`' rustdoc. Semantics are
  unchanged (env wins over the programmatic value; never read per
  frame), with one improvement: an unparseable value now logs a
  `log::warn!` instead of being silently ignored.
- `roxlap-gpu` itself no longer reads the environment:
  `ROXLAP_GPU_POWER` is resolved by the facade into
  `GpuRendererSettings::power_preference`. Only affects code driving
  `roxlap_gpu::GpuRenderer` directly (not via `roxlap-render`) — set
  `power_preference` yourself there.
- `RenderOptions` is now `Clone`.

### Added: workspace-wide `missing_docs` (QE.0 debt paid)

- The ~260 undocumented public items left in `roxlap-formats` /
  `roxlap-core` / `roxlap-gpu` (mostly voxlap-parity struct fields
  and GPU-mirror layouts) now carry real rustdoc — meaning, units,
  packing, sentinels — written from their read sites (WGSL shaders
  included), not from their names.
- `missing_docs = "warn"` moved from per-crate attributes into
  `[workspace.lints.rust]`: with CI's `-D warnings`, every new public
  item in any library crate must be documented to build.

### Changed: the colour family — `VoxColor` / `Rgb` / `OverlayColor` newtypes (QE-B6, breaking)

- Every public colour parameter that used to be a bare `u32` (or
  `i32`) with an implicit packing is now one of three
  `#[repr(transparent)]` newtypes, so mixing the packings is a
  compile error instead of an over-bright voxel or an invisible line:
  - **`VoxColor`** — voxel colours: RGB + a *brightness* byte (not
    alpha; `0x80` = neutral). Edits, colfuncs, `voxel_color`,
    `RayHit::color`, KV6 builders, debris sampling.
  - **`Rgb`** — plain colour: sprite/particle/actor tints, sky / fog /
    clear colours, colour→material map keys.
  - **`OverlayColor`** — real alpha; the overlay APIs only
    (`Line3.color`, image-sprite tints).
- All three re-export from `roxlap-formats`, `roxlap-scene` and
  `roxlap-render`; the wire word is the public `.0` field, so file
  formats, GPU buffers and golden hashes are byte-identical.
- Migration (mechanical):

  | was | now |
  |---|---|
  | `set_sphere(.., Some(0x80_4d_8a_3a))` | `set_sphere(.., Some(VoxColor::rgb(0x4d, 0x8a, 0x3a)))` |
  | colfunc `\|x,y,z\| 0x80_....u32` | colfunc `\|x,y,z\| VoxColor(0x80_....)` |
  | `params.sky_color = 0x0099_b3d9` | `params.sky_color = Rgb(0x0099_b3d9)` |
  | `set_actor_tint(id, 0x00ff_8040)` | `set_actor_tint(id, Rgb::new(0xff, 0x80, 0x40))` |
  | `Line3 { color: 0x80ff_d040, .. }` | `Line3 { color: OverlayColor::rgba(0xff, 0xd0, 0x40, 0x80), .. }` |
  | material map `&[(0x00ff_ffff, 2)]` | `&[(Rgb(0x00ff_ffff), 2)]` |

  When you genuinely need the raw word (framebuffer comparisons,
  custom wire formats), take `.0` / wrap with the tuple constructor.

### Added: `Frame` guard + `clear_sprites` (QE-B6)

- `SceneRenderer::frame(scene, camera, params)` returns a `Frame`
  guard — the type-state form of the `render → overlays →
  present`/`paint_egui` protocol. Overlays draw with the camera the
  frame was rendered with (the guard holds it), presenting consumes
  the guard (double present can't compile), and dropping an
  unfinished frame presents it (forgetting can't happen). **Purely
  additive**: the split `render`/`present` calls remain for hosts
  that need custom control flow between the stages.
- `SceneRenderer::clear_sprites()` — the explicit scene-switch verb
  for "drop every model/instance/clip/character/billboard and stale
  every handle", replacing the "register an empty `SpriteSet`" idiom.
  `set_sprites`' docs now state its replace-the-world semantics up
  front.

### Changed: typed lighting bakes — `Grid::bake(BakeMode)` (QE-B6)

- The voxlap magic-`u32` lightmode is gone from the public bake API:

  | was | now |
  |---|---|
  | `grid.bake_lightmode(1)` | `grid.bake(BakeMode::Directional)` |
  | `grid.bake_lightmode_with_ao(3, ao)` | `grid.bake(BakeMode::AmbientOcclusion(ao))` |
  | `grid.bake_lightmode_bbox(lo, hi, 1)` | `grid.bake_bbox(lo, hi, BakeMode::Directional)` |

  The old methods remain as `#[deprecated]` forwarders for one minor
  release. `BakeMode` and `AoParams` re-export from `roxlap_scene`.
- Fix folded in: `bake_lightmode_bbox` silently baked AO with
  *default* params; `bake_bbox(.., BakeMode::AmbientOcclusion(ao))`
  honours the ones you pass.

### Changed: `ImageId` is generational (QE-B6)

- Image slots are reused after `drop_image`, and `ImageId` was a bare
  positional index — a stale handle silently aliased whatever texture
  re-took its slot (the one handle family with that hazard left).
  `ImageId` now carries the slot's generation: stale handles resolve
  to safe no-ops in `draw_images` / `pick_image` / `drop_image`.
- Migration: nothing to change for code that treats `ImageId` as
  opaque (the intended use). Code that constructed or compared raw
  indices no longer compiles — hold the handle `upload_image`
  returned instead.

### Changed: GPU sprite mip-LOD default (visual parity with CPU)

- The GPU sprite pass stepped to coarser sprite mips once a mip-0
  voxel projected below **4** screen pixels; the CPU backend has no
  sprite LOD at all. Downsampling collapses thin/hollow structure —
  most visibly, translucent models: a hollow glass model's front/back
  sheets merge into a solid volume, so glass read *denser/paler* on
  GPU at moderate distances (the "window looks different per backend"
  report). The default is now **1.0** (mip only once a voxel goes
  sub-pixel), which is pixel-identical to CPU in the A/B probe.
- New knob: `RenderOptions::gpu_sprite_lod_px` (env override
  `ROXLAP_GPU_SPRITE_LOD_PX`). **Old behaviour:** set it to `4.0` —
  worth trying if distant-sprite cost matters more than fidelity.

### Fixed: GPU sky panorama orientation

- The GPU sky sampler used `atan2(dir.x, dir.y)` where the CPU
  renderer uses `atan2(dir.y, dir.x)` — the swapped arguments rotate
  the panorama 90° and mirror the heading, so the two backends showed
  *different panorama content* at the same camera (hills on CPU where
  GPU showed city). Argument order now matches the CPU; panorama
  content aligns across backends (verified on live captures).
  Residual: the GPU samples with linear filtering vs the CPU's
  nearest-texel — sub-texel softness only.

### Internal: WGSL shared-snippet extraction (QE.8)

- The five helpers duplicated (with drift) between `scene_dda.wgsl`
  and `sprite_model_dda.wgsl` — `shield_parallel`, `apply_fog`,
  `point_falloff`, `spot_cone`, `cel_band` (+ `T_INF`) — now live once
  in `shaders/common.wgsl`, prepended at shader assembly. The raw
  `.wgsl` base files are no longer standalone-valid; naga-validation
  covers all four assembled variants. No render change (verified
  within run-to-run noise on live GPU captures; device tests green).

### Security / robustness (fuzzing)

- New `cargo-fuzz` harnesses (`crates/roxlap-formats/fuzz`, 8 targets:
  every parser + the scene-snapshot envelope) found two
  crafted-input bugs, both fixed:
  - `roxlap_formats::vxl::parse` — an unvalidated `vsid` reserved a
    `vsid² + 1`-entry offset table before reading any column data
    (capacity-overflow panic / multi-GB allocation). Rejected as the
    new `ParseError::BadVsid` (every column needs ≥ 4 bytes, so
    `vsid²` is bounded by the file size).
  - `roxlap_formats::kfa::parse` — zero hinges + a crafted frame
    count looped/allocated unboundedly (frame rows are 2 bytes ×
    hinges, i.e. zero bytes — no read could fail). Rejected as the
    new `ParseError::FramesWithoutHinges`.
  - Migration: both `ParseError` enums gained a variant — exhaustive
    `match`es on them need one new arm. Files rejected by the new
    checks were never loadable (they crashed the process).

### Fixed

- `SceneRenderer::supports` reported `TranslucentSpriteMaterials` and
  `TranslucentTerrain` as unsupported on the GPU backend. Both GPU
  paths have existed (and been visually verified) since the TV stage
  — TV.3 per-voxel sprite materials, TV.6 terrain accumulation; the
  QE.7a parity table shipped stale. `supports` now returns `true` for
  both on either backend; the `Feature` rustdoc table and the book's
  lighting chapter are corrected to match. No behavioural change to
  rendering itself — only the capability probe (hosts that branched
  on it will now take their translucent path on GPU, which is what
  they asked for).

## [0.22.0] — 2026-07-05

Stage **PS** — the particle system (`roxlap_render::ParticleSystem`
over dynamic sprite instances), PS.0–PS.5, purely additive. Plus the
"Particles" demo tab with water/lighting/crosshair polish and the
click-perf investigation probes.

### Changed: scene-demo "Particles" polish (water, lights, crosshair)

- The fountain is **water** now: droplets carry an alpha-blend
  `MAT_WATER` (150), and a translucent **pool** sits under it — grid
  voxels in a unique water colour mapped via `set_terrain_materials`
  (TV.5/TV.6 terrain transparency), inside a recoloured pad rim.
- **Runtime lighting**: warm shadow-casting sun + a cool fill over the
  pool + a warm accent by the smoke column, over a freshly baked
  ambient byte (`bake_lightmode(1)`); every explosion adds a brief
  orange point-light **flash** that burns down over 0.45 s.
- **Perf probes** (env-gated, zero cost when off):
  `ROXLAP_AUTOFIRE=1` fires a scripted explosion every 4 s at fresh
  floor spots and logs per-second frame stats (`sec: avg/max/frames`)
  plus >40 ms spikes — the click hitch, measurable headless;
  `ROXLAP_NOFLASH=1` disables the explosion light flash (the A/B for
  its per-pixel light-loop cost). Probe findings on the dev box
  (Intel Xe, 430×260): the carve + chunk refresh cause **no**
  measurable hitch (fresh crater per shot, no frame > 40 ms); the
  one-off ~74 ms frame is app warm-up (frame 2, fires with or without
  explosions); the apparent post-click elevation is the **fountain
  filling to its ~380-droplet steady state** over its first ~3 s, not
  the explosion.
- **Explosion load halved**: sparks 50 → 24 per click, debris capped
  at 32 via the new **`ParticleSystem::set_carve_debris_cap`** — a
  per-system knob over the `CARVE_DEBRIS_CAP` (96) default, clamped
  ≥ 1, so hosts can tune per-explosion cost without touching the
  crater size.
- **Particles are lit** now: the fountain droplets use the default
  `FaceNormal` lighting (they pick up the sun, the pool fill and the
  explosion flashes) and the smoke uses `WorldUp` (stable sun shading
  the warm accent tints). Only the sparks stay `FullBright` — they are
  emissive by design. The earlier defs had opted the two big effects
  out of lighting (`FullBright`/`AmbientOnly`), which read as
  "particles ignore the rig".
- **Crosshair + centre-aim**: four world-space `Line3` ticks (no depth
  test, FOV-stable size) mark the screen centre, and the left-click
  pick now goes through the **centre** — FPS-style, since mouse-look
  owns the pointer — instead of the invisible cursor position.
- Fix uncovered by the pool work: `set_rect(Some(colour))` over an
  already-solid region merges spans but keeps the old colours, so the
  PS.4 "landing pad" recolour was silently invisible. The demo now
  carves-then-inserts to recolour (an engine-side recolour story is a
  separate question).

### Added (PS.5): debris-from-carve helper — stage PS closes

- **`ParticleSystem::carve_debris(scene, grid, centre, radius,
  outward, &def)`** — the "shoot the wall, the wall's colours fly off"
  effect as one call: samples the solid voxel colours inside the ball,
  `set_sphere(…, None)` carves it, then bursts one debris particle per
  sampled voxel — positioned at its voxel's world centre
  (transform-correct for rotated grids), **tinted with its own
  colour** (a `tint_end` lerp starts from that colour), and kicked
  radially away from the crater at a speed from `outward` on top of
  the def's velocity terms. Big carves stride-sample an even subset
  down to `CARVE_DEBRIS_CAP` (96) so one explosion can't monopolise
  the pool; the budget applies on top and counts drops. The demo
  scene's explosion now uses it.
- **Perf verdict (10k-particle stress, release)**: `update` + `sync`
  with worst-case per-frame alpha *and* tint churn = **~225 µs/frame**
  (~1.4 % of a 60 FPS frame). The PS.1 "alpha/tint have no batch API"
  hazard closes as *not needed*: each setter is a CPU-side vec write
  and the GPU instance upload is already coalesced once per frame.
  A `#[ignore]`d probe test (`stress_10k_probe`) reproduces the
  measurement; a threshold-free 10k stress test guards behaviour.

### Added (PS.4): scene-demo "Particles" tab

- New demo scene between Spotlight and Doom: a bouncing **fountain**
  (cone emission + gravity + `Bounce` off the arena floor), a buoyant
  **smoke column** (sphere-shaped emitter, alpha-blend material,
  grow + spin + fade-in/out + tint-to-dark over life), and
  **left-click explosions** — `pick` → `set_sphere(None)` crater +
  additive full-bright sparks (`Kill` on contact) + tumbling debris
  (`Bounce`, spin, shrink). The whole per-frame protocol is one
  `ParticleSystem::tick_with_scene(renderer, dt, &scene)`; HUD shows
  live/dropped counts. Verified live on both backends.
- scene-demo: **`ROXLAP_SCENE=<name>`** — start on a scene by its
  (case-insensitive) menu name instead of scene 0.

### Added (PS.3): particle voxel collision

- **`CollisionMode`** (`ParticleEmitterDef::collision`) — `None`
  (default, pass-through) / `Kill` (impact sparks, raindrops) /
  `Bounce { restitution }` (reflects the velocity on the axes whose
  voxel boundary was crossed this step, then scales it by
  `restitution` — deliberately arcade).
- **`ParticleSystem::update_with_scene(dt, &Scene)`** and
  **`tick_with_scene(renderer, dt, &Scene)`** — the colliding
  variants; the scene-free `update`/`tick` never collide. The test is
  a point sample of each post-step position nudged half a voxel along
  the velocity (`Scene::resolve_voxel` reused), so contact registers
  slightly early, fast particles can tunnel through sub-step-thin
  walls, and zero-velocity particles never re-collide — documented
  effect-grade physics, not a physics engine.

### Added (PS.2): full emitter palette

The effect-design layer on the PS.0/PS.1 base (all additive; the
PS.0 API was still unreleased, so its two touched fields changed in
place — `fade_frac` is now `fade_out_frac`, and `VelocityDef` gained a
`cone` field / lost `Copy`):

- **`EmitterShape`** (`ParticleEmitterDef::shape`) — spawn-position
  distribution: `Point` (default) / `Sphere { radius }` (uniform in
  the ball) / `Box { half }`.
- **`ConeDef`** (`VelocityDef::cone`) — directional emission: a random
  direction uniform on the spherical cap within `half_angle_deg`
  (degrees, the `SpotLight` convention) of `axis`, at a `speed` range
  along it; composes additively with `base` + isotropic `spread`.
  Fountains, muzzle flashes, impact sprays.
- **`spin: Range<f32>`** — per-particle yaw rate about the world
  vertical (rad/s, e.g. `-3.0..3.0` for tumbling debris); the rendered
  basis rotates accordingly.
- **Over-life curves** — `scale_end: Option<f32>` (growing smoke,
  shrinking sparks), `tint_end: Option<u32>` (per-channel lerp,
  white-hot → ember), and `fade_in_frac` (alpha ramp at birth;
  overlapping in/out windows take the darker). Lerping tints ride the
  same change-only per-instance sync writes as alpha.

The facade binding for the PS.0 core; with it the particle system is
usable end to end:

- **`ParticleSystem::sync(&mut SceneRenderer)`** — despawns the dead,
  spawns newborns *pre-posed* (no one-frame axis-aligned flash) with
  their one-time material/lighting/shadow/tint setup, moves everything
  else through one `set_sprite_instance_transforms` batch, and writes
  per-instance alpha only when the fade value actually changed.
  **`tick(&mut renderer, dt)`** = `update` + `sync`, mirroring the
  facade's own `tick` naming. A spawn against a stale
  `SpriteModelId` kills the particle and counts it
  (`stale_model_kills()`).
- **GPU cull fix (internal)** — the sprite cull sphere and the LOD
  pick assumed a unit basis; instances scaled **up** under-culled and
  could pop at screen edges. `SpriteInstanceTransform` now carries
  `max_scale` (longest basis column, in the former pad slot — GPU
  layout unchanged), and the cull radius = model bound radius ×
  `max_scale`, maintained across `upload`/`append_instances`/
  `update_transforms`/`set_instance_model`. The LOD pick scales the
  projected voxel size the same way, so a 2× instance holds its fine
  mip proportionally longer. Unit-basis sprites are byte-identical.
- **CPU scaled-basis parity verified** — new `dda_sprite` test: the
  same cube at 2×/0.5× basis covers ~4×/~0.25× the pixels, the
  contract particle scale-over-life (PS.2) will rely on.

### Added (PS.0): particle-system core — `roxlap_render::ParticleSystem`

Stage PS opens (`docs/porting/PORTING-PARTICLES.md`): a host-owned
particle layer built on the facade's dynamic sprite instances. PS.0 is
the renderer-free simulation half — purely additive, no facade methods
touched:

- **`ParticleSystem`** — emitters + a budgeted particle pool with a
  deterministic seeded RNG (in-module PCG32, no new dependency): same
  seed + same `dt` sequence ⇒ bit-identical simulation.
- **`ParticleEmitterDef`** (construct via `::new(model)` — no
  `Default`, a def needs a live `SpriteModelId`): position,
  `SpawnMode::{Rate, Burst, Manual}`, lifetime range, base velocity +
  isotropic spread, gravity (+z is DOWN — gravity is positive z),
  linear drag, uniform scale, trailing alpha fade, tint, TV material,
  `BillboardLighting`, `ShadowFlags` (default **off** for particles).
- **`EmitterId`** — epoch-generational like every other handle family;
  stale handles are safe no-ops. `remove_emitter` retires: spawning
  stops, in-flight particles live out their lifetimes.
- **Budget** — `set_max_particles` (default 4096,
  `DEFAULT_MAX_PARTICLES`); when full, spawns are *dropped* (never
  evicts live particles) and `dropped_spawns()` counts them — no
  silent cap.
- `update(dt)` — semi-implicit Euler + age/fade, no renderer, no
  window: unit-testable. The facade binding (`sync`) lands in PS.1;
  until then `drain_dead_instances()` already exposes the ids a
  custom renderer must free.

### Changed — internal (QE.8): roxlap-gpu de-monolithed

- **Dead pipelines deleted** — `GpuRenderer::render_chunk` /
  `render_grid` (the GPU.0/GPU.3 single-chunk and single-grid
  baselines, superseded by the scene marcher since GPU.5) plus their
  resources, uniforms, `GpuGridResident`, and the `chunk_dda.wgsl` /
  `grid_dda.wgsl` shaders — ~900 lines with **zero callers** outside
  their own definitions. `GridUpload` / `bounding_box_of` (used by the
  live scene-upload path) stay. If you called these directly:
  `render_scene` with a one-grid `SceneUpload` is the replacement.
- **`lib.rs` split by pass** (5.7k → ~3.5k lines): `overlay.rs`
  (deferred lines + image quads + egui paint), `lights.rs` (dynamic
  light packing/upload), `readback.rs` (blocking depth/colour
  readbacks + unproject), `shader_src.rs` (shader-source splicing +
  the naga validation test). Pure code moves; public API re-exported
  unchanged from the crate root.
- **`FrameDirty`** — the three loose cross-frame booleans
  (`scene_lights_dirty`, `sprite_lights_dirty`, `scene_depth_valid`)
  grouped into one struct whose lifecycle rules (notably
  "sprite-lights is cleared only by the conditional sprite pass") are
  documented on the fields they guard, closing the QE review's last
  "discipline-only invariant".
- WGSL shared-snippet extraction (sky/occupancy/light-loop copies
  across `scene_dda` / `sprite_model_dda`) is deliberately **not**
  done: the copies are structurally similar but not byte-identical
  (different camera sources / variable names), so merging means
  parameterising shader code — a change that needs render-output
  verification on a real GPU, not just the naga validation available
  in CI. Tracked as owed in `docs/porting/PORTING-QUALITY.md`,
  best done alongside the TV.3b/TV.6 shader work.

### Added — QE.7a: GPU frame capture + capability probing

- **Screenshots now work on the GPU backend** —
  `SceneRenderer::request_capture` / `take_capture` implement a
  blocking colour readback of the most recent frame at the logical
  resolution (post-SSAA/posterize, pre-upscale), closing the biggest
  backend-parity gap (photo modes, bug reports, golden tests ran
  CPU-only before). Hotkey-grade cost, like `pick_depth`; returns
  `None` on the wasm GPU path (WebGPU can't block).
- **`SceneRenderer::supports(Feature)`** — the queryable form of the
  CPU/GPU parity table (capture, sky panorama, sprite carve,
  translucent sprite/terrain materials, free-vs-blocking pick), which
  previously lived in scattered doc sentences and tribal knowledge.
  The table itself is on the `Feature` docs.

### Changed — **breaking** (QE.7b): API wart batch

| Was | Now | Migration |
|---|---|---|
| `RenderOptions.want_gpu: bool` | `backend: BackendPreference` | `true` → `PreferGpu`, `false` → `Cpu`; **new**: `RequireGpu` fails construction (`RenderError::GpuInit`) instead of silently software-rendering — for CI/benchmark rigs |
| `speed_q8: i32` (6 signatures + `BillboardActorDef.speed_q8`) | `speed: f32` (`1.0` = authored rate) | `speed = speed_q8 as f32 / 256.0`; clip clocks keep Q8 internally, `.rkc`'s on-disk `ClipPlayback.speed_q8` is unchanged (wire format) |
| `set_sprite_instance_shadow_flags(id, true, false)` | takes `ShadowFlags { casts, receives }` | wrap the bools; `ShadowFlags::default()` = both on (the spawn default). `BillboardActorDef.{casts_shadow, receives_shadow}` merged into `shadows: ShadowFlags` |
| `get_clip_instance_frame` | `clip_instance_frame` | the only `get_` prefix in the crate; deprecated forwarding shim kept for one minor release |
| `ActorState.name: &'static str` | `String` | `.to_owned()` at literals; actor definitions can now come from data files without `Box::leak` |

Still deferred (tracked in `docs/porting/PORTING-QUALITY.md`): the
`PackedColor` newtype family, generational `ImageId`, splitting
`set_sprites`' all-family reset, collapsing the `_with_materials`
method variants, a `Frame` guard for the render/present protocol, and
a typed `BakeMode` for `Grid::bake_lightmode`.

### Added — QE.6: MagicaVoxel import + hostile-input hardening

- **MagicaVoxel `.vox` importer** (`roxlap_formats::vox`, QE.6a) —
  the bridge from the industry-standard voxel editor:
  `vox::parse(bytes)` → models + palette,
  `VoxFile::to_kv6_models()` → ready-to-register KV6 sprite models
  (`add_sprite_model` / `SpriteSet`). Hand-parsed in the house style
  (shared cursor, typed `ParseError`, hardened by construction — a
  crafted file errors, never panics/hangs/allocation-bombs); **no new
  dependencies**. Covers `SIZE`/`XYZI`/`RGBA` (multi-model files
  yield models in file order); the scene graph / materials are
  skipped for now. z-up → z-down mapped so editor-upright models stay
  upright; palette → voxlap-packed `0x80RRGGBB`; files without a
  palette get the official default.
- **Parser hardening sweep** (QE.6b) — adversarial-input fixes across
  every format, each pinned by a new adversarial test:
  - **Allocation bombs defused**: every length-prefixed
    `Vec::with_capacity` in `.kv6` / `.kfa` / `.rkc` / `.rvc` is now
    clamped by the bytes actually remaining (a 36-byte file claiming
    `u32::MAX` voxels used to attempt a ~32 GiB allocation — process
    abort — before the reads could fail; now it errors `Truncated`).
  - **`.kfa`/`.rkc` skeleton validation at parse**: an out-of-range
    hinge/bone `parent` used to panic out-of-bounds later, and
    **cyclic parents hung the loader forever** (denial-of-service on
    load); both are now `ParseError`s (`BadHingeParent`/`HingeCycle`,
    `BadSkeleton`).
  - **`.rkc` cross-reference validation**: an attachment mesh/clip
    index past its table is now `BadAttachmentIndex` at parse (the
    doc contract "parse keeps indices in range" was previously
    claimed but unchecked — the runtime panicked instead).
  - **`.rvc` dimension cap**: META dims are validated (≤ 4096 per
    axis, non-zero) before anything derives an allocation — a crafted
    header could previously drive huge allocs or a usize overflow on
    wasm32.
  - **GIF/PNG importer caps default on**: `max_dims` now defaults to
    `Some([4096, 4096, 4096])` (a hostile GIF can declare a ~16 GiB
    canvas). Migration: set `max_dims: None` explicitly for truly
    unbounded imports; there is still no silent downscale.
  - `cargo-fuzz` harnesses for the `&[u8] → Result` parsers remain
    owed (tooling unavailable in the dev environment); the adversarial
    classes found by review are covered by deterministic tests.

### Added — QE.5: streamed-edit persistence + versioned save files

- **`ChunkStore`** (roxlap-scene, QE.5a) — the persistence hook that
  makes edit-the-world games possible on streamed grids. Previously,
  walking away from an edited chunk and coming back **silently
  reverted the player's edits** to generator output (evict +
  deterministic re-generate). With a store attached
  (`Grid::set_chunk_store`): the eviction pass hands every edited
  chunk (`chunk_version != 0`) to `ChunkStore::store` before dropping
  it, and stream-in consults `ChunkStore::load` **before** the
  generator — a stored chunk restores with its persisted edit version,
  and wins even where `should_generate` declines the index. Pristine
  chunks (version 0) are regenerable and skip the store. Under
  `Scene::pump_streaming` the loads run on the background pool
  (blocking IO is fine there); `store` runs inline during eviction —
  keep it cheap. A store alone (no generator) also works. Default
  (`None`) keeps the old behaviour.
- **Versioned snapshot wire format** (QE.5b) —
  `Scene::save_snapshot() -> Vec<u8>` / `Scene::load_snapshot(bytes)`:
  magic `RXSS` + little-endian `u32` version + bincode payload.
  Loading dispatches on the version, so an old save either loads
  correctly or fails loudly (`SnapshotLoadError::UnsupportedVersion`)
  — never the silent misparse that bare positional bincode gave. A
  checked-in v1 fixture test freezes backward compatibility.
  Migration: saves written before QE.5b (bare bincode, no envelope)
  fail with `BadMagic` — decode them with the engine version that
  wrote them and re-save. `Scene::to_snapshot`'s plain serde value
  stays available for custom codecs.
- **Snapshots now carry grid configuration** (QE.5b) —
  `GridSnapshot` gained `name`, `render_sky`, `mip_levels_override`,
  `lod_thresholds`, `stream_radius` (all restored by
  `from_snapshot`; previously every config field silently reset to
  defaults on load). The `generator`/`store` hooks are host code and
  can't serialise — rebind them after loading, keyed on the new
  **`Grid::name`** tag (grid ids are runtime-opaque, so `name` is the
  stable save-file identity hosts rebind against).
  `LodThresholds` + `StreamRadius` now derive serde.
  **Payload-shape note:** the `SceneSnapshot` serde shape changed;
  pre-QE.5 *bare-bincode* blobs are not decodable by this version
  (see the envelope migration note above). Self-describing formats
  (JSON etc.) of old snapshots still deserialise (`#[serde(default)]`
  on every new field).

### Changed — internal (QE.3): one `SceneState`, one dirty-tracking entry point

No API change; two structural debts retired while small:

- **QE.3a — the facade owns scene bookkeeping once.** The material
  palette, terrain colour→material map, and the per-instance
  clip/frame table were previously kept in *duplicate* by the CPU and
  GPU backends (every new feature meant three coordinated edits, and
  the copies had already drifted: the GPU had change-detection the CPU
  lacked). They now live in one facade-owned `SceneState` passed into
  the backends' render passes; backends keep only the genuinely
  divergent reactions (CPU flipbook draw, GPU model-id/instance-buffer
  writes + device palette mirror) behind small `apply_*` hooks. The
  PF.5 same-frame guard on clip playback now covers both backends
  (was GPU-only). Spawning against a missing model/registry now books
  no facade handle at all (previously the backends could silently
  append nothing while the facade still minted a handle).
- **QE.3b — `Grid`'s dirty tracking has one entry point.**
  `Grid::chunk_versions` is private (read via the new
  `Grid::chunk_versions()` accessor / `chunk_version(idx)`); the
  version/extent/counter triple is only ever mutated together through
  `bump_chunk_version[_bbox]` + crate-internal helpers, so the three
  can no longer desync. Eviction now also drops a chunk's accumulated
  `DirtyExtent` (pre-QE.3b it leaked until a consumer happened to take
  it). On the GPU side the two parallel per-grid vectors
  (`versions` + `grid_mutations`) merged into one `GridSync` struct
  whose only-advance-on-complete-sync invariant is documented on the
  field it guards.

### Changed — **breaking** (QE.2): `FrameParams` is `#[non_exhaustive]`, one projection for both backends

`FrameParams` can no longer be built with a struct literal outside
roxlap-render — construct with `FrameParams::new(&settings)` and
override fields. In exchange, **future field additions stop being
breaking changes** (pre-QE.2, every added field broke every host's
literal), and the three per-backend projection knobs are gone:

| Removed field | Where it went | Migration |
|---|---|---|
| `gpu_fov_y_rad` | derived from `settings`: `fov_y = 2·atan(yres/2 / hz)` | want an explicit FOV? `settings = settings.with_fov_y(rad)` (new `OpticastSettings` helper) — it now applies to **both** backends |
| `gpu_max_outer_steps` | derived from `settings.max_scan_dist` (`/CHUNK_SIZE_XY + 4`, the formula every host already used) | set `settings.max_scan_dist` |
| `gpu_mip_scan_dist` | `RenderOptions::gpu_mip_scan_dist` (construction-time; default 64.0) | set it in `RenderOptions`; the `ROXLAP_GPU_MIP_SCAN_DIST` env var still overrides |

Before/after:

```rust,ignore
// before                                   // after
let frame = FrameParams {                   let mut frame = FrameParams::new(&settings);
    settings: &settings,                    frame.sky_color = sky;
    sky_color: sky,                         frame.fog_color = sky;
    sky: None,                              frame.fog_max_scan_dist = settings.max_scan_dist;
    fog_color: sky,                         // gpu_* fields: see the table above
    fog_max_scan_dist: msd,
    treat_z_max_as_air: true,               // `new` defaults: treat_z_max_as_air = true,
    gpu_mip_scan_dist: 64.0,                // draw_sprites = true, side_shades = [0;6],
    gpu_max_outer_steps: n,                 // lights = None — override what differs.
    gpu_fov_y_rad: 60f32.to_radians(),
    draw_sprites: true,
    side_shades: [0; 6],
    lights: None,
};
```

**Visual note:** the GPU backend previously rendered whatever
`gpu_fov_y_rad` said (typically a hard-coded 60°) while the CPU
backend rendered the `OpticastSettings` FOV (≈73.7° for the default
4:3 `for_oracle_framebuffer`). They now always match; if you relied on
the old GPU-side 60°, apply `settings.with_fov_y(60f32.to_radians())`
— to both backends, which is the point.

### Changed — **breaking** (QE.2b): construction diagnostics + `try_new`

- **`SceneRenderer::try_new(window, size, &opts) -> Result<Self,
  RenderError>`** (native) — the honest constructor: a GPU-init
  failure still falls back to CPU (now logged at `warn` via the
  [`log`] facade), and the `Err` fires only when even the last-resort
  CPU software surface can't bind. `SceneRenderer::new` stays and
  keeps the old behaviour, but its doc no longer claims "never fails"
  — it panics where it always secretly did (softbuffer init), with a
  clear message. Migration: nothing required; switch to `try_new` if
  you want to show your own error UI.
- **Library diagnostics moved from `eprintln!` to the `log` facade**
  (GPU-fallback reason, CPU light-budget demotion at `warn`;
  scene-upload/refresh traces at `info`/`debug`). Migration: install
  any logger to keep seeing them — the demos + quickstart now init
  `env_logger` with a `warn` default filter (`RUST_LOG` overrides), so
  their stderr output is unchanged.

### Added — QE.2

- `OpticastSettings::with_fov_y(rad)` — set an explicit vertical FOV
  by computing the focal length (`hz`); both backends follow it.
- `FrameParams::fov_y_rad()` / `FrameParams::gpu_outer_steps()` — the
  derived projection values, exposed for hosts/tools that need them.
- `RenderOptions::{gpu_mip_scan_dist, gpu_chunk_upload_budget,
  gpu_clip_upload_budget}` (QE.2c) — the last two were previously
  reachable **only** via `ROXLAP_GPU_CHUNK_BUDGET` /
  `ROXLAP_GPU_CLIP_BUDGET` env vars, which a shipped game can't
  reasonably set; the env vars remain as user-side overrides. `0` =
  unbounded. Defaults unchanged (2 / 8 / 64.0).

### Changed — **breaking** (QE.1c): spawn methods return `Option`

Every facade spawn that could previously fail *silently* — handing back
a sentinel id that resolved to nothing, so a misconfigured entity just
didn't exist with zero signals — now returns `Option<…>` and spawns
nothing on `None`:

| Method | Was | Now |
|---|---|---|
| `add_sprite_instance` / `add_sprite_instance_posed` | `SpriteInstanceId` (sentinel on stale model) | `Option<SpriteInstanceId>` |
| `add_clip_instance_posed` / `add_clip_instance_playing` | `SpriteInstanceId` (sentinel on stale clip) | `Option<SpriteInstanceId>` |
| `add_billboard_instance` | `SpriteInstanceId` (sentinel on stale clip) | `Option<SpriteInstanceId>` |
| `add_billboard_actor` | `BillboardActorId` (sentinel on empty def / stale clip) | `Option<BillboardActorId>` |
| `add_streaming_clip_instance` | `StreamingInstanceId` (sentinel on stale clip) | `Option<StreamingInstanceId>` |

Migration: where the source handle is freshly registered (the common
case — you just called `add_sprite_model` / `add_voxel_clip`), append
`.expect("model just registered")` to keep the old
can't-actually-fail behaviour; where failure is possible (data-driven
spawns), handle the `None`. To get literally the old semantics —
ignore the failure and carry a dead handle — there is no dead handle
anymore; store the `Option` and skip `None` at use sites.

### Added

- **`SceneRenderer::tick(camera, dt)`** (QE.1b) — one call drives every
  facade-owned animated collection in the right order (auto-playing
  clip players → all characters → billboard actors → billboard
  facing), replacing the 4-to-5-call per-frame protocol hosts had to
  know (a missed call meant frozen animation or unfaced billboards —
  silently). The fine-grained methods (`advance_voxel_clips`,
  `advance_character`, `update_billboard_actors`,
  `face_billboards_to`) stay public and unchanged for hosts that need
  custom per-entity `dt` or ordering; `tick` is exactly equivalent to
  calling them in the order above. KFA sprites driven via
  `update_kfa_poses` remain a separate call (host-owned skeletons).

- **`FrameParams::new(settings)`** (QE.0) — a constructor with
  sensible defaults for every field except the CPU `OpticastSettings`,
  so hosts stop copying 12-field struct literals. Notably it derives
  the **GPU projection from the CPU settings** (`fov_y = 2·atan(yres/2
  / hz)`), so both backends render the same field of view by default —
  previously a host had to keep `OpticastSettings::hz` and
  `gpu_fov_y_rad` in sync by hand. All fields stay public; construct
  with `new` and override what differs. Defaults: sky/fog =
  `RenderOptions::default().clear_sky` with CPU fog off, sprites on,
  no side shades, no dynamic lights, `treat_z_max_as_air = true`,
  `gpu_mip_scan_dist = 64`, step budget from
  `OpticastSettings::max_scan_dist`.
- **`quickstart` example** (`cargo run -p roxlap-render --example
  quickstart`) — a minimal winit "hello voxel world" (window + tiny
  scene + orbit camera), and a matching **"Use it in your game"**
  README section whose snippet is compiled as a doctest of
  `roxlap-render` (a `#[cfg(doctest)] #[doc = include_str!]` hook), so
  README code can no longer rot silently — the previous README snippet
  called API deleted many releases ago.

### Deprecated

- `RenderOptions::cpu_max_grid_vsid` and
  `RenderOptions::cpu_render_threads` — **both have been ignored since
  the DDA renderer replaced the strip-parallel opticast**; setting them
  has had no effect for many releases. Migration: delete the fields
  from your `RenderOptions` literal (use `..RenderOptions::default()`);
  to bound CPU render parallelism, set the standard
  `RAYON_NUM_THREADS` env var instead. The fields will be removed in a
  QE-series breaking release.

### Changed

- **Docs/onboarding sweep (QE.0)** — the 17 `PORTING-*.md` stage docs
  moved from the repository root to `docs/porting/` (all README /
  CHANGELOG / rustdoc references updated); README gained the
  quickstart section and now points docs.rs readers at `roxlap-render`
  / `roxlap-scene` first; the stale "Multicore" README section was
  rewritten (its code sample called deleted API); every hand-rolled
  camera basis in demos/tests now delegates to
  `Camera::from_yaw_pitch` (byte-identical math, one canonical
  implementation); ~55 broken rustdoc links fixed workspace-wide, and
  stale crate docs (RF.0-era facade skeleton, "GPU-only" dynamic
  lighting — CPU support landed in 0.18) rewritten;
  `#![warn(missing_docs)]` now guards `roxlap-render` +
  `roxlap-scene`; CHANGELOG version links restored for 0.3.0–0.21.0.
- **Build friction** — the workspace-root nightly `rust-toolchain.toml`
  pin moved into the two web crates (the only nightly consumers, for
  wasm `-Z build-std` threads); a fresh clone now builds on the
  developer's stable toolchain, matching CI. To get the old behaviour
  back (nightly everywhere), copy
  `crates/roxlap-web/rust-toolchain.toml` to the repo root. The
  workspace also gained `default-members` excluding `roxlap-sdl-demo`,
  so plain `cargo build` / `cargo test` no longer needs system SDL2
  headers — build it explicitly with `-p roxlap-sdl-demo` or
  `--workspace`.
- **Internal (QE.1a)** — the facade's five hand-rolled epoch slotmaps
  (`DynModelMap`, `DynClipMap`, `CharMap`, `StreamingClipMap`,
  `BillboardActorMap`) collapsed into one generic `EpochSlotMap<I>`
  (~200 lines deleted; no behaviour change — the model map keeps its
  deliberate positional-ids-survive-`set_sprites` semantics via
  `reset_live`).

## [0.21.0] — 2026-07-01

### Added

- **Spot (cone) lights** (stage SL) — a runtime directional cone light on
  **both** backends: `LightRig.spots: &[SpotLight]` alongside the sun +
  point lights. A `SpotLight` is a point light with a `direction` (cone axis)
  and soft `inner_angle_deg` / `outer_angle_deg` half-angles (a `smoothstep`
  falloff; `outer == inner` ⇒ a hard edge; `>= 180°` ⇒ an omnidirectional
  point light). Internally each spot folds into the same per-grid point-light
  array / GPU buffer (`GpuPointLight` grew 48→64 B) / shader loop, so spots
  reuse the distance falloff, hard voxel shadows, per-grid transform, and cel
  banding, and share the point-light count + shadow-caster budgets (points
  take priority). The cone factor gates the loop before the shadow march, so
  an off-cone spot skips it. Point-light-only rigs stay byte-identical. A new
  **Spotlight** scene-demo scene showcases it on its own — a near-dark room lit
  by a single spot, with a flashlight mode (`F`, the cone rides the camera), a
  sweeping searchlight, and a live-adjustable cone angle (`[` / `]`). See
  `PORTING-SPOTLIGHT.md`.
- **Per-actor runtime tint** — `SceneRenderer::set_actor_tint(BillboardActorId, tint)`,
  the per-actor counterpart to `set_sprite_instance_tint`, routes an
  `0x00RRGGBB` colour multiply to the actor's clip instance (works on both
  backends). `0x00FF_FFFF` (white) is a no-op; returns `false` on a stale id.

## [0.20.0] — 2026-07-01

### Added

- **Live render-pipeline HUD** (RP.3) — the scene-demo gains a "Render pipeline"
  egui panel (top-right) that drives `set_render_resolution` / `set_ssaa` /
  `set_posterize` at runtime: resolution mode (Native / Fixed `w×h` / Scale),
  SSAA factor, posterize levels + dither (none / Bayer / blue-noise), with the
  live logical + march sizes shown. The `ROXLAP_RENDER_RES` / `ROXLAP_SSAA` /
  `ROXLAP_POSTERIZE` / `ROXLAP_DITHER` env vars now just seed the panel's
  initial state. README + per-crate docs updated. Closes the RP pipeline
  (RP.0 fixed target + RP.1 SSAA + RP.2 posterize + RP.3 controls).
- **Posterize + dither** (RP.2) — `SceneRenderer::set_posterize(Option<PosterizeConfig>)`
  applies a reduced-palette post at the logical resolution in the resolve step
  (after the SSAA downfilter, before the nearest upscale), so each hard pixel
  quantizes once. Per-channel `levels_r/g/b` (`<= 1` ⇒ untouched) +
  `DitherMode {None, Bayer4x4, BlueNoise}` (blue-noise = texture-free
  interleaved-gradient noise). GPU folds it into `scene_resolve.wgsl` (posterize
  fields written per-frame in the resolve uniform — no pipeline rebuild); CPU
  into `posterize_pixel` (exact per-channel quantization). `None` posterize +
  `ssaa == 1` stays byte-identical. scene-demo: `ROXLAP_POSTERIZE=N` +
  `ROXLAP_DITHER=none|bayer|blue`.
- **SSAA + box-downfilter resolve** (RP.1) — `SceneRenderer::set_ssaa(factor)`
  (clamped `1..=4`) supersamples the retro grid: the raycaster marches at
  `logical × factor`, then a box-downfilter resolves back to the logical grid
  before the nearest upscale — anti-aliasing edges + reducing rotation/movement
  shimmer while keeping hard pixels. `render_dims()` now reports the march size
  (`logical × ssaa`); `logical_dims()` the resolved grid. GPU: a
  `scene_resolve.wgsl` compute pass (framebuffer→resolve buffer; the blit reads
  the logical resolve buffer). CPU: an exact integer box average
  (`downfilter_pixel`). `ssaa == 1` is a byte-exact identity, so the RP.0 paths
  are unchanged. scene-demo: `ROXLAP_SSAA` (default 1; CPU pays the full N² ray
  cost, so opt-in).
- **Fixed-resolution render target** (macro-stage RP, RP.0) — the scene now
  marches into a fixed **logical** render target that is nearest-upscaled to the
  window, so the per-pixel raycaster's cost — and thus the frame rate — stops
  depending on the window size. New `SceneRenderer::set_render_resolution`
  (`RenderResolution {Native, Fixed {w, h}, Scale(f32)}`) plus
  `render_dims()` / `logical_dims()` introspection. Mirrored on both backends:
  the GPU scene/sprite passes + framebuffer/depth run at the render size and
  `scene_blit.wgsl` integer-nearest-upscales to the swapchain; the CPU backend
  upscales its logical framebuffer into a native-size output and rasterises the
  egui HUD there so it stays crisp. Debug-line / image overlays + screen→world
  picking map window pixels into the render grid. **`Native` (the default) is
  byte-identical to pre-RP rendering.** The scene-demo defaults to an
  `860×520` grid; override with `ROXLAP_RENDER_RES` (`native` | `WxH` |
  `<scale>`). Foundation for the RP.1 SSAA + RP.2 posterize/dither passes.

## [0.19.0] — 2026-06-30

### Added

- **GIF billboard sprites** (macro-stage BB) — Doom/Build-style flat,
  camera-facing animated cutouts that are first-class lighting citizens (they
  cast + receive the dynamic shadows + lighting from the DL/XS stages). Each
  GIF frame is voxelized into a flat 1-voxel slab and played back as an
  ordinary voxel clip, so shadows, lighting, materials, and playback all come
  for free.
  - **`gif_import`** (BB.0) — `voxel_clip_from_gif(bytes, &GifImportOpts)`
    decodes an animated GIF (with disposal compositing) into a `VoxelClip` of
    flat `[W, thickness, H]` slabs (transparent pixels → cutout, GIF delays →
    clip durations). In `roxlap-formats` behind the `gif` feature, re-exported
    from `roxlap-render` behind its own `gif` feature.
  - **`set_clip_instance_clip`** (BB.1) — retarget a live clip instance onto a
    different clip (restart at frame 0, keep transform + clock policy) via the
    existing GPU model-swap — the primitive behind directional + state swaps,
    no remove/respawn.
  - **Billboard orientation** (BB.2) — `BillboardMode {None, Cylindrical,
    Spherical}` + `add_billboard_instance` / `set_billboard_mode` /
    `set_billboard_position` / `face_billboards_to(&camera)` (one batched
    transform flush). Cylindrical (default) keeps the slab vertical so its cast
    shadow stays sane as the camera orbits.
  - **Per-instance shadow flags** (BB.3) —
    `set_sprite_instance_shadow_flags(id, casts, receives)` toggles an
    instance's XS.4 shadow participation live (the per-instance counterpart to
    `Sprite::with_casts_shadow` / `with_receives_shadow`).
  - **`BillboardActor`** (BB.4) — a high-level directional actor:
    `add_billboard_actor` / `set_actor_state` / `set_actor_transform` /
    `remove_billboard_actor` / `update_billboard_actors(&camera, dt)`. The
    renderer picks the directional (N-way) clip from the view angle, plays a
    named-state animation, and faces it to the camera.
  - **"Doom" demo scene** (BB.5) — an 8-directional walking monster (casts +
    receives the sun's shadows) + a flickering non-casting flame + a standalone
    signpost billboard, all synthesised as GIFs at startup and imported through
    `gif_import` (dogfooding the full path). New scene in `roxlap-scene-demo`.
  - **`BillboardLighting`** (BB.2b) — per-instance shading mode (sprite
    `flags` bits 6/7) so a camera-facing billboard needn't suffer the
    camera-dependent `N·L` of its (camera-tracking) face normal: `FaceNormal`
    (default), `WorldUp` (a fixed world-up normal — stable directional
    shading), `AmbientOnly` (flat cutout, ambient term only), and `FullBright`
    (**emissive** — the colour at full intensity, ignoring lighting; the right
    look for glows like fire/spell auras). Both backends (`shade_sprite_lit`
    WGSL + the CPU `shade_dynamic_mode` at both sprite shade sites); set via
    `set_sprite_instance_lighting` / `BillboardActorDef::lighting`. Also
    `set_actor_lighting` to change a `BillboardActor`'s mode at runtime.
    `FaceNormal` is byte-identical to the prior look.
  - **PNG / APNG importer** (`png_import`, `png` feature on `roxlap-formats` /
    re-exported from `roxlap-render`) — the truecolor counterpart of
    `gif_import`: `voxel_clip_from_png_frames` (a sequence of independent PNG
    files, same size) and `voxel_clip_from_apng` (a single animated PNG, with
    dispose/blend compositing). PNG's 8-bit alpha is resolved as a cutout at a
    configurable `alpha_cutoff`. The voxelization core is shared with
    `gif_import` (`slab` module).
- **Per-instance sprite RGB tint** — each sprite instance carries a packed
  `0x00RRGGBB` tint that multiplies its voxel colours (per channel), so
  instances of one model can be recoloured cheaply. `0x00FF_FFFF` (white, the
  default) is a no-op, so existing sprites render byte-identically. Set it on a
  model via `Sprite::with_tint` / the `tint` field, or per dynamic instance via
  `SceneRenderer::set_sprite_instance_tint`. Both backends (CPU
  `dda_sprite` + GPU `sprite_model_dda`); no new bindings (packed into the
  instance record's free slot). Tint is colour only — transparency stays on
  the translucent-material + `alpha_mul` path.

## [0.18.0] — 2026-06-30

### Added

- **Cross-scene shadows + lit translucent sprites** (macro-stage XS, extends
  DL). Three related gaps closed:
  - **Lit translucent sprite layers** (XS.0) — translucent sprite/clip voxels
    are now shaded by the dynamic-lighting rig (sun + point lights + cel +
    ramp, flat per voxel) like opaque sprites and translucent *terrain*
    already were, on both backends. Previously the accumulate paths used the
    flat baked colour (`cast_local_layers` on the CPU, `march_instance_layers`
    in `sprite_model_dda.wgsl`); both now light each layer via the model-local
    face normal. A disabled rig is byte-identical.
  - **Cross-grid hard shadows, CPU** (XS.1) — a shadow ray cast from a voxel
    in one grid now tests *every* grid in the scene, so e.g. a ship grid drops
    a shadow on the ground grid. A world-space `SceneOccluder`
    (`roxlap-scene`) implements the new `roxlap_core::WorldOccluder` trait; the
    CPU DDA reaches it through `DdaEnv::world_shadow` + the hit grid's
    local→world transform. Built once per frame (only when a caster is active),
    borrowing the scene immutably — the composed render loop was split into a
    `&mut` cache-prep pass and an immutable render pass so the occluder can
    coexist with it. No caster ⇒ no occluder built (unchanged).
  - **Sprite shadows, CPU** (XS.2) — sprites now both **cast** and **receive**
    hard shadows. A `SpriteOccluder` (decoded sprite volumes + world poses)
    implements `WorldOccluder` and is composited with the grid occluder
    (`CompositeOccluder`): the terrain render sees it (sprites darken the
    ground), and the sprite pass queries it (a sprite is darkened by terrain or
    another sprite between it and a caster). Built only when a caster is active.
  - **Cross-grid hard shadows, GPU** (XS.3) — the scene-DDA shader's
    `shadow_occluded` now runs per grid inside a `shadow_occluded_world`
    wrapper: a shadow ray is lifted to world space and tested against every
    grid, so a caster in one grid shadows another (matching the CPU). Each
    grid's world transform (origin + local→world rotation) is packed into the
    existing per-grid camera buffer (binding 15) — no new storage buffer (the
    16-buffer limit is saturated) — via `GridWorldTransform`. Identity
    transforms reproduce the prior intra-grid shadows byte-for-byte.
  - **Per-sprite shadow flags** (XS.4) — a sprite can opt out of casting and/or
    receiving hard shadows via `SPRITE_FLAG_NO_SHADOW_CAST` /
    `SPRITE_FLAG_NO_SHADOW_RECEIVE` (both default to participating), with
    `Sprite::{casts_shadow, receives_shadow, with_casts_shadow,
    with_receives_shadow}` helpers. Honored on both backends.
  - **GPU sprites receive terrain shadows** (XS.4.2) — on devices that grant
    enough storage buffers (`GpuRenderer::sprite_shadows_capable()`), the sprite
    pass marches the terrain occupancy (the scene pass's exact ABI, spliced in
    from `sprite_terrain_shadow.wgsl`) so a sprite is darkened where terrain (or
    another grid) blocks the sun / a point light. Needed raising the device
    storage-buffer request to 22 (`pick_required_limits`); devices below that
    fall back to unshadowed GPU sprites. Per-instance `receives_shadow` honored.
  - **GPU sprites cast onto terrain** (XS.4.3) — the mirror: on a capable
    device the scene pass's `shadow_occluded_world` also marches the visible
    sprite volumes (the sprite registry bound at 19..21, spliced in from
    `scene_sprite_shadow.wgsl`), so a sprite drops a shadow on the ground.
    Per-instance `casts_shadow` honored. With XS.4.2 this completes
    bidirectional GPU sprite shadows, at full CPU/GPU parity.
- **Voxel ambient occlusion** (macro-stage AO) — a CPU bake pass that writes
  per-voxel ambient occlusion into the brightness byte, which the dynamic
  lighting (DL) reads as its ambient/AO fill: open surfaces stay bright while
  crevices, inside corners, and contact points (pillar bases, the monument's
  foot) darken — even where no sun/point light directly shades them. Reuses the
  existing lighting bake (`EstNormCache`): `EstNormCache::ambient_occlusion` is
  computed **per exposed face** (for each air-facing voxel face, how much solid
  sits in front of it), so it darkens **only concave** edges — flat faces and
  convex edges stay open, no "pillow" outline. A new `lightmode == 3` bakes it.
  Both backends benefit at zero render cost (it's the same byte they already
  multiply in). Tunable via `AoParams { strength, radius, min_floor }`
  (`bake_ao_pub`); the "Lighting" demo bakes AO into its floor/pillars/monument
  and retunes the depth live with `N`/`M`. (CPU bake; the byte feeds GPU + CPU
  alike.) For **stacked grids**, `EstNormCache::build_with_reader_z` reads the
  chunks above/below (`chz±1`) for the bake's `±ESTNORMRAD` z-padding, so AO
  (and the directional `estnorm` bake) stay continuous across a chunk z-seam
  instead of seeing fake air-above / bedrock-below at the boundary; the
  scene-graph `bake_lightmode` and the demo bakes use it.

- **Dynamic lighting** (macro-stage DL; `PORTING-DYNLIGHT.md`) — runtime
  lighting layered on the scene-DDA raymarcher: one coloured directional
  **sun**, several coloured **point lights**, and **stylized hard voxel
  shadows** cast by the sun and a chosen subset of point lights (the rest
  shadowless). Started GPU-only, then brought to full **CPU parity** (CPU.1 +
  CPU.2 below). The baked brightness byte is reinterpreted as the ambient/AO
  fill (`out = albedo·ambient + Σ direct`). Lights are per-frame via
  `FrameParams.lights`
  (`LightRig { sun, points, ambient, shadow_strength, shadow_bias_voxels,
  shadow_max_dist }`); `None` ⇒ byte-identical to pre-DL. New `DirectionalLight`
  / `PointLight` / `LightRig` types in `roxlap-render`.
  - **Sun + point lights** (DL.1/DL.2) — `shade_lit` in `scene_dda.wgsl`:
    raw albedo × ambient + N·L diffuse per light, using the DDA's hit-face
    normal (free) for terrain. Point lights attenuate with a smooth quadratic
    falloff and a hard radius cut. Lights are transformed into each grid's
    local frame on the CPU (mirroring the per-grid cameras); the per-grid sun
    direction rides in the camera struct to stay within the GPU's 16
    storage-buffer limit.
  - **Hard shadows** (DL.3) — `shadow_occluded`, a dedicated intra-grid shadow
    DDA (chunk-skipping outer loop + mip-0 inner voxel walk) reusing the scene
    occupancy. Shadow-ray origin is biased along the surface normal (anti-acne);
    the in-shadow floor is `1 − shadow_strength`. `MAX_SHADOW_CASTERS` caps the
    casters (excess demoted to shadowless with a warning). Cross-grid shadows
    deferred.
  - **Lit sprites** (DL.4) — opaque sprites/clips shade with the sun + point
    lights using their **true per-voxel normals** (voxlap `univec[256]` mapped
    from each voxel's `dir` index, rotated to world). World-space lights for the
    sprite pass; sprite shadows deferred.
  - **Stylized (retro) lighting** (DL.6) — smooth N·L reads as generic Phong
    and flattens the voxel identity, so terrain lighting gains an opt-in retro
    look: `LightRig.bands` (cel quantization — the sun key + each point factor
    snap to `bands + 1` discrete levels) + a gradient-map ramp from
    `LightRig.shadow_tint` (cool, unlit) to the sun colour (warm, lit), giving
    terraced, hue-shifted shading where shadows tint cool instead of just
    darkening. `bands == 0` keeps the smooth path byte-identical. The
    "Lighting" demo's `J` toggles stylized ↔ smooth.
  - **"Lighting" demo scene** — a sweeping sun + three orbiting coloured point
    lights over a pillared floor; `P` pauses the sun, `K` toggles sun shadows,
    `L` toggles the point lights, `J` toggles stylized/smooth.
  - **Stylized sprites + clips** (DL.7) — the cel banding + gradient-ramp +
    flat-per-voxel stylization now extends to the GPU sprite/clip pass (using
    each voxel's true normal), so animated characters and voxel clips match the
    terrain's retro look. `style_bands == 0` keeps the smooth path.
  - **CPU diffuse lighting** (CPU.1) — the dynamic lighting now also runs on the
    **CPU** backend (sun + point lights + cel + ramp, flat per voxel), terrain
    **and** sprites/clips, so the same `FrameParams.lights` rig lights both
    backends. Diffuse is arithmetic-only, so it's effectively free on the
    (bandwidth-bound) CPU path. `lights: None` keeps the CPU path byte-identical
    to the baked-brightness render. (CPU sprites use the DDA face normal — flat
    per voxel — since the CPU sprite store has no per-voxel normals.)
  - **CPU hard shadows** (CPU.2) — the CPU backend now casts the sun + point
    shadows too, so both backends are on full parity. A shadow ray marches a
    3D-DDA toward each caster through the same render `Sampler` occupancy the
    camera ray uses (bounded by `shadow_max_dist` / the light distance), and an
    occluded sample keeps `1 − shadow_strength` of that caster; the same
    `MAX_SHADOW_CASTERS` cap as the GPU applies (excess point casters demoted
    with a warning). Only marched when a caster is flagged and
    `shadow_strength > 0`, so an unshadowed rig (and `lights: None`) stays
    march-free and byte-identical. Sprites are unshadowed (matching the GPU).
    This is the slow CPU fallback's slowest path, but correct and on visual
    parity with the GPU.

## [0.17.0] — 2026-06-28

### Added

- **`SceneRenderer::wait_idle`** — blocks until the active backend has drained
  all in-flight work and releases any acquired-but-unpresented swapchain frame
  (GPU: `device.poll(Wait)`; CPU: no-op). Call it at shutdown before dropping
  the renderer and its window.

- **`BlendMode::Volumetric`** (Beer–Lambert) — the thickness-aware transparency
  mode for *filled* volumes (true smoke, fog, murky water), the deferred
  follow-up to the per-span `AlphaBlend`. Where `AlphaBlend` composites one
  alpha per surface run (opacity independent of thickness — ideal for
  shells/glass), `Volumetric` weights each voxel's opacity by the ray's path
  length through it: per-cell effective opacity `1 − (1 − alpha)^seg_len`
  (`seg_len` in voxel units), so a boundary sliver contributes ≈0 (no
  voxel-grid dicing) and a filled volume thickens smoothly with depth. Lands on
  both backends and both passes (sprites/clips + terrain), CPU pinned then GPU
  matched. New `Material::volumetric(alpha)`. The "Transparency" demo gains a
  filled volumetric fog cloud whose core reads denser than its rim.
  - **`Kv6::from_fn_keep_interior`** — surface extraction normally culls every
    enclosed voxel (a solid cube is a hollow shell), which would defeat
    Volumetric (a filled cloud would render as front+back faces with air
    between). This variant keeps interior voxels whose colour a predicate
    accepts, so translucent/volumetric bodies stay solid through while opaque
    interiors are still dropped (the storage win). (Terrain `.vxl` stores solid
    runs, so its interiors were already traversable — this gap was kv6-only.)

- **Mixed-material animated clips** — per-voxel materials (TV.3) now extend to
  voxel clips (`.rvc`), the animated analogue of
  `add_sprite_model_with_materials`: `SceneRenderer::add_voxel_clip_with_materials`
  classifies every frame's voxels into per-voxel material ids by a
  colour→material map, so an animated clip can mix opaque and translucent
  voxels (an opaque torch handle around an additive flame, a pulsing glass
  orb) on both backends. Previously clips could only carry a whole-instance
  uniform material; the per-voxel path was wired for static sprites but not
  for clip frames. An empty map is byte-identical to `add_voxel_clip`. The
  "Transparency" demo scene gains a pulsing glass-orb clip dogfooding it.
  - The per-voxel materials are also preserved across the **in-place
    single-frame edit** (`update_clip_frame`, which now re-classifies the
    edited frame from the clip's registered map) and the **streaming-clip**
    path: `add_streaming_clip_with_materials` + `refresh_sprite_model_with_materials`
    re-apply the colour→material map on every per-frame model re-upload.

### Fixed

- **Clean GPU teardown on exit** — the native demos (`scene-demo`, `cave-demo`,
  `sdl-demo`) now drain the GPU and drop the renderer (wgpu device/queue/surface)
  *before* the window, via a winit `exiting`/`suspended` handler (and a
  renderer-before-window field order so the panic-unwind path tears down the
  same way). Previously an exit could yank the swapchain mid-frame / drop the
  window before the surface, leaving the driver or compositor showing stale
  buffers — the "leftover triangles / flicker" after an unclean exit. (The
  runtime already reconfigured the surface on `Lost`/`Outdated`; this fixes the
  shutdown side.)

## [0.16.0] — 2026-06-28

### Added

- **Transparent voxels** (macro-stage TV; `PORTING-TRANSPARENCY.md`):
  alpha-blended and additive voxels for effects (smoke, fire, spell auras,
  muzzle flashes) and glass/water, on both the CPU and GPU backends. The
  per-pixel front-to-back 3D-DDA renderer composites translucent voxels in
  visit order, so it is order-correct without any depth sorting; an
  all-opaque scene renders byte-for-byte as before.
  - **Material model** — `roxlap_formats::material`: `BlendMode`
    (`Opaque` / `AlphaBlend` / `Additive`), `Material { alpha, mode }`, and
    a 256-entry `MaterialTable` global palette (id 0 is permanently
    `Material::OPAQUE`). Facade `SceneRenderer::define_material` / `material`.
  - **Sprites & clips** — per-pixel accumulate-and-continue in the CPU
    `dda_sprite` raycaster and the GPU `sprite_model_dda.wgsl` pass:
    `Additive` (commutative glow) and `AlphaBlend` (`over`), a per-instance
    `alpha_mul` (`set_sprite_instance_material` / `set_sprite_instance_alpha`)
    for cheap fade animation, and **per-span compositing** (one alpha layer
    per contiguous solid run / material change) so a translucent shell no
    longer reads as a voxel grid.
  - **Mixed-material models** (TV.3) — a single model can mix opaque and
    translucent voxels (opaque frame + glass) via a colour→material map:
    `add_sprite_model_with_materials`.
  - **Terrain** (TV.4–TV.6) — glass/water as world (grid) geometry, resolved
    from a global terrain colour→material map at render time
    (`SceneRenderer::set_terrain_materials`) — no `.vxl` format change. CPU
    `dda` + GPU `scene_dda.wgsl` both accumulate front-to-back.
  - **Demo** — a "Transparency" scene (glass pane, additive glow, pulsing
    smoke, a mixed opaque-frame+glass window, and a world glass wall).
  - CPU `render_sky_fill`: the panorama sky now fills every background pixel
    (sprite/effect-only views + the margins around small grids), matching the
    GPU.

## [0.15.0] — 2026-06-27

### Added

- **Animated voxel-sprite clips (`.rvc`) — "GIF/MP4 for voxel models"**
  (macro-stage VCL; `PORTING-VOXEL-CLIP.md`). A fixed-bbox sequence of
  voxel frames encoded as keyframes + inter-frame diffs, for effects like
  flame, spells, and muzzle flashes — decoded to a runtime *flipbook* and
  played back by selecting a frame per render (no per-frame volume
  re-upload).
  - `roxlap_formats::voxel_clip`: `VoxelClip` / `VoxelFrame` (dense-column
    layout matching the GPU sprite model) + `DecodedClip`; I/P codec
    (`from_frames` / `decode`), `serialize` / `parse` (`RVCL` chunked
    container), `LoopMode`, and `frame_at` playback math.
  - `.kv6` authoring bridge: `VoxelFrame::from_kv6` (re-index a voxel
    sprite into one clip frame) + `VoxelClip::from_kv6_frames` (encode a
    sequence of same-dims `.kv6` frames straight into a clip), so clips can
    be built from existing voxel models, not just procedurally.
  - `VoxelClip::from_frames_auto` / `from_kv6_frames_auto` — auto-choose
    keyframe vs. delta per frame (the codec's I-frame decision) instead of a
    fixed interval: a frame is keyframed on a "scene change" (its delta would
    be ≥ 60% of a keyframe) or to cap keyframe spacing (`max_keyframe_gap`),
    so seek points + size adapt to the content.
  - `.rvc` per-chunk deflate (format **v2**, via `miniz_oxide` — the
    formats crate's first runtime dependency). Each chunk is deflated when
    that shrinks it (a `flags` byte in the envelope; occupancy bitmasks +
    colour runs compress hugely), else stored raw. v1 files still parse.
  - `voxel_clip::StreamingClip` — a seekable, **O(1-frame)-memory** cursor
    over a clip's I/P stream (replays deltas from the nearest keyframe),
    the streaming alternative to `DecodedClip` for huge clips: holds one
    reconstructed frame + the compact encoded stream instead of N full
    frames. Plus `VoxelFrame::to_kv6` (the `from_kv6` inverse) to materialise
    a frame as a flat-lit `.kv6` model.
  - `SceneRenderer` streaming-clip facade: `add_streaming_clip` (one model +
    the cursor, vs the flipbook's N volumes) / `add_streaming_clip_instance`
    / `set_streaming_clip_frame` (seek + re-upload the single model) /
    `remove_streaming_clip` (+ `StreamingClipId`). For huge clips where the
    flipbook's resident N-frame footprint is too costly; all instances of a
    streaming clip share its one model (same current frame).
  - `voxel_clip::pad_stats` / `PadStats` (+ `DecodedClip::pad_stats`) — a
    clip is a *fixed* bbox, so a frame whose content fills only a corner
    still pays the full per-frame occupancy. Reports the declared `dims` vs.
    the tight `content_dims` (`pad_ratio` / `is_wasteful`) so an asset
    pipeline can warn; the encoder stays side-effect-free.
  - `SceneRenderer` auto-playing clips: `add_clip_instance_playing`
    (flipbook) / `play_streaming_clip` (streaming) attach a per-clip
    playback clock (Q8 speed + start phase), advanced by a single
    `advance_voxel_clips(dt)` — the host no longer hand-drives `frame_at` +
    `set_clip_instance_frame` per instance per frame.
  - **Editor/authoring API for clips + characters.** Queries:
    `clip_frame_count`, `clip_metadata` (`ClipMetadata`: dims / pivot /
    scale / loop / per-frame durations), `get_clip_instance_frame` — so an
    inspector + timeline scrubber needn't shadow the `DecodedClip`. Live
    auto-player control (play/pause/scrub): `set_clip_instance_paused` /
    `is_clip_instance_paused` / `set_clip_instance_speed` /
    `set_clip_instance_clock_ms` (scrub) / `clip_instance_clock_ms` (and the
    per-clip `set_streaming_clip_paused` / `_speed` / `_clock_ms` /
    `is_streaming_clip_paused` / `streaming_clip_clock_ms` analogues for
    streaming players). `update_clip_frame(id, frame, &VoxelFrame)` re-uploads one frame in
    place (O(1 frame), vs remove + re-add). `remove_character` now frees the
    models + clips it registered (no leak when hot-swapping). New
    `set_character_world_transform` teleports a character (re-solve + re-pose)
    without ticking its animation / clip clocks.
  - Editor-API hardening: `add_streaming_clip_instance` now returns a
    distinct `StreamingInstanceId` (+ `set_streaming_instance_transform` /
    `remove_streaming_instance`) — a streaming clip's frame is per-clip, so
    the type makes "scrub two instances independently" a compile error rather
    than a silent coupling. The clip / character / streaming slotmaps bump a
    generation epoch on `set_sprites`, so a handle held across a reset
    resolves to `None` instead of aliasing whatever re-took its slot.
    `upload_image` returns `Option<ImageId>` (was `ImageId(0)` on error,
    indistinguishable from the first valid id).
  - GPU flipbook: `sprite_model_from_clip_frame` (field-move upload) +
    `SpriteRegistryResident::set_instance_model` (the per-frame select).
  - CPU flipbook: `roxlap_core::ClipFlipbook` (cached `SpriteDense` per
    frame) + the generalised `draw_sprite_dense`.
  - `SceneRenderer` facade: `add_voxel_clip` / `add_clip_instance_posed`
    / `set_clip_instance_frame` (+ `VoxelClipId`).
- **RKC v3 — multi-attachment bones.** `Character`'s bones now carry a
  list of `Attachment`s (static KV6 meshes and/or animated voxel clips,
  each with a `local_offset` + `ClipPlayback`) instead of a single mesh;
  `MeshRef::Clip` indexes a new `Character::voxel_clips` (`VCLP` chunk).
  v2/v1 files are rejected (regenerate from demiurg).
- **Character attachment runtime.** `SceneRenderer::add_character` /
  `advance_character` / `remove_character` (+ `CharacterId`) emit one
  renderer instance per bone attachment — static meshes sit on their
  bones, clip attachments play back on their own clocks, all driven by the
  skeletal animation. `roxlap_core::kfa_draw::compose_attachment` composes
  a bone's solved world transform with an attachment's local offset. The
  scene-demo dogfoods it with a procedural flame clip on coco's swinging
  arm.

### Fixed

- **Voxel-clip hardening** (review pass over the VCL stage). Defensive fixes
  to the new clip code: `SpriteModelRegistry::set_instance_model` now guards
  `chain_id` (a tombstoned/out-of-range chain no longer index-panics — new
  `model_checked`); the CPU backend's `set_sprites` clears `clip_books` to
  match the GPU (clip indices restart at 0 on both; no leak);
  `remove_voxel_clip` detaches instances still pointing at the removed clip
  on both backends; `VoxelFrame::from_kv6` bounds-checks a malformed kv6
  (`ylen` / `voxels` disagreeing with the header) instead of panicking;
  `.rvc` inflate caps an untrusted `raw_len` at 64 MiB (decompression-bomb
  guard); and `frame_at` / `total_ms` compute in u64 / saturate so a very
  long clip can't overflow `2·total`.
- **GPU scene upload truncated dense chunks' colours** (`roxlap-gpu`).
  The per-chunk colour stride was a fixed `COLORS_PER_CHUNK_WORDS`
  (65536 u32s), sized for sparse terrain chunks (~36 k colours). A
  *fully dense* chunk — e.g. the cave demo's single 128×128×256 chunk
  (~207 k colours across its mip ladder) — overflowed the stride and had
  its colour data truncated; since columns upload in `y·vsid + x` order,
  the high-`y` spatial half of the chunk rendered **black** on the GPU
  backend (the CPU backend was unaffected). The stride is now **adaptive
  per grid** — grown to fit the grid's densest chunk, floored at the old
  default — so dense chunks upload in full while sparse grids keep the
  small stride (and now use slightly less memory). `GpuSceneResident`
  carries the per-grid stride so streamed re-uploads (`refresh_chunk`)
  address colours identically. Regression test:
  `scene_dda_dense_chunk_colours_not_truncated`.

### Changed

- **GPU flipbook-clip registration flushes the staging pool in batches**
  (`roxlap-gpu` / `roxlap-render`; #4). `add_voxel_clip` uploads N frame
  volumes at once; a big flipbook (or many clips registered in one frame)
  could stage that many `write_buffer`s before the next submit and exhaust
  the device staging pool — the same crash the `chunk_upload_budget` guards
  against, which then panics egui-wgpu. It now flushes (`GpuRenderer::
  flush_writes`, an empty submit) every `ROXLAP_GPU_CLIP_BUDGET` frames
  (default 8; `0` = unbounded). Streaming clips upload one model, so they
  sidestep the spike entirely.
- **`roxlap-scene-demo` refactored into a menu-driven multi-scene
  showcase** (demo-only; macro-stage DS, `PORTING-DEMO-SCENES.md`). The
  2296-line kitchen-sink `App` (≈34 fields, ≈20 hotkeys, every feature
  live at once) is split into a thin host (`host.rs`) + a `DemoScene`
  trait (`scene_api.rs`) + one module per scene (`scenes/`). The host
  owns the window, `SceneRenderer`, egui HUD, shared fly-camera +
  mouse-look, and FPS; `Tab` opens a scene picker. Six scenes each
  showcase one feature cluster: **World** (streaming hills + ship +
  collision-fly), **Sprites** (coco field + shoot-to-carve + streaming
  ring), **Animation** (KFA arm + flame-clip character), **Picking**
  (top-down `view_ray`/`pick`), **Primitives** (`draw_lines` /
  `draw_images` / `pick_image`), and **Empty**. Pruned the rarely-used
  bits: `ROXLAP_AUTOFLY`, the in-app bench, the `H` A/B pose toggle, the
  `F` frame-capture, `ROXLAP_NO_SPINNER`, and `ROXLAP_GPU_NO_SPRITES`.
  Kept `ROXLAP_GPU` / `STATIC` / `RKC*` / `KFA_DUMP` / `FPS_LOG` /
  `GPU_MIP_SCAN_DIST` / `SPRITE_GRID`.
- **`roxlap-cave-demo` migrated onto the `SceneRenderer` facade**
  (demo-only). The cave world — exactly one scene chunk
  (`128 × 128 × 256`) — is now a single-grid, single-chunk
  `roxlap_scene::Scene` (identity transform, chunk `(0, 0, 0)`,
  materialised via `CaveChunkGenerator`). The demo renders through
  `SceneRenderer::{render, present}` instead of the hand-rolled
  `softbuffer` + `render_dda_parallel` loop, so it gains the GPU backend
  for free: run with `ROXLAP_GPU=1` (automatic CPU fallback). Carves +
  relights go through `Grid::set_sphere_with_colfunc` +
  `bake_lightmode`; collision reads the cave chunk via `getcube`;
  plasma bullets are now voxel-sphere sprites driven by the dynamic
  sprite API (`add_sprite_model` / `add_sprite_instance_posed` /
  `set_sprite_instance_transforms` / `remove_sprite_instance`). The
  per-chunk generator seed (`FNV(base_seed, (0,0,0))`) makes the cave
  equivalent but not byte-identical to the prior direct-seed output.
  Drops the `softbuffer` dependency; adds `roxlap-scene` + `roxlap-render`.

## [0.14.0] — 2026-06-25

This release lands the **DDA** macro-stage (DDA.0–DDA.10): the CPU
renderer is now a **clean-room per-pixel 3D-DDA over an 8³ brickmap**
that replaces voxlap's opticast outright — fixing the long-standing
voxlap-inherent artifact classes (silhouette notch, floor hairlines,
axis-aligned mip beams) — and the last voxlap-derived engine code is
excised, so the **Ken-Silverman commercial-use caveat is dropped: roxlap
is now MIT OR Apache-2.0 with no third-party restriction** (free for
commercial use). On top of that, an additive **dynamic sprite-model +
per-instance-transform API on `SceneRenderer`** lets a physics-driven
caller stream sprite models in/out and orient instances per frame without
bypassing to `GpuRenderer`. Breaking vs 0.13.0: `render_scene` /
`render_scene_composed` take a `CpuFog` value instead of `&mut
ScratchPool`, and `FrameParams::sprite_lighting` is replaced by a plain
`draw_sprites: bool`.

### Added

- **Clean-room CPU renderer** `roxlap_core::dda`: per-pixel 3D-DDA over
  an 8³ brickmap. Public API: `render_dda`, `render_dda_parallel`
  (rayon tile bands), `DdaEnv` (sky / fog / side-shades), `BrickCache`
  (cross-frame occupancy cache), `RasterSink`, `pixel_ray`,
  `effective_mip`.
- **Clean-room KV6 sprite raycaster** `roxlap_core::dda_sprite::draw_sprite_dda`
  — per-pixel ray cast through a sprite's KV6, depth-composited against
  the shared z-buffer.
- `roxlap_core::raster_target::RasterTarget` — the framebuffer/z-buffer
  compositing primitive shared by the DDA terrain + sprite passes.
- `roxlap_scene::render::CpuFog { color, max_scan_dist, side_shades }`
  — the CPU fog/side-shade config passed into the scene render entry
  points.
- **Dynamic sprite-model + per-instance-transform API on `SceneRenderer`**
  (purely additive — no existing signature changed): stream unique
  procedural sprite models in and out and orient placed instances per
  frame entirely through the facade, without dropping to `GpuRenderer`.
  New items in `roxlap_render`:
  - `SceneRenderer::add_sprite_model(&Kv6) -> SpriteModelId` — register
    one model incrementally (GPU appends an LOD chain; works before any
    `set_sprites`).
  - `SceneRenderer::remove_sprite_model(SpriteModelId) -> bool` — free a
    model's voxel data in place (ids never reused, so other handles stay
    valid); `false` on a stale handle.
  - `SceneRenderer::compact_sprite_models()` — reclaim the GPU buffer
    holes left by removed models (no-op on the CPU backend).
  - `SceneRenderer::add_sprite_instance_posed(SpriteModelId, DynSpriteTransform)`
    — spawn an instance already oriented (no one-frame axis-aligned
    flash).
  - `SceneRenderer::set_sprite_instance_transform(SpriteInstanceId, DynSpriteTransform)`
    and the batched `set_sprite_instance_transforms(&[(…, …)])` — update
    placed instances' position + orientation per frame (the GPU backend
    coalesces a frame's updates into a single buffer upload).
  - `DynSpriteTransform { pos, right, up, forward }` — the per-instance
    pose (model→world basis columns; `det ≠ 0`, identity by default; a
    degenerate basis silently skips the instance).
  - `SpriteModelId` is now a generational handle (`{ slot, gen }`,
    fields private — externally unchanged); a removed model's handle
    resolves to nothing → safe no-op.
- `roxlap_gpu::sprite_model`: `SpriteModel::empty()` and
  `SpriteModelRegistry::{remove, is_live}` — in-place free of a model
  chain's voxel data, preserving ids (no remap), backing the facade's
  `remove_sprite_model`.

### Changed

- **DDA is the default and only CPU renderer.** It fixes the
  long-standing voxlap-inherent artifact classes (silhouette notch,
  floor hairlines, axis-aligned mip beams). `render_scene` /
  `render_scene_composed` take a `CpuFog` value where they previously
  took `&mut ScratchPool`.
- `FrameParams::sprite_lighting: Option<&SpriteLighting>` → a plain
  `draw_sprites: bool` (both backends now draw sprites flat-lit; the GPU
  sprite pass uses its identity colour table to match the CPU path).
- **Engine math reimplemented independently** for a clean license:
  lighting bake (`world_lighting`), camera projection (`camera_math`),
  and the KFA bone solver (`kfa_draw`) rewritten from first principles;
  the `.vxl` slab editor (`roxlap-formats::edit`) de-ported (independent
  expression, byte-compatible with the format — existing assets and
  outputs are unchanged).
- README / crate descriptions reframed: roxlap is an independent engine
  that interoperates with Voxlap's file formats, not a line-by-line port.

### Removed

- **Breaking:** the voxlap renderer and its public API are gone from
  `roxlap-core` (~14k LOC): `opticast()`, `OpticastOutcome`,
  `rasterizer::ScratchPool`, `scalar_rasterizer::ScalarRasterizer`,
  `sprite::{draw_sprite, DrawTarget, SpriteLighting, sprite_colmul}`,
  `kfa_draw::draw_kfa_sprite`, and the `grouscan` / `scan_loops` /
  `drawtile` / `gline` / `column_walk` / `opticast_prelude` /
  `projection` / `ray_step` / `ptfaces16` modules. `OpticastSettings`
  is kept (now just projection + scan-distance settings).
  `solve_kfa_limbs` is kept (pose the limbs, then draw via
  `draw_sprite_dda`).
- `roxlap_core::meltsphere` removed (unused voxlap-derived sphere-carve).
- The `roxlap-oracle` crate (voxlap-C render-hash diff harness) removed.

### License

- **Dropped the Ken-Silverman commercial-use caveat.** roxlap is now an
  independent implementation containing no Voxlap C source; dual
  MIT/Apache-2.0 now applies to commercial use as well. The crates still
  interoperate with Voxlap's on-disk formats (`.vxl` / `.kv6` / `.kvx` /
  `.kfa`), which are not subject to copyright.

## [0.13.0] — 2026-06-22

This release lands two larger threads — a full **rigged-character pipeline**
(an on-disk `.rkc` container plus a TRS/quaternion bone rig replacing the
legacy single-angle hinge) and an **incremental GPU sprite path** (stream
models and instances in and out without rebuilding the resident registry) —
alongside a horizontal scene-flip primitive, image-sprite alpha cutoff +
picking, and a handful of render/GPU correctness fixes. One small breaking
change: `OpticastSettings`/`ScanContext.anginc` is now `f32`.

### Added

- **`.rkc` rigged-character container (`roxlap-formats::character`).** The
  on-disk form of a whole animated voxel character — `RKCH` magic, chunked
  `META`/`MSHS`/`BONS`/`CLPS`, reusing kv6 mesh blobs and the kfa
  `Hinge`/`Seq` layout. Forward-compat by construction: unknown top-level
  chunks and unknown clip kinds are preserved verbatim on resave, and typed
  `mesh_kind`/clip-kind discriminants leave room for voxel-video.
  `to_kfa_sprite` builds a renderable `KfaSprite`. `scene-demo`'s `build_kfa`
  authors a `Character` and round-trips it through serialize/parse,
  dogfooding the format end to end.
- **`.rkc` disk loader + lossy `.kfa` export.** `Character::to_kfa(clip,
  kv6_name)` writes a voxlap-toolchain `.kfa` (skeleton + one clip + a single
  kv6 filename), the interop writer scoped out of the format itself.
  `scene-demo` gains a runtime source selector mirroring how a host loads
  characters: `ROXLAP_RKC=<path>` loads + parses an `.rkc` from disk (falling
  back to the built-in character on failure), `ROXLAP_RKC_DUMP` /
  `ROXLAP_KFA_DUMP` write the authored character as `.rkc` or the lossy `.kfa`.
- **TRS bone rig + quaternion math (`roxlap-formats::xform`).** A `Quat`
  (axis-angle, normalize, rotate, `Mul`, `nlerp`, `from_euler`/`to_euler`
  intrinsic ZYX, `from_basis` via Shepperd's method, `conjugate`) and a
  `BoneXform { t, r, s }` local transform with `blend`,
  `from_hinge_angle`/`hinge_angle`. `setlimb` now composes TRS (rotate the
  velcro frame by the quaternion, offset the child anchor, scale the basis),
  with the math split into a pure `limb_xform` helper. Round-trip tested.
- **Incremental GPU sprite lifecycle (`roxlap-gpu`).** Stream sprites in and
  out without the full volume + buffer rebuild of `set_sprite_instances`:
  `append_sprite_instances`/`remove_sprite_instance`/`sprite_instance_count`
  (amortised O(1) push + power-of-two grow / O(1) swap_remove),
  `add_sprite_model` (amortised O(new model voxels) LOD-chain upload),
  `remove_sprite_model` + `compact_sprite_models` (tombstone + free-list reuse
  + buffer compaction, all without remapping caller ids), and
  `dead_sprite_model_count` as the fragmentation signal. Device-backed tests
  readback-verify free-list reuse, grow boundaries, and post-compaction
  offsets.
- **Unified incremental sprite-instance API (`SceneRenderer::{add,remove}_sprite_instance`).**
  Both backends get a cheap add/remove path with stable `SpriteInstanceId`
  handles (a gen-guarded slotmap absorbs the backends' swap-remove indexing).
  The `scene-demo` gains a streamed spinner (a rotating ring of colour-sphere
  sprites added/removed each frame) exercising the path on whichever backend
  is active.
- **Horizontal scene flip (`SceneRenderer::set_flip_x`).** Mirrors the scene
  in X right before display on both backends (CPU framebuffer flip; GPU
  `scene_blit.wgsl` X-mirror gated by a per-frame uniform flag), so a host can
  correct a left-handed render while the egui overlay stays upright. Line and
  image overlays mirror their NDC X and depth lookup to match.
- **Image-sprite alpha cutoff + screen→sprite picking.** `ImageSprite::alpha_cutoff`
  discards (does not blend) texels below the threshold on both backends, for
  crisp pixel-art edges; `SceneRenderer::pick_image` resolves the nearest image
  sprite under a pixel to its hit `uv`/texel (`ImagePickHit`), alpha-aware
  (transparent texels see through) and occlusion-aware (depth-tested sprites
  behind geometry are rejected). Geometry factored into the unit-tested
  `ray_quad_uv`; the GPU backend keeps a CPU shadow copy of each upload.

### Changed

- **`OpticastSettings`/`ScanContext.anginc` is now `f32`** (was `i32`).
  `anginc < 1` supersamples the angular ray fan (more ray planes), masking the
  thin-geometry silhouette holes the grouscan marcher drops on isolated voxels
  — a density knob, not a fix. `anginc = 1.0` is byte-identical to the old
  integer `1`, so all goldens are unchanged.
- **`.kfa` runtime poser now carries full TRS.** `KfaSprite::{kfaval, frmval}`,
  `set_animation`, and the `.rkc` clip store (`ClipData::Skeletal::frmval`,
  container `VERSION` 2) hold `BoneXform` instead of one Q15 hinge angle per
  bone; `animsprite` Phase 3 blends with `BoneXform::blend` (lerp
  translation/scale, nlerp rotation). Keyframes pose identically; in-between
  interpolation moves from linear-angle to nlerp (exact at the midpoint for
  same-axis rotations). The lossy `.kfa` export collapses each frame to its
  hinge angle about the bone axis.

### Fixed

- **GPU streaming no longer freezes on a batch of newly-streamed chunks.**
  `refresh_dirty` now caps per-frame chunk installs to a budget (default 4,
  `ROXLAP_GPU_CHUNK_BUDGET`, 0 = unbounded); leftover dirty chunks ride
  subsequent frames. Evictions stay unbounded.
- **GPU side-shade at a chunk-boundary-flush surface.** `march_grid` now seeds
  `hit_axis` from the face the ray crossed to enter each chunk, so a voxel
  solid at the chunk-entry point gets its real face normal instead of the
  hardcoded z pair (which split the surface bright/dark at the horizon).
- **GPU depth test for mirrored lines/images.** `line.wgsl`/`image.wgsl` now
  mirror their depth-buffer lookup back to `width-1-px` under `flip_x`, so
  overlays depth-test against the correct column instead of the mirror
  position.
- **`grouscan::run_phases` degenerate-cf hang.** An unconditional
  `MAX_PHASE_STEPS` (100M) cap converts a degenerate slab-split spin into a
  bounded, loudly-reported bail-out instead of a silent CI hang. The bound
  sits far above real rendering and the synthetic-prologue fixtures.

## [0.12.0] — 2026-06-16

Two additive features, no breaking changes. The renderer gains a
world-placed 2D **image-sprite** primitive — a flat RGBA texture drawn as
a depth-composited quad in world space — and `roxlap-core` gains canonical
right-handed `Camera` constructors so hosts stop hand-rolling (and
mis-handing) the camera basis.

### Added

- **World-placed 2D image sprites (`SceneRenderer::draw_images`).** A
  renderer primitive that draws an RGBA texture as a flat quad positioned
  in world space, composited with the scene depth buffer so voxel geometry
  occludes it correctly (not a screen-space overlay, not a voxel slab).
  Follows the `draw_lines` lineage: upload a texture once
  (`upload_image` → `ImageId`, released by `drop_image`), then draw it
  per frame between `render` and `present`/`paint_egui` from an
  `ImageSprite { image, origin, facing, size, tint, depth_test,
  double_sided }`, where `ImageFacing` is either world-fixed
  (`World { u, v }`) or camera-facing (`Billboard { up }`). `origin` is
  the top-left corner; `size` scales `u`/`v` (1 texel = 1 voxel for traced
  pixel-art). UVs are perspective-correct on both backends (CPU: a
  near-clipped textured-triangle rasteriser into the same framebuffer/
  z-buffer; GPU: `image.wgsl`, re-homogenised quads + a manual depth test
  against the scene-DDA `best_t`, nearest sampling, straight-alpha
  over-blend). Depth-tested sprites are occluded with a bias to avoid
  z-fighting on a coincident face; `double_sided: false` back-face-culls
  world quads. Each sprite carries an `alpha_cutoff` (texels below it are
  discarded outright, not blended — crisp pixel-art edges, and the same
  threshold defines pick-solidity). The first consumer is the demiurg
  voxel editor's reference overlay; the `scene-demo` `I` hotkey toggles a
  demo reference quad.
- **`SceneRenderer::pick_image` (screen→sprite picking).** The nearest
  image sprite under a window pixel, resolving which `uv` / source texel
  was hit (`ImagePickHit`). Transparent texels (below `alpha_cutoff`) are
  see-through — the pick passes through them to a sprite behind — and a
  `depth_test` sprite occluded by nearer scene geometry is rejected
  (shares `pick`'s depth convention). Lets an editor eyedrop a colour off
  the reference or snap placement to its pixel grid; the GPU backend keeps
  a CPU shadow copy of each upload for the alpha test.
- **`SceneRenderer::project_point` (`world → screen`).** The backend-correct
  inverse of `view_ray`: projects a world point to window pixels under the
  last frame's projection (CPU `setcamera` `hx/hy/hz`, GPU vertical-FOV
  pinhole), so hosts never reconstruct it themselves.
- **Canonical camera constructors `Camera::from_yaw_pitch`,
  `Camera::orbit`, and `Camera::look_at`.** All three build the
  right-handed `(right, down, forward)` basis the engine actually
  renders with (`right × down = +forward`), reproducing
  `oracle.c::set_camera_yaw_pitch`. Projects previously hand-rolled
  `right = [-sin yaw, cos yaw, 0]`; copying that form by hand is the
  usual source of a left-handed basis, which silently makes the sprite
  frustum cull reject every sprite. New crate-level docs explain the
  z-down world's inherent horizontal mirror versus a real camera and
  why an un-mirrored world must be handled consumer-side (mirror a
  world axis or negate yaw) rather than by flipping `right`. `Camera`'s
  placeholder `Default` is now documented as left-handed and not for
  interactive use.

## [0.11.0] — 2026-06-15

Multi-grid scenes get cheaper on both backends. The GPU renderer's
16-grid-per-scene cap is gone (per-grid cameras moved to a runtime-sized
storage buffer), and the CPU compositor now scissors each grid to its
projected screen footprint instead of paying a full frame per grid. The
only breaking change is the removal of the now-meaningless
`roxlap_gpu::MAX_SCENE_GRIDS` constant.

### Performance

- **Per-grid screen scissor for the CPU multi-grid compositor.**
  `render_scene_composed` previously rendered every in-range grid as a
  near-full-frame opticast plus several full-screen memory passes (temp
  reset, sentinel sweep, compose), so a scene of N grids cost ≈ N full
  frames regardless of how little of the screen each grid covered. Each
  grid is now projected to a conservative screen rectangle and (a) skipped
  outright when it falls off-screen on either axis, (b) opticast-clipped to
  its vertical band via the existing `y_start`/`y_end` strip path, and (c)
  has its temp reset / sentinel sweep / compose restricted to that band.
  Byte-identical to the old output (a new test renders a scene with the
  scissor on and off and asserts the framebuffer matches). The horizontal
  opticast *march* is **not** clipped — the radar's column-indexed
  `angstart` table isn't reset per grid, so column-clipping reads stale
  entries and crashes at extreme poses; that remains future work.

### Changed

- **Lifted the 16-grid-per-scene cap on the GPU renderer.** The shader's
  per-grid cameras moved out of a fixed `array<…, 16>` uniform field into
  a runtime-sized storage buffer (`scene_dda.wgsl` binding 15), so a scene
  can now hold any number of grids — the only ceiling is the device's
  storage limits (per-grid metadata + voxel data were already unbounded
  storage arrays, and the CPU path never had a cap). Removes the
  `MAX_SCENE_GRIDS` constant, the `GpuSceneResident::upload` /
  `render_scene` grid-count asserts, and the facade's grid-skip cap.
  Per-pixel cost stays linear in the *uploaded* grid count (no cross-grid
  culling on the GPU), so a host with many grids should still upload only
  the visible ones. **Breaking:** `roxlap_gpu::MAX_SCENE_GRIDS` is gone.

## [0.10.0] — 2026-06-15

A small, additive release: editing a single registered sprite model no
longer re-uploads the whole sprite field. The renderer keeps a slack-backed
suballocator for each model's variable-length colour/normal data, so a
carve or recolour touches only that one model's GPU bytes — and the edit
API is keyed by a stable `SpriteModelId` handle rather than a positional
index. No published crate's existing call breaks: `set_sprites` simply
gains a return value (handles) and `Kv6` is newly re-exported from
`roxlap-render`.

### Added

- **Incremental single-sprite refresh — `SceneRenderer::refresh_sprite_model`.**
  Editing one sprite model's geometry (a carve or recolour) no longer
  re-uploads the **whole** sprite registry. `set_sprites` now returns a
  `Vec<SpriteModelId>` — one stable handle per model — so callers never
  track the positional index themselves; `refresh_sprite_model(id, &Kv6)`
  brings that one model's stored geometry up to date after a content edit,
  leaving every other model and the entire instance set untouched (an edit
  never moves or adds an instance). It is a **backend-agnostic content
  refresh**, not a GPU upload: on the GPU backend the model's `colors` /
  `dirs` arrays are written through a slack-backed **suballocator** — in
  place when they fit, relocated (with a `model_meta` rewrite) when a carve
  grows the surface-voxel count past the slot's slack, and only on a
  buffer-tail overflow are the colour/dir buffers grown + the registry
  repacked — while the dims-fixed `occupancy` / `color_offsets` arrays are
  always written in place; on the CPU backend it swaps the edited `kv6`
  into each instance of the model. The `G`-carve hotkey and the scene-demo
  shoot-to-carve both move onto this path, so a shot re-uploads ~one
  model's bytes instead of the full 256-instance field. The sprite
  registry's GPU buffers are now `STORAGE | COPY_DST` with over-allocation
  to back the in-place writes. `set_sprites` remains the bulk/setup path
  for adding/removing models or changing instances. `Kv6` is now
  re-exported from `roxlap-render`.

## [0.9.0] — 2026-06-15

The GPU renderer reaches the browser. `roxlap-gpu` and the `roxlap-render`
facade now compile and run on `wasm32-unknown-unknown` (WebGPU), and both
web demos — `roxlap-web` and `roxlap-cave-web` — are rebuilt **on top of
`roxlap-render`** instead of calling `roxlap-core` opticast directly. In a
WebGPU browser they render via the wgpu compute marcher; elsewhere the
facade falls back to the CPU opticast path, now presented through a WebGL2
blit owned by the facade. This closes the last *architectural* gap from
the GPU and scene-graph roadmaps; the release stays **0.9.x** rather than
1.0 because engine limitations remain open (e.g. VC.6 mip-N multi-chz,
the deep-mip axis-aligned beam mitigation) and the API is not yet frozen.
No public API of the existing native crates is broken; the change is
additive (new wasm constructors + one new `Grid` method). This release
also lands three editor (demiurg) handoffs — kv6 surface normals, a
dense-model → `.vxl` export path, and depth-tested overlay lines.

### Added

- **Depth-tested 3D line drawing — `SceneRenderer::draw_lines`.** A
  world-space line-overlay pass on the `roxlap-render` facade, drawn
  **between `render` and `present`/`paint_egui`** and dispatched
  per-backend, for editor gizmos / debug geometry (bounding boxes, floor
  grids, origin axes, paths). `draw_lines(camera, &[Line3])` projects each
  segment with the frame's own projection and **depth-tests it against the
  frame's z-buffer**, so rendered geometry occludes lines behind it.
  `Line3 { a, b: [f64;3], color: u32 /*0xAARRGGBB*/, width_px: f32,
  depth_test: bool }` — alpha-blended by the colour's high byte, screen-
  space thickness via `width_px`, and `depth_test: false` for always-on-
  top overlays (e.g. a hover highlight). Both backends honour the same
  semantics in **their own depth metric** (CPU: perpendicular distance;
  GPU: euclidean `best_t`): the CPU backend rasterises the segments into
  the framebuffer with a perspective-correct depth interpolation, while
  the GPU backend expands them to screen-space quads (`line.wgsl`) and
  composites a `LoadOp::Load` pass that samples the scene-DDA depth
  buffer. The GPU scene pass now **always writes depth** (was gated on
  sprites being present), which also makes `pick_depth` work on a
  sprite-less GPU frame. egui overlay ordering is unaffected — the lines
  land in the framebuffer, so `paint_egui` still draws panels on top.
- **`Vxl::from_dense` + `Vxl::empty` — one-call dense-model → `.vxl`
  export.** `Vxl::from_dense(vsid, |x,y,z| -> Option<u32>)` builds a world
  from a dense occupancy + `0x80RRGGBB` colour closure (z-down,
  `z ∈ [0,256)`), and `Vxl::empty(vsid)` is the blessed all-air
  constructor — both seed voxlap's `loadnul` slab shape and edit it, so
  callers never hand-roll the slab format. `from_dense(..)` +
  `vxl::serialize(..)` is now the whole export path. Relatedly,
  **`vxl::serialize` now round-trips a post-edit `Vxl`**: it rebuilds the
  column data in column-index order (was raw-dumping `data`, which is
  wrong once `voxalloc` scatters columns), so any edited world — not just
  a freshly-parsed one — serialises to a valid `.vxl`. Byte-identical for
  unedited worlds.
- **kv6 per-voxel shading + a public `normal → dir` quantiser.**
  `Kv6::from_fn_shaded` builds a model with real surface normals
  (`Voxel::dir`) and exposed-face masks (`Voxel::vis`) instead of the
  flat `dir = 0`, `vis = 63` of `from_fn`, so procedurally-authored
  models shade with a directional gradient on the CPU sprite path (which
  reads `kv6colmul[dir]`) like an authored `.kv6`. `Kv6::recompute_surface`
  refreshes `vis`/`dir` after editing a model's voxels. The voxlap
  `univec[256]` direction table moved to `roxlap-formats`
  (`roxlap_formats::equivec`, re-exported from `roxlap-core`) so kv6 can
  reach it without a circular dependency, and gained
  `equivec::nearest_dir(n) -> u8`. The `vis` face-bit order is calibrated
  against `coco.kv6`'s authored bytes (all six faces).
- **`roxlap-gpu` + `roxlap-render` build for `wasm32` (WebGPU).** New async,
  canvas-based constructors — `GpuRenderer::new_from_canvas` and
  `SceneRenderer::new_from_canvas_async` — create the wgpu surface from an
  `HtmlCanvasElement` via `SurfaceTarget::Canvas`. The browser drives the
  adapter/device futures through its event loop (no `pollster`). The
  generic window-handle constructors stay the native path; the wasm
  constructors drop the `Send + Sync` bound (wgpu types are `!Send` on the
  `+atomics` shared-memory build, and the browser host is single-threaded).
- **CPU fallback presents on the web.** The facade's CPU backend, which
  uses `softbuffer` on native, presents its composited framebuffer through
  a WebGL2 texture-blit (`cpu_blit.rs`) on wasm. So a browser without
  WebGPU still renders, via CPU opticast — same `SceneRenderer`, same API.
- **Graphics stack upgraded to wgpu 29** (from 22) + naga 29 + egui/
  egui-wgpu 0.34. wgpu 22 unconditionally sent the WebGPU-spec-removed
  `maxInterStageShaderComponents` device limit, which current Chrome/Dawn
  reject — so WebGPU device creation failed on modern browsers. wgpu 29
  tracks the current spec. The wasm GPU init also now probes the adapter
  **and** device before binding the canvas surface, so any GPU-init
  failure leaves the canvas pristine for the WebGL2 CPU fallback (no
  more "no webgl2 context" crash).
- **`Grid::bake_lightmode(lightmode)`** in `roxlap-scene` — bakes voxlap
  `estnorm`/`updatevxl` per-voxel lighting into every materialised chunk's
  brightness bytes, neighbour-aware at chunk-XY seams. A reusable engine
  API (previously a scene-demo-only helper); the cave web demo bakes with
  it after generation and after each carve.

### Changed

- **`roxlap-web`** renders a procedural terraced-hills world through the
  facade (WebGPU, CPU fallback). The bundled `oracle.vxl` parse + the
  hand-rolled WebGL2 blit are gone (the blit now lives in the facade); the
  resolved backend (WebGPU vs CPU) is logged to the console.
- **`roxlap-cave-web`** renders the procedural cave as a single-chunk
  `Scene` grid through the facade. Flying + per-voxel collision + runtime
  carving run against the scene; plasma bullets are now facade **sprites**
  (glowing voxel spheres) that carve a crater with a local lightmode-1
  re-bake on impact, which the GPU path re-uploads via per-chunk dirty
  tracking. `F`/`R` regenerate the cave in place.

## [0.8.0] — 2026-06-13

Ports voxlap's KFA animation-curve playback (`animsprite`) and brings the
GPU sprite path to parity with the CPU: animated KFA sprites, directional
normal-based sprite lighting, and sprite rendering in grid-less
(model-viewer) scenes, plus a grid-lighting hook on the render API.
**Breaking:** `FrameParams` gains a `side_shades: [i8; 6]` field (default
`[0; 6]`); `HeadlessSceneRenderer::new` takes a `&wgpu::Queue` (so it
uploads its default sky); `KfaSprite` gains public
`frmval`/`seq`/`kfatim`/`okfatim` fields and `SpriteModel` a public `dirs`
field (built by the existing constructors — only direct struct-literal
construction is affected).

### Added

- **Grid side-shade lighting hook (`FrameParams.side_shades`).** Plumbs
  voxlap's `setsideshades(top, bot, left, right, up, down)` through the
  render API — the grid-scan analogue of `sprite_lighting` — on **both**
  backends. The CPU rasteriser applies it via `gcsub`; the GPU scene-DDA
  pass darkens a hit voxel's brightness by the hit face's shade (the face
  from the DDA's last-stepped axis), reducing the alpha-brightness before
  the `/128` divide exactly like `grouscan_shade`. With a flat (un-baked)
  brightness it's pure runtime side-shading; with baked light it stacks,
  as voxlap does. Default `[0; 6]` keeps `sideshademode` off
  (byte-identical to before). A host passes its engine's `side_shades()`
  so the board shades by the same sun.

### Fixed

- **GPU sprite-only / empty-scene rendering.** Sprites now render on the
  GPU backend when the scene has no voxel grids (a pure model/sprite
  viewer). Previously the facade short-circuited a grid-less scene to a
  bare clear and never ran the sprite pass, so only the clear colour
  showed (CPU rendered the sprite fine — a backend asymmetry). The
  no-grids-but-sprites case now renders through a cached zero-grid
  resident so the scene pass fills the sky background + far depth and
  the sprite pass composites over it. Also fixed a latent degenerate
  sky direction: with zero grids the sky ray was `(0,0,1)`, whose
  `atan2(0,0)` panorama lookup sampled black — the scene shader now
  derives the sky direction from a dedicated world camera in the
  uniform (which also retires the "grid 0 is the world frame"
  assumption for the sky on rotated grids).

### Added

- **KFA animation-curve playback (`animsprite`).** Faithful port of
  voxlap's `animsprite` (`voxlap5.c:11125`): `KfaSprite::set_animation`
  attaches the parsed `frmval` + `seq` tables and
  `KfaSprite::animsprite(dt_ms)` advances the time cursor (honouring
  `!target` loop/jump entries) and recomputes `kfaval[]` by wrap-aware,
  fixed-point keyframe interpolation. Previously the host had to drive
  `kfaval[]` by hand; baked `.kfa` curves now play back.
- **KFA sprites on the GPU backend.** `SceneRenderer::set_kfa_sprites`
  registers each limb's KV6 as an instanced model once, and
  `update_kfa_poses` re-poses them every frame — GPU via a cheap
  transform-only instance-buffer update (no model-volume re-upload,
  `GpuRenderer::update_sprite_instance_transforms`), CPU by re-solving
  limb transforms. Bone posing was factored out of `draw_kfa_sprite`
  into the reusable `roxlap_core::kfa_draw::solve_kfa_limbs`. The
  scene-demo shows an `animsprite`-driven swinging arm on both backends.
- **Directional sprite lighting on the GPU backend.** The GPU sprite
  pass now shades KV6 voxels by surface normal, matching the CPU
  rasteriser instead of rendering flat colours. Each voxel's `dir` is
  uploaded alongside its colour, and the renderer builds voxlap's
  `kv6colmul[256]` table per instance via the new
  `roxlap_core::sprite::sprite_colmul` (reusing the exact
  `update_reflects` math); the shader applies the same per-channel
  `mulhi`/saturate modulation. Tables are rebuilt each frame so rotating
  KFA limbs re-shade correctly. A naga-based unit test statically
  validates every WGSL shader.

## [0.7.0] — 2026-06-12

Decouples the renderer from winit (binds to any `raw-window-handle`
provider), adds an **egui overlay seam** on both backends, and ships an
SDL2 host demo. Breaking: `SceneRenderer::new` is generic + takes a size,
and `render` no longer presents (add a `present()` / `paint_egui()`
call). GPU sprite-camera + sky/fog parity bugs fixed along the way.

### Added

- **egui overlay seam (`hud` feature).** A renderer-level path to draw an
  egui UI (HUD, debug panels, menus) on top of the rendered frame, on
  **both** backends. `SceneRenderer::paint_egui(jobs, textures,
  pixels_per_point)` takes the host's tessellated egui output and:
  - GPU — paints via `egui-wgpu` (`LoadOp::Load` over the marcher's
    frame);
  - CPU — software-rasterises the tessellation (textured triangles,
    font/image atlas, premultiplied src-over) into the framebuffer.

  The host runs egui itself (e.g. `egui` + `egui-winit`); roxlap consumes
  only `egui::ClippedPrimitive`s + the `TexturesDelta`. `roxlap-render`
  re-exports `egui` (under `hud`) so the host builds against the exact
  version. Pin: `egui`/`egui-wgpu` `0.29` (→ wgpu 22), `egui-winit`
  `0.29` (→ winit 0.30). `roxlap-scene-demo` shows a live FPS/pose HUD
  (toggle with `F1`).

- **`roxlap-sdl-demo`** — an SDL2 host demo (WASD + mouse-look fly
  camera over a small voxel scene) that drives the exact same
  `SceneRenderer` as the winit `roxlap-scene-demo`, proving the windowing
  decoupling end-to-end. Includes a `Send + Sync` raw-handle adapter
  pattern for window providers (like SDL) whose window type is
  `!Send`/`!Sync`. `nix develop` now provides `SDL2`.

### Changed

- **`render` no longer presents — split into `render` + `present`.** To
  let a host slot a UI pass between the world and the swap, `SceneRenderer::
  render` now *composites without presenting* and the frame is finished
  by exactly one of `present` (no overlay) or `paint_egui` (egui
  overlay). The CPU backend composites into an owned framebuffer; the GPU
  backend acquires-but-defers the swapchain frame (`GpuRenderer` gained
  `render_clear_deferred` + `present`; `render_scene` no longer presents).
  **Migration:** add a `renderer.present()` call after each
  `renderer.render(...)` (or use `paint_egui`). Hosts that don't draw a
  UI need only the one extra call.

- **`roxlap-render` / `roxlap-gpu` decoupled from winit.** The window
  binding was nominal — both backends only ever needed the window's
  `raw-window-handle` traits (softbuffer + wgpu) plus its pixel size.
  `SceneRenderer::new` and `GpuRenderer::new` / `new_blocking` are now
  generic over any `W: HasWindowHandle + HasDisplayHandle + Send + Sync
  + 'static` (winit, SDL, GLFW, a custom surface, …) and take the
  initial framebuffer size as an explicit `(u32, u32)` argument instead
  of calling `winit`'s `inner_size()`. The CPU backend tracks its size
  from `resize()` (which hosts already call on resize events) rather
  than polling the window each frame. `roxlap-render` re-exports
  `HasWindowHandle` / `HasDisplayHandle` so hosts need no direct
  `raw-window-handle` dependency. winit moves to a dev-dependency
  (examples/doctests only). **Migration:** pass the window size to
  `new`, e.g. `SceneRenderer::new(window, (w, h), &opts)`. Removed the
  unused `GpuRenderer::window()` getter.

### Fixed

- **GPU instanced sprites projected through the wrong camera.** The GPU
  sprite pass (frustum-cull/tile-bin + model-DDA) used `cameras[0]` —
  grid 0's *local* camera — to project world-space sprite instances. It
  only looked correct when grid 0 sat at identity; any non-identity
  transform on the first grid shifted every sprite by that grid's
  origin/rotation. `render_scene` now takes an explicit world
  `sprite_camera` (the facade passes the untransformed world camera) and
  both sprite sites use it. The CPU path was already correct (it draws
  sprites with the world camera directly). Also drops the sprite pass's
  `!cameras.is_empty()` guard, so sprites no longer require any grid.

- **GPU sky / fog parity with the CPU path.** The GPU backend ignored
  `FrameParams::{sky_color, fog_color, fog_max_scan_dist}` — it sampled
  its own default grey sky texture and never applied distance fog, so it
  diverged from the CPU path (grey vs. flat-colour sky, no fog vs. fog).
  `SceneRenderer::render` now mirrors `sky_color` onto a 1×1 GPU sky
  texture each frame (unless the host uploaded a real panorama via
  `set_sky_panorama`) and forwards `fog_color` / `fog_max_scan_dist` to
  the GPU marcher. (GPU fog is a smoothstep where the CPU LUT is linear
  — same endpoints, slightly different mid-curve.)

## [0.6.1] — 2026-06-10

### Changed

- **README rewrite to current scope.** The workspace README (the
  crates.io / docs.rs pitch page for every published crate, via
  `readme.workspace`) was stuck around the R10/R11 CPU-only era. It now
  covers the `roxlap-scene` scene graph (multi-grid f64 world +
  rotation, streaming, snapshots, world queries), the optional
  `roxlap-gpu` compute renderer, and the unified `roxlap-render` facade
  with screen→world picking — drops the now-inaccurate "no GPU" framing,
  adds the four missing crates to the table and `roxlap-scene-demo` to
  the quick-start, and refreshes Status (published; S1–S7 + GPU.0–13 +
  picking landed). Docs only; no code or API changes.

## [0.6.0] — 2026-06-10

A GPU outer-DDA perf win (GPU.13.0) plus a **screen→world picking and
voxel-query API** for downstream engines (turning a click — or any
world ray — into a grid + voxel + colour). Purely additive; no
public-API breakage versus 0.5.0.

### Added

- **GPU.13.0 — chunk-AABB outer-DDA early-out.** `scene_dda.wgsl`'s
  outer chunk-DDA now stops the moment a ray leaves a grid's occupied
  chunk-AABB along its travel direction, instead of stepping empty
  space to `max_outer_steps`. Cuts the high-altitude / horizon
  overscan where sky and beyond-terrain rays cross many empty chunks.
  The AABB lives in `GridStaticMeta` (112→144 bytes) and is maintained
  live — `GpuSceneResident::refresh_chunk` / `evict_chunk` recompute
  and re-upload it so streamed-in terrain is never skipped. Demo gains
  an `H` hotkey toggling a high-altitude top-down vantage for the FPS
  A/B. Render output is byte-stable (the early-out only skips empty
  space). On a 3070, the user's flagged high-altitude pose went from
  the slow case to ~600 FPS.

#### Screen→world picking + voxel queries

- **`roxlap-render`** — `SceneRenderer::pick(scene, camera, x, y) ->
  Option<PickHit>` resolves a pixel to its world point + owning grid +
  grid-local voxel in one call. Supporting API: `pixel_ray`, the
  canonical `view_ray` (returns a `Ray` — the one unproject both
  backends honour), and `pick_depth` (per-pixel world-t; CPU reads its
  z-buffer, GPU stages the depth buffer at click time). New `Ray` and
  `PickHit` types. Each backend caches its last-frame projection so
  callers never reconstruct it.
- **`roxlap-scene`** — `Scene::raycast(origin, dir, max_dist) ->
  Option<RayHit>`: a renderer-independent voxel DDA across grids
  (per-grid local-space marching, transform-correct for rotated /
  translated grids; nearest hit wins) for line-of-sight, projectiles,
  and off-screen / backend-agnostic picking. `Scene::resolve_voxel`
  maps a world surface point to its grid + voxel. `Grid::voxel_solid`
  and `Grid::voxel_color` query a grid-local voxel. New `RayHit` type.
- **`roxlap-formats`** — `Vxl::voxel_color(x, y, z)` reads a textured
  voxel's packed colour straight from the slab chain (no decompress).
- **`roxlap-gpu`** — `GpuRenderer::{read_depth_pixel, pixel_ray}` +
  the standalone `pinhole_pixel_ray`; the scene depth buffer gained
  `COPY_SRC` + a `MAP_READ` staging buffer for readback.
- **`roxlap-scene-demo`** — `C` toggles a top-down pick mode: a cursor
  sprite follows the mouse on a ground plane, and left-click resolves
  and prints the true grid + grid-local voxel.

## [0.5.0] — 2026-06-09

The **GPU compute-shader renderer** arc (GPU.0–GPU.12) plus the
**`roxlap-render`** unified renderer facade. Two new published crates
(`roxlap-gpu`, `roxlap-render`); the existing crates are unchanged
versus 0.4.2 and are version-bumped only to keep the workspace
unified. No public-API breakage.

### Added

#### `roxlap-gpu` — GPU compute-shader renderer (new crate, first publish)

- A WGPU + WGSL compute-shader voxel renderer alongside the CPU
  opticast — "approximately the same retro look, much faster", freeing
  the CPU budget for game logic. Two-level Amanatides-Woo DDA (outer
  chunk grid + inner voxel), per-chunk occupancy/colour decompress +
  upload, multi-grid scene composition with per-grid f64→f32 camera
  transforms, panoramic sky + fog, per-chunk edit/streaming
  invalidation, and a KV6 sprite path (instanced model-DDA with CPU
  frustum cull, screen-tile binning, far-LOD mips, and structural
  runtime edits).
- **GPU.11 — scene-grid LOD.** Each chunk's full mip ladder is
  uploaded (a second occupancy/colour set per level) and the marcher
  picks a mip per chunk by entry distance. ~2.15× FPS at horizon views
  on an RTX 3070 Laptop (58→125); tunable via
  `ROXLAP_GPU_MIP_SCAN_DIST` (default 64, matching the CPU
  `mip_scan_dist`).
- Sibling to the CPU path, not a replacement: the byte-exact voxlap
  oracle stays CPU-only. Falls back gracefully when no WGPU adapter is
  available.

#### `roxlap-render` — unified CPU/GPU renderer facade (new crate, first publish)

- One `SceneRenderer` over the CPU opticast (presented via
  `softbuffer`) and the GPU compute marcher (presented via `wgpu`).
  Owns presentation, the `Scene`→GPU upload / dirty-chunk refresh /
  per-grid camera transform bridge, the CPU compositor + scratch
  pool, the sprite reps (CPU draw + GPU registry + carve), and
  framebuffer capture — and **falls back to the CPU backend
  automatically** when GPU init fails (the wasm/driver-gap path).
  Hosts pick a backend with one call and render with one method.

### Fixed

#### `roxlap-gpu` — bedrock-as-solid (opaque cliff/wall faces)

- Vertical wall and cliff faces rendered as sky holes (only the
  textured top voxel showed). The decompressor dropped the implicit
  voxlap bedrock interior below a surface to save memory; it is now
  marked solid in a **second occupancy bitmap** used for hit-testing,
  while colours stay textured-only — a bedrock hit inherits the
  surface colour above it. Bedrock costs one bit, not a colour, so the
  colour array is unchanged; the empty-chunk placeholder stays air so
  floating objects (the ship) don't grow a spurious floor plane.

## [0.4.2] — 2026-06-03

Axis-aligned-mip-beams resolution + `phase_remiporend`
multi-chz reload. The beam bug that motivated the 0.4.x
mitigation cascade is gone (incidentally fixed by the VC.5 /
VC.6.2 / PRR multi-chz install path); demo mitigations
reverted to the original aggressive config. `phase_remiporend`
closes the last VC.6 follow-up from 0.4.1's Known limits.
No public-API breakage vs 0.4.1.

### Fixed

#### `roxlap-core` + `roxlap-scene-demo` — AAMB (axis-aligned-mip-beams resolution)

- **The axis-aligned-mip-beams artifact is RESOLVED.** Originally
  reported 2026-05-12: faint world-axis green columns at deep
  mip-N + near-axis-aligned rays. Multi-session investigation
  CF.0..CF.3.C in `project_cf_narrowing_multi_session_plan.md`
  concluded that cf-narrowing could not fix it. Re-audit at
  2026-06-03 finds the bug GONE — every multi-chunk beam test
  reports **0 beam pixels** at `ml=6` across `msd=8/64/256/1024`.
  Likely fixed incidentally by the VC.5 / VC.6.2 / PRR multi-chz
  install cascade (`phase_after_delete_kept_presync`'s column-step
  and `phase_remiporend`'s reload both now route through
  `build_owned_column_multi_chz`, which appears to have closed
  the corner case the beam relied on). Confirmed:
  `dump_green_beam_pose_diff` (scan=1024, msd=64, ml=6 vs ml=1)
  reports 0 beam pixels (was 6404 at S5.3); `dump_spawn_pose_diff`
  reports 0 (was 5379).
- **Demo mitigations reverted**. `roxlap-scene-demo`:
  - `SCAN_DIST_MAX` 1500 → 1024.
  - `settings.mip_levels` 4 → 6, `settings.mip_scan_dist` 128 →
    64 (the live demo now exercises the full 6-mip ladder at
    msd=64).
  - Ship grid's `mip_levels_override = Some(1)` retired — ship
    now renders with the full mip ladder. Bench engine-only
    moves from 43 FPS (mitigated, NSP 68.9 %) to 75 FPS (revert,
    NSP 34.4 %) — coarser mips render less terrain faster at
    the bench camera.
- **`crates/roxlap-core/src/cf_narrow.rs` deleted** (~523 LOC) —
  the rejected experiment + 9 unit tests retired. The three env-
  var gates (`ROXLAP_CF_NARROW`, `ROXLAP_CF_NARROW_PER_COLUMN`,
  `ROXLAP_CF_NARROW_PER_COLUMN_NO_I1`) and their LazyLock caches
  also gone from `grouscan.rs`.
- **AAMB.1 single-chunk multi-mip crash fixed**. The audit
  surfaced a pre-existing arithmetic underflow at
  `grouscan.rs::phase_remiporend` where
  `state.ixy_sptr_col_idx - state.mip_base_offsets[old_mip]`
  panicked for axis-aligned poses on single-chunk grids at
  `msd ≥ 64, ml = 6` (negative `cy_mip` masking into a column
  index that landed just below the mip-OLD sub-table boundary).
  Fix: defensive bail to `Phase::Startsky` when the index slips
  below `mip_old_base`. `axis_aligned_single_chunk_multi_mip_*`
  and `axis_aligned_single_chunk_pitched_up` flip from PANIC to
  PASS; multi-chunk beam tests stay green.

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
[`PORTING-SCENE.md`](docs/porting/PORTING-SCENE.md).

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
[`PORTING-SCENE.md`](docs/porting/PORTING-SCENE.md).

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

[0.2.0]: https://github.com/NCrashed/roxlap/compare/v0.1.1...v0.2.0

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
    [`PORTING-MULTICORE.md`](docs/porting/PORTING-MULTICORE.md).

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

[Unreleased]: https://github.com/NCrashed/roxlap/compare/v0.28.0...HEAD
[0.28.0]: https://github.com/NCrashed/roxlap/compare/v0.27.0...v0.28.0
[0.27.0]: https://github.com/NCrashed/roxlap/compare/v0.26.0...v0.27.0
[0.26.0]: https://github.com/NCrashed/roxlap/compare/v0.25.0...v0.26.0
[0.25.0]: https://github.com/NCrashed/roxlap/compare/v0.24.0...v0.25.0
[0.24.0]: https://github.com/NCrashed/roxlap/compare/v0.23.0...v0.24.0
[0.23.0]: https://github.com/NCrashed/roxlap/compare/v0.22.0...v0.23.0
[0.22.0]: https://github.com/NCrashed/roxlap/compare/v0.21.0...v0.22.0
[0.21.0]: https://github.com/NCrashed/roxlap/compare/v0.20.0...v0.21.0
[0.20.0]: https://github.com/NCrashed/roxlap/compare/v0.19.0...v0.20.0
[0.19.0]: https://github.com/NCrashed/roxlap/compare/v0.18.0...v0.19.0
[0.18.0]: https://github.com/NCrashed/roxlap/compare/v0.17.0...v0.18.0
[0.17.0]: https://github.com/NCrashed/roxlap/compare/v0.16.0...v0.17.0
[0.16.0]: https://github.com/NCrashed/roxlap/compare/v0.15.0...v0.16.0
[0.15.0]: https://github.com/NCrashed/roxlap/compare/v0.14.0...v0.15.0
[0.14.0]: https://github.com/NCrashed/roxlap/compare/v0.13.0...v0.14.0
[0.13.0]: https://github.com/NCrashed/roxlap/compare/v0.12.0...v0.13.0
[0.12.0]: https://github.com/NCrashed/roxlap/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/NCrashed/roxlap/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/NCrashed/roxlap/compare/8263621...v0.10.0
[0.9.0]: https://github.com/NCrashed/roxlap/compare/v0.8.0...8263621
[0.8.0]: https://github.com/NCrashed/roxlap/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/NCrashed/roxlap/compare/v0.6.1...v0.7.0
[0.6.1]: https://github.com/NCrashed/roxlap/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/NCrashed/roxlap/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/NCrashed/roxlap/compare/v0.4.2...v0.5.0
[0.4.2]: https://github.com/NCrashed/roxlap/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/NCrashed/roxlap/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/NCrashed/roxlap/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/NCrashed/roxlap/compare/v0.2.0...v0.3.0
