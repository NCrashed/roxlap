# roxlap — engine quality review & improvement plan (Stage QE)

Full-engine review taken 2026-07-03 at workspace 0.21.0 (~72k LOC Rust +
~3.1k LOC WGSL across 12 crates). Four parallel review passes: public API
ergonomics, core/GPU architecture, data/content layer, and newcomer
onboarding. This is the **entry doc** for the quality/ergonomics stage —
tag **QE**. A fresh-context session should read it top to bottom before
touching code.

## API stability policy for this stage — LOCKED

Decision taken with the engine author 2026-07-03:

**Breaking API changes are allowed now.** We are pre-1.0 and the user base
is small; this stage is the right time to fix API shape rather than
accrete compatibility shims. The rules:

1. **Every breaking change must ship with a migration path.** The
   CHANGELOG entry (and, for large breaks, a section in this doc) must
   contain an explicit *old → new* mapping: the removed/renamed item, the
   replacement call, and a copy-pasteable before/after snippet.
2. **Old behaviour must stay reachable.** If a default changes or a
   behaviour is removed, the migration notes must say exactly how to get
   the previous behaviour back (a constructor flag, an options field, an
   explicit call). "Upgrade and re-tune" is not acceptable; "set
   `RenderOptions.x = old_value`" is.
3. **Deprecate-in-place where cheap.** When a `#[deprecated(note =
   "use X")]` forwarding shim costs a few lines, keep it for one minor
   release so downstream code gets a compiler hint instead of a hard
   break. When the shim would be structural (e.g. `FrameParams` field
   layout), break cleanly and document per rule 1.
4. **Byte-identical render output is NOT required** across QE changes,
   but visual parity is: golden scenes may be re-frozen with a note, and
   any intentional visual change must be called out in the CHANGELOG.

---

## Verdict

The codebase is in better shape than its size suggests. Crate layering is
disciplined (`formats → core → scene`; `roxlap-gpu` depends only on
`formats` + wgpu; only `roxlap-render` knows both scene and gpu). Unsafe
is small and quarantined in two rayon-parallelism sites. The legacy asm
port is isolated behind clean seams (`GridView`, `RasterTarget`,
`OpticastSettings`). 699 tests, ~100% rustdoc coverage on the public
surface, byte-exact round-trips for every file format.

The debt falls into three classes:

- **(A) "engine demo → real game" gaps** — streaming loses player edits,
  snapshots are voxel-only and unversioned, no `.vox` import.
- **(B) facade API debt** — `SceneRenderer` is a ~93-method god object
  with sentinel-handle errors and an implicit multi-call tick protocol.
- **(C) fresh architecture debt** — duplicated CPU/GPU bookkeeping in the
  facade and dirty-flag sprawl from the PF series. Cheapest to fix now;
  gets more expensive every release.

---

## P0 — game blockers (class A)

### QE-A1. Streaming eviction silently destroys player edits

`evict_grid_chunks_with_cam` (`roxlap-scene/src/lib.rs:1054-1090`)
unconditionally removes any chunk past `r_evict`, including its
`chunk_versions` entry. Player digs/builds, walks away, comes back — the
world has reverted. The `ChunkGenerator` docs even state "evict +
re-stream is sound under no-persistence" (`streaming.rs:43-44`) — sound
only if nobody edited the chunk.

Fix: a `ChunkStore` beside `ChunkGenerator` — an eviction hook
(`on_evict(chunk_idx, &Vxl, version)`) so dirty chunks (version ≠ 0) can
be persisted, and a load-side "restore from store vs generate" check.
Also integrate with snapshots: today `Scene::to_snapshot` saves only
materialised chunks, so a streamed world's save depends on where the
camera stood.

### QE-A2. Scene snapshots: no wire version, voxel-only coverage

`SceneSnapshot`/`GridSnapshot` (`snapshot.rs:84-107`) are bare serde
derives over bincode: no magic, no version field. `#[serde(default)]`
does NOT provide evolution in positional bincode — a pre-S7.2 save fed to
the current struct is undefined/format-dependent and untested. Coverage:
grid transforms + chunk bytes + versions, nothing else. Per-grid config
(`render_sky`, `mip_levels_override`, `stream_radius`, generator, LOD
thresholds) resets to defaults on restore; the entire roxlap-render layer
(sprites, clips, actors, lights, materials, fog/sky, pipeline settings)
is not serializable at all. You can save *terrain*, not a *game*.

Fix, in order: (1) versioned envelope — magic + explicit `u32` version +
a checked-in old-version fixture test; (2) extend `GridSnapshot` with
grid config; (3) either a facade-level `RendererSnapshot` companion or
user-assignable string tags on grids/instances so hosts can rebind their
own save data (slotmap ids are runtime-only today).

### QE-A3. No MagicaVoxel `.vox` import

Zero hits for `.vox`/`dot_vox` anywhere. MagicaVoxel is the de-facto
voxel authoring tool; without it every asset goes through ancient voxlap
tooling or code. The `dot_vox` crate plus existing `Kv6::from_fn` /
`Vxl::from_dense` make this ~a day of work. Highest-leverage single
content-pipeline feature.

### QE-A4. Parser hardening (hostile input)

Truncation is handled everywhere (shared bounds-checked cursor in
`bytes.rs` — good), but:

- **Allocation bombs**: `Vec::with_capacity(untrusted count)` in every
  format except `.vxl` — `kv6.rs:552`, `kfa.rs:147-163`,
  `character.rs:512+`, `voxel_clip.rs:1402-1404` (up to 16 GB despite the
  64 MiB inflate cap). Fix pattern already in-house: `.vxl`'s
  `FileTooLarge` gate; use `n.min(cursor.remaining()/elem_size)`.
- **Loader hang / OOB panic on malformed skeletons**: `Hinge::parent` is
  an unchecked `i32` (`kfa.rs:257`); cyclic parents make `sort_hinges`
  (`kfa.rs:600-622`) loop forever; `parent >= len` panics OOB at
  `kfa.rs:612`. Validate range + acyclicity at parse.
- **`.rvc` META dims unvalidated** (`voxel_clip.rs:1412-1420`) —
  `cols * owpc` can overflow usize on wasm32 or attempt huge allocs.
- **RKC cross-reference indices unvalidated** (`character.rs:547`) —
  the doc claim that parse keeps mesh indices in range is false.
- **No fuzzing**: all parsers are `&[u8] → Result`, so `cargo fuzz`
  harnesses are nearly free and would regression-guard the above.

Matters the moment mod support / downloadable content exists.

---

## Facade API — ergonomics (class B; breaking changes welcome per policy)

### QE-B1. `SceneRenderer` god object: ~93 methods, 6 hand-rolled slotmaps

`DynInstanceMap`, `DynModelMap`, `DynClipMap`, `CharMap`,
`StreamingClipMap`, `BillboardActorMap` (`roxlap-render/src/lib.rs:140`,
`:195`, `:265`, `:318`, `:386`, `:864`) — the last three are
byte-identical epoch slotmaps. One generic `SlotMap<T>` deletes ~250
lines and stops the copy-per-feature trend (SL, BB, TV each added one).
Longer term: typed handles with methods (`clips().add(…)` →
`ClipHandle::spawn(…)`) instead of 93 flat methods.

### QE-B2. Implicit 5-call tick protocol → one `renderer.tick(camera, dt)`

Hosts must call, in order, before `render`: `advance_voxel_clips(dt)`,
`update_billboard_actors(camera, dt)`, `face_billboards_to(camera)`,
`advance_character(id, dt)` per character, `update_kfa_poses(…)`. A
missed call is a silent visual bug. A single `tick(camera, dt)` subsuming
all collection-level updates is the highest-value single API addition —
non-breaking, keep the fine-grained calls for hosts that need them.

### QE-B3. `FrameParams`: breaking-change magnet + backend grab bag

12 public fields, no `Default` (holds `&OpticastSettings`), not
`#[non_exhaustive]` (`lib.rs:916-964`) — every added field broke every
host's struct literal. Mixes CPU-only (`settings`, fog fields,
`treat_z_max_as_air`) and GPU-only (`gpu_fov_y_rad`,
`gpu_mip_scan_dist`, `gpu_max_outer_steps`) knobs. Worst: **FOV has two
unrelated sources** — CPU from `OpticastSettings::hx/hy/hz`, GPU from
`gpu_fov_y_rad`; same host code renders different fields of view per
backend.

Fix (breaking, per policy): `#[non_exhaustive]` +
`FrameParams::new(settings)` constructor with builder-style setters; one
shared `fov_y` deriving both projections; move `gpu_*` tuning into
`RenderOptions`/`GpuRendererSettings`. Migration note: list every field's
new home; old per-frame literal → `FrameParams::new(settings)
.fov_y(f).sky(…)…`.

### QE-B4. Error model: sentinels, silence, no error type

- Spawn methods return `SpriteInstanceId{u32::MAX,u32::MAX}` on failure
  (`add_sprite_instance_posed` :2074, `add_clip_instance_posed` :2453,
  `add_billboard_actor` :2627, `add_streaming_clip_instance` :2945) — a
  misconfigured actor silently doesn't exist. → return `Option`/`Result`
  (breaking; migration: wrap old call in `.unwrap_or(STALE)` to keep old
  semantics).
- No error enum in roxlap-render at all; `SceneRenderer::new` documented
  "Never fails" but the CPU fallback `expect`s on softbuffer
  (`cpu.rs:610-611`) and every present can panic (`cpu.rs:2002-2005`).
  GPU-init failure is reported via `eprintln!` only. → `RenderError` +
  `try_new` (keep infallible `new` as the wrapping fallback), route
  diagnostics through `log`.
- `bool` returns applied inconsistently across setters (`set_actor_state`
  returns bool, `set_actor_transform` returns `()`); pick one rule.

### QE-B5. Backend parity: silent no-ops, no capability query

| Capability | CPU | GPU | Today |
|---|---|---|---|
| `request_capture`/`take_capture` | ✅ | ❌ | silent no-op — **screenshots impossible on the GPU backend** |
| `set_sky_panorama` | ❌ | ✅ | silent no-op; sky configured two different ways per backend |
| `carve_active_sprite` | ❌ (0) | ✅ | silent 0 |
| translucent sprite/terrain materials | ✅ | ⚠ pending | silent visual divergence |
| `FrameParams.lights` | ✅ | ✅ | **docs stale**: `light.rs:1-10` + `lib.rs:957-963` still say "GPU-only" — false since CPU.1/2 |

Fix: implement GPU capture (readback of the resolve target — well-trodden
wgpu); add `supports(Feature) -> bool` or `Result<(), Unsupported>`
returns; fix the stale lights docs; put a parity table in crate docs.

### QE-B6. Small recurring API warts (batch fixable, mostly breaking)

- `speed_q8: i32` in 6 signatures + `BillboardActorDef` — internal Q8
  leak; → `speed: f32` (migration: `speed = q8 as f32 / 256.0`).
- Bool-pair `set_sprite_instance_shadow_flags(id, true, false)`
  (`lib.rs:2332`) → `ShadowFlags` bitflags/struct.
- Four `_with_materials` method variants (`:2016`, `:2183`, `:2392`,
  `:2902`) → options param with `Default`.
- `get_clip_instance_frame` (`:2830`) — the only `get_` prefix in the
  crate → `clip_instance_frame`.
- `ActorState.name: &'static str` (`:778`) — actor defs can't come from
  data files without `Box::leak` → `String`/`Cow`.
- **Four colour packings** (`Line3` `0xAARRGGBB`, tint `0x00RRGGBB`,
  voxlap BGRA w/ brightness-in-alpha, `colfunc → i32`) → `PackedColor`
  newtype family.
- Env-only tuning `ROXLAP_GPU_CLIP_BUDGET`/`ROXLAP_GPU_CHUNK_BUDGET`
  (`roxlap-render/src/gpu.rs:215,237`) → `RenderOptions` fields (env vars
  stay as user-side overrides).
- `RenderOptions.want_gpu: bool` can't express "GPU or fail" →
  `enum BackendPreference { Cpu, PreferGpu, RequireGpu }`.
- `Grid::bake_lightmode(u32)` magic number (`chunks.rs:214`) →
  `enum BakeMode`; note `bake_lightmode_bbox` silently ignores AO params
  (`chunks.rs:344`).
- `ImageId` is positional, non-generational (slot reuse aliasing admitted
  in docs `lib.rs:1723-1726`) — every other handle family solved this.
- `SpriteInstanceDesc.model: usize` raw index (`lib.rs:93-96`) vs
  `SpriteModelId` everywhere post-`set_sprites`.
- `set_sprites` nukes every handle family (`:1965-1979`) — split static
  world sprites from the dynamic registry, or at least return a token
  making the invalidation explicit.
- The `render → overlays → present`/`paint_egui` protocol is enforced by
  docs only; a `Frame` guard object would make misuse unrepresentable.
- Doc corruption: `add_sprite_model` doc spliced mid-sentence into
  `define_material`'s (`lib.rs:2121`/`:2163`).

---

## Architecture debt (class C — fix while small)

### QE-C1. Facade CPU/GPU bookkeeping duplication

~200-250 LOC identical between `cpu.rs` (2410 LOC) and `gpu.rs` (1899
LOC): clip detach/append/frame ops, tints, shadow/lighting flags,
materials state (parallel sites e.g. `remove_voxel_clip` cpu.rs:909-920
vs gpu.rs:626-649). Cost is not LOC but **three coordinated edits per
feature** (facade arm + cpu + gpu), and the copies already drifted (GPU
tracks `materials_dirty`/`transforms_dirty`; CPU doesn't).

Fix: NOT a `dyn RenderBackend` rewrite. Extract shared `SceneState`
(materials, terrain materials, `dyn_clip` mappings, clip metadata) owned
by `SceneRenderer`; backends shrink to genuinely divergent ops
(upload/refresh/draw/present) receiving `&SceneState` + small
`DirtyFlags`. Keep enum dispatch. Caveats: both backend signatures must
take `&mut Scene` (CPU needs it); CPU-only caches (PF.8 dense decode)
stay in `CpuBackend`.

### QE-C2. Dirty-tracking flag soup (5 overlapping mechanisms)

`Grid::chunk_versions` (bumped from 4 files), `Grid::chunk_dirty`,
`Grid::mutations`, `GpuBackend::versions` mirror, `GpuBackend::
grid_mutations` (updated only on *complete* sync — discipline-only
invariant, gpu.rs:1551) + per-frame GPU booleans (`sprite_lights_dirty`
cleared only inside the *conditional* sprite pass,
`roxlap-gpu/src/lib.rs:3002`).

Fix: one `GridDirtyState` on `Grid` with a single
`invalidate_chunks(chunk, extent)` entry point + grid-level generation
counter; consumers hold a typed `LastSync` snapshot (replaces the
mirror + `grid_mutations`). GPU-side: fold booleans into a `FrameDirty`
struct with `begin_frame()/end_frame()`. Do this BEFORE the next perf
series adds a sixth counter.

### QE-C3. `roxlap-gpu/src/lib.rs` monolith (5,685 lines) + dead pipelines

`render_scene` ~529 lines (:2675), `build_scene_dda` ~422 (:3990).
`chunk_dda.wgsl`/`grid_dda.wgsl` pipelines (`render_chunk` :2068,
`render_grid` :2385) are dead in production — GPU.0/GPU.3 baselines
superseded by scene_dda; only tests/examples call them. Delete or
feature-gate them, then split lib.rs by pass (`init` / `scene_pass` /
`sprite_pass` / `overlay` / `resolve` / `shader_source`); split the
53-field `GpuRenderer` struct along the same seams.

### QE-C4. WGSL duplication, no include mechanism

13 shaders / 3,062 lines. Sky code ×4 copies, occupancy lookup ×5, light
loop drifted between `scene_dda.wgsl:732-770` and
`sprite_model_dda.wgsl:329-360`, camera setup ×2. The splice mechanism
already exists (`sprite_shader_source` :5459 / `scene_shader_source`
:5484 marker blocks) — generalise to build-time concatenation of shared
snippets (`sky.wgsl`, `camera.wgsl`, `occupancy.wgsl`, then
`lighting.wgsl`). Deleting QE-C3's dead shaders removes half the copies
for free.

### QE-C5. Test blind spots: the facade and CPU/GPU parity

699 tests, but: **roxlap-render has no `tests/` dir at all** — exactly
where QE-C1's duplication lives; no CPU↔GPU frame-diff harness (GPU test
comment admits "exact CPU parity needs visual inspection"); exactly ONE
golden-hash render oracle (`render.rs:2603`) guards the whole CPU
composed path; CI never executes the GPU path (adapter-gated skip).

Fix: facade integration suite driving add/remove/retarget cycles on the
CPU backend; CPU-vs-GPU pixel-delta harness gated on adapter presence;
freeze 3-5 golden scenes (lit, transparent, multi-grid).

### QE-C6. Env-var config sprawl

21 ad-hoc `ROXLAP_*` vars parsed at scattered call sites. History lesson
(PR.1): uncached `env::var_os` in hot loops once cost 3× FPS — every new
scattered env read is a latent hazard. Fix: one `RenderConfig` struct,
`from_env()` called once at construction, threaded into backends; gives
API users programmatic control of env-only knobs.

### QE-C7. Cheap hygiene

`// SAFETY:` contracts on the two `unsafe impl Send/Sync`
(`raster_target.rs:36,44`, `world_lighting.rs`) stating the disjointness
invariants — guards future tile-range refactors.

---

## Onboarding — quick wins (~a day, no engine changes)

1. **README's only code example calls deleted API** (`README.md:201-213`
   — `ScratchPool::new_parallel`, `DrawTarget::new`,
   `draw_sprites_parallel` don't exist anywhere) and reaches every
   crates.io page via `readme.workspace = true`. Delete/replace.
2. **No quickstart anywhere.** Add a "Use it in your game" README section
   (the two `Cargo.toml` lines: `roxlap-render` + `roxlap-scene`) plus
   `roxlap-render/examples/quickstart.rs` (CI-compiled, can't rot).
   Today's minimal hello-world is ~120-150 lines, dominated by winit
   boilerplate + a 14-line `FrameParams` literal; with QE-B3 and a
   template, ~40 lines is reachable.
3. README docs.rs links point only at roxlap-core/formats — the two
   crates NOT to start from; add render/scene first.
4. Move the 16 `PORTING-*.md` files (7,811 lines) to `docs/porting/`;
   update README + rustdoc references.
5. `missing_docs = "warn"` in `[workspace.lints.rust]` — passes today,
   locks in the crate's best asset.
6. Fix broken doc links (`roxlap_core::rasterizer::ScratchPool`,
   `roxlap-render/src/lib.rs:1122,1127`), stale "RF.0 skeleton" /
   "future stages will add…" crate docs (`roxlap-scene/src/lib.rs:8-25`,
   `roxlap-render/src/lib.rs:24-26`), abandoned CHANGELOG link refs
   (only 4 of 26 versions defined).
7. Switch all three demos to `Camera::from_yaw_pitch` — they hand-roll
   exactly the basis code `roxlap-core/src/camera.rs:60-136` warns about
   (chirality footgun; newcomers copy demos).
8. `rust-toolchain.toml` pins nightly repo-wide but only wasm-threads
   needs it — scope the pin; `cargo build --workspace` fails without
   system SDL2 (`default-members` excluding roxlap-sdl-demo, or
   `sdl2/bundled`).
9. `roxlap-host` naming trap: it's a legacy demo binary, not a host
   helper. Rename or document; consider an optional `run(app)` winit
   wrapper crate (removes the single largest LOC tax on hosts).
10. `impl` a `FrameParams` constructor even before the full QE-B3 break.

---

## Suggested order of work

| Phase | Contents | Breaking? |
|---|---|---|
| QE.0 | Onboarding quick-win batch (items 1-10 above) | no |
| QE.1 | `renderer.tick()` + generic `SlotMap<T>` dedup + spawn methods → `Option` | tick/slotmap no; spawns yes (mechanical) |
| QE.2 | `FrameParams::new` + `#[non_exhaustive]` + unified FOV; `RenderError`/`try_new`; budgets → `RenderOptions` | yes — biggest migration note |
| QE.3 | `SceneState` extraction (QE-C1) + dirty-tracking centralisation (QE-C2) | internal |
| QE.4 | Facade test suite + CPU/GPU diff harness + golden scenes (QE-C5) | no |
| QE.5 | Streaming persistence (`ChunkStore`, QE-A1) + versioned snapshot (QE-A2) | snapshot wire format yes |
| QE.6 | `.vox` import (QE-A3) + parser hardening + fuzz targets (QE-A4) | no |
| QE.7 | GPU capture + `supports()` (QE-B5); B6 wart batch | B6 yes |
| QE.8 | roxlap-gpu split + WGSL snippets (QE-C3/C4) | internal |

Phases QE.1-QE.3 are the ones that get more expensive every release they
wait. Each phase lands as its own commit series with CHANGELOG migration
notes per the policy above.
