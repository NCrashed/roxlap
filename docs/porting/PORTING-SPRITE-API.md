# roxlap — SceneRenderer sprite API: incremental models + per-instance transforms

Start-of-stage brief and locked decisions for **surfacing dynamic
sprite-model management and per-instance orientation through the
`SceneRenderer` facade**, so a physics-driven demo (rotating
asteroids/debris/projectiles that stream in and out) can run entirely
through `SceneRenderer` without bypassing it to the `GpuRenderer`.
Companion to [PORTING-GPU.md](PORTING-GPU.md) (the GPU backend) and
[PORTING-SCENE.md](PORTING-SCENE.md) (the scene layer).

This document is a **start-of-stage brief**. A fresh-context session
should read it top to bottom before touching code. The change is
**purely additive** — no existing public signature changes.

## Why

`SceneRenderer` is the facade over the CPU (software 3D-DDA) and GPU
(wgpu) backends. The `GpuRenderer` already has the full machinery for
dynamic sprite-model lifecycle and per-instance transforms, but the
facade only exposes a subset, forcing physics-driven callers to drop
to `GpuRenderer` directly. Four gaps (reported by an engine user
migrating a demo to the scene API):

1. **No per-frame orientation.** `add_sprite_instance` places by
   position only; there is no way to update a placed instance's
   rotation per frame. `GpuRenderer::update_sprite_instance_transforms`
   exists but is not surfaced. → rotating objects impossible without
   bypass.
2. **No incremental model registration.** Only `set_sprites(SpriteSet)`
   (full replace). Each procedurally-generated asteroid is a unique
   model; there is no `add_sprite_model` to append one to the live set.
3. **No model removal.** `remove_sprite_instance` exists, but nothing
   to remove a registered *model* when an asteroid leaves range.
4. **No `compact_sprite_models`.** No facade equivalent to reclaim GPU
   memory from tombstoned models.

Plus an underlying constraint: **`SpriteModelRegistry`
(`roxlap-gpu/src/sprite_model.rs:160`) is append-only** — no `remove`,
no `compact`. Removal/compaction today live only on the GPU-resident
side; the CPU-side registry `entries`/`chains` grow unboundedly.

## Locked decisions

Taken with the engine author 2026-06-25:

1. **Model handle = slotmap + generation.** `SpriteModelId` becomes
   `{ slot: u32, gen: u32 }`, mirroring `SpriteInstanceId`. A stale
   handle after removal resolves to `None` → safe no-op. (Slots are
   tombstoned in place and **never reused** because GPU chain ids are
   append-only, so `gen` stays 0 today; it is wired for a future
   compacting registry and keeps the two handle types symmetric.)
2. **Reclamation = in-place free, keep slots.** New
   `SpriteModelRegistry::remove(chain_id)` drops the dead chain's
   voxel data (replace each entry's `SpriteModel` with an empty
   placeholder) but keeps the slot → chain ids stay stable, **no id
   remap**. `compact_sprite_models()` repacks the GPU buffers via the
   existing `gpu.compact_sprite_models(registry)` (dead-aware; reads
   only live entries).
3. **Single PR.** All four facade methods + `DynSpriteTransform` +
   batch variant + registry free, together. The demo needs all of it.
4. **Include posed add.** Besides `set_sprite_instance_transform`, add
   `add_sprite_instance_posed(model, DynSpriteTransform)` so an
   instance can spawn pre-rotated in one call (no one-frame
   axis-aligned flash for objects spawned mid-flight).

## Code map (as of 2026-06-25)

Facade — `crates/roxlap-render/src/lib.rs`:
- `SceneRenderer { inner: BackendImpl, dyn_map: DynInstanceMap }`
  (`:553`), dispatch enum `BackendImpl::{Cpu,Gpu}` (`:546`).
- `SpriteModelId(pub(crate) usize)` (`:88`) — positional, minted only
  in `set_sprites` (`:951`).
- `SpriteInstanceId { slot, gen }` (`:99`) + `DynInstanceMap` (`:109`)
  — the model for the new `DynModelMap`.
- existing sprite methods: `set_sprites` (`:951`),
  `refresh_sprite_model` (`:981`), `add_sprite_instance` (`:998`),
  `remove_sprite_instance` (`:1010`), `dynamic_sprite_count` (`:1025`).
- mirror pattern: `match &mut self.inner { Cpu(c)=>…, Gpu(g)=>… }`.

GPU backend — `crates/roxlap-render/src/gpu.rs`:
- `GpuBackend` fields: `sprite_registry: Option<SpriteModelRegistry>`,
  `sprite_instances: Vec<SpriteInstance>`, `sprite_basis: Vec<Sprite>`,
  `sprite_models_tpl: Vec<Sprite>`, `dyn_count`,
  `sprite_model_ids: Vec<u32>` (host model index → chain id) (`:43`).
- `add_dyn_instance` (`:239`), `remove_dyn_instance` (`:265`).

CPU backend — `crates/roxlap-render/src/cpu.rs`:
- `CpuBackend` fields: `sprites`, `sprite_models`, `models: Vec<Sprite>`
  (templates), `dyn_sprites`, `dyn_models`, `kfa_limbs` (`:286`).
- `set_sprites` (`:462`), `add_dyn_instance` (`:485`),
  `remove_dyn_instance` (`:500`), `update_sprite_model` (`:516`).
- draws `Sprite`s directly via `draw_sprite_dda`
  (`roxlap-core/src/dda_sprite.rs:185`); degenerate-basis guard at
  `dda_sprite.rs:90`.

GPU lib — `crates/roxlap-gpu/src/lib.rs`:
- `add_sprite_model(registry, chain_id)` (`:3342`, zero-instance upload
  path establishes residency if none yet),
  `remove_sprite_model(chain_id)` (`:3369`),
  `compact_sprite_models(registry)` (`:3380`),
  `append_sprite_instances(registry, &[…]) -> u32` (`:3296`),
  `remove_sprite_instance(index) -> Option<usize>` (`:3316`),
  `update_sprite_instance_transforms(&[…])` (`:3416`, **full ordered
  slice**, zip-by-position).

GPU registry/resident — `crates/roxlap-gpu/src/sprite_model.rs`:
- `SpriteModelRegistry { entries, chains }` (`:160`) — append-only;
  `add`/`add_lod`/`fork`, `model`, `len`. **No remove/compact.**
- `SpriteInstanceTransform { inv_rot:[[f32;4];3], pos }` (`:124`),
  `from_sprite` inverts `[s|h|f]` (`:139`).
- `SpriteRegistryResident` (`:620`): `remove_model(chain_id)` (`:1283`,
  tombstones `dead[e]`, frees colors slots, empties chain),
  `compact(device,queue,registry)` (`:1307`, dead-aware repack;
  preserves ids), `update_transforms(&[…])` (`:922`, writes `cull`).

`Sprite` (voxlap convention) — `crates/roxlap-formats/src/sprite.rs:52`:
`{ kv6, p:[f32;3], s:[f32;3], h:[f32;3], f:[f32;3], flags }`. `s/h/f`
are the model→world basis **columns** (local +x/+y/+z). Both backends
just invert `[s|h|f]`; **chirality is irrelevant for sprites** (the
`right × down == forward` rule is camera-only). Only hard constraint:
**det ≠ 0**.

## New public surface — `roxlap-render/src/lib.rs`

```rust
/// Orientation + position for a dynamic sprite instance.
/// `right`/`up`/`forward` are the instance's local axes in world space
/// (the columns of the model→world rotation). Must be non-singular
/// (det ≠ 0); need not be orthonormal. Defaults to identity.
#[derive(Clone, Copy, Debug)]
pub struct DynSpriteTransform {
    pub pos: [f32; 3],
    pub right:   [f32; 3], // ↦ Sprite.s  (kv6 local +x)
    pub up:      [f32; 3], // ↦ Sprite.h  (kv6 local +y)
    pub forward: [f32; 3], // ↦ Sprite.f  (kv6 local +z)
}
impl Default for DynSpriteTransform { /* identity basis */ }

impl SceneRenderer {
    pub fn add_sprite_model(&mut self, kv6: &Kv6) -> SpriteModelId;
    pub fn remove_sprite_model(&mut self, id: SpriteModelId) -> bool; // false on stale
    pub fn compact_sprite_models(&mut self);

    pub fn add_sprite_instance_posed(
        &mut self, model: SpriteModelId, xf: DynSpriteTransform,
    ) -> SpriteInstanceId;

    pub fn set_sprite_instance_transform(
        &mut self, id: SpriteInstanceId, xf: DynSpriteTransform,
    ); // no-op on stale
    pub fn set_sprite_instance_transforms(
        &mut self, updates: &[(SpriteInstanceId, DynSpriteTransform)],
    );
}
```

`right/up/forward → s/h/f` is a direct copy. A near-singular basis
falls through the existing degenerate guards (`dda_sprite.rs:90`,
`mat3_inverse` at `sprite_model.rs:140`) → instance silently skips
rather than panics. Document this.

### Handle change — `SpriteModelId`

`SpriteModelId` (fields already `pub(crate)`, so externally
non-breaking) becomes `{ slot: u32, gen: u32 }`. Add a `DynModelMap`
next to `DynInstanceMap` — same shape **minus** the swap-remove
`moved` fixup (model slots tombstone in place, never reused).
`set_sprites` and `refresh_sprite_model` route through it; their public
signatures are unchanged.

## Backend work

### GPU backend (`gpu.rs`)
- `add_model(kv6) -> usize`: lazily create empty registry + resident
  if `sprite_registry` is `None`; `chain = registry.add_lod(
  build_sprite_model(kv6), 4)`; `gpu.add_sprite_model(&registry,
  chain)`; push `sprite_model_ids`/`sprite_models_tpl`; return host idx.
- `remove_model(host_idx)`: `chain = sprite_model_ids[host_idx]`;
  **`gpu.remove_sprite_model(chain)` first** (sets resident `dead`),
  then `registry.remove(chain)` (frees voxel data); mark template slot
  dead in place. Ordering matters — assert/comment it.
- `compact_models()`: `gpu.compact_sprite_models(&registry)`.
- `add_dyn_instance_posed(model_idx, xf)`: like `add_dyn_instance`
  (`:239`) but write `s/h/f` from `xf`; old `add_dyn_instance` becomes
  an identity-basis caller.
- **Transform flush (perf-critical):**
  `update_sprite_instance_transforms` takes the *whole ordered slice*
  (`sprite_model.rs:922`) → a naïve per-instance setter is O(n) each
  call → O(n²)/frame. Instead: setter mutates
  `self.sprite_instances[gpu_index].transform` + sets a
  `transforms_dirty` flag; `render()` flushes once via the full-slice
  call when dirty (the per-frame `cull_bin_upload` already re-reads
  `cull`, so no extra GPU upload). Batch setter mutates all, flips the
  flag once. *Follow-up (not this PR): a targeted
  `update_transform(index, …)` on the resident to drop the O(n)
  memcpy.*

### CPU backend (`cpu.rs`)
- `add_model(kv6) -> usize`: push an axis-aligned `Sprite` template
  (kv6 clone) into `self.models`; return index.
- `remove_model(host_idx)`: replace `models[host_idx]` with an empty
  placeholder, keep slot (parity with GPU tombstone; existing
  instances keep their own kv6 clones and draw until instance-removed).
- `compact_models()`: no-op.
- `add_dyn_instance_posed(model_idx, xf)`: clone template, set
  `p/s/h/f` from `xf`.
- `set_dyn_instance_transform(idx, xf)`: O(1) write of `p/s/h/f` on
  `dyn_sprites[idx]`.

### GPU registry (`roxlap-gpu/src/sprite_model.rs`)
- `SpriteModel::empty()` (or `clear()` freeing the volume `Vec`s).
- `SpriteModelRegistry::remove(&mut self, chain_id)`: for each entry in
  `chains[chain_id]`, swap `entries[e]` to an empty placeholder; set
  `chains[chain_id] = Vec::new()`. **Slots/ids preserved** → no remap,
  entry-id alignment with the resident intact. Safe because the
  resident `compact`/`repack_colors_dirs` only touch live entries (skip
  via `dead[e]`), and `remove_model` runs first to set those
  tombstones. Add `is_live(chain_id) -> bool`. Pure-CPU edit → CI-testable.

## Tests
- **roxlap-gpu (pure CPU):** `registry.remove` frees data, chain
  empty, `len()` unchanged, ids stay valid; `add_lod → remove →
  add_lod` keeps later ids stable.
- **roxlap-render (CPU backend, CI-safe):** lifecycle on `SceneRenderer`
  forced to CPU — `add_sprite_model → add_sprite_instance_posed →
  set_sprite_instance_transform → remove_sprite_instance →
  remove_sprite_model → compact_sprite_models`; assert
  `dynamic_sprite_count` and that stale handles no-op (return
  false / no panic).
- **Render correctness:** small deterministic CPU render of one posed
  sprite vs axis-aligned; assert the silhouette/center pixel shifts.
  dda_sprite is clean-room — pin a tiny pixel assertion, not a hash.
- GPU paths: covered by demo dogfood (no headless wgpu fixture in CI).

## Docs / dogfood
- Doc-comment all new items; CHANGELOG `[Unreleased] Added`.
- Dogfood in `roxlap-scene-demo`: a cluster of `add_sprite_instance_posed`
  sprites spun per-frame via `set_sprite_instance_transforms`
  (exercises both backends + dirty-flush; manual GPU verification).

## Commit sequencing (one PR)
1. `roxlap-gpu`: `SpriteModel::empty` + `SpriteModelRegistry::{remove,is_live}` + tests.
2. `roxlap-render`: `DynSpriteTransform` + `SpriteModelId` slotmap (`DynModelMap`) + re-point `set_sprites`/`refresh_sprite_model`.
3. `roxlap-render`: CPU backend model add/remove/compact + posed/transform setters.
4. `roxlap-render`: GPU backend model add/remove/compact + posed add + dirty-flush transforms.
5. Facade methods wiring all four + batch variant; render-correctness + lifecycle tests.
6. scene-demo dogfood + CHANGELOG + doc comments.

## Risks / watch-items
- **O(n) transform flush** — mitigated by once-per-frame dirty-flush;
  targeted-update is a noted follow-up.
- **Lazy registry creation** — `add_sprite_model` before `set_sprites`
  must establish residency cleanly (`gpu.add_sprite_model`'s
  zero-instance upload path handles "no registry yet", `lib.rs:3342`).
- **remove ordering** — `gpu.remove_sprite_model(chain)` must precede
  `registry.remove(chain)`.
- **CPU memory parity** — CPU keeps empty model placeholders (not
  reclaimed); acceptable + documented (heavy per-instance kv6 clones
  free at instance-remove).
- Fully additive — existing `set_sprites`/`add_sprite_instance`/
  `remove_sprite_instance` callers compile unchanged.
