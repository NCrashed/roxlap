# roxlap — particle system over sprite instancing (Stage PS)

Entry doc written 2026-07-04 at workspace 0.21.0, right after the QE
quality series closed. This is the **entry doc** for the particle-system
stage — tag **PS**. A fresh-context session should read it top to bottom
before touching code.

## Status

- PS.0 — LANDED 2026-07-04 (this doc + pure-simulation core,
  `roxlap-render/src/particles.rs`, 9 unit tests).
- PS.1 — LANDED 2026-07-04: `sync`/`tick` facade binding through a
  crate-internal `ParticleFacade` seam (mock-tested — a real backend
  needs a window); GPU cull now scales the bounding sphere + LOD pick
  by `SpriteInstanceTransform::max_scale` (hazard 1 fixed at all three
  radius sites); CPU scaled-basis parity **verified** (hazard 2 —
  `dda_sprite` smoke test: 2×/0.5× basis ⇒ ~4×/0.25× pixel coverage).
- PS.2 — LANDED 2026-07-05: full emitter palette — `EmitterShape`
  (Point/Sphere/Box), `ConeDef` cone emission (degrees, SpotLight
  convention), `spin` (yaw about world vertical), `scale_end` /
  `tint_end` over-life lerps, `fade_in_frac`. Renames while
  unreleased: `fade_frac` → `fade_out_frac`; `VelocityDef` grew
  `cone` (lost `Copy`).
- PS.3..5 — not started.

## Goal

A small, engine-owned particle layer — emitters, lifetimes, velocity,
fade — built **on top of the existing dynamic-sprite instancing**, so
every particle is a posed kv6 sprite instance and inherits everything
sprites already have: both backends, per-instance tint/alpha/material,
volumetric + additive TV materials for smoke/fire, shadow flags,
`BillboardLighting::FullBright` for sparks, GPU incremental instances
(streaming-spawn safe since [[project_gpu_incremental_sprite_instances]]).

Explicitly **not** in scope: GPU-simulated particles (compute-updated),
soft particles, screen-space effects. The engine's look is crisp voxel
sprites; thousands, not millions.

## Why this is cheap

The facade already provides every primitive the runtime loop needs:

| Need | Existing API (roxlap-render/src/lib.rs) |
|---|---|
| spawn pre-posed (no axis-flash frame) | `add_sprite_instance_posed` :2363 → `Option<SpriteInstanceId>` |
| despawn O(1) | `remove_sprite_instance` :2381 (swap-remove, other handles stay valid) |
| move N per frame | `set_sprite_instance_transforms(&[(id, xf)])` :2552 — the batch path |
| fade out | `set_sprite_instance_alpha(id, 0..=255)` :2589 |
| recolour | `set_sprite_instance_tint(id, 0x00RRGGBB)` :2606 |
| smoke/fire look | `set_sprite_instance_material` + TV `Material` palette (alpha/additive/volumetric) |
| unlit sparks | `set_sprite_instance_lighting(id, BillboardLighting::FullBright)` :2654 |
| perf hygiene | `set_sprite_instance_shadow_flags` :2627 — particles default to no-cast/no-receive |
| scale | `DynSpriteTransform` basis columns are free-form; GPU inverts a general matrix (`mat3_inverse`, sprite_model.rs:906 — "rotation + non-unit scale") |

What's missing is only the middle layer: emitter definitions, the
per-frame age/velocity/fade integration, id pooling, and a demo.

## Design

### Ownership: a host-owned system that *drives* the facade

`ParticleSystem` is a standalone type in a new `roxlap-render`
module (`src/particles.rs`), **not** new methods on `SceneRenderer` —
the facade is already a ~93-method god object (QE-B1) and particles
don't need private facade state. The host owns the system and calls:

```rust
let mut particles = ParticleSystem::new(seed);
let em = particles.add_emitter(ParticleEmitterDef {
    spawn: SpawnMode::Rate(120.0),
    lifetime: 0.8..1.6,
    velocity: VelocityDef { base: [0.0, 0.0, -14.0], spread: 3.0 },
    gravity: [0.0, 0.0, 22.0],   // +z is DOWN (voxlap convention)
    ..ParticleEmitterDef::new(model) // model = the puff/spark kv6
});                                  // (no Default — a def needs a live model)
// per frame:
particles.update(dt);            // pure simulation, no renderer
particles.sync(&mut renderer);   // spawn/despawn/batch-move/alpha/tint
```

`update` (pure sim) and `sync` (facade writes) are split so the core
integrates + budgets under plain unit tests with **no renderer, no
window, no GPU**. A convenience `tick(&mut renderer, dt)` does both —
mirroring the facade's own QE.1b `tick` naming.

### Simulation core (PS.0)

- `Particle`: pos/vel `[f32; 3]`, age, lifetime, base scale, spin
  (yaw-rate about world z for PS.0), tint/alpha state, instance id
  (`Option<SpriteInstanceId>` — `None` until first sync).
- Integration: semi-implicit Euler (`vel += (gravity - drag·vel)·dt;
  pos += vel·dt`). Enough at particle scale; not a physics engine.
- **Deterministic RNG**: a tiny in-module PCG32 seeded at
  `ParticleSystem::new(seed)`. No new dependency; fixed iteration order
  ⇒ same seed + same dt sequence = identical sim. Golden-testable.
- **Budget**: `max_particles` per system (default 4096). When full,
  spawn requests are dropped (never evict live particles — visible
  pop); a `dropped_spawns()` counter makes the cap observable instead
  of silent (QE "no silent caps" lesson).
- Emitter handles: reuse the crate-internal `EpochSlotMap` (QE.1a) with
  a new `EmitterId` — same stale-handle semantics as every other family.
- Removal inside the pool: `swap_remove`, mirroring the facade's own
  dyn-instance bookkeeping.

### Facade binding (PS.1)

Per `sync`:

1. dead particles → `remove_sprite_instance` (O(1) each);
2. newborn → `add_sprite_instance_posed` (pre-posed — documented
   streaming-spawn path, no one-frame axis-aligned flash) + one-time
   material/lighting/shadow-flag setup; a `None` return (stale model)
   kills the particle and counts it;
3. live → one `set_sprite_instance_transforms` batch;
4. alpha/tint → per-instance calls **only when the u8/u32 actually
   changed** this frame (they're not batched in the facade; see
   Hazards).

Scale-over-life = basis columns × s(t). Clamp s ≥ 0.05: a degenerate
basis makes the instance silently skip drawing (documented on
`add_sprite_instance_posed` :2360) — fine as a kill, wrong as a fade.

### Emitter palette (PS.0 minimal → PS.2 full)

PS.0 ships: point emitter, `SpawnMode::{Rate(f32), Burst(u32)}`,
lifetime range, base velocity + isotropic spread, gravity, drag,
alpha fade-out over the last fraction of life, constant tint.

PS.2 extends: shapes `Point | Sphere{r} | Box{half} | Cone{dir,
half_angle}` (cone = directional spread — fountains, muzzle flashes),
spin, scale-over-life, tint lerp (start→end), fade-in, per-emitter
`BillboardLighting` + material.

### Voxel collision (PS.3)

Optional per-emitter `CollisionMode { None, Kill, Bounce { restitution:
f32 } }` via `Scene::resolve_voxel` point-sampling at the post-step
position (needs `update_with_scene(dt, &Scene)` — the plain `update`
stays scene-free). Point-sampling tunnels at high speed; acceptable for
effects, documented. Bounce reflects `vel` about the axis of entry
(cheapest normal estimate: which of the 3 axes crossed a voxel boundary
this step).

### Demo + debris (PS.4/PS.5)

New scene-demo tab "Particles": fountain (cone + gravity + bounce),
smoke column (volumetric material, slow rise — note rise = **-z**),
click-to-explode (picking → burst emitter at hit + `set_sphere(None)`
carve). PS.5 promotes the debris pattern into an engine helper:
sample voxel colours in the carve sphere *before* carving, then burst
one tinted debris particle per sampled voxel (cap + stride for big
carves) — the "shoot the wall, the wall's colours fly off" effect.

## Hazards (read before each stage)

1. **GPU frustum cull assumes unit basis** — `bound_radius`
   (roxlap-gpu/src/sprite_model.rs:559) doesn't scale with basis
   length; instances scaled **up** under-cull and pop at screen edges.
   PS.1 must multiply the cull radius by the instance's max basis
   column length (cheap, at cull site). Scale-down is conservative-safe.
2. **CPU backend scaled-basis parity is unverified.** The voxlap
   heritage says `s/h/f` magnitude = scale, but no test covers it.
   PS.1 adds a CPU render smoke test with a 2× and 0.5× instance.
3. **Alpha/tint have no batch API** — per-particle facade calls. Fine
   at 4k particles; the PS.5 perf pass measures it (esp. GPU staging
   writes) and only then do we consider adding a batch setter to the
   facade.
4. **Axes**: +z is DOWN. Gravity is *positive* z; smoke rises with
   *negative* z velocity. Get this wrong and every effect is upside
   down but "works".
5. **f32 world positions** — sprites (and thus particles) live in f32
   world space; far-from-origin jitter is a known sprite property, not
   a PS regression.
6. **`sync` ordering**: despawn before spawn (frees pool + instance
   slots), spawn before batch-move (newborns get their frame-0 pose
   from the posed add, so they must not be in the move batch twice).
7. Per-frame heap traffic: reuse persistent `Vec` scratch buffers for
   the transform batch (PF-series lesson — no per-frame allocs in the
   hot loop).

## Stage list

| Stage | Contents | Breaking? |
|---|---|---|
| PS.0 | Entry doc + pure-sim core in `roxlap-render/src/particles.rs`: defs, PCG32, `update`, budget, unit tests (determinism, lifetime, budget, fade) | no — additive |
| PS.1 | `sync`/`tick` facade binding + id pooling; GPU cull-radius scale fix; CPU scaled-basis smoke test | no (gpu cull fix is internal) |
| PS.2 | Full emitter palette: shapes, cone spread, spin, scale/tint-over-life, fade-in | no |
| PS.3 | Voxel collision Kill/Bounce via `Scene::resolve_voxel` (`update_with_scene`) | no |
| PS.4 | scene-demo "Particles" tab: fountain, smoke, click-explosion | no |
| PS.5 | Debris-from-carve helper + perf/stress pass (10k particles, alpha/tint call cost, budget behaviour) | no |

Versioning: purely additive ⇒ next minor (0.22.0) when the stage
closes. Each PS.n lands as its own commit with a CHANGELOG entry.
