# Particles

`ParticleSystem` is roxlap's effects engine: fountains, smoke, sparks,
debris. It is deliberately a *host-side* system built on the sprite
API from [chapter 7](sprites.md) — every particle is an ordinary kv6
sprite instance, so particles are lit, shadowed and materialed like
everything else, and they collide with the actual voxel world.

The snippets come from a runnable example — a bouncing water
fountain, a buoyant smoke column, and a scripted explosion every four
seconds that carves real craters:

```sh
cargo run --release -p roxlap-render --example book_particles
```

## The system and its emitters

One `ParticleSystem` per scene, seeded — the same seed replays the
same effects. Effects are **emitters**, described declaratively by a
`ParticleEmitterDef`: construct with `new(model)` and override what
the effect needs. The two ambient effects show most of the vocabulary:

```rust,noplayground
{{#include ../../../crates/roxlap-render/examples/book_particles.rs:system}}
```

The def's fields group into three concerns:

- **Spawning** — `spawn` (`Rate(n)` per second with fractional
  accumulation, `Burst(n)` once on add, `Manual` + explicit
  `burst()` calls), `shape` (point / sphere / box around `pos`),
  `lifetime` (a sampled range).
- **Motion** — `velocity` (a fixed `base` + isotropic `spread` + an
  optional `ConeDef`, composed by addition), `gravity` (**positive z
  is down**: the default `[0, 0, 22]` falls; smoke rises with a
  negative-z term), `drag`, `spin`.
- **Look** — `scale` → `scale_end` (growing smoke, shrinking sparks),
  `fade_in_frac` / `fade_out_frac` (alpha ramps at the ends of life),
  `tint` → `tint_end` (white-hot → ember), `material` (the palette
  from [chapter 6](lighting.md) — smoke wants alpha-blend, sparks
  additive), `lighting`, `shadows`.

Note the model trick: all three effects share plain **white** models —
the per-particle tint does the colouring, so one 2-voxel cube serves
as droplet and spark both.

Shadows default to **off** in both directions for particles —
hundreds of shadow-casting instances is a perf trap; opt back in per
effect.

## The per-frame protocol

```rust,noplayground
{{#include ../../../crates/roxlap-render/examples/book_particles.rs:tick}}
```

One call simulates, collides, and mirrors live particles into sprite
instances (through the same batched-transform path as any bulk sprite
motion). Use `tick(renderer, dt)` when nothing collides —
`tick_with_scene` is what enables `collision`:

- `CollisionMode::None` — pass through (default).
- `Kill` — die on contact: impact sparks, raindrops.
- `Bounce { restitution }` — arcade reflection off voxel faces.

The collision test is a point sample nudged along the velocity —
cheap and honest about what it is: fast particles can tunnel through
one-voxel walls, resting particles stop being tested. It's an effects
system, not a physics engine.

## Explosions that change the world: `carve_debris`

The signature roxlap effect — because particles and world share one
voxel vocabulary, an explosion can *become* its debris:

```rust,noplayground
{{#include ../../../crates/roxlap-render/examples/book_particles.rs:explosion}}
```

`carve_debris` samples the crater's voxel colours, carves the sphere
out of the grid (the same edit path as [chapter 3](scene-graph.md),
lighting re-bake included), and spawns one tinted debris particle per
removed surface voxel with a radial kick. The floor's own colours
tumble away — no artist-authored debris needed.

Budgets keep it bounded: `set_carve_debris_cap` limits debris per
carve (default `CARVE_DEBRIS_CAP`), `set_max_particles` caps the
whole system (default `DEFAULT_MAX_PARTICLES`; overflow spawns are
dropped and counted in `dropped_spawns`, never reallocated). For HUD
counters read `particle_count()` / `particles()`.

Performance envelope: the PS-stage benchmark holds ~10 000 live
particles at ≈225 µs of simulation per frame — effects budget-level,
not gameplay-limiting. The knobs that matter when a scene gets heavy:
crater radius (a radius-4 carve samples ~70 voxels), burst sizes, and
keeping particle shadows off.

## Further reading

- The **Particles** demo scene — this chapter's effects plus
  crosshair-aimed interactive explosions and explosion light flashes
  (`ROXLAP_SCENE=Particles cargo run --release -p roxlap-scene-demo`).
- [`PORTING-PARTICLES.md`](https://github.com/NCrashed/roxlap/blob/master/docs/porting/PORTING-PARTICLES.md)
  — the PS-stage design history and benchmarks.
