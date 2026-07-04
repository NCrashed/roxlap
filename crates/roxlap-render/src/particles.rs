//! PS.0 — particle-system core: emitters + pure simulation.
//!
//! A [`ParticleSystem`] is a **host-owned** layer over the facade's
//! dynamic sprite instances (see `docs/porting/PORTING-PARTICLES.md`):
//! every live particle will become one posed kv6 instance, so particles
//! inherit both backends, per-instance tint/alpha/material, TV
//! volumetric/additive looks and [`BillboardLighting`] for free. This
//! module is deliberately **not** part of [`SceneRenderer`]'s method
//! surface — the system *drives* the facade through its public API.
//!
//! PS.0 ships the renderer-free half: emitter definitions, the
//! deterministic per-frame integration ([`ParticleSystem::update`]),
//! and the particle budget. The facade binding (`sync`, PS.1) consumes
//! the state this half maintains — newborn particles carry
//! `instance == None`, dead ones queue their [`SpriteInstanceId`] for
//! removal in [`ParticleSystem::drain_dead_instances`].
//!
//! Axes reminder: **+z is DOWN** (voxlap convention) — gravity is
//! positive z, smoke rises with negative z velocity.
//!
//! [`SceneRenderer`]: crate::SceneRenderer

use std::ops::Range;

use crate::{
    BillboardLighting, EpochSlotMap, ShadowFlags, SlotHandle, SpriteInstanceId, SpriteModelId,
};

/// Stable handle to an emitter inside one [`ParticleSystem`] — the
/// result of [`add_emitter`](ParticleSystem::add_emitter), passed to
/// the per-emitter setters and [`burst`](ParticleSystem::burst).
/// Epoch-generational like every other facade handle family: a removed
/// emitter's handle resolves to a safe no-op, never to another emitter.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct EmitterId {
    slot: u32,
    gen: u32,
}

// `impl_slot_handle!` is textually scoped to lib.rs; the trait is
// crate-visible, so the two-method impl is written out here.
impl SlotHandle for EmitterId {
    fn mint(slot: u32, gen: u32) -> Self {
        Self { slot, gen }
    }
    fn parts(self) -> (u32, u32) {
        (self.slot, self.gen)
    }
}

/// How an emitter produces particles.
#[derive(Clone, Copy, Debug)]
pub enum SpawnMode {
    /// Continuous emission at `n` particles per second; fractional
    /// particles accumulate across frames, so `Rate(0.5)` spawns one
    /// particle every two seconds regardless of frame rate.
    Rate(f32),
    /// One burst of `n` particles the moment the emitter is added,
    /// then nothing (explosions). Further bursts via
    /// [`burst`](ParticleSystem::burst).
    Burst(u32),
    /// Nothing automatic — the host calls
    /// [`burst`](ParticleSystem::burst) itself.
    Manual,
}

/// Initial particle velocity: a fixed base plus an isotropic spread.
#[derive(Clone, Copy, Debug, Default)]
pub struct VelocityDef {
    /// Velocity every particle starts from, world units/second.
    /// Remember +z is down: a fountain fires with negative z.
    pub base: [f32; 3],
    /// Magnitude of the random isotropic kick added to `base`: each
    /// particle adds a uniformly random direction scaled by a uniform
    /// `0..spread` speed. `0.0` (default) = no randomness.
    pub spread: f32,
}

/// Recipe for [`add_emitter`](ParticleSystem::add_emitter). Construct
/// with [`ParticleEmitterDef::new`] and override what the effect needs;
/// there is no `Default` because a def is meaningless without a live
/// [`SpriteModelId`].
#[derive(Clone, Debug)]
pub struct ParticleEmitterDef {
    /// The kv6 sprite model every particle instantiates (a puff, a
    /// spark, a shard) — from
    /// [`add_sprite_model`](crate::SceneRenderer::add_sprite_model).
    pub model: SpriteModelId,
    /// Emitter position, world space. Movable later via
    /// [`set_emitter_pos`](ParticleSystem::set_emitter_pos).
    pub pos: [f32; 3],
    /// Spawn behaviour (default [`SpawnMode::Manual`]).
    pub spawn: SpawnMode,
    /// Per-particle lifetime, seconds, sampled uniformly. A degenerate
    /// range (`end <= start`) collapses to `start`. Clamped to ≥ 1 ms.
    pub lifetime: Range<f32>,
    /// Initial velocity distribution.
    pub velocity: VelocityDef,
    /// Constant acceleration, world units/s² — gravity is **positive
    /// z** (default `[0, 0, 22]`, a decent arcade fall).
    pub gravity: [f32; 3],
    /// Linear drag coefficient, 1/s: each step removes
    /// `drag · vel · dt`. `0.0` = ballistic; smoke wants ~1-3.
    pub drag: f32,
    /// Uniform base scale applied to the instance basis (`1.0` =
    /// authored model size). PS.1 clamps the rendered scale ≥ 0.05 —
    /// a degenerate basis silently skips drawing.
    pub scale: f32,
    /// Fraction of the lifetime over which alpha fades 255 → 0 at the
    /// end (`0.25` = the last quarter). `0.0` = no fade, particles
    /// vanish at full opacity.
    pub fade_frac: f32,
    /// Per-particle RGB tint, packed `0x00RRGGBB` (white = no-op).
    pub tint: u32,
    /// Voxel-material id for every particle (TV palette; `0` opaque).
    /// Smoke wants an alpha/volumetric material, sparks additive.
    pub material: u8,
    /// Shading-normal mode (default [`BillboardLighting::FaceNormal`];
    /// glowing effects want [`BillboardLighting::FullBright`]).
    pub lighting: BillboardLighting,
    /// Shadow participation. Defaults to **neither cast nor receive** —
    /// hundreds of shadow-casting particles are a perf trap; opt back
    /// in per effect.
    pub shadows: ShadowFlags,
}

impl ParticleEmitterDef {
    /// A def with every field at its documented default: manual spawn
    /// at the origin, 1-2 s lifetime, no initial velocity, arcade
    /// gravity, no drag, unit scale, fade over the last quarter, no
    /// tint, opaque material, face-normal lighting, shadows off.
    #[must_use]
    pub fn new(model: SpriteModelId) -> Self {
        Self {
            model,
            pos: [0.0, 0.0, 0.0],
            spawn: SpawnMode::Manual,
            lifetime: 1.0..2.0,
            velocity: VelocityDef::default(),
            gravity: [0.0, 0.0, 22.0],
            drag: 0.0,
            scale: 1.0,
            fade_frac: 0.25,
            tint: 0x00FF_FFFF,
            material: 0,
            lighting: BillboardLighting::FaceNormal,
            shadows: ShadowFlags {
                casts: false,
                receives: false,
            },
        }
    }
}

/// One live particle — a read-only view for hosts (HUD counters,
/// custom overlays); the system owns the mutation.
#[derive(Clone, Copy, Debug)]
pub struct Particle {
    /// World position.
    pub pos: [f32; 3],
    /// World velocity, units/second.
    pub vel: [f32; 3],
    /// Seconds lived so far.
    pub age: f32,
    /// Seconds this particle lives in total.
    pub lifetime: f32,
    /// Uniform render scale (the emitter's [`ParticleEmitterDef::scale`]).
    pub scale: f32,
    /// Current alpha 0..=255, driven by the emitter's fade curve.
    pub alpha: u8,
    /// Packed `0x00RRGGBB` tint.
    pub tint: u32,
    /// Owning emitter slot — resolves render params (model, material,
    /// lighting) at sync time.
    pub(crate) emitter_slot: u32,
    /// The facade instance backing this particle: `None` until the
    /// first `sync` (PS.1) spawns it.
    pub(crate) instance: Option<SpriteInstanceId>,
}

/// Per-emitter live state.
struct EmitterState {
    def: ParticleEmitterDef,
    /// Fractional particles owed by [`SpawnMode::Rate`].
    spawn_acc: f64,
    /// Live particles owned by this emitter — a retired emitter's
    /// state is kept until this drains to 0 (particles read their
    /// def's gravity/drag every step).
    live: u32,
    /// [`ParticleSystem::remove_emitter`] called: stop spawning, free
    /// the state once `live == 0`.
    retired: bool,
}

/// PCG32 (Melissa O'Neill's `pcg32_oneseq`): 64-bit state, 32-bit
/// output. Deterministic and tiny — same seed + same `dt` sequence ⇒
/// bit-identical simulation, so effects are golden-testable. Not
/// cryptographic, deliberately.
struct Pcg32 {
    state: u64,
}

impl Pcg32 {
    const MULT: u64 = 6_364_136_223_846_793_005;
    const INC: u64 = 1_442_695_040_888_963_407;

    fn new(seed: u64) -> Self {
        let mut rng = Self {
            state: seed.wrapping_add(Self::INC),
        };
        rng.next_u32();
        rng
    }

    fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(Self::MULT).wrapping_add(Self::INC);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        xorshifted.rotate_right((old >> 59) as u32)
    }

    /// Uniform in `[0, 1)` with 24 bits of mantissa.
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * (1.0 / (1u32 << 24) as f32)
    }

    /// Uniform in `[start, end)`; a degenerate range yields `start`.
    fn range_f32(&mut self, r: &Range<f32>) -> f32 {
        if r.end <= r.start {
            return r.start;
        }
        r.start + (r.end - r.start) * self.next_f32()
    }

    /// Uniform direction on the unit sphere (cube-rejection — no
    /// trig, deterministic step count per accepted sample stream).
    fn unit_vec(&mut self) -> [f32; 3] {
        loop {
            let v = [
                self.next_f32() * 2.0 - 1.0,
                self.next_f32() * 2.0 - 1.0,
                self.next_f32() * 2.0 - 1.0,
            ];
            let len2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
            if len2 > 1e-4 && len2 <= 1.0 {
                let inv = 1.0 / len2.sqrt();
                return [v[0] * inv, v[1] * inv, v[2] * inv];
            }
        }
    }
}

/// Default [`ParticleSystem`] particle budget.
pub const DEFAULT_MAX_PARTICLES: usize = 4096;

/// A self-contained particle simulation: emitters, a shared particle
/// pool with a budget, and a deterministic seeded RNG. Host-owned;
/// per frame call [`update`](Self::update) (pure simulation, no
/// renderer) then — from PS.1 — `sync(&mut SceneRenderer)` to mirror
/// the pool into dynamic sprite instances.
///
/// Budget semantics: when the pool is full, **spawns are dropped**
/// (never evict a live particle — a visible pop);
/// [`dropped_spawns`](Self::dropped_spawns) makes the cap observable
/// instead of silent.
pub struct ParticleSystem {
    rng: Pcg32,
    map: EpochSlotMap<EmitterId>,
    /// Parallel to `map`'s slots; `None` once a retired emitter drains.
    emitters: Vec<Option<EmitterState>>,
    particles: Vec<Particle>,
    /// Instances whose particles died since the last drain — PS.1's
    /// `sync` removes these from the facade.
    dead_instances: Vec<SpriteInstanceId>,
    max_particles: usize,
    dropped_spawns: u64,
}

impl ParticleSystem {
    /// A system with the given RNG seed and the default budget
    /// ([`DEFAULT_MAX_PARTICLES`]). Same seed + same call sequence ⇒
    /// bit-identical simulation.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            rng: Pcg32::new(seed),
            map: EpochSlotMap::default(),
            emitters: Vec::new(),
            particles: Vec::new(),
            dead_instances: Vec::new(),
            max_particles: DEFAULT_MAX_PARTICLES,
            dropped_spawns: 0,
        }
    }

    /// Set the particle budget. Lowering it below the current live
    /// count kills nothing — it only gates future spawns.
    pub fn set_max_particles(&mut self, max: usize) {
        self.max_particles = max;
    }

    /// Register an emitter. [`SpawnMode::Burst`] fires immediately.
    pub fn add_emitter(&mut self, def: ParticleEmitterDef) -> EmitterId {
        let slot = self.emitters.len() as u32;
        let id = self.map.alloc(slot);
        let burst = match def.spawn {
            SpawnMode::Burst(n) => n,
            _ => 0,
        };
        self.emitters.push(Some(EmitterState {
            def,
            spawn_acc: 0.0,
            live: 0,
            retired: false,
        }));
        if burst > 0 {
            self.spawn_from(slot as usize, burst);
        }
        id
    }

    /// Retire an emitter: it stops spawning immediately and its handle
    /// goes stale, but particles already in flight live out their
    /// lifetimes (the state drains away with the last one). Returns
    /// `false` on a stale handle.
    pub fn remove_emitter(&mut self, id: EmitterId) -> bool {
        let Some(slot) = self.map.index(id) else {
            return false;
        };
        if !self.map.remove(id) {
            return false;
        }
        let state = self.emitters[slot].as_mut().expect("live map ⇒ state");
        if state.live == 0 {
            self.emitters[slot] = None;
        } else {
            state.retired = true;
        }
        true
    }

    /// Move an emitter (attach effects to moving things). Returns
    /// `false` on a stale handle.
    pub fn set_emitter_pos(&mut self, id: EmitterId, pos: [f32; 3]) -> bool {
        let Some(slot) = self.map.index(id) else {
            return false;
        };
        let state = self.emitters[slot].as_mut().expect("live map ⇒ state");
        state.def.pos = pos;
        true
    }

    /// Spawn `n` particles from `id` right now (any [`SpawnMode`]).
    /// Returns how many actually spawned (budget may drop the rest);
    /// `0` on a stale handle.
    pub fn burst(&mut self, id: EmitterId, n: u32) -> u32 {
        let Some(slot) = self.map.index(id) else {
            return 0;
        };
        self.spawn_from(slot, n)
    }

    /// Advance the simulation by `dt` seconds: integrate + age + fade
    /// every particle, retire the dead (their facade instances queue
    /// in [`drain_dead_instances`](Self::drain_dead_instances)), then
    /// run [`SpawnMode::Rate`] emitters. Pure simulation — no facade
    /// calls, unit-testable without a window or GPU.
    pub fn update(&mut self, dt: f64) {
        let dtf = dt.max(0.0) as f32;

        // 1. Age + semi-implicit Euler + fade. Split field borrows:
        //    defs are read-only here.
        let emitters = &self.emitters;
        for p in &mut self.particles {
            p.age += dtf;
            if p.age >= p.lifetime {
                continue; // swept below
            }
            let def = &emitters[p.emitter_slot as usize]
                .as_ref()
                .expect("live particle ⇒ emitter state retained")
                .def;
            for a in 0..3 {
                p.vel[a] += (def.gravity[a] - def.drag * p.vel[a]) * dtf;
                p.pos[a] += p.vel[a] * dtf;
            }
            p.alpha = fade_alpha(p.age, p.lifetime, def.fade_frac);
        }

        // 2. Kill sweep (swap-remove keeps the pool dense).
        let mut i = 0;
        while i < self.particles.len() {
            if self.particles[i].age >= self.particles[i].lifetime {
                let p = self.particles.swap_remove(i);
                if let Some(inst) = p.instance {
                    self.dead_instances.push(inst);
                }
                self.on_particle_died(p.emitter_slot as usize);
            } else {
                i += 1;
            }
        }

        // 3. Rate spawning (after the sweep, so freed budget is
        //    available the same frame; newborns keep age 0 and the
        //    emitter position — their first integration is next frame,
        //    matching the pre-posed facade spawn).
        for slot in 0..self.emitters.len() {
            let n = {
                let Some(state) = self.emitters[slot].as_mut() else {
                    continue;
                };
                if state.retired {
                    continue;
                }
                let SpawnMode::Rate(rate) = state.def.spawn else {
                    continue;
                };
                state.spawn_acc += f64::from(rate) * dt.max(0.0);
                let n = state.spawn_acc.floor();
                state.spawn_acc -= n;
                n as u32
            };
            if n > 0 {
                self.spawn_from(slot, n);
            }
        }
    }

    /// The live particles, unordered (the pool swap-removes).
    #[must_use]
    pub fn particles(&self) -> &[Particle] {
        &self.particles
    }

    /// Number of live particles.
    #[must_use]
    pub fn particle_count(&self) -> usize {
        self.particles.len()
    }

    /// Number of active (non-retired) emitters.
    #[must_use]
    pub fn emitter_count(&self) -> usize {
        self.emitters
            .iter()
            .filter(|e| e.as_ref().is_some_and(|s| !s.retired))
            .count()
    }

    /// Spawns dropped by the budget since construction. A steadily
    /// climbing value means the effect design outruns
    /// [`set_max_particles`](Self::set_max_particles).
    #[must_use]
    pub fn dropped_spawns(&self) -> u64 {
        self.dropped_spawns
    }

    /// Drain the facade instances of particles that died since the
    /// last drain. `sync` (PS.1) removes each via
    /// [`remove_sprite_instance`](crate::SceneRenderer::remove_sprite_instance);
    /// hosts doing their own rendering can consume it directly.
    pub fn drain_dead_instances(&mut self) -> impl Iterator<Item = SpriteInstanceId> + '_ {
        self.dead_instances.drain(..)
    }

    /// Spawn up to `n` particles from emitter `slot`; returns how many
    /// fit the budget.
    fn spawn_from(&mut self, slot: usize, n: u32) -> u32 {
        let state = self.emitters[slot]
            .as_mut()
            .expect("spawn_from callers hold a live slot");
        let def = state.def.clone(); // stack-only, cheap
        let mut spawned = 0;
        for _ in 0..n {
            if self.particles.len() >= self.max_particles {
                self.dropped_spawns += u64::from(n - spawned);
                break;
            }
            let mut vel = def.velocity.base;
            if def.velocity.spread > 0.0 {
                let dir = self.rng.unit_vec();
                let speed = self.rng.next_f32() * def.velocity.spread;
                for a in 0..3 {
                    vel[a] += dir[a] * speed;
                }
            }
            let lifetime = self.rng.range_f32(&def.lifetime).max(1e-3);
            self.particles.push(Particle {
                pos: def.pos,
                vel,
                age: 0.0,
                lifetime,
                scale: def.scale,
                alpha: fade_alpha(0.0, lifetime, def.fade_frac),
                tint: def.tint,
                emitter_slot: slot as u32,
                instance: None,
            });
            spawned += 1;
        }
        // Re-borrow: the RNG borrow above forced dropping `state`.
        self.emitters[slot]
            .as_mut()
            .expect("slot unchanged during spawn")
            .live += spawned;
        spawned
    }

    /// Bookkeeping for one particle death: decrement the emitter's
    /// live count and free a drained retired emitter.
    fn on_particle_died(&mut self, slot: usize) {
        let state = self.emitters[slot]
            .as_mut()
            .expect("live particle ⇒ emitter state retained");
        state.live -= 1;
        if state.retired && state.live == 0 {
            self.emitters[slot] = None;
        }
    }
}

/// Alpha for `age` of `lifetime` with a fade over the trailing
/// `fade_frac`: 255 until the fade window, then linear to 0 at death.
fn fade_alpha(age: f32, lifetime: f32, fade_frac: f32) -> u8 {
    if fade_frac <= 0.0 {
        return 255;
    }
    let frac = age / lifetime;
    let fade_start = 1.0 - fade_frac.min(1.0);
    if frac <= fade_start {
        return 255;
    }
    let k = (1.0 - frac) / fade_frac.min(1.0);
    (k.clamp(0.0, 1.0) * 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A model handle for tests. The sim core never dereferences it —
    /// only `sync` (PS.1) resolves models — so a minted dummy is fine.
    fn dummy_model() -> SpriteModelId {
        SpriteModelId::mint(0, 0)
    }

    fn base_def() -> ParticleEmitterDef {
        ParticleEmitterDef {
            spawn: SpawnMode::Manual,
            lifetime: 1.0..1.0,
            gravity: [0.0, 0.0, 0.0],
            fade_frac: 0.0,
            ..ParticleEmitterDef::new(dummy_model())
        }
    }

    #[test]
    fn same_seed_is_bit_identical() {
        let run = || {
            let mut sys = ParticleSystem::new(0x00C0_FFEE);
            let em = sys.add_emitter(ParticleEmitterDef {
                spawn: SpawnMode::Rate(120.0),
                lifetime: 0.3..0.9,
                velocity: VelocityDef {
                    base: [0.0, 0.0, -10.0],
                    spread: 4.0,
                },
                gravity: [0.0, 0.0, 22.0],
                ..ParticleEmitterDef::new(dummy_model())
            });
            sys.burst(em, 7);
            for _ in 0..60 {
                sys.update(1.0 / 60.0);
            }
            sys.particles()
                .iter()
                .map(|p| (p.pos, p.vel, p.age, p.lifetime))
                .collect::<Vec<_>>()
        };
        let (a, b) = (run(), run());
        assert_eq!(a.len(), b.len());
        for (pa, pb) in a.iter().zip(&b) {
            assert_eq!(pa, pb, "same seed must be bit-identical");
        }
        assert!(!a.is_empty());
    }

    #[test]
    fn rate_accumulates_across_frames() {
        let mut sys = ParticleSystem::new(1);
        sys.add_emitter(ParticleEmitterDef {
            spawn: SpawnMode::Rate(10.0),
            lifetime: 100.0..100.0,
            ..base_def()
        });
        for _ in 0..10 {
            sys.update(0.1);
        }
        assert_eq!(sys.particle_count(), 10);

        // Sub-particle rates accumulate: 0.5/s over 2 s = 1 particle.
        let mut slow = ParticleSystem::new(2);
        slow.add_emitter(ParticleEmitterDef {
            spawn: SpawnMode::Rate(0.5),
            lifetime: 100.0..100.0,
            ..base_def()
        });
        for _ in 0..20 {
            slow.update(0.1);
        }
        assert_eq!(slow.particle_count(), 1);
    }

    #[test]
    fn burst_mode_fires_on_add() {
        let mut sys = ParticleSystem::new(3);
        sys.add_emitter(ParticleEmitterDef {
            spawn: SpawnMode::Burst(5),
            ..base_def()
        });
        assert_eq!(sys.particle_count(), 5);
    }

    #[test]
    fn budget_drops_spawns_and_counts_them() {
        let mut sys = ParticleSystem::new(4);
        sys.set_max_particles(5);
        let em = sys.add_emitter(base_def());
        assert_eq!(sys.burst(em, 10), 5);
        assert_eq!(sys.particle_count(), 5);
        assert_eq!(sys.dropped_spawns(), 5);
    }

    #[test]
    fn particles_die_at_lifetime() {
        let mut sys = ParticleSystem::new(5);
        let em = sys.add_emitter(ParticleEmitterDef {
            lifetime: 0.5..0.5,
            ..base_def()
        });
        sys.burst(em, 3);
        sys.update(0.3);
        assert_eq!(sys.particle_count(), 3);
        sys.update(0.3); // age 0.6 ≥ 0.5
        assert_eq!(sys.particle_count(), 0);
        // Never synced ⇒ no facade instances owed.
        assert_eq!(sys.drain_dead_instances().count(), 0);
    }

    #[test]
    fn semi_implicit_euler_gravity() {
        let mut sys = ParticleSystem::new(6);
        let em = sys.add_emitter(ParticleEmitterDef {
            gravity: [0.0, 0.0, 10.0],
            lifetime: 100.0..100.0,
            ..base_def()
        });
        sys.burst(em, 1);
        sys.update(0.1);
        let p = sys.particles()[0];
        // vel += g·dt first, then pos += vel·dt.
        assert!((p.vel[2] - 1.0).abs() < 1e-6);
        assert!((p.pos[2] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn fade_curve_hits_endpoints() {
        assert_eq!(fade_alpha(0.0, 1.0, 0.5), 255);
        assert_eq!(fade_alpha(0.5, 1.0, 0.5), 255); // window edge
        assert_eq!(fade_alpha(0.75, 1.0, 0.5), 127); // mid-fade
        assert_eq!(fade_alpha(1.0, 1.0, 0.5), 0);
        assert_eq!(fade_alpha(0.99, 1.0, 0.0), 255); // no fade
    }

    #[test]
    fn retired_emitter_drains_then_frees() {
        let mut sys = ParticleSystem::new(7);
        let em = sys.add_emitter(ParticleEmitterDef {
            spawn: SpawnMode::Rate(1000.0),
            lifetime: 0.2..0.2,
            ..base_def()
        });
        sys.update(0.01);
        assert!(sys.particle_count() > 0);
        assert!(sys.remove_emitter(em));
        assert_eq!(sys.emitter_count(), 0);
        // Stale-handle ops are safe no-ops.
        assert!(!sys.remove_emitter(em));
        assert!(!sys.set_emitter_pos(em, [1.0, 2.0, 3.0]));
        assert_eq!(sys.burst(em, 10), 0);
        // In-flight particles live out their lifetime, then the slot
        // frees.
        let live = sys.particle_count();
        sys.update(0.3);
        assert_eq!(sys.particle_count(), 0);
        assert!(live > 0);
        assert!(sys.emitters.iter().all(Option::is_none));
    }

    #[test]
    fn moved_emitter_spawns_at_new_pos() {
        let mut sys = ParticleSystem::new(8);
        let em = sys.add_emitter(base_def());
        assert!(sys.set_emitter_pos(em, [5.0, 6.0, 7.0]));
        sys.burst(em, 1);
        assert_eq!(sys.particles()[0].pos, [5.0, 6.0, 7.0]);
    }
}
