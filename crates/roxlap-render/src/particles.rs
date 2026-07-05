//! Stage PS — particle system over sprite instancing.
//!
//! A [`ParticleSystem`] is a **host-owned** layer over the facade's
//! dynamic sprite instances (see `docs/porting/PORTING-PARTICLES.md`):
//! every live particle is one posed kv6 instance, so particles inherit
//! both backends, per-instance tint/alpha/material, TV
//! volumetric/additive looks and [`BillboardLighting`] for free. This
//! module is deliberately **not** part of [`SceneRenderer`]'s method
//! surface — the system *drives* the facade through its public API.
//!
//! Two halves: the renderer-free simulation
//! ([`ParticleSystem::update`] — emitters, deterministic integration,
//! over-life curves, budget; PS.0/PS.2) and the facade binding
//! ([`ParticleSystem::sync`] / [`tick`](ParticleSystem::tick) —
//! spawn/despawn/batch-move; PS.1). Hosts doing their own rendering
//! can use `update` alone and consume
//! [`ParticleSystem::drain_dead_instances`] directly.
//!
//! Axes reminder: **+z is DOWN** (voxlap convention) — gravity is
//! positive z, smoke rises with negative z velocity.
//!
//! [`SceneRenderer`]: crate::SceneRenderer

use std::ops::Range;

use crate::{
    BillboardLighting, DynSpriteTransform, EpochSlotMap, SceneRenderer, ShadowFlags, SlotHandle,
    SpriteInstanceId, SpriteModelId,
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

/// Where new particles appear, relative to the emitter position (PS.2).
#[derive(Clone, Copy, Debug, Default)]
pub enum EmitterShape {
    /// Every particle starts exactly at the emitter position (default).
    #[default]
    Point,
    /// Uniform inside a ball of this radius.
    Sphere {
        /// Ball radius, world units.
        radius: f32,
    },
    /// Uniform inside an axis-aligned box of these half-extents.
    Box {
        /// Half-extents per axis, world units.
        half: [f32; 3],
    },
}

/// Directional cone emission (PS.2) — fountains, muzzle flashes,
/// impact sprays: a random direction within `half_angle_deg` of
/// `axis`, at a speed sampled from `speed`.
#[derive(Clone, Debug)]
pub struct ConeDef {
    /// Cone axis (need not be unit length; a zero vector falls back to
    /// straight **up**, `[0, 0, -1]` — +z is down).
    pub axis: [f32; 3],
    /// Cone half-angle in **degrees** (the [`SpotLight`] convention):
    /// `0` = a beam along the axis, `180` = fully isotropic. Clamped
    /// to `0..=180`.
    ///
    /// [`SpotLight`]: crate::SpotLight
    pub half_angle_deg: f32,
    /// Speed along the sampled direction, world units/second, uniform.
    pub speed: Range<f32>,
}

/// Initial particle velocity: a fixed base, an isotropic spread, and
/// an optional directional cone — the three compose by addition.
#[derive(Clone, Debug, Default)]
pub struct VelocityDef {
    /// Velocity every particle starts from, world units/second.
    /// Remember +z is down: a fountain fires with negative z.
    pub base: [f32; 3],
    /// Magnitude of the random isotropic kick added to `base`: each
    /// particle adds a uniformly random direction scaled by a uniform
    /// `0..spread` speed. `0.0` (default) = no randomness.
    pub spread: f32,
    /// Optional cone kick added on top (PS.2). `None` (default) = no
    /// directional term.
    pub cone: Option<ConeDef>,
}

/// How particles react to solid voxels (PS.3) — checked only by the
/// `_with_scene` update/tick variants; the scene-free ones never
/// collide. The test is a point sample of the post-step position,
/// nudged half a voxel along the velocity ([`Scene::resolve_voxel`]'s
/// picking nudge, reused): contact registers slightly before visual
/// interpenetration, fast particles can tunnel through walls thinner
/// than one step's travel, and a **resting** particle (zero velocity)
/// is never re-tested — all acceptable for effects, none of this is a
/// physics engine.
///
/// [`Scene::resolve_voxel`]: roxlap_scene::Scene::resolve_voxel
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum CollisionMode {
    /// No voxel interaction (default) — particles pass through.
    #[default]
    None,
    /// Die on contact (impact sparks, raindrops).
    Kill,
    /// Reflect off the surface: velocity flips on the axes whose voxel
    /// boundary was crossed this step (the cheapest normal estimate —
    /// exact for axis-aligned grids, approximate for rotated ones),
    /// then the whole velocity scales by `restitution` — tangential
    /// damping included, deliberately arcade.
    Bounce {
        /// Energy kept per bounce, `0..=1` (`0.5` = half).
        restitution: f32,
    },
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
    /// Spawn-position distribution around `pos` (PS.2; default
    /// [`EmitterShape::Point`]).
    pub shape: EmitterShape,
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
    /// Voxel-collision reaction (PS.3; default [`CollisionMode::None`]).
    /// Only the `_with_scene` update/tick variants test it.
    pub collision: CollisionMode,
    /// Uniform base scale applied to the instance basis (`1.0` =
    /// authored model size). The rendered scale clamps to ≥ 0.05 —
    /// a degenerate basis silently skips drawing.
    pub scale: f32,
    /// End-of-life scale (PS.2): the particle lerps `scale →
    /// scale_end` over its lifetime — growing smoke, shrinking sparks.
    /// `None` (default) = constant.
    pub scale_end: Option<f32>,
    /// Spin rate about the world vertical, **radians/second**, sampled
    /// uniformly per particle (PS.2) — a symmetric range like
    /// `-3.0..3.0` spins debris both ways. Default `0.0..0.0` = no
    /// spin.
    pub spin: Range<f32>,
    /// Fraction of the lifetime over which alpha ramps 0 → 255 at the
    /// **start** (PS.2) — smoke that condenses in rather than pops.
    /// `0.0` (default) = born fully opaque.
    pub fade_in_frac: f32,
    /// Fraction of the lifetime over which alpha fades 255 → 0 at the
    /// end (`0.25` = the last quarter). `0.0` = no fade, particles
    /// vanish at full opacity.
    pub fade_out_frac: f32,
    /// Per-particle RGB tint, packed `0x00RRGGBB` (white = no-op).
    pub tint: u32,
    /// End-of-life tint (PS.2): lerped per channel from `tint` over
    /// the lifetime — white-hot → red → dark ember. `None` (default) =
    /// constant.
    pub tint_end: Option<u32>,
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
    /// at a point at the origin, 1-2 s lifetime, no initial velocity,
    /// arcade gravity, no drag, constant unit scale, no spin, fade out
    /// over the last quarter, constant no-op tint, opaque material,
    /// face-normal lighting, shadows off.
    #[must_use]
    pub fn new(model: SpriteModelId) -> Self {
        Self {
            model,
            pos: [0.0, 0.0, 0.0],
            shape: EmitterShape::Point,
            spawn: SpawnMode::Manual,
            lifetime: 1.0..2.0,
            velocity: VelocityDef::default(),
            gravity: [0.0, 0.0, 22.0],
            drag: 0.0,
            collision: CollisionMode::None,
            scale: 1.0,
            scale_end: None,
            spin: 0.0..0.0,
            fade_in_frac: 0.0,
            fade_out_frac: 0.25,
            tint: 0x00FF_FFFF,
            tint_end: None,
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
    /// Current uniform render scale (lerps `scale → scale_end` when
    /// the emitter sets an end scale).
    pub scale: f32,
    /// Current rotation about the world vertical, radians (PS.2).
    pub yaw: f32,
    /// Sampled spin rate, radians/second (PS.2).
    pub spin_rate: f32,
    /// Current alpha 0..=255, driven by the emitter's fade curves.
    pub alpha: u8,
    /// Current packed `0x00RRGGBB` tint (lerps `tint → tint_end` when
    /// the emitter sets an end tint).
    pub tint: u32,
    /// Owning emitter slot — resolves render params (model, material,
    /// lighting) at sync time.
    pub(crate) emitter_slot: u32,
    /// The facade instance backing this particle: `None` until the
    /// first [`sync`](ParticleSystem::sync) spawns it.
    pub(crate) instance: Option<SpriteInstanceId>,
    /// The alpha last written to the facade (which defaults a fresh
    /// instance to 255) — `sync` calls the per-instance alpha setter
    /// only when this differs, since alpha has no batch API.
    pub(crate) last_alpha: u8,
    /// The tint last written to the facade (fresh-instance default:
    /// white) — same change-only discipline as `last_alpha`.
    pub(crate) last_tint: u32,
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
    /// Instances whose particles died since the last drain —
    /// [`sync`](Self::sync) removes these from the facade.
    dead_instances: Vec<SpriteInstanceId>,
    max_particles: usize,
    dropped_spawns: u64,
    stale_model_kills: u64,
    /// Persistent scratch for the per-frame transform batch (PF
    /// lesson: no per-frame allocs in the hot loop).
    xf_scratch: Vec<(SpriteInstanceId, DynSpriteTransform)>,
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
            stale_model_kills: 0,
            xf_scratch: Vec::new(),
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
    /// calls, unit-testable without a window or GPU. Never collides;
    /// for [`CollisionMode`] emitters use
    /// [`update_with_scene`](Self::update_with_scene).
    pub fn update(&mut self, dt: f64) {
        self.step(dt, None);
    }

    /// [`update`](Self::update) + voxel collision (PS.3): emitters
    /// with a non-[`None`](CollisionMode::None) [`CollisionMode`] test
    /// each particle's post-step position against `scene`'s solid
    /// voxels. See [`CollisionMode`] for the sampling caveats.
    pub fn update_with_scene(&mut self, dt: f64, scene: &roxlap_scene::Scene) {
        self.step(dt, Some(scene));
    }

    /// The shared body of the `update` variants.
    fn step(&mut self, dt: f64, scene: Option<&roxlap_scene::Scene>) {
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
            let prev = p.pos;
            for a in 0..3 {
                p.vel[a] += (def.gravity[a] - def.drag * p.vel[a]) * dtf;
                p.pos[a] += p.vel[a] * dtf;
            }
            // PS.3 — voxel collision at the post-step position (see
            // `CollisionMode` for the sampling caveats).
            if let Some(scene) = scene {
                if !matches!(def.collision, CollisionMode::None)
                    && scene_solid_ahead(scene, p.pos, p.vel)
                {
                    match def.collision {
                        CollisionMode::Kill => {
                            p.age = p.lifetime; // swept below
                            continue;
                        }
                        CollisionMode::Bounce { restitution } => {
                            // Reflect the axes whose voxel boundary
                            // was crossed this step; a same-cell hit
                            // (spawned against a wall / nudge-early
                            // contact) reverses outright.
                            let mut crossed = false;
                            for ((prev_a, pos_a), vel_a) in prev.iter().zip(&p.pos).zip(&mut p.vel)
                            {
                                if prev_a.floor() != pos_a.floor() {
                                    *vel_a = -*vel_a;
                                    crossed = true;
                                }
                            }
                            for vel_a in &mut p.vel {
                                if !crossed {
                                    *vel_a = -*vel_a;
                                }
                                *vel_a *= restitution;
                            }
                            p.pos = prev;
                        }
                        CollisionMode::None => unreachable!("guarded above"),
                    }
                }
            }
            p.yaw += p.spin_rate * dtf;
            // Over-life curves (PS.2), all keyed on the life fraction.
            let frac = p.age / p.lifetime;
            if let Some(end) = def.scale_end {
                p.scale = def.scale + (end - def.scale) * frac;
            }
            if let Some(end) = def.tint_end {
                p.tint = lerp_tint(def.tint, end, frac);
            }
            p.alpha = fade_alpha(p.age, p.lifetime, def.fade_in_frac, def.fade_out_frac);
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
    /// last drain. [`sync`](Self::sync) removes each via
    /// [`remove_sprite_instance`](crate::SceneRenderer::remove_sprite_instance);
    /// hosts doing their own rendering can consume it directly.
    pub fn drain_dead_instances(&mut self) -> impl Iterator<Item = SpriteInstanceId> + '_ {
        self.dead_instances.drain(..)
    }

    /// Newborn spawns that failed because the emitter's
    /// [`SpriteModelId`] went stale (the model was removed / the
    /// registry was reset). Each failure kills its particle — a
    /// climbing value means an emitter outlived its model.
    #[must_use]
    pub fn stale_model_kills(&self) -> u64 {
        self.stale_model_kills
    }

    /// Mirror the simulation into `renderer` (PS.1): despawn the dead,
    /// spawn newborns **pre-posed** (the documented streaming-spawn
    /// path — no one-frame axis-aligned flash) with their one-time
    /// material/lighting/shadow/tint setup, batch-move everything else
    /// via one
    /// [`set_sprite_instance_transforms`](SceneRenderer::set_sprite_instance_transforms),
    /// and write alpha only for particles whose fade actually changed
    /// (alpha has no batch API). Call after
    /// [`update`](Self::update), before
    /// [`render`](SceneRenderer::render) — or use
    /// [`tick`](Self::tick).
    pub fn sync(&mut self, renderer: &mut SceneRenderer) {
        self.sync_with(renderer);
    }

    /// [`update`](Self::update) + [`sync`](Self::sync) in one call —
    /// the per-frame default, named after the facade's own
    /// [`tick`](SceneRenderer::tick).
    pub fn tick(&mut self, renderer: &mut SceneRenderer, dt: f64) {
        self.step(dt, None);
        self.sync_with(renderer);
    }

    /// [`update_with_scene`](Self::update_with_scene) +
    /// [`sync`](Self::sync) — [`tick`](Self::tick) for hosts using
    /// [`CollisionMode`] emitters. Call before handing the same scene
    /// to [`render`](SceneRenderer::render).
    pub fn tick_with_scene(
        &mut self,
        renderer: &mut SceneRenderer,
        dt: f64,
        scene: &roxlap_scene::Scene,
    ) {
        self.step(dt, Some(scene));
        self.sync_with(renderer);
    }

    /// [`sync`](Self::sync) against the internal facade seam — the
    /// testable core (see [`ParticleFacade`]).
    fn sync_with<F: ParticleFacade>(&mut self, facade: &mut F) {
        // 1. Dead first — frees instance slots before this frame's
        //    spawns reuse them.
        for id in self.dead_instances.drain(..) {
            facade.despawn(id);
        }

        // 2. One pass: spawn newborns, batch-collect live moves.
        let mut batch = std::mem::take(&mut self.xf_scratch);
        batch.clear();
        let mut i = 0;
        while i < self.particles.len() {
            if self.particles[i].instance.is_none() {
                let (slot, xf) = {
                    let p = &self.particles[i];
                    (p.emitter_slot as usize, particle_xf(p))
                };
                let (model, material, lighting, shadows) = {
                    let st = self.emitters[slot]
                        .as_ref()
                        .expect("live particle ⇒ emitter state retained");
                    (
                        st.def.model,
                        st.def.material,
                        st.def.lighting,
                        st.def.shadows,
                    )
                };
                let Some(id) = facade.spawn(model, xf) else {
                    // Stale model handle — this particle can never
                    // render; kill it now rather than retry per frame.
                    self.stale_model_kills += 1;
                    self.particles.swap_remove(i);
                    self.on_particle_died(slot);
                    continue;
                };
                // One-time setup, skipping facade defaults (spawn
                // already left the instance there).
                if material != 0 {
                    facade.set_material(id, material);
                }
                if lighting != BillboardLighting::default() {
                    facade.set_lighting(id, lighting);
                }
                if shadows != ShadowFlags::default() {
                    facade.set_shadows(id, shadows);
                }
                let p = &mut self.particles[i];
                p.instance = Some(id);
                // Fresh-instance facade defaults; the change-only
                // blocks below write the real values (tint rides them
                // too — it can lerp per frame from PS.2 on).
                p.last_alpha = 255;
                p.last_tint = 0x00FF_FFFF;
            } else {
                // Live: frame-0 pose came from the posed spawn, so
                // only pre-existing instances join the move batch.
                let p = &self.particles[i];
                batch.push((p.instance.expect("checked above"), particle_xf(p)));
            }
            let p = &mut self.particles[i];
            if p.alpha != p.last_alpha {
                facade.set_alpha(p.instance.expect("set above"), p.alpha);
                p.last_alpha = p.alpha;
            }
            if p.tint != p.last_tint {
                facade.set_tint(p.instance.expect("set above"), p.tint);
                p.last_tint = p.tint;
            }
            i += 1;
        }
        if !batch.is_empty() {
            facade.set_transforms(&batch);
        }
        self.xf_scratch = batch;
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
            // Position: emitter pos + the shape's offset (PS.2).
            let mut pos = def.pos;
            match def.shape {
                EmitterShape::Point => {}
                EmitterShape::Sphere { radius } => {
                    // Uniform in the ball: direction × r·∛u.
                    let dir = self.rng.unit_vec();
                    let r = radius * self.rng.next_f32().cbrt();
                    for a in 0..3 {
                        pos[a] += dir[a] * r;
                    }
                }
                EmitterShape::Box { half } => {
                    for a in 0..3 {
                        pos[a] += (self.rng.next_f32() * 2.0 - 1.0) * half[a];
                    }
                }
            }

            // Velocity: base + isotropic spread + cone, by addition.
            let mut vel = def.velocity.base;
            if def.velocity.spread > 0.0 {
                let dir = self.rng.unit_vec();
                let speed = self.rng.next_f32() * def.velocity.spread;
                for a in 0..3 {
                    vel[a] += dir[a] * speed;
                }
            }
            if let Some(cone) = &def.velocity.cone {
                let dir = cone_dir(&mut self.rng, cone.axis, cone.half_angle_deg);
                let speed = self.rng.range_f32(&cone.speed);
                for a in 0..3 {
                    vel[a] += dir[a] * speed;
                }
            }

            let lifetime = self.rng.range_f32(&def.lifetime).max(1e-3);
            self.particles.push(Particle {
                pos,
                vel,
                age: 0.0,
                lifetime,
                scale: def.scale,
                yaw: 0.0,
                spin_rate: self.rng.range_f32(&def.spin),
                alpha: fade_alpha(0.0, lifetime, def.fade_in_frac, def.fade_out_frac),
                tint: def.tint,
                emitter_slot: slot as u32,
                instance: None,
                last_alpha: 255,
                last_tint: 0x00FF_FFFF,
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

/// The pose a particle renders at: position + a basis rotated by
/// [`Particle::yaw`] about the world vertical and uniformly scaled by
/// [`Particle::scale`]. The scale clamps to ≥ 0.05 — a degenerate
/// basis makes the facade silently skip drawing, which is right for a
/// kill but wrong for a fade.
fn particle_xf(p: &Particle) -> DynSpriteTransform {
    let k = p.scale.max(0.05);
    let (c, s) = (p.yaw.cos() * k, p.yaw.sin() * k);
    DynSpriteTransform {
        pos: p.pos,
        right: [c, s, 0.0],
        up: [-s, c, 0.0],
        forward: [0.0, 0.0, k],
    }
}

/// Uniform random direction within `half_angle_deg` of `axis` —
/// uniform over the spherical cap, so a wide cone doesn't bunch at the
/// rim. A degenerate axis falls back to straight up (`[0, 0, -1]`).
fn cone_dir(rng: &mut Pcg32, axis: [f32; 3], half_angle_deg: f32) -> [f32; 3] {
    let len2 = axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2];
    let w = if len2 > 1e-12 {
        let inv = 1.0 / len2.sqrt();
        [axis[0] * inv, axis[1] * inv, axis[2] * inv]
    } else {
        [0.0, 0.0, -1.0]
    };
    // cos θ uniform on [cos half, 1] ⇒ uniform area on the cap.
    let cos_half = half_angle_deg.clamp(0.0, 180.0).to_radians().cos();
    let cz = 1.0 - rng.next_f32() * (1.0 - cos_half);
    let sz = (1.0 - cz * cz).max(0.0).sqrt();
    let phi = rng.next_f32() * std::f32::consts::TAU;
    // Any orthonormal frame around `w`: cross with the axis `w` leans
    // on least.
    let t = if w[0].abs() < 0.5 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let u = {
        let c = [
            w[1] * t[2] - w[2] * t[1],
            w[2] * t[0] - w[0] * t[2],
            w[0] * t[1] - w[1] * t[0],
        ];
        let inv = 1.0 / (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
        [c[0] * inv, c[1] * inv, c[2] * inv]
    };
    let v = [
        w[1] * u[2] - w[2] * u[1],
        w[2] * u[0] - w[0] * u[2],
        w[0] * u[1] - w[1] * u[0],
    ];
    let (cp, sp) = (phi.cos() * sz, phi.sin() * sz);
    [
        w[0] * cz + u[0] * cp + v[0] * sp,
        w[1] * cz + u[1] * cp + v[1] * sp,
        w[2] * cz + u[2] * cp + v[2] * sp,
    ]
}

/// Whether a solid voxel sits at `pos` nudged half a voxel along
/// `vel` — [`Scene::resolve_voxel`]'s picking nudge doing collision
/// look-ahead duty. Zero velocity returns `false` (a resting particle
/// never re-collides).
///
/// [`Scene::resolve_voxel`]: roxlap_scene::Scene::resolve_voxel
fn scene_solid_ahead(scene: &roxlap_scene::Scene, pos: [f32; 3], vel: [f32; 3]) -> bool {
    let world = glam::DVec3::new(f64::from(pos[0]), f64::from(pos[1]), f64::from(pos[2]));
    let dir = glam::DVec3::new(f64::from(vel[0]), f64::from(vel[1]), f64::from(vel[2]));
    scene.resolve_voxel(world, dir).is_some()
}

/// Per-channel lerp of two packed `0x00RRGGBB` tints.
fn lerp_tint(a: u32, b: u32, t: f32) -> u32 {
    let t = t.clamp(0.0, 1.0);
    let ch = |sh: u32| {
        let (ca, cb) = ((a >> sh) & 0xff, (b >> sh) & 0xff);
        let m = ca as f32 + (cb as f32 - ca as f32) * t;
        ((m as u32) & 0xff) << sh
    };
    ch(16) | ch(8) | ch(0)
}

/// The slice of [`SceneRenderer`] that [`ParticleSystem::sync`]
/// drives — an internal seam so the binding logic (spawn ordering,
/// one-time setup, change-only alpha writes) is unit-testable with a
/// mock, since constructing a real backend needs a window. The facade
/// impl is pure forwarding.
pub(crate) trait ParticleFacade {
    fn spawn(&mut self, model: SpriteModelId, xf: DynSpriteTransform) -> Option<SpriteInstanceId>;
    fn despawn(&mut self, id: SpriteInstanceId);
    fn set_transforms(&mut self, batch: &[(SpriteInstanceId, DynSpriteTransform)]);
    fn set_alpha(&mut self, id: SpriteInstanceId, alpha: u8);
    fn set_tint(&mut self, id: SpriteInstanceId, tint: u32);
    fn set_material(&mut self, id: SpriteInstanceId, material: u8);
    fn set_lighting(&mut self, id: SpriteInstanceId, mode: BillboardLighting);
    fn set_shadows(&mut self, id: SpriteInstanceId, flags: ShadowFlags);
}

impl ParticleFacade for SceneRenderer {
    fn spawn(&mut self, model: SpriteModelId, xf: DynSpriteTransform) -> Option<SpriteInstanceId> {
        self.add_sprite_instance_posed(model, xf)
    }
    fn despawn(&mut self, id: SpriteInstanceId) {
        // A stale handle here means the host reset the registry
        // (`set_sprites`) — already gone, nothing owed.
        self.remove_sprite_instance(id);
    }
    fn set_transforms(&mut self, batch: &[(SpriteInstanceId, DynSpriteTransform)]) {
        self.set_sprite_instance_transforms(batch);
    }
    fn set_alpha(&mut self, id: SpriteInstanceId, alpha: u8) {
        self.set_sprite_instance_alpha(id, alpha);
    }
    fn set_tint(&mut self, id: SpriteInstanceId, tint: u32) {
        self.set_sprite_instance_tint(id, tint);
    }
    fn set_material(&mut self, id: SpriteInstanceId, material: u8) {
        self.set_sprite_instance_material(id, material);
    }
    fn set_lighting(&mut self, id: SpriteInstanceId, mode: BillboardLighting) {
        self.set_sprite_instance_lighting(id, mode);
    }
    fn set_shadows(&mut self, id: SpriteInstanceId, flags: ShadowFlags) {
        self.set_sprite_instance_shadow_flags(id, flags);
    }
}

/// Alpha for `age` of `lifetime`: ramps 0 → 255 over the leading
/// `in_frac` (PS.2), 255 → 0 over the trailing `out_frac`, 255 in
/// between; overlapping windows take the darker of the two.
fn fade_alpha(age: f32, lifetime: f32, in_frac: f32, out_frac: f32) -> u8 {
    let frac = age / lifetime;
    let mut a = 1.0_f32;
    if in_frac > 0.0 {
        a = a.min((frac / in_frac.min(1.0)).clamp(0.0, 1.0));
    }
    if out_frac > 0.0 {
        a = a.min(((1.0 - frac) / out_frac.min(1.0)).clamp(0.0, 1.0));
    }
    (a * 255.0) as u8
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
            fade_out_frac: 0.0,
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
                    ..VelocityDef::default()
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
        assert_eq!(fade_alpha(0.0, 1.0, 0.0, 0.5), 255);
        assert_eq!(fade_alpha(0.5, 1.0, 0.0, 0.5), 255); // window edge
        assert_eq!(fade_alpha(0.75, 1.0, 0.0, 0.5), 127); // mid-fade
        assert_eq!(fade_alpha(1.0, 1.0, 0.0, 0.5), 0);
        assert_eq!(fade_alpha(0.99, 1.0, 0.0, 0.0), 255); // no fade
                                                          // Fade-in (PS.2): born dark, ramps up, then the out-window
                                                          // takes over; overlap picks the darker.
        assert_eq!(fade_alpha(0.0, 1.0, 0.25, 0.0), 0);
        assert_eq!(fade_alpha(0.125, 1.0, 0.25, 0.0), 127);
        assert_eq!(fade_alpha(0.25, 1.0, 0.25, 0.0), 255);
        assert_eq!(fade_alpha(0.5, 1.0, 1.0, 1.0), 127); // crossover
    }

    #[test]
    fn shapes_sample_within_bounds() {
        let mut sys = ParticleSystem::new(20);
        let sphere = sys.add_emitter(ParticleEmitterDef {
            pos: [10.0, 0.0, 0.0],
            shape: EmitterShape::Sphere { radius: 3.0 },
            ..base_def()
        });
        let boxy = sys.add_emitter(ParticleEmitterDef {
            pos: [-10.0, 0.0, 0.0],
            shape: EmitterShape::Box {
                half: [1.0, 2.0, 0.5],
            },
            ..base_def()
        });
        sys.burst(sphere, 64);
        sys.burst(boxy, 64);
        let mut spread = false;
        for p in sys.particles() {
            if p.pos[0] > 0.0 {
                let d = [p.pos[0] - 10.0, p.pos[1], p.pos[2]];
                let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                assert!(r <= 3.0 + 1e-4, "sphere sample outside radius: {r}");
                spread |= r > 0.1;
            } else {
                let d = [p.pos[0] + 10.0, p.pos[1], p.pos[2]];
                assert!(
                    d[0].abs() <= 1.0 + 1e-4
                        && d[1].abs() <= 2.0 + 1e-4
                        && d[2].abs() <= 0.5 + 1e-4,
                    "box sample outside half-extents: {d:?}"
                );
                spread |= d[1].abs() > 1.0; // uses the wide axis
            }
        }
        assert!(spread, "samples must actually spread out");
    }

    #[test]
    fn cone_stays_within_half_angle() {
        let mut sys = ParticleSystem::new(21);
        let axis = [0.0, 1.0, -1.0]; // non-unit on purpose
        let em = sys.add_emitter(ParticleEmitterDef {
            velocity: VelocityDef {
                cone: Some(ConeDef {
                    axis,
                    half_angle_deg: 30.0,
                    speed: 5.0..10.0,
                }),
                ..VelocityDef::default()
            },
            ..base_def()
        });
        sys.burst(em, 64);
        let inv = 1.0 / (2.0_f32).sqrt();
        let w = [0.0, inv, -inv];
        let cos_half = 30.0_f32.to_radians().cos();
        for p in sys.particles() {
            let sp = (p.vel[0] * p.vel[0] + p.vel[1] * p.vel[1] + p.vel[2] * p.vel[2]).sqrt();
            assert!(
                (5.0 - 1e-3..10.0 + 1e-3).contains(&sp),
                "speed in range: {sp}"
            );
            let cosang = (p.vel[0] * w[0] + p.vel[1] * w[1] + p.vel[2] * w[2]) / sp;
            assert!(
                cosang >= cos_half - 1e-4,
                "direction within the cone: cos {cosang} < {cos_half}"
            );
        }
        // A zero axis falls back to straight up (-z).
        assert_eq!(
            cone_dir(&mut Pcg32::new(1), [0.0; 3], 0.0),
            [0.0, 0.0, -1.0]
        );
    }

    #[test]
    fn spin_rotates_pose_and_keeps_scale() {
        let mut sys = ParticleSystem::new(22);
        let em = sys.add_emitter(ParticleEmitterDef {
            spin: 2.0..2.0,
            scale: 3.0,
            lifetime: 100.0..100.0,
            ..base_def()
        });
        sys.burst(em, 1);
        sys.update(0.25); // yaw = 0.5 rad
        let p = sys.particles()[0];
        assert!((p.yaw - 0.5).abs() < 1e-6);
        let xf = particle_xf(&p);
        // Columns stay orthogonal with length == scale.
        let len = |v: [f32; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        assert!((len(xf.right) - 3.0).abs() < 1e-5);
        assert!((len(xf.up) - 3.0).abs() < 1e-5);
        assert!(xf.right[1].abs() > 0.1, "yaw actually rotates the basis");
        let dot = xf.right[0] * xf.up[0] + xf.right[1] * xf.up[1];
        assert!(dot.abs() < 1e-5);
    }

    #[test]
    fn scale_and_tint_lerp_over_life() {
        let mut sys = ParticleSystem::new(23);
        let em = sys.add_emitter(ParticleEmitterDef {
            lifetime: 1.0..1.0,
            scale: 1.0,
            scale_end: Some(3.0),
            tint: 0x00FF_0000,
            tint_end: Some(0x0000_00FF),
            ..base_def()
        });
        sys.burst(em, 1);
        sys.update(0.5);
        let p = sys.particles()[0];
        assert!((p.scale - 2.0).abs() < 1e-5, "mid-life scale: {}", p.scale);
        let (r, b) = ((p.tint >> 16) & 0xff, p.tint & 0xff);
        assert!(
            (126..=128).contains(&r) && (126..=128).contains(&b),
            "mid tint: {:#08x}",
            p.tint
        );

        // sync writes the lerping tint every frame it changes.
        let mut f = Mock::default();
        sys.sync_with(&mut f);
        assert_eq!(f.tints.len(), 1);
        sys.update(0.1);
        sys.sync_with(&mut f);
        assert_eq!(f.tints.len(), 2, "changed tint re-syncs");
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

    /// Records every facade call `sync` makes; mints sequential ids.
    #[derive(Default)]
    struct Mock {
        next_slot: u32,
        fail_spawn: bool,
        spawns: Vec<(SpriteModelId, DynSpriteTransform)>,
        despawns: Vec<SpriteInstanceId>,
        batch_sizes: Vec<usize>,
        alphas: Vec<(SpriteInstanceId, u8)>,
        tints: Vec<u32>,
        materials: Vec<u8>,
        lightings: Vec<BillboardLighting>,
        shadows: Vec<ShadowFlags>,
    }

    impl ParticleFacade for Mock {
        fn spawn(
            &mut self,
            model: SpriteModelId,
            xf: DynSpriteTransform,
        ) -> Option<SpriteInstanceId> {
            if self.fail_spawn {
                return None;
            }
            self.spawns.push((model, xf));
            let id = SpriteInstanceId {
                slot: self.next_slot,
                gen: 0,
            };
            self.next_slot += 1;
            Some(id)
        }
        fn despawn(&mut self, id: SpriteInstanceId) {
            self.despawns.push(id);
        }
        fn set_transforms(&mut self, batch: &[(SpriteInstanceId, DynSpriteTransform)]) {
            self.batch_sizes.push(batch.len());
        }
        fn set_alpha(&mut self, id: SpriteInstanceId, alpha: u8) {
            self.alphas.push((id, alpha));
        }
        fn set_tint(&mut self, _id: SpriteInstanceId, tint: u32) {
            self.tints.push(tint);
        }
        fn set_material(&mut self, _id: SpriteInstanceId, material: u8) {
            self.materials.push(material);
        }
        fn set_lighting(&mut self, _id: SpriteInstanceId, mode: BillboardLighting) {
            self.lightings.push(mode);
        }
        fn set_shadows(&mut self, _id: SpriteInstanceId, flags: ShadowFlags) {
            self.shadows.push(flags);
        }
    }

    #[test]
    fn sync_spawns_once_then_batch_moves() {
        let mut sys = ParticleSystem::new(10);
        let em = sys.add_emitter(ParticleEmitterDef {
            lifetime: 100.0..100.0,
            ..base_def()
        });
        sys.burst(em, 3);
        let mut f = Mock::default();

        sys.sync_with(&mut f);
        // Newborns spawn pre-posed — no move batch for them.
        assert_eq!(f.spawns.len(), 3);
        assert!(f.batch_sizes.is_empty());
        // Particle shadows default off ≠ facade default (cast+receive)
        // ⇒ set once each; everything else is at facade defaults.
        assert_eq!(f.shadows.len(), 3);
        assert!(f.materials.is_empty() && f.tints.is_empty() && f.lightings.is_empty());
        assert!(f.alphas.is_empty(), "alpha 255 == facade default");

        // Next frame: no new spawns, one batch of 3.
        sys.update(0.01);
        sys.sync_with(&mut f);
        assert_eq!(f.spawns.len(), 3);
        assert_eq!(f.batch_sizes, vec![3]);
    }

    #[test]
    fn sync_one_time_setup_honours_def() {
        let mut sys = ParticleSystem::new(11);
        let em = sys.add_emitter(ParticleEmitterDef {
            lifetime: 100.0..100.0,
            tint: 0x00FF_0000,
            material: 5,
            lighting: BillboardLighting::FullBright,
            shadows: ShadowFlags::default(), // back to facade default
            ..base_def()
        });
        sys.burst(em, 1);
        let mut f = Mock::default();
        sys.sync_with(&mut f);
        assert_eq!(f.materials, vec![5]);
        assert_eq!(f.tints, vec![0x00FF_0000]);
        assert_eq!(f.lightings, vec![BillboardLighting::FullBright]);
        assert!(f.shadows.is_empty(), "facade-default shadows skip the call");
        // Re-sync repeats none of it: still one call per family.
        sys.update(0.01);
        sys.sync_with(&mut f);
        assert_eq!(f.materials.len() + f.tints.len() + f.lightings.len(), 3);
    }

    #[test]
    fn sync_despawns_dead_and_writes_alpha_on_change_only() {
        let mut sys = ParticleSystem::new(12);
        let em = sys.add_emitter(ParticleEmitterDef {
            lifetime: 1.0..1.0,
            fade_out_frac: 0.5,
            ..base_def()
        });
        sys.burst(em, 2);
        let mut f = Mock::default();
        sys.sync_with(&mut f);

        // Pre-fade window: alpha stays 255, no writes.
        sys.update(0.25);
        sys.sync_with(&mut f);
        assert!(f.alphas.is_empty());

        // Inside the fade window: one write per particle per change.
        sys.update(0.5); // age 0.75 ⇒ alpha 127
        sys.sync_with(&mut f);
        assert_eq!(f.alphas.len(), 2);
        assert!(f.alphas.iter().all(|&(_, a)| a == 127));

        // Death ⇒ despawn of exactly the minted ids.
        sys.update(0.5);
        assert_eq!(sys.particle_count(), 0);
        sys.sync_with(&mut f);
        assert_eq!(f.despawns.len(), 2);
    }

    #[test]
    fn stale_model_spawn_kills_particle() {
        let mut sys = ParticleSystem::new(13);
        let em = sys.add_emitter(base_def());
        sys.burst(em, 2);
        let mut f = Mock {
            fail_spawn: true,
            ..Mock::default()
        };
        sys.sync_with(&mut f);
        assert_eq!(sys.particle_count(), 0);
        assert_eq!(sys.stale_model_kills(), 2);
        // The emitter's live count drained cleanly: removing it frees
        // the slot immediately.
        assert!(sys.remove_emitter(em));
        assert!(sys.emitters.iter().all(Option::is_none));
    }

    #[test]
    fn particle_xf_scales_and_clamps() {
        let p = Particle {
            pos: [1.0, 2.0, 3.0],
            vel: [0.0; 3],
            age: 0.0,
            lifetime: 1.0,
            scale: 2.0,
            yaw: 0.0,
            spin_rate: 0.0,
            alpha: 255,
            tint: 0,
            emitter_slot: 0,
            instance: None,
            last_alpha: 255,
            last_tint: 0x00FF_FFFF,
        };
        let xf = particle_xf(&p);
        assert_eq!(xf.pos, [1.0, 2.0, 3.0]);
        assert_eq!((xf.right[0], xf.up[1], xf.forward[2]), (2.0, 2.0, 2.0));
        // Degenerate scale clamps to the visible minimum.
        let tiny = particle_xf(&Particle { scale: 0.0, ..p });
        assert_eq!(tiny.right[0], 0.05);
    }

    /// A scene with a solid slab: z ∈ 10..=12 across generous xy.
    fn scene_with_floor() -> roxlap_scene::Scene {
        use glam::{DVec3, IVec3};
        let mut scene = roxlap_scene::Scene::new();
        let id = scene.add_grid(roxlap_scene::GridTransform::at(DVec3::ZERO));
        let g = scene.grid_mut(id).expect("fresh grid");
        g.set_rect(
            IVec3::new(-16, -16, 10),
            IVec3::new(16, 16, 12),
            Some(0x80FF_FFFF),
        );
        scene
    }

    /// An emitter dropping one particle straight down (+z) at the slab.
    fn falling(collision: CollisionMode) -> ParticleEmitterDef {
        ParticleEmitterDef {
            pos: [0.0, 0.0, 5.0],
            velocity: VelocityDef {
                base: [0.0, 0.0, 20.0],
                ..VelocityDef::default()
            },
            lifetime: 100.0..100.0,
            collision,
            ..base_def() // zero gravity — constant fall speed
        }
    }

    #[test]
    fn collision_kill_dies_on_contact() {
        let scene = scene_with_floor();
        let mut sys = ParticleSystem::new(30);
        let em = sys.add_emitter(falling(CollisionMode::Kill));
        sys.burst(em, 1);
        for _ in 0..20 {
            sys.update_with_scene(0.05, &scene);
        }
        assert_eq!(sys.particle_count(), 0, "dies at the slab");

        // The scene-free update never collides, whatever the mode.
        let mut free = ParticleSystem::new(30);
        let em = free.add_emitter(falling(CollisionMode::Kill));
        free.burst(em, 1);
        for _ in 0..20 {
            free.update(0.05);
        }
        assert_eq!(free.particle_count(), 1);
        assert!(free.particles()[0].pos[2] > 12.0, "fell straight through");
    }

    #[test]
    fn collision_none_passes_through() {
        let scene = scene_with_floor();
        let mut sys = ParticleSystem::new(31);
        let em = sys.add_emitter(falling(CollisionMode::None));
        sys.burst(em, 1);
        for _ in 0..20 {
            sys.update_with_scene(0.05, &scene);
        }
        assert!(sys.particles()[0].pos[2] > 12.0);
    }

    #[test]
    fn collision_bounce_reflects_and_damps() {
        let scene = scene_with_floor();
        let mut sys = ParticleSystem::new(32);
        let em = sys.add_emitter(falling(CollisionMode::Bounce { restitution: 0.5 }));
        sys.burst(em, 1);
        let mut bounced = false;
        for _ in 0..40 {
            sys.update_with_scene(0.05, &scene);
            let p = sys.particles()[0];
            assert!(p.pos[2] < 10.5, "never sinks into the slab: {}", p.pos[2]);
            if p.vel[2] < 0.0 {
                bounced = true;
            }
        }
        assert!(bounced, "velocity reflected upward");
        let p = sys.particles()[0];
        assert!(
            p.vel[2].abs() <= 10.0 + 1e-3,
            "restitution halves the speed: {}",
            p.vel[2]
        );
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
