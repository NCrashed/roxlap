//! Character controller (stage CC.1 — see
//! `docs/porting/PORTING-CONTROLLER.md`).
//!
//! A walking body over a [`Scene`]: substepped per-axis
//! move-and-slide against the [`crate::collide`] probe, gravity,
//! jumping, ground / head-bump detection — and swimming (stage WT.1):
//! a `Walk`-mode body submerged past
//! [`CharacterDef::swim_enter_frac`] of its height in a
//! [`crate::WaterVolume`] enters the swim state automatically
//! (buoyancy, vertical drag, `jump` strokes up / `sink` strokes down,
//! a jump with the head above water breaches). Not to be confused with
//! `.rkc` characters (`roxlap-formats`' animated-model container) —
//! this is what you *stand on the ground* with; a `.rkc` character is
//! what you *draw* (CC.4 connects the two).
//!
//! Conventions (get the signs wrong and everything compiles and
//! "works" upside down): **+z is DOWN**, so gravity is *positive* z,
//! a jump impulse is *negative* z, the body's feet are its
//! largest-z end and its head is at `feet.z - height`. Positions and
//! velocities are f64 world space, matching the camera and
//! [`crate::GridTransform`].
//!
//! Movement is deterministic — pure f64, no RNG, a fixed substep
//! rule — so trajectories are unit-testable: same scene + same input
//! sequence = identical path.

use glam::{DVec2, DVec3};

use crate::collide::{box_overlaps_solid, Solidity};
use crate::Scene;

/// Collision skin: contact rests this far off the blocking plane so
/// the next probe of the resting pose stays clear.
const SKIN: f64 = 1e-3;

/// Hard cap on substeps per `walk` call — an anti-hang guard, far
/// above any sane displacement (at the default radius this is ~4000
/// voxels per call). Past it the displacement is truncated to keep
/// the no-tunnel guarantee rather than probing less often.
const MAX_SUBSTEPS: u32 = 10_000;

/// Construction-time parameters of a [`CharacterBody`]. Distances in
/// voxels (= world units), times in seconds.
#[derive(Debug, Clone, Copy)]
pub struct CharacterDef {
    /// Half-extent of the collision box in x and y.
    pub radius: f64,
    /// Feet → head extent. The body occupies
    /// `z ∈ [feet.z - height, feet.z]` (+z is down).
    pub height: f64,
    /// Feet → eye distance for [`CharacterBody::eye_pos`] (the
    /// camera anchor), along the same up-is-−z axis.
    pub eye_height: f64,
    /// Gravity acceleration, **positive** (+z is down).
    pub gravity: f64,
    /// Initial upward speed of a jump, applied as **negative** z
    /// velocity.
    pub jump_speed: f64,
    /// Terminal fall speed (+z velocity cap). Besides realism, this
    /// bounds the per-frame probe cost: substeps scale with
    /// displacement, so an unbounded fall gets linearly more
    /// expensive the longer it lasts (the CC.5 stress probe caught
    /// exactly that).
    pub max_fall_speed: f64,
    /// Target horizontal speed while walking.
    pub walk_speed: f64,
    /// How fast the horizontal velocity approaches the wish
    /// direction on the ground, in speed units per second. Also the
    /// stopping (friction) rate — with no input the target is zero.
    pub accel_ground: f64,
    /// Same, airborne — low, so jumps keep their momentum but retain
    /// a little steering.
    pub accel_air: f64,
    /// Auto-step height: a grounded body blocked horizontally climbs
    /// ledges up to this many voxels tall, if the lifted body fits
    /// and finds ground on the far side. `1.05` clears 1-voxel
    /// stairs; set `0.0` to disable.
    pub step_up: f64,
    /// Grace window after walking off an edge during which a jump
    /// still fires (seconds) — "coyote time".
    pub coyote_time: f64,
    /// How long a jump pressed in mid-air stays queued and fires on
    /// landing (seconds).
    pub jump_buffer: f64,
    /// Target speed in [`MoveMode::Fly`] / [`MoveMode::Noclip`],
    /// where the full 3D `wish` steers.
    pub fly_speed: f64,
    /// Acceleration toward the wish in the fly modes. The default
    /// `f64::INFINITY` means instant start/stop — a fly camera, not
    /// a body with inertia (the demos' classic feel); set a finite
    /// rate for drifty flight.
    pub fly_accel: f64,
    /// What counts as solid (bedrock-placeholder policy — must match
    /// how the host *renders* the world; see
    /// [`Solidity::bedrock_blocks`]).
    pub solidity: Solidity,
    /// WT.1 — upward acceleration at FULL submersion, **positive**
    /// (applied toward −z, fighting `gravity`). With no stroke input
    /// the body settles where `gravity == buoyancy · fraction`; the
    /// defaults (24 / 32) float it at 75% submerged — head above the
    /// surface.
    pub buoyancy: f64,
    /// WT.1 — vertical velocity damping in water, per second
    /// (exponential). Caps stroke/fall speeds under water and kills
    /// the surface bob; higher = thicker water.
    pub water_drag: f64,
    /// WT.1 — target horizontal speed while swimming.
    pub swim_speed: f64,
    /// WT.1 — acceleration while swimming: the horizontal approach
    /// rate AND the vertical stroke strength (`jump` up / `sink`
    /// down).
    pub swim_accel: f64,
    /// WT.1 — the swim state ENGAGES at this submerged fraction of
    /// the body height. The default `0.6` deliberately lets the
    /// default 1.8-body WADE through minimal authored water (one
    /// voxel over the floor = fraction ≈ 0.56): a decorative puddle
    /// must not swap walking for swimming.
    pub swim_enter_frac: f64,
    /// …and RELEASES below this one. The gap is deliberate hysteresis:
    /// a bobbing body at the waterline must not flicker between
    /// swimming and walking (`enter > exit`). The EFFECTIVE release
    /// threshold is additionally clamped below the no-stroke bobbing
    /// equilibrium (`gravity / buoyancy`) at runtime, so a very
    /// buoyant "cork" tuning can't limit-cycle at the waterline.
    pub swim_exit_frac: f64,
}

impl Default for CharacterDef {
    fn default() -> Self {
        Self {
            radius: 0.4,
            height: 1.8,
            eye_height: 1.62,
            gravity: 24.0,
            jump_speed: 9.0,
            max_fall_speed: 60.0,
            walk_speed: 6.0,
            accel_ground: 40.0,
            accel_air: 8.0,
            step_up: 1.05,
            coyote_time: 0.12,
            jump_buffer: 0.12,
            fly_speed: 12.0,
            fly_accel: f64::INFINITY,
            solidity: Solidity::default(),
            buoyancy: 32.0,
            water_drag: 3.0,
            swim_speed: 4.0,
            swim_accel: 20.0,
            swim_enter_frac: 0.6,
            swim_exit_frac: 0.35,
        }
    }
}

/// How [`CharacterBody::walk`] moves the body.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MoveMode {
    /// Grounded movement: gravity, jumping, step-up, slide. The
    /// default.
    #[default]
    Walk,
    /// The demos' fly camera: no gravity, the full 3D `wish` steers
    /// at [`CharacterDef::fly_speed`], collision still slides.
    Fly,
    /// Fly without any collision probes.
    Noclip,
}

/// Per-frame input to [`CharacterBody::walk`].
#[derive(Debug, Clone, Copy, Default)]
pub struct WalkInput {
    /// Wish direction, world space. In [`MoveMode::Walk`] only x/y
    /// steer; the fly modes use all three components. Length is
    /// clamped to 1, so passing a raw WASD sum is fine — scale it
    /// *down* for analog part-speed input.
    pub wish: DVec3,
    /// Jump this frame. Fires when grounded or within the coyote
    /// window; otherwise it stays buffered for
    /// [`CharacterDef::jump_buffer`] seconds and fires on landing.
    /// While swimming (WT.1) it strokes the body UP instead — and a
    /// jump with the head already above the surface breaches into a
    /// normal jump.
    pub jump: bool,
    /// WT.1 — stroke DOWN while swimming (dive). Ignored on land and
    /// in the fly modes (they steer with the full 3D `wish`).
    pub sink: bool,
}

/// A walking body: feet-positioned collision box + velocity +
/// contact flags. Construct with [`CharacterBody::new`], place with
/// [`teleport`](Self::teleport), then call
/// [`walk`](Self::walk) once per frame.
#[derive(Debug, Clone, Copy)]
// Four bools, all genuinely independent contact/state flags (ground,
// head bump, swim, breach latch) — not an encoded state machine.
#[allow(clippy::struct_excessive_bools)]
pub struct CharacterBody {
    def: CharacterDef,
    mode: MoveMode,
    /// FEET position — the box is `pos ± radius` in x/y and
    /// `[pos.z - height, pos.z]` in z.
    pos: DVec3,
    vel: DVec3,
    on_ground: bool,
    hit_head: bool,
    /// Seconds since the body was last grounded (coyote window);
    /// `INFINITY` once a jump consumes it.
    since_grounded: f64,
    /// Seconds of buffered-jump validity left.
    jump_buffer_left: f64,
    /// WT.1 — the hysteretic swim state (see
    /// [`CharacterDef::swim_enter_frac`]). Only meaningful in
    /// [`MoveMode::Walk`]; the fly modes clear it.
    swimming: bool,
    /// WT.1 — breach one-shot: set when a breach jump fires, held
    /// until the jump input releases. Without it a HELD jump at the
    /// surface would re-impulse to full `-jump_speed` every frame
    /// (no ballistic decay — the body leaves any pool at jump speed
    /// and overshoots a normal jump's apex).
    breach_latched: bool,
}

impl CharacterBody {
    /// A body at the world origin with zero velocity — call
    /// [`teleport`](Self::teleport) before the first `walk`.
    #[must_use]
    pub fn new(def: CharacterDef) -> Self {
        Self {
            def,
            mode: MoveMode::Walk,
            pos: DVec3::ZERO,
            vel: DVec3::ZERO,
            on_ground: false,
            hit_head: false,
            since_grounded: f64::INFINITY,
            jump_buffer_left: 0.0,
            swimming: false,
            breach_latched: false,
        }
    }

    /// Current movement mode.
    #[must_use]
    pub fn mode(&self) -> MoveMode {
        self.mode
    }

    /// Switch movement mode. Velocity is kept — dropping out of
    /// `Fly` mid-air falls with whatever speed you had.
    pub fn set_mode(&mut self, mode: MoveMode) {
        self.mode = mode;
        // WT.1 — the fly modes ignore water. Clear the swim state
        // HERE, not just on the next walk(): a host polling
        // `is_swimming()` right after the switch (cutscene camera, no
        // walk() calls) must not see a stale underwater tint/lowpass.
        if mode != MoveMode::Walk {
            self.swimming = false;
            self.breach_latched = false;
        }
    }

    /// The construction parameters.
    #[must_use]
    pub fn def(&self) -> &CharacterDef {
        &self.def
    }

    /// Tune parameters at runtime — sprint (`walk_speed` /
    /// `fly_speed`), variable jump height, etc. Growing `radius` /
    /// `height` while standing in a tight spot can leave the body
    /// overlapping solid; the stuck-escape rule makes that
    /// recoverable rather than a jam.
    pub fn def_mut(&mut self) -> &mut CharacterDef {
        &mut self.def
    }

    /// Feet position, world space.
    #[must_use]
    pub fn pos(&self) -> DVec3 {
        self.pos
    }

    /// Eye position — the camera anchor: `eye_height` *above* the
    /// feet, i.e. toward −z.
    #[must_use]
    pub fn eye_pos(&self) -> DVec3 {
        self.pos - DVec3::new(0.0, 0.0, self.def.eye_height)
    }

    /// Current velocity, world units per second.
    #[must_use]
    pub fn vel(&self) -> DVec3 {
        self.vel
    }

    /// Overwrite the velocity — knockback, launch pads, spawn state.
    pub fn set_vel(&mut self, vel: DVec3) {
        self.vel = vel;
    }

    /// `true` while the feet rest on solid ground (skin probe below
    /// the feet, updated by [`walk`](Self::walk)).
    #[must_use]
    pub fn on_ground(&self) -> bool {
        self.on_ground
    }

    /// `true` if the head hit a ceiling during the last
    /// [`walk`](Self::walk).
    #[must_use]
    pub fn hit_head(&self) -> bool {
        self.hit_head
    }

    /// WT.1 — `true` while the body is in the swim state (submerged
    /// past [`CharacterDef::swim_enter_frac`], held until it falls
    /// below [`CharacterDef::swim_exit_frac`]). Updated by
    /// [`walk`](Self::walk); always `false` in the fly modes.
    #[must_use]
    pub fn is_swimming(&self) -> bool {
        self.swimming
    }

    /// WT.1 — fraction of the body height below a water surface,
    /// `0.0..=1.0`: the [`Scene::water_depth_at`] point depth at the
    /// FEET over [`CharacterDef::height`] (exact for the
    /// world-horizontal surfaces water is authored with —
    /// PORTING-WATER.md decision 1: there is deliberately no
    /// box-overlap query). This is the BODY-STATE number (it drives
    /// the swim state) — do NOT wire underwater tint/audio to it: at
    /// the default bobbing equilibrium it reads 0.75 while the head
    /// (and camera) are in the air. Use [`Self::eye_in_water`] for
    /// those.
    ///
    /// Continuity guard: feet just below an authored volume's BOTTOM
    /// face (a pool volume that stops short of the floor) read as
    /// fully submerged while the head is still in water — without
    /// this, the fraction would snap 0.55+ → 0.0 mid-dive and strobe
    /// the swim state faster than hysteresis can absorb. (Author
    /// volumes down to the floor anyway.)
    #[must_use]
    pub fn submerged_fraction(&self, scene: &Scene) -> f64 {
        if let Some((_, depth)) = scene.water_depth_at(self.pos) {
            return (depth / self.def.height).clamp(0.0, 1.0);
        }
        let head = self.pos - DVec3::new(0.0, 0.0, self.def.height);
        if scene.in_water(head) {
            1.0
        } else {
            0.0
        }
    }

    /// WT.1 — `true` while the EYE (the camera anchor,
    /// [`Self::eye_pos`]) is under a water surface. THE hook for the
    /// underwater feel — WT.2 tint, WT.3 listener lowpass: it flips
    /// exactly when the player's view goes under, unlike
    /// [`Self::submerged_fraction`] (feet-depth, 0.75 at the surface
    /// bob with a dry camera).
    #[must_use]
    pub fn eye_in_water(&self, scene: &Scene) -> bool {
        scene.in_water(self.eye_pos())
    }

    /// Reposition the feet, KEEPING velocity and contact state — for
    /// world-bounds clamps and moving-platform style corrections.
    /// For spawns and respawns use [`teleport`](Self::teleport).
    pub fn set_pos(&mut self, pos: DVec3) {
        self.pos = pos;
    }

    /// Hard-place the feet at `pos`, zeroing velocity and contact
    /// flags. No collision check — placing inside solid is allowed
    /// (the stuck-escape rule below makes it recoverable).
    pub fn teleport(&mut self, pos: DVec3) {
        self.pos = pos;
        self.vel = DVec3::ZERO;
        self.on_ground = false;
        self.hit_head = false;
        self.since_grounded = f64::INFINITY;
        self.jump_buffer_left = 0.0;
        self.swimming = false;
        self.breach_latched = false;
    }

    /// Advance the body by `dt` seconds against `scene`. Behaviour
    /// follows the current [`MoveMode`]; everything below describes
    /// [`MoveMode::Walk`].
    ///
    /// Integration order: horizontal velocity approaches
    /// `wish · walk_speed` at the ground/air accel rate; gravity;
    /// jump if grounded; then the displacement is applied in
    /// substeps of at most `radius` per axis (the no-tunnel
    /// guarantee), each substep moving x, then y, then z, clamping
    /// flush against the blocking cell plane and zeroing that
    /// velocity component on contact — unless a grounded horizontal
    /// block steps up a ledge ([`CharacterDef::step_up`]). +z contact
    /// grounds the body; −z contact is a head bump.
    ///
    /// **Stuck escape** (kept verbatim from the demos): if the body
    /// *starts* the frame overlapping solid — an edit carved under
    /// the player, a bake reclassified a column — the whole frame
    /// moves without collision so the player can escape rather than
    /// jam.
    pub fn walk(&mut self, scene: &Scene, dt: f64, input: WalkInput) {
        self.hit_head = false;
        if dt <= 0.0 {
            return;
        }
        match self.mode {
            MoveMode::Walk => self.walk_grounded(scene, dt, input),
            MoveMode::Fly => self.fly(scene, dt, input, true),
            MoveMode::Noclip => self.fly(scene, dt, input, false),
        }
    }

    fn walk_grounded(&mut self, scene: &Scene, dt: f64, input: WalkInput) {
        // -- WT.1: hysteretic swim state ----------------------------
        // A dry scene short-circuits in `water_depth_at` (no volumes),
        // so the pre-WT walk path runs unperturbed.
        let frac = self.submerged_fraction(scene);
        // BOTH thresholds are clamped around the no-stroke bobbing
        // equilibrium (gravity/buoyancy) so the resting point always
        // lies INSIDE the hysteresis band and the flag keeps tracking
        // the truth for extreme tunings: release below the
        // equilibrium (0.9× — a cork bobbing at 0.3 must not release
        // at swim_exit_frac = 0.35), engage no higher than just above
        // it (1.1× — a cork's bob never reaches the default 0.6).
        // The band alone can't prevent waterline strobing, though —
        // that is the passive-force continuity's job (see the
        // integration below). The default tuning (equilibrium 0.75)
        // leaves both knobs untouched: 0.9·0.75 > 0.35 and
        // 1.1·0.75 > 0.6.
        let equilibrium = if self.def.buoyancy > 0.0 {
            self.def.gravity / self.def.buoyancy
        } else {
            f64::INFINITY
        };
        let exit = self.def.swim_exit_frac.min(0.9 * equilibrium);
        let enter = self
            .def
            .swim_enter_frac
            .min(1.1 * equilibrium)
            .max(exit + 0.01);
        if self.swimming {
            if frac < exit {
                self.swimming = false;
            }
        } else if frac >= enter {
            self.swimming = true;
        }
        if self.swimming {
            self.swim(scene, dt, input, frac);
            return;
        }
        self.breach_latched = false; // walking re-arms the breach

        // -- integrate velocity ------------------------------------
        let accel = if self.on_ground {
            self.def.accel_ground
        } else {
            self.def.accel_air
        };
        self.steer_horizontal(input.wish, self.def.walk_speed, accel, dt);

        // WT.1 — the PASSIVE water forces (buoyancy + drag, both
        // scaled by the submerged fraction) apply in BOTH states; the
        // swim flag gates only the CONTROLS (strokes, speeds, coyote).
        // This continuity is load-bearing: if the walking body felt
        // dry gravity inside water, the force would jump by
        // `buoyancy · frac` at every engage/release crossing — a
        // relay oscillator whose hysteresis loop pumps energy each
        // cycle, so a buoyant body strobes swim → walk at the
        // waterline forever instead of settling (drag alone cannot
        // outrun the pump; measured before this fix). Besides
        // stability it is also the right feel — wading is floaty,
        // a splashing landing is cushioned. `frac == 0` reduces to
        // exactly the pre-WT integration (bit-identical, pinned).
        self.vel.z = (self.vel.z + (self.def.gravity - self.def.buoyancy * frac) * dt)
            .min(self.def.max_fall_speed);
        if frac > 0.0 {
            self.vel.z -= self.vel.z * (self.def.water_drag * frac * dt).min(1.0);
        }

        // -- jump: buffered press + coyote window ------------------
        if input.jump {
            self.jump_buffer_left = self.def.jump_buffer;
        }
        let can_jump = self.on_ground || self.since_grounded <= self.def.coyote_time;
        if self.jump_buffer_left > 0.0 && can_jump {
            self.vel.z = -self.def.jump_speed;
            self.jump_buffer_left = 0.0;
            // Consume the coyote window — no double jumps off it.
            self.since_grounded = f64::INFINITY;
            self.on_ground = false;
        }
        self.jump_buffer_left = (self.jump_buffer_left - dt).max(0.0);

        // -- move --------------------------------------------------
        if self.slide_move(scene, dt, true) {
            // Stuck escape ran: no contact state this frame (and no
            // coyote jumps off the inside of a wall).
            self.since_grounded = f64::INFINITY;
            return;
        }

        // -- ground flag + coyote clock ----------------------------
        self.on_ground = self.ground_probe(scene);
        if self.on_ground {
            self.since_grounded = 0.0;
        } else {
            self.since_grounded += dt;
        }
    }

    /// WT.1 — swimming: buoyancy scaled by the submerged fraction
    /// fights gravity (equilibrium bobbing at the surface), vertical
    /// drag caps every speed, `jump`/`sink` stroke up/down, and a
    /// jump with the head above the surface breaches into a real
    /// jump (one-shot — release the jump input to re-arm). Collision
    /// is the same slide as walking — pool walls and floor still
    /// block (step-up assists a grounded wade out).
    fn swim(&mut self, scene: &Scene, dt: f64, input: WalkInput, frac: f64) {
        // Horizontal: like walking, at swim speed/accel.
        self.steer_horizontal(input.wish, self.def.swim_speed, self.def.swim_accel, dt);

        // Breach FIRST (the stroke/drag it replaces would only be
        // overwritten): a jump with the head above the surface is a
        // real jump, not a stroke. ONE-SHOT via the latch — a held
        // jump must not re-impulse to full `-jump_speed` every frame
        // (the body would leave any pool at jump speed with no
        // ballistic decay and overshoot a normal jump's apex).
        let head = self.pos - DVec3::new(0.0, 0.0, self.def.height);
        let breach = input.jump && !self.breach_latched && !scene.in_water(head);
        self.breach_latched = input.jump && (self.breach_latched || breach);
        if breach {
            self.vel.z = -self.def.jump_speed;
        } else {
            // Vertical (+z is DOWN): gravity pulls down (+), buoyancy
            // pushes up (−) in proportion to how much of the body is
            // under; a stroke adds on top. With no stroke the body
            // settles where gravity == buoyancy·frac — bobbing at the
            // surface, above the enter threshold at the defaults (and
            // the release threshold is clamped under the equilibrium
            // for any tuning — see `walk_grounded`).
            let stroke = f64::from(input.sink) - f64::from(input.jump);
            self.vel.z += (self.def.gravity - self.def.buoyancy * frac) * dt;
            self.vel.z += stroke * self.def.swim_accel * dt;
            // Exponential drag — the water is thick.
            self.vel.z -= self.vel.z * (self.def.water_drag * dt).min(1.0);
            // The dry terminal cap still binds: a legal weak-drag,
            // zero-buoyancy tuning must not sink faster than the walk
            // path is allowed to fall.
            self.vel.z = self.vel.z.min(self.def.max_fall_speed);
        }

        // Swimming neither grants coyote time nor buffers jumps
        // (jump means "stroke up" here; a press near the shore must
        // not fire a stale jump on landing).
        self.since_grounded = f64::INFINITY;
        self.jump_buffer_left = 0.0;

        if self.slide_move(scene, dt, true) {
            return; // stuck escape — no contact state this frame
        }
        self.on_ground = self.ground_probe(scene);
    }

    /// The walk/swim horizontal steer: clamp the wish's x/y to unit
    /// length, approach `wish · speed` at `accel` (also the friction
    /// rate — zero wish brakes).
    fn steer_horizontal(&mut self, wish: DVec3, speed: f64, accel: f64, dt: f64) {
        let wish = wish.truncate();
        let wish = if wish.length_squared() > 1.0 {
            wish.normalize()
        } else {
            wish
        };
        let horizontal = move_toward(self.vel.truncate(), wish * speed, accel * dt);
        self.vel.x = horizontal.x;
        self.vel.y = horizontal.y;
    }

    /// `Fly` / `Noclip`: the full 3D wish steers at `fly_speed`, no
    /// gravity, no jumping; `collide` picks slide-vs-ghost.
    fn fly(&mut self, scene: &Scene, dt: f64, input: WalkInput, collide: bool) {
        // WT.1 — the fly modes ignore water entirely.
        self.swimming = false;
        self.breach_latched = false;
        let wish = if input.wish.length_squared() > 1.0 {
            input.wish.normalize()
        } else {
            input.wish
        };
        let target = wish * self.def.fly_speed;
        let max_delta = self.def.fly_accel * dt;
        let delta = target - self.vel;
        let len = delta.length();
        self.vel = if len <= max_delta || len < 1e-12 {
            target
        } else {
            self.vel + delta * (max_delta / len)
        };

        if collide {
            if !self.slide_move(scene, dt, false) {
                self.on_ground = self.ground_probe(scene);
            }
        } else {
            self.pos += self.vel * dt;
            self.on_ground = false;
        }
        self.since_grounded = f64::INFINITY;
        self.jump_buffer_left = 0.0;
    }

    /// The shared substepped per-axis mover; `step_up` enables the
    /// grounded auto-step retry on horizontal blocks. Returns `true`
    /// when the stuck-escape rule fired (the caller must not derive
    /// contact state from this frame).
    fn slide_move(&mut self, scene: &Scene, dt: f64, step_up: bool) -> bool {
        let mut disp = self.vel * dt;

        if self.blocked_at(scene, self.pos) {
            // Stuck escape: free move, no probes, flags cleared.
            self.pos += disp;
            self.on_ground = false;
            return true;
        }

        let max_step = self.def.radius.min(0.5);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let substeps = ((disp.abs().max_element() / max_step).ceil() as u32).clamp(1, MAX_SUBSTEPS);
        if disp.abs().max_element() > f64::from(MAX_SUBSTEPS) * max_step {
            // Anti-hang truncation — keeps probes-per-cell dense.
            disp *= f64::from(MAX_SUBSTEPS) * max_step / disp.abs().max_element();
        }
        let step = disp / f64::from(substeps);

        for _ in 0..substeps {
            for axis in 0..3 {
                if self.move_axis(scene, axis, step[axis]) {
                    if axis < 2
                        && step_up
                        && self.on_ground
                        && self.def.step_up > 0.0
                        && self.try_step_up(scene, axis, step[axis])
                    {
                        // Climbed the ledge — keep the velocity.
                        continue;
                    }
                    if axis == 2 && step.z < 0.0 {
                        self.hit_head = true;
                    }
                    self.vel[axis] = 0.0;
                }
            }
        }
        false
    }

    /// Auto-step (CC.2): lift by `step_up` (up = −z), redo the
    /// blocked horizontal move, then snap back down and require
    /// ground under the new spot. All-or-nothing: any stage failing
    /// reverts to the pre-step pose and the caller slides as usual.
    fn try_step_up(&mut self, scene: &Scene, axis: usize, delta: f64) -> bool {
        let saved = self.pos;

        let mut lifted = self.pos;
        lifted.z -= self.def.step_up;
        if self.blocked_at(scene, lifted) {
            return false; // no headroom for the lifted body
        }
        let mut over = lifted;
        over[axis] += delta;
        if self.blocked_at(scene, over) {
            return false; // ledge taller than step_up (or a wall)
        }
        self.pos = over;

        // Snap down: must land within step_up, else it wasn't a
        // ledge (walking off into air is the normal fall path, not a
        // step).
        if self.move_axis(scene, 2, self.def.step_up + SKIN) {
            true
        } else {
            self.pos = saved;
            false
        }
    }

    /// Thin skin box just below the feet. After a landing clamp the
    /// feet rest SKIN off the surface plane, so a 2·SKIN deep probe
    /// reaches into the floor cell.
    fn ground_probe(&self, scene: &Scene) -> bool {
        let (bmin, bmax) = self.box_at(self.pos);
        box_overlaps_solid(
            scene,
            DVec3::new(bmin.x, bmin.y, bmax.z),
            DVec3::new(bmax.x, bmax.y, bmax.z + 2.0 * SKIN),
            self.def.solidity,
        )
    }

    /// Collision box corners for feet position `pos`.
    fn box_at(&self, pos: DVec3) -> (DVec3, DVec3) {
        let r = self.def.radius;
        (
            DVec3::new(pos.x - r, pos.y - r, pos.z - self.def.height),
            DVec3::new(pos.x + r, pos.y + r, pos.z),
        )
    }

    fn blocked_at(&self, scene: &Scene, pos: DVec3) -> bool {
        let (bmin, bmax) = self.box_at(pos);
        box_overlaps_solid(scene, bmin, bmax, self.def.solidity)
    }

    /// Move the feet along `axis` by `delta`; on block, clamp flush
    /// (`SKIN` off the integer cell plane the leading face crossed)
    /// or, if even the clamped pose probes solid (rotated-grid
    /// geometry — its planes aren't world-axis planes), reject the
    /// axis move entirely (the demos' behaviour). Returns `true` if
    /// the axis was blocked.
    fn move_axis(&mut self, scene: &Scene, axis: usize, delta: f64) -> bool {
        if delta == 0.0 {
            return false;
        }
        let mut candidate = self.pos;
        candidate[axis] += delta;
        if !self.blocked_at(scene, candidate) {
            self.pos = candidate;
            return false;
        }

        // Leading-face offset from the feet position along this axis.
        let (min_off, max_off) = {
            let (bmin, bmax) = self.box_at(DVec3::ZERO);
            (bmin[axis], bmax[axis])
        };
        let clamped = if delta > 0.0 {
            // Leading face = max face; it entered cell
            // `floor(face)` — rest SKIN before that plane.
            (candidate[axis] + max_off).floor() - SKIN - max_off
        } else {
            // Leading face = min face; the entered cell's far
            // boundary is `floor(face) + 1`.
            (candidate[axis] + min_off).floor() + 1.0 + SKIN - min_off
        };
        let mut flush = self.pos;
        flush[axis] = clamped;
        // Never clamp *past* the attempted move, and only accept a
        // pose the probe agrees is clear.
        let overshoots = (clamped - self.pos[axis]).abs() > delta.abs() + SKIN;
        if !overshoots && !self.blocked_at(scene, flush) {
            self.pos = flush;
        }
        true
    }
}

/// Move `from` toward `to` by at most `max_delta` (a deterministic
/// approach — no overshoot, exact arrival).
fn move_toward(from: DVec2, to: DVec2, max_delta: f64) -> DVec2 {
    let delta = to - from;
    let len = delta.length();
    if len <= max_delta || len < 1e-12 {
        to
    } else {
        from + delta * (max_delta / len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GridTransform, VoxColor};
    use glam::IVec3;

    const DT: f64 = 1.0 / 60.0;

    /// Flat ground: solid slab z ∈ 100..=110 over x/y ∈ 60..=160,
    /// grid at the world origin — floor *surface* plane at z = 100.
    fn ground_scene() -> Scene {
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::identity());
        let grid = scene.grid_mut(id).expect("grid present");
        grid.set_rect(
            IVec3::new(60, 60, 100),
            IVec3::new(160, 160, 110),
            Some(VoxColor(0x80_50_90_50)),
        );
        scene
    }

    fn body_on_ground(scene: &Scene) -> CharacterBody {
        let mut body = CharacterBody::new(CharacterDef::default());
        body.teleport(DVec3::new(100.0, 100.0, 95.0));
        // Settle: fall to the floor.
        for _ in 0..120 {
            body.walk(scene, DT, WalkInput::default());
        }
        assert!(body.on_ground(), "settle: body must land");
        body
    }

    #[test]
    fn falls_and_lands_flush_on_the_surface_plane() {
        let scene = ground_scene();
        let body = body_on_ground(&scene);
        // Feet rest exactly SKIN above (−z of) the z=100 plane.
        assert!(
            (body.pos().z - (100.0 - SKIN)).abs() < 1e-9,
            "feet at {} != {}",
            body.pos().z,
            100.0 - SKIN
        );
        assert_eq!(body.vel().z, 0.0);
        assert!(!body.hit_head());
    }

    #[test]
    fn walk_reaches_and_holds_walk_speed() {
        let scene = ground_scene();
        let mut body = body_on_ground(&scene);
        let input = WalkInput {
            wish: DVec3::new(1.0, 0.0, 0.0),
            jump: false,
            sink: false,
        };
        for _ in 0..180 {
            body.walk(&scene, DT, input);
        }
        let speed = body.vel().truncate().length();
        assert!(
            (speed - body.def().walk_speed).abs() < 1e-9,
            "converged speed {speed}"
        );
        assert!(body.on_ground());
    }

    #[test]
    fn walk_into_wall_clamps_flush_and_slides() {
        let mut scene = ground_scene();
        {
            let id = scene.grids().next().expect("grid").0;
            let grid = scene.grid_mut(id).expect("grid");
            // Tall wall filling x ∈ 105..=106 in front of the body.
            grid.set_rect(
                IVec3::new(105, 60, 80),
                IVec3::new(106, 160, 110),
                Some(VoxColor(0x80_90_50_50)),
            );
        }
        let mut body = body_on_ground(&scene);
        let y0 = body.pos().y;
        let input = WalkInput {
            wish: DVec3::new(1.0, 1.0, 0.0),
            jump: false,
            sink: false,
        };
        for _ in 0..240 {
            body.walk(&scene, DT, input);
        }
        // x face flush against the wall plane at x = 105…
        let expected_x = 105.0 - body.def().radius - SKIN;
        assert!(
            (body.pos().x - expected_x).abs() < 1e-9,
            "x {} != flush {}",
            body.pos().x,
            expected_x
        );
        assert_eq!(body.vel().x, 0.0, "blocked axis velocity zeroed");
        // …while y keeps sliding.
        assert!(body.pos().y > y0 + 3.0, "slid along the wall");
    }

    #[test]
    fn jump_apex_matches_ballistics_and_relands() {
        let scene = ground_scene();
        let mut body = body_on_ground(&scene);
        let start_z = body.pos().z;
        let def = *body.def();

        body.walk(
            &scene,
            DT,
            WalkInput {
                wish: DVec3::ZERO,
                jump: true,
                sink: false,
            },
        );
        assert!(!body.on_ground(), "airborne after jump");

        let mut apex_rise = 0.0f64;
        let mut relanded = false;
        for _ in 0..240 {
            body.walk(&scene, DT, WalkInput::default());
            apex_rise = apex_rise.max(start_z - body.pos().z);
            if body.on_ground() {
                relanded = true;
                break;
            }
        }
        assert!(relanded, "must land again");
        // v²/2g, with discrete-integration slack.
        let ideal = def.jump_speed * def.jump_speed / (2.0 * def.gravity);
        assert!(
            (apex_rise - ideal).abs() < 0.2,
            "apex {apex_rise} vs ideal {ideal}"
        );
        assert!((body.pos().z - start_z).abs() < 1e-9, "back on the floor");
    }

    #[test]
    fn head_bump_stops_the_jump() {
        let mut scene = ground_scene();
        {
            let id = scene.grids().next().expect("grid").0;
            let grid = scene.grid_mut(id).expect("grid");
            // Ceiling slab: cells z ∈ 96..=97, so its underside
            // plane is z = 98 — ~2.1 voxels of clearance over the
            // 1.8 body.
            grid.set_rect(
                IVec3::new(60, 60, 96),
                IVec3::new(160, 160, 97),
                Some(VoxColor(0x80_50_50_90)),
            );
        }
        // Start INSIDE the gap (between the ceiling underside at
        // z = 98 and the floor plane at z = 100) — a spawn above the
        // ceiling slab would settle on top of it instead.
        let mut body = CharacterBody::new(CharacterDef::default());
        body.teleport(DVec3::new(100.0, 100.0, 99.9));
        for _ in 0..30 {
            body.walk(&scene, DT, WalkInput::default());
        }
        assert!(body.on_ground(), "settled in the gap");
        body.walk(
            &scene,
            DT,
            WalkInput {
                wish: DVec3::ZERO,
                jump: true,
                sink: false,
            },
        );
        let mut bumped = false;
        for _ in 0..120 {
            body.walk(&scene, DT, WalkInput::default());
            if body.hit_head() {
                bumped = true;
                // Head face flush under the ceiling plane, upward
                // velocity killed.
                let head = body.pos().z - body.def().height;
                assert!((head - (98.0 + SKIN)).abs() < 1e-9, "head at {head}");
                assert!(body.vel().z >= 0.0);
            }
            if body.on_ground() && bumped {
                break;
            }
        }
        assert!(bumped, "must bump the ceiling");
        assert!(body.on_ground(), "falls back to the floor");
    }

    #[test]
    fn fast_fall_does_not_tunnel_thin_floor() {
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::identity());
        let grid = scene.grid_mut(id).expect("grid present");
        // One voxel thick floor at z = 100.
        grid.set_rect(
            IVec3::new(60, 60, 100),
            IVec3::new(160, 160, 100),
            Some(VoxColor(0x80_70_70_70)),
        );
        let mut body = CharacterBody::new(CharacterDef {
            // The test drives 200 v/s deliberately — lift the
            // terminal-velocity clamp out of the way.
            max_fall_speed: 400.0,
            ..CharacterDef::default()
        });
        body.teleport(DVec3::new(100.0, 100.0, 60.0));
        body.set_vel(DVec3::new(0.0, 0.0, 200.0)); // 20 voxels per step
        for _ in 0..5 {
            body.walk(&scene, 0.1, WalkInput::default());
        }
        assert!(
            (body.pos().z - (100.0 - SKIN)).abs() < 1e-9,
            "feet at {} — tunneled?",
            body.pos().z
        );
        assert!(body.on_ground());
    }

    #[test]
    fn stuck_body_escapes_freely() {
        let scene = ground_scene();
        let mut body = CharacterBody::new(CharacterDef::default());
        // Feet buried mid-slab: the frame must move without probes.
        body.teleport(DVec3::new(100.0, 100.0, 105.0));
        assert!({
            let (bmin, bmax) = body.box_at(body.pos());
            box_overlaps_solid(&scene, bmin, bmax, Solidity::default())
        });
        let z0 = body.pos().z;
        body.walk(
            &scene,
            DT,
            WalkInput {
                wish: DVec3::new(1.0, 0.0, 0.0),
                jump: false,
                sink: false,
            },
        );
        assert!(body.pos().z > z0, "gravity still applies while stuck");
        assert!(body.pos().x > 100.0, "input still applies while stuck");
        assert!(!body.on_ground());
    }

    /// Floor plus a raised platform (top plane z = 99, one voxel
    /// above the z = 100 floor) covering x >= 105.
    fn ledge_scene(ledge_top_z: i32) -> Scene {
        let mut scene = ground_scene();
        let id = scene.grids().next().expect("grid").0;
        let grid = scene.grid_mut(id).expect("grid");
        grid.set_rect(
            IVec3::new(105, 60, ledge_top_z),
            IVec3::new(160, 160, 99),
            Some(VoxColor(0x80_80_80_40)),
        );
        scene
    }

    #[test]
    fn step_up_climbs_a_one_voxel_ledge() {
        let scene = ledge_scene(99); // 1 voxel proud of the floor
        let mut body = body_on_ground(&scene);
        let input = WalkInput {
            wish: DVec3::new(1.0, 0.0, 0.0),
            jump: false,
            sink: false,
        };
        for _ in 0..240 {
            body.walk(&scene, DT, input);
        }
        assert!(body.pos().x > 106.0, "walked onto the ledge");
        assert!(
            (body.pos().z - (99.0 - SKIN)).abs() < 1e-9,
            "feet on the ledge plane, got {}",
            body.pos().z
        );
        assert!(body.on_ground());
    }

    #[test]
    fn step_up_refuses_a_two_voxel_wall() {
        let scene = ledge_scene(98); // 2 voxels proud — too tall
        let mut body = body_on_ground(&scene);
        let input = WalkInput {
            wish: DVec3::new(1.0, 0.0, 0.0),
            jump: false,
            sink: false,
        };
        for _ in 0..240 {
            body.walk(&scene, DT, input);
        }
        let expected_x = 105.0 - body.def().radius - SKIN;
        assert!(
            (body.pos().x - expected_x).abs() < 1e-9,
            "clamped at {}, expected flush {expected_x}",
            body.pos().x
        );
        assert!((body.pos().z - (100.0 - SKIN)).abs() < 1e-9, "stayed down");
    }

    #[test]
    fn step_up_needs_headroom() {
        let mut scene = ledge_scene(99);
        {
            let id = scene.grids().next().expect("grid").0;
            let grid = scene.grid_mut(id).expect("grid");
            // Ceiling low enough that the LIFTED body cannot fit
            // over the ledge (lifted head reaches z ≈ 97.1; cell 97
            // spans [97, 98)).
            grid.set_rect(
                IVec3::new(104, 60, 97),
                IVec3::new(160, 160, 97),
                Some(VoxColor(0x80_40_40_80)),
            );
        }
        let mut body = body_on_ground(&scene);
        let input = WalkInput {
            wish: DVec3::new(1.0, 0.0, 0.0),
            jump: false,
            sink: false,
        };
        for _ in 0..240 {
            body.walk(&scene, DT, input);
        }
        let expected_x = 105.0 - body.def().radius - SKIN;
        assert!(
            (body.pos().x - expected_x).abs() < 1e-9,
            "no headroom ⇒ no step, got x {}",
            body.pos().x
        );
    }

    #[test]
    fn coyote_jump_after_walking_off_an_edge() {
        // Floor ends at x = 160; walk off it, then jump 3 frames
        // late — inside the 0.12 s coyote window.
        let scene = ground_scene();
        let mut body = body_on_ground(&scene);
        body.teleport(DVec3::new(159.0, 100.0, 95.0));
        for _ in 0..120 {
            body.walk(&scene, DT, WalkInput::default());
        }
        assert!(body.on_ground());
        let input = WalkInput {
            wish: DVec3::new(1.0, 0.0, 0.0),
            jump: false,
            sink: false,
        };
        while body.on_ground() {
            body.walk(&scene, DT, input);
        }
        body.walk(&scene, DT, input);
        body.walk(&scene, DT, input);
        body.walk(
            &scene,
            DT,
            WalkInput {
                wish: DVec3::new(1.0, 0.0, 0.0),
                jump: true,
                sink: false,
            },
        );
        assert!(
            body.vel().z < -0.5 * body.def().jump_speed,
            "coyote jump fired, vel.z = {}",
            body.vel().z
        );
    }

    #[test]
    fn coyote_does_not_double_jump() {
        let scene = ground_scene();
        let mut body = body_on_ground(&scene);
        body.walk(
            &scene,
            DT,
            WalkInput {
                wish: DVec3::ZERO,
                jump: true,
                sink: false,
            },
        );
        let rising = body.vel().z;
        assert!(rising < 0.0);
        // A second press right after must not re-fire off the coyote
        // window (nor may the buffer hold it until landing — wait
        // out the buffer first).
        for _ in 0..30 {
            body.walk(&scene, DT, WalkInput::default());
        }
        let before = body.vel().z;
        body.walk(
            &scene,
            DT,
            WalkInput {
                wish: DVec3::ZERO,
                jump: true,
                sink: false,
            },
        );
        assert!(
            body.vel().z > before,
            "still decelerating upward/falling — no mid-air re-jump"
        );
    }

    #[test]
    fn buffered_jump_fires_on_landing() {
        let scene = ground_scene();
        let mut body = CharacterBody::new(CharacterDef::default());
        body.teleport(DVec3::new(100.0, 100.0, 99.9));
        // Press jump while still falling, just before touchdown
        // (~0.1 voxels up ⇒ touchdown ≈ 0.08 s < the 0.12 s buffer;
        // from 0.5 voxels the fall takes ~0.2 s and the buffer
        // rightly expires).
        body.walk(
            &scene,
            DT,
            WalkInput {
                wish: DVec3::ZERO,
                jump: true,
                sink: false,
            },
        );
        assert!(!body.on_ground(), "still airborne at press");
        let mut jumped = false;
        for _ in 0..30 {
            body.walk(&scene, DT, WalkInput::default());
            if body.vel().z <= -0.9 * body.def().jump_speed {
                jumped = true;
                break;
            }
        }
        assert!(jumped, "buffered jump fired on landing");
    }

    #[test]
    fn fly_mode_hovers_and_slides() {
        let mut scene = ground_scene();
        {
            let id = scene.grids().next().expect("grid").0;
            let grid = scene.grid_mut(id).expect("grid");
            grid.set_rect(
                IVec3::new(105, 60, 80),
                IVec3::new(106, 160, 110),
                Some(VoxColor(0x80_90_50_50)),
            );
        }
        let mut body = CharacterBody::new(CharacterDef::default());
        body.set_mode(MoveMode::Fly);
        body.teleport(DVec3::new(100.0, 100.0, 95.0));
        // Hovers: no gravity.
        for _ in 0..60 {
            body.walk(&scene, DT, WalkInput::default());
        }
        assert_eq!(body.pos().z, 95.0, "no gravity in fly mode");
        // Slides against the wall like the demo fly cameras.
        let input = WalkInput {
            wish: DVec3::new(1.0, 0.0, -0.2),
            jump: false,
            sink: false,
        };
        for _ in 0..240 {
            body.walk(&scene, DT, input);
        }
        let expected_x = 105.0 - body.def().radius - SKIN;
        assert!(
            (body.pos().x - expected_x).abs() < 1e-9,
            "fly clamps at the wall, got {}",
            body.pos().x
        );
        assert!(body.pos().z < 95.0, "the -z wish component climbed");
    }

    #[test]
    fn fly_stops_instantly_when_wish_drops() {
        // The cave-demo regression: walk() with a zero wish must
        // stop the body (default fly_accel is instant); a host that
        // skips walk() on idle keeps stale velocity instead.
        let scene = ground_scene();
        let mut body = CharacterBody::new(CharacterDef::default());
        body.set_mode(MoveMode::Fly);
        body.teleport(DVec3::new(100.0, 100.0, 95.0));
        body.walk(
            &scene,
            DT,
            WalkInput {
                wish: DVec3::new(1.0, 0.0, 0.0),
                jump: false,
                sink: false,
            },
        );
        assert!(
            (body.vel().x - body.def().fly_speed).abs() < 1e-9,
            "instant spin-up to fly_speed"
        );
        let x_after_release = {
            body.walk(&scene, DT, WalkInput::default());
            body.pos().x
        };
        assert_eq!(body.vel(), DVec3::ZERO, "instant stop on zero wish");
        body.walk(&scene, DT, WalkInput::default());
        assert_eq!(body.pos().x, x_after_release, "no drift while idle");
    }

    #[test]
    fn noclip_passes_through_the_wall() {
        let mut scene = ground_scene();
        {
            let id = scene.grids().next().expect("grid").0;
            let grid = scene.grid_mut(id).expect("grid");
            grid.set_rect(
                IVec3::new(105, 60, 80),
                IVec3::new(106, 160, 110),
                Some(VoxColor(0x80_90_50_50)),
            );
        }
        let mut body = CharacterBody::new(CharacterDef::default());
        body.set_mode(MoveMode::Noclip);
        body.teleport(DVec3::new(100.0, 100.0, 95.0));
        let input = WalkInput {
            wish: DVec3::new(1.0, 0.0, 0.0),
            jump: false,
            sink: false,
        };
        for _ in 0..240 {
            body.walk(&scene, DT, input);
        }
        assert!(
            body.pos().x > 110.0,
            "ghosted through, x = {}",
            body.pos().x
        );
        assert!(!body.on_ground());
    }

    /// CC.5 perf probe (not a gate — run by hand):
    /// `cargo test -p roxlap-scene --release --lib -- --ignored
    /// character::tests::stress_probe --nocapture`. Measures walk()
    /// per frame for one wall-hugging walker (worst-ish case: probes
    /// + flush clamps every substep) and for a 100-NPC crowd.
    #[test]
    #[ignore = "perf probe, run by hand with --ignored --nocapture"]
    fn stress_probe() {
        let mut scene = ground_scene();
        {
            let id = scene.grids().next().expect("grid").0;
            let grid = scene.grid_mut(id).expect("grid");
            grid.set_rect(
                IVec3::new(105, 60, 80),
                IVec3::new(106, 160, 110),
                Some(VoxColor(0x80_90_50_50)),
            );
        }
        let input = WalkInput {
            wish: DVec3::new(1.0, 0.0, 0.0), // straight INTO the wall
            jump: false,
            sink: false,
        };

        let mut body = body_on_ground(&scene);
        let t0 = std::time::Instant::now();
        const FRAMES: u32 = 60_000;
        for _ in 0..FRAMES {
            body.walk(&scene, DT, input);
        }
        let per_frame = t0.elapsed() / FRAMES;
        println!("single wall-hugging walker: {per_frame:?}/frame");

        let mut crowd: Vec<CharacterBody> = (0..100)
            .map(|i| {
                let mut b = CharacterBody::new(CharacterDef::default());
                #[allow(clippy::cast_lossless)]
                b.teleport(DVec3::new(
                    70.0 + f64::from(i % 10) * 3.0,
                    70.0 + f64::from(i / 10) * 3.0,
                    95.0,
                ));
                b
            })
            .collect();
        for _ in 0..120 {
            for b in &mut crowd {
                b.walk(&scene, DT, WalkInput::default());
            }
        }
        let t0 = std::time::Instant::now();
        const CROWD_FRAMES: u32 = 600;
        for _ in 0..CROWD_FRAMES {
            for b in &mut crowd {
                b.walk(&scene, DT, input);
            }
        }
        let per_crowd_frame = t0.elapsed() / CROWD_FRAMES;
        println!("100-NPC crowd: {per_crowd_frame:?}/frame (all 100 walking)");
    }

    #[test]
    fn fall_speed_caps_at_terminal_velocity() {
        // No floor: a long fall must clamp at max_fall_speed (this
        // also bounds substep count — an unbounded fall gets
        // linearly more expensive per frame).
        let scene = Scene::new();
        let mut body = CharacterBody::new(CharacterDef::default());
        body.teleport(DVec3::new(0.0, 0.0, 0.0));
        for _ in 0..600 {
            body.walk(&scene, DT, WalkInput::default());
        }
        assert_eq!(body.vel().z, body.def().max_fall_speed);
    }

    #[test]
    fn deterministic_trajectory() {
        let scene = ground_scene();
        let run = || {
            let mut body = CharacterBody::new(CharacterDef::default());
            body.teleport(DVec3::new(100.0, 100.0, 95.0));
            let mut trace = Vec::new();
            for i in 0..180 {
                body.walk(
                    &scene,
                    DT,
                    WalkInput {
                        wish: DVec3::new(1.0, 0.3, 0.0),
                        jump: i == 90,
                        sink: false,
                    },
                );
                trace.push(body.pos());
            }
            trace
        };
        assert_eq!(run(), run(), "same input ⇒ bit-identical trajectory");
    }

    // ---- WT.1: swimming --------------------------------------------

    /// A deep pool: solid basin floor at z = 120, water surface plane
    /// at z = 100 (20 voxels of water above the floor). Physics water
    /// only — no visual voxels needed.
    fn pool_scene() -> Scene {
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::identity());
        let grid = scene.grid_mut(id).expect("grid present");
        grid.set_rect(
            IVec3::new(60, 60, 120),
            IVec3::new(160, 160, 130),
            Some(VoxColor(0x80_50_90_50)),
        );
        grid.add_water_volume(IVec3::new(60, 60, 100), IVec3::new(160, 160, 119));
        scene
    }

    /// The no-stroke bobbing equilibrium: gravity == buoyancy · frac.
    fn equilibrium_frac(def: &CharacterDef) -> f64 {
        def.gravity / def.buoyancy
    }

    #[test]
    fn floats_to_equilibrium_and_never_flickers() {
        let scene = pool_scene();
        let mut body = CharacterBody::new(CharacterDef::default());
        // Drop in deep: feet at z = 115 (fully submerged).
        body.teleport(DVec3::new(100.0, 100.0, 115.0));
        let mut engaged_at = None;
        for i in 0..600 {
            body.walk(&scene, DT, WalkInput::default());
            if body.is_swimming() && engaged_at.is_none() {
                engaged_at = Some(i);
            }
            // Hysteresis gate: once the swim state engages it must
            // hold through the whole rise + surface bob.
            if engaged_at.is_some() {
                assert!(body.is_swimming(), "swim state flickered at frame {i}");
            }
        }
        assert!(engaged_at.is_some(), "fully submerged body must swim");
        // Settled: negligible vertical motion at the predicted bob.
        assert!(body.vel().z.abs() < 0.05, "still bobbing: {}", body.vel().z);
        let frac = body.submerged_fraction(&scene);
        let eq = equilibrium_frac(body.def());
        assert!(
            (frac - eq).abs() < 0.03,
            "equilibrium fraction {frac} != {eq}"
        );
        // Head above the surface at the default tuning (z-down:
        // smaller z is higher).
        let head_z = body.pos().z - body.def().height;
        assert!(head_z < 100.0, "head under water at rest: {head_z}");
        assert!(!body.on_ground(), "floating, not standing");
    }

    #[test]
    fn sink_dives_and_jump_strokes_up() {
        let scene = pool_scene();
        let mut body = CharacterBody::new(CharacterDef::default());
        body.teleport(DVec3::new(100.0, 100.0, 102.0));
        // Settle at the surface first.
        for _ in 0..300 {
            body.walk(&scene, DT, WalkInput::default());
        }
        let surface_z = body.pos().z;

        // Hold sink: the body dives (+z grows).
        let dive = WalkInput {
            sink: true,
            ..WalkInput::default()
        };
        for _ in 0..120 {
            body.walk(&scene, DT, dive);
        }
        assert!(
            body.pos().z > surface_z + 1.0,
            "dive made no depth: {} vs {surface_z}",
            body.pos().z
        );
        assert!(body.is_swimming());

        // Hold jump: the body strokes back up (−z).
        let rise = WalkInput {
            jump: true,
            ..WalkInput::default()
        };
        let deep_z = body.pos().z;
        for _ in 0..60 {
            body.walk(&scene, DT, rise);
        }
        assert!(
            body.pos().z < deep_z - 1.0,
            "stroke up made no height: {} vs {deep_z}",
            body.pos().z
        );
    }

    #[test]
    fn breach_jump_fires_with_head_above_water() {
        let scene = pool_scene();
        let mut body = CharacterBody::new(CharacterDef::default());
        body.teleport(DVec3::new(100.0, 100.0, 102.0));
        for _ in 0..300 {
            body.walk(&scene, DT, WalkInput::default());
        }
        // At equilibrium the head is above the surface — a jump is a
        // BREACH (full jump impulse), not a stroke.
        assert!(body.is_swimming());
        let head_z = body.pos().z - body.def().height;
        assert!(head_z < 100.0, "test setup: head must be above water");
        body.walk(
            &scene,
            DT,
            WalkInput {
                jump: true,
                ..WalkInput::default()
            },
        );
        assert!(
            body.vel().z < -0.8 * body.def().jump_speed,
            "breach impulse missing: vel.z = {}",
            body.vel().z
        );
    }

    #[test]
    fn wading_below_enter_fraction_stays_walking() {
        // The minimal authored water — ONE voxel over the walkable
        // floor (surface z = 99 above the z = 100 plane) — must stay
        // wadable for the DEFAULT body: 1.0 deep on a 1.8 body is a
        // fraction of ≈ 0.56, under the 0.6 enter threshold (the
        // threshold default was picked for exactly this: a
        // decorative puddle must not swap walking for swimming).
        let mut scene = ground_scene();
        let id = scene.grids().next().expect("grid").0;
        scene
            .grid_mut(id)
            .expect("grid")
            .add_water_volume(IVec3::new(60, 60, 99), IVec3::new(160, 160, 99));
        let mut body = body_on_ground(&scene);
        let frac = body.submerged_fraction(&scene);
        assert!(
            frac > 0.5 && frac < body.def().swim_enter_frac,
            "test setup: wading fraction {frac}"
        );
        for _ in 0..120 {
            body.walk(&scene, DT, WalkInput::default());
        }
        assert!(!body.is_swimming(), "wading must not engage the swim state");
        assert!(body.on_ground(), "wading body stands on the floor");
    }

    #[test]
    fn breach_is_one_shot_until_jump_releases() {
        let scene = pool_scene();
        let mut body = CharacterBody::new(CharacterDef::default());
        body.teleport(DVec3::new(100.0, 100.0, 102.0));
        for _ in 0..300 {
            body.walk(&scene, DT, WalkInput::default());
        }
        let held = WalkInput {
            jump: true,
            ..WalkInput::default()
        };
        // First held frame: the breach impulse, exactly.
        body.walk(&scene, DT, held);
        assert_eq!(body.vel().z, -body.def().jump_speed, "breach fired");
        // Still held: NO re-impulse — the very next frame the
        // velocity must have integrated away from the exact impulse
        // value (the old bug pinned it to -jump_speed every frame).
        body.walk(&scene, DT, held);
        assert!(
            (body.vel().z - -body.def().jump_speed).abs() > 1e-9,
            "held jump re-impulsed: vel.z = {}",
            body.vel().z
        );
        // Release: fall back into the pool and settle at the bob
        // again (the breach launched the body clear of the water) —
        // then a fresh press re-arms the latch and breaches again.
        for _ in 0..300 {
            body.walk(&scene, DT, WalkInput::default());
        }
        assert!(body.is_swimming(), "back at the surface bob");
        body.walk(&scene, DT, held);
        assert_eq!(body.vel().z, -body.def().jump_speed, "re-armed breach");
    }

    #[test]
    fn underwater_sink_respects_the_terminal_fall_cap() {
        // A legal tuning with no buoyancy and near-zero drag must not
        // sink faster than the dry terminal cap (gravity/drag alone
        // would reach 240 u/s here — 4× the cap — and could launch
        // the body out of the volume's bottom face).
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::identity());
        let grid = scene.grid_mut(id).expect("grid present");
        // A very deep column of water, no floor for a while.
        grid.add_water_volume(IVec3::new(60, 60, 100), IVec3::new(160, 160, 900));
        let def = CharacterDef {
            buoyancy: 0.0,
            water_drag: 0.1,
            ..CharacterDef::default()
        };
        let mut body = CharacterBody::new(def);
        body.teleport(DVec3::new(100.0, 100.0, 110.0));
        for _ in 0..600 {
            body.walk(&scene, DT, WalkInput::default());
            assert!(
                body.vel().z <= body.def().max_fall_speed + 1e-9,
                "sink speed {} exceeded the terminal cap",
                body.vel().z
            );
        }
        assert!(body.is_swimming(), "zero buoyancy: still under water");
    }

    #[test]
    fn cork_tuning_does_not_limit_cycle_at_the_waterline() {
        // buoyancy 80 / gravity 24 → equilibrium fraction 0.3, BELOW
        // swim_exit_frac (0.35). Without the runtime clamp on the
        // release threshold this strobes swim → walk forever.
        let scene = pool_scene();
        let mut body = CharacterBody::new(CharacterDef {
            buoyancy: 80.0,
            ..CharacterDef::default()
        });
        body.teleport(DVec3::new(100.0, 100.0, 115.0));
        // The strong buoyancy legitimately POPS the cork clear of the
        // water first (airborne = not swimming — physical, not a
        // flicker); let the splash oscillation decay…
        for _ in 0..600 {
            body.walk(&scene, DT, WalkInput::default());
        }
        // …then the settled bob must hold ONE state steadily
        // (whichever the last threshold crossing left — the clamped
        // band contains the equilibrium, so either flag is honest).
        // Without continuous passive water forces this strobes every
        // few frames: the walk path's dry gravity vs the swim path's
        // buoyancy made a relay oscillator across the band.
        let settled = body.is_swimming();
        for i in 0..300 {
            body.walk(&scene, DT, WalkInput::default());
            assert_eq!(
                body.is_swimming(),
                settled,
                "settled cork flickered at frame {i}"
            );
        }
        // Floating high: equilibrium ≈ 0.3 of the body under water.
        let frac = body.submerged_fraction(&scene);
        assert!((frac - 0.3).abs() < 0.05, "cork equilibrium {frac}");
    }

    #[test]
    fn fraction_survives_a_volume_that_stops_short_of_the_floor() {
        // An authored volume whose bottom face ends ABOVE the basin
        // floor: a diving body's feet exit through the bottom while
        // the head is still deep in water. The fraction must read
        // fully-submerged there, not snap to dry (the discontinuity
        // would strobe the swim state).
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::identity());
        let grid = scene.grid_mut(id).expect("grid present");
        grid.set_rect(
            IVec3::new(60, 60, 120),
            IVec3::new(160, 160, 130),
            Some(VoxColor(0x80_50_90_50)),
        );
        // Water stops at z = 110 — ten voxels short of the floor.
        grid.add_water_volume(IVec3::new(60, 60, 100), IVec3::new(160, 160, 109));
        let mut body = CharacterBody::new(CharacterDef::default());
        // Feet below the volume's bottom face, head well inside it.
        body.teleport(DVec3::new(100.0, 100.0, 111.0));
        assert_eq!(
            body.submerged_fraction(&scene),
            1.0,
            "feet under the volume's bottom + head in water = fully submerged"
        );
        // Diving straight through: the guard covers the straddle band
        // (feet out, head in), so the ONLY release happens when the
        // whole body has passed below the volume — one clean
        // swim→walk transition, no strobing across the bottom face.
        let mut transitions = 0;
        let mut was_swimming = false;
        for _ in 0..300 {
            body.walk(
                &scene,
                DT,
                WalkInput {
                    sink: true,
                    ..WalkInput::default()
                },
            );
            if was_swimming && !body.is_swimming() {
                transitions += 1;
            }
            was_swimming = body.is_swimming();
        }
        assert!(transitions <= 1, "strobed across the face: {transitions}");
        assert!(!body.is_swimming(), "fully below the volume = not in water");
        assert!(body.on_ground(), "landed on the basin floor");
    }

    #[test]
    fn eye_in_water_flips_at_the_camera_not_the_feet() {
        let scene = pool_scene();
        let mut body = CharacterBody::new(CharacterDef::default());
        body.teleport(DVec3::new(100.0, 100.0, 102.0));
        for _ in 0..300 {
            body.walk(&scene, DT, WalkInput::default());
        }
        // Bobbing at the surface: swimming, but the camera is dry —
        // the tint/lowpass hook must be off.
        assert!(body.is_swimming());
        assert!(!body.eye_in_water(&scene), "surface bob: dry camera");
        // Dive: the eye goes under.
        for _ in 0..120 {
            body.walk(
                &scene,
                DT,
                WalkInput {
                    sink: true,
                    ..WalkInput::default()
                },
            );
        }
        assert!(body.eye_in_water(&scene), "diving: submerged camera");
    }

    #[test]
    fn dry_walk_is_unaffected_by_far_away_water() {
        // Byte-identity gate: the same input sequence over the same
        // floor must produce the SAME trajectory whether or not the
        // scene has water somewhere else.
        let run = |with_water: bool| {
            let mut scene = ground_scene();
            if with_water {
                let id = scene.grids().next().expect("grid").0;
                scene
                    .grid_mut(id)
                    .expect("grid")
                    .add_water_volume(IVec3::new(150, 150, 90), IVec3::new(160, 160, 99));
            }
            let mut body = CharacterBody::new(CharacterDef::default());
            body.teleport(DVec3::new(100.0, 100.0, 95.0));
            let mut trace = Vec::new();
            for i in 0..240 {
                body.walk(
                    &scene,
                    DT,
                    WalkInput {
                        wish: DVec3::new(-1.0, 0.4, 0.0),
                        jump: i == 60,
                        sink: false,
                    },
                );
                trace.push(body.pos());
            }
            trace
        };
        assert_eq!(run(false), run(true), "dry path must be unperturbed");
    }

    #[test]
    fn swims_identically_on_a_scaled_grid() {
        // SC discipline: the same pool authored on a vws = 0.5 grid
        // (double the voxel counts, half the size each) floats the
        // body to the same WORLD equilibrium.
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::at_scale(DVec3::ZERO, 0.5));
        let grid = scene.grid_mut(id).expect("grid present");
        grid.set_rect(
            IVec3::new(120, 120, 240),
            IVec3::new(320, 320, 260),
            Some(VoxColor(0x80_50_90_50)),
        );
        grid.add_water_volume(IVec3::new(120, 120, 200), IVec3::new(320, 320, 239));
        let mut body = CharacterBody::new(CharacterDef::default());
        body.teleport(DVec3::new(100.0, 100.0, 115.0));
        for _ in 0..600 {
            body.walk(&scene, DT, WalkInput::default());
        }
        assert!(body.is_swimming());
        let frac = body.submerged_fraction(&scene);
        let eq = equilibrium_frac(body.def());
        assert!(
            (frac - eq).abs() < 0.03,
            "scaled-grid equilibrium {frac} != {eq}"
        );
        // World surface plane: local z 200 × 0.5 = world z 100 — the
        // same waterline as the identity pool.
        let expect_feet = 100.0 + eq * body.def().height;
        assert!(
            (body.pos().z - expect_feet).abs() < 0.1,
            "feet at {} != {expect_feet}",
            body.pos().z
        );
    }

    #[test]
    fn fly_mode_ignores_water_and_clears_swim_state() {
        let scene = pool_scene();
        let mut body = CharacterBody::new(CharacterDef::default());
        body.teleport(DVec3::new(100.0, 100.0, 110.0));
        for _ in 0..120 {
            body.walk(&scene, DT, WalkInput::default());
        }
        assert!(body.is_swimming(), "walk mode swims in the pool");
        body.set_mode(MoveMode::Fly);
        body.walk(&scene, DT, WalkInput::default());
        assert!(!body.is_swimming(), "fly cleared the swim state");
        // Hovering: no gravity, no buoyancy — velocity stays zero.
        for _ in 0..60 {
            body.walk(&scene, DT, WalkInput::default());
        }
        assert_eq!(body.vel(), DVec3::ZERO, "fly ignores water forces");
    }
}
