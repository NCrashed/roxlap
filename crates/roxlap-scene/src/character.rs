//! Character controller (stage CC.1 — see
//! `docs/porting/PORTING-CONTROLLER.md`).
//!
//! A walking body over a [`Scene`]: substepped per-axis
//! move-and-slide against the [`crate::collide`] probe, gravity,
//! jumping, ground / head-bump detection. Not to be confused with
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
    /// What counts as solid (bedrock-placeholder policy — must match
    /// how the host *renders* the world; see
    /// [`Solidity::bedrock_blocks`]).
    pub solidity: Solidity,
}

impl Default for CharacterDef {
    fn default() -> Self {
        Self {
            radius: 0.4,
            height: 1.8,
            eye_height: 1.62,
            gravity: 24.0,
            jump_speed: 9.0,
            walk_speed: 6.0,
            accel_ground: 40.0,
            accel_air: 8.0,
            step_up: 1.05,
            coyote_time: 0.12,
            jump_buffer: 0.12,
            fly_speed: 12.0,
            solidity: Solidity::default(),
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
    pub jump: bool,
}

/// A walking body: feet-positioned collision box + velocity +
/// contact flags. Construct with [`CharacterBody::new`], place with
/// [`teleport`](Self::teleport), then call
/// [`walk`](Self::walk) once per frame.
#[derive(Debug, Clone, Copy)]
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
    }

    /// The construction parameters (read-only; a body's shape does
    /// not change after construction).
    #[must_use]
    pub fn def(&self) -> &CharacterDef {
        &self.def
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
        // -- integrate velocity ------------------------------------
        let wish = input.wish.truncate();
        let wish = if wish.length_squared() > 1.0 {
            wish.normalize()
        } else {
            wish
        };
        let target = wish * self.def.walk_speed;
        let accel = if self.on_ground {
            self.def.accel_ground
        } else {
            self.def.accel_air
        };
        let horizontal = move_toward(self.vel.truncate(), target, accel * dt);
        self.vel.x = horizontal.x;
        self.vel.y = horizontal.y;

        self.vel.z += self.def.gravity * dt;

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

    /// `Fly` / `Noclip`: the full 3D wish steers at `fly_speed`, no
    /// gravity, no jumping; `collide` picks slide-vs-ghost.
    fn fly(&mut self, scene: &Scene, dt: f64, input: WalkInput, collide: bool) {
        let wish = if input.wish.length_squared() > 1.0 {
            input.wish.normalize()
        } else {
            input.wish
        };
        let target = wish * self.def.fly_speed;
        let max_delta = self.def.accel_ground * dt;
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
        let mut body = CharacterBody::new(CharacterDef::default());
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
                    },
                );
                trace.push(body.pos());
            }
            trace
        };
        assert_eq!(run(), run(), "same input ⇒ bit-identical trajectory");
    }
}
