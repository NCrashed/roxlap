//! DT.2 — [`DebrisSystem`]: falling voxel islands (see
//! `docs/porting/PORTING-DESTRUCTION.md`).
//!
//! The crumble half of the destruction stage: a detected
//! [`Island`] is extracted from its grid, becomes a KV6 sprite, falls
//! under gravity with a slow cosmetic spin, and reports an impact the
//! moment its footprint touches solid ground — where DT.3's shatter
//! takes over. Host-owned and renderer-agnostic like
//! [`ParticleSystem`](crate::ParticleSystem), and split the same way:
//! [`DebrisSystem::spawn_island`] + [`DebrisSystem::update`] are pure
//! simulation over a [`Scene`] (unit-testable, no facade calls);
//! [`DebrisSystem::sync`] mirrors bodies into dynamic sprite
//! instances (spawn pre-posed, batch-move, despawn);
//! [`DebrisSystem::tick`] is the per-frame one-shot.
//!
//! Physics is deliberately minimal (locked in the entry doc): bodies
//! fall **world-vertically** (+z, the voxlap z-down convention) with
//! a terminal-speed clamp; the yaw spin is cosmetic only — collision
//! always tests the **unrotated** world AABB via
//! [`box_overlaps_solid`], binary-searching the contact so a landed
//! body rests flush on the voxel plane it hit. A grid's rotation is
//! likewise not applied to the falling sprite's pose (v1 targets
//! axis-aligned grids; the cave demo's is identity).

use glam::{DVec3, IVec3};
use roxlap_scene::islands::Island;
use roxlap_scene::{box_overlaps_solid, BakeMode, GridId, Scene, Solidity};

use crate::{DynSpriteTransform, SceneRenderer, SpriteInstanceId, SpriteModelId};

/// One landed island, drained via [`DebrisSystem::drain_impacts`]:
/// everything DT.3's shatter needs.
#[derive(Debug)]
pub struct DebrisImpact {
    /// The island's voxels + colours (grid-local, as detected) — the
    /// shatter burst samples these, not the sprite model.
    pub island: Island,
    /// The grid the island was extracted from.
    pub grid: GridId,
    /// World position of the island's pivot (bbox centre) at rest —
    /// flush on the surface it hit.
    pub pos: DVec3,
    /// Impact speed in world units/second (≤ the terminal clamp).
    pub speed: f64,
}

/// One falling island: simulation state + the (lazily created) sprite
/// handles `sync` manages.
struct DebrisBody {
    island: Island,
    grid: GridId,
    model: Option<SpriteModelId>,
    inst: Option<SpriteInstanceId>,
    /// World position of the pivot (bbox centre).
    pos: DVec3,
    /// Vertical speed, world units/s, +z (down).
    vel: f64,
    /// Cosmetic spin about the world vertical.
    yaw: f32,
    yaw_rate: f32,
    /// Half-extents of the unrotated world AABB.
    half: DVec3,
    /// The source grid's `voxel_world_size` — the uniform basis scale
    /// the sprite renders at.
    scale: f32,
}

/// Host-owned falling-island simulation + facade binding. Construct
/// once, [`Self::spawn_island`] per detected island,
/// [`Self::tick`] per frame, [`Self::drain_impacts`] to shatter.
pub struct DebrisSystem {
    /// Gravity in world units/s², +z is down. Matches the particle
    /// default so debris and dust fall together.
    pub gravity: f64,
    /// Terminal-speed clamp on the fall, world units/s.
    pub terminal_speed: f64,
    /// Cosmetic spin cap, rad/s: each island gets a deterministic
    /// per-island rate in `[-max, max]` (hashed from its position —
    /// no RNG state, replays are bit-identical).
    pub max_yaw_rate: f32,
    /// What counts as solid ground (bedrock-placeholder policy) —
    /// the same knob [`box_overlaps_solid`] takes.
    pub solidity: Solidity,
    bodies: Vec<DebrisBody>,
    impacts: Vec<DebrisImpact>,
    /// Handles of retired bodies awaiting facade removal in `sync`.
    dead: Vec<(Option<SpriteModelId>, Option<SpriteInstanceId>)>,
    /// Reused per-sync transform batch (no per-frame allocation).
    move_scratch: Vec<(SpriteInstanceId, DynSpriteTransform)>,
}

impl Default for DebrisSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl DebrisSystem {
    /// A system with the stock tuning: particle-matched gravity, a
    /// generous terminal clamp, a barely-perceptible spin.
    #[must_use]
    pub fn new() -> Self {
        Self {
            gravity: 22.0,
            terminal_speed: 60.0,
            max_yaw_rate: 0.6,
            solidity: Solidity::default(),
            bodies: Vec::new(),
            impacts: Vec::new(),
            dead: Vec::new(),
            move_scratch: Vec::new(),
        }
    }

    /// Extract `island` from `grid` (via [`Island::extract`] — carve +
    /// `bake_bbox`; **mips are the caller's obligation**, same as any
    /// edit) and register it as a falling body at the exact world pose
    /// of the voxels it replaced. The sprite model/instance appear on
    /// the next [`Self::sync`]. Returns `false` (and does nothing) for
    /// a missing grid or an empty island.
    ///
    /// **Spawn inside geometry** (an L-shaped island whose bbox wraps
    /// a surviving support, a detach flush under a shelf): the body's
    /// AABB already overlaps foreign solid, so it cannot fall — it is
    /// queued as an **immediate [`DebrisImpact`] at zero speed** (the
    /// shatter replaces it in place) and never becomes a body. An
    /// explicit policy, not an accident of the contact search: the
    /// accepted degradation of AABB-only collision (decision 4).
    pub fn spawn_island(
        &mut self,
        scene: &mut Scene,
        grid: GridId,
        island: Island,
        bake: BakeMode,
    ) -> bool {
        if island.voxels.is_empty() {
            return false;
        }
        let Some(g) = scene.grid_mut(grid) else {
            return false;
        };
        let transform = g.transform;
        island.extract(g, bake);
        let pos = island.world_pivot(&transform);
        let dims = (island.bbox.1 - island.bbox.0 + IVec3::ONE).as_dvec3();
        let vws = transform.voxel_world_size;
        #[allow(clippy::cast_possible_truncation)]
        let body = DebrisBody {
            yaw_rate: spin_from_pos(island.bbox.0, self.max_yaw_rate),
            island,
            grid,
            model: None,
            inst: None,
            pos,
            vel: 0.0,
            yaw: 0.0,
            half: dims * (vws * 0.5),
            scale: vws as f32,
        };
        if aabb_hits_ground(scene, &body, body.pos.z, self.solidity) {
            self.impacts.push(DebrisImpact {
                island: body.island,
                grid: body.grid,
                pos: body.pos,
                speed: 0.0,
            });
            return true;
        }
        self.bodies.push(body);
        true
    }

    /// Advance every body by `dt` seconds: semi-implicit Euler with
    /// the terminal clamp, cosmetic yaw, then the vertical collision
    /// march against `scene`. The frame's displacement is walked in
    /// **substeps of half the collision window** (`half.z + vws/2` —
    /// the thinnest thing a voxel scene can contain is one voxel), so
    /// a fast body cannot tunnel through a one-voxel shelf on a slow
    /// frame (`vel·dt` is unbounded: hosts pass real dt, hitches
    /// included). A blocked substep is binary-searched to flush
    /// contact; the body retires as a [`DebrisImpact`]. Pure
    /// simulation — no facade calls (see [`Self::sync`]).
    #[allow(clippy::cast_possible_truncation)]
    pub fn update(&mut self, scene: &Scene, dt: f64) {
        let dtf = dt as f32;
        let mut i = 0;
        while i < self.bodies.len() {
            let b = &mut self.bodies[i];
            b.vel = (b.vel + self.gravity * dt).min(self.terminal_speed);
            b.yaw += b.yaw_rate * dtf;
            // Substep stride: half the overlap window of the box vs a
            // one-voxel plate (window = 2·half.z + vws), so consecutive
            // endpoint tests cannot straddle the thinnest obstacle.
            let stride = b.half.z + 0.5 * f64::from(b.scale);
            let mut remaining = b.vel * dt;
            let mut blocked_at = None;
            while remaining > 0.0 {
                let step = remaining.min(stride);
                let target = b.pos.z + step;
                if aabb_hits_ground(scene, b, target, self.solidity) {
                    blocked_at = Some(target);
                    break;
                }
                b.pos.z = target;
                remaining -= step;
            }
            let Some(target) = blocked_at else {
                i += 1;
                continue;
            };
            // Contact: binary-search z between the last free position
            // and the blocked target — lands flush on the voxel plane
            // (within the anti-flush epsilon the AABB is shrunk by).
            let (mut lo, mut hi) = (b.pos.z, target);
            for _ in 0..16 {
                let mid = 0.5 * (lo + hi);
                if aabb_hits_ground(scene, b, mid, self.solidity) {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            b.pos.z = lo;
            let body = self.bodies.swap_remove(i);
            self.dead.push((body.model, body.inst));
            self.impacts.push(DebrisImpact {
                island: body.island,
                grid: body.grid,
                pos: body.pos,
                speed: body.vel,
            });
        }
    }

    /// Mirror the simulation into the facade: retire handles of landed
    /// bodies, spawn a model + pre-posed instance for each newborn
    /// (no axis-aligned first-frame flash), and batch-move everything
    /// alive via one
    /// [`set_sprite_instance_transforms`](SceneRenderer::set_sprite_instance_transforms).
    /// The facade tombstones removed models — a host shattering many
    /// islands should call
    /// [`compact_sprite_models`](SceneRenderer::compact_sprite_models)
    /// periodically.
    pub fn sync(&mut self, renderer: &mut SceneRenderer) {
        self.sync_with(renderer);
    }

    fn sync_with<F: DebrisFacade>(&mut self, facade: &mut F) {
        for (model, inst) in self.dead.drain(..) {
            if let Some(inst) = inst {
                facade.despawn(inst);
            }
            if let Some(model) = model {
                facade.remove_model(model);
            }
        }
        for b in &mut self.bodies {
            let xf = debris_xf(b.pos, b.yaw, b.scale);
            if let Some(inst) = b.inst {
                self.move_scratch.push((inst, xf));
            } else {
                let model = facade.add_model(&b.island.to_kv6());
                if let Some(inst) = facade.spawn(model, xf) {
                    b.model = Some(model);
                    b.inst = Some(inst);
                } else {
                    // Degenerate pose (should not happen for a yaw
                    // basis) — drop the model, retry next sync.
                    facade.remove_model(model);
                }
            }
        }
        if !self.move_scratch.is_empty() {
            facade.set_transforms(&self.move_scratch);
        }
        self.move_scratch.clear();
    }

    /// One-shot per-frame step: [`Self::update`] then [`Self::sync`].
    pub fn tick(&mut self, renderer: &mut SceneRenderer, scene: &Scene, dt: f64) {
        self.update(scene, dt);
        self.sync_with(renderer);
    }

    /// Drain the impacts accumulated by [`Self::update`] — DT.3's
    /// shatter consumes these (each carries its island's voxels).
    pub fn drain_impacts(&mut self) -> std::vec::Drain<'_, DebrisImpact> {
        self.impacts.drain(..)
    }

    /// Number of bodies currently falling.
    #[must_use]
    pub fn debris_count(&self) -> usize {
        self.bodies.len()
    }
}

/// The slice of [`SceneRenderer`] that [`DebrisSystem::sync`] drives —
/// the same internal seam [`ParticleFacade`](crate::particles) uses,
/// grown by the model half (each island is its own KV6 model), so the
/// binding logic is unit-testable with a mock (a real backend needs a
/// window). The facade impl is pure forwarding.
pub(crate) trait DebrisFacade {
    fn add_model(&mut self, kv6: &roxlap_formats::kv6::Kv6) -> SpriteModelId;
    fn remove_model(&mut self, id: SpriteModelId);
    fn spawn(&mut self, model: SpriteModelId, xf: DynSpriteTransform) -> Option<SpriteInstanceId>;
    fn despawn(&mut self, id: SpriteInstanceId);
    fn set_transforms(&mut self, batch: &[(SpriteInstanceId, DynSpriteTransform)]);
}

impl DebrisFacade for SceneRenderer {
    fn add_model(&mut self, kv6: &roxlap_formats::kv6::Kv6) -> SpriteModelId {
        self.add_sprite_model(kv6)
    }
    fn remove_model(&mut self, id: SpriteModelId) {
        self.remove_sprite_model(id);
    }
    fn spawn(&mut self, model: SpriteModelId, xf: DynSpriteTransform) -> Option<SpriteInstanceId> {
        self.add_sprite_instance_posed(model, xf)
    }
    fn despawn(&mut self, id: SpriteInstanceId) {
        self.remove_sprite_instance(id);
    }
    fn set_transforms(&mut self, batch: &[(SpriteInstanceId, DynSpriteTransform)]) {
        self.set_sprite_instance_transforms(batch);
    }
}

/// Would the body's unrotated AABB, centred at `(b.pos.x, b.pos.y,
/// z)`, overlap solid ground? The box is shrunk by a small epsilon on
/// every side so flush contact (an island resting exactly on a voxel
/// plane, or cut flush against a wall) does not read as a collision —
/// only real penetration does.
fn aabb_hits_ground(scene: &Scene, b: &DebrisBody, z: f64, solidity: Solidity) -> bool {
    let eps = DVec3::splat(1.0e-3 * f64::from(b.scale));
    let c = DVec3::new(b.pos.x, b.pos.y, z);
    box_overlaps_solid(scene, c - b.half + eps, c + b.half - eps, solidity)
}

/// The pose a debris body renders at: pivot position + a basis rotated
/// by `yaw` about the world vertical and uniformly scaled by the
/// source grid's `voxel_world_size` (the same scaled-basis convention
/// as the particle pass — right × up chirality preserved).
#[allow(clippy::cast_possible_truncation)]
fn debris_xf(pos: DVec3, yaw: f32, k: f32) -> DynSpriteTransform {
    let (c, s) = (yaw.cos() * k, yaw.sin() * k);
    DynSpriteTransform {
        pos: [pos.x as f32, pos.y as f32, pos.z as f32],
        right: [c, s, 0.0],
        up: [-s, c, 0.0],
        forward: [0.0, 0.0, k],
    }
}

/// Deterministic per-island spin rate in `[-max, max]`, hashed from
/// the island's bbox-min voxel — stateless, so identical scenes
/// crumble identically.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn spin_from_pos(p: IVec3, max: f32) -> f32 {
    let h = (p.x.wrapping_mul(73_856_093)
        ^ p.y.wrapping_mul(19_349_663)
        ^ p.z.wrapping_mul(83_492_791)) as u32;
    (f64::from(h) / f64::from(u32::MAX) * 2.0 - 1.0) as f32 * max
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SlotHandle;
    use roxlap_scene::islands::detect_islands;
    use roxlap_scene::{GridTransform, VoxColor};

    const STONE: VoxColor = VoxColor(0x80B0_8040);

    /// A scene with one identity grid: a supported pillar at (2,2)
    /// with a beam sticking out at z=100, already cut — plus the
    /// detected 3-voxel island (x ∈ [4,6], y=2, z=100), ready to spawn.
    fn scene_with_island() -> (Scene, GridId, Island) {
        let mut scene = Scene::new();
        let grid = scene.add_grid(GridTransform::identity());
        let g = scene.grid_mut(grid).expect("grid");
        g.set_rect(IVec3::new(2, 2, 100), IVec3::new(2, 2, 255), Some(STONE));
        g.set_rect(IVec3::new(3, 2, 100), IVec3::new(6, 2, 100), Some(STONE));
        g.set_voxel(IVec3::new(3, 2, 100), None);
        let islands = detect_islands(g, IVec3::new(3, 2, 100), IVec3::new(3, 2, 100), 4096);
        assert_eq!(islands.len(), 1);
        (scene, grid, islands[0].clone())
    }

    /// In empty air the body falls monotonically (z-down ⇒ z grows)
    /// and its speed saturates at the terminal clamp.
    #[test]
    fn falls_monotonically_and_clamps() {
        let (mut scene, grid, island) = scene_with_island();
        let mut sys = DebrisSystem::new();
        assert!(sys.spawn_island(&mut scene, grid, island, BakeMode::Directional));
        assert_eq!(sys.debris_count(), 1);

        let mut last_z = sys.bodies[0].pos.z;
        // 5 s: saturation takes terminal/gravity ≈ 2.7 s; nothing to
        // land on (the island's columns are placeholder bedrock only,
        // which the default Solidity does not count as solid).
        for _ in 0..300 {
            sys.update(&scene, 1.0 / 60.0);
            if sys.bodies.is_empty() {
                break;
            }
            let z = sys.bodies[0].pos.z;
            assert!(z > last_z, "falling means z strictly grows (z-down)");
            last_z = z;
        }
        assert!(
            !sys.bodies.is_empty(),
            "nothing to land on above bedrock yet"
        );
        assert!(
            sys.bodies[0].vel <= sys.terminal_speed + 1e-9,
            "speed clamps at terminal"
        );
        // Long enough to have saturated.
        assert!((sys.bodies[0].vel - sys.terminal_speed).abs() < 1e-9);
    }

    /// With a floor below, the body lands flush on the voxel plane
    /// (no penetration), exactly one impact fires, and the body +
    /// its voxels come back in the event.
    #[test]
    fn lands_flush_and_fires_once() {
        let (mut scene, grid, island) = scene_with_island();
        // A solid floor plate at z=140 under the island's footprint.
        scene.grid_mut(grid).expect("grid").set_rect(
            IVec3::new(0, 0, 140),
            IVec3::new(12, 6, 140),
            Some(STONE),
        );

        let mut sys = DebrisSystem::new();
        let half_z = 0.5; // 1-voxel-tall island
        assert!(sys.spawn_island(&mut scene, grid, island, BakeMode::Directional));

        for _ in 0..600 {
            sys.update(&scene, 1.0 / 60.0);
        }
        assert_eq!(sys.debris_count(), 0, "the body landed and retired");
        let impacts: Vec<DebrisImpact> = sys.drain_impacts().collect();
        assert_eq!(impacts.len(), 1, "exactly one impact");
        let hit = &impacts[0];
        assert_eq!(hit.island.voxels.len(), 3, "voxels ride along for DT.3");
        assert!(hit.speed > 0.0);
        // Flush: the AABB bottom rests on the floor's top plane z=140
        // (within the anti-flush epsilon + binary-search tolerance).
        let bottom = hit.pos.z + half_z;
        assert!(
            (bottom - 140.0).abs() < 5.0e-3,
            "AABB bottom {bottom} must sit on the z=140 plane"
        );
        // And further updates change nothing.
        sys.update(&scene, 1.0);
        assert_eq!(sys.drain_impacts().count(), 0);
    }

    /// Tunneling guard (maintainer review, DT.2): a fast body's
    /// single-frame displacement can dwarf the overlap window of a
    /// one-voxel island over a one-voxel shelf (window = 2). Tuned so
    /// the FIRST update moves 8 units clean across the shelf — the
    /// endpoint-only collision test deterministically skipped this;
    /// the substepped march must not. (Stock tuning at 30 FPS sits
    /// right at the window boundary — same failure, just alignment-
    /// dependent; this pins the mechanism, not the tuning.)
    #[test]
    fn fast_body_cannot_tunnel_thin_shelf() {
        let (mut scene, grid, island) = scene_with_island();
        // Shelf 4 voxels under the island (pivot z=100.5 → crossed
        // mid-first-step).
        scene.grid_mut(grid).expect("grid").set_rect(
            IVec3::new(0, 0, 104),
            IVec3::new(12, 6, 104),
            Some(STONE),
        );

        let mut sys = DebrisSystem::new();
        sys.gravity = 7200.0; // one tick at dt=1/30 ⇒ v = terminal
        sys.terminal_speed = 240.0; // step = 8 units/frame ≫ window 2
        assert!(sys.spawn_island(&mut scene, grid, island, BakeMode::Directional));
        sys.update(&scene, 1.0 / 30.0);
        let impacts: Vec<DebrisImpact> = sys.drain_impacts().collect();
        assert_eq!(impacts.len(), 1, "the shelf must not be tunnelled through");
        let bottom = impacts[0].pos.z + 0.5;
        assert!(
            (bottom - 104.0).abs() < 5.0e-3,
            "flush on the shelf: bottom {bottom}"
        );
    }

    /// Two bodies with floors at different heights land independently
    /// — pins the `swap_remove` mid-iteration path (one body retires
    /// while the other keeps falling; batch sizes shrink 2 → 1 → 0).
    #[test]
    fn two_bodies_land_independently() {
        let mut scene = Scene::new();
        let grid = scene.add_grid(GridTransform::identity());
        let g = scene.grid_mut(grid).expect("grid");
        // Two cut beams far apart, floors at different depths.
        for (y, floor_z) in [(2, 120), (30, 200)] {
            g.set_rect(IVec3::new(2, y, 100), IVec3::new(2, y, 255), Some(STONE));
            g.set_rect(IVec3::new(3, y, 100), IVec3::new(6, y, 100), Some(STONE));
            g.set_voxel(IVec3::new(3, y, 100), None);
            g.set_rect(
                IVec3::new(0, y - 2, floor_z),
                IVec3::new(12, y + 2, floor_z),
                Some(STONE),
            );
        }
        let mut islands = Vec::new();
        for y in [2, 30] {
            islands.extend(detect_islands(
                g,
                IVec3::new(3, y, 100),
                IVec3::new(3, y, 100),
                4096,
            ));
        }
        assert_eq!(islands.len(), 2);

        let mut sys = DebrisSystem::new();
        let mut f = Mock::default();
        for isl in islands {
            assert!(sys.spawn_island(&mut scene, grid, isl, BakeMode::Directional));
        }
        let mut landings = Vec::new();
        for frame in 0..900 {
            sys.update(&scene, 1.0 / 60.0);
            sys.sync_with(&mut f);
            for _ in sys.drain_impacts() {
                landings.push(frame);
            }
        }
        assert_eq!(landings.len(), 2, "both bodies land");
        assert!(
            landings[0] < landings[1],
            "the shallower floor catches its body first"
        );
        assert_eq!(sys.debris_count(), 0);
        assert!(f.batch_sizes.contains(&2), "both flew together");
        assert!(
            f.batch_sizes.contains(&1),
            "one kept flying after the first landed (swap_remove path)"
        );
        assert_eq!(f.despawns.len(), 2);
        assert_eq!(f.models_removed.len(), 2);
    }

    /// Spawn inside geometry (the island bbox wraps foreign solid):
    /// the explicit policy — an immediate zero-speed impact, no body.
    #[test]
    fn spawn_inside_geometry_shatters_in_place() {
        let mut scene = Scene::new();
        let grid = scene.add_grid(GridTransform::identity());
        let g = scene.grid_mut(grid).expect("grid");
        // A supported pillar the hand-built island's bbox wraps.
        g.set_rect(IVec3::new(5, 5, 150), IVec3::new(5, 5, 255), Some(STONE));
        // Two voxels either side of it (solid, so extract has work).
        g.set_voxel(IVec3::new(4, 5, 150), Some(STONE));
        g.set_voxel(IVec3::new(6, 5, 150), Some(STONE));
        let island = Island {
            voxels: vec![
                (IVec3::new(4, 5, 150), STONE),
                (IVec3::new(6, 5, 150), STONE),
            ],
            bbox: (IVec3::new(4, 5, 150), IVec3::new(6, 5, 150)),
        };

        let mut sys = DebrisSystem::new();
        assert!(sys.spawn_island(&mut scene, grid, island, BakeMode::Directional));
        assert_eq!(sys.debris_count(), 0, "never becomes a body");
        let impacts: Vec<DebrisImpact> = sys.drain_impacts().collect();
        assert_eq!(impacts.len(), 1);
        assert_eq!(impacts[0].speed, 0.0, "explicit zero-speed impact");
        let g = scene.grid(grid).expect("grid");
        assert!(!g.voxel_solid(IVec3::new(4, 5, 150)), "island extracted");
        assert!(g.voxel_solid(IVec3::new(5, 5, 150)), "the pillar survives");
    }

    /// Spawn extracts the island: its voxels are air in the grid the
    /// moment the body exists (the sprite twin replaces them).
    #[test]
    fn spawn_extracts_from_grid() {
        let (mut scene, grid, island) = scene_with_island();
        let voxels: Vec<IVec3> = island.voxels.iter().map(|&(v, _)| v).collect();
        let mut sys = DebrisSystem::new();
        assert!(sys.spawn_island(&mut scene, grid, island, BakeMode::Directional));
        let g = scene.grid(grid).expect("grid");
        for v in voxels {
            assert!(!g.voxel_solid(v), "{v} extracted");
        }
    }

    /// Mock facade recording the binding traffic `sync_with` emits.
    #[derive(Default)]
    struct Mock {
        next: u32,
        models_added: usize,
        models_removed: Vec<SpriteModelId>,
        spawns: Vec<(SpriteModelId, DynSpriteTransform)>,
        despawns: Vec<SpriteInstanceId>,
        batch_sizes: Vec<usize>,
    }

    impl DebrisFacade for Mock {
        fn add_model(&mut self, _kv6: &roxlap_formats::kv6::Kv6) -> SpriteModelId {
            self.models_added += 1;
            SpriteModelId::mint(0, self.next)
        }
        fn remove_model(&mut self, id: SpriteModelId) {
            self.models_removed.push(id);
        }
        fn spawn(
            &mut self,
            model: SpriteModelId,
            xf: DynSpriteTransform,
        ) -> Option<SpriteInstanceId> {
            self.spawns.push((model, xf));
            let id = SpriteInstanceId {
                slot: self.next,
                gen: 0,
            };
            self.next += 1;
            Some(id)
        }
        fn despawn(&mut self, id: SpriteInstanceId) {
            self.despawns.push(id);
        }
        fn set_transforms(&mut self, batch: &[(SpriteInstanceId, DynSpriteTransform)]) {
            self.batch_sizes.push(batch.len());
        }
    }

    /// The full binding lifecycle through the facade seam: one model +
    /// one pre-posed spawn at the island's world pivot with the yaw
    /// basis (scaled), batch moves while falling, despawn + model
    /// removal after landing — nothing leaked.
    #[test]
    fn sync_lifecycle_through_mock() {
        let (mut scene, grid, island) = scene_with_island();
        let pivot = island.world_pivot(&scene.grid(grid).expect("grid").transform);
        scene.grid_mut(grid).expect("grid").set_rect(
            IVec3::new(0, 0, 140),
            IVec3::new(12, 6, 140),
            Some(STONE),
        );

        let mut sys = DebrisSystem::new();
        let mut f = Mock::default();
        assert!(sys.spawn_island(&mut scene, grid, island, BakeMode::Directional));
        sys.sync_with(&mut f);
        assert_eq!(f.models_added, 1);
        assert_eq!(f.spawns.len(), 1, "pre-posed spawn, no move batch yet");
        let xf = f.spawns[0].1;
        assert!(
            (f64::from(xf.pos[2]) - pivot.z).abs() < 1e-4,
            "spawned at the world pivot"
        );
        assert!((xf.forward[2] - 1.0).abs() < 1e-6, "identity-scale basis");
        assert_eq!(f.batch_sizes.len(), 0);

        // Fall to the floor, syncing per frame like a host would.
        for _ in 0..600 {
            sys.update(&scene, 1.0 / 60.0);
            sys.sync_with(&mut f);
        }
        assert_eq!(sys.debris_count(), 0);
        assert!(!f.batch_sizes.is_empty(), "live frames batch-moved");
        assert!(f.batch_sizes.iter().all(|&n| n == 1));
        assert_eq!(f.despawns.len(), 1, "instance retired on impact");
        assert_eq!(f.models_removed.len(), 1, "model retired on impact");
    }

    /// An empty island or a stale grid id is refused.
    #[test]
    fn spawn_rejects_empty_or_missing() {
        let (mut scene, grid, island) = scene_with_island();
        let mut sys = DebrisSystem::new();
        assert!(!sys.spawn_island(
            &mut scene,
            grid,
            Island {
                voxels: Vec::new(),
                bbox: (IVec3::MAX, IVec3::MIN),
            },
            BakeMode::Directional,
        ));
        let stale = {
            let mut other = Scene::new();
            // A grid id from a different scene: unknown here.
            other.add_grid(GridTransform::identity());
            other.add_grid(GridTransform::identity())
        };
        assert!(!sys.spawn_island(&mut scene, stale, island, BakeMode::Directional));
        assert_eq!(sys.debris_count(), 0);
    }
}
