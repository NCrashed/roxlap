//! The **Boarding** scene (stage OC) — the occlusion-cutout keyhole in
//! third person: the Decks shiplet rebuilt with walkable stairwells, a
//! CC [`CharacterBody`] steered over its decks (auto-swims the flooded
//! bilge), and a shoulder follow-camera that stays OUTSIDE the hull —
//! the walls between the camera and the character melt inside the
//! screen-space keyhole ([`ViewCutout`]) instead of being
//! raycast-boomed around. `PgUp`/`PgDn` additionally slide the CA deck
//! clip: the two cuts COMPOSE — that composition is the scene's point
//! (CA cuts ceilings above, OC cuts walls in front).
//!
//! The cutout is a view aid, not world removal: hidden walls keep
//! casting sun shadow into the keyhole and keep their collision.
//!
//! Controls: WASD walks · Space jumps (strokes up when swimming) ·
//! Shift sinks · mouse orbits · wheel resizes the keyhole · `K`
//! keyhole on/off · `V` deck view follows the character / manual ·
//! `C` shoulder ↔ tele-iso deck orbit · `PgUp`/`PgDn` manual decks.

use glam::DVec3;
use roxlap_render::{
    ActorState, BillboardActorDef, BillboardActorId, BillboardLighting, BillboardMode,
    DirectionalLight, Kv6, LightRig, LoopMode, Material, PointLight, ShadowFlags, ViewCutout,
    VoxelClip,
};
use roxlap_scene::{
    voxel_global, world_to_grid_local, BakeLight, BakeMode, CharacterBody, CharacterDef, DeckBand,
    FogOfWar, FowObserver, FowTwin, GridId, LightGate, Scene, VisionConfig, WalkInput,
};
use winit::keyboard::KeyCode;

use super::decks::{
    build_ship_world, cabin_lights, DECK_CLIPS, FLOOR_TOPS, HULL_X, HULL_Y, MAT_WATER, ROOF_Z,
    SHIP_ORIGIN_Z, WATER_RGB,
};
use super::world::{figure_frame, FIGURE_H_VOX};
use crate::scene_api::{
    frame_params, opticast_settings, CameraPose, CameraRig, DemoScene, SceneCtx, SceneInput,
};

/// Character proportions in ship voxels (≈ ×6.7 the stock human-scale
/// [`CharacterDef`]): 12 tall in 20-voxel rooms, matching the 24-slab
/// figure drawn at [`FIGURE_SCALE`].
const BODY_HEIGHT: f64 = 12.0;
const BODY_RADIUS: f64 = 2.5;
const EYE_HEIGHT: f64 = 10.0;
/// Clears the 4-voxel stair steps with a little margin.
const STEP_UP: f64 = 4.4;
const WALK_SPEED: f64 = 26.0;
/// Jump apex ≈ 5.2 voxels (v²/2g) — clears one stair step.
const JUMP_SPEED: f64 = 38.0;
const GRAVITY: f64 = 140.0;

/// Figure scale: 24 slab voxels × 0.5 = the 12-voxel body height.
const FIGURE_SCALE: f32 = 0.5;

/// Shoulder-boom length (world units). No raycast clamp on purpose —
/// letting the boom cross the hull is what the keyhole is FOR.
const BOOM_DIST: f64 = 70.0;

/// The `C` camera mode — the Decks tele-iso preset following the
/// character: narrow FOV + distant orbit (the SS13-style deck view),
/// with the GPU LOD range raised past the orbit distance while active.
const TELE_FOV_Y: f32 = 0.15;
const TELE_DIST: f64 = 1150.0;
const TELE_SCAN_DIST: i32 = 2048;
const TELE_MIP_SCAN: f32 = 640.0;

/// Keyhole defaults (logical px); the wheel steps the radius.
const RADIUS_DEFAULT: f32 = 110.0;
const RADIUS_STEP: f32 = 10.0;
const RADIUS_RANGE: (f32, f32) = (30.0, 320.0);
/// Wide taper band: the reveal distance falls off across it, so the
/// rim reads as a bevelled funnel instead of a hard cylinder.
const FEATHER_PX: f32 = 24.0;
/// Reveal margin: how far short of the character column the cut
/// stops. Small on purpose — an obstacle the character stands right
/// behind must melt all the way to the body (visual-pass round 4).
const MARGIN: f32 = 1.5;
/// Focus plane bias: the focus sits at chest height (feet − 6); +6.5
/// puts the cutting plane AT the feet (floor level, after the floor()
/// in the per-grid conversion), so front walls cut all the way down
/// to the boots while the floor plate itself stays. (Hazard 7's
/// original head-level tune left waist-high wall stumps in front of
/// the character — the visual-pass round-3 report.)
const Z_BIAS: f32 = 6.5;
/// Chest height above the feet (−z), the keyhole focus anchor.
const CHEST: f64 = 6.0;

/// Spawn feet, clear of the crate and the stairwells. Half a voxel
/// ABOVE the surface (z-down: −0.5): feet EXACTLY at the boundary
/// make the first downward sweep degenerate and the body falls
/// through its own floor; a short drop lands with the proper
/// collision skin. Capture hook: `ROXLAP_BOARD_DECK=0..=2` spawns on
/// that deck's floor (2 = the flooded bilge) instead of the top deck.
fn spawn_feet() -> DVec3 {
    let deck = std::env::var("ROXLAP_BOARD_DECK")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .map_or(0, |d| d.min(FLOOR_TOPS.len() - 1));
    DVec3::new(
        44.0,
        64.0,
        f64::from(FLOOR_TOPS[deck]) + SHIP_ORIGIN_Z - 0.5,
    )
}

const START_YAW: f64 = std::f64::consts::FRAC_PI_2;
const START_PITCH: f64 = 0.55;

/// FW.5 — how often the scripted "noise behind the wall" fires (seconds)
/// and where its source sits (grid-local voxel, on the top deck behind a
/// bulkhead from spawn), so the heard pocket is clearly OFF the observer.
const NOISE_PERIOD: f64 = 2.0;
const NOISE_SOURCE_LOCAL: [f64; 3] = [96.0, 64.0, 252.0];

/// FW.5 — grid-local z of the flooded-bilge water surface. Kept in sync
/// with the flood fill in [`build_ship_world`] (`FLOOR_TOPS[2] - 8`);
/// the bottom deck's fog eye sits just above it (see `ship_vision_config`
/// #2) so the observer isn't blocked by the solid water below.
const BILGE_WATER_TOP: i32 = FLOOR_TOPS[2] - 8;

/// FW.5 — the ship's fog-of-war vision: one deck band per walkable deck,
/// derived from the same [`DECK_CLIPS`]/[`FLOOR_TOPS`] the CA clip uses
/// (band = clip plane .. floor, z-down). `light_gate` (the `G` toggle)
/// caps vision at unlit surfaces.
fn ship_vision_config(light_gate: bool) -> VisionConfig {
    let decks: Vec<DeckBand> = (0..FLOOR_TOPS.len())
        .map(|i| {
            let z_bottom = FLOOR_TOPS[i];
            let z_top = DECK_CLIPS[i + 1].expect("decks 1..=3 are clipped");
            // The eye band is no longer here — it rides the observer
            // (`fow_observer`'s `eye_z`), so LOS tracks the character's
            // real eye over stairs / ramps / a floating swim, not a
            // fixed floor-relative slab.
            DeckBand { z_top, z_bottom }
        })
        .collect();
    let mut cfg = VisionConfig::for_decks(decks);
    // Ship-scale: a ~100° cone reaching most of a deck + a small
    // 360° peripheral, quicker memory decay than the defaults.
    cfg.cone_half_angle = 0.9;
    // Visual-pass round 5 (#4): a wider angular rim + fast fade-in so the
    // FOV edge and the turning leading-edge settle to full brightness
    // instead of trailing as dim lines against the unseen black.
    cfg.cone_taper = 0.22;
    cfg.fade_in = 600.0;
    cfg.range = 60.0;
    cfg.peripheral_range = 8.0;
    cfg.memory_decay = 2.0;
    if light_gate {
        // Visual-pass round 5 (#3): the ship bakes its cabin lamps
        // (`BakeMode::PointLights`) into the brightness byte — a dim base
        // (~50–75 at the room ends) rising to a bright pool (~130–155)
        // under each lamp. The knee (`lit_brightness - softness` = 85)
        // sits between them, so lamp-lit floor reads as seen while the
        // dim ends stay Memory even inside the cone — "you only KNOW what
        // the lamps light."
        cfg.light_gate = Some(LightGate {
            lit_brightness: 130,
            softness: 45,
            emissive_colors: Vec::new(),
        });
    }
    cfg
}

/// FW.5 (#3) — bake the cabin lamps statically into the ship's per-voxel
/// brightness byte ([`BakeMode::PointLights`]: a dim base plus a bright
/// pool per lamp), so the fog light gate has real contrast to read: lamp
/// pools clear its knee (lit → seen) while unlit corners fall below it
/// (dark → held at Memory even when in the cone). Boarding-only — the
/// Decks scene keeps its uniform base — and the dynamic cabin lights
/// still render on top for colour + the sprite fill.
fn bake_ship_lamps(scene: &mut Scene, ship: GridId) {
    let Some(grid) = scene.grid_mut(ship) else {
        return;
    };
    #[allow(clippy::cast_possible_truncation)]
    let lights: Vec<BakeLight> = FLOOR_TOPS
        .iter()
        .map(|&ft| BakeLight {
            // Grid-local voxel frame (the ship origin only offsets z), one
            // lamp per deck at the same spot the dynamic `cabin_lights`
            // sit — 18 voxels above the floor.
            pos: glam::Vec3::new(64.0, 56.0, (ft - 18) as f32),
            radius: 80.0,
            strength: 30_000.0,
        })
        .collect();
    grid.bake_lights = lights;
    grid.bake(BakeMode::PointLights);
}

// Four independent user toggles (deck-follow, keyhole, camera mode,
// first-enter radius latch) — not an encoded state machine.
#[allow(clippy::struct_excessive_bools)]
pub struct BoardingScene {
    scene: Scene,
    ship: GridId,
    /// Current deck view, indexing [`DECK_CLIPS`] (0 = whole ship).
    deck: usize,
    /// `V` — deck view follows the character (the CA clip exposes the
    /// deck the body is on, so the hull around the keyhole reads as a
    /// deck plan instead of the roof). `PgUp`/`PgDn` switch to manual.
    auto_clip: bool,
    /// `K` — the keyhole on/off.
    keyhole_on: bool,
    /// `C` — camera mode: shoulder boom (false) or the Decks tele-iso
    /// orbit (true), both following the character.
    tele: bool,
    /// GPU LOD range to restore when leaving tele mode / the scene.
    saved_mip_scan: Option<f32>,
    /// The walking body (CC stage; auto-swims the bilge — WT).
    body: CharacterBody,
    /// The third-person figure posed at the body every frame.
    actor: Option<BillboardActorId>,
    /// Keyhole radius (logical px), wheel-adjusted. Sized to the
    /// logical frame on the FIRST enter only — a wheel-tuned radius
    /// survives tab switches like every other Boarding toggle.
    radius_px: f32,
    radius_inited: bool,
    /// Per-deck cabin lights + this frame's footprint-culled survivors
    /// (the Decks light-cull pattern, decision 4).
    cabin_lights: Vec<PointLight>,
    lit_points: Vec<PointLight>,
    /// FW.5 — fog of war. `F` toggles it: on, the ship becomes a
    /// `render_excluded` real grid shown through the known `twin`, and
    /// `render` passes the mask so only the character's cone / memory /
    /// heard pockets render. `G` toggles the light gate.
    fow: FogOfWar,
    twin: Option<FowTwin>,
    fow_on: bool,
    light_gate_on: bool,
    /// Scripted "noise behind the wall" countdown (seconds).
    noise_timer: f64,
}

impl BoardingScene {
    #[must_use]
    pub fn new() -> Self {
        // The walkable hull: stairwells carved through each deck floor.
        let (mut scene, ship) = build_ship_world(true);
        // FW.5 (#3): bake the cabin lamps so the fog light gate has
        // brightness contrast to read (dim corners vs lit pools).
        bake_ship_lamps(&mut scene, ship);
        let mut body = CharacterBody::new(CharacterDef {
            radius: BODY_RADIUS,
            height: BODY_HEIGHT,
            eye_height: EYE_HEIGHT,
            step_up: STEP_UP,
            walk_speed: WALK_SPEED,
            jump_speed: JUMP_SPEED,
            gravity: GRAVITY,
            max_fall_speed: 200.0,
            accel_ground: 260.0,
            accel_air: 50.0,
            // WT tunings scaled with the body: enough buoyancy to bob
            // at the bilge waterline, swim about half walking speed.
            buoyancy: 210.0,
            swim_speed: 16.0,
            swim_accel: 130.0,
            ..CharacterDef::default()
        });
        body.teleport(spawn_feet());
        Self {
            scene,
            ship,
            deck: 0,
            auto_clip: true,
            keyhole_on: true,
            tele: false,
            saved_mip_scan: None,
            body,
            actor: None,
            radius_px: RADIUS_DEFAULT,
            radius_inited: false,
            cabin_lights: cabin_lights(),
            lit_points: Vec::new(),
            fow: FogOfWar::new(ship_vision_config(false)),
            twin: None,
            fow_on: false,
            light_gate_on: false,
            noise_timer: NOISE_PERIOD,
        }
    }

    /// Visual-pass round 6 (#2) — the fog deck the body is IN, 0-based
    /// into [`ship_vision_config`]'s decks. Uses the body CENTRE against
    /// the deck floors, NOT the head like the CA clip's `deck_for_body`:
    /// standing on a column lifts the head above the deck's ceiling, so
    /// the head-pick jumps to the deck ABOVE and the fog then reveals the
    /// wrong floor (the player saw nothing on the deck they stood on).
    /// The centre stays inside the deck the feet rest on. Clamped to the
    /// bottom deck (never `None`).
    fn fog_deck_index(&self) -> usize {
        #[allow(clippy::cast_possible_truncation)]
        let local_center = self.body.pos().z - SHIP_ORIGIN_Z - BODY_HEIGHT * 0.5;
        FLOOR_TOPS
            .iter()
            .position(|&ft| local_center <= f64::from(ft))
            .unwrap_or(FLOOR_TOPS.len() - 1)
    }

    /// FW.5 — the character's fog observer: grid-local cell, facing (the
    /// camera's horizontal aim), and the fog deck the body is in.
    fn fow_observer(&self, ctx: &SceneCtx) -> FowObserver {
        let transform = self
            .scene
            .grid(self.ship)
            .map(|g| g.transform)
            .unwrap_or_default();
        let glp = world_to_grid_local(self.body.pos(), &transform);
        let v = voxel_global(glp.chunk, glp.voxel);
        let fwd = ctx.cam.camera().forward;
        // The character's real eye, grid-local (z-down: EYE_HEIGHT above
        // the feet is a SMALLER z). This rides stairs / ramps / the swim
        // instead of a fixed floor slab. One clamp remains: in the
        // flooded bilge the solid water blocks LOS, so keep the eye band
        // clear of the waterline (a swimmer sees ACROSS the surface, not
        // walled in by it) — the only water-specific bit, pending a
        // material-aware LOS sample.
        #[allow(clippy::cast_possible_truncation)]
        let eye_z = (self.body.pos().z - EYE_HEIGHT - SHIP_ORIGIN_Z) as i32;
        let eye_z = eye_z.min(BILGE_WATER_TOP - 3);
        #[allow(clippy::cast_possible_truncation)]
        FowObserver {
            cell: glam::IVec2::new(v.x, v.y),
            facing: glam::Vec2::new(fwd[0] as f32, fwd[1] as f32),
            deck: self.fog_deck_index(),
            eye_z,
        }
    }

    /// FW.5 — turn the fog on/off: attach / detach the known twin (which
    /// flips the ship to `render_excluded` and renders the twin instead).
    fn set_fow(&mut self, on: bool) {
        if on && self.twin.is_none() {
            self.twin = Some(FowTwin::attach(&mut self.scene, self.ship));
        } else if !on {
            if let Some(t) = self.twin.take() {
                t.detach(&mut self.scene);
            }
        }
        self.fow_on = on;
        eprintln!("boarding: fog of war = {}", if on { "ON" } else { "off" });
    }

    /// The deck view the character's position calls for: the deck the
    /// body is on (1..=3), or "whole ship" while on/above the roof or
    /// outside the hull footprint.
    ///
    /// Decided by the HEAD, not the feet: the clip may only switch to
    /// the deck below once the WHOLE body is under the upper deck's
    /// floor level. A feet-based switch fires on the top stair steps
    /// while the figure's anchor (its volume centre) still sits above
    /// the new clip plane — the CA sprite footprint rule then hides
    /// it and the player vanishes mid-staircase. Step heights (4)
    /// keep the head threshold strictly between step levels, so the
    /// view doesn't flicker while climbing.
    fn deck_for_body(&self) -> usize {
        let feet = self.body.pos();
        let local_head = feet.z - SHIP_ORIGIN_Z - BODY_HEIGHT;
        #[allow(clippy::cast_possible_truncation)]
        let (x, y) = (feet.x.floor() as i32, feet.y.floor() as i32);
        let inside_xy = x >= HULL_X.0 && x <= HULL_X.1 && y >= HULL_Y.0 && y <= HULL_Y.1;
        if !inside_xy || local_head < f64::from(ROOF_Z) - 2.0 {
            return 0;
        }
        FLOOR_TOPS
            .iter()
            .position(|&ft| local_head <= f64::from(ft) + 0.5)
            .map_or(0, |d| d + 1)
    }

    fn deck_label(&self) -> &'static str {
        [
            "whole ship",
            "deck 1 (top)",
            "deck 2 (mid)",
            "deck 3 (flooded)",
        ][self.deck]
    }

    fn set_deck(&mut self, deck: usize) {
        self.deck = deck;
        self.scene.set_grid_z_clip(self.ship, DECK_CLIPS[deck]);
        eprintln!("boarding: deck view {}", self.deck_label());
    }

    /// The keyhole focus: chest height on the body.
    fn focus(&self) -> DVec3 {
        self.body.pos() - DVec3::new(0.0, 0.0, CHEST)
    }

    /// The figure's anchor: clip pivot is the volume CENTRE, so half
    /// the figure's height above (−z of) the feet.
    #[allow(clippy::cast_possible_truncation)]
    fn actor_pos(&self) -> [f32; 3] {
        let feet = self.body.pos();
        let half_h = f64::from(FIGURE_H_VOX) * f64::from(FIGURE_SCALE) * 0.5;
        [feet.x as f32, feet.y as f32, (feet.z - half_h) as f32]
    }

    /// Register the figure's walk/idle states and spawn it at the body
    /// (same silhouette as the World scene, drawn at ship scale).
    fn spawn_actor(&mut self, ctx: &mut SceneCtx) -> Option<BillboardActorId> {
        let clip = |frames: &[Kv6]| {
            VoxelClip::from_kv6_frames(frames, 1.0, LoopMode::Loop, &[], 220, 1)
                .expect("figure frames are non-empty + same dims")
        };
        let walk = ctx.renderer.add_voxel_clip(
            &clip(&[figure_frame(3), figure_frame(0)])
                .decode()
                .expect("clip decodes"),
        );
        let idle = ctx
            .renderer
            .add_voxel_clip(&clip(&[figure_frame(1)]).decode().expect("clip decodes"));
        let def = BillboardActorDef {
            states: vec![
                ActorState {
                    name: "walk".to_owned(),
                    dirs: vec![walk],
                },
                ActorState {
                    name: "idle".to_owned(),
                    dirs: vec![idle],
                },
            ],
            mode: BillboardMode::Cylindrical,
            lighting: BillboardLighting::FaceNormal,
            speed: 1.0,
            scale: FIGURE_SCALE,
            shadows: ShadowFlags::default(),
        };
        ctx.renderer
            .add_billboard_actor(def, self.actor_pos(), ctx.cam.yaw)
    }

    /// Place the host camera on the shoulder boom: back from the chest
    /// along the current (yaw, pitch) view direction. Deliberately NOT
    /// clamped by a raycast — the keyhole handles the hull in between.
    fn boom_camera(&self, ctx: &mut SceneCtx) {
        let f = crate::scene::camera_for_yaw_pitch([0.0; 3], ctx.cam.yaw, ctx.cam.pitch).forward;
        let focus = self.focus();
        ctx.cam.pos = [
            focus.x - f[0] * BOOM_DIST,
            focus.y - f[1] * BOOM_DIST,
            focus.z - f[2] * BOOM_DIST,
        ];
    }
}

impl DemoScene for BoardingScene {
    fn name(&self) -> &'static str {
        "Boarding"
    }

    fn controls(&self) -> &'static str {
        "WASD walks · Space jumps · mouse orbits · wheel: keyhole radius · K: keyhole · V: deck-follow cut · C: shoulder/tele-iso · PgUp/PgDn manual decks · F: fog of war · G: fog light gate"
    }

    fn start_pose(&self) -> CameraPose {
        let f = crate::scene::camera_for_yaw_pitch([0.0; 3], START_YAW, START_PITCH).forward;
        let focus = spawn_feet() - DVec3::new(0.0, 0.0, CHEST);
        CameraPose {
            pos: [
                focus.x - f[0] * BOOM_DIST,
                focus.y - f[1] * BOOM_DIST,
                focus.z - f[2] * BOOM_DIST,
            ],
            yaw: START_YAW,
            pitch: START_PITCH,
        }
    }

    fn enter(&mut self, ctx: &mut SceneCtx) {
        // Size the default keyhole to the LOGICAL frame (the radius is
        // in logical pixels, so a fixed value covers half as much of a
        // 2× logical grid) — on the FIRST enter only; a wheel-tuned
        // radius survives tab switches like the other toggles.
        if !self.radius_inited {
            self.radius_inited = true;
            let (lw, lh) = ctx.renderer.logical_dims();
            #[allow(clippy::cast_precision_loss)]
            let base = 0.42 * lw.min(lh) as f32;
            self.radius_px = base.clamp(RADIUS_RANGE.0, RADIUS_RANGE.1);
        }
        // The bilge flood is the WT Beer–Lambert water material.
        ctx.renderer
            .define_material(MAT_WATER, Material::volumetric(30));
        ctx.renderer
            .set_terrain_materials(&[(WATER_RGB.rgb_part(), MAT_WATER)]);
        // Content layers were reset on scene switch — respawn the
        // figure and restore this visit's deck cut / tele LOD range.
        self.actor = self.spawn_actor(ctx);
        self.body.teleport(spawn_feet());
        self.scene.set_grid_z_clip(self.ship, DECK_CLIPS[self.deck]);
        if self.tele {
            self.saved_mip_scan = Some(ctx.renderer.gpu_mip_scan_dist());
            ctx.renderer.set_gpu_mip_scan_dist(TELE_MIP_SCAN);
        }
        // FW.5 — re-arm the fog twin if the toggle was left on (`exit`
        // detached it so the ship renders normally in the other tabs).
        if self.fow_on && self.twin.is_none() {
            self.twin = Some(FowTwin::attach(&mut self.scene, self.ship));
        }
    }

    fn update(&mut self, ctx: &mut SceneCtx, dt: f64) {
        // Clamp the physics step: the first frame after startup (or a
        // scene switch / window drag) arrives with seconds of wall
        // time, and at ship-scale gravity one un-clamped step drives
        // the body through a 4-voxel deck plate.
        let dt = dt.min(0.05);
        // WASD steers in the camera's frame; Walk mode uses only x/y.
        let wish = DVec3::from(ctx.cam.wish_dir(ctx.input));
        self.body.walk(
            &self.scene,
            dt,
            WalkInput {
                wish,
                jump: ctx.input.up,
                sink: ctx.input.down,
            },
        );

        // `V` — deck view follows the character: expose the deck the
        // body is on, so the hull around the keyhole reads as a deck
        // plan instead of the roof (the CA/OC composition).
        if self.auto_clip {
            let want = self.deck_for_body();
            if want != self.deck {
                self.set_deck(want);
            }
        }

        // Pose + animate the figure, then place the camera BEFORE the
        // tick so the cylindrical billboard faces the real camera.
        if let Some(actor) = self.actor {
            ctx.renderer
                .set_actor_transform(actor, self.actor_pos(), ctx.cam.yaw);
            let moving = self.body.vel().truncate().length() > 1.0;
            ctx.renderer
                .set_actor_state(actor, if moving { "walk" } else { "idle" });
        }
        if self.tele {
            // The Decks tele-iso orbit, re-derived from the current
            // yaw/pitch every frame so mouse-look circles the body.
            let f = self.focus();
            let rig = CameraRig::tele_iso([f.x, f.y, f.z], ctx.cam.yaw, ctx.cam.pitch, TELE_DIST);
            ctx.cam.pos = rig.pos;
        } else {
            self.boom_camera(ctx);
        }
        if self.actor.is_some() {
            ctx.renderer.tick(&ctx.cam.camera(), dt);
        }

        // FW.5 — advance the fog of war for the character, sync the known
        // twin (the geometry the observer has seen, frozen), and fire the
        // scripted "noise behind the wall" so a heard pocket reveals live
        // geometry the character can't see.
        if self.fow_on {
            let obs = self.fow_observer(ctx);
            #[allow(clippy::cast_possible_truncation)]
            let dtf = dt as f32;
            if let Some(g) = self.scene.grid(self.ship) {
                self.fow.update(g, &obs, dtf);
            }
            if let Some(twin) = &mut self.twin {
                if !twin.sync(&mut self.scene, &self.fow) {
                    // The twin was lost (shouldn't happen here) — re-arm.
                    self.twin = None;
                    self.fow_on = false;
                }
            }
            self.noise_timer -= dt;
            if self.noise_timer <= 0.0 {
                self.noise_timer = NOISE_PERIOD;
                let src = DVec3::new(
                    NOISE_SOURCE_LOCAL[0],
                    NOISE_SOURCE_LOCAL[1],
                    NOISE_SOURCE_LOCAL[2] + SHIP_ORIGIN_Z,
                );
                let listener = self.body.pos() - DVec3::new(0.0, 0.0, EYE_HEIGHT);
                roxlap_audio::hear_source(
                    &self.scene,
                    &mut self.fow,
                    self.ship,
                    src,
                    listener,
                    1.4,
                    &roxlap_audio::AcousticsConfig::default(),
                );
            }
        }

        // Decision 4 — the Decks light-cull pattern: lamps on
        // deck-clip-hidden decks stop lighting the exposed ones.
        self.lit_points.clear();
        self.lit_points.extend(
            self.cabin_lights
                .iter()
                .filter(|l| {
                    !self.scene.cutaway_hides_point(DVec3::new(
                        f64::from(l.position[0]),
                        f64::from(l.position[1]),
                        f64::from(l.position[2]),
                    ))
                })
                .copied(),
        );
    }

    fn on_input(&mut self, ctx: &mut SceneCtx, ev: &SceneInput) {
        match ev {
            SceneInput::Key {
                code: KeyCode::PageDown,
                pressed: true,
            } => {
                self.auto_clip = false;
                let d = (self.deck + 1).min(DECK_CLIPS.len() - 1);
                self.set_deck(d);
            }
            SceneInput::Key {
                code: KeyCode::PageUp,
                pressed: true,
            } => {
                self.auto_clip = false;
                self.set_deck(self.deck.saturating_sub(1));
            }
            SceneInput::Key {
                code: KeyCode::KeyK,
                pressed: true,
            } => {
                self.keyhole_on = !self.keyhole_on;
                eprintln!(
                    "boarding: keyhole = {}",
                    if self.keyhole_on { "ON" } else { "OFF" }
                );
            }
            SceneInput::Key {
                code: KeyCode::KeyV,
                pressed: true,
            } => {
                self.auto_clip = !self.auto_clip;
                eprintln!(
                    "boarding: deck view = {}",
                    if self.auto_clip {
                        "FOLLOWS the character (PgUp/PgDn for manual)"
                    } else {
                        "MANUAL (PgUp/PgDn; V to follow again)"
                    }
                );
            }
            SceneInput::Key {
                code: KeyCode::KeyC,
                pressed: true,
            } => {
                self.tele = !self.tele;
                if self.tele {
                    // Distance LOD would coarsen the ship from the
                    // tele orbit; hold mip 0 past the working range.
                    self.saved_mip_scan = Some(ctx.renderer.gpu_mip_scan_dist());
                    ctx.renderer.set_gpu_mip_scan_dist(TELE_MIP_SCAN);
                } else if let Some(d) = self.saved_mip_scan.take() {
                    ctx.renderer.set_gpu_mip_scan_dist(d);
                }
                eprintln!(
                    "boarding: camera = {}",
                    if self.tele {
                        "TELE-ISO deck orbit (C back to shoulder)"
                    } else {
                        "SHOULDER"
                    }
                );
            }
            SceneInput::Key {
                code: KeyCode::KeyF,
                pressed: true,
            } => {
                let on = !self.fow_on;
                self.set_fow(on);
            }
            SceneInput::Key {
                code: KeyCode::KeyG,
                pressed: true,
            } => {
                self.light_gate_on = !self.light_gate_on;
                self.fow.set_config(ship_vision_config(self.light_gate_on));
                eprintln!(
                    "boarding: fog light gate = {}",
                    if self.light_gate_on { "ON" } else { "off" }
                );
            }
            SceneInput::Wheel { dy } => {
                #[allow(clippy::cast_possible_truncation)]
                let step = *dy as f32 * RADIUS_STEP;
                self.radius_px = (self.radius_px + step).clamp(RADIUS_RANGE.0, RADIUS_RANGE.1);
                eprintln!("boarding: keyhole radius = {:.0} px", self.radius_px);
            }
            _ => {}
        }
    }

    fn render(&mut self, ctx: &mut SceneCtx) {
        // Shoulder mode renders with the host projection; tele mode
        // with the Decks narrow-FOV preset (the "iso" flattening).
        let (settings, scan) = if self.tele {
            (
                opticast_settings(ctx.size, TELE_SCAN_DIST).with_fov_y(TELE_FOV_Y),
                TELE_SCAN_DIST,
            )
        } else {
            (opticast_settings(ctx.size, ctx.scan_dist), ctx.scan_dist)
        };
        let mut frame = frame_params(ctx.engine, &settings, scan);
        // The Decks sun: with the keyhole open, the cut hull still
        // shadows the interior — visibly a VIEW cut, not removal.
        frame.lights = Some(LightRig {
            sun: Some(DirectionalLight {
                direction: [0.45, 0.35, 0.82],
                color: [1.0, 0.95, 0.85],
                intensity: 1.1,
                casts_shadow: true,
            }),
            points: &self.lit_points,
            spots: &[],
            ambient: [0.4, 0.42, 0.48],
            shadow_strength: 0.8,
            shadow_bias_voxels: 1.5,
            shadow_max_dist: 110.0,
            bands: 0,
            shadow_tint: [0.0; 3],
        });
        // OC.3 — the keyhole, keyed to the character's chest (`K`
        // toggles it). The facade derives everything else.
        if self.keyhole_on {
            #[allow(clippy::cast_possible_truncation)]
            let focus = self.focus();
            frame.view_cutout = Some(ViewCutout {
                focus_world: [focus.x as f32, focus.y as f32, focus.z as f32],
                radius_px: self.radius_px,
                feather_px: FEATHER_PX,
                margin: MARGIN,
                z_bias: Z_BIAS,
            });
        }
        // FW.5 — the fog mask, keyed to the known twin grid (renders in
        // place of the render-excluded ship). Composes with the CA deck
        // clip + the OC keyhole above — all three cuts at once.
        if self.fow_on {
            if let Some(twin) = &self.twin {
                frame.fow = Some((twin.twin(), &self.fow));
            }
        }
        let camera = ctx.cam.camera();
        ctx.renderer.render(&mut self.scene, &camera, &frame);
    }

    fn exit(&mut self, ctx: &mut SceneCtx) {
        // Restore the entry-time GPU LOD range for the other scenes.
        if let Some(d) = self.saved_mip_scan.take() {
            ctx.renderer.set_gpu_mip_scan_dist(d);
        }
        // FW.5 — detach the twin so the ship isn't left render-excluded
        // in the other tabs (the `fow_on` flag survives; `enter` re-arms).
        if let Some(t) = self.twin.take() {
            t.detach(&mut self.scene);
        }
    }

    fn hud_lines(&self) -> Vec<String> {
        vec![
            format!(
                "keyhole: {} {:.0} px (K/wheel) · deck view: {}{} · camera: {}",
                if self.keyhole_on { "ON" } else { "off" },
                self.radius_px,
                self.deck_label(),
                if self.auto_clip {
                    " (follows, V)"
                } else {
                    " (manual, V)"
                },
                if self.tele {
                    "tele-iso (C)"
                } else {
                    "shoulder (C)"
                },
            ),
            format!(
                "body: {} · cabin lights lit: {}/{}",
                if self.body.is_swimming() {
                    "swimming"
                } else if self.body.on_ground() {
                    "walking"
                } else {
                    "airborne"
                },
                self.lit_points.len(),
                self.cabin_lights.len()
            ),
            format!(
                "fog of war: {} (F) · light gate: {} (G){}",
                if self.fow_on { "ON" } else { "off" },
                if self.light_gate_on { "ON" } else { "off" },
                if self.fow_on {
                    " · a scripted noise reveals a heard pocket every 2 s"
                } else {
                    ""
                },
            ),
        ]
    }
}
