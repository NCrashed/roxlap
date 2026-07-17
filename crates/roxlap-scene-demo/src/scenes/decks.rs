//! The **Decks** scene (stage CA) — cutaway deck rendering: a 3-deck
//! shiplet (stacked chz chunks) on a ground plate, viewed through the
//! tele-iso preset (narrow FOV + distant orbit camera). `PgUp`/`PgDn`
//! slide the cut plane one deck at a time via
//! [`roxlap_scene::Scene::set_grid_z_clip`]; the exposed cross-section
//! renders with the run-top cut-face rule, hidden decks stop casting
//! sun shadows (CA.2/CA.3), actors on hidden decks vanish under the
//! footprint rule while the one on the ground beside the ship stays
//! (CA.4), and the bottom deck is flooded (hazard 6: the water voxels
//! live in the slabs, so they clip with everything else).
//!
//! Lights are game-managed (locked decision 4): the engine does NOT
//! filter point lights above the plane, so this scene culls them with
//! the same footprint rule the sprites use —
//! [`roxlap_scene::Scene::cutaway_hides_point`] — the pattern the book
//! documents.
//!
//! Controls: mouse orbits the ship · WASD pans the focus ·
//! `PgUp`/`PgDn` raise/cut decks · left-click picks on the cut.

use glam::{DVec3, IVec3};
use roxlap_render::{DirectionalLight, LightRig, Material, PointLight, SpriteModelId, SpriteSet};
use roxlap_scene::{GridId, GridTransform, Scene, VoxColor};
use winit::event::MouseButton;
use winit::keyboard::KeyCode;

use crate::build_sprites;
use crate::scene_api::{
    frame_params, opticast_settings, CameraPose, CameraRig, DemoScene, SceneCtx, SceneInput,
};

// ---- ship geometry (ship-grid-local, z-down) -------------------------
// pub(crate): the Boarding scene (stage OC) walks the same shiplet.

/// Hull XY extent (inclusive).
pub(crate) const HULL_X: (i32, i32) = (24, 103);
pub(crate) const HULL_Y: (i32, i32) = (32, 95);
/// Roof plate top (the shallowest hull voxel).
pub(crate) const ROOF_Z: i32 = 240;
/// Top of each deck's floor plate, top deck first. Floors are 4 voxels
/// thick; the last one is the hull bottom (321). The chz=0/chz=1 chunk
/// boundary (z = 256) falls INSIDE deck 1, so the hull genuinely spans
/// stacked chunks — the case the absolute-z clip formula exists for.
/// The bilge is 8 voxels TALLER than the upper decks (OC.3 visual
/// pass: with the 8-deep flood a swimming 12-voxel body floated with
/// its head a couple of voxels under the old ceiling).
pub(crate) const FLOOR_TOPS: [i32; 3] = [262, 286, 318];
/// The cut plane per deck view: `z_clip` = the first VISIBLE layer.
/// Index 0 = whole ship (no clip), 1..=3 = expose that deck.
pub(crate) const DECK_CLIPS: [Option<i32>; 4] = [None, Some(242), Some(266), Some(290)];
/// Ship grid world origin: hull bottom (local 321) lands flush on the
/// ground surface (world 419).
pub(crate) const SHIP_ORIGIN_Z: f64 = 98.0;
/// Ground surface (world z, z-down: the plate spans 420..440).
pub(crate) const GROUND_TOP: i32 = 420;

const HULL_RGB: VoxColor = VoxColor(0x80_62_66_70);
/// Per-deck floor colours (top → bottom): wood, steel, dark bilge.
const FLOOR_RGB: [VoxColor; 3] = [
    VoxColor(0x80_8A_5E_38),
    VoxColor(0x80_5C_6C_7C),
    VoxColor(0x80_46_46_56),
];
const GROUND_RGB: VoxColor = VoxColor(0x80_4A_5E_3C);
/// Flood water (bottom deck) — mapped to a Beer–Lambert volumetric
/// material in `enter`, WT-style.
pub(crate) const WATER_RGB: VoxColor = VoxColor(0x80_28_54_98);
pub(crate) const MAT_WATER: u8 = 1;

/// OC.3 — stairwell footprints (grid-local): one per deck pair, on
/// opposite room sides. 8 steps × 4 voxels along x, each dropping 3
/// (the 24-voxel deck spacing); a walking body with `step_up > 3`
/// climbs them back up.
const STAIR_X: (i32, i32) = (64, 96);
const STAIR_YS: [(i32, i32); 2] = [(38, 46), (82, 90)];

/// Build the Decks/Boarding world: a two-chunk ground plate + the
/// 3-deck shiplet (its own grid, stacked chz chunks) with the flooded
/// bilge. `with_stairs` additionally carves a stairwell through each
/// deck floor and builds the 3-voxel steps the Boarding character
/// walks — the plain Decks view keeps the sealed hull its captures pin.
pub(crate) fn build_ship_world(with_stairs: bool) -> (Scene, GridId) {
    let mut scene = Scene::new();

    // Ground: two chunks along x (world XY [0,256) × [0,128)) so
    // the beside-the-ship actor can stand OUTSIDE the ship grid's
    // chunk footprint — the footprint rule's "never affected" case.
    let ground = scene.add_grid(GridTransform::identity());
    scene.grid_mut(ground).unwrap().set_rect(
        IVec3::new(0, 0, GROUND_TOP),
        IVec3::new(255, 127, GROUND_TOP + 20),
        Some(GROUND_RGB),
    );

    // The shiplet, its own grid so the clip stays grid-local.
    let ship = scene.add_grid(GridTransform::at(DVec3::new(0.0, 0.0, SHIP_ORIGIN_Z)));
    {
        let g = scene.grid_mut(ship).unwrap();
        let (x0, x1) = HULL_X;
        let (y0, y1) = HULL_Y;
        // Roof plate.
        g.set_rect(
            IVec3::new(x0, y0, ROOF_Z),
            IVec3::new(x1, y1, ROOF_Z + 1),
            Some(HULL_RGB),
        );
        // Perimeter walls, roof to hull bottom.
        let zb = FLOOR_TOPS[2] + 3;
        for (wx0, wy0, wx1, wy1) in [
            (x0, y0, x1, y0 + 1),
            (x0, y1 - 1, x1, y1),
            (x0, y0, x0 + 1, y1),
            (x1 - 1, y0, x1, y1),
        ] {
            g.set_rect(
                IVec3::new(wx0, wy0, ROOF_Z),
                IVec3::new(wx1, wy1, zb),
                Some(HULL_RGB),
            );
        }
        // Deck floors (the last is the hull bottom).
        for (d, &ft) in FLOOR_TOPS.iter().enumerate() {
            g.set_rect(
                IVec3::new(x0, y0, ft),
                IVec3::new(x1, y1, ft + 3),
                Some(FLOOR_RGB[d]),
            );
        }
        // Hazard 6 — flood the bottom deck: ordinary water voxels in
        // the slabs (they clip with everything else) + the physics
        // volume for completeness. Placed BEFORE the crates so the
        // bilge crate overwrites its water cells and reads as a
        // submerged solid through the Beer–Lambert surface, not a
        // washed-away hole.
        g.set_rect(
            IVec3::new(x0 + 2, y0 + 2, FLOOR_TOPS[2] - 8),
            IVec3::new(x1 - 2, y1 - 2, FLOOR_TOPS[2] - 1),
            Some(WATER_RGB),
        );
        g.add_water_volume(
            IVec3::new(x0 + 2, y0 + 2, FLOOR_TOPS[2] - 8),
            IVec3::new(x1 - 2, y1 - 2, FLOOR_TOPS[2] - 1),
        );
        // A crate per deck so the interiors read as rooms. 12
        // voxels tall — 4 above the 8-deep flood, so the bilge one
        // pokes out of the water while its base reads through the
        // Beer–Lambert surface (it overwrote its water cells).
        for &ft in &FLOOR_TOPS {
            g.set_rect(
                IVec3::new(70, 58, ft - 12),
                IVec3::new(77, 65, ft - 1),
                Some(VoxColor(0x80_7A_5A_30)),
            );
        }
        // OC.3 — the Boarding stairwells: per deck pair, carve the
        // shaft through the UPPER deck's floor plate and build the
        // descending steps in the room below (solid columns down to
        // the lower plate, so water cells they cross are overwritten).
        // Steps are 4 voxels tall (`step_up > 4` climbs them back up),
        // so the deck drops (24 and 32) divide evenly.
        if with_stairs {
            for (d, &(sy0, sy1)) in STAIR_YS.iter().enumerate() {
                let plate = FLOOR_TOPS[d];
                let lower = FLOOR_TOPS[d + 1];
                let (sx0, _sx1) = STAIR_X;
                // `drop/4` steps of 4 voxels: step i's top at
                // plate + 4(i+1), filled down to the lower plate. The
                // LAST slot (i = n−1, top == lower) is the lower deck
                // floor itself — building it would be a degenerate
                // inverted z-rect that `set_rect` normalises into a
                // 1-voxel hull-grey sill over the floor (and over the
                // bilge flood) that the body trips on.
                let n = (lower - plate) / 4;
                // Shaft: the plate becomes air EXACTLY over the steps
                // (a longer hole would leave a full-deck drop past the
                // last step).
                g.set_rect(
                    IVec3::new(sx0, sy0, plate),
                    IVec3::new(sx0 + 4 * n - 1, sy1 - 1, plate + 3),
                    None,
                );
                for i in 0..n - 1 {
                    let top = plate + 4 * (i + 1);
                    g.set_rect(
                        IVec3::new(sx0 + 4 * i, sy0, top),
                        IVec3::new(sx0 + 4 * i + 3, sy1 - 1, lower - 1),
                        Some(HULL_RGB),
                    );
                }
            }
        }
    }
    (scene, ship)
}

/// The per-deck cabin lights (world coords), centred in each room —
/// shared by Decks and Boarding, culled game-side by the footprint
/// rule (locked decision 4: the game culls, not the engine).
pub(crate) fn cabin_lights() -> Vec<PointLight> {
    #[allow(clippy::cast_possible_truncation)]
    FLOOR_TOPS
        .iter()
        .zip([
            [1.0_f32, 0.78, 0.45], // warm top deck
            [0.45, 0.9, 1.0],      // cyan mid deck
            [0.35, 0.55, 1.0],     // blue bilge
        ])
        .map(|(&ft, color)| PointLight {
            position: [64.0, 56.0, (f64::from(ft) - 18.0 + SHIP_ORIGIN_Z) as f32],
            color,
            intensity: 3.5,
            radius: 70.0,
            casts_shadow: false,
        })
        .collect()
}

// ---- tele-iso camera --------------------------------------------------

/// Narrow vertical FOV (radians) — the "tele" in tele-iso.
const TELE_FOV_Y: f32 = 0.15;
/// Orbit distance: at 0.15 rad the frame spans ≈ `0.15 × dist` voxels
/// vertically, so 1150 shows the whole shiplet with margin.
const TELE_DIST: f64 = 1150.0;
/// Ray-march far enough to cover the tele distance plus the scene
/// (camera ≈ 1150 out + ~400 of world + orbit slack) — the GPU outer
/// step budget derives from this, so keep it tight.
const TELE_SCAN_DIST: i32 = 2048;
/// GPU LOD scan-distance override while the scene is active: chosen so
/// the PRIMARY rays still march mip 0 at the tele distance
/// (`1150 / 640 < 2` ⇒ level 0) while the sun-shadow marches — whose
/// mip is picked from `t_base + travel` (PF.2) — degrade to mip 1 a
/// short way from the receiver instead of walking hundreds of mip-0
/// voxels per lit pixel (the GPU shadow march has no brick-level
/// empty-space skip; measured 12 → 30+ FPS on NVK at 860×520).
const TELE_MIP_SCAN: f32 = 640.0;
const START_YAW: f64 = 0.7;
const START_PITCH: f64 = 0.85;
/// Focus pans at this speed (voxels/sec) under WASD.
const PAN_SPEED: f64 = 90.0;

pub struct DecksScene {
    scene: Scene,
    ship: GridId,
    /// Current deck view, indexing [`DECK_CLIPS`] (0 = whole ship).
    deck: usize,
    /// Orbit focus (world) — the ship's mid-height centre; WASD pans it.
    focus: [f64; 3],
    /// All four actors' world positions (3 on ship decks + 1 on the
    /// ground beside the hull), re-registered on every `enter`.
    actors: Vec<[f32; 3]>,
    actor_model: Option<SpriteModelId>,
    /// The per-deck cabin lights, world-space. Culled per frame by the
    /// footprint rule (decision 4: the game culls, not the engine).
    cabin_lights: Vec<PointLight>,
    /// Lights that survived this frame's cull (borrowed by `LightRig`).
    lit_points: Vec<PointLight>,
    /// The GPU LOD scan distance to restore on `exit` (the tele camera
    /// sits ~1150 voxels out — distance LOD would coarsen the whole
    /// ship to deep mips, so `enter` raises it past the working range).
    /// `Option` because `0.0` ("LOD off", e.g.
    /// `ROXLAP_GPU_MIP_SCAN_DIST=0`) is a legal value to restore.
    saved_mip_scan: Option<f32>,
}

impl DecksScene {
    #[must_use]
    pub fn new() -> Self {
        // The sealed hull (no stairwells) — the geometry the CA
        // captures pin; Boarding builds the walkable variant.
        let (scene, ship) = build_ship_world(false);

        // World-space actor spots: one per deck (standing on its floor,
        // inside the hull) + one on the ground beside the ship. The
        // deck actors hide with their deck (footprint rule); the ground
        // one never does.
        #[allow(clippy::cast_possible_truncation)]
        let actors: Vec<[f32; 3]> = FLOOR_TOPS
            .iter()
            .map(|&ft| [54.0, 56.0, (f64::from(ft) - 12.0 + SHIP_ORIGIN_Z) as f32])
            .chain(std::iter::once([150.0, 56.0, GROUND_TOP as f32 - 12.0]))
            .collect();

        let cabin_lights = cabin_lights();

        // BK.7-style capture hook: `ROXLAP_DECK=0..=3` starts on that
        // deck view (gallery shots of the cut without keyboard input).
        let deck = std::env::var("ROXLAP_DECK")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .map_or(0, |d| d.min(DECK_CLIPS.len() - 1));
        let mut this = Self {
            scene,
            ship,
            deck: 0,
            focus: [64.0, 64.0, f64::from(ROOF_Z + 40) + SHIP_ORIGIN_Z],
            actors,
            actor_model: None,
            cabin_lights,
            lit_points: Vec::new(),
            saved_mip_scan: None,
        };
        if deck != 0 {
            this.set_deck(deck);
        }
        this
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
        eprintln!("decks: viewing {}", self.deck_label());
    }
}

impl DemoScene for DecksScene {
    fn name(&self) -> &'static str {
        "Decks"
    }

    fn controls(&self) -> &'static str {
        "PgUp/PgDn cut decks · mouse orbits · WASD pans · left-click picks on the cut"
    }

    fn start_pose(&self) -> CameraPose {
        let rig = CameraRig::tele_iso(
            [64.0, 64.0, f64::from(ROOF_Z + 40) + SHIP_ORIGIN_Z],
            START_YAW,
            START_PITCH,
            TELE_DIST,
        );
        CameraPose {
            pos: rig.pos,
            yaw: rig.yaw,
            pitch: rig.pitch,
        }
    }

    fn enter(&mut self, ctx: &mut SceneCtx) {
        // GPU distance LOD would march the whole ship at deep mips from
        // the tele distance; hold mip 0 out past the orbit range.
        self.saved_mip_scan = Some(ctx.renderer.gpu_mip_scan_dist());
        ctx.renderer.set_gpu_mip_scan_dist(TELE_MIP_SCAN);

        // The flood water is a Beer–Lambert volumetric material (WT
        // pattern) so the submerged bilge reads through the surface.
        ctx.renderer
            .define_material(MAT_WATER, Material::volumetric(30));
        ctx.renderer
            .set_terrain_materials(&[(WATER_RGB.rgb_part(), MAT_WATER)]);

        // Actors: the shared coco sprite, one instance per spot.
        let models = ctx.renderer.set_sprites(&SpriteSet {
            models: build_sprites().first().cloned().into_iter().collect(),
            instances: Vec::new(),
            carve_model: None,
        });
        self.actor_model = models.first().copied();
        if let Some(m) = self.actor_model {
            for p in &self.actors {
                ctx.renderer.add_sprite_instance(m, *p);
            }
        }
        // Restore this visit's deck cut (scene state survives switches).
        self.scene.set_grid_z_clip(self.ship, DECK_CLIPS[self.deck]);
    }

    fn update(&mut self, ctx: &mut SceneCtx, dt: f64) {
        // WASD pans the focus in the ground plane, camera-relative.
        let cam = ctx.cam.camera();
        let (mut px, mut py) = (0.0_f64, 0.0_f64);
        // Screen-forward projected to XY; right is already horizontal.
        let fl = (cam.forward[0] * cam.forward[0] + cam.forward[1] * cam.forward[1]).sqrt();
        if fl > 1e-6 {
            let (fx, fy) = (cam.forward[0] / fl, cam.forward[1] / fl);
            if ctx.input.forward {
                px += fx;
                py += fy;
            }
            if ctx.input.back {
                px -= fx;
                py -= fy;
            }
        }
        if ctx.input.right {
            px += cam.right[0];
            py += cam.right[1];
        }
        if ctx.input.left {
            px -= cam.right[0];
            py -= cam.right[1];
        }
        self.focus[0] += px * PAN_SPEED * dt;
        self.focus[1] += py * PAN_SPEED * dt;

        // Tele-iso orbit: re-derive the position from the CURRENT
        // yaw/pitch (mouse-look) so the camera circles the focus.
        let rig = CameraRig::tele_iso(self.focus, ctx.cam.yaw, ctx.cam.pitch, TELE_DIST);
        ctx.cam.pos = rig.pos;

        // Decision 4 — the light-cull pattern: the engine does not
        // filter lights above the cut, so cull them with the SAME
        // footprint rule the sprites use. A lamp on a hidden deck
        // would otherwise keep lighting the exposed deck below.
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
                let d = (self.deck + 1).min(DECK_CLIPS.len() - 1);
                self.set_deck(d);
            }
            SceneInput::Key {
                code: KeyCode::PageUp,
                pressed: true,
            } => {
                self.set_deck(self.deck.saturating_sub(1));
            }
            SceneInput::Mouse {
                button: MouseButton::Left,
                pressed: true,
            } => {
                // CA.4 — click-to-select on the cut: the clip-aware
                // raycast lands on what the cut render shows, not on
                // the hidden hull above it.
                let cam = ctx.cam.camera();
                let o = DVec3::from_array(cam.pos);
                let d = DVec3::from_array(cam.forward);
                match self.scene.raycast_clipped(o, d, 8192.0) {
                    Some(h) => eprintln!(
                        "decks: centre pick → grid #{} voxel ({}, {}, {}) at t={:.1}",
                        h.grid.raw(),
                        h.voxel.x,
                        h.voxel.y,
                        h.voxel.z,
                        h.t,
                    ),
                    None => eprintln!("decks: centre pick → sky"),
                }
            }
            _ => {}
        }
    }

    fn render(&mut self, ctx: &mut SceneCtx) {
        // Narrow tele FOV via the core helper (both backends derive
        // their projection from these settings).
        let settings = opticast_settings(ctx.size, TELE_SCAN_DIST).with_fov_y(TELE_FOV_Y);
        let mut frame = frame_params(ctx.engine, &settings, TELE_SCAN_DIST);
        // A shadowing sun proves CA.2/CA.3 live: with the roof cut
        // away the interior is sun-lit, not shadowed by hidden decks.
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
            // Just enough ray to cover the hull: the farthest real
            // occluder (ground-tip → roof edge along the sun) is ~90
            // world units out — every extra unit is marched per lit pixel.
            shadow_max_dist: 110.0,
            bands: 0,
            shadow_tint: [0.0; 3],
        });
        let camera = ctx.cam.camera();
        ctx.renderer.render(&mut self.scene, &camera, &frame);
    }

    fn exit(&mut self, ctx: &mut SceneCtx) {
        // Restore the entry-time GPU LOD distance for the other scenes
        // (their cameras sit inside the world). `0.0` = "LOD off" is a
        // legal saved value, hence the Option, not a magic zero.
        if let Some(d) = self.saved_mip_scan.take() {
            ctx.renderer.set_gpu_mip_scan_dist(d);
        }
    }

    fn hud_lines(&self) -> Vec<String> {
        vec![
            format!("deck view: {}", self.deck_label()),
            format!(
                "cabin lights lit: {}/{} (hidden decks culled game-side)",
                self.lit_points.len(),
                self.cabin_lights.len()
            ),
        ]
    }
}
