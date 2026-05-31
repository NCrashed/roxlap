//! Scene assembly entry point — builds the demo `Scene` and
//! returns it along with a sensible starting camera.
//!
//! S3.x: 2 single-chunk axis-aligned grids — ground patch + ship.
//! Both grid origins are picked so the camera can frame both
//! without OOB-camera weirdness.

use glam::{DQuat, DVec3, IVec3};
use roxlap_core::Camera;
use roxlap_scene::{Grid, GridId, GridTransform, Scene, CHUNK_SIZE_XY, CHUNK_SIZE_Z};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Angular velocity (rad/s) of the ship grid's Z-axis spin when
/// `R` toggles spin on. Z is voxlap's vertical (down) axis — this
/// is the "UFO spinning on its own axis" yaw. Picked low enough to
/// read as "slow majestic spin" rather than disorienting blur —
/// ~21 s per full revolution.
pub const SHIP_SPIN_RATE_Z: f64 = 0.3;

/// Angular velocity (rad/s) of the X-axis + Y-axis spins. Half the
/// Z rate so the ship tumbles slowly enough to read each pose
/// distinctly. The X+Y rotations are the real voxel-raycasting
/// stress test: non-axis-aligned rays through a chunk grid that's
/// continuously tilting on its horizontal axes exercise paths
/// (mip-N column-step, cross-chunk DDA, cf-narrowing at unusual
/// orientations) that the Z-only spin doesn't touch.
pub const SHIP_SPIN_RATE_XY: f64 = 0.15;

use crate::{markers, ship, terrain};

/// Camera basis for "yaw=0, pitch=0 looks +x, voxlap-z down".
/// Same convention as the existing `roxlap-host` demo.
fn camera_for_yaw_pitch(pos: [f64; 3], yaw: f64, pitch: f64) -> Camera {
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    Camera {
        pos,
        right: [-sy, cy, 0.0],
        down: [-cy * sp, -sy * sp, cp],
        forward: [cy * cp, sy * cp, sp],
    }
}

/// Build the demo scene + initial camera state.
///
/// Layout (world coords, voxlap z-down: smaller z = up):
/// - **Ground** grid origin at world `(0, 0, 0)` — terrain spans
///   `GROUND_CHUNKS_X × GROUND_CHUNKS_Y` chunks centred on the grid
///   origin (S4.1: chunk-XY `[-16..16) × [-16..16)`, world XY
///   `[-2048..2048)`, z ∈ [80..255]).
/// - **Ship** grid origin at world `(0, 500, -100)` — saucer
///   spanning `SHIP_CHUNKS_X × SHIP_CHUNKS_Y` chunks centred on the
///   ship grid origin (S4.2: chunk-XY `[-2..2) × [-3..3)`, grid-local
///   `[-256..256) × [-384..384) × [0..256)`). Body equator at
///   grid-local z=64 → world z=-36 (same altitude as the S3.x ship).
///   The +y origin offset puts the saucer visibly ahead of the
///   spawn camera instead of engulfing it.
///
/// Initial camera at world `(0, -120, 50)` (over the south-edge
/// midline of the centred ground) looking +y, sees the saucer
/// floating ahead and slightly above the horizon with the centred
/// terrain stretching to the horizon.
pub fn build_demo() -> SceneAndCamera {
    let mut scene = Scene::new();

    let ground_id = scene.add_grid(GridTransform::at(DVec3::new(0.0, 0.0, 0.0)));
    // S4B.6.f: `ROXLAP_STACKED_GROUND=1` switches the ground to a
    // 2-chunk-tall stack (chz=0 all-air, chz=1 hilly terrain). The
    // camera spawns in chz=0 air-gap and uses S4B.6.e's
    // cross-chunk look-down (seed-time) to see chz=1 terrain.
    // Static path (unset = default for the legacy regression tests
    // via `ROXLAP_STATIC=1`); S7.6 path replaces it with streaming
    // hills below.
    let stacked_ground = std::env::var("ROXLAP_STACKED_GROUND").is_ok();
    // S7.6: streaming hills by default. `ROXLAP_STATIC=1` falls
    // back to the historical statically-built 32×32 ground
    // (needed for repro tests + visual A/B against the streaming
    // variant).
    let use_static_ground = std::env::var("ROXLAP_STATIC").is_ok() || stacked_ground;
    if stacked_ground {
        terrain::build_ground_stacked(scene.grid_mut(ground_id).expect("ground grid present"));
    } else if use_static_ground {
        terrain::build_ground(scene.grid_mut(ground_id).expect("ground grid present"));
    } else {
        // Streaming variant — attach the generator, set a small
        // `r_active` so chunks visibly load + unload as the camera
        // moves. r_active = 256 keeps `chunks_z ≤ 3` at the spawn
        // camera z = 50 (ducks the
        // [[s7-4-landed]] opticast gylookup overflow) and gives a
        // ~2-chunk visible radius around the camera. r_evict =
        // 384 is the usual 1.5× hysteresis band.
        let g = scene.grid_mut(ground_id).expect("ground grid present");
        g.set_generator(Some(Arc::new(terrain::HillsChunkGenerator)));
        g.stream_radius = roxlap_scene::StreamRadius::new(256.0, 384.0);
        // No initial pump — let the user see chunks stream in over
        // the first ~1 second as the rayon pool fills the active
        // ball. Visually demonstrates the streaming pipeline from
        // frame 0.
    }

    let ship_id = scene.add_grid(GridTransform::at(DVec3::new(0.0, 500.0, -100.0)));
    {
        let ship = scene.grid_mut(ship_id).expect("ship grid present");
        ship::build_ship(ship);
        // S5.2-followup: with the ship rotating each frame, its
        // grid-local sky lookup rotates with it. Leaving the ship's
        // sky on lets per-pixel min-z noise between the ground's
        // sky-z and the ship's sky-z (which differ because the ray
        // basis differs once rotation is applied) allow some of the
        // ship's rotated sky panorama to bleed into the composed
        // framebuffer. Disabling sky for the ship grid masks those
        // pixels via [`render::SKY_MASK_SENTINEL`] so only the
        // ground grid contributes the sky panorama.
        ship.render_sky = false;
        // S5.2-followup #2 (fake-column artifact). The
        // axis-aligned-mip-beams artifact (multi-mip + near-axis-
        // aligned ray + cf-cancellation, see
        // `project_axis_aligned_mip_beams.md`) is dramatically
        // worse for the rotating ship grid: each rotation
        // orientation creates new ray-vs-axis configurations that
        // can trigger the cf-cancellation. The proper fix is
        // cf-narrowing at remiporend (deferred engine work). For
        // the ship — which is small enough (~500-voxel diameter)
        // that multi-mip provides no perceptible LOD benefit at
        // the demo's camera distances — cap at mip-0 to side-step
        // the bug. Identity-rotation render stays clean; verified
        // by `ship_fake_column_glitch_diag::try_single_mip=true`.
        ship.mip_levels_override = Some(1);
    }

    // Bake lightmode-1 directional shading into every chunk's slab
    // alpha bytes. Pure surface-normal-based gradient — voxlap's
    // `(tp.y * 0.5 + tp.z) * 64 + 103.5` formula. Done once at
    // scene-build time; the renderer reads the baked values via
    // hrend / vrend without needing further setup.
    //
    // Done BEFORE marker grids are added so the marker chunks
    // (built via fragmented `set_rect` calls — see
    // `markers::build_one_marker`) don't pass through
    // `generate_mips` inside the bake. That call hits the
    // [[mip_attempt]] index-out-of-bounds for fragmented chunks.
    // Markers stay unlit (flat colour stripes) which is fine for
    // a LOD-tier visual validation demo.
    bake_lightmode_1(&mut scene);

    // S6.6: 10 striped marker pillars along world +y. Built AFTER
    // `bake_lightmode_1` so the markers skip the bake's
    // generate_mips pass (see above). The `B` hotkey toggles their
    // `lod_thresholds` between always-Near (default — pre-S6.6
    // behaviour) and the tuned Near/Far config.
    let marker_ids = markers::build_markers(&mut scene);

    // For the stacked-ground variant the terrain sits in chz=1
    // (world z=256..511). Spawn deeper into chz=0's air-gap (z=200,
    // = 56 voxels above chz=1's top) and pitch down so the
    // cross-chunk look-down hits the terrain in the default view.
    let (initial_pos, initial_pitch) = if stacked_ground {
        ([0.0, -120.0, 200.0], -0.35)
    } else {
        ([0.0, -120.0, 50.0], 0.0)
    };
    let initial_yaw = std::f64::consts::FRAC_PI_2; // looks +y
    let camera = camera_for_yaw_pitch(initial_pos, initial_yaw, initial_pitch);

    SceneAndCamera {
        scene,
        camera,
        cam_pos: initial_pos,
        yaw: initial_yaw,
        pitch: initial_pitch,
        ship_id,
        ship_angles: [0.0; 3],
        spin_enabled: false,
        marker_ids,
        lod_billboards_on: false,
        // Streaming on whenever the ground grid uses the generator
        // path; static (`ROXLAP_STATIC=1`) keeps it off.
        streaming_enabled: !use_static_ground,
    }
}

/// What [`build_demo`] returns. The camera state is split into
/// `(pos, yaw, pitch)` plus the materialised `Camera` so the
/// app can mutate `pos` / `yaw` / `pitch` in response to input
/// and reconstruct the basis each frame.
///
/// S5.2: `ship_id` + `ship_angle` + `spin_enabled` drive the
/// per-frame ship rotation. Toggling `spin_enabled` (R hotkey)
/// makes the saucer slowly rotate about its own Z axis via
/// [`tick_ship_spin`].
pub struct SceneAndCamera {
    pub scene: Scene,
    pub camera: Camera,
    pub cam_pos: [f64; 3],
    pub yaw: f64,
    pub pitch: f64,
    /// Grid id of the ship — held so the per-frame spin tick can
    /// update its [`GridTransform::rotation`] without re-iterating
    /// the scene.
    pub ship_id: GridId,
    /// Current ship rotation angles around grid-local X / Y / Z
    /// (radians). Tracked as f64 so the quaternion is rebuilt from
    /// absolute angles each frame, avoiding the slow drift that
    /// an incremental quat multiply would accumulate. Order:
    /// `[x, y, z]`.
    pub ship_angles: [f64; 3],
    /// `R` toggles this; `false` freezes the ship at its current
    /// `ship_angles` (lighting bake stays as-is — see S5.3).
    pub spin_enabled: bool,
    /// S6.6: grid ids of the 10 LOD marker pillars (closest first).
    /// Held so the `B` hotkey can flip their `lod_thresholds`
    /// without re-iterating the scene.
    pub marker_ids: Vec<GridId>,
    /// S6.6: `B` toggles this. `false` ⇒ marker grids render as
    /// `Lod::Near` (full voxel — pre-S6.6 baseline); `true` ⇒
    /// distance-keyed Near/Far split via the tuned thresholds in
    /// [`crate::markers`].
    pub lod_billboards_on: bool,
    /// S7.6: streaming mode. `true` when [`build_demo`] used the
    /// `HillsChunkGenerator` streaming path (the default); `false`
    /// when the historical static ground was selected via
    /// `ROXLAP_STATIC=1` or `ROXLAP_STACKED_GROUND=1`. Drives the
    /// per-frame [`Scene::pump_streaming`] call + the `T`
    /// telemetry hotkey in `main.rs`.
    pub streaming_enabled: bool,
}

impl SceneAndCamera {
    /// Recompute the camera basis from the current `(pos, yaw,
    /// pitch)` state. Call after any input that touches them.
    pub fn refresh_camera(&mut self) {
        self.camera = camera_for_yaw_pitch(self.cam_pos, self.yaw, self.pitch);
    }

    /// S5.2: advance the ship grid's rotation by `dt` seconds.
    /// No-op when `spin_enabled` is `false`. Rebuilds the
    /// quaternion from the absolute angles each frame so f64
    /// arithmetic stays the source of truth — incremental quat
    /// multiplication would drift over thousands of frames.
    ///
    /// Three independent axes:
    /// - X: `SHIP_SPIN_RATE_XY` rad/s — tilts the saucer
    ///   forward/back.
    /// - Y: `SHIP_SPIN_RATE_XY` rad/s — rolls the saucer
    ///   side-to-side.
    /// - Z: `SHIP_SPIN_RATE_Z` rad/s — the "UFO yaw" (vertical
    ///   axis, voxlap z-down).
    ///
    /// X and Y at half the Z rate gives a slow tumbling motion
    /// that's the real voxel-raycasting correctness stress: each
    /// frame the rasterizer sees rays through a chunk grid at a
    /// non-axis-aligned, continuously-changing orientation.
    /// Composition order: `R_z · R_y · R_x` (X applied first to
    /// any grid-local vector, then Y, then Z).
    /// S6.6: flip the marker grids' [`LodThresholds`] between
    /// the always-Near default and the tuned billboards config.
    /// Toggles [`Self::lod_billboards_on`] and returns the new
    /// value so the caller can echo it to the console.
    pub fn toggle_billboards_lod(&mut self) -> bool {
        self.lod_billboards_on = !self.lod_billboards_on;
        markers::set_billboards_lod(&mut self.scene, &self.marker_ids, self.lod_billboards_on);
        self.lod_billboards_on
    }

    pub fn tick_ship_spin(&mut self, dt: f64) {
        if !self.spin_enabled {
            return;
        }
        let rates = [SHIP_SPIN_RATE_XY, SHIP_SPIN_RATE_XY, SHIP_SPIN_RATE_Z];
        let two_pi = std::f64::consts::TAU;
        for (angle, rate) in self.ship_angles.iter_mut().zip(rates.iter()) {
            *angle += rate * dt;
            // Wrap into [0, 2π) so the f64 stays bounded over a
            // long session. Visually identical.
            if *angle >= two_pi {
                *angle -= two_pi;
            }
        }
        let qx = DQuat::from_rotation_x(self.ship_angles[0]);
        let qy = DQuat::from_rotation_y(self.ship_angles[1]);
        let qz = DQuat::from_rotation_z(self.ship_angles[2]);
        let ship = self
            .scene
            .grid_mut(self.ship_id)
            .expect("ship grid registered");
        ship.transform.rotation = qz * qy * qx;
    }
}

/// Bake voxlap's lightmode-1 (sun-style directional) shading into
/// every chunk of every grid. Mode 1 uses surface normals only —
/// no `LightSrc` consulted — so we pass an empty `lights` slice.
///
/// **S4B.4.b chunk-aware bake.** For each populated chunk:
/// 1. Build an `EstNormCache` (in roxlap-core) with a closure
///    that resolves chunk-local `(px, py)` queries to whichever
///    chunk owns that position via `Grid::chunk(IVec3)`. The
///    closure spans `(-RAD..chunk_size+RAD)` so the 5×5×5
///    neighbourhood vote pulls from neighbouring chunks seamlessly.
/// 2. Mutably borrow the target chunk and call
///    `update_lighting_chunk` to write alpha bytes within the
///    chunk's footprint only.
///
/// Replaces the S4B.4.a combined-view materialisation. Same
/// chunk-edge-seam fix as before (cross-chunk estnorm reads); now
/// without stitching a giant flat buffer for the bake.
///
/// **Per-chunk bake region (not whole-grid).** A whole-grid
/// `update_lighting` at vsid=4096 would allocate a 500 MB+
/// `EstNormCache` bit table; the per-chunk loop keeps each cache
/// at ~135 KB (132²×8 bytes).
///
/// **S5.3: lighting is grid-local — sun rotates with the grid.**
/// `compute_brightness` in `roxlap-core::world_lighting` bakes a
/// hardcoded sun direction `(0, 0.5, 1)` (see
/// `(tp[1] * 0.5 + tp[2]) * 64 + 103.5`) into each voxel's alpha
/// byte. `tp` is the surface normal in the chunk's grid-local
/// frame, so the "sun" lives in **grid-local space**. When a grid
/// rotates (the S5 ship demo), the bake doesn't change; rays
/// project the rotated surface normals back to camera, so the lit
/// side appears to rotate with the grid — visually "the ship is
/// lit by a fixed local sun that turns with it".
///
/// This was a deliberate v1 decision per `PORTING-SCENE.md` § S5
/// (see `[[project_s5_3_landed]]`). The alternative — rotate the
/// world-space sun into grid-local at bake time AND re-bake on
/// rotation change — is a 0.3.x follow-up. Acceptable trade-off
/// because: (a) the per-chunk bake is non-trivial cost, (b) the
/// visual difference is minor for slowly-rotating scenes, and
/// (c) re-baking each frame would dominate the render budget.

/// Diagnostic re-export for `repro.rs` to bake lighting+mips into a
/// scene it builds itself (ring-artifact isolation tests). Gated
/// behind `cfg(test)` so the release binary doesn't carry the
/// dead-code entry point.
#[cfg(test)]
pub fn bake_lightmode_1_pub(scene: &mut Scene) {
    bake_lightmode_1(scene);
}

// chx_v / chy_v are voxlap-canonical paired names.
#[allow(clippy::cast_possible_wrap, clippy::similar_names)]
fn bake_lightmode_1(scene: &mut Scene) {
    const LIGHTMODE: u32 = 1;
    // S7.6: skip streaming grids — they bake themselves on
    // stream-in inside their `ChunkGenerator::generate`. A
    // scene-wide bake here would only catch the few chunks that
    // happened to be loaded at the moment of the call, leaving
    // subsequently-streamed chunks unlit.
    let ids: Vec<GridId> = scene
        .grids()
        .filter_map(|(id, grid)| {
            if grid.generator.is_some() {
                None
            } else {
                Some(id)
            }
        })
        .collect();
    for id in ids {
        let grid = scene.grid_mut(id).expect("grid present");
        let chunk_idxs: Vec<IVec3> = grid.chunks.keys().copied().collect();
        if chunk_idxs.is_empty() {
            continue;
        }
        let cs_xy = CHUNK_SIZE_XY as i32;
        let cs_z = CHUNK_SIZE_Z as i32;

        for chunk_idx in &chunk_idxs {
            let target_chx = chunk_idx.x;
            let target_chy = chunk_idx.y;
            let target_chz = chunk_idx.z;

            // Cache build (immutable grid borrow). The closure
            // resolves chunk-local `(px, py)` — which can extend
            // `±ESTNORMRAD` outside the target chunk — into the
            // neighbour chunk that owns that voxel-column. Padding
            // straddling unpopulated chunks returns None (= treat
            // as full air), matching the historical OOB behaviour.
            // S4B.6.f: reader queries `target_chz` so stacked grids
            // bake each chunk-z layer independently. Cross-chz
            // neighbour reads (ESTNORMRAD padding extending into
            // chz±1 at top/bottom of a chunk) still clip on the z
            // boundary — that's a follow-up.
            let cache = {
                let grid_ref: &Grid = &*grid;
                let reader = |px: i32, py: i32| -> Option<&[u8]> {
                    let neighbour_chx = target_chx + px.div_euclid(cs_xy);
                    let neighbour_chy = target_chy + py.div_euclid(cs_xy);
                    let in_chunk_x = px.rem_euclid(cs_xy);
                    let in_chunk_y = py.rem_euclid(cs_xy);
                    let chunk =
                        grid_ref.chunk(IVec3::new(neighbour_chx, neighbour_chy, target_chz))?;
                    let col_idx = (in_chunk_y as u32) * CHUNK_SIZE_XY + (in_chunk_x as u32);
                    let off = chunk.column_offset[col_idx as usize] as usize;
                    Some(&chunk.data[off..])
                };
                roxlap_core::EstNormCache::build_with_reader(reader, 0, 0, cs_xy, cs_xy)
            };
            // Immutable grid borrow released.

            // Mutable target-chunk borrow + write phase.
            let target_chunk = grid.chunks.get_mut(chunk_idx).expect("populated");
            roxlap_core::apply_lighting_with_cache(
                &mut target_chunk.data,
                &target_chunk.column_offset,
                CHUNK_SIZE_XY,
                0,
                0,
                0,
                cs_xy,
                cs_xy,
                cs_z,
                &cache,
                LIGHTMODE,
                &[],
            );
        }

        // S4B.5 (2026-05-12): generate per-chunk mips after the
        // lighting bake. 6 levels covers a 2048-voxel ray-depth
        // ladder when paired with `mip_scan_dist=64` (64·2⁵ = 2048),
        // matching the live demo's max scan distance. Build cost
        // and the chunk's grown mip tables are ~1.25× the mip-0
        // footprint per chunk — small vs the bake cost.
        for chunk_idx in &chunk_idxs {
            let chunk = grid.chunks.get_mut(chunk_idx).expect("populated");
            chunk.generate_mips(6);
        }
    }
}

/// Per-chunk lighting + mip bake driver for streaming grids.
///
/// The post-S7.6-hills patch: removed the in-isolation bake from
/// [`crate::terrain::HillsChunkGenerator::generate`] because the
/// generator runs on the streaming rayon pool with no access to
/// the live [`Grid`] — its estnorm reader had to return `None` for
/// every voxel past its chunk's own face, producing visible
/// chunk-edge brightness banding wherever two chunks met.
///
/// This tracker runs on the **main thread** right after
/// [`Scene::pump_streaming`] each frame and bakes lightmode-1
/// shading + regenerates mips for any chunks that arrived since
/// the last call. Critically it also re-bakes the four cardinal
/// neighbours of each newly-installed chunk so the seam between
/// "I was baked when my neighbour wasn't there" and "I now have a
/// neighbour" resolves immediately.
///
/// Steady state: no streaming → no rebakes. Camera moving slowly
/// → 1–5 newly-installed chunks per frame → 1–5 + 4 rebakes = ~5–25
/// per frame, ~7 ms each.
pub struct StreamingBakeTracker {
    /// Per-grid set of chunk indices whose alpha bytes have already
    /// been baked against the current neighbour set. Cleared on
    /// eviction (the tracker's `process` retains only currently-
    /// present indices).
    baked: HashMap<GridId, HashSet<IVec3>>,
}

impl StreamingBakeTracker {
    #[must_use]
    pub fn new() -> Self {
        Self {
            baked: HashMap::new(),
        }
    }

    /// Bake any chunks that streamed in since the last call + the
    /// loaded cardinal neighbours of each. Skips chunks at
    /// `chz != 0` — they're bedrock-only placeholders in the
    /// current demo (caves / hills both live in the chz=0 layer).
    pub fn process(&mut self, scene: &mut Scene) {
        let streaming_ids: Vec<GridId> = scene
            .grids()
            .filter_map(|(id, g)| {
                if g.generator.is_some() {
                    Some(id)
                } else {
                    None
                }
            })
            .collect();
        for id in streaming_ids {
            self.process_grid(scene, id);
        }
    }

    fn process_grid(&mut self, scene: &mut Scene, id: GridId) {
        // Snapshot the current chz=0 chunk set.
        let grid = scene.grid(id).expect("grid present");
        let current: HashSet<IVec3> = grid.chunks.keys().filter(|i| i.z == 0).copied().collect();

        let baked_set = self.baked.entry(id).or_default();
        // Drop evicted chunks from the tracker so re-streaming
        // re-bakes from scratch.
        baked_set.retain(|idx| current.contains(idx));

        let newly_installed: Vec<IVec3> = current
            .iter()
            .filter(|idx| !baked_set.contains(*idx))
            .copied()
            .collect();
        if newly_installed.is_empty() {
            return;
        }

        // Rebake set: newly installed + their 4 cardinal neighbours
        // that are also loaded. The neighbour rebake resolves the
        // pre-existing chunks' "I had no neighbour over there"
        // estnorm gradient now that a real neighbour is present.
        // Dedupe via the HashSet.
        let mut to_bake: HashSet<IVec3> = HashSet::new();
        for &idx in &newly_installed {
            to_bake.insert(idx);
            for delta in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)] {
                let n = IVec3::new(idx.x + delta.0, idx.y + delta.1, idx.z + delta.2);
                if current.contains(&n) {
                    to_bake.insert(n);
                }
            }
        }

        let grid = scene.grid_mut(id).expect("grid present");
        for &target_idx in &to_bake {
            bake_single_chunk_neighbour_aware(grid, target_idx);
            // Regenerate mips since alpha bytes (the brightness
            // byte) drive the mip lookup tables.
            let target = grid.chunks.get_mut(&target_idx).expect("populated");
            remip_post_edit(target, 4);
            baked_set.insert(target_idx);
        }
    }
}

/// `Vxl::generate_mips` wrapper that handles the stale-sentinel bug
/// hit when re-mipping an already-edited chunk.
///
/// **Bug.** `Vxl::generate_mips` calls `reset_to_single_mip` first,
/// which slices `self.data[..column_offset[n_cols]]` to drop any
/// previously-built mip-N. That sentinel `column_offset[n_cols]`
/// is set at chunk creation to the initial seed-data length
/// (= `vsid² × 8 = 131072` for our chunks) and is **never bumped**
/// when `voxalloc` scatters columns into the edit pool past it.
/// On the second `generate_mips` call against a post-edit chunk,
/// the slice destroys all real column data → subsequent column
/// reads OOB-panic.
///
/// **Fix.** Recompute "end of mip-0 data" before each rebuild by
/// walking columns + summing `slng` lengths, then patch the
/// sentinel. After this, `reset_to_single_mip`'s truncation slices
/// to a value that preserves all post-edit column data.
fn remip_post_edit(vxl: &mut roxlap_formats::vxl::Vxl, max_mips: u32) {
    let n_cols = (vxl.vsid as usize) * (vxl.vsid as usize);
    let mut max_end: u32 = 0;
    for i in 0..n_cols {
        let start = vxl.column_offset[i] as usize;
        let len_bytes = roxlap_formats::vxl::slng(&vxl.data[start..]);
        max_end = max_end.max(u32::try_from(start + len_bytes).expect("end fits in u32"));
    }
    if vxl.column_offset[n_cols] != max_end {
        let mut new_offsets = vxl.column_offset.to_vec();
        new_offsets[n_cols] = max_end;
        vxl.column_offset = new_offsets.into_boxed_slice();
    }
    vxl.generate_mips(max_mips);
}

impl Default for StreamingBakeTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Run lightmode-1 directional bake on `chunk_idx` of `grid` using a
/// reader that resolves to neighbour chunks via [`Grid::chunk`].
///
/// Same shape as the inner loop body of [`bake_lightmode_1`]; pulled
/// out so [`StreamingBakeTracker`] can call it per-install. Unlike
/// the in-generator bake this previously replaced, queries past the
/// target chunk's own faces resolve to the actual neighbour's data
/// (when it's loaded), so estnorm at chunk boundaries is consistent
/// with what the neighbour will compute.
#[allow(clippy::cast_possible_wrap)]
fn bake_single_chunk_neighbour_aware(grid: &mut Grid, chunk_idx: IVec3) {
    const LIGHTMODE: u32 = 1;
    let cs_xy = CHUNK_SIZE_XY as i32;
    let cs_z = CHUNK_SIZE_Z as i32;
    let target_chx = chunk_idx.x;
    let target_chy = chunk_idx.y;
    let target_chz = chunk_idx.z;

    let cache = {
        let grid_ref: &Grid = &*grid;
        let reader = |px: i32, py: i32| -> Option<&[u8]> {
            let neighbour_chx = target_chx + px.div_euclid(cs_xy);
            let neighbour_chy = target_chy + py.div_euclid(cs_xy);
            let in_chunk_x = px.rem_euclid(cs_xy);
            let in_chunk_y = py.rem_euclid(cs_xy);
            let chunk = grid_ref.chunk(IVec3::new(neighbour_chx, neighbour_chy, target_chz))?;
            let col_idx = (in_chunk_y as u32) * CHUNK_SIZE_XY + (in_chunk_x as u32);
            let off = chunk.column_offset[col_idx as usize] as usize;
            Some(&chunk.data[off..])
        };
        roxlap_core::EstNormCache::build_with_reader(reader, 0, 0, cs_xy, cs_xy)
    };

    let target = grid
        .chunks
        .get_mut(&chunk_idx)
        .expect("target chunk populated");
    roxlap_core::apply_lighting_with_cache(
        &mut target.data,
        &target.column_offset,
        CHUNK_SIZE_XY,
        0,
        0,
        0,
        cs_xy,
        cs_xy,
        cs_z,
        &cache,
        LIGHTMODE,
        &[],
    );
}

#[cfg(test)]
mod tests_s7_6 {
    use super::*;
    use crate::terrain::HillsChunkGenerator;
    use roxlap_scene::ChunkGenerator;

    /// Regression test for the `reset_to_single_mip` stale-sentinel
    /// bug (panic at vxl.rs:599 with index = 131073, len = 131072
    /// on the second `generate_mips`). Simulates the chunk lifecycle
    /// the `StreamingBakeTracker` exercises:
    ///
    /// 1. Generate one hills chunk (post-edit, no mips).
    /// 2. First mip build (works — `reset_to_single_mip` early-
    ///    returns because there's nothing past mip-0 yet).
    /// 3. Second mip build (would panic without `remip_post_edit`
    ///    because the stale sentinel would truncate the data
    ///    buffer back to its pre-edit-pool footprint).
    ///
    /// The test passes iff no panic + the chunk has 4 mip levels
    /// after both rebuilds.
    #[test]
    fn remip_post_edit_handles_second_generate_mips_call() {
        let mut vxl = HillsChunkGenerator.generate(IVec3::ZERO);
        // First mip build — must succeed.
        remip_post_edit(&mut vxl, 4);
        assert_eq!(vxl.mip_count(), 4, "first mip build produced 4 mips");
        // Second mip build (the path the bug fired on). Must not
        // panic + must still produce 4 mips.
        remip_post_edit(&mut vxl, 4);
        assert_eq!(vxl.mip_count(), 4, "second mip build still produces 4 mips");
        // Third for good measure — confirms no incremental damage.
        remip_post_edit(&mut vxl, 4);
        assert_eq!(vxl.mip_count(), 4, "third mip build OK");
    }

    /// S7.6 entry-point smoke test: `build_demo` (now the streaming
    /// hills variant by default) returns a scene with the streaming
    /// flag set, the ground grid has a generator + non-disabled
    /// stream_radius, and ship + 10 markers are still registered.
    #[test]
    fn build_demo_streaming_default_has_generator_on_ground_grid() {
        let s = build_demo();
        assert!(
            s.streaming_enabled,
            "streaming is the default for build_demo"
        );
        // Ground (raw=0) + ship (raw=1) + `markers::NUM_MARKERS`
        // pillars.
        assert_eq!(s.scene.grid_count(), 1 + 1 + crate::markers::NUM_MARKERS);
        let ground = s
            .scene
            .grids()
            .min_by_key(|(id, _)| id.raw())
            .map(|(_, g)| g)
            .expect("at least one grid");
        assert!(ground.generator.is_some(), "ground grid must be streaming");
        assert!(
            !ground.stream_radius.is_disabled(),
            "stream_radius must be active"
        );
    }

    /// S7.6 bounded-memory smoke test: pump many frames with a
    /// fixed camera; the streaming grid's chunk count must NOT
    /// grow without bound. The stream_radius from `build_demo`
    /// (256 / 384) caps the materialised set to chunks within a
    /// 384-voxel ball — ~25-50 chunks at the demo spawn.
    #[test]
    fn streaming_demo_chunk_count_stays_bounded_under_repeated_pump() {
        let mut s = build_demo();
        let cam = DVec3::from_array(s.cam_pos);
        // 50 pumps gives the async pool ample time to drain on any
        // reasonable CI host. After this, every chunk inside
        // r_active is either present or in-flight; nothing further
        // gets dispatched.
        for _ in 0..50 {
            s.scene.pump_streaming(cam);
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        // Find the streaming grid (lowest GridId by add order).
        let ground_id = s
            .scene
            .grids()
            .map(|(id, _)| id)
            .min_by_key(|id| id.raw())
            .expect("ground grid registered");
        let grid = s.scene.grid(ground_id).expect("ground grid");
        // Bound derived from `r_evict = 600`: chunks within
        // 600 voxels of the camera, where each chunk is
        // 128x128x256. Worst case (cube approximation):
        // (2 * (600/128) + 2)^2 * (2 * (600/256) + 2) ≈ 12^2 * 6 = 864.
        // In practice the spherical filter keeps it much lower; the
        // test just guards against "infinite growth" regressions.
        assert!(
            grid.chunk_count() < 2000,
            "chunk count unbounded: {}",
            grid.chunk_count()
        );
    }
}
