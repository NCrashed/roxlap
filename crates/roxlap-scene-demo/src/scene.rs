//! Scene assembly entry point — builds the demo `Scene` and
//! returns it along with a sensible starting camera.
//!
//! S3.x: 2 single-chunk axis-aligned grids — ground patch + ship.
//! Both grid origins are picked so the camera can frame both
//! without OOB-camera weirdness.

use glam::{DVec3, IVec3};
use roxlap_core::Camera;
use roxlap_scene::{GridId, GridTransform, Scene, CHUNK_SIZE_XY, CHUNK_SIZE_Z};

use crate::{ship, terrain};

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
    terrain::build_ground(scene.grid_mut(ground_id).expect("ground grid present"));

    let ship_id = scene.add_grid(GridTransform::at(DVec3::new(0.0, 500.0, -100.0)));
    ship::build_ship(scene.grid_mut(ship_id).expect("ship grid present"));

    // Bake lightmode-1 directional shading into every chunk's slab
    // alpha bytes. Pure surface-normal-based gradient — voxlap's
    // `(tp.y * 0.5 + tp.z) * 64 + 103.5` formula. Done once at
    // scene-build time; the renderer reads the baked values via
    // hrend / vrend without needing further setup.
    bake_lightmode_1(&mut scene);

    let initial_pos = [0.0, -120.0, 50.0];
    let initial_yaw = std::f64::consts::FRAC_PI_2; // looks +y
    let initial_pitch = 0.0;
    let camera = camera_for_yaw_pitch(initial_pos, initial_yaw, initial_pitch);

    SceneAndCamera {
        scene,
        camera,
        cam_pos: initial_pos,
        yaw: initial_yaw,
        pitch: initial_pitch,
    }
}

/// What [`build_demo`] returns. The camera state is split into
/// `(pos, yaw, pitch)` plus the materialised `Camera` so the
/// app can mutate `pos` / `yaw` / `pitch` in response to input
/// and reconstruct the basis each frame.
pub struct SceneAndCamera {
    pub scene: Scene,
    pub camera: Camera,
    pub cam_pos: [f64; 3],
    pub yaw: f64,
    pub pitch: f64,
}

impl SceneAndCamera {
    /// Recompute the camera basis from the current `(pos, yaw,
    /// pitch)` state. Call after any input that touches them.
    pub fn refresh_camera(&mut self) {
        self.camera = camera_for_yaw_pitch(self.cam_pos, self.yaw, self.pitch);
    }
}

/// Bake voxlap's lightmode-1 (sun-style directional) shading into
/// every chunk of every grid. Mode 1 uses surface normals only —
/// no `LightSrc` consulted — so we pass an empty `lights` slice.
///
/// **S4.1 fix for the chunk-edge lighting seam** — preserved
/// through S4B.4.a's combined-view retirement. Per-chunk bakes
/// at S4.0 produced a visible brightness jump at every chunk
/// boundary because `estnorm`'s 5×5×5 neighbourhood vote treated
/// neighbour chunks as all-air. The fix routes the bake through
/// a freshly-built `CombinedGridView`: each chunk's bake region
/// is still `(chunk_x0..chunk_x1) × (chunk_y0..chunk_y1)`, but the
/// `(world_data, column_offsets, vsid)` triple is the combined
/// view — so `EstNormCache`'s padding reads neighbour-chunk
/// voxels naturally. After all chunks bake, we
/// [`CombinedGridView::sync_alpha_to_chunks`] to copy the post-
/// bake alpha bytes back into source chunks; then drop the
/// combined view (no caching on `Grid` post-S4B.4.a).
///
/// S4B.4.b plan: replace the combined-view materialisation with
/// a chunk-aware `EstNormCache::build` reader, removing the
/// last `CombinedGridView` user.
///
/// **Per-chunk bake region (not whole-grid).** A whole-grid
/// `update_lighting` call at vsid=4096 would allocate a 500 MB+
/// `EstNormCache` bit table; the per-chunk loop keeps each cache
/// at ~135 KB (132²×8 bytes) and still gets correct cross-chunk
/// neighbourhood sampling.

// chx_v / chy_v are voxlap-canonical paired names.
#[allow(clippy::cast_possible_wrap, clippy::similar_names)]
fn bake_lightmode_1(scene: &mut Scene) {
    use roxlap_scene::CombinedGridView;

    const LIGHTMODE: u32 = 1;
    let ids: Vec<GridId> = scene.grids().map(|(id, _)| id).collect();
    for id in ids {
        let grid = scene.grid_mut(id).expect("grid present");
        let chunk_idxs: Vec<IVec3> = grid.chunks.keys().copied().collect();
        if chunk_idxs.is_empty() {
            continue;
        }
        let cs_xy = CHUNK_SIZE_XY as i32;
        let cs_z = CHUNK_SIZE_Z as i32;

        // S4B.4.a: build a local combined view per grid for the
        // duration of the bake. Drops once the alpha bytes are
        // propagated back.
        let mut combined = CombinedGridView::build(&grid.chunks);
        let origin_chunk = combined.origin_chunk;
        for chunk_idx in &chunk_idxs {
            if chunk_idx.z != 0 {
                // S4.0 combined-view scope: chz=0 only.
                continue;
            }
            let chx_v = chunk_idx.x - origin_chunk.x;
            let chy_v = chunk_idx.y - origin_chunk.y;
            let x0 = chx_v * cs_xy;
            let y0 = chy_v * cs_xy;
            let x1 = x0 + cs_xy;
            let y1 = y0 + cs_xy;
            roxlap_core::update_lighting(
                &mut combined.data,
                &combined.column_offset,
                combined.vsid,
                x0,
                y0,
                0,
                x1,
                y1,
                cs_z,
                LIGHTMODE,
                &[],
            );
        }
        combined.sync_alpha_to_chunks(&mut grid.chunks);

        // NOTE (2026-05-11): mip generation attempted as a perf
        // fix for the vsid=4096 demo. The `compilerle` emit-only-
        // top-of-column-floor-voxels bug makes mip-1+ slab data
        // unrenderable. Deferred to S6.
        // See `project_mip_attempt.md`.
    }
}
