//! Scene assembly entry point — builds the demo `Scene` and
//! returns it along with a sensible starting camera.
//!
//! S3.x: 2 single-chunk axis-aligned grids — ground patch + ship.
//! Both grid origins are picked so the camera can frame both
//! without OOB-camera weirdness.

use glam::DVec3;
use roxlap_core::Camera;
use roxlap_scene::{GridTransform, Scene};

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
/// - **Ground** grid origin at world `(0, 0, 0)` — terrain in
///   chunk-local z ∈ [80..255].
/// - **Ship** grid origin at world `(0, 0, -100)` — ship body
///   in chunk-local z ∈ [56..72] → world z ∈ [-44..-28].
///   The ship sits comfortably in the sky band above the terrain
///   (terrain peaks at world z ≈ 80; sky is z < 80).
///
/// Initial camera at world `(0, -120, 50)` looking +y, sees the
/// ship dead-ahead with the terrain visible below the horizon.
pub fn build_demo() -> SceneAndCamera {
    let mut scene = Scene::new();

    let ground_id = scene.add_grid(GridTransform::at(DVec3::new(0.0, 0.0, 0.0)));
    terrain::build_ground(scene.grid_mut(ground_id).expect("ground grid present"));

    let ship_id = scene.add_grid(GridTransform::at(DVec3::new(0.0, 0.0, -100.0)));
    ship::build_ship(scene.grid_mut(ship_id).expect("ship grid present"));

    let initial_pos = [64.0, -120.0, 50.0];
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
