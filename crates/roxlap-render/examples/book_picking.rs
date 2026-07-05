//! Companion example for the book's "Picking & world queries" chapter
//! (`docs/book/src/picking.md`) — the chapter pulls its snippets from
//! here via `// ANCHOR:` markers, so everything it shows compiles.
//!
//! An orbiting camera over a pillar field: the voxel under the screen
//! centre is outlined every frame (the backend-free `view_ray` +
//! `Scene::raycast` path), and a left-click carves a small crater at
//! the voxel under the mouse cursor (the depth-based `pick` path):
//!
//! ```sh
//! cargo run --release -p roxlap-render --example book_picking
//! ROXLAP_GPU=0 cargo run --release -p roxlap-render --example book_picking  # force CPU
//! ```
//!
//! Keep the anchors when editing; `docs/book/check-anchors.sh` (run by
//! the CI `book` job) goes red if one disappears.

use std::sync::Arc;
use std::time::Instant;

use glam::{DVec3, IVec3};
use roxlap_core::opticast::OpticastSettings;
use roxlap_core::Camera;
use roxlap_render::{BackendPreference, FrameParams, Line3, RenderOptions, SceneRenderer};
use roxlap_scene::{GridTransform, Scene};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

const GRASS: u32 = 0x80_4d_8a_3a;
const PILLAR: u32 = 0x80_b0_a0_88;
const SKY: u32 = 0x00_8f_bc_d4;

fn build_scene() -> Scene {
    let mut scene = Scene::new();
    let id = scene.add_grid(GridTransform::at(DVec3::ZERO));
    let grid = scene.grid_mut(id).expect("grid just added");
    grid.set_rect(
        IVec3::new(-128, -128, 210),
        IVec3::new(127, 127, 254),
        Some(GRASS),
    );
    for i in -2..=2 {
        for j in -2..=2 {
            let (x, y) = (i * 45, j * 45);
            grid.set_rect(
                IVec3::new(x - 5, y - 5, 175),
                IVec3::new(x + 5, y + 5, 209),
                Some(PILLAR),
            );
        }
    }
    grid.bake_lightmode(1);
    scene
}

/// The 12 edges of a unit voxel's AABB as always-on-top overlay lines.
fn voxel_outline(world_voxel: IVec3, color: u32) -> Vec<Line3> {
    let lo = [
        f64::from(world_voxel.x),
        f64::from(world_voxel.y),
        f64::from(world_voxel.z),
    ];
    let hi = [lo[0] + 1.0, lo[1] + 1.0, lo[2] + 1.0];
    let p = [
        [lo[0], lo[1], lo[2]],
        [hi[0], lo[1], lo[2]],
        [lo[0], hi[1], lo[2]],
        [hi[0], hi[1], lo[2]],
        [lo[0], lo[1], hi[2]],
        [hi[0], lo[1], hi[2]],
        [lo[0], hi[1], hi[2]],
        [hi[0], hi[1], hi[2]],
    ];
    const EDGES: [(usize, usize); 12] = [
        (0, 1),
        (1, 3),
        (3, 2),
        (2, 0),
        (4, 5),
        (5, 7),
        (7, 6),
        (6, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ];
    EDGES
        .map(|(a, b)| Line3 {
            a: p[a],
            b: p[b],
            color,
            width_px: 2.0,
            depth_test: false, // hover highlight: visible through geometry
        })
        .to_vec()
}

/// `renderer` before `window` so it drops first.
#[derive(Default)]
struct App {
    renderer: Option<SceneRenderer>,
    window: Option<Arc<Window>>,
    scene: Option<Scene>,
    started: Option<Instant>,
    /// Last cursor position in window pixels (for the click pick).
    cursor: (u32, u32),
    /// Camera of the last rendered frame (picking unprojects with it).
    last_camera: Option<Camera>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("roxlap book_picking"))
                .expect("create window"),
        );
        let size = window.inner_size();
        let backend = if std::env::var_os("ROXLAP_GPU").map_or(true, |v| v != "0") {
            BackendPreference::PreferGpu
        } else {
            BackendPreference::Cpu
        };
        let opts = RenderOptions {
            backend,
            clear_sky: SKY,
            ..RenderOptions::default()
        };
        self.renderer = Some(SceneRenderer::new(
            window.clone(),
            (size.width, size.height),
            &opts,
        ));
        self.window = Some(window);
        self.scene = Some(build_scene());
        self.started = Some(Instant::now());
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let (Some(renderer), Some(scene)) = (self.renderer.as_mut(), self.scene.as_mut()) else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => renderer.resize(size.width.max(1), size.height.max(1)),
            WindowEvent::CursorMoved { position, .. } => {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                {
                    self.cursor = (position.x.max(0.0) as u32, position.y.max(0.0) as u32);
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                let Some(camera) = self.last_camera else {
                    return;
                };
                let (mx, my) = self.cursor;
                // ANCHOR: click_pick
                // Click-time picking: `pick` unprojects the pixel with
                // the last frame's projection, reads its depth there,
                // and resolves the hit to (grid, grid-local voxel).
                // Depth is free on CPU; on GPU it blocks on a readback
                // — a click-time call, not a per-frame one.
                if let Some(hit) = renderer.pick(scene, &camera, mx, my) {
                    let grid = scene.grid_mut(hit.grid).expect("hit grid exists");
                    // Carve a crater, then re-light just the hole.
                    grid.set_sphere(hit.voxel, 4, None);
                    let r = IVec3::splat(4);
                    grid.bake_lightmode_bbox(hit.voxel - r, hit.voxel + r, 1);
                }
                // ANCHOR_END: click_pick
            }
            WindowEvent::RedrawRequested => {
                let t = self.started.map_or(0.0, |s| s.elapsed().as_secs_f64());
                let camera = Camera::orbit(t * 0.15, 0.45, 260.0, [0.0, 0.0, 195.0]);
                let window = self.window.as_ref().expect("window outlives renderer");
                let size = window.inner_size();
                let (w, h) = (size.width.max(1), size.height.max(1));
                let settings = OpticastSettings::for_oracle_framebuffer(w, h);
                let mut frame = FrameParams::new(&settings);
                frame.sky_color = SKY;
                frame.fog_color = SKY;
                renderer.render(scene, &camera, &frame);

                // ANCHOR: hover_raycast
                // Per-frame hover: unproject the screen centre with
                // `view_ray`, march the scene with `Scene::raycast`.
                // No depth buffer involved — identical on CPU and GPU
                // and cheap enough to run every frame.
                let centre = (f64::from(w) * 0.5, f64::from(h) * 0.5);
                let hover = renderer
                    .view_ray(&camera, centre.0, centre.1)
                    .and_then(|ray| scene.raycast(ray.origin, ray.dir, 2048.0));
                if let Some(hit) = hover {
                    // `hit.voxel` is grid-local; this grid sits at the
                    // world origin, so the outline needs no transform.
                    renderer.draw_lines(&camera, &voxel_outline(hit.voxel, 0xff_ff_e0_40));
                }
                // ANCHOR_END: hover_raycast

                renderer.present();
                self.last_camera = Some(camera);
                window.request_redraw();
            }
            _ => {}
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.wait_idle();
        }
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let event_loop = EventLoop::new().expect("create event loop");
    event_loop
        .run_app(&mut App::default())
        .expect("run event loop");
}
