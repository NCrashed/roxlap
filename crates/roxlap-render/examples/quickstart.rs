//! Minimal roxlap "hello voxel world": open a winit window, build a
//! tiny scene (a grass plain + a blue dome), and render it with the
//! unified [`SceneRenderer`] facade — a slow orbit camera so the scene
//! reads as 3D.
//!
//! ```sh
//! cargo run --release -p roxlap-render --example quickstart
//! ROXLAP_GPU=0 cargo run --release -p roxlap-render --example quickstart  # force CPU
//! ```
//!
//! The renderer asks for the GPU backend and **falls back to CPU
//! automatically** when WGPU init fails, so this runs anywhere.
//!
//! To depend on roxlap from your own game, add to `Cargo.toml`:
//!
//! ```toml
//! roxlap-render = "0.21"   # SceneRenderer facade (this crate)
//! roxlap-scene  = "0.21"   # Scene / Grid / edits / streaming
//! roxlap-core   = "0.21"   # Camera + OpticastSettings
//! ```

use std::sync::Arc;
use std::time::Instant;

use glam::{DVec3, IVec3};
use roxlap_core::opticast::OpticastSettings;
use roxlap_core::Camera;
use roxlap_render::{FrameParams, RenderOptions, SceneRenderer};
use roxlap_scene::{GridTransform, Scene};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

/// Voxel colours are voxlap-packed `0x80_RR_GG_BB` — the high byte is
/// the flat shading intensity, the low 24 bits the RGB colour.
const GRASS: u32 = 0x80_4d_8a_3a;
const DOME: u32 = 0x80_40_60_c0;
/// Sky colour in the framebuffer's native `0x00_RR_GG_BB` packing.
const SKY: u32 = 0x00_8f_bc_d4;

/// A one-grid scene. Grid-local voxel coordinates; **+z is down**, so
/// the ground surface at `z = 210` fills downward toward the bedrock
/// placeholder at `z = 255`, and "up" is toward smaller z.
fn build_scene() -> Scene {
    let mut scene = Scene::new();
    let id = scene.add_grid(GridTransform::at(DVec3::ZERO));
    let grid = scene.grid_mut(id).expect("grid just added");
    grid.set_rect(
        IVec3::new(-128, -128, 210),
        IVec3::new(127, 127, 254),
        Some(GRASS),
    );
    grid.set_sphere(IVec3::new(0, 0, 205), 30, Some(DOME));
    scene
}

/// `renderer` is declared **before** `window` so it drops first: the
/// render surface must release its raw window handles while the window
/// is still alive.
#[derive(Default)]
struct App {
    renderer: Option<SceneRenderer>,
    window: Option<Arc<Window>>,
    scene: Option<Scene>,
    started: Option<Instant>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("roxlap quickstart"))
                .expect("create window"),
        );
        let size = window.inner_size();
        let want_gpu = std::env::var_os("ROXLAP_GPU").map_or(true, |v| v != "0");
        let opts = RenderOptions {
            want_gpu,
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
            WindowEvent::RedrawRequested => {
                // Orbit the dome. `Camera::orbit` / `from_yaw_pitch` /
                // `look_at` produce the canonical right-handed basis the
                // sprite frustum cull needs — don't hand-roll one.
                let t = self.started.map_or(0.0, |s| s.elapsed().as_secs_f64());
                let camera = Camera::orbit(t * 0.3, 0.35, 220.0, [0.0, 0.0, 195.0]);

                let window = self.window.as_ref().expect("window outlives renderer");
                let size = window.inner_size();
                let settings =
                    OpticastSettings::for_oracle_framebuffer(size.width.max(1), size.height.max(1));
                // Defaults for everything but the sky; the GPU
                // projection is derived from `settings`, so both
                // backends show the same field of view.
                let mut frame = FrameParams::new(&settings);
                frame.sky_color = SKY;
                frame.fog_color = SKY;

                renderer.render(scene, &camera, &frame);
                renderer.present(); // render() composites; present() finishes
                window.request_redraw();
            }
            _ => {}
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        // Drain in-flight GPU work before surface/window teardown so
        // quitting never yanks the swapchain mid-submission.
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.wait_idle();
        }
    }
}

fn main() {
    // roxlap-render reports diagnostics (e.g. why GPU init failed and
    // it fell back to CPU) through the `log` facade — install any
    // logger to see them. RUST_LOG=debug for more detail.
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let event_loop = EventLoop::new().expect("create event loop");
    event_loop
        .run_app(&mut App::default())
        .expect("run event loop");
}
