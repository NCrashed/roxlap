//! Companion example for the book's "Rendering & backends" and "The
//! render pipeline" chapters (`docs/book/src/rendering.md` /
//! `render-pipeline.md`) — both pull their snippets from here via
//! `// ANCHOR:` markers, so everything they show compiles.
//!
//! A foggy plain with a dome and a pillar avenue, rendered through the
//! full retro pipeline (fixed 320×180 logical grid + 2× SSAA +
//! posterize with Bayer dither) with a wireframe overlay gizmo:
//!
//! ```sh
//! cargo run --release -p roxlap-render --example book_pipeline
//! ROXLAP_GPU=0 cargo run --release -p roxlap-render --example book_pipeline  # force CPU
//! ```
//!
//! Keep the anchors when editing; `docs/book/check-anchors.sh` (run by
//! the CI `book` job) goes red if one disappears.

use std::sync::Arc;
use std::time::Instant;

use glam::{DVec3, IVec3};
use roxlap_core::opticast::OpticastSettings;
use roxlap_core::Camera;
use roxlap_render::{
    Backend, BackendPreference, DitherMode, Feature, FrameParams, Line3, PosterizeConfig,
    RenderOptions, RenderResolution, SceneRenderer,
};
use roxlap_scene::{GridTransform, Scene};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

const GRASS: u32 = 0x80_4d_8a_3a;
const DOME: u32 = 0x80_40_60_c0;
const PILLAR: u32 = 0x80_b0_a0_88;
/// Framebuffer/sky packing is `0x00_RR_GG_BB` (no intensity byte).
const SKY: u32 = 0x00_8f_bc_d4;

/// A plain + a dome + an avenue of pillars marching away from the
/// camera, so the fog and the posterized colour ramps have something
/// to chew on.
fn build_scene() -> Scene {
    let mut scene = Scene::new();
    let id = scene.add_grid(GridTransform::at(DVec3::ZERO));
    let grid = scene.grid_mut(id).expect("grid just added");
    grid.set_rect(
        IVec3::new(-320, -320, 210),
        IVec3::new(319, 319, 254),
        Some(GRASS),
    );
    grid.set_sphere(IVec3::new(0, 0, 205), 30, Some(DOME));
    for i in 0..12 {
        let y = -280 + i * 48;
        for x in [-90, 90] {
            grid.set_rect(
                IVec3::new(x - 4, y - 4, 160),
                IVec3::new(x + 4, y + 4, 209),
                Some(PILLAR),
            );
        }
    }
    scene
}

// ANCHOR: gizmo
/// The 12 edges of an axis-aligned box as depth-tested overlay lines
/// — the shape of most editor gizmos.
fn wire_box(lo: [f64; 3], hi: [f64; 3], color: u32) -> Vec<Line3> {
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
        (2, 0), // z = lo face (the top — +z is down)
        (4, 5),
        (5, 7),
        (7, 6),
        (6, 4), // z = hi face (the bottom)
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7), // vertical edges
    ];
    EDGES
        .map(|(a, b)| Line3 {
            a: p[a],
            b: p[b],
            color, // 0xAARRGGBB — high byte is alpha here, not shading
            width_px: 1.5,
            depth_test: true, // occluded by nearer voxels
        })
        .to_vec()
}
// ANCHOR_END: gizmo

/// `renderer` before `window` so it drops first (surface handles must
/// release while the window is alive).
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
                .create_window(Window::default_attributes().with_title("roxlap book_pipeline"))
                .expect("create window"),
        );
        let size = window.inner_size();
        let want_gpu = std::env::var_os("ROXLAP_GPU").map_or(true, |v| v != "0");

        // ANCHOR: backend_select
        // PreferGpu tries WGPU and falls back to the CPU renderer with
        // a `log::warn` when init fails; RequireGpu turns that failure
        // into an error instead (benchmark rigs must not silently
        // measure a software render). Plain Cpu never touches WGPU.
        let opts = RenderOptions {
            backend: if want_gpu {
                BackendPreference::PreferGpu
            } else {
                BackendPreference::Cpu
            },
            clear_sky: SKY,
            ..RenderOptions::default()
        };
        let mut renderer =
            match SceneRenderer::try_new(window.clone(), (size.width, size.height), &opts) {
                Ok(r) => r,
                Err(e) => {
                    // Even the last-resort CPU software surface failed —
                    // there is nothing to draw on.
                    eprintln!("cannot create a render surface: {e}");
                    event_loop.exit();
                    return;
                }
            };
        // ANCHOR_END: backend_select

        // ANCHOR: supports
        // Backend capabilities differ below the parity line; methods on
        // the unsupported side degrade to documented no-ops. Query once
        // at startup and pick a strategy — don't guess per frame.
        match renderer.backend() {
            Backend::Gpu => log::info!(
                "GPU backend: {}",
                renderer.adapter_info().unwrap_or("unknown adapter")
            ),
            Backend::Cpu => log::info!("CPU backend (software per-pixel DDA)"),
        }
        if !renderer.supports(Feature::FreePickDepth) {
            // GPU depth reads block on a device poll: fine per click,
            // wrong per frame — so pick a reticle strategy up front.
            log::info!("per-frame depth picks are expensive here; use view_ray + raycast");
        }
        // ANCHOR_END: supports

        // ANCHOR: pipeline
        // The retro pipeline, configured once (each takes effect from
        // the next render): march a fixed 320×180 logical grid — frame
        // cost stops tracking the window size — with 2×2 supersampling
        // folded back down per logical pixel, then quantize to 6
        // levels per channel with an ordered Bayer dither before the
        // nearest-neighbour upscale to the window.
        renderer.set_render_resolution(RenderResolution::Fixed { w: 320, h: 180 });
        renderer.set_ssaa(2);
        renderer.set_posterize(Some(PosterizeConfig::uniform(6, DitherMode::Bayer4x4)));
        let (rw, rh) = renderer.render_dims(); // 640×360 marched…
        let (lw, lh) = renderer.logical_dims(); // …resolved to 320×180
        log::info!("marching {rw}×{rh}, resolving to {lw}×{lh}, upscaling to the window");
        // ANCHOR_END: pipeline

        self.renderer = Some(renderer);
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
                let t = self.started.map_or(0.0, |s| s.elapsed().as_secs_f64());
                let camera = Camera::orbit(t * 0.25, 0.3, 260.0, [0.0, 0.0, 195.0]);
                let window = self.window.as_ref().expect("window outlives renderer");
                let size = window.inner_size();

                // ANCHOR: frame_params
                // FrameParams::new gives working defaults for everything
                // but `settings`; override what differs. Both backends
                // derive their projection from `settings`, so CPU and
                // GPU show the same field of view.
                let settings =
                    OpticastSettings::for_oracle_framebuffer(size.width.max(1), size.height.max(1));
                let mut frame = FrameParams::new(&settings);
                frame.sky_color = SKY;
                // CPU fog: distant voxels fade into the sky colour,
                // fully fogged at 700 voxels (0 = fog off).
                frame.fog_color = SKY;
                frame.fog_max_scan_dist = 700;
                // ANCHOR_END: frame_params

                // ANCHOR: overlay
                renderer.render(scene, &camera, &frame);
                // Overlays go between render and present: they draw into
                // the composited frame using its camera, projection and
                // depth buffer — here a gizmo box around the dome.
                let gizmo = wire_box([-32.0, -32.0, 173.0], [32.0, 32.0, 237.0], 0xff_ff_d0_40);
                renderer.draw_lines(&camera, &gizmo);
                // Exactly one of present() / paint_egui(..) finishes it.
                renderer.present();
                // ANCHOR_END: overlay

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
