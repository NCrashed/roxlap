//! Companion example for the book's "Sprites & animation" chapter
//! (`docs/book/src/sprites.md`) — the chapter pulls its snippets from
//! here via `// ANCHOR:` markers, so everything it shows compiles.
//!
//! Three gems on a plain (one plain, one double-size via the basis,
//! one spinning per frame), a pulsing animated voxel clip, and the
//! same clip as a camera-facing Doom-style billboard:
//!
//! ```sh
//! cargo run --release -p roxlap-render --example book_sprites
//! ROXLAP_GPU=0 cargo run --release -p roxlap-render --example book_sprites  # force CPU
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
    BackendPreference, BillboardMode, BillboardUp, DynSpriteTransform, FrameParams, Kv6, LoopMode,
    RenderOptions, Rgb, SceneRenderer, SpriteInstanceId, VoxColor, VoxelClip,
};
use roxlap_scene::{GridTransform, Scene};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

const GRASS: VoxColor = VoxColor(0x80_4d_8a_3a);
const GEM: VoxColor = VoxColor(0x80_c8_50_d0);
const ORB: VoxColor = VoxColor(0x80_d0_a0_50);
const SKY: Rgb = Rgb(0x00_8f_bc_d4);

fn build_scene() -> Scene {
    let mut scene = Scene::new();
    let id = scene.add_grid(GridTransform::at(DVec3::ZERO));
    let grid = scene.grid_mut(id).expect("grid just added");
    grid.set_rect(
        IVec3::new(-128, -128, 210),
        IVec3::new(127, 127, 254),
        Some(GRASS),
    );
    scene
}

/// A faceted gem: a sphere with the corners shaved off.
fn gem_kv6() -> Kv6 {
    const DIM: u32 = 14;
    let c = (DIM as f32 - 1.0) * 0.5;
    Kv6::from_fn(DIM, DIM, DIM, |x, y, z| {
        let (dx, dy, dz) = (x as f32 - c, y as f32 - c, z as f32 - c);
        let round = dx * dx + dy * dy + dz * dz <= (c * 0.98).powi(2);
        let cut = dx.abs() + dy.abs() + dz.abs() <= c * 1.35;
        (round && cut).then_some(GEM)
    })
}

// ANCHOR: clip_build
/// A pulsing orb as an animated **voxel clip** — the "GIF for voxels":
/// four kv6 frames sharing one bounding box, encoded as keyframe +
/// deltas. 150 ms per frame, looping.
fn orb_clip() -> VoxelClip {
    const DIM: u32 = 18;
    let c = (DIM as f32 - 1.0) * 0.5;
    let frames: Vec<Kv6> = [5.0_f32, 6.5, 8.0, 6.5]
        .iter()
        .map(|&r| {
            let r2 = r * r;
            Kv6::from_fn(DIM, DIM, DIM, |x, y, z| {
                let (dx, dy, dz) = (x as f32 - c, y as f32 - c, z as f32 - c);
                (dx * dx + dy * dy + dz * dz <= r2).then_some(ORB)
            })
        })
        .collect();
    VoxelClip::from_kv6_frames(&frames, 1.0, LoopMode::Loop, &[], 150, 1)
        .expect("frames are non-empty and share dims")
}
// ANCHOR_END: clip_build

/// `renderer` before `window` so it drops first.
#[derive(Default)]
struct App {
    renderer: Option<SceneRenderer>,
    window: Option<Arc<Window>>,
    scene: Option<Scene>,
    started: Option<Instant>,
    last_frame: Option<Instant>,
    /// The gem instance whose transform is rewritten every frame.
    spinner: Option<SpriteInstanceId>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("roxlap book_sprites"))
                .expect("create window"),
        );
        let size = window.inner_size();
        let backend = if std::env::var_os("ROXLAP_GPU").is_none_or(|v| v != "0") {
            BackendPreference::PreferGpu
        } else {
            BackendPreference::Cpu
        };
        let opts = RenderOptions {
            backend,
            clear_sky: SKY,
            ..RenderOptions::default()
        };
        let mut renderer = SceneRenderer::new(window.clone(), (size.width, size.height), &opts);

        // ANCHOR: model_instances
        // One model, many instances: register the kv6 once, then place
        // posed instances. A pose is a position + a local→world basis;
        // scaling the basis scales the model — there is no separate
        // scale field.
        let gem = renderer.add_sprite_model(&gem_kv6());
        renderer
            .add_sprite_instance_posed(
                gem,
                DynSpriteTransform {
                    pos: [-50.0, 0.0, 202.0],
                    ..DynSpriteTransform::default() // identity basis: authored size
                },
            )
            .expect("model just registered");
        renderer
            .add_sprite_instance_posed(
                gem,
                DynSpriteTransform {
                    pos: [0.0, 0.0, 198.0],
                    right: [2.0, 0.0, 0.0], // basis ×2 ⇒ the gem renders
                    up: [0.0, 2.0, 0.0],    // at twice its authored size
                    forward: [0.0, 0.0, 2.0],
                },
            )
            .expect("model just registered");
        let spinner = renderer
            .add_sprite_instance_posed(
                gem,
                DynSpriteTransform {
                    pos: [50.0, 0.0, 202.0],
                    ..DynSpriteTransform::default()
                },
            )
            .expect("model just registered");
        // ANCHOR_END: model_instances

        // ANCHOR: clip_instances
        // Register the clip once, then instance it. `_playing` starts
        // its per-instance player: `tick` (or `advance_voxel_clips`)
        // advances the clock and swaps frames — O(changed frame), not
        // O(volume).
        let clip = renderer.add_voxel_clip(&orb_clip().decode().expect("self-authored clip"));
        renderer
            .add_clip_instance_playing(
                clip,
                DynSpriteTransform {
                    pos: [0.0, -60.0, 200.0],
                    ..DynSpriteTransform::default()
                },
                1.0, // playback speed
                0,   // start phase, ms
            )
            .expect("clip just registered");
        // ANCHOR_END: clip_instances

        // ANCHOR: billboard
        // The same clip as a Doom-style billboard: a camera-facing
        // instance. Cylindrical = yaw-only facing (stays vertical, the
        // Build-engine look); Spherical also pitches with the view.
        let card = renderer
            .add_billboard_instance(clip, [0.0, 60.0, 200.0], BillboardMode::Cylindrical)
            .expect("clip just registered");
        // Which way is up in the image is the independent knob: World (the
        // default), Camera (never leans on screen — for a rolled camera), or
        // Axis(v) for a card standing on a body with an up of its own.
        renderer.set_billboard_up(card, BillboardUp::World);
        // ANCHOR_END: billboard

        self.renderer = Some(renderer);
        self.window = Some(window);
        self.scene = Some(build_scene());
        self.started = Some(Instant::now());
        self.last_frame = Some(Instant::now());
        self.spinner = spinner.into();
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
                let now = Instant::now();
                let dt = self
                    .last_frame
                    .map_or(0.0, |l| (now - l).as_secs_f64())
                    .min(0.1);
                self.last_frame = Some(now);
                let camera = Camera::orbit(t * 0.2, 0.3, 200.0, [0.0, 0.0, 195.0]);

                // ANCHOR: per_frame
                // Rewrite the spinner's basis: a yaw rotation about the
                // world vertical. Per-frame transform writes are the
                // intended path — they update in place, no re-upload.
                if let Some(id) = self.spinner {
                    let (s, c) = (t * 1.5).sin_cos();
                    let (s, c) = (s as f32, c as f32);
                    renderer.set_sprite_instance_transform(
                        id,
                        DynSpriteTransform {
                            pos: [50.0, 0.0, 202.0],
                            right: [c, s, 0.0],
                            up: [-s, c, 0.0],
                            forward: [0.0, 0.0, 1.0],
                        },
                    );
                }
                // One call advances everything facade-owned: clip
                // players, characters, billboard actors, and the
                // camera-facing of plain billboards.
                renderer.tick(&camera, dt);
                // ANCHOR_END: per_frame

                let window = self.window.as_ref().expect("window outlives renderer");
                let size = window.inner_size();
                let settings =
                    OpticastSettings::for_oracle_framebuffer(size.width.max(1), size.height.max(1));
                let mut frame = FrameParams::new(&settings);
                frame.sky_color = SKY;
                frame.fog_color = SKY;
                renderer.render(scene, &camera, &frame);
                renderer.present();
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
