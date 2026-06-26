//! The thin demo host (stage DS): owns the window, [`SceneRenderer`],
//! egui, FPS, the shared fly-camera + mouse-look, the scan distance, and
//! the scene menu — and drives the active [`DemoScene`]. All
//! feature-specific content lives in the scenes (`scenes/`).

use std::sync::Arc;
use std::time::Instant;

use roxlap_core::Engine;
use roxlap_render::{Backend, RenderOptions, SceneRenderer, SpriteSet};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{DeviceEvent, DeviceId, ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

use crate::scene_api::{CameraRig, DemoScene, InputState, SceneCtx, SceneInput};
use crate::scenes::{
    animation::AnimationScene, empty::EmptyScene, picking::PickingScene,
    primitives::PrimitivesScene, sprites::SpritesScene, world::WorldScene,
};
use crate::{
    load_png_sky, load_png_sky_rgba, MAX_GRID_VSID, RENDER_THREADS, SCAN_DIST_INITIAL,
    SCAN_DIST_MAX, SCAN_DIST_MIN, SCAN_DIST_STEP, SKY_PNG,
};

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

/// An empty sprite set — used to reset the renderer's content layers
/// (static + dynamic + clip + character) when switching scenes.
fn empty_sprite_set() -> SpriteSet {
    SpriteSet {
        models: Vec::new(),
        instances: Vec::new(),
        carve_model: None,
    }
}

pub struct Host {
    window: Option<Arc<Window>>,
    renderer: Option<SceneRenderer>,
    engine: Engine,
    cam: CameraRig,
    input: InputState,
    grabbed: bool,
    /// Accumulated mouse-look delta (px) since the last frame.
    look_accum: (f64, f64),
    scan_dist: i32,
    scenes: Vec<Box<dyn DemoScene>>,
    active: usize,
    /// Scene index to switch to after the egui pass (menu click).
    pending_switch: Option<usize>,
    menu_open: bool,
    last_frame: Instant,
    egui_ctx: egui::Context,
    egui_state: Option<egui_winit::State>,
    hud_on: bool,
    title_base: String,
    fps_frames: u32,
    fps_last: Instant,
    last_fps: f64,
}

impl Host {
    #[must_use]
    pub fn new() -> Self {
        let mut engine = Engine::new();
        let sky = load_png_sky(SKY_PNG).unwrap_or_else(|e| {
            eprintln!("sky: PNG decode failed ({e}); falling back to blue gradient");
            roxlap_core::sky::Sky::blue_gradient()
        });
        engine.set_sky(Some(sky));

        let scenes: Vec<Box<dyn DemoScene>> = vec![
            Box::new(WorldScene::new()),
            Box::new(SpritesScene::new()),
            Box::new(AnimationScene::new()),
            Box::new(PickingScene::new()),
            Box::new(PrimitivesScene::new()),
            Box::new(EmptyScene::new()),
        ];
        let cam = CameraRig::from_pose(scenes[0].start_pose());

        Self {
            window: None,
            renderer: None,
            engine,
            cam,
            input: InputState::default(),
            grabbed: false,
            look_accum: (0.0, 0.0),
            scan_dist: SCAN_DIST_INITIAL,
            scenes,
            active: 0,
            pending_switch: None,
            menu_open: false,
            last_frame: Instant::now(),
            egui_ctx: egui::Context::default(),
            egui_state: None,
            hud_on: true,
            title_base: "roxlap-scene-demo".to_string(),
            fps_frames: 0,
            fps_last: Instant::now(),
            last_fps: 0.0,
        }
    }

    /// Forward a scene-local input event to the active scene (split-borrow
    /// of the host's renderer/cam/engine).
    fn scene_input(&mut self, ev: SceneInput) {
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };
        let size = self.window.as_ref().map_or((WIDTH, HEIGHT), |w| {
            let s = w.inner_size();
            (s.width.max(1), s.height.max(1))
        });
        let mut ctx = SceneCtx {
            renderer,
            cam: &mut self.cam,
            input: self.input,
            size,
            engine: &mut self.engine,
            scan_dist: self.scan_dist,
        };
        self.scenes[self.active].on_input(&mut ctx, &ev);
    }

    fn set_grab(&mut self, grab: bool) {
        let Some(window) = self.window.as_ref() else {
            return;
        };
        if grab {
            let _ = window
                .set_cursor_grab(CursorGrabMode::Confined)
                .or_else(|_| window.set_cursor_grab(CursorGrabMode::Locked));
            window.set_cursor_visible(false);
            self.grabbed = true;
        } else {
            let _ = window.set_cursor_grab(CursorGrabMode::None);
            window.set_cursor_visible(true);
            self.grabbed = false;
        }
    }

    /// Apply a pending scene switch: exit the old scene, reset the
    /// renderer's content layers, enter the new scene at its start pose.
    fn switch_to(&mut self, idx: usize, size: (u32, u32)) {
        if idx >= self.scenes.len() || idx == self.active {
            return;
        }
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };
        {
            let mut ctx = SceneCtx {
                renderer,
                cam: &mut self.cam,
                input: self.input,
                size,
                engine: &mut self.engine,
                scan_dist: self.scan_dist,
            };
            self.scenes[self.active].exit(&mut ctx);
        }
        // Drop all registered content (static + dynamic + clip + character).
        renderer.set_sprites(&empty_sprite_set());
        self.active = idx;
        self.cam = CameraRig::from_pose(self.scenes[idx].start_pose());
        let mut ctx = SceneCtx {
            renderer,
            cam: &mut self.cam,
            input: self.input,
            size,
            engine: &mut self.engine,
            scan_dist: self.scan_dist,
        };
        self.scenes[idx].enter(&mut ctx);
        eprintln!("scene → {}", self.scenes[idx].name());
    }

    fn tick_fps(&mut self) {
        self.fps_frames += 1;
        let now = Instant::now();
        let dt = (now - self.fps_last).as_secs_f32();
        if dt < 0.5 {
            return;
        }
        let fps = f64::from(self.fps_frames) / f64::from(dt);
        self.last_fps = fps;
        if let Some(window) = self.window.as_ref() {
            window.set_title(&format!("{} — {:.1} FPS", self.title_base, fps));
        }
        if std::env::var_os("ROXLAP_FPS_LOG").is_some() {
            eprintln!("fps: {fps:.1}");
        }
        self.fps_frames = 0;
        self.fps_last = now;
    }

    fn redraw(&mut self) {
        let size = {
            let Some(window) = self.window.as_ref() else {
                return;
            };
            let s = window.inner_size();
            (s.width, s.height)
        };
        if size.0 == 0 || size.1 == 0 {
            return;
        }

        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f64();
        self.last_frame = now;

        // Mouse-look (accumulated since last frame), then per-scene update +
        // render — split-borrow the disjoint host fields.
        let (lx, ly) = std::mem::take(&mut self.look_accum);
        self.cam.apply_look(lx, ly);
        {
            let Some(renderer) = self.renderer.as_mut() else {
                return;
            };
            let mut ctx = SceneCtx {
                renderer,
                cam: &mut self.cam,
                input: self.input,
                size,
                engine: &mut self.engine,
                scan_dist: self.scan_dist,
            };
            let scene = &mut self.scenes[self.active];
            scene.update(&mut ctx, dt);
            scene.render(&mut ctx);
        }

        self.present_frame();

        if let Some(i) = self.pending_switch.take() {
            self.switch_to(i, size);
        }
        if let Some(w) = self.window.as_ref() {
            w.request_redraw();
        }
        self.tick_fps();
    }

    /// Finish the frame: overlay the egui HUD + (if open) the scene menu,
    /// then `paint_egui`; or a plain `present` when the HUD is off.
    fn present_frame(&mut self) {
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };
        let hud_ready = self.hud_on && self.window.is_some() && self.egui_state.is_some();
        if !hud_ready {
            renderer.present();
            return;
        }
        let backend = match renderer.backend() {
            Backend::Gpu => "GPU",
            Backend::Cpu => "CPU",
        };
        // Snapshot the scene-derived strings before the egui closures so the
        // `scenes` borrow doesn't overlap the `pending_switch` write.
        let (sname, sctrl, slines) = {
            let s = &self.scenes[self.active];
            (s.name(), s.controls(), s.hud_lines())
        };
        let names: Vec<&'static str> = self.scenes.iter().map(|s| s.name()).collect();

        let window = self.window.as_ref().expect("hud_ready");
        let state = self.egui_state.as_mut().expect("hud_ready");
        let raw = state.take_egui_input(window);
        self.egui_ctx.begin_pass(raw);

        hud_panel(
            &self.egui_ctx,
            backend,
            self.last_fps,
            self.cam,
            self.scan_dist,
            sname,
            sctrl,
            &slines,
        );
        if self.menu_open {
            if let Some(pick) = scene_menu(&self.egui_ctx, &names, self.active) {
                self.pending_switch = Some(pick);
                self.menu_open = false;
            }
        }

        let full = self.egui_ctx.end_pass();
        state.handle_platform_output(window, full.platform_output);
        let ppp = self.egui_ctx.pixels_per_point();
        let jobs = self.egui_ctx.tessellate(full.shapes, ppp);
        renderer.paint_egui(&jobs, &full.textures_delta, ppp);
    }
}

impl ApplicationHandler for Host {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_title("roxlap-scene-demo")
            .with_inner_size(LogicalSize::new(WIDTH, HEIGHT));
        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .expect("winit: create_window"),
        );

        let want_gpu = std::env::var_os("ROXLAP_GPU").is_some_and(|v| v != "0" && !v.is_empty());
        let opts = RenderOptions {
            want_gpu,
            cpu_max_grid_vsid: MAX_GRID_VSID,
            cpu_render_threads: RENDER_THREADS,
            ..RenderOptions::default()
        };
        let init = window.inner_size();
        let mut renderer = SceneRenderer::new(window.clone(), (init.width, init.height), &opts);

        self.title_base = if let Some(info) = renderer.adapter_info() {
            eprintln!("roxlap-render: GPU backend — {info}");
            format!("roxlap-scene-demo (GPU: {info})")
        } else {
            eprintln!("roxlap-render: CPU backend");
            "roxlap-scene-demo (CPU)".to_string()
        };
        window.set_title(&self.title_base);

        match load_png_sky_rgba(SKY_PNG) {
            Ok((rgba, w, h)) => {
                renderer.set_sky_panorama(&rgba, w, h);
                eprintln!("roxlap-render: sky panorama uploaded ({w}×{h})");
            }
            Err(e) => eprintln!("roxlap-render: sky decode failed ({e})"),
        }

        self.egui_state = Some(egui_winit::State::new(
            self.egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &*window,
            None,
            None,
            Some(2048),
        ));

        // Enter the active scene (register its content).
        let size = (init.width.max(1), init.height.max(1));
        {
            let mut ctx = SceneCtx {
                renderer: &mut renderer,
                cam: &mut self.cam,
                input: self.input,
                size,
                engine: &mut self.engine,
                scan_dist: self.scan_dist,
            };
            self.scenes[self.active].enter(&mut ctx);
        }
        eprintln!("scene → {} (Tab: menu)", self.scenes[self.active].name());

        self.renderer = Some(renderer);
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        // egui passthrough (never consumed) so the menu/HUD is interactive.
        if let (Some(window), Some(state)) = (self.window.as_ref(), self.egui_state.as_mut()) {
            let _ = state.on_window_event(window, &event);
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(sz) => {
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.resize(sz.width, sz.height);
                }
            }
            WindowEvent::RedrawRequested => self.redraw(),
            WindowEvent::CursorMoved { position, .. } => {
                self.scene_input(SceneInput::CursorMoved {
                    x: position.x,
                    y: position.y,
                });
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button,
                ..
            } => {
                if self.menu_open {
                    // egui consumes the click on the menu.
                } else if button == MouseButton::Left && !self.grabbed {
                    self.set_grab(true);
                } else {
                    self.scene_input(SceneInput::Mouse {
                        button,
                        pressed: true,
                    });
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state,
                        ..
                    },
                ..
            } => {
                let pressed = state == ElementState::Pressed;
                match code {
                    KeyCode::KeyW => self.input.forward = pressed,
                    KeyCode::KeyS => self.input.back = pressed,
                    KeyCode::KeyA => self.input.left = pressed,
                    KeyCode::KeyD => self.input.right = pressed,
                    KeyCode::Space => self.input.up = pressed,
                    KeyCode::ShiftLeft => self.input.down = pressed,
                    KeyCode::ControlLeft => self.input.fast = pressed,
                    KeyCode::Tab if pressed => {
                        self.menu_open = !self.menu_open;
                        if self.menu_open {
                            self.set_grab(false);
                        }
                    }
                    KeyCode::F1 if pressed => self.hud_on = !self.hud_on,
                    KeyCode::Equal | KeyCode::NumpadAdd if pressed => {
                        self.scan_dist = (self.scan_dist + SCAN_DIST_STEP).min(SCAN_DIST_MAX);
                        eprintln!("scan_dist = {}", self.scan_dist);
                    }
                    KeyCode::Minus | KeyCode::NumpadSubtract if pressed => {
                        self.scan_dist = (self.scan_dist - SCAN_DIST_STEP).max(SCAN_DIST_MIN);
                        eprintln!("scan_dist = {}", self.scan_dist);
                    }
                    KeyCode::Escape if pressed => {
                        if self.menu_open {
                            self.menu_open = false;
                        } else if self.grabbed {
                            self.set_grab(false);
                        } else {
                            event_loop.exit();
                        }
                    }
                    other => self.scene_input(SceneInput::Key {
                        code: other,
                        pressed,
                    }),
                }
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _el: &ActiveEventLoop, _id: DeviceId, event: DeviceEvent) {
        if !self.grabbed {
            return;
        }
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            self.look_accum.0 += dx;
            self.look_accum.1 += dy;
        }
    }
}

/// The fixed top-left HUD info box: backend / FPS / camera + scan distance,
/// the active scene's name + controls, and any per-scene live lines.
#[allow(clippy::too_many_arguments)]
fn hud_panel(
    ctx: &egui::Context,
    backend: &str,
    fps: f64,
    cam: CameraRig,
    scan: i32,
    scene_name: &str,
    scene_controls: &str,
    scene_lines: &[String],
) {
    egui::Area::new(egui::Id::new("roxlap-hud"))
        .fixed_pos(egui::pos2(8.0, 8.0))
        .show(ctx, |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.label(format!("roxlap-scene-demo — {backend} · {fps:.0} FPS"));
                ui.label(format!("scene: {scene_name}   (Tab: menu)"));
                ui.label(format!(
                    "pos ({:.0}, {:.0}, {:.0})   scan {scan}",
                    cam.pos[0], cam.pos[1], cam.pos[2]
                ));
                ui.label(scene_controls);
                for line in scene_lines {
                    ui.label(line);
                }
                ui.separator();
                ui.label("F1: HUD");
            });
        });
}

/// The scene-picker panel (opened with `Tab`). Returns the index a button
/// click selected, if any.
fn scene_menu(ctx: &egui::Context, names: &[&'static str], active: usize) -> Option<usize> {
    let mut pick = None;
    egui::Window::new("Scenes")
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
        .show(ctx, |ui| {
            ui.label("Pick a demo scene:");
            for (i, name) in names.iter().enumerate() {
                let label = if i == active {
                    format!("▶ {name}")
                } else {
                    format!("   {name}")
                };
                if ui.button(label).clicked() {
                    pick = Some(i);
                }
            }
            ui.separator();
            ui.label("Tab / Esc to close");
        });
    pick
}
