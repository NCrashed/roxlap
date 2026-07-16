//! The thin demo host (stage DS): owns the window, [`SceneRenderer`],
//! egui, FPS, the shared fly-camera + mouse-look, the scan distance, and
//! the scene menu — and drives the active [`DemoScene`]. All
//! feature-specific content lives in the scenes (`scenes/`).

use std::sync::Arc;
use std::time::Instant;

use roxlap_core::Engine;
use roxlap_render::{
    Backend, BackendPreference, DitherMode, PosterizeConfig, RenderOptions, RenderResolution,
    SceneRenderer,
};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{DeviceEvent, DeviceId, ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

use crate::scene_api::{CameraPose, CameraRig, DemoScene, InputState, SceneCtx, SceneInput};
use crate::scenes::{
    animation::AnimationScene, audio::AudioScene, decks::DecksScene, doom::DoomScene,
    empty::EmptyScene, lighting::LightingScene, particles::ParticlesScene, picking::PickingScene,
    primitives::PrimitivesScene, scale::ScaleScene, spotlight::SpotlightScene,
    sprites::SpritesScene, transparency::TransparencyScene, world::WorldScene,
};
use crate::{
    load_png_sky, load_png_sky_rgba, SCAN_DIST_INITIAL, SCAN_DIST_MAX, SCAN_DIST_MIN,
    SCAN_DIST_STEP, SKY_PNG,
};

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

/// RP.0 default logical render grid (the marcher's fixed pixel grid; the
/// window upscales onto it). Decouples FPS from window size.
const RENDER_RES_W: u32 = 860;
const RENDER_RES_H: u32 = 520;

/// Parse `ROXLAP_RENDER_RES` into a [`RenderResolution`]. Accepts `native`,
/// `WxH` (e.g. `640x360`), or a bare scale factor (e.g. `0.5`). Anything
/// unset / unparseable falls back to the fixed [`RENDER_RES_W`]×[`RENDER_RES_H`]
/// default.
fn parse_render_res() -> RenderResolution {
    let default = RenderResolution::Fixed {
        w: RENDER_RES_W,
        h: RENDER_RES_H,
    };
    let Some(raw) = std::env::var_os("ROXLAP_RENDER_RES") else {
        return default;
    };
    let s = raw.to_string_lossy();
    let s = s.trim();
    if s.eq_ignore_ascii_case("native") {
        return RenderResolution::Native;
    }
    if let Some((w, h)) = s.split_once(['x', 'X']) {
        if let (Ok(w), Ok(h)) = (w.trim().parse::<u32>(), h.trim().parse::<u32>()) {
            if w > 0 && h > 0 {
                return RenderResolution::Fixed { w, h };
            }
        }
    }
    if let Ok(f) = s.parse::<f32>() {
        if f > 0.0 {
            return RenderResolution::Scale(f);
        }
    }
    eprintln!("roxlap-scene-demo: unparseable ROXLAP_RENDER_RES={s:?}; using default");
    default
}

/// Parse `ROXLAP_SSAA` into a supersampling factor (clamped `1..=4`).
/// Default `1` (off). RP.1.
fn parse_ssaa() -> u8 {
    std::env::var("ROXLAP_SSAA")
        .ok()
        .and_then(|s| s.trim().parse::<u8>().ok())
        .unwrap_or(1)
        .clamp(1, 4)
}

/// Parse `ROXLAP_POSTERIZE` (per-channel level count) + `ROXLAP_DITHER`
/// (`none`|`bayer`|`blue`) into an optional [`PosterizeConfig`] (RP.2).
/// Unset / `0` / `1` ⇒ disabled. Dither defaults to blue-noise.
fn parse_posterize() -> Option<PosterizeConfig> {
    let levels = std::env::var("ROXLAP_POSTERIZE")
        .ok()
        .and_then(|s| s.trim().parse::<u8>().ok())?;
    if levels <= 1 {
        return None;
    }
    let dither = match std::env::var("ROXLAP_DITHER")
        .ok()
        .as_deref()
        .map(str::trim)
    {
        Some("none") => DitherMode::None,
        Some("bayer") => DitherMode::Bayer4x4,
        _ => DitherMode::BlueNoise,
    };
    Some(PosterizeConfig::uniform(levels, dither))
}

/// RP.3 — which [`RenderResolution`] variant the HUD is editing.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ResMode {
    Native,
    Fixed,
    Scale,
}

/// RP.3 — live-editable render-pipeline state backing the HUD controls
/// (resolution / SSAA / posterize / dither). Seeded from the env vars, then
/// mutated by the egui panel; changes are pushed to the renderer each frame.
#[derive(Clone, Copy, PartialEq)]
struct PipelineUi {
    res_mode: ResMode,
    fixed_w: u32,
    fixed_h: u32,
    scale: f32,
    ssaa: u8,
    posterize_on: bool,
    levels: u8,
    dither: DitherMode,
}

impl PipelineUi {
    /// Seed from the `ROXLAP_*` env vars (the same parse the CLI uses).
    fn from_env() -> Self {
        let (res_mode, fixed_w, fixed_h, scale) = match parse_render_res() {
            RenderResolution::Native => (ResMode::Native, RENDER_RES_W, RENDER_RES_H, 0.5),
            RenderResolution::Fixed { w, h } => (ResMode::Fixed, w, h, 0.5),
            RenderResolution::Scale(f) => (ResMode::Scale, RENDER_RES_W, RENDER_RES_H, f),
        };
        let (posterize_on, levels, dither) = match parse_posterize() {
            Some(p) => (true, p.levels_r, p.dither),
            None => (false, 4, DitherMode::BlueNoise),
        };
        Self {
            res_mode,
            fixed_w,
            fixed_h,
            scale,
            ssaa: parse_ssaa(),
            posterize_on,
            levels,
            dither,
        }
    }

    fn resolution(&self) -> RenderResolution {
        match self.res_mode {
            ResMode::Native => RenderResolution::Native,
            ResMode::Fixed => RenderResolution::Fixed {
                w: self.fixed_w,
                h: self.fixed_h,
            },
            ResMode::Scale => RenderResolution::Scale(self.scale),
        }
    }

    fn posterize(&self) -> Option<PosterizeConfig> {
        self.posterize_on
            .then(|| PosterizeConfig::uniform(self.levels, self.dither))
    }

    /// Push the whole pipeline state to the renderer.
    fn apply(&self, renderer: &mut SceneRenderer) {
        renderer.set_render_resolution(self.resolution());
        renderer.set_ssaa(self.ssaa);
        renderer.set_posterize(self.posterize());
    }
}

pub struct Host {
    // Field order matters for teardown: the renderer owns the wgpu
    // surface/device, which must drop *before* the window they were created
    // from. Rust drops fields top-to-bottom, so `renderer` is declared before
    // `window` — this keeps the order correct even on the panic-unwind path
    // where `exiting` (the graceful teardown) never runs.
    renderer: Option<SceneRenderer>,
    window: Option<Arc<Window>>,
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
    /// RP.3 — live render-pipeline settings edited by the HUD panel.
    pipe: PipelineUi,
    title_base: String,
    fps_frames: u32,
    fps_last: Instant,
    last_fps: f64,
    /// BK.7 — README-gallery capture mode (`ROXLAP_CAPTURE=<dir>`): a
    /// PPM frame is written every `ROXLAP_CAPTURE_MS` (default 80) of
    /// wall time until `ROXLAP_CAPTURE_FRAMES` (default 40) are down,
    /// then the process exits. HUD forced off. Assemble with
    /// `magick -delay 8 <dir>/frame_*.ppm -layers Optimize out.gif`.
    capture_dir: Option<std::path::PathBuf>,
    capture_ms: u64,
    capture_left: u32,
    capture_idx: u32,
    capture_last: Instant,
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
            Box::new(TransparencyScene::new()),
            Box::new(LightingScene::new()),
            Box::new(SpotlightScene::new()),
            Box::new(ScaleScene::new()),
            Box::new(ParticlesScene::new()),
            Box::new(AudioScene::new()),
            Box::new(DoomScene::new()),
            Box::new(DecksScene::new()),
            Box::new(PickingScene::new()),
            Box::new(PrimitivesScene::new()),
            Box::new(EmptyScene::new()),
        ];
        // PS.4 — optional initial scene by menu name (case-insensitive),
        // e.g. `ROXLAP_SCENE=Particles`; unknown names note-and-fall-back
        // to the first scene.
        let active = std::env::var("ROXLAP_SCENE").map_or(0, |want| {
            scenes
                .iter()
                .position(|s| s.name().eq_ignore_ascii_case(&want))
                .unwrap_or_else(|| {
                    eprintln!("ROXLAP_SCENE={want:?} matches no scene; starting on scene 0");
                    0
                })
        });
        // BK.7 — `ROXLAP_CAMERA=x,y,z,yaw,pitch` overrides the scene's
        // start pose (gallery-capture framing, pose-pinned repros).
        let pose = std::env::var("ROXLAP_CAMERA")
            .ok()
            .and_then(|v| {
                let n: Vec<f64> = v.split(',').filter_map(|p| p.trim().parse().ok()).collect();
                (n.len() == 5).then(|| CameraPose {
                    pos: [n[0], n[1], n[2]],
                    yaw: n[3],
                    pitch: n[4],
                })
            })
            .unwrap_or_else(|| scenes[active].start_pose());
        let cam = CameraRig::from_pose(pose);

        // BK.7 — gallery capture mode (see the field docs).
        let capture_dir = std::env::var_os("ROXLAP_CAPTURE").map(std::path::PathBuf::from);
        if let Some(d) = &capture_dir {
            std::fs::create_dir_all(d).expect("ROXLAP_CAPTURE dir must be creatable");
        }
        let env_u64 = |k: &str, default: u64| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(default)
        };

        Self {
            renderer: None,
            window: None,
            engine,
            cam,
            input: InputState::default(),
            grabbed: false,
            look_accum: (0.0, 0.0),
            scan_dist: SCAN_DIST_INITIAL,
            scenes,
            active,
            pending_switch: None,
            menu_open: false,
            last_frame: Instant::now(),
            egui_ctx: egui::Context::default(),
            egui_state: None,
            // Captures want clean frames — no HUD panels baked in.
            hud_on: capture_dir.is_none(),
            pipe: PipelineUi::from_env(),
            title_base: "roxlap-scene-demo".to_string(),
            fps_frames: 0,
            fps_last: Instant::now(),
            last_fps: 0.0,
            capture_ms: env_u64("ROXLAP_CAPTURE_MS", 80),
            #[allow(clippy::cast_possible_truncation)]
            capture_left: env_u64("ROXLAP_CAPTURE_FRAMES", 40) as u32,
            capture_idx: 0,
            capture_last: Instant::now(),
            capture_dir,
        }
    }

    /// BK.7 — write the frame `redraw` armed via `request_capture` as
    /// a P6 PPM into the capture dir; exit once the budget is spent.
    fn save_capture(&mut self) {
        let Some(dir) = self.capture_dir.clone() else {
            return;
        };
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };
        let Some((px, w, h)) = renderer.take_capture() else {
            return;
        };
        let mut ppm = format!("P6\n{w} {h}\n255\n").into_bytes();
        ppm.reserve(px.len() * 3);
        for p in &px {
            #[allow(clippy::cast_possible_truncation)]
            ppm.extend_from_slice(&[(p >> 16) as u8, (p >> 8) as u8, *p as u8]);
        }
        let path = dir.join(format!("frame_{:04}.ppm", self.capture_idx));
        if let Err(e) = std::fs::write(&path, ppm) {
            eprintln!("capture: writing {} failed: {e}", path.display());
        }
        self.capture_idx += 1;
        self.capture_left -= 1;
        if self.capture_left == 0 {
            eprintln!(
                "capture: done, {} frames in {}",
                self.capture_idx,
                dir.display()
            );
            self.teardown();
            std::process::exit(0);
        }
    }

    /// Tear the renderer + window down in the correct order for a clean GPU
    /// shutdown: drain in-flight GPU work, then drop the renderer (releasing
    /// the wgpu device/queue/surface), then the egui state, then the window.
    /// Dropping the surface/device before the window — with the queue idle and
    /// no acquired frame — is what keeps an exit from leaving the driver /
    /// compositor showing stale buffers (the leftover-triangles/flicker bug).
    /// Idempotent: safe if already torn down.
    fn teardown(&mut self) {
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.wait_idle();
        }
        self.renderer = None;
        self.egui_state = None;
        self.window = None;
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
        renderer.clear_sprites();
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

        // BK.7 — arm a capture before the render when the gallery
        // recorder is on and the wall-clock interval elapsed; the
        // matching `save_capture` runs after `present_frame`.
        let capture_this = self.capture_dir.is_some()
            && self.capture_left > 0
            && now.duration_since(self.capture_last).as_millis() >= u128::from(self.capture_ms);
        if capture_this {
            self.capture_last = now;
            if let Some(r) = self.renderer.as_mut() {
                r.request_capture();
            }
        }

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
        if capture_this {
            self.save_capture();
        }

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
        // RP.3 — live render-pipeline controls. Edit a local copy, then push
        // any change to the renderer after the pass (keeps the `renderer`
        // borrow disjoint from `self.pipe`).
        let mut pipe = self.pipe;
        pipeline_panel(
            &self.egui_ctx,
            &mut pipe,
            renderer.logical_dims(),
            renderer.render_dims(),
        );
        if self.menu_open {
            if let Some(pick) = scene_menu(&self.egui_ctx, &names, self.active) {
                self.pending_switch = Some(pick);
                self.menu_open = false;
            }
        }

        let full = self.egui_ctx.end_pass();
        state.handle_platform_output(window, full.platform_output);
        if pipe != self.pipe {
            pipe.apply(renderer);
            self.pipe = pipe;
        }
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

        // QE.7b - BackendPreference replaces want_gpu.
        let backend = if std::env::var_os("ROXLAP_GPU").is_some_and(|v| v != "0" && !v.is_empty()) {
            BackendPreference::PreferGpu
        } else {
            BackendPreference::Cpu
        };
        let opts = RenderOptions {
            backend,
            ..RenderOptions::default()
        };
        let init = window.inner_size();
        let mut renderer = SceneRenderer::new(window.clone(), (init.width, init.height), &opts);

        // The default fixed grid targets a discrete GPU; on modest
        // hardware (CPU backend / integrated GPU) halve it so the demos
        // stay interactive. An explicit `ROXLAP_RENDER_RES` always wins,
        // and the HUD panel can still raise it live.
        if std::env::var_os("ROXLAP_RENDER_RES").is_none()
            && renderer.is_low_power()
            && self.pipe.res_mode == ResMode::Fixed
            && (self.pipe.fixed_w, self.pipe.fixed_h) == (RENDER_RES_W, RENDER_RES_H)
        {
            self.pipe.fixed_w = RENDER_RES_W / 2;
            self.pipe.fixed_h = RENDER_RES_H / 2;
            eprintln!(
                "roxlap-scene-demo: low-power renderer — render resolution defaults to {}×{}",
                self.pipe.fixed_w, self.pipe.fixed_h
            );
        }

        // RP.0/1/2 — apply the render-pipeline settings (seeded from the
        // `ROXLAP_RENDER_RES`/`SSAA`/`POSTERIZE`/`DITHER` env vars in
        // `Host::new`, then live-editable from the HUD's "Render pipeline"
        // panel): fixed logical grid + SSAA + posterize.
        self.pipe.apply(&mut renderer);
        let (lw, lh) = renderer.logical_dims();
        let (rw, rh) = renderer.render_dims();
        eprintln!(
            "roxlap-render: render resolution {:?} → {lw}×{lh} logical, ssaa {} → {rw}×{rh} march; posterize {:?}",
            self.pipe.resolution(),
            self.pipe.ssaa,
            self.pipe.posterize()
        );

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

    /// Graceful shutdown: winit calls this once the event loop is told to
    /// exit (`event_loop.exit()` from a close/Esc). Tear the GPU down cleanly
    /// here so an exit never yanks the swapchain mid-frame.
    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        self.teardown();
    }

    /// The platform asked us to release the window/surface (Android-style;
    /// rare on desktop). Drop the GPU resources cleanly too — `resumed`
    /// rebuilds them. Same clean-teardown path as `exiting`.
    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        self.teardown();
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

/// RP.3 — the live render-pipeline panel (top-right). Mutates `p` in place;
/// the caller diffs it against the previous state and pushes any change to the
/// renderer. `logical`/`march` are shown for orientation.
fn pipeline_panel(ctx: &egui::Context, p: &mut PipelineUi, logical: (u32, u32), march: (u32, u32)) {
    egui::Window::new("Render pipeline")
        .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-8.0, 8.0))
        .default_open(false)
        .resizable(false)
        .show(ctx, |ui| {
            ui.label(format!(
                "logical {}×{}  ·  march {}×{}",
                logical.0, logical.1, march.0, march.1
            ));
            ui.separator();

            ui.label("Resolution (RP.0)");
            ui.horizontal(|ui| {
                ui.radio_value(&mut p.res_mode, ResMode::Native, "Native");
                ui.radio_value(&mut p.res_mode, ResMode::Fixed, "Fixed");
                ui.radio_value(&mut p.res_mode, ResMode::Scale, "Scale");
            });
            match p.res_mode {
                ResMode::Fixed => {
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::DragValue::new(&mut p.fixed_w)
                                .range(64..=4096)
                                .prefix("w "),
                        );
                        ui.add(
                            egui::DragValue::new(&mut p.fixed_h)
                                .range(64..=4096)
                                .prefix("h "),
                        );
                    });
                }
                ResMode::Scale => {
                    ui.add(egui::Slider::new(&mut p.scale, 0.1..=1.0).text("scale"));
                }
                ResMode::Native => {}
            }
            ui.separator();

            ui.label("SSAA (RP.1)");
            ui.add(egui::Slider::new(&mut p.ssaa, 1..=4).text("×N² rays"));
            ui.separator();

            ui.checkbox(&mut p.posterize_on, "Posterize (RP.2)");
            if p.posterize_on {
                ui.add(egui::Slider::new(&mut p.levels, 2..=16).text("levels/ch"));
                egui::ComboBox::from_label("dither")
                    .selected_text(match p.dither {
                        DitherMode::None => "none",
                        DitherMode::Bayer4x4 => "bayer 4×4",
                        DitherMode::BlueNoise => "blue noise",
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut p.dither, DitherMode::None, "none");
                        ui.selectable_value(&mut p.dither, DitherMode::Bayer4x4, "bayer 4×4");
                        ui.selectable_value(&mut p.dither, DitherMode::BlueNoise, "blue noise");
                    });
            }
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
