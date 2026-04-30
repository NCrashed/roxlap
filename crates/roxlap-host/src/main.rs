//! roxlap-host — winit + softbuffer demo host.
//!
//! Stage R4.3a-rewire-4: opens a window, loads a real `.vxl` world
//! (oracle.vxl.gz from the workspace assets), and on every
//! `RedrawRequested` event runs `opticast` with the
//! `ScalarRasterizer`. The real R4.3a-rewire-3b gline + grouscan
//! chain produces voxel colours from the loaded world.
//!
//! Controls:
//! - `Esc` or window close → exit.

use std::io::Read;
use std::num::NonZeroU32;
use std::rc::Rc;

use flate2::read::GzDecoder;
use roxlap_core::opticast;
use roxlap_core::rasterizer::ScanScratch;
use roxlap_core::scalar_rasterizer::ScalarRasterizer;
use roxlap_core::Camera;
use roxlap_core::Engine;
use roxlap_core::OpticastSettings;
use roxlap_formats::vxl;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

/// Embedded gzipped oracle world. Same fixture roxlap-formats uses
/// in its parser tests — no extra disk I/O at startup.
const ORACLE_VXL_GZ: &[u8] = include_bytes!("../../../assets/oracle.vxl.gz");

struct App {
    /// Window handle. Wrapped in `Rc` because softbuffer's `Context`
    /// and `Surface` each take a clone — both need the same handle
    /// type, so we share.
    window: Option<Rc<Window>>,
    surface: Option<softbuffer::Surface<Rc<Window>, Rc<Window>>>,
    engine: Engine,
    /// f32 z-buffer, allocated lazily / re-sized on first redraw and
    /// resized on window-resize.
    zbuffer: Vec<f32>,
    /// `ScanScratch` (radar / angstart / lastx / uurend), reused across
    /// frames. Sized at app construction for the initial window
    /// resolution; resized on window-resize.
    scratch: ScanScratch,
    /// World loaded from `oracle.vxl.gz`. `vxl.data` is the flat
    /// slab buffer, `vxl.column_offset` the per-column byte offsets;
    /// `vxl.vsid` the world dimension; `vxl.ipo`/`ist`/`ihe`/`ifo`
    /// the saved camera.
    vxl: vxl::Vxl,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_title("roxlap (R4.3a-rewire-4 — oracle.vxl)")
            .with_inner_size(LogicalSize::new(f64::from(WIDTH), f64::from(HEIGHT)));
        let window = Rc::new(
            event_loop
                .create_window(attrs)
                .expect("winit: create_window"),
        );
        let context = softbuffer::Context::new(window.clone()).expect("softbuffer: Context::new");
        let surface =
            softbuffer::Surface::new(&context, window.clone()).expect("softbuffer: Surface::new");
        self.window = Some(window);
        self.surface = Some(surface);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Named(NamedKey::Escape),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => event_loop.exit(),

            WindowEvent::RedrawRequested => {
                let (Some(window), Some(surface)) = (self.window.as_ref(), self.surface.as_mut())
                else {
                    return;
                };
                let size = window.inner_size();
                let (Some(w_nz), Some(h_nz)) =
                    (NonZeroU32::new(size.width), NonZeroU32::new(size.height))
                else {
                    return;
                };
                surface.resize(w_nz, h_nz).expect("softbuffer: resize");

                // Make sure the zbuffer + scratch fit this frame's
                // resolution. Cheap when unchanged.
                let pixel_count = (size.width as usize) * (size.height as usize);
                if self.zbuffer.len() < pixel_count {
                    self.zbuffer.resize(pixel_count, 0.0);
                }
                if self.scratch.uurend_half_stride < size.width as usize {
                    self.scratch =
                        ScanScratch::new_for_size(size.width, size.height, self.vxl.vsid);
                }

                // Wire engine sky colour onto scratch so grouscan's
                // startsky has the right (col, dist) for any radar
                // slot it drains. The `dist` argument here seeds an
                // initial value; gline (R4.4) overwrites
                // `scratch.skycast.dist` per ray based on the
                // frustum-edge clip outcome — `gxmax` when the ray
                // is bounded by scan-distance, `i32::MAX` when by
                // world edge.
                // u32 → i32 reinterpret (preserves bits; `cast_signed`
                // is 1.87+, beyond the workspace MSRV).
                let sky_col_i = i32::from_ne_bytes(self.engine.sky_color().to_ne_bytes());
                self.scratch.set_skycast(sky_col_i, 0);

                // Engine fog → ScanScratch foglut. Rebuilds the
                // 2048-entry table only when fog params change in
                // practice (set_fog itself is the only producer);
                // the foglut alloc lives across frames so this is
                // cheap on steady-state.
                let fog_col_i = i32::from_ne_bytes(self.engine.fog_color().to_ne_bytes());
                self.scratch
                    .set_fog(fog_col_i, self.engine.fog_max_scan_dist());

                let mut buffer = surface.buffer_mut().expect("softbuffer: buffer_mut");
                // Pre-fill with sky-blue so any pixel opticast leaves
                // untouched reads as sky.
                let sky = self.engine.sky_color();
                for px in buffer.iter_mut() {
                    *px = sky;
                }

                // Looking-down camera at the .vxl's saved position.
                // Oracle's saved basis is `ist=[1,0,0]`,
                // `ihe=[0,0,1]`, `ifo=[0,1,0]` — a level
                // look-along-y. That basis makes
                // `derive_gline_frustum` collapse per-scanline (vd0
                // == vd1 because right[1] = down[0] = forward[2] =
                // 0), which makes `gi0 = (vd0 - vd1)/leng = 0`,
                // which freezes drawflor's cross-sign test on the
                // first iteration → no floor pixels written, only
                // sky. The oracle binary's actual rendered poses
                // (oracle.c:261-271) use a yawed/pitched basis
                // computed per-frame; the saved camera is a
                // placeholder for `loadvxl`. Until R4.4 ships a
                // proper engine-level camera-pose API, the demo
                // host uses an unambiguously non-degenerate
                // looking-down basis and keeps the saved position
                // (which is in air above the playable carve).
                let cam = Camera {
                    pos: self.vxl.ipo,
                    right: [1.0, 0.0, 0.0],
                    down: [0.0, 1.0, 0.0],
                    forward: [0.0, 0.0, 1.0],
                };

                let settings = OpticastSettings::for_oracle_framebuffer(size.width, size.height);
                let pitch_pixels = size.width as usize;
                // Scope the rasterizer so its &mut buffer borrow ends
                // before we present the buffer.
                {
                    let mut rasterizer = ScalarRasterizer::new(
                        &mut buffer,
                        &mut self.zbuffer,
                        pitch_pixels,
                        &self.vxl.data,
                        &self.vxl.column_offset,
                        self.vxl.vsid,
                    );
                    let _ = opticast(
                        &mut rasterizer,
                        &mut self.scratch,
                        &cam,
                        &settings,
                        self.vxl.vsid,
                        &self.vxl.data,
                        &self.vxl.column_offset,
                    );
                }
                buffer.present().expect("softbuffer: present");
            }

            _ => {}
        }
    }
}

/// Decompress + parse the embedded oracle world. Errors are fatal:
/// a malformed asset means the binary is broken, not a runtime
/// recoverable state.
fn load_oracle_vxl() -> vxl::Vxl {
    let mut decoder = GzDecoder::new(ORACLE_VXL_GZ);
    let mut bytes = Vec::with_capacity(40 * 1024 * 1024);
    decoder
        .read_to_end(&mut bytes)
        .expect("gunzip oracle.vxl.gz");
    vxl::parse(&bytes).expect("parse oracle.vxl")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Wait);

    let vxl_world = load_oracle_vxl();
    let initial_scratch = ScanScratch::new_for_size(WIDTH, HEIGHT, vxl_world.vsid);

    let mut app = App {
        window: None,
        surface: None,
        engine: Engine::new(),
        zbuffer: Vec::new(),
        scratch: initial_scratch,
        vxl: vxl_world,
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}
