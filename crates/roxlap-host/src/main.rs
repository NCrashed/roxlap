//! roxlap-host — winit + softbuffer demo host.
//!
//! Stage R4.3a: opens a window, allocates a softbuffer surface, and on
//! every `RedrawRequested` event runs `opticast` with the
//! `ScalarRasterizer` and the placeholder gline. The visible scene
//! pixels show as magenta on a sky-blue background; once R4.3b+
//! lands the real grouscan ray-cast, the magenta is replaced with
//! actual voxel colours.
//!
//! Controls:
//! - `Esc` or window close → exit.

use std::num::NonZeroU32;
use std::rc::Rc;

use roxlap_core::opticast;
use roxlap_core::rasterizer::ScanScratch;
use roxlap_core::scalar_rasterizer::ScalarRasterizer;
use roxlap_core::Camera;
use roxlap_core::Engine;
use roxlap_core::OpticastSettings;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;
const VSID: u32 = 2048;

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
    /// Synthetic single-slab column the placeholder gline ignores
    /// but `opticast` checks via `camera_column_air_gap`. Moves up
    /// the priority list once R4.3b loads a real `.vxl` world.
    column_data: Vec<u8>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_title("roxlap (R4.3a — magenta placeholder)")
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
                    self.scratch = ScanScratch::new_for_size(size.width, size.height, VSID);
                }

                let mut buffer = surface.buffer_mut().expect("softbuffer: buffer_mut");
                // Pre-fill with sky-blue so any pixel opticast leaves
                // untouched reads as sky.
                let sky = self.engine.sky_color();
                for px in buffer.iter_mut() {
                    *px = sky;
                }

                // Looking-down camera so all four scan-quadrants engage
                // and the magenta placeholder fills a noticeable
                // region of the screen. Replaced with engine-managed
                // camera state once R4.3b+ has a real world to look at.
                let cam = Camera {
                    pos: [1024.0, 1024.0, 128.0],
                    right: [1.0, 0.0, 0.0],
                    down: [0.0, 1.0, 0.0],
                    forward: [0.0, 0.0, 1.0],
                };

                let settings = OpticastSettings::for_oracle_framebuffer(size.width, size.height);
                let pitch_pixels = size.width as usize;
                // Scope the rasterizer so its &mut buffer borrow ends
                // before we present the buffer.
                {
                    let mut rasterizer =
                        ScalarRasterizer::new(&mut buffer, &mut self.zbuffer, pitch_pixels);
                    let _ = opticast(
                        &mut rasterizer,
                        &mut self.scratch,
                        &cam,
                        &settings,
                        VSID,
                        &self.column_data,
                    );
                }
                buffer.present().expect("softbuffer: present");
            }

            _ => {}
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Wait);
    let mut app = App {
        window: None,
        surface: None,
        engine: Engine::new(),
        zbuffer: Vec::new(),
        scratch: ScanScratch::new_for_size(WIDTH, HEIGHT, VSID),
        // Single-slab synthetic column at z = 200..254. The placeholder
        // gline doesn't read this, but opticast's camera_column_air_gap
        // walks it to confirm the camera (z = 128) is in air above
        // the slab; otherwise opticast would early-out and skip
        // rendering.
        column_data: vec![0u8, 200, 254, 0],
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}
