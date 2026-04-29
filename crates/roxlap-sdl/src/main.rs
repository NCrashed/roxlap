//! roxlap-sdl — winit + softbuffer demo host.
//!
//! Stage R3: opens a window, allocates a softbuffer surface, and on
//! every `RedrawRequested` event asks the [`Engine`] to render into
//! it. Today's render is a sky-blue fill; R4 will swap in the actual
//! rasterizer behind the same call.
//!
//! Controls:
//! - `Esc` or window close → exit.

use std::num::NonZeroU32;
use std::rc::Rc;

use roxlap_core::Engine;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

struct App {
    /// Window handle. Wrapped in `Rc` because softbuffer's `Context`
    /// and `Surface` each take a clone — both need the same handle
    /// type, so we share.
    window: Option<Rc<Window>>,
    surface: Option<softbuffer::Surface<Rc<Window>, Rc<Window>>>,
    engine: Engine,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_title("roxlap (R3 stub)")
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

                let mut buffer = surface.buffer_mut().expect("softbuffer: buffer_mut");
                // softbuffer expects 0x00RRGGBB pixels (high byte
                // ignored). Voxlap's 0x80RRGGBB packing has the
                // brightness bit in the high byte; softbuffer drops
                // it harmlessly, so we hand the engine's u32 buffer
                // straight through.
                self.engine
                    .render(&mut buffer, size.width, size.height, size.width);
                buffer.present().expect("softbuffer: present");
            }

            _ => {}
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new()?;
    // Wait for input/expose events instead of busy-polling. R4's
    // animation/profiling work may want Poll; today's static sky
    // fill doesn't.
    event_loop.set_control_flow(ControlFlow::Wait);
    let mut app = App {
        window: None,
        surface: None,
        engine: Engine::new(),
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}
