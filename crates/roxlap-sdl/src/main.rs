//! roxlap-sdl — SDL2 demo host.
//!
//! Stage R3: opens a window, allocates a streaming ARGB texture, and
//! per-frame asks the [`Engine`] to render into it. Today's render is a
//! sky-blue fill; R4 will swap in the actual rasterizer behind the
//! same call.
//!
//! Controls:
//! - `Esc` or window close → exit.

use roxlap_core::Engine;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::pixels::PixelFormatEnum;

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sdl = sdl2::init()?;
    let video = sdl.video()?;
    let window = video
        .window("roxlap-sdl (R3 stub)", WIDTH, HEIGHT)
        .position_centered()
        .build()?;
    let mut canvas = window.into_canvas().present_vsync().build()?;
    let texture_creator = canvas.texture_creator();
    let mut texture =
        texture_creator.create_texture_streaming(PixelFormatEnum::ARGB8888, WIDTH, HEIGHT)?;

    let mut engine = Engine::new();
    let mut events = sdl.event_pump()?;

    'main: loop {
        for ev in events.poll_iter() {
            match ev {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => break 'main,
                _ => {}
            }
        }

        texture.with_lock(None, |bytes: &mut [u8], pitch_bytes: usize| {
            let pitch_pixels = u32::try_from(pitch_bytes / 4).expect("SDL2 pitch fits in u32");
            let pixels = argb_pixels_mut(bytes);
            engine.render(pixels, WIDTH, HEIGHT, pitch_pixels);
        })?;

        canvas.clear();
        canvas.copy(&texture, None, None)?;
        canvas.present();
    }

    Ok(())
}

/// Reinterpret an SDL2 `with_lock` byte buffer as a slice of u32 ARGB
/// pixels. Safe under SDL2's guarantees: ARGB8888 textures are aligned
/// to 4 bytes and have a row pitch that is always a multiple of 4.
//
// clippy::cast_ptr_alignment fires because *u8 → *u32 is a strict-
// alignment increase in the abstract; the runtime asserts below prove
// alignment and length, so this cast is sound for SDL2's pixel buffers.
#[allow(clippy::cast_ptr_alignment)]
fn argb_pixels_mut(bytes: &mut [u8]) -> &mut [u32] {
    assert_eq!(bytes.len() % 4, 0, "ARGB buffer must have len % 4 == 0");
    assert_eq!(
        bytes.as_ptr().align_offset(core::mem::align_of::<u32>()),
        0,
        "ARGB buffer must be u32-aligned",
    );
    // SAFETY: the asserts above prove the slice is valid for the cast.
    // The lifetime of the returned slice is tied to `bytes` via the
    // borrow checker — it ends when this function returns.
    unsafe { core::slice::from_raw_parts_mut(bytes.as_mut_ptr().cast::<u32>(), bytes.len() / 4) }
}
