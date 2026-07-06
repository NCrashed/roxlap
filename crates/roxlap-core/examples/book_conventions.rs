//! Companion example for the book's "Concepts & conventions" chapter
//! (`docs/book/src/concepts.md`) — the chapter pulls its snippets from
//! here via `// ANCHOR:` markers, so everything it shows compiles and
//! its assertions actually ran. Headless: no window, no renderer.
//!
//! ```sh
//! cargo run -p roxlap-core --example book_conventions
//! ```
//!
//! Keep the anchors when editing; `docs/book/check-anchors.sh` (run by
//! the CI `book` job) goes red if one disappears.

use roxlap_core::Camera;
use roxlap_formats::{OverlayColor, Rgb, VoxColor};

// ANCHOR: packed_color
fn packed_colors() {
    // The QE-B6 colour family: three packings, three types — mixing
    // them up is a compile error, not an over-bright voxel.
    // VoxColor = RGB + a brightness byte (NOT alpha; 0x80 = neutral,
    // lighting bakes rewrite it per voxel — see the lighting chapter).
    let grass = VoxColor::rgb(0x4d, 0x8a, 0x3a);
    assert_eq!(grass.0, 0x80_4d_8a_3a); // .0 is the raw wire word
    assert_eq!(grass.with_brightness(0xff).0, 0xff_4d_8a_3a);
    // Rgb = plain colour: tints, sky/fog, material-map keys.
    assert_eq!(Rgb::new(0x8f, 0xbc, 0xd4).0, 0x00_8f_bc_d4);
    // OverlayColor = REAL alpha — only the overlay-line API uses it.
    assert_eq!(OverlayColor::rgba(0xff, 0xd0, 0x40, 0x80).0, 0x80_ff_d0_40);
}
// ANCHOR_END: packed_color

/// `a × b`, for checking basis chirality by hand.
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn approx_eq(a: [f64; 3], b: [f64; 3]) -> bool {
    a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-12)
}

fn main() {
    // The grass colour the quickstart uses, packed by hand.
    packed_colors();

    // ANCHOR: z_down
    // +z points DOWN. Positive pitch aims the camera downward (the
    // forward axis gains a positive z component); "up" in the world
    // is toward smaller z.
    // A camera pitched down has forward pointing at +z:
    let cam = Camera::from_yaw_pitch([0.0, 0.0, 128.0], 0.0, 0.4);
    assert!(cam.forward[2] > 0.0);
    // ANCHOR_END: z_down

    // ANCHOR: camera_basis
    // The canonical constructors (`from_yaw_pitch` / `orbit` /
    // `look_at`) produce the right-handed basis the engine requires:
    // right × down == +forward. The sprite frustum cull depends on
    // this chirality — a hand-rolled basis that gets it backwards
    // renders the terrain fine and silently culls every sprite.
    let cam = Camera::from_yaw_pitch([0.0, 0.0, 128.0], 0.6, 0.2);
    assert!(approx_eq(cross(cam.right, cam.down), cam.forward));

    // `Camera::default()` is the trap: its placeholder basis (from
    // the .vxl header convention) is LEFT-handed. Never build an
    // interactive camera by rotating `default()` — construct one.
    let trap = Camera::default();
    let anti = cross(trap.right, trap.down);
    // right × down == -forward here: the wrong chirality.
    assert!(approx_eq(anti, [0.0, -1.0, 0.0]));
    // ANCHOR_END: camera_basis

    println!("book_conventions: all convention assertions hold");
}
