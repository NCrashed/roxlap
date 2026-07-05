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

// ANCHOR: packed_color
/// Pack RGB into roxlap's voxel colour format. The low 24 bits are
/// the colour; the high byte is the per-voxel *shading intensity* —
/// NOT alpha. `0x80` is the neutral "unlit" default; lighting bakes
/// rewrite that byte per voxel (see the lighting chapter).
fn voxel_color(r: u8, g: u8, b: u8) -> u32 {
    0x80 << 24 | u32::from(r) << 16 | u32::from(g) << 8 | u32::from(b)
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
    assert_eq!(voxel_color(0x4d, 0x8a, 0x3a), 0x80_4d_8a_3a);

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
