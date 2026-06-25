//! roxlap engine core.
//!
//! A Rust voxel engine. The CPU renderer is a clean-room per-pixel
//! 3D-DDA over an 8³ brickmap ([`dda`] + [`dda_sprite`]); it reads the
//! voxlap-compatible `.vxl` / KV6 / KFA formats from `roxlap-formats`
//! but the rendering algorithm is not voxlap-derived. See
//! `PORTING-RUST.md` at the workspace root for the project history.
//!
//! # World handedness and the horizontal mirror
//!
//! roxlap's world is **z-down**: `x` = east, `y` = north, `z` points
//! *down* into the map. Physically the triple (east, north, up) is
//! left-handed, so a faithfully-projected view is **horizontally
//! mirrored** relative to a real camera at the same heading — the
//! viewer's geometric *right* lands on screen-*left*. This is Voxlap's
//! native convention; it is baked into the projection, the frustum
//! cull, the scan loops, and every bit-exact oracle golden, and is
//! reproduced deliberately. It is **not** a bug.
//!
//! The camera basis the engine actually renders with is
//! right-handed in the `(right, down, forward)` sense
//! (`right × down = +forward`). Build it with
//! [`Camera::from_yaw_pitch`], [`Camera::orbit`], or
//! [`Camera::look_at`] — never by rotating [`Camera::default`], whose
//! placeholder basis is left-handed and will make the sprite cull
//! reject every sprite.
//!
//! ## I want an un-mirrored world
//!
//! Do **not** "fix" the mirror by negating `right` in the basis. That
//! flips the chirality to `right × down = -forward`; the world still
//! renders, but the sprite frustum cull then rejects every sprite
//! (and you diverge from the oracle goldens). The mirror is in the
//! projection, not the basis.
//!
//! Handle it on the **consumer side** instead — pick one and stay
//! consistent:
//!
//! - mirror a single world axis in your scene → world mapping (e.g.
//!   negate world-x when you place content), or
//! - negate your yaw input so headings sweep the opposite way.
//!
//! Either re-establishes the chirality your project expects without
//! touching the engine. (A true engine-side de-mirror would mean
//! negating screen-x in *both* the grid raycaster and the sprite
//! rasteriser while preserving the cull's normal winding — a large,
//! golden-breaking change that diverges from Voxlap, and is out of
//! scope here.)

mod camera;
pub mod camera_math;
pub mod dda;
pub mod dda_sprite;
mod engine;
pub(crate) mod fixed;
pub mod grid_view;
pub mod kfa_draw;
pub mod opticast;
pub mod raster_target;
pub mod ray_aabb;
pub mod sky;
pub mod world_lighting;
pub mod world_query;

pub use camera::Camera;
pub use dda::{
    effective_mip, pixel_ray, render_dda, render_dda_parallel, BrickCache, DdaEnv, PixelSink,
    RasterSink,
};
pub use dda_sprite::{draw_sprite_dda, draw_sprite_dense, ClipFlipbook, SpriteDense};
pub use engine::{Engine, LightSrc, DEFAULT_KV6COL};
pub use grid_view::{ChunkGrid, GridView};
pub use opticast::OpticastSettings;
pub use world_lighting::{
    apply_lighting_with_cache, update_lighting, update_lighting_chunk, EstNormCache,
};
