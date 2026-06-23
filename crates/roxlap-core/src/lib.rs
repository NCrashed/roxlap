//! roxlap engine core.
//!
//! A pure-Rust port of Ken Silverman's Voxlap voxel engine. See
//! `PORTING-RUST.md` at the workspace root for the substage roadmap.
//!
//! Stage R3 lands the public [`Engine`] / [`Camera`] surface with a
//! sky-fill stub renderer. R4 replaces the stub with the full
//! opticast + grouscan algorithm.
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
pub(crate) mod column_walk;
pub mod dda;
pub mod drawtile;
mod engine;
// `equivec` (voxlap's normal table) moved to roxlap-formats so kv6
// model builders can fill per-voxel `dir` without a circular dep;
// re-exported here so in-crate `crate::equivec::…` paths (the sprite
// `kv6colmul` build) keep resolving.
pub(crate) use roxlap_formats::equivec;
pub(crate) mod fixed;
pub mod gline;
pub mod grid_view;
pub(crate) mod grouscan;
pub mod kfa_draw;
pub mod meltsphere;
pub mod opticast;
pub mod opticast_prelude;
pub(crate) mod projection;
pub(crate) mod ptfaces16;
pub mod rasterizer;
pub mod ray_aabb;
pub(crate) mod ray_step;
pub mod scalar_rasterizer;
pub(crate) mod scan_loops;
pub mod sky;
pub mod sprite;
pub mod world_lighting;
pub mod world_query;

pub use camera::Camera;
pub use dda::{pixel_ray, render_dda, render_dda_parallel, DdaEnv, PixelSink, RasterSink};
pub use engine::{Engine, LightSrc, DEFAULT_KV6COL};
pub use grid_view::{ChunkGrid, GridView};
pub use opticast::{opticast, OpticastOutcome, OpticastSettings};
pub use world_lighting::{
    apply_lighting_with_cache, update_lighting, update_lighting_chunk, EstNormCache,
};
