//! roxlap engine core.
//!
//! A pure-Rust port of Ken Silverman's Voxlap voxel engine. See
//! `PORTING-RUST.md` at the workspace root for the substage roadmap.
//!
//! Stage R3 lands the public [`Engine`] / [`Camera`] surface with a
//! sky-fill stub renderer. R4 replaces the stub with the full
//! opticast + grouscan algorithm.

mod camera;
pub mod camera_math;
pub(crate) mod column_walk;
pub mod drawtile;
mod engine;
pub(crate) mod equivec;
pub(crate) mod fixed;
pub mod gline;
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
pub use engine::{Engine, LightSrc, DEFAULT_KV6COL};
pub use opticast::{opticast, OpticastOutcome, OpticastSettings};
pub use world_lighting::update_lighting;
