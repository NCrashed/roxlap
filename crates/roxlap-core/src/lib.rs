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
pub mod column_walk;
mod engine;
pub mod fixed;
pub mod gline;
pub mod grouscan;
pub mod opticast;
pub mod opticast_prelude;
pub mod projection;
pub mod rasterizer;
pub mod ray_step;
pub mod scalar_rasterizer;
pub mod scan_loops;
pub mod world_query;

pub use camera::Camera;
pub use engine::Engine;
pub use opticast::{opticast, OpticastOutcome, OpticastSettings};
