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
mod engine;

pub use camera::Camera;
pub use engine::Engine;
