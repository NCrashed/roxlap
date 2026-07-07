//! Selectable demo scenes (stage DS) — each a [`DemoScene`] showcasing one
//! cluster of engine features, picked from the host's scene menu.
//!
//! DS.0 ships the `Empty` placeholder; the focused scenes (World, Sprites,
//! Animation, Picking, Primitives) land in DS.1–DS.4.
//!
//! [`DemoScene`]: crate::scene_api::DemoScene

pub mod animation;
pub mod doom;
pub mod empty;
pub mod lighting;
pub mod particles;
pub mod picking;
pub mod primitives;
pub mod scale;
pub mod spotlight;
pub mod sprites;
pub mod transparency;
pub mod world;
