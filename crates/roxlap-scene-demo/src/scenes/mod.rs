//! Selectable demo scenes (stage DS) — each a [`DemoScene`] showcasing one
//! cluster of engine features, picked from the host's scene menu.
//!
//! DS.0 ships the `Empty` placeholder; the focused scenes (World, Sprites,
//! Animation, Picking, Primitives) land in DS.1–DS.4.
//!
//! [`DemoScene`]: crate::scene_api::DemoScene

#![allow(dead_code)] // wired into the host in DS.1

pub mod empty;
