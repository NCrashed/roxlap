//! QE-C6 — the single site where the render facade reads `ROXLAP_*`
//! process-environment overrides.
//!
//! Every knob here has a programmatic [`RenderOptions`] counterpart;
//! the env var is a user-side escape hatch (re-tune a shipped binary
//! from the shell without a rebuild). Overrides are resolved **once**,
//! at `SceneRenderer` construction, and never read per frame — the
//! PR.1 history lesson: an uncached `env::var_os` in a hot loop once
//! cost 3× FPS. The variable table lives on [`RenderOptions`]'s
//! rustdoc (the type users actually read); keep the two in sync.
//!
//! Demo/app-level variables (`ROXLAP_GPU`, `ROXLAP_SCENE`,
//! `ROXLAP_CAPTURE*`, `ROXLAP_STATIC`, …) are parsed by their host
//! binaries, not by the library — a library crate deciding backend
//! selection from the environment behind the host's back is exactly
//! the sprawl this module closes out.

use crate::RenderOptions;
use roxlap_gpu::PowerPreference;

/// The `ROXLAP_*` overrides found in the process environment, each
/// `None` when unset (or unparseable, or on wasm — no environment).
// The shared `gpu_` prefix is deliberate: each field spells the
// `RenderOptions` field it overrides.
#[allow(clippy::struct_field_names)]
#[derive(Default)]
pub(crate) struct EnvOverrides {
    /// `ROXLAP_GPU_POWER=low|high` → `RenderOptions::gpu.power_preference`.
    pub gpu_power: Option<PowerPreference>,
    /// `ROXLAP_GPU_SPRITE_LOD_PX` → [`RenderOptions::gpu_sprite_lod_px`].
    pub gpu_sprite_lod_px: Option<f32>,
    /// `ROXLAP_GPU_MIP_SCAN_DIST` → [`RenderOptions::gpu_mip_scan_dist`].
    pub gpu_mip_scan_dist: Option<f32>,
    /// `ROXLAP_GPU_CHUNK_BUDGET` → [`RenderOptions::gpu_chunk_upload_budget`].
    pub gpu_chunk_upload_budget: Option<u32>,
    /// `ROXLAP_GPU_CLIP_BUDGET` → [`RenderOptions::gpu_clip_upload_budget`].
    pub gpu_clip_upload_budget: Option<u32>,
}

/// Read one env var, parsed; unparseable values are dropped with a
/// `log::warn!` (construction-time only, so the cost is nil) instead
/// of silently falling back — a typo'd override that quietly does
/// nothing is the worst failure mode an escape hatch can have.
#[cfg(not(target_arch = "wasm32"))]
fn parse_var<T: std::str::FromStr>(var: &str) -> Option<T> {
    let raw = std::env::var(var).ok()?;
    match raw.parse() {
        Ok(v) => Some(v),
        Err(_) => {
            log::warn!("ignoring unparseable env override {var}={raw}");
            None
        }
    }
}

impl EnvOverrides {
    /// Snapshot the recognised `ROXLAP_*` variables. On wasm there is
    /// no process environment: every override is `None` and the
    /// [`RenderOptions`] values apply as-is.
    pub(crate) fn from_env() -> Self {
        #[cfg(target_arch = "wasm32")]
        {
            Self::default()
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let gpu_power = match std::env::var("ROXLAP_GPU_POWER").as_deref() {
                Ok("low") => Some(PowerPreference::Low),
                Ok("high") => Some(PowerPreference::High),
                Ok(other) => {
                    log::warn!("ignoring unparseable env override ROXLAP_GPU_POWER={other}");
                    None
                }
                Err(_) => None,
            };
            Self {
                gpu_power,
                gpu_sprite_lod_px: parse_var("ROXLAP_GPU_SPRITE_LOD_PX"),
                gpu_mip_scan_dist: parse_var("ROXLAP_GPU_MIP_SCAN_DIST"),
                gpu_chunk_upload_budget: parse_var("ROXLAP_GPU_CHUNK_BUDGET"),
                gpu_clip_upload_budget: parse_var("ROXLAP_GPU_CLIP_BUDGET"),
            }
        }
    }

    /// `opts` with every present override applied — the env wins over
    /// the programmatic value, field by field.
    pub(crate) fn applied(&self, opts: &RenderOptions) -> RenderOptions {
        let mut o = opts.clone();
        if let Some(p) = self.gpu_power {
            o.gpu.power_preference = p;
        }
        if let Some(v) = self.gpu_sprite_lod_px {
            o.gpu_sprite_lod_px = v;
        }
        if let Some(v) = self.gpu_mip_scan_dist {
            o.gpu_mip_scan_dist = v;
        }
        if let Some(v) = self.gpu_chunk_upload_budget {
            o.gpu_chunk_upload_budget = v;
        }
        if let Some(v) = self.gpu_clip_upload_budget {
            o.gpu_clip_upload_budget = v;
        }
        o
    }
}
