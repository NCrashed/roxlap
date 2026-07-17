//! FW.4 — the shared fog-of-war sprite-cull context, resolved ONCE per
//! frame and used by BOTH backends (`cpu.rs` and `gpu.rs`). Review perf
//! #2: the resolve + per-sprite verdict were duplicated verbatim across
//! the two paths — exactly the mechanism that produced the CPU/GPU
//! divergences earlier review rounds caught. One helper, one rule.

use roxlap_scene::{FogOfWar, GridId, Scene};

/// The per-frame fog-of-war sprite hide test for the fog grid: its
/// transform + footprint + mask, resolved from `FrameParams::fow`.
pub(crate) struct FogSpriteCull<'a> {
    fow: &'a FogOfWar,
    transform: roxlap_scene::GridTransform,
    /// Grid-local cell bounds `[lo, hi)` of the ship (open space outside
    /// is never hidden — hazard 3 / review #1).
    footprint: (glam::IVec2, glam::IVec2),
}

impl<'a> FogSpriteCull<'a> {
    /// Resolve from `FrameParams::fow` + the scene. `None` when there is
    /// no fog grid, or its footprint is unknown (an empty grid) — the
    /// backends then apply no sprite fog.
    pub(crate) fn resolve(scene: &Scene, fow: Option<(GridId, &'a FogOfWar)>) -> Option<Self> {
        let (gid, fow) = fow?;
        let grid = scene.grid(gid)?;
        let footprint = grid.footprint_cells()?;
        Some(Self {
            fow,
            transform: grid.transform,
            footprint,
        })
    }

    /// Is a sprite at world `center` hidden by the fog? Delegates to the
    /// single [`FogOfWar::hides_sprite`] rule (footprint + Visible-only).
    pub(crate) fn hides(&self, center: [f32; 3]) -> bool {
        self.fow.hides_sprite(
            &self.transform,
            self.footprint,
            glam::DVec3::new(
                f64::from(center[0]),
                f64::from(center[1]),
                f64::from(center[2]),
            ),
        )
    }

    /// FW.4 — the GPU sprite-cull skip-cache key: changes iff the fog
    /// could re-classify a sprite — the Visible set changed
    /// (`sprite_epoch`, review perf #1, NOT `mask_version` which bumps on
    /// every fade) OR the fog grid moved/rotated (`transform`, review
    /// #4). The top bit is always set so a present fog is never `0`, the
    /// value reserved for "no fog" — a fresh mask at `sprite_epoch 0`
    /// still differs from None (review #3).
    pub(crate) fn cull_key(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut mix = |v: u64| {
            h ^= v;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        };
        mix(self.fow.sprite_epoch());
        let t = &self.transform;
        mix(t.origin.x.to_bits());
        mix(t.origin.y.to_bits());
        mix(t.origin.z.to_bits());
        mix(t.rotation.x.to_bits());
        mix(t.rotation.y.to_bits());
        mix(t.rotation.z.to_bits());
        mix(t.rotation.w.to_bits());
        mix(t.voxel_world_size.to_bits());
        h | (1u64 << 63)
    }
}
