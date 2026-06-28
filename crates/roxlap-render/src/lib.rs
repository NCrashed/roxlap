//! roxlap-render — unified CPU/GPU renderer facade.
//!
//! One [`SceneRenderer`] hides the choice between the CPU opticast
//! path (`roxlap-core` / `roxlap-scene`, presented via `softbuffer`)
//! and the GPU compute-shader path (`roxlap-gpu`, presented via its
//! own wgpu surface). Construction picks the GPU backend when asked
//! and able, and **falls back to CPU automatically** when WGPU init
//! fails — so a host never has to branch on GPU availability or carry
//! the `Scene`→GPU upload/refresh/transform glue itself.
//!
//! Hosts stay thin: build a `Scene`, advance it from input, then call
//! [`SceneRenderer::render`] each frame. The facade owns the window
//! surface, the framebuffer/z-buffer (CPU) or the resident scene +
//! dirty-chunk tracking (GPU), and presentation.
//!
//! The per-frame flow is `render` → *(optional overlays)* → finish.
//! Between [`SceneRenderer::render`] and the finishing
//! [`SceneRenderer::present`] / [`SceneRenderer::paint_egui`] call, a
//! host may overlay depth-tested world-space lines with
//! [`SceneRenderer::draw_lines`] (editor gizmos, debug geometry — see
//! [`Line3`]); they land in the framebuffer, occluded by the rendered
//! scene, with egui still painting panels on top.
//!
//! This is the RF.0 skeleton: backend selection + fallback + a
//! clear-to-sky frame. RF.1/RF.2 fill in the real CPU/GPU scene
//! render; RF.3 adds sprites; RF.4 adds framebuffer capture.

#![forbid(unsafe_code)]

mod cpu;
/// WebGL2 framebuffer presenter for the CPU backend on wasm (the
/// browser has no `softbuffer`).
#[cfg(target_arch = "wasm32")]
mod cpu_blit;
#[cfg(feature = "hud")]
mod cpu_egui;
mod gpu;

#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;

use roxlap_core::kfa_draw::{compose_attachment, solve_kfa_limbs};
use roxlap_core::opticast::OpticastSettings;
use roxlap_core::sky::Sky;
use roxlap_core::Camera;
use roxlap_formats::voxel_clip::frame_at;
use roxlap_scene::Scene;

pub use roxlap_formats::character::{Attachment, Character, MeshRef};
pub use roxlap_formats::kfa::KfaSprite;
pub use roxlap_formats::kv6::Kv6;
pub use roxlap_formats::material::{BlendMode, Material};
pub use roxlap_formats::sprite::Sprite;
pub use roxlap_formats::voxel_clip::{
    DecodeError, DecodedClip, LoopMode, StreamingClip, VoxelClip, VoxelFrame,
};
pub use roxlap_gpu::{GpuInitError, GpuRendererSettings, PowerPreference};
// Re-exported so hosts can name the [`SceneRenderer::new`] bounds
// without adding a direct `raw-window-handle` dependency of their own.
pub use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
// Re-exported so hosts feed [`SceneRenderer::paint_egui`] from the exact
// egui version the renderer was built against (`hud` feature).
#[cfg(feature = "hud")]
pub use egui;

use crate::cpu::CpuBackend;
use crate::gpu::GpuBackend;

/// Type-erased display handle stored by the CPU backend's softbuffer
/// surface. `raw-window-handle` implements `HasDisplayHandle` for
/// `Arc<H>` (`H: ?Sized`), and the bare trait object implements its
/// own object-safe trait — so `Arc<W>` coerces to `Arc<DynDisplay>`
/// for any provider `W`.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) type DynDisplay = dyn HasDisplayHandle + Send + Sync + 'static;
/// Type-erased window handle counterpart to [`DynDisplay`].
#[cfg(not(target_arch = "wasm32"))]
pub(crate) type DynWindow = dyn HasWindowHandle + Send + Sync + 'static;

/// One placed sprite instance: which [`SpriteSet::models`] entry and
/// where in the world.
pub struct SpriteInstanceDesc {
    pub model: usize,
    pub pos: [f32; 3],
}

/// Stable handle to a registered sprite model, returned (one per
/// [`SpriteSet::models`] entry, in order) by
/// [`SceneRenderer::set_sprites`]. Pass it to
/// [`refresh_sprite_model`](SceneRenderer::refresh_sprite_model) to
/// re-register that model's geometry after a content edit — so callers
/// never track the positional `usize` index themselves. Opaque on
/// purpose: there is no arithmetic to do on it.
///
/// Also returned by [`SceneRenderer::add_sprite_model`] for an
/// incrementally registered model, and accepted by
/// [`remove_sprite_model`](SceneRenderer::remove_sprite_model). A handle
/// to a removed model is **stale**: it resolves to nothing, so passing
/// it anywhere is a safe no-op. The `gen` (generation) field guards a
/// future compacting registry; it stays `0` today because model slots
/// are tombstoned in place and never reused (GPU chain ids are
/// append-only).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SpriteModelId {
    pub(crate) slot: u32,
    pub(crate) gen: u32,
}

/// Stable handle to a **dynamically added** sprite instance — the result
/// of [`SceneRenderer::add_sprite_instance`], passed to
/// [`remove_sprite_instance`](SceneRenderer::remove_sprite_instance).
///
/// Backends remove instances by swap (O(1)), which moves another instance
/// into the freed slot; this handle survives that because the facade keeps
/// the id↔slot mapping up to date. The generation guards against a stale
/// handle aliasing a recycled slot.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SpriteInstanceId {
    slot: u32,
    gen: u32,
}

/// Facade-side slotmap that turns the backends' swap-remove indexing into
/// stable [`SpriteInstanceId`] handles. Both backends keep their dynamic
/// instances as a tail sublist indexed `0..n`; `order[dyn_index]` is the
/// owning slot, and a removal fixes up the one slot whose instance was
/// swapped into the hole.
#[derive(Default)]
struct DynInstanceMap {
    /// Per slot: `(generation, Some(dyn_index) while live)`.
    slots: Vec<(u32, Option<u32>)>,
    /// Per live `dyn_index`: the owning slot. Parallel to the backends'
    /// dynamic sublist (so `order.len()` == the dynamic instance count).
    order: Vec<u32>,
    free: Vec<u32>,
}

impl DynInstanceMap {
    /// Register a freshly appended instance (always at `dyn_index ==
    /// order.len()`); returns its stable handle.
    fn alloc(&mut self, dyn_index: u32) -> SpriteInstanceId {
        debug_assert_eq!(self.order.len() as u32, dyn_index);
        let slot = self.free.pop().unwrap_or_else(|| {
            self.slots.push((0, None));
            (self.slots.len() - 1) as u32
        });
        let gen = self.slots[slot as usize].0;
        self.slots[slot as usize].1 = Some(dyn_index);
        self.order.push(slot);
        SpriteInstanceId { slot, gen }
    }

    /// Resolve a handle to its current backend `dyn_index`, or `None` if
    /// it's stale / already removed.
    fn dyn_index(&self, id: SpriteInstanceId) -> Option<u32> {
        let (gen, idx) = *self.slots.get(id.slot as usize)?;
        (gen == id.gen).then_some(idx).flatten()
    }

    /// Apply a removal: the backend swap-removed `removed` and reported
    /// `moved` (the old-last `dyn_index` that slid into `removed`, or
    /// `None` if `removed` was itself the last).
    fn remove(&mut self, id: SpriteInstanceId, removed: u32, moved: Option<u32>) {
        self.slots[id.slot as usize].1 = None;
        self.slots[id.slot as usize].0 += 1; // bump generation
        self.free.push(id.slot);
        if let Some(last) = moved {
            let moved_slot = self.order[last as usize];
            self.slots[moved_slot as usize].1 = Some(removed);
            self.order[removed as usize] = moved_slot;
        }
        self.order.pop();
    }
}

/// Facade-side slotmap for registered sprite **models**, mirroring
/// [`DynInstanceMap`] but **without** the swap-remove fixup: a model
/// slot maps 1:1 to the backends' positional model index (the GPU LOD
/// chain id), which is append-only and never reused. A removed model
/// tombstones its slot *in place* (the backend frees the voxel data but
/// keeps the id), so a stale [`SpriteModelId`] resolves to `None` → a
/// safe no-op rather than aliasing another model.
#[derive(Default)]
struct DynModelMap {
    /// Per slot (== backend model index): `(generation, live)`. Slots are
    /// never reused, so `generation` stays `0`; `live` flips to `false`
    /// on removal.
    slots: Vec<(u32, bool)>,
}

impl DynModelMap {
    /// Reset to `n` live models with ids `0..n` — used by
    /// [`SceneRenderer::set_sprites`], which rebuilds the whole model set
    /// positionally (model index = chain id on both backends).
    fn reset(&mut self, n: usize) {
        self.slots.clear();
        self.slots.resize(n, (0, true));
    }

    /// Register a freshly appended model at positional index
    /// `model_index` (always the new `slots.len()`); returns its handle.
    fn alloc(&mut self, model_index: u32) -> SpriteModelId {
        debug_assert_eq!(self.slots.len() as u32, model_index);
        self.slots.push((0, true));
        SpriteModelId {
            slot: model_index,
            gen: 0,
        }
    }

    /// Resolve a handle to its backend model index, or `None` if it's
    /// stale / already removed.
    fn model_index(&self, id: SpriteModelId) -> Option<usize> {
        let (gen, live) = *self.slots.get(id.slot as usize)?;
        (gen == id.gen && live).then_some(id.slot as usize)
    }

    /// Tombstone a model slot in place. Returns `false` if the handle is
    /// stale / already removed.
    fn remove(&mut self, id: SpriteModelId) -> bool {
        let Some(slot) = self.slots.get_mut(id.slot as usize) else {
            return false;
        };
        if slot.0 != id.gen || !slot.1 {
            return false;
        }
        slot.1 = false;
        true
    }
}

/// Stable handle to a registered animated voxel clip (VCL.4) — the
/// result of [`SceneRenderer::add_voxel_clip`], passed to
/// [`add_clip_instance_posed`](SceneRenderer::add_clip_instance_posed)
/// and [`remove_voxel_clip`](SceneRenderer::remove_voxel_clip). Like
/// [`SpriteModelId`], a removed clip's handle is stale → a safe no-op.
/// Reset by [`set_sprites`](SceneRenderer::set_sprites) (which drops the
/// dynamic + clip layers).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct VoxelClipId {
    slot: u32,
    gen: u32,
}

/// Facade-side slotmap for registered voxel clips — mirrors
/// [`DynModelMap`]: a clip slot maps 1:1 to the backends' positional clip
/// index (append-only, tombstoned in place on removal, never reused).
///
/// `reset` clears the slots **and bumps `epoch`**, which is baked into each
/// minted id's `gen`. A handle from before a `set_sprites` therefore carries
/// the old epoch and resolves to `None` rather than silently aliasing the
/// new clip that re-took its slot.
#[derive(Default)]
struct DynClipMap {
    /// Per slot: `(epoch_at_alloc, live)`.
    slots: Vec<(u32, bool)>,
    epoch: u32,
}

impl DynClipMap {
    fn alloc(&mut self, clip_index: u32) -> VoxelClipId {
        debug_assert_eq!(self.slots.len() as u32, clip_index);
        self.slots.push((self.epoch, true));
        VoxelClipId {
            slot: clip_index,
            gen: self.epoch,
        }
    }

    fn clip_index(&self, id: VoxelClipId) -> Option<usize> {
        let (gen, live) = *self.slots.get(id.slot as usize)?;
        (gen == id.gen && live).then_some(id.slot as usize)
    }

    fn remove(&mut self, id: VoxelClipId) -> bool {
        let Some(slot) = self.slots.get_mut(id.slot as usize) else {
            return false;
        };
        if slot.0 != id.gen || !slot.1 {
            return false;
        }
        slot.1 = false;
        true
    }

    fn reset(&mut self) {
        self.slots.clear();
        self.epoch = self.epoch.wrapping_add(1);
    }
}

/// Stable handle to a registered animated character (VCL.6) — the result
/// of [`SceneRenderer::add_character`], advanced each frame with
/// [`advance_character`](SceneRenderer::advance_character) and dropped with
/// [`remove_character`](SceneRenderer::remove_character). Reset by
/// [`set_sprites`](SceneRenderer::set_sprites).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct CharacterId {
    slot: u32,
    gen: u32,
}

/// Facade-side slotmap for registered characters (mirrors [`DynClipMap`],
/// including the epoch bump on `reset` so a pre-`set_sprites` handle
/// resolves to `None` instead of aliasing a new character).
#[derive(Default)]
struct CharMap {
    /// Per slot: `(epoch_at_alloc, live)`.
    slots: Vec<(u32, bool)>,
    epoch: u32,
}

impl CharMap {
    fn alloc(&mut self, index: u32) -> CharacterId {
        debug_assert_eq!(self.slots.len() as u32, index);
        self.slots.push((self.epoch, true));
        CharacterId {
            slot: index,
            gen: self.epoch,
        }
    }
    fn index(&self, id: CharacterId) -> Option<usize> {
        let (gen, live) = *self.slots.get(id.slot as usize)?;
        (gen == id.gen && live).then_some(id.slot as usize)
    }
    fn remove(&mut self, id: CharacterId) -> bool {
        let Some(slot) = self.slots.get_mut(id.slot as usize) else {
            return false;
        };
        if slot.0 != id.gen || !slot.1 {
            return false;
        }
        slot.1 = false;
        true
    }
    fn reset(&mut self) {
        self.slots.clear();
        self.epoch = self.epoch.wrapping_add(1);
    }
}

/// Stable handle to a registered **streaming** voxel clip (follow-up #3) —
/// the result of [`SceneRenderer::add_streaming_clip`], advanced with
/// [`set_streaming_clip_frame`](SceneRenderer::set_streaming_clip_frame) and
/// dropped with
/// [`remove_streaming_clip`](SceneRenderer::remove_streaming_clip). Reset by
/// [`set_sprites`](SceneRenderer::set_sprites).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct StreamingClipId {
    slot: u32,
    gen: u32,
}

/// Handle to an instance of a streaming clip
/// ([`add_streaming_clip_instance`](SceneRenderer::add_streaming_clip_instance)).
///
/// Deliberately **distinct** from [`SpriteInstanceId`]: a streaming clip's
/// frame is per-*clip* (all its instances share one re-uploaded model,
/// advanced by
/// [`set_streaming_clip_frame`](SceneRenderer::set_streaming_clip_frame)), so
/// a streaming instance is *not* accepted by the per-instance
/// [`set_clip_instance_frame`](SceneRenderer::set_clip_instance_frame) —
/// trying to scrub two instances of one streaming clip independently is a
/// compile error, not a silent coupling. (Use a flipbook clip for
/// per-instance frames.) Move it with
/// [`set_streaming_instance_transform`](SceneRenderer::set_streaming_instance_transform)
/// and drop it with
/// [`remove_streaming_instance`](SceneRenderer::remove_streaming_instance).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct StreamingInstanceId(SpriteInstanceId);

/// Facade-side slotmap for streaming clips (mirrors [`CharMap`], epoch bump
/// on `reset` included).
#[derive(Default)]
struct StreamingClipMap {
    /// Per slot: `(epoch_at_alloc, live)`.
    slots: Vec<(u32, bool)>,
    epoch: u32,
}

impl StreamingClipMap {
    fn alloc(&mut self, index: u32) -> StreamingClipId {
        debug_assert_eq!(self.slots.len() as u32, index);
        self.slots.push((self.epoch, true));
        StreamingClipId {
            slot: index,
            gen: self.epoch,
        }
    }
    fn index(&self, id: StreamingClipId) -> Option<usize> {
        let (gen, live) = *self.slots.get(id.slot as usize)?;
        (gen == id.gen && live).then_some(id.slot as usize)
    }
    fn remove(&mut self, id: StreamingClipId) -> bool {
        let Some(slot) = self.slots.get_mut(id.slot as usize) else {
            return false;
        };
        if slot.0 != id.gen || !slot.1 {
            return false;
        }
        slot.1 = false;
        true
    }
    fn reset(&mut self) {
        self.slots.clear();
        self.epoch = self.epoch.wrapping_add(1);
    }
}

/// One registered streaming clip: the seekable cursor + the single sprite
/// model it re-uploads each frame, plus the dims/pivot used to rebuild it.
struct StreamingClipState {
    cursor: StreamingClip,
    model: SpriteModelId,
    dims: [u32; 3],
    pivot: [f32; 3],
    /// Colour→material map (TV.3), empty for an all-opaque streaming clip.
    /// Re-applied on every per-frame re-upload so the streamed model keeps
    /// its per-voxel materials as it advances.
    material_map: Vec<(u32, u8)>,
}

/// Per-clip-attachment playback clock (VCL.6): the timing it needs to
/// resolve a frame, plus its own accumulating clock.
struct ClipClock {
    durations: Vec<u32>,
    loop_mode: LoopMode,
    /// Playback rate, Q8 (256 = 1×).
    speed_q8: i32,
    /// Accumulated playback time (ms), seeded from the attachment's
    /// `start_phase_ms`.
    clock_ms: f64,
}

impl ClipClock {
    /// Advance the clock by `dt` seconds at its Q8 `speed` and return the
    /// frame to show. Shared by character attachments and standalone clip
    /// players. A negative clock (rewind past 0) reads as frame 0 but is
    /// kept signed so resuming forward is continuous.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn tick(&mut self, dt: f64) -> u32 {
        self.clock_ms += dt * 1000.0 * f64::from(self.speed_q8) / 256.0;
        frame_at(
            &self.durations,
            self.loop_mode,
            self.clock_ms.max(0.0) as u32,
        ) as u32
    }
}

/// Facade-side metadata captured for a registered flipbook clip, so editor
/// queries + the auto-player don't shadow the `DecodedClip`.
struct ClipMeta {
    dims: [u32; 3],
    pivot: [f32; 3],
    voxel_world_size: f32,
    durations: Vec<u32>,
    loop_mode: LoopMode,
    /// Colour→material map the clip was registered with (TV.3), empty for an
    /// all-opaque clip. Retained so an in-place
    /// [`update_clip_frame`](SceneRenderer::update_clip_frame) re-classifies
    /// the edited frame's voxels instead of dropping its per-voxel materials.
    material_map: Vec<(u32, u8)>,
}

/// Public metadata for a registered clip — the inspector view returned by
/// [`SceneRenderer::clip_metadata`].
#[derive(Clone, Debug, PartialEq)]
pub struct ClipMetadata {
    /// Fixed bounding box (voxels).
    pub dims: [u32; 3],
    /// Model pivot (the kv6 pivot frames share).
    pub pivot: [f32; 3],
    /// Render scale (1 voxel = this many world units).
    pub voxel_world_size: f32,
    /// Playback wrap behaviour.
    pub loop_mode: LoopMode,
    /// Number of frames.
    pub frame_count: usize,
    /// Per-frame durations (ms), one per frame.
    pub durations: Vec<u32>,
    /// Total loop length (ms) — sum of `durations`.
    pub total_ms: u32,
}

/// What an auto-advancing [`ClipPlayer`] (#6) drives each
/// [`advance_voxel_clips`](SceneRenderer::advance_voxel_clips). A flipbook
/// clip's frame is per-instance; a streaming clip's is per-clip (its
/// instances share one model), so the targets differ.
#[derive(Clone, Copy)]
enum PlayerTarget {
    Flipbook(SpriteInstanceId),
    Streaming(StreamingClipId),
}

/// A standalone clip given its own playback clock (#6): the host calls
/// `advance_voxel_clips(dt)` once instead of hand-driving `frame_at` +
/// `set_clip_instance_frame`.
struct ClipPlayer {
    target: PlayerTarget,
    clock: ClipClock,
    /// When `true`, [`advance_voxel_clips`](SceneRenderer::advance_voxel_clips)
    /// leaves the clock (and frame) untouched — the editor's play/pause.
    paused: bool,
}

/// One live bone attachment: which bone drives it, its local offset, the
/// renderer instance it owns, and (for a clip target) its playback clock.
struct AttachInst {
    bone: usize,
    local_offset: roxlap_formats::xform::BoneXform,
    inst: SpriteInstanceId,
    clip: Option<ClipClock>,
}

/// A live animated character: the hinge skeleton (the bone-transform
/// solver) + one [`AttachInst`] per bone attachment.
struct CharInstance {
    skeleton: KfaSprite,
    attaches: Vec<AttachInst>,
    /// Sprite models + voxel clips this character registered, so
    /// [`remove_character`](SceneRenderer::remove_character) can free them
    /// (otherwise they leak until the next `set_sprites`).
    models: Vec<SpriteModelId>,
    clips: Vec<VoxelClipId>,
}

/// Orientation + position for a dynamic sprite instance — the per-frame
/// pose passed to [`SceneRenderer::add_sprite_instance_posed`] and
/// [`set_sprite_instance_transform`](SceneRenderer::set_sprite_instance_transform).
///
/// `right`/`up`/`forward` are the instance's local axes expressed in
/// world space (the columns of the model→world rotation), mapping
/// directly onto the underlying [`Sprite`]'s `s`/`h`/`f` (kv6 local
/// +x/+y/+z). They **must** be non-singular (`det ≠ 0`) but need not be
/// orthonormal — a uniform/non-uniform scale or shear is fine. A
/// near-singular basis falls through the renderer's degenerate-basis
/// guards and the instance silently skips that frame rather than
/// panicking. [`Default`] is the identity basis (axis-aligned).
#[derive(Clone, Copy, Debug)]
pub struct DynSpriteTransform {
    /// Instance world position (the kv6 pivot maps here).
    pub pos: [f32; 3],
    /// Local +x in world space ↦ [`Sprite::s`].
    pub right: [f32; 3],
    /// Local +y in world space ↦ [`Sprite::h`].
    pub up: [f32; 3],
    /// Local +z in world space ↦ [`Sprite::f`].
    pub forward: [f32; 3],
}

impl Default for DynSpriteTransform {
    fn default() -> Self {
        Self {
            pos: [0.0, 0.0, 0.0],
            right: [1.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
            forward: [0.0, 0.0, 1.0],
        }
    }
}

impl DynSpriteTransform {
    /// Stamp this pose onto a [`Sprite`] in place: `pos → p`,
    /// `right/up/forward → s/h/f` (a direct copy — the basis is the
    /// model→world columns). Both backends keep the rest of the template
    /// (`kv6`, `flags`) and only overwrite the pose.
    pub(crate) fn apply_to(self, s: &mut Sprite) {
        s.p = self.pos;
        s.s = self.right;
        s.h = self.up;
        s.f = self.forward;
    }
}

/// Backend-agnostic sprite description. The facade builds the CPU
/// per-instance draw list and the GPU instanced registry from the
/// same data, so both backends show identical sprites. The host owns
/// content (which models, where, recolouring) — building a recoloured
/// variant is just a second [`Sprite`] model with edited `kv6.voxels`.
pub struct SpriteSet {
    /// Distinct voxel models (KV6 + base orientation). Instances index
    /// into this; their position overrides the model's.
    pub models: Vec<Sprite>,
    pub instances: Vec<SpriteInstanceDesc>,
    /// Model the [`SceneRenderer::carve_active_sprite`] hotkey edits
    /// (GPU only, mirroring the demo's `G`-carve). `None` disables it.
    pub carve_model: Option<usize>,
}

/// Per-frame inputs both backends consume. The host builds the
/// [`OpticastSettings`] (it owns scan distance etc.); the facade does
/// everything else (pool config, sky fill, render, present).
pub struct FrameParams<'a> {
    /// CPU opticast settings (scan distance, mip ladder, framebuffer
    /// geometry). Ignored by the GPU backend.
    pub settings: &'a OpticastSettings,
    /// Packed engine sky colour: the CPU sky-miss fill + skycast, and
    /// the clear colour if no scene renders.
    pub sky_color: u32,
    /// Optional sky panorama for the CPU rasterizer's sky sampling.
    pub sky: Option<&'a Sky>,
    /// CPU fog: packed colour + max scan distance (voxels). `0` scan
    /// distance disables CPU fog.
    pub fog_color: u32,
    pub fog_max_scan_dist: i32,
    /// CPU: treat z=255 as air (avoids the S1.X bedrock path for
    /// out-of-bounds cameras).
    pub treat_z_max_as_air: bool,
    /// GPU scene-grid LOD scan distance (world units); see GPU.11.1.
    /// Ignored by the CPU backend.
    pub gpu_mip_scan_dist: f32,
    /// GPU outer-DDA step budget (chunks). Ignored by the CPU backend.
    pub gpu_max_outer_steps: u32,
    /// GPU vertical field of view (radians). Ignored by the CPU
    /// backend (it derives projection from [`OpticastSettings`]).
    pub gpu_fov_y_rad: f32,
    /// Whether to draw the renderer's sprites this frame. Both backends
    /// draw KV6 sprites flat-lit (the clean-room DDA sprite raycaster on
    /// CPU; uploaded model colours on GPU), so no host-supplied lighting
    /// is needed — this is just the on/off opt-in. `false` skips sprite
    /// drawing.
    pub draw_sprites: bool,
    /// Per-face directional shading for the voxel grids — voxlap's
    /// `setsideshades(top, bot, left, right, up, down)`, the grid-scan
    /// analogue of [`draw_sprites`](Self::draw_sprites). Each
    /// entry darkens the faces pointing that way; the host typically
    /// passes its engine's `side_shades()`. The default `[0; 6]` keeps
    /// `sideshademode` off (no per-side shading), so existing hosts and
    /// the oracle goldens are unaffected. Applied each frame by **both**
    /// backends: the CPU rasteriser via `gcsub`, and the GPU scene-DDA
    /// pass by darkening a hit voxel's brightness by the hit face's
    /// shade (the face taken from the DDA's last-stepped axis).
    pub side_shades: [i8; 6],
}

/// Result of [`SceneRenderer::pick`] — a resolved screen→world voxel
/// hit. `world` is the surface point (`cam.pos + t · normalize(ray)`);
/// `grid` + `voxel` are the owning grid and its **grid-local** voxel
/// (transform-correct for rotated / translated grids).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct PickHit {
    pub world: [f32; 3],
    pub grid: roxlap_scene::GridId,
    pub voxel: glam::IVec3,
}

/// A world-space view ray: the canonical unproject output of
/// [`SceneRenderer::view_ray`]. `dir` is unit-length. Feed it straight
/// to [`roxlap_scene::Scene::raycast`] for depth-free, backend-agnostic
/// voxel picking (`scene.raycast(ray.origin, ray.dir, max_dist)`), or
/// intersect it with a plane for tile selection.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Ray {
    pub origin: glam::DVec3,
    pub dir: glam::DVec3,
}

/// A world-space line segment to draw over a rendered frame via
/// [`SceneRenderer::draw_lines`] — editor gizmos (bounding boxes, floor
/// grids, axes, hover wireframes), debug paths, etc.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Line3 {
    /// World-space endpoints (voxel units), in the same frame the
    /// rendered scene + `camera` use.
    pub a: [f64; 3],
    pub b: [f64; 3],
    /// `0xAARRGGBB` — the high byte is an alpha blend factor (`0xFF`
    /// opaque, `0x00` invisible), the low 24 bits the RGB colour.
    pub color: u32,
    /// Screen-space thickness in pixels (`<= 1.0` draws a 1px line).
    pub width_px: f32,
    /// `true`: the segment is occluded by nearer rendered geometry
    /// (depth-tested against the frame's z-buffer). `false`: always on
    /// top (e.g. a hover highlight that should show through the model).
    pub depth_test: bool,
}

/// A handle to an uploaded image-sprite texture, returned by
/// [`SceneRenderer::upload_image`]. Positional (like [`SpriteModelId`]):
/// it indexes the backend's texture store. Pass it in an [`ImageSprite`]
/// for [`SceneRenderer::draw_images`], or to
/// [`drop_image`](SceneRenderer::drop_image) to release it. Opaque on
/// purpose — there's no arithmetic to do on it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ImageId(pub(crate) usize);

/// How an [`ImageSprite`]'s quad is oriented in the world.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ImageFacing {
    /// Fixed in world space: the quad lies in the plane spanned by `u`
    /// (the image's +column / width direction) and `v` (its +row /
    /// height direction). Both are world-space directions; their length
    /// is ignored (the quad is sized by [`ImageSprite::size`]), so pass
    /// the plane's axes directly. Row 0 of the image is the `origin`
    /// edge and rows grow along `v`.
    World { u: [f32; 3], v: [f32; 3] },
    /// Always faces the camera (billboard); `up` is the world direction
    /// the image's top edge points toward (e.g. world `-Z` for the
    /// scene-demo's z-down world, or any "up" the host prefers).
    Billboard { up: [f32; 3] },
}

/// One placed 2D image sprite for the current frame: a flat textured
/// quad in world space, composited over the rendered scene with the
/// frame's depth buffer (so the voxel model can occlude it). Built per
/// frame and passed to [`SceneRenderer::draw_images`], mirroring
/// [`Line3`] / [`SceneRenderer::draw_lines`]. The texture is uploaded
/// once via [`SceneRenderer::upload_image`] and referenced by [`image`].
///
/// [`image`]: ImageSprite::image
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ImageSprite {
    /// The uploaded texture to draw (from [`SceneRenderer::upload_image`]).
    pub image: ImageId,
    /// World position of the quad's **top-left** corner — the image's
    /// `(column 0, row 0)` texel. The quad extends `size[0]` along the
    /// facing's `u` and `size[1]` along its `v`.
    pub origin: [f32; 3],
    /// World orientation of the quad — fixed in world or camera-facing.
    pub facing: ImageFacing,
    /// World size of the quad along `u` and `v`. For pixel-art traced at
    /// 1 texel = 1 voxel, pass `[width as f32, height as f32]`.
    pub size: [f32; 2],
    /// Multiplied into every sampled texel (tint + opacity), `0xAARRGGBB`.
    /// `0xFFFFFFFF` draws the texture unchanged; the high byte scales
    /// the texel alpha (e.g. `0x80FFFFFF` = 50 % opacity).
    pub tint: u32,
    /// Alpha cutoff in `0.0..=1.0`. Texels whose **own** alpha is below
    /// this are discarded outright (not blended) — crisp pixel-art edges
    /// instead of a semi-transparent haze, and the same threshold decides
    /// what [`SceneRenderer::pick_image`] treats as solid. `0.0` keeps the
    /// plain straight-alpha over-blend (every non-zero texel draws).
    pub alpha_cutoff: f32,
    /// `true`: occluded by nearer rendered geometry (depth-tested against
    /// the frame's depth buffer, with a bias so a quad resting on a
    /// coincident voxel face doesn't z-fight). `false`: always on top.
    pub depth_test: bool,
    /// `true`: draw regardless of which way the quad faces (no backface
    /// cull) — what reference images usually want. `false`: cull when the
    /// quad faces away from the camera. Ignored for
    /// [`ImageFacing::Billboard`] (it always faces the camera).
    pub double_sided: bool,
}

/// Backend-agnostic resolved quad: four world corners (`TL, TR, BL, BR`,
/// with UVs `(0,0) (1,0) (0,1) (1,1)`) + the texture to map. The facade
/// resolves [`ImageSprite::facing`] into corners and culls back-facing
/// quads once, so both backends draw from the same geometry.
#[derive(Clone, Copy, Debug)]
pub(crate) struct QuadDraw {
    pub corners: [[f32; 3]; 4],
    pub image: ImageId,
    pub tint: u32,
    pub depth_test: bool,
    pub alpha_cutoff: f32,
}

/// Result of [`SceneRenderer::pick_image`] — a resolved screen→sprite hit.
/// `uv` is the normalised position within the quad (`(0,0)` = top-left
/// corner); `texel` is the matching source-image pixel; `world` is the
/// hit point; `t` is its euclidean distance from the camera.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ImagePickHit {
    pub image: ImageId,
    pub uv: [f32; 2],
    pub texel: (u32, u32),
    pub world: [f32; 3],
    pub t: f32,
}

/// Which renderer a [`SceneRenderer`] resolved to at construction.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Backend {
    /// `roxlap-core` opticast, presented via `softbuffer`.
    Cpu,
    /// `roxlap-gpu` compute marcher, presented via wgpu.
    Gpu,
}

/// Construction-time options for [`SceneRenderer::new`].
pub struct RenderOptions {
    /// Try the GPU backend first. When `false`, or when GPU init
    /// fails, the renderer uses the CPU backend.
    pub want_gpu: bool,
    /// Settings forwarded to [`roxlap_gpu::GpuRenderer`] when the GPU
    /// backend is selected.
    pub gpu: GpuRendererSettings,
    /// Packed `0x00RRGGBB` (alpha ignored) the empty/clear frame fills
    /// with until a scene render lands. Also the CPU sky-miss colour
    /// default if a frame supplies none.
    pub clear_sky: u32,
    /// CPU [`ScratchPool`](roxlap_core::rasterizer::ScratchPool) `lastx`
    /// sizing — the largest combined grid `vsid` the CPU rasterizer
    /// will see. Pre-sizing keeps later frames allocation-free.
    pub cpu_max_grid_vsid: u32,
    /// CPU strip-parallel render thread count (capped to the rayon
    /// pool). One [`ScratchPool`](roxlap_core::rasterizer::ScratchPool)
    /// slot per thread.
    pub cpu_render_threads: usize,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            want_gpu: false,
            gpu: GpuRendererSettings::default(),
            clear_sky: 0x0099_b3d9,
            // 32 chunks × CHUNK_SIZE_XY — the scene-demo's widest
            // combined ground grid.
            cpu_max_grid_vsid: 32 * roxlap_scene::CHUNK_SIZE_XY,
            cpu_render_threads: 4,
        }
    }
}

/// Depth-test slack (same spirit as the backends' `DEPTH_BIAS`) so a
/// [`SceneRenderer::pick_image`] hit on a sprite resting on a coincident
/// voxel face isn't rejected as "occluded".
const PICK_DEPTH_BIAS: f32 = 0.5;

// --- image-sprite geometry helpers (shared by both backends) ---

fn v_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn v_add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn v_scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn v_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn v_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn v_norm(a: [f32; 3]) -> [f32; 3] {
    let len = v_dot(a, a).sqrt();
    if len < 1e-12 {
        a
    } else {
        v_scale(a, 1.0 / len)
    }
}

/// Intersect a ray (`origin` + `dir`, `dir` un-normalised) with a quad
/// `[TL, TR, BL, BR]` and return `(uv, t)` for a front/back hit inside
/// the quad — `uv` in `0..=1` (`(0,0)` = `TL`), `t` the ray parameter
/// (`hit = origin + dir·t`). `None` for a parallel ray, a hit behind the
/// origin, a degenerate quad, or a hit outside the `u`/`v` span. Solves
/// affine coords exactly for a (possibly skew) parallelogram. Standalone
/// so the geometry is unit-testable without a renderer.
fn ray_quad_uv(
    origin: [f32; 3],
    dir: [f32; 3],
    corners: &[[f32; 3]; 4],
) -> Option<([f32; 2], f32)> {
    let [tl, tr, bl, _br] = *corners;
    let ue = v_sub(tr, tl); // +u edge (width)
    let ve = v_sub(bl, tl); // +v edge (height)
    let n = v_cross(ue, ve);
    let denom = v_dot(dir, n);
    if denom.abs() < 1e-12 {
        return None; // ray parallel to the quad's plane
    }
    let t = v_dot(v_sub(tl, origin), n) / denom;
    if t <= 1e-6 {
        return None; // behind / at the origin
    }
    let p = v_add(origin, v_scale(dir, t));
    let rel = v_sub(p, tl);
    let guu = v_dot(ue, ue);
    let guv = v_dot(ue, ve);
    let gvv = v_dot(ve, ve);
    let det = guu * gvv - guv * guv;
    if det.abs() < 1e-12 {
        return None; // degenerate quad
    }
    let wu = v_dot(rel, ue);
    let wv = v_dot(rel, ve);
    let a = (gvv * wu - guv * wv) / det;
    let b = (guu * wv - guv * wu) / det;
    if !(0.0..=1.0).contains(&a) || !(0.0..=1.0).contains(&b) {
        return None; // outside the quad
    }
    Some(([a, b], t))
}

/// Resolve an [`ImageSprite`] into its four world corners (`TL, TR, BL,
/// BR`), or `None` when a `double_sided == false` world quad faces away
/// from the camera (back-face cull) or its plane is degenerate. The
/// camera basis is used only for [`ImageFacing::Billboard`] and the cull
/// test.
fn resolve_quad(sprite: &ImageSprite, camera: &Camera) -> Option<QuadDraw> {
    let cam_pos = [
        camera.pos[0] as f32,
        camera.pos[1] as f32,
        camera.pos[2] as f32,
    ];
    let cam_fwd = v_norm([
        camera.forward[0] as f32,
        camera.forward[1] as f32,
        camera.forward[2] as f32,
    ]);

    let (u_hat, v_hat) = match sprite.facing {
        ImageFacing::World { u, v } => (v_norm(u), v_norm(v)),
        ImageFacing::Billboard { up } => {
            // Horizontal axis ⟂ both the view direction and `up`; fall
            // back to the camera right when `up` is parallel to the view.
            let mut u_hat = v_norm(v_cross(up, cam_fwd));
            if v_dot(u_hat, u_hat) < 1e-12 {
                u_hat = v_norm([
                    camera.right[0] as f32,
                    camera.right[1] as f32,
                    camera.right[2] as f32,
                ]);
            }
            // Vertical axis ⟂ both, pointing *down* (rows grow downward)
            // so the top edge ends up toward `up`.
            let mut v_hat = v_norm(v_cross(cam_fwd, u_hat));
            if v_dot(v_hat, up) > 0.0 {
                v_hat = v_scale(v_hat, -1.0);
            }
            (u_hat, v_hat)
        }
    };

    let du = v_scale(u_hat, sprite.size[0]);
    let dv = v_scale(v_hat, sprite.size[1]);
    let tl = sprite.origin;
    let tr = v_add(tl, du);
    let bl = v_add(tl, dv);
    let br = v_add(tr, dv);

    // Back-face cull for fixed world quads (billboards always face us).
    if !sprite.double_sided {
        if let ImageFacing::World { .. } = sprite.facing {
            let normal = v_cross(du, dv);
            // Front-facing when the quad normal points toward the camera.
            if v_dot(normal, v_sub(cam_pos, tl)) <= 0.0 {
                return None;
            }
        }
    }

    Some(QuadDraw {
        corners: [tl, tr, bl, br],
        image: sprite.image,
        tint: sprite.tint,
        depth_test: sprite.depth_test,
        alpha_cutoff: sprite.alpha_cutoff,
    })
}

/// Renderer-internal backend; never exposes wgpu or softbuffer types.
/// The GPU variant owns the whole wgpu device/queue/pipelines, so
/// it's boxed to keep the enum small.
enum BackendImpl {
    // Both variants boxed so the enum stays small regardless of which
    // backend's state is larger (clippy::large_enum_variant).
    Cpu(Box<CpuBackend>),
    Gpu(Box<GpuBackend>),
}

/// Unified renderer over the CPU and GPU paths. See the crate docs.
pub struct SceneRenderer {
    inner: BackendImpl,
    /// Handles for dynamically added sprite instances (see
    /// [`Self::add_sprite_instance`]). Reset by [`Self::set_sprites`].
    dyn_map: DynInstanceMap,
    /// Handles for registered sprite models (see [`Self::add_sprite_model`]
    /// and the models returned by [`Self::set_sprites`]). Reset by
    /// [`Self::set_sprites`].
    model_map: DynModelMap,
    /// Handles for registered animated voxel clips (see
    /// [`Self::add_voxel_clip`]). Reset by [`Self::set_sprites`].
    clip_map: DynClipMap,
    /// Handles for registered animated characters (see
    /// [`Self::add_character`]). Reset by [`Self::set_sprites`].
    char_map: CharMap,
    /// Live character runtimes, parallel to `char_map` slots (VCL.6).
    char_instances: Vec<CharInstance>,
    /// Handles for registered streaming clips (see
    /// [`Self::add_streaming_clip`]). Reset by [`Self::set_sprites`].
    streaming_map: StreamingClipMap,
    /// Streaming-clip runtimes (cursor + one re-uploaded model), parallel
    /// to `streaming_map` slots; `None` once removed (#3).
    streaming_clips: Vec<Option<StreamingClipState>>,
    /// Metadata per registered flipbook clip, indexed by the backend clip
    /// index (parallel to `clip_map`). Captured at [`Self::add_voxel_clip`]
    /// so the editor queries ([`Self::clip_metadata`]) + the auto-player
    /// don't have to re-pass / shadow the `DecodedClip`. Reset by
    /// [`Self::set_sprites`].
    clip_meta: Vec<ClipMeta>,
    /// Auto-advancing clip players (#6); ticked by
    /// [`Self::advance_voxel_clips`]. Reset by [`Self::set_sprites`].
    clip_players: Vec<ClipPlayer>,
}

impl SceneRenderer {
    /// Build a renderer for `window` — any [`raw-window-handle`]
    /// provider (winit, SDL, GLFW, …) in an `Arc`. `size` is the
    /// window's initial physical framebuffer size in pixels; thereafter
    /// the host reports changes via [`Self::resize`]. Passing the size
    /// explicitly keeps the facade decoupled from any one windowing
    /// library's size API.
    ///
    /// Selects the GPU backend when `opts.want_gpu` and WGPU
    /// initialises; otherwise the CPU backend. **Never fails** — a
    /// missing/incompatible GPU silently yields the CPU path (the
    /// message is logged to stderr).
    ///
    /// [`raw-window-handle`]: raw_window_handle
    #[cfg(not(target_arch = "wasm32"))]
    #[must_use]
    pub fn new<W>(window: Arc<W>, size: (u32, u32), opts: &RenderOptions) -> Self
    where
        W: HasWindowHandle + HasDisplayHandle + Send + Sync + 'static,
    {
        if opts.want_gpu {
            match GpuBackend::new(window.clone(), size, opts) {
                Ok(g) => {
                    return Self {
                        inner: BackendImpl::Gpu(Box::new(g)),
                        dyn_map: DynInstanceMap::default(),
                        model_map: DynModelMap::default(),
                        clip_map: DynClipMap::default(),
                        char_map: CharMap::default(),
                        char_instances: Vec::new(),
                        streaming_map: StreamingClipMap::default(),
                        streaming_clips: Vec::new(),
                        clip_meta: Vec::new(),
                        clip_players: Vec::new(),
                    };
                }
                Err(e) => {
                    eprintln!(
                        "roxlap-render: GPU init failed ({e}); falling back to the CPU renderer",
                    );
                }
            }
        }
        Self {
            inner: BackendImpl::Cpu(Box::new(CpuBackend::new(window, size, opts))),
            dyn_map: DynInstanceMap::default(),
            model_map: DynModelMap::default(),
            clip_map: DynClipMap::default(),
            char_map: CharMap::default(),
            char_instances: Vec::new(),
            streaming_map: StreamingClipMap::default(),
            streaming_clips: Vec::new(),
            clip_meta: Vec::new(),
            clip_players: Vec::new(),
        }
    }

    /// wasm/WebGPU build-time entry: build a renderer over an HTML
    /// `canvas`. `size` is the canvas's initial framebuffer size in
    /// pixels; the host reports later changes via [`Self::resize`].
    ///
    /// Async because the browser drives wgpu's adapter/device requests
    /// through its event loop — `await` it inside a
    /// `wasm_bindgen_futures::spawn_local` task. Selects the GPU
    /// (WebGPU) backend when `opts.want_gpu` and WebGPU is available;
    /// otherwise (no WebGPU, or init failed) it falls back to the CPU
    /// opticast path presented through a WebGL2 blit on the same canvas.
    /// **Never fails** — the message is logged to the browser console.
    #[cfg(target_arch = "wasm32")]
    pub async fn new_from_canvas_async(
        canvas: web_sys::HtmlCanvasElement,
        size: (u32, u32),
        opts: &RenderOptions,
    ) -> Self {
        if opts.want_gpu {
            // `SurfaceTarget::Canvas` moves the canvas into wgpu, so the
            // GPU attempt gets a clone — the CPU fallback keeps the
            // original if WebGPU init fails.
            match GpuBackend::new_async(canvas.clone(), size, opts).await {
                Ok(g) => {
                    return Self {
                        inner: BackendImpl::Gpu(Box::new(g)),
                        dyn_map: DynInstanceMap::default(),
                        model_map: DynModelMap::default(),
                        clip_map: DynClipMap::default(),
                        char_map: CharMap::default(),
                        char_instances: Vec::new(),
                        streaming_map: StreamingClipMap::default(),
                        streaming_clips: Vec::new(),
                        clip_meta: Vec::new(),
                        clip_players: Vec::new(),
                    };
                }
                Err(e) => {
                    web_sys::console::warn_1(
                        &format!("roxlap-render: WebGPU init failed ({e}); using the CPU renderer")
                            .into(),
                    );
                }
            }
        }
        Self {
            inner: BackendImpl::Cpu(Box::new(CpuBackend::new_from_canvas(canvas, size, opts))),
            dyn_map: DynInstanceMap::default(),
            model_map: DynModelMap::default(),
            clip_map: DynClipMap::default(),
            char_map: CharMap::default(),
            char_instances: Vec::new(),
            streaming_map: StreamingClipMap::default(),
            streaming_clips: Vec::new(),
            clip_meta: Vec::new(),
            clip_players: Vec::new(),
        }
    }

    /// Which backend was selected.
    #[must_use]
    pub fn backend(&self) -> Backend {
        match self.inner {
            BackendImpl::Cpu(_) => Backend::Cpu,
            BackendImpl::Gpu(_) => Backend::Gpu,
        }
    }

    /// The GPU adapter description when on the GPU backend, else
    /// `None`.
    #[must_use]
    pub fn adapter_info(&self) -> Option<&str> {
        match &self.inner {
            BackendImpl::Gpu(g) => Some(g.adapter_info()),
            BackendImpl::Cpu(_) => None,
        }
    }

    /// Upload an equirectangular sky panorama (RGBA8, `w×h`) for the
    /// GPU marcher's sky sampling. No-op on the CPU backend, which
    /// samples the [`Sky`] passed in each [`FrameParams`] instead.
    pub fn set_sky_panorama(&mut self, rgba: &[u8], w: u32, h: u32) {
        if let BackendImpl::Gpu(g) = &mut self.inner {
            g.set_sky_panorama(rgba, w, h);
        }
    }

    /// Follow a window resize. CPU resizes its framebuffer lazily, so
    /// this only matters to the GPU swapchain — but it's safe to call
    /// for both.
    pub fn resize(&mut self, width: u32, height: u32) {
        match &mut self.inner {
            BackendImpl::Cpu(c) => c.resize(width, height),
            BackendImpl::Gpu(g) => g.resize(width, height),
        }
    }

    /// Composite `scene` from `camera` with `frame` params into the
    /// backend's frame buffer — **without presenting**. The CPU backend
    /// fills sky + runs the opticast compositor into an owned buffer;
    /// the GPU backend uploads/refreshes the scene, runs the compute
    /// marcher + sprite pass, and acquires (but does not present) the
    /// swapchain frame.
    ///
    /// Finish the frame with exactly one of [`present`](Self::present)
    /// (no overlay) or [`paint_egui`](Self::paint_egui) (UI overlay).
    /// Calling `render` again without finishing drops the pending frame.
    pub fn render(&mut self, scene: &mut Scene, camera: &Camera, frame: &FrameParams) {
        match &mut self.inner {
            BackendImpl::Cpu(c) => c.render(scene, camera, frame),
            BackendImpl::Gpu(g) => g.render(scene, camera, frame),
        }
    }

    /// Draw world-space [`Line3`] segments over the frame
    /// [`render`](Self::render) composited, using that frame's camera +
    /// projection + depth buffer. Call **after** [`render`](Self::render)
    /// and **before** [`present`](Self::present) /
    /// [`paint_egui`](Self::paint_egui) — the lines land in the
    /// framebuffer, so a subsequent `paint_egui` still draws its panels
    /// on top.
    ///
    /// `camera` must be the one the last frame rendered with (the
    /// projection is taken from that frame). Depth-tested segments
    /// (`Line3::depth_test`) are occluded by nearer rendered geometry;
    /// always-on-top segments ignore depth. See [`Line3`] for colour /
    /// width / blend semantics.
    pub fn draw_lines(&mut self, camera: &Camera, lines: &[Line3]) {
        match &mut self.inner {
            BackendImpl::Cpu(c) => c.draw_lines(camera, lines),
            BackendImpl::Gpu(g) => g.draw_lines(camera, lines),
        }
    }

    /// Upload (or replace) an RGBA8 image and return a stable [`ImageId`]
    /// to reference it in [`draw_images`](Self::draw_images). `rgba` is
    /// row-major, `width * height * 4` bytes, **straight** (un-premultiplied)
    /// alpha. The texture is retained until [`drop_image`](Self::drop_image),
    /// so the per-frame draw call stays cheap. Sampling is
    /// nearest-neighbour (pixel-art friendly — no blurring).
    ///
    /// Returns `None` for malformed input — a wrong byte count
    /// (`!= width·height·4`) or a zero dimension — so a bad upload can't be
    /// confused with the first valid id (`ImageId(0)`).
    pub fn upload_image(&mut self, rgba: &[u8], width: u32, height: u32) -> Option<ImageId> {
        if width == 0 || height == 0 || rgba.len() != (width as usize) * (height as usize) * 4 {
            return None;
        }
        Some(match &mut self.inner {
            BackendImpl::Cpu(c) => c.upload_image(rgba, width, height),
            BackendImpl::Gpu(g) => g.upload_image(rgba, width, height),
        })
    }

    /// Release a texture uploaded with [`upload_image`](Self::upload_image).
    /// The id must not be reused afterwards (a later `upload_image` may
    /// hand the slot back out under a fresh id).
    pub fn drop_image(&mut self, id: ImageId) {
        match &mut self.inner {
            BackendImpl::Cpu(c) => c.drop_image(id),
            BackendImpl::Gpu(g) => g.drop_image(id),
        }
    }

    /// Draw 2D [`ImageSprite`]s over the frame [`render`](Self::render)
    /// composited — flat textured quads placed in world space, using that
    /// frame's camera + projection + depth buffer. Same contract as
    /// [`draw_lines`](Self::draw_lines): call **after** [`render`](Self::render)
    /// and **before** [`present`](Self::present) / [`paint_egui`](Self::paint_egui).
    ///
    /// UVs are perspective-correct (no affine warp on an obliquely-viewed
    /// quad). Depth-tested sprites are occluded by nearer rendered
    /// geometry (with a bias to avoid z-fighting on a coincident face);
    /// the texture's straight alpha + the [`ImageSprite::tint`] composite
    /// over the scene. `camera` must be the one the last frame rendered.
    pub fn draw_images(&mut self, camera: &Camera, images: &[ImageSprite]) {
        if images.is_empty() {
            return;
        }
        let quads: Vec<QuadDraw> = images
            .iter()
            .filter_map(|s| resolve_quad(s, camera))
            .collect();
        if quads.is_empty() {
            return;
        }
        match &mut self.inner {
            BackendImpl::Cpu(c) => c.draw_images(camera, &quads),
            BackendImpl::Gpu(g) => g.draw_images(camera, &quads),
        }
    }

    /// Project a world point to window pixel coordinates `(x, y)` under
    /// the projection the **last frame** rendered with — the backend-correct
    /// `world → screen` inverse of [`view_ray`](Self::view_ray). `None`
    /// before the first frame or for a point at/behind the camera near
    /// plane.
    ///
    /// Both backends honour their own projection (CPU `setcamera`
    /// `hx/hy/hz`, GPU vertical-FOV pinhole), so hosts never reconstruct
    /// it themselves. The returned `(x, y)` may fall outside `[0, w) ×
    /// [0, h)` for points off-screen but in front of the camera.
    #[must_use]
    pub fn project_point(&self, camera: &Camera, world: [f32; 3]) -> Option<(f32, f32)> {
        match &self.inner {
            BackendImpl::Cpu(c) => c.project_point(camera, world),
            BackendImpl::Gpu(g) => g.project_point(camera, world),
        }
    }

    /// Screen→sprite pick: the nearest [`ImageSprite`] hit under window
    /// pixel `(x, y)`, resolving which texel was clicked. `sprites` is the
    /// same list passed to [`draw_images`](Self::draw_images) (image
    /// sprites are immediate-mode, so the caller owns the set). `None` for
    /// a miss.
    ///
    /// The ray is intersected with each quad's plane and mapped to its
    /// `uv` / source texel. A texel whose alpha is below the sprite's
    /// [`ImageSprite::alpha_cutoff`] (and any fully-transparent texel) is
    /// **see-through** — the pick passes through it to a sprite behind.
    /// For [`depth_test`](ImageSprite::depth_test) sprites the hit is
    /// rejected when nearer scene geometry occludes that pixel (shares the
    /// depth convention + bias of [`pick`](Self::pick); on the GPU backend
    /// the occlusion test costs a click-time depth readback).
    #[must_use]
    pub fn pick_image(
        &self,
        camera: &Camera,
        x: f64,
        y: f64,
        sprites: &[ImageSprite],
    ) -> Option<ImagePickHit> {
        if sprites.is_empty() {
            return None;
        }
        let dir = self.pixel_ray(camera, x, y)?;
        let dir = [dir[0] as f32, dir[1] as f32, dir[2] as f32];
        let dir_len = v_dot(dir, dir).sqrt();
        if dir_len < 1e-9 {
            return None;
        }
        let origin = [
            camera.pos[0] as f32,
            camera.pos[1] as f32,
            camera.pos[2] as f32,
        ];
        // Scene surface distance under this pixel (sky / no-hit → None);
        // used to occlude depth-tested sprites. Same metric as `pick`.
        let scene_t = self.pick_depth(x as u32, y as u32);

        let mut best: Option<ImagePickHit> = None;
        for sprite in sprites {
            // Reuse the render-path resolve (back-face cull included), so
            // a single-sided quad that isn't drawn also can't be picked.
            let Some(q) = resolve_quad(sprite, camera) else {
                continue;
            };
            let Some(([a, b], t)) = ray_quad_uv(origin, dir, &q.corners) else {
                continue; // miss / parallel / behind
            };
            let d_eucl = t * dir_len;
            if best.is_some_and(|cur| d_eucl >= cur.t) {
                continue; // a nearer sprite already won
            }
            let p = v_add(origin, v_scale(dir, t));

            let Some((iw, ih)) = self.image_dims(sprite.image) else {
                continue; // dropped / unknown image
            };
            let tx = ((a * iw as f32) as i32).clamp(0, iw as i32 - 1) as u32;
            let ty = ((b * ih as f32) as i32).clamp(0, ih as i32 - 1) as u32;

            // See-through test: a texel is solid when its alpha clears the
            // cutoff (and a fully-transparent texel is never solid).
            let cutoff_u8 = (sprite.alpha_cutoff.clamp(0.0, 1.0) * 255.0) as u32;
            let solid_thresh = cutoff_u8.max(1);
            if u32::from(self.image_alpha_at(sprite.image, tx, ty)) < solid_thresh {
                continue;
            }

            // Occlusion: a depth-tested sprite behind nearer geometry loses.
            if sprite.depth_test {
                if let Some(st) = scene_t {
                    if d_eucl > st + PICK_DEPTH_BIAS {
                        continue;
                    }
                }
            }

            best = Some(ImagePickHit {
                image: sprite.image,
                uv: [a, b],
                texel: (tx, ty),
                world: p,
                t: d_eucl,
            });
        }
        best
    }

    /// Source dimensions of an uploaded image, or `None` if the id was
    /// dropped / never uploaded. Internal helper for [`Self::pick_image`].
    fn image_dims(&self, id: ImageId) -> Option<(u32, u32)> {
        match &self.inner {
            BackendImpl::Cpu(c) => c.image_dims(id),
            BackendImpl::Gpu(g) => g.image_dims(id),
        }
    }

    /// Alpha byte of texel `(tx, ty)` in an uploaded image (`0` for an
    /// unknown id / out-of-range texel). Internal helper for
    /// [`Self::pick_image`].
    fn image_alpha_at(&self, id: ImageId, tx: u32, ty: u32) -> u8 {
        match &self.inner {
            BackendImpl::Cpu(c) => c.image_alpha_at(id, tx, ty),
            BackendImpl::Gpu(g) => g.image_alpha_at(id, tx, ty),
        }
    }

    /// Mirror the rendered 3D scene horizontally before display. The flip is
    /// applied *before* any egui overlay, so the UI stays upright while the
    /// viewport un-mirrors — a fix for the engine's left-handed render.
    /// Supported on both backends (CPU reverses the framebuffer rows; GPU
    /// mirrors the scene blit + line/image overlays). Picking/projection are
    /// unchanged, so a host that flips must mirror its cursor X (`width - x`)
    /// for ray casts.
    pub fn set_flip_x(&mut self, flip: bool) {
        match &mut self.inner {
            BackendImpl::Cpu(c) => c.set_flip_x(flip),
            BackendImpl::Gpu(g) => g.set_flip_x(flip),
        }
    }

    /// Present the frame [`render`](Self::render) composited, with no UI
    /// overlay. Pairs with `render`; use [`paint_egui`](Self::paint_egui)
    /// instead to overlay an egui UI before presenting.
    pub fn present(&mut self) {
        match &mut self.inner {
            BackendImpl::Cpu(c) => c.present(),
            BackendImpl::Gpu(g) => g.present(),
        }
    }

    /// Overlay an egui UI on the frame [`render`](Self::render)
    /// composited, then present it (`hud` feature). The host runs egui
    /// itself (e.g. `egui` + `egui-winit`) and passes the tessellated
    /// `jobs` ([`egui::Context::tessellate`]) and the per-frame
    /// `textures` delta from [`egui::FullOutput`]; `pixels_per_point` is
    /// the UI scale (`ctx.pixels_per_point()`).
    ///
    /// The GPU backend paints via `egui-wgpu`; the CPU backend
    /// software-rasterises the tessellation into its framebuffer. Use
    /// this **instead of** [`present`](Self::present) — both finish the
    /// frame.
    #[cfg(feature = "hud")]
    pub fn paint_egui(
        &mut self,
        jobs: &[egui::ClippedPrimitive],
        textures: &egui::TexturesDelta,
        pixels_per_point: f32,
    ) {
        match &mut self.inner {
            BackendImpl::Cpu(c) => c.paint_egui(jobs, textures, pixels_per_point),
            BackendImpl::Gpu(g) => g.paint_egui(jobs, textures, pixels_per_point),
        }
    }

    /// Register sprite models + instances. The CPU backend builds a
    /// per-instance draw list; the GPU backend builds an instanced
    /// model registry. Call once at setup (or again to replace).
    pub fn set_sprites(&mut self, set: &SpriteSet) -> Vec<SpriteModelId> {
        match &mut self.inner {
            BackendImpl::Cpu(c) => c.set_sprites(set),
            BackendImpl::Gpu(g) => g.set_sprites(set),
        }
        // A fresh sprite set replaces the instance world, so any
        // previously added dynamic instances + models are gone — drop their
        // handles and re-seat the model slotmap with `set.models.len()`
        // live ids `0..n` (model index = chain id on both backends).
        self.dyn_map = DynInstanceMap::default();
        self.model_map.reset(set.models.len());
        // A full sprite rebuild drops the dynamic + clip layers on both
        // backends (the GPU registry is replaced), so reset the clip +
        // character maps too.
        self.clip_map.reset();
        self.char_map.reset();
        self.char_instances.clear();
        self.streaming_map.reset();
        self.streaming_clips.clear();
        self.clip_meta.clear();
        self.clip_players.clear();
        (0..set.models.len() as u32)
            .map(|slot| SpriteModelId { slot, gen: 0 })
            .collect()
    }

    /// Re-register one sprite model's geometry after you've edited its
    /// content (a carve or recolour of its `kv6`). `model` is the
    /// [`SpriteModelId`] handed back by [`set_sprites`](Self::set_sprites);
    /// `kv6` is the model's **new** geometry — the caller owns the source
    /// of truth (e.g. a dense carve grid the surface-only `kv6` can't
    /// represent) and supplies the refreshed mesh here.
    ///
    /// This is a **backend-agnostic content refresh**, not a GPU upload:
    /// the renderer brings its stored model up to date however its active
    /// backend needs to. The instance set is left untouched (an edit never
    /// moves or adds an instance), so on the GPU backend only that one
    /// model's voxel data is re-uploaded — through a slack-backed
    /// suballocator, one model's bytes rather than the whole registry —
    /// while the CPU backend swaps the cached `kv6` into each instance of
    /// the model. Use [`set_sprites`](Self::set_sprites) to add/remove
    /// models or change the instance set.
    pub fn refresh_sprite_model(&mut self, model: SpriteModelId, kv6: &Kv6) {
        let Some(idx) = self.model_map.model_index(model) else {
            return; // stale / removed handle → no-op
        };
        match &mut self.inner {
            BackendImpl::Cpu(c) => c.update_sprite_model(idx, kv6),
            BackendImpl::Gpu(g) => g.update_sprite_model(idx, kv6),
        }
    }

    /// Like [`refresh_sprite_model`](Self::refresh_sprite_model) but also
    /// re-classifies the refreshed voxels into per-voxel material ids by
    /// colour (TV.3) via `material_map` — used by the material-aware streaming
    /// clip path so a re-uploaded frame keeps its per-voxel materials. An
    /// empty map matches `refresh_sprite_model`.
    pub fn refresh_sprite_model_with_materials(
        &mut self,
        model: SpriteModelId,
        kv6: &Kv6,
        material_map: &[(u32, u8)],
    ) {
        let Some(idx) = self.model_map.model_index(model) else {
            return; // stale / removed handle → no-op
        };
        match &mut self.inner {
            BackendImpl::Cpu(c) => {
                c.update_sprite_model_with_materials(idx, kv6, Some(material_map))
            }
            BackendImpl::Gpu(g) => g.update_sprite_model_with_materials(idx, kv6, material_map),
        }
    }

    /// Add one sprite instance of an already-registered `model` at world
    /// `pos`, **incrementally** — the cheap streaming-spawn path that both
    /// backends now share (GPU: append to the instance buffer, growing by
    /// powers of two; CPU: push one pre-posed [`Sprite`]). Returns a
    /// stable [`SpriteInstanceId`] for later removal.
    ///
    /// `model` must be a [`SpriteModelId`] from the current
    /// [`set_sprites`](Self::set_sprites) (a model registered there, even
    /// with zero initial instances). Dynamic instances live *after* the
    /// static set + any KFA limbs, so register those first.
    pub fn add_sprite_instance(&mut self, model: SpriteModelId, pos: [f32; 3]) -> SpriteInstanceId {
        self.add_sprite_instance_posed(
            model,
            DynSpriteTransform {
                pos,
                ..DynSpriteTransform::default()
            },
        )
    }

    /// Add one sprite instance of an already-registered `model`,
    /// pre-posed with the orientation in `xf` — the streaming-spawn path
    /// for objects that appear mid-flight already rotated (so there's no
    /// one-frame axis-aligned flash before the first
    /// [`set_sprite_instance_transform`](Self::set_sprite_instance_transform)).
    /// Otherwise identical to
    /// [`add_sprite_instance`](Self::add_sprite_instance) (which is just
    /// this with the identity basis). Returns a stable
    /// [`SpriteInstanceId`].
    ///
    /// A stale/removed `model` handle spawns nothing and returns a handle
    /// that is itself already stale (it resolves to no instance). `xf`'s
    /// basis must be non-singular; a degenerate one makes the instance
    /// silently skip drawing (see [`DynSpriteTransform`]).
    pub fn add_sprite_instance_posed(
        &mut self,
        model: SpriteModelId,
        xf: DynSpriteTransform,
    ) -> SpriteInstanceId {
        let Some(idx) = self.model_map.model_index(model) else {
            // Stale model → spawn nothing; hand back a sentinel id that
            // resolves to no live instance (a safe no-op everywhere).
            return SpriteInstanceId {
                slot: u32::MAX,
                gen: u32::MAX,
            };
        };
        let dyn_index = match &mut self.inner {
            BackendImpl::Cpu(c) => c.add_dyn_instance_posed(idx, xf),
            BackendImpl::Gpu(g) => g.add_dyn_instance_posed(idx, xf),
        };
        self.dyn_map.alloc(dyn_index as u32)
    }

    /// Remove a dynamic sprite instance added by
    /// [`add_sprite_instance`](Self::add_sprite_instance). O(1) on both
    /// backends (swap-remove); other dynamic handles stay valid. Returns
    /// `false` if the handle is stale / already removed.
    pub fn remove_sprite_instance(&mut self, id: SpriteInstanceId) -> bool {
        let Some(dyn_index) = self.dyn_map.dyn_index(id) else {
            return false;
        };
        let moved = match &mut self.inner {
            BackendImpl::Cpu(c) => c.remove_dyn_instance(dyn_index as usize),
            BackendImpl::Gpu(g) => g.remove_dyn_instance(dyn_index as usize),
        };
        self.dyn_map.remove(id, dyn_index, moved.map(|m| m as u32));
        true
    }

    /// Number of live dynamic sprite instances (those added via
    /// [`add_sprite_instance`](Self::add_sprite_instance)).
    #[must_use]
    pub fn dynamic_sprite_count(&self) -> usize {
        self.dyn_map.order.len()
    }

    /// Register one new sprite **model** incrementally from `kv6`,
    /// **without** rebuilding the existing model set — the streaming-in
    /// counterpart to [`add_sprite_instance`](Self::add_sprite_instance)
    /// for unique generated geometry (procedural asteroids, debris).
    /// Returns a stable [`SpriteModelId`] usable immediately with
    /// [`add_sprite_instance`](Self::add_sprite_instance) /
    /// [`add_sprite_instance_posed`](Self::add_sprite_instance_posed).
    ///
    /// Works before any [`set_sprites`](Self::set_sprites) (it establishes
    /// residency on the GPU backend's first model). The GPU backend
    /// appends one LOD chain to the resident registry (amortised O(model
    /// Define a global voxel **material** (TV stage): the opacity + blend
    /// mode that a per-voxel material id resolves to. The renderer owns one
    /// 256-entry palette shared by every model and grid.
    ///
    /// Id `0` is permanently [`Material::OPAQUE`] — the value every voxel
    /// without explicit material data resolves to — and **cannot** be
    /// redefined; passing `id == 0` is a no-op that returns `false`. Any
    /// other id returns `true`.
    ///
    /// While no translucent material is defined the renderer stays on the
    /// fully-opaque fast path, so this is inert until first called. See
    /// `PORTING-TRANSPARENCY.md`.
    pub fn define_material(&mut self, id: u8, mat: Material) -> bool {
        match &mut self.inner {
            BackendImpl::Cpu(c) => c.define_material(id, mat),
            BackendImpl::Gpu(g) => g.define_material(id, mat),
        }
    }

    /// The [`Material`] currently at palette `id` ([`Material::OPAQUE`] for
    /// any id never passed to [`define_material`](Self::define_material)).
    #[must_use]
    pub fn material(&self, id: u8) -> Material {
        match &self.inner {
            BackendImpl::Cpu(c) => c.material(id),
            BackendImpl::Gpu(g) => g.material(id),
        }
    }

    /// Set the **terrain** colour→material map (TV.4): pairs of `(rgb,
    /// material_id)` that make matching-colour world (grid) voxels translucent
    /// — glass walls, water pools. The materials themselves are defined via
    /// [`define_material`](Self::define_material). An empty map (the default)
    /// keeps all terrain opaque. The CPU backend composites these today; the
    /// GPU backend renders them once the TV.6 device path lands.
    pub fn set_terrain_materials(&mut self, map: &[(u32, u8)]) {
        match &mut self.inner {
            BackendImpl::Cpu(c) => c.set_terrain_materials(map),
            BackendImpl::Gpu(g) => g.set_terrain_materials(map),
        }
    }

    /// voxels)); the CPU backend pushes an axis-aligned template.
    pub fn add_sprite_model(&mut self, kv6: &Kv6) -> SpriteModelId {
        let model_index = match &mut self.inner {
            BackendImpl::Cpu(c) => c.add_model(kv6),
            BackendImpl::Gpu(g) => g.add_model(kv6),
        };
        self.model_map.alloc(model_index as u32)
    }

    /// Register a **mixed-material** sprite model (TV.3): `material_map` pairs
    /// a voxel RGB colour (`0xRRGGBB`) with a material id (defined via
    /// [`define_material`](Self::define_material)), so a single model can mix
    /// opaque and translucent voxels — an opaque window frame around glass, a
    /// bottle around a translucent potion. Voxels whose colour isn't in the
    /// map are opaque (material 0). Like [`add_sprite_model`](Self::add_sprite_model)
    /// otherwise.
    ///
    /// The CPU backend composites per-voxel materials today; the GPU backend
    /// carries the data and renders per-voxel materials once the TV.3b device
    /// path lands (until then it uses the instance's uniform material).
    pub fn add_sprite_model_with_materials(
        &mut self,
        kv6: &Kv6,
        material_map: &[(u32, u8)],
    ) -> SpriteModelId {
        let model_index = match &mut self.inner {
            BackendImpl::Cpu(c) => c.add_model_with_materials(kv6, material_map),
            BackendImpl::Gpu(g) => g.add_model_with_materials(kv6, material_map),
        };
        self.model_map.alloc(model_index as u32)
    }

    /// Remove a registered sprite model, freeing its voxel data. Returns
    /// `false` if `id` is stale / already removed.
    ///
    /// The model's slot is tombstoned **in place**: its id is never
    /// reused, so every other [`SpriteModelId`] stays valid (no remap).
    /// Existing instances of the removed model are **not** dropped here —
    /// they linger but draw as nothing on the GPU backend (the CPU
    /// backend keeps each instance's own kv6 clone, so they keep drawing
    /// until removed via
    /// [`remove_sprite_instance`](Self::remove_sprite_instance)); remove
    /// them when convenient. Call
    /// [`compact_sprite_models`](Self::compact_sprite_models) afterwards
    /// to reclaim the GPU buffer holes.
    pub fn remove_sprite_model(&mut self, id: SpriteModelId) -> bool {
        let Some(idx) = self.model_map.model_index(id) else {
            return false;
        };
        match &mut self.inner {
            BackendImpl::Cpu(c) => c.remove_model(idx),
            BackendImpl::Gpu(g) => g.remove_model(idx),
        }
        self.model_map.remove(id)
    }

    /// Reclaim the GPU buffer space left by
    /// [`remove_sprite_model`](Self::remove_sprite_model) by repacking the
    /// resident registry to its live models only. Model ids are preserved
    /// (no remap). O(live voxel volume) — call it when many models have
    /// been removed, not every frame. No-op on the CPU backend (which
    /// keeps cheap empty placeholders) and when nothing was removed.
    pub fn compact_sprite_models(&mut self) {
        match &mut self.inner {
            BackendImpl::Cpu(c) => c.compact_models(),
            BackendImpl::Gpu(g) => g.compact_models(),
        }
    }

    /// Update one dynamic instance's full pose (position + orientation)
    /// for this frame. `id` is from
    /// [`add_sprite_instance`](Self::add_sprite_instance) /
    /// [`add_sprite_instance_posed`](Self::add_sprite_instance_posed). A
    /// stale / removed handle is a no-op.
    ///
    /// For many instances per frame prefer
    /// [`set_sprite_instance_transforms`](Self::set_sprite_instance_transforms):
    /// the GPU backend flushes all pending pose changes to the device
    /// once per [`render`](Self::render), so a per-instance call here is
    /// still O(1) device work, but the batch variant avoids re-walking
    /// the slotmap.
    pub fn set_sprite_instance_transform(&mut self, id: SpriteInstanceId, xf: DynSpriteTransform) {
        let Some(dyn_index) = self.dyn_map.dyn_index(id) else {
            return;
        };
        match &mut self.inner {
            BackendImpl::Cpu(c) => c.set_dyn_instance_transform(dyn_index as usize, xf),
            BackendImpl::Gpu(g) => g.set_dyn_instance_transform(dyn_index as usize, xf),
        }
    }

    /// Batch form of
    /// [`set_sprite_instance_transform`](Self::set_sprite_instance_transform)
    /// — apply many `(instance, pose)` updates in one call. Stale handles
    /// in `updates` are skipped. On the GPU backend this marks the
    /// instance buffer dirty once and uploads the new poses a single time
    /// at the next [`render`](Self::render), so spinning a whole cluster
    /// of instances per frame is one device upload, not one per instance.
    pub fn set_sprite_instance_transforms(
        &mut self,
        updates: &[(SpriteInstanceId, DynSpriteTransform)],
    ) {
        for &(id, xf) in updates {
            let Some(dyn_index) = self.dyn_map.dyn_index(id) else {
                continue;
            };
            match &mut self.inner {
                BackendImpl::Cpu(c) => c.set_dyn_instance_transform(dyn_index as usize, xf),
                BackendImpl::Gpu(g) => g.set_dyn_instance_transform(dyn_index as usize, xf),
            }
        }
    }

    /// Set sprite instance `id`'s voxel-material id (TV stage) — indexes the
    /// global palette defined via [`define_material`](Self::define_material)
    /// for this whole instance's opacity + blend mode. `0` (the default) is
    /// opaque. Stale handles are ignored.
    ///
    /// Only the CPU backend composites translucent sprites today; the GPU
    /// backend retains the value for the forthcoming device-side path (see
    /// `PORTING-TRANSPARENCY.md`).
    pub fn set_sprite_instance_material(&mut self, id: SpriteInstanceId, material: u8) {
        let Some(dyn_index) = self.dyn_map.dyn_index(id) else {
            return;
        };
        match &mut self.inner {
            BackendImpl::Cpu(c) => c.set_dyn_instance_material(dyn_index as usize, material),
            BackendImpl::Gpu(g) => g.set_dyn_instance_material(dyn_index as usize, material),
        }
    }

    /// Set sprite instance `id`'s per-instance alpha multiplier (TV stage),
    /// `0..=255` (`255` = unscaled). Scales the material's opacity so an
    /// effect can fade out by cheap per-frame updates without re-uploading
    /// its volume. Stale handles are ignored.
    pub fn set_sprite_instance_alpha(&mut self, id: SpriteInstanceId, alpha_mul: u8) {
        let Some(dyn_index) = self.dyn_map.dyn_index(id) else {
            return;
        };
        match &mut self.inner {
            BackendImpl::Cpu(c) => c.set_dyn_instance_alpha(dyn_index as usize, alpha_mul),
            BackendImpl::Gpu(g) => g.set_dyn_instance_alpha(dyn_index as usize, alpha_mul),
        }
    }

    // ---- animated voxel clips (VCL.4) ------------------------------------

    /// Register an animated voxel clip ("GIF/MP4 for voxels"): decode all
    /// its frames and upload the flipbook to the active backend (GPU: one
    /// LOD chain per frame; CPU: a cached dense grid per frame). Returns a
    /// [`VoxelClipId`] to spawn instances of it via
    /// [`add_clip_instance_posed`](Self::add_clip_instance_posed).
    ///
    /// Build the [`DecodedClip`] from a `.rvc` via
    /// [`VoxelClip::decode`](roxlap_formats::voxel_clip::VoxelClip::decode).
    /// Like [`add_sprite_model`](Self::add_sprite_model), this works before
    /// any [`set_sprites`](Self::set_sprites); a later `set_sprites`
    /// **drops** all registered clips (re-register afterwards).
    pub fn add_voxel_clip(&mut self, clip: &DecodedClip) -> VoxelClipId {
        self.add_voxel_clip_with_materials(clip, &[])
    }

    /// Register a **mixed-material** animated voxel clip (TV.3): the clip
    /// analogue of
    /// [`add_sprite_model_with_materials`](Self::add_sprite_model_with_materials).
    /// `material_map` pairs a voxel RGB colour (`0xRRGGBB`) with a material id
    /// (defined via [`define_material`](Self::define_material)), classifying
    /// every frame's voxels so an animated clip can mix opaque and translucent
    /// voxels — an opaque torch handle around an additive flame, a spinning
    /// glass orb. Voxels whose colour isn't in the map stay opaque
    /// (material 0). Like [`add_voxel_clip`](Self::add_voxel_clip) otherwise.
    pub fn add_voxel_clip_with_materials(
        &mut self,
        clip: &DecodedClip,
        material_map: &[(u32, u8)],
    ) -> VoxelClipId {
        let clip_index = match &mut self.inner {
            BackendImpl::Cpu(c) => c.add_voxel_clip_with_materials(clip, material_map),
            BackendImpl::Gpu(g) => g.add_voxel_clip_with_materials(clip, material_map),
        };
        // Capture metadata for editor queries + #6 auto-play; clip indices
        // are sequential and parallel to `clip_meta`.
        debug_assert_eq!(clip_index, self.clip_meta.len());
        self.clip_meta.push(ClipMeta {
            dims: clip.dims,
            pivot: clip.pivot,
            voxel_world_size: clip.voxel_world_size,
            durations: clip.durations.clone(),
            loop_mode: clip.loop_mode,
            material_map: material_map.to_vec(),
        });
        self.clip_map.alloc(clip_index as u32)
    }

    /// Remove a registered clip, freeing its per-frame volumes. Instances
    /// of it linger but draw nothing until removed via
    /// [`remove_sprite_instance`](Self::remove_sprite_instance). Returns
    /// `false` if `id` is stale / already removed.
    pub fn remove_voxel_clip(&mut self, id: VoxelClipId) -> bool {
        let Some(clip_index) = self.clip_map.clip_index(id) else {
            return false;
        };
        match &mut self.inner {
            BackendImpl::Cpu(c) => c.remove_voxel_clip(clip_index),
            BackendImpl::Gpu(g) => g.remove_voxel_clip(clip_index),
        }
        self.clip_map.remove(id)
    }

    /// Spawn an instance of clip `clip`, posed by `xf`, starting on frame
    /// 0. Returns a [`SpriteInstanceId`] — a clip instance is a dynamic
    /// sprite instance, so move it with
    /// [`set_sprite_instance_transform`](Self::set_sprite_instance_transform),
    /// advance its frame with
    /// [`set_clip_instance_frame`](Self::set_clip_instance_frame), and drop
    /// it with [`remove_sprite_instance`](Self::remove_sprite_instance).
    /// A stale `clip` handle yields an instance id that resolves to nothing
    /// (a safe no-op everywhere).
    ///
    /// This instance has **no playback clock**: drive its frame yourself via
    /// [`set_clip_instance_frame`](Self::set_clip_instance_frame) (frame-based
    /// scrubbing). For *clock*-based control — auto-advance, play/pause, or
    /// [`set_clip_instance_clock_ms`](Self::set_clip_instance_clock_ms)
    /// scrubbing — spawn with
    /// [`add_clip_instance_playing`](Self::add_clip_instance_playing) instead
    /// (the player-control methods no-op on an instance with no player).
    pub fn add_clip_instance_posed(
        &mut self,
        clip: VoxelClipId,
        xf: DynSpriteTransform,
    ) -> SpriteInstanceId {
        let Some(clip_index) = self.clip_map.clip_index(clip) else {
            return SpriteInstanceId {
                slot: u32::MAX,
                gen: u32::MAX,
            };
        };
        let dyn_index = match &mut self.inner {
            BackendImpl::Cpu(c) => c.add_clip_instance(clip_index, xf),
            BackendImpl::Gpu(g) => g.add_clip_instance(clip_index, xf),
        };
        self.dyn_map.alloc(dyn_index as u32)
    }

    /// Select which frame a clip instance shows — the per-frame playback
    /// step. Cheap on both backends (GPU: swap the instance's model id;
    /// CPU: select the cached frame grid), with no volume re-upload. Drive
    /// it from a playback clock via
    /// [`DecodedClip::frame_at`](roxlap_formats::voxel_clip::DecodedClip::frame_at).
    /// No-op on a stale id or a non-clip instance.
    pub fn set_clip_instance_frame(&mut self, id: SpriteInstanceId, frame: u32) {
        let Some(dyn_index) = self.dyn_map.dyn_index(id) else {
            return;
        };
        match &mut self.inner {
            BackendImpl::Cpu(c) => c.set_clip_frame(dyn_index as usize, frame as usize),
            BackendImpl::Gpu(g) => g.set_clip_frame(dyn_index as usize, frame as usize),
        }
    }

    // ---- clip queries (editor inspector) ---------------------------------

    /// Frame count of a registered flipbook clip, or `None` if `id` is
    /// stale. (Same as `clip_metadata(id)?.frame_count`, without the clone.)
    #[must_use]
    pub fn clip_frame_count(&self, id: VoxelClipId) -> Option<usize> {
        let idx = self.clip_map.clip_index(id)?;
        Some(self.clip_meta[idx].durations.len())
    }

    /// Inspector metadata (dims / pivot / scale / loop mode / per-frame
    /// durations) of a registered flipbook clip, or `None` if `id` is stale
    /// — so an editor needn't shadow the source [`DecodedClip`].
    #[must_use]
    pub fn clip_metadata(&self, id: VoxelClipId) -> Option<ClipMetadata> {
        let idx = self.clip_map.clip_index(id)?;
        let m = &self.clip_meta[idx];
        Some(ClipMetadata {
            dims: m.dims,
            pivot: m.pivot,
            voxel_world_size: m.voxel_world_size,
            loop_mode: m.loop_mode,
            frame_count: m.durations.len(),
            durations: m.durations.clone(),
            total_ms: m
                .durations
                .iter()
                .fold(0u32, |acc, &d| acc.saturating_add(d)),
        })
    }

    /// Which frame a clip instance is currently showing (the timeline
    /// scrubber's read-back), or `None` if `id` isn't a live clip instance.
    #[must_use]
    pub fn get_clip_instance_frame(&self, id: SpriteInstanceId) -> Option<u32> {
        let dyn_index = self.dyn_map.dyn_index(id)? as usize;
        let frame = match &self.inner {
            BackendImpl::Cpu(c) => c.clip_instance_frame(dyn_index),
            BackendImpl::Gpu(g) => g.clip_instance_frame(dyn_index),
        }?;
        u32::try_from(frame).ok()
    }

    /// Re-upload a **single** `frame` of registered clip `id` in place — the
    /// editor's one-voxel paint, O(1 frame) instead of `remove_voxel_clip` +
    /// `add_voxel_clip` (which rebuilds all N volumes). `vf` must fit the
    /// clip's fixed `dims`. Returns `false` on a stale `id`, an out-of-range
    /// `frame`, or a frame that fails the clip's layout (so it can't corrupt
    /// the flipbook).
    pub fn update_clip_frame(&mut self, id: VoxelClipId, frame: u32, vf: &VoxelFrame) -> bool {
        let Some(clip_index) = self.clip_map.clip_index(id) else {
            return false;
        };
        let m = &self.clip_meta[clip_index];
        let (dims, pivot, vws) = (m.dims, m.pivot, m.voxel_world_size);
        if vf.validate(dims).is_err() {
            return false;
        }
        // Re-classify with the clip's registered colour→material map (TV.3) so
        // an in-place frame edit keeps the clip's per-voxel materials.
        let material_map = m.material_map.clone();
        let frame = frame as usize;
        match &mut self.inner {
            BackendImpl::Cpu(c) => {
                c.update_clip_frame(clip_index, frame, vf, dims, pivot, &material_map)
            }
            BackendImpl::Gpu(g) => {
                g.update_clip_frame(clip_index, frame, vf, dims, pivot, vws, &material_map)
            }
        }
    }

    // ---- streaming voxel clips (#3) --------------------------------------

    /// Register a **streaming** voxel clip — `O(1-frame)` memory (one sprite
    /// model + the compact encoded stream) rather than the N-volume flipbook
    /// [`add_voxel_clip`](Self::add_voxel_clip) builds, for huge clips where
    /// N frames are too costly to hold resident. Builds the model from frame
    /// 0; advance it with
    /// [`set_streaming_clip_frame`](Self::set_streaming_clip_frame). Spawn
    /// instances with
    /// [`add_streaming_clip_instance`](Self::add_streaming_clip_instance) —
    /// note that, unlike a flipbook, **all** instances of a streaming clip
    /// share its one model and so always show the same (current) frame.
    ///
    /// Takes the *encoded* [`VoxelClip`] (not a [`DecodedClip`]) — the whole
    /// point is to avoid materialising every frame.
    ///
    /// # Errors
    /// [`DecodeError`] if the clip's frame stream is empty or doesn't begin
    /// with a keyframe.
    pub fn add_streaming_clip(&mut self, clip: &VoxelClip) -> Result<StreamingClipId, DecodeError> {
        self.add_streaming_clip_with_materials(clip, &[])
    }

    /// Register a **mixed-material** streaming voxel clip (TV.3): the streaming
    /// analogue of
    /// [`add_voxel_clip_with_materials`](Self::add_voxel_clip_with_materials).
    /// `material_map` pairs a voxel RGB colour with a material id (defined via
    /// [`define_material`](Self::define_material)); it is re-applied on every
    /// per-frame re-upload, so the single streamed model keeps its per-voxel
    /// materials as the clip advances. An empty map is identical to
    /// [`add_streaming_clip`](Self::add_streaming_clip).
    ///
    /// # Errors
    /// As [`add_streaming_clip`](Self::add_streaming_clip).
    pub fn add_streaming_clip_with_materials(
        &mut self,
        clip: &VoxelClip,
        material_map: &[(u32, u8)],
    ) -> Result<StreamingClipId, DecodeError> {
        let cursor = StreamingClip::new(clip)?;
        let dims = cursor.dims();
        let pivot = cursor.pivot();
        let kv6 = cursor.current_frame().to_kv6(dims, pivot);
        let model = self.add_sprite_model_with_materials(&kv6, material_map);
        let index = self.streaming_clips.len() as u32;
        self.streaming_clips.push(Some(StreamingClipState {
            cursor,
            model,
            dims,
            pivot,
            material_map: material_map.to_vec(),
        }));
        Ok(self.streaming_map.alloc(index))
    }

    /// Spawn an instance of streaming clip `id`, posed by `xf`. Returns a
    /// [`SpriteInstanceId`] — move it with
    /// [`set_sprite_instance_transform`](Self::set_sprite_instance_transform)
    /// and drop it with
    /// [`remove_sprite_instance`](Self::remove_sprite_instance), like any
    /// dynamic instance. All instances of one streaming clip share its single
    /// model. A stale `id` yields a no-op instance handle.
    pub fn add_streaming_clip_instance(
        &mut self,
        id: StreamingClipId,
        xf: DynSpriteTransform,
    ) -> StreamingInstanceId {
        let model = self
            .streaming_map
            .index(id)
            .and_then(|idx| self.streaming_clips[idx].as_ref())
            .map(|s| s.model);
        let inst = match model {
            Some(model) => self.add_sprite_instance_posed(model, xf),
            None => SpriteInstanceId {
                slot: u32::MAX,
                gen: u32::MAX,
            },
        };
        StreamingInstanceId(inst)
    }

    /// Re-pose a streaming-clip instance (world transform). No-op on a stale
    /// handle.
    pub fn set_streaming_instance_transform(
        &mut self,
        id: StreamingInstanceId,
        xf: DynSpriteTransform,
    ) {
        self.set_sprite_instance_transform(id.0, xf);
    }

    /// Remove a streaming-clip instance. Returns `false` if `id` is stale.
    pub fn remove_streaming_instance(&mut self, id: StreamingInstanceId) -> bool {
        self.remove_sprite_instance(id.0)
    }

    /// Advance a streaming clip to `frame`: seek the cursor and re-upload its
    /// single model — the per-frame streaming step (one volume re-upload,
    /// vs the flipbook's cheap model-select). Updates **every** instance of
    /// the clip at once. Drive it from a clock via
    /// [`frame_at`](roxlap_formats::voxel_clip::frame_at). No-op on a stale
    /// id; `frame` is clamped to the last.
    pub fn set_streaming_clip_frame(&mut self, id: StreamingClipId, frame: u32) {
        let Some(idx) = self.streaming_map.index(id) else {
            return;
        };
        let Some((model, kv6, material_map)) = self.streaming_clips[idx].as_mut().and_then(|s| {
            let vf = s.cursor.seek(frame as usize).ok()?;
            Some((s.model, vf.to_kv6(s.dims, s.pivot), s.material_map.clone()))
        }) else {
            return;
        };
        self.refresh_sprite_model_with_materials(model, &kv6, &material_map);
    }

    /// Remove a streaming clip: free its model and drop the cursor (the
    /// memory win for huge clips). Instances linger but draw nothing until
    /// removed. Returns `false` if `id` is stale / already removed.
    pub fn remove_streaming_clip(&mut self, id: StreamingClipId) -> bool {
        let Some(idx) = self.streaming_map.index(id) else {
            return false;
        };
        let model = self.streaming_clips[idx].as_ref().map(|s| s.model);
        self.streaming_clips[idx] = None;
        if let Some(model) = model {
            self.remove_sprite_model(model);
        }
        self.streaming_map.remove(id)
    }

    // ---- auto-advancing clip players (#6) --------------------------------

    /// Spawn a flipbook-clip instance that **plays itself**: like
    /// [`add_clip_instance_posed`](Self::add_clip_instance_posed), but the
    /// facade tracks a playback clock so a single
    /// [`advance_voxel_clips`](Self::advance_voxel_clips) call advances every
    /// such instance — no per-frame `frame_at` + `set_clip_instance_frame`
    /// bookkeeping in the host. `speed_q8` is the Q8 playback rate (`256` =
    /// 1×); `start_phase_ms` offsets the clock (stagger copies of one clip).
    /// A stale `clip` yields a no-op instance handle and no player.
    pub fn add_clip_instance_playing(
        &mut self,
        clip: VoxelClipId,
        xf: DynSpriteTransform,
        speed_q8: i32,
        start_phase_ms: u32,
    ) -> SpriteInstanceId {
        let Some(clip_index) = self.clip_map.clip_index(clip) else {
            return SpriteInstanceId {
                slot: u32::MAX,
                gen: u32::MAX,
            };
        };
        let meta = &self.clip_meta[clip_index];
        let clock = ClipClock {
            durations: meta.durations.clone(),
            loop_mode: meta.loop_mode,
            speed_q8,
            clock_ms: f64::from(start_phase_ms),
        };
        let inst = self.add_clip_instance_posed(clip, xf);
        self.clip_players.push(ClipPlayer {
            target: PlayerTarget::Flipbook(inst),
            clock,
            paused: false,
        });
        inst
    }

    /// Give a streaming clip ([`add_streaming_clip`](Self::add_streaming_clip))
    /// its own playback clock, advanced by
    /// [`advance_voxel_clips`](Self::advance_voxel_clips). A streaming clip's
    /// frame is per-clip (all its instances share one model), so this is
    /// keyed on the clip, not an instance — register instances separately
    /// with
    /// [`add_streaming_clip_instance`](Self::add_streaming_clip_instance).
    /// No-op on a stale `clip`.
    ///
    /// Control the player (play/pause/scrub) via
    /// [`set_streaming_clip_paused`](Self::set_streaming_clip_paused) /
    /// [`set_streaming_clip_speed`](Self::set_streaming_clip_speed) /
    /// [`set_streaming_clip_clock_ms`](Self::set_streaming_clip_clock_ms), the
    /// per-clip analogues of the flipbook `set_clip_instance_*` methods.
    pub fn play_streaming_clip(
        &mut self,
        clip: StreamingClipId,
        speed_q8: i32,
        start_phase_ms: u32,
    ) {
        let Some(idx) = self.streaming_map.index(clip) else {
            return;
        };
        let Some(state) = self.streaming_clips[idx].as_ref() else {
            return;
        };
        let clock = ClipClock {
            durations: state.cursor.durations().to_vec(),
            loop_mode: state.cursor.loop_mode(),
            speed_q8,
            clock_ms: f64::from(start_phase_ms),
        };
        self.clip_players.push(ClipPlayer {
            target: PlayerTarget::Streaming(clip),
            clock,
            paused: false,
        });
    }

    /// Advance every auto-playing clip ([`add_clip_instance_playing`] /
    /// [`play_streaming_clip`]) by `dt` seconds: tick each clock, resolve its
    /// frame via [`frame_at`](roxlap_formats::voxel_clip::frame_at), and
    /// apply it. Players whose instance / clip was removed are pruned. Call
    /// once per frame.
    ///
    /// [`add_clip_instance_playing`]: Self::add_clip_instance_playing
    /// [`play_streaming_clip`]: Self::play_streaming_clip
    pub fn advance_voxel_clips(&mut self, dt: f64) {
        // Phase 1: tick clocks → (target, frame), pruning dead players.
        // Borrow only the maps (disjoint from `clip_players`).
        let dyn_map = &self.dyn_map;
        let streaming_map = &self.streaming_map;
        let mut updates: Vec<(PlayerTarget, u32)> = Vec::new();
        self.clip_players.retain_mut(|p| {
            let alive = match p.target {
                PlayerTarget::Flipbook(inst) => dyn_map.dyn_index(inst).is_some(),
                PlayerTarget::Streaming(clip) => streaming_map.index(clip).is_some(),
            };
            if !alive {
                return false;
            }
            // A paused player keeps its clock + frame (the editor's pause).
            if !p.paused {
                updates.push((p.target, p.clock.tick(dt)));
            }
            true
        });
        // Phase 2: apply (borrows self mutably, disjoint from the above).
        for (target, frame) in updates {
            self.apply_player_frame(target, frame);
        }
    }

    /// Apply a resolved frame to a player's target (flipbook instance vs.
    /// streaming clip).
    fn apply_player_frame(&mut self, target: PlayerTarget, frame: u32) {
        match target {
            PlayerTarget::Flipbook(inst) => self.set_clip_instance_frame(inst, frame),
            PlayerTarget::Streaming(clip) => self.set_streaming_clip_frame(clip, frame),
        }
    }

    /// Find the auto-player driving flipbook instance `inst`, if any.
    fn flipbook_player_mut(&mut self, inst: SpriteInstanceId) -> Option<&mut ClipPlayer> {
        self.clip_players
            .iter_mut()
            .find(|p| matches!(p.target, PlayerTarget::Flipbook(i) if i == inst))
    }

    /// Pause / resume the auto-player driving clip instance `id` (the
    /// editor's play/pause). No-op if `id` has no player.
    pub fn set_clip_instance_paused(&mut self, id: SpriteInstanceId, paused: bool) {
        if let Some(p) = self.flipbook_player_mut(id) {
            p.paused = paused;
        }
    }

    /// Whether clip instance `id`'s auto-player is paused, or `None` if it
    /// has no player.
    #[must_use]
    pub fn is_clip_instance_paused(&self, id: SpriteInstanceId) -> Option<bool> {
        self.clip_players
            .iter()
            .find(|p| matches!(p.target, PlayerTarget::Flipbook(i) if i == id))
            .map(|p| p.paused)
    }

    /// Set the playback speed (Q8: `256` = 1×, negative = reverse) of clip
    /// instance `id`'s auto-player. No-op if `id` has no player.
    pub fn set_clip_instance_speed(&mut self, id: SpriteInstanceId, speed_q8: i32) {
        if let Some(p) = self.flipbook_player_mut(id) {
            p.clock.speed_q8 = speed_q8;
        }
    }

    /// **Scrub**: set clip instance `id`'s playback clock to `clock_ms` and
    /// immediately show the matching frame (works while paused). No-op if
    /// `id` has no player.
    pub fn set_clip_instance_clock_ms(&mut self, id: SpriteInstanceId, clock_ms: f64) {
        let Some((target, frame)) = self.flipbook_player_mut(id).map(|p| {
            p.clock.clock_ms = clock_ms;
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let frame = frame_at(
                &p.clock.durations,
                p.clock.loop_mode,
                clock_ms.max(0.0) as u32,
            ) as u32;
            (p.target, frame)
        }) else {
            return;
        };
        self.apply_player_frame(target, frame);
    }

    /// Clip instance `id`'s current playback-clock position (ms), or `None`
    /// if it has no player — the scrubber's read-back.
    #[must_use]
    pub fn clip_instance_clock_ms(&self, id: SpriteInstanceId) -> Option<f64> {
        self.clip_players
            .iter()
            .find(|p| matches!(p.target, PlayerTarget::Flipbook(i) if i == id))
            .map(|p| p.clock.clock_ms)
    }

    /// Find the auto-player driving streaming clip `clip`, if any (a player
    /// registered via [`play_streaming_clip`](Self::play_streaming_clip)).
    fn streaming_player_mut(&mut self, clip: StreamingClipId) -> Option<&mut ClipPlayer> {
        self.clip_players
            .iter_mut()
            .find(|p| matches!(p.target, PlayerTarget::Streaming(c) if c == clip))
    }

    /// Pause / resume a streaming clip's auto-player
    /// ([`play_streaming_clip`](Self::play_streaming_clip)). No-op if `clip`
    /// has no player.
    pub fn set_streaming_clip_paused(&mut self, clip: StreamingClipId, paused: bool) {
        if let Some(p) = self.streaming_player_mut(clip) {
            p.paused = paused;
        }
    }

    /// Whether streaming clip `clip`'s auto-player is paused, or `None` if it
    /// has no player.
    #[must_use]
    pub fn is_streaming_clip_paused(&self, clip: StreamingClipId) -> Option<bool> {
        self.clip_players
            .iter()
            .find(|p| matches!(p.target, PlayerTarget::Streaming(c) if c == clip))
            .map(|p| p.paused)
    }

    /// Set the playback speed (Q8: `256` = 1×, negative = reverse) of
    /// streaming clip `clip`'s auto-player. No-op if `clip` has no player.
    pub fn set_streaming_clip_speed(&mut self, clip: StreamingClipId, speed_q8: i32) {
        if let Some(p) = self.streaming_player_mut(clip) {
            p.clock.speed_q8 = speed_q8;
        }
    }

    /// **Scrub** a streaming clip: set its auto-player's clock to `clock_ms`
    /// and immediately show the matching frame (works while paused). No-op if
    /// `clip` has no player.
    pub fn set_streaming_clip_clock_ms(&mut self, clip: StreamingClipId, clock_ms: f64) {
        let Some((target, frame)) = self.streaming_player_mut(clip).map(|p| {
            p.clock.clock_ms = clock_ms;
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let frame = frame_at(
                &p.clock.durations,
                p.clock.loop_mode,
                clock_ms.max(0.0) as u32,
            ) as u32;
            (p.target, frame)
        }) else {
            return;
        };
        self.apply_player_frame(target, frame);
    }

    /// Streaming clip `clip`'s current playback-clock position (ms), or
    /// `None` if it has no player — the scrubber's read-back.
    #[must_use]
    pub fn streaming_clip_clock_ms(&self, clip: StreamingClipId) -> Option<f64> {
        self.clip_players
            .iter()
            .find(|p| matches!(p.target, PlayerTarget::Streaming(c) if c == clip))
            .map(|p| p.clock.clock_ms)
    }

    // ---- animated characters (VCL.6) -------------------------------------

    /// Register an animated character (RKC v3): upload its meshes as sprite
    /// models + its embedded voxel clips as flipbooks, then spawn one
    /// renderer instance **per bone attachment** — a static mesh sits at
    /// its bone, a clip attachment plays back on its own clock. `clip`
    /// selects a skeletal animation clip to drive the bones (`None` =
    /// rest pose). Returns a [`CharacterId`]; advance it each frame with
    /// [`advance_character`](Self::advance_character).
    ///
    /// Like clips, this works before any [`set_sprites`](Self::set_sprites);
    /// a later `set_sprites` drops all registered characters.
    pub fn add_character(&mut self, ch: &Character, clip: Option<usize>) -> CharacterId {
        // 1. Meshes → sprite models.
        let model_ids: Vec<SpriteModelId> =
            ch.meshes.iter().map(|m| self.add_sprite_model(m)).collect();
        // 2. Voxel clips → flipbooks; keep each one's timing for the clocks.
        let clip_regs: Vec<Option<(VoxelClipId, Vec<u32>, LoopMode)>> = ch
            .voxel_clips
            .iter()
            .map(|vc| {
                vc.decode().ok().map(|d| {
                    let id = self.add_voxel_clip(&d);
                    (id, d.durations, d.loop_mode)
                })
            })
            .collect();
        // 3. Build + solve the skeleton (rest pose → bone transforms).
        let mut skeleton = ch.to_kfa_sprite(clip);
        solve_kfa_limbs(&mut skeleton);
        // 4. One instance per attachment, posed by bone × local_offset.
        let mut attaches = Vec::new();
        for (bi, bone) in ch.bones.iter().enumerate() {
            let limb = &skeleton.limbs[bi];
            for att in &bone.attachments {
                let (s, h, f, p) =
                    compose_attachment(limb.s, limb.h, limb.f, limb.p, &att.local_offset);
                let xf = DynSpriteTransform {
                    pos: p,
                    right: s,
                    up: h,
                    forward: f,
                };
                match att.target {
                    MeshRef::Static(mi) => {
                        if let Some(&mid) = model_ids.get(mi) {
                            let inst = self.add_sprite_instance_posed(mid, xf);
                            attaches.push(AttachInst {
                                bone: bi,
                                local_offset: att.local_offset,
                                inst,
                                clip: None,
                            });
                        }
                    }
                    MeshRef::Clip(ci) => {
                        if let Some(Some((cid, durations, loop_mode))) = clip_regs.get(ci) {
                            let inst = self.add_clip_instance_posed(*cid, xf);
                            attaches.push(AttachInst {
                                bone: bi,
                                local_offset: att.local_offset,
                                inst,
                                clip: Some(ClipClock {
                                    durations: durations.clone(),
                                    loop_mode: *loop_mode,
                                    speed_q8: att.playback.speed_q8,
                                    clock_ms: f64::from(att.playback.start_phase_ms),
                                }),
                            });
                        }
                    }
                }
            }
        }
        let clips: Vec<VoxelClipId> = clip_regs
            .iter()
            .filter_map(|r| r.as_ref().map(|(cid, _, _)| *cid))
            .collect();
        let idx = self.char_instances.len();
        self.char_instances.push(CharInstance {
            skeleton,
            attaches,
            models: model_ids,
            clips,
        });
        self.char_map.alloc(idx as u32)
    }

    /// Advance a character by `dt` seconds: tick its skeletal animation +
    /// each clip attachment's clock, then re-pose every attachment
    /// (bone × local_offset) and select each clip's current frame. No-op on
    /// a stale id.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn advance_character(&mut self, id: CharacterId, dt: f64) {
        let Some(idx) = self.char_map.index(id) else {
            return;
        };
        // Phase 1: solve the skeleton + compute each attachment's update,
        // borrowing only `char_instances[idx]`.
        let updates: Vec<(SpriteInstanceId, DynSpriteTransform, Option<u32>)> = {
            let CharInstance {
                skeleton, attaches, ..
            } = &mut self.char_instances[idx];
            skeleton.animsprite((dt * 1000.0) as i32);
            solve_kfa_limbs(skeleton);
            attaches
                .iter_mut()
                .map(|a| {
                    let limb = &skeleton.limbs[a.bone];
                    let (s, h, f, p) =
                        compose_attachment(limb.s, limb.h, limb.f, limb.p, &a.local_offset);
                    let xf = DynSpriteTransform {
                        pos: p,
                        right: s,
                        up: h,
                        forward: f,
                    };
                    let frame = a.clip.as_mut().map(|c| c.tick(dt));
                    (a.inst, xf, frame)
                })
                .collect()
        };
        // Phase 2: apply via the facade primitives (disjoint from
        // `char_instances`).
        for (inst, xf, frame) in updates {
            self.set_sprite_instance_transform(inst, xf);
            if let Some(f) = frame {
                self.set_clip_instance_frame(inst, f);
            }
        }
    }

    /// Move/re-orient a character to a new world transform `xf` (the root
    /// limb's world pose) **without** ticking its animation or clip clocks —
    /// a teleport that holds the current animation frame (e.g. dragging a
    /// paused character in an editor). Re-solves the skeleton from the new
    /// root + re-poses every attachment; clip frames are left as-is. No-op on
    /// a stale id.
    pub fn set_character_world_transform(&mut self, id: CharacterId, xf: DynSpriteTransform) {
        let Some(idx) = self.char_map.index(id) else {
            return;
        };
        // Phase 1: set the root pose + re-solve (no animsprite), then compute
        // each attachment's new transform — borrowing only `char_instances`.
        let updates: Vec<(SpriteInstanceId, DynSpriteTransform)> = {
            let CharInstance {
                skeleton, attaches, ..
            } = &mut self.char_instances[idx];
            skeleton.p = xf.pos;
            skeleton.s = xf.right;
            skeleton.h = xf.up;
            skeleton.f = xf.forward;
            solve_kfa_limbs(skeleton);
            attaches
                .iter()
                .map(|a| {
                    let limb = &skeleton.limbs[a.bone];
                    let (s, h, f, p) =
                        compose_attachment(limb.s, limb.h, limb.f, limb.p, &a.local_offset);
                    (
                        a.inst,
                        DynSpriteTransform {
                            pos: p,
                            right: s,
                            up: h,
                            forward: f,
                        },
                    )
                })
                .collect()
        };
        // Phase 2: apply (clip frames untouched — clocks didn't tick).
        for (inst, t) in updates {
            self.set_sprite_instance_transform(inst, t);
        }
    }

    /// Remove a character, dropping all its attachment instances **and**
    /// freeing the sprite models + voxel clips it registered. Returns
    /// `false` if `id` is stale.
    pub fn remove_character(&mut self, id: CharacterId) -> bool {
        let Some(idx) = self.char_map.index(id) else {
            return false;
        };
        let insts: Vec<SpriteInstanceId> = self.char_instances[idx]
            .attaches
            .iter()
            .map(|a| a.inst)
            .collect();
        for inst in insts {
            self.remove_sprite_instance(inst);
        }
        self.char_instances[idx].attaches.clear();
        // Free the models + clips this character registered (else they leak
        // until a `set_sprites` — costly for an editor hot-swapping all
        // session). `mem::take` so the per-id frees can borrow `self`.
        let models = std::mem::take(&mut self.char_instances[idx].models);
        let clips = std::mem::take(&mut self.char_instances[idx].clips);
        for model in models {
            self.remove_sprite_model(model);
        }
        for clip in clips {
            self.remove_voxel_clip(clip);
        }
        self.char_map.remove(id)
    }

    /// Register animated KFA sprites (one or more bone hierarchies).
    /// The GPU backend uploads each limb's kv6 as an instanced model
    /// **once** (appended to the sprite registry) and seeds the limb
    /// instances at their current pose; the CPU backend caches the
    /// posed limbs for drawing. Call once at setup, after
    /// [`set_sprites`](Self::set_sprites), then drive motion per frame
    /// with [`update_kfa_poses`](Self::update_kfa_poses).
    ///
    /// Limbs are posed from the sprites' current
    /// [`kfaval`](roxlap_formats::kfa::KfaSprite::kfaval) (advance
    /// [`animsprite`](roxlap_formats::kfa::KfaSprite::animsprite) first
    /// if using a baked curve), so `kfas` is taken `&mut`.
    pub fn set_kfa_sprites(&mut self, kfas: &mut [KfaSprite]) {
        match &mut self.inner {
            BackendImpl::Cpu(c) => c.set_kfa_sprites(kfas),
            BackendImpl::Gpu(g) => g.set_kfa_sprites(kfas),
        }
    }

    /// Re-pose the registered KFA sprites from their current
    /// `kfaval[]`. Call each frame after advancing the animation
    /// (`kfa.animsprite(dt_ms)` or poking `kfaval[]`). The GPU backend
    /// takes the cheap transform-only update (no model-volume
    /// re-upload); the CPU backend re-solves limb transforms for the
    /// next [`render`](Self::render). Must follow a
    /// [`set_kfa_sprites`](Self::set_kfa_sprites) with the same sprites.
    pub fn update_kfa_poses(&mut self, kfas: &mut [KfaSprite]) {
        match &mut self.inner {
            BackendImpl::Cpu(c) => c.update_kfa_poses(kfas),
            BackendImpl::Gpu(g) => g.update_kfa_poses(kfas),
        }
    }

    /// Carve the next z-layer off the [`SpriteSet::carve_model`] and
    /// re-upload (the demo's `G` hotkey + GPU.12 copy-on-modify). GPU
    /// only; a no-op on the CPU backend. Returns the voxels removed.
    pub fn carve_active_sprite(&mut self) -> u32 {
        match &mut self.inner {
            BackendImpl::Cpu(_) => 0,
            BackendImpl::Gpu(g) => g.carve_active_sprite(),
        }
    }

    /// Request that the next [`render`](Self::render) capture its
    /// framebuffer for [`take_capture`](Self::take_capture). CPU only
    /// (the GPU swapchain isn't read back) — a no-op on GPU.
    pub fn request_capture(&mut self) {
        if let BackendImpl::Cpu(c) = &mut self.inner {
            c.request_capture();
        }
    }

    /// Take the most recently captured frame as packed `0x00RRGGBB`
    /// pixels + dimensions, or `None` if no capture is ready / GPU.
    pub fn take_capture(&mut self) -> Option<(Vec<u32>, u32, u32)> {
        match &mut self.inner {
            BackendImpl::Cpu(c) => c.take_capture(),
            BackendImpl::Gpu(_) => None,
        }
    }

    /// Screen→world picking input: the world-space hit distance `t` at
    /// window pixel `(x, y)` from the **last rendered frame**, or `None`
    /// for out-of-bounds pixels and sky / no-hit. The host reconstructs
    /// the world hit point as `cam.pos + t * normalize(ray_dir)`, where
    /// `ray_dir` is the same per-pixel ray the frame was rendered with
    /// (see the backend's projection).
    ///
    /// `t` is the distance to the nearest **scene-grid** surface
    /// (terrain + grids); sprites do not occlude it (the sprite pass
    /// reads depth read-only), so a cursor sprite under the pointer is
    /// transparent to the pick.
    ///
    /// Cost: the CPU backend reads its in-memory z-buffer (free); the
    /// GPU backend stages the depth buffer and blocks on a device poll
    /// (cheap at click time — do not call every frame). The GPU path
    /// only has depth when the last frame drew sprites (`write_depth`).
    #[must_use]
    pub fn pick_depth(&self, x: u32, y: u32) -> Option<f32> {
        match &self.inner {
            BackendImpl::Cpu(c) => c.pick_depth(x, y),
            BackendImpl::Gpu(g) => g.pick_depth(x, y),
        }
    }

    /// World-space view-ray direction (un-normalised) for window pixel
    /// `(x, y)`, under the projection the **last frame** rendered with.
    /// The backends differ (CPU `setcamera` vs GPU vertical-FOV
    /// pinhole), so this hides which one is active. `None` before the
    /// first frame. Intersect it with a plane for tile picking, or feed
    /// it to [`Self::pick`] for a voxel.
    #[must_use]
    pub fn pixel_ray(&self, camera: &Camera, x: f64, y: f64) -> Option<[f64; 3]> {
        match &self.inner {
            BackendImpl::Cpu(c) => c.pixel_ray(camera, x, y),
            BackendImpl::Gpu(g) => g.pixel_ray(camera, x, y),
        }
    }

    /// Canonical screen→world unproject: the full view [`Ray`]
    /// (`camera.pos` origin + unit direction) for window pixel
    /// `(x, y)`, under whichever projection the last frame used. The
    /// one entry point both backends honour — hosts never reconstruct
    /// the projection. `None` before the first frame or for a
    /// degenerate ray.
    ///
    /// Compose with [`roxlap_scene::Scene::raycast`] for depth-free
    /// picking that's identical on CPU and GPU:
    /// `renderer.view_ray(cam, x, y).and_then(|r| scene.raycast(r.origin, r.dir, max))`.
    #[must_use]
    pub fn view_ray(&self, camera: &Camera, x: f64, y: f64) -> Option<Ray> {
        let d = self.pixel_ray(camera, x, y)?;
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        if len < 1e-12 {
            return None;
        }
        Some(Ray {
            origin: glam::DVec3::from_array([camera.pos[0], camera.pos[1], camera.pos[2]]),
            dir: glam::DVec3::new(d[0] / len, d[1] / len, d[2] / len),
        })
    }

    /// One-call screen→world voxel pick: unproject pixel `(x, y)` with
    /// the active backend's projection, read the last frame's depth
    /// there, reconstruct the world hit, and resolve it to the owning
    /// grid + grid-local voxel via [`Scene::resolve_voxel`]. `None` on
    /// sky / no-hit, or when no grid claims the surface.
    ///
    /// `scene` and `camera` must be the ones the last frame rendered;
    /// the projection (size + FOV / `hx,hy,hz`) is taken from that
    /// frame. Cheap on CPU (in-memory z-buffer); on GPU it stages the
    /// depth buffer (a click-time device poll — not per frame).
    #[must_use]
    pub fn pick(&self, scene: &Scene, camera: &Camera, x: u32, y: u32) -> Option<PickHit> {
        let dir = self.pixel_ray(camera, f64::from(x), f64::from(y))?;
        let t = f64::from(self.pick_depth(x, y)?);
        let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        if len < 1e-9 {
            return None;
        }
        let s = t / len; // world = cam.pos + t · (dir / |dir|)
        let world = glam::DVec3::new(
            camera.pos[0] + dir[0] * s,
            camera.pos[1] + dir[1] * s,
            camera.pos[2] + dir[2] * s,
        );
        let (grid, voxel) = scene.resolve_voxel(world, glam::DVec3::from_array(dir))?;
        #[allow(clippy::cast_possible_truncation)]
        let world_f32 = [world.x as f32, world.y as f32, world.z as f32];
        Some(PickHit {
            world: world_f32,
            grid,
            voxel,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The handle map must survive the backends' swap-remove indexing:
    /// drive a model `DynInstanceMap` against a `Vec` "backend" that
    /// swap-removes, and check every live handle keeps resolving to its
    /// own payload through a sequence of adds + removes.
    #[test]
    fn dyn_instance_map_survives_swap_removes() {
        let mut map = DynInstanceMap::default();
        // The "backend": payload per dynamic index; swap_remove mirrors
        // both backends' remove_dyn_instance.
        let mut backend: Vec<u32> = Vec::new();
        // Our bookkeeping: handle -> the payload we expect it to address.
        let mut expect: Vec<(SpriteInstanceId, u32)> = Vec::new();

        let add = |map: &mut DynInstanceMap,
                   backend: &mut Vec<u32>,
                   expect: &mut Vec<(SpriteInstanceId, u32)>,
                   payload: u32| {
            let dyn_index = backend.len() as u32;
            backend.push(payload);
            let id = map.alloc(dyn_index);
            expect.push((id, payload));
        };

        for p in 0..6 {
            add(&mut map, &mut backend, &mut expect, p);
        }

        // Remove a middle handle (payload 2) and a later one (payload 4),
        // plus the current last — covering swap and no-swap paths.
        for victim_payload in [2u32, 4, 5] {
            let pos = expect
                .iter()
                .position(|&(_, p)| p == victim_payload)
                .unwrap();
            let (id, _) = expect.remove(pos);
            let dyn_index = map.dyn_index(id).expect("live handle resolves");
            // Backend swap-remove + report moved index (old last), exactly
            // like remove_dyn_instance on both backends.
            let last = backend.len() - 1;
            backend.swap_remove(dyn_index as usize);
            let moved = (dyn_index as usize != last).then_some(last as u32);
            map.remove(id, dyn_index, moved);
            // The removed handle is now stale.
            assert!(map.dyn_index(id).is_none(), "removed handle is stale");
        }

        // Every surviving handle still resolves to its own payload.
        for &(id, payload) in &expect {
            let idx = map.dyn_index(id).expect("survivor resolves");
            assert_eq!(
                backend[idx as usize], payload,
                "handle addresses its payload"
            );
        }
        assert_eq!(map.order.len(), backend.len());
        assert_eq!(backend.len(), expect.len());
    }

    /// The model slotmap mints stable ids, resolves only live handles,
    /// and never reuses a slot — so a removed model's id stays dead and
    /// every other id survives the remove.
    #[test]
    fn dyn_model_map_lifecycle() {
        let mut map = DynModelMap::default();
        // `set_sprites(3 models)` seeds ids 0..3, all live.
        map.reset(3);
        let ids: Vec<SpriteModelId> = (0..3).map(|s| SpriteModelId { slot: s, gen: 0 }).collect();
        for (i, &id) in ids.iter().enumerate() {
            assert_eq!(map.model_index(id), Some(i));
        }

        // Incrementally add a fourth model.
        let extra = map.alloc(3);
        assert_eq!(extra, SpriteModelId { slot: 3, gen: 0 });
        assert_eq!(map.model_index(extra), Some(3));

        // Remove model 1: its handle goes stale, the rest stay valid.
        assert!(map.remove(ids[1]));
        assert_eq!(map.model_index(ids[1]), None);
        assert_eq!(map.model_index(ids[0]), Some(0));
        assert_eq!(map.model_index(ids[2]), Some(2));
        assert_eq!(map.model_index(extra), Some(3));

        // Double remove / stale removal is a no-op returning false.
        assert!(!map.remove(ids[1]));

        // A bogus / out-of-range handle resolves to nothing, no panic.
        let bogus = SpriteModelId { slot: 999, gen: 0 };
        assert_eq!(map.model_index(bogus), None);
        assert!(!map.remove(bogus));

        // A handle with a mismatched generation never resolves (guards a
        // future compacting registry).
        let wrong_gen = SpriteModelId { slot: 0, gen: 7 };
        assert_eq!(map.model_index(wrong_gen), None);
    }

    /// The voxel-clip slotmap (VCL.4) mints stable ids, resolves only live
    /// handles, tombstones in place, and `reset` clears it — mirroring the
    /// model slotmap, since clips register append-only too.
    #[test]
    fn dyn_clip_map_lifecycle() {
        let mut map = DynClipMap::default();
        // Two clips registered incrementally (indices 0, 1).
        let c0 = map.alloc(0);
        let c1 = map.alloc(1);
        assert_eq!(c0, VoxelClipId { slot: 0, gen: 0 });
        assert_eq!(map.clip_index(c0), Some(0));
        assert_eq!(map.clip_index(c1), Some(1));

        // Remove clip 0: stale handle, clip 1 stays valid; slot not reused.
        assert!(map.remove(c0));
        assert_eq!(map.clip_index(c0), None);
        assert_eq!(map.clip_index(c1), Some(1));
        // Double / stale / out-of-range removes are false, no panic.
        assert!(!map.remove(c0));
        assert!(!map.remove(VoxelClipId { slot: 99, gen: 0 }));
        // Mismatched generation never resolves.
        assert_eq!(map.clip_index(VoxelClipId { slot: 1, gen: 5 }), None);

        // `set_sprites` resets the clip layer → ids restart at slot 0, but
        // the epoch bumps so old handles don't alias the new clips.
        map.reset();
        assert_eq!(map.clip_index(c1), None, "reset invalidates old handles");
        let again = map.alloc(0); // re-takes slot 0 under the new epoch
        assert_eq!(again, VoxelClipId { slot: 0, gen: 1 });
        assert_eq!(map.clip_index(again), Some(0));
        // The footgun fix: c0 (slot 0, old epoch) must NOT resolve to the new
        // clip now occupying slot 0.
        assert_eq!(
            map.clip_index(c0),
            None,
            "a pre-reset handle must not alias a new clip on the same slot"
        );
    }

    /// The character slotmap (VCL.6) mints stable ids, resolves only live
    /// handles, tombstones in place, and `reset` clears it.
    #[test]
    fn char_map_lifecycle() {
        let mut map = CharMap::default();
        let a = map.alloc(0);
        let b = map.alloc(1);
        assert_eq!(a, CharacterId { slot: 0, gen: 0 });
        assert_eq!(map.index(a), Some(0));
        assert_eq!(map.index(b), Some(1));

        assert!(map.remove(a));
        assert_eq!(map.index(a), None);
        assert_eq!(map.index(b), Some(1));
        assert!(!map.remove(a)); // double remove is a no-op
        assert!(!map.remove(CharacterId { slot: 9, gen: 0 }));
        assert_eq!(map.index(CharacterId { slot: 1, gen: 7 }), None);

        map.reset();
        assert_eq!(map.index(b), None);
        assert_eq!(map.alloc(0), CharacterId { slot: 0, gen: 1 });
        assert_eq!(map.index(a), None, "pre-reset handle must not alias slot 0");
    }

    /// The streaming-clip slotmap (#3) mints stable ids, resolves only live
    /// handles, tombstones in place, and `reset` clears it.
    #[test]
    fn streaming_clip_map_lifecycle() {
        let mut map = StreamingClipMap::default();
        let a = map.alloc(0);
        let b = map.alloc(1);
        assert_eq!(a, StreamingClipId { slot: 0, gen: 0 });
        assert_eq!(map.index(a), Some(0));
        assert_eq!(map.index(b), Some(1));

        assert!(map.remove(a));
        assert_eq!(map.index(a), None);
        assert_eq!(map.index(b), Some(1));
        assert!(!map.remove(a)); // double remove is a no-op
        assert!(!map.remove(StreamingClipId { slot: 9, gen: 0 }));
        assert_eq!(map.index(StreamingClipId { slot: 1, gen: 7 }), None);

        map.reset();
        assert_eq!(map.index(b), None);
        assert_eq!(map.alloc(0), StreamingClipId { slot: 0, gen: 1 });
        assert_eq!(map.index(a), None, "pre-reset handle must not alias slot 0");
    }

    /// The shared clip-playback clock (#6 / VCL.6): `tick` accumulates time
    /// at its Q8 speed, resolves the frame, honours `start_phase`, and reads
    /// a rewound (negative) clock as frame 0.
    #[test]
    fn clip_clock_tick_advances_and_resolves_frames() {
        // 3 frames, 100 ms each → total 300 ms, looping.
        let mut c = ClipClock {
            durations: vec![100, 100, 100],
            loop_mode: LoopMode::Loop,
            speed_q8: 256, // 1×
            clock_ms: 0.0,
        };
        assert_eq!(c.tick(0.0), 0); // t=0 → frame 0
        assert_eq!(c.tick(0.10), 1); // t=100 → frame 1 (100 is not < 100)
        assert_eq!(c.clock_ms as u32, 100);
        assert_eq!(c.tick(0.15), 2); // t=250 → frame 2
        assert_eq!(c.tick(0.10), 0); // t=350 → 350%300=50 → frame 0
                                     // 0.5× speed advances half as fast.
        let mut slow = ClipClock {
            durations: vec![100, 100],
            loop_mode: LoopMode::Once,
            speed_q8: 128, // 0.5×
            clock_ms: 0.0,
        };
        assert_eq!(slow.tick(0.20), 1); // 200ms wall → 100ms clock → frame 1
        assert!((slow.clock_ms - 100.0).abs() < 1e-6);
        // start_phase seeds the clock; negative clock reads as frame 0.
        let mut phased = ClipClock {
            durations: vec![50, 50, 50],
            loop_mode: LoopMode::Loop,
            speed_q8: -256, // rewind
            clock_ms: 50.0, // start mid frame 1
        };
        assert_eq!(phased.tick(0.10), 0); // 50 - 100 = -50 → max(0)=0 → frame 0
        assert!(phased.clock_ms < 0.0); // kept signed
    }

    #[test]
    fn dyn_sprite_transform_default_is_identity_and_applies() {
        let xf = DynSpriteTransform::default();
        assert_eq!(xf.pos, [0.0, 0.0, 0.0]);
        assert_eq!(xf.right, [1.0, 0.0, 0.0]);
        assert_eq!(xf.up, [0.0, 1.0, 0.0]);
        assert_eq!(xf.forward, [0.0, 0.0, 1.0]);

        let mut s = Sprite::axis_aligned(
            roxlap_formats::kv6::Kv6::solid_cube(2, 0x80_FF_FF_FF),
            [9.0, 9.0, 9.0],
        );
        let posed = DynSpriteTransform {
            pos: [1.0, 2.0, 3.0],
            right: [0.0, 0.0, 1.0],
            up: [0.0, 1.0, 0.0],
            forward: [1.0, 0.0, 0.0],
        };
        posed.apply_to(&mut s);
        assert_eq!(s.p, [1.0, 2.0, 3.0]);
        assert_eq!(s.s, [0.0, 0.0, 1.0]);
        assert_eq!(s.h, [0.0, 1.0, 0.0]);
        assert_eq!(s.f, [1.0, 0.0, 0.0]);
    }

    #[test]
    fn options_default_is_cpu_intent() {
        let o = RenderOptions::default();
        assert!(!o.want_gpu);
        assert_eq!(o.clear_sky & 0xFF00_0000, 0, "clear_sky is 0x00RRGGBB");
    }

    /// A camera at the origin looking down +Y (voxlap z-down world): right
    /// = +X, down = +Z, forward = +Y. Handedness `right × down == forward`.
    fn cam_looking_y() -> Camera {
        Camera {
            pos: [0.0, 0.0, 0.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 0.0, 1.0],
            forward: [0.0, 1.0, 0.0],
        }
    }

    #[test]
    fn world_quad_corner_layout() {
        // Top-left at (-5, 10, -5); u = +X (width), v = +Z (down). A
        // 10×10 quad facing the camera (its +Y normal points back at us).
        let sprite = ImageSprite {
            image: ImageId(0),
            origin: [-5.0, 10.0, -5.0],
            facing: ImageFacing::World {
                u: [1.0, 0.0, 0.0],
                v: [0.0, 0.0, 1.0],
            },
            size: [10.0, 10.0],
            tint: 0xFFFF_FFFF,
            alpha_cutoff: 0.0,
            depth_test: true,
            double_sided: true,
        };
        let q = resolve_quad(&sprite, &cam_looking_y()).expect("front-facing");
        assert_eq!(q.corners[0], [-5.0, 10.0, -5.0], "TL = origin");
        assert_eq!(q.corners[1], [5.0, 10.0, -5.0], "TR = origin + u·size");
        assert_eq!(q.corners[2], [-5.0, 10.0, 5.0], "BL = origin + v·size");
        assert_eq!(q.corners[3], [5.0, 10.0, 5.0], "BR = origin + u + v");
    }

    #[test]
    fn world_quad_backface_culls_when_single_sided() {
        // Same plane but spanned so its normal (u × v) points *away* from
        // the camera: swap u/v so the winding flips.
        let sprite = ImageSprite {
            image: ImageId(0),
            origin: [-5.0, 10.0, -5.0],
            facing: ImageFacing::World {
                u: [0.0, 0.0, 1.0], // v-ish
                v: [1.0, 0.0, 0.0], // u-ish → normal flips to -Y... toward camera?
            },
            size: [10.0, 10.0],
            tint: 0xFFFF_FFFF,
            alpha_cutoff: 0.0,
            depth_test: true,
            double_sided: false,
        };
        // With double_sided=false one of the two windings must cull; the
        // opposite winding must draw. Exactly one of the two resolves.
        let a = resolve_quad(&sprite, &cam_looking_y()).is_some();
        let mut flipped = sprite;
        flipped.facing = ImageFacing::World {
            u: [1.0, 0.0, 0.0],
            v: [0.0, 0.0, 1.0],
        };
        let b = resolve_quad(&flipped, &cam_looking_y()).is_some();
        assert!(a ^ b, "exactly one winding is front-facing");
    }

    #[test]
    fn double_sided_never_culls() {
        let mut sprite = ImageSprite {
            image: ImageId(0),
            origin: [-5.0, 10.0, -5.0],
            facing: ImageFacing::World {
                u: [0.0, 0.0, 1.0],
                v: [1.0, 0.0, 0.0],
            },
            size: [10.0, 10.0],
            tint: 0xFFFF_FFFF,
            alpha_cutoff: 0.0,
            depth_test: true,
            double_sided: true,
        };
        assert!(resolve_quad(&sprite, &cam_looking_y()).is_some());
        sprite.facing = ImageFacing::World {
            u: [1.0, 0.0, 0.0],
            v: [0.0, 0.0, 1.0],
        };
        assert!(resolve_quad(&sprite, &cam_looking_y()).is_some());
    }

    #[test]
    fn ray_quad_uv_center_and_corners() {
        // 10×10 quad on the y=10 plane: TL(-5,10,-5) u=+X v=+Z. Camera at
        // origin looking +Y. A ray straight at the quad centre → uv (.5,.5).
        let corners = [
            [-5.0, 10.0, -5.0], // TL
            [5.0, 10.0, -5.0],  // TR
            [-5.0, 10.0, 5.0],  // BL
            [5.0, 10.0, 5.0],   // BR
        ];
        let (uv, t) = ray_quad_uv([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], &corners).expect("center hit");
        assert!(
            (uv[0] - 0.5).abs() < 1e-5 && (uv[1] - 0.5).abs() < 1e-5,
            "centre → (.5,.5)"
        );
        assert!((t - 10.0).abs() < 1e-4, "t = plane distance");
        // Ray toward the TL corner texel region (−x, +y, −z) → uv near (0,0).
        let (uv_tl, _) = ray_quad_uv([0.0, 0.0, 0.0], [-4.0, 10.0, -4.0], &corners).unwrap();
        assert!(uv_tl[0] < 0.2 && uv_tl[1] < 0.2, "toward TL → small uv");
    }

    #[test]
    fn ray_quad_uv_misses_outside_and_behind() {
        let corners = [
            [-5.0, 10.0, -5.0],
            [5.0, 10.0, -5.0],
            [-5.0, 10.0, 5.0],
            [5.0, 10.0, 5.0],
        ];
        // Ray pointing away (−Y) never reaches the +Y plane in front.
        assert!(ray_quad_uv([0.0, 0.0, 0.0], [0.0, -1.0, 0.0], &corners).is_none());
        // Ray parallel to the quad plane (in +X) → no intersection.
        assert!(ray_quad_uv([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], &corners).is_none());
        // Ray hitting the plane far outside the quad → outside uv.
        assert!(ray_quad_uv([100.0, 0.0, 0.0], [0.0, 1.0, 0.0], &corners).is_none());
    }

    #[test]
    fn billboard_axes_orthogonal_and_top_toward_up() {
        // World up = -Z (z-down world). The billboard's v (top→bottom)
        // must point away from `up`, and u/v must be ⟂ the view direction.
        let up = [0.0, 0.0, -1.0];
        let sprite = ImageSprite {
            image: ImageId(0),
            origin: [0.0, 50.0, 0.0],
            facing: ImageFacing::Billboard { up },
            size: [4.0, 4.0],
            tint: 0xFFFF_FFFF,
            alpha_cutoff: 0.0,
            depth_test: false,
            double_sided: false, // billboards must NEVER cull
        };
        let q = resolve_quad(&sprite, &cam_looking_y()).expect("billboard always faces camera");
        let u = v_sub(q.corners[1], q.corners[0]); // TR - TL = u·size
        let v = v_sub(q.corners[2], q.corners[0]); // BL - TL = v·size
        let fwd = [0.0, 1.0, 0.0];
        assert!(v_dot(u, fwd).abs() < 1e-5, "u ⟂ view");
        assert!(v_dot(v, fwd).abs() < 1e-5, "v ⟂ view");
        assert!(v_dot(u, v).abs() < 1e-5, "u ⟂ v");
        assert!(
            v_dot(v, up) < 0.0,
            "rows grow away from `up` (top edge toward up)"
        );
    }
}
