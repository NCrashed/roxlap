# roxlap — GIF billboard sprites: camera-facing animated cutouts (Substage BB)

Start-of-stage brief and locked decisions for **GIF billboard sprites** —
Doom/Build-style flat, camera-facing animated cutouts, authored as one
`.gif` per animation, integrated as first-class citizens of the 3D world:
they **receive** dynamic shadows + lighting and **cast** their own
shadows, play back by time, and switch animation by game state (including
view-angle "rotations"). Companion to
[PORTING-VOXEL-CLIP.md](PORTING-VOXEL-CLIP.md) (the `.rvc` flipbook this
builds on), [PORTING-SPRITE-API.md](PORTING-SPRITE-API.md) (the dynamic
instance + per-instance transform API), [PORTING-DYNLIGHT.md](PORTING-DYNLIGHT.md)
(sprite lighting DL.7 + sprite shadows cast/receive XS.4.2/XS.4.3), and
[PORTING-TRANSPARENCY.md](PORTING-TRANSPARENCY.md) (per-voxel materials /
alpha / additive).

This is a **start-of-stage brief**. A fresh-context session should read it
top to bottom before touching code. The stage tag is **BB**. It targets
the next minor (**0.19.0** unless AO claims it first — maintainer sets the
number at release). The change is **purely additive** — no existing public
signature changes; the opaque/no-billboard world stays byte-identical.

## Why

The engine user wants to drop animated 2D sprites — monsters, projectiles,
fire, pickups — into the voxel world the way Doom/Build did: a flat image
that always turns to face the player, plays an animation over time, and
swaps to a different animation depending on game state (idle → walk →
attack → die) and view angle (the 8 "rotation" frames). Authoring is **one
`.gif` per animation**. And — unlike Doom — these must be real lighting
citizens: a torch must light them, the sun must shadow them, and they must
cast their own shadow onto the terrain.

roxlap has, as of XS.4.3 + DL.7 + TV + VCL, *already built every hard part
of this — for voxels*:

- **Sprites cast + receive dynamic shadows** on both backends, per-instance
  configurable (XS.4.2 receive, XS.4.3 cast; flags `SPRITE_FLAG_NO_SHADOW_CAST`/
  `_RECEIVE`, `roxlap-formats/src/sprite.rs:41-49`).
- **Sprites are lit** by sun + point lights (DL.7 GPU / CPU-sprites), retro
  cel/flat-per-voxel look.
- **`.rvc` voxel-clips** are a working "GIF for voxels": a flipbook decoded
  once, played by a per-instance clock (`advance_voxel_clips`,
  `roxlap-render/src/lib.rs:2230`), with per-frame durations, loop modes,
  per-voxel materials (`add_voxel_clip_with_materials`, `:1852`).
- **TV materials** give per-voxel/per-instance alpha + `Additive`/`AlphaBlend`/
  `Volumetric` (fire/spell glow, smoke).

So the cheapest, most consistent path is **not a new render primitive** but
to express a GIF billboard as a *flat voxel-clip* and lean on all of the
above. What is genuinely missing is small and well-scoped (see Gaps).

## Key enabling facts

- **A GIF frame is exactly a 1-voxel-thick clip frame.** A `VoxelFrame`
  (`roxlap-formats/src/voxel_clip.rs:102`) is dense-column occupancy +
  colours. A GIF frame of `W×H` opaque/transparent pixels maps to a slab of
  dims `[W, 1, H]`: opaque pixel → one voxel `0x80RRGGBB` at `(col, 0, row)`,
  transparent pixel → air (cutout). No interior voxels exist in a 1-thick
  slab, so `Kv6::from_fn` (`kv6.rs:163`) culls nothing — but we build the
  `VoxelFrame` columns directly (skip the kv6 round-trip) for speed.

- **Shadows / lighting / alpha / playback come for free.** Because the
  billboard *is* a voxel-clip instance, it inherits XS.4 cast+receive
  shadows, DL.7 lighting, TV materials, and the VCL clock — zero new
  rendering, shadow, or lighting code. This is the entire reason BB is a
  small stage.

- **Billboarding is a per-frame transform, already the cheap axis.**
  `set_sprite_instance_transform[s]` (`lib.rs:1745/1762`) is batched +
  coalesced into one device write per frame (`transforms_dirty`). Orienting
  every billboard to the camera each frame is just one batched transform
  flush — no volume re-upload, the same cost as the existing posed-sprite
  demo.

- **Retargeting an instance to another flipbook is already possible at the
  GPU layer.** `SpriteRegistryResident::set_instance_model(cull_idx,
  chain_id)` (the VCL.2 per-frame "select model" primitive) can point an
  instance at *any* registered chain — including another clip's frame range.
  So "switch which GIF this instance plays" (`set_clip_instance_clip`) is a
  thin facade wrapper over the same mechanism `set_clip_instance_frame`
  already uses, not new machinery.

- **`from_frames_auto` already picks keyframes + diffs.** The clip encoder
  (`voxel_clip.rs:903/from_frames`, the `_auto` variant) takes a frame list
  + durations and emits the I/P-coded `.rvc`. The GIF importer just feeds it
  decoded slabs — no new codec.

- **Default-off is byte-identical.** A scene with no billboard instances
  renders exactly as today; a billboard instance is an ordinary clip
  instance whose transform happens to be camera-derived. The headline
  regression gate (`lights==None` / opaque-unchanged) holds trivially.

## Gaps (what does NOT exist yet)

1. **Image import.** Workspace has only `png` + `flate2` (`roxlap-host/Cargo.toml`).
   No `gif`/`image` crate. Need a GIF (and trivially PNG-sequence) → clip
   importer.
2. **Billboarding.** `roxlap-scene/src/billboard.rs` is the *LOD impostor*
   cache for far grids — unrelated. There is **no** per-sprite camera-facing
   anywhere; orientation is fully manual today.
3. **Clip switching on a live instance.** `set_clip_instance_frame`
   (`lib.rs:1931`) moves *within* a clip; there is **no**
   `set_clip_instance_clip` to swap the whole flipbook (needed for
   directional rotations + state changes). Today's only path is
   remove + respawn.
4. **Per-instance shadow flags.** Shadow cast/receive flags live on the
   `Sprite` template at registration (`sprite.rs:83` `flags`), with no
   per-instance facade setter.

## Locked decisions

Taken with the engine author 2026-06-30:

1. **Voxel-slab billboards, not textured cards.** Each GIF frame is
   voxelized into a flat 1-thick `VoxelFrame`; a billboard is a voxel-clip
   instance. Reuses XS.4 shadows, DL.7 lighting, TV materials, VCL playback
   verbatim — no new render/shadow/lighting primitive. (Rejected: true 2D
   textured quads — crisper 2D + less memory, but a from-scratch render
   primitive in the DDA compute shader **plus** a from-scratch shadow-ray ∩
   quad ∩ alpha path that reuses none of XS.4; ~3–4× the work. Noted as a
   possible future high-fidelity path, §Risks R4.)

2. **Cylindrical billboarding by default; spherical per-instance.** Yaw-only,
   `up = world up`. Keeps the slab vertical so an overhead sun casts a
   stable, sane shadow that does not spin as the camera orbits (the
   spherical-billboard pathology). `BillboardMode::{None, Cylindrical,
   Spherical}` per instance; `Spherical` is opt-in for face-camera-always
   effects (coins, particles) where the rotating shadow is acceptable or the
   instance is a non-caster.

3. **Two API tiers, both shipped.** Low-level primitives
   (`set_clip_instance_clip`, the billboard mode + `face_billboards_to`,
   per-instance shadow flags) **and** a high-level `BillboardActor` that
   owns directional (8-way) selection + a named-state machine. Game code may
   use either. The actor is built *entirely* on the primitives — no private
   backend reach.

4. **GIF = 1-bit cutout; semi-alpha + glow via materials.** GIF carries only
   a palette + a single transparent index → pure cutout (the common case,
   no compositing). For glow effects (fire, spells) the whole clip is tagged
   with an `Additive` material via the existing `add_voxel_clip_with_materials`
   colour→material map (`&[(u32,u8)]`). True per-pixel semi-alpha is a
   PNG/APNG-sequence importer concern, deferred (§Risks R5).

5. **Importer is feature-gated in `roxlap-formats`.** New optional feature
   `gif` adds the `gif` crate + a `gif_import` module producing a
   `VoxelClip`/`DecodedClip` (a format type belongs with the format). Core
   stays dependency-light when the feature is off. (PNG already lives in
   `roxlap-host`; a PNG-sequence importer can mirror this later, feature
   `png-seq`.)

6. **Lighting normal for billboards is configurable.** A camera-facing slab's
   front-face normal points at the camera, so naive N·L changes as you orbit
   (R1). A per-instance `BillboardLighting::{FaceNormal, WorldUp,
   AmbientOnly}` chooses: the real face normal (default; matches DL.7 retro
   look), a fixed world-up normal (stable shading), or ambient/unlit (flat,
   most Doom-faithful). Implemented at the shade site with the data already
   present — no new buffers.

7. **No new crate.** Importer in `roxlap-formats` (feature `gif`); the
   `set_clip_instance_clip` primitive + billboard orientation + per-instance
   shadow flags on the `roxlap-render` facade, mirrored by hand into
   `cpu.rs` + `gpu.rs` (duck-typed, no backend trait — the standing
   pattern); `BillboardActor` lives in `roxlap-render` next to the clip
   player.

## The data model

```rust
// roxlap-formats/src/gif_import.rs   (new, feature = "gif")

pub struct GifImportOpts {
    pub voxel_world_size: f32,     // world size of one pixel-voxel (default 1.0)
    pub thickness: u32,            // slab depth in voxels (default 1; >1 for shadow body)
    pub pivot: Pivot,              // BottomCenter (feet on ground) | Center | Custom([f32;3])
    pub loop_mode: LoopMode,       // produced clip's playback (default Loop)
    pub default_frame_ms: u32,     // used when a GIF frame delay is 0 (default 100)
    pub keyframe_gap: u32,         // forwarded to from_frames_auto (default 8)
    pub max_dims: Option<[u32;3]>, // reject oversize GIFs explicitly (no silent downscale)
}
impl Default for GifImportOpts { /* 1px voxels, 1 thick, bottom-center, Loop, 100ms */ }
```

> **BB.0 deviation from locked decision #4 (material_map dropped).** The
> importer is geometry-only: a `VoxelClip` carries colour, not materials. A
> whole-clip glow (fire/spells) is a *per-instance* concern — define an
> additive `Material` and call `set_sprite_instance_material` on the clip
> instance — so `material_map` was removed from `GifImportOpts` rather than
> threaded through a format type that can't hold it. Per-voxel material maps
> remain available at registration via the existing
> `add_voxel_clip_with_materials`.

```rust

/// Decode an animated GIF into a VoxelClip of flat camera-facing slabs.
/// Per-frame GIF delays (centiseconds) → clip durations (ms). Loops by default.
/// Disposal methods honored (restore-to-bg / restore-previous) so partial
/// frames compose correctly before voxelization.
pub fn voxel_clip_from_gif(bytes: &[u8], opts: &GifImportOpts)
    -> Result<VoxelClip, GifImportError>;
```

Slab axis convention (voxlap `s/h/f` = local +x/+y/+z columns): image
**column → local +x (right)**, image **row → local +z (up, top row =
high z)**, **local +y = depth (1 voxel)**, normal toward the viewer. So
`Cylindrical` orientation sets `forward = +y = horiz(camera - pos)`,
`up = +z = world_up`, `right = +x = up × forward`.

```rust
// roxlap-render/src/lib.rs  (facade additions)

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BillboardMode { #[default] None, Cylindrical, Spherical }

#[derive(Clone, Copy, Debug, Default)]
pub enum BillboardLighting { #[default] FaceNormal, WorldUp, AmbientOnly }

impl SceneRenderer {
    // --- BB.1: switch which flipbook a live instance plays ---
    /// Retarget an existing clip instance onto a different clip (resets to
    /// frame 0, keeps transform + clock policy). false on stale handle.
    pub fn set_clip_instance_clip(&mut self, id: SpriteInstanceId, clip: VoxelClipId) -> bool;

    // --- BB.2: billboard orientation ---
    /// Spawn a clip instance flagged to auto-orient toward the camera.
    pub fn add_billboard_instance(
        &mut self, clip: VoxelClipId, pos: [f32; 3], mode: BillboardMode,
    ) -> SpriteInstanceId;
    pub fn set_billboard_mode(&mut self, id: SpriteInstanceId, mode: BillboardMode);
    pub fn set_billboard_lighting(&mut self, id: SpriteInstanceId, l: BillboardLighting);
    /// Re-orient every billboard-flagged instance toward `camera` (one
    /// batched transform flush). Call once per frame before render, like
    /// advance_voxel_clips(dt). Position comes from the instance's stored pos.
    pub fn face_billboards_to(&mut self, camera: &Camera);
    /// Move a billboard (keeps the auto-orientation).
    pub fn set_billboard_position(&mut self, id: SpriteInstanceId, pos: [f32; 3]);

    // --- BB.3: per-instance shadow flags ---
    pub fn set_sprite_instance_shadow_flags(
        &mut self, id: SpriteInstanceId, casts: bool, receives: bool,
    );
}
```

```rust
// roxlap-render/src/billboard_actor.rs  (new, BB.4 — built on the above)

pub struct BillboardActorId { slot: u32, gen: u32 }

/// One animation state = its directional clip variants. `dirs.len()` may be
/// 1 (non-directional), 8 (classic Doom), or any N (uniform angular bins).
pub struct ActorState { pub name: &'static str, pub dirs: Vec<VoxelClipId> }

pub struct BillboardActorDef {
    pub states: Vec<ActorState>,
    pub mode: BillboardMode,            // default Cylindrical
    pub lighting: BillboardLighting,
    pub casts_shadow: bool, pub receives_shadow: bool,
}

impl SceneRenderer {
    pub fn add_billboard_actor(&mut self, def: BillboardActorDef, pos: [f32;3], facing_yaw: f32)
        -> BillboardActorId;
    pub fn set_actor_state(&mut self, id: BillboardActorId, state: &str); // restarts clock
    pub fn set_actor_transform(&mut self, id: BillboardActorId, pos: [f32;3], facing_yaw: f32);
    pub fn remove_billboard_actor(&mut self, id: BillboardActorId) -> bool;
    /// Per frame: for each actor, pick the directional clip from
    /// angle(camera→actor vs actor.facing) → set_clip_instance_clip if it
    /// changed → face_billboards_to → advance the playback clock.
    pub fn update_billboard_actors(&mut self, camera: &Camera, dt: f64);
}
```

## Engine work (per gap)

**BB.0 — GIF importer** (`roxlap-formats`, feature `gif`). Decode with the
`gif` crate, apply disposal/compositing to get full RGBA frames, voxelize
each into a `VoxelFrame` slab (build columns directly; transparent →
air; thickness>1 duplicates the silhouette along +y), map GIF delays →
durations, hand to `VoxelClip::from_frames_auto` (`voxel_clip.rs`). Reject
> `max_dims` with an error (no silent downscale). Pure CPU, fully
CI-testable.

**BB.1 — `set_clip_instance_clip`** (facade + `cpu.rs` + `gpu.rs`). Track
each clip instance's owning `VoxelClipId` in the facade (the `DynInstanceMap`
side-table already keyed by `SpriteInstanceId`). Retarget:
- GPU: rebind the instance's resident model to the new clip's frame-0 chain
  via the existing `set_instance_model` (VCL.2) and update the facade's
  instance→clip + frame=0 bookkeeping; the per-frame cull upload picks it up
  (no volume write).
- CPU: point the instance's dense-grid set at the new clip's cached frames.
Stale/`gen`-mismatched handle ⇒ `false`, no panic (the standing slotmap
discipline). Headless GPU test (2 clips, swap changes the rendered colour)
+ CPU lifecycle test.

**BB.2 — billboarding** (facade + both backends). A billboard flag + stored
world `pos` + `BillboardMode`/`BillboardLighting` per instance (extend the
`DynInstanceMap` record). `add_billboard_instance` = `add_clip_instance_posed`
with the flag set + an identity-or-derived initial pose.
`face_billboards_to(camera)` walks flagged instances, computes the
`Cylindrical`/`Spherical` basis (see convention above), and calls the
existing batched `set_sprite_instance_transforms` once. `BillboardLighting`
selects the normal at the shade site (`FaceNormal` = today's DL.7 path;
`WorldUp` = override the sprite-shade normal to world +z; `AmbientOnly` =
skip direct light). Unit-test the basis math (cylindrical yaw-only;
spherical full) under camera rotation + a non-axis-aligned camera; assert
`right × up ≈ forward` orientation sanity (cf. the voxlap basis-chirality
footgun memo).

**BB.3 — per-instance shadow flags** (facade + both backends). Mutate the
instance's `flags` (`SPRITE_FLAG_NO_SHADOW_CAST`/`_RECEIVE`) live: GPU sets
the bit on the `Instance` record + marks the cull/instance buffer dirty; CPU
sets it on the per-instance `Sprite`. Default for billboards = cast + receive
(what the user wants). Test: a flagged-non-caster billboard drops out of the
occluder; a non-receiver shades unshadowed.

**BB.4 — `BillboardActor`** (`roxlap-render`, on the primitives). A slotmap
of actors; each owns its instance, current state, facing yaw, world pos.
`update_billboard_actors`: per actor, bin `angle(camera_dir_to_actor,
actor.facing)` into `dirs.len()` sectors → desired clip; if it differs from
the showing clip, `set_clip_instance_clip`; then `face_billboards_to`
covers orientation; `advance_voxel_clips`-style clock tick drives the frame.
Directional binning is pure math — unit-test the 8-way sector boundaries +
state swap.

**BB.5 — demo + docs.** A "Doom" demo scene: a GIF-imported monster actor
that walks (directional rotations visible as you strafe around it), casts a
sun shadow on the DL floor + receives the pillar shadows; plus an additive
GIF fire effect; toggles for billboard mode + lighting normal. README
feature row + per-crate docs + CHANGELOG `[Unreleased]`; workspace +
internal-dep version bump. Ship a tiny generated GIF (or check a small one
in) so the demo has an asset.

## Code map (as of 2026-06-30)

Formats — `crates/roxlap-formats/src/`:
- `voxel_clip.rs:102` `VoxelFrame`, `:345` `DecodedClip`, `:903` `from_frames`,
  `from_frames_auto`, `:132` `from_kv6`, `:192` `to_kv6`.
- `kv6.rs:163` `from_fn`, `:188` `from_fn_keep_interior`, `:216` `from_fn_shaded`.
- `sprite.rs:41-49` shadow-flag consts, `:83` `flags` field, `:158-191`
  `casts_shadow`/`receives_shadow`/`with_*` helpers.
- `material.rs` `BlendMode`/`Material`; colour→material map type `&[(u32,u8)]`.
- `lib.rs` module list (add `#[cfg(feature="gif")] pub mod gif_import;`);
  `Cargo.toml` (add optional `gif` dep + `gif` feature).

Facade — `crates/roxlap-render/src/lib.rs`:
- clip API: `:1839` `add_voxel_clip`, `:1852` `add_voxel_clip_with_materials`,
  `:1907` `add_clip_instance_posed`, `:2154` `add_clip_instance_playing`,
  `:1931` `set_clip_instance_frame`, `:2230` `advance_voxel_clips`,
  player ctrl `:2274-2320`.
- sprite instance API: `:544` `DynSpriteTransform`, `:1568`
  `add_sprite_instance_posed`, `:1745/1762` `set_sprite_instance_transform[s]`,
  `:1785` `set_sprite_instance_material`, `:1799` `set_sprite_instance_alpha`.
- `DynInstanceMap` (instance side-table — extend with billboard fields +
  owning clip id) ; `SpriteInstanceId`/`VoxelClipId` slotmaps.

GPU — `crates/roxlap-gpu/src/sprite_model.rs`:
- `SpriteRegistryResident::set_instance_model(cull_idx, chain_id)` (the
  retarget mechanism BB.1 reuses), `update_transforms` (the batched flush
  BB.2 reuses), `Instance.flags` (BB.3).

CPU — `crates/roxlap-core/src/dda_sprite.rs` + `roxlap-render/src/cpu.rs`:
- clip dense-grid cache + per-instance `Sprite` (flags + transform live
  here); `SpriteShade` (lighting/material ctx — the shade site BB.2's
  `BillboardLighting` normal-select touches).

⚠️ Do **not** confuse `roxlap-scene/src/billboard.rs` (far-grid LOD
impostor cache, S6.2) with this stage — unrelated despite the name.

## Sub-substage roadmap

| Stage | Scope | Gate |
|---|---|---|
| **BB.0** ✅ | GIF importer (`roxlap-formats` feature `gif`): decode + disposal compositing + voxelize → flat `VoxelFrame` slabs + delays→durations → `from_frames_auto`. `GifImportOpts` (thickness, pivot, loop_mode, default_frame_ms, keyframe_gap, max_dims; `material_map` dropped — see note). | **Done** — `gif_import.rs` (`Pivot`/`GifImportOpts`/`GifImportError`/`voxel_clip_from_gif`); `gif` 0.13 dep behind off-by-default `gif` feature. 5 unit tests (dims/durations/cutout, thickness extrude, black≠air-sentinel, oversize reject, bottom-center pivot); clippy clean; formats default 192+2+4 green; workspace check clean. |
| **BB.1** ✅ | `set_clip_instance_clip` primitive (facade + CPU + GPU). GPU repoints the instance's resident model at the new clip's frame-0 chain via the existing `set_sprite_instance_model` (no volume re-upload); CPU rebinds `dyn_clip`. Both reset to frame 0; facade resolves instance + clip handles (stale ⇒ `false`). When the instance has an auto-player, its timeline is retargeted to the new clip (new `ClipClock::retarget`: swap durations + loop, restart clock, keep speed/paused). | **Done** — builds + clippy clean; `clip_clock_retarget_swaps_timeline_restarts_keeps_speed` unit test (33 render tests green); workspace check clean. **Test note:** roxlap-render has no headless backend harness (a CPU backend needs a real softbuffer surface; GPU tests live in roxlap-gpu), so the new *facade* logic is unit-tested via `ClipClock::retarget`, the GPU model-swap reuses the VCL.2-tested `set_sprite_instance_model` primitive, and the dyn_clip rebind mirrors `set_clip_frame` — backend behaviour is demo-verified (standing GPU-visual caveat). |
| **BB.2** ✅ | **Billboard orientation.** `BillboardMode {None, Cylindrical, Spherical}` (public); per-instance `BillboardRec {id,pos,mode}` side-table on the facade (reset by `set_sprites`); `add_billboard_instance(clip,pos,mode)`, `set_billboard_mode`, `set_billboard_position`, and `face_billboards_to(&camera)` — one batched `set_sprite_instance_transforms` flush, pruning removed instances (mirrors `advance_voxel_clips`). Pure `billboard_transform` maps the slab's local axes (x=image-horiz, y=normal→camera, z=image-vert) so cylindrical stays vertical (forward = world up `-z`) and spherical fully faces; non-mirrored (image-right = screen-right). World up = `[0,0,-1]` (voxlap z-down; `const BILLBOARD_UP`). | **Done** — builds + clippy clean; 3 basis unit tests (cylindrical upright + height-independent; spherical tilts + normal = dir-to-camera; degenerate/None ⇒ skip; all assert orthonormality); 36 render tests green; workspace check clean. Orientation is per-frame transform only ⇒ no shader change; off (no billboards) ⇒ byte-identical. |
| **BB.2b** ✅ | **`BillboardLighting {FaceNormal, WorldUp, AmbientOnly, FullBright}`** — per-instance normal/shading select at the sprite shade site (locked decision #6, risk R1). Sprite `flags` bits 6/7 (`SPRITE_FLAG_LIGHT_WORLD_UP`/`_AMBIENT_ONLY`); **both bits = `FullBright`** (decoders check it first). GPU `shade_sprite_lit` (`sprite_model_dda.wgsl`): both ⇒ return `albedo` (emissive); bit7 ⇒ `albedo·ambient`; bit6 ⇒ `n_world = (0,0,-1)`. CPU: shared `shade_dynamic_mode` at **both** sprite shade sites (opaque + translucent) — `WorldUp` swaps the up normal, `AmbientOnly` copies the rig with `sun=false, points=[], bands=0`, `FullBright` returns `f32_to_rgb(albedo)`. Facade `set_sprite_instance_lighting` + `BillboardActorDef::lighting` (+ shared `apply_lighting_flags`). Default `FaceNormal` = the DL.7 path verbatim. `FullBright` added after the flame read too dark under `AmbientOnly` (a glow must be emissive, not ambient-dimmed). | **Done** — builds + clippy clean; `apply_lighting_flags_*` (render) + `sprite_light_mode_world_up_and_ambient_only` incl. `FullBright` (core); naga validates the sprite shader; render tests green. Demo "Doom": flame actor is `FullBright`, `L` cycles the signpost through all four modes. GPU visual user-verified. |
| **BB.3** ✅ | Per-instance `set_sprite_instance_shadow_flags(id, casts, receives)` (facade + both backends, live). Shared `apply_shadow_flags` read-modify-writes the XS.4 bits (4/5) on the instance's `Sprite`/`SpriteInstance`, preserving other bits; GPU rides the coalesced transform flush (`flags` already re-uploaded per XS.4). The per-instance counterpart to the template-level `Sprite::with_casts_shadow`/`with_receives_shadow`. | **Done** — builds + clippy clean; `apply_shadow_flags_toggles_bits_and_preserves_others` unit test (all 4 combos + unrelated-bit preservation); 37 render tests + workspace check green. Honored-on-render behaviour (non-caster excluded / non-receiver unshadowed) is the already-landed XS.4 path — demo-verified. |
| **BB.4** ✅ | `BillboardActor` (facade-managed): `BillboardActorId` slotmap + `ActorState {name, dirs}` + `BillboardActorDef {states, mode, speed_q8, casts/receives_shadow}`. `add_billboard_actor` (owns one `add_clip_instance_posed` instance, applies shadow flags, seeds a per-state `ClipClock`), `set_actor_state` (reselect + restart clock), `set_actor_transform`, `remove_billboard_actor`, `update_billboard_actors(&camera, dt)` (two-phase like `advance_voxel_clips`: pick dir-clip by bearing → swap clip only on change → tick clock → frame → batched face-camera). Pure `dir_index` binning (front = camera in facing dir, CCW). Actor instances live outside `self.billboards` (no double-orientation). Built entirely on BB.1+BB.2+the clip clock — no backend reach. | **Done** — builds + clippy clean; `dir_index_bins_view_angle_front_ccw` unit test (N=1/4/8, facing rotation, overhead degenerate); 38 render tests + workspace check green. Runtime swap/orient is demo-verified. |
| **BB.5** ✅ | "Doom" demo scene (`scenes/doom.rs`): an 8-directional walking monster (casts + receives the sweeping sun's shadows; per-direction hue marker; `Q`/`E` turn, `Space` walk/idle), a flickering 1-dir non-casting flame actor, and a standalone signpost billboard (`face_billboards_to`). **All sprites are synthesised as animated GIFs at startup and imported through `gif_import`** — dogfooding the whole encode → import → voxelize path (the demo deps the `gif` encoder + `roxlap-render/gif`). `roxlap-render` re-exports `gif_import` behind its own `gif` feature. Registered in `host.rs`/`scenes/mod.rs`; CHANGELOG `[Unreleased]` + README updated. | **Done** — build + clippy clean (workspace, all-targets); workspace tests green. GPU + interactive **visual user-verified** (headless CI has no display — standing caveat). Version bump left to the maintainer (work sits under `[Unreleased]`). |

| **BB.6** ✅ | **`BillboardUp {World, Camera, Axis([f32;3])}`** — the second, independent half of orientation (BB.2 fixed the image vertical to the world-up constant; that `BILLBOARD_UP` YAGNI note is retired). `BillboardMode` picks the *normal*, `BillboardUp` the *roll* about it: `Camera` = `-camera.down` (screen-locked — a **rolled** camera, e.g. one riding a rotating grid, no longer leans upright art), `Axis` = an app-supplied world axis (a grid's up ⇒ cards stand on a tilted deck; `Cylindrical` yaws about that axis, `bb_reject` generalising the old `[x,y,0]` flatten). Degenerate axis / view-along-axis falls back world-up → fixed, never dropped (the CC.4 position-always-lands rule holds). Per instance `set_billboard_up`; per actor `BillboardActorDef::up` (breaking: new field) + `set_actor_up`. **Actor frame:** `ActorFacing {Yaw(f64), Dir([f32;3])}` + `set_actor_facing`/`set_actor_pose`; `dir_index` takes the ground axis (`actor_ground_axis`: an explicit `Axis`, else world — `Camera` is about screen roll, not the floor) and measures the signed bearing about it, so an actor on a turning deck keeps its sector. The world-yaw path is kept **verbatim** as a fast path ⇒ pre-BB.6 actors bin bit-for-bit. | **Done** — builds + clippy + fmt clean; 5 new unit tests (rolled camera ⇒ image vertical = camera up while `World` still returns `BILLBOARD_UP`; tilted-deck `Axis` ⇒ image vertical = the deck axis + normal in the deck plane + anchor fixed; zero-axis ≡ `World`; view-along-axis falls back orthonormal; `Dir` ≡ `Yaw` on a world floor over a 24×7 sweep; sector invariance under a 39°-deck rotation over 64 azimuths, with the naive world-yaw composition drifting — the reported bug pinned); 97 render tests + workspace green. Demo "Doom": `R` rolls the camera, `U` cycles the monster + signpost through the three ups. |

One sub-stage per commit, each green on `cargo test/clippy/build
--workspace`. BB.0 (pure formats) + BB.1 land first; billboarding + shadow
flags next; the actor + demo close the stage. **BB.6** re-opened the stage
in 2026-08 for the up-axis generalisation (see the row above); the
originating request is `docs/handover-billboard-camera-up.md`.

## Post-stage follow-ups (landed)

- **`FullBright` lighting mode** — added to `BillboardLighting` after the demo
  flame read too dark under `AmbientOnly` (a glow must be emissive, not
  ambient-dimmed). Encoded as **both** flag bits 6|7 (decoders check it first):
  returns the voxel colour at full intensity, ignoring the rig. Both backends.
- **`set_actor_lighting(id, mode)`** — change a `BillboardActor`'s lighting mode
  at runtime (the per-actor counterpart to `BillboardActorDef::lighting`,
  routed through `set_sprite_instance_lighting`). Demo: `O` cycles the monster.
- **PNG / APNG importer** (`png_import`, `png` feature) — the truecolor
  counterpart of `gif_import`: `voxel_clip_from_png_frames` (a sequence of
  same-size PNG files) + `voxel_clip_from_apng` (a single animated PNG, with
  APNG dispose/blend compositing). PNG's 8-bit alpha is a cutout at a
  configurable `alpha_cutoff` (per-pixel translucency isn't preserved — a
  `VoxelClip` is RGB-only; use a translucent instance material for a uniform
  fade). The voxelization core (`voxelize_rgba` + `assemble_clip` + `Pivot`)
  was extracted to a shared `slab` module that both importers use. 4 png unit
  tests (sequence/cutout/size-mismatch/APNG); `gif_import` refactored onto
  `slab` with its 5 tests still green.

## Tests

- **Formats (CI):** GIF decode → slab voxelization (colour/cutout/dims),
  delays→durations, disposal compositing on a partial-frame GIF, oversize
  rejection; clip encode→`decode()` identity.
- **Primitive (CI):** `set_clip_instance_clip` swaps the rendered model
  (headless GPU + CPU); transform + clock policy preserved across the swap;
  stale handle no-op.
- **Billboard math (CI):** cylindrical = yaw-only (up stays world-up under
  pitch); spherical fully faces; basis non-singular + correctly handed for a
  rotated camera.
- **Actor (CI):** N-way sector boundaries pick the expected directional
  clip; `set_actor_state` swaps clip + restarts the clock.
- **Regression anchor:** a scene with no billboard instances renders
  byte-identical to pre-BB (the standing opaque/`lights==None` gate).
- **Visual:** BB.5 demo (manual GPU verification, no display in CI).

## Risks / watch-items

- **R1 — camera-dependent shading.** A camera-facing slab's front normal
  tracks the camera, so `FaceNormal` lighting shifts as you orbit.
  Mitigation: `BillboardLighting::{WorldUp, AmbientOnly}` (locked decision
  #6) — `AmbientOnly` is the most Doom-faithful (flat), `WorldUp` keeps
  stable directional shading. Default `FaceNormal` matches DL.7.

- **R2 — thin shadow edge-on to the light.** A 1-voxel slab parallel to the
  light casts a near-zero shadow. Mitigation: cylindrical billboard (slab
  stays vertical → overhead sun always has body) + `thickness > 1` opt-in in
  `GifImportOpts` (extrudes the silhouette along +y for a real shadow
  volume).

- **R3 — memory on big / many-directional GIFs.** 64×64 × 8 dirs × N frames
  is sizeable. Mitigation: a clip is registered **once** and shared by all
  instances/actors of a type (100 monsters = 1 clip set + 100 cheap
  instances); tight per-frame bbox crop in the importer; `max_dims` guard;
  LOD via `add_lod` per frame later. Document the per-type cost.

- **R4 — voxel-block aesthetic vs crisp 2D.** Slabs render nearest/blocky
  (ideal for pixel art) but show cube edges on the silhouette / at extreme
  angles. Accepted as the voxel-engine look (locked decision #1). The true
  textured-card path stays a documented future option if crisp upscaled 2D
  is ever needed.

- **R5 — GIF is 1-bit alpha only.** No per-pixel semi-transparency from a
  GIF (locked decision #4). Glow via a whole-clip `Additive` material;
  smooth alpha edges need a PNG/APNG-sequence importer (deferred, mirrors
  BB.0 with the existing `png` dep).

- **R6 — directional source assets.** 8-way "rotations" need 8 GIFs (or a
  convention) per state; the actor bins by angle but cannot synthesize
  unseen views. Mitigation: `dirs.len()==1` degrades to a single
  non-directional clip; document the N-GIF authoring convention.

- **R7 — cross-pass ordering (inherited TV v1 limit).** A translucent
  billboard between glass terrain and the wall behind composites over the
  baked glass, not under it (TV decision #6). Unchanged by BB; documented.

## Validation (every sub-substage)

- `cargo test/clippy/build --workspace` green; **no-billboard ⇒
  byte-identical** is the headline regression gate.
- Billboard basis math unit-tested independent of any backend (rotation +
  chirality), per the voxlap basis-chirality discipline.
- "No silent caps": importer rejects oversize GIFs and over-`max_dims`
  frames with an error/`log::warn!`, never a silent downscale or truncation.
- GPU/interactive paths dogfooded in the BB.5 demo (manual visual — headless
  CI has no display, standing caveat).
