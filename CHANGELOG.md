# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Live render-pipeline HUD** (RP.3) — the scene-demo gains a "Render pipeline"
  egui panel (top-right) that drives `set_render_resolution` / `set_ssaa` /
  `set_posterize` at runtime: resolution mode (Native / Fixed `w×h` / Scale),
  SSAA factor, posterize levels + dither (none / Bayer / blue-noise), with the
  live logical + march sizes shown. The `ROXLAP_RENDER_RES` / `ROXLAP_SSAA` /
  `ROXLAP_POSTERIZE` / `ROXLAP_DITHER` env vars now just seed the panel's
  initial state. README + per-crate docs updated. Closes the RP pipeline
  (RP.0 fixed target + RP.1 SSAA + RP.2 posterize + RP.3 controls).
- **Posterize + dither** (RP.2) — `SceneRenderer::set_posterize(Option<PosterizeConfig>)`
  applies a reduced-palette post at the logical resolution in the resolve step
  (after the SSAA downfilter, before the nearest upscale), so each hard pixel
  quantizes once. Per-channel `levels_r/g/b` (`<= 1` ⇒ untouched) +
  `DitherMode {None, Bayer4x4, BlueNoise}` (blue-noise = texture-free
  interleaved-gradient noise). GPU folds it into `scene_resolve.wgsl` (posterize
  fields written per-frame in the resolve uniform — no pipeline rebuild); CPU
  into `posterize_pixel` (exact per-channel quantization). `None` posterize +
  `ssaa == 1` stays byte-identical. scene-demo: `ROXLAP_POSTERIZE=N` +
  `ROXLAP_DITHER=none|bayer|blue`.
- **SSAA + box-downfilter resolve** (RP.1) — `SceneRenderer::set_ssaa(factor)`
  (clamped `1..=4`) supersamples the retro grid: the raycaster marches at
  `logical × factor`, then a box-downfilter resolves back to the logical grid
  before the nearest upscale — anti-aliasing edges + reducing rotation/movement
  shimmer while keeping hard pixels. `render_dims()` now reports the march size
  (`logical × ssaa`); `logical_dims()` the resolved grid. GPU: a
  `scene_resolve.wgsl` compute pass (framebuffer→resolve buffer; the blit reads
  the logical resolve buffer). CPU: an exact integer box average
  (`downfilter_pixel`). `ssaa == 1` is a byte-exact identity, so the RP.0 paths
  are unchanged. scene-demo: `ROXLAP_SSAA` (default 1; CPU pays the full N² ray
  cost, so opt-in).
- **Fixed-resolution render target** (macro-stage RP, RP.0) — the scene now
  marches into a fixed **logical** render target that is nearest-upscaled to the
  window, so the per-pixel raycaster's cost — and thus the frame rate — stops
  depending on the window size. New `SceneRenderer::set_render_resolution`
  (`RenderResolution {Native, Fixed {w, h}, Scale(f32)}`) plus
  `render_dims()` / `logical_dims()` introspection. Mirrored on both backends:
  the GPU scene/sprite passes + framebuffer/depth run at the render size and
  `scene_blit.wgsl` integer-nearest-upscales to the swapchain; the CPU backend
  upscales its logical framebuffer into a native-size output and rasterises the
  egui HUD there so it stays crisp. Debug-line / image overlays + screen→world
  picking map window pixels into the render grid. **`Native` (the default) is
  byte-identical to pre-RP rendering.** The scene-demo defaults to an
  `860×520` grid; override with `ROXLAP_RENDER_RES` (`native` | `WxH` |
  `<scale>`). Foundation for the RP.1 SSAA + RP.2 posterize/dither passes.

## [0.19.0] — 2026-06-30

### Added

- **GIF billboard sprites** (macro-stage BB) — Doom/Build-style flat,
  camera-facing animated cutouts that are first-class lighting citizens (they
  cast + receive the dynamic shadows + lighting from the DL/XS stages). Each
  GIF frame is voxelized into a flat 1-voxel slab and played back as an
  ordinary voxel clip, so shadows, lighting, materials, and playback all come
  for free.
  - **`gif_import`** (BB.0) — `voxel_clip_from_gif(bytes, &GifImportOpts)`
    decodes an animated GIF (with disposal compositing) into a `VoxelClip` of
    flat `[W, thickness, H]` slabs (transparent pixels → cutout, GIF delays →
    clip durations). In `roxlap-formats` behind the `gif` feature, re-exported
    from `roxlap-render` behind its own `gif` feature.
  - **`set_clip_instance_clip`** (BB.1) — retarget a live clip instance onto a
    different clip (restart at frame 0, keep transform + clock policy) via the
    existing GPU model-swap — the primitive behind directional + state swaps,
    no remove/respawn.
  - **Billboard orientation** (BB.2) — `BillboardMode {None, Cylindrical,
    Spherical}` + `add_billboard_instance` / `set_billboard_mode` /
    `set_billboard_position` / `face_billboards_to(&camera)` (one batched
    transform flush). Cylindrical (default) keeps the slab vertical so its cast
    shadow stays sane as the camera orbits.
  - **Per-instance shadow flags** (BB.3) —
    `set_sprite_instance_shadow_flags(id, casts, receives)` toggles an
    instance's XS.4 shadow participation live (the per-instance counterpart to
    `Sprite::with_casts_shadow` / `with_receives_shadow`).
  - **`BillboardActor`** (BB.4) — a high-level directional actor:
    `add_billboard_actor` / `set_actor_state` / `set_actor_transform` /
    `remove_billboard_actor` / `update_billboard_actors(&camera, dt)`. The
    renderer picks the directional (N-way) clip from the view angle, plays a
    named-state animation, and faces it to the camera.
  - **"Doom" demo scene** (BB.5) — an 8-directional walking monster (casts +
    receives the sun's shadows) + a flickering non-casting flame + a standalone
    signpost billboard, all synthesised as GIFs at startup and imported through
    `gif_import` (dogfooding the full path). New scene in `roxlap-scene-demo`.
  - **`BillboardLighting`** (BB.2b) — per-instance shading mode (sprite
    `flags` bits 6/7) so a camera-facing billboard needn't suffer the
    camera-dependent `N·L` of its (camera-tracking) face normal: `FaceNormal`
    (default), `WorldUp` (a fixed world-up normal — stable directional
    shading), `AmbientOnly` (flat cutout, ambient term only), and `FullBright`
    (**emissive** — the colour at full intensity, ignoring lighting; the right
    look for glows like fire/spell auras). Both backends (`shade_sprite_lit`
    WGSL + the CPU `shade_dynamic_mode` at both sprite shade sites); set via
    `set_sprite_instance_lighting` / `BillboardActorDef::lighting`. Also
    `set_actor_lighting` to change a `BillboardActor`'s mode at runtime.
    `FaceNormal` is byte-identical to the prior look.
  - **PNG / APNG importer** (`png_import`, `png` feature on `roxlap-formats` /
    re-exported from `roxlap-render`) — the truecolor counterpart of
    `gif_import`: `voxel_clip_from_png_frames` (a sequence of independent PNG
    files, same size) and `voxel_clip_from_apng` (a single animated PNG, with
    dispose/blend compositing). PNG's 8-bit alpha is resolved as a cutout at a
    configurable `alpha_cutoff`. The voxelization core is shared with
    `gif_import` (`slab` module).
- **Per-instance sprite RGB tint** — each sprite instance carries a packed
  `0x00RRGGBB` tint that multiplies its voxel colours (per channel), so
  instances of one model can be recoloured cheaply. `0x00FF_FFFF` (white, the
  default) is a no-op, so existing sprites render byte-identically. Set it on a
  model via `Sprite::with_tint` / the `tint` field, or per dynamic instance via
  `SceneRenderer::set_sprite_instance_tint`. Both backends (CPU
  `dda_sprite` + GPU `sprite_model_dda`); no new bindings (packed into the
  instance record's free slot). Tint is colour only — transparency stays on
  the translucent-material + `alpha_mul` path.

## [0.18.0] — 2026-06-30

### Added

- **Cross-scene shadows + lit translucent sprites** (macro-stage XS, extends
  DL). Three related gaps closed:
  - **Lit translucent sprite layers** (XS.0) — translucent sprite/clip voxels
    are now shaded by the dynamic-lighting rig (sun + point lights + cel +
    ramp, flat per voxel) like opaque sprites and translucent *terrain*
    already were, on both backends. Previously the accumulate paths used the
    flat baked colour (`cast_local_layers` on the CPU, `march_instance_layers`
    in `sprite_model_dda.wgsl`); both now light each layer via the model-local
    face normal. A disabled rig is byte-identical.
  - **Cross-grid hard shadows, CPU** (XS.1) — a shadow ray cast from a voxel
    in one grid now tests *every* grid in the scene, so e.g. a ship grid drops
    a shadow on the ground grid. A world-space `SceneOccluder`
    (`roxlap-scene`) implements the new `roxlap_core::WorldOccluder` trait; the
    CPU DDA reaches it through `DdaEnv::world_shadow` + the hit grid's
    local→world transform. Built once per frame (only when a caster is active),
    borrowing the scene immutably — the composed render loop was split into a
    `&mut` cache-prep pass and an immutable render pass so the occluder can
    coexist with it. No caster ⇒ no occluder built (unchanged).
  - **Sprite shadows, CPU** (XS.2) — sprites now both **cast** and **receive**
    hard shadows. A `SpriteOccluder` (decoded sprite volumes + world poses)
    implements `WorldOccluder` and is composited with the grid occluder
    (`CompositeOccluder`): the terrain render sees it (sprites darken the
    ground), and the sprite pass queries it (a sprite is darkened by terrain or
    another sprite between it and a caster). Built only when a caster is active.
  - **Cross-grid hard shadows, GPU** (XS.3) — the scene-DDA shader's
    `shadow_occluded` now runs per grid inside a `shadow_occluded_world`
    wrapper: a shadow ray is lifted to world space and tested against every
    grid, so a caster in one grid shadows another (matching the CPU). Each
    grid's world transform (origin + local→world rotation) is packed into the
    existing per-grid camera buffer (binding 15) — no new storage buffer (the
    16-buffer limit is saturated) — via `GridWorldTransform`. Identity
    transforms reproduce the prior intra-grid shadows byte-for-byte.
  - **Per-sprite shadow flags** (XS.4) — a sprite can opt out of casting and/or
    receiving hard shadows via `SPRITE_FLAG_NO_SHADOW_CAST` /
    `SPRITE_FLAG_NO_SHADOW_RECEIVE` (both default to participating), with
    `Sprite::{casts_shadow, receives_shadow, with_casts_shadow,
    with_receives_shadow}` helpers. Honored on both backends.
  - **GPU sprites receive terrain shadows** (XS.4.2) — on devices that grant
    enough storage buffers (`GpuRenderer::sprite_shadows_capable()`), the sprite
    pass marches the terrain occupancy (the scene pass's exact ABI, spliced in
    from `sprite_terrain_shadow.wgsl`) so a sprite is darkened where terrain (or
    another grid) blocks the sun / a point light. Needed raising the device
    storage-buffer request to 22 (`pick_required_limits`); devices below that
    fall back to unshadowed GPU sprites. Per-instance `receives_shadow` honored.
  - **GPU sprites cast onto terrain** (XS.4.3) — the mirror: on a capable
    device the scene pass's `shadow_occluded_world` also marches the visible
    sprite volumes (the sprite registry bound at 19..21, spliced in from
    `scene_sprite_shadow.wgsl`), so a sprite drops a shadow on the ground.
    Per-instance `casts_shadow` honored. With XS.4.2 this completes
    bidirectional GPU sprite shadows, at full CPU/GPU parity.
- **Voxel ambient occlusion** (macro-stage AO) — a CPU bake pass that writes
  per-voxel ambient occlusion into the brightness byte, which the dynamic
  lighting (DL) reads as its ambient/AO fill: open surfaces stay bright while
  crevices, inside corners, and contact points (pillar bases, the monument's
  foot) darken — even where no sun/point light directly shades them. Reuses the
  existing lighting bake (`EstNormCache`): `EstNormCache::ambient_occlusion` is
  computed **per exposed face** (for each air-facing voxel face, how much solid
  sits in front of it), so it darkens **only concave** edges — flat faces and
  convex edges stay open, no "pillow" outline. A new `lightmode == 3` bakes it.
  Both backends benefit at zero render cost (it's the same byte they already
  multiply in). Tunable via `AoParams { strength, radius, min_floor }`
  (`bake_ao_pub`); the "Lighting" demo bakes AO into its floor/pillars/monument
  and retunes the depth live with `N`/`M`. (CPU bake; the byte feeds GPU + CPU
  alike.) For **stacked grids**, `EstNormCache::build_with_reader_z` reads the
  chunks above/below (`chz±1`) for the bake's `±ESTNORMRAD` z-padding, so AO
  (and the directional `estnorm` bake) stay continuous across a chunk z-seam
  instead of seeing fake air-above / bedrock-below at the boundary; the
  scene-graph `bake_lightmode` and the demo bakes use it.

- **Dynamic lighting** (macro-stage DL; `PORTING-DYNLIGHT.md`) — runtime
  lighting layered on the scene-DDA raymarcher: one coloured directional
  **sun**, several coloured **point lights**, and **stylized hard voxel
  shadows** cast by the sun and a chosen subset of point lights (the rest
  shadowless). Started GPU-only, then brought to full **CPU parity** (CPU.1 +
  CPU.2 below). The baked brightness byte is reinterpreted as the ambient/AO
  fill (`out = albedo·ambient + Σ direct`). Lights are per-frame via
  `FrameParams.lights`
  (`LightRig { sun, points, ambient, shadow_strength, shadow_bias_voxels,
  shadow_max_dist }`); `None` ⇒ byte-identical to pre-DL. New `DirectionalLight`
  / `PointLight` / `LightRig` types in `roxlap-render`.
  - **Sun + point lights** (DL.1/DL.2) — `shade_lit` in `scene_dda.wgsl`:
    raw albedo × ambient + N·L diffuse per light, using the DDA's hit-face
    normal (free) for terrain. Point lights attenuate with a smooth quadratic
    falloff and a hard radius cut. Lights are transformed into each grid's
    local frame on the CPU (mirroring the per-grid cameras); the per-grid sun
    direction rides in the camera struct to stay within the GPU's 16
    storage-buffer limit.
  - **Hard shadows** (DL.3) — `shadow_occluded`, a dedicated intra-grid shadow
    DDA (chunk-skipping outer loop + mip-0 inner voxel walk) reusing the scene
    occupancy. Shadow-ray origin is biased along the surface normal (anti-acne);
    the in-shadow floor is `1 − shadow_strength`. `MAX_SHADOW_CASTERS` caps the
    casters (excess demoted to shadowless with a warning). Cross-grid shadows
    deferred.
  - **Lit sprites** (DL.4) — opaque sprites/clips shade with the sun + point
    lights using their **true per-voxel normals** (voxlap `univec[256]` mapped
    from each voxel's `dir` index, rotated to world). World-space lights for the
    sprite pass; sprite shadows deferred.
  - **Stylized (retro) lighting** (DL.6) — smooth N·L reads as generic Phong
    and flattens the voxel identity, so terrain lighting gains an opt-in retro
    look: `LightRig.bands` (cel quantization — the sun key + each point factor
    snap to `bands + 1` discrete levels) + a gradient-map ramp from
    `LightRig.shadow_tint` (cool, unlit) to the sun colour (warm, lit), giving
    terraced, hue-shifted shading where shadows tint cool instead of just
    darkening. `bands == 0` keeps the smooth path byte-identical. The
    "Lighting" demo's `J` toggles stylized ↔ smooth.
  - **"Lighting" demo scene** — a sweeping sun + three orbiting coloured point
    lights over a pillared floor; `P` pauses the sun, `K` toggles sun shadows,
    `L` toggles the point lights, `J` toggles stylized/smooth.
  - **Stylized sprites + clips** (DL.7) — the cel banding + gradient-ramp +
    flat-per-voxel stylization now extends to the GPU sprite/clip pass (using
    each voxel's true normal), so animated characters and voxel clips match the
    terrain's retro look. `style_bands == 0` keeps the smooth path.
  - **CPU diffuse lighting** (CPU.1) — the dynamic lighting now also runs on the
    **CPU** backend (sun + point lights + cel + ramp, flat per voxel), terrain
    **and** sprites/clips, so the same `FrameParams.lights` rig lights both
    backends. Diffuse is arithmetic-only, so it's effectively free on the
    (bandwidth-bound) CPU path. `lights: None` keeps the CPU path byte-identical
    to the baked-brightness render. (CPU sprites use the DDA face normal — flat
    per voxel — since the CPU sprite store has no per-voxel normals.)
  - **CPU hard shadows** (CPU.2) — the CPU backend now casts the sun + point
    shadows too, so both backends are on full parity. A shadow ray marches a
    3D-DDA toward each caster through the same render `Sampler` occupancy the
    camera ray uses (bounded by `shadow_max_dist` / the light distance), and an
    occluded sample keeps `1 − shadow_strength` of that caster; the same
    `MAX_SHADOW_CASTERS` cap as the GPU applies (excess point casters demoted
    with a warning). Only marched when a caster is flagged and
    `shadow_strength > 0`, so an unshadowed rig (and `lights: None`) stays
    march-free and byte-identical. Sprites are unshadowed (matching the GPU).
    This is the slow CPU fallback's slowest path, but correct and on visual
    parity with the GPU.

## [0.17.0] — 2026-06-28

### Added

- **`SceneRenderer::wait_idle`** — blocks until the active backend has drained
  all in-flight work and releases any acquired-but-unpresented swapchain frame
  (GPU: `device.poll(Wait)`; CPU: no-op). Call it at shutdown before dropping
  the renderer and its window.

- **`BlendMode::Volumetric`** (Beer–Lambert) — the thickness-aware transparency
  mode for *filled* volumes (true smoke, fog, murky water), the deferred
  follow-up to the per-span `AlphaBlend`. Where `AlphaBlend` composites one
  alpha per surface run (opacity independent of thickness — ideal for
  shells/glass), `Volumetric` weights each voxel's opacity by the ray's path
  length through it: per-cell effective opacity `1 − (1 − alpha)^seg_len`
  (`seg_len` in voxel units), so a boundary sliver contributes ≈0 (no
  voxel-grid dicing) and a filled volume thickens smoothly with depth. Lands on
  both backends and both passes (sprites/clips + terrain), CPU pinned then GPU
  matched. New `Material::volumetric(alpha)`. The "Transparency" demo gains a
  filled volumetric fog cloud whose core reads denser than its rim.
  - **`Kv6::from_fn_keep_interior`** — surface extraction normally culls every
    enclosed voxel (a solid cube is a hollow shell), which would defeat
    Volumetric (a filled cloud would render as front+back faces with air
    between). This variant keeps interior voxels whose colour a predicate
    accepts, so translucent/volumetric bodies stay solid through while opaque
    interiors are still dropped (the storage win). (Terrain `.vxl` stores solid
    runs, so its interiors were already traversable — this gap was kv6-only.)

- **Mixed-material animated clips** — per-voxel materials (TV.3) now extend to
  voxel clips (`.rvc`), the animated analogue of
  `add_sprite_model_with_materials`: `SceneRenderer::add_voxel_clip_with_materials`
  classifies every frame's voxels into per-voxel material ids by a
  colour→material map, so an animated clip can mix opaque and translucent
  voxels (an opaque torch handle around an additive flame, a pulsing glass
  orb) on both backends. Previously clips could only carry a whole-instance
  uniform material; the per-voxel path was wired for static sprites but not
  for clip frames. An empty map is byte-identical to `add_voxel_clip`. The
  "Transparency" demo scene gains a pulsing glass-orb clip dogfooding it.
  - The per-voxel materials are also preserved across the **in-place
    single-frame edit** (`update_clip_frame`, which now re-classifies the
    edited frame from the clip's registered map) and the **streaming-clip**
    path: `add_streaming_clip_with_materials` + `refresh_sprite_model_with_materials`
    re-apply the colour→material map on every per-frame model re-upload.

### Fixed

- **Clean GPU teardown on exit** — the native demos (`scene-demo`, `cave-demo`,
  `sdl-demo`) now drain the GPU and drop the renderer (wgpu device/queue/surface)
  *before* the window, via a winit `exiting`/`suspended` handler (and a
  renderer-before-window field order so the panic-unwind path tears down the
  same way). Previously an exit could yank the swapchain mid-frame / drop the
  window before the surface, leaving the driver or compositor showing stale
  buffers — the "leftover triangles / flicker" after an unclean exit. (The
  runtime already reconfigured the surface on `Lost`/`Outdated`; this fixes the
  shutdown side.)

## [0.16.0] — 2026-06-28

### Added

- **Transparent voxels** (macro-stage TV; `PORTING-TRANSPARENCY.md`):
  alpha-blended and additive voxels for effects (smoke, fire, spell auras,
  muzzle flashes) and glass/water, on both the CPU and GPU backends. The
  per-pixel front-to-back 3D-DDA renderer composites translucent voxels in
  visit order, so it is order-correct without any depth sorting; an
  all-opaque scene renders byte-for-byte as before.
  - **Material model** — `roxlap_formats::material`: `BlendMode`
    (`Opaque` / `AlphaBlend` / `Additive`), `Material { alpha, mode }`, and
    a 256-entry `MaterialTable` global palette (id 0 is permanently
    `Material::OPAQUE`). Facade `SceneRenderer::define_material` / `material`.
  - **Sprites & clips** — per-pixel accumulate-and-continue in the CPU
    `dda_sprite` raycaster and the GPU `sprite_model_dda.wgsl` pass:
    `Additive` (commutative glow) and `AlphaBlend` (`over`), a per-instance
    `alpha_mul` (`set_sprite_instance_material` / `set_sprite_instance_alpha`)
    for cheap fade animation, and **per-span compositing** (one alpha layer
    per contiguous solid run / material change) so a translucent shell no
    longer reads as a voxel grid.
  - **Mixed-material models** (TV.3) — a single model can mix opaque and
    translucent voxels (opaque frame + glass) via a colour→material map:
    `add_sprite_model_with_materials`.
  - **Terrain** (TV.4–TV.6) — glass/water as world (grid) geometry, resolved
    from a global terrain colour→material map at render time
    (`SceneRenderer::set_terrain_materials`) — no `.vxl` format change. CPU
    `dda` + GPU `scene_dda.wgsl` both accumulate front-to-back.
  - **Demo** — a "Transparency" scene (glass pane, additive glow, pulsing
    smoke, a mixed opaque-frame+glass window, and a world glass wall).
  - CPU `render_sky_fill`: the panorama sky now fills every background pixel
    (sprite/effect-only views + the margins around small grids), matching the
    GPU.

## [0.15.0] — 2026-06-27

### Added

- **Animated voxel-sprite clips (`.rvc`) — "GIF/MP4 for voxel models"**
  (macro-stage VCL; `PORTING-VOXEL-CLIP.md`). A fixed-bbox sequence of
  voxel frames encoded as keyframes + inter-frame diffs, for effects like
  flame, spells, and muzzle flashes — decoded to a runtime *flipbook* and
  played back by selecting a frame per render (no per-frame volume
  re-upload).
  - `roxlap_formats::voxel_clip`: `VoxelClip` / `VoxelFrame` (dense-column
    layout matching the GPU sprite model) + `DecodedClip`; I/P codec
    (`from_frames` / `decode`), `serialize` / `parse` (`RVCL` chunked
    container), `LoopMode`, and `frame_at` playback math.
  - `.kv6` authoring bridge: `VoxelFrame::from_kv6` (re-index a voxel
    sprite into one clip frame) + `VoxelClip::from_kv6_frames` (encode a
    sequence of same-dims `.kv6` frames straight into a clip), so clips can
    be built from existing voxel models, not just procedurally.
  - `VoxelClip::from_frames_auto` / `from_kv6_frames_auto` — auto-choose
    keyframe vs. delta per frame (the codec's I-frame decision) instead of a
    fixed interval: a frame is keyframed on a "scene change" (its delta would
    be ≥ 60% of a keyframe) or to cap keyframe spacing (`max_keyframe_gap`),
    so seek points + size adapt to the content.
  - `.rvc` per-chunk deflate (format **v2**, via `miniz_oxide` — the
    formats crate's first runtime dependency). Each chunk is deflated when
    that shrinks it (a `flags` byte in the envelope; occupancy bitmasks +
    colour runs compress hugely), else stored raw. v1 files still parse.
  - `voxel_clip::StreamingClip` — a seekable, **O(1-frame)-memory** cursor
    over a clip's I/P stream (replays deltas from the nearest keyframe),
    the streaming alternative to `DecodedClip` for huge clips: holds one
    reconstructed frame + the compact encoded stream instead of N full
    frames. Plus `VoxelFrame::to_kv6` (the `from_kv6` inverse) to materialise
    a frame as a flat-lit `.kv6` model.
  - `SceneRenderer` streaming-clip facade: `add_streaming_clip` (one model +
    the cursor, vs the flipbook's N volumes) / `add_streaming_clip_instance`
    / `set_streaming_clip_frame` (seek + re-upload the single model) /
    `remove_streaming_clip` (+ `StreamingClipId`). For huge clips where the
    flipbook's resident N-frame footprint is too costly; all instances of a
    streaming clip share its one model (same current frame).
  - `voxel_clip::pad_stats` / `PadStats` (+ `DecodedClip::pad_stats`) — a
    clip is a *fixed* bbox, so a frame whose content fills only a corner
    still pays the full per-frame occupancy. Reports the declared `dims` vs.
    the tight `content_dims` (`pad_ratio` / `is_wasteful`) so an asset
    pipeline can warn; the encoder stays side-effect-free.
  - `SceneRenderer` auto-playing clips: `add_clip_instance_playing`
    (flipbook) / `play_streaming_clip` (streaming) attach a per-clip
    playback clock (Q8 speed + start phase), advanced by a single
    `advance_voxel_clips(dt)` — the host no longer hand-drives `frame_at` +
    `set_clip_instance_frame` per instance per frame.
  - **Editor/authoring API for clips + characters.** Queries:
    `clip_frame_count`, `clip_metadata` (`ClipMetadata`: dims / pivot /
    scale / loop / per-frame durations), `get_clip_instance_frame` — so an
    inspector + timeline scrubber needn't shadow the `DecodedClip`. Live
    auto-player control (play/pause/scrub): `set_clip_instance_paused` /
    `is_clip_instance_paused` / `set_clip_instance_speed` /
    `set_clip_instance_clock_ms` (scrub) / `clip_instance_clock_ms` (and the
    per-clip `set_streaming_clip_paused` / `_speed` / `_clock_ms` /
    `is_streaming_clip_paused` / `streaming_clip_clock_ms` analogues for
    streaming players). `update_clip_frame(id, frame, &VoxelFrame)` re-uploads one frame in
    place (O(1 frame), vs remove + re-add). `remove_character` now frees the
    models + clips it registered (no leak when hot-swapping). New
    `set_character_world_transform` teleports a character (re-solve + re-pose)
    without ticking its animation / clip clocks.
  - Editor-API hardening: `add_streaming_clip_instance` now returns a
    distinct `StreamingInstanceId` (+ `set_streaming_instance_transform` /
    `remove_streaming_instance`) — a streaming clip's frame is per-clip, so
    the type makes "scrub two instances independently" a compile error rather
    than a silent coupling. The clip / character / streaming slotmaps bump a
    generation epoch on `set_sprites`, so a handle held across a reset
    resolves to `None` instead of aliasing whatever re-took its slot.
    `upload_image` returns `Option<ImageId>` (was `ImageId(0)` on error,
    indistinguishable from the first valid id).
  - GPU flipbook: `sprite_model_from_clip_frame` (field-move upload) +
    `SpriteRegistryResident::set_instance_model` (the per-frame select).
  - CPU flipbook: `roxlap_core::ClipFlipbook` (cached `SpriteDense` per
    frame) + the generalised `draw_sprite_dense`.
  - `SceneRenderer` facade: `add_voxel_clip` / `add_clip_instance_posed`
    / `set_clip_instance_frame` (+ `VoxelClipId`).
- **RKC v3 — multi-attachment bones.** `Character`'s bones now carry a
  list of `Attachment`s (static KV6 meshes and/or animated voxel clips,
  each with a `local_offset` + `ClipPlayback`) instead of a single mesh;
  `MeshRef::Clip` indexes a new `Character::voxel_clips` (`VCLP` chunk).
  v2/v1 files are rejected (regenerate from demiurg).
- **Character attachment runtime.** `SceneRenderer::add_character` /
  `advance_character` / `remove_character` (+ `CharacterId`) emit one
  renderer instance per bone attachment — static meshes sit on their
  bones, clip attachments play back on their own clocks, all driven by the
  skeletal animation. `roxlap_core::kfa_draw::compose_attachment` composes
  a bone's solved world transform with an attachment's local offset. The
  scene-demo dogfoods it with a procedural flame clip on coco's swinging
  arm.

### Fixed

- **Voxel-clip hardening** (review pass over the VCL stage). Defensive fixes
  to the new clip code: `SpriteModelRegistry::set_instance_model` now guards
  `chain_id` (a tombstoned/out-of-range chain no longer index-panics — new
  `model_checked`); the CPU backend's `set_sprites` clears `clip_books` to
  match the GPU (clip indices restart at 0 on both; no leak);
  `remove_voxel_clip` detaches instances still pointing at the removed clip
  on both backends; `VoxelFrame::from_kv6` bounds-checks a malformed kv6
  (`ylen` / `voxels` disagreeing with the header) instead of panicking;
  `.rvc` inflate caps an untrusted `raw_len` at 64 MiB (decompression-bomb
  guard); and `frame_at` / `total_ms` compute in u64 / saturate so a very
  long clip can't overflow `2·total`.
- **GPU scene upload truncated dense chunks' colours** (`roxlap-gpu`).
  The per-chunk colour stride was a fixed `COLORS_PER_CHUNK_WORDS`
  (65536 u32s), sized for sparse terrain chunks (~36 k colours). A
  *fully dense* chunk — e.g. the cave demo's single 128×128×256 chunk
  (~207 k colours across its mip ladder) — overflowed the stride and had
  its colour data truncated; since columns upload in `y·vsid + x` order,
  the high-`y` spatial half of the chunk rendered **black** on the GPU
  backend (the CPU backend was unaffected). The stride is now **adaptive
  per grid** — grown to fit the grid's densest chunk, floored at the old
  default — so dense chunks upload in full while sparse grids keep the
  small stride (and now use slightly less memory). `GpuSceneResident`
  carries the per-grid stride so streamed re-uploads (`refresh_chunk`)
  address colours identically. Regression test:
  `scene_dda_dense_chunk_colours_not_truncated`.

### Changed

- **GPU flipbook-clip registration flushes the staging pool in batches**
  (`roxlap-gpu` / `roxlap-render`; #4). `add_voxel_clip` uploads N frame
  volumes at once; a big flipbook (or many clips registered in one frame)
  could stage that many `write_buffer`s before the next submit and exhaust
  the device staging pool — the same crash the `chunk_upload_budget` guards
  against, which then panics egui-wgpu. It now flushes (`GpuRenderer::
  flush_writes`, an empty submit) every `ROXLAP_GPU_CLIP_BUDGET` frames
  (default 8; `0` = unbounded). Streaming clips upload one model, so they
  sidestep the spike entirely.
- **`roxlap-scene-demo` refactored into a menu-driven multi-scene
  showcase** (demo-only; macro-stage DS, `PORTING-DEMO-SCENES.md`). The
  2296-line kitchen-sink `App` (≈34 fields, ≈20 hotkeys, every feature
  live at once) is split into a thin host (`host.rs`) + a `DemoScene`
  trait (`scene_api.rs`) + one module per scene (`scenes/`). The host
  owns the window, `SceneRenderer`, egui HUD, shared fly-camera +
  mouse-look, and FPS; `Tab` opens a scene picker. Six scenes each
  showcase one feature cluster: **World** (streaming hills + ship +
  collision-fly), **Sprites** (coco field + shoot-to-carve + streaming
  ring), **Animation** (KFA arm + flame-clip character), **Picking**
  (top-down `view_ray`/`pick`), **Primitives** (`draw_lines` /
  `draw_images` / `pick_image`), and **Empty**. Pruned the rarely-used
  bits: `ROXLAP_AUTOFLY`, the in-app bench, the `H` A/B pose toggle, the
  `F` frame-capture, `ROXLAP_NO_SPINNER`, and `ROXLAP_GPU_NO_SPRITES`.
  Kept `ROXLAP_GPU` / `STATIC` / `RKC*` / `KFA_DUMP` / `FPS_LOG` /
  `GPU_MIP_SCAN_DIST` / `SPRITE_GRID`.
- **`roxlap-cave-demo` migrated onto the `SceneRenderer` facade**
  (demo-only). The cave world — exactly one scene chunk
  (`128 × 128 × 256`) — is now a single-grid, single-chunk
  `roxlap_scene::Scene` (identity transform, chunk `(0, 0, 0)`,
  materialised via `CaveChunkGenerator`). The demo renders through
  `SceneRenderer::{render, present}` instead of the hand-rolled
  `softbuffer` + `render_dda_parallel` loop, so it gains the GPU backend
  for free: run with `ROXLAP_GPU=1` (automatic CPU fallback). Carves +
  relights go through `Grid::set_sphere_with_colfunc` +
  `bake_lightmode`; collision reads the cave chunk via `getcube`;
  plasma bullets are now voxel-sphere sprites driven by the dynamic
  sprite API (`add_sprite_model` / `add_sprite_instance_posed` /
  `set_sprite_instance_transforms` / `remove_sprite_instance`). The
  per-chunk generator seed (`FNV(base_seed, (0,0,0))`) makes the cave
  equivalent but not byte-identical to the prior direct-seed output.
  Drops the `softbuffer` dependency; adds `roxlap-scene` + `roxlap-render`.

## [0.14.0] — 2026-06-25

This release lands the **DDA** macro-stage (DDA.0–DDA.10): the CPU
renderer is now a **clean-room per-pixel 3D-DDA over an 8³ brickmap**
that replaces voxlap's opticast outright — fixing the long-standing
voxlap-inherent artifact classes (silhouette notch, floor hairlines,
axis-aligned mip beams) — and the last voxlap-derived engine code is
excised, so the **Ken-Silverman commercial-use caveat is dropped: roxlap
is now MIT OR Apache-2.0 with no third-party restriction** (free for
commercial use). On top of that, an additive **dynamic sprite-model +
per-instance-transform API on `SceneRenderer`** lets a physics-driven
caller stream sprite models in/out and orient instances per frame without
bypassing to `GpuRenderer`. Breaking vs 0.13.0: `render_scene` /
`render_scene_composed` take a `CpuFog` value instead of `&mut
ScratchPool`, and `FrameParams::sprite_lighting` is replaced by a plain
`draw_sprites: bool`.

### Added

- **Clean-room CPU renderer** `roxlap_core::dda`: per-pixel 3D-DDA over
  an 8³ brickmap. Public API: `render_dda`, `render_dda_parallel`
  (rayon tile bands), `DdaEnv` (sky / fog / side-shades), `BrickCache`
  (cross-frame occupancy cache), `RasterSink`, `pixel_ray`,
  `effective_mip`.
- **Clean-room KV6 sprite raycaster** `roxlap_core::dda_sprite::draw_sprite_dda`
  — per-pixel ray cast through a sprite's KV6, depth-composited against
  the shared z-buffer.
- `roxlap_core::raster_target::RasterTarget` — the framebuffer/z-buffer
  compositing primitive shared by the DDA terrain + sprite passes.
- `roxlap_scene::render::CpuFog { color, max_scan_dist, side_shades }`
  — the CPU fog/side-shade config passed into the scene render entry
  points.
- **Dynamic sprite-model + per-instance-transform API on `SceneRenderer`**
  (purely additive — no existing signature changed): stream unique
  procedural sprite models in and out and orient placed instances per
  frame entirely through the facade, without dropping to `GpuRenderer`.
  New items in `roxlap_render`:
  - `SceneRenderer::add_sprite_model(&Kv6) -> SpriteModelId` — register
    one model incrementally (GPU appends an LOD chain; works before any
    `set_sprites`).
  - `SceneRenderer::remove_sprite_model(SpriteModelId) -> bool` — free a
    model's voxel data in place (ids never reused, so other handles stay
    valid); `false` on a stale handle.
  - `SceneRenderer::compact_sprite_models()` — reclaim the GPU buffer
    holes left by removed models (no-op on the CPU backend).
  - `SceneRenderer::add_sprite_instance_posed(SpriteModelId, DynSpriteTransform)`
    — spawn an instance already oriented (no one-frame axis-aligned
    flash).
  - `SceneRenderer::set_sprite_instance_transform(SpriteInstanceId, DynSpriteTransform)`
    and the batched `set_sprite_instance_transforms(&[(…, …)])` — update
    placed instances' position + orientation per frame (the GPU backend
    coalesces a frame's updates into a single buffer upload).
  - `DynSpriteTransform { pos, right, up, forward }` — the per-instance
    pose (model→world basis columns; `det ≠ 0`, identity by default; a
    degenerate basis silently skips the instance).
  - `SpriteModelId` is now a generational handle (`{ slot, gen }`,
    fields private — externally unchanged); a removed model's handle
    resolves to nothing → safe no-op.
- `roxlap_gpu::sprite_model`: `SpriteModel::empty()` and
  `SpriteModelRegistry::{remove, is_live}` — in-place free of a model
  chain's voxel data, preserving ids (no remap), backing the facade's
  `remove_sprite_model`.

### Changed

- **DDA is the default and only CPU renderer.** It fixes the
  long-standing voxlap-inherent artifact classes (silhouette notch,
  floor hairlines, axis-aligned mip beams). `render_scene` /
  `render_scene_composed` take a `CpuFog` value where they previously
  took `&mut ScratchPool`.
- `FrameParams::sprite_lighting: Option<&SpriteLighting>` → a plain
  `draw_sprites: bool` (both backends now draw sprites flat-lit; the GPU
  sprite pass uses its identity colour table to match the CPU path).
- **Engine math reimplemented independently** for a clean license:
  lighting bake (`world_lighting`), camera projection (`camera_math`),
  and the KFA bone solver (`kfa_draw`) rewritten from first principles;
  the `.vxl` slab editor (`roxlap-formats::edit`) de-ported (independent
  expression, byte-compatible with the format — existing assets and
  outputs are unchanged).
- README / crate descriptions reframed: roxlap is an independent engine
  that interoperates with Voxlap's file formats, not a line-by-line port.

### Removed

- **Breaking:** the voxlap renderer and its public API are gone from
  `roxlap-core` (~14k LOC): `opticast()`, `OpticastOutcome`,
  `rasterizer::ScratchPool`, `scalar_rasterizer::ScalarRasterizer`,
  `sprite::{draw_sprite, DrawTarget, SpriteLighting, sprite_colmul}`,
  `kfa_draw::draw_kfa_sprite`, and the `grouscan` / `scan_loops` /
  `drawtile` / `gline` / `column_walk` / `opticast_prelude` /
  `projection` / `ray_step` / `ptfaces16` modules. `OpticastSettings`
  is kept (now just projection + scan-distance settings).
  `solve_kfa_limbs` is kept (pose the limbs, then draw via
  `draw_sprite_dda`).
- `roxlap_core::meltsphere` removed (unused voxlap-derived sphere-carve).
- The `roxlap-oracle` crate (voxlap-C render-hash diff harness) removed.

### License

- **Dropped the Ken-Silverman commercial-use caveat.** roxlap is now an
  independent implementation containing no Voxlap C source; dual
  MIT/Apache-2.0 now applies to commercial use as well. The crates still
  interoperate with Voxlap's on-disk formats (`.vxl` / `.kv6` / `.kvx` /
  `.kfa`), which are not subject to copyright.

## [0.13.0] — 2026-06-22

This release lands two larger threads — a full **rigged-character pipeline**
(an on-disk `.rkc` container plus a TRS/quaternion bone rig replacing the
legacy single-angle hinge) and an **incremental GPU sprite path** (stream
models and instances in and out without rebuilding the resident registry) —
alongside a horizontal scene-flip primitive, image-sprite alpha cutoff +
picking, and a handful of render/GPU correctness fixes. One small breaking
change: `OpticastSettings`/`ScanContext.anginc` is now `f32`.

### Added

- **`.rkc` rigged-character container (`roxlap-formats::character`).** The
  on-disk form of a whole animated voxel character — `RKCH` magic, chunked
  `META`/`MSHS`/`BONS`/`CLPS`, reusing kv6 mesh blobs and the kfa
  `Hinge`/`Seq` layout. Forward-compat by construction: unknown top-level
  chunks and unknown clip kinds are preserved verbatim on resave, and typed
  `mesh_kind`/clip-kind discriminants leave room for voxel-video.
  `to_kfa_sprite` builds a renderable `KfaSprite`. `scene-demo`'s `build_kfa`
  authors a `Character` and round-trips it through serialize/parse,
  dogfooding the format end to end.
- **`.rkc` disk loader + lossy `.kfa` export.** `Character::to_kfa(clip,
  kv6_name)` writes a voxlap-toolchain `.kfa` (skeleton + one clip + a single
  kv6 filename), the interop writer scoped out of the format itself.
  `scene-demo` gains a runtime source selector mirroring how a host loads
  characters: `ROXLAP_RKC=<path>` loads + parses an `.rkc` from disk (falling
  back to the built-in character on failure), `ROXLAP_RKC_DUMP` /
  `ROXLAP_KFA_DUMP` write the authored character as `.rkc` or the lossy `.kfa`.
- **TRS bone rig + quaternion math (`roxlap-formats::xform`).** A `Quat`
  (axis-angle, normalize, rotate, `Mul`, `nlerp`, `from_euler`/`to_euler`
  intrinsic ZYX, `from_basis` via Shepperd's method, `conjugate`) and a
  `BoneXform { t, r, s }` local transform with `blend`,
  `from_hinge_angle`/`hinge_angle`. `setlimb` now composes TRS (rotate the
  velcro frame by the quaternion, offset the child anchor, scale the basis),
  with the math split into a pure `limb_xform` helper. Round-trip tested.
- **Incremental GPU sprite lifecycle (`roxlap-gpu`).** Stream sprites in and
  out without the full volume + buffer rebuild of `set_sprite_instances`:
  `append_sprite_instances`/`remove_sprite_instance`/`sprite_instance_count`
  (amortised O(1) push + power-of-two grow / O(1) swap_remove),
  `add_sprite_model` (amortised O(new model voxels) LOD-chain upload),
  `remove_sprite_model` + `compact_sprite_models` (tombstone + free-list reuse
  + buffer compaction, all without remapping caller ids), and
  `dead_sprite_model_count` as the fragmentation signal. Device-backed tests
  readback-verify free-list reuse, grow boundaries, and post-compaction
  offsets.
- **Unified incremental sprite-instance API (`SceneRenderer::{add,remove}_sprite_instance`).**
  Both backends get a cheap add/remove path with stable `SpriteInstanceId`
  handles (a gen-guarded slotmap absorbs the backends' swap-remove indexing).
  The `scene-demo` gains a streamed spinner (a rotating ring of colour-sphere
  sprites added/removed each frame) exercising the path on whichever backend
  is active.
- **Horizontal scene flip (`SceneRenderer::set_flip_x`).** Mirrors the scene
  in X right before display on both backends (CPU framebuffer flip; GPU
  `scene_blit.wgsl` X-mirror gated by a per-frame uniform flag), so a host can
  correct a left-handed render while the egui overlay stays upright. Line and
  image overlays mirror their NDC X and depth lookup to match.
- **Image-sprite alpha cutoff + screen→sprite picking.** `ImageSprite::alpha_cutoff`
  discards (does not blend) texels below the threshold on both backends, for
  crisp pixel-art edges; `SceneRenderer::pick_image` resolves the nearest image
  sprite under a pixel to its hit `uv`/texel (`ImagePickHit`), alpha-aware
  (transparent texels see through) and occlusion-aware (depth-tested sprites
  behind geometry are rejected). Geometry factored into the unit-tested
  `ray_quad_uv`; the GPU backend keeps a CPU shadow copy of each upload.

### Changed

- **`OpticastSettings`/`ScanContext.anginc` is now `f32`** (was `i32`).
  `anginc < 1` supersamples the angular ray fan (more ray planes), masking the
  thin-geometry silhouette holes the grouscan marcher drops on isolated voxels
  — a density knob, not a fix. `anginc = 1.0` is byte-identical to the old
  integer `1`, so all goldens are unchanged.
- **`.kfa` runtime poser now carries full TRS.** `KfaSprite::{kfaval, frmval}`,
  `set_animation`, and the `.rkc` clip store (`ClipData::Skeletal::frmval`,
  container `VERSION` 2) hold `BoneXform` instead of one Q15 hinge angle per
  bone; `animsprite` Phase 3 blends with `BoneXform::blend` (lerp
  translation/scale, nlerp rotation). Keyframes pose identically; in-between
  interpolation moves from linear-angle to nlerp (exact at the midpoint for
  same-axis rotations). The lossy `.kfa` export collapses each frame to its
  hinge angle about the bone axis.

### Fixed

- **GPU streaming no longer freezes on a batch of newly-streamed chunks.**
  `refresh_dirty` now caps per-frame chunk installs to a budget (default 4,
  `ROXLAP_GPU_CHUNK_BUDGET`, 0 = unbounded); leftover dirty chunks ride
  subsequent frames. Evictions stay unbounded.
- **GPU side-shade at a chunk-boundary-flush surface.** `march_grid` now seeds
  `hit_axis` from the face the ray crossed to enter each chunk, so a voxel
  solid at the chunk-entry point gets its real face normal instead of the
  hardcoded z pair (which split the surface bright/dark at the horizon).
- **GPU depth test for mirrored lines/images.** `line.wgsl`/`image.wgsl` now
  mirror their depth-buffer lookup back to `width-1-px` under `flip_x`, so
  overlays depth-test against the correct column instead of the mirror
  position.
- **`grouscan::run_phases` degenerate-cf hang.** An unconditional
  `MAX_PHASE_STEPS` (100M) cap converts a degenerate slab-split spin into a
  bounded, loudly-reported bail-out instead of a silent CI hang. The bound
  sits far above real rendering and the synthetic-prologue fixtures.

## [0.12.0] — 2026-06-16

Two additive features, no breaking changes. The renderer gains a
world-placed 2D **image-sprite** primitive — a flat RGBA texture drawn as
a depth-composited quad in world space — and `roxlap-core` gains canonical
right-handed `Camera` constructors so hosts stop hand-rolling (and
mis-handing) the camera basis.

### Added

- **World-placed 2D image sprites (`SceneRenderer::draw_images`).** A
  renderer primitive that draws an RGBA texture as a flat quad positioned
  in world space, composited with the scene depth buffer so voxel geometry
  occludes it correctly (not a screen-space overlay, not a voxel slab).
  Follows the `draw_lines` lineage: upload a texture once
  (`upload_image` → `ImageId`, released by `drop_image`), then draw it
  per frame between `render` and `present`/`paint_egui` from an
  `ImageSprite { image, origin, facing, size, tint, depth_test,
  double_sided }`, where `ImageFacing` is either world-fixed
  (`World { u, v }`) or camera-facing (`Billboard { up }`). `origin` is
  the top-left corner; `size` scales `u`/`v` (1 texel = 1 voxel for traced
  pixel-art). UVs are perspective-correct on both backends (CPU: a
  near-clipped textured-triangle rasteriser into the same framebuffer/
  z-buffer; GPU: `image.wgsl`, re-homogenised quads + a manual depth test
  against the scene-DDA `best_t`, nearest sampling, straight-alpha
  over-blend). Depth-tested sprites are occluded with a bias to avoid
  z-fighting on a coincident face; `double_sided: false` back-face-culls
  world quads. Each sprite carries an `alpha_cutoff` (texels below it are
  discarded outright, not blended — crisp pixel-art edges, and the same
  threshold defines pick-solidity). The first consumer is the demiurg
  voxel editor's reference overlay; the `scene-demo` `I` hotkey toggles a
  demo reference quad.
- **`SceneRenderer::pick_image` (screen→sprite picking).** The nearest
  image sprite under a window pixel, resolving which `uv` / source texel
  was hit (`ImagePickHit`). Transparent texels (below `alpha_cutoff`) are
  see-through — the pick passes through them to a sprite behind — and a
  `depth_test` sprite occluded by nearer scene geometry is rejected
  (shares `pick`'s depth convention). Lets an editor eyedrop a colour off
  the reference or snap placement to its pixel grid; the GPU backend keeps
  a CPU shadow copy of each upload for the alpha test.
- **`SceneRenderer::project_point` (`world → screen`).** The backend-correct
  inverse of `view_ray`: projects a world point to window pixels under the
  last frame's projection (CPU `setcamera` `hx/hy/hz`, GPU vertical-FOV
  pinhole), so hosts never reconstruct it themselves.
- **Canonical camera constructors `Camera::from_yaw_pitch`,
  `Camera::orbit`, and `Camera::look_at`.** All three build the
  right-handed `(right, down, forward)` basis the engine actually
  renders with (`right × down = +forward`), reproducing
  `oracle.c::set_camera_yaw_pitch`. Projects previously hand-rolled
  `right = [-sin yaw, cos yaw, 0]`; copying that form by hand is the
  usual source of a left-handed basis, which silently makes the sprite
  frustum cull reject every sprite. New crate-level docs explain the
  z-down world's inherent horizontal mirror versus a real camera and
  why an un-mirrored world must be handled consumer-side (mirror a
  world axis or negate yaw) rather than by flipping `right`. `Camera`'s
  placeholder `Default` is now documented as left-handed and not for
  interactive use.

## [0.11.0] — 2026-06-15

Multi-grid scenes get cheaper on both backends. The GPU renderer's
16-grid-per-scene cap is gone (per-grid cameras moved to a runtime-sized
storage buffer), and the CPU compositor now scissors each grid to its
projected screen footprint instead of paying a full frame per grid. The
only breaking change is the removal of the now-meaningless
`roxlap_gpu::MAX_SCENE_GRIDS` constant.

### Performance

- **Per-grid screen scissor for the CPU multi-grid compositor.**
  `render_scene_composed` previously rendered every in-range grid as a
  near-full-frame opticast plus several full-screen memory passes (temp
  reset, sentinel sweep, compose), so a scene of N grids cost ≈ N full
  frames regardless of how little of the screen each grid covered. Each
  grid is now projected to a conservative screen rectangle and (a) skipped
  outright when it falls off-screen on either axis, (b) opticast-clipped to
  its vertical band via the existing `y_start`/`y_end` strip path, and (c)
  has its temp reset / sentinel sweep / compose restricted to that band.
  Byte-identical to the old output (a new test renders a scene with the
  scissor on and off and asserts the framebuffer matches). The horizontal
  opticast *march* is **not** clipped — the radar's column-indexed
  `angstart` table isn't reset per grid, so column-clipping reads stale
  entries and crashes at extreme poses; that remains future work.

### Changed

- **Lifted the 16-grid-per-scene cap on the GPU renderer.** The shader's
  per-grid cameras moved out of a fixed `array<…, 16>` uniform field into
  a runtime-sized storage buffer (`scene_dda.wgsl` binding 15), so a scene
  can now hold any number of grids — the only ceiling is the device's
  storage limits (per-grid metadata + voxel data were already unbounded
  storage arrays, and the CPU path never had a cap). Removes the
  `MAX_SCENE_GRIDS` constant, the `GpuSceneResident::upload` /
  `render_scene` grid-count asserts, and the facade's grid-skip cap.
  Per-pixel cost stays linear in the *uploaded* grid count (no cross-grid
  culling on the GPU), so a host with many grids should still upload only
  the visible ones. **Breaking:** `roxlap_gpu::MAX_SCENE_GRIDS` is gone.

## [0.10.0] — 2026-06-15

A small, additive release: editing a single registered sprite model no
longer re-uploads the whole sprite field. The renderer keeps a slack-backed
suballocator for each model's variable-length colour/normal data, so a
carve or recolour touches only that one model's GPU bytes — and the edit
API is keyed by a stable `SpriteModelId` handle rather than a positional
index. No published crate's existing call breaks: `set_sprites` simply
gains a return value (handles) and `Kv6` is newly re-exported from
`roxlap-render`.

### Added

- **Incremental single-sprite refresh — `SceneRenderer::refresh_sprite_model`.**
  Editing one sprite model's geometry (a carve or recolour) no longer
  re-uploads the **whole** sprite registry. `set_sprites` now returns a
  `Vec<SpriteModelId>` — one stable handle per model — so callers never
  track the positional index themselves; `refresh_sprite_model(id, &Kv6)`
  brings that one model's stored geometry up to date after a content edit,
  leaving every other model and the entire instance set untouched (an edit
  never moves or adds an instance). It is a **backend-agnostic content
  refresh**, not a GPU upload: on the GPU backend the model's `colors` /
  `dirs` arrays are written through a slack-backed **suballocator** — in
  place when they fit, relocated (with a `model_meta` rewrite) when a carve
  grows the surface-voxel count past the slot's slack, and only on a
  buffer-tail overflow are the colour/dir buffers grown + the registry
  repacked — while the dims-fixed `occupancy` / `color_offsets` arrays are
  always written in place; on the CPU backend it swaps the edited `kv6`
  into each instance of the model. The `G`-carve hotkey and the scene-demo
  shoot-to-carve both move onto this path, so a shot re-uploads ~one
  model's bytes instead of the full 256-instance field. The sprite
  registry's GPU buffers are now `STORAGE | COPY_DST` with over-allocation
  to back the in-place writes. `set_sprites` remains the bulk/setup path
  for adding/removing models or changing instances. `Kv6` is now
  re-exported from `roxlap-render`.

## [0.9.0] — 2026-06-15

The GPU renderer reaches the browser. `roxlap-gpu` and the `roxlap-render`
facade now compile and run on `wasm32-unknown-unknown` (WebGPU), and both
web demos — `roxlap-web` and `roxlap-cave-web` — are rebuilt **on top of
`roxlap-render`** instead of calling `roxlap-core` opticast directly. In a
WebGPU browser they render via the wgpu compute marcher; elsewhere the
facade falls back to the CPU opticast path, now presented through a WebGL2
blit owned by the facade. This closes the last *architectural* gap from
the GPU and scene-graph roadmaps; the release stays **0.9.x** rather than
1.0 because engine limitations remain open (e.g. VC.6 mip-N multi-chz,
the deep-mip axis-aligned beam mitigation) and the API is not yet frozen.
No public API of the existing native crates is broken; the change is
additive (new wasm constructors + one new `Grid` method). This release
also lands three editor (demiurg) handoffs — kv6 surface normals, a
dense-model → `.vxl` export path, and depth-tested overlay lines.

### Added

- **Depth-tested 3D line drawing — `SceneRenderer::draw_lines`.** A
  world-space line-overlay pass on the `roxlap-render` facade, drawn
  **between `render` and `present`/`paint_egui`** and dispatched
  per-backend, for editor gizmos / debug geometry (bounding boxes, floor
  grids, origin axes, paths). `draw_lines(camera, &[Line3])` projects each
  segment with the frame's own projection and **depth-tests it against the
  frame's z-buffer**, so rendered geometry occludes lines behind it.
  `Line3 { a, b: [f64;3], color: u32 /*0xAARRGGBB*/, width_px: f32,
  depth_test: bool }` — alpha-blended by the colour's high byte, screen-
  space thickness via `width_px`, and `depth_test: false` for always-on-
  top overlays (e.g. a hover highlight). Both backends honour the same
  semantics in **their own depth metric** (CPU: perpendicular distance;
  GPU: euclidean `best_t`): the CPU backend rasterises the segments into
  the framebuffer with a perspective-correct depth interpolation, while
  the GPU backend expands them to screen-space quads (`line.wgsl`) and
  composites a `LoadOp::Load` pass that samples the scene-DDA depth
  buffer. The GPU scene pass now **always writes depth** (was gated on
  sprites being present), which also makes `pick_depth` work on a
  sprite-less GPU frame. egui overlay ordering is unaffected — the lines
  land in the framebuffer, so `paint_egui` still draws panels on top.
- **`Vxl::from_dense` + `Vxl::empty` — one-call dense-model → `.vxl`
  export.** `Vxl::from_dense(vsid, |x,y,z| -> Option<u32>)` builds a world
  from a dense occupancy + `0x80RRGGBB` colour closure (z-down,
  `z ∈ [0,256)`), and `Vxl::empty(vsid)` is the blessed all-air
  constructor — both seed voxlap's `loadnul` slab shape and edit it, so
  callers never hand-roll the slab format. `from_dense(..)` +
  `vxl::serialize(..)` is now the whole export path. Relatedly,
  **`vxl::serialize` now round-trips a post-edit `Vxl`**: it rebuilds the
  column data in column-index order (was raw-dumping `data`, which is
  wrong once `voxalloc` scatters columns), so any edited world — not just
  a freshly-parsed one — serialises to a valid `.vxl`. Byte-identical for
  unedited worlds.
- **kv6 per-voxel shading + a public `normal → dir` quantiser.**
  `Kv6::from_fn_shaded` builds a model with real surface normals
  (`Voxel::dir`) and exposed-face masks (`Voxel::vis`) instead of the
  flat `dir = 0`, `vis = 63` of `from_fn`, so procedurally-authored
  models shade with a directional gradient on the CPU sprite path (which
  reads `kv6colmul[dir]`) like an authored `.kv6`. `Kv6::recompute_surface`
  refreshes `vis`/`dir` after editing a model's voxels. The voxlap
  `univec[256]` direction table moved to `roxlap-formats`
  (`roxlap_formats::equivec`, re-exported from `roxlap-core`) so kv6 can
  reach it without a circular dependency, and gained
  `equivec::nearest_dir(n) -> u8`. The `vis` face-bit order is calibrated
  against `coco.kv6`'s authored bytes (all six faces).
- **`roxlap-gpu` + `roxlap-render` build for `wasm32` (WebGPU).** New async,
  canvas-based constructors — `GpuRenderer::new_from_canvas` and
  `SceneRenderer::new_from_canvas_async` — create the wgpu surface from an
  `HtmlCanvasElement` via `SurfaceTarget::Canvas`. The browser drives the
  adapter/device futures through its event loop (no `pollster`). The
  generic window-handle constructors stay the native path; the wasm
  constructors drop the `Send + Sync` bound (wgpu types are `!Send` on the
  `+atomics` shared-memory build, and the browser host is single-threaded).
- **CPU fallback presents on the web.** The facade's CPU backend, which
  uses `softbuffer` on native, presents its composited framebuffer through
  a WebGL2 texture-blit (`cpu_blit.rs`) on wasm. So a browser without
  WebGPU still renders, via CPU opticast — same `SceneRenderer`, same API.
- **Graphics stack upgraded to wgpu 29** (from 22) + naga 29 + egui/
  egui-wgpu 0.34. wgpu 22 unconditionally sent the WebGPU-spec-removed
  `maxInterStageShaderComponents` device limit, which current Chrome/Dawn
  reject — so WebGPU device creation failed on modern browsers. wgpu 29
  tracks the current spec. The wasm GPU init also now probes the adapter
  **and** device before binding the canvas surface, so any GPU-init
  failure leaves the canvas pristine for the WebGL2 CPU fallback (no
  more "no webgl2 context" crash).
- **`Grid::bake_lightmode(lightmode)`** in `roxlap-scene` — bakes voxlap
  `estnorm`/`updatevxl` per-voxel lighting into every materialised chunk's
  brightness bytes, neighbour-aware at chunk-XY seams. A reusable engine
  API (previously a scene-demo-only helper); the cave web demo bakes with
  it after generation and after each carve.

### Changed

- **`roxlap-web`** renders a procedural terraced-hills world through the
  facade (WebGPU, CPU fallback). The bundled `oracle.vxl` parse + the
  hand-rolled WebGL2 blit are gone (the blit now lives in the facade); the
  resolved backend (WebGPU vs CPU) is logged to the console.
- **`roxlap-cave-web`** renders the procedural cave as a single-chunk
  `Scene` grid through the facade. Flying + per-voxel collision + runtime
  carving run against the scene; plasma bullets are now facade **sprites**
  (glowing voxel spheres) that carve a crater with a local lightmode-1
  re-bake on impact, which the GPU path re-uploads via per-chunk dirty
  tracking. `F`/`R` regenerate the cave in place.

## [0.8.0] — 2026-06-13

Ports voxlap's KFA animation-curve playback (`animsprite`) and brings the
GPU sprite path to parity with the CPU: animated KFA sprites, directional
normal-based sprite lighting, and sprite rendering in grid-less
(model-viewer) scenes, plus a grid-lighting hook on the render API.
**Breaking:** `FrameParams` gains a `side_shades: [i8; 6]` field (default
`[0; 6]`); `HeadlessSceneRenderer::new` takes a `&wgpu::Queue` (so it
uploads its default sky); `KfaSprite` gains public
`frmval`/`seq`/`kfatim`/`okfatim` fields and `SpriteModel` a public `dirs`
field (built by the existing constructors — only direct struct-literal
construction is affected).

### Added

- **Grid side-shade lighting hook (`FrameParams.side_shades`).** Plumbs
  voxlap's `setsideshades(top, bot, left, right, up, down)` through the
  render API — the grid-scan analogue of `sprite_lighting` — on **both**
  backends. The CPU rasteriser applies it via `gcsub`; the GPU scene-DDA
  pass darkens a hit voxel's brightness by the hit face's shade (the face
  from the DDA's last-stepped axis), reducing the alpha-brightness before
  the `/128` divide exactly like `grouscan_shade`. With a flat (un-baked)
  brightness it's pure runtime side-shading; with baked light it stacks,
  as voxlap does. Default `[0; 6]` keeps `sideshademode` off
  (byte-identical to before). A host passes its engine's `side_shades()`
  so the board shades by the same sun.

### Fixed

- **GPU sprite-only / empty-scene rendering.** Sprites now render on the
  GPU backend when the scene has no voxel grids (a pure model/sprite
  viewer). Previously the facade short-circuited a grid-less scene to a
  bare clear and never ran the sprite pass, so only the clear colour
  showed (CPU rendered the sprite fine — a backend asymmetry). The
  no-grids-but-sprites case now renders through a cached zero-grid
  resident so the scene pass fills the sky background + far depth and
  the sprite pass composites over it. Also fixed a latent degenerate
  sky direction: with zero grids the sky ray was `(0,0,1)`, whose
  `atan2(0,0)` panorama lookup sampled black — the scene shader now
  derives the sky direction from a dedicated world camera in the
  uniform (which also retires the "grid 0 is the world frame"
  assumption for the sky on rotated grids).

### Added

- **KFA animation-curve playback (`animsprite`).** Faithful port of
  voxlap's `animsprite` (`voxlap5.c:11125`): `KfaSprite::set_animation`
  attaches the parsed `frmval` + `seq` tables and
  `KfaSprite::animsprite(dt_ms)` advances the time cursor (honouring
  `!target` loop/jump entries) and recomputes `kfaval[]` by wrap-aware,
  fixed-point keyframe interpolation. Previously the host had to drive
  `kfaval[]` by hand; baked `.kfa` curves now play back.
- **KFA sprites on the GPU backend.** `SceneRenderer::set_kfa_sprites`
  registers each limb's KV6 as an instanced model once, and
  `update_kfa_poses` re-poses them every frame — GPU via a cheap
  transform-only instance-buffer update (no model-volume re-upload,
  `GpuRenderer::update_sprite_instance_transforms`), CPU by re-solving
  limb transforms. Bone posing was factored out of `draw_kfa_sprite`
  into the reusable `roxlap_core::kfa_draw::solve_kfa_limbs`. The
  scene-demo shows an `animsprite`-driven swinging arm on both backends.
- **Directional sprite lighting on the GPU backend.** The GPU sprite
  pass now shades KV6 voxels by surface normal, matching the CPU
  rasteriser instead of rendering flat colours. Each voxel's `dir` is
  uploaded alongside its colour, and the renderer builds voxlap's
  `kv6colmul[256]` table per instance via the new
  `roxlap_core::sprite::sprite_colmul` (reusing the exact
  `update_reflects` math); the shader applies the same per-channel
  `mulhi`/saturate modulation. Tables are rebuilt each frame so rotating
  KFA limbs re-shade correctly. A naga-based unit test statically
  validates every WGSL shader.

## [0.7.0] — 2026-06-12

Decouples the renderer from winit (binds to any `raw-window-handle`
provider), adds an **egui overlay seam** on both backends, and ships an
SDL2 host demo. Breaking: `SceneRenderer::new` is generic + takes a size,
and `render` no longer presents (add a `present()` / `paint_egui()`
call). GPU sprite-camera + sky/fog parity bugs fixed along the way.

### Added

- **egui overlay seam (`hud` feature).** A renderer-level path to draw an
  egui UI (HUD, debug panels, menus) on top of the rendered frame, on
  **both** backends. `SceneRenderer::paint_egui(jobs, textures,
  pixels_per_point)` takes the host's tessellated egui output and:
  - GPU — paints via `egui-wgpu` (`LoadOp::Load` over the marcher's
    frame);
  - CPU — software-rasterises the tessellation (textured triangles,
    font/image atlas, premultiplied src-over) into the framebuffer.

  The host runs egui itself (e.g. `egui` + `egui-winit`); roxlap consumes
  only `egui::ClippedPrimitive`s + the `TexturesDelta`. `roxlap-render`
  re-exports `egui` (under `hud`) so the host builds against the exact
  version. Pin: `egui`/`egui-wgpu` `0.29` (→ wgpu 22), `egui-winit`
  `0.29` (→ winit 0.30). `roxlap-scene-demo` shows a live FPS/pose HUD
  (toggle with `F1`).

- **`roxlap-sdl-demo`** — an SDL2 host demo (WASD + mouse-look fly
  camera over a small voxel scene) that drives the exact same
  `SceneRenderer` as the winit `roxlap-scene-demo`, proving the windowing
  decoupling end-to-end. Includes a `Send + Sync` raw-handle adapter
  pattern for window providers (like SDL) whose window type is
  `!Send`/`!Sync`. `nix develop` now provides `SDL2`.

### Changed

- **`render` no longer presents — split into `render` + `present`.** To
  let a host slot a UI pass between the world and the swap, `SceneRenderer::
  render` now *composites without presenting* and the frame is finished
  by exactly one of `present` (no overlay) or `paint_egui` (egui
  overlay). The CPU backend composites into an owned framebuffer; the GPU
  backend acquires-but-defers the swapchain frame (`GpuRenderer` gained
  `render_clear_deferred` + `present`; `render_scene` no longer presents).
  **Migration:** add a `renderer.present()` call after each
  `renderer.render(...)` (or use `paint_egui`). Hosts that don't draw a
  UI need only the one extra call.

- **`roxlap-render` / `roxlap-gpu` decoupled from winit.** The window
  binding was nominal — both backends only ever needed the window's
  `raw-window-handle` traits (softbuffer + wgpu) plus its pixel size.
  `SceneRenderer::new` and `GpuRenderer::new` / `new_blocking` are now
  generic over any `W: HasWindowHandle + HasDisplayHandle + Send + Sync
  + 'static` (winit, SDL, GLFW, a custom surface, …) and take the
  initial framebuffer size as an explicit `(u32, u32)` argument instead
  of calling `winit`'s `inner_size()`. The CPU backend tracks its size
  from `resize()` (which hosts already call on resize events) rather
  than polling the window each frame. `roxlap-render` re-exports
  `HasWindowHandle` / `HasDisplayHandle` so hosts need no direct
  `raw-window-handle` dependency. winit moves to a dev-dependency
  (examples/doctests only). **Migration:** pass the window size to
  `new`, e.g. `SceneRenderer::new(window, (w, h), &opts)`. Removed the
  unused `GpuRenderer::window()` getter.

### Fixed

- **GPU instanced sprites projected through the wrong camera.** The GPU
  sprite pass (frustum-cull/tile-bin + model-DDA) used `cameras[0]` —
  grid 0's *local* camera — to project world-space sprite instances. It
  only looked correct when grid 0 sat at identity; any non-identity
  transform on the first grid shifted every sprite by that grid's
  origin/rotation. `render_scene` now takes an explicit world
  `sprite_camera` (the facade passes the untransformed world camera) and
  both sprite sites use it. The CPU path was already correct (it draws
  sprites with the world camera directly). Also drops the sprite pass's
  `!cameras.is_empty()` guard, so sprites no longer require any grid.

- **GPU sky / fog parity with the CPU path.** The GPU backend ignored
  `FrameParams::{sky_color, fog_color, fog_max_scan_dist}` — it sampled
  its own default grey sky texture and never applied distance fog, so it
  diverged from the CPU path (grey vs. flat-colour sky, no fog vs. fog).
  `SceneRenderer::render` now mirrors `sky_color` onto a 1×1 GPU sky
  texture each frame (unless the host uploaded a real panorama via
  `set_sky_panorama`) and forwards `fog_color` / `fog_max_scan_dist` to
  the GPU marcher. (GPU fog is a smoothstep where the CPU LUT is linear
  — same endpoints, slightly different mid-curve.)

## [0.6.1] — 2026-06-10

### Changed

- **README rewrite to current scope.** The workspace README (the
  crates.io / docs.rs pitch page for every published crate, via
  `readme.workspace`) was stuck around the R10/R11 CPU-only era. It now
  covers the `roxlap-scene` scene graph (multi-grid f64 world +
  rotation, streaming, snapshots, world queries), the optional
  `roxlap-gpu` compute renderer, and the unified `roxlap-render` facade
  with screen→world picking — drops the now-inaccurate "no GPU" framing,
  adds the four missing crates to the table and `roxlap-scene-demo` to
  the quick-start, and refreshes Status (published; S1–S7 + GPU.0–13 +
  picking landed). Docs only; no code or API changes.

## [0.6.0] — 2026-06-10

A GPU outer-DDA perf win (GPU.13.0) plus a **screen→world picking and
voxel-query API** for downstream engines (turning a click — or any
world ray — into a grid + voxel + colour). Purely additive; no
public-API breakage versus 0.5.0.

### Added

- **GPU.13.0 — chunk-AABB outer-DDA early-out.** `scene_dda.wgsl`'s
  outer chunk-DDA now stops the moment a ray leaves a grid's occupied
  chunk-AABB along its travel direction, instead of stepping empty
  space to `max_outer_steps`. Cuts the high-altitude / horizon
  overscan where sky and beyond-terrain rays cross many empty chunks.
  The AABB lives in `GridStaticMeta` (112→144 bytes) and is maintained
  live — `GpuSceneResident::refresh_chunk` / `evict_chunk` recompute
  and re-upload it so streamed-in terrain is never skipped. Demo gains
  an `H` hotkey toggling a high-altitude top-down vantage for the FPS
  A/B. Render output is byte-stable (the early-out only skips empty
  space). On a 3070, the user's flagged high-altitude pose went from
  the slow case to ~600 FPS.

#### Screen→world picking + voxel queries

- **`roxlap-render`** — `SceneRenderer::pick(scene, camera, x, y) ->
  Option<PickHit>` resolves a pixel to its world point + owning grid +
  grid-local voxel in one call. Supporting API: `pixel_ray`, the
  canonical `view_ray` (returns a `Ray` — the one unproject both
  backends honour), and `pick_depth` (per-pixel world-t; CPU reads its
  z-buffer, GPU stages the depth buffer at click time). New `Ray` and
  `PickHit` types. Each backend caches its last-frame projection so
  callers never reconstruct it.
- **`roxlap-scene`** — `Scene::raycast(origin, dir, max_dist) ->
  Option<RayHit>`: a renderer-independent voxel DDA across grids
  (per-grid local-space marching, transform-correct for rotated /
  translated grids; nearest hit wins) for line-of-sight, projectiles,
  and off-screen / backend-agnostic picking. `Scene::resolve_voxel`
  maps a world surface point to its grid + voxel. `Grid::voxel_solid`
  and `Grid::voxel_color` query a grid-local voxel. New `RayHit` type.
- **`roxlap-formats`** — `Vxl::voxel_color(x, y, z)` reads a textured
  voxel's packed colour straight from the slab chain (no decompress).
- **`roxlap-gpu`** — `GpuRenderer::{read_depth_pixel, pixel_ray}` +
  the standalone `pinhole_pixel_ray`; the scene depth buffer gained
  `COPY_SRC` + a `MAP_READ` staging buffer for readback.
- **`roxlap-scene-demo`** — `C` toggles a top-down pick mode: a cursor
  sprite follows the mouse on a ground plane, and left-click resolves
  and prints the true grid + grid-local voxel.

## [0.5.0] — 2026-06-09

The **GPU compute-shader renderer** arc (GPU.0–GPU.12) plus the
**`roxlap-render`** unified renderer facade. Two new published crates
(`roxlap-gpu`, `roxlap-render`); the existing crates are unchanged
versus 0.4.2 and are version-bumped only to keep the workspace
unified. No public-API breakage.

### Added

#### `roxlap-gpu` — GPU compute-shader renderer (new crate, first publish)

- A WGPU + WGSL compute-shader voxel renderer alongside the CPU
  opticast — "approximately the same retro look, much faster", freeing
  the CPU budget for game logic. Two-level Amanatides-Woo DDA (outer
  chunk grid + inner voxel), per-chunk occupancy/colour decompress +
  upload, multi-grid scene composition with per-grid f64→f32 camera
  transforms, panoramic sky + fog, per-chunk edit/streaming
  invalidation, and a KV6 sprite path (instanced model-DDA with CPU
  frustum cull, screen-tile binning, far-LOD mips, and structural
  runtime edits).
- **GPU.11 — scene-grid LOD.** Each chunk's full mip ladder is
  uploaded (a second occupancy/colour set per level) and the marcher
  picks a mip per chunk by entry distance. ~2.15× FPS at horizon views
  on an RTX 3070 Laptop (58→125); tunable via
  `ROXLAP_GPU_MIP_SCAN_DIST` (default 64, matching the CPU
  `mip_scan_dist`).
- Sibling to the CPU path, not a replacement: the byte-exact voxlap
  oracle stays CPU-only. Falls back gracefully when no WGPU adapter is
  available.

#### `roxlap-render` — unified CPU/GPU renderer facade (new crate, first publish)

- One `SceneRenderer` over the CPU opticast (presented via
  `softbuffer`) and the GPU compute marcher (presented via `wgpu`).
  Owns presentation, the `Scene`→GPU upload / dirty-chunk refresh /
  per-grid camera transform bridge, the CPU compositor + scratch
  pool, the sprite reps (CPU draw + GPU registry + carve), and
  framebuffer capture — and **falls back to the CPU backend
  automatically** when GPU init fails (the wasm/driver-gap path).
  Hosts pick a backend with one call and render with one method.

### Fixed

#### `roxlap-gpu` — bedrock-as-solid (opaque cliff/wall faces)

- Vertical wall and cliff faces rendered as sky holes (only the
  textured top voxel showed). The decompressor dropped the implicit
  voxlap bedrock interior below a surface to save memory; it is now
  marked solid in a **second occupancy bitmap** used for hit-testing,
  while colours stay textured-only — a bedrock hit inherits the
  surface colour above it. Bedrock costs one bit, not a colour, so the
  colour array is unchanged; the empty-chunk placeholder stays air so
  floating objects (the ship) don't grow a spurious floor plane.

## [0.4.2] — 2026-06-03

Axis-aligned-mip-beams resolution + `phase_remiporend`
multi-chz reload. The beam bug that motivated the 0.4.x
mitigation cascade is gone (incidentally fixed by the VC.5 /
VC.6.2 / PRR multi-chz install path); demo mitigations
reverted to the original aggressive config. `phase_remiporend`
closes the last VC.6 follow-up from 0.4.1's Known limits.
No public-API breakage vs 0.4.1.

### Fixed

#### `roxlap-core` + `roxlap-scene-demo` — AAMB (axis-aligned-mip-beams resolution)

- **The axis-aligned-mip-beams artifact is RESOLVED.** Originally
  reported 2026-05-12: faint world-axis green columns at deep
  mip-N + near-axis-aligned rays. Multi-session investigation
  CF.0..CF.3.C in `project_cf_narrowing_multi_session_plan.md`
  concluded that cf-narrowing could not fix it. Re-audit at
  2026-06-03 finds the bug GONE — every multi-chunk beam test
  reports **0 beam pixels** at `ml=6` across `msd=8/64/256/1024`.
  Likely fixed incidentally by the VC.5 / VC.6.2 / PRR multi-chz
  install cascade (`phase_after_delete_kept_presync`'s column-step
  and `phase_remiporend`'s reload both now route through
  `build_owned_column_multi_chz`, which appears to have closed
  the corner case the beam relied on). Confirmed:
  `dump_green_beam_pose_diff` (scan=1024, msd=64, ml=6 vs ml=1)
  reports 0 beam pixels (was 6404 at S5.3); `dump_spawn_pose_diff`
  reports 0 (was 5379).
- **Demo mitigations reverted**. `roxlap-scene-demo`:
  - `SCAN_DIST_MAX` 1500 → 1024.
  - `settings.mip_levels` 4 → 6, `settings.mip_scan_dist` 128 →
    64 (the live demo now exercises the full 6-mip ladder at
    msd=64).
  - Ship grid's `mip_levels_override = Some(1)` retired — ship
    now renders with the full mip ladder. Bench engine-only
    moves from 43 FPS (mitigated, NSP 68.9 %) to 75 FPS (revert,
    NSP 34.4 %) — coarser mips render less terrain faster at
    the bench camera.
- **`crates/roxlap-core/src/cf_narrow.rs` deleted** (~523 LOC) —
  the rejected experiment + 9 unit tests retired. The three env-
  var gates (`ROXLAP_CF_NARROW`, `ROXLAP_CF_NARROW_PER_COLUMN`,
  `ROXLAP_CF_NARROW_PER_COLUMN_NO_I1`) and their LazyLock caches
  also gone from `grouscan.rs`.
- **AAMB.1 single-chunk multi-mip crash fixed**. The audit
  surfaced a pre-existing arithmetic underflow at
  `grouscan.rs::phase_remiporend` where
  `state.ixy_sptr_col_idx - state.mip_base_offsets[old_mip]`
  panicked for axis-aligned poses on single-chunk grids at
  `msd ≥ 64, ml = 6` (negative `cy_mip` masking into a column
  index that landed just below the mip-OLD sub-table boundary).
  Fix: defensive bail to `Phase::Startsky` when the index slips
  below `mip_old_base`. `axis_aligned_single_chunk_multi_mip_*`
  and `axis_aligned_single_chunk_pitched_up` flip from PANIC to
  PASS; multi-chunk beam tests stay green.

#### `roxlap-core` — PRR (phase_remiporend multi-chz reload)

- **`phase_remiporend`'s post-mip-transition column reload**
  switched from single-chz install to the same multi-chz
  builder VC.6.2 already routes the column-step through.
  Closes the last VC.6 follow-up from 0.4.1's
  Known-limits list. At the camera's own chunk-XY past a mip
  transition, multi-chz scenes (`chunks_z > 1` + content at
  `chz != seed_chz`) need the stitched virtual column or the
  intermediate drawing phases (drawfwall / drawcwall / drawflor)
  read the camera-chunk's placeholder bedrock and bypass to
  sky before the next column-step's multi-chz install fires.
- VC.6.0 fixture (4×4×3 chunk grid, chz=0 camera looking
  down at chz=2 floor) reveals the effect: `total_red`
  104 511 → 105 442 (+931 voxel in-fills, +0.9 %). Visual
  shape unchanged (the trapezoidal floor projection); the
  fix in-fills voxels that were previously sky-bypassed.
- VC.6.2 regression pin re-anchored to
  `0xd8eb_5565_84f2_f30d` (was `0x577e_6879_b86e_f758`).
- User's `z=-19.44` hills demo still byte-stable at VC.5
  baseline `0x15e3_21a1_012a_6109` — the steep pitch
  terminates rays in the camera-chunk; mip transitions
  don't fire there.

## [0.4.1] — 2026-06-01

Mip-N multi-chz column-step fix + column-borrow perf cleanup.
Closes the engine-level mip-N gap that 0.4.0's VC.5 left open
(distant-XY rays at mip-N now stitch chz layers the same way
mip-0 already did), and reverts the structural per-column-step
memcpy cost the VC arc accidentally introduced at 0.2.0. No
public-API breakage vs 0.4.0.

### Performance

#### `roxlap-core` — CB (column borrow)

- **`state.column` switched to `ColumnSource<'a>` enum** —
  `Borrowed(&'a [u8])` for single-chz / N=1 installs (zero
  allocation, zero memcpy per column-step) and
  `Owned(Vec<u8>)` for multi-chz stitched chains. `Deref<Target
  = [u8]>` keeps every `state.column[i]` / `.len()` / `.get(i)`
  call site working unchanged.
- Reverts the structural perf cost the VC arc accidentally
  introduced at 0.2.0 (when `state.column` flipped from `&'a
  [u8]` to `Vec<u8>` to support multi-chz stitching). For
  chunks_z = 1 grids — including the demo's ground (vsid =
  4096) and ship (vsid = 768) — every column-step install now
  borrows back into the chunk's `slab_buf` instead of copying
  the chain bytes into an owned Vec.
- Install sites updated: seed install (`from_seed`), mip-0
  multi-chunk column-step's N=1 fast path, mip-N multi-chunk
  column-step (VC.6.2's swap), single-chunk single-chz column-
  step (both mip-0 and mip-N), and `phase_remiporend`'s
  mid-ray mip-transition reload. Multi-chz stitched installs
  stay on the `Owned` Vec path unchanged.
- Bench gain (4-thread, ROXLAP_STATIC=1, vsid=4096 ground +
  vsid=768 ship, max_scan_dist=512, mip_levels=4): engine-only
  best 52.1 FPS, mean ~46 FPS (vs 0.4.0 baseline 42.9 FPS = ~
  +10-20%; bench has ±5 FPS run-to-run noise). Lower than the
  ~25% structural-gap estimate in [[perf-recovery-landed]] —
  the Deref-through-enum match adds per-read overhead that
  partly offsets the install-time savings. Architectural
  cleanup stands regardless: `install_owned_column` is no
  longer called in production (kept as a chain-walker reference
  for tests).
- Byte-stable across the entire test corpus: VC.5 baseline
  hash `0x15e3_21a1_012a_6109`, VC.6.2 fix hash
  `0x577e_6879_b86e_f758`, oracle 10 MATCH + 2 pre-existing
  CPU divergence all unchanged.

### Fixed

#### `roxlap-core` — VC.6 (mip-N multi-chz column-step)

- **Mip-N multi-chz column-step install** (VC.6.0..VC.6.2).
  Closes the mip-N gap left open by 0.4.0's VC.5 (mip-0 multi-
  chz). `build_owned_column_multi_chz` and `emit_chunk_chain`
  both grew a `mip_level: u32` parameter. The mip-N column-step
  branch at `grouscan.rs::phase_after_delete_kept_presync` now
  calls the multi-chz builder with `mip_level = state.gmipcnt`,
  so distant-XY rays at mip-N stitch chz layers the same way
  mip-0 already did. Intermediate bedrock placeholders are
  stripped at the scaled sentinel `0xff >> mip_level`.

  Visible improvement (synthetic 4×4×3 grid, chz=0 camera looking
  down at a chz=2 floor, `mip_levels=4 + mip_scan_dist=64`):

  | Metric         | Pre-VC.6.2 | Post-VC.6.2 | Δ     |
  |----------------|-----------|-------------|-------|
  | bottom_half red pixels | 19 087 | 104 511 | 5.5× |

  Render shape: small camera-chunk-only red blob → full
  trapezoidal floor at perspective. New regression pin:
  `roxlap_scene_demo::vc6_repro::vc6_2_mip_n_multi_chz_*` at
  hash `0x577e_6879_b86e_f758`.

  User's `z=-19.44` hills demo byte-stable at VC.5 baseline
  (`0x15e3_21a1_012a_6109`) — the steep pitch terminates rays
  inside the camera's own chunk-XY footprint, so the mip-N
  column-step branch never fires at distant XY for that pose.
  New scenes with shallower-pitch cameras over stacked content
  benefit.

### Removed

- **`try_handoff_chunk_z_down`** removed (VC.6.3). The
  rasterizer's mid-render chunk-Z handoff was the
  pre-VC.5 mechanism for stacked-grid rendering; VC.5's mip-0 +
  VC.6.2's mip-N multi-chz install supersede it (every chz is
  pre-stitched into one virtual column at install time, with
  intermediate bedrocks stripped). Audit confirmed the helper
  was dead in all three reachable configurations (mip-0 multi-
  chunk, single-chunk, mip-N gated out). `phase_draw_flor`'s
  bedrock-as-air bypass now falls through to `Phase::AfterDelete`
  unconditionally when the sentinel is hit. Byte-stable: every
  existing render hash unchanged (VC.5 baseline + VC.6.2 fix
  hash both verified post-removal).

### Known limits

- **`phase_remiporend` single-chz reload** at mip transition
  mid-render. The VC.6.0 fixture pose doesn't expose this
  because the chz=2 floor is hit after rays have already
  crossed chunk-XY (column-step's multi-chz path took over).
  Other topologies — content at `chz != seed_chz` visible at
  the camera's own chunk-XY past a mip transition — could
  surface it. Deferred until a real scene demonstrates the
  gap.
- **Edge tearing at deep mip-N** at the back of the VC.6.0
  fixture render. Same area-of-code as the open
  `axis-aligned-mip-beams` artifact (cf-narrowing at remiporend).
- `opticast` `gylookup` overflow at `chunks_z ≥ 4` per grid —
  unchanged from 0.4.0.

## [0.4.0] — 2026-06-01

Virtual-Column-rewrite + perf-recovery release. Closes the S4B.6.j
cross-chunk look-down rendering limitation properly (no longer
mitigated; the live demo materialises the full chunk-z stack), and
restores demo perf to ~3× the pre-VC.5 floor at the spawn pose.

### Fixed

#### `roxlap-core` — Virtual Column rewrite (VC.0..VC.7)

- **S4B.6.j cross-chunk look-down rendering** is fixed at the
  engine level. The rasterizer's column-step now stitches a
  multi-chz virtual column at every chunk-XY crossing — distant
  XY columns see the correct world-z slabs instead of falling
  into the previous handoff's broken `+ chunk_size_z` arithmetic.
  `roxlap_scene::render`'s `stacked_chz0_distant_mountain_visible
  _from_chz0_camera` test (the sanctioned VC.0 fail pin) is now
  GREEN.
- New per-slab chain builders + multi-chz install path:
  `build_owned_column_from_chain`, `build_owned_column_multi_chz`,
  `emit_chunk_chain`. The seed-time path and the per-column-step
  install both route through these so distant columns inherit the
  same multi-chz stitching the camera column gets.
- New parallel `column_z_base: Vec<i32>` translation table —
  per-slab world-z lookup decoupled from the chunk-local slab
  bytes. Lets the rasterizer keep voxlap's u8 slab format while
  the engine reasons about world-z natively.
- Camera-above-stacked-grid: `camera_chunk_air_gap` clamped to
  `origin_chunk_z`; `gline` matches. Cameras above the grid no
  longer collapse to chz=0.

#### `roxlap-scene-demo` — VC.7 cleanup

- Reverted the `HillsChunkGenerator::should_generate` mitigation
  from 0.3.0. The live demo now materialises the full chz stack
  and exercises the engine-level multi-chz path. Repurposed
  `camera_above_hills_only_streams_chz0_chunks` →
  `camera_above_hills_streams_full_chz_stack` as a regression test
  for VC.5's behaviour.

### Performance

- **PR.1 — env-var caching** (`roxlap-core`): the 7
  `std::env::var{,_os}` reads in `grouscan.rs` + `cf_narrow.rs`
  (`ROXLAP_TRACE_PHASES`, `ROXLAP_TRACE_STARTSKY`,
  `ROXLAP_CF_NARROW`, `ROXLAP_CF_NARROW_PER_COLUMN`,
  `ROXLAP_CF_NARROW_PER_COLUMN_NO_I1`, `ROXLAP_CF_NARROW_NOP`)
  are now read exactly once per process via module-private
  `LazyLock<bool>` caches instead of per-call. `run_phases` alone
  fires per-ray (~480K calls / frame at vsid=4096), so the
  previous getenv chain accounted for ~29% of render-frame CPU
  time. Diagnostic flags are not expected to change at runtime,
  so behaviour is preserved.
- **PR.2 — per-grid bounding-sphere distance cull**
  (`roxlap-scene`): `render_scene_composed`'s Near/Mid arm now
  skips the per-grid opticast pass when the grid's bounding
  sphere is entirely beyond `OpticastSettings::max_scan_dist`.
  Each opticast walks ~width\*height rays even when none reach a
  voxel, so far-away grids (marker pillars, distant pickups)
  otherwise paid ~9 ms each per frame. Safe: no ray can reach a
  grid whose closest sphere point is past the scan distance.
- **PR.3 — single-chunk fast path**
  (`roxlap-scene::render_scene_composed`): grids that are exactly
  1 chunk at index `(0, 0, 0)` reuse the
  `GridView::from_single_vxl` view that `chunk_xyz_backing`
  already populated. The rasterizer then takes the single-chunk
  branch in `phase_after_delete_kept_presync` — no per-column-
  step `chunk_at_xyz` / IVec2 equality / `Option::is_some`.
  Bench-neutral in the scene-demo's spawn pose (only one
  qualifying grid is in range; the rest are culled by PR.2),
  architectural prep for callers adding small single-chunk grids.

Combined bench at the demo's spawn pose (4-thread, vsid=4096
ground + vsid=768 ship, max_scan_dist=512, mip_levels=4):

| Config                    | 0.3.0 | 0.4.0 | Δ     |
|---------------------------|-------|-------|-------|
| Live demo (markers on)    | 11.7  | 35.3  | +202% |
| Engine only (no markers)  | ~17   | 46.5  | +173% |

### Added

- `ROXLAP_NO_MARKERS=1` env knob in `roxlap-scene-demo` —
  bench-only switch to skip the 5 marker pillars when measuring
  the core renderer's cost in isolation. Live demo defaults
  unchanged.

### Known limits

- **VC.6 — mip-N multi-chz** is deferred. The mip-N column-step
  still uses single-chz install; demo poses A/B/C/D + the ship
  don't exercise multi-mip multi-chz, so no active regression.
- **opticast `gylookup` overflow** at `chunks_z ≥ 4` per grid
  (surfaced at S7.4, still deferred).
- Per-strip parallel scheduler tops out at 4 worker threads.

## [0.3.0] — 2026-05-31

LOD-and-streaming release: per-grid Far-tier billboard impostors,
mid-tier mip overrides, and an end-to-end streaming + procedural
generation pipeline. Closes the S6 + S7 macro-stages of
[`PORTING-SCENE.md`](PORTING-SCENE.md).

### Added

#### `roxlap-scene` — S6 (Far-LOD billboards)

- **Per-grid LOD picker** (`roxlap_scene::lod`): new
  `LodThresholds { r_near, r_mid, mid_mip_levels, mid_mip_scan_dist }`
  + `Lod::{Near, Mid, Far}` enum + `select_lod(camera_world_pos,
  transform, thresholds)`. Wired into `Grid::lod_thresholds` and
  consulted by `render_scene_composed`. Default
  `LodThresholds::always_near` is byte-identical with the pre-S6
  path.
- **Mid-tier mip overrides** (`Grid::lod_thresholds.mid_mip_*`):
  per-grid Mid-tier override for `OpticastSettings::mip_levels` /
  `mip_scan_dist`. Falls back to caller's settings when both
  fields are `None`. Plays nicely with the existing
  `Grid::mip_levels_override` cap.
- **Billboard impostor cache** (`roxlap_scene::billboard`): new
  `BillboardCache` + `BillboardSnapshot` with 26 canonical
  viewpoints (6 face + 12 edge + 8 corner). Rendered via opticast
  at `D = 8 × bounding_radius` near-orthographic camera against
  the runtime sky. Lazy: built on first Far-tier entry per grid,
  cleared by edits + eviction + stream-in (S7.4).
- **Far-tier blit** (`roxlap_scene::render::billboard_blit_into`):
  walks `BillboardCache::pick_nearest`, projects the grid centre
  via the camera basis, stamps the impostor's RGBA pixels with a
  constant z. Skips by the sky-sentinel (`0x00_00_00_00`) so
  background pixels in the impostor don't write to the
  framebuffer.

#### `roxlap-scene` — S7 (streaming + procedural generation)

- **`ChunkGenerator` trait** (`roxlap_scene::streaming`):
  `Debug + Send + Sync` pluggable per-chunk generator with
  `generate(chunk_idx) -> Vxl` and a `should_generate(chunk_idx)
  -> bool` filter (default `true`). `Grid::generator:
  Option<Arc<dyn ChunkGenerator>>` carries the generator;
  `Grid::ensure_chunk_generated` is the synchronous helper.
- **`StreamRadius { r_active, r_evict }`**: per-grid streaming
  policy in grid-local voxel units. `DISABLED` sentinel (`r_active
  = 0`, `r_evict = ∞`) is the default — pre-S7.1 grids keep their
  "absent stays absent" semantics. `new()` panics on `r_evict <
  r_active`, NaN, or negative.
- **Per-chunk version counter**
  (`Grid::chunk_versions: HashMap<IVec3, u64>`): edits bump;
  `ensure_chunk_generated` does NOT. Survives the
  `SceneSnapshot` round-trip via `#[serde(default)]` so pre-S7.2
  snapshots deserialise cleanly.
- **`Scene::pump_streaming_sync(camera_world_pos)`** (S7.1) +
  **`Scene::pump_streaming(camera_world_pos)`** (S7.3, async):
  per-frame drain + evict + dispatch. Async path uses a dedicated
  `rayon::ThreadPool` + `crossbeam_channel` inbox so chunk
  generation doesn't compete with R12's render pool. Race
  detection: each ChunkResult carries `version_at_dispatch`,
  installation gated on `chunk_version(idx) ==
  version_at_dispatch && !chunks.contains_key(idx)`. Eviction
  also drops `pending_gen` entries so stale results past
  `r_evict` are discarded. `Scene::set_streaming_threads(n)`
  reconfigures the pool (drops + rebuilds; channel survives).
- **Stream-in invalidates billboards** (S7.4): both
  `ensure_chunk_generated` and the async drain clear
  `Grid::billboards` after install so the impostor cache rebuilds
  with the new bounding sphere.
- **`CaveChunkGenerator<G>`** (`roxlap_scene::cavegen`): generic
  adapter wrapping any `roxlap_cavegen::Generator<Params =
  CaveParams>` (works with `BlueCaveGenerator`,
  `MagCaveGenerator`). `chunk_idx.z != 0` returns a bedrock-only
  chunk; `chz = 0` derives a per-chunk seed via FNV-1a of `(base_seed,
  chunk_idx.x, chunk_idx.y)` and calls the inner preset at
  `vsid = CHUNK_SIZE_XY`. Visible chunk-boundary seams documented
  as a v1 limitation; continuous-cave deferred.

#### `roxlap-scene-demo` (default mode now streaming)

- **Streaming-hills demo by default**: `build_demo` attaches a
  `HillsChunkGenerator` to the ground grid with
  `StreamRadius::new(256.0, 384.0)`. Chunks visibly load + unload
  as the camera moves. `ROXLAP_STATIC=1` restores the
  historical 32×32 statically-built ground for regression /
  visual-A-B work. `T` prints `chunks=N pending=N
  radius=A/E` per streaming grid.
- **`StreamingBakeTracker`**: per-frame lighting + mip bake
  driver. Bakes newly-installed chunks + their four cardinal
  neighbours via a `Grid::chunk`-resolving `EstNormCache` reader,
  so chunk-edge brightness banding resolves as chunks settle
  around the camera. Bake-on-stream-in replaces the previous
  in-isolation bake inside `HillsChunkGenerator`, which had no
  neighbour context and produced visible seams.

#### `roxlap-formats`

- `Vxl::reset_to_single_mip` (called from `generate_mips`) now
  walks columns + `slng` to recompute the actual end of mip-0
  data instead of trusting the chunk-creation-time sentinel. Fixes
  an OOB panic on the **second** `generate_mips` call against any
  chunk that had been edited (= had `voxalloc`-driven column
  scatter past the original sentinel). Surfaced by S7.6's
  streaming bake tracker; pre-existing in `roxlap-formats`.

### Public-API breakage

- `roxlap_scene::Grid::generator` field type:
  `Option<Box<dyn ChunkGenerator>>` → `Option<Arc<dyn ChunkGenerator>>`.
  Required so S7.3's async dispatch can clone the generator into
  background rayon tasks. Callers that constructed with
  `Box::new(...)` should switch to `Arc::new(...)`.
- `ChunkGenerator` gained `should_generate(&self, _idx: IVec3) ->
  bool` (default `true`). Existing implementations keep current
  semantics; opt in by overriding.

### Known limits (still open)

- **S4B.6.j cross-chunk look-down rendering** at non-camera XY
  columns still hits the `chunk_world_z_base` mismatch on
  column-step + handoff (rendered surface appears 256 voxels
  below expected). The fix lives in a "virtual stitched-column"
  rasterizer rewrite estimated at 2-4 weeks
  (`memory/project_s4b_6_chz_multi_research.md`); deferred to
  post-0.3.0. Mitigated in the streaming-hills demo by
  `HillsChunkGenerator::should_generate` declining `chz != 0`
  so the camera-above-grid path never materialises the
  placeholder chunks that trigger the bug.
- **`opticast_prelude::derive_prelude`'s `gylookup`** multiplies
  `(chunks_z * 512) * PREC` and overflows `i32` at `chunks_z ≥
  4`. Streaming demo uses `r_active = 256` to stay at `chunks_z =
  3`. Proper fix (wrapping_mul or i64 indexing) deferred.
- **Edits across eviction**: a user edit to a streamed chunk is
  lost when the chunk is evicted + re-streamed. Persistent
  dirty-flag handling is deferred to 0.3.x.
- **CaveChunkGenerator boundary seams**: per-chunk independent
  Worley seeds. Continuous-cave neighbour-aware pool deferred to
  0.3.x.

[0.3.0]: https://github.com/NCrashed/roxlap/releases/tag/v0.3.0

## [0.2.0] — 2026-05-27

Scene-graph release: many independent chunked voxel grids, each with
f64 world position and `Quat` rotation. Substages S1..S5 of
[`PORTING-SCENE.md`](PORTING-SCENE.md).

### Added

#### `roxlap-scene` (new crate)

- New crate. Scene-graph layer above the per-grid voxel renderer.
  Many `Grid` objects, each a sparse `IVec3 → Chunk` map; each
  grid carries a `GridTransform { position: DVec3, rotation:
  DQuat, scale: f64 }` so worlds can be unbounded in size and
  rotated to arbitrary orientation.
- **Address math** (`roxlap_scene::addr`):
  `world_to_grid_local` / `voxel_split` / `voxel_global` /
  `grid_local_to_world`. Handles negative / boundary / rotated
  cases (15 property tests).
- **Edit API** (`roxlap_scene::edit`): `Scene::set_voxel` /
  `set_rect` / `set_sphere`. Multi-chunk decomposition delegates
  per-chunk writes to `roxlap_formats::edit`. Empty chunks keep
  voxlap's bedrock placeholder at z=255.
- **Snapshot + serde** (`roxlap_scene::snapshot`):
  `SceneSnapshot` / `GridSnapshot`, `Scene::to_snapshot` /
  `from_snapshot`. 100-chunk round-trip via bincode validated.
  `compact_serialize_chunk` rebuilds in column-index order to work
  around `vxl::serialize`'s post-edit non-round-trip.
- **Multi-grid render** (`roxlap_scene::render`):
  `render_scene` (last-grid-wins single-buffer) +
  `render_scene_composed` (per-grid temp buffers + min-z merge).
  Translates the world camera into grid-local space per grid and
  dispatches to `roxlap_core::opticast`.
- **Cross-chunk gline** (S4): 3D DDA over chunk index space
  inside a single grid. New `GridView` / `ChunkGrid` types in
  `roxlap_core` wrap a multi-chunk sparse map; `opticast` walks
  chunk-by-chunk with explicit handoff at boundaries (mid-render
  and at seed-time). Power-of-two chunk_size_xy → arithmetic-shift
  fast path (+22% FPS vs. the pre-S4 single-chunk view).
- **Per-grid rotation** (S5): `world_camera_to_grid_local` inverts
  the grid quat to bring camera basis + position into grid-local
  space; the rasterizer sees an axis-aligned grid and is unaware
  of rotation. Identity-rotation is byte-identical to S4 output;
  180°-Z is bit-exact via exact-arithmetic quat. Per-grid
  lighting bake stays in grid-local space — a rotating ship is
  "lit by a sun that turns with it".

#### `roxlap-core`

- **`GridView<'a>`**: new four-field (vsid, slab_buf,
  column_offsets, mip_base_offsets) borrow-style abstraction
  threaded through every `opticast` / `ScalarRasterizer::new`
  callsite. Replaces ad-hoc `&Vxl` + `usize` argument tuples.
- **`ChunkGrid<'a>`**: sparse multi-chunk view; `GridView`
  carries an optional pointer to a `ChunkGrid` and dispatches
  `chunk_at_xy` / `chunk_at_xyz` via the table for multi-chunk,
  falls back to `Some(Self)`-for-`[0,0,0]` for single-chunk.
- **Cross-chunk rasterizer state**: `GrouscanState` carries
  `current_chunk_idx_xy` / `current_chunk_z` / `current_chunk_exists`;
  column-step routes by `chunk_size_xy == vsid → flat fast path`
  vs `chunk_size_xy < vsid → chunk-swap via chunk_at_xy`. Single-
  chunk path byte-identical to 0.1.x.
- **Z-stacked chunks**: `ChunkGrid` grows `chunks_z` /
  `origin_chunk_z` / `chunk_size_z`. `gylookup` table widened to
  `((chunks_z * 512) >> mip) + 4`. Mid-render handoff for
  look-down across stacked chunks. Slab byte reads operate in
  world-z (= chunk-local + `camera_chunk_z * 256`).
- `Grid::mip_levels_override` field: per-grid clamp on mip
  levels — sidesteps the axis-aligned-mip-N artifact for small
  rotating grids by forcing mip-0.
- Engine bug fixes that surfaced during S4/S5 integration:
  - **OOB-XY chunk-edge streaking** — opticast now seeds OOB-XY
    columns from the `(0, 255, 0)` bedrock placeholder rather
    than a synthesised cf entry. (`outside_orbit` golden refrozen.)
  - **Below-floor sky distortion** — `phase_draw_{cwall,fwall,
    ceil,flor}` drains now write `cx0/cy0/cx1/cy1` alongside
    `i0/i1`, fixing the sky lookup offset.
  - **Below-bedrock all-sky** — camera at z > 255 with
    `treat_z_max_as_air` now synthesises a bedrock air gap
    instead of returning `None`.
  - **Ship-grid mip-N black wall** — `phase_remiporend` reloads
    `state.column` only when `current_chunk_exists`, avoiding an
    OOB march into the wrong sub-table.
  - **Camera above grid skipped** — `chunk_at_xyz` clamps the
    effective chz to `origin_chunk_z` for unstacked grids so a
    camera at world z < 0 renders correctly.
  - **Three rotated-grid camera-placement bugs** — grouscan mip-N
    column-step underflow, chz clamp for camera-below-grid,
    cz_local "below the column" branch.
  - **Mip-N slab_z_at scale fix** — `slab_z_at` shifts the chunk
    world-z base by `>> gmipcnt` so mip-N reads in stacked
    chunks land at the correct world-z.
  - **Pose-D `phase_draw_fwall` bedrock guard** — read
    `column[vptr+1]` as a raw byte rather than casting via
    `slab_z_at` (mirrors drawflor's fix).
- 1 added oracle pose (`outside_orbit`, S1, roxlap-only golden).
  Existing 12 voxlap-comparable oracle hashes byte-identical to
  0.1.x.

#### Public-API breakage

- Everything that took `&Vxl` + raw `vsid` / `slab_buf` /
  `column_offsets` arguments to opticast / grouscan now takes
  a single `GridView<'a>`. ~18 call sites migrated internally;
  downstream callers of `opticast` need the same.
- `roxlap_scene` is new — no migration burden, but pulls in
  `glam` as a public dep.
- `Grid::combined_world` and friends (an Approach-C scaffold
  that briefly existed during S4) — DELETED in S4B.4.b in favour
  of `Grid::chunk(idx)` reader + per-chunk bake closure. Never
  shipped in 0.1.x.

#### Not (yet) included

- Per-grid LOD tier selection / billboards / planet sphere proxies
  (S6 — coming in 0.3.0).
- Streaming + procedural generation (S7 — coming in 0.3.0).
- chz multi-rendering: rasterizer state densely coupled to a
  single column, so rendering content from two stacked chunks
  through one column is unsupported. Scene-demo poses A/B/C/D
  unaffected; full investigation log in memory.

[0.2.0]: https://github.com/NCrashed/roxlap/releases/tag/v0.2.0

## [0.1.1] — 2026-05-07

- docs.rs metadata patch. No code changes.

## [0.1.0] — 2026-05-07

Initial public release of the roxlap workspace.

### Added

#### `roxlap-formats`

- `.vxl` (heightmap voxel world) parser + serialiser, including the
  multi-mip extension (`generate_mips`) and per-mip column-offset
  tables.
- `.kv6` (voxel sprite) parser + serialiser, including the optional
  `"SPal"` palette trailer.
- `.kvx` (legacy voxel sprite) parser + serialiser.
- `.kfa` (kv6 animation rig) parser + serialiser, plus the
  `KfaSprite` host-facing scene type and the `sort_hinges`
  topological-order helper.
- `Sprite` data type — kv6 + world-space pose + flag bitfield —
  with the `SPRITE_FLAG_*` constants.
- All parsers round-trip byte-equally on every fixture they accept.
- **Voxel-edit module** (`roxlap_formats::edit`):
  - Slab-pool allocator on `Vxl` — `vbit` / `vbiti` fields plus
    `reserve_edit_capacity` / `voxalloc` / `voxdealloc`. Port of
    voxlap's `vbit` bitmap free-list (`voxlap5.c:822/841`).
  - Low-level b2 z-range buffer ops `delslab` / `insslab` (port of
    voxlap5.c:4231/4259) plus the slab encode / decode helpers
    `expandrle` / `compilerle` (voxlap5.c:4131/4154) and the
    `slng` slab-chain length walker (voxlap5.c:814).
  - `ScumCtx` — column-edit batch context with the rolling 3-row
    radar-buffer cache (voxlap5.c:4431/4544/4507's
    `scum2_line` / `scum2_finish` / `scum2`). Closure-based
    `with_column` API skips the redundant `expandrle` when
    successive edits land on the same column.
  - High-level region wrappers `set_spans` / `set_cube` /
    `set_sphere` / `set_rect` plus `*_with_colfunc` variants that
    accept any `FnMut(i32, i32, i32) -> i32` colour callback —
    closure captures replace voxlap's `vx5.colfunc` /
    `vx5.curcol` global-state dance.
  - Byte-equality validated against voxlap C: a `setspans` carve
    on a 5×5 column patch produces output byte-identical to the
    voxlap-C reference (fixture captured from voxlaptest's new
    `edit_fixture` harness binary).

#### `roxlap-core`

- Pure-Rust port of voxlap's `opticast` raycaster and `grouscan`
  per-ray voxel-column rasterizer (R4.x).
- Multi-mip rendering via per-mip column-offset tables (R4.5).
- Textured panoramic sky via the `Sky` type and `phase_startsky`
  textured-fill branch (R4.4).
- Per-side wall-face shading (`set_side_shades`, sideshademode swap)
  matching voxlap's `setsideshades` ABI.
- x86_64 SSE2 batches for the four scanline rasterizer paths
  (`hrend` / `vrend` / `hrendzfog` / `vrendzfog`).
- KV6 sprite rendering: 4-plane frustum cull, per-voxel rasterizer,
  9-arm slab walk, alpha-byte face shading, and lightmode-2 point-
  light shading via `update_reflects` (R6.0–R6.5).
- KFA-animated sprites with bone hierarchy, hinge math, and the
  `kfadraw` per-frame transform pipeline (R6.6).
- World voxel lighting (`update_lighting`) — voxlap's
  `updatelighting` baked-intensity pass (R6.5).
- 2D textured-quad blit (`drawtile`) covering voxlap's three quality
  modes.
- High-level `Engine` + `Camera` types with idiomatic Rust
  constructors and getters; `OpticastSettings::for_oracle_framebuffer`
  convenience builder.
- **Multicore CPU rendering** (R12) — three rayon-backed
  parallelism axes:
  - **Per-strip opticast** via `ScratchPool::new_parallel(.., n)`:
    splits the framebuffer into `n` row strips, each running an
    independent opticast pass over its strip's y-range. New
    `OpticastSettings::y_start` / `y_end` clip the projection
    + scan-loop iteration. Peaks at ~1.5× on 4 strips for the
    oracle pose suite (i7-12700H); per-strip ray-fan
    discretisation drifts sub-pixel across `n`, so goldens are
    frozen at `n=1` (single-strip = full frame, byte-stable).
  - **`update_lighting` parallel bake**: outer y-loop is
    `rayon::par_iter`, per-column writes via raw-pointer view
    under the voxalloc disjoint-byte-range invariant. ~3.4×
    speedup on the oracle bake region — turns dynamic /
    per-edit relighting from "pause-the-game" to interactive.
    Bit-identical to sequential.
  - **`draw_sprites_parallel`**: new entry point that par_iters
    over `&[Sprite]` with z-test arbitrating concurrent fb / zb
    writes. `DrawTarget` refactored to `Copy + Send + Sync` raw-
    pointer view. ~4–6× speedup on synthetic many-sprite
    scenes; sprite oracle goldens unchanged (existing 2-sprite
    poses are non-overlapping). Tied-z races on overlapping
    sprites are non-deterministic but visually identical.
  - Three new oracle bench subcommands report scaling curves:
    `bench --threads N`, `bench-lighting`, `bench-sprites`.
  - Full design + measured numbers in
    [`PORTING-MULTICORE.md`](PORTING-MULTICORE.md).

#### `roxlap-host`

- Interactive demo binary (`cargo run -p roxlap-host`) — winit +
  softbuffer window with WASD + mouse-look fly-through over the
  bundled oracle voxel world.
- Animated KFA sprite + procedural rotation demo.
- Textured panoramic sky from the bundled `assets/sky.png`.
- Frame-capture key (`F` writes `roxlap-capture.{txt,ppm}` for
  off-line repro of any rendering artifact).
- World-voxel-lighting toggle (`L`).

#### `roxlap-cavegen`

- New crate. Procedural cave generation; depends only on
  `roxlap-formats` (no renderer).
- `Generator` trait + `CaveParams` struct (seed + 5 shape knobs).
- `BlueCaveGenerator` and `MagCaveGenerator` presets — visual /
  parameter defaults tuned to match Ken + Tom Dobrowolski's 2003
  *Justfly* demo screenshots (`caveblue2m.jpg`,
  `cavemag3m.jpg`).
- Hand-rolled classic 3D Perlin noise + fBm sum (`PerlinNoise3D`)
  and Worley-distance classification + dense-grid emit
  (`worley_classify_grid`, `place_seeds`, `classify_voxel`,
  `classify_voxel_with_perlin`). In-house deterministic
  `SplitMix64` PRNG; no `cmake` / C++ build deps for downstream
  users.
- `pack_dense_grid_to_vxl` — folds a `(VSID × VSID × MAXZDIM)`
  voxel mask + colour grid into voxlap's slab format via the
  `roxlap-formats::edit` pipeline.

#### `roxlap-cave-demo`

- New bin crate. Procedural-cave showcase
  (`cargo run -p roxlap-cave-demo`).
- Cave-gen on startup (`BlueCaveGenerator`, vsid = 128, ~1-2 s).
- WASD + mouse-look fly through with per-axis collision-checked
  movement (camera slides along walls instead of clipping).
- Plasma-bullet projectiles: LMB spawns a hot-pink bullet that
  flies along camera-forward, shrinks with distance for
  along-axis depth perception, and carves a sphere on impact.
- `F` toggles blue ↔ mag cave preset (regenerates the world);
  `R` regenerates with the next seed (preset preserved).
- Fog enabled (low-24-bit colour, no brightness bit per the
  voxlap-fork's `set_fogcol` brightness-bit gotcha).
- Spawn-bubble carve at world centre on init + on every
  regenerate, so the camera never starts inside a wall.

#### `roxlap-oracle`

- Cross-engine render-hash oracle (R8): renders 12 fixed test poses,
  FNV-1a-64 hashes each framebuffer, diffs against
  `tests/golden-hashes.txt`. CI (`.github/workflows/ci.yml`) gates
  every push.
- 5 of 12 oracle poses bit-exact with voxlaptest's C engine output;
  the remaining 7 frozen as roxlap goldens after visual verification
  (sub-pixel rounding noise from `_mm_rcp_ps`'s 12-bit approximation
  varies across CPU vendors).
- `cargo run -p roxlap-oracle -- diff` and the lower-level
  `cmd_debug_gline` subcommand for porting / debugging workflows.

### Documentation

- README rewrite: pitch shape, screenshot, quick-start, crate table,
  links to docs.rs.
- Per-crate Cargo metadata for crates.io discovery (keywords,
  categories, documentation).
- This CHANGELOG.

### Out of scope (for 0.1.0)

- ARM NEON port (R9): scalar fallback used on aarch64.
- wasm32 SIMD + browser host (R10): scalar fallback used on wasm32.
- Voxlap's animation-curve playback (`animsprite` + per-frame
  interpolation): the host drives `kfaval[]` directly.
- Sprite no-z (`SPRITE_FLAG_NO_Z`) overlay rendering: data type
  defines the flag, renderer skips it.
- Multi-mip + voxel-lighting integration with `roxlap-formats::edit`
  ops: edits operate on mip-0 only; the cave demo runs single-mip
  with no baked lighting.
- Spatial-bucketing acceleration for `roxlap-cavegen`'s Worley
  classify: brute-force `O(vsid² × MAXZDIM × seed_count)` today
  (~2 s at vsid=128, longer at vsid=256).
- A third cave-demo preset matching Ken's `caverock1m.jpg`
  rocky-cave reference: requires a sphere-stack generator,
  different algorithm than Worley.
- Full byte-fixture coverage of the edit ops against voxlap C
  (only one `setspans` fixture today; `setcube` / `setsphere` /
  `setrect` rely on round-trip self-consistency tests).

[0.1.1]: https://github.com/NCrashed/roxlap/releases/tag/v0.1.1
[0.1.0]: https://github.com/NCrashed/roxlap/releases/tag/v0.1.0
