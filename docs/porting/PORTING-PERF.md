# PORTING-PERF.md — Performance stage (PF)

Entry doc for the cross-backend performance pass. Produced from a full audit of
both backends (2026-07-02, four parallel deep-reads: CPU hot path, WGSL
shaders, GPU host, scene/streaming). Nothing below is implemented unless its
stage is marked LANDED.

Scope fact confirmed by the audit: the voxlap opticast/gline/grouscan column
renderer is **gone** — the DDA stage replaced it wholesale with a per-pixel
3D-DDA (`crates/roxlap-core/src/dda.rs`); `opticast.rs` is only a settings
struct. All optimization work targets the DDA pipeline.

Golden rule for every stage: byte-identical output where the change is purely
mechanical (hoists, caches, guards); where output may drift (shadow-mip,
per-run volumetric lighting) the stage must say so, gate it or refreeze
goldens deliberately, and verify with the headless GPU diff-harness / CPU hash
tests.

---

## Cross-cutting themes

1. **Shadows dominate both backends.** CPU: heap alloc + full RLE decode per
   shadow-ray step. GPU: shadow rays march mip-0 ignoring the resident mip
   ladder, and every DDA step re-reads ~5 dependent storage words.
2. **Per-frame rebuilds of static data.** Fresh GPU buffers + bind groups
   every frame; sprite pass fully re-culls/re-bins/re-uploads; streaming clips
   rebuild the model per render frame instead of per animation frame.
3. **Edits pay for the whole chunk.** One bullet hole = chunk clone + full
   `generate_mips` + full decompress + full upload (~45 ms/carve batch);
   lighting bake is serial, full-chunk, main-thread (up to ~175 ms hitches).

---

## Findings — CPU backend (roxlap-core + roxlap-scene)

### C1 [HIGH, hours] Shadow/raycast voxel query allocates + full-decodes the column per step
`vxl_voxel_solid` (`crates/roxlap-scene/src/chunks.rs:92-110`): `vec![maxzdim; 516]`
(~2 KB malloc) + `expandrle` of the entire column per single-voxel test.
Callers in hot loops:
- CPU cross-grid shadow march `occluded_in_grid` (`crates/roxlap-scene/src/occluder.rs:158-184`,
  cap `SHADOW_MAX_STEPS = 4096` at `occluder.rs:25`) — per step per shaded
  pixel per casting light.
- `Scene::raycast` → `voxel_dda` (`crates/roxlap-scene/src/lib.rs:100-181`) —
  a 512-unit gameplay raycast ≈ 1500 allocs+decodes.
- `Scene::resolve_voxel` (`lib.rs:635`), demo collision.

Fix: (a) rewrite as an in-place slab-chain walk with early-out at z — pattern
already exists in `GridView::voxel_run_top_mip`
(`crates/roxlap-core/src/grid_view.rs:279-308`) and `Vxl::voxel_color`
(`crates/roxlap-formats/src/vxl.rs:255-290`); ~20 lines, no alloc.
(b) cache the current chunk `&Vxl` across DDA steps (re-lookup only on chunk
cross). (c) longer-term: route shadow rays through per-grid `Sampler` +
`BrickCache` for empty-space skip.

### C2 [HIGH, ~1-2 days] Sprites re-densified per draw; occluder re-densifies again; KFA deep-clones Kv6 per frame
- `draw_sprite_dda_shaded` decodes KV6 → `SpriteDense` per sprite per frame
  (`crates/roxlap-core/src/dda_sprite.rs:831-839`, admitted in comment).
- Shadow occluder rebuild densifies every caster again each frame
  (`crates/roxlap-render/src/cpu.rs:1206-1228`; clip dense frames `d.clone()`d
  at `cpu.rs:1219`).
- `update_kfa_poses` clones full limb Kv6 voxel data per frame
  (`crates/roxlap-render/src/cpu.rs:969-976`; `Sprite` owns `Kv6` by value,
  `crates/roxlap-formats/src/sprite.rs:78-83`).

Fix: cache `SpriteDense` per model in `CpuRenderer` (invalidate on
`update_sprite_model`); share with the occluder via `Arc<SpriteDense>`;
keep `kfa_limbs` as pose records referencing `kfa.limbs` (or `Arc<Kv6>`).

### C3 [MED-HIGH, medium] Single-grid shadow march has no empty-space skip
`SamplerShadow::occluded` (`crates/roxlap-core/src/dda.rs:1229-1262`,
`SHADOW_MAX_STEPS = 1024` at `dda.rs:1214`) steps one mip-cell per iteration;
primary rays get brick/super skip (`dda.rs:1381-1462`), shadow rays don't.
Fix: factor an occupancy-only variant of `cell_walk_skip`'s fast-forward.

### C4 [MED, trivial each] Light-loop micro fixes
- `sqrt` before radius cull: `shade_dynamic` (`dda.rs:515-549`, sqrt at 521) —
  test `dist2 < radius²` first.
- Per-hit `env.lights.points.iter().any(|p| p.casts_shadow)` scan
  (`dda.rs:1475-1499`) — precompute per grid in `grid_local_lights`
  (`crates/roxlap-scene/src/render.rs:120-185`) or `DdaEnv` build
  (`render.rs:962-976`).
- No light-vs-grid-AABB cull in `grid_local_lights` — drop lights whose sphere
  misses the grid AABB (bounds already at `render.rs:782`).
- `material_for_color` linear scan runs twice per opaque hit
  (`dda.rs:1514` and `dda.rs:1529`; scan in
  `crates/roxlap-formats/src/material.rs:193-201`) — return `(Material, id)`
  once; pre-index the map in `DdaEnv`.
- Spot-cone math itself is cheap (dot + smoothstep, `dda.rs:303-309`,
  correctly ordered before the march) — no action.

### C5 [MED, trivial] Rayon band granularity = 1 band per thread
`render_dda_parallel` (`dda.rs:1747-1756`): `band = rows.div_ceil(nthreads)` —
sky rows finish instantly, horizon rows dominate, threads idle. Fix: 4-8-row
bands + work stealing. Typically +10-30 % wall clock. Byte-identical.

### C6 [MED, hours] Full-frame temp buffers + sequential compose/sky/sprites
- `render_scene_composed_scissored` allocates+initializes `temp_fb`/`temp_zb`
  per frame (`crates/roxlap-scene/src/render.rs:618-619`; ≈7.4 MB @720p, ×16
  at 4×SSAA). Also per-frame: `eff_mips` HashMap (`:627`), per-grid
  `chunk_xyz_backing()` (`:766` → `chunks.rs:325-370`), `light_scratch`
  (`:943`).
- `render_sky_fill` sequential with `acos`/`atan2` per pixel
  (`dda.rs:697-718`, called from `cpu.rs:1255-1268`).
- Sprite pass entirely single-threaded (`cpu.rs:1320-1371` →
  `dda_sprite.rs:961-1104`).
- `compose_rect`/`fill_rect_*` single-threaded memory-bandwidth loops
  (`render.rs:438-473, 991-1004`).

Fix: persistent scratch pair owned by the backend; rayon rows for sky fill,
compose, and sprite rect; world-AABB early-reject per `SpriteOccEntry`
(`dda_sprite.rs:367-373`).

### C7 [MED, low-medium] Scissor is vertical-band-only — dead voxlap constraint
`render.rs:794-826`: rect widened to full width because the old radar
`angstart` couldn't x-clip. DDA pixels are independent — keep `r.x0..r.x1`,
plumb an x-range into the parallel driver. Wins scenes with many small grids.

### C8 [LOW-MED] Misc
- Redundant occupancy tests per dense cell: `super_occupied` + `brick_occupied`
  then `Sampler::hit` redoes locate+select+brick test
  (`dda.rs:1381-1391`, `1135-1152`).
- `surface_color_mip` walks the slab chain up to 3× per hit
  (`grid_view.rs:360-364`) — fuse into one walk.
- Per-frame O(#chunks) HashMap sweeps per grid: `chunk_xyz_backing` (2 sweeps),
  `grid_bounds` (`billboard.rs:362-388`), `SceneOccluder::build` runs twice
  per frame when shadows on (`render.rs:658` and `cpu.rs:1288-1290`),
  `ensure_dda_bricks` (`scene/lib.rs:380-402`). Cache bounds on `Grid` keyed
  by edit version; reuse phase-A occluder.

---

## Findings — GPU shaders (roxlap-gpu/shaders)

### G1 [HIGH, small+localized] Per-DDA-step redundant storage reads of chunk constants
`voxel_solid_in` (`scene_dda.wgsl:272-277`, per inner step at `:843`, shadow
`:516`) re-derives everything per call: `col_word_base_mip` (`:251-258`) reads
`vsid`, `occupancy_offset`, `occ_words_per_slot`, `mip_occ_rel[mip]`;
`mip_occ_block_words` (`:263-266`) re-reads `vsid` — ~5 dependent storage
loads to fetch one occupancy word, all constant per chunk. Consecutive z-steps
hit the same occ word up to 32×. Same pattern in
`sprite_terrain_shadow.wgsl:65-82`.
Fix: hoist per-chunk constants into registers at chunk entry (before the loop
at `:842`); cache `(word_index, word_value)` across steps. Single best flat
win — applies to every primary and shadow step. Byte-identical.

### G2 [HIGH, near-one-line] Shadow march ignores the mip ladder
`scene_dda.wgsl:516` and `sprite_terrain_shadow.wgsl:159` hardcode mip 0;
mips are resident (GPU.11) and `pick_mip` exists (`scene_dda.wgsl:460-467`).
March at `pick_mip(t_pixel + t_along_ray)` (or fixed mip ≥1 past ~32 units):
2-8× fewer steps. NOT byte-identical — verify with the headless diff harness,
acceptable drift for stylized hard shadows; consider env/setting gate.

### G3 [MED-HIGH, mechanical] Double voxel fetch per hit; volumetric cells shade per cell
- `voxel_packed_in` evaluated inside `shade_lit`/`voxel_color_in` (`:631`/`:328`)
  and AGAIN at `scene_dda.wgsl:871` — each call = popcount rank scan of up to
  8 occ words + 3 dependent loads. Same in sprites:
  `sprite_model_dda.wgsl:513` vs `:257`.
- Translucent/volumetric accumulate path calls full `shade_lit` (sun + N
  lights + shadow marches) per CELL (`scene_dda.wgsl:853-861`) — Beer-Lambert
  water = dozens of cells/pixel. Light once per contiguous run.
Fix: return `packed` from shade functions; per-run lighting for volumetrics
(second part NOT byte-identical for translucent scenes — gate/refreeze).

### G4 [MED] Full 144-byte GridStaticMeta struct copies twice per outer chunk step
`slot_idx_of` (`scene_dda.wgsl:371-378`) and `chunk_has_content` (`:404-417`)
each do `let m = grid_static_meta[g];` (naga materializes both 6-word mip
arrays); `aabb_passed` (`:389-401`) re-reads aabb per step. Runs per outer
step in primary (`:799-801`) and shadow (`:497-498`) marches + in
`sprite_terrain_shadow.wgsl:83-111`. Fix: hoist to locals once per
march invocation. Byte-identical.

### G5 [MED] Point-light loop: full 64-B load + length() before radius reject
`scene_dda.wgsl:687-691` (and `sprite_model_dda.wgsl:327-331`): loads all four
vec4s + `length()` for all ≤32 lights per shaded pixel/cell; `max_t` at
`:702-703` recomputes a length that ≈ `dist` — reuse it. Fix: slim
`vec4(pos, radius)` pre-scan array, or per-tile light lists later.

### G6 [MED] Shadow-setup work even when nothing casts + occluder test order
`shade_lit` computes `shadow_origin_w = grid_local_to_world(...)` (4 vec4
reads of `grid_cameras[g]`, `scene_dda.wgsl:649-652`) unconditionally per hit.
Compute lazily in the first shadowed branch. In `shadow_occluded_world`
(`:591`) sprites are tested before grids — with sprites present, grids are
usually the likelier occluder; reorder.

### G7 [MED, translucency-gated] `terrain_material_id` linear scan per translucent cell
`scene_dda.wgsl:213-221` called at `:872` — O(map_count) storage reads per
cell. Fix: per-voxel material-id byte uploaded parallel to `all_colors` (as
sprites do with `materials_vox`).

### G8 [LOW-MED] Misc shader
- `pow()` per volumetric cell (`scene_dda.wgsl:887`,
  `sprite_model_dda.wgsl:545`): precompute `log2(1-a)` into the `Mat` palette
  pad, use `exp2`.
- Sprite translucent mode runs the two-sweep structure for ALL tiles
  (`sprite_model_dda.wgsl:602-680`) — add per-tile "has translucent" bit in
  `tile_ranges` spare bits.
- `ray_dir_len` (`scene_dda.wgsl:752`) is `length` of a normalized vector —
  always 1.0, delete.
- Legacy `grid_dda.wgsl:262-273` dead per-step block (only on the GPU.4
  single-grid path) — delete or ignore.
- Stylized mode (`style_bands >= 1`) samples the voxel centre (`:647`) ⇒ sun
  shadow constant per voxel — adjacent pixels redundantly recompute; possible
  per-workgroup dedup later (high effort).
- Workgroup shape 8×8 on variable-length rays: try 8×4/16×16 empirically only
  after G1-G5 land.

### Verified fine (no action)
Per-pixel normalize once per grid loop; `pick_mip` per chunk entry (no mid-chunk
restart); occ paging branch uniform; one encoder/one submit for the main
scene; blit nearest/integer negligible; opaque lanes exit early; translucent
terrain path fully gated behind `terrain_has_translucent`.

---

## Findings — GPU host (roxlap-gpu + roxlap-render)

### H1 [HIGH, ~1 day] Per-frame storage-buffer creation → per-frame bind-group rebuild
`upload_grid_cameras` (`roxlap-gpu/src/lib.rs:1077-1086`, called `:2573`) and
`pack_scene_lights`→`upload_grid_point_lights` (`:1046-1064`, called `:2567`)
call `device.create_buffer_init` EVERY frame; sprite path creates another
light buffer (`:2790`). Changed handles force rebuilding the 22-entry scene
bind group (`:2643-2754`) and 23-entry sprite bind group (`:2938-2942`) per
frame. Fix: persistent COPY_DST grow-on-need buffers + `queue.write_buffer`;
cache bind groups, invalidate on regrow/resident swap. Also: engine deep-clones
`scene_lights` per frame (`:2566`) — borrow instead.

### H2 [HIGH, 1-2 days] Sprite pass: full cull+bin+colmul rebuild/upload per frame
`cull_bin_upload` (`roxlap-gpu/src/sprite_model.rs:1726-1888`, called `:2511`):
6+ fresh Vecs, full rewrite of instances/tile_ranges/tile_instances/colmul.
Worst: `visible_colmul` = 512 u32 = 2 KiB per visible instance per frame
(`:1815-1818`, upload `:1885`) — ~1 MiB/frame at 500 sprites, tables
effectively static. Fix: persistent per-instance colmul buffer indexed
indirectly (or skip while identity); reuse workspace Vecs; dirty-flag skip
when camera+instances+transforms unchanged.

### H3 [HIGH, hours] Streaming-clip playback rebuilds+re-uploads model every render frame
`set_streaming_clip_frame` (`roxlap-render/src/lib.rs:2943-2954`, called per
player per frame from `advance_voxel_clips` `:3078-3090`): unconditional
`to_kv6` (full dense decode) + `material_map.clone()` +
`refresh_sprite_model_with_materials` → full model + LOD-chain rebuild +
GPU upload (`roxlap-render/src/gpu.rs:863-880`). 10 fps clip @144 fps render
= 14× wasted rebuilds. Same missing guard in flipbook `set_clip_frame`
(`gpu.rs:663-679`, also hit per actor from `update_billboard_actors`
`lib.rs:2760`). Fix: `last_applied_frame` early-out; pass material map by
slice.

### H4 [HIGH for interactivity, hours] `pick_depth` copies the WHOLE depth buffer + blocking poll
`read_depth_pixel` (`roxlap-gpu/src/lib.rs:4182-4218`): copies w*h*4 bytes
(8.3 MB @1440p, `:4193-4194`), submits, `device.poll(wait)` (`:4202`) — full
CPU↔GPU sync per call (`pick_depth` `roxlap-render/src/gpu.rs:889-894`,
`pick_image` `roxlap-render/src/lib.rs:1784`). Fix: copy only the 4-byte
pixel (offset `(y*w+x)*4`); optionally async one-frame-late resolve.

### H5 [MED-HIGH, 1-2 weeks] Single-voxel edit ⇒ full-chunk remip+decompress+upload
Chain: `set_voxel`/`set_sphere` → version bump → `refresh_dirty`
(`roxlap-render/src/gpu.rs:1395-1448`) → `decompress_chunk`
(`roxlap-gpu/src/decompress.rs:213-264`; `vxl.clone()` + `generate_mips` at
`:224-225` when mips short) → `refresh_chunk` ~4 write_buffers per mip
(`roxlap-gpu/src/scene.rs:609-647`). `Vxl::generate_mips`
(`roxlap-formats/src/vxl.rs:388`) always `reset_to_single_mip` (`:352-369`,
walks ALL vsid² columns). Cave demo: ~45 ms/carve batch (`main.rs:947-950`),
whole-Vxl clone per batch (`main.rs:907`). Fix: per-chunk dirty voxel bbox;
incremental remip of affected column groups; column-range partial GPU refresh.

### H6 [MED, hours] Resolve pass + small uniform writes when RP features off
`roxlap-gpu/src/lib.rs:2976-2984`: resolve compute dispatch runs always; with
ssaa==1 + posterize None it is an identity full-screen copy. Also
unconditional 8-B flip write (`:2533-2537`) + 16-B posterize write
(`:2545-2549`). Fix: skip dispatch, second cached blit bind group reading
`framebuffer` directly; change-detect the small writes.

### H7 [MED, ~1 day] Per-frame overlay bind groups + one submit per overlay pass
One bind group per image quad per frame (`lib.rs:3460-3487`), lines bind group
per frame (`:3138-3151`); separate encoder+submit each for lines (`:3201`),
images (`:3519`), egui (`:3771`) + scene (`:3006`) ⇒ 2-5 submits/frame. Fix:
cache bind groups keyed by (ImageId, depth generation); record overlays into
the scene encoder; 1-2 submits/frame.

### H8 [MED, hours] Lights + materials re-synced every frame regardless of change
- `sync_lights` (`roxlap-render/src/gpu.rs:1456-1556`) rebuilds all per-grid
  light Vecs per frame; engine clones `Vec<Vec<GpuLight>>` (`lib.rs:2566`).
- `set_sprite_materials` + `set_scene_terrain_materials` write 2 KiB palettes
  every frame (`gpu.rs:1079-1083` → `roxlap-gpu/src/lib.rs:4608-4652`).
Fix: dirty flags on `define_material`/`set_terrain_materials`/light changes.

### H9 [MED at scale, hours] Per-frame O(all chunks) scans
- `resident_matches_scene` collects+sorts grid ids per frame
  (`gpu.rs:1298-1302`, duplicated `:1308-1312`).
- `refresh_dirty` iterates every chunk of every grid, 2 HashMap lookups each
  (`gpu.rs:1412-1416`); stale-eviction Vec per grid per frame (`:1434-1438`).
- Streaming eviction filters all chunk keys per grid per frame
  (`scene/lib.rs:888-904`); `for_each_chunk_in_radius` re-tests all candidates
  (`:936-968`); `ensure_dda_bricks` sweep (`:380-403`);
  `StreamingBakeTracker` rebuilds a HashSet per frame
  (`scene-demo/src/scene.rs:529`).
Fix: per-grid "any chunk changed" counter bumped by edits/installs; run
eviction only when camera moved > ½ chunk; cache sorted grid-id snapshot.

### H10 [MED on CPU backend, 1-2 days] CPU occluders rebuilt from scratch per frame
`roxlap-render/src/cpu.rs:1206-1228` re-decodes every caster
(`SpriteDense::from_kv6`) + `SceneOccluder::build(scene)` per frame
(`cpu.rs:1289`). Fix: cache dense per model; cache scene occluder against
grid versions. (Overlaps C2 — do together.)

### H11 [LOW-MED, minutes-hours] Misc host
- Per-frame `eprintln!` over light cap (`roxlap-gpu/src/lib.rs:1012-1015`,
  `1040-1044`; `roxlap-render/src/cpu.rs:1168-1173`; streaming log
  `gpu.rs:1445-1447`) — warn-once flags.
- `ColorsAllocator` free-list: no coalescing of adjacent free blocks
  (`sprite_model.rs:1909-2062`) — sort by offset + merge on free.
- `sync_aabb` O(pool-slots) rescan per chunk install/evict
  (`roxlap-gpu/src/scene.rs:754-792`) — incremental min/max.
- Small per-frame facade allocs (`gpu.rs:1089,1093,1168-1184,1259-1275`;
  `lib.rs:2553-2556`; `roxlap-render/src/lib.rs:2545,2728,2753`) — scratch
  reuse, do opportunistically.

### Verified fine (no action)
One submit for the main scene; `present()` is only `surf_tex.present()`;
depth readback on-demand only; CPU fixed-res path short-circuits (ssaa==1 no
posterize ⇒ `CpuSrc::Frame`); chunk upload budget bounds streaming spikes;
KFA transform flush coalesced once per frame; no miniz inflate on the frame
path; no snapshot/serialize on the frame path.

---

## Findings — scene layer / lighting bake / demos

### S1 [HIGH, 2-4 days] Lighting bake: serial, full-chunk, main thread
`StreamingBakeTracker::process_grid` (`scene-demo/src/scene.rs:490-577`): 1-5
installs → 5-25 full-chunk rebakes ≈7 ms each = up to ~175 ms in one frame;
each = `EstNormCache::build_with_reader_z` + `apply_lighting_with_cache` over
128×128×256 + `generate_mips(4)` (`:567`) + full GPU re-upload (`:574`).
Library `Grid::bake_lightmode` (`scene/chunks.rs:163-211`) is all-chunks,
serial, no bbox variant. Fixes: per-frame bake budget (mirror
`chunk_upload_budget`); parallelize (read phase is embarrassingly parallel);
public bbox-limited neighbour-aware rebake on `Grid` — primitive exists
(`world_lighting.rs:566` `update_lighting`; cave-demo `relight_bbox`
`main.rs:963-987` shows 0.04 ms vs 4-7 ms).

### S2 [MED-HIGH, ~1 day] `Scene::raycast` — no grid-AABB clip, no chunk skip
`scene/lib.rs:665-692` + `voxel_dda` `:100-181`: marches every grid
voxel-by-voxel from origin. Fix: clip ray to populated voxel AABB (pattern at
`occluder.rs:188-211`); skip missing chunks by advancing to chunk exit.
Combined with C1 ⇒ ~100× cheaper gameplay raycasts.

### S3 [LOW-MED, 1 day] First-Far-frame billboard cache build is a synchronous hitch
`scene/render.rs:631-634`: `BillboardCache::build` = 26 × res² renders inline;
every edit/stream event nukes the cache. Fix: build 1 snapshot (needed view)
per frame, fill others lazily.

### S4 [LOW, minutes] `frame_at` re-sums all durations per call
`voxel_clip.rs:404-439`, per player per frame via `ClipClock::tick`
(`roxlap-render/lib.rs:452-460`). Cache total + prefix sums.

### S5 [demo hygiene, hours] picking demo rebuilds the whole sprite registry per frame
`scene-demo/src/scenes/picking.rs:101-113`: `set_sprites` with cloned
`Vec<Sprite>` (full Kv6) per frame to move a cursor. Use instance transforms.
Bad example for API users.

---

## Stage plan

Order chosen by (impact ÷ effort), byte-identical mechanical wins first.

- **PF.0** — this document. ✅
- **PF.1 — LANDED** — G1 + G4: per-march meta-field hoists (no whole-struct
  copies), per-(chunk, mip) solid-word-base hoist + last-occ-word cache in
  both marchers of `scene_dda.wgsl` and in `sprite_terrain_shadow.wgsl`;
  `slot_idx_of`/`aabb_passed`/`chunk_has_content` scalarised. Byte-identical.
  Verified: naga validation (both spliced variants) + full headless GPU suite.
- **PF.2 — LANDED** — G2: scene shadow march follows the primary ray's LOD
  (`pick_mip(t_pixel + t_along_shadow_ray)`); `mip_scan_dist == 0` restores
  the all-mip-0 march, and shadows inside `mip_scan_dist` are unchanged.
  + G6 lazy `shadow_origin_w` lift in `shade_lit`. Follow-up deferred: the
  sprite pass's terrain-shadow march (`sprite_terrain_shadow.wgsl`) stays
  mip-0 — its `Uniforms` lack `mip_scan_dist`; add it when touching that
  struct next. G8 `ray_dir_len` deletion folded into PF.3 (volumetric-path
  float drift). Verified: full headless GPU suite (shadow tests are
  near-field ⇒ mip-0 ⇒ identical).
- **PF.3 — LANDED** — G3 + G5 + G8(ray_dir_len): single voxel fetch per hit
  (`voxel_packed_in` fetched once in march_grid, passed into
  `shade_lit`/`voxel_color_in`; sprite `voxel_index` computed once per hit,
  passed into `shade_sprite_lit`/`model_color`); light loops read only
  pos+radius (16 B) for a squared-distance reject before loading the full
  64-B light + sqrt; shadow-ray `length(to_light)` replaced by `dist` (the
  parameterisation `origin + t·(to_light/dist)` lands exactly on the light
  at `t == dist`); `ray_dir_len` deleted (ray is unit — volumetric float
  bits may drift ULP-level). Deferred from G3: per-run volumetric lighting
  (behavioral, revisit with G7). Verified: full GPU suite incl. point/spot
  brightness tests.
- **PF.4 — LANDED** — H1: `FramePackBuffers` — persistent grow-only (pow2,
  `line_vbuf` pattern) storage buffers for grid cameras + scene point lights
  + sprite world lights, written via `queue.write_buffer` (was
  `create_buffer_init` ×2-3 per frame); the 22-entry scene and 23-entry
  sprite bind groups are cached and keyed on the exact resources they bound
  (wgpu 29 identity `PartialEq`) — any regrow / resident swap / `scene_dda`
  rebuild / sky swap / registry growth misses the cache automatically, no
  manual event tracking. `pack_scene_lights` split to pure packing (headless
  keeps `create_buffer_init`); the per-frame `scene_lights` deep clone is
  gone. Samplers excluded from the key (init-stable only). Verified: full
  GPU suite + 60 s live scene-demo run on the GPU backend (NVK), incl. a
  post-init sky replacement (exercises the view key); sprite BG path is
  compile-verified + symmetric (no input injection available for a live
  sprite scene).
- **PF.5 — LANDED** — guards batch:
  - H3: streaming clips track `last_applied` (resolved/clamped index) —
    same-frame re-application skips `to_kv6` + LOD rebuild + GPU upload;
    flipbook `set_clip_frame` (GPU) gets the matching same-frame early-out.
  - H8: facade `materials_dirty` (set by `define_material` /
    `set_terrain_materials`, starts true; both device pipelines re-seed
    palettes at build so a pre-pipeline change is never lost);
    `GpuRenderer::set_scene_lights` compares (`SceneLights: PartialEq`)
    and no-ops on an unchanged rig; `render_scene` re-packs + re-uploads
    scene lights only on `scene_lights_dirty || grid_count` change (sun-dir
    injection into the fresh cam_vec extracted to `inject_grid_sun_dirs`,
    runs every frame; `sun_flags`/`point_count` cached); sprite world
    lights get their own `sprite_lights_dirty` (their upload only runs
    with sprites visible). Bonus: the over-cap eprintlns inside
    `pack_scene_lights` now fire once per rig change.
  - H6: `blit_bg_direct` reads the march framebuffer; `render_scene` skips
    the resolve pass when ssaa==1 && posterize off (identity copy). The
    24 B/frame flip+posterize uniform writes kept (negligible).
  - H4: `read_depth_pixel` copies ONLY the picked pixel's 4 bytes (was the
    whole w*h*4 depth buffer) before the blocking poll.
  - H11: CPU shadow-demotion eprintln fires once per change.
  Verified: full GPU+render suites; two 30 s live GPU scene-demo runs —
  default (identity-resolve direct blit) and `ROXLAP_SSAA=2
  ROXLAP_POSTERIZE=6` (resolve path) — no validation errors.
- **PF.6 — LANDED** — C1 + S2:
  - `vxl_voxel_solid` walks the slab chain in place with an early-out at z
    (mirrors `expandrle`'s run derivation one run at a time) — the
    per-query `vec![516]` + whole-column decode is gone from every CPU
    shadow-ray step / raycast step / collision probe. Equivalence test vs
    the expandrle reference added (multi-slab cave columns).
  - `SolidSampler` (chunk-cached `Grid::voxel_solid`): one chunks-HashMap
    probe per chunk crossing instead of per step; used by the shadow
    oracle (`occluded_in_grid`) and the raycast DDA.
  - `Scene::raycast`/`voxel_dda`: ray pre-clipped to the populated chunk
    AABB (miss = one slab test; distant grids marched AT the box), and a
    voxel in an absent chunk fast-forwards to the chunk's exit face
    instead of stepping ≤128 voxels through it.
  Measured (throwaway release micro-bench, 20k mixed 600-unit raycasts on
  a 4×4-chunk cave terrain): 25.55 → 1.99 µs/ray (12.8×), identical hits.
  Full workspace suite green (incl. golden hashes). Occluder still steps
  absent chunks voxel-wise (jump deferred; sampler+alloc-free covers the
  dominant cost there).
- **PF.7 — LANDED** — C4 + C5 + C6:
  - C4: squared-distance light reject before the sqrt (`shade_lit_cpu`);
    "anything casts?" `any()` scan hoisted out of the hit block (ray-
    invariant); ONE colour→material scan per hit (id + material resolved
    together, `terrain_material` deleted); point lights culled against the
    grid's world bounding sphere (+`shadow_bias`+1 slack ⇒ byte-identical)
    BEFORE the per-grid transform in `grid_local_lights`.
  - C5: `render_dda_parallel` bands = fixed 8 rows + rayon work stealing
    (was rows/nthreads — sky rows finished instantly, horizon rows
    dominated, threads idled per frame). Bit-identical.
  - C6: `SceneRenderScratch` (temp fb/zb pair sized-not-cleared — every
    read pixel is fill_rect-initialised per grid; light scratch; eff_mips
    map) owned by `CpuBackend` via the new
    `render_scene_composed_with_materials_scratch`; `compose_rect` and
    `render_sky_fill` parallelised over disjoint rows (both crates already
    depend on rayon incl. wasm). Kills two full-frame allocations + their
    init writes per frame.
  Deferred to PF.8: sprite-pass row parallelism + `SpriteOccEntry`
  world-AABB early-reject (rides the sprite caching work).
  Measured: World scene (streaming hills, static spawn pose, release,
  no dynamic lights) 6.4 → 8.0 FPS (+25%). Full workspace suite green
  (golden hashes + scissor byte-identity regression).
- **PF.8 — LANDED** — C2/H10 (+PF.7 deferrals):
  - CPU backend caches ONE dense decode per model template / per KFA limb
    (`DenseCacheEntry` = `Arc<SpriteDense>` + kv6 shape key). The draw
    pass and the shadow-occluder build share it — previously every caster
    was re-densified per occluder rebuild AND every instance re-densified
    again per draw call, every frame. Invalidation: explicit on
    `set_sprites` / `update_sprite_model` / `remove_model`; the shape key
    (dims+voxel count) backstops lingering post-tombstone instances,
    which fall back to an inline decode.
  - `SpriteOccluder::push` takes `Arc<SpriteDense>`; `ClipFlipbook`
    frames are `Arc`-shared (`frame_arc`) — the occluder no longer
    deep-clones the current clip frame per rebuild.
  - `update_kfa_poses` is pose-only (p/s/h/f + display fields): the old
    path deep-cloned every limb incl. its Kv6 EVERY pose update.
    Registration (`set_kfa_sprites`) still full-clones once; a limb-count
    mismatch falls back to full rebuild.
  - `draw_sprite_dense_shaded` parallelises rows via rayon for footprints
    ≥ 64×64 px (small rects stay serial — per-sprite overhead). Rows
    disjoint, per-pixel reads own-index only; bit-identical.
  - `SpriteOccEntry` world-AABB pre-reject dropped: the local-space AABB
    clip already runs right after a ~18-flop transform — not worth it.
  Verified: core+render+scene-demo suites green (incl. golden hashes);
  World CPU FPS unchanged (~8, few sprites there — the win scales with
  sprite count); GPU smoke run clean. Sprite-heavy live check needs a
  manual Tab to the Sprites/Doom scene.
- **PF.9 — LANDED** — C3, extended to the REAL hot path (the audited
  `SamplerShadow` turned out effectively dead on the facade path — the
  scene occluder is always chosen when shadows are on):
  - `BrickCache` gains a public occupancy view
    (`brick_occupied_at`/`super_occupied_at`, `Option<bool>`; `None` =
    no map cached → caller steps densely).
  - `SceneOccluder::occluded_in_grid`: three-tier empty-space skip —
    ABSENT chunk jumps its whole 128×128×256 box (the PF.6 deferral),
    empty 64³ super-brick / 8³ brick (mip-0 maps, ensured by render
    phase A before the occluder build) jump their boxes. Global 8/64
    alignment coincides with the chunk-local maps (chunk dims are
    multiples of 64; `>>` floors negatives). Mid-LOD grids without
    mip-0 maps fall back densely.
  - `SamplerShadow::occluded` (core) gets the same super/brick skip.
  - Bit-compatibility: skips cross only guaranteed-empty boxes; landing
    reuses `cell_walk_skip`'s exit-axis-pinned cell (leak-free); the
    step budget is consumed in Manhattan cell distance so the
    `SHADOW_MAX_STEPS` truncation fires at the identical point. Pinned
    by an 8000-ray equivalence test vs the pre-PF.9 dense loop kept as
    oracle (caves, thin walls, absent-chunk gap, axis-parallel rays,
    budget-truncated rays, with AND without brick maps).
  Measured (throwaway release bench, 40k sun-like 512-unit rays, cave
  terrain + pillars): 5.84 → 2.51 µs/ray (2.3×, identical 9453 hits) —
  on top of PF.6's gains. Full workspace green.
- **PF.10 — LANDED** — H2: sprite cull/bin/colmul caching:
  - Identity-colmul fast path (the big one): the facade never sets
    per-instance colmul (sprites render flat-lit), yet every frame
    packed + uploaded 512 u32 = 2 KiB per visible instance. While no
    `set_instance_colmul` call has happened (`any_colmul == false`) the
    per-visible tables are neither built nor uploaded — the buffer holds
    a lazily-written identity fill (re-filled on growth only).
  - Whole-cull skip: `cull_bin_upload` keys on (frustum bits, screen,
    tile, lod) + "registry unchanged" (every mutating method —
    append/remove instance, transforms, set model/colmul, update/add/
    remove model, compact — nulls `last_cull`); a matching key returns
    the cached counts with zero CPU work and zero uploads (the buffers
    already hold this frame's data). Static camera + static sprites ⇒
    free; moving camera still pays the (now colmul-free) cull.
  - `CullScratch`: the 6+ per-frame Vecs (visible/boxes/colmul/counts/
    tile_ranges/tile_instances/cursor) become reused workspace fields.
  - `storage_dst_u32` buffers gain COPY_SRC (free) so harnesses can read
    contents back.
  Verified: new device gate `tests/sprite_cull.rs` (same-key call returns
  the cached result; transform update invalidates and drops the moved
  instance; colmul readback = identity pattern on the fast path and the
  packed custom table after `set_instance_colmul`); full GPU+render
  suites; 45 s live demo on the iGPU (`ROXLAP_GPU_POWER=low` — the dGPU
  present path is driver-broken, see the NVK memo), no panics.
- **PF.11 — LANDED** — S1:
  - `Grid::bake_lightmode[_with_ao]`: wave-parallel estnorm-cache phase
    (waves of `current_num_threads` caches build against the immutable
    grid; the apply phase stays outer-serial — it's already row-parallel
    inside `apply_lighting_with_cache`). Byte-identical by construction.
    Measured: 16-chunk full bake 32.9 → 10.6 ms (3.1×). The scene-demo's
    duplicated per-chunk bake driver now delegates here (keeping its mip
    pass), so scene builds get the speedup too.
  - `Grid::bake_lightmode_bbox(lo, hi, mode)` — the runtime-edit rebake
    primitive: pads the write region by ±ESTNORMRAD internally (pass the
    geometric edit extent, `update_lighting`'s convention), re-bakes the
    touched strip of every chunk the padded region crosses (xy AND
    stacked-z seams), bumps touched chunk versions. Measured 0.089 ms
    per bullet-sized rebake (~23× cheaper than a full-chunk bake). Mips
    intentionally not regenerated (near-field reads mip 0; PF.12 owns
    partial remip). Pinned by `bbox_rebake_matches_full_rebake`: edit
    straddling a chunk seam → full re-bake vs bbox re-bake byte-equal.
    `ESTNORMRAD` re-exported from roxlap-core.
  - `StreamingBakeTracker` gains a per-grid pending FIFO + per-frame
    budget (default 4, `with_budget(usize::MAX)` = old behaviour): a
    streaming burst's 5–25 full-chunk rebakes (~175 ms single-frame
    hitch) spread over frames at ≤ ~28 ms each; a queued chunk renders
    unbaked for those few frames.
  Full workspace green; live streaming runs on both backends (GPU via
  iGPU `ROXLAP_GPU_POWER=low`) clean.
- **PF.12.a/.b — LANDED** — H5 (part 1 of 2):
  - **.a dirty-extent plumbing**: `Grid` tracks a per-chunk
    `DirtyExtent` (`Full` | inclusive chunk-local `Bbox`) alongside
    versions. `bump_chunk_version` marks Full;
    `bump_chunk_version_bbox` merges extents; `take_chunk_dirty`
    consumes. Edit paths (`set_voxel/rect/sphere`) and
    `bake_lightmode_bbox` report extents padded ±1 voxel — an edit
    rewrites the exposed-face records of ADJACENT columns too (found
    empirically: a carve's wall repaint relocates the r+1 ring). The
    GPU facade's `refresh_dirty` (now `&mut Scene`) consumes extents at
    chunk sync (the .c partial upload will key on them).
  - **.b incremental re-mip**: `Vxl::remip_bbox(x0,y0,x1,y1,max)` —
    byte-identical to `generate_mips` (shared `rebuild_mips` driver);
    clean columns memcpy their old level bytes (`MipReuse` in
    `build_mip_level`), only the ±1-padded dirty column rect re-runs
    the 2×2 downsampler per level. Equivalence test: 2 alternating
    edit+remip rounds vs full rebuilds, corner edits, voxalloc-scattered
    pools — data/offsets/mip tables byte-equal.
  - **Latent pool-corruption landmine FIXED**: every mip rebuild left
    `vbit` stale, marking the fresh level segments as FREE — the next
    voxalloc edit silently clobbered level bytes (old code self-healed
    at the next full re-mip; renders in between read corrupt mips, and
    `remip_bbox` trusts old level bytes outright). `rebuild_mips` now
    ends with `resync_vbit()` (slng-walk per column — the
    `reserve_edit_capacity` window trick is invalid for scattered
    pools); interior gaps become legitimately allocatable.
  - cave-demo's carve worker uses `remip_bbox` over the batch bbox
    (padded by FIRE_RADIUS + ESTNORMRAD) instead of the full
    `generate_mips` that motivated its clone+worker design.
  Measured (release, 128² scene chunk, 30 carve rounds): 0.78 →
  0.26 ms/round (3×), byte-identical; the ratio scales with chunk area
  (clean columns are memcpy) — cave-scale chunks gain far more. Full
  workspace green.
- **PF.12.c — LANDED** — partial GPU chunk refresh (PF.12 closed):
  - `GpuSceneResident::refresh_chunk_partial(queue, scene, chunk, vxl,
    x0, y0, x1, y1)`: re-derives ONLY the dirty column rect per mip
    (per-column `decompress_column` reuse via the vsid=1 trick) and
    writes row-runs of occupancy (textured + solid) and colours.
    Colours are prefix-packed, so the method verifies every dirty
    column's colour COUNT against a new CPU mirror of the slot's
    `color_offsets` window (`color_offsets_shadow`, ~87 KB per 128²
    chunk, maintained by upload/refresh/evict); any count change —
    or a slot-identity/mip/stride mismatch — returns `false` with
    NOTHING written and the caller falls back to the full path.
    Count-stable extents (lighting re-bakes via `bake_lightmode_bbox`,
    recolours) now upload a few KB instead of decompressing + rewriting
    the whole ~1–2 MB chunk ladder.
  - Facade `refresh_dirty` restructured into candidate → take-extents →
    refresh phases: a budgeted candidate with a `DirtyExtent::Bbox` and
    a previously-synced slot takes the partial path (extent padded ±1
    again, mirroring `remip_bbox`'s belt); everything else takes the
    full path unchanged.
  - Resident storage buffers gain COPY_SRC (free) for test readback.
  Verified: device gate `tests/partial_refresh.rs` — a count-stable
  recolour partially refreshed is byte-identical (occupancy pages +
  color_offsets + colors readback) to a full refresh of the same edit;
  a count-changing carve is refused with a bit-exact untouched
  resident. Full workspace green; live streaming demo run (iGPU) clean.
- **PF.13 — LANDED** — leftovers batch (closes the PF series):
  - **C7 x-scissor**: `OpticastSettings` gains `x_start`/`x_end` +
    `with_x_range` (DDA pixels are fully independent; the historical
    full-width-only constraint came from the deleted voxlap radar's
    `angstart`). Both DDA pixel loops iterate the x-range; the per-grid
    screen scissor now clips BOTH axes, so a small on-screen grid (the
    ship) renders its true rect instead of full-width rows. Covered by
    the existing scissor=false byte-identity regression.
  - **H9 mutation counter**: `Grid::mutations` (u64, bumped by every
    chunk-content mutation: `bump_chunk_version[_bbox]`, `ensure_chunk`
    insert, generator/streaming installs, evictions, snapshot restore)
    + `Grid::mutation_counter()`. Consumers skip O(all chunks) scans on
    quiet frames: `ensure_dda_bricks` early-outs via a
    `(counter, requested_mip) → effective_mip` memo (measured 11.1 µs →
    ~0 per quiet frame at 256 chunks), and the GPU facade's
    `refresh_dirty` skips the whole per-chunk version poll +
    stale-eviction scan per grid (`grid_mutations` snapshot, recorded
    ONLY when a pass completes without budget break / failed refresh).
  - **S4 clip prefix sums**: `duration_prefix_sums` + `frame_at_prefix`
    (binary search, results identical incl. zero-duration frames —
    equivalence test sweeps all loop modes) in roxlap-formats;
    `ClipClock` caches the prefix vector so every per-frame `tick` is
    O(log n) instead of re-summing the timeline.
  - **S5 picking demo**: registers the cursor/marker models ONCE per
    scene activation and moves the cursor via
    `set_sprite_instance_transform` (markers spawn incrementally on
    click) — was a full `set_sprites` registry rebuild + 2 kv6 clones +
    GPU re-upload EVERY frame.
  - **G8**: deleted the dead per-step bounds computation in
    `grid_dda.wgsl`'s outer loop (empty `if` body — pure wasted ALU).
  - **H7-lite**: line/image overlay bind groups cached by depth-buffer
    identity (wgpu 29 identity `PartialEq`); image BGs keyed by image id
    and evicted on drop/slot-reuse. A static HUD costs zero bind-group
    creations per frame (was one per quad per frame).
  - **Low-power default resolution**: new `GpuRenderer::low_power()`
    (adapter `device_type != DiscreteGpu`) surfaced as
    `SceneRenderer::is_low_power()` (CPU backend ⇒ always true); the
    scene-demo halves its fixed render-grid default to 430×260 on such
    renderers (the 860×520 default targeted a discrete card and ran
    ~6 FPS on CPU/iGPU). Explicit `ROXLAP_RENDER_RES` and live HUD
    edits always win; the halving only replaces the untouched default.
  Verified: full workspace green (oracle hashes unmoved), clippy clean,
  25 s live GPU run (iGPU `ROXLAP_GPU_POWER=low`, streaming World) + 15 s
  CPU run clean.
  **Consciously deferred** (small or risky relative to payoff): G7
  per-voxel material byte in the sprite path, `pow`→`exp2` in shaders,
  sprite-tile translucent bit, S3 billboard lazy cache fill, H7 full
  submit-merging of overlay passes, C8 render-bounds caching, and the
  optional H9 extension (skip streaming eviction until the camera moves
  ½ chunk).

Verification per stage: `cargo test` workspace (CPU golden hashes must not
move unless the stage says so), the GPU headless diff harness for shader
stages, plus a manual FPS spot-check in scene-demo/cave-demo where relevant.
