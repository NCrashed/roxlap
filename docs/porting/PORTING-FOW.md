# roxlap — fog of war / field of view (Stage FW)

Entry doc written 2026-07-17 at workspace 0.29.0 + uncommitted OC tail.
This is the **entry doc** for the fog-of-war stage — tag **FW**.

## Status — FW.0 + FW.1 + FW.2 + FW.3 + FW.4 LANDED (2026-07-18); FW.5 not started

- **FW.0** — `roxlap-scene/src/fow.rs`: `FogOfWar` + `VisionConfig` +
  `DeckBand` + `LightGate` + `FowObserver` (+ `CellState`, all
  re-exported); per-deck sparse 128×128 mask tiles (byte = 2-bit state
  + 6-bit intensity), classic 8-octant recursive shadowcast,
  lazy per-cell opacity/light tiles keyed by an FNV mix of
  `chunk_version`s + config generation, LOS recompute keyed on
  (cell, quantized facing, deck, `mutation_counter`, config gen),
  fade/decay state machine with float-truth transitions (byte is the
  rounding — no dt-quantization stall), heard blobs with TTL +
  Visible-priority, `mask_version` for the FW.3 upload gate,
  `for_each_live_cell` for the FW.1 twin sync. 10 unit gates: cone /
  peripheral / occlusion / cache invalidation on edit / gradual fade /
  decay-to-floor + re-see reset / light gate incl. emissive key /
  deck independence / heard reveal + fade / mask-version quiescence.
  Fixture note: inserting over an already-solid voxel keeps the OLD
  colour (`set_cube` is a geometric no-op there) — light-gate fixtures
  must author lit/dark regions disjoint.
- **FW.0 review round (2026-07-17)** — 6 correctness + 4 perf, all
  landed (+7 regression gates, 17 total):
  1. **Light gate read the ceiling, not the floor** — `sample_cell`
     took the first solid from `min(z_top, eye_top)`, so a `DeckBand`
     whose `z_top` includes the ceiling plate sampled the dark
     underside of the deck above → fully-lit rooms stayed Unseen
     forever. Floor surface is now the first solid at `z >= eye_top`.
  2. **Full `Grid::bake` was invisible to version-keyed caches** —
     `bake_u32` rewrote brightness bytes without bumping chunk
     versions (unlike `bake_bbox`). Fixed in `chunks.rs` (helps the
     GPU dirty-chunk poller too, not just FW). Turn the ship's lights
     on + re-bake → rooms now re-light.
  3. **Stream-in seen through fresh walls** — a generated chunk keeps
     `chunk_version == 0` (absent value), so opacity tiles cached
     while absent survived materialisation. Tile/edit keys now fold in
     chunk PRESENCE (`chunk(idx).is_some()`), flipping on
     absent→present at v0.
  4. **Ceiling emissive became a wall** — the emissive early-return
     marked the cell blocked when the lamp sat above the eye band,
     carving Unseen wedges. Emissive now sets a lit-override flag and
     never blocks; blocked stays eye-band-only.
  5. **Heard stomped just-set Visible** — `update_heard` ran after
     `recompute_los` and demoted fallen-out heard cells unconditionally;
     turning toward a noise flipped the cell Visible then instantly
     back to Memory. Demotion + stamp now skip cells the active deck
     sees.
  6. **`quantize_facing` sentinel collision** — `-1` was both "no
     facing" and a legit ~-0.18° bucket. Sentinel moved to `i32::MIN`;
     buckets `rem_euclid(2048)` (also folds the ±π wrap).
  Perf: (1) LOS key hangs on a bounded `local_edit_key` over the view
  radius's chunks, not the grid-wide `mutation_counter` (far-side
  debris/streaming no longer forces a radius-96 shadowcast); (2)
  `sample_cell` borrows the chunk once per `chz` instead of a per-voxel
  `voxel_split`+HashMap probe (~340k on a first-fill tile); (3) tile
  key memoised per tile-transition in `LosScan`, not rehashed per
  visited cell; (4) heard blobs precompute their cells once at `hear()`
  and the per-tick union rebuilds only on blob add/expiry (no per-tick
  `sqrt` fan-out).
- **FW.1** — known-twin grid (`fow.rs` `FowTwin` + two `Grid` flags).
  `Grid::render_excluded` (real grid: simulated, never drawn — skipped
  by BOTH the primary render and the shadow occluder, so unseen
  geometry casts no shadow) and `Grid::presentation_only` (twin: drawn,
  never queried). Two centralised `Scene` iterators enforce the split:
  `render_grids[_mut]` (`!render_excluded`) drive CPU render / Phase-A
  caches / occluder / GPU residency / CPU sprite-cutaway; `query_grids
  [_mut]` (`!presentation_only`) drive raycast / resolve_voxel /
  cutaway_hides_point / collision / water / audio thickness / streaming.
  With no FW grids both yield every grid ⇒ byte-identical. `FowTwin::
  attach` registers the twin + sets the flags; `sync(scene, fow)`
  mirrors the real grid's render config (transform / render_sky / mip
  override / lod / z_clip) then copy-on-first-seen / re-syncs (whole
  `Vxl` clone) every chunk under a live cell whose `(version, present)`
  signature changed; `detach` restores. Twin excluded from snapshots
  (derived state — NO wire change; host re-arms fog after load, real
  grid's `render_excluded` defaults off). 8 gates: render/query split,
  excluded-real-still-raycasts, copy-on-first-seen, edit-behind-wall
  invisible-until-seen, memory-frozen-after-leaving, chunk-granular leak
  (documents the accepted v1 bound), detach, snapshot-excludes-twin.
  Audit: 28 grid-iteration sites swept (found one the entry doc missed —
  `cpu.rs` sprite-cutaway — routed through `render_grids`).
- **FW.1 review round (2026-07-17)** — 7 correctness/lifecycle + 3 perf,
  all landed (+6 regression gates, 31 fow total):
  1. **Empty twin fell out of GPU residency forever** — `upload_scene`
     skipped a chunkless static grid from `grid_ids` yet still recorded
     its id as resident, so `resident_matches_scene` kept matching and
     `refresh_dirty` never installed the twin's later chunks (whole
     fogged world invisible on GPU). Twin now counts as "dynamic"
     (`gpu_residency_hint.is_some()`), registered even when empty — the
     streaming-grid pattern.
  2. **Twin GPU pool didn't grow with exploration** — a static grid
     sizes its chunk-space region AND modular slot pool from the
     chunks resident at first upload; later-explored chunks (a tall
     ship spans many `chz`) fell outside → unaddressable or aliased
     occupied slots (rooms flicker), CPU/GPU divergence. New
     `Grid::gpu_residency_hint (origin_chunk, chunks_dims)` set by
     `FowTwin` to the REAL grid's full chunk bbox; `upload_scene` sizes
     region + `default_pool_dims` from it. (colors_stride truncation
     for denser-later chunks is inherited from the streaming path — a
     shared pre-existing limitation, noted not newly introduced.)
  3/7. **Real-grid eviction wiped twin memory / phantom versions** —
     sync keyed copies on `(version, present)`; an evicted real chunk
     flipped present→false and the old sync REMOVED it from the twin
     (geometry in view vanished) and bumped a version for a chunk that
     exists nowhere (unbounded `chunk_versions`/`chunk_dirty` growth +
     defeated PF.13 quiet-skip). Sync now NEVER removes for an absent
     real chunk — the last-seen copy stays as memory — and only records
     the sig when a memory copy exists.
  4. **Cutaway footprint divergence** — `cutaway_hides_point` was on
     `query_grids` (real, full footprint) while the sprite cull reads
     `render_grids` (twin, seen chunks); CA.4 requires one rule.
     `cutaway_hides_point` moved to `render_grids` (both = the drawn
     twin). Byte-identical with no FW.
  5. **Snapshot silently killed fog** — `sync` now returns `#[must_use]
     bool`: `false` when either grid is gone (post-load / rollback),
     the explicit "host must re-arm" signal the API lacked.
  6. **sync didn't clear twin.billboards** — a copying sync now nulls
     the Far impostor cache (the S7.4 contract every other chunk-set
     path honours), else a frozen first-look impostor at distance.
  Perf: (P1) re-sync bumps the twin with the real edit's drained bbox
  (`take_chunk_dirty` on the never-rendered real grid — also stops that
  orphaned map leaking) instead of `Full`, so the GPU re-uploads only
  the delta; first-seen still `Full`. (P2) quiet-frame early-out keyed
  on `(mask_version, real mutation_counter)` skips the ~πr² live-cell
  rescan when nothing changed. (P3) the exclusion rule is single-sourced
  in `Grid::renderable` / `Grid::queryable`; `query_grids_mut` (was dead)
  now drives `pump_streaming_sync`; snapshot routes through
  `query_grids`. Full-chunk CPU clone on copy stays per decision 2 (the
  column-wise copy is the deferred follow-up).
- **FW.2** — CPU styling. New in `roxlap-core::dda`: `FowStyler` trait
  (`Send + Sync`, one `verdict(x,y,z)` per hit) + `FowVerdict`
  (`Hide` | `Show { dynamic, dim, desaturate }`, `LIVE` const) +
  `fow_style` (dim×desaturate colour op, identity at `{1,0}`) +
  `DdaEnv::fow`. In `cell_walk_skip`'s hit branch the verdict is
  evaluated once per SOLID hit inside the occupancy `.filter` (the
  cutout's "price the rule on real hits only" pattern): `Hide` reads as
  air (marcher continues — hides within-chunk unseen geometry the twin
  copied for a neighbour); `Show { dynamic:false }` forces the baked
  `shade` path (memory/heard never relit by a live rig) then
  `fow_style` dims+desaturates before fog. `roxlap-scene::FowRender`
  implements the trait over a `FogOfWar`: Unseen→Hide, Visible→Show
  (dynamic, dim toward memory at the cone edge by intensity so the FOV
  boundary is a smooth taper — NO dither), Memory/Heard→Show (baked,
  dim×decay + desaturate). Wired per-grid via `ComposedFrameParams::fow`
  / `FrameParams::fow` (`(GridId, &FogOfWar)`) — set only for the twin
  grid; every other grid and `None` are byte-identical. mip-N hits shift
  `cellc << mip` to the mip-0 lookup. Gates: 5 core (fow_style unit,
  Hide-vanish, dim/desaturate matches helper, memory-skips-dynamic-rig,
  LIVE byte-identity) + 3 scene (verdict state mapping, memory
  baked+dim, composed `fow=None` byte-identity).
- **FW.2 review round (2026-07-18)** — 8 correctness (info-leak cluster
  + visual) + 2 perf, all landed (+6 gates):
  1. **Hidden walls cast shadows** — the Hide verdict was primary-ray
     only; shadow marches (single-grid `SamplerShadow` AND the cross-grid
     `SceneOccluder`/WorldShadow used by composed renders) blocked light
     on unseen geometry → the shadow silhouette leaked. Both are now
     fog-filtered: a `Hide` cell occludes nothing. `SceneOccluder::build`
     takes `Option<(GridId, &FogOfWar)>`, threaded from `render.rs`
     (terrain) and `cpu.rs` (sprite-receive).
  2. **mip-N low-corner leak** — a coarse cell was classified by its low
     corner (straddling a deck ceiling → out-of-band → leak; a Visible
     corner → pop). Now sampled at the coarse cell's CENTRE
     (`(cellc << mip) + half`), primary + shadow.
  3. **Far-LOD impostor bypassed fog** — the twin mirrored the real
     grid's `lod_thresholds` incl. Far, and Far blits a raw impostor. The
     twin mirror now forces `r_mid = INFINITY` (never Far — stays on the
     fog-aware Near/Mid DDA at any distance).
  4. **Out-of-band z rendered LIVE** — a z in NO deck band (inter-deck
     hull, sub-floor a chunk-granular copy dragged in) defaulted to
     `LIVE`. Now `Hide` (untracked = unknown = hidden).
  5. **Visible→Memory seam pop** — Memory used a different dim curve, so
     a cone-centre cell (dim 1) jumped to `memory_dim` the frame it
     flipped to Memory. Visible and Memory now share ONE
     `memory_dim + (1-memory_dim)·t` dim + `memory_desaturate·(1-t)`
     curve keyed on intensity; only `dynamic` differs → continuous, the
     fade_out taper carries it down.
  6. **Memory lost emissive** — `dynamic:false` had replaced the emissive
     branch with baked `shade` → a remembered crystal went dark while its
     baked halo stayed. Emissive is intrinsic and now ALWAYS wins;
     `dynamic:false` gates only the dynamic rig.
  7. **GPU silently dropped `fow`** — the GPU backend now `debug_assert`s
     + warns once when `FrameParams::fow` is set (CPU-only until FW.3).
  8. **Vacuous composed gate** — the byte-identity test compared two
     `fow=None` renders. Replaced with a real one: an all-Unseen fog
     HIDES the rendered twin (sky), a seen fog SHOWS it.
  Perf: (perf#2) the cheap keyhole `cut_z_mip` gate now runs BEFORE the
  fog verdict, so a hit the keyhole discards pays no classification;
  (perf#1) the per-deck mask/opacity maps use a zero-dep Fx-style integer
  hasher (Sync-safe — the styler is shared across rayon workers, so a
  faster hash, not a mutable cache) instead of SipHash. Gates: shadow no
  longer leaks (Δlum), out-of-band Hide, seam no-pop, emissive kept,
  composed hides-unseen-shows-seen.
- **FW.3** — GPU parity. One `fog_mask` storage buffer (binding 22 —
  19..21 are the conditional sprite-cast slots) with a self-describing
  17-word header (`FOG_*` word offsets: enabled / grid index / deck
  count / origin cell / width / height / memory dim+desaturate / up to 4
  `(z_top, z_bottom)` deck bands) then the deck-major row-major mask
  bytes packed 4/u32. Header + single buffer ⇒ NO change to the
  intricate `Uniforms` layout. WGSL `fow_lookup(g, cxm, cym, czm, mip)`
  (mirrors `FowRender`: LIVE unless enabled & `g == FOG_GRID`; mip-cell →
  mip-0 CENTRE `<< mip + half`; deck_for_z, out-of-band/Unseen → Hide;
  unified `dim`/`desat` taper; `dynamic = state==Visible`) + `fow_apply_
  style`. Applied in `scene_dda.wgsl`'s hit branch (`!fow.hidden` in the
  gate, emissive-wins + `dynamic`-gated rig + `fow_apply_style` before
  fog) AND in `shadow_occluded` (a Hide cell occludes nothing — the CPU
  `SceneOccluder` fix, review #1, on GPU). Disabled (word 0 == 0, a
  1-word dummy) reads one word ⇒ byte-identical (all 30 existing headless
  gates green). Host: `roxlap_scene::GpuFowMask` + `FogOfWar::gpu_mask
  (origin_cell, w, h)` flattens the sparse tiles over the twin's
  residency-hint bbox; `roxlap-render` `pack_fog_mask` builds the words +
  `GpuBackend::sync_fog_mask` uploads them via `GpuRenderer::set_fog_mask`
  (recreates the buffer → invalidates the cached bind group),
  version-gated on `(fog grid, mask_version)` and cleared when
  `FrameParams::fow` goes away. Bindings added to BOTH scene_dda BGLs +
  bind groups (main + headless) and `HeadlessSceneRenderer::set_fog_mask`.
  The FW.2 GPU `debug_assert` is gone. Gates: 2 headless (Hide→sky /
  Visible→floor / disabled byte-identical; Memory dims) + `pack_fog_mask`
  header pin + `gpu_mask` flatten. Traps: bindings 19–21 are the
  conditional sprite-cast slots (use 22); `<< mip` (not `>> mip`) for
  mip-0 coords; std140 fields dodged entirely via the storage-buffer
  header.
- **FW.3 review round (2026-07-18)** — 7 correctness (GPU mask state) +
  3 perf/hygiene, all landed. Root cause of #1/#2/#7: the version-gate
  key `(grid, mask_version)` under-specified the uploaded bytes AND
  `sync_fog_mask` ran BEFORE `upload_scene` (stale `grid_ids`).
  1. **Resize/SSAA silently disabled fog** — rebuilding the scene-DDA
     pipeline resets `fog_mask_buf` to the disabled dummy, but the key
     said "loaded". Added `GpuRenderer::scene_dda_generation()` (bumped
     per pipeline build) to the key.
  2. **Stale positional `FOG_GRID`** — the header bakes the twin's
     resident SLOT index, which shifts on a scene switch / grid add. The
     slot is now in the key AND `sync_fog_mask` runs AFTER `upload_scene`
     (current `grid_ids`).
  3. **`hint == None` ⇒ whole grid Hidden** — the `(w=0, h=0)` fallback
     still uploaded `FOG_ENABLED=1`, so every cell failed `lx < 0` → all
     Hidden (a direct-`FogOfWar` grid vanished on GPU). Now falls back to
     a DISABLED mask (LIVE).
  4. **Silent >4-deck truncate** — `pack_fog_mask` warns once when the
     deck count exceeds the shader's `FOG_MAX_DECKS`.
  5/6. **Fog-target switch / lost twin didn't clear + no guard** — a
     non-resident or hint-less fog grid now clears any resident mask
     (`clear_fog_mask`) and warns once (`warn_fog_grid_not_resident`,
     the replacement for the removed FW.2 guard), instead of leaving the
     old ship fogged / the new one unstyled.
  7. **One-frame full reveal on enable** — fixed by #2's move after
     `upload_scene`: the twin's first fogged frame has a live slot, so
     the mask uploads before the dispatch that draws it.
  Perf: (perf#1) `set_fog_mask` writes IN PLACE via `write_buffer` when
  the mask fits the buffer (an intensity fade re-uploads the same
  geometry every frame) — no bind-group rebuild / alloc; only a size
  growth recreates. (perf#2) the WGSL `fow_lookup` runs only AFTER the
  cheap `z_clip_mip`/`cut_hidden` gates. (perf#3) the header packer +
  `FOG_*` layout live in ONE place — `roxlap_gpu::fow::pack_fog_mask`
  (owner of the WGSL constants); `roxlap-render` and the headless test
  both call it (the hand-rolled test header is gone). New gates:
  `roxlap_gpu::fow` header-layout + deck-truncation; `roxlap-gpu` gains a
  direct `log` dep for the warn.
- **FW.4** — sprites + audio reveal.
  *Sprites (decision 8)*: `FogOfWar::hides_sprite(transform, world)` —
  world → grid-local cell + deck → true for Memory / Unseen / off-deck
  (shown only in Visible / Heard), the per-sprite mirror of
  `cutaway_hides_point`. **Binary hide** on both backends (no alpha fade
  in v1 — kept backend-symmetric; the fade + live-sprite ghost are
  follow-ups). CPU: OR'd into the `hidden_flags` closure in `cpu.rs`
  (was `cutaway_flags`) using `frame.fow` + the fog grid's transform. GPU:
  a `fog_hidden: Option<&dyn Fn([f32;3])->bool>` + `fog_version` threaded
  `render_scene → cull_bin_upload` (the `CullKey`/PF.10 skip cache folds
  in `fog_version`), tested against each instance's world `center` beside
  the cutaway clips; `roxlap-render` builds the closure over `frame.fow`
  (grid transform copied out — no scene borrow). Both drop the sprite
  from the visible set (and its shadow), exactly like the CA cull.
  *Audio (decision 6)*: `FogOfWar::hear_world(transform, world, loudness)`
  maps a world source to its cell/deck and stamps a heard blob;
  `roxlap_audio::hear_source(scene, fow, transform, source, listener,
  base_loudness, cfg)` folds in `source_acoustics` transmission
  (`loudness = base × transmission`) so a walled source reveals a smaller
  pocket, a buried one nothing. Gates: `hides_sprite` state mapping,
  `hear_world` map+off-deck, GPU `cull_applies_fog_hide` (headless),
  `hear_source` transmission-scaled reveal (audio).

## Goal

SS13-style *knowledge-based* visibility for third-person play: the
player renders in realtime only what the character currently sees — a
facing cone plus a small 360° peripheral radius; everything seen
before renders as it was *at the last moment it was seen* (frozen
geometry AND frozen baked light, dimmed); never-seen space is black;
audible sound sources punch temporary live "heard" pockets through the
fog (behind the back, behind walls). Per-grid opt-in (the ship gets
it, streaming terrain does not). Both backends, hash-gated, zero cost
/ byte-identical when disabled.

Non-goal: information hiding at the simulation level. The lockstep
model replicates full state to every client; FW is presentation only —
the classic RTS tradeoff, accepted. Nothing in the sim, collision,
audio, or pathing may ever read the FW mask.

## Locked design decisions

1. **Render the KNOWN twin, simulate the real grid.** A FW-enabled
   grid gets a shadow copy ("known grid"): chunks copy-on-first-seen,
   then re-synced each frame ONLY where the mask says Visible/Heard
   (v1 granularity: whole chunk, see decision 2). The renderer draws
   the known twin; the real grid is excluded from rendering; the known
   twin is excluded from collision/raycast/audio (see hazard 2). No
   second DDA pass, no per-hit data-source branch — "memory" is just
   the twin not being updated. Baked brightness lives in the packed
   colour high byte, so the twin freezes *lighting* at last-seen for
   free (an explosion re-bake in an unseen compartment stays
   invisible).
2. **v1 sync is chunk-granular, column-wise is the follow-up.** A real
   chunk re-syncs into the twin when its `chunk_dirty` extent
   intersects at least one Visible/Heard cell. Accepted v1 leak: an
   edit in the hidden part of a partially-visible chunk becomes
   visible early (bounded by one chunk). The fix — per-column RLE copy
   of only the visible columns — is deliberately deferred (FW.2 notes
   the seam so the follow-up is mechanical).
3. **Per-cell mask, host computes, backends apply** (the OC pattern).
   One byte per mip-0 XY column per deck: 2 bits state
   (Unseen / Memory / Visible / Heard) + 6 bits intensity (fade /
   hysteresis — boundaries lerp over a few frames, no flicker on
   turn). Computed by a 2D recursive shadowcast in GRID-LOCAL frame
   (ship rotation/movement free, S5 precedent) over a per-cell opacity
   bitmap ("solid in the deck's eye z-band?") cached and invalidated
   by `chunk_versions`. Persistent state lives in a scene-side
   `FogOfWar` struct (owns the mask layers, the known twin id, the
   config) with an explicit `update(&mut scene, observer, dt)`;
   renderers read it as grid state, the GPU re-uploads the mask
   storage buffer only when its version bumps.
4. **Styling per state, applied in the hit branch after CA/OC gates.**
   Visible: full shading (bake + dynamic sun/points). Memory: baked
   component only × configurable dim factor + configurable cold
   desaturation, dynamic light contributions scaled to zero by the
   mask (a torch in the visible zone must not relight memories).
   Unseen: background colour, light loop skipped entirely (perf win).
   Heard: live data (decision 6) with the Memory styling. Boundary =
   smooth intensity taper — **NO dither anywhere** (OC round-1
   lesson, and explicitly rejected for FW).
5. **Light gates vision (configurable).** During shadowcast a cell
   only reaches full Visible intensity if its floor-surface baked
   brightness (+ emissive, EV materials count as lit) clears a
   configurable threshold with a soft knee; below it the cell caps at
   Memory. `VisionConfig::light_gate: Option<LightGate>` — None =
   geometric visibility only. Dynamic point lights are NOT sampled by
   the gate in v1 (bake + emissive only; noted as follow-up).
6. **Audio reveal.** Host feeds heard events (`source_pos`, loudness —
   convenience helper wraps `source_acoustics` so transmission/gain
   are engine-computed); each event above threshold stamps a Heard
   blob of radius r(loudness) around the source with a fade-out
   timer. Heard cells sync the known twin like Visible ones — hearing
   shows the *current* state, dimmed. No cutout needed: the camera is
   overhead, OC already solves camera occlusion.
7. **Memory decays.** Memory-cell intensity fades toward a
   configurable floor at a configurable rate (frame-rate independent,
   dt-driven); intensity modulates the dim factor. Re-seeing resets
   it.
8. **Sprites follow the mask.** Instances whose world position maps to
   a Memory/Unseen cell of a FW grid are hidden (the CA
   `cutaway_hides_point` footprint pattern, extended); near the
   boundary `alpha_mul` fades to avoid pop. Ghost "last-seen actor"
   rendering is explicitly NOT v1 — own go/no-go later (OC.4
   precedent). Billboards: out of scope — FW grids don't use the
   Far-LOD billboard path (user decision).
9. **v1 vision is per-deck 2D.** Mask layers are per deck (memory
   persists for every deck); shadowcast runs only on the active deck's
   z-band (the same levels as CA `DECK_CLIPS`, deck selection follows
   the HEAD — CA round-3 lesson). Floor holes / shafts / windows
   between decks (true 3D visibility) are an explicit follow-up, like
   OC.4 was.
10. **Presentation-only, per-grid opt-in, disabled ⇒ byte-identical.**
    No FW state defaults to existing behaviour and reproduces every
    golden hash on both backends (the standing invariant; the mask
    gate must be a one-compare sentinel in the hot loop, the
    CA/OC pattern).

## Substages

- **FW.0 — mask core (no rendering).** `roxlap-scene` `fow` module:
  `FogOfWar` + `VisionConfig` (cone half-angle, range, peripheral
  radius, deck z-bands, light gate, decay rates, dim/tint factors),
  per-deck cell layers, recursive shadowcast, opacity cache keyed by
  `chunk_versions`, state machine with intensity fade + hysteresis,
  light gate (decision 5), decay (decision 7). Pure CPU, no render
  change. Gate: unit tests — cone/peripheral classification, wall
  occlusion, opacity-cache invalidation on edit, decay curve, light
  gate threshold, deck-layer independence.
- **FW.1 — known twin.** Copy-on-first-seen chunk materialisation into
  a twin grid (same transform, registered in the scene), per-frame
  chunk re-sync where dirty ∩ Visible/Heard (decision 2), render/sim
  exclusion flags (hazard 2). Gate: edit-behind-wall test (mutate
  unseen geometry → render hash of the twin unchanged; walk over →
  hash updates), re-bake-behind-wall test (frozen light), sync-leak
  bound test documenting the v1 chunk granularity.
- **FW.2 — CPU styling.** Mask into `DdaEnv` (the `z_clip`/`cutout`
  precedent); per-hit state lookup + dim/desaturate/dynamic-light
  scaling in the shading path; Unseen skips the light loop. Gate:
  golden fixtures per state, boundary-taper determinism, disabled
  byte-identity, vws-scaled grid pin (OC hazard-3 precedent).
- **FW.3 — GPU parity.** Mask storage buffer (+ version-bump upload),
  WGSL apply next to the `z_clip_mip`/cutout gates in `scene_dda.wgsl`
  hit branch, per-grid enable lane. Headless gates in
  `roxlap-gpu/tests/scene_render.rs` mirroring FW.2 (classification
  agreement away from the taper band + exact styled colours — the
  CA.3/OC.2 harness patterns wholesale).
- **FW.4 — sprites + audio reveal.** Sprite hide/fade by mask
  (decision 8) on both backends; heard events API + `source_acoustics`
  convenience wrapper + Heard blob stamping/fade (decision 6). Gate:
  sprite-behind-fog headless pin; heard blob reveals a live edit
  behind a wall then fades back to Memory.
- **FW.5 — demo + book + CHANGELOG.** Boarding grows FW: `F` toggle,
  config sliders (cone angle/range, light gate on/off, decay), a
  scripted "noise behind the wall" source to show the heard pocket;
  composition with CA deck-follow + OC keyhole is the demo's point.
  Book: rendering chapter "Fog of war" section (when to use which:
  CA = decks, OC = third person camera, FW = knowledge); demo tour
  row; CHANGELOG. Next cut minor (additive APIs). Gate: user visual
  pass.

## Hazards

1. **Twin-grid identity leaks.** The known twin must be excluded from
   `Scene::raycast`, collision queries, `SceneOccluder`, audio
   marches, and streaming/billboard bookkeeping, while the real grid
   must be excluded from BOTH render passes (primary AND shadow — a
   sun shadow cast by unseen real geometry is an info leak; shadows
   must come from the twin). Needs two explicit per-grid flags
   (`render_excluded` / `presentation_only`) checked at every grid
   iteration site — audit them all in FW.1, pin each with a test.
2. **Chunk-granular sync leak (accepted v1).** Decision 2's bound is
   one chunk; keep the sync-site column-shaped (iterate visible cells
   → owning chunks) so the column-wise follow-up only swaps the copy
   primitive, not the scan.
3. **Mask frame vs multi-grid scenes.** The mask is grid-local to ONE
   FW grid; other grids (sea, debris) render normally. Sprite hiding
   needs world→FW-grid mapping (the `cutaway_hides_point` transform
   path); actors outside the FW grid's footprint are never hidden —
   decide in FW.4 whether space outside the hull is "always visible"
   (recommended: yes, it's space).
4. **Shadowcast cost.** ~π·range² cells per update (range 100 → ~31k)
   — fine, but recompute only when observer cell/facing/deck/opacity
   version changed; decay ticks touch only non-Unseen cells already
   tracked in a dirty list.
5. **DT debris / destruction on a FW grid.** `voxel_debris` spawns
   real-grid edits + debris sprites; an unseen explosion must neither
   update the twin (decision 1 handles it) nor show its debris sprites
   (decision 8 handles it) — pin both in one FW.4 test.
6. **Emissive + light gate.** Decision 5 samples bake + emissive; a
   pitch-dark corridor with an emissive strip must stay visible —
   fixture in FW.0. Dynamic-only light sources (a carried torch) do
   NOT open vision in v1; document in the book section.
7. **Heard blob vs deck layers.** A source on another deck stamps that
   deck's layer, not the active one; the active-deck render shows
   nothing but audio still plays — correct, but pin it so nobody
   "fixes" it into a cross-deck leak.
8. **Intensity taper vs posterize.** The RP posterize runs in resolve
   after the march; the FW taper is per-cell intensity, coarse enough
   not to moiré — but verify with posterize on in FW.5 (OC hazard-6
   precedent).
9. **Observer in motion between decks.** Deck-band switches follow the
   head (decision 9); during stair transit the previous deck's layer
   flips Visible→Memory — hysteresis (decision 3) must cover the
   1–2 frame flicker window; pin with the Boarding stairs.

## Code map (as of 2026-07-17)

Scene — `crates/roxlap-scene`:
- `src/lib.rs:580-639` `Grid::chunk_versions` / `chunk_dirty` / `mutations` — sync triggers + opacity-cache keys (FW.0/FW.1)
- `src/lib.rs:662` `Grid::z_clip` — per-grid state precedent; FW flags land beside it
- `src/lib.rs:752` `Grid::cutaway_hides_point` — sprite footprint-hide pattern to extend (FW.4)
- `src/occluder.rs:60-75` — shadow-march grid iteration; twin/real exclusion audit site (hazard 1)
- new `src/fow.rs` — `FogOfWar`, `VisionConfig`, mask layers, shadowcast (FW.0)

CPU — `crates/roxlap-core`:
- `src/dda.rs:85-95` `DdaEnv` (`z_clip`, cutout) — mask ref + enable sentinel (FW.2)
- `src/dda.rs:159-227` `CpuLights` — dynamic-light application to scale by mask (FW.2)

GPU — `crates/roxlap-gpu`:
- `shaders/scene_dda.wgsl:1036-1381` hit branch — apply after the `z_clip_mip` + cutout gates (FW.3)
- `src/lib.rs:862-905` per-grid camera + `Uniforms` — enable lane + mask buffer binding (FW.3)
- `src/sprite_model.rs:2030-2032` CA footprint cull — sprite mask check joins here (FW.4)
- `tests/scene_render.rs` — CA.3/OC.2 harness patterns to clone

Facade — `crates/roxlap-render`:
- `src/lib.rs:929-995` `FrameParams` — no new per-frame field expected (FW is grid/scene state, decision 3); styling knobs live in `VisionConfig`

Audio — `crates/roxlap-audio`:
- `src/lib.rs:204-270` `path_thickness[_weighted]`, `source_acoustics` — heard-event convenience wrapper input (FW.4)

Demo — `crates/roxlap-scene-demo`:
- `src/scenes/boarding.rs:113-221` — FW demo host: deck follow, `K`/`V`/`C` hotkeys to extend with `F` (FW.5)
