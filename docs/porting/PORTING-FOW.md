# roxlap — fog of war / field of view (Stage FW)

Entry doc written 2026-07-17 at workspace 0.29.0 + uncommitted OC tail.
This is the **entry doc** for the fog-of-war stage — tag **FW**.

## Status — entry doc; FW.0..5 not started

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
