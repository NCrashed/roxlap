# PORTING-WATER.md — water + swimming (Stage WT)

Entry doc written 2026-07-13, right after the PW platform wave closed
(workspace 0.28.0 + unreleased PW work). This is the **entry doc** for
the water stage — tag **WT**. A fresh-context session should read it
top to bottom before touching code. Recon: one thorough sweep
2026-07-13 (CC.4 hook, CharacterBody internals, Volumetric paths,
audio absorption, frame/post hooks — findings below are verified
file:line facts, not guesses).

## Status — OPEN (scope locked by user 2026-07-13)

- WT.0 — LANDED 2026-07-13: `roxlap-scene/src/water.rs` —
  `WaterVolume { lo, hi }` (inclusive grid-local voxel corners,
  normalised in `new`; continuous span `[lo, hi+1)`; surface plane
  `z = lo.z`), `Grid::water_volumes` (pub Vec, `bake_lights`-style) +
  `add_water_volume` + `water_depth_local` (deepest across
  overlapping volumes), `Scene::water_depth_at(world) ->
  Option<(GridId, f64)>` in WORLD units (via
  `streaming::world_to_grid_local_pos`, depth × vws on the way out) +
  `in_water`. Snapshot wire v2 → **v3**: trailing
  `GridSnapshot.water_volumes` (corners re-normalised on load —
  untrusted bytes), frozen `GridSnapshotV2`/`SceneSnapshotV2` shadow
  shapes, `snapshot_v3.rxs` fixture generated + checked in, v1/v2
  fixtures untouched and asserted to restore DRY. 9 water unit tests
  (surface/bottom half-open bounds, overlap-max, identity + vws=0.5 +
  90°-yaw world queries, dry-scene, swapped-corner equivalence,
  equal-depth tie-break) + wire tests 6/6. Scene tests, clippy, fmt,
  workspace rustdoc green; no downstream `GridSnapshot` literals
  existed.
  **Review round (user, 2026-07-13) — 8 findings closed:** corner
  normalisation moved INTO `depth_local` (pub fields made an
  inverted volume dry live but wet after load — live must equal
  restored; load now restores verbatim); `water_depth_at` rewritten
  as one `max_by` with an explicit smallest-`GridId` tie-break
  (HashMap iteration made equal-depth winners non-deterministic —
  WT.2/3 hang state off the id); the envelope-version assert pins
  the LITERAL `3` again (comparing against `SNAPSHOT_VERSION` was a
  tautology); the v3 fixture slimmed 131 KB → 212 bytes (a stray
  `set_voxel` froze a whole chunk into a forever-fixture; chunk
  decoding is v1/v2's job); shadow `From`s chained V1 → V2 → live
  (old links frozen forever; only the last is rewritten per bump);
  `in_water` short-circuits (the per-actor-per-frame query);
  decision 1 + the substage spec de-rotted (point queries only,
  fraction derives in WT.1 — no box query exists).
- WT.1 — LANDED 2026-07-13: swimming in `CharacterBody`
  (character.rs). Hysteretic swim state at the top of the Walk path:
  `submerged_fraction` = `clamp(water_depth_at(feet) / height)`
  (pub — hosts wire tint/audio to the same number), engage ≥
  `swim_enter_frac` (0.5), release < `swim_exit_frac` (0.35); the
  no-stroke equilibrium `gravity/buoyancy` (24/32 = 0.75) sits above
  the enter threshold, so the surface bob can't flicker the state
  (pinned by test). `swim()`: horizontal `move_toward` at
  `swim_speed`/`swim_accel`; vertical `+= (gravity − buoyancy·frac)·dt
  + stroke·swim_accel·dt` then exponential `water_drag` (stroke =
  `sink` − `jump`, +z down); **breach** = `jump` with the head out of
  the water (`!scene.in_water(head)`) → full `-jump_speed` impulse
  (knob-free; fires-while-held matches the walk path's held-jump
  behaviour). Swimming grants no coyote and buffers no jumps; the fly
  modes clear the state; `teleport` resets it. New `CharacterDef`
  fields (defaulted): `buoyancy`, `water_drag`, `swim_speed`,
  `swim_accel`, `swim_enter_frac`, `swim_exit_frac`; new
  `WalkInput.sink` (pub-field growth — all in-tree literal users
  patched: 2 demos, 2 gallery scenes, cave-web, book example).
  7 tests: float-to-equilibrium + no-flicker, dive/stroke-up, breach
  impulse, wading below enter keeps walking, dry-walk byte-identity
  with far-away water, scaled-grid (vws 0.5) same WORLD equilibrium,
  fly clears + ignores.
  **Review round (user, 2026-07-14) — 10 findings closed:**
  1. `submerged_fraction` continuity guard: feet below an authored
     volume's BOTTOM face with the head still in water read as fully
     submerged (was: snap to 0.0 mid-dive → state strobe hysteresis
     can't absorb). Diving all the way THROUGH a short volume is one
     clean swim→walk exit (pinned).
  2. Breach is a ONE-SHOT (`breach_latched`, re-armed on jump
     release / walking / fly / teleport): a held jump re-impulsed to
     full `-jump_speed` every frame — the body left any pool at jump
     speed with no ballistic decay. Pinned: second held frame must
     NOT read exactly `-jump_speed`; release + press re-breaches.
  3. `eye_in_water` added — THE hook for WT.2 tint / WT.3 lowpass.
     `submerged_fraction` is feet-depth (0.75 at the bob with a DRY
     camera) and its doc now says so.
  4. **The big one — waterline relay oscillator.** Clamping release
     below the equilibrium wasn't enough: a cork (buoyancy 80) still
     strobed, because the walk path applied DRY gravity inside water
     — the force jumped by `buoyancy·frac` at every state crossing,
     and the hysteresis loop pumps energy each cycle (dry-gravity
     fall across the released band re-accelerates the body every
     re-entry; drag can't outrun it). FIX: the passive water forces
     (buoyancy + drag, `frac`-scaled) apply in BOTH states —
     continuous across the boundary; the swim flag gates only
     controls (strokes/speeds/coyote). Bonus: wading is floaty,
     splashing landings cushion. Both thresholds also clamp around
     the equilibrium (0.9× / 1.1×) so the band always contains the
     resting point. Cork test: settle, then 300 frames of ONE stable
     state (either flag is honest inside the band).
  5. The dry terminal cap (`max_fall_speed`) binds under water too
     (a zero-buoyancy weak-drag tuning could sink at 4× the cap).
  6. `set_mode(Fly/Noclip)` clears the swim state immediately (a
     host polling `is_swimming()` between set_mode and the next
     walk() saw a stale underwater state).
  7. Default `swim_enter_frac` 0.5 → 0.6: minimal authored water
     (one voxel over the floor = frac ≈ 0.56 on the default body)
     stays WADABLE — a decorative puddle must not swap walking for
     swimming (was: needed a taller test body to wade at all).
  8. Horizontal steer deduplicated (`steer_horizontal`, walk + swim).
  9. Breach branches FIRST (stroke/drag on the breach frame were
     dead computation).
  10. All 6 external `WalkInput` literal sites use
     `..WalkInput::default()` — the next field won't churn 6 files.
  Tests 7 → 13 (breach one-shot/re-arm, terminal cap, cork
  stability, short-volume continuity + clean exit, eye_in_water,
  wading with the DEFAULT body). Scene 255 tests; workspace clippy
  (+`struct_excessive_bools` allow — 4 genuinely independent flags),
  fmt, docs, anchors, wasm clippy, book example run — all green.
- WT.2 — LANDED 2026-07-14: full-screen tint in the resolve step.
  - Facade: `Tint { color: Rgb, strength: f32 }` +
    `FrameParams.tint: Option<Tint>` (per-frame, unlike the
    renderer-level posterize; pub-field growth — in-tree all
    construction goes through `FrameParams::new`, no literals
    existed). Pipeline position: after the SSAA downfilter, BEFORE
    the posterize quantize (grade first — the reduced palette stays
    palette-shaped under the tint). Strength `0.0` folds to `None`
    in BOTH facade paths, so the identity claim is structural.
  - CPU: `tint_pixel` (per-channel f32 lerp + round-to-nearest) in
    `resolve_scene` between downfilter and posterize;
    `resolve_active()` includes the tint (a tint alone activates the
    resolve).
  - GPU: the `Resolve` uniform's `_pad0/_pad1` became
    `tint_rgb`/`tint_strength` (ZERO layout change — the pads were
    load-bearing spare room); per-frame write extends the existing
    offset-20 posterize refresh; the colour repacks facade
    `0x00RRGGBB` → little-endian R-low for `unpack4x8unorm`; the
    identity-resolve fast-path gate (PF.5 H6) includes
    `tint.is_none()`. `mix` in unorm space + `pack4x8unorm`
    round-to-nearest = the CPU arithmetic exactly (pinned by a CPU
    test that emulates the unorm round-trip over a sample grid).
  - Tests: strength-0 byte-identity, strength-1 flat paint,
    per-channel lerp + GPU-arithmetic equivalence; naga validates
    the edited shader; full gpu suite (21 scene_render on a live
    adapter) + render 92 + workspace clippy/fmt/docs + wasm clippy
    green. Demo wiring (eye_in_water → tint) lands with WT.4.
  **Review round (user, 2026-07-14) — 7 findings closed; the tint is
  now INTEGER end-to-end:**
  1. The f32 design was NOT bit-exact across backends: CPU
     `round(c+(t−c)s)` vs WGSL `mix` + `pack4x8unorm` provably drift
     ±1 on ~2.5% of byte pairs (user brute-forced 256×256×9; e.g.
     c=7,t=2,s=0.1 → 7 vs 6) — and the "GPU emulation" test used the
     CPU's own formula, a tautology that could never catch it.
     REDESIGN: strength quantizes to 8 bits (`Tint::quantized`, the
     single fold — closes the duplicated-closure finding too) and
     both backends run the identical u32 expression
     `(c·(255−s₈)+t·s₈+127)/255`; the GPU byte-quantizes the SSAA box
     average first (`floor(0.5+255c)`, = pack4x8unorm semantics),
     closing the second drift (CPU rounded the downfilter before the
     lerp, GPU lerped the raw average). Bit-exactness is now BY
     CONSTRUCTION — immune to driver float contraction; tests pin
     properties (endpoint brute force, boundedness, monotonicity, the
     reviewer's witness pair) instead of pretending to emulate.
  2. CPU capture (QE.7a) snapshotted the raw MARCH buffer at render
     time — a tinted/posterized frame captured without its grade
     (pre-existing for posterize!). Moved to the present-side
     pipeline (`capture_logical` after `resolve_scene`, in both
     `present` and `paint_egui` — pre-UI, like the GPU) — captures
     are now the post-resolve logical image on both backends.
  3. GPU capture's identity-gate MIRROR in readback.rs was not
     updated with the render-path gate — a tinted default-settings
     frame captured from the ungraded march buffer. Fixed + a
     drift-trap comment tying the two conditions together.
  4. `GpuRenderer::set_tint` now enforces its own contract (folds
     strength 0 → None) — a direct crates.io user's fade-to-zero no
     longer parks the identity fast path permanently.
  5. Host-side LE repack deleted: the shader extracts the facade's
     `0x00RRGGBB` with shifts — ONE packing convention on the path.
  6. `tint_channel` doc warns off the workspace's two other RGB
     lerps (particles truncates, cavegen rounds via f32).
  7. CPU resolve builds three 256-entry LUTs per frame (colour +
     strength are frame constants) instead of per-pixel arithmetic.
  Known seam (documented on `Tint`): CPU overlays rasterise before
  the resolve (graded), GPU overlays draw after it (full-brightness).
- WT.3 — LANDED 2026-07-14: the underwater listener muffle.
  **Deliberate deviation from decision 3**: NOT a `ListenerAcoustics`
  field but a trait method — `AudioOut::set_listener_lowpass(cutoff_hz)`
  (default no-op, the `set_source_pitch` precedent). Rationale: the
  reverb environment rides a ~2 Hz probe cadence with ~1 s tweens; a
  muffle wired through it would lag a head-dip by up to a second. The
  dunk is a FRAME-rate signal (`CharacterBody::eye_in_water`) —
  hosts call the method per frame.
  - `LISTENER_LOWPASS_OPEN_HZ` (20 kHz) = "not submerged"; pinned
    equal to `AcousticsConfig::default().open_cutoff_hz` — one
    definition of "open" across the per-source and listener paths.
  - kira: one `FilterBuilder` on `AudioManagerSettings`'s
    `main_track_builder` — the MASTER path, so every spatial voice
    AND the reverb send dull together; FAST (~120 ms) tween (a dip
    is an event, a slow ramp smears it); cutoff floored at 40 Hz.
  - Tests: default-no-op inheritance via the mock + the open-constant
    coherence pin; audio 31 tests both feature variants; kira-on-wasm
    clippy green (the cave-web audio stack invocation). Demo wiring
    (eye_in_water → cutoff + the water material's absorption entry)
    lands with WT.4; listening pass owed there.
  **Review round (user, 2026-07-14) — 6 findings closed:**
  1. NaN passed THROUGH the clamp (`f32::clamp` keeps NaN) and one
     NaN cutoff NaN-poisons the SVF integrators — the whole mix goes
     silent PERMANENTLY (no later valid call recovers). Non-finite
     input now rejected before the clamp ("hold the current muffle").
  2. Per-frame drive RESTARTED the 120 ms tween every frame (kira
     `Parameter::set` semantics) — the dunk smeared to ~3×, exactly
     the lag the per-frame method exists to avoid. The backend now
     stores the last clamped target and only a CHANGED value tweens.
  3. The master biquad was unconditional — every KiraAudio host
     (waterless games, wasm) paid attenuation + phase shift near the
     top octave + per-sample DSP with no opt-out. Now OPT-IN:
     `KiraAudio::with_options(pool, listener_lowpass)` (the filter
     can only be built at construction in kira); `new`/
     `with_capacity` keep the exact pre-WT.3 master path and inherit
     the trait no-op. `DEFAULT_POOL` went pub so opting in doesn't
     restate the magic pool size.
  4. THREE unsynced "open = 20 kHz" definitions (constant, config
     default, per-source builder literal) → ONE `crate::OPEN_CUTOFF_HZ`;
     `LISTENER_LOWPASS_OPEN_HZ` is defined AS it and the pin test
     died (coherence by construction).
  5. The magic 40.0 floor became `LISTENER_LOWPASS_FLOOR_HZ` beside
     the file's other named knobs, doc'd with the NaN story.
  6. The no-op test now asserts the MOCK'S STATE is untouched
     (voices, applies, stops, registrations) — not merely that two
     calls don't panic; it feeds NaN too. Plus
     `LISTENER_LOWPASS_SUBMERGED_HZ` (700 Hz) — the one named
     submerged default WT.4/WT.5 call sites share, so the listening
     pass retunes one line, not N demos.
- WT.4 — LANDED 2026-07-14: the native cave demo floods.
  **Design correction found by test**: the planned solid volumetric
  FILL destroys submerged surface colours — refilling cavities merges
  water into the rock slabs and the RLE format drops every
  interior-boundary colour (colourless by design; the crystal test
  caught black submerged crystals). Shipped design: a 2-voxel
  volumetric **surface SHELL** where air crosses `WATERLINE_Z = 170`
  (grazing looks accumulate thickness ⇒ cheap Fresnel; straight down
  stays clear enough to spot submerged crystals), physics = ONE
  `WaterVolume` over the whole band below, underwater look = the
  WT.2 tint (not murky geometry). The shell is the CC.4 veto's
  textbook case (exactly 2 voxels — the format limit): the body's
  `Solidity.passable = water_passes` crosses it to swim; plasma
  bullets ALSO pass it colour-keyed (they detonate on the pool floor,
  the boom muffled by the water's absorption on the way up).
  - Order is load-bearing: crystals plant BEFORE the flood (the
    planter must not grow on the shell), flood before the bake.
  - Wiring: `Material::volumetric(70)` + colour→material rows
    (render + AU2 absorption 0.5 via `DemoAudio`'s config), V-key
    fly ⇄ walk toggle, Space/Shift = jump-breach/sink (held keys pass
    through `WalkInput` — the fly path never reads them),
    `UNDERWATER_TINT` from `eye_in_water` per frame, the WT.3
    lowpass via `KiraAudio::with_options(DEFAULT_POOL, true)` +
    `DemoAudio::set_submerged`, splash particle ring on feet-crossing
    with speed (entry + exit; `prev_feet_depth` reset on regen so a
    respawn can't phantom-splash).
  - Seams documented in the module doc + `flood_below_waterline`:
    static volume (craters below the line still swim), fully-sealed
    submerged cavities have no visible surface plane.
  - 6 tests green incl. `cave_floods_below_the_waterline` (shell
    colour + veto + wet physics + deep-air-pocket swims + dry spawn)
    and the crystal test that caught the fill bug.
  - **Visual + listening pass PASSED** (user, 2026-07-14) with one
    follow-up, fixed: walking used the fly "drone" body (the
    ±PLAYER_RADIUS probe cube, eye at its centre) — ankle-height
    camera. Now per-mode proportions: the V-toggle stands a human
    frame up for Walk (`WALK_HEIGHT` 1.8 / `WALK_EYE_HEIGHT` 1.62,
    radius unchanged) and restores the drone cube for Fly; the
    eye→feet teleport sync and the world-top z-clamp read the
    CURRENT def instead of the old hardcoded drone numbers. Bonus:
    the swim thresholds were tuned for a 1.8 body all along — the
    0.6 drone swam at 0.36 depth; the human frame wades/swims at the
    intended depths, and the bob leaves the eye just above the
    surface.
  **Review round (user, 2026-07-14) — 6 findings + 4 cleanups
  closed:**
  1. The 2-layer shell grew a colourless `UnexposedSolid` cell
     wherever it landed on a floor 1 voxel down (air 170–171, rock
     172: the merged span's second cell is interior — both colour
     vetoes blind there: invisible step for the body, surface
     detonation for plasma). New fill rule keeps every water voxel
     in its span's COLOURED zone: layer 1 fills wherever air (even
     on rock — it's the merged span's top, always coloured); layer 2
     only over MORE air. Shallow shores get a 1-voxel sheet with a
     1-voxel air slice above the floor.
  2. Held Space bunny-hopped up the shore (the engine re-buffers a
     held jump every frame): the demo passes the held key while
     SWIMMING (continuous stroke; breach is engine-latched) but the
     PRESS EDGE while walking (`up_was_down`).
  3. Space/Shift leaked into the 3D wish in Walk mode — the
     normalisation taxed horizontal speed −29% while stroking. The z
     components now feed the wish only in the fly modes.
  4. `DebrisSystem` didn't know the water veto — detached rock stood
     ON the 2-voxel sheet like concrete. `debris.solidity.passable`
     now shares `water_passes`. (Accepted cosmetic seam: shatter
     PARTICLES' `Bounce` still treats the sheet as solid —
     `ParticleSystem` has no solidity plumbing; they fade in ≤1.4 s.)
  5. Splash gating: Walk mode only (the fly START mode crossed
     z = 170 and rang phantom rings) and only where the shell
     actually exists in the feet's column (a carved crater / sealed
     pocket has no visible surface for a ring).
  6. The WT.4 test had been inserted MID-doc-comment of the EV.4
     crystal test (its contract text was orphaned) — both docs
     reattached.
  Cleanups: ONE `material_map()` feeds renderer + fracture + audio
  (the three hand-written lists had already drifted — water missing
  from the fracture map dropped shot-out shell pieces as opaque
  rock; crystals in the audio map are a no-op, unmapped ids absorb
  at 1.0); the flood uses ONE batched `set_spans` (y-outer contract)
  instead of ≤16 384 per-column `set_rect` ScumCtx builds; bullets
  classify water from the getcube they already did (one lookup, one
  classification path); `eye_in_water` evaluates once per frame and
  feeds both tint and lowpass.
- WT.5 — LANDED 2026-07-14; **STAGE CLOSED** (pending the web
  browser pass, owed below).
  - cave-web floods with the native design verbatim: shell fill rule
    (coloured-zone invariant) + one `WaterVolume`, batched
    `set_spans`, body/bullet/debris water vetoes, V-toggle with
    per-mode proportions, edge-triggered walk jump vs held swim
    stroke, wish-z gated to fly modes, tint + lowpass from ONE
    `eye_in_water` per frame, splash ring (Walk + real-shell gated),
    `material_map()` feeding renderer + fracture + audio,
    `KiraAudio::with_options` opt-in. Touch UI has no mode toggle —
    desktop keys only (documented v1 limit). wasm clippy both CI
    invocations + release wasm build green.
  - Book: scene-graph grew a "Water & swimming" section (physics
    volumes vs visual shell, the auto-swim contract, `eye_in_water`
    as THE feel hook) with a new runnable `water` anchor in
    `book_controller.rs` (float-to-bob + dry-camera + dive
    assertions); the audio chapter documents the listener lowpass.
    Deviation from the plan: no demo-tour row — the water showcase
    is the cave demo, not a gallery scene (demo-tour covers
    scene-demo only).
  - CHANGELOG had the feature entries; status/memory closed.
- **Owed at stage close**: the user's WEB pass (trunk serve
  --features audio: V-walk, wade, dive — tint/muffle/splash/crystals
  — and CPU-fallback perf with the shell on screen; the fog wall
  caps rays, expected fine). PERF G8 (`exp2` volumetric rewrite)
  stays deferred — the shell is thin, no demo pressure.

## Goal

Swimmable water: the cave demo's lower chambers flood, the player
wades in, floats, dives for crystals, and the world responds — the
frame tints blue-green under the surface, sounds go muffled, bullets
still carve above the waterline. Native and web (cave-web has full
parity since PW.0b; water keeps it).

## Audit facts the design leans on (verified 2026-07-13)

- **The CC.4 passable-veto cannot carry deep water.** `Solidity {
  passable: Option<fn(VoxColor) -> bool> }` (collide.rs:38-82) works
  on `Cube::Color` hits only; slab interiors are `Cube::UnexposedSolid`
  — colourless BY FORMAT (.vxl stores surface colours) — and always
  block. A passable water body works to ~2 voxels of thickness; a
  filled pool has a colourless core the veto never sees. This is the
  load-bearing fact behind decision 1.
- **Volumetric rendering is ready.** `BlendMode::Volumetric`
  (material.rs:141) does per-cell Beer–Lambert on BOTH backends
  (dda.rs:1726-1745; scene_dda.wgsl:1030-1046): `eff_a = 1-(1-a)^seg_len`.
  Terrain .vxl solid runs traverse correctly (interior-retention is a
  KV6-sprite concern only). Known costs, both deferred with owners:
  per-cell `pow()` (PERF G8: precompute `log2(1-a)`, use `exp2`) and
  per-cell `shade_lit` (G3). Fine for a demo pool; measure in WT.4.
- **Gravity lives in one line.** `walk_grounded()` integrates
  `vel.z += gravity*dt` capped at `max_fall_speed`
  (character.rs:315); MoveMode = Walk/Fly/Noclip; z is DOWN (gravity
  positive, jump impulse negative).
- **Audio is one config entry away from "muffled through water"** —
  `AcousticsConfig.material_map` + `absorption` (AU2.0) already
  weight per-material thickness. What does NOT exist: a listener-side
  global lowpass (each source has its own occlusion lowpass;
  `apply_listener` only drives the reverb) — decision 3 adds it.
- **No full-screen tint exists.** FrameParams has fog/sky colours
  (distance-based); the only post is posterize (RP.2), applied at the
  logical resolution in the resolve step on both backends — the
  natural place to hang a tint (decision 3).
- **The demo water fake** (Particles scene) is a 1-2 voxel AlphaBlend
  shell — exactly the thing WT replaces with real volumetric fill +
  volume physics.
- **SC interplay**: grids carry `voxel_world_size`; water volumes are
  declared in grid-local voxels, so every world-side query must scale
  by vws (the same boundary rule as everything since SC).

## Locked design decisions (user, 2026-07-13)

1. **Physics water = `WaterVolume` list on the Grid** (grid-local
   voxel AABBs; the surface is the volume's TOP face — min z, z-down).
   API is **point queries only**: `Grid::add_water_volume` (+ the pub
   Vec for removal/edits, `bake_lights`-style) and
   `Scene::water_depth_at(world) -> Option<(GridId, f64)>` /
   `in_water`. The body's **submerged fraction is NOT a volume
   query** — WT.1 derives it from the centre-line point depth vs the
   body height (`clamp(depth_at(feet) / height)`), which is exact for
   the world-horizontal surfaces water is authored with; do not go
   hunting for a box-overlap query, it does not exist. Deterministic
   (equal-depth ties break to the smallest `GridId`), cheap (few
   AABBs), persists in the snapshot (version bump v2 → v3, SC.snap
   pattern: sibling field, old fixtures still load, missing field =
   no water). The CC.4 veto stays for what it is good at: thin
   pass-through curtains. Visuals are separate: the host fills the
   same region with Volumetric voxels (no new render representation).
2. **Swimming engages automatically by submersion.** No new
   MoveMode. A Walk-mode body whose submerged fraction crosses a
   threshold (with hysteresis) enters the swim state: gravity blends
   toward buoyancy (net upward near full submersion), vertical drag,
   slower accel; `jump` = swim up, a new `WalkInput.sink` = swim
   down; `jump` at the surface breaches into a normal jump.
   `CharacterBody::is_swimming()` for hosts. Fly/Noclip ignore water
   (the cave demo grows a mode toggle in WT.4 so flying explorers
   can opt into wading).
3. **Small engine hooks for the underwater feel** (both reusable
   beyond water):
   - Render: `FrameParams`-driven full-screen tint (colour +
     strength), applied in the resolve step next to posterize on
     BOTH backends. `None` = byte-identical output (the usual gate).
   - Audio: `ListenerAcoustics` grows a listener lowpass field
     (default = no-op); the kira backend puts one filter on the
     master path and tweens it. Water material gets an `absorption`
     entry in the demo config (~0.5 — muffled but audible).
4. **Demo target = flood the cave demo** (native WT.4, web WT.5 —
   parity held): a waterline through the lower chambers, volumetric
   fill + matching WaterVolume, splash particles on entry/exit
   (`voxel_debris` burst at the crossing point), tint + lowpass below
   the surface. Crystals under water = the dive incentive.
5. **Byte-identity gates everywhere**: no water volumes, tint `None`,
   lowpass default ⇒ bit-identical frames + sound on both backends.
   The swim state must not perturb a dry body's walk integration.

## Substages

- **WT.0 — water volumes (roxlap-scene).** `WaterVolume { lo: IVec3,
  hi: IVec3 }` (grid-local, inclusive, surface = lo.z); Grid API +
  point-depth queries (world↔local boundary with vws; see decision 1
  for what deliberately does NOT exist); snapshot v3 (fixture +
  v1/v2-loads); unit tests incl. scaled + rotated grids.
- **WT.1 — swimming (CharacterBody).** Submersion sampling against
  the scene's water volumes; swim state with hysteresis; CharacterDef
  grows `buoyancy`, `swim_speed`, `swim_accel`, `water_drag`,
  `submerge_threshold` (all defaulted so existing defs compile);
  `WalkInput.sink`; breach-jump at the surface. Tests: float to rest
  at the surface, dive/rise controls, walk-out on a shore slope, dry
  path byte-identical, threshold hysteresis (no mode flicker at the
  waterline), scaled-grid water.
- **WT.2 — underwater tint (roxlap-render).** FrameParams tint
  (colour + strength 0..1) composited in the resolve step (CPU +
  scene_resolve.wgsl), orthogonal to posterize; `None` byte-identical
  (pinned); headless GPU/CPU parity test.
- **WT.3 — underwater audio (roxlap-audio).** `ListenerAcoustics`
  lowpass field + kira master filter (default no-op, pinned);
  DemoAudio-side: submerged listener drives the lowpass + water
  material absorption entry. Unit test on the pure side; listening
  pass owed at WT.4.
- **WT.4 — cave-demo flood (native).** Waterline plane in the lower
  cave: volumetric water fill (define material, colour → material
  map — the SAME map pattern as crystals) + `WaterVolume`; V-key
  fly ⇄ walk toggle so swimming is reachable; splash particles at
  surface crossings; tint + lowpass wired to submersion; bullets
  vs water documented (v1: carving below the waterline leaves the
  volume as-is — an air pocket LOOKS dry but still swims; known v1
  seam). Visual + listening pass owed.
- **WT.5 — web parity + docs + close.** cave-web gets the same flood
  (PW.0b parity held; watch CPU-fallback perf with volumetric fill —
  the fog wall at 128 caps ray length, measure before tuning); book
  (scene-graph: water volumes + swimming; demo-tour rows; audio +
  platforms cross-refs), CHANGELOG, status, memory.

## Hazards

1. **z-DOWN sign traps.** Buoyancy is a NEGATIVE-z acceleration; the
   surface is the volume's MIN z. Every review of WT.1 should read
   the signs twice (the controller chapter's "compiles fine and runs
   upside down" warning applies verbatim).
2. **vws boundary.** Water volumes are grid-local; a scaled grid's
   world waterline moves with `voxel_world_size`. Pin with a scaled
   test in WT.0 AND WT.1 (SC discipline).
3. **Volumetric cost in deep water.** Per-cell `pow()` + per-cell
   lighting through many water cells per pixel. The cave fog wall
   caps ray length; if WT.4/WT.5 profiling still hurts, pull PERF G8
   (`exp2` rewrite) into the stage rather than shipping a slow demo.
4. **Carve-vs-water incoherence** (decision 4's v1 seam): shooting
   below the waterline makes an air pocket that still swims.
   Document; a real water-settling sim is explicitly out of scope.
5. **Snapshot compatibility.** v3 must load v2 fixtures (SC.snap
   precedent has the shadow-struct recipe); add the reverse-rot
   fixture test BEFORE wiring demos.
6. **Waterline flicker.** Submersion threshold without hysteresis
   oscillates at the surface (bobbing body + discrete sampling);
   decision 2 mandates hysteresis — test it explicitly.
7. **Web CPU fallback.** 320×256 + volumetric fill may still dip —
   the trunk-profile lesson says measure release builds only.
