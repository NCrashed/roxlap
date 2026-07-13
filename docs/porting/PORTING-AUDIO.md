# roxlap — voxel-aware acoustics + audio playback (Stage AU)

Entry doc written 2026-07-07 at workspace 0.24.0+ (post-EV, post
release-gate quality pass). This is the **entry doc** for the audio
stage — tag **AU**. A fresh-context session should read it top to
bottom before touching code. Recon sources: engine-infrastructure
sweep + audio-crate landscape review (both 2026-07-07, verified
against crates.io/docs.rs).

## Status — STAGE AU CLOSED 2026-07-07 (AU.0–4 all landed)

Voxel-aware audio shipped: occlusion muffling through rock + cavity
reverb, `roxlap-audio` crate (pure core + optional kira backend), cave
demo showcase (`--features audio`), book Audio chapter. User listening
pass passed ("работает отлично"). Purely additive to the workspace
(new crate + a demo feature + a book chapter) ⇒ folds into the next
minor cut. Owed beyond the stage: wasm audio, Doppler, HRTF,
per-material acoustics (all noted as deferred below). **2026-07-13:
macro-stage AU2 (per-material + Doppler + gallery tab; wasm still
deferred) — entry doc at the END of this file.**

- AU.0 — LANDED 2026-07-07: `roxlap-audio` crate (workspace member,
  cavegen-style metadata); `AcousticsConfig` / `SourceAcoustics`;
  `path_thickness` (per-grid Amanatides–Woo accumulating exact
  in-solid segment lengths, transform-aware like `Scene::raycast`);
  `source_acoustics` (direct + fixed 8-spoke jitter ring →
  transmission → log-lerp cutoff + linear dB). 8 unit tests:
  exact perpendicular/diagonal thickness, monotonic muffling
  (1/2/5-voxel walls), doorway leak (NB: the ring converges toward
  the listener — at a wall 40% of the way the ±1.5 jitter is ±0.9
  voxels; tests must place slots inside that reach), chunk-seam,
  cross-grid sum, max-distance cap, determinism. Review round
  (user, same day) added: rotated-grid world-true thickness test
  (the one previously untested claim), a one-entry chunk borrow
  cache in the march (new public
  `Grid::chunk_voxel_solid(&Vxl, UVec3)` in roxlap-scene — one
  HashMap probe per crossed chunk, not per voxel),
  beyond-max-distance now reports `clear()` (was "buried" — see
  tuning notes), and the bedrock-plane caveat (doc + pinning test).
  10 tests total.
- AU.1 — LANDED 2026-07-07: `cavity` module — `probe_cavity` (fixed
  32-ray golden-spiral fan over `Scene::raycast`), `CavityProbe` /
  `ListenerAcoustics` / `CavityConfig`, and `CavityEstimator`
  (per-update exponential smoothing, default 0.75 = the prior art's
  `(3·old+new)/4`; first update seeds from raw; `reset()` for
  teleports). Review round (user, same day) reworked the mapping —
  the original leaked audible reverb outdoors (mix 0.25 at feedback
  0.63 in an open field, hidden by a relative test assert):
  (a) room size now comes from `enclosed_free_path` (hit rays only —
  escaped rays must not read open sky as a huge hall; all-miss
  sentinel = cap, moot since mix is 0 there), and (b) the wet mix
  hits zero at `outdoor_openness` (default 0.5) — standing on ground
  the lower hemisphere always hits, so raw openness tops out near
  0.5 outdoors and must read as FULLY dry. Tests use ABSOLUTE bounds
  (`dry.mix < 0.05`, `dry.feedback < min+0.15`) so this can't
  regress behind a wetter cavern. 5 tests total.
- AU.2 — LANDED 2026-07-07: playback backend behind the `kira`
  feature (off by default — CI's default build/test never touches
  it, no ALSA on the runner). `synth` module (deterministic LCG
  shot/impact/hum `SoundBuffer`s, no binary assets); `backend` module
  = the `AudioOut` trait boundary (every kira type stays behind it,
  hazard 1) + public `SourcePool` voice policy (pure, unit-tested:
  one-shots stealable oldest-first, loops held till stopped —
  reusable by custom backends); `kira_out::KiraAudio` implements it
  with the entry-doc topology (one `SendTrack`+`Reverb`, a pool of 24
  `SpatialTrack`s each with a lowpass `Filter` + per-source reverb
  send; 120 ms source tweens / 1 s listener tweens). `audio_probe`
  example (`--features kira`) walks a listener out of a sealed room
  through a doorway for a live listen. **kira 0.12 API reality
  check** (research sketch was partly wrong): filter/reverb param
  setters ARE runtime (macro-generated `set_cutoff/set_feedback/…(v,
  Tween)`), but `StaticSoundData` lives at
  `kira::sound::static_sound`, `SpatialTrackHandle` has NO `stop`
  (keep the `StaticSoundHandle` and stop THAT for loops), positions
  are `mint` types (via `glam/mint`), and `set_send` returns a
  `Result`. Native-only; flake gained `alsa-lib` (build+runtime) for
  the dev shell. 21 crate tests (synth shapes/loop/determinism +
  pool policy + a device-free `AudioOut` mock pinning the trait
  contract) — all backend-agnostic, so they run in CI's default
  (no-kira) build.
  Review round (user, same day):
  - **Reverb wired as a proper aux send**: the reverb runs FULLY WET
    (`Mix(1.0)`), room wetness is the send-track VOLUME (`mix_to_db`),
    NOT the reverb dry/wet. The original `Mix(0..0.5)` on a parallel
    send re-added each source's dry signal to the master — sources got
    LOUDER and comb-filtered the MORE open the space (backwards).
  - **Voice stealing now cuts the old sound**: dropping a
    `StaticSoundHandle` doesn't stop it in kira, so `start` `stop`s
    the old sound (15 ms fade) before reusing the slot.
  - Renamed tweens (`FAST_TWEEN_MS` pose+occlusion / `ENV_TWEEN_MS`
    reverb env); documented f64→f32 `mint_pos` truncation and the
    deliberate 96 (spatial) < 128 (occlusion) distance nesting.
- AU.3 — LANDED 2026-07-07: cave-demo integration behind the `audio`
  feature (off by default — the default build/CI/ship stays silent,
  no kira/cpal/ALSA). `crates/roxlap-cave-demo/src/audio.rs`:
  `DemoAudio` registers synth shot/impact/hum; `fire` → shot at the
  muzzle (occlusion-shaded once), `impacts` → boom per carve (capped
  2/frame), `tick` → listener pose every frame + reverb at 2 Hz + hum
  occlusion at 4 Hz. **Voice budget (the AU.2 review's concern)
  solved by distance-cull**: only the nearest `MAX_HUMS = 8` crystals
  within `HUM_RADIUS = 80` loop at once, started/stopped as the
  listener moves — one-shots never starved. `DemoAudio::new` returns
  `None` on no device ⇒ silent. In-solid guard: the cavity update is
  skipped while the camera is buried (else the reverb collapses to a
  sealed box). Listener orientation from the camera basis (positions
  + orientation both in roxlap world coords, so kira's azimuthal
  panning is self-consistent). Regen (F/R) calls `audio_reset` (stop
  all hums, clear reverb history — crystal indices change meaning).
  cfg'd `audio_*` helper methods on `App` keep the call sites clean
  with no-op twins.
  Review round (user, same day):
  - **One-shots now start already muffled** (was: reset-to-clear then
    a 120 ms occlusion ramp that arrived AFTER the ~120 ms shot
    envelope decayed — the attack, the audible tell, went unoccluded).
    AU.2 border fix: `AudioOut::play`/`play_loop` gained an
    `Option<&SourceAcoustics>` initial applied at `tween(0)` before
    `track.play`; the demo shades shots/booms/entering-hums at spawn.
  - **Near-set de-thrashed**: recompute throttled to `NEAR_HZ = 5`
    (off the per-frame path), with radius hysteresis (enter 80 / exit
    92) and cap-membership hysteresis (an active hum sorts 6 units
    nearer so a marginal newcomer can't evict it). Extracted to a pure
    `select_near` — 4 unit tests (cap, radius hysteresis, cap
    hysteresis, empty/far), device-free.
  - `reset` zeroes the throttle timers; documented the identity-grid
    assumption (world == grid-local) and the eye-centre-only in-solid
    guard as caveats.
- AU.4 — LANDED 2026-07-07: book "Audio" chapter (`docs/book/src/audio.md`,
  SUMMARY position 9 — renumbered the shifted cross-refs in
  demo-tour/rendering/scene-graph/sprites) + a device-free
  `book_audio` example (core only, no kira: prints occlusion +
  cavity params for a wall/room/doorway scene, anchored into the
  chapter) + CHANGELOG. Book renders (mdbook via `nix shell
  nixpkgs#mdbook`), check-anchors green. **Stage AU CLOSED** — user
  listening pass passed 2026-07-07.
- Deferred beyond the stage: wasm audio (see decision 7), per-material
  acoustics, Doppler, scene-demo tab.

## Goal

Sound that *knows about the voxels*:

1. **Occlusion/muffling** — a sound behind a voxel wall gets quieter
   and darker (lowpass), proportional to how much rock the path
   crosses; a doorway leaks sound realistically (partial occlusion).
2. **Cavity reverb** — a shot in a large cavern rings; the same shot
   in a crawl-space is dry; outdoors is drier still. Reverb decay and
   wet mix follow the measured space around the listener.
3. A **`roxlap-audio` crate**: a backend-agnostic acoustics core
   (parameters from `&Scene`) plus an optional playback backend, and
   the cave demo wired up as the live showcase.

## Locked design decisions

1. **Two layers, hard boundary.** The acoustics core is pure
   parameter computation — no audio device, no audio thread, fully
   unit-testable against synthetic scenes. Output per source:
   `{ gain_db, lowpass_cutoff_hz, reverb_send_db }`; output per
   listener: `{ reverb_feedback, reverb_damping, reverb_mix }`.
   Playback lives behind a cargo feature and a small trait, so the
   backend is swappable (kira has a history of API redesigns — pin
   the minor, keep the boundary clean).
2. **Playback backend = kira 0.12** (verified 0.12.1, 2026-05-25;
   MIT/Apache-2.0 dual; pure-Rust DSP, cpal links OS audio only).
   The required topology exists natively, zero custom DSP: one
   `SpatialTrack` per source carrying a runtime-tweenable
   `Filter` (lowpass) + volume + a per-source **send** into one
   shared `SendTrack` holding a `Reverb` whose
   feedback/damping/mix tween at runtime. Runner-up fyrox-sound 1.0
   (HRTF, but MIT-only, per-bus filters, drags Fyrox infra);
   firewheel is the one to re-evaluate in a year.
3. **Occlusion = 9 jittered thickness rays** (Sound Physics
   Remastered's recipe: 1 direct + 8 offset marches source→listener),
   each accumulating **solid path length** (voxel units) through the
   scene — not boolean hits. The engine has no thickness query today
   (`Scene::raycast` returns first hit; Beer–Lambert thickness lives
   only inside the renderers), so AU.0 builds a small DDA
   thickness march over public `Grid::voxel_solid`/chunk queries
   inside roxlap-audio; if it proves generally useful, upstream a
   `Scene::raycast_thickness` later. Mapping: summed thickness →
   lowpass cutoff (exponential toward ~800 Hz) + gain reduction;
   the 9-ray average gives soft doorway transitions.
4. **Cavity estimate = 32-ray golden-spiral fan from the listener**,
   free path per ray (capped ~64 voxels) + sky-escape fraction
   (rays that exit the populated AABB upward). Mean free path →
   reverb feedback (decay); enclosure fraction → wet mix;
   sky-openness kills reverb outdoors (Sound Filters' pattern).
   Heavy smoothing — `(3·old + new) / 4` per update — so the
   environment converges over seconds (Teardown ships exactly this
   lag; players read it as natural).
5. **Cheap by construction, decoupled from frame rate.** Budget per
   acoustic tick ≈ 9 rays × N audible sources at ~4 Hz per source
   (round-robin) + 32 listener rays at ~2 Hz. Prior art runs this
   on far weaker query stacks than roxlap's chunk-cached DDA
   (~hundreds of ray segments per second total — noise). Host-owned
   system following the ParticleSystem pattern: pure
   `update(dt, &Scene, listener, sources)` + `tick(backend)` sync.
6. **Demo sounds are synthesized** — no binary assets (house
   pattern: the Doom scene synthesizes its GIFs). Shot transient =
   filtered noise burst; impact boom = low sine + noise tail;
   crystal hum = detuned sine pair looping. Generated once at
   startup into `StaticSoundData`.
7. **wasm deferred out of the stage.** kira runs on wasm via cpal's
   WebAudio path (no atomics/COOP-COEP; streaming sounds are
   desktop-only; AudioContext must start from a user gesture — the
   web demos' existing click-to-pointer-lock flow is the natural
   gate). Nothing in the design blocks it; it's scoped out of AU to
   keep the stage shippable. Same for Doppler (kira roadmap, not
   ours), HRTF, and per-material reflectivity/absorption (the
   colour→material map gives a hook when wanted).

## Substages

- **AU.0 — crate + occlusion core.** `roxlap-audio` scaffold
  (cavegen-style Cargo.toml, workspace lints, members +
  default-members); acoustics types; the thickness DDA march;
  `occlusion(scene, src, listener) -> SourceAcoustics` with the
  9-ray average. Tests: 0 walls ⇒ open params; 1/2/5-voxel walls ⇒
  monotonically stronger muffling; doorway ⇒ partial; cross-chunk
  and cross-grid paths.
- **AU.1 — cavity estimator.** Golden-spiral fan, mean free path,
  sky fraction, smoothing state. `listener_env(scene, pos, dt) ->
  ListenerAcoustics`. Tests: synthetic closed box (small vs large ⇒
  decay ordering), open plane ⇒ dry, half-open pit in between;
  smoothing converges monotonically.
- **AU.2 — kira backend** (`feature = "kira"`, native-only in AU).
  `AudioOut` trait boundary; kira impl with the
  spatial-track/send-track topology; tween times 120 ms (per-source)
  / 1 s (environment); synthesized test tones; a `#[ignore]`d
  listening probe binary for the maintainer's ears.
- **AU.3 — cave-demo integration.** Listener = camera; sources:
  gunshot (at muzzle), impact boom (at carve centre — the existing
  `impacts` queue), crystal hum loops (at each `BakeLight` position,
  EV synergy); reverb follows the cavern. Tuning pass on both
  presets. Feature-gated so `--no-default-features` still builds
  silent.
- **AU.4 — docs.** Book chapter ("Sound in a voxel world"),
  CHANGELOG, this doc's status, memory note.

## Tuning notes (post-AU.0 review)

- **Listener-side undersampling**: the jitter ring sits at the SOURCE
  and every ray converges through the listener point, so "listener
  pressed against a wall / standing in a doorway" is sampled by one
  point. SPR ships this too; if it reads badly in the demo, symmetric
  jitter (offset both endpoints) is the fix — revisit in AU.3 tuning.
- **`rotation.inverse()` per ray×grid** in `path_thickness` — trivial
  cost today; hoist per-grid if the function ever goes hot.
- **Doorway-test geometry** is tied to the default
  `jitter_radius = 1.5` (±0.9 voxels at a wall 40% along the path) —
  re-derive the slot position if the default changes.
- **max_distance semantics locked**: beyond the budget the source
  reports `clear()` (never "buried") — distance attenuation belongs
  to the spatial backend, and clamping to muffled would put a lowpass
  step function at the boundary.
- **Bedrock plane counts as solid** (documented on `grid_thickness`,
  pinned by a test): keep acoustic endpoints above z = 255.
- **Listener inside solid** reads as a tiny sealed box (max wet, min
  feedback) — documented on `probe_cavity`. AU.3 must guard the
  camera-clipped-into-wall case (skip the update or nudge the probe
  point out of solid).
- **Two separate distance budgets by design**: occlusion
  `max_distance = 128` (as far as sources are audible) vs cavity
  `max_ray_dist = 64` (only the local room) — cross-documented on
  both configs so tuning doesn't conflate them.
- **The cavity fan rides `Scene::raycast`**, not AU.0's
  chunk-borrow-cached thickness march — when profiling AU.2/3, the
  32 listener rays have `raycast`'s cost profile (its own march, no
  borrow cache).

## Hazards

1. **kira API churn** — 0.9→0.10→0.12 each broke the track API. Pin
   the minor; keep every kira type behind the `AudioOut` trait; the
   acoustics core must compile without the feature.
2. **Audio-thread allocation discipline** — kira's handles are
   cheap, but create tracks up front and reuse; a per-shot track
   allocation storm is the classic mistake. Pool N spatial tracks
   (sources beyond the pool steal the quietest).
3. **Parameter zippering** — always set via tweens, never raw jumps;
   the 4 Hz acoustic tick + 120 ms tweens must overlap or fast
   listeners hear steps.
4. **Ray budget creep** — the estimator must stay round-robin;
   "just re-trace everything every frame" reads fine at 3 sources
   and melts at 50.
5. **Synth sounds that read as programmer art** — accept it for the
   demo (retro engine, retro bleeps), but keep the asset path open:
   `StaticSoundData::from_cursor` loads ogg/wav if the maintainer
   drops real files in later.

---

# Macro-stage AU2 — per-material acoustics, reverb character, Doppler, gallery tab

Entry written 2026-07-13, right after the 0.27.0 cut. Scoped with the
user: the **deep** per-material form (absorption AND reverb
character), the deferred scene-demo audio tab, and a DIY Doppler.
**wasm audio stays deferred** (explicitly excluded from this wave);
HRTF stays rejected (AU decision — Fyrox infra drag).

**Semver: this wave's cut is 0.28.0 (minor), never a patch** —
AU2.0/2.1 grew public config structs and added a `pub` field to
`CavityProbe` (Copy + PartialEq: exhaustive destructuring downstream
breaks). Source-level breaking only, the EV `Material.emissive`
class — minor per house policy (maintainer review).

## Status

- AU2.0 — LANDED 2026-07-13: `AcousticsConfig` grew `material_map` +
  `absorption` (plain data, defaults empty); new
  `path_thickness_weighted` (public) shares one walk with
  `path_thickness` — per solid cell, `Vxl::voxel_color` → material →
  multiplier, interior cells carry the last surface colour forward
  (1.0 before the first); `source_acoustics` routes through it.
  Activation gate = non-empty `absorption` (a map alone changes
  nothing); the empty-table path multiplies by an exact `1.0`, so all
  22 pre-AU2 pinned tests pass untouched. New tests (3): bit-identity
  (`to_bits` on the walk + config equality), glass wall = exactly
  0.3× thickness + higher transmission + unmapped-id no-op, and the
  interior carry-forward on a glass monolith (with a precondition
  assert that mid-run cells really are colour-less). roxlap-audio
  grew a `roxlap-formats` dep (`material_for_color`). 25 tests,
  clippy 0.
- AU2.1 — LANDED 2026-07-13: `CavityConfig` grew `material_map` +
  `damping_override` (activation gate = non-empty override; the
  `damping` field is now "the default material's damping" + the
  fallback for unmapped/colour-less hits). `CavityProbe` grew
  `damping`: mean per-material damping over hit rays
  (`RayHit.color` → `material_for_color` → override entry); the
  estimator smooths it like the other probe fields — EXCEPT the
  empty-table case, which passes the config constant through
  untouched (blending a constant with itself can drift a ULP; the
  legacy contract is exact, pinned via `to_bits` through TWO updates
  so the blend branch is exercised). `ListenerAcoustics
  .reverb_damping` = the smoothed probe value. Tests (3): legacy
  bit-exactness; glass box (0.1) vs stone box (default) + the value
  reaching listener params; mixed-wall averaging. Test-infra note:
  the AU-era `room()` helper CARVES its interior — carved faces are
  zero-RGB (colour-less) and unclassifiable, so AU2.1 tests build a
  `boxed_room` from six painted slabs instead (the same reason the
  cave demo's crater walls needed an explicit colfunc). 28 audio
  tests, clippy 0.
- AU2.2 — LANDED 2026-07-13: pure `doppler_factor(source_pos,
  source_vel, listener_pos, listener_vel, c)` +
  `DEFAULT_SPEED_OF_SOUND = 90.0` (game-tuned knob, doc says why not
  343). The two-sided formula made monotone at the edges: numerator
  floors at 0, denominator at `0.05·c` (a supersonic approach clamps
  to the ceiling instead of flipping sign), result clamps to
  `[0.5, 2.0]`; rest is `c/c` — exactly `1.0`. `AudioOut` grew
  **default-no-op** `set_source_pitch(id, factor)` (existing
  backends compile untouched); `KiraAudio` maps it to
  `set_playback_rate` with the FAST (~120 ms) tween. Cave demo: the
  crystal hums get per-frame Doppler from the **listener's velocity**
  (position delta per tick, teleport-guarded at 200 u/s and reset on
  regen — a warp reads as rest, not a pitch spike); crystals are
  static so `source_vel = 0`. Tests (2): signs + exact-rest +
  perpendicular-neutral; clamps + degenerate inputs. Env note: this
  shell CAN build the `audio` feature after all —
  `PKG_CONFIG_PATH=<nix-store alsa-lib dev>/lib/pkgconfig` (the
  earlier "needs full dev shell" claim was a missing search path,
  not a missing library); demo clippy + tests green WITH audio.
  30 audio tests, clippy 0.
- AU2.3 — LANDED 2026-07-13: the **Audio** scene
  (`scene-demo/src/scenes/audio.rs`, registered unconditionally).
  Design upgrade over the plan: the acoustics core is pure, so the
  tab exists in EVERY build showing the live numbers (`book_audio`
  made walkable — occlusion per source, reverb environment, Doppler,
  listener speed); the new scene-demo `audio` feature
  (`roxlap-audio/kira`) only adds playback (three looping zone hums,
  full parameter mirroring incl. per-frame Doppler pitch). World:
  floor + sealed stone room (doorway on +x) + a big **glass** cavern
  (entrance on −x) — built from PAINTED slabs with gaps assembled
  around the doorways, never carved (the AU2.1 zero-RGB lesson); the
  same colour map drives rendering, so the cavern renders as actual
  alpha-blend glass AND classifies as low-damping for the reverb.
  Tables ACTIVE: glass → absorption 0.35 + damping 0.1. **Hazard-2
  profiling debt DISCHARGED** with numbers (release probe
  `weighted_tables_probe`, #[ignore]d, roxlap-audio):
  `source_acoustics` through 4 walls — 8.03 µs empty vs 8.10 µs
  active tables (+0.8%); `probe_cavity` — 7.2 vs 6.9 µs (noise). The
  feared per-cell `voxel_color` cost doesn't materialise: colours
  are only sampled on solid cells, and audio rays cross few. The
  scene's HUD also shows the measured per-update acoustics cost
  live. Both feature variants compile, clippy 0; 30 audio + 18
  scene-demo tests green.
- AU2.4 — LANDED 2026-07-13 — **MACRO-STAGE AU2 COMPLETE** (owed: the
  user's listening pass — scene-demo Audio tab with
  `--features audio`, and the cave demo's Doppler-bent hums; can ride
  the next session). Book: the Audio chapter grew **"Materials:
  glass sounds thin"** (same-map-three-consumers pitch, absorption +
  damping through the extended device-free `book_audio` — a glass
  pane heard both ways, 0.47 → 0.77 transmission, and a painted
  glass booth reading damping 0.10; the material-0 convention and
  the carve/paint caveat stated in prose) and **"Doppler"** (the
  factor table: ×1.44 approaching / ×0.56 receding / ×1.00 still;
  the 90 u/s game-tuning rationale; loops-only guidance); the cave
  demo section mentions the bending hums, and a new paragraph points
  at the gallery's Audio tab. `check-anchors` green (two new
  anchors: `materials`, `doppler`). CHANGELOG: the AU2 wave under
  [Unreleased], closing with the minor-bump note. Wave cut = 0.28.0
  (the semver obligation above).

## Locked design decisions

1. **Material tables are plain data on the configs**, the DT
   side-table pattern (`DebrisSystem::set_fracture_patterns`):
   `AcousticsConfig` grows `material_map: Vec<(Rgb, u8)>` — the SAME
   colour→material terrain map the renderer and the debris system
   already take — plus `absorption: Vec<(u8, f32)>`, a material →
   effective-thickness multiplier (stone 1.0, glass/crystal ~0.35;
   unmapped = 1.0). **Empty tables must stay bit-identical to today**
   — the AU tests pin exact values and must pass untouched.
2. **Weighted thickness** (AU2.0): `grid_thickness` samples the voxel
   colour of each SOLID cell it crosses (colour → material → weight;
   effective thickness = Σ segment × weight). The `.vxl` format
   stores surface colours only, so interior cells inherit the last
   colour seen along the ray (the same carry-forward
   `Island::build`/`voxel_color` convention); before any coloured
   cell the weight is 1.0. Keep the one-entry chunk borrow; only
   sample colour for solid cells.
3. **Reverb character** (AU2.1): `probe_cavity` already receives
   `RayHit.color` per wall hit — classify each hit through the same
   map; `CavityConfig` grows `material_map` + `damping_override:
   Vec<(u8, f32)>`. The probe's effective damping = mean over hit
   rays of the per-material damping (unmapped/no-colour hits use
   `cfg.damping`); rides the existing `(3·old + new)/4` smoothing
   into `ListenerAcoustics.reverb_damping`. A glass cavern rings
   brighter than a stone one.
4. **Doppler is core-computed, backend-applied** (AU2.2): pure
   `doppler_factor(source_pos, source_vel, listener_pos,
   listener_vel, speed_of_sound) -> f32`, clamped to `[0.5, 2.0]`;
   `AudioOut` grows a **default-no-op** `set_source_pitch(id,
   factor)` (existing backends keep compiling); `KiraAudio` maps it
   to `set_playback_rate` with the house ~120 ms tween. Speed of
   sound is a knob (default 90 u/s — audible against 60 u/s
   bullets). Pitch-bending a one-shot mid-boom sounds wrong — hosts
   should apply Doppler to loops and long tails (demo: crystal hums
   vs a moving listener).
5. **Gallery tab** (AU2.3): scene-demo grows an optional `audio`
   feature mirroring the cave demo's, plus an Audio scene: sealed
   room + doorway + open field + cavern, one looping synth source per
   zone, HUD readout of transmission / openness / damping — the
   `book_audio` numbers made walkable.

## Substages

- **AU2.0 — weighted thickness (roxlap-audio).** Tables on
  `AcousticsConfig`; colour sampler + carry-forward in
  `grid_thickness`; `source_acoustics` unchanged on empty tables.
  Tests: empty tables bit-identical (existing pinned tests
  untouched); a glass wall vs a stone wall of equal thickness →
  higher transmission; interior carry-forward pinned (thick wall,
  surface colour rules the interior).
- **AU2.1 — reverb character (roxlap-audio).** Tables on
  `CavityConfig`; per-hit classification in `probe_cavity` (returns
  effective damping in `CavityProbe`); estimator smooths it. Tests:
  glass box vs stone box damping; empty tables identical; mixed-wall
  box averages.
- **AU2.2 — Doppler.** `doppler_factor` + `AudioOut::set_source_pitch`
  (default no-op) + `KiraAudio` impl; cave demo: the listener's own
  velocity vs the crystal hums (fly past a crystal — the hum bends).
  Tests: approaching > 1 > receding; symmetric at equal speeds;
  clamps; zero velocity = 1 exactly.
- **AU2.3 — scene-demo Audio tab** (behind `audio`): the walkable
  scene above; F1 HUD lines for the live numbers.
- **AU2.4 — docs.** Book Audio chapter grows "Materials & Doppler"
  (extend `book_audio` — still device-free); CHANGELOG; this status;
  memory note.

## Hazards

1. **Pinned-value tests.** The AU test suite pins exact floats;
   AU2.0/2.1 must route empty-table paths through the EXACT same
   arithmetic (no reordering of float ops).
2. **Colour lookups cost more than solid tests.** `voxel_color` walks
   the slab chain per cell; keep it gated on solid cells only and
   behind the chunk borrow. Audio rays are short and ~4 Hz per
   source — fine — but re-profile the demo tick with ACTIVE tables.
   **OWED at AU2.3** (maintainer review): AU2.0 landed core-only, no
   demo wires the tables yet — the profile is honest only once the
   gallery tab (or the cave demo) runs weighted queries per tick.
3. **Interior cells have no colour** — the carry-forward starts
   empty; a segment entering solid mid-wall (buried source) weights
   1.0 until the first coloured cell. Accepted; document on the
   config.
4. **Doppler on one-shots** reads as a glitch, not physics — the API
   is per-source, the guidance (docs + demo) applies it to loops.
5. **kira playback-rate units** — factor is a ratio (1.0 = neutral);
   tween it like the other params or fast flybys zipper.
