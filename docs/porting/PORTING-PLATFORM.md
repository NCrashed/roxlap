# PORTING-PLATFORM.md — platform wave (PW)

Entry doc written 2026-07-13, right after the 0.28.0 cut. This is the
**entry doc** for the platform wave — tag **PW**: one stage bundling
the three long-deferred cross-platform tails so they stop haunting
other stages' deferral lists. A fresh-context session should read it
top to bottom before touching code.

- **wasm audio** (deferred at AU close, re-deferred out of AU2): the
  cave web demo gets the full voxel-aware soundscape.
- **wasm GPU depth-picking** (deferred at GW.1): `pick_depth` on the
  browser GPU path returns `None` today; the CPU fallback picks.
- **CI matrix** (deferred at R8): macOS and Linux-aarch64 runners join
  the x86_64-linux-only pipeline.

## Status

- PW.0 — LANDED 2026-07-13: cave-web grew the `audio` feature
  (`roxlap-audio/kira`, optional; `trunk serve --features audio`) and
  `src/audio.rs` — the trimmed `DemoAudio` port (`WebAudio`): shots
  at the muzzle + carve booms (both occlusion-shaded at spawn, capped
  2/frame) + cavity reverb at 2 Hz with the buried-eye guard; no hums
  /Doppler (the web cave has no crystals — noted for whenever they
  port). **The gesture is the constructor** (decision 2): built
  lazily in the first pointer-lock click AND the first touchstart;
  reverb history resets on preset/reseed regen. kira compiled for
  wasm32 untouched — hazard 3 did NOT fire: `cargo tree` shows
  everything on the pinned wasm-bindgen v0.2.117 (cpal 0.17.3 rides
  it). Verified under the flake's pinned nightly: check + clippy for
  both feature variants (the 2 clippy warnings present are
  PRE-existing wasm-path lints — cpu.rs:611 needless_pass_by_value,
  cave-web lib.rs:555 map_unwrap_or — never gated because the CI
  wasm job only runs `check`; candidates for PW.2). **Drive-by bug
  fix**: cave-web's bullet impact used
  `set_sphere(…, Some(CARVE_COLOR))` — INSERTING solid painted balls
  instead of the documented crater; now
  `set_sphere_with_colfunc(…, SpanOp::Carve, …)` like the native
  demo (carve + painted crater walls). Owed: the user's **browser
  listening pass** (`trunk serve --features audio`: click → shots
  and booms muffle behind rock, reverb swells in chambers — and
  craters now actually excavate).
  **Listening-pass follow-up (user, 2026-07-13)**: audio works, but
  the CPU fallback's DDA can't hold 640×512 in wasm (low FPS since
  the DDA migration). Fix: backend-conditional resolution — WebGPU
  keeps 640×512, the CPU path drops to quarter-pixel 320×256
  (`renderer.resize` after backend detection; input mapping follows
  automatically since the handlers read `canvas.width()` live), with
  a fixed 640-wide CSS size + the existing `image-rendering:
  pixelated` so the on-screen picture stays identical in spirit —
  crisp-retro upscale. roxlap-web (the legacy demo) has the same
  CPU-res exposure — candidate for the same two-line treatment if
  anyone still exercises it.
- PW.0b — LANDED 2026-07-13 (user follow-up: "не хватает кристаллов,
  их гудения и падений островов"): cave-web reaches full parity with
  the native cave demo.
  - **Bullets → dynamic sprite API** (prerequisite: the old per-frame
    `set_sprites` rebuild RESETS the dynamic instance world, which
    would have deleted debris + particles every frame).
  - **Crystals**: `plant_crystals` extracted from the native demo into
    `roxlap_scene::cavegen` (shared fn + `CrystalParams`; the
    `bake_light` book anchor moved with it, `lighting.md` include
    repointed, `check-anchors.sh` green). Same colours + salts →
    identical caves grow identical crystals. Web gets lightmode 2 +
    `BakeMode::PointLights`, the translucent+emissive material, and
    `generate_mips` at regen.
  - **Crumble**: synchronous in-frame (no carve worker on the web —
    the 128³ cave affords it): per-hit `detect_islands` → immediate
    `spawn_island` (extraction prevents duplicate re-finds),
    `DebrisSystem` + landing shatter `ParticleSystem` + booms, model
    pool compaction every 32 shatters. Carves now relight with
    `bake_bbox(PointLights)` + `remip_bbox` over the edit extent —
    closing TWO pre-existing web bugs: whole-chunk `Directional`
    re-bake (would erase glow pools) and no re-mip after carves. A
    THIRD found in review: the spawn bubble was `set_sphere(Some(…))`
    — an INSERTED solid ball, not a carved bubble (same bug class as
    the PW.0 crater fix).
  - **Audio**: web `WebAudio` is now the full native `DemoAudio` —
    crystal hums (distance-culled MAX_HUMS=8, enter 80 / exit 92 /
    cap-hysteresis 6, occlusion at 4 Hz) + AU2 Doppler from listener
    velocity (200 u/s teleport guard). `select_near` copied verbatim;
    its unit tests live in the native demo (this crate is wasm-only).
  - Verified: wasm32 check + clippy both feature variants (only the 2
    pre-existing lints), roxlap-scene 234 lib tests, cave-demo 5
    tests (incl. `crystals_planted_and_lit` against the delegate),
    rustdoc -D warnings, check-anchors. Browser pass PASSED
    (2026-07-13): crystals + hums + falling islands confirmed live.
- PW.1 — LANDED 2026-07-13: wasm GPU depth-picking per decision 4.
  - `roxlap-gpu/src/pending_pick.rs` — `PendingPick`, the pure
    one-in-flight state machine (request → in-flight → complete →
    re-arm; clicks during flight coalesce away — the next call re-arms
    with its own, newest pixel; the completed result keeps its pixel).
    4 native unit tests.
  - Driver `GpuRenderer::read_depth_pixel_async` (new pub API ⇒ minor):
    harvest-if-resolved → re-arm-for-this-pixel → return latest
    completed. Per-pick 4-byte staging buffer owned by the state (NOT
    the shared `depth_readback`) — the copy executes at submit time,
    so a resize/scene swap between calls can't invalidate an in-flight
    pick, and no generation counter is needed. `map_async` result
    rides an `Arc<Mutex>` cell (fresh per submission), so the driver
    compiles on every target; the browser event loop resolves the map
    between RAF frames.
  - Facade: the wasm `pick_depth` stub (`None`) now calls the async
    driver; `SceneRenderer::pick_depth`/`pick` docs state the
    one-frame-latency contract (poll next frame; result may be the
    previous pixel's). Native path byte-untouched. Stale doc claim
    fixed en route: depth is ALWAYS written since L3.1 — picking never
    needed sprites in the frame.
  - cave-web probe: `P` polls `pick()` at the canvas centre (≤30
    frames) and console.logs the world hit + voxel — works on both
    wasm backends (CPU resolves on the first poll).
  - Verified: gpu 52 + render 89 native lib tests, clippy clean,
    workspace rustdoc -D warnings, wasm check+clippy both variants
    (only the 2 pre-existing lints). Owed: the user's manual browser
    check (press P over a wall → coordinates in the console).
- PW.2 — LANDED 2026-07-13: CI matrix per decision 5.
  - Two new jobs: `test-macos` (macos-latest, Apple Silicon) and
    `test-linux-arm` (ubuntu-24.04-arm), both running the standard
    `cargo test --workspace --release --exclude roxlap-sdl-demo`. The
    two first-time-proofs (DDA hash portability on ARM; GPU tests
    under Metal) are called out in the job comment with the agreed
    responses (pin per-arch / scope mac to build-only).
  - wasm-check UPGRADED check → clippy (`components: clippy` on the
    pinned nightly) and split into two invocations: default-feature
    web crates, then the audio stack
    (`roxlap-cave-web/audio + roxlap-audio/kira` — the PW.0 gate).
    The three wasm-only pedantic lints that had accumulated unseen
    (check never ran clippy) are FIXED: cpu.rs `new_from_canvas`
    takes `&canvas` (needless_pass_by_value; internal call site only
    — the public `new_from_canvas_async` signature is unchanged), and
    `map(...).unwrap_or(4)` → `map_or(4, ...)` in BOTH web demos'
    `navigator_hardware_concurrency` (the roxlap-web copy was
    invisible until the audio-stack invocation ran it).
  - Verified locally: both clippy invocations with the exact CI
    RUSTFLAGS pass clean under the pinned nightly; fmt + native
    clippy green; YAML parses (11 jobs). Owed: watching the FIRST
    mac + ARM runs on push (decision 6's experiment).
- PW.3 — docs + close: NOT STARTED

## Audit facts the design leans on (verified 2026-07-13)

- **kira 0.12.1 is wasm-ready out of the box**: its own manifest
  switches to `cpal` with the `wasm-bindgen` feature (+
  `send_wrapper`) under `cfg(target_arch = "wasm32")` — our `kira`
  feature needs no per-platform dependency surgery. The
  `roxlap-audio` CORE is already pure (no threads, no `Instant`
  outside an `#[ignore]`d probe); `kira_out.rs` uses only
  `std::time::Duration` (wasm-safe).
- **cave-web is a modern facade host**: `SceneRenderer` with
  WebGPU→CPU/WebGL2 fallback, RAF loop, and — key — a **first-click
  pointer-lock request** (`lib.rs:667`) that is the natural
  user-gesture moment the browser requires before an `AudioContext`
  may produce sound. Touch has the same gate (first tap). Trunk build
  already ships COOP/COEP + atomics (wasm-bindgen pinned `=0.2.117`,
  nightly toolchain pinned for build-std).
- **roxlap-web is the legacy oracle demo** — kept building, but it
  gets no audio (cave-web is the showcase).
- **Picking**: native `pick_depth` reads 4 bytes back with a blocking
  `map_async` + `device.poll(wait_indefinitely)`
  (`roxlap-gpu/src/readback.rs:34-73`); the wasm gate at
  `roxlap-render/src/gpu.rs:883-887` returns `None` because WebGPU
  has no blocking readback — `map_async` resolves on browser
  event-loop turns. There is NO existing async/frame-latency
  readback infrastructure to reuse.
- **CI today** (`.github/workflows/ci.yml`): 9 jobs, all
  `ubuntu-latest` x86_64 — fmt, clippy, wasm-check (type-check of the
  two web crates on pinned nightly with `+atomics` flags), test
  (`--release`, minus roxlap-sdl-demo), msrv 1.92, docs, smoke-fuzz,
  book, deploy-book. GPU tests silently skip without an adapter.
- **Per-arch goldens are a legacy phantom**: PORTING-RUST.md's
  deferral note ("each architecture needs its own goldens") predates
  the DDA renderer. `golden-hashes-aarch64.txt`, `wasm-hashes.txt`
  and the `roxlap-oracle` harness are all GONE from the tree — they
  died with opticast. The DDA-era hash tests are plain IEEE f32 (no
  `rcp` intrinsics, no fast-math), expected bit-portable across
  x86/ARM; the aarch64 job's first run is the proof.

## Locked design decisions

1. **Audio target = cave-web only**, behind an `audio` cargo feature
   mirroring the native cave demo (off by default; a plain trunk
   build pulls in no audio stack). roxlap-web stays silent (legacy).
2. **The gesture IS the constructor.** `KiraAudio` (and the whole
   demo-audio wrapper) is built lazily on the **first pointer-lock
   click / first touch** — never in `start()`. Constructing an
   `AudioContext` before a user gesture leaves it suspended and, in
   some browsers, permanently mute; building it inside the gesture
   handler sidesteps the resume dance entirely.
3. **cave-web ports the native demo's audio wiring** (the
   `DemoAudio` pattern: shot at the muzzle, boom per carve impact,
   distance-culled crystal hums with occlusion + cavity reverb +
   AU2 Doppler from the camera velocity). Where practical the tuning
   constants are copied verbatim — the browser should sound like the
   native demo.
4. **wasm picking = one-frame-latency async, re-armed per call.** A
   small platform-neutral `PendingPick` state machine (unit-tested
   natively): `pick_depth(x, y)` on wasm submits the 4-byte copy +
   `map_async` for THIS click and returns the **latest completed**
   result (usually `None` on the first call, the value on the next).
   The facade doc states the semantics; hosts needing synchronous
   picks keep the CPU backend (unchanged default fallback).
5. **CI grows two runner jobs, not a matrix rewrite**:
   `macos-latest` (Apple Silicon) and `ubuntu-24.04-arm`, each
   running the standard `cargo test --workspace --release` (minus
   sdl-demo). GPU tests self-skip without an adapter; on the mac
   runner Metal MAY be reachable — if those tests prove flaky in CI,
   the follow-up is scoping the mac job to build-only, not deleting
   it. The wasm-check job additionally checks
   `roxlap-audio --features kira` for wasm32 (the PW.0 gate) and
   `roxlap-cave-web --features audio`.
6. **No per-arch goldens revival.** If an ARM run flips a hash test,
   that is a *finding* (pin per-arch expectations then, with the
   R9-era file naming); the wave does not pre-build machinery for a
   problem the DDA renderer probably retired.

## Substages

- **PW.0 — wasm audio (cave-web).** `roxlap-cave-web` grows the
  `audio` feature (`roxlap-audio/kira`); an `audio.rs` module ports
  the native `DemoAudio` (fire / impacts / hums / Doppler / cavity,
  same constants); lazy construction inside the pointer-lock click +
  first-touch handlers; wasm-check locally via the pinned nightly.
  Owed: the user's **browser listening pass** (trunk serve, click,
  shoot, fly past a crystal).
- **PW.1 — wasm GPU depth-picking.** `PendingPick` in roxlap-gpu
  (platform-neutral, unit-tested: request → in-flight → completed →
  re-arm, stale-click coalescing); wasm `pick_depth` wires it to
  `map_async` (callback into an `Rc<Cell>`), native path untouched;
  facade docs updated (one-frame latency on wasm GPU). Owed: manual
  browser check (a click-pick probe in cave-web behind a key).
- **PW.2 — CI matrix.** Two new jobs (macos-latest,
  ubuntu-24.04-arm) + the wasm-check extension from decision 5.
  Verified on push — watch the first runs (mac Metal and ARM hash
  portability are both first-time-proofs).
- **PW.3 — docs + close.** Book Platforms chapter: wasm audio (the
  gesture rule), picking semantics table (native sync / wasm
  one-frame / CPU always-sync), CI coverage statement; CHANGELOG;
  this status; memory.

## Hazards

1. **AudioContext autoplay policy.** Anything constructed before the
   gesture may stay suspended forever on some browsers. Decision 2
   (lazy construction in the handler) is the defence — do not
   "optimise" it into `start()`.
2. **Audio underruns on main-thread jank.** cpal's wasm backend
   drives audio from the browser's audio callback; heavy RAF frames
   (big carves) may crackle. Accept for the demo; note in the book.
   The rayon worker pool (COOP/COEP) is already in place and does
   not conflict.
3. **The nightly + `=0.2.117` wasm-bindgen pins.** kira/cpal/web-sys
   additions must build under the pinned nightly and must not drag a
   newer wasm-bindgen (the nixpkgs CLI would reject it). Check
   `cargo tree` before committing to versions.
4. **wasm pick semantics are observably different** (one-frame
   latency, result may correspond to a slightly stale depth buffer).
   Documented, not hidden; CPU backend remains the sync option.
5. **mac runner Metal is untested territory** — the GPU test suite
   may genuinely execute there for the first time in CI. Flakiness →
   scope the job to build-only as the follow-up (decision 5).
6. **ARM hash portability is assumed, not yet proven** (decision 6).
   The first `ubuntu-24.04-arm` run is the experiment; a red hash
   test there is a finding to pin, not a broken wave.
