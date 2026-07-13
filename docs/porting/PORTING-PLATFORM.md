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
- PW.1 — wasm GPU depth-picking: NOT STARTED
- PW.2 — CI matrix (macOS + aarch64 + wasm-audio check): NOT STARTED
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
