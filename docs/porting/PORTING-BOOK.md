# roxlap — the engine book (Stage BK)

Entry doc written 2026-07-05 at workspace 0.22.0, right after stage PS
(particles) closed and shipped. This is the **entry doc** for the
documentation-book stage — tag **BK**. A fresh-context session should
read it top to bottom before touching anything; it bakes in the
documentation audit so no re-exploration is needed.

## Status

- BK.0 — **LANDED 2026-07-05.** All six decisions confirmed by the
  author as proposed. Delivered: mdbook in the dev shell (nixpkgs
  0.5.2, pin CI to match), `docs/book/` skeleton (book.toml +
  SUMMARY + foreword + 12 chapter stubs), chapter 1 written
  (anchors `colors`/`build_scene`/`init`/`render_frame`/`teardown`
  in `quickstart.rs` — per-frame code extracted to a top-level
  `render_frame` fn so includes aren't 16-deep indented; deps toml
  included from README via HTML-comment anchors), CI `book` job,
  README links the book (top + Documentation).
  **Correction to decision 4:** mdbook 0.5.2 does NOT fail on broken
  includes — missing file logs ERROR and exits 0, missing anchor is
  *silent empty output*. The real gate is `docs/book/check-anchors.sh`
  (resolves every `{{#include}}`, verifies file + ANCHOR/ANCHOR_END;
  verified red on both failure modes); CI runs it before `mdbook
  build` and additionally greps the build log for WARN/ERROR.
- BK.1 — **LANDED 2026-07-05.** Ch. 2 (concepts) + ch. 3 (scene graph)
  written. Two new headless, assertion-checked anchor-bearing examples
  (run them after API changes — their asserts are the book's facts):
  - `roxlap-core/examples/book_conventions.rs` — anchors
    `packed_color` / `z_down` / `camera_basis` (incl. the
    `Camera::default()` left-handed trap, asserted).
  - `roxlap-scene/examples/book_scene_graph.rs` — anchors
    `scene_grids` / `edits` / `queries` / `recolour` / `colfunc` /
    `snapshot` / `generator` / `streaming` / `chunk_store` /
    `streaming_edit`. The recolour gotcha and the
    ChunkStore evict→re-stream edit survival are *asserted*, not just
    stated. Teaching note learned while writing: after walking away,
    the active set follows the camera (old region evicts, new one
    streams in) — `chunk_count() == 0` is the wrong expectation.
  Authoring gotcha: rustfmt glues a `// ANCHOR_END` line to a
  preceding trailing comment (indents it to the comment column) —
  never end an anchor block with a line that carries a trailing
  comment. `Vxl::empty(CHUNK_SIZE_XY)` is the canonical chunk shape
  for generators (identical to what `ensure_chunk` makes) — the book
  teaches it instead of the demo's grid-detach idiom.
- BK.2 — **LANDED 2026-07-05.** Ch. 4 (rendering & backends) + ch. 5
  (render pipeline). One shared anchor example
  `roxlap-render/examples/book_pipeline.rs` (windowed, quickstart-
  style; verified live for 8 s on the CPU backend) — anchors
  `gizmo` / `backend_select` / `supports` / `pipeline` /
  `frame_params` / `overlay`. Decisions taken while writing:
  - The `Feature` parity table is NOT copied into the book (ch. 4
    links the rustdoc — the table changes as parity gaps close).
  - `paint_egui` has no anchor: the `hud` feature is non-default, so
    a `required-features` example would be skipped by CI's
    `--all-targets` (silent rot). Ch. 5 shows the 3-line tessellate/
    paint idiom inline (≤3 lines = policy-legal) + links the
    scene-demo host.
  - MSRV gotcha: `Option::is_none_or` trips `clippy::incompatible_msrv`
    — examples use `map_or` like the quickstart.
- BK.3 — **LANDED 2026-07-05.** Ch. 6 (lighting & materials). Anchor
  example `roxlap-render/examples/book_lighting.rs` (windowed,
  verified live 8 s CPU) — anchors `bake` / `volumetric` /
  `materials` / `light_rig`: AO bake (lightmode 3 + AoParams),
  per-frame LightRig (sun sweep + 2 points + spot cone + stylized
  bands), material palette (glass terrain wall + volumetric fog cloud
  via `from_fn_keep_interior`). Drive-by fix: `LightRig`'s rustdoc
  said "GPU-only" — stale since CPU.1/CPU.2; now says both backends.
- BK.4 — **LANDED 2026-07-05.** Ch. 7 (sprites & animation) + ch. 8
  (particles). Two anchor examples (both verified live on CPU):
  - `book_sprites.rs` — anchors `clip_build` / `model_instances` /
    `clip_instances` / `billboard` / `per_frame` (scale-via-basis,
    spinning instance, clip player, cylindrical billboard, tick()).
  - `book_particles.rs` — anchors `system` / `tick` / `explosion`
    (fountain + smoke defs, tick_with_scene, carve_debris with
    scripted 4 s explosions).
  Characters (.rkc) + KFA covered in prose only (asset-driven; the
  Animation scene + roxlap-host are the worked examples). GIF/PNG
  billboard import prose-only for the same reason as paint_egui
  (non-default features ⇒ a required-features example rots).
- BK.5 — **LANDED 2026-07-05.** Ch. 9 (asset pipeline) + ch. 10
  (picking & queries). Anchor examples:
  - `roxlap-formats/examples/book_assets.rs` — headless,
    assert-checked; anchors `vox` / `kv6_roundtrip` / `rvc_roundtrip`
    / `vxl_roundtrip`. Synthesises a minimal .vox **in code** (SIZE +
    XYZI, default palette) so the import anchor needs no binary asset;
    asserts byte-stable serialize(parse(x)) == x for kv6.
  - `roxlap-render/examples/book_picking.rs` — windowed (verified
    live 8 s CPU); anchors `hover_raycast` (view_ray + Scene::raycast
    per frame) / `click_pick` (pick → carve → bake_lightmode_bbox).
    Teaches the two-path split: raycast per-frame, pick click-time
    (GPU depth readback blocks).
  Collision stays prose (points at demo collision.rs; CC stage will
  supersede). gif/png import prose-only (non-default features).
- BK.6 — **LANDED 2026-07-05.** Ch. 11 (platforms) + ch. 12 (tuning)
  + ch. 13 (demo tour) — prose chapters, no new anchor examples
  (nothing here is snippet-shaped; wasm code is cfg-gated off native
  CI so anchors there would be unverified). The copy-prone tables per
  hazard 4: engine env vars (4: GPU_MIP_SCAN_DIST / GPU_CHUNK_BUDGET /
  GPU_CLIP_BUDGET / GPU_POWER) live in tuning.md, demo vars in
  demo-tour.md — **both carry an HTML source-of-truth comment naming
  the grep**; ROXLAP_GPU itself is documented as a demo convention,
  not an engine variable. Parity table again rustdoc-linked, not
  copied. Grep-verified 23 vars total on 2026-07-05; ROXLAP_DITHER
  values are none|bayer|blue (default blue).
  **All 13 chapters now written** — content-complete; BK.7 remains.
- BK.7 — **LANDED 2026-07-05. Stage BK CLOSED.** Author confirmed
  gh-pages deploy + agent-side GIF capture. Delivered:
  - **gh-pages**: CI `book` job uploads a Pages artifact on master
    pushes; new `deploy-book` job publishes via actions/deploy-pages
    to https://ncrashed.github.io/roxlap/ (repo Settings → Pages must
    be set to "GitHub Actions" once). `site-url = "/roxlap/"` in
    book.toml. README + all four crate rustdoc headers link the
    published URL ("New to roxlap? The roxlap book…").
  - **GIF gallery**: `docs/gallery/{lighting,particles,doom,
    transparency}.gif` (~324 KB total) in a README table. Captured
    via a new demo-host recorder: `ROXLAP_CAPTURE=<dir>` (+`_MS`,
    `_FRAMES`) writes PPMs on a wall-clock interval then exits, HUD
    forced off; `ROXLAP_CAMERA=x,y,z,yaw,pitch` overrides the start
    pose (Transparency needed reframing; Animation's start pose shows
    empty sky — its GIF was dropped, re-shoot after re-framing).
    Recipe: 480×270 fixed res + POSTERIZE=6 + DITHER=bayer, 40 frames
    @80 ms, `magick -delay 8 -loop 0 frames -layers Optimize out.gif`.
  - **Editorial pass** (avoid-ai-writing audit, docs profile): prose
    came out clean on vocabulary/template patterns; fixed the one
    structural tell (three chapters opened with the same
    "Everything …" shape) + a "showcases"; foreword updated now that
    all chapters exist.
  Both new env vars added to demo-tour.md's table (the source-of-truth
  grep found them immediately — the gate works).

## Goal

A user-facing **mdBook** that teaches roxlap to a game developer who
has never seen the codebase. Today that person has a 332-line README,
docs.rs API reference, and nothing in between — the engine's entire
accumulated knowledge (conventions, architecture, asset workflows,
tuning) lives in 18 internal `PORTING-*.md` stage histories (~6.4k
lines) written for *us*, not for users. The book repackages that
knowledge; the porting docs stay untouched as historical records.

## Audit facts (2026-07-04/05, verified — do not re-survey)

- `docs/porting/`: 18 stage docs, internal history only. No mdBook, no
  tutorials, no architecture guide anywhere.
- README.md (332 lines): good pitch + quickstart + crate table +
  runnable-demo commands. One static screenshot for 11 demo scenes.
- CHANGELOG.md: exemplary *migration* reference (was→now tables), not
  a teaching document.
- Rustdoc: `missing_docs` enforced on roxlap-render + roxlap-scene
  (CI `-D warnings`); README is compiled as a doctest in roxlap-render
  (the anti-rot precedent this stage generalises). roxlap-gpu /
  -formats / -core still owe ~250 field docs (QE leftover, separate).
- Examples (CI-compiled via `--all-targets`):
  `roxlap-render/examples/quickstart.rs` (~140 lines, winit + orbit),
  `roxlap-formats/examples/parse_kv6.rs`,
  `roxlap-gpu/examples/probe.rs`. **There is no
  `roxlap-core/examples/hello.rs`** — an earlier survey hallucinated
  it (it saw the legacy `roxlap-host` demo binary). Verify every file
  reference before citing.
- Demo scenes = the de-facto feature gallery (11 tabs in
  roxlap-scene-demo): World, Sprites, Animation, Transparency,
  Lighting, Spotlight, Particles, Doom, Picking, Primitives, Empty.
  `ROXLAP_SCENE=<name>` starts on any of them.
- CI: fmt + clippy `-D warnings` + `test --workspace` (examples
  compile-checked). No docs job.
- **mdbook is NOT in the dev shell** — add to `flake.nix` `packages`
  (~line 67) in BK.0; re-enter `nix develop` after.

## Decisions to lock at BK.0 (proposed; confirm with the author)

1. **Tool/layout**: mdBook, source at `docs/book/` (`book.toml` +
   `src/SUMMARY.md` + one file per chapter). Build output ignored.
2. **Language**: English (repo convention; README/rustdoc are English).
3. **Anti-rot policy (the load-bearing rule)**: every code snippet
   longer than ~3 lines lives as a **compile-tested artifact** — a
   crate example or a rustdoc doctest — marked with `// ANCHOR:` /
   `// ANCHOR_END:` comments and pulled into the book via mdBook's
   `{{#include ../../crates/...rs:anchor}}`. Never paste copies.
   (`mdbook test` is NOT used — snippets need workspace deps it can't
   see. CI compiling the examples is the guard.) The README's only
   code block already follows the spirit via its doctest.
4. **CI**: a job that installs mdbook (nix or `cargo install --locked
   mdbook`) and runs `mdbook build docs/book` — a broken include =
   red CI. Deployment (gh-pages) is a maintainer decision — ask at
   BK.7, not before.
5. **README relationship**: README stays the pitch + 40-line
   quickstart and gains a prominent link to the book; depth moves to
   the book. Never fork the same prose into both.
6. **PORTING docs**: unchanged, linked from the book where design
   rationale helps ("why per-pixel DDA replaced opticast" →
   PORTING-DDA).

## Chapter plan → source mapping

| # | Chapter | Feeds from |
|---|---|---|
| 1 | Introduction & quickstart | README, `quickstart.rs` |
| 2 | Concepts & conventions | **+z is DOWN**; packed colours `0x80_RR_GG_BB` (brightness-in-alpha!); voxel = 1 world unit; camera basis + chirality footgun (`roxlap-core/src/camera.rs` warning, `Camera::from_yaw_pitch`); f64 world / f32 sprites; PORTING-RUST/-SCENE intros |
| 3 | The scene graph | grids/chunks/GridTransform; edits (`set_voxel/rect/sphere`, colfunc, SpanOp; **recolour gotcha: `set_rect(Some)` over solid keeps old colours — carve then insert**); snapshots (versioned envelope, QE.5); streaming + `ChunkGenerator` + `ChunkStore`; PORTING-SCENE |
| 4 | Rendering & backends | SceneRenderer facade; `BackendPreference` + auto-fallback; `render → overlays → present`/`paint_egui` protocol; `FrameParams::new` builder; `supports()` parity query (QE.7); CPU DDA vs GPU compute overview; PORTING-DDA/-GPU |
| 5 | Render pipeline | fixed logical res / `Scale` / `Native`, SSAA, posterize + dither, egui HUD; PORTING-PIPELINE |
| 6 | Lighting & materials | baked `bake_lightmode` + AO as the ambient byte; runtime `LightRig` (sun/point/spot, stylized shadows, bands); TV materials (alpha/additive/**volumetric needs `from_fn_keep_interior`**); `set_terrain_materials` water/glass; PORTING-DYNLIGHT/-SPOTLIGHT/-TRANSPARENCY |
| 7 | Sprites & animation | kv6 models + dynamic instances (+scale via basis); voxel clips `.rvc`; characters `.rkc`; KFA rigs; billboards + actors (Doom-style); PORTING-SPRITE-API/-VOXEL-CLIP/-BILLBOARD |
| 8 | Particles | `ParticleSystem` end-to-end (the Particles demo as the worked example); PORTING-PARTICLES |
| 9 | Asset pipeline | MagicaVoxel `.vox` import; `.kv6`/`.kvx`/`.vxl`/`.kfa` readers; GIF/PNG → clips; snapshot save format; formats rustdoc + QE.6 |
| 10 | Picking & world queries | `pick`/`view_ray`/`pick_depth`; `Scene::raycast`/`resolve_voxel`; the demo's hand-rolled collision as the interim pattern (CC stage will supersede) |
| 11 | Platforms | wasm (WebGPU + WebGL2 fallback, threads/COOP-COEP, size numbers), SDL/raw-window-handle genericity; PORTING-WASM |
| 12 | Performance & tuning | env-var table (21 `ROXLAP_*` — grep-verify at each stage close; QE-C6 `RenderConfig` is the eventual fix); `RenderOptions` knobs; mip/LOD + scan distances; streaming radii; PF lessons (no per-frame env reads, batch transforms); CPU↔GPU parity table (QE-B5 state) |
| 13 | Demo tour | 11 scenes ↔ features each showcases; `ROXLAP_SCENE`, probe env vars |

## Stage list

| Phase | Contents |
|---|---|
| BK.0 | Confirm decisions; mdbook → flake dev shell; `docs/book/` skeleton (book.toml, SUMMARY with all chapters stubbed); chapter 1 (from README + quickstart include-anchors); CI `mdbook build` job; README links the book |
| BK.1 | Ch. 2 concepts + ch. 3 scene graph |
| BK.2 | Ch. 4 rendering + ch. 5 pipeline |
| BK.3 | Ch. 6 lighting & materials |
| BK.4 | Ch. 7 sprites/animation + ch. 8 particles |
| BK.5 | Ch. 9 assets + ch. 10 picking/queries |
| BK.6 | Ch. 11 platforms + ch. 12 tuning + ch. 13 demo tour |
| BK.7 | Editorial pass (AI-ism sweep, voice consistency); README GIF gallery; docs.rs ↔ book crosslinks; ask author about gh-pages deploy |

Optional follow-on (own mini-stage, not BK): `roxlap-cli` — `.vox →
.kv6/.rvc` conversion + snapshot inspection; high indie value, zero
coupling to the book.

## Hazards

1. **Snippet rot** is the failure mode of every engine book — the
   include-anchor policy (decision 3) is mandatory, not aspirational.
   New anchors go into `quickstart.rs`-style examples per chapter
   (e.g. `examples/book_lighting.rs`), CI-compiled.
2. **Voice drift / bloat**: chapters are teaching material, not stage
   diaries — resist pasting porting-doc prose wholesale; it explains
   *how we got here*, the book explains *how to use it*.
3. Facts about the engine change under the book (e.g. QE-B6 leftovers
   will rename APIs) — each BK phase ends with `mdbook build` green in
   CI, and API mentions prefer linked rustdoc over restated
   signatures.
4. Env-var table and parity table are copy-prone — both get a "source
   of truth" comment naming the grep / the QE-B5 table they must match.
5. The book adds a second consumer of `quickstart.rs` — editing it now
   breaks two things (doctest + include); fine, but know it.

Versioning: docs + flake/CI only — no crate release. Book ships
in-repo; deployment decided at BK.7.
