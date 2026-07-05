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
- BK.1..7 — not started.

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
