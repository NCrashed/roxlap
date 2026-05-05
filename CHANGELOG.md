# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-05-XX

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
- Multicore CPU rendering (R12): single-threaded today.
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

[0.1.0]: https://github.com/NCrashed/roxlap/releases/tag/v0.1.0
