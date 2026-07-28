# Handover: stale chunk mips after runtime `set_rect` edits ("invisible walls")

Status: **CLOSED 2026-07-28** — root cause was NOT stale mips but the
voxlap format's uncarvable chunk-local z=255 voxel landing on the
digger's sim surface (see the ROOT CAUSE section below, appended
2026-07-25). Fixed by the **CT carve-through-floor stage**
(`docs/porting/PORTING-CARVE.md`, CT.0..CT.9): empty-column sentinel
encoding, carve-through edit ops, placeholder retirement, walker/
renderer/collision agreement on both backends, islands un-anchoring,
snapshot wire v5. Verified fixed in the monada digger by the reporter
(visual pass, 2026-07-28). Ships in the next minor cut.

Originally: **open bug**, reported 2026-07-25 from the monada digger demo
(roxlap 0.30.0 as published on crates.io). This doc is written for a
fresh session working in THIS repo; everything needed to reproduce,
diagnose and validate is spelled out. File:line references below are
against the 0.30.0 sources (crates.io mirror at
`~/.cargo/registry/src/index.crates.io-*/roxlap-*-0.30.0/`); the repo's
`crates/roxlap-*` trees should match modulo post-0.30 work.

## Symptom

In the monada digger demo (a vehicle drilling tunnels through a voxel
terrain grid, carving via `Grid::set_rect(.., None)` at ~10 cells per
tick), digging DOWNWARD quickly produces "invisible walls": the player
carves cells that are genuinely gone from the grid (raycasts agree they
are air), but the render shows coarse ghost geometry — in the
reporter's words, *"почти сразу карта переключается на mip 1 и мы будто
копаем невидимые стены, что примерно отображаются мипкартой выше
уровня"* — the scene switches to mip 1 almost immediately and the
ghost walls match what a coarser mip of the PRE-CARVE terrain would
look like.

Two independent facts make this surprising:

1. The host explicitly sets `RenderOptions::gpu_mip_scan_dist: 8192.0`
   (monada-host `src/lib.rs` ~1802, a carry-over from the ship demo's
   deck-cutaway fix, commented "Keep the whole scene at GPU mip-0") —
   at that threshold the whole 2560-world-unit scene should never leave
   mip 0 on the GPU.
2. The affected grid carries `mip_levels_override = Some(1)`
   (documented in `roxlap-scene/src/lib.rs:493-502` as "Some(1) =
   mip-0 only").

So EITHER something samples coarse mips despite both knobs, OR the
knobs don't reach the sampler for this grid configuration.

## The affected grid's configuration (unusual, likely relevant)

The digger's terrain is ONE large grid with a **non-unit
`voxel_world_size = 16`** (isotropic "one grid voxel per sim cell"):

- `GridTransform { origin: (0, 0, 100), rotation: IDENTITY, voxel_world_size: 16.0 }`
- extent ~160×160 grid voxels in x/y, z spans roughly −24..+15
  (grid-local voxels; negative coordinates in x and z — the grid is
  addressed at `(-x-1, y, -z-1)` from sim cells),
- `mip_levels_override = Some(1)`, `render_sky` default (true),
- a per-frame `z_clip` follows the vehicle (deck cutaway),
- runtime edits: `set_rect(c, c, None)` hole-punches at ~10 cells/tick
  while drilling, plus large `set_rect(.., Some(colour))` fills at map
  init.

vws = 16 matters twice: (a) each mip-0 voxel is already 16 world units,
so any distance-based mip policy in world units fires 16× "closer" in
voxel terms than on a vws-1 grid; (b) if any threshold comparison mixes
world units with grid-local (already ÷vws) units, this grid is exactly
where it shows.

## Primary suspect: the scene edit path never regenerates mips

Established by reading the 0.30.0 sources:

- `Scene::set_rect` → `apply_set_rect` (`roxlap-scene/src/edit.rs:280-302`)
  edits the chunk `Vxl` via `roxlap_formats::edit::set_rect` and calls
  `bump_chunk_version_bbox` — and NOTHING else. No mip regeneration.
- `Vxl::generate_mips` / `Vxl::remip_bbox` exist
  (`roxlap-formats/src/vxl.rs:389` and around :346-399) — but in the
  whole of roxlap-scene 0.30.0 they are invoked **only from tests**
  (`chunks.rs:1075-1126`, `remip_bbox_matches_generate_mips`). No
  production caller.
- The engine's own docs assign remipping to "the caller":
  - `roxlap-scene/src/islands.rs:62-64`: "caller rendering distant
    chunks must remip the touched columns (`Vxl::remip_bbox`) exactly
    as it already does for its carves" — but the scene's carve path
    (above) does NOT do it;
  - `roxlap-scene/src/chunks.rs:410-416` (`bake_bbox`): "Mip
    regeneration is NOT performed — near-field renders read mip 0;
    callers streaming distant edited chunks should remip as they
    already do for edits."

Consequence: **any sampler that reads mip ≥ 1 after a runtime edit sees
the pre-edit solid** — which is exactly the reported ghost geometry.
The remaining question is only *why* a mip ≥ 1 is sampled at all given
the two knobs above.

## Why is mip ≥ 1 sampled? Three hypotheses to check, in order

1. **`ROXLAP_GPU_MIP_SCAN_DIST` env override.** The GPU threshold is
   "seeded from `RenderOptions::gpu_mip_scan_dist` (env …)"
   (`roxlap-render/src/gpu.rs:156-159`; env table at `lib.rs:1551`).
   If the dev shell exports a small value from old experiments, it
   silently overrides the host's 8192. One-minute check:
   `echo $ROXLAP_GPU_MIP_SCAN_DIST`, then run monada-digger with
   `ROXLAP_GPU_MIP_SCAN_DIST=100000`. If this fixes it, the bug
   reduces to the stale-mip issue above (still worth fixing) plus a
   footgun-y env default.
2. **Units mismatch on non-unit `voxel_world_size`.** The mip-select
   distance is documented in world units, but per-grid rendering
   rebases the camera into grid-local voxels by dividing everything by
   vws (`roxlap-scene/src/render.rs:101-122`, SC notes; GPU:
   `grid_cameras`/`grid_local_camera`, `gpu.rs:1822-1916`). If the
   mip-select compare happens in grid-local units against a threshold
   held in world units (or vice versa) the effective threshold is off
   by 16× on this grid. Check where `set_scene_mip_scan_dist`
   (`gpu.rs:1210, 994-1000`) lands in the shader/driver compare and
   which space the per-step distance is in. NB a ÷16 error would make
   mips fire LATER, not sooner — a ×16 error (grid-local distance
   compared against a world threshold ÷ vws somewhere else) fires
   sooner. Also check the CPU DDA brick path: monada's per-frame
   `OpticastSettings` has `mip_levels: 1, mip_scan_dist: 4`
   (`roxlap-core/src/opticast.rs:99-105` defaults via
   `for_oracle_framebuffer`), so the CPU backend *should* be mip-free —
   if the repro persists with `ROXLAP_GPU=0`, the story is different
   and hypothesis 2/3 moves to the CPU brick cache
   (`ensure_dda_bricks`, `chunks.rs` PF.13 notes).
3. **`mip_levels_override` not honoured by the sampler.** The field is
   documented "Some(1) = mip-0 only" (`roxlap-scene/src/lib.rs:501`)
   and monada sets it on the terrain grid AND on every small
   rotating prop grid. Verify the GPU sampler actually clamps its mip
   choice to the grid's available levels rather than assuming a full
   ladder — a sampler that steps to mip 1 on a 1-level grid reads
   whatever bytes sit there (stale or garbage), which also matches the
   symptom.

## Suggested engine-side fixes (any one suffices; the first is the real one)

- **Remip on edit**: `apply_set_rect` / `apply_set_sphere` /
  `set_rect_with_colfunc` call `Vxl::remip_bbox` over the edited bbox
  (plus the defensive belt `remip_bbox` already documents) whenever the
  chunk has more than one mip level. Cost: the digger carves ~10
  cells/tick; `remip_bbox` was measured cheap for bullet-hole-scale
  edits (`chunks.rs:410-416` cites 0.04 ms vs 4-7 ms for full-grid).
  This makes coarse mips *honest* and fixes every consumer regardless
  of thresholds.
- **Honour `mip_levels_override` end-to-end**: clamp the mip select to
  the grid's level count on both backends. Cheap, and turns the
  monada workaround (Some(1) on the terrain grid) into a guarantee.
- **Threshold unit audit** for non-unit vws grids (hypothesis 2), if
  the audit finds a mix-up.

## How to reproduce & validate

Repro (in `~/dev/monada`, working tree already carries the demo):

```sh
cargo run -p monada-digger        # GPU by default; ROXLAP_GPU=0 for CPU
# drive to the mountain (E to drill), pitch down with F, dig a few
# cells deep — ghost walls appear almost immediately on the affected
# setup; the drill cuts through them (the sim store is correct).
```

Validation after an engine fix: same manual run — walls must match the
carved store everywhere. monada's determinism goldens
(`monada-hashes.txt`) hash only sim state, never the render, so a
roxlap render fix CANNOT break them; monada picks the fix up via a
version bump in its workspace `Cargo.toml` (currently pinned to
crates.io 0.30.0 precisely so the local roxlap working copy does not
perturb monada builds — publish 0.30.1 or temporarily switch monada to
a path dep for verification, then back).

## What the consumer (monada) already does — do not double-guard

- `RenderOptions::gpu_mip_scan_dist: 8192.0` at renderer construction.
- Terrain grid + all prop grids: `mip_levels_override = Some(1)`;
  prop grids also `render_sky = false`.
- Per-frame CPU `OpticastSettings`: `mip_levels: 1`.
- The deck `z_clip` on the terrain grid changes per frame (known
  0.30 interaction: coarse mips ignore `z_clip` — the ship demo's
  original reason for the 8192 threshold; a mip fix should keep that
  case in mind).

## Verification results (2026-07-25, against the repo working tree + the
## crates.io 0.30.0 sources monada actually locks)

Each claim above was checked in code; several change the picture.

### Confirmed facts

- **Edits never remip** — TRUE. `apply_set_rect` / `apply_set_sphere` /
  `set_rect_with_colfunc` (`roxlap-scene/src/edit.rs:280-330`) only call
  `bump_chunk_version[_bbox]`; `Vxl::remip_bbox`/`generate_mips` have
  zero production callers in roxlap-scene (tests only).
- **`mip_levels_override` is ignored by the GPU backend** — TRUE, but
  not the way hypothesis 3 guessed. The GPU never reads the field (no
  reference outside roxlap-scene); `decompress_chunk`
  (`roxlap-gpu/src/decompress.rs:237-252`) always builds the FULL
  `gpu_mip_count(vsid)` ladder itself — when the `Vxl` has fewer mips it
  clones and `generate_mips` at upload. So the sampler never reads
  garbage; `grid_static_meta.mip_count` is always the full ladder and
  `pick_mip` clamps only to that (`scene_dda.wgsl:595-602`). The CPU
  backend DOES honour the override (`roxlap-scene/src/render.rs:1283`,
  `global_mip_cap`).

### The twist: on the digger's config, GPU mips are NOT stale

Because the terrain chunks carry `mip_count() == 1` (override = Some(1),
runtime-filled — nothing ever generated mips), the partial-refresh fast
path bails on its precondition (`roxlap-gpu/src/scene.rs:886`,
`vxl.mip_count() < layout.mip_count → false`) and EVERY edit falls back
to a full `decompress_chunk` (`roxlap-render/src/gpu.rs:1641-1677`),
which regenerates the whole 6-level ladder FRESH from post-edit mip-0.
Two consequences:

1. The stale-mip suspect does not apply to the digger. (It IS real for
   grids whose chunks already carry ≥ 6 mips: there
   `refresh_chunk_partial` re-derives all levels from the Vxl's own
   never-remipped mip tables — stale — and the full path reads them
   directly. "Remip on edit" is still a correct engine fix for THAT
   case.)
2. Even an HONEST mip 1 reproduces the symptom on this grid: a 1-cell
   tunnel aggregates back into solid at 2×2×2 (monada's own comment at
   `map_render.rs:1089-1094` says exactly this). So **remip-on-edit
   would NOT fix the digger** — the fix must stop mip ≥ 1 from being
   sampled at all.

### Hypotheses 1-3 status

- **H1 (env override)**: plumbing verified clean — opts seed the backend
  at construction (`roxlap-render/src/gpu.rs:226`) and are re-pushed
  every frame inside the main `fn render`
  (`gpu.rs:1125`/`1210`). This dev shell has no
  `ROXLAP_GPU_MIP_SCAN_DIST` set. Still worth `echo`-ing in the repro
  shell, but no engine bug here.
- **H2 (units mismatch)**: REFUTED in the "fires sooner" direction. The
  primary march picks `pick_mip(t_enter / vws)`
  (`scene_dda.wgsl:1291`, projected-size LOD, present in published
  0.30.0 too) — a vws=16 grid coarsens 16× LATER: mip 1 needs
  t ≥ 2·8192·16 = 262 144 world units. The shadow path uses unscaled
  world t (`:700`) → needs t ≥ 16 384. The scene is ~2 560 units. The
  camera rebase (`grid_local_camera`, roxlap-render `gpu.rs:1911`) does
  not divide by vws; the marcher scales chunk dims by vws, so t is
  world units — consistent. Negative chunk indices (the digger grid
  spans chz −1..0, negative chx) are handled: `p_chunk` comes from a
  true floor (`scene_dda.wgsl:1233`), pow2 `&`-mask slot lookup is
  negative-safe (`:504-510`) with an identity check (`:539-549`), and
  `z_clip` mip math is coherent for negative z.
- **H3 (override not honoured)**: see above — true as a fact, wrong
  about the mechanism (no garbage reads, and the ladder is fresh on
  this grid).

### Net verdict

With 8192 verified to reach `u.mip_scan_dist` every frame, **neither
backend can select mip ≥ 1 anywhere in the digger scene** (CPU:
`mip_levels: 1` + `LodThresholds::default() == always_near()` + the
override cap). So either the threshold does not actually arrive in the
failing run, or the ghost walls are not distance-mip LOD at all.

Discriminating experiment (2 min, run on the failing setup):

1. Check the stderr line `monada-host: GPU backend — …` vs
   `CPU backend` (NVK PRIME breakage on this box historically forced
   CPU fallback — see `reference_nvk_prime_explicit_sync_breakage`).
2. Run with `ROXLAP_GPU_MIP_SCAN_DIST=0` — `0` DISABLES LOD entirely
   (`pick_mip` returns 0 unconditionally, `scene_dda.wgsl:596`).
   - Ghosts persist at 0 ⇒ not mips at all → suspect the upload/sync
     path or `z_clip` (note: monada locks crates.io 0.30.0, which lacks
     the 0.30.1 `ceil(z_clip / 2^mip)` fix — though that leak also
     needs mip ≥ 1).
   - Ghosts gone at 0 but present at 8192 ⇒ `pick_mip` really fires →
     instrument the uniform value (something clobbers it).

### Revised fix ranking (superseded — see ROOT CAUSE below)

1. **Honour `mip_levels_override` on the GPU** (per-grid mip cap in
   `grid_static_meta`, clamp `pick_mip`) — still a worthwhile
   guarantee, but NOT the digger's bug.
2. **Remip on edit** — still right for honesty of coarse mips on
   full-ladder grids (the partial-refresh path re-uploads stale
   tables), but it does not fix the digger.
3. Threshold unit audit — DONE, no mix-up found.

## ROOT CAUSE (2026-07-25, reproduced headless — NOT mips at all)

Reproduced in `crates/roxlap-gpu/tests/digger_repro.rs` and
`roxlap-scene/src/render.rs::digger_bedrock_boundary_layer_cpu_probe`
(both `#[ignore]`d KNOWN-RED — run with `--ignored`), with the digger's
exact shape: vws = 16, negative chunk indices, terrain slab crossing the
chz −1|0 boundary, a shaft carved through the boundary, camera near.

**A voxel at chunk-local z = 255 can never be carved.** Voxlap's
`delslab` clamps `z1` to `MAXZDIM−1`
(`roxlap-formats/src/edit.rs`; pinned by its own test
`set_rect_clamps_to_world`: carving an entire column leaves the solid
run `[255, MAXZDIM)`) — the RLE column format cannot represent "no
floor". For single-chz grids that voxel is deep underground and
invisible. The digger's mirrored addressing (`grid_z = −sim_z − 1`)
puts **sim z = 0 — the surface, the most-carved layer of the game — at
chunk-local z 255 of chunk chz = −1**. Every surface-cell carve
silently leaves that voxel solid in the render grid:

- GPU repro: centre ray down the carved shaft stops at depth 284 =
  grid z −1 (the stuck voxel) instead of 444 (the shaft floor).
  **Identical at `mip_scan_dist` 8192 and 64** — threshold-independent,
  marched at mip 0; the "mip 1" impression came from the stuck layer
  sketching the pre-carve surface (GPU bedrock hits inherit the colour
  of the textured surface above — smeared/blocky when that surface is
  itself carved).
- Scene-level probe: after `Grid::set_rect(.., None)` over grid
  z −8..8, `grid.voxel_solid((-16,16,-1))` returns **true** — engine
  collision/raycasts see the wall too (monada's own sim store doesn't,
  hence "the drill cuts through ghosts").
- `z_clip` HIDES the layer when the deck clip drops below it (the
  deck-clip variant of the GPU test passes) — explaining why the ghosts
  come and go with the roof-probe deck cutaway while drilling.
- "либо внизу, либо вверху": digging down hits the layer from above at
  sim 0; digging up from a tunnel hits the same layer from below.

Secondary CPU find (same probe): the CPU marcher falls through the
carve-exposed UNTEXTURED top surface (hits the textured floor at depth
540 instead of solid z 9 at 444) — the scene-level fill+carve leaves the
newly exposed top without a colour, and the CPU DDA only draws textured
cells. Likely invisible in monada (per-cell paints texture everything),
but it breaks CPU/GPU parity at carve sites in general.

### Fix directions

- **Engine (real fix)**: legalize carve-through-floor — let a column
  become genuinely empty (or carve local z 255) at the formats level,
  with both samplers, the VC stitcher, collide and GPU decompress
  following. This is a format-invariant change (voxlap requires a
  floor span) — a proper substage, not a patch.
- **Engine (cheaper, partial)**: on multi-chz grids, treat a
  1-voxel-thick solid at local z 255 whose neighbours (local 254 and
  the chunk-below's local 0) are air as a carve artifact and skip it in
  both samplers + `voxel_solid`. Heuristic — can mis-hide a legitimate
  1-voxel floor plate sitting exactly on a chunk seam.
- **monada (immediate unblock)**: shift the volume addressing so the
  diggable band never touches chunk-local z 255 — e.g. map sim z = 0
  to grid z ≈ +100 instead of −1 (adjust `GridTransform.origin.z` by
  the same amount to keep world placement); the whole −24..+15 sim band
  then lives inside chz 0, and the uncarvable plane sits in the
  indestructible bedrock frame where carves never reach.
