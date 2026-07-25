# roxlap — carve-through-floor (Stage CT)

Entry doc written 2026-07-25 at workspace 0.30.1.
This is the **entry doc** for the carve-through-floor stage — tag **CT**.

## Why

`delslab` clamps `y1` to `MAXZDIM-1` (`roxlap-formats/src/edit.rs:39`),
so **chunk-local z = 255 can never be carved** — the voxlap RLE column
format cannot even *represent* "no floor": every walker seeds the
terminal run as `[slab[1], 256)`, and `slab[1]` is a `u8`. For
single-chz grids the stuck voxel is deep underground and invisible.
The monada digger (volume terrain, `grid_z = −sim_z − 1`) put **sim
z = 0 — the surface, the most-carved layer of the game — at local 255
of chunk chz = −1**: every surface carve silently leaves a solid voxel
in the render grid → the "invisible walls" report
(`docs/handover-stale-mips-volume-edits.md`, ROOT CAUSE section).

Worse: the bedrock **placeholder** columns of a materialised-but-
unpainted region carry the same z=255 voxel, so the ghost layer spans
whole chunk footprints, not just carved cells.

Repro gates already in-tree, `#[ignore]`d KNOWN-RED (run `--ignored`):

- `crates/roxlap-gpu/tests/digger_repro.rs` — headless GPU shaft
  through the chz −1|0 boundary; ghost depth 284 vs expected 444.
- `roxlap-scene/src/render.rs::digger_bedrock_boundary_layer_cpu_probe`
  — scene-level; also pins the SECOND bug (below).

A second, independent defect found by the same probe: **the CPU DDA
couples the hit verdict to the colour fetch** (`Sampler::hit`,
`roxlap-core/src/dda.rs:1405-1407` → `surface_color_mip`,
`grid_view.rs:357-367`). A carve paints newly-exposed tops colour 0
(`edit.rs:1008-1017`), `voxel_color_mip` treats RGB 0 as untextured
(`grid_view.rs:400-404`), so the marcher steps *through* solid
uncoloured voxels to the next textured surface (CPU probe hits depth
540 instead of 444). The GPU decouples the two (`solid_occupancy` +
colour inheritance, `roxlap-gpu/src/decompress.rs:23-33`) — a CPU/GPU
parity break at every carve site.

## Locked design decisions

1. **Empty-column encoding: a 4-byte sentinel slab, never a
   zero-length column.** `slng`, `column_data`, and the snapshot
   serializer index `slab[0..4]` unconditionally — a 0-byte column is
   an OOB. The sentinel is the natural degenerate: a column whose ONLY
   slab is terminal with an *empty* floor-colour list —
   canonical bytes `[nextptr=0, z1=255, z1c=254, z0=0]`
   (`n_floor = z1c − z1 + 1 = 0`). Read predicate:
   `column_data.len() == 4 && slab[2] < slab[1]`. `parse` already
   tolerates it (`vxl.rs:1362-1381`, `last_size = 4`); no writer today
   (voxlap's or ours — `compilerle` always emits ≥ 1 floor colour)
   ever produces it, so reinterpretation is safe.
2. **The bedrock placeholder convention dies.** `Vxl::empty` /
   `seeded_air` columns become truly empty (the sentinel) instead of
   the `[255, 256)` bedrock voxel. The placeholder existed for the
   opticast/VC era; the production renderer is the DDA backend
   (opticast and the VC stitcher are deleted — `opticast.rs` is
   settings-only), and both backends handle "no solid in column" as
   plain sky/pass-through already. The S1.X-era tests that lock the
   placeholder (`chunks.rs:812-843`) are replaced, not appeased.
3. **Carve retexture: newly-exposed voxels inherit a colour instead
   of 0.** The single decision point is `compilerle`'s colfunc branch
   (`edit.rs:342-349`): the default carve colfunc changes from
   `VoxColor(0)` to "inherit the nearest surviving colour record of
   the same column" (voxlap behaviour); `set_*_with_colfunc` callers
   keep full control. RGB-0 stays the *untextured* sentinel — but the
   default carve no longer manufactures it.
4. **CPU hit verdict comes from solidity, not colour.**
   `Sampler::hit` gates on `voxel_run_top_mip` (the run walk the
   bricks are built from); colour resolves separately with the GPU's
   fallback ladder (own colour → run-top colour → `BEDROCK_RGB`).
   Backends align on the RGB-0 policy and a parity gate pins them.
5. **Islands semantics: carving the floor out un-anchors.** The DT
   support model anchors a column's final run at z=255
   (`islands.rs:7-12`, `:334-335`). With carve-through, an emptied
   column contributes NO anchored run — hanging remains become debris.
   That is the *physically correct* reading, and it is a deliberate
   behaviour change, gated by a new DT test.
6. **Version: next cut is MINOR with a loud behaviour note.** No API
   breaks planned (additive helpers only), but carve semantics change
   (`[0,256)` now empties a column) and snapshots containing sentinel
   columns render floors on ≤ 0.30 readers. Snapshot wire gets a
   version bump with the frozen-shadow-shape test pattern (as CA.0
   did for v4).

## Substages

- **CT.0 — sentinel + helpers (no behaviour change).**
  `roxlap-formats`: `EMPTY_COLUMN_SLAB` constant,
  `Vxl::column_is_empty(idx)`, doc on the encoding. RED formats tests
  written for the target behaviour (carve `[0,256)` → empty column;
  `voxel_color`/walkers report air). Gate: helpers round-trip through
  `parse`/serialize; full suite byte-identical.
- **CT.1 — edit ops carve through.** `delslab` drops the clamp
  (empty spans list = sentinel-first); `expandrle` on the sentinel
  emits zero runs; `compilerle` emits the canonical sentinel when the
  run list is empty; `insslab` re-inserts into an empty column;
  `set_rect`/`set_sphere`/`set_cube`/`set_spans` + `ScumCtx`
  flush path carry it. `delslab_y1_clamped_to_maxzdim_minus_1` and
  `set_rect_clamps_to_world` are *inverted* into carve-through pins.
  Gate: formats suite + property round-trips (expand→compile→expand).
- **CT.2 — placeholder retirement.** `Vxl::empty`/`seeded_air` emit
  sentinel columns; S1.X placeholder tests replaced; grep-audit every
  `bedrock`/`placeholder` reliance (collide's `bedrock_blocks` stays —
  it is *policy* for real floors, now no longer masking phantom ones).
  Gate: full workspace suite; cave/scene demos boot via ROXLAP_CAPTURE.
- **CT.3 — mip downsampler.** `build_mip_level` (`vxl.rs:891-1170`):
  phase 1 skips empty children, phase 2's bedrock-terminator /
  `curz >= MAXZDIM` invariants reworked; 4 empty children → empty dest
  column. HIGHEST-RISK substage — raw pointer-style slab walk. Gate:
  PF.12 `remip_bbox == generate_mips` byte-equivalence extended with
  empty columns; edit fuzz corpus (QE) grown with carve-through cases.
- **CT.4 — core walkers.** `grid_view.rs` (`column_slab_mip`,
  `voxel_run_top_mip`, `for_each_run_mip`, `surface_color_mip`),
  `world_query::getcube` (empty → air, incl. the `UnexposedSolid`
  bottom branch), `SolidSampler`, lighting (`EstNormCache`,
  `shade_column` — already graceful, pin it). `BrickCache` rides
  `for_each_run_mip` for free. Gate: core suite + the CPU probe's
  bedrock half turns green.
- **CT.5 — scene agreement.** `vxl_voxel_solid` (`chunks.rs:101-126`),
  `Scene::raycast`(+`_clipped`), `islands::chunk_column_runs` (empty →
  no anchored run; NEW DT test "carve floor out → island detaches"),
  collide reconciliation, fow/audio ride `chunk_voxel_solid` — verify
  with their suites. Gate: scene+audio suites; collide slab-trap tests.
- **CT.6 — CPU hit/colour decouple + carve retexture.** Decisions 3+4:
  `compilerle` default inherit-colour, `Sampler::hit` solidity verdict
  with colour fallback, RGB-0 policy aligned with GPU
  (`decompress.rs:380-395`), CPU/GPU parity gate at a carve site.
  Independent of CT.0-5 — may land first as a standalone fix. Gate:
  the CPU probe's fall-through half (540 → 444) turns green.
- **CT.7 — GPU verification.** `decompress_column` already treats a
  surfaceless column as air — pin with a unit test; verify
  `refresh_chunk_partial` re-derive, `vox_z_lo/hi` content box, and
  chunk-occupancy bit when a chunk empties entirely. Un-`#[ignore]`
  `digger_repro.rs` — all four tests green. Gate: full headless GPU
  suite (cutaway, fow, emissive, digger).
- **CT.8 — snapshot wire + CLI.** Wire version bump (frozen shadow
  shape for the old version, fixtures for v1..current stay green);
  `compact_serialize_chunk` round-trips sentinels; CLI extract paths
  (vxl/kv6/vox) sanity-checked on a carved-through chunk. Gate:
  snapshot suite + fixture matrix.
- **CT.9 — validation + docs.** monada digger via path-dep: drill
  through sim z=0 both directions, no ghosts, deck-clip interplay
  visually checked (user pass). Book: editing chapter gains the
  empty-column encoding + the digger case study; CHANGELOG names the
  behaviour changes (carve-through, retexture default, islands
  un-anchoring, snapshot bump). Handover doc closed out.

## Hazards

- **H1 — downsampler OOB / mip corruption** (CT.3): the slab walk is
  raw index arithmetic; empty children violate both phase invariants.
  Mitigation: byte-equivalence + fuzz before any renderer work builds
  on it.
- **H2 — islands cascade**: maps that relied on the uncarvable floor
  as implicit support will now shed debris when players dig it out.
  Intentional; flagged in CHANGELOG, DT test pins it.
- **H3 — placeholder reliance**: opticast-era assumptions
  (S1.X below-bedrock camera, OOB-XY streak seeding) are gone with
  opticast, but CT.2 runs a grep-audit before flipping `Vxl::empty`.
- **H4 — snapshot forward-compat**: ≤ 0.30 readers parse sentinel
  columns fine but render a z=255 floor. Wire bump + release note;
  no silent corruption.
- **H5 — RGB-0 divergence**: GPU treats in-range RGB-0 as air, CPU as
  pass-through; CT.6 must land ONE policy on both, or the parity gate
  will (correctly) fail.
- **H6 — perf**: sentinel check is one compare at walk seed; brick
  and GPU builds unchanged. Watch the CT.3 downsampler inner loop —
  no per-voxel branch, only per-column.
- **H7 — external `.vxl` files**: no known writer emits the sentinel;
  if one does, "empty column" is the sane reading. Documented, not
  guarded.

## Definition of done

All four `digger_repro`/CPU-probe tests un-`#[ignore]`d and green on
both backends; formats fuzz clean; workspace suite green; monada
digger drills through the surface with no ghost layer (user visual
pass); book + CHANGELOG updated. Next cut minor (0.31.0).
