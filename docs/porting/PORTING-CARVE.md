# roxlap — carve-through-floor (Stage CT)

Entry doc written 2026-07-25 at workspace 0.30.1.
This is the **entry doc** for the carve-through-floor stage — tag **CT**.

## Status

- **CT.0 — LANDED 2026-07-26.** `EMPTY_COLUMN_SLAB` +
  `slab_is_empty_column` + `Vxl::column_is_empty` + module-doc section
  in `roxlap-formats/src/vxl.rs`; `slng`/`parse`/`serialize` needed no
  changes (their `n_floor = 0` handling already covers the sentinel).
  Green pins: sentinel shape (incl. "bedrock placeholder is NOT
  empty"), read-as-air, byte-equal serialize→parse round-trip. Target
  tests `carve_full_height_empties_column` (CT.1) and
  `vxl_empty_columns_are_sentinel_empty` (CT.2) are in-tree KNOWN-RED
  `#[ignore]`, verified failing for the right reasons. formats suite
  217 green / clippy / doc clean.
- **CT.1 — LANDED 2026-07-27.** Edit ops carve through the floor:
  - `delslab` de-clamped: `y1 == MAXZDIM` carves through the bottom —
    the consumed run becomes the PURE TERMINATOR `[MAXZDIM, MAXZDIM]`
    (zero-length run, the spans image of the sentinel), a truncated
    survivor gets the terminator appended (growth still ≤ 1 pair).
  - **Encoding correction vs the CT.0 plan**: voxlap's `compilerle`
    legitimately emits a terminal slab with an EXACTLY-empty floor
    list (`z1c = z1 − 1`) for a buried column bottom (the `dacnt`
    sub-slab + `p_z == MAXZDIM−1` exit) — walkers merge it into the
    previous run as solid-to-bedrock. The air sentinel is therefore
    the OVER-empty form **`z1c ≤ z1 − 2`**:
    `EMPTY_COLUMN_SLAB = [0, 255, 253, 0]`, and the same shape as a
    chain tail (`[0, 255, 253, z0]`) means "air below the last run".
    Discovered by `scum2_batch_edits_multiple_columns_same_row` going
    red on legitimate buried-bottom re-encodes.
  - `expandrle` decodes both sentinel forms to the pure terminator;
    `compilerle` emits them (empty-column guard + air-terminal at the
    rlendit2 boundary); `insslab` works against the pure terminator
    unchanged (verified by trace + tests).
  - Placeholder pins for CT.2: `Vxl::seeded_air` AND roxlap-scene's
    own `empty_chunk_vxl` (chunks.rs — found carving `z1 = 255` and
    silently seeding sentinel chunks) carve `[0, 255)` with
    TODO(CT.2) notes, keeping the bedrock placeholder until the
    walker/test sweep.
  - Tests: old clamp pins inverted (`delslab_y1_reaches_world_bottom`,
    `set_rect_carves_through_world_bottom`); new: insslab-into-empty,
    expandrle sentinel, full-height carve → sentinel, bottom carve
    with survivor (bytes + round-trip + re-insert), buried-bottom NOT
    sentinel shape pins, `generate_mips` no-panic on sentinel columns
    (the GPU upload path re-mips carved chunks every refresh — mip
    OUTPUT correctness stays CT.3).
  - **The GPU half of the digger bug is already fixed**: the sentinel
    decompresses to air (`have_surface` gate), so all four
    `digger_repro.rs` tests are GREEN and un-`#[ignore]`d — now
    regression gates. The CPU probe stays KNOWN-RED for its CT.6 half
    only (hit/colour coupling; the bedrock-layer half is gone).
  - Suites: formats 222 / core 143 / scene 304 / gpu headless / render
    / audio all green; clippy + doc clean.
- **CT.6 — LANDED 2026-07-27** (out of order, as planned — it was
  independent). CPU hit/colour decouple + carve retexture:
  - `VoxColor::BEDROCK_FALLBACK` (`0x8040_4040`, roxlap-formats) —
    the shared last-resort colour for solid-but-recordless cells.
  - Carve retexture: `compilerle`'s colfunc branch inherits the
    nearest non-zero original record when the colfunc returns `0`
    ("engine picks" — the plain-carve default); a column with no
    usable record (bedrock placeholder) keeps the `0` sentinel.
    `nearest_original_color` walks the colour table (floor+ceiling).
  - `GridView::surface_color_mip` reworked: hit verdict from
    `voxel_run_top_mip`, colour ladder own-record → run-top →
    `BEDROCK_FALLBACK`; an EXPLICIT zero-RGB record still reads as
    air (GPU parity). New `voxel_color_raw_mip` (records incl. zero;
    `None` = no record) underneath.
  - **Pulled in CT.4's grid-view half**: `voxel_run_top_mip` /
    `for_each_run_mip` learned the air-terminal (`z1c + 1 < z1`
    terminal → no bottom run) — without it the decoupled hit verdict
    HIT the phantom `[255,256)` run the walkers still emitted for
    sentinel columns (pre-CT.6 the colour coupling coincidentally hid
    it). CT.4's remainder: `getcube`, `SolidSampler`, lighting,
    scene's `vxl_voxel_solid` / islands / collide (CT.5).
  - GPU mirror: `expand_solid_runs` learned both sentinel forms (the
    chain-tail case previously left a phantom solid at z=255 — caught
    by the new parity gate at exactly one voxel).
  - NEW gate `roxlap-render/tests/carve_parity.rs`: per-voxel CPU
    verdict (`surface_color_mip`) vs GPU `solid_occupancy` +
    textured-colour parity over fill/pocket/bottom-carve/empty/
    placeholder shapes. Pure CPU (no device) — runs everywhere.
  - The CPU digger probe is GREEN and un-`#[ignore]`d (depth 444) —
    both halves of the digger bug are now fixed on both backends.
  - Test fallout, both semantic inversions: scene's
    `set_sphere_with_colfunc_paints_exposed_interior` contrast half
    (plain carve now inherits, not black); gpu `partial_refresh`
    count-change case rebuilt around a full-column carve (a
    same-height carve is count-STABLE under inherit).
  - **Observation (pre-existing footgun, not CT)**: `remip_bbox`
    compacts the vbuf EXACT-FIT and drops all edit headroom — the
    partial-refresh test tops the pool back up after remip; flag for
    a QE follow-up (reserve or document).
  - Suites: formats 222 / core 143 / scene 305 / gpu (all 6 targets)
    / render (incl. new parity) / audio green; clippy 0 / doc clean.
- **CT.2 — LANDED 2026-07-27** (placeholder retirement), pulling in
  most of CT.3 and the rest of CT.4:
  - Both seeds flipped: `Vxl::seeded_air` and scene's
    `empty_chunk_vxl` carve the full `[0, 256)` — fresh chunks/worlds
    are empty-sentinel columns, truly all-air. `from_dense` worlds no
    longer grow a phantom ground voxel.
  - **CT.4 rest**: `world_query::getcube` learned the air-terminal
    (the sentinel read as `UnexposedSolid` at z∈{254,255} — caught by
    the debris test landing on a phantom that had MOVED to z=254,
    past collide's z==255 policy special-case). `SolidSampler` is
    just a chunk cache over the already-taught `vxl_voxel_solid` —
    nothing to do. Scene's `vxl_voxel_solid` taught here too (the
    inverted S1.X tests needed honest answers).
  - **CT.3's empty-child half**: `build_mip_level` handles sentinel
    sources (seed-exhaust + flatten skip), air-terminal chain tails
    (state-2 advance guard kills the phantom on-event; the off at z0
    still fires), per-source `anchored` flags, and a reworked tail:
    all-empty dest → sentinel, un-anchored dest → air-terminal (same
    ceiling list, different header). Plus the subtle one: **voxlap's
    placeholder events were load-bearing** — they pulled the last
    pending records out of the catch-up loops; a bedrock-anchored
    cell now runs one synthetic bottom-flush pass at exhaustion
    (air-tailed cells must NOT get it — they flush at their
    off-events, and padding them would re-grow phantoms).
    Pinned by `generate_mips_empty_and_mixed_cells` (formats) + the
    GPU `solid_mips_are_child_supersets` cross-level gate + PF.12
    `remip_bbox == generate_mips` byte-equivalence still green.
    Remaining for CT.3 proper: fuzz-corpus growth over CT shapes.
  - Inverted pins: S1.X `empty_chunk_*` (air everywhere incl. z=255,
    columns are sentinels), cavegen implicit-air chunk, collide
    `no_phantom_bedrock_plane_under_either_policy` (the
    `bedrock_blocks` opt-in plane is gone — the flag now governs only
    genuine z=255 content), audio
    `empty_chunk_bottom_plane_is_acoustically_air` (+ the
    `grid_thickness` caveat doc), islands dense-oracle seeds only
    from genuinely solid z=255, debris free-fall comment.
  - Suites: formats 224 / core 143 / scene 305 / gpu (all, incl.
    cutout + superset) / render / audio green; clippy 0 / doc clean.
- **CT.5 — LANDED 2026-07-27.** Islands honest un-anchoring:
  `chunk_column_runs` yields no runs for the empty sentinel and closes
  an air-terminal column's last run UN-anchored; the module doc's
  support rationale rewritten (anchoredness is a fact of the bytes,
  not the old "z=255 is uncarvable" invariant). NEW DT gate
  `carve_through_floor_detaches_island`: dig a bottom-standing
  pillar's base out through the floor → the hanging top comes back as
  an island (exact voxel count + bbox pinned). Multi-chz mid-stack
  anchoring nuances deliberately untouched (pre-existing). Suites
  green (scene 306), clippy clean.
- **CT.7 + CT.3-fuzz — LANDED 2026-07-28.**
  - CT.7: the leftovers were already handled by existing code —
    `refresh_chunk` clears the chunk-occupancy bit
    (`!colors.is_empty()`) and resets the slot z-extent
    (`solid_z_extent → None`) — they only lacked a pin. NEW gate
    `carve_all_and_refresh_renders_sky` (digger_repro.rs): fill a
    chunk, render, carve EVERYTHING, refresh in place, re-render →
    every pixel sky.
  - CT.3-fuzz: NEW `edit` fuzz target — an op-stream driver
    (insert/carve rects, sphere carves, bottom-reaching carves) living
    in the library (`roxlap_formats::fuzz_driver`) so the committed
    seeds also run as a stable-CI unit test
    (`edit_fuzz_seeds_hold_invariants`); invariants: byte-stable
    round-trip, sane per-column runs (incl. sentinel/pure-terminator),
    mip ladder builds + walks. 4 corpus seeds committed; CI's
    smoke-fuzz picks the target up automatically (`cargo fuzz list`).
  - **The fuzzer earned its keep in under a minute**: `nextptr` is a
    u8 dword count, and the CT.1 air-terminal made the closing slab
    NON-terminal — so a 255+-record fully-exposed stretch (an insert
    into an empty-sentinel column with exposed sides) overflowed it;
    pre-CT that slab was always the uncapped terminal. Fix:
    `compilerle` splits an oversized slab into degenerate-continuation
    slabs (`z0 == z1` — every walker already folds them back; the
    ceiling arithmetic stays exact for a contiguous record list), and
    `MAXCSIZ` grew 1028 → 1088 for the extra headers. Pinned by
    `oversized_exposed_run_splits_slabs` + the crash input promoted to
    the corpus (`seed-oversized-slab-split`). 3-minute soak: 77k runs
    clean.
- Remaining: CT.8 snapshot wire bump, CT.9 monada validation + book +
  CHANGELOG + handover close-out.

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
