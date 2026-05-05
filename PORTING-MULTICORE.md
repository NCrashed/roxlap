# roxlap — Multicore CPU rendering (Substage R12)

Sub-substage roadmap and locked decisions for parallelising the
roxlap renderer with [`rayon`](https://crates.io/crates/rayon).
Companion to [PORTING-RUST.md](PORTING-RUST.md) — the broader
substage roadmap that places R12 between R11 polish and R9 NEON.

## Goal

Multicore CPU rendering on x86_64. Target on a balanced 8-core
consumer CPU: **~3–5× throughput** vs the current single-threaded
baseline on the 12-pose oracle render suite at 640×480.

Refreshed single-threaded baseline (2026-05-05, captured by
`cargo run --release -p roxlap-oracle -- bench --iters 50` on the
dev machine):

| pose | mean ms |
|---|---|
| north | 7.55 |
| east | 7.55 |
| diag_down | 10.95 |
| high_down | 13.16 |
| sprite_front | 7.02 |
| sprite_above | 8.38 |
| sprite_iso | 10.21 |
| sprite_coco | 9.45 |
| diag_down_lit | 11.24 |
| tile_1x | 11.59 |
| tile_half | 11.68 |
| tile_blend | 11.33 |
| **mean across 12 poses** | **10.01 ms (99.9 fps)** |

R12 lands before R9 NEON / R10 wasm because the cross-arch ports
inherit whatever rendering *structure* exists when they ship; doing
the SMP restructure once (here) is cheaper than three times (one
per arch).

## Locked decisions

| # | Decision | Consequence |
|---|---|---|
| 1 | **`rayon` for the threading layer.** Rayon's work-stealing pool, `par_iter`, and `join` cover every parallelism axis we need (per-quadrant, per-tile, per-column lighting bake). | One workspace dependency. `RAYON_NUM_THREADS=1` is the kill-switch / single-threaded repro. No hand-rolled `std::thread` pool. |
| 2 | **`ScratchPool` is a host-owned sibling type, not a rasterizer-internal pool.** `ScratchPool::new(xres, yres, vsid)` for single-threaded; `::new_parallel(.., n_threads)` for N slots. Opticast takes `&mut ScratchPool` (slot 0 in R12.1; slots 0..3 in R12.2; per-tile in R12.3). | The "rasterizer owns the pool" form was the original intent, but Rust borrow-rules make it awkward — `ScalarRasterizer` is generic and can't field-project a per-thread `ScanScratch` out of `&mut self` while simultaneously calling free functions like `top_quadrant` that take `(&mut R, &mut ScanScratch, &ctx)`. Host-owned pool side-steps this and keeps allocations long-lived (one ~7.6 MB allocation per slot survives across frames; the rasterizer remains the per-frame object). Per-frame setters (`pool.set_skycast` / `set_fog` / `set_side_shades`) broadcast to all slots so each thread sees current frame state on its private slot. |
| 3 | **CI stays single-threaded; multicore is opt-in.** The 12 voxlap-C oracle goldens are byte-stable bit-exactness gates, not perf gates — they run with `n_threads = 1` regardless of host threading. | Bit-exact regression detection survives R12 untouched. A separate per-thread oracle run (added in R12.5) asserts hash equality across `--threads {1, 2, 4, 8}`. |
| 4 | **Per-quadrant first (R12.2), per-tile second (R12.3).** Per-quadrant's seam already exists in `scan_loops.rs`; landing it first proves the per-thread `ScanScratch` plumbing before tackling the larger tile restructure. | Two perf milestones (~2–3× then ~3–5×) instead of one big-bang. Each ships as its own commit with its own bench numbers. |
| 5 | **`split_at_mut` where possible, raw pointers only where required.** Per-tile row strips are contiguous and split safely; per-quadrant wedges are pixel-disjoint but row-overlapping, so quadrants either use unsafe disjoint pointers with a wedge invariant or per-thread render-then-merge. R12.2 picks one based on Rust borrow-checker friction. | Maximises safe-Rust coverage. Any unsafe is localised to one or two functions with a documented invariant. |
| 6 | **Pixel-write determinism per thread.** Each thread writes only to its own region (its quadrant wedge or its tile row range). `gscanptr`, `radar[]`, `uurend[]`, `cf[]` are all in per-thread `ScanScratch`. No cross-thread atomics on the hot path. | Output is deterministic regardless of `rayon`'s work-stealing schedule. Same input → same bytes. |
| 7 | **Default thread count = `rayon`'s global pool default.** That's `num_cpus::get()` unless overridden by `RAYON_NUM_THREADS`. Hosts that want explicit control pass `n_threads` to `new_parallel`. | One less knob to document. Standard rayon ergonomics. |

## Sub-substage roadmap

| # | Scope | Estimate | Validation |
|---|---|---|---|
| **R12.0** | Author `PORTING-MULTICORE.md` (this doc) + add `rayon` workspace dep. No code wired yet. | 0.5 d | Doc lands; `cargo build` green; `cargo test` green. |
| **R12.1** | Introduce `ScratchPool` (host-owned `Vec<ScanScratch>`). Opticast signature changes from `&mut ScanScratch` to `&mut ScratchPool`; R12.1 indexes slot 0 only. `ScratchPool::new` (1 slot, single-threaded) and `::new_parallel(.., n_threads)` (N slots). Per-frame setters broadcast to every slot. | 1–2 d | All existing tests pass; oracle goldens byte-stable at the default thread count (still 1, just now plumbed through the pool). |
| **R12.2** | Per-quadrant 4-way via `rayon::join`. Each quadrant runs against its own pool slot. Framebuffer/zbuffer split via raw pointers + a wedge-disjoint invariant *or* per-thread render-then-merge — pick whichever is cleaner under the borrow checker. | 3–5 d | Oracle goldens byte-stable across `--threads 1` and `--threads 4`; bench shows ≥2× on at least 6/12 poses. |
| **R12.3** | Per-tile N-way via `rayon::par_iter` over `Vec<RowStrip>`. Each strip has a restricted frustum + its own `ScanScratch` slot sized by the strip's pixel range. Strips ARE contiguous in row-major, so `framebuffer.split_at_mut` works (no unsafe). | 5–7 d | `bench --threads N` scaling curve for N ∈ {1, 2, 4, 8}; oracle goldens byte-stable at every N. |
| **R12.4** | `update_lighting` per-column outer loop → `par_iter`. Sprites (`Engine::sprites`) → `par_iter` with z-test arbitrating final pixel writes. | 1–2 d | World-bake bench number; sprite oracle goldens (5 poses) byte-stable at every thread count. |
| **R12.5** | `roxlap-oracle bench --threads N` flag + scaling docs in `README.md` (or a new `BENCH.md`). Per-thread oracle (`oracle diff --threads N`) added to CI as a non-blocking job. | 0.5 d | Bench output reproducible across thread counts; CI green. |

**Total scope**: ~2–3 weeks for R12.1–R12.4. R12.0 + R12.5 are
bookends.

## Where the parallelism lives

Three viable axes, increasing in payoff and in implementation
cost.

### Per-quadrant — 4-way, simpler (R12.2)

`opticast` already structures into top / right / bottom / left
quadrant functions in `crates/roxlap-core/src/scan_loops.rs`. The
four quadrants ray-cast over four geometrically-disjoint screen
wedges (rays partitioned by which axis dominates the gline
direction).

- **Pro**: lowest-risk milestone. The seam already exists;
  quadrants are pure functions of `&mut ScanScratch` + `&mut
  Rasterizer`. Validates the per-thread scratch plumbing before
  R12.3.
- **Con**: ~3× speedup ceiling — load imbalance is real (looking
  down a corridor puts 60–70 % of the work in one quadrant). Caps
  at 4 threads.

### Per-tile / row-strip — N-way, the real prize (R12.3)

Split the framebuffer into N contiguous row strips (e.g., 30
strips of 16 rows for a 480-row screen). Each strip runs its own
`derive_prelude` (frustum corners restricted to its y-range) +
`opticast` over that strip's pixel range.

- **Pro**: scales near-linearly to core count (8 cores → ~6×).
  Tile boundaries align to cache lines (64 B = 16 px in u32). Maps
  cleanly to `rayon::par_iter` over a `Vec<RowStrip>`.
- **Con**: requires re-deriving the frustum per strip (small
  per-frame cost, but a real port). Each strip's `ScanScratch` is
  sized by the strip's pixel range, not the full screen — so
  per-strip memory is *less* than per-quadrant, but total memory
  scales with strip count.

### Sprite + lighting bake — free correctness wins (R12.4)

- Sprites with z-test render mutually independently.
  `rayon::par_iter` over `&[Sprite]` with per-sprite framebuffer
  views; the z-test arbitrates pixel write order so output is
  deterministic. Limited gain (the host has ~2 sprites) but free.
- `update_lighting` is embarrassingly parallel per column.
  Already an `for x in 0..vsid { for y in 0..vsid { ... } }`
  shape. Convert the outer loop to `par_iter`.

## API change shape (chosen: host-owned `ScratchPool`)

Three designs evaluated; (C) is what landed in R12.1.

### (A) Caller owns a raw `Vec<ScanScratch>`

```rust
let mut scratches: Vec<ScanScratch> = (0..n_threads)
    .map(|_| ScanScratch::new_for_size(xres, yres, vsid))
    .collect();
opticast_parallel(&mut rasterizer, &mut scratches, ...);
```

- Pro: no API surprises; caller controls allocation.
- Con: every host crate has to know about the pool's broadcast
  ergonomics (per-frame `set_fog` etc. on each entry).

### (B) Rasterizer owns the pool

```rust
let mut rasterizer = ScalarRasterizer::new_parallel(
    fb, zb, pitch, ..., n_threads,
);
opticast(&mut rasterizer, ...);  // internally par_iter when n_threads > 1
```

- Pro: clean public API; `Engine::render(...)` stays one call.
- **Con: doesn't compile cleanly.** `ScalarRasterizer` is generic
  and the scan-loop dispatchers (`top_quadrant`, etc.) take
  `(&mut R, &mut ScanScratch, &ctx)`. Field-projecting a per-thread
  `ScanScratch` out of `&mut rasterizer` while *also* passing
  `&mut rasterizer` to those free functions is a re-borrow conflict
  the compiler rejects (no split-borrow accessor exists for a
  generic type). Escapes are all costly: unsafe pointer
  acrobatics, interior mutability with runtime overhead, or
  reshaping the trait surface to drop the scratch parameter (a
  larger restructure). Worse, the rasterizer is reconstructed each
  frame because it borrows `&mut framebuffer` / `&mut zbuffer`,
  which makes per-frame allocation of the pool wasteful (the pool
  is ~7.6 MB per slot).

### (C) Host-owned `ScratchPool` *(chosen, R12.1)*

```rust
// long-lived, on the App / oracle struct
let mut pool = ScratchPool::new_parallel(xres, yres, vsid, n_threads);

// per frame
let mut rasterizer = ScalarRasterizer::new(fb, zb, pitch, ..., vsid);
opticast(&mut rasterizer, &mut pool, &cam, &settings, ...);
```

- Pro: compiles cleanly with no unsafe; pool's lifetime matches
  the host (long-lived); rasterizer stays the per-frame object;
  `n_threads` knob lives on the data structure that actually owns
  the threads' working memory.
- Pro: per-frame setters (`pool.set_skycast` / `set_fog` /
  `set_side_shades`) broadcast to all slots, so once R12.2 fans
  out, each thread already sees the current frame's state on its
  private slot.
- Con: pool is one extra type for hosts to know about; total
  public-API delta is one `ScanScratch` field renamed to a
  `ScratchPool` field.

`roxlap-oracle`'s bench / test fixtures use `ScratchPool::new` (1
slot) so byte-stability gates run identically to single-threaded.
Multicore is opt-in via `::new_parallel`.

## Memory cost

Per-thread `ScanScratch` at 640×480 / vsid = 2048:

| field | size | per thread |
|---|---|---|
| `radar` (xres × 6 × 256 × CastDat) | 640 × 6 × 256 × 8 B | 7.5 MB |
| `angstart` (xres × 4) | 10 KB | 10 KB |
| `lastx` (max(yres, vsid) × i32) | 8 KB | 8 KB |
| `uurend` (2 × xres × i32) | 5 KB | 5 KB |
| `cf` (CF_LEN × CfType) | 8 KB | 8 KB |
| **Total** | | **~7.6 MB** |

- 4 threads at 640×480: ~30 MB. Fine.
- 8 threads at 1920×1080 (3× xres): ~70 MB per thread × 8 ≈
  560 MB. Heavy on a 16 GB machine; trivial on a 32 GB workstation.
- **Tile-based mitigation** (R12.3): per-strip `ScanScratch`
  sized by strip's pixel range. 8 strips × (xres / 8) wide each
  → per-strip `radar` is 1/8 the per-thread version. Net per-tile
  memory ≈ per-thread / N.

## Risks

### Borrow-checker fights with framebuffer / zbuffer slicing

The 4 quadrant wedges are *pixel-disjoint* but *row-overlapping*
in row-major storage. `split_at_mut` requires contiguous slices;
the wedges aren't. Two mitigations:

- **Unsafe disjoint pointers**: each thread holds `*mut u32` to
  the same framebuffer + a runtime invariant that says "I only
  write to pixels in my wedge". Sound, but loses the borrow-checker
  safety net. Encapsulated in one struct with documented
  invariants.
- **Per-thread render-then-merge**: each thread renders to its
  own framebuffer, then a final z-tested pass composites. Adds
  4× framebuffer + 4× zbuffer (5 MB extra at 640×480, 32 MB at
  1080 p). Safe. Costs one frame of memory bandwidth on merge.

For tile-based (R12.3), `split_at_mut` works — strips ARE
contiguous in row-major. Use this and skip the unsafe.

### Determinism across thread counts

`rayon::par_iter` work-stealing means iteration order varies
between runs. The renderer must produce identical pixels
regardless of thread schedule. Each tile / quadrant writes to its
own region, so final pixel values don't depend on order — and
`gscanptr`, `radar[]`, `uurend[]`, `cf[]` are all per-thread by
virtue of per-thread `ScanScratch`. Pixel writes are partitioned.
Should hold.

**Validation**: per-thread oracle (`roxlap-oracle diff
--threads N`) for N ∈ {1, 2, 4, 8} asserts hash equality. Land
in R12.5 alongside the bench `--threads` flag.

### Cache thrashing

8 threads writing to interleaved cache lines = false-sharing
disaster. Mitigations:

- Tile boundaries align to cache lines: 64 B = 16 px (u32) =
  pad strip height to multiples of `64 / (4 * xres)` — for
  640×480, 16 rows fits one cache-line stride. Should be naturally
  clean.
- `ScanScratch` fields are `Vec`, so they're heap-allocated and
  don't share cache lines across threads.

### Debug ergonomics

Parallel bugs are non-deterministic. Mitigations:

- `--threads 1` (or `RAYON_NUM_THREADS=1`) flag for repro.
- Oracle goldens stay byte-stable as the regression gate at every
  thread count.
- Bench prints which thread count it ran at so flaky runs can
  always be re-checked single-threaded.

## Bench projection

| variant | ms / frame | fps | speedup |
|---|---|---|---|
| **single-threaded (today)** | 10.0 | 100 | 1× |
| R12.2 per-quadrant 4-way | ~4–5 | 200–250 | 2–3× |
| R12.3 per-tile 8-way | ~2–3 | 350–500 | 3–5× |
| R12.3 per-tile 16-way (server) | ~1.5–2 | 500–700 | 5–7× |

Projections based on voxlap's algorithm structure + typical rayon
overhead. Actual numbers will land within ±30 % of these. The
bench harness from CD.x (`roxlap-oracle bench`) captures
min / p50 / mean / p99 / max per pose — same harness drives R12
validation.

## Reading list (for the implementing session)

1. `crates/roxlap-core/src/opticast.rs` — driver loop; the entry
   point that becomes `par_iter`-shaped in R12.3.
2. `crates/roxlap-core/src/scan_loops.rs` — 4 quadrant functions
   (`top_quadrant` / `right_quadrant` / `bottom_quadrant` /
   `left_quadrant`); R12.2's seam.
3. `crates/roxlap-core/src/rasterizer.rs::ScanScratch` — current
   ownership; new pool design lives here.
4. `crates/roxlap-core/src/scalar_rasterizer.rs::ScalarRasterizer`
   — current `&mut framebuffer` / `&mut zbuffer` borrow shape;
   needs the `new_parallel` constructor.
5. `crates/roxlap-core/src/world_lighting.rs::update_lighting` —
   embarrassingly parallel per-column outer loop.
6. `crates/roxlap-oracle/src/main.rs::cmd_bench` — driver for
   scaling measurements; add `--threads N` here in R12.5.
7. rayon docs: `par_iter`, `join`, `ThreadPool`,
   `current_thread_index`. Standard pattern: "render N
   independent tiles" maps to `par_iter` + per-tile state via a
   pool slot keyed on `current_thread_index()`.

## Out of scope

- GPU offload. roxlap stays a software rasterizer; that's a
  defining decision in PORTING-RUST.md.
- Cross-process / cross-machine parallelism. Rayon is intra-process
  only.
- Async / `tokio`. The renderer is CPU-bound; async adds latency,
  not throughput.
- Custom thread affinity / NUMA pinning. Rayon's defaults are
  good enough for the consumer-CPU target. Server NUMA tuning is
  a downstream concern.

## How to apply

R12.0 (this doc + `rayon` dep) is the cheapest first commit —
unblocks everything else without committing to a specific
parallelism axis. Land that, then R12.1 (`ScanScratch` pool) is
the load-bearing structural change every subsequent sub-substage
depends on.

If R12 is deferred entirely, R9 NEON is the fallback. The cost
of skipping R12 is +30–50 % effort on R9 later (re-restructure
for SMP after the NEON port has already replicated the
single-threaded structure).
