# roxlap — CPU renderer rewrite: 3D-DDA + brickmap (Substage DDA)

Sub-substage roadmap and locked decisions for **replacing voxlap's
column-coherent opticast CPU renderer with a per-pixel 3D-DDA +
brickmap renderer**. Companion to [PORTING-RUST.md](PORTING-RUST.md)
(the original R1..R12 voxlap port), [PORTING-SCENE.md](PORTING-SCENE.md)
(the scene-graph layer) and the VC virtual-column work (memory
`project_vc_scope`).

This document is the **start-of-stage brief**. A fresh-context session
should read it top to bottom before touching code.

## Why

The voxlap algorithm is not an honest 3D raycast — it is **2.5D
column-coherent**: for each screen column it DDA-steps along the
ground plane and fills the vertical via RLE column spans with `cf`
boundary interpolation. All of roxlap's long-standing artifact
classes are direct consequences of that structure, not of the port:

- per-ray quantization **notch** on silhouettes (`tiny_grid_1x1x1`,
  accepted as voxlap-inherent in `project_single_voxel_silhouette_rootcause`)
- floor **hairlines** (`gdz` precision)
- axis-aligned **green beams** at deep mip-N (`cross_sign`
  cancellation, `project_axis_aligned_mip_beams`)
- sub-pixel **`_mm_rcp_ps` divergence** across engines
- the entire **cross-chunk look-down** complexity (S4B.6.j → the
  whole VC.0..VC.7 virtual-column rewrite)

These do not close while we stay column-coherent — we only mitigate
them with mip params. They are especially visible in the **voxel
editor**. A per-pixel 3D-DDA makes every ray independent → no
column/row coherence shortcuts → none of the stitching seams. As a
bonus it (a) parallelises into true tiles (bit-identical across
thread counts, unlike voxlap's `--threads 1`-frozen goldens) and
(b) is **clean-room** w.r.t. Ken Silverman's voxlap, which unblocks
a single permissive license (see DDA.10).

## Goal & invariants

Replace `opticast` with a per-pixel **3D-DDA (Amanatides–Woo) +
brickmap** CPU backend under three hard invariants:

1. **Correctness** — structurally kill the artifact classes above.
2. **Speed** — ≥ voxlap via empty-space skipping (bricks) + SIMD ray
   packets + true tile parallelism.
3. **Clean-room** — new code is not derivative of voxlap source. This
   is an architecture-level constraint, not a final polish step.

End state: voxlap renderer (`grouscan` + `opticast` + `scan_loops`
+ the VC virtual-column machinery) is **deleted**; DDA is the only
CPU backend; the repo ships under a single permissive license.

## Cut seam (locked)

Replace at the `opticast(rasterizer, pool, camera, settings,
GridView)` boundary (`crates/roxlap-core/src/opticast.rs:180`). Scene
already produces a `GridView`/`ChunkGrid` spatial index
(`grid_view.rs:52/118`) and hands `(&mut [u32] color, &mut [f32] z)`.
The new backend exposes a sibling `render_dda(...)` with the same
buffer conventions (`0x80RRGGBB`; z = perpendicular distance,
smaller = closer) and the same `compose_into` (`render.rs:229`).
Voxel data is read via `GridView::chunk_at_xyz` (`grid_view.rs:314`).

## Reuse vs delete

| Layer | Fate |
|---|---|
| `GridView` / `ChunkGrid` / `chunk_at_xyz` | **Reuse** — the spatial index for the DDA outer loop |
| `RasterTarget` (fb/zb raw-ptr disjoint writes, `scalar_rasterizer.rs:76`) | **Reuse** as the pixel sink |
| `compose_into`, `render_scene_composed` (`render.rs:229/383`) | **Reuse** — only change who they call |
| `grouscan.rs` (4939 lines, entire `phase_*` machine) | **Delete** (DDA.9) |
| `opticast.rs` / `scan_loops.rs` / `projection.rs` / `ray_step.rs` / `column_walk.rs` | **Delete** (DDA.9) |
| `column_z_base` virtual-column stack (all of VC.0..VC.7) | **Delete** — cross-chz becomes a plain step to the neighbour cell |
| `Vxl` storage | **Read** for now (source for brick cache); replace/clean in DDA.3/DDA.10 |

## Locked decisions

| # | Decision | Consequence |
|---|----------|-------------|
| 1 | **Cut at `opticast()`, add `render_dda()` sibling first** (not in-place rewrite). | voxlap stays runnable behind a flag through DDA.0..DDA.8; safe A/B. |
| 2 | **Brick storage = lazy derived cache from `Vxl`**, locally invalidated per edit. | Editor stays responsive; full native brick storage deferred to a later 0.x, not blocking this stage. |
| 3 | **Brick size 8³, occupancy = one `u64` bitmask per brick.** | Cheap empty-skip; maps onto SIMD; two-level DDA (bricks → voxels). |
| 4 | **Normals/shading = independent scheme** (central-difference gradient over brick occupancy), NOT a port of `estnorm`. | Both simpler and license-clean (see DDA.10). |
| 5 | **Tile parallelism, not strips.** | Per-pixel rays are independent → square tiles via rayon → **bit-identical across thread counts** (voxlap goldens are frozen at `--threads 1` due to per-strip discretization). |
| 6 | **Bit-exactness vs voxlap goldens is abandoned by design** (different sampling). | Re-freeze fresh DDA goldens; keep the oracle as an image-vs-image regression, not a bit gate. |
| 7 | **Keep `.vxl` on-disk compatibility.** | File formats aren't the licensing risk; *copied parsing code* is (DDA.10). Compatibility stays; the reader must be an independent implementation. |

## Architecture sketch

```rust
// roxlap-core/src/dda/

pub trait PixelSink {            // thin wrapper over RasterTarget
    fn put(&mut self, x: u32, y: u32, color: u32, dist: f32);
}

pub fn render_dda(
    camera: &Camera,
    settings: &OpticastSettings, // reuse: xres/yres, tile bounds, projection, mip, max_scan_dist
    grid: GridView,              // reuse: chunk_at_xyz spatial index
    sink: &mut impl PixelSink,
);

// Per pixel:
//   primary ray = camera origin + dir(pixel)
//   outer 3D-DDA over ChunkGrid cells          (chunk_at_xyz)
//     -> per chunk: 3D-DDA over 8^3 bricks      (occupancy u64 skip)
//        -> per occupied brick: dense 3D-DDA over voxels
//           -> first solid hit: color + central-diff normal + shade
//              write (color, perp-distance) to sink; stop ray
```

## Sub-substage roadmap

| Stage | Scope | Gate |
|---|---|---|
| **DDA.0** | `roxlap-core/src/dda/` scaffold; `PixelSink` over `RasterTarget`; `render_dda` clears to sky; runtime toggle `ROXLAP_DDA=1` in `render_scene` (`render.rs:185`). | Builds; flag toggles; voxlap path untouched & green. |
| **DDA.1** | Single-chunk dense 3D-DDA, no bricks, no shading; flat voxel color. | **`tiny_grid_1x1x1` silhouette correct** (no notch). Freeze 3–4 visual goldens. |
| **DDA.2** | Camera-in-solid / air-gap, near-far, `max_scan_dist`, sky + fog parity. | outside/inside-camera poses sane; fog parity. |
| **DDA.3** | Brickmap: 8³ occupancy grid per chunk + two-level empty-space skip; brick cache from `Vxl`, local per-edit invalidation. | Perf vs voxlap baseline; correctness unchanged from DDA.2. |
| **DDA.4** | Cross-chunk + cross-chz via `chunk_at_xyz` outer DDA. | **VC.0 pin + S4B.6.j look-down repros GREEN for free** (no virtual column). |
| **DDA.5** | Shading/lighting with clean-room normals; directional + `side_shades` + per-impact relight for the editor. | lit poses correct; editor relight works. |
| **DDA.6** | LOD/mip: brick mip pick by distance; Far-LOD billboards (S6) still work. | **`axis_aligned_beam_repro` GREEN**; perf at demo vsid=4096. |
| **DDA.7** | SIMD ray packets (4-wide f32x4 SSE/NEON/wasm) + tile rayon. | perf target met; **determinism across thread counts**. |
| **DDA.8** | KV6 sprites in the DDA path (raytrace KV6 voxels or depth-correct splat). | sprite poses render + depth-composite. |
| **DDA.9** | DDA = default; repoint oracle/goldens; **delete** grouscan / opticast / scan_loops / VC machinery. | voxlap code gone; all tests green on DDA; perf acceptable. |
| **DDA.10** | **License audit & cleanup** — identify and excise every voxlap-derived piece blocking a single permissive license. | No voxlap-derived code in published crates; per-crate verdict documented; license flipped. |

## DDA.10 — License audit & cleanup (the deliverable for "free license")

**Why this is its own stage.** roxlap is currently effectively
double-licensed: the *Rust code* is MIT/Apache, but the README
(lines 273–291) carries a standing caveat that the **algorithms and
on-disk formats are Ken Silverman's voxlap**, royalty-free for
non-commercial use only, with commercial use requiring a license
from Ken. That caveat — not the MIT/Apache text — is what stops a
user from shipping a commercial Steam game on roxlap.

Removing the voxlap *renderer* is necessary but **not sufficient**.
The whole derivative-of-voxlap tail must be gone from the published
crates before the caveat can be dropped.

> Disclaimer: this is an engineering plan, not legal advice. Get a
> lawyer's sign-off before changing the license text or making
> commercial-use claims.

**Principle.** A *file format* is generally not copyrightable;
*copied code/structure* is. The DDA algorithm (Amanatides–Woo) and
brickmap are independently published — not Ken's — so the renderer
itself clears. The risk lives in the surrounding ported code.

### Per-crate audit checklist (each item → verdict: clean / derivative / rewrite)

- [ ] **roxlap-core renderer** — deleted in DDA.9. Confirm no
      residual `grouscan`/`opticast`/`scan_loops`/VC code remains.
- [ ] **roxlap-core lighting** — `estnorm`/`updatereflects`/world
      lighting bake. Likely voxlap-derived → **rewrite clean-room**
      (DDA.5 already replaces normals; finish the lighting bake).
- [ ] **roxlap-formats `Vxl`** — slab/RLE parse, `voxel_color`,
      `generate_mips`, edit API. Audit whether it is a line-by-line
      port of voxlap C. Keep `.vxl` *compatibility*; ensure the
      reader/writer is an **independent implementation** (rewrite the
      parts that are direct ports).
- [ ] **roxlap-formats KV6 / KFA** — sprite + anim loaders. Same
      treatment: format compat OK, ported code not.
- [ ] **roxlap-cavegen** — confirm original (likely clean).
- [ ] **roxlap-oracle** — *literally* compares against voxlap C
      output. **Dev tool: do not publish.** Keep out of the
      distributed crate set (or under its own non-commercial terms);
      it then does not taint the distribution.
- [ ] **assets/oracle.vxl.gz, meltsphere/coco KV6, cave-refs/** —
      audit provenance of bundled data files; replace any
      voxlap-shipped sample assets with originals.
- [ ] **README/NOTICE/headers** — remove the Ken-Silverman
      commercial caveat once the above are clean; add an attribution
      note (format interop / inspiration) that does not impose
      use restrictions.

### Exit criteria

1. Every checklist item has a written verdict; all `derivative`
   items resolved to `rewrite`/`removed`.
2. No published crate contains voxlap-derived code; the oracle (the
   one unavoidable derivative) is dev-only / unpublished.
3. README license section reduced to plain MIT/Apache dual (or single
   chosen permissive license) with no commercial-use caveat.
4. CHANGELOG entry documents the relicensing and what changed.

## Risks

- **R1 — naive DDA is slower than voxlap.** Expected until
  DDA.3/DDA.7. Snapshot baseline like `project_s4b_baselines`; don't
  panic before bricks + packets land.
- **R2 — KV6 sprites** are a self-contained chunk; DDA.8 may be
  larger than it looks.
- **R3 — oracle bit-exactness gone by design.** Re-freeze; keep it as
  image regression.
- **R4 — clean-room "derivative by knowledge".** The author has read
  voxlap source. Mitigation: use only independently-published
  algorithms (DDA/brickmap), don't copy code or structure; DDA.5
  normals + DDA.10 rewrites exist partly for this reason.
- **R5 — effort.** Comparable to S4-B/VC. Rough: DDA.0–.2 ~1–1.5 wk,
  DDA.3–.6 ~3–4 wk, DDA.7–.9 ~2–3 wk, DDA.10 depends on audit. ~6–10
  weeks total; targets 0.5.0 / 0.6.0.

## Validation (every sub-substage)

- `cargo test` across the workspace stays green (voxlap goldens
  unchanged until DDA.9).
- New DDA visual goldens frozen per stage (image hash, not
  voxlap-bit).
- The four artifact repros are the headline acceptance gates and must
  flip to GREEN at their stage: `tiny_grid_1x1x1` (DDA.1),
  VC.0 pin + S4B.6.j (DDA.4), `axis_aligned_beam_repro` (DDA.6).
- Perf tracked against the `project_s4b_baselines` snapshot from
  DDA.3 onward.
