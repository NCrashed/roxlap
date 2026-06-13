# roxlap — GPU compute-shader renderer (Substage G)

Sub-substage roadmap and locked decisions for adding a **GPU
compute-shader renderer** alongside the existing CPU opticast.
Companion to [PORTING-RUST.md](PORTING-RUST.md) (the original
R1..R12 port), [PORTING-SCENE.md](PORTING-SCENE.md) (the
scene-graph engine), [PORTING-MULTICORE.md](PORTING-MULTICORE.md)
(R12 multicore), and [PORTING-WASM.md](PORTING-WASM.md) (R10 wasm
+ R10.X web demos).

This is a **new sibling renderer**, not an acceleration of
opticast. The CPU path stays as the byte-exact voxlap reference +
the oracle gate; the GPU path is "approximately the same look,
much faster" on platforms where it's available.

This document is the **start-of-stage brief**. A fresh-context
session should read this top to bottom before touching code.

## Status as of 2026-06-03

- 0.4.2 release-cut complete (PRR + AAMB landed). All major
  planned features in the original PORTING-RUST + PORTING-SCENE
  roadmaps are done.
- CPU opticast at the demo's spawn pose hits ~75 FPS engine-only
  (post-AAMB demo config). Game logic still needs to share that
  CPU budget — typical real games want 10-30% of frame time for
  AI / physics / scripting, which currently competes with render.
- User's motivation (2026-06-03): **free up the CPU budget for
  game logic** by moving the renderer to GPU, while **keeping
  voxlap's retro low-res aesthetic** (blocky pixels, no
  anti-aliasing, no PBR — the look is the feature).

## Goal

A WGPU-based compute-shader renderer that consumes the existing
`roxlap-scene::Scene` (grids, transforms, chunks) and produces a
framebuffer + z-buffer at framerates that leave most of the
frame budget free for game logic. Pixel-perfect equivalence with
the CPU path is **not** a goal; visual equivalence at the retro
aesthetic is.

End-state: the demo binary opens a window, picks the GPU
renderer if WGPU is available, falls back to CPU opticast
otherwise. The roxlap-web demos do the same via WebGPU.
Single-frame budget on a 2020-era iGPU at 640×480: <2 ms render
+ leftover for everything else.

## Locked decisions

| # | Decision | Consequence |
|---|----------|-------------|
| 1 | **Algorithm: DDA with coarse skip.** Two-level Bresenham/Amanatides — outer DDA over chunk grid, inner DDA over a chunk's voxel bitmap. Skip empty chunks via a 1-bit-per-chunk occupancy texture. | Simplest GPU-friendly approach; preserves voxlap's blocky aesthetic; no SVO/SDF complexity. |
| 2 | **Resolution: low (target 320×240..854×480).** Retro look is the feature. | At 640×480 = 307k pixels, even iGPUs handle this in <2 ms. SVO/SDF complexity buys nothing at this scale. |
| 3 | **Backend: WGPU (compute shaders, WGSL).** Cross-platform: Vulkan/Metal/DX12 native + WebGPU in browser. | Single shader codebase, works in the existing roxlap-web demos. |
| 4 | **New crate: `roxlap-gpu`.** Mirrors `roxlap-scene::render` shape; consumes `Scene` + `Camera` + sky/fog, produces framebuffer. | CPU opticast (`roxlap-core`) remains untouched as the byte-exact reference. Hosts pick a renderer at startup. |
| 5 | **Voxel data: decompress at upload time** into per-chunk (occupancy bitmap, color array, color offset map). | Removes slab-walking from the shader hot path; trades GPU memory for shader simplicity. ~130 KiB per chunk + variable color data. |
| 6 | **Lighting: pre-baked alpha bytes from the CPU bake.** Same as the CPU path. No runtime lighting on GPU in v1. | Single-pass shader. Emissive / dynamic light deferred. |
| 7 | **KV6 sprites: same algorithm.** Treat each KV6 as a tiny voxel grid in its own local frame; DDA it with the same shader. | Unified path for chunks + sprites; close-up sprites preserve parallax + retro silhouette. |
| 8 | **Edit invalidation: per-chunk dirty flag, re-upload on dirty.** | Simple. Adequate for hand edits + the demo's edit rate. Bulk-edit streaming gets per-column granularity in a later substage if it matters. |
| 9 | **NOT a goal: pixel-perfect parity with CPU path.** | Oracle's 12 byte-exact poses stay on the CPU path only. GPU has its own visual goldens + perf gates. |
| 10 | **NOT a goal: replacing CPU opticast.** | CPU path stays as the byte-exact voxlap reference + the oracle gate forever. GPU is a sibling. |

## Why DDA, not SVO or SDF

Three GPU voxel-rendering algorithm families exist; at this
project's resolution + aesthetic, **DDA wins by simplicity**.

| Algorithm | Strength | Weakness | Verdict here |
|-----------|----------|----------|--------------|
| **DDA grid marching** (this plan) | Embarrassingly parallel; one thread per pixel; ~50 lines of WGSL each for outer + inner DDA; no antialiasing changes the retro look | Per-voxel cost dominates at high res; needs voxel data in random-access form (not slab) | **Selected.** At 307k pixels + 512-voxel max scan, total work is <150M ops/frame — trivial. |
| **SVO traversal** (Crassin "Gigavoxels", Laine & Karras) | Asymptotic O(log n) skipping for huge sparse worlds; mature literature | Per-ray octree-stack management; building + editing octrees is expensive; doesn't fit per-chunk edits well | Skip. Complexity earns its keep at millions-of-rays scale; at 307k rays the constant overhead exceeds the savings. |
| **SDF cone tracing** (Teardown) | Gorgeous AA + global illumination; very fast on modern GPUs | Bakes voxels into SDF (lossy + expensive); explicit anti-aliasing destroys the blocky look | Skip. AA is the opposite of what the user wants. |

Modern voxel games that look like the user wants (low-res,
blocky, retro) almost all use DDA. The complexity of SVO / SDF
exists for a different problem space.

## Architecture

```
roxlap-formats                  (unchanged — voxel storage)
   ↓
roxlap-scene                    (unchanged — Scene + Grid + transform + streaming)
   ↙              ↘
roxlap-core                  roxlap-gpu (NEW)
(CPU opticast,               (WGPU compute shader renderer)
 byte-exact voxlap)
```

`roxlap-gpu` exposes roughly:

```rust
pub struct GpuRenderer<'window> {
    // WGPU device, queue, surface, pipeline, bind groups...
}

impl<'window> GpuRenderer<'window> {
    pub async fn new(window: &Window, settings: GpuRendererSettings) -> Result<Self>;

    /// Mirror of roxlap-scene's render_scene_composed shape.
    /// scene + camera + sky → framebuffer presented to the surface.
    pub fn render(&mut self, scene: &Scene, camera: &Camera, sky: Option<&Sky>);

    /// Tell the renderer which chunks are dirty since the last render.
    /// scene-graph edits already track this; we just plumb it in.
    pub fn invalidate_chunks(&mut self, grid: GridId, chunks: &[IVec3]);
}
```

Host binaries (`roxlap-scene-demo`, `roxlap-web`, `roxlap-host`)
pick a renderer at startup:

```rust
let render_backend = if want_gpu && GpuRenderer::is_available() {
    Backend::Gpu(GpuRenderer::new(...)?)
} else {
    Backend::Cpu(roxlap_scene::render::render_scene_composed_setup(...))
};
```

## Data representation

Per chunk, uploaded to GPU once at chunk-arrival (streaming or
initial load) and re-uploaded on edit:

| Resource | Shape | Size (CHUNK_SIZE_XY=128, CHUNK_SIZE_Z=256) |
|----------|-------|---------------------------------------------|
| `occupancy[chx, chy, chz][x, y, z]` | 1 bit per voxel, packed u32 array | 128·128·256 / 8 = **64 KiB / chunk** |
| `color_offsets[chx, chy, chz][x, y]` | u32 per column = base index into color array | 128·128·4 = **64 KiB / chunk** |
| `colors[grid_id][...]` | packed u32 per occupied voxel, concatenated per chunk | typically **30-100 KiB / chunk** for terrain |

Plus per-grid resources:

| Resource | Shape |
|----------|-------|
| `chunk_occupancy[chx, chy, chz]` | 1 bit per chunk ("has any voxels at all") |
| `chunk_meta[grid_id][...]` | per-chunk: pointer to occupancy bitmap, color array offset/length |
| `grid_transform[grid_id]` | mat4 (world → grid local) |

Memory budget at the demo's 32×32 chunks_z=1 ground grid:
1024 chunks × (64 KiB + 64 KiB + ~50 KiB color) = ~180 MiB GPU
memory. Comfortable on every iGPU since 2018. Streaming makes
this strictly the working set, not the whole world.

## Shader structure (sketched)

One compute shader, dispatched once per frame at
`workgroup_size = (8, 8, 1)` over the framebuffer. Each thread =
one output pixel.

```wgsl
@compute @workgroup_size(8, 8)
fn render_frame(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel = vec2<i32>(gid.xy);
    if (pixel.x >= screen_size.x || pixel.y >= screen_size.y) {
        return;
    }

    let ray = camera_ray_for_pixel(pixel);
    var best_depth = max_scan_dist;
    var best_color = sky_color(ray.dir);

    // For each grid (small N, typically 2..10)
    for (var g = 0u; g < num_grids; g++) {
        let local_ray = transform_world_to_grid_local(ray, grids[g].transform);

        // Outer DDA over chunk grid (skip empty chunks via occupancy bit)
        let outer = dda_chunk_grid(g, local_ray, best_depth);

        if (outer.hit) {
            let inner = dda_chunk_voxels(g, outer.chunk_idx, local_ray, outer.t_enter);
            if (inner.hit && inner.depth < best_depth) {
                best_depth = inner.depth;
                best_color = apply_fog(inner.color, inner.depth);
            }
        }
    }

    output_color[pixel.y * stride + pixel.x] = pack_rgba(best_color);
    output_depth[pixel.y * stride + pixel.x] = best_depth;
}
```

`dda_chunk_grid` and `dda_chunk_voxels` are each ~50 lines of
WGSL implementing the standard 3D Amanatides-Woo voxel
traversal: one cell-step per loop iteration, t-min selection
across the three axes.

Multi-grid composition is implicit in the per-pixel `best_depth`
tracking — no separate compose pass.

## Sub-substages

| # | Scope | Est | Validation gate |
|---|-------|-----|-----------------|
| **GPU.0** | **Probe.** Standalone WGPU test program: hand-built voxel chunk (just enough to make a recognisable shape), single compute shader, measure FPS at 320×240, 640×480, 1280×720 on the developer's hardware. ~300 LOC. Goal: empirically confirm the FPS ceiling and the "GPU stays idle most of the frame" claim. | **1-2 d** | Probe runs; FPS recorded on at least one iGPU and one dGPU; numbers documented in the memo. **If the probe shows the algorithm can't hit target FPS, the whole stage closes here.** |
| **GPU.1** | **Crate skeleton.** New `roxlap-gpu` crate. WGPU device + surface + swap-chain setup, empty render pipeline that clears to a colour, framebuffer presents. Host binary opt-in via env var. | 3-4 d | `roxlap-scene-demo --gpu` shows a coloured background; no voxels yet; CPU path stays default. |
| **GPU.2** | **Voxel data upload.** Decompress one chunk on CPU into (occupancy bitmap, color array, color offset map) and upload to GPU. Validate by reading back a known voxel via a debug shader. | 4-5 d | Read-back round-trip verifies a hand-set voxel; benchmark single-chunk upload time. |
| **GPU.3** | **Inner DDA shader.** Implement `dda_chunk_voxels` against a single uploaded chunk. Camera math + per-pixel ray + voxel hit + color output. No multi-grid, no chunk skipping, no sprites, no transforms. | 1-2 w | Single-chunk demo renders at expected FPS; visual comparison against CPU path matches at the retro aesthetic. |
| **GPU.4** | **Outer DDA + chunk skipping.** Multiple chunks in a grid. Chunk occupancy texture. Outer DDA walks chunk indices, inner DDA fires per-chunk. | 1 w | Demo's full 32×32 ground grid renders; FPS within probe's projection; visual continuity at chunk seams. |
| **GPU.5** | **Multi-grid + per-grid transform.** Loop over grids, transform world ray → grid-local ray per grid, composite via best_depth. | 4-5 d | Demo's ground + ship render together; depth correct at occlusion seams; ship rotation works. |
| **GPU.6** | **Edit invalidation pipeline.** Per-chunk dirty flag in `roxlap-scene::Grid` (or a host-side tracker). Before each render, re-decompress + re-upload all dirty chunks. | 4-5 d | Editing voxels at runtime updates the GPU render within one frame; bench upload bandwidth at edit-heavy workloads. |
| **GPU.7** | **Streaming integration.** `Scene::pump_streaming` hands new chunks to a main-thread upload queue; queue drains one or more chunks per frame; capped depth so a burst doesn't stall the render. | 4-5 d | Streaming-hills demo works on GPU; chunk pops minimal; no upload stalls visible. |
| **GPU.8** | **Sky + fog.** Panoramic sky texture upload; fog blend. Match the CPU path's visual at typical poses (not pixel-equal). | 3-4 d | Demo sky looks "right"; fog dims distant terrain. |
| **GPU.9** | **KV6 sprite path.** Each KV6 = small voxel grid, uploaded once, treated as a per-grid entity in the shader. KFA = frame-swap which sprite is active. | 1-2 w | Saucer ship renders correctly at all rotations; sprite pickups visible. |
| **GPU.X** | **Stage close.** WGSL shader cleanup, error-path audit, fallback-to-CPU when WGPU init fails, web demo verification, README + CHANGELOG, memo. | 3-4 d | All gates green; cuts a 0.5.0 release if this is a public surface. |

**Total**: ~6-10 weeks across GPU.0..GPU.X depending on how
deep the sprite + streaming integration goes.

The stage **fails fast at GPU.0** if the probe shows the FPS
ceiling isn't where we expect. That's the only point with a
real "abandon" exit; everything after it is on a predictable
trajectory.

## Risk + mitigations

**R-GPU.0.A — GPU memory pressure at larger worlds.**
- 1024 chunks × ~180 KiB = 180 MiB GPU. The demo is fine.
  Procedural infinite worlds need streaming to keep working
  set bounded.
- Mitigation: GPU.7 streaming integration; eviction matches
  CPU-side `Scene::pump_streaming`'s r_evict radius.

**R-GPU.0.B — Edit bandwidth at bulk procedural rates.**
- Streaming a chunk = ~150 KiB upload. At 30 chunks/s (= the
  S7.6 hills demo's worst case): 4.5 MiB/s. Comfortable.
- For bigger procedural workloads (e.g. a full planet
  regen), the per-chunk uploads might stall the queue.
  Mitigation: cap upload queue at N chunks per frame; spread
  large regens across multiple frames. Don't pre-optimise.

**R-GPU.0.C — Shader divergence at chunk-XY transitions.**
- Adjacent pixels can be in different chunks at chunk-XY
  boundaries → SIMT lanes diverge. Probably noisy.
- Mitigation: use tile-based dispatch with workgroup-shared
  bookkeeping; in practice modern GPUs handle this fine.
  Measure during GPU.4.

**R-GPU.0.D — WebGPU compute-shader feature gaps.**
- WebGPU's compute support is mature but some features
  (storage textures with arbitrary formats, atomic operations
  on storage buffers) have spec restrictions.
- Mitigation: keep the shader simple; use plain storage
  buffers + writeonly framebuffer texture; avoid atomics in
  v1 (we don't need them — first-hit wins per pixel).

**R-GPU.0.E — Per-grid rotation precision.**
- World position math at planet scales needs f64; WGSL only
  has f32. Same problem the CPU path solves with f64 grid
  origins.
- Mitigation: keep ray origin/direction in **grid-local f32**
  (do the world→local transform on CPU per frame, upload as
  uniforms). Each grid renders in its own f32 frame; the
  inter-grid f64 math stays on CPU. Same trick the CPU path
  uses.

**R-GPU.0.F — Driver compatibility.**
- Some Linux iGPUs have weak Vulkan/Mesa support.
- Mitigation: fallback to CPU path when GpuRenderer::new
  fails; document supported drivers in README.

## Validation gates

The CPU oracle (12 byte-exact poses) does NOT apply to the GPU
path — different algorithm, different rounding. The GPU path
needs its own validation:

1. **Per-substage visual gate.** A scripted "render N reference
   scenes, dump PPMs" pass at GPU.3, GPU.4, GPU.5, GPU.7, GPU.9.
   Compare against the CPU path's PPMs at the same camera —
   look for "approximately the same"; quantify by pixel
   difference (e.g. <2 % of pixels disagree by more than 8 grey
   levels).
2. **Per-substage FPS gate.** Bench at 640×480 at the demo's
   spawn pose; target ≥ 200 FPS on a 2020-era iGPU.
3. **CPU renderer stays untouched.** Every existing oracle +
   hash test (VC.5 baseline, VC.6.2, scene-demo 17/17, oracle
   10 MATCH + 2 CPU divergence) keeps passing. The GPU path
   adds, doesn't replace.
4. **Streaming + edit + rotation regression.** The streaming-
   hills demo, the ship-rotation poses, and the edit-flow tests
   all pass on the GPU path with expected visual fidelity.

## Out of scope (v1)

- **Anti-aliasing.** The retro look is the feature.
- **Dynamic lighting / emissive voxels.** Pre-baked lightmode-1
  alpha bytes are enough for the demo.
- **Reflections / shadows / GI.** Beyond v1.
- **DXVK / MoltenVK gymnastics.** WGPU handles platform choice;
  we don't ship per-platform shader variants.
- **Pixel-perfect parity with CPU.** Visual equivalence at the
  retro aesthetic is enough.
- **A WebGL2 fallback.** WebGPU is the web target. WebGL2
  doesn't have compute shaders; users on browsers without
  WebGPU support get the CPU path.

## Reading list (for the implementing session)

Required before GPU.0:
1. This doc, top to bottom.
2. [Amanatides & Woo's voxel traversal paper](http://www.cse.yorku.ca/~amana/research/grid.pdf)
   — the canonical 3D DDA algorithm we're implementing.
3. [WGPU's compute-shader hello-world example](https://github.com/gfx-rs/wgpu/tree/trunk/examples/src/hello_compute)
   for the rust + WGSL setup pattern.
4. `crates/roxlap-scene/src/render.rs` — what the new
   `roxlap-gpu::render` signature should mirror.
5. `crates/roxlap-scene/src/lib.rs` — Scene + Grid +
   GridTransform shapes the renderer consumes.

Useful background:
6. [Sébastien Hillaire's "DDA on a GPU" notes](https://www.shadertoy.com/view/4dX3zl)
   — typical performance characteristics.
7. [Crassin's Gigavoxels paper](https://maverick.inria.fr/Membres/Cyril.Crassin/thesis/CCrassinThesis_EN_Web.pdf)
   — for the road not taken (SVO).
8. `crates/roxlap-core/src/opticast.rs` — the CPU reference,
   for visual comparison + understanding the original
   algorithm we're approximating.

## Where this lands

- New crate `crates/roxlap-gpu/` — alongside `roxlap-core`,
  `roxlap-scene`, etc.
- WGSL shaders live in `crates/roxlap-gpu/shaders/`.
- Probe (GPU.0) lives in `crates/roxlap-gpu/examples/probe.rs`
  — self-contained, no Scene dependency.
- Host integration via a `--gpu` env var or feature flag on
  `roxlap-scene-demo`, `roxlap-web`, `roxlap-host`.
- Public version bump: `roxlap-core` stays at 0.4.x;
  `roxlap-gpu 0.1.0` releases independently. No breaking
  changes to existing public surfaces.

## How to enter GPU.0 in a fresh session

1. Read this file.
2. Read the required items in the reading list.
3. Start in `crates/roxlap-gpu/examples/probe.rs` (will need
   `cargo new --lib roxlap-gpu` + adding `examples/probe.rs`).
4. Target: a single static voxel chunk uploaded as
   (occupancy bitmap + color array); a compute shader that
   DDA-marches it from a fixed camera; results blitted to a
   window via WGPU surface; FPS counter in title bar.
5. Run on the developer's iGPU + dGPU + (if accessible) a
   recent phone via WebGPU. Record numbers.
6. Decide: continue to GPU.1 (yes, with confidence about the
   ceiling) or close the stage with a probe-result memo
   (algorithm doesn't hit target).

## Naming + version note

Substage prefix is `GPU` (GPU.0..GPU.X) within this document,
matching the per-stage prefix convention used by VC.0..VC.7
and CB / PRR / AAMB. Nested sub-substages can use `GPU.<n>.<m>`
when needed.

Public version targets (tentative):
- `roxlap-gpu 0.1.0` after GPU.5 (multi-grid + transform).
- `roxlap-gpu 0.2.0` after GPU.9 (sprite path complete).

The umbrella roxlap version (`roxlap-core` + `roxlap-scene`) is
independent; GPU work doesn't force a major bump there.

## GW — GPU renderer on the web (landed 2026-06-13, roxlap 1.0.0)

The last gap: bring the GPU path to the browser (WebGPU) and rebuild
the wasm demos on `roxlap-render` instead of direct `roxlap-core`
opticast. Landed as sub-substages GW.0..GW.4; cut **roxlap 1.0.0**.

| # | Scope |
|---|-------|
| GW.0 | `roxlap-gpu` builds for `wasm32`. `pollster`/`headless` gated native; `new_from_canvas` (`SurfaceTarget::Canvas`, no `Send+Sync`) + a shared `finish_init`. `read_depth_pixel` left compiled (facade just doesn't call it on wasm). |
| GW.1 | `roxlap-render` builds for `wasm32`. `softbuffer` native-only; CPU backend presents via a ported WebGL2 blit (`cpu_blit.rs`) on wasm; async `new_from_canvas_async` (GPU-first, CPU fallback, canvas cloned before the GPU attempt); wasm `pick_depth` returns `None`. |
| GW.2 | `roxlap-web` on the facade — procedural terraced-hills `Scene`, async init after `init_thread_pool`, RAF `render`+`present`, kept input + bench. |
| GW.3 | `roxlap-cave-web` on the facade — single-chunk cave `Scene`, scene collision (`Grid::voxel_solid`) + carving (`Grid::set_sphere` + new `Grid::bake_lightmode`), bullets as facade sprites. |
| GW.4 | 1.0.0: console backend message, docs, workspace 0.8.0→1.0.0, CHANGELOG. |

### Why these decisions

- **wgpu `!Send+!Sync` on wasm.** The wasm build uses `+atomics` +
  shared memory, so wgpu's `fragile-send-sync-non-atomic-wasm` does
  not apply. The single-threaded `Rc<RefCell>` browser host makes
  that fine; the canvas constructors carry no `Send+Sync` bound.
- **No blocking on wasm.** `pollster::block_on` and
  `device.poll(Wait)` have no WebGPU equivalent, so init is async and
  GPU depth-picking is deferred (the CPU fallback still picks).
- **CPU fallback kept.** WebGPU isn't universal (Safari/Firefox lag),
  so the facade's CPU opticast path stays available on the web,
  presented via WebGL2 instead of softbuffer.

### Browser-only caveats (not caught by `cargo build`)

- WebGPU may be disabled under COEP `require-corp` in some browsers;
  the CPU fallback covers it, but whether GPU activates in the
  threaded build is unverifiable offline.
- `!Send` wgpu resources must stay on the main thread (they do — the
  host is `Rc<RefCell>`; rayon workers only touch the CPU
  compositor's framebuffer slices).
- GPU click-picking returns `None` on wasm (deferred); CPU picks.
