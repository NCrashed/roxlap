# roxlap-scene-demo

Interactive showcase of the [`roxlap-scene`](../roxlap-scene)
scene-graph engine: a free-flying camera over a procedurally-
generated planet surface with rotated multi-chunk voxel grids in
the sky.

## Vision (full-power demo)

Once the scene-graph stage closes (S7 in
[`PORTING-SCENE.md`](../../PORTING-SCENE.md)) the demo will
showcase every load-bearing capability of the engine:

- **Streamed planet surface** — effectively infinite procedural
  terrain (Worley + Perlin), grass / dirt / stone material based
  on slope and depth, hills, valleys, caves. Chunks load and
  evict around the camera.
- **Rotated, moving ships** — multi-chunk voxel grids in the sky,
  each with its own `DQuat` orientation. Ships drift, rotate,
  and approach the planet; the per-grid raycast composes them
  into the same framebuffer as the planet.
- **LOD spectrum** — flying from inside a ship corridor (1 m
  voxel detail) out to orbital altitude (planet rendered as a
  far-LOD billboard / sphere proxy) without a render-path break.
- **Composition correctness** — overlapping grids depth-merge
  correctly, including ship-on-planet shadows along occlusion
  seams.

## Demo evolution

The demo grows alongside the scene-graph substages — each
landed substage unlocks the next slice of the showcase. The
table also serves as the milestone gate: each row's "demo
delta" lands in the same commit / PR as the underlying
scene-graph work.

| Substage   | Status   | Demo capability                                                                                                         |
|------------|----------|-------------------------------------------------------------------------------------------------------------------------|
| **S3.x**   | landed   | 2 single-chunk axis-aligned grids: 128² hilly ground patch + 128³ ship hull. WASD + mouse-look free-fly camera.         |
| **S4.0**   | landed   | Ground extends to **2×1×1 chunks** (256×128×256) via the per-grid combined-world stitch (Approach C). Cross-chunk seam validated. |
| **S4.1**   | landed   | Ground extends to **32×32×1 chunks** (4096×4096×256). Terrain builder switched to batched `set_spans` per chunk (~150× fewer edit calls). ~7 s startup. |
| **S4.2**   | landed   | Ship grows to **4×6×1 chunks** centred on the ship grid origin (saucer ~220×350×12 voxels). Vspan-batched per chunk; multi-chunk axis-aligned validation alongside the planet-scale ground. |
| **S4.3**   | next     | Vertical-stack validation test — a 1×1×8 chunk grid (256-tall column world) per `PORTING-SCENE.md` § S4 acceptance criterion. Closes the S4 substage. |
| **S5**     | after S4 | Ship picks up an arbitrary `DQuat` orientation (default 45°/45°/45° pitch/yaw/roll). Camera math handles rotation.      |
| **S6**     | after S5 | LOD switching: ground renders as billboard at far distance, coarse-mip mid-range, full voxel close. Two ships at LOD-mid. |
| **S7**     | after S6 | Ground generates on demand from the camera; flying past the loaded radius streams new chunks in / evicts old.          |
| **S+**     | wishlist | Multiple planets with biome variation; ship physics; in-ship walking; planet-to-planet travel.                          |

The current S3.x scaffold isn't the final showcase — it's the
**vertical slice** that wires the scene-graph API end-to-end
through a real winit window so each substage's progress can be
seen and felt without standing up a new binary every time.

## Build + run

```sh
cargo run --release -p roxlap-scene-demo
```

`--release` matters: the per-pixel scan loops are too slow in
debug for an interactive frame rate.

## Scenes

The demo is a thin host driving one **scene** at a time, each a
focused showcase of one engine feature cluster. Press `Tab` to
open the scene menu and click a scene to switch; switching resets
all sprite / clip / character layers and snaps the camera to the
scene's start pose.

| Scene          | Showcases                                                                                  |
|----------------|--------------------------------------------------------------------------------------------|
| **World**      | Streaming hills + a rotating ship, flown with collision; chunk streaming, multi-grid composition, LOD billboards. `R` ship spin · `B` LOD billboards · `T` streaming telemetry. |
| **Sprites**    | A green/red `coco` field, a shoot-to-carve dense blob, and a streaming sprite ring. Left-click shoots the blob · `G` carves the red model's next layer. |
| **Animation**  | A KFA `animsprite`-driven swinging arm + a flame-clip character (RKC v3 attachment runtime). Both animate on their own. |
| **Picking**    | Top-down `view_ray` / `pick` (screen→world): a ground-plane cursor + left-click drops a marker at the picked surface voxel. |
| **Primitives** | Depth-tested debug-line gizmos (`draw_lines`) + a 2D reference quad (`draw_images`); left-click runs the `pick_image` eyedropper. |
| **Empty**      | A blank world (sky only) — the minimal `DemoScene`.                                         |

Each scene's specific controls show in the on-screen HUD (`F1`).

## Controls

Global (host-owned) controls, available in every scene:

| Input                 | Action                                                                            |
|-----------------------|-----------------------------------------------------------------------------------|
| `Tab`                 | Open / close the scene menu                                                       |
| `F1`                  | Toggle the HUD (backend, FPS, camera, active scene + its controls)               |
| Click in window       | Grab cursor (mouse-look active); a grabbed click is forwarded to the scene        |
| `Esc`                 | Close the menu, else release the cursor, else close the window                    |
| `W` / `A` / `S` / `D` | Forward / strafe-left / back / strafe-right (camera frame)                        |
| `Space` / `LShift`    | Up / down (world frame, voxlap convention `−z` / `+z`)                            |
| `LCtrl`               | Hold for 4× speed                                                                 |
| Mouse                 | Look around (yaw + pitch)                                                         |
| `+` / `-`             | Increase / decrease scan distance by 64 voxels (64..1024). Larger values showcase multi-mip's distant-LOD pixel band; smaller values cap mip-0 work for low-end CPUs. |

The camera collides with solid voxels (per-axis sliding, ±0.3
voxel skin) so you can't fly through the ground or the ship's
hull. World-bounded scenes still let the camera escape past the
chunk edges since out-of-grid positions count as air.

Lighting is voxlap's **lightmode 1** directional sun bake — surface
normals are shaded `(tp.y * 0.5 + tp.z) * 64 + 103.5` clamped to
`[0, 255]`. Baked into each chunk's slab alpha bytes once at
scene-build time; no per-frame light cost.

## Where the code lives

- `src/host.rs` — the thin host: winit `ApplicationHandler`, the
  [`SceneRenderer`] (CPU or GPU backend), the shared fly-camera +
  mouse-look, the egui HUD + scene menu, FPS, and the active
  `Box<dyn DemoScene>` + scene registry.
- `src/scene_api.rs` — the `DemoScene` trait every scene
  implements, the `SceneCtx` handed to it each frame, the
  `CameraRig`, and the shared `opticast_settings` / `frame_params`
  helpers.
- `src/scenes/` — one module per scene (`world`, `sprites`,
  `animation`, `picking`, `primitives`, `empty`); each owns its
  world content + per-scene update / input / render / overlays.
- `src/main.rs` — content helpers reused across scenes
  (`build_sprites`, `CarveTarget`, `Spinner`, `build_kfa`,
  `flame_character`, the debug-line / image / pick primitives) +
  `fn main` (hands off to [`host::Host`]).
- `src/scene.rs` — `build_demo()` (the World scene's streaming
  terrain + ship) and the streaming bake tracker.
- `src/terrain.rs` / `src/ship.rs` — terrain heightmap → chunk
  voxels and the ship hull geometry.

[`Scene`]: ../roxlap-scene/src/lib.rs
[`roxlap_scene::render::render_scene_composed`]: ../roxlap-scene/src/render.rs
