# Demo tour

`roxlap-scene-demo` is the engine's living feature gallery: eleven
scenes behind one host, each built to showcase one subsystem — and
each chapter of this book uses "its" scene as the worked example.

```sh
cargo run --release -p roxlap-scene-demo                 # CPU backend
ROXLAP_GPU=1 cargo run --release -p roxlap-scene-demo    # GPU backend
ROXLAP_SCENE=Doom cargo run --release -p roxlap-scene-demo  # start on a scene
```

Global controls: `Tab` opens the scene menu, `F1` toggles the HUD
(FPS, camera, per-scene status, and the live **Render pipeline**
panel from [chapter 5](render-pipeline.md)), `Esc` releases the
cursor. WASD + mouse-look everywhere.

## The scenes

| Scene | Showcases | Book chapter |
|---|---|---|
| **World** | Streaming hills + a rotating ship: multi-grid composition, chunk streaming, LOD billboards, collision-checked flight. `R` spins the ship, `T` prints streaming stats, `H` jumps to a top-down vantage. | [3](scene-graph.md) |
| **Sprites** | Sprite models + the dynamic add/remove API: a checkerboard `coco` field, a shoot-to-carve blob, a streaming spinner ring, per-instance tints. | [7](sprites.md) |
| **Animation** | KFA skeletal animation + an `.rkc` character with a flame voxel-clip attachment, side by side. | [7](sprites.md) |
| **Transparency** | Alpha glass, additive glow, pulsing smoke, a mixed-material window, a volumetric fog cloud — over translucent terrain. | [6](lighting.md) |
| **Lighting** | The runtime rig: sweeping shadow-casting sun + orbiting point lights, stylized bands (`J`, `[`/`]`), live AO retuning (`N`/`M`). | [6](lighting.md) |
| **Spotlight** | One spot cone in a dark room: `F` flashlight mode, `O` sweep, `[`/`]` cone width, `K` shadows. | [6](lighting.md) |
| **Particles** | The particle system: fountain over a water pool, smoke column, crosshair-aimed explosions that carve craters. | [8](particles.md) |
| **Audio** | The acoustics core made walkable: stone room / open field / glass cavern with live occlusion, reverb environment and Doppler numbers in the HUD (from the pure core — every build has the tab; `--features audio` also plays it). | [9](audio.md) |
| **Scale** | Per-grid `voxel_world_size`: a chunky planet (`4.0`) + a fine bobbing ship (`0.25`) at a true 16× ratio in one scene; `P` pauses the bob, `K` toggles sun shadows. | [3](scene-graph.md) |
| **Doom** | GIF billboard sprites + an 8-way directional actor (`Q`/`E` turn it, `Space` walk/idle) — synthesised as GIFs at startup, so the full import path runs live. | [7](sprites.md) |
| **Picking** | Top-down cursor picking: `view_ray`/`pick`, snap-to-surface, click-to-mark. | [12](picking.md) |
| **Primitives** | Overlay gizmos (`draw_lines`) + 2D image quads (`draw_images`) + the `pick_image` eyedropper. | [4](rendering.md) |
| **Empty** | Just sky — the minimal `DemoScene` contract, and a handy GPU-vs-CPU present-cost baseline. | — |

## Demo environment variables

These belong to the demo host, not the library (the engine's own
variables are in [chapter 14](tuning.md)):

<!-- Source of truth: grep -rhoE '"ROXLAP_[A-Z_0-9]+"' crates/ --include='*.rs' | sort -u,
     minus the four engine vars listed in tuning.md. Re-verify at stage closes. -->

| Variable | Effect |
|---|---|
| `ROXLAP_GPU=1`/`0` | Prefer the GPU backend / force CPU (all demos + examples) |
| `ROXLAP_SCENE=<name>` | Start on a scene (table above) |
| `ROXLAP_RENDER_RES=WxH\|native\|<scale>`, `ROXLAP_SSAA=n`, `ROXLAP_POSTERIZE=n`, `ROXLAP_DITHER=bayer\|blue\|none` | Seed the render-pipeline panel from the environment |
| `ROXLAP_STATIC=1` | World scene: static 32×32 ground instead of streaming hills (A/B + repro) |
| `ROXLAP_STACKED_GROUND=1` | World scene: the stacked-chunk ground variant (regression repros) |
| `ROXLAP_NO_MARKERS=1` | World scene: drop the LOD marker pillars (engine-only perf runs) |
| `ROXLAP_SPRITE_GRID=n` | Sprites scene: coco field size (default 4) |
| `ROXLAP_NO_CRUMBLE=1` | Cave demo: plain carves, no island crumble ([chapter 10](destruction.md)) |
| `ROXLAP_RKC=path`, `ROXLAP_RKC_DUMP=path`, `ROXLAP_KFA_DUMP=path` | Animation scene: load / dump character containers |
| `ROXLAP_AUTOFIRE=1`, `ROXLAP_NOFLASH=1`, `ROXLAP_FPS_LOG=1` | Particles / host probes: scripted explosions, flash A/B, per-second FPS log |
| `ROXLAP_CAPTURE=<dir>` (+ `_MS`, `_FRAMES`), `ROXLAP_CAMERA=x,y,z,yaw,pitch` | Gallery recorder: write PPM frames on a wall-clock interval, then exit; override the start pose for framing |

(`roxlap-host` and the GPU probe example carry a few more of the same
flavour — `ROXLAP_HOST_SPRITE_*`, `ROXLAP_GPU_RES` — documented in
their own sources.)

## Beyond the gallery

Two more demos ship in the workspace and double as integration
references: **`roxlap-cave-demo`** (procedural caves, plasma-bullet
carving with local relight — the README quickstart; rock the carve
disconnects **crumbles**: it detaches, falls and bursts into debris,
crystals shattering into glowing plates, [chapter 10](destruction.md);
run it with `--features audio` for voxel-muffled sound,
[chapter 9](audio.md)) and the **web demos** from
[chapter 13](platforms.md). And the book's own
examples (`quickstart`, `book_*` in `roxlap-render`, `roxlap-scene`,
`roxlap-core`, `roxlap-formats`, `roxlap-audio`) are deliberately
minimal single-topic hosts — often the better starting point for your
own project.

For a *complete game* on the engine, see the community-built
[**roxlap-game-demo**](https://github.com/Aminion/roxlap-game-demo):
a mining ship in procedurally generated asteroid fields — physics
flight, cannon + tractor beam, crystal collection, an energy economy
— everything this book covers, assembled into one playable loop.
