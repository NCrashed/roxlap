# roxlap — character controller (Stage CC)

Entry doc written 2026-07-06 at workspace 0.22.0 (+ unreleased QE-B6
colour newtypes / missing_docs / QE-C6), queued at the 2026-07-06
engine-backlog triage. This is the **entry doc** for the
character-controller stage — tag **CC**. A fresh-context session
should read it top to bottom before touching code.

Naming disambiguation up front: this stage is about a **walking
body** — collision, gravity, jumping. It is *not* about
`roxlap-formats/src/character.rs` (the `.rkc` animated-model
container, stage RKC) — a `.rkc` character is what you *draw*, a
character controller is what you *stand on the ground with*. CC.4
connects the two.

## Status — OPEN (CC.0 + CC.1 landed; CC.2 next)

## Phase log

- CC.1 — LANDED 2026-07-06: `roxlap-scene/src/character.rs` —
  `CharacterDef` (radius/height/eye_height, gravity/jump_speed,
  walk_speed + `accel_ground`/`accel_air` approach model, `Solidity`)
  / `CharacterBody` (feet-positioned, `walk(scene, dt, WalkInput)`)
  / `WalkInput { wish, jump }`. Substeps ≤ radius (MAX_SUBSTEPS
  anti-hang truncation), per-axis x→y→z with flush plane clamp
  (SKIN = 1e-3; rotated-geometry fallback = reject axis), +z contact
  grounds / −z bumps, end-of-frame skin probe drives `on_ground`,
  stuck-escape kept verbatim. 8 trajectory tests: land flush on the
  plane, walk-speed convergence, wall clamp+slide, jump apex vs
  v²/2g, head bump flush under ceiling, 20-voxel/step fall onto a
  1-voxel floor (no tunnel), stuck escape, bit-identical determinism.
  Test lesson: spawn fixtures INSIDE the intended gap — a body
  dropped from above a ceiling slab settles on top of it.

- CC.0 — LANDED 2026-07-06: `roxlap-scene/src/collide.rs` —
  `Solidity` (bedrock knob only, `Copy`),
  `box_overlaps_solid`/`point_overlaps_solid`/`grid_box_overlaps_solid`
  (the single-grid form is public for the cave demos' one-grid
  hosts). Exact floor-range probe for axis-aligned grids,
  corner-AABB conservative for rotated. 7 unit tests: the three
  ported scene-demo tests, slab-interior `UnexposedSolid`, bedrock
  policy both ways, rotated-grid block/far-air, point≡degenerate-box.
  Demos untouched (their copies die in CC.3).

## Goal

An engine-owned first-person/third-person character controller over
`Scene`: axis-swept move-and-slide, gravity, jumping, ground
detection, auto step-up, fly/noclip modes. Host code becomes:

```rust
use roxlap_scene::{CharacterBody, CharacterDef, WalkInput};

let mut body = CharacterBody::new(CharacterDef {
    radius: 0.4,      // xy half-extent of the collision box, voxels
    height: 1.8,      // feet → head, extends UP i.e. toward -z
    eye_height: 1.62, // feet → camera, same axis
    step_up: 1.05,    // auto-step ledges up to this many voxels
    gravity: 24.0,    // +z is DOWN, so gravity is POSITIVE
    jump_speed: 9.0,  // applied as NEGATIVE z velocity
    ..CharacterDef::default()
});
body.teleport(DVec3::new(64.0, 64.0, 100.0)); // FEET position

// per frame:
body.walk(&scene, dt, WalkInput { wish, jump: space_pressed });
let cam = Camera::from_yaw_pitch(body.eye_pos().into(), yaw, pitch);
if body.on_ground() { /* footsteps, fall-damage reset, … */ }
```

Explicitly **not** in scope: a physics engine (no dynamic rigid
bodies, no forces on the world), riding moving/rotating grids (the
S5 ship-spin demo — standing on it will NOT carry the player; see
Hazards), swimming (deferred behind the CC.4 material hook),
crouching, NPC pathfinding, collision against sprites/clips (grids
only — sprites are props and effects).

## Prior art (read before designing anything)

### In-tree: the same hack, three times

All three interactive demos ship the *same* fly-camera collision,
copy-pasted and drifting:

| Copy | Shape |
|---|---|
| `roxlap-scene-demo/src/collision.rs` (212 lines, the most evolved) | point-in-cube probe `pos ± 0.3` over every grid, per-axis slide, bedrock-placeholder skip, already-stuck escape |
| `roxlap-cave-demo/src/main.rs:1032-…` | same probe against the single `Vxl` chunk |
| `roxlap-cave-web/src/lib.rs:225-…` | same again, wasm copy |

None has gravity, jumping, or ground detection — they are fly
cameras. The scene-demo copy carries the two hard-won lessons this
stage must keep (its unit tests move to the engine in CC.0):

1. **Solidity comes from `roxlap_core::world_query::getcube`** —
   `Cube::Color` *and* `Cube::UnexposedSolid` block (slab interiors
   are solid material; treating them as air lets the camera inside
   the saucer body), `Cube::Air` does not.
2. **The bedrock placeholder is a policy, not a fact.** Voxlap
   auto-maintains a solid voxel at chunk-local `z = 255`; the
   scene-demo renders it as air (`treat_z_max_as_air`) and its
   collision must agree or an invisible wall appears at the grid's
   bottom plane (the user-reported z=155 bug, fixed 2026-05).
   Whether z=255 blocks must stay a **knob** — the cave demo's
   bedrock is real floor.

### Voxlap: `clipmove` — studied and REJECTED

The original engine's controller
(`~/dev/voxlaptest/voxlap/voxlap5.c:4233`) is `clipmove`: an
iterative swept-**sphere** trace (`sphtrace`, :4040) with up to 3
slide iterations (plane slide, then edge slide via cross product),
plus `findmaxcr` (:3717) which *shrinks the sphere radius* to whatever
currently fits ("Shrinking radius error control hack", his words).
We do not port it:

- There is no oracle value — this is gameplay feel, not rendering;
  nothing downstream needs bit-parity with voxlap movement.
- The sphere-vs-voxel-face/edge/corner case analysis is the
  hairiest scalar code in the file, and the shrinking-radius hack
  exists precisely because it wasn't robust.
- It assumes one grid, `VSID` bounds, and voxlap's `MAXZDIM` clamp.

What we *do* keep from it: the slide formulation (project the
remaining displacement onto the contact plane and continue), the
idea that a controller is `p += v` with the world clipping `v`, and
the axis conventions.

## Design (locked)

### Home: `roxlap-scene`

The controller needs `Scene` and nothing else — no renderer, no
window; fully unit-testable headless (the PS.0 lesson: pure core
first). Two new modules:

- `roxlap-scene/src/collide.rs` — the query layer: solidity probe +
  box-overlap tests, promoted from the demo copies.
- `roxlap-scene/src/character.rs` — `CharacterDef` / `CharacterBody`
  / `WalkInput` on top of it.

Demos delete their copies in CC.3. `roxlap-render` re-exports
nothing new (games already depend on `roxlap-scene` directly).

### Body: an AABB, probed cell-exactly

The body is an axis-aligned box in world space: `radius` half-extent
in x/y, `height` along z, positioned by its **feet** (`pos.z` is the
lowest… i.e. *largest*-z point of the body — feet DOWN means feet at
the +z end; head at `pos.z - height`). f64 throughout, matching the
camera and `GridTransform` world (the precision-model chapter).

The overlap probe enumerates exactly the voxel cells the box
intersects, per grid (the `floor`-range loop `collision.rs` already
uses), and asks `getcube` per cell via `voxel_split` + chunk lookup:

- **Axis-aligned grids** (identity rotation): exact — translate,
  floor, loop. This is 99% of gameplay content.
- **Rotated grids**: transform the world box's 8 corners into grid
  local space with `addr::world_to_grid_local`, take the local AABB
  of the corners, probe that. Conservative (blocks slightly early
  near rotated geometry — a fat OBB approximation), never leaky.
  Documented on the API; exactness for rotated grids is out of
  scope (an OBB-vs-voxel SAT is not worth it for a controller).

A body of 0.8×0.8×1.8 spans 2×2×3 cells when axis-snapped
(worst case 3×3×3–4) — a probe is a few dozen `getcube`s. No
caching until measured (PF lesson: the naive loop was never the
bottleneck; `env::var` in the loop was).

### Solidity policy

```rust
pub struct Solidity {
    /// Chunk-local z = CHUNK_SIZE_Z-1 placeholder blocks? Default
    /// `false` — matches the demos' `treat_z_max_as_air` rendering.
    pub bedrock_blocks: bool,
    // CC.4: material hook — per-(grid, voxel) veto so glass stays
    // solid but water/ladders/foliage can pass. Deliberately absent
    // until then; do NOT add a closure field speculatively.
}
```

### Movement: substepped per-axis move-and-slide

Per `walk(dt)`:

1. Integrate: `vel.z += gravity·dt` (unless flying); horizontal
   `vel` toward `wish` (accel/friction split ground vs air — small
   constants in `CharacterDef`, Quake-style but not configurable
   beyond that in v1).
2. Split the displacement into substeps of length ≤ `radius` (never
   tunnel: one substep can't skip past a cell the box would touch).
3. Per substep, move axis-by-axis (x, y, then z): try the axis
   move; if the probe blocks, binary-search is NOT needed — clamp
   the coordinate flush against the blocking cell plane (the box
   and the cells are both axis-aligned; the contact plane is an
   integer plane, computable directly), zero that velocity
   component. Sliding falls out per-axis, exactly like the demos'
   `slide_with_collision`, but flush instead of reject-whole-axis
   (no more visible gap when hugging a wall).
4. **Step-up** (CC.2): when a *horizontal* axis blocks and
   `on_ground`, retry the same axis move with the body lifted
   `step_up` voxels; accept if it fits AND the landing probe finds
   ground under the new spot; else fall back to the slide.
5. z-axis contact sets flags: moving +z into solid ⇒ `on_ground`
   (and `vel.z = 0`); moving −z into solid ⇒ head bump
   (`vel.z = 0`).
6. **Already-stuck escape** (kept from the demos, verbatim
   semantics): if the start-of-frame pose already overlaps solid
   (edit under the player, bake reclassification), all axes move
   freely this frame so the player can escape. Active depenetration
   is deferred — the escape rule has years of demo mileage.

`on_ground()` between frames = probe a `1e-3`-skin box just below
the feet. Coyote time + jump buffering are two small timers in
CC.2 (feel, cheap, testable).

Modes: `Mode::{Walk, Fly, Noclip}` — `Fly` is exactly today's demo
behaviour (slide, no gravity), `Noclip` skips probes entirely. The
demos' F-key flight stays available through the same body.

### Determinism

Pure f64, no RNG, fixed substep rule ⇒ same scene + same input
sequence = identical trajectory. Unit tests pin trajectories the way
PS.0 pinned sim goldens: walk-into-wall slides along it, fall lands
exactly on the surface plane, jump apex, step-up onto a 1-voxel
ledge, refusal at a 2-voxel wall, bedrock knob both ways, stuck
escape, rotated-grid conservative block.

## Hazards (read before each stage)

1. **+z is DOWN.** Gravity is *positive* z, jump impulse *negative*,
   "head" is at `pos.z - height`, falling is z *increasing*. Every
   sign error here compiles and "works" upside down (the PS hazard
   list said the same; it bit anyway).
2. **Bedrock placeholder policy** — collision and rendering MUST
   agree per app or the invisible-wall bug returns (scene-demo:
   placeholder = air; cave-demo: its floor is real voxels well above
   z=255 so either setting works — but set it consciously).
   Corollary: with `bedrock_blocks: false` and gravity, a hole
   carved to the chunk bottom drops the player out of the world —
   hosts should set a kill-z (documented, not engine policy).
3. **Tunneling** — the substep rule is the guarantee; do not "fast
   path" large displacements past it. Falling at 60 voxels/s at
   60 fps is 1 voxel/frame ≈ 2–3 substeps: cheap.
4. **Rotated grids are conservative, moving grids don't carry.**
   The R-hotkey spinning ship: player standing on it stays put
   (world-frame) while the deck rotates away — expected, documented.
   Platform carry (velocity inheritance) is a future stage if a game
   ever needs it; it needs per-frame grid-transform deltas.
5. **Multi-grid overlap ORs** — any grid's solid blocks; overlapping
   grids just both probe (same as `is_blocked_in_scene` today).
6. **Don't regress the demo lessons**: `UnexposedSolid` blocks
   (saucer-interior test), out-of-grid = air (fly past footprints),
   stuck = escape. Port the scene-demo unit tests INTO
   `roxlap-scene` in CC.0 before writing any new movement code —
   they are the regression net.
7. **f32 attach jitter** — a third-person `.rkc`/billboard actor
   posed at the body inherits the f32 sprite world; far-from-origin
   jitter is the known sprite property, not a CC bug (PS hazard 5).
8. **`getcube` cost discipline** — the probe loop calls
   `grid.chunk(idx)` per cell (a `HashMap` hit each). At controller
   scale this is nothing; resist adding a chunk cache before a
   profile says otherwise (PF lesson — measure first).

## Stage list

| Stage | Contents | Breaking? |
|---|---|---|
| CC.0 | Entry doc + `roxlap-scene/src/collide.rs`: `Solidity`, box-overlap probe (exact axis-aligned / conservative rotated), ported scene-demo tests + new rotated-grid test | no — additive |
| CC.1 | `character.rs`: `CharacterDef`/`CharacterBody`/`WalkInput`, substepped per-axis move-and-slide, gravity, jump, ground/head flags, flush-clamp contact; trajectory unit tests | no |
| CC.2 | Feel pass: auto step-up + landing check, coyote time, jump buffering, `Mode::{Walk,Fly,Noclip}` | no |
| CC.3 | Demo migration: scene-demo Walk/Fly toggle on the new body (`collision.rs` deleted), cave-demo + cave-web on the same core (their copies deleted) | no (demo-internal) |
| CC.4 | Material-aware `Solidity` hook (TV materials: glass solid, water pass-through) + third-person demo: billboard actor / voxel-clip posed at the body | no |
| CC.5 | Perf sanity probe + book snippet (anchored example, `check-anchors.sh` gate) + CHANGELOG roll-up | docs |

Versioning: purely additive ⇒ rides the next minor. Each CC.n lands
as its own commit with a CHANGELOG entry; CC.3 is the
delete-three-copies payoff commit.
