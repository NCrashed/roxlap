# Handover: billboards keep world up, so a ROLLED camera leans them

Status: **IMPLEMENTED 2026-08-08 as BB.6** — see `PORTING-BILLBOARD.md`
(BB.6 row) and the CHANGELOG's `[Unreleased]`. What landed, versus the
proposal below:

- The **second spelling** was taken: `BillboardUp {World, Camera, Axis([f32;3])}`
  is an independent knob on `BillboardMode`, not a fourth mode. `Axis` was
  added on top of the two the doc asked for — it is the *physical* answer for
  the ship case (the card stands on the deck, so it neither leans off its
  anchor nor needs the camera to be the one riding the grid), and it
  subsumes `World`.
- `Cylindrical` now yaws about the chosen axis rather than the world
  vertical, so cylindrical-with-a-deck-up is a card upright *on the deck*.
- The **adjacent** `dir_index` item was taken too: `ActorFacing {Yaw, Dir}`
  plus the actor's up axis measures the bearing in the actor's own frame, so
  monada's consumer-side `billboard_yaw` workaround can be deleted (pass
  `ActorFacing::Dir(hull_rot * nose)` + `BillboardUp::Axis(deck_up)`, ideally
  through the one-call `set_actor_pose`).
- Both properties the doc asked to keep are pinned by tests: the anchor never
  moves, and the world-up default is unchanged (bit-for-bit — the world-yaw
  sector path is kept verbatim as a fast path).

Original request follows.

---

Status: **feature request**, reported 2026-08-08 from the monada ship demo
(roxlap 0.31.1 as published on crates.io; the repo's `crates/roxlap-render`
tree matches — line numbers below are against this working copy). Written for
a fresh session in THIS repo: the motivation, the exact code, the options and
what a fix has to keep are all spelled out. Nothing here is a bug in the
classic sense — it is the `BILLBOARD_UP` YAGNI note at
`crates/roxlap-render/src/lib.rs:658-661` coming due.

## Why now: a camera that RIDES a rotating grid

monada grew a `camera_grid(grid)` verb: the host turns the whole orbit frame —
basis *and* eye offset — by a `Scene` grid's rotation, so a spaceship's deck
holds still on screen while the starfield sweeps past. It is not a look; it is
a correctness fix. Entities bound to a grid carry **grid-local** positions, so
with a world-fixed camera the map's view-relative input steered in the ship's
frame while the player watched the world's, and "forward" pointed somewhere new
every tick the hull turned.

The ship's hull tumbles about a TILTED axis (mostly yaw with a lean), so the
camera now genuinely **rolls**: `Camera::right` / `down` are no longer level
with the world. Everything drawn from a grid follows correctly — the hull, the
crates (posed via `DynSpriteTransform`), the fog twin. The crew do not: they
are `BillboardActor`s, and their cards stay pinned to world up, so on screen
they lean by the camera's roll angle while standing on a deck that looks level.

## What the code does today

`billboard_transform` (`crates/roxlap-render/src/lib.rs:682`) builds the slab
basis from the camera **position** only:

```rust
const BILLBOARD_UP: [f32; 3] = [0.0, 0.0, -1.0];        // :662

let ny = match mode {                                    // :703-707
    BillboardMode::Cylindrical => bb_norm([to_cam[0], to_cam[1], 0.0])…,
    BillboardMode::Spherical   => bb_norm(to_cam)…,
    BillboardMode::None        => return None,
};
let nx = bb_norm(bb_cross(BILLBOARD_UP, ny))…;           // image horizontal
let nz = bb_cross(ny, nx);                               // image vertical
```

So the two modes differ only in the **normal**; the image's vertical comes from
the world-up constant in both. The unit tests pin exactly that:
`billboard_cylindrical_faces_camera_upright_and_ignores_height` asserts
`xf.forward == BILLBOARD_UP` (:4684), and the spherical test asserts only that
it *differs* (:4717).

Callers already hold what a fix needs: `face_billboards_to(camera)` (:3272) and
`update_billboard_actors(camera, dt)` (:3439) both take `&Camera`, whose
`right` / `down` / `forward` are the screen axes. Only `billboard_transform`
narrows that to `cam: [f64; 3]`.

## The proposal

Let the app choose where a card's vertical comes from. Sketch (naming is
yours — this is the shape, not a patch):

```rust
pub enum BillboardMode {
    None,
    Cylindrical,          // unchanged: image up = world up
    Spherical,            // unchanged: normal = view dir, image up ≈ world up
    /// Screen-locked: normal = view direction, image up = the CAMERA's up
    /// (`-camera.down`). The card never leans relative to the viewer, so a
    /// rolled or grid-riding camera keeps upright art upright.
    CameraUp,
}
```

with `billboard_transform(pos, camera: &Camera, mode)` and, for the new arm,
`up = -camera.down` in place of `BILLBOARD_UP` (falling back to the constant if
`up × normal` degenerates). An equally good spelling is a separate
`BillboardUp { World, Camera }` axis on the existing modes — that composes
better (cylindrical-with-camera-up is meaningful too: a card that yaws to the
camera but rolls with it), at the cost of a second knob on
`BillboardActorDef` (:803).

**Recommendation:** the second one, if the extra field is acceptable —
"which way does the card face" and "which way is up in the image" really are
independent, and the cross-product code is already shared.

Two properties worth keeping in whatever lands:

- **The anchor must not move.** Rolling the card about its own normal pivots it
  around the instance anchor, so a character's feet stay planted where they
  are. That is the failure `Spherical` has at steep pitch (the reason monada's
  `actor_def` picks `Cylindrical`, with a comment about the body leaning off its
  ground anchor) — a camera-up mode must not reintroduce it.
- **Shadows stay the app's call.** The mode docs already note that a card whose
  orientation tracks the camera casts a shadow that rotates as you orbit. That
  is exactly why this belongs in the enum rather than becoming the default:
  Doom/Build-style scenes want world-up cards and sane shadows, a ship interior
  wants screen-up crew.

Back-compat: additive. `Cylindrical` / `Spherical` keep their bases and their
two tests; only maps that ask for the new mode change. On the monada side,
`actor_def` in `monada-host/src/map_render.rs` is the single call site to flip.

## Validating it

No visual pass needed for the core claim — a unit test alongside the two above
covers it: build a camera whose basis is rolled about `forward` (e.g. rotate
`right` / `down` by 30°), call `billboard_transform` for a card at the origin,
and assert the resulting image vertical is parallel to `-camera.down`, while
`Cylindrical` still returns `BILLBOARD_UP`. The visual pass then is the ship
demo (`cargo run -p monada-ship` in the monada repo): the hull tumbles about
`(0.3, 0, 1)`, the camera rides it, and the crew should stay upright on screen
instead of leaning with the deck.

## Adjacent, already solved consumer-side — but you may want it here

roxlap picks a `BillboardActor`'s directional sprite in `dir_index` (:880) from
`bearing(camera − actor) − facing_yaw`, **both world-space**. A consumer whose
actors ride a rotating grid must therefore hand over a facing composed into
world — and composing means *rotate, then project to horizontal*, which does
not commute with a tilted rotation. The two projections turn by different
amounts, so the chosen sprite drifts: at the ship's tumble, 0.27 rad — a third
of a 45° sector — and the crew member visibly turns on the spot while standing
still.

monada fixed this on its side (`billboard_yaw` in `map_render.rs`): measure the
angle in the grid's own frame, where the nose is defined and a grid-riding
camera sits still, then feed roxlap a world yaw that reproduces it. Identity
and yaw-only rotations come out bit-identical to before, so nothing else moved.

It is worth knowing upstream because **every** consumer with actors on a
rotating body will have to rediscover it. If you would rather own it here, the
smallest version is to accept the actor's facing as a direction vector plus an
optional frame (or a full quaternion) instead of a world yaw, and do the
grid-frame bearing inside `dir_index`. Not urgent — monada is unblocked — but
the current signature quietly assumes actors only ever stand on a world-aligned
floor.

## Paperwork if this lands

`docs/porting/PORTING-BILLBOARD.md` carries the BB stage log (BB.0–BB.5, all
done); this would be **BB.6**. The `BILLBOARD_UP` comment at :658-661 says
plainly that generalising it is unexposed YAGNI — that note is the thing being
retired, so it should go with the change rather than be left contradicting the
code. CHANGELOG entry + a minor version cut; monada pins roxlap from crates.io,
so it picks it up on the next bump.
