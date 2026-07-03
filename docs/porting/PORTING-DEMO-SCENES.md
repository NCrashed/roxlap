# roxlap-scene-demo → menu-driven multi-scene showcase

Start-of-stage brief and locked decisions for **refactoring
`roxlap-scene-demo` from a single kitchen-sink `App` into a thin host +
a set of focused, menu-selectable demo scenes**, each showcasing one
cluster of engine features. Demo-only — no library change. Stage tag
**DS**.

This is a **start-of-stage brief**. A fresh-context session should read
it top to bottom before touching code.

## Why

`crates/roxlap-scene-demo/src/main.rs` is **2296 lines** with a ~34-field
`App` and ~20 hotkeys that mash together, all live at once:

- streaming hills terrain + a rotating ship + fly-camera collision,
- a checkerboarded coco sprite field + a shoot-to-carve target (`G`),
- a streaming "spinner" ring (dynamic sprite add/remove dogfood),
- a KFA swinging arm + (VCL.7) a flame voxel-clip character,
- screen→world picking (`C`) with dropped markers,
- world-placed image sprites (`I`) + debug-line gizmos (`L`),
- an in-app bench (`B`), `F`-capture, an A/B pose toggle (`H`),
- and a pile of `ROXLAP_*` env knobs.

Every new feature makes it worse to read, and the features visually
collide. The fix: **one feature cluster per scene**, picked from an
in-app menu, behind a small `DemoScene` trait, so `main.rs` becomes a
thin host and each scene is a self-contained module.

(`repro.rs`, 2320 lines, is a `#[cfg(test)]` regression module — *not*
demo runtime — and is untouched by this stage.)

## Locked decisions

Taken with the engine author 2026-06-26:

1. **Menu = an egui scene-picker panel, toggled by a key.** Reuse the
   existing egui HUD seam (`SceneRenderer::paint_egui`); a key (e.g.
   `Tab`) opens/closes a "Scenes" panel listing the scenes to click.
   (Rejected: number-key-only switching — less discoverable.)
2. **Five focused scenes:** World, Sprites, Animation, Picking,
   Primitives (see below). One feature cluster each.
3. **Big-bang full split** — one PR that reorganises `main.rs` into the
   host + all five scene modules (not an incremental trickle).
4. **Prune rarely-used bits while refactoring** (see "Pruning"). Lean the
   demo rather than carry every legacy knob into the new structure.

## Target architecture

```
main.rs            host: window, SceneRenderer, egui, FPS, the scene
                   menu, the shared fly-camera + mouse-look, sky; owns the
                   active `Box<dyn DemoScene>` + the scene registry.
scene_api.rs       the DemoScene trait + SceneCtx + SceneInput + CameraRig.
scenes/
  world.rs         streaming hills + ship + collision-fly + capture.
  sprites.rs       coco field + shoot-to-carve + spinner.
  animation.rs     KFA arm + flame character + voxel clips.
  picking.rs       top-down cursor + dropped markers.
  primitives.rs    image sprites + debug-line gizmos.
collision.rs ship.rs terrain.rs markers.rs kv6_sprite.rs scene.rs
                   reused as-is (mostly by world.rs / animation.rs).
```

### The trait + context

```rust
/// One selectable demo scene. The host owns the window / renderer / egui /
/// camera; a scene owns its world content + per-scene update, input, and
/// overlays.
pub trait DemoScene {
    fn name(&self) -> &str;              // menu label
    fn controls(&self) -> &str;          // HUD help text

    /// Become active: build the world + register sprites/clips/characters.
    /// The host has just reset the renderer's content layers (empty
    /// `set_sprites`) and will set the camera from `start_pose`.
    fn enter(&mut self, ctx: &mut SceneCtx);
    fn start_pose(&self) -> CameraPose;  // where the camera begins

    /// Per-frame: advance animation/streaming, apply movement (collision is
    /// the scene's call — it owns its world), tick clip/character clocks.
    fn update(&mut self, ctx: &mut SceneCtx, dt: f64);

    /// A scene-local key/mouse event (the host already consumed the
    /// universal ones: movement, mouse-look, Tab-menu, F1-HUD, Esc).
    fn on_input(&mut self, ctx: &mut SceneCtx, ev: &SceneInput);

    /// Render this scene's world (`renderer.render` + post overlays via
    /// `draw_lines`/`draw_images`). Does NOT present — the host finishes
    /// the frame with egui/HUD.
    fn render(&mut self, ctx: &mut SceneCtx, frame: &FrameParams);

    /// Optional per-scene `FrameParams` tweaks over the host default
    /// (scan dist, sky, fog, gpu knobs). Default: use the host's.
    fn frame_params<'a>(&self, host: HostFrame<'a>) -> FrameParams<'a> { host.default() }

    /// Drop scene-specific non-renderer state (the host resets the
    /// renderer content layers itself on switch).
    fn exit(&mut self, ctx: &mut SceneCtx) {}
}
```

`SceneCtx<'a>` borrows the host's pieces a scene legitimately needs:
`renderer: &mut SceneRenderer`, `cam: &mut CameraRig` (pos/yaw/pitch +
`Camera` build), `input: &InputState` (held keys + mouse delta),
`size: (u32, u32)`, `engine: &Engine`. The shared fly-camera +
mouse-look live in the host; **movement application** is delegated to
each scene's `update` (World collision-checks against its grid; Picking
ignores movement and sits top-down).

### The host loop (per frame)

1. compute `dt`; apply accumulated mouse-look to `cam` (universal).
2. `active.update(ctx, dt)` — movement + collision + animation + streaming.
3. `active.render(ctx, &frame)` — `renderer.render(&mut world, &cam, &frame)`
   then overlays. No present.
4. host HUD: egui pass combining FPS/pos/backend + `active.controls()` +
   (if open) the **scene-picker panel**; finish via `paint_egui`, else
   `present`.
5. `request_redraw`.

### Scene switching

On a menu pick: `active.exit(ctx)` → host `renderer.set_sprites(&EMPTY)`
(resets static + dynamic + clip + character layers, VCL.4/6) → swap
`active` → `new.enter(ctx)` → set `cam` to `new.start_pose()`. Clean
teardown, no leaked content.

## The five scenes (what moves where)

| Scene | Owns (from today's `main.rs`/modules) | Demonstrates |
|---|---|---|
| **World** | `build_demo` (hills + ship), `StreamingBakeTracker`, `collision::slide_with_collision`, `tick_ship_spin`, scan-dist `+/-`, `T` streaming info, `R` ship spin, `F` capture, `H`? (drop) | scene-graph, chunk streaming, multi-grid, LOD billboards |
| **Sprites** | `build_sprite_set` (coco green/red field), `CarveTarget` + `fire`/`G`, `Spinner` | sprite models, `set_sprites`, dynamic add/remove API, sprite carving |
| **Animation** | `authored_character`/`build_kfa` + KFA path, `flame_character` + `add_character`/`advance_character`, the `ROXLAP_RKC*/KFA_DUMP` tooling | KFA skeletal anim, RKC v3, voxel clips, the attachment runtime |
| **Picking** | `toggle_pick_mode` state (top-down cam, cursor, `placed` markers, `pick_models`), `plane_hit`, `view_ray` | `view_ray`/`pick`, screen→world, sprite placement |
| **Primitives** | `demo_image_sprite`/`I` + `upload_image`, `debug_overlay_lines`/`L` | `draw_images`, `draw_lines` (depth-tested gizmos) |

Each scene module exposes a `pub fn new() -> Box<dyn DemoScene>` (or a
unit struct) and is registered in the host's `scenes: Vec<Box<dyn
DemoScene>>` (the menu iterates it).

## The menu

- `Tab` toggles a `menu_open` flag on the host.
- When open, an egui side panel lists `scenes[i].name()`; a click sets
  `pending_switch = Some(i)`, applied after the egui pass (so the borrow
  of `active` during render is clean).
- The HUD always shows the active scene name + `controls()`; the
  per-scene controls replace today's monolithic key list.

## Pruning (locked decision 4)

Drop or fold:
- `ROXLAP_AUTOFLY` (debug auto-fly) — remove.
- the in-app bench (`B`, 300-frame timing) — remove (FPS HUD + external
  profiling cover it).
- the `H` A/B saved-pose toggle — remove (each scene has a sensible
  `start_pose`; free-fly covers the rest).
- `ROXLAP_NO_SPINNER` / `ROXLAP_GPU_NO_SPRITES` — remove (switch scenes
  instead of env-gating content).
- `lod_billboards_on` toggle — fold into the World scene (keep the
  feature, lose the global flag).

Keep: `ROXLAP_GPU`, `ROXLAP_STATIC` (World), `ROXLAP_RKC` / `RKC_DUMP` /
`KFA_DUMP` (Animation tooling), `ROXLAP_FPS_LOG`,
`ROXLAP_GPU_MIP_SCAN_DIST`, `ROXLAP_SPRITE_GRID` (Sprites).

## Code map (as of 2026-06-26)

- `main.rs` (2296): the `App` god-struct + `redraw`/`resumed`/`tick_camera`
  /`toggle_pick_mode`/`hud_panel`/`fire`/`build_*` — the source to split.
- `scene.rs` (773): `build_demo`, `SceneAndCamera`, `StreamingBakeTracker`,
  `bake_lightmode_1_pub` — reused by World.
- `terrain.rs` / `ship.rs` / `markers.rs` (462/174/180): content builders —
  reused by World.
- `collision.rs` (211): `slide_with_collision` — reused by World.
- `kv6_sprite.rs` (20): `load_coco_kv6` — reused by Sprites/Animation.
- `repro.rs` (2320, `#[cfg(test)]`): regression test — untouched.
- Facade seams the host uses: `SceneRenderer::{render, present, paint_egui,
  set_sprites, resize, view_ray, draw_lines, draw_images, backend,
  adapter_info, set_sky_panorama}`.

## Commit-sized chunks (one PR)

Big-bang, but stage the diff so each commit builds + the demo runs:

1. **DS.0 — scaffold:** `scene_api.rs` (trait + `SceneCtx` + `CameraRig` +
   `SceneInput`), the thin host shell (window/renderer/egui/FPS/camera/menu)
   driving a single placeholder scene. Demo runs (empty scene + menu).
2. **DS.1 — World scene** (the biggest; proves the host loop end-to-end).
3. **DS.2 — Sprites scene.**
4. **DS.3 — Animation scene** (incl. the flame dogfood + `.kfa`/`.rkc`
   tooling).
5. **DS.4 — Picking + Primitives scenes.**
6. **DS.5 — prune** the dropped knobs/keys, delete the dead `App` cruft,
   update the demo `README` + module docs + CHANGELOG (demo-only).

## Tests

- Demo binaries have no render gate in CI; the per-scene **content
  builders** are what's testable: move `flame_clip`/`flame_character`
  tests + any `build_*` invariants into the relevant scene module's
  `#[cfg(test)]`. Keep `repro.rs` + the `character_tests` running.
- Validate by **running** each scene on both backends (CPU + `ROXLAP_GPU=1`):
  switch via the menu, exercise each scene's controls, confirm clean
  teardown (no ghost sprites after a switch).
- Gate: `cargo build -p roxlap-scene-demo` + `cargo clippy` stay green.

## Risks / watch-items

- **R1 borrow choreography.** `active.render` borrows `renderer` +
  `active`; the menu click mutates `active`. Defer the switch until after
  the egui pass (`pending_switch`), and split scene update/render so the
  host never holds two conflicting `&mut`. (Same pattern as
  `advance_character`'s two-phase apply.)
- **R2 teardown completeness.** A scene switch must fully reset renderer
  content — `set_sprites(&EMPTY)` already resets dynamic/clip/character
  layers (VCL.4/6); verify no static images/lines linger (those are
  per-frame, so fine).
- **R3 shared vs per-scene camera.** Picking wants a fixed top-down cam,
  World a collision-fly cam. The host owns mouse-look; movement is the
  scene's `update`. Keep `CameraRig` minimal (pos/yaw/pitch) so scenes
  fully control behaviour.
- **R4 pruning regressions.** Removing the bench / autofly / pose-toggle
  drops capability; documented + intentional. Don't remove `ROXLAP_STATIC`
  / the `.rkc` tooling (still useful).
- **No library change** — purely a demo reorganisation. ~2–3 days.
