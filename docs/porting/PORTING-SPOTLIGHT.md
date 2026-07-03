# roxlap — spot (cone) lights: directional point lights on both backends (Substage SL)

Start-of-stage brief and locked decisions for **spot lights** — a runtime,
CPU-**and**-GPU cone light: a point light with a direction and an angular
cutoff, so it lights only a cone instead of the full sphere. Companion to
[PORTING-DYNLIGHT.md](PORTING-DYNLIGHT.md) (the dynamic-lighting layer this
extends — read it first), [PORTING-DDA.md](PORTING-DDA.md), and
[PORTING-GPU.md](PORTING-GPU.md).

This is a **start-of-stage brief**. A fresh-context session should read it
top to bottom before touching code. The stage tag is **SL**. It targets
**0.21.0** (0.20.0 is already cut, so SL lands under `[Unreleased]` + a
0.20→0.21 workspace bump). Unlike DL's original GPU-only scope, SL is
implemented on **both** backends, because the CPU dynamic-lighting path
already landed (CPU.1/CPU.2) and a spot is a trivial extension of it.

## Why

Dynamic lighting today (stage DL) offers a directional **sun** and
omnidirectional **point lights**. There is no way to author a *cone*: a
torch beam, a car headlight, a stage spotlight, a security lamp, a flashlight
in a cave. demiurg has asked for one. A spotlight is the last of the three
canonical real-time light types (directional / point / spot) and the only
one missing.

## Key enabling facts

- **A spotlight is a point light plus an angular mask.** It shares *every*
  other property with a point light: position, colour, intensity, radius,
  quadratic distance falloff (`point_falloff`), the shadow ray (surface →
  light position), the per-grid transform, and cel banding. The only
  addition is a **cone axis** (unit direction the light shines along) and a
  **soft angular cutoff** (inner half-angle = full brightness, outer
  half-angle = zero). Everything else is reused verbatim. This mirrors the
  DL doc's guiding principle — *reuse the machinery, don't invent a new one*.

- **A point light IS a spotlight with a 180° cone.** Setting the outer
  cutoff to `cos_outer = -1` makes the cone cover the whole sphere, so a
  point light is exactly a degenerate spot. We fold spots into the **same**
  light array / GPU buffer / shader loop and gate the cone factor to `1.0`
  when the light is not a spot — so **point-light-only scenes render
  pixel-for-pixel identically** (the SL regression gate).

- **There are exactly three shading sites, and they already share a loop.**
  The point-light diffuse loop lives in three places, all structurally
  identical:
  - CPU: `roxlap-core/src/dda.rs` `shade_dynamic` (the shared terrain +
    sprite core), the `for p in l.points` loop (~dda.rs:481–512).
  - GPU terrain: `roxlap-gpu/shaders/scene_dda.wgsl` `shade_lit`, the
    `grid_point_lights` loop (~:671–694).
  - GPU sprite: `roxlap-gpu/shaders/sprite_model_dda.wgsl`, the
    `point_lights` loop (~:312–329).
  The cone factor is one extra multiply into the existing `f` in each. No
  new binding, no new buffer, no new pass.

- **One GPU struct feeds both GPU passes.** `GpuPointLight` (lib.rs:943,
  std430) is uploaded by `upload_grid_point_lights` for both the scene pass
  (binding 18) and the sprite pass (binding 15). Growing it once updates
  both.

- **Both transforms already exist.** A spot needs its *position* rotated +
  translated into grid-local space (`grid_local_point`, already used for
  point lights) and its *axis* rotated only (`grid_local_sun_dir` /
  `grid_dir_to_world`, already used for the sun). No new math.

- **Shadows come for free, and spots are cheaper than points in shadow.**
  The shadow ray is unchanged (surface → light position). Better: the cone
  gates out most of the sphere, so we early-out (`cone <= 0 ⇒ continue`)
  *before* the shadow march — a shadow-casting spot marches fewer rays than
  the equivalent point light.

## Locked decisions

Taken with the engine author 2026-07-01:

1. **Both backends.** SL lands on CPU (`shade_dynamic`) and GPU (both
   shaders). The CPU dynamic-lighting path already exists; a spot is a
   one-multiply extension, so the DL-era "GPU-only" carve-out does not
   apply.

2. **Spot is a distinct public type, unified internally.** `SpotLight` is a
   new public type on `roxlap-render` (in tone with the existing
   `DirectionalLight` / `PointLight` split), exposed as `LightRig.spots`.
   Internally every spot is folded into the **same** per-grid point-light
   array / GPU buffer / shader loop — a point light is the `cos_outer == -1`
   degenerate. One code path, two public ergonomic types.

3. **Soft cone edge (inner + outer half-angles).** The cutoff is a
   `smoothstep(cos_outer, cos_inner, cos_angle)`. A hard edge is the
   `outer == inner` special case, so both looks share one code path. Angles
   are authored in **degrees** (half-angles); the facade converts to cosines
   once at pack time.

4. **Spots share the point-light budgets.** They are point lights
   internally, so they consume the same `MAX_POINT_LIGHTS` (32) count budget
   and the same `MAX_SHADOW_CASTERS` (4) shadow-caster budget, with the same
   never-silent demotion warning.

5. **Per-frame, via `FrameParams.lights` (`LightRig`).** No stateful spot
   setters — spots flow through the existing per-frame rig exactly like the
   sun and points. `spots: &[]` (the default) ⇒ exactly today's render.

## The data model

### Public facade type (`roxlap-render/src/light.rs`)

```rust
/// A spot (cone) light — a point light with a direction and a soft angular
/// cutoff, so it lights only a cone. World space. Internally folded into the
/// point-light path (a point light is the 180°-cone degenerate).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpotLight {
    /// World-space position (voxel units).
    pub position: [f32; 3],
    /// Cone **axis** — the unit direction the light shines along (the way the
    /// light travels). Normalized by the backend.
    pub direction: [f32; 3],
    /// Linear RGB, `0..1`.
    pub color: [f32; 3],
    pub intensity: f32,
    /// Hard distance cutoff (voxel units); past it the light is zero.
    pub radius: f32,
    /// Inner half-angle in **degrees**: within it, full angular brightness.
    pub inner_angle_deg: f32,
    /// Outer half-angle in **degrees**: past it, zero. Between the two the
    /// cone soft-falls off (`smoothstep`). `outer == inner` ⇒ a hard edge.
    /// Clamped `0 <= inner <= outer <= 180`.
    pub outer_angle_deg: f32,
    /// Whether this spot casts a stylized hard shadow (shares the
    /// `MAX_SHADOW_CASTERS` budget with the sun + point lights).
    pub casts_shadow: bool,
}
```

`LightRig` gains `pub spots: &'a [SpotLight]` (default `&[]`). Nothing else
in the rig changes.

### Internal light structs grow (shared point/spot layout)

`CpuPointLight` (dda.rs:88), `GpuPointLight` (lib.rs:943) and the `PointLight`
structs in both `.wgsl` files each gain:

```
spot_dir:  [f32; 3]   // cone axis, grid-local (scene) or world (sprite)
cos_inner: f32
cos_outer: f32        // == -1.0 for a pure point light ⇒ cone factor ≡ 1
```

`GpuPointLight` grows **48 → 64 bytes** — four `vec4<f32>`, std430-clean:

```
vec4: pos.xyz,       radius
vec4: color.rgb,     intensity
vec4: spot_dir.xyz,  cos_outer
vec4: cos_inner,     casts_shadow(u32 bitcast), _pad, _pad
```

(The current three `_pad` words hold 12 bytes; a spot needs 5 floats = 20
bytes, so the struct grows honestly to the next 16-byte multiple.) Growing
the **one** struct updates both GPU passes (`upload_grid_point_lights` at
lib.rs:955 and the sprite `sprite_pts` path at lib.rs:2764).

## The cone factor (identical in all three sites)

`l` in every loop is already the unit vector **from the surface to the
light** (`d3 / dist`). The cone axis `spot_dir` points **away** from the
light. The angle between the light→surface ray and the axis has cosine
`dot(-l, spot_dir)`:

```wgsl
let cd = dot(-l, spot_dir);                       // 1 = dead on axis
// A pure point light (cos_outer == -1) skips the mask and stays == 1.0,
// guaranteeing byte-identical output for non-spot scenes:
var cone = 1.0;
if (cos_outer > -0.999) {
    cone = smoothstep(cos_outer, cos_inner, cd);  // 0 outside, 1 inside inner
}
if (cone <= 0.0) { continue; }                    // early-out before shadow march
var f = ndl * atten * cone * sh;
if (styled) { f = cel_band(f, bands); }           // cone folds in before cel, like atten
```

The CPU `shade_dynamic` gets the scalar equivalent (a `smoothstep` helper +
the same `cos_outer > -0.999` guard + `continue`).

## Renderer / facade changes

- **`sync_lights`** (`roxlap-render/src/gpu.rs:1456`) and the **CPU light
  block** (`roxlap-render/src/cpu.rs:1102`): in the same per-grid loop that
  already builds the point array, append `rig.spots`, setting `spot_dir`
  (axis via `grid_local_sun_dir` for the scene pass / raw world for the
  sprite pass), `cos_inner = cos(inner_deg)`, `cos_outer = cos(outer_deg)`,
  and the transformed `position`. Existing point lights get
  `cos_outer = -1.0`.
- The **shadow-caster budget** loop (cpu.rs:1119–1149) folds spots into the
  same `budget` accounting *after* the sun and points; the same demotion
  warning covers them.
- The **world-space copies** for the sprite pass (gpu.rs:1503–1519) append
  the spots too, with the axis left in world space.

## Sub-substage roadmap

- **SL.0** — grow `CpuPointLight` / `GpuPointLight` / both `.wgsl`
  `PointLight` structs + the ABI (buffer stride, uniform, bindings unchanged
  in count). Thread `spot_dir` / `cos_inner` / `cos_outer` through pack +
  upload; existing lights set `cos_outer = -1.0`; the cone factor is
  hard-wired to `1.0` (not yet read from the fields). **Gate: sun + point
  goldens byte-identical; GPU diff-harness zero-diff.**
- **SL.1** — the cone factor in all three shading sites (CPU
  `shade_dynamic`, `scene_dda.wgsl`, `sprite_model_dda.wgsl`) + the
  `cone <= 0` early-out. Spots now actually cone. Point lights
  (`cos_outer == -1`) still identical.
- **SL.2** — facade: the `SpotLight` public type, `LightRig.spots`, the
  degrees→cosines conversion, and the fold into the per-grid point array in
  `sync_lights` + the CPU block, sharing the caster budget.
- **SL.3** — tests: a CPU unit test in `dda.rs` (`shade_dynamic`: a sample
  inside the cone is lit, one outside is dark; `outer == inner` gives a hard
  edge; a `cos_outer == -1` spot equals the matching point light) + a GPU
  headless diff-harness case (spot vs point with `cos_outer == -1` ⇒
  identical).
- **SL.4** — demo + docs: a spot "searchlight" in the DL demo scene + a HUD
  control (position / cone angles / colour), CHANGELOG under
  `[Unreleased]`, per-crate rustdoc, and the 0.20→0.21 workspace bump.

## Risks

- **std430 stride.** Growing `GpuPointLight` to 64 B must keep the `#[repr(C)]`
  layout matching the WGSL `PointLight` exactly (four `vec4`). Validate with
  a `size_of` assert + the SL.0 zero-diff gate. This is the single most
  likely place to introduce a silent corruption.
- **`bitcast` of `casts_shadow`.** Packed as a `u32` in the last `vec4`; read
  it with `bitcast<u32>` in WGSL, not a float compare.
- **CPU cost.** Spots add a `smoothstep` + a dot per light per lit pixel on
  the already-slow CPU fallback — negligible next to the shadow march, and
  the `cone <= 0` early-out *saves* shadow rays. No action needed, but note
  it in the CPU perf memo.
- **Sprite-pass axis space.** The scene pass wants `spot_dir` grid-local; the
  sprite pass wants it world-space (sprites shade in world space). Two
  different transforms of the same authored axis — mirror exactly how the
  sun dir is already double-computed (`grid_sun_dirs` vs `world_sun_dir`).

## Validation (every sub-substage)

1. `cargo build` + `cargo clippy` clean across the workspace.
2. The existing DL goldens (sun + point) stay byte-identical through SL.0–SL.2
   (a spot-free rig must not move a pixel).
3. The GPU headless diff-harness: a `cos_outer == -1` spot equals the
   matching point light bit-for-bit.
4. The new SL.3 cone unit test passes on CPU; the demo spot visibly cones on
   both backends (`ROXLAP_GPU=0/1`).
