# roxlap — dynamic lighting: runtime sun + point lights + stylized voxel shadows (Substage DL)

Start-of-stage brief and locked decisions for **dynamic lighting** — a
runtime, GPU-only lighting model layered on top of the existing baked
ambient byte: one directional **sun** (colored), several colored
**point lights**, and **stylized hard voxel shadows** cast by the sun and
a small number of chosen point lights (the rest are shadowless). Companion
to [PORTING-DDA.md](PORTING-DDA.md) (the per-pixel 3D-DDA renderer this
shades on top of), [PORTING-GPU.md](PORTING-GPU.md) (the compute-shader
raymarcher we extend), [PORTING-TRANSPARENCY.md](PORTING-TRANSPARENCY.md)
(the per-voxel `Material` layer, orthogonal to lighting), and
[PORTING-SPRITE-API.md](PORTING-SPRITE-API.md) (sprite instances, which
get lit in DL.4).

A follow-on stage **AO** (ambient occlusion baked into the same ambient
byte) is designed at the end of this doc and targets the next minor after
DL.

This is a **start-of-stage brief**. A fresh-context session should read it
top to bottom before touching code. The stage tag is **DL**. It is
**GPU-only** (the CPU rasterizer is deliberately left unlit — see Locked
decision #1). It targets **0.18.0** (0.17.0 was already cut for the alpha
fix, so DL lands under `[Unreleased]` + a 0.17→0.18 workspace bump).

## Why

All lighting in roxlap today is **baked, static, and direction-frozen**.

- A CPU pre-pass (`roxlap-core/src/world_lighting.rs:360`
  `update_lighting`, `:737` `compute_brightness`) writes a single `u8`
  **brightness** into the high byte of every voxel's `0x80RRGGBB` color
  word. lightmode 1 bakes a hardcoded directional sun
  (`i = (n.y*0.5 + n.z)*64 + 103.5`); lightmode 2 bakes point lights from
  `Engine::lights`. The demo bakes lightmode 1 with an **empty** light
  list (`roxlap-scene-demo/src/scene.rs:360` `bake_lightmode_1`).
- Both renderers only **multiply that byte in**. GPU:
  `scene_dda.wgsl:272-280` `voxel_color_in` — `brightness = max(0, a -
  face_shade) * (1/128)`, then `rgb * brightness/255`. There is **no
  light position, no light color, no normal, no shadow** anywhere in the
  shader. The only per-frame visual knobs are sky, fog, and the per-face
  `side_shades` constant.
- Per-voxel **normals exist only transiently** inside `EstNormCache`
  during the bake (5×5×5 occupancy gradient, `world_lighting.rs:298`
  `estnorm`) and are discarded — the GPU never sees a normal.

So the sun cannot move, lights cannot be colored or animated, nothing
casts a shadow, and a torch dropped into a cave does nothing. We want a
first-class **runtime** lighting layer so demiurg can author day/night,
colored spell glows, muzzle flashes, and dramatic shadows — and preview
them live through the engine.

## Key enabling facts

- **The DDA already knows the surface normal — for free.** At every
  terrain hit the marcher knows which face was crossed: `hit_axis ∈
  {0,1,2}` plus the sign of `ray_dir` on that axis. That is an
  axis-aligned face normal with **zero extra work** — it is already
  computed today to pick the `side_shade` (`scene_dda.wgsl:229-237`
  `side_shade_for`, used at the hit site `:507-530`). N·L lighting needs
  exactly this. No per-voxel normal storage required for terrain.

- **Shadows are just another DDA march.** The GPU is already a 3D-DDA
  raymarcher with a ready occlusion test (`scene_dda.wgsl:217-222`
  `voxel_solid_in`, traversal `march_grid` `:410-595`). A shadow ray is
  the same traversal from the hit point toward a light, returning on the
  first solid voxel. We reuse the machinery; we do not invent a new one.

- **The hit/shade site is one well-isolated block.** `scene_dda.wgsl:507-530`
  is the single place a terrain surface color is produced. All new
  shading lands there, behind an "is dynamic lighting enabled" branch.
  When no lights are set the old `voxel_color_in` path runs verbatim —
  the **byte-identical regression gate** holds trivially.

- **Per-grid camera plumbing is the template for per-grid lights.** The
  world camera is already transformed into each grid's local space CPU-side
  and uploaded as a runtime-sized storage buffer
  (`SceneDdaPerGridCamera` `lib.rs:766-790`, `upload_grid_cameras`
  `lib.rs:2055`, consumed `scene_dda.wgsl:598-628`). Lights are
  transformed and uploaded **exactly the same way** — positions as points,
  the sun direction as a vector — so the shader always works in grid-local
  space (where the ray already lives) and rigid transforms preserve
  distances/dot-products.

- **Sprites already carry true per-voxel normals.** The sprite pass
  (`sprite_model_dda.wgsl`, `dirs` binding 10, `model_color` `:129-142`)
  has real surface normals and a per-voxel-normal modulation table. Once
  terrain is lit, sprites can be lit *more* accurately than terrain (true
  normals, not face normals) in DL.4.

- **The ambient byte is now free to mean "ambient".** Per the locked
  decision below, the existing baked brightness byte is reinterpreted as
  an **ambient / AO fill term**, and the AO follow-on rebakes it as true
  occlusion. Direct/directional light becomes the dynamic layer.

## Locked decisions

Taken with the engine author 2026-06-28:

1. **GPU-only.** Dynamic lighting and shadows are implemented only in the
   GPU backend (`roxlap-gpu` + `scene_dda.wgsl` / `sprite_model_dda.wgsl`).
   The CPU rasterizer is already the slow fallback; per-pixel shadow
   marches would make it unusable. The CPU path keeps multiplying the
   baked ambient byte and **ignores** any configured lights — documented,
   not silent.

2. **The baked brightness byte becomes the ambient/AO channel.** The
   high byte stops representing a static sun and is reinterpreted as a
   per-voxel **ambient** multiplier. Direct light (sun + points) is the
   runtime layer, composited on top: `out = albedo*ambient + Σ direct`.
   The AO follow-on (below) rebakes this byte as occlusion. During DL the
   existing baked content is treated as the ambient term verbatim (interim
   soft fill); the bake content itself changes only in the AO stage.

3. **Lighting is per-frame, via `FrameParams`.** Lights flow through the
   existing per-frame config struct (where sky/fog/side_shades already
   live — there are deliberately no stateful lighting setters on
   `SceneRenderer`). The sun moves, lights animate, day/night cycles — all
   by mutating the next frame's `FrameParams`. No re-bake, no scene
   mutation. Default `None` ⇒ exactly today's render.

4. **Shadows are stylized hard voxel shadows with a strength floor.** One
   shadow ray per shadow-casting light per primary hit; binary occlusion;
   the shadow does **not** go to absolute black — a tunable
   `shadow_strength ∈ [0,1]` controls how much of that light is removed in
   shadow (1.0 = full black, default ~0.7). Blocky edges are embraced as
   the voxel aesthetic. No penumbra/PCSS (rejected as too expensive on the
   per-pixel DDA budget).

5. **Bounded counts; shadow casters are a strict subset.** `MAX_POINT_LIGHTS`
   (default 32) point lights total; at most `MAX_SHADOW_CASTERS` (default
   4, including the sun) actually cast shadows. Non-shadow lights are
   cheap (N·L + attenuation only). Over-cap lights are dropped with a
   `log()` warning — **no silent truncation** (standing discipline). Each
   `PointLight` carries its own `casts_shadow` flag; the backend keeps the
   first N flagged casters and demotes the rest to shadowless with a warn.

6. **Intra-grid shadows in DL.3; cross-grid shadows deferred.** A shadow
   ray marches only within the hit voxel's own grid. Cross-grid shadows
   (a ship casting onto terrain) are a known limitation, deferred behind a
   noted risk — the per-grid-local light transform makes single-grid the
   natural unit. Self-shadow acne is killed by biasing the shadow-ray
   origin ~1.5 voxels along the face normal.

7. **Crate placement.** New light types live in `roxlap-render` next to
   `FrameParams` (they are render config, not a serialized voxel format —
   unlike `Material`, which lives in `roxlap-formats`). GPU packing/upload
   lives in `roxlap-gpu`. The facade (`roxlap-render/src/lib.rs`
   `SceneRenderer`) is duck-typed across `cpu.rs`/`gpu.rs` by hand — no
   backend trait. If light rigs ever need to serialize into scene
   snapshots, the types move to `roxlap-formats` then (not now).

## The data model

```rust
// roxlap-render/src/light.rs  (new), re-exported from lib.rs

/// One directional light (the sun). World space. At most one per frame.
#[derive(Clone, Copy, Debug)]
pub struct DirectionalLight {
    /// Direction the light TRAVELS (from sun toward scene), world space,
    /// normalized. N·L uses the negation.
    pub direction: [f32; 3],
    /// Linear RGB, 0..1 (may exceed 1 for HDR-ish punch; clamped at output).
    pub color: [f32; 3],
    pub intensity: f32,
    pub casts_shadow: bool,
}

/// A colored point light. World space. Hard radius cutoff (cube/quadratic).
#[derive(Clone, Copy, Debug)]
pub struct PointLight {
    pub position: [f32; 3], // world
    pub color: [f32; 3],    // linear RGB
    pub intensity: f32,
    /// Hard cutoff: contributes nothing beyond `radius` (in world units).
    pub radius: f32,
    pub casts_shadow: bool,
}

/// The whole per-frame light environment. Borrowed into FrameParams.
#[derive(Clone, Copy, Debug, Default)]
pub struct LightRig<'a> {
    pub sun: Option<DirectionalLight>,
    pub points: &'a [PointLight],
    /// Multiplier applied to the baked ambient byte (global ambient tint /
    /// level). Default [1.0; 3] ⇒ ambient byte used as-is.
    pub ambient: [f32; 3],
    /// Stylized-shadow knobs (locked decision #4/#5).
    pub shadow_strength: f32,   // 0 = no shadow, 1 = black. default 0.7
    pub shadow_bias_voxels: f32, // default ~1.5
    pub shadow_max_dist: f32,   // sun shadow ray length cap (world units)
}

// roxlap-render/src/lib.rs  (FrameParams, extend)
pub struct FrameParams<'a> {
    // ...existing: sky_color, sky, fog_color, fog_max_scan_dist, side_shades...
    pub lights: Option<LightRig<'a>>, // None ⇒ today's render, byte-identical
}
```

GPU side (`roxlap-gpu`), mirroring the existing material/camera plumbing:

```rust
// roxlap-gpu/src/lib.rs
const MAX_POINT_LIGHTS: usize = 32;
const MAX_SHADOW_CASTERS: usize = 4; // sun + up to 3 points, by default

#[repr(C)] struct GpuPointLight {   // per (grid, light), uploaded local-space
    pos: [f32; 3], radius: f32,
    color: [f32; 3], intensity: f32,
    casts_shadow: u32, _pad: [u32; 3],
}
// Extend SceneDdaUniform (lib.rs:794+, packed :2056-2087) with:
//   sun_dir_local handled per-grid (see below), sun_color, sun_intensity,
//   sun_flags (enabled | casts_shadow), point_light_count,
//   ambient_color, shadow_strength, shadow_bias, shadow_max_dist.
// One new runtime-sized storage buffer (mirrors grid_cameras @binding 15):
//   @binding 18  grid_point_lights : array<GpuPointLight>  // grid_idx*count + i
// NOTE (landed in DL.0): the device storage-buffer limit (16/stage) was
// already saturated at 15, so the per-grid sun direction is NOT a separate
// buffer — it rides in `SceneDdaPerGridCamera.sun_dir` (binding 15). One
// new storage buffer total (point lights), keeping scene_dda at 16.
```

`gpu.rs` (`sync_sky_and_fog` sibling, ~`:813`) gains `sync_lights(frame)`:
for each registered grid, transform every point-light position and the sun
direction into that grid's local space (reuse `world_camera_to_grid_local`'s
rotation/translation), pack, and upload. Point lights past `MAX_POINT_LIGHTS`
and shadow-casters past `MAX_SHADOW_CASTERS` are dropped with a `log::warn!`.

## Renderer changes

**CPU.** None functional. The CPU backend ignores `frame.lights` (locked
decision #1) and keeps the existing baked-ambient multiply. A one-line
`log::debug!` notes lights were supplied but skipped. This keeps the
**CPU byte-identical regression gate** intact for free.

**GPU — terrain.** All work is at the hit/shade site (`scene_dda.wgsl:507-530`).
When `point_light_count==0 && sun disabled` the original `voxel_color_in`
path runs unchanged (regression gate). Otherwise:

```wgsl
let albedo  = unpack_albedo(packed);                 // rgb/255, NO brightness fold
let ambient = ambient_byte(packed, face_shade);      // old brightness byte, repurposed
let N       = face_normal(hit_axis, ray_dir);        // axis normal, grid-local
var lit     = albedo * u.ambient_color * ambient;

if (u.sun_flags & SUN_ENABLED) != 0u {
    let L   = -grid_sun_dir[g].xyz;                  // toward the sun
    let ndl = max(0.0, dot(N, L));
    var sh  = 1.0;
    if (u.sun_flags & SUN_SHADOW) != 0u {
        sh = shadow_factor(g, slot_id, hit_local, L, u.shadow_max_dist);
    }
    lit += albedo * u.sun_color * u.sun_intensity * ndl * sh;
}
for (var i=0u; i<u.point_light_count; i=i+1u) {
    let pl = grid_point_lights[g*MAX + i];
    let d3 = pl.pos - hit_local; let d = length(d3);
    if (d < pl.radius) {
        let L = d3 / d;
        let ndl = max(0.0, dot(N, L));
        let atten = falloff(d, pl.radius);           // (1 - d/radius)^2, hard cutoff
        var sh = 1.0;
        if (pl.casts_shadow != 0u) { sh = shadow_factor(g, slot_id, hit_local, L, d); }
        lit += albedo * pl.color * pl.intensity * ndl * atten * sh;
    }
}
out.color = apply_fog(lit, t_hit);                   // fog unchanged (:359-363)
```

`shadow_factor` (new): bias `origin = hit_local + N*shadow_bias_voxels`,
3D-DDA toward `L` capped at `max_t` and `shadow_max_steps`, return
`(1 - shadow_strength)` on first `voxel_solid_in` hit before `max_t`, else
`1.0`. Intra-grid only (locked decision #6). It is a thin reuse of the
existing inner-voxel DDA loop, not a copy of `march_grid`.

**GPU — sprites (DL.4).** `sprite_model_dda.wgsl` applies the same
sun+point math but with the **true per-voxel normal** (`dirs`, binding 10)
instead of a face normal, giving smoother sprite shading. Sprite shadow
rays march the terrain grid (so characters are shadowed by terrain);
sprite-casts-onto-terrain is deferred with cross-grid shadows.

**Parity.** No bit-exact gate — dynamic lighting is clean-room and
GPU-only (there is nothing to match the CPU against). The headline
regression gate is **`lights==None` ⇒ output byte-identical to pre-DL**.
Visual correctness is image-regression-with-tolerance + manual dogf0od in
the demo (headless CI has no display).

## Authoring API (facade)

No new stateful setters — lighting is per-frame (locked decision #3). The
surface is just the new `FrameParams.lights` field plus the re-exported
`DirectionalLight` / `PointLight` / `LightRig` types from
`roxlap-render`. A host builds a `LightRig` each frame and assigns it:

```rust
let sun = DirectionalLight {
    direction: [-0.3, -0.5, -0.8], color: [1.0, 0.95, 0.85],
    intensity: 1.0, casts_shadow: true,
};
let torches = [PointLight { position: muzzle, color: [1.0,0.6,0.2],
                            intensity: 3.0, radius: 40.0, casts_shadow: true }];
let frame = FrameParams {
    lights: Some(LightRig { sun: Some(sun), points: &torches,
                            ambient: [0.6,0.65,0.8], shadow_strength: 0.7,
                            ..Default::default() }),
    ..frame
};
renderer.render(&mut scene, camera, &frame);
```

## Sub-substage roadmap

| Stage | Scope | Gate |
|---|---|---|
| **DL.0** ✅ | Types + plumbing, **no render change**. `light.rs` (`DirectionalLight`/`PointLight`/`LightRig`), `FrameParams.lights`, GPU uniform fields + the per-grid point-light storage buffer + bind-group wiring + `sync_lights` upload. Shader receives the buffers but ignores them. | **Done** — builds + clippy clean (all targets); 23 test binaries green incl. the headless scene-DDA renders (byte-identical, `lights==None`). **Note:** the device `max_storage_buffers_per_shader_stage` is 16 and scene_dda was already at 15 — so the per-grid **sun direction rides in `PerGridCamera.sun_dir` (binding 15)**, and only point lights got a new storage buffer (binding 18). Bindings 18 + the camera `sun_dir` are bound but unread until DL.1. |
| **DL.1** ✅ | **GPU directional sun**, diffuse, no shadow. Albedo/ambient split + `face_normal` + N·L in `shade_lit` (scene_dda.wgsl); ambient byte × `ambient_color`; sun via `grid_cameras[g].sun_dir`. Lit path gated on `sun_flags` bit 2 (`SceneLights.enabled`); off ⇒ `voxel_color_in` verbatim. Per-grid transform extracted to `grid_local_sun_dir`/`grid_local_point` (facade); light packing shared via `pack_scene_lights` (surface + headless). | **Done** — builds + clippy clean; 3 transform unit tests (sign/rotation/translation) + headless GPU test `scene_dda_sun_lights_floor_by_facing` (sun-above > baked > … ; facing beats back-facing). Off-path byte-identical (existing headless renders unchanged). |
| **DL.2** ✅ | **GPU point lights**, diffuse, no shadow. Shader loop in `shade_lit` over `grid_point_lights[g*count+i]`: N·L × `point_falloff` (smooth quadratic, hard radius cut), needs the grid-local `hit_pos` (= `ray_origin + t_hit·ray_dir`). Per-grid position transform (`grid_local_point`) + `MAX_POINT_LIGHTS` cap + over-cap `warn` already landed in DL.0/DL.1 (`pack_scene_lights`). | **Done** — builds + clippy clean; headless GPU test `scene_dda_point_light_brightens_by_distance_and_facing` (near>baked, near>far falloff, back-facing≈no light). Off-path byte-identical. |
| **DL.3** ✅ | **GPU stylized hard shadows**. `shadow_occluded(g, origin, dir, max_t)` — dedicated intra-grid DDA (outer chunk-skip + inner mip-0 voxel walk, bounded by `max_outer_steps` + `shadow_max_steps`). Sun + per-light `casts_shadow`; shadow-ray origin biased `n*shadow_bias` (anti-acne); in-shadow factor `1-shadow_strength` (`ambient_color.w`). `MAX_SHADOW_CASTERS` cap in `pack_scene_lights` (sun first, excess point casters demoted + `warn`). Intra-grid only (cross-grid deferred, R5). | **Done** — builds + clippy clean; headless GPU test `scene_dda_sun_shadow_darkens_occluded_floor` (wall casts sun shadow on visible floor; shadows-on darker than shadows-off). Off-path byte-identical. |
| **DL.4** ✅ | **Sprites lit** by sun + point lights using **true per-voxel normals** (voxlap `univec[256]`, binding 14; `dir`→model normal→world via `transpose(inv_rot)`). World-space lights for the sprite pass (`SceneLights.world_sun_dir`/`world_points`, sprite uniform light fields, world point-light buffer binding 15). `shade_sprite_lit` in `march_instance` gated on `sun_flags` bit 2; off ⇒ `model_color` unchanged. **Opaque sprites only**; sprite shadows + lit translucent sprites deferred (R5/R6). | **Done** — builds + clippy clean; `wgsl_shaders_validate` (naga) covers the sprite shader; off-path unchanged. Sprite **visual** is demo-verified (no headless sprite renderer). |
| **DL.5** ✅ | **"Lighting" demo scene** (`scenes/lighting.rs`): sweeping warm sun (casts shadows) + 3 orbiting coloured point lights (2 shadow-casting, 1 not) over a pillared floor + monument; `P`/`K`/`L` toggles. Registered in `host.rs` + `scenes/mod.rs`. README scenes table + CHANGELOG `[Unreleased]` entry; workspace + internal-dep version bump 0.17→0.18 (0.17.0 was already released for the alpha fix). | **Done** — build + clippy clean; full workspace tests green; demo GPU visual user-verified (headless CI has no display). WASD free-fly via `ctx.cam.fly_free`. |
| **DL.6** ✅ | **Stylized lighting** (retro, terrain) — smooth Phong reads "generic" and flattens the voxel identity; this adds **cel banding (A)** + **gradient-map ramp (C)**. `LightRig.bands` (0 = smooth) + `LightRig.shadow_tint`. In `shade_lit`: when `style_bands > 0`, the sun key + each point factor quantize via `cel_band` (round to `bands+1` levels) and the banded sun key gradient-maps `shadow_tint` (cool, unlit) → sun colour (warm, lit) — hue-shifted terraces, shadows tint cool instead of darkening. `bands == 0` keeps the smooth DL.1–3 path byte-for-byte. **Flat per-voxel (option D, folded in):** the stylized path samples lighting at the **voxel centre** (`select(hit_pos, vox_center, styled)`) not the per-pixel hit — point lights + shadow edges become flat per voxel-face (blocky pools, the retro fix for the "smooth radial gradient" tell), at O(1) (no extra pass / buffer). Demo `J` toggles stylized/smooth, `[`/`]` adjust band count live (default 6). (Sprites still smooth — mirror later if wanted.) | **Done** — build + clippy clean; headless `scene_dda_cel_banding_terraces_sun` (two N·L collapse to one band ⇔ stylized equal, smooth differs); naga validates. **Watch:** WGSL `vec3<u32>` pad has 16-byte align (≠ Rust `[u32;3]`) → use 3 scalar pads to keep the uniform size matched. |

| **CPU.1** ✅ | **CPU diffuse stylized lighting** — revises locked decision #1: the CPU backend now lights too (sun + point lights + cel + ramp + flat-per-voxel), **no shadows** (the per-pixel shadow march stays GPU-only). `CpuLights`/`CpuPointLight` + `DdaEnv.lights` in roxlap-core; `shade_lit_cpu` (mirror of GPU `shade_lit`, samples at the voxel centre) at the `cast_ray` hit, gated on `env.lights.enabled` (else the baked `shade`, byte-identical). `render_scene_composed{_with_materials,_scissored}` gain a world-space `CpuLights` param, transformed per grid via `grid_local_lights` (mirrors `world_camera_to_grid_local`); `cpu.rs` builds it from `FrameParams.lights`. Diffuse is ALU-only ⇒ ~free (the CPU path is bandwidth-bound). | **Done** — build + clippy clean; unit tests `shade_lit_cpu_sun_lights_by_facing` / `shade_lit_cpu_cel_terraces_sun` / `cel_band_quantizes_and_collapses`; `lights==None` ⇒ baked path byte-identical (oracle/CPU goldens unaffected). |
| **CPU.2** ✅ | **CPU hard shadows** (revises locked decision #1 further): the CPU backend now casts sun + point-light shadows too, so both backends are on full parity. A `ShadowTester` trait + `SamplerShadow` (in `dda.rs`) march a 3D-DDA toward the light, reusing the render `Sampler`'s occupancy (`sampler.hit().is_some()` — a shadow ray is blocked by the same surfaces the camera sees) and the same `[lo_c, hi_c)` voxel-box bounds, capped by `shadow_max_dist` / the light distance + `SHADOW_MAX_STEPS`. `shade_dynamic` gained an `Option<&mut dyn ShadowTester>`: the sun key and each flagged point factor scale by `1 - shadow_strength` when occluded (bias `shadow_bias` off the surface normal kills acne). `CpuLights`/`CpuPointLight` gained `sun_casts_shadow` / `casts_shadow` / `shadow_strength` / `shadow_bias` / `shadow_max_dist`; `cpu.rs` fills them from the rig and mirrors the GPU `MAX_SHADOW_CASTERS` cap (sun first, excess point casters demoted with an `eprintln` warning — never silent). Only built when a caster is actually flagged **and** `shadow_strength > 0`, so the no-shadow rig stays march-free; sprites pass `None` (no sprite shadows, matching the GPU). The march is the slow fallback's slowest path but correct. | **Done** — build + clippy clean; `shade_dynamic_sun_shadow_darkens` (the shadow math, mock tester) + `sampler_shadow_march_casts_sun_shadow` (a wall on a floor under a grazing sun darkens the scene >2%); `lights==None` ⇒ baked path byte-identical. |
| **CPU-sprites** ✅ | **CPU stylized sprites + clips** — the DL.7 look on the CPU sprite/clip path (`dda_sprite.rs`). Lighting core extracted to `dda::shade_dynamic(albedo, ao, n, sample, l)` (shared with terrain); `cast_local` now also returns the hit's model-local **face normal** + cell (CPU `SpriteDense` has no per-voxel normals → use the DDA face normal, flat per voxel, consistent retro). The opaque branch in `draw_sprite_dense_shaded` rotates the local normal + voxel centre to world via the instance basis (s,h,f) and calls `shade_dynamic`; `SpriteShade` gained `lights: CpuLights` (fed from `cpu.rs`'s world rig). Disabled ⇒ baked `shade`, byte-identical. No sprite shadows. | **Done** — build + clippy clean; `cast_local_reports_face_normal` unit test + existing sprite render tests unchanged; visual via the "Lighting" demo on the CPU backend. |

| **DL.7** ✅ | **Stylized sprites + clips** — extend DL.6 (cel + ramp + flat-per-voxel) to the GPU sprite/clip pass (`sprite_model_dda.wgsl` `shade_sprite_lit`): `cel_band` on the sun key + point factors; banded sun key gradient-maps `shadow_tint` → sun colour; flat per voxel samples at the **world voxel centre** (`inst.pos + transpose(inv_rot)·((p+0.5−pivot)·voxel_world_size)`). Sprite uniform gains `shadow_tint` + `style_bands` (fed from `SceneLights`). Clips share the path (a clip frame is a sprite-model instance) ⇒ free. `style_bands == 0` ⇒ the DL.4 smooth path. No sprite shadows. Demo "Lighting" gains a static voxel sphere + a pulsing voxel clip to show it. **Normal source (fixed):** the surface normal is the **DDA hit-face normal** (`march_instance` tracks the crossed axis), NOT the per-voxel `univec[dir]` table — procedural/clip kv6 may not populate `dir`, which gave zero/garbage normals (unlit static sprites, edge blowouts, no per-face colour). Face normals match terrain + CPU and the flat-per-voxel look. The univec buffer/binding (14) was removed. | **Done** — build + clippy clean; naga validates the sprite shader; sprite/clip **visual** demo-verified (no headless sprite renderer). |

**Stage DL complete.** All sub-substages landed (DL.0–DL.7) + **CPU.1** (CPU diffuse lighting) + **CPU.2** (CPU hard shadows — both backends now at full parity).

**Macro-stage XS (cross-scene shadows + lit translucent sprites)** — closing the DL deferred list, both backends:
- **XS.0** ✅ — **Lit translucent sprite layers.** The sprite/clip accumulate paths shaded layers with the flat baked colour; now each layer is lit (`shade_dynamic` on the CPU `cast_local_layers`, `shade_sprite_lit` in the GPU `march_instance_layers`) via the model-local face normal, matching opaque sprites + translucent terrain (already lit on both backends). Disabled rig ⇒ byte-identical. Tests: `translucent_sprite_layers_are_lit` (CPU); naga validates the GPU shader.
- **XS.1** ✅ — **Cross-grid hard shadows, CPU.** A shadow ray now tests the whole scene, not just the hit grid. New `roxlap_core::WorldOccluder` trait + `DdaEnv::world_shadow` (a `WorldShadowCtx`: the occluder + the hit grid's local→world transform; `WorldShadow` lifts the grid-local ray to world). `roxlap-scene::occluder::SceneOccluder` marches every grid (`Grid::voxel_solid`, mip-0, AABB-clipped) in world space. Built once per frame when a caster is active; the composed render loop split into a `&mut` cache-prep phase + an immutable render phase so the occluder coexists with the render borrow. Test: `cross_grid_sun_shadow_darkens_other_grid`.
- **XS.2** ✅ — **CPU sprite shadows (cast + receive).** `SpriteOccluder` (decoded `SpriteDense` volumes + world poses, `WorldOccluder`) marches each sprite's dense occupancy for a world ray. `CompositeOccluder` ORs grid + sprite occluders. cpu.rs builds the sprite occluder (static + KFA + dynamic/clip frames) when a caster is active, passes it into the terrain render (sprites **cast** onto terrain — new `sprite_occluder` param on `render_scene_composed_with_materials`/`_scissored`, composited with the per-frame grid occluder), and after the terrain render builds the grid occluder + composites for the sprite pass so sprites **receive** (their `SpriteShade.shadow` queries it; opaque + translucent layers, world-space identity `WorldShadowCtx`). Tests: `sprite_occluder_blocks_ray_through_volume`, `sprite_receives_hard_shadow`.
- **XS.3** ✅ — **GPU cross-grid shadows.** `scene_dda.wgsl`: each grid's world transform (origin + local→world rotation columns) packed into `PerGridCamera` (binding 15 — no new buffer; the 16-buffer limit is saturated). New `shadow_occluded_world(origin_w, dir_w, max_t)` loops every grid, transforms the world ray into each grid's local frame (`world_to_grid_local`) and runs the per-grid `shadow_occluded`; `shade_lit` lifts its sun + point shadow rays to world (`grid_local_to_world`/`grid_dir_to_world`) and calls it. Rust: `SceneDdaPerGridCamera` gains `world_origin`/`rot0/1/2`; new pub `GridWorldTransform` + `render_scene`/headless `render_with_transforms` take a parallel `&[GridWorldTransform]` (gpu.rs builds them from `grid.transform`). Identity ⇒ prior intra-grid shadows byte-identical. Test: headless `scene_dda_cross_grid_sun_shadow` (grid B's wall, at a world offset, shadows grid A's floor).
- **XS.4** (in progress) — **per-sprite shadow flags + GPU sprite shadows.** The GPU side hits a hard wall: full cross-pass occupancy needs >16 storage buffers per pass (scene pass is **16/16 saturated**; sprite pass 14/16 + ~8 terrain-occupancy buffers = 22), and the device cap is held at `min(adapter, 16)` for portability. Decision (with author): **raise the cap + gate on capability** (capable devices get GPU sprite shadows; others fall back gracefully). Staged:
  - **XS.4.0** ✅ — per-sprite `casts_shadow`/`receives_shadow` flags (`SPRITE_FLAG_NO_SHADOW_CAST`/`_RECEIVE`, default participating; `Sprite::with_*`/`casts_shadow`/`receives_shadow` helpers). Honored on the **CPU** backend (non-caster excluded from the occluder; non-receiver passes `shadow: None`). Test `sprite_shadow_flags_default_on_and_toggle`.
  - **XS.4.1** ✅ — raised `pick_required_limits` to `min(adapter, SPRITE_SHADOW_MIN_STORAGE_BUFFERS=22)`; `GpuRenderer::sprite_shadows_capable()` reflects whether the device granted ≥22 (graceful fallback to today's unshadowed GPU sprites otherwise). The dev GPU here grants 22 (capable), so XS.4.2/.3 are headlessly verifiable.
  - **XS.4.2** ✅ — GPU sprites **receive** terrain shadows. On capable devices the renderer splices `sprite_terrain_shadow.wgsl` over the stub `shadow_occluded_world` in `sprite_model_dda.wgsl` (string substitution on `//XS4_STUB_BEGIN..END`), binding the terrain occupancy set (bindings 16..23 = occ pages 0..3 + chunk occupancy + slot index + grid meta + grid cameras) — the sprite pass reaches 22 storage buffers. `shade_sprite_lit` queries `shadow_occluded_world` for the sun (sun_flags bit1) + flagged point lights, honoring the per-instance `flags` (bit5 = NO_RECEIVE). Sprite uniform gained the paging+shadow fields; `Instance` gained `flags` (plumbed from `roxlap_formats::sprite` through the facade). `bgl`/bind-group conditionally add 16..23; non-capable keeps the 14-binding stub path. naga validates **both** variants; GPU visual is demo-verified (no headless sprite renderer, as for DL.4/.7).
  - **XS.4.3** ✅ — GPU sprites **cast** onto terrain. Mirror of XS.4.2: on capable devices the renderer splices `scene_sprite_shadow.wgsl` over a `sprites_occlude` stub in `scene_dda.wgsl` (`//XS4C_STUB_BEGIN..END`), binding the sprite registry (19 = instances, 20 = model meta, 21 = occupancy; scene pass → 19 storage buffers). `shadow_occluded_world` now also calls `sprites_occlude`, which loops the visible instances (`u.sprite_cast_count`, new uniform field) and marches each sprite volume, honoring the per-instance `flags` (bit4 = NO_CAST). A capable device with no sprite registry binds an 80-byte dummy at 19..21 (`SceneDdaResources::sprite_cast_dummy`) with count 0. naga validates the capable scene variant too. **Bidirectional GPU sprite shadows complete; full CPU/GPU parity.** GPU visual demo-verified (no headless sprite renderer).

**Macro-stage XS complete** — cross-grid shadows, lit translucent layers, and sprite shadows (cast + receive) all land on both backends, per-sprite configurable.

Other deferred (future): soft/penumbra shadows, **option B** (screen-space palette + ordered/Bayer dithering) — deferred deliberately: it only pays off as the tail of an **advanced post pipeline** (downsample → a cubemap/temporal pass to stabilize the dither noise under camera rotation → palette quantize → dither). Dithering alone, per-frame in screen space, would crawl/shimmer when the camera turns; do it once that pipeline exists, not before. Next macro-stage: **AO** (bake voxel ambient occlusion into the ambient byte — see the follow-on section below; targets 0.19.0).

**CPU lands nothing here** (locked decision #1); the usual "CPU first, GPU
mirrors" discipline is inverted for this stage because the feature is
GPU-exclusive. The standing regression discipline is preserved by the
`lights==None` byte-identical gate at every stage.

## Risks

- **R1 — Shadow-ray cost.** N shadow casters = N extra DDA marches per
  primary hit; on the per-pixel budget this can dominate. Mitigation:
  hard `MAX_SHADOW_CASTERS=4`, `shadow_max_steps`/`shadow_max_dist` caps,
  shadowless point lights as the common case, and shadows behind the
  per-light `casts_shadow` flag. Bench in DL.3 against the DL.0 baseline
  FPS; if a cast is too slow, document the cap (no silent slowdown).

- **R2 — Self-shadow acne / peter-panning.** Biasing the shadow origin
  along the face normal trades acne for detached contact shadows.
  Mitigation: bias in **voxel units** (`shadow_bias_voxels`, default 1.5)
  tuned in DL.3; the axis-aligned face normal makes the bias exact (no
  smooth-normal slope-scaling needed).

- **R3 — Double-lighting during the DL interim.** Until the AO stage
  rebakes the ambient byte, the byte still holds the old directional bake,
  so a flat surface gets baked-directional × ambient + dynamic-sun. This
  may read as too bright/contrasty. Mitigation: ship a `ambient` multiplier
  (default <1) and flag it; the AO stage removes the directionality from
  the byte for the proper look. Accept interim approximation.

- **R4 — Per-grid light transform correctness.** Lights are rotated into
  each grid's local frame; a chirality/translation bug silently mislights
  rotated grids (cf. the documented voxlap basis-chirality footgun).
  Mitigation: reuse the *same* `world_camera_to_grid_local` path the
  cameras use; add unit tests asserting a known light hits a known grid
  voxel from the expected direction under rotation.

- **R5 — Cross-grid shadows missing.** A ship will not shadow the terrain
  below it (intra-grid only). Mitigation: documented limitation; deferred
  to a future stage that marches shadow rays across the grid set (the
  outer chunk-grid DDA in `march_grid` is the hook). Not in DL scope.

- **R6 — Translucent terrain interaction.** The TV accumulate path
  (`u.terrain_has_translucent`) composites multiple voxels per ray. DL
  lights the **opaque backstop hit**; per-layer lit translucency is
  deferred (translucent layers keep ambient-only shading). Mitigation:
  scope DL.1-3 to the opaque hit; note translucent-lit as future work.

## Validation (every sub-substage)

- `cargo test` across the workspace stays green; the **`lights==None`
  byte-identical** output is the headline regression gate at every stage
  (existing demo scenes render exactly as before when no rig is set).
- CPU-side per-grid light-transform math is unit-tested (rotation +
  translation), independent of the GPU.
- No bit-exact GPU gate (clean-room, GPU-only) — visual parity is
  image-regression-with-tolerance plus manual dogfood in the DL.5 demo.
- naga WGSL validation + wgpu device tests run in CI; **pixel-visual
  checks are manual** (headless CI has no display).
- "No silent caps": every dropped light / demoted shadow caster emits a
  `log::warn!`.
- DL.3 records an FPS delta vs the DL.0 baseline so shadow cost is
  explicit, never silent.

---

# Follow-on stage: voxel ambient occlusion (Substage AO)

Designed now because it **shares the ambient byte** that DL frees up
(locked decision #2). Separate stage, **targets 0.19.0**, tag **AO**.

## Approach (locked)

**Bake AO into the ambient byte** (chosen over runtime per-pixel AO). A
CPU pre-pass computes per-voxel occlusion from surrounding voxels and
writes `ambient_level * (1 - ao)` into the high byte — the exact channel
DL now reads as ambient. Result: crevices, contacts, and inside corners
darken; both CPU and GPU benefit with **zero shader change** (they already
multiply the byte). Cheap at render time; cost is paid once at bake /
stream-in.

Rationale vs runtime AO: the engine already has the whole bake
infrastructure — `EstNormCache` (`world_lighting.rs:159`) decodes 5×5×5
(and wider) occupancy bitsets cross-chunk via `build_with_reader`, and
`apply_lighting_with_cache` (`:561`) is the write path. AO is a new
`compute_brightness` variant reusing that occupancy. Runtime per-pixel AO
(corner smooth-lighting) is left as an optional future refinement.

## Sketch

- **AO.0** ✅ — `lightmode == 3` AO bake in `world_lighting.rs`:
  `EstNormCache::ambient_occlusion(x,y,z,n)` samples the `±ESTNORMRAD`
  neighbourhood on the voxel's **air side** (`offset·n < 0`), inverse-distance
  weighted, → occluded fraction `0..1`; `ao_byte` writes `128·(1 − AO_STRENGTH·ao)`
  (open = 128, crevices darker) into the brightness byte (the DL ambient/AO
  channel). Dispatched in `shade_column`. Unit tests: `ambient_occlusion_darkens_next_to_a_wall`
  + `lightmode3_bakes_ambient_occlusion` (open floor = 128; voxel beside a wall darker).
- **AO.1** ✅ — wired into the scene bake: `bake_lightmode_1` generalized to
  `bake_lightmode(scene, mode)` + `bake_ao_pub` (mode 3); the "Lighting" demo
  scene bakes AO into its floor/pillars/monument at build time. (Streaming
  re-bake reuses the same cross-chunk `build_with_reader` reader; not exercised
  by the static Lighting scene.)
- **AO.2** (partial) — AO is **concave-only**, computed **per exposed face**
  (normal-free): for each axis face whose immediate neighbour is air, sample
  the half-space *in front* of it for solids. The earlier estnorm-gradient
  hemisphere had a **"pillow" bug** — near a convex edge the gradient normal
  tilts toward the solid bulk, so a voxel's own folded-over surface (e.g. a
  pillar's top above its side face) fell on the "air side" and counted as
  occlusion → a 1-voxel dark border one step inside *every* convex edge. The
  per-face method has no normal to tilt, so flat faces + convex edges read `0`
  at any radius; only a perpendicular solid in front of an exposed face (a
  concave corner) occludes. Verified by `ao_only_darkens_concave_not_convex`
  (synthetic) + `ao_only_concave_on_setrect_pillar` (real `set_rect` pillar:
  the bug only showed on the "one-in-from-edge" voxels, which the scan-style
  test catches). **Tunable** via `AoParams { strength, radius, min_floor }`
  (re-exported from roxlap-core), threaded through `apply_lighting_with_cache`
  → `shade_column` → `ao_byte`; `update_lighting`/`update_lighting_chunk` pass
  the default (the engine doesn't tune AO). `bake_ao_pub(scene, strength,
  radius)` exposes it; the "Lighting" demo retunes AO depth live with `N`/`M`
  (re-bakes the one-chunk scene). `min_floor` caps how dark a crevice can get.
- **Cross-chunk ±z seam continuity** (stacked grids). `EstNormCache::solid`
  used to hardcode the out-of-window z boundary (`z < 0 → air`,
  `z ≥ MAXZDIM → bedrock`), so for a stacked grid the bake's `±ESTNORMRAD`
  z-padding saw fake air/bedrock at a chunk's top/bottom face instead of the
  neighbour chunk — an AO (and `estnorm`) discontinuity at the z-seam. New
  `EstNormCache::build_with_reader_z` takes a `chz`-aware reader
  (`Fn(x, y, chz_delta) → Option<&[u8]>`, `0` / `-1` = above / `+1` = below)
  and fills per-column `z_below` / `z_above` overlays (the `ESTNORMRAD` voxels
  just past each boundary, read from the chunk above/below; an absent
  neighbour falls back to the old air/bedrock boundary, so a topmost /
  bottommost chunk is unchanged). The plain `build_with_reader` leaves the
  overlays empty → single-layer bakes stay byte-identical. The scene-graph
  `bake_lightmode` and both demo bake sites use the z-aware build. Verified by
  `ao_z_seam_reads_stacked_neighbour`.
- **AO.3** (deferred) — broader demo showcase + README; CHANGELOG entry landed
  under `[Unreleased]` (AO ships in the same unreleased batch as DL at 0.18.0,
  not a separate 0.19.0 bump — 0.18.0 is not yet released).

## AO risks

- **AO.R1 — Chunk-border seams.** Per-chunk bake must read neighbor
  occupancy or AO discontinues at chunk edges. Mitigation: reuse the
  `build_with_reader` cross-chunk closure already used for lighting seams.
- **AO.R2 — Re-bake cost on edits/streaming.** AO is baked, so every
  voxel edit dirties a neighborhood. Mitigation: region-bounded re-bake
  (the lighting path is already row/region-parallel via rayon); accept
  baked-AO latency on edits as the cost of cheap render-time AO.
- **AO.R3 — Interaction with DL ambient multiplier.** AO and the
  `LightRig.ambient` multiplier both scale the same term. Mitigation:
  define AO as occlusion of ambient only (`ambient_color * ao_byte`),
  direct light unaffected by AO (standard separation), documented.
