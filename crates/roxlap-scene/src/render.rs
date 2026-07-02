//! Scene-level rendering — drives [`roxlap_core::opticast::opticast`]
//! across the grids of a [`Scene`].
//!
//! Two entry points:
//!
//! - [`render_scene_composed`] (recommended for multi-grid scenes):
//!   per grid, allocates a temporary framebuffer + zbuffer, runs
//!   opticast into the temp, then merges into the shared output via
//!   per-pixel min-z. Correctly composites overlapping grid output.
//! - [`render_scene`] (single-grid trusting caller): writes every
//!   grid directly into the shared rasterizer. For single-grid
//!   scenes this matches a direct opticast call byte-for-byte; for
//!   multi-grid it's last-grid-wins (sky writes from grid B
//!   overwrite grid A's hits). Useful for tests / single-grid
//!   sanity checks.
//!
//! ## S4B.2.e: Approach B multi-chunk dispatch
//!
//! Both APIs route per-grid rendering through
//! [`crate::Grid::chunk_xy_backing`] → [`roxlap_core::ChunkGrid`] →
//! [`roxlap_core::GridView::from_chunk_grid`] → [`opticast`].
//! `opticast`'s prelude looks up the camera's chunk via
//! [`roxlap_core::GridView::chunk_at_xy`]; the grouscan column-step
//! swaps the active per-chunk `(slab_buf, column_offsets)` when
//! rays cross a chunk-XY boundary. The combined-world stitch
//! (Approach C, S4.0..S4.2) is no longer in the render path — the
//! lighting bake still uses it until S4B.4 lands a per-chunk bake.
//!
//! Per-grid rotation (S5) and per-grid LOD (S6) plug in at the
//! same dispatch point: rotate the world camera into grid-local
//! before the chunk-grid lookup, then dispatch coarse / fine /
//! billboard based on grid-camera distance.

// `fb` / `zb` (framebuffer / zbuffer) and the `_fb` / `_zb` suffixes
// throughout this module are voxlap-canonical pairs — drilling them
// apart with longer names just hurts readability.
#![allow(clippy::similar_names)]

use glam::DVec3;
use roxlap_core::dda::{render_dda_parallel, CpuLights, CpuPointLight, DdaEnv};
use roxlap_core::opticast::OpticastSettings;
use roxlap_core::sky::Sky;
use roxlap_core::Camera;
use roxlap_formats::material::MaterialTable;

use crate::billboard::{self, BillboardCache, DEFAULT_RESOLUTION as BILLBOARD_RESOLUTION};
use crate::chunks;
use crate::lod::Lod;
use crate::occluder::SceneOccluder;
use crate::{GridId, GridTransform, Scene, CHUNK_SIZE_XY};
use roxlap_core::{CompositeOccluder, WorldOccluder, WorldShadowCtx};
use std::collections::HashMap;

/// Sentinel colour stamped into a `render_sky = false` grid's
/// temporary framebuffer wherever the rasterizer would have drawn
/// sky. After opticast, [`render_scene_composed`] walks the temp
/// buffer and resets `temp_zb` to [`f32::INFINITY`] for any pixel
/// still carrying this value — those pixels then always lose
/// [`compose_into`]'s min-z test and the underlying grid's sky
/// (or another grid's hit) wins.
///
/// Alpha byte is `0x00`. Voxlap voxel slabs carry an alpha-encoded
/// shade in `[0x00, 0x80]`, but a `0x00` alpha **with this exact
/// RGB pattern** is exceedingly unlikely to occur on a real hit
/// (the lit-voxel path produces alpha ≥ 0x40 in practice). Bit
/// pattern is also visually distinct (cyan-ish neon) if anything
/// ever leaks through to the screen, making the bug obvious.
const SKY_MASK_SENTINEL: u32 = 0x00_DE_AD_BE;

/// CPU fog + per-face shading config for the DDA backend, passed by
/// value into the scene render entry points (replaces the old
/// `&mut ScratchPool` parameter the voxlap path threaded fog through).
///
/// `max_scan_dist <= 0` disables fog (no distance blend). Otherwise the
/// DDA renderer linearly ramps a hit's colour toward [`Self::color`]
/// over `max_scan_dist` voxels. `side_shades` darkens each of the six
/// voxel faces — `[x-, x+, y-, y+, z-, z+]`.
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuFog {
    /// Low-24-bit RGB fog colour.
    pub color: u32,
    /// Distance (voxels) at which fog is fully opaque; `<= 0` ⇒ fog OFF.
    pub max_scan_dist: i32,
    /// Per-face brightness reduction `[x-, x+, y-, y+, z-, z+]`.
    pub side_shades: [i8; 6],
}

/// Project a world-space [`Camera`] into a grid's local frame:
/// translate by `-transform.origin`, then apply
/// `transform.rotation.inverse()` to the position and the
/// orthonormal basis (`right` / `down` / `forward`).
///
/// Identity rotation collapses to pure translation, byte-identical
/// to the pre-S5 path (`DQuat::IDENTITY * v == v`). For a rotated
/// grid the rasterizer still sees an axis-aligned chunk grid —
/// rotation is invisible below this layer per PORTING-SCENE.md § S5.
///
/// The basis is rotated as a free vector (no translation
/// component); position is rotated about the grid origin.
fn world_camera_to_grid_local(camera: &Camera, transform: &GridTransform) -> Camera {
    let inv = transform.rotation.inverse();
    let world_offset = DVec3::from_array(camera.pos) - transform.origin;
    let local_pos = inv * world_offset;
    let local_right = inv * DVec3::from_array(camera.right);
    let local_down = inv * DVec3::from_array(camera.down);
    let local_forward = inv * DVec3::from_array(camera.forward);
    Camera {
        pos: local_pos.to_array(),
        right: local_right.to_array(),
        down: local_down.to_array(),
        forward: local_forward.to_array(),
    }
}

/// CPU.1 — transform world-space dynamic lights into a grid's local frame
/// (the same translate + inverse-rotation as [`world_camera_to_grid_local`]):
/// point positions are points (origin-relative + inverse-rotated); the sun
/// direction is a vector (inverse-rotated only). Point lights land in `scratch`
/// so the returned [`CpuLights`] can borrow them for the grid's render.
///
/// PF.7 (C4) — `grid_sphere` is the grid's world-space bounding sphere
/// `(centre, radius)`: a light whose reach-sphere can't touch it (with
/// slack for the shadow-bias sample offset) is dropped BEFORE the
/// transform, so the per-hit light loop never sees it. Conservative ⇒
/// byte-identical (a dropped light's `point_falloff` would be 0 at every
/// reachable sample anyway). `None` skips the cull.
fn grid_local_lights<'a>(
    world: &CpuLights<'_>,
    transform: &GridTransform,
    scratch: &'a mut Vec<CpuPointLight>,
    grid_sphere: Option<(DVec3, f64)>,
) -> CpuLights<'a> {
    scratch.clear();
    if !world.enabled {
        return CpuLights::default();
    }
    let inv = transform.rotation.inverse();
    #[allow(clippy::cast_possible_truncation)]
    let sun_dir = if world.sun {
        let d = inv
            * DVec3::new(
                f64::from(world.sun_dir[0]),
                f64::from(world.sun_dir[1]),
                f64::from(world.sun_dir[2]),
            );
        [d.x as f32, d.y as f32, d.z as f32]
    } else {
        [0.0; 3]
    };
    // Shade samples sit on voxel surfaces inside the bounding sphere,
    // nudged up to `shadow_bias` along the normal — expand by that plus
    // a unit of float slack.
    let cull_slack = f64::from(world.shadow_bias) + 1.0;
    for p in world.points {
        if let Some((centre, radius)) = grid_sphere {
            let lp = DVec3::new(
                f64::from(p.pos[0]),
                f64::from(p.pos[1]),
                f64::from(p.pos[2]),
            );
            if (lp - centre).length() > f64::from(p.radius) + radius + cull_slack {
                continue;
            }
        }
        let lp = inv
            * (DVec3::new(
                f64::from(p.pos[0]),
                f64::from(p.pos[1]),
                f64::from(p.pos[2]),
            ) - transform.origin);
        // SL — the cone axis is a vector: inverse-rotate only (no origin).
        let sd = inv
            * DVec3::new(
                f64::from(p.spot_dir[0]),
                f64::from(p.spot_dir[1]),
                f64::from(p.spot_dir[2]),
            );
        #[allow(clippy::cast_possible_truncation)]
        scratch.push(CpuPointLight {
            pos: [lp.x as f32, lp.y as f32, lp.z as f32],
            color: p.color,
            intensity: p.intensity,
            radius: p.radius,
            casts_shadow: p.casts_shadow,
            spot_dir: [sd.x as f32, sd.y as f32, sd.z as f32],
            cos_inner: p.cos_inner,
            cos_outer: p.cos_outer,
        });
    }
    CpuLights {
        enabled: true,
        sun: world.sun,
        sun_dir,
        sun_color: world.sun_color,
        sun_intensity: world.sun_intensity,
        sun_casts_shadow: world.sun_casts_shadow,
        points: scratch.as_slice(),
        ambient: world.ambient,
        bands: world.bands,
        shadow_tint: world.shadow_tint,
        // CPU.2 — shadows: the rig is world-space here; shadow distances are
        // grid-uniform (no scaling), so they carry through unchanged.
        shadow_strength: world.shadow_strength,
        shadow_bias: world.shadow_bias,
        shadow_max_dist: world.shadow_max_dist,
    }
}

/// Outcome of a [`render_scene`] / [`render_scene_composed`] call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderOutcome {
    /// At least one grid produced a render.
    Rendered {
        /// Number of grids that were drawn.
        grids_drawn: usize,
    },
    /// No grid rendered — the scene was empty (no populated grids).
    Empty,
}

/// Render every grid in `scene` directly into `(fb, zb)` — no
/// per-grid temp buffer, no compose merge. For multi-grid scenes
/// this is last-grid-wins (later grids' opticast writes overwrite
/// earlier grids' pixels indiscriminately, including sky), so it's
/// only correct for single-grid scenes.
///
/// Use this when you have one grid and want the byte-stable
/// PR.3: pick the cheapest `GridView` constructor that matches the
/// grid's chunk layout.
///
/// Trivial-single-chunk grids (1 chunk at index `(0, 0, 0)`) bypass
/// the multi-chunk rasterizer path: `GridView::from_single_vxl`
/// leaves `chunk_grid = None`, so `phase_after_delete_kept_presync`
/// takes the cheaper single-chunk branch instead of doing
/// `chunk_at_xyz` + IVec2-equality + `Option::is_some` per
/// column-step. Markers / pickups / small ships qualify.
///
/// Multi-chunk grids (ground, larger ships) fall through to
/// `from_chunk_grid` with the supplied `ChunkGrid`.
fn single_chunk_fast_path<'a>(
    backing: &'a chunks::ChunkXyBacking<'a>,
    cg: &'a roxlap_core::ChunkGrid<'a>,
) -> roxlap_core::GridView<'a> {
    if backing.chunks_x == 1
        && backing.chunks_y == 1
        && backing.chunks_z == 1
        && backing.origin_chunk_xy == [0, 0]
        && backing.origin_chunk_z == 0
    {
        // chunk_xyz_backing populates each `Vec<Option<GridView>>`
        // slot via `GridView::from_single_vxl`, which leaves
        // `chunk_grid = None`. Reuse that directly.
        if let Some(single) = backing.chunks[0] {
            return single;
        }
    }
    roxlap_core::GridView::from_chunk_grid(cg, CHUNK_SIZE_XY)
}

/// matches-direct-opticast property — the test suite uses it as a
/// sanity check that the combined-world stitch + render harness
/// doesn't drift vs. a raw [`opticast`] call.
///
/// Caller pre-fills `fb` with the desired sky colour and `zb` with
/// any value (typically `0.0` matching the per-chunk renderer's
/// convention or `f32::INFINITY` for compose-friendly init); the
/// rasterizer overwrites both per pixel that gets a hit.
#[allow(clippy::too_many_arguments)]
pub fn render_scene(
    fb: &mut [u32],
    zb: &mut [f32],
    pitch_pixels: usize,
    width: u32,
    height: u32,
    fog: CpuFog,
    scene: &mut Scene,
    camera: &Camera,
    settings: &OpticastSettings,
    sky: Option<&Sky>,
) -> RenderOutcome {
    debug_assert_eq!(fb.len(), zb.len());
    let pixel_count = (width as usize) * (height as usize);
    debug_assert_eq!(fb.len(), pixel_count);

    let mut grids_drawn = 0usize;
    for (_id, grid) in scene.grids_mut() {
        // S4B.2.e: Approach B render path. World → grid-local
        // camera transform doesn't need a voxel-offset adjustment
        // anymore — Approach B's chunks live at their signed
        // (chx, chy) indices and `chunk_at_xy` handles negative-
        // index lookups natively.
        //
        // S5.0: per-grid arbitrary rotation. The local camera is
        // built by `world_camera_to_grid_local` — translation +
        // inverse-rotation of the basis. Identity rotation keeps
        // this byte-identical to the pre-S5 translate-only form.
        // DDA.7: refresh the cross-frame brick cache (needs `&mut grid`)
        // before borrowing the grid immutably for `backing`.
        let dda_mip = grid.ensure_dda_bricks(0);
        let Some(backing) = grid.chunk_xyz_backing() else {
            // Empty grid (no populated chz=0 chunks) — skip.
            continue;
        };
        let local_cam = world_camera_to_grid_local(camera, &grid.transform);
        let cg = roxlap_core::ChunkGrid {
            chunks: &backing.chunks,
            origin_chunk_xy: backing.origin_chunk_xy,
            origin_chunk_z: backing.origin_chunk_z,
            chunks_x: backing.chunks_x,
            chunks_y: backing.chunks_y,
            chunks_z: backing.chunks_z,
        };
        let grid_view = single_chunk_fast_path(&backing, &cg);
        // DDA backend. The direct path doesn't pre-fill, so seed sky
        // (black) + far depth here — DDA leaves misses untouched.
        for px in fb.iter_mut() {
            *px = 0;
        }
        for d in zb.iter_mut() {
            *d = f32::INFINITY;
        }
        let fog_on = fog.max_scan_dist > 0;
        #[allow(clippy::cast_precision_loss)]
        let env = DdaEnv {
            sky,
            fog_color: if fog_on { fog.color } else { 0 },
            fog_max_dist: if fog_on {
                fog.max_scan_dist.max(1) as f32
            } else {
                0.0
            },
            side_shades: fog.side_shades,
            // The direct (non-composed) path is opaque-only; terrain
            // materials flow through render_scene_composed_with_materials.
            materials: None,
            terrain_materials: &[],
            // The direct path is unlit (lighting flows through the composed
            // path); keep it on the baked-byte shade.
            lights: CpuLights::default(),
            world_shadow: None,
        };
        render_dda_parallel(
            &local_cam,
            settings,
            grid_view,
            fb,
            zb,
            pitch_pixels,
            &env,
            &grid.dda_brick_cache,
            dda_mip,
        );
        grids_drawn += 1;
    }
    if grids_drawn == 0 {
        RenderOutcome::Empty
    } else {
        RenderOutcome::Rendered { grids_drawn }
    }
}

/// Per-pixel "min-z wins" merge of `(temp_fb, temp_zb)` into
/// `(shared_fb, shared_zb)`.
///
/// Voxlap's z-buffer convention: `z` = perpendicular distance from
/// camera; **smaller `z` = closer to camera**. This helper picks
/// the closer pixel per slot. Sky pixels emerge with a large `z`
/// (`scratch.skycast.dist`, set to `gxmax` or `i32::MAX` per
/// `phase_startsky`) so they always lose to any hit's finite
/// distance.
///
/// `temp_fb` / `temp_zb` are read-only inputs; both must have the
/// same length as `shared_fb` / `shared_zb` (debug-asserted).
pub fn compose_into(
    shared_fb: &mut [u32],
    shared_zb: &mut [f32],
    temp_fb: &[u32],
    temp_zb: &[f32],
) {
    debug_assert_eq!(shared_fb.len(), shared_zb.len());
    debug_assert_eq!(shared_fb.len(), temp_fb.len());
    debug_assert_eq!(shared_fb.len(), temp_zb.len());
    for i in 0..shared_fb.len() {
        if temp_zb[i] < shared_zb[i] {
            shared_fb[i] = temp_fb[i];
            shared_zb[i] = temp_zb[i];
        }
    }
}

/// Half-open screen rectangle `[x0, x1) × [y0, y1)` a grid's
/// projection is confined to — the scissor [`render_scene_composed`]
/// uses to render and compose each grid only within its screen
/// footprint instead of over the whole frame.
#[derive(Clone, Copy, Debug)]
struct ScreenRect {
    x0: u32,
    x1: u32,
    y0: u32,
    y1: u32,
}

impl ScreenRect {
    fn is_empty(self) -> bool {
        self.x0 >= self.x1 || self.y0 >= self.y1
    }
}

/// Project a world-space bounding sphere `(centre, radius)` to a
/// conservative screen rectangle under opticast's pinhole — focal `hz`,
/// principal point `(hx, hy)`, ray for pixel `(px, py)` being
/// `(px-hx)·right + (py-hy)·down + hz·forward` (camera_math). Returns:
///
/// - `Some(rect)` clamped to the viewport when the sphere is safely in
///   front of the camera. The rect may be **empty** (sphere off to one
///   side) → the grid can't appear, so the caller skips it entirely.
/// - `None` when the camera is inside or near the sphere (forward-depth
///   `z ≤ radius`), where a finite screen bound is unsafe → the caller
///   must render the grid full-frame.
///
/// Conservative on purpose (never clips a pixel the full render would
/// touch): the projected radius uses the over-estimate `hz·R/(z−R)`
/// (exact is `hz·R/√(z²−R²)`) and pads by `anginc + 1`, matching the
/// projection's `anginc` viewport padding.
fn project_sphere_to_screen(
    camera: &Camera,
    centre: DVec3,
    radius: f64,
    settings: &OpticastSettings,
) -> Option<ScreenRect> {
    let d = centre - DVec3::from_array(camera.pos);
    let z = d.dot(DVec3::from_array(camera.forward));
    if z <= radius {
        return None; // camera inside / in front of the sphere shell
    }
    let x = d.dot(DVec3::from_array(camera.right));
    let y = d.dot(DVec3::from_array(camera.down));
    let (hx, hy, hz) = (
        f64::from(settings.hx),
        f64::from(settings.hy),
        f64::from(settings.hz),
    );
    let sr = hz * radius / (z - radius); // over-estimated screen radius
    let sx = hx + x / z * hz;
    let sy = hy + y / z * hz;
    let pad = f64::from(settings.anginc) + 1.0;
    let (xres, yres) = (f64::from(settings.xres), f64::from(settings.yres));
    let clamp = |v: f64, hi: f64| v.clamp(0.0, hi);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    Some(ScreenRect {
        x0: clamp((sx - sr - pad).floor(), xres) as u32,
        x1: clamp((sx + sr + pad).ceil(), xres) as u32,
        y0: clamp((sy - sr - pad).floor(), yres) as u32,
        y1: clamp((sy + sr + pad).ceil(), yres) as u32,
    })
}

/// Fill each `rect` row of a `u32` buffer (row stride `pitch`) with
/// `val` — the scissored analogue of `slice.fill(val)`.
fn fill_rect_u32(buf: &mut [u32], pitch: usize, rect: ScreenRect, val: u32) {
    for y in rect.y0..rect.y1 {
        let row = y as usize * pitch;
        buf[row + rect.x0 as usize..row + rect.x1 as usize].fill(val);
    }
}

/// Fill each `rect` row of an `f32` buffer (row stride `pitch`) with `val`.
fn fill_rect_f32(buf: &mut [f32], pitch: usize, rect: ScreenRect, val: f32) {
    for y in rect.y0..rect.y1 {
        let row = y as usize * pitch;
        buf[row + rect.x0 as usize..row + rect.x1 as usize].fill(val);
    }
}

/// Min-z compose `temp_*` into `fb`/`zb` over `rect` only — the
/// scissored analogue of [`compose_into`]. A `temp` pixel wins where its
/// `z` is strictly smaller than the destination's.
///
/// PF.7 (C6) — rayon rows: a memory-bandwidth loop repeated per grid per
/// frame; rows are disjoint (`par_chunks_mut` of both destinations),
/// sources read-only. Bit-identical.
fn compose_rect(
    fb: &mut [u32],
    zb: &mut [f32],
    temp_fb: &[u32],
    temp_zb: &[f32],
    pitch: usize,
    rect: ScreenRect,
) {
    use rayon::prelude::*;
    let (y0, y1) = (rect.y0 as usize, rect.y1 as usize);
    let (x0, x1) = (rect.x0 as usize, rect.x1 as usize);
    if y0 >= y1 {
        return;
    }
    // The last row may be short of a full `pitch` when the buffer is
    // exactly `width*height` — clamp the slice end.
    let end = (y1 * pitch).min(fb.len());
    fb[y0 * pitch..end]
        .par_chunks_mut(pitch)
        .zip(zb[y0 * pitch..end].par_chunks_mut(pitch))
        .enumerate()
        .for_each(|(dy, (frow, zrow))| {
            let row = (y0 + dy) * pitch;
            for x in x0..x1 {
                if temp_zb[row + x] < zrow[x] {
                    zrow[x] = temp_zb[row + x];
                    frow[x] = temp_fb[row + x];
                }
            }
        });
}

/// PF.7 (C6) — reusable scratch for the composed scene render: the
/// per-grid temp framebuffer/z-buffer pair (was two full-frame `vec!`
/// allocations + initialising writes per call — ≈7.4 MB at 720p, ×16
/// under 4×SSAA), the per-grid light scratch, and the phase-A mip map.
/// Own one per renderer and pass it to
/// [`render_scene_composed_with_materials_scratch`]; the buffers grow to
/// the frame size on first use and are reused verbatim afterwards (every
/// pixel the render reads is filled per grid first, so no per-frame
/// clear is needed).
#[derive(Default)]
pub struct SceneRenderScratch {
    temp_fb: Vec<u32>,
    temp_zb: Vec<f32>,
    lights: Vec<CpuPointLight>,
    eff_mips: HashMap<GridId, u32>,
}

/// Render every grid in `scene` with per-grid temporary buffers +
/// z-buffer composition. The canonical multi-grid scene render
/// path.
///
/// Algorithm:
/// 1. Caller pre-fills `fb` with the desired sky colour and `zb`
///    with [`f32::INFINITY`] (so any rendered pixel wins the
///    initial composition).
/// 2. For each grid, allocate a temporary `(temp_fb, temp_zb)` of
///    the same size, pre-fill them with sky / `INFINITY`, and run
///    [`opticast`] into them via a [`ScalarRasterizer`] over the
///    temporary buffers AND the grid's combined-world view (S4.0).
/// 3. Merge the temporary buffers into the shared `(fb, zb)` via
///    [`compose_into`] — closer pixels (smaller `z`) win.
///
/// Pixel correctness across overlapping grids: sky pixels emerge
/// with `z` = `gxmax` / `i32::MAX` (a very large value), so they
/// always lose to any hit. Hits compete on actual perpendicular
/// distance — the closer grid's surface is what gets composited.
///
/// `pitch_pixels` is the framebuffer's row stride in pixels (×4 for
/// bytes). `width` × `height` must equal `fb.len()` /
/// `zb.len()`. `sky` is the optional textured sky resource the
/// rasterizer threads through to `phase_startsky`; `None` ⇒ solid
/// `pool.skycast` fill.
///
/// **Heap allocation per call:** two `Vec` allocations per grid (a
/// temp framebuffer and zbuffer). For repeated frame rendering an
/// owned scratch struct that pre-allocates these is the obvious
/// optimisation; deferred until profiling shows it matters.
#[allow(clippy::too_many_arguments)]
pub fn render_scene_composed(
    fb: &mut [u32],
    zb: &mut [f32],
    pitch_pixels: usize,
    width: u32,
    height: u32,
    fog: CpuFog,
    scene: &mut Scene,
    camera: &Camera,
    settings: &OpticastSettings,
    sky_color: u32,
    sky: Option<&Sky>,
) -> RenderOutcome {
    render_scene_composed_scissored(
        fb,
        zb,
        pitch_pixels,
        width,
        height,
        fog,
        scene,
        camera,
        settings,
        sky_color,
        sky,
        true,
        None,
        &[],
        CpuLights::default(),
        None,
        &mut SceneRenderScratch::default(),
    )
}

/// [`render_scene_composed`] with TV terrain materials: `materials` is the
/// global palette and `terrain_materials` the colour→material map; together
/// they make matching-colour terrain voxels translucent (front-to-back
/// composited). An empty map / `None` palette renders identically to
/// [`render_scene_composed`].
#[allow(clippy::too_many_arguments)]
pub fn render_scene_composed_with_materials(
    fb: &mut [u32],
    zb: &mut [f32],
    pitch_pixels: usize,
    width: u32,
    height: u32,
    fog: CpuFog,
    scene: &mut Scene,
    camera: &Camera,
    settings: &OpticastSettings,
    sky_color: u32,
    sky: Option<&Sky>,
    materials: Option<&MaterialTable>,
    terrain_materials: &[(u32, u8)],
    lights: CpuLights<'_>,
    // XS.2 — sprite-cast shadow occluder (so sprites darken terrain). `None` ⇒
    // grids-only shadows.
    sprite_occluder: Option<&dyn WorldOccluder>,
) -> RenderOutcome {
    render_scene_composed_scissored(
        fb,
        zb,
        pitch_pixels,
        width,
        height,
        fog,
        scene,
        camera,
        settings,
        sky_color,
        sky,
        true,
        materials,
        terrain_materials,
        lights,
        sprite_occluder,
        &mut SceneRenderScratch::default(),
    )
}

/// [`render_scene_composed_with_materials`] with a caller-owned
/// [`SceneRenderScratch`] (PF.7) — the per-frame render path: the temp
/// buffer pair and per-grid scratch are reused across frames instead of
/// re-allocated per call.
#[allow(clippy::too_many_arguments)]
pub fn render_scene_composed_with_materials_scratch(
    fb: &mut [u32],
    zb: &mut [f32],
    pitch_pixels: usize,
    width: u32,
    height: u32,
    fog: CpuFog,
    scene: &mut Scene,
    camera: &Camera,
    settings: &OpticastSettings,
    sky_color: u32,
    sky: Option<&Sky>,
    materials: Option<&MaterialTable>,
    terrain_materials: &[(u32, u8)],
    lights: CpuLights<'_>,
    sprite_occluder: Option<&dyn WorldOccluder>,
    scratch: &mut SceneRenderScratch,
) -> RenderOutcome {
    render_scene_composed_scissored(
        fb,
        zb,
        pitch_pixels,
        width,
        height,
        fog,
        scene,
        camera,
        settings,
        sky_color,
        sky,
        true,
        materials,
        terrain_materials,
        lights,
        sprite_occluder,
        scratch,
    )
}

/// Backing implementation of [`render_scene_composed`] with the
/// per-grid screen-AABB scissor toggleable. `scissor = true` is the
/// production path; the regression test renders the same scene with
/// `false` (full-frame per grid, the pre-scissor behaviour) and asserts
/// the framebuffer is byte-identical — the scissor must be a pure
/// speed-up, never change a pixel.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn render_scene_composed_scissored(
    fb: &mut [u32],
    zb: &mut [f32],
    pitch_pixels: usize,
    width: u32,
    height: u32,
    fog: CpuFog,
    scene: &mut Scene,
    camera: &Camera,
    settings: &OpticastSettings,
    sky_color: u32,
    sky: Option<&Sky>,
    scissor: bool,
    materials: Option<&MaterialTable>,
    terrain_materials: &[(u32, u8)],
    // CPU.1 — world-space dynamic lights, transformed per grid in the loop.
    lights: CpuLights<'_>,
    // XS.2 — world-space occluder for sprite volumes (so sprites cast shadows
    // onto terrain). Composited with the per-frame grid occluder. `None` ⇒
    // grids only.
    sprite_occluder: Option<&dyn WorldOccluder>,
    // PF.7 — caller-owned reusable buffers (see [`SceneRenderScratch`]).
    scratch: &mut SceneRenderScratch,
) -> RenderOutcome {
    debug_assert_eq!(fb.len(), zb.len());
    let pixel_count = (width as usize) * (height as usize);
    debug_assert_eq!(fb.len(), pixel_count);

    let mut grids_drawn = 0usize;
    // PF.7 (C6) — size (don't clear) the temp pair: every pixel the
    // render reads inside a grid's rect is `fill_rect_*`-initialised for
    // that grid first, and `compose_rect` reads only within the rect, so
    // stale contents outside are never observed. This removes two
    // full-frame allocations AND their initialising writes per call.
    let scratch = &mut *scratch;
    scratch.temp_fb.resize(pixel_count, 0);
    scratch.temp_zb.resize(pixel_count, f32::INFINITY);
    let temp_fb = &mut scratch.temp_fb[..pixel_count];
    let temp_zb = &mut scratch.temp_zb[..pixel_count];

    // XS.1 — phase A (`&mut`): materialise the per-frame caches the render
    // reads — DDA brick caches (Near/Mid) and Far-tier billboard impostors —
    // and record each grid's effective DDA mip. Hoisting these out of the
    // render loop lets phase B run over `&Scene` immutably, so the cross-grid
    // shadow occluder (which also borrows the scene) can coexist with it.
    let cam_world = DVec3::from_array(camera.pos);
    let eff_mips = &mut scratch.eff_mips;
    eff_mips.clear();
    for (id, grid) in scene.grids_mut() {
        let lod = grid.select_lod(cam_world);
        if lod == Lod::Far {
            if !grid.chunks.is_empty() && grid.billboards.is_none() {
                let cache = BillboardCache::build(grid, BILLBOARD_RESOLUTION);
                grid.billboards = Some(cache);
            }
            continue; // Far blits an impostor; no brick cache / mip needed.
        }
        let req = match lod {
            Lod::Mid => grid
                .lod_thresholds
                .mid_mip_levels
                .map_or(0, |n| n.saturating_sub(1)),
            Lod::Near | Lod::Far => 0,
        };
        eff_mips.insert(id, grid.ensure_dda_bricks(req));
    }

    // Reborrow immutably for phase B + the shadow occluder.
    let scene: &Scene = scene;

    // XS.1 — cross-grid hard shadows: build the world-space scene occluder
    // once when shadows are actually active (a caster flagged + non-zero
    // strength), so the shadow ray at a terrain hit tests every grid, not
    // just the one it hit. `None` ⇒ the single-grid `SamplerShadow` path.
    let shadows_on = lights.enabled
        && lights.shadow_strength > 0.0
        && (lights.sun_casts_shadow || lights.points.iter().any(|p| p.casts_shadow));
    let grid_occ = shadows_on
        .then(|| SceneOccluder::build(scene))
        .filter(|o| !o.is_empty());
    // XS.2 — combine the grid occluder with the sprite occluder (sprites cast
    // onto terrain). `composite_store` backs the borrow when both are present.
    let composite_store;
    let active_occluder: Option<&dyn WorldOccluder> = if shadows_on {
        match (grid_occ.as_ref(), sprite_occluder) {
            (Some(g), Some(s)) => {
                composite_store = CompositeOccluder { a: g, b: s };
                Some(&composite_store)
            }
            (Some(g), None) => Some(g),
            (None, Some(s)) => Some(s),
            (None, None) => None,
        }
    } else {
        None
    };

    for (grid_id, grid) in scene.grids() {
        // S6.0/S6.1: per-grid LOD tier dispatch. The picker keys
        // off the grid's `lod_thresholds` and the world-space
        // camera. Default thresholds are `always_near` so every
        // grid lands on `Lod::Near` and the framebuffer stays
        // byte-identical to the pre-S6 path.
        //
        // S6.1: `Mid` applies the grid's `mid_mip_levels` /
        // `mid_mip_scan_dist` overrides (if `Some`) on top of the
        // base settings, biasing the grid into coarser mips. With
        // both `None`, Mid renders identically to Near (graceful
        // degrade — callers opt into the Mid plumbing via
        // `LodThresholds::from_radius_with_mid_mip`).
        //
        // S6.3: `Far` skips the opticast path entirely — render
        // dispatches into the billboard impostor blit (below). The
        // LOD enum is computed before `chunk_xyz_backing` because
        // the Far branch needs `&mut grid` for the lazy cache
        // populate, which conflicts with the `&grid` lifetime
        // backing's tied to.
        let lod = grid.select_lod(DVec3::from_array(camera.pos));

        if lod == Lod::Far {
            // S6.3: Far-tier billboard blit. The impostor cache was built in
            // phase A (above); this immutable pass only reads it.
            //
            // Empty grids have nothing to impostor; skip.
            if grid.chunks.is_empty() {
                continue;
            }
            // Grid bounds + world-space centre. Rotation preserves
            // length, so `bounds.radius` is the world-space radius.
            let bounds = billboard::grid_bounds(grid);
            let centre_world = grid.transform.origin + grid.transform.rotation * bounds.centre;
            // Query direction = unit vector from grid centre TO
            // camera, in grid-local space (snapshots' `view_dir`s
            // live in that frame).
            let cam_pos = DVec3::from_array(camera.pos);
            let centre_to_cam_world = cam_pos - centre_world;
            let ctc_len = centre_to_cam_world.length();
            if !ctc_len.is_finite() || ctc_len < 1e-9 {
                // Camera essentially at grid centre — pick_nearest
                // is ill-defined. Skip; a future frame at a
                // resolvable pose will render normally.
                continue;
            }
            let query_dir_world = centre_to_cam_world / ctc_len;
            let query_dir_local = grid.transform.rotation.inverse() * query_dir_world;
            // Cache was populated in phase A for non-empty Far grids; if it's
            // somehow absent, skip (a future frame re-enters Far and builds).
            let Some(cache) = grid.billboards.as_ref() else {
                continue;
            };
            // pick_nearest only returns None for empty caches; the phase-A
            // build produced a 26-snapshot cache so this resolves.
            let Some(snapshot) = cache.pick_nearest(query_dir_local) else {
                continue;
            };
            billboard::billboard_blit_into(
                fb,
                zb,
                pitch_pixels,
                width,
                height,
                snapshot,
                centre_world,
                bounds.radius,
                camera,
                settings,
            );
            grids_drawn += 1;
            continue;
        }

        // S4B.2.e: Approach B render path. See `render_scene`'s
        // body for the camera transform + ChunkGrid construction
        // commentary; the only difference is this writes to
        // (temp_fb, temp_zb) and composes via `compose_into`.
        // S5.0: per-grid rotation flows via the shared helper.
        //
        // DDA.7: refresh the cross-frame brick cache (needs `&mut grid`)
        // before the immutable `backing` borrow. Render mip by LOD tier:
        // Near = full detail, Mid = coarser (clamped to built mips).
        // Mid tier: coarsen by the grid's `mid_mip_levels` override
        // (a level count → uniform DDA mip `n-1`). No override ⇒ mip
        // 0, i.e. byte-identical to Near (the override is opt-in).
        // Effective DDA mip: the brick cache was ensured in phase A; reuse the
        // mip it resolved (Near/Mid grids are recorded; default 0 otherwise).
        let dda_eff_mip = eff_mips.get(&grid_id).copied().unwrap_or(0);
        let Some(backing) = grid.chunk_xyz_backing() else {
            continue;
        };

        // Out-of-range early-out: skip the per-grid opticast pass
        // when the grid's bounding sphere is entirely beyond
        // `max_scan_dist`. Each opticast call walks ~width*height
        // rays even when no ray reaches a voxel, so far-away marker
        // pillars / pickups otherwise cost ~9 ms each at the bench
        // pose. Safe: if the closest point of the sphere is past
        // max_scan_dist, no ray can possibly reach the grid, so
        // dropping the opticast pass is byte-identical.
        //
        // `grid_bounds` walks `grid.chunks.keys()`; for the ground's
        // ~1024 chunks it costs ~10 µs amortised against the ~50 ms
        // it might save by culling 4-of-5 markers in the live demo.
        let bounds = billboard::grid_bounds(grid);
        let centre_world = grid.transform.origin + grid.transform.rotation * bounds.centre;
        let cam_pos = DVec3::from_array(camera.pos);
        let dist_to_centre = (centre_world - cam_pos).length();
        if dist_to_centre - bounds.radius > f64::from(settings.max_scan_dist) {
            continue;
        }

        // Per-grid screen-space scissor: confine this grid's opticast +
        // temp reset + compose to the true screen rect its projection
        // spans, and skip the grid entirely when it projects fully
        // off-screen on either axis. `project_sphere_to_screen` is
        // conservative (over-estimates the footprint), so the rendered
        // pixels stay byte-identical to the full-frame path — only the
        // work shrinks.
        //
        // PF.13 (C7) — the horizontal extent now clips the render too.
        // The historical full-width-only constraint guarded the deleted
        // voxlap radar's column-indexed `angstart` (never reset per
        // grid, so x-clipping read stale entries at extreme poses); the
        // DDA renderer's pixels are fully independent, so the x band is
        // as safe as the long-proven y band. Small grids (markers,
        // pickups, ships) stop paying full-width rows of render + fill
        // + compose. `None` (camera inside/near the sphere) renders
        // full-frame; `scissor = false` disables it all for the
        // byte-identity regression test.
        let full_rect = ScreenRect {
            x0: 0,
            x1: width,
            y0: 0,
            y1: height,
        };
        let rect = if scissor {
            match project_sphere_to_screen(camera, centre_world, bounds.radius, settings) {
                // Off-screen on either axis → the grid can't appear.
                Some(r) if r.is_empty() => continue,
                Some(r) => r,
                None => full_rect,
            }
        } else {
            full_rect
        };

        // S5.2-followup: per-grid sky opt-out. Grids with
        // `render_sky = false` (e.g. a rotating ship) must not
        // contribute sky pixels — the grid-local sky lookup
        // rotates with the grid and visibly fights the world's
        // sky during compose. Implementation: stamp a sentinel
        // colour into temp_fb everywhere the rasterizer would
        // paint sky, then walk the buffer post-opticast and
        // mark sentinel pixels as `INFINITY` in temp_zb so
        // [`compose_into`]'s min-z test always drops them.
        let owns_sky = grid.render_sky;
        let local_sky_color = if owns_sky {
            sky_color
        } else {
            SKY_MASK_SENTINEL
        };

        // Reset temp to sky / INFINITY so each grid starts fresh —
        // only within the grid's screen rect (opticast writes nothing
        // outside it, and the rect-limited compose reads nothing there).
        fill_rect_u32(temp_fb, pitch_pixels, rect, local_sky_color);
        fill_rect_f32(temp_zb, pitch_pixels, rect, f32::INFINITY);

        let local_cam = world_camera_to_grid_local(camera, &grid.transform);
        let cg = roxlap_core::ChunkGrid {
            chunks: &backing.chunks,
            origin_chunk_xy: backing.origin_chunk_xy,
            origin_chunk_z: backing.origin_chunk_z,
            chunks_x: backing.chunks_x,
            chunks_y: backing.chunks_y,
            chunks_z: backing.chunks_z,
        };
        let grid_view = single_chunk_fast_path(&backing, &cg);

        // Build the per-grid settings by layering three opt-in
        // overrides on top of the caller's `settings`:
        //
        //   1. (S6.1) `lod_thresholds.mid_mip_levels` /
        //      `mid_mip_scan_dist` — applied iff `lod == Mid`.
        //      Biases the grid into coarser mips via the existing
        //      multi-mip path. None ⇒ Mid degrades to Near's
        //      settings (graceful).
        //   2. (S5.2-followup) `Grid::mip_levels_override` — global
        //      per-grid cap applied at ALL tiers. Preserves the
        //      ship anti-axis-aligned-beam workaround through Mid
        //      tier (so a rotating ship pinned at mip-0 stays at
        //      mip-0 even when distant).
        //
        // Layer order: Mid overrides first, then global cap. Both
        // mip_levels overrides are clamped to `[1, base.mip_levels]`
        // since the base is the maximum the renderer can use
        // (chunk's `chunk_mips`-min logic inside scalar_rasterizer
        // applies further per-chunk).
        let per_grid_settings;
        let active_settings = {
            let base_mip_levels = settings.mip_levels;
            let base_mip_scan = settings.mip_scan_dist;
            let lod_mip_levels = match lod {
                Lod::Mid => grid.lod_thresholds.mid_mip_levels,
                Lod::Near | Lod::Far => None,
            };
            let lod_mip_scan = match lod {
                Lod::Mid => grid.lod_thresholds.mid_mip_scan_dist,
                Lod::Near | Lod::Far => None,
            };
            let global_mip_cap = grid.mip_levels_override;
            let needs_override =
                lod_mip_levels.is_some() || lod_mip_scan.is_some() || global_mip_cap.is_some();
            if needs_override {
                // Resolve mip_levels: start with base, apply LOD
                // override (clamped to base), then apply global cap.
                let mut mip_levels =
                    lod_mip_levels.map_or(base_mip_levels, |n| n.clamp(1, base_mip_levels));
                if let Some(cap) = global_mip_cap {
                    mip_levels = mip_levels.min(cap.clamp(1, base_mip_levels));
                }
                // Resolve mip_scan_dist: LOD override clamps to
                // `min(base, override)` — the override only makes
                // transitions kick in CLOSER, never farther. The
                // renderer floors at 4 internally so we don't
                // bottom-clamp here.
                let mip_scan_dist = lod_mip_scan.map_or(base_mip_scan, |d| base_mip_scan.min(d));
                per_grid_settings = OpticastSettings {
                    mip_levels,
                    mip_scan_dist,
                    ..*settings
                };
                &per_grid_settings
            } else {
                settings
            }
        };

        // PF.13 (C7) — 2D scissor: restrict the render to the grid's
        // true screen rect. The y strip is the long-proven path; the x
        // band joins it now that the radar-era `angstart` fragility is
        // gone (see the rect computation above). Byte-identical to the
        // full frame when the rect is `0..width × 0..height`.
        let scissored = (*active_settings)
            .with_y_range(rect.y0, rect.y1)
            .with_x_range(rect.x0, rect.x1);
        // DDA backend. temp_fb / temp_zb are already pre-filled with
        // sky / INFINITY for this grid's rect, so a miss with no
        // textured sky yields the correct solid sky.
        //
        // Fog is config-driven: on iff the caller set `max_scan_dist > 0`
        // in `fog`. Off → no blend, so exact-colour tests and unfogged
        // hosts are unaffected. Linear ramp toward the configured fog
        // colour over `max_scan_dist`. Sky texture is suppressed for
        // `!owns_sky` grids so the textured-sky branch doesn't bypass
        // the sentinel.
        let fog_on = fog.max_scan_dist > 0;
        // CPU.1 — transform the world lights into this grid's local frame
        // (the reused point scratch lives for the grid's render below).
        // PF.7 — lights that can't reach the grid's bounding sphere are
        // culled (`bounds`/`centre_world` computed for the distance cull
        // above).
        let local_lights = grid_local_lights(
            &lights,
            &grid.transform,
            &mut scratch.lights,
            Some((centre_world, bounds.radius)),
        );
        // XS.1 — cross-grid shadows: hand the shade the scene-wide occluder
        // plus this grid's local→world transform, so a grid-local shadow ray
        // is lifted to world space and tested against every grid. `cols[i]`
        // is the world image of grid-local axis `i` (the rotation's columns).
        let world_shadow = active_occluder.map(|occ| {
            let r = grid.transform.rotation;
            let col = |v: DVec3| {
                let w = r * v;
                [w.x as f32, w.y as f32, w.z as f32]
            };
            let o = grid.transform.origin;
            WorldShadowCtx {
                occluder: occ,
                origin: [o.x as f32, o.y as f32, o.z as f32],
                cols: [col(DVec3::X), col(DVec3::Y), col(DVec3::Z)],
            }
        });
        #[allow(clippy::cast_precision_loss)]
        let env = DdaEnv {
            sky: if owns_sky { sky } else { None },
            fog_color: if fog_on { fog.color } else { 0 },
            fog_max_dist: if fog_on {
                fog.max_scan_dist.max(1) as f32
            } else {
                0.0
            },
            side_shades: fog.side_shades,
            materials,
            terrain_materials,
            lights: local_lights,
            world_shadow,
        };
        // Effective render mip + brick cache were prepared above
        // (DDA.6 uniform per-grid mip, DDA.7 cross-frame cache).
        render_dda_parallel(
            &local_cam,
            &scissored,
            grid_view,
            temp_fb,
            temp_zb,
            pitch_pixels,
            &env,
            &grid.dda_brick_cache,
            dda_eff_mip,
        );

        if !owns_sky {
            // Mask sentinel pixels so compose drops them — only within
            // the grid's rect (opticast wrote nothing outside it).
            for y in rect.y0..rect.y1 {
                let row = y as usize * pitch_pixels;
                for i in row + rect.x0 as usize..row + rect.x1 as usize {
                    if temp_fb[i] == SKY_MASK_SENTINEL {
                        temp_zb[i] = f32::INFINITY;
                    }
                }
            }
        }

        compose_rect(fb, zb, temp_fb, temp_zb, pitch_pixels, rect);
        grids_drawn += 1;
    }

    if grids_drawn == 0 {
        RenderOutcome::Empty
    } else {
        RenderOutcome::Rendered { grids_drawn }
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::{GridTransform, Scene, CHUNK_SIZE_XY};
    use glam::{DVec3, IVec3};
    use roxlap_core::opticast::OpticastSettings;
    use roxlap_core::{Camera, Engine};

    const XRES: u32 = 320;
    const YRES: u32 = 200;

    /// Build a single-grid scene at the given world origin with a
    /// recognisable shape inside its chunk (0, 0, 0): a 16-voxel
    /// box plus a 6-radius sphere. Returns `(scene, grid_id)`.
    fn build_one_grid_scene(world_origin: DVec3) -> (Scene, crate::GridId) {
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::at(world_origin));
        let grid = scene.grid_mut(id).unwrap();
        // Box covering [40..56]³ in chunk-local coords.
        grid.set_rect(
            IVec3::new(40, 40, 40),
            IVec3::new(55, 55, 55),
            Some(0x80_88_88_88),
        );
        // Sphere at (80, 80, 80) radius 6.
        grid.set_sphere(IVec3::new(80, 80, 80), 6, Some(0x80_22_aa_22));
        (scene, id)
    }

    fn camera_at(pos: [f64; 3]) -> Camera {
        // Look +y axis; voxlap z-down convention. Right-handed:
        // right × down == forward.
        Camera {
            pos,
            right: [-1.0, 0.0, 0.0],
            down: [0.0, 0.0, 1.0],
            forward: [0.0, 1.0, 0.0],
        }
    }

    /// Spin up an engine + framebuffers ready for one `render_scene`
    /// pass. `_pool_vsid` is retained for call-site compatibility but
    /// the DDA backend needs no pre-sized scratch pool.
    fn render_setup(_pool_vsid: u32) -> (Engine, Vec<u32>, Vec<f32>) {
        let engine = Engine::new();
        let sky = engine.sky_color();
        let pixel_count = (XRES as usize) * (YRES as usize);
        let framebuffer = vec![sky; pixel_count];
        let zbuffer = vec![0.0f32; pixel_count];
        (engine, framebuffer, zbuffer)
    }

    /// Render `scene` via [`render_scene`] (single-grid no-compose
    /// path) and return the resulting framebuffer.
    fn render_via_scene(scene: &mut Scene, camera: &Camera) -> Vec<u32> {
        let (_engine, mut fb, mut zb) = render_setup(CHUNK_SIZE_XY);
        let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
        let outcome = render_scene(
            &mut fb,
            &mut zb,
            XRES as usize,
            XRES,
            YRES,
            CpuFog::default(),
            scene,
            camera,
            &settings,
            None,
        );
        assert_eq!(outcome, RenderOutcome::Rendered { grids_drawn: 1 });
        fb
    }

    /// XS.1 — cross-grid hard shadows: a block in grid **B** casts a sun
    /// shadow onto the floor of grid **A**. Renders the two-grid scene with
    /// the sun shadow-casting vs not; the shadow only exists if the shadow
    /// ray from A's floor crossed into B, so the shadowed render must be
    /// strictly (and non-trivially) darker.
    #[test]
    fn cross_grid_sun_shadow_darkens_other_grid() {
        // Grid A: a wide floor at world z∈[60,62]. Grid B (same origin): a
        // 10-tall block at x∈[50,60]. Sun grazes from +x and above, so B's
        // shadow lands on A's floor at x≈[40,50] — visible to a straight-down
        // camera (B itself occludes only x∈[50,60]).
        let mut scene = Scene::new();
        let a = scene.add_grid(GridTransform::at(DVec3::ZERO));
        scene.grid_mut(a).unwrap().set_rect(
            IVec3::new(30, 30, 60),
            IVec3::new(90, 90, 62),
            Some(0x80_88_88_88),
        );
        let b = scene.add_grid(GridTransform::at(DVec3::ZERO));
        scene.grid_mut(b).unwrap().set_rect(
            IVec3::new(50, 50, 40),
            IVec3::new(60, 60, 50),
            Some(0x80_60_60_60),
        );

        // Straight-down camera over the floor (voxlap z-down ⇒ forward +z).
        let cam = Camera {
            pos: [55.0, 55.0, 6.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 1.0, 0.0],
            forward: [0.0, 0.0, 1.0],
        };
        let inv = 1.0f32 / 2.0f32.sqrt();
        let base = CpuLights {
            enabled: true,
            sun: true,
            sun_dir: [inv, 0.0, -inv], // to-sun: +x and up
            sun_color: [1.0; 3],
            sun_intensity: 1.0,
            ambient: [0.3; 3],
            shadow_strength: 0.85,
            shadow_bias: 1.5,
            shadow_max_dist: 128.0,
            ..CpuLights::default()
        };
        let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
        let mut sum_lum = |lights: CpuLights| -> u64 {
            let n = (XRES as usize) * (YRES as usize);
            let mut fb = vec![0u32; n];
            let mut zb = vec![f32::INFINITY; n];
            render_scene_composed_scissored(
                &mut fb,
                &mut zb,
                XRES as usize,
                XRES,
                YRES,
                CpuFog::default(),
                &mut scene,
                &cam,
                &settings,
                0x0011_2233,
                None,
                false,
                None,
                &[],
                lights,
                None,
                &mut SceneRenderScratch::default(),
            );
            fb.iter()
                .map(|&p| u64::from((p & 0xff) + ((p >> 8) & 0xff) + ((p >> 16) & 0xff)))
                .sum()
        };
        let lit = sum_lum(CpuLights {
            sun_casts_shadow: false,
            ..base
        });
        let shadowed = sum_lum(CpuLights {
            sun_casts_shadow: true,
            ..base
        });
        assert!(
            shadowed < lit,
            "B's shadow must darken A's floor: shadowed={shadowed} lit={lit}"
        );
        assert!(
            (lit - shadowed) * 200 > lit,
            "cross-grid shadow should remove >0.5% of total luminance: lit={lit} shadowed={shadowed}"
        );
    }

    // ---- S5.0: world_camera_to_grid_local helper ----

    /// Identity rotation: pos translates by `-origin`; basis is
    /// untouched. This is the byte-identical-to-pre-S5 contract.
    #[test]
    fn world_camera_to_grid_local_identity_rotation_translates_pos_only() {
        let camera = Camera {
            pos: [110.0, 220.0, 330.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 0.0, 1.0],
            forward: [0.0, 1.0, 0.0],
        };
        let transform = GridTransform::at(DVec3::new(100.0, 200.0, 300.0));
        let local = super::world_camera_to_grid_local(&camera, &transform);
        // Basis must be bit-for-bit unchanged for the identity case.
        assert_eq!(local.right, camera.right);
        assert_eq!(local.down, camera.down);
        assert_eq!(local.forward, camera.forward);
        // Pos translates by `-origin`.
        for (got, want) in local.pos.iter().zip([10.0, 20.0, 30.0].iter()) {
            assert!((got - want).abs() < 1e-12, "pos got={got} want={want}");
        }
    }

    /// 90° rotation about +Z: grid-local `+x` aligns with world `+y`.
    /// World camera at `(0, 10, 0)` looking world `+y` lives in
    /// grid-local at `(10, 0, 0)` looking grid-local `+x`.
    #[test]
    fn world_camera_to_grid_local_90deg_z_rotates_basis_and_pos() {
        use glam::DQuat;
        let camera = Camera {
            pos: [0.0, 10.0, 0.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 0.0, 1.0],
            forward: [0.0, 1.0, 0.0],
        };
        let transform = GridTransform {
            origin: DVec3::ZERO,
            rotation: DQuat::from_rotation_z(std::f64::consts::FRAC_PI_2),
        };
        let local = super::world_camera_to_grid_local(&camera, &transform);
        // World +y == grid-local +x.
        let approx_eq =
            |a: [f64; 3], b: [f64; 3]| a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < 1e-9);
        assert!(
            approx_eq(local.pos, [10.0, 0.0, 0.0]),
            "pos={:?} expected ~(10, 0, 0)",
            local.pos
        );
        // World +x (right) maps to grid-local -y.
        assert!(
            approx_eq(local.right, [0.0, -1.0, 0.0]),
            "right={:?} expected ~(0, -1, 0)",
            local.right
        );
        // World +z (down) is unchanged — it's the rotation axis.
        assert!(
            approx_eq(local.down, [0.0, 0.0, 1.0]),
            "down={:?} expected ~(0, 0, 1)",
            local.down
        );
        // World +y (forward) maps to grid-local +x.
        assert!(
            approx_eq(local.forward, [1.0, 0.0, 0.0]),
            "forward={:?} expected ~(1, 0, 0)",
            local.forward
        );
    }

    /// Basis orthonormality + handedness both survive the
    /// inverse-rotation transform. Property: any unit-quaternion
    /// conjugation preserves the input basis's orthonormality AND
    /// its handedness (rotations are orientation-preserving).
    #[test]
    fn world_camera_to_grid_local_preserves_basis_orthonormality() {
        use glam::DQuat;
        // Right-handed voxlap basis (`right × down == forward`):
        // looking +y, right = -x makes the cross product land on +y.
        let camera = Camera {
            pos: [3.0, -5.0, 7.0],
            right: [-1.0, 0.0, 0.0],
            down: [0.0, 0.0, 1.0],
            forward: [0.0, 1.0, 0.0],
        };
        let transform = GridTransform {
            origin: DVec3::new(1.0, 2.0, 3.0),
            rotation: DQuat::from_axis_angle(glam::DVec3::new(0.3, 0.8, 0.5).normalize(), 0.7),
        };
        let local = super::world_camera_to_grid_local(&camera, &transform);
        let r = DVec3::from_array(local.right);
        let d = DVec3::from_array(local.down);
        let f = DVec3::from_array(local.forward);
        // Norms ≈ 1.
        for v in [r, d, f] {
            assert!(
                (v.length_squared() - 1.0).abs() < 1e-12,
                "basis vec {v:?} not unit length"
            );
        }
        // Orthogonality.
        assert!(r.dot(d).abs() < 1e-12, "right·down = {}", r.dot(d));
        assert!(r.dot(f).abs() < 1e-12, "right·forward = {}", r.dot(f));
        assert!(d.dot(f).abs() < 1e-12, "down·forward = {}", d.dot(f));
        // Right-handed: right × down == forward (voxlap convention).
        let cross = r.cross(d);
        assert!(
            (cross - f).length() < 1e-12,
            "right×down={cross:?} forward={f:?}"
        );
    }

    // ---- S5.1: rotated-grid render correctness ----

    /// Build a single-grid scene at the given transform with a
    /// marker box near one corner of chunk (0, 0, 0). Returns the
    /// scene and the marker colour. Picking a single chunk + small
    /// box keeps the test compact while still exercising the gline
    /// + grouscan path through the rotated frame.
    fn build_one_grid_marker_scene(transform: GridTransform) -> (Scene, crate::GridId, u32) {
        let mut scene = Scene::new();
        let id = scene.add_grid(transform);
        let grid = scene.grid_mut(id).unwrap();
        // Bright marker box at chunk-local (40..56, 40..56, 40..56).
        grid.set_rect(
            IVec3::new(40, 40, 40),
            IVec3::new(55, 55, 55),
            Some(0x80_55_aa_22), // distinctive green
        );
        (scene, id, 0x80_55_aa_22)
    }

    /// Pin S5.1's central equivalence: rotating both the grid and the
    /// camera by the SAME rotation around the grid's origin must
    /// leave the rendered framebuffer unchanged — the grid-local
    /// camera pose collapses to the same values in both scenarios.
    ///
    /// We use `DQuat::from_xyzw(0.0, 0.0, 1.0, 0.0)`, the
    /// 180°-around-Z unit quaternion. This rotation acts on vectors
    /// as `(x, y, z) → (-x, -y, z)`, which only multiplies f64
    /// components by 0 or ±1 — bit-exact under glam's standard quat
    /// conjugation formula. Other angles (e.g. 90°) would introduce
    /// sub-1e-15 noise from sin/cos, breaking byte-identity at
    /// chunk / voxel boundaries.
    #[test]
    fn s5_1_180deg_z_rotated_grid_byte_identical_to_axis_aligned() {
        use glam::DQuat;
        // Right-handed voxlap basis (right × down == forward).
        let axis_aligned_camera = Camera {
            pos: [40.0, -20.0, 50.0],
            right: [-1.0, 0.0, 0.0],
            down: [0.0, 0.0, 1.0],
            forward: [0.0, 1.0, 0.0],
        };
        // R_z(180°): (x, y, z) → (-x, -y, z).
        let rotated_camera = Camera {
            pos: [-40.0, 20.0, 50.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 0.0, 1.0],
            forward: [0.0, -1.0, 0.0],
        };
        // Sanity: prove the exact-arithmetic rotation lands on the
        // baseline. If glam ever changes its quat*vec formula in a
        // way that loses exactness here, the next two assertions
        // catch it before the framebuffer comparison.
        let q = DQuat::from_xyzw(0.0, 0.0, 1.0, 0.0);
        let rot_pos = q * DVec3::from_array(axis_aligned_camera.pos);
        let rot_fwd = q * DVec3::from_array(axis_aligned_camera.forward);
        assert_eq!(rot_pos.to_array(), rotated_camera.pos);
        assert_eq!(rot_fwd.to_array(), rotated_camera.forward);

        let (mut scene_a, _, _) = build_one_grid_marker_scene(GridTransform::identity());
        let fb_a = render_via_scene(&mut scene_a, &axis_aligned_camera);

        let (mut scene_b, _, _) = build_one_grid_marker_scene(GridTransform {
            origin: DVec3::ZERO,
            rotation: q,
        });
        let fb_b = render_via_scene(&mut scene_b, &rotated_camera);

        assert_eq!(
            fb_a, fb_b,
            "rotating both grid and camera by R about the grid origin must leave the framebuffer unchanged"
        );
    }

    /// 45° smoke test: rotated grid renders to something non-trivial
    /// without panicking. No equivalence assertion (45° quat math is
    /// approximate at f64 level; that path is exercised structurally,
    /// not bit-exactly). Camera is placed at a fixed world pose where
    /// — under the rotation — the marker box stays inside the view
    /// frustum.
    #[test]
    fn s5_1_45deg_z_rotated_grid_renders_marker() {
        use glam::DQuat;
        let rotation = DQuat::from_rotation_z(std::f64::consts::FRAC_PI_4);
        let (mut scene, _, marker) = build_one_grid_marker_scene(GridTransform {
            origin: DVec3::ZERO,
            rotation,
        });

        // World position of the marker's centre. Grid-local
        // (47.5, 47.5, 47.5) → world `rotation * (47.5, 47.5, 47.5)`.
        // R_z(45°): (47.5, 47.5, 47.5) → (0, 67.18, 47.5) (the x/y
        // components combine into a single +y vector at √2 * 47.5).
        let marker_world = rotation * DVec3::new(47.5, 47.5, 47.5);
        // Camera 80 units south of the marker on the world Y axis,
        // looking +y at the same z. RH basis.
        let camera = Camera {
            pos: [marker_world.x, marker_world.y - 80.0, marker_world.z],
            right: [-1.0, 0.0, 0.0],
            down: [0.0, 0.0, 1.0],
            forward: [0.0, 1.0, 0.0],
        };

        let (_engine, mut fb, mut zb) = render_setup(CHUNK_SIZE_XY);
        let fog = CpuFog::default();
        let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
        let outcome = render_scene(
            &mut fb,
            &mut zb,
            XRES as usize,
            XRES,
            YRES,
            fog,
            &mut scene,
            &camera,
            &settings,
            None,
        );
        assert_eq!(outcome, RenderOutcome::Rendered { grids_drawn: 1 });
        let marker_count = fb.iter().filter(|&&p| p == marker).count();
        assert!(
            marker_count > 50,
            "45°-rotated marker box should be visible — got {marker_count} marker pixels"
        );
    }

    // ---- S5.2-followup: per-grid render_sky opt-out ----

    /// Two-grid scene where grid B sits behind grid A along +y;
    /// grid A is opaque only in the centre of the framebuffer, so
    /// the camera's view through grid A is mostly "ray miss". When
    /// `A.render_sky = false`, the pixels around A's silhouette
    /// must remain whatever grid B (or the shared pre-fill)
    /// painted — NOT A's grid-local sky colour. This pins the
    /// sentinel-mask path: without it, A's sky would write into
    /// the composed framebuffer wherever its sky-z happened to win
    /// the min-z race with B's sky-z.
    #[test]
    fn render_sky_false_drops_grid_sky_pixels() {
        use crate::{GridId, GridTransform};

        // Grid B (far, sky owner) — a wide floor of distinct
        // colour spanning chunk-local x/y so most rays land on it.
        let mut scene = Scene::new();
        let _b_id: GridId = scene.add_grid(GridTransform::at(DVec3::new(0.0, 600.0, 0.0)));
        // Find grid B's id (HashMap iteration; we only just added
        // one grid, so its id is whichever the iterator yields).
        let b_id = scene.grids().next().unwrap().0;
        scene.grid_mut(b_id).unwrap().set_rect(
            IVec3::new(0, 0, 100),
            IVec3::new(127, 127, 110),
            Some(0x80_22_88_22), // green floor
        );

        // Grid A (near, sky disabled) — a SMALL marker box that
        // covers only a fraction of the screen. Most pixels of A's
        // local render are sky.
        let a_id = scene.add_grid(GridTransform::at(DVec3::new(0.0, 200.0, 0.0)));
        scene.grid_mut(a_id).unwrap().set_rect(
            IVec3::new(60, 60, 60),
            IVec3::new(67, 67, 67),
            Some(0x80_aa_22_22), // red cube
        );
        scene.grid_mut(a_id).unwrap().render_sky = false;

        let unique_sky: u32 = 0xFF_AB_CD_EF;
        let (_engine, fog, _) = make_composed_pool(CHUNK_SIZE_XY);
        let mut fb = vec![unique_sky; pixel_count(XRES, YRES)];
        let mut zb = vec![f32::INFINITY; pixel_count(XRES, YRES)];
        let camera = camera_at([64.0, 0.0, 100.0]);
        let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
        let outcome = render_scene_composed(
            &mut fb,
            &mut zb,
            XRES as usize,
            XRES,
            YRES,
            fog,
            &mut scene,
            &camera,
            &settings,
            unique_sky,
            None,
        );
        assert_eq!(outcome, RenderOutcome::Rendered { grids_drawn: 2 });

        // The sentinel must never appear in the composed output —
        // every sentinel pixel must have been masked out before
        // compose. If any leak through, the test catches it.
        let leaked = fb
            .iter()
            .filter(|&&p| p == super::SKY_MASK_SENTINEL)
            .count();
        assert_eq!(
            leaked, 0,
            "SKY_MASK_SENTINEL leaked into composed framebuffer ({leaked} pixels)"
        );
        // Grid A's hit (red cube) must still render — render_sky=false
        // only affects sky pixels, not hits.
        let red_count = fb.iter().filter(|&&p| p == 0x80_aa_22_22).count();
        assert!(
            red_count > 0,
            "red cube from sky-disabled grid A is missing — render_sky=false should only mask sky"
        );
        // Grid B's floor must be visible past grid A's silhouette
        // (the sky-disabled grid doesn't hide B's render).
        let green_count = fb.iter().filter(|&&p| p == 0x80_22_88_22).count();
        assert!(
            green_count > 0,
            "grid B's floor invisible — grid A's masked sky may have overwritten it"
        );
    }

    /// Identity-rotation, single-grid scene with `render_sky = false`
    /// must produce a sentinel-free framebuffer. Sanity test for the
    /// trivial 1-grid case (no second grid to compose against).
    #[test]
    fn render_sky_false_single_grid_no_sentinel_leak() {
        let (mut scene, id, _) = build_one_grid_marker_scene(GridTransform::identity());
        scene.grid_mut(id).unwrap().render_sky = false;
        let unique_sky: u32 = 0xFF_12_34_56;
        let (_engine, fog, _) = make_composed_pool(CHUNK_SIZE_XY);
        let mut fb = vec![unique_sky; pixel_count(XRES, YRES)];
        let mut zb = vec![f32::INFINITY; pixel_count(XRES, YRES)];
        let camera = camera_at([64.0, 0.0, 64.0]);
        let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
        let outcome = render_scene_composed(
            &mut fb,
            &mut zb,
            XRES as usize,
            XRES,
            YRES,
            fog,
            &mut scene,
            &camera,
            &settings,
            unique_sky,
            None,
        );
        assert_eq!(outcome, RenderOutcome::Rendered { grids_drawn: 1 });
        let leaked = fb
            .iter()
            .filter(|&&p| p == super::SKY_MASK_SENTINEL)
            .count();
        assert_eq!(leaked, 0, "SKY_MASK_SENTINEL leaked ({leaked} pixels)");
        // Pixels that would have been the grid's sky now show
        // through to the pre-fill (unique_sky).
        let prefill_count = fb.iter().filter(|&&p| p == unique_sky).count();
        assert!(
            prefill_count > 0,
            "no pre-fill pixels survived — render_sky=false should leave non-hit pixels untouched"
        );
    }

    // DDA.9: `render_scene_at_origin_matches_direct_opticast` and
    // `render_scene_translated_grid_matches_grid_local_opticast` were
    // removed — they asserted the scene render byte-matches voxlap
    // `opticast`, which no longer holds now that the scene's CPU backend
    // is the DDA renderer (different, intentionally non-bit-exact). The
    // grid-local camera transform they also exercised is covered by the
    // `stacked_*` / two-grid composition tests below.

    #[test]
    fn empty_scene_returns_empty_outcome() {
        let mut scene = Scene::new();
        let (_engine, mut fb, mut zb) = render_setup(CHUNK_SIZE_XY);
        let fog = CpuFog::default();
        let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
        let outcome = render_scene(
            &mut fb,
            &mut zb,
            XRES as usize,
            XRES,
            YRES,
            fog,
            &mut scene,
            &camera_at([0.0, 0.0, 0.0]),
            &settings,
            None,
        );
        assert_eq!(outcome, RenderOutcome::Empty);
    }

    // ---- S3.1 / S4.0: render_scene_composed + 2-grid composition ----

    /// Build a 2-grid scene with two distinguishable boxes placed
    /// side-by-side in world space along the camera's right axis.
    /// Each grid holds one chunk (`(0, 0, 0)`) containing a single
    /// 16-voxel box with a uniquely-coloured surface so the
    /// composited framebuffer is partitionable by colour.
    fn build_two_grid_side_by_side() -> (Scene, u32, u32) {
        let mut scene = Scene::new();
        // Grid 0 at world (0, 200, 0): box centred chunk-local (64, 64, 100).
        let g0 = scene.add_grid(GridTransform::at(DVec3::new(0.0, 200.0, 0.0)));
        scene.grid_mut(g0).unwrap().set_rect(
            IVec3::new(56, 56, 92),
            IVec3::new(71, 71, 107),
            Some(0x80_88_22_22), // dark red
        );
        // Grid 1 at world (200, 200, 0): box centred chunk-local (64, 64, 100).
        let _g1 = scene.add_grid(GridTransform::at(DVec3::new(200.0, 200.0, 0.0)));
        // Borrow-checker dance: re-borrow grid 1 mutably.
        let g1_id = scene
            .grids()
            .filter(|(id, _)| *id != g0)
            .map(|(id, _)| id)
            .next()
            .unwrap();
        scene.grid_mut(g1_id).unwrap().set_rect(
            IVec3::new(56, 56, 92),
            IVec3::new(71, 71, 107),
            Some(0x80_22_22_88), // dark blue
        );
        (scene, 0x80_88_22_22, 0x80_22_22_88)
    }

    /// Engine + default (off) fog config + sky colour for the
    /// composed-render tests. `_pool_vsid` retained for call-site
    /// compatibility; the DDA backend needs no scratch pool.
    fn make_composed_pool(_pool_vsid: u32) -> (Engine, CpuFog, u32) {
        let engine = Engine::new();
        let sky_color = engine.sky_color();
        (engine, CpuFog::default(), sky_color)
    }

    fn pixel_count(width: u32, height: u32) -> usize {
        (width as usize) * (height as usize)
    }

    #[test]
    fn compose_into_takes_smaller_z() {
        let mut shared_fb = vec![0xff_ff_ff_ff_u32; 4];
        let mut shared_zb = vec![10.0f32; 4];
        let temp_fb = [0xaa_aa_aa_aa, 0x11_22_33_44, 0x55_66_77_88, 0xde_ad_be_ef];
        let temp_zb = [5.0f32, 20.0, 10.0, f32::INFINITY];
        compose_into(&mut shared_fb, &mut shared_zb, &temp_fb, &temp_zb);
        // i=0: 5 < 10 → take temp.
        assert_eq!(shared_fb[0], 0xaa_aa_aa_aa);
        assert_eq!(shared_zb[0], 5.0);
        // i=1: 20 > 10 → keep shared.
        assert_eq!(shared_fb[1], 0xff_ff_ff_ff);
        assert_eq!(shared_zb[1], 10.0);
        // i=2: 10 == 10 → keep shared (`<` not `<=`).
        assert_eq!(shared_fb[2], 0xff_ff_ff_ff);
        // i=3: INFINITY > 10 → keep shared.
        assert_eq!(shared_fb[3], 0xff_ff_ff_ff);
    }

    #[test]
    fn render_scene_composed_two_grids_both_visible() {
        // Camera positioned to see both grids' boxes. Grid 0's box
        // at world (~64, ~264, ~100); grid 1's box at world
        // (~264, ~264, ~100). Camera at world (160, 100, 100)
        // looking +y centres both in view.
        let (mut scene, red, blue) = build_two_grid_side_by_side();
        let (_engine, fog, sky_color) = make_composed_pool(CHUNK_SIZE_XY);
        let mut fb = vec![sky_color; pixel_count(XRES, YRES)];
        let mut zb = vec![f32::INFINITY; pixel_count(XRES, YRES)];

        let camera = camera_at([160.0, 100.0, 100.0]);
        let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
        let outcome = render_scene_composed(
            &mut fb,
            &mut zb,
            XRES as usize,
            XRES,
            YRES,
            fog,
            &mut scene,
            &camera,
            &settings,
            sky_color,
            None,
        );
        assert_eq!(outcome, RenderOutcome::Rendered { grids_drawn: 2 });

        // Both colours should appear somewhere in the framebuffer.
        let red_count = fb.iter().filter(|&&p| p == red).count();
        let blue_count = fb.iter().filter(|&&p| p == blue).count();
        assert!(
            red_count > 0,
            "no red pixels: grid 0 (red box) not visible after compose"
        );
        assert!(
            blue_count > 0,
            "no blue pixels: grid 1 (blue box) not visible after compose"
        );
    }

    /// The per-grid screen scissor (vertical band + lateral/vertical
    /// off-screen cull + rect-limited memory passes) must be a pure
    /// speed-up: rendering a multi-grid scene with it on
    /// (`render_scene_composed`) must produce a **byte-identical**
    /// framebuffer to rendering each grid full-frame
    /// (`scissor = false`). Includes a third grid placed off the left
    /// edge but within scan distance, so the lateral cull (scissor on)
    /// vs a sky-only full render (scissor off) must still agree pixel
    /// for pixel.
    #[test]
    fn scissor_render_is_byte_identical_to_full_frame() {
        let (mut scene, red, blue) = build_two_grid_side_by_side();
        // Third grid far to the +x side at the camera's depth: within
        // max_scan_dist (so the distance cull doesn't fire) but its box
        // projects off the left screen edge → screen-culled with the
        // scissor, sky-only when rendered full-frame.
        let g2 = scene.add_grid(GridTransform::at(DVec3::new(700.0, 130.0, 0.0)));
        let g2_id = scene
            .grids()
            .map(|(id, _)| id)
            .max_by_key(|id| id.raw())
            .unwrap();
        let _ = g2;
        scene.grid_mut(g2_id).unwrap().set_rect(
            IVec3::new(56, 56, 92),
            IVec3::new(71, 71, 107),
            Some(0x80_22_88_22), // green — must never appear (off-screen)
        );

        let camera = camera_at([160.0, 100.0, 100.0]);
        let render = |scene: &mut Scene, scissor: bool| -> Vec<u32> {
            let (_engine, fog, sky_color) = make_composed_pool(CHUNK_SIZE_XY);
            let mut fb = vec![sky_color; pixel_count(XRES, YRES)];
            let mut zb = vec![f32::INFINITY; pixel_count(XRES, YRES)];
            let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
            render_scene_composed_scissored(
                &mut fb,
                &mut zb,
                XRES as usize,
                XRES,
                YRES,
                fog,
                scene,
                &camera,
                &settings,
                sky_color,
                None,
                scissor,
                None,
                &[],
                CpuLights::default(),
                None,
                &mut SceneRenderScratch::default(),
            );
            fb
        };

        let scissored = render(&mut scene, true);
        let full = render(&mut scene, false);
        assert_eq!(
            scissored, full,
            "the screen scissor changed the framebuffer — it must be a pure speed-up",
        );
        // Sanity: the scene actually drew content (not a vacuous all-sky
        // match), and the off-screen green grid never appears.
        assert!(scissored.iter().any(|&p| p == red || p == blue));
        assert!(
            !scissored.contains(&0x80_22_88_22),
            "off-screen grid leaked pixels",
        );
    }

    #[test]
    fn render_scene_composed_grid_a_in_front_of_grid_b() {
        // Two grids stacked along +y so grid A (closer) occludes
        // grid B (farther). After composition only grid A's colour
        // should appear on the overlap.
        let mut scene = Scene::new();
        let g_a = scene.add_grid(GridTransform::at(DVec3::new(0.0, 50.0, 0.0)));
        scene.grid_mut(g_a).unwrap().set_rect(
            IVec3::new(56, 56, 92),
            IVec3::new(71, 71, 107),
            Some(0x80_aa_00_00), // red
        );
        let _g_b = scene.add_grid(GridTransform::at(DVec3::new(0.0, 200.0, 0.0)));
        let g_b_id = scene
            .grids()
            .filter(|(id, _)| *id != g_a)
            .map(|(id, _)| id)
            .next()
            .unwrap();
        scene.grid_mut(g_b_id).unwrap().set_rect(
            IVec3::new(56, 56, 92),
            IVec3::new(71, 71, 107),
            Some(0x80_00_00_aa), // blue
        );

        let (_engine, fog, sky_color) = make_composed_pool(CHUNK_SIZE_XY);
        let mut fb = vec![sky_color; pixel_count(XRES, YRES)];
        let mut zb = vec![f32::INFINITY; pixel_count(XRES, YRES)];

        // Camera at (64, -10, 100) looking +y — both boxes line up
        // along the camera's forward axis.
        let camera = camera_at([64.0, -10.0, 100.0]);
        let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
        let outcome = render_scene_composed(
            &mut fb,
            &mut zb,
            XRES as usize,
            XRES,
            YRES,
            fog,
            &mut scene,
            &camera,
            &settings,
            sky_color,
            None,
        );
        assert_eq!(outcome, RenderOutcome::Rendered { grids_drawn: 2 });

        // Red (closer grid) should be visible. Blue (farther grid)
        // may peek around the edges but the central pixels should
        // be red where both boxes project.
        let red_count = fb.iter().filter(|&&p| p == 0x80_aa_00_00).count();
        assert!(
            red_count > 0,
            "expected red pixels (closer box should win z-test)"
        );

        // Reverse the registration order (force grid B drawn first)
        // and verify that's irrelevant — composition is commutative.
        let mut scene2 = Scene::new();
        let g_b2 = scene2.add_grid(GridTransform::at(DVec3::new(0.0, 200.0, 0.0)));
        scene2.grid_mut(g_b2).unwrap().set_rect(
            IVec3::new(56, 56, 92),
            IVec3::new(71, 71, 107),
            Some(0x80_00_00_aa),
        );
        let g_a2 = scene2.add_grid(GridTransform::at(DVec3::new(0.0, 50.0, 0.0)));
        scene2.grid_mut(g_a2).unwrap().set_rect(
            IVec3::new(56, 56, 92),
            IVec3::new(71, 71, 107),
            Some(0x80_aa_00_00),
        );

        let mut fb2 = vec![sky_color; pixel_count(XRES, YRES)];
        let mut zb2 = vec![f32::INFINITY; pixel_count(XRES, YRES)];
        let outcome2 = render_scene_composed(
            &mut fb2,
            &mut zb2,
            XRES as usize,
            XRES,
            YRES,
            fog,
            &mut scene2,
            &camera,
            &settings,
            sky_color,
            None,
        );
        assert_eq!(outcome2, RenderOutcome::Rendered { grids_drawn: 2 });
        assert_eq!(
            fb, fb2,
            "composition should be order-independent — same scene in different add order should produce identical output"
        );
    }

    // ---- S6.1: Mid-tier mip overrides ----

    /// Build a multi-mip-friendly grid: solid floor spanning the
    /// whole chunk at z=100..254 + `generate_mips(3)`. This is the
    /// same setup `vxl_generate_mips_on_set_voxel_chunk_renders`
    /// uses and is known to render at `mip_levels = 3,
    /// mip_scan_dist = 32`.
    ///
    /// Returns `(scene, grid_id)`. The Mid test sets the camera
    /// inside the chunk so chunk-local rays reach the floor at
    /// short distances; that lets the Mid override use
    /// `mip_scan_dist = 16` without busting the ray budget
    /// (`mip_scan_dist * 2^(mip_levels-1) = 16 * 4 = 64` covers the
    /// distance from camera to floor).
    fn build_mip_visible_grid(world_origin: DVec3) -> (Scene, crate::GridId) {
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::at(world_origin));
        let grid = scene.grid_mut(id).unwrap();
        // Solid floor across the entire chunk at z=100..254.
        grid.set_rect(
            IVec3::new(0, 0, 100),
            IVec3::new(127, 127, 254),
            Some(0x80_88_88_88),
        );
        // Build the per-chunk mip ladder so `gmipnum` can grow past 1.
        grid.chunk_mut(IVec3::ZERO).unwrap().generate_mips(3);
        (scene, id)
    }

    /// Render `scene` via composed path with `mip_levels = 3,
    /// mip_scan_dist = 32` — same values the working
    /// `vxl_generate_mips_on_set_voxel_chunk_renders` test uses.
    /// Returns the framebuffer.
    fn render_with_multi_mip(scene: &mut Scene, camera: &Camera) -> Vec<u32> {
        let (_engine, fog, sky_color) = make_composed_pool(CHUNK_SIZE_XY);
        let mut fb = vec![sky_color; pixel_count(XRES, YRES)];
        let mut zb = vec![f32::INFINITY; pixel_count(XRES, YRES)];
        let mut settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
        settings.mip_levels = 3;
        settings.mip_scan_dist = 32;
        let outcome = render_scene_composed(
            &mut fb,
            &mut zb,
            XRES as usize,
            XRES,
            YRES,
            fog,
            scene,
            camera,
            &settings,
            sky_color,
            None,
        );
        assert_eq!(outcome, RenderOutcome::Rendered { grids_drawn: 1 });
        fb
    }

    // DDA.9: `s6_1_mid_overrides_produce_different_framebuffer_than_near`
    // was removed. It encoded voxlap's mip-*transition* semantics
    // (mid_mip_levels=Some(1) caps in-grid mip transitions, differing
    // from Near's mip0→1→2 distance ramp). The DDA renderer uses a
    // *uniform* per-grid mip (no in-grid transition), so Some(1) → mip 0
    // = identical to Near. DDA mip coarsening is covered by
    // `roxlap_core::dda` `mip_render_is_coarse_but_complete`; the LOD-Mid
    // wiring by `s6_1_mid_without_overrides_byte_identical_to_near`.

    /// Mid tier with `mid_mip_levels = None` AND
    /// `mid_mip_scan_dist = None` must produce a byte-identical
    /// framebuffer to Near. This is the graceful-degrade contract
    /// — callers can opt into the Mid plumbing without committing
    /// to a mip override and stay byte-stable.
    #[test]
    fn s6_1_mid_without_overrides_byte_identical_to_near() {
        let camera = camera_at([64.0, 0.0, 64.0]);

        // Scene A: default thresholds → Near.
        let (mut scene_a, _) = build_mip_visible_grid(DVec3::ZERO);
        let fb_near = render_with_multi_mip(&mut scene_a, &camera);

        // Scene B: thresholds force Mid but no mip overrides set.
        let (mut scene_b, b_id) = build_mip_visible_grid(DVec3::ZERO);
        scene_b.grid_mut(b_id).unwrap().lod_thresholds = crate::LodThresholds {
            r_near: 0.0,
            r_mid: f64::INFINITY,
            mid_mip_levels: None,
            mid_mip_scan_dist: None,
        };
        let lod = scene_b
            .grid(b_id)
            .unwrap()
            .select_lod(DVec3::from_array(camera.pos));
        assert_eq!(lod, Lod::Mid);
        let fb_mid = render_with_multi_mip(&mut scene_b, &camera);

        // Byte-identical: Mid with no overrides degrades cleanly.
        assert_eq!(
            fb_near, fb_mid,
            "Mid with both overrides=None must byte-match Near"
        );
    }

    // DDA.9: `s6_1_global_mip_cap_survives_mid_tier` was removed. It
    // pinned voxlap's `mip_levels_override` global cap composing with the
    // Mid override — the ship anti-axis-aligned-beam workaround. The DDA
    // renderer has no axis-aligned mip beam (honest per-cell traversal),
    // so the workaround / global cap is obsolete and the DDA path doesn't
    // consult `mip_levels_override`.

    // ---- S6.3: Far-tier billboard blit ----

    /// Force Far tier via `r_near = 0, r_mid = 0`: any non-zero
    /// camera-to-grid distance lands on `Lod::Far`. Renders a small
    /// grid at world (0, 200, 0) with default-radius thresholds
    /// turned all-Far. The composed framebuffer must contain
    /// non-sky pixels from the impostor blit.
    #[test]
    fn s6_3_far_tier_blits_non_sky_pixels() {
        let (mut scene, id) = build_one_grid_scene(DVec3::new(0.0, 200.0, 0.0));
        scene.grid_mut(id).unwrap().lod_thresholds = crate::LodThresholds {
            r_near: 0.0,
            r_mid: 0.0,
            mid_mip_levels: None,
            mid_mip_scan_dist: None,
        };

        let camera = camera_at([64.0, 0.0, 100.0]);
        let (_engine, fog, sky_color) = make_composed_pool(CHUNK_SIZE_XY);
        let mut fb = vec![sky_color; pixel_count(XRES, YRES)];
        let mut zb = vec![f32::INFINITY; pixel_count(XRES, YRES)];
        let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
        let outcome = render_scene_composed(
            &mut fb,
            &mut zb,
            XRES as usize,
            XRES,
            YRES,
            fog,
            &mut scene,
            &camera,
            &settings,
            sky_color,
            None,
        );
        assert_eq!(outcome, RenderOutcome::Rendered { grids_drawn: 1 });

        // Sanity: picker actually picked Far.
        let lod = scene
            .grid(id)
            .unwrap()
            .select_lod(DVec3::from_array(camera.pos));
        assert_eq!(lod, Lod::Far);

        // Impostor must paint at least some non-sky pixels.
        let non_sky = fb.iter().filter(|&&p| p != sky_color).count();
        assert!(
            non_sky > 0,
            "Far-tier render produced no non-sky pixels — billboard blit not firing"
        );
    }

    /// Lazy populate: cache starts `None`, becomes `Some` after the
    /// first Far render.
    #[test]
    fn s6_3_far_render_lazily_populates_cache() {
        let (mut scene, id) = build_one_grid_scene(DVec3::new(0.0, 200.0, 0.0));
        scene.grid_mut(id).unwrap().lod_thresholds = crate::LodThresholds {
            r_near: 0.0,
            r_mid: 0.0,
            mid_mip_levels: None,
            mid_mip_scan_dist: None,
        };
        assert!(scene.grid(id).unwrap().billboards.is_none());

        let camera = camera_at([64.0, 0.0, 100.0]);
        let (_engine, fog, sky_color) = make_composed_pool(CHUNK_SIZE_XY);
        let mut fb = vec![sky_color; pixel_count(XRES, YRES)];
        let mut zb = vec![f32::INFINITY; pixel_count(XRES, YRES)];
        let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
        let _ = render_scene_composed(
            &mut fb,
            &mut zb,
            XRES as usize,
            XRES,
            YRES,
            fog,
            &mut scene,
            &camera,
            &settings,
            sky_color,
            None,
        );
        let cache = scene
            .grid(id)
            .unwrap()
            .billboards
            .as_ref()
            .expect("Far render should have populated billboards");
        assert_eq!(cache.len(), 26);
    }

    /// Edit invalidates the cache; a subsequent Far render rebuilds.
    #[test]
    fn s6_3_edit_invalidates_then_far_render_rebuilds() {
        let (mut scene, id) = build_one_grid_scene(DVec3::new(0.0, 200.0, 0.0));
        scene.grid_mut(id).unwrap().lod_thresholds = crate::LodThresholds {
            r_near: 0.0,
            r_mid: 0.0,
            mid_mip_levels: None,
            mid_mip_scan_dist: None,
        };
        let camera = camera_at([64.0, 0.0, 100.0]);
        let (_engine, fog, sky_color) = make_composed_pool(CHUNK_SIZE_XY);
        let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);

        // First Far render → cache built.
        let mut fb1 = vec![sky_color; pixel_count(XRES, YRES)];
        let mut zb1 = vec![f32::INFINITY; pixel_count(XRES, YRES)];
        let _ = render_scene_composed(
            &mut fb1,
            &mut zb1,
            XRES as usize,
            XRES,
            YRES,
            fog,
            &mut scene,
            &camera,
            &settings,
            sky_color,
            None,
        );
        assert!(scene.grid(id).unwrap().billboards.is_some());

        // Edit invalidates.
        scene
            .grid_mut(id)
            .unwrap()
            .set_voxel(IVec3::new(70, 70, 70), Some(0x80_aa_aa_22));
        assert!(scene.grid(id).unwrap().billboards.is_none());

        // Second Far render rebuilds.
        let mut fb2 = vec![sky_color; pixel_count(XRES, YRES)];
        let mut zb2 = vec![f32::INFINITY; pixel_count(XRES, YRES)];
        let _ = render_scene_composed(
            &mut fb2,
            &mut zb2,
            XRES as usize,
            XRES,
            YRES,
            fog,
            &mut scene,
            &camera,
            &settings,
            sky_color,
            None,
        );
        assert!(scene.grid(id).unwrap().billboards.is_some());
    }

    /// Hybrid scene: one Near grid + one Far grid. Both must render
    /// visibly; the Far grid via blit, the Near grid via opticast.
    /// Sanity check that the two paths cohabit one
    /// `render_scene_composed` call.
    #[test]
    fn s6_3_near_and_far_grids_in_same_scene() {
        let mut scene = Scene::new();
        // Grid A: stays Near (default thresholds). Solid box at
        // world (-30..-20, 190..210, 50..70).
        let a_id = scene.add_grid(GridTransform::at(DVec3::new(-100.0, 200.0, 0.0)));
        scene.grid_mut(a_id).unwrap().set_rect(
            IVec3::new(70, 0, 50),
            IVec3::new(85, 15, 70),
            Some(0x80_22_88_22), // green
        );
        // Grid B: forced Far. Box at world (~100, 200, 100).
        let b_id = scene.add_grid(GridTransform::at(DVec3::new(100.0, 200.0, 0.0)));
        scene.grid_mut(b_id).unwrap().set_rect(
            IVec3::new(0, 0, 80),
            IVec3::new(20, 20, 110),
            Some(0x80_aa_22_22), // red
        );
        scene.grid_mut(b_id).unwrap().lod_thresholds = crate::LodThresholds {
            r_near: 0.0,
            r_mid: 0.0,
            mid_mip_levels: None,
            mid_mip_scan_dist: None,
        };

        let camera = camera_at([0.0, 0.0, 80.0]);
        // Confirm A is Near, B is Far for this pose.
        assert_eq!(
            scene
                .grid(a_id)
                .unwrap()
                .select_lod(DVec3::from_array(camera.pos)),
            Lod::Near
        );
        assert_eq!(
            scene
                .grid(b_id)
                .unwrap()
                .select_lod(DVec3::from_array(camera.pos)),
            Lod::Far
        );

        let (_engine, fog, sky_color) = make_composed_pool(CHUNK_SIZE_XY);
        let mut fb = vec![sky_color; pixel_count(XRES, YRES)];
        let mut zb = vec![f32::INFINITY; pixel_count(XRES, YRES)];
        let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
        let outcome = render_scene_composed(
            &mut fb,
            &mut zb,
            XRES as usize,
            XRES,
            YRES,
            fog,
            &mut scene,
            &camera,
            &settings,
            sky_color,
            None,
        );
        assert_eq!(outcome, RenderOutcome::Rendered { grids_drawn: 2 });

        // Each grid should contribute visible pixels.
        let non_sky = fb.iter().filter(|&&p| p != sky_color).count();
        assert!(
            non_sky > 20,
            "hybrid scene produced too few non-sky pixels ({non_sky}); one tier may have failed"
        );
    }

    /// Empty grid at Far tier: skipped silently (no panic, no
    /// allocation), `billboards` stays `None`.
    #[test]
    fn s6_3_empty_grid_at_far_is_skipped() {
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::at(DVec3::new(100.0, 200.0, 0.0)));
        scene.grid_mut(id).unwrap().lod_thresholds = crate::LodThresholds {
            r_near: 0.0,
            r_mid: 0.0,
            mid_mip_levels: None,
            mid_mip_scan_dist: None,
        };

        let camera = camera_at([0.0, 0.0, 100.0]);
        let (_engine, fog, sky_color) = make_composed_pool(CHUNK_SIZE_XY);
        let mut fb = vec![sky_color; pixel_count(XRES, YRES)];
        let mut zb = vec![f32::INFINITY; pixel_count(XRES, YRES)];
        let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
        let outcome = render_scene_composed(
            &mut fb,
            &mut zb,
            XRES as usize,
            XRES,
            YRES,
            fog,
            &mut scene,
            &camera,
            &settings,
            sky_color,
            None,
        );
        // No grids contributed.
        assert_eq!(outcome, RenderOutcome::Empty);
        // Cache must NOT have been built for an empty grid.
        assert!(scene.grid(id).unwrap().billboards.is_none());
        // Framebuffer unchanged.
        assert!(fb.iter().all(|&p| p == sky_color));
    }

    // ---- S6.0: LOD picker wired but every tier falls through to Near ----

    /// Threshold-invariance: a grid rendered with the S6 derived
    /// thresholds (`from_radius` of the actual bounding sphere) must
    /// produce a framebuffer byte-identical to the same grid with
    /// default `always_near` thresholds, because S6.0 takes the
    /// `Near` arm of the match for all three tiers. This is the
    /// regression test for the S6.0 contract.
    #[test]
    fn render_scene_composed_lod_threshold_invariance() {
        // Scene A: default thresholds (always_near).
        let (mut scene_a, _a_id) = build_one_grid_scene(DVec3::new(0.0, 200.0, 0.0));
        let cam = camera_at([64.0, 0.0, 100.0]);
        let (_engine, fog, sky_color) = make_composed_pool(CHUNK_SIZE_XY);
        let mut fb_a = vec![sky_color; pixel_count(XRES, YRES)];
        let mut zb_a = vec![f32::INFINITY; pixel_count(XRES, YRES)];
        let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
        let outcome_a = render_scene_composed(
            &mut fb_a,
            &mut zb_a,
            XRES as usize,
            XRES,
            YRES,
            fog,
            &mut scene_a,
            &cam,
            &settings,
            sky_color,
            None,
        );
        assert_eq!(outcome_a, RenderOutcome::Rendered { grids_drawn: 1 });

        // Scene B: thresholds derived from the grid's bounding
        // radius. At this camera distance the grid lands on Mid or
        // Far; if S6.0 ever stops falling through to Near, this test
        // catches the divergence.
        let (mut scene_b, b_id) = build_one_grid_scene(DVec3::new(0.0, 200.0, 0.0));
        let radius = scene_b.grid(b_id).unwrap().bounding_radius();
        assert!(
            radius > 0.0,
            "bounding_radius should be > 0 for a populated grid"
        );
        scene_b.grid_mut(b_id).unwrap().lod_thresholds = crate::LodThresholds::from_radius(radius);
        // Sanity: the camera is far enough that the picker no longer
        // returns Near (otherwise the invariance test would be vacuous).
        let lod = scene_b
            .grid(b_id)
            .unwrap()
            .select_lod(DVec3::from_array(cam.pos));
        assert_ne!(
            lod,
            Lod::Near,
            "camera should land in Mid or Far for derived thresholds — got {lod:?}",
        );

        let mut fb_b = vec![sky_color; pixel_count(XRES, YRES)];
        let mut zb_b = vec![f32::INFINITY; pixel_count(XRES, YRES)];
        let outcome_b = render_scene_composed(
            &mut fb_b,
            &mut zb_b,
            XRES as usize,
            XRES,
            YRES,
            fog,
            &mut scene_b,
            &cam,
            &settings,
            sky_color,
            None,
        );
        assert_eq!(outcome_b, RenderOutcome::Rendered { grids_drawn: 1 });

        // Byte-identity is the S6.0 contract — Mid/Far still take
        // the Near arm.
        assert_eq!(
            fb_a, fb_b,
            "S6.0 framebuffer must be byte-identical regardless of LOD thresholds"
        );
    }

    #[test]
    fn render_scene_composed_empty_scene_returns_empty() {
        let mut scene = Scene::new();
        let (_engine, fog, sky_color) = make_composed_pool(CHUNK_SIZE_XY);
        let mut fb = vec![sky_color; pixel_count(XRES, YRES)];
        let mut zb = vec![f32::INFINITY; pixel_count(XRES, YRES)];
        let camera = camera_at([0.0, 0.0, 0.0]);
        let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
        let outcome = render_scene_composed(
            &mut fb,
            &mut zb,
            XRES as usize,
            XRES,
            YRES,
            fog,
            &mut scene,
            &camera,
            &settings,
            sky_color,
            None,
        );
        assert_eq!(outcome, RenderOutcome::Empty);
        // fb should be unchanged (still all sky).
        assert!(fb.iter().all(|&p| p == sky_color));
    }

    /// FNV-1a 64-bit hash. Same offset/prime as the
    /// `roxlap-oracle::fnv1a64` helper used by the wasm-render
    /// goldens; pinning a render hash here is the same flavour of
    /// regression catch.
    fn fnv1a64(data: &[u8]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in data {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    // ---- S4.0 cross-chunk smoke test ----

    /// Two-chunk-wide grid: a recognisable shape spans the chunk
    /// boundary at `virtual_x = 128`. The render must not have a
    /// horizontal seam line at the boundary.
    #[test]
    fn render_scene_two_chunk_x_grid_no_seam() {
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::at(DVec3::new(0.0, 200.0, 0.0)));
        let g = scene.grid_mut(id).unwrap();
        // 100-voxel-tall stripe spanning x=[120..136] across the
        // x=128 chunk seam at z=200, y=[60..68]. After bake-free
        // render, every column in the stripe paints the same colour
        // at the same z; a seam at x=128 would show as missing
        // pixels in the column at virtual_x=128 / 129 / ...
        g.set_rect(
            IVec3::new(120, 60, 200),
            IVec3::new(136, 67, 215),
            Some(0x80_aa_55_22),
        );
        // Sanity: ensure both chunks were materialised.
        assert_eq!(g.chunk_count(), 2);

        // Render with a camera positioned to look at the stripe
        // straight on. Stripe at world (120..136, 260..268, 200..215).
        // Camera at (128, 100, 207) looking +y centres on it.
        let (_engine, fog, sky_color) = make_composed_pool(2 * CHUNK_SIZE_XY);
        let mut fb = vec![sky_color; pixel_count(XRES, YRES)];
        let mut zb = vec![f32::INFINITY; pixel_count(XRES, YRES)];
        let camera = camera_at([128.0, 100.0, 207.0]);
        let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
        let outcome = render_scene_composed(
            &mut fb,
            &mut zb,
            XRES as usize,
            XRES,
            YRES,
            fog,
            &mut scene,
            &camera,
            &settings,
            sky_color,
            None,
        );
        assert_eq!(outcome, RenderOutcome::Rendered { grids_drawn: 1 });

        // Stripe colour should appear in roughly the centre of the
        // framebuffer. A chunk-edge seam would manifest as a thin
        // sky-coloured vertical line splitting the stripe in two.
        let stripe = 0x80_aa_55_22;
        let stripe_count = fb.iter().filter(|&&p| p == stripe).count();
        assert!(
            stripe_count > 200,
            "stripe rendered too few pixels ({stripe_count}) — chunks may not be stitching"
        );

        // Walk the centre row left-to-right looking for a sky-pixel
        // gap inside a stripe run. A gap 1+ pixels wide flags a
        // chunk-edge seam.
        let centre_y = (YRES / 2) as usize;
        let row_start = centre_y * (XRES as usize);
        let row = &fb[row_start..row_start + (XRES as usize)];
        let mut in_stripe = false;
        let mut seam_gaps = 0usize;
        for &px in row {
            if px == stripe {
                in_stripe = true;
            } else if in_stripe && px == sky_color {
                // Stripe ended; if we re-enter it on this row that's
                // a seam.
                if row.iter().skip_while(|&&p| p != px).any(|&p| p == stripe) {
                    // Look ahead for any further stripe pixel.
                    seam_gaps += 1;
                }
                in_stripe = false;
            }
        }
        // We allow seam_gaps to count the legitimate "stripe ended,
        // didn't restart" transition once; more than that means
        // multiple disjoint runs on the row → seam.
        assert!(
            seam_gaps <= 1,
            "centre row has {seam_gaps} disjoint stripe runs — expected 1 (chunk-edge seam suspected)"
        );
    }

    // DDA.9: the voxlap-era mip regression tests here
    // (`vxl_generate_mips_on_set_voxel_chunk_renders` + the byte-exact
    // 2-chunk opticast pin) were removed — they drove voxlap `opticast` +
    // `ScalarRasterizer` directly, a path no longer reachable from this
    // consumer crate. The DDA mip ladder + multi-mip render is covered by
    // `render_with_mips_present_still_renders_mip0` and the
    // `stacked_*_multi_mip` tests below.

    /// Mip-0 preservation when mips are generated on the combined
    /// view but `mip_levels = 1` in the rasterizer's settings.
    /// Confirms `generate_mips` only APPENDS data — mip-0
    /// prefix is unchanged.
    #[test]
    fn render_with_mips_present_still_renders_mip0() {
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::at(DVec3::ZERO));
        scene.grid_mut(id).unwrap().set_rect(
            IVec3::new(40, 40, 40),
            IVec3::new(55, 55, 55),
            Some(0x80_88_88_88),
        );
        // S4B.4.a: force mip-1..mip-2 generation on the single
        // chunk directly (the Grid's combined-view cache API was
        // removed). The chunk's own Vxl::generate_mips builds its
        // own mip tables and the renderer happens to render through
        // them via Approach B's chunk_at_xy lookup.
        {
            let grid = scene.grid_mut(id).unwrap();
            let chunk = grid.chunks.get_mut(&IVec3::ZERO).unwrap();
            chunk.generate_mips(3);
        }

        let (_engine, fog, sky_color) = make_composed_pool(CHUNK_SIZE_XY);
        let mut fb = vec![sky_color; pixel_count(XRES, YRES)];
        let mut zb = vec![f32::INFINITY; pixel_count(XRES, YRES)];
        let camera = camera_at([64.0, 0.0, 64.0]);
        // mip_scan_dist huge → renderer never transitions past mip-0
        // so this test pins mip-0 correctness only.
        let mut settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
        settings.mip_scan_dist = 100_000;
        let outcome = render_scene_composed(
            &mut fb,
            &mut zb,
            XRES as usize,
            XRES,
            YRES,
            fog,
            &mut scene,
            &camera,
            &settings,
            sky_color,
            None,
        );
        assert_eq!(outcome, RenderOutcome::Rendered { grids_drawn: 1 });
        let non_sky = fb.iter().filter(|&&p| p != sky_color).count();
        assert!(
            non_sky > 0,
            "render of single-grid scene with mips present rendered all-sky: mip-0 may be corrupted by generate_mips"
        );
    }

    #[test]
    fn render_scene_two_chunk_x_grid_hash_is_stable() {
        // Frozen 2026-05-10 at S4.0 landing on x86_64.
        // DDA.9: re-frozen to the DDA renderer's output (was the
        // voxlap-opticast golden 0x215e_d66d_7359_4725).
        const GOLDEN: u64 = 0x492e_c4bb_718f_d7e5;
        // Same scene shape as `render_scene_two_chunk_x_grid_no_seam`
        // — kept distinct so the hash assertion doesn't share its
        // setup with the structural seam check.
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::at(DVec3::new(0.0, 200.0, 0.0)));
        scene.grid_mut(id).unwrap().set_rect(
            IVec3::new(120, 60, 200),
            IVec3::new(136, 67, 215),
            Some(0x80_aa_55_22),
        );
        let (_engine, fog, sky_color) = make_composed_pool(2 * CHUNK_SIZE_XY);
        let mut fb = vec![sky_color; pixel_count(XRES, YRES)];
        let mut zb = vec![f32::INFINITY; pixel_count(XRES, YRES)];
        let camera = camera_at([128.0, 100.0, 207.0]);
        let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
        let outcome = render_scene_composed(
            &mut fb,
            &mut zb,
            XRES as usize,
            XRES,
            YRES,
            fog,
            &mut scene,
            &camera,
            &settings,
            sky_color,
            None,
        );
        assert_eq!(outcome, RenderOutcome::Rendered { grids_drawn: 1 });

        let bytes: Vec<u8> = fb.iter().flat_map(|p| p.to_ne_bytes()).collect();
        let hash = fnv1a64(&bytes);
        if GOLDEN == SENTINEL {
            // First-run capture mode — print the hash so the
            // developer can paste it into GOLDEN above.
            eprintln!("render_scene_two_chunk_x_grid_hash_is_stable: capture hash = 0x{hash:016x}");
            panic!("GOLDEN is the SENTINEL placeholder — paste 0x{hash:016x} into GOLDEN above");
        }
        assert_eq!(
            hash, GOLDEN,
            "2-chunk render hash drifted: expected 0x{GOLDEN:016x}, got 0x{hash:016x}"
        );
    }

    /// Sentinel for first-run hash capture in
    /// [`render_scene_two_chunk_x_grid_hash_is_stable`]. Replace
    /// `GOLDEN`'s definition with the printed value once captured.
    const SENTINEL: u64 = 0xDEAD_BEEF_DEAD_BEEF;

    /// S4B.6.c: stacked-grid scaffold — camera in chz=1 (= world
    /// z=256..511) of a 2-chunk-tall grid should render its own
    /// chunk's terrain. Verifies cf seed + slab-byte reads + chunk-
    /// XY swaps all use world-z consistently.
    ///
    /// Cross-chunk look-down (= camera in chz=0 sees terrain in
    /// chz=1) needs cf z range extension at air-gap-lookup time;
    /// that's a follow-up to S4B.6.c.
    #[test]
    fn stacked_two_chunk_z_camera_in_chz1_sees_own_chunk_floor() {
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::at(DVec3::ZERO));
        let g = scene.grid_mut(id).unwrap();
        // chz=0: all-air (materialised so chunk_xyz_backing enumerates).
        g.ensure_chunk(IVec3::new(0, 0, 0));
        // chz=1: floor at local z=50 (= world z=306).
        g.set_rect(
            IVec3::new(60, 60, 306),
            IVec3::new(72, 72, 310),
            Some(0x80_33_66_99),
        );
        assert!(g.chunk(IVec3::new(0, 0, 1)).is_some());

        let (_engine, fog, sky_color) = make_composed_pool(2 * CHUNK_SIZE_XY);
        let mut fb = vec![sky_color; pixel_count(XRES, YRES)];
        let mut zb = vec![f32::INFINITY; pixel_count(XRES, YRES)];
        // Camera at world (66, 66, 280) — directly above the
        // floor at world z=306. Look STRAIGHT DOWN (z increases =
        // down in voxlap z-down).
        let camera = Camera {
            pos: [66.0, 66.0, 280.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 1.0, 0.0],
            forward: [0.0, 0.0, 1.0],
        };
        let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
        let outcome = render_scene_composed(
            &mut fb,
            &mut zb,
            XRES as usize,
            XRES,
            YRES,
            fog,
            &mut scene,
            &camera,
            &settings,
            sky_color,
            None,
        );
        assert_eq!(outcome, RenderOutcome::Rendered { grids_drawn: 1 });
        let floor_count = fb.iter().filter(|&&p| p == 0x80_33_66_99).count();
        assert!(
            floor_count > 100,
            "camera at chz=1 with floor in same chunk should see it — got {floor_count} floor pixels"
        );
    }

    /// S4B.6.e: cross-chunk look-down. Camera in chz=0's all-air
    /// chunk should see chz=1's floor below it. This was deferred
    /// from S4B.6.c because the cf seed's z range capped at the
    /// camera-chunk's bedrock (world z=255); S4B.6.e extends the
    /// air-gap walk in `camera_chunk_air_gap` to step into the
    /// next chunk down when the camera's column is all-air-bedrock,
    /// and the rasterizer routes state.column / slab_buf to the
    /// chunk holding the real floor via `seed_chunk_z`.
    #[test]
    fn stacked_two_chunk_z_camera_in_chz0_sees_chz1_floor() {
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::at(DVec3::ZERO));
        let g = scene.grid_mut(id).unwrap();
        // chz=0: all-air. Materialised so chunk_xyz_backing
        // enumerates it.
        g.ensure_chunk(IVec3::new(0, 0, 0));
        // chz=1: floor at world z=306..310 (= local z=50..54).
        g.set_rect(
            IVec3::new(60, 60, 306),
            IVec3::new(72, 72, 310),
            Some(0x80_77_aa_44),
        );
        assert!(g.chunk(IVec3::new(0, 0, 1)).is_some());

        let (_engine, fog, sky_color) = make_composed_pool(2 * CHUNK_SIZE_XY);
        let mut fb = vec![sky_color; pixel_count(XRES, YRES)];
        let mut zb = vec![f32::INFINITY; pixel_count(XRES, YRES)];
        // Camera at world (66, 66, 100) — in chz=0's all-air
        // chunk. Look STRAIGHT DOWN (z+) toward chz=1's floor at
        // world z=306.
        let camera = Camera {
            pos: [66.0, 66.0, 100.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 1.0, 0.0],
            forward: [0.0, 0.0, 1.0],
        };
        let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
        let outcome = render_scene_composed(
            &mut fb,
            &mut zb,
            XRES as usize,
            XRES,
            YRES,
            fog,
            &mut scene,
            &camera,
            &settings,
            sky_color,
            None,
        );
        assert_eq!(outcome, RenderOutcome::Rendered { grids_drawn: 1 });
        let floor_count = fb.iter().filter(|&&p| p == 0x80_77_aa_44).count();
        assert!(
            floor_count > 50,
            "camera in chz=0 air-gap should see chz=1 floor via cross-chunk look-down — got {floor_count} floor pixels"
        );
    }

    /// S4B.6.l KNOWN LIMITATION → RESOLVED by VC.5 (2026-05-31).
    /// Camera at chz=0 with all-air-bedrock at the camera's own
    /// XY column (seed_chz=1 via cross-chunk look-down). A DIFFERENT
    /// XY column has chz=0 content (= a distant mountain entirely
    /// inside chz=0). Pre-VC.5 the chunk-XY swap read chz=1 chunks
    /// across the DDA, so the chz=0 mountain was invisible. VC.5's
    /// multi-chz column-step install stitches every chz layer at the
    /// new XY column; the chz=0 mountain renders correctly.
    ///
    /// VC.0 pin (2026-05-31): re-enabled (was `#[ignore]`'d). VC.5
    /// flipped it from failing (mountain_chz0 = 0) to passing.
    #[test]
    fn stacked_chz0_distant_mountain_visible_from_chz0_camera() {
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::at(DVec3::ZERO));
        let g = scene.grid_mut(id).unwrap();
        // chz=0 mountain at a column DISTANT from the camera —
        // entirely in chz=0 (world z=100..200), so chz=1 at the
        // same XY is all-air-bedrock.
        g.set_rect(
            IVec3::new(100, 100, 100),
            IVec3::new(124, 124, 200),
            Some(0x80_aa_55_22), // distinct brown
        );
        // chz=1 hills filling the floor at world z=336..360 across
        // the chunk EXCEPT a hole around the mountain XY (so the
        // mountain doesn't sit on a green tower).
        g.set_rect(
            IVec3::new(0, 0, 336),
            IVec3::new(128, 128, 360),
            Some(0x80_22_88_44),
        );
        g.set_rect(IVec3::new(100, 100, 336), IVec3::new(124, 124, 360), None);
        // Materialise chz=0 + chz=1 (chz=0 has the mountain; chz=1
        // has the hills).
        assert!(g.chunk(IVec3::new(0, 0, 0)).is_some());
        assert!(g.chunk(IVec3::new(0, 0, 1)).is_some());

        let (_engine, fog, sky_color) = make_composed_pool(CHUNK_SIZE_XY);
        let mut fb = vec![sky_color; pixel_count(XRES, YRES)];
        let mut zb = vec![f32::INFINITY; pixel_count(XRES, YRES)];
        // Camera at (40, 40, 60) — chz=0 air, FAR from the mountain
        // XY (100..124, 100..124). Yaw=π/4 (look toward +x+y =
        // mountain direction), pitch=0.72 rad (≈ 41° down) so the
        // ray bisecting the screen aims at the chz=0 mountain centre
        // ≈ (112, 112, 150).
        let (sy, cy) = (std::f64::consts::FRAC_PI_4).sin_cos();
        let (sp, cp) = 0.72_f64.sin_cos();
        let camera = Camera {
            pos: [40.0, 40.0, 60.0],
            right: [-sy, cy, 0.0],
            down: [-cy * sp, -sy * sp, cp],
            forward: [cy * cp, sy * cp, sp],
        };
        let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
        let outcome = render_scene_composed(
            &mut fb,
            &mut zb,
            XRES as usize,
            XRES,
            YRES,
            fog,
            &mut scene,
            &camera,
            &settings,
            sky_color,
            None,
        );
        assert_eq!(outcome, RenderOutcome::Rendered { grids_drawn: 1 });
        let mountain_count = fb.iter().filter(|&&p| p == 0x80_aa_55_22).count();
        let hill_count = fb.iter().filter(|&&p| p == 0x80_22_88_44).count();
        eprintln!("chz0-distant-mountain: mountain_chz0={mountain_count} hill_chz1={hill_count}");
        // chz=1 hills are reachable via seed-time cross-chunk
        // look-down.
        assert!(
            hill_count > 50,
            "expected chz=1 hills via cross-chunk look-down — got {hill_count}"
        );
        // The proper-fix assertion: chz=0 distant mountain SHOULD be
        // visible. Currently fails — pins the limitation.
        assert!(
            mountain_count > 50,
            "expected chz=0 distant mountain visible — got {mountain_count} (S4B.6.l limitation)"
        );
    }

    /// S4B.6.h: mid-render chunk-Z handoff. Camera column has
    /// content in chz=0 (= a mountain at the camera's XY) so
    /// seed-time cross-chunk look-down does NOT fire — seed_chz=0.
    /// As rays DDA across the scene, they visit XY columns where
    /// chz=0 is all-air-bedrock. Mid-render handoff should swap
    /// state to chz=1's column at those XY positions and reveal
    /// hill content sitting under the camera's chz=0 layer.
    ///
    /// This is the "tall mountains breaching chunk-Z boundary"
    /// case the demo aims for.
    #[test]
    fn mid_render_handoff_reveals_chz1_hills_under_mountain_camera() {
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::at(DVec3::ZERO));
        let g = scene.grid_mut(id).unwrap();
        // chz=0: a small "mountain peak" at the camera's XY.
        // Mountain at world z=150..200 — solid block.
        g.set_rect(
            IVec3::new(60, 60, 150),
            IVec3::new(72, 72, 200),
            Some(0x80_88_44_22), // brown mountain
        );
        // chz=1: hills at world z=336..360 across the WHOLE chunk
        // (so DDA rays hit them when chz=0 is air).
        g.set_rect(
            IVec3::new(0, 0, 336),
            IVec3::new(128, 128, 360),
            Some(0x80_22_88_44), // green hills
        );
        // Carve a hole in chz=1's hill at the mountain's footprint
        // so the mountain doesn't appear to "float" on green.
        g.set_rect(IVec3::new(60, 60, 336), IVec3::new(72, 72, 360), None);
        assert!(g.chunk(IVec3::new(0, 0, 0)).is_some());
        assert!(g.chunk(IVec3::new(0, 0, 1)).is_some());

        let (_engine, fog, sky_color) = make_composed_pool(2 * CHUNK_SIZE_XY);
        let mut fb = vec![sky_color; pixel_count(XRES, YRES)];
        let mut zb = vec![f32::INFINITY; pixel_count(XRES, YRES)];
        // Camera at world (66, 66, 100) — directly above the
        // mountain peak (at z=150). Camera column has the
        // mountain in chz=0. Look straight down.
        let camera = Camera {
            pos: [66.0, 66.0, 100.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 1.0, 0.0],
            forward: [0.0, 0.0, 1.0],
        };
        let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
        let outcome = render_scene_composed(
            &mut fb,
            &mut zb,
            XRES as usize,
            XRES,
            YRES,
            fog,
            &mut scene,
            &camera,
            &settings,
            sky_color,
            None,
        );
        assert_eq!(outcome, RenderOutcome::Rendered { grids_drawn: 1 });
        let mountain_count = fb.iter().filter(|&&p| p == 0x80_88_44_22).count();
        let hill_count = fb.iter().filter(|&&p| p == 0x80_22_88_44).count();
        // Verify the hills render at approximately the correct
        // world-z by sampling the z-buffer at hill pixels. Camera
        // at z=100 looking straight down; hills at world z=336.
        // Expected depth = 236 for directly-below pixels. If
        // state.z1 stays stuck at the mountain peak's z=150 the
        // hills would render with depth ≈ 50 → orders of magnitude
        // off.
        let mut hill_depths: Vec<f32> = fb
            .iter()
            .zip(zb.iter())
            .filter_map(|(&p, &d)| if p == 0x80_22_88_44 { Some(d) } else { None })
            .collect();
        hill_depths.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_hill_depth = hill_depths[hill_depths.len() / 2];
        eprintln!(
            "mid-render handoff: mountain={mountain_count} hill={hill_count} median_hill_depth={median_hill_depth:.1}"
        );
        assert!(
            mountain_count > 50,
            "should see mountain peak via chz=0 — got {mountain_count} mountain pixels"
        );
        assert!(
            hill_count > 50,
            "should see chz=1 hills via mid-render handoff — got {hill_count} hill pixels"
        );
        assert!(
            (median_hill_depth - 236.0).abs() < 80.0,
            "hill median depth should be ≈236 (camera→z=336); got {median_hill_depth:.1} — state.z1 may be stale at the mountain peak's z"
        );
    }

    /// S4B.6.g: cross-chunk look-down under multi-mip. Same scene
    /// as `stacked_two_chunk_z_camera_in_chz0_sees_chz1_floor` but
    /// with `mip_levels=2, mip_scan_dist=16` so the rasterizer
    /// transitions to mip-1 well within the chz=1 terrain. Locks in
    /// the slab_z_at mip-N offset fix (= `chunk_world_z_base >>
    /// gmipcnt`). Pre-fix produced a green / brown "wall in a circle
    /// around the camera" because mip-1 rendered the floor at
    /// world-z ≈ 178 instead of 306.
    #[test]
    fn stacked_two_chunk_z_camera_in_chz0_sees_chz1_floor_multi_mip() {
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::at(DVec3::ZERO));
        let g = scene.grid_mut(id).unwrap();
        g.ensure_chunk(IVec3::new(0, 0, 0));
        g.set_rect(
            IVec3::new(60, 60, 306),
            IVec3::new(72, 72, 310),
            Some(0x80_77_aa_44),
        );
        assert!(g.chunk(IVec3::new(0, 0, 1)).is_some());

        let (_engine, fog, sky_color) = make_composed_pool(2 * CHUNK_SIZE_XY);
        let mut fb = vec![sky_color; pixel_count(XRES, YRES)];
        let mut zb = vec![f32::INFINITY; pixel_count(XRES, YRES)];
        let camera = Camera {
            pos: [66.0, 66.0, 100.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 1.0, 0.0],
            forward: [0.0, 0.0, 1.0],
        };
        let mut settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
        settings.mip_levels = 2;
        settings.mip_scan_dist = 16;
        let outcome = render_scene_composed(
            &mut fb,
            &mut zb,
            XRES as usize,
            XRES,
            YRES,
            fog,
            &mut scene,
            &camera,
            &settings,
            sky_color,
            None,
        );
        assert_eq!(outcome, RenderOutcome::Rendered { grids_drawn: 1 });
        let floor_count = fb.iter().filter(|&&p| p == 0x80_77_aa_44).count();
        assert!(
            floor_count > 50,
            "multi-mip cross-chunk look-down should still see chz=1 floor — got {floor_count} floor pixels"
        );
    }

    /// S4B.6.d: 3-chunk-tall stack stresses the widened gylookup
    /// (`(chunks_z * 512) >> mip + 4` per mip). Pre-S4B.6.d, gylookup
    /// was hardcoded at `(512 >> mip) + 4`, which would OOB or alias
    /// for any z > 511. This test renders a floor at world z=562
    /// (= chz=2, local z=50) with the camera at world z=540, looking
    /// straight down. Multi-mip is on so we exercise the mip slide
    /// path in `phase_remiporend` that scales `advance` by chunks_z.
    #[test]
    fn stacked_three_chunk_z_camera_in_chz2_sees_own_chunk_floor_multi_mip() {
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::at(DVec3::ZERO));
        let g = scene.grid_mut(id).unwrap();
        // Materialise chz=0 + chz=1 so chunk_xyz_backing enumerates
        // the full stack.
        g.ensure_chunk(IVec3::new(0, 0, 0));
        g.ensure_chunk(IVec3::new(0, 0, 1));
        // chz=2: floor at world z=562..566 (= local z=50..54).
        g.set_rect(
            IVec3::new(60, 60, 562),
            IVec3::new(72, 72, 566),
            Some(0x80_aa_55_22),
        );
        assert!(g.chunk(IVec3::new(0, 0, 2)).is_some());

        let (_engine, fog, sky_color) = make_composed_pool(2 * CHUNK_SIZE_XY);
        let mut fb = vec![sky_color; pixel_count(XRES, YRES)];
        let mut zb = vec![f32::INFINITY; pixel_count(XRES, YRES)];
        let camera = Camera {
            pos: [66.0, 66.0, 540.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 1.0, 0.0],
            forward: [0.0, 0.0, 1.0],
        };
        // Multi-mip on to exercise the gylookup-slide path.
        let mut settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
        settings.mip_levels = 2;
        settings.mip_scan_dist = 16;
        let outcome = render_scene_composed(
            &mut fb,
            &mut zb,
            XRES as usize,
            XRES,
            YRES,
            fog,
            &mut scene,
            &camera,
            &settings,
            sky_color,
            None,
        );
        assert_eq!(outcome, RenderOutcome::Rendered { grids_drawn: 1 });
        let floor_count = fb.iter().filter(|&&p| p == 0x80_aa_55_22).count();
        assert!(
            floor_count > 100,
            "camera at chz=2 with floor in same chunk should see it — got {floor_count} floor pixels"
        );
    }

    // ---- S7.4: render integration with streaming ----

    /// Floor-stamping generator for S7.4 render tests. Produces a
    /// 10-voxel-thick floor at the bottom of every chunk it
    /// generates (chunk-local `z = 230..239`, all xy). Visible as
    /// a green stripe along the bottom of the framebuffer when
    /// the camera looks +y across populated chunks.
    #[derive(Debug)]
    struct FloorGenerator;

    impl crate::ChunkGenerator for FloorGenerator {
        fn generate(&self, _chunk_idx: IVec3) -> roxlap_formats::vxl::Vxl {
            // Lean on `Grid::ensure_chunk` for the empty-chunk
            // builder, then carve a floor via `set_rect`. Detach
            // the chunk from the temporary grid and return it.
            let mut tmp = crate::Grid::new(GridTransform::identity());
            tmp.ensure_chunk(IVec3::ZERO);
            let mut vxl = tmp.chunks.remove(&IVec3::ZERO).unwrap();
            #[allow(clippy::cast_possible_wrap)]
            roxlap_formats::edit::set_rect(
                &mut vxl,
                glam::IVec3::new(0, 0, 230).into(),
                glam::IVec3::new((CHUNK_SIZE_XY - 1) as i32, (CHUNK_SIZE_XY - 1) as i32, 239)
                    .into(),
                Some(0x80_22_aa_22),
            );
            vxl
        }
    }

    #[test]
    fn render_scene_composed_unpumped_streaming_grid_renders_all_sky() {
        // S7.4(a): a grid with a generator + active stream radius
        // but no pump_streaming call has zero chunks. The render
        // walks the grid (chunk_xyz_backing returns None for an
        // empty chunk map → grid is skipped), framebuffer stays
        // sky.
        use std::sync::Arc;
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::at(DVec3::ZERO));
        let g = scene.grid_mut(id).unwrap();
        g.set_generator(Some(Arc::new(FloorGenerator)));
        g.stream_radius = crate::StreamRadius::new(300.0, 600.0);
        assert!(g.chunks.is_empty(), "no pump yet → no chunks");

        let (_engine, fog, sky_color) = make_composed_pool(CHUNK_SIZE_XY);
        let mut fb = vec![sky_color; pixel_count(XRES, YRES)];
        let mut zb = vec![f32::INFINITY; pixel_count(XRES, YRES)];
        // Camera at (64, -100, 200) looking +y so it would see
        // chunks ahead once they exist.
        let camera = camera_at([64.0, -100.0, 200.0]);
        let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
        let _ = render_scene_composed(
            &mut fb,
            &mut zb,
            XRES as usize,
            XRES,
            YRES,
            fog,
            &mut scene,
            &camera,
            &settings,
            sky_color,
            None,
        );
        // Empty grid path skips opticast → framebuffer untouched.
        assert!(
            fb.iter().all(|&p| p == sky_color),
            "unpumped streaming grid must render as all sky"
        );
    }

    #[test]
    fn render_scene_composed_picks_up_streamed_chunks_after_sync_pump() {
        // S7.4(a): once the streaming pump installs chunks, the
        // next render shows them. Using pump_streaming_sync for
        // deterministic timing — pump_streaming (async) lands
        // the same way modulo a frame of latency.
        use std::sync::Arc;
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::at(DVec3::ZERO));
        let g = scene.grid_mut(id).unwrap();
        g.set_generator(Some(Arc::new(FloorGenerator)));
        // Cover chunks ahead of the camera (y=0, y=128, y=256).
        g.stream_radius = crate::StreamRadius::new(300.0, 600.0);

        // Render BEFORE pump: zero floor pixels.
        let (_engine, fog, sky_color) = make_composed_pool(CHUNK_SIZE_XY);
        let mut fb = vec![sky_color; pixel_count(XRES, YRES)];
        let mut zb = vec![f32::INFINITY; pixel_count(XRES, YRES)];
        let camera = camera_at([64.0, -100.0, 200.0]);
        let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
        let _ = render_scene_composed(
            &mut fb,
            &mut zb,
            XRES as usize,
            XRES,
            YRES,
            fog,
            &mut scene,
            &camera,
            &settings,
            sky_color,
            None,
        );
        let pre_floor = fb.iter().filter(|&&p| p == 0x80_22_aa_22).count();
        assert_eq!(pre_floor, 0, "pre-pump frame has no streamed chunks");

        // Pump synchronously — `world_pos` matches the camera so
        // chunks ahead of it (within r_active = 300) stream in.
        scene.pump_streaming_sync(DVec3::new(64.0, -100.0, 200.0));
        let g = scene.grid(id).unwrap();
        assert!(
            !g.chunks.is_empty(),
            "pump should have streamed at least one chunk"
        );

        // Render AFTER pump: the floor should now be visible. Reset
        // the framebuffer to sky first.
        fb.iter_mut().for_each(|p| *p = sky_color);
        zb.iter_mut().for_each(|z| *z = f32::INFINITY);
        let outcome = render_scene_composed(
            &mut fb,
            &mut zb,
            XRES as usize,
            XRES,
            YRES,
            fog,
            &mut scene,
            &camera,
            &settings,
            sky_color,
            None,
        );
        assert_eq!(outcome, RenderOutcome::Rendered { grids_drawn: 1 });
        let post_floor = fb.iter().filter(|&&p| p == 0x80_22_aa_22).count();
        assert!(
            post_floor > 100,
            "post-pump frame should show the streamed floor — got {post_floor} green pixels"
        );
    }

    #[test]
    fn render_scene_composed_partial_streaming_renders_pending_chunks_as_air() {
        // S7.4(a): mixed state — some r_active chunks are
        // materialised, others are still pending (not in
        // `chunks`). The render must treat pending chunks as
        // implicit-air. Verified by stamping one chunk via the
        // generator + skipping the others, then confirming the
        // framebuffer has fewer floor pixels than the
        // fully-pumped baseline.
        use std::sync::Arc;
        let mut scene = Scene::new();
        let id = scene.add_grid(GridTransform::at(DVec3::ZERO));
        let g = scene.grid_mut(id).unwrap();
        g.set_generator(Some(Arc::new(FloorGenerator)));
        // r_active must be set so the later pump_streaming_sync
        // sanity-check actually streams more chunks in.
        g.stream_radius = crate::StreamRadius::new(400.0, 800.0);

        // Materialise ONLY chunk (0, 0, 0) manually via the
        // sync helper — leave (0, 1, 0), (0, 2, 0) absent.
        let installed = g.ensure_chunk_generated(IVec3::ZERO);
        assert!(installed, "manual install of one chunk");
        assert_eq!(g.chunks.len(), 1);
        // Make sure (0, 1, 0), (0, 2, 0) are NOT present.
        assert!(g.chunk(IVec3::new(0, 1, 0)).is_none());
        assert!(g.chunk(IVec3::new(0, 2, 0)).is_none());

        let (_engine, fog, sky_color) = make_composed_pool(CHUNK_SIZE_XY);
        let mut fb = vec![sky_color; pixel_count(XRES, YRES)];
        let mut zb = vec![f32::INFINITY; pixel_count(XRES, YRES)];
        // Camera inside chunk (0, 0, 0); looking +y means the
        // floor of (0, 0, 0) gets rendered until the ray walks
        // off the chunk into implicit-air space at y=128. No
        // floor pixels past that distance.
        let camera = camera_at([64.0, 32.0, 200.0]);
        let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
        let _ = render_scene_composed(
            &mut fb,
            &mut zb,
            XRES as usize,
            XRES,
            YRES,
            fog,
            &mut scene,
            &camera,
            &settings,
            sky_color,
            None,
        );
        let floor_pixels = fb.iter().filter(|&&p| p == 0x80_22_aa_22).count();
        // Visible floor inside chunk (0,0,0); pending neighbours
        // contribute nothing. The number isn't pinned exactly —
        // it just needs to be non-zero (we have content) and
        // less than what a fully-streamed scene would produce.
        assert!(
            floor_pixels > 0,
            "should see at least some floor from the loaded chunk"
        );
        // Sanity: stream the missing chunks; verify the floor
        // pixel count goes up.
        scene.pump_streaming_sync(DVec3::new(64.0, 32.0, 200.0));
        assert!(scene.grid(id).unwrap().chunk_count() >= 2);
        fb.iter_mut().for_each(|p| *p = sky_color);
        zb.iter_mut().for_each(|z| *z = f32::INFINITY);
        let _ = render_scene_composed(
            &mut fb,
            &mut zb,
            XRES as usize,
            XRES,
            YRES,
            fog,
            &mut scene,
            &camera,
            &settings,
            sky_color,
            None,
        );
        let floor_pixels_full = fb.iter().filter(|&&p| p == 0x80_22_aa_22).count();
        assert!(
            floor_pixels_full > floor_pixels,
            "fully-streamed scene should show more floor than partial: \
             partial={floor_pixels} full={floor_pixels_full}"
        );
    }
}
