// GPU.5 — multi-grid scene marcher.
//
// For each grid in 0..grid_count:
//   build the grid-local ray from per_grid_camera[i]
//   outer DDA over chunks (skip via chunk_occupancy)
//   inner DDA over voxels (bounded to current chunk)
//   on hit at world-t < best_t: update best_color + best_t
// emit best_color (or sky if no hit).
//
// All grids' chunks share one set of storage buffers; per-grid
// offsets live in `grid_static_meta`. Per-grid camera state lives
// in the `per_grid_camera` uniform array (computed CPU-side each
// frame via inverse `GridTransform`).
//
// `t` is in WORLD units. Comparing best_t across grids works
// because each grid's per-grid camera is the WORLD camera
// transformed into grid-local — the `t` along the local ray equals
// the world-space `t` (rigid transforms preserve distance).

const OCC_WORDS_PER_COLUMN: u32 = 8u; // CHUNK_Z (256) / 32
const CHUNK_Z: u32 = 256u;
const MAX_INNER_STEPS: u32 = 768u;
const MAX_GPU_MIPS: u32 = 6u; // GPU.11 — must match scene::MAX_GPU_MIPS
const T_INF: f32 = 1.0e30;

struct PerGridCamera {
    pos: vec3<f32>,
    _pad0: f32,
    right: vec3<f32>,
    _pad1: f32,
    down: vec3<f32>,
    _pad2: f32,
    forward: vec3<f32>,
    _pad3: f32,
    // DL — unit direction TO the sun in this grid's local frame (xyz; w
    // unused). Packed here instead of a separate per-grid storage buffer
    // (the 16 storage-buffer limit is already saturated). Zero ⇒ no sun
    // (the uniform's `sun_flags` gates whether it's used).
    sun_dir: vec4<f32>,
};

struct GridStaticMeta {
    occupancy_offset: u32,
    color_offsets_offset: u32,
    colors_offset: u32,
    chunk_colors_base_offset: u32,
    chunk_occupancy_offset: u32,
    slot_chunk_idx_offset: u32,
    vsid: u32,
    total_slots: u32,
    pool_dims: vec3<u32>,
    _pad0: u32,
    // GPU.11 — per-slot strides spanning the whole mip ladder, plus
    // per-mip within-slot relative offsets. mip_*_rel[0] == 0 so
    // mip-0 reads index exactly as the pre-mip layout did.
    occ_words_per_slot: u32,
    offsets_words_per_slot: u32,
    mip_count: u32,
    _pad1: u32,
    mip_occ_rel: array<u32, MAX_GPU_MIPS>,
    mip_coff_rel: array<u32, MAX_GPU_MIPS>,
    // GPU.13.0 — occupied chunk-AABB (inclusive) in chunk-index space.
    // `vec3<i32>` aligns to 16 here (mip_coff_rel ends 16-aligned), so
    // these mirror the host's `[i32;3] + pad` pair exactly (112→144).
    aabb_min: vec3<i32>,
    aabb_max: vec3<i32>,
};

struct Uniforms {
    fov_y_rad: f32,
    grid_count: u32,
    max_outer_steps: u32,
    _pad0: u32,
    screen_size: vec2<u32>,
    _pad1: vec2<u32>,
    // GPU.8 fog. `fog_color.rgb` is the colour we blend toward at
    // far distances. `fog_color.w` is `fog_near`, packed with the
    // colour to keep std140 alignment simple.
    fog_color: vec4<f32>,
    fog_far: f32,
    // GPU.9: gate the depth-buffer write. When the sprite pass is
    // active this is 1 and `render_scene` records `best_t` per
    // pixel; otherwise 0 and the no-sprite path stays unchanged.
    write_depth: u32,
    // Occupancy paging: words per storage page, and the number of
    // real pages. `occ_num_pages == 1` (multi-GiB GPUs) takes a
    // branch-free single-page read.
    occ_page_words: u32,
    occ_num_pages: u32,
    // GPU.11.1 — scene-grid LOD. A chunk entered at world-t `t` is
    // marched at mip level `floor(log2(max(t, msd) / msd))`, clamped
    // to the grid's `mip_count`. `0` disables LOD (always mip-0).
    // Tunable for the axis-aligned-mip-beams mitigation (11.2).
    mip_scan_dist: f32,
    // TV.6 — 1 if any terrain material is translucent (gates the
    // accumulate path; 0 ⇒ unchanged first-hit opaque march).
    terrain_has_translucent: u32,
    // TV.6 — number of (rgb, material_id) entries in `terrain_map`.
    terrain_map_count: u32,
    _pad4: u32,
    // World camera used purely to derive the per-pixel sky direction.
    // Always valid (even with grid_count == 0, where no grid ray
    // exists), so a grid-less scene still paints a proper sky instead
    // of a degenerate (0,0,1) → atan2(0,0) → black sample.
    sky_cam: PerGridCamera,
    // Per-face directional shading (voxlap setsideshades), as the
    // alpha-brightness reduction applied at a voxel hit. Each value is
    // the u8 shade intensity (0..255) subtracted from the voxel's
    // brightness byte before the /128 divide — matching the CPU
    // `grouscan_shade`. side_shades0 = (top, bot, left, right),
    // side_shades1 = (up, down, _, _). All-zero = no shading.
    side_shades0: vec4<i32>,
    side_shades1: vec4<i32>,
    // ── DL — dynamic lighting (appended; all-zero ⇒ pre-DL render) ──
    // rgb = sun colour, w = sun intensity.
    sun_color: vec4<f32>,
    // rgb = ambient multiplier on the baked byte, w = shadow strength.
    ambient_color: vec4<f32>,
    // bit0 = sun enabled, bit1 = sun casts shadow.
    sun_flags: u32,
    point_light_count: u32,
    shadow_max_steps: u32,
    _pad5: u32,
    shadow_bias: f32,
    shadow_max_dist: f32,
    _pad6: vec2<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
// Occupancy is split across up to MAX_OCC_PAGES (=4) storage
// bindings so no single binding exceeds the device limit. Page 0 is
// binding 1; pages 1..3 are bindings 12..14. `occ_word()` maps a
// global word index to its page. See scene::split_occupancy_pages.
@group(0) @binding(1) var<storage, read> occ_page0: array<u32>;
@group(0) @binding(2) var<storage, read> all_color_offsets: array<u32>;
@group(0) @binding(3) var<storage, read> all_colors: array<u32>;
@group(0) @binding(4) var<storage, read> all_chunk_colors_base: array<u32>;
@group(0) @binding(5) var<storage, read> all_chunk_occupancy: array<u32>;
@group(0) @binding(6) var<storage, read> grid_static_meta: array<GridStaticMeta>;
// GPU.7: per-slot chunk_idx, vec3<i32> with std430 16-byte stride.
@group(0) @binding(7) var<storage, read> all_slot_chunk_idx: array<vec3<i32>>;
// Framebuffer as a storage BUFFER (packed `rgba8unorm` per pixel),
// not a storage texture: Chrome's Dawn lays out write storage
// textures with GPU-optimal tiling that the sampled read-back
// disagrees with, producing a 128×256-tiled image. A linear buffer
// + an explicit `screen_size.x` stride is layout-unambiguous on every
// backend (the depth buffer already uses this).
@group(0) @binding(8) var<storage, read_write> output: array<u32>;
// GPU.8: panoramic sky.
@group(0) @binding(9) var sky_texture: texture_2d<f32>;
@group(0) @binding(10) var sky_sampler: sampler;
// GPU.9: per-pixel world-t depth (f32 bits as u32). Written here
// when `u.write_depth != 0`, read+tested by the sprite splatter.
@group(0) @binding(11) var<storage, read_write> depth_buffer: array<u32>;
// Occupancy pages 1..3 (page 0 is binding 1). Unused pages bind a
// 1-word dummy and are never indexed.
@group(0) @binding(12) var<storage, read> occ_page1: array<u32>;
@group(0) @binding(13) var<storage, read> occ_page2: array<u32>;
@group(0) @binding(14) var<storage, read> occ_page3: array<u32>;
// Per-grid world->grid cameras, one per grid (`grid_count` of them).
// Moved out of the uniform (was a fixed `array<…, 16>`) into a runtime-
// sized storage array so a scene can hold any number of grids — the cap
// is now the device's storage limit, not a baked-in 16. The shader only
// indexes `0..grid_count`, so a grid-less scene binds a 1-element dummy.
@group(0) @binding(15) var<storage, read> grid_cameras: array<PerGridCamera>;
// TV.6 — global voxel-material palette (256), `mode`: 0=Opaque,
// 1=AlphaBlend, 2=Additive; alpha normalised 0..1.
struct Mat { alpha: f32, mode: u32 };
@group(0) @binding(16) var<storage, read> materials_pal: array<Mat>;
// TV.6 — terrain colour→material map: `.x` = rgb (0xRRGGBB), `.y` =
// material id. A hit voxel's colour is matched here to find its material.
@group(0) @binding(17) var<storage, read> terrain_map: array<vec2<u32>>;
// DL — dynamic lighting. One point light in a grid's local frame (std430,
// 48 bytes). Mirrors `GpuPointLight` in lib.rs.
struct PointLight {
    pos: vec3<f32>,
    radius: f32,
    color: vec3<f32>,
    intensity: f32,
    casts_shadow: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
};
// DL binding 18 — per-grid point lights, grid-major: grid g's lights at
// [g*point_light_count .. (g+1)*point_light_count]. DL.0 binds the buffer
// but the hit-site shading that reads it lands in DL.1+. (The per-grid sun
// direction rides in `grid_cameras[g].sun_dir`, binding 15.)
@group(0) @binding(18) var<storage, read> grid_point_lights: array<PointLight>;

// TV.6 — material id for a terrain voxel colour (linear scan of the small
// map); 0 (opaque) when unmapped or the map is empty.
fn terrain_material_id(packed: u32) -> u32 {
    let rgb = packed & 0x00ffffffu;
    for (var i: u32 = 0u; i < u.terrain_map_count; i = i + 1u) {
        if ((terrain_map[i].x & 0x00ffffffu) == rgb) {
            return terrain_map[i].y;
        }
    }
    return 0u;
}

// Read one occupancy word by global index, selecting its page.
// Single-page scenes (multi-GiB GPUs) skip the division — the
// branch is uniform across the workgroup, so it's effectively free.
fn occ_word(i: u32) -> u32 {
    if (u.occ_num_pages <= 1u) {
        return occ_page0[i];
    }
    let page = i / u.occ_page_words;
    let local = i % u.occ_page_words;
    if (page == 0u) { return occ_page0[local]; }
    if (page == 1u) { return occ_page1[local]; }
    if (page == 2u) { return occ_page2[local]; }
    return occ_page3[local];
}

// GPU.11.1 — occupancy words per column at `mip`
// (`(CHUNK_Z >> mip) / 32`, min 1). Mirrors
// `decompress::occ_words_per_column_for_mip`.
fn occ_words_per_col_for_mip(mip: u32) -> u32 {
    return max(1u, (CHUNK_Z >> mip) / 32u);
}

// GPU.11.1 — word base of column `(p_voxel.x, p_voxel.y)`'s occupancy
// at `mip` within slot `meta_id`. Indexes `grid_static_meta`
// **directly** (storage address space): WGSL forbids dynamic
// indexing of an array member once the struct is copied into a value
// `let`. `mip_occ_rel[mip]` is the within-slot start of that mip's
// sub-block (0 for mip-0).
fn col_word_base_mip(g: u32, meta_id: u32, mip: u32, p_voxel: vec3<i32>) -> u32 {
    let vsid_mip = grid_static_meta[g].vsid >> mip;
    let col_idx = u32(p_voxel.x) + u32(p_voxel.y) * vsid_mip;
    let occ_base = grid_static_meta[g].occupancy_offset
        + meta_id * grid_static_meta[g].occ_words_per_slot
        + grid_static_meta[g].mip_occ_rel[mip];
    return occ_base + col_idx * occ_words_per_col_for_mip(mip);
}

// Within-slot word stride of one mip's textured occupancy block; the
// SOLID occupancy block sits immediately after it (cliff-face fix). So
// the solid word base for a column == its textured base + this.
fn mip_occ_block_words(g: u32, mip: u32) -> u32 {
    let vsid_mip = grid_static_meta[g].vsid >> mip;
    return vsid_mip * vsid_mip * occ_words_per_col_for_mip(mip);
}

// GPU — hit-test against the SOLID bitmap (textured surfaces + bedrock
// interior) so vertical wall/cliff faces are opaque. The textured
// bitmap (used for colour rank) is the first block; solid is the
// second.
fn voxel_solid_in(g: u32, meta_id: u32, mip: u32, p_voxel: vec3<i32>) -> bool {
    let solid_base = col_word_base_mip(g, meta_id, mip, p_voxel) + mip_occ_block_words(g, mip);
    let z_word = u32(p_voxel.z) >> 5u;
    let z_bit = u32(p_voxel.z) & 31u;
    return (occ_word(solid_base + z_word) & (1u << z_bit)) != 0u;
}

// Per-face side-shade intensity for a voxel hit, mirroring the CPU's
// gcsub-lane selection: z-faces → top/bot (ceiling/floor), x-faces →
// left/right, y-faces → up/down, with the pair chosen by the ray's
// direction sign along that axis (= voxlap's gixy-sign select).
// `axis`: 0=x, 1=y, 2=z.
fn side_shade_for(axis: i32, ray_dir: vec3<f32>) -> f32 {
    if (axis == 2) {
        // ray going +z (down, voxlap z-down) hits a floor → bot, else ceiling → top
        return f32(select(u.side_shades0.x, u.side_shades0.y, ray_dir.z >= 0.0));
    } else if (axis == 0) {
        return f32(select(u.side_shades0.z, u.side_shades0.w, ray_dir.x >= 0.0));
    }
    return f32(select(u.side_shades1.x, u.side_shades1.y, ray_dir.y >= 0.0));
}

// The raw packed `0x__RRGGBB` colour of a voxel (the value `voxel_color_in`
// shades). TV.6 uses it for the terrain colour→material lookup.
fn voxel_packed_in(g: u32, meta_id: u32, mip: u32, p_voxel: vec3<i32>) -> u32 {
    let vsid_mip = grid_static_meta[g].vsid >> mip;
    let col_idx = u32(p_voxel.x) + u32(p_voxel.y) * vsid_mip;
    let col_word_base = col_word_base_mip(g, meta_id, mip, p_voxel);
    let z_word = u32(p_voxel.z) >> 5u;
    let z_bit = u32(p_voxel.z) & 31u;
    var rank: u32 = 0u;
    for (var w: u32 = 0u; w < z_word; w = w + 1u) {
        rank = rank + countOneBits(occ_word(col_word_base + w));
    }
    var mask: u32 = 0u;
    if (z_bit > 0u) {
        mask = (1u << z_bit) - 1u;
    }
    let z_word_bits = occ_word(col_word_base + z_word);
    rank = rank + countOneBits(z_word_bits & mask);
    let is_textured = (z_word_bits & (1u << z_bit)) != 0u;
    var color_index = rank;
    if (!is_textured && rank > 0u) {
        color_index = rank - 1u;
    }
    let offsets_base = grid_static_meta[g].color_offsets_offset
        + meta_id * grid_static_meta[g].offsets_words_per_slot
        + grid_static_meta[g].mip_coff_rel[mip];
    let chunk_local_offset = all_color_offsets[offsets_base + col_idx];
    let chunk_colors_base =
        all_chunk_colors_base[grid_static_meta[g].chunk_colors_base_offset + meta_id];
    return all_colors[grid_static_meta[g].colors_offset + chunk_colors_base
        + chunk_local_offset + color_index];
}

fn voxel_color_in(g: u32, meta_id: u32, mip: u32, p_voxel: vec3<i32>, face_shade: f32) -> vec3<f32> {
    let packed = voxel_packed_in(g, meta_id, mip, p_voxel);
    let a = f32((packed >> 24u) & 0xffu);
    let r = f32((packed >> 16u) & 0xffu);
    let g_chan = f32((packed >> 8u) & 0xffu);
    let b = f32(packed & 0xffu);
    let brightness = max(0.0, a - face_shade) * (1.0 / 128.0);
    return vec3<f32>(r, g_chan, b) * (brightness / 255.0);
}

// DL — face normal in grid-local space: it points back along the ray on
// the crossed axis (toward the incoming ray). `axis` ∈ {0,1,2} is the
// DDA's last-stepped axis (the same value side-shading uses).
fn face_normal(axis: i32, ray_dir: vec3<f32>) -> vec3<f32> {
    if (axis == 0) { return vec3<f32>(-sign(ray_dir.x), 0.0, 0.0); }
    if (axis == 1) { return vec3<f32>(0.0, -sign(ray_dir.y), 0.0); }
    return vec3<f32>(0.0, 0.0, -sign(ray_dir.z));
}

// DL — dynamic-lighting surface shade (pre-fog), same 0..~2 scale as
// `voxel_color_in`. The baked brightness byte is reinterpreted as the
// ambient/AO term (× `ambient_color`); the sun adds an N·L diffuse term on
// top of the raw albedo. Point lights + shadows land in DL.2 / DL.3. Only
// taken when dynamic lighting is active (`sun_flags` bit 2) — otherwise the
// hit site uses `voxel_color_in` verbatim (the byte-identical pre-DL path).
fn shade_lit(
    g: u32,
    meta_id: u32,
    mip: u32,
    p_voxel: vec3<i32>,
    face_shade: f32,
    hit_axis: i32,
    ray_dir: vec3<f32>,
) -> vec3<f32> {
    let packed = voxel_packed_in(g, meta_id, mip, p_voxel);
    let a = f32((packed >> 24u) & 0xffu);
    let albedo = vec3<f32>(
        f32((packed >> 16u) & 0xffu),
        f32((packed >> 8u) & 0xffu),
        f32(packed & 0xffu),
    ) * (1.0 / 255.0);
    // Ambient term = baked brightness byte (with the per-face side-shade),
    // matching `voxel_color_in`'s brightness, scaled by the ambient mult.
    let ambient = max(0.0, a - face_shade) * (1.0 / 128.0);
    var lit = albedo * u.ambient_color.rgb * ambient;
    // Directional sun: N·L diffuse on the raw albedo. No shadow yet (DL.3).
    if ((u.sun_flags & 1u) != 0u) {
        let n = face_normal(hit_axis, ray_dir);
        let l = grid_cameras[g].sun_dir.xyz; // unit, TO the sun, grid-local
        let ndl = max(0.0, dot(n, l));
        lit = lit + albedo * u.sun_color.rgb * u.sun_color.w * ndl;
    }
    return lit;
}

// GPU.7 modular slot lookup. `pool_dims` are powers of 2 (asserted
// on the host), so `chunk_idx & (pool_dims - 1)` is the slot index
// per axis. Slot identity must be verified against
// `all_slot_chunk_idx` — multiple chunk_idx values can map to the
// same slot under the pool's collision invariant.
fn slot_idx_of(g: u32, chunk_idx: vec3<i32>) -> u32 {
    let m = grid_static_meta[g];
    let mask = vec3<i32>(m.pool_dims) - vec3<i32>(1, 1, 1);
    let s = chunk_idx & mask;
    return u32(s.x)
        + u32(s.y) * m.pool_dims.x
        + u32(s.z) * m.pool_dims.x * m.pool_dims.y;
}

// GPU.13.0 — has the outer DDA left the grid's occupied chunk-AABB
// for good? A 3D-DDA ray is inside the box only while all three axes
// are within `[aabb_min, aabb_max]`; once it crosses the far slab on
// any axis (in its travel direction) it can never re-enter, so no
// resident chunk lies ahead. An axis the ray is parallel to (`step ==
// 0`) and already outside the box means the ray misses the grid
// entirely. Either way the caller returns `out` (sky / no closer hit).
// The empty-grid sentinel (min = i32::MAX, max = i32::MIN) makes every
// branch fire immediately, so an empty grid contributes nothing.
fn aabb_passed(g: u32, p: vec3<i32>, step: vec3<i32>) -> bool {
    let mn = grid_static_meta[g].aabb_min;
    let mx = grid_static_meta[g].aabb_max;
    if (step.x > 0 && p.x > mx.x) { return true; }
    if (step.x < 0 && p.x < mn.x) { return true; }
    if (step.x == 0 && (p.x < mn.x || p.x > mx.x)) { return true; }
    if (step.y > 0 && p.y > mx.y) { return true; }
    if (step.y < 0 && p.y < mn.y) { return true; }
    if (step.y == 0 && (p.y < mn.y || p.y > mx.y)) { return true; }
    if (step.z > 0 && p.z > mx.z) { return true; }
    if (step.z < 0 && p.z < mn.z) { return true; }
    if (step.z == 0 && (p.z < mn.z || p.z > mx.z)) { return true; }
    return false;
}

fn chunk_has_content(g: u32, slot_idx: u32, chunk_idx: vec3<i32>) -> bool {
    let m = grid_static_meta[g];
    // Identity check: does this slot actually hold the chunk the
    // outer DDA is visiting? An empty slot's sentinel
    // (i32::MIN, i32::MIN, i32::MIN) fails this check.
    // vec3<i32> entries are at `slot_chunk_idx_offset/4 + slot_idx`
    // since WGSL `array<vec3<i32>>` uses 16-byte stride.
    let stored = all_slot_chunk_idx[m.slot_chunk_idx_offset / 4u + slot_idx];
    if (stored.x != chunk_idx.x || stored.y != chunk_idx.y || stored.z != chunk_idx.z) {
        return false;
    }
    return (all_chunk_occupancy[m.chunk_occupancy_offset + (slot_idx >> 5u)]
        & (1u << (slot_idx & 31u))) != 0u;
}

// Voxlap-convention sky sample. The bundled `assets/sky.png` is
// `width = elevation` (horizon → zenith), `height = azimuth`
// (wraps 360°) — the OPPOSITE axes of a standard equirectangular
// panorama. We sample `(elevation, azimuth)` in `(u, v)` to match
// the CPU rasterizer's orientation, and rely on the sampler's
// `Repeat` mode on both axes (elevation values stay in [0, 1] so
// Repeat is a no-op there; azimuth needs the wrap).
fn sky_color(dir: vec3<f32>) -> vec3<f32> {
    let pi = 3.1415926535897932;
    let azimuth = atan2(dir.x, dir.y) * (0.5 / pi) + 0.5;
    let elevation = clamp(acos(-dir.z) * (1.0 / pi), 0.0, 1.0);
    return textureSampleLevel(
        sky_texture,
        sky_sampler,
        vec2<f32>(elevation, azimuth),
        0.0,
    ).rgb;
}

// GPU.8 fog blend. `t` is the world-space hit distance; below
// `fog_near` the hit shows through fully; above `fog_far` only the
// fog colour shows. Linear ramp (not smoothstep) to match the CPU /
// DDA fog `clamp((t - near) / (far - near))` — smoothstep's S-curve
// gave visibly weaker mid-distance fog than the DDA renderer.
fn apply_fog(hit_color: vec3<f32>, t: f32) -> vec3<f32> {
    let fog_near = u.fog_color.w;
    let factor = clamp((t - fog_near) / max(u.fog_far - fog_near, 1e-6), 0.0, 1.0);
    return mix(hit_color, u.fog_color.rgb, factor);
}

fn shield_parallel(t_max: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    var t = t_max;
    if (dir.x == 0.0) { t.x = T_INF; }
    if (dir.y == 0.0) { t.y = T_INF; }
    if (dir.z == 0.0) { t.z = T_INF; }
    return t;
}

// GPU.11.1 — choose the mip a chunk is marched at, from the world-t
// at which the ray enters it. mip-0 inside `mip_scan_dist`, then one
// coarser level per distance-octave, clamped to the grid's ladder.
fn pick_mip(t: f32, mip_count: u32) -> u32 {
    if (u.mip_scan_dist <= 0.0 || mip_count <= 1u) {
        return 0u;
    }
    let ratio = max(t, u.mip_scan_dist) / u.mip_scan_dist;
    let lvl = u32(floor(log2(ratio)));
    return min(lvl, mip_count - 1u);
}

// March one grid; return (hit, t, color). `best_t` is the world-t
// threshold the caller already found in earlier grids; we early-out
// once our outer t passes it.
struct GridHit {
    hit: bool,
    t: f32,
    color: vec3<f32>,
};

// TV.6 — finalize a translucent terrain ray that exited the grid: composite
// the accumulated layers over the sky. `touched == false` ⇒ no contribution,
// returns a non-hit (identical to the opaque path's plain exit). Depth is a
// large finite value (far) so it loses to nearer opaque grids but still wins
// over the T_INF sky seed in `render_scene`.
fn finalize_sky_grid(touched: bool, accum: vec3<f32>, trans: f32, ray_dir: vec3<f32>) -> GridHit {
    var o: GridHit;
    o.hit = touched;
    o.t = 1.0e29;
    o.color = vec3<f32>(0.0);
    if (touched) {
        o.color = accum + trans * sky_color(ray_dir);
    }
    return o;
}

fn march_grid(
    g: u32,
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    best_t: f32,
) -> GridHit {
    let m = grid_static_meta[g];
    let chunk_dim = vec3<f32>(f32(m.vsid), f32(m.vsid), f32(CHUNK_Z));
    // World ray length per `t` unit; divided by a voxel's size it turns a
    // cell's `t` span into its path length in voxel units (Volumetric weight).
    let ray_dir_len = length(ray_dir);

    var p_chunk = vec3<i32>(floor(ray_origin / chunk_dim));
    let step_chunk = vec3<i32>(sign(ray_dir));
    let t_delta_chunk = abs(chunk_dim / ray_dir);
    let next_boundary_chunk = vec3<f32>(
        select(f32(p_chunk.x), f32(p_chunk.x + 1), step_chunk.x > 0) * chunk_dim.x,
        select(f32(p_chunk.y), f32(p_chunk.y + 1), step_chunk.y > 0) * chunk_dim.y,
        select(f32(p_chunk.z), f32(p_chunk.z + 1), step_chunk.z > 0) * chunk_dim.z,
    );
    var t_max_chunk = shield_parallel(
        (next_boundary_chunk - ray_origin) / ray_dir,
        ray_dir,
    );

    var t_enter: f32 = 0.0;
    // Axis crossed to enter the current chunk (= the face normal of a
    // voxel that is already solid at the chunk-entry point). Seeds
    // `hit_axis` for the `iv==0` case so a surface flush with the chunk
    // boundary gets its real face axis, not a hardcoded z. Defaults to z
    // only for the first chunk (t_enter==0, ray starts inside it).
    var entry_axis: i32 = 2;
    var out: GridHit;
    out.hit = false;
    out.t = T_INF;
    out.color = vec3<f32>(0.0);

    // TV.6 — front-to-back translucent accumulation (gated on
    // `terrain_has_translucent`; while off, `touched` stays false and every
    // return is the unchanged opaque result). `prev_*` drive per-span
    // compositing; reset per chunk (a seam at chunk borders is acceptable).
    var accum = vec3<f32>(0.0);
    var trans = 1.0;
    var touched = false;
    var prev_solid = false;
    var prev_mat = 0u;

    for (var step: u32 = 0u; step < u.max_outer_steps; step = step + 1u) {
        if (t_enter > best_t) {
            return finalize_sky_grid(touched, accum, trans, ray_dir);
        }
        // GPU.13.0 — once the ray has left the occupied chunk-AABB
        // along its travel direction, no resident chunk lies ahead:
        // stop instead of stepping empty space to max_outer_steps.
        if (aabb_passed(g, p_chunk, step_chunk)) {
            return finalize_sky_grid(touched, accum, trans, ray_dir);
        }
        let slot_id = slot_idx_of(g, p_chunk);
        prev_solid = false; // fresh chunk: start a new solid run
        if (chunk_has_content(g, slot_id, p_chunk)) {
            // GPU.11.1 — pick the mip for this chunk by entry distance.
            // Voxels are `vsize` world units; the chunk holds
            // `vsid>>mip` × `vsid>>mip` × `CHUNK_Z>>mip` of them.
            let mip = pick_mip(t_enter, m.mip_count);
            let vsize = f32(1u << mip);
            let vsid_mip = i32(m.vsid >> mip);
            let cz_mip = i32(CHUNK_Z >> mip);

            let entry_world = ray_origin + t_enter * ray_dir;
            let chunk_origin_world = vec3<f32>(p_chunk) * chunk_dim;
            let entry_in_chunk = entry_world - chunk_origin_world;
            var p_voxel = vec3<i32>(floor(entry_in_chunk / vsize));
            p_voxel = clamp(
                p_voxel,
                vec3<i32>(0),
                vec3<i32>(vsid_mip - 1, vsid_mip - 1, cz_mip - 1),
            );

            // Voxel boundaries are at integer-mip-coord * vsize.
            let next_voxel_world = vec3<f32>(
                select(f32(p_voxel.x), f32(p_voxel.x + 1), step_chunk.x > 0) * vsize
                    + chunk_origin_world.x,
                select(f32(p_voxel.y), f32(p_voxel.y + 1), step_chunk.y > 0) * vsize
                    + chunk_origin_world.y,
                select(f32(p_voxel.z), f32(p_voxel.z + 1), step_chunk.z > 0) * vsize
                    + chunk_origin_world.z,
            );
            var t_max_voxel = shield_parallel(
                (next_voxel_world - ray_origin) / ray_dir,
                ray_dir,
            );
            let t_delta_voxel = abs(vsize / ray_dir);
            var t_hit: f32 = t_enter;
            // Axis of the last voxel step = the hit face normal (for
            // side-shading). An iv==0 hit (solid at the chunk-entry point)
            // takes no inner step, so seed with the chunk-entry axis — the
            // face the ray crossed to enter this chunk. Surfaces hit after
            // any inner travel overwrite this with the real stepped axis.
            var hit_axis: i32 = entry_axis;

            for (var iv: u32 = 0u; iv < MAX_INNER_STEPS; iv = iv + 1u) {
                if (voxel_solid_in(g, slot_id, mip, p_voxel)) {
                    if (t_hit >= best_t) {
                        return finalize_sky_grid(touched, accum, trans, ray_dir);
                    }
                    let shade = side_shade_for(hit_axis, ray_dir);
                    // DL — lit path (ambient + sun) when dynamic lighting is
                    // active (sun_flags bit 2); else the baked-only path,
                    // byte-identical to pre-DL.
                    var base_color: vec3<f32>;
                    if ((u.sun_flags & 4u) != 0u) {
                        base_color = shade_lit(g, slot_id, mip, p_voxel, shade, hit_axis, ray_dir);
                    } else {
                        base_color = voxel_color_in(g, slot_id, mip, p_voxel, shade);
                    }
                    let lit = apply_fog(base_color, t_hit);
                    if (u.terrain_has_translucent == 0u) {
                        // Opaque fast-path: unchanged first hit.
                        out.hit = true;
                        out.t = t_hit;
                        out.color = lit;
                        return out;
                    }
                    let packed = voxel_packed_in(g, slot_id, mip, p_voxel);
                    let mat_id = terrain_material_id(packed);
                    let mm = materials_pal[mat_id];
                    if (mm.mode == 0u) {
                        // Opaque surface backs the translucent layers in front.
                        out.hit = true;
                        out.t = t_hit;
                        out.color = select(lit, accum + trans * lit, touched);
                        return out;
                    }
                    let a = mm.alpha;
                    if (mm.mode == 3u) {
                        // Volumetric (Beer–Lambert): per-cell opacity weighted
                        // by the ray's path length (voxel units); occludes.
                        let t_exit = min(t_max_voxel.x, min(t_max_voxel.y, t_max_voxel.z));
                        let seg_len = max(t_exit - t_hit, 0.0) * ray_dir_len / vsize;
                        let eff_a = 1.0 - pow(1.0 - a, seg_len);
                        accum = accum + trans * eff_a * lit;
                        trans = trans * (1.0 - eff_a);
                        touched = true;
                        prev_mat = mat_id;
                        if (trans < (1.0 / 256.0)) {
                            out.hit = true;
                            out.t = t_hit;
                            out.color = accum;
                            return out;
                        }
                    } else if (!prev_solid || mat_id != prev_mat) {
                        // AlphaBlend / Additive: one layer per solid-run entry
                        // / material change (thickness-independent).
                        accum = accum + trans * a * lit;
                        if (mm.mode != 2u) { trans = trans * (1.0 - a); }
                        touched = true;
                        prev_mat = mat_id;
                        if (trans < (1.0 / 256.0)) {
                            out.hit = true;
                            out.t = t_hit;
                            out.color = accum;
                            return out;
                        }
                    }
                    prev_solid = true;
                } else {
                    prev_solid = false;
                }
                if (t_max_voxel.x < t_max_voxel.y && t_max_voxel.x < t_max_voxel.z) {
                    t_hit = t_max_voxel.x;
                    p_voxel.x = p_voxel.x + step_chunk.x;
                    t_max_voxel.x = t_max_voxel.x + t_delta_voxel.x;
                    hit_axis = 0;
                    if (p_voxel.x < 0 || p_voxel.x >= vsid_mip) {
                        break;
                    }
                } else if (t_max_voxel.y < t_max_voxel.z) {
                    t_hit = t_max_voxel.y;
                    p_voxel.y = p_voxel.y + step_chunk.y;
                    t_max_voxel.y = t_max_voxel.y + t_delta_voxel.y;
                    hit_axis = 1;
                    if (p_voxel.y < 0 || p_voxel.y >= vsid_mip) {
                        break;
                    }
                } else {
                    t_hit = t_max_voxel.z;
                    p_voxel.z = p_voxel.z + step_chunk.z;
                    t_max_voxel.z = t_max_voxel.z + t_delta_voxel.z;
                    hit_axis = 2;
                    if (p_voxel.z < 0 || p_voxel.z >= cz_mip) {
                        break;
                    }
                }
            }
        }

        if (t_max_chunk.x < t_max_chunk.y && t_max_chunk.x < t_max_chunk.z) {
            t_enter = t_max_chunk.x;
            p_chunk.x = p_chunk.x + step_chunk.x;
            t_max_chunk.x = t_max_chunk.x + t_delta_chunk.x;
            entry_axis = 0;
        } else if (t_max_chunk.y < t_max_chunk.z) {
            t_enter = t_max_chunk.y;
            p_chunk.y = p_chunk.y + step_chunk.y;
            t_max_chunk.y = t_max_chunk.y + t_delta_chunk.y;
            entry_axis = 1;
        } else {
            t_enter = t_max_chunk.z;
            p_chunk.z = p_chunk.z + step_chunk.z;
            t_max_chunk.z = t_max_chunk.z + t_delta_chunk.z;
            entry_axis = 2;
        }
    }
    return finalize_sky_grid(touched, accum, trans, ray_dir);
}

@compute @workgroup_size(8, 8)
fn render_scene(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.screen_size.x || gid.y >= u.screen_size.y) {
        return;
    }

    let aspect = f32(u.screen_size.x) / f32(u.screen_size.y);
    let half_h = tan(u.fov_y_rad * 0.5);
    let half_w = half_h * aspect;
    let ndc_x = (f32(gid.x) + 0.5) / f32(u.screen_size.x) * 2.0 - 1.0;
    let ndc_y_top_pos = 1.0 - (f32(gid.y) + 0.5) / f32(u.screen_size.y) * 2.0;

    var best_t: f32 = T_INF;
    // Sky direction = the per-pixel ray of the dedicated world/sky
    // camera. Valid regardless of grid_count (a grid-less scene has no
    // grid ray), so a sprite-only / empty scene paints a real sky.
    let sky_dir = normalize(
        u.sky_cam.forward
        + ndc_x * half_w * u.sky_cam.right
        - ndc_y_top_pos * half_h * u.sky_cam.down
    );
    var best_color = vec3<f32>(0.6, 0.7, 0.85);
    var any_hit = false;

    for (var g: u32 = 0u; g < u.grid_count; g = g + 1u) {
        let cam = grid_cameras[g];
        let ray_dir = normalize(
            cam.forward
            + ndc_x * half_w * cam.right
            - ndc_y_top_pos * half_h * cam.down
        );
        let hit = march_grid(g, cam.pos, ray_dir, best_t);
        if (hit.hit && hit.t < best_t) {
            best_t = hit.t;
            best_color = hit.color;
            any_hit = true;
        }
    }
    if (!any_hit) {
        best_color = sky_color(sky_dir);
    }

    output[gid.y * u.screen_size.x + gid.x] = pack4x8unorm(vec4<f32>(best_color, 1.0));
    if (u.write_depth != 0u) {
        let pix_idx = gid.y * u.screen_size.x + gid.x;
        depth_buffer[pix_idx] = bitcast<u32>(best_t);
    }
}
