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
const MAX_GRIDS: u32 = 16u;
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
    cameras: array<PerGridCamera, 16>,
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
    _pad2: u32,
    _pad3: u32,
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

fn voxel_color_in(g: u32, meta_id: u32, mip: u32, p_voxel: vec3<i32>, face_shade: f32) -> vec3<f32> {
    let vsid_mip = grid_static_meta[g].vsid >> mip;
    let col_idx = u32(p_voxel.x) + u32(p_voxel.y) * vsid_mip;
    let col_word_base = col_word_base_mip(g, meta_id, mip, p_voxel);
    let z_word = u32(p_voxel.z) >> 5u;
    let z_bit = u32(p_voxel.z) & 31u;

    // Rank = number of TEXTURED voxels below z. Indexes the colour.
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

    // A bedrock hit (solid but not textured) inherits the colour of the
    // textured surface directly above it: that's `rank - 1` (rank here
    // counts surfaces strictly above). A textured hit uses `rank`.
    let is_textured = (z_word_bits & (1u << z_bit)) != 0u;
    var color_index = rank;
    if (!is_textured && rank > 0u) {
        color_index = rank - 1u;
    }

    // Cumulative-within-slot colour offsets: the mip's sub-table
    // lives at `mip_coff_rel[mip]`, and its values already include
    // every finer mip's colour count, so `chunk_colors_base + value
    // + index` indexes the slot's concatenated colour block directly.
    let offsets_base = grid_static_meta[g].color_offsets_offset
        + meta_id * grid_static_meta[g].offsets_words_per_slot
        + grid_static_meta[g].mip_coff_rel[mip];
    let chunk_local_offset = all_color_offsets[offsets_base + col_idx];
    let chunk_colors_base =
        all_chunk_colors_base[grid_static_meta[g].chunk_colors_base_offset + meta_id];
    let packed = all_colors[grid_static_meta[g].colors_offset + chunk_colors_base
        + chunk_local_offset + color_index];

    let a = f32((packed >> 24u) & 0xffu);
    let r = f32((packed >> 16u) & 0xffu);
    let g_chan = f32((packed >> 8u) & 0xffu);
    let b = f32(packed & 0xffu);
    // Side-shade: reduce the brightness byte by the hit face's shade
    // before the /128 divide (CPU grouscan_shade equivalent). With no
    // baked light (flat a=0x80) this is pure runtime side-shading; with
    // baked light it stacks, exactly like voxlap.
    let brightness = max(0.0, a - face_shade) * (1.0 / 128.0);
    return vec3<f32>(r, g_chan, b) * (brightness / 255.0);
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
// fog colour shows. Smoothstep gives a soft mid-band.
fn apply_fog(hit_color: vec3<f32>, t: f32) -> vec3<f32> {
    let fog_near = u.fog_color.w;
    let factor = smoothstep(fog_near, u.fog_far, t);
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

fn march_grid(
    g: u32,
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    best_t: f32,
) -> GridHit {
    let m = grid_static_meta[g];
    let chunk_dim = vec3<f32>(f32(m.vsid), f32(m.vsid), f32(CHUNK_Z));

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
    var out: GridHit;
    out.hit = false;
    out.t = T_INF;
    out.color = vec3<f32>(0.0);

    for (var step: u32 = 0u; step < u.max_outer_steps; step = step + 1u) {
        if (t_enter > best_t) {
            return out; // no closer hit possible in this grid
        }
        // GPU.13.0 — once the ray has left the occupied chunk-AABB
        // along its travel direction, no resident chunk lies ahead:
        // stop instead of stepping empty space to max_outer_steps.
        if (aabb_passed(g, p_chunk, step_chunk)) {
            return out;
        }
        let slot_id = slot_idx_of(g, p_chunk);
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
            // side-shading). Defaults to z for an iv==0 hit (camera
            // embedded in solid — a 1-voxel edge case); surfaces hit
            // after any travel use the real last-stepped axis.
            var hit_axis: i32 = 2;

            for (var iv: u32 = 0u; iv < MAX_INNER_STEPS; iv = iv + 1u) {
                if (voxel_solid_in(g, slot_id, mip, p_voxel)) {
                    if (t_hit < best_t) {
                        out.hit = true;
                        out.t = t_hit;
                        let shade = side_shade_for(hit_axis, ray_dir);
                        out.color = apply_fog(
                            voxel_color_in(g, slot_id, mip, p_voxel, shade),
                            t_hit,
                        );
                        return out;
                    } else {
                        return out;
                    }
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
        } else if (t_max_chunk.y < t_max_chunk.z) {
            t_enter = t_max_chunk.y;
            p_chunk.y = p_chunk.y + step_chunk.y;
            t_max_chunk.y = t_max_chunk.y + t_delta_chunk.y;
        } else {
            t_enter = t_max_chunk.z;
            p_chunk.z = p_chunk.z + step_chunk.z;
            t_max_chunk.z = t_max_chunk.z + t_delta_chunk.z;
        }
    }
    return out;
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
        let cam = u.cameras[g];
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
