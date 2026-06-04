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
    vsid: u32,
    total_chunks: u32,
    _pad0: u32,
    chunks_dims: vec3<u32>,
    _pad1: u32,
    origin_chunk: vec3<i32>,
    _pad2: u32,
};

struct Uniforms {
    fov_y_rad: f32,
    grid_count: u32,
    max_outer_steps: u32,
    _pad0: u32,
    screen_size: vec2<u32>,
    _pad1: vec2<u32>,
    cameras: array<PerGridCamera, 16>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> all_occupancy: array<u32>;
@group(0) @binding(2) var<storage, read> all_color_offsets: array<u32>;
@group(0) @binding(3) var<storage, read> all_colors: array<u32>;
@group(0) @binding(4) var<storage, read> all_chunk_colors_base: array<u32>;
@group(0) @binding(5) var<storage, read> all_chunk_occupancy: array<u32>;
@group(0) @binding(6) var<storage, read> grid_static_meta: array<GridStaticMeta>;
@group(0) @binding(7) var output: texture_storage_2d<rgba8unorm, write>;

fn voxel_solid_in(g: u32, meta_id: u32, p_voxel: vec3<i32>) -> bool {
    let m = grid_static_meta[g];
    let col_idx = u32(p_voxel.x) + u32(p_voxel.y) * m.vsid;
    let cols_per_chunk = m.vsid * m.vsid;
    let occ_base = m.occupancy_offset + meta_id * cols_per_chunk * OCC_WORDS_PER_COLUMN;
    let col_word_base = occ_base + col_idx * OCC_WORDS_PER_COLUMN;
    let z_word = u32(p_voxel.z) >> 5u;
    let z_bit = u32(p_voxel.z) & 31u;
    return (all_occupancy[col_word_base + z_word] & (1u << z_bit)) != 0u;
}

fn voxel_color_in(g: u32, meta_id: u32, p_voxel: vec3<i32>) -> vec3<f32> {
    let m = grid_static_meta[g];
    let col_idx = u32(p_voxel.x) + u32(p_voxel.y) * m.vsid;
    let cols_per_chunk = m.vsid * m.vsid;
    let occ_base = m.occupancy_offset + meta_id * cols_per_chunk * OCC_WORDS_PER_COLUMN;
    let col_word_base = occ_base + col_idx * OCC_WORDS_PER_COLUMN;
    let z_word = u32(p_voxel.z) >> 5u;
    let z_bit = u32(p_voxel.z) & 31u;

    var rank: u32 = 0u;
    for (var w: u32 = 0u; w < z_word; w = w + 1u) {
        rank = rank + countOneBits(all_occupancy[col_word_base + w]);
    }
    var mask: u32 = 0u;
    if (z_bit > 0u) {
        mask = (1u << z_bit) - 1u;
    }
    rank = rank + countOneBits(all_occupancy[col_word_base + z_word] & mask);

    let offsets_base = m.color_offsets_offset + meta_id * (cols_per_chunk + 1u);
    let chunk_local_offset = all_color_offsets[offsets_base + col_idx];
    let chunk_colors_base = all_chunk_colors_base[m.chunk_colors_base_offset + meta_id];
    let packed = all_colors[m.colors_offset + chunk_colors_base + chunk_local_offset + rank];

    let a = f32((packed >> 24u) & 0xffu);
    let r = f32((packed >> 16u) & 0xffu);
    let g_chan = f32((packed >> 8u) & 0xffu);
    let b = f32(packed & 0xffu);
    let brightness = a * (1.0 / 128.0);
    return vec3<f32>(r, g_chan, b) * (brightness / 255.0);
}

fn meta_idx_of(g: u32, chunk_idx: vec3<i32>) -> i32 {
    let m = grid_static_meta[g];
    let rel = chunk_idx - m.origin_chunk;
    if (rel.x < 0 || rel.y < 0 || rel.z < 0 ||
        u32(rel.x) >= m.chunks_dims.x ||
        u32(rel.y) >= m.chunks_dims.y ||
        u32(rel.z) >= m.chunks_dims.z) {
        return -1;
    }
    return rel.x
        + rel.y * i32(m.chunks_dims.x)
        + rel.z * i32(m.chunks_dims.x * m.chunks_dims.y);
}

fn chunk_has_content(g: u32, meta_id: i32) -> bool {
    if (meta_id < 0) {
        return false;
    }
    let m = grid_static_meta[g];
    let mi = u32(meta_id);
    return (all_chunk_occupancy[m.chunk_occupancy_offset + (mi >> 5u)] & (1u << (mi & 31u))) != 0u;
}

fn sky_color(dir: vec3<f32>) -> vec3<f32> {
    let down_amount = clamp(dir.z * 0.5 + 0.5, 0.0, 1.0);
    let zenith = vec3<f32>(0.18, 0.28, 0.55);
    let horizon = vec3<f32>(0.66, 0.74, 0.88);
    return mix(zenith, horizon, down_amount);
}

fn shield_parallel(t_max: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    var t = t_max;
    if (dir.x == 0.0) { t.x = T_INF; }
    if (dir.y == 0.0) { t.y = T_INF; }
    if (dir.z == 0.0) { t.z = T_INF; }
    return t;
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
        let meta_id = meta_idx_of(g, p_chunk);
        if (chunk_has_content(g, meta_id)) {
            let t_chunk_exit = min(t_max_chunk.x, min(t_max_chunk.y, t_max_chunk.z));
            let entry_world = ray_origin + t_enter * ray_dir;
            let chunk_origin_world = vec3<f32>(p_chunk) * chunk_dim;
            let entry_in_chunk = entry_world - chunk_origin_world;
            var p_voxel = vec3<i32>(floor(entry_in_chunk));
            p_voxel = clamp(
                p_voxel,
                vec3<i32>(0),
                vec3<i32>(i32(m.vsid - 1u), i32(m.vsid - 1u), i32(CHUNK_Z - 1u)),
            );

            let next_voxel_world = vec3<f32>(
                select(f32(p_voxel.x), f32(p_voxel.x + 1), step_chunk.x > 0)
                    + chunk_origin_world.x,
                select(f32(p_voxel.y), f32(p_voxel.y + 1), step_chunk.y > 0)
                    + chunk_origin_world.y,
                select(f32(p_voxel.z), f32(p_voxel.z + 1), step_chunk.z > 0)
                    + chunk_origin_world.z,
            );
            var t_max_voxel = shield_parallel(
                (next_voxel_world - ray_origin) / ray_dir,
                ray_dir,
            );
            let t_delta_voxel = abs(1.0 / ray_dir);
            var t_hit: f32 = t_enter;

            for (var iv: u32 = 0u; iv < MAX_INNER_STEPS; iv = iv + 1u) {
                if (voxel_solid_in(g, u32(meta_id), p_voxel)) {
                    if (t_hit < best_t) {
                        out.hit = true;
                        out.t = t_hit;
                        out.color = voxel_color_in(g, u32(meta_id), p_voxel);
                        return out;
                    } else {
                        return out;
                    }
                }
                if (t_max_voxel.x < t_max_voxel.y && t_max_voxel.x < t_max_voxel.z) {
                    t_hit = t_max_voxel.x;
                    p_voxel.x = p_voxel.x + step_chunk.x;
                    t_max_voxel.x = t_max_voxel.x + t_delta_voxel.x;
                    if (p_voxel.x < 0 || u32(p_voxel.x) >= m.vsid) {
                        break;
                    }
                } else if (t_max_voxel.y < t_max_voxel.z) {
                    t_hit = t_max_voxel.y;
                    p_voxel.y = p_voxel.y + step_chunk.y;
                    t_max_voxel.y = t_max_voxel.y + t_delta_voxel.y;
                    if (p_voxel.y < 0 || u32(p_voxel.y) >= m.vsid) {
                        break;
                    }
                } else {
                    t_hit = t_max_voxel.z;
                    p_voxel.z = p_voxel.z + step_chunk.z;
                    t_max_voxel.z = t_max_voxel.z + t_delta_voxel.z;
                    if (p_voxel.z < 0 || u32(p_voxel.z) >= CHUNK_Z) {
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
    // Sky-direction we paint on miss is derived from grid 0's
    // camera (the world-frame ground grid). For demos with the
    // ground at identity transform this is just the world camera.
    var sky_dir = vec3<f32>(0.0, 0.0, 1.0);
    var best_color = vec3<f32>(0.6, 0.7, 0.85);
    var any_hit = false;

    for (var g: u32 = 0u; g < u.grid_count; g = g + 1u) {
        let cam = u.cameras[g];
        let ray_dir = normalize(
            cam.forward
            + ndc_x * half_w * cam.right
            - ndc_y_top_pos * half_h * cam.down
        );
        if (g == 0u) {
            sky_dir = ray_dir;
        }
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

    textureStore(output, vec2<i32>(gid.xy), vec4<f32>(best_color, 1.0));
}
