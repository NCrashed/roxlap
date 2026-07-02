// GPU.4 — outer DDA over chunks + inner DDA over voxels.
//
// Each pixel: build a ray from the camera basis, run an
// Amanatides–Woo DDA over chunk indices. At each step, the
// chunk-occupancy bitmap says "any voxels here?". If yes, run an
// inner Amanatides–Woo DDA bounded to that chunk's voxel range; if
// no, advance to the next chunk in one outer step.
//
// Coordinate convention:
// * `camera_pos` is in grid-local voxel units (host translates from
//   world coords). For grid 0 at identity transform, world == grid.
// * Z is DOWN (voxlap). `camera_down` is the +z direction; pixel
//   (0, 0) at the top of the screen maps to -camera_down.
// * Chunks are XY × CHUNK_Z voxels (typically 128 × 128 × 256).
//   `chunks_dims` is the count along each axis; the grid spans
//   chunk indices [origin_chunk, origin_chunk + chunks_dims).
// * Bedrock voxels are NOT in occupancy (bedrock-as-air refactor —
//   `decompress.rs` strips them). Rays through bedrock fall to sky.

const OCC_WORDS_PER_COLUMN: u32 = 8u; // CHUNK_Z (256) / 32
const CHUNK_Z: u32 = 256u;
// Worst-case inner steps per chunk = vsid + vsid + CHUNK_Z ~= 512.
// Keep loose so a near-axis-aligned ray crossing the chunk diagonal
// terminates cleanly.
const MAX_INNER_STEPS: u32 = 768u;

struct Uniforms {
    camera_pos: vec3<f32>,
    _pad0: f32,
    camera_right: vec3<f32>,
    _pad1: f32,
    camera_down: vec3<f32>,
    _pad2: f32,
    camera_forward: vec3<f32>,
    fov_y_rad: f32,
    screen_size: vec2<u32>,
    vsid: u32,
    max_outer_steps: u32,
    chunks_dims: vec3<u32>,
    _pad3: u32,
    origin_chunk: vec3<i32>,
    _pad4: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> occupancy: array<u32>;
@group(0) @binding(2) var<storage, read> color_offsets: array<u32>;
@group(0) @binding(3) var<storage, read> colors: array<u32>;
@group(0) @binding(4) var<storage, read> chunk_colors_base: array<u32>;
@group(0) @binding(5) var<storage, read> chunk_occupancy: array<u32>;
@group(0) @binding(6) var output: texture_storage_2d<rgba8unorm, write>;

// ---- helpers --------------------------------------------------------------

fn meta_idx_of(chunk_idx: vec3<i32>) -> i32 {
    let rel = chunk_idx - u.origin_chunk;
    if (rel.x < 0 || rel.y < 0 || rel.z < 0 ||
        u32(rel.x) >= u.chunks_dims.x ||
        u32(rel.y) >= u.chunks_dims.y ||
        u32(rel.z) >= u.chunks_dims.z) {
        return -1;
    }
    return rel.x
        + rel.y * i32(u.chunks_dims.x)
        + rel.z * i32(u.chunks_dims.x * u.chunks_dims.y);
}

fn chunk_has_content(meta_id: i32) -> bool {
    if (meta_id < 0) {
        return false;
    }
    let mi = u32(meta_id);
    return (chunk_occupancy[mi >> 5u] & (1u << (mi & 31u))) != 0u;
}

fn voxel_solid(meta_id: u32, p_in_chunk: vec3<i32>) -> bool {
    let col_idx = u32(p_in_chunk.x) + u32(p_in_chunk.y) * u.vsid;
    let cols_per_chunk = u.vsid * u.vsid;
    let occ_base = meta_id * cols_per_chunk * OCC_WORDS_PER_COLUMN;
    let col_word_base = occ_base + col_idx * OCC_WORDS_PER_COLUMN;
    let z_word = u32(p_in_chunk.z) >> 5u;
    let z_bit = u32(p_in_chunk.z) & 31u;
    return (occupancy[col_word_base + z_word] & (1u << z_bit)) != 0u;
}

fn voxel_color(meta_id: u32, p_in_chunk: vec3<i32>) -> vec3<f32> {
    let col_idx = u32(p_in_chunk.x) + u32(p_in_chunk.y) * u.vsid;
    let cols_per_chunk = u.vsid * u.vsid;
    let occ_base = meta_id * cols_per_chunk * OCC_WORDS_PER_COLUMN;
    let col_word_base = occ_base + col_idx * OCC_WORDS_PER_COLUMN;
    let z_word = u32(p_in_chunk.z) >> 5u;
    let z_bit = u32(p_in_chunk.z) & 31u;

    var rank: u32 = 0u;
    for (var w: u32 = 0u; w < z_word; w = w + 1u) {
        rank = rank + countOneBits(occupancy[col_word_base + w]);
    }
    var mask: u32 = 0u;
    if (z_bit > 0u) {
        mask = (1u << z_bit) - 1u;
    }
    rank = rank + countOneBits(occupancy[col_word_base + z_word] & mask);

    let offsets_base = meta_id * (cols_per_chunk + 1u);
    let chunk_local_offset = color_offsets[offsets_base + col_idx];
    let packed = colors[chunk_colors_base[meta_id] + chunk_local_offset + rank];

    let a = f32((packed >> 24u) & 0xffu);
    let r = f32((packed >> 16u) & 0xffu);
    let g = f32((packed >> 8u) & 0xffu);
    let b = f32(packed & 0xffu);
    let brightness = a * (1.0 / 128.0);
    return vec3<f32>(r, g, b) * (brightness / 255.0);
}

fn sky_color(dir: vec3<f32>) -> vec3<f32> {
    let down_amount = clamp(dir.z * 0.5 + 0.5, 0.0, 1.0);
    let zenith = vec3<f32>(0.18, 0.28, 0.55);
    let horizon = vec3<f32>(0.66, 0.74, 0.88);
    return mix(zenith, horizon, down_amount);
}

// Set `t_max` infinity for axes the ray is parallel to. WGSL has no
// f32::INFINITY constant; use a very large value instead.
fn shield_parallel(t_max: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    var t = t_max;
    if (dir.x == 0.0) { t.x = 1.0e30; }
    if (dir.y == 0.0) { t.y = 1.0e30; }
    if (dir.z == 0.0) { t.z = 1.0e30; }
    return t;
}

// ---- main marcher ---------------------------------------------------------

@compute @workgroup_size(8, 8)
fn render_grid(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.screen_size.x || gid.y >= u.screen_size.y) {
        return;
    }

    // Build the per-pixel ray.
    let aspect = f32(u.screen_size.x) / f32(u.screen_size.y);
    let half_h = tan(u.fov_y_rad * 0.5);
    let half_w = half_h * aspect;
    let ndc_x = (f32(gid.x) + 0.5) / f32(u.screen_size.x) * 2.0 - 1.0;
    let ndc_y_top_pos = 1.0 - (f32(gid.y) + 0.5) / f32(u.screen_size.y) * 2.0;
    let dir = normalize(
        u.camera_forward
        + ndc_x * half_w * u.camera_right
        - ndc_y_top_pos * half_h * u.camera_down
    );

    let chunk_dim = vec3<f32>(f32(u.vsid), f32(u.vsid), f32(CHUNK_Z));

    // Outer DDA setup in chunk-space (each chunk is a "cell").
    var p_chunk = vec3<i32>(floor(u.camera_pos / chunk_dim));
    let step_chunk = vec3<i32>(sign(dir));
    let t_delta_chunk = abs(chunk_dim / dir);
    let next_boundary_chunk = vec3<f32>(
        select(f32(p_chunk.x), f32(p_chunk.x + 1), step_chunk.x > 0) * chunk_dim.x,
        select(f32(p_chunk.y), f32(p_chunk.y + 1), step_chunk.y > 0) * chunk_dim.y,
        select(f32(p_chunk.z), f32(p_chunk.z + 1), step_chunk.z > 0) * chunk_dim.z,
    );
    var t_max_chunk = shield_parallel(
        (next_boundary_chunk - u.camera_pos) / dir,
        dir,
    );

    // `t_enter` = world-units t at which the ray entered the
    // current chunk. Used to pick the inner DDA's voxel-entry
    // point.
    var t_enter: f32 = 0.0;
    var hit_color = sky_color(dir);
    var done = false;

    for (var step: u32 = 0u; step < u.max_outer_steps; step = step + 1u) {
        let meta_id = meta_idx_of(p_chunk);
        if (chunk_has_content(meta_id)) {
            // Inner DDA bounded to this chunk.
            let t_chunk_exit = min(t_max_chunk.x, min(t_max_chunk.y, t_max_chunk.z));
            // Voxel coords inside the chunk. Compute from the
            // ray's entry position; clamp to the chunk's [0, vsid)
            // × [0, CHUNK_Z) range to absorb float-rounding error
            // at chunk boundaries.
            let entry_world = u.camera_pos + t_enter * dir;
            let chunk_origin_world = vec3<f32>(p_chunk) * chunk_dim;
            let entry_in_chunk = entry_world - chunk_origin_world;
            var p_voxel = vec3<i32>(floor(entry_in_chunk));
            p_voxel = clamp(
                p_voxel,
                vec3<i32>(0),
                vec3<i32>(i32(u.vsid - 1u), i32(u.vsid - 1u), i32(CHUNK_Z - 1u)),
            );

            // Voxel-level DDA. `t_max_voxel` is in WORLD units so
            // we can compare directly against `t_chunk_exit`.
            let next_voxel_world = vec3<f32>(
                select(f32(p_voxel.x), f32(p_voxel.x + 1), step_chunk.x > 0)
                    + chunk_origin_world.x,
                select(f32(p_voxel.y), f32(p_voxel.y + 1), step_chunk.y > 0)
                    + chunk_origin_world.y,
                select(f32(p_voxel.z), f32(p_voxel.z + 1), step_chunk.z > 0)
                    + chunk_origin_world.z,
            );
            var t_max_voxel = shield_parallel(
                (next_voxel_world - u.camera_pos) / dir,
                dir,
            );
            let t_delta_voxel = abs(1.0 / dir);

            for (var iv: u32 = 0u; iv < MAX_INNER_STEPS; iv = iv + 1u) {
                if (voxel_solid(u32(meta_id), p_voxel)) {
                    hit_color = voxel_color(u32(meta_id), p_voxel);
                    done = true;
                    break;
                }
                // Step voxel; if we leave the chunk, return to the
                // outer DDA. Use the smallest-`t_max` axis like the
                // standard Amanatides–Woo.
                if (t_max_voxel.x < t_max_voxel.y && t_max_voxel.x < t_max_voxel.z) {
                    p_voxel.x = p_voxel.x + step_chunk.x;
                    t_max_voxel.x = t_max_voxel.x + t_delta_voxel.x;
                    if (p_voxel.x < 0 || u32(p_voxel.x) >= u.vsid) {
                        break;
                    }
                } else if (t_max_voxel.y < t_max_voxel.z) {
                    p_voxel.y = p_voxel.y + step_chunk.y;
                    t_max_voxel.y = t_max_voxel.y + t_delta_voxel.y;
                    if (p_voxel.y < 0 || u32(p_voxel.y) >= u.vsid) {
                        break;
                    }
                } else {
                    p_voxel.z = p_voxel.z + step_chunk.z;
                    t_max_voxel.z = t_max_voxel.z + t_delta_voxel.z;
                    if (p_voxel.z < 0 || u32(p_voxel.z) >= CHUNK_Z) {
                        break;
                    }
                }
            }
            if (done) {
                break;
            }
        }

        // Outer step: advance to the next chunk along the ray.
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

        // (PF.13/G8 — no outside-the-bbox early-out here on purpose:
        // a ray can leave through one face while still moving toward
        // voxels behind another; `max_outer_steps` terminates wayward
        // rays. A dead per-step bounds computation used to sit here.)
    }

    textureStore(output, vec2<i32>(gid.xy), vec4<f32>(hit_color, 1.0));
}
