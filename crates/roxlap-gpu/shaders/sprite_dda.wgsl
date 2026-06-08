// GPU.9 — KV6 voxel sprite splatter compute shader.
// One thread per sprite voxel. Project via cameras[0] (world
// camera; grid 0 = ground at identity). Plain depth-test against
// depth_buffer (NO atomics — NVK suspected; lavapipe-validated).

const T_INF: f32 = 1.0e30;

struct PerGridCamera {
    pos: vec3<f32>, _pad0: f32,
    right: vec3<f32>, _pad1: f32,
    down: vec3<f32>, _pad2: f32,
    forward: vec3<f32>, _pad3: f32,
};
struct Uniforms {
    fov_y_rad: f32, grid_count: u32, max_outer_steps: u32, _pad0: u32,
    screen_size: vec2<u32>, _pad1: vec2<u32>,
    cameras: array<PerGridCamera, 16>,
    fog_color: vec4<f32>, fog_far: f32,
    write_depth: u32, _pad2: f32, _pad3: f32,
};
struct SpriteVoxel { world_pos: vec3<f32>, color: u32, };

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> sprite_voxels: array<SpriteVoxel>;
@group(0) @binding(2) var<storage, read_write> depth_buffer: array<u32>;
@group(0) @binding(3) var output: texture_storage_2d<rgba8unorm, write>;

fn apply_fog(hit_color: vec3<f32>, t: f32) -> vec3<f32> {
    let fog_near = u.fog_color.w;
    let factor = smoothstep(fog_near, u.fog_far, t);
    return mix(hit_color, u.fog_color.rgb, factor);
}

@compute @workgroup_size(64, 1, 1)
fn splat(@builtin(global_invocation_id) gid: vec3<u32>) {
    let voxel_count = arrayLength(&sprite_voxels);
    if (gid.x >= voxel_count) { return; }
    let v = sprite_voxels[gid.x];
    let cam = u.cameras[0];
    let rel = v.world_pos - cam.pos;
    let z_cam = dot(rel, cam.forward);
    if (z_cam <= 0.1) { return; }
    let x_cam = dot(rel, cam.right);
    let y_cam = dot(rel, cam.down);
    let aspect = f32(u.screen_size.x) / f32(u.screen_size.y);
    let half_h = tan(u.fov_y_rad * 0.5);
    let half_w = half_h * aspect;
    let cx = f32(u.screen_size.x) * 0.5;
    let cy = f32(u.screen_size.y) * 0.5;
    // Screen projection must invert scene_dda's ray generation
    // exactly so the splat lands where the world DDA would have hit
    // the same point. scene_dda builds the ray as
    //   forward + ndc_x*half_w*right - ndc_y_top_pos*half_h*down
    // with ndc_y_top_pos = +1 at the TOP. Inverting that gives a
    // `+` on the y term here (a `-` mirrors the sprite vertically,
    // which reads as the sprite "orbiting" against camera pitch).
    let sx_f = cx + (x_cam / z_cam) * (cx / half_w);
    let sy_f = cy + (y_cam / z_cam) * (cy / half_h);
    let voxel_t = length(rel);

    // Decode the voxel colour + lightmode-1 brightness once; every
    // pixel of this voxel's splat shares it.
    let packed = v.color;
    let a = f32((packed >> 24u) & 0xffu);
    let r = f32((packed >> 16u) & 0xffu);
    let g = f32((packed >> 8u) & 0xffu);
    let b = f32(packed & 0xffu);
    let brightness = a * (1.0 / 128.0);
    var col = vec3<f32>(r, g, b) * (brightness / 255.0);
    col = apply_fog(col, voxel_t);
    let out_col = vec4<f32>(col, 1.0);

    // Multi-pixel splat. A unit-size voxel projects to
    // `px_per_world / z_cam` screen pixels; `cx/half_w == cy/half_h`
    // so the scale is isotropic. Neighbouring voxels are 1 world
    // unit apart, so a square of that full side tiles edge-to-edge
    // and the model reads solid up close. Half-extent is clamped so
    // an extreme close-up can't explode the per-thread work.
    let px_per_world = cx / half_w;
    let splat_px = px_per_world / z_cam; // voxel_world_size = 1.0
    // Inflate ×1.3 so neighbouring splats overlap instead of merely
    // abutting: a face-depth-sized square leaves seams on surfaces
    // tilted away from the camera, where adjacent voxels are spaced
    // wider on screen than the face-depth estimate. The clamp only
    // exists to bound per-thread work when the camera is nearly
    // touching a voxel — 128 keeps the model solid until you're
    // essentially inside the sprite.
    let half_ext = i32(clamp(ceil(splat_px * 0.65), 0.0, 128.0));

    let cx_i = i32(floor(sx_f));
    let cy_i = i32(floor(sy_f));
    // Coarse cull: skip voxels whose whole splat is off-screen.
    if (cx_i + half_ext < 0 || cx_i - half_ext >= i32(u.screen_size.x)
        || cy_i + half_ext < 0 || cy_i - half_ext >= i32(u.screen_size.y)) {
        return;
    }

    for (var dy = -half_ext; dy <= half_ext; dy = dy + 1) {
        let py = cy_i + dy;
        if (py < 0 || u32(py) >= u.screen_size.y) { continue; }
        for (var dx = -half_ext; dx <= half_ext; dx = dx + 1) {
            let px = cx_i + dx;
            if (px < 0 || u32(px) >= u.screen_size.x) { continue; }
            let pix_idx = u32(py) * u.screen_size.x + u32(px);
            let cur_t = bitcast<f32>(depth_buffer[pix_idx]);
            if (voxel_t >= cur_t) { continue; }
            depth_buffer[pix_idx] = bitcast<u32>(voxel_t);
            textureStore(output, vec2<i32>(px, py), out_col);
        }
    }
}
