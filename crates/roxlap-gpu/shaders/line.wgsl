// roxlap-gpu line.wgsl — depth-tested 3D debug lines (L3.2).
//
// Editor gizmos / debug overlays (bounding boxes, floor grids, axes)
// drawn over the marched frame. Geometry is expanded to screen-space
// quads CPU-side (`build_line_vertices`); each vertex carries:
//   - NDC position (the quad corner),
//   - the euclidean scene-depth of its source endpoint (= the marcher's
//     `best_t` metric: world distance from the camera),
//   - an alpha-blended colour, and
//   - a per-vertex depth-test flag.
//
// The fragment shader compares the interpolated euclidean depth against
// the scene-DDA depth buffer (`best_t`, smaller = closer; `T_INF` for
// sky) so terrain in front occludes a line behind it. `depth_test = 0`
// lines (e.g. a hover highlight) skip the test and always draw on top.
// Alpha blending is the pipeline's job (src-alpha over the loaded frame).

struct Params {
    screen_w: u32,
    screen_h: u32,
    depth_bias: f32,
    // 1 = no scene depth buffer bound (sprite-only / empty scene) →
    // skip the test so the dummy 1-word buffer is never indexed.
    no_depth: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> depth_buf: array<u32>;

struct VsIn {
    @location(0) pos: vec2<f32>,
    @location(1) depth: f32,
    @location(2) depth_test: f32,
    @location(3) color: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) depth: f32,
    @location(2) depth_test: f32,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var o: VsOut;
    o.clip = vec4<f32>(in.pos, 0.0, 1.0);
    o.color = in.color;
    o.depth = in.depth;
    o.depth_test = in.depth_test;
    return o;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    if (in.depth_test > 0.5 && params.no_depth == 0u) {
        // `clip` is the framebuffer pixel coordinate in the fragment
        // stage (origin top-left) — matches the depth buffer's row-major
        // `gid.y * screen_w + gid.x` layout.
        let px = u32(in.clip.x);
        let py = u32(in.clip.y);
        if (px < params.screen_w && py < params.screen_h) {
            let scene_t = bitcast<f32>(depth_buf[py * params.screen_w + px]);
            if (in.depth > scene_t + params.depth_bias) {
                discard;
            }
        }
    }
    return in.color;
}
