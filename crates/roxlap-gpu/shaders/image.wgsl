// roxlap-gpu image.wgsl — depth-tested world-space 2D image sprites.
//
// A flat textured quad placed in world space, composited over the
// marched frame (editor reference images, decals, blueprints). One
// dimension up from `line.wgsl`: instead of a solid colour the fragment
// samples an RGBA texture, multiplied by a per-quad tint.
//
// Vertices are projected + near-clipped CPU-side (`build_image_vertices`)
// and arrive as `(ndc.xy, w, depth, depth_test, uv)`. The vertex stage
// re-homogenises them — `clip = vec4(ndc * w, 0, w)` — so the rasterizer
// interpolates `uv` with full perspective correction (no affine warp on
// an obliquely-viewed quad). `depth` is the source vertex's euclidean
// world distance (= the marcher's `best_t`), perspective-interpolated for
// the manual depth test against the scene-DDA depth buffer.

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
@group(0) @binding(2) var img_tex: texture_2d<f32>;
@group(0) @binding(3) var img_samp: sampler;

struct VsIn {
    @location(0) ndc: vec2<f32>,
    @location(1) w: f32,
    @location(2) depth: f32,
    @location(3) depth_test: f32,
    @location(4) cutoff: f32,
    @location(5) uv: vec2<f32>,
    @location(6) tint: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) tint: vec4<f32>,
    @location(2) depth: f32,
    @location(3) depth_test: f32,
    @location(4) cutoff: f32,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var o: VsOut;
    // Re-homogenise so uv / depth interpolate perspective-correctly.
    // z = 0 keeps the vertex inside the [0, w] clip range (no depth
    // attachment — the depth test is manual in the fragment stage).
    o.clip = vec4<f32>(in.ndc * in.w, 0.0, in.w);
    o.uv = in.uv;
    o.tint = in.tint;
    o.depth = in.depth;
    o.depth_test = in.depth_test;
    o.cutoff = in.cutoff;
    return o;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    if (in.depth_test > 0.5 && params.no_depth == 0u) {
        let px = u32(in.clip.x);
        let py = u32(in.clip.y);
        if (px < params.screen_w && py < params.screen_h) {
            let scene_t = bitcast<f32>(depth_buf[py * params.screen_w + px]);
            if (in.depth > scene_t + params.depth_bias) {
                discard;
            }
        }
    }
    let texel = textureSample(img_tex, img_samp, in.uv);
    // Alpha cutoff: discard below-threshold texels (crisp pixel-art edges).
    if (texel.a < in.cutoff) {
        discard;
    }
    // Straight alpha in; the pipeline's ALPHA_BLENDING does the over-blend.
    return vec4<f32>(texel.rgb * in.tint.rgb, texel.a * in.tint.a);
}
