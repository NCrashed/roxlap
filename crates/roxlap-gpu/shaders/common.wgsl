// QE.8 — helpers shared VERBATIM by the two DDA marchers
// (`scene_dda.wgsl` + `sprite_model_dda.wgsl`). Prepended to both by
// `shader_src.rs` (WGSL module scope is order-independent), so the
// bodies exist exactly once — the pre-QE.8 copies had already drifted
// in names/comments. Everything here may reference only the `u`
// uniform fields both shaders declare identically (`fog_color`,
// `fog_far`).

const T_INF: f32 = 1.0e30;

// Ray/DDA guard: axes the ray is parallel to never advance (their
// next-boundary t stays at infinity).
fn shield_parallel(t_max: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    var t = t_max;
    if (dir.x == 0.0) { t.x = T_INF; }
    if (dir.y == 0.0) { t.y = T_INF; }
    if (dir.z == 0.0) { t.z = T_INF; }
    return t;
}

// GPU.8 fog blend. `t` is the world-space hit distance; below
// `fog_near` (packed in `fog_color.w`) the hit shows through fully;
// above `fog_far` only the fog colour shows. Linear ramp (not
// smoothstep) to match the CPU / DDA fog
// `clamp((t - near) / (far - near))` — smoothstep's S-curve gave
// visibly weaker mid-distance fog than the DDA renderer.
fn apply_fog(hit_color: vec3<f32>, t: f32) -> vec3<f32> {
    let fog_near = u.fog_color.w;
    let factor = clamp((t - fog_near) / max(u.fog_far - fog_near, 1e-6), 0.0, 1.0);
    return mix(hit_color, u.fog_color.rgb, factor);
}

// DL.4 — point-light distance falloff: smooth quadratic from 1 at the
// light to 0 at `radius` (hard cut).
fn point_falloff(d: f32, radius: f32) -> f32 {
    let x = clamp(1.0 - d / radius, 0.0, 1.0);
    return x * x;
}

// SL — spot (cone) angular mask. `ldir` = unit dir from surface TO
// light; `axis` = cone axis. 1.0 for a pure point light
// (`cos_outer <= -0.999`); else soft smoothstep outer→inner, hard
// `step` when the angles coincide (WGSL smoothstep is undefined for
// equal edges).
fn spot_cone(ldir: vec3<f32>, axis: vec3<f32>, cos_inner: f32, cos_outer: f32) -> f32 {
    if (cos_outer <= -0.999) { return 1.0; }
    let cd = dot(-ldir, axis);
    if (cos_inner <= cos_outer) { return step(cos_outer, cd); }
    return smoothstep(cos_outer, cos_inner, cd);
}

// DL.6 — cel quantization: snap a 0..1 factor to `bands + 1` discrete
// levels.
fn cel_band(x: f32, bands: u32) -> f32 {
    let b = f32(bands);
    return clamp(round(x * b) / b, 0.0, 1.0);
}
