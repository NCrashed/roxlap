//! QE.8b — WGSL shader-source assembly, split verbatim out of
//! `lib.rs`: the XS.4 splice helpers that swap the sprite/scene
//! shadow stubs for the real cross-pass snippets on capable devices,
//! plus the naga validation test covering every shader (including
//! both spliced variants).

/// XS.4.2 — build the sprite-pass shader source. On a sprite-shadow-capable
/// device, splice `sprite_terrain_shadow.wgsl` over the `//XS4_STUB_BEGIN`..
/// `//XS4_STUB_END` block so `shadow_occluded_world` becomes the real terrain
/// march (+ the occupancy bindings 16..23); otherwise the stub keeps GPU
/// sprites unshadowed. The base file is always valid WGSL (the stub variant),
/// so `wgsl_shaders_validate` covers the fallback path.
pub(crate) fn sprite_shader_source(capable: bool) -> String {
    let base = include_str!("../shaders/sprite_model_dda.wgsl");
    if !capable {
        return base.to_string();
    }
    let snippet = include_str!("../shaders/sprite_terrain_shadow.wgsl");
    const BEGIN: &str = "//XS4_STUB_BEGIN";
    const END: &str = "//XS4_STUB_END";
    let (Some(b), Some(e)) = (base.find(BEGIN), base.find(END)) else {
        // Markers missing — fail loud rather than silently shipping the stub.
        panic!("sprite_model_dda.wgsl: XS4 stub markers not found");
    };
    let e_end = e + END.len();
    let mut out = String::with_capacity(base.len() + snippet.len());
    out.push_str(&base[..b]);
    out.push_str(snippet);
    out.push_str(&base[e_end..]);
    out
}

/// XS.4.3 — build the scene-pass shader source. On a sprite-shadow-capable
/// device, splice `scene_sprite_shadow.wgsl` over the `//XS4C_STUB_BEGIN`..
/// `//XS4C_STUB_END` block so `sprites_occlude` marches the sprite registry
/// (+ bindings 19..21) and terrain receives sprite-cast shadows; otherwise the
/// stub returns false. The base file is always valid WGSL (the stub variant).
pub(crate) fn scene_shader_source(capable: bool) -> String {
    let base = include_str!("../shaders/scene_dda.wgsl");
    if !capable {
        return base.to_string();
    }
    let snippet = include_str!("../shaders/scene_sprite_shadow.wgsl");
    const BEGIN: &str = "//XS4C_STUB_BEGIN";
    const END: &str = "//XS4C_STUB_END";
    let (Some(b), Some(e)) = (base.find(BEGIN), base.find(END)) else {
        panic!("scene_dda.wgsl: XS4C stub markers not found");
    };
    let e_end = e + END.len();
    let mut out = String::with_capacity(base.len() + snippet.len());
    out.push_str(&base[..b]);
    out.push_str(snippet);
    out.push_str(&base[e_end..]);
    out
}

#[cfg(test)]
mod tests {
    /// Statically validate every WGSL shader with naga (the same
    /// front-end + validator wgpu runs at pipeline creation), so shader
    /// edits — e.g. the GPU.10 sprite lighting bindings — are caught in
    /// CI without needing a GPU device.
    #[test]
    fn wgsl_shaders_validate() {
        let shaders: &[(&str, &str)] = &[
            (
                "sprite_model_dda.wgsl",
                include_str!("../shaders/sprite_model_dda.wgsl"),
            ),
            ("scene_dda.wgsl", include_str!("../shaders/scene_dda.wgsl")),
            ("blit.wgsl", include_str!("../shaders/blit.wgsl")),
            (
                "scene_blit.wgsl",
                include_str!("../shaders/scene_blit.wgsl"),
            ),
            (
                "scene_resolve.wgsl",
                include_str!("../shaders/scene_resolve.wgsl"),
            ),
            ("line.wgsl", include_str!("../shaders/line.wgsl")),
            ("image.wgsl", include_str!("../shaders/image.wgsl")),
        ];
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        for (name, src) in shaders {
            let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| {
                panic!("{name}: WGSL parse failed:\n{}", e.emit_to_string(src))
            });
            validator
                .validate(&module)
                .unwrap_or_else(|e| panic!("{name}: WGSL validation failed: {e:?}"));
        }
        // XS.4.2 — the raw `sprite_model_dda.wgsl` above is the unshadowed STUB
        // variant; also validate the sprite-shadow-CAPABLE spliced variant (the
        // terrain-shadow snippet injected) that capable devices build.
        let capable = super::sprite_shader_source(true);
        let module = naga::front::wgsl::parse_str(&capable).unwrap_or_else(|e| {
            panic!(
                "sprite_model_dda.wgsl (capable): parse failed:\n{}",
                e.emit_to_string(&capable)
            )
        });
        validator.validate(&module).unwrap_or_else(|e| {
            panic!("sprite_model_dda.wgsl (capable): validation failed: {e:?}")
        });
        // XS.4.3 — the capable scene variant (sprite-cast snippet spliced in).
        let scene_cap = super::scene_shader_source(true);
        let module = naga::front::wgsl::parse_str(&scene_cap).unwrap_or_else(|e| {
            panic!(
                "scene_dda.wgsl (capable): parse failed:\n{}",
                e.emit_to_string(&scene_cap)
            )
        });
        validator
            .validate(&module)
            .unwrap_or_else(|e| panic!("scene_dda.wgsl (capable): validation failed: {e:?}"));
    }
}
