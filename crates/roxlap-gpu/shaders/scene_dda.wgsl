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

struct PerGridCamera {
    pos: vec3<f32>,
    _pad0: f32,
    right: vec3<f32>,
    _pad1: f32,
    down: vec3<f32>,
    _pad2: f32,
    forward: vec3<f32>,
    // CA.0 — the grid's cutaway clip riding the old `_pad3` lane:
    // grid-local absolute voxel z, a REAL i32 lane (an f32 bit-carrier
    // would be a subnormal for any clip in 1..=0x7fffff, which the
    // WGSL spec permits implementations to flush to zero on load).
    // i32::MIN = disabled. Hide cells with `z < z_clip`.
    z_clip: i32,
    // DL — unit direction TO the sun in this grid's local frame (xyz; w
    // unused). Packed here instead of a separate per-grid storage buffer
    // (the 16 storage-buffer limit is already saturated). Zero ⇒ no sun
    // (the uniform's `sun_flags` gates whether it's used).
    sun_dir: vec4<f32>,
    // XS.3 — this grid's world transform, for cross-grid shadows: a shadow
    // ray (grid-local in the grid being shaded) is lifted to world space and
    // tested against every grid. `world_origin.xyz` is the grid origin;
    // `rot0/1/2.xyz` are the local→world rotation columns (world images of
    // grid-local axes x/y/z). Packed here for the same buffer-limit reason.
    // OC.0 — `rot0.w` carries the view cutout's focus-plane z in this
    // grid's local frame (an exact integer-valued f32, NOT a
    // bit-carrier; grid-local absolute voxel z, z-down; `i32::MIN` =
    // no cutout — the hit branch's first-gate sentinel).
    world_origin: vec4<f32>,
    rot0: vec4<f32>,
    rot1: vec4<f32>,
    rot2: vec4<f32>,
    // OC — the view cutout's focus in THIS grid's frame (xyz,
    // march/world units; w spare), converted host-side in f64 by the
    // same shared helper the CPU path uses — the kernel does no
    // world→grid conversion of its own (review finding: the shader
    // copy of the formula was a per-ray cost AND a second language to
    // keep in lockstep). Zero while the cutout is off.
    cutout_focus_local: vec4<f32>,
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
    // `@size(16)` is REQUIRED, not cosmetic: a bare `vec3<i32>` is 12
    // bytes and WGSL places the next member at its own alignment (4
    // for the u32 array below) — offset 140, while the host's
    // explicit `[i32;3] + pad` pairs put it at 144: every tail field
    // after the first bare vec3 read 4 bytes early (the GPU.13.1
    // pyramid silently read `mip_off[3]` as its level count — 0 in
    // small pools, i.e. the pyramid never fired). 112→144 padded.
    @size(16) aabb_min: vec3<i32>,
    @size(16) aabb_max: vec3<i32>,
    // GPU.13.1 — chunk-occupancy pyramid: word offsets of levels 1..=4
    // in `all_chunk_occupancy` (entry l-1 = level l) + the level count
    // above L0. Mirrors the host's [u32;4] + u32 + [u32;3] pad
    // (144→176).
    chunk_occ_mip_off: array<u32, 4>,
    chunk_occ_levels: u32,
    // CA perf — grid-local SOLID voxel z-extent (inclusive): clamps
    // the marchers' entry fast-forward box + ray caps to the real
    // content slice of the chunk stack. Inverted sentinel when empty.
    vox_z_lo: i32,
    vox_z_hi: i32,
    _pad4: u32,
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
    // TV.6 — 1 if any mapped terrain material is translucent OR
    // emissive (EV.2) — gates the material lookup + accumulate path;
    // 0 ⇒ unchanged first-hit opaque march.
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
    // DL.6 — stylized lighting: cel banding + gradient-map ramp.
    shadow_tint: vec4<f32>, // rgb = cool unlit end of the sun ramp
    style_bands: u32,       // 0 = smooth; ≥1 = quantize to bands+1 levels
    // XS.4.3 — visible sprite-instance count for the sprite-cast shadow march.
    sprite_cast_count: u32,
    // Two scalar pads (NOT vec2<u32> — keep the Rust `[u32; 2]` layout).
    _pad7b: u32,
    _pad7c: u32,
    // OC.0 — view cutout ("keyhole"):
    // `cutout_a = (margin, tan_outer, tan_inner, enable)` — how far
    // short of the character column the reveal stops (world units) +
    // the view-cone half-angle tangents (resolution-invariant) +
    // 1.0 while a cutout is set;
    // `cutout_b` = spare (the focus rides per grid in
    // `PerGridCamera.cutout_focus_local`, pre-converted host-side).
    cutout_a: vec4<f32>,
    cutout_b: vec4<f32>,
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
// 1=AlphaBlend, 2=Additive; alpha normalised 0..1. EV.2 — `emissive` is
// the pre-scaled over-bright factor (~1.0..2.0, see `MaterialGpu`), or
// 0.0 for a normal lit material. 16-byte stride, lockstep with lib.rs.
struct Mat { alpha: f32, mode: u32, emissive: f32, _pad: u32 };
@group(0) @binding(16) var<storage, read> materials_pal: array<Mat>;
// TV.6 — terrain colour→material map: `.x` = rgb (0xRRGGBB), `.y` =
// material id. A hit voxel's colour is matched here to find its material.
@group(0) @binding(17) var<storage, read> terrain_map: array<vec2<u32>>;
// DL — dynamic lighting. One point light in a grid's local frame (std430,
// 64 bytes — four vec4). Mirrors `GpuPointLight` in lib.rs. SL added the
// spot (cone) fields; `cos_outer == -1.0` marks an omnidirectional point.
struct PointLight {
    pos: vec3<f32>,
    radius: f32,
    color: vec3<f32>,
    intensity: f32,
    spot_dir: vec3<f32>,
    cos_outer: f32,
    cos_inner: f32,
    casts_shadow: u32,
    _p0: u32,
    _p1: u32,
};
// DL binding 18 — per-grid point lights, grid-major: grid g's lights at
// [g*point_light_count .. (g+1)*point_light_count]. DL.0 binds the buffer
// but the hit-site shading that reads it lands in DL.1+. (The per-grid sun
// direction rides in `grid_cameras[g].sun_dir`, binding 15.)
@group(0) @binding(18) var<storage, read> grid_point_lights: array<PointLight>;

// FW.3 — fog-of-war mask. A single self-describing storage buffer: a
// small header (word offsets below), then the deck-major, row-major
// mask bytes packed 4 cells / u32. Word 0 == 0 disables the whole thing
// (a 1-word dummy is bound when no fog grid is active), so a scene with
// no fog reads one word and skips — byte-identical to pre-FW. Mirrors
// `roxlap_scene::GpuFowMask`; the header layout must stay in lockstep
// with `roxlap-render`'s packer. Binding 22 — 19..21 are the
// conditional sprite-cast slots, so 22 is always free.
@group(0) @binding(22) var<storage, read> fog_mask: array<u32>;
const FOG_ENABLED: u32 = 0u;    // 0 = off
const FOG_GRID: u32 = 1u;       // scene-grid index the mask applies to
const FOG_DECKS_N: u32 = 2u;    // number of decks
const FOG_ORIGIN_X: u32 = 3u;   // grid-local cell of buffer (0,0), i32
const FOG_ORIGIN_Y: u32 = 4u;
const FOG_WIDTH: u32 = 5u;      // cells
const FOG_HEIGHT: u32 = 6u;
const FOG_MEM_DIM: u32 = 7u;    // f32 bits
const FOG_MEM_DESAT: u32 = 8u;  // f32 bits
const FOG_DECK_BASE: u32 = 9u;  // (z_top, z_bottom) i32 pairs per deck
const FOG_MAX_DECKS: u32 = 4u;
const FOG_ACTIVE_DECK: u32 = 17u; // observer's deck (voxel classified vs it)
const FOG_CELLS_BASE: u32 = 18u; // = FOG_ACTIVE_DECK + 1

// FW.3 — the per-cell verdict, mirroring `roxlap_scene::FowRender`.
// `hidden` ⇒ treat the cell as air (marcher/shadow continues);
// otherwise `dim`/`desat` style the surface and `dynamic` gates the
// light rig (0 = memory, baked only). `LIVE` default = shown, full.
struct FowV { hidden: bool, dim: f32, desat: f32, dynamic: bool };

// Look up the verdict for a hit in grid `g` at mip-cell `(cxm, cym,
// czm)` (grid-local; `czm` is the absolute mip-z). Returns LIVE when
// the fog is off or `g` is not the fog grid (caller need not pre-check).
fn fow_lookup(g: u32, cxm: i32, cym: i32, czm: i32, mip: u32) -> FowV {
    var v = FowV(false, 1.0, 0.0, true);
    if (fog_mask[FOG_ENABLED] == 0u || g != fog_mask[FOG_GRID]) {
        return v;
    }
    // mip-cell → mip-0 grid-local voxel at the coarse cell's CENTRE
    // (review #2 — low-corner sampling leaked / popped at mip ≥ 1).
    let half = (i32(1) << mip) >> 1u;
    let m0x = (cxm << mip) + half;
    let m0y = (cym << mip) + half;
    let m0z = (czm << mip) + half;
    // Visual-pass round 9 (#1): classify each voxel by its OWN deck
    // (deck_for_z), then gate by that deck's state. A deck BELOW the
    // observer occludes OPAQUE-DARK when unseen (returns an opaque black
    // hit, dim 0), so you can't see down through your floor into an
    // unexplored basement; a deck at or ABOVE the active one stays
    // transparent (v.hidden) so a deck you're under still shows through
    // (the swim). Round 6's window (claiming the whole next deck) let the
    // lower deck render live.
    let active_deck_i = i32(fog_mask[FOG_ACTIVE_DECK]);
    let dc = i32(fog_mask[FOG_DECKS_N]);
    let a_floor = bitcast<i32>(fog_mask[FOG_DECK_BASE + u32(active_deck_i) * 2u + 1u]);
    var deck: i32 = -1;
    for (var d: i32 = 0; d < dc; d = d + 1) {
        let zt = bitcast<i32>(fog_mask[FOG_DECK_BASE + u32(d) * 2u]);
        let zb = bitcast<i32>(fog_mask[FOG_DECK_BASE + u32(d) * 2u + 1u]);
        if (m0z >= zt && m0z <= zb) { deck = d; break; }
    }
    if (deck < 0) {
        // z in a gap between bands. Below the active floor = a sub-floor
        // shaft → occlude dark; above = transparent.
        if (m0z > a_floor) {
            return FowV(false, 0.0, 0.0, false);
        }
        v.hidden = true;
        return v;
    }
    let ox = bitcast<i32>(fog_mask[FOG_ORIGIN_X]);
    let oy = bitcast<i32>(fog_mask[FOG_ORIGIN_Y]);
    let w = i32(fog_mask[FOG_WIDTH]);
    let h = i32(fog_mask[FOG_HEIGHT]);
    let lx = m0x - ox;
    let ly = m0y - oy;
    if (lx < 0 || lx >= w || ly < 0 || ly >= h) { v.hidden = true; return v; }
    let idx = u32(deck) * u32(w) * u32(h) + u32(ly) * u32(w) + u32(lx);
    let word = fog_mask[FOG_CELLS_BASE + (idx >> 2u)];
    let mbyte = (word >> ((idx & 3u) * 8u)) & 0xffu;
    let state = mbyte >> 6u;
    if (state == 0u) { // Unseen
        // Below the observer's deck → occlude opaque-dark; else transparent.
        if (deck > active_deck_i) {
            return FowV(false, 0.0, 0.0, false);
        }
        v.hidden = true;
        return v;
    }
    let inten = f32(mbyte & 63u) * (1.0 / 63.0);
    let mdim = bitcast<f32>(fog_mask[FOG_MEM_DIM]);
    let mdesat = bitcast<f32>(fog_mask[FOG_MEM_DESAT]);
    v.dynamic = (state == 2u); // Visible
    // Visual-pass round 6 (#3): a Visible cell is UNSTYLED (full
    // brightness + colour) so the cone rim / peripheral ring no longer
    // read as dim/desaturated smudges. The intensity taper shapes only
    // Memory/Heard; the Visible→Memory handoff still lands continuous
    // (a freshly-demoted t≈1 cell is dim≈1 / desat≈0).
    if (v.dynamic) {
        v.dim = 1.0;
        v.desat = 0.0;
    } else {
        v.dim = mdim + (1.0 - mdim) * inten;
        v.desat = mdesat * (1.0 - inten);
    }
    return v;
}

// FW.3 — apply a verdict's dim + desaturate to a linear-RGB surface
// colour (matches the CPU `fow_style`; identity at dim=1, desat=0).
fn fow_apply_style(c: vec3<f32>, dim: f32, desat: f32) -> vec3<f32> {
    let luma = dot(c, vec3<f32>(0.299, 0.587, 0.114));
    return (c + (luma - c) * desat) * dim;
}

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

// GPU — hit-tests read the SOLID bitmap (textured surfaces + bedrock
// interior) so vertical wall/cliff faces are opaque. Within a slot each
// mip stores the textured block first (used by `voxel_packed_in` for the
// colour rank), then the same-size SOLID block
// (`vsid_mip² · occ_words_per_col`) immediately after it. PF.1: both
// marchers compute the solid word base once per (chunk, mip) and test
// bits inline with a last-word cache, instead of re-deriving the whole
// address chain from `grid_static_meta` on every DDA step.

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

// PF.3 — takes the pre-fetched packed voxel colour: the hit site fetches it
// exactly once (`voxel_packed_in` is a popcount rank scan of up to 8 occ
// words + 3 dependent loads) and shares it with the material lookup.
fn voxel_color_in(packed: u32, face_shade: f32) -> vec3<f32> {
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

// GPU.7 modular slot lookup. `pool_dims` are powers of 2 (asserted
// on the host), so `chunk_idx & (pool_dims - 1)` is the slot index
// per axis. Slot identity must be verified against
// `all_slot_chunk_idx` — multiple chunk_idx values can map to the
// same slot under the pool's collision invariant. PF.1: takes the
// caller-hoisted `pool_dims` instead of `g` — `grid_static_meta[g]`
// here made naga materialise the full 144-byte struct (both mip
// arrays included) once per outer DDA step.
fn slot_idx_of(pool_dims: vec3<u32>, chunk_idx: vec3<i32>) -> u32 {
    let mask = vec3<i32>(pool_dims) - vec3<i32>(1, 1, 1);
    let s = chunk_idx & mask;
    return u32(s.x)
        + u32(s.y) * pool_dims.x
        + u32(s.z) * pool_dims.x * pool_dims.y;
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
// PF.1: `mn`/`mx` are caller-hoisted (they're loop-invariant; reading
// them from storage per outer step was pure waste).
fn aabb_passed(mn: vec3<i32>, mx: vec3<i32>, p: vec3<i32>, step: vec3<i32>) -> bool {
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

// PF.1: `slot_base` = the caller-hoisted `slot_chunk_idx_offset / 4`
// (vec3<i32> entries use 16-byte stride, hence the /4), `chunk_occ_off`
// = the hoisted `chunk_occupancy_offset` — both loop-invariant.
fn chunk_has_content(slot_base: u32, chunk_occ_off: u32, slot_idx: u32, chunk_idx: vec3<i32>) -> bool {
    // Identity check: does this slot actually hold the chunk the
    // outer DDA is visiting? An empty slot's sentinel
    // (i32::MIN, i32::MIN, i32::MIN) fails this check.
    let stored = all_slot_chunk_idx[slot_base + slot_idx];
    if (stored.x != chunk_idx.x || stored.y != chunk_idx.y || stored.z != chunk_idx.z) {
        return false;
    }
    return (all_chunk_occupancy[chunk_occ_off + (slot_idx >> 5u)]
        & (1u << (slot_idx & 31u))) != 0u;
}

// GPU.13.1 — is the level-`lvl` chunk-occupancy pyramid cell that
// chunk `c` maps into EMPTY? The pyramid lives in slot space (the
// modular pool), so the cell's bit is the OR of every resident slot
// in the 2^lvl slot-block — a clear bit proves every chunk block
// aliasing it (including the ray's) holds no resident voxels, and
// the outer DDA may cross the whole block without touching memory
// again. A set bit may be an aliased false positive; the caller just
// descends (conservative, like `chunk_has_content`'s identity check).
fn chunk_block_empty(g: u32, pool_dims: vec3<u32>, lvl: u32, c: vec3<i32>) -> bool {
    let d = max(pool_dims >> vec3<u32>(lvl), vec3<u32>(1u));
    // Arithmetic >> floors negative chunk indices onto their block;
    // the pow2 mask is the same modular trick as `slot_idx_of`.
    let cell = (c >> vec3<u32>(lvl)) & (vec3<i32>(d) - vec3<i32>(1));
    let idx = u32(cell.x) + u32(cell.y) * d.x + u32(cell.z) * d.x * d.y;
    let off = grid_static_meta[g].chunk_occ_mip_off[lvl - 1u];
    return (all_chunk_occupancy[off + (idx >> 5u)] & (1u << (idx & 31u))) == 0u;
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
    // QE.8 follow-up: CPU sample_sky uses atan2(y, x) — the swapped
    // argument order here rotated the panorama 90° AND mirrored the
    // heading, so the two backends showed different panorama content
    // at the same camera.
    let azimuth = atan2(dir.y, dir.x) * (0.5 / pi) + 0.5;
    let elevation = clamp(acos(-dir.z) * (1.0 / pi), 0.0, 1.0);
    return textureSampleLevel(
        sky_texture,
        sky_sampler,
        vec2<f32>(elevation, azimuth),
        0.0,
    ).rgb;
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

// DL.3 — stylized hard shadow test. Is the segment from `origin` along unit
// `dir` blocked by a solid voxel within grid `g` before `max_t`? Intra-grid
// only (locked decision): outer chunk DDA skips empty chunks, inner voxel
// DDA hit-tests the solid bitmap. Bounded by `u.max_outer_steps` chunks and
// `u.shadow_max_steps` total voxel steps, so it always terminates. Returns
// `true` on the first occluder.
//
// PF.2 — the march follows the primary ray's LOD policy: each chunk is
// marched at `pick_mip(t_base + t_enter)`, where `t_base` is the shaded
// pixel's own world-t. Near shadows (inside `mip_scan_dist`) stay
// mip-0-exact; far shadows step coarser cells (2-8× fewer steps).
// `mip_scan_dist == 0` (LOD off) keeps the pre-PF.2 all-mip-0 march.
fn shadow_occluded(g: u32, origin: vec3<f32>, dir: vec3<f32>, max_t: f32, t_base: f32) -> bool {
    // PF.1 — hoist every loop-invariant meta field ONCE per march. Field
    // reads (not `let m = grid_static_meta[g]`): the whole-struct copy
    // makes naga materialise both 6-word mip arrays.
    let vsid = grid_static_meta[g].vsid;
    let mip_count = grid_static_meta[g].mip_count;
    let occ_off = grid_static_meta[g].occupancy_offset;
    let occ_words_slot = grid_static_meta[g].occ_words_per_slot;
    let pool_dims = grid_static_meta[g].pool_dims;
    let slot_base = grid_static_meta[g].slot_chunk_idx_offset / 4u;
    let chunk_occ_off = grid_static_meta[g].chunk_occupancy_offset;
    let aabb_mn = grid_static_meta[g].aabb_min;
    let aabb_mx = grid_static_meta[g].aabb_max;
    let occ_levels = grid_static_meta[g].chunk_occ_levels;
    // SC.4 — world-local march (chunk/voxel dims × vws); the world shadow ray
    // `origin`/`dir` and `max_t` are already world, so a scaled grid occludes
    // at its true world footprint and `shadow_max_dist` stays world-uniform.
    let vws = grid_cameras[g].world_origin.w;
    let chunk_dim = vec3<f32>(f32(vsid), f32(vsid), f32(CHUNK_Z)) * vws;
    // CA.3 — cutaway: hidden cells occlude nothing ("world as if
    // removed"), each grid applying its OWN clip — mirrors the CPU
    // SceneOccluder. Sentinel i32::MIN survives the `>> mip` below
    // any real cell, so no enable flag.
    let z_clip = grid_cameras[g].z_clip;

    // CA follow-up — fast-forward to the content box (see march_grid;
    // voxel-granular in Z, cutaway-clipped): a cross-grid shadow ray
    // from another grid's surface skips the approach void in one slab
    // test; a ray that never reaches this grid's box within `max_t`
    // rejects without loading anything else; and the ray CAP tightens
    // to the box exit — nothing past it can occlude, so up-rays stop
    // at the hull top instead of marching to the chunk-stack edge.
    var t_ff: f32 = 0.0;
    var t_cap = max_t;
    {
        let box_lo = vec3<f32>(
            f32(aabb_mn.x) * chunk_dim.x,
            f32(aabb_mn.y) * chunk_dim.y,
            f32(max(grid_static_meta[g].vox_z_lo, z_clip)) * vws,
        );
        let box_hi = vec3<f32>(
            f32(aabb_mx.x + 1) * chunk_dim.x,
            f32(aabb_mx.y + 1) * chunk_dim.y,
            f32(grid_static_meta[g].vox_z_hi + 1) * vws,
        );
        let t_a = (box_lo - origin) / dir;
        let t_b = (box_hi - origin) / dir;
        let t_lo = min(t_a, t_b);
        let t_hi = max(t_a, t_b);
        let t_in = max(max(t_lo.x, t_lo.y), t_lo.z);
        let t_out = min(min(t_hi.x, t_hi.y), t_hi.z);
        if (t_out < max(t_in, 0.0) || t_in > max_t) {
            return false;
        }
        if (t_out < 1e29) {
            t_cap = min(max_t, t_out + 1.0);
        }
        let ff = t_in - 1.0;
        if (ff > 1.0 && ff < 1e30) {
            t_ff = ff;
        }
    }

    var p_chunk = vec3<i32>(floor((origin + dir * t_ff) / chunk_dim));
    let step_chunk = vec3<i32>(sign(dir));
    let t_delta_chunk = abs(chunk_dim / dir);
    let next_boundary_chunk = vec3<f32>(
        select(f32(p_chunk.x), f32(p_chunk.x + 1), step_chunk.x > 0) * chunk_dim.x,
        select(f32(p_chunk.y), f32(p_chunk.y + 1), step_chunk.y > 0) * chunk_dim.y,
        select(f32(p_chunk.z), f32(p_chunk.z + 1), step_chunk.z > 0) * chunk_dim.z,
    );
    var t_max_chunk = shield_parallel((next_boundary_chunk - origin) / dir, dir);
    var t_enter: f32 = t_ff;
    var steps: u32 = 0u;

    for (var oc: u32 = 0u; oc < u.max_outer_steps; oc = oc + 1u) {
        if (t_enter > t_cap) { return false; }
        // Left the occupied chunk-AABB along the ray ⇒ nothing ahead.
        if (aabb_passed(aabb_mn, aabb_mx, p_chunk, step_chunk)) { return false; }
        let slot_id = slot_idx_of(pool_dims, p_chunk);
        let has_content = chunk_has_content(slot_base, chunk_occ_off, slot_id, p_chunk);
        if (has_content) {
            // PF.2 — mip for this chunk from the receiver's screen distance
            // plus the travel along the shadow ray (mirrors march_grid).
            let mip = pick_mip(t_base + t_enter, mip_count);
            let vsize = f32(1u << mip) * vws; // SC.4 — world-unit voxel size
            let vsid_mip_u = vsid >> mip;
            let vsid_mip = i32(vsid_mip_u);
            let cz_mip = i32(CHUNK_Z >> mip);
            // CA.3 — same floor formula as march_grid / the CPU sampler.
            let z_clip_mip = z_clip >> mip;
            // CA follow-up — empty-block skip: the SOLID bitmaps of
            // coarser mip levels double as brick maps (a clear bit
            // proves every descendant cell empty — gated by the
            // `solid_mips_are_child_supersets` test). Marching mip m,
            // test the block containing the cell, coarsest tier first
            // (32³ then 8³ at mip 0, like the CPU super/brick pair);
            // empty ⇒ jump the whole box. This is what keeps long
            // shadow rays affordable at fine mips.
            var skip_gap = 0u;
            var skip_vsid = 0u;
            var skip_wpc = 0u;
            var skip_col_base = 0u;
            var sup_gap = 0u;
            var sup_vsid = 0u;
            var sup_wpc = 0u;
            var sup_col_base = 0u;
            if (mip + 2u < mip_count) {
                let l = min(mip + 3u, mip_count - 1u);
                skip_gap = l - mip;
                skip_vsid = vsid >> l;
                skip_wpc = occ_words_per_col_for_mip(l);
                skip_col_base = occ_off
                    + slot_id * occ_words_slot
                    + grid_static_meta[g].mip_occ_rel[l]
                    + skip_vsid * skip_vsid * skip_wpc;
                let l2 = mip_count - 1u;
                if (l2 > l) {
                    sup_gap = l2 - mip;
                    sup_vsid = vsid >> l2;
                    sup_wpc = occ_words_per_col_for_mip(l2);
                    sup_col_base = occ_off
                        + slot_id * occ_words_slot
                        + grid_static_meta[g].mip_occ_rel[l2]
                        + sup_vsid * sup_vsid * sup_wpc;
                }
            }
            // PF.1 — solid word base for this (slot, mip), hoisted out of
            // the inner loop, plus a last-word cache.
            let wpc = occ_words_per_col_for_mip(mip);
            let solid_col_base = occ_off
                + slot_id * occ_words_slot
                + grid_static_meta[g].mip_occ_rel[mip]
                + vsid_mip_u * vsid_mip_u * wpc;
            var occ_idx_cached: u32 = 0xffffffffu;
            var occ_word_cached: u32 = 0u;
            let entry_world = origin + t_enter * dir;
            let chunk_origin_world = vec3<f32>(p_chunk) * chunk_dim;
            let entry_in_chunk = entry_world - chunk_origin_world;
            var p_voxel = clamp(
                vec3<i32>(floor(entry_in_chunk / vsize)),
                vec3<i32>(0),
                vec3<i32>(vsid_mip - 1, vsid_mip - 1, cz_mip - 1),
            );
            let next_voxel_world = vec3<f32>(
                select(f32(p_voxel.x), f32(p_voxel.x + 1), step_chunk.x > 0) * vsize
                    + chunk_origin_world.x,
                select(f32(p_voxel.y), f32(p_voxel.y + 1), step_chunk.y > 0) * vsize
                    + chunk_origin_world.y,
                select(f32(p_voxel.z), f32(p_voxel.z + 1), step_chunk.z > 0) * vsize
                    + chunk_origin_world.z,
            );
            var t_max_voxel = shield_parallel((next_voxel_world - origin) / dir, dir);
            let t_delta_voxel = abs(vsize / dir);
            // SC — coarse-mip self-shadow acne: at mip>0 the ray's ORIGIN cell
            // is a big coarse block that also contains the shading surface, so
            // the ray would immediately self-hit it (the thin "shell" on
            // distant / scaled terrain). Skip that first cell — but only when
            // the ray originates in this chunk (`t_enter == 0`), so a real
            // cross-grid occluder (entered at `t_enter > 0`) is never skipped,
            // there's no overshoot, and mip 0 stays byte-identical.
            var skip_origin_cell = (t_enter == 0.0) && (mip > 0u);
            loop {
                // CA follow-up — empty-block skip (see the hoist above):
                // if the containing coarse block is solid-free, jump to
                // its exit, coarsest tier first. Axis advances use exact
                // crossing COUNTS from t-differences (mirrors the CPU
                // landing).
                if (skip_gap > 0u) {
                    var gap = 0u;
                    if (sup_gap > 0u) {
                        let bs = vec3<u32>(p_voxel) >> vec3<u32>(sup_gap);
                        let bws = sup_col_base + (bs.x + bs.y * sup_vsid) * sup_wpc + (bs.z >> 5u);
                        if ((occ_word(bws) & (1u << (bs.z & 31u))) == 0u) {
                            gap = sup_gap;
                        }
                    }
                    if (gap == 0u) {
                        let b = vec3<u32>(p_voxel) >> vec3<u32>(skip_gap);
                        let bw = skip_col_base + (b.x + b.y * skip_vsid) * skip_wpc + (b.z >> 5u);
                        if ((occ_word(bw) & (1u << (b.z & 31u))) == 0u) {
                            gap = skip_gap;
                        }
                    }
                    if (gap > 0u) {
                        // The block was empty, so the (unsampled) origin
                        // cell was air — nothing left to suppress.
                        skip_origin_cell = false;
                        let bmask = vec3<i32>(i32((1u << gap) - 1u));
                        // Cells to the block edge along the travel
                        // direction, then the per-axis block-exit t.
                        let to_edge = select(
                            p_voxel & bmask,
                            bmask - (p_voxel & bmask),
                            step_chunk > vec3<i32>(0),
                        );
                        let t_exit_raw = t_max_voxel + vec3<f32>(to_edge) * t_delta_voxel;
                        let t_exit = select(t_exit_raw, vec3<f32>(1e30), step_chunk == vec3<i32>(0));
                        let t_box = min(t_exit.x, min(t_exit.y, t_exit.z));
                        if (t_box > t_cap) { return false; }
                        // Crossings of each axis at t ≤ t_box (`max(…, 0)`
                        // keeps the parallel-axis lanes NaN-free).
                        let num = max(vec3<f32>(t_box) - t_max_voxel, vec3<f32>(0.0));
                        let crossed = t_max_voxel <= vec3<f32>(t_box);
                        let k = select(
                            vec3<i32>(0),
                            vec3<i32>(num / t_delta_voxel) + vec3<i32>(1),
                            crossed,
                        );
                        p_voxel = p_voxel + k * step_chunk;
                        t_max_voxel = select(
                            t_max_voxel + vec3<f32>(k) * t_delta_voxel,
                            t_max_voxel,
                            k == vec3<i32>(0),
                        );
                        steps = steps + 1u;
                        if (steps >= u.shadow_max_steps) { return false; }
                        if (p_voxel.x < 0 || p_voxel.x >= vsid_mip
                            || p_voxel.y < 0 || p_voxel.y >= vsid_mip
                            || p_voxel.z < 0 || p_voxel.z >= cz_mip) { break; }
                        continue;
                    }
                }
                // Solid-bit test with a last-word cache: consecutive
                // z-steps land in the same 32-voxel word up to 32 times.
                let z_u = u32(p_voxel.z);
                let widx = solid_col_base
                    + (u32(p_voxel.x) + u32(p_voxel.y) * vsid_mip_u) * wpc
                    + (z_u >> 5u);
                if (widx != occ_idx_cached) {
                    occ_idx_cached = widx;
                    occ_word_cached = occ_word(widx);
                }
                // CA.3 — clipped-away cells (abs mip-z above the cut,
                // i.e. < z_clip_mip) never occlude.
                // FW.3 review #1 — nor does a fog-Hidden cell: an unseen
                // wall must cast no shadow, or its silhouette leaks onto
                // visible floor (the CPU `SceneOccluder` fix, on GPU).
                if (!skip_origin_cell
                    && (p_chunk.z * cz_mip + p_voxel.z >= z_clip_mip)
                    && (occ_word_cached & (1u << (z_u & 31u))) != 0u
                    && !fow_lookup(
                        g,
                        p_chunk.x * i32(vsid_mip_u) + p_voxel.x,
                        p_chunk.y * i32(vsid_mip_u) + p_voxel.y,
                        p_chunk.z * cz_mip + p_voxel.z,
                        mip,
                    ).hidden) { return true; }
                skip_origin_cell = false;
                steps = steps + 1u;
                if (steps >= u.shadow_max_steps) { return false; }
                if (t_max_voxel.x < t_max_voxel.y && t_max_voxel.x < t_max_voxel.z) {
                    if (t_max_voxel.x > t_cap) { return false; }
                    p_voxel.x = p_voxel.x + step_chunk.x;
                    t_max_voxel.x = t_max_voxel.x + t_delta_voxel.x;
                    if (p_voxel.x < 0 || p_voxel.x >= vsid_mip) { break; }
                } else if (t_max_voxel.y < t_max_voxel.z) {
                    if (t_max_voxel.y > t_cap) { return false; }
                    p_voxel.y = p_voxel.y + step_chunk.y;
                    t_max_voxel.y = t_max_voxel.y + t_delta_voxel.y;
                    if (p_voxel.y < 0 || p_voxel.y >= vsid_mip) { break; }
                } else {
                    if (t_max_voxel.z > t_cap) { return false; }
                    p_voxel.z = p_voxel.z + step_chunk.z;
                    t_max_voxel.z = t_max_voxel.z + t_delta_voxel.z;
                    if (p_voxel.z < 0 || p_voxel.z >= cz_mip) { break; }
                }
            }
        }
        // GPU.13.1 — same read-free empty-block skip as march_grid
        // (see there): shadow rays cross the same empty chunks.
        var skip_lvl: u32 = 0u;
        if (!has_content && occ_levels > 0u) {
            loop {
                if (skip_lvl >= occ_levels) { break; }
                if (!chunk_block_empty(g, pool_dims, skip_lvl + 1u, p_chunk)) { break; }
                skip_lvl = skip_lvl + 1u;
            }
        }
        let block_id = p_chunk >> vec3<u32>(skip_lvl);
        for (var sstep: u32 = 0u; sstep < 64u; sstep = sstep + 1u) {
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
            if (skip_lvl == 0u) { break; }
            if (any((p_chunk >> vec3<u32>(skip_lvl)) != block_id)) { break; }
            if (t_enter > t_cap) { break; }
        }
    }
    return false;
}

// XS.3 — grid g local → world: `world = origin + R·local` (R columns = rot0/1/2).
fn grid_local_to_world(g: u32, p: vec3<f32>) -> vec3<f32> {
    let c = grid_cameras[g];
    return c.world_origin.xyz + c.rot0.xyz * p.x + c.rot1.xyz * p.y + c.rot2.xyz * p.z;
}
// XS.3 — grid g local → world for a direction (rotation only).
fn grid_dir_to_world(g: u32, d: vec3<f32>) -> vec3<f32> {
    let c = grid_cameras[g];
    return c.rot0.xyz * d.x + c.rot1.xyz * d.y + c.rot2.xyz * d.z;
}
// XS.3 — world → grid h local: `local = Rᵀ·(world − origin)` (Rᵀ rows = rot0/1/2).
fn world_to_grid_local(g: u32, w: vec3<f32>) -> vec3<f32> {
    let c = grid_cameras[g];
    let v = w - c.world_origin.xyz;
    return vec3<f32>(dot(c.rot0.xyz, v), dot(c.rot1.xyz, v), dot(c.rot2.xyz, v));
}
// XS.3 — world → grid h local for a direction (rotation only).
fn world_dir_to_grid_local(g: u32, d: vec3<f32>) -> vec3<f32> {
    let c = grid_cameras[g];
    return vec3<f32>(dot(c.rot0.xyz, d), dot(c.rot1.xyz, d), dot(c.rot2.xyz, d));
}

// XS.3 — cross-grid hard shadow: a world-space shadow ray tested against EVERY
// grid (transformed into each grid's local frame), so a caster in one grid
// shadows surfaces in another. Returns `true` on the first occluder. The grid
// the ray came from is included (its self-shadow is the old intra-grid test).
//
// XS.4.3 — `sprites_occlude` also tests sprite volumes so sprites CAST onto
// terrain. On sprite-shadow-capable devices the renderer splices
// scene_sprite_shadow.wgsl over the stub below (binds the sprite registry at
// 19..21 + the real march); otherwise the stub returns false (no sprite cast).
//XS4C_STUB_BEGIN
fn sprites_occlude(origin_w: vec3<f32>, dir_w: vec3<f32>, max_t: f32) -> bool {
    return false;
}
//XS4C_STUB_END
// PF.2 — `t_base` = the shaded pixel's own world-t, driving the shadow
// march's mip pick (sprite volumes have no mips; `sprites_occlude` ignores it).
fn shadow_occluded_world(origin_w: vec3<f32>, dir_w: vec3<f32>, max_t: f32, t_base: f32) -> bool {
    if (sprites_occlude(origin_w, dir_w, max_t)) {
        return true;
    }
    for (var g: u32 = 0u; g < u.grid_count; g = g + 1u) {
        let o = world_to_grid_local(g, origin_w);
        let d = world_dir_to_grid_local(g, dir_w);
        if (shadow_occluded(g, o, d, max_t, t_base)) {
            return true;
        }
    }
    return false;
}

// DL — dynamic-lighting surface shade (pre-fog), same 0..~2 scale as
// `voxel_color_in`. The baked brightness byte is the ambient/AO term; the
// sun + point lights add N·L diffuse. Two looks: **smooth** (`style_bands
// == 0`, physically-ish) and **stylized** (DL.6, `style_bands ≥ 1`): the
// sun key + each point factor quantize to bands (cel), and the banded sun
// key gradient-maps `shadow_tint` (cool) → sun colour (warm) — retro
// hue-shifted terracing instead of generic Phong. Only taken when dynamic
// lighting is active (`sun_flags` bit 2); else `voxel_color_in` verbatim.
fn shade_lit(
    g: u32,
    packed: u32, // PF.3 — pre-fetched by the hit site (single voxel fetch)
    face_shade: f32,
    hit_axis: i32,
    ray_dir: vec3<f32>,
    hit_pos: vec3<f32>,
    vox_center: vec3<f32>,
    t_hit: f32,
) -> vec3<f32> {
    let a = f32((packed >> 24u) & 0xffu);
    let albedo = vec3<f32>(
        f32((packed >> 16u) & 0xffu),
        f32((packed >> 8u) & 0xffu),
        f32(packed & 0xffu),
    ) * (1.0 / 255.0);
    // Baked brightness byte (with the per-face side-shade) = the ambient/AO
    // scalar, matching `voxel_color_in`'s brightness.
    let ao = max(0.0, a - face_shade) * (1.0 / 128.0);
    let styled = u.style_bands > 0u;
    // Surface normal (grid-local) — shared by the sun + point lights.
    let n = face_normal(hit_axis, ray_dir);
    // DL.6 — stylized lighting samples at the VOXEL CENTRE (flat per voxel:
    // the whole face gets one shade → blocky pools + blocky shadow edges,
    // the retro look), smooth samples at the per-pixel hit (gradients).
    let sample = select(hit_pos, vox_center, styled);
    // Shadow-ray origin: bias off the surface along the normal to avoid
    // self-shadow acne. Shared by every caster. XS.3 — also its world-space
    // form, so the shadow ray can be tested against every grid (cross-grid).
    // PF.2 (G6) — the world-space lift (4 vec4 reads of grid_cameras[g] +
    // 9 fma) is computed lazily, only when a caster actually fires a ray.
    // SC.4 — `shadow_bias` is in VOXELS, but the shadow marches in a
    // world-scale frame, so scale it by the grid's voxel_world_size. On a
    // big-voxel (vws > 1) grid a fixed world bias is far too small — e.g. the
    // vws=4 planet biased only 1.5/4 = 0.375 voxels → self-shadow acne. `×
    // vws` restores the intended 1.5-voxel offset without overshooting a
    // nearby occluder (scaling by the coarse-mip factor too would push the
    // origin metres off a big grid and skip past a low-hovering caster).
    // vws == 1 ⇒ × 1, byte-identical.
    let vws = grid_cameras[g].world_origin.w;
    let shadow_origin = sample + n * (u.shadow_bias * vws);
    var shadow_origin_w = vec3<f32>(0.0);
    var sow_ready = false;
    // Light remaining in shadow (the strength floor); 1.0 ⇒ unshadowed.
    let in_shadow = 1.0 - u.ambient_color.w;

    // Sun key (0..1): N·L × shadow factor.
    var sun_key = 0.0;
    if ((u.sun_flags & 1u) != 0u) {
        let l = grid_cameras[g].sun_dir.xyz; // unit, TO the sun, grid-local
        let ndl = max(0.0, dot(n, l));
        if (ndl > 0.0) {
            var sh = 1.0;
            if ((u.sun_flags & 2u) != 0u) {
                if (!sow_ready) {
                    shadow_origin_w = grid_local_to_world(g, shadow_origin);
                    sow_ready = true;
                }
                if (shadow_occluded_world(shadow_origin_w, grid_dir_to_world(g, l), u.shadow_max_dist, t_hit)) {
                    sh = in_shadow;
                }
            }
            sun_key = ndl * sh;
        }
    }

    // Base term: ambient + sun. Smooth = additive; stylized = gradient map.
    var lit: vec3<f32>;
    if (styled) {
        let key = cel_band(sun_key, u.style_bands);
        let warm = u.sun_color.rgb * u.sun_color.w;
        lit = albedo * mix(u.shadow_tint.rgb, warm, key) * ao;
    } else {
        lit = albedo * u.ambient_color.rgb * ao
            + albedo * u.sun_color.rgb * u.sun_color.w * sun_key;
    }

    // Point lights: per-grid, grid-major rows at [g*count .. (g+1)*count].
    // N·L × distance falloff, hard-cut at the light's radius; shadow rays
    // march to the light (intra-grid). Stylized ⇒ the factor is celled too.
    let count = u.point_light_count;
    let base = g * count;
    for (var i: u32 = 0u; i < count; i = i + 1u) {
        // PF.3 (G5) — cheap radius reject first: read only pos+radius (16 B,
        // not the whole 64-B light) and compare SQUARED distances (no sqrt
        // for the common out-of-range light).
        let lpos = grid_point_lights[base + i].pos;
        let lrad = grid_point_lights[base + i].radius;
        let d3 = lpos - sample;
        let d2 = dot(d3, d3);
        if (d2 < lrad * lrad && d2 > 1e-8) {
            let pl = grid_point_lights[base + i];
            let dist = sqrt(d2);
            let l = d3 / dist;
            let ndl = max(0.0, dot(n, l));
            // SL — spot cone mask (1.0 for a pure point light). Computed
            // before the shadow march so a spot skips it entirely off-cone.
            let cone = spot_cone(l, pl.spot_dir, pl.cos_inner, pl.cos_outer);
            if (ndl > 0.0 && cone > 0.0) {
                let atten = point_falloff(dist, lrad);
                var sh = 1.0;
                if (pl.casts_shadow != 0u) {
                    // PF.3 — the ray origin is biased off the surface, but the
                    // parameterisation `origin + t·(to_light/dist)` still lands
                    // exactly on the light at `t == dist`, so `dist` replaces
                    // the old `length(to_light)` (one sqrt saved per caster).
                    let to_light = lpos - shadow_origin;
                    if (!sow_ready) {
                        shadow_origin_w = grid_local_to_world(g, shadow_origin);
                        sow_ready = true;
                    }
                    if (shadow_occluded_world(shadow_origin_w, grid_dir_to_world(g, to_light / dist), dist, t_hit)) {
                        sh = in_shadow;
                    }
                }
                var f = ndl * atten * cone * sh;
                if (styled) { f = cel_band(f, u.style_bands); }
                lit = lit + albedo * pl.color * pl.intensity * f;
            }
        }
    }
    return lit;
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
    // PF.1 — hoist loop-invariant meta fields once per march; a whole-struct
    // `let m = grid_static_meta[g]` would make naga materialise both 6-word
    // mip arrays (mip_occ_rel is instead read per chunk entry, where `mip`
    // is known — WGSL forbids dynamically indexing an array member of a
    // value copy anyway).
    let vsid = grid_static_meta[g].vsid;
    let mip_count = grid_static_meta[g].mip_count;
    let occ_off = grid_static_meta[g].occupancy_offset;
    let occ_words_slot = grid_static_meta[g].occ_words_per_slot;
    let pool_dims = grid_static_meta[g].pool_dims;
    let slot_base = grid_static_meta[g].slot_chunk_idx_offset / 4u;
    let chunk_occ_off = grid_static_meta[g].chunk_occupancy_offset;
    let aabb_mn = grid_static_meta[g].aabb_min;
    let aabb_mx = grid_static_meta[g].aabb_max;
    let occ_levels = grid_static_meta[g].chunk_occ_levels;
    // SC.4 — per-grid voxel_world_size (world units per voxel). Scaling the
    // chunk + voxel cell dims by it makes the WHOLE march run in world-local
    // units: `t` (unit ray_dir) stays world, so `best_t` compares across
    // grids, voxel indexing (entry_in_chunk / vsize) stays correct, and the
    // volumetric seg_len (t_span / vsize) stays a voxel count. 1.0 ⇒ identity.
    let vws = grid_cameras[g].world_origin.w;
    let chunk_dim = vec3<f32>(f32(vsid), f32(vsid), f32(CHUNK_Z)) * vws;
    // CA.3 — per-grid cutaway clip (grid-local absolute voxel z; cells
    // with `z < z_clip` read as air). The disabled sentinel (i32::MIN)
    // stays below any real cell even after the per-chunk `>> mip`, so
    // no separate enable flag is needed.
    let z_clip = grid_cameras[g].z_clip;
    // OC.2 — the keyhole cone (CPU `CpuCutout` mirror, per-CELL
    // classification): the focus-plane z rides the per-grid camera
    // (integer-valued f32 lane, `i32::MIN` sentinel = off — the
    // FIRST, one-compare gate in the hit branch); the grid-local
    // focus arrives PRE-CONVERTED in `cutout_focus_local` (host-side
    // f64, shared with the CPU path), and the cone axis is
    // eye→focus. Everything here is march-frame world units.
    let cut_focus_z = i32(grid_cameras[g].rot0.w);
    let cut_focus_local = grid_cameras[g].cutout_focus_local.xyz;
    var cut_axis = vec3<f32>(0.0);
    if (u.cutout_a.w > 0.5) {
        let a = cut_focus_local - ray_origin;
        let al = length(a);
        if (al > 1e-6) {
            cut_axis = a / al;
        }
    }

    // CA follow-up — fast-forward to the grid's chunk AABB: a distant
    // eye (the tele-iso deck camera sits ~1000+ world units out)
    // otherwise walks EVERY pixel's ray chunk-by-chunk through the
    // void just to reach the scene, per grid. One slab test either
    // rejects the ray outright (sky) or advances it to ~1 world unit
    // before the box (the backoff swallows the f32 cancellation noise
    // of a large-t position reconstruction, so the DDA still enters
    // through a true outside chunk). The empty-grid AABB sentinel
    // (min > max) rejects here too. NaN lanes (origin exactly on a
    // face of a parallel axis) fail the compares → conservative
    // fall-through to the plain from-origin march.
    var t_ff: f32 = 0.0;
    // Ray t past which nothing of this grid can be hit (the box exit,
    // +1 world unit of the same backoff slack) — checked in the outer
    // loop so up-rays leave a tall-but-thin grid at its content top
    // instead of chunk-stepping to the chunk-AABB edge.
    var t_gone: f32 = 1e30;
    {
        // Voxel-granular in Z (vox_z_lo/hi: the grid's real solid
        // slice — a chz-spanning ship is ~70 voxels of hull inside a
        // 512-voxel chunk stack), chunk-granular in XY. The cutaway
        // clip narrows the top for free (`z < z_clip` renders as air;
        // the i32::MIN sentinel never wins the max).
        let box_lo = vec3<f32>(
            f32(aabb_mn.x) * chunk_dim.x,
            f32(aabb_mn.y) * chunk_dim.y,
            f32(max(grid_static_meta[g].vox_z_lo, z_clip)) * vws,
        );
        let box_hi = vec3<f32>(
            f32(aabb_mx.x + 1) * chunk_dim.x,
            f32(aabb_mx.y + 1) * chunk_dim.y,
            f32(grid_static_meta[g].vox_z_hi + 1) * vws,
        );
        let t_a = (box_lo - ray_origin) / ray_dir;
        let t_b = (box_hi - ray_origin) / ray_dir;
        let t_lo = min(t_a, t_b);
        let t_hi = max(t_a, t_b);
        let t_in = max(max(t_lo.x, t_lo.y), t_lo.z);
        let t_out = min(min(t_hi.x, t_hi.y), t_hi.z);
        if (t_out < max(t_in, 0.0)) {
            // Nothing of this grid on the ray — pristine sky (the
            // translucent accumulator hasn't started yet).
            return finalize_sky_grid(false, vec3<f32>(0.0), 1.0, ray_dir);
        }
        if (t_out < 1e29) {
            t_gone = t_out + 1.0;
        }
        let ff = t_in - 1.0;
        if (ff > 1.0 && ff < 1e30) {
            t_ff = ff;
        }
    }

    var p_chunk = vec3<i32>(floor((ray_origin + ray_dir * t_ff) / chunk_dim));
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

    var t_enter: f32 = t_ff;
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
        if (t_enter > best_t || t_enter > t_gone) {
            return finalize_sky_grid(touched, accum, trans, ray_dir);
        }
        // GPU.13.0 — once the ray has left the occupied chunk-AABB
        // along its travel direction, no resident chunk lies ahead:
        // stop instead of stepping empty space to max_outer_steps.
        if (aabb_passed(aabb_mn, aabb_mx, p_chunk, step_chunk)) {
            return finalize_sky_grid(touched, accum, trans, ray_dir);
        }
        let slot_id = slot_idx_of(pool_dims, p_chunk);
        prev_solid = false; // fresh chunk: start a new solid run
        let has_content = chunk_has_content(slot_base, chunk_occ_off, slot_id, p_chunk);
        if (has_content) {
            // GPU.11.1 — pick the mip for this chunk by entry distance.
            // Voxels are `vsize` world units; the chunk holds
            // `vsid>>mip` × `vsid>>mip` × `CHUNK_Z>>mip` of them.
            // SC — projected-size LOD: a voxel of world size `vws` at distance
            // `t` projects like a vws=1 voxel at `t/vws`, so a fine grid
            // (vws<1) coarsens sooner and a coarse grid (vws>1) stays fine
            // longer — matched screen detail. Shadows keep the unscaled
            // distance (coarsening a fine occluder could drop its shadow).
            // vws == 1 ⇒ identity.
            let mip = pick_mip(t_enter / vws, mip_count);
            let vsize = f32(1u << mip) * vws; // SC.4 — world-unit voxel size
            let vsid_mip_u = vsid >> mip;
            let vsid_mip = i32(vsid_mip_u);
            let cz_mip = i32(CHUNK_Z >> mip);
            // CA.3 — clip plane in mip-cells. Arithmetic `>>` floors
            // toward -∞ — the IDENTICAL formula to the CPU sampler's
            // `z_clip >> mip` (the CA.3 parity gate pins the pair).
            let z_clip_mip = z_clip >> mip;
            // OC.2 — focus plane in mip-cells, the same floor formula
            // as the CPU's `focus_z >> sampler.mip` (OC.2 parity gate).
            let cut_z_mip = cut_focus_z >> mip;
            // CA follow-up — empty-block skip (see shadow_occluded):
            // the coarse SOLID mips double as brick maps, so rays
            // cross in-chunk air in 2^gap-cell jumps instead of
            // per-voxel steps. TWO tiers, like the CPU's brick/super
            // pair: the coarsest level (32³ at mip 0) is tried first,
            // then the mip+3 level (8³). Skipping solid-free boxes
            // can't change any hit (mip-superset gate) — pure speed.
            var skip_gap = 0u;
            var skip_vsid = 0u;
            var skip_wpc = 0u;
            var skip_col_base = 0u;
            var sup_gap = 0u;
            var sup_vsid = 0u;
            var sup_wpc = 0u;
            var sup_col_base = 0u;
            if (mip + 2u < mip_count) {
                let l = min(mip + 3u, mip_count - 1u);
                skip_gap = l - mip;
                skip_vsid = vsid >> l;
                skip_wpc = occ_words_per_col_for_mip(l);
                skip_col_base = occ_off
                    + slot_id * occ_words_slot
                    + grid_static_meta[g].mip_occ_rel[l]
                    + skip_vsid * skip_vsid * skip_wpc;
                let l2 = mip_count - 1u;
                if (l2 > l) {
                    sup_gap = l2 - mip;
                    sup_vsid = vsid >> l2;
                    sup_wpc = occ_words_per_col_for_mip(l2);
                    sup_col_base = occ_off
                        + slot_id * occ_words_slot
                        + grid_static_meta[g].mip_occ_rel[l2]
                        + sup_vsid * sup_vsid * sup_wpc;
                }
            }
            // PF.1 — solid-occupancy word base for this (slot, mip), hoisted
            // out of the inner loop (textured block first, solid after it),
            // plus a last-word cache: consecutive z-steps land in the same
            // 32-voxel occupancy word up to 32 times.
            let wpc = occ_words_per_col_for_mip(mip);
            let solid_col_base = occ_off
                + slot_id * occ_words_slot
                + grid_static_meta[g].mip_occ_rel[mip]
                + vsid_mip_u * vsid_mip_u * wpc;
            var occ_idx_cached: u32 = 0xffffffffu;
            var occ_word_cached: u32 = 0u;

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
                // CA follow-up — empty-block skip (mirror of the shadow
                // march): jump solid-free coarse blocks, coarsest tier
                // first. The landing cell is entered at `t_box` through
                // the exit axis, so `t_hit`/`hit_axis` update exactly
                // as a dense step would; skipped air resets the
                // translucent run.
                if (skip_gap > 0u) {
                    var gap = 0u;
                    if (sup_gap > 0u) {
                        let bs = vec3<u32>(p_voxel) >> vec3<u32>(sup_gap);
                        let bws = sup_col_base + (bs.x + bs.y * sup_vsid) * sup_wpc + (bs.z >> 5u);
                        if ((occ_word(bws) & (1u << (bs.z & 31u))) == 0u) {
                            gap = sup_gap;
                        }
                    }
                    if (gap == 0u) {
                        let b = vec3<u32>(p_voxel) >> vec3<u32>(skip_gap);
                        let bw = skip_col_base + (b.x + b.y * skip_vsid) * skip_wpc + (b.z >> 5u);
                        if ((occ_word(bw) & (1u << (b.z & 31u))) == 0u) {
                            gap = skip_gap;
                        }
                    }
                    if (gap > 0u) {
                        let bmask = vec3<i32>(i32((1u << gap) - 1u));
                        let to_edge = select(
                            p_voxel & bmask,
                            bmask - (p_voxel & bmask),
                            step_chunk > vec3<i32>(0),
                        );
                        let t_exit_raw = t_max_voxel + vec3<f32>(to_edge) * t_delta_voxel;
                        let t_exit = select(t_exit_raw, vec3<f32>(1e30), step_chunk == vec3<i32>(0));
                        let t_box = min(t_exit.x, min(t_exit.y, t_exit.z));
                        if (t_box >= best_t) {
                            return finalize_sky_grid(touched, accum, trans, ray_dir);
                        }
                        let num = max(vec3<f32>(t_box) - t_max_voxel, vec3<f32>(0.0));
                        let crossed = t_max_voxel <= vec3<f32>(t_box);
                        let k = select(
                            vec3<i32>(0),
                            vec3<i32>(num / t_delta_voxel) + vec3<i32>(1),
                            crossed,
                        );
                        p_voxel = p_voxel + k * step_chunk;
                        t_max_voxel = select(
                            t_max_voxel + vec3<f32>(k) * t_delta_voxel,
                            t_max_voxel,
                            k == vec3<i32>(0),
                        );
                        t_hit = t_box;
                        if (t_exit.x <= t_exit.y && t_exit.x <= t_exit.z) {
                            hit_axis = 0;
                        } else if (t_exit.y <= t_exit.z) {
                            hit_axis = 1;
                        } else {
                            hit_axis = 2;
                        }
                        prev_solid = false;
                        if (p_voxel.x < 0 || p_voxel.x >= vsid_mip
                            || p_voxel.y < 0 || p_voxel.y >= vsid_mip
                            || p_voxel.z < 0 || p_voxel.z >= cz_mip) { break; }
                        continue;
                    }
                }
                let z_u = u32(p_voxel.z);
                let widx = solid_col_base
                    + (u32(p_voxel.x) + u32(p_voxel.y) * vsid_mip_u) * wpc
                    + (z_u >> 5u);
                if (widx != occ_idx_cached) {
                    occ_idx_cached = widx;
                    occ_word_cached = occ_word(widx);
                }
                // CA.3 — cells above the cutaway plane (grid-local
                // absolute mip-z < z_clip_mip) read as air, resetting
                // `prev_solid` like real air so translucent runs
                // restart at the cut exactly as on the CPU.
                // OC.2 — keyhole hide rule (view cutout), decided per
                // CELL by its centre (whole cubes in or out — the CPU
                // rule verbatim): a cell above the focus plane whose
                // centre lies inside the tapered eye→focus cone
                // closer than the reveal distance reads as air too —
                // PRIMARY rays only; the shadow marches never take
                // this branch. The z compare is first and never-true
                // when the cutout is off (`i32::MIN` sentinel), so
                // the cone math stays off the disabled path.
                let cell_z_abs = p_chunk.z * cz_mip + p_voxel.z;
                let cell_occupied = (occ_word_cached & (1u << (z_u & 31u))) != 0u;
                var cut_hidden = false;
                // The occupancy bit gates the cone math (two sqrt) so
                // only actual solid cells pay it — never the far more
                // numerous marched air cells above the plane.
                if (cell_occupied && cell_z_abs < cut_z_mip) {
                    let pc = chunk_origin_world
                        + (vec3<f32>(p_voxel) + vec3<f32>(0.5)) * vsize;
                    let dv = pc - ray_origin;
                    let along = dot(dv, cut_axis);
                    if (along > 0.0) {
                        let d2 = dot(dv, dv);
                        let tan_c = sqrt(max(d2 - along * along, 0.0)) / along;
                        // Radial taper: full reveal inside the inner
                        // cone, linear to zero at the outer (hard edge
                        // when they coincide; the subnormal-free max
                        // keeps the degenerate division finite).
                        let s = clamp(
                            (u.cutout_a.y - tan_c)
                                / max(u.cutout_a.y - u.cutout_a.z, 1.17549435e-38),
                            0.0,
                            1.0,
                        );
                        // Column-hugging reveal (the CPU rule
                        // verbatim): reference = the nearest
                        // character-column point at the cell's own
                        // height — the plane z (feet, mip-0 voxels →
                        // world via vws) mirrored around the focus
                        // gives the column top (head).
                        let plane = f32(cut_focus_z) * vws;
                        let top = 2.0 * cut_focus_local.z - plane;
                        let rz = clamp(pc.z, min(top, plane), max(top, plane));
                        let dr = vec3<f32>(
                            cut_focus_local.x,
                            cut_focus_local.y,
                            rz,
                        ) - ray_origin;
                        let ref_dist = length(dr);
                        cut_hidden = sqrt(d2)
                            < max(ref_dist - u.cutout_a.x, 0.0) * s;
                    }
                }
                // FW.3 — fog-of-war verdict, priced only AFTER the cheap
                // clip/cutout gates (review perf #2): a cell the cutaway
                // or keyhole already discards never pays the deck scan +
                // mask reads. `Hide` cells read as air, so the marcher
                // continues past unseen geometry (same as a clipped cell);
                // `Show` carries dim/desaturate/dynamic into the shade.
                var fow = FowV(false, 1.0, 0.0, true);
                if (cell_occupied && cell_z_abs >= z_clip_mip && !cut_hidden) {
                    let cell_x_mip = p_chunk.x * i32(vsid_mip_u) + p_voxel.x;
                    let cell_y_mip = p_chunk.y * i32(vsid_mip_u) + p_voxel.y;
                    fow = fow_lookup(g, cell_x_mip, cell_y_mip, cell_z_abs, mip);
                }
                if (cell_z_abs >= z_clip_mip
                    && cell_occupied
                    && !cut_hidden
                    && !fow.hidden) {
                    if (t_hit >= best_t) {
                        return finalize_sky_grid(touched, accum, trans, ray_dir);
                    }
                    let shade = side_shade_for(hit_axis, ray_dir);
                    // PF.3 — ONE voxel fetch per hit (rank scan + 3 dependent
                    // loads), shared by the shade paths and the material
                    // lookup (previously fetched twice on translucent scenes).
                    let packed = voxel_packed_in(g, slot_id, mip, p_voxel);
                    // EV.2 — material lookup FIRST (mirrors the CPU hit
                    // order in dda.rs): an emissive material bypasses both
                    // shade paths below. With the gate off the lookup is
                    // skipped and everything is bit-identical to pre-EV.
                    var mat_id = 0u;
                    var mm = Mat(1.0, 0u, 0.0, 0u);
                    if (u.terrain_has_translucent != 0u) {
                        mat_id = terrain_material_id(packed);
                        mm = materials_pal[mat_id];
                    }
                    // DL — lit path (ambient + sun + point lights) when
                    // dynamic lighting is active (sun_flags bit 2); else the
                    // baked-only path, byte-identical to pre-DL. The hit
                    // position (grid-local) feeds point-light distance/dir.
                    // EV.2 — an emissive material outranks both: over-bright
                    // albedo, no face shade, no baked byte, no rig.
                    var base_color: vec3<f32>;
                    if (mm.emissive > 0.0) {
                        // FW.3 review #6 — emissive is intrinsic and wins
                        // even in memory (a remembered glowing crystal
                        // stays lit, dimmed below), never dark rock.
                        let albedo = vec3<f32>(
                            f32((packed >> 16u) & 0xffu),
                            f32((packed >> 8u) & 0xffu),
                            f32(packed & 0xffu),
                        ) * (1.0 / 255.0);
                        base_color = min(albedo * mm.emissive, vec3<f32>(1.0));
                    } else if ((u.sun_flags & 4u) != 0u && fow.dynamic) {
                        // FW.3 — memory (`dynamic == false`) suppresses the
                        // dynamic rig: a live light never relights a
                        // remembered room.
                        let hit_pos = ray_origin + t_hit * ray_dir;
                        // Voxel centre (grid-local) for flat per-voxel stylized
                        // lighting; ignored by the smooth path.
                        let vox_center = chunk_origin_world
                            + (vec3<f32>(p_voxel) + vec3<f32>(0.5)) * vsize;
                        base_color = shade_lit(g, packed, shade, hit_axis, ray_dir, hit_pos, vox_center, t_hit);
                    } else {
                        base_color = voxel_color_in(packed, shade);
                    }
                    // FW.3 — the memory / FOV-edge taper on the surface,
                    // before distance fog (identity for a full-visible cell).
                    base_color = fow_apply_style(base_color, fow.dim, fow.desat);
                    let lit = apply_fog(base_color, t_hit);
                    if (u.terrain_has_translucent == 0u) {
                        // Opaque fast-path: unchanged first hit.
                        out.hit = true;
                        out.t = t_hit;
                        out.color = lit;
                        return out;
                    }
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
                        // PF.3 — `ray_dir` is unit (normalized in render_scene),
                        // so the t-span IS the world path length; /vsize only.
                        let t_exit = min(t_max_voxel.x, min(t_max_voxel.y, t_max_voxel.z));
                        let seg_len = max(t_exit - t_hit, 0.0) / vsize;
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

        // GPU.13.1 — over an empty chunk, climb the occupancy pyramid
        // to the highest level whose block is provably empty. Inside
        // that block the incremental steps below run WITHOUT the
        // per-chunk slot/occupancy reads (and the whole block costs
        // ONE outer step of the `max_outer_steps` budget). The block
        // test is pure integer (`p_chunk >> lvl` leaves `block_id`),
        // so occupied chunks are still entered through the exact same
        // incremental `t_max_chunk` sums — render output is
        // byte-identical, only the control flow over empty space
        // changes.
        var skip_lvl: u32 = 0u;
        if (!has_content && occ_levels > 0u) {
            loop {
                if (skip_lvl >= occ_levels) { break; }
                if (!chunk_block_empty(g, pool_dims, skip_lvl + 1u, p_chunk)) { break; }
                skip_lvl = skip_lvl + 1u;
            }
        }
        let block_id = p_chunk >> vec3<u32>(skip_lvl);

        // Advance at least one chunk; while a skip level is active,
        // keep stepping (read-free) until the ray leaves the block.
        // Bound: a 2^4 block crosses at most 3·16 boundaries; 64 is
        // a safe anti-hang guard.
        for (var sstep: u32 = 0u; sstep < 64u; sstep = sstep + 1u) {
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
            if (skip_lvl == 0u) { break; }
            if (any((p_chunk >> vec3<u32>(skip_lvl)) != block_id)) { break; }
            if (t_enter > best_t) { break; }
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
