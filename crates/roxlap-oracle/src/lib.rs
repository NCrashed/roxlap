//! Shared engine-facing helpers used by both the native oracle
//! binary (`src/main.rs`) and the wasm-bindgen-test goldens
//! suite (`tests/wasm_render.rs`).
//!
//! The bin and the wasm test render the same 12 poses through
//! the same `render_pose` body so any divergence between them
//! is a code-side bug, not a fixture mismatch. Native-only
//! concerns (PPM dump, golden-diff CLI, bench harness) live in
//! the bin and don't need to be exposed here.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::io::Read;

use flate2::read::GzDecoder;
use roxlap_core::camera_math;
use roxlap_core::opticast::opticast;
use roxlap_core::rasterizer::ScratchPool;
use roxlap_core::scalar_rasterizer::ScalarRasterizer;
use roxlap_core::sprite::{draw_sprite, DrawTarget, SpriteLighting};
use roxlap_core::{Camera, Engine, OpticastSettings};
use roxlap_formats::sprite::Sprite;
use roxlap_formats::{kv6, vxl};

pub const XRES: u32 = 640;
pub const YRES: u32 = 480;

/// Embedded gzipped oracle world. ~207 KB on disk; ~37 MB in
/// memory after gunzip + parse.
pub const ORACLE_VXL_GZ: &[u8] = include_bytes!("../../../assets/oracle.vxl.gz");

/// Voxlap-C oracle's `meltsphere`-extracted sprite at radius 8.
/// Bit-exact with `voxlaptest/build-clang/oracle-run/sprite_meltsphere.kv6`.
pub const SPRITE_MELTSPHERE_KV6: &[u8] =
    include_bytes!("../../roxlap-core/tests/fixtures/sprite_meltsphere.kv6");
/// Voxlap-C oracle's coco-melt sprite (radius 12 around a stamped
/// coco kvx in the world).
pub const SPRITE_COCO_MELT_KV6: &[u8] =
    include_bytes!("../../roxlap-core/tests/fixtures/sprite_coco_melt.kv6");

/// Which sprite an oracle pose draws after opticast (or `None` for
/// the opticast-only poses).
#[derive(Debug, Clone, Copy)]
pub enum SpriteKind {
    /// Axis-aligned `g_sprite` at world `(1050, 1050, 175)` —
    /// meltsphere'd from the world voxel grid (radius 8).
    MeltsphereAxis,
    /// Rotated `g_coco_sprite` at world `(1110, 1080, 175)` with
    /// basis = 120° rotation about z.
    CocoRot120,
}

/// `tile` field of a `Pose`. Mirror of voxlaptest's
/// `oracle.c:283-290` switch:
/// - `None`: no tile overlay.
/// - `OneX`: 1× zoom, alpha-disabled.
/// - `Half`: 0.5× zoom, alpha-disabled.
/// - `Blend`: 1.5× zoom + alpha-modulate-and-blend.
#[derive(Debug, Clone, Copy)]
pub enum TileKind {
    OneX,
    Half,
    Blend,
}

/// One render pose. Mirror of voxlaptest's `struct pose`. Sprite
/// poses get a `sprite` reference; the `lit` flag triggers a
/// one-shot lightmode-2 bake of the world's voxel intensities
/// before the first lit-flagged pose renders. The `tile` flag
/// dispatches a post-opticast `drawtile` overlay across the
/// three voxlap C `drawtile` paths.
#[derive(Debug, Clone, Copy)]
pub struct Pose {
    pub name: &'static str,
    pub px: f64,
    pub py: f64,
    pub pz: f64,
    pub yaw: f64,
    pub pitch: f64,
    pub sprite: Option<SpriteKind>,
    pub lit: bool,
    pub tile: Option<TileKind>,
}

/// The 12 oracle poses we render — 4 opticast-only + 4 sprite +
/// 1 lit + 3 tile, in the same name+order as voxlaptest's
/// `tests/oracle/oracle.c` so the diff is line-stable.
pub const POSES: &[Pose] = &[
    Pose {
        name: "north",
        px: 1024.0,
        py: 1024.0,
        pz: 128.0,
        yaw: std::f64::consts::FRAC_PI_2,
        pitch: 0.0,
        sprite: None,
        lit: false,
        tile: None,
    },
    Pose {
        name: "east",
        px: 1024.0,
        py: 1024.0,
        pz: 128.0,
        yaw: 0.0,
        pitch: 0.0,
        sprite: None,
        lit: false,
        tile: None,
    },
    Pose {
        name: "diag_down",
        px: 1000.0,
        py: 1000.0,
        pz: 110.0,
        yaw: std::f64::consts::FRAC_PI_4,
        pitch: 0.4,
        sprite: None,
        lit: false,
        tile: None,
    },
    Pose {
        name: "high_down",
        px: 1024.0,
        py: 1024.0,
        pz: 90.0,
        yaw: std::f64::consts::FRAC_PI_2,
        pitch: 0.7,
        sprite: None,
        lit: false,
        tile: None,
    },
    Pose {
        name: "sprite_front",
        px: 1020.0,
        py: 1050.0,
        pz: 175.0,
        yaw: 0.0,
        pitch: 0.0,
        sprite: Some(SpriteKind::MeltsphereAxis),
        lit: false,
        tile: None,
    },
    Pose {
        name: "sprite_above",
        px: 1050.0,
        py: 1050.0,
        pz: 150.0,
        yaw: 0.0,
        pitch: 1.3,
        sprite: Some(SpriteKind::MeltsphereAxis),
        lit: false,
        tile: None,
    },
    Pose {
        name: "sprite_iso",
        px: 1020.0,
        py: 1020.0,
        pz: 160.0,
        yaw: std::f64::consts::FRAC_PI_4,
        pitch: 0.4,
        sprite: Some(SpriteKind::MeltsphereAxis),
        lit: false,
        tile: None,
    },
    Pose {
        name: "sprite_coco",
        px: 1080.0,
        py: 1050.0,
        pz: 160.0,
        yaw: std::f64::consts::FRAC_PI_4,
        pitch: 0.4,
        sprite: Some(SpriteKind::CocoRot120),
        lit: false,
        tile: None,
    },
    Pose {
        name: "diag_down_lit",
        px: 1000.0,
        py: 1000.0,
        pz: 110.0,
        yaw: std::f64::consts::FRAC_PI_4,
        pitch: 0.4,
        sprite: None,
        lit: true,
        tile: None,
    },
    Pose {
        name: "tile_1x",
        px: 1000.0,
        py: 1000.0,
        pz: 110.0,
        yaw: std::f64::consts::FRAC_PI_4,
        pitch: 0.4,
        sprite: None,
        lit: false,
        tile: Some(TileKind::OneX),
    },
    Pose {
        name: "tile_half",
        px: 1000.0,
        py: 1000.0,
        pz: 110.0,
        yaw: std::f64::consts::FRAC_PI_4,
        pitch: 0.4,
        sprite: None,
        lit: false,
        tile: Some(TileKind::Half),
    },
    Pose {
        name: "tile_blend",
        px: 1000.0,
        py: 1000.0,
        pz: 110.0,
        yaw: std::f64::consts::FRAC_PI_4,
        pitch: 0.4,
        sprite: None,
        lit: false,
        tile: Some(TileKind::Blend),
    },
];

/// Voxlap's `BR(rgb) = 0x80000000 | rgb`.
#[allow(clippy::cast_possible_wrap)]
#[must_use]
pub const fn br(rgb: u32) -> i32 {
    (0x8000_0000_u32 | rgb) as i32
}

pub const TILE_SIZE: i32 = 16;

/// Voxlap C oracle's procedural test tile (oracle.c:83-95).
#[must_use]
pub fn build_test_tile() -> Vec<i32> {
    let n = (TILE_SIZE * TILE_SIZE) as usize;
    let mut out = vec![0i32; n];
    let half = TILE_SIZE / 2;
    for y in 0..TILE_SIZE {
        for x in 0..TILE_SIZE {
            let cx = x - half;
            let cy = y - half;
            let argb = if cx == 0 || cy == 0 {
                br(0x00ff_d050)
            } else if (((x >> 1) ^ (y >> 1)) & 1) == 0 {
                br(0x00c0_3030)
            } else {
                br(0x0030_c030)
            };
            #[allow(clippy::cast_sign_loss)]
            let idx = (y * TILE_SIZE + x) as usize;
            out[idx] = argb;
        }
    }
    out
}

/// Build a [`Sprite`] for a given pose.
///
/// # Panics
/// If the embedded `meltsphere` or `coco` KV6 fixture bytes can't
/// parse — they're built into the binary, so a panic here means
/// the build is broken, not a runtime fault.
#[must_use]
pub fn sprite_for_pose(kind: SpriteKind) -> Sprite {
    match kind {
        SpriteKind::MeltsphereAxis => {
            let kv = kv6::parse(SPRITE_MELTSPHERE_KV6).expect("parse meltsphere fixture");
            Sprite::axis_aligned(kv, [1050.0, 1050.0, 175.0])
        }
        SpriteKind::CocoRot120 => {
            // 120° rotation about z (oracle.c:248-251).
            let theta = 2.0_f32 * std::f32::consts::PI / 3.0_f32;
            let ca = theta.cos();
            let sa = theta.sin();
            let kv = kv6::parse(SPRITE_COCO_MELT_KV6).expect("parse coco fixture");
            Sprite {
                kv6: kv,
                p: [1110.0, 1080.0, 175.0],
                s: [ca, sa, 0.0],
                h: [-sa, ca, 0.0],
                f: [0.0, 0.0, 1.0],
                flags: 0,
            }
        }
    }
}

/// FNV-1a 64-bit, byte-for-byte the same as voxlaptest's
/// `tests/oracle/oracle.c::fnv1a64` (offset basis
/// `0xcbf29ce484222325`, prime `0x100000001b3`).
#[must_use]
pub fn fnv1a64(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Build a Voxlap-style RH camera basis from yaw + pitch.
/// Mirror of `oracle.c:set_camera_yaw_pitch` (minus the
/// `dorthonormalize`, which is a no-op for any yaw/pitch).
#[must_use]
pub fn camera_for_pose(pose: &Pose) -> Camera {
    let cy = pose.yaw.cos();
    let sy = pose.yaw.sin();
    let cp = pose.pitch.cos();
    let sp = pose.pitch.sin();
    Camera {
        pos: [pose.px, pose.py, pose.pz],
        right: [-sy, cy, 0.0],
        down: [-cy * sp, -sy * sp, cp],
        forward: [cy * cp, sy * cp, sp],
    }
}

/// Decompress + parse the embedded oracle world.
///
/// # Panics
/// If the embedded `oracle.vxl.gz` bytes fail to gunzip or parse.
/// The asset is built into the binary, so a panic here means the
/// build is broken.
#[must_use]
pub fn load_oracle_vxl() -> vxl::Vxl {
    let mut decoder = GzDecoder::new(ORACLE_VXL_GZ);
    let mut bytes = Vec::with_capacity(40 * 1024 * 1024);
    decoder
        .read_to_end(&mut bytes)
        .expect("gunzip oracle.vxl.gz");
    vxl::parse(&bytes).expect("parse oracle.vxl")
}

/// Render one pose and return the FNV-1a64 hash of the framebuffer
/// bytes — same semantic as voxlap's `fnv1a64(g_fb,
/// sizeof(g_fb))`. `framebuffer` and `zbuffer` are owned by the
/// caller so they can be reused across poses.
pub fn render_pose(
    engine: &Engine,
    vxl: &vxl::Vxl,
    pose: &Pose,
    framebuffer: &mut [u32],
    zbuffer: &mut [f32],
    pool: &mut ScratchPool,
) -> u64 {
    let sky = engine.sky_color();
    for px in framebuffer.iter_mut() {
        *px = sky;
    }

    let sky_col_i = i32::from_ne_bytes(engine.sky_color().to_ne_bytes());
    pool.set_skycast(sky_col_i, 0);
    let fog_col_i = i32::from_ne_bytes(engine.fog_color().to_ne_bytes());
    pool.set_fog(fog_col_i, engine.fog_max_scan_dist());

    let cam = camera_for_pose(pose);
    let settings = OpticastSettings::for_oracle_framebuffer(XRES, YRES);
    let pitch_pixels = XRES as usize;

    {
        let mut rasterizer = ScalarRasterizer::new(
            framebuffer,
            zbuffer,
            pitch_pixels,
            &vxl.data,
            &vxl.column_offset,
            &vxl.mip_base_offsets,
            vxl.vsid,
        );
        let _ = opticast(
            &mut rasterizer,
            pool,
            &cam,
            &settings,
            vxl.vsid,
            &vxl.data,
            &vxl.column_offset,
        );
    }

    if let Some(kind) = pose.sprite {
        let sprite = sprite_for_pose(kind);
        let cam_state =
            camera_math::derive(&cam, XRES, YRES, settings.hx, settings.hy, settings.hz);
        let mut target = DrawTarget::new(framebuffer, zbuffer, pitch_pixels, XRES, YRES);
        let lighting = SpriteLighting::default_oracle();
        let _ = draw_sprite(&mut target, &cam_state, &settings, &lighting, &sprite);
    }

    if let Some(kind) = pose.tile {
        let tile_pixels = build_test_tile();
        let mut target = DrawTarget::new(framebuffer, zbuffer, pitch_pixels, XRES, YRES);
        #[allow(clippy::cast_possible_wrap)]
        let (xz, yz, black, white) = match kind {
            TileKind::OneX => (1 << 16, 1 << 16, br(0), br(0)),
            TileKind::Half => (32768, 32768, br(0), br(0)),
            TileKind::Blend => (
                (3 << 16) / 2,
                (3 << 16) / 2,
                0x4010_3060_u32 as i32,
                0xc0e0_a0c0_u32 as i32,
            ),
        };
        roxlap_core::drawtile::drawtile(
            &mut target,
            &tile_pixels,
            TILE_SIZE * 4,
            TILE_SIZE,
            TILE_SIZE,
            8 << 16,
            8 << 16,
            320 << 16,
            240 << 16,
            xz,
            yz,
            black,
            white,
        );
    }

    let mut bytes = Vec::with_capacity(framebuffer.len() * 4);
    for &px in framebuffer.iter() {
        bytes.extend_from_slice(&px.to_ne_bytes());
    }
    fnv1a64(&bytes)
}

/// One-shot lightmode-2 bake matching voxlap C oracle's
/// `lighting_baked` flag — point light at `(1100, 1100, 70)`,
/// radius 600, intensity 1.0, baked over the
/// `800..1248 × 800..1248 × 0..200` world region. Mutates `vxl`
/// in place; the change is sticky across subsequent poses.
pub fn bake_oracle_lighting(engine: &mut Engine, vxl: &mut vxl::Vxl) {
    engine.set_lightmode(2);
    engine.add_light(roxlap_core::LightSrc {
        pos: [1100.0, 1100.0, 70.0],
        r2: 600.0 * 600.0,
        sc: 1.0,
    });
    roxlap_core::update_lighting(
        &mut vxl.data,
        &vxl.column_offset,
        vxl.vsid,
        800,
        800,
        0,
        1248,
        1248,
        200,
        engine.lightmode(),
        engine.lights(),
    );
}

/// Render every pose in `POSES`, returning `(name, fnv1a64)`
/// pairs. Handles the lighting bake on the first lit-flagged pose
/// (sticky thereafter, matching the C oracle). The bake mutates
/// `vxl` in place — pass an exclusive borrow.
pub fn render_all_poses(engine: &mut Engine, vxl: &mut vxl::Vxl) -> Vec<(&'static str, u64)> {
    let mut framebuffer = vec![0u32; (XRES * YRES) as usize];
    let mut zbuffer = vec![0f32; (XRES * YRES) as usize];
    let mut pool = ScratchPool::new(XRES, YRES, vxl.vsid);
    let mut rows = Vec::with_capacity(POSES.len());
    let mut lighting_baked = false;
    for pose in POSES {
        if pose.lit && !lighting_baked {
            bake_oracle_lighting(engine, vxl);
            lighting_baked = true;
        }
        let hash = render_pose(engine, vxl, pose, &mut framebuffer, &mut zbuffer, &mut pool);
        rows.push((pose.name, hash));
    }
    rows
}
