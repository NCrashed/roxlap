//! R6.0e: byte-equality validation of roxlap's `meltsphere` against
//! voxlap C's `meltsphere`.
//!
//! Strategy:
//! 1. Load the oracle.vxl.gz fixture (produced by voxlaptest's
//!    `tests/oracle/oracle.c` after `build_scene()`, so it carries
//!    the carved-cavity + painted-stripe + setkvx world state that
//!    feeds both meltsphere calls).
//! 2. Run roxlap's `meltsphere` at the same hit points and radii
//!    that oracle.c uses (voxlap5.c default `curpow = 2.0`).
//! 3. Serialise the resulting `Kv6` and byte-compare against the
//!    dumped reference `.kv6` files in `tests/fixtures/`.
//!
//! Reference fixtures are produced by the voxlap C oracle with
//! `ROXLAP_DUMP_SPRITES=<dir>` set; the dumped format mirrors
//! voxlap's `loadkv6` exactly (Kvxl magic + header + voxels + xlen
//! + ylen, all little-endian on x86).

use std::io::Read;

use flate2::read::GzDecoder;

use roxlap_core::meltsphere::{meltsphere, PowerTables};
use roxlap_formats::{kv6, vxl};

const ORACLE_VXL_GZ: &[u8] = include_bytes!("../../../assets/oracle.vxl.gz");
const SPRITE_MELTSPHERE_KV6: &[u8] = include_bytes!("fixtures/sprite_meltsphere.kv6");
const SPRITE_COCO_MELT_KV6: &[u8] = include_bytes!("fixtures/sprite_coco_melt.kv6");

fn load_oracle_world() -> vxl::Vxl {
    let mut decoder = GzDecoder::new(ORACLE_VXL_GZ);
    let mut bytes = Vec::with_capacity(40 * 1024 * 1024);
    decoder
        .read_to_end(&mut bytes)
        .expect("ungzip oracle.vxl.gz");
    vxl::parse(&bytes).expect("parse oracle.vxl")
}

#[test]
fn meltsphere_matches_voxlap_c_at_oracle_sphere_hit() {
    // Same call oracle.c makes after carving + painting the
    // (594..606, 594..606, 93..107) red/green/blue stripe block:
    //   meltsphere(&g_sprite, &c={600,600,100}, 8);
    let world = load_oracle_world();
    let pt = PowerTables::new();
    let out = meltsphere(
        &world.data,
        &world.column_offset,
        world.vsid,
        [600, 600, 100],
        8,
        2.0,
        &pt,
    )
    .expect("oracle hit should produce a non-empty sprite");

    let got = kv6::serialize(&out.kv6);
    assert_eq!(
        got.len(),
        SPRITE_MELTSPHERE_KV6.len(),
        "kv6 byte length differs (got {}, want {})",
        got.len(),
        SPRITE_MELTSPHERE_KV6.len()
    );
    assert_eq!(
        got, SPRITE_MELTSPHERE_KV6,
        "kv6 bytes differ from voxlap C's meltsphere dump"
    );
}

#[test]
fn meltsphere_matches_voxlap_c_at_oracle_coco_hit() {
    // oracle.c after carving (640..680, 640..680, 95..125) and
    // setkvx-stamping assets/coco.kvx at (660, 660, 110):
    //   meltsphere(&g_coco_sprite, &c={660,660,110}, 12);
    let world = load_oracle_world();
    let pt = PowerTables::new();
    let out = meltsphere(
        &world.data,
        &world.column_offset,
        world.vsid,
        [660, 660, 110],
        12,
        2.0,
        &pt,
    )
    .expect("oracle coco hit should produce a non-empty sprite");

    let got = kv6::serialize(&out.kv6);
    assert_eq!(
        got.len(),
        SPRITE_COCO_MELT_KV6.len(),
        "coco kv6 byte length differs (got {}, want {})",
        got.len(),
        SPRITE_COCO_MELT_KV6.len()
    );
    assert_eq!(
        got, SPRITE_COCO_MELT_KV6,
        "coco kv6 bytes differ from voxlap C's meltsphere dump"
    );
}
