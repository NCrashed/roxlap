//! Companion example for the book's "The asset pipeline" chapter
//! (`docs/book/src/assets.md`) — the chapter pulls its snippets from
//! here via `// ANCHOR:` markers, so everything it shows compiles and
//! its assertions actually ran. Headless: no window, no renderer.
//!
//! ```sh
//! cargo run -p roxlap-formats --example book_assets
//! ```
//!
//! Keep the anchors when editing; `docs/book/check-anchors.sh` (run by
//! the CI `book` job) goes red if one disappears.

use roxlap_formats::voxel_clip::{LoopMode, VoxelClip};
use roxlap_formats::vxl::Vxl;
use roxlap_formats::{kv6, vox};

/// A minimal in-memory `.vox` file (what MagicaVoxel writes): the
/// `"VOX "` header, a `MAIN` container, one `SIZE` + `XYZI` model pair
/// — 3×3×3 with two voxels — and no `RGBA` chunk, so parsing falls
/// back to the official default palette.
fn synthesize_vox() -> Vec<u8> {
    let mut v = Vec::new();
    let chunk = |v: &mut Vec<u8>, id: &[u8; 4], content: &[u8], children: u32| {
        v.extend_from_slice(id);
        v.extend_from_slice(&u32::try_from(content.len()).expect("small").to_le_bytes());
        v.extend_from_slice(&children.to_le_bytes());
        v.extend_from_slice(content);
    };
    v.extend_from_slice(b"VOX ");
    v.extend_from_slice(&150u32.to_le_bytes()); // format version
    let size = [3u32, 3, 3].map(u32::to_le_bytes).concat();
    let mut xyzi = 2u32.to_le_bytes().to_vec(); // voxel count…
    xyzi.extend_from_slice(&[0, 0, 0, 1]); // …(x, y, z, palette index)
    xyzi.extend_from_slice(&[2, 2, 2, 7]);
    let children = u32::try_from(12 + size.len() + 12 + xyzi.len()).expect("small");
    chunk(&mut v, b"MAIN", &[], children);
    chunk(&mut v, b"SIZE", &size, 0);
    chunk(&mut v, b"XYZI", &xyzi, 0);
    v
}

fn main() {
    // ANCHOR: vox
    // MagicaVoxel import: parse the .vox bytes, convert each model to
    // a Kv6 ready for `add_sprite_model`. MagicaVoxel is z-UP; the
    // conversion flips to roxlap's z-down, so a model that is right-
    // side-up in the editor is right-side-up in the engine.
    let bytes = synthesize_vox();
    let file = vox::parse(&bytes).expect("valid .vox");
    assert_eq!(file.models.len(), 1);
    assert_eq!(file.models[0].voxels.len(), 2);
    let models = file.to_kv6_models();
    assert_eq!((models[0].xsiz, models[0].ysiz, models[0].zsiz), (3, 3, 3));
    // ANCHOR_END: vox

    // ANCHOR: kv6_roundtrip
    // Every reader has a symmetric writer, and the round trip is
    // byte-stable — serialize(parse(bytes)) == bytes — so tools can
    // rewrite assets without churn.
    let gem = kv6::serialize(&models[0]);
    let reparsed = kv6::parse(&gem).expect("self-authored kv6");
    assert_eq!(kv6::serialize(&reparsed), gem);
    // ANCHOR_END: kv6_roundtrip

    // ANCHOR: rvc_roundtrip
    // Animated clips: kv6 frames → .rvc bytes → back. All frames must
    // share one bounding box (clips are fixed-bbox); the decode yields
    // the flipbook the renderer registers (`add_voxel_clip`).
    let frames: Vec<_> = [1u32, 2, 3, 2] // a pulsing cube, Chebyshev radius
        .iter()
        .map(|&r| {
            kv6::Kv6::from_fn(7, 7, 7, |x, y, z| {
                let d = |v: u32| (i64::from(v) - 3).unsigned_abs();
                (d(x).max(d(y)).max(d(z)) <= u64::from(r)).then_some(0x80_d0_a0_50)
            })
        })
        .collect();
    let clip = VoxelClip::from_kv6_frames(&frames, 1.0, LoopMode::Loop, &[], 150, 1)
        .expect("frames share dims");
    let rvc = clip.serialize(); // the on-disk .rvc
    let reparsed = VoxelClip::parse(&rvc).expect("self-authored .rvc");
    assert_eq!(reparsed.decode().expect("decodes").frames.len(), 4);
    // ANCHOR_END: rvc_roundtrip

    // ANCHOR: vxl_roundtrip
    // Worlds: build a Vxl from a dense predicate (the one-call "model
    // → slab format" path), then round-trip the .vxl wire bytes. This
    // is also the per-chunk encoding scene snapshots use internally.
    let world = Vxl::from_dense(64, |x, y, z| {
        (z >= 200 && x + y < 100).then_some(0x80_4d_8a_3a)
    });
    let bytes = roxlap_formats::vxl::serialize(&world);
    let reparsed = roxlap_formats::vxl::parse(&bytes).expect("self-authored .vxl");
    assert_eq!(reparsed.voxel_color(10, 10, 200), Some(0x80_4d_8a_3a));
    // Above the terrain is air.
    assert_eq!(reparsed.voxel_color(10, 10, 100), None);
    // ANCHOR_END: vxl_roundtrip

    println!("book_assets: all asset round-trips hold");
}
