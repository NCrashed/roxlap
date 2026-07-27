//! CT.6 — CPU/GPU hit-verdict parity at carve sites
//! (docs/porting/PORTING-CARVE.md).
//!
//! The CPU DDA's hit verdict (`GridView::surface_color_mip`) and the
//! GPU marcher's hit-test bitmap (`decompress_column`'s
//! `solid_occupancy`) must agree per voxel — including the CT shapes:
//! implicit interior under a rect fill, carve-exposed tops (inherited
//! colours), pockets, bottom-reaching carves with survivors, fully
//! emptied columns (the sentinel) and the legacy z=255 bedrock
//! placeholder with its zero-RGB record. Pre-CT.6 the CPU coupled the
//! verdict to the colour fetch and stepped THROUGH solid uncoloured
//! cells — this gate pins the decoupled semantics on both backends.
//!
//! Pure CPU test: no GPU device needed (`decompress_chunk` is host
//! code), so it runs everywhere.

use roxlap_core::grid_view::GridView;
use roxlap_formats::color::VoxColor;
use roxlap_formats::edit::set_rect;
use roxlap_formats::vxl::Vxl;
use roxlap_gpu::decompress_chunk;

#[test]
fn cpu_gpu_hit_verdicts_agree_at_carve_sites() {
    const TER: VoxColor = VoxColor(0x80_aa_bb_00);
    let vsid = 16u32;
    let mut vxl = Vxl::empty(vsid); // placeholder columns: z=255, RGB-0
                                    // Terrain slab with implicit (record-less) interior.
    set_rect(&mut vxl, [2, 2, 100], [13, 13, 140], Some(TER));
    // Mid-column pocket (exposes an untextured top → CT.6 inherit).
    set_rect(&mut vxl, [4, 4, 110], [6, 6, 120], None);
    // Bottom-reaching carve with a survivor (air-terminal tail).
    set_rect(&mut vxl, [8, 8, 120], [8, 8, 255], None);
    // Full-column carve (empty sentinel).
    set_rect(&mut vxl, [10, 10, 0], [10, 10, 255], None);

    let up = decompress_chunk(&vxl);
    let m0 = &up.mips[0];
    let wpc = m0.occ_words_per_col as usize;
    let view = GridView::from_single_vxl(&vxl);

    let mut mismatches = 0u32;
    for y in 0..vsid {
        for x in 0..vsid {
            let col = (x + y * vsid) as usize;
            for z in 0..256u32 {
                let w = col * wpc + (z / 32) as usize;
                let gpu_solid = m0.solid_occupancy[w] & (1 << (z % 32)) != 0;
                let cpu_hit = view.surface_color_mip(x, y, z, 0);
                if cpu_hit.is_some() != gpu_solid {
                    mismatches += 1;
                    if mismatches <= 8 {
                        eprintln!(
                            "verdict mismatch at ({x},{y},{z}): cpu={cpu_hit:?} gpu_solid={gpu_solid}"
                        );
                    }
                }
                // Textured-colour parity: where the GPU has a usable
                // (non-zero-RGB) record, the CPU must hit with the
                // same colour.
                if let Some(c) = up.voxel_at(x, y, z) {
                    if c & 0x00ff_ffff != 0 {
                        assert_eq!(
                            cpu_hit.map(|v| v.0),
                            Some(c),
                            "textured colour diverges at ({x},{y},{z})"
                        );
                    }
                }
            }
        }
    }
    assert_eq!(mismatches, 0, "CPU/GPU hit verdicts diverged");
}
