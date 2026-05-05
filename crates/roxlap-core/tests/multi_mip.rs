//! R4.5e: end-to-end multi-mip rendering smoke test.
//!
//! Exercises the [`phase_remiporend`] body landed in R4.5d by
//! rendering the oracle world after `Vxl::generate_mips(4)`, with
//! [`OpticastSettings::mip_levels`] = 4. The same scene rendered
//! with `mip_levels = 1` is used as a baseline.
//!
//! There is no voxlap C oracle for this path: voxlap's
//! `vxlmipuse == 1` everywhere in voxlaptest, so its goldens never
//! exercise the mip transition. The validation here is therefore
//! roxlap-only:
//!
//! 1. Both renders complete without panicking (state stays
//!    in-bounds across the mip transition).
//! 2. The framebuffer hashes differ — the multi-mip render
//!    actually drives `phase_remiporend`'s body, otherwise the
//!    branch would short-circuit on `gmipnum == 1` and produce
//!    the same pixels as the baseline.
//! 3. Hashes are pinned so future changes to `phase_remiporend`
//!    have to declare intent (bump the constants).
//!
//! [`phase_remiporend`]: roxlap_core::grouscan::Phase

use std::io::Read;

use flate2::read::GzDecoder;

use roxlap_core::{
    opticast, rasterizer::ScratchPool, scalar_rasterizer::ScalarRasterizer, Camera,
    OpticastOutcome, OpticastSettings,
};
use roxlap_formats::vxl;

const ORACLE_VXL_GZ: &[u8] = include_bytes!("../../../assets/oracle.vxl.gz");

const XRES: u32 = 320;
const YRES: u32 = 240;

fn load_oracle_world() -> vxl::Vxl {
    let mut decoder = GzDecoder::new(ORACLE_VXL_GZ);
    let mut bytes = Vec::with_capacity(40 * 1024 * 1024);
    decoder
        .read_to_end(&mut bytes)
        .expect("ungzip oracle.vxl.gz");
    vxl::parse(&bytes).expect("parse oracle.vxl")
}

/// FNV-1a 64-bit, byte-wise — same hash voxlap-oracle uses for its
/// `roxlap-hashes.txt`. Stable across runs and platforms; cheap to
/// compute on the entire framebuffer.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

/// Render once, returning a hash of the framebuffer bytes.
#[allow(clippy::cast_precision_loss)]
fn render_and_hash(vxl: &vxl::Vxl, mip_levels: u32) -> u64 {
    let pixel_count = (XRES as usize) * (YRES as usize);
    let mut framebuffer = vec![0u32; pixel_count];
    let mut zbuffer = vec![0.0f32; pixel_count];
    let mut pool = ScratchPool::new(XRES, YRES, vxl.vsid);

    // Place camera inside the carved cavity (oracle world's playable
    // region is x=800..1248, y=800..1248) near the northern edge,
    // looking +y down the long axis so rays travel hundreds of
    // voxels and exercise the mip transition.
    let cam = Camera {
        pos: [1024.0, 850.0, 100.0],
        right: [1.0, 0.0, 0.0],
        down: [0.0, 0.0, 1.0],
        forward: [0.0, 1.0, 0.0],
    };

    // Match for_oracle_framebuffer's hx/hy/hz convention but with
    // the multi-mip dial turned on. mip_scan_dist = 4 is voxlap's
    // default; max_scan_dist = 1024 is the oracle's.
    let half_w = (XRES as f32) * 0.5;
    let half_h = (YRES as f32) * 0.5;
    let settings = OpticastSettings {
        xres: XRES,
        yres: YRES,
        y_start: 0,
        y_end: YRES,
        hx: half_w,
        hy: half_h,
        hz: half_w,
        anginc: 1,
        mip_levels,
        mip_scan_dist: 4,
        max_scan_dist: 1024,
    };

    {
        let mut rasterizer = ScalarRasterizer::new(
            &mut framebuffer,
            &mut zbuffer,
            XRES as usize,
            &vxl.data,
            &vxl.column_offset,
            &vxl.mip_base_offsets,
            vxl.vsid,
        );
        let outcome = opticast(
            &mut rasterizer,
            &mut pool,
            &cam,
            &settings,
            vxl.vsid,
            &vxl.data,
            &vxl.column_offset,
        );
        assert_eq!(outcome, OpticastOutcome::Rendered);
    }

    let bytes = bytemuck_cast(&framebuffer);
    fnv1a64(bytes)
}

/// Treat the `[u32]` framebuffer as raw bytes for hashing. Plain
/// `align_to` since the slice is contiguous and the alignment is
/// trivially valid for a `u32` source.
fn bytemuck_cast(fb: &[u32]) -> &[u8] {
    let (head, body, tail) = unsafe { fb.align_to::<u8>() };
    assert!(head.is_empty() && tail.is_empty(), "u32 → u8 reslice clean");
    body
}

#[test]
fn multi_mip_renders_without_panic_and_diverges_from_single_mip() {
    let mut vxl = load_oracle_world();
    let baseline = render_and_hash(&vxl, 1);

    vxl.generate_mips(4);
    assert_eq!(vxl.mip_count(), 4);

    let multi = render_and_hash(&vxl, 4);

    // Sanity: both renders produced a non-empty hash.
    assert_ne!(baseline, 0);
    assert_ne!(multi, 0);
    // The mip transition fired, so the framebuffer differs from the
    // single-mip baseline. If `phase_remiporend`'s `gmipnum > 1`
    // gate failed (e.g. multi_mip plumbing regressed), this would
    // produce the same hash as the baseline.
    assert_ne!(
        baseline, multi,
        "multi-mip render should differ from single-mip baseline \
         (baseline={baseline:#x}, multi={multi:#x})"
    );
}

/// Single-mip baseline pinned at the test camera. Pinned to catch
/// regressions in the rest of the pipeline; should not change when
/// only `phase_remiporend` or `Vxl::generate_mips` are touched. No
/// voxlap C reference — this is roxlap's own golden.
const BASELINE_SINGLE_MIP: u64 = 0xc287_c058_2874_9ce7;
/// Multi-mip render after `Vxl::generate_mips(4)` + `mip_levels = 4`.
/// Pinned to detect drift in `phase_remiporend` or
/// `Vxl::generate_mips`.
const MULTI_MIP_4: u64 = 0x156e_d408_6987_e325;

#[test]
fn multi_mip_hashes_match_pinned_goldens() {
    let mut vxl = load_oracle_world();
    let baseline = render_and_hash(&vxl, 1);
    assert_eq!(
        baseline, BASELINE_SINGLE_MIP,
        "single-mip baseline drift: {baseline:#018x}"
    );

    vxl.generate_mips(4);
    let multi = render_and_hash(&vxl, 4);
    assert_eq!(multi, MULTI_MIP_4, "multi-mip golden drift: {multi:#018x}");
}
