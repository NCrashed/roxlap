//! Per-voxel world queries against a `.vxl` column-slab world.
//!
//! Port of voxlap's `getcube` — the engine's
//! random-access voxel lookup. Used by sphere/region edits
//! (`meltsphere`, `setrect`, etc.) and any logic that needs to ask
//! "what's at (x, y, z)?" without going through a ray cast.
//!
//! Voxlap's slab format (mirroring `roxlap-formats::vxl`):
//!
//! ```text
//! slab record:
//!   byte 0  nextptr  — advance to next slab in dwords (== 0 last slab)
//!   byte 1  z1       — top z of floor-colour list
//!   byte 2  z1c      — bottom z of floor-colour list MINUS 1
//!   byte 3  z0       — ceiling z (additional slabs); dummy in first
//!   <floor colours>  (z1c - z1 + 1) × 4 bytes (BGRA each)
//!   <ceil colours>   n_ceil × 4 bytes — ceiling colours of the *next*
//!                    slab (i.e. visible bottom of *this* slab's
//!                    solid mass, observed from inside the air pocket
//!                    between this slab and the next).
//! ```
//!
//! The ceiling colours sit between this slab's floor colours and the
//! next slab's header — so they're addressed at *negative* offsets
//! from the next slab's header, which is what voxlap's getcube
//! exploits via `&v[(z - v[3]) * 4]` with `z < v[3]`.
//!
//! `n_ceil` is derived as `n_ceil = nextptr - 2 - (z1c - z1)`. Voxlap
//! stores the *negative* of this as `ceilnum = z1c - z1 - nextptr + 2`
//! and uses it as a signed comparison threshold.

// The slab walker is a pointer-arithmetic port; the casts mirror C's
// implicit narrowing/sign-loss (e.g. voxlap's `(uint32_t)(x|y) >= VSID`
// out-of-bounds test). Calls inside this module's loop are guarded by
// the precondition checks above.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

/// Result of a [`getcube`] query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cube {
    /// Empty space (sky, air pocket, or out-of-bounds (x, y)).
    Air,
    /// Solid material whose colour isn't stored in the slab list —
    /// either deep below the deepest slab, or in the hidden interior
    /// between a slab's floor and the next slab's ceiling colours.
    UnexposedSolid,
    /// Visible voxel; `0xAARRGGBB` in voxlap's storage convention
    /// (the slab bytes are little-endian B,G,R,intensity → reading
    /// as `u32_le` lands intensity in the high byte).
    Color(u32),
}

/// Look up the voxel at `(x, y, z)` in a column-slab world.
///
/// `slab_buf` + `column_offsets` + `vsid` describe the world the same
/// way the rasterizer takes them: `slab_buf` is the concatenated raw
/// slab bytes, `column_offsets[i]..column_offsets[i + 1]` is column
/// `i = y * vsid + x`'s slab range, and `vsid` is the square map
/// dimension. `column_offsets.len() == vsid² + 1`.
///
/// Out-of-bounds `(x, y)` returns [`Cube::Air`] (matches voxlap C's
/// `(uint32_t)(x|y) >= VSID` early return). `z` is *not* range-checked
/// — the caller is expected to clamp to `[0, MAXZDIM)` if needed.
///
/// # Panics
///
/// Panics on a malformed column whose slab walker would step past the
/// column's data range. Voxlap's loader (`roxlap-formats::vxl::parse`)
/// validates slab structure on parse, so any `Vxl` that round-trips
/// the parser is safe to query.
#[must_use]
pub fn getcube(slab_buf: &[u8], column_offsets: &[u32], vsid: u32, x: i32, y: i32, z: i32) -> Cube {
    // Voxlap's `(uint32_t)(x|y) >= VSID` test: rejects negatives and
    // anything past the map edge in one bitwise comparison.
    if x < 0 || y < 0 || (x as u32) >= vsid || (y as u32) >= vsid {
        return Cube::Air;
    }
    let col_idx = (y as u32 * vsid + x as u32) as usize;
    let col_start = column_offsets[col_idx] as usize;
    // The slab walker self-terminates on `nextptr == 0`, so we don't
    // need an explicit end bound. Using `column_offsets[idx + 1]` as
    // the end was wrong post-edit: voxalloc scatters columns across
    // vbuf, so adjacent indices in the offset table are no longer
    // adjacent in the buffer (and may even overlap or appear in
    // reverse order). The slab walker reads only what each column's
    // own nextptr chain says, so slicing to end-of-buffer is safe.
    let col = &slab_buf[col_start..];

    // Walk the slab chain. `pos` is the byte offset within `col` of
    // the current slab's header.
    let mut pos: usize = 0;
    loop {
        let nextptr = i32::from(col[pos]);
        let z1 = i32::from(col[pos + 1]);
        let z1c = i32::from(col[pos + 2]);
        // col[pos + 3] is z0 (additional slab) or dummy (first slab);
        // only consulted after the `pos += nextptr*4` advance below.

        if z <= z1c {
            if z < z1 {
                // Above the visible top of this slab — air pocket.
                return Cube::Air;
            }
            // Floor colour at byte offset (z - z1)*4 + 4 from header.
            let off = pos + (z - z1) as usize * 4 + 4;
            return Cube::Color(read_color(col, off));
        }

        // Voxlap's signed-stride trick: ceilnum is the *negative* of
        // the actual ceiling-colour count of the next slab. Always
        // ≤ 0; equal to 0 when the next slab has no ceiling colours.
        let ceilnum = z1c - z1 - nextptr + 2;

        if nextptr == 0 {
            // Last slab and z > z1c: deepest unexposed material.
            return Cube::UnexposedSolid;
        }

        pos += nextptr as usize * 4;
        let next_z0 = i32::from(col[pos + 3]);
        if z < next_z0 {
            let dz = z - next_z0; // dz < 0 here.
            if dz < ceilnum {
                // Above the ceiling colour list — hidden interior.
                return Cube::UnexposedSolid;
            }
            // Ceiling colour at *negative* byte offset dz*4 from the
            // next slab's header. dz ∈ [ceilnum, -1].
            let off = (pos as isize + (dz * 4) as isize) as usize;
            return Cube::Color(read_color(col, off));
        }
        // z >= next_z0: still in the air pocket above next_z0's air
        // gap, or below it. Fall through and re-test against the next
        // slab as the new "current" slab on the next iteration.
    }
}

#[inline]
fn read_color(col: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([col[off], col[off + 1], col[off + 2], col[off + 3]])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Single solid slab covering z = 5..=14 (10 visible floor voxels).
    /// Header [nextptr=0, z1=5, z1c=14, dummy=0]. Each colour record
    /// encodes its z value in B so we can verify the lookup picks the
    /// right offset.
    fn single_slab_5_to_14() -> Vec<u8> {
        let mut col = vec![0u8, 5, 14, 0];
        for z in 5..=14u8 {
            col.extend_from_slice(&[z, 0xaa, 0xbb, 0x80]);
        }
        col
    }

    /// Two slabs separated by a one-voxel air gap.
    ///
    /// Slab 0 = solid mass at z=10..28 (floor at z=10..14, hidden
    /// interior z=15..25, ceiling at z=26..28). Air pocket at z=29.
    /// Slab 1 = solid mass at z=30..39 (floor at z=30..39).
    ///
    /// Layout:
    ///   slab 0 header  [nextptr=9, z1=10, z1c=14, dummy=0]    4 bytes
    ///   slab 0 floor   z=10..14 (5 colours, B byte = z)      20 bytes
    ///   slab 1 ceil    z=26..28 (3 colours, B byte = z)      12 bytes
    ///   slab 1 header  [nextptr=0, z1=30, z1c=39, z0=29]      4 bytes
    ///   slab 1 floor   z=30..39 (10 colours, B byte = z)     40 bytes
    ///
    /// `nextptr_0 = (4 + 20 + 12) / 4 = 9` dwords.
    /// `ceilnum_0 = z1c - z1 - nextptr + 2 = 14 - 10 - 9 + 2 = -3`,
    /// so `n_ceil = 3` ✓.
    /// `z0_1 = 29` puts the ceiling colours at `z = z0 - 3 .. z0 - 1`
    ///        `= 26 .. 28` (matching the ceil-fill loop above).
    fn two_slabs_with_ceiling() -> Vec<u8> {
        let mut col = vec![9u8, 10, 14, 0];
        for z in 10..=14u8 {
            col.extend_from_slice(&[z, 0x10, 0x20, 0x80]);
        }
        for z in 26..=28u8 {
            col.extend_from_slice(&[z, 0x30, 0x40, 0x80]);
        }
        col.extend_from_slice(&[0u8, 30, 39, 29]);
        for z in 30..=39u8 {
            col.extend_from_slice(&[z, 0x50, 0x60, 0x80]);
        }
        col
    }

    fn world_with(col: Vec<u8>) -> (Vec<u8>, Vec<u32>) {
        // 1×1 world: column 0 = `col`, column_offset = [0, col.len()].
        let len = col.len() as u32;
        (col, vec![0, len])
    }

    #[test]
    fn out_of_bounds_xy_returns_air() {
        let (buf, off) = world_with(single_slab_5_to_14());
        assert_eq!(getcube(&buf, &off, 1, -1, 0, 7), Cube::Air);
        assert_eq!(getcube(&buf, &off, 1, 0, -1, 7), Cube::Air);
        assert_eq!(getcube(&buf, &off, 1, 1, 0, 7), Cube::Air);
        assert_eq!(getcube(&buf, &off, 1, 0, 1, 7), Cube::Air);
    }

    #[test]
    fn above_first_slab_returns_air() {
        let (buf, off) = world_with(single_slab_5_to_14());
        // z = 0..4 is above z1=5 → air.
        for z in 0..5 {
            assert_eq!(getcube(&buf, &off, 1, 0, 0, z), Cube::Air, "z={z}");
        }
    }

    #[test]
    fn floor_color_returned_for_visible_voxels() {
        let (buf, off) = world_with(single_slab_5_to_14());
        // z=5..=14 are floor colours; B = z, G = 0xaa, R = 0xbb, alpha = 0x80.
        for z in 5..=14 {
            let want = u32::from_le_bytes([z as u8, 0xaa, 0xbb, 0x80]);
            assert_eq!(getcube(&buf, &off, 1, 0, 0, z), Cube::Color(want), "z={z}");
        }
    }

    #[test]
    fn below_last_slab_returns_unexposed_solid() {
        let (buf, off) = world_with(single_slab_5_to_14());
        // z > z1c = 14 with nextptr = 0 → unexposed.
        for z in 15..40 {
            assert_eq!(
                getcube(&buf, &off, 1, 0, 0, z),
                Cube::UnexposedSolid,
                "z={z}"
            );
        }
    }

    #[test]
    fn second_slab_floor_resolved() {
        let (buf, off) = world_with(two_slabs_with_ceiling());
        // z=30..=39 are second slab's floor colours.
        for z in 30..=39 {
            let want = u32::from_le_bytes([z as u8, 0x50, 0x60, 0x80]);
            assert_eq!(getcube(&buf, &off, 1, 0, 0, z), Cube::Color(want), "z={z}");
        }
    }

    #[test]
    fn slab0_ceiling_colors_resolved() {
        let (buf, off) = world_with(two_slabs_with_ceiling());
        // Slab 0's ceiling at z=26..=28 (visible bottom of slab 0,
        // looking up from inside the air pocket below it).
        for z in 26..=28 {
            let want = u32::from_le_bytes([z as u8, 0x30, 0x40, 0x80]);
            assert_eq!(getcube(&buf, &off, 1, 0, 0, z), Cube::Color(want), "z={z}");
        }
    }

    #[test]
    fn slab0_hidden_interior_returns_unexposed() {
        let (buf, off) = world_with(two_slabs_with_ceiling());
        // z=15..=25 is slab 0's hidden interior (between floor at
        // z=14 and ceiling at z=26). Walker falls through slab 0's
        // top, advances to slab 1, sees z < z0_next=29, dz < ceilnum
        // = -3 so dz=z-29 ∈ [-14, -4] all satisfy dz < -3.
        for z in 15..=25 {
            assert_eq!(
                getcube(&buf, &off, 1, 0, 0, z),
                Cube::UnexposedSolid,
                "z={z}"
            );
        }
    }

    #[test]
    fn air_pocket_between_slabs_returns_air() {
        let (buf, off) = world_with(two_slabs_with_ceiling());
        // z=29 is the one-voxel air pocket between slab 0's bottom
        // (z=28 ceiling) and slab 1's top (z=30 floor). z0_next=29
        // so `z < z0_next` is false; loop continues with slab 1 as
        // the new "current" slab. z <= z1c_1 = 39 and z < z1_1 = 30
        // → Cube::Air.
        assert_eq!(getcube(&buf, &off, 1, 0, 0, 29), Cube::Air);
    }

    #[test]
    fn deep_below_last_slab_unexposed() {
        let (buf, off) = world_with(two_slabs_with_ceiling());
        // z >= 40 is below slab 1's floor (z1c=39) with nextptr=0.
        for z in 40..50 {
            assert_eq!(
                getcube(&buf, &off, 1, 0, 0, z),
                Cube::UnexposedSolid,
                "z={z}"
            );
        }
    }
}
