//! Per-column slab walker used at the top of opticast to locate which
//! air gap contains the camera. Port of the `gstartv` walk in
//! `voxlaptest`'s `opticast` (`voxlap5.c:2314..2325`).
//!
//! Voxlap stores each map column as a chain of slab records (see
//! `roxlap-formats::vxl`):
//!
//! ```text
//! slab header (4 bytes):
//!   byte 0  nextptr  — offset to next slab in dwords (0 = last)
//!   byte 1  z1       — top z of floor-colour list
//!   byte 2  z1c      — bottom z of floor-colour list MINUS 1
//!   byte 3  z0       — ceiling z (additional slabs); dummy in first
//! ```
//!
//! [`camera_column_air_gap`] walks the chain to find the air gap (the
//! `[z0, z1)` range with no voxels) the camera is in, or reports
//! `None` if the camera is inside solid voxel material (in which case
//! voxlap returns from opticast early — no render this frame).

/// Walk a column's slab list to find which air gap contains z = `cz`.
///
/// Returns:
/// - `Some((z0, z1, vptr_offset))` — voxlap's `gstartz0` /
///   `gstartz1` / `v - *ixy_sptr_col` triple. Camera sits in the
///   air gap with top z = `z0` and bottom z = `z1`. `vptr_offset`
///   is the byte offset within `column` to the slab whose top
///   bounds the air gap from below. `0` means the air gap is
///   above the first slab (z0 = 0); a non-zero value means the
///   camera is in an interior air gap and `vptr_offset` points to
///   the slab header below it. R4.3a-rewire-3 grouscan uses this
///   to dispatch the initial drawflor (top-of-column) vs drawceil
///   (interior) branch.
/// - `None` — camera is inside a slab (hidden interior or below the
///   final slab); opticast returns early with no render.
#[must_use]
pub fn camera_column_air_gap(column: &[u8], cz: i32) -> Option<(i32, i32, usize)> {
    if column.len() < 4 {
        return None;
    }
    let first_z1 = i32::from(column[1]);
    if cz < first_z1 {
        // Air above the first slab — gstartz0 = 0 in voxlap's else
        // branch. vptr_offset = 0 marks "column-top" for grouscan.
        return Some((0, first_z1, 0));
    }

    let mut pos = 0usize;
    loop {
        let nextptr = column[pos];
        if nextptr == 0 {
            // Reached the last slab without finding an air gap above
            // the camera; voxlap returns early (no render).
            return None;
        }
        pos = pos.checked_add(usize::from(nextptr) * 4)?;
        // Need 4 header bytes at the new position.
        if pos.checked_add(4)? > column.len() {
            return None;
        }
        let z1 = i32::from(column[pos + 1]);
        if cz < z1 {
            let z0 = i32::from(column[pos + 3]);
            if cz < z0 {
                // Hidden interior between this slab's ceiling colours
                // and floor colours — no render.
                return None;
            }
            return Some((z0, z1, pos));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Single solid slab at z = 5..15 (inclusive of bottom). Last slab,
    /// no air-gap below it. Header [0, 5, 14, 0].
    fn single_slab_5_15() -> Vec<u8> {
        // 4-byte header. Floor colours would be (z1c - z1 + 1) = 10
        // entries × 4 bytes = 40 bytes; we omit them here because the
        // walker only reads the header.
        vec![0, 5, 14, 0]
    }

    /// Two slabs with an air gap at z = 20..30.
    ///
    /// Slab 0 (first): visible block at z = 10..14, 5 voxels.
    ///   header = [nextptr=6, z1=10, z1c=14, dummy=0]
    ///   5 colour records × 4 bytes = 20 bytes
    ///   total slab size = nextptr × 4 = 24 bytes ✓
    ///
    /// Slab 1 (last): visible block at z = 30..39, 10 voxels above
    /// the air starting at z = 20.
    ///   header = [nextptr=0, z1=30, z1c=39, z0=20]
    ///   10 colour records × 4 bytes = 40 bytes
    fn two_slabs_air_at_20_30() -> Vec<u8> {
        let mut col = Vec::new();
        // Slab 0 header.
        col.extend_from_slice(&[6, 10, 14, 0]);
        // 5 colour records (content irrelevant — the walker doesn't read).
        col.resize(col.len() + 5 * 4, 0xab);
        // Slab 1 header.
        col.extend_from_slice(&[0, 30, 39, 20]);
        // 10 colour records.
        col.resize(col.len() + 10 * 4, 0xcd);
        col
    }

    #[test]
    fn camera_above_first_slab_returns_zero_to_z1() {
        let col = single_slab_5_15();
        assert_eq!(camera_column_air_gap(&col, 0), Some((0, 5, 0)));
        assert_eq!(camera_column_air_gap(&col, 4), Some((0, 5, 0)));
    }

    #[test]
    fn camera_inside_solid_returns_none() {
        let col = single_slab_5_15();
        // cz = 10 is inside the only slab. We walk forward, hit
        // nextptr = 0, return None.
        assert_eq!(camera_column_air_gap(&col, 10), None);
        // cz = 100 (way past the column) does the same thing.
        assert_eq!(camera_column_air_gap(&col, 100), None);
    }

    #[test]
    fn camera_above_first_slab_in_two_slab_column() {
        let col = two_slabs_air_at_20_30();
        // cz = 5 is above slab 0's z1 = 10.
        // Column-top — vptr_offset = 0.
        assert_eq!(camera_column_air_gap(&col, 5), Some((0, 10, 0)));
    }

    #[test]
    fn camera_in_air_gap_between_slabs() {
        let col = two_slabs_air_at_20_30();
        // cz = 25 is in the [20, 30) gap above slab 1.
        // Interior gap — vptr_offset points to slab 1's header.
        // Slab 0's nextptr = 6 → 24-byte stride (6 × 4); slab 1
        // starts at column[24].
        assert_eq!(camera_column_air_gap(&col, 25), Some((20, 30, 24)));
        assert_eq!(camera_column_air_gap(&col, 20), Some((20, 30, 24)));
        assert_eq!(camera_column_air_gap(&col, 29), Some((20, 30, 24)));
    }

    #[test]
    fn camera_in_first_slabs_hidden_interior_returns_none() {
        let col = two_slabs_air_at_20_30();
        // cz = 12 is inside slab 0 (z1 = 10..14). The walker advances
        // to slab 1, which has z1 = 30 (cz < z1) and z0 = 20 (cz < z0)
        // → hidden interior → None.
        assert_eq!(camera_column_air_gap(&col, 12), None);
    }

    #[test]
    fn camera_in_last_slab_returns_none() {
        let col = two_slabs_air_at_20_30();
        // cz = 35 is inside slab 1 (z1 = 30..39). The walker advances
        // to slab 1, sees cz >= z1 = 30, hits nextptr = 0 → None.
        assert_eq!(camera_column_air_gap(&col, 35), None);
        // cz = 1000 — same path.
        assert_eq!(camera_column_air_gap(&col, 1000), None);
    }

    #[test]
    fn malformed_too_short_returns_none() {
        // Less than a single header.
        assert_eq!(camera_column_air_gap(&[1, 2, 3], 10), None);
    }

    #[test]
    fn malformed_nextptr_overruns_returns_none() {
        // Single slab with nextptr = 99 but no following slab.
        let col = vec![99, 10, 14, 0];
        // cz = 100 forces walk-forward; nextptr advances past EOF.
        assert_eq!(camera_column_air_gap(&col, 100), None);
    }
}
