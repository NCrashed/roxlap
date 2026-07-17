//! FW.3 — the fog-of-war `fog_mask` storage-buffer packer, owned here
//! (the crate that owns the WGSL `FOG_*` constants) so the header layout
//! has ONE source. `roxlap-render` and the headless tests both call
//! [`pack_fog_mask`] instead of hand-rolling the header — a layout
//! change lands in one place, not three.
//!
//! Buffer layout (mirrors `scene_dda.wgsl`'s `FOG_*` consts):
//! word 0 `FOG_ENABLED` (0 = off) · 1 `FOG_GRID` (per-grid slot) · 2
//! `FOG_DECKS_N` · 3–4 origin cell (i32) · 5 width · 6 height · 7–8
//! `memory_dim`/`memory_desaturate` (f32 bits) · 9.. up to
//! [`FOG_MAX_DECKS`] `(z_top, z_bottom)` i32 pairs · [`FOG_HEADER_WORDS`]..
//! deck-major, row-major mask bytes packed 4 cells / u32.

/// Header word count before the packed cells (`FOG_CELLS_BASE` in WGSL).
pub const FOG_HEADER_WORDS: usize = 17;
/// Max decks the shader's `fog_mask` deck loop scans (`FOG_MAX_DECKS`).
pub const FOG_MAX_DECKS: usize = 4;

/// FW.3 — pack a fog mask into `fog_mask` storage-buffer words.
///
/// - `grid_index` — the per-grid camera slot the shader gates on
///   (`FOG_GRID`).
/// - `origin_cell` — grid-local cell of the buffer's `(0, 0)`.
/// - `width` / `height` — buffer size in cells.
/// - `decks` — `(z_top, z_bottom)` inclusive bands. **Truncated to
///   [`FOG_MAX_DECKS`]** (with a `debug_assert` + one-time warn): the
///   shader only scans that many, so a taller deck stack would silently
///   drop its top floors — the truncation is now loud.
/// - `cells` — deck-major, row-major mask bytes (`d*w*h + y*w + x`).
///
/// Returns `FOG_HEADER_WORDS + ceil(cells.len()/4)` words.
#[must_use]
pub fn pack_fog_mask(
    grid_index: u32,
    origin_cell: [i32; 2],
    width: u32,
    height: u32,
    decks: &[[i32; 2]],
    memory_dim: f32,
    memory_desaturate: f32,
    cells: &[u8],
) -> Vec<u32> {
    if decks.len() > FOG_MAX_DECKS {
        // Loud (not a silent drop): the shader scans only FOG_MAX_DECKS,
        // so a taller stack would lose its top floors on GPU with no log.
        use std::sync::atomic::{AtomicBool, Ordering};
        static WARNED: AtomicBool = AtomicBool::new(false);
        if !WARNED.swap(true, Ordering::Relaxed) {
            log::warn!(
                "fog-of-war mask has {} decks; the GPU shader scans only {FOG_MAX_DECKS} — \
                 decks {FOG_MAX_DECKS}+ render Hidden on GPU (split the ship or raise \
                 FOG_MAX_DECKS)",
                decks.len(),
            );
        }
    }
    let deck_count = decks.len().min(FOG_MAX_DECKS);
    let cell_words = cells.len().div_ceil(4);
    let mut w = vec![0u32; FOG_HEADER_WORDS + cell_words];
    w[0] = 1; // FOG_ENABLED
    w[1] = grid_index; // FOG_GRID
    w[2] = deck_count as u32; // FOG_DECKS_N
    w[3] = origin_cell[0] as u32; // FOG_ORIGIN_X (bitcast i32)
    w[4] = origin_cell[1] as u32; // FOG_ORIGIN_Y
    w[5] = width; // FOG_WIDTH
    w[6] = height; // FOG_HEIGHT
    w[7] = memory_dim.to_bits(); // FOG_MEM_DIM
    w[8] = memory_desaturate.to_bits(); // FOG_MEM_DESAT
    for (d, band) in decks.iter().take(FOG_MAX_DECKS).enumerate() {
        w[9 + d * 2] = band[0] as u32; // z_top
        w[9 + d * 2 + 1] = band[1] as u32; // z_bottom
    }
    for (i, chunk) in cells.chunks(4).enumerate() {
        let mut word = 0u32;
        for (j, &b) in chunk.iter().enumerate() {
            word |= u32::from(b) << (j * 8);
        }
        w[FOG_HEADER_WORDS + i] = word;
    }
    w
}

/// FW.3 — the disabled fog buffer (`FOG_ENABLED == 0`): the shader reads
/// one word and skips, so a scene with no fog is byte-identical. Bound as
/// the dummy at build and written when `FrameParams::fow` clears.
#[must_use]
pub fn disabled_fog_mask() -> Vec<u32> {
    vec![0u32; 4]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_layout() {
        let w = pack_fog_mask(
            3,
            [-128, 256],
            2,
            1,
            &[[10, 20], [30, 40]],
            0.5,
            0.25,
            &[0xC1, 0x02, 0x83, 0x00],
        );
        assert_eq!(w[0], 1);
        assert_eq!(w[1], 3);
        assert_eq!(w[2], 2);
        assert_eq!(w[3] as i32, -128);
        assert_eq!(w[4], 256);
        assert_eq!(w[5], 2);
        assert_eq!(w[6], 1);
        assert_eq!(f32::from_bits(w[7]), 0.5);
        assert_eq!(f32::from_bits(w[8]), 0.25);
        assert_eq!((w[9] as i32, w[10] as i32), (10, 20));
        assert_eq!((w[11] as i32, w[12] as i32), (30, 40));
        assert_eq!(w[17], 0x0083_02C1);
        assert_eq!(w.len(), 18);
    }

    #[test]
    fn truncates_excess_decks() {
        let decks = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]; // 5 > MAX 4
        let w = pack_fog_mask(0, [0, 0], 1, 1, &decks, 1.0, 0.0, &[0]);
        assert_eq!(w[2], FOG_MAX_DECKS as u32, "deck count clamps to MAX");
    }
}
