//! FW.3 — the fog-of-war `fog_mask` storage-buffer packer, owned here
//! (the crate that owns the WGSL `FOG_*` constants) so the header layout
//! has ONE source. `roxlap-render` and the headless tests both call
//! [`pack_fog_mask`] instead of hand-rolling the header — a layout
//! change lands in one place, not three.
//!
//! Buffer layout (mirrors `scene_dda.wgsl`'s `FOG_*` consts):
//! word 0 `FOG_FLAGS` (0 = off; bit 0 on, bit 1
//! [`FOG_FLAG_UNSEEN_OCCLUDES`], bits 8-12 [`FOG_SPAN_SHIFT`]) · 1
//! `FOG_GRID` (per-grid slot) · 2
//! `FOG_DECKS_N` · 3–4 origin cell (i32) · 5 width · 6 height · 7–8
//! `memory_dim`/`memory_desaturate` (f32 bits) · 9.. up to
//! [`FOG_MAX_DECKS`] `(z_top, z_bottom)` i32 pairs · then
//! `FOG_ACTIVE_DECK` (the observer's deck — every voxel is classified
//! against it, not its own z) · [`FOG_HEADER_WORDS`].. deck-major,
//! row-major mask bytes packed 4 cells / u32.

/// Header word count before the packed cells (`FOG_CELLS_BASE` in WGSL).
pub const FOG_HEADER_WORDS: usize = 18;
/// Word 0, bit 0: fog is on at all. A zero word 0 is the disabled buffer.
pub const FOG_FLAG_ON: u32 = 1;
/// Word 0, bit 1: never-seen ground on the observer's OWN deck renders
/// opaque black rather than as air
/// ([`VisionConfig::unseen_occludes`](../../roxlap_scene/fow/struct.VisionConfig.html)).
///
/// A flag bit in word 0 rather than a new header word, because the header
/// layout is mirrored by hand in three places and widening it is the
/// change most likely to go wrong quietly.
pub const FOG_FLAG_UNSEEN_OCCLUDES: u32 = 2;
/// Word 0, bits 8–12: `log2` of the mask's cell span — how many grid
/// columns one mask cell covers, per axis
/// ([`VisionConfig::cell_span`](../../roxlap_scene/fow/struct.VisionConfig.html)).
/// The shader shifts a voxel's grid-local XY down by this before
/// indexing, so `0` is the mask-cell-per-column mask this was before a
/// span existed.
///
/// In the flags word for the reason [`FOG_FLAG_UNSEEN_OCCLUDES`] is: the
/// header layout is mirrored by hand in three places, and a span is five
/// bits, not a word.
pub const FOG_SPAN_SHIFT: u32 = 8;
/// Max decks the shader's `fog_mask` deck loop scans (`FOG_MAX_DECKS`).
pub const FOG_MAX_DECKS: usize = 4;
/// Word index of the observer's active deck (`FOG_ACTIVE_DECK` in WGSL),
/// just past the [`FOG_MAX_DECKS`] band pairs (`9 + 2*4`).
pub const FOG_ACTIVE_DECK: usize = 17;

/// FW.3 — pack a fog mask into `fog_mask` storage-buffer words.
///
/// - `grid_index` — the per-grid camera slot the shader gates on
///   (`FOG_GRID`).
/// - `origin_cell` — grid-local MASK cell of the buffer's `(0, 0)`.
/// - `width` / `height` — buffer size in mask cells.
/// - `cell_span` — grid columns one mask cell covers per axis; rounded
///   DOWN to a power of two and packed as its `log2` (`FOG_SPAN_SHIFT`),
///   since the shader shifts rather than divides. `1` is a cell per
///   column.
/// - `decks` — `(z_top, z_bottom)` inclusive bands. **Truncated to
///   [`FOG_MAX_DECKS`]** (with a `debug_assert` + one-time warn): the
///   shader only scans that many, so a taller deck stack would silently
///   drop its top floors — the truncation is now loud.
/// - `active_deck` — the observer's deck; every rendered voxel is
///   classified against this layer regardless of its own z (so stairs
///   and sub-floor seen through a hole read on the deck you stand on).
///   Clamped to the deck count.
/// - `cells` — deck-major, row-major mask bytes (`d*w*h + y*w + x`).
///
/// Returns `FOG_HEADER_WORDS + ceil(cells.len()/4)` words.
#[must_use]
pub fn pack_fog_mask(
    grid_index: u32,
    origin_cell: [i32; 2],
    width: u32,
    height: u32,
    cell_span: u32,
    decks: &[[i32; 2]],
    active_deck: usize,
    memory_dim: f32,
    memory_desaturate: f32,
    unseen_occludes: bool,
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
    w[0] = FOG_FLAG_ON
        | if unseen_occludes {
            FOG_FLAG_UNSEEN_OCCLUDES
        } else {
            0
        }
        // A span that is not a power of two is a caller bug; round it
        // down rather than shift by a fraction, which the shader cannot.
        | (cell_span.max(1).ilog2() << FOG_SPAN_SHIFT);
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
    // Clamp the active deck to a valid layer so the shader index is
    // always in-bounds even if a caller passes a stale deck.
    w[FOG_ACTIVE_DECK] = active_deck.min(deck_count.saturating_sub(1)) as u32;
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
            1,
            &[[10, 20], [30, 40]],
            1,
            0.5,
            0.25,
            false,
            &[0xC1, 0x02, 0x83, 0x00],
        );
        assert_eq!(w[0], FOG_FLAG_ON);
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
        assert_eq!(w[FOG_ACTIVE_DECK], 1, "active deck");
        assert_eq!(w[18], 0x0083_02C1);
        assert_eq!(w.len(), 19);
    }

    #[test]
    fn truncates_excess_decks() {
        let decks = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]; // 5 > MAX 4
        let w = pack_fog_mask(0, [0, 0], 1, 1, 1, &decks, 0, 1.0, 0.0, false, &[0]);
        assert_eq!(w[2], FOG_MAX_DECKS as u32, "deck count clamps to MAX");
    }

    /// The span rides word 0 above the flag bits, as its `log2` — the
    /// shader shifts by it, so a span that is not a power of two is
    /// rounded down rather than turned into a fractional shift nothing
    /// can perform. A span of one leaves word 0 exactly as it was before
    /// spans existed, which is what keeps every hull mask byte-stable.
    #[test]
    fn the_cell_span_rides_word_zero_as_a_shift() {
        let pack = |span| pack_fog_mask(0, [0, 0], 1, 1, span, &[[0, 1]], 0, 1.0, 0.0, false, &[0]);
        assert_eq!(pack(1)[0], FOG_FLAG_ON, "span 1 is shift 0");
        assert_eq!(pack(16)[0] >> FOG_SPAN_SHIFT, 4);
        assert_eq!(pack(0)[0] >> FOG_SPAN_SHIFT, 0, "no span is one column");
        assert_eq!(pack(24)[0] >> FOG_SPAN_SHIFT, 4, "rounded down to 16");
        // …and the flags underneath are untouched by any of it.
        assert_eq!(pack(16)[0] & 0xff, FOG_FLAG_ON);
    }

    /// The occlude-unexplored choice rides word 0 beside the on bit, so a
    /// shader that reads word 0 as a plain boolean would silently treat a
    /// shrouded map as fog-off. Both bits, checked together.
    #[test]
    fn unseen_occludes_is_a_flag_beside_the_on_bit() {
        let plain = pack_fog_mask(0, [0, 0], 1, 1, 1, &[[0, 1]], 0, 1.0, 0.0, false, &[0]);
        assert_eq!(plain[0], FOG_FLAG_ON);
        assert_eq!(plain[0] & FOG_FLAG_UNSEEN_OCCLUDES, 0);

        let shrouded = pack_fog_mask(0, [0, 0], 1, 1, 1, &[[0, 1]], 0, 1.0, 0.0, true, &[0]);
        assert_eq!(shrouded[0] & FOG_FLAG_ON, FOG_FLAG_ON, "still on");
        assert_eq!(shrouded[0] & FOG_FLAG_UNSEEN_OCCLUDES, FOG_FLAG_UNSEEN_OCCLUDES);
        assert_ne!(shrouded[0], 0, "a zero word 0 is the disabled buffer");
    }

    #[test]
    fn active_deck_clamps_to_deck_count() {
        // A stale/over-range active deck must not index past the layers.
        let w = pack_fog_mask(0, [0, 0], 1, 1, 1, &[[0, 1], [2, 3]], 9, 1.0, 0.0, false, &[0]);
        assert_eq!(w[FOG_ACTIVE_DECK], 1, "clamped to last valid deck");
    }
}
