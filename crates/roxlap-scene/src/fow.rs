//! FW.0 — fog-of-war mask core (`PORTING-FOW.md`): SS13-style
//! *knowledge* visibility, computed host-side, applied by renderers
//! in later substages.
//!
//! A [`FogOfWar`] tracks, per deck and per mip-0 XY column ("cell"),
//! what the observed character *knows*: never seen ([`CellState::Unseen`]),
//! seen before ([`CellState::Memory`]), currently in the facing cone /
//! peripheral radius ([`CellState::Visible`]), or revealed by a sound
//! source ([`CellState::Heard`]). Each cell carries a 6-bit intensity
//! that fades between states (the boundary is a smooth per-cell
//! taper — **no dither**, the OC round-1 lesson) and decays while in
//! Memory.
//!
//! Conventions (chapter 2: **+z is DOWN**):
//! - Cells are grid-local mip-0 XY columns; a [`DeckBand`] limits each
//!   deck's z extent (`z_top` is the ceiling — the *smaller* z).
//! - Line of sight is a 2D recursive shadowcast on the active deck: a
//!   cell blocks when any solid voxel intersects the band's eye-level
//!   z range. Cell opacity (and the light-gate sample) is cached per
//!   XY chunk tile and invalidated by [`Grid::chunk_version`]s, so
//!   edits are picked up without rescanning quiet chunks.
//! - The optional [`LightGate`] keeps *dark* cells from becoming
//!   Visible: geometric LOS is scaled by the baked brightness of the
//!   first solid surface under the eye (plus an emissive colour-key
//!   list — an emissive strip lights its cell regardless of bake).
//!   Dynamic lights do NOT open vision in v1 (entry-doc decision 5).
//!
//! This module is **presentation-only** state (entry-doc decision 10):
//! nothing in the simulation, collision, audio, or pathing may read
//! the mask, and no snapshot wire change is involved (savegame
//! persistence of explored area is an explicit follow-up). Renderers
//! consume it in FW.2/FW.3; the known-twin sync (FW.1) walks
//! [`FogOfWar::for_each_live_cell`].

use std::collections::HashMap;

use glam::{IVec2, IVec3, UVec3, Vec2};
use roxlap_formats::color::Rgb;
use roxlap_formats::vxl::Vxl;

use crate::{Grid, CHUNK_SIZE_XY};

/// Maximum cell intensity (6 bits of the mask byte).
pub const INTENSITY_MAX: u8 = 63;

/// XY size of one mask/opacity tile — locked to the chunk footprint so
/// tile keys are chunk indices and version invalidation is 1:1.
const TILE: i32 = CHUNK_SIZE_XY as i32;
const TILE_CELLS: usize = (TILE * TILE) as usize;
const TILE_WORDS: usize = TILE_CELLS / 64;

/// Intensity settle threshold — transitions below this snap and stop
/// ticking.
const SETTLE_EPS: f32 = 0.25;

/// What the observer knows about a cell. Packed into the top 2 bits
/// of the mask byte (intensity in the low 6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CellState {
    /// Never seen — renderers draw background (FW.2).
    Unseen = 0,
    /// Seen before — the known twin's frozen last-seen look, dimmed.
    Memory = 1,
    /// In the current facing cone or peripheral radius — live.
    Visible = 2,
    /// Revealed by a heard sound source — live data, Memory styling.
    Heard = 3,
}

impl CellState {
    fn from_bits(bits: u8) -> Self {
        match bits & 3 {
            1 => Self::Memory,
            2 => Self::Visible,
            3 => Self::Heard,
            _ => Self::Unseen,
        }
    }
}

/// One deck's z extent, grid-local, inclusive, z-down (`z_top` is the
/// ceiling — numerically the smallest z). The caller derives these
/// from the same levels as CA's deck clips.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeckBand {
    /// Ceiling z (inclusive, smallest value).
    pub z_top: i32,
    /// Floor z (inclusive, largest value) — the light gate scans down
    /// to here for the surface brightness sample.
    pub z_bottom: i32,
    /// Top of the eye-level opacity band (inclusive).
    pub eye_top: i32,
    /// Bottom of the eye-level opacity band (inclusive). A cell
    /// blocks LOS when any solid voxel lies in `[eye_top, eye_bottom]`.
    pub eye_bottom: i32,
}

impl DeckBand {
    /// Does grid-local `z` fall inside this deck's `[z_top, z_bottom]`?
    #[must_use]
    pub fn contains_z(&self, z: i32) -> bool {
        (self.z_top..=self.z_bottom).contains(&z)
    }
}

/// Decision 5 — light gates vision: a cell only reaches full Visible
/// intensity when its surface is lit. Samples the **baked** brightness
/// byte (colour high byte) of the first solid voxel scanning down from
/// `eye_top` to the deck floor; emissive colour keys count as fully
/// lit anywhere in that scan. Dynamic point lights are not sampled in
/// v1.
#[derive(Debug, Clone, PartialEq)]
pub struct LightGate {
    /// Baked-brightness byte at/above which the cell is fully lit
    /// (bakes centre around `0x80` neutral).
    pub lit_brightness: u8,
    /// Soft knee below the threshold: brightness in
    /// `[lit_brightness - softness, lit_brightness]` scales visibility
    /// linearly. `0` = hard cut.
    pub softness: u8,
    /// Colour keys (brightness-stripped, the engine's colour→material
    /// map convention) treated as self-lit — emissive strips keep
    /// their cell visible in a pitch-dark corridor.
    pub emissive_colors: Vec<Rgb>,
}

impl LightGate {
    /// Visibility factor `0..=1` for a surface of baked brightness
    /// `b`.
    #[must_use]
    pub fn lit_factor(&self, b: u8) -> f32 {
        if b >= self.lit_brightness {
            return 1.0;
        }
        if self.softness == 0 {
            return 0.0;
        }
        let knee = self.lit_brightness.saturating_sub(self.softness);
        if b <= knee {
            return 0.0;
        }
        f32::from(b - knee) / f32::from(self.softness)
    }
}

/// All FW tuning in one place. Geometry fields are in cells (mip-0
/// voxels) and radians; rates are intensity units (`0..=63`) per
/// second. The styling knobs at the bottom are read by the renderers
/// from FW.2 on — unread in FW.0.
#[derive(Debug, Clone, PartialEq)]
pub struct VisionConfig {
    /// Half-angle of the facing cone, radians.
    pub cone_half_angle: f32,
    /// Angular taper band *inside* the half-angle over which intensity
    /// falls to zero (smooth cone rim), radians.
    pub cone_taper: f32,
    /// Facing-cone reach, cells.
    pub range: f32,
    /// 360° close-range vision, cells (an observer is never blind in
    /// their own back at arm's length).
    pub peripheral_range: f32,
    /// Radial taper band inside `range` / `peripheral_range` /
    /// heard-blob radii, cells.
    pub edge_taper: f32,
    /// The decks, index-aligned with [`FowObserver::deck`] and every
    /// per-deck query. Must stay the same length across
    /// [`FogOfWar::set_config`] calls (layers are keyed by index).
    pub decks: Vec<DeckBand>,
    /// Decision 5 — `None` = geometric visibility only.
    pub light_gate: Option<LightGate>,
    /// Intensity rise rate toward a live (Visible/Heard) target.
    pub fade_in: f32,
    /// Intensity fall rate when leaving sight, down to
    /// `memory_intensity`.
    pub fade_out: f32,
    /// Where a freshly-demoted Memory cell settles before slow decay.
    pub memory_intensity: u8,
    /// Decision 7 — slow decay rate from `memory_intensity` toward
    /// `memory_floor`. `0` = memories never fade.
    pub memory_decay: f32,
    /// Decayed-memory minimum intensity — old memories stay barely
    /// readable rather than vanishing back to Unseen.
    pub memory_floor: u8,
    /// Heard-blob radius in cells per unit of loudness.
    pub heard_radius: f32,
    /// Seconds a heard blob stays live before fading out.
    pub heard_duration: f32,
    /// FW.2+ styling: Memory/Heard brightness multiplier.
    pub memory_dim: f32,
    /// FW.2+ styling: Memory/Heard desaturation amount `0..=1`.
    pub memory_desaturate: f32,
}

impl VisionConfig {
    /// Defaults tuned for ship scale (~1 voxel ≈ 10 cm): a 120° cone
    /// reaching 96 cells, 12-cell peripheral, light gate off.
    #[must_use]
    pub fn for_decks(decks: Vec<DeckBand>) -> Self {
        Self {
            cone_half_angle: 1.05,
            cone_taper: 0.12,
            range: 96.0,
            peripheral_range: 12.0,
            edge_taper: 4.0,
            decks,
            light_gate: None,
            fade_in: 240.0,
            fade_out: 120.0,
            memory_intensity: 26,
            memory_decay: 0.8,
            memory_floor: 10,
            heard_radius: 10.0,
            heard_duration: 1.5,
            memory_dim: 0.45,
            memory_desaturate: 0.6,
        }
    }

    /// Index of the deck whose `[z_top, z_bottom]` contains grid-local
    /// `z` (the head-follow deck pick, CA round-3 convention: feed the
    /// HEAD z, not the feet).
    #[must_use]
    pub fn deck_for_z(&self, z: i32) -> Option<usize> {
        self.decks.iter().position(|d| d.contains_z(z))
    }
}

/// Who is looking: grid-local cell, facing, and active deck index.
/// The caller converts from world space ([`crate::world_to_grid_local`])
/// — keeping the mask grid-local makes ship rotation/movement free.
#[derive(Debug, Clone, Copy)]
pub struct FowObserver {
    /// Observer cell (grid-local mip-0 XY column).
    pub cell: IVec2,
    /// Facing direction, grid-local XY. Zero disables the cone
    /// (peripheral-only vision).
    pub facing: Vec2,
    /// Active deck index into [`VisionConfig::decks`].
    pub deck: usize,
}

/// One 128×128 mask tile: byte per cell, state in bits 6–7, intensity
/// in bits 0–5.
struct MaskTile {
    bytes: Box<[u8; TILE_CELLS]>,
}

impl MaskTile {
    fn new() -> Self {
        Self {
            bytes: vec![0u8; TILE_CELLS].into_boxed_slice().try_into().unwrap(),
        }
    }
}

/// Lazily-computed per-cell opacity + light sample for one XY chunk
/// tile, keyed by the chunk-version mix so edits invalidate it.
struct OpacityTile {
    key: u64,
    computed: Box<[u64; TILE_WORDS]>,
    blocked: Box<[u64; TILE_WORDS]>,
    /// Quantised lit factor `0..=255` (255 = fully lit); only
    /// meaningful where `computed` is set.
    lit: Box<[u8; TILE_CELLS]>,
}

impl OpacityTile {
    fn new(key: u64) -> Self {
        Self {
            key,
            computed: vec![0u64; TILE_WORDS]
                .into_boxed_slice()
                .try_into()
                .unwrap(),
            blocked: vec![0u64; TILE_WORDS]
                .into_boxed_slice()
                .try_into()
                .unwrap(),
            lit: vec![0u8; TILE_CELLS].into_boxed_slice().try_into().unwrap(),
        }
    }
}

/// Per-deck layers: sparse mask + opacity tiles keyed by chunk XY
/// index.
#[derive(Default)]
struct DeckLayer {
    mask: HashMap<(i32, i32), MaskTile>,
    opacity: HashMap<(i32, i32), OpacityTile>,
}

fn split_cell(cell: IVec2) -> ((i32, i32), usize) {
    let tx = cell.x.div_euclid(TILE);
    let ty = cell.y.div_euclid(TILE);
    let ix = cell.x.rem_euclid(TILE);
    let iy = cell.y.rem_euclid(TILE);
    ((tx, ty), (iy * TILE + ix) as usize)
}

impl DeckLayer {
    fn byte(&self, cell: IVec2) -> u8 {
        let (tile, idx) = split_cell(cell);
        self.mask.get(&tile).map_or(0, |t| t.bytes[idx])
    }

    /// Write a mask byte; returns whether the stored value changed.
    fn write(&mut self, cell: IVec2, val: u8) -> bool {
        let (tile, idx) = split_cell(cell);
        if val == 0 && !self.mask.contains_key(&tile) {
            return false;
        }
        let t = self.mask.entry(tile).or_insert_with(MaskTile::new);
        let changed = t.bytes[idx] != val;
        t.bytes[idx] = val;
        changed
    }
}

fn pack(state: CellState, intensity: u8) -> u8 {
    ((state as u8) << 6) | intensity.min(INTENSITY_MAX)
}

#[inline]
fn fnv(h: &mut u64, v: u64) {
    *h ^= v;
    *h = h.wrapping_mul(0x0000_0100_0000_01b3);
}

/// Mix one chunk's change-signature: the edit version AND its
/// materialisation state. Presence matters because a generated /
/// streamed-in chunk keeps `chunk_version == 0` (same as absent) —
/// keying on the version alone would reuse a stale "all air" opacity
/// tile after the chunk appears (walls behind the fresh geometry stay
/// visible). Folding presence in flips the key on the absent→present
/// transition too.
#[inline]
fn mix_chunk_sig(h: &mut u64, grid: &Grid, idx: IVec3) {
    fnv(h, grid.chunk_version(idx));
    fnv(h, u64::from(grid.chunk(idx).is_some()));
}

fn band_chz_range(band: &DeckBand) -> (i32, i32) {
    let cs_z = crate::CHUNK_SIZE_Z as i32;
    (
        band.z_top.min(band.eye_top).div_euclid(cs_z),
        band.z_bottom.max(band.eye_bottom).div_euclid(cs_z),
    )
}

/// FNV-1a mix of the chunk signatures (plus the config generation, so
/// light-gate retunes re-sample) backing one XY tile of a deck band.
fn tile_key(grid: &Grid, band: &DeckBand, config_gen: u64, tx: i32, ty: i32) -> u64 {
    let (chz_lo, chz_hi) = band_chz_range(band);
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    fnv(&mut h, config_gen);
    for chz in chz_lo..=chz_hi {
        mix_chunk_sig(&mut h, grid, IVec3::new(tx, ty, chz));
    }
    h
}

/// FNV-1a mix of every chunk signature the LOS scan could touch from
/// `origin` at `radius` (cells) on `band`. Keys the LOS recompute on
/// edits *within view* only — far-side debris / streaming elsewhere on
/// the ship no longer forces a full radius-96 shadowcast every frame
/// (the old key hung on the grid-wide `mutation_counter`).
fn local_edit_key(
    grid: &Grid,
    band: &DeckBand,
    config_gen: u64,
    origin: IVec2,
    radius: i32,
) -> u64 {
    let cs_xy = CHUNK_SIZE_XY as i32;
    let cx_lo = (origin.x - radius).div_euclid(cs_xy);
    let cx_hi = (origin.x + radius).div_euclid(cs_xy);
    let cy_lo = (origin.y - radius).div_euclid(cs_xy);
    let cy_hi = (origin.y + radius).div_euclid(cs_xy);
    let (chz_lo, chz_hi) = band_chz_range(band);
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    fnv(&mut h, config_gen);
    for chz in chz_lo..=chz_hi {
        for cy in cy_lo..=cy_hi {
            for cx in cx_lo..=cx_hi {
                mix_chunk_sig(&mut h, grid, IVec3::new(cx, cy, chz));
            }
        }
    }
    h
}

/// One live heard reveal. Its covered cells + intensities are computed
/// ONCE at [`FogOfWar::hear`] time (they never change over the TTL), so
/// the per-tick union carries no `sqrt`.
struct HeardBlob {
    deck: usize,
    cells: Vec<(IVec2, f32)>,
    ttl: f32,
}

/// The fog-of-war state for ONE FW-enabled grid (entry-doc decision
/// 3): per-deck mask layers, the LOS cache, heard blobs, and the
/// fade/decay state machine. Owned by the host beside the scene;
/// call [`Self::update`] once per tick with the observed character's
/// grid-local pose.
pub struct FogOfWar {
    config: VisionConfig,
    config_gen: u64,
    layers: Vec<DeckLayer>,
    /// Currently-visible cells of `visible_deck` → target intensity.
    visible: HashMap<(i32, i32), f32>,
    visible_deck: usize,
    los_key: Option<(IVec2, i32, usize, u64, u64)>,
    heard: Vec<HeardBlob>,
    /// Set when a blob was added, so the next tick rebuilds the union;
    /// blob expiry sets it inline. Steady state skips the rebuild.
    heard_dirty: bool,
    /// Cells currently stamped Heard → target intensity.
    heard_cells: HashMap<(usize, i32, i32), f32>,
    /// Cells mid-fade/decay → current fractional intensity (the float
    /// is the truth while transitioning; the mask byte is its
    /// rounding).
    transitions: HashMap<(usize, i32, i32), f32>,
    mask_version: u64,
}

impl FogOfWar {
    /// A fresh all-Unseen fog for `config`.
    #[must_use]
    pub fn new(config: VisionConfig) -> Self {
        let layers = config.decks.iter().map(|_| DeckLayer::default()).collect();
        Self {
            config,
            config_gen: 0,
            layers,
            visible: HashMap::new(),
            visible_deck: 0,
            los_key: None,
            heard: Vec::new(),
            heard_dirty: false,
            heard_cells: HashMap::new(),
            transitions: HashMap::new(),
            mask_version: 0,
        }
    }

    /// Current tuning.
    #[must_use]
    pub fn config(&self) -> &VisionConfig {
        &self.config
    }

    /// Replace the tuning (sliders in the demo). The deck list must
    /// keep its length — layers are keyed by index; explored memory
    /// survives the swap. Forces a LOS + light-sample recompute.
    ///
    /// # Panics
    /// If `config.decks.len()` differs from the current one.
    pub fn set_config(&mut self, config: VisionConfig) {
        assert_eq!(
            config.decks.len(),
            self.config.decks.len(),
            "FW deck count is fixed per FogOfWar (layers are index-keyed)"
        );
        self.config = config;
        self.config_gen += 1;
    }

    /// Bumped whenever any mask byte changes — FW.3 gates the GPU
    /// mask re-upload on it.
    #[must_use]
    pub fn mask_version(&self) -> u64 {
        self.mask_version
    }

    /// The deck the last [`Self::update`] ran LOS on.
    #[must_use]
    pub fn visible_deck(&self) -> usize {
        self.visible_deck
    }

    /// What the observer knows about `cell` on `deck`, plus the
    /// current fade intensity `0..=63`.
    #[must_use]
    pub fn state(&self, deck: usize, cell: IVec2) -> (CellState, u8) {
        let Some(layer) = self.layers.get(deck) else {
            return (CellState::Unseen, 0);
        };
        let b = layer.byte(cell);
        (CellState::from_bits(b >> 6), b & INTENSITY_MAX)
    }

    /// Decision 6 — a sound source at `cell` on `deck` was heard at
    /// `loudness` (feed it `source_acoustics` transmission×gain, or
    /// any game-side loudness): stamps a live Heard blob of radius
    /// `heard_radius × loudness` for `heard_duration` seconds.
    /// Hearing goes through walls by design — no LOS test.
    pub fn hear(&mut self, deck: usize, cell: IVec2, loudness: f32) {
        if deck >= self.config.decks.len() || loudness <= 0.0 {
            return;
        }
        let r = self.config.heard_radius * loudness;
        let ri = r.ceil() as i32;
        let taper = self.config.edge_taper.max(1.0);
        let mut cells = Vec::new();
        for dy in -ri..=ri {
            for dx in -ri..=ri {
                let d = f64::from(dx * dx + dy * dy).sqrt() as f32;
                let t = ((r - d) / taper).clamp(0.0, 1.0);
                if t > 0.0 {
                    cells.push((cell + IVec2::new(dx, dy), t * f32::from(INTENSITY_MAX)));
                }
            }
        }
        if cells.is_empty() {
            return;
        }
        self.heard.push(HeardBlob {
            deck,
            cells,
            ttl: self.config.heard_duration,
        });
        self.heard_dirty = true;
    }

    /// Every live cell — Visible and Heard — with its state. FW.1's
    /// known-twin sync walks this to decide which chunks re-copy.
    pub fn for_each_live_cell(&self, mut f: impl FnMut(usize, IVec2, CellState)) {
        for &(x, y) in self.visible.keys() {
            f(self.visible_deck, IVec2::new(x, y), CellState::Visible);
        }
        for &(deck, x, y) in self.heard_cells.keys() {
            // A cell the active deck also sees is reported once, as
            // Visible (it wins) — skip the Heard duplicate.
            if deck == self.visible_deck && self.visible.contains_key(&(x, y)) {
                continue;
            }
            f(deck, IVec2::new(x, y), CellState::Heard);
        }
    }

    /// Raw mask tiles of one deck (tile key = chunk XY index; 128×128
    /// row-major bytes) — the FW.3 GPU upload path.
    pub fn deck_tiles(&self, deck: usize) -> impl Iterator<Item = (IVec2, &[u8])> {
        self.layers.get(deck).into_iter().flat_map(|l| {
            l.mask
                .iter()
                .map(|(&(tx, ty), t)| (IVec2::new(tx, ty), &t.bytes[..]))
        })
    }

    /// Advance the fog one tick: recompute LOS if the observer moved /
    /// turned / switched deck or the grid was edited (the
    /// [`Grid::mutation_counter`] key), restamp heard blobs, and step
    /// every fading cell by `dt` seconds.
    pub fn update(&mut self, grid: &Grid, observer: &FowObserver, dt: f32) {
        let mut changed = false;

        // Deck switch: demote the whole visible set of the old deck.
        if observer.deck != self.visible_deck && !self.visible.is_empty() {
            let old = std::mem::take(&mut self.visible);
            let deck = self.visible_deck;
            for (cell, _) in old {
                changed |= self.demote_to_memory(deck, IVec2::new(cell.0, cell.1));
            }
            self.los_key = None;
        }
        self.visible_deck = observer.deck;

        // LOS recompute when the key moves.
        if observer.deck < self.config.decks.len() {
            let facing_q = quantize_facing(observer.facing);
            let band = self.config.decks[observer.deck];
            let radius = self.config.range.max(self.config.peripheral_range).ceil() as i32;
            let edit_key = local_edit_key(grid, &band, self.config_gen, observer.cell, radius);
            let key = (
                observer.cell,
                facing_q,
                observer.deck,
                edit_key,
                self.config_gen,
            );
            if self.los_key != Some(key) {
                self.los_key = Some(key);
                changed |= self.recompute_los(grid, observer);
            }
        } else if !self.visible.is_empty() {
            // Observer off every configured deck: nothing is visible.
            let old = std::mem::take(&mut self.visible);
            for (cell, _) in old {
                changed |= self.demote_to_memory(
                    observer.deck.min(self.layers.len()),
                    IVec2::new(cell.0, cell.1),
                );
            }
        }

        changed |= self.update_heard(dt);
        changed |= self.tick_transitions(dt);

        if changed {
            self.mask_version += 1;
        }
    }

    /// Rebuild the visible set via shadowcast; diff against the old
    /// one, stamping states and queueing fades. Returns whether any
    /// mask byte changed.
    fn recompute_los(&mut self, grid: &Grid, observer: &FowObserver) -> bool {
        let deck = observer.deck;
        let band = self.config.decks[deck];
        let mut new_visible: HashMap<(i32, i32), f32> = HashMap::new();
        {
            let layer = &mut self.layers[deck];
            let mut scan = LosScan {
                grid,
                cfg: &self.config,
                band: &band,
                config_gen: self.config_gen,
                layer,
                origin: observer.cell,
                facing: normalize_facing(observer.facing),
                out: &mut new_visible,
                key_memo: None,
            };
            scan.run();
        }

        let mut changed = false;
        // Demotions: previously visible, no longer.
        let old = std::mem::take(&mut self.visible);
        for &cell in old.keys() {
            if !new_visible.contains_key(&cell) {
                changed |= self.demote_to_memory(deck, IVec2::new(cell.0, cell.1));
            }
        }
        // Stamps: state bits flip to Visible immediately; intensity
        // fades via the transition map.
        for (&(x, y), &target) in &new_visible {
            let cell = IVec2::new(x, y);
            let cur = self.layers[deck].byte(cell);
            let intensity = cur & INTENSITY_MAX;
            changed |= self.layers[deck].write(cell, pack(CellState::Visible, intensity));
            let key = (deck, x, y);
            if (f32::from(intensity) - target).abs() > SETTLE_EPS {
                self.transitions
                    .entry(key)
                    .or_insert_with(|| f32::from(intensity));
            }
        }
        self.visible = new_visible;
        changed
    }

    /// Flip a cell to Memory (keeping its intensity) and queue its
    /// decay. No-op on Unseen cells.
    fn demote_to_memory(&mut self, deck: usize, cell: IVec2) -> bool {
        let Some(layer) = self.layers.get_mut(deck) else {
            return false;
        };
        let b = layer.byte(cell);
        if CellState::from_bits(b >> 6) == CellState::Unseen {
            return false;
        }
        let intensity = b & INTENSITY_MAX;
        let changed = layer.write(cell, pack(CellState::Memory, intensity));
        self.transitions
            .entry((deck, cell.x, cell.y))
            .or_insert_with(|| f32::from(intensity));
        changed
    }

    /// Tick blob TTLs, rebuild the heard union only when the blob set
    /// changed (perf: the union carries no `sqrt` — blob cells are
    /// precomputed), then stamp the Heard byte for every union cell the
    /// active-deck LOS pass didn't already claim as Visible.
    fn update_heard(&mut self, dt: f32) -> bool {
        let mut changed = false;
        let before = self.heard.len();
        for blob in &mut self.heard {
            blob.ttl -= dt;
        }
        self.heard.retain(|b| b.ttl > 0.0);
        let expired = self.heard.len() != before;

        // The union depends only on the blob set — rebuild solely when
        // a blob was added (`heard_dirty`) or expired. Visible-priority
        // is NOT baked in here (it changes as the observer moves); it is
        // applied at stamp time below, so union stays blob-only.
        if self.heard_dirty || expired {
            self.heard_dirty = false;
            let mut now: HashMap<(usize, i32, i32), f32> = HashMap::new();
            for blob in &self.heard {
                for &(cell, target) in &blob.cells {
                    let e = now.entry((blob.deck, cell.x, cell.y)).or_insert(0.0);
                    *e = e.max(target);
                }
            }
            // Demote cells that fell out of every blob — but NOT one the
            // just-run LOS pass marked Visible (the player turned toward
            // the noise): stomping it back to Memory left a live cell
            // dead ahead rendering as memory until the observer moved.
            let old = std::mem::take(&mut self.heard_cells);
            for &key in old.keys() {
                if now.contains_key(&key) {
                    continue;
                }
                if key.0 == self.visible_deck && self.visible.contains_key(&(key.1, key.2)) {
                    continue;
                }
                changed |= self.demote_to_memory(key.0, IVec2::new(key.1, key.2));
            }
            self.heard_cells = now;
        }

        // Stamp the Heard byte for union cells (skipping any the active
        // deck currently sees) and queue their fades. Runs every tick so
        // a cell that just left the visible cone reverts Visible→Heard.
        for (&(deck, x, y), &target) in &self.heard_cells {
            if deck == self.visible_deck && self.visible.contains_key(&(x, y)) {
                continue;
            }
            let cell = IVec2::new(x, y);
            let cur = self.layers[deck].byte(cell);
            let intensity = cur & INTENSITY_MAX;
            changed |= self.layers[deck].write(cell, pack(CellState::Heard, intensity));
            if (f32::from(intensity) - target).abs() > SETTLE_EPS {
                self.transitions
                    .entry((deck, x, y))
                    .or_insert_with(|| f32::from(intensity));
            }
        }
        changed
    }

    /// Step every transitioning cell toward its current target.
    fn tick_transitions(&mut self, dt: f32) -> bool {
        if self.transitions.is_empty() {
            return false;
        }
        let cfg = &self.config;
        let mut changed = false;
        let mut settled: Vec<(usize, i32, i32)> = Vec::new();
        let visible_deck = self.visible_deck;
        for (&key, cur) in &mut self.transitions {
            let (deck, x, y) = key;
            let cell = IVec2::new(x, y);
            // Target + rate from current membership.
            let (target, rate) = if deck == visible_deck && self.visible.contains_key(&(x, y)) {
                let t = self.visible[&(x, y)];
                (t, if *cur < t { cfg.fade_in } else { cfg.fade_out })
            } else if let Some(&t) = self.heard_cells.get(&key) {
                (t, if *cur < t { cfg.fade_in } else { cfg.fade_out })
            } else {
                // Memory: fast fall to memory_intensity, then slow
                // decay to memory_floor (0 decay = stop at sustain).
                let sustain = f32::from(cfg.memory_intensity);
                if *cur > sustain {
                    (sustain.max(f32::from(cfg.memory_floor)), cfg.fade_out)
                } else if cfg.memory_decay > 0.0 {
                    (f32::from(cfg.memory_floor).min(*cur), cfg.memory_decay)
                } else {
                    (*cur, f32::INFINITY)
                }
            };

            let step = rate * dt;
            if (*cur - target).abs() <= step.max(SETTLE_EPS) {
                *cur = target;
            } else if *cur < target {
                *cur += step;
            } else {
                *cur -= step;
            }

            let layer = &mut self.layers[deck];
            let b = layer.byte(cell);
            let state = CellState::from_bits(b >> 6);
            let rounded = (cur.round().clamp(0.0, f32::from(INTENSITY_MAX))) as u8;
            changed |= layer.write(cell, pack(state, rounded));
            if (*cur - target).abs() <= SETTLE_EPS {
                // Memory cells settle for good only once fully decayed
                // (or decay is off); live cells settle at their target.
                let memory_pending = state == CellState::Memory
                    && cfg.memory_decay > 0.0
                    && *cur > f32::from(cfg.memory_floor) + SETTLE_EPS;
                if !memory_pending {
                    settled.push(key);
                }
            }
        }
        for key in settled {
            self.transitions.remove(&key);
        }
        changed
    }
}

fn normalize_facing(f: Vec2) -> Option<Vec2> {
    let len = f.length();
    if len < 1e-6 {
        None
    } else {
        Some(f / len)
    }
}

/// Facing quantised to one of 2048 buckets `[0, 2048)`, or `i32::MIN`
/// for "no facing" (peripheral-only). The sentinel sits outside the
/// bucket range so a near-zero-angle facing (e.g. `(1, -0.003)`) can
/// never alias it — aliasing froze the cone on/off across the
/// facing↔zero transition. The `rem_euclid` also folds the ±π wrap
/// (bucket 2048 ≡ 0) so opposite-signed representations of the same
/// heading share a bucket.
fn quantize_facing(f: Vec2) -> i32 {
    match normalize_facing(f) {
        None => i32::MIN,
        Some(n) => {
            let ang = f64::from(n.y).atan2(f64::from(n.x));
            let raw = (ang / std::f64::consts::TAU * 2048.0).round() as i32;
            raw.rem_euclid(2048)
        }
    }
}

/// One LOS recompute: recursive shadowcasting (the classic RogueBasin
/// 8-octant algorithm) over the deck's opacity, classifying every
/// LOS-clear cell against the cone/peripheral shape and the light
/// gate.
struct LosScan<'a> {
    grid: &'a Grid,
    cfg: &'a VisionConfig,
    band: &'a DeckBand,
    config_gen: u64,
    layer: &'a mut DeckLayer,
    origin: IVec2,
    facing: Option<Vec2>,
    out: &'a mut HashMap<(i32, i32), f32>,
    /// Last (tile coords, tile_key) — shadowcasting has strong tile
    /// locality, so this reuses the FNV mix across the run of cells in
    /// the same tile instead of rehashing every chz version per cell.
    key_memo: Option<((i32, i32), u64)>,
}

/// Octant transforms (xx, xy, yx, yy).
const OCTANTS: [[i32; 4]; 8] = [
    [1, 0, 0, 1],
    [0, 1, 1, 0],
    [0, -1, 1, 0],
    [-1, 0, 0, 1],
    [-1, 0, 0, -1],
    [0, -1, -1, 0],
    [0, 1, -1, 0],
    [1, 0, 0, -1],
];

impl LosScan<'_> {
    fn radius(&self) -> i32 {
        self.cfg.range.max(self.cfg.peripheral_range).ceil() as i32
    }

    fn run(&mut self) {
        // The observer's own cell is always visible (gate-exempt: you
        // know where you stand even in the dark).
        self.out
            .insert((self.origin.x, self.origin.y), f32::from(INTENSITY_MAX));
        for m in OCTANTS {
            self.cast(1, 1.0, 0.0, m);
        }
    }

    /// Is LOS blocked at `cell`? Lazily computes + caches the cell's
    /// opacity/light sample.
    fn blocked_at(&mut self, cell: IVec2) -> bool {
        self.sample(cell).0
    }

    /// (blocked, lit factor 0..=1) for `cell`, from the version-keyed
    /// tile cache.
    fn sample(&mut self, cell: IVec2) -> (bool, f32) {
        let ((tx, ty), idx) = split_cell(cell);
        let key = match self.key_memo {
            Some((t, k)) if t == (tx, ty) => k,
            _ => {
                let k = tile_key(self.grid, self.band, self.config_gen, tx, ty);
                self.key_memo = Some(((tx, ty), k));
                k
            }
        };
        let tile = self
            .layer
            .opacity
            .entry((tx, ty))
            .or_insert_with(|| OpacityTile::new(key));
        if tile.key != key {
            *tile = OpacityTile::new(key);
        }
        let (w, bit) = (idx / 64, 1u64 << (idx % 64));
        if tile.computed[w] & bit == 0 {
            let (blocked, lit) =
                sample_cell(self.grid, self.band, self.cfg.light_gate.as_ref(), cell);
            tile.computed[w] |= bit;
            if blocked {
                tile.blocked[w] |= bit;
            }
            tile.lit[idx] = (lit * 255.0).round() as u8;
        }
        (tile.blocked[w] & bit != 0, f32::from(tile.lit[idx]) / 255.0)
    }

    /// Classify an LOS-clear (or wall-face) cell at offset
    /// `(dx, dy)` from the origin and record its target intensity.
    fn visit(&mut self, cell: IVec2, dx: i32, dy: i32) {
        let d = f64::from(dx * dx + dy * dy).sqrt() as f32;
        let taper = self.cfg.edge_taper.max(1.0);
        let mut vis = 0.0f32;
        if d <= self.cfg.peripheral_range {
            vis = ((self.cfg.peripheral_range - d) / taper).clamp(0.0, 1.0);
        }
        if let Some(f) = self.facing {
            if d <= self.cfg.range && d > 0.0 {
                let dir = Vec2::new(dx as f32, dy as f32) / d;
                let ang = dir.dot(f).clamp(-1.0, 1.0).acos();
                if ang <= self.cfg.cone_half_angle {
                    let ang_t = ((self.cfg.cone_half_angle - ang) / self.cfg.cone_taper.max(1e-3))
                        .clamp(0.0, 1.0);
                    let rad_t = ((self.cfg.range - d) / taper).clamp(0.0, 1.0);
                    vis = vis.max(ang_t.min(rad_t));
                }
            }
        }
        if vis <= 0.0 {
            return;
        }
        let lit = self.sample(cell).1;
        vis *= lit;
        if vis <= 0.0 {
            return;
        }
        let target = vis * f32::from(INTENSITY_MAX);
        let e = self.out.entry((cell.x, cell.y)).or_insert(0.0);
        *e = e.max(target);
    }

    /// One octant of recursive shadowcasting.
    fn cast(&mut self, row: i32, mut start: f64, end: f64, m: [i32; 4]) {
        if start < end {
            return;
        }
        let radius = self.radius();
        let r2 = i64::from(radius) * i64::from(radius);
        let mut new_start = start;
        let mut blocked = false;
        let mut j = row;
        while j <= radius && !blocked {
            let dy = -j;
            for dx in -j..=0 {
                let l_slope = (f64::from(dx) - 0.5) / (f64::from(dy) + 0.5);
                let r_slope = (f64::from(dx) + 0.5) / (f64::from(dy) - 0.5);
                if start < r_slope {
                    continue;
                }
                if end > l_slope {
                    break;
                }
                let sx = dx * m[0] + dy * m[1];
                let sy = dx * m[2] + dy * m[3];
                let cell = self.origin + IVec2::new(sx, sy);
                if i64::from(dx) * i64::from(dx) + i64::from(dy) * i64::from(dy) <= r2 {
                    self.visit(cell, sx, sy);
                }
                let blk = self.blocked_at(cell);
                if blocked {
                    if blk {
                        new_start = r_slope;
                    } else {
                        blocked = false;
                        start = new_start;
                    }
                } else if blk && j < radius {
                    blocked = true;
                    self.cast(j + 1, start, l_slope, m);
                    new_start = r_slope;
                }
            }
            j += 1;
        }
    }
}

/// Column scan for one cell, returning `(blocked, lit 0..=1)`:
/// - **blocked** — any solid voxel in the eye band `[eye_top,
///   eye_bottom]` (LOS occlusion); a ceiling / floor voxel never
///   blocks.
/// - **lit** — with a gate, the gate factor of the **floor** surface:
///   the first solid at or below `eye_top` scanning down to the deck
///   floor (a ceiling plate above the eye is NOT the surface — reading
///   its dark underside kept fully-lit rooms permanently unseen). An
///   emissive colour key ANYWHERE in the deck band forces `1.0`
///   (ceiling lamps light their cell) without marking it blocked. No
///   gate ⇒ always `1.0`; no floor found under a gate ⇒ `0.0`.
///
/// Borrows the chunk once per z-layer (`chz`) rather than a per-voxel
/// grid probe — a first-fill of a 128² tile was ~340k `voxel_split` +
/// HashMap lookups.
fn sample_cell(grid: &Grid, band: &DeckBand, gate: Option<&LightGate>, cell: IVec2) -> (bool, f32) {
    let cs_xy = CHUNK_SIZE_XY as i32;
    let cs_z = crate::CHUNK_SIZE_Z as i32;
    let chx = cell.x.div_euclid(cs_xy);
    let chy = cell.y.div_euclid(cs_xy);
    let ix = cell.x.rem_euclid(cs_xy) as u32;
    let iy = cell.y.rem_euclid(cs_xy) as u32;

    let z_lo = band.z_top.min(band.eye_top);
    let z_hi = band.z_bottom.max(band.eye_bottom);

    let mut blocked = false;
    let mut surface: Option<u8> = None;
    let mut lit_override = false;

    let mut cur_chz = i32::MIN;
    let mut chunk: Option<&Vxl> = None;
    for z in z_lo..=z_hi {
        let chz = z.div_euclid(cs_z);
        if chz != cur_chz {
            cur_chz = chz;
            chunk = grid.chunk(IVec3::new(chx, chy, chz));
        }
        let Some(vxl) = chunk else {
            continue;
        };
        let iz = z.rem_euclid(cs_z) as u32;
        if !Grid::chunk_voxel_solid(vxl, UVec3::new(ix, iy, iz)) {
            continue;
        }
        if (band.eye_top..=band.eye_bottom).contains(&z) {
            blocked = true;
        }
        if let Some(g) = gate {
            let color = vxl.voxel_color(ix, iy, iz);
            if let Some(c) = color {
                if g.emissive_colors.contains(&c.rgb_part()) {
                    lit_override = true;
                }
            }
            // Floor surface = first solid at or below eye level; skip
            // the ceiling region above the eye.
            if surface.is_none() && z >= band.eye_top {
                surface = Some(color.map_or(0x80, |c| (c.0 >> 24) as u8));
            }
        } else if blocked {
            // No gate: once the eye band is occluded there is nothing
            // more to learn from this column.
            return (true, 1.0);
        }
    }

    let lit = match gate {
        None => 1.0,
        Some(_) if lit_override => 1.0,
        Some(g) => surface.map_or(0.0, |b| g.lit_factor(b)),
    };
    (blocked, lit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GridTransform;
    use roxlap_formats::color::VoxColor;

    const FLOOR_Z: i32 = 100;

    fn band() -> DeckBand {
        DeckBand {
            z_top: 80,
            z_bottom: FLOOR_Z,
            eye_top: 92,
            eye_bottom: 97,
        }
    }

    /// Flat lit floor `[-64, 63]²` at `FLOOR_Z`, one deck.
    fn open_room() -> Grid {
        let mut g = Grid::new(GridTransform::identity());
        g.set_rect(
            IVec3::new(-64, -64, FLOOR_Z),
            IVec3::new(63, 63, FLOOR_Z),
            Some(VoxColor::rgb(120, 120, 120)),
        );
        g
    }

    fn cfg() -> VisionConfig {
        let mut c = VisionConfig::for_decks(vec![band()]);
        c.range = 40.0;
        c.peripheral_range = 12.0;
        c
    }

    fn observer(cell: IVec2, facing: Vec2) -> FowObserver {
        FowObserver {
            cell,
            facing,
            deck: 0,
        }
    }

    /// A wall column blocking the eye band at `(x, y)`.
    fn wall(g: &mut Grid, x: i32, y: i32) {
        g.set_rect(
            IVec3::new(x, y, 88),
            IVec3::new(x, y, FLOOR_Z),
            Some(VoxColor::rgb(200, 60, 60)),
        );
    }

    const SETTLE: f32 = 1000.0;

    #[test]
    fn cone_and_peripheral_classification() {
        let g = open_room();
        let mut fow = FogOfWar::new(cfg());
        fow.update(&g, &observer(IVec2::new(0, 0), Vec2::X), SETTLE);

        // Ahead, inside the cone.
        assert_eq!(fow.state(0, IVec2::new(20, 0)).0, CellState::Visible);
        // Behind, inside the peripheral radius.
        assert_eq!(fow.state(0, IVec2::new(-6, 0)).0, CellState::Visible);
        // Behind, beyond the peripheral radius.
        assert_eq!(fow.state(0, IVec2::new(-25, 0)).0, CellState::Unseen);
        // Ahead but beyond range.
        assert_eq!(fow.state(0, IVec2::new(60, 0)).0, CellState::Unseen);
        // Full intensity well inside the cone.
        assert_eq!(fow.state(0, IVec2::new(20, 0)).1, INTENSITY_MAX);
    }

    #[test]
    fn wall_blocks_line_of_sight() {
        let mut g = open_room();
        for y in -12..=12 {
            wall(&mut g, 15, y);
        }
        let mut fow = FogOfWar::new(cfg());
        fow.update(&g, &observer(IVec2::new(0, 0), Vec2::X), SETTLE);

        // The wall face itself is visible…
        assert_eq!(fow.state(0, IVec2::new(15, 0)).0, CellState::Visible);
        // …the room behind it is not.
        assert_eq!(fow.state(0, IVec2::new(25, 0)).0, CellState::Unseen);
        // Open cells before the wall are visible.
        assert_eq!(fow.state(0, IVec2::new(10, 0)).0, CellState::Visible);
    }

    #[test]
    fn edit_invalidates_opacity_cache() {
        let mut g = open_room();
        let mut fow = FogOfWar::new(cfg());
        fow.update(&g, &observer(IVec2::new(0, 0), Vec2::X), SETTLE);
        assert_eq!(fow.state(0, IVec2::new(25, 0)).0, CellState::Visible);

        // Build the wall AFTER first sight; the edit bumps chunk
        // versions + the mutation counter, so the same observer pose
        // recomputes against fresh opacity.
        for y in -12..=12 {
            wall(&mut g, 15, y);
        }
        fow.update(&g, &observer(IVec2::new(0, 0), Vec2::X), SETTLE);
        assert_eq!(fow.state(0, IVec2::new(25, 0)).0, CellState::Memory);
        assert_eq!(fow.state(0, IVec2::new(15, 0)).0, CellState::Visible);
    }

    #[test]
    fn memory_decays_to_floor_and_reseeing_resets() {
        let g = open_room();
        let mut c = cfg();
        c.memory_decay = 8.0;
        let floor = c.memory_floor;
        let sustain = c.memory_intensity;
        let mut fow = FogOfWar::new(c);
        let cell = IVec2::new(20, 0);
        fow.update(&g, &observer(IVec2::new(0, 0), Vec2::X), SETTLE);
        assert_eq!(fow.state(0, cell), (CellState::Visible, INTENSITY_MAX));

        // Turn around: the cell leaves the cone and falls to Memory.
        fow.update(&g, &observer(IVec2::new(0, 0), -Vec2::X), 0.05);
        assert_eq!(fow.state(0, cell).0, CellState::Memory);
        let after_fall = fow.state(0, cell).1;
        assert!(after_fall < INTENSITY_MAX);

        // Long fade-out settles at the sustain level, then decay
        // grinds it down to the floor and stops.
        fow.update(&g, &observer(IVec2::new(0, 0), -Vec2::X), 1.0);
        let at_sustain = fow.state(0, cell).1;
        assert!(at_sustain <= sustain);
        fow.update(&g, &observer(IVec2::new(0, 0), -Vec2::X), 30.0);
        assert_eq!(fow.state(0, cell).1, floor);

        // Re-seeing restores full Visible.
        fow.update(&g, &observer(IVec2::new(0, 0), Vec2::X), SETTLE);
        assert_eq!(fow.state(0, cell), (CellState::Visible, INTENSITY_MAX));
    }

    #[test]
    fn intensity_fades_gradually() {
        let g = open_room();
        let mut fow = FogOfWar::new(cfg());
        let cell = IVec2::new(20, 0);
        // fade_in 240/s × 0.05 s = 12 units — mid-fade after one tick.
        fow.update(&g, &observer(IVec2::new(0, 0), Vec2::X), 0.05);
        let (state, mid) = fow.state(0, cell);
        assert_eq!(state, CellState::Visible);
        assert!(mid > 0 && mid < INTENSITY_MAX, "mid-fade, got {mid}");
        fow.update(&g, &observer(IVec2::new(0, 0), Vec2::X), SETTLE);
        assert_eq!(fow.state(0, cell).1, INTENSITY_MAX);
    }

    #[test]
    fn light_gate_caps_dark_cells() {
        // Lit floor up to x = 9, dark floor beyond (built disjoint —
        // inserting over an existing solid voxel keeps its colour, so
        // an overwrite would silently stay lit).
        let mut g = Grid::new(GridTransform::identity());
        g.set_rect(
            IVec3::new(-64, -64, FLOOR_Z),
            IVec3::new(9, 63, FLOOR_Z),
            Some(VoxColor::rgb(120, 120, 120)),
        );
        g.set_rect(
            IVec3::new(10, -64, FLOOR_Z),
            IVec3::new(63, 63, FLOOR_Z),
            Some(VoxColor::rgb(120, 120, 120).with_brightness(20)),
        );
        // An emissive strip voxel on the dark side.
        let emissive = VoxColor::rgb(80, 255, 160);
        g.set_voxel(
            IVec3::new(30, 5, FLOOR_Z - 1),
            Some(emissive.with_brightness(20)),
        );

        let mut c = cfg();
        c.light_gate = Some(LightGate {
            lit_brightness: 100,
            softness: 0,
            emissive_colors: vec![emissive.rgb_part()],
        });
        let mut fow = FogOfWar::new(c);
        fow.update(&g, &observer(IVec2::new(0, 0), Vec2::X), SETTLE);

        // Lit floor (VoxColor::rgb default brightness 0x80 = 128).
        assert_eq!(fow.state(0, IVec2::new(5, 0)).0, CellState::Visible);
        // Dark floor in the cone stays unknown.
        assert_eq!(fow.state(0, IVec2::new(20, 0)).0, CellState::Unseen);
        // The emissive cell is visible in the dark.
        assert_eq!(fow.state(0, IVec2::new(30, 5)).0, CellState::Visible);
    }

    #[test]
    fn deck_layers_are_independent() {
        let mut g = open_room();
        // A second, lower deck with its own floor.
        g.set_rect(
            IVec3::new(-64, -64, 130),
            IVec3::new(63, 63, 130),
            Some(VoxColor::rgb(90, 90, 140)),
        );
        let lower = DeckBand {
            z_top: 110,
            z_bottom: 130,
            eye_top: 122,
            eye_bottom: 127,
        };
        let mut c = cfg();
        c.decks = vec![band(), lower];
        let mut fow = FogOfWar::new(c);

        fow.update(&g, &observer(IVec2::new(0, 0), Vec2::X), SETTLE);
        assert_eq!(fow.state(0, IVec2::new(20, 0)).0, CellState::Visible);
        assert_eq!(fow.state(1, IVec2::new(20, 0)).0, CellState::Unseen);

        // Descend to deck 1: deck 0 demotes to Memory, deck 1 lights up.
        let mut obs = observer(IVec2::new(0, 0), Vec2::X);
        obs.deck = 1;
        fow.update(&g, &obs, SETTLE);
        assert_eq!(fow.state(0, IVec2::new(20, 0)).0, CellState::Memory);
        assert_eq!(fow.state(1, IVec2::new(20, 0)).0, CellState::Visible);
    }

    #[test]
    fn heard_blob_reveals_behind_wall_then_fades() {
        let mut g = open_room();
        for y in -12..=12 {
            wall(&mut g, 15, y);
        }
        let mut fow = FogOfWar::new(cfg());
        let src = IVec2::new(25, 0);
        fow.update(&g, &observer(IVec2::new(0, 0), Vec2::X), SETTLE);
        assert_eq!(fow.state(0, src).0, CellState::Unseen);

        // A noise behind the wall: revealed without LOS.
        fow.hear(0, src, 1.0);
        fow.update(&g, &observer(IVec2::new(0, 0), Vec2::X), 0.5);
        assert_eq!(fow.state(0, src).0, CellState::Heard);
        assert!(fow.state(0, IVec2::new(27, 2)).1 > 0);

        // Past the TTL the blob expires and the pocket becomes Memory.
        fow.update(&g, &observer(IVec2::new(0, 0), Vec2::X), 2.0);
        assert_eq!(fow.state(0, src).0, CellState::Memory);
    }

    #[test]
    fn mask_version_tracks_changes() {
        let g = open_room();
        let mut fow = FogOfWar::new(cfg());
        let v0 = fow.mask_version();
        fow.update(&g, &observer(IVec2::new(0, 0), Vec2::X), SETTLE);
        let v1 = fow.mask_version();
        assert!(v1 > v0);
        // Same pose, settled fades: nothing changes.
        fow.update(&g, &observer(IVec2::new(0, 0), Vec2::X), 1.0);
        assert_eq!(fow.mask_version(), v1);
    }

    #[test]
    fn live_cells_cover_visible_and_heard() {
        let mut g = open_room();
        for y in -12..=12 {
            wall(&mut g, 15, y);
        }
        let mut fow = FogOfWar::new(cfg());
        fow.hear(0, IVec2::new(25, 0), 0.5);
        fow.update(&g, &observer(IVec2::new(0, 0), Vec2::X), 0.5);
        let mut visible = 0usize;
        let mut heard = 0usize;
        fow.for_each_live_cell(|deck, _, state| {
            assert_eq!(deck, 0);
            match state {
                CellState::Visible => visible += 1,
                CellState::Heard => heard += 1,
                _ => panic!("live cells are Visible or Heard"),
            }
        });
        assert!(visible > 0 && heard > 0);
    }

    // ---- regression tests for the FW.0 review round ----

    /// Bug 1 — the light gate must sample the FLOOR under the eye, not
    /// a ceiling plate above it. A fully-lit floor room with a dark
    /// ceiling stayed permanently Unseen.
    #[test]
    fn light_gate_reads_floor_not_ceiling() {
        let mut g = open_room(); // lit floor at z=100 (brightness 0x80)
                                 // Dark ceiling plate above the eye band (z=85 < eye_top=92)
                                 // over the far half of the room.
        g.set_rect(
            IVec3::new(10, -32, 85),
            IVec3::new(40, 32, 85),
            Some(VoxColor::rgb(120, 120, 120).with_brightness(15)),
        );
        let mut c = cfg();
        c.light_gate = Some(LightGate {
            lit_brightness: 100,
            softness: 0,
            emissive_colors: Vec::new(),
        });
        let mut fow = FogOfWar::new(c);
        fow.update(&g, &observer(IVec2::new(0, 0), Vec2::X), SETTLE);
        // Floor under the dark ceiling is lit → Visible (the ceiling is
        // not the surface the gate reads).
        assert_eq!(fow.state(0, IVec2::new(20, 0)).0, CellState::Visible);
    }

    /// Bug 2 — a full `Grid::bake` rewrites brightness bytes; it must
    /// bump chunk versions so the FW opacity/light cache re-reads.
    #[test]
    fn full_bake_bumps_chunk_version() {
        let g = open_room();
        let (chunk_idx, _) = crate::voxel_split(IVec3::new(0, 0, FLOOR_Z));
        let v0 = g.chunk_version(chunk_idx);
        let mut g = g;
        g.bake(crate::BakeMode::Directional);
        assert!(
            g.chunk_version(chunk_idx) > v0,
            "full bake must bump the version (was {v0})"
        );
    }

    /// Bug 3 — a chunk materialised at version 0 (generation / stream-in
    /// keeps `chunk_version == 0`) must still invalidate the opacity
    /// tile: the key folds in chunk presence, so absent→present flips it
    /// even with no version change.
    #[test]
    fn tile_key_changes_on_materialisation_at_version_zero() {
        let mut g = Grid::new(GridTransform::identity());
        let b = band();
        let k_absent = tile_key(&g, &b, 0, 0, 0);
        // Materialise chunk (0,0,0) as empty air — mirrors a stream-in:
        // version stays 0, only presence changes.
        let _ = g.ensure_chunk(IVec3::new(0, 0, 0));
        assert_eq!(g.chunk_version(IVec3::new(0, 0, 0)), 0);
        let k_present = tile_key(&g, &b, 0, 0, 0);
        assert_ne!(
            k_absent, k_present,
            "presence must change the tile key at version 0"
        );
    }

    /// Bug 4 — an emissive voxel on the CEILING (above the eye band)
    /// lights its cell without blocking LOS. It used to be treated as a
    /// wall, carving an Unseen wedge behind each ceiling lamp.
    #[test]
    fn ceiling_emissive_lights_without_blocking() {
        let mut g = open_room();
        let emissive = VoxColor::rgb(80, 255, 160);
        // Lamp on the ceiling (z=85, above eye_top=92).
        g.set_voxel(IVec3::new(20, 0, 85), Some(emissive.with_brightness(30)));
        let mut c = cfg();
        c.light_gate = Some(LightGate {
            lit_brightness: 100,
            softness: 0,
            emissive_colors: vec![emissive.rgb_part()],
        });
        let mut fow = FogOfWar::new(c);
        fow.update(&g, &observer(IVec2::new(0, 0), Vec2::X), SETTLE);
        // The lamp cell is lit and does NOT block: a cell behind it
        // stays visible.
        assert_eq!(fow.state(0, IVec2::new(20, 0)).0, CellState::Visible);
        assert_eq!(fow.state(0, IVec2::new(25, 0)).0, CellState::Visible);
    }

    /// Bug 5 — turning toward a heard source must leave the cell
    /// Visible, not have `update_heard` immediately stomp it to Memory.
    #[test]
    fn facing_a_heard_source_stays_visible() {
        let g = open_room();
        let mut fow = FogOfWar::new(cfg());
        let src = IVec2::new(25, 0);
        // Heard while facing away (source behind, outside the cone).
        fow.hear(0, src, 1.0);
        fow.update(&g, &observer(IVec2::new(0, 0), -Vec2::X), 0.1);
        assert_eq!(fow.state(0, src).0, CellState::Heard);
        // Turn toward it (blob still alive): LOS claims it Visible and
        // the heard pass must not overwrite that.
        fow.update(&g, &observer(IVec2::new(0, 0), Vec2::X), 0.1);
        assert_eq!(fow.state(0, src).0, CellState::Visible);
    }

    /// Bug 6 — a near-zero-angle facing must not alias the "no facing"
    /// sentinel: switching from it to zero facing has to recompute LOS.
    #[test]
    fn near_zero_facing_distinct_from_none() {
        let g = open_room();
        let mut fow = FogOfWar::new(cfg());
        let cell = IVec2::new(30, 0); // in cone (range 40), past peripheral (12)
                                      // Facing ≈ -0.003 rad: its old bucket was -1, colliding with the
                                      // sentinel.
        fow.update(
            &g,
            &observer(IVec2::new(0, 0), Vec2::new(1.0, -0.003)),
            SETTLE,
        );
        assert_eq!(fow.state(0, cell).0, CellState::Visible);
        // Drop to no facing: must recompute (key differs) and the cone
        // cell falls out of sight.
        fow.update(&g, &observer(IVec2::new(0, 0), Vec2::ZERO), SETTLE);
        assert_eq!(fow.state(0, cell).0, CellState::Memory);
    }

    /// Perf 1 — a far-side edit outside the view radius must NOT force a
    /// LOS recompute (the key is local now, not the grid-wide mutation
    /// counter).
    #[test]
    fn far_edit_does_not_bump_mask_version() {
        let mut g = open_room();
        let mut fow = FogOfWar::new(cfg());
        fow.update(&g, &observer(IVec2::new(0, 0), Vec2::X), SETTLE);
        let v = fow.mask_version();
        // Edit far outside the radius-40 view (a whole chunk away).
        g.set_voxel(IVec3::new(600, 600, FLOOR_Z), Some(VoxColor::rgb(1, 2, 3)));
        fow.update(&g, &observer(IVec2::new(0, 0), Vec2::X), 0.0);
        assert_eq!(fow.mask_version(), v, "far edit should not recompute LOS");
    }
}
