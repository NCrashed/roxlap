//! DT.0 — floating-island detection, the destruction stage's core
//! (see `docs/porting/PORTING-DESTRUCTION.md`).
//!
//! After a carve, voxel regions that no longer connect to support must
//! crumble. [`detect_islands`] finds them with a budgeted breadth-first
//! flood over the RLE **runs** of the affected columns — a run is one
//! BFS node (chunk-format-native, no dense decode), 6-connected,
//! seeded from every run touching the carved bbox (±1 pad). A
//! component that reaches a **bedrock-anchored** run (the final run of
//! any materialised chunk's column: the `.vxl` format extends it to
//! the column bottom, and its bottom voxel at local z=255 is
//! uncarvable — `delslab` clamps below `MAXZDIM` — so a component
//! containing it can never physically fall, chz stacks included) or
//! grows past `budget` voxels is supported and abandoned early; only
//! components that exhaust under budget without finding support come
//! back as [`Island`]s.
//!
//! Cost per call ≈ O(Σ min(component size, budget)) over the distinct
//! components touching the bbox. False negatives are by design: a
//! detached region larger than `budget` stays floating (voxlap behaved
//! the same way) — raise the budget when a host wants bigger crumbles.

use std::collections::{HashMap, VecDeque};

use glam::{DVec3, IVec3, Vec3};
use roxlap_formats::color::VoxColor;
use roxlap_formats::kv6::Kv6;
use roxlap_formats::vxl::Vxl;

use crate::{grid_local_to_world, BakeMode, Grid, GridTransform, CHUNK_SIZE_XY, CHUNK_SIZE_Z};

/// Default flood budget in voxels: a component that grows past this is
/// declared supported (early exit). Tuned for carve-sized overhangs;
/// hosts wanting building-sized crumbles pass a bigger budget.
pub const DEFAULT_ISLAND_BUDGET: usize = 4096;

/// One detached voxel region: everything a caller needs to extract it
/// from the grid and hand it to a debris system.
#[derive(Debug, Clone)]
pub struct Island {
    /// Every solid voxel of the component, grid-local coordinates plus
    /// the colour it had at detection time (untextured interior cells
    /// inherit the nearest colour above them in the column).
    pub voxels: Vec<(IVec3, VoxColor)>,
    /// Inclusive grid-local bounds of `voxels`.
    pub bbox: (IVec3, IVec3),
}

impl Island {
    /// DT.1 — carve this island's voxels out of `grid`, then re-bake
    /// the touched region so lighting on the newly exposed faces is
    /// correct: one [`Grid::set_rect`] carve per contiguous column
    /// segment (the voxel list is stored run-major, so coalescing is a
    /// single pass) and one [`Grid::bake_bbox`] over the island bbox
    /// (its internal padding relights the surroundings; the edits and
    /// the bake bump the touched chunk versions). After extraction, a
    /// re-run of [`detect_islands`] over the same region finds
    /// nothing.
    ///
    /// **Mip ladders are NOT regenerated** — the same contract as
    /// every edit primitive ([`Grid::bake_bbox`] documents it too): a
    /// caller rendering distant chunks must remip the touched columns
    /// (`Vxl::remip_bbox`) exactly as it already does for its carves,
    /// or mip-N keeps drawing the extracted island beyond
    /// `mip_scan_dist` while its sprite twin falls next to it.
    pub fn extract(&self, grid: &mut Grid, bake: BakeMode) {
        if self.voxels.is_empty() {
            return;
        }
        let mut pending: Option<(IVec3, IVec3)> = None;
        for &(v, _) in &self.voxels {
            if let Some((lo, hi)) = pending {
                if v.x == hi.x && v.y == hi.y && v.z == hi.z + 1 {
                    pending = Some((lo, v));
                    continue;
                }
                grid.set_rect(lo, hi, None);
            }
            pending = Some((v, v));
        }
        if let Some((lo, hi)) = pending {
            grid.set_rect(lo, hi, None);
        }
        let (lo, hi) = self.bbox;
        grid.bake_bbox(lo, hi, bake);
    }

    /// DT.1 — build the island's KV6 sprite model: dims are the
    /// inclusive bbox extent, voxels sit at bbox-relative positions,
    /// and colours are the detected ones with the high byte normalised
    /// to `0x80` (full-bright — the sprite paths do their own
    /// lighting, and a baked brightness byte would double-darken).
    /// Surface-only ([`Kv6::from_fn`]): the interior is invisible in
    /// flight, and a shatter samples [`Self::voxels`], not the model.
    /// The model pivot is the bbox centre (`from_fn`'s convention) —
    /// pose it with [`Self::world_pivot`].
    #[must_use]
    #[allow(clippy::cast_possible_wrap)]
    pub fn to_kv6(&self) -> Kv6 {
        let (lo, hi) = self.bbox;
        let dims = (hi - lo + IVec3::ONE).as_uvec3();
        let map: HashMap<IVec3, u32> = self
            .voxels
            .iter()
            .map(|&(v, c)| (v - lo, (c.0 & 0x00ff_ffff) | 0x8000_0000))
            .collect();
        Kv6::from_fn(dims.x, dims.y, dims.z, |x, y, z| {
            map.get(&IVec3::new(x as i32, y as i32, z as i32))
                .map(|&c| VoxColor(c))
        })
    }

    /// DT.1 — the world-space position of the island's **minimum
    /// corner** (`bbox.0`, voxel corner), through the grid transform
    /// with `voxel_world_size` honoured (SC).
    #[must_use]
    pub fn world_origin(&self, transform: &GridTransform) -> DVec3 {
        let (chunk, in_chunk) = crate::voxel_split(self.bbox.0);
        grid_local_to_world(chunk, in_chunk, Vec3::ZERO, transform)
    }

    /// DT.1 — the world-space position of the island's **bbox centre**
    /// — where [`Self::to_kv6`]'s pivot sits, i.e. the position a
    /// debris system spawns the sprite instance at so the model lands
    /// exactly over the voxels it replaced.
    #[must_use]
    pub fn world_pivot(&self, transform: &GridTransform) -> DVec3 {
        let (chunk, in_chunk) = crate::voxel_split(self.bbox.0);
        let dims = (self.bbox.1 - self.bbox.0 + IVec3::ONE).as_vec3();
        grid_local_to_world(chunk, in_chunk, dims * 0.5, transform)
    }
}

/// DT.5 — how a detached region breaks apart before it falls, keyed
/// by material in a debris system's side table (deliberately NOT a
/// `Material` field — the render palette stays render-only).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FracturePattern {
    /// One piece — the default for unmapped materials.
    Whole,
    /// Rounded rubble (stone): jittered-Voronoi cells on a grid of
    /// roughly `cell`-voxel spacing. `cell` is clamped to ≥ 2.
    Chunks {
        /// Approximate fragment edge length in voxels.
        cell: u32,
    },
    /// Sharp plates (glass/crystal): the region sliced by 1–7
    /// near-parallel planes of one random orientation into `plates`
    /// slabs (clamped to `2..=8`), boundaries jittered.
    Shards {
        /// Number of plates to slice into.
        plates: u32,
    },
}

/// Tiny deterministic generator for the partitioners (SplitMix64) —
/// keeps `split` reproducible from its explicit seed with no RNG
/// state anywhere else.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in `[0, 1)`.
    #[allow(clippy::cast_precision_loss)]
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Uniform integer in `[0, n)` (`n ≥ 1`).
    #[allow(clippy::cast_possible_truncation)]
    fn below(&mut self, n: u32) -> i32 {
        (self.next() % u64::from(n.max(1))) as i32
    }
}

impl Island {
    /// DT.5 — partition this island into fragments per `pattern`: a
    /// **disjoint cover** of the original voxels (colours ride along),
    /// each fragment with a recomputed bbox, deterministic in
    /// `(island, pattern, seed)`. [`FracturePattern::Whole`] returns
    /// the island unsplit; degenerate cases (an island smaller than a
    /// cell, all sites in one plate) collapse gracefully to fewer
    /// fragments. A fragment may be internally disconnected (a
    /// Voronoi cell straddling a concave gap) — it falls as one piece
    /// (accepted v1 simplification).
    #[must_use]
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    pub fn split(&self, pattern: FracturePattern, seed: u64) -> Vec<Island> {
        let assign: Vec<usize> = match pattern {
            FracturePattern::Whole => return vec![self.clone()],
            FracturePattern::Chunks { cell } => {
                let cell = i32::try_from(cell.max(2)).unwrap_or(i32::MAX);
                let mut rng = SplitMix(seed ^ 0xC0C0);
                // One jittered Voronoi site per **occupied** cell
                // (keyed by the voxel's cell coordinate, first-seen
                // order — deterministic): cost scales with the island,
                // O(voxels × occupied cells), never with its bbox
                // volume — a long sparse island (a shot-off diagonal
                // vein) has a huge bbox but few occupied cells, and a
                // bbox-grid of sites would OOM on it (maintainer
                // review). Every voxel joins its nearest site (ties →
                // the lower site index).
                let mut cell_site: HashMap<IVec3, usize> = HashMap::new();
                let mut sites: Vec<IVec3> = Vec::new();
                for &(v, _) in &self.voxels {
                    let cc = IVec3::new(
                        v.x.div_euclid(cell),
                        v.y.div_euclid(cell),
                        v.z.div_euclid(cell),
                    );
                    if let std::collections::hash_map::Entry::Vacant(e) = cell_site.entry(cc) {
                        let jitter = IVec3::new(
                            rng.below(cell as u32),
                            rng.below(cell as u32),
                            rng.below(cell as u32),
                        );
                        e.insert(sites.len());
                        sites.push(cc * cell + jitter);
                    }
                }
                self.voxels
                    .iter()
                    .map(|&(v, _)| {
                        let mut best = 0usize;
                        let mut best_d = i64::MAX;
                        for (i, s) in sites.iter().enumerate() {
                            let d = (v - *s).as_i64vec3().length_squared();
                            if d < best_d {
                                best_d = d;
                                best = i;
                            }
                        }
                        best
                    })
                    .collect()
            }
            FracturePattern::Shards { plates } => {
                let k = plates.clamp(2, 8);
                let mut rng = SplitMix(seed ^ 0x5AAD);
                // One random unit normal (uniform on the sphere), then
                // k slabs over the projection range with jittered
                // boundaries — near-parallel plates.
                let z = 2.0 * rng.unit() - 1.0;
                let phi = std::f64::consts::TAU * rng.unit();
                let r = (1.0 - z * z).max(0.0).sqrt();
                let n = glam::DVec3::new(r * phi.cos(), r * phi.sin(), z);
                let projs: Vec<f64> = self
                    .voxels
                    .iter()
                    .map(|&(v, _)| v.as_dvec3().dot(n))
                    .collect();
                let (mut pmin, mut pmax) = (f64::MAX, f64::MIN);
                for &p in &projs {
                    pmin = pmin.min(p);
                    pmax = pmax.max(p);
                }
                let range = (pmax - pmin).max(1e-9);
                let step = range / f64::from(k);
                let bounds: Vec<f64> = (1..k)
                    .map(|i| pmin + step * (f64::from(i) + 0.4 * (rng.unit() - 0.5)))
                    .collect();
                projs
                    .iter()
                    .map(|&p| bounds.iter().filter(|&&b| p >= b).count())
                    .collect()
            }
        };
        // Group by fragment id, preserving first-seen order (stable,
        // deterministic), dropping empty cells/plates.
        let mut order: Vec<usize> = Vec::new();
        let mut groups: HashMap<usize, Vec<(IVec3, VoxColor)>> = HashMap::new();
        for (i, &(v, c)) in self.voxels.iter().enumerate() {
            let g = assign[i];
            groups.entry(g).or_insert_with(|| {
                order.push(g);
                Vec::new()
            });
            groups.get_mut(&g).expect("just inserted").push((v, c));
        }
        order
            .into_iter()
            .map(|g| {
                let voxels = groups.remove(&g).expect("grouped above");
                let mut lo = IVec3::MAX;
                let mut hi = IVec3::MIN;
                for &(v, _) in &voxels {
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
                Island {
                    voxels,
                    bbox: (lo, hi),
                }
            })
            .collect()
    }
}

/// One solid z-run of a column in **grid-local** z (chz stacking
/// already applied), `[top, bot)` half-open, z-down. `anchored` marks
/// the bedrock run that counts as support.
#[derive(Clone, Copy)]
struct Run {
    top: i32,
    bot: i32,
    anchored: bool,
}

/// Decode one chunk-local column's solid runs, mirroring the
/// `vxl_voxel_solid` slab walk (`chunks.rs`): run k spans
/// `[top_k, bot_k)`; the final run extends to the column bottom and is
/// flagged `is_final` (the format's implicit bedrock).
fn chunk_column_runs(vxl: &Vxl, x: u32, y: u32, mut f: impl FnMut(i32, i32, bool)) {
    let idx = (y * vxl.vsid + x) as usize;
    let slab = vxl.column_data(idx);
    let mut top = i32::from(slab[1]);
    let mut v = 0usize;
    while slab[v] != 0 {
        v += usize::from(slab[v]) * 4;
        if slab[v + 3] >= slab[v + 1] {
            // Degenerate slab (no air gap above): merges into the
            // current run — same skip `expandrle` takes.
            continue;
        }
        let bot = i32::from(slab[v + 3]);
        f(top, bot, false);
        top = i32::from(slab[v + 1]);
    }
    #[allow(clippy::cast_possible_wrap)]
    f(top, CHUNK_SIZE_Z as i32, true);
}

/// Tiny multiply-mix hasher for the `(x, y)` column keys: the flood's
/// hot loop is one map probe per neighbour column per node, and the
/// default SipHash dominated it (interior coordinates are not
/// DoS-facing, so the strong hash buys nothing here).
#[derive(Default)]
struct ColHash(u64);

impl std::hash::Hasher for ColHash {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
        }
    }
    #[allow(clippy::cast_sign_loss)]
    fn write_i32(&mut self, i: i32) {
        // Rotate between the tuple's two lanes so (x, y) ≠ (y, x).
        self.0 = (self.0 ^ u64::from(i as u32)).wrapping_mul(0xf135_7aea_2e62_a9c5);
        self.0 = self.0.rotate_left(26);
    }
}

type ColMap<V> = HashMap<(i32, i32), V, std::hash::BuildHasherDefault<ColHash>>;

/// Per-call flood state. Columns decode lazily into one shared
/// **arena** (`runs` + the parallel `visited` marks) — a per-column
/// `(start, len)` range in the map instead of per-column heap
/// allocations, because the budget-exit worst case (a thin plate:
/// thousands of one-voxel runs) is dominated by per-column setup, not
/// by the BFS itself. A one-entry chunk cache covers the same reason:
/// consecutive columns almost always live in the same chunk.
struct Flood<'a> {
    grid: &'a Grid,
    /// Sorted chz list per (chx, chy) — which chunks stack under an XY
    /// footprint (missing entries are implicit air).
    stacks: ColMap<Vec<i32>>,
    /// Column key → `(start, len)` range into `runs`/`visited`.
    columns: ColMap<(u32, u32)>,
    /// Run arena, grid-local z, in column-decode order.
    runs: Vec<Run>,
    /// Parallel to `runs`: id of the traversal that first reached the
    /// run, `0` = unvisited.
    visited: Vec<u32>,
    /// One-entry `Grid::chunk` cache (the `chunks` HashMap probe is
    /// SipHash — measurable at thousands of columns per call).
    last_chunk: Option<(IVec3, Option<&'a Vxl>)>,
}

impl<'a> Flood<'a> {
    fn new(grid: &'a Grid) -> Self {
        let mut stacks: ColMap<Vec<i32>> = ColMap::default();
        for k in grid.chunks.keys() {
            stacks.entry((k.x, k.y)).or_default().push(k.z);
        }
        for v in stacks.values_mut() {
            v.sort_unstable();
        }
        Self {
            grid,
            stacks,
            columns: ColMap::default(),
            runs: Vec::new(),
            visited: Vec::new(),
            last_chunk: None,
        }
    }

    /// The arena range of grid-local column `(x, y)`, decoding it on
    /// first touch: each present chunk in the chz stack contributes
    /// its runs offset by `chz * CHUNK_SIZE_Z`; runs meeting exactly
    /// at a chunk border merge; **every** chunk's final run is a
    /// bedrock anchor — its local z=255 voxel is format-pinned
    /// (uncarvable), so even on chz stacks a component containing it
    /// cannot fall. (Gating the anchor on "no chunk below" was wrong:
    /// it let the placeholder-bedrock sheet of an upper chunk flood
    /// into components as 128×128 phantom voxels.) Split field
    /// borrows keep the whole decode allocation-free (arena append).
    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    fn column_range(&mut self, x: i32, y: i32) -> (u32, u32) {
        if let Some(&r) = self.columns.get(&(x, y)) {
            return r;
        }
        let Self {
            grid,
            stacks,
            columns,
            runs,
            visited,
            last_chunk,
        } = self;
        let start = runs.len();
        let chx = x.div_euclid(CHUNK_SIZE_XY as i32);
        let chy = y.div_euclid(CHUNK_SIZE_XY as i32);
        let lx = x.rem_euclid(CHUNK_SIZE_XY as i32) as u32;
        let ly = y.rem_euclid(CHUNK_SIZE_XY as i32) as u32;
        let stack: &[i32] = stacks.get(&(chx, chy)).map_or(&[], Vec::as_slice);
        for &chz in stack {
            let idx = IVec3::new(chx, chy, chz);
            let vxl = match *last_chunk {
                Some((c, v)) if c == idx => v,
                _ => {
                    let v = grid.chunk(idx);
                    *last_chunk = Some((idx, v));
                    v
                }
            };
            let Some(vxl) = vxl else { continue };
            let base = chz * CHUNK_SIZE_Z as i32;
            chunk_column_runs(vxl, lx, ly, |top, bot, is_final| {
                let (top_g, bot_g) = (base + top, base + bot);
                let anchored = is_final;
                if runs.len() > start {
                    if let Some(last) = runs.last_mut() {
                        if last.bot == top_g {
                            last.bot = bot_g;
                            last.anchored |= anchored;
                            return;
                        }
                    }
                }
                runs.push(Run {
                    top: top_g,
                    bot: bot_g,
                    anchored,
                });
            });
        }
        visited.resize(runs.len(), 0);
        let r = (start as u32, (runs.len() - start) as u32);
        columns.insert((x, y), r);
        r
    }
}

/// Detect voxel regions detached from support after an edit inside the
/// grid-local bbox `[carve_lo, carve_hi]` (inclusive, any corner
/// order). Seeds every solid run within the bbox padded by ±1 voxel —
/// anything the carve could have disconnected touches that pad — and
/// floods each unvisited component with 6-connectivity. Components
/// that reach a bedrock-anchored run, exceed `budget` voxels, or merge
/// into a previously-flooded (supported) region are dropped; the rest
/// are returned as [`Island`]s, deterministic in both order and
/// content for a given grid state.
///
/// Runs where the carve itself ran — call it **after** the edit.
#[must_use]
pub fn detect_islands(grid: &Grid, carve_lo: IVec3, carve_hi: IVec3, budget: usize) -> Vec<Island> {
    let lo = carve_lo.min(carve_hi) - IVec3::ONE;
    let hi = carve_lo.max(carve_hi) + IVec3::ONE;

    let mut fl = Flood::new(grid);
    let mut islands = Vec::new();
    let mut traversal: u32 = 0;

    for y in lo.y..=hi.y {
        for x in lo.x..=hi.x {
            let (start, len) = fl.column_range(x, y);
            for i in start..start + len {
                let seed = fl.runs[i as usize];
                // Seed only runs intersecting the padded z window.
                if seed.bot <= lo.z || seed.top > hi.z {
                    continue;
                }
                if fl.visited[i as usize] != 0 {
                    continue;
                }
                traversal += 1;
                fl.visited[i as usize] = traversal;
                if let Some(comp) = flood_component(&mut fl, (x, y), seed, budget, traversal) {
                    islands.push(build_island(grid, &comp));
                }
            }
        }
    }
    islands
}

/// Flood one component from `seed`; `Some(runs)` iff it is a genuine
/// island (exhausted under budget, no bedrock anchor, no merge into an
/// earlier — necessarily supported — traversal's territory).
fn flood_component(
    fl: &mut Flood<'_>,
    seed_col: (i32, i32),
    seed: Run,
    budget: usize,
    traversal: u32,
) -> Option<Vec<(i32, i32, Run)>> {
    let mut queue: VecDeque<(i32, i32, Run)> = VecDeque::new();
    queue.push_back((seed_col.0, seed_col.1, seed));

    let mut comp: Vec<(i32, i32, Run)> = Vec::new();
    let mut count = 0usize;

    while let Some((cx, cy, r)) = queue.pop_front() {
        if r.anchored {
            return None; // bedrock support
        }
        count += usize::try_from(r.bot - r.top).unwrap_or(usize::MAX);
        if count > budget {
            return None; // too big to crumble — assume supported
        }
        comp.push((cx, cy, r));
        for (nx, ny) in [(cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)] {
            let (start, len) = fl.column_range(nx, ny);
            for i in (start as usize)..(start + len) as usize {
                let nr = fl.runs[i];
                if nr.top >= r.bot || nr.bot <= r.top {
                    continue; // no z overlap ⇒ not 6-adjacent
                }
                let v = fl.visited[i];
                if v == traversal {
                    continue;
                }
                if v != 0 {
                    // A different traversal reached this run first.
                    // Completed islands are closed components (nothing
                    // outside them touches them), so this can only be
                    // the territory of a supported-aborted flood ⇒ we
                    // are the same component ⇒ supported.
                    return None;
                }
                fl.visited[i] = traversal;
                queue.push_back((nx, ny, nr));
            }
        }
    }
    Some(comp)
}

/// Materialise a detected component into an [`Island`]: enumerate its
/// voxels with colours (untextured interior cells inherit the nearest
/// colour above in the run — the `.vxl` format only stores surface
/// colours) and fold the inclusive bbox.
fn build_island(grid: &Grid, comp: &[(i32, i32, Run)]) -> Island {
    let mut voxels = Vec::new();
    let mut lo = IVec3::MAX;
    let mut hi = IVec3::MIN;
    for &(x, y, r) in comp {
        let mut last = VoxColor(0x8080_8080);
        for z in r.top..r.bot {
            let v = IVec3::new(x, y, z);
            let c = grid.voxel_color(v).unwrap_or(last);
            last = c;
            voxels.push((v, c));
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    Island {
        voxels,
        bbox: (lo, hi),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GridTransform;
    use roxlap_formats::color::VoxColor;

    const STONE: VoxColor = VoxColor(0x80B0_8040);

    fn grid() -> Grid {
        Grid::new(GridTransform::identity())
    }

    /// Vertical pillar `[z0, 255]` at `(x, y)` — merged with the
    /// bedrock run, i.e. genuinely supported.
    fn pillar(g: &mut Grid, x: i32, y: i32, z0: i32) {
        g.set_rect(IVec3::new(x, y, z0), IVec3::new(x, y, 255), Some(STONE));
    }

    fn island_positions(i: &Island) -> Vec<IVec3> {
        let mut v: Vec<IVec3> = i.voxels.iter().map(|&(p, _)| p).collect();
        v.sort_by_key(|p| (p.z, p.y, p.x));
        v
    }

    /// Cutting a horizontal beam one voxel from its pillar detaches
    /// exactly the tip — voxels, colours and bbox all exact.
    #[test]
    fn beam_cut_detaches_tip() {
        let mut g = grid();
        pillar(&mut g, 2, 2, 100);
        // Beam sticking out along +x at z=100: x ∈ [3, 6].
        g.set_rect(IVec3::new(3, 2, 100), IVec3::new(6, 2, 100), Some(STONE));

        // No cut yet: everything hangs off the pillar.
        assert!(detect_islands(&g, IVec3::new(3, 2, 100), IVec3::new(3, 2, 100), 4096).is_empty());

        // Cut at x=3: the tip x ∈ [4, 6] is an island.
        g.set_voxel(IVec3::new(3, 2, 100), None);
        let islands = detect_islands(&g, IVec3::new(3, 2, 100), IVec3::new(3, 2, 100), 4096);
        assert_eq!(islands.len(), 1);
        let isl = &islands[0];
        assert_eq!(
            island_positions(isl),
            vec![
                IVec3::new(4, 2, 100),
                IVec3::new(5, 2, 100),
                IVec3::new(6, 2, 100)
            ]
        );
        assert!(isl.voxels.iter().all(|&(_, c)| c == STONE));
        assert_eq!(isl.bbox, (IVec3::new(4, 2, 100), IVec3::new(6, 2, 100)));
    }

    /// An arch: cutting one leg leaves the beam supported by the
    /// other; cutting both detaches it.
    #[test]
    fn arch_needs_both_legs_cut() {
        let mut g = grid();
        pillar(&mut g, 2, 5, 101);
        pillar(&mut g, 8, 5, 101);
        g.set_rect(IVec3::new(2, 5, 100), IVec3::new(8, 5, 100), Some(STONE));

        // Sever the left leg below the beam.
        g.set_voxel(IVec3::new(2, 5, 101), None);
        assert!(
            detect_islands(&g, IVec3::new(2, 5, 101), IVec3::new(2, 5, 101), 4096).is_empty(),
            "beam still hangs off the right leg"
        );

        // Sever the right leg too: the whole beam comes down.
        g.set_voxel(IVec3::new(8, 5, 101), None);
        let islands = detect_islands(&g, IVec3::new(8, 5, 101), IVec3::new(8, 5, 101), 4096);
        assert_eq!(islands.len(), 1);
        assert_eq!(islands[0].voxels.len(), 7, "beam x ∈ [2, 8] at z=100");
    }

    /// An island whose body crosses the x = 127/128 chunk border is
    /// found whole — the flood walks the chunk HashMap, not one Vxl.
    #[test]
    fn island_crosses_chunk_border() {
        let mut g = grid();
        pillar(&mut g, 133, 5, 101);
        g.set_rect(
            IVec3::new(124, 5, 100),
            IVec3::new(133, 5, 100),
            Some(STONE),
        );

        g.set_voxel(IVec3::new(132, 5, 100), None);
        let islands = detect_islands(&g, IVec3::new(132, 5, 100), IVec3::new(132, 5, 100), 4096);
        assert_eq!(islands.len(), 1);
        assert_eq!(islands[0].voxels.len(), 8, "x ∈ [124, 131] at z=100");
        let (lo, hi) = islands[0].bbox;
        assert!(lo.x < 128 && hi.x >= 128, "bbox spans the border");
    }

    /// A detached plate bigger than the budget is treated as supported
    /// (stays floating, by design); a raised budget finds it.
    #[test]
    fn budget_exceeded_stays_supported() {
        let mut g = grid();
        pillar(&mut g, 10, 10, 101);
        g.set_rect(IVec3::new(1, 1, 100), IVec3::new(20, 20, 100), Some(STONE));

        g.set_voxel(IVec3::new(10, 10, 101), None);
        let cut = (IVec3::new(10, 10, 101), IVec3::new(10, 10, 101));
        assert!(
            detect_islands(&g, cut.0, cut.1, 100).is_empty(),
            "400-voxel plate exceeds a 100-voxel budget"
        );
        let islands = detect_islands(&g, cut.0, cut.1, 10_000);
        assert_eq!(islands.len(), 1);
        assert_eq!(islands[0].voxels.len(), 400);
    }

    /// Carving where nothing was solid detects nothing.
    #[test]
    fn carve_in_air_finds_nothing() {
        let mut g = grid();
        pillar(&mut g, 2, 2, 100);
        assert!(
            detect_islands(&g, IVec3::new(50, 50, 50), IVec3::new(52, 52, 52), 4096).is_empty()
        );
    }

    /// Two separate islands from one carve come back as two islands:
    /// both beams hang on a single keystone voxel atop the pillar;
    /// carving it detaches each side independently.
    #[test]
    fn two_islands_from_one_carve() {
        let mut g = grid();
        pillar(&mut g, 5, 5, 101);
        // The keystone both beams connect through, sitting on the pillar.
        g.set_voxel(IVec3::new(5, 5, 100), Some(STONE));
        // Two beams off the keystone, along +x and −x.
        g.set_rect(IVec3::new(6, 5, 100), IVec3::new(8, 5, 100), Some(STONE));
        g.set_rect(IVec3::new(2, 5, 100), IVec3::new(4, 5, 100), Some(STONE));

        // With the keystone in place everything is supported.
        assert!(detect_islands(&g, IVec3::new(5, 5, 100), IVec3::new(5, 5, 100), 4096).is_empty());

        // Carve the keystone: each beam floats on its own.
        g.set_voxel(IVec3::new(5, 5, 100), None);
        let islands = detect_islands(&g, IVec3::new(5, 5, 100), IVec3::new(5, 5, 100), 4096);
        assert_eq!(islands.len(), 2);
        let mut sizes: Vec<usize> = islands.iter().map(|i| i.voxels.len()).collect();
        sizes.sort_unstable();
        assert_eq!(sizes, vec![3, 3]);
    }

    /// A pillar through the chz = 0/1 chunk border: every chunk's
    /// local z=255 is format-pinned placeholder bedrock (`delslab`
    /// clamps below `MAXZDIM`, so it is uncarvable — a component
    /// containing it can never fall). A cut below the border must
    /// detach exactly the segment under the cut — not report the
    /// pinned upper segment, and not flood the upper chunk's 128×128
    /// bedrock sheet into a phantom mega-island.
    #[test]
    fn stacked_chz_bedrock_is_anchored() {
        let mut g = grid();
        // Column (5,5), z ∈ [200, 400] — crosses the border at z=256
        // and merges with chunk 0's pinned bedrock at z=255 on the way.
        g.set_rect(IVec3::new(5, 5, 200), IVec3::new(5, 5, 400), Some(STONE));

        g.set_voxel(IVec3::new(5, 5, 300), None);
        let islands = detect_islands(&g, IVec3::new(5, 5, 300), IVec3::new(5, 5, 300), 100_000);
        assert_eq!(islands.len(), 1, "only the under-cut segment falls");
        assert_eq!(islands[0].voxels.len(), 100, "z ∈ [301, 400]");
        assert_eq!(
            islands[0].bbox,
            (IVec3::new(5, 5, 301), IVec3::new(5, 5, 400))
        );
    }

    /// Manual perf probe (house-style, `--ignored --nocapture`): the
    /// entry-doc gates are supported-exit (budget path) < 0.5 ms and
    /// detach cost ∝ island size.
    #[test]
    #[ignore = "manual perf probe — cargo test -p roxlap-scene --lib islands -- --ignored --nocapture"]
    fn perf_probe() {
        // Worst case: a carve into a huge anchored plate — the flood
        // must burn through DEFAULT_ISLAND_BUDGET and exit.
        let mut g = grid();
        for (x, y) in [(1, 1), (98, 1), (1, 98), (98, 98)] {
            pillar(&mut g, x, y, 101);
        }
        g.set_rect(IVec3::new(0, 0, 100), IVec3::new(99, 99, 100), Some(STONE));
        g.set_sphere(IVec3::new(50, 50, 100), 4, None);
        let t = std::time::Instant::now();
        let n = detect_islands(
            &g,
            IVec3::new(46, 46, 96),
            IVec3::new(54, 54, 104),
            DEFAULT_ISLAND_BUDGET,
        )
        .len();
        let supported_exit = t.elapsed();
        assert_eq!(n, 0);

        // Typical detach: a 64-voxel beam cut at its root.
        let mut g = grid();
        pillar(&mut g, 2, 2, 100);
        g.set_rect(IVec3::new(3, 2, 100), IVec3::new(66, 2, 100), Some(STONE));
        g.set_voxel(IVec3::new(3, 2, 100), None);
        let t = std::time::Instant::now();
        let islands = detect_islands(
            &g,
            IVec3::new(3, 2, 100),
            IVec3::new(3, 2, 100),
            DEFAULT_ISLAND_BUDGET,
        );
        let detach = t.elapsed();
        assert_eq!(islands[0].voxels.len(), 63);

        eprintln!("supported-exit (budget {DEFAULT_ISLAND_BUDGET}): {supported_exit:?}");
        eprintln!("63-voxel beam detach: {detach:?}");
    }

    // ---------- DT.1: extraction → sprite ----------

    /// Extraction carves exactly the island's voxels (support stays),
    /// bumps the chunk version, and a re-run of detection over the
    /// same carve region finds nothing.
    #[test]
    fn extract_leaves_air_and_detection_clean() {
        let mut g = grid();
        pillar(&mut g, 2, 2, 100);
        g.set_rect(IVec3::new(3, 2, 100), IVec3::new(6, 2, 100), Some(STONE));
        g.set_voxel(IVec3::new(3, 2, 100), None);
        let cut = (IVec3::new(3, 2, 100), IVec3::new(3, 2, 100));
        let islands = detect_islands(&g, cut.0, cut.1, 4096);
        let isl = islands[0].clone();

        let v_before = g.chunk_version(IVec3::ZERO);
        isl.extract(&mut g, BakeMode::Directional);

        for &(v, _) in &isl.voxels {
            assert!(!g.voxel_solid(v), "extracted voxel {v} must be air");
        }
        assert!(
            g.voxel_solid(IVec3::new(2, 2, 100)),
            "the supported pillar stays"
        );
        assert!(
            detect_islands(&g, cut.0, cut.1, 4096).is_empty(),
            "nothing left to detach"
        );
        assert!(
            g.chunk_version(IVec3::ZERO) > v_before,
            "renderers must see the extraction"
        );
    }

    /// Extraction of a vertical run (one column, many voxels)
    /// coalesces into span carves and clears the whole segment —
    /// the stacked-chz island from `stacked_chz_bedrock_is_anchored`.
    #[test]
    fn extract_vertical_run_across_chz() {
        let mut g = grid();
        g.set_rect(IVec3::new(5, 5, 200), IVec3::new(5, 5, 400), Some(STONE));
        g.set_voxel(IVec3::new(5, 5, 300), None);
        let islands = detect_islands(&g, IVec3::new(5, 5, 300), IVec3::new(5, 5, 300), 100_000);
        islands[0].extract(&mut g, BakeMode::Directional);
        for z in 301..=400 {
            assert!(!g.voxel_solid(IVec3::new(5, 5, z)), "z={z} must be air");
        }
        assert!(
            g.voxel_solid(IVec3::new(5, 5, 299)),
            "the pinned upper segment stays"
        );
    }

    /// The KV6 model matches the island: dims = bbox extent, one
    /// (surface) voxel per island voxel for a thin beam, colours
    /// normalised to full-bright `0x80RRGGBB` — including a voxel
    /// whose grid colour carries a dim baked brightness byte.
    #[test]
    fn to_kv6_matches_bbox_and_colours() {
        const DIM_STONE: VoxColor = VoxColor(0x40B0_8040); // baked-dark byte
        let mut g = grid();
        pillar(&mut g, 2, 2, 100);
        // Per-voxel inserts into air (an insert over an existing solid
        // voxel does not repaint): x=5 carries the dim brightness byte.
        for x in [3, 4, 6] {
            g.set_voxel(IVec3::new(x, 2, 100), Some(STONE));
        }
        g.set_voxel(IVec3::new(5, 2, 100), Some(DIM_STONE));
        g.set_voxel(IVec3::new(3, 2, 100), None);
        let islands = detect_islands(&g, IVec3::new(3, 2, 100), IVec3::new(3, 2, 100), 4096);
        let isl = &islands[0];
        assert!(
            isl.voxels
                .iter()
                .any(|&(v, c)| v == IVec3::new(5, 2, 100) && c == DIM_STONE),
            "the island records the raw (dim) grid colour"
        );
        let kv6 = isl.to_kv6();
        assert_eq!((kv6.xsiz, kv6.ysiz, kv6.zsiz), (3, 1, 1));
        assert_eq!(kv6.voxels.len(), 3, "thin beam: every voxel is surface");
        assert!(
            kv6.voxels.iter().all(|v| v.col == STONE.0),
            "the model normalises every brightness byte to 0x80"
        );
    }

    /// A hand-constructed empty island (detection never yields one)
    /// is a no-op: no edits, no bake over the degenerate default bbox.
    #[test]
    fn extract_empty_island_is_noop() {
        let mut g = grid();
        pillar(&mut g, 2, 2, 100);
        let v = g.chunk_version(IVec3::ZERO);
        Island {
            voxels: Vec::new(),
            bbox: (IVec3::MAX, IVec3::MIN),
        }
        .extract(&mut g, BakeMode::Directional);
        assert_eq!(
            g.chunk_version(IVec3::ZERO),
            v,
            "no-op leaves versions alone"
        );
    }

    /// World anchors honour the grid transform including
    /// `voxel_world_size` (SC): the origin is the bbox-min corner, the
    /// pivot the bbox centre (where `to_kv6`'s pivot sits).
    #[test]
    fn world_anchors_honour_scale() {
        let isl = Island {
            voxels: Vec::new(),
            bbox: (IVec3::new(4, 2, 100), IVec3::new(6, 2, 100)),
        };
        let t = GridTransform::at_scale(DVec3::new(10.0, 20.0, 30.0), 2.0);
        assert_eq!(isl.world_origin(&t), DVec3::new(18.0, 24.0, 230.0));
        assert_eq!(isl.world_pivot(&t), DVec3::new(21.0, 25.0, 231.0));
    }

    // ---------- DT.5: fracture partitioners ----------

    /// A hand-built solid box island for the partition tests.
    fn box_island(dims: IVec3) -> Island {
        let mut voxels = Vec::new();
        for z in 0..dims.z {
            for y in 0..dims.y {
                for x in 0..dims.x {
                    voxels.push((IVec3::new(x, y, z), STONE));
                }
            }
        }
        Island {
            voxels,
            bbox: (IVec3::ZERO, dims - IVec3::ONE),
        }
    }

    /// The union of the fragments is exactly the original voxel set,
    /// pairwise disjoint, with correct per-fragment bboxes.
    fn assert_disjoint_cover(original: &Island, frags: &[Island]) {
        let mut seen = std::collections::HashSet::new();
        for f in frags {
            let (mut lo, mut hi) = (IVec3::MAX, IVec3::MIN);
            for &(v, _) in &f.voxels {
                assert!(seen.insert(v), "voxel {v} appears in two fragments");
                lo = lo.min(v);
                hi = hi.max(v);
            }
            assert_eq!(f.bbox, (lo, hi), "fragment bbox recomputed");
        }
        assert_eq!(
            seen.len(),
            original.voxels.len(),
            "fragments cover every original voxel"
        );
        for &(v, _) in &original.voxels {
            assert!(seen.contains(&v));
        }
    }

    /// `Chunks` yields several rounded cells forming a disjoint cover,
    /// bit-identically for the same seed and differently-grouped for
    /// another.
    #[test]
    fn chunks_split_is_disjoint_cover_and_deterministic() {
        let isl = box_island(IVec3::new(12, 12, 6));
        let frags = isl.split(FracturePattern::Chunks { cell: 4 }, 7);
        assert!(frags.len() > 3, "a 12×12×6 box breaks into several cells");
        assert_disjoint_cover(&isl, &frags);

        let again = isl.split(FracturePattern::Chunks { cell: 4 }, 7);
        assert_eq!(frags.len(), again.len());
        for (a, b) in frags.iter().zip(&again) {
            assert_eq!(a.voxels, b.voxels, "same seed ⇒ bit-identical split");
        }
    }

    /// `Shards` slices into the requested number of roughly-equal
    /// plates; each plate is thin along some direction (the slicing
    /// normal — found here by sampling), unlike the island itself.
    #[test]
    fn shards_split_into_planar_plates() {
        let isl = box_island(IVec3::new(12, 12, 12));
        let frags = isl.split(FracturePattern::Shards { plates: 3 }, 11);
        assert_eq!(frags.len(), 3, "three plates from a solid cube");
        assert_disjoint_cover(&isl, &frags);
        let total = isl.voxels.len() as f64;
        for f in &frags {
            let share = f.voxels.len() as f64 / total;
            assert!(
                (0.15..=0.55).contains(&share),
                "roughly equal plates (share {share:.2})"
            );
            // Planarity metric: over sampled directions, the plate's
            // thinnest extent is well under the cube's 12-voxel edge —
            // a compact (non-planar) third of the cube could not be.
            let mut best = f64::MAX;
            let mut rng = SplitMix(99);
            for _ in 0..256 {
                let z = 2.0 * rng.unit() - 1.0;
                let phi = std::f64::consts::TAU * rng.unit();
                let r = (1.0 - z * z).max(0.0).sqrt();
                let d = glam::DVec3::new(r * phi.cos(), r * phi.sin(), z);
                let (mut pmin, mut pmax) = (f64::MAX, f64::MIN);
                for &(v, _) in &f.voxels {
                    let p = v.as_dvec3().dot(d);
                    pmin = pmin.min(p);
                    pmax = pmax.max(p);
                }
                best = best.min(pmax - pmin);
            }
            assert!(
                best < 7.0,
                "each plate is thin along the slicing normal (got {best:.2})"
            );
        }
    }

    /// A long sparse diagonal island (huge bbox, few voxels) splits
    /// in time and memory proportional to the ISLAND, not its bbox —
    /// the bbox-grid site placement this replaces would have built
    /// millions of sites here (maintainer review, DT.5).
    #[test]
    fn chunks_split_scales_with_island_not_bbox() {
        let n = 200;
        let voxels: Vec<(IVec3, VoxColor)> = (0..n)
            .map(|i| (IVec3::new(i, i, i.rem_euclid(256)), STONE))
            .collect();
        let isl = Island {
            bbox: (IVec3::ZERO, IVec3::new(n - 1, n - 1, 199)),
            voxels,
        };
        let frags = isl.split(FracturePattern::Chunks { cell: 2 }, 3);
        assert!(frags.len() > 10, "a diagonal snake breaks into many bits");
        assert_disjoint_cover(&isl, &frags);
    }

    /// `Whole` and degenerate inputs pass through unchanged.
    #[test]
    fn whole_and_degenerate_splits_pass_through() {
        let isl = box_island(IVec3::new(3, 2, 1));
        let whole = isl.split(FracturePattern::Whole, 1);
        assert_eq!(whole.len(), 1);
        assert_eq!(whole[0].voxels, isl.voxels);
        // A cell far larger than the island ⇒ one fragment.
        let coarse = isl.split(FracturePattern::Chunks { cell: 64 }, 1);
        assert_eq!(coarse.len(), 1);
        assert_disjoint_cover(&isl, &coarse);
    }

    /// Property test: on a randomised scene, `detect_islands` over the
    /// whole region agrees exactly with a naive dense oracle (flood
    /// from the bedrock plane; solid voxels not reached = islands).
    #[test]
    fn matches_dense_oracle() {
        // Deterministic xorshift.
        let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut rnd = move |m: i32| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            #[allow(clippy::cast_possible_truncation)]
            ((s >> 33) as i32).rem_euclid(m)
        };

        let (nx, ny) = (24, 24);
        let mut g = grid();
        // A few anchored pillars…
        for _ in 0..4 {
            pillar(&mut g, rnd(nx), rnd(ny), 200);
        }
        // …and a scatter of small blobs, some touching pillars, some
        // not — clamped inside the region so the dense oracle sees the
        // whole world it reasons about.
        for _ in 0..40 {
            let p = IVec3::new(rnd(nx), rnd(ny), 190 + rnd(30));
            let q = (p + IVec3::new(rnd(3), rnd(3), rnd(3))).min(IVec3::new(nx - 1, ny - 1, 255));
            g.set_rect(p, q, Some(STONE));
        }
        // One carve through the middle.
        g.set_sphere(IVec3::new(nx / 2, ny / 2, 205), 6, None);

        // Oracle: 6-conn flood from every SOLID voxel at z=255 (a
        // final run reaching the bottom is anchored). CT.2 — columns
        // no fill touched are the empty sentinel now, so z=255 is no
        // longer a universal bedrock plane: seed only where solid.
        let solid = |v: IVec3| v.x >= 0 && v.x < nx && v.y >= 0 && v.y < ny && g.voxel_solid(v);
        let mut reached = std::collections::HashSet::new();
        let mut q = VecDeque::new();
        for x in 0..nx {
            for y in 0..ny {
                let v = IVec3::new(x, y, 255);
                if g.voxel_solid(v) {
                    reached.insert(v);
                    q.push_back(v);
                }
            }
        }
        while let Some(v) = q.pop_front() {
            for d in [
                IVec3::X,
                IVec3::NEG_X,
                IVec3::Y,
                IVec3::NEG_Y,
                IVec3::Z,
                IVec3::NEG_Z,
            ] {
                let n = v + d;
                if n.z >= 0 && n.z < 256 && solid(n) && reached.insert(n) {
                    q.push_back(n);
                }
            }
        }
        let mut oracle: Vec<IVec3> = Vec::new();
        for x in 0..nx {
            for y in 0..ny {
                for z in 0..256 {
                    let v = IVec3::new(x, y, z);
                    if solid(v) && !reached.contains(&v) {
                        oracle.push(v);
                    }
                }
            }
        }
        oracle.sort_by_key(|p| (p.z, p.y, p.x));

        // detect_islands seeded over the whole region.
        let islands = detect_islands(
            &g,
            IVec3::new(0, 0, 0),
            IVec3::new(nx - 1, ny - 1, 255),
            1_000_000,
        );
        let mut got: Vec<IVec3> = islands
            .iter()
            .flat_map(|i| i.voxels.iter().map(|&(p, _)| p))
            .collect();
        got.sort_by_key(|p| (p.z, p.y, p.x));
        assert_eq!(got, oracle, "span-BFS must agree with the dense oracle");
    }
}
