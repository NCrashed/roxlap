//! DT.0 — floating-island detection, the destruction stage's core
//! (see `docs/porting/PORTING-DESTRUCTION.md`).
//!
//! After a carve, voxel regions that no longer connect to support must
//! crumble. [`detect_islands`] finds them with a budgeted breadth-first
//! flood over the RLE **runs** of the affected columns — a run is one
//! BFS node (chunk-format-native, no dense decode), 6-connected,
//! seeded from every run touching the carved bbox (±1 pad). A
//! component that reaches a **bedrock-anchored** run (the final run of
//! a column in a chunk with no chunk below it — the `.vxl` format
//! extends it to the column bottom) or grows past `budget` voxels is
//! supported and abandoned early; only components that exhaust under
//! budget without finding support come back as [`Island`]s.
//!
//! Cost per call ≈ O(Σ min(component size, budget)) over the distinct
//! components touching the bbox. False negatives are by design: a
//! detached region larger than `budget` stays floating (voxlap behaved
//! the same way) — raise the budget when a host wants bigger crumbles.

use std::collections::{HashMap, VecDeque};

use glam::IVec3;
use roxlap_formats::color::VoxColor;
use roxlap_formats::vxl::Vxl;

use crate::{Grid, CHUNK_SIZE_XY, CHUNK_SIZE_Z};

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
    /// at a chunk border merge; the final run of a chunk with **no
    /// chunk below it** is the bedrock anchor (support). Split field
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
        for (i, &chz) in stack.iter().enumerate() {
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
            let below_present = stack.get(i + 1) == Some(&(chz + 1));
            chunk_column_runs(vxl, lx, ly, |top, bot, is_final| {
                let (top_g, bot_g) = (base + top, base + bot);
                let anchored = is_final && !below_present;
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

    /// Two separate islands from one carve come back as two islands.
    #[test]
    fn two_islands_from_one_carve() {
        let mut g = grid();
        pillar(&mut g, 5, 5, 101);
        // Two beams off the same pillar, along +x and −x.
        g.set_rect(IVec3::new(6, 5, 100), IVec3::new(8, 5, 100), Some(STONE));
        g.set_rect(IVec3::new(2, 5, 100), IVec3::new(4, 5, 100), Some(STONE));
        // The pillar top holds both; carve it out.
        g.set_voxel(IVec3::new(5, 5, 100), None);
        g.set_voxel(IVec3::new(5, 5, 101), None);
        // (both beams hung on (5,5,100); with it gone each side floats)
        let islands = detect_islands(&g, IVec3::new(5, 5, 100), IVec3::new(5, 5, 101), 4096);
        assert_eq!(islands.len(), 2);
        let mut sizes: Vec<usize> = islands.iter().map(|i| i.voxels.len()).collect();
        sizes.sort_unstable();
        assert_eq!(sizes, vec![3, 3]);
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

        // Oracle: 6-conn flood from every solid voxel at z=255 (the
        // bedrock plane all final runs share), dense over the region.
        let solid = |v: IVec3| v.x >= 0 && v.x < nx && v.y >= 0 && v.y < ny && g.voxel_solid(v);
        let mut reached = std::collections::HashSet::new();
        let mut q = VecDeque::new();
        for x in 0..nx {
            for y in 0..ny {
                let v = IVec3::new(x, y, 255);
                assert!(g.voxel_solid(v), "bedrock plane is always solid");
                reached.insert(v);
                q.push_back(v);
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
