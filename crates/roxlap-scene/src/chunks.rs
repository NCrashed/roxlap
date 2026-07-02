//! Sparse chunk storage helpers.
//!
//! A grid's [`Grid::chunks`] map holds populated chunks keyed by
//! their `(chx, chy, chz)` index. A missing entry is an implicit
//! all-air chunk; this module provides the constructor for fresh
//! all-air chunks plus the `chunk` / `chunk_mut` / `ensure_chunk`
//! lookup API.
//!
//! [`Grid::chunks`]: crate::Grid::chunks

use glam::IVec3;
use roxlap_formats::edit::{set_spans, Vspan};
use roxlap_formats::vxl::Vxl;

use crate::{Grid, CHUNK_SIZE_XY, CHUNK_SIZE_Z};

/// Bytes of edit-pool headroom reserved per chunk on creation.
/// 256 bytes/column × 128² columns ≈ 4 MiB; a generous budget for
/// runtime edits within a single chunk before [`voxalloc`] starts
/// returning out-of-space. Tunable later if memory becomes an
/// issue.
///
/// [`voxalloc`]: roxlap_formats::vxl::Vxl::voxalloc
const CHUNK_EDIT_HEADROOM_PER_COLUMN: usize = 256;

/// Construct a fresh all-air [`Vxl`] sized for one chunk
/// (`vsid = CHUNK_SIZE_XY`).
///
/// Strategy mirrors `roxlap_cavegen::pack_dense_grid_to_vxl`: seed
/// each column with one solid voxel at z=0 + implicit-solid below
/// (the voxlap "loadnul" shape), then carve the entire z range to
/// air via [`set_spans`]. Finishes with [`Vxl::reserve_edit_capacity`]
/// so subsequent runtime edits don't need a separate upgrade pass.
///
/// This is the canonical empty-chunk constructor — every code
/// path that materialises a sparse chunk goes through it (see
/// [`Grid::ensure_chunk`]).
pub(crate) fn empty_chunk_vxl() -> Vxl {
    let vsid = CHUNK_SIZE_XY;
    let n_cols = (vsid as usize) * (vsid as usize);

    // 1. Seed: every column = 4-byte slab header + 1 colour. Colour
    //    is irrelevant — the whole column gets carved below.
    let mut data: Vec<u8> = Vec::with_capacity(n_cols * 8);
    let mut column_offset: Vec<u32> = Vec::with_capacity(n_cols + 1);
    for _ in 0..n_cols {
        column_offset.push(u32::try_from(data.len()).expect("offset fits in u32"));
        data.extend_from_slice(&[0, 0, 0, 0]); // header
        data.extend_from_slice(&[0, 0, 0, 0]); // 1 placeholder colour
    }
    column_offset.push(u32::try_from(data.len()).expect("offset fits in u32"));

    let mut vxl = Vxl {
        vsid,
        // Per-grid placement lives on `GridTransform`; the per-chunk
        // Vxl's intrinsic camera fields are unused at this layer.
        ipo: [0.0; 3],
        ist: [1.0, 0.0, 0.0],
        ihe: [0.0, 0.0, 1.0],
        ifo: [0.0, 1.0, 0.0],
        data: data.into_boxed_slice(),
        column_offset: column_offset.into_boxed_slice(),
        mip_base_offsets: Box::new([0, n_cols + 1]),
        vbit: Box::new([]),
        vbiti: 0,
    };
    vxl.reserve_edit_capacity(n_cols * CHUNK_EDIT_HEADROOM_PER_COLUMN);

    // 2. Carve [0, 255] in every column to make it all-air.
    //    `Vspan.z1` is inclusive per voxlap's vspans convention.
    let mut spans: Vec<Vspan> = Vec::with_capacity(n_cols);
    for y in 0..vsid {
        for x in 0..vsid {
            spans.push(Vspan {
                x,
                y,
                z0: 0,
                z1: u8::MAX,
            });
        }
    }
    set_spans(&mut vxl, &spans, None);

    vxl
}

/// True if voxel `(x, y, z)` is solid within one chunk's [`Vxl`] —
/// i.e. covered by a solid run in column `(x, y)`. `(x, y)` are
/// `< CHUNK_SIZE_XY`, `z < CHUNK_SIZE_Z`.
///
/// PF.6 — walks the slab chain in place with an early-out at `z`,
/// mirroring `expandrle`'s run derivation ONE RUN AT A TIME instead of
/// heap-allocating a 516-int buffer and decoding the whole column per
/// query (this sits on every CPU shadow-ray step and every
/// `Scene::raycast` step). Run k spans `[top_k, bot_k)` where `top_0 =
/// slab[1]`, each following non-degenerate slab header closes the
/// previous run at `slab[v+3]` and opens the next at `slab[v+1]`, and
/// the final run extends to the column bottom — exactly the list
/// `expandrle` would emit.
#[allow(clippy::cast_possible_wrap)]
pub(crate) fn vxl_voxel_solid(vxl: &Vxl, x: u32, y: u32, z: u32) -> bool {
    let idx = (y * vxl.vsid + x) as usize;
    let slab = vxl.column_data(idx);
    let z = z as i32;
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
        if z < bot {
            // z is above this run's bottom: solid iff inside the run
            // (below `top` would be the preceding air gap).
            return z >= top;
        }
        top = i32::from(slab[v + 1]);
    }
    // Last run extends to the column bottom (bedrock).
    z >= top
}

/// PF.6 — chunk-cached solid sampler for DDA marches. Consecutive steps
/// almost always stay inside one chunk; caching the last `chunks`
/// HashMap probe turns the per-step cost into one compare + the in-place
/// slab walk. `chunk_at` exposes presence so callers can skip absent
/// (all-air) chunks wholesale.
pub(crate) struct SolidSampler<'a> {
    grid: &'a Grid,
    cached_idx: IVec3,
    cached: Option<&'a Vxl>,
    primed: bool,
}

impl<'a> SolidSampler<'a> {
    /// The chunk holding grid-local voxel-space chunk index `chunk_idx`,
    /// through the one-entry cache.
    pub(crate) fn chunk_at(&mut self, chunk_idx: IVec3) -> Option<&'a Vxl> {
        if !self.primed || chunk_idx != self.cached_idx {
            self.cached = self.grid.chunk(chunk_idx);
            self.cached_idx = chunk_idx;
            self.primed = true;
        }
        self.cached
    }

    /// [`Grid::voxel_solid`] through the cache.
    pub(crate) fn solid(&mut self, voxel: IVec3) -> bool {
        let (chunk_idx, in_chunk) = crate::voxel_split(voxel);
        match self.chunk_at(chunk_idx) {
            Some(vxl) => vxl_voxel_solid(vxl, in_chunk.x, in_chunk.y, in_chunk.z),
            None => false,
        }
    }
}

impl Grid {
    /// True if the grid-local integer voxel `voxel` is solid (inside a
    /// solid run of its chunk). An implicit-air or absent chunk reads
    /// as `false`. `voxel` is a grid-local voxel coordinate
    /// (pre-transform) — get one from a world point via
    /// [`crate::world_to_grid_local`] + [`crate::voxel_global`]. Useful
    /// for picking, collision, and world queries.
    #[must_use]
    pub fn voxel_solid(&self, voxel: IVec3) -> bool {
        let (chunk_idx, in_chunk) = crate::voxel_split(voxel);
        match self.chunk(chunk_idx) {
            Some(vxl) => vxl_voxel_solid(vxl, in_chunk.x, in_chunk.y, in_chunk.z),
            None => false,
        }
    }

    /// PF.6 — a chunk-cached [`SolidSampler`] over this grid, for DDA
    /// marches that issue many [`Self::voxel_solid`]-style queries with
    /// strong chunk locality (shadow rays, `Scene::raycast`).
    pub(crate) fn solid_sampler(&self) -> SolidSampler<'_> {
        SolidSampler {
            grid: self,
            cached_idx: IVec3::ZERO,
            cached: None,
            primed: false,
        }
    }

    /// Packed BGRA colour of the textured voxel at grid-local `voxel`,
    /// or `None` for air / untextured cells. Thin wrapper over
    /// [`roxlap_formats::vxl::Vxl::voxel_color`] after the chunk split —
    /// the colour-inspection companion to [`Self::voxel_solid`]. Use it
    /// to read back what a pick / raycast hit looks like.
    #[must_use]
    pub fn voxel_color(&self, voxel: IVec3) -> Option<u32> {
        let (chunk_idx, in_chunk) = crate::voxel_split(voxel);
        self.chunk(chunk_idx)?
            .voxel_color(in_chunk.x, in_chunk.y, in_chunk.z)
    }

    /// Borrow the chunk at `chunk_idx` if it has been materialised.
    /// `None` means the chunk is implicitly all-air.
    #[must_use]
    pub fn chunk(&self, chunk_idx: IVec3) -> Option<&Vxl> {
        self.chunks.get(&chunk_idx)
    }

    /// Bake per-voxel lighting (voxlap `updatevxl`/`estnorm` shading)
    /// into every materialised chunk's brightness bytes, in place.
    /// `lightmode` is voxlap's mode (1 = directional estnorm shading,
    /// the look the cave + terrain demos use). Both the CPU rasteriser
    /// and the GPU marcher read these pre-baked brightness bytes, so
    /// call this once after building a grid and again over edited
    /// chunks after a carve (then bump their versions so the GPU
    /// re-uploads — edits already do, via [`Grid::set_voxel`] &c.).
    ///
    /// Each chunk is baked neighbour-aware in all three axes: estnorm's
    /// (and AO's) ±2-voxel padding that crosses a chunk face reads the
    /// actual neighbour chunk (when populated) — XY neighbours and, for
    /// stacked grids, the chunks above/below on `chz±1` — so brightness
    /// / occlusion is continuous across every seam. Point lights aren't
    /// applied (directional-only) — matching the demos' bake. No-op for
    /// an empty grid.
    pub fn bake_lightmode(&mut self, lightmode: u32) {
        #[allow(clippy::cast_possible_wrap)]
        let cs_xy = CHUNK_SIZE_XY as i32;
        #[allow(clippy::cast_possible_wrap)]
        let cs_z = CHUNK_SIZE_Z as i32;
        let chunk_idxs: Vec<IVec3> = self.chunks.keys().copied().collect();
        for chunk_idx in chunk_idxs {
            // Build the estnorm cache from an immutable grid borrow: the
            // reader resolves a chunk-local `(px, py)` (which may extend
            // ±ESTNORMRAD outside the target chunk) into the neighbour
            // chunk that owns that voxel column, same `chz`. Padding over
            // an unpopulated neighbour returns `None` (= treated as air).
            let cache = {
                let grid_ref: &Self = &*self;
                // `chz_delta` walks the stacked neighbour in z so the
                // ESTNORMRAD padding crossing a chunk's top/bottom face
                // reads the chunk above/below (continuous AO + estnorm
                // across a z-seam); `0` is the target layer.
                let reader = |px: i32, py: i32, chz_delta: i32| -> Option<&[u8]> {
                    let nb_chx = chunk_idx.x + px.div_euclid(cs_xy);
                    let nb_chy = chunk_idx.y + py.div_euclid(cs_xy);
                    let in_x = px.rem_euclid(cs_xy);
                    let in_y = py.rem_euclid(cs_xy);
                    let chunk =
                        grid_ref.chunk(IVec3::new(nb_chx, nb_chy, chunk_idx.z + chz_delta))?;
                    let col_idx = (in_y as u32) * CHUNK_SIZE_XY + (in_x as u32);
                    let off = chunk.column_offset[col_idx as usize] as usize;
                    Some(&chunk.data[off..])
                };
                roxlap_core::EstNormCache::build_with_reader_z(reader, 0, 0, cs_xy, cs_xy)
            };
            let target = self.chunks.get_mut(&chunk_idx).expect("populated chunk");
            roxlap_core::apply_lighting_with_cache(
                &mut target.data,
                &target.column_offset,
                CHUNK_SIZE_XY,
                0,
                0,
                0,
                cs_xy,
                cs_xy,
                cs_z,
                &cache,
                lightmode,
                &[],
                roxlap_core::AoParams::default(),
            );
        }
    }

    /// Mutably borrow a materialised chunk. Returns `None` for
    /// implicit-air chunks; use [`Grid::ensure_chunk`] when you
    /// need a `&mut Vxl` for an edit that may write voxels.
    pub fn chunk_mut(&mut self, chunk_idx: IVec3) -> Option<&mut Vxl> {
        self.chunks.get_mut(&chunk_idx)
    }

    /// Borrow `chunk_idx`'s [`Vxl`], creating an empty all-air
    /// chunk first if it doesn't exist yet. The returned `&mut`
    /// is valid for editing via [`roxlap_formats::edit`] — the new
    /// chunk has [`Vxl::reserve_edit_capacity`] already applied.
    pub fn ensure_chunk(&mut self, chunk_idx: IVec3) -> &mut Vxl {
        self.chunks.entry(chunk_idx).or_insert_with(empty_chunk_vxl)
    }

    /// Number of materialised chunks. Implicit-air chunks don't
    /// count.
    #[must_use]
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// S4B.2.c.3: build a per-chunk [`roxlap_core::GridView`] table
    /// over this grid's XY chunk footprint at `chz = 0`.
    ///
    /// Returns `None` if no chz=0 chunk is populated (the entire
    /// grid would render as implicit air anyway).
    ///
    /// Iterates `chunks` once to find the chx/chy bounding box,
    /// then a second time to fill the row-major
    /// `Vec<Option<GridView<'_>>>`. Empty XY slots (implicit-air
    /// chunks inside the box) get `None`.
    ///
    /// Pair with [`roxlap_core::ChunkGrid`] + [`roxlap_core::
    /// GridView::from_chunk_grid`] to drive the Approach B render
    /// path:
    ///
    /// ```ignore
    /// let backing = grid.chunk_xy_backing().unwrap();
    /// let cg = roxlap_core::ChunkGrid {
    ///     chunks: &backing.chunks,
    ///     origin_chunk_xy: backing.origin_chunk_xy,
    ///     chunks_x: backing.chunks_x,
    ///     chunks_y: backing.chunks_y,
    /// };
    /// let view = roxlap_core::GridView::from_chunk_grid(
    ///     &cg, crate::CHUNK_SIZE_XY,
    /// );
    /// ```
    ///
    /// Only chz=0 chunks contribute (multi-z handoff lands in
    /// S4B.3); higher-chz chunks in [`Self::chunks`] are ignored
    /// here.
    #[must_use]
    pub fn chunk_xy_backing(&self) -> Option<ChunkXyBacking<'_>> {
        let mut min_x = i32::MAX;
        let mut min_y = i32::MAX;
        let mut max_x = i32::MIN;
        let mut max_y = i32::MIN;
        let mut any = false;
        for chunk_idx in self.chunks.keys() {
            if chunk_idx.z != 0 {
                continue;
            }
            min_x = min_x.min(chunk_idx.x);
            min_y = min_y.min(chunk_idx.y);
            max_x = max_x.max(chunk_idx.x);
            max_y = max_y.max(chunk_idx.y);
            any = true;
        }
        if !any {
            return None;
        }
        #[allow(clippy::cast_sign_loss)]
        let chunks_x = (max_x - min_x + 1) as u32;
        #[allow(clippy::cast_sign_loss)]
        let chunks_y = (max_y - min_y + 1) as u32;
        let mut table: Vec<Option<roxlap_core::GridView<'_>>> =
            vec![None; (chunks_x * chunks_y) as usize];
        for (chunk_idx, vxl) in &self.chunks {
            if chunk_idx.z != 0 {
                continue;
            }
            let dx = chunk_idx.x - min_x;
            let dy = chunk_idx.y - min_y;
            #[allow(clippy::cast_sign_loss)]
            let i = (dy as u32 * chunks_x + dx as u32) as usize;
            table[i] = Some(roxlap_core::GridView::from_single_vxl(vxl));
        }
        Some(ChunkXyBacking {
            chunks: table,
            origin_chunk_xy: [min_x, min_y],
            origin_chunk_z: 0,
            chunks_x,
            chunks_y,
            chunks_z: 1,
        })
    }

    /// S4B.6.a: 3D-aware version of [`Self::chunk_xy_backing`].
    /// Enumerates ALL chunks across the chx/chy/chz bounding box
    /// (not just `chz=0`) so a stacked grid can be rendered once
    /// S4B.6.c switches the rasterizer to a chunk-z-aware column
    /// walker.
    ///
    /// Iterates `chunks` once for the XYZ bbox, then a second time
    /// to fill the row-major-per-z `Vec<Option<GridView<'_>>>`.
    /// Index layout: `[(dz * chunks_y + dy) * chunks_x + dx]` —
    /// matches [`roxlap_core::ChunkGrid`]'s indexing exactly.
    ///
    /// Returns `None` for empty grids.
    #[must_use]
    pub fn chunk_xyz_backing(&self) -> Option<ChunkXyBacking<'_>> {
        let mut min_x = i32::MAX;
        let mut min_y = i32::MAX;
        let mut min_z = i32::MAX;
        let mut max_x = i32::MIN;
        let mut max_y = i32::MIN;
        let mut max_z = i32::MIN;
        let mut any = false;
        for chunk_idx in self.chunks.keys() {
            min_x = min_x.min(chunk_idx.x);
            min_y = min_y.min(chunk_idx.y);
            min_z = min_z.min(chunk_idx.z);
            max_x = max_x.max(chunk_idx.x);
            max_y = max_y.max(chunk_idx.y);
            max_z = max_z.max(chunk_idx.z);
            any = true;
        }
        if !any {
            return None;
        }
        #[allow(clippy::cast_sign_loss)]
        let chunks_x = (max_x - min_x + 1) as u32;
        #[allow(clippy::cast_sign_loss)]
        let chunks_y = (max_y - min_y + 1) as u32;
        #[allow(clippy::cast_sign_loss)]
        let chunks_z = (max_z - min_z + 1) as u32;
        let mut table: Vec<Option<roxlap_core::GridView<'_>>> =
            vec![None; (chunks_x * chunks_y * chunks_z) as usize];
        for (chunk_idx, vxl) in &self.chunks {
            let dx = chunk_idx.x - min_x;
            let dy = chunk_idx.y - min_y;
            let dz = chunk_idx.z - min_z;
            #[allow(clippy::cast_sign_loss)]
            let (dx, dy, dz) = (dx as u32, dy as u32, dz as u32);
            let i = ((dz * chunks_y + dy) * chunks_x + dx) as usize;
            table[i] = Some(roxlap_core::GridView::from_single_vxl(vxl));
        }
        Some(ChunkXyBacking {
            chunks: table,
            origin_chunk_xy: [min_x, min_y],
            origin_chunk_z: min_z,
            chunks_x,
            chunks_y,
            chunks_z,
        })
    }
}

/// S4B.2.c.3: chx/chy chunk table built from a [`Grid`].
///
/// Owns the `Vec<Option<GridView>>` so [`roxlap_core::ChunkGrid`]
/// (which borrows the table) can live alongside the GridView
/// constructed from it. Used by the Approach B render path —
/// see [`Grid::chunk_xy_backing`].
///
/// S4B.6.a: gained `chunks_z` + `origin_chunk_z` for tall-world
/// support. Pre-S4B.6.a `chunk_xy_backing` always populates these
/// as `chunks_z=1 origin_chunk_z=0` (= chz=0 only); S4B.6.c will
/// switch the render path to `chunk_xyz_backing` once the
/// rasterizer is stack-aware.
pub struct ChunkXyBacking<'a> {
    /// Per-chunk views over the chx/chy/chz extent.
    /// Length `chunks_x * chunks_y * chunks_z`; index layout
    /// `[(dz * chunks_y + dy) * chunks_x + dx]`. `None` for
    /// implicit-air chunks inside the bbox.
    pub chunks: Vec<Option<roxlap_core::GridView<'a>>>,
    /// XY index of the chunk at `chunks[0]` — the minimum chx/chy
    /// among populated chunks at `chz = origin_chunk_z`.
    pub origin_chunk_xy: [i32; 2],
    /// Z index of the chunk at `chunks[0]`.
    pub origin_chunk_z: i32,
    /// Number of chunks along the X axis. Row stride.
    pub chunks_x: u32,
    /// Number of chunks along the Y axis.
    pub chunks_y: u32,
    /// Number of chunks along the Z axis. `1` from
    /// [`Grid::chunk_xy_backing`]; `>1` only via the S4B.6.a
    /// [`Grid::chunk_xyz_backing`].
    pub chunks_z: u32,
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::{GridTransform, CHUNK_SIZE_Z};

    /// Decode `column`'s slab bytes and return `true` iff `z` is
    /// covered by any solid run. Mirrors voxlap's column-walk
    /// semantics — the b2 buffer is `[top0, bot0, top1, bot1, ...,
    /// MAXZDIM_sentinel]`, with each `[top, bot)` pair denoting a
    /// solid range.
    pub(crate) fn voxel_is_solid(vxl: &Vxl, x: u32, y: u32, z: u32) -> bool {
        super::vxl_voxel_solid(vxl, x, y, z)
    }

    #[test]
    fn voxel_solid_reflects_set_voxel() {
        // Grid::voxel_solid (the public picking query) reads back an
        // edit: the set voxel is solid, its neighbour is air, and an
        // unmaterialised chunk reads as air.
        let mut g = Grid::new(GridTransform::identity());
        g.set_voxel(IVec3::new(5, 6, 7), Some(0x80_aa_bb_cc));
        assert!(g.voxel_solid(IVec3::new(5, 6, 7)), "set voxel is solid");
        assert!(!g.voxel_solid(IVec3::new(5, 6, 8)), "neighbour is air");
        assert!(
            !g.voxel_solid(IVec3::new(900, 900, 7)),
            "absent chunk reads as air",
        );
    }

    #[test]
    fn voxel_solid_handles_negative_coords() {
        // Negative grid-local voxels decompose via div_euclid (addr
        // semantics); the query must follow the same split.
        let mut g = Grid::new(GridTransform::identity());
        g.set_voxel(IVec3::new(-1, -1, 10), Some(0x80_11_22_33));
        assert!(g.voxel_solid(IVec3::new(-1, -1, 10)));
        assert!(!g.voxel_solid(IVec3::new(-1, -1, 11)));
    }

    #[test]
    fn empty_chunk_has_correct_vsid() {
        let vxl = empty_chunk_vxl();
        assert_eq!(vxl.vsid, CHUNK_SIZE_XY);
    }

    #[test]
    fn empty_chunk_is_all_air() {
        let vxl = empty_chunk_vxl();
        // Sample a few representative voxels — full coverage is in
        // `empty_chunk_no_voxel_solid_anywhere` below.
        for &(x, y, z) in &[
            (0u32, 0u32, 0u32),
            (0, 0, 100),
            (0, 0, 200),
            (CHUNK_SIZE_XY - 1, CHUNK_SIZE_XY - 1, 0),
            (64, 64, 128),
        ] {
            assert!(
                !voxel_is_solid(&vxl, x, y, z),
                "voxel ({x}, {y}, {z}) should be air"
            );
        }
    }

    #[test]
    fn empty_chunk_air_above_bedrock_on_grid_sample() {
        // Stride 16 across the chunk catches structural breakage
        // (a corner column wrong, a z-band wrong, etc.) without the
        // 4M-query cost of a brute-force scan in debug mode.
        // Voxlap's slab format keeps z=255 solid as the "below the
        // world" sentinel; the renderer's `treat_z_max_as_air` flag
        // handles displaying it as transparent. See
        // `project_below_bedrock_all_sky.md` for the S1.X fix.
        let vxl = empty_chunk_vxl();
        let bedrock_z = CHUNK_SIZE_Z - 1;
        for y in (0..CHUNK_SIZE_XY).step_by(16) {
            for x in (0..CHUNK_SIZE_XY).step_by(16) {
                for z in (0..bedrock_z).step_by(16) {
                    assert!(
                        !voxel_is_solid(&vxl, x, y, z),
                        "voxel ({x}, {y}, {z}) leaked solid in empty chunk"
                    );
                }
                // bedrock z is solid (placeholder).
                assert!(voxel_is_solid(&vxl, x, y, bedrock_z));
            }
        }
    }

    #[test]
    fn empty_chunk_keeps_bedrock_placeholder() {
        // Voxlap's invariant: every column carries an implicit
        // solid voxel at z = MAXZDIM-1 = 255 even after a full
        // carve. The renderer reads this as the bedrock placeholder.
        let vxl = empty_chunk_vxl();
        assert!(voxel_is_solid(&vxl, 0, 0, CHUNK_SIZE_Z - 1));
        assert!(voxel_is_solid(&vxl, 64, 64, CHUNK_SIZE_Z - 1));
    }

    #[test]
    fn ensure_chunk_creates_when_missing() {
        let mut g = Grid::new(GridTransform::identity());
        assert_eq!(g.chunk_count(), 0);
        assert!(g.chunk(IVec3::ZERO).is_none());
        let _ = g.ensure_chunk(IVec3::ZERO);
        assert_eq!(g.chunk_count(), 1);
        assert!(g.chunk(IVec3::ZERO).is_some());
    }

    #[test]
    fn ensure_chunk_returns_existing() {
        // Calling ensure_chunk a second time on the same index
        // doesn't replace the chunk. Verify by writing through the
        // first call and reading through the second.
        let mut g = Grid::new(GridTransform::identity());
        let chunk = IVec3::new(2, -1, 0);
        g.ensure_chunk(chunk);
        // Voxel local (5, 6, 7) inside chunk (2, -1, 0) is
        // grid-local global (2*128 + 5, -1*128 + 6, 0*256 + 7) =
        // (261, -122, 7).
        g.set_voxel(IVec3::new(261, -122, 7), Some(0x80_aa_bb_cc));
        let vxl = g.ensure_chunk(chunk);
        assert!(voxel_is_solid(vxl, 5, 6, 7));
        assert_eq!(g.chunk_count(), 1);
    }

    #[test]
    fn chunk_mut_returns_none_for_missing() {
        let mut g = Grid::new(GridTransform::identity());
        assert!(g.chunk_mut(IVec3::ZERO).is_none());
    }

    /// S4B.6.a: legacy `chunk_xy_backing` ignores chz!=0 chunks and
    /// always returns `chunks_z=1 origin_chunk_z=0`. Sanity-check
    /// the new field defaults so pre-S4B.6 render path stays
    /// byte-identical.
    #[test]
    fn chunk_xy_backing_returns_chunks_z_one() {
        let mut g = Grid::new(GridTransform::identity());
        g.ensure_chunk(IVec3::new(0, 0, 0));
        g.ensure_chunk(IVec3::new(1, 0, 0));
        // Add a chunk at chz=1 — chunk_xy_backing should ignore it.
        g.ensure_chunk(IVec3::new(0, 0, 1));
        let backing = g.chunk_xy_backing().expect("two chz=0 chunks present");
        assert_eq!(backing.chunks_z, 1);
        assert_eq!(backing.origin_chunk_z, 0);
        assert_eq!(backing.chunks_x, 2);
        assert_eq!(backing.chunks_y, 1);
        assert_eq!(backing.chunks.len(), 2);
    }

    /// S4B.6.a: new `chunk_xyz_backing` enumerates ALL chunks
    /// including chz!=0. Indexing layout must match
    /// `roxlap_core::ChunkGrid`: row-major per z slab.
    #[test]
    fn chunk_xyz_backing_with_stacked_chunks_enumerates_all_z() {
        let mut g = Grid::new(GridTransform::identity());
        g.ensure_chunk(IVec3::new(0, 0, 0));
        g.ensure_chunk(IVec3::new(1, 0, 0));
        g.ensure_chunk(IVec3::new(0, 0, 1));
        // Leave (1, 0, 1) implicit-air.
        let backing = g.chunk_xyz_backing().expect("at least one chunk");
        assert_eq!(backing.chunks_x, 2);
        assert_eq!(backing.chunks_y, 1);
        assert_eq!(backing.chunks_z, 2);
        assert_eq!(backing.origin_chunk_xy, [0, 0]);
        assert_eq!(backing.origin_chunk_z, 0);
        assert_eq!(backing.chunks.len(), 4); // dims [2, 1, 2]
                                             // Index layout: [(dz * chunks_y + dy) * chunks_x + dx]
                                             // (0, 0, 0) → dx=0, dy=0, dz=0 → 0
                                             // (1, 0, 0) → dx=1, dy=0, dz=0 → 1
                                             // (0, 0, 1) → dx=0, dy=0, dz=1 → 2
                                             // (1, 0, 1) → dx=1, dy=0, dz=1 → 3 (implicit-air → None)
        assert!(backing.chunks[0].is_some(), "(0, 0, 0) present");
        assert!(backing.chunks[1].is_some(), "(1, 0, 0) present");
        assert!(backing.chunks[2].is_some(), "(0, 0, 1) present");
        assert!(backing.chunks[3].is_none(), "(1, 0, 1) implicit-air");
    }

    /// S4B.6.a: chunk_xyz_backing with negative chz origin —
    /// origin_chunk_z = min_z must reflect the actual minimum chz.
    #[test]
    fn chunk_xyz_backing_with_negative_origin_chunk_z() {
        let mut g = Grid::new(GridTransform::identity());
        g.ensure_chunk(IVec3::new(0, 0, -2));
        g.ensure_chunk(IVec3::new(0, 0, 0));
        let backing = g.chunk_xyz_backing().expect("at least one chunk");
        assert_eq!(backing.chunks_z, 3); // chz range [-2, 0]
        assert_eq!(backing.origin_chunk_z, -2);
        assert!(backing.chunks[0].is_some(), "chz=-2 → dz=0");
        assert!(backing.chunks[1].is_none(), "chz=-1 → dz=1 implicit-air");
        assert!(backing.chunks[2].is_some(), "chz=0 → dz=2");
    }

    /// PF.6 — the in-place slab walk in `vxl_voxel_solid` must agree
    /// with the `expandrle` run list (the pre-PF.6 reference decoder)
    /// for every z, on columns with multiple slabs (caves), single
    /// slabs, and untouched bedrock.
    #[test]
    fn vxl_voxel_solid_matches_expandrle_reference() {
        use roxlap_formats::edit::expandrle;

        let mut g = Grid::new(GridTransform::identity());
        // Carve two separate air pockets into one column (multi-slab)
        // plus a sphere for varied neighbours.
        g.set_rect(IVec3::new(3, 4, 40), IVec3::new(4, 5, 60), None);
        g.set_rect(IVec3::new(3, 4, 100), IVec3::new(4, 5, 140), None);
        g.set_voxel(IVec3::new(3, 4, 120), Some(0x80_11_22_33)); // island inside the pocket
        g.set_sphere(IVec3::new(8, 8, 80), 6, None);
        let vxl = g.chunk(IVec3::ZERO).expect("chunk materialised");

        let maxzdim = CHUNK_SIZE_Z as i32;
        for (x, y) in [(3u32, 4u32), (8, 8), (0, 0), (8, 2), (12, 8)] {
            // Reference: full expandrle decode → run-list scan.
            let column = vxl.column_data((y * vxl.vsid + x) as usize);
            let mut b2 = vec![maxzdim; 2 * (CHUNK_SIZE_Z as usize) + 4];
            expandrle(column, &mut b2);
            for z in 0..CHUNK_SIZE_Z {
                let zi = z as i32;
                let mut reference = false;
                let mut i = 0;
                while b2[i] < maxzdim {
                    if zi >= b2[i] && zi < b2[i + 1] {
                        reference = true;
                        break;
                    }
                    i += 2;
                }
                assert_eq!(
                    vxl_voxel_solid(vxl, x, y, z),
                    reference,
                    "column ({x}, {y}) z={z} disagrees with expandrle",
                );
            }
        }
    }
}
