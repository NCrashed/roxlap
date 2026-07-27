//! Per-frame voxel-world borrow shape.
//!
//! Wraps the `(vsid, slab_buf, column_offsets, mip_base_offsets)`
//! tuple the renderer ([`crate::dda`]) reads. A single
//! [`roxlap_formats::vxl::Vxl`] is one chunk; a [`ChunkGrid`] composes
//! many chunks behind the same borrow shape.
//!
//! Substage S4B.0 introduced the shape as a pure rename — opticast
//! drove a single flat world behind a typed borrow. Subsequent
//! S4B.x sub-substages grow this into a multi-chunk view:
//!
//! * S4B.1 — carry the camera's chunk index alongside the borrow.
//! * S4B.2.a — `chunk_size_xy` field + `chunk_at_xy` method.
//!   Today's single-chunk callers set `chunk_size_xy = vsid` and
//!   the lookup only succeeds for `[0, 0]`.
//! * S4B.2.b — grouscan column-step calls `chunk_at_xy` and swaps
//!   active per-chunk `(slab_buf, column_offsets)` when `(cx, cy)`
//!   crosses a chunk boundary. Single-chunk goldens byte-identical.
//! * S4B.2.c.1 (this file) — [`ChunkGrid`] backend so `chunk_at_xy`
//!   can resolve real per-chunk borrows. Pure infrastructure; no
//!   caller integration yet.
//! * S4B.2.c.2 — scene-side multi-chunk constructor + 32×32
//!   ground seam test.
//! * S4B.3 — chunk-z extent + handoff for cross-chunk-Z rays.
//!
//! See `project_s4_b_plan.md` for the full sub-substage plan.

use roxlap_formats::vxl::Vxl;

/// Z extent of a single chunk's slab table, in voxels. Voxlap's
/// slab byte format encodes z as `u8`, so each chunk covers exactly
/// 256 voxels along Z. Tall worlds stack chunks vertically (see
/// `memory/project_s4b_6_z_stacking_plan.md`).
pub const CHUNK_SIZE_Z: u32 = 256;

/// Per-frame, zero-copy borrow of one grid's voxel world — the
/// `(vsid, slab_buf, column_offsets, mip_base_offsets)` tuple the DDA
/// renderer reads, plus the optional [`ChunkGrid`] backend for
/// multi-chunk grids. `Copy` so callers pass it by value: every field
/// is a shared borrow or a small integer.
///
/// Fields are public on purpose. External callers usually go through
/// [`from_single_vxl`](GridView::from_single_vxl) /
/// [`from_parts`](GridView::from_parts), but the engine's internals
/// destructure directly.
#[derive(Clone, Copy)]
pub struct GridView<'a> {
    /// Square dimension of the currently-active chunk view (matches
    /// the source `Vxl`'s `vsid` for single-chunk callers). The
    /// per-chunk `column_offsets` table holds `(vsid² + 1)` entries.
    pub vsid: u32,
    /// S4B.2.a: square dimension of each chunk in XY voxel units.
    /// For today's single-chunk callers, `chunk_size_xy == vsid` so
    /// `(cx, cy)` in `[0, vsid)` never crosses a chunk boundary.
    /// For multi-chunk callers (S4B.2.c+), `chunk_size_xy` is the
    /// per-chunk dimension (typically 128) and `vsid` is the same
    /// per-chunk value — they may diverge in S4B.4 when `GridView`
    /// stops carrying a "default" chunk's flat fields.
    pub chunk_size_xy: u32,
    /// S4B.6.a: Z extent of each chunk in voxel units. Locked to
    /// `MAXZDIM = 256` for every chunk (voxlap's slab byte format
    /// uses a `u8` z field). Tall worlds stack chunks vertically
    /// rather than extending this constant — see
    /// `memory/project_s4b_6_z_stacking_plan.md`. Pre-S4B.6.a
    /// callers set this to `256` (the only valid value) and the
    /// rasterizer ignores chunk-z boundaries; S4B.6.c will start
    /// consuming the field for cross-chunk-z column walks.
    pub chunk_size_z: u32,
    /// Flat slab byte buffer for every column at every built mip.
    pub slab_buf: &'a [u8],
    /// Per-column byte offsets into [`Self::slab_buf`], concatenated
    /// across every mip's sub-table. Mip-0 occupies indices
    /// `mip_base_offsets[0]..mip_base_offsets[1]`.
    pub column_offsets: &'a [u32],
    /// Mip-level boundaries inside [`Self::column_offsets`].
    /// Length `mip_count + 1`; trailing sentinel equals
    /// `column_offsets.len()`. Single-mip callers pass
    /// `&[0, vsid² + 1]`.
    pub mip_base_offsets: &'a [usize],
    /// S4B.2.c.1: chunk-grid backend. `None` for single-chunk views
    /// (e.g. anything built via [`Self::from_single_vxl`] /
    /// [`Self::from_parts`]) — [`Self::chunk_at_xy`] falls back to
    /// the `Some(Self) for [0, 0]` behaviour. `Some(&...)` for
    /// multi-chunk views built via [`Self::from_chunk_grid`] —
    /// [`Self::chunk_at_xy`] consults the table.
    pub chunk_grid: Option<&'a ChunkGrid<'a>>,
}

/// S4B.2.c.1: chunk-grid metadata for multi-chunk [`GridView`]
/// lookups.
///
/// Stores a 2D table of optional per-chunk [`GridView`] borrows so
/// [`GridView::chunk_at_xy`] can resolve an XY chunk index to the
/// matching chunk's `(slab_buf, column_offsets, ...)` view. Empty
/// table entries (`None`) signal "no chunk at that index" — the
/// grouscan column-step treats them as fully-air (the chunk renders
/// as sky / empty until the ray crosses into a populated chunk).
///
/// Sized to match the chunk grid's full XYZ footprint:
/// `chunks.len() == chunks_x * chunks_y * chunks_z`. Chunk at
/// relative position `(dx, dy, dz)` from
/// `(origin_chunk_xy, origin_chunk_z)` lives at index
/// `(dz * chunks_y + dy) * chunks_x + dx`. The per-chunk
/// [`GridView`] entries should have `chunk_grid: None` (they
/// describe individual chunks, not the parent grid).
///
/// S4B.6.a: `chunks_z` + `origin_chunk_z` added for tall worlds.
/// Pre-S4B.6 callers built `ChunkGrid` with implicit `chunks_z=1
/// origin_chunk_z=0`; the new layout is backwards-compatible —
/// `dz=0` substitution gives the same index `dy * chunks_x + dx`,
/// so a flat `chunks_x * chunks_y` slice indexes identically.
#[derive(Clone, Copy)]
pub struct ChunkGrid<'a> {
    /// Per-chunk views. Length `chunks_x * chunks_y * chunks_z`.
    /// Index layout: `[(dz * chunks_y + dy) * chunks_x + dx]`.
    pub chunks: &'a [Option<GridView<'a>>],
    /// XY index of the chunk at `chunks[0]`. Subsequent chunks lie
    /// along `+x` (next in the row) then `+y` (next row).
    pub origin_chunk_xy: [i32; 2],
    /// Z index of the chunk at `chunks[0]`. Subsequent z slabs lie
    /// at `+chunks_x * chunks_y` strides into [`Self::chunks`].
    pub origin_chunk_z: i32,
    /// Number of chunks along the X axis. Row stride in
    /// [`Self::chunks`].
    pub chunks_x: u32,
    /// Number of chunks along the Y axis.
    pub chunks_y: u32,
    /// Number of chunks along the Z axis. `1` for non-stacked
    /// worlds (matches every pre-S4B.6.a caller).
    pub chunks_z: u32,
}

impl<'a> GridView<'a> {
    /// Build from explicit fields. Test fixtures use this directly;
    /// production callers usually go through
    /// [`from_single_vxl`](Self::from_single_vxl).
    ///
    /// Sets `chunk_size_xy = vsid` (single-chunk semantics). Use
    /// [`with_chunk_size_xy`](Self::with_chunk_size_xy) to mark the
    /// view as part of a chunk grid.
    #[must_use]
    pub fn from_parts(
        vsid: u32,
        slab_buf: &'a [u8],
        column_offsets: &'a [u32],
        mip_base_offsets: &'a [usize],
    ) -> Self {
        Self {
            vsid,
            chunk_size_xy: vsid,
            chunk_size_z: CHUNK_SIZE_Z,
            slab_buf,
            column_offsets,
            mip_base_offsets,
            chunk_grid: None,
        }
    }

    /// Borrow a parsed `.vxl` map as a single-chunk grid view. The
    /// scene-graph stage's eventual multi-chunk constructor will
    /// live alongside this one (`from_grid` over
    /// `roxlap_scene::Grid`).
    #[must_use]
    pub fn from_single_vxl(vxl: &'a Vxl) -> Self {
        Self {
            vsid: vxl.vsid,
            chunk_size_xy: vxl.vsid,
            chunk_size_z: CHUNK_SIZE_Z,
            slab_buf: &vxl.data,
            column_offsets: &vxl.column_offset,
            mip_base_offsets: &vxl.mip_base_offsets,
            chunk_grid: None,
        }
    }

    /// S4B.2.c.1: build a multi-chunk view from a [`ChunkGrid`].
    ///
    /// The returned [`GridView`]'s flat `(vsid, slab_buf,
    /// column_offsets, mip_base_offsets)` fields are seeded from
    /// the first populated chunk in the grid (so opticast's prelude
    /// has a sensible default before its camera-chunk lookup
    /// refreshes them). `chunk_size_xy` carries the caller-supplied
    /// per-chunk dimension; [`Self::chunk_grid`] points at
    /// `chunk_grid` so [`Self::chunk_at_xy`] resolves every
    /// in-range index to its actual chunk borrow.
    ///
    /// Empty-grid case: if every chunk is `None`, the flat fields
    /// fall back to empty slices and `vsid = chunk_size_xy`. The
    /// grouscan column-step swap will see `chunk_at_xy → None` for
    /// every index and render the whole grid as sky.
    #[must_use]
    pub fn from_chunk_grid(chunk_grid: &'a ChunkGrid<'a>, chunk_size_xy: u32) -> Self {
        let default_chunk = chunk_grid.chunks.iter().find_map(|c| c.as_ref()).copied();
        match default_chunk {
            Some(c) => Self {
                vsid: c.vsid,
                chunk_size_xy,
                chunk_size_z: CHUNK_SIZE_Z,
                slab_buf: c.slab_buf,
                column_offsets: c.column_offsets,
                mip_base_offsets: c.mip_base_offsets,
                chunk_grid: Some(chunk_grid),
            },
            None => Self {
                vsid: chunk_size_xy,
                chunk_size_xy,
                chunk_size_z: CHUNK_SIZE_Z,
                slab_buf: &[],
                column_offsets: &[],
                mip_base_offsets: &[],
                chunk_grid: Some(chunk_grid),
            },
        }
    }

    /// S4B.2.a builder: override [`Self::chunk_size_xy`]. Multi-chunk
    /// callers (S4B.2.c+) use this to mark the view as one chunk of
    /// a larger grid. Today no caller needs it; the existence makes
    /// the seam testable in isolation.
    #[must_use]
    pub fn with_chunk_size_xy(mut self, chunk_size_xy: u32) -> Self {
        self.chunk_size_xy = chunk_size_xy;
        self
    }

    /// Number of built mip levels (mip-0 always counts). `0` only for a
    /// degenerate empty view.
    #[must_use]
    pub fn mip_count(&self) -> u32 {
        #[allow(clippy::cast_possible_truncation)]
        {
            self.mip_base_offsets.len().saturating_sub(1) as u32
        }
    }

    /// Raw slab-chain bytes for column `(x, y)` at mip `mip`, or `None`
    /// if out of range / unbacked / `mip` not built. At mip-N the
    /// per-chunk grid is `(vsid >> N)²` columns.
    fn column_slab_mip(&self, x: u32, y: u32, mip: u32) -> Option<&'a [u8]> {
        if mip >= self.mip_count() {
            return None;
        }
        let vsid_m = (self.vsid >> mip).max(1);
        if x >= vsid_m || y >= vsid_m {
            return None;
        }
        let base = self.mip_base_offsets[mip as usize];
        let col_idx = base + (y * vsid_m + x) as usize;
        let start = *self.column_offsets.get(col_idx)? as usize;
        // Return the unbounded tail from the column's start (not a
        // `slng`-bounded slice): the slab-chain decoders self-terminate
        // at `nextptr == 0`, so computing the exact length up front is a
        // redundant full-chain walk — the dominant cost when
        // `surface_color` is called per cell. `.get()` inside the
        // decoders still guards the buffer end against malformed data.
        self.slab_buf.get(start..)
    }

    /// Top z of the solid run containing voxel `(x, y, z)` at mip 0, or
    /// `None` if the voxel is air. See [`Self::voxel_run_top_mip`].
    #[must_use]
    pub fn voxel_run_top(&self, x: u32, y: u32, z: u32) -> Option<i32> {
        self.voxel_run_top_mip(x, y, z, 0)
    }

    /// Top z of the solid run containing voxel `(x, y, z)` at mip `mip`,
    /// or `None` if air. Coordinates are in mip-`mip` units
    /// (`x, y ∈ [0, vsid >> mip)`, `z ∈ [0, CHUNK_SIZE_Z >> mip)`).
    ///
    /// Mirrors [`roxlap_formats::edit::expandrle`]'s run decomposition
    /// (the same `[top, bot)` solid runs the scene's `voxel_solid`
    /// uses), walking the slab chain in place. The returned top is
    /// always a colour-list voxel, so callers can shade an interior /
    /// side-face hit with `voxel_color_mip(x, y, top, mip)`.
    #[must_use]
    pub fn voxel_run_top_mip(&self, x: u32, y: u32, z: u32, mip: u32) -> Option<i32> {
        let maxz = (CHUNK_SIZE_Z >> mip) as i32;
        if i64::from(z) >= i64::from(maxz) {
            return None;
        }
        let slab = self.column_slab_mip(x, y, mip)?;
        #[allow(clippy::cast_possible_wrap)]
        let zi = z as i32;
        // First run opens at the first slab's z1 (`expandrle uind[0]`).
        let mut top = i32::from(slab[1]);
        let mut v = 0usize;
        loop {
            let nextptr = usize::from(slab[v]);
            if nextptr == 0 {
                // CT.6 — air-terminal slab (`z1c + 1 < z1`, the
                // empty-column sentinel / its chain-tail form): the
                // column ends in AIR — the `top` opened above is the
                // sentinel's own z1, not a real run. NB `z1c == z1-1`
                // is voxlap's legitimate buried-bottom terminal and
                // must keep the bedrock run.
                if i32::from(slab[v + 2]) + 1 < i32::from(slab[v + 1]) {
                    return None;
                }
                // Last run extends to bedrock: [top, maxz).
                return (zi >= top && zi < maxz).then_some(top);
            }
            v += nextptr * 4;
            let ze = i32::from(slab[v + 3]);
            let z1 = i32::from(slab[v + 1]);
            if ze >= z1 {
                continue; // degenerate slab — run continues
            }
            // Current run closes at `ze`; the next opens at `z1`.
            if zi >= top && zi < ze {
                return Some(top);
            }
            top = z1;
        }
    }

    /// Call `f(top, bot)` for each solid run `[top, bot)` of column
    /// `(x, y)` at mip 0. See [`Self::for_each_run_mip`].
    pub fn for_each_run(&self, x: u32, y: u32, f: impl FnMut(i32, i32)) {
        self.for_each_run_mip(x, y, 0, f);
    }

    /// Call `f(top, bot)` for each solid run `[top, bot)` of column
    /// `(x, y)` at mip `mip`, top-to-bottom (the
    /// [`roxlap_formats::edit::expandrle`] decomposition). No-op for an
    /// out-of-range / unbacked column. The brickmap builder
    /// ([`crate::dda`]) uses this to mark occupied bricks.
    pub fn for_each_run_mip(&self, x: u32, y: u32, mip: u32, mut f: impl FnMut(i32, i32)) {
        let Some(slab) = self.column_slab_mip(x, y, mip) else {
            return;
        };
        let maxz = (CHUNK_SIZE_Z >> mip) as i32;
        let mut top = i32::from(slab[1]);
        let mut v = 0usize;
        loop {
            let nextptr = usize::from(slab[v]);
            if nextptr == 0 {
                // CT.6 — air-terminal (see `voxel_run_top_mip`): the
                // column ends in air, no bottom run to emit.
                if i32::from(slab[v + 2]) + 1 < i32::from(slab[v + 1]) {
                    return;
                }
                f(top, maxz); // last run extends to bedrock
                return;
            }
            v += nextptr * 4;
            let ze = i32::from(slab[v + 3]);
            let z1 = i32::from(slab[v + 1]);
            if ze >= z1 {
                continue; // degenerate slab — run continues
            }
            f(top, ze);
            top = z1;
        }
    }

    /// DDA hit colour for voxel `(x, y, z)` at mip 0. See
    /// [`Self::surface_color_mip`].
    #[must_use]
    pub fn surface_color(&self, x: u32, y: u32, z: u32) -> Option<roxlap_formats::VoxColor> {
        self.surface_color_mip(x, y, z, 0)
    }

    /// DDA hit colour for voxel `(x, y, z)` at mip `mip`: the display
    /// colour if the cell hits, or `None` for air.
    ///
    /// CT.6 — the hit VERDICT comes from the solid-run walk, not from
    /// the colour fetch (pre-CT.6 a solid-but-uncoloured cell returned
    /// `None` and the marcher stepped THROUGH solid matter — the
    /// carve-exposed-top fall-through bug). Colour resolves as a
    /// ladder: the cell's own record → its run-top's record (the
    /// surface "bleeds" down a cliff face) →
    /// [`roxlap_formats::VoxColor::BEDROCK_FALLBACK`]. One exception
    /// keeps GPU parity: a cell whose column carries an EXPLICIT
    /// zero-RGB record (the untextured placeholder sentinel) still
    /// reads as air — the GPU decompressor leaves the same cells out
    /// of its solid bitmap.
    #[must_use]
    pub fn surface_color_mip(
        &self,
        x: u32,
        y: u32,
        z: u32,
        mip: u32,
    ) -> Option<roxlap_formats::VoxColor> {
        let top = self.voxel_run_top_mip(x, y, z, mip)?;
        match self.voxel_color_raw_mip(x, y, z, mip) {
            Some(c) if c.0 & 0x00ff_ffff != 0 => Some(c),
            // Explicit zero-RGB record = untextured-air sentinel
            // (GPU `decompress_column` parity).
            Some(_) => None,
            None => Some(
                u32::try_from(top)
                    .ok()
                    .and_then(|t| self.voxel_color_raw_mip(x, y, t, mip))
                    .filter(|c| c.0 & 0x00ff_ffff != 0)
                    .unwrap_or(roxlap_formats::VoxColor::BEDROCK_FALLBACK),
            ),
        }
    }

    /// Surface colour of voxel `(x, y, z)` at mip 0. See
    /// [`Self::voxel_color_mip`].
    #[must_use]
    pub fn voxel_color(&self, x: u32, y: u32, z: u32) -> Option<roxlap_formats::VoxColor> {
        self.voxel_color_mip(x, y, z, 0)
    }

    /// Surface colour of voxel `(x, y, z)` at mip `mip` (mip-`mip`
    /// coordinates), or `None` for an empty / out-of-range cell.
    ///
    /// Decodes the column's slab chain directly from the [`GridView`]
    /// borrow (the same `vbuf` floor + ceiling list walk
    /// [`roxlap_formats::vxl::Vxl::voxel_color`] performs). The returned
    /// `u32` is packed `0xAARRGGBB`; a zero-RGB cell (empty-chunk
    /// placeholder) reads as `None`. Surface voxels only — use
    /// [`Self::surface_color_mip`] for the full solid-cell hit test.
    #[must_use]
    pub fn voxel_color_mip(
        &self,
        x: u32,
        y: u32,
        z: u32,
        mip: u32,
    ) -> Option<roxlap_formats::VoxColor> {
        // Zero RGB = empty-chunk placeholder → untextured.
        self.voxel_color_raw_mip(x, y, z, mip)
            .filter(|c| c.0 & 0x00ff_ffff != 0)
    }

    /// CT.6 — like [`Self::voxel_color_mip`] but WITHOUT the zero-RGB
    /// filter: `Some` for any explicit colour record (including the
    /// untextured `0` sentinel), `None` only when the cell has no
    /// record at all. The distinction drives the hit ladder in
    /// [`Self::surface_color_mip`]: an explicit `0` is placeholder
    /// air, an absent record is solid interior.
    #[must_use]
    pub fn voxel_color_raw_mip(
        &self,
        x: u32,
        y: u32,
        z: u32,
        mip: u32,
    ) -> Option<roxlap_formats::VoxColor> {
        let maxz = (CHUNK_SIZE_Z >> mip) as i32;
        if i64::from(z) >= i64::from(maxz) {
            return None;
        }
        let slab = self.column_slab_mip(x, y, mip)?;
        #[allow(clippy::cast_possible_wrap)]
        let zi = z as i32;
        let texel = |b: &[u8]| -> Option<roxlap_formats::VoxColor> {
            let rgb = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            Some(roxlap_formats::VoxColor(rgb))
        };
        let mut v = 0usize;
        loop {
            // Floor colour list of the current slab: z ∈ [z1, z1c].
            let z_start = i32::from(slab[v + 1]);
            let z1c = i32::from(slab[v + 2]);
            if zi >= z_start && zi <= z1c {
                let off = v + 4 + ((zi - z_start) as usize) * 4;
                return texel(slab.get(off..off + 4)?);
            }
            let nextptr = slab[v];
            if nextptr == 0 {
                return None; // last slab, z not in its floor list
            }
            let (prev_z1, prev_z1c, prev_nextptr) = (z_start, z1c, i32::from(nextptr));
            v += usize::from(nextptr) * 4;
            // Ceiling colour list for the NEW slab — stored in the tail
            // of the previous slab's bytes (voxlap `vbuf` convention).
            let ze = i32::from(slab[v + 3]);
            let ceil_z_start = ze + prev_z1c - prev_z1 - prev_nextptr + 2;
            if zi >= ceil_z_start && zi < ze {
                let ceil_n = (ze - ceil_z_start) as usize;
                let ceil_start = v - ceil_n * 4;
                let off = ceil_start + ((zi - ceil_z_start) as usize) * 4;
                return texel(slab.get(off..off + 4)?);
            }
        }
    }

    /// S4B.2.d: voxel-space XY axis-aligned bounding box of the
    /// grid. Returns `([xmin, ymin], [xmax, ymax])` in voxel units;
    /// the grid contains voxels with coordinates in
    /// `[xmin, xmax) × [ymin, ymax)`.
    ///
    /// - Single-chunk (`chunk_grid: None`): returns
    ///   `([0, 0], [vsid, vsid])`. Byte-identical to the historical
    ///   single-chunk world-edge math.
    /// - Multi-chunk (`chunk_grid: Some(&cg)`): derived from
    ///   `cg.origin_chunk_xy + cg.chunks_x/y * chunk_size_xy`.
    ///
    /// Used as the renderer's outer DDA bounds (the world-edge box the
    /// per-pixel ray is clipped to).
    #[must_use]
    pub fn aabb_xy(&self) -> ([i32; 2], [i32; 2]) {
        if let Some(cg) = self.chunk_grid {
            #[allow(clippy::cast_possible_wrap)]
            let cs = self.chunk_size_xy as i32;
            #[allow(clippy::cast_possible_wrap)]
            let chunks_x = cg.chunks_x as i32;
            #[allow(clippy::cast_possible_wrap)]
            let chunks_y = cg.chunks_y as i32;
            let xmin = cg.origin_chunk_xy[0] * cs;
            let ymin = cg.origin_chunk_xy[1] * cs;
            let xmax = xmin + chunks_x * cs;
            let ymax = ymin + chunks_y * cs;
            ([xmin, ymin], [xmax, ymax])
        } else {
            #[allow(clippy::cast_possible_wrap)]
            let v = self.vsid as i32;
            ([0, 0], [v, v])
        }
    }

    /// Full voxel-space bounding box of the grid: `([x0, y0, z0],
    /// [x1, y1, z1])` half-open in grid-local voxel coordinates. XY
    /// comes from [`Self::aabb_xy`]; Z spans the chunk grid's
    /// `chunks_z` layers (`origin_chunk_z * CHUNK_SIZE_Z` upward), or a
    /// single `[0, CHUNK_SIZE_Z)` chunk when un-stacked. The DDA
    /// renderer ([`crate::dda`]) uses this as the outer traversal box.
    #[must_use]
    pub fn voxel_bounds(&self) -> ([i32; 3], [i32; 3]) {
        let ([x0, y0], [x1, y1]) = self.aabb_xy();
        let csz = CHUNK_SIZE_Z as i32;
        let (z0, z1) = if let Some(cg) = self.chunk_grid {
            #[allow(clippy::cast_possible_wrap)]
            let chunks_z = cg.chunks_z as i32;
            (
                cg.origin_chunk_z * csz,
                (cg.origin_chunk_z + chunks_z) * csz,
            )
        } else {
            (0, csz)
        };
        ([x0, y0, z0], [x1, y1, z1])
    }

    /// S4B.2.a: chunk lookup for the cross-chunk-XY DDA.
    ///
    /// Returns the [`GridView`] for the chunk at XY index
    /// `chunk_idx` if one exists, `None` otherwise. Routes by
    /// [`Self::chunk_grid`]:
    ///
    /// * `chunk_grid: Some(&cg)` — multi-chunk view. Resolves
    ///   `chunk_idx` against `cg.origin_chunk_xy` /
    ///   `cg.chunks_x` / `cg.chunks_y` and returns `cg.chunks[..]`
    ///   at the matching slot (which may itself be `None` for
    ///   empty chunks).
    /// * `chunk_grid: None` — single-chunk view. Returns `Some(Self)`
    ///   for `[0, 0]`, `None` for any other index. Matches today's
    ///   single-chunk callers (every `from_single_vxl` /
    ///   `from_parts` consumer).
    ///
    /// The grouscan column-step treats `None` as an empty chunk
    /// (renders as sky / empty until the ray re-enters a populated
    /// chunk).
    #[must_use]
    pub fn chunk_at_xy(&self, chunk_idx: [i32; 2]) -> Option<GridView<'a>> {
        // Defer to chunk_at_xyz at the grid's `origin_chunk_z` (=
        // 0 for non-stacked worlds, the "current" z layer for
        // S4B.6+ stacked worlds). Pre-S4B.6.a behaviour is preserved
        // because `chunks_z=1, origin_chunk_z=0` is the default.
        let z = self.chunk_grid.map_or(0, |cg| cg.origin_chunk_z);
        self.chunk_at_xyz([chunk_idx[0], chunk_idx[1], z])
    }

    /// S4B.6.a: 3D chunk lookup for the future cross-chunk-Z DDA.
    ///
    /// Same dispatch contract as [`Self::chunk_at_xy`] but extended
    /// to a z axis:
    ///
    /// * `chunk_grid: Some(&cg)` — multi-chunk view. Resolves
    ///   `chunk_idx` against `cg.origin_chunk_xy` /
    ///   `cg.origin_chunk_z` / `cg.chunks_x` / `cg.chunks_y` /
    ///   `cg.chunks_z` and returns `cg.chunks[..]` at the matching
    ///   slot (which may itself be `None` for empty chunks).
    /// * `chunk_grid: None` — single-chunk view. Returns
    ///   `Some(Self)` for `[0, 0, *]`, `None` otherwise. The z
    ///   index is ignored because single-chunk callers carry an
    ///   un-stacked world — the camera's chz, even when non-zero
    ///   (e.g. camera at world z >= 256 below the chunk's bedrock
    ///   with `treat_z_max_as_air`), still refers to the same one
    ///   chunk. S4B.6.c will start treating chz as a separator
    ///   only for multi-chunk grids.
    #[must_use]
    pub fn chunk_at_xyz(&self, chunk_idx: [i32; 3]) -> Option<GridView<'a>> {
        if let Some(cg) = self.chunk_grid {
            let dx = chunk_idx[0] - cg.origin_chunk_xy[0];
            let dy = chunk_idx[1] - cg.origin_chunk_xy[1];
            let dz = chunk_idx[2] - cg.origin_chunk_z;
            if dx < 0 || dy < 0 || dz < 0 {
                return None;
            }
            #[allow(clippy::cast_sign_loss)]
            let (dx, dy, dz) = (dx as u32, dy as u32, dz as u32);
            if dx >= cg.chunks_x || dy >= cg.chunks_y || dz >= cg.chunks_z {
                return None;
            }
            let i = (dz as usize * cg.chunks_y as usize + dy as usize) * cg.chunks_x as usize
                + dx as usize;
            cg.chunks.get(i).copied().flatten()
        } else if chunk_idx[0] == 0 && chunk_idx[1] == 0 {
            Some(*self)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_parts_preserves_fields_byte_identically() {
        let slab = [0u8, 200, 254, 0];
        let cols = [0u32, 4];
        let mips = [0usize, 2];
        let gv = GridView::from_parts(1, &slab, &cols, &mips);
        assert_eq!(gv.vsid, 1);
        assert_eq!(gv.slab_buf, &slab[..]);
        assert_eq!(gv.column_offsets, &cols[..]);
        assert_eq!(gv.mip_base_offsets, &mips[..]);
    }

    #[test]
    fn grid_view_is_copy() {
        // Compile-time check: GridView must be Copy so opticast +
        // ScalarRasterizer can stash independent copies without a
        // borrow-checker dance.
        fn assert_copy<T: Copy>() {}
        assert_copy::<GridView<'_>>();
    }

    #[test]
    fn from_parts_defaults_chunk_size_xy_to_vsid() {
        let mips = [0usize, 2];
        let gv = GridView::from_parts(2048, &[], &[], &mips);
        assert_eq!(gv.chunk_size_xy, 2048);
    }

    #[test]
    fn chunk_at_xy_returns_self_for_origin_chunk() {
        let slab = [0u8, 200, 254, 0];
        let cols = [0u32, 4];
        let mips = [0usize, 2];
        let gv = GridView::from_parts(1, &slab, &cols, &mips);
        let inner = gv.chunk_at_xy([0, 0]).expect("origin chunk present");
        assert_eq!(inner.vsid, gv.vsid);
        assert_eq!(inner.slab_buf, gv.slab_buf);
        assert_eq!(inner.column_offsets, gv.column_offsets);
    }

    #[test]
    fn chunk_at_xy_returns_none_for_off_origin_idx() {
        let mips = [0usize, 2];
        let gv = GridView::from_parts(1, &[], &[], &mips);
        assert!(gv.chunk_at_xy([1, 0]).is_none());
        assert!(gv.chunk_at_xy([-1, 0]).is_none());
        assert!(gv.chunk_at_xy([0, 1]).is_none());
        assert!(gv.chunk_at_xy([5, -7]).is_none());
    }

    #[test]
    fn with_chunk_size_xy_overrides_default() {
        let mips = [0usize, 2];
        let gv = GridView::from_parts(2048, &[], &[], &mips).with_chunk_size_xy(128);
        assert_eq!(gv.vsid, 2048);
        assert_eq!(gv.chunk_size_xy, 128);
    }

    /// S4B.2.c.1: tiny 2-chunk-x-stripe ChunkGrid scaffolding for
    /// the multi-chunk lookup tests. Two synthetic 1×1 chunks at
    /// XY indices [0, 0] and [1, 0]. Their slab/column data is
    /// distinct so the lookup tests can tell them apart.
    #[allow(clippy::type_complexity)] // test fixture: a bag of parallel arrays
    fn build_two_chunk_x_stripe() -> ([u8; 4], [u8; 4], [u32; 2], [u32; 2], [usize; 2], [usize; 2])
    {
        let slab_a = [10u8, 200, 254, 0];
        let slab_b = [20u8, 200, 254, 0];
        let cols_a = [0u32, 4];
        let cols_b = [0u32, 4];
        let mips_a = [0usize, 2];
        let mips_b = [0usize, 2];
        (slab_a, slab_b, cols_a, cols_b, mips_a, mips_b)
    }

    #[test]
    fn chunk_at_xy_via_chunk_grid_returns_in_range_chunks() {
        let (slab_a, slab_b, cols_a, cols_b, mips_a, mips_b) = build_two_chunk_x_stripe();
        let chunks = [
            Some(GridView::from_parts(64, &slab_a, &cols_a, &mips_a)),
            Some(GridView::from_parts(64, &slab_b, &cols_b, &mips_b)),
        ];
        let cg = ChunkGrid {
            chunks: &chunks,
            origin_chunk_xy: [0, 0],
            chunks_x: 2,
            chunks_y: 1,
            chunks_z: 1,
            origin_chunk_z: 0,
        };
        let gv = GridView::from_chunk_grid(&cg, 64);
        // [0, 0] resolves to chunk A.
        let c0 = gv.chunk_at_xy([0, 0]).expect("chunk [0, 0] present");
        assert_eq!(c0.slab_buf, &slab_a[..]);
        // [1, 0] resolves to chunk B.
        let c1 = gv.chunk_at_xy([1, 0]).expect("chunk [1, 0] present");
        assert_eq!(c1.slab_buf, &slab_b[..]);
        // [0, 0] and [1, 0] carry distinct slab buffers.
        assert_ne!(c0.slab_buf, c1.slab_buf);
    }

    #[test]
    fn chunk_at_xy_via_chunk_grid_returns_none_out_of_range() {
        let (slab_a, _, cols_a, _, mips_a, _) = build_two_chunk_x_stripe();
        let chunks = [
            Some(GridView::from_parts(64, &slab_a, &cols_a, &mips_a)),
            None,
        ];
        let cg = ChunkGrid {
            chunks: &chunks,
            origin_chunk_xy: [0, 0],
            chunks_x: 2,
            chunks_y: 1,
            chunks_z: 1,
            origin_chunk_z: 0,
        };
        let gv = GridView::from_chunk_grid(&cg, 64);
        // [-1, 0] is below origin.
        assert!(gv.chunk_at_xy([-1, 0]).is_none());
        // [0, -1] is below origin (y).
        assert!(gv.chunk_at_xy([0, -1]).is_none());
        // [2, 0] is past chunks_x.
        assert!(gv.chunk_at_xy([2, 0]).is_none());
        // [0, 1] is past chunks_y.
        assert!(gv.chunk_at_xy([0, 1]).is_none());
        // [1, 0] is an empty slot inside the grid.
        assert!(gv.chunk_at_xy([1, 0]).is_none());
    }

    #[test]
    fn chunk_at_xy_handles_negative_origin() {
        let (slab_a, slab_b, cols_a, cols_b, mips_a, mips_b) = build_two_chunk_x_stripe();
        // Origin at [-1, 0]: chunks live at XY indices [-1, 0] and [0, 0].
        let chunks = [
            Some(GridView::from_parts(64, &slab_a, &cols_a, &mips_a)),
            Some(GridView::from_parts(64, &slab_b, &cols_b, &mips_b)),
        ];
        let cg = ChunkGrid {
            chunks: &chunks,
            origin_chunk_xy: [-1, 0],
            chunks_x: 2,
            chunks_y: 1,
            chunks_z: 1,
            origin_chunk_z: 0,
        };
        let gv = GridView::from_chunk_grid(&cg, 64);
        let cm1 = gv.chunk_at_xy([-1, 0]).expect("chunk [-1, 0] present");
        assert_eq!(cm1.slab_buf, &slab_a[..]);
        let c0 = gv.chunk_at_xy([0, 0]).expect("chunk [0, 0] present");
        assert_eq!(c0.slab_buf, &slab_b[..]);
        // Past the right edge.
        assert!(gv.chunk_at_xy([1, 0]).is_none());
        // Past the left edge.
        assert!(gv.chunk_at_xy([-2, 0]).is_none());
    }

    #[test]
    fn from_chunk_grid_seeds_flat_fields_from_first_populated_chunk() {
        let (slab_a, slab_b, cols_a, cols_b, mips_a, mips_b) = build_two_chunk_x_stripe();
        // First slot empty, second populated. Flat fields should
        // seed from chunk B.
        let chunks = [
            None,
            Some(GridView::from_parts(64, &slab_b, &cols_b, &mips_b)),
        ];
        let cg = ChunkGrid {
            chunks: &chunks,
            origin_chunk_xy: [0, 0],
            chunks_x: 2,
            chunks_y: 1,
            chunks_z: 1,
            origin_chunk_z: 0,
        };
        let gv = GridView::from_chunk_grid(&cg, 64);
        assert_eq!(gv.slab_buf, &slab_b[..]);
        assert_eq!(gv.chunk_size_xy, 64);
        // Single-chunk fields stay populated; tests beyond rely on
        // opticast's prelude refreshing them via the camera lookup.
        let _ = (slab_a, cols_a, mips_a);
    }

    #[test]
    fn aabb_xy_single_chunk_returns_0_to_vsid() {
        let mips = [0usize, 2];
        let gv = GridView::from_parts(2048, &[], &[], &mips);
        let (lo, hi) = gv.aabb_xy();
        assert_eq!(lo, [0, 0]);
        assert_eq!(hi, [2048, 2048]);
    }

    #[test]
    fn aabb_xy_multi_chunk_covers_full_extent() {
        let (slab_a, _, cols_a, _, mips_a, _) = build_two_chunk_x_stripe();
        // 2-wide × 3-tall chunk grid starting at origin [-1, 0].
        // Each chunk is 128² (matches the constructor's chunk_size_xy).
        let chunks = [
            Some(GridView::from_parts(128, &slab_a, &cols_a, &mips_a)),
            None,
            None,
            None,
            None,
            None,
        ];
        let cg = ChunkGrid {
            chunks: &chunks,
            origin_chunk_xy: [-1, 0],
            chunks_x: 2,
            chunks_y: 3,
            chunks_z: 1,
            origin_chunk_z: 0,
        };
        let gv = GridView::from_chunk_grid(&cg, 128);
        let (lo, hi) = gv.aabb_xy();
        // xmin = -1 * 128 = -128; xmax = (-1 + 2) * 128 = 128.
        // ymin = 0; ymax = 3 * 128 = 384.
        assert_eq!(lo, [-128, 0]);
        assert_eq!(hi, [128, 384]);
    }

    #[test]
    fn from_chunk_grid_empty_table_falls_back_to_empty_slices() {
        let chunks: [Option<GridView<'_>>; 1] = [None];
        let cg = ChunkGrid {
            chunks: &chunks,
            origin_chunk_xy: [0, 0],
            chunks_x: 1,
            chunks_y: 1,
            chunks_z: 1,
            origin_chunk_z: 0,
        };
        let gv = GridView::from_chunk_grid(&cg, 128);
        assert!(gv.slab_buf.is_empty());
        assert!(gv.column_offsets.is_empty());
        assert!(gv.mip_base_offsets.is_empty());
        assert_eq!(gv.vsid, 128);
        assert_eq!(gv.chunk_size_xy, 128);
        // The chunk_grid backend still gates chunk_at_xy.
        assert!(gv.chunk_at_xy([0, 0]).is_none());
    }

    /// S4B.6.a: `chunk_at_xyz` resolves the z axis correctly on a
    /// stacked grid. Two chunks at chz=0 and chz=1 each have their
    /// own slab buffer; xyz lookup must return the matching one.
    #[test]
    fn chunk_at_xyz_resolves_stacked_chunks() {
        let slab0 = [0u8, 100, 100, 0];
        let slab1 = [0u8, 200, 200, 0];
        let cols = [0u32, 4];
        let mips = [0usize, 2];
        let c0 = GridView::from_parts(1, &slab0, &cols, &mips);
        let c1 = GridView::from_parts(1, &slab1, &cols, &mips);
        let chunks = [Some(c0), Some(c1)];
        let cg = ChunkGrid {
            chunks: &chunks,
            origin_chunk_xy: [0, 0],
            origin_chunk_z: 0,
            chunks_x: 1,
            chunks_y: 1,
            chunks_z: 2,
        };
        let gv = GridView::from_chunk_grid(&cg, 1);
        let v0 = gv.chunk_at_xyz([0, 0, 0]).expect("chz=0 present");
        assert_eq!(v0.slab_buf, &slab0[..]);
        let v1 = gv.chunk_at_xyz([0, 0, 1]).expect("chz=1 present");
        assert_eq!(v1.slab_buf, &slab1[..]);
        // OOR z returns None.
        assert!(gv.chunk_at_xyz([0, 0, 2]).is_none());
        assert!(gv.chunk_at_xyz([0, 0, -1]).is_none());
    }

    /// S4B.6.a: `chunk_at_xy` defers to `chunk_at_xyz` at
    /// `origin_chunk_z`. For a stacked grid centred at chz=-1,
    /// `chunk_at_xy` returns the chz=-1 layer.
    #[test]
    fn chunk_at_xy_returns_origin_z_layer() {
        let slab_z_neg1 = [0u8, 50, 50, 0];
        let slab_z_0 = [0u8, 100, 100, 0];
        let cols = [0u32, 4];
        let mips = [0usize, 2];
        let cn = GridView::from_parts(1, &slab_z_neg1, &cols, &mips);
        let c0 = GridView::from_parts(1, &slab_z_0, &cols, &mips);
        let chunks = [Some(cn), Some(c0)];
        let cg = ChunkGrid {
            chunks: &chunks,
            origin_chunk_xy: [0, 0],
            origin_chunk_z: -1,
            chunks_x: 1,
            chunks_y: 1,
            chunks_z: 2,
        };
        let gv = GridView::from_chunk_grid(&cg, 1);
        // chunk_at_xy returns origin_chunk_z layer = chz=-1.
        let via_xy = gv.chunk_at_xy([0, 0]).expect("origin layer present");
        assert_eq!(via_xy.slab_buf, &slab_z_neg1[..]);
        // chunk_at_xyz with explicit chz=0 returns the upper layer.
        let v0 = gv.chunk_at_xyz([0, 0, 0]).expect("chz=0 present");
        assert_eq!(v0.slab_buf, &slab_z_0[..]);
    }
}
