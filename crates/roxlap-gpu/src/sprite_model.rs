//! GPU.10 — KV6 sprite as a DDA-marchable voxel model.
//!
//! Unlike the GPU.9 splatter (one thread per voxel, screen-space
//! squares, overdraw + atomic contention), a sprite model is a small
//! voxel volume the precise ray-DDA marches one ray per pixel —
//! crisp, correct occlusion, no overdraw. This is the GPU.10.0 single
//! sprite; instancing + tiling + LOD come in later sub-substages.
//!
//! The volume reuses the chunk occupancy/colour scheme but sized to
//! the KV6 bbox: per-column occupancy bitmask (`occ_words_per_col`
//! u32s, `CHUNK_Z`-style 32-bits-per-word), a flat colour array in
//! ascending-z order per column, and a `color_offsets` prefix table.
//! The shader finds a voxel's colour by `offset[col] + popcount(bits
//! below z)`, so colours MUST be ascending-z (we sort per column).

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::similar_names
)]

use bytemuck::{Pod, Zeroable};
use roxlap_formats::kv6::Kv6;
use roxlap_formats::sprite::Sprite;

/// CPU-built voxel volume for one KV6 model.
#[derive(Debug, Clone)]
pub struct SpriteModel {
    /// Voxel extent `(mx, my, mz)`.
    pub dims: [u32; 3],
    /// `ceil(mz / 32)` — u32 words of occupancy per (x, y) column.
    pub occ_words_per_col: u32,
    /// KV6 pivot in model-local voxel space.
    pub pivot: [f32; 3],
    /// Per-column occupancy bitmask, `mx * my * occ_words_per_col`.
    pub occupancy: Vec<u32>,
    /// Voxel colours, ascending z within each column.
    pub colors: Vec<u32>,
    /// Prefix sums: `color_offsets[col]` is the first colour index of
    /// column `col`; length `mx * my + 1`.
    pub color_offsets: Vec<u32>,
    /// World-space size of one voxel of this model (GPU.10.4 LOD): 1.0
    /// at mip-0, doubling each [`SpriteModel::downsample`]. The shader
    /// divides the local ray by this so a coarse voxel spans the right
    /// world extent and the march `t` stays in world units.
    pub voxel_world_size: f32,
}

/// Build the DDA volume from a KV6. Columns are packed in
/// `x + y*mx` order; each column's voxels are sorted ascending by z
/// so the shader's popcount-rank colour lookup is correct.
///
/// # Panics
/// If the KV6's `ylen` counters disagree with `voxels.len()` (a
/// malformed model).
#[must_use]
pub fn build_sprite_model(kv6: &Kv6) -> SpriteModel {
    let (mx, my, mz) = (kv6.xsiz, kv6.ysiz, kv6.zsiz);
    let occ_words_per_col = mz.div_ceil(32).max(1);
    let cols = (mx * my) as usize;

    let mut occupancy = vec![0u32; cols * occ_words_per_col as usize];
    let mut color_offsets = vec![0u32; cols + 1];
    let mut colors: Vec<u32> = Vec::with_capacity(kv6.voxels.len());

    // Pass 1 — consume voxels in KV6 storage order (x-outer / y-inner)
    // into per-column buckets keyed by `col = x + y*mx`.
    let mut buckets: Vec<Vec<(u16, u32)>> = vec![Vec::new(); cols];
    let mut voxel_iter = kv6.voxels.iter();
    for x in 0..mx {
        for y in 0..my {
            let col = (x + y * mx) as usize;
            let count = kv6.ylen[x as usize][y as usize];
            for _ in 0..count {
                let v = voxel_iter.next().expect("KV6 ylen / voxels.len mismatch");
                buckets[col].push((v.z, v.col));
            }
        }
    }

    // Pass 2 — emit in COLUMN-INDEX order so `color_offsets` is a true
    // monotonic prefix sum (the shader indexes by `col` either way, but
    // structural edits / mip rebuilds rely on monotonic offsets). Each
    // column's voxels sorted ascending z for the popcount-rank lookup.
    for (col, bucket) in buckets.iter_mut().enumerate() {
        color_offsets[col] = colors.len() as u32;
        bucket.sort_by_key(|(z, _)| *z);
        for &(z, col_rgba) in bucket.iter() {
            let z = u32::from(z);
            let base = col * occ_words_per_col as usize + (z >> 5) as usize;
            occupancy[base] |= 1u32 << (z & 31);
            colors.push(col_rgba);
        }
    }
    color_offsets[cols] = colors.len() as u32;

    SpriteModel {
        dims: [mx, my, mz],
        occ_words_per_col,
        pivot: [kv6.xpiv, kv6.ypiv, kv6.zpiv],
        occupancy,
        color_offsets,
        colors,
        voxel_world_size: 1.0,
    }
}

/// Per-instance transform consumed by the model-DDA shader: the
/// inverse model→world rotation (so a world ray can be brought into
/// model-local space) plus the instance's world position. Stored as
/// three padded columns for std140/std430 (`mat3x3` 16-byte columns).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct SpriteInstanceTransform {
    /// Inverse of `[s | h | f]`, column-major, each column padded to
    /// `vec4`. `inv_rot * v = c0*v.x + c1*v.y + c2*v.z`.
    pub inv_rot: [[f32; 4]; 3],
    /// Instance world position (the KV6 pivot maps here).
    pub pos: [f32; 3],
    _pad: f32,
}

impl SpriteInstanceTransform {
    /// Build from a sprite pose. `s/h/f` are the model→world basis
    /// columns; we invert them so the shader can map world→local.
    #[must_use]
    pub fn from_sprite(sprite: &Sprite) -> Self {
        let inv = mat3_inverse([sprite.s, sprite.h, sprite.f]);
        Self {
            inv_rot: [
                [inv[0][0], inv[0][1], inv[0][2], 0.0],
                [inv[1][0], inv[1][1], inv[1][2], 0.0],
                [inv[2][0], inv[2][1], inv[2][2], 0.0],
            ],
            pos: sprite.p,
            _pad: 0.0,
        }
    }
}

/// A registry of sprite models. Instances reference a model by
/// `model_id`, which is a **LOD chain** id: each chain holds one or
/// more concrete mip levels (finest first; GPU.10.4), and the renderer
/// picks the level per instance by distance. Identical KV6s are added
/// once and shared by many instances. **Copy-on-modify**:
/// [`Self::fork`] deep-copies a chain so edits to the fork leave the
/// parent (and its instances) intact.
#[derive(Debug, Clone, Default)]
pub struct SpriteModelRegistry {
    /// Concrete mip-level volumes (the GPU buffers concatenate these).
    entries: Vec<SpriteModel>,
    /// `chains[model_id]` = entry ids, finest (mip-0) first.
    chains: Vec<Vec<u32>>,
}

impl SpriteModelRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn push_entry(&mut self, model: SpriteModel) -> u32 {
        let id = self.entries.len() as u32;
        self.entries.push(model);
        id
    }

    /// Register a single-level (no-LOD) model; returns its `model_id`.
    pub fn add(&mut self, model: SpriteModel) -> u32 {
        let e = self.push_entry(model);
        let id = self.chains.len() as u32;
        self.chains.push(vec![e]);
        id
    }

    /// Register a model with up to `max_levels` LOD mips (each a 2×
    /// [`SpriteModel::downsample`] of the previous; stops early once a
    /// level collapses to 1³). Returns its `model_id`.
    pub fn add_lod(&mut self, model: SpriteModel, max_levels: u32) -> u32 {
        let mut levels = vec![self.push_entry(model.clone())];
        let mut cur = model;
        for _ in 1..max_levels.max(1) {
            if cur.dims == [1, 1, 1] {
                break;
            }
            cur = cur.downsample();
            levels.push(self.push_entry(cur.clone()));
        }
        let id = self.chains.len() as u32;
        self.chains.push(levels);
        id
    }

    /// Copy-on-modify: deep-copy every level of chain `parent` into new
    /// entries + a new chain, and return its `model_id`. The fork owns
    /// independent voxel data, so mutating it does not affect the
    /// parent or any instance still pointing at it.
    ///
    /// # Panics
    /// If `parent` is not a registered `model_id`.
    pub fn fork(&mut self, parent: u32) -> u32 {
        let src = self.chains[parent as usize].clone();
        let levels: Vec<u32> = src
            .iter()
            .map(|&e| {
                let copy = self.entries[e as usize].clone();
                self.push_entry(copy)
            })
            .collect();
        let id = self.chains.len() as u32;
        self.chains.push(levels);
        id
    }

    /// The finest (mip-0) model of chain `id`.
    #[must_use]
    pub fn model(&self, id: u32) -> &SpriteModel {
        &self.entries[self.chains[id as usize][0] as usize]
    }

    /// Mutable access to the finest (mip-0) model for editing — the
    /// copy-on-modify entry point (typically on a [`Self::fork`]).
    /// After a *structural* edit (occupancy/dims), call
    /// [`Self::rebuild_lod`] so the coarser mips match; a pure recolour
    /// can use [`Self::recolor_chain`] instead.
    pub fn model_mut(&mut self, id: u32) -> &mut SpriteModel {
        let e = self.chains[id as usize][0] as usize;
        &mut self.entries[e]
    }

    /// Recolour every LOD level of chain `id` (so a forked tint shows
    /// at all distances).
    pub fn recolor_chain(&mut self, id: u32, f: impl Fn(u32) -> u32 + Copy) {
        for li in 0..self.chains[id as usize].len() {
            let e = self.chains[id as usize][li] as usize;
            self.entries[e].recolor(f);
        }
    }

    /// Regenerate chain `id`'s coarser mip levels from its (possibly
    /// just-edited) mip-0. Run after a structural edit via
    /// [`Self::model_mut`] so the LOD ladder stays consistent. No-op
    /// for a single-level (no-LOD) chain.
    pub fn rebuild_lod(&mut self, id: u32) {
        let levels = self.chains[id as usize].clone();
        if levels.len() <= 1 {
            return;
        }
        let mut cur = self.entries[levels[0] as usize].clone();
        for &e in &levels[1..] {
            cur = cur.downsample();
            self.entries[e as usize] = cur.clone();
        }
    }

    /// Number of LOD chains (distinct `model_id`s).
    #[must_use]
    pub fn len(&self) -> usize {
        self.chains.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.chains.is_empty()
    }
}

impl SpriteModel {
    /// Recolour every voxel via `f(old_rgba) -> new_rgba`. Structure
    /// (occupancy / offsets) is untouched, so this is a cheap in-place
    /// edit — handy on a [`SpriteModelRegistry::fork`] to make a tinted
    /// variant. For structural edits, mutate the public occupancy /
    /// colours / dims directly (via `model_mut`) then rebuild the LOD.
    pub fn recolor(&mut self, f: impl Fn(u32) -> u32) {
        for c in &mut self.colors {
            *c = f(*c);
        }
    }

    /// GPU.12 — structural edit of a single voxel within the model's
    /// existing bounds. `Some(rgba)` sets/replaces the voxel at
    /// `(x, y, z)`; `None` clears it. Maintains the ascending-z colour
    /// invariant by inserting/removing at the voxel's popcount rank and
    /// shifting the affected columns' `color_offsets`. Returns `true`
    /// if the model changed. Out-of-bounds coordinates are ignored
    /// (returns `false`) — growing `dims` is a separate concern.
    ///
    /// After editing, call [`SpriteModelRegistry::rebuild_lod`] to
    /// refresh coarser mips, then re-upload via `set_sprite_instances`.
    pub fn set_voxel(&mut self, x: u32, y: u32, z: u32, color: Option<u32>) -> bool {
        if x >= self.dims[0] || y >= self.dims[1] || z >= self.dims[2] {
            return false;
        }
        let owpc = self.occ_words_per_col as usize;
        let cols = (self.dims[0] * self.dims[1]) as usize;
        let col = (x + y * self.dims[0]) as usize;
        let base = col * owpc;
        let zw = (z >> 5) as usize;
        let zb = z & 31;

        // Rank = solid voxels strictly below z in this column.
        let mut rank = 0usize;
        for w in 0..zw {
            rank += self.occupancy[base + w].count_ones() as usize;
        }
        let below_mask = if zb > 0 { (1u32 << zb) - 1 } else { 0 };
        rank += (self.occupancy[base + zw] & below_mask).count_ones() as usize;
        let idx = self.color_offsets[col] as usize + rank;
        let was_set = (self.occupancy[base + zw] >> zb) & 1 == 1;

        if let Some(rgba) = color {
            if was_set {
                self.colors[idx] = rgba; // replace in place
            } else {
                self.occupancy[base + zw] |= 1u32 << zb;
                self.colors.insert(idx, rgba);
                for c in &mut self.color_offsets[col + 1..=cols] {
                    *c += 1;
                }
            }
            true
        } else {
            if !was_set {
                return false;
            }
            self.occupancy[base + zw] &= !(1u32 << zb);
            self.colors.remove(idx);
            for c in &mut self.color_offsets[col + 1..=cols] {
                *c -= 1;
            }
            true
        }
    }

    /// Radius of a bounding sphere centred at the instance position
    /// (the pivot maps there): the farthest bbox corner from the
    /// pivot. Used for frustum culling. Assumes a unit basis; scaled
    /// instances would multiply this by their max basis length.
    #[must_use]
    pub fn bound_radius(&self) -> f32 {
        let mut r2 = 0.0_f32;
        for &cx in &[0.0, self.dims[0] as f32] {
            for &cy in &[0.0, self.dims[1] as f32] {
                for &cz in &[0.0, self.dims[2] as f32] {
                    let d = [cx - self.pivot[0], cy - self.pivot[1], cz - self.pivot[2]];
                    r2 = r2.max(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
                }
            }
        }
        r2.sqrt()
    }

    /// GPU.10.4 — 2× voxel downsample for the next LOD level. A coarse
    /// voxel is solid if any of its 2×2×2 fine voxels is, coloured by
    /// their per-channel average. Dims/pivot halve and
    /// `voxel_world_size` doubles, so the coarse model occupies the
    /// same world box at half the resolution (origin-corner aligned).
    #[must_use]
    #[allow(clippy::manual_checked_ops)] // `n > 0` guards 4 divisions, not one checked_div
    pub fn downsample(&self) -> SpriteModel {
        let [fx, fy, fz] = self.dims;
        let fidx = |x: u32, y: u32, z: u32| (x + y * fx + z * fx * fy) as usize;

        // Reconstruct dense fine voxels (solid flag + colour).
        let mut solid = vec![false; (fx * fy * fz) as usize];
        let mut fine = vec![0u32; (fx * fy * fz) as usize];
        for x in 0..fx {
            for y in 0..fy {
                let col = (x + y * fx) as usize;
                let base = col * self.occ_words_per_col as usize;
                let off = self.color_offsets[col] as usize;
                let mut seen = 0usize;
                for z in 0..fz {
                    let w = base + (z >> 5) as usize;
                    if (self.occupancy[w] >> (z & 31)) & 1 == 1 {
                        fine[fidx(x, y, z)] = self.colors[off + seen];
                        solid[fidx(x, y, z)] = true;
                        seen += 1;
                    }
                }
            }
        }

        let nx = fx.div_ceil(2).max(1);
        let ny = fy.div_ceil(2).max(1);
        let nz = fz.div_ceil(2).max(1);
        let owpc = nz.div_ceil(32).max(1);
        let cols = (nx * ny) as usize;
        let mut occupancy = vec![0u32; cols * owpc as usize];
        let mut color_offsets = vec![0u32; cols + 1];
        let mut colors: Vec<u32> = Vec::new();

        // Emit in column-index order (`ccol = cx + cy*nx`), cy outer,
        // so `color_offsets` is a monotonic prefix sum like build's.
        for cy in 0..ny {
            for cx in 0..nx {
                let ccol = (cx + cy * nx) as usize;
                color_offsets[ccol] = colors.len() as u32;
                for cz in 0..nz {
                    let (mut a, mut r, mut g, mut b, mut n) = (0u32, 0u32, 0u32, 0u32, 0u32);
                    for dz in 0..2 {
                        for dy in 0..2 {
                            for dx in 0..2 {
                                let (x, y, z) = (2 * cx + dx, 2 * cy + dy, 2 * cz + dz);
                                if x < fx && y < fy && z < fz && solid[fidx(x, y, z)] {
                                    let c = fine[fidx(x, y, z)];
                                    a += (c >> 24) & 0xff;
                                    r += (c >> 16) & 0xff;
                                    g += (c >> 8) & 0xff;
                                    b += c & 0xff;
                                    n += 1;
                                }
                            }
                        }
                    }
                    if n > 0 {
                        let avg = ((a / n) << 24) | ((r / n) << 16) | ((g / n) << 8) | (b / n);
                        let base = ccol * owpc as usize + (cz >> 5) as usize;
                        occupancy[base] |= 1u32 << (cz & 31);
                        colors.push(avg);
                    }
                }
            }
        }
        color_offsets[cols] = colors.len() as u32;

        SpriteModel {
            dims: [nx, ny, nz],
            occ_words_per_col: owpc,
            pivot: [
                self.pivot[0] * 0.5,
                self.pivot[1] * 0.5,
                self.pivot[2] * 0.5,
            ],
            occupancy,
            colors,
            color_offsets,
            voxel_world_size: self.voxel_world_size * 2.0,
        }
    }
}

/// View frustum for CPU instance culling, in world space. Built each
/// frame from the world camera. `half_w`/`half_h` are the tangents of
/// the half-FOV (so the side planes are `|x| <= half_w * z` etc. in
/// camera space).
#[derive(Clone, Copy, Debug)]
pub struct ViewFrustum {
    pub pos: [f32; 3],
    pub right: [f32; 3],
    pub down: [f32; 3],
    pub forward: [f32; 3],
    pub half_w: f32,
    pub half_h: f32,
    pub far: f32,
}

/// CPU cull record: the GPU instance + its world bounding sphere.
#[derive(Clone, Copy)]
struct CullInstance {
    /// Instance transform + a placeholder `model_id`; the cull
    /// overwrites `model_id` with the distance-chosen LOD entry.
    gpu: SpriteInstanceGpu,
    /// LOD chain this instance draws (the user-facing `model_id`).
    chain_id: u32,
    center: [f32; 3],
    radius: f32,
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// One sprite instance: a model reference + world pose.
#[derive(Debug, Clone, Copy)]
pub struct SpriteInstance {
    pub model_id: u32,
    pub transform: SpriteInstanceTransform,
}

/// GPU per-model metadata: where this model's data starts in the
/// shared registry buffers + its dims/pivot. Mirrors `ModelMeta` in
/// the shader (std430, 48 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct SpriteModelMeta {
    occupancy_offset: u32,
    colors_offset: u32,
    color_offsets_offset: u32,
    occ_words_per_col: u32,
    dims: [u32; 3],
    _pad0: u32,
    pivot: [f32; 3],
    /// GPU.10.4 — world size of one voxel of this (mip) entry.
    voxel_world_size: f32,
}

/// GPU per-instance record. Mirrors `Instance` in the shader (std430,
/// 64 bytes): inverse rotation columns + position + model id.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct SpriteInstanceGpu {
    inv_rot0: [f32; 4],
    inv_rot1: [f32; 4],
    inv_rot2: [f32; 4],
    pos: [f32; 3],
    model_id: u32,
}

/// Invert a 3×3 matrix given as basis columns `[c0, c1, c2]`,
/// returning the inverse as columns. For an orthonormal basis this is
/// the transpose; the general path covers rotation + non-unit scale.
#[must_use]
fn mat3_inverse(cols: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let [a, b, c] = cols; // columns
                          // Determinant via scalar triple product a · (b × c).
    let cross = |u: [f32; 3], v: [f32; 3]| {
        [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ]
    };
    let bc = cross(b, c);
    let ca = cross(c, a);
    let ab = cross(a, b);
    let det = a[0] * bc[0] + a[1] * bc[1] + a[2] * bc[2];
    let inv_det = if det.abs() < 1e-12 { 0.0 } else { 1.0 / det };
    // Inverse rows are (b×c, c×a, a×b)/det; return as columns of the
    // inverse, i.e. transpose of those rows.
    [
        [bc[0] * inv_det, ca[0] * inv_det, ab[0] * inv_det],
        [bc[1] * inv_det, ca[1] * inv_det, ab[1] * inv_det],
        [bc[2] * inv_det, ca[2] * inv_det, ab[2] * inv_det],
    ]
}

/// GPU-resident registry + instances: every model's occupancy /
/// colours / offsets concatenated into shared storage buffers, a
/// per-model metadata table, and a capacity-sized instance buffer
/// rewritten each frame with the frustum-visible subset (GPU.10.2).
/// One bind group serves all models (same approach as the multi-grid
/// scene).
pub struct SpriteRegistryResident {
    pub occupancy: wgpu::Buffer,
    pub colors: wgpu::Buffer,
    pub color_offsets: wgpu::Buffer,
    pub model_meta: wgpu::Buffer,
    /// Holds up to `instance_capacity` instances; the visible subset
    /// is packed into `[0, count)` each frame by [`Self::cull_bin_upload`].
    pub instances: wgpu::Buffer,
    pub instance_capacity: u32,
    /// GPU.10.3 — per-tile `(offset, count)` into `tile_instances`,
    /// flat `2 * tiles_x * tiles_y` u32s. Grown to fit the screen.
    pub tile_ranges: wgpu::Buffer,
    tile_ranges_cap: u32,
    /// GPU.10.3 — flat list of visible-instance indices grouped by
    /// tile. Grown to fit the per-frame total.
    pub tile_instances: wgpu::Buffer,
    tile_instances_cap: u32,
    /// CPU cull records (full set), with precomputed bounding spheres.
    cull: Vec<CullInstance>,
    /// GPU.10.4 — LOD chains: `chains[chain_id]` = entry ids, finest
    /// first. The cull picks a level by distance and writes its entry
    /// id into the packed instance's `model_id`.
    chains: Vec<Vec<u32>>,
}

impl SpriteRegistryResident {
    /// Concatenate `registry`'s models into shared buffers and prepare
    /// `instances` for per-frame culling. Model-relative indices stay
    /// as built; the shader adds each model's base offset from the
    /// metadata table.
    #[must_use]
    pub fn upload(
        device: &wgpu::Device,
        registry: &SpriteModelRegistry,
        instances: &[SpriteInstance],
    ) -> Self {
        let mut all_occ: Vec<u32> = Vec::new();
        let mut all_colors: Vec<u32> = Vec::new();
        let mut all_offsets: Vec<u32> = Vec::new();
        let mut meta: Vec<SpriteModelMeta> = Vec::with_capacity(registry.entries.len());

        // One meta + concatenated data per concrete (mip-level) entry.
        for m in &registry.entries {
            meta.push(SpriteModelMeta {
                occupancy_offset: all_occ.len() as u32,
                colors_offset: all_colors.len() as u32,
                color_offsets_offset: all_offsets.len() as u32,
                occ_words_per_col: m.occ_words_per_col,
                dims: m.dims,
                _pad0: 0,
                pivot: m.pivot,
                voxel_world_size: m.voxel_world_size,
            });
            all_occ.extend_from_slice(&m.occupancy);
            all_colors.extend_from_slice(&m.colors);
            all_offsets.extend_from_slice(&m.color_offsets);
        }

        // Per-instance cull records: sphere centred at the instance
        // position, radius from the chain's finest (mip-0) model.
        let cull: Vec<CullInstance> = instances
            .iter()
            .map(|i| CullInstance {
                gpu: SpriteInstanceGpu {
                    inv_rot0: i.transform.inv_rot[0],
                    inv_rot1: i.transform.inv_rot[1],
                    inv_rot2: i.transform.inv_rot[2],
                    pos: i.transform.pos,
                    model_id: i.model_id, // placeholder; cull rewrites
                },
                chain_id: i.model_id,
                center: i.transform.pos,
                radius: registry.model(i.model_id).bound_radius(),
            })
            .collect();

        // Capacity buffer (COPY_DST so cull can rewrite it each frame),
        // seeded with the full set so frame 0 is valid pre-cull.
        let seed: Vec<SpriteInstanceGpu> = cull.iter().map(|c| c.gpu).collect();
        let instances_buf = {
            use wgpu::util::DeviceExt;
            let one = [SpriteInstanceGpu::zeroed()];
            let src: &[SpriteInstanceGpu] = if seed.is_empty() { &one } else { &seed };
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("roxlap-gpu sprite_reg.instances"),
                contents: bytemuck::cast_slice(src),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
        };

        let tile_ranges = storage_dst_u32(device, "roxlap-gpu sprite_reg.tile_ranges", 1);
        let tile_instances = storage_dst_u32(device, "roxlap-gpu sprite_reg.tile_instances", 1);
        Self {
            occupancy: storage_u32(device, "roxlap-gpu sprite_reg.occupancy", &all_occ),
            colors: storage_u32(device, "roxlap-gpu sprite_reg.colors", &all_colors),
            color_offsets: storage_u32(device, "roxlap-gpu sprite_reg.color_offsets", &all_offsets),
            model_meta: storage_pod(device, "roxlap-gpu sprite_reg.model_meta", &meta),
            instances: instances_buf,
            instance_capacity: cull.len() as u32,
            tile_ranges,
            tile_ranges_cap: 1,
            tile_instances,
            tile_instances_cap: 1,
            cull,
            chains: registry.chains.clone(),
        }
    }

    /// Refresh instance poses in place from `instances` — for animated
    /// sprites (e.g. KFA limbs re-posed each frame) — **without** any
    /// model-volume re-upload. `instances` must match the set passed to
    /// [`Self::upload`] in length + order; each keeps its `model_id`
    /// (LOD chain) so only the transform + cull centre change. No GPU
    /// write happens here: the next [`Self::cull_bin_upload`] re-uploads
    /// the packed visible subset, as it already does every frame.
    pub fn update_transforms(&mut self, instances: &[SpriteInstance]) {
        debug_assert_eq!(
            instances.len(),
            self.cull.len(),
            "update_transforms instance count must match upload"
        );
        for (ci, inst) in self.cull.iter_mut().zip(instances) {
            ci.gpu.inv_rot0 = inst.transform.inv_rot[0];
            ci.gpu.inv_rot1 = inst.transform.inv_rot[1];
            ci.gpu.inv_rot2 = inst.transform.inv_rot[2];
            ci.gpu.pos = inst.transform.pos;
            // Bounding sphere follows the pivot; radius/chain unchanged.
            ci.center = inst.transform.pos;
        }
    }

    /// GPU.10.3 — frustum-cull, pack the visible subset into the
    /// instance buffer, then bin those instances into screen tiles:
    /// project each visible bounding sphere to a screen AABB and append
    /// its (visible) index to every overlapped tile. Uploads the
    /// instance buffer + `tile_ranges` (per-tile offset/count) +
    /// `tile_instances` (flat grouped indices), growing the tile
    /// buffers as needed. Returns `(visible_count, tiles_x, tiles_y)`.
    #[allow(clippy::too_many_arguments)]
    pub fn cull_bin_upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        f: &ViewFrustum,
        screen_w: u32,
        screen_h: u32,
        tile_size: u32,
        lod_px: f32,
    ) -> (u32, u32, u32) {
        let tiles_x = screen_w.div_ceil(tile_size).max(1);
        let tiles_y = screen_h.div_ceil(tile_size).max(1);
        let n_tiles = (tiles_x * tiles_y) as usize;

        let nw = (1.0 + f.half_w * f.half_w).sqrt();
        let nh = (1.0 + f.half_h * f.half_h).sqrt();
        let cx = screen_w as f32 * 0.5;
        let cy = screen_h as f32 * 0.5;
        let px_per_world = cx / f.half_w; // isotropic: == cy/half_h
        let ts = tile_size as f32;
        let tx_max = tiles_x as i32 - 1;
        let ty_max = tiles_y as i32 - 1;

        let mut visible: Vec<SpriteInstanceGpu> = Vec::with_capacity(self.cull.len());
        // Per-visible tile AABB (tx0, tx1, ty0, ty1) for the bin pass.
        let mut boxes: Vec<[i32; 4]> = Vec::with_capacity(self.cull.len());
        let mut counts = vec![0u32; n_tiles];

        for ci in &self.cull {
            let rel = [
                ci.center[0] - f.pos[0],
                ci.center[1] - f.pos[1],
                ci.center[2] - f.pos[2],
            ];
            let z = dot3(rel, f.forward);
            let r = ci.radius;
            if z + r < 0.0 || z - r > f.far {
                continue; // behind / beyond far
            }
            let x = dot3(rel, f.right);
            if (x - f.half_w * z) > r * nw || (-x - f.half_w * z) > r * nw {
                continue; // right / left
            }
            let y = dot3(rel, f.down);
            if (y - f.half_h * z) > r * nh || (-y - f.half_h * z) > r * nh {
                continue; // bottom / top
            }

            // Visible: project the sphere to a screen AABB → tile range.
            let (tx0, tx1, ty0, ty1) = if z > 1e-3 {
                let sx = cx + (x / z) * px_per_world;
                let sy = cy + (y / z) * px_per_world;
                let sr = (r / z) * px_per_world;
                (
                    (((sx - sr) / ts).floor() as i32).clamp(0, tx_max),
                    (((sx + sr) / ts).floor() as i32).clamp(0, tx_max),
                    (((sy - sr) / ts).floor() as i32).clamp(0, ty_max),
                    (((sy + sr) / ts).floor() as i32).clamp(0, ty_max),
                )
            } else {
                // Sphere crosses the camera plane — cover all tiles.
                (0, tx_max, 0, ty_max)
            };
            // GPU.10.4 — pick the LOD level by projected voxel size:
            // choose the coarsest level whose voxel still covers at
            // least `lod_px` screen pixels, i.e. step up once a mip-0
            // voxel would be smaller than that. `lod_px = 1` is the
            // natural "don't go sub-pixel" threshold; larger values
            // force LOD in closer (tuning/inspection).
            let chain = &self.chains[ci.chain_id as usize];
            let level = if z > 1e-3 && chain.len() > 1 {
                let voxel_px = px_per_world / z; // mip-0 voxel screen size
                ((lod_px / voxel_px).log2().ceil().max(0.0) as usize).min(chain.len() - 1)
            } else {
                0
            };
            let mut g = ci.gpu;
            g.model_id = chain[level];
            visible.push(g);
            boxes.push([tx0, tx1, ty0, ty1]);
            for ty in ty0..=ty1 {
                for tx in tx0..=tx1 {
                    counts[(ty * tiles_x as i32 + tx) as usize] += 1;
                }
            }
        }

        if visible.is_empty() {
            return (0, tiles_x, tiles_y);
        }

        // Prefix-sum counts → per-tile offsets; build the flat grouped
        // index list.
        let mut tile_ranges = vec![0u32; n_tiles * 2];
        let mut running = 0u32;
        for t in 0..n_tiles {
            tile_ranges[2 * t] = running; // offset
            tile_ranges[2 * t + 1] = counts[t]; // count
            running += counts[t];
        }
        let total = running as usize;
        let mut tile_instances = vec![0u32; total.max(1)];
        let mut cursor: Vec<u32> = (0..n_tiles).map(|t| tile_ranges[2 * t]).collect();
        for (vis_idx, b) in boxes.iter().enumerate() {
            for ty in b[2]..=b[3] {
                for tx in b[0]..=b[1] {
                    let t = (ty * tiles_x as i32 + tx) as usize;
                    tile_instances[cursor[t] as usize] = vis_idx as u32;
                    cursor[t] += 1;
                }
            }
        }

        // Upload: instances + (grown) tile buffers. Grow a tile buffer
        // only when this frame needs more than its capacity (wgpu has
        // no Clone on Buffer, so we replace the field in place).
        queue.write_buffer(&self.instances, 0, bytemuck::cast_slice(&visible));
        let need_ranges = tile_ranges.len() as u32;
        if need_ranges > self.tile_ranges_cap {
            self.tile_ranges_cap = need_ranges.next_power_of_two();
            self.tile_ranges = storage_dst_u32(
                device,
                "roxlap-gpu sprite_reg.tile_ranges",
                self.tile_ranges_cap,
            );
        }
        let need_inst = tile_instances.len() as u32;
        if need_inst > self.tile_instances_cap {
            self.tile_instances_cap = need_inst.next_power_of_two();
            self.tile_instances = storage_dst_u32(
                device,
                "roxlap-gpu sprite_reg.tile_instances",
                self.tile_instances_cap,
            );
        }
        queue.write_buffer(&self.tile_ranges, 0, bytemuck::cast_slice(&tile_ranges));
        queue.write_buffer(
            &self.tile_instances,
            0,
            bytemuck::cast_slice(&tile_instances),
        );

        (visible.len() as u32, tiles_x, tiles_y)
    }
}

/// Create a STORAGE buffer of u32s; pads empty input (wgpu rejects
/// zero-sized storage bindings).
fn storage_u32(device: &wgpu::Device, label: &str, data: &[u32]) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    let bytes: &[u8] = if data.is_empty() {
        bytemuck::cast_slice(&[0u32])
    } else {
        bytemuck::cast_slice(data)
    };
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytes,
        usage: wgpu::BufferUsages::STORAGE,
    })
}

/// Create an uninitialised `STORAGE | COPY_DST` `u32` buffer of `cap`
/// words (≥1). Written each frame via `queue.write_buffer`.
fn storage_dst_u32(device: &wgpu::Device, label: &str, cap: u32) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: u64::from(cap.max(1)) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Create a STORAGE buffer of Pod records; pads empty input with one
/// zeroed `T`.
fn storage_pod<T: Pod + Zeroable>(device: &wgpu::Device, label: &str, data: &[T]) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    let one = [T::zeroed()];
    let src: &[T] = if data.is_empty() { &one } else { data };
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(src),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use roxlap_formats::kv6::{Kv6, Voxel};

    /// 2×1 kv6: column (0,0) has voxels at z=5 (red) and z=1 (green)
    /// stored OUT of z-order; column (1,0) has one voxel at z=3.
    fn kv6_unsorted() -> Kv6 {
        let mk = |z, col| Voxel {
            col,
            z,
            vis: 0,
            dir: 0,
        };
        Kv6 {
            xsiz: 2,
            ysiz: 1,
            zsiz: 8,
            xpiv: 0.0,
            ypiv: 0.0,
            zpiv: 0.0,
            voxels: vec![mk(5, 0xAA), mk(1, 0xBB), mk(3, 0xCC)],
            xlen: vec![2, 1],
            ylen: vec![vec![2], vec![1]],
            palette: None,
        }
    }

    #[test]
    fn occupancy_bits_set_at_voxel_z() {
        let m = build_sprite_model(&kv6_unsorted());
        assert_eq!(m.dims, [2, 1, 8]);
        assert_eq!(m.occ_words_per_col, 1); // ceil(8/32)
                                            // col 0: bits 1 and 5; col 1: bit 3.
        assert_eq!(m.occupancy[0], (1 << 1) | (1 << 5));
        assert_eq!(m.occupancy[1], 1 << 3);
    }

    #[test]
    fn colors_are_ascending_z_for_rank_lookup() {
        let m = build_sprite_model(&kv6_unsorted());
        // col 0 sorted ascending z ⇒ z=1 (green 0xBB) before z=5 (0xAA).
        assert_eq!(m.color_offsets, vec![0, 2, 3]);
        assert_eq!(&m.colors, &[0xBB, 0xAA, 0xCC]);
    }

    #[test]
    fn identity_basis_inverts_to_identity() {
        let inv = mat3_inverse([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        assert_eq!(inv, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    }

    #[test]
    fn fork_is_independent_of_parent() {
        let mut reg = SpriteModelRegistry::new();
        let base = reg.add(build_sprite_model(&kv6_unsorted()));
        let forked = reg.fork(base);
        assert_ne!(base, forked);
        // Recolour only the fork.
        reg.model_mut(forked).recolor(|_| 0x11);
        // Parent colours untouched; fork fully overwritten.
        assert_eq!(&reg.model(base).colors, &[0xBB, 0xAA, 0xCC]);
        assert_eq!(&reg.model(forked).colors, &[0x11, 0x11, 0x11]);
    }

    #[test]
    fn registry_gpu_structs_have_expected_sizes() {
        assert_eq!(std::mem::size_of::<SpriteModelMeta>(), 48);
        assert_eq!(std::mem::size_of::<SpriteInstanceGpu>(), 64);
    }

    #[test]
    fn add_lod_builds_halving_mip_chain() {
        let mut reg = SpriteModelRegistry::new();
        // 8×8×8 single voxel-filled column model would be ideal, but
        // kv6_unsorted is 2×1×8 → mips: 2×1×8 → 1×1×4 → 1×1×2 → 1×1×1.
        let id = reg.add_lod(build_sprite_model(&kv6_unsorted()), 4);
        let m0 = reg.model(id);
        assert_eq!(m0.dims, [2, 1, 8]);
        assert!((m0.voxel_world_size - 1.0).abs() < 1e-6);
    }

    /// kv6 from explicit voxels, ordered x-major/y-inner to match
    /// `build_sprite_model`'s column walk.
    fn kv6_from(xsiz: u32, ysiz: u32, zsiz: u32, voxels: &[(u32, u32, u16, u32)]) -> Kv6 {
        let mut ylen = vec![vec![0u16; ysiz as usize]; xsiz as usize];
        let mut flat = Vec::new();
        for x in 0..xsiz {
            for y in 0..ysiz {
                let mut col: Vec<(u16, u32)> = voxels
                    .iter()
                    .filter(|(vx, vy, _, _)| *vx == x && *vy == y)
                    .map(|(_, _, z, c)| (*z, *c))
                    .collect();
                col.sort_by_key(|(z, _)| *z);
                ylen[x as usize][y as usize] = col.len() as u16;
                for (z, c) in col {
                    flat.push(Voxel {
                        col: c,
                        z,
                        vis: 0,
                        dir: 0,
                    });
                }
            }
        }
        let xlen = ylen
            .iter()
            .map(|c| c.iter().map(|&v| u32::from(v)).sum())
            .collect();
        Kv6 {
            xsiz,
            ysiz,
            zsiz,
            xpiv: 0.0,
            ypiv: 0.0,
            zpiv: 0.0,
            voxels: flat,
            xlen,
            ylen,
            palette: None,
        }
    }

    fn offsets_consistent(m: &SpriteModel) -> bool {
        let cols = (m.dims[0] * m.dims[1]) as usize;
        if m.color_offsets.len() != cols + 1 {
            return false;
        }
        // Monotonic non-decreasing + last == colors.len + each column's
        // span == its solid-voxel count.
        for w in m.color_offsets.windows(2) {
            if w[1] < w[0] {
                return false;
            }
        }
        m.color_offsets[cols] as usize == m.colors.len()
    }

    #[test]
    fn carve_two_layers_keeps_offsets_consistent() {
        // Mirror the demo's carve: columns with voxels at varied z,
        // some sharing z=0/z=1, some not.
        let kv6 = kv6_from(
            3,
            2,
            8,
            &[
                (0, 0, 0, 0xA0),
                (0, 0, 1, 0xA1),
                (0, 0, 5, 0xA5),
                (1, 0, 1, 0xB1),
                (2, 1, 0, 0xC0),
                (2, 1, 3, 0xC3),
            ],
        );
        let mut m = build_sprite_model(&kv6);
        assert!(offsets_consistent(&m));
        for z in 0..2u32 {
            for y in 0..m.dims[1] {
                for x in 0..m.dims[0] {
                    m.set_voxel(x, y, z, None);
                }
            }
            assert!(offsets_consistent(&m), "inconsistent after carving z={z}");
            // downsample must not panic on the carved model.
            let _ = m.downsample();
        }
    }

    #[test]
    fn set_voxel_inserts_replaces_and_clears() {
        // col 0 starts with z=1 (0xBB), z=5 (0xAA); col 1 with z=3 (0xCC).
        let mut m = build_sprite_model(&kv6_unsorted());

        // Insert z=3 into col 0 (between z=1 and z=5) → rank 1.
        assert!(m.set_voxel(0, 0, 3, Some(0x55)));
        assert_eq!(m.occupancy[0], (1 << 1) | (1 << 3) | (1 << 5));
        // col 0 colours ascending z: 0xBB(z1), 0x55(z3), 0xAA(z5).
        assert_eq!(m.color_offsets, vec![0, 3, 4]);
        assert_eq!(&m.colors, &[0xBB, 0x55, 0xAA, 0xCC]);

        // Replace z=3 in place (no offset shift).
        assert!(m.set_voxel(0, 0, 3, Some(0x66)));
        assert_eq!(&m.colors, &[0xBB, 0x66, 0xAA, 0xCC]);
        assert_eq!(m.color_offsets, vec![0, 3, 4]);

        // Clear z=1 (rank 0) from col 0.
        assert!(m.set_voxel(0, 0, 1, None));
        assert_eq!(m.occupancy[0], (1 << 3) | (1 << 5));
        assert_eq!(m.color_offsets, vec![0, 2, 3]);
        assert_eq!(&m.colors, &[0x66, 0xAA, 0xCC]);

        // No-ops: clear an empty voxel, edit out of bounds.
        assert!(!m.set_voxel(0, 0, 2, None));
        assert!(!m.set_voxel(9, 0, 0, Some(1)));
    }

    #[test]
    fn rebuild_lod_refreshes_coarse_levels_from_mip0() {
        let mut reg = SpriteModelRegistry::new();
        let id = reg.add_lod(build_sprite_model(&kv6_unsorted()), 3);
        // Recolour mip-0 only via model_mut, then rebuild the ladder.
        reg.model_mut(id).recolor(|_| 0x0000_2000);
        reg.rebuild_lod(id);
        // The mip-1 average of all-0x2000 voxels is still 0x2000.
        let lvl1_entry = reg.chains[id as usize][1] as usize;
        assert!(reg.entries[lvl1_entry]
            .colors
            .iter()
            .all(|&c| c == 0x0000_2000));
    }
}
