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
use roxlap_formats::color::Rgb;
use roxlap_formats::kv6::Kv6;
use roxlap_formats::material::material_for_color;
use roxlap_formats::sprite::Sprite;
use roxlap_formats::voxel_clip::{DecodedClip, VoxelFrame};

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
    /// Per-voxel surface-normal index (`Kv6::Voxel::dir`, 0..256),
    /// parallel to [`colors`](Self::colors). The GPU sprite shader uses
    /// it to index the per-instance `kv6colmul` lighting table, matching
    /// the CPU rasteriser's normal-based shading.
    pub dirs: Vec<u32>,
    /// Prefix sums: `color_offsets[col]` is the first colour index of
    /// column `col`; length `mx * my + 1`.
    pub color_offsets: Vec<u32>,
    /// Per-voxel material id (TV.3), parallel to [`colors`](Self::colors).
    /// **Empty** means the model has no per-voxel materials — every voxel
    /// uses the instance's uniform material (the TV.1/TV.2 path). A non-empty
    /// array gives mixed-material models (opaque frame + glass). Built by
    /// [`build_sprite_model_with_materials`].
    pub materials: Vec<u8>,
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
    build_sprite_model_inner(kv6, &[])
}

/// Build the DDA volume from a KV6, classifying each voxel into a per-voxel
/// **material id** by colour (TV.3 mixed models) via `material_map`
/// (`(rgb, material_id)` pairs; see
/// [`material_for_color`]).
/// An empty map produces a model with no per-voxel materials (identical to
/// [`build_sprite_model`]).
///
/// # Panics
/// As [`build_sprite_model`].
#[must_use]
pub fn build_sprite_model_with_materials(kv6: &Kv6, material_map: &[(Rgb, u8)]) -> SpriteModel {
    build_sprite_model_inner(kv6, material_map)
}

fn build_sprite_model_inner(kv6: &Kv6, material_map: &[(Rgb, u8)]) -> SpriteModel {
    let (mx, my, mz) = (kv6.xsiz, kv6.ysiz, kv6.zsiz);
    let occ_words_per_col = mz.div_ceil(32).max(1);
    let cols = (mx * my) as usize;
    let want_mats = !material_map.is_empty();

    let mut occupancy = vec![0u32; cols * occ_words_per_col as usize];
    let mut color_offsets = vec![0u32; cols + 1];
    let mut colors: Vec<u32> = Vec::with_capacity(kv6.voxels.len());
    let mut dirs: Vec<u32> = Vec::with_capacity(kv6.voxels.len());
    let mut materials: Vec<u8> = if want_mats {
        Vec::with_capacity(kv6.voxels.len())
    } else {
        Vec::new()
    };

    // Pass 1 — consume voxels in KV6 storage order (x-outer / y-inner)
    // into per-column buckets keyed by `col = x + y*mx`. Each entry is
    // `(z, colour, normal-dir)`.
    let mut buckets: Vec<Vec<(u16, u32, u8)>> = vec![Vec::new(); cols];
    let mut voxel_iter = kv6.voxels.iter();
    for x in 0..mx {
        for y in 0..my {
            let col = (x + y * mx) as usize;
            let count = kv6.ylen[x as usize][y as usize];
            for _ in 0..count {
                let v = voxel_iter.next().expect("KV6 ylen / voxels.len mismatch");
                buckets[col].push((v.z, v.col, v.dir));
            }
        }
    }

    // Pass 2 — emit in COLUMN-INDEX order so `color_offsets` is a true
    // monotonic prefix sum (the shader indexes by `col` either way, but
    // structural edits / mip rebuilds rely on monotonic offsets). Each
    // column's voxels sorted ascending z for the popcount-rank lookup.
    for (col, bucket) in buckets.iter_mut().enumerate() {
        color_offsets[col] = colors.len() as u32;
        bucket.sort_by_key(|(z, _, _)| *z);
        for &(z, col_rgba, dir) in bucket.iter() {
            let z = u32::from(z);
            let base = col * occ_words_per_col as usize + (z >> 5) as usize;
            occupancy[base] |= 1u32 << (z & 31);
            colors.push(col_rgba);
            dirs.push(u32::from(dir));
            if want_mats {
                materials.push(material_for_color(material_map, col_rgba));
            }
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
        dirs,
        materials,
        voxel_world_size: 1.0,
    }
}

/// Build a [`SpriteModel`] directly from a decoded voxel-clip frame
/// (VCL.2). The [`VoxelFrame`] dense-column layout is byte-for-byte the
/// [`SpriteModel`] layout that [`build_sprite_model`] produces, so this is
/// a field move — no per-column bucket-sort. `dirs` is the frame's
/// surface-normal LUT indices (from [`DecodedClip::dirs`]), parallel to
/// `frame.colors`.
///
/// # Panics
/// In debug, if `dirs.len() != frame.colors.len()` or the field shapes
/// don't match `dims` (the same invariants [`build_sprite_model`] upholds).
#[must_use]
pub fn sprite_model_from_voxel_frame(
    frame: &VoxelFrame,
    dirs: &[u32],
    dims: [u32; 3],
    pivot: [f32; 3],
    voxel_world_size: f32,
) -> SpriteModel {
    sprite_model_from_voxel_frame_with_materials(frame, dirs, dims, pivot, voxel_world_size, &[])
}

/// Like [`sprite_model_from_voxel_frame`] but classifies each voxel into a
/// per-voxel **material id** by colour (TV.3 mixed models) via `material_map`
/// (`(rgb, material_id)` pairs). An empty map produces a model with no
/// per-voxel materials (identical to [`sprite_model_from_voxel_frame`]).
///
/// # Panics
/// As [`sprite_model_from_voxel_frame`].
#[must_use]
pub fn sprite_model_from_voxel_frame_with_materials(
    frame: &VoxelFrame,
    dirs: &[u32],
    dims: [u32; 3],
    pivot: [f32; 3],
    voxel_world_size: f32,
    material_map: &[(Rgb, u8)],
) -> SpriteModel {
    let occ_words_per_col = dims[2].div_ceil(32).max(1);
    let cols = (dims[0] * dims[1]) as usize;
    debug_assert_eq!(frame.occupancy.len(), cols * occ_words_per_col as usize);
    debug_assert_eq!(frame.color_offsets.len(), cols + 1);
    debug_assert_eq!(dirs.len(), frame.colors.len());
    // Per-voxel materials are parallel to `colors` (popcount-rank order), so
    // classify the frame's colour run directly — no re-index needed.
    let materials: Vec<u8> = if material_map.is_empty() {
        Vec::new()
    } else {
        frame
            .colors
            .iter()
            .map(|&c| material_for_color(material_map, c))
            .collect()
    };
    SpriteModel {
        dims,
        occ_words_per_col,
        pivot,
        occupancy: frame.occupancy.clone(),
        colors: frame.colors.clone(),
        dirs: dirs.to_vec(),
        color_offsets: frame.color_offsets.clone(),
        materials,
        voxel_world_size,
    }
}

/// Build the [`SpriteModel`] for frame `frame` of a decoded clip — the
/// per-frame model uploaded into a flipbook chain (VCL.2).
///
/// # Panics
/// If `frame` is out of range, or the frame fails the layout invariants.
#[must_use]
pub fn sprite_model_from_clip_frame(clip: &DecodedClip, frame: usize) -> SpriteModel {
    sprite_model_from_clip_frame_with_materials(clip, frame, &[])
}

/// Like [`sprite_model_from_clip_frame`] but classifies the frame's voxels
/// into per-voxel material ids by colour (TV.3 mixed models) via
/// `material_map`. An empty map is identical to [`sprite_model_from_clip_frame`].
///
/// # Panics
/// If `frame` is out of range, or the frame fails the layout invariants.
#[must_use]
pub fn sprite_model_from_clip_frame_with_materials(
    clip: &DecodedClip,
    frame: usize,
    material_map: &[(Rgb, u8)],
) -> SpriteModel {
    sprite_model_from_voxel_frame_with_materials(
        &clip.frames[frame],
        &clip.dirs[frame],
        clip.dims,
        clip.pivot,
        clip.voxel_world_size,
        material_map,
    )
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
    /// Longest model→world basis column length (PS.1) — `1.0` for the
    /// orthonormal poses every pre-PS caller uses. The CPU cull
    /// multiplies the model's unit-basis [`SpriteModel::bound_radius`]
    /// by it (exact for rotation × uniform-or-per-axis scale; a
    /// sheared basis can still exceed it, which nothing produces
    /// today). Rides the former std430 pad slot, so the GPU layout is
    /// unchanged.
    pub max_scale: f32,
}

impl SpriteInstanceTransform {
    /// Build from a sprite pose. `s/h/f` are the model→world basis
    /// columns; we invert them so the shader can map world→local, and
    /// keep the longest column length for cull-sphere / LOD scaling.
    #[must_use]
    pub fn from_sprite(sprite: &Sprite) -> Self {
        let inv = mat3_inverse([sprite.s, sprite.h, sprite.f]);
        let len = |c: [f32; 3]| (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
        Self {
            inv_rot: [
                [inv[0][0], inv[0][1], inv[0][2], 0.0],
                [inv[1][0], inv[1][1], inv[1][2], 0.0],
                [inv[2][0], inv[2][1], inv[2][2], 0.0],
            ],
            pos: sprite.p,
            max_scale: len(sprite.s).max(len(sprite.h)).max(len(sprite.f)),
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

    /// Like [`Self::model`] but returns `None` for an out-of-range or
    /// tombstoned (emptied) chain instead of panicking — the guarded form
    /// for public primitives handed an arbitrary `chain_id`.
    #[must_use]
    pub fn model_checked(&self, id: u32) -> Option<&SpriteModel> {
        let entry = *self.chains.get(id as usize)?.first()?;
        self.entries.get(entry as usize)
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

    /// Free chain `chain_id`'s voxel data **in place**: replace each of
    /// its LOD entries with [`SpriteModel::empty`] and clear the chain.
    /// Entry ids and every other `model_id` are **preserved** (the chain
    /// becomes empty, its entries become placeholders), so no id remap is
    /// needed and the resident registry's entry alignment stays intact.
    ///
    /// This is safe to pair with the resident side because
    /// [`SpriteRegistryResident::remove_model`] tombstones the same
    /// entries (`dead[e]`) and [`compact`](SpriteRegistryResident::compact)
    /// reads only live entries — so the resident never touches the empty
    /// placeholders left here. Call `remove_model` (resident) **before**
    /// this so those tombstones are set. No-op if `chain_id` is out of
    /// range or already removed.
    pub fn remove(&mut self, chain_id: u32) {
        let Some(entries) = self.chains.get(chain_id as usize) else {
            return;
        };
        // Clone the small id list so we can mutate `entries` while iterating.
        let entries = entries.clone();
        for e in entries {
            self.entries[e as usize] = SpriteModel::empty();
        }
        self.chains[chain_id as usize] = Vec::new(); // tombstone (slot kept)
    }

    /// Whether `chain_id` is a live (registered, not [`removed`](Self::remove))
    /// model. `false` for an out-of-range id or a tombstoned chain.
    #[must_use]
    pub fn is_live(&self, chain_id: u32) -> bool {
        self.chains
            .get(chain_id as usize)
            .is_some_and(|c| !c.is_empty())
    }

    /// Number of LOD chains (distinct `model_id`s). Counts tombstoned
    /// (removed) chains too — ids are never reused, so this is also the
    /// next id that [`Self::add`] / [`Self::add_lod`] will mint.
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
    /// An empty (zero-voxel, zero-extent) placeholder model. Used by
    /// [`SpriteModelRegistry::remove`] to free a removed chain's voxel
    /// data while keeping its entry slot, so ids stay stable. Carries no
    /// occupancy/colours; `color_offsets` is the single-element prefix
    /// `[0]` (`cols + 1` with `cols == 0`), keeping the structural
    /// invariant intact for any code that inspects it.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            dims: [0, 0, 0],
            occ_words_per_col: 1,
            pivot: [0.0, 0.0, 0.0],
            occupancy: Vec::new(),
            colors: Vec::new(),
            dirs: Vec::new(),
            color_offsets: vec![0],
            materials: Vec::new(),
            voxel_world_size: 1.0,
        }
    }

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
                self.colors[idx] = rgba; // replace in place (keeps dir)
            } else {
                self.occupancy[base + zw] |= 1u32 << zb;
                self.colors.insert(idx, rgba);
                // No normal supplied by this API — default to dir 0 (the
                // sole caller, the carve hotkey, only ever clears).
                self.dirs.insert(idx, 0);
                if !self.materials.is_empty() {
                    self.materials.insert(idx, 0); // new voxel → opaque material
                }
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
            self.dirs.remove(idx);
            if !self.materials.is_empty() {
                self.materials.remove(idx);
            }
            for c in &mut self.color_offsets[col + 1..=cols] {
                *c -= 1;
            }
            true
        }
    }

    /// Radius of a bounding sphere centred at the instance position
    /// (the pivot maps there): the farthest bbox corner from the
    /// pivot, in **model units** (a unit basis). The cull multiplies
    /// it by each instance's longest basis column
    /// ([`SpriteInstanceTransform::max_scale`], PS.1), so scaled
    /// instances stay conservatively bounded.
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

        // Reconstruct dense fine voxels (solid flag + colour + normal + TV
        // material).
        let has_mats = !self.materials.is_empty();
        let mut solid = vec![false; (fx * fy * fz) as usize];
        let mut fine = vec![0u32; (fx * fy * fz) as usize];
        let mut fine_dir = vec![0u32; (fx * fy * fz) as usize];
        let mut fine_mat = vec![0u8; (fx * fy * fz) as usize];
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
                        fine_dir[fidx(x, y, z)] = self.dirs[off + seen];
                        if has_mats {
                            fine_mat[fidx(x, y, z)] = self.materials[off + seen];
                        }
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
        let mut dirs: Vec<u32> = Vec::new();
        let mut materials: Vec<u8> = Vec::new();

        // Emit in column-index order (`ccol = cx + cy*nx`), cy outer,
        // so `color_offsets` is a monotonic prefix sum like build's.
        for cy in 0..ny {
            for cx in 0..nx {
                let ccol = (cx + cy * nx) as usize;
                color_offsets[ccol] = colors.len() as u32;
                for cz in 0..nz {
                    let (mut a, mut r, mut g, mut b, mut n) = (0u32, 0u32, 0u32, 0u32, 0u32);
                    // Normals + materials don't average meaningfully — keep
                    // the first solid child's `dir` / material for the coarse
                    // voxel.
                    let mut rep_dir = 0u32;
                    let mut rep_mat = 0u8;
                    for dz in 0..2 {
                        for dy in 0..2 {
                            for dx in 0..2 {
                                let (x, y, z) = (2 * cx + dx, 2 * cy + dy, 2 * cz + dz);
                                if x < fx && y < fy && z < fz && solid[fidx(x, y, z)] {
                                    let c = fine[fidx(x, y, z)];
                                    if n == 0 {
                                        rep_dir = fine_dir[fidx(x, y, z)];
                                        rep_mat = fine_mat[fidx(x, y, z)];
                                    }
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
                        dirs.push(rep_dir);
                        if has_mats {
                            materials.push(rep_mat);
                        }
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
            dirs,
            color_offsets,
            materials,
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
/// Not `Copy` — carries a boxed 256-entry `kv6colmul` table.
#[derive(Clone)]
struct CullInstance {
    /// Instance transform + a placeholder `model_id`; the cull
    /// overwrites `model_id` with the distance-chosen LOD entry.
    gpu: SpriteInstanceGpu,
    /// LOD chain this instance draws (the user-facing `model_id`).
    chain_id: u32,
    center: [f32; 3],
    /// World-space bounding-sphere radius — the cached product
    /// `model_radius × max_scale`, kept so the hot cull loop reads one
    /// float (PS.1).
    radius: f32,
    /// The chain's unit-basis [`SpriteModel::bound_radius`], reseeded
    /// by [`SpriteRegistryResident::set_instance_model`].
    model_radius: f32,
    /// Longest basis column of the current pose (PS.1) — scaled
    /// instances (particles) grow/shrink `radius` and the LOD pick
    /// with it.
    max_scale: f32,
    /// voxlap `kv6colmul[256]` — per-surface-normal colour modulation
    /// for this instance's pose + lighting. Defaults to identity
    /// (`0x0100` in every channel lane → unshaded) until the facade sets
    /// it via [`SpriteRegistryResident::set_instance_colmul`]. Packed
    /// into the `colmul` GPU buffer (in visible order) each frame.
    colmul: Box<[u64; 256]>,
}

/// Identity `kv6colmul` table: every channel lane = `0x0100`, so the
/// shader's `(rgb[c] << 8) * 0x0100 >> 16 == rgb[c]` — i.e. no shading.
fn identity_colmul() -> Box<[u64; 256]> {
    const LANE: u64 = 0x0100;
    let w = LANE | (LANE << 16) | (LANE << 32) | (LANE << 48);
    Box::new([w; 256])
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// PF.10 — everything `cull_bin_upload`'s result depends on besides the
/// registry contents (float fields compared bitwise). Paired with the
/// "registry changed" invalidation (`last_cull = None` in every mutating
/// method): when the key matches the previous frame's, the cull, the
/// binning, and all four buffer uploads are skipped — the buffers already
/// hold exactly this frame's data.
#[derive(Clone, Copy, PartialEq)]
struct CullKey {
    frustum: [u32; 15],
    screen: [u32; 4],
}

impl CullKey {
    fn new(f: &ViewFrustum, screen_w: u32, screen_h: u32, tile_size: u32, lod_px: f32) -> Self {
        let b = |v: f32| v.to_bits();
        Self {
            frustum: [
                b(f.pos[0]),
                b(f.pos[1]),
                b(f.pos[2]),
                b(f.right[0]),
                b(f.right[1]),
                b(f.right[2]),
                b(f.down[0]),
                b(f.down[1]),
                b(f.down[2]),
                b(f.forward[0]),
                b(f.forward[1]),
                b(f.forward[2]),
                b(f.half_w),
                b(f.half_h),
                b(f.far),
            ],
            screen: [screen_w, screen_h, tile_size, lod_px.to_bits()],
        }
    }
}

/// PF.10 — reusable cull/bin workspace (was 6+ fresh `Vec`s per frame).
#[derive(Default)]
struct CullScratch {
    visible: Vec<SpriteInstanceGpu>,
    boxes: Vec<[i32; 4]>,
    colmul: Vec<u32>,
    counts: Vec<u32>,
    tile_ranges: Vec<u32>,
    tile_instances: Vec<u32>,
    cursor: Vec<u32>,
}

/// Build one CPU cull record from a user [`SpriteInstance`]: pack the
/// transform, seed the bounding sphere from the chain's finest model, and
/// start `colmul` at identity. Shared by the full
/// [`SpriteRegistryResident::upload`] and the incremental
/// [`SpriteRegistryResident::append_instances`].
fn make_cull(registry: &SpriteModelRegistry, i: &SpriteInstance) -> CullInstance {
    let model_radius = registry.model(i.model_id).bound_radius();
    CullInstance {
        gpu: SpriteInstanceGpu {
            inv_rot0: i.transform.inv_rot[0],
            inv_rot1: i.transform.inv_rot[1],
            inv_rot2: i.transform.inv_rot[2],
            pos: i.transform.pos,
            model_id: i.model_id, // placeholder; cull rewrites per frame
            material: u32::from(i.material),
            alpha_mul: f32::from(i.alpha_mul) / 255.0,
            flags: i.flags,
            tint: i.tint,
        },
        chain_id: i.model_id,
        center: i.transform.pos,
        radius: model_radius * i.transform.max_scale,
        model_radius,
        max_scale: i.transform.max_scale,
        colmul: identity_colmul(),
    }
}

/// Allocate the `instances` capacity buffer (`STORAGE | COPY_DST`) sized
/// for `cap` records (≥1). Left uninitialised — `cull_bin_upload`
/// rewrites it (offset 0) each frame, and `append_instances` seeds the
/// live records after a grow.
fn instances_buffer(device: &wgpu::Device, cap: u32) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("roxlap-gpu sprite_reg.instances"),
        size: u64::from(cap.max(1)) * std::mem::size_of::<SpriteInstanceGpu>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// One sprite instance: a model reference + world pose.
#[derive(Debug, Clone, Copy)]
pub struct SpriteInstance {
    pub model_id: u32,
    pub transform: SpriteInstanceTransform,
    /// Voxel-material id (TV stage): indexes the renderer's global material
    /// palette for this instance's opacity + blend mode. `0` (the default)
    /// is opaque, so an unset instance renders unchanged.
    pub material: u8,
    /// Per-instance alpha multiplier (TV stage), `0..=255` (`255` =
    /// unscaled, the default).
    pub alpha_mul: u8,
    /// XS.4 — sprite shadow flags (`roxlap_formats::sprite` bits 4/5:
    /// `NO_SHADOW_CAST` / `NO_SHADOW_RECEIVE`). `0` (default) ⇒ casts +
    /// receives. Only honoured when the device is sprite-shadow capable.
    pub flags: u32,
    /// Per-instance RGB tint, packed `0x00RRGGBB` (white `0x00FF_FFFF` = no-op).
    pub tint: u32,
}

impl SpriteInstance {
    /// A model reference + pose with the default opaque material
    /// (`material = 0`, `alpha_mul = 255`), shadows on (`flags = 0`), and no
    /// tint (`0x00FF_FFFF`).
    #[must_use]
    pub fn new(model_id: u32, transform: SpriteInstanceTransform) -> Self {
        Self {
            model_id,
            transform,
            material: 0,
            alpha_mul: 255,
            flags: 0,
            tint: 0x00FF_FFFF,
        }
    }
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
    /// TV.3 — 1 if this model has per-voxel materials (`materials_vox` is
    /// populated for it); 0 ⇒ use the instance's uniform material.
    has_vox_materials: u32,
    pivot: [f32; 3],
    /// GPU.10.4 — world size of one voxel of this (mip) entry.
    voxel_world_size: f32,
}

/// GPU per-instance record. Mirrors `Instance` in the shader (std430,
/// 80 bytes): inverse rotation columns + position + model id + the TV
/// material id and per-instance alpha multiplier.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct SpriteInstanceGpu {
    inv_rot0: [f32; 4],
    inv_rot1: [f32; 4],
    inv_rot2: [f32; 4],
    pos: [f32; 3],
    model_id: u32,
    /// TV: material id into the global palette (binding 12).
    material: u32,
    /// TV: per-instance alpha multiplier, normalised to `0..=1`.
    alpha_mul: f32,
    /// XS.4 — sprite shadow flags (mirror of `roxlap_formats::sprite` bits 4/5):
    /// bit4 = NO_SHADOW_CAST, bit5 = NO_SHADOW_RECEIVE. `0` ⇒ casts + receives.
    flags: u32,
    /// Per-instance RGB tint, packed `0x00RRGGBB` (white `0x00FF_FFFF` = no-op).
    tint: u32,
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
    /// Per-voxel surface-normal index, concatenated across models in the
    /// same layout as [`colors`](Self::colors). The shader indexes the
    /// per-instance `kv6colmul` table by it.
    pub dirs: wgpu::Buffer,
    /// Per-voxel material id (TV.3), same layout as [`colors`](Self::colors)
    /// (one u32 per voxel). `0` for models without per-voxel materials; the
    /// per-model `has_vox_materials` flag in `model_meta` says whether to use
    /// it (else the shader falls back to the instance's uniform material).
    pub materials_vox: wgpu::Buffer,
    pub color_offsets: wgpu::Buffer,
    pub model_meta: wgpu::Buffer,
    /// Holds up to `instance_capacity` instances; the visible subset
    /// is packed into `[0, count)` each frame by [`Self::cull_bin_upload`].
    pub instances: wgpu::Buffer,
    pub instance_capacity: u32,
    /// Per-visible-instance `kv6colmul[256]` tables, packed in the same
    /// order as the `instances` buffer each frame (two u32 per u64
    /// entry: lanes 0|1 then 2|3). Sized `instance_capacity * 256 * 2`
    /// u32; rewritten by [`Self::cull_bin_upload`].
    pub colmul: wgpu::Buffer,
    colmul_cap: u32,
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
    /// GPU.12 incremental — CPU mirror of the GPU `model_meta` table, one
    /// per concrete entry. [`Self::update_model`] reads the fixed
    /// occupancy/color_offsets bases from here and rewrites the changed
    /// `colors_offset` on a relocation.
    meta: Vec<SpriteModelMeta>,
    /// GPU.12 incremental — per-entry placement of `colors`/`dirs` in the
    /// shared buffers (drives both; same offsets/ranks). Lets an edit
    /// re-upload one model's data without touching the others.
    colors_alloc: ColorsAllocator,
    /// PF.10 — the (frustum, screen) key + result of the last
    /// `cull_bin_upload`; `None` after any registry mutation. A matching
    /// key skips the whole cull/bin/upload (buffers already current).
    last_cull: Option<(CullKey, (u32, u32, u32))>,
    /// PF.10 — true once ANY per-instance colmul table was set. While
    /// false every table is identity, so the 2 KiB-per-visible-instance
    /// rebuild + upload is skipped; the buffer is identity-filled lazily
    /// instead (`colmul_identity`).
    any_colmul: bool,
    /// PF.10 — whether the whole `colmul` buffer currently holds the
    /// identity pattern (reset on growth).
    colmul_identity: bool,
    /// PF.10 — reusable cull/bin workspace.
    scratch: CullScratch,
    /// Per-entry word length of the dims-fixed `occupancy` and
    /// `color_offsets` arrays, kept so [`Self::update_model`] can assert a
    /// carve never changed dims (which would invalidate the in-place
    /// writes — growing dims is out of scope, handled by a full re-upload).
    occ_lens: Vec<u32>,
    coloff_lens: Vec<u32>,
    /// Used / allocated words of the tightly-concatenated `occupancy`
    /// buffer. `add_model` bump-appends at `occ_used`; when it would pass
    /// `occ_cap` the buffer is grown (with slack) and rebuilt from the
    /// registry. (`colors`/`dirs` track theirs in [`ColorsAllocator`].)
    occ_used: u32,
    occ_cap: u32,
    /// Used / allocated words of the tightly-concatenated `color_offsets`
    /// buffer — same growth scheme as `occ_*`.
    coloff_used: u32,
    coloff_cap: u32,
    /// Allocated record count of the `model_meta` buffer; `add_model`
    /// grows it (with slack) when the entry count passes it.
    meta_cap: u32,
    /// Per-entry tombstone: `true` once its model was removed
    /// ([`Self::remove_model`]). Dead entries keep their `meta` slot (so
    /// entry ids — and the caller's `chain_id`s — stay stable) but their
    /// colours are freed for reuse and they contribute nothing to a
    /// repack / [`Self::compact`]. Parallel to `meta`.
    dead: Vec<bool>,
}

/// Which tightly-concatenated registry buffer [`SpriteRegistryResident::
/// sync_concat`] is operating on.
#[derive(Clone, Copy)]
enum ConcatBuf {
    Occupancy,
    ColorOffsets,
}

/// The model's source array for a given [`ConcatBuf`] — a free fn (not a
/// closure) so the returned borrow keeps `m`'s lifetime.
fn concat_data(m: &SpriteModel, which: ConcatBuf) -> &[u32] {
    match which {
        ConcatBuf::Occupancy => &m.occupancy,
        ConcatBuf::ColorOffsets => &m.color_offsets,
    }
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
        // `occupancy` + `color_offsets` are dims-fixed → tightly
        // concatenated (never grow on a carve). `colors` + `dirs` are
        // variable → laid out by the suballocator with per-slot slack so
        // an incremental edit can rewrite one model in place.
        let entry_lens: Vec<u32> = registry
            .entries
            .iter()
            .map(|m| m.colors.len() as u32)
            .collect();
        let colors_alloc = ColorsAllocator::new(&entry_lens);
        let cap_total = colors_alloc.cap_total();

        let mut all_occ: Vec<u32> = Vec::new();
        let mut all_offsets: Vec<u32> = Vec::new();
        let mut all_colors: Vec<u32> = vec![0; cap_total as usize];
        let mut all_dirs: Vec<u32> = vec![0; cap_total as usize];
        let mut all_materials: Vec<u32> = vec![0; cap_total as usize];
        let mut meta: Vec<SpriteModelMeta> = Vec::with_capacity(registry.entries.len());
        let mut occ_lens: Vec<u32> = Vec::with_capacity(registry.entries.len());
        let mut coloff_lens: Vec<u32> = Vec::with_capacity(registry.entries.len());

        // One meta + placed data per concrete (mip-level) entry.
        for (e, m) in registry.entries.iter().enumerate() {
            let slot = colors_alloc.slot(e);
            meta.push(SpriteModelMeta {
                occupancy_offset: all_occ.len() as u32,
                colors_offset: slot.off,
                color_offsets_offset: all_offsets.len() as u32,
                occ_words_per_col: m.occ_words_per_col,
                dims: m.dims,
                has_vox_materials: u32::from(!m.materials.is_empty()),
                pivot: m.pivot,
                voxel_world_size: m.voxel_world_size,
            });
            occ_lens.push(m.occupancy.len() as u32);
            coloff_lens.push(m.color_offsets.len() as u32);
            all_occ.extend_from_slice(&m.occupancy);
            all_offsets.extend_from_slice(&m.color_offsets);
            let off = slot.off as usize;
            all_colors[off..off + m.colors.len()].copy_from_slice(&m.colors);
            all_dirs[off..off + m.dirs.len()].copy_from_slice(&m.dirs);
            for (i, &mat) in m.materials.iter().enumerate() {
                all_materials[off + i] = u32::from(mat);
            }
        }

        // Per-instance cull records: sphere centred at the instance
        // position, radius from the chain's finest (mip-0) model.
        // `colmul` starts at identity (unshaded) until the facade sets
        // per-instance lighting via `set_instance_colmul`.
        let cull: Vec<CullInstance> = instances.iter().map(|i| make_cull(registry, i)).collect();

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
        // colmul: 256 entries × 2 u32 per visible instance. Sized to the
        // full instance set (worst case all visible); rewritten per frame.
        let colmul_cap = (cull.len() as u32).max(1) * 256 * 2;
        let colmul = storage_dst_u32(device, "roxlap-gpu sprite_reg.colmul", colmul_cap);
        Self {
            occupancy: storage_dst_u32_cap(
                device,
                "roxlap-gpu sprite_reg.occupancy",
                &all_occ,
                all_occ.len() as u32,
            ),
            colors: storage_dst_u32_cap(
                device,
                "roxlap-gpu sprite_reg.colors",
                &all_colors,
                cap_total,
            ),
            dirs: storage_dst_u32_cap(device, "roxlap-gpu sprite_reg.dirs", &all_dirs, cap_total),
            materials_vox: storage_dst_u32_cap(
                device,
                "roxlap-gpu sprite_reg.materials_vox",
                &all_materials,
                cap_total,
            ),
            color_offsets: storage_dst_u32_cap(
                device,
                "roxlap-gpu sprite_reg.color_offsets",
                &all_offsets,
                all_offsets.len() as u32,
            ),
            model_meta: storage_dst_pod(device, "roxlap-gpu sprite_reg.model_meta", &meta),
            instances: instances_buf,
            instance_capacity: cull.len() as u32,
            colmul,
            colmul_cap,
            tile_ranges,
            tile_ranges_cap: 1,
            tile_instances,
            tile_instances_cap: 1,
            cull,
            chains: registry.chains.clone(),
            last_cull: None,
            any_colmul: false,
            colmul_identity: false,
            scratch: CullScratch::default(),
            occ_used: all_occ.len() as u32,
            occ_cap: all_occ.len() as u32,
            coloff_used: all_offsets.len() as u32,
            coloff_cap: all_offsets.len() as u32,
            meta_cap: meta.len() as u32,
            dead: vec![false; meta.len()],
            meta,
            colors_alloc,
            occ_lens,
            coloff_lens,
        }
    }

    /// Number of resident instances (the cull set length).
    #[must_use]
    pub fn instance_count(&self) -> usize {
        self.cull.len()
    }

    /// Append new instances **without** re-uploading any model volume —
    /// the incremental counterpart to [`Self::upload`], for streaming
    /// spawns (asteroids, projectiles, …). Returns the index of the first
    /// appended instance; the block occupies `[base, base + N)`.
    ///
    /// The model volumes are untouched, so every appended instance must
    /// reference a `model_id` (LOD chain) that was already present in the
    /// `registry` passed to [`Self::upload`]. Registering a *new* model
    /// still requires a full [`Self::upload`] (its voxels must be laid
    /// into the shared buffers). `registry` here is only read for the new
    /// instances' bound-sphere radii and must be the resident one.
    ///
    /// The `instances` GPU buffer is only *grown* here (power-of-two,
    /// amortised O(1)); its contents are **not** written. [`Self::
    /// cull_bin_upload`] rewrites the whole visible range from `cull` every
    /// frame before the sprite pass reads it — exactly as for the static
    /// instances — so appending only needs to extend `cull` and ensure
    /// capacity. Writing the buffer here too caused a mid-frame
    /// write-while-in-flight hazard on some drivers (a stray full-screen
    /// flash on append). `colmul` likewise grows lazily in
    /// `cull_bin_upload`. After a removal the capacity is not shrunk.
    pub fn append_instances(
        &mut self,
        device: &wgpu::Device,
        registry: &SpriteModelRegistry,
        instances: &[SpriteInstance],
    ) -> u32 {
        let base = self.cull.len() as u32;
        if instances.is_empty() {
            return base;
        }
        self.last_cull = None; // PF.10 — instance set changed
        for i in instances {
            debug_assert!(
                (i.model_id as usize) < self.chains.len(),
                "append_instances: model_id {} not resident (run upload to register new models)",
                i.model_id
            );
            self.cull.push(make_cull(registry, i));
        }
        let need = self.cull.len() as u32;
        if need > self.instance_capacity {
            // Grow power-of-two and recreate the buffer (the next frame's
            // bind group picks up the new handle). No seed write — the
            // per-frame cull_bin_upload populates it.
            self.instance_capacity = need.next_power_of_two();
            self.instances = instances_buffer(device, self.instance_capacity);
        }
        base
    }

    /// Remove the instance at `index` by swap-remove — O(1), no GPU work
    /// (the next [`Self::cull_bin_upload`] repacks the visible set from
    /// the shrunk cull list). Capacity is retained for reuse.
    ///
    /// Returns `Some(old_last)` when a different instance was moved into
    /// `index` to fill the hole (its index changed from `old_last` to
    /// `index` — callers holding instance handles must fix up that one),
    /// or `None` if `index` was the last element or out of range. Because
    /// this reorders, any [`Self::set_instance_colmul`] table set by
    /// position should be re-applied after a removal.
    pub fn remove_instance(&mut self, index: usize) -> Option<usize> {
        if index >= self.cull.len() {
            return None;
        }
        self.last_cull = None; // PF.10 — instance set changed
        let last = self.cull.len() - 1;
        self.cull.swap_remove(index);
        (index != last).then_some(last)
    }

    /// Set the per-instance `kv6colmul[256]` lighting tables (voxlap's
    /// `update_reflects` output), in the same order/length as the
    /// instances passed to [`Self::upload`]. The next
    /// [`Self::cull_bin_upload`] packs the visible subset to the GPU.
    /// Instances beyond `tables.len()` keep their previous tables.
    pub fn set_instance_colmul(&mut self, tables: &[[u64; 256]]) {
        // PF.10 — leaves the identity fast path for good: from here on the
        // per-visible tables are rebuilt + uploaded each cull.
        self.any_colmul = true;
        self.last_cull = None;
        for (ci, t) in self.cull.iter_mut().zip(tables) {
            ci.colmul.copy_from_slice(t);
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
        self.last_cull = None; // PF.10 — poses changed
        for (ci, inst) in self.cull.iter_mut().zip(instances) {
            ci.gpu.inv_rot0 = inst.transform.inv_rot[0];
            ci.gpu.inv_rot1 = inst.transform.inv_rot[1];
            ci.gpu.inv_rot2 = inst.transform.inv_rot[2];
            ci.gpu.pos = inst.transform.pos;
            // TV: material id + alpha multiplier ride the same coalesced
            // update as the pose (set via the facade's per-instance setters).
            ci.gpu.material = u32::from(inst.material);
            ci.gpu.alpha_mul = f32::from(inst.alpha_mul) / 255.0;
            // XS.4 shadow flags + per-instance RGB tint also ride this flush,
            // so `set_dyn_instance_tint` (and any flag change) takes effect.
            ci.gpu.flags = inst.flags;
            ci.gpu.tint = inst.tint;
            // Bounding sphere follows the pivot and rescales with the
            // pose's longest basis column (PS.1 — scaled particles
            // must not under-cull); the chain is unchanged.
            ci.center = inst.transform.pos;
            ci.max_scale = inst.transform.max_scale;
            ci.radius = ci.model_radius * inst.transform.max_scale;
        }
    }

    /// Repoint instance `idx` at a different LOD chain — the per-frame
    /// **flipbook** step for animated voxel clips (VCL.2). The instance's
    /// transform / colmul are untouched; only which model's volume it
    /// draws changes. The new chain's volume must already be resident
    /// (uploaded via [`Self::add_model`] / [`Self::upload`]); `registry`
    /// is the one those uploads used (so the bounding radius reseeds from
    /// the new model). Like [`Self::update_transforms`], this is a CPU-side
    /// rewrite — the next [`Self::cull_bin_upload`] re-uploads the packed
    /// visible subset, so it costs nothing extra on the GPU. No-op if `idx`
    /// is out of range.
    ///
    /// All frames of a clip share the same `dims`, so a flipbook swap
    /// leaves the bounding radius unchanged; reseeding it anyway keeps the
    /// method correct for arbitrary chain swaps.
    pub fn set_instance_model(
        &mut self,
        registry: &SpriteModelRegistry,
        idx: usize,
        chain_id: u32,
    ) {
        self.last_cull = None; // PF.10 — model binding changed
                               // Guard `chain_id` (the `cull.get_mut` below only covers `idx`): a
                               // public caller could pass an out-of-range / tombstoned chain, which
                               // `registry.model` would index-panic on.
        let Some(model_radius) = registry
            .model_checked(chain_id)
            .map(SpriteModel::bound_radius)
        else {
            return;
        };
        let Some(ci) = self.cull.get_mut(idx) else {
            return;
        };
        ci.chain_id = chain_id;
        ci.gpu.model_id = chain_id; // placeholder; cull rewrites to the LOD entry
        ci.model_radius = model_radius;
        ci.radius = model_radius * ci.max_scale;
    }

    /// GPU.12 incremental — re-upload only the entries of LOD chain
    /// `chain_id` after an in-place edit (carve / recolour) of its model,
    /// **without** rebuilding the whole registry. `registry` must be the
    /// same registry uploaded (same entry ids), with chain `chain_id`'s
    /// entries already edited (`model_mut` + `rebuild_lod`).
    ///
    /// For each entry: occupancy + color_offsets are dims-fixed, so they
    /// are written in place; colors + dirs (variable, parallel) go through
    /// the suballocator — written in place when they fit the slack,
    /// relocated (with a `model_meta` rewrite) when they outgrow it, and
    /// only when the buffer tail overflows are colors/dirs grown + the
    /// whole registry repacked. Instances / cull / colmul are untouched
    /// (a carve never moves an instance or grows its bounds) — that is the
    /// win over [`Self::upload`].
    ///
    /// # Panics (debug)
    /// If an entry's dims changed (occupancy / color_offsets length), which
    /// the in-place path can't absorb — growing dims needs a full
    /// re-upload via [`Self::upload`].
    pub fn update_model(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        registry: &SpriteModelRegistry,
        chain_id: u32,
    ) {
        self.last_cull = None; // PF.10 — model volume changed
        let entries = self.chains[chain_id as usize].clone();
        let mut grew = false;
        for &e in &entries {
            let e = e as usize;
            let m = &registry.entries[e];

            // Dims-fixed arrays: assert unchanged, then write in place.
            debug_assert_eq!(
                m.occupancy.len() as u32,
                self.occ_lens[e],
                "update_model: entry {e} occupancy length changed (dims grew?)"
            );
            debug_assert_eq!(
                m.color_offsets.len() as u32,
                self.coloff_lens[e],
                "update_model: entry {e} color_offsets length changed (dims grew?)"
            );
            queue.write_buffer(
                &self.occupancy,
                u64::from(self.meta[e].occupancy_offset) * 4,
                bytemuck::cast_slice(&m.occupancy),
            );
            queue.write_buffer(
                &self.color_offsets,
                u64::from(self.meta[e].color_offsets_offset) * 4,
                bytemuck::cast_slice(&m.color_offsets),
            );

            // Variable colors/dirs via the suballocator.
            let new_len = m.colors.len() as u32;
            match self.colors_alloc.place(e, new_len) {
                Some(off) => {
                    queue.write_buffer(
                        &self.colors,
                        u64::from(off) * 4,
                        bytemuck::cast_slice(&m.colors),
                    );
                    queue.write_buffer(
                        &self.dirs,
                        u64::from(off) * 4,
                        bytemuck::cast_slice(&m.dirs),
                    );
                    let mats: Vec<u32> = m.materials.iter().map(|&x| u32::from(x)).collect();
                    queue.write_buffer(
                        &self.materials_vox,
                        u64::from(off) * 4,
                        bytemuck::cast_slice(&mats),
                    );
                    if self.meta[e].colors_offset != off {
                        // Relocated — rewrite this entry's meta record.
                        self.meta[e].colors_offset = off;
                        queue.write_buffer(
                            &self.model_meta,
                            (e * std::mem::size_of::<SpriteModelMeta>()) as u64,
                            bytemuck::bytes_of(&self.meta[e]),
                        );
                    }
                }
                None => grew = true,
            }
        }

        // Buffer overflow on at least one entry → grow colors/dirs and
        // repack the WHOLE registry (rare; offsets for every entry move).
        if grew {
            self.grow_and_repack(device, queue, registry);
        }
    }

    /// Grow the `colors`/`dirs` buffers and repack every entry compactly
    /// (with fresh slack) when an [`Self::update_model`] edit overflowed
    /// the buffer tail. Recreates both buffers (the next frame's bind
    /// group picks up the new handles) and rewrites every `model_meta`
    /// `colors_offset`. O(registry) but rare — logged so a growth burst
    /// is visible.
    fn grow_and_repack(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        registry: &SpriteModelRegistry,
    ) {
        self.repack_colors_dirs(device, registry);
        // Every entry's colors_offset moved → rewrite the whole meta table.
        queue.write_buffer(&self.model_meta, 0, bytemuck::cast_slice(&self.meta));
    }

    /// Repack `colors`/`dirs` compactly (with fresh slack) from the full
    /// `registry`, recreating both buffers and updating every CPU
    /// `meta[e].colors_offset`. Does **not** touch the GPU `model_meta`
    /// buffer — the caller writes it ([`Self::grow_and_repack`] writes the
    /// whole table; [`Self::add_model`] writes it once after all entries
    /// are placed). O(registry) but rare — logged so a growth burst is
    /// visible.
    fn repack_colors_dirs(&mut self, device: &wgpu::Device, registry: &SpriteModelRegistry) {
        // Dead (removed) entries collapse to 0 length so they reclaim no
        // space; live entries keep their colours.
        let new_lens: Vec<u32> = registry
            .entries
            .iter()
            .enumerate()
            .map(|(e, m)| {
                if self.dead[e] {
                    0
                } else {
                    m.colors.len() as u32
                }
            })
            .collect();
        self.colors_alloc.repack(&new_lens);
        let cap_total = self.colors_alloc.cap_total();

        let mut all_colors = vec![0u32; cap_total as usize];
        let mut all_dirs = vec![0u32; cap_total as usize];
        let mut all_materials = vec![0u32; cap_total as usize];
        for (e, m) in registry.entries.iter().enumerate() {
            if self.dead[e] {
                self.meta[e].colors_offset = 0;
                continue;
            }
            let off = self.colors_alloc.slot(e).off as usize;
            all_colors[off..off + m.colors.len()].copy_from_slice(&m.colors);
            all_dirs[off..off + m.dirs.len()].copy_from_slice(&m.dirs);
            for (i, &mat) in m.materials.iter().enumerate() {
                all_materials[off + i] = u32::from(mat);
            }
            self.meta[e].colors_offset = off as u32;
        }
        self.colors = storage_dst_u32_cap(
            device,
            "roxlap-gpu sprite_reg.colors",
            &all_colors,
            cap_total,
        );
        self.dirs = storage_dst_u32_cap(device, "roxlap-gpu sprite_reg.dirs", &all_dirs, cap_total);
        self.materials_vox = storage_dst_u32_cap(
            device,
            "roxlap-gpu sprite_reg.materials_vox",
            &all_materials,
            cap_total,
        );
        eprintln!(
            "roxlap-gpu: sprite registry colors/dirs/materials grew + repacked to {cap_total} words"
        );
    }

    /// Append a new model (its full LOD chain) to the resident registry
    /// **without** re-uploading the existing models' volumes — the
    /// incremental counterpart to a full [`Self::upload`], for streaming
    /// in new geometry (unique asteroids, generated meshes).
    ///
    /// Contract (mirrors [`Self::update_model`]): the caller owns the
    /// `SpriteModelRegistry`, has just appended this chain to it (e.g. via
    /// [`SpriteModelRegistry::add_lod`]), and passes the resulting
    /// `chain_id`. The chain's entries must be the registry's newest (ids
    /// `>= ` the resident entry count) — entries are append-only.
    ///
    /// The large `colors`/`dirs`/`occupancy`/`color_offsets` buffers carry
    /// slack and bump-append the new entries in place; a buffer that
    /// overflows is grown (with slack) and rebuilt once from the registry
    /// (amortised O(1) per add). The small `model_meta` table is rewritten
    /// each call. After this, [`Self::append_instances`] can reference the
    /// new `chain_id`.
    pub fn add_model(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        registry: &SpriteModelRegistry,
        chain_id: u32,
    ) {
        self.last_cull = None; // PF.10 — chain set changed
        let entries = registry.chains[chain_id as usize].clone();
        debug_assert_eq!(
            chain_id as usize,
            self.chains.len(),
            "add_model: chains must be appended in order"
        );

        // CPU bookkeeping: assign each new entry a tight occ/coloff offset
        // and an allocator slot for colors/dirs. `need_colors_grow` marks
        // a slot that didn't fit → a colors/dirs repack below.
        let mut need_colors_grow = false;
        for &e in &entries {
            let e = e as usize;
            debug_assert_eq!(
                e,
                self.meta.len(),
                "add_model: entries must be appended in order"
            );
            let m = &registry.entries[e];
            let occ_off = self.occ_used;
            let coloff_off = self.coloff_used;
            self.occ_used += m.occupancy.len() as u32;
            self.coloff_used += m.color_offsets.len() as u32;
            let colors_off = match self.colors_alloc.push(m.colors.len() as u32) {
                Some(off) => off,
                None => {
                    need_colors_grow = true;
                    0 // placeholder; repack assigns the real offset
                }
            };
            self.meta.push(SpriteModelMeta {
                occupancy_offset: occ_off,
                colors_offset: colors_off,
                color_offsets_offset: coloff_off,
                occ_words_per_col: m.occ_words_per_col,
                dims: m.dims,
                has_vox_materials: u32::from(!m.materials.is_empty()),
                pivot: m.pivot,
                voxel_world_size: m.voxel_world_size,
            });
            self.occ_lens.push(m.occupancy.len() as u32);
            self.coloff_lens.push(m.color_offsets.len() as u32);
            self.dead.push(false);
        }
        self.chains.push(entries.clone());

        // occupancy + color_offsets: grow+rebuild on overflow, else write
        // the new tails in place.
        self.sync_concat(device, queue, registry, &entries, ConcatBuf::Occupancy);
        self.sync_concat(device, queue, registry, &entries, ConcatBuf::ColorOffsets);

        // colors/dirs: repack on overflow (rebuilds both + every CPU
        // colors_offset), else write the new entries at their slots.
        if need_colors_grow {
            self.repack_colors_dirs(device, registry);
        } else {
            for &e in &entries {
                let e = e as usize;
                let m = &registry.entries[e];
                let off = u64::from(self.meta[e].colors_offset) * 4;
                queue.write_buffer(&self.colors, off, bytemuck::cast_slice(&m.colors));
                queue.write_buffer(&self.dirs, off, bytemuck::cast_slice(&m.dirs));
                let mats: Vec<u32> = m.materials.iter().map(|&x| u32::from(x)).collect();
                queue.write_buffer(&self.materials_vox, off, bytemuck::cast_slice(&mats));
            }
        }

        // model_meta: grow the record buffer if needed, then rewrite the
        // whole (small) table — covers both new records and any
        // colors_offset relocations from a repack.
        let count = self.meta.len() as u32;
        if count > self.meta_cap {
            self.meta_cap = grow_records(count);
            self.model_meta = storage_dst_pod_cap(
                device,
                "roxlap-gpu sprite_reg.model_meta",
                &self.meta,
                self.meta_cap,
            );
        } else {
            queue.write_buffer(&self.model_meta, 0, bytemuck::cast_slice(&self.meta));
        }
    }

    /// Sync one tightly-concatenated buffer (`occupancy` or
    /// `color_offsets`) after `add_model` appended `new_entries`: if the
    /// used length now exceeds capacity, grow (with slack) and rebuild the
    /// whole buffer from the registry; otherwise write just the appended
    /// tails at their offsets.
    fn sync_concat(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        registry: &SpriteModelRegistry,
        new_entries: &[u32],
        which: ConcatBuf,
    ) {
        let (used, cap) = match which {
            ConcatBuf::Occupancy => (self.occ_used, self.occ_cap),
            ConcatBuf::ColorOffsets => (self.coloff_used, self.coloff_cap),
        };
        if used > cap {
            let new_cap = grow_words(used);
            let all: Vec<u32> = registry
                .entries
                .iter()
                .flat_map(|m| concat_data(m, which).iter().copied())
                .collect();
            let label = match which {
                ConcatBuf::Occupancy => "roxlap-gpu sprite_reg.occupancy",
                ConcatBuf::ColorOffsets => "roxlap-gpu sprite_reg.color_offsets",
            };
            let buf = storage_dst_u32_cap(device, label, &all, new_cap);
            match which {
                ConcatBuf::Occupancy => {
                    self.occupancy = buf;
                    self.occ_cap = new_cap;
                }
                ConcatBuf::ColorOffsets => {
                    self.color_offsets = buf;
                    self.coloff_cap = new_cap;
                }
            }
        } else {
            let target = match which {
                ConcatBuf::Occupancy => &self.occupancy,
                ConcatBuf::ColorOffsets => &self.color_offsets,
            };
            for &e in new_entries {
                let e = e as usize;
                let off = match which {
                    ConcatBuf::Occupancy => self.meta[e].occupancy_offset,
                    ConcatBuf::ColorOffsets => self.meta[e].color_offsets_offset,
                };
                queue.write_buffer(
                    target,
                    u64::from(off) * 4,
                    bytemuck::cast_slice(concat_data(&registry.entries[e], which)),
                );
            }
        }
    }

    /// Number of removed-but-not-yet-compacted models (tombstoned chains).
    /// A caller streams `add_model` / `remove_model` and calls
    /// [`Self::compact`] once this (relative to [`Self::live_model_count`])
    /// crosses a threshold.
    #[must_use]
    pub fn dead_model_count(&self) -> usize {
        self.chains.iter().filter(|c| c.is_empty()).count()
    }

    /// Number of live (non-removed) models.
    #[must_use]
    pub fn live_model_count(&self) -> usize {
        self.chains.iter().filter(|c| !c.is_empty()).count()
    }

    /// Remove a model (tombstone its LOD chain) — the counterpart to
    /// [`Self::add_model`]. O(chain length): marks the chain's entries
    /// dead and frees their `colors`/`dirs` slots for reuse by a later
    /// `add_model`. The `occupancy` / `color_offsets` holes are **not**
    /// reclaimed until [`Self::compact`]; entry ids (and the caller's other
    /// `chain_id`s) stay stable.
    ///
    /// Instances of the removed chain are **not** dropped here — they
    /// linger in the cull set but draw as nothing (skipped in
    /// [`Self::cull_bin_upload`]); the caller removes them via
    /// [`Self::remove_instance`] when convenient. A no-op if `chain_id` is
    /// out of range or already removed.
    pub fn remove_model(&mut self, chain_id: u32) {
        let Some(entries) = self.chains.get(chain_id as usize).cloned() else {
            return;
        };
        if entries.is_empty() {
            return; // already removed
        }
        self.last_cull = None; // PF.10 — tombstone changes visibility
        for &e in &entries {
            let e = e as usize;
            self.dead[e] = true;
            self.colors_alloc.free(e);
        }
        self.chains[chain_id as usize] = Vec::new(); // tombstone
    }

    /// Reclaim the holes left by [`Self::remove_model`]: rebuild the shared
    /// volume buffers from the live entries only, dropping every dead
    /// entry's data. Entry ids and `chain_id`s are preserved (dead entries
    /// keep a zero-length `meta` tombstone), so the caller's handles stay
    /// valid and no remap is needed.
    ///
    /// `registry` must be the resident one (entry ids 1:1, as for
    /// [`Self::add_model`] / [`Self::update_model`]). O(live volume) —
    /// call it when [`Self::dead_model_count`] is high, not every frame.
    pub fn compact(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        registry: &SpriteModelRegistry,
    ) {
        self.last_cull = None; // PF.10 — entry ids / chains renumbered
                               // occupancy + color_offsets: re-pack live entries tightly, rewrite
                               // each live entry's meta offset, zero the dead ones.
        self.compact_concat(device, registry, ConcatBuf::Occupancy);
        self.compact_concat(device, registry, ConcatBuf::ColorOffsets);
        // colors/dirs: the dead-aware repack already drops dead entries.
        self.repack_colors_dirs(device, registry);
        // model_meta: rewrite the (unchanged-length) table with the new
        // offsets. Buffer count didn't change, so no grow needed.
        queue.write_buffer(&self.model_meta, 0, bytemuck::cast_slice(&self.meta));
    }

    /// Rebuild one tightly-concatenated buffer from live entries only
    /// (used by [`Self::compact`]): assign each live entry a fresh tight
    /// offset, zero dead entries' offset, and recreate the buffer with
    /// slack.
    fn compact_concat(
        &mut self,
        device: &wgpu::Device,
        registry: &SpriteModelRegistry,
        which: ConcatBuf,
    ) {
        let mut all: Vec<u32> = Vec::new();
        for e in 0..self.meta.len() {
            if self.dead[e] {
                match which {
                    ConcatBuf::Occupancy => self.meta[e].occupancy_offset = 0,
                    ConcatBuf::ColorOffsets => self.meta[e].color_offsets_offset = 0,
                }
                continue;
            }
            let off = all.len() as u32;
            match which {
                ConcatBuf::Occupancy => self.meta[e].occupancy_offset = off,
                ConcatBuf::ColorOffsets => self.meta[e].color_offsets_offset = off,
            }
            all.extend_from_slice(concat_data(&registry.entries[e], which));
        }
        let used = all.len() as u32;
        let cap = grow_words(used);
        let (label, buf) = match which {
            ConcatBuf::Occupancy => ("roxlap-gpu sprite_reg.occupancy", &mut self.occupancy),
            ConcatBuf::ColorOffsets => (
                "roxlap-gpu sprite_reg.color_offsets",
                &mut self.color_offsets,
            ),
        };
        *buf = storage_dst_u32_cap(device, label, &all, cap);
        match which {
            ConcatBuf::Occupancy => {
                self.occ_used = used;
                self.occ_cap = cap;
            }
            ConcatBuf::ColorOffsets => {
                self.coloff_used = used;
                self.coloff_cap = cap;
            }
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

        // PF.10 — nothing changed since the last cull (same registry
        // state, same view, same screen): the four buffers already hold
        // exactly this frame's data — skip the whole cull/bin/upload.
        let key = CullKey::new(f, screen_w, screen_h, tile_size, lod_px);
        if let Some((k, res)) = self.last_cull {
            if k == key {
                return res;
            }
        }

        let nw = (1.0 + f.half_w * f.half_w).sqrt();
        let nh = (1.0 + f.half_h * f.half_h).sqrt();
        let cx = screen_w as f32 * 0.5;
        let cy = screen_h as f32 * 0.5;
        let px_per_world = cx / f.half_w; // isotropic: == cy/half_h
        let ts = tile_size as f32;
        let tx_max = tiles_x as i32 - 1;
        let ty_max = tiles_y as i32 - 1;

        // PF.10 — reused workspace (was 6+ fresh Vecs per frame).
        let scratch = &mut self.scratch;
        let visible = &mut scratch.visible;
        visible.clear();
        // Per-visible tile AABB (tx0, tx1, ty0, ty1) for the bin pass.
        let boxes = &mut scratch.boxes;
        boxes.clear();
        // Per-visible kv6colmul tables, flattened to two u32 per u64
        // entry (lanes 0|1, then 2|3), packed in visible order so the
        // shader indexes `colmul[inst_idx*512 + dir*2 + {0,1}]`. PF.10 —
        // built ONLY once a non-identity table exists (`any_colmul`);
        // until then the buffer holds a lazily-written identity fill and
        // the ~2 KiB-per-visible-instance rebuild + upload is skipped.
        let visible_colmul = &mut scratch.colmul;
        visible_colmul.clear();
        let counts = &mut scratch.counts;
        counts.clear();
        counts.resize(n_tiles, 0u32);
        let pack_colmul = self.any_colmul;

        for ci in &self.cull {
            // Skip instances of a removed model (tombstoned chain) — they
            // linger in `cull` until the caller drops them, but draw as
            // nothing.
            if self.chains[ci.chain_id as usize].is_empty() {
                continue;
            }
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
                // Mip-0 voxel screen size; a scaled instance's voxels
                // are `max_scale`× larger in world, so it holds the
                // fine mip proportionally longer (PS.1).
                let voxel_px = px_per_world * ci.max_scale / z;
                ((lod_px / voxel_px).log2().ceil().max(0.0) as usize).min(chain.len() - 1)
            } else {
                0
            };
            let mut g = ci.gpu;
            g.model_id = chain[level];
            visible.push(g);
            boxes.push([tx0, tx1, ty0, ty1]);
            if pack_colmul {
                for &w in ci.colmul.iter() {
                    visible_colmul.push((w & 0xffff_ffff) as u32);
                    visible_colmul.push((w >> 32) as u32);
                }
            }
            for ty in ty0..=ty1 {
                for tx in tx0..=tx1 {
                    counts[(ty * tiles_x as i32 + tx) as usize] += 1;
                }
            }
        }

        if visible.is_empty() {
            let res = (0, tiles_x, tiles_y);
            self.last_cull = Some((key, res));
            return res;
        }

        // Prefix-sum counts → per-tile offsets; build the flat grouped
        // index list.
        let tile_ranges = &mut scratch.tile_ranges;
        tile_ranges.clear();
        tile_ranges.resize(n_tiles * 2, 0u32);
        let mut running = 0u32;
        for t in 0..n_tiles {
            tile_ranges[2 * t] = running; // offset
            tile_ranges[2 * t + 1] = counts[t]; // count
            running += counts[t];
        }
        let total = running as usize;
        let tile_instances = &mut scratch.tile_instances;
        tile_instances.clear();
        tile_instances.resize(total.max(1), 0u32);
        let cursor = &mut scratch.cursor;
        cursor.clear();
        cursor.extend((0..n_tiles).map(|t| tile_ranges[2 * t]));
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
        queue.write_buffer(&self.instances, 0, bytemuck::cast_slice(visible));
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
        queue.write_buffer(&self.tile_ranges, 0, bytemuck::cast_slice(tile_ranges));
        queue.write_buffer(
            &self.tile_instances,
            0,
            bytemuck::cast_slice(tile_instances),
        );
        if pack_colmul {
            let need_colmul = visible_colmul.len() as u32;
            if need_colmul > self.colmul_cap {
                self.colmul_cap = need_colmul.next_power_of_two();
                self.colmul =
                    storage_dst_u32(device, "roxlap-gpu sprite_reg.colmul", self.colmul_cap);
                self.colmul_identity = false;
            }
            queue.write_buffer(&self.colmul, 0, bytemuck::cast_slice(visible_colmul));
        } else {
            // PF.10 — identity fast path: every table is identity, so the
            // buffer content is a constant repeating pattern. (Re)fill it
            // only on first use / growth; per-frame upload skipped.
            let need_colmul = visible.len() as u32 * 512;
            if need_colmul > self.colmul_cap {
                self.colmul_cap = need_colmul.next_power_of_two();
                self.colmul =
                    storage_dst_u32(device, "roxlap-gpu sprite_reg.colmul", self.colmul_cap);
                self.colmul_identity = false;
            }
            if !self.colmul_identity {
                let w = identity_colmul()[0];
                let (lo, hi) = ((w & 0xffff_ffff) as u32, (w >> 32) as u32);
                let fill: Vec<u32> = (0..self.colmul_cap)
                    .map(|i| if i & 1 == 0 { lo } else { hi })
                    .collect();
                queue.write_buffer(&self.colmul, 0, bytemuck::cast_slice(&fill));
                self.colmul_identity = true;
            }
        }

        let res = (visible.len() as u32, tiles_x, tiles_y);
        self.last_cull = Some((key, res));
        res
    }
}

/// GPU.12 incremental — per-entry placement of one model's `colors`
/// (and the parallel `dirs`) within the shared registry buffers: a
/// `[off, off+cap)` word window holding `len` live words. `cap >= len`
/// gives slack so a carve that *grows* the surface-voxel count can be
/// rewritten in place without relocating.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ColorSlot {
    off: u32,
    cap: u32,
    len: u32,
}

/// First-fit suballocator over the parallel `colors`/`dirs` buffers
/// (same offsets/ranks → one allocator drives both). Each registry
/// entry owns a [`ColorSlot`]; growth past a slot's `cap` relocates it
/// (freeing the old block) via the free list or a bump tail, and only
/// when the tail would exceed `cap_total` does the caller grow + repack
/// the whole buffer. Pure (no GPU) so it unit-tests on its own.
#[derive(Debug, Default)]
struct ColorsAllocator {
    /// Per-entry slot, indexed by entry id.
    slots: Vec<ColorSlot>,
    /// Freed `(off, cap)` blocks available for first-fit reuse.
    free: Vec<(u32, u32)>,
    /// Next bump-allocation position (words).
    tail: u32,
    /// Total buffer capacity in words.
    cap_total: u32,
}

/// Slack-padded capacity for a `len`-word array: +25% + 16 words, so a
/// few extra surface voxels from a carve fit without relocating.
fn slot_cap(len: u32) -> u32 {
    len + len / 4 + 16
}

/// Slack capacity (words) for a grown concatenated buffer: +50% + 256, so
/// a burst of `add_model` calls bump-appends rather than re-growing every
/// time. Matches [`ColorsAllocator`]'s `cap_total` headroom.
fn grow_words(used: u32) -> u32 {
    used + used / 2 + 256
}

/// Slack capacity (records) for a grown `model_meta` buffer: +50% + 8.
fn grow_records(count: u32) -> u32 {
    count + count / 2 + 8
}

impl ColorsAllocator {
    /// Lay every entry out contiguously (with per-slot slack) and add a
    /// global tail headroom so early growth bump-allocates rather than
    /// repacks.
    fn new(entry_lens: &[u32]) -> Self {
        let mut a = Self::default();
        a.repack(entry_lens);
        a
    }

    fn slot(&self, entry: usize) -> ColorSlot {
        self.slots[entry]
    }

    fn cap_total(&self) -> u32 {
        self.cap_total
    }

    /// Repack ALL entries compactly to fit `new_lens`, resetting the
    /// free list + tail and choosing a fresh `cap_total` with headroom.
    /// Used at initial build and on a buffer grow.
    fn repack(&mut self, new_lens: &[u32]) {
        self.free.clear();
        let mut off = 0u32;
        let mut slots = Vec::with_capacity(new_lens.len());
        for &len in new_lens {
            // A 0-length (dead / removed) entry takes no space — keeps a
            // tombstone slot so entry ids stay positional.
            let cap = if len == 0 { 0 } else { slot_cap(len) };
            slots.push(ColorSlot { off, cap, len });
            off += cap;
        }
        self.slots = slots;
        self.tail = off;
        // Global headroom: +50% + 256 words.
        self.cap_total = off + off / 2 + 256;
    }

    /// Place `new_len` words for `entry`. Returns `Some(off)` with the
    /// (possibly relocated) slot offset, or `None` if the buffer must
    /// grow + repack. On relocation the old block is pushed to the free
    /// list; an in-place fit returns the unchanged offset.
    fn place(&mut self, entry: usize, new_len: u32) -> Option<u32> {
        let cur = self.slots[entry];
        if new_len <= cur.cap {
            self.slots[entry] = ColorSlot {
                len: new_len,
                ..cur
            };
            return Some(cur.off);
        }
        let old = (cur.off, cur.cap);
        // First-fit a freed block big enough for the live data.
        if let Some(i) = self.free.iter().position(|&(_, c)| c >= new_len) {
            let (off, cap) = self.free.remove(i);
            self.free.push(old);
            self.slots[entry] = ColorSlot {
                off,
                cap,
                len: new_len,
            };
            return Some(off);
        }
        // Bump the tail if there's room.
        let want = slot_cap(new_len);
        if self.tail + want <= self.cap_total {
            let off = self.tail;
            self.tail += want;
            self.free.push(old);
            self.slots[entry] = ColorSlot {
                off,
                cap: want,
                len: new_len,
            };
            return Some(off);
        }
        None
    }

    /// Append a slot for a brand-new entry of `new_len` words (used by
    /// [`SpriteRegistryResident::add_model`]). Returns `Some(off)` placed
    /// via the free list or the bump tail, or `None` if the buffer must
    /// grow + repack — in which case **no** slot is pushed (the caller's
    /// repack rebuilds every slot from scratch).
    fn push(&mut self, new_len: u32) -> Option<u32> {
        if let Some(i) = self.free.iter().position(|&(_, c)| c >= new_len) {
            let (off, cap) = self.free.remove(i);
            self.slots.push(ColorSlot {
                off,
                cap,
                len: new_len,
            });
            return Some(off);
        }
        let want = slot_cap(new_len);
        if self.tail + want <= self.cap_total {
            let off = self.tail;
            self.tail += want;
            self.slots.push(ColorSlot {
                off,
                cap: want,
                len: new_len,
            });
            return Some(off);
        }
        None
    }

    /// Free `entry`'s slot back to the pool ([`SpriteRegistryResident::
    /// remove_model`]). Its `(off, cap)` block joins the free list for
    /// first-fit reuse by a later [`Self::push`]; the slot is zeroed so a
    /// repack treats it as a 0-length tombstone.
    fn free(&mut self, entry: usize) {
        let s = self.slots[entry];
        if s.cap > 0 {
            self.free.push((s.off, s.cap));
        }
        self.slots[entry] = ColorSlot {
            off: 0,
            cap: 0,
            len: 0,
        };
    }
}

/// Create a STORAGE buffer of u32s; pads empty input (wgpu rejects
/// zero-sized storage bindings).
#[allow(dead_code)]
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
        // COPY_SRC so test/debug harnesses can read the contents back
        // (PF.10's cull gate does); free at runtime.
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

/// Create a `STORAGE | COPY_DST` `u32` buffer of `cap` words (≥ data
/// length, ≥ 1), initialised with `data` at offset 0 and the tail left
/// zeroed. Unlike [`storage_u32`] (STORAGE-only, exact-size) this both
/// reserves spare capacity and is `COPY_DST`, so the incremental
/// [`SpriteRegistryResident::update_model`] can `write_buffer` a growing
/// `colors`/`dirs` array in place. Filled via `mapped_at_creation` so no
/// queue is needed at upload time.
fn storage_dst_u32_cap(device: &wgpu::Device, label: &str, data: &[u32], cap: u32) -> wgpu::Buffer {
    let cap = cap.max(data.len() as u32).max(1);
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: u64::from(cap) * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: true,
    });
    if !data.is_empty() {
        buf.slice(..(data.len() as u64 * 4))
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(data));
    }
    buf.unmap();
    buf
}

/// Create a `STORAGE | COPY_DST` buffer of Pod records, exact-size
/// (≥ 1, zero-padded), so individual records can be rewritten in place
/// by [`SpriteRegistryResident::update_model`] on a relocation. The
/// record *count* never changes on an incremental edit (no model is
/// added/removed), so no slack is needed here.
fn storage_dst_pod<T: Pod + Zeroable>(
    device: &wgpu::Device,
    label: &str,
    data: &[T],
) -> wgpu::Buffer {
    let one = [T::zeroed()];
    let src: &[T] = if data.is_empty() { &one } else { data };
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: std::mem::size_of_val(src) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: true,
    });
    buf.slice(..)
        .get_mapped_range_mut()
        .copy_from_slice(bytemuck::cast_slice(src));
    buf.unmap();
    buf
}

/// Create a `STORAGE | COPY_DST` Pod buffer holding `cap` records
/// (≥ `data.len()`, ≥ 1), initialised with `data` at record 0 and the
/// tail zeroed. The slack lets [`SpriteRegistryResident::add_model`] grow
/// the `model_meta` table without re-growing on every add.
fn storage_dst_pod_cap<T: Pod + Zeroable>(
    device: &wgpu::Device,
    label: &str,
    data: &[T],
    cap: u32,
) -> wgpu::Buffer {
    let rec = std::mem::size_of::<T>() as u64;
    let cap = u64::from(cap.max(data.len() as u32).max(1));
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: cap * rec,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: true,
    });
    if !data.is_empty() {
        buf.slice(..(data.len() as u64 * rec))
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(data));
    }
    buf.unmap();
    buf
}

/// Create a STORAGE buffer of Pod records; pads empty input with one
/// zeroed `T`.
#[allow(dead_code)]
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
    fn remove_frees_chain_data_keeps_ids_stable() {
        let mut reg = SpriteModelRegistry::new();
        let a = reg.add_lod(build_sprite_model(&kv6_unsorted()), 4);
        let b = reg.add_lod(build_sprite_model(&kv6_unsorted()), 4);
        let len_before = reg.len();
        assert!(reg.is_live(a) && reg.is_live(b));

        reg.remove(a);
        // Chain `a` is tombstoned (its entries are freed to empty models;
        // they're unreachable via `model()` now — that's the tombstone).
        assert!(!reg.is_live(a));
        // `b` is untouched and still live; `len()` (next id) is unchanged.
        assert!(reg.is_live(b));
        assert_eq!(&reg.model(b).colors, &[0xBB, 0xAA, 0xCC]);
        assert_eq!(reg.len(), len_before);

        // A later add mints a fresh id past the tombstone (no slot reuse).
        let c = reg.add_lod(build_sprite_model(&kv6_unsorted()), 4);
        assert_eq!(c, len_before as u32);
        assert!(reg.is_live(c));
        // `b`'s id stayed valid across the remove + add round-trip.
        assert_eq!(&reg.model(b).colors, &[0xBB, 0xAA, 0xCC]);
    }

    #[test]
    fn model_checked_guards_out_of_range_and_tombstoned() {
        // The guard `set_instance_model` relies on: `model()` would
        // index-panic on these, `model_checked` returns `None`.
        let mut reg = SpriteModelRegistry::new();
        let a = reg.add_lod(build_sprite_model(&kv6_unsorted()), 4);
        assert!(reg.model_checked(a).is_some());
        assert!(reg.model_checked(9999).is_none(), "out of range → None");
        reg.remove(a);
        assert!(reg.model_checked(a).is_none(), "tombstoned chain → None");
    }

    #[test]
    fn remove_is_idempotent_and_bounds_safe() {
        let mut reg = SpriteModelRegistry::new();
        let a = reg.add(build_sprite_model(&kv6_unsorted()));
        reg.remove(a);
        reg.remove(a); // already removed → no-op, no panic
        reg.remove(999); // out of range → no-op
        assert!(!reg.is_live(a));
        assert!(!reg.is_live(999));
    }

    #[test]
    fn registry_gpu_structs_have_expected_sizes() {
        assert_eq!(std::mem::size_of::<SpriteModelMeta>(), 48);
        // TV — grew 64 → 80 with the per-instance material id + alpha_mul
        // (+ 8 bytes pad to keep the 16-byte std430 stride).
        assert_eq!(std::mem::size_of::<SpriteInstanceGpu>(), 80);
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

    // ---- GPU.12 incremental: colors/dirs suballocator -----------------

    /// Every slot fits its data, has slack, doesn't overlap the next, and
    /// the buffer reserves tail headroom past the last slot.
    fn alloc_invariants(a: &ColorsAllocator, lens: &[u32]) {
        let mut prev_end = 0u32;
        for (e, &len) in lens.iter().enumerate() {
            let s = a.slot(e);
            assert_eq!(s.len, len, "slot {e} len");
            assert!(s.cap >= s.len, "slot {e} cap >= len");
            // In a freshly repacked layout slots are in entry order.
            assert!(s.off >= prev_end, "slot {e} overlaps previous");
            assert!(s.off + s.cap <= a.cap_total(), "slot {e} past cap_total");
            prev_end = s.off + s.cap;
        }
        assert!(a.cap_total() >= prev_end, "tail headroom");
    }

    #[test]
    fn allocator_new_lays_out_with_slack_and_headroom() {
        let lens = [10u32, 0, 64, 7];
        let a = ColorsAllocator::new(&lens);
        alloc_invariants(&a, &lens);
        // Slack: a 64-word slot has cap > 64 so a small carve-grow fits.
        assert!(a.slot(2).cap > 64);
        // Headroom past the bump tail for early growth.
        assert!(a.cap_total() > a.slot(3).off + a.slot(3).cap);
    }

    #[test]
    fn allocator_place_in_place_when_within_cap() {
        let mut a = ColorsAllocator::new(&[10, 20]);
        let off0 = a.slot(0).off;
        let cap0 = a.slot(0).cap;
        // Shrink: still the same slot.
        assert_eq!(a.place(0, 5), Some(off0));
        assert_eq!(a.slot(0).len, 5);
        assert_eq!(a.slot(0).cap, cap0);
        // Grow within slack: same offset, no relocation.
        assert_eq!(a.place(0, cap0), Some(off0));
        assert_eq!(a.slot(0).off, off0);
        assert!(a.free.is_empty(), "no relocation should free anything");
    }

    #[test]
    fn allocator_place_relocates_to_tail_and_frees_old() {
        let mut a = ColorsAllocator::new(&[10, 20]);
        let old0 = (a.slot(0).off, a.slot(0).cap);
        let tail_before = a.tail;
        // Overgrow entry 0 past its cap → relocate to the bump tail.
        let new_len = a.slot(0).cap + 5;
        let off = a.place(0, new_len).expect("fits in headroom");
        assert_eq!(off, tail_before, "relocated to old tail");
        assert_eq!(a.slot(0).off, off);
        assert_eq!(a.slot(0).len, new_len);
        assert!(a.free.contains(&old0), "old slot freed");
    }

    #[test]
    fn allocator_reuses_freed_block_first_fit() {
        // Entry 0 has a large slot; entry 1 a tiny one, so growing 1 must
        // relocate (it can't fit in place) and lands in 0's freed block.
        let mut a = ColorsAllocator::new(&[10, 2]);
        let old0 = (a.slot(0).off, a.slot(0).cap);
        // Relocate entry 0 to the tail, freeing its original block.
        let _ = a.place(0, a.slot(0).cap + 5).unwrap();
        assert!(a.free.contains(&old0));
        // Grow entry 1 past its (tiny) cap but ≤ the freed block's cap →
        // first-fit reuses that block rather than bumping the tail.
        let new1 = a.slot(1).cap + 1;
        assert!(new1 <= old0.1, "freed block big enough");
        let off = a.place(1, new1).expect("reuses freed block");
        assert_eq!(off, old0.0, "first-fit reused the freed slot offset");
        assert!(!a.free.contains(&old0), "freed block consumed");
    }

    #[test]
    fn allocator_signals_grow_then_repack_restores() {
        let mut a = ColorsAllocator::new(&[8, 8]);
        // Force overflow: ask for far more than cap_total.
        let huge = a.cap_total() + 100;
        assert_eq!(a.place(0, huge), None, "overflow must signal grow");
        // Repack with the new lengths compacts + grows the buffer.
        a.repack(&[huge, 8]);
        alloc_invariants(&a, &[huge, 8]);
        assert!(a.cap_total() > huge);
        // After repack the entry now fits in place.
        assert_eq!(a.place(0, huge), Some(a.slot(0).off));
    }

    /// Drive the allocator like a real carve loop (mirroring
    /// `update_model`): one model's colour count drifts up and down
    /// across many edits while two neighbours stay put. Growth is
    /// absorbed in place / via the free list / by the bump tail, and on
    /// the rare overflow we repack (as `update_model` does). After every
    /// edit the live `[off, off+len)` windows must stay disjoint.
    #[test]
    fn allocator_carve_loop_keeps_live_windows_disjoint() {
        let mut a = ColorsAllocator::new(&[40, 12, 40]);
        let mut lens = [40u32, 12, 40];
        // A deterministic up/down walk of entry 1's length, incl. a jump
        // that forces at least one grow+repack.
        let walk = [13u32, 30, 60, 18, 9, 80, 80, 25, 200, 7];
        let mut grew = false;
        for &len in &walk {
            lens[1] = len;
            // Entry 1 re-placed; on overflow, repack the whole set.
            if a.place(1, len).is_none() {
                grew = true;
                a.repack(&lens);
            } else {
                // Neighbours fit in place every time.
                assert_eq!(a.place(0, 40), Some(a.slot(0).off));
                assert_eq!(a.place(2, 40), Some(a.slot(2).off));
            }
            assert_eq!(a.slot(1).len, len);

            // No two entries' live windows overlap.
            let mut wins: Vec<(u32, u32)> =
                (0..3).map(|e| (a.slot(e).off, a.slot(e).len)).collect();
            wins.sort_by_key(|w| w.0);
            for pair in wins.windows(2) {
                let (o0, l0) = pair[0];
                let (o1, _) = pair[1];
                assert!(o0 + l0 <= o1, "live windows overlap: {pair:?}");
            }
        }
        assert!(grew, "the 200-word jump should have forced a repack");
    }

    // --- incremental instance path (device-backed; skips w/o adapter) ---

    fn headless() -> Option<crate::HeadlessGpu> {
        match crate::HeadlessGpu::new_blocking(crate::GpuRendererSettings::default()) {
            Ok(h) => Some(h),
            Err(e) => {
                eprintln!("[skip] no GPU adapter reachable: {e}");
                None
            }
        }
    }

    fn one_model_registry() -> (SpriteModelRegistry, u32) {
        let mut reg = SpriteModelRegistry::new();
        let id = reg.add(build_sprite_model(&kv6_unsorted()));
        (reg, id)
    }

    fn inst(model_id: u32, pos: [f32; 3]) -> SpriteInstance {
        use roxlap_formats::sprite::Sprite;
        SpriteInstance::new(
            model_id,
            SpriteInstanceTransform::from_sprite(&Sprite::axis_aligned(kv6_unsorted(), pos)),
        )
    }

    /// PS.1 — a scaled basis grows the cull sphere with the pose: the
    /// transform keeps the longest basis column, and `make_cull` seeds
    /// `radius = model.bound_radius() × max_scale`, so scaled-up
    /// instances (particles) no longer under-cull at screen edges.
    #[test]
    fn scaled_basis_scales_cull_radius() {
        use roxlap_formats::sprite::Sprite;

        let mut reg = SpriteModelRegistry::new();
        let chain = reg.add(build_sprite_model(&kv6_unsorted()));
        let model_r = reg.model(chain).bound_radius();

        let scaled = |k: f32, pos: [f32; 3]| {
            let mut s = Sprite::axis_aligned(kv6_unsorted(), pos);
            for a in 0..3 {
                s.s[a] *= k;
                s.h[a] *= k;
                s.f[a] *= k;
            }
            SpriteInstanceTransform::from_sprite(&s)
        };

        // Unit basis: max_scale 1, radius = the model's (float-exact).
        let unit = inst(chain, [0.0; 3]);
        assert_eq!(unit.transform.max_scale, 1.0);
        assert_eq!(make_cull(&reg, &unit).radius, model_r);

        // 2× uniform scale doubles both.
        let xf2 = scaled(2.0, [0.0; 3]);
        assert!((xf2.max_scale - 2.0).abs() < 1e-6);
        let big = SpriteInstance::new(chain, xf2);
        assert!((make_cull(&reg, &big).radius - 2.0 * model_r).abs() < 1e-4);

        // Anisotropic scale takes the longest column.
        let mut s = Sprite::axis_aligned(kv6_unsorted(), [0.0; 3]);
        for a in 0..3 {
            s.h[a] *= 3.0;
            s.f[a] *= 0.5;
        }
        let xf = SpriteInstanceTransform::from_sprite(&s);
        assert!((xf.max_scale - 3.0).abs() < 1e-6);
    }

    #[test]
    fn append_grows_count_and_capacity_pow2() {
        let Some(h) = headless() else { return };
        let (reg, m) = one_model_registry();
        let mut res = SpriteRegistryResident::upload(&h.device, &reg, &[inst(m, [0.0; 3])]);
        assert_eq!(res.instance_count(), 1);
        assert_eq!(res.instance_capacity, 1);

        // Append 4 → count 5, capacity grows to next_pow2(5) = 8.
        let more: Vec<_> = (1..=4).map(|i| inst(m, [i as f32, 0.0, 0.0])).collect();
        let base = res.append_instances(&h.device, &reg, &more);
        assert_eq!(base, 1, "first appended index follows the seed instance");
        assert_eq!(res.instance_count(), 5);
        assert_eq!(res.instance_capacity, 8, "power-of-two growth");

        // A second append that still fits keeps the same capacity (no realloc).
        let base2 = res.append_instances(&h.device, &reg, &[inst(m, [9.0, 0.0, 0.0])]);
        assert_eq!(base2, 5);
        assert_eq!(res.instance_count(), 6);
        assert_eq!(res.instance_capacity, 8, "fits existing capacity, no grow");
    }

    #[test]
    fn append_empty_is_noop() {
        let Some(h) = headless() else { return };
        let (reg, m) = one_model_registry();
        let mut res = SpriteRegistryResident::upload(&h.device, &reg, &[inst(m, [0.0; 3])]);
        let base = res.append_instances(&h.device, &reg, &[]);
        assert_eq!(base, 1);
        assert_eq!(res.instance_count(), 1);
        assert_eq!(res.instance_capacity, 1);
    }

    /// Read `words` u32s back from a GPU buffer (needs COPY_SRC).
    fn read_u32(h: &crate::HeadlessGpu, buf: &wgpu::Buffer, words: u64) -> Vec<u32> {
        let bytes = words * 4;
        let staging = h.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = h
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        enc.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
        h.queue.submit(std::iter::once(enc.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        h.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let out = bytemuck::cast_slice::<u8, u32>(&data).to_vec();
        drop(data);
        staging.unmap();
        out
    }

    /// A second distinct model so add_model has real new geometry to lay
    /// down (different dims + colours from `kv6_unsorted`).
    fn kv6_other() -> Kv6 {
        let mk = |z, col| Voxel {
            col,
            z,
            vis: 0,
            dir: 0,
        };
        Kv6 {
            xsiz: 1,
            ysiz: 1,
            zsiz: 4,
            xpiv: 0.0,
            ypiv: 0.0,
            zpiv: 0.0,
            voxels: vec![mk(0, 0x11), mk(2, 0x22)],
            xlen: vec![2],
            ylen: vec![vec![2]],
            palette: None,
        }
    }

    /// add_model lays the new model's volume on the GPU at the offsets its
    /// meta record claims — verified by reading the shared buffers back
    /// and matching each entry against its source SpriteModel.
    #[test]
    fn add_model_uploads_new_volume_incrementally() {
        let Some(h) = headless() else { return };

        // Residency starts with model A only.
        let mut reg = SpriteModelRegistry::new();
        let a = reg.add(build_sprite_model(&kv6_unsorted()));
        let mut res = SpriteRegistryResident::upload(&h.device, &reg, &[inst(a, [0.0; 3])]);
        assert_eq!(res.chains.len(), 1);
        let entries_before = res.meta.len();

        // Append model B (single-level) to the registry, then sync it.
        let b = reg.add(build_sprite_model(&kv6_other()));
        res.add_model(&h.device, &h.queue, &reg, b);
        assert_eq!(res.chains.len(), 2);
        assert_eq!(res.meta.len(), entries_before + 1, "one new entry");

        // Read the shared buffers back and check EVERY entry's data sits
        // where its meta record points — both the pre-existing A and the
        // newly streamed B.
        let occ = read_u32(&h, &res.occupancy, u64::from(res.occ_cap));
        let coloff = read_u32(&h, &res.color_offsets, u64::from(res.coloff_cap));
        let cols = read_u32(&h, &res.colors, u64::from(res.colors_alloc.cap_total()));
        for (e, m) in reg.entries.iter().enumerate() {
            let meta = res.meta[e];
            let oo = meta.occupancy_offset as usize;
            assert_eq!(
                &occ[oo..oo + m.occupancy.len()],
                &m.occupancy[..],
                "occ entry {e}"
            );
            let co = meta.color_offsets_offset as usize;
            assert_eq!(
                &coloff[co..co + m.color_offsets.len()],
                &m.color_offsets[..],
                "color_offsets entry {e}"
            );
            let cc = meta.colors_offset as usize;
            assert_eq!(
                &cols[cc..cc + m.colors.len()],
                &m.colors[..],
                "colors entry {e}"
            );
        }

        // And an instance of the freshly-added model can now be appended.
        let base = res.append_instances(&h.device, &reg, &[inst(b, [5.0, 0.0, 0.0])]);
        assert_eq!(base, 1);
        assert_eq!(res.instance_count(), 2);
    }

    /// Adding many small models forces the volume buffers to grow + rebuild
    /// at least once; every entry must still read back correctly across the
    /// grow boundary.
    #[test]
    fn add_model_survives_buffer_growth() {
        let Some(h) = headless() else { return };
        let mut reg = SpriteModelRegistry::new();
        let a = reg.add(build_sprite_model(&kv6_unsorted()));
        let mut res = SpriteRegistryResident::upload(&h.device, &reg, &[inst(a, [0.0; 3])]);
        let occ_cap0 = res.occ_cap;

        // 40 adds — occupancy starts exact-sized (cap == used), so the very
        // first add overflows and grows; later ones ride the slack.
        for _ in 0..40 {
            let id = reg.add(build_sprite_model(&kv6_other()));
            res.add_model(&h.device, &h.queue, &reg, id);
        }
        assert_eq!(res.chains.len(), 41);
        assert!(res.occ_cap > occ_cap0, "occupancy buffer grew");

        let occ = read_u32(&h, &res.occupancy, u64::from(res.occ_cap));
        let cols = read_u32(&h, &res.colors, u64::from(res.colors_alloc.cap_total()));
        for (e, m) in reg.entries.iter().enumerate() {
            let meta = res.meta[e];
            let oo = meta.occupancy_offset as usize;
            assert_eq!(
                &occ[oo..oo + m.occupancy.len()],
                &m.occupancy[..],
                "occ entry {e}"
            );
            let cc = meta.colors_offset as usize;
            assert_eq!(
                &cols[cc..cc + m.colors.len()],
                &m.colors[..],
                "colors entry {e}"
            );
        }
    }

    /// VCL.2 — a decoded voxel clip's frames register as a flipbook of LOD
    /// chains, and `set_instance_model` flips which frame an instance
    /// draws. The cull state it updates is exactly what
    /// `cull_bin_upload` packs into the GPU instance buffer each frame, so
    /// TV.3 (clip wiring): `sprite_model_from_clip_frame_with_materials`
    /// classifies a clip frame's voxels into a per-voxel `materials` array
    /// (parallel to `colors`) by colour; an empty map leaves it empty (the
    /// all-opaque clip), identical to `sprite_model_from_clip_frame`.
    #[test]
    fn clip_frame_with_materials_classifies_by_color() {
        use roxlap_formats::voxel_clip::{LoopMode, VoxelClip, VoxelFrame};

        let dims = [1u32, 1, 4];
        let owpc = dims[2].div_ceil(32).max(1) as usize; // 1
        let glass = 0x80AA_BBCC;
        let stone = 0x8011_2233;
        let frame = VoxelFrame {
            occupancy: {
                let mut occ = vec![0u32; owpc];
                occ[0] |= (1 << 0) | (1 << 1);
                occ
            },
            colors: vec![stone, glass], // ascending z: z=0 stone, z=1 glass
            color_offsets: vec![0, 2],
        };
        let clip = VoxelClip::from_frames(
            dims,
            [0.5, 0.5, 2.0],
            1.0,
            LoopMode::Loop,
            &[frame],
            &[],
            33,
            0,
        );
        let decoded = clip.decode().expect("decode");

        // Map only the glass colour → material 2; stone stays opaque (0).
        let m = sprite_model_from_clip_frame_with_materials(&decoded, 0, &[(Rgb(0x00AA_BBCC), 2)]);
        assert_eq!(
            m.materials.len(),
            m.colors.len(),
            "materials parallel to colors"
        );
        // `colors` is in popcount-rank (ascending z) order: stone then glass.
        assert_eq!(
            m.materials,
            vec![0u8, 2u8],
            "stone opaque, glass material 2"
        );

        // Empty map ⇒ no per-voxel materials, identical to the plain builder.
        let plain = sprite_model_from_clip_frame(&decoded, 0);
        let plain_mat = sprite_model_from_clip_frame_with_materials(&decoded, 0, &[]);
        assert!(plain.materials.is_empty());
        assert!(plain_mat.materials.is_empty());
        assert_eq!(plain.colors, plain_mat.colors);
    }

    /// TV.3 (streaming-clip refresh path): `build_sprite_model_with_materials`
    /// — the builder behind `GpuBackend::update_sprite_model_with_materials`,
    /// which a streaming clip re-runs each frame — classifies a kv6's voxels
    /// into a per-voxel `materials` array (popcount-rank order) by colour.
    #[test]
    fn build_with_materials_classifies_by_color() {
        let glass = 0x80AA_BBCC;
        let stone = 0x8011_2233;
        // One column (x=0,y=0), two voxels: z=0 stone, z=1 glass.
        let kv6 = kv6_from(1, 1, 4, &[(0, 0, 0, stone), (0, 0, 1, glass)]);

        let m = build_sprite_model_with_materials(&kv6, &[(Rgb(0x00AA_BBCC), 2)]);
        assert_eq!(
            m.materials.len(),
            m.colors.len(),
            "materials parallel to colors"
        );
        assert_eq!(
            m.materials,
            vec![0u8, 2u8],
            "stone opaque, glass material 2"
        );

        // Empty map ⇒ no per-voxel materials, identical to `build_sprite_model`.
        let plain = build_sprite_model(&kv6);
        let plain_mat = build_sprite_model_with_materials(&kv6, &[]);
        assert!(plain.materials.is_empty());
        assert!(plain_mat.materials.is_empty());
        assert_eq!(plain.colors, plain_mat.colors);
    }

    /// flipping `chain_id` redirects the rendered instance to the new
    /// frame's resident volume.
    #[test]
    fn voxel_clip_flipbook_set_instance_model() {
        use roxlap_formats::voxel_clip::{LoopMode, VoxelClip, VoxelFrame};
        let Some(h) = headless() else { return };

        // Two distinct frames of a 1×1×4 clip: frame 0 has a voxel at z=0;
        // frame 1 adds z=1 — different occupancy + a longer colour run.
        let dims = [1u32, 1, 4];
        let owpc = dims[2].div_ceil(32).max(1) as usize; // 1
        let mk_frame = |zs: &[u32], cols: &[u32]| -> VoxelFrame {
            let mut occ = vec![0u32; owpc];
            for &z in zs {
                occ[(z >> 5) as usize] |= 1u32 << (z & 31);
            }
            VoxelFrame {
                occupancy: occ,
                colors: cols.to_vec(),
                color_offsets: vec![0, cols.len() as u32],
            }
        };
        let f0 = mk_frame(&[0], &[0x8011_2233]);
        let f1 = mk_frame(&[0, 1], &[0x8011_2233, 0x80AA_BBCC]);
        let clip = VoxelClip::from_frames(
            dims,
            [0.5, 0.5, 2.0],
            1.0,
            LoopMode::Loop,
            &[f0, f1],
            &[],
            33,
            0,
        );
        let decoded = clip.decode().expect("decode");

        // Each frame → a single-level chain; both volumes resident + distinct.
        let mut reg = SpriteModelRegistry::new();
        let c0 = reg.add(sprite_model_from_clip_frame(&decoded, 0));
        let c1 = reg.add(sprite_model_from_clip_frame(&decoded, 1));
        assert_eq!(reg.model(c0).colors.len(), 1);
        assert_eq!(reg.model(c1).colors.len(), 2);

        // One instance, in front of the test frustum, drawing frame 0.
        let mut res = SpriteRegistryResident::upload(&h.device, &reg, &[inst(c0, [0.0, 0.0, 5.0])]);
        assert_eq!(res.cull[0].chain_id, c0);

        // Flip to frame 1: the cull now draws chain c1 (radius reseeded).
        res.set_instance_model(&reg, 0, c1);
        assert_eq!(res.cull[0].chain_id, c1);
        assert_eq!(res.cull[0].radius, reg.model(c1).bound_radius());

        // The next cull packs the new chain into the GPU instance buffer
        // (visible, no panic).
        let f = test_frustum();
        let (visible, _, _) = res.cull_bin_upload(&h.device, &h.queue, &f, 64, 64, 16, 1.0);
        assert_eq!(visible, 1);

        // …and back to frame 0.
        res.set_instance_model(&reg, 0, c0);
        assert_eq!(res.cull[0].chain_id, c0);

        // Out-of-range index is a safe no-op.
        res.set_instance_model(&reg, 99, c1);
        assert_eq!(res.cull[0].chain_id, c0);
    }

    fn test_frustum() -> ViewFrustum {
        ViewFrustum {
            pos: [0.0, 0.0, 0.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 1.0, 0.0],
            forward: [0.0, 0.0, 1.0],
            half_w: 1.0,
            half_h: 1.0,
            far: 10_000.0,
        }
    }

    #[test]
    fn remove_model_tombstones_frees_and_reuses() {
        let Some(h) = headless() else { return };
        // Residency with models A and B, one instance each.
        let mut reg = SpriteModelRegistry::new();
        let a = reg.add(build_sprite_model(&kv6_unsorted()));
        let b = reg.add(build_sprite_model(&kv6_other()));
        let mut res = SpriteRegistryResident::upload(
            &h.device,
            &reg,
            &[inst(a, [0.0; 3]), inst(b, [1.0, 0.0, 0.0])],
        );
        assert_eq!(res.live_model_count(), 2);
        assert_eq!(res.dead_model_count(), 0);

        // Remove B → tombstoned, its colours freed into the pool.
        res.remove_model(b);
        assert_eq!(res.live_model_count(), 1);
        assert_eq!(res.dead_model_count(), 1);
        assert_eq!(res.dead.iter().filter(|&&d| d).count(), 1, "one entry dead");
        assert!(!res.colors_alloc.free.is_empty(), "B's colour slot freed");

        // Adding C reuses the freed slot (free-list first-fit).
        let c = reg.add(build_sprite_model(&kv6_other()));
        res.add_model(&h.device, &h.queue, &reg, c);
        assert_eq!(res.live_model_count(), 2);

        // A and C read back correctly; B is dead (skipped).
        let cols = read_u32(&h, &res.colors, u64::from(res.colors_alloc.cap_total()));
        for e in [a as usize, c as usize] {
            let m = &reg.entries[e];
            let cc = res.meta[e].colors_offset as usize;
            assert_eq!(
                &cols[cc..cc + m.colors.len()],
                &m.colors[..],
                "colors entry {e}"
            );
        }

        // The lingering instance of removed B is skipped without panic.
        let f = test_frustum();
        let _ = res.cull_bin_upload(&h.device, &h.queue, &f, 64, 64, 16, 1.0);
    }

    #[test]
    fn compact_reclaims_holes_keeps_ids_stable() {
        let Some(h) = headless() else { return };
        let mut reg = SpriteModelRegistry::new();
        let a = reg.add(build_sprite_model(&kv6_unsorted()));
        let b = reg.add(build_sprite_model(&kv6_other()));
        let c = reg.add(build_sprite_model(&kv6_other()));
        let mut res = SpriteRegistryResident::upload(
            &h.device,
            &reg,
            &[inst(a, [0.0; 3]), inst(b, [1.0; 3]), inst(c, [2.0; 3])],
        );
        let occ_used_full = res.occ_used;

        // Remove the middle model, then compact.
        res.remove_model(b);
        res.compact(&h.device, &h.queue, &reg);

        // Holes reclaimed: occupancy now only covers A + C.
        let live_occ: u32 = [a, c]
            .iter()
            .map(|&e| reg.entries[e as usize].occupancy.len() as u32)
            .sum();
        assert_eq!(res.occ_used, live_occ);
        assert!(res.occ_used < occ_used_full, "compaction shrank occupancy");
        // Dead entry keeps a zeroed tombstone; ids unchanged.
        assert_eq!(res.meta[b as usize].occupancy_offset, 0);
        assert_eq!(res.live_model_count(), 2);
        assert_eq!(res.dead_model_count(), 1);

        // Live entries read back correctly at their new offsets.
        let occ = read_u32(&h, &res.occupancy, u64::from(res.occ_cap));
        let cols = read_u32(&h, &res.colors, u64::from(res.colors_alloc.cap_total()));
        for &e in &[a as usize, c as usize] {
            let m = &reg.entries[e];
            let oo = res.meta[e].occupancy_offset as usize;
            assert_eq!(
                &occ[oo..oo + m.occupancy.len()],
                &m.occupancy[..],
                "occ {e}"
            );
            let cc = res.meta[e].colors_offset as usize;
            assert_eq!(&cols[cc..cc + m.colors.len()], &m.colors[..], "cols {e}");
        }

        // Chain ids still valid: C's chain still resolves; B's is empty.
        assert!(!res.chains[c as usize].is_empty());
        assert!(res.chains[b as usize].is_empty());
    }

    #[test]
    fn remove_swap_semantics_and_capacity_retained() {
        let Some(h) = headless() else { return };
        let (reg, m) = one_model_registry();
        let seed: Vec<_> = (0..4).map(|i| inst(m, [i as f32, 0.0, 0.0])).collect();
        let mut res = SpriteRegistryResident::upload(&h.device, &reg, &seed);
        assert_eq!(res.instance_count(), 4);
        let cap = res.instance_capacity;

        // Remove a middle element → the previous last (idx 3) moved into it.
        assert_eq!(res.remove_instance(1), Some(3));
        assert_eq!(res.instance_count(), 3);

        // Remove the current last (idx 2) → nothing moved.
        assert_eq!(res.remove_instance(2), None);
        assert_eq!(res.instance_count(), 2);

        // Out of range → None.
        assert_eq!(res.remove_instance(99), None);
        assert_eq!(res.instance_count(), 2);

        // Capacity is retained for reuse (no shrink).
        assert_eq!(res.instance_capacity, cap);
    }
}
