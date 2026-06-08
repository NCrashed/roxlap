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
    clippy::many_single_char_names
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

    let mut voxel_iter = kv6.voxels.iter();
    for x in 0..mx {
        for y in 0..my {
            let col = (x + y * mx) as usize;
            color_offsets[col] = colors.len() as u32;
            let count = kv6.ylen[x as usize][y as usize];
            // Collect the column's voxels and sort ascending z so the
            // occupancy popcount rank matches the colour order.
            let mut column: Vec<(u16, u32)> = (0..count)
                .map(|_| {
                    let v = voxel_iter.next().expect("KV6 ylen / voxels.len mismatch");
                    (v.z, v.col)
                })
                .collect();
            column.sort_by_key(|(z, _)| *z);
            for (z, col_rgba) in column {
                let z = u32::from(z);
                let base = col * occ_words_per_col as usize + (z >> 5) as usize;
                occupancy[base] |= 1u32 << (z & 31);
                colors.push(col_rgba);
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

/// A registry of sprite models. Instances reference a model by index
/// (`model_id`); identical KV6s are added once and shared by many
/// instances. **Copy-on-modify**: [`Self::fork`] deep-copies a model
/// so edits to the fork leave the parent (and its instances) intact.
#[derive(Debug, Clone, Default)]
pub struct SpriteModelRegistry {
    models: Vec<SpriteModel>,
}

impl SpriteModelRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self { models: Vec::new() }
    }

    /// Register a model and return its `model_id`.
    pub fn add(&mut self, model: SpriteModel) -> u32 {
        let id = self.models.len() as u32;
        self.models.push(model);
        id
    }

    /// Copy-on-modify: deep-copy model `parent` into a new entry and
    /// return its `model_id`. The fork owns independent voxel data, so
    /// mutating it (e.g. via [`Self::model_mut`]) does not affect the
    /// parent or any instance still pointing at it.
    ///
    /// # Panics
    /// If `parent` is not a registered model id.
    pub fn fork(&mut self, parent: u32) -> u32 {
        let copy = self.models[parent as usize].clone();
        self.add(copy)
    }

    #[must_use]
    pub fn model(&self, id: u32) -> &SpriteModel {
        &self.models[id as usize]
    }

    /// Mutable access for editing a (typically forked) model.
    pub fn model_mut(&mut self, id: u32) -> &mut SpriteModel {
        &mut self.models[id as usize]
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.models.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }
}

impl SpriteModel {
    /// Recolour every voxel via `f(old_rgba) -> new_rgba`. Structure
    /// (occupancy / offsets) is untouched, so this is a cheap in-place
    /// edit — handy on a [`SpriteModelRegistry::fork`] to make a tinted
    /// variant. Structural edits (add/remove voxels) come in GPU.10.5.
    pub fn recolor(&mut self, f: impl Fn(u32) -> u32) {
        for c in &mut self.colors {
            *c = f(*c);
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
    gpu: SpriteInstanceGpu,
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
    _pad1: f32,
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
    /// is packed into `[0, count)` each frame by [`Self::cull_visible`].
    pub instances: wgpu::Buffer,
    pub instance_capacity: u32,
    /// CPU cull records (full set), with precomputed bounding spheres.
    cull: Vec<CullInstance>,
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
        let mut meta: Vec<SpriteModelMeta> = Vec::with_capacity(registry.models.len());

        for m in &registry.models {
            meta.push(SpriteModelMeta {
                occupancy_offset: all_occ.len() as u32,
                colors_offset: all_colors.len() as u32,
                color_offsets_offset: all_offsets.len() as u32,
                occ_words_per_col: m.occ_words_per_col,
                dims: m.dims,
                _pad0: 0,
                pivot: m.pivot,
                _pad1: 0.0,
            });
            all_occ.extend_from_slice(&m.occupancy);
            all_colors.extend_from_slice(&m.colors);
            all_offsets.extend_from_slice(&m.color_offsets);
        }

        // Precompute per-model bound radius, then the per-instance cull
        // records (sphere centred at the instance position).
        let radii: Vec<f32> = registry
            .models
            .iter()
            .map(SpriteModel::bound_radius)
            .collect();
        let cull: Vec<CullInstance> = instances
            .iter()
            .map(|i| CullInstance {
                gpu: SpriteInstanceGpu {
                    inv_rot0: i.transform.inv_rot[0],
                    inv_rot1: i.transform.inv_rot[1],
                    inv_rot2: i.transform.inv_rot[2],
                    pos: i.transform.pos,
                    model_id: i.model_id,
                },
                center: i.transform.pos,
                radius: radii.get(i.model_id as usize).copied().unwrap_or(0.0),
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

        Self {
            occupancy: storage_u32(device, "roxlap-gpu sprite_reg.occupancy", &all_occ),
            colors: storage_u32(device, "roxlap-gpu sprite_reg.colors", &all_colors),
            color_offsets: storage_u32(device, "roxlap-gpu sprite_reg.color_offsets", &all_offsets),
            model_meta: storage_pod(device, "roxlap-gpu sprite_reg.model_meta", &meta),
            instances: instances_buf,
            instance_capacity: cull.len() as u32,
            cull,
        }
    }

    /// Frustum-cull the full instance set, pack the visible subset into
    /// the front of the instance buffer, and return the visible count.
    /// Conservative sphere test against the 4 side planes + the
    /// near(behind)/far planes. Allocates a small scratch vec per call.
    pub fn cull_visible(&self, queue: &wgpu::Queue, f: &ViewFrustum) -> u32 {
        let nw = (1.0 + f.half_w * f.half_w).sqrt();
        let nh = (1.0 + f.half_h * f.half_h).sqrt();
        let mut visible: Vec<SpriteInstanceGpu> = Vec::with_capacity(self.cull.len());
        for ci in &self.cull {
            let rel = [
                ci.center[0] - f.pos[0],
                ci.center[1] - f.pos[1],
                ci.center[2] - f.pos[2],
            ];
            let z = dot3(rel, f.forward);
            let r = ci.radius;
            if z + r < 0.0 || z - r > f.far {
                continue; // behind the camera or beyond far
            }
            let x = dot3(rel, f.right);
            if (x - f.half_w * z) > r * nw || (-x - f.half_w * z) > r * nw {
                continue; // right / left plane
            }
            let y = dot3(rel, f.down);
            if (y - f.half_h * z) > r * nh || (-y - f.half_h * z) > r * nh {
                continue; // bottom / top plane
            }
            visible.push(ci.gpu);
        }
        if !visible.is_empty() {
            queue.write_buffer(&self.instances, 0, bytemuck::cast_slice(&visible));
        }
        visible.len() as u32
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
}
