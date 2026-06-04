// GPU.2 validator — one-thread compute shader that reads a single
// voxel from the uploaded (occupancy, color_offsets, colors)
// storage buffers and writes its colour to `out[0]`. Output is 0
// for empty voxels.
//
// Mirrors the CPU `ChunkUpload::voxel_at` logic. Used to prove the
// upload round-trips without bit corruption.

struct Probe {
    coord: vec3<u32>,
    vsid: u32,
    chunk_z: u32,
    // Padding to align the storage buffer to 16 bytes. WGSL doesn't
    // need it but it keeps the bytemuck Pod struct on the Rust side
    // straightforward.
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> probe: Probe;
@group(0) @binding(1) var<storage, read> occupancy: array<u32>;
@group(0) @binding(2) var<storage, read> color_offsets: array<u32>;
@group(0) @binding(3) var<storage, read> colors: array<u32>;
@group(0) @binding(4) var<storage, read_write> out: array<u32, 1>;

fn voxel_solid(p: vec3<u32>) -> bool {
    if (p.x >= probe.vsid || p.y >= probe.vsid || p.z >= probe.chunk_z) {
        return false;
    }
    let i = p.x + p.y * probe.vsid + p.z * probe.vsid * probe.vsid;
    let word = occupancy[i >> 5u];
    return (word & (1u << (i & 31u))) != 0u;
}

@compute @workgroup_size(1)
fn debug_read() {
    let p = probe.coord;
    if (!voxel_solid(p)) {
        out[0] = 0u;
        return;
    }
    // Count solid voxels at z < probe.z in column (x, y) → that's
    // the rank within color_offsets.
    var rank: u32 = 0u;
    let col_base = p.x + p.y * probe.vsid;
    for (var z: u32 = 0u; z < p.z; z = z + 1u) {
        let i = p.x + p.y * probe.vsid + z * probe.vsid * probe.vsid;
        let word = occupancy[i >> 5u];
        let bit = (word >> (i & 31u)) & 1u;
        rank = rank + bit;
    }
    let base = color_offsets[col_base];
    out[0] = colors[base + rank];
}
