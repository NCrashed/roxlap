// GPU.2 validator — one-thread compute shader that reads a single
// voxel from the uploaded (occupancy, color_offsets, colors)
// storage buffers and writes its colour to `out[0]`. Output is 0
// for empty voxels.
//
// Occupancy layout (z-innermost, post-GPU.3 layout swap):
//   column (x, y) owns 8 contiguous u32 words at
//   `col_word_base = (x + y*vsid)*8`. Bit `z & 31` in word
//   `col_word_base + z/32` is voxel (x, y, z)'s occupancy.
//
// Rank-count of solid voxels at z' < z = sum of `countOneBits` over
// the `z/32` full words plus a masked partial. Mirrors
// `ChunkUpload::voxel_at` field-for-field.

const OCC_WORDS_PER_COLUMN: u32 = 8u; // CHUNK_Z (256) / 32

struct Probe {
    coord: vec3<u32>,
    vsid: u32,
    chunk_z: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> probe: Probe;
@group(0) @binding(1) var<storage, read> occupancy: array<u32>;
@group(0) @binding(2) var<storage, read> color_offsets: array<u32>;
@group(0) @binding(3) var<storage, read> colors: array<u32>;
@group(0) @binding(4) var<storage, read_write> out: array<u32, 1>;

@compute @workgroup_size(1)
fn debug_read() {
    let p = probe.coord;
    if (p.x >= probe.vsid || p.y >= probe.vsid || p.z >= probe.chunk_z) {
        out[0] = 0u;
        return;
    }
    let col_idx = p.x + p.y * probe.vsid;
    let col_word_base = col_idx * OCC_WORDS_PER_COLUMN;
    let z_word = p.z >> 5u;
    let z_bit = p.z & 31u;
    let solid = (occupancy[col_word_base + z_word] >> z_bit) & 1u;
    if (solid == 0u) {
        out[0] = 0u;
        return;
    }
    // Rank: popcount the full words below z, plus mask the partial.
    var rank: u32 = 0u;
    for (var w: u32 = 0u; w < z_word; w = w + 1u) {
        rank = rank + countOneBits(occupancy[col_word_base + w]);
    }
    var mask: u32 = 0u;
    if (z_bit > 0u) {
        mask = (1u << z_bit) - 1u;
    }
    rank = rank + countOneBits(occupancy[col_word_base + z_word] & mask);

    out[0] = colors[color_offsets[col_idx] + rank];
}
