//! Companion example for the book's "The scene graph" chapter
//! (`docs/book/src/scene-graph.md`) — the chapter pulls its snippets
//! from here via `// ANCHOR:` markers, so everything it shows compiles
//! and its assertions actually ran. Headless: builds, edits, saves and
//! streams a scene without ever opening a window.
//!
//! ```sh
//! cargo run -p roxlap-scene --example book_scene_graph
//! ```
//!
//! Keep the anchors when editing; `docs/book/check-anchors.sh` (run by
//! the CI `book` job) goes red if one disappears.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use glam::{DQuat, DVec3, IVec3};
use roxlap_formats::vxl::Vxl;
use roxlap_scene::{
    ChunkGenerator, ChunkStore, GridTransform, Scene, SpanOp, StreamRadius, VoxColor, CHUNK_SIZE_XY,
};

const GRASS: VoxColor = VoxColor::rgb(0x4d, 0x8a, 0x3a);
const STONE: VoxColor = VoxColor::rgb(0x8a, 0x8a, 0x8a);
const RED: VoxColor = VoxColor::rgb(0xc0, 0x30, 0x30);

// ANCHOR: generator
/// A deterministic floor generator: chunk layer `z = 0` gets a solid
/// slab, checkerboarded per chunk so streamed tiles are visible;
/// every other layer is left as air. `generate` must be a pure
/// function of `chunk_idx` (plus the generator's own config) — that
/// determinism is what makes evict + re-stream sound.
#[derive(Debug)]
struct FloorGenerator;

impl ChunkGenerator for FloorGenerator {
    fn generate(&self, chunk_idx: IVec3) -> Vxl {
        // `Vxl::empty(CHUNK_SIZE_XY)` is the canonical all-air chunk
        // shape — the same one `Grid::ensure_chunk` materialises.
        let mut vxl = Vxl::empty(CHUNK_SIZE_XY);
        if chunk_idx.z == 0 {
            let light = (chunk_idx.x + chunk_idx.y).rem_euclid(2) == 0;
            let color = if light {
                VoxColor::rgb(0x74, 0x9e, 0x58)
            } else {
                VoxColor::rgb(0x5d, 0x84, 0x46)
            };
            roxlap_formats::edit::set_rect(&mut vxl, [0, 0, 240], [127, 127, 255], Some(color));
        }
        vxl
    }
}
// ANCHOR_END: generator

// ANCHOR: chunk_store
/// Minimal in-memory [`ChunkStore`]: edited chunks handed over at
/// eviction, returned on stream-in (so player edits survive walking
/// away). A real game would write a save-file region instead.
#[derive(Debug, Default)]
struct MemoryStore {
    chunks: Mutex<HashMap<IVec3, (Vxl, u64)>>,
}

impl ChunkStore for MemoryStore {
    fn store(&self, chunk_idx: IVec3, vxl: &Vxl, version: u64) {
        let entry = (vxl.clone(), version);
        self.chunks
            .lock()
            .expect("store poisoned")
            .insert(chunk_idx, entry);
    }

    fn load(&self, chunk_idx: IVec3) -> Option<(Vxl, u64)> {
        self.chunks
            .lock()
            .expect("store poisoned")
            .get(&chunk_idx)
            .cloned()
    }
}
// ANCHOR_END: chunk_store

fn main() {
    // ANCHOR: scene_grids
    let mut scene = Scene::new();
    // A static "world" grid at the origin…
    let ground_id = scene.add_grid(GridTransform::at(DVec3::ZERO));
    // …and a second grid placed and rotated independently — its own
    // f64 world position + quaternion, same chunked voxel payload.
    let ship_id = scene.add_grid(GridTransform {
        origin: DVec3::new(300.0, -80.0, -40.0),
        rotation: DQuat::from_rotation_z(0.5),
        voxel_world_size: 1.0,
    });
    // Object grids should not paint their own (grid-local, rotating)
    // sky — leave the sky to the world grid.
    scene.grid_mut(ship_id).expect("just added").render_sky = false;
    // ANCHOR_END: scene_grids

    // ANCHOR: edits
    let ground = scene.grid_mut(ground_id).expect("just added");
    // Solid slab: grid-local voxel coords, inclusive on both ends,
    // decomposed across as many chunks as it spans (here 4×4).
    ground.set_rect(
        IVec3::new(-200, -200, 200),
        IVec3::new(199, 199, 254),
        Some(GRASS),
    );
    // `Some(colour)` inserts, `None` carves back to air.
    ground.set_sphere(IVec3::new(0, 0, 195), 12, Some(STONE));
    ground.set_rect(IVec3::new(-3, -200, 190), IVec3::new(3, 199, 196), None);
    // ANCHOR_END: edits

    // ANCHOR: queries
    // Solid tests: inside the slab, then inside the carved tunnel.
    assert!(ground.voxel_solid(IVec3::new(0, 0, 210)));
    assert!(!ground.voxel_solid(IVec3::new(0, 100, 193)));
    // Colour queries answer on surface voxels; deep interior cells
    // are untextured in the slab format and read as `None`.
    assert_eq!(ground.voxel_color(IVec3::new(50, 50, 200)), Some(GRASS));
    // ANCHOR_END: queries

    // ANCHOR: recolour
    // GOTCHA: inserting over already-solid voxels does NOT repaint
    // them — span insertion only fills air. Recolour = carve, then
    // insert.
    let (lo, hi) = (IVec3::new(20, 20, 200), IVec3::new(24, 24, 204));
    ground.set_rect(lo, hi, Some(RED));
    assert_eq!(ground.voxel_color(IVec3::new(22, 22, 200)), Some(GRASS)); // unchanged!
    ground.set_rect(lo, hi, None); // carve…
    ground.set_rect(lo, hi, Some(RED)); // …then insert
    assert_eq!(ground.voxel_color(IVec3::new(22, 22, 200)), Some(RED));
    // ANCHOR_END: recolour

    // ANCHOR: colfunc
    // A carve exposes fresh interior walls; the plain edits paint
    // them colour 0 (black). The `_with_colfunc` variants ask a
    // closure instead — it sees grid-local coordinates, so gradients
    // stay continuous across chunk seams. Here: a crater whose walls
    // darken with depth.
    ground.set_sphere_with_colfunc(IVec3::new(60, 60, 200), 8, SpanOp::Carve, |_x, _y, z| {
        let depth = (z - 192).clamp(0, 15) as u8;
        VoxColor::rgb(0x6b, 0x40 + depth * 3, 0x2f)
    });
    // ANCHOR_END: colfunc

    // ANCHOR: snapshot
    // Name grids before saving: ids survive the round-trip, but the
    // generator / store hooks are host code and cannot be serialised
    // — the name is your key for rebinding them after a load.
    scene.grid_mut(ground_id).expect("ground exists").name = Some("ground".into());
    let bytes = scene.save_snapshot(); // versioned envelope
    let restored = Scene::load_snapshot(&bytes).expect("self-authored snapshot");
    assert_eq!(restored.grid_count(), 2);
    let (_, ground2) = restored
        .grids()
        .find(|(_, g)| g.name.as_deref() == Some("ground"))
        .expect("rebind by name");
    assert_eq!(ground2.voxel_color(IVec3::new(22, 22, 200)), Some(RED));
    // ANCHOR_END: snapshot

    // ANCHOR: streaming
    // Streaming: attach a generator + radii to a grid, then pump once
    // per frame with the camera's world position. Chunks within
    // `r_active` stream in; chunks beyond `r_evict` drop out; the band
    // between is hysteresis so a camera hovering on a boundary
    // doesn't thrash.
    let stream_id = scene.add_grid(GridTransform::at(DVec3::new(0.0, 4096.0, 0.0)));
    let store = Arc::new(MemoryStore::default());
    let grid = scene.grid_mut(stream_id).expect("just added");
    grid.set_generator(Some(Arc::new(FloorGenerator)));
    grid.set_chunk_store(Some(store));
    grid.stream_radius = StreamRadius::new(256.0, 384.0);

    let camera_near = DVec3::new(0.0, 4096.0, 100.0);
    scene.pump_streaming_sync(camera_near); // real games: `pump_streaming`
    let grid = scene.grid(stream_id).expect("still there");
    assert!(grid.chunk_count() > 0); // floor streamed in around the camera
    assert!(grid.voxel_solid(IVec3::new(5, 5, 240)));
    // ANCHOR_END: streaming

    // ANCHOR: streaming_edit
    // Edit the streamed floor, walk far away, come back. The active
    // set follows the camera: the old region evicts (edited chunks
    // are handed to the ChunkStore first) while a fresh set streams
    // in around the new position. On return the store wins over the
    // generator, so the edit survives; without a store, eviction
    // would silently revert the chunk to generator output.
    scene
        .grid_mut(stream_id)
        .expect("still there")
        .set_voxel(IVec3::new(5, 5, 240), None); // dig a hole
    scene.pump_streaming_sync(camera_near + DVec3::new(0.0, 100_000.0, 0.0));
    let grid = scene.grid(stream_id).expect("still there");
    assert!(grid.chunk(IVec3::ZERO).is_none()); // home chunk evicted…
    assert!(grid.chunk_count() > 0); // …but the far region streamed in
    scene.pump_streaming_sync(camera_near);
    let grid = scene.grid(stream_id).expect("re-streamed");
    // The hole survived evict + re-stream.
    assert!(!grid.voxel_solid(IVec3::new(5, 5, 240)));
    // ANCHOR_END: streaming_edit

    println!("book_scene_graph: all scene-graph assertions hold");
}
