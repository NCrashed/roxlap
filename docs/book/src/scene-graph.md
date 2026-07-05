# The scene graph

> **Stub** — this chapter is planned for stage BK.1 (see
> [`PORTING-BOOK.md`](https://github.com/NCrashed/roxlap/blob/master/docs/porting/PORTING-BOOK.md)).

Will cover grids, chunks, and `GridTransform`; runtime edits
(`set_voxel` / `set_rect` / `set_sphere`, colour callbacks, span ops —
including the recolour gotcha: `set_rect(Some(..))` over solid voxels
keeps the old colours, so carve first, then insert); serde snapshots;
chunk streaming with `ChunkGenerator` + `ChunkStore`.

Until then: [docs.rs/roxlap-scene](https://docs.rs/roxlap-scene).
