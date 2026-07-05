# Rendering & backends

> **Stub** — this chapter is planned for stage BK.2 (see
> [`PORTING-BOOK.md`](https://github.com/NCrashed/roxlap/blob/master/docs/porting/PORTING-BOOK.md)).

Will cover the `SceneRenderer` facade in full: `BackendPreference` and
automatic CPU fallback; the `render → overlays → present` /
`paint_egui` frame protocol; the `FrameParams::new` builder; the
`supports()` backend-parity query; and an overview of how the CPU
per-pixel DDA and the GPU compute marcher differ.

Until then: [docs.rs/roxlap-render](https://docs.rs/roxlap-render).
