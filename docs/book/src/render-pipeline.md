# The render pipeline

> **Stub** — this chapter is planned for stage BK.2 (see
> [`PORTING-BOOK.md`](https://github.com/NCrashed/roxlap/blob/master/docs/porting/PORTING-BOOK.md)).

Will cover the fixed-resolution post pipeline: logical render
resolution (`Scale` / `Native`) that decouples frame rate from window
size, SSAA supersampling, posterize + dither (Bayer / blue-noise) for
the reduced-palette retro look, and the egui HUD.

Until then: `set_render_resolution` / `set_ssaa` / `set_posterize` in
[docs.rs/roxlap-render](https://docs.rs/roxlap-render), and the live
"Render pipeline" HUD panel in `roxlap-scene-demo`.
