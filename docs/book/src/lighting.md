# Lighting & materials

> **Stub** — this chapter is planned for stage BK.3 (see
> [`PORTING-BOOK.md`](https://github.com/NCrashed/roxlap/blob/master/docs/porting/PORTING-BOOK.md)).

Will cover baked lighting (`bake_lightmode`, ambient-occlusion in the
ambient byte); the runtime `LightRig` (sun, point and spot lights,
stylized voxel shadows, banding); transparent-voxel materials (alpha,
additive, and Beer–Lambert volumetric — which needs
`from_fn_keep_interior`, since kv6 import culls interiors); and
terrain materials for water and glass.

Until then: the Lighting / Spotlight / Transparency demo scenes, and
[docs.rs/roxlap-render](https://docs.rs/roxlap-render).
