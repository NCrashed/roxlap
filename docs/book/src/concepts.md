# Concepts & conventions

> **Stub** — this chapter is planned for stage BK.1 (see
> [`PORTING-BOOK.md`](https://github.com/NCrashed/roxlap/blob/master/docs/porting/PORTING-BOOK.md)).

Will cover the load-bearing conventions: **+z is DOWN**; packed colours
`0x80_RR_GG_BB` (brightness in the high byte, not alpha); one voxel =
one world unit; the camera basis and the chirality footgun
(right × down must equal forward — use the `Camera` constructors); f64
world coordinates vs f32 sprite transforms.

Until then: the [quickstart](introduction.md) flags the first two, and
the `Camera` rustdoc in
[`roxlap-core`](https://docs.rs/roxlap-core) documents the basis rules.
