# roxlap-cli — asset tool (mini-stage CLI)

Written 2026-07-06, right after stage BK closed. The optional
follow-on PORTING-BOOK.md proposed ("high indie value, zero coupling
to the book"), delivered as a single mini-stage — small enough that
this doc is a record, not a plan.

## What landed

`crates/roxlap-cli` — a binary crate over `roxlap-formats` +
`roxlap-scene`, no other dependencies (arguments are hand-parsed, the
house style; no CLI framework). Three subcommands:

- `info <file>` — identify + summarise. Detection is magic-based
  (`VOX `, `Kvxl`, `Kwlk`, `RVCL`, `RKCH`, `RXSS`) with an extension
  fallback for the two formats that have no magic (`.kvx`, `.vxl`).
  Snapshot info goes through `Scene::load_snapshot` and prints the
  grid list: id, name, chunk count, edited-chunk count
  (`chunk_versions() != 0`), origin.
- `vox2kv6 <in> <out>` — MagicaVoxel → kv6; a multi-model file fans
  out to `out_0.kv6`, `out_1.kv6`, ….
- `vox2rvc <in> <out> [frame_ms]` — models become the frames of one
  looping clip via `from_kv6_frames_auto` (cost-driven keyframe/delta
  choice); default 80 ms/frame.

Exit codes 0/1/2 (ok / operation failed / usage). Unit tests
synthesise a `.vox` in memory (same approach as the book's
`book_assets` example) and cover sniffing, summaries, both
conversions and the snapshot listing.

Wired up: workspace `members` + `default-members`, README crate
table, and a "The command-line tool" section in the book's asset
chapter. Publishing to crates.io is the maintainer's call at the next
release cut (the crate carries full publish metadata).

## Notes for future sessions

- `voxel_clip::ParseError` is Debug-only (no `Display`) — unlike the
  other formats' errors. The CLI formats it with `{e:?}`.
- Follow-ons LANDED 2026-07-06: `gif2rvc` / `png2rvc` (the cli now
  enables the formats crate's `gif`/`png` features; png2rvc takes an
  APNG/still or a frame list) and `kv62vox` over the new
  `vox::VoxFile::from_kv6` + `vox::serialize` (exact ≤ 255 colours,
  6-bit-bucket quantise beyond; pivot lost — `.vox` has none).
  Live-verified: magick-made GIF/PNG assets → rvc → `info`, and
  coco.kv6 → vox → kv6 keeps all 148 voxels. Still open: snapshot
  chunk extraction (no demand yet). Publish owed at the next release
  cut.
