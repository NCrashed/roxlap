# roxlap — animated voxel-sprite clips (`.rvc`) + multi-attachment RKC

Start-of-stage brief and locked decisions for **animated voxel sprites**
— a "GIF/MP4 for voxel models": a fixed-bbox sequence of voxel frames
encoded as keyframes + inter-frame diffs, for effects like flame, spells,
muzzle flashes, magic auras. Plus the **RKC integration** that lets a
character bone carry *several* attachments (today: exactly one mesh per
bone), each of which can be a static `.kv6` **or** an animated clip.
Companion to [PORTING-SCENE.md](PORTING-SCENE.md),
[PORTING-GPU.md](PORTING-GPU.md), and
[PORTING-SPRITE-API.md](PORTING-SPRITE-API.md) (the dynamic sprite API
this builds on).

This is a **start-of-stage brief**. A fresh-context session should read
it top to bottom before touching code. The stage tag is **VCL**.

## Why

`roxlap` animates voxels two ways today, both *transform-only*:

1. **Skeletal** — `.rkc` `Character` / `KfaSprite`: a bone hierarchy
   whose hinge transforms animate over keyframes (`ClipData::Skeletal`,
   `character.rs:126`). Each bone draws **exactly one** mesh
   (`Bone.mesh: MeshRef`, `character.rs:98`; `MeshRef::Static(usize)` is
   the only variant, `character.rs:111`).
2. **Per-instance pose** — `set_sprite_instance_transforms` (the
   PORTING-SPRITE-API stage): cheap, batched, coalesced per frame.

Neither animates the **voxel volume itself**. A flame's *shape* changes
frame to frame; no bone transform expresses that. There is no
time-varying voxel data anywhere in the crate, and no temporal/diff
encoding for voxel volumes (the only delta coding is `.vxl`'s *spatial*
column RLE). We want a first-class "voxel video" that the engine plays
back cheaply and that demiurg authors + previews **through the engine**
(editor and runtime pixel-identical).

## Key enabling facts

- **The render pipeline already separates the cheap axis from the
  expensive one.** Per-instance transform updates are batched +
  coalesced into one device write (`transforms_dirty`, `gpu.rs:565`);
  *volume* upload is the expensive, "assumed-static" path
  (`build_sprite_model` + `SpriteRegistryResident::upload`,
  `sprite_model.rs:65,768`). So we **never re-upload voxels per frame** —
  we pre-decode every frame, upload them all once, and per frame only
  *select* which one shows.
- **The GPU sprite model layout IS our frame layout.** `SpriteModel`
  (`sprite_model.rs:30`) is: `dims[3]`, per-column `occupancy` bitmask
  (`Vec<u32>`), ascending-z `colors`/`dirs` runs, `color_offsets` prefix
  sums, `pivot`, `voxel_world_size`. A clip frame stored in that exact
  layout uploads with **no `build_sprite_model` bucket-sort** and diffs
  cleanly per column.
- **Effects are tiny.** A 24-frame flame at ~200 voxels/frame in a
  ~16³ bbox ≈ ~30 KB GPU resident. Pre-uploading every frame
  (flipbook) is the obvious default; memory is a non-issue at effect
  scale.
- **The `.rkc` chunk envelope already anticipates this.** `character.rs`
  is a forward-compatible `magic + version + [tag|len|payload]`
  container with `extra_chunks` preservation; its doc explicitly names a
  future `MeshRef::Video(usize)` / new `mesh_kind` as the extension seam
  (`character.rs:59-68,114`).
- **The KFA path is the template** for "register once, drive per frame
  cheaply": `set_kfa_sprites` uploads each limb volume once + seeds one
  instance (`gpu.rs:365`); `update_kfa_poses` only rewrites transforms
  (`gpu.rs:399`, no volume re-upload). We mirror this for clips, adding
  one new per-frame primitive: *select an instance's model*.

## Locked decisions

Taken with the engine author 2026-06-25:

1. **Flipbook runtime.** Decode all N frames at load, upload all as
   models, and per frame **swap which model the instance shows**. No
   per-frame volume re-upload. (Rejected: streaming one model + applying
   a column-diff per frame via a budgeted `update_model` — kept as a
   noted future option for very large clips, §"Risks".)
2. **Dense-column frame type** (not "decode each frame to `Kv6`"). A
   clip frame is stored directly in the `SpriteModel`-style column
   layout. This makes the diff codec clean (per-column) and the GPU
   upload near-zero-cost; it requires new *CPU* decode plumbing (a
   raycaster path that consumes the column layout) parallel to today's
   `Kv6`-only `dda_sprite`.
3. **`.rkc` bumps to v3** with a clean data model: `Bone.mesh: MeshRef`
   → `Bone.attachments: Vec<Attachment>`. The format is young (v2), so a
   clean break beats a split data model. v2 files are rejected with
   `UnsupportedVersion` (no migration; re-export from demiurg).
4. **Frames step, never blend.** Voxel occupancy can't be meaningfully
   lerped, so playback selects the current frame discretely (unlike
   skeletal clips' `BoneXform::blend`). Per-frame durations + a loop mode
   drive selection.
5. **No new crate.** `voxel_clip` lands in `roxlap-formats`; GPU/CPU
   decode in `roxlap-gpu`/`roxlap-core`; the player API on the
   `roxlap-render` facade. `roxlap-formats` stays dependency-free of
   `roxlap-gpu` — the canonical `VoxelFrame` is backend-agnostic and each
   backend converts it (GPU → `SpriteModel`; CPU → dense grid).

## The format — `VoxelClip` (`.rvc`)

### `VoxelFrame` (the backend-agnostic frame)

Mirrors `SpriteModel` (`sprite_model.rs:30`) so GPU upload is a move, not
a rebuild:

```rust
// roxlap-formats/src/voxel_clip.rs
pub struct VoxelFrame {
    pub occupancy: Vec<u32>,      // per-column z-bitmask; occ_words_per_col * mx*my words
    pub colors: Vec<u32>,         // ascending-z per column, packed 0x80RRGGBB
    pub color_offsets: Vec<u32>,  // prefix sums, len mx*my + 1  (colors[off[c]..off[c+1]] = column c)
    // dirs (surface-normal LUT indices) are recomputed at decode from the
    // reconstructed full-frame occupancy (compute_vis_dir, kv6.rs:62),
    // so the on-disk codec stays "occupancy + colour" only.
}
```

Clip-level (shared by every frame, fixed): `dims: [u32;3]` (the union
bbox of all frames — fixed dims keep diffs + the GPU in-place path
valid), `pivot: [f32;3]`, `voxel_world_size: f32`, `occ_words_per_col`.

### Container layout (RKCH-style, forward-compat)

```
magic    b"RVCL"
version  u16 = 1
chunks   [tag(4) | len(u32) | payload]  until EOF;  unknown tags → extra_chunks
  META : dims[3] u32, pivot[3] f32, voxel_world_size f32, frame_count u32,
         loop_mode u8 {Loop=0, Once=1, PingPong=2}, default_frame_ms u32
  FRMS : the frame stream (codec below)
  TIME : optional per-frame durations (u32 ms × frame_count) + an optional
         Seq{tim,frm} schedule (reused from kfa.rs:78) for non-uniform /
         loop-point control; absent ⇒ uniform default_frame_ms, plain loop
```

Reuse `bytes::Cursor` (LE), the `write_chunk` back-patch-length idiom
(`character.rs:541`), and the `extra_chunks` skip/re-emit pattern.

### Frame codec (FRMS)

Each frame is tagged I (keyframe) or P (diff-from-previous):

- **I-frame** — full `VoxelFrame`: `occupancy` words, `color_offsets`,
  `colors`. Optionally deflate (dev-dep `flate2` already vendored;
  gate behind a per-chunk "compressed" flag). Frame 0 is always I; emit
  a periodic I every K frames so seeking/looping never replays the whole
  stream.
- **P-frame** — a **sparse changed-column list**: `changed_count`
  (varint), then per changed column `{ col_index (varint), new
  occ_word(s), color_run_len (varint), color_run[] }`. Decode = clone the
  previous full frame, overwrite each listed column's occupancy +
  colours, rebuild `color_offsets` (prefix sum). Unchanged columns are
  skipped — for flame this is a handful of columns per frame.

`VoxelClip::decode() -> DecodedClip { frames: Vec<VoxelFrame>, durations,
loop_mode }` expands I/P → full frames once at load + computes each
frame's `dirs`. Decode is CPU-only and `Send`, so a big clip can decode
off-thread (the cave-demo carve-worker pattern).

## RKC v3 — multi-attachment bones

`character.rs` changes (VERSION 2 → 3):

```rust
pub struct Bone {
    pub name: String,
    pub attachments: Vec<Attachment>,   // was: mesh: MeshRef
    pub hinge: Hinge,
}
pub struct Attachment {
    pub target: MeshRef,            // Static(kv6_idx) | Clip(voxel_clip_idx)
    pub local_offset: BoneXform,    // place/orient this attachment on the bone
    pub playback: ClipPlayback,     // loop mode, speed, start phase (ignored for Static)
}
pub enum MeshRef {
    Static(usize),                  // mesh_kind 0 → Character::meshes[idx]
    Clip(usize),                    // mesh_kind 1 → Character::voxel_clips[idx]  (NEW)
}
pub struct ClipPlayback { pub loop_mode: LoopMode, pub speed_q8: i32, pub start_phase_ms: u32 }

pub struct Character {
    // …unchanged: name, root, meshes, bones, clips, extra_chunks…
    pub voxel_clips: Vec<VoxelClip>,   // NEW — VCLP chunk
}
```

- New top-level chunk `VCLP` = `count u32` + length-prefixed embedded
  `voxel_clip::serialize` blobs (mirrors how `MSHS` embeds `kv6::serialize`,
  `character.rs:561`).
- `BONS` payload per bone grows an attachment list (`count` +
  `[mesh_kind|index|local_offset(40B BoneXform)|playback]`). This is the
  v2→v3 break.
- `to_kfa_sprite` (`character.rs:322`): now produces a sprite **per
  attachment** (not per bone), posed by `bone_world × local_offset`.
- `to_kfa` (lossy voxlap export, `character.rs:361`): can't represent
  clips/multi-attachment — export the **first `Static` attachment** per
  bone, warn + drop the rest. Document.

Backward-compat shape: a bone with a single `Static` attachment at
identity offset == today's behaviour.

## Engine rendering (CPU + GPU)

Mirror the KFA "register once / drive per frame" split.

### GPU (`roxlap-gpu` + `roxlap-render/src/gpu.rs`)

- `VoxelFrame → SpriteModel`: trivial field move (same layout); skip
  `build_sprite_model`. Optionally `add_lod` per frame for distance LOD
  (more memory; gate per clip).
- **New low-level primitive — select an instance's model.** Today an
  instance's `model_id` is fixed at creation and only rewritten by the
  per-frame cull from its chain (`cull_bin_upload`, `sprite_model.rs:1436`).
  Add `SpriteRegistryResident::set_instance_model(cull_idx, chain_id)`
  (rewrite `cull[i].chain` / model ref — CPU-side, picked up by the next
  cull upload, no volume write) + a `GpuRenderer` passthrough. This is
  the per-frame flipbook step and is as cheap as a transform update.
- Register a clip = `add_lod`/`add_model` each frame's `SpriteModel`
  once → a contiguous chain-id range = the flipbook. Per frame:
  `set_instance_model(instance, flipbook[frame_idx])`.

### CPU (`roxlap-core/src/dda_sprite.rs` + `roxlap-render/src/cpu.rs`)

- `dda_sprite` today rebuilds a dense grid from a `Kv6` **every frame**
  (`Kv6Dense::build`, `dda_sprite.rs:44` — its hotspot). For clips: add
  a `VoxelFrame → dense (occ+col+dir)` decode and **cache all frames'
  dense grids at register**; per frame select `grids[frame_idx]` and
  raycast it (the existing `cast_local` march, generalised off `Kv6Dense`
  to a shared dense view). No per-frame rebuild.

### Facade API (`roxlap-render/src/lib.rs`)

```rust
pub struct VoxelClipId { slot: u32, gen: u32 }   // like SpriteModelId

impl SceneRenderer {
    pub fn add_voxel_clip(&mut self, clip: &DecodedClip) -> VoxelClipId;   // uploads the flipbook
    pub fn remove_voxel_clip(&mut self, id: VoxelClipId) -> bool;
    pub fn add_clip_instance_posed(&mut self, clip: VoxelClipId, xf: DynSpriteTransform) -> SpriteInstanceId;
    pub fn set_clip_instance_frame(&mut self, id: SpriteInstanceId, frame: u32);  // model-select
}
```

Plus a higher-level driver mirroring `update_kfa_poses`:
`advance_voxel_clips(dt)` ticks each clip instance's playback clock →
frame index → `set_clip_instance_frame` + (for bone attachments)
`set_sprite_instance_transform`. The character runtime (the
attachment-aware evolution of `KfaSprite`) solves bone transforms, then
emits one instance per attachment: `Static` → static model + transform;
`Clip` → flipbook + frame + transform, each with its **own** playback
clock (a flame loops regardless of the skeletal clip).

## Code map (as of 2026-06-25)

Formats — `crates/roxlap-formats/src/`:
- `kv6.rs:30,99,116` `Voxel`/`Kv6`; `compute_vis_dir` (`:62`) — reuse for
  decode-time `dirs`. `serialize`/`parse` (`:574,492`).
- `character.rs:78,98,111,119,126` `Character`/`Bone`/`MeshRef`/`Clip`/
  `ClipData`; chunk I/O `:289,220,541`; `to_kfa_sprite` `:322`,
  `to_kfa` `:361`; VERSION `:50`.
- `kfa.rs:78,86,301,415` `Seq`/`Kfa`/`KfaSprite`/`animsprite` (timing
  model to reuse).
- `xform.rs:197,220,250` `BoneXform` (`from_hinge_angle`, `blend`).
- `bytes.rs` LE `Cursor`; `lib.rs` module list (add `pub mod voxel_clip;`).

GPU — `crates/roxlap-gpu/src/sprite_model.rs`:
- `SpriteModel` (`:30`), `build_sprite_model` (`:65`),
  `SpriteModelRegistry::{add,add_lod,update_model,remove,compact}`
  (`:181,191,1013,281,1363`), resident `update_transforms`/`upload`/
  `cull_bin_upload` (`:978,768,1436`), `CullInstance` model ref (the
  per-frame select target). Shader `shaders/sprite_model_dda.wgsl`.

CPU — `crates/roxlap-core/src/dda_sprite.rs`:
- `draw_sprite_dda` (`:185`), `Kv6Dense::build` (`:44`, the per-frame
  rebuild to cache away), `cast_local` (`:126`).

Facade — `crates/roxlap-render/src/`:
- `lib.rs` sprite API (`:1075,1107,1206,1263,1280,1307,1321`),
  `SpriteModelId`/`SpriteInstanceId`/`DynModelMap`/`DynInstanceMap`
  (`:97,111,176,121`).
- `gpu.rs` KFA path `set_kfa_sprites`/`update_kfa_poses` (`:365,399`),
  `transforms_dirty` flush (`:565`), `chunk_upload_budget` (`:103`, the
  budget pattern to copy if needed).
- `cpu.rs` sprite draw loop (`:694`), `set_kfa_sprites`/`update_kfa_poses`
  (`:596,602`).

## Sub-stages (VCL.0 – VCL.7)

- **VCL.0 — `voxel_clip` format.** `VoxelFrame` + `VoxelClip` container +
  `serialize`/`parse` + I/P codec + `decode()` (with decode-time `dirs`).
  Unit tests: round-trip byte-equal; `decode(P-frame) == authored full
  frame`; union-bbox/dims; loop/seek.
- **VCL.1 — encoder.** Author N full frames → pick I-frames (frame 0 +
  every K) → column-diff the rest. `VoxelClip::from_frames(&[VoxelFrame],
  timing)`. Tests: encode→decode identity; diff size shrinks on a
  mostly-static clip.
- **VCL.2 — GPU flipbook.** `VoxelFrame → SpriteModel`; upload a clip as a
  chain-id range; `set_instance_model` primitive on resident +
  `GpuRenderer`. Headless gpu test (a 2-frame clip; selecting frame 1
  changes the rendered pixel — the `scene_render.rs` adapter runs in CI).
- **VCL.3 — CPU flipbook.** `VoxelFrame → dense` cache; generalise
  `dda_sprite` off a shared dense view; per-frame select. Pixel test:
  posed frame 0 vs frame 1 differ.
- **VCL.4 — facade API.** `VoxelClipId` + `add_voxel_clip` /
  `add_clip_instance_posed` / `set_clip_instance_frame` /
  `advance_voxel_clips`; CPU-backend lifecycle test (CI-safe).
- **VCL.5 — RKC v3.** `Bone.attachments`, `Attachment`, `MeshRef::Clip`,
  `Character.voxel_clips`, `VCLP` chunk, v3 serialize/parse (reject v2),
  `to_kfa_sprite`/`to_kfa` updates. Round-trip tests incl. a clip
  attachment + multi-attachment bone.
- **VCL.6 — character attachment runtime.** Evolve `KfaSprite` (or a new
  `CharacterInstance`) to emit one instance per attachment, with
  per-clip playback clocks; facade integration (the attachment-aware
  successor to `set_kfa_sprites`/`update_kfa_poses`).
- **VCL.7 — dogfood + docs.** scene-demo: a looping flame `.rvc` attached
  to coco's hinged arm (exercises multi-attachment + clip playback on
  both backends). demiurg authoring notes; CHANGELOG; per-crate docs.
  Optional: a tiny procedural flame generator that emits a `VoxelClip`
  (so the demo ships an asset without a hand-painted one).

## Tests

- **Formats (CI):** codec round-trip + diff-decode equality + decode-time
  `dirs` parity vs a from-scratch `compute_vis_dir`.
- **GPU (CI, software adapter — `scene_render.rs` already runs headless):**
  upload a 2-frame flipbook, assert `set_instance_model` swaps the
  rendered colour; instance-model select leaves transforms intact.
- **CPU (CI):** dense-cache frame select renders frame N's silhouette;
  no per-frame rebuild (assert the cache is built once).
- **RKC v3 (CI):** byte round-trip of a `Character` with a `Clip`
  attachment + a 2-attachment bone; `extra_chunks` preserved; v2 rejected.
- **Render correctness:** pin a tiny pixel assertion per backend (clip
  frame 0 vs 1), not a hash (dda_sprite is clean-room).

## Risks / watch-items

- **R1 GPU register-time upload spike.** A flipbook uploads N volumes at
  register. One-time, but if many clips register in one frame it can hit
  the staging-pool exhaustion the chunk budget (`gpu.rs:103`) exists to
  prevent. Register clips at load, or spread across frames with a budget.
- **R2 CPU per-frame cost.** The win depends on **caching** decoded dense
  grids at register; never run `Kv6Dense::build`-style rebuilds per
  frame. Memory = N dense grids per clip (fine at effect scale).
- **R3 fixed-dims waste.** All frames pad to the union bbox; a clip with
  one giant frame over-allocates the rest. Acceptable for effects; the
  encoder can warn on a high pad ratio.
- **R4 lossy `.kfa` export** can't carry clips/multi-attachment — keep
  first `Static`, warn. Documented.
- **R5 streaming-diff (deferred).** For a very large clip the flipbook's
  memory could matter; the alternative is one model + a budgeted
  per-frame `update_model` column-diff (the codec already has the diffs).
  Keep behind a per-clip threshold; not built in this stage.
- **R6 no inter-frame blend** — by design (occupancy can't lerp). If
  smooth motion is wanted, that's a higher frame rate or a procedural
  generator, not blending.

## Commit sequencing

One sub-stage per commit (VCL.0 … VCL.7), each green on `cargo
test/clippy/build --workspace`. Formats + encoder land first (pure-CPU,
fully CI-testable); GPU/CPU flipbook + facade next; RKC v3 + the
attachment runtime last; demo/docs close the stage. Targets roxlap
0.15.0 (new public formats + facade surface + an `.rkc` major bump).
