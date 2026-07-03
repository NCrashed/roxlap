# roxlap — fixed-resolution render target + posterize/SSAA post (Substage RP)

Start-of-stage brief and locked decisions for the **render-pipeline** rework:
render the scene into a **fixed-size offscreen target**, then **upscale-blit**
it to the swapchain, with an optional **posterize + dither** post pass and
**supersampling (SSAA)**. The tag is **RP**. A fresh-context session should
read this top to bottom before touching code.

Companion docs: [PORTING-GPU.md](PORTING-GPU.md) (the WGPU compute renderer +
`scene_blit.wgsl` this hooks into), [PORTING-DDA.md](PORTING-DDA.md) (the CPU
per-pixel raycaster whose cost is per-pixel), and the render facade
(`roxlap-render`, the `SceneRenderer` over CPU+GPU).

The change is **purely additive**: the default resolution mode is `Native`
(logical resolution == window), which is **byte-identical** to today. The
posterize/SSAA path is opt-in. The headline regression gate
(`Native` + no posterize ⇒ pixel-identical to pre-RP) holds trivially.

## Why

Today the renderer writes straight to the swapchain/window surface at **window
resolution**, on both backends:

- **GPU** — the `scene_dda` compute shader writes a packed-`rgba8` storage
  buffer sized `window_w × window_h`
  (`roxlap-gpu/src/lib.rs:643-679` `SceneDdaResources.framebuffer`,
  `:987-991` shader store), then `scene_blit.wgsl` fullscreen-triangles it
  onto the swapchain (`lib.rs:2796-2817`). Storage buffer + swapchain are both
  sized at window resolution (`surface_config.width/height`).
- **CPU** — the raycaster fills an owned `framebuffer: Vec<u32>` grown lazily
  to **window size** (`roxlap-render/src/cpu.rs:288-378`, render `:855-893`),
  then software-blits via softbuffer (desktop, `:1456-1475`) or a WebGL2
  fullscreen quad (wasm, `:1479-1489`).

Because both backends are **per-pixel raycasters**, frame cost scales directly
with the pixel count — so **FPS is coupled to window size**. Resizing the
window silently changes the framerate. This is the primary pain RP fixes.

The fix is the standard one: render the scene into a **decoupled, fixed-size**
target and resample it to the window. That also creates the natural seam for a
**posterize** post (the engine user wants a configurable reduced-palette retro
look) and for **SSAA** (the right anti-shimmer tool for a crisp-retro
aesthetic — see Locked decisions).

### Two separate problems (do not conflate)

1. **FPS coupled to window size** — pure consequence of per-pixel cost. Solved
   completely by the fixed-size target (RP.0) and nothing else.
2. **Shimmer / crawl** on camera rotation + movement — this is **aliasing**
   (undersampling the voxel signal), *amplified* by posterization (small
   colour changes cross quantization boundaries and flip). Addressed by
   **SSAA + blue-noise dither** (RP.1 + RP.2), not by geometric tricks.

## Locked decisions

Taken with the engine author 2026-07-01. The author's stated priorities:
**(a) FPS stability is the main pain**, **(b) the target aesthetic is "crisp
retro"** (sharp limited palette, hard pixels — voxlap/PSX vibe).

1. **Three-resolution model.** A frame flows through up to three resolutions:
   - **render_dims** — where the raycaster actually runs (compute / CPU). With
     SSAA, `render_dims = logical_dims × ssaa`.
   - **logical_dims** — the fixed "retro pixel grid" the image is resolved and
     posterized at (e.g. `960×540`, or `window × scale`). This is the buffer
     that decouples cost from window size.
   - **swapchain** — the native window. The logical image is **nearest**-upscaled
     onto it (chunky retro pixels).

   So `render_dims ≥ logical_dims ≤ swapchain`. For crisp retro: small fixed
   `logical_dims`, `render_dims = logical × ssaa` for clean pre-quantize
   sampling, `swapchain = window` (nearest upscale of the posterized grid).

2. **Default `Native` is byte-identical.** `RenderResolution::Native` ⇒
   `logical_dims = window`, `ssaa = 1`, no posterize ⇒ the resolve+blit
   collapse to today's straight blit. Pixel-identical regression gate.

3. **Crisp-retro resolve order.** Resolve pass: **box-downfilter**
   `render_dims → logical_dims` (averages the SSAA samples → smooth pre-quantize
   signal), then **blue-noise dither**, then **quantize** (posterize). Final
   blit: **nearest** `logical_dims → swapchain`. Downfilter-before-quantize is
   the whole point — it makes the quantization buckets stable frame-to-frame,
   which is the real anti-shimmer for a posterized look. (Rejected: quantize
   before downfilter — reintroduces crawl.)

4. **SSAA, not TAA, for shimmer.** SSAA attacks aliasing at the sampling source
   and is compatible with the crisp-retro look (the final upscale is still
   nearest, pixels stay hard). **TAA is explicitly rejected** here: it blurs the
   limited palette and ghosts on disocclusion — against the chosen aesthetic.
   (Motion vectors would be nearly free in a DDA raycaster — every pixel knows
   its exact world hit point — so TAA stays a *documented future option* if the
   aesthetic ever shifts toward "smooth & stable"; see Risks R5.)

5. **No cubemap-around-camera, no world-probe texel-splatting.** Both were
   considered for shimmer and **rejected**:
   - A camera-centred cubemap only stabilizes *rotation*, costs *more* pixels
     (6 faces), adds seams, and only pays off as an amortized angular render
     *cache* in the "can't re-render the screen fast enough" regime — which is
     not roxlap's (CPU ~40–70 FPS, GPU hundreds). Dominated by SSAA for this
     goal.
   - World-anchored texel-splatting is really a **radiance/temporal cache** for
     GI / many-lights reuse, not an anti-aliasing tool — wrong instrument, huge
     complexity (disocclusion, invalidation, memory).

   Both are recorded as future options **only if goals change** (cubemap →
   360°/VR/stereo; world-probe → global illumination), not for RP.

6. **egui HUD stays native-res, composited AFTER upscale on BOTH backends.**
   The HUD must be crisp regardless of `logical_dims`.
   - GPU already does this correctly: `paint_egui` runs a separate render pass
     with `LoadOp::Load` on the swapchain at `surface_config.width/height`
     (`roxlap-gpu/src/lib.rs:3509-3578`) — it composites over the *blit output*,
     so it is naturally native-res. **No change needed.**
   - CPU is **wrong today**: `paint_egui` software-rasterizes egui *into the
     scene framebuffer* before blit (`cpu.rs:1495-1520`, `cpu_egui.rs`). Once
     the framebuffer is the small `logical_dims`, the HUD would upscale with the
     scene. **RP.0 must move the CPU egui raster to a native-size output buffer,
     after the logical→native upscale.**

7. **No new crate.** Resolution + posterize knobs live on the `SceneRenderer`
   facade (`roxlap-render/src/lib.rs`), mirrored by hand into `cpu.rs` + `gpu.rs`
   (the standing duck-typed pattern — no backend trait). GPU post is WGSL
   (extend/duplicate `scene_blit.wgsl` into a resolve pass + a nearest-upscale
   pass); CPU post is plain Rust in the blit path.

## The data model

```rust
// roxlap-render/src/lib.rs  (facade additions)

/// Decouples the raycaster's pixel count from the window size.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RenderResolution {
    /// logical_dims == window. Default. Byte-identical to pre-RP.
    Native,
    /// Fixed logical grid, independent of window (the retro pixel grid).
    Fixed { w: u32, h: u32 },
    /// logical_dims = round(window * factor). 0.5 = quarter the pixels.
    Scale(f32),
}
impl Default for RenderResolution { fn default() -> Self { Self::Native } }

/// Per-channel quantization. levels <= 1 ⇒ that channel is untouched.
#[derive(Clone, Copy, Debug)]
pub struct PosterizeConfig {
    pub levels_r: u8,
    pub levels_g: u8,
    pub levels_b: u8,
    pub dither: DitherMode,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DitherMode { #[default] None, Bayer4x4, BlueNoise }

impl SceneRenderer {
    /// Set the logical (fixed) render resolution. `Native` ⇒ today's behaviour.
    pub fn set_render_resolution(&mut self, res: RenderResolution);
    /// Supersampling factor. 1 = off; 2 = 2×2 samples per logical pixel.
    /// render_dims = logical_dims * factor. (GPU affords 2; CPU costs 4× rays.)
    pub fn set_ssaa(&mut self, factor: u8);
    /// Reduced-palette post. `None` = off (default).
    pub fn set_posterize(&mut self, cfg: Option<PosterizeConfig>);
    /// Introspection: the resolution the raycaster actually runs at this frame.
    pub fn render_dims(&self) -> (u32, u32);
    pub fn logical_dims(&self) -> (u32, u32);
}
```

Resolution resolution (sic) happens once per frame from the current window
size: `logical_dims = match res { Native => window, Fixed{w,h} => (w,h),
Scale(f) => round(window*f) }`; `render_dims = logical_dims * ssaa`.

## Engine work (per stage)

**RP.0 — fixed-size offscreen + nearest upscale** (both backends + facade).
Introduce `logical_dims` (decoupled from window) and a final nearest-upscale to
the swapchain. `render_dims == logical_dims` at this stage (no SSAA yet).
- **Facade:** `RenderResolution` enum + `set_render_resolution` +
  `render_dims`/`logical_dims`; compute `logical_dims` per frame; plumb to the
  active backend on resize/first-render.
- **GPU:** size `SceneDdaResources.framebuffer` + `depth_buffer` to
  `render_dims` instead of `surface_config` (`lib.rs:643-679`); dispatch compute
  over `render_dims` (`:2777-2783`); the sprite pass writes the same buffer at
  render_dims (no special-case). `scene_blit.wgsl` becomes a **nearest upscale**:
  sample `fb[ floor(uv * logical_dims) ]` for each swapchain pixel; extend the
  `blit_dims` uniform to carry both src(logical) + dst(swapchain) dims
  (`:2796-2817`).
- **CPU:** `framebuffer`/`zbuffer` sized to `render_dims` (`cpu.rs:855-893`); a
  new native-size **output buffer**; `blit_and_present` nearest-upscales
  framebuffer→output, then presents the output (softbuffer/WebGL2). **Move egui:
  `paint_egui` rasterizes into the native output AFTER upscale** (locked
  decision #6), not into the small framebuffer.
- `Native` ⇒ `logical==window`, upscale is identity ⇒ **byte-identical**.

**RP.1 — SSAA + resolve (box downfilter)** (both backends).
`render_dims = logical_dims × ssaa`. Add a **resolve** step that box-averages
`render_dims → logical_dims`:
- **GPU:** a resolve pass (compute or fragment) reading the `render_dims`
  framebuffer, writing a `logical_dims` buffer (box average of the `ssaa²`
  samples). The existing blit then nearest-upscales the logical buffer.
- **CPU:** the upscale step generalises to a **resampler** — box-average when
  `render_dims > logical_dims`, nearest when equal. (Note CPU cost: ssaa=2 ⇒ 4×
  the rays — keep ssaa a knob; default 1 on CPU, 2 acceptable on GPU.)
- No posterize yet ⇒ pure anti-aliasing; visually smoother edges, stable under
  rotation/movement.

**RP.2 — posterize + dither** (both backends, in the resolve step).
Configurable per-channel quantization, applied **at logical_dims, after the box
downfilter, before nothing else** (it is the last op before the nearest
upscale). Blue-noise (or Bayer) dither **before** quantize.
- **GPU:** fold into the resolve pass — `sample → box average → dither threshold
  → quantize → store logical buffer`. Blue-noise via a small tiled noise texture
  (or a cheap hash); Bayer via a 4×4 constant matrix. Pass `PosterizeConfig` as
  a uniform.
- **CPU:** same math in the resampler in Rust.
- `posterize == None` ⇒ resolve is the RP.1 box-average verbatim.

**RP.3 — demo + docs.** Wire the knobs into the scene-demo HUD (egui sliders:
resolution mode + fixed-grid size, SSAA factor, posterize levels + dither mode)
so the retro look + FPS-decoupling is demonstrable and tunable live. README
feature row + per-crate docs + CHANGELOG `[Unreleased]`. Confirm FPS is now
**invariant to window size** (the headline win) and shimmer is reduced under
SSAA+dither.

## Code map (as of 2026-07-01)

Facade — `crates/roxlap-render/src/`:
- `lib.rs:1483-1488` `resize()` (dispatch), `:1500-1505` `render()`,
  `:1729-1734` `present()`, `:1765-1775` `paint_egui()`. Add the resolution +
  posterize knobs + per-frame `logical_dims`/`render_dims` computation here.
- `cpu.rs:288-378` `CpuBackend` (`framebuffer: Vec<u32>`, `zbuffer`,
  `last_dims`, `present_target`); `:855-893` `render()` (lazy resize to window —
  retarget to render_dims); `:1171-1176` `present()` → `:1456-1475` softbuffer
  blit / `:1479-1489` WebGL2 blit (insert the resampler + native output here);
  `:1495-1520` `paint_egui()` + `cpu_egui.rs` `EguiRaster` (move to native
  output, post-upscale).
- `gpu.rs:910-912` `resize()`, `:1061-1063` `present()` (dispatch).

GPU — `crates/roxlap-gpu/src/`:
- `lib.rs:643-679` `SceneDdaResources` (`framebuffer`/`depth_buffer`/`blit_dims`
  storage; `storage_size`) — size to render_dims; `:487-590` `GpuRenderer`
  (`surface_config`, `scene_dda`, `pending_frame`); `:1580-1592` `resize()`
  (surface reconfig + pipeline invalidation); `:2281-2823` `render_scene()`
  (`:2777-2783` compute dispatch → resize to render_dims; `:2796-2817` blit
  render pass → becomes resolve + nearest-upscale); `:3509-3578` `paint_egui()`
  (native-res, `LoadOp::Load` — leave as is).
- `shaders/scene_dda.wgsl:987-991` framebuffer store (dims come from a uniform);
  `shaders/scene_blit.wgsl:28-42` fullscreen-triangle blit → split into a
  **resolve** shader (box downfilter + dither + posterize, render→logical) + a
  **nearest-upscale** blit (logical→swapchain). `shaders/sprite_model_dda.wgsl`
  writes the same framebuffer/depth at render_dims (no special-case).

CPU — `crates/roxlap-core/src/`:
- the per-pixel raycaster fills the facade `framebuffer` at whatever dims it is
  handed — retargeting it to render_dims is a dims change at the call site, no
  core algorithm change.

⚠️ Do **not** size the `depth_buffer` (GPU sprite depth) or `zbuffer` (CPU) to
the swapchain — they must track **render_dims** alongside the colour buffer, or
sprite occlusion desyncs from the scene.

## Sub-substage roadmap

| Stage | Scope | Gate |
|---|---|---|
| **RP.0** ✅ | Fixed-size offscreen `logical_dims` (decoupled from window) + nearest upscale to swapchain (both backends + facade). `RenderResolution {Native, Fixed, Scale}` + `set_render_resolution` + `render_dims()`/`logical_dims()`. GPU: `framebuffer`/`depth_buffer`/compute dispatch/`screen_size`/sprite-cull all at render size; `scene_blit.wgsl` integer-nearest-upscales (`blit_dims` carries src+dst, size 16→32); line/image overlays project at render aspect + scale `clip.xy`→render for the depth lookup (`LineParams.depth_w/h`); pick maps window→render. CPU: framebuffer/zbuffer at logical + native `output` buffer + nearest resampler; egui rasterises into the native output post-upscale; pick maps window→logical. `render_dims == logical_dims` (no SSAA). | **Done** — `Native` byte-identical by construction (integer nearest map ⇒ identity; flip/egui order preserved); `Fixed/Scale` ⇒ FPS invariant to window size. `cargo test/clippy/build --workspace` green (`render_resolution_logical_for` unit test; `wgsl_shaders_validate` covers the 3 edited shaders; wasm-target check clean). Demo defaults to `Fixed{860,520}`, `ROXLAP_RENDER_RES` override. GPU + interactive **visual user-verification pending** (headless CI has no display — standing caveat). |
| **RP.1** ✅ | SSAA: `render_dims = logical × ssaa` + resolve (box downfilter march→logical). `set_ssaa(factor)` (clamp 1..=4). GPU: `scene_resolve.wgsl` compute pass framebuffer(march)→`resolve_buf`(logical); blit reads `resolve_buf` (always — `ssaa==1` is a byte-exact 1×1 copy). CPU: `downfilter_pixel` integer box average → `resolve` buffer, then the RP.0 nearest upscale; `CpuSrc {Frame,Resolve,Output}` selects the present source. `render_dims()`=march, `logical_dims()`=logical; pick/overlays key off march (depth buffer is march-sized). | **Done** — `ssaa==1` byte-exact identity (RP.0 paths unchanged); `ssaa==2/4` runs on CPU + RTX 3070, 0 panics, predictable march cost. `downfilter_pixel` unit tests (identity/uniform/rounding); `wgsl_shaders_validate` covers `scene_resolve.wgsl`; clippy 0; full workspace green. Demo `ROXLAP_SSAA` (default 1). GPU/interactive **visual user-verification pending**. |
| **RP.2** ✅ | Posterize + dither in the resolve step. `PosterizeConfig {levels_r/g/b, dither}` + `DitherMode {None, Bayer4x4, BlueNoise}` + `set_posterize(Option<…>)`. Box-average → dither → quantize at logical_dims, before the nearest upscale. GPU: folded into `scene_resolve.wgsl` (`quantize` + `dither_offset`; posterize fields written per-frame into the resolve uniform at offset 20 — no rebuild). CPU: `posterize_pixel`/`quantize_channel`/`dither_offset` (`resolve_scene` now runs when posterize is set even at ssaa==1). BlueNoise = texture-free interleaved-gradient noise (identical CPU/GPU formula). | **Done** — `posterize==None` ⇒ RP.1 verbatim (levels=[1,1,1] ⇒ untouched, byte-identical); quantization stable thanks to downfilter→dither→quantize order. `quantize_channel`/`posterize_pixel` unit tests (identity/2-level/4-level palette/per-channel/dither-varies); `wgsl_shaders_validate` covers the extended resolve shader; clippy 0; full workspace green. Demo `ROXLAP_POSTERIZE=N` + `ROXLAP_DITHER`. Ran live both backends (levels 2/4/6/8 × none/bayer/blue × ssaa 1/2), 0 panics. GPU/interactive **visual user-verification pending**. |
| **RP.3** ✅ | scene-demo "Render pipeline" egui panel (`pipeline_panel` + `PipelineUi` in `host.rs`): resolution mode (Native/Fixed `w×h`/Scale) + SSAA slider + posterize toggle/levels/dither combo, live logical+march readout. Edits a `PipelineUi` copy, diffs vs the stored state, pushes changes via `PipelineUi::apply` (the env vars now seed the initial state). README + per-crate docs + CHANGELOG `[Unreleased]`. | **Done** — builds + clippy 0; both backends run the panel, 0 panics; live diff-apply rebuilds the GPU scene_dda on a dims change / writes posterize per-frame. Version bump left to maintainer. GPU + interactive **visual user-verification pending** (headless CI has no display — standing caveat). |

One sub-stage per commit, each green on `cargo test/clippy/build --workspace`.
RP.0 lands the decoupling (the FPS win) first; SSAA + posterize layer on top.

## Tests

- **Regression anchor (CI):** `RenderResolution::Native` + `posterize==None`
  renders **pixel-identical** to pre-RP — the headline gate (mirror the GPU
  `HeadlessSceneRenderer` diff harness used in GPU.11.0; CPU compares the
  framebuffer hash).
- **Resolution decoupling (CI/headless):** `Fixed{w,h}` produces an
  `w×h` logical buffer regardless of window size; `render_dims`/`logical_dims`
  introspection returns the expected values for `Native`/`Fixed`/`Scale`.
- **Resampler (CI):** box downfilter of a known `render_dims` pattern →
  `logical_dims` matches a hand-computed average; nearest upscale of a known
  logical buffer → swapchain matches expected pixel replication.
- **Posterize (CI):** per-channel quantization to N levels maps known inputs to
  expected buckets; `levels<=1` is identity; `DitherMode::None` is deterministic.
- **Visual:** RP.3 demo (manual GPU verification, no display in CI) — FPS
  invariance to window size + retro look.

## Risks / watch-items

- **R1 — depth/colour resolution desync.** GPU `depth_buffer` / CPU `zbuffer`
  must track `render_dims`, not swapchain. Mitigation: size them together at the
  same call site; sprite-occlusion test in the demo.

- **R2 — CPU SSAA cost.** `ssaa=2` quadruples CPU rays. Mitigation: ssaa is a
  knob, default 1 on CPU; document the cost; the fixed `logical_dims` already
  bounds the base ray count (the FPS-stability win is independent of SSAA).

- **R3 — egui upscaled on CPU (regression trap).** Today's CPU `paint_egui`
  rasterizes into the framebuffer pre-blit; if not moved to the native output
  it will blur with the scene. Mitigation: RP.0 explicitly relocates it
  (locked decision #6); a HUD-sharpness visual check in RP.3.

- **R4 — posterize amplifies shimmer if mis-ordered.** Quantizing before the
  box downfilter reintroduces crawl. Mitigation: enforced resolve order
  (downfilter → dither → quantize, locked decision #3); the dither is the
  palette-stabilizer.

- **R5 — aesthetic lock-in to crisp-retro.** SSAA + nearest upscale is chosen
  for the hard-pixel look and rejects TAA. If the goal ever shifts to
  "smooth & stable", TAA is a clean future add (free motion vectors from the
  DDA world-hit point) — recorded, not built.

- **R6 — wasm parity.** The WebGL2 blit (`cpu.rs:1479-1489`) must gain the same
  resample + native-output + egui-post-upscale path as desktop softbuffer.
  Mitigation: keep the resampler backend-agnostic (operates on the `Vec<u32>`
  buffers); only the final present differs.

## Validation (every sub-substage)

- `cargo test/clippy/build --workspace` green; **`Native` + no posterize ⇒
  byte-identical** is the headline regression gate.
- Resampler + posterize math unit-tested independent of any backend.
- "No silent caps": `Fixed`/`Scale` that exceed device storage limits or round
  to zero are rejected/clamped with a `log::warn!`, never silently.
- GPU/interactive paths dogfooded in the RP.3 demo (manual visual — headless CI
  has no display, standing caveat).
