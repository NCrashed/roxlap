# roxlap-web

Browser demo of the roxlap engine — same `opticast` raycaster +
scalar rasterizer the native demos use, compiled to
`wasm32-unknown-unknown` with WebAssembly SIMD (`simd128`)
batches, running on a `<canvas>` via wasm-bindgen + 2D
`putImageData` blits. Single-threaded for v1 (rayon's wasm
threads via SharedArrayBuffer + COOP/COEP are deferred to a
future R10.X).

## Quick start

The flake dev shell (`nix develop`) ships
[`trunk`](https://trunkrs.dev/) and `wasm-bindgen-cli` already.
From the repo root:

```sh
nix develop
cd crates/roxlap-web
trunk serve         # opens http://localhost:8080
```

Trunk hot-reloads the wasm bundle on edits to `src/lib.rs` /
`index.html` / `Trunk.toml`. First build is ~15 s; subsequent
incremental builds are sub-second.

For a production deploy:

```sh
trunk build --release
# dist/ now contains index.html + the wasm + JS shim.
# Upload the directory to any static host.
```

`trunk build --release` runs `wasm-opt` on the wasm output, which
typically halves the binary (debug ~900 KB → release ~460 KB).

### Cross-origin isolation (required for wasm threads)

R10.X.2's `wasm-bindgen-rayon` thread pool relies on
`SharedArrayBuffer`, which the browser only enables when the
page is **cross-origin isolated**. The static host serving
`dist/` must attach two response headers to the HTML and wasm:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

Trunk's dev server attaches these automatically (see
`Trunk.toml` `[serve.headers]`). For production:

* **Cloudflare Pages**: add a `_headers` file at the dist root.
* **Netlify**: `netlify.toml` `[[headers]]` block.
* **GitHub Pages**: not currently supported — needs a custom
  domain + Cloudflare in front, or use a different host.
* **Custom nginx / caddy / traefik**: add the headers per the
  upstream docs.

Without these headers, the demo loads but `initThreadPool`
hangs at the worker spawn step.

## Controls

| key | action |
|---|---|
| W / A / S / D / arrows | translate camera (horizontal plane) |
| Space / Shift | translate camera up / down |
| click canvas | request pointer lock for mouse-look |
| mouse | yaw + pitch (pointer-lock active) |
| Esc | release pointer lock |
| **B** | run a 300-frame bench, dump min / p50 / mean / p99 / max ms + fps to console |

The devtools **Console** logs:

- `parsed oracle.vxl in N ms` on first load.
- `frame N | mean M.M ms over last 60` once per second of run.
- `bench: 300 frames | min ... | p50 ... | mean ... | p99 ... | max ... ms — N fps` after pressing B.

## Performance

WebAssembly SIMD (`simd128`) is enabled via
`.cargo/config.toml`'s `target-feature=+simd128`. The release
wasm contains 700+ SIMD instructions including
`f32x4_sqrt` + `f32x4_div` from the `hrend` / `vrend` 4-pixel
batches plus rustc auto-vectorization on the surrounding code.

Wasm performance is typically 1.5–2× slower than native on the
same machine — same algorithm, different instruction set, no
multi-threading. Native baseline at 640×480 single-threaded is
~11 ms / frame on an i7-12700H; expect ~17–22 ms / frame in
Chrome 91+ / Firefox 89+ / Safari 16.4+ on the same machine.

The actual number is what the `B` bench prints — copy-paste the
output rather than relying on this estimate.

## Bundle size

| component | dev profile | release profile (post-`wasm-opt`) |
|---|---|---|
| `roxlap-web-*.wasm` | ~760 KB | ~360 KB |
| `roxlap-web-*.js` (wasm-bindgen shim) | ~18 KB | ~14 KB |
| `index.html` | 2 KB | 2 KB |
| **total** | **~780 KB** | **~376 KB** |

Of that, ~207 KB is the embedded `oracle.vxl.gz` (the world the
demo renders) and ~1.2 MB → ~150 KB after wasm-opt is the
sky / sprite assets `include_bytes!`-baked into the wasm.

## Goldens

The wasm renderer has its own
[`crates/roxlap-oracle/tests/wasm-hashes.txt`](../roxlap-oracle/tests/wasm-hashes.txt)
goldens file — wasm SIMD's full-precision `f32x4_sqrt` +
`f32x4_div` produces different bytes than x86_64's
`_mm_rsqrt_ps` (12-bit approximation) or aarch64's
`vrsqrteq_f32` + Newton (~16-bit). The CI matrix's wasm job
runs all 12 oracle poses through `wasm-bindgen-test` under Node
and diffs against that file:

```sh
cargo test --target wasm32-unknown-unknown -p roxlap-oracle --test wasm_render
```

## Browser support

Requires WebAssembly SIMD (`simd128`):

- Chrome / Edge 91+ (May 2021)
- Firefox 89+ (June 2021)
- Safari 16.4+ (March 2023)

Older browsers fail at the wasm loader stage. A scalar-only
fallback build is tracked as R10.X follow-up work; ~94 % of
2026 browsers ship simd128, so the priority is low.
