# roxlap-sdl-demo

An **SDL2** host for the roxlap scene-graph engine.

It exists to demonstrate one thing: [`roxlap-render`](../roxlap-render)
is not tied to winit. The renderer binds to whatever
[`raw-window-handle`](https://docs.rs/raw-window-handle) provider you
hand it — winit (see [`roxlap-scene-demo`](../roxlap-scene-demo)), SDL2
(this crate), GLFW, or your own surface. The engine only ever sees the
two `raw-window-handle` traits plus an explicit pixel size.

## Run

```sh
nix develop          # provides SDL2 + the toolchain
cargo run -p roxlap-sdl-demo
# GPU (wgpu) backend, falls back to CPU softbuffer automatically:
ROXLAP_GPU=1 cargo run -p roxlap-sdl-demo
```

Without Nix, install SDL2 development libraries first (e.g.
`apt install libsdl2-dev`, `brew install sdl2`) so the `sdl2` crate can
link against them via `pkg-config`.

### Controls

| Key            | Action                |
| -------------- | --------------------- |
| `W` `A` `S` `D` | Move (horizontal)    |
| `Space` / `Shift` | Fly up / down      |
| Mouse          | Look (relative mode)  |
| `Esc`          | Quit                  |

## How the binding works

`roxlap-render`'s `SceneRenderer::new` is generic:

```rust
pub fn new<W>(window: Arc<W>, size: (u32, u32), opts: &RenderOptions) -> Self
where
    W: HasWindowHandle + HasDisplayHandle + Send + Sync + 'static,
```

winit's `Window` satisfies these bounds directly. SDL2's window does
**not** — it is `!Send`/`!Sync` (bound to its creating thread), while
wgpu's `Surface<'static>` requires `Send + Sync`. So this demo snapshots
the window's two raw handles once and wraps them in a small
`Send + Sync` adapter (`SdlWindowHandle` in [`src/main.rs`](src/main.rs)).

The adapter owns no SDL state — only `Copy` raw handles — so the
`unsafe impl Send + Sync` is sound **as long as the backing SDL window
outlives it**. `main` guarantees that by keeping the `Window` alive for
the whole program and dropping the renderer first. The same pattern
applies to any window provider whose handle type isn't already
`Send + Sync`.

Presentation is owned entirely by the renderer: the SDL window is
created *without* an SDL renderer, and softbuffer (CPU) / wgpu (GPU)
draws straight to its surface via the raw handle.
