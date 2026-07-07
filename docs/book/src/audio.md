# Audio

Most voxel engines bolt sound on as an afterthought: a positional
source, a distance falloff, done. roxlap's `roxlap-audio` crate does
what the *renderer* does — it reads the actual voxels. A shot behind a
wall is muffled by exactly as much rock as the sound crosses; a gun
fired in a cavern rings, the same gun in a crawl-space is dry.

The crate splits cleanly in two, and the split is the whole design:

- The **acoustics core** turns a [`Scene`](scene-graph.md) plus a
  source and listener into plain numbers — an occlusion gain + lowpass
  cutoff per source, a reverb feedback + wet mix per listener. It owns
  no audio device, spawns no thread, and is fully deterministic, so
  it's unit-tested against synthetic caves and you can call it from any
  thread.
- A **playback backend** applies those numbers. The built-in one wraps
  [kira](https://crates.io/crates/kira) behind the `kira` cargo
  feature; a host with its own audio stack implements the `AudioOut`
  trait instead.

The snippets below come from a runnable example that prints the numbers
the core computes for a little scene — no audio device needed:

```sh
cargo run -p roxlap-audio --example book_audio
```

## Occlusion: hearing through rock

A source in a sealed room, a listener outside, a wall with a doorway:

```rust,noplayground
{{#include ../../../crates/roxlap-audio/examples/book_audio.rs:scene}}
```

Occlusion is a small fan of rays from the source to the listener. Each
ray accumulates the **solid thickness** it crosses — not a yes/no hit,
the actual voxel path length — and the average becomes a *transmission*
in `0..=1` that drives a muffling lowpass and a volume drop:

```rust,noplayground
{{#include ../../../crates/roxlap-audio/examples/book_audio.rs:occlusion}}
```

```text
   line of sight: transmission 1.00  cutoff  20000 Hz  gain -0.0 dB
through the wall: transmission 0.17  cutoff   1399 Hz  gain -19.8 dB
past the doorway: transmission 0.36  cutoff   2527 Hz  gain -15.4 dB
```

Thickness is why this reads as sound and not as a gate: one voxel of
wall muffles less than five, and because the fan spreads, a doorway off
the direct line lets *some* rays through — the listener past the door
hears the source brighter and louder than through the sealed wall, but
still not clear. Sound leaks around corners instead of snapping on and
off. `AcousticsConfig` tunes the fan (ray count, jitter radius,
per-voxel absorption, the cutoff endpoints).

## Reverb: the size of the space

The second half fires a fan the other way — from the **listener**,
outward in every direction. How far the rays travel before hitting a
wall reads as room size; the fraction that escape to open sky reads as
openness:

```rust,noplayground
{{#include ../../../crates/roxlap-audio/examples/book_audio.rs:cavity}}
```

```text
     in the room: openness 0.00  reverb feedback 0.53  mix 0.50
 out in the open: openness 0.59  reverb feedback 0.66  mix 0.00
```

A `CavityEstimator` maps that to reverb feedback (decay) and a wet mix,
and — crucially — **smooths it over time**. Real rooms don't change
acoustics the instant you step through a doorway, and a per-frame probe
would jitter the reverb audibly; the estimator eases toward each new
reading over a second or so, which players read as natural. Call it at
a couple of hertz, not every frame, and feed the result to the reverb
with a long (~1 s) tween. Note the open-air line: even outdoors, half
the fan hits the ground the listener stands on (openness `0.59`, not
`1.0`) — so dryness can't come from openness reaching 1. Instead the
wet mix falls to **zero** once openness crosses a threshold (default
`0.5`, an open-ground listener's floor), which is what makes outdoors
read as dry however far the sky-bound rays fly. `CavityConfig` tunes
the fan size, that outdoor-openness threshold, and the
feedback / wet-mix ranges.

## Playback: the `AudioOut` boundary

The core hands you parameters; a backend turns them into sound. That
seam is the `AudioOut` trait — `register` a sound, `set_listener`,
`play` / `play_loop` a source, and `apply_source` / `apply_listener`
the computed acoustics:

```rust,noplayground
// Per acoustic tick (roxlap-audio computes, the backend applies):
let occ = source_acoustics(&scene, source, listener, &acfg);
audio.play(shot, source, Some(&occ));          // starts already muffled
let env = cavity.update(&scene, listener);     // ~2 Hz
audio.apply_listener(&env);                     // ~1 s tween
```

The built-in `KiraAudio` (feature `kira`) implements it with a proper
aux-send topology: one shared reverb on a send track, a pool of spatial
voices each carrying its own lowpass filter and a per-source reverb
send. Parameters are applied with **tweens** — fast (~120 ms) for
per-source occlusion and the listener pose, slow (~1 s) for the reverb
environment — so nothing zippers. A fresh one-shot, though, starts
*already* at its occluded parameters (`play`'s `initial` argument): a
gunshot's envelope is gone in a tenth of a second, long before a ramp
would arrive, so it must be muffled from the first sample. Voices are a
fixed pool allocated up front and reused — one-shots steal the oldest
finished voice when the pool is full; loops hold their slot until
stopped.

The `kira` backend is native-only and needs an audio device (ALSA on
Linux). It's off by default, so a plain build pulls in no audio stack
at all; wasm support (kira runs on WebAudio) is scoped for a later
stage.

## The cave demo

The cave demo is the worked example — build it with the `audio`
feature:

```sh
cargo run --release -p roxlap-cave-demo --features audio
```

Every plasma shot cracks at the muzzle, each impact booms at the crater
it carves, and every glowing crystal hums — all muffled by the rock
between them and you, with reverb that swells as you fly into a big
chamber and dries as you leave it. The crystal hums show the voice
budget at work: only the nearest handful loop at once (with hysteresis
so a crystal at the edge of earshot doesn't flicker), started and
stopped as you move, so a crystal-rich cave never drowns out the shots.
The wiring — one small `DemoAudio` struct driving the core and the kira
backend from the demo's fire / carve / per-frame hooks — lives in
[`crates/roxlap-cave-demo/src/audio.rs`](https://github.com/NCrashed/roxlap/blob/master/crates/roxlap-cave-demo/src/audio.rs).

## Further reading

- [`PORTING-AUDIO.md`](https://github.com/NCrashed/roxlap/blob/master/docs/porting/PORTING-AUDIO.md)
  — the AU-stage design history: the ray budgets and smoothing borrowed
  from Sound Physics Remastered, Teardown and Overwatch, why kira, and
  the aux-bus / voice-stealing lessons.
- [The scene graph](scene-graph.md) and
  [Picking & world queries](picking.md) — the `Scene::raycast` and
  voxel-query machinery the acoustic rays are built on.
