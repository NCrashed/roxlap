# Foreword

This book teaches [roxlap](https://github.com/NCrashed/roxlap) — a
pure-Rust voxel-scene engine descended from Ken Silverman's Voxlap — to
a game developer who has never seen the codebase. It sits between the
[README](https://github.com/NCrashed/roxlap#readme) (the pitch and a
40-line quickstart) and the [docs.rs API
reference](https://docs.rs/roxlap-render) (every type and function):
the README tells you *whether* you want roxlap, docs.rs tells you *what
each item does*, and this book tells you *how the pieces fit together
and how to build a game with them*.

Three other kinds of documentation exist and are linked from here where
they help, but are not replaced by the book:

- **docs.rs** — the API reference. The book prefers linking a type over
  restating its signature, so the reference is always the fresher of
  the two.
- **`docs/porting/PORTING-*.md`** — internal stage histories: design
  rationale, dead ends, benchmarks. Read them when you want to know
  *why* something is built the way it is.
- **The demo scenes** (`roxlap-scene-demo`) — the runnable feature
  gallery. Every chapter's topic has a demo tab you can fly around in;
  the [demo tour](demo-tour.md) maps features to scenes.

Code in this book is never pasted by hand: every snippet longer than a
few lines is included directly from a compile-tested crate example or
doctest, so if the book and the engine disagree, CI is already red.

The book tracks the engine: chapters are re-checked as stages land,
and every include is CI-gated, so a snippet that stops compiling turns
the build red before it can mislead you. How the book itself was built
(and its anti-rot policy) is recorded in
[`docs/porting/PORTING-BOOK.md`](https://github.com/NCrashed/roxlap/blob/master/docs/porting/PORTING-BOOK.md).
