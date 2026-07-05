{
  description = "roxlap — pure-Rust port of Ken Silverman's Voxlap voxel engine";

  inputs = {
    nixpkgs.url = "flake:nixpkgs";
    # R10.X.2: nightly Rust toolchain for wasm-bindgen-rayon. Needs
    # `-Z build-std` to rebuild std with `+atomics` enabled, plus
    # `rust-src` available as a component. rust-overlay is the
    # idiomatic Nix way to get a custom rustc.
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, rust-overlay }:
    let
      forAllSystems = f:
        nixpkgs.lib.genAttrs [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ]
          (system: f {
            pkgs = import nixpkgs {
              inherit system;
              overlays = [ rust-overlay.overlays.default ];
            };
          });
    in {
      devShells = forAllSystems ({ pkgs }:
        let
          # Runtime libs winit + softbuffer dlopen on Linux. Listed
          # both for X11 (most NixOS desktops today) and Wayland
          # (Sway, GNOME on Wayland, …) so the demo Just Works on
          # either backend. macOS uses CoreGraphics natively and
          # needs none of these.
          linuxRuntimeLibs = with pkgs; [
            libxkbcommon
            wayland
            # Top-level names since nixpkgs flattened the xorg.* set.
            libx11
            libxcursor
            libxi
            libxrandr
            libxcb
            # GPU.0: wgpu dlopens libvulkan.so.1 at runtime to reach
            # the Mesa/Nvidia ICDs. NixOS desktops carry the ICDs
            # themselves via hardware.graphics; the loader has to be
            # findable on LD_LIBRARY_PATH for non-NixOS-managed
            # binaries (i.e. ours).
            vulkan-loader
            # `roxlap-sdl-demo` links libSDL2 (the `sdl2` crate finds
            # it via pkg-config — see `buildInputs` below — and the
            # runtime loader needs it on LD_LIBRARY_PATH too).
            SDL2
          ];

          # R10.X.2: pin a specific nightly via rust-toolchain.toml.
          # The toolchain bundles `rust-src` (required by
          # `-Z build-std`) and the wasm32-unknown-unknown target so
          # cargo can rebuild std with `+atomics` for wasm threads.
          # QE.0 scoped the pin to the web crates (plain rustup clones
          # build on stable, matching CI); the devShell keeps reading
          # the same file so one source-of-truth controls both.
          rustToolchain =
            pkgs.rust-bin.fromRustupToolchainFile
            ./crates/roxlap-web/rust-toolchain.toml;
        in {
          default = pkgs.mkShell {
            packages = with pkgs; [
              # R10.X.2: nightly Rust toolchain replaces the previous
              # stable `rustc + cargo + rustfmt + clippy +
              # rust-analyzer` set. The fenix/rust-overlay-built
              # bundle includes all of those plus rust-src and the
              # wasm32-unknown-unknown target.
              rustToolchain
              pkg-config
              # R10.0: wasm32-unknown-unknown needs an LLD-class
              # linker; nixpkgs's `rustc` doesn't bundle `rust-lld`,
              # so we provide the system one. Cargo finds it via
              # `lld` on PATH (rustc's wasm target spec invokes the
              # bare name).
              lld
              # R10.1: `wasm-bindgen-test` runs the test wasm under
              # Node (V8). `wasm-bindgen-cli` produces the JS shim
              # (its `wasm-bindgen-test-runner` is the cargo runner);
              # Node executes the shim.
              wasm-bindgen-cli
              nodejs
              # R10.2: `trunk` is the dev-server / bundler for the
              # `roxlap-web` + `roxlap-cave-web` crates.
              trunk
              # Image inspection — imagemagick converts PPM frame dumps
              # to PNG for eyeballing renderer output and pixel diffing.
              imagemagick
              # BK.0: the user-facing engine book lives in `docs/book/`
              # (mdBook). `mdbook serve docs/book` for live preview;
              # broken include-anchors are caught by
              # `docs/book/check-anchors.sh` (CI runs it — mdbook alone
              # exits 0 on broken includes).
              mdbook
            ] ++ pkgs.lib.optionals pkgs.stdenv.isLinux linuxRuntimeLibs;

            # `sdl2` (roxlap-sdl-demo) links libSDL2 at build time via
            # pkg-config; mkShell exposes `buildInputs` on
            # PKG_CONFIG_PATH so `sdl2.pc` is discoverable. macOS uses
            # the SDL2 framework / Homebrew copy on the default path.
            buildInputs = pkgs.lib.optionals pkgs.stdenv.isLinux [ pkgs.SDL2 ];

            # winit + softbuffer link these via dlopen at runtime;
            # nix mkShell only sets PATH / PKG_CONFIG_PATH, so we add
            # the lib search path explicitly. macOS skips this — the
            # Cocoa / Metal frameworks are always on the dyld path.
            shellHook = pkgs.lib.optionalString pkgs.stdenv.isLinux ''
              export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath linuxRuntimeLibs}:''${LD_LIBRARY_PATH:-}"
            '';
          };
        });
    };
}
