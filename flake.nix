{
  description = "roxlap — pure-Rust port of Ken Silverman's Voxlap voxel engine";

  inputs.nixpkgs.url = "flake:nixpkgs";

  outputs = { self, nixpkgs }:
    let
      forAllSystems = f:
        nixpkgs.lib.genAttrs [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ]
          (system: f { pkgs = import nixpkgs { inherit system; }; });
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
          ];
        in {
          default = pkgs.mkShell {
            packages = with pkgs; [
              # Rust toolchain. Stable until we need nightly (wasm
              # SIMD features in R10 may push us there).
              rustc
              cargo
              rustfmt
              clippy
              rust-analyzer
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
              # `roxlap-web` crate. `trunk serve` from
              # `crates/roxlap-web/` starts a hot-reloading
              # localhost:8080 demo; `trunk build --release` emits
              # the production static bundle.
              trunk
              # Image inspection — imagemagick reads roxlap-oracle's
              # PPM dumps and voxlap C oracle's PNG outputs for byte-
              # level pixel diffing across the two engines.
              imagemagick
            ] ++ pkgs.lib.optionals pkgs.stdenv.isLinux linuxRuntimeLibs;

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
