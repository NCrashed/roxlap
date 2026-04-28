{
  description = "roxlap — pure-Rust port of Ken Silverman's Voxlap voxel engine";

  inputs.nixpkgs.url = "flake:nixpkgs";

  outputs = { self, nixpkgs }:
    let
      forAllSystems = f:
        nixpkgs.lib.genAttrs [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ]
          (system: f { pkgs = import nixpkgs { inherit system; }; });
    in {
      devShells = forAllSystems ({ pkgs }: {
        default = pkgs.mkShell {
          packages = with pkgs; [
            # Rust toolchain. Stable until we need nightly (wasm SIMD
            # features may push us there in stage R10).
            rustc
            cargo
            rustfmt
            clippy
            rust-analyzer

            # R3+ — SDL2 host. Listed now so `nix develop` is sufficient
            # for every substage; the dependency itself is added to
            # crates/roxlap-sdl/Cargo.toml in R3.
            SDL2
            pkg-config
          ];
        };
      });
    };
}
