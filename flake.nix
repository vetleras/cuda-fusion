{
  outputs = {nixpkgs, ...}: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
      config.allowUnfree = true;
    };
  in {
    devShells."${system}".default =
      pkgs.mkShell.override {
        stdenv = pkgs.clangStdenv;
      } {
        # https://discourse.nixos.org/t/non-interactive-bash-errors-from-flake-nix-mkshell/33310
        # This fixes a bug where the terminal in VS Code is weird
        buildInputs = [pkgs.bashInteractive];

        packages = [
          pkgs.linuxPackages.nvidia_x11
          pkgs.cudatoolkit
          pkgs.llvmPackages_latest.llvm
        ];

        shellHook = ''
          export CUDA_PATH=${pkgs.cudatoolkit}
          export LIBCLANG_PATH=${pkgs.llvmPackages.libclang.lib}/lib

          rustup default stable
          cargo install ptx-linker
          rustup toolchain install nightly-2022-10-13
          rustup target add --toolchain nightly-2022-10-13-x86_64-unknown-linux-gnu nvptx64-nvidia-cuda
          rustup component add rust-src --toolchain nightly-2022-10-13-x86_64-unknown-linux-gnu
          rustup default nightly-2024-05-01
          code .
          exit
        '';
      };
  };
}
