# Important note: this flake does not attempt to create a fully isolated, 'pure'
# Python environment for InvokeAI. Instead, it depends on local invocations of
# virtualenv/pip to install the required (binary) packages, most importantly the
# prebuilt binary pytorch packages with CUDA support.
# ML Python packages with CUDA support, like pytorch, are notoriously expensive
# to compile so it's purposefuly not what this flake does.

{
  description = "An (impure) flake to develop on InvokeAI.";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };

      python = pkgs.python310;

      mkShell = { dir, install }:
        let
          setupScript = pkgs.writeScript "setup-invokai" ''
            # This must be sourced using 'source', not executed.
            ${python}/bin/python -m venv ${dir}
            ${dir}/bin/python -m pip install ${install}
            # ${dir}/bin/python -c 'import torch; assert(torch.cuda.is_available())'
            source ${dir}/bin/activate
          '';
        in
        pkgs.mkShell rec {
          buildInputs = with pkgs; [
            # Backend: graphics, CUDA.
            cudaPackages.cudnn
            cudaPackages.cuda_nvrtc
            cudatoolkit
            pkgconfig
            libconfig
            cmake
            blas
            freeglut
            glib
            gperf
            procps
            libGL
            libGLU
            linuxPackages.nvidia_x11
            python
            (opencv4.override {
              enableGtk3 = true;
              enableFfmpeg = true;
              enableCuda = true;
              enableUnfree = true;
            })
            stdenv.cc
            stdenv.cc.cc.lib
            xorg.libX11
            xorg.libXext
            xorg.libXi
            xorg.libXmu
            xorg.libXrandr
            xorg.libXv
            zlib

            # Pre-commit hooks.
            black

            # Frontend.
            yarn
            nodejs
          ];
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath buildInputs;
          CUDA_PATH = pkgs.cudatoolkit;
          EXTRA_LDFLAGS = "-L${pkgs.linuxPackages.nvidia_x11}/lib";
          shellHook = ''
            if [[ -f "${dir}/bin/activate" ]]; then
              source "${dir}/bin/activate"
              echo "Using Python: $(which python)"
            else
              echo "Use 'source ${setupScript}' to set up the environment."
            fi
          '';
        };
    in
    {
      devShells.${system} = rec {
        develop = mkShell { dir = "venv"; install = "-e '.[xformers]' --extra-index-url https://download.pytorch.org/whl/cu118"; };
        default = develop;
      };
    };
}
