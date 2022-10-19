{ pkgs ? import <nixpkgs> {}
  , lib ? pkgs.lib
  , stdenv ? pkgs.stdenv
  , fetchurl ? pkgs.fetchurl
  , runCommand ? pkgs.runCommand
  , makeWrapper ? pkgs.makeWrapper
  , mkShell ? pkgs.mkShell
  , buildFHSUserEnv ? pkgs.buildFHSUserEnv
  , frameworks ? pkgs.darwin.apple_sdk.frameworks
}:

# Setup InvokeAI environment using nix
# Simple usage:
# nix-shell
# python3 scripts/preload_models.py
# python3 scripts/invoke.py -h

let
  conda-shell = { url, sha256, installPath, packages, shellHook }:
  let
    src = fetchurl { inherit url sha256; };
    libPath = lib.makeLibraryPath ([] ++ lib.optionals (stdenv.isLinux) [ pkgs.zlib ]);
    condaArch = if stdenv.system == "aarch64-darwin" then "osx-arm64" else "";
    installer =
    if stdenv.isDarwin then
      runCommand "conda-install" {
        nativeBuildInputs = [ makeWrapper ];
      } ''
        mkdir -p $out/bin
        cp ${src} $out/bin/miniconda-installer.sh
        chmod +x $out/bin/miniconda-installer.sh
        makeWrapper                       \
          $out/bin/miniconda-installer.sh \
          $out/bin/conda-install          \
          --add-flags "-p ${installPath}" \
          --add-flags "-b"
      ''
    else if stdenv.isLinux then
      runCommand "conda-install" {
        nativeBuildInputs = [ makeWrapper ];
        buildInputs = [ pkgs.zlib ];
      }
      # on line 10, we have 'unset LD_LIBRARY_PATH'
      # we have to comment it out however in a way that the number of bytes in the
      # file does not change. So we replace the 'u' in the line with a '#'
      # The reason is that the binary payload is encoded as number
      # of bytes from the top of the installer script
      # and unsetting the library path prevents the zlib library from being discovered
      ''
        mkdir -p $out/bin
        sed 's/unset LD_LIBRARY_PATH/#nset LD_LIBRARY_PATH/' ${src} > $out/bin/miniconda-installer.sh
        chmod +x $out/bin/miniconda-installer.sh
        makeWrapper                       \
          $out/bin/miniconda-installer.sh \
          $out/bin/conda-install          \
          --add-flags "-p ${installPath}" \
          --add-flags "-b"                \
          --prefix "LD_LIBRARY_PATH" : "${libPath}"
      ''
    else {};

    hook = ''
      export CONDA_SUBDIR=${condaArch}
    '' +  shellHook;

    fhs = buildFHSUserEnv {
      name = "conda-shell";
      targetPkgs = pkgs: [ stdenv.cc pkgs.git installer ] ++ packages;
      profile = hook;
      runScript = "bash";
    };

    shell = mkShell {
      shellHook = if stdenv.isDarwin then hook else "conda-shell; exit";
      packages = if stdenv.isDarwin then [ pkgs.git installer ] ++ packages else [ fhs ];
    };
  in shell;

  packages = with pkgs; [
    cmake
    protobuf
    libiconv
    rustc
    cargo
    rustPlatform.bindgenHook
  ];

  env = {
    aarch64-darwin = {
      envFile = "environment-mac.yml";
      condaPath = (builtins.toString ./.) + "/.conda";
      ptrSize = "8";
    };
    x86_64-linux =  {
      envFile = "environment.yml";
      condaPath = (builtins.toString ./.) + "/.conda";
      ptrSize = "8";
    };
  };

  envFile = env.${stdenv.system}.envFile;
  installPath = env.${stdenv.system}.condaPath;
  ptrSize = env.${stdenv.system}.ptrSize;
  shellHook = ''
    conda-install

    # tmpdir is too small in nix
    export TMPDIR="${installPath}/tmp"

    # Add conda to PATH
    export PATH="${installPath}/bin:$PATH"

    # Allows `conda activate` to work properly
    source ${installPath}/etc/profile.d/conda.sh

    # Paths for gcc if compiling some C sources with pip
    export NIX_CFLAGS_COMPILE="-I${installPath}/include -I$TMPDIR/include"
    export NIX_CFLAGS_LINK="-L${installPath}/lib $BINDGEN_EXTRA_CLANG_ARGS"

    export PIP_EXISTS_ACTION=w

    # rust-onig fails (think it writes config.h to wrong location)
    mkdir -p "$TMPDIR/include"
    cat <<'EOF' > "$TMPDIR/include/config.h"
    #define HAVE_PROTOTYPES 1
    #define STDC_HEADERS 1
    #define HAVE_STRING_H 1
    #define HAVE_STDARG_H 1
    #define HAVE_STDLIB_H 1
    #define HAVE_LIMITS_H 1
    #define HAVE_INTTYPES_H 1
    #define SIZEOF_INT 4
    #define SIZEOF_SHORT 2
    #define SIZEOF_LONG ${ptrSize}
    #define SIZEOF_VOIDP ${ptrSize}
    #define SIZEOF_LONG_LONG 8
    EOF

    conda env create -f "${envFile}" || conda env update --prune -f "${envFile}"
    conda activate invokeai
  '';

  version = "4.12.0";
  conda = {
    aarch64-darwin = {
      shell = conda-shell {
        inherit shellHook installPath;
        url = "https://repo.anaconda.com/miniconda/Miniconda3-py39_${version}-MacOSX-arm64.sh";
        sha256 = "4bd112168cc33f8a4a60d3ef7e72b52a85972d588cd065be803eb21d73b625ef";
        packages = [ frameworks.Security ] ++ packages;
      };
    };
    x86_64-linux =  {
      shell = conda-shell {
        inherit shellHook installPath;
        url = "https://repo.continuum.io/miniconda/Miniconda3-py39_${version}-Linux-x86_64.sh";
        sha256 = "78f39f9bae971ec1ae7969f0516017f2413f17796670f7040725dd83fcff5689";
        packages = with pkgs; [ libGL glib ] ++ packages;
      };
    };
  };
in conda.${stdenv.system}.shell
