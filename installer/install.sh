#!/usr/bin/env bash

set -euo pipefail
IFS=$'\n\t'

function _err_exit {
    if test "$1" -ne 0
    then
        echo -e "Error code $1; Error caught was '$2'"
        read -p "Press any key to exit..."
        exit
    fi
}

# This script will install git (if not found on the PATH variable)
#  using micromamba (an 8mb static-linked single-file binary, conda replacement).
# For users who already have git, this step will be skipped.

# Next, it'll download the project's source code.
# Then it will download a self-contained, standalone Python and unpack it.
# Finally, it'll create the Python virtual environment and preload the models.

# This enables a user to install this project without manually installing git or Python

export no_cache_dir="--no-cache-dir"
if [ $# -ge 1 ]; then
    if [ "$1" = "use-cache" ]; then
        export no_cache_dir=""
    fi
fi

echo "$no_cache_dir"

echo -e "\n***** Installing InvokeAI... *****\n"


OS_NAME=$(uname -s)
case "${OS_NAME}" in
    Linux*)     OS_NAME="linux";;
    Darwin*)    OS_NAME="darwin";;
    *)          echo -e "\n----- Unknown OS: $OS_NAME! This script runs only on Linux or macOS -----\n" && exit
esac

OS_ARCH=$(uname -m)
case "${OS_ARCH}" in
    x86_64*)    ;;
    arm64*)     ;;
    *)          echo -e "\n----- Unknown system architecture: $OS_ARCH! This script runs only on x86_64 or arm64 -----\n" && exit
esac

# https://mamba.readthedocs.io/en/latest/installation.html
MAMBA_OS_NAME=$OS_NAME
MAMBA_ARCH=$OS_ARCH
if [ "$OS_NAME" == "darwin" ]; then
    MAMBA_OS_NAME="osx"
fi

if [ "$OS_ARCH" == "linux" ]; then
    MAMBA_ARCH="aarch64"
fi

if [ "$OS_ARCH" == "x86_64" ]; then
    MAMBA_ARCH="64"
fi

PY_ARCH=$OS_ARCH
if [ "$OS_ARCH" == "arm64" ]; then
    PY_ARCH="aarch64"
fi

# Compute device ('cd' segment of reqs files) detect goes here
# This needs a ton of work
# Suggestions:
#   - lspci
#   - check $PATH for nvidia-smi, gtt CUDA/GPU version from output
#   - Surely there's a similar utility for AMD?
CD="cuda"
if [ "$OS_NAME" == "darwin" ] && [ "$OS_ARCH" == "arm64" ]; then
    CD="mps"
fi

# config
INSTALL_ENV_DIR="$(pwd)/installer_files/env"
MICROMAMBA_DOWNLOAD_URL="https://micro.mamba.pm/api/micromamba/${MAMBA_OS_NAME}-${MAMBA_ARCH}/latest"
RELEASE_URL=https://github.com/invoke-ai/InvokeAI
RELEASE_SOURCEBALL=/archive/refs/heads/main.tar.gz
PYTHON_BUILD_STANDALONE_URL=https://github.com/indygreg/python-build-standalone/releases/download
if [ "$OS_NAME" == "darwin" ]; then
    PYTHON_BUILD_STANDALONE=20221002/cpython-3.10.7+20221002-${PY_ARCH}-apple-darwin-install_only.tar.gz
elif [ "$OS_NAME" == "linux" ]; then
    PYTHON_BUILD_STANDALONE=20221002/cpython-3.10.7+20221002-${PY_ARCH}-unknown-linux-gnu-install_only.tar.gz
fi

PACKAGES_TO_INSTALL=""

if ! hash "git" &>/dev/null; then PACKAGES_TO_INSTALL="$PACKAGES_TO_INSTALL git"; fi

# (if necessary) install git and conda into a contained environment
if [ "$PACKAGES_TO_INSTALL" != "" ]; then
    # download micromamba
    echo -e "\n***** Downloading micromamba from $MICROMAMBA_DOWNLOAD_URL to micromamba *****\n"

    curl -L "$MICROMAMBA_DOWNLOAD_URL" | tar -xvjO bin/micromamba > micromamba

    chmod u+x ./micromamba

    # test the mamba binary
    echo -e "\n***** Micromamba version: *****\n"
    ./micromamba --version

    # create the installer env
    if [ ! -e "$INSTALL_ENV_DIR" ]; then
        ./micromamba create -y --prefix "$INSTALL_ENV_DIR"
    fi

    echo -e "\n***** Packages to install:$PACKAGES_TO_INSTALL *****\n"

    ./micromamba install -y --prefix "$INSTALL_ENV_DIR" -c conda-forge "$PACKAGES_TO_INSTALL"

    if [ ! -e "$INSTALL_ENV_DIR" ]; then
        echo -e "\n----- There was a problem while initializing micromamba. Cannot continue. -----\n"
        exit
    fi
fi

rm -f micromamba.exe

export PATH="$INSTALL_ENV_DIR/bin:$PATH"

# Download/unpack/clean up InvokeAI release sourceball
_err_msg="\n----- InvokeAI source download failed -----\n"
curl -L $RELEASE_URL/$RELEASE_SOURCEBALL --output InvokeAI.tgz
_err_exit $? _err_msg
_err_msg="\n----- InvokeAI source unpack failed -----\n"
tar -zxf InvokeAI.tgz
_err_exit $? _err_msg

rm -f InvokeAI.tgz

_err_msg="\n----- InvokeAI source copy failed -----\n"
cd InvokeAI-*
cp -r . ..
_err_exit $? _err_msg
cd ..

# cleanup
rm -rf InvokeAI-*/
rm -rf .dev_scripts/ .github/ docker-build/ tests/ requirements.in requirements-mkdocs.txt shell.nix

echo -e "\n***** Unpacked InvokeAI source *****\n"

# Download/unpack/clean up python-build-standalone
_err_msg="\n----- Python download failed -----\n"
curl -L $PYTHON_BUILD_STANDALONE_URL/$PYTHON_BUILD_STANDALONE --output python.tgz
_err_exit $? _err_msg
_err_msg="\n----- Python unpack failed -----\n"
tar -zxf python.tgz
_err_exit $? _err_msg

rm -f python.tgz

echo -e "\n***** Unpacked python-build-standalone *****\n"

# create venv
_err_msg="\n----- problem creating venv -----\n"

if [ "$OS_NAME" == "darwin" ]; then
    # patch sysconfig so that extensions can build properly
    # adapted from https://github.com/cashapp/hermit-packages/commit/fcba384663892f4d9cfb35e8639ff7a28166ee43
    PYTHON_INSTALL_DIR="$(pwd)/python"
    SYSCONFIG="$(echo python/lib/python*/_sysconfigdata_*.py)"
    TMPFILE="$(mktemp)"
    chmod +w "${SYSCONFIG}"
    cp "${SYSCONFIG}" "${TMPFILE}"
    sed "s,'/install,'${PYTHON_INSTALL_DIR},g" "${TMPFILE}" > "${SYSCONFIG}"
    rm -f "${TMPFILE}"
fi

./python/bin/python3 -E -s -m venv .venv
_err_exit $? _err_msg
source .venv/bin/activate

echo -e "\n***** Created Python virtual environment *****\n"

# Print venv's Python version
_err_msg="\n----- problem calling venv's python -----\n"
echo -e "We're running under"
.venv/bin/python3 --version
_err_exit $? _err_msg

_err_msg="\n----- pip update failed -----\n"
.venv/bin/python3 -m pip install "$no_cache_dir" --no-warn-script-location --upgrade pip wheel
_err_exit $? _err_msg

echo -e "\n***** Updated pip and wheel *****\n"

_err_msg="\n----- requirements file copy failed -----\n"
cp installer/py3.10-${OS_NAME}-"${OS_ARCH}"-${CD}-reqs.txt requirements.txt
_err_exit $? _err_msg

_err_msg="\n----- main pip install failed -----\n"
.venv/bin/python3 -m pip install "$no_cache_dir" --no-warn-script-location -r requirements.txt
_err_exit $? _err_msg

echo -e "\n***** Installed Python dependencies *****\n"

_err_msg="\n----- InvokeAI setup failed -----\n"
.venv/bin/python3 -m pip install "$no_cache_dir" --no-warn-script-location -e .
_err_exit $? _err_msg

echo -e "\n***** Installed InvokeAI *****\n"

cp installer/invoke.sh .
echo -e "\n***** Installed invoke launcher script ******\n"

# more cleanup
rm -rf installer/ installer_files/

# preload the models
.venv/bin/python3 scripts/configure_invokeai.py
_err_msg="\n----- model download clone failed -----\n"
_err_exit $? _err_msg
deactivate

echo -e "\n***** Finished downloading models *****\n"

echo "All done! Run the command './invoke.sh' to start InvokeAI."
read -p "Press any key to exit..."
exit
