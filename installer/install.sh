#!/bin/bash

# This script will install git and conda (if not found on the PATH variable)
#  using micromamba (an 8mb static-linked single-file binary, conda replacement).
# For users who already have git and conda, this step will be skipped.

# Next, it'll checkout the project's git repo, if necessary.
# Finally, it'll create the conda environment and preload the models.

# This enables a user to install this project without manually installing conda and git.


OS_NAME=$(uname -s)
case "${OS_NAME}" in
    Linux*)     OS_NAME="linux";;
    Darwin*)    OS_NAME="mac";;
    *)          echo "Unknown OS: $OS_NAME! This script runs only on Linux or Mac" && exit
esac

OS_ARCH=$(uname -m)
case "${OS_ARCH}" in
    x86_64*)    OS_ARCH="x64";;
    arm64*)     OS_ARCH="arm64";;
    *)          echo "Unknown system architecture: $OS_ARCH! This script runs only on x86_64 or arm64" && exit
esac

# config
export MAMBA_ROOT_PREFIX="$(pwd)/installer_files/mamba"
INSTALL_ENV_DIR="$(pwd)/installer_files/env"
MICROMAMBA_BINARY_FILE="$(pwd)/installer_files/micromamba_${OS_NAME}_${OS_ARCH}"

# figure out whether git and conda needs to be installed
PACKAGES_TO_INSTALL=""

if ! hash "conda" &>/dev/null; then PACKAGES_TO_INSTALL="$PACKAGES_TO_INSTALL conda"; fi
if ! hash "git" &>/dev/null; then PACKAGES_TO_INSTALL="$PACKAGES_TO_INSTALL git"; fi

# (if necessary) install git and conda into a contained environment
if [ "$PACKAGES_TO_INSTALL" != "" ]; then
    # initialize micromamba
    if [ ! -e "$MAMBA_ROOT_PREFIX" ]; then
        mkdir -p "$MAMBA_ROOT_PREFIX"
        cp "$MICROMAMBA_BINARY_FILE" "$MAMBA_ROOT_PREFIX/micromamba"

        # test the mamba binary
        echo "Micromamba version:"
        "$MAMBA_ROOT_PREFIX/micromamba" --version
    fi

    # create the installer env
    if [ ! -e "$INSTALL_ENV_DIR" ]; then
        "$MAMBA_ROOT_PREFIX/micromamba" create -y --prefix "$INSTALL_ENV_DIR"
    fi

    echo "Packages to install:$PACKAGES_TO_INSTALL"

    "$MAMBA_ROOT_PREFIX/micromamba" install -y --prefix "$INSTALL_ENV_DIR" -c conda-forge $PACKAGES_TO_INSTALL

    if [ ! -e "$INSTALL_ENV_DIR" ]; then
        echo "There was a problem while initializing micromamba. Cannot continue."
        exit
    fi
fi

if [ -e "$INSTALL_ENV_DIR" ]; then export PATH="$INSTALL_ENV_DIR/bin:$PATH"; fi

# get the repo (and load into the current directory)
if [ ! -e ".git" ]; then
    git config --global init.defaultBranch main
    git init
    git remote add origin https://github.com/cmdr2/InvokeAI.git
    git fetch
    git checkout origin/main -ft
fi

# create the environment
CONDA_BASEPATH=$(conda info --base)
source "$CONDA_BASEPATH/etc/profile.d/conda.sh" # otherwise conda complains about 'shell not initialized' (needed when running in a script)

if [ "$OS_NAME" == "mac" ]; then
    PIP_EXISTS_ACTION=w CONDA_SUBDIR=osx-${OS_ARCH} conda env create -f environment-mac.yml
else
    conda env create -f environment.yml
fi

conda activate invokeai

# preload the models
python scripts/preload_models.py

# make the models dir
mkdir models/ldm/stable-diffusion-v1

# tell the user that they need to download the ckpt
WEIGHTS_DOC_URL="https://invoke-ai.github.io/InvokeAI/installation/INSTALL_LINUX/"
if [ "$OS_NAME" == "mac" ]; then
    WEIGHTS_DOC_URL="https://invoke-ai.github.io/InvokeAI/installation/INSTALL_MAC/"
fi

echo ""
echo "Now you need to install the weights for the stable diffusion model."
echo "Please follow the steps at $WEIGHTS_DOC_URL to complete the installation"

# it would be nice if the weights downloaded automatically, and didn't need the user to do this manually.
