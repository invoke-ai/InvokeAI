#!/bin/bash

# This script will install git and conda (if not found on the PATH variable)
#  using micromamba (an 8mb static-linked single-file binary, conda replacement).
# For users who already have git and conda, this step will be skipped.

# Next, it'll checkout the project's git repo, if necessary.
# Finally, it'll create the conda environment and preload the models.

# This enables a user to install this project without manually installing conda and git.

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "Installing InvokeAI.."
echo ""

OS_NAME=$(uname -s)
case "${OS_NAME}" in
    Linux*)     OS_NAME="linux";;
    Darwin*)    OS_NAME="osx";;
    *)          echo "Unknown OS: $OS_NAME! This script runs only on Linux or Mac" && exit
esac

OS_ARCH=$(uname -m)
case "${OS_ARCH}" in
    x86_64*)    OS_ARCH="64";;
    arm64*)     OS_ARCH="arm64";;
    *)          echo "Unknown system architecture: $OS_ARCH! This script runs only on x86_64 or arm64" && exit
esac

# https://mamba.readthedocs.io/en/latest/installation.html
if [ "$OS_NAME" == "linux" ] && [ "$OS_ARCH" == "arm64" ]; then OS_ARCH="aarch64"; fi

# config
export MAMBA_ROOT_PREFIX="$(pwd)/installer_files/mamba"
INSTALL_ENV_DIR="$(pwd)/installer_files/env"
MICROMAMBA_DOWNLOAD_URL="https://micro.mamba.pm/api/micromamba/${OS_NAME}-${OS_ARCH}/latest"
REPO_URL="https://github.com/invoke-ai/InvokeAI.git"
umamba_exists="F"

# figure out whether git and conda needs to be installed
if [ -e "$INSTALL_ENV_DIR" ]; then export PATH="$INSTALL_ENV_DIR/bin:$PATH"; fi

PACKAGES_TO_INSTALL=""
if ! $(which conda) -V &>/dev/null; then PACKAGES_TO_INSTALL="$PACKAGES_TO_INSTALL conda"; fi
if ! which git &>/dev/null; then PACKAGES_TO_INSTALL="$PACKAGES_TO_INSTALL git"; fi

if "$MAMBA_ROOT_PREFIX/micromamba" --version &>/dev/null; then umamba_exists="T"; fi

# (if necessary) install git and conda into a contained environment
if [ "$PACKAGES_TO_INSTALL" != "" ]; then
    # download micromamba
    if [ "$umamba_exists" == "F" ]; then
        echo "Downloading micromamba from $MICROMAMBA_DOWNLOAD_URL to $MAMBA_ROOT_PREFIX/micromamba"

        mkdir -p "$MAMBA_ROOT_PREFIX"
        curl -L "$MICROMAMBA_DOWNLOAD_URL" | tar -xvjO bin/micromamba > "$MAMBA_ROOT_PREFIX/micromamba"

        chmod u+x "$MAMBA_ROOT_PREFIX/micromamba"

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
    git init
    git config --local init.defaultBranch main
    git remote add origin "$REPO_URL"
    git fetch
    git checkout origin/main -ft
fi

# create the environment
CONDA_BASEPATH=$(conda info --base)
source "$CONDA_BASEPATH/etc/profile.d/conda.sh" # otherwise conda complains about 'shell not initialized' (needed when running in a script)

conda activate
if [ "$OS_NAME" == "osx" ]; then
    echo "macOS detected. Installing MPS and CPU support."
    ln -sf environments-and-requirements/environment-mac.yml environment.yml
else
    if (lsmod | grep amdgpu) &>/dev/null ; then
	echo "Linux system with AMD GPU driver detected. Installing ROCm and CPU support"
	ln -sf environments-and-requirements/environment-lin-amd.yml environment.yml
    else
	echo "Linux system detected. Installing CUDA and CPU support."
	ln -sf environments-and-requirements/environment-lin-cuda.yml environment.yml
    fi
fi
conda env update

status=$?

if test $status -ne 0
then
   echo "Something went wrong while installing Python libraries and cannot continue."
   echo "Please visit https://invoke-ai.github.io/InvokeAI/#installation for alternative"
   echo "installation methods"
else
    ln -sf ./source_installer/invoke.sh .
    ln -sf ./source_installer/update.sh .

    conda activate invokeai
    # preload the models
    echo "Calling the preload_models.py script"
    python scripts/preload_models.py
    status=$?
    if test $status -ne 0
       then
	   echo "The preload_models.py script crashed or was cancelled."
           echo "InvokeAI is not ready to run. Try again by running"
           echo "update.sh in this directory."
       else
           # tell the user their next steps
	   echo "You can now start generating images by running invoke.sh (inside this folder), using ./invoke.sh"
       fi
fi

conda activate invokeai
