#!/bin/bash


INSTALL_ENV_DIR="$(pwd)/installer_files/env"
if [ -e "$INSTALL_ENV_DIR" ]; then export PATH="$INSTALL_ENV_DIR/bin:$PATH"; fi

# update the repo
if [ -e ".git" ]; then
    git pull
fi

CONDA_BASEPATH=$(conda info --base)
source "$CONDA_BASEPATH/etc/profile.d/conda.sh" # otherwise conda complains about 'shell not initialized' (needed when running in a script)

conda activate invokeai

OS_NAME=$(uname -s)
case "${OS_NAME}" in
    Linux*)     conda env update;;
    Darwin*)    conda env update -f environment-mac.yml;;
    *)          echo "Unknown OS: $OS_NAME! This script runs only on Linux or Mac" && exit
esac

python scripts/preload_models.py
