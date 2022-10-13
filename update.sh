#!/bin/bash

INSTALL_ENV_DIR="$(pwd)/installer_files/env"
if [ -e "$INSTALL_ENV_DIR" ]; then export PATH="$PATH:$INSTALL_ENV_DIR/bin"; fi

# update the repo
if [ -e ".git" ]; then
    git pull
fi

OS_NAME=$(uname -s)
case "${OS_NAME}" in
    Linux*)     conda env update;;
    Darwin*)    conda env update -f environment-mac.yml;;
    *)          echo "Unknown OS: $OS_NAME! This script runs only on Linux or Mac" && exit
esac
