#!/bin/bash

if [ "$0" == "bash" ]; then
    # check if conda exists, otherwise use micromamba
    CONDA_COMMAND="conda"

    if ! hash "conda" &>/dev/null; then CONDA_COMMAND="micromamba"; fi

    # initialize micromamba, if using that
    if [ "$CONDA_COMMAND" == "micromamba" ]; then
        export MAMBA_ROOT_PREFIX="$(pwd)/installer_files/mamba"
        INSTALL_ENV_DIR="$(pwd)/installer_files/env"

        if [ ! -e "$MAMBA_ROOT_PREFIX" ]; then
            echo "Have you run install.sh?"
            exit
        fi

        eval "$($MAMBA_ROOT_PREFIX/micromamba shell hook -s posix)"

        micromamba activate "$INSTALL_ENV_DIR"
    )

    $CONDA_COMMAND activate invokeai
else
    bash --init-file invoke.sh
fi
