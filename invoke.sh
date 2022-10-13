#!/bin/bash

if [ "$0" == "bash" ]; then
    INSTALL_ENV_DIR="$(pwd)/installer_files/env"
    if [ -e "$INSTALL_ENV_DIR" ]; then export PATH="$PATH:$INSTALL_ENV_DIR/bin"; fi

    CONDA_BASEPATH=$(conda info --base)
    source "$CONDA_BASEPATH/etc/profile.d/conda.sh" # otherwise conda complains about 'shell not initialized' (needed when running in a script)

    conda activate invokeai

    echo "Ready to dream.."
else
    bash --init-file invoke.sh
fi
