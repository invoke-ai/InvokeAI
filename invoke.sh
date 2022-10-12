#!/bin/bash

if [ "$0" == "bash" ]; then
    INSTALL_ENV_DIR="$(pwd)/installer_files/env"
    if [ -e "$INSTALL_ENV_DIR" ]; then export PATH="$PATH;$INSTALL_ENV_DIR/bin"; fi

    conda activate invokeai

    echo "Ready to dream.."
else
    bash --init-file invoke.sh
fi
