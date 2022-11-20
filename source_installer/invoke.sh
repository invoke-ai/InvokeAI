#!/bin/bash

cd "$(dirname "${BASH_SOURCE[0]}")"

INSTALL_ENV_DIR="$(pwd)/installer_files/env"
if [ -e "$INSTALL_ENV_DIR" ]; then export PATH="$INSTALL_ENV_DIR/bin:$PATH"; fi

CONDA_BASEPATH=$(conda info --base)
source "$CONDA_BASEPATH/etc/profile.d/conda.sh" # otherwise conda complains about 'shell not initialized' (needed when running in a script)

conda activate invokeai

if [ "$0" != "bash" ]; then
    echo "Do you want to generate images using the"
    echo "1. command-line"
    echo "2. browser-based UI"
    echo "3. open the developer console"
    read -p "Please enter 1, 2, or 3: " yn
    case $yn in
        1 ) printf "\nStarting the InvokeAI command-line..\n"; python scripts/invoke.py;;
        2 ) printf "\nStarting the InvokeAI browser-based UI..\n"; python scripts/invoke.py --web;;
        3 ) printf "\nDeveloper Console:\n"; file_name=$(basename "${BASH_SOURCE[0]}"); bash --init-file "$file_name";;
        * ) echo "Invalid selection"; exit;;
    esac
else # in developer console
    python --version
    echo "Press ^D to exit"
    export PS1="(InvokeAI) \u@\h \w> "
fi
