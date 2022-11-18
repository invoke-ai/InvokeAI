#!/usr/bin/env bash

set -euo pipefail
IFS=$'\n\t'

source .venv/bin/activate

if [ "$0" != "bash" ]; then
    echo "Do you want to generate images using the"
    echo "1. command-line"
    echo "2. browser-based UI"
    echo "3. open the developer console"
    read -p "Please enter 1, 2, or 3: " yn
    case $yn in
        1 ) printf "\nStarting the InvokeAI command-line..\n"; .venv/bin/python scripts/invoke.py;;
        2 ) printf "\nStarting the InvokeAI browser-based UI..\n"; .venv/bin/python scripts/invoke.py --web;;
        3 ) printf "\nDeveloper Console:\n"; file_name=$(basename "${BASH_SOURCE[0]}"); bash --init-file "$file_name";;
        * ) echo "Invalid selection"; exit;;
    esac
else # in developer console
    echo "'python' command is:"
    which python
    echo "'python' version is:"
    python --version
    echo "Type 'exit' to quit this shell"
fi
