#!/bin/bash

####
# This launch script assumes that:
# 1. it is located in the runtime directory,
# 2. the .venv is also located in the runtime directory and is named exactly that
#
# If both of the above are not true, this script will likely not work as intended.
# Activate the virtual environment and run `invoke.py` directly.
####

set -eu

# ensure we're in the correct folder in case user's CWD is somewhere else
scriptdir=$(dirname "$0")
cd "$scriptdir"

. .venv/bin/activate

export INVOKEAI_ROOT="$scriptdir"

# set required env var for torch on mac MPS
if [ "$(uname -s)" == "Darwin" ]; then
    export PYTORCH_ENABLE_MPS_FALLBACK=1
fi

while true
do
if [ "$0" != "bash" ]; then
    echo "Do you want to generate images using the"
    echo "1. command-line interface"
    echo "2. browser-based UI"
    echo "3. run textual inversion training"
    echo "4. merge models (diffusers type only)"
    echo "5. download and install models"
    echo "6. change InvokeAI startup options"
    echo "7. re-run the configure script to fix a broken install"
    echo "8. open the developer console"
    echo "9. update InvokeAI"
    echo "10. command-line help"
    echo "Q - Quit"
    echo ""
    read -p "Please enter 1-10, Q: [2] " yn
    choice=${yn:='2'}
    case $choice in
        1)
            echo "Starting the InvokeAI command-line..."
            invokeai $@
            ;;
        2)
            echo "Starting the InvokeAI browser-based UI..."
            invokeai --web $@
            ;;
        3)
            echo "Starting Textual Inversion:"
            invokeai-ti --gui $@
            ;;
        4)
            echo "Merging Models:"
            invokeai-merge --gui $@
            ;;
        5)
            invokeai-model-install --root ${INVOKEAI_ROOT}
            ;;
        6)
            invokeai-configure --root ${INVOKEAI_ROOT} --skip-sd-weights --skip-support-models
            ;;
        7)
            invokeai-configure --root ${INVOKEAI_ROOT} --yes --default_only
	    ;;
	8)
	    echo "Developer Console:"
            file_name=$(basename "${BASH_SOURCE[0]}")
            bash --init-file "$file_name"
            ;;
        9)
	    echo "Update:"
            invokeai-update
            ;;
        10)
            invokeai --help
            ;;
	[qQ])
            exit 0
            ;;
        *)
            echo "Invalid selection"
            exit;;
    esac
else # in developer console
    python --version
    echo "Press ^D to exit"
    export PS1="(InvokeAI) \u@\h \w> "
fi
done
