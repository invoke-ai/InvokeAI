#!/usr/bin/env sh

set -eu

. .venv/bin/activate

echo "Do you want to generate images using the"
echo "1. command-line"
echo "2. browser-based UI"
echo "OR"
echo "3. open the developer console"
echo "Please enter 1, 2, or 3:"
read choice

case $choice in
    1)
        printf "\nStarting the InvokeAI command-line..\n";
        .venv/bin/python scripts/invoke.py;
    ;;
    2)
        printf "\nStarting the InvokeAI browser-based UI..\n";
        .venv/bin/python scripts/invoke.py --web;
    ;;
    3)
        printf "\nDeveloper Console:\n";
        printf "Python command is:\n\t";
        which python;
        printf "Python version is:\n\t";
        python --version;
        echo "*************************"
        echo "You are now in your user shell ($SHELL) with the local InvokeAI Python virtual environment activated,";
        echo "so that you can troubleshoot this InvokeAI installation as necessary.";
        printf "*************************\n"
        echo "*** Type \`exit\` to quit this shell and deactivate the Python virtual environment *** ";
        /usr/bin/env "$SHELL";
    ;;
    *)
        echo "Invalid selection";
        exit
    ;;
esac
