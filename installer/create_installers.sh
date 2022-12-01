#!/usr/bin/env bash

set -euo pipefail
IFS=$'\n\t'

current_branch=$(git branch --show-current)

echo "Be certain that you're in the 'installer' directory before continuing."
read -p "Press any key to continue, or CTRL-C to exit..."

echo "Also be sure that '$current_branch' is the correct branch for this release, and"
echo "that this matches RELEASE_SOURCEBALL in install.sh and install.bat, and that the"
echo "py3.10-*-reqs.txt files are all correct and up to date for branch '$current_branch'."
read -p "Press any key to continue, or CTRL-C to exit..."

# make the installer zip for linux and mac
rm -rf InvokeAI
mkdir -p InvokeAI
cp install.sh InvokeAI
cp readme.txt InvokeAI

zip -r InvokeAI-linux_on_$current_branch.zip InvokeAI
zip -r InvokeAI-mac_on_$current_branch.zip InvokeAI

# make the installer zip for windows
rm -rf InvokeAI
mkdir -p InvokeAI
cp install.bat InvokeAI
cp readme.txt InvokeAI
cp WinLongPathsEnabled.reg InvokeAI

zip -r InvokeAI-windows_on_$current_branch.zip InvokeAI

rm -rf InvokeAI

echo "The installer zips are ready for distribution."
