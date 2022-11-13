#!/usr/bin/env bash

set -euo pipefail
IFS=$'\n\t'

echo "Be certain that you're in the 'installer' directory before continuing."
read -p "Press any key to continue, or CTRL-C to exit..."

# make the installer zip for linux and mac
rm -rf InvokeAI
mkdir -p InvokeAI
cp install.sh InvokeAI
cp readme.txt InvokeAI

zip -r InvokeAI-linux.zip InvokeAI
zip -r InvokeAI-mac.zip InvokeAI

# make the installer zip for windows
rm -rf InvokeAI
mkdir -p InvokeAI
cp install.bat InvokeAI
cp readme.txt InvokeAI
cp WinLongPathsEnabled.reg InvokeAI

zip -r InvokeAI-windows.zip InvokeAI

rm -rf InvokeAI

echo "The installer zips are ready for distribution."
