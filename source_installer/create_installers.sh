#!/bin/bash

cd "$(dirname "${BASH_SOURCE[0]}")"

# make the installer zip for linux and mac
rm -rf invokeAI
mkdir -p invokeAI
cp install.sh invokeAI
cp readme.txt invokeAI

zip -r invokeAI-src-installer-linux.zip invokeAI
zip -r invokeAI-src-installer-mac.zip invokeAI

# make the installer zip for windows
rm -rf invokeAI
mkdir -p invokeAI
cp install.bat invokeAI
cp readme.txt invokeAI
cp WinLongPathsEnabled.reg invokeAI

zip -r invokeAI-src-installer-windows.zip invokeAI

echo "The installer zips are ready to be distributed.."
