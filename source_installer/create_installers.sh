#!/bin/bash

cd "$(dirname "${BASH_SOURCE[0]}")"

VERSION='2.2.3'

# make the installer zip for linux and mac
rm -rf invokeAI
mkdir -p invokeAI
cp install.sh.in invokeAI/install.sh
chmod a+x invokeAI/install.sh
cp readme.txt invokeAI

zip -r invokeAI-src-installer-$VERSION-linux.zip invokeAI
zip -r invokeAI-src-installer-$VERSION-mac.zip invokeAI

# make the installer zip for windows
rm -rf invokeAI
mkdir -p invokeAI
cp install.bat.in invokeAI/install.bat
cp readme.txt invokeAI
cp WinLongPathsEnabled.reg invokeAI

zip -r invokeAI-src-installer-$VERSION-windows.zip invokeAI

rm -rf invokeAI
echo "The installer zips are ready to be distributed.."
