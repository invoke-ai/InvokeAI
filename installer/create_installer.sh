#!/bin/bash

cd "$(dirname "$0")"

VERSION=$(grep ^VERSION ../setup.py | awk '{ print $3 }' | sed "s/'//g" )
VERSION="$VERSION-p5"

echo "Be certain that you're in the 'installer' directory before continuing."
read -p "Press any key to continue, or CTRL-C to exit..."

echo Building installer zip fles for InvokeAI v$VERSION

# get rid of any old ones
rm *.zip

rm -rf InvokeAI-Installer
mkdir InvokeAI-Installer

cp -pr ../environments-and-requirements templates readme.txt InvokeAI-Installer/
mkdir InvokeAI-Installer/templates/rootdir

cp -pr ../configs InvokeAI-Installer/templates/rootdir/

mkdir InvokeAI-Installer/templates/rootdir/{outputs,embeddings,models}

cp install.sh.in InvokeAI-Installer/install.sh
chmod a+rx InvokeAI-Installer/install.sh

zip -r InvokeAI-installer-$VERSION-linux.zip InvokeAI-Installer
zip -r InvokeAI-installer-$VERSION-mac.zip InvokeAI-Installer

# now do the windows installer
rm InvokeAI-Installer/install.sh
cp install.bat.in InvokeAI-Installer/install.bat
cp WinLongPathsEnabled.reg InvokeAI-Installer/

# this gets rid of the "-e ." at the end of the windows requirements file
# because it is easier to do it now than in the .bat install script
egrep -v '^-e .' InvokeAI-Installer/environments-and-requirements/requirements-win-colab-cuda.txt > InvokeAI-Installer/requirements.txt
cp InvokeAI-Installer/requirements.txt InvokeAI-Installer/environments-and-requirements/requirements-win-colab-cuda.txt
zip -r InvokeAI-installer-$VERSION-windows.zip InvokeAI-Installer

# clean up
rm -rf InvokeAI-Installer


exit 0


