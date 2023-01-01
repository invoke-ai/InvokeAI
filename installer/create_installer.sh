#!/bin/bash

cd "$(dirname "$0")"

VERSION=$(grep ^VERSION ../setup.py | awk '{ print $3 }' | sed "s/'//g" )
PATCH=""
VERSION="v${VERSION}${PATCH}"

echo "Be certain that you're in the 'installer' directory before continuing."
read -p "Press any key to continue, or CTRL-C to exit..."

git commit -a

if ! git tag $VERSION ; then
    echo "Existing/invalid tag"
    exit -1
fi

git push origin :refs/tags/latest
git tag -fa latest

echo Building installer zip fles for InvokeAI $VERSION

# get rid of any old ones
rm *.zip

rm -rf InvokeAI-Installer
mkdir InvokeAI-Installer

cp -pr ../environments-and-requirements templates readme.txt InvokeAI-Installer/
mkdir InvokeAI-Installer/templates/rootdir

cp -pr ../configs InvokeAI-Installer/templates/rootdir/

mkdir InvokeAI-Installer/templates/rootdir/{outputs,embeddings,models}

perl -p -e "s/^INVOKEAI_VERSION=.*/INVOKEAI_VERSION=\"$VERSION\"/" install.sh.in > InvokeAI-Installer/install.sh
chmod a+rx InvokeAI-Installer/install.sh

zip -r InvokeAI-installer-$VERSION-linux.zip InvokeAI-Installer
zip -r InvokeAI-installer-$VERSION-mac.zip InvokeAI-Installer

# now do the windows installer
rm InvokeAI-Installer/install.sh
perl -p -e "s/^set INVOKEAI_VERSION=.*/set INVOKEAI_VERSION=$VERSION/" install.bat.in > InvokeAI-Installer/install.bat
cp WinLongPathsEnabled.reg InvokeAI-Installer/

# this gets rid of the "-e ." at the end of the windows requirements file
# because it is easier to do it now than in the .bat install script
egrep -v '^-e .' InvokeAI-Installer/environments-and-requirements/requirements-win-colab-cuda.txt > InvokeAI-Installer/requirements.txt
cp InvokeAI-Installer/requirements.txt InvokeAI-Installer/environments-and-requirements/requirements-win-colab-cuda.txt
zip -r InvokeAI-installer-$VERSION-windows.zip InvokeAI-Installer

mkdir tmp
cp templates/update.sh.in tmp/update.sh
cp templates/update.bat.in tmp/update.bat
chmod +x tmp/update.sh
chmod +x tmp/update.bat
cd tmp
zip InvokeAI-updater-$VERSION.zip update.sh update.bat
cd ..
mv tmp/InvokeAI-updater-$VERSION.zip .

# clean up
rm -rf InvokeAI-Installer tmp


exit 0


