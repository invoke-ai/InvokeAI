#!/bin/bash

cd "$(dirname "$0")"

VERSION=$(grep ^VERSION ../setup.py | awk '{ print $3 }' | sed "s/'//g" )

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

zip -r InvokeAI-simple-installer-$VERSION-linux.zip InvokeAI-Installer
zip -r InvokeAI-simple-installer-$VERSION-mac.zip InvokeAI-Installer

# clean up
rm -rf InvokeAI-Installer
exit 0


