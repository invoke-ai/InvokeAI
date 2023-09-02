#!/bin/bash

set -e

cd "$(dirname "$0")"

if [[ -v "VIRTUAL_ENV" ]]; then
    # we can't just call 'deactivate' because this function is not exported
    # to the environment of this script from the bash process that runs the script
    echo "A virtual environment is activated. Please deactivate it before proceeding".
    exit -1
fi

VERSION=$(cd ..; python -c "from invokeai.version import __version__ as version; print(version)")
PATCH=""
VERSION="v${VERSION}${PATCH}"
LATEST_TAG="v3.0-latest"

echo Building installer for version $VERSION
echo "Be certain that you're in the 'installer' directory before continuing."
read -p "Press any key to continue, or CTRL-C to exit..."

read -e -p "Tag this repo with '${VERSION}' and '${LATEST_TAG}'? [n]: " input
RESPONSE=${input:='n'}
if [ "$RESPONSE" == 'y' ]; then

    git push origin :refs/tags/$VERSION
    if ! git tag -fa $VERSION ; then
	    echo "Existing/invalid tag"
	    exit -1
    fi

    git push origin :refs/tags/$LATEST_TAG
    git tag -fa $LATEST_TAG

    echo "remember to push --tags!"
fi

# ----------------------

echo Building the wheel

# install the 'build' package in the user site packages, if needed
# could be improved by using a temporary venv, but it's tiny and harmless
if [[ $(python -c 'from importlib.util import find_spec; print(find_spec("build") is None)') == "True" ]]; then
    pip install --user build
fi

rm -r ../build
python -m build --wheel --outdir dist/ ../.

# ----------------------

echo Building installer zip fles for InvokeAI $VERSION

# get rid of any old ones
rm -f *.zip
rm -rf InvokeAI-Installer

# copy content
mkdir InvokeAI-Installer
for f in templates lib *.txt *.reg; do
    cp -r ${f} InvokeAI-Installer/
done

# Move the wheel
mv dist/*.whl InvokeAI-Installer/lib/

# Install scripts
# Mac/Linux
cp install.sh.in InvokeAI-Installer/install.sh
chmod a+x InvokeAI-Installer/install.sh

# Windows
perl -p -e "s/^set INVOKEAI_VERSION=.*/set INVOKEAI_VERSION=$VERSION/" install.bat.in > InvokeAI-Installer/install.bat
cp WinLongPathsEnabled.reg InvokeAI-Installer/

# Zip everything up
zip -r InvokeAI-installer-$VERSION.zip InvokeAI-Installer

# clean up
rm -rf InvokeAI-Installer tmp dist

exit 0
