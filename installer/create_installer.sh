#!/bin/bash

set -e

BCYAN="\033[1;36m"
BYELLOW="\033[1;33m"
BGREEN="\033[1;32m"
BRED="\033[1;31m"
RED="\033[31m"
RESET="\033[0m"

function git_show {
    git show -s --format=oneline --abbrev-commit "$1" | cat
}

if [[ ! -z "${VIRTUAL_ENV}" ]]; then
    # we can't just call 'deactivate' because this function is not exported
    # to the environment of this script from the bash process that runs the script
    echo -e "${BRED}A virtual environment is activated. Please deactivate it before proceeding.${RESET}"
    exit -1
fi

cd "$(dirname "$0")"

VERSION=$(
    cd ..
    python3 -c "from invokeai.version import __version__ as version; print(version)"
)
VERSION="v${VERSION}"

if [[ ! -z ${CI} ]]; then
    echo
    echo -e "${BCYAN}CI environment detected${RESET}"
    echo
else
    echo
    echo -e "${BYELLOW}This script must be run from the installer directory!${RESET}"
    echo "The current working directory is $(pwd)"
    read -p "If that looks right, press any key to proceed, or CTRL-C to exit..."
    echo
fi

echo -e "${BGREEN}HEAD${RESET}:"
git_show HEAD
echo

# ---------------------- FRONTEND ----------------------

pushd ../invokeai/frontend/web >/dev/null
echo "Installing frontend dependencies..."
echo
pnpm i --frozen-lockfile
echo
if [[ ! -z ${CI} ]]; then
    echo "Building frontend without checks..."
    # In CI, we have already done the frontend checks and can just build
    pnpm vite build
else
    echo "Running checks and building frontend..."
    # This runs all the frontend checks and builds
    pnpm build
fi
echo
popd

# ---------------------- BACKEND ----------------------

echo
echo "Building wheel..."
echo

# install the 'build' package in the user site packages, if needed
# could be improved by using a temporary venv, but it's tiny and harmless
if [[ $(python3 -c 'from importlib.util import find_spec; print(find_spec("build") is None)') == "True" ]]; then
    pip install --user build
fi

rm -rf ../build

python3 -m build --outdir dist/ ../.

# ----------------------

echo
echo "Building installer zip files for InvokeAI ${VERSION}..."
echo

# get rid of any old ones
rm -f *.zip
rm -rf InvokeAI-Installer

# copy content
mkdir InvokeAI-Installer
for f in templates *.txt *.reg; do
    cp -r ${f} InvokeAI-Installer/
done
mkdir InvokeAI-Installer/lib
cp lib/*.py InvokeAI-Installer/lib

# Install scripts
# Mac/Linux
cp install.sh.in InvokeAI-Installer/install.sh
chmod a+x InvokeAI-Installer/install.sh

# Windows
cp install.bat.in InvokeAI-Installer/install.bat
cp WinLongPathsEnabled.reg InvokeAI-Installer/

FILENAME=InvokeAI-installer-$VERSION.zip

# Zip everything up
zip -r ${FILENAME} InvokeAI-Installer

echo
echo -e "${BGREEN}Built installer: ./${FILENAME}${RESET}"
echo -e "${BGREEN}Built PyPi distribution: ./dist${RESET}"

# clean up, but only if we are not in a github action
if [[ -z ${CI} ]]; then
    echo
    echo "Cleaning up intermediate build files..."
    rm -rf InvokeAI-Installer tmp ../invokeai/frontend/web/dist/
fi

if [[ ! -z ${CI} ]]; then
    echo
    echo "Setting GitHub action outputs..."
    echo "INSTALLER_FILENAME=${FILENAME}" >>$GITHUB_OUTPUT
    echo "INSTALLER_PATH=installer/${FILENAME}" >>$GITHUB_OUTPUT
    echo "DIST_PATH=installer/dist/" >>$GITHUB_OUTPUT
fi

exit 0
