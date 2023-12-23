#!/bin/bash

set -e

BCYAN="\033[1;36m"
BYELLOW="\033[1;33m"
BGREEN="\033[1;32m"
BRED="\033[1;31m"
RED="\033[31m"
RESET="\033[0m"

function is_bin_in_path {
    builtin type -P "$1" &>/dev/null
}

function git_show {
    git show -s --format=oneline --abbrev-commit "$1" | cat
}

if [[ -v "VIRTUAL_ENV" ]]; then
    # we can't just call 'deactivate' because this function is not exported
    # to the environment of this script from the bash process that runs the script
    echo -e "${BRED}A virtual environment is activated. Please deactivate it before proceeding.${RESET}"
    exit -1
fi

cd "$(dirname "$0")"

echo
echo -e "${BYELLOW}This script must be run from the installer directory!${RESET}"
echo "The current working directory is $(pwd)"
read -p "If that looks right, press any key to proceed, or CTRL-C to exit..."
echo

# Some machines only have `python3` in PATH, others have `python` - make an alias.
# We can use a function to approximate an alias within a non-interactive shell.
if ! is_bin_in_path python && is_bin_in_path python3; then
    function python {
        python3 "$@"
    }
fi

VERSION=$(
    cd ..
    python -c "from invokeai.version import __version__ as version; print(version)"
)
VERSION="v${VERSION}"

echo -e "${BGREEN}HEAD${RESET}:"
git_show HEAD
echo

# ---------------------- FRONTEND ----------------------

pushd ../invokeai/frontend/web >/dev/null
echo
echo "Installing frontend dependencies..."
echo
pnpm i --frozen-lockfile
echo
echo "Building frontend..."
if [[ -v CI ]]; then
    # In CI, we have already done the frontend checks and can just build
    pnpm vite build
else
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
if [[ $(python -c 'from importlib.util import find_spec; print(find_spec("build") is None)') == "True" ]]; then
    pip install --user build
fi

rm -rf ../build

python -m build --outdir dist/ ../.

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
zip -r $FILENAME InvokeAI-Installer

if [[ ! -v CI ]]; then
    # clean up, but only if we are not in a github action
    rm -rf InvokeAI-Installer tmp dist ../invokeai/frontend/web/dist/
fi

if [[ -v CI ]]; then
    # Set the output variable for github action
    echo "INSTALLER_FILENAME=$FILENAME" >>$GITHUB_OUTPUT
    echo "INSTALLER_PATH=installer/$FILENAME" >>$GITHUB_OUTPUT
    echo "DIST_PATH=installer/dist/" >>$GITHUB_OUTPUT
fi

exit 0
