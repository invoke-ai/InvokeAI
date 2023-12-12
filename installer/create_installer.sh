#!/bin/bash

set -e

BCYAN="\e[1;36m"
BYELLOW="\e[1;33m"
BGREEN="\e[1;32m"
BRED="\e[1;31m"
RED="\e[31m"
RESET="\e[0m"

function is_bin_in_path {
    builtin type -P "$1" &>/dev/null
}

function git_show {
    git show -s --format='%h %s' $1
}

cd "$(dirname "$0")"

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

if [[ -v "VIRTUAL_ENV" ]]; then
    # we can't just call 'deactivate' because this function is not exported
    # to the environment of this script from the bash process that runs the script
    echo -e "${BRED}A virtual environment is activated. Please deactivate it before proceeding.${RESET}"
    exit -1
fi

VERSION=$(
    cd ..
    python -c "from invokeai.version import __version__ as version; print(version)"
)
PATCH=""
VERSION="v${VERSION}${PATCH}"

echo -e "${BGREEN}HEAD${RESET}:"
git_show
echo

# ---------------------- FRONTEND ----------------------

pushd ../invokeai/frontend/web >/dev/null
echo
echo "Installing frontend dependencies..."
echo
pnpm i --frozen-lockfile
echo
echo "Building frontend..."
echo
pnpm build
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

python -m build --wheel --outdir dist/ ../.

# ----------------------

echo
echo "Building installer zip files for InvokeAI ${VERSION}..."
echo

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
perl -p -e "s/^set INVOKEAI_VERSION=.*/set INVOKEAI_VERSION=$VERSION/" install.bat.in >InvokeAI-Installer/install.bat
cp WinLongPathsEnabled.reg InvokeAI-Installer/

# Zip everything up
zip -r InvokeAI-installer-$VERSION.zip InvokeAI-Installer

# clean up
rm -rf InvokeAI-Installer tmp dist

exit 0
