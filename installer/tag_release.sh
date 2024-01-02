#!/bin/bash

set -e

BCYAN="\e[1;36m"
BYELLOW="\e[1;33m"
BGREEN="\e[1;32m"
BRED="\e[1;31m"
RED="\e[31m"
RESET="\e[0m"

function does_tag_exist {
    git rev-parse --quiet --verify "refs/tags/$1" >/dev/null
}

function git_show_ref {
    git show-ref --dereference $1 --abbrev 7
}

function git_show {
    git show -s --format='%h %s' $1
}

VERSION=$(
    cd ..
    python -c "from invokeai.version import __version__ as version; print(version)"
)
PATCH=""
MAJOR_VERSION=$(echo $VERSION | sed 's/\..*$//')
VERSION="v${VERSION}${PATCH}"
LATEST_TAG="v${MAJOR_VERSION}-latest"

if does_tag_exist $VERSION; then
    echo -e "${BCYAN}${VERSION}${RESET} already exists:"
    git_show_ref tags/$VERSION
    echo
fi
if does_tag_exist $LATEST_TAG; then
    echo -e "${BCYAN}${LATEST_TAG}${RESET} already exists:"
    git_show_ref tags/$LATEST_TAG
    echo
fi

echo -e "${BGREEN}HEAD${RESET}:"
git_show
echo

echo -e -n "Create tags ${BCYAN}${VERSION}${RESET} and ${BCYAN}${LATEST_TAG}${RESET} @ ${BGREEN}HEAD${RESET}, ${RED}deleting existing tags on remote${RESET}? "
read -e -p 'y/n [n]: ' input
RESPONSE=${input:='n'}
if [ "$RESPONSE" == 'y' ]; then
    echo
    echo -e "Deleting ${BCYAN}${VERSION}${RESET} tag on remote..."
    git push --delete origin $VERSION

    echo -e "Tagging ${BGREEN}HEAD${RESET} with ${BCYAN}${VERSION}${RESET} locally..."
    if ! git tag -fa $VERSION; then
        echo "Existing/invalid tag"
        exit -1
    fi

    echo -e "Deleting ${BCYAN}${LATEST_TAG}${RESET} tag on remote..."
    git push --delete origin $LATEST_TAG

    echo -e "Tagging ${BGREEN}HEAD${RESET} with ${BCYAN}${LATEST_TAG}${RESET} locally..."
    git tag -fa $LATEST_TAG

    echo -e "Pushing updated tags to remote..."
    git push origin --tags
fi
exit 0
