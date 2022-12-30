#!/usr/bin/env bash

# Variables shared by build.sh and run.sh
REPOSITORY_NAME=${REPOSITORY_NAME:-$(basename "$(git rev-parse --show-toplevel)")}
VOLUMENAME=${VOLUMENAME:-${REPOSITORY_NAME,,}_data}
ARCH=${ARCH:-$(uname -m)}
PLATFORM=${PLATFORM:-Linux/${ARCH}}
CONTAINER_FLAVOR=${CONTAINER_FLAVOR:-cuda}
INVOKEAI_BRANCH=$(git branch --show)
INVOKEAI_TAG=${REPOSITORY_NAME,,}-${CONTAINER_FLAVOR}:${INVOKEAI_TAG:-${INVOKEAI_BRANCH/\//-}}
