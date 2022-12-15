#!/usr/bin/env bash

# Variables shared by build.sh and run.sh
REPOSITORY_NAME=${REPOSITORY_NAME:-$(basename "$(git rev-parse --show-toplevel)")}
VOLUMENAME=${VOLUMENAME:-${REPOSITORY_NAME,,}_data}
ARCH=${ARCH:-$(arch)}
PLATFORM=${PLATFORM:-Linux/${ARCH}}
CONTAINER_FLAVOR=${CONTAINER_FLAVOR:-cuda}
INVOKEAI_TAG=${REPOSITORY_NAME,,}-${CONTAINER_FLAVOR}:${INVOKEAI_TAG:-latest}
