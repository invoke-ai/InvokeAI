#!/usr/bin/env bash

# Variables shared by build.sh and run.sh
repository_name=${REPOSITORY_NAME:-$(basename "$(git rev-parse --show-toplevel)")}
volumename=${VOLUMENAME:-${repository_name,,}_data}
arch=${ARCH:-$(arch)}
platform=${PLATFORM:-Linux/${arch}}
container_flavor=${CONTAINER_FLAVOR:-cuda}
invokeai_tag=${repository_name,,}-${container_flavor}:${INVOKEAI_TAG:-latest}
