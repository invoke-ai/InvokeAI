#!/usr/bin/env bash

repository_name=$(basename "$(git rev-parse --show-toplevel)")
repository_name_lc=${repository_name,,}
volumename=${VOLUMENAME:-${repository_name_lc}_data}
arch=${ARCH:-$(arch)}
platform=${PLATFORM:-Linux/${arch}}
container_flavor=${CONTAINER_FLAVOR:-cuda}
invokeai_tag=${repository_name_lc}-${container_flavor}:${INVOKEAI_TAG:-latest}
gpus=${GPU_FLAGS:+--gpus=${GPU_FLAGS}}

export repository_name
export volumename
export arch
export platform
export invokeai_tag
export gpus
