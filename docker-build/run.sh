#!/usr/bin/env bash
set -e

source ./docker-build/env.sh || echo "please run from repository root" || exit 1

docker run \
  --interactive \
  --tty \
  --rm \
  --platform="$platform" \
  --name="$project_name" \
  --hostname="$project_name" \
  --mount="source=$volumename,target=/data" \
  --publish=9090:9090 \
  "$invokeai_tag" ${1:+$@}
