#!/usr/bin/env bash
set -e

source ./docker-build/env.sh || echo "please run from repository root" || exit 1

echo -e "You are using these values:\n"
echo -e "volumename:\t ${volumename}"
echo -e "invokeai_tag:\t ${invokeai_tag}\n"

docker run \
  --interactive \
  --tty \
  --rm \
  --platform="$platform" \
  --name="$project_name" \
  --hostname="$project_name" \
  --mount="source=$volumename,target=/data" \
  --publish=9090:9090 \
  --cap-add=sys_nice \
  "$invokeai_tag" ${1:+$@}
