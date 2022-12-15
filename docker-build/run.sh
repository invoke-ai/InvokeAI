#!/usr/bin/env bash
set -e

source ./docker-build/env.sh \
  || echo "please run from repository root" \
  || exit 1

# check if HUGGINGFACE_TOKEN is available
HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN:?Please set your token for Huggingface as HUGGINGFACE_TOKEN}

echo -e "You are using these values:\n"
echo -e "Volumename:\t ${VOLUMENAME}"
echo -e "Invokeai_tag:\t ${INVOKEAI_TAG}\n"

docker run \
  --interactive \
  --tty \
  --rm \
  --platform="$PLATFORM" \
  --name="${REPOSITORY_NAME,,}" \
  --hostname="${REPOSITORY_NAME,,}" \
  --mount="source=$VOLUMENAME,target=/data" \
  --env="HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}" \
  --publish=9090:9090 \
  --cap-add=sys_nice \
  ${GPU_FLAGS:+--gpus=${GPU_FLAGS}} \
  "$INVOKEAI_TAG" ${1:+$@}
