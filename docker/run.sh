#!/usr/bin/env bash
set -e

# How to use: https://invoke-ai.github.io/InvokeAI/installation/INSTALL_DOCKER/#run-the-container
# IMPORTANT: You need to have a token on huggingface.co to be able to download the checkpoints!!!

SCRIPTDIR=$(dirname "$0")
cd "$SCRIPTDIR" || exit 1

source ./env.sh

echo -e "You are using these values:\n"
echo -e "Volumename:\t${VOLUMENAME}"
echo -e "Invokeai_tag:\t${CONTAINER_IMAGE}"
echo -e "local Models:\t${MODELSPATH:-unset}\n"

docker run \
  --interactive \
  --tty \
  --rm \
  --platform="${PLATFORM}" \
  --name="${REPOSITORY_NAME,,}" \
  --hostname="${REPOSITORY_NAME,,}" \
  --mount=source="${VOLUMENAME}",target=/data \
  ${MODELSPATH:+-u "$(id -u):$(id -g)"} \
  ${MODELSPATH:+--mount="type=bind,source=${MODELSPATH},target=/data/models"} \
  ${HUGGING_FACE_HUB_TOKEN:+--env="HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}"} \
  --publish=9090:9090 \
  --cap-add=sys_nice \
  ${GPU_FLAGS:+--gpus="${GPU_FLAGS}"} \
  "${CONTAINER_IMAGE}" ${1:+$@}
