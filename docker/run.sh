#!/usr/bin/env bash
set -e

# How to use: https://invoke-ai.github.io/InvokeAI/installation/040_INSTALL_DOCKER/

SCRIPTDIR=$(dirname "${BASH_SOURCE[0]}")
cd "$SCRIPTDIR" || exit 1

source ./env.sh

echo -e "You are using these values:\n"
echo -e "Container engine:\t${CONTAINER_ENGINE}"
echo -e "Volumename:\t${VOLUMENAME}"
echo -e "Invokeai_tag:\t${CONTAINER_IMAGE}"
echo -e "local Models:\t${MODELSPATH:-unset}\n"

if [[ "${CONTAINER_ENGINE}" == "podman" ]]; then
   PODMAN_ARGS="--user=appuser:appuser"
   unset PLATFORM #causes problems
fi

"${CONTAINER_ENGINE}" run \
  --interactive \
  --tty \
  --rm \
  ${PLATFORM+--platform="${PLATFORM}"} \
  --name="${REPOSITORY_NAME,,}" \
  --hostname="${REPOSITORY_NAME,,}" \
  --mount type=volume,src="${VOLUMENAME}",target=/data \
  --mount type=bind,source="$(pwd)"/outputs,target=/data/outputs \
  ${MODELSPATH:+--mount="type=bind,source=${MODELSPATH},target=/data/models"} \
  ${HUGGING_FACE_HUB_TOKEN:+--env="HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}"} \
  --publish=9090:9090 \
  --cap-add=sys_nice \
  ${PODMAN_ARGS:+"${PODMAN_ARGS}"} \
  ${GPU_FLAGS:+--gpus="${GPU_FLAGS}"} \
  "${CONTAINER_IMAGE}" ${@:+$@}

# Remove Trash folder
for f in outputs/.Trash*; do
  if [ -e "$f" ]; then
    rm -Rf "$f"
    break
  fi
done
