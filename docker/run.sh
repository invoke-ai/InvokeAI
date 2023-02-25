#!/usr/bin/env bash
set -e

# How to use: https://invoke-ai.github.io/InvokeAI/installation/040_INSTALL_DOCKER/

SCRIPTDIR=$(dirname "${BASH_SOURCE[0]}")
cd "$SCRIPTDIR" || exit 1

source ./env.sh

# Create outputs directory if it does not exist
[[ -d ./outputs ]] || mkdir ./outputs

echo -e "You are using these values:\n"
echo -e "Volumename:\t${VOLUMENAME}"
echo -e "Invokeai_tag:\t${CONTAINER_IMAGE}"
echo -e "local Models:\t${MODELSPATH:-unset}\n"

docker run \
  --interactive \
  --tty \
  --rm \
  --platform="${PLATFORM}" \
  --name="${REPOSITORY_NAME}" \
  --hostname="${REPOSITORY_NAME}" \
  --mount type=volume,volume-driver=local,source="${VOLUMENAME}",target=/data \
  --mount type=bind,source="$(pwd)"/outputs/,target=/data/outputs/ \
  ${MODELSPATH:+--mount="type=bind,source=${MODELSPATH},target=/data/models"} \
  ${HUGGING_FACE_HUB_TOKEN:+--env="HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}"} \
  --publish=9090:9090 \
  --cap-add=sys_nice \
  ${GPU_FLAGS:+--gpus="${GPU_FLAGS}"} \
  "${CONTAINER_IMAGE}" ${@:+$@}

echo -e "\nCleaning trash folder ..."
for f in outputs/.Trash*; do
  if [ -e "$f" ]; then
    rm -Rf "$f"
    break
  fi
done
