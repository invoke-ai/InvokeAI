#!/usr/bin/env bash
set -e

# How to use: https://invoke-ai.github.io/InvokeAI/installation/INSTALL_DOCKER/#setup
#
# Some possible pip extra-index urls (cuda 11.7 is available without extra url):
#
#   CUDA 11.6:  https://download.pytorch.org/whl/cu116
#   ROCm 5.2:   https://download.pytorch.org/whl/rocm5.2
#   CPU:        https://download.pytorch.org/whl/cpu
#
#   as found on https://pytorch.org/get-started/locally/

cd "$(dirname "$0")" || exit 1

source ./env.sh

DOCKERFILE=${INVOKE_DOCKERFILE:-"./Dockerfile"}

# print the settings
echo -e "You are using these values:\n"
echo -e "Dockerfile:\t ${DOCKERFILE}"
echo -e "extra-index-url: ${PIP_EXTRA_INDEX_URL:-none}"
echo -e "Volumename:\t ${VOLUMENAME}"
echo -e "arch:\t\t ${ARCH}"
echo -e "Platform:\t ${PLATFORM}"
echo -e "Invokeai_tag:\t ${INVOKEAI_TAG}\n"

if [[ -n "$(docker volume ls -f name="${VOLUMENAME}" -q)" ]]; then
    echo -e "Volume already exists\n"
else
    echo -n "createing docker volume "
    docker volume create "${VOLUMENAME}"
fi

# Build Container
docker build \
    --platform="${PLATFORM}" \
    --tag="${INVOKEAI_TAG}" \
    ${PIP_EXTRA_INDEX_URL:+--build-arg=PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL}"} \
    --file="${DOCKERFILE}" \
    ..
