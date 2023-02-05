#!/usr/bin/env bash
set -e

# How to use: https://invoke-ai.github.io/InvokeAI/installation/INSTALL_DOCKER/#setup
# Some possible pip extra-index urls (cuda 11.7 is available without extra url):
#   CUDA 11.6:  https://download.pytorch.org/whl/cu116
#   ROCm 5.2:   https://download.pytorch.org/whl/rocm5.2
#   CPU:        https://download.pytorch.org/whl/cpu
#   as found on https://pytorch.org/get-started/locally/

SCRIPTDIR=$(dirname "$0")
cd "$SCRIPTDIR" || exit 1

source ./env.sh

DOCKERFILE=${INVOKE_DOCKERFILE:-Dockerfile}

# print the settings
echo -e "You are using these values:\n"
echo -e "Dockerfile:\t\t${DOCKERFILE}"
echo -e "index-url:\t\t${PIP_EXTRA_INDEX_URL:-none}"
echo -e "Volumename:\t\t${VOLUMENAME}"
echo -e "Platform:\t\t${PLATFORM}"
echo -e "Registry:\t\t${CONTAINER_REGISTRY}"
echo -e "Repository:\t\t${CONTAINER_REPOSITORY}"
echo -e "Container Tag:\t\t${CONTAINER_TAG}"
echo -e "Container Image:\t${CONTAINER_IMAGE}\n"

# Create docker volume
if [[ -n "$(docker volume ls -f name="${VOLUMENAME}" -q)" ]]; then
    echo -e "Volume already exists\n"
else
    echo -n "createing docker volume "
    docker volume create "${VOLUMENAME}"
fi

# Build Container
DOCKER_BUILDKIT=1 docker build \
    --platform="${PLATFORM}" \
    --tag="${CONTAINER_IMAGE}" \
    ${PIP_EXTRA_INDEX_URL:+--build-arg="PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL}"} \
    ${PIP_PACKAGE:+--build-arg="PIP_PACKAGE=${PIP_PACKAGE}"} \
    --file="${DOCKERFILE}" \
    ..
