#!/usr/bin/env bash
set -e

# How to use: https://invoke-ai.github.io/InvokeAI/installation/INSTALL_DOCKER/#setup

source ./docker-build/env.sh \
  || echo "please execute docker-build/build.sh from repository root" \
  || exit 1

PIP_REQUIREMENTS=${PIP_REQUIREMENTS:-requirements-lin-cuda.txt}
DOCKERFILE=${INVOKE_DOCKERFILE:-docker-build/Dockerfile}

# print the settings
echo -e "You are using these values:\n"
echo -e "Dockerfile:\t ${DOCKERFILE}"
echo -e "Requirements:\t ${PIP_REQUIREMENTS}"
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
  --build-arg="PIP_REQUIREMENTS=${PIP_REQUIREMENTS}" \
  --file="${DOCKERFILE}" \
  .
