#!/usr/bin/env bash

# This file is used to set environment variables for the build.sh and run.sh scripts.

# Try to detect the container flavor if no PIP_EXTRA_INDEX_URL got specified
if [[ -z "$PIP_EXTRA_INDEX_URL" ]]; then

  # Activate virtual environment if not already activated and exists
  if [[ -z $VIRTUAL_ENV ]]; then
    [[ -e "$(dirname "${BASH_SOURCE[0]}")/../.venv/bin/activate" ]] \
      && source "$(dirname "${BASH_SOURCE[0]}")/../.venv/bin/activate" \
      && echo "Activated virtual environment: $VIRTUAL_ENV"
  fi

  # Decide which container flavor to build if not specified
  if [[ -z "$CONTAINER_FLAVOR" ]] && python -c "import torch" &>/dev/null; then
    # Check for CUDA and ROCm
    CUDA_AVAILABLE=$(python -c "import torch;print(torch.cuda.is_available())")
    ROCM_AVAILABLE=$(python -c "import torch;print(torch.version.hip is not None)")
    if [[ "${CUDA_AVAILABLE}" == "True" ]]; then
      CONTAINER_FLAVOR="cuda"
    elif [[ "${ROCM_AVAILABLE}" == "True" ]]; then
      CONTAINER_FLAVOR="rocm"
    else
      CONTAINER_FLAVOR="cpu"
    fi
  fi

  # Set PIP_EXTRA_INDEX_URL based on container flavor
  if [[ "$CONTAINER_FLAVOR" == "rocm" ]]; then
    PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/rocm"
  elif [[ "$CONTAINER_FLAVOR" == "cpu" ]]; then
    PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu"
  # elif [[ -z "$CONTAINER_FLAVOR" || "$CONTAINER_FLAVOR" == "cuda" ]]; then
  #   PIP_PACKAGE=${PIP_PACKAGE-".[xformers]"}
  fi
fi

# Variables shared by build.sh and run.sh
REPOSITORY_NAME="${REPOSITORY_NAME-$(basename "$(git rev-parse --show-toplevel)")}"
REPOSITORY_NAME="${REPOSITORY_NAME,,}"
VOLUMENAME="${VOLUMENAME-"${REPOSITORY_NAME}_data"}"
ARCH="${ARCH-$(uname -m)}"
PLATFORM="${PLATFORM-linux/${ARCH}}"
INVOKEAI_BRANCH="${INVOKEAI_BRANCH-$(git branch --show)}"
CONTAINER_REGISTRY="${CONTAINER_REGISTRY-"ghcr.io"}"
CONTAINER_REPOSITORY="${CONTAINER_REPOSITORY-"$(whoami)/${REPOSITORY_NAME}"}"
CONTAINER_FLAVOR="${CONTAINER_FLAVOR-cuda}"
CONTAINER_TAG="${CONTAINER_TAG-"${INVOKEAI_BRANCH##*/}-${CONTAINER_FLAVOR}"}"
CONTAINER_IMAGE="${CONTAINER_REGISTRY}/${CONTAINER_REPOSITORY}:${CONTAINER_TAG}"
CONTAINER_IMAGE="${CONTAINER_IMAGE,,}"
