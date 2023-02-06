#!/usr/bin/env bash

if [[ -z "$PIP_EXTRA_INDEX_URL" ]]; then
  # Decide which container flavor to build if not specified
  if [[ -z "$CONTAINER_FLAVOR" ]] && python -c "import torch" &>/dev/null; then
    # Check for CUDA and ROCm
    CUDA_AVAILABLE=$(python -c "import torch;print(torch.cuda.is_available())")
    ROCM_AVAILABLE=$(python -c "import torch;print(torch.version.hip is not None)")
    if [[ "$(uname -s)" != "Darwin" && "${CUDA_AVAILABLE}" == "True" ]]; then
      CONTAINER_FLAVOR="cuda"
    elif [[ "$(uname -s)" != "Darwin" && "${ROCM_AVAILABLE}" == "True" ]]; then
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
VOLUMENAME="${VOLUMENAME-"${REPOSITORY_NAME,,}_data"}"
ARCH="${ARCH-$(uname -m)}"
PLATFORM="${PLATFORM-Linux/${ARCH}}"
INVOKEAI_BRANCH="${INVOKEAI_BRANCH-$(git branch --show)}"
CONTAINER_REGISTRY="${CONTAINER_REGISTRY-"ghcr.io"}"
CONTAINER_REPOSITORY="${CONTAINER_REPOSITORY-"$(whoami)/${REPOSITORY_NAME}"}"
CONTAINER_FLAVOR="${CONTAINER_FLAVOR-cuda}"
CONTAINER_TAG="${CONTAINER_TAG-"${INVOKEAI_BRANCH##*/}-${CONTAINER_FLAVOR}"}"
CONTAINER_IMAGE="${CONTAINER_REGISTRY}/${CONTAINER_REPOSITORY}:${CONTAINER_TAG}"
CONTAINER_IMAGE="${CONTAINER_IMAGE,,}"
