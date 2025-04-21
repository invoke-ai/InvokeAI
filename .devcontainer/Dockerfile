FROM mcr.microsoft.com/devcontainers/typescript-node:22
# The default username for most devcontainer-base-images is 'vscode', but the typescript-node image uses 'node'.
ARG USERNAME=node

RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt update && apt install -y --no-install-recommends \
    ca-certificates \
    gosu \
    libglib2.0-0 \
    libgl1 \
    libglx-mesa0 \
    build-essential \
    libopencv-dev \
    libstdc++-12-dev

# Link amdgpu.ids for ROCm builds
# contributed by https://github.com/Rubonnek
RUN mkdir -p "/opt/amdgpu/share/libdrm" &&\
ln -s "/usr/share/libdrm/amdgpu.ids" "/opt/amdgpu/share/libdrm/amdgpu.ids"

ARG COMPUTE_DEVICE=cuda
ARG INVOKEAI_SRC=/home/${USERNAME}/src/invokeai
ARG INVOKEAI_ROOT=/home/${USERNAME}/invokeai-data
ARG UV_PROJECT_ENVIRONMENT=/home/${USERNAME}/.venv
ARG HF_HOME=/home/${USERNAME}/.invokeai-cache/huggingface

ENV \
    VIRTUAL_ENV=${UV_PROJECT_ENVIRONMENT} \
    UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT}

ENV \
    COMPUTE_DEVICE=${COMPUTE_DEVICE} \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_MANAGED_PYTHON=1 \
    INVOKEAI_HOST=0.0.0.0 \
    INVOKEAI_PORT=9090 \
    INVOKEAI_ROOT=${INVOKEAI_ROOT} \
    HF_HOME=${HF_HOME}

# Set Python version
# NOTE: this is the version of Python that will be installed by `uv`
ARG PYTHON_VERSION=3.12
ENV \
PYTHON_VERSION=${PYTHON_VERSION} \
UV_PYTHON=${PYTHON_VERSION}

# Volumes for the dependency stores/caches
ARG CACHE_DIR=/home/${USERNAME}/.cache
# Setting XDG_CACHE_HOME will affect the location of the cache for pnpm and uv, and possibly other tools
# https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html
# We set XDG_DATA_HOME to a non-volume-mounted location to guarantee that UV pre-installs python. (if its mounted it will only be installed after container start)
ENV XDG_CACHE_HOME=${CACHE_DIR} \
    XDG_DATA_HOME=/home/${USERNAME}/.data \
    GIT_LFS_CACHE=${CACHE_DIR}/git-lfs \
    PATH="$UV_PROJECT_ENVIRONMENT/bin:$CACHE_DIR/pnpm:$CACHE_DIR/uv:$PATH" \
    PYTHONPATH="$PYTHONPATH:${INVOKEAI_SRC}"

# Predeclare volume mounting points for directories which will be written to.
# This guarantees that even if the devcontainer doesnt mount a volume for these paths, their contents will still be handled by a volume, 
# rather than being written to the container image layers itself.
VOLUME [ "${CACHE_DIR}", "${INVOKEAI_SRC}", "${INVOKEAI_ROOT}", "${HF_HOME}"]

# Create & set ownership of directories
RUN set -eux; \
    mkdir -p ${CACHE_DIR} && \
    mkdir -p ${CACHE_DIR}/uv && \
    mkdir -p ${CACHE_DIR}/pnpm && \
    mkdir -p ${CACHE_DIR}/node && \
    mkdir -p ${CACHE_DIR}/Microsoft && \
    mkdir -p ${INVOKEAI_ROOT}/models/.download_cache && \
    mkdir -p ${UV_PROJECT_ENVIRONMENT} && \
    chown --recursive ${USERNAME}:${USERNAME} ${CACHE_DIR} && \
    chown --recursive ${USERNAME}:${USERNAME} ${INVOKEAI_ROOT} && \
    chown --recursive ${USERNAME}:${USERNAME} ${UV_PROJECT_ENVIRONMENT} && \
    mkdir -p /home/${USERNAME}/.data && \
    chown --recursive ${USERNAME}:${USERNAME} /home/${USERNAME}/.data

# Setup PNPM
RUN corepack use pnpm && corepack enable

# Install `uv` for package management
COPY --from=ghcr.io/astral-sh/uv:0.6 /uv /uvx /bin/

RUN uv venv && uv python install ${PYTHON_VERSION}
ENV UV_PYTHON_DOWNLOADS=never

WORKDIR ${INVOKEAI_SRC}