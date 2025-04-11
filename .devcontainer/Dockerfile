FROM mcr.microsoft.com/devcontainers/typescript-node:22

ARG USERNAME=vscode
ARG CONTAINER_UID=1000
ARG CONTAINER_GID=1000

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

ARG GPU_DRIVER=cuda
ARG INVOKEAI_SRC=/home/${USERNAME}/src/invokeai
ARG INVOKEAI_ROOT=/home/${USERNAME}/invokeai-data
ARG UV_PROJECT_ENVIRONMENT=/home/${USERNAME}/.venv

ENV \
    VIRTUAL_ENV=${UV_PROJECT_ENVIRONMENT} \
    UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT}

ENV \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_MANAGED_PYTHON=1 \
    INVOKEAI_HOST=0.0.0.0 \
    INVOKEAI_PORT=9090 \
    CONTAINER_UID=${CONTAINER_UID:-1000} \
    CONTAINER_GID=${CONTAINER_GID:-1000}

# Set the secondary index for UV to the appropriate PyTorch index URL based on the GPU driver and architecture
# NOTE: this is the index URL that will be used by `uv` to install PyTorch and other packages
# Instead of using an envar we write the value to the user-level config file for UV
RUN if [ "$TARGETPLATFORM" = "linux/arm64" ] || [ "$GPU_DRIVER" = "cpu" ]; then export UV_INDEX="https://download.pytorch.org/whl/cpu"; \
    elif [ "$GPU_DRIVER" = "rocm" ]; then export UV_INDEX="https://download.pytorch.org/whl/rocm6.2"; \
    else export UV_INDEX="https://download.pytorch.org/whl/cu126"; \
    fi \
    && mkdir -p /home/${USERNAME}/.config/uv/ \
    && echo "[[index]]\nurl = '$(echo $UV_INDEX)'\n" > /home/${USERNAME}/.config/uv/uv.toml

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
ENV XDG_CACHE_HOME=${CACHE_DIR} \
    GIT_LFS_CACHE=${CACHE_DIR}/git-lfs \
    PATH="$UV_PROJECT_ENVIRONMENT/bin:$CACHE_DIR/pnpm:$CACHE_DIR/uv:$PATH" \
    PYTHONPATH="$PYTHONPATH:${INVOKEAI_SRC}"
    # PYTHONPATH="$UV_PROJECT_ENVIRONMENT/lib/python${PYTHON_VERSION}/site-packages:$PYTHONPATH" \

# Setup cache volume for dependencies
VOLUME [ "${CACHE_DIR}", "${INVOKEAI_ROOT}", "${INVOKEAI_SRC}" ]

# Create & set ownership of directories
RUN set -eux; \
    mkdir -p ${CACHE_DIR} && \
    mkdir -p ${CACHE_DIR}/pnpm && \
    mkdir -p ${CACHE_DIR}/uv && \
    mkdir -p ${INVOKEAI_ROOT} && \
    mkdir -p ${UV_PROJECT_ENVIRONMENT} && \
    chown --recursive ${USERNAME}:${USERNAME} ${CACHE_DIR} && \
    chown --recursive ${USERNAME}:${USERNAME} ${INVOKEAI_ROOT} && \
    chown --recursive ${USERNAME}:${USERNAME} ${UV_PROJECT_ENVIRONMENT} 

# Setup PNPM
RUN corepack use pnpm && corepack enable

# Install `uv` for package management
# TODO: Shouldnt this be put in the users .local dir?
COPY --from=ghcr.io/astral-sh/uv /uv /uvx /bin/

# RUN --mount=type=cache,target=/home/${USERNAME}/.cache/uv \
RUN uv python install ${PYTHON_VERSION}

WORKDIR ${INVOKEAI_SRC}

# RUN --mount=type=cache,target=/home/${USERNAME}/.cache/uv \
#     --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
#     --mount=type=bind,source=uv.lock,target=uv.lock \
#     # this is just to get the package manager to recognize that the project exists, without making changes to the docker layer
#     --mount=type=bind,source=invokeai/version,target=invokeai/version \
#     uv sync --frozen --inexact --active --all-extras --no-extra onnx-directml

# # build patchmatch
# RUN ln -sf --relative /usr/lib/${USERNAME}-linux-gnu/pkgconfig/{opencv4,opencv}.pc && python -c "from patchmatch import patch_match"
