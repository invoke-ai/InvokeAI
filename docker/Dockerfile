# syntax=docker/dockerfile:1.4

## Builder stage

FROM library/ubuntu:24.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive
RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt update && apt-get install -y \
        build-essential \
        git

# Install `uv` for package management
COPY --from=ghcr.io/astral-sh/uv:0.5.5 /uv /uvx /bin/

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV INVOKEAI_SRC=/opt/invokeai
ENV PYTHON_VERSION=3.11
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

ARG GPU_DRIVER=cuda
ARG TARGETPLATFORM="linux/amd64"
# unused but available
ARG BUILDPLATFORM

# Switch to the `ubuntu` user to work around dependency issues with uv-installed python
RUN mkdir -p ${VIRTUAL_ENV} && \
    mkdir -p ${INVOKEAI_SRC} && \
    chmod -R a+w /opt
USER ubuntu

# Install python and create the venv
RUN uv python install ${PYTHON_VERSION} && \
    uv venv --relocatable --prompt "invoke" --python ${PYTHON_VERSION} ${VIRTUAL_ENV}

WORKDIR ${INVOKEAI_SRC}
COPY invokeai ./invokeai
COPY pyproject.toml ./

# Editable mode helps use the same image for development:
# the local working copy can be bind-mounted into the image
# at path defined by ${INVOKEAI_SRC}
# NOTE: there are no pytorch builds for arm64 + cuda, only cpu
# x86_64/CUDA is the default
RUN --mount=type=cache,target=/home/ubuntu/.cache/uv,uid=1000,gid=1000 \
    if [ "$TARGETPLATFORM" = "linux/arm64" ] || [ "$GPU_DRIVER" = "cpu" ]; then \
        extra_index_url_arg="--extra-index-url https://download.pytorch.org/whl/cpu"; \
    elif [ "$GPU_DRIVER" = "rocm" ]; then \
        extra_index_url_arg="--extra-index-url https://download.pytorch.org/whl/rocm6.1"; \
    else \
        extra_index_url_arg="--extra-index-url https://download.pytorch.org/whl/cu124"; \
    fi && \
    uv pip install --python ${PYTHON_VERSION} $extra_index_url_arg -e "."

#### Build the Web UI ------------------------------------

FROM docker.io/node:22-slim AS web-builder
ENV PNPM_HOME="/pnpm"
ENV PATH="$PNPM_HOME:$PATH"
RUN corepack use pnpm@8.x
RUN corepack enable

WORKDIR /build
COPY invokeai/frontend/web/ ./
RUN --mount=type=cache,target=/pnpm/store \
    pnpm install --frozen-lockfile
RUN npx vite build

#### Runtime stage ---------------------------------------

FROM library/ubuntu:24.04 AS runtime

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt update && apt install -y --no-install-recommends \
        git \
        curl \
        vim \
        tmux \
        ncdu \
        iotop \
        bzip2 \
        gosu \
        magic-wormhole \
        libglib2.0-0 \
        libgl1 \
        libglx-mesa0 \
        build-essential \
        libopencv-dev \
        libstdc++-10-dev &&\
    apt-get clean && apt-get autoclean

ENV INVOKEAI_SRC=/opt/invokeai
ENV VIRTUAL_ENV=/opt/venv
ENV PYTHON_VERSION=3.11
ENV INVOKEAI_ROOT=/invokeai
ENV INVOKEAI_HOST=0.0.0.0
ENV INVOKEAI_PORT=9090
ENV PATH="$VIRTUAL_ENV/bin:$INVOKEAI_SRC:$PATH"
ENV CONTAINER_UID=${CONTAINER_UID:-1000}
ENV CONTAINER_GID=${CONTAINER_GID:-1000}

# Install `uv` for package management
# and install python for the ubuntu user (expected to exist on ubuntu >=24.x)
# this is too tiny to optimize with multi-stage builds, but maybe we'll come back to it
COPY --from=ghcr.io/astral-sh/uv:0.5.5 /uv /uvx /bin/
USER ubuntu
RUN uv python install ${PYTHON_VERSION}
USER root

# --link requires buldkit w/ dockerfile syntax 1.4
COPY --link --from=builder ${INVOKEAI_SRC} ${INVOKEAI_SRC}
COPY --link --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}
COPY --link --from=web-builder /build/dist ${INVOKEAI_SRC}/invokeai/frontend/web/dist

# Link amdgpu.ids for ROCm builds
# contributed by https://github.com/Rubonnek
RUN mkdir -p "/opt/amdgpu/share/libdrm" &&\
  ln -s "/usr/share/libdrm/amdgpu.ids" "/opt/amdgpu/share/libdrm/amdgpu.ids"

WORKDIR ${INVOKEAI_SRC}

# build patchmatch
RUN cd /usr/lib/$(uname -p)-linux-gnu/pkgconfig/ && ln -sf opencv4.pc opencv.pc
RUN python -c "from patchmatch import patch_match"

RUN mkdir -p ${INVOKEAI_ROOT} && chown -R ${CONTAINER_UID}:${CONTAINER_GID} ${INVOKEAI_ROOT}

COPY docker/docker-entrypoint.sh ./
ENTRYPOINT ["/opt/invokeai/docker-entrypoint.sh"]
CMD ["invokeai-web"]
