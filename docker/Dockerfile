# syntax=docker/dockerfile:1.4

## Builder stage

FROM library/ubuntu:22.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive
RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt update && apt-get install -y \
        git \
        python3.10-venv \
        python3-pip \
        build-essential

ENV INVOKEAI_SRC=/opt/invokeai
ENV VIRTUAL_ENV=/opt/venv/invokeai

ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ARG TORCH_VERSION=2.0.1
ARG TORCHVISION_VERSION=0.15.2
ARG GPU_DRIVER=cuda
ARG TARGETPLATFORM="linux/amd64"
# unused but available
ARG BUILDPLATFORM

WORKDIR ${INVOKEAI_SRC}

# Install pytorch before all other pip packages
# NOTE: there are no pytorch builds for arm64 + cuda, only cpu
# x86_64/CUDA is default
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m venv ${VIRTUAL_ENV} &&\
    if [ "$TARGETPLATFORM" = "linux/arm64" ] || [ "$GPU_DRIVER" = "cpu" ]; then \
        extra_index_url_arg="--extra-index-url https://download.pytorch.org/whl/cpu"; \
    elif [ "$GPU_DRIVER" = "rocm" ]; then \
        extra_index_url_arg="--extra-index-url https://download.pytorch.org/whl/rocm5.4.2"; \
    else \
        extra_index_url_arg="--extra-index-url https://download.pytorch.org/whl/cu118"; \
    fi &&\
    pip install $extra_index_url_arg \
        torch==$TORCH_VERSION \
        torchvision==$TORCHVISION_VERSION

# Install the local package.
# Editable mode helps use the same image for development:
# the local working copy can be bind-mounted into the image
# at path defined by ${INVOKEAI_SRC}
COPY invokeai ./invokeai
COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/pip \
    # xformers + triton fails to install on arm64
    if [ "$GPU_DRIVER" = "cuda" ] && [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
        pip install -e ".[xformers]"; \
    else \
        pip install -e "."; \
    fi

# #### Build the Web UI ------------------------------------

FROM node:18 AS web-builder
WORKDIR /build
COPY invokeai/frontend/web/ ./
RUN --mount=type=cache,target=/usr/lib/node_modules \
    npm install --include dev
RUN --mount=type=cache,target=/usr/lib/node_modules \
    yarn vite build


#### Runtime stage ---------------------------------------

FROM library/ubuntu:22.04 AS runtime

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
        libglib2.0-0 \
        libgl1-mesa-glx \
        python3-venv \
        python3-pip \
        build-essential \
        libopencv-dev \
        libstdc++-10-dev &&\
    apt-get clean && apt-get autoclean

# globally add magic-wormhole
# for ease of transferring data to and from the container
# when running in sandboxed cloud environments; e.g. Runpod etc.
RUN pip install magic-wormhole

ENV INVOKEAI_SRC=/opt/invokeai
ENV VIRTUAL_ENV=/opt/venv/invokeai
ENV INVOKEAI_ROOT=/invokeai
ENV PATH="$VIRTUAL_ENV/bin:$INVOKEAI_SRC:$PATH"

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
RUN python3 -c "from patchmatch import patch_match"

# Create unprivileged user and make the local dir
RUN useradd --create-home --shell /bin/bash -u 1000 --comment "container local user" invoke
RUN mkdir -p ${INVOKEAI_ROOT} && chown -R invoke:invoke ${INVOKEAI_ROOT}

COPY docker/docker-entrypoint.sh ./
ENTRYPOINT ["/opt/invokeai/docker-entrypoint.sh"]
CMD ["invokeai-web", "--host", "0.0.0.0"]
