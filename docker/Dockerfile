# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.9
##################
##  base image  ##
##################
FROM --platform=${TARGETPLATFORM} python:${PYTHON_VERSION}-slim AS python-base
ENV DEBIAN_FRONTEND=noninteractive
LABEL org.opencontainers.image.authors="mauwii@outlook.de"

RUN apt update && \
    apt install curl gnupg2 software-properties-common -y && \
    curl -sSL https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
    apt-add-repository https://packages.microsoft.com/debian/11/prod && \
    apt update && \
    apt install fuse3 blobfuse2 -y

# Prepare apt for buildkit cache
RUN rm -f /etc/apt/apt.conf.d/docker-clean \
    && echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' >/etc/apt/apt.conf.d/keep-cache

# Install dependencies
RUN \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
    && apt-get install -y \
    --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libopencv-dev

# Set working directory and env
ARG APPDIR=/usr/src
ARG APPNAME=InvokeAI
WORKDIR ${APPDIR}
ENV PATH ${APPDIR}/${APPNAME}/bin:$PATH
# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1
# Don't fall back to legacy build system
ENV PIP_USE_PEP517=1

# #######################
# ##  build pyproject  ##
# #######################
FROM python-base AS pyproject-builder

# Install build dependencies
RUN \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
    && apt-get install -y \
    --no-install-recommends \
    build-essential \
    gcc \
    python3-dev \
    python3-venv

# Prepare pip for buildkit cache
ARG PIP_CACHE_DIR=/var/cache/buildkit/pip
ENV PIP_CACHE_DIR ${PIP_CACHE_DIR}
RUN mkdir -p ${PIP_CACHE_DIR}

# Create virtual environment
RUN --mount=type=cache,target=${PIP_CACHE_DIR} \
    python3 -m venv "${APPNAME}" \
    --upgrade-deps

# Install requirements
COPY --link pyproject.toml .
COPY --link invokeai/version/invokeai_version.py invokeai/version/__init__.py invokeai/version/
ARG PIP_EXTRA_INDEX_URL
ENV PIP_EXTRA_INDEX_URL ${PIP_EXTRA_INDEX_URL}
RUN --mount=type=cache,target=${PIP_CACHE_DIR} \
    "${APPNAME}"/bin/pip install .

# Install pyproject.toml
COPY --link . .
RUN --mount=type=cache,target=${PIP_CACHE_DIR} \
    "${APPNAME}/bin/pip" install .

# Build patchmatch
RUN python3 -c "from patchmatch import patch_match"

#####################
##  runtime image  ##
#####################
FROM python-base AS runtime

# Create volume directory
ARG VOLUME_DIR=/data
RUN mkdir -p "${VOLUME_DIR}" \
    && mkdir -p /blob_data

# Setup runtime environment
COPY --from=pyproject-builder ${APPDIR}/${APPNAME} ${APPNAME}
COPY ./invokeai/configs/resleeve_configs.yml /data
COPY ./invokeai/configs/blobfuse2/config.yaml /blob_configs/config_blobfuse.yaml
COPY ./invokeai/configs/certificates/BaltimoreCyberTrustRoot.crt.pem /certificates/db_certificate.crt.pem
COPY ./docker/app.sh /usr/src/app.sh
ENV INVOKEAI_ROOT ${VOLUME_DIR}
ENV TRANSFORMERS_CACHE ${VOLUME_DIR}/.cache
ENV INVOKE_MODEL_RECONFIGURE "--yes --default_only"
EXPOSE 9090
CMD ["bash", "/usr/src/app.sh"]
