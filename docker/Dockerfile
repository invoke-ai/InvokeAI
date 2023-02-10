# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.9
##################
##  base image  ##
##################
FROM python:${PYTHON_VERSION}-slim AS python-base

LABEL org.opencontainers.image.authors="mauwii@outlook.de"

# prepare for buildkit cache
RUN rm -f /etc/apt/apt.conf.d/docker-clean \
  && echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' >/etc/apt/apt.conf.d/keep-cache

# Install necesarry packages
RUN \
  --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  apt-get update \
  && apt-get install -y \
    --no-install-recommends \
    libgl1-mesa-glx=20.3.* \
    libglib2.0-0=2.66.* \
    libopencv-dev=4.5.*

# set working directory and env
ARG APPDIR=/usr/src
ARG APPNAME=InvokeAI
WORKDIR ${APPDIR}
ENV PATH ${APPDIR}/${APPNAME}/bin:$PATH
# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1
# don't fall back to legacy build system
ENV PIP_USE_PEP517=1

#######################
##  build pyproject  ##
#######################
FROM python-base AS pyproject-builder

# Install dependencies
RUN \
  --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  apt-get update \
  && apt-get install -y \
    --no-install-recommends \
    build-essential=12.9 \
    gcc=4:10.2.* \
    python3-dev=3.9.*

# prepare pip for buildkit cache
ARG PIP_CACHE_DIR=/var/cache/buildkit/pip
ENV PIP_CACHE_DIR ${PIP_CACHE_DIR}
RUN mkdir -p ${PIP_CACHE_DIR}

# create virtual environment
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
  python3 -m venv "${APPNAME}" \
  --upgrade-deps

# copy sources
COPY --link . .

# install pyproject.toml
ARG PIP_EXTRA_INDEX_URL
ENV PIP_EXTRA_INDEX_URL ${PIP_EXTRA_INDEX_URL}
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
  "${APPNAME}/bin/pip" install .

# build patchmatch
RUN python3 -c "from patchmatch import patch_match"

#####################
##  runtime image  ##
#####################
FROM python-base AS runtime

# Create a new User
ARG UNAME=appuser
RUN useradd \
  --no-log-init \
  -m \
  -U \
  "${UNAME}"

# create volume directory
ARG VOLUME_DIR=/data
RUN mkdir -p "${VOLUME_DIR}" \
  && chown -R "${UNAME}" "${VOLUME_DIR}"

# setup runtime environment
USER ${UNAME}
COPY --chown=${UNAME} --from=pyproject-builder ${APPDIR}/${APPNAME} ${APPNAME}
ENV INVOKEAI_ROOT ${VOLUME_DIR}
ENV TRANSFORMERS_CACHE ${VOLUME_DIR}/.cache
ENV INVOKE_MODEL_RECONFIGURE "--yes --default_only"
EXPOSE 9090
ENTRYPOINT [ "invokeai" ]
CMD [ "--web", "--host", "0.0.0.0", "--port", "9090" ]
VOLUME [ "${VOLUME_DIR}" ]
