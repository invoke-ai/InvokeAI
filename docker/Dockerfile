# syntax=docker/dockerfile:1

# Maintained by Matthias Wild <mauwii@outlook.de>

ARG PYTHON_VERSION=3.9
##################
##  base image  ##
##################
FROM python:${PYTHON_VERSION}-slim AS python-base

# prepare for buildkit cache
RUN rm -f /etc/apt/apt.conf.d/docker-clean

# Install necesarry packages
RUN \
  --mount=type=cache,target=/var/cache/apt,sharing=locked \
  apt-get update \
  && apt-get install \
    -yqq \
    --no-install-recommends \
    libgl1-mesa-glx=20.3.* \
    libglib2.0-0=2.66.* \
    libopencv-dev=4.5.* \
  && rm -rf /var/lib/apt/lists/*

# set working directory and path
ARG APPDIR=/usr/src
ARG APPNAME=InvokeAI
WORKDIR ${APPDIR}
ENV PATH=${APPDIR}/${APPNAME}/bin:$PATH

#######################
##  build pyproject  ##
#######################
FROM python-base AS pyproject-builder
ENV PIP_USE_PEP517=1

# prepare for buildkit cache
ARG PIP_CACHE_DIR=/var/cache/buildkit/pip
ENV PIP_CACHE_DIR ${PIP_CACHE_DIR}
RUN mkdir -p ${PIP_CACHE_DIR}

# Install dependencies
RUN \
  --mount=type=cache,target=${PIP_CACHE_DIR} \
  --mount=type=cache,target=/var/cache/apt,sharing=locked \
  apt-get update \
  && apt-get install \
    -yqq \
    --no-install-recommends \
    build-essential=12.9 \
    gcc=4:10.2.* \
    python3-dev=3.9.* \
  && rm -rf /var/lib/apt/lists/*

# create virtual environment
RUN --mount=type=cache,target=${PIP_CACHE_DIR} \
  python3 -m venv "${APPNAME}" \
  --upgrade-deps

# copy sources
COPY --link . .

# install pyproject.toml
ARG PIP_EXTRA_INDEX_URL
ENV PIP_EXTRA_INDEX_URL ${PIP_EXTRA_INDEX_URL}
ARG PIP_PACKAGE=.
RUN --mount=type=cache,target=${PIP_CACHE_DIR} \
  "${APPDIR}/${APPNAME}/bin/pip" install ${PIP_PACKAGE}

# build patchmatch
RUN python3 -c "from patchmatch import patch_match"

#####################
##  runtime image  ##
#####################
FROM python-base AS runtime

# setup environment
COPY --from=pyproject-builder --link ${APPDIR}/${APPNAME} ${APPDIR}/${APPNAME}
ENV INVOKEAI_ROOT=/data
ENV INVOKE_MODEL_RECONFIGURE="--yes --default_only"

# set Entrypoint and default CMD
ENTRYPOINT [ "invokeai" ]
CMD [ "--web", "--host=0.0.0.0" ]
VOLUME [ "/data" ]
