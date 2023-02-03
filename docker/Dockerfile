# syntax=docker/dockerfile:1

# Maintained by Matthias Wild <mauwii@outlook.de>

ARG PYTHON_VERSION=3.9
##################
### base image ###
##################
FROM python:${PYTHON_VERSION}-slim AS python-base

# Install necesarry packages
RUN \
  --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  apt-get update \
  && apt-get install -y \
    --no-install-recommends \
    libgl1-mesa-glx=20.3.* \
    libglib2.0-0=2.66.* \
    libopencv-dev=4.5.* \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# set working directory and path
ARG APPDIR=/usr/src
ARG APPNAME=InvokeAI
WORKDIR ${APPDIR}
ENV PATH=${APPDIR}/${APPNAME}/bin:$PATH

######################
### build frontend ###
######################
FROM node:lts as frontend-builder

# Copy Sources
ARG APPDIR=/usr/src
WORKDIR ${APPDIR}
COPY --link . .

# install dependencies and build frontend
WORKDIR ${APPDIR}/invokeai/frontend
RUN \
  --mount=type=cache,target=/usr/local/share/.cache/yarn/v6 \
  yarn install \
  --prefer-offline \
  --frozen-lockfile \
  --non-interactive \
  --production=false \
  && yarn build

###################################
### install python dependencies ###
###################################
FROM python-base AS pyproject-builder

# Install dependencies
RUN \
  --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  apt-get update \
  && apt-get install -y \
    --no-install-recommends \
    gcc=4:10.2.* \
    python3-dev=3.9.* \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# create virtual environment
RUN python3 -m venv "${APPNAME}" \
  --upgrade-deps

# copy sources
COPY --from=frontend-builder ${APPDIR} .

# install pyproject.toml
ARG PIP_EXTRA_INDEX_URL
ENV PIP_EXTRA_INDEX_URL ${PIP_EXTRA_INDEX_URL}
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
  "${APPDIR}/${APPNAME}/bin/pip" install \
  --use-pep517 \
  .

#####################
### runtime image ###
#####################
FROM python-base AS runtime

# setup environment
COPY --from=pyproject-builder ${APPDIR}/${APPNAME} ${APPDIR}/${APPNAME}
ENV INVOKEAI_ROOT=/data
ENV INVOKE_MODEL_RECONFIGURE="--yes --default_only"

# build patchmatch
RUN \
  --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  apt-get update \
  && apt-get install -y \
    --no-install-recommends \
    build-essential=12.9 \
  && PYTHONDONTWRITEBYTECODE=1 \
    python3 -c "from patchmatch import patch_match" \
  && apt-get remove -y \
    --autoremove \
    build-essential \
  && apt-get autoclean \
  && rm -rf /var/lib/apt/lists/*

# set Entrypoint and default CMD
ENTRYPOINT [ "invokeai" ]
CMD [ "--web", "--host=0.0.0.0" ]
VOLUME [ "/data" ]
