# syntax=docker/dockerfile:1
FROM python:3.9-slim AS python-base

# use bash
SHELL [ "/bin/bash", "-c" ]

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

ARG APPDIR=/usr/src/app
ENV APPDIR ${APPDIR}
WORKDIR ${APPDIR}

FROM python-base AS builder

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

# copy sources
COPY --link . .
ARG PIP_EXTRA_INDEX_URL
ENV PIP_EXTRA_INDEX_URL ${PIP_EXTRA_INDEX_URL}

# install requirements
RUN python3 -m venv invokeai \
  && ${APPDIR}/invokeai/bin/pip \
    install \
    --no-cache-dir \
    --use-pep517 \
    .

FROM python-base AS runtime

# setup environment
COPY --link . .
COPY --from=builder ${APPDIR}/invokeai ${APPDIR}/invokeai
ENV PATH=${APPDIR}/invokeai/bin:$PATH
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
ENTRYPOINT [ "invoke" ]
CMD [ "--web", "--host=0.0.0.0" ]
VOLUME [ "/data" ]
