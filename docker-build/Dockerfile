FROM python:3.10-slim AS builder

# use bash
SHELL [ "/bin/bash", "-c" ]

# Install necesarry packages
RUN apt-get update \
  && apt-get install -y \
    --no-install-recommends \
    gcc=4:10.2.* \
    libgl1-mesa-glx=20.3.* \
    libglib2.0-0=2.66.* \
    python3-dev=3.9.* \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# set WORKDIR, PATH and copy sources
ARG APPDIR=/usr/src/app
ARG PIP_EXTRA_INDEX_URL
ENV \
  PATH=${APPDIR}/.venv/bin:$PATH \
  PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL}

WORKDIR ${APPDIR}
COPY . .

# install requirements
RUN python3 -m venv .venv \
  && pip install \
    --no-cache-dir \
    --use-pep517 \
    .

FROM python:3.10-slim AS runtime

# setup environment
ARG APPDIR=/usr/src/app
WORKDIR ${APPDIR}
COPY --from=builder ${APPDIR} .
ENV \
  PATH=${APPDIR}/.venv/bin:$PATH \
  INVOKEAI_ROOT=/data \
  INVOKE_MODEL_RECONFIGURE=--yes

# Install necesarry packages
RUN apt-get update \
  && apt-get install -y \
    --no-install-recommends \
    build-essential=12.9 \
    libgl1-mesa-glx=20.3.* \
    libglib2.0-0=2.66.* \
    libopencv-dev=4.5.* \
  && python3 -c "from patchmatch import patch_match" \
  && apt-get remove -y \
    --autoremove \
    build-essential \
  && apt-get autoclean \
  && rm -rf /var/lib/apt/lists/*

# set Entrypoint and default CMD
ENTRYPOINT [ "invoke" ]
CMD [ "--web", "--host=0.0.0.0" ]
