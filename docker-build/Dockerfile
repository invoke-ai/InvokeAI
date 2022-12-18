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
ARG WORKDIR=/usr/src/app
WORKDIR ${WORKDIR}
ENV PATH ${WORKDIR}/.venv/bin:$PATH
ARG PIP_REQUIREMENTS=requirements-lin-cuda.txt
COPY . ./environments-and-requirements/${PIP_REQUIREMENTS} ./

# install requirements
RUN python3 -m venv .venv \
  && pip install \
    --upgrade \
    --no-cache-dir \
    'pip>=22.3.1' \
    'setuptools>=65.5.0' \
    'wheel>=0.38.4' \
  && pip install \
    --no-cache-dir \
    -r ${PIP_REQUIREMENTS}

FROM python:3.10-slim AS runtime

# Install necesarry packages
RUN apt-get update \
  && apt-get install -y \
    --no-install-recommends \
    build-essential=12.9 \
    libgl1-mesa-glx=20.3.* \
    libglib2.0-0=2.66.* \
    libopencv-dev=4.5.* \
    python3-opencv=4.5.* \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# setup environment
ARG WORKDIR=/usr/src/app
WORKDIR ${WORKDIR}
COPY --from=builder ${WORKDIR} .
ENV PATH=${WORKDIR}/.venv/bin:$PATH
ENV INVOKEAI_ROOT /data
ENV INVOKE_MODEL_RECONFIGURE --yes

# Initialize patchmatch
RUN ln -sf \
  /usr/lib/"$(arch)"-linux-gnu/pkgconfig/opencv4.pc \
  /usr/lib/"$(arch)"-linux-gnu/pkgconfig/opencv.pc \
  && python3 -c "from patchmatch import patch_match"

# set Entrypoint and default CMD
ENTRYPOINT [ "python3", "scripts/invoke.py" ]
CMD [ "--web", "--host=0.0.0.0" ]
