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

# set workdir, PATH and copy sources
WORKDIR /usr/src/app
ENV PATH /usr/src/app/.venv/bin:$PATH
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
    libgl1-mesa-glx=20.3.* \
    libglib2.0-0=2.66.* \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app
COPY --from=builder /usr/src/app .

# set Environment, Entrypoint and default CMD
ENV INVOKEAI_ROOT /data
ENV INVOKE_MODEL_RECONFIGURE --yes
ENV PATH=/usr/src/app/.venv/bin:$PATH

ENTRYPOINT [ "python3", "scripts/invoke.py" ]
CMD [ "--web", "--host=0.0.0.0" ]
