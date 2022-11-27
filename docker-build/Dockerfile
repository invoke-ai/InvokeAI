FROM ubuntu:22.10

# use bash
SHELL [ "/bin/bash", "-c" ]

# Install necesarry packages
RUN apt-get update \
  && apt-get install -y \
    --no-install-recommends \
    build-essential \
    gcc \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    pip \
    python3 \
    python3-dev \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

ARG PIP_REQUIREMENTS=requirements-lin-cuda.txt
ARG PROJECT_NAME=invokeai
ARG INVOKEAI_ROOT=/data
ENV INVOKEAI_ROOT=${INVOKEAI_ROOT}

# set workdir and copy sources
WORKDIR /${PROJECT_NAME}
COPY . .

# install requirements and link outputs folder
RUN cp \
  ./environments-and-requirements/${PIP_REQUIREMENTS} \
  ${PIP_REQUIREMENTS} \
  && pip install \
    --no-cache-dir \
    -r ${PIP_REQUIREMENTS} \
  && ln -sf /data/outputs /${PROJECT_NAME}/outputs

# set Entrypoint and default CMD
ENTRYPOINT [ "python3" ]
CMD [ "scripts/invoke.py", "--web", "--host", "0.0.0.0" ]
