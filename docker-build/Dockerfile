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

# set workdir and copy sources
WORKDIR /invokeai
ARG PIP_REQUIREMENTS=requirements-lin-cuda.txt
COPY . ./environments-and-requirements/${PIP_REQUIREMENTS} ./

# install requirements and link outputs folder
RUN pip install \
  --no-cache-dir \
  -r ${PIP_REQUIREMENTS}

# set Environment, Entrypoint and default CMD
ENV INVOKEAI_ROOT /data
ENTRYPOINT [ "python3", "scripts/invoke.py", "--outdir=/data/outputs" ]
CMD [ "--web", "--host=0.0.0.0" ]
