FROM ubuntu AS get_miniconda

SHELL ["/bin/bash", "-c"]

# install wget
RUN apt-get update \
  && apt-get install -y \
    wget \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# download and install miniconda
ARG conda_version=py39_4.12.0-Linux-x86_64
ARG conda_prefix=/opt/conda
RUN wget --progress=dot:giga -O /miniconda.sh \
  https://repo.anaconda.com/miniconda/Miniconda3-${conda_version}.sh \
  && bash /miniconda.sh -b -p ${conda_prefix} \
  && rm -f /miniconda.sh

FROM ubuntu AS invokeai

# use bash
SHELL [ "/bin/bash", "-c" ]

# clean bashrc
RUN echo "" > ~/.bashrc

# Install necesarry packages
RUN apt-get update \
  && apt-get install -y \
    --no-install-recommends \
    gcc \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    pip \
    python3 \
    python3-dev \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# clone repository and create symlinks
ARG invokeai_git=https://github.com/invoke-ai/InvokeAI.git
ARG project_name=invokeai
RUN git clone ${invokeai_git} /${project_name} \
  && mkdir /${project_name}/models/ldm/stable-diffusion-v1 \
  && ln -s /data/models/sd-v1-4.ckpt /${project_name}/models/ldm/stable-diffusion-v1/model.ckpt \
  && ln -s /data/outputs/ /${project_name}/outputs

# set workdir
WORKDIR /${project_name}

# install conda env and preload models
ARG conda_prefix=/opt/conda
ARG conda_env_file=environment.yml
COPY --from=get_miniconda ${conda_prefix} ${conda_prefix}
RUN source ${conda_prefix}/etc/profile.d/conda.sh \
  && conda init bash \
  && source ~/.bashrc \
  && conda env create \
    --name ${project_name} \
    --file ${conda_env_file} \
  && rm -Rf ~/.cache \
  && conda clean -afy \
  && echo "conda activate ${project_name}" >> ~/.bashrc \
  && ln -s /data/models/GFPGANv1.4.pth ./src/gfpgan/experiments/pretrained_models/GFPGANv1.4.pth \
  && conda activate ${project_name} \
  && python scripts/preload_models.py

# Copy entrypoint and set env
ENV CONDA_PREFIX=${conda_prefix}
ENV PROJECT_NAME=${project_name}
COPY docker-build/entrypoint.sh /
ENTRYPOINT [ "/entrypoint.sh" ]
